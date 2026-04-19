# M3 — 4B smoke + calibration-binding diagnostics

**Opened:** 2026-04-19 by Codex (GPT-5) + Claude (Opus 4.7) review.
**Model:** `google/gemma-3-4b-it` (revision `093f9f388b`), float32, TSUBAME H100.
**Scope:** 2 candidates (tiger, eagle), 3 questions, 4 conditions. Not the
full M3 calibration — this is the diagnostic that explains why M3 did **not**
scale to ~2k runs on schedule.

## TL;DR

- **Mechanics pass on 4B.** Model loads, `Ready` parses, question-turn
  activations capture, `turn_activation_paths` populate in the manifest,
  `PROMPT_TEMPLATE_ID=v2-2026-04-19` flows through. No plumbing bug.
- **Under every index-based calibration variant tried, `eagle`'s
  `Is it a mammal?` answer is `Yes` — wrong.** The model does not behave
  as if a specific animal is held in mind when the secret is bound by list
  position, even with an explicit per-turn "your secret is the animal at
  position #N" reminder.
- **Under name-based binding ("Your secret animal is eagle"), answers are
  correct 6/6.**
- A generic "remember the same secret" reminder at each turn *made tiger's
  answers worse* — it shifted toward the candidate-list average rather than
  reinforcing the specific secret. So it's not just that the reminder
  failed to help; it confirms the per-turn prompt can over-influence the
  answer relative to whatever is held at Ready.

**Scientific read (tentative):** at 4B, a bare index pointer ("#11 in the
list") does not reliably **instantiate** a specific-entity representation at
the Ready position — it seems to leave the model biased toward
candidate-list priors (mostly-mammal in our 20-animal bank). Name-based
binding does instantiate it. This matters for the headline claim:
self-chosen decodability requires that there *be* a secret to decode. We
should treat calibration as "probe-training infra *conditional on* the
binding actually producing an instantiated secret," and test that condition
explicitly at 4B before scaling.

## Runs

All run artifacts on TSUBAME at
`/gs/fs/tga-sip_arase/tyrone/twenty_questions_interp/`:

| Job | Script | Purpose | Log |
|---|---|---|---|
| `tq_m3_smoke` (7216361) | `jobs/tq_m3_smoke.sh` | Current D-06 index prompt, per-run permutation | `logs/tq_m3_smoke.7216361.out` |
| `tq_turn_diag` (7216366) | `jobs/tq_m3_turn_prompt_diag.sh` | Index + generic per-turn reminder | `logs/tq_turn_diag.7216366.out` |
| `tq_index_diag` (7216372) | `jobs/tq_m3_index_reminder_diag.sh` | Index + explicit "position #N" per-turn reminder | `logs/tq_index_diag.7216372.out` |
| `tq_name_diag` (7216369) | `jobs/tq_m3_name_cal_diag.sh` | **Name-based** ("Your secret animal is X"), otherwise identical | `logs/tq_name_diag.7216369.out` |

Bank ground truth for the three probe questions (`data/answers.csv`):

| Animal | `is_mammal` | `can_fly` | `can_swim` |
|---|---|---|---|
| tiger | **Yes** | No | **Yes** |
| eagle | No | **Yes** | No |

Note: `can_swim` is phrased as *regular* swimming. The bank marks tiger=Yes
(they are strong swimmers behaviourally), which is the answer most humans
would hesitate on. Don't read too much into tiger/can_swim below.

## Per-condition transcripts

### A. Current D-06 index prompt (smoke)

Seeds produced `tiger @ position 13`, `eagle @ position 1`.

```
cal_tiger_00  (secret_displayed_index=13)
  is_mammal: Yes ✓
  can_fly:   No  ✓
  can_swim:  No  ✗  (bank says Yes; tiger/can_swim is the one bank answer most LLMs disagree with)

cal_eagle_00  (secret_displayed_index=1)
  is_mammal: Yes ✗  ← eagle is a bird
  can_fly:   Yes ✓
  can_swim:  Yes ✗  ← eagles don't regularly swim
```

### B. Index + generic reminder ("Remember the same secret animal you already fixed…")

Same displayed permutation as A.

```
SECRET tiger
  is_mammal: Yes ✓
  can_fly:   Yes ✗  ← tiger now says it can fly
  can_swim:  Yes ✓

SECRET eagle
  is_mammal: Yes ✗  ← unchanged failure
  can_fly:   Yes ✓
  can_swim:  No  ✓
```

Tiger's `can_fly` flipped from No (correct) in condition A to Yes (wrong)
in condition B. The reminder *hurts*.

### C. Index + explicit position reminder ("…the animal at position #N from the original candidate list")

Identity permutation (`tiger @ idx 1`, `eagle @ idx 11`).

```
SECRET tiger (IDX 1)
  is_mammal: Yes ✓
  can_fly:   No  ✓
  can_swim:  No  ✗  (same bank disagreement as A)

SECRET eagle (IDX 11)
  is_mammal: Yes ✗  ← failure persists with explicit position
  can_fly:   Yes ✓
  can_swim:  No  ✓
```

Eagle's `is_mammal` failure survives even when every turn restates
"your secret is the animal at position #11". This is the strongest
evidence that the failure is not "the model forgot which index" —
it is that indexing didn't bind the *concept* in the first place.

### D. Name-based control ("Your secret animal is X")

Same 3 questions, but binding via the animal's name instead of its list
index.

```
SECRET tiger
  is_mammal: Yes ✓
  can_fly:   No  ✓
  can_swim:  Yes ✓

SECRET eagle
  is_mammal: No  ✓  ← fixed
  can_fly:   Yes ✓
  can_swim:  No  ✓
```

**6/6 correct.** The only thing that changed between C and D is whether the
secret is referenced by list index or by name.

## What this means for the research, not just the infra

The calibration condition is supposed to be a **probe-training harness**:
give the model a secret, capture Ready, train a decoder that we then apply
to self-chosen runs. That only works if the Ready-state representation
actually encodes the instantiated secret.

The smoke suggests a research-level claim worth testing, not just a
prompt-engineering workaround:

> **H-binding.** At 4B, referencing the secret by list index at
> commitment time does not reliably produce an instantiated-entity
> representation at Ready. Referencing it by name does. Self-chosen,
> where the model names its own secret internally, is closer to the
> name-based regime than to the index-based one.

This is directly relevant to the blog claim because it *supports* the
expected transfer story (name-based calibration probes → self-chosen
Ready-state) rather than breaking it. The cost is that D-06's motivation
("avoid training a decoder on the literal surface token of the name") is
now in tension with getting a binding at all. That tension is resolvable
(see below) but we should decide it in DECISIONS.md, not implicitly.

### Concrete decodable predictions from H-binding

These are cheap to measure once the next smoke runs, and each one informs
what we think the Ready state *is*:

1. **NC LOO at 4B should be lower under index calibration than under name
   calibration**, at every post-13 layer. If index NC LOO is near chance
   while name NC LOO is well above, that's a quantitative version of "index
   doesn't instantiate."
2. **Under index calibration, two runs with the same secret at different
   displayed positions should be *less* similar at Ready than under name
   calibration**, controlling for everything else. The natural metric is
   within-class centroid concentration (or NC in-class cosine).
3. **Under name calibration, within-class cosine should be high, but so
   should surface-token confound**: the literal animal-name token at
   Ready shouldn't match the secret (the prompt tells the model not to
   state the name), but it could still be tokenwise near. Worth a direct
   check on the last generated token before "Ready" gets emitted.
4. **If we add "answer question turns" to calibration, name-based answer
   correctness >> index-based answer correctness (scaled to more
   candidates × questions).** The smoke already suggests this for 2×3;
   worth confirming at 20×5.

None of these require full M3 scale to do — they fit in small jobs on H100
with a single-digit count of runs per condition.

## Why not just fix the prompt and move on

Two reasons to treat this as a research artifact, not an infra hiccup:

1. **The phenomenon is reproducible and category-specific.** Eagle's
   `is_mammal` failure held across three index variants and one generic
   reminder, and name-based flipped it back instantly. That is the
   signature of a representational effect, not a decoding-temperature
   artifact.
2. **The "reminder hurts tiger" finding is independent of the eagle
   failure.** Condition B shows that a per-turn reminder can push the
   answer toward a candidate-pool prior even when the underlying secret
   representation is sane (tiger did answer correctly without the
   reminder). That constrains what we think the Ready-state → answer
   pathway is doing.

## Recommended next smoke

Before touching the full 2k calibration:

- **4B, 4 candidates × 2 seeds × 4 conditions × 5 questions.** Conditions:
  (a) current index, (b) index + position reminder, (c) name, (d) name
  with paraphrased binding ("You have chosen X as your secret"). Scoring
  gate: conditions (c) and (d) must hit ≥ 95% answer correctness before
  any 2k-scale run. Persist per-turn activations on all conditions so we
  can cheaply fit NC/LR at Ready between variants.
- **Candidates chosen to span categories**, not tiger + eagle alone:
  e.g., `tiger, eagle, frog, salmon` (mammal / bird / amphibian / fish).
  The eagle failure may be specifically about "bird in a mammal-heavy
  list"; we need to know if it's category-general.
- Decision D-06 should be **explicitly re-examined** in DECISIONS.md on
  the basis of this smoke, not edited. If we move to name-based
  calibration, add a new D-NN that reverses D-06 and explains the
  trade-off (token-confound vs. binding-failure).

## Handoff-ready artifacts

- TSUBAME: `runs/calibration/cal_{tiger,eagle}_00/` each has
  `manifest.json`, `activations.pt`, and three `turn_NN_activations.pt`
  files. These are the only 4B runs on disk so far.
- Job scripts: `jobs/tq_m3_*.sh` — keep as the template for the 4-condition
  smoke.
- D-17 in DECISIONS.md already blocks scaling. This note is the
  supporting evidence for D-17 and the scaffold for the next decision
  (revisiting D-06).
