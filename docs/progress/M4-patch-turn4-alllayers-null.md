# M4 phase 2b — All-layer L1-L48 single-position patch is also null; pre-answer position is not the reveal-token bottleneck

**Run date:** 2026-04-25 by Claude (Opus 4.7).
**Source collection:** `runs/diag/selfchosen_ready_20bank_12b_turn4scale_20260421/`
(M3 default-prompt n=80 scale-up; balanced 4 classes — `cow`, `dog`,
`elephant`, `horse` — 20 runs/class).
**Patch script:** `scripts/patch_turn4.py` at commit `9384758`.
**Job:** `tq_m4_patch_turn4_12b_alllayers_20260425.sh` (gpu_h MIG slice,
job `7260501`, ~2 min wall after model load).
**Machine-readable report:** `runs/m4_patch_turn4_12b_default_L1-48all.json`.

## Scope

Decisive position-vs-layer test. Same target collection and trial design
as phases 1/2a (5 src × 5 tgt per class, 16 cells, 400 patched trials +
20 baselines). The change: patch **all 48 decoder-block layers
simultaneously** at the same single turn-4 pre-answer position. This
replaces the entire residual stream at that position with src's
representation — no layer escapes the intervention.

Per the phase 2a writeup (D-36), the structural hypotheses entering this
phase are:

1. **Earlier-layer locus.** Class info is in L1-L26 at the pre-answer
   position; late layers carry it forward inertly. Phase 2b replaces
   *both* early and late layers, so a positive result here lights up
   the early-layer hypothesis.
2. **Other-position locus.** The pre-answer position is not the
   bottleneck regardless of layer. Phase 2b nulls under this
   hypothesis.

## Result — null on argmax, near-null on logit-diff

### Argmax flip-rate

```
  src\tgt |      cow |      dog | elephant |    horse
  ---------------------------------------------------
      cow |    92.0% |     0.0% |     0.0% |     0.0%
      dog |     0.0% |   100.0% |     0.0% |     0.0%
 elephant |     0.0% |     0.0% |   100.0% |     0.0%
    horse |     0.0% |     0.0% |     0.0% |   100.0%
```

Off-diagonals are 0% across every cell. The phase 2a `cow→horse` 20%
"effect" — which D-35 traced to one non-deterministic horse target
(`attempt_588`) — has now disappeared in phase 2b. Patching the entire
residual stack apparently overwrites whatever stochastic state was
producing the `attempt_588` cow drift in phase 1/2a.

The cow→cow self-patch drops from 100% (phase 1/2a) to 92% — 2/25 cow
targets reveal horse instead of cow when their residual stack at the
pre-answer position is replaced with another cow run's residual stack.
The patch is no longer a no-op even within-class because src and tgt
are different runs, and overwriting all 48 layers at one position is a
strong-enough perturbation to occasionally tip the noise. Importantly,
**the perturbation tips toward the model's prior (horse) — not toward
src's class**.

### Logit-difference deltas (patched − baseline)

```
  src\tgt |      cow |      dog | elephant |    horse
  ---------------------------------------------------
      cow |    +0.00 |    -0.46 |    -0.15 |    -0.37
      dog |    -0.10 |    +0.00 |    -0.23 |    -0.41
 elephant |    +0.03 |    -0.43 |    +0.00 |    -0.44
    horse |    +0.23 |    +0.03 |    -0.06 |    +0.00
```

Diagonals exactly 0 (correct). Phase 2b deltas are 3-4× larger in
magnitude than phase 2a (range ±0.46 vs ±0.12), but **predominantly
negative** on off-diagonals — patches push *away* from the src class,
not toward it. This pattern is consistent with "stronger residual
disruption increases noise but does not introduce the src signal."
The horse row is the only mostly-positive row (+0.23, +0.03, -0.06),
and even there the largest delta is +0.23 logits against natural class
margins of +2 to +10 — i.e. ≤10% of the baseline gap.

### Off-diagonal flip distributions show prior leakage

Several off-diagonal cells reveal a horse-leakage signature:

- `dog→cow`: dist `{cow: 23, horse: 2}` (2/25 went to horse, not dog)
- `dog→horse`: dist `{horse: 23, cow: 2}` (2/25 went to cow, not dog)
- `elephant→cow`: dist `{cow: 24, horse: 1}` (1/25 went to horse)

The 2-class fallback set is `{cow, horse}` — the top-2 attractor classes
in the M3 less_obvious-variant (38.5% horse + 33.0% cow = 71.5% prior
mass; D-34). When the residual stream at the pre-answer position is
heavily disrupted, the model occasionally falls back to those priors.
**It does not adopt src's class** — even when src is dog or elephant,
the model rarely produces dog or elephant from a non-matching target.

### Methodological caveat (carried over from D-36)

The first-step logit-diff metric is unreliable for dog (reveals don't
begin with the ` Dog` token at step 0; baseline first-token margin is
*negative* for dog targets while parsed reveal is dog 100%). Treat dog
deltas as suspect; cow/elephant/horse rows remain reliable.

## Interpretation

Together, phase 1 (L29 single-layer), phase 2a (L27-L48 band), and
phase 2b (L1-L48 all-layer) — all at the turn-4 pre-answer position —
rule out **every layer-axis variant** of "the pre-answer position is
the reveal bottleneck." The intervention scales:

| phase | layers patched | flip rate (off-diag) | logit-diff delta range |
|---|---|---:|---|
| 1 | L29 only | 0% (sans `attempt_588`) | not measured |
| 2a | L27-L48 (22 layers) | 0% (sans `attempt_588`) | ±0.12 |
| 2b | **L1-L48 (all 48)** | **0%** | ±0.46 (mostly negative) |

Tripling the layer scope from phase 1 → phase 2b produces no shift in
flip rate and only a small increase in logit-diff *noise*, never a
consistent push toward src.

**Conclusion:** the turn-4 pre-answer position is **not** in the
causal path of the reveal token, regardless of which layers the
intervention targets. The class identity is decodable at this position
(M3 turn-4 LR LOO 0.79 @ L31, ~3.2× chance) but the residual at this
position is *not* what drives the model's reveal. The reveal must be
computed via attention to *other token positions* in the dialogue
prefix.

This is hypothesis (2) from D-36. It also matches the dissociation
documented in "Causality ≠ Decodability" (arXiv 2510.09794) and the
broader finding that probe-decodability and causal effect can diverge
sharply (Heimersheim & Nanda 2024, arXiv 2404.15255).

## What positions could carry the reveal-causal signal?

Several plausible candidates, in order of "earliest commit":

1. **Turn-0 Ready output position.** In the self-chosen condition, the
   model is asked to silently choose an animal and reply `Ready`. The
   class commitment plausibly lives in the residual stream at the
   `Ready` token (or just before it). If so, patching that position
   should flip reveals.
2. **Per-turn pre-answer positions (turn 1, 2, 3).** Already saved on
   TSUBAME (`turn_01_..._activations.pt`), so each is a free
   single-position-all-layer experiment.
3. **Turn-4 user message tokens** (the question text positions, not
   the post-question scaffolding). These are where turn-4-specific info
   enters the residual stream.
4. **Distributed across the entire dialogue prefix.** If no single
   position is the bottleneck, the class info is encoded redundantly
   across positions and a position-band patch is the only way to move
   reveals.

The cheapest first experiment uses already-saved per-turn activations:
patch each of `turn_01`, `turn_02`, `turn_03` pre-answer positions in
turn under all-layer mode. ~6 min total walltime. If any flips
reveals, the bottleneck is at that turn. If all null, escalate to a
multi-position window patch (requires live src activation capture
during the experiment).

## Decision consequence — phase 2c

**Phase 2c-i (cheap, no new captures): per-turn pre-answer all-layer
patch sweep.** Run `patch_turn4.py` with the activation file pointed
at `turn_01_activations.pt`, `turn_02_..._activations.pt`,
`turn_03_..._activations.pt` in sequence, `--layers 1,...,48`, same
n=80 collection. The script needs a small extension to accept a
`--turn` flag that points at the right activation file and computes
the right tgt position. Three jobs (or one bundled job), ~2 min each.

**Phase 2c-ii (escalation if 2c-i is null): position-band patch.**
Add a `--position-window K` flag and on-the-fly src forward-pass
capture across the last K positions of the chosen turn's prefill.
Patch a window of K positions simultaneously. K=5 captures the
chat-template scaffolding (`<end_of_turn>\n<start_of_turn>model\n`),
which is token-aligned across runs. Larger K extends into the
question text and requires careful handling of cross-run alignment.

## Side observation — `attempt_588` instability is solvable by patching

In phase 1/2a, `attempt_588` produced the only off-diagonal "effect"
because its baseline was already drifting between horse and cow.
Phase 2b's stronger intervention completely overwrites that drift —
the cow→horse 20% from phase 1/2a is gone in phase 2b. This is not a
fix for the underlying non-determinism (which still affects baseline
generation), but it does mean the all-layer patch produces clean
matrices on every run. So phase 2c experiments can use the existing
collection without first solving the `attempt_588` reproducibility
issue.

## Artifacts

- `runs/m4_patch_turn4_12b_default_L1-48all.json` — full per-trial
  records + per-cell summary with both metrics (400 patched + 20
  baselines).
- `scripts/patch_turn4.py` at `0a4913c` — extended for layer band and
  logit capture.
- Reference: Heimersheim & Nanda 2024, *How to use and interpret
  activation patching* (arXiv 2404.15255); "Causality ≠ Decodability"
  arXiv 2510.09794.
