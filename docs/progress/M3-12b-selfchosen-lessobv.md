# M3 — 12B self-chosen `less_obvious` prompt variant: attractor partially moves, turn-4 signal generalizes

**Run date:** 2026-04-24 by Claude (Opus 4.7).
**Source collection:** `runs/diag/selfchosen_ready_20bank_12b_lessobv_20260422/`
from job `7246440` (600 attempts, T=0.0 greedy, 101 kept runs).
**Prompt template id:** `v2-2026-04-19-less_obvious` (suffixed variant of
the default self-chosen prompt; adds the sentence *"Pick a less obvious
animal from the list — avoid the most stereotypical first choices."*)
**Machine-readable report:** `runs/m3_12b_selfchosen_lessobv_turns.json`.
**Detail table:** `docs/progress/M3-12b-selfchosen-lessobv-detail.md`.

## Scope

Follow-up to `M3-12b-selfchosen-diversity-T07.md` / D-33. T=0.7 sampling
did not broaden the 12B self-chosen attractor beyond
`{elephant, cow, dog, horse}`. The question this run answers is whether
the attractor is *prompt*-fragile at greedy temperature, and — if so —
whether the turn-4 LR LOO signal generalizes to new classes or is
specific to the original 4-cluster.

Everything else held identical: `google/gemma-3-12b-it`, bfloat16,
T=0.0, 20-bank, 4-question panel, `--n-per-candidate 20`,
`--stop-when-n-classes-hit-quota 8`, 600 attempts max.

## Result 1 — diversity

Reveal parse 100%, Ready parse 100%. Full attempt distribution over 600
greedy attempts:

| class | count | share | note |
|---|---:|---:|---|
| horse | 231 | 38.5% | persists |
| cow | 198 | 33.0% | persists |
| gorilla | 102 | 17.0% | **new** |
| elephant | 47 | 7.8% | persists, shrunk |
| kangaroo | 21 | 3.5% | **new** |
| penguin | 1 | 0.2% | **new** |
| dog | 0 | 0 | **displaced** |
| 13 others | 0 | 0 | still never realized |

- Distinct parsed classes: **6** (vs 4 at default T=0.0 and T=0.7).
- Top-1 share: 38.5% (essentially unchanged).
- Effective classes (entropy-based): **3.90** (vs 3.46 at T=0.7).

Quota-hit classes (`n >= 20`): `{elephant, cow, horse, gorilla,
kangaroo}`. That is a 5-class realized set; the `--stop-when-n-classes-
hit-quota 8` early stop did **not** fire — the run completed all 600
attempts and the 8th class never hit quota.

## Result 2 — turn-4 decode on the new class set

Running `decode_turns.py --selection kept --turns 1,4 --layers all` on
the 101 kept runs (includes `penguin=1` as a noisy singleton, so
effective LR chance is ~0.198, basically 0.20):

| turn | LR peak | LR mean L27–48 | NC peak | NC mean L27–48 |
|---|---:|---:|---:|---:|
| 1 | 0.51 @ L30 | 0.43 | 0.35 @ L32/L34 | 0.32 |
| **4** | **0.54 @ L29** | **0.51** | **0.48 @ L48** | **0.42** |

LR turn-4 is ≥0.48 across L27–L48 (broad band, not a peak). Peak at
L29 mirrors the default scale-up's peak location (L29–L31).

## Comparison to the default-prompt scale-up

| axis | default n=80 (4 classes) | less_obvious n=101 (5 classes) |
|---|---:|---:|
| chance | 0.25 | 0.20 |
| turn-4 LR peak | 0.79 @ L31 | 0.54 @ L29 |
| turn-4 LR peak / chance | **3.16×** | **2.70×** |
| turn-4 LR L27–48 mean | 0.73 | 0.51 |
| turn-4 LR mean / chance | **2.92×** | **2.55×** |
| turn-4 NC peak | 0.66 @ L29 | 0.48 @ L48 |
| turn-4 NC peak / chance | 2.65× | 2.40× |

The ratio to chance drops from ~3× to ~2.7× but does not collapse.

## Interpretation

1. **The attractor is partly prompt-fragile.** A single added sentence
   displaces `dog` entirely, promotes `gorilla` and `kangaroo` from
   never-seen to quota-hit, and lets `penguin` peek through once. This
   is scientifically informative: it rules out the strongest version of
   the "12B prior is frozen on 4 animals" story. The model has
   non-trivial mass on other candidates when asked — it just does not
   emit them under the default wording.
2. **The attractor is also partly robust.** 14/20 bank classes remain
   at zero realization. The nudge shifted the mode but did not
   broaden it to the full bank; horse+cow+elephant together still take
   79% of mass. So the residual concentration is a real 12B property,
   not a prompt artifact.
3. **Turn-4 pre-answer LR LOO generalizes to new classes.** Kangaroo
   and gorilla were never realized in any prior 12B collection, yet
   the late-layer LR at turn 4 still decodes at 2.7× chance across
   the broader 5-class mix. This is the first generalization check we
   have on the M3 probe position, and it holds. If the decodable
   feature were keyed on e.g. a farm-animal / pet cluster unique to
   `{elephant,cow,dog,horse}`, we would expect the signal to break
   once kangaroo and gorilla dilute the set. It weakens but does not
   break.
4. **The LR-peak layer is stable.** L29–L31 remains the peak band in
   both collections; the locked-in probe position from D-31 transfers
   to the variant without re-tuning.
5. **Turn 1 strengthens a little relative to turn 4.** LR turn-1 mean
   0.43 and peak 0.51 is meaningfully closer to turn-4's 0.51 / 0.54
   than it was in the default scale-up (where turn-1 L27-48 mean was
   0.43 vs turn-4 0.73 — a large gap). One reading: with a broader
   class set, more of the commitment signal is available earlier in
   the dialogue, because there is more genuine information to encode
   rather than a near-degenerate posterior. This is suggestive only —
   the n is small and the comparison is not strictly controlled.

## Decision consequence

1. **Headline result updated.** The M3 self-chosen decodability claim
   is now: turn-4 pre-answer LR LOO ~2.7-3.2× chance across two
   realized-class regimes (4-class default, 5-class variant), peak at
   L29-L31, with the signal generalizing to classes not seen in the
   default attractor.
2. **Stop chasing diversity via prompt gymnastics.** One variant
   probe was informative; further prompt tweaking would just be
   attractor tourism. The 12B prompt-prior is real; we have
   characterized it.
3. **Move to M4.** The next artifact is causal patching at turn-4
   L29-L31 between classes, using the existing n=80 default collection
   as the cleanest base (balanced 4 classes, n=20 each). The
   less_obvious collection stays available as a supplementary dataset
   for a generalization check on the patch effect.
4. **Tag the variant class identities as a scale-axis hint.** At 4B
   the default attractor was `{salmon, frog, ...}`; at 12B default it
   is `{elephant, cow, dog, horse}`; at 12B less_obvious it shifts to
   include `{gorilla, kangaroo}`. Whether 27B+ has a narrower or
   broader default attractor, and whether the less_obvious shift
   magnitude grows or shrinks with scale, is now a concrete observable
   for the scale-up.

## Artifacts

- `runs/diag/selfchosen_ready_20bank_12b_lessobv_20260422/` (101 kept
  runs, full per-turn activations)
- `runs/m3_12b_selfchosen_lessobv_turns.json`
- `docs/progress/M3-12b-selfchosen-lessobv-detail.md`
- Job: `7246440` (600 attempts, 463s wall on 1 H100).
