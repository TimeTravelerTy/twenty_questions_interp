# STATUS

> **First file any agent reads.** The `Next concrete step` is always actionable
> without reading anything else. Update the `Last updated` line on every session.

**Current milestone:** M4 — phases 1/2a/2b sweep across {L29, L27-L48, L1-L48} layer scope at the turn-4 pre-answer position all null on both metrics. The pre-answer position is decisively not the reveal-token bottleneck. Phase 2c-i sweeps across earlier turns' pre-answer positions (free, uses existing saved activations).
**Last agent:** Claude
**Last updated:** 2026-04-25 (M4 phase 2b done in ~2min walltime on job `7260501`: patched all 48 decoder layers simultaneously at the turn-4 pre-answer position. **Null on argmax (every off-diagonal cell 0% flip), near-null on logit-diff** (deltas ±0.46 range, mostly *negative* on off-diagonals — patches push *away* from src). Cow→cow self-patch drops to 92% (horse leakage from prior fallback). Conclusion across phases 1/2a/2b: turn-4 pre-answer position is not the reveal-token bottleneck regardless of layer scope. The M3 turn-4 LR LOO 0.79 signal is *legible* but *off the causal path*. See `docs/progress/M4-patch-turn4-alllayers-null.md` and DECISIONS D-37. **Phase 2c-i:** cheap per-turn sweep using existing `turn_01..turn_04_activations.pt` files; `patch_turn4.py` to be extended with `--turn N` flag, then one bundled job runs all-layer patches at turn 1, 2, 3 pre-answer positions in sequence.)

**North star:** *Calibration is infra; the scientific claim is self-chosen only.*
Do not headline calibration-only results.

---

## Next concrete step

Prior notes, newest first:
**`docs/progress/M3-12b-selfchosen-turn4scale.md`** (n=80 scale-up on job `7232075`; turn-4 locked),
`docs/progress/M3-12b-selfchosen-turns.md` (turn-1..4 sweep on job `7230807` kept runs),
`docs/progress/M3-12b-selfchosen-direct.md` (job `7230807`; Ready direct-fit LOO local),
`docs/progress/M3-12b-selfchosen-transfer.md` (job `7230657`; plus retrospective 20-bank slice),
`docs/progress/M3-12b-pilot-readouts.md` (job `7226576`; readouts local),
`docs/progress/M3-selfchosen-20bank.md` (jobs `7223018`, `7226501`, `7226502`, `7226538`, `7226546`, `7226547`),
`docs/progress/M3-selfchosen-ready-T07.md` (job `7219788`),
`docs/progress/M3-selfchosen-ready-smoke.md` (job `7218660`),
`docs/progress/M3-h-persistence.md` (job `7218322`),
`docs/progress/M3-binding-bank-audit.md`,
`docs/progress/M3-3cond-binding-smoke.md` (job `7218265`),
`docs/progress/M3-4cond-binding-smoke.md`, `docs/progress/M3-4b-smoke-diagnostics.md`.

**Result so far (12B calibration is good; turn-4 self-chosen is the first useful probe position):**

1. **20-bank prompt beats the 4-candidate panel.** Greedy 4B no longer
   collapses to only salmon/frog. Over 200 attempts: dolphin 104 / penguin 85 /
   shark 5 / crocodile 2 / horse 2 / cow 1 / salmon 1; 7 classes appear and 5
   reach quota `n=2`.
2. **4B Ready geometry is still weak.** On the realized 5-way subset
   (`dolphin,horse,penguin,crocodile,shark`), self-chosen best post-L13 NC is
   **10.0%** (chance 20%) and post-L13 within-between contrast is **+4.70e-06**.
3. **Matched persistence is strong.** State A and B both hit **100% NC at L21**,
   cross A→B is **90%**, and State B contrast is **+4.87e-04**. Self-chosen is
   about **104× weaker than matched persistence State B** on contrast.
4. **Matched calibration still fails at 4B.** `name_paraphrase` scores only
   **34/40 = 85.0%** on the same 5-way subset, so it is not ready to standardize.
5. **12B improves self-chosen behavior, but the old 4-question panel was broken.**
   Greedy 12B over 100 attempts realizes 4 classes and reaches **37.5% NC** at
   L15, but the legacy panel is non-diagnostic there because every class shares
   the same `1,0,0,1` fingerprint.
6. **A six-question matched panel fixes that and clears the gate.** On
   `is_carnivore,is_larger_than_human,is_domesticated,lives_in_africa,produces_dairy_milk,is_ridden_by_humans`,
   matched persistence stays strong (`7226546`: 45/48 = 93.8%, NC-A 100% at L30)
   and matched calibration with **`name_paraphrase`** passes (`7226547`:
   **47/48 = 97.9%**).
7. **The narrowed 4-way self-chosen prompt is a dead end at 12B.** Re-running
   self-chosen directly on `{elephant,cow,dog,horse}` (job `7230657`) realizes
   only `cow` and `horse` over 160 attempts; `elephant` and `dog` never appear.
8. **Calibration -> self-chosen Ready transfer still fails at 12B.** On the
   balanced kept 4-way run, agreement vs reveal at `L6/L17/L27/L48` is
   NC `0.35/0.00/0.00/0.10` and LR `0.00/0.00/0.00/0.10`. On the earlier
   natural 20-bank 12B self-chosen slice (2 x `{elephant,cow,dog,horse}`),
   transfer is only `0.125-0.25`, i.e. chance.
9. **Self-chosen Ready direct-fit is also weak (decisive).** Job `7230807`,
   300 attempts on the 20-bank prompt, 40 kept runs balanced 10/class.
   LOO over all 49 layers: NC mean 0.23 / max 0.45 @ L14; LR mean 0.27 /
   max 0.45 @ L4 (chance 0.25). Best layers disagree between NC and LR,
   consistent with noise + a weak underlying signal. Calibration on the
   same 4 classes saturates at 1.00 from L6 (LR) / L27 (NC). So at 12B,
   Ready-state self-chosen geometry itself is the bottleneck, **not**
   calibration->self-chosen transfer.
10. **Turn-4 pre-answer is materially stronger than Ready.** On the same 40
    kept self-chosen runs, turn 4 pre-answer reaches NC mean **0.40** /
    max **0.625 @ L44** and LR mean **0.40** / max **0.60 @ L42**. The
    signal forms a broad late-layer band (L27-48 means: NC **0.549**,
    LR **0.539**). Turn 1 is moderate, turns 2-3 are weak. So the right
    lesson is not "later is always better"; it is that **turn 4 pre-answer**
    is the first clearly probe-usable self-chosen position we have found.
11. **This turn-4 signal is a latent-state result, not public-history leakage.**
    On the realized kept subset `{elephant,cow,dog,horse}`, the 4-question
    panel is degenerate (`1,0,0,1` for every class). So the turn-4 decode
    cannot be coming from publicly distinguishing yes/no history.
12. **Scale-up (n=80, 20/class) crystallizes the turn-4 signal decisively.**
    Job `7232075`, same 20-bank prompt, same 4-question panel. On turn 4:
    LR LOO **0.787 @ L31** / NC **0.662 @ L29** (chance 0.25). Broad
    coherent L27-48 band: **LR mean 0.731**, NC mean 0.558. Compared to
    the n=40 pilot, LR jumps +0.19 in both mean and max while NC barely
    moves — consistent with linearly separable geometry that was only
    regularization-starved at n=40. Turn 1 L27-48 LR mean 0.431 remains
    above chance but well below turn 4. The STATUS ~70% threshold for
    locking this probe position is cleared.

Bottom line: at **12B**, the self-chosen class code is decodable at
**turn-4 pre-answer, late layers (L26-L48, peaks near L29-L31)**. LR
0.79 at L31 is ~3.2x chance and the signal is coherent across depth. That
position is locked for M4.

**Pilot done (2026-04-21):** 100-run 12B `name_paraphrase` calibration on
`{elephant,cow,dog,horse}` with the six-question panel (job `7226576`). Local
Ready readouts across all 49 layers:

- **LR LOO saturates at 1.00 by L6**, stays saturated to L48.
- **NC LOO climbs 0.25→0.66 @ L7→0.93 @ L16→1.00 from L27.**
- Six binary attribute decoders all 1.00 from L7 (trivial at 4 classes × 25).

The LR ≫ NC gap at L6–L26 is still the main calibration-side structural signal:
candidate identity is linearly available from ~1/4 depth, but class clusters are
not spherical until much deeper. Transfer, however, is now the decisive result:
see `docs/progress/M3-12b-selfchosen-transfer.md`.

**Next concrete step (M4 phase 2c-i):** sweep across earlier-turn
pre-answer positions using already-saved activations. Each attempt
directory has `turn_01_activations.pt` through `turn_04_activations.pt`
on TSUBAME — same shape (49, 3840), captured at each turn's pre-answer
position. So per-turn patching is free (no new captures needed).

Plan:
1. Extend `scripts/patch_turn4.py` with `--turn N` (1..4, default 4) —
   selects which turn's pre-answer position to patch and which
   `turn_NN_activations.pt` file to load as the src vector.
   Generalize the existing `_context_up_to_q4_preanswer` to
   `_context_up_to_qN_preanswer` parameterized by N.
2. Bundle one job script that runs the all-layer patch at turn 1,
   turn 2, turn 3 in sequence (one model load amortized over three
   experiments). Output: three JSON files
   `runs/m4_patch_turnN_12b_default_L1-48all.json` for N in 1, 2, 3.

Decision tree on phase 2c-i outcomes:

- **Some turn flips reveals**: the bottleneck is at *that* turn's
  pre-answer position. Narrow further by layer band; that turn's
  pre-answer position becomes the new probe locus.
- **All three turns null**: no single-position patch on any
  pre-answer position is sufficient. Either the bottleneck is at
  positions we haven't tested (turn-0 Ready output, turn-4 question
  text positions), or class info is encoded redundantly across
  positions. Phase 2c-ii then escalates to a **position-band patch**
  with on-the-fly src capture (extend the script to forward-pass src
  through `_context_up_to_qN_preanswer` and grab residuals at the last
  K positions), patching tgt's last K positions simultaneously.

**Methodological follow-up (deferred to whichever phase produces a
positive signal):** the first-step logit-diff metric is unreliable for
classes whose reveal doesn't begin with the animal-name token (dog at
phase 2a). When a positive signal emerges, upgrade the metric to scan
several generation steps and locate where the animal-name token first
becomes argmax-favored.

**Side investigation (non-blocking):** `attempt_588` is non-deterministic
at `do_sample=False`. On-disk reveal `horse`, replayed reveal `cow`. Same
prompt, same kwargs. Suspect KV-cache or attention-impl drift in
bfloat16. Worth ~30min of investigation before relying on horse trials
in subsequent patching work.

**Do not:**
- repeat 4-way narrowed self-chosen prompts (collapse is established)
- probe "State A/B" in the self-chosen condition — the concept does not
  apply there
- spend more cycles on Ready-state self-chosen decoding as the main branch
- sweep more positions on the 80-run dataset — turn-4 L26-48 is locked
- run a bare single-position layer sweep as the next M4 experiment.
  Heimersheim & Nanda 2024 recommend low-granularity (band) interventions
  *first*, then refine. A bare layer sweep nulls under both the
  redundancy and off-path interpretations and is non-diagnostic.

**Open threads on the backlog:**
- **Bigger-model ladder (D-05):** 4B's salmon attractor may be model-specific.
  At 12B+ the distribution could sharpen differently or flatten; re-run both
  `diagnose_persistence.py` and `diagnose_selfchosen_ready.py` when we scale.
- **Bank improvement:** disputed cells in the 20×30 table
  (`docs/progress/M3-binding-bank-audit.md`) touched `frog.has_four_legs`,
  which showed up in both self-chosen smokes' 53% correctness number. Not
  blocking the 20-candidate run, still worth a broader audit pass.

Scientific ground truth remains self-chosen. Keep publishing only self-chosen
results as the headline.

Full plan is at `~/.claude/plans/here-is-a-project-calm-hummingbird.md` and
`docs/PLAN.md` (scientific).

---

## Milestone tracker

- [x] **M0 — Repo bootstrap.** Skeleton, pyproject, docs seeded. See commit history.
- [x] **M1 — Data artifacts + feasible-set utility.** 20 animals × 30 questions × 0/1
      table; pairwise-distinguishability floor relaxed to 2 (D-14); 13 tests pass.
- [x] **M2 — Ready-state decoder smoke test** (Gemma 3 1B, local CPU). 160 calibration
      + 40 self-chosen runs; LR LOO 0.38 @ L15, attribute decoder 0.89 @ L17. See
      `docs/progress/M2-ready-smoke-test.md`.
- [ ] **M3 — TSUBAME + Gemma 3 4B, full calibration dataset (~2k).** With question turns.
- [ ] **M4 — Self-chosen full study + causal patching.**
- [ ] **M5 — Transcoder / SAE feature case studies.**
- [ ] **M6 — Blog post draft.**

---

## Open questions

- HF license approval for Gemma 3 1B is pending. Without it, M2 scales only to
  open models (Qwen 2.5 0.5B, Llama-3.2 if granted, etc.). The plan keeps
  Gemma 3 as the target (D-05); using a non-Gemma model for M2's scientific
  exit criterion would break downstream SAE/transcoder reuse at M5.
- Exact middle-layer target for Gemma 3 1B (26 layers → probably layer 13–18).
  Defer: M2 captures *all* layers, picked empirically from the sweep.
- Question-regime choice for M2 dialogues (ambiguity-first vs. disambiguation-first).
  Defer to M3; M2 is Ready-only, no questions yet.
- Whether to include a 21st "I don't know / none of the above" option to catch
  non-commitment. Deferred; current bet is the model will pick one under the prompt.

---

## Known risks

- **Probe transfer from calibration to self-chosen may fail.** The core scientific
  risk. Mitigation: build the feasible-set control (`S_t`) into the first cut so
  we can ask whether the signal exceeds what public dialogue trivially reveals.
- **Instruction-following failures on Gemma 3 1B.** The 1B may not reliably emit
  `Ready` alone, may leak the secret, may refuse self-choice. Track parse-success
  rate in M2 as an explicit metric.
- **Gemma Scope / circuit-tracer maturity.** SAE artifacts exist for Gemma 3 but
  circuit-tracer's NNsight backend is flagged experimental. Plan to do readouts
  and patching *without* circuit-tracer first; add it only for the case studies.
- **Introspection unreliability.** Treat the end-of-game reveal only as sanity
  check, never as ground truth for `ẑ_0`.

---

## Handoff protocol (reminder)

1. Read this file first.
2. Do the `Next concrete step`.
3. On milestone close, write `docs/progress/M<n>-<slug>.md`.
4. Append non-obvious choices to `docs/DECISIONS.md` with today's date.
5. Update this file's `Current milestone`, `Last agent`, `Last updated`, and
   `Next concrete step`. Commit with a milestone-prefixed message (`M1: ...`).
