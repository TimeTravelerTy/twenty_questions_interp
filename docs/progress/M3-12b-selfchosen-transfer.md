# M3 — 12B calibration -> self-chosen Ready transfer fails

**Run date:** 2026-04-21 by Codex.
**Calibration source:** `runs/calibration/12b_name_paraphrase_4way_pilot_20260420/`
(100 runs = 25 x `{elephant,cow,dog,horse}`).
**New self-chosen run:** job `7230657`, artifacts under
`runs/diag/selfchosen_ready_12b_4way_transfer_20260421/`.
**Supporting slice:** retrospective transfer on the earlier 20-bank 12B run,
`runs/diag/selfchosen_ready_20bank_12b_20260420/`.
**Model:** `google/gemma-3-12b-it`, H100, bfloat16.

## Scope

Test the first real scientific question opened by the successful 12B calibration
pilot: do Ready-state probes trained on calibration transfer to self-chosen
Ready at the same model size?

The calibration side is fixed from `M3-12b-pilot-readouts.md`: LR LOO is
perfect from L6, NC is nearly perfect by L17 and perfect by L27. We therefore
score transfer at the four decisive layers:

- **L6** — earliest layer where LR is perfect on calibration
- **L17** — NC turn-on layer
- **L27** — NC saturation layer
- **L48** — final layer

Two self-chosen testbeds were checked:

1. A **new explicit 4-way self-chosen run** on
   `{elephant,cow,dog,horse}` using the discriminative six-question panel:
   `is_carnivore,is_larger_than_human,is_domesticated,lives_in_africa,`
   `produces_dairy_milk,is_ridden_by_humans`.
2. A **retrospective slice of the earlier 20-bank 12B self-chosen run**,
   restricted to the kept reveals on the same four animals.

That second check matters because Ready is captured **before** any question
turns, so the earlier 20-bank run is still valid evidence for Ready-state
transfer even though its later question panel was different.

## Result

### 1. New explicit 4-way self-chosen run (`7230657`)

The narrowed self-chosen prompt did **not** broaden the realized class set.
It collapsed harder than the natural 20-bank prompt:

- 160 attempts
- kept counts: `cow=10`, `horse=10`, `elephant=0`, `dog=0`
- primary correctness on kept runs: `120/120 = 100%`

Transfer from calibration probes onto the balanced kept 20-run self-chosen
subset:

| layer | calibration NC LOO | calibration LR LOO | self-chosen NC agree | self-chosen LR agree |
|---|---:|---:|---:|---:|
| 6 | 0.52 | 1.00 | 0.35 | 0.00 |
| 17 | 0.98 | 1.00 | 0.00 | 0.00 |
| 27 | 1.00 | 1.00 | 0.00 | 0.00 |
| 48 | 1.00 | 1.00 | 0.10 | 0.10 |

This is bad even relative to the **collapsed realized set**. Since only
`cow/horse` appear, a constant predictor over those two classes would score
50%; every transferred probe is below that.

The failure mode is not random noise. The transferred decoders mostly collapse
the entire self-chosen set onto a single **absent** calibration class:

- L6 LR predicts **`elephant` for all 20 runs**
- L17 NC and LR both predict **`elephant` for all 20 runs**
- L27 NC and LR both predict **`dog` for all 20 runs**

### 2. Retrospective 20-bank slice on the same four animals

From the earlier 20-bank 12B self-chosen run, the kept slice contains exactly
2 runs each of `{elephant,cow,dog,horse}` (8 total). Transfer on that natural
self-chosen prompt is still poor:

| layer | calibration NC LOO | calibration LR LOO | self-chosen NC agree | self-chosen LR agree |
|---|---:|---:|---:|---:|
| 6 | 0.52 | 1.00 | 0.125 | 0.25 |
| 17 | 0.98 | 1.00 | 0.25 | 0.25 |
| 27 | 1.00 | 1.00 | 0.25 | 0.25 |
| 48 | 1.00 | 1.00 | 0.25 | 0.25 |

Chance here is **25%**, and that is essentially where the transfer stays.

Again the failure is class collapse, not merely low confidence:

- L6 LR predicts **`cow` for all 8 runs**
- L17 NC and LR both predict **`elephant` for all 8 runs**
- L27 NC and LR both predict **`dog` for all 8 runs**
- L48 NC predicts **`dog` for all 8 runs** while LR predicts **`elephant` for all 8**

## Interpretation

This is the decisive result for the current branch:

> **At 12B, a calibration regime can be internally excellent and still fail to
> transfer to self-chosen Ready.**

The 100-run calibration pilot is not weak. LRs are perfect from L6 onward and
NC is perfect from L27 onward. Yet when those same readouts are applied to
self-chosen Ready, agreement is chance or worse on the natural 20-bank slice
and even worse on the narrowed 4-way prompt.

So the D-23 lesson was **not** just a 4B artifact. Probe-training location
still matters at 12B:

- calibration Ready is usable as **infrastructure**
- calibration-trained Ready probes are **not** usable as self-chosen readouts

The explicit 4-way self-chosen prompt should also be treated as a dead end for
the main line. It collapses the reveal distribution from
`{elephant,cow,dog,horse}` on the 20-bank prompt down to only `{cow,horse}`.

## Decision consequence

1. Do **not** use calibration-trained Ready probes as the headline readout on
   self-chosen 12B runs.
2. Do **not** keep narrowing the self-chosen prompt to the realized 4-way
   subset. The natural 20-bank prompt is better behaved.
3. The next productive step is a larger **12B self-chosen 20-bank** collection
   and **direct** Ready-state probe fitting on self-chosen reveal labels.

## Artifacts

- Supporting 20-bank slice table:
  `docs/progress/M3-12b-selfchosen-transfer-20bank-slice.md`
- Machine-readable reports:
  `runs/m3_12b_selfchosen_transfer.json`,
  `runs/m3_12b_selfchosen_transfer_20bank.json`
