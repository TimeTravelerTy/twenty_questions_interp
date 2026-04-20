# M3 — 20-bank self-chosen smoke + matched 5-way controls

**Run date:** 2026-04-20 by Codex.

Primary run:
- **Self-chosen 20-bank (4B)** — job `7223018`, `google/gemma-3-4b-it`, H100, `bfloat16`,
  greedy (`T=0.0`), 200 attempts, 4 primary questions.

Follow-up controls on the realized 5-class subset:
- **Matched persistence 5-way (4B)** — job `7226501`, `google/gemma-3-4b-it`,
  H100, `float32`, candidates `dolphin,horse,penguin,crocodile,shark`, 2 seeds.
- **Matched calibration 5-way, single schema `name_paraphrase` (4B)** —
  job `7226502`, same model/hardware/candidate subset, 2 seeds.

## Why this run mattered

The 4-candidate self-chosen panel was closed as a 4-way test at 4B because it
collapsed almost entirely to salmon/frog. The 20-bank prompt was the direct test
of whether the collapse was mostly a panel artifact or a deeper 4B competence
limit.

The follow-up matched controls answer two separate questions:

1. If the 20-bank self-chosen run realizes a broader class set, does Ready-state
   geometry on that set actually look like persistence State A or State B?
2. Does the best current calibration schema (`name_paraphrase`) finally clear the
   semantic gate when restricted to the same, more realistic self-chosen subset?

## 1. Self-chosen 20-bank (4B) — broader distribution, but still not probe-ready

### Reveal histogram over 200 attempts

| candidate | count |
|---|---:|
| `dolphin` | **104** |
| `penguin` | **85** |
| `shark` | 5 |
| `crocodile` | 2 |
| `horse` | 2 |
| `cow` | 1 |
| `salmon` | 1 |

Everything else: `0`.

This is a real improvement over the 4-candidate salmon/frog collapse: the
20-bank prompt realizes **7 classes** total and reaches quota `n=2` for **5**
classes. But the distribution is still extremely skewed. Under greedy decoding,
4B is effectively a `dolphin/penguin` chooser with a small tail.

### Balanced Ready-state analysis on the 5 realized classes

Balanced subset: `dolphin,horse,penguin,crocodile,shark` with `n=2` each.

- Best post-L13 NC layer: **L13**
- Best post-L13 NC: **10.0%** (chance is **20.0%**)
- Post-L13 within-vs-between contrast: **+4.70e-06**
- Primary correctness across kept runs: **33/48 = 68.75%**

Interpretation: the 20-bank prompt fixed the *class-collapse problem* enough to
make a 5-way comparison possible, but it did **not** put 4B self-chosen Ready
into a regime where candidate identity is cleanly separable by simple residual
geometry.

## 2. Matched persistence 5-way (4B) — entity signal remains strong

Candidates matched to the self-chosen realized subset:
`dolphin,horse,penguin,crocodile,shark`.

Topline:

- Verbalization match: **10/10**
- Primary correctness: **31/40 = 77.5%**
- Best post-L13 layer by NC-A: **L21**
- NC-A at L21: **100.0%**
- NC-B at L21: **100.0%**
- Cross A→B at L21: **90.0%**
- Post-L13 within-vs-between contrast:
  - State A: **+1.36e-02**
  - State B: **+4.87e-04**

So the matched persistence regime remains cleanly decodable on this 5-way subset,
and still shows the same strong ordering seen earlier: **A >> B**.

## 3. Matched calibration 5-way, `name_paraphrase` only (4B) — still fails the gate

This was the best current single calibration schema and the user's suggested
pragmatic branch: pick one regime and move forward **if** it passes threshold.

Result:

- Ready parse: **10/10**
- Primary correctness: **34/40 = 85.0%**
- Secondary `can_swim`: **9/10 = 90.0%**
- Post-L13 within-vs-between contrast: **+5.41e-04**

This is effectively the same semantic ceiling as the earlier 4-candidate
name-based runs (84–90%), not a breakthrough. It still misses the standing
`>=95%` gate by a wide margin.

## 4. Matched self-chosen vs persistence comparison

Using the matched 5-class persistence control:

- Persistence reference layer (best by State A): **L21**
- Self-chosen NC at L21: **0.0%**
- Persistence NC at L21: **100.0%** for both A and B
- Self-chosen post-L13 contrast: **+4.70e-06**
- Persistence State B post-L13 contrast: **+4.87e-04**
- Persistence State A post-L13 contrast: **+1.36e-02**

By the existing vote rule this is **mixed** (`nc=tie`, `contrast=state_b`), but
the scale is the important point:

> self-chosen 20-bank Ready is still about **104× weaker than persistence
> State B** on post-L13 within-vs-between contrast.

So the broader prompt improved class diversity, not representational strength.

## Decision

Two things are now established at 4B:

1. **The 20-bank prompt is better than the 4-candidate panel** for self-chosen
   diversity. Keep it.
2. **Calibration is still not ready to standardize.** Even the best single
   schema (`name_paraphrase`) fails the semantic gate on the matched realized
   subset.

Conclusion: do **not** launch the full ~2k calibration run at 4B. The next
productive branch is the model ladder, not another 4B calibration prompt sweep.

## Next step

## 5. 12B self-chosen follow-up (greedy, same 20-bank prompt)

Follow-up run after the 4B calibration failure:

- **Self-chosen 20-bank (12B)** — job `7226538`, `google/gemma-3-12b-it`,
  H100, `bfloat16`, greedy, 100 attempts, same 4-question panel.

### What changed at 12B

Reveal histogram over 100 attempts:

| candidate | count |
|---|---:|
| `cow` | **44** |
| `horse` | **29** |
| `elephant` | **18** |
| `dog` | **7** |
| everything else | 0 |

Quota `n=2` is reached for **4 classes**: `elephant,cow,dog,horse`.

Balanced Ready-state analysis on those 4 classes:

- Best post-L13 NC layer: **L15**
- Best post-L13 NC: **37.5%** (chance **25.0%**)
- Post-L13 within-vs-between contrast: **+1.08e-05**
- Primary correctness on kept runs: **32/32 = 100%**

### Why this is not the clean win it first looks like

The old 4-question panel (`is_mammal,is_bird,lives_primarily_in_water,has_four_legs`)
turns out to be **degenerate** on the realized 12B subset:

- `elephant`: `1,0,0,1`
- `cow`: `1,0,0,1`
- `dog`: `1,0,0,1`
- `horse`: `1,0,0,1`

So the perfect 32/32 correctness number is not evidence that 12B has solved the
semantic calibration problem. It mostly means the question panel stopped being
diagnostic once the model settled into a mammal-heavy cluster.

### Immediate correction

Rather than re-running self-chosen again, the right fix is to keep the realized
12B subset and switch the *matched controls* to a nontrivial question set that
actually separates these four animals:

`is_carnivore, is_larger_than_human, is_domesticated, lives_in_africa,`
`produces_dairy_milk, is_ridden_by_humans`

Those six questions give distinct fingerprints to `elephant,cow,dog,horse`.

Submitted follow-up jobs:

- **Matched persistence 12B, 4-way** — job `7226546`
- **Matched calibration 12B, `name_paraphrase`, 4-way** — job `7226547`

## 6. 12B matched controls on a discriminative 4-way panel

Both follow-up jobs finished on the realized 12B subset
`elephant,cow,dog,horse` using the six-question panel:

`is_carnivore, is_larger_than_human, is_domesticated, lives_in_africa,`
`produces_dairy_milk, is_ridden_by_humans`

### Matched persistence 12B (`7226546`)

- Verbalization match: **8/8**
- Primary correctness: **45/48 = 93.8%**
- Best post-L13 layer by NC-A: **L30**
- NC-A at L30: **100.0%**
- NC-B at L30: **75.0%**
- Cross A→B at L30: **25.0%**
- Post-L13 within-vs-between contrast:
  - State A: **+9.93e-03**
  - State B: **+2.75e-04**

So the persistence regime still shows a strong, nontrivial entity representation
at 12B on this 4-way panel.

### Matched calibration 12B, `name_paraphrase` (`7226547`)

- Ready parse: **8/8**
- Primary correctness: **47/48 = 97.9%**
- Post-L13 within-vs-between contrast: **+2.33e-04**

This is the first calibration regime that actually clears the standing
`>=95%` semantic gate.

## Decision update

At **12B**, the project now has a viable single calibration schema:
`name_paraphrase`.

That does **not** retroactively rescue 4B. It means the model ladder did what it
was supposed to do: move us into a competence regime where calibration can stop
being the blocker and become usable probe-training infrastructure.

## Current next step

Use the now-standardized **12B `name_paraphrase`** schema to launch a pilot
Ready-state calibration collection on the validated 4-way subset
`elephant,cow,dog,horse`, then train the first dense readouts at Ready on that
pilot before deciding whether to scale the same schema to a broader candidate set.
