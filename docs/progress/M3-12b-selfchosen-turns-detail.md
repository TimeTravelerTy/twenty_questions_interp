# M3 - 12B self-chosen turn-position decoder sweep

Run dir: `runs/diag/selfchosen_ready_20bank_12b_directfit_20260421`.
Model: google/gemma-3-12b-it. Classes: 4 (chance = 0.250).

| turn | NC mean | NC median | NC best | NC best layer | LR mean | LR median | LR best | LR best layer |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.31 | 0.35 | 0.45 | 9 | 0.31 | 0.31 | 0.47 | 47 |
| 2 | 0.21 | 0.20 | 0.50 | 16 | 0.26 | 0.26 | 0.40 | 46 |
| 3 | 0.23 | 0.20 | 0.45 | 18 | 0.23 | 0.23 | 0.38 | 15 |
| 4 | 0.40 | 0.46 | 0.62 | 44 | 0.40 | 0.45 | 0.60 | 42 |
