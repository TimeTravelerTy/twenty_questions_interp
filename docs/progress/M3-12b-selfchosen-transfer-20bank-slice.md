# M3 - 12B calibration-to-self-chosen Ready transfer (20-bank slice)

Runs: 100 calibration, 8 self-chosen.
Model: google/gemma-3-12b-it. Classes: 4 (NC/LR chance = 0.250).

| layer | NC LOO | LR LOO | SC-NC | SC-LR |
|---|---|---|---|---|
| 6 | 0.52 | 1.00 | 0.12 | 0.25 |
| 17 | 0.98 | 1.00 | 0.25 | 0.25 |
| 27 | 1.00 | 1.00 | 0.25 | 0.25 |
| 48 | 1.00 | 1.00 | 0.25 | 0.25 |
