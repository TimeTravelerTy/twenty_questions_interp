"""M4 phase 2c-iii — Per-anchor × per-layer LR LOO and class centroids.

Reads per-run `.pt` files written by `capture_positional_residuals.py`,
runs leave-one-run-out logistic regression and nearest-centroid probes
at every (anchor, layer) cell, and writes a JSON report plus a single
compact `.pt` file with class-centroid vectors per (anchor, layer).
The class-centroid file is the steering-vector ingredient for the
phase 2d intervention (patch only along class-discriminating direction).

Outputs:
- {out_prefix}.json — anchor × layer LR LOO + NC LOO accuracy heatmaps,
  plus {anchor: peak_layer, peak_lr, peak_nc} summary.
- {out_prefix}_centroids.pt — dict with:
    "anchor_labels": list[str]
    "layers": list[int]   # 0..n_layers (49 for Gemma 3 12B)
    "class_ids": list[str]
    "centroids": tensor (n_anchors, n_layers, n_classes, hidden_dim)
    "counts": dict[class_id, int]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twenty_q.readouts import loo_accuracy_logreg, loo_accuracy_nearest_centroid


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in-dir", required=True, help="Directory of per-run capture .pt files.")
    p.add_argument("--out-prefix", required=True,
                   help="Output path prefix; writes <prefix>.json + <prefix>_centroids.pt")
    p.add_argument("--min-class-count", type=int, default=5,
                   help="Skip class probing if any class has fewer kept runs than this. "
                        "Default 5 — matches the n=20/class default scale-up balanced.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    in_dir = Path(args.in_dir).resolve()
    out_prefix = Path(args.out_prefix).resolve()
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    pts = sorted(in_dir.glob("attempt_*.pt"))
    pts = [p for p in pts if not p.name.endswith("_FAILED.pt")]
    if not pts:
        print(f"No capture files in {in_dir}", file=sys.stderr)
        return 1
    print(f"Loading {len(pts)} capture files from {in_dir}")

    # Load and verify all anchor schemas match.
    samples: list[dict] = []
    for p in pts:
        d = torch.load(p, map_location="cpu", weights_only=False)
        samples.append(d)
    anchor_labels = samples[0]["anchor_labels"]
    n_layers = samples[0]["residuals"].shape[1]
    hidden = samples[0]["residuals"].shape[2]
    for d in samples:
        if d["anchor_labels"] != anchor_labels:
            print(f"Anchor labels mismatch in {d['run_id']}: {d['anchor_labels']}", file=sys.stderr)
            return 2

    classes_present = sorted({d["class"] for d in samples})
    counts = {c: sum(1 for d in samples if d["class"] == c) for c in classes_present}
    print(f"Classes: {classes_present}  counts: {counts}")
    if any(counts[c] < args.min_class_count for c in classes_present):
        print(f"WARNING: some class has < {args.min_class_count} runs; probing may be noisy")

    chance = 1.0 / len(classes_present)

    # Stack residuals: (n_runs, n_anchors, n_layers, hidden).
    X_full = torch.stack([d["residuals"] for d in samples], dim=0).numpy()
    y = [d["class"] for d in samples]
    n_runs = X_full.shape[0]
    n_anchors = X_full.shape[1]
    print(f"X shape: {X_full.shape}; n_runs={n_runs} n_anchors={n_anchors} "
          f"n_layers={n_layers} hidden={hidden} chance={chance:.3f}")

    # Per-(anchor, layer) probe-fit.
    lr_grid = np.zeros((n_anchors, n_layers))
    nc_grid = np.zeros((n_anchors, n_layers))
    centroids = torch.zeros((n_anchors, n_layers, len(classes_present), hidden), dtype=torch.float32)

    for ai, alabel in enumerate(anchor_labels):
        for L in range(n_layers):
            X_aL = X_full[:, ai, L, :]  # (n_runs, hidden)
            try:
                lr_grid[ai, L] = loo_accuracy_logreg(X_aL, y, classes_present)
            except Exception as e:
                lr_grid[ai, L] = float("nan")
                print(f"  LR fit failed at anchor={alabel} L={L}: {e}", file=sys.stderr)
            try:
                nc_grid[ai, L] = loo_accuracy_nearest_centroid(X_aL, y, classes_present)
            except Exception as e:
                nc_grid[ai, L] = float("nan")
                print(f"  NC fit failed at anchor={alabel} L={L}: {e}", file=sys.stderr)
            # Centroid per class at this (anchor, layer)
            for ci, cid in enumerate(classes_present):
                mask = np.array([yy == cid for yy in y])
                if mask.sum() == 0:
                    continue
                centroids[ai, L, ci, :] = torch.from_numpy(X_aL[mask].mean(axis=0))
        print(f"  anchor={alabel:20s}  LR peak {lr_grid[ai].max():.3f} @ L{int(lr_grid[ai].argmax())}  "
              f"NC peak {nc_grid[ai].max():.3f} @ L{int(nc_grid[ai].argmax())}")

    # Per-anchor peaks, plus the max over the late band (L27-L48) which is what M3 highlighted.
    summary = {}
    for ai, alabel in enumerate(anchor_labels):
        lr_row = lr_grid[ai]
        nc_row = nc_grid[ai]
        late_band = slice(27, 49)
        summary[alabel] = {
            "lr_peak": float(np.nanmax(lr_row)),
            "lr_peak_layer": int(np.nanargmax(lr_row)),
            "nc_peak": float(np.nanmax(nc_row)),
            "nc_peak_layer": int(np.nanargmax(nc_row)),
            "lr_late_band_mean": float(np.nanmean(lr_row[late_band])),
            "lr_late_band_max": float(np.nanmax(lr_row[late_band])),
            "nc_late_band_mean": float(np.nanmean(nc_row[late_band])),
            "nc_late_band_max": float(np.nanmax(nc_row[late_band])),
        }

    report = {
        "in_dir": str(in_dir),
        "n_runs": n_runs,
        "classes": classes_present,
        "class_counts": counts,
        "chance": chance,
        "n_layers": n_layers,
        "hidden": hidden,
        "anchor_labels": anchor_labels,
        "lr_grid": lr_grid.tolist(),  # (n_anchors, n_layers)
        "nc_grid": nc_grid.tolist(),
        "summary": summary,
    }
    json_path = out_prefix.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {json_path}")

    centroid_path = Path(str(out_prefix) + "_centroids.pt")
    torch.save({
        "anchor_labels": anchor_labels,
        "layers": list(range(n_layers)),
        "class_ids": classes_present,
        "centroids": centroids,
        "counts": counts,
    }, centroid_path)
    print(f"Wrote {centroid_path} (centroids shape {tuple(centroids.shape)})")

    # Console heatmap of LR LOO ratio-to-chance.
    print()
    print(f"LR LOO accuracy / chance ({chance:.3f}). Cells where LR > 1.5x chance marked '*'.")
    print(f"{'anchor':22s}  " + "  ".join(f"L{L:02d}" for L in [0, 6, 13, 20, 27, 29, 31, 35, 40, 44, 48]))
    cols = [0, 6, 13, 20, 27, 29, 31, 35, 40, 44, 48]
    for ai, alabel in enumerate(anchor_labels):
        cells = []
        for L in cols:
            v = lr_grid[ai, L] / chance
            mark = "*" if v >= 1.5 else " "
            cells.append(f"{v:4.2f}{mark}")
        print(f"  {alabel:20s}  " + " ".join(cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
