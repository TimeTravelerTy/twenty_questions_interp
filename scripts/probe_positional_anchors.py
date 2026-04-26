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
    p.add_argument("--n-per-class", type=int, default=None,
                   help="If set, subsample to first K runs per class for LR/NC LOO "
                        "(matches M3 balanced scale-up methodology). Centroids are "
                        "still computed over ALL kept runs of each class.")
    p.add_argument("--layers", type=str, default=None,
                   help="Comma-separated layer indices to probe, e.g. '0,29,48'. "
                        "Default = all 49 layers. Centroids are stored for ALL layers regardless.")
    p.add_argument("--min-class-count", type=int, default=5,
                   help="Warn if any class has fewer kept runs than this.")
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
    counts_full = {c: sum(1 for d in samples if d["class"] == c) for c in classes_present}
    print(f"Classes (all loaded): {classes_present}  counts: {counts_full}")
    if any(counts_full[c] < args.min_class_count for c in classes_present):
        print(f"WARNING: some class has < {args.min_class_count} runs; probing may be noisy")

    chance = 1.0 / len(classes_present)

    # ---- Centroids over ALL samples, per class ----
    # X_all_full[run_idx, anchor, layer, :] for centroid computation
    X_all_full = torch.stack([d["residuals"] for d in samples], dim=0).numpy()
    y_all = [d["class"] for d in samples]

    # ---- Subsample for LR/NC LOO ----
    if args.n_per_class is not None:
        # Take the first K runs of each class (sorted by run_id, deterministic)
        per_class_buckets: dict[str, list[int]] = {c: [] for c in classes_present}
        # samples is already sorted by file path = sorted by run_id.
        for i, d in enumerate(samples):
            per_class_buckets[d["class"]].append(i)
        kept_indices: list[int] = []
        for c in classes_present:
            kept_indices.extend(per_class_buckets[c][: args.n_per_class])
        kept_indices.sort()
        X_full = X_all_full[kept_indices]
        y = [y_all[i] for i in kept_indices]
        counts_loo = {c: sum(1 for yy in y if yy == c) for c in classes_present}
        print(f"LR/NC LOO subsample (n_per_class={args.n_per_class}): "
              f"counts={counts_loo} total={len(y)}")
    else:
        X_full = X_all_full
        y = y_all
        counts_loo = counts_full

    # ---- Layer subset for probing (centroids cover all layers regardless) ----
    if args.layers is not None:
        layer_indices = [int(x) for x in args.layers.split(",") if x.strip()]
    else:
        layer_indices = list(range(n_layers))
    print(f"Probing {len(layer_indices)} layer(s): {layer_indices}")

    n_runs = X_full.shape[0]
    n_anchors = X_full.shape[1]
    print(f"X shape (LOO subset): {X_full.shape}; n_runs={n_runs} n_anchors={n_anchors} "
          f"n_layers={n_layers} hidden={hidden} chance={chance:.3f}")

    # Per-(anchor, layer) probe-fit. Cells outside `layer_indices` are NaN.
    lr_grid = np.full((n_anchors, n_layers), np.nan)
    nc_grid = np.full((n_anchors, n_layers), np.nan)
    centroids = torch.zeros((n_anchors, n_layers, len(classes_present), hidden), dtype=torch.float32)

    # Centroids on ALL samples, all anchors, all layers.
    y_all_arr = np.array(y_all)
    for ci, cid in enumerate(classes_present):
        mask = (y_all_arr == cid)
        if mask.sum() == 0:
            continue
        # X_all_full has shape (n_all, n_anchors, n_layers, hidden); centroid (n_anchors, n_layers, hidden)
        centroids[:, :, ci, :] = torch.from_numpy(X_all_full[mask].mean(axis=0))

    for ai, alabel in enumerate(anchor_labels):
        for L in layer_indices:
            X_aL = X_full[:, ai, L, :]  # (n_runs_loo, hidden)
            try:
                lr_grid[ai, L] = loo_accuracy_logreg(X_aL, y, classes_present)
            except Exception as e:
                print(f"  LR fit failed at anchor={alabel} L={L}: {e}", file=sys.stderr)
            try:
                nc_grid[ai, L] = loo_accuracy_nearest_centroid(X_aL, y, classes_present)
            except Exception as e:
                print(f"  NC fit failed at anchor={alabel} L={L}: {e}", file=sys.stderr)
        # Per-anchor peak across the probed layers only
        probed = lr_grid[ai, layer_indices]
        nc_probed = nc_grid[ai, layer_indices]
        lr_peak_idx = int(np.nanargmax(probed)) if not np.all(np.isnan(probed)) else 0
        nc_peak_idx = int(np.nanargmax(nc_probed)) if not np.all(np.isnan(nc_probed)) else 0
        print(f"  anchor={alabel:20s}  LR peak {probed[lr_peak_idx]:.3f} @ L{layer_indices[lr_peak_idx]}  "
              f"NC peak {nc_probed[nc_peak_idx]:.3f} @ L{layer_indices[nc_peak_idx]}")

    # Per-anchor peaks. Use only the layers we actually probed; report nan-safe.
    summary = {}
    for ai, alabel in enumerate(anchor_labels):
        lr_row = lr_grid[ai]
        nc_row = nc_grid[ai]
        finite_lr = ~np.isnan(lr_row)
        finite_nc = ~np.isnan(nc_row)
        late_band = [L for L in layer_indices if 27 <= L <= 48]
        summary[alabel] = {
            "lr_peak": float(np.nanmax(lr_row)) if finite_lr.any() else None,
            "lr_peak_layer": int(np.nanargmax(lr_row)) if finite_lr.any() else None,
            "nc_peak": float(np.nanmax(nc_row)) if finite_nc.any() else None,
            "nc_peak_layer": int(np.nanargmax(nc_row)) if finite_nc.any() else None,
            "lr_late_band_mean": float(np.nanmean(lr_row[late_band])) if late_band else None,
            "lr_late_band_max": float(np.nanmax(lr_row[late_band])) if late_band else None,
            "nc_late_band_mean": float(np.nanmean(nc_row[late_band])) if late_band else None,
            "nc_late_band_max": float(np.nanmax(nc_row[late_band])) if late_band else None,
        }

    # JSON-safe grids (nan -> None)
    def _grid_to_jsonable(g: np.ndarray) -> list:
        return [[None if np.isnan(v) else float(v) for v in row] for row in g]

    report = {
        "in_dir": str(in_dir),
        "n_runs_total_loaded": len(samples),
        "n_runs_loo": n_runs,
        "n_per_class_loo": args.n_per_class,
        "layer_indices_probed": layer_indices,
        "classes": classes_present,
        "class_counts_full": counts_full,
        "class_counts_loo": counts_loo,
        "chance": chance,
        "n_layers": n_layers,
        "hidden": hidden,
        "anchor_labels": anchor_labels,
        "lr_grid": _grid_to_jsonable(lr_grid),  # (n_anchors, n_layers); None outside probed cols
        "nc_grid": _grid_to_jsonable(nc_grid),
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
    print(f"LR LOO accuracy / chance ({chance:.3f}). Cells where LR > 1.5x chance marked '*'; '   .' = not probed.")
    cols = [L for L in [0, 6, 13, 20, 27, 29, 31, 35, 40, 44, 48] if L in layer_indices]
    if not cols:
        cols = layer_indices[: min(len(layer_indices), 11)]
    print(f"  {'anchor':20s}  " + "  ".join(f"L{L:02d}" for L in cols))
    for ai, alabel in enumerate(anchor_labels):
        cells = []
        for L in cols:
            v = lr_grid[ai, L]
            if np.isnan(v):
                cells.append("   .")
            else:
                ratio = v / chance
                mark = "*" if ratio >= 1.5 else " "
                cells.append(f"{ratio:4.2f}{mark}")
        print(f"  {alabel:20s}  " + " ".join(cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
