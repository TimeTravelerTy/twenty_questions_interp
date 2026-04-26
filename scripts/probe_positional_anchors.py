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
    print(f"Streaming over {len(pts)} capture files from {in_dir}")

    # Streaming pass: discover schema from first file, accumulate per-class
    # centroid sums + counts, and stash residuals for runs in the LOO subsample.
    # This keeps peak memory at: centroid_sums (~36 MB) + LOO_subsample stack
    # (~720 MB for 80x12x49x3840 fp32) + transient one-file (~4.5 MB).
    first = torch.load(pts[0], map_location="cpu", weights_only=False)
    anchor_labels = first["anchor_labels"]
    n_layers = first["residuals"].shape[1]
    hidden = first["residuals"].shape[2]
    n_anchors = len(anchor_labels)
    del first

    # Per-class running sums for centroid computation. Keys filled as we discover classes.
    centroid_sums: dict[str, torch.Tensor] = {}
    centroid_counts: dict[str, int] = {}

    # First pass to enumerate classes + collect (run_id, class) → file path.
    # We do this by reading each file twice (once for class metadata, once
    # for residuals) to avoid holding 600 dicts in memory.
    print("Pass 1: scanning class labels...")
    file_class: list[tuple[Path, str, str]] = []
    for p in pts:
        d = torch.load(p, map_location="cpu", weights_only=False)
        if d["anchor_labels"] != anchor_labels:
            print(f"Anchor labels mismatch in {d['run_id']}: {d['anchor_labels']}", file=sys.stderr)
            return 2
        file_class.append((p, d["class"], d["run_id"]))
        del d
    classes_present = sorted({c for _, c, _ in file_class})
    counts_full = {c: sum(1 for _, cc, _ in file_class if cc == c) for c in classes_present}
    print(f"Classes (all loaded): {classes_present}  counts: {counts_full}")
    if any(counts_full[c] < args.min_class_count for c in classes_present):
        print(f"WARNING: some class has < {args.min_class_count} runs; probing may be noisy")
    chance = 1.0 / len(classes_present)

    # Decide LOO subsample membership now (deterministic: first K per class).
    loo_indices_in_files: list[int] = []
    if args.n_per_class is not None:
        per_class_buckets: dict[str, list[int]] = {c: [] for c in classes_present}
        for i, (_, c, _) in enumerate(file_class):
            per_class_buckets[c].append(i)
        for c in classes_present:
            loo_indices_in_files.extend(per_class_buckets[c][: args.n_per_class])
        loo_indices_in_files.sort()
    else:
        loo_indices_in_files = list(range(len(file_class)))
    loo_set = set(loo_indices_in_files)
    counts_loo = {c: sum(1 for i in loo_indices_in_files if file_class[i][1] == c) for c in classes_present}
    print(f"LR/NC LOO subsample (n_per_class={args.n_per_class}): "
          f"counts={counts_loo} total={len(loo_indices_in_files)}")

    # Pass 2: stream residuals; accumulate per-class sums; collect LOO subsample.
    print("Pass 2: streaming residuals (centroid sums + LOO subsample)...")
    loo_residuals_list: list[torch.Tensor] = []
    loo_y: list[str] = []
    import time as _time
    t0 = _time.time()
    for i, (p, cls, _rid) in enumerate(file_class):
        d = torch.load(p, map_location="cpu", weights_only=False)
        r = d["residuals"]  # (n_anchors, n_layers, hidden) fp32
        # Centroid running sum
        if cls not in centroid_sums:
            centroid_sums[cls] = torch.zeros((n_anchors, n_layers, hidden), dtype=torch.float32)
            centroid_counts[cls] = 0
        centroid_sums[cls] += r
        centroid_counts[cls] += 1
        if i in loo_set:
            loo_residuals_list.append(r.clone())
            loo_y.append(cls)
        del d, r
        if (i + 1) % 100 == 0:
            print(f"  pass2 [{i+1}/{len(file_class)}] elapsed {_time.time()-t0:.1f}s", flush=True)
    print(f"Pass 2 done in {_time.time()-t0:.1f}s")

    # Build centroids tensor: (n_anchors, n_layers, n_classes, hidden)
    centroids = torch.zeros((n_anchors, n_layers, len(classes_present), hidden), dtype=torch.float32)
    for ci, cid in enumerate(classes_present):
        if centroid_counts.get(cid, 0) == 0:
            continue
        centroids[:, :, ci, :] = centroid_sums[cid] / centroid_counts[cid]
    # Free per-class sums
    centroid_sums.clear()

    # Stack LOO subsample: (n_loo, n_anchors, n_layers, hidden)
    if loo_residuals_list:
        X_full = torch.stack(loo_residuals_list, dim=0).numpy()
        loo_residuals_list.clear()
    else:
        print("No LOO subsample collected", file=sys.stderr)
        return 3
    y = loo_y

    # ---- Layer subset for probing (centroids cover all layers regardless) ----
    if args.layers is not None:
        layer_indices = [int(x) for x in args.layers.split(",") if x.strip()]
    else:
        layer_indices = list(range(n_layers))
    print(f"Probing {len(layer_indices)} layer(s): {layer_indices}")

    n_runs = X_full.shape[0]
    print(f"X shape (LOO subset): {X_full.shape}; n_runs={n_runs} n_anchors={n_anchors} "
          f"n_layers={n_layers} hidden={hidden} chance={chance:.3f}")

    # Per-(anchor, layer) probe-fit. Cells outside `layer_indices` are NaN.
    lr_grid = np.full((n_anchors, n_layers), np.nan)
    nc_grid = np.full((n_anchors, n_layers), np.nan)

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
        "n_runs_total_loaded": len(file_class),
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
        "counts": counts_full,
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
