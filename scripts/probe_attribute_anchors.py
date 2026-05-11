"""M5 Phase A4b (attribute-bundle hypothesis) — per-anchor x per-layer
binary LR LOO for EACH bank attribute, on the v2 self-chosen capture.

Tests whether the residual encodes individual *answer-relevant binary
attributes* (the "answer-sufficient attribute bundle" framing from the
original plan) more readily than the 4-way class identity that M3
already measured.

For each capture run, we look up the binary value of each requested
attribute from data/answers.csv using the run's reveal class. The
attribute values for {cow, dog, elephant, horse} under the matched
6-question panel are mostly 3v1 splits, plus one 2v2 (`is_ridden_by_humans`).
We then fit a binary LR LOO at each (anchor, layer, attribute) cell.

Output JSON contains:
- per-attribute (n_anchors x n_layers) accuracy grid
- per-attribute peak (best anchor, best layer)
- the per-class attribute table actually used (for sanity)
- the 4-way class LR LOO grid as a baseline reference (optional via --include-class)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twenty_q.banks import load_bank
from twenty_q.readouts import loo_accuracy_binary, loo_accuracy_logreg


DEFAULT_ATTRIBUTES = (
    "is_carnivore,is_larger_than_human,is_domesticated,"
    "lives_in_africa,produces_dairy_milk,is_ridden_by_humans"
)
DEFAULT_CLASSES = "cow,dog,elephant,horse"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in-dir", required=True,
                   help="Directory of capture_positional_residuals .pt files.")
    p.add_argument("--out", required=True, help="Output JSON path.")
    p.add_argument("--attributes", default=DEFAULT_ATTRIBUTES,
                   help="Comma-separated bank question ids.")
    p.add_argument("--classes", default=DEFAULT_CLASSES,
                   help="Comma-separated reveal-class allowlist; runs with "
                        "other reveal classes are dropped.")
    p.add_argument("--n-per-class", type=int, default=20,
                   help="Balanced subsample per class.")
    p.add_argument("--layers", default=None,
                   help="Comma-separated layer indices to probe. "
                        "Default = all 49.")
    p.add_argument("--lr-c", type=float, default=1.0)
    p.add_argument("--include-class", action="store_true",
                   help="Also compute 4-way class LR LOO at each "
                        "(anchor, layer) cell for reference.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    in_dir = Path(args.in_dir).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bank = load_bank()
    attr_ids = [s.strip() for s in args.attributes.split(",") if s.strip()]
    class_allow = [s.strip() for s in args.classes.split(",") if s.strip()]
    for cid in class_allow:
        for aid in attr_ids:
            # validate bank lookup
            _ = bank.answer(cid, aid)

    # Per-class attribute table (sanity)
    attr_table = {cid: {aid: int(bank.answer(cid, aid)) for aid in attr_ids}
                  for cid in class_allow}

    pts = sorted(in_dir.glob("attempt_*.pt"))
    pts = [p for p in pts if not p.name.endswith("_FAILED.pt")]
    if not pts:
        print(f"No capture files in {in_dir}", file=sys.stderr)
        return 1

    # Pass 1: filter to allowed classes; balance.
    print(f"Pass 1: scanning {len(pts)} capture files...")
    keep: list[tuple[Path, str]] = []
    for p in pts:
        d = torch.load(p, map_location="cpu", weights_only=False)
        cls = d.get("class")
        if cls in class_allow:
            keep.append((p, cls))
        del d

    by_class: dict[str, list[Path]] = {c: [] for c in class_allow}
    for p, c in keep:
        by_class[c].append(p)

    counts_full = {c: len(by_class[c]) for c in class_allow}
    print(f"Class counts (all kept): {counts_full}")
    for c in class_allow:
        if len(by_class[c]) < args.n_per_class:
            print(f"ERROR: class {c} has {len(by_class[c])} runs, need "
                  f"{args.n_per_class}", file=sys.stderr)
            return 2

    # Deterministic balanced subset: first N per class
    selection: list[tuple[Path, str]] = []
    for c in class_allow:
        for p in by_class[c][: args.n_per_class]:
            selection.append((p, c))
    print(f"Balanced subset: {len(selection)} runs "
          f"({args.n_per_class}/class)")

    # Pass 2: load residuals into one big array (n_runs, n_anchors, n_layers, hidden)
    first = torch.load(selection[0][0], map_location="cpu", weights_only=False)
    anchor_labels = first["anchor_labels"]
    n_anchors, n_layers, hidden = first["residuals"].shape
    print(f"Schema: n_anchors={n_anchors} ({anchor_labels}) n_layers={n_layers} hidden={hidden}")

    X_full = np.zeros((len(selection), n_anchors, n_layers, hidden), dtype=np.float32)
    y_class: list[str] = []
    for i, (p, c) in enumerate(selection):
        d = torch.load(p, map_location="cpu", weights_only=False)
        X_full[i] = d["residuals"].numpy()
        y_class.append(c)
        del d

    if args.layers is not None:
        layer_indices = [int(x) for x in args.layers.split(",") if x.strip()]
    else:
        layer_indices = list(range(n_layers))
    print(f"Probing {len(layer_indices)} layers x {n_anchors} anchors x "
          f"{len(attr_ids)} attributes = "
          f"{len(layer_indices) * n_anchors * len(attr_ids)} LR LOO fits")

    # Per-attribute binary label vector
    attr_labels: dict[str, list[int]] = {
        aid: [int(bank.answer(c, aid)) for c in y_class] for aid in attr_ids
    }
    # Majority baseline per attribute
    majority = {aid: float(max(np.mean(v), 1 - np.mean(v)))
                for aid, v in attr_labels.items()}
    print(f"Attribute majority baselines: {majority}")

    # Allocate output grids
    attr_grids = {aid: np.full((n_anchors, n_layers), np.nan)
                  for aid in attr_ids}
    class_grid = np.full((n_anchors, n_layers), np.nan) if args.include_class else None

    t0 = time.time()
    total_cells = n_anchors * len(layer_indices)
    cell_idx = 0
    for ai, alab in enumerate(anchor_labels):
        for L in layer_indices:
            X = X_full[:, ai, L, :]  # (n_runs, hidden)
            for aid in attr_ids:
                acc, _maj = loo_accuracy_binary(X, attr_labels[aid], C=args.lr_c)
                attr_grids[aid][ai, L] = acc
            if class_grid is not None:
                try:
                    class_grid[ai, L] = loo_accuracy_logreg(X, y_class, class_allow, C=args.lr_c)
                except Exception as e:
                    print(f"  class LR fit failed at anchor={alab} L={L}: {e}",
                          file=sys.stderr)
            cell_idx += 1
            if cell_idx % 50 == 0 or cell_idx == total_cells:
                elapsed = time.time() - t0
                rate = cell_idx / max(elapsed, 1e-6)
                eta = (total_cells - cell_idx) / max(rate, 1e-6)
                print(f"  [{cell_idx}/{total_cells}] anchor={alab} L={L} "
                      f"({rate:.2f} cells/s, ETA {eta/60:.1f}min)", flush=True)
        # Per-attribute peak this anchor
        peak_per_attr = {
            aid: float(np.nanmax(attr_grids[aid][ai, layer_indices]))
            for aid in attr_ids
        }
        print(f"  anchor={alab:20s}  peaks: " +
              "  ".join(f"{aid[:18]}={peak_per_attr[aid]:.3f}"
                       for aid in attr_ids))

    elapsed = time.time() - t0
    print(f"\nAll cells done in {elapsed:.1f}s")

    # Build summary
    def _grid_to_jsonable(g: np.ndarray) -> list:
        return [[None if np.isnan(v) else float(v) for v in row] for row in g]

    summary_per_attr = {}
    for aid in attr_ids:
        g = attr_grids[aid]
        finite = ~np.isnan(g)
        if not finite.any():
            summary_per_attr[aid] = None
            continue
        flat_idx = int(np.nanargmax(g))
        ai, L = divmod(flat_idx, n_layers)
        per_anchor_peak = []
        for ai2, alab in enumerate(anchor_labels):
            row = g[ai2]
            row_finite = ~np.isnan(row)
            if not row_finite.any():
                per_anchor_peak.append(None)
                continue
            L_peak = int(np.nanargmax(row))
            per_anchor_peak.append({
                "anchor": alab,
                "peak_acc": float(row[L_peak]),
                "peak_layer": L_peak,
                "late_band_mean": float(np.nanmean([row[L] for L in range(27, 49)])),
            })
        summary_per_attr[aid] = {
            "global_peak_acc": float(g[ai, L]),
            "global_peak_anchor": anchor_labels[ai],
            "global_peak_layer": L,
            "majority_baseline": majority[aid],
            "per_anchor": per_anchor_peak,
        }

    out = {
        "in_dir": str(in_dir),
        "classes": class_allow,
        "n_per_class": args.n_per_class,
        "class_counts_full": counts_full,
        "attributes": attr_ids,
        "attribute_table": attr_table,
        "anchor_labels": anchor_labels,
        "n_layers": n_layers,
        "layer_indices_probed": layer_indices,
        "lr_c": args.lr_c,
        "elapsed_s": elapsed,
        "majority_per_attribute": majority,
        "attr_grids": {aid: _grid_to_jsonable(attr_grids[aid]) for aid in attr_ids},
        "class_grid": _grid_to_jsonable(class_grid) if class_grid is not None else None,
        "summary_per_attribute": summary_per_attr,
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
