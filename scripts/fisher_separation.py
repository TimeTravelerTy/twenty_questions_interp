"""Normalized class separation (d-prime / Fisher) at given (anchor, layer) cells.

Reconciles "high decodability, tiny difference-of-means norm" by separating
the raw mean gap from the within-class scatter. For each cell and each class
pair (a, b):

  raw           = ||mu_a - mu_b||                      (the CAA steering norm)
  d_prime       = ||mu_a - mu_b|| / pooled_within_std   (scale-free SNR along
                  the mean-difference direction; this is what a linear probe
                  cares about)
  raw_over_act  = ||mu_a - mu_b|| / mean(||x||)         (gap relative to ambient
                  activation magnitude; what steering cares about)

Aggregates the mean over all class pairs per cell. shark droppable.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from itertools import combinations
import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in-dir", required=True)
    p.add_argument("--cells", nargs="+", required=True,
                   help="anchor:layer pairs, e.g. end_ready:16 pre_answer_q1:38")
    p.add_argument("--drop-class", nargs="+", default=None)
    p.add_argument("--n-per-class", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    pts = sorted(Path(args.in_dir).glob("attempt_*.pt"))
    pts = [p for p in pts if not p.name.endswith("_FAILED.pt")]
    drop = set(args.drop_class or [])

    first = torch.load(pts[0], map_location="cpu", weights_only=False)
    anchor_labels = list(first["anchor_labels"])

    # Group residual vectors by class for every requested cell.
    cells = []
    for c in args.cells:
        anc, lay = c.split(":")
        cells.append((anc, int(lay)))

    by_cell = {c: {} for c in cells}  # cell -> class -> list[np.ndarray]
    for p in pts:
        d = torch.load(p, map_location="cpu", weights_only=False)
        cls = d["class"]
        if cls in drop:
            continue
        res = d["residuals"]  # (n_anchors, n_layers, hidden) fp32
        for (anc, lay) in cells:
            ai = anchor_labels.index(anc)
            v = res[ai, lay, :].float().numpy()
            by_cell[(anc, lay)].setdefault(cls, []).append(v)

    for cell in cells:
        anc, lay = cell
        cls_vecs = {}
        for cls, lst in sorted(by_cell[cell].items()):
            X = np.stack(lst, 0)
            if args.n_per_class is not None:
                X = X[: args.n_per_class]
            cls_vecs[cls] = X
        classes = sorted(cls_vecs)
        # ambient activation norm
        allX = np.concatenate([cls_vecs[c] for c in classes], 0)
        act_norm = float(np.linalg.norm(allX, axis=1).mean())

        raws, dprimes, raw_over_acts = [], [], []
        for a, b in combinations(classes, 2):
            Xa, Xb = cls_vecs[a], cls_vecs[b]
            mu_a, mu_b = Xa.mean(0), Xb.mean(0)
            diff = mu_a - mu_b
            raw = float(np.linalg.norm(diff))
            u = diff / (raw + 1e-9)
            # within-class std of projections onto the unit mean-diff direction
            pa = Xa @ u
            pb = Xb @ u
            pooled_std = float(np.sqrt(0.5 * (pa.var(ddof=1) + pb.var(ddof=1))))
            dprimes.append(raw / (pooled_std + 1e-9))
            raws.append(raw)
            raw_over_acts.append(raw / act_norm)
        n = len(classes)
        print(f"\n=== {anc} @ L{lay}  ({n} classes, "
              f"{sum(len(cls_vecs[c]) for c in classes)} runs) ===")
        print(f"  mean activation norm ||x||      : {act_norm:10.2f}")
        print(f"  raw ||mu_a-mu_b|| (steering norm): {np.mean(raws):10.3f}  "
              f"(median {np.median(raws):.3f})")
        print(f"  raw / ||x||  (steering leverage) : {np.mean(raw_over_acts):10.5f}")
        print(f"  d-prime = raw / within-std (SNR) : {np.mean(dprimes):10.3f}  "
              f"(median {np.median(dprimes):.3f})")


if __name__ == "__main__":
    main()
