"""M5b — per-class probe accuracy and direction-geometry analysis.

Local analysis (no model load) over a v2 capture directory. For each
class, fits a leave-one-out logistic regression at a chosen anchor +
layer and reports per-class accuracy + the pairwise cosine similarity
of class-mean-difference directions. Used to interpret the steering
sweep: classes with high per-class probe accuracy should be the most
steerable IF the direction is causally load-bearing; pairs whose
direction is near a more populated class's direction will show
cross-talk.

Output: a small JSON with
  - per_class_lr_acc
  - per_class_mean_norm
  - direction_norm[src][tgt]
  - cosine_to_other_directions (top-2 nearest per direction)
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src-residuals-dir", required=True)
    p.add_argument("--anchor", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--out-json", required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    src_dir = Path(args.src_residuals_dir).resolve()

    # Load residuals per run at (anchor, layer).
    per_class: dict[str, list[torch.Tensor]] = defaultdict(list)
    n_files = 0
    skipped = 0
    for pt in sorted(src_dir.glob("*.pt")):
        try:
            d = torch.load(pt, map_location="cpu", weights_only=False)
        except (RuntimeError, EOFError) as e:
            skipped += 1
            continue
        cid = d.get("class")
        if cid is None:
            continue
        anchor_labels = list(d["anchor_labels"])
        if args.anchor not in anchor_labels:
            continue
        a_idx = anchor_labels.index(args.anchor)
        resid = d["residuals"]  # (K, n_layers+1, hidden)
        if resid.shape[1] <= args.layer:
            continue
        per_class[cid].append(resid[a_idx, args.layer].to(torch.float32))
        n_files += 1
    classes = sorted(per_class.keys())
    print(f"Loaded {n_files} captures, skipped {skipped} corrupted, "
          f"across {len(classes)} classes: "
          f"{ {c: len(per_class[c]) for c in classes} }")

    # Stack into X, y for LR.
    Xs, ys = [], []
    for c in classes:
        X = torch.stack(per_class[c], dim=0).numpy()
        Xs.append(X)
        ys.append(np.full(len(X), c))
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    n = X.shape[0]
    print(f"X shape: {X.shape}, classes: {classes}")

    # LOO LR. Sklearn refit per LOO fold; use class_weight='balanced' to handle
    # unequal class counts. Predict the held-out point.
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)
    confusion = defaultdict(lambda: defaultdict(int))
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        if len(set(y[mask])) < 2:
            continue
        clf = LogisticRegression(max_iter=2000, C=1.0,
                                  class_weight="balanced")
        clf.fit(X[mask], y[mask])
        pred = clf.predict(X[i:i+1])[0]
        true = y[i]
        per_class_total[true] += 1
        confusion[true][pred] += 1
        if pred == true:
            per_class_correct[true] += 1
    per_class_acc = {
        c: per_class_correct[c] / per_class_total[c] if per_class_total[c] else None
        for c in classes
    }
    overall_acc = sum(per_class_correct.values()) / sum(per_class_total.values())
    print(f"LR LOO overall acc: {overall_acc:.3f}  (chance={1/len(classes):.3f})")
    for c in classes:
        print(f"  {c}: {per_class_acc[c]:.3f}  ({per_class_correct[c]}/{per_class_total[c]})")

    # Class means + pairwise cosine of (μ_i - μ_j) directions.
    means = {c: torch.stack(per_class[c]).mean(dim=0).numpy() for c in classes}
    mean_norms = {c: float(np.linalg.norm(means[c])) for c in classes}

    direction_norm: dict[str, dict[str, float]] = {}
    direction_vec: dict[tuple[str, str], np.ndarray] = {}
    for s in classes:
        direction_norm[s] = {}
        for t in classes:
            if s == t:
                continue
            d = means[s] - means[t]
            direction_norm[s][t] = float(np.linalg.norm(d))
            direction_vec[(s, t)] = d / max(np.linalg.norm(d), 1e-9)

    # For each direction (s, t), find its top-2 most-aligned other directions.
    cosine_nbrs: dict[str, dict[str, list[tuple[str, float]]]] = {}
    keys = list(direction_vec.keys())
    for (s, t), v in direction_vec.items():
        sims = []
        for (s2, t2), v2 in direction_vec.items():
            if (s2, t2) == (s, t):
                continue
            sims.append((f"{s2}->{t2}", float(np.dot(v, v2))))
        sims.sort(key=lambda x: -x[1])
        cosine_nbrs.setdefault(s, {})[t] = sims[:3]

    # Predict which steering pairs should work best:
    # heuristic = (per_class_acc[src] + per_class_acc[tgt]) * direction_norm[src][tgt]
    pair_score = []
    for s in classes:
        for t in classes:
            if s == t:
                continue
            sa = per_class_acc.get(s) or 0
            ta = per_class_acc.get(t) or 0
            score = (sa + ta) / 2 * direction_norm[s][t]
            pair_score.append((s, t, score, direction_norm[s][t], sa, ta))
    pair_score.sort(key=lambda x: -x[2])
    print("\nTop 10 predicted-most-steerable pairs (heuristic: avg_acc * dir_norm):")
    for s, t, sc, dn, sa, ta in pair_score[:10]:
        print(f"  {s:>9} -> {t:<9} score={sc:.2f}  dir_norm={dn:.2f}  acc(src,tgt)={sa:.2f},{ta:.2f}")

    out = {
        "src_residuals_dir": str(src_dir),
        "anchor": args.anchor,
        "layer": args.layer,
        "n_runs_per_class": {c: len(per_class[c]) for c in classes},
        "lr_loo_overall_acc": overall_acc,
        "per_class_acc": per_class_acc,
        "confusion": {true: dict(preds) for true, preds in confusion.items()},
        "class_mean_norm": mean_norms,
        "direction_norm": direction_norm,
        "cosine_nearest": cosine_nbrs,
        "predicted_steerable_ranking": [
            {"src": s, "tgt": t, "score": sc, "dir_norm": dn,
             "acc_src": sa, "acc_tgt": ta}
            for s, t, sc, dn, sa, ta in pair_score
        ],
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
