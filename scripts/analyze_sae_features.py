"""M5 Phase A3.1 + A4 — SAE feature analysis for class-discriminative
firings and probe-reproducibility.

Reads the firings file written by `sae_feature_firings.py` and produces
two analyses keyed off a single (anchor, layer, SAE) slice — typically
`pre_answer_q4` × Gemma Scope 2 12b-it resid_post layer_31:

  A3.1 — Per-feature class-discrimination ranking. Computes per-class
         mean activation and per-class firing rate, then an ANOVA-style
         F statistic over class-conditioned activations. Output: top-N
         features ranked by F, with effect sizes and per-class means.
  A4   — Sparse-classifier reproducibility. Trains a 4-class logistic
         regression with leave-one-out CV on the sparse SAE feature
         matrix (n_runs × d_sae). Reports LOO accuracy and compares
         against the M3 residual probe LR LOO at the matching (anchor,
         layer) — the AxBench question.

Notes
-----
- A3.2 (turn-progressive) and A3.3 (yes/no-conditional) need multi-anchor
  firings (currently encoded one anchor per run). Add later by calling
  `sae_feature_firings.py` over more anchors and extending this script.
- A4 calibration-attribute pickup (A4b) needs M3 calibration runs through
  the same SAE; deferred until first-cut self-chosen result is in.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--firings", required=True,
                   help="Output of sae_feature_firings.py (.pt).")
    p.add_argument("--anchor", default=None,
                   help="If firings cover multiple anchors, restrict to "
                        "this anchor label.")
    p.add_argument("--top-n", type=int, default=30,
                   help="How many top class-discriminative features to "
                        "report in the JSON output.")
    p.add_argument("--lr-c", type=float, default=1.0,
                   help="LogisticRegression C parameter for the sparse "
                        "classifier.")
    p.add_argument("--balance", type=int, default=None,
                   help="If set, randomly subsample N runs per class for "
                        "the analysis (apples-to-apples with M3's "
                        "balanced 20/class LR LOO).")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed for --balance subsampling.")
    p.add_argument("--drop-class", nargs="+", default=None,
                   help="Class name(s) to exclude (e.g. degenerate 'shark' "
                        "with too few samples to balance/LOO).")
    p.add_argument("--out", required=True, help="Output JSON path.")
    return p.parse_args()


def _build_sparse_matrix(records: list[dict], anchor: str | None,
                         balance: int | None = None, seed: int = 0
                         ) -> tuple[
    np.ndarray, list[str], np.ndarray, list[str], int, list[int]
]:
    """Build a dense (n_runs, d_sae_active) feature matrix from sparse
    top-k records, plus a global feature-id list (only features that
    fire in at least one run) and a class array.

    If `balance` is set, randomly subsample `balance` runs per class
    (deterministic in `seed`) to produce a balanced K-way pool.

    Returns (X, class_names, classes_idx, run_ids, d_sae_observed_max).
    """
    if anchor is not None:
        records = [r for r in records if r["anchor"] == anchor]
    if not records:
        raise SystemExit("No records left after anchor filter")
    if balance is not None:
        rng = np.random.default_rng(seed)
        by_class: dict[str, list] = {}
        for r in records:
            by_class.setdefault(str(r["class"]), []).append(r)
        balanced: list = []
        for cls, items in sorted(by_class.items()):
            if len(items) < balance:
                raise SystemExit(
                    f"Class {cls!r} has only {len(items)} records, "
                    f"cannot --balance to {balance}"
                )
            order = rng.permutation(len(items))
            balanced.extend(items[i] for i in order[:balance])
        records = balanced

    feat_set: set[int] = set()
    for r in records:
        feat_set.update(int(i) for i in r["feature_idx"].tolist())
    feat_ids = sorted(feat_set)
    feat_index = {fid: j for j, fid in enumerate(feat_ids)}

    n_runs = len(records)
    X = np.zeros((n_runs, len(feat_ids)), dtype=np.float32)
    classes_str: list[str] = []
    run_ids: list[str] = []
    for i, r in enumerate(records):
        idx = r["feature_idx"].tolist()
        act = r["activation"].tolist()
        for fid, a in zip(idx, act):
            j = feat_index[int(fid)]
            X[i, j] = float(a)
        classes_str.append(str(r["class"]))
        run_ids.append(str(r["run_id"]))

    class_names = sorted(set(classes_str))
    name_to_idx = {n: k for k, n in enumerate(class_names)}
    y = np.array([name_to_idx[c] for c in classes_str], dtype=np.int64)
    max_seen = max(feat_ids) + 1 if feat_ids else 0
    return X, class_names, y, run_ids, max_seen, feat_ids


def _per_feature_anova(X: np.ndarray, y: np.ndarray, class_names: list[str]
                       ) -> list[dict]:
    """Per-feature one-way ANOVA F over class-conditioned activations.

    Reports F, effect size (max - min class mean), per-class mean activations
    and per-class firing rates. Stable on sparse zero-heavy columns
    (returns F=0 when within-class variance is zero across all classes).
    """
    n, d = X.shape
    K = len(class_names)
    out = []
    grand_mean = X.mean(axis=0)
    # Per-class means and variances
    per_class_mean = np.zeros((K, d), dtype=np.float64)
    per_class_var = np.zeros((K, d), dtype=np.float64)
    per_class_n = np.zeros(K, dtype=np.int64)
    per_class_fire_rate = np.zeros((K, d), dtype=np.float64)
    for k in range(K):
        mask = (y == k)
        per_class_n[k] = mask.sum()
        if mask.sum() == 0:
            continue
        per_class_mean[k] = X[mask].mean(axis=0)
        per_class_var[k] = X[mask].var(axis=0, ddof=1) if mask.sum() > 1 else 0.0
        per_class_fire_rate[k] = (X[mask] > 0).mean(axis=0)
    # Between-group SS
    diff_to_grand = per_class_mean - grand_mean[None, :]
    ss_between = (per_class_n[:, None] * (diff_to_grand ** 2)).sum(axis=0)
    # Within-group SS
    ss_within = (per_class_var * np.maximum(per_class_n[:, None] - 1, 0)).sum(axis=0)
    df_between = K - 1
    df_within = max(int(per_class_n.sum() - K), 1)
    ms_between = ss_between / df_between
    ms_within = np.where(ss_within > 0, ss_within / df_within, np.inf)
    F = np.where(ms_within > 0, ms_between / ms_within, 0.0)
    # Effect size: max - min class mean
    eff = per_class_mean.max(axis=0) - per_class_mean.min(axis=0)
    for j in range(d):
        out.append({
            "feat_col": j,
            "F": float(F[j]),
            "effect": float(eff[j]),
            "per_class_mean": [float(x) for x in per_class_mean[:, j]],
            "per_class_fire_rate": [float(x) for x in per_class_fire_rate[:, j]],
        })
    return out


def _loo_lr(X: np.ndarray, y: np.ndarray, C: float) -> dict:
    """4-class logistic regression with leave-one-out CV.

    Uses sklearn; returns per-class accuracy + overall LOO accuracy.
    """
    from sklearn.linear_model import LogisticRegression  # type: ignore
    n, d = X.shape
    preds = np.zeros(n, dtype=np.int64)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
        clf.fit(X[mask], y[mask])
        preds[i] = clf.predict(X[i:i+1])[0]
    acc = float((preds == y).mean())
    K = int(y.max()) + 1
    per_class = []
    for k in range(K):
        m = (y == k)
        per_class.append(float((preds[m] == k).mean()) if m.sum() else float("nan"))
    return {"loo_accuracy": acc, "per_class_accuracy": per_class, "n": int(n)}


def main() -> int:
    args = parse_args()
    payload = torch.load(args.firings, map_location="cpu", weights_only=False)
    records = payload["records"]
    meta = payload.get("meta", {})

    print(f"Loaded {len(records)} firing records from {args.firings}")
    if args.anchor:
        print(f"Filtering to anchor={args.anchor}")
    if args.drop_class:
        before = len(records)
        records = [r for r in records if str(r["class"]) not in set(args.drop_class)]
        print(f"Dropped classes {args.drop_class}: {before} -> {len(records)} records")

    X, class_names, y, run_ids, d_sae_observed, feat_ids = _build_sparse_matrix(
        records, anchor=args.anchor, balance=args.balance, seed=args.seed,
    )
    print(f"Built feature matrix: shape={X.shape}, "
          f"classes={class_names}, n_per_class={[int((y==k).sum()) for k in range(len(class_names))]}")
    print(f"Active features (any nonzero): {X.shape[1]} of d_sae={meta.get('d_sae')}")

    t0 = time.time()
    feat_stats = _per_feature_anova(X, y, class_names)
    feat_stats.sort(key=lambda d: d["F"], reverse=True)
    top = feat_stats[: args.top_n]

    feat_id_by_col = {j: feat_ids[j] for j in range(len(feat_ids))}

    print(f"\nTop-{args.top_n} class-discriminative SAE features at "
          f"({args.anchor or '<all>'}, layer={meta.get('block_id')}):")
    print(f"{'rank':>4} {'feat_id':>8} {'F':>10} {'effect':>10}  per_class_mean  per_class_fire_rate")
    for r, t in enumerate(top, start=1):
        fid = feat_id_by_col[t["feat_col"]]
        pc_mean = ", ".join(f"{x:.3f}" for x in t["per_class_mean"])
        pc_fire = ", ".join(f"{x:.2f}" for x in t["per_class_fire_rate"])
        print(f"{r:>4} {fid:>8} {t['F']:>10.3f} {t['effect']:>10.4f}  [{pc_mean}]  [{pc_fire}]")

    print(f"\nFitting LOO logistic regression (n={X.shape[0]}, d={X.shape[1]}, C={args.lr_c}) ...")
    lr_out = _loo_lr(X, y, args.lr_c)
    print(f"LOO accuracy = {lr_out['loo_accuracy']:.4f} "
          f"(per-class: {lr_out['per_class_accuracy']})")
    elapsed = time.time() - t0

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "meta": {
            "firings": str(Path(args.firings).resolve()),
            "anchor": args.anchor,
            "lr_c": args.lr_c,
            "n_records": len(records),
            "n_features_active": int(X.shape[1]),
            "class_names": class_names,
            "n_per_class": [int((y == k).sum()) for k in range(len(class_names))],
            "elapsed_s": elapsed,
            "sae_meta": meta,
        },
        "loo_lr": lr_out,
        "top_features": [
            {
                "feat_id": feat_id_by_col[t["feat_col"]],
                "F": t["F"],
                "effect": t["effect"],
                "per_class_mean": t["per_class_mean"],
                "per_class_fire_rate": t["per_class_fire_rate"],
            }
            for t in top
        ],
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path} ({elapsed:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
