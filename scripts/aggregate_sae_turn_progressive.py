"""M5 Phase A3.2 — turn-progressive aggregation.

Reads the four per-anchor JSONs produced by analyze_sae_features.py at
pre_answer_q1..q4 and reports:

  (1) Per-turn LR LOO accuracy (sparse classifier).
  (2) Per-turn unique active feature count and per-class fire-rate
      summaries.
  (3) Top-F feature overlap across turns: same identities or different?
  (4) Trajectory of q4's top-N features evaluated at q1..q3:
      per-class mean activation and fire rate, to test whether the
      "horse-asymmetric" axis at q4 is built up monotonically or appears
      late.

Output: a single combined JSON + console summary.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", nargs="+", required=True,
                   help="Per-anchor analysis JSONs (q1, q2, q3, q4 order).")
    p.add_argument("--anchor-labels", nargs="+",
                   default=["pre_answer_q1", "pre_answer_q2",
                            "pre_answer_q3", "pre_answer_q4"],
                   help="Labels matching --inputs order.")
    p.add_argument("--top-n-overlap", type=int, default=10,
                   help="How many top-F features per turn to use for "
                        "overlap analysis.")
    p.add_argument("--out", required=True, help="Output JSON path.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if len(args.inputs) != len(args.anchor_labels):
        raise SystemExit("--inputs and --anchor-labels must match in length")

    payloads = {}
    for label, path in zip(args.anchor_labels, args.inputs):
        with open(path) as f:
            payloads[label] = json.load(f)

    # (1) + (2): per-turn summary
    per_turn = []
    top_ids_by_turn: dict[str, list[int]] = {}
    for label in args.anchor_labels:
        p = payloads[label]
        meta = p["meta"]
        loo = p["loo_lr"]
        top = p["top_features"]
        top_ids_by_turn[label] = [int(t["feat_id"]) for t in top]
        per_turn.append({
            "anchor": label,
            "n_per_class": meta["n_per_class"],
            "class_names": meta["class_names"],
            "n_features_active": meta["n_features_active"],
            "loo_accuracy": loo["loo_accuracy"],
            "loo_per_class": loo["per_class_accuracy"],
            "top_features_preview": [
                {
                    "feat_id": int(t["feat_id"]),
                    "F": t["F"],
                    "effect": t["effect"],
                    "per_class_mean": t["per_class_mean"],
                    "per_class_fire_rate": t["per_class_fire_rate"],
                }
                for t in top[:5]
            ],
        })

    # (3) Overlap of top-N feature ids across turns
    top_n = args.top_n_overlap
    top_sets = {label: set(top_ids_by_turn[label][:top_n])
                for label in args.anchor_labels}
    overlap_table = {}
    for a in args.anchor_labels:
        row = {}
        for b in args.anchor_labels:
            inter = top_sets[a] & top_sets[b]
            row[b] = {"intersection": len(inter), "ids": sorted(inter)}
        overlap_table[a] = row

    # All-turns intersection
    intersect_all = set.intersection(*top_sets.values()) if top_sets else set()

    # (4) q4 top-N feature trajectory across q1..q3
    last_label = args.anchor_labels[-1]
    q4_top = payloads[last_label]["top_features"][:top_n]
    q4_top_ids = [int(t["feat_id"]) for t in q4_top]

    # For each q4-top feature, look up per_class_mean / fire_rate at every
    # turn (if present in that turn's top_features). If not present, it
    # might still be in the active set but not top-30; we report missing.
    # Note: top_features in analyze_sae_features.py is capped at --top-n
    # (default 30), so this only catches features that made the top-N
    # at every turn. Adequate for a first pass.
    feat_trajectory = {}
    for fid in q4_top_ids:
        traj = {}
        for label in args.anchor_labels:
            entry = next(
                (t for t in payloads[label]["top_features"]
                 if int(t["feat_id"]) == fid),
                None,
            )
            if entry is None:
                traj[label] = None
            else:
                traj[label] = {
                    "F": entry["F"],
                    "effect": entry["effect"],
                    "per_class_mean": entry["per_class_mean"],
                    "per_class_fire_rate": entry["per_class_fire_rate"],
                }
        feat_trajectory[str(fid)] = traj

    out = {
        "anchors": args.anchor_labels,
        "per_turn_summary": per_turn,
        "top_overlap": {
            "top_n": top_n,
            "pairwise": overlap_table,
            "all_turns_intersection": sorted(intersect_all),
        },
        "q4_top_feature_trajectory": feat_trajectory,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    # Console summary
    print(f"=== Turn-progressive M5 SAE summary (top_n={top_n}) ===")
    print(f"{'anchor':>16} {'n_active':>10} {'LR LOO':>9}  per-class LOO")
    for row in per_turn:
        pc = "[" + ",".join(f"{x:.2f}" for x in row["loo_per_class"]) + "]"
        print(f"{row['anchor']:>16} {row['n_features_active']:>10} "
              f"{row['loo_accuracy']:>9.3f}  {pc}")
    print()
    print(f"Pairwise top-{top_n} feature overlaps:")
    print(f"{'':>16} " + " ".join(f"{b:>16}" for b in args.anchor_labels))
    for a in args.anchor_labels:
        cells = " ".join(
            f"{overlap_table[a][b]['intersection']:>16d}"
            for b in args.anchor_labels
        )
        print(f"{a:>16} {cells}")
    print(f"\nAll-4-turns intersection ({len(intersect_all)} features): "
          f"{sorted(intersect_all)}")
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
