"""Ready-state decoder smoke test (M2.2c).

Scans every layer's residual-stream activation captured at the Ready position:
  - Nearest-centroid LOO accuracy on calibration (20-way candidate identity).
  - Multinomial logistic regression LOO on calibration.
  - Binary attribute decoders LOO on calibration (one per question).

On self-chosen runs, reports decoder-vs-reveal agreement (where reveal parsed).

Writes a layer x metric table to `docs/progress/M2-ready-smoke-test.md` and a
per-run JSON summary under `runs/m2_report.json`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from twenty_q.banks import load_bank
from twenty_q.config import CALIBRATION_RUNS_DIR, REPO_ROOT, SELFCHOSEN_RUNS_DIR
from twenty_q.manifest import RunManifest
from twenty_q.readouts import (
    attribute_labels,
    fit_nearest_centroid,
    loo_accuracy_binary,
    loo_accuracy_logreg,
    loo_accuracy_nearest_centroid,
)


def load_runs(run_root: Path) -> list[tuple[RunManifest, np.ndarray]]:
    """Returns (manifest, activations[n_layers+1, hidden_size]) for each run in `run_root`."""
    out: list[tuple[RunManifest, np.ndarray]] = []
    if not run_root.exists():
        return out
    for sub in sorted(run_root.iterdir()):
        m_path = sub / "manifest.json"
        a_path = sub / "activations.pt"
        if not (m_path.exists() and a_path.exists()):
            continue
        manifest = RunManifest.load(m_path)
        activations = torch.load(a_path, map_location="cpu").numpy()
        out.append((manifest, activations))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--attr-subset", default="is_mammal,is_bird,is_carnivore,lives_in_africa",
                   help="Comma-separated question ids for binary-attribute sweep.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    bank = load_bank()

    cal = load_runs(CALIBRATION_RUNS_DIR)
    sc = load_runs(SELFCHOSEN_RUNS_DIR)
    if not cal:
        print("No calibration runs found; run scripts/run_calibration.py first.", file=sys.stderr)
        return 2

    cal_secrets = [m.secret_canonical_id for m, _ in cal]
    cal_X_all = np.stack([a for _, a in cal], axis=0)  # (N, n_layers+1, hidden_size)
    print(f"Calibration runs: {len(cal)}   activations shape: {cal_X_all.shape}")

    if sc:
        sc_X_all = np.stack([a for _, a in sc], axis=0)
        print(f"Self-chosen runs: {len(sc)}   activations shape: {sc_X_all.shape}")
    else:
        sc_X_all = None
        print("Self-chosen runs: 0 (skipping self-chosen transfer).")

    n_layers_plus_1 = cal_X_all.shape[1]
    class_ids = list(bank.candidate_ids)
    attr_ids = [x.strip() for x in args.attr_subset.split(",") if x.strip()]

    report = {
        "n_calibration_runs": len(cal),
        "n_selfchosen_runs": len(sc),
        "class_ids": class_ids,
        "layers": [],
    }
    lines: list[str] = []
    lines.append("| layer | NC LOO | LR LOO | " + " | ".join(
        f"{a[:18]}" for a in attr_ids
    ) + " | SC-agree |")
    lines.append("|---|---|---|" + "|".join("---" for _ in attr_ids) + "|---|")

    for layer in range(n_layers_plus_1):
        X_cal = cal_X_all[:, layer, :]
        nc_acc = loo_accuracy_nearest_centroid(X_cal, cal_secrets, class_ids)
        lr_acc = loo_accuracy_logreg(X_cal, cal_secrets, class_ids)
        attr_accs: dict[str, tuple[float, float]] = {}
        for qid in attr_ids:
            y = attribute_labels(cal_secrets, bank, qid)
            attr_accs[qid] = loo_accuracy_binary(X_cal, y)

        # Self-chosen transfer: fit centroid on ALL calibration, predict SC, compare to reveal.
        sc_agreement: float | None = None
        if sc_X_all is not None:
            dec = fit_nearest_centroid(X_cal, cal_secrets, class_ids)
            sc_preds = dec.predict(sc_X_all[:, layer, :])
            matched = [(m.reveal_canonical_id, p) for (m, _), p in zip(sc, sc_preds)]
            with_reveal = [(r, p) for r, p in matched if r is not None]
            if with_reveal:
                sc_agreement = sum(1 for r, p in with_reveal if r == p) / len(with_reveal)

        row = {
            "layer": layer,
            "nearest_centroid_loo": nc_acc,
            "logreg_loo": lr_acc,
            "attribute_loo": {k: v[0] for k, v in attr_accs.items()},
            "attribute_majority_baseline": {k: v[1] for k, v in attr_accs.items()},
            "selfchosen_decoder_vs_reveal_agreement": sc_agreement,
        }
        report["layers"].append(row)

        attr_cells = " | ".join(f"{attr_accs[q][0]:.2f}" for q in attr_ids)
        sc_cell = f"{sc_agreement:.2f}" if sc_agreement is not None else "n/a"
        lines.append(f"| {layer} | {nc_acc:.2f} | {lr_acc:.2f} | {attr_cells} | {sc_cell} |")
        print(f"  layer {layer:2d}  NC={nc_acc:.2f}  LR={lr_acc:.2f}  attrs={ {k: f'{v[0]:.2f}' for k, v in attr_accs.items()} }  SC={sc_cell}")

    # Persist machine-readable report + markdown table.
    report_path = REPO_ROOT / "runs" / "m2_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {report_path}")

    md_path = REPO_ROOT / "docs" / "progress" / "M2-ready-smoke-test.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# M2 - Ready-state decoder smoke test\n\n"
        f"Runs: {len(cal)} calibration, {len(sc)} self-chosen.\n"
        f"Chance baselines: 20-way = 0.05; attribute majority varies per question.\n\n"
    )
    md_path.write_text(header + "\n".join(lines) + "\n")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
