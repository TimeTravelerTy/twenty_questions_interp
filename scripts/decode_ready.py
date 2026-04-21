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
    fit_logreg,
    fit_nearest_centroid,
    loo_accuracy_binary,
    loo_accuracy_logreg,
    loo_accuracy_nearest_centroid,
)


def load_runs(
    run_root: Path,
    model_name: str | None = None,
    prompt_template_id: str | None = None,
    selection: str = "all",
) -> list[tuple[RunManifest, np.ndarray]]:
    """Returns (manifest, activations[n_layers+1, hidden_size]) for each run in `run_root`.

    If `model_name` or `prompt_template_id` is given, runs whose manifest does
    not match are skipped (with a stderr warning). This prevents silently
    mixing incompatible datasets once M3 adds 4B runs or new prompts.
    """
    out: list[tuple[RunManifest, np.ndarray]] = []
    skipped: list[str] = []
    if not run_root.exists():
        return out
    run_paths: list[tuple[Path, Path, str]] = []
    results_path = run_root / "results.json"
    if selection == "kept" and results_path.exists():
        with results_path.open() as f:
            results = json.load(f)
        for row in results.get("kept_rows", []):
            manifest_path = Path(row["manifest_path"])
            activation_path = Path(row["ready_state_path"])
            if not manifest_path.exists():
                manifest_path = run_root / row["run_id"] / "manifest.json"
            if not activation_path.exists():
                activation_path = run_root / row["run_id"] / "activations.pt"
            run_paths.append((manifest_path, activation_path, manifest_path.parent.name))
    else:
        for sub in sorted(run_root.iterdir()):
            m_path = sub / "manifest.json"
            a_path = sub / "activations.pt"
            if not (m_path.exists() and a_path.exists()):
                continue
            run_paths.append((m_path, a_path, sub.name))

    for m_path, a_path, label in run_paths:
        manifest = RunManifest.load(m_path)
        if model_name is not None and manifest.model_name != model_name:
            skipped.append(f"{label} (model_name={manifest.model_name!r})")
            continue
        if prompt_template_id is not None and manifest.prompt_template_id != prompt_template_id:
            skipped.append(f"{label} (prompt_template_id={manifest.prompt_template_id!r})")
            continue
        activations = torch.load(a_path, map_location="cpu").numpy()
        out.append((manifest, activations))
    if skipped:
        print(f"[load_runs] Skipped {len(skipped)} incompatible run(s) in {run_root}:",
              file=sys.stderr)
        for s in skipped[:10]:
            print(f"  - {s}", file=sys.stderr)
        if len(skipped) > 10:
            print(f"  ... and {len(skipped) - 10} more", file=sys.stderr)
    return out


def parse_layer_selector(layer_spec: str, n_layers_plus_1: int) -> list[int]:
    if layer_spec.strip().lower() == "all":
        return list(range(n_layers_plus_1))

    layers: list[int] = []
    seen: set[int] = set()
    for raw in layer_spec.split(","):
        raw = raw.strip()
        if not raw:
            continue
        layer = int(raw)
        if layer < 0 or layer >= n_layers_plus_1:
            raise ValueError(
                f"Layer {layer} out of range for activations with {n_layers_plus_1} layers."
            )
        if layer not in seen:
            layers.append(layer)
            seen.add(layer)
    if not layers:
        raise ValueError("No valid layers selected.")
    return layers


def agreement_with_reveal(
    runs: list[tuple[RunManifest, np.ndarray]], preds: list[str]
) -> float | None:
    matched = [(m.reveal_canonical_id, p) for (m, _), p in zip(runs, preds, strict=True)]
    with_reveal = [(r, p) for r, p in matched if r is not None]
    if not with_reveal:
        return None
    return sum(1 for r, p in with_reveal if r == p) / len(with_reveal)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--attr-subset", default="is_mammal,is_bird,is_carnivore,lives_in_africa",
                   help="Comma-separated question ids for binary-attribute sweep.")
    p.add_argument("--layers", default="all",
                   help="Comma-separated layer ids to score, or 'all'.")
    p.add_argument("--model-name", default=None,
                   help="Restrict to runs with this model_name. Defaults to the "
                        "model_name of the first calibration run found.")
    p.add_argument("--prompt-template-id", default=None,
                   help="Restrict to runs with this prompt_template_id. Note: "
                        "calibration and self-chosen use different templates; "
                        "this filter is applied per-condition independently.")
    p.add_argument("--cal-dir", default=str(CALIBRATION_RUNS_DIR),
                   help="Directory containing calibration run subdirectories.")
    p.add_argument("--sc-dir", default=str(SELFCHOSEN_RUNS_DIR),
                   help="Directory containing self-chosen run subdirectories.")
    p.add_argument("--sc-selection", default="all", choices=["all", "kept"],
                   help="Which self-chosen runs to score when sc-dir is a diagnostic output.")
    p.add_argument("--out-report-json",
                   default=str(REPO_ROOT / "runs" / "m2_report.json"),
                   help="Path for machine-readable JSON report.")
    p.add_argument("--out-md",
                   default=str(REPO_ROOT / "docs" / "progress" / "M2-ready-smoke-test.md"),
                   help="Path for markdown layer-sweep table.")
    p.add_argument("--title", default="M2 - Ready-state decoder smoke test",
                   help="Heading used in the generated markdown report.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    bank = load_bank()
    cal_dir = Path(args.cal_dir)
    sc_dir = Path(args.sc_dir)

    # First pass without filter to auto-detect model_name if the user didn't set one.
    model_name = args.model_name
    if model_name is None:
        probe = load_runs(cal_dir)
        if not probe:
            print(f"No calibration runs found under {cal_dir}; "
                  "run scripts/run_calibration.py first.", file=sys.stderr)
            return 2
        model_name = probe[0][0].model_name
        print(f"[decode_ready] Auto-filtering on model_name={model_name!r} "
              f"(override with --model-name).")

    cal = load_runs(cal_dir, model_name=model_name,
                    prompt_template_id=args.prompt_template_id)
    sc = load_runs(sc_dir, model_name=model_name,
                   prompt_template_id=args.prompt_template_id,
                   selection=args.sc_selection)
    if not cal:
        print("No calibration runs matched the model_name/prompt filter.", file=sys.stderr)
        return 2
    # Sanity: all retained runs must agree on hidden_size.
    hidden_sizes = {m.hidden_size for m, _ in cal + sc if m.hidden_size is not None}
    if len(hidden_sizes) > 1:
        print(f"[decode_ready] Mixed hidden_sizes {hidden_sizes}; aborting.", file=sys.stderr)
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
    # Restrict the decoder's class set to classes actually present in the calibration
    # data. Passing the full bank poisons NC centroids with NaNs for any class that
    # has no runs (the original M2 smoke had all 20, so this only surfaced once a
    # pilot used a subset).
    class_ids = sorted(set(cal_secrets))
    attr_ids = []
    if args.attr_subset.strip().lower() not in {"", "none"}:
        attr_ids = [x.strip() for x in args.attr_subset.split(",") if x.strip()]
    try:
        layers = parse_layer_selector(args.layers, n_layers_plus_1)
    except ValueError as exc:
        print(f"[decode_ready] {exc}", file=sys.stderr)
        return 2

    report = {
        "filter": {
            "model_name": model_name,
            "prompt_template_id": args.prompt_template_id,
        },
        "n_calibration_runs": len(cal),
        "n_selfchosen_runs": len(sc),
        "class_ids": class_ids,
        "attribute_ids": attr_ids,
        "selected_layers": layers,
        "layers": [],
    }
    lines: list[str] = []
    table_header = ["layer", "NC LOO", "LR LOO"] + [
        a[:18] for a in attr_ids
    ] + ["SC-NC", "SC-LR"]
    lines.append("| " + " | ".join(table_header) + " |")
    lines.append("|" + "|".join("---" for _ in table_header) + "|")

    for layer in layers:
        X_cal = cal_X_all[:, layer, :]
        nc_acc = loo_accuracy_nearest_centroid(X_cal, cal_secrets, class_ids)
        lr_acc = loo_accuracy_logreg(X_cal, cal_secrets, class_ids)
        attr_accs: dict[str, tuple[float, float]] = {}
        for qid in attr_ids:
            y = attribute_labels(cal_secrets, bank, qid)
            attr_accs[qid] = loo_accuracy_binary(X_cal, y)

        # Self-chosen transfer: fit decoders on ALL calibration, predict SC, compare to reveal.
        sc_agreement_nc: float | None = None
        sc_agreement_lr: float | None = None
        if sc_X_all is not None:
            dec_nc = fit_nearest_centroid(X_cal, cal_secrets, class_ids)
            sc_agreement_nc = agreement_with_reveal(
                sc, dec_nc.predict(sc_X_all[:, layer, :])
            )
            dec_lr = fit_logreg(X_cal, cal_secrets)
            sc_agreement_lr = agreement_with_reveal(
                sc, dec_lr.predict(sc_X_all[:, layer, :])
            )

        row = {
            "layer": layer,
            "nearest_centroid_loo": nc_acc,
            "logreg_loo": lr_acc,
            "attribute_loo": {k: v[0] for k, v in attr_accs.items()},
            "attribute_majority_baseline": {k: v[1] for k, v in attr_accs.items()},
            "selfchosen_decoder_vs_reveal_agreement": sc_agreement_nc,
            "selfchosen_nearest_centroid_vs_reveal_agreement": sc_agreement_nc,
            "selfchosen_logreg_vs_reveal_agreement": sc_agreement_lr,
        }
        report["layers"].append(row)

        row_cells = [f"{layer}", f"{nc_acc:.2f}", f"{lr_acc:.2f}"]
        row_cells.extend(f"{attr_accs[q][0]:.2f}" for q in attr_ids)
        row_cells.append(
            f"{sc_agreement_nc:.2f}" if sc_agreement_nc is not None else "n/a"
        )
        row_cells.append(
            f"{sc_agreement_lr:.2f}" if sc_agreement_lr is not None else "n/a"
        )
        lines.append("| " + " | ".join(row_cells) + " |")
        attr_print = {k: f"{v[0]:.2f}" for k, v in attr_accs.items()}
        attrs_blurb = f"  attrs={attr_print}" if attr_ids else ""
        sc_nc_cell = f"{sc_agreement_nc:.2f}" if sc_agreement_nc is not None else "n/a"
        sc_lr_cell = f"{sc_agreement_lr:.2f}" if sc_agreement_lr is not None else "n/a"
        print(
            f"  layer {layer:2d}  NC={nc_acc:.2f}  LR={lr_acc:.2f}  "
            f"SC-NC={sc_nc_cell}  SC-LR={sc_lr_cell}{attrs_blurb}"
        )

    # Persist machine-readable report + markdown table.
    report_path = Path(args.out_report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {report_path}")

    md_path = Path(args.out_md)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    n_classes = len(set(cal_secrets))
    chance = 1.0 / n_classes if n_classes else 0.0
    header = (
        f"# {args.title}\n\n"
        f"Runs: {len(cal)} calibration, {len(sc)} self-chosen.\n"
        f"Model: {model_name}. Classes: {n_classes} "
        f"(NC/LR chance = {chance:.3f}).\n\n"
    )
    md_path.write_text(header + "\n".join(lines) + "\n")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
