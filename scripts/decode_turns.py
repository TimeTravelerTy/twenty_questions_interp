"""Decode self-chosen turn-position activations on kept diagnostic runs.

This is the follow-through from STATUS.md / D-29: Ready-state direct-fit at
12B is weak, so the next question is whether the chosen class becomes more
decodable at pre-answer positions later in the dialogue.

For each requested turn index, this script loads the all-layer activation
tensor captured immediately before that answer, then runs layerwise leave-one-
run-out nearest-centroid and logistic-regression decoding against the reveal
label on the selected runs.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

from twenty_q.config import REPO_ROOT
from twenty_q.manifest import RunManifest
from twenty_q.readouts import loo_accuracy_logreg, loo_accuracy_nearest_centroid


def parse_turn_selector(turn_spec: str) -> list[int]:
    turns: list[int] = []
    seen: set[int] = set()
    for raw in turn_spec.split(","):
        raw = raw.strip()
        if not raw:
            continue
        turn = int(raw)
        if turn < 1:
            raise ValueError("Turn indices are 1-based and must be >= 1.")
        if turn not in seen:
            turns.append(turn)
            seen.add(turn)
    if not turns:
        raise ValueError("No valid turns selected.")
    return turns


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


def load_turn_runs(
    run_root: Path,
    *,
    turn_idx: int,
    selection: str = "all",
    model_name: str | None = None,
    prompt_template_id: str | None = None,
) -> list[tuple[RunManifest, np.ndarray]]:
    out: list[tuple[RunManifest, np.ndarray]] = []
    skipped: list[str] = []
    if not run_root.exists():
        return out

    run_paths: list[tuple[Path, Path, str]] = []
    results_path = run_root / "results.json"
    if selection == "kept" and results_path.exists():
        with results_path.open() as f:
            results = json.load(f)
        candidate_rows = results.get("kept_rows", [])
    else:
        candidate_rows = [{"run_id": sub.name} for sub in sorted(run_root.iterdir()) if sub.is_dir()]

    for row in candidate_rows:
        run_id = row["run_id"]
        manifest_path = run_root / run_id / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = RunManifest.load(manifest_path)
        turn_path_raw = manifest.turn_activation_paths.get(turn_idx)
        if turn_path_raw is None:
            skipped.append(f"{run_id} (missing turn {turn_idx})")
            continue
        turn_path = Path(turn_path_raw)
        if not turn_path.exists():
            turn_path = run_root / run_id / f"turn_{turn_idx:02d}_activations.pt"
        run_paths.append((manifest_path, turn_path, run_id))

    for manifest_path, activation_path, label in run_paths:
        manifest = RunManifest.load(manifest_path)
        if model_name is not None and manifest.model_name != model_name:
            skipped.append(f"{label} (model_name={manifest.model_name!r})")
            continue
        if prompt_template_id is not None and manifest.prompt_template_id != prompt_template_id:
            skipped.append(f"{label} (prompt_template_id={manifest.prompt_template_id!r})")
            continue
        if not activation_path.exists():
            skipped.append(f"{label} (missing {activation_path.name})")
            continue
        activations = torch.load(activation_path, map_location="cpu").numpy()
        out.append((manifest, activations))

    if skipped:
        print(
            f"[decode_turns] Skipped {len(skipped)} incompatible/missing run(s) for turn {turn_idx}:",
            file=sys.stderr,
        )
        for s in skipped[:10]:
            print(f"  - {s}", file=sys.stderr)
        if len(skipped) > 10:
            print(f"  ... and {len(skipped) - 10} more", file=sys.stderr)
    return out


def summarize_turn(layers: list[int], accs: list[float]) -> dict[str, float]:
    pairs = list(zip(layers, accs, strict=True))
    body = [acc for layer, acc in pairs if layer != 0] or accs
    best_layer, best_acc = max(pairs, key=lambda x: x[1])
    return {
        "mean_excluding_l0": float(statistics.fmean(body)),
        "median_excluding_l0": float(statistics.median(body)),
        "best_layer": int(best_layer),
        "best_accuracy": float(best_acc),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, help="Diagnostic run directory root.")
    p.add_argument("--selection", default="kept", choices=["all", "kept"])
    p.add_argument("--turns", default="1,2,3,4",
                   help="Comma-separated 1-based turn indices to score.")
    p.add_argument("--layers", default="all",
                   help="Comma-separated layer ids to score, or 'all'.")
    p.add_argument("--model-name", default=None)
    p.add_argument("--prompt-template-id", default=None)
    p.add_argument(
        "--out-report-json",
        default=str(REPO_ROOT / "runs" / "turn_report.json"),
    )
    p.add_argument(
        "--out-md",
        default=str(REPO_ROOT / "docs" / "progress" / "turn-report.md"),
    )
    p.add_argument(
        "--title",
        default="Turn-position self-chosen decoder sweep",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_root = Path(args.run_dir)
    try:
        turns = parse_turn_selector(args.turns)
    except ValueError as exc:
        print(f"[decode_turns] {exc}", file=sys.stderr)
        return 2

    model_name = args.model_name
    if model_name is None:
        probe = load_turn_runs(run_root, turn_idx=turns[0], selection=args.selection)
        if not probe:
            print(f"No runs found under {run_root} for turn {turns[0]}.", file=sys.stderr)
            return 2
        model_name = probe[0][0].model_name
        print(f"[decode_turns] Auto-filtering on model_name={model_name!r}.")

    turn_reports: list[dict] = []
    lines: list[str] = []
    lines.append("| turn | NC mean | NC median | NC best | NC best layer | LR mean | LR median | LR best | LR best layer |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")

    class_ids_global: list[str] | None = None
    chance: float | None = None

    for turn_idx in turns:
        runs = load_turn_runs(
            run_root,
            turn_idx=turn_idx,
            selection=args.selection,
            model_name=model_name,
            prompt_template_id=args.prompt_template_id,
        )
        if not runs:
            print(f"[decode_turns] No runs loaded for turn {turn_idx}.", file=sys.stderr)
            return 2

        labels: list[str] = []
        for manifest, _ in runs:
            label = manifest.reveal_canonical_id
            if label is None:
                print(
                    f"[decode_turns] run {manifest.run_id!r} missing reveal_canonical_id.",
                    file=sys.stderr,
                )
                return 2
            labels.append(label)
        class_ids = sorted(set(labels))
        class_ids_global = class_ids_global or class_ids
        chance = 1.0 / len(class_ids)

        X_all = np.stack([a for _, a in runs], axis=0)
        try:
            layers = parse_layer_selector(args.layers, X_all.shape[1])
        except ValueError as exc:
            print(f"[decode_turns] {exc}", file=sys.stderr)
            return 2

        nc_by_layer: list[float] = []
        lr_by_layer: list[float] = []
        for layer in layers:
            X = X_all[:, layer, :]
            nc = loo_accuracy_nearest_centroid(X, labels, class_ids)
            lr = loo_accuracy_logreg(X, labels, class_ids)
            nc_by_layer.append(nc)
            lr_by_layer.append(lr)
            print(
                f"  turn {turn_idx:02d} layer {layer:02d}  NC={nc:.2f}  LR={lr:.2f}"
            )

        nc_summary = summarize_turn(layers, nc_by_layer)
        lr_summary = summarize_turn(layers, lr_by_layer)

        turn_report = {
            "turn": turn_idx,
            "n_runs": len(runs),
            "class_ids": class_ids,
            "selected_layers": layers,
            "nearest_centroid_loo_by_layer": {
                str(layer): acc for layer, acc in zip(layers, nc_by_layer, strict=True)
            },
            "logreg_loo_by_layer": {
                str(layer): acc for layer, acc in zip(layers, lr_by_layer, strict=True)
            },
            "nearest_centroid_summary": nc_summary,
            "logreg_summary": lr_summary,
        }
        turn_reports.append(turn_report)
        lines.append(
            "| "
            + " | ".join(
                [
                    str(turn_idx),
                    f"{nc_summary['mean_excluding_l0']:.2f}",
                    f"{nc_summary['median_excluding_l0']:.2f}",
                    f"{nc_summary['best_accuracy']:.2f}",
                    str(nc_summary["best_layer"]),
                    f"{lr_summary['mean_excluding_l0']:.2f}",
                    f"{lr_summary['median_excluding_l0']:.2f}",
                    f"{lr_summary['best_accuracy']:.2f}",
                    str(lr_summary["best_layer"]),
                ]
            )
            + " |"
        )

    report = {
        "filter": {
            "model_name": model_name,
            "prompt_template_id": args.prompt_template_id,
            "selection": args.selection,
        },
        "run_dir": str(run_root),
        "class_ids": class_ids_global,
        "chance": chance,
        "turns": turn_reports,
    }

    report_path = Path(args.out_report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {report_path}")

    md_path = Path(args.out_md)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        f"# {args.title}\n\n"
        f"Run dir: `{run_root}`.\n"
        f"Model: {model_name}. Classes: {len(class_ids_global or [])} "
        f"(chance = {(chance or 0.0):.3f}).\n\n"
    )
    md_path.write_text(header + "\n".join(lines) + "\n")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
