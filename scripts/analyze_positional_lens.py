"""M4 D-42 — Lens-based cross-class baseline + rank-1 check at one anchor.

Reads per-run residual `.pt` files written by `capture_positional_residuals.py`
(format: `{ "anchor_labels": [...], "residuals": (K, n_layers, hidden),
"class": canonical_id, "run_id": ... }`), applies the model's final RMSNorm
+ lm_head at every layer at one chosen anchor (default `pre_reveal_gen`),
and computes for each run the per-layer logits over the bank-class
first-tokens.

Aggregates across runs:

- `mean_own_minus_non_own[L]`: own-class logit minus mean non-own-bank-class
  logit, averaged across runs. The L25-L30 numbers are the headline used in
  D-41 / D-42.
- `rank1_top_class[L]` and its share: most-frequent rank-1 class among bank-
  class lens logits at layer L (pooled across runs). Tests "is the rank-1
  class generic-prior driven, not run-specific?"
- `own_is_top_rate[L]`: how often the run's own class equals the top class.

Output is a small JSON summary; keeps the multi-GB residuals in place.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twenty_q.banks import load_bank
from twenty_q.config import MODEL_MAIN
from twenty_q.dialogue import load_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--residuals-dir", required=True,
                   help="Output dir of capture_positional_residuals.py.")
    p.add_argument("--out", required=True, help="Path for the summary JSON.")
    p.add_argument("--anchor", default="pre_reveal_gen",
                   help="Anchor label (must match one captured).")
    p.add_argument("--model", default=MODEL_MAIN)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["float32", "bfloat16", "float16"])
    return p.parse_args()


def _find_unembed_path(model):
    norm = None
    for path in ("model.norm", "model.language_model.norm",
                 "language_model.model.norm", "language_model.norm"):
        obj = model
        ok = True
        for part in path.split("."):
            if not hasattr(obj, part):
                ok = False
                break
            obj = getattr(obj, part)
        if ok:
            norm = obj
            break
    if norm is None:
        raise RuntimeError("Could not locate final norm module for logit lens")
    lm_head = model.get_output_embeddings()
    if lm_head is None:
        lm_head = model.lm_head
    return norm, lm_head


def _bank_class_first_token_ids(handle, bank) -> dict[str, int]:
    out: dict[str, int] = {}
    for c in bank.candidates:
        ids = handle.tokenizer(" " + c.display, add_special_tokens=False)["input_ids"]
        if not ids:
            ids = handle.tokenizer(c.display, add_special_tokens=False)["input_ids"]
        if not ids:
            raise RuntimeError(f"Empty tokenization for {c.display!r}")
        out[c.id] = ids[0]
    return out


@torch.no_grad()
def main() -> int:
    args = parse_args()
    res_dir = Path(args.residuals_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    handle = load_model(args.model, device=args.device, dtype=args.dtype)
    norm, lm_head = _find_unembed_path(handle.model)

    bank = load_bank()
    class_token_ids = _bank_class_first_token_ids(handle, bank)
    class_ids_list = list(class_token_ids.keys())
    n_classes = len(class_ids_list)
    token_ids_t = torch.tensor(
        [class_token_ids[c] for c in class_ids_list],
        device=handle.model.device,
    )

    pt_files = sorted(p for p in res_dir.glob("*.pt") if "_FAILED" not in p.name)
    print(f"Found {len(pt_files)} residual files in {res_dir}")
    print(f"Anchor: {args.anchor}")

    per_layer_diffs: list[list[float]] = []
    per_layer_rank1: list[list[str]] = []
    per_layer_own_is_top: list[list[bool]] = []
    # per-class per-layer split on own/non-own membership
    # logits_by_class_own[c][L] = list of (c-th class) logits when own=c
    # logits_by_class_non_own[c][L] = list of (c-th class) logits when own!=c
    logits_by_class_own: list[list[list[float]]] | None = None
    logits_by_class_non_own: list[list[list[float]]] | None = None
    n_layers_seen: int | None = None

    used = 0
    skipped = 0
    for pt_path in pt_files:
        bundle = torch.load(pt_path, map_location="cpu", weights_only=False)
        anchor_labels: list[str] = bundle["anchor_labels"]
        if args.anchor not in anchor_labels:
            print(f"  WARN anchor {args.anchor} not in {pt_path.name}; have {anchor_labels}")
            skipped += 1
            continue
        a_idx = anchor_labels.index(args.anchor)
        residuals = bundle["residuals"]  # (K, n_layers, hidden), float32 cpu
        own_id = bundle["class"]
        if own_id not in class_ids_list:
            skipped += 1
            continue
        own_idx = class_ids_list.index(own_id)

        n_layers = residuals.shape[1]
        if n_layers_seen is None:
            n_layers_seen = n_layers
            per_layer_diffs = [[] for _ in range(n_layers)]
            per_layer_rank1 = [[] for _ in range(n_layers)]
            per_layer_own_is_top = [[] for _ in range(n_layers)]
            logits_by_class_own = [[[] for _ in range(n_layers)] for _ in range(n_classes)]
            logits_by_class_non_own = [[[] for _ in range(n_layers)] for _ in range(n_classes)]

        # move just this run's anchor slice to device
        slice_dev = residuals[a_idx].to(handle.model.device, dtype=handle.model.dtype)
        for L in range(n_layers):
            r = slice_dev[L]
            normed = norm(r.unsqueeze(0)).squeeze(0)
            logits_full = lm_head(normed.unsqueeze(0)).squeeze(0).to(torch.float32)
            class_logits = logits_full[token_ids_t].cpu()
            own_lg = float(class_logits[own_idx])
            mask = torch.ones(n_classes, dtype=torch.bool)
            mask[own_idx] = False
            non_own_mean = float(class_logits[mask].mean())
            per_layer_diffs[L].append(own_lg - non_own_mean)
            top_idx = int(torch.argmax(class_logits))
            per_layer_rank1[L].append(class_ids_list[top_idx])
            per_layer_own_is_top[L].append(top_idx == own_idx)
            for ci in range(n_classes):
                lg = float(class_logits[ci])
                if ci == own_idx:
                    logits_by_class_own[ci][L].append(lg)
                else:
                    logits_by_class_non_own[ci][L].append(lg)
        used += 1

    if used == 0:
        print("No runs processed.")
        return 1

    summary: dict[str, Any] = {
        "anchor": args.anchor,
        "n_runs_used": used,
        "n_runs_skipped": skipped,
        "n_layers": n_layers_seen,
        "class_ids_order": class_ids_list,
        "per_layer": [],
    }
    for L in range(n_layers_seen):
        diffs = per_layer_diffs[L]
        c = Counter(per_layer_rank1[L])
        top_class, top_count = c.most_common(1)[0]
        own_top = per_layer_own_is_top[L]
        # D-41 style own-elevation: per-class (mean own - mean non-own); average over classes that occur as own
        per_class_elev: dict[str, float] = {}
        for ci, cid in enumerate(class_ids_list):
            own_l = logits_by_class_own[ci][L]
            non_l = logits_by_class_non_own[ci][L]
            if not own_l or not non_l:
                continue
            per_class_elev[cid] = float(sum(own_l) / len(own_l) - sum(non_l) / len(non_l))
        own_elev_mean = (
            float(sum(per_class_elev.values()) / len(per_class_elev))
            if per_class_elev
            else 0.0
        )
        summary["per_layer"].append({
            "layer": L,
            "mean_own_minus_non_own": float(sum(diffs) / len(diffs)),
            "median_own_minus_non_own": float(sorted(diffs)[len(diffs) // 2]),
            "rank1_top_class": top_class,
            "rank1_top_class_share": top_count / used,
            "own_is_top_rate": sum(own_top) / used,
            "own_elevation_per_class": per_class_elev,
            "own_elevation_mean_over_classes": own_elev_mean,
        })

    out_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out_path} ({used} runs, {skipped} skipped, {n_layers_seen} layers)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
