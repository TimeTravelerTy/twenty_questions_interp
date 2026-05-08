"""M5 Phase A1–A2 — SAE feature encoding of captured residuals.

Loads a single Gemma Scope 2 SAE and encodes residuals at one (layer,
anchor-set) slice from M4 / M5 capture `.pt` files, persisting the top-k
feature activations per (run, anchor). Also reports reconstruction
quality (MSE / FVU) so Phase A's sanity check (A1) can be done in the
same pass.

Layer indexing convention. The capture `.pt` files store residuals at
49 indices (1 embedding + 48 decoder-block outputs). Capture-index L is
`outputs.hidden_states[L]` from HuggingFace transformers — L=0 is the
embedding output (= input to block 0), and L=i for i>=1 is the output
of decoder block (i-1), i.e. resid_post[i-1]. For Gemma Scope 2's
`layer_N` SAE family (trained on the residual stream *after* decoder
block N), feed capture-index N+1. The CLI takes the capture-index
directly; `--block-id` is reported for sanity-check logging only.

Output schema (torch.save'd list of dicts):
  [
    {"run_id": str, "class": str, "anchor": str,
     "feature_idx": int32 tensor (k_active,),
     "activation":  float32 tensor (k_active,),
     "recon_mse": float, "input_norm_sq": float},
    ...
  ]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--residuals-dir", required=True,
                   help="Output dir of capture_positional_residuals.py "
                        "(per-run .pt files, anchor-labelled).")
    p.add_argument("--sae-release", required=True,
                   help="sae_lens release id, e.g. "
                        "'google/gemma-scope-2-12b-it-res'.")
    p.add_argument("--sae-id", required=True,
                   help="sae_lens SAE id within the release, e.g. "
                        "'layer_30/width_64k/average_l0_45'.")
    p.add_argument("--capture-index", type=int, required=True,
                   help="Index into the 49-layer capture tensor "
                        "(= block_id + 1 for resid_post-after-block).")
    p.add_argument("--block-id", type=int, default=None,
                   help="For logging only: the SAE's nominal block id "
                        "(should equal capture-index - 1 for resid_post).")
    p.add_argument("--anchors", nargs="+", default=None,
                   help="Subset of anchor labels to encode. "
                        "Default: all anchors stored in each capture.")
    p.add_argument("--top-k", type=int, default=64,
                   help="Persist top-k feature activations per (run, anchor).")
    p.add_argument("--out", required=True,
                   help="Output .pt file. Parent dirs created if needed.")
    p.add_argument("--device", default="auto",
                   choices=["auto", "cpu", "cuda"])
    p.add_argument("--dtype", default="bfloat16",
                   choices=["float32", "bfloat16", "float16"])
    p.add_argument("--limit", type=int, default=None,
                   help="Process at most this many capture files (debug).")
    return p.parse_args()


def _resolve_device(arg: str) -> str:
    if arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return arg


def _load_sae(release: str, sae_id: str, device: str) -> tuple[object, dict]:
    from sae_lens import SAE  # heavy import; defer until argparse done
    sae, cfg_dict, _sparsity = SAE.from_pretrained(
        release=release, sae_id=sae_id, device=device,
    )
    sae.eval()
    return sae, cfg_dict


def main() -> int:
    args = parse_args()
    device = _resolve_device(args.device)
    dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.dtype]

    if args.block_id is not None and args.block_id != args.capture_index - 1:
        print(f"WARN block-id={args.block_id} but capture-index-1="
              f"{args.capture_index - 1}; mismatch is suspicious for "
              f"resid_post SAEs.", file=sys.stderr)

    print(f"device={device} dtype={args.dtype}")
    print(f"sae_release={args.sae_release} sae_id={args.sae_id}")
    print(f"capture_index={args.capture_index} (block_id="
          f"{args.capture_index - 1} for resid_post)")
    sae, sae_cfg = _load_sae(args.sae_release, args.sae_id, device)
    d_in = getattr(sae, "d_in", None) or sae_cfg.get("d_in")
    d_sae = getattr(sae, "d_sae", None) or sae_cfg.get("d_sae")
    print(f"loaded SAE d_in={d_in} d_sae={d_sae}")

    files = sorted(Path(args.residuals_dir).glob("*.pt"))
    files = [f for f in files if not f.name.endswith("_FAILED.pt")]
    if args.limit is not None:
        files = files[: args.limit]
    print(f"Found {len(files)} capture files in {args.residuals_dir}")

    out_records: list[dict] = []
    skipped: list[str] = []
    total_anchors_processed = 0
    sum_mse = 0.0
    sum_input_sqnorm = 0.0
    t0 = time.time()

    with torch.no_grad():
        for fi, f in enumerate(files):
            data = torch.load(f, map_location="cpu", weights_only=False)
            if "residuals" not in data:
                skipped.append(f.name)
                continue
            anchors = data["anchor_labels"]
            run_id = data.get("run_id", f.stem)
            cls = data.get("class", "?")
            residuals = data["residuals"]  # (K, n_layers, hidden), float32

            keep_idx = [
                (i, lab) for i, lab in enumerate(anchors)
                if args.anchors is None or lab in args.anchors
            ]
            if not keep_idx:
                continue

            for i, label in keep_idx:
                vec = residuals[i, args.capture_index, :].to(device=device, dtype=dtype)
                if d_in is not None and vec.shape[-1] != d_in:
                    raise RuntimeError(
                        f"Residual hidden_dim={vec.shape[-1]} != SAE d_in={d_in} "
                        f"for capture-index {args.capture_index}, anchor {label}"
                    )
                feats = sae.encode(vec.unsqueeze(0))[0]  # (d_sae,)
                recon = sae.decode(feats.unsqueeze(0))[0]
                err = (recon - vec).float()
                mse = float((err * err).mean().item())
                input_sqnorm = float((vec.float() * vec.float()).sum().item())
                sum_mse += mse
                sum_input_sqnorm += input_sqnorm

                k = min(args.top_k, feats.shape[0])
                top_acts, top_idx = torch.topk(feats.float(), k)
                active = top_acts > 0
                out_records.append({
                    "run_id": run_id,
                    "class": cls,
                    "anchor": label,
                    "feature_idx": top_idx[active].to(torch.int32).cpu(),
                    "activation": top_acts[active].cpu(),
                    "recon_mse": mse,
                    "input_norm_sq": input_sqnorm,
                })
                total_anchors_processed += 1

            if (fi + 1) % 50 == 0 or fi == len(files) - 1:
                elapsed = time.time() - t0
                rate = (fi + 1) / elapsed if elapsed > 0 else 0.0
                print(f"  [{fi+1}/{len(files)}] {run_id} ({cls}) "
                      f"({rate:.2f} files/s)", flush=True)

    elapsed = time.time() - t0
    n = max(total_anchors_processed, 1)
    mean_mse = sum_mse / n
    # FVU at the per-vector level: averaged ratio MSE / (||x||^2 / d_in)
    # is a reasonable scalar; report mean variance-explained equivalents
    mean_input_var = (sum_input_sqnorm / n) / (d_in or 1)
    fvu = mean_mse / mean_input_var if mean_input_var > 0 else float("nan")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "records": out_records,
        "meta": {
            "sae_release": args.sae_release,
            "sae_id": args.sae_id,
            "capture_index": args.capture_index,
            "block_id": args.block_id,
            "residuals_dir": str(Path(args.residuals_dir).resolve()),
            "n_files": len(files),
            "n_skipped": len(skipped),
            "n_records": len(out_records),
            "top_k": args.top_k,
            "anchors_filter": args.anchors,
            "d_in": d_in,
            "d_sae": d_sae,
            "mean_recon_mse": mean_mse,
            "mean_input_var_per_dim": mean_input_var,
            "fvu_estimate": fvu,
            "elapsed_s": elapsed,
        },
    }
    torch.save(payload, out_path)
    print(f"\nDone: {len(out_records)} firings from {total_anchors_processed} "
          f"anchor evaluations, FVU≈{fvu:.4f} in {elapsed:.1f}s")
    print(f"Wrote {out_path}")
    if skipped:
        print(f"Skipped {len(skipped)} FAILED capture files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
