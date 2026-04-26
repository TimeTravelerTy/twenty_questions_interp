"""M4 phase 2c-iii — Positional residual capture across structural anchors.

For each kept run in a self-chosen collection, rebuild the FULL prefix
(system + Ready + 4 turns + reveal user message + add_generation_prompt)
and do one forward pass with `output_hidden_states=True`. Capture the
residual stream at ~12 structural anchor positions tied to chat-template
role boundaries:

  end_user_prompt    — end of the combined system+user opening
  end_ready          — end of the model's "Ready" turn
  end_user_qN        — end of turn N's user question      (N=1..4)
  end_model_qN       — end of turn N's model yes/no answer (N=1..4)
  end_reveal_user    — end of the reveal user message
  pre_reveal_gen     — last position (just before reveal generation)

Output: one `.pt` file per run with the full anchor-positions tensor of
shape (K_anchors, n_layers+1, hidden_dim), plus role-position metadata.

The saved residuals support a probe-fitting analysis (per anchor × layer
LR LOO accuracy) and serve as the steering-vector ingredients for the
followup phase 2d (patch only along class-discriminating direction).

Notes:
- Different runs may have different seq_len because turn-N questions
  vary in length. Anchor labels are the alignment unit, not positions.
- We capture *all* layers (49 = 1 embedding + 48 decoder blocks). One
  full prefix forward pass per run costs ~few seconds on a 12B model.
- Storage budget: 12 anchors × 49 layers × 3840 hidden × 2 bytes ~= 4.5
  MB / run × 80 runs ~= 360 MB total.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twenty_q.banks import load_bank
from twenty_q.config import MODEL_MAIN
from twenty_q.dialogue import (
    REVEAL_USER_MESSAGE,
    ModelHandle,
    _build_chat_input_ids,
    _history_to_chat_turns,
    load_model,
)
from twenty_q.manifest import RunManifest
from twenty_q.permutations import Permutation
from twenty_q.prompts import self_chosen_prompt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True, help="Diagnostic run directory.")
    p.add_argument("--out-dir", required=True, help="Where to write per-run .pt files.")
    p.add_argument("--model", default=MODEL_MAIN)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    p.add_argument("--prompt-variant", default="default")
    p.add_argument("--limit", type=int, default=None,
                   help="Process at most this many kept runs (debug only).")
    return p.parse_args()


def _load_kept_manifests(run_dir: Path) -> list[RunManifest]:
    out: list[RunManifest] = []
    for a in sorted(run_dir.glob("attempt_*")):
        mp = a / "manifest.json"
        if not mp.exists():
            continue
        m = RunManifest.load(mp)
        if m.reveal_canonical_id is not None:
            out.append(m)
    return out


def _build_full_prefix_inputs(
    handle: ModelHandle,
    manifest: RunManifest,
    bank: Any,
    prompt_variant: str,
) -> dict[str, torch.Tensor]:
    """Build the full chat context up to (and including) the reveal user
    message, with add_generation_prompt=True so the last token is the
    pre-reveal-generation position.
    """
    display_names = {c.id: c.display for c in bank.candidates}
    perm = Permutation(order=tuple(manifest.permutation))
    rendered = self_chosen_prompt(perm, display_names, variant=prompt_variant)
    extra = [
        *_history_to_chat_turns(manifest.ready_raw_output, list(manifest.turns)),
        {"role": "user", "content": REVEAL_USER_MESSAGE},
    ]
    return _build_chat_input_ids(handle, rendered, extra_turns=extra)


# Expected order of <end_of_turn> tokens in a full self-chosen prefix
# (4-question panel + reveal user message). 11 of them.
ANCHOR_LABELS_AT_EOT = [
    "end_user_prompt",   # 0: end of combined system+user opening
    "end_ready",         # 1: end of model's "Ready" turn
    "end_user_q1",       # 2: end of turn-1 user
    "end_model_q1",      # 3: end of turn-1 model answer
    "end_user_q2",
    "end_model_q2",
    "end_user_q3",
    "end_model_q3",
    "end_user_q4",
    "end_model_q4",
    "end_reveal_user",   # 10: end of reveal user message
]


def _find_anchors(
    tokenizer: Any, input_ids: torch.Tensor
) -> dict[str, int]:
    """Return a {label: position_index} dict.

    Looks for `<end_of_turn>` tokens in order; the count must match
    ANCHOR_LABELS_AT_EOT or we abort with a diagnostic. The final
    `pre_reveal_gen` anchor is the last position of input_ids
    (the model is about to start generating the reveal there).
    """
    eot_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    if eot_id is None or eot_id == tokenizer.unk_token_id:
        raise RuntimeError("Tokenizer has no <end_of_turn> special token")
    ids = input_ids[0]
    eot_positions = (ids == eot_id).nonzero(as_tuple=True)[0].tolist()

    anchors: dict[str, int] = {}
    if len(eot_positions) != len(ANCHOR_LABELS_AT_EOT):
        # Don't abort — record what we got so a debug pass can investigate.
        anchors["__DEBUG_eot_positions__"] = eot_positions  # type: ignore
        anchors["__DEBUG_expected__"] = len(ANCHOR_LABELS_AT_EOT)  # type: ignore
        return anchors
    for label, pos in zip(ANCHOR_LABELS_AT_EOT, eot_positions, strict=True):
        anchors[label] = int(pos)
    anchors["pre_reveal_gen"] = int(ids.shape[0] - 1)
    return anchors


@torch.no_grad()
def _capture_run(
    handle: ModelHandle,
    manifest: RunManifest,
    bank: Any,
    prompt_variant: str,
) -> dict[str, Any]:
    inputs = _build_full_prefix_inputs(handle, manifest, bank, prompt_variant)
    anchors = _find_anchors(handle.tokenizer, inputs["input_ids"])
    if "__DEBUG_eot_positions__" in anchors:
        return {"failed_anchors": anchors, "input_ids": inputs["input_ids"][0].cpu()}

    outputs = handle.model(
        **inputs,
        output_hidden_states=True,
        return_dict=True,
        use_cache=False,
    )
    # outputs.hidden_states is a tuple of (1, seq_len, hidden) tensors,
    # length n_layers+1 (embedding output + each block output).
    n_layers = len(outputs.hidden_states)
    hidden = outputs.hidden_states[0].shape[-1]

    # Anchor tensor shape: (K, n_layers, hidden), float32 on CPU.
    anchor_labels = list(anchors.keys())
    K = len(anchor_labels)
    out_tensor = torch.zeros((K, n_layers, hidden), dtype=torch.float32)
    for i, label in enumerate(anchor_labels):
        pos = anchors[label]
        for L in range(n_layers):
            out_tensor[i, L, :] = outputs.hidden_states[L][0, pos, :].float().cpu()
    return {
        "anchor_labels": anchor_labels,
        "anchor_positions": [anchors[L] for L in anchor_labels],
        "residuals": out_tensor,
        "seq_len": int(inputs["input_ids"].shape[1]),
        "class": manifest.reveal_canonical_id,
        "run_id": manifest.run_id,
    }


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    bank = load_bank()
    manifests = _load_kept_manifests(run_dir)
    if args.limit is not None:
        manifests = manifests[: args.limit]
    print(f"Found {len(manifests)} kept runs in {run_dir}")

    dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.dtype]
    handle = load_model(args.model, device=args.device, dtype=dtype)

    failed: list[str] = []
    t0 = time.time()
    for i, m in enumerate(manifests):
        try:
            data = _capture_run(handle, m, bank, args.prompt_variant)
            if "failed_anchors" in data:
                failed.append(m.run_id)
                print(f"  [{i+1}/{len(manifests)}] {m.run_id} ({m.reveal_canonical_id}): "
                      f"ANCHOR MISMATCH (got {len(data['failed_anchors'].get('__DEBUG_eot_positions__', []))} EOTs, "
                      f"expected {data['failed_anchors'].get('__DEBUG_expected__')})", flush=True)
                torch.save(data, out_dir / f"{m.run_id}_FAILED.pt")
                continue
            torch.save(data, out_dir / f"{m.run_id}.pt")
            if (i + 1) % 10 == 0 or i == len(manifests) - 1:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                print(f"  [{i+1}/{len(manifests)}] {m.run_id} ({m.reveal_canonical_id}) "
                      f"seq_len={data['seq_len']} "
                      f"K={len(data['anchor_labels'])} "
                      f"({rate:.2f} runs/s)", flush=True)
        except Exception as e:
            print(f"  [{i+1}/{len(manifests)}] {m.run_id}: FAILED {type(e).__name__}: {e}", file=sys.stderr)
            failed.append(m.run_id)
            continue

    elapsed = time.time() - t0
    print(f"\nDone: {len(manifests) - len(failed)}/{len(manifests)} OK, "
          f"{len(failed)} failed in {elapsed:.1f}s")
    if failed:
        print(f"Failed runs: {failed}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
