"""H-persistence diagnostic for `verbalized_index` at 4B.

Per STATUS.md / D-19, the hypothesis is that at 4B the instantiated-entity
representation does not persist across the chat-turn boundary, even when name
retrieval from the index succeeds. This script measures that directly.

For each run of the `verbalized_index` condition we capture two all-layer
hidden states at parallel "model about to speak" positions:

- **State A** — end of turn-1 user prompt, before the model verbalizes the
  name. At this position the model should have just completed index->name
  retrieval, so the entity representation should be maximally active.
- **State B** — end of turn-2 user prompt (after the model has verbalized the
  name and the adversary has asked for Ready), before the model emits Ready.
  A full assistant+user round has elapsed since State A.

Both captures use `add_generation_prompt=True`, i.e. the same kind of position
as M2/M3 Ready capture, so they are directly comparable. The same script also
runs through the primary question set after Ready so we can correlate
representational persistence with the usual answer-drift pattern.

After all runs, per layer we compute:

- NC LOO accuracy at State A (chance 25% on 4 candidates)
- NC LOO accuracy at State B
- Cross-state transfer accuracy: fit centroids on State A, predict State B
- Within-vs-between cosine contrast at each state

If A decodes the entity but B does not, and the cross-state transfer is poor,
H-persistence is directly measured.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from twenty_q.banks import Bank, load_bank
from twenty_q.config import MODEL_MAIN
from twenty_q.dialogue import ModelHandle, load_model, parse_ready, parse_yes_no
from twenty_q.permutations import Permutation, shuffle_candidates
from twenty_q.prompts import PROMPT_TEMPLATE_ID, question_turn_prompt
from twenty_q.readouts import (
    layerwise_cross_nearest_centroid,
    layerwise_loo_accuracy_nearest_centroid,
    within_between_contrast,
)

DEFAULT_CANDIDATES = ("tiger", "eagle", "frog", "salmon")
DEFAULT_PRIMARY_QUESTION_IDS = (
    "is_mammal",
    "is_bird",
    "lives_primarily_in_water",
    "has_four_legs",
)
DEFAULT_SEEDS = (0, 1)


SYSTEM_PROMPT = (
    "You are playing a 20-Questions-style game. An adversary will ask you "
    "yes/no questions about a single secret animal that you are holding in "
    "mind. You must answer each question truthfully and concisely with "
    "either 'Yes' or 'No' about the secret animal. Do not reveal the "
    "identity of the secret animal under any circumstances."
)

VERBALIZED_LOCKIN_USER = (
    "Good. Keep that same animal as your secret for the entire game. Do not "
    "state the name of the animal again. When your choice is locked in, reply "
    "with only the single word: Ready"
)


@dataclass(frozen=True)
class CaptureResult:
    states: torch.Tensor  # (n_layers, hidden)
    generated_raw: str


def _seed_for(cid: str, k: int) -> int:
    digest = hashlib.sha256(f"verbalized_index:{cid}:{k}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def _candidate_list_lines(perm: Permutation, display_names: dict[str, str]) -> str:
    return "\n".join(
        f"{i}. {display_names[cid]}" for i, cid in enumerate(perm.order, start=1)
    )


def _turn1_user(
    perm: Permutation,
    display_names: dict[str, str],
    secret_displayed_index: int,
) -> str:
    return (
        f"{SYSTEM_PROMPT}\n\n"
        "Here is the list of candidate animals:\n\n"
        f"{_candidate_list_lines(perm, display_names)}\n\n"
        f"Your secret is the animal shown at position #{secret_displayed_index} "
        "in the list above. First, identify which animal that is by replying "
        "with only the animal's name on a single line — nothing else."
    )


def _capture_last_state(
    handle: ModelHandle, messages: list[dict[str, str]]
) -> torch.Tensor:
    """Run the chat template through the model and return per-layer hidden state at
    the last token position. Uses `add_generation_prompt=True` so the position is
    the 'model about to speak' boundary.
    """
    out = handle.tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    if hasattr(out, "to"):
        out = out.to(handle.model.device)
    else:
        out = {k: v.to(handle.model.device) for k, v in out.items()}

    with torch.no_grad():
        outputs = handle.model(
            **out,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        states = torch.stack(
            [h[:, -1, :].squeeze(0).float().cpu() for h in outputs.hidden_states],
            dim=0,
        )
    return states


def _capture_and_generate(
    handle: ModelHandle,
    messages: list[dict[str, str]],
    max_new_tokens: int,
) -> CaptureResult:
    """Capture the pre-generation hidden state and greedy-decode the reply."""
    out = handle.tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    if hasattr(out, "to"):
        out = out.to(handle.model.device)
    else:
        out = {k: v.to(handle.model.device) for k, v in out.items()}
    input_ids = out["input_ids"]

    with torch.no_grad():
        outputs = handle.model(
            **out,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        states = torch.stack(
            [h[:, -1, :].squeeze(0).float().cpu() for h in outputs.hidden_states],
            dim=0,
        )
        gen = handle.model.generate(
            **out,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=handle.tokenizer.eos_token_id,
        )
    new_tokens = gen[0, input_ids.shape[1]:]
    raw = handle.tokenizer.decode(new_tokens, skip_special_tokens=True)
    return CaptureResult(states=states, generated_raw=raw)


def _run_one(
    handle: ModelHandle,
    bank: Bank,
    cid: str,
    k: int,
    question_ids: tuple[str, ...],
    run_dir: Path,
) -> dict[str, Any]:
    display_names = {c.id: c.display for c in bank.candidates}
    seed = _seed_for(cid, k)
    perm = shuffle_candidates(bank.candidate_ids, seed=seed)
    pos = perm.displayed_index(cid)
    display_name = display_names[cid]
    run_dir.mkdir(parents=True, exist_ok=True)

    # Turn 1: index prompt. Capture State A (pre-verbalization) AND generate name.
    turn1_user = _turn1_user(perm, display_names, pos)
    messages: list[dict[str, str]] = [{"role": "user", "content": turn1_user}]
    turn1 = _capture_and_generate(handle, messages, max_new_tokens=16)
    state_a_path = run_dir / "state_a_turn1_pre_verbalization.pt"
    torch.save(turn1.states, state_a_path)
    raw_name = turn1.generated_raw
    cleaned_name = raw_name.strip().splitlines()[0].strip().strip(".") if raw_name else ""
    name_matches = cleaned_name.lower() == display_name.lower()
    messages.append({"role": "assistant", "content": raw_name})

    # Turn 2: lock-in user. Capture State B (pre-Ready) AND generate Ready.
    messages.append({"role": "user", "content": VERBALIZED_LOCKIN_USER})
    turn2 = _capture_and_generate(handle, messages, max_new_tokens=8)
    state_b_path = run_dir / "state_b_turn2_pre_ready.pt"
    torch.save(turn2.states, state_b_path)
    ready_raw = turn2.generated_raw
    ready_ok = parse_ready(ready_raw)
    messages.append({"role": "assistant", "content": ready_raw.strip()})

    # Primary questions (reported for cross-check with correctness drift).
    q_lookup = {q.id: q for q in bank.questions}
    answers: list[dict[str, Any]] = []
    for turn_idx, qid in enumerate(question_ids, start=3):
        q = q_lookup[qid]
        messages.append({"role": "user", "content": question_turn_prompt(q.text)})
        cap = _capture_and_generate(handle, messages, max_new_tokens=8)
        torch.save(cap.states, run_dir / f"q{turn_idx - 2:02d}_pre_answer.pt")
        messages.append({"role": "assistant", "content": cap.generated_raw.strip()})
        parsed = parse_yes_no(cap.generated_raw)
        bank_answer = bool(bank.answer(cid, qid))
        answers.append(
            {
                "qid": qid,
                "question_text": q.text,
                "raw": cap.generated_raw,
                "parsed": parsed,
                "bank": bank_answer,
                "correct": (parsed == bank_answer) if parsed is not None else None,
            }
        )

    record = {
        "run_id": f"{cid}_{k:02d}",
        "cid": cid,
        "seed_tag": f"verbalized_index:{cid}:{k}",
        "seed": seed,
        "position": pos,
        "permutation": list(perm.order),
        "prompt_template_id": PROMPT_TEMPLATE_ID,
        "verbalized_name_raw": raw_name,
        "verbalized_name_cleaned": cleaned_name,
        "verbalized_name_matches": name_matches,
        "ready_raw": ready_raw,
        "ready_ok": ready_ok,
        "state_a_path": str(state_a_path),
        "state_b_path": str(state_b_path),
        "answers": answers,
    }
    with (run_dir / "result.json").open("w") as f:
        json.dump(record, f, indent=2, default=str)
    return record


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL_MAIN)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    p.add_argument("--out-dir", default="runs/diag/persistence_smoke")
    p.add_argument("--candidates", default=",".join(DEFAULT_CANDIDATES))
    p.add_argument(
        "--question-ids",
        default=",".join(DEFAULT_PRIMARY_QUESTION_IDS),
        help="Primary yes/no questions to ask after Ready for correctness cross-check.",
    )
    p.add_argument("--seeds", default=",".join(str(x) for x in DEFAULT_SEEDS))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    bank = load_bank()
    candidates = tuple(x.strip() for x in args.candidates.split(",") if x.strip())
    question_ids = tuple(x.strip() for x in args.question_ids.split(",") if x.strip())
    seeds = tuple(int(x.strip()) for x in args.seeds.split(",") if x.strip())

    unknown = sorted(set(candidates) - set(bank.candidate_ids))
    if unknown:
        print(f"Unknown candidate ids: {unknown}", file=sys.stderr)
        return 2
    q_lookup = {q.id: q for q in bank.questions}
    unknown_q = sorted(set(question_ids) - set(q_lookup))
    if unknown_q:
        print(f"Unknown question ids: {unknown_q}", file=sys.stderr)
        return 2

    dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.dtype]

    print(f"Loading {args.model} on {args.device} ({args.dtype}) ...")
    t0 = time.time()
    handle = load_model(args.model, device=args.device, dtype=dtype)
    print(f"  loaded in {time.time() - t0:.1f}s")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    states_a: dict[str, list[torch.Tensor]] = {}
    states_b: dict[str, list[torch.Tensor]] = {}

    t0 = time.time()
    for cid in candidates:
        for k in seeds:
            run_id = f"{cid}_{k:02d}"
            print(f"  run {run_id} ...")
            run_dir = out_dir / run_id
            rec = _run_one(
                handle=handle,
                bank=bank,
                cid=cid,
                k=k,
                question_ids=question_ids,
                run_dir=run_dir,
            )
            rows.append(rec)
            states_a.setdefault(cid, []).append(torch.load(rec["state_a_path"]))
            states_b.setdefault(cid, []).append(torch.load(rec["state_b_path"]))
    print(f"  {len(rows)} runs in {time.time() - t0:.1f}s")

    # Analysis.
    class_ids = sorted({r["cid"] for r in rows})
    ordered_a = [t for cid in class_ids for t in states_a.get(cid, [])]
    ordered_b = [t for cid in class_ids for t in states_b.get(cid, [])]
    labels = [cid for cid in class_ids for _ in states_a.get(cid, [])]

    nc_a = layerwise_loo_accuracy_nearest_centroid(ordered_a, labels, class_ids)
    nc_b = layerwise_loo_accuracy_nearest_centroid(ordered_b, labels, class_ids)
    cross_a_to_b = layerwise_cross_nearest_centroid(
        ordered_a, labels, ordered_b, labels, class_ids
    )

    contrast_a = within_between_contrast(states_a)
    contrast_b = within_between_contrast(states_b)

    # Correctness cross-check (per candidate + overall).
    n_total = 0
    n_correct = 0
    n_name_match = 0
    per_cid_correctness: dict[str, dict[str, Any]] = {}
    for r in rows:
        per = per_cid_correctness.setdefault(
            r["cid"],
            {"n_runs": 0, "n_name_match": 0, "n_ready_ok": 0, "n_q": 0, "n_correct": 0},
        )
        per["n_runs"] += 1
        per["n_name_match"] += int(r["verbalized_name_matches"])
        per["n_ready_ok"] += int(r["ready_ok"])
        n_name_match += int(r["verbalized_name_matches"])
        for a in r["answers"]:
            per["n_q"] += 1
            n_total += 1
            if a["correct"]:
                per["n_correct"] += 1
                n_correct += 1
    for per in per_cid_correctness.values():
        per["pct_correct"] = per["n_correct"] / per["n_q"] if per["n_q"] else 0.0

    # Picking a "best" post-13 layer by NC-A accuracy for a compact headline.
    n_layers = len(nc_a)
    post13_best_layer = 13 + int(np.argmax(nc_a[13:])) if n_layers > 13 else int(np.argmax(nc_a))

    results: dict[str, Any] = {
        "model": args.model,
        "model_revision": handle.model_revision,
        "tokenizer_revision": handle.tokenizer_revision,
        "torch_dtype": args.dtype,
        "prompt_template_id": PROMPT_TEMPLATE_ID,
        "candidates": list(candidates),
        "question_ids": list(question_ids),
        "seeds": list(seeds),
        "condition": "verbalized_index",
        "n_runs": len(rows),
        "class_ids": class_ids,
        "rows": rows,
        "persistence": {
            "n_layers": n_layers,
            "nc_loo_state_a_by_layer": nc_a,
            "nc_loo_state_b_by_layer": nc_b,
            "nc_cross_a_to_b_by_layer": cross_a_to_b,
            "state_a_within_between": contrast_a,
            "state_b_within_between": contrast_b,
            "post13_best_layer_by_a": post13_best_layer,
            "post13_best_acc": {
                "state_a": nc_a[post13_best_layer],
                "state_b": nc_b[post13_best_layer],
                "cross_a_to_b": cross_a_to_b[post13_best_layer],
            },
        },
        "correctness": {
            "n_runs": len(rows),
            "n_name_match": n_name_match,
            "pct_name_match": n_name_match / len(rows) if rows else 0.0,
            "n_q_total": n_total,
            "n_q_correct": n_correct,
            "pct_correct": n_correct / n_total if n_total else 0.0,
            "per_candidate": per_cid_correctness,
        },
    }

    out_path = out_dir / "results.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n== H-persistence summary ==")
    print(f"  verbalization match: {n_name_match}/{len(rows)}")
    print(
        f"  primary correctness: {n_correct}/{n_total} "
        f"({(n_correct / n_total * 100.0) if n_total else 0.0:.1f}%)"
    )
    print(
        f"  layer {post13_best_layer} (best post-13 by NC-A): "
        f"NC-A={nc_a[post13_best_layer]:.2%}, "
        f"NC-B={nc_b[post13_best_layer]:.2%}, "
        f"cross A->B={cross_a_to_b[post13_best_layer]:.2%}"
    )
    print(
        f"  post-13 within-between contrast: "
        f"A={contrast_a.get('contrast_post13', float('nan')):+.2e}, "
        f"B={contrast_b.get('contrast_post13', float('nan')):+.2e}"
    )
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
