"""Diagnose whether index-based calibration binds a specific entity at 4B.

Runs three conditions end-to-end on the same seeds:

  A  bare retrieval (no secrecy): "Here is the list... what animal is at
     position #N?" Does the model retrieve the right name? Cheapest possible
     check that position indexing works at all.

  B  verbalized index binding: same index framing, but the model is asked to
     *state the animal at position #N* in an assistant turn before being told
     to commit and say Ready. The name enters the attention trail but the
     index framing is preserved. If this passes answer-correctness it keeps
     D-06's intent alive.

  C  current D-06 prompt: baseline.

Scoring gate: answer correctness vs. the bank. No activation capture — this
script is choosing *which* calibration condition to train probes with, not
training probes. Writes a single JSON summary under the provided --out-dir.

Usage:
    uv run python scripts/diagnose_index_binding.py \\
        --model google/gemma-3-4b-it --out-dir runs/diag/binding_smoke
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

from twenty_q.banks import Bank, load_bank
from twenty_q.config import MODEL_MAIN
from twenty_q.dialogue import ModelHandle, load_model, parse_ready, parse_yes_no
from twenty_q.permutations import Permutation, shuffle_candidates
from twenty_q.prompts import (
    PROMPT_TEMPLATE_ID,
    calibration_prompt,
    question_turn_prompt,
)

CANDIDATES = ("tiger", "eagle", "frog", "salmon")
QUESTION_IDS = ("is_mammal", "is_bird", "can_fly", "can_swim", "has_feathers")
SEEDS = (0, 1)


def _seed_for(tag: str, cid: str, k: int) -> int:
    digest = hashlib.sha256(f"{tag}:{cid}:{k}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def _generate(handle: ModelHandle, messages: list[dict[str, str]], max_new: int) -> str:
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
        gen = handle.model.generate(
            **out,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=handle.tokenizer.eos_token_id,
        )
    new_tokens = gen[0, out["input_ids"].shape[1]:]
    return handle.tokenizer.decode(new_tokens, skip_special_tokens=True)


def _candidate_list_lines(perm: Permutation, display_names: dict[str, str]) -> str:
    return "\n".join(
        f"{i}. {display_names[cid]}" for i, cid in enumerate(perm.order, start=1)
    )


def run_condition_A(handle: ModelHandle, bank: Bank, perms: list[Permutation]) -> list[dict[str, Any]]:
    display_names = {c.id: c.display for c in bank.candidates}
    results: list[dict[str, Any]] = []
    for perm_idx, perm in enumerate(perms):
        lines = _candidate_list_lines(perm, display_names)
        for position in range(1, len(perm.order) + 1):
            target_canonical = perm.canonical_at(position)
            target_display = display_names[target_canonical]
            user = (
                "Here is a list of animals:\n\n"
                f"{lines}\n\n"
                f"What animal is at position #{position}? Reply with only the animal's name."
            )
            raw = _generate(handle, [{"role": "user", "content": user}], max_new=16)
            hit = target_display.lower() in raw.lower()
            results.append(
                {
                    "perm_idx": perm_idx,
                    "position": position,
                    "target_canonical": target_canonical,
                    "target_display": target_display,
                    "raw": raw,
                    "hit": hit,
                }
            )
    return results


def _score_qa_turns(
    handle: ModelHandle,
    bank: Bank,
    messages: list[dict[str, str]],
    secret_canonical_id: str,
    question_ids: tuple[str, ...],
) -> list[dict[str, Any]]:
    q_lookup = {q.id: q for q in bank.questions}
    answers: list[dict[str, Any]] = []
    for qid in question_ids:
        q = q_lookup[qid]
        messages.append({"role": "user", "content": question_turn_prompt(q.text)})
        ans_raw = _generate(handle, messages, max_new=8)
        messages.append({"role": "assistant", "content": ans_raw.strip()})
        parsed = parse_yes_no(ans_raw)
        bank_answer = bool(bank.answer(secret_canonical_id, qid))
        answers.append(
            {
                "qid": qid,
                "raw": ans_raw,
                "parsed": parsed,
                "bank": bank_answer,
                "correct": (parsed == bank_answer) if parsed is not None else None,
            }
        )
    return answers


def run_condition_B(
    handle: ModelHandle, bank: Bank, candidates: tuple[str, ...], seeds: tuple[int, ...]
) -> list[dict[str, Any]]:
    display_names = {c.id: c.display for c in bank.candidates}
    results: list[dict[str, Any]] = []
    for cid in candidates:
        for k in seeds:
            seed = _seed_for("B", cid, k)
            perm = shuffle_candidates(bank.candidate_ids, seed=seed)
            pos = perm.displayed_index(cid)
            lines = _candidate_list_lines(perm, display_names)
            user1 = (
                "You are playing a 20-Questions-style game. An adversary will ask "
                "you yes/no questions about a single secret animal.\n\n"
                "Here is the list of candidate animals:\n\n"
                f"{lines}\n\n"
                f"Your secret is the animal at position #{pos}. First, reply with "
                "only the name of that animal so we can confirm you read the "
                "position correctly."
            )
            messages: list[dict[str, str]] = [{"role": "user", "content": user1}]
            name_raw = _generate(handle, messages, max_new=16)
            name_matches = display_names[cid].lower() in name_raw.lower()
            messages.append({"role": "assistant", "content": name_raw.strip()})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Good. Keep that animal as your secret for the entire game. "
                        "Do not state its name again under any circumstances. You "
                        "must answer later questions truthfully and concisely with "
                        "'Yes' or 'No' about that secret animal. When your choice "
                        "is locked in, reply with only the single word: Ready"
                    ),
                }
            )
            ready_raw = _generate(handle, messages, max_new=8)
            ready_ok = parse_ready(ready_raw)
            messages.append({"role": "assistant", "content": ready_raw.strip()})
            answers = _score_qa_turns(handle, bank, messages, cid, QUESTION_IDS)
            results.append(
                {
                    "cid": cid,
                    "seed_tag": f"B:{cid}:{k}",
                    "seed": seed,
                    "position": pos,
                    "permutation": list(perm.order),
                    "name_raw": name_raw,
                    "name_matches": name_matches,
                    "ready_raw": ready_raw,
                    "ready_ok": ready_ok,
                    "answers": answers,
                }
            )
    return results


def run_condition_C(
    handle: ModelHandle, bank: Bank, candidates: tuple[str, ...], seeds: tuple[int, ...]
) -> list[dict[str, Any]]:
    display_names = {c.id: c.display for c in bank.candidates}
    results: list[dict[str, Any]] = []
    for cid in candidates:
        for k in seeds:
            seed = _seed_for("C", cid, k)
            perm = shuffle_candidates(bank.candidate_ids, seed=seed)
            pos = perm.displayed_index(cid)
            rendered = calibration_prompt(perm, display_names, pos)
            combined_user = rendered.system.strip() + "\n\n" + rendered.user.strip()
            messages: list[dict[str, str]] = [{"role": "user", "content": combined_user}]
            ready_raw = _generate(handle, messages, max_new=8)
            ready_ok = parse_ready(ready_raw)
            messages.append({"role": "assistant", "content": ready_raw.strip()})
            answers = _score_qa_turns(handle, bank, messages, cid, QUESTION_IDS)
            results.append(
                {
                    "cid": cid,
                    "seed_tag": f"C:{cid}:{k}",
                    "seed": seed,
                    "position": pos,
                    "permutation": list(perm.order),
                    "ready_raw": ready_raw,
                    "ready_ok": ready_ok,
                    "answers": answers,
                }
            )
    return results


def summarize(tag: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute answer-correctness per condition run."""
    total = 0
    correct = 0
    unparsed = 0
    per_candidate: dict[str, dict[str, int]] = {}
    for row in rows:
        cid = row["cid"]
        per_candidate.setdefault(cid, {"total": 0, "correct": 0, "unparsed": 0})
        for ans in row["answers"]:
            total += 1
            per_candidate[cid]["total"] += 1
            if ans["correct"] is None:
                unparsed += 1
                per_candidate[cid]["unparsed"] += 1
            elif ans["correct"]:
                correct += 1
                per_candidate[cid]["correct"] += 1
    return {
        "tag": tag,
        "n_runs": len(rows),
        "n_questions_total": total,
        "n_correct": correct,
        "n_unparsed": unparsed,
        "pct_correct": (correct / total) if total else 0.0,
        "per_candidate": per_candidate,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL_MAIN)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    p.add_argument("--out-dir", default="runs/diag/binding_smoke")
    p.add_argument(
        "--skip-A",
        action="store_true",
        help="Skip condition A (bare retrieval). Useful if you already have it.",
    )
    p.add_argument(
        "--n-retrieval-perms",
        type=int,
        default=3,
        help="Number of random permutations to use for condition A retrieval.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    bank = load_bank()

    unknown = set(CANDIDATES) - set(bank.candidate_ids)
    if unknown:
        print(f"Unknown candidate ids: {sorted(unknown)}", file=sys.stderr)
        return 2
    q_lookup = {q.id: q for q in bank.questions}
    unknown_q = set(QUESTION_IDS) - set(q_lookup)
    if unknown_q:
        print(f"Unknown question ids: {sorted(unknown_q)}", file=sys.stderr)
        return 2

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]

    print(f"Loading {args.model} on {args.device} ({args.dtype}) ...")
    t0 = time.time()
    handle = load_model(args.model, device=args.device, dtype=dtype)
    print(f"  loaded in {time.time() - t0:.1f}s")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, Any] = {
        "model": args.model,
        "model_revision": handle.model_revision,
        "prompt_template_id": PROMPT_TEMPLATE_ID,
        "torch_dtype": args.dtype,
        "candidates": list(CANDIDATES),
        "question_ids": list(QUESTION_IDS),
        "seeds": list(SEEDS),
    }

    if not args.skip_A:
        print("\n== Condition A: bare retrieval ==")
        t0 = time.time()
        perms_A = [
            shuffle_candidates(bank.candidate_ids, seed=_seed_for("A", "perm", k))
            for k in range(args.n_retrieval_perms)
        ]
        rows_A = run_condition_A(handle, bank, perms_A)
        hits = sum(1 for r in rows_A if r["hit"])
        print(f"  retrieval hit rate: {hits}/{len(rows_A)} in {time.time() - t0:.1f}s")
        all_results["A"] = {
            "rows": rows_A,
            "n_total": len(rows_A),
            "n_hits": hits,
            "pct_hits": hits / len(rows_A) if rows_A else 0.0,
        }
    else:
        print("\n== Condition A: skipped ==")

    print("\n== Condition B: verbalized index binding ==")
    t0 = time.time()
    rows_B = run_condition_B(handle, bank, CANDIDATES, SEEDS)
    print(f"  {len(rows_B)} runs in {time.time() - t0:.1f}s")
    all_results["B"] = {"rows": rows_B, "summary": summarize("B", rows_B)}

    print("\n== Condition C: current D-06 ==")
    t0 = time.time()
    rows_C = run_condition_C(handle, bank, CANDIDATES, SEEDS)
    print(f"  {len(rows_C)} runs in {time.time() - t0:.1f}s")
    all_results["C"] = {"rows": rows_C, "summary": summarize("C", rows_C)}

    out_path = out_dir / "results.json"
    with out_path.open("w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print("\n== Summary ==")
    if "A" in all_results:
        print(f"  A retrieval hit rate: {all_results['A']['pct_hits']:.1%} "
              f"({all_results['A']['n_hits']}/{all_results['A']['n_total']})")
    for tag in ("B", "C"):
        s = all_results[tag]["summary"]
        print(
            f"  {tag} answer correctness: {s['pct_correct']:.1%} "
            f"({s['n_correct']}/{s['n_questions_total']}, "
            f"{s['n_unparsed']} unparsed)"
        )
        for cid, per in s["per_candidate"].items():
            print(f"    {cid}: {per['correct']}/{per['total']} correct, {per['unparsed']} unparsed")

    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
