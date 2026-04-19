"""Run the M3 calibration-binding smoke described in STATUS.md.

Six calibration conditions are defined; the default selection runs the three
non-index variants used in the post-4cond follow-up:

  index             current D-06 index-based prompt (legacy reference)
  index_reminder    index + explicit per-turn position reminder (legacy)
  name              name-based ("Your secret animal is X") (legacy)
  name_paraphrase   name paraphrase ("You have chosen X as your secret animal")
  name_strict       name paraphrase + "answer only about X, not the average
                    animal in the candidate list"
  verbalized_index  two-turn binding: model first verbalizes the animal at the
                    indexed position, then locks it in and emits Ready

For each run it:

- captures all-layer activations at the token position immediately before `Ready`
- continues through fixed yes/no question turns
- captures all-layer pre-answer activations for every turn
- scores answer correctness against the bank, split into a primary set
  (decision gate) and a secondary set (sanity reporting only)
- summarizes Ready-state representations as both within-secret cosine and a
  within-vs-between contrast across candidates

It writes one JSON summary plus per-run activation tensors under `--out-dir`.
The intended use is to decide whether D-06 should be reversed before launching the
full ~2k M3 calibration run.

Usage:
    uv run python scripts/diagnose_index_binding.py \
        --model google/gemma-3-4b-it --device auto
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

import torch

from twenty_q.banks import Bank, load_bank
from twenty_q.config import MODEL_MAIN
from twenty_q.dialogue import ModelHandle, load_model, parse_ready, parse_yes_no
from twenty_q.permutations import Permutation, shuffle_candidates
from twenty_q.prompts import (
    PROMPT_TEMPLATE_ID,
    RenderedPrompt,
    calibration_prompt,
    question_turn_prompt,
)

DEFAULT_CANDIDATES = ("tiger", "eagle", "frog", "salmon")
DEFAULT_QUESTION_IDS = (
    "is_mammal",
    "is_bird",
    "lives_primarily_in_water",
    "has_four_legs",
    "can_swim",
)
DEFAULT_PRIMARY_QUESTION_IDS = (
    "is_mammal",
    "is_bird",
    "lives_primarily_in_water",
    "has_four_legs",
)
DEFAULT_SEEDS = (0, 1)
DEFAULT_CONDITION_TAGS = ("name_paraphrase", "name_strict", "verbalized_index")


@dataclass(frozen=True)
class ConditionSpec:
    tag: str
    label: str


CONDITIONS = (
    ConditionSpec(tag="index", label="Current D-06 index binding"),
    ConditionSpec(tag="index_reminder", label="Index binding + explicit position reminder"),
    ConditionSpec(tag="name", label='Name binding ("Your secret animal is X")'),
    ConditionSpec(
        tag="name_paraphrase",
        label='Name paraphrase ("You have chosen X as your secret animal")',
    ),
    ConditionSpec(
        tag="name_strict",
        label='Name paraphrase + "answer only about X, not the average animal"',
    ),
    ConditionSpec(
        tag="verbalized_index",
        label="Two-turn binding: model verbalizes name from indexed position, then locks in",
    ),
)


def _seed_for(tag: str, cid: str, k: int) -> int:
    digest = hashlib.sha256(f"{tag}:{cid}:{k}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def _candidate_list_lines(perm: Permutation, display_names: dict[str, str]) -> str:
    return "\n".join(
        f"{i}. {display_names[cid]}" for i, cid in enumerate(perm.order, start=1)
    )


def _ready_capture_and_generate(
    handle: ModelHandle,
    messages: list[dict[str, str]],
    max_new_tokens: int = 8,
) -> tuple[torch.Tensor, str]:
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
    return states, raw


def _question_prompt(question_text: str, reminder: str | None = None) -> str:
    base = question_turn_prompt(question_text)
    if reminder is None:
        return base
    return f"{reminder.strip()}\n\n{base}"


def _build_condition_prompt(
    condition: ConditionSpec,
    perm: Permutation,
    display_names: dict[str, str],
    secret_displayed_index: int,
    secret_display_name: str,
) -> RenderedPrompt:
    if condition.tag == "index":
        return calibration_prompt(perm, display_names, secret_displayed_index)

    system = (
        "You are playing a 20-Questions-style game. An adversary will ask you "
        "yes/no questions about a single secret animal that you are holding in "
        "mind. You must answer each question truthfully and concisely with "
        "either 'Yes' or 'No' about the secret animal. Do not reveal the "
        "identity of the secret animal under any circumstances."
    )
    if condition.tag == "index_reminder":
        user = (
            "Here is the list of candidate animals:\n\n"
            f"{_candidate_list_lines(perm, display_names)}\n\n"
            f"Your secret is the animal shown at position #{secret_displayed_index} "
            "in the list above. Keep that choice fixed and secret for the entire "
            "game. Do not state the name of the animal. When your choice is locked "
            "in, reply with only the single word: Ready"
        )
    elif condition.tag == "name":
        user = (
            "Here is the list of candidate animals:\n\n"
            f"{_candidate_list_lines(perm, display_names)}\n\n"
            f"Your secret animal is {secret_display_name}. Keep that choice fixed "
            "and secret for the entire game. Do not state the name of the animal "
            "again. When your choice is locked in, reply with only the single word: Ready"
        )
    elif condition.tag == "name_paraphrase":
        user = (
            "Here is the list of candidate animals:\n\n"
            f"{_candidate_list_lines(perm, display_names)}\n\n"
            f"You have chosen {secret_display_name} as your secret animal. Keep using "
            "that same animal for the entire game. Do not state the name of the animal "
            "again. When your choice is locked in, reply with only the single word: Ready"
        )
    elif condition.tag == "name_strict":
        user = (
            "Here is the list of candidate animals:\n\n"
            f"{_candidate_list_lines(perm, display_names)}\n\n"
            f"You have chosen {secret_display_name} as your secret animal. Keep using "
            "that same animal for the entire game. When the adversary asks a yes/no "
            f"question, answer it only about {secret_display_name} specifically — not "
            "about the average animal in the candidate list, and not about a related "
            "animal. Do not state the name of the animal again. When your choice is "
            "locked in, reply with only the single word: Ready"
        )
    elif condition.tag == "verbalized_index":
        # The first user turn used to elicit the model's verbalization of the
        # animal at the indexed position. The second turn (lock-in + Ready) is
        # injected by `_seed_binding_messages` once the model has spoken the
        # name. Returning a single RenderedPrompt here would be misleading; the
        # caller routes verbalized_index through `_seed_binding_messages`.
        user = (
            "Here is the list of candidate animals:\n\n"
            f"{_candidate_list_lines(perm, display_names)}\n\n"
            f"Your secret is the animal shown at position #{secret_displayed_index} "
            "in the list above. First, identify which animal that is by replying "
            "with only the animal's name on a single line — nothing else."
        )
    else:
        raise ValueError(f"Unknown condition {condition.tag!r}")
    return RenderedPrompt(system=system, user=user)


VERBALIZED_LOCKIN_USER = (
    "Good. Keep that same animal as your secret for the entire game. Do not "
    "state the name of the animal again. When your choice is locked in, reply "
    "with only the single word: Ready"
)


def _question_reminder(condition: ConditionSpec, secret_displayed_index: int) -> str | None:
    if condition.tag == "index_reminder":
        return (
            "Remember: your secret is the animal at position "
            f"#{secret_displayed_index} from the original candidate list. "
            "Answer only about that animal."
        )
    return None


def _combined_user(rendered: RenderedPrompt) -> str:
    return rendered.system.strip() + "\n\n" + rendered.user.strip()


def _generate_reply(
    handle: ModelHandle,
    messages: list[dict[str, str]],
    max_new_tokens: int = 16,
) -> str:
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
        gen = handle.model.generate(
            **out,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=handle.tokenizer.eos_token_id,
        )
    new_tokens = gen[0, input_ids.shape[1]:]
    return handle.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _seed_binding_messages(
    handle: ModelHandle,
    condition: ConditionSpec,
    rendered: RenderedPrompt,
    secret_display_name: str,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Build the message list to pass to Ready capture.

    For most conditions this is just the single combined-user turn. For
    verbalized_index it additionally elicits the model's name verbalization
    and appends the lock-in instruction so Ready capture is one turn removed
    from the verbalized name token.
    """
    messages: list[dict[str, str]] = [
        {"role": "user", "content": _combined_user(rendered)}
    ]
    extra: dict[str, Any] = {}
    if condition.tag != "verbalized_index":
        return messages, extra

    raw_name = _generate_reply(handle, messages, max_new_tokens=16)
    cleaned = raw_name.strip().splitlines()[0].strip().strip(".") if raw_name else ""
    matches = cleaned.lower() == secret_display_name.lower()
    extra = {
        "verbalized_name_raw": raw_name,
        "verbalized_name_cleaned": cleaned,
        "verbalized_name_matches": matches,
        "expected_name": secret_display_name,
    }
    messages.append({"role": "assistant", "content": raw_name})
    messages.append({"role": "user", "content": VERBALIZED_LOCKIN_USER})
    return messages, extra


def _score_qa_turns(
    handle: ModelHandle,
    bank: Bank,
    messages: list[dict[str, str]],
    secret_canonical_id: str,
    question_ids: tuple[str, ...],
    reminder: str | None,
    run_dir: Path,
) -> list[dict[str, Any]]:
    q_lookup = {q.id: q for q in bank.questions}
    answers: list[dict[str, Any]] = []
    for turn_idx, qid in enumerate(question_ids, start=1):
        q = q_lookup[qid]
        user_content = _question_prompt(q.text, reminder=reminder)
        messages.append({"role": "user", "content": user_content})
        activations, ans_raw = _ready_capture_and_generate(
            handle,
            messages,
            max_new_tokens=8,
        )
        act_path = run_dir / f"turn_{turn_idx:02d}_activations.pt"
        torch.save(activations, act_path)
        messages.append({"role": "assistant", "content": ans_raw.strip()})
        parsed = parse_yes_no(ans_raw)
        bank_answer = bool(bank.answer(secret_canonical_id, qid))
        answers.append(
            {
                "qid": qid,
                "question_text": q.text,
                "raw": ans_raw,
                "parsed": parsed,
                "bank": bank_answer,
                "correct": (parsed == bank_answer) if parsed is not None else None,
                "activation_path": str(act_path),
            }
        )
    return answers


def _run_condition(
    handle: ModelHandle,
    bank: Bank,
    condition: ConditionSpec,
    candidates: tuple[str, ...],
    seeds: tuple[int, ...],
    question_ids: tuple[str, ...],
    out_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, list[torch.Tensor]]]:
    display_names = {c.id: c.display for c in bank.candidates}
    rows: list[dict[str, Any]] = []
    ready_states: dict[str, list[torch.Tensor]] = {}

    for cid in candidates:
        for k in seeds:
            seed = _seed_for(condition.tag, cid, k)
            perm = shuffle_candidates(bank.candidate_ids, seed=seed)
            pos = perm.displayed_index(cid)
            display_name = display_names[cid]
            rendered = _build_condition_prompt(
                condition=condition,
                perm=perm,
                display_names=display_names,
                secret_displayed_index=pos,
                secret_display_name=display_name,
            )
            reminder = _question_reminder(condition, pos)
            run_id = f"{condition.tag}_{cid}_{k:02d}"
            run_dir = out_dir / condition.tag / run_id
            run_dir.mkdir(parents=True, exist_ok=True)

            messages, binding_extra = _seed_binding_messages(
                handle=handle,
                condition=condition,
                rendered=rendered,
                secret_display_name=display_name,
            )
            ready_activations, ready_raw = _ready_capture_and_generate(handle, messages)
            ready_act_path = run_dir / "activations.pt"
            torch.save(ready_activations, ready_act_path)
            messages.append({"role": "assistant", "content": ready_raw.strip()})
            answers = _score_qa_turns(
                handle=handle,
                bank=bank,
                messages=messages,
                secret_canonical_id=cid,
                question_ids=question_ids,
                reminder=reminder,
                run_dir=run_dir,
            )

            row = {
                "run_id": run_id,
                "condition": condition.tag,
                "condition_label": condition.label,
                "cid": cid,
                "seed_tag": f"{condition.tag}:{cid}:{k}",
                "seed": seed,
                "position": pos,
                "permutation": list(perm.order),
                "prompt_template_id": PROMPT_TEMPLATE_ID,
                "ready_raw": ready_raw,
                "ready_ok": parse_ready(ready_raw),
                "ready_activation_path": str(ready_act_path),
                "answers": answers,
                "binding_extra": binding_extra,
            }
            with (run_dir / "result.json").open("w") as f:
                json.dump(row, f, indent=2, default=str)
            rows.append(row)
            ready_states.setdefault(cid, []).append(ready_activations)

    return rows, ready_states


def _cosine_summary(ready_states: dict[str, list[torch.Tensor]]) -> dict[str, Any]:
    per_candidate: dict[str, Any] = {}
    within_pairs: list[torch.Tensor] = []
    between_pairs: list[torch.Tensor] = []
    cids = list(ready_states.keys())

    for cid, tensors in ready_states.items():
        if len(tensors) < 2:
            per_candidate[cid] = {"n_runs": len(tensors), "mean_pairwise_cosine_by_layer": []}
            continue
        pair_cosines: list[torch.Tensor] = []
        for i in range(len(tensors)):
            for j in range(i + 1, len(tensors)):
                pair_cosines.append(
                    torch.nn.functional.cosine_similarity(
                        tensors[i],
                        tensors[j],
                        dim=1,
                    )
                )
        stacked = torch.stack(pair_cosines, dim=0)
        mean_by_layer = stacked.mean(dim=0)
        per_candidate[cid] = {
            "n_runs": len(tensors),
            "mean_pairwise_cosine_by_layer": [float(x) for x in mean_by_layer.tolist()],
            "mean_pairwise_cosine_post13": float(mean_by_layer[13:].mean().item()),
        }
        within_pairs.append(stacked)

    for a_idx in range(len(cids)):
        for b_idx in range(a_idx + 1, len(cids)):
            a_tensors = ready_states[cids[a_idx]]
            b_tensors = ready_states[cids[b_idx]]
            for ta in a_tensors:
                for tb in b_tensors:
                    between_pairs.append(
                        torch.nn.functional.cosine_similarity(ta, tb, dim=1)
                    )

    overall: dict[str, Any] = {}
    if within_pairs:
        within_all = torch.cat(within_pairs, dim=0)
        within_mean = within_all.mean(dim=0)
        overall["within"] = {
            "n_pairs": int(within_all.shape[0]),
            "mean_cosine_by_layer": [float(x) for x in within_mean.tolist()],
            "mean_cosine_post13": float(within_mean[13:].mean().item()),
        }
        # Back-compat aliases used by the existing 4cond progress note format.
        overall["mean_pairwise_cosine_by_layer"] = overall["within"]["mean_cosine_by_layer"]
        overall["mean_pairwise_cosine_post13"] = overall["within"]["mean_cosine_post13"]
    if between_pairs:
        between_all = torch.stack(between_pairs, dim=0)
        between_mean = between_all.mean(dim=0)
        overall["between"] = {
            "n_pairs": int(between_all.shape[0]),
            "mean_cosine_by_layer": [float(x) for x in between_mean.tolist()],
            "mean_cosine_post13": float(between_mean[13:].mean().item()),
        }
    if "within" in overall and "between" in overall:
        contrast_by_layer = [
            w - b
            for w, b in zip(
                overall["within"]["mean_cosine_by_layer"],
                overall["between"]["mean_cosine_by_layer"],
                strict=True,
            )
        ]
        overall["contrast"] = {
            "within_minus_between_by_layer": contrast_by_layer,
            "within_minus_between_post13": float(
                sum(contrast_by_layer[13:]) / max(1, len(contrast_by_layer[13:]))
            ),
        }
    return {"per_candidate": per_candidate, "overall": overall}


def _empty_split_counts() -> dict[str, int]:
    return {"n_questions_total": 0, "n_correct": 0, "n_unparsed": 0}


def _split_pct(counts: dict[str, int]) -> float:
    n = counts["n_questions_total"]
    return counts["n_correct"] / n if n else 0.0


def _correctness_summary(
    rows: list[dict[str, Any]],
    primary_question_ids: tuple[str, ...],
) -> dict[str, Any]:
    primary_set = set(primary_question_ids)
    total = _empty_split_counts()
    primary = _empty_split_counts()
    secondary = _empty_split_counts()
    ready_ok = 0
    per_candidate: dict[str, dict[str, Any]] = {}
    for row in rows:
        cid = row["cid"]
        cand = per_candidate.setdefault(
            cid,
            {
                "n_runs": 0,
                "n_ready_ok": 0,
                "total": _empty_split_counts(),
                "primary": _empty_split_counts(),
                "secondary": _empty_split_counts(),
            },
        )
        cand["n_runs"] += 1
        ready_ok += int(row["ready_ok"])
        cand["n_ready_ok"] += int(row["ready_ok"])
        for ans in row["answers"]:
            bucket = primary if ans["qid"] in primary_set else secondary
            for counts in (total, bucket, cand["total"], cand[
                "primary" if ans["qid"] in primary_set else "secondary"
            ]):
                counts["n_questions_total"] += 1
                if ans["correct"] is None:
                    counts["n_unparsed"] += 1
                elif ans["correct"]:
                    counts["n_correct"] += 1

    summary = {
        "n_runs": len(rows),
        "n_ready_ok": ready_ok,
        "pct_ready_ok": (ready_ok / len(rows)) if rows else 0.0,
        "primary_question_ids": list(primary_question_ids),
        "total": {**total, "pct_correct": _split_pct(total)},
        "primary": {**primary, "pct_correct": _split_pct(primary)},
        "secondary": {**secondary, "pct_correct": _split_pct(secondary)},
        "per_candidate": per_candidate,
    }
    for _cid, per in per_candidate.items():
        per["pct_ready_ok"] = per["n_ready_ok"] / per["n_runs"] if per["n_runs"] else 0.0
        for split_name in ("total", "primary", "secondary"):
            per[split_name]["pct_correct"] = _split_pct(per[split_name])
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL_MAIN)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    p.add_argument("--out-dir", default="runs/diag/binding_smoke")
    p.add_argument(
        "--candidates",
        default=",".join(DEFAULT_CANDIDATES),
        help="Comma-separated candidate ids to use.",
    )
    p.add_argument(
        "--question-ids",
        default=",".join(DEFAULT_QUESTION_IDS),
        help="Comma-separated question ids to ask after Ready.",
    )
    p.add_argument(
        "--primary-question-ids",
        default=",".join(DEFAULT_PRIMARY_QUESTION_IDS),
        help=(
            "Comma-separated subset of --question-ids that count toward the "
            "primary correctness gate; the rest are reported as secondary only."
        ),
    )
    p.add_argument(
        "--seeds",
        default=",".join(str(x) for x in DEFAULT_SEEDS),
        help="Comma-separated small integer seed indices per candidate/condition.",
    )
    p.add_argument(
        "--conditions",
        default=",".join(DEFAULT_CONDITION_TAGS),
        help=(
            "Comma-separated subset of conditions. Available: "
            f"{','.join(c.tag for c in CONDITIONS)}"
        ),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    bank = load_bank()

    candidates = tuple(x.strip() for x in args.candidates.split(",") if x.strip())
    question_ids = tuple(x.strip() for x in args.question_ids.split(",") if x.strip())
    primary_question_ids = tuple(
        x.strip() for x in args.primary_question_ids.split(",") if x.strip()
    )
    seeds = tuple(int(x.strip()) for x in args.seeds.split(",") if x.strip())
    condition_lookup = {c.tag: c for c in CONDITIONS}
    selected_conditions = tuple(
        condition_lookup[tag.strip()] for tag in args.conditions.split(",") if tag.strip()
    )

    unknown_candidates = sorted(set(candidates) - set(bank.candidate_ids))
    if unknown_candidates:
        print(f"Unknown candidate ids: {unknown_candidates}", file=sys.stderr)
        return 2
    q_lookup = {q.id: q for q in bank.questions}
    unknown_questions = sorted(set(question_ids) - set(q_lookup))
    if unknown_questions:
        print(f"Unknown question ids: {unknown_questions}", file=sys.stderr)
        return 2
    primary_outside_asked = sorted(set(primary_question_ids) - set(question_ids))
    if primary_outside_asked:
        print(
            f"Primary question ids not in --question-ids: {primary_outside_asked}",
            file=sys.stderr,
        )
        return 2
    unknown_conditions = sorted(
        {
            tag.strip()
            for tag in args.conditions.split(",")
            if tag.strip() and tag.strip() not in condition_lookup
        }
    )
    if unknown_conditions:
        print(f"Unknown conditions: {unknown_conditions}", file=sys.stderr)
        return 2
    if not selected_conditions:
        print("No conditions selected.", file=sys.stderr)
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

    results: dict[str, Any] = {
        "model": args.model,
        "model_revision": handle.model_revision,
        "tokenizer_revision": handle.tokenizer_revision,
        "torch_dtype": args.dtype,
        "prompt_template_id": PROMPT_TEMPLATE_ID,
        "candidates": list(candidates),
        "question_ids": list(question_ids),
        "primary_question_ids": list(primary_question_ids),
        "seeds": list(seeds),
        "conditions": [c.tag for c in selected_conditions],
    }

    for condition in selected_conditions:
        print(f"\n== {condition.tag}: {condition.label} ==")
        t0 = time.time()
        rows, ready_states = _run_condition(
            handle=handle,
            bank=bank,
            condition=condition,
            candidates=candidates,
            seeds=seeds,
            question_ids=question_ids,
            out_dir=out_dir,
        )
        correctness = _correctness_summary(rows, primary_question_ids)
        cosine = _cosine_summary(ready_states)
        results[condition.tag] = {
            "label": condition.label,
            "rows": rows,
            "correctness": correctness,
            "ready_cosine": cosine,
        }
        contrast = cosine["overall"].get("contrast", {}).get(
            "within_minus_between_post13", float("nan")
        )
        print(
            f"  {len(rows)} runs in {time.time() - t0:.1f}s; "
            f"primary {correctness['primary']['pct_correct']:.1%} "
            f"({correctness['primary']['n_correct']}/{correctness['primary']['n_questions_total']}); "
            f"secondary {correctness['secondary']['pct_correct']:.1%} "
            f"({correctness['secondary']['n_correct']}/{correctness['secondary']['n_questions_total']}); "
            f"post-13 within-cos "
            f"{cosine['overall'].get('mean_pairwise_cosine_post13', float('nan')):.5f}; "
            f"post-13 within-between contrast {contrast:+.2e}"
        )

    out_path = out_dir / "results.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n== Summary ==")
    for condition in selected_conditions:
        payload = results[condition.tag]
        correctness = payload["correctness"]
        cosine = payload["ready_cosine"]["overall"]
        contrast = cosine.get("contrast", {}).get(
            "within_minus_between_post13", float("nan")
        )
        print(
            f"  {condition.tag}: "
            f"ready {correctness['pct_ready_ok']:.1%} "
            f"({correctness['n_ready_ok']}/{correctness['n_runs']}), "
            f"primary {correctness['primary']['pct_correct']:.1%} "
            f"({correctness['primary']['n_correct']}/{correctness['primary']['n_questions_total']}), "
            f"secondary {correctness['secondary']['pct_correct']:.1%} "
            f"({correctness['secondary']['n_correct']}/{correctness['secondary']['n_questions_total']}), "
            f"unparsed={correctness['total']['n_unparsed']}, "
            f"within-cos post13={cosine.get('mean_pairwise_cosine_post13', float('nan')):.5f}, "
            f"contrast post13={contrast:+.2e}"
        )
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
