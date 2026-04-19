"""Offline rescoring helpers for binding-smoke result payloads."""
from __future__ import annotations

from copy import deepcopy
from typing import Any


def _empty_split_counts() -> dict[str, int]:
    return {"n_questions_total": 0, "n_correct": 0, "n_unparsed": 0}


def _split_pct(counts: dict[str, int]) -> float:
    n = counts["n_questions_total"]
    return counts["n_correct"] / n if n else 0.0


def summarize_rows(
    rows: list[dict[str, Any]],
    primary_question_ids: tuple[str, ...],
) -> dict[str, Any]:
    """Summarize correctness with primary/secondary splits."""
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
            bucket_name = "primary" if ans["qid"] in primary_set else "secondary"
            bucket = primary if bucket_name == "primary" else secondary
            for counts in (total, bucket, cand["total"], cand[bucket_name]):
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
    for per in per_candidate.values():
        per["pct_ready_ok"] = per["n_ready_ok"] / per["n_runs"] if per["n_runs"] else 0.0
        for split_name in ("total", "primary", "secondary"):
            per[split_name]["pct_correct"] = _split_pct(per[split_name])
    return summary


def parse_override(spec: str) -> tuple[str, str, bool]:
    """Parse `candidate.question_id=0|1|true|false`."""
    try:
        lhs, rhs = spec.split("=", 1)
        candidate, question_id = lhs.split(".", 1)
    except ValueError as exc:
        raise ValueError(
            f"Invalid override {spec!r}; expected candidate.question_id=0|1"
        ) from exc

    rhs_l = rhs.strip().lower()
    if rhs_l in {"1", "true", "yes"}:
        value = True
    elif rhs_l in {"0", "false", "no"}:
        value = False
    else:
        raise ValueError(
            f"Invalid override value in {spec!r}; expected 0/1/true/false"
        )
    return candidate.strip(), question_id.strip(), value


def apply_overrides_to_rows(
    rows: list[dict[str, Any]],
    overrides: dict[tuple[str, str], bool],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return rescored rows plus an audit trail of changed answers."""
    rescored = deepcopy(rows)
    changes: list[dict[str, Any]] = []

    for row in rescored:
        cid = row["cid"]
        for ans in row["answers"]:
            key = (cid, ans["qid"])
            if key not in overrides:
                continue
            old_bank = bool(ans["bank"])
            new_bank = overrides[key]
            old_correct = ans["correct"]
            ans["bank"] = new_bank
            ans["correct"] = (ans["parsed"] == new_bank) if ans["parsed"] is not None else None
            changes.append(
                {
                    "run_id": row["run_id"],
                    "candidate": cid,
                    "question_id": ans["qid"],
                    "raw": ans["raw"],
                    "old_bank": old_bank,
                    "new_bank": new_bank,
                    "old_correct": old_correct,
                    "new_correct": ans["correct"],
                }
            )
    return rescored, changes

