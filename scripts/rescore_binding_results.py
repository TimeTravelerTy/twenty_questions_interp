"""Offline rescoring for binding-smoke results under disputed bank cells.

This does not edit the canonical answer table. It rewrites the stored `bank`
labels inside an existing `results.json`, recomputes correctness summaries, and
prints the delta per condition.

Usage:
    uv run python scripts/rescore_binding_results.py \
        --results-json runs/diag/binding_smoke_5_20260419/results.json \
        --override frog.has_four_legs=0 \
        --override frog.lives_primarily_in_water=0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from twenty_q.binding_audit import apply_overrides_to_rows, parse_override, summarize_rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results-json", required=True, help="Path to an existing binding results.json")
    p.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override candidate.question_id=0|1; may be repeated",
    )
    p.add_argument(
        "--write-json",
        default="",
        help="Optional path to write the rescored payload JSON",
    )
    return p.parse_args()


def _format_split(summary: dict[str, Any], split: str) -> str:
    bucket = summary[split]
    return f"{bucket['n_correct']}/{bucket['n_questions_total']} ({bucket['pct_correct']:.1%})"


def main() -> int:
    args = parse_args()
    payload = json.loads(Path(args.results_json).read_text())

    overrides: dict[tuple[str, str], bool] = {}
    for spec in args.override:
        candidate, question_id, value = parse_override(spec)
        overrides[(candidate, question_id)] = value

    primary_question_ids = tuple(payload.get("primary_question_ids", []))
    rescored_payload = json.loads(json.dumps(payload))
    audit_by_condition: dict[str, list[dict[str, Any]]] = {}

    print("Overrides:")
    for (candidate, question_id), value in sorted(overrides.items()):
        print(f"  {candidate}.{question_id} -> {int(value)}")
    if not overrides:
        print("  (none)")

    print("\nPer-condition delta:")
    for condition in payload["conditions"]:
        rows = payload[condition]["rows"]
        rescored_rows, changes = apply_overrides_to_rows(rows, overrides)
        original_summary = summarize_rows(rows, primary_question_ids)
        rescored_summary = summarize_rows(rescored_rows, primary_question_ids)
        rescored_payload[condition]["rows"] = rescored_rows
        rescored_payload[condition]["correctness_rescored"] = rescored_summary
        audit_by_condition[condition] = changes

        print(f"\n== {condition} ==")
        print(f"  original primary:  {_format_split(original_summary, 'primary')}")
        print(f"  rescored primary:  {_format_split(rescored_summary, 'primary')}")
        print(f"  original secondary:{_format_split(original_summary, 'secondary')}")
        print(f"  rescored secondary:{_format_split(rescored_summary, 'secondary')}")
        if changes:
            print("  changed answers:")
            for change in changes:
                old = "correct" if change["old_correct"] else "wrong"
                new = "correct" if change["new_correct"] else "wrong"
                print(
                    f"    {change['run_id']} {change['question_id']}: "
                    f"bank {int(change['old_bank'])}->{int(change['new_bank'])}, {old}->{new}"
                )
        else:
            print("  changed answers: none")

    rescored_payload["audit_overrides"] = {
        f"{candidate}.{question_id}": value
        for (candidate, question_id), value in sorted(overrides.items())
    }
    rescored_payload["audit_changes"] = audit_by_condition

    if args.write_json:
        out_path = Path(args.write_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(rescored_payload, indent=2))
        print(f"\nWrote rescored payload to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
