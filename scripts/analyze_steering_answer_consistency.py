"""Posthoc answer-consistency metrics for rollout steering runs.

Raw answer flips only compare regenerated answers to the target run's
answers. That misses cases where source and target animals share the same
answer for a question. This script compares regenerated answers against the
bank-expected answer vector for both the source class and target class.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twenty_q.banks import load_bank  # noqa: E402
from twenty_q.manifest import RunManifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True,
                   help="Original self-chosen run dir containing attempt_*/manifest.json")
    p.add_argument("--steer-json", required=True,
                   help="Output JSON from scripts/steer_class_direction.py --answer-rollout")
    p.add_argument("--out-json", default=None)
    return p.parse_args()


def _load_manifests(run_dir: Path) -> dict[str, RunManifest]:
    out: dict[str, RunManifest] = {}
    for mp in sorted(run_dir.glob("attempt_*/manifest.json")):
        m = RunManifest.load(mp)
        out[m.run_id] = m
    return out


def _add_counts(dst: dict[str, int], src: dict[str, int]) -> None:
    for k, v in src.items():
        dst[k] = dst.get(k, 0) + v


def _trial_counts(record: dict[str, Any], manifest: RunManifest, bank: Any) -> dict[str, int]:
    src = record["src_class"]
    tgt = record["tgt_class"]
    answers = record.get("regen_answers_bool") or []
    counts = {
        "slots": 0,
        "parsed_slots": 0,
        "unparsed_slots": 0,
        "source_expected_yes": 0,
        "target_expected_yes": 0,
        "source_target_same": 0,
        "source_target_different": 0,
        "parsed_source_target_different": 0,
        "match_source_all": 0,
        "match_target_all": 0,
        "match_both_all": 0,
        "match_neither_all": 0,
        "match_source_diagnostic": 0,
        "match_target_diagnostic": 0,
        "match_neither_diagnostic": 0,
    }
    for turn, got in zip(manifest.turns, answers, strict=False):
        qid = turn.question_id
        src_expected = bool(bank.answer(src, qid))
        tgt_expected = bool(bank.answer(tgt, qid))
        counts["slots"] += 1
        counts["source_expected_yes"] += int(src_expected)
        counts["target_expected_yes"] += int(tgt_expected)
        same = src_expected == tgt_expected
        counts["source_target_same"] += int(same)
        counts["source_target_different"] += int(not same)
        if got is None:
            counts["unparsed_slots"] += 1
            continue
        counts["parsed_slots"] += 1
        match_source = got == src_expected
        match_target = got == tgt_expected
        counts["match_source_all"] += int(match_source)
        counts["match_target_all"] += int(match_target)
        counts["match_both_all"] += int(match_source and match_target)
        counts["match_neither_all"] += int((not match_source) and (not match_target))
        if not same:
            counts["parsed_source_target_different"] += 1
            counts["match_source_diagnostic"] += int(match_source)
            counts["match_target_diagnostic"] += int(match_target)
            counts["match_neither_diagnostic"] += int((not match_source) and (not match_target))
    return counts


def _rate(num: int, den: int) -> float | None:
    return None if den == 0 else num / den


def _pct(value: float | None) -> str:
    return "   n/a" if value is None else f"{100*value:5.1f}%"


def _summarize_counts(counts: dict[str, int]) -> dict[str, Any]:
    slots = counts.get("slots", 0)
    parsed = counts.get("parsed_slots", 0)
    diag = counts.get("source_target_different", 0)
    parsed_diag = counts.get("parsed_source_target_different", 0)
    return {
        **counts,
        "parsed_rate": _rate(parsed, slots),
        "source_target_different_rate": _rate(diag, slots),
        "match_source_all_rate": _rate(counts.get("match_source_all", 0), parsed),
        "match_target_all_rate": _rate(counts.get("match_target_all", 0), parsed),
        "match_both_all_rate": _rate(counts.get("match_both_all", 0), parsed),
        "match_neither_all_rate": _rate(counts.get("match_neither_all", 0), parsed),
        "match_source_diagnostic_rate": _rate(
            counts.get("match_source_diagnostic", 0), parsed_diag
        ),
        "match_target_diagnostic_rate": _rate(
            counts.get("match_target_diagnostic", 0), parsed_diag
        ),
    }


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    steer_path = Path(args.steer_json)
    bank = load_bank()
    manifests = _load_manifests(run_dir)
    data = json.loads(steer_path.read_text())

    by_alpha: dict[str, dict[str, int]] = {}
    by_alpha_reveal_src: dict[str, dict[str, int]] = {}
    by_alpha_not_reveal_src: dict[str, dict[str, int]] = {}

    for r in data["steered_trials"]:
        rid = r["tgt_run"]
        if rid not in manifests:
            raise KeyError(f"missing manifest for {rid}")
        alpha = str(r["alpha"])
        counts = _trial_counts(r, manifests[rid], bank)
        _add_counts(by_alpha.setdefault(alpha, {}), counts)
        if r.get("steered_canonical") == r.get("src_class"):
            _add_counts(by_alpha_reveal_src.setdefault(alpha, {}), counts)
        else:
            _add_counts(by_alpha_not_reveal_src.setdefault(alpha, {}), counts)

    result = {
        "run_dir": str(run_dir),
        "steer_json": str(steer_path),
        "direction_anchor": data.get("direction_anchor"),
        "direction_layer": data.get("direction_layer"),
        "steer_layer": data.get("steer_layer"),
        "by_alpha": {a: _summarize_counts(c) for a, c in sorted(by_alpha.items(), key=lambda x: float(x[0]))},
        "by_alpha_reveal_source": {
            a: _summarize_counts(c)
            for a, c in sorted(by_alpha_reveal_src.items(), key=lambda x: float(x[0]))
        },
        "by_alpha_not_reveal_source": {
            a: _summarize_counts(c)
            for a, c in sorted(by_alpha_not_reveal_src.items(), key=lambda x: float(x[0]))
        },
    }

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))

    print(f"{data.get('direction_anchor')} L{data.get('direction_layer')}")
    print("alpha | diag% | source@diag | target@diag | source@all | target@all")
    for alpha, s in result["by_alpha"].items():
        print(
            f"{float(alpha):>4.1f} | "
            f"{_pct(s['source_target_different_rate']):>6} | "
            f"{_pct(s['match_source_diagnostic_rate']):>11} | "
            f"{_pct(s['match_target_diagnostic_rate']):>11} | "
            f"{_pct(s['match_source_all_rate']):>10} | "
            f"{_pct(s['match_target_all_rate']):>10}"
        )

    print("\nReveal-to-source subset: source@diag / target@diag")
    for alpha in result["by_alpha"]:
        s = result["by_alpha_reveal_source"].get(alpha)
        if s is None:
            print(f"{float(alpha):>4.1f}:    n/a /    n/a")
            continue
        print(
            f"{float(alpha):>4.1f}: "
            f"{_pct(s['match_source_diagnostic_rate'])} / "
            f"{_pct(s['match_target_diagnostic_rate'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
