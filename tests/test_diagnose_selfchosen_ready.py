import json
import importlib.util
from pathlib import Path


def _load_module():
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "diagnose_selfchosen_ready.py"
    spec = importlib.util.spec_from_file_location("diagnose_selfchosen_ready", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


diagnose_selfchosen_ready = _load_module()
_compare_against_persistence = diagnose_selfchosen_ready._compare_against_persistence
_classes_at_quota = diagnose_selfchosen_ready._classes_at_quota
_should_stop_collection = diagnose_selfchosen_ready._should_stop_collection
_summarize_attempt_distribution = diagnose_selfchosen_ready._summarize_attempt_distribution


def _write_persistence_fixture(path: Path, class_ids: list[str]) -> None:
    path.write_text(
        json.dumps(
            {
                "class_ids": class_ids,
                "persistence": {
                    "post13_best_layer_by_a": 13,
                    "nc_loo_state_a_by_layer": [0.0] * 30,
                    "nc_loo_state_b_by_layer": [1.0] * 30,
                    "state_a_within_between": {"contrast_post13": 0.1},
                    "state_b_within_between": {"contrast_post13": 0.2},
                },
            }
        )
    )


def test_compare_against_persistence_skips_mismatched_class_ids(tmp_path: Path):
    persistence_path = tmp_path / "persistence.json"
    _write_persistence_fixture(persistence_path, ["tiger", "eagle", "frog", "salmon"])

    result = _compare_against_persistence(
        ["dolphin", "horse"],
        [0.0] * 30,
        {"contrast_post13": 0.0},
        persistence_path,
    )

    assert result == {
        "persistence_results_path": str(persistence_path),
        "comparison_skipped_reason": (
            "class_ids_mismatch: self_chosen=['dolphin', 'horse'] "
            "persistence=['tiger', 'eagle', 'frog', 'salmon']"
        ),
    }


def test_compare_against_persistence_returns_vote_for_matched_class_ids(tmp_path: Path):
    persistence_path = tmp_path / "persistence.json"
    _write_persistence_fixture(persistence_path, ["dolphin", "horse"])

    result = _compare_against_persistence(
        ["dolphin", "horse"],
        [0.0] * 30,
        {"contrast_post13": 0.19},
        persistence_path,
    )

    assert result == {
        "persistence_results_path": str(persistence_path),
        "reference_layer": 13,
        "ready_nc_at_reference_layer": 0.0,
        "reference_nc": {"state_a": 0.0, "state_b": 1.0},
        "ready_contrast_post13": 0.19,
        "reference_contrast_post13": {"state_a": 0.1, "state_b": 0.2},
        "vote_by_metric": {
            "nc_at_reference_layer": "state_a",
            "contrast_post13": "state_b",
        },
        "overall": "mixed",
    }


def test_compare_against_persistence_accepts_reordered_matched_class_ids(tmp_path: Path):
    persistence_path = tmp_path / "persistence.json"
    _write_persistence_fixture(persistence_path, ["horse", "dolphin"])

    result = _compare_against_persistence(
        ["dolphin", "horse"],
        [0.0] * 30,
        {"contrast_post13": 0.19},
        persistence_path,
    )

    assert result["overall"] == "mixed"
    assert result["reference_nc"] == {"state_a": 0.0, "state_b": 1.0}


def test_classes_at_quota_returns_only_filled_classes():
    counts = {"elephant": 20, "cow": 19, "dog": 20, "horse": 3}

    assert _classes_at_quota(counts, 20) == ["elephant", "dog"]


def test_should_stop_collection_respects_partial_quota_target():
    counts = {"elephant": 20, "cow": 20, "dog": 5, "horse": 0}

    assert _should_stop_collection(
        counts,
        20,
        stop_when_n_classes_hit_quota=2,
    )
    assert not _should_stop_collection(
        counts,
        20,
        stop_when_n_classes_hit_quota=3,
    )


def test_summarize_attempt_distribution_reports_diversity_and_parse_rates():
    rows = [
        {
            "ready_ok": True,
            "reveal_canonical_id": "elephant",
            "n_answer_parsed": 4,
            "n_answer_correct": 4,
        },
        {
            "ready_ok": False,
            "reveal_canonical_id": "cow",
            "n_answer_parsed": 3,
            "n_answer_correct": 2,
        },
        {
            "ready_ok": True,
            "reveal_canonical_id": None,
            "n_answer_parsed": 0,
            "n_answer_correct": 0,
        },
    ]

    summary = _summarize_attempt_distribution(
        rows,
        candidate_ids=["elephant", "cow", "dog"],
        n_questions=4,
    )

    assert summary["counts_by_candidate"] == {"elephant": 1, "cow": 1, "dog": 0}
    assert summary["n_distinct_candidates_parsed"] == 2
    assert summary["ready_parse_success"] == 2 / 3
    assert summary["reveal_parse_success"] == 2 / 3
    assert summary["answer_parse_success"] == 7 / 12
    assert summary["answer_correct_on_parsed"] == 6 / 7
    assert summary["parsed_reveal_top1_share"] == 0.5
    assert summary["parsed_reveal_effective_classes"] == 2.0
