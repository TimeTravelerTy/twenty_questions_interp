import json
import importlib.util
from pathlib import Path


def _load_compare_fn():
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "diagnose_selfchosen_ready.py"
    spec = importlib.util.spec_from_file_location("diagnose_selfchosen_ready", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._compare_against_persistence


_compare_against_persistence = _load_compare_fn()


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
