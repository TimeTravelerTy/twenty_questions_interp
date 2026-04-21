import importlib.util
import json
from pathlib import Path

import torch


def _load_module():
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "decode_turns.py"
    spec = importlib.util.spec_from_file_location("decode_turns", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


decode_turns = _load_module()


def test_parse_turn_selector_dedupes_and_preserves_order():
    assert decode_turns.parse_turn_selector("2,1,2,4") == [2, 1, 4]


def test_load_turn_runs_uses_local_fallback_when_manifest_path_is_missing(tmp_path: Path):
    run_root = tmp_path / "diag"
    run_dir = run_root / "attempt_000"
    run_dir.mkdir(parents=True)
    torch.save(torch.ones(3, 4), run_dir / "turn_01_activations.pt")
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "attempt_000",
                "condition": "self_chosen",
                "model_name": "m",
                "model_revision": "r",
                "tokenizer_revision": "t",
                "torch_dtype": "float32",
                "device": "cpu",
                "prompt_template_id": "p",
                "seed": 1,
                "permutation": ["cow", "horse"],
                "reveal_canonical_id": "cow",
                "turn_activation_paths": {
                    "1": "/does/not/exist/turn_01_activations.pt",
                },
            }
        )
    )
    (run_root / "results.json").write_text(
        json.dumps(
            {
                "kept_rows": [
                    {
                        "run_id": "attempt_000",
                    }
                ]
            }
        )
    )

    runs = decode_turns.load_turn_runs(run_root, turn_idx=1, selection="kept")

    assert len(runs) == 1
    manifest, activations = runs[0]
    assert manifest.run_id == "attempt_000"
    assert activations.shape == (3, 4)
