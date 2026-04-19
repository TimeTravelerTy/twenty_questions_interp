from twenty_q.binding_audit import apply_overrides_to_rows, parse_override, summarize_rows


def test_parse_override_accepts_boolish_values():
    assert parse_override("frog.has_four_legs=0") == ("frog", "has_four_legs", False)
    assert parse_override("eagle.can_swim=true") == ("eagle", "can_swim", True)


def test_apply_overrides_and_summary_recompute():
    rows = [
        {
            "run_id": "name_paraphrase_frog_00",
            "cid": "frog",
            "ready_ok": True,
            "answers": [
                {
                    "qid": "has_four_legs",
                    "raw": "No",
                    "parsed": False,
                    "bank": True,
                    "correct": False,
                },
                {
                    "qid": "can_swim",
                    "raw": "Yes",
                    "parsed": True,
                    "bank": True,
                    "correct": True,
                },
            ],
        }
    ]
    rescored, changes = apply_overrides_to_rows(rows, {("frog", "has_four_legs"): False})
    assert len(changes) == 1
    assert changes[0]["old_correct"] is False
    assert changes[0]["new_correct"] is True

    summary = summarize_rows(rescored, ("has_four_legs",))
    assert summary["primary"]["n_correct"] == 1
    assert summary["primary"]["n_questions_total"] == 1
    assert summary["secondary"]["n_correct"] == 1
    assert summary["secondary"]["n_questions_total"] == 1
