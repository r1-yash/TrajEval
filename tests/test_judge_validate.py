# Tests for judge output validation: every rule that turns a model answer
# into a JudgeResult -- and every answer that must instead raise
# JudgeOutputError (no guessing, no silent repair, no retries).

import json

import pytest

from trajecteval import Trajectory
from trajecteval.judge.validate import JudgeOutputError, parse_judge_output

STEPS = 3


def _trajectory(steps: int = STEPS) -> Trajectory:
    return Trajectory.from_dict(
        {
            "task_id": "t",
            "steps": [
                {
                    "index": i,
                    "state_before": {},
                    "action": {"name": "do"},
                    "state_after": {},
                }
                for i in range(1, steps + 1)
            ],
        }
    )


def _output(**overrides) -> str:
    # A well-formed answer, with any field overridable -- and removable by
    # passing None for notes or by using _raw_output for key deletion.
    data = {"verdict": "pass", "reason": "clean run", "score": 0.9, "notes": []}
    data.update(overrides)
    return json.dumps(data)


def _raw_output(**changes) -> str:
    # Same, but a None value deletes the key (for missing-key cases).
    data = {"verdict": "pass", "reason": "clean run", "score": 0.9, "notes": []}
    for key, value in changes.items():
        if value is None:
            data.pop(key, None)
        else:
            data[key] = value
    return json.dumps(data)


# ------------------------------------------------------------------ happy path

def test_valid_pass_with_no_notes():
    result = parse_judge_output(_output(), _trajectory())
    assert result.verdict.value == "PASS"
    assert result.reason == "clean run"
    assert result.notes == ()


def test_valid_warn_with_a_whole_episode_note():
    raw = _output(
        verdict="warn",
        notes=[{"step": None, "kind": "thrash", "detail": "slow overall"}],
    )
    result = parse_judge_output(raw, _trajectory())
    assert result.verdict.value == "WARN"
    note = result.notes[0]
    assert note.step is None
    assert note.kind == "thrash"


def test_valid_fail_citing_a_real_step():
    raw = _output(
        verdict="fail",
        reason="step 2 repeats step 1",
        notes=[{"step": 2, "kind": "repeated_action", "detail": "same search twice"}],
    )
    result = parse_judge_output(raw, _trajectory())
    assert result.verdict.value == "FAIL"
    assert result.notes[0].step == 2


def test_last_step_number_is_valid():
    raw = _output(
        verdict="fail",
        notes=[{"step": STEPS, "kind": "detour", "detail": "wasted end"}],
    )
    assert parse_judge_output(raw, _trajectory()).notes[0].step == STEPS


# ------------------------------------------------------------- shape violations

def test_invalid_json_raises_with_the_json_error():
    with pytest.raises(JudgeOutputError, match="not valid JSON"):
        parse_judge_output("{not json", _trajectory())


def test_markdown_fenced_json_is_not_repaired():
    # No fence-stripping: the prompt demands bare JSON, and silently
    # unwrapping fences is the first step toward silently "fixing" answers.
    raw = '```json\n{"verdict": "pass", "reason": "ok", "notes": []}\n```'
    with pytest.raises(JudgeOutputError, match="not valid JSON"):
        parse_judge_output(raw, _trajectory())


def test_empty_text_raises():
    with pytest.raises(JudgeOutputError, match="no text"):
        parse_judge_output("   ", _trajectory())


def test_top_level_must_be_an_object():
    with pytest.raises(JudgeOutputError, match="must be a JSON object"):
        parse_judge_output('"pass"', _trajectory())


def test_missing_top_level_key_raises_naming_it():
    with pytest.raises(JudgeOutputError, match="missing notes"):
        parse_judge_output(_raw_output(notes=None), _trajectory())


def test_missing_score_raises_naming_it():
    # The score is part of the closed shape; an answer without one is
    # unusable, not something to default.
    with pytest.raises(JudgeOutputError, match="missing score"):
        parse_judge_output(_raw_output(score=None), _trajectory())


def test_unknown_top_level_key_raises_naming_it():
    # "confidence" is not in the closed shape; the score, by contrast, is.
    raw = _output(confidence=0.9)
    with pytest.raises(JudgeOutputError, match="unknown confidence"):
        parse_judge_output(raw, _trajectory())


# ------------------------------------------------------------- score rules

@pytest.mark.parametrize("score", [0, 1, 0.5, 0.9, 1.0])
def test_valid_scores_are_kept(score):
    result = parse_judge_output(_output(score=score), _trajectory())
    assert result.score == score


def test_score_is_normalized_to_float():
    assert parse_judge_output(_output(score=1), _trajectory()).score == 1.0


@pytest.mark.parametrize("score", [1.01, -0.1, True, "0.9", None, [0.9]])
def test_bad_scores_make_the_answer_unusable(score):
    # No clamping, no guessing: a bool is not a number here either, and a
    # missing key (None) is caught by the missing-score rule.
    with pytest.raises(JudgeOutputError, match="'score'"):
        parse_judge_output(_output(score=score), _trajectory())


# ------------------------------------------------------------ verdict violations

def test_verdict_is_case_sensitive():
    # "PASS" is an unknown verdict, not something to fold to pass.
    with pytest.raises(JudgeOutputError, match="unknown verdict 'PASS'"):
        parse_judge_output(_output(verdict="PASS"), _trajectory())


def test_unknown_verdict_raises():
    with pytest.raises(JudgeOutputError, match="unknown verdict 'excellent'"):
        parse_judge_output(_output(verdict="excellent"), _trajectory())


def test_non_string_verdict_raises():
    with pytest.raises(JudgeOutputError, match="unknown verdict 1"):
        parse_judge_output(_output(verdict=1), _trajectory())


def test_reason_must_be_a_non_empty_string():
    with pytest.raises(JudgeOutputError, match="'reason'"):
        parse_judge_output(_output(reason="  "), _trajectory())
    with pytest.raises(JudgeOutputError, match="'reason'"):
        parse_judge_output(_output(reason=7), _trajectory())


# ------------------------------------------------------------ note violations

def test_notes_must_be_a_list():
    with pytest.raises(JudgeOutputError, match="'notes' must be a list"):
        parse_judge_output(_output(notes={}), _trajectory())


def test_note_must_be_an_object():
    with pytest.raises(JudgeOutputError, match=r"notes\[0\] must be an object"):
        parse_judge_output(_output(notes=[2]), _trajectory())


def test_note_keys_are_closed():
    with pytest.raises(JudgeOutputError, match=r"notes\[0\] keys are wrong"):
        parse_judge_output(_output(notes=[{"step": 1, "kind": "x"}]), _trajectory())
    with pytest.raises(JudgeOutputError, match="unknown confidence"):
        parse_judge_output(
            _output(
                notes=[{"step": 1, "kind": "x", "detail": "d", "confidence": 0.5}]
            ),
            _trajectory(),
        )


def test_note_step_type_is_checked():
    for bad in ("2", 2.0, True):
        with pytest.raises(JudgeOutputError, match=r"notes\[0\]\.step"):
            parse_judge_output(
                _output(notes=[{"step": bad, "kind": "x", "detail": "d"}]),
                _trajectory(),
            )


def test_note_step_zero_is_rejected_with_bad_step_recorded():
    raw = _output(notes=[{"step": 0, "kind": "x", "detail": "d"}])
    with pytest.raises(JudgeOutputError, match="1-based") as excinfo:
        parse_judge_output(raw, _trajectory())
    assert excinfo.value.bad_step == 0


def test_note_kind_and_detail_must_be_non_empty():
    with pytest.raises(JudgeOutputError, match=r"notes\[0\]\.kind"):
        parse_judge_output(
            _output(notes=[{"step": 1, "kind": "", "detail": "d"}]), _trajectory()
        )
    with pytest.raises(JudgeOutputError, match=r"notes\[0\]\.detail"):
        parse_judge_output(
            _output(notes=[{"step": 1, "kind": "x", "detail": 3}]), _trajectory()
        )


# ------------------------------------------------------------ step existence

def test_note_citing_a_nonexistent_step_raises_naming_both_numbers():
    raw = _output(
        verdict="fail",
        notes=[{"step": 7, "kind": "detour", "detail": "ghost step"}],
    )
    with pytest.raises(JudgeOutputError, match="step 7 .* 3 step") as excinfo:
        parse_judge_output(raw, _trajectory())
    assert excinfo.value.bad_step == 7


def test_bad_step_defaults_to_none():
    with pytest.raises(JudgeOutputError) as excinfo:
        parse_judge_output("{", _trajectory())
    assert excinfo.value.bad_step is None


# ----------------------------------------------------------- verdict discipline

def test_warn_with_no_notes_raises():
    with pytest.raises(JudgeOutputError, match="'warn' verdict needs"):
        parse_judge_output(_output(verdict="warn"), _trajectory())


def test_fail_with_only_whole_episode_notes_raises():
    # A FAIL nothing can point at is unusable -- ERROR, not a kept verdict.
    raw = _output(
        verdict="fail", notes=[{"step": None, "kind": "vibe", "detail": "felt slow"}]
    )
    with pytest.raises(JudgeOutputError, match="'fail' verdict needs"):
        parse_judge_output(raw, _trajectory())


def test_fail_with_one_citing_note_and_extra_stepless_notes_is_fine():
    raw = _output(
        verdict="fail",
        notes=[
            {"step": 1, "kind": "repeated_action", "detail": "loop start"},
            {"step": None, "kind": "thrash", "detail": "whole episode slow"},
        ],
    )
    assert parse_judge_output(raw, _trajectory()).verdict.value == "FAIL"


# ------------------------------------------------------------------- excerpt

def test_error_carries_the_start_of_the_raw_answer():
    raw = "x" * 500
    with pytest.raises(JudgeOutputError) as excinfo:
        parse_judge_output(raw, _trajectory())
    assert excinfo.value.output_excerpt == "x" * 200


def test_error_excerpt_is_the_original_text_even_when_stripped():
    raw = '  {"verdict": "pass"'
    with pytest.raises(JudgeOutputError) as excinfo:
        parse_judge_output(raw, _trajectory())
    assert excinfo.value.output_excerpt == raw[:200]
