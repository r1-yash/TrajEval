# Tests for TrajectoryQualityGrader (the adapter) and ScriptedJudge (the
# test double): the Judge contract folded into the Grader contract, the
# can't-judge lanes, and the proof that an empty trajectory never costs a
# judge call.

import json

import pytest

from trajecteval import (
    Dimension,
    GraderResult,
    JudgeNote,
    JudgeResult,
    LLMClientError,
    ScriptedJudge,
    TaskSpec,
    Trajectory,
    TrajectoryQualityGrader,
    TrajectoryTooLongError,
    Verdict,
    task_id_mismatch_result,
)
from trajecteval.judge.validate import JudgeOutputError


def _spec(task_id: str = "travel_booking") -> TaskSpec:
    return TaskSpec(task_id=task_id)


def _trajectory(steps: int = 3, *, task_id: str = "travel_booking") -> Trajectory:
    return Trajectory.from_dict(
        {
            "task_id": task_id,
            "steps": [
                {
                    "index": i,
                    "state_before": {},
                    "action": {"name": "do", "args": {"n": i}},
                    "state_after": {},
                }
                for i in range(1, steps + 1)
            ],
        }
    )


def _judge(*results: JudgeResult, raises: Exception | None = None) -> ScriptedJudge:
    return ScriptedJudge(results or None, raises=raises)


# ------------------------------------------------------------- conversion

def test_judge_result_becomes_a_grader_result():
    scripted = _judge(
        JudgeResult(
            verdict=Verdict.FAIL,
            reason="step 2 repeats step 1",
            notes=(
                JudgeNote(step=2, kind="repeated_action", detail="same search"),
                JudgeNote(step=None, kind="thrash", detail="overall slow"),
            ),
        )
    )
    result = TrajectoryQualityGrader(scripted).grade(_spec(), _trajectory())
    assert isinstance(result, GraderResult)
    assert result.dimension is Dimension.TRAJECTORY_QUALITY
    assert result.verdict is Verdict.FAIL
    assert result.reason == "step 2 repeats step 1"
    assert len(result.evidence) == 2
    cited = result.evidence[0]
    assert cited.kind == "repeated_action"
    assert cited.step == 2
    assert cited.value == "same search"
    whole = result.evidence[1]
    assert whole.step is None


def test_pass_with_no_notes_yields_no_evidence():
    result = TrajectoryQualityGrader(_judge()).grade(_spec(), _trajectory())
    assert result.verdict is Verdict.PASS
    assert result.evidence == ()


def test_score_becomes_a_whole_episode_evidence_item():
    # The score is secondary and display-only, but it still travels as
    # typed evidence -- step None, the number itself in value.
    scripted = _judge(
        JudgeResult(
            verdict=Verdict.WARN,
            reason="one redundant step",
            notes=(JudgeNote(step=2, kind="thrash", detail="repeated search"),),
            score=0.72,
        )
    )
    result = TrajectoryQualityGrader(scripted).grade(_spec(), _trajectory())
    assert result.verdict is Verdict.WARN
    score_item = result.evidence[-1]
    assert score_item.kind == "quality_score"
    assert score_item.step is None
    assert score_item.value == 0.72


def test_pass_may_carry_the_score_as_evidence():
    # results.py demands evidence only for FAIL/WARN; a PASS with a score
    # must survive construction (this is the score's main lane).
    scripted = _judge(JudgeResult(verdict=Verdict.PASS, reason="clean", score=0.95))
    result = TrajectoryQualityGrader(scripted).grade(_spec(), _trajectory())
    assert result.verdict is Verdict.PASS
    assert [item.kind for item in result.evidence] == ["quality_score"]


def test_no_score_means_no_score_evidence():
    # A judge that returns no score (the scripted default) adds nothing.
    result = TrajectoryQualityGrader(_judge()).grade(_spec(), _trajectory())
    assert all(item.kind != "quality_score" for item in result.evidence)


def test_judge_result_score_must_be_in_range():
    with pytest.raises(ValueError, match="score"):
        JudgeResult(verdict=Verdict.PASS, reason="clean", score=1.5)


def test_warn_keeps_its_note_as_evidence():
    scripted = _judge(
        JudgeResult(
            verdict=Verdict.WARN,
            reason="one redundant step",
            notes=(JudgeNote(step=1, kind="repeated_action", detail="retry"),),
        )
    )
    result = TrajectoryQualityGrader(scripted).grade(_spec(), _trajectory())
    assert result.verdict is Verdict.WARN
    assert result.evidence[0].step == 1


# ------------------------------------------------- ordering: mismatch first

def test_task_id_mismatch_short_circuits_before_any_judge_call():
    scripted = _judge()
    result = TrajectoryQualityGrader(scripted).grade(
        _spec("travel_booking"), _trajectory(task_id="airbnb_booking")
    )
    assert result.verdict is Verdict.ERROR
    assert result.evidence[0].kind == "task_id_mismatch"
    assert scripted.calls == []


def test_wrong_pairing_beats_a_judge_that_would_raise():
    # Pinned order: the diagnostic never depends on judge health.
    scripted = _judge(raises=RuntimeError("judge is down"))
    result = TrajectoryQualityGrader(scripted).grade(
        _spec("travel_booking"), _trajectory(task_id="airbnb_booking")
    )
    assert result.verdict is Verdict.ERROR
    assert scripted.calls == []


# -------------------------------------------------- empty: zero judge calls

def test_empty_trajectory_warns_without_calling_the_judge():
    scripted = _judge()
    result = TrajectoryQualityGrader(scripted).grade(
        _spec(), Trajectory(task_id="travel_booking")
    )
    assert result.verdict is Verdict.WARN
    assert result.evidence[0].kind == "empty_trajectory"
    assert result.evidence[0].step is None
    assert scripted.calls == []  # a paid model is never called here


# ------------------------------------------------------- can't-judge lane

def test_unusable_output_becomes_error_naming_the_bad_step():
    scripted = _judge(
        raises=JudgeOutputError(
            "note cites step 9 but the trajectory has 3 step(s)",
            bad_step=9,
            output_excerpt='{"verdict": "fail"',
        )
    )
    result = TrajectoryQualityGrader(scripted).grade(_spec(), _trajectory())
    assert result.verdict is Verdict.ERROR
    assert "unusable output" in result.reason
    evidence = result.evidence[0]
    assert evidence.kind == "judge_invalid_output"
    assert evidence.step is None
    assert evidence.value["bad_step"] == 9
    assert evidence.value["output_excerpt"] == '{"verdict": "fail"'
    assert "step 9" in evidence.value["problem"]


def test_unusable_output_without_excerpt_omits_that_key():
    scripted = _judge(raises=JudgeOutputError("not valid JSON"))
    result = TrajectoryQualityGrader(scripted).grade(_spec(), _trajectory())
    value = result.evidence[0].value
    assert value == {"problem": "not valid JSON"}
    assert "output_excerpt" not in value


# --------------------------------------------------- judge-unavailable lane

def test_client_error_becomes_error_judge_unavailable():
    # Network/auth/rate-limit/timeout is a setup problem, NOT a bug: ERROR
    # with status + capped provider detail as evidence, never grader_crash.
    scripted = _judge(
        raises=LLMClientError(
            "judge request failed: HTTP 401 from https://api.groq.com/openai/v1: invalid key",
            status=401,
            detail="invalid key",
        )
    )
    result = TrajectoryQualityGrader(scripted).grade(_spec(), _trajectory())
    assert result.verdict is Verdict.ERROR
    assert "could not be reached" in result.reason
    evidence = result.evidence[0]
    assert evidence.kind == "judge_unavailable"
    assert evidence.step is None
    assert evidence.value["status"] == 401
    assert evidence.value["detail"] == "invalid key"
    assert "HTTP 401" in evidence.value["problem"]


def test_client_error_evidence_never_carries_a_key():
    # The client keeps the key in the Authorization header only; the
    # evidence is built from problem/status/detail, none of which hold it.
    scripted = _judge(
        raises=LLMClientError(
            "judge request failed: HTTP 401", status=401, detail="invalid key"
        )
    )
    result = TrajectoryQualityGrader(scripted).grade(_spec(), _trajectory())
    assert "sk-" not in json.dumps(result.evidence[0].value)


def test_too_long_trajectory_becomes_error_with_size_and_limit():
    scripted = _judge(raises=TrajectoryTooLongError(size=30_000, limit=24_000))
    result = TrajectoryQualityGrader(scripted).grade(_spec(), _trajectory())
    assert result.verdict is Verdict.ERROR
    assert "trajectory too long to judge" in result.reason
    assert "30000" in result.reason and "24000" in result.reason
    evidence = result.evidence[0]
    assert evidence.kind == "trajectory_too_long"
    assert evidence.value == {"size": 30_000, "limit": 24_000}


# ------------------------------------------------------ crash lane (genuine bugs)

def test_genuine_judge_bug_still_propagates_to_the_crash_lane():
    # Not every failure is a lane above: a real bug (here: a plain
    # RuntimeError) keeps its type so evaluate() reports "crashed: ...".
    scripted = _judge(raises=RuntimeError("connection reset"))
    with pytest.raises(RuntimeError, match="connection reset"):
        TrajectoryQualityGrader(scripted).grade(_spec(), _trajectory())


def test_judge_returning_junk_raises_instead_of_guessing():
    class _JunkJudge:
        def judge(self, spec, trajectory):
            return "PASS"

    with pytest.raises(TypeError, match="expected JudgeResult"):
        TrajectoryQualityGrader(_JunkJudge()).grade(_spec(), _trajectory())


# ------------------------------------------------------------ ScriptedJudge

def test_scripted_judge_counts_calls_with_task_and_size():
    scripted = _judge(
        JudgeResult(verdict=Verdict.WARN, reason="mild", notes=(JudgeNote(1, "detour", "x"),)),
        JudgeResult(verdict=Verdict.PASS, reason="clean"),
    )
    grader = TrajectoryQualityGrader(scripted)
    grader.grade(_spec(), _trajectory(steps=3))
    grader.grade(_spec(), _trajectory(steps=5))
    assert scripted.calls == [("travel_booking", 3), ("travel_booking", 5)]


def test_scripted_judge_pops_then_repeats_the_last_result():
    scripted = _judge(
        JudgeResult(verdict=Verdict.WARN, reason="first", notes=(JudgeNote(1, "detour", "x"),)),
        JudgeResult(verdict=Verdict.FAIL, reason="second", notes=(JudgeNote(2, "thrash", "y"),)),
    )
    grader = TrajectoryQualityGrader(scripted)
    assert grader.grade(_spec(), _trajectory()).reason == "first"
    assert grader.grade(_spec(), _trajectory()).reason == "second"
    assert grader.grade(_spec(), _trajectory()).reason == "second"


def test_scripted_judge_rejects_an_empty_preset_list():
    with pytest.raises(ValueError, match="at least one"):
        ScriptedJudge([])


def test_scripted_judge_satisfies_the_judge_protocol():
    from trajecteval import Judge

    assert isinstance(_judge(), Judge)
