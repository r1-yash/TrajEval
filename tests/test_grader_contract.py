"""Tests for the grader contract: the protocol, and the can't-judge path.

The stub graders here are *test doubles*, not real graders (those are Step 6).
They pin the two promises this step makes: any object with a ``grade`` method
qualifies, and a spec missing its section yields an ERROR verdict with
diagnostic evidence -- never an exception.
"""

import pytest

from trajecteval import (
    Dimension,
    Grader,
    GraderResult,
    TaskSpec,
    Trajectory,
    Verdict,
    missing_section_result,
)

TRAJ = Trajectory(task_id="travel_booking")


class _StubBoundsGrader:
    """Stands in for the Step 6 bounds grader: section first, then judge."""

    dimension = Dimension.BOUNDS

    def grade(self, spec: TaskSpec, trajectory: Trajectory) -> GraderResult:
        blocked = missing_section_result(spec, self.dimension, "allowed_actions")
        if blocked is not None:
            return blocked
        return GraderResult(
            dimension=self.dimension,
            verdict=Verdict.PASS,
            reason="stub: section present, nothing else checked yet",
        )


class _NoDimensionGrader:
    """Has grade() but no dimension -- must NOT qualify as a Grader."""

    def grade(self, spec: TaskSpec, trajectory: Trajectory) -> GraderResult:
        raise AssertionError("never called")


def test_grader_protocol_requires_dimension():
    # The report layer reads grader.dimension *without* calling grade (it
    # needs the column even when grade crashes), so dimension is required.
    assert isinstance(_StubBoundsGrader(), Grader)
    assert _StubBoundsGrader().dimension is Dimension.BOUNDS
    assert not isinstance(_NoDimensionGrader(), Grader)


def test_protocol_accepts_any_grade_object():
    # runtime_checkable isinstance checks the *shape* (attributes present);
    # the argument types are the type checker's job, not isinstance's.
    assert isinstance(_StubBoundsGrader(), Grader)
    assert not isinstance(object(), Grader)
    assert not isinstance({}, Grader)


def test_stub_grader_fills_out_the_report_card():
    result = _StubBoundsGrader().grade(
        TaskSpec(task_id="travel_booking", allowed_actions={"search_flights": {}}), TRAJ
    )
    assert isinstance(result, GraderResult)
    assert result.verdict is Verdict.PASS


def test_missing_section_returns_none_when_present():
    spec = TaskSpec(task_id="t", allowed_actions={})
    assert missing_section_result(spec, Dimension.BOUNDS, "allowed_actions") is None


def test_missing_section_is_error_not_crash():
    # The adopted rule: a spec without 'allowed_actions' -> the bounds grader
    # returns ERROR, with evidence pointing at the spec (step=None), not at
    # the trajectory -- a diagnostic, never an accusation.
    result = missing_section_result(TaskSpec(task_id="t"), Dimension.BOUNDS, "allowed_actions")
    assert result is not None
    assert result.verdict is Verdict.ERROR
    assert "allowed_actions" in result.reason
    evidence = result.evidence[0]
    assert evidence.kind == "missing_spec_field"
    assert evidence.field_name == "allowed_actions"
    assert evidence.step is None


def test_stub_grader_reports_absent_section_as_error():
    result = _StubBoundsGrader().grade(TaskSpec(task_id="travel_booking"), TRAJ)
    assert result.verdict is Verdict.ERROR
    assert result.evidence[0].field_name == "allowed_actions"


def test_every_verdict_from_stub_still_obeys_result_invariants():
    # Contract results are ordinary GraderResults: reason and dimension live,
    # so Step 7 can render them with no special cases.
    result = missing_section_result(TaskSpec(task_id="t"), Dimension.BOUNDS, "allowed_actions")
    assert result.reason.strip()
    assert result.dimension is Dimension.BOUNDS
    with pytest.raises(ValueError, match="must cite at least one evidence"):
        GraderResult(
            dimension=Dimension.BOUNDS, verdict=Verdict.FAIL, reason="no evidence"
        )
