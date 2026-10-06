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
    task_id_mismatch_result,
)

TRAJ = Trajectory(task_id="travel_booking")
OTHER_TRAJ = Trajectory(task_id="airbnb_booking")


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


def test_task_id_match_returns_none():
    spec = TaskSpec(task_id="travel_booking")
    assert task_id_mismatch_result(spec, Dimension.BOUNDS, TRAJ) is None


def test_task_id_mismatch_is_diagnostic_error():
    # Wrong pairing: ERROR, reason names BOTH ids, evidence points at the
    # spec side (step=None) -- a diagnostic, never an accusation against
    # the trajectory, which may be perfectly fine on its own.
    spec = TaskSpec(task_id="travel_booking")
    result = task_id_mismatch_result(spec, Dimension.BOUNDS, OTHER_TRAJ)
    assert result is not None
    assert result.verdict is Verdict.ERROR
    assert "'travel_booking'" in result.reason
    assert "'airbnb_booking'" in result.reason
    assert result.dimension is Dimension.BOUNDS  # attributed to the CALLER's column
    evidence = result.evidence[0]
    assert evidence.kind == "task_id_mismatch"
    assert evidence.step is None
    assert evidence.field_name == "task_id"
    assert evidence.value == {"spec": "travel_booking", "trajectory": "airbnb_booking"}


def test_mismatch_check_comes_before_section_check():
    # Documented call order, made executable: a wrong pairing must not be
    # masked by an unrelated missing section (spec here has no
    # allowed_actions AND is paired with the wrong trajectory).
    class _OrderedGrader:
        dimension = Dimension.BOUNDS

        def grade(self, spec: TaskSpec, trajectory: Trajectory) -> GraderResult:
            return (
                task_id_mismatch_result(spec, self.dimension, trajectory)
                or missing_section_result(spec, self.dimension, "allowed_actions")
                or GraderResult(
                    dimension=self.dimension,
                    verdict=Verdict.PASS,
                    reason="stub: paired and complete",
                )
            )

    result = _OrderedGrader().grade(TaskSpec(task_id="travel_booking"), OTHER_TRAJ)
    assert result.verdict is Verdict.ERROR
    assert result.evidence[0].kind == "task_id_mismatch"


def test_missing_section_rejects_unknown_section_names():
    spec = TaskSpec(task_id="t")
    # "allowed_action" is a typo; "task_id" is the dangerous one -- a real,
    # non-None attribute that would silently report "present".
    for bad in ("allowed_action", "task_id", "__class__"):
        with pytest.raises(ValueError, match="invalid section"):
            missing_section_result(spec, Dimension.BOUNDS, bad)


def test_missing_section_rejects_non_string_section():
    spec = TaskSpec(task_id="t")
    with pytest.raises(ValueError, match="invalid section None"):
        missing_section_result(spec, Dimension.BOUNDS, None)


def test_missing_section_accepts_all_three_rule_sections():
    spec = TaskSpec(task_id="t")
    for section, dimension in (
        ("expected_final_state", Dimension.FINAL_STATE),
        ("allowed_actions", Dimension.BOUNDS),
        ("critical_error_patterns", Dimension.CRITICAL),
    ):
        assert missing_section_result(spec, dimension, section) is not None


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
