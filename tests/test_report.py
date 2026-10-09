# Tests for evaluate(): the foreman -- same rules for every grader, crashes
# degrade one column, gates run before any judging, no overall verdict.

import json
from pathlib import Path

import pytest

from trajecteval import (
    Action,
    BoundsGrader,
    CriticalGrader,
    Dimension,
    FinalStateGrader,
    GraderResult,
    Report,
    Step,
    TaskSpec,
    Trajectory,
    Verdict,
    evaluate,
    load_spec,
    render_json,
)

REPO = Path(__file__).resolve().parents[1]
TASKS = REPO / "tasks"
FIXTURES = REPO / "tests" / "fixtures" / "travel_booking"
TABLE = json.loads((FIXTURES / "expected_verdicts.json").read_text(encoding="utf-8"))

# The three columns the deterministic graders own. trajectory_quality is
# declared intent only (Step 9's judge), so the answer key checks just these.
DETERMINISTIC = ("final_state", "bounds", "critical")


def _spec() -> TaskSpec:
    return load_spec("travel_booking", tasks_dir=TASKS)


def _fixture(name: str) -> Trajectory:
    return Trajectory.from_json_file(FIXTURES / f"{name}.json")


class _CrashingGrader:
    # A grader whose code raises mid-inspection (a bug, not bad input).

    dimension = Dimension.TRAJECTORY_QUALITY

    def grade(self, spec, trajectory):
        raise ZeroDivisionError("boom")


class _JunkReturningGrader:
    # A grader that returns something that is not a GraderResult at all.

    dimension = Dimension.TRAJECTORY_QUALITY

    def grade(self, spec, trajectory):
        return "PASS"


class _WrongDimensionGrader:
    # A grader whose result claims a column it does not own.

    dimension = Dimension.TRAJECTORY_QUALITY

    def grade(self, spec, trajectory):
        return GraderResult(
            dimension=Dimension.BOUNDS, verdict=Verdict.PASS, reason="wrong column"
        )


class _DimensionlessGrader:
    def grade(self, spec, trajectory):
        return None


class _TypoDimensionGrader:
    dimension = "bound"  # not a Dimension member

    def grade(self, spec, trajectory):
        return None


class _RecordingGrader:
    # Counts grade() calls, proving the ValueError gates run before any judging.

    dimension = Dimension.TRAJECTORY_QUALITY

    def __init__(self):
        self.calls = 0

    def grade(self, spec, trajectory):
        self.calls += 1
        return GraderResult(
            dimension=self.dimension, verdict=Verdict.PASS, reason="fine"
        )


def _quality_by_dimension(report: Report) -> dict[str, GraderResult]:
    return {result.dimension.value: result for result in report.results}


def test_default_graders_cover_the_three_deterministic_columns():
    report = evaluate(_fixture("clean"), _spec())
    assert tuple(r.dimension for r in report.results) == (
        Dimension.FINAL_STATE,
        Dimension.BOUNDS,
        Dimension.CRITICAL,
    )


@pytest.mark.parametrize("name", sorted(TABLE["verdicts"]), ids=sorted(TABLE["verdicts"]))
def test_pipeline_matches_the_answer_key(name: str):
    # Whole-pipeline reproduction: evaluate() over each fixture agrees with
    # expected_verdicts.json on every deterministic column.
    report = evaluate(_fixture(name), _spec())
    results = _quality_by_dimension(report)
    dimensions = [d for d in DETERMINISTIC if d not in TABLE["declared_only"]]
    for dimension in dimensions:
        assert results[dimension].verdict.value == TABLE["verdicts"][name][dimension]


def test_a_crashing_grader_degrades_only_its_own_column():
    # The final-state column is judged normally; the crashed column says so.
    report = evaluate(_fixture("clean"), _spec(), graders=[FinalStateGrader(), _CrashingGrader()])
    results = _quality_by_dimension(report)
    assert results["final_state"].verdict is Verdict.PASS
    crashed = results["trajectory_quality"]
    assert crashed.verdict is Verdict.ERROR
    assert crashed.evidence[0].kind == "grader_crash"
    assert crashed.evidence[0].step is None
    assert "ZeroDivisionError" in crashed.reason


def test_crash_evidence_carries_type_and_message_but_no_traceback():
    report = evaluate(_fixture("clean"), _spec(), graders=[_CrashingGrader()])
    value = report.results[0].evidence[0].value
    assert value == {"exception": "ZeroDivisionError", "message": "boom"}


def test_non_result_return_goes_in_the_crash_lane():
    report = evaluate(_fixture("clean"), _spec(), graders=[_JunkReturningGrader()])
    result = report.results[0]
    assert result.verdict is Verdict.ERROR
    assert result.evidence[0].kind == "grader_crash"
    assert result.evidence[0].value["exception"] == "TypeError"


def test_wrong_dimension_return_goes_in_the_crash_lane():
    report = evaluate(_fixture("clean"), _spec(), graders=[_WrongDimensionGrader()])
    result = report.results[0]
    assert result.verdict is Verdict.ERROR
    assert result.evidence[0].kind == "grader_crash"
    assert result.evidence[0].value["exception"] == "ValueError"
    assert "bounds" in result.evidence[0].value["message"]


def test_empty_grader_list_raises():
    with pytest.raises(ValueError, match="at least one grader"):
        evaluate(_fixture("clean"), _spec(), graders=[])


def test_grader_without_dimension_raises_before_judging():
    recorder = _RecordingGrader()
    with pytest.raises(ValueError, match="dimension"):
        evaluate(_fixture("clean"), _spec(), graders=[recorder, _DimensionlessGrader()])
    assert recorder.calls == 0  # nothing was judged before the gate fired


def test_typo_dimension_raises_before_judging():
    with pytest.raises(ValueError, match="valid dimension"):
        evaluate(_fixture("clean"), _spec(), graders=[_TypoDimensionGrader()])


def test_duplicate_dimension_raises_before_any_grading():
    first, second = _RecordingGrader(), _RecordingGrader()
    with pytest.raises(ValueError, match="duplicate"):
        evaluate(_fixture("clean"), _spec(), graders=[first, second])
    assert first.calls == 0 and second.calls == 0


def test_report_carries_both_task_ids():
    report = evaluate(_fixture("clean"), _spec())
    assert report.task_id == "travel_booking"
    assert report.trajectory_task_id == "travel_booking"


def test_results_follow_the_given_grader_order():
    report = evaluate(
        _fixture("clean"), _spec(), graders=[CriticalGrader(), FinalStateGrader(), BoundsGrader()]
    )
    assert tuple(r.dimension for r in report.results) == (
        Dimension.CRITICAL,
        Dimension.FINAL_STATE,
        Dimension.BOUNDS,
    )


def test_only_the_given_graders_run():
    report = evaluate(_fixture("clean"), _spec(), graders=[FinalStateGrader()])
    assert len(report.results) == 1


def test_discontinuities_land_in_the_report():
    # Graders never inspect discontinuities; the report carries them for the
    # human (Step 2's find_discontinuities, finally seen by someone).
    trajectory = Trajectory(
        task_id="travel_booking",
        steps=(
            Step(index=1, state_before={}, action=Action(name="a"), state_after={"x": 1}),
            Step(index=2, state_before={"x": 2}, action=Action(name="b"), state_after={"x": 2}),
        ),
    )
    report = evaluate(trajectory, _spec(), graders=[FinalStateGrader()])
    assert len(report.discontinuities) == 1
    discontinuity = report.discontinuities[0]
    assert discontinuity.step_index == 1
    assert discontinuity.differing_keys == ("x",)


def test_the_report_dict_has_exactly_the_four_fields():
    # No overall verdict, no surprises: the columns stay visible (FAQ).
    report = evaluate(_fixture("clean"), _spec())
    assert set(report.to_dict()) == {
        "task_id",
        "trajectory_task_id",
        "results",
        "discontinuities",
    }


def test_evaluate_does_not_mutate_inputs():
    spec, trajectory = _spec(), _fixture("first_class_overlap")
    spec_before, trajectory_before = spec.to_dict(), trajectory.to_dict()
    evaluate(trajectory, spec)
    assert spec.to_dict() == spec_before
    assert trajectory.to_dict() == trajectory_before


def test_render_json_is_exactly_the_report_dict():
    report = evaluate(_fixture("failed"), _spec())
    assert json.loads(render_json(report)) == report.to_dict()


def test_render_json_round_trips_through_from_dict():
    # to_dict -> dumps -> loads -> from_dict == the original report.
    report = evaluate(_fixture("failed"), _spec())
    assert Report.from_dict(json.loads(render_json(report))) == report


def test_render_json_is_deterministic():
    # Same report, same bytes: a saved report can be diffed later (Step 8).
    report = evaluate(_fixture("clean"), _spec())
    assert render_json(report) == render_json(report)
