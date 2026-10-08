# Tests for the final-state grader: reproduces the answer key's final_state column and pins its evidence.

import json
from pathlib import Path

import pytest

from trajecteval import (
    Action,
    Dimension,
    FinalStateGrader,
    Grader,
    Step,
    TaskSpec,
    Trajectory,
    Verdict,
    load_spec,
)

REPO = Path(__file__).resolve().parents[1]
TASKS = REPO / "tasks"
FIXTURES = REPO / "tests" / "fixtures" / "travel_booking"
TABLE = json.loads((FIXTURES / "expected_verdicts.json").read_text(encoding="utf-8"))

GRADER = FinalStateGrader()


def _spec() -> TaskSpec:
    return load_spec("travel_booking", tasks_dir=TASKS)


def _fixture(name: str) -> Trajectory:
    return Trajectory.from_json_file(FIXTURES / f"{name}.json")


def test_grader_satisfies_the_protocol_with_the_right_dimension():
    # The report layer reads .dimension without calling grade().
    assert isinstance(GRADER, Grader)
    assert FinalStateGrader().dimension is Dimension.FINAL_STATE


@pytest.mark.parametrize("name", sorted(TABLE["verdicts"]), ids=sorted(TABLE["verdicts"]))
def test_final_state_matches_the_answer_key(name: str):
    if "final_state" in TABLE["declared_only"]:
        pytest.skip("final_state is declared intent only")
    result = GRADER.grade(_spec(), _fixture(name))
    assert result.verdict.value == TABLE["verdicts"][name]["final_state"]
    assert result.dimension is Dimension.FINAL_STATE


def test_failed_cites_passengers_at_the_last_step():
    # status matches, passengers does not: one citation, the right one,
    # on the step whose state_after was judged (the last).
    result = GRADER.grade(_spec(), _fixture("failed"))
    assert result.verdict is Verdict.FAIL
    assert len(result.evidence) == 1
    evidence = result.evidence[0]
    assert evidence.kind == "final_state_value_mismatch"
    assert evidence.step == 3
    assert evidence.field_name == "passengers"
    assert evidence.value == {"expected": 1, "actual": 2}


def test_clean_passes_with_no_evidence():
    # The false-positive guard: extra fields (searches, flight, cabin,
    # payment_method) are ignored, and a PASS cites nothing.
    result = GRADER.grade(_spec(), _fixture("clean"))
    assert result.verdict is Verdict.PASS
    assert result.evidence == ()


def test_empty_trajectory_fails_whole_trajectory():
    # Zero steps: the finish line was never reached -- step=None, and the
    # evidence shows what was expected (accusation-free, factual).
    result = GRADER.grade(_spec(), _fixture("empty"))
    assert result.verdict is Verdict.FAIL
    evidence = result.evidence[0]
    assert evidence.kind == "final_state_not_reached"
    assert evidence.step is None
    assert evidence.value == {"expected": {"status": "confirmed", "passengers": 1}}


def test_empty_final_state_section_warns_instead_of_passing():
    # A rulebook with no rules must not earn a vacuous PASS (blanket-WARN
    # decision): WARN, kind=empty_section, spec-side (step=None).
    spec = TaskSpec(task_id="travel_booking", expected_final_state={})
    result = GRADER.grade(spec, _fixture("clean"))
    assert result.verdict is Verdict.WARN
    evidence = result.evidence[0]
    assert evidence.kind == "empty_section"
    assert evidence.step is None
    assert evidence.field_name == "expected_final_state"


def test_wrong_pairing_is_error_not_a_judgment():
    result = GRADER.grade(_spec(), Trajectory(task_id="airbnb_booking"))
    assert result.verdict is Verdict.ERROR
    assert result.evidence[0].kind == "task_id_mismatch"


def test_missing_section_is_error_naming_it():
    result = GRADER.grade(TaskSpec(task_id="travel_booking"), _fixture("clean"))
    assert result.verdict is Verdict.ERROR
    evidence = result.evidence[0]
    assert evidence.kind == "missing_spec_field"
    assert evidence.field_name == "expected_final_state"
    assert evidence.step is None
    assert "expected_final_state" in result.reason


def test_pairing_check_runs_before_section_check():
    # Spec has NO sections and the trajectory is for another task: the
    # mismatch must win, proving task_id_mismatch_result was called first.
    result = GRADER.grade(
        TaskSpec(task_id="travel_booking"), Trajectory(task_id="airbnb_booking")
    )
    assert result.evidence[0].kind == "task_id_mismatch"


def test_every_bad_field_is_cited_not_just_the_first():
    spec = TaskSpec(
        task_id="travel_booking",
        expected_final_state={"status": "pending", "passengers": 4, "confirmation": "X1"},
    )
    result = GRADER.grade(spec, _fixture("clean"))
    assert result.verdict is Verdict.FAIL
    assert [(e.field_name, e.kind) for e in result.evidence] == [
        ("status", "final_state_value_mismatch"),
        ("passengers", "final_state_value_mismatch"),
        ("confirmation", "final_state_field_missing"),
    ]


def test_nested_final_state_cites_the_dotted_path():
    trajectory = Trajectory(
        task_id="travel_booking",
        steps=(
            Step(
                index=1,
                state_before={},
                action=Action(name="book_flight"),
                state_after={"trip": {"cabin": "economy"}},
            ),
        ),
    )
    spec = TaskSpec(task_id="travel_booking", expected_final_state={"trip": {"cabin": "business"}})
    result = GRADER.grade(spec, trajectory)
    assert result.verdict is Verdict.FAIL
    evidence = result.evidence[0]
    assert evidence.field_name == "trip.cabin"
    assert evidence.step == 1
    assert evidence.value == {"expected": "business", "actual": "economy"}


def test_grading_does_not_mutate_inputs():
    spec, trajectory = _spec(), _fixture("first_class_overlap")
    spec_before, trajectory_before = spec.to_dict(), trajectory.to_dict()
    GRADER.grade(spec, trajectory)
    assert spec.to_dict() == spec_before
    assert trajectory.to_dict() == trajectory_before
