# Tests for the critical grader: reproduces the answer key's critical column; edges built in code.

import json
from pathlib import Path

import pytest

from trajecteval import (
    Action,
    CriticalGrader,
    CriticalPattern,
    Dimension,
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

GRADER = CriticalGrader()


def _spec() -> TaskSpec:
    return load_spec("travel_booking", tasks_dir=TASKS)


def _fixture(name: str) -> Trajectory:
    return Trajectory.from_json_file(FIXTURES / f"{name}.json")


def _trajectory(*actions: tuple[str, dict]) -> Trajectory:
    # Build a step-continuous trajectory from (name, args) pairs.
    steps = []
    state: dict = {}
    for position, (name, args) in enumerate(actions, start=1):
        steps.append(
            Step(
                index=position,
                state_before=dict(state),
                action=Action(name=name, args=args),
                state_after=dict(state),
            )
        )
    return Trajectory(task_id="travel_booking", steps=tuple(steps))


def test_grader_satisfies_the_protocol_with_the_right_dimension():
    assert isinstance(GRADER, Grader)
    assert CriticalGrader().dimension is Dimension.CRITICAL


@pytest.mark.parametrize("name", sorted(TABLE["verdicts"]), ids=sorted(TABLE["verdicts"]))
def test_critical_matches_the_answer_key(name: str):
    if "critical" in TABLE["declared_only"]:
        pytest.skip("critical is declared intent only")
    result = GRADER.grade(_spec(), _fixture(name))
    assert result.verdict.value == TABLE["verdicts"][name]["critical"]
    assert result.dimension is Dimension.CRITICAL


def test_critical_recovered_cites_only_the_points_payment():
    # Steps 4-5 undo the mistake inside the rules -- the scan still flags
    # step 3 and nothing else: recovery is sticky, not amnesiac.
    result = GRADER.grade(_spec(), _fixture("critical_recovered"))
    assert result.verdict is Verdict.FAIL
    assert len(result.evidence) == 1
    evidence = result.evidence[0]
    assert evidence.kind == "critical_error"
    assert evidence.step == 3
    assert evidence.field_name == "pay"
    assert evidence.value == {"action": "pay", "args": {"method": "points"}}
    assert "description" not in evidence.value  # travel spec ships none (TBD by hand)


def test_first_class_overlap_cites_the_first_class_booking():
    result = GRADER.grade(_spec(), _fixture("first_class_overlap"))
    assert result.verdict is Verdict.FAIL
    assert len(result.evidence) == 1
    evidence = result.evidence[0]
    assert evidence.kind == "critical_error"
    assert evidence.step == 2
    assert evidence.field_name == "book_flight"
    assert evidence.value == {"action": "book_flight", "args": {"cabin": "first"}}


def test_clean_passes_with_no_evidence():
    result = GRADER.grade(_spec(), _fixture("clean"))
    assert result.verdict is Verdict.PASS
    assert result.evidence == ()


def test_empty_trajectory_warns_that_nothing_was_checked():
    result = GRADER.grade(_spec(), _fixture("empty"))
    assert result.verdict is Verdict.WARN
    evidence = result.evidence[0]
    assert evidence.kind == "empty_trajectory"
    assert evidence.step is None
    assert "no patterns were checked" in result.reason


def test_empty_critical_patterns_warns_instead_of_passing():
    # Blanket-WARN decision: a rulebook with no red lines earns no PASS.
    spec = TaskSpec(task_id="travel_booking", critical_error_patterns=())
    result = GRADER.grade(spec, _fixture("clean"))
    assert result.verdict is Verdict.WARN
    evidence = result.evidence[0]
    assert evidence.kind == "empty_section"
    assert evidence.step is None
    assert evidence.field_name == "critical_error_patterns"


def test_wrong_pairing_is_error_not_a_judgment():
    result = GRADER.grade(_spec(), Trajectory(task_id="airbnb_booking"))
    assert result.verdict is Verdict.ERROR
    assert result.evidence[0].kind == "task_id_mismatch"


def test_missing_section_is_error_naming_it():
    result = GRADER.grade(TaskSpec(task_id="travel_booking"), _fixture("clean"))
    assert result.verdict is Verdict.ERROR
    evidence = result.evidence[0]
    assert evidence.kind == "missing_spec_field"
    assert evidence.field_name == "critical_error_patterns"
    assert evidence.step is None
    assert "critical_error_patterns" in result.reason


def test_pairing_check_runs_before_section_check():
    # No sections AND a foreign trajectory: mismatch must win, proving
    # task_id_mismatch_result ran before missing_section_result.
    result = GRADER.grade(
        TaskSpec(task_id="travel_booking"), Trajectory(task_id="airbnb_booking")
    )
    assert result.evidence[0].kind == "task_id_mismatch"


def test_two_matches_in_one_trajectory_report_both():
    spec = TaskSpec(
        task_id="travel_booking",
        critical_error_patterns=(
            CriticalPattern(action="pay", args={"method": "points"}),
            CriticalPattern(action="book_flight", args={"cabin": "first"}),
        ),
    )
    trajectory = _trajectory(
        ("pay", {"method": "points"}),
        ("book_flight", {"cabin": "first"}),
    )
    result = GRADER.grade(spec, trajectory)
    assert result.verdict is Verdict.FAIL
    assert [(e.kind, e.step, e.field_name) for e in result.evidence] == [
        ("critical_error", 1, "pay"),
        ("critical_error", 2, "book_flight"),
    ]


def test_pattern_without_args_flags_every_call_of_its_action():
    # Omitted args = any call of the action is the forbidden move; other
    # actions are untouched, and both calls are cited.
    spec = TaskSpec(
        task_id="travel_booking",
        critical_error_patterns=(CriticalPattern(action="pay"),),
    )
    trajectory = _trajectory(
        ("search_flights", {"origin": "SFO"}),
        ("pay", {"method": "card"}),
        ("pay", {"method": "cash"}),
    )
    result = GRADER.grade(spec, trajectory)
    assert result.verdict is Verdict.FAIL
    assert [e.step for e in result.evidence] == [2, 3]


def test_description_rides_in_evidence_and_reason_when_present():
    spec = TaskSpec(
        task_id="travel_booking",
        critical_error_patterns=(
            CriticalPattern(
                action="pay",
                args={"method": "points"},
                description="burning points on a paid route",
            ),
        ),
    )
    result = GRADER.grade(spec, _trajectory(("pay", {"method": "points"})))
    assert result.verdict is Verdict.FAIL
    assert result.evidence[0].value["description"] == "burning points on a paid route"
    assert "burning points on a paid route" in result.reason


def test_reason_falls_back_to_action_and_args_without_description():
    # The travel spec has no descriptions yet (TBD by hand): the reason
    # must still say WHAT happened, mechanically.
    result = GRADER.grade(_spec(), _trajectory(("pay", {"method": "points"})))
    assert result.verdict is Verdict.FAIL
    assert "step 1: pay {'method': 'points'}" in result.reason


def test_grading_does_not_mutate_inputs():
    spec, trajectory = _spec(), _fixture("critical_recovered")
    spec_before, trajectory_before = spec.to_dict(), trajectory.to_dict()
    GRADER.grade(spec, trajectory)
    assert spec.to_dict() == spec_before
    assert trajectory.to_dict() == trajectory_before
