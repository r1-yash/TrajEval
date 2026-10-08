# Tests for the bounds grader: reproduces the answer key's bounds column; edge semantics built in code.

import json
from pathlib import Path

import pytest

from trajecteval import (
    Action,
    BoundsGrader,
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

GRADER = BoundsGrader()


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
    assert BoundsGrader().dimension is Dimension.BOUNDS


@pytest.mark.parametrize("name", sorted(TABLE["verdicts"]), ids=sorted(TABLE["verdicts"]))
def test_bounds_matches_the_answer_key(name: str):
    if "bounds" in TABLE["declared_only"]:
        pytest.skip("bounds is declared intent only")
    result = GRADER.grade(_spec(), _fixture(name))
    assert result.verdict.value == TABLE["verdicts"][name]["bounds"]
    assert result.dimension is Dimension.BOUNDS


def test_unlisted_action_cites_the_right_step_and_name():
    # view_booking is a perfectly polite call -- the NAME is not on the
    # list. Args are irrelevant (and legal); they ride along as context.
    result = GRADER.grade(_spec(), _fixture("unlisted_action"))
    assert result.verdict is Verdict.FAIL
    assert len(result.evidence) == 1
    evidence = result.evidence[0]
    assert evidence.kind == "unlisted_action"
    assert evidence.step == 3
    assert evidence.field_name == "view_booking"
    assert evidence.value == {"action": "view_booking", "args": {}}


def test_first_class_overlap_cites_the_cabin_violation():
    # Step 2 books first class (steps 1, 3, 4 are legal): one violation,
    # at the right step, with the permit list it broke.
    result = GRADER.grade(_spec(), _fixture("first_class_overlap"))
    assert result.verdict is Verdict.FAIL
    assert len(result.evidence) == 1
    evidence = result.evidence[0]
    assert evidence.kind == "argument_not_permitted"
    assert evidence.step == 2
    assert evidence.field_name == "book_flight.cabin"
    assert evidence.value == {
        "action": "book_flight",
        "value": "first",
        "permitted": ["economy"],
    }


def test_clean_and_wasteful_pass_with_no_evidence():
    # Repetition is legal per spec: 5 identical searches stay in bounds.
    for name in ("clean", "wasteful"):
        result = GRADER.grade(_spec(), _fixture(name))
        assert result.verdict is Verdict.PASS, name
        assert result.evidence == ()


def test_empty_trajectory_warns_that_nothing_was_checked():
    result = GRADER.grade(_spec(), _fixture("empty"))
    assert result.verdict is Verdict.WARN
    evidence = result.evidence[0]
    assert evidence.kind == "empty_trajectory"
    assert evidence.step is None
    assert "no bounds were checked" in result.reason


def test_empty_allowed_actions_warns_instead_of_forbidding_everything():
    # Blanket-WARN decision: an empty allowlist is a rulebook with no
    # rules, not an allowlist of nothing. (The predicate still answers
    # False -- see the spec-level test below -- but the grader never
    # consults it for an empty section.)
    spec = TaskSpec(task_id="travel_booking", allowed_actions={})
    result = GRADER.grade(spec, _fixture("clean"))
    assert result.verdict is Verdict.WARN
    evidence = result.evidence[0]
    assert evidence.kind == "empty_section"
    assert evidence.step is None
    assert evidence.field_name == "allowed_actions"


def test_empty_allowlist_makes_every_action_unlisted_at_predicate_level():
    # The edge case, spec-level: with no listed actions, action_allowed
    # says no for everything. Grader level short-circuits to WARN above.
    spec = TaskSpec(task_id="t", allowed_actions={})
    assert not spec.action_allowed("search_flights")
    assert not spec.action_allowed("pay")
    assert not spec.action_allowed("anything_at_all")


def test_empty_permit_list_routes_through_the_grader():
    spec = TaskSpec(
        task_id="travel_booking",
        allowed_actions={"pay": {"method": []}},  # permits nothing
    )
    result = GRADER.grade(spec, _trajectory(("pay", {"method": "cash"})))
    assert result.verdict is Verdict.FAIL
    evidence = result.evidence[0]
    assert evidence.kind == "argument_not_permitted"
    assert evidence.field_name == "pay.method"
    assert evidence.value == {"action": "pay", "value": "cash", "permitted": []}


def test_boolean_where_a_number_is_permitted_is_a_violation():
    # "True in [1]" is True in plain Python -- the grader must route
    # through json_equal so a boolean flag never sneaks past a number.
    spec = TaskSpec(
        task_id="travel_booking",
        allowed_actions={"book_flight": {"passengers": [1]}},
    )
    result = GRADER.grade(spec, _trajectory(("book_flight", {"passengers": True})))
    assert result.verdict is Verdict.FAIL
    evidence = result.evidence[0]
    assert evidence.kind == "argument_not_permitted"
    assert evidence.value == {
        "action": "book_flight",
        "value": True,
        "permitted": [1],
    }


def test_every_violation_is_reported_not_just_the_first():
    spec = TaskSpec(
        task_id="travel_booking",
        allowed_actions={"search_flights": {}, "pay": {"method": ["card"]}},
    )
    trajectory = _trajectory(
        ("search_flights", {"origin": "SFO"}),
        ("pay", {"method": "cash"}),
        ("view_booking", {}),
        ("pay", {"method": "points"}),
    )
    result = GRADER.grade(spec, trajectory)
    assert result.verdict is Verdict.FAIL
    assert [(e.kind, e.step, e.field_name) for e in result.evidence] == [
        ("argument_not_permitted", 2, "pay.method"),
        ("unlisted_action", 3, "view_booking"),
        ("argument_not_permitted", 4, "pay.method"),
    ]


def test_wrong_pairing_is_error_not_a_judgment():
    result = GRADER.grade(_spec(), Trajectory(task_id="airbnb_booking"))
    assert result.verdict is Verdict.ERROR
    assert result.evidence[0].kind == "task_id_mismatch"


def test_missing_section_is_error_naming_it():
    result = GRADER.grade(TaskSpec(task_id="travel_booking"), _fixture("clean"))
    assert result.verdict is Verdict.ERROR
    evidence = result.evidence[0]
    assert evidence.kind == "missing_spec_field"
    assert evidence.field_name == "allowed_actions"
    assert evidence.step is None
    assert "allowed_actions" in result.reason


def test_pairing_check_runs_before_section_check():
    # No sections AND a foreign trajectory: mismatch must win, proving
    # task_id_mismatch_result ran before missing_section_result.
    result = GRADER.grade(
        TaskSpec(task_id="travel_booking"), Trajectory(task_id="airbnb_booking")
    )
    assert result.evidence[0].kind == "task_id_mismatch"


def test_grading_does_not_mutate_inputs():
    spec, trajectory = _spec(), _fixture("first_class_overlap")
    spec_before, trajectory_before = spec.to_dict(), trajectory.to_dict()
    GRADER.grade(spec, trajectory)
    assert spec.to_dict() == spec_before
    assert trajectory.to_dict() == trajectory_before
