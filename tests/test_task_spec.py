"""Tests for task specs: load-by-id, the pinned schema, and the two-door errors."""

from pathlib import Path

import pytest

from trajecteval import Action, CriticalPattern, SpecError, Step, TaskSpec, load_spec

REPO = Path(__file__).resolve().parents[1]
TASKS = REPO / "tasks"


def test_load_travel_booking_by_id():
    spec = load_spec("travel_booking", tasks_dir=TASKS)
    assert spec.task_id == "travel_booking"
    assert spec.expected_final_state == {"status": "confirmed", "passengers": 1}
    assert spec.allowed_actions == {
        "search_flights": {},
        "book_flight": {"cabin": ["economy"]},
        "pay": {"method": ["card"]},
    }
    assert [p.action for p in spec.critical_error_patterns] == ["book_flight", "pay"]
    assert spec.metadata["budget"] == 1200


def test_default_tasks_dir_is_cwd(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(REPO)
    assert load_spec("travel_booking").task_id == "travel_booking"


def test_spec_round_trip_equality():
    spec = load_spec("travel_booking", tasks_dir=TASKS)
    assert TaskSpec.from_dict(spec.to_dict()) == spec


def test_to_dict_omits_absent_sections():
    data = TaskSpec(task_id="bare").to_dict()
    assert set(data) == {"task_id", "metadata"}
    assert TaskSpec.from_dict(data) == TaskSpec(task_id="bare")


def test_missing_file_raises_spec_error(tmp_path: Path):
    with pytest.raises(SpecError, match="cannot read"):
        load_spec("no_such_task", tasks_dir=tmp_path)


def test_invalid_json_raises_spec_error(tmp_path: Path):
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(SpecError, match="is not valid JSON"):
        load_spec("broken", tasks_dir=tmp_path)


def test_top_level_must_be_object(tmp_path: Path):
    (tmp_path / "listy.json").write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(SpecError, match="expected an object, got list"):
        load_spec("listy", tasks_dir=tmp_path)


def test_task_id_cannot_be_a_path(tmp_path: Path):
    # Rejected before any file access -- an id is a file name, not a path.
    for bad in ("../evil", "a/b", "UPPER", "has space", ".hidden", "_lead"):
        with pytest.raises(SpecError, match="task id"):
            load_spec(bad, tasks_dir=tmp_path)


def test_empty_task_id_rejected():
    with pytest.raises(SpecError, match="task id"):
        load_spec("", tasks_dir=TASKS)


def test_missing_task_id_in_dict():
    with pytest.raises(SpecError, match="missing required field 'task_id'"):
        TaskSpec.from_dict({})


def test_unknown_top_level_key_rejected():
    # A hand-written rule file with an unknown key is a typo, not a new rule.
    with pytest.raises(SpecError, match=r"unknown key\(s\) 'allowedAction'"):
        TaskSpec.from_dict({"task_id": "x", "allowedAction": {}})


def test_wrong_section_types_rejected():
    with pytest.raises(SpecError, match="spec 'expected_final_state': expected an object"):
        TaskSpec(task_id="x", expected_final_state=["status"])
    with pytest.raises(SpecError, match="spec 'allowed_actions': expected an object"):
        TaskSpec(task_id="x", allowed_actions="everything goes")
    with pytest.raises(SpecError, match="spec 'metadata': expected an object"):
        TaskSpec(task_id="x", metadata=["not", "a", "dict"])


def test_allowed_action_value_must_be_permit_object():
    with pytest.raises(SpecError, match=r"spec 'allowed_actions'\['book_flight'\]"):
        TaskSpec(task_id="x", allowed_actions={"book_flight": "economy"})


def test_permit_must_be_a_list():
    with pytest.raises(SpecError, match="expected a list of permitted values"):
        TaskSpec(task_id="x", allowed_actions={"book_flight": {"cabin": "economy"}})


def test_pattern_requires_action():
    with pytest.raises(SpecError, match="missing required field 'action'"):
        CriticalPattern.from_dict({"args": {"cabin": "first"}})


def test_pattern_rejects_unknown_keys():
    with pytest.raises(SpecError, match=r"unknown key\(s\) 'step'"):
        CriticalPattern.from_dict({"action": "pay", "step": 3})


def test_pattern_requires_non_empty_action():
    with pytest.raises(SpecError, match="critical pattern 'action'"):
        CriticalPattern(action="")


def test_patterns_must_be_a_list():
    with pytest.raises(SpecError, match="spec 'critical_error_patterns': expected a list"):
        TaskSpec.from_dict({"task_id": "x", "critical_error_patterns": {"action": "pay"}})


def test_absent_sections_are_none_empty_sections_stay_empty():
    absent = TaskSpec(task_id="t")
    assert absent.expected_final_state is None
    assert absent.allowed_actions is None
    assert absent.critical_error_patterns is None

    empty = TaskSpec.from_dict(
        {
            "task_id": "t",
            "expected_final_state": {},
            "allowed_actions": {},
            "critical_error_patterns": [],
        }
    )
    assert empty.expected_final_state == {}
    assert empty.allowed_actions == {}
    assert empty.critical_error_patterns == ()


def test_pattern_list_is_coerced_to_tuple():
    spec = TaskSpec(task_id="t", critical_error_patterns=[CriticalPattern(action="pay")])
    assert isinstance(spec.critical_error_patterns, tuple)


def _book_step(cabin: str | None = "economy") -> Step:
    args = {"cabin": cabin} if cabin is not None else {}
    return Step(
        index=4,
        state_before={},
        action=Action(name="book_flight", args=args),
        state_after={},
    )


def test_pattern_matches_exact_action_and_args():
    pattern = CriticalPattern(action="book_flight", args={"cabin": "first"})
    assert pattern.matches(_book_step(cabin="first"))
    assert not pattern.matches(_book_step(cabin="economy"))  # right action, wrong value


def test_pattern_never_matches_a_missing_argument():
    pattern = CriticalPattern(action="book_flight", args={"cabin": "first"})
    assert not pattern.matches(_book_step(cabin=None))  # not making the move != making it


def test_pattern_without_args_matches_any_call():
    pattern = CriticalPattern(action="cancel_booking")
    assert pattern.matches(_book_step(cabin="economy")) is False  # wrong action
    cancel = Step(
        index=9,
        state_before={},
        action=Action(name="cancel_booking", args={"reason": "changed plans"}),
        state_after={},
    )
    assert pattern.matches(cancel)
