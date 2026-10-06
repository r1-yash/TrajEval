"""Tests for task specs: load-by-id, the pinned schema, and the two-door errors."""

import json
from pathlib import Path

import pytest

from trajecteval import Action, CriticalPattern, RULE_SECTIONS, SpecError, Step, TaskSpec, load_spec

REPO = Path(__file__).resolve().parents[1]
TASKS = REPO / "tasks"


def test_load_travel_booking_by_id():
    spec = load_spec("travel_booking", tasks_dir=TASKS)
    assert spec.task_id == "travel_booking"
    assert spec.expected_final_state == {"status": "confirmed", "passengers": 1}
    assert spec.allowed_actions == {
        "search_flights": {},
        "book_flight": {"cabin": ["economy"]},
        # points is deliberately bounds-legal: its only failing dimension is
        # critical (dimension isolation for the Step 5 fixtures)
        "pay": {"method": ["card", "points"]},
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


def test_load_spec_rejects_id_mismatch(tmp_path: Path):
    # A copied/renamed file: tasks/foo.json declares task_id "bar". Loading
    # it silently would hand callers a spec whose id disagrees with the one
    # they asked for -- and with every trajectory pairing built from it.
    (tmp_path / "foo.json").write_text(json.dumps({"task_id": "bar"}), encoding="utf-8")
    with pytest.raises(SpecError) as exc_info:
        load_spec("foo", tasks_dir=tmp_path)
    message = str(exc_info.value)
    assert "'bar'" in message  # the id the file declares
    assert "'foo'" in message  # the id it was loaded as


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


def test_action_allowed_distinguishes_listed_empty_and_missing():
    spec = TaskSpec(task_id="t", allowed_actions={"search": {}})
    assert spec.action_allowed("search")          # listed with {} -> allowed
    assert not spec.action_allowed("book_flight")  # unlisted -> out of bounds


def test_empty_permit_map_constrains_no_arguments():
    spec = TaskSpec(task_id="t", allowed_actions={"search": {}})
    assert spec.argument_violations("search", {"q": "cheap flights", "limit": 5}) == ()


def test_empty_permit_list_forbids_any_supplied_value():
    spec = TaskSpec(task_id="t", allowed_actions={"pay": {"method": []}})
    assert spec.argument_violations("pay", {"method": "cash"}) == (("method", "cash"),)
    assert spec.argument_violations("pay", {"amount": 100}) == ()  # nothing supplied


def test_argument_outside_permit_map_is_unconstrained():
    spec = TaskSpec(task_id="t", allowed_actions={"pay": {"method": ["card"]}})
    assert spec.argument_violations("pay", {"method": "card", "memo": "team offsite"}) == ()


def test_listed_but_absent_argument_is_not_a_violation():
    # Permits police what happened; they can never require an argument --
    # completeness is expected_final_state's job.
    spec = TaskSpec(task_id="t", allowed_actions={"pay": {"method": ["card"]}})
    assert spec.argument_violations("pay", {}) == ()


def test_argument_violations_reports_each_forbidden_argument():
    spec = TaskSpec(
        task_id="t",
        allowed_actions={"pay": {"method": ["card"], "currency": ["USD"]}},
    )
    assert spec.argument_violations(
        "pay", {"method": "cash", "currency": "BTC", "memo": "hi"}
    ) == (("method", "cash"), ("currency", "BTC"))  # memo: unconstrained


def test_action_predicates_require_section_checked_first():
    spec = TaskSpec(task_id="t")  # section absent -- "can't judge", not "forbidden"
    with pytest.raises(ValueError, match="missing_section_result"):
        spec.action_allowed("search")
    with pytest.raises(ValueError, match="missing_section_result"):
        spec.argument_violations("search", {})


def test_argument_violations_requires_a_listed_action():
    spec = TaskSpec(task_id="t", allowed_actions={"pay": {}})
    with pytest.raises(ValueError, match="action_allowed"):
        spec.argument_violations("search", {"q": "x"})


def test_json_null_section_equals_omitted():
    for section in RULE_SECTIONS:
        assert TaskSpec.from_dict({"task_id": "t", section: None}) == TaskSpec.from_dict(
            {"task_id": "t"}
        )


def test_json_null_metadata_normalizes_to_empty():
    assert TaskSpec.from_dict({"task_id": "t", "metadata": None}) == TaskSpec.from_dict(
        {"task_id": "t"}
    )


def test_pattern_description_round_trips():
    pattern = CriticalPattern(
        action="pay", args={"method": "points"}, description="pay with the card"
    )
    assert pattern.to_dict()["description"] == "pay with the card"
    assert CriticalPattern.from_dict(pattern.to_dict()) == pattern


def test_pattern_description_is_optional_and_null_is_omitted():
    # travel_booking.json ships without descriptions (strings TBD by hand):
    # absent and explicit null both land on None, to_dict omits the key.
    pattern = CriticalPattern(action="pay")
    assert pattern.description is None
    assert "description" not in pattern.to_dict()
    assert CriticalPattern.from_dict({"action": "pay"}).description is None
    assert CriticalPattern.from_dict(
        {"action": "pay", "description": None}
    ).description is None


def test_pattern_description_must_be_non_empty_when_present():
    with pytest.raises(SpecError, match="critical pattern 'description'"):
        CriticalPattern(action="pay", description="")


def test_pattern_description_does_not_affect_matching():
    # Narrative only: nothing parses it, matches() ignores it.
    without = CriticalPattern(action="book_flight", args={"cabin": "first"})
    with_desc = CriticalPattern(
        action="book_flight", args={"cabin": "first"}, description="over budget"
    )
    assert with_desc.matches(_book_step(cabin="first"))
    assert not with_desc.matches(_book_step(cabin="economy"))
    assert with_desc.matches(_book_step(cabin="first")) == without.matches(
        _book_step(cabin="first")
    )


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
