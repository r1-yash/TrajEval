"""Tests for Step 2: domain models (Action, Step, Trajectory, Discontinuity)."""

from pathlib import Path

import pytest

from trajecteval.errors import TrajectoryError
from trajecteval.models import Action, Discontinuity, Step, Trajectory

FIXTURES = Path(__file__).parent / "fixtures"


def make_trajectory_dict(task_id="t1", steps=None):
    """Small helper so tests state only what they care about."""
    if steps is None:
        steps = [
            {
                "index": 1,
                "state_before": {"v": 0},
                "action": {"name": "inc", "args": {"by": 1}},
                "state_after": {"v": 1},
            }
        ]
    return {"task_id": task_id, "steps": steps}


def test_load_minimal_fixture():
    traj = Trajectory.from_json_file(FIXTURES / "minimal.json")
    assert traj.task_id == "toy-edit-001"
    assert len(traj.steps) == 2
    assert traj.metadata["agent"] == "toy-agent"
    # unknown top-level key "source" was swept into metadata
    assert traj.metadata["source"] == "fixture"


def test_nested_attribute_access():
    traj = Trajectory.from_json_file(FIXTURES / "minimal.json")
    assert traj.steps[0].action.name == "read_file"
    assert traj.steps[1].action.args == {"path": "a.txt", "old": "hello", "new": "hi"}
    assert traj.steps[1].state_after == {"files": {"a.txt": "hi"}}


def test_missing_step_key_raises_naming_step():
    data = make_trajectory_dict()
    del data["steps"][0]["state_after"]
    with pytest.raises(TrajectoryError, match="step 1: missing 'state_after'"):
        Trajectory.from_dict(data)


def test_index_gap_raises():
    data = make_trajectory_dict(steps=[
        {"index": 1, "state_before": {}, "action": {"name": "a"}, "state_after": {}},
        {"index": 2, "state_before": {}, "action": {"name": "a"}, "state_after": {}},
        {"index": 4, "state_before": {}, "action": {"name": "a"}, "state_after": {}},
    ])
    with pytest.raises(TrajectoryError, match="step 3: .*gap.*expected 3, got 4"):
        Trajectory.from_dict(data)


def test_index_duplicate_raises():
    data = make_trajectory_dict(steps=[
        {"index": 1, "state_before": {}, "action": {"name": "a"}, "state_after": {}},
        {"index": 1, "state_before": {}, "action": {"name": "a"}, "state_after": {}},
    ])
    with pytest.raises(TrajectoryError, match="step 2: .*duplicate.*expected 2, got 1"):
        Trajectory.from_dict(data)


def test_missing_task_id_raises():
    with pytest.raises(TrajectoryError, match="missing required field 'task_id'"):
        Trajectory.from_dict({"steps": []})


def test_empty_trajectory_is_valid_with_no_mismatches():
    traj = Trajectory.from_dict({"task_id": "empty", "steps": []})
    assert traj.steps == ()  # Trajectory is frozen; steps is a tuple
    assert traj.find_discontinuities() == []


def test_discontinuity_is_reported_not_raised():
    # Loads fine, but the mismatch between step 1's state_after and step 2's
    # state_before (both "files" and an extra sneaky key) is reported.
    traj = Trajectory.from_json_file(FIXTURES / "discontinuous.json")
    assert len(traj.steps) == 3  # it loaded!

    mismatches = traj.find_discontinuities()
    assert len(mismatches) == 1
    m = mismatches[0]
    assert isinstance(m, Discontinuity)
    assert m.step_index == 1
    assert m.next_step_index == 2
    assert m.differing_keys == ["files", "sneaky.txt"]
    # and step 2 -> 3 agrees, so only the one mismatch
    assert str(m) == "steps 1->2 disagree in keys: files, sneaky.txt"


def test_unknown_keys_swept_into_metadata():
    data = make_trajectory_dict()
    data["pipeline"] = "ci"
    traj = Trajectory.from_dict(data)
    assert traj.metadata["pipeline"] == "ci"


def test_collision_metadata_wins_over_top_level():
    data = make_trajectory_dict()
    data["source"] = "top-level-loses"
    data["metadata"] = {"source": "metadata-wins", "other": 1}
    traj = Trajectory.from_dict(data)
    assert traj.metadata["source"] == "metadata-wins"
    assert traj.metadata["other"] == 1


def test_round_trip_preserves_objects():
    """Round-trip must reproduce the *objects* (identity of every field),
    not merely dicts that happen to serialize the same way."""
    original = Trajectory.from_json_file(FIXTURES / "minimal.json")
    reloaded = Trajectory.from_dict(original.to_dict())

    assert reloaded == original  # dataclass __eq__ compares every field
    assert reloaded.task_id == original.task_id
    assert reloaded.metadata == original.metadata
    assert reloaded.steps == original.steps
    for a, b in zip(reloaded.steps, original.steps):
        assert a == b
        assert a.action == b.action
        assert a.state_before == b.state_before
        assert a.state_after == b.state_after


# ---------------------------------------------------------------- action rules


def test_action_args_must_be_a_dict():
    data = make_trajectory_dict()
    data["steps"][0]["action"]["args"] = "by=1"
    with pytest.raises(TrajectoryError, match="step 1: 'action': 'args' must be an object"):
        Trajectory.from_dict(data)


def test_action_name_missing_raises():
    data = make_trajectory_dict()
    del data["steps"][0]["action"]["name"]
    with pytest.raises(TrajectoryError, match="step 1: 'action': missing 'name'"):
        Trajectory.from_dict(data)


def test_action_without_args_defaults_to_empty():
    data = make_trajectory_dict()
    del data["steps"][0]["action"]["args"]
    traj = Trajectory.from_dict(data)
    assert traj.steps[0].action.args == {}


def test_bool_index_rejected():
    # bool is a subclass of int in Python; True must not pass as index 1.
    data = make_trajectory_dict(steps=[
        {"index": True, "state_before": {}, "action": {"name": "a"}, "state_after": {}},
    ])
    with pytest.raises(TrajectoryError, match="step 1: 'index' must be an int, got bool"):
        Trajectory.from_dict(data)


# ------------------------------------------------------- file-level errors


def test_invalid_json_file_raises(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(TrajectoryError, match="is not valid JSON"):
        Trajectory.from_json_file(bad)


def test_unreadable_path_raises(tmp_path):
    missing = tmp_path / "nope.json"
    with pytest.raises(TrajectoryError, match="cannot read"):
        Trajectory.from_json_file(missing)


def test_directory_path_raises(tmp_path):
    with pytest.raises(TrajectoryError, match="cannot read"):
        Trajectory.from_json_file(tmp_path)
