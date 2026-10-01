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
    assert traj.metadata["agent"] == "toy-sandbox"
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
    assert traj.steps == []
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


def test_round_trip_through_to_dict():
    original = Trajectory.from_json_file(FIXTURES / "minimal.json")
    reloaded = Trajectory.from_dict(original.to_dict())
    assert reloaded.to_dict() == original.to_dict()
    assert reloaded.task_id == original.task_id
    assert len(reloaded.steps) == len(original.steps)
