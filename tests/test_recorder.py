"""Step 11 recorder tests.

The key one is the replay test: every hand-made fixture's actions are fed
through a fresh environment + recorder, and the recorded trajectory must
match the fixture's states exactly and grade to the same deterministic
verdicts. If a fixture cannot be replayed, this fails first -- the fixtures
are the ground truth, not the environment.

Plus the recorder's own contracts: a rejected action is still recorded (and
re-raised), recorded states are independent copies, and the file round trip
goes through the existing loader. Offline and deterministic, like the rest
of the default suite.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trajecteval import evaluate, load_spec
from trajecteval.envs import ActionRejected, TravelBookingEnvironment
from trajecteval.models import Action, Trajectory
from trajecteval.recorder import Recorder

REPO = Path(__file__).resolve().parent.parent
FIXTURES = REPO / "tests" / "fixtures" / "travel_booking"
TABLE = json.loads(
    (FIXTURES / "expected_verdicts.json").read_text(encoding="utf-8")
)["verdicts"]
# Every fixture, including empty (zero actions replays to a zero-step trace).
FIXTURE_NAMES = [
    "clean",
    "wasteful",
    "critical_recovered",
    "first_class_overlap",
    "failed",
    "unlisted_action",
    "empty",
]


def _replay(name: str) -> Trajectory:
    # Feed a fixture's actions through a fresh world + recorder. The fixture
    # is only the source of actions; every state comes from the world.
    fixture = Trajectory.from_json_file(FIXTURES / f"{name}.json")
    recorder = Recorder(TravelBookingEnvironment())
    for step in fixture.steps:
        try:
            recorder.act(step.action.name, step.action.args)
        except ActionRejected:
            pass  # a rejected attempt is recorded too; the replay continues
    return recorder.trajectory()


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_every_fixture_replays_exactly(name: str) -> None:
    fixture = Trajectory.from_json_file(FIXTURES / f"{name}.json")
    replayed = _replay(name)

    # (a) loads cleanly through the existing loader and is continuous.
    loaded = Trajectory.from_dict(replayed.to_dict())
    assert loaded.find_discontinuities() == []

    # (b) every recorded state matches the fixture's states exactly. The
    # replay carries no fixture metadata, so the claim is per-step, not
    # whole-file.
    assert len(replayed.steps) == len(fixture.steps)
    for recorded, original in zip(replayed.steps, fixture.steps):
        assert recorded.index == original.index
        assert recorded.action == original.action
        assert recorded.state_before == original.state_before
        assert recorded.state_after == original.state_after

    # (c) the deterministic graders reproduce the answer key (the three
    # non-judged columns; trajectory_quality is declared-only there).
    spec = load_spec("travel_booking", tasks_dir=REPO / "tasks")
    report = evaluate(replayed, spec)
    verdicts = {result.dimension.value: result.verdict.value for result in report.results}
    assert verdicts == {
        dimension: TABLE[name][dimension]
        for dimension in ("final_state", "bounds", "critical")
    }


def test_a_rejected_action_is_recorded_then_reraised() -> None:
    # Approved contract: act() re-raises ActionRejected -- but the attempt
    # was recorded first (state_after equals state_before), and the world
    # itself is unchanged.
    env = TravelBookingEnvironment()
    recorder = Recorder(env)
    with pytest.raises(ActionRejected):
        recorder.act("pay", {"method": "card"})  # nothing is booked yet
    assert env.current_state() == {"status": "new"}

    step = recorder.trajectory().steps[0]
    assert step.index == 1
    assert step.action.name == "pay"
    assert step.state_before == step.state_after == {"status": "new"}


def test_recorded_states_are_independent_copies() -> None:
    # The aliasing bug, pinned from both sides: neither the world moving on
    # nor the caller mutating act()'s return may rewrite a recorded step.
    env = TravelBookingEnvironment()
    recorder = Recorder(env)
    recorder.act("search_flights", {"origin": "SFO", "destination": "NRT"})
    returned = recorder.act(
        "book_flight", {"flight": "TR420", "cabin": "economy", "passengers": 1}
    )
    trajectory = recorder.trajectory()

    returned["cabin"] = "MUTATED"  # caller mutates what it was handed
    env.run_action(Action("pay", {"method": "card"}))  # world moves on

    first, second = trajectory.steps
    assert first.state_after == {"status": "searching", "searches": 1}
    assert second.state_after["cabin"] == "economy"


def test_trajectory_file_round_trip(tmp_path: Path) -> None:
    # to_json_file writes what Trajectory.from_json_file reads; the caller's
    # metadata rides along into the recorded trajectory.
    recorder = Recorder(
        TravelBookingEnvironment(), metadata={"agent": "scripted-demo"}
    )
    recorder.act("search_flights", {"origin": "SFO", "destination": "NRT"})
    recorder.act(
        "book_flight", {"flight": "TR420", "cabin": "economy", "passengers": 1}
    )
    path = tmp_path / "recorded.json"
    recorder.trajectory().to_json_file(path)
    loaded = Trajectory.from_json_file(path)
    assert loaded == recorder.trajectory()
    assert loaded.metadata == {"agent": "scripted-demo"}
