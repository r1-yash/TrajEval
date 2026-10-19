# Step 12 agent-loop tests: a scripted fake model client drives the REAL
# environment and Recorder -- no network, no key, no spend, no sleeping
# (every run injects a no-op pause). One test per loop behavior.
#
# The live test lives in test_agent_live.py behind the marker.

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from trajecteval.agent import run_agent
from trajecteval.envs import TravelBookingEnvironment
from trajecteval.models import Trajectory
from trajecteval.report import evaluate
from trajecteval.task_spec import load_spec

REPO = Path(__file__).resolve().parent.parent

# The instruction is the loop's input, not its contract: a fixed stub keeps
# these tests independent of the real instruction's wording.
INSTRUCTION = "Book a flight."


class FakeClient:
    # Canned replies, no network: complete() records the prompt it was
    # handed and pops the next reply (the last one repeats forever, so a
    # looping model is one string long).

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if len(self._replies) > 1:
            return self._replies.pop(0)
        return self._replies[0]


def _action(name: str, **args: Any) -> str:
    return json.dumps({"action": name, "args": args})


DONE = json.dumps({"done": True})


def _no_pause() -> None:
    pass  # tests never sleep


def _run(client: FakeClient, *, max_steps: int = 12, model: str = "fake-model"):
    return run_agent(
        INSTRUCTION,
        TravelBookingEnvironment(),
        client,
        model=model,
        max_steps=max_steps,
        pause=_no_pause,
    )


def test_clean_run_ends_done_and_is_gradeable() -> None:
    client = FakeClient(
        [
            _action("search_flights", origin="SFO", destination="NRT"),
            _action("book_flight", flight="TR420", cabin="economy", passengers=1),
            _action("pay", method="card"),
            DONE,
        ]
    )
    trajectory = _run(client)
    assert trajectory.metadata["stop_reason"] == "done"
    assert len(trajectory.steps) == 3
    assert len(client.calls) == 4

    # It loads through the existing loader and grades like any trajectory.
    loaded = Trajectory.from_dict(trajectory.to_dict())
    report = evaluate(loaded, load_spec("travel_booking", tasks_dir=REPO / "tasks"))
    verdicts = {r.dimension.value: r.verdict.value for r in report.results}
    assert verdicts == {"final_state": "PASS", "bounds": "PASS", "critical": "PASS"}


def test_rejected_action_is_recorded_fed_back_and_the_run_continues() -> None:
    client = FakeClient(
        [
            _action("pay", method="card"),  # impossible: nothing is booked
            _action("search_flights", origin="SFO", destination="NRT"),
            _action("book_flight", flight="TR420", cabin="economy", passengers=1),
            _action("pay", method="card"),
            DONE,
        ]
    )
    trajectory = _run(client)
    assert trajectory.metadata["stop_reason"] == "done"
    assert len(trajectory.steps) == 4  # the rejected attempt included
    first = trajectory.steps[0]
    assert first.state_before == first.state_after  # recorded, unchanged
    # The world's rejection message came back to the model in the next turn.
    assert "requires status" in client.calls[1][1]


def test_step_limit_stops_a_looping_model_and_returns_the_trajectory() -> None:
    client = FakeClient([_action("search_flights", origin="SFO", destination="NRT")])
    trajectory = _run(client, max_steps=3)
    assert trajectory.metadata["stop_reason"] == "step_limit"
    assert len(trajectory.steps) == 3
    assert len(client.calls) == 3  # no call after the limit


def test_three_consecutive_bad_replies_stop_the_run_without_steps() -> None:
    client = FakeClient(["not json", '{"foo": 1}', '["action"]'])
    trajectory = _run(client)
    assert trajectory.metadata["stop_reason"] == "bad_output"
    assert trajectory.steps == ()
    assert len(client.calls) == 3


def test_bad_replies_are_never_recorded_and_do_not_stick() -> None:
    client = FakeClient(
        ["garbage", _action("search_flights", origin="SFO", destination="NRT"), DONE]
    )
    trajectory = _run(client)
    assert trajectory.metadata["stop_reason"] == "done"
    assert len(trajectory.steps) == 1  # only the real action, not the bad reply
    assert trajectory.steps[0].action.name == "search_flights"
    assert len(client.calls) == 3


def test_metadata_carries_model_date_limit_and_stop_reason() -> None:
    client = FakeClient([_action("search_flights", origin="SFO", destination="NRT"), DONE])
    trajectory = _run(client, max_steps=5, model="provider/some-model")
    assert trajectory.metadata == {
        "model": "provider/some-model",
        "date": date.today().isoformat(),
        "max_steps": 5,
        "stop_reason": "done",
    }
