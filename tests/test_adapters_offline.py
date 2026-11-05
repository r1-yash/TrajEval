"""Tests for the offline external-run adapter foundation.

The toy adapter below is deliberately not any real external format: it maps
small synthetic run records into the format-neutral ``ExternalStep`` boundary.
That keeps these tests focused on the shared translation contract rather than
on Debuggernaut (or any other) field names.
"""

from __future__ import annotations

from typing import Any

import pytest

from trajecteval import (
    AdapterError,
    ExternalStep,
    OfflineRunAdapter,
    Trajectory,
    UNAVAILABLE_STATE_KEY,
    is_unavailable_state,
    unavailable_state,
)


class ToyRunAdapter(OfflineRunAdapter):
    """A tiny synthetic format: run_id / events / run_metadata."""

    def extract_task_id(self, record: dict[str, Any]) -> str:
        return record.get("run_id")

    def extract_steps(self, record: dict[str, Any]) -> list[ExternalStep]:
        events = record.get("events")
        if not isinstance(events, list):
            raise AdapterError(
                f"external run: 'events' must be a list, got {type(events).__name__}"
            )
        steps: list[ExternalStep] = []
        for position, event in enumerate(events, start=1):
            if not isinstance(event, dict):
                raise AdapterError(
                    f"external event {position}: expected an object, "
                    f"got {type(event).__name__}"
                )
            action = event.get("action")
            if not isinstance(action, dict):
                raise AdapterError(
                    f"external event {position}: 'action' must be an object, "
                    f"got {type(action).__name__}"
                )
            steps.append(
                ExternalStep(
                    action_name=action.get("name"),
                    action_args=action.get("args", {}),
                    state_before=event.get("before"),
                    state_after=event.get("after"),
                )
            )
        return steps

    def extract_metadata(self, record: dict[str, Any]) -> dict[str, Any]:
        return record.get("run_metadata", {})


def make_record() -> dict[str, Any]:
    return {
        "run_id": "toy-run-001",
        "events": [
            {
                # Deliberately not in ascending external numbering: conversion
                # follows sequence order and assigns TrajEval's 1-based indices.
                "sequence": 20,
                "action": {
                    "name": "search_flights",
                    "args": {
                        "origin": "SFO",
                        "filters": {"cabins": ["economy"], "max_stops": 1},
                    },
                },
                "before": {"status": "new"},
                "after": {"status": "searching", "results": 2},
            },
            {
                "sequence": 10,
                "action": {
                    "name": "book_flight",
                    "args": {"flight": "TR420", "passengers": 1},
                },
                "before": {"status": "searching", "results": 2},
                "after": {"status": "booked", "flight": "TR420"},
            },
            {
                "sequence": 30,
                "action": {"name": "pay", "args": {"method": "card"}},
                "before": {"status": "booked", "flight": "TR420"},
                "after": {"status": "confirmed", "flight": "TR420"},
            },
        ],
        "run_metadata": {
            "model": "toy-agent-1",
            "stop_reason": "done",
            "source": {"harness": "synthetic", "attempt": 2},
        },
    }


def test_converts_a_synthetic_run_into_the_existing_trajectory_model() -> None:
    trajectory = ToyRunAdapter().to_trajectory(make_record())

    # The conversion produces the one existing Trajectory model, not a
    # competing external-run representation.
    assert isinstance(trajectory, Trajectory)
    assert trajectory.task_id == "toy-run-001"
    assert len(trajectory.steps) == 3
    # It is a first-class Trajectory: the existing round-trip sees no
    # second model or adapter-only shape.
    assert Trajectory.from_dict(trajectory.to_dict()) == trajectory


def test_multiple_steps_keep_source_order_and_get_canonical_indices() -> None:
    trajectory = ToyRunAdapter().to_trajectory(make_record())

    assert [step.index for step in trajectory.steps] == [1, 2, 3]
    assert [step.action.name for step in trajectory.steps] == [
        "search_flights",
        "book_flight",
        "pay",
    ]
    assert trajectory.steps[0].state_before == {"status": "new"}
    assert trajectory.steps[0].state_after == {"status": "searching", "results": 2}
    assert trajectory.steps[1].state_before == {"status": "searching", "results": 2}


def test_action_arguments_are_preserved_exactly() -> None:
    trajectory = ToyRunAdapter().to_trajectory(make_record())

    assert trajectory.steps[0].action.args == {
        "origin": "SFO",
        "filters": {"cabins": ["economy"], "max_stops": 1},
    }
    assert trajectory.steps[1].action.args == {"flight": "TR420", "passengers": 1}
    assert trajectory.steps[2].action.args == {"method": "card"}


def test_run_metadata_is_preserved_without_interpretation() -> None:
    record = make_record()
    trajectory = ToyRunAdapter().to_trajectory(record)

    assert trajectory.metadata == {
        "model": "toy-agent-1",
        "stop_reason": "done",
        "source": {"harness": "synthetic", "attempt": 2},
    }

    # The trajectory owns copies: mutating the source record later cannot
    # rewrite what the evaluator will see.
    record["run_metadata"]["model"] = "mutated-after-conversion"
    record["events"][0]["before"]["status"] = "mutated-after-conversion"
    assert trajectory.metadata["model"] == "toy-agent-1"
    assert trajectory.steps[0].state_before == {"status": "new"}


def test_missing_states_use_an_explicit_marker_and_empty_state_stays_empty() -> None:
    record = make_record()
    record["events"] = [
        {
            # The source recorded neither snapshot.
            "action": {"name": "search_flights", "args": {"origin": "SFO"}},
        },
        {
            # This source snapshot really was empty -- not missing.
            "action": {"name": "noop", "args": {}},
            "before": {},
            "after": {},
        },
        {
            # Only one side was unavailable.
            "action": {"name": "pay", "args": {"method": "card"}},
            "before": {"status": "booked"},
        },
    ]
    trajectory = ToyRunAdapter().to_trajectory(record)

    first, second, third = trajectory.steps
    assert is_unavailable_state(first.state_before)
    assert first.state_before == unavailable_state()
    assert is_unavailable_state(first.state_after)
    assert first.state_before == first.state_after == unavailable_state()

    assert second.state_before == {}
    assert second.state_after == {}
    assert not is_unavailable_state(second.state_before)

    assert third.state_before == {"status": "booked"}
    assert is_unavailable_state(third.state_after)


def test_missing_run_metadata_does_not_invent_any() -> None:
    record = make_record()
    del record["run_metadata"]

    trajectory = ToyRunAdapter().to_trajectory(record)
    assert trajectory.metadata == {}


def test_malformed_top_level_record_raises_a_clear_error() -> None:
    with pytest.raises(AdapterError, match=r"external run: expected an object"):
        ToyRunAdapter().to_trajectory(["not", "a", "mapping"])


def test_missing_task_id_raises_a_clear_error() -> None:
    record = make_record()
    del record["run_id"]

    with pytest.raises(
        AdapterError, match=r"external run: 'task_id' must be a non-empty string"
    ):
        ToyRunAdapter().to_trajectory(record)


def test_malformed_events_raises_a_clear_error() -> None:
    record = make_record()
    record["events"] = {"not": "a list"}

    with pytest.raises(AdapterError, match=r"external run: 'events' must be a list"):
        ToyRunAdapter().to_trajectory(record)


def test_missing_action_name_raises_a_clear_error() -> None:
    record = make_record()
    del record["events"][0]["action"]["name"]

    with pytest.raises(
        AdapterError, match=r"external step 1: 'action_name' must be a non-empty string"
    ):
        ToyRunAdapter().to_trajectory(record)


def test_non_mapping_action_arguments_raise_a_clear_error() -> None:
    record = make_record()
    record["events"][0]["action"]["args"] = "origin=SFO"

    with pytest.raises(
        AdapterError, match=r"external step 1: 'action_args' must be a mapping"
    ):
        ToyRunAdapter().to_trajectory(record)


def test_non_mapping_state_raises_a_clear_error() -> None:
    record = make_record()
    record["events"][0]["before"] = ["not", "a", "state"]

    with pytest.raises(
        AdapterError,
        match=r"external step 1: 'state_before': expected a mapping or None",
    ):
        ToyRunAdapter().to_trajectory(record)


def test_non_mapping_metadata_raises_a_clear_error() -> None:
    record = make_record()
    record["run_metadata"] = "model=toy-agent-1"

    with pytest.raises(
        AdapterError, match=r"external run: 'metadata' must be a mapping"
    ):
        ToyRunAdapter().to_trajectory(record)


def test_source_state_may_not_use_the_reserved_unavailable_marker() -> None:
    record = make_record()
    record["events"][0]["after"] = {UNAVAILABLE_STATE_KEY: "unavailable"}

    with pytest.raises(AdapterError, match=r"reserved key"):
        ToyRunAdapter().to_trajectory(record)
