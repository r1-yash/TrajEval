# Simulated worlds, one module per task. The Environment protocol and its
# rejection lane live in base.py; every task environment keeps that promise.
#
# The registry below is how a task id finds its world (and its agent
# instruction): Step 13's second task registers itself with one entry here.
# An unknown task id is a SpecError -- the CLI turns it into one clear line.

from __future__ import annotations

from trajecteval.envs.base import ActionRejected, Environment
from trajecteval.envs.travel_booking import (
    TRAVEL_BOOKING_INSTRUCTION,
    TravelBookingEnvironment,
)
from trajecteval.errors import SpecError

# task id -> (fresh-world factory, plain-text task instruction for an agent).
_ENVIRONMENTS: dict[str, tuple[type, str]] = {
    "travel_booking": (TravelBookingEnvironment, TRAVEL_BOOKING_INSTRUCTION),
}


def _lookup(task_id: str) -> tuple[type, str]:
    entry = _ENVIRONMENTS.get(task_id)
    if entry is None:
        raise SpecError(
            f"no simulated environment for task {task_id!r}; "
            f"known tasks: {', '.join(sorted(_ENVIRONMENTS))}"
        )
    return entry


def make_environment(task_id: str) -> Environment:
    # A FRESH world per call: one run always starts from the initial state.
    factory, _ = _lookup(task_id)
    return factory()


def agent_instruction(task_id: str) -> str:
    _, instruction = _lookup(task_id)
    return instruction


__all__ = [
    "ActionRejected",
    "Environment",
    "TRAVEL_BOOKING_INSTRUCTION",
    "TravelBookingEnvironment",
    "agent_instruction",
    "make_environment",
]
