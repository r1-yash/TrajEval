# The environment interface: what the recorder needs from ANY simulated world.
#
# An Environment is the world the agent acts in. The recorder (recorder.py)
# is environment-agnostic -- it works with any object that keeps this
# promise, so task two (Step 13) gets a new environment and reuses the
# recorder unchanged. Two kinds of failure live apart on purpose:
#
# - ActionRejected -- the world CANNOT do this (unknown action name, paying
#   before anything is booked). Normal gameplay: the environment guarantees
#   the state is unchanged, and the recorder keeps the attempt as a step.
#   Deliberately NOT a TrajEvalError: that lane means "an input file is
#   structurally broken", and nothing here is broken.
# - A genuine bug (e.g. a handler crashing on malformed args) propagates as
#   usual -- it is a bug, not a rejection.

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from trajecteval.models import Action


class ActionRejected(Exception):
    """The world cannot physically perform this action; state unchanged."""


@runtime_checkable
class Environment(Protocol):
    # The smallest promise the recorder needs: which task this world is, a
    # snapshot of its state, and one action applied. Protocol bodies never
    # run (same style as the Grader protocol in graders/base.py).

    task_id: str

    def current_state(self) -> dict[str, Any]:
        # A snapshot of the world right now; an independent copy, so a
        # caller can never reach into the world's internals.
        ...  # pragma: no cover - protocol body, never executed

    def run_action(self, action: Action) -> dict[str, Any]:
        # Apply one action and return the new state. Raises ActionRejected
        # (with the state guaranteed unchanged) when physically impossible.
        ...  # pragma: no cover - protocol body, never executed
