"""Offline translation of already-loaded external agent runs into trajectories.

This is a *boundary*, not an evaluator. A format-specific subclass knows how
one external run format stores its fields and implements the three
``extract_*`` hooks; the base class does the shared, format-neutral work:

* accept an already-loaded mapping -- no HTTP, model, agent, or external
  repository access happens here;
* assign the canonical gapless 1-based ``Step.index`` in sequence order
  (the external record's own numbering, if any, is deliberately not trusted);
* deep-copy actions, states, and metadata so the resulting trajectory does
  not alias the source record;
* build the existing :class:`trajecteval.models.Trajectory` through
  ``Trajectory.from_dict`` so there is one trajectory model and one validator;
* reject malformed records with :class:`trajecteval.errors.AdapterError`.

It never decides PASS/FAIL, reads a task spec, or interprets an action.

Unavailable state
-----------------
The existing ``Step`` model requires ``state_before`` and ``state_after`` to
be dicts. An external record may simply not contain a state snapshot. At the
normalized boundary that is represented by ``ExternalStep.state_before`` /
``ExternalStep.state_after`` being ``None`` (the provider omits it). The
converted trajectory uses an explicit one-key marker instead of ``{}``:

    {"__trajecteval_unavailable_state__": "unavailable"}

``{}`` is deliberately reserved for a state the source *did* record and that
really was empty. The marker key is reserved: a source state containing it is
rejected rather than silently mistaken for "unavailable". Callers can use
``is_unavailable_state()`` rather than comparing the dict by hand.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from trajecteval.errors import AdapterError, TrajectoryError
from trajecteval.models import Trajectory

# Reserved key for the "the source did not record this state" marker. A real
# source state must never use it (checked in _state_dict below), so the marker
# can never be confused with recorded data.
UNAVAILABLE_STATE_KEY = "__trajecteval_unavailable_state__"
_UNAVAILABLE_STATE_VALUE = "unavailable"


def unavailable_state() -> dict[str, Any]:
    """Return a fresh marker dict for a state the source did not record."""
    return {UNAVAILABLE_STATE_KEY: _UNAVAILABLE_STATE_VALUE}


def is_unavailable_state(state: Mapping[str, Any]) -> bool:
    """Whether ``state`` is exactly the unavailable-state marker."""
    return (
        isinstance(state, Mapping)
        and len(state) == 1
        and state.get(UNAVAILABLE_STATE_KEY) == _UNAVAILABLE_STATE_VALUE
    )


@dataclass(frozen=True)
class ExternalStep:
    """One action from an external run, normalized but not yet a ``Step``.

    ``state_before`` / ``state_after`` are ``None`` when the source record did
    not contain that snapshot. An empty dict means the source recorded a
    genuinely empty state. The base adapter assigns the final 1-based index.
    """

    action_name: str
    action_args: Mapping[str, Any] = field(default_factory=dict)
    state_before: Mapping[str, Any] | None = None
    state_after: Mapping[str, Any] | None = None


class OfflineRunAdapter(ABC):
    """Translate one already-loaded external run record into a ``Trajectory``.

    Subclasses are format-specific: they only pull fields out of their own
    record shape. They must not grade, execute anything, or reach the network.
    """

    def to_trajectory(self, record: Mapping[str, Any]) -> Trajectory:
        """Convert one already-loaded run record into the existing model.

        The returned ``Trajectory`` is validated by ``Trajectory.from_dict``.
        Raises ``AdapterError`` when the record is malformed or when the
        extracted values cannot form a valid trajectory.
        """
        if not isinstance(record, Mapping):
            raise AdapterError(
                f"external run: expected an object, got {type(record).__name__}"
            )

        task_id = self.extract_task_id(record)
        if not isinstance(task_id, str) or not task_id:
            raise AdapterError(
                f"external run: 'task_id' must be a non-empty string, "
                f"got {type(task_id).__name__}"
            )

        raw_steps = self.extract_steps(record)
        if isinstance(raw_steps, (str, bytes, bytearray)) or not isinstance(
            raw_steps, Sequence
        ):
            raise AdapterError(
                f"external run: 'steps' must be a sequence of ExternalStep "
                f"objects, got {type(raw_steps).__name__}"
            )

        raw_metadata = self.extract_metadata(record)
        if raw_metadata is None:
            metadata: dict[str, Any] = {}
        elif not isinstance(raw_metadata, Mapping):
            raise AdapterError(
                f"external run: 'metadata' must be a mapping, "
                f"got {type(raw_metadata).__name__}"
            )
        else:
            metadata = copy.deepcopy(dict(raw_metadata))

        steps: list[dict[str, Any]] = []
        for position, external_step in enumerate(raw_steps, start=1):
            where = f"external step {position}"
            if not isinstance(external_step, ExternalStep):
                raise AdapterError(
                    f"{where}: expected an ExternalStep, "
                    f"got {type(external_step).__name__}"
                )
            if not isinstance(external_step.action_name, str) or not (
                external_step.action_name
            ):
                raise AdapterError(
                    f"{where}: 'action_name' must be a non-empty string, "
                    f"got {type(external_step.action_name).__name__}"
                )
            if not isinstance(external_step.action_args, Mapping):
                raise AdapterError(
                    f"{where}: 'action_args' must be a mapping, "
                    f"got {type(external_step.action_args).__name__}"
                )

            steps.append(
                {
                    # External ordering wins; the Trajectory contract is a
                    # gapless 1-based sequence, which the adapter owns.
                    "index": position,
                    "state_before": _state_dict(
                        external_step.state_before, f"{where}: 'state_before'"
                    ),
                    "action": {
                        "name": external_step.action_name,
                        # Deep-copied so later mutation of the source record
                        # cannot rewrite the converted trajectory.
                        "args": copy.deepcopy(dict(external_step.action_args)),
                    },
                    "state_after": _state_dict(
                        external_step.state_after, f"{where}: 'state_after'"
                    ),
                }
            )

        data = {"task_id": task_id, "metadata": metadata, "steps": steps}
        try:
            return Trajectory.from_dict(data)
        except TrajectoryError as exc:
            raise AdapterError(
                f"external run cannot be converted into a trajectory: {exc}"
            ) from exc

    @abstractmethod
    def extract_task_id(self, record: Mapping[str, Any]) -> str:
        """Return the task id from this format's record."""

    @abstractmethod
    def extract_steps(self, record: Mapping[str, Any]) -> Sequence[ExternalStep]:
        """Return this format's actions in execution order."""

    def extract_metadata(self, record: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return useful run metadata from this format's record.

        The default preserves no metadata; format-specific adapters should
        override this when their record carries run-level information such as
        the model, stop reason, or source identifiers.
        """
        return {}


def _state_dict(value: Mapping[str, Any] | None, where: str) -> dict[str, Any]:
    """Copy one external state, mapping ``None`` to the explicit marker."""
    if value is None:
        return unavailable_state()
    if not isinstance(value, Mapping):
        raise AdapterError(
            f"{where}: expected a mapping or None, got {type(value).__name__}"
        )
    # The marker is reserved vocabulary, not real state: reject a source
    # state that uses it so "unavailable" can never masquerade as data.
    if UNAVAILABLE_STATE_KEY in value:
        raise AdapterError(
            f"{where}: source state uses reserved key {UNAVAILABLE_STATE_KEY!r}; "
            "omit the state to mark it unavailable"
        )
    return copy.deepcopy(dict(value))
