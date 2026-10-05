##basically my step 2 is validating and saving trajectory not is json(too messy risky) to a more structured format

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from trajecteval.errors import TrajectoryError

# The keys every trajectory file must have. Anything else at the top level is
# treated as extra metadata and swept into ``metadata``.
_KNOWN_TOP_LEVEL = ("task_id", "metadata", "steps")

# The keys every step object must have.
_STEP_REQUIRED_KEYS = ("index", "state_before", "action", "state_after")

# Every action must at least say what it did; "args" is optional and
# defaults to {} (only the *name* is structural).
_ACTION_REQUIRED_KEYS = ("name",)


#make sure its a dict and raise error if not
def _require_dict(value: Any, where: str) -> dict[str, Any]:
    """Narrow ``value`` to a dict or raise with *where* in the message."""
    if not isinstance(value, dict):
        raise TrajectoryError(f"{where}: expected an object, got {type(value).__name__}")
    return value

#make sure the dict has all the required keys and raise error if not
def _require_keys(obj: dict[str, Any], keys: tuple[str, ...], where: str) -> None:
    """Raise TrajectoryError naming ``where`` for the first missing key."""
    for key in keys:
        if key not in obj:
            raise TrajectoryError(f"{where}: missing '{key}'")

#this class represents an action taken by the agent at a specific step in the trajectory. It contains the name of the action and any associated arguments. The class provides methods to create an instance from a dictionary and to convert an instance back to a dictionary format.
@dataclass(frozen=True) ##here we freeze the dataclass to make it immutable, which is a good practice for data structures that represent a fixed state, like an action in a trajectory so that llm cannot later modify it 
class Action:
    """What the agent did at one step.

    ``args`` stays a plain dict on purpose: different tasks have different
    argument shapes, and typing them here would couple this universal
    structure to one domain's payload.
    """

    name: str
    args: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Any, where: str = "action") -> Action:
        data = _require_dict(data, where)
        _require_keys(data, _ACTION_REQUIRED_KEYS, where)
        # args, when present, must be a dict -- a stringified or numeric args
        # blob is a structural problem (we could never grade against it later).
        args = data.get("args", {})
        if not isinstance(args, dict):
            raise TrajectoryError(
                f"{where}: 'args' must be an object, got {type(args).__name__}"
            )
        return cls(name=data["name"], args=args)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "args": self.args}

#this class represents a single step in an episode of the trajectory. It contains the index of the step, the state of the environment before and after the action, and the action taken by the agent. The class provides methods to create an instance from a dictionary and to convert an instance back to a dictionary format.
@dataclass(frozen=True)
class Step:
    """One point in an episode: state before, action, state after."""

    index: int
    state_before: dict[str, Any]
    action: Action
    state_after: dict[str, Any]

    @classmethod
    def from_dict(cls, data: Any, position: int) -> Step:
        # position tell me which step this is in the list that had errors, so I can give a more helpful error message
        #index is for the step number in the trajectory, starting from 1. It is used to identify the step in error messages and to ensure that the steps are in the correct order.
        where = f"step {position + 1}"
        data = _require_dict(data, where)
        _require_keys(data, _STEP_REQUIRED_KEYS, where)

        index = data["index"]
        # bool is a subclass of int in Python (isinstance(True, int) is True),
        # so reject it explicitly before the int check.
        if isinstance(index, bool) or not isinstance(index, int):
            raise TrajectoryError(f"{where}: 'index' must be an int, got {type(index).__name__}")

        state_before = _require_dict(data["state_before"], f"{where}: 'state_before'")
        state_after = _require_dict(data["state_after"], f"{where}: 'state_after'")
        action = Action.from_dict(data["action"], where=f"{where}: 'action'")

        return cls(
            index=index,
            state_before=state_before,
            action=action,
            state_after=state_after,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "state_before": self.state_before,
            "action": self.action.to_dict(),
            "state_after": self.state_after,
        }


@dataclass(frozen=True)
class Discontinuity:
    """A mismatch between one step's state_after and the next step's
    state_before -- evidence the *recorder* that wrote the log is buggy.

    Reported (never raised) so the trace still loads and the anomaly can be
    inspected and included in a report.
    """

    step_index: int        # the step whose state_after disagrees
    next_step_index: int   # the following step, whose state_before disagrees
    differing_keys: list[str]  # top-level keys where the two snapshots differ

    def __str__(self) -> str:
        keys = ", ".join(self.differing_keys)
        return (
            f"steps {self.step_index}->{self.next_step_index} disagree in keys: {keys}"
        )


# Frozen on purpose: validation happens once, at load time. After that nothing
# may rebind task_id/steps/metadata -- graders must all see the same episode.
# steps is a tuple (not a list) because frozen only stops *rebinding*; making
# the container immutable too means the sequence itself can't be appended to.
# (Freezing is shallow: nested state dicts remain mutable.)
@dataclass(frozen=True)
class Trajectory:
    """A complete episode: the ordered steps an agent took for one task."""

    task_id: str
    steps: tuple[Step, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ load

    @classmethod
    def from_dict(cls, data: Any) -> Trajectory:
        """Parse + validate a plain dict into a Trajectory.

        Raises TrajectoryError only for structural problems
        """
        data = _require_dict(data, "trajectory")

        if "task_id" not in data or data["task_id"] is None:
            raise TrajectoryError("missing required field 'task_id'")
        task_id = data["task_id"]
        if not isinstance(task_id, str) or not task_id:
            raise TrajectoryError(
                f"'task_id' must be a non-empty string, got {type(task_id).__name__}"
            )

        if "steps" not in data:
            raise TrajectoryError("missing required field 'steps'")
        raw_steps = data["steps"]
        if not isinstance(raw_steps, list):
            raise TrajectoryError(
                f"'steps' must be a list, got {type(raw_steps).__name__}"
            )

        # Validate each step and ensure the indices are gapless and duplicate-free and make it into a list of Step objects from steps.
        steps = [Step.from_dict(raw, position=i) for i, raw in enumerate(raw_steps)]
        cls._validate_indices(steps)

        # get raw metadata from the data dict, if it exists. If it does not exist, use an empty dict. Then ensure that the raw metadata is a dict and raise an error if it is not.
        # Finally, add any extra top-level keys from the data dict to the metadata dict, unless they are already present in the metadata dict.
        raw_metadata = data.get("metadata", {})
        metadata: dict[str, Any] = (
            dict(_require_dict(raw_metadata, "'metadata'")) if raw_metadata is not None else {}
        )
        for key, value in data.items():
            if key in _KNOWN_TOP_LEVEL:
                continue
            metadata.setdefault(key, value)

        return cls(task_id=task_id, steps=tuple(steps), metadata=metadata)

    @classmethod
    def from_json_file(cls, path: str | Path) -> Trajectory:
        """Open a JSON file → read it → convert the JSON into a Python dictionary → validate it → create and return a Trajectory object"""
        path = Path(path)
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise TrajectoryError(f"cannot read {path}: {exc}") from exc
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise TrajectoryError(f"{path} is not valid JSON: {exc}") from exc
        return cls.from_dict(data)

    @staticmethod
    def _validate_indices(steps: list[Step]) -> None:
        """Enforce a gapless, duplicate-free 1-based index sequence."""
        expected = 1
        for step in steps:
            if step.index != expected:
                kind = (
                    "duplicate" if step.index in (s.index for s in steps[:expected - 1]) else "gap"
                )
                raise TrajectoryError(
                    f"step {expected}: index sequence has a {kind} -- "
                    f"expected {expected}, got {step.index}"
                )
            expected += 1

    # ----------------------------------------------------------------- query

    def find_discontinuities(self) -> list[Discontinuity]:
        """the state after of a step should be equa to state before of the next step, if not then dicontinuity is there."""
        mismatches: list[Discontinuity] = []
        for current, following in zip(self.steps, self.steps[1:]):
            if current.state_after == following.state_before:
                continue
            differing = sorted(
                {
                    *current.state_after.keys() - following.state_before.keys(),
                    *following.state_before.keys() - current.state_after.keys(),
                    *(k for k in current.state_after.keys() & following.state_before.keys()
                      if current.state_after[k] != following.state_before[k]),
                }
            )
            mismatches.append(
                Discontinuity(
                    step_index=current.index,
                    next_step_index=following.index,
                    differing_keys=differing,
                )
            )
        return mismatches

    # ----------------------------------------------------------------- save

    def to_dict(self) -> dict[str, Any]:
        """return the valiadted raw json data to now a structured format json"""
        out: dict[str, Any] = {"task_id": self.task_id}
        if self.metadata:
            out["metadata"] = self.metadata
        out["steps"] = [step.to_dict() for step in self.steps]
        return out

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
