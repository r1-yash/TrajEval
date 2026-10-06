"""Task specs: the rulebook, one plain JSON file per task.

Graders know *how* to check; a spec says *what* to check. A spec carries
three rule sections plus metadata:

- ``expected_final_state``    field -> value the final state must equal
- ``allowed_actions``         action -> per-argument permit lists
- ``critical_error_patterns`` exact moves that count as critical mistakes

Absent vs. empty is meaningful: a section set to ``None`` (omitted) means a
grader that needs it *cannot judge* and must return an ERROR result naming
the section; an empty ``{}`` / ``()`` means the section exists but lists
nothing to check.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from trajecteval.errors import SpecError
from trajecteval.models import Step

# The full key vocabulary. Anything else at the top level of a spec file is a
# typo -- rejected at load, not silently ignored. "metadata" is the extension
# valve for annotations that are not rules.
_KNOWN_KEYS = frozenset(
    {"task_id", "expected_final_state", "allowed_actions", "critical_error_patterns", "metadata"}
)

# A task id is a file name stem, never a path: no separators, no "..".
# Lowercase keeps ids stable across operating systems.
_TASK_ID_OK = frozenset("abcdefghijklmnopqrstuvwxyz0123456789_-")


def _require_object(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SpecError(f"{where}: expected an object, got {type(value).__name__}")
    return value


def _require_non_empty_str(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise SpecError(f"{where}: expected a non-empty string, got {value!r}")
    return value


def _require_task_id(task_id: Any) -> str:
    """Validate a task id before it is ever used to build a file path."""
    _require_non_empty_str(task_id, "task id")
    bad = [ch for ch in task_id if ch not in _TASK_ID_OK]
    if bad or task_id[0] in "_-":
        raise SpecError(
            f"invalid task id {task_id!r}: an id is a file name, not a path -- "
            "use only lowercase letters, digits, '_' or '-', starting with a letter or digit"
        )
    return task_id


@dataclass(frozen=True)
class CriticalPattern:
    """One forbidden move, pinned exactly.

    ``args`` matches *positively*: every key in the pattern must be present in
    the step's arguments with an equal value. An absent pattern's ``args``
    matches any call of the action; an absent *argument* never matches (not
    making a move is not making the forbidden move).
    """

    action: str
    args: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty_str(self.action, "critical pattern 'action'")
        _require_object(self.args, "critical pattern 'args'")

    def matches(self, step: Step) -> bool:
        if step.action.name != self.action:
            return False
        return all(
            key in step.action.args and step.action.args[key] == value
            for key, value in self.args.items()
        )

    def to_dict(self) -> dict[str, Any]:
        return {"action": self.action, "args": dict(self.args)}

    @classmethod
    def from_dict(cls, data: Any, where: str = "critical pattern") -> CriticalPattern:
        data = _require_object(data, where)
        if "action" not in data:
            raise SpecError(f"{where}: missing required field 'action'")
        unknown = sorted(set(data) - {"action", "args"})
        if unknown:
            raise SpecError(f"{where}: unknown key(s) {', '.join(repr(k) for k in unknown)}")
        args = data.get("args", {})
        _require_object(args, f"{where} 'args'")
        return cls(action=data["action"], args=args)


@dataclass(frozen=True)
class TaskSpec:
    """One task's rules, loaded from ``tasks/<task_id>.json``.

    Sections default to ``None`` (absent), which is distinct from an empty
    section: graders ask ``missing_section_result`` before judging.
    ``from_dict`` and ``__post_init__`` validate the same way, so a spec built
    from a file and one built in code obey one schema.
    """

    task_id: str
    expected_final_state: dict[str, Any] | None = None
    allowed_actions: dict[str, dict[str, list[Any]]] | None = None
    critical_error_patterns: tuple[CriticalPattern, ...] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_task_id(self.task_id)

        if self.expected_final_state is not None:
            _require_object(self.expected_final_state, "spec 'expected_final_state'")

        if self.allowed_actions is not None:
            _require_object(self.allowed_actions, "spec 'allowed_actions'")
            for action, permits in self.allowed_actions.items():
                _require_non_empty_str(action, "spec 'allowed_actions' key")
                _require_object(permits, f"spec 'allowed_actions'[{action!r}]")
                for arg, allowed in permits.items():
                    _require_non_empty_str(arg, f"spec 'allowed_actions'[{action!r}] key")
                    if not isinstance(allowed, list):
                        raise SpecError(
                            f"spec 'allowed_actions'[{action!r}][{arg!r}]: "
                            f"expected a list of permitted values, got {type(allowed).__name__}"
                        )

        if self.critical_error_patterns is not None:
            if not isinstance(self.critical_error_patterns, (list, tuple)):
                raise SpecError(
                    "spec 'critical_error_patterns': expected a list of patterns, "
                    f"got {type(self.critical_error_patterns).__name__}"
                )
            coerced: list[CriticalPattern] = []
            for position, item in enumerate(self.critical_error_patterns):
                if not isinstance(item, CriticalPattern):
                    raise SpecError(
                        f"spec 'critical_error_patterns[{position}]': "
                        f"expected a CriticalPattern, got {type(item).__name__}"
                    )
                coerced.append(item)
            object.__setattr__(self, "critical_error_patterns", tuple(coerced))

        if not isinstance(self.metadata, dict):
            raise SpecError(f"spec 'metadata': expected an object, got {type(self.metadata).__name__}")

    def to_dict(self) -> dict[str, Any]:
        """Round-trippable plain dict; absent sections stay absent (omitted)."""
        data: dict[str, Any] = {"task_id": self.task_id}
        if self.expected_final_state is not None:
            data["expected_final_state"] = dict(self.expected_final_state)
        if self.allowed_actions is not None:
            data["allowed_actions"] = {
                action: dict(permits) for action, permits in self.allowed_actions.items()
            }
        if self.critical_error_patterns is not None:
            data["critical_error_patterns"] = [p.to_dict() for p in self.critical_error_patterns]
        data["metadata"] = dict(self.metadata)
        return data

    @classmethod
    def from_dict(cls, data: Any) -> TaskSpec:
        data = _require_object(data, "spec")
        if "task_id" not in data:
            raise SpecError("spec: missing required field 'task_id'")
        unknown = sorted(set(data) - _KNOWN_KEYS)
        if unknown:
            raise SpecError(
                f"spec: unknown key(s) {', '.join(repr(k) for k in unknown)} -- "
                "known keys are task_id, expected_final_state, allowed_actions, "
                "critical_error_patterns, metadata"
            )

        patterns: tuple[CriticalPattern, ...] | None = None
        if data.get("critical_error_patterns") is not None:
            raw = data["critical_error_patterns"]
            if not isinstance(raw, list):
                raise SpecError(
                    "spec 'critical_error_patterns': expected a list of patterns, "
                    f"got {type(raw).__name__}"
                )
            patterns = tuple(
                CriticalPattern.from_dict(item, where=f"spec 'critical_error_patterns[{position}]'")
                for position, item in enumerate(raw)
            )

        metadata = data.get("metadata", {})
        _require_object(metadata, "spec 'metadata'")

        return cls(
            task_id=data["task_id"],
            expected_final_state=data.get("expected_final_state"),
            allowed_actions=data.get("allowed_actions"),
            critical_error_patterns=patterns,
            metadata=metadata,
        )


def load_spec(task_id: str, tasks_dir: str | Path | None = None) -> TaskSpec:
    """Load ``tasks/<task_id>.json``. No registry: add a file, add a task.

    ``tasks_dir`` defaults to a ``tasks`` folder in the current directory.
    File problems (missing file, invalid JSON) and schema problems both raise
    ``SpecError`` -- "the rulebook you pointed me at is broken".
    """
    _require_task_id(task_id)
    directory = Path(tasks_dir) if tasks_dir is not None else Path.cwd() / "tasks"
    path = directory / f"{task_id}.json"

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SpecError(f"cannot read {path}: {exc}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SpecError(f"{path} is not valid JSON: {exc}") from exc
    return TaskSpec.from_dict(data)
