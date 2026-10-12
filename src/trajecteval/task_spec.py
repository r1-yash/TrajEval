

# Task specs: the rulebook -- one JSON file per task; sections are absent (ERROR), empty (WARN "no rules to check"), or judged.

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from trajecteval.errors import SpecError
from trajecteval.matching import contains_failures, json_contains, json_equal

# The three matchers live in matching.py (models.py needs json_equal too;
# importing them back here keeps every existing `from
# trajecteval.task_spec import json_equal` working unchanged).
from trajecteval.models import Step

# The rule sections -- the three things a spec can be missing. ``_KNOWN_KEYS``
# below derives from this, and graders validate section names against it
# (a typo'd section must raise ValueError, not crash getattr or silently
# judge nothing).
RULE_SECTIONS = (
    "expected_final_state",
    "allowed_actions",
    "critical_error_patterns",
)

# The full key vocabulary. Anything else at the top level of a spec file is a
# typo -- rejected at load, not silently ignored. "metadata" is the extension
# valve for annotations that are not rules.
_KNOWN_KEYS = frozenset((*RULE_SECTIONS, "task_id", "metadata"))

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
    # Validate a task id before it is ever used to build a file path.
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
    # One forbidden move: action + args matched positively (absent args match any call); description narrates why, never parsed.

    action: str
    args: dict[str, Any] = field(default_factory=dict)
    description: str | None = None

    def __post_init__(self) -> None:
        _require_non_empty_str(self.action, "critical pattern 'action'")
        _require_object(self.args, "critical pattern 'args'")
        if self.description is not None:
            _require_non_empty_str(self.description, "critical pattern 'description'")

    def matches(self, step: Step) -> bool:
        if step.action.name != self.action:
            return False
        return all(
            key in step.action.args and json_equal(step.action.args[key], value)
            for key, value in self.args.items()
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"action": self.action, "args": dict(self.args)}
        if self.description is not None:
            data["description"] = self.description
        return data

    @classmethod
    def from_dict(cls, data: Any, where: str = "critical pattern") -> CriticalPattern:
        data = _require_object(data, where)
        if "action" not in data:
            raise SpecError(f"{where}: missing required field 'action'")
        unknown = sorted(set(data) - {"action", "args", "description"})
        if unknown:
            raise SpecError(f"{where}: unknown key(s) {', '.join(repr(k) for k in unknown)}")
        args = data.get("args", {})
        _require_object(args, f"{where} 'args'")
        return cls(action=data["action"], args=args, description=data.get("description"))


@dataclass(frozen=True)
class TaskSpec:
    # One task's rules; sections default to None (absent) vs {} (empty), validated the same in __post_init__ and from_dict.

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

    def action_allowed(self, action: str) -> bool:
        # Whether allowed_actions lists this action; ValueError when the section is absent (call missing_section_result first).
        if self.allowed_actions is None:
            raise ValueError(
                f"allowed_actions section is absent for task {self.task_id!r}; "
                "call missing_section_result(...) first"
            )
        return action in self.allowed_actions

    def argument_violations(
        self, action: str, args: dict[str, Any]
    ) -> tuple[tuple[str, Any], ...]:
        # Supplied arguments the action's permit lists forbid: ((arg, value), ...); needs the section present and the action listed.
        if self.allowed_actions is None:
            raise ValueError(
                f"allowed_actions section is absent for task {self.task_id!r}; "
                "call missing_section_result(...) first"
            )
        if action not in self.allowed_actions:
            raise ValueError(
                f"action {action!r} is not listed in allowed_actions; "
                "check action_allowed(...) first"
            )
        permits = self.allowed_actions[action]
        violations: list[tuple[str, Any]] = []
        for arg, value in args.items():
            allowed_values = permits.get(arg)
            if allowed_values is None:  # arg not in the permit map -> unconstrained
                continue
            # Membership through the shared matcher, not `in`/`==`: in
            # Python ``True in [1]`` is True, which would let a boolean
            # sneak past a numeric permit list.
            if not any(json_equal(value, allowed) for allowed in allowed_values):
                violations.append((arg, value))
        return tuple(violations)

    def to_dict(self) -> dict[str, Any]:
        # Round-trippable plain dict; absent sections stay absent (omitted).
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

        raw_metadata = data.get("metadata")  # absent and explicit null both mean omitted
        metadata = {} if raw_metadata is None else raw_metadata
        _require_object(metadata, "spec 'metadata'")

        return cls(
            task_id=data["task_id"],
            expected_final_state=data.get("expected_final_state"),
            allowed_actions=data.get("allowed_actions"),
            critical_error_patterns=patterns,
            metadata=metadata,
        )


def load_spec(task_id: str, tasks_dir: str | Path | None = None) -> TaskSpec:
    # Load tasks/<task_id>.json (no registry: add a file, add a task); broken files and a task_id mismatch raise SpecError.
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
    spec = TaskSpec.from_dict(data)
    if spec.task_id != task_id:
        raise SpecError(
            f"task id mismatch: {path} declares task_id {spec.task_id!r} "
            f"but was loaded as {task_id!r}"
        )
    return spec
