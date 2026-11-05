"""Adapter for real Debuggernaut benchmark run records (schema version 4).

Debuggernaut records separate three kinds of information, and this adapter
keeps that separation:

* ``attempt_records[*].exploration`` is the actual structured tool history.
  Each entry (``tool``, ``arguments``, ``ok``, ``result_size``, ...) becomes
  one TrajEval ``Step``, in source order.
* ``calls`` is provider/LLM accounting, not tool history. It is preserved in
  trajectory metadata and is never converted into an action.
* attempt outcomes -- the proposed file/diff, test result, diagnosis, proposal
  validation attempts, guard events, and proposal failures -- are evidence,
  not agent tool calls. They stay in trajectory metadata, where each attempt
  records the ``step_indices`` its exploration produced.

Debuggernaut does not record repository snapshots around tool calls, so every
converted step uses the generic adapter's explicit unavailable-state marker
(``None`` at the ``ExternalStep`` boundary). No file or repository state is
reconstructed from tool names, results, diffs, or test output.

The task id is derived from the real ``repository`` and ``issue`` fields as a
stable, lowercase TrajEval id (for example
``pre-commit__pre-commit-hooks__1038``). The original repository and issue
remain in metadata.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from typing import Any

from trajecteval.adapters.offline import ExternalStep, OfflineRunAdapter
from trajecteval.errors import AdapterError

# Same id vocabulary as TaskSpec: a task id is a file name stem, not a path.
_TASK_ID_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789_-")


class DebuggernautAdapter(OfflineRunAdapter):
    """Translate one already-loaded Debuggernaut run record.

    This is an offline boundary. It performs no I/O, starts no agent, calls no
    provider, and never reads Debuggernaut proposal artifacts referenced by
    the record.
    """

    def extract_task_id(self, record: Mapping[str, Any]) -> str:
        repository = _require_non_empty_str(
            record.get("repository"), "external run: 'repository'"
        )
        issue = record.get("issue")
        if isinstance(issue, bool) or not isinstance(issue, int) or issue < 1:
            raise AdapterError(
                f"external run: 'issue' must be a positive int, "
                f"got {type(issue).__name__}"
            )

        slug = repository.lower().replace("/", "__")
        slug = re.sub(r"[^a-z0-9_-]+", "_", slug).strip("_-")
        if not slug or not set(slug) <= _TASK_ID_CHARS:
            raise AdapterError(
                f"external run: 'repository' {repository!r} cannot form a task id"
            )
        return f"{slug}__{issue}"

    def extract_steps(self, record: Mapping[str, Any]) -> list[ExternalStep]:
        steps: list[ExternalStep] = []
        for attempt_position, attempt in enumerate(
            _attempt_records(record), start=1
        ):
            where = f"attempt record {attempt_position}"
            for action_position, entry in enumerate(
                _exploration(attempt, where), start=1
            ):
                action_where = f"{where}: exploration {action_position}"
                steps.append(
                    ExternalStep(
                        action_name=_tool_name(entry, action_where),
                        action_args=_arguments(entry, action_where),
                        # Debuggernaut records tool metadata, not repository
                        # snapshots. Never reconstruct state from tool output.
                        state_before=None,
                        state_after=None,
                    )
                )
        return steps

    def extract_metadata(self, record: Mapping[str, Any]) -> dict[str, Any]:
        # Every top-level field is useful run metadata; attempt_records is
        # replaced with a version whose exploration entries carry step_index.
        metadata = copy.deepcopy(
            {key: value for key, value in record.items() if key != "attempt_records"}
        )

        attempts: list[dict[str, Any]] = []
        next_step_index = 1
        for attempt_position, attempt in enumerate(
            _attempt_records(record), start=1
        ):
            where = f"attempt record {attempt_position}"
            attempt_metadata = copy.deepcopy(dict(attempt))
            exploration = _exploration(attempt, where)
            step_indices: list[int] = []

            if attempt.get("exploration") is not None:
                enriched_exploration: list[dict[str, Any]] = []
                for action_position, entry in enumerate(exploration, start=1):
                    action_where = f"{where}: exploration {action_position}"
                    # Validate exactly as extract_steps does, so metadata and
                    # steps can never describe different exploration records.
                    _tool_name(entry, action_where)
                    _arguments(entry, action_where)
                    step_index = next_step_index
                    next_step_index += 1
                    step_indices.append(step_index)
                    enriched = copy.deepcopy(dict(entry))
                    enriched["step_index"] = step_index
                    enriched_exploration.append(enriched)
                attempt_metadata["exploration"] = enriched_exploration

            attempt_metadata["step_indices"] = step_indices
            attempts.append(attempt_metadata)

        if "attempt_records" in record:
            metadata["attempt_records"] = attempts
        return metadata


def _attempt_records(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return the source attempt records, or none when the field is absent."""
    raw = record.get("attempt_records", [])
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise AdapterError(
            f"external run: 'attempt_records' must be a list, "
            f"got {type(raw).__name__}"
        )
    attempts: list[Mapping[str, Any]] = []
    for position, attempt in enumerate(raw, start=1):
        if not isinstance(attempt, Mapping):
            raise AdapterError(
                f"attempt record {position}: expected an object, "
                f"got {type(attempt).__name__}"
            )
        attempt_number = attempt.get("attempt")
        if isinstance(attempt_number, bool) or not isinstance(attempt_number, int):
            raise AdapterError(
                f"attempt record {position}: 'attempt' must be an int, "
                f"got {type(attempt_number).__name__}"
            )
        attempts.append(attempt)
    return attempts


def _exploration(
    attempt: Mapping[str, Any], where: str
) -> list[Mapping[str, Any]]:
    """Return one attempt's structured tool history; absent means none."""
    raw = attempt.get("exploration", [])
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise AdapterError(
            f"{where}: 'exploration' must be a list, got {type(raw).__name__}"
        )
    entries: list[Mapping[str, Any]] = []
    for position, entry in enumerate(raw, start=1):
        if not isinstance(entry, Mapping):
            raise AdapterError(
                f"{where}: exploration {position}: expected an object, "
                f"got {type(entry).__name__}"
            )
        entries.append(entry)
    return entries


def _tool_name(entry: Mapping[str, Any], where: str) -> str:
    tool = entry.get("tool")
    if not isinstance(tool, str) or not tool:
        raise AdapterError(
            f"{where}: 'tool' must be a non-empty string, "
            f"got {type(tool).__name__}"
        )
    return tool


def _arguments(entry: Mapping[str, Any], where: str) -> Mapping[str, Any]:
    arguments = entry.get("arguments", {})
    if not isinstance(arguments, Mapping):
        raise AdapterError(
            f"{where}: 'arguments' must be a mapping, "
            f"got {type(arguments).__name__}"
        )
    return arguments


def _require_non_empty_str(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise AdapterError(
            f"{where} must be a non-empty string, got {type(value).__name__}"
        )
    return value
