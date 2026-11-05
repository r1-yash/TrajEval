"""Tests for the real-format Debuggernaut offline adapter.

Fixtures are verbatim copies of actual Debuggernaut benchmark records; see
``tests/fixtures/debuggernaut/README.md`` for the source paths. The tests also
derive incomplete variants by deleting fields from those real records, so the
older partially recorded failures are covered without inventing a schema.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from trajecteval import (
    AdapterError,
    DebuggernautAdapter,
    Trajectory,
    is_unavailable_state,
    unavailable_state,
)

FIXTURES = Path(__file__).parent / "fixtures" / "debuggernaut"


def load_fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def flat_exploration(
    record: dict[str, Any],
) -> list[tuple[int, str, dict[str, Any]]]:
    return [
        (attempt["attempt"], entry["tool"], entry["arguments"])
        for attempt in record.get("attempt_records", [])
        for entry in attempt.get("exploration", [])
    ]


def test_successful_debuggernaut_run_converts_to_trajectory() -> None:
    record = load_fixture("success_pre_commit_hooks_1038.json")

    trajectory = DebuggernautAdapter().to_trajectory(record)

    assert isinstance(trajectory, Trajectory)
    assert trajectory.task_id == "pre-commit__pre-commit-hooks__1038"
    assert len(trajectory.steps) == 10
    assert Trajectory.from_dict(trajectory.to_dict()) == trajectory


def test_exploration_actions_keep_source_order() -> None:
    record = load_fixture("success_pre_commit_hooks_1038.json")
    trajectory = DebuggernautAdapter().to_trajectory(record)

    assert [step.action.name for step in trajectory.steps] == [
        "list_directory",
        "search_code",
        "read_file",
        "list_directory",
        "read_file",
        "search_code",
        "read_file",
        "read_file",
        "read_file",
        "read_file",
    ]
    assert [
        (step.action.name, step.action.args) for step in trajectory.steps
    ] == [(tool, args) for _, tool, args in flat_exploration(record)]


def test_tool_arguments_are_preserved_exactly() -> None:
    trajectory = DebuggernautAdapter().to_trajectory(
        load_fixture("success_pre_commit_hooks_1038.json")
    )

    assert trajectory.steps[0].action.args == {"dir_path": "."}
    assert trajectory.steps[1].action.args == {"query": "pretty-format-json"}
    assert trajectory.steps[2].action.args == {
        "path": "pre_commit_hooks/pretty_format_json.py"
    }
    assert trajectory.steps[4].action.args == {
        "path": "tests/pretty_format_json_test.py"
    }


def test_useful_run_metadata_is_preserved_at_trajectory_level() -> None:
    record = load_fixture("success_pre_commit_hooks_1038.json")
    trajectory = DebuggernautAdapter().to_trajectory(record)
    metadata = trajectory.metadata

    assert metadata["schema_version"] == 4
    assert metadata["repository"] == "pre-commit/pre-commit-hooks"
    assert metadata["issue"] == 1038
    assert metadata["commit"] == "8c24e2c2e6b964feb04e1a93a72200ee84bd115a"
    assert metadata["provider"] == "tokenharbor"
    assert metadata["base_url_host"] == "tokenharbor.ai"
    assert metadata["model"] == "deepseek-v4.1-flash:free"
    assert metadata["status"] == "success"
    assert metadata["failure_class"] is None
    assert metadata["attempts"] == 1
    assert metadata["exploration_rounds"] == 8
    assert metadata["llm_calls"] == 9
    assert metadata["input_tokens"] == 77320
    assert metadata["output_tokens"] == 3484
    assert metadata["total_tokens"] == 80804
    assert metadata["test_patch"] == {
        "applied": True,
        "file_paths": ["tests/pretty_format_json_test.py"],
    }
    assert metadata["baseline"]["status"] == "fails_as_expected"
    assert metadata["environment"]["python_version"] == "3.11.17"
    assert len(metadata["calls"]) == 9
    assert metadata["calls"][0]["purpose"] == "exploration"
    assert metadata["guard_events"] == []

    attempt = metadata["attempt_records"][0]
    assert attempt["attempt"] == 1
    assert attempt["file_path"] == "pre_commit_hooks/pretty_format_json.py"
    assert attempt["step_indices"] == list(range(1, 11))
    assert [
        entry["step_index"] for entry in attempt["exploration"]
    ] == list(range(1, 11))
    assert attempt["proposal_artifact"]["path"] == (
        "artifacts/20261102T144640Z_pre-commit-pre-commit-hooks_1038/"
        "attempt-1-proposal.json"
    )


def test_multiple_attempts_remain_distinguishable() -> None:
    record = load_fixture("multi_attempt_mashumaro_184.json")
    trajectory = DebuggernautAdapter().to_trajectory(record)
    attempts = trajectory.metadata["attempt_records"]

    assert trajectory.task_id == "fatal1ty__mashumaro__184"
    assert [attempt["attempt"] for attempt in attempts] == [1, 2, 3]
    assert [len(attempt["step_indices"]) for attempt in attempts] == [13, 16, 15]
    assert len(trajectory.steps) == 44
    assert [
        (step.action.name, step.action.args) for step in trajectory.steps
    ] == [(tool, args) for _, tool, args in flat_exploration(record)]

    expected_start = 1
    for attempt in attempts:
        expected_indices = list(
            range(expected_start, expected_start + len(attempt["step_indices"]))
        )
        assert attempt["step_indices"] == expected_indices
        assert [
            entry["step_index"] for entry in attempt["exploration"]
        ] == expected_indices
        expected_start += len(expected_indices)

    # The first action of attempts 2 and 3 starts at step 14 and step 30.
    assert trajectory.steps[13].action.name == "read_file"
    assert trajectory.steps[29].action.name == "read_file"


def test_test_outcomes_and_diagnosis_are_preserved() -> None:
    record = load_fixture("multi_attempt_mashumaro_184.json")
    trajectory = DebuggernautAdapter().to_trajectory(record)
    attempts = trajectory.metadata["attempt_records"]

    first = attempts[0]
    assert first["test"]["passed"] is False
    assert first["test"]["return_code"] == 2
    assert first["test"]["timed_out"] is False
    assert first["test"]["started"] is True
    assert "IndentationError" in first["test"]["output_tail"]
    assert first["diagnosis"]["reason"] == "wrong_fix"
    assert "invalid" in first["diagnosis"]["explanation"]

    assert attempts[1]["diagnosis"]["reason"] == "wrong_fix"
    assert "incomplete/incorrect" in attempts[1]["diagnosis"]["explanation"]
    assert attempts[2]["diagnosis"] is None


def test_proposal_and_fix_information_are_preserved() -> None:
    record = load_fixture("multi_attempt_mashumaro_184.json")
    trajectory = DebuggernautAdapter().to_trajectory(record)
    first = trajectory.metadata["attempt_records"][0]

    assert first["file_path"] == "mashumaro/core/meta/types/unpack.py"
    assert "DiscriminatedUnionUnpackerBuilder" in first["reasoning"]
    assert first["diff"].startswith("--- a/mashumaro/core/meta/types/unpack.py")
    assert first["diff_truncated"] is True
    assert first["exploration_rounds"] == 10
    assert first["exploration_truncated"] is False
    assert first["avoid_file_paths_after"] == []
    assert first["proposal_attempts"] == [
        {
            "attempt": 1,
            "valid": True,
            "raw_response_present": True,
            "parsed_proposal_present": True,
            "validation_error": None,
        }
    ]
    assert first["proposal_artifact"]["includes"] == [
        "original_content",
        "new_content",
        "proposal_attempts",
    ]


def test_every_step_uses_the_explicit_unavailable_state_marker() -> None:
    trajectory = DebuggernautAdapter().to_trajectory(
        load_fixture("success_pre_commit_hooks_1038.json")
    )

    assert trajectory.steps
    for step in trajectory.steps:
        assert is_unavailable_state(step.state_before)
        assert is_unavailable_state(step.state_after)
        assert step.state_before == unavailable_state()
        assert step.state_after == unavailable_state()
        # An empty dict would claim Debuggernaut recorded a genuinely empty
        # repository state, which these records never do.
        assert step.state_before != {}
        assert step.state_after != {}


def test_proposal_failure_record_has_no_fabricated_tool_steps() -> None:
    record = load_fixture("proposal_failure_sqlparse_763.json")
    trajectory = DebuggernautAdapter().to_trajectory(record)
    metadata = trajectory.metadata

    # calls[] contains provider/LLM records, not structured exploration. An
    # empty attempt_records list therefore means no actual tool history.
    assert trajectory.steps == ()
    assert len(metadata["calls"]) == 12
    assert metadata["attempt_records"] == []
    assert metadata["attempts"] == 0
    assert metadata["failure_class"] == "agent"
    assert "Fix proposal failed schema validation" in metadata["status_reason"]

    failure = metadata["proposal_failures"][0]
    assert failure["attempt"] == 1
    assert [item["valid"] for item in failure["proposal_attempts"]] == [
        False,
        False,
        False,
    ]
    assert failure["proposal_attempts"][0]["validation_error"] == (
        "new_content: Field required; reasoning: Field required"
    )
    assert failure["proposal_artifact"]["path"] == (
        "artifacts/20261102T140451Z_andialbrecht-sqlparse_763/"
        "proposal-failure-1.json"
    )


def test_older_record_without_exploration_rounds_still_converts() -> None:
    record = load_fixture("toolcall_limit_sqlparse_763.json")
    trajectory = DebuggernautAdapter().to_trajectory(record)
    metadata = trajectory.metadata

    assert trajectory.steps == ()
    assert metadata["attempt_records"] == []
    assert "exploration_rounds" not in metadata
    assert metadata["llm_calls"] == 4
    assert len(metadata["calls"]) == 4
    assert "tool-call limit" in metadata["status_reason"]


def test_missing_exploration_fields_do_not_crash_conversion() -> None:
    record = load_fixture("success_pre_commit_hooks_1038.json")
    del record["attempt_records"][0]["exploration"]

    trajectory = DebuggernautAdapter().to_trajectory(record)

    assert trajectory.steps == ()
    attempt = trajectory.metadata["attempt_records"][0]
    assert attempt["step_indices"] == []
    assert "exploration" not in attempt

    record_without_attempts = load_fixture("success_pre_commit_hooks_1038.json")
    del record_without_attempts["attempt_records"]
    trajectory_without_attempts = DebuggernautAdapter().to_trajectory(
        record_without_attempts
    )
    assert trajectory_without_attempts.steps == ()
    assert "attempt_records" not in trajectory_without_attempts.metadata


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        pytest.param(
            lambda record: record.pop("repository"),
            r"external run: 'repository' must be a non-empty string",
            id="missing-repository",
        ),
        pytest.param(
            lambda record: record.__setitem__("issue", "1038"),
            r"external run: 'issue' must be a positive int",
            id="non-integer-issue",
        ),
        pytest.param(
            lambda record: record.__setitem__("attempt_records", {}),
            r"external run: 'attempt_records' must be a list",
            id="attempt-records-not-list",
        ),
        pytest.param(
            lambda record: record["attempt_records"][0].pop("attempt"),
            r"attempt record 1: 'attempt' must be an int",
            id="missing-attempt-number",
        ),
        pytest.param(
            lambda record: record["attempt_records"][0].__setitem__(
                "exploration", "not-a-list"
            ),
            r"attempt record 1: 'exploration' must be a list",
            id="exploration-not-list",
        ),
        pytest.param(
            lambda record: record["attempt_records"][0]["exploration"][0].pop(
                "tool"
            ),
            r"attempt record 1: exploration 1: 'tool' must be a non-empty string",
            id="missing-tool-name",
        ),
        pytest.param(
            lambda record: record["attempt_records"][0]["exploration"][
                0
            ].__setitem__("arguments", "not-a-mapping"),
            r"attempt record 1: exploration 1: 'arguments' must be a mapping",
            id="arguments-not-mapping",
        ),
    ],
)
def test_malformed_debuggernaut_records_raise_adapter_error(
    mutate: Any, message: str
) -> None:
    record = copy.deepcopy(load_fixture("success_pre_commit_hooks_1038.json"))
    mutate(record)

    with pytest.raises(AdapterError, match=message):
        DebuggernautAdapter().to_trajectory(record)
