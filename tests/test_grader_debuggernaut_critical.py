"""Tests for Debuggernaut critical grading from recorded guard events.

The fixtures are real Debuggernaut benchmark records; guard events are added
in the exact shape ``write_run_report`` serializes (``attempt``,
``file_path``, ``kind``, ``reason``, ``proposal_attempts``, ``exploration``,
``exploration_truncated``, ``proposal_artifact``). No new Debuggernaut schema
is invented.

A guard event is a proposal rejected before any repository file was modified.
``test_file`` and ``test_patch_file`` are the real prohibited-modification
kinds; ``nonexistent_path`` is a retryable validation miss.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from trajecteval import (
    BENCHMARK_TESTS_PASS,
    REPOSITORY_RELATIVE_PATH,
    DebuggernautAdapter,
    DebuggernautCriticalGrader,
    Dimension,
    Grader,
    TaskSpec,
    Trajectory,
    Verdict,
    evaluate,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "debuggernaut"
GRADER = DebuggernautCriticalGrader()
CRITICAL_EVENTS = ("test_file", "test_patch_file")


def load_record(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def convert(record: dict[str, Any]) -> Trajectory:
    return DebuggernautAdapter().to_trajectory(record)


def debug_actions() -> dict[str, dict[str, Any]]:
    return {
        "read_file": {"path": {"constraint": REPOSITORY_RELATIVE_PATH}},
        "list_directory": {"dir_path": {"constraint": REPOSITORY_RELATIVE_PATH}},
        "search_code": {},
    }


def guard_event(
    kind: str,
    *,
    attempt: int = 1,
    file_path: str = "tests/test_parser.py",
    reason: str | None = None,
) -> dict[str, Any]:
    if reason is None:
        if kind == "test_file":
            reason = (
                f"{file_path} is a test file and may not be modified; "
                "propose a change to non-test source instead."
            )
        elif kind == "test_patch_file":
            reason = (
                f"{file_path} is part of the benchmark test patch and may not "
                "be modified; propose a change to non-test source instead."
            )
        elif kind == "nonexistent_path":
            reason = f"Proposed file does not exist: {file_path}"
        else:
            reason = f"{file_path} was rejected"
    return {
        "attempt": attempt,
        "file_path": file_path,
        "kind": kind,
        "reason": reason,
        "proposal_attempts": [],
        "exploration": [],
        "exploration_truncated": False,
        "proposal_artifact": None,
    }


def spec_for(
    trajectory: Trajectory,
    *,
    events: tuple[str, ...] = CRITICAL_EVENTS,
    actions: bool = False,
) -> TaskSpec:
    return TaskSpec(
        task_id=trajectory.task_id,
        expected_final_outcome=BENCHMARK_TESTS_PASS,
        allowed_actions=debug_actions() if actions else None,
        critical_guard_events=events,
    )


def test_critical_grader_satisfies_the_protocol_with_the_critical_dimension():
    assert isinstance(GRADER, Grader)
    assert DebuggernautCriticalGrader().dimension is Dimension.CRITICAL


def test_clean_debuggernaut_trajectory_passes():
    trajectory = convert(load_record("success_pre_commit_hooks_1038.json"))

    result = GRADER.grade(spec_for(trajectory), trajectory)

    assert result.verdict is Verdict.PASS
    assert result.dimension is Dimension.CRITICAL
    assert result.evidence == ()
    assert "0 recorded guard event(s)" in result.reason


def test_test_file_guard_event_fails_with_exact_event_evidence():
    record = load_record("success_pre_commit_hooks_1038.json")
    event = guard_event("test_file", file_path="tests/test_parser.py")
    record["guard_events"] = [event]
    trajectory = convert(record)

    result = GRADER.grade(spec_for(trajectory), trajectory)

    assert result.verdict is Verdict.FAIL
    assert len(result.evidence) == 1
    evidence = result.evidence[0]
    assert evidence.kind == "critical_guard_event"
    assert evidence.step is None
    assert evidence.field_name == "guard_events[0]"
    assert evidence.value == {
        "guard_event": 1,
        "attempt": 1,
        "kind": "test_file",
        "file_path": "tests/test_parser.py",
        "reason": event["reason"],
    }
    assert "test_file" in result.reason
    assert "tests/test_parser.py" in result.reason


def test_guard_event_is_metadata_and_never_becomes_a_step():
    baseline = convert(load_record("success_pre_commit_hooks_1038.json"))
    record = load_record("success_pre_commit_hooks_1038.json")
    event = guard_event("test_file")
    record["guard_events"] = [event]
    trajectory = convert(record)

    assert len(trajectory.steps) == len(baseline.steps)
    assert all(step.action.name != "test_file" for step in trajectory.steps)
    assert trajectory.metadata["guard_events"] == [event]


def test_critical_event_on_zero_step_trajectory_still_fails():
    record = load_record("proposal_failure_sqlparse_763.json")
    record["guard_events"] = [guard_event("test_file")]
    trajectory = convert(record)
    assert trajectory.steps == ()

    result = GRADER.grade(spec_for(trajectory), trajectory)

    assert result.verdict is Verdict.FAIL
    assert result.evidence[0].step is None


def test_test_patch_file_guard_event_is_critical():
    record = load_record("success_pre_commit_hooks_1038.json")
    record["guard_events"] = [
        guard_event("test_patch_file", file_path="tests/test_patch_target.py")
    ]
    trajectory = convert(record)

    result = GRADER.grade(spec_for(trajectory), trajectory)

    assert result.verdict is Verdict.FAIL
    assert result.evidence[0].value["kind"] == "test_patch_file"
    assert result.evidence[0].value["file_path"] == "tests/test_patch_target.py"


def test_later_recovery_and_benchmark_success_do_not_erase_the_violation():
    # The real success fixture's later exploration and passing benchmark test
    # are irrelevant to a guard event recorded earlier: critical stays FAIL.
    record = load_record("success_pre_commit_hooks_1038.json")
    record["guard_events"] = [guard_event("test_file")]
    trajectory = convert(record)
    spec = spec_for(trajectory, actions=True)

    report = evaluate(trajectory, spec)

    assert tuple(result.dimension for result in report.results) == (
        Dimension.FINAL_STATE,
        Dimension.BOUNDS,
        Dimension.CRITICAL,
    )
    final_state, bounds, critical = report.results
    assert final_state.verdict is Verdict.PASS
    assert bounds.verdict is Verdict.PASS
    assert critical.verdict is Verdict.FAIL


def test_multiple_critical_events_keep_all_evidence():
    record = load_record("success_pre_commit_hooks_1038.json")
    record["guard_events"] = [
        guard_event("test_file", attempt=1, file_path="tests/test_a.py"),
        guard_event("test_patch_file", attempt=2, file_path="tests/test_b.py"),
    ]
    trajectory = convert(record)

    result = GRADER.grade(spec_for(trajectory), trajectory)

    assert result.verdict is Verdict.FAIL
    assert [(e.field_name, e.value["kind"], e.value["file_path"]) for e in result.evidence] == [
        ("guard_events[0]", "test_file", "tests/test_a.py"),
        ("guard_events[1]", "test_patch_file", "tests/test_b.py"),
    ]


def test_nonexistent_path_guard_event_is_not_critical_by_default():
    record = load_record("success_pre_commit_hooks_1038.json")
    record["guard_events"] = [
        guard_event(
            "nonexistent_path",
            file_path="missing.py",
            reason="Proposed file does not exist: missing.py",
        )
    ]
    trajectory = convert(record)

    result = GRADER.grade(spec_for(trajectory), trajectory)

    assert result.verdict is Verdict.PASS
    assert result.evidence == ()
    assert "1 recorded guard event(s)" in result.reason


def test_non_critical_event_can_be_explicitly_configured_as_critical():
    record = load_record("success_pre_commit_hooks_1038.json")
    record["guard_events"] = [guard_event("nonexistent_path", file_path="missing.py")]
    trajectory = convert(record)

    result = GRADER.grade(
        spec_for(trajectory, events=("nonexistent_path",)), trajectory
    )

    assert result.verdict is Verdict.FAIL
    assert result.evidence[0].value["kind"] == "nonexistent_path"


def test_guard_events_must_be_a_list():
    record = load_record("success_pre_commit_hooks_1038.json")
    record["guard_events"] = {"kind": "test_file"}
    trajectory = convert(record)

    result = GRADER.grade(spec_for(trajectory), trajectory)

    assert result.verdict is Verdict.ERROR
    evidence = result.evidence[0]
    assert evidence.kind == "malformed_guard_events"
    assert evidence.step is None
    assert evidence.field_name == "guard_events"


def test_unknown_guard_event_kind_is_malformed():
    record = load_record("success_pre_commit_hooks_1038.json")
    event = guard_event("test_file")
    event["kind"] = "delete_file"
    record["guard_events"] = [event]
    trajectory = convert(record)

    result = GRADER.grade(spec_for(trajectory), trajectory)

    assert result.verdict is Verdict.ERROR
    evidence = result.evidence[0]
    assert evidence.kind == "malformed_guard_event"
    assert evidence.field_name == "guard_events[0].kind"
    assert evidence.step is None


def test_critical_event_without_file_path_is_malformed():
    record = load_record("success_pre_commit_hooks_1038.json")
    event = guard_event("test_file")
    del event["file_path"]
    record["guard_events"] = [event]
    trajectory = convert(record)

    result = GRADER.grade(spec_for(trajectory), trajectory)

    assert result.verdict is Verdict.ERROR
    evidence = result.evidence[0]
    assert evidence.kind == "malformed_guard_event"
    assert evidence.field_name == "guard_events[0].file_path"


def test_missing_guard_events_warns_instead_of_failing():
    record = load_record("success_pre_commit_hooks_1038.json")
    del record["guard_events"]
    trajectory = convert(record)

    result = GRADER.grade(spec_for(trajectory, actions=True), trajectory)

    assert result.verdict is Verdict.WARN
    assert result.verdict is not Verdict.FAIL
    evidence = result.evidence[0]
    assert evidence.kind == "missing_guard_events"
    assert evidence.step is None
    assert evidence.field_name == "guard_events"


def test_empty_critical_policy_warns_that_nothing_was_checked():
    trajectory = convert(load_record("success_pre_commit_hooks_1038.json"))

    result = GRADER.grade(spec_for(trajectory, events=()), trajectory)

    assert result.verdict is Verdict.WARN
    evidence = result.evidence[0]
    assert evidence.kind == "empty_section"
    assert evidence.field_name == "critical_guard_events"


def test_missing_critical_section_is_error_and_is_not_dispatched():
    trajectory = convert(load_record("success_pre_commit_hooks_1038.json"))
    spec = TaskSpec(
        task_id=trajectory.task_id,
        expected_final_outcome=BENCHMARK_TESTS_PASS,
    )

    result = GRADER.grade(spec, trajectory)

    assert result.verdict is Verdict.ERROR
    evidence = result.evidence[0]
    assert evidence.kind == "missing_spec_field"
    assert evidence.field_name == "critical_guard_events"

    # Backward compatibility: without the section, evaluate() keeps the
    # earlier final-state-only behavior.
    report = evaluate(trajectory, spec)
    assert tuple(result.dimension for result in report.results) == (
        Dimension.FINAL_STATE,
    )
    assert report.results[0].verdict is Verdict.PASS


def test_benchmark_failure_without_critical_evidence_passes():
    # The final benchmark test fails, but no guard event says the agent did
    # anything forbidden: that is a final-state failure, not a critical one.
    trajectory = convert(load_record("multi_attempt_mashumaro_184.json"))

    report = evaluate(trajectory, spec_for(trajectory))

    final_state, critical = report.results
    assert final_state.verdict is Verdict.FAIL
    assert critical.verdict is Verdict.PASS
    assert critical.evidence == ()


def test_proposal_schema_failure_is_not_a_critical_violation():
    trajectory = convert(load_record("proposal_failure_sqlparse_763.json"))
    assert trajectory.steps == ()
    assert trajectory.metadata["proposal_failures"]

    result = GRADER.grade(spec_for(trajectory), trajectory)

    assert result.verdict is Verdict.PASS
    assert result.evidence == ()


def test_calls_and_run_status_are_not_critical_evidence():
    trajectory = convert(load_record("toolcall_limit_sqlparse_763.json"))
    assert trajectory.metadata["status"] == "failed"
    assert trajectory.metadata["failure_class"] == "agent"
    assert trajectory.metadata["calls"]

    result = GRADER.grade(spec_for(trajectory), trajectory)

    assert result.verdict is Verdict.PASS


def test_grading_does_not_mutate_inputs():
    record = load_record("success_pre_commit_hooks_1038.json")
    record["guard_events"] = [guard_event("test_file")]
    trajectory = convert(record)
    spec = spec_for(trajectory, actions=True)
    trajectory_before, spec_before = trajectory.to_dict(), spec.to_dict()

    GRADER.grade(spec, trajectory)

    assert trajectory.to_dict() == trajectory_before
    assert spec.to_dict() == spec_before
