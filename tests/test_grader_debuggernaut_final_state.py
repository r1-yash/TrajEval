"""Tests for Debuggernaut final-state grading.

The fixtures are the real Debuggernaut records used by the adapter tests. Edge
cases are derived in memory from those records by changing only the recorded
metadata under test (for example, removing the final test result); no new
Debuggernaut schema is invented.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from trajecteval import (
    BENCHMARK_TESTS_PASS,
    DebuggernautAdapter,
    DebuggernautFinalStateGrader,
    Dimension,
    FinalStateGrader,
    Grader,
    TaskSpec,
    Trajectory,
    Verdict,
    evaluate,
    load_spec,
)

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "fixtures" / "debuggernaut"
GRADER = DebuggernautFinalStateGrader()


def load_record(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def convert(record: dict[str, Any]) -> Trajectory:
    return DebuggernautAdapter().to_trajectory(record)


def spec_for(trajectory: Trajectory) -> TaskSpec:
    return TaskSpec(
        task_id=trajectory.task_id,
        expected_final_outcome=BENCHMARK_TESTS_PASS,
    )


def success_trajectory() -> Trajectory:
    return convert(load_record("success_pre_commit_hooks_1038.json"))


def test_grader_satisfies_the_protocol_with_the_final_state_dimension():
    assert isinstance(GRADER, Grader)
    assert DebuggernautFinalStateGrader().dimension is Dimension.FINAL_STATE


def test_successful_benchmark_run_passes():
    trajectory = success_trajectory()

    result = GRADER.grade(spec_for(trajectory), trajectory)

    assert result.verdict is Verdict.PASS
    assert result.dimension is Dimension.FINAL_STATE
    assert result.evidence == ()
    assert "final attempt 1" in result.reason
    assert "success" in result.reason


def test_failed_final_benchmark_test_fails_and_cites_the_last_attempt():
    trajectory = convert(load_record("multi_attempt_mashumaro_184.json"))

    result = GRADER.grade(spec_for(trajectory), trajectory)

    assert result.verdict is Verdict.FAIL
    assert result.dimension is Dimension.FINAL_STATE
    assert len(result.evidence) == 1
    evidence = result.evidence[0]
    assert evidence.kind == "benchmark_test_failed"
    assert evidence.step is None
    assert evidence.field_name == "attempt_records[-1].test"
    assert evidence.value == {
        "attempt": 3,
        "attempt_count": 3,
        "run_status": "failed",
        "failure_class": "agent",
        "test": {
            "passed": False,
            "started": True,
            "timed_out": False,
            "return_code": 2,
        },
    }
    assert "final attempt 3" in result.reason


def test_final_passing_attempt_wins_over_earlier_failures():
    record = load_record("multi_attempt_mashumaro_184.json")
    record["status"] = "success"
    record["failure_class"] = None
    record["attempt_records"][-1]["test"].update(
        {"passed": True, "return_code": 0, "timed_out": False, "started": True}
    )
    trajectory = convert(record)

    result = GRADER.grade(spec_for(trajectory), trajectory)

    assert result.verdict is Verdict.PASS
    assert "final attempt 3" in result.reason


def test_later_attempt_without_a_test_does_not_inherit_an_earlier_failure():
    record = load_record("multi_attempt_mashumaro_184.json")
    del record["attempt_records"][-1]["test"]
    trajectory = convert(record)

    result = GRADER.grade(spec_for(trajectory), trajectory)

    assert result.verdict is Verdict.ERROR
    assert result.evidence[0].kind == "missing_benchmark_test"
    assert result.evidence[0].field_name == "attempt_records[-1].test"


def test_reliable_failed_test_wins_over_an_infrastructure_label():
    record = load_record("multi_attempt_mashumaro_184.json")
    record["status"] = "provider_unavailable"
    record["failure_class"] = "infrastructure"
    trajectory = convert(record)

    result = GRADER.grade(spec_for(trajectory), trajectory)

    assert result.verdict is Verdict.FAIL
    assert result.evidence[0].value["run_status"] == "provider_unavailable"
    assert result.evidence[0].value["failure_class"] == "infrastructure"


def test_provider_failure_without_a_benchmark_result_is_error():
    record = load_record("success_pre_commit_hooks_1038.json")
    record["status"] = "provider_unavailable"
    record["failure_class"] = "infrastructure"
    record["status_reason"] = "Provider unavailable"
    record["attempt_records"] = []
    trajectory = convert(record)

    result = GRADER.grade(spec_for(trajectory), trajectory)

    assert result.verdict is Verdict.ERROR
    assert result.evidence[0].kind == "missing_benchmark_test"
    assert result.evidence[0].field_name == "attempt_records"
    assert "unavailable" in result.reason


def test_proposal_failure_without_a_benchmark_result_is_error():
    trajectory = convert(load_record("proposal_failure_sqlparse_763.json"))

    result = GRADER.grade(spec_for(trajectory), trajectory)

    assert result.verdict is Verdict.ERROR
    assert result.evidence[0].kind == "missing_benchmark_test"
    assert result.evidence[0].value == {"attempt_count": 0}


@pytest.mark.parametrize(
    ("field", "value", "kind"),
    [
        pytest.param("started", False, "benchmark_test_not_started", id="not-started"),
        pytest.param("timed_out", True, "benchmark_test_timed_out", id="timed-out"),
    ],
)
def test_interrupted_or_incomplete_test_outcome_is_error(
    field: str, value: bool, kind: str
) -> None:
    record = load_record("multi_attempt_mashumaro_184.json")
    record["attempt_records"][-1]["test"][field] = value
    record["status"] = "errored"
    record["failure_class"] = "infrastructure"
    trajectory = convert(record)

    result = GRADER.grade(spec_for(trajectory), trajectory)

    assert result.verdict is Verdict.ERROR
    assert result.evidence[0].kind == kind
    assert result.evidence[0].field_name == f"attempt_records[-1].test.{field}"


@pytest.mark.parametrize(
    ("mutate", "kind"),
    [
        pytest.param(
            lambda test: test.pop("passed"),
            "malformed_benchmark_test",
            id="missing-passed",
        ),
        pytest.param(
            lambda test: test.__setitem__("passed", "yes"),
            "malformed_benchmark_test",
            id="non-boolean-passed",
        ),
        pytest.param(
            lambda test: test.pop("started"),
            "malformed_benchmark_test",
            id="missing-started",
        ),
        pytest.param(
            lambda test: test.__setitem__("timed_out", "no"),
            "malformed_benchmark_test",
            id="non-boolean-timed-out",
        ),
        pytest.param(
            lambda test: test.pop("return_code"),
            "malformed_benchmark_test",
            id="missing-return-code",
        ),
        pytest.param(
            lambda test: test.__setitem__("return_code", "2"),
            "malformed_benchmark_test",
            id="non-integer-return-code",
        ),
        pytest.param(
            lambda test: test.update({"passed": False, "return_code": 0}),
            "inconsistent_benchmark_test",
            id="failed-test-with-zero-return-code",
        ),
    ],
)
def test_malformed_test_metadata_is_error(mutate: Any, kind: str) -> None:
    record = load_record("success_pre_commit_hooks_1038.json")
    mutate(record["attempt_records"][-1]["test"])
    trajectory = convert(record)

    result = GRADER.grade(spec_for(trajectory), trajectory)

    assert result.verdict is Verdict.ERROR
    assert result.evidence[0].kind == kind


def test_missing_final_test_result_is_error():
    record = load_record("success_pre_commit_hooks_1038.json")
    del record["attempt_records"][-1]["test"]
    trajectory = convert(record)

    result = GRADER.grade(spec_for(trajectory), trajectory)

    assert result.verdict is Verdict.ERROR
    assert result.evidence[0].kind == "missing_benchmark_test"
    assert result.evidence[0].field_name == "attempt_records[-1].test"


@pytest.mark.parametrize(
    ("metadata", "kind"),
    [
        pytest.param({}, "missing_benchmark_test", id="no-attempt-records"),
        pytest.param(
            {"attempt_records": {}}, "missing_benchmark_test", id="not-a-list"
        ),
        pytest.param(
            {"attempt_records": ["not-an-object"]},
            "malformed_benchmark_test",
            id="final-attempt-not-an-object",
        ),
    ],
)
def test_missing_or_malformed_attempt_metadata_is_error(
    metadata: dict[str, Any], kind: str
) -> None:
    base = success_trajectory()
    without_attempts = {
        key: value
        for key, value in base.metadata.items()
        if key != "attempt_records"
    }
    trajectory = Trajectory(
        task_id=base.task_id,
        steps=base.steps,
        metadata={**without_attempts, **metadata},
    )

    result = GRADER.grade(spec_for(trajectory), trajectory)

    assert result.verdict is Verdict.ERROR
    assert result.evidence[0].kind == kind
    assert result.evidence[0].step is None


def test_missing_outcome_section_is_error_not_fail():
    trajectory = success_trajectory()

    result = GRADER.grade(TaskSpec(task_id=trajectory.task_id), trajectory)

    assert result.verdict is Verdict.ERROR
    assert result.evidence[0].kind == "missing_spec_field"
    assert result.evidence[0].field_name == "expected_final_outcome"


def test_wrong_pairing_is_error_before_section_check():
    trajectory = success_trajectory()
    spec = TaskSpec(task_id="other__repo__1")

    result = GRADER.grade(spec, trajectory)

    assert result.verdict is Verdict.ERROR
    assert result.evidence[0].kind == "task_id_mismatch"


def test_evaluate_dispatches_benchmark_specs_to_the_debuggernaut_grader():
    trajectory = success_trajectory()

    report = evaluate(trajectory, spec_for(trajectory))

    assert tuple(result.dimension for result in report.results) == (
        Dimension.FINAL_STATE,
    )
    assert report.results[0].verdict is Verdict.PASS


def test_existing_travel_booking_final_state_evaluation_is_unchanged():
    spec = load_spec("travel_booking", tasks_dir=REPO / "tasks")
    trajectory = Trajectory.from_json_file(
        REPO / "tests" / "fixtures" / "travel_booking" / "clean.json"
    )

    direct = FinalStateGrader().grade(spec, trajectory)
    report = evaluate(trajectory, spec)

    assert direct.verdict is Verdict.PASS
    assert tuple(result.dimension for result in report.results) == (
        Dimension.FINAL_STATE,
        Dimension.BOUNDS,
        Dimension.CRITICAL,
    )
    assert report.results[0].verdict is Verdict.PASS


def test_grading_does_not_mutate_the_trajectory_or_spec():
    trajectory = convert(load_record("multi_attempt_mashumaro_184.json"))
    spec = spec_for(trajectory)
    trajectory_before, spec_before = trajectory.to_dict(), spec.to_dict()

    GRADER.grade(spec, trajectory)

    assert trajectory.to_dict() == trajectory_before
    assert spec.to_dict() == spec_before


def test_calls_are_metadata_not_a_fallback_for_a_missing_test_result():
    record = load_record("success_pre_commit_hooks_1038.json")
    record["attempt_records"] = []
    trajectory = convert(record)
    assert trajectory.metadata["calls"]
    assert trajectory.steps == ()

    result = GRADER.grade(spec_for(trajectory), trajectory)

    assert result.verdict is Verdict.ERROR
    assert result.evidence[0].kind == "missing_benchmark_test"
