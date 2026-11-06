"""Debuggernaut final-state grading from the recorded benchmark outcome.

Debuggernaut does not record repository snapshots around tool calls, so the
adapter deliberately marks every step's state unavailable. Its final state is
therefore the benchmark's own recorded test execution, not a filesystem
snapshot.

The rule is:

* PASS -- the final attempt's benchmark test started, finished, and passed.
* FAIL -- the final attempt's benchmark test started, finished, and failed.
* ERROR -- no usable final test record, malformed fields, a test that never
  started or timed out, or contradictory pass/return-code data.

A provider or infrastructure failure with no recorded benchmark test is an
ERROR: it is not evidence that the agent's fix was wrong. When the record does
contain a reliable final test outcome, that outcome is the source of truth even
if the run later stopped for an unrelated reason.

This grader is offline. It never runs pytest, reads a repository, calls a
provider, or reconstructs state from a step.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from trajecteval.graders.base import missing_section_result, task_id_mismatch_result
from trajecteval.models import Trajectory
from trajecteval.results import Dimension, Evidence, GraderResult, Verdict
from trajecteval.task_spec import BENCHMARK_TESTS_PASS, TaskSpec

SECTION = "expected_final_outcome"
_ATTEMPT_RECORDS = "attempt_records"
_TEST_FIELD = f"{_ATTEMPT_RECORDS}[-1].test"


class DebuggernautFinalStateGrader:
    """The final_state column for a recorded Debuggernaut benchmark run."""

    dimension = Dimension.FINAL_STATE

    def grade(self, spec: TaskSpec, trajectory: Trajectory) -> GraderResult:
        mismatch = task_id_mismatch_result(spec, self.dimension, trajectory)
        if mismatch is not None:
            return mismatch
        blocked = missing_section_result(spec, self.dimension, SECTION)
        if blocked is not None:
            return blocked

        expected = spec.expected_final_outcome  # present; helper proved it
        if expected != BENCHMARK_TESTS_PASS:
            return _error(
                kind="unsupported_spec_expectation",
                field_name=SECTION,
                reason=(
                    f"spec expects {expected!r}, but the Debuggernaut final-state "
                    f"grader only supports {BENCHMARK_TESTS_PASS!r}"
                ),
                value={"expected": expected, "supported": [BENCHMARK_TESTS_PASS]},
            )
        return _grade_recorded_outcome(trajectory)


def _grade_recorded_outcome(trajectory: Trajectory) -> GraderResult:
    metadata = trajectory.metadata
    attempts = metadata.get(_ATTEMPT_RECORDS)
    if not isinstance(attempts, list):
        return _error(
            kind="missing_benchmark_test",
            field_name=_ATTEMPT_RECORDS,
            reason=(
                "trajectory metadata has no usable 'attempt_records' list; "
                "cannot determine the benchmark test outcome"
            ),
            value={
                "present": _ATTEMPT_RECORDS in metadata,
                "expected": "a list of attempt records",
            },
        )
    if not attempts:
        return _error(
            kind="missing_benchmark_test",
            field_name=_ATTEMPT_RECORDS,
            reason=(
                "trajectory metadata has no attempt records; the benchmark "
                "test outcome is unavailable"
            ),
            value={"attempt_count": 0},
        )

    final_attempt = attempts[-1]
    if not isinstance(final_attempt, Mapping):
        return _error(
            kind="malformed_benchmark_test",
            field_name=f"{_ATTEMPT_RECORDS}[-1]",
            reason=(
                "the final attempt record is not an object; cannot determine "
                "the benchmark test outcome"
            ),
            value={"type": type(final_attempt).__name__},
        )

    attempt_number = _attempt_number(final_attempt, len(attempts))
    test = final_attempt.get("test")
    if not isinstance(test, Mapping):
        return _error(
            kind="missing_benchmark_test",
            field_name=_TEST_FIELD,
            reason=(
                f"final attempt {attempt_number} has no recorded benchmark "
                "test result"
            ),
            value={"attempt": attempt_number, "expected": "a recorded test object"},
        )

    started = test.get("started")
    if not isinstance(started, bool):
        return _malformed(
            attempt_number,
            "started",
            reason=(
                f"final attempt {attempt_number} has no valid 'started' flag; "
                "cannot tell whether the benchmark test ran"
            ),
        )
    if not started:
        return _error(
            kind="benchmark_test_not_started",
            field_name=f"{_TEST_FIELD}.started",
            reason=(
                f"the benchmark test in final attempt {attempt_number} never "
                "started; its outcome is unavailable"
            ),
            value={"attempt": attempt_number, "started": False},
        )

    timed_out = test.get("timed_out")
    if not isinstance(timed_out, bool):
        return _malformed(
            attempt_number,
            "timed_out",
            reason=(
                f"final attempt {attempt_number} has no valid 'timed_out' flag; "
                "cannot tell whether the benchmark test finished"
            ),
        )
    if timed_out:
        return _error(
            kind="benchmark_test_timed_out",
            field_name=f"{_TEST_FIELD}.timed_out",
            reason=(
                f"the benchmark test in final attempt {attempt_number} timed "
                "out; it did not produce a reliable outcome"
            ),
            value={"attempt": attempt_number, "timed_out": True},
        )

    passed = test.get("passed")
    if not isinstance(passed, bool):
        return _malformed(
            attempt_number,
            "passed",
            reason=(
                f"final attempt {attempt_number} has no valid 'passed' flag; "
                "cannot determine the benchmark test outcome"
            ),
        )

    return_code = test.get("return_code")
    if isinstance(return_code, bool) or not isinstance(return_code, int):
        return _malformed(
            attempt_number,
            "return_code",
            reason=(
                f"final attempt {attempt_number} has no valid 'return_code'; "
                "cannot trust its test outcome"
            ),
            value=return_code,
        )
    if passed != (return_code == 0):
        return _error(
            kind="inconsistent_benchmark_test",
            field_name=_TEST_FIELD,
            reason=(
                f"final attempt {attempt_number} records passed={passed!r} "
                f"with return_code={return_code}; the test result contradicts "
                "itself"
            ),
            value={
                "attempt": attempt_number,
                "passed": passed,
                "return_code": return_code,
            },
        )

    if passed:
        return GraderResult(
            dimension=Dimension.FINAL_STATE,
            verdict=Verdict.PASS,
            reason=(
                f"benchmark tests passed in final attempt {attempt_number}"
                f"{_run_suffix(metadata)}"
            ),
        )

    return GraderResult(
        dimension=Dimension.FINAL_STATE,
        verdict=Verdict.FAIL,
        reason=(
            f"benchmark tests failed in final attempt {attempt_number}"
            f"{_run_suffix(metadata)}"
        ),
        evidence=(
            Evidence(
                kind="benchmark_test_failed",
                step=None,
                field_name=_TEST_FIELD,
                value={
                    "attempt": attempt_number,
                    "attempt_count": len(attempts),
                    "run_status": _optional_str(metadata.get("status")),
                    "failure_class": _optional_str(metadata.get("failure_class")),
                    "test": {
                        "passed": passed,
                        "started": started,
                        "timed_out": timed_out,
                        "return_code": return_code,
                    },
                },
            ),
        ),
    )


def _malformed(
    attempt_number: int,
    field: str,
    *,
    reason: str,
    value: Any = None,
) -> GraderResult:
    return _error(
        kind="malformed_benchmark_test",
        field_name=f"{_TEST_FIELD}.{field}",
        reason=reason,
        value={
            "attempt": attempt_number,
            "field": field,
            "actual_type": type(value).__name__,
        },
    )


def _error(
    *,
    kind: str,
    field_name: str,
    reason: str,
    value: Any = None,
) -> GraderResult:
    return GraderResult(
        dimension=Dimension.FINAL_STATE,
        verdict=Verdict.ERROR,
        reason=reason,
        evidence=(
            Evidence(
                kind=kind,
                step=None,
                field_name=field_name,
                value=value,
            ),
        ),
    )


def _attempt_number(attempt: Mapping[str, Any], fallback: int) -> int:
    value = attempt.get("attempt")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return fallback


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _run_suffix(metadata: Mapping[str, Any]) -> str:
    status = _optional_str(metadata.get("status"))
    failure_class = _optional_str(metadata.get("failure_class"))
    parts = []
    if status is not None:
        parts.append(f"run status {status!r}")
    if failure_class is not None:
        parts.append(f"failure class {failure_class!r}")
    return f" ({', '.join(parts)})" if parts else ""
