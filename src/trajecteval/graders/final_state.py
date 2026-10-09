#this just checks last step of final grader

from __future__ import annotations

from trajecteval.graders.base import missing_section_result, task_id_mismatch_result
from trajecteval.models import Trajectory
from trajecteval.results import Dimension, Evidence, GraderResult, Verdict
from trajecteval.task_spec import TaskSpec, contains_failures

SECTION = "expected_final_state"


class FinalStateGrader:
    # The final_state column of the report card.

    dimension = Dimension.FINAL_STATE

    def grade(self, spec: TaskSpec, trajectory: Trajectory) -> GraderResult:
        mismatch = task_id_mismatch_result(spec, self.dimension, trajectory)
        if mismatch is not None:
            return mismatch
        blocked = missing_section_result(spec, self.dimension, SECTION)
        if blocked is not None:
            return blocked
        expected = spec.expected_final_state  # present; helper proved it
        if not expected:
            return GraderResult(
                dimension=self.dimension,
                verdict=Verdict.WARN,
                reason=f"{SECTION} is empty: no rules to check",
                evidence=(Evidence(kind="empty_section", step=None, field_name=SECTION),),
            )
        if not trajectory.steps:
            return GraderResult(
                dimension=self.dimension,
                verdict=Verdict.FAIL,
                reason="trajectory has no steps: the expected final state was never reached",
                evidence=(
                    Evidence(
                        kind="final_state_not_reached",
                        step=None,
                        value={"expected": dict(expected)},
                    ),
                ),
            ) 
        last = trajectory.steps[-1] ##last element of sequence
        failures = contains_failures(last.state_after, expected)
        if not failures:
            return GraderResult(
                dimension=self.dimension,
                verdict=Verdict.PASS,
                reason=(
                    f"final state at step {last.index} matches "
                    f"{SECTION} ({len(expected)} field(s))"
                ),
            )
        evidence = tuple(
            Evidence(
                kind=(
                    "final_state_field_missing"
                    if kind == "missing"
                    else "final_state_value_mismatch"
                ),
                step=last.index,
                field_name=path or None,
                value=(
                    {"expected": wanted} if kind == "missing"
                    else {"expected": wanted, "actual": seen}
                ),
            )
            for path, kind, wanted, seen in failures
        )
        return GraderResult(
            dimension=self.dimension,
            verdict=Verdict.FAIL,
            reason=(
                f"final state at step {last.index}: {len(failures)} of "
                f"{len(expected)} expected field(s) missing or wrong"
            ),
            evidence=evidence,
        )
