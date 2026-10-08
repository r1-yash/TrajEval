# Critical grader: scan the whole history for critical_error_patterns matches; recovery never erases one.

from __future__ import annotations

from trajecteval.graders.base import missing_section_result, task_id_mismatch_result
from trajecteval.models import Trajectory
from trajecteval.results import Dimension, Evidence, GraderResult, Verdict
from trajecteval.task_spec import TaskSpec

SECTION = "critical_error_patterns"


class CriticalGrader:
    # The critical column of the report card.

    dimension = Dimension.CRITICAL

    def grade(self, spec: TaskSpec, trajectory: Trajectory) -> GraderResult:
        mismatch = task_id_mismatch_result(spec, self.dimension, trajectory)
        if mismatch is not None:
            return mismatch
        blocked = missing_section_result(spec, self.dimension, SECTION)
        if blocked is not None:
            return blocked
        patterns = spec.critical_error_patterns  # present; helper proved it
        if not patterns:
            return GraderResult(
                dimension=self.dimension,
                verdict=Verdict.WARN,
                reason=f"{SECTION} is empty: no rules to check",
                evidence=(Evidence(kind="empty_section", step=None, field_name=SECTION),),
            )
        if not trajectory.steps:
            return GraderResult(
                dimension=self.dimension,
                verdict=Verdict.WARN,
                reason="trajectory has no steps: no patterns were checked",
                evidence=(Evidence(kind="empty_trajectory", step=None),),
            )
        hits: list[Evidence] = []
        for step in trajectory.steps:
            for pattern in patterns:
                if not pattern.matches(step):
                    continue
                value: dict = {"action": pattern.action, "args": dict(pattern.args)}
                if pattern.description is not None:
                    value["description"] = pattern.description
                hits.append(
                    Evidence(
                        kind="critical_error",
                        step=step.index,
                        field_name=pattern.action,
                        value=value,
                    )
                )
        if not hits:
            return GraderResult(
                dimension=self.dimension,
                verdict=Verdict.PASS,
                reason=(
                    f"no {SECTION} matched in {len(trajectory.steps)} step(s)"
                ),
            )
        fragments = []
        for hit in hits:
            fragment = f"step {hit.step}: {hit.value['action']} {hit.value['args']!r}"
            if "description" in hit.value:
                fragment = f"{fragment} -- {hit.value['description']}"
            fragments.append(fragment)
        return GraderResult(
            dimension=self.dimension,
            verdict=Verdict.FAIL,
            reason=(
                f"{len(hits)} critical mistake(s): " + "; ".join(fragments)
            ),
            evidence=tuple(hits),
        )
