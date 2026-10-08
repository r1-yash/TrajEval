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
    #basically the things we applied in base, like missing section or mismatch we are using it here via import 
    def grade(self, spec: TaskSpec, trajectory: Trajectory) -> GraderResult:
        mismatch = task_id_mismatch_result(spec, self.dimension, trajectory)
        if mismatch is not None:
            return mismatch
        blocked = missing_section_result(spec, self.dimension, SECTION)
        if blocked is not None:
            return blocked
        #westore the critical error patterns from the spec and check if they are present and if agent did it then we flag it BUT
        #if there are no critical errors pattern defined by user then we just a warning 
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
                verdict=Verdict.WARN, #we check if agent did anything, if not we do warning that no steps taken so we cant say that agent passed all critical error checs 
                reason="trajectory has no steps: no patterns were checked",
                evidence=(Evidence(kind="empty_trajectory", step=None),),
            )

        #everthing before this was to check that whether my critcalgrader has things to check, are they valid

        hits: list[Evidence] = [] #empty list to store mistakes
        for step in trajectory.steps: #with complexity of n2 we check that does my step1 violates any pattern ? 
            for pattern in patterns:
                if not pattern.matches(step):
                    continue
                value: dict = {"action": pattern.action, "args": dict(pattern.args)} # This collects information about what critical-error pattern was violated.
                if pattern.description is not None: ## desc if it has one, like - test file must never be deleted 
                    value["description"] = pattern.description 
                hits.append(
                    Evidence( ## we keep on adding evidence as needed 
                        kind="critical_error",
                        step=step.index,
                        field_name=pattern.action,
                        value=value,
                    )
                )
        if not hits:
            return GraderResult(
                dimension=self.dimension,
                verdict=Verdict.PASS, ## no critical errors foudn by gent so we pass 
                reason=(
                    f"no {SECTION} matched in {len(trajectory.steps)} step(s)"
                ),
            )
        fragments = []
        for hit in hits: ##means we go errors, it wil contain human readable mistakes 
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
            evidence=tuple(hits), #so we return a tuple of evidence readable 
        )
