# The grader contract: dimension + grade(); graders return ERROR results instead of raising (Step 7's evaluate() wraps crashes).

from __future__ import annotations

from typing import Protocol, runtime_checkable

from trajecteval.models import Trajectory
from trajecteval.results import Dimension, Evidence, GraderResult, Verdict
from trajecteval.task_spec import RULE_SECTIONS, TaskSpec


@runtime_checkable
class Grader(Protocol):
    # Protocol: anything with a `dimension` (the column it owns) and a grade(spec, trajectory) method.

    dimension: Dimension

    def grade(self, spec: TaskSpec, trajectory: Trajectory) -> GraderResult:
        # Judge one dimension of one trajectory against one task's rules.
        ...  # pragma: no cover - protocol body, never executed


def task_id_mismatch_result(
    spec: TaskSpec, dimension: Dimension, trajectory: Trajectory
) -> GraderResult | None:
    # ERROR when spec and trajectory name different tasks (diagnostic, step=None); every grader calls this first.
    if spec.task_id == trajectory.task_id:
        return None
    return GraderResult(
        dimension=dimension,
        verdict=Verdict.ERROR,
        reason=(
            f"spec is for task {spec.task_id!r} but trajectory is for "
            f"{trajectory.task_id!r}; wrong pairing -- "
            f"cannot judge the {dimension.value} dimension"
        ),
        evidence=(
            Evidence(
                kind="task_id_mismatch",
                step=None,
                field_name="task_id",
                value={"spec": spec.task_id, "trajectory": trajectory.task_id},
            ),
        ),
    )


def missing_section_result(
    spec: TaskSpec, dimension: Dimension, section: str
) -> GraderResult | None:
    # ERROR naming a missing section, called second; present-but-empty returns None (the grader WARNs empty_section).
    # Validate before getattr: a typo'd section would raise a bare
    # AttributeError, and a real-but-wrong attribute like "task_id" would
    # silently report "present". A bad section name is a grader-code bug,
    # so ValueError (the grader lane), not SpecError (the data lane).
    if section not in RULE_SECTIONS:
        raise ValueError(
            f"invalid section {section!r}; expected one of {', '.join(RULE_SECTIONS)}"
        )
    if getattr(spec, section) is not None:
        return None
    return GraderResult(
        dimension=dimension,
        verdict=Verdict.ERROR,
        reason=(
            f"spec for task {spec.task_id!r} has no {section!r} section; "
            f"cannot judge the {dimension.value} dimension"
        ),
        evidence=(
            Evidence(kind="missing_spec_field", step=None, field_name=section),
        ),
    )
