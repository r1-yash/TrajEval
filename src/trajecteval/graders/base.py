"""The grader contract: who counts as a grader, and how one admits it can't judge.

A grader is any object with ``grade(spec, trajectory) -> GraderResult`` -- a
hiring contract, not a base class. Whoever it is, it fills out the same report
card, which is why Step 7 can loop over graders with zero special cases.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from trajecteval.models import Trajectory
from trajecteval.results import Dimension, Evidence, GraderResult, Verdict
from trajecteval.task_spec import TaskSpec


@runtime_checkable
class Grader(Protocol):
    """Anything with a ``grade`` method of the right shape is a grader."""

    def grade(self, spec: TaskSpec, trajectory: Trajectory) -> GraderResult:
        """Judge one dimension of one trajectory against one task's rules."""
        ...  # pragma: no cover - protocol body, never executed


def missing_section_result(
    spec: TaskSpec, dimension: Dimension, section: str
) -> GraderResult | None:
    """ERROR result when the spec lacks ``section``; None when it is present.

    Every grader calls this before checking anything: if the rulebook is
    missing the page this grader needs, it cannot judge -- so it says which
    section is missing (ERROR verdict, diagnostic evidence pointing at the
    spec, ``step=None``) instead of crashing or guessing. An *empty* section
    is present and judged normally; only an absent one is an ERROR.
    """
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
