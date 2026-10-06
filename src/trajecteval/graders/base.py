"""The grader contract: who counts as a grader, and how one admits it can't judge.

A grader is any object with a ``dimension`` (the comparison-table column it
owns) and a ``grade(spec, trajectory) -> GraderResult`` method -- a hiring
contract, not a base class. Whoever it is, it fills out the same report card,
which is why Step 7 can loop over graders with zero special cases.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from trajecteval.models import Trajectory
from trajecteval.results import Dimension, Evidence, GraderResult, Verdict
from trajecteval.task_spec import RULE_SECTIONS, TaskSpec


@runtime_checkable
class Grader(Protocol):
    """Anything with a ``dimension`` and a ``grade`` method is a grader.

    ``dimension`` is load-bearing, not decoration: the report layer reads it
    without calling ``grade`` -- to know which column this grader owns, and
    to attribute an ERROR result to that column if ``grade`` ever crashes
    (Step 7's wrapper). ``isinstance`` checks attribute presence only; the
    argument types of ``grade`` are the type checker's job.
    """

    dimension: Dimension

    def grade(self, spec: TaskSpec, trajectory: Trajectory) -> GraderResult:
        """Judge one dimension of one trajectory against one task's rules."""
        ...  # pragma: no cover - protocol body, never executed


def task_id_mismatch_result(
    spec: TaskSpec, dimension: Dimension, trajectory: Trajectory
) -> GraderResult | None:
    """ERROR result when spec and trajectory are for different tasks; None otherwise.

    Every grader calls this **first**, before ``missing_section_result``: if
    the pairing is wrong, the spec's sections are irrelevant. A mismatch is a
    *pairing* problem, not a verdict on either file -- so the evidence is a
    diagnostic naming both ids (``step=None``, spec-side), never an
    accusation against the trajectory, whose own task_id may be perfectly
    fine.
    """
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
    """ERROR result when the spec lacks ``section``; None when it is present.

    Call order in a grader: ``task_id_mismatch_result`` first, then this --
    pairing, then sections. Every grader calls this before checking anything:
    if the rulebook is missing the page this grader needs, it cannot judge --
    so it says which section is missing (ERROR verdict, diagnostic evidence
    pointing at the spec, ``step=None``) instead of crashing or guessing. An
    *empty* section is present and judged normally; only an absent one is an
    ERROR.
    """
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
