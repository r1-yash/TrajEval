# evaluate(): run the graders, survive a crash, assemble the Report.

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Sequence

from trajecteval.errors import ReportError, TrajectoryError
from trajecteval.graders import BoundsGrader, CriticalGrader, FinalStateGrader, Grader
from trajecteval.models import Discontinuity, Trajectory
from trajecteval.results import Dimension, Evidence, GraderResult, Verdict
from trajecteval.task_spec import TaskSpec

##the to_dict and from_dict are a pattern called serialization and deserialization. for converting to python dits tuples, and for converting to json and back.
@dataclass(frozen=True)
class Report:
    # One trajectory's full report card: both task ids, one result per column,
    # recorder discontinuities. Deliberately no overall verdict (README FAQ:
    # collapsing columns into one score destroys information).

    task_id: str
    trajectory_task_id: str
    results: tuple[GraderResult, ...] = () #the ... means that it can contain any number of objects
    discontinuities: tuple[Discontinuity, ...] = ()

    '''the results are pass, warn fail thing and the discontinuties refers to anomalies liek state after not matching state before'''
    def __post_init__(self) -> None:
        # Accept lists at construction (ergonomic), store tuples (frozen class,
        # no mutable fields -- same rule as Trajectory.steps and differing_keys).
        if isinstance(self.results, list): ##checking what we get is a list or not
            object.__setattr__(self, "results", tuple(self.results)) #and here we convert it to a tuple because we want it to be be immutable, 
        if isinstance(self.discontinuities, list):
            object.__setattr__(self, "discontinuities", tuple(self.discontinuities)) 

    def to_dict(self) -> dict[str, Any]:
        # Round-trippable plain dict: from_dict(to_dict(report)) == report.
        return {
            "task_id": self.task_id,
            "trajectory_task_id": self.trajectory_task_id,
            "results": [result.to_dict() for result in self.results],
            "discontinuities": [item.to_dict() for item in self.discontinuities],
        }

    @classmethod
    def from_dict(cls, data: Any) -> Report:
        # Inverse of to_dict, in the saved-report file lane: every structural
        # problem raises ReportError (a TrajEvalError, same except-catch as
        # SpecError/TrajectoryError), wrapping the nested ValueError from
        # results.py and TrajectoryError from models.py where they occur.
        if not isinstance(data, dict):
            raise ReportError(f"report must be an object, got {type(data).__name__}")
        for key in ("task_id", "trajectory_task_id"):
            if key not in data:
                raise ReportError(f"report missing '{key}'")
            value = data[key]
            if not isinstance(value, str) or not value:
                raise ReportError(
                    f"report '{key}' must be a non-empty string, got {value!r}"
                )
        raw_results = data.get("results", [])
        if not isinstance(raw_results, list):
            raise ReportError(
                f"report 'results' must be a list, got {type(raw_results).__name__}"
            )
        raw_discontinuities = data.get("discontinuities", [])
        if not isinstance(raw_discontinuities, list):
            raise ReportError(
                "report 'discontinuities' must be a list, "
                f"got {type(raw_discontinuities).__name__}"
            )
        try:
            results = tuple(GraderResult.from_dict(item) for item in raw_results)
            discontinuities = tuple(
                Discontinuity.from_dict(item) for item in raw_discontinuities
            )
        except (ValueError, TrajectoryError) as exc:
            raise ReportError(f"broken report: {exc}") from exc
        seen: list[Dimension] = []
        for result in results:
            if result.dimension in seen:
                raise ReportError(
                    f"report has two results for dimension {result.dimension.value}; "
                    "one column, one verdict"
                )
            seen.append(result.dimension)
        return cls(
            task_id=data["task_id"],
            trajectory_task_id=data["trajectory_task_id"],
            results=results,
            discontinuities=discontinuities,
        )


def _default_graders() -> tuple[Grader, ...]:
    # The three deterministic columns, in dimension order.
    return (FinalStateGrader(), BoundsGrader(), CriticalGrader())


def _require_judgeable(graders: Sequence[Grader]) -> tuple[Dimension, ...]:
    # its a security check before it enters grader and checks are -  empty list, valid dimension, one grader per
    # column -- raises ValueError before any grade() call: broken caller code, same lane as every other ValueError in the project.
    if not graders:
        raise ValueError("evaluate() needs at least one grader; got an empty list")
    dimensions: list[Dimension] = []
    for grader in graders:
        raw = getattr(grader, "dimension", None)
        try:
            dimension = raw if isinstance(raw, Dimension) else Dimension(raw)
        except (ValueError, TypeError):
            raise ValueError(
                f"grader {grader!r} has no valid dimension attribute; "
                f"expected one of {', '.join(d.value for d in Dimension)}"
            ) from None
        if dimension in dimensions:
            raise ValueError(
                f"duplicate grader for dimension {dimension.value}; "
                "each column needs at most one grader"
            )
        dimensions.append(dimension)
    return tuple(dimensions)


def _crash_result(dimension: Dimension, exc: Exception) -> GraderResult:
    # A crashed grader degrades its own column and nothing else: ERROR with
    # kind=grader_crash (exception type + message, never a traceback).
    return GraderResult(
        dimension=dimension,
        verdict=Verdict.ERROR,
        reason=f"the {dimension.value} grader crashed: {type(exc).__name__}",
        evidence=(
            Evidence(
                kind="grader_crash",
                step=None,
                value={"exception": type(exc).__name__, "message": str(exc)},
            ),
        ),
    )


def evaluate(
    trajectory: Trajectory,
    spec: TaskSpec,
    graders: Sequence[Grader] | None = None,
) -> Report:
    # The foreman, not an inspector: hand every grader the same spec and the
    # same trajectory, keep one broken grader to its own column, and assemble
    # the report card. KeyboardInterrupt/SystemExit are the operator, not a
    # bug, so only Exception is caught.
    chosen = _default_graders() if graders is None else tuple(graders)
    dimensions = _require_judgeable(chosen)
    results: list[GraderResult] = []
    for grader, dimension in zip(chosen, dimensions):
        try:
            result = grader.grade(spec, trajectory)
            if not isinstance(result, GraderResult):
                raise TypeError(
                    f"grader for {dimension.value} returned "
                    f"{type(result).__name__}, expected GraderResult"
                )
            if result.dimension is not dimension:
                raise ValueError(
                    f"grader for {dimension.value} returned a result for "
                    f"{result.dimension.value}"
                )
        except Exception as exc:
            results.append(_crash_result(dimension, exc))
        else:
            results.append(result)
    return Report(
        task_id=spec.task_id,
        trajectory_task_id=trajectory.task_id,
        results=tuple(results),
        discontinuities=tuple(trajectory.find_discontinuities()),
    )


def render_json(report: Report, *, indent: int = 2) -> str:
    # The same report as bytes for machines: plain JSON, no envelope, no
    # trailing newline (Trajectory.to_json sets the precedent). Round-trips
    # through Report.from_dict unchanged.
    return json.dumps(report.to_dict(), indent=indent, ensure_ascii=False)


def render_text(report: Report) -> str:
    # The same report as words for humans: one line per column, evidence
    # indented under it, and -- the honesty rule -- every Dimension member
    # with no result gets a 'not evaluated' line, so a three-column report
    # is never mistaken for a complete one.
    lines = [f"task {report.task_id} (trajectory {report.trajectory_task_id})"]
    for result in report.results:
        lines.append(f"  {result.dimension.value} {result.verdict.value}: {result.reason}")
        for item in result.evidence:
            where = f"step {item.step}" if item.step is not None else "no step"
            if item.field_name is not None:
                where = f"{where}, {item.field_name}"
            # Display-only: a None value is 'no citation to show', so the
            # ': ...' tail is dropped -- JSON keeps "value": null.
            tail = "" if item.value is None else f": {json.dumps(item.value, ensure_ascii=False)}"
            lines.append(f"    {item.kind} ({where}){tail}")
    evaluated = {result.dimension for result in report.results}
    for dimension in Dimension:
        if dimension not in evaluated:
            lines.append(f"  not evaluated: {dimension.value}")
    for item in report.discontinuities:
        lines.append(f"  discontinuity: {item}")
    return "\n".join(lines)
