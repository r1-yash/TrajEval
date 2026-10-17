# The trajectory-quality column: adapts the Judge contract to the Grader
# contract. Judge code lives here, never in graders/ -- deterministic code
# and model-judged code do not share a module.

from __future__ import annotations

from typing import Any

from trajecteval.graders.base import task_id_mismatch_result
from trajecteval.judge.base import Judge, JudgeResult
from trajecteval.judge.validate import JudgeOutputError
from trajecteval.models import Trajectory
from trajecteval.results import Dimension, Evidence, GraderResult, Verdict
from trajecteval.task_spec import TaskSpec


def _invalid_output_result(dimension: Dimension, exc: JudgeOutputError) -> GraderResult:
    # The model answered, but not in a shape we can trust: ERROR (can't
    # judge), never a guessed verdict. Evidence carries the problem, the bad
    # step number when one was cited, and a short excerpt of the raw answer.
    value: dict[str, Any] = {"problem": str(exc)}
    if exc.bad_step is not None:
        value["bad_step"] = exc.bad_step
    if exc.output_excerpt:
        value["output_excerpt"] = exc.output_excerpt
    return GraderResult(
        dimension=dimension,
        verdict=Verdict.ERROR,
        reason=f"the trajectory-quality judge returned unusable output: {exc}",
        evidence=(Evidence(kind="judge_invalid_output", step=None, value=value),),
    )


class TrajectoryQualityGrader:
    # The trajectory_quality column of the report card: same Grader contract
    # as the deterministic three, but the verdict comes from a judge.

    dimension = Dimension.TRAJECTORY_QUALITY

    def __init__(self, judge: Judge) -> None:
        self._judge = judge

    def grade(self, spec: TaskSpec, trajectory: Trajectory) -> GraderResult:
        # Pinned order, same as the deterministic graders: wrong pairing
        # first, so an unrelated judge failure never masks it.
        mismatch = task_id_mismatch_result(spec, self.dimension, trajectory)
        if mismatch is not None:
            return mismatch
        if not trajectory.steps:
            # Nothing to judge. Calling a paid model to say "no path" would
            # spend money on an answer we already know; and an episode with
            # no actions did not demonstrably work cleanly, so WARN -- never
            # PASS. Same precedent as bounds/critical on an empty trajectory.
            return GraderResult(
                dimension=self.dimension,
                verdict=Verdict.WARN,
                reason="trajectory has no steps: no path to judge",
                evidence=(Evidence(kind="empty_trajectory", step=None),),
            )
        try:
            result = self._judge.judge(spec, trajectory)
        except JudgeOutputError as exc:
            # Unusable answer: the can't-judge lane. Any OTHER exception
            # (transport, judge bug) propagates, which evaluate() turns into
            # the crash lane -- the two failures stay tellable apart.
            return _invalid_output_result(self.dimension, exc)
        if not isinstance(result, JudgeResult):
            raise TypeError(
                f"judge returned {type(result).__name__}, expected JudgeResult"
            )
        return GraderResult(
            dimension=self.dimension,
            verdict=result.verdict,
            reason=result.reason,
            # Every note becomes evidence: kind is the open label, the step
            # citation survives, and the detail rides in value (JSON-native).
            evidence=tuple(
                Evidence(kind=note.kind, step=note.step, value={"detail": note.detail})
                for note in result.notes
            ),
        )
