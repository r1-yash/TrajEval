# A scripted judge for tests: hands back preset JudgeResult objects and
# counts its calls, so unit tests exercise the judge seam without ever
# spending API money or touching the network.

from __future__ import annotations

from collections.abc import Sequence

from trajecteval.judge.base import JudgeResult
from trajecteval.models import Trajectory
from trajecteval.results import Verdict
from trajecteval.task_spec import TaskSpec


class ScriptedJudge:
    # Implements the Judge protocol. judge() records the call, then raises
    # whatever you scripted it to raise, or pops the next preset result; the
    # last preset repeats when the queue runs out.

    def __init__(
        self,
        results: Sequence[JudgeResult] | None = None,
        *,
        raises: Exception | None = None,
    ) -> None:
        # (task_id, step count) per call -- tests assert both "called" and
        # "called exactly zero times".
        self.calls: list[tuple[str, int]] = []
        self._results = list(
            results
            if results is not None
            else [JudgeResult(verdict=Verdict.PASS, reason="scripted pass")]
        )
        if not self._results:
            raise ValueError("ScriptedJudge needs at least one preset result")
        self._raises = raises

    def judge(self, spec: TaskSpec, trajectory: Trajectory) -> JudgeResult:
        self.calls.append((spec.task_id, len(trajectory.steps)))
        if self._raises is not None:
            raise self._raises
        if len(self._results) > 1:
            return self._results.pop(0)
        return self._results[0]
