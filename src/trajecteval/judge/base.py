# The judge contract: what a judge is called, what it must return, and the
# rules its output must satisfy. The offline fake and the future LLM client
# both stand on this seam, so tests never need the network.

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from trajecteval.models import Trajectory
from trajecteval.results import Verdict
from trajecteval.task_spec import TaskSpec

# The judged set: PASS, FAIL, WARN. ERROR is deliberately absent -- it is the
# pipeline's crash lane, not the judge's to hand out.
_JUDGE_VERDICTS = (Verdict.PASS, Verdict.FAIL, Verdict.WARN)


# One observation the judge made, citing the step it is about (None = whole
# episode). kind is an open string -- same looseness as Evidence.kind, so the
# real judge's vocabulary is not frozen by a toy's.
@dataclass(frozen=True)
class JudgeNote:
    step: int | None
    kind: str
    detail: str = ""

    def __post_init__(self) -> None:
        # A note citing step 9 of a 5-step trajectory is broken judge code;
        # the check lives here so the citation is valid from birth.
        if self.step is not None and (
            isinstance(self.step, bool)
            or not isinstance(self.step, int)
            or self.step < 1
        ):
            raise ValueError(
                f"judge note step must be a 1-based int or None, got {self.step!r}"
            )
        if not isinstance(self.kind, str) or not self.kind:
            raise ValueError(
                f"judge note kind must be a non-empty string, got {self.kind!r}"
            )
        if not isinstance(self.detail, str):
            raise ValueError(
                f"judge note detail must be a string, got {self.detail!r}"
            )


# What a judge returns: a verdict, a one-sentence reason, and the notes that
# back it. The quality grader converts this to a GraderResult in-process, so
# -- unlike every other result type here -- it never crosses a file boundary
# and carries no to_dict/from_dict.
@dataclass(frozen=True)
class JudgeResult:
    verdict: Verdict
    reason: str
    notes: tuple[JudgeNote, ...] = ()

    def __post_init__(self) -> None:
        # Accept lists at construction, store tuples -- same rule as Report.
        if isinstance(self.notes, list):
            object.__setattr__(self, "notes", tuple(self.notes))
        if not isinstance(self.verdict, Verdict) or self.verdict not in _JUDGE_VERDICTS:
            raise ValueError(
                f"judge verdict must be PASS, FAIL, or WARN, got {self.verdict!r}; "
                "a judge that cannot judge raises, which becomes the crash lane"
            )
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError(
                f"judge reason must be a non-empty string, got {self.reason!r}"
            )
        for note in self.notes:
            if not isinstance(note, JudgeNote):
                raise ValueError(
                    f"judge notes must be JudgeNote items, got {note!r}"
                )


# runtime_checkable so isinstance() can check a judge at evaluate()'s gate.
@runtime_checkable
class Judge(Protocol):

    def judge(self, spec: TaskSpec, trajectory: Trajectory) -> JudgeResult:
        # Judge one trajectory's quality against one task; raise if unable.
        ...  # pragma: no cover - protocol body, never executed
