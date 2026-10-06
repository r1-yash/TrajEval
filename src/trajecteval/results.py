"""Shared result vocabulary: what every grader and the judge report back.

One report card, filled out the same way by everyone. Step 7's report
renderer loops over ``GraderResult`` objects and prints verdict + reason +
evidence **without any per-grader special cases** — that genericity is the
whole reason this module exists and why it was built before any grader.

Verdict semantics
-----------------
PASS    a rule was satisfied. May cite evidence — e.g. a *recovery note*
        ("step 9 undid a near-miss"), which the sticky-mistake rule
        promises to surface.
FAIL    a rule was broken. Must cite evidence.
WARN    **no rule was broken, but something noteworthy happened** (e.g. an
        empty trajectory: bounds checked nothing, so a clean PASS would
        oversell it). Must cite the noteworthy thing as evidence.
        Deterministic graders MAY emit WARN, not only the LLM judge — but if
        an actual rule was broken, that is FAIL, never WARN.
ERROR   the grader itself could not judge (missing spec field, malformed
        input). May cite evidence pointing at *why judging was blocked*
        (typically a malformed spec field, step=None) — a diagnostic, never
        an accusation against the trajectory. ERROR is never conflated with
        FAIL: distinct ``Verdict`` members, unequal as values, and calling
        for opposite responses (fix the setup vs. fix the trajectory).

Evidence is neutral
-------------------
Evidence is **facts a result cites, not proof of a problem**. The verdict
carries the interpretation; the evidence just points at what was observed.
Step 7's renderer must therefore present evidence uniformly
(``step · kind · field_name=value``) and must never assemble a "problems"
list from evidence presence alone — a PASS's recovery note and a FAIL's
smoking gun look identical at the evidence level, and that is the point.

Construction invariants (enforced in __post_init__, raised as ValueError —
these are bugs in grader *code*, not bad trajectory files):

* ``dimension`` (a closed ``Dimension`` member) and ``reason`` are
  non-empty — Step 7 prints a reason line for every result, so a blank one
  is a grader that didn't explain what it checked.
* FAIL and WARN carry at least one evidence item ("evidence or it didn't
  happen"); PASS and ERROR may carry any number.
* each evidence item is an ``Evidence`` with a non-empty ``kind`` and, when
  present, a 1-based ``step``.

``dimension`` is a closed enum (not a free string) because the comparison
table keys columns by it — a typo like "bound" would otherwise silently
become a fifth column. Evidence ``kind``, by contrast, is an *open* string:
kinds are owned by the producers (each grader's findings), verdicts and
dimensions are consumed by the report and comparison table.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class Verdict(StrEnum):
    """Closed set of outcomes. Not a free string: 'fail' or 'Fail' are not
    verdicts. StrEnum members *are* their string value, so JSON output needs
    no converter and ``Verdict("FAIL")`` rejects unknown values."""

    PASS = "PASS"
    FAIL = "FAIL"
    WARN = "WARN"
    ERROR = "ERROR"


class Dimension(StrEnum):
    """Closed set of grading axes — the four dimensions of TrajEval.

    The comparison table keys its columns by these, so a typo like "bound"
    must fail at the grader's birth instead of silently creating a fifth
    column later. Same closed-set logic as Verdict."""

    FINAL_STATE = "final_state"
    BOUNDS = "bounds"
    CRITICAL = "critical"
    TRAJECTORY_QUALITY = "trajectory_quality"


def _require_non_empty_str(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{where} must be a non-empty string, got {value!r}")
    return value


@dataclass(frozen=True)
class Evidence:
    """Typed **facts cited by** a result: what was observed, and where.

    ``step=None`` means the finding applies to the **whole trajectory**, not
    any single step (e.g. "this episode has no actions at all"). None is a
    meaningful value here, not missing data — and it must survive JSON
    round-trips as null.

    Evidence is neutral: it does not assert pass or fail. The surrounding
    ``GraderResult.verdict`` says how to read it.
    """

    kind: str                      # open label, e.g. "price_exceeds_limit"
    step: int | None = None        # 1-based step index, or None = whole episode
    # Named field_name (not 'field') so it never collides with
    # dataclasses.field if that import is ever added to this module.
    field_name: str | None = None  # the specific field involved, if any
    value: Any = None              # the specific value involved, if any

    def __post_init__(self) -> None:
        _require_non_empty_str(self.kind, "evidence 'kind'")
        if self.step is not None:
            # bool is a subclass of int; reject it explicitly (same trap as Step 2)
            if isinstance(self.step, bool) or not isinstance(self.step, int) or self.step < 1:
                raise ValueError(
                    f"evidence 'step' must be a 1-based int or None, got {self.step!r}"
                )
        if self.field_name is not None:
            _require_non_empty_str(self.field_name, "evidence 'field_name'")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "step": self.step,
            "field_name": self.field_name,
            "value": self.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Evidence:
        if not isinstance(data, dict):
            raise ValueError(f"evidence must be an object, got {type(data).__name__}")
        return cls(
            kind=data["kind"],
            step=data.get("step"),
            field_name=data.get("field_name"),
            value=data.get("value"),
        )

# Invariants enforced in __post_init__ (see module docs for the reasoning):
#   FAIL/WARN: at least one evidence item (assert something -> must show it)
#   ERROR:     any number -- evidence cites *why judging was blocked*
#              (typically a malformed spec field, step=None), never an
#              accusation against the trajectory (that's FAIL's job)
#   plus: evidence stored as a tuple of Evidence; dimension/verdict coerced
#   into their closed enums; dimension and reason non-empty strings
@dataclass(frozen=True)
class GraderResult:
    """One dimension's verdict for one trajectory — the shared report card."""

    dimension: Dimension       # closed set: final_state | bounds | critical | trajectory_quality
    verdict: Verdict
    reason: str
    evidence: tuple[Evidence, ...] = ()

    def __post_init__(self) -> None:
        _require_non_empty_str(self.dimension, "result 'dimension'")
        # Coerce plain strings ("bounds") into Dimension members; a typo like
        # "bound" raises with our message instead of becoming a new column.
        if not isinstance(self.dimension, Dimension):
            try:
                object.__setattr__(self, "dimension", Dimension(self.dimension))
            except ValueError:
                raise ValueError(
                    f"invalid dimension {self.dimension!r}; "
                    f"expected one of {', '.join(d.value for d in Dimension)}"
                ) from None

        # Coerce plain strings ("FAIL") into Verdict members; unknown values
        # raise with our message instead of a bare enum ValueError.
        if not isinstance(self.verdict, Verdict):
            try:
                object.__setattr__(self, "verdict", Verdict(self.verdict))
            except ValueError:
                raise ValueError(
                    f"invalid verdict {self.verdict!r}; "
                    f"expected one of {', '.join(v.value for v in Verdict)}"
                ) from None

        _require_non_empty_str(self.reason, "result 'reason'")

        # Accept a list at construction (ergonomic), store a tuple (immutable).
        if isinstance(self.evidence, list):
            object.__setattr__(self, "evidence", tuple(self.evidence))
        if not isinstance(self.evidence, tuple):
            raise ValueError(
                f"result 'evidence' must be a tuple of Evidence, "
                f"got {type(self.evidence).__name__}"
            )
        for item in self.evidence:
            if not isinstance(item, Evidence):
                raise ValueError(
                    f"result 'evidence' items must be Evidence, "
                    f"got {type(item).__name__}"
                )

        # Verdict-specific rules: FAIL/WARN assert something (proof mandatory);
        # ERROR may cite the evidence that blocked judging, but never accuses.
        if self.verdict in (Verdict.FAIL, Verdict.WARN) and not self.evidence:
            raise ValueError(f"{self.verdict.value} results must cite at least one evidence item")

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly plain dict. Verdict and dimension become plain strings."""
        return {
            "dimension": self.dimension.value,
            "verdict": self.verdict.value,
            "reason": self.reason,
            "evidence": [item.to_dict() for item in self.evidence],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GraderResult:
        if not isinstance(data, dict):
            raise ValueError(f"result must be an object, got {type(data).__name__}")
        for key in ("dimension", "verdict", "reason"):
            if key not in data:
                raise ValueError(f"result missing '{key}'")
        return cls(
            dimension=data["dimension"],     # __post_init__ coerces str -> Dimension
            verdict=data["verdict"],   # __post_init__ coerces str -> Verdict
            reason=data["reason"],
            evidence=tuple(
                Evidence.from_dict(item) for item in data.get("evidence", ())
            ),
        )
