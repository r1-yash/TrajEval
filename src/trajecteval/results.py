

from __future__ import annotations

import math
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

    The comparison table keys its columns by these, so a typo like "bound" or "bounds"
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


def _require_json_native(value: Any, where: str) -> None:
    """Reject values that would not survive a JSON round trip unchanged.

    Tuples come back as lists, int dict keys come back as strings, and
    NaN/Infinity are not valid JSON -- each would silently break the
    ``from_dict(to_dict(x)) == x`` guarantee the Step 7 report relies on.
    """
    if value is None or isinstance(value, (bool, str)):
        return
    if isinstance(value, int):  # bool is a subclass of int; handled above
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{where} must be JSON-native, got non-finite float {value!r}")
        return
    if isinstance(value, list):
        for item in value:
            _require_json_native(item, f"{where}[...]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(
                    f"{where} keys must be strings (JSON coerces them), "
                    f"got {type(key).__name__} key {key!r}"
                )
            _require_json_native(item, f"{where}[{key!r}]")
        return
    raise ValueError(f"{where} must be JSON-native, got {type(value).__name__}")


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

    kind: str                      
    step: int | None = None        
    field_name: str | None = None 
    value: Any = None              

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
        _require_json_native(self.value, "evidence 'value'")

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
        if "kind" not in data:
            raise ValueError("evidence missing 'kind'")
        return cls(
            kind=data["kind"],
            step=data.get("step"),
            field_name=data.get("field_name"),
            value=data.get("value"),
        )

#   FAIL/WARN: at least one evidence item (assert something -> must show it)
#   ERROR:     any number -- evidence cites *why judging was blocked*
#              (typically a malformed spec field, step=None), never an
#              accusation against the trajectory (that's FAIL's job)
#   plus: evidence stored as a tuple of Evidence; dimension/verdict coerced

#this class basically represents the result of grading a trajectory along a specific dimension, including the verdict, reason, and any supporting evidence.
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
        raw_evidence = data.get("evidence", [])
        # Check the container before iterating: a dict would silently iterate
        # its keys, an int would raise a bare TypeError.
        if not isinstance(raw_evidence, list):
            raise ValueError(
                f"result 'evidence' must be a list, got {type(raw_evidence).__name__}"
            )
        return cls(
            dimension=data["dimension"],    
            verdict=data["verdict"],   
            reason=data["reason"],
            evidence=tuple(Evidence.from_dict(item) for item in raw_evidence),
        )
