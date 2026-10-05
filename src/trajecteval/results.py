
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


def _require_non_empty_str(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{where} must be a non-empty string, got {value!r}")
    return value


@dataclass(frozen=True)
class Evidence:
    """Typed proof attached to a result: what was found, and where.

    step=None, means the finding applies to the whole trajectory, not
    any single step (e.g. "this episode has no actions at all"). None is a
    meaningful value here, not missing data or anything and it must survive JSON
    round-trips as null.
    """

    kind: str                      # open label, e.g. "price_exceeds_limit"
    step: int | None = None        # 1-based step index, or None = whole episode
    field: str | None = None       # the specific field involved, if any
    value: Any = None              # the specific value involved, if any

    def __post_init__(self) -> None:
        _require_non_empty_str(self.kind, "evidence 'kind'")
        if self.step is not None:
            # bool is a subclass of int; reject it explicitly (same trap as Step 2)
            if isinstance(self.step, bool) or not isinstance(self.step, int) or self.step < 1:
                raise ValueError(
                    f"evidence 'step' must be a 1-based int or None, got {self.step!r}"
                )
        if self.field is not None:
            _require_non_empty_str(self.field, "evidence 'field'")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "step": self.step,
            "field": self.field,
            "value": self.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Evidence:
        if not isinstance(data, dict):
            raise ValueError(f"evidence must be an object, got {type(data).__name__}")
        return cls(
            kind=data["kind"],
            step=data.get("step"),
            field=data.get("field"),
            value=data.get("value"),
        )

#if error then no evidence, if fail/warn then at least one evidence in this class
# it also does not make sense to have a pass with evidence, so we also enforce that
# and we also enforce that the evidence is a tuple of Evidence objects
# and we also enforce that the dimension and reason are non-empty strings
@dataclass(frozen=True)
class GraderResult:
    """One dimension's verdict for one trajectory — the shared report card."""

    dimension: str                 # which axis: "final_state", "bounds", ...
    verdict: Verdict
    reason: str
    evidence: tuple[Evidence, ...] = ()

    def __post_init__(self) -> None:
        _require_non_empty_str(self.dimension, "result 'dimension'")

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

        # Verdict-specific rules: ERROR asserts nothing (so no proof);
        # FAIL/WARN assert something (so proof is mandatory).
        if self.verdict is Verdict.ERROR and self.evidence:
            raise ValueError("ERROR results must carry no evidence (no judgment was made)")
        if self.verdict in (Verdict.FAIL, Verdict.WARN) and not self.evidence:
            raise ValueError(f"{self.verdict.value} results must cite at least one evidence item")

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly plain dict. Verdict becomes its plain string."""
        return {
            "dimension": self.dimension,
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
            dimension=data["dimension"],
            verdict=data["verdict"],   # __post_init__ coerces str -> Verdict
            reason=data["reason"],
            evidence=tuple(
                Evidence.from_dict(item) for item in data.get("evidence", ())
            ),
        )
