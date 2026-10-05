
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

    ``step=None`` means the finding applies to the **whole trajectory**, not
    any single step (e.g. "this episode has no actions at all"). None is a
    meaningful value here, not missing data — and it must survive JSON
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
