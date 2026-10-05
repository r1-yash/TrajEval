
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

