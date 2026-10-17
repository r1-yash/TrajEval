# Strict validation of a judge's raw text answer: parse the JSON, check the
# shape and every rule, or raise JudgeOutputError. Never guesses, never
# repairs -- an unusable answer becomes an ERROR column, not a made-up verdict.

from __future__ import annotations

import json
from typing import Any

from trajecteval.judge.base import JudgeNote, JudgeResult
from trajecteval.models import Trajectory
from trajecteval.results import Verdict

# The three lowercase words the prompt demands; case is part of the contract
# ("PASS" is an unknown verdict, not something to fold).
_VERDICT_WORDS = {"pass": Verdict.PASS, "warn": Verdict.WARN, "fail": Verdict.FAIL}

# The closed shape of the answer: exactly these keys, no more, no less.
_TOP_KEYS = frozenset(("verdict", "reason", "notes"))
_NOTE_KEYS = frozenset(("step", "kind", "detail"))

# How much of the raw answer rides along in an error / evidence excerpt.
_EXCERPT_LIMIT = 200


class JudgeOutputError(Exception):
    # The judge's answer failed validation. The quality grader turns this
    # into an ERROR result (the can't-judge lane); it is not a TrajEvalError
    # (no input file is broken) and never a verdict.

    def __init__(
        self,
        message: str,
        *,
        bad_step: int | None = None,
        output_excerpt: str = "",
    ) -> None:
        super().__init__(message)
        # The step number a note cited that does not exist, when that is the
        # problem -- the ERROR evidence names it.
        self.bad_step = bad_step
        # The start of the raw answer, for debugging the provider's output.
        self.output_excerpt = output_excerpt


def _fail(
    message: str, raw: str, *, bad_step: int | None = None
) -> JudgeOutputError:
    return JudgeOutputError(
        message, bad_step=bad_step, output_excerpt=raw[:_EXCERPT_LIMIT]
    )


def _parse_note(item: Any, position: int, raw: str) -> JudgeNote:
    where = f"notes[{position}]"
    if not isinstance(item, dict):
        raise _fail(f"{where} must be an object, got {type(item).__name__}", raw)
    keys = set(item)
    if keys != _NOTE_KEYS:
        missing = sorted(_NOTE_KEYS - keys)
        extra = sorted(keys - _NOTE_KEYS)
        parts = []
        if missing:
            parts.append(f"missing {', '.join(missing)}")
        if extra:
            parts.append(f"unknown {', '.join(extra)}")
        raise _fail(
            f"{where} keys are wrong ({'; '.join(parts)}); "
            "expected exactly step, kind, detail",
            raw,
        )
    step = item["step"]
    if step is not None and (isinstance(step, bool) or not isinstance(step, int)):
        raise _fail(f"{where}.step must be a positive int or null, got {step!r}", raw)
    if step is not None and step < 1:
        raise _fail(f"{where}.step must be 1-based, got {step}", raw, bad_step=step)
    kind = item["kind"]
    if not isinstance(kind, str) or not kind:
        raise _fail(f"{where}.kind must be a non-empty string, got {kind!r}", raw)
    detail = item["detail"]
    if not isinstance(detail, str) or not detail:
        raise _fail(f"{where}.detail must be a non-empty string, got {detail!r}", raw)
    return JudgeNote(step=step, kind=kind, detail=detail)


def parse_judge_output(raw: str, trajectory: Trajectory) -> JudgeResult:
    # raw is the model's text; the trajectory is needed because every cited
    # step number must exist in *this* episode (loaders guarantee gapless
    # 1-based indices, so "step <= len(steps)" is the whole check).
    text = raw.strip()
    if not text:
        raise _fail("judge returned no text", raw)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise _fail(f"judge output is not valid JSON: {exc}", raw) from None
    if not isinstance(data, dict):
        raise _fail(
            f"judge output must be a JSON object, got {type(data).__name__}", raw
        )
    keys = set(data)
    if keys != _TOP_KEYS:
        missing = sorted(_TOP_KEYS - keys)
        extra = sorted(keys - _TOP_KEYS)
        parts = []
        if missing:
            parts.append(f"missing {', '.join(missing)}")
        if extra:
            parts.append(f"unknown {', '.join(extra)}")
        raise _fail(
            f"judge output keys are wrong ({'; '.join(parts)}); "
            "expected exactly verdict, reason, notes",
            raw,
        )
    verdict = _VERDICT_WORDS.get(data["verdict"])
    if verdict is None:
        raise _fail(
            f"unknown verdict {data['verdict']!r}; "
            "expected exactly 'pass', 'warn', or 'fail' (lowercase)",
            raw,
        )
    reason = data["reason"]
    if not isinstance(reason, str) or not reason.strip():
        raise _fail(f"judge 'reason' must be a non-empty string, got {reason!r}", raw)
    raw_notes = data["notes"]
    if not isinstance(raw_notes, list):
        raise _fail(
            f"judge 'notes' must be a list, got {type(raw_notes).__name__}", raw
        )
    notes = tuple(
        _parse_note(item, position, raw) for position, item in enumerate(raw_notes)
    )
    for note in notes:
        if note.step is not None and note.step > len(trajectory.steps):
            raise _fail(
                f"note cites step {note.step} but the trajectory has "
                f"{len(trajectory.steps)} step(s)",
                raw,
                bad_step=note.step,
            )
    # The verdict rules from the prompt, enforced on this side too: a WARN
    # with nothing behind it and a FAIL with no citable step are both
    # unusable answers, not verdicts to keep.
    if verdict is Verdict.WARN and not notes:
        raise _fail("a 'warn' verdict needs at least one note", raw)
    if verdict is Verdict.FAIL and not any(
        note.step is not None for note in notes
    ):
        raise _fail(
            "a 'fail' verdict needs at least one note citing a real step number",
            raw,
        )
    return JudgeResult(verdict=verdict, reason=reason.strip(), notes=notes)
