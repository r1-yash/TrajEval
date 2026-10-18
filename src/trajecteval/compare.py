# compare(): several Report objects as one table -- every dimension visible,
# caller-ordered rows, and cells that cite steps without replacing the report.

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from trajecteval.report import Report
from trajecteval.results import Dimension, GraderResult, Verdict

# A dimension with no result renders this marker -- the honesty rule from
# render_text's 'not evaluated' lines: a column is never dropped.
NOT_EVALUATED = "not evaluated"

# A column that could not be judged at all gets this bang prefix so ERROR can
# never be misread as FAIL in plain text; renderers print a legend spelling
# it out.
ERROR_MARK = "!ERROR"

# How many stepped-evidence step numbers a hint lists before summarizing the
# rest as '+N more' (keeps cell width bounded).
_HINT_STEPS = 3


def _cell_text(report: Report, dimension: Dimension) -> str:
    # One cell: the verdict plus the hint rule. Missing result ->
    # 'not evaluated'; ERROR -> the bang marker, never a hint (its evidence
    # is diagnostic, not a citation about the trajectory); PASS -> bare
    # (a clean cell stays clean); FAIL/WARN -> sorted deduplicated step
    # numbers, or '(no single step)' when its evidence cites none. A
    # quality_score evidence item (the judge's 0-1 number) renders beside
    # the verdict; only the quality grader ever emits one, and ERROR lanes
    # carry none.
    for result in report.results:
        if result.dimension is dimension:
            break
    else:
        return NOT_EVALUATED
    score = next(
        (item.value for item in result.evidence if item.kind == "quality_score"),
        None,
    )
    scored = f" {score:.2f}" if isinstance(score, (int, float)) else ""
    if result.verdict is Verdict.ERROR:
        return ERROR_MARK
    if result.verdict is Verdict.PASS:
        return f"PASS{scored}"
    # GraderResult rejects evidence-less FAIL/WARN, so evidence exists here.
    steps = sorted({item.step for item in result.evidence if item.step is not None})
    if not steps:
        # Whole-episode or spec-side finding (empty_section, final_state
        # never reached): no single step is to blame.
        return f"{result.verdict.value}{scored} (no single step)"
    if len(steps) == 1:
        return f"{result.verdict.value}{scored} (step {steps[0]})"
    if len(steps) <= _HINT_STEPS:
        return f"{result.verdict.value}{scored} (steps {', '.join(str(s) for s in steps)})"
    shown = ", ".join(str(s) for s in steps[:_HINT_STEPS])
    return (
        f"{result.verdict.value}{scored} "
        f"(steps {shown}, +{len(steps) - _HINT_STEPS} more)"
    )


def _task_text(report: Report) -> str:
    # The spec's task id; a trace graded against a different spec shows both
    # ids, because disagreeing ids are the whole story of that row.
    if report.task_id == report.trajectory_task_id:
        return report.task_id
    return f"{report.task_id} / {report.trajectory_task_id}"


def _discont_text(report: Report) -> str:
    # Recorder discontinuities as a count (or '-' when none): cross-trace,
    # "3 of 7 traces have recorder gaps" is the pattern only this view shows.
    return str(len(report.discontinuities)) if report.discontinuities else "-"


def _headers(columns: tuple[Dimension, ...]) -> list[str]:
    return ["trace", "task", "discont", *[column.value for column in columns]]


def _row_cells(
    label: str, report: Report, columns: tuple[Dimension, ...]
) -> list[str]:
    # Shared by both renderers so the text and markdown tables cannot drift.
    return [
        label,
        _task_text(report),
        _discont_text(report),
        *[_cell_text(report, column) for column in columns],
    ]


@dataclass(frozen=True)
class Comparison:
    # Validated (label, report) pairs in the caller's order plus the fixed
    # column set. Reports carry no name of their own -- every travel_booking
    # fixture shares a task_id -- so labels come from the caller (Step 10
    # passes file stems). No overall verdict anywhere (README FAQ).

    entries: tuple[tuple[str, Report], ...]
    columns: tuple[Dimension, ...] = tuple(Dimension)

    def cell(self, label: str, column: Dimension) -> str:
        # Public cell lookup: exactly the text the renderers put in the table.
        for entry_label, report in self.entries:
            if entry_label == label:
                return _cell_text(report, column)
        raise KeyError(label)


def compare(reports: Sequence[tuple[str, Report]]) -> Comparison:
    # Caller-order rows, caller-supplied labels. Empty list, a bad label, a
    # non-Report, or a duplicated label are caller mistakes -> ValueError,
    # the same pre-judging gate style as evaluate()'s grader list.
    entries = tuple(reports)
    if not entries:
        raise ValueError("compare() needs at least one report; got an empty list")
    seen: list[str] = []
    for entry in entries:
        if not isinstance(entry, (tuple, list)) or len(entry) != 2:
            raise ValueError(
                f"compare() takes (label, report) pairs; got {entry!r}"
            )
        label, report = entry
        if not isinstance(label, str) or not label:
            raise ValueError(f"report label must be a non-empty string, got {label!r}")
        if not isinstance(report, Report):
            raise ValueError(
                f"report for label {label!r} must be a Report, "
                f"got {type(report).__name__}"
            )
        # Two rows with one name cannot be told apart by a human; a mapping
        # would have silently dropped one, so pairs are checked here.
        if label in seen:
            raise ValueError(
                f"duplicate report label {label!r}; rows must be tellable apart"
            )
        seen.append(label)
    return Comparison(entries=entries)


def render_table(comparison: Comparison) -> str:
    # Plain aligned text: metadata first (trace, task, discont), then every
    # Dimension member in enum order -- a missing result renders the
    # not-evaluated marker, a column is never dropped -- and the legend as
    # the final line even when no ERROR is present (format stability).
    headers = _headers(comparison.columns)
    rows = [
        _row_cells(label, report, comparison.columns)
        for label, report in comparison.entries
    ]
    widths = [
        max([len(headers[i]), *(len(row[i]) for row in rows)])
        for i in range(len(headers))
    ]

    def _pad(row: list[str]) -> str:
        # Two-space gutters, cells padded to the widest entry; the last
        # column is not padded, so no line ends in spaces.
        return "  ".join(
            cell.ljust(widths[i]) if i < len(row) - 1 else cell
            for i, cell in enumerate(row)
        )

    lines = [_pad(headers)]
    lines.extend(_pad(row) for row in rows)
    lines.append(
        f"legend: {ERROR_MARK} = could not judge (setup problem); "
        "FAIL/WARN = judged -- details in the full report"
    )
    return "\n".join(lines)


def render_table_markdown(comparison: Comparison) -> str:
    # The same cells as render_table in Markdown pipes: no padding, a
    # separator row after the header, legend as a trailing italic line.
    lines = [
        "| " + " | ".join(_headers(comparison.columns)) + " |",
        "|" + "|".join("---" for _ in _headers(comparison.columns)) + "|",
    ]
    for label, report in comparison.entries:
        lines.append(
            "| " + " | ".join(_row_cells(label, report, comparison.columns)) + " |"
        )
    lines.append(
        f"*legend: `{ERROR_MARK}` = could not judge (setup problem); "
        "FAIL/WARN = judged -- details in the full report*"
    )
    return "\n".join(lines)
