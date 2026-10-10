# compare(): one row per trace, one column per dimension -- side by side.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from trajecteval.report import Report
from trajecteval.results import Dimension, Verdict

# The gap marker, in the same words render_text uses for a missing column.
NOT_EVALUATED = "not evaluated"


@dataclass(frozen=True)
class Comparison:
    # The table itself: row labels, the fixed column set, and one cell string
    # per (trace, dimension). Cells are display-ready text -- verdict, with a
    # step hint for FAIL/WARN citations, or the NOT_EVALUATED marker.
    traces: tuple[str, ...]
    columns: tuple[Dimension, ...]
    cells: Mapping[str, Mapping[str, str]]  # trace -> dimension value -> cell

    def cell(self, trace: str, dimension: Dimension) -> str:
        return self.cells[trace][dimension.value]


def _cell_text(result: Any) -> str:
    # A result becomes one cell: the verdict, plus "(step N)" for FAIL/WARN
    # that cite a step (the first one that does). ERROR stays bare -- crash
    # and missing-section evidence are spec-side or whole-trace, no one step.
    if result is None:
        return NOT_EVALUATED
    verdict = result.verdict.value
    if result.verdict in (Verdict.FAIL, Verdict.WARN):
        for item in result.evidence:
            if item.step is not None:
                return f"{verdict} (step {item.step})"
    return verdict


def compare(reports: Mapping[str, Report]) -> Comparison:
    # Side-by-side view across traces. Columns are ALWAYS every Dimension
    # member in enum order: a dimension nobody evaluated still shows, full of
    # markers -- never a silently dropped column (the Step 7 honesty rule,
    # applied across traces). Row order follows the mapping's insertion order.
    if not reports:
        raise ValueError("compare() needs at least one report")
    columns = tuple(Dimension)
    traces = tuple(reports)
    cells: dict[str, dict[str, str]] = {}
    for name, report in reports.items():
        by_dimension = {result.dimension: result for result in report.results}
        cells[name] = {column.value: _cell_text(by_dimension.get(column)) for column in columns}
    return Comparison(traces=traces, columns=columns, cells=cells)


def render_table(comparison: Comparison) -> str:
    # Plain aligned text: a 'trace' label column plus one column per
    # dimension, each padded to its widest cell. No box drawing, no color --
    # greppable output that diff cleanly (the CLI prints this).
    headers = ["trace", *(column.value for column in comparison.columns)]
    rows = [
        [name, *(comparison.cells[name][column.value] for column in comparison.columns)]
        for name in comparison.traces
    ]
    widths = [
        max(len(row[index]) for row in [headers, *rows]) for index in range(len(headers))
    ]

    def line(cells: list[str]) -> str:
        return "  ".join(cell.ljust(width) for cell, width in zip(cells, widths)).rstrip()

    return "\n".join([line(headers), *(line(row) for row in rows)])
