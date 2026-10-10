# Tests for compare(): fixed columns across traces, the gap marker, and the
# exact text layout of the table.

from pathlib import Path

import pytest

from trajecteval import (
    Dimension,
    Evidence,
    GraderResult,
    Report,
    Trajectory,
    Verdict,
    compare,
    evaluate,
    load_spec,
    render_table,
)

REPO = Path(__file__).resolve().parents[1]
TASKS = REPO / "tasks"
FIXTURES = REPO / "tests" / "fixtures" / "travel_booking"


def _result(dimension: Dimension, verdict: Verdict, *steps: int) -> GraderResult:
    evidence = tuple(
        Evidence(kind="citation", step=step) for step in steps
    )
    return GraderResult(dimension=dimension, verdict=verdict, reason="r", evidence=evidence)


def _report(*results: GraderResult) -> Report:
    return Report(task_id="t", trajectory_task_id="t", results=results)


FULL = _report(
    _result(Dimension.FINAL_STATE, Verdict.PASS),
    _result(Dimension.BOUNDS, Verdict.PASS),
    _result(Dimension.CRITICAL, Verdict.PASS),
)


def test_columns_are_every_dimension_in_enum_order():
    table = compare({"a": FULL})
    assert table.columns == tuple(Dimension)


def test_verdict_cells_come_from_each_reports_results():
    partial = _report(
        _result(Dimension.FINAL_STATE, Verdict.FAIL, 2),
        _result(Dimension.BOUNDS, Verdict.PASS),
    )
    table = compare({"clean": FULL, "partial": partial})
    assert table.cell("clean", Dimension.CRITICAL) == "PASS"
    assert table.cell("partial", Dimension.FINAL_STATE) == "FAIL (step 2)"
    assert table.cell("partial", Dimension.BOUNDS) == "PASS"


def test_missing_dimension_shows_the_marker_not_a_dropped_column():
    # The Step 7 honesty rule across traces: the column exists, full of markers.
    partial = _report(_result(Dimension.FINAL_STATE, Verdict.PASS))
    table = compare({"partial": partial})
    assert Dimension.BOUNDS in table.columns
    assert table.cell("partial", Dimension.BOUNDS) == "not evaluated"
    assert table.cell("partial", Dimension.TRAJECTORY_QUALITY) == "not evaluated"


def test_fail_cell_cites_the_first_evidence_that_has_a_step():
    result = GraderResult(
        dimension=Dimension.BOUNDS,
        verdict=Verdict.FAIL,
        reason="r",
        evidence=(
            Evidence(kind="whole_trace", step=None),
            Evidence(kind="unlisted_action", step=3),
            Evidence(kind="unlisted_action", step=5),
        ),
    )
    table = compare({"t": _report(result)})
    assert table.cell("t", Dimension.BOUNDS) == "FAIL (step 3)"


def test_warn_with_only_stepless_evidence_stays_bare():
    result = GraderResult(
        dimension=Dimension.FINAL_STATE,
        verdict=Verdict.WARN,
        reason="r",
        evidence=(Evidence(kind="empty_section", field_name="expected_final_state"),),
    )
    table = compare({"t": _report(result)})
    assert table.cell("t", Dimension.FINAL_STATE) == "WARN"


def test_error_cells_stay_bare():
    # Crash and missing-section evidence are spec-side or whole-trace; no step hint.
    result = GraderResult(
        dimension=Dimension.CRITICAL,
        verdict=Verdict.ERROR,
        reason="r",
        evidence=(Evidence(kind="grader_crash", value={"exception": "ValueError"}),),
    )
    table = compare({"t": _report(result)})
    assert table.cell("t", Dimension.CRITICAL) == "ERROR"


def test_row_order_follows_the_given_mapping():
    table = compare({"second": FULL, "first": FULL})
    assert table.traces == ("second", "first")


def test_empty_mapping_raises():
    with pytest.raises(ValueError, match="at least one report"):
        compare({})


def test_render_table_pins_the_exact_layout():
    clean = FULL
    delayed = _report(
        _result(Dimension.FINAL_STATE, Verdict.FAIL, 2),
        _result(Dimension.BOUNDS, Verdict.PASS),
        GraderResult(
            dimension=Dimension.CRITICAL,
            verdict=Verdict.WARN,
            reason="r",
            evidence=(Evidence(kind="empty_section"),),  # stepless -> bare WARN
        ),
    )
    table = compare({"clean": clean, "delayed": delayed})
    assert render_table(table) == "\n".join(
        [
            "trace    final_state    bounds  critical  trajectory_quality",
            "clean    PASS           PASS    PASS      not evaluated",
            "delayed  FAIL (step 2)  PASS    WARN      not evaluated",
        ]
    )


def test_table_over_real_reports():
    # The end-to-end shape the CLI will print in Step 10.
    spec = load_spec("travel_booking", tasks_dir=TASKS)
    clean = evaluate(Trajectory.from_json_file(FIXTURES / "clean.json"), spec)
    unlisted = evaluate(Trajectory.from_json_file(FIXTURES / "unlisted_action.json"), spec)
    table = compare({"clean": clean, "unlisted_action": unlisted})
    assert table.cell("clean", Dimension.BOUNDS) == "PASS"
    assert table.cell("unlisted_action", Dimension.BOUNDS) == "FAIL (step 3)"
    assert table.cell("unlisted_action", Dimension.TRAJECTORY_QUALITY) == "not evaluated"
