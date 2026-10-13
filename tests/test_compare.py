# Tests for compare(): the cross-trace table's data half -- caller-ordered
# rows, every Dimension member a column, the hint rule, and the gates.

import json
from pathlib import Path

import pytest

from trajecteval import (
    Dimension,
    Evidence,
    GraderResult,
    Report,
    Verdict,
    compare,
    evaluate,
    load_spec,
)
from trajecteval.models import Trajectory

REPO = Path(__file__).resolve().parents[1]
TASKS = REPO / "tasks"
FIXTURES = REPO / "tests" / "fixtures" / "travel_booking"
TABLE = json.loads((FIXTURES / "expected_verdicts.json").read_text(encoding="utf-8"))

# The three columns the deterministic graders own; the answer key knows
# only these until Step 9's judge exists.
DETERMINISTIC = ("final_state", "bounds", "critical")


def _spec():
    return load_spec("travel_booking", tasks_dir=TASKS)


def _fixture(name: str) -> Trajectory:
    return Trajectory.from_json_file(FIXTURES / f"{name}.json")


def _report_with(*results: GraderResult) -> Report:
    return Report(task_id="t", trajectory_task_id="t", results=results)


def _result(verdict: Verdict, *steps: int, kind: str = "some_kind") -> GraderResult:
    return GraderResult(
        dimension=Dimension.FINAL_STATE,
        verdict=verdict,
        reason="because",
        evidence=tuple(Evidence(kind=kind, step=step) for step in steps),
    )


# -- the seven fixtures through the table, against the answer key -----------


@pytest.mark.parametrize("name", sorted(TABLE["verdicts"]), ids=sorted(TABLE["verdicts"]))
def test_table_cells_match_the_answer_key(name: str):
    comparison = compare([(name, evaluate(_fixture(name), _spec()))])
    for dimension in DETERMINISTIC:
        # The hint is display sugar after the verdict word; the verdict
        # itself must equal the key.
        cell = comparison.cell(name, Dimension(dimension))
        assert cell.split(" (")[0] == TABLE["verdicts"][name][dimension]


@pytest.mark.parametrize("name", sorted(TABLE["verdicts"]), ids=sorted(TABLE["verdicts"]))
def test_trajectory_quality_is_never_evaluated_for_the_fixtures(name: str):
    # Step 9's judge does not exist yet: every real table shows the marker
    # in that column, never a dropped column and never a silent PASS.
    comparison = compare([(name, evaluate(_fixture(name), _spec()))])
    assert comparison.cell(name, Dimension.TRAJECTORY_QUALITY) == "not evaluated"


def test_columns_are_all_dimensions_in_enum_order():
    comparison = compare([("a", _report_with())])
    assert comparison.columns == (
        Dimension.FINAL_STATE,
        Dimension.BOUNDS,
        Dimension.CRITICAL,
        Dimension.TRAJECTORY_QUALITY,
    )


def test_a_missing_dimension_renders_the_marker():
    comparison = compare([("a", _report_with(_result(Verdict.PASS, 1)))])
    assert comparison.cell("a", Dimension.FINAL_STATE) == "PASS"
    assert comparison.cell("a", Dimension.BOUNDS) == "not evaluated"


def test_cell_for_an_unknown_label_raises_key_error():
    comparison = compare([("a", _report_with())])
    with pytest.raises(KeyError):
        comparison.cell("b", Dimension.BOUNDS)


# -- the hint rule, one test per branch -------------------------------------


def test_hint_one_step_is_singular():
    comparison = compare([("a", _report_with(_result(Verdict.FAIL, 3)))])
    assert comparison.cell("a", Dimension.FINAL_STATE) == "FAIL (step 3)"


def test_hint_two_or_three_steps_are_listed():
    two = compare([("a", _report_with(_result(Verdict.FAIL, 2, 4)))])
    assert two.cell("a", Dimension.FINAL_STATE) == "FAIL (steps 2, 4)"
    three = compare([("a", _report_with(_result(Verdict.WARN, 1, 2, 3)))])
    assert three.cell("a", Dimension.FINAL_STATE) == "WARN (steps 1, 2, 3)"


def test_hint_more_than_three_steps_is_summarized():
    comparison = compare([("a", _report_with(_result(Verdict.FAIL, 1, 2, 3, 4, 5)))])
    assert comparison.cell("a", Dimension.FINAL_STATE) == "FAIL (steps 1, 2, 3, +2 more)"


def test_hint_steps_are_sorted_and_deduplicated():
    comparison = compare([("a", _report_with(_result(Verdict.FAIL, 5, 2, 5)))])
    assert comparison.cell("a", Dimension.FINAL_STATE) == "FAIL (steps 2, 5)"


def test_hint_stepless_evidence_says_no_single_step():
    # Whole-episode or spec-side findings (empty_section, final state never
    # reached): evidence exists, but no step is to blame.
    result = GraderResult(
        dimension=Dimension.FINAL_STATE,
        verdict=Verdict.WARN,
        reason="no rules to check",
        evidence=(Evidence(kind="empty_section", field_name="expected_final_state"),),
    )
    comparison = compare([("a", _report_with(result))])
    assert comparison.cell("a", Dimension.FINAL_STATE) == "WARN (no single step)"


def test_pass_cells_stay_bare_even_with_stepped_evidence():
    # A clean cell stays clean; the recovery story lives in the full report.
    comparison = compare([("a", _report_with(_result(Verdict.PASS, 3, 4)))])
    assert comparison.cell("a", Dimension.FINAL_STATE) == "PASS"


def test_error_cells_are_marked_and_hint_free():
    # ERROR is a judging failure, not a verdict about the trajectory: the
    # bang marker keeps it from reading as FAIL, and its diagnostic
    # evidence never becomes a hint.
    crash = GraderResult(
        dimension=Dimension.BOUNDS,
        verdict=Verdict.ERROR,
        reason="grader crashed",
        evidence=(Evidence(kind="grader_crash", value={"exception": "ZeroDivisionError"}),),
    )
    comparison = compare([("a", _report_with(crash))])
    assert comparison.cell("a", Dimension.BOUNDS) == "!ERROR"


# -- the gates: labels and pairs are the caller's responsibility -----------


def test_an_empty_list_raises():
    with pytest.raises(ValueError, match="at least one report"):
        compare([])


def test_a_duplicate_label_raises():
    report = _report_with()
    with pytest.raises(ValueError, match="duplicate report label 'clean'"):
        compare([("clean", report), ("clean", report)])


def test_an_empty_or_non_string_label_raises():
    report = _report_with()
    with pytest.raises(ValueError, match="label must be a non-empty string"):
        compare([("", report)])
    with pytest.raises(ValueError, match="label must be a non-empty string"):
        compare([(7, report)])


def test_a_non_report_entry_raises():
    with pytest.raises(ValueError, match="must be a Report"):
        compare([("a", "not a report")])


def test_a_malformed_pair_raises():
    with pytest.raises(ValueError, match="\\(label, report\\) pairs"):
        compare([("a",)])


def test_row_order_is_the_callers_order_never_sorted():
    labels = ["zeta", "alpha", "mid"]
    comparison = compare([(label, _report_with()) for label in labels])
    assert [label for label, _ in comparison.entries] == labels


def test_compare_is_reproducible_from_saved_reports(tmp_path):
    # The real Step 8 flow: save each report, load it back, compare. A
    # loaded report renders the same cells as the original.
    report = evaluate(_fixture("unlisted_action"), _spec())
    path = tmp_path / "unlisted_action.json"
    report.to_json_file(path)
    reloaded = Report.from_json_file(path)
    assert compare([("t", reloaded)]).cell("t", Dimension.BOUNDS) == compare(
        [("t", report)]
    ).cell("t", Dimension.BOUNDS)
