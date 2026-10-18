# Tests for compare(): the cross-trace table's data half -- caller-ordered
# rows, every Dimension member a column, the hint rule, and the gates.

import json
from pathlib import Path

import pytest

from trajecteval import (
    Dimension,
    Discontinuity,
    Evidence,
    GraderResult,
    Report,
    Verdict,
    compare,
    evaluate,
    load_spec,
    render_table,
    render_table_markdown,
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


# -- the judge's 0-1 score, rendered beside the verdict ---------------------


def _quality(verdict: Verdict, *steps: int, score: float) -> GraderResult:
    evidence = tuple(Evidence(kind="some_kind", step=step) for step in steps)
    return GraderResult(
        dimension=Dimension.TRAJECTORY_QUALITY,
        verdict=verdict,
        reason="because",
        evidence=(*evidence, Evidence(kind="quality_score", value=score)),
    )


def test_score_renders_beside_the_verdict_word():
    comparison = compare([("a", _report_with(_quality(Verdict.WARN, 2, score=0.72)))])
    assert comparison.cell("a", Dimension.TRAJECTORY_QUALITY) == "WARN 0.72 (step 2)"


def test_pass_cell_shows_the_score_when_one_exists():
    comparison = compare([("a", _report_with(_quality(Verdict.PASS, score=0.91)))])
    assert comparison.cell("a", Dimension.TRAJECTORY_QUALITY) == "PASS 0.91"


def test_fail_cell_shows_the_score_and_the_step_hint_too():
    comparison = compare(
        [("a", _report_with(_quality(Verdict.FAIL, 2, 4, score=0.4)))]
    )
    assert comparison.cell("a", Dimension.TRAJECTORY_QUALITY) == "FAIL 0.40 (steps 2, 4)"
    # The +N-more path keeps the score too (this exact branch once forgot it).
    many = compare([("b", _report_with(_quality(Verdict.FAIL, 1, 2, 3, 4, 5, score=0.4)))])
    assert many.cell("b", Dimension.TRAJECTORY_QUALITY) == "FAIL 0.40 (steps 1, 2, 3, +2 more)"


def test_score_item_never_becomes_a_step_hint():
    # quality_score cites no step; only real step numbers shape the hint.
    comparison = compare([("a", _report_with(_quality(Verdict.WARN, score=0.6)))])
    assert (
        comparison.cell("a", Dimension.TRAJECTORY_QUALITY)
        == "WARN 0.60 (no single step)"
    )


def test_cells_without_a_score_render_exactly_as_before():
    # No quality_score evidence anywhere -> the pre-score strings, byte for
    # byte (this is the empty/ERROR/no-judge lane).
    comparison = compare([("a", _report_with(_result(Verdict.PASS, 3)))])
    assert comparison.cell("a", Dimension.FINAL_STATE) == "PASS"


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


# -- render_table: plain aligned text ---------------------------------------


def test_text_table_format_is_pinned():
    # Two rows, fixed input: the exact bytes a caller sees, spaces included.
    passes = _report_with(
        GraderResult(
            dimension=Dimension.FINAL_STATE, verdict=Verdict.PASS, reason="good"
        )
    )
    boom = Report(
        task_id="t",
        trajectory_task_id="copy",
        results=(
            GraderResult(
                dimension=Dimension.FINAL_STATE,
                verdict=Verdict.FAIL,
                reason="never arrived",
                evidence=(Evidence(kind="final_state_not_reached", step=3),),
            ),
        ),
        discontinuities=(
            Discontinuity(step_index=1, next_step_index=2, differing_keys=("x",)),
            Discontinuity(step_index=2, next_step_index=3, differing_keys=("y",)),
        ),
    )
    text = render_table(compare([("clean", passes), ("boom", boom)]))
    assert text == "\n".join(
        [
            "trace  task      discont  final_state    bounds         critical       trajectory_quality",
            "clean  t         -        PASS           not evaluated  not evaluated  not evaluated",
            "boom   t / copy  2        FAIL (step 3)  not evaluated  not evaluated  not evaluated",
            "legend: !ERROR = could not judge (setup problem); FAIL/WARN = judged -- details in the full report",
        ]
    )


def test_text_table_task_column_shows_both_ids_when_they_differ():
    # The spec's task id normally; a copied trace graded against another
    # spec shows both ids, because the disagreement is the row's story.
    same = Report(task_id="t", trajectory_task_id="t")
    differ = Report(task_id="spec_a", trajectory_task_id="copied_trace")
    text = render_table(compare([("a", same), ("b", differ)]))
    header, row_a, row_b = text.splitlines()[:3]
    col = header.index("task")
    assert row_a[col : col + 1] == "t"
    assert row_b[col:].startswith("spec_a / copied_trace")


def test_text_table_discont_column_counts_and_dashes():
    clean = Report(task_id="t", trajectory_task_id="t")
    gappy = Report(
        task_id="t",
        trajectory_task_id="t",
        discontinuities=(
            Discontinuity(step_index=1, next_step_index=2, differing_keys=("a",)),
            Discontinuity(step_index=2, next_step_index=3, differing_keys=("b",)),
        ),
    )
    header, row_a, row_b = render_table(compare([("a", clean), ("b", gappy)])).splitlines()[:3]
    col = header.index("discont")
    assert row_a[col] == "-"  # no recorder gaps: a dash, not a zero or blank
    assert row_b[col : col + 1] == "2"


def test_text_table_legend_is_last_even_without_errors():
    text = render_table(compare([("a", _report_with(_result(Verdict.PASS, 1)))]))
    assert text.splitlines()[-1] == (
        "legend: !ERROR = could not judge (setup problem); "
        "FAIL/WARN = judged -- details in the full report"
    )


def test_text_table_error_cells_read_differently_from_fail_cells():
    # The same table, both states: a FAIL row carries the verdict word and
    # no bang; an ERROR row carries the bang and no verdict word.
    crash = GraderResult(
        dimension=Dimension.BOUNDS,
        verdict=Verdict.ERROR,
        reason="the bounds grader crashed: ZeroDivisionError",
        evidence=(Evidence(kind="grader_crash"),),
    )
    fail_row_report = _report_with(_result(Verdict.FAIL, 3))
    crash_row_report = Report(
        task_id="t",
        trajectory_task_id="t",
        results=(_result(Verdict.PASS, 1), crash),
    )
    lines = render_table(
        compare([("fail", fail_row_report), ("crash", crash_row_report)])
    ).splitlines()
    assert "FAIL (step 3)" in lines[1] and "!ERROR" not in lines[1]
    assert "!ERROR" in lines[2] and "FAIL" not in lines[2]


def test_text_table_long_labels_keep_alignment():
    long_label = "a-very-long-trace-label-from-a-file-stem"
    text = render_table(
        compare([("s", _report_with(_result(Verdict.PASS, 1))), (long_label, _report_with(_result(Verdict.PASS, 1)))])
    )
    lines = text.splitlines()
    # The last column is never padded, so no line ends in spaces...
    assert all(not line.endswith(" ") for line in lines)
    # ...and the final_state cell sits at the same column in every row.
    col = lines[0].index("final_state")
    for row in lines[1:-1]:
        assert row[col : col + 4] == "PASS"


def test_text_table_all_seven_fixtures_render_one_row_each():
    names = sorted(TABLE["verdicts"])
    comparison = compare([(name, evaluate(_fixture(name), _spec())) for name in names])
    lines = render_table(comparison).splitlines()
    assert len(lines) == len(names) + 2  # header + seven rows + legend
    for name, row in zip(names, lines[1:-1]):
        assert row.startswith(name)
        assert "travel_booking" in row
        assert "not evaluated" in row  # trajectory_quality, every row
    for column in ("trace", "task", "discont", "final_state", "bounds", "critical", "trajectory_quality"):
        assert column in lines[0]


# -- render_table_markdown: the same cells in Markdown pipes ----------------


def test_markdown_table_format_is_pinned():
    passes = _report_with(
        GraderResult(
            dimension=Dimension.FINAL_STATE, verdict=Verdict.PASS, reason="good"
        )
    )
    boom = Report(
        task_id="t",
        trajectory_task_id="copy",
        results=(
            GraderResult(
                dimension=Dimension.FINAL_STATE,
                verdict=Verdict.FAIL,
                reason="never arrived",
                evidence=(Evidence(kind="final_state_not_reached", step=3),),
            ),
        ),
        discontinuities=(
            Discontinuity(step_index=1, next_step_index=2, differing_keys=("x",)),
        ),
    )
    text = render_table_markdown(compare([("clean", passes), ("boom", boom)]))
    assert text == "\n".join(
        [
            "| trace | task | discont | final_state | bounds | critical | trajectory_quality |",
            "|---|---|---|---|---|---|---|",
            "| clean | t | - | PASS | not evaluated | not evaluated | not evaluated |",
            "| boom | t / copy | 1 | FAIL (step 3) | not evaluated | not evaluated | not evaluated |",
            "*legend: `!ERROR` = could not judge (setup problem); FAIL/WARN = judged -- details in the full report*",
        ]
    )


def test_markdown_and_text_tables_share_the_same_cells():
    # One cell rule (_row_cells), two renderers: the markdown table can
    # never drift from the text table's content.
    crash = GraderResult(
        dimension=Dimension.BOUNDS,
        verdict=Verdict.ERROR,
        reason="the bounds grader crashed: ZeroDivisionError",
        evidence=(Evidence(kind="grader_crash"),),
    )
    report = Report(
        task_id="spec_a",
        trajectory_task_id="copied",
        results=(_result(Verdict.FAIL, 3), crash),
        discontinuities=(
            Discontinuity(step_index=1, next_step_index=2, differing_keys=("x",)),
        ),
    )
    comparison = compare([("r", report)])
    text, md = render_table(comparison), render_table_markdown(comparison)
    for cell in ("!ERROR", "FAIL (step 3)", "spec_a / copied", "1", "not evaluated"):
        assert cell in text and cell in md


def test_markdown_table_keeps_every_dimension_column():
    comparison = compare([("a", _report_with(_result(Verdict.PASS, 1)))])
    header = render_table_markdown(comparison).splitlines()[0]
    assert header == (
        "| trace | task | discont | final_state | bounds | critical | trajectory_quality |"
    )
