"""Step 5 fixtures: they load, they are continuous, and the answer key fits.

Ground truth lives next to the trajectories in
``tests/fixtures/travel_booking/expected_verdicts.json``. These tests keep
the fixture set and the key honest *before* any grader exists: Step 6 reads
the same file and compares real verdicts against it, so a fixture drifting
from the table (or the table from the fixtures) fails here first.
"""

import json
from pathlib import Path

import pytest

from trajecteval import Dimension, Trajectory, Verdict

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "travel_booking"
TABLE = FIXTURES / "expected_verdicts.json"

# The seven task-one fixtures. The answer key must mirror exactly this set --
# a fixture nobody declared a verdict for, or a verdict for a file that does
# not exist, is dead ground truth.
EXPECTED_FIXTURES = {
    "clean",
    "wasteful",
    "critical_recovered",
    "first_class_overlap",
    "failed",
    "unlisted_action",
    "empty",
}


def _table() -> dict:
    return json.loads(TABLE.read_text(encoding="utf-8"))


def _fixture_paths() -> list[Path]:
    return sorted(
        path for path in FIXTURES.glob("*.json") if path.stem in EXPECTED_FIXTURES
    )


def test_fixture_files_match_the_answer_key_exactly():
    assert {path.stem for path in _fixture_paths()} == EXPECTED_FIXTURES
    assert set(_table()["verdicts"]) == EXPECTED_FIXTURES


@pytest.mark.parametrize(
    "path", _fixture_paths(), ids=lambda path: path.stem
)
def test_fixture_loads_and_pairs_with_the_spec(path: Path):
    trajectory = Trajectory.from_json_file(path)
    assert trajectory.task_id == "travel_booking"  # pairs with tasks/travel_booking.json
    assert trajectory.metadata["fixture"] == path.stem
    if path.stem == "empty":
        assert trajectory.steps == ()  # zero steps, deliberately
    else:
        assert trajectory.steps  # every other fixture has actions to judge


@pytest.mark.parametrize(
    "path", _fixture_paths(), ids=lambda path: path.stem
)
def test_fixture_is_step_continuous(path: Path):
    # Ground truth must be free of recorder-style anomalies: each step's
    # state_after is the next step's state_before (trivially true for empty).
    trajectory = Trajectory.from_json_file(path)
    assert trajectory.find_discontinuities() == []


def test_table_covers_every_dimension_with_the_closed_sets():
    dimensions = {dimension.value for dimension in Dimension}
    verdicts = {verdict.value for verdict in Verdict}
    for name, row in _table()["verdicts"].items():
        assert set(row) == dimensions, f"{name}: missing or extra dimension"
        for dimension, verdict in row.items():
            assert verdict in verdicts, f"{name}.{dimension}: {verdict!r} not a Verdict"


def test_quality_column_is_declared_intent_only():
    table = _table()
    assert table["declared_only"] == ["trajectory_quality"]
    note = table["notes"]["trajectory_quality"]
    # The hand-off to Step 9 must stay spelled out: the real judge is
    # nondeterministic, so it may only be held to clean = PASS and
    # wasteful != PASS -- the other quality values are intent, not contract.
    assert "nondeterministic" in note
    assert "Step 9" in note


def test_every_dimension_has_a_fixture_that_fails_only_it():
    """One wrong thing per fixture: for each dimension, at least one fixture
    is non-PASS in that dimension and PASS in every other *enforced*
    dimension (columns in ``declared_only`` are intent, not contract, until
    Step 9 -- so e.g. critical_recovered may carry a WARN quality column and
    still isolate critical).

    This is the isolation property the graders rely on: a grader that fails
    the wrong column, or fails everything, is caught by a fixture whose other
    columns are clean -- a borrowed FAIL would be indistinguishable otherwise.
    """
    table = _table()
    verdicts = table["verdicts"]
    declared = set(table["declared_only"])
    for dimension in Dimension:
        isolating = [
            name
            for name, row in verdicts.items()
            if row[dimension.value] != "PASS"
            and all(
                row[other.value] == "PASS"
                for other in Dimension
                if other is not dimension and other.value not in declared
            )
        ]
        assert isolating, (
            f"no fixture isolates the {dimension.value} dimension "
            f"(non-PASS there, PASS everywhere else that is enforced)"
        )
