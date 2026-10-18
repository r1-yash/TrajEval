"""Step 10 CLI tests: ``traject evaluate`` as thin glue over the library.

Offline only, same rule as the rest of the default suite: every test calls
``main(argv)`` directly and reads its integer return as the exit code. The
real judge is never constructed -- the ``--judge`` tests swap in the existing
``ScriptedJudge`` fake by monkeypatching ``trajecteval.cli.LLMJudge``, and
one test proves the swap is total: without ``--judge`` the judge class is
never even instantiated, so no network call can happen.

Exit-code contract under test: 0 = the command ran (there is no overall
pass/fail -- a FAIL verdict is a result, not a failure); 1 = it could not
run (one clear line, no traceback); argparse's 2 = usage error.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

import trajecteval.cli as cli
from trajecteval.cli import main
from trajecteval.judge.base import JudgeResult, LLMClientError
from trajecteval.judge.fake import ScriptedJudge
from trajecteval.judge.llm import KEY_ENV
from trajecteval.results import Verdict

REPO = Path(__file__).resolve().parent.parent
FIXTURES = REPO / "tests" / "fixtures" / "travel_booking"


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    # A miniature project directory: tasks/ next to the trace files, and the
    # cwd moved there -- exactly the layout load_spec() expects when a user
    # runs `traject evaluate` from their own project root.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "tasks").mkdir()
    shutil.copy(REPO / "tasks" / "travel_booking.json", tmp_path / "tasks")
    for name in ("clean", "wasteful"):
        shutil.copy(FIXTURES / f"{name}.json", tmp_path / f"{name}.json")
    return tmp_path


# -- happy paths: one trace -> detailed report; several -> comparison table --


def test_one_trace_prints_the_detailed_report(workspace: Path, capsys) -> None:
    code = main(["evaluate", "clean.json"])
    out = capsys.readouterr().out
    assert code == 0
    assert "task travel_booking (trajectory travel_booking)" in out
    # Without --judge the quality column is honestly marked, never guessed.
    assert "not evaluated: trajectory_quality" in out


def test_several_traces_print_the_comparison_table(
    workspace: Path, capsys
) -> None:
    code = main(["evaluate", "clean.json", "wasteful.json"])
    out = capsys.readouterr().out
    assert code == 0
    # Row labels are file stems; the legend is the last line.
    assert "clean" in out and "wasteful" in out
    assert out.rstrip().startswith("trace")
    assert "legend:" in out


# -- --judge: the fake stands in for the real judge, no network, no spend --


def test_judge_flag_calls_the_judge_once_per_trace(
    workspace: Path, monkeypatch, capsys
) -> None:
    fake = ScriptedJudge(
        [JudgeResult(verdict=Verdict.PASS, reason="scripted pass", score=0.95)]
    )
    monkeypatch.setattr(cli, "LLMJudge", lambda: fake)
    code = main(["evaluate", "clean.json", "--judge"])
    out = capsys.readouterr().out
    assert code == 0
    assert len(fake.calls) == 1
    # The scripted verdict and score ride the quality line, as designed.
    assert "PASS 0.95" in out


def test_without_the_flag_the_judge_is_never_constructed(
    workspace: Path, monkeypatch, capsys
) -> None:
    # The money-safety contract: the ONLY path to a network call runs through
    # constructing LLMJudge, and --judge is the only thing that constructs it.
    def tripwire():
        raise AssertionError("the judge must not be constructed without --judge")

    monkeypatch.setattr(cli, "LLMJudge", tripwire)
    code = main(["evaluate", "clean.json", "wasteful.json"])
    out = capsys.readouterr().out
    assert code == 0
    # In the table, the quality column renders the bare not-evaluated marker.
    assert "trajectory_quality" in out and "not evaluated" in out


def test_a_judge_failure_is_an_error_cell_not_a_command_failure(
    workspace: Path, monkeypatch, capsys
) -> None:
    # A provider failure (rate limit, bad key at the API, timeout) reaches
    # evaluate() as LLMClientError and becomes an ERROR column; the command
    # itself still ran, so it still exits 0.
    fake = ScriptedJudge(
        raises=LLMClientError("judge request failed: HTTP 429", status=429)
    )
    monkeypatch.setattr(cli, "LLMJudge", lambda: fake)
    code = main(["evaluate", "clean.json", "--judge"])
    out = capsys.readouterr().out
    assert code == 0
    assert "trajectory_quality ERROR" in out


# -- broken input: one clear line on stderr, exit 1, no traceback, no output --


def test_unknown_task_is_one_clear_line(workspace: Path, capsys) -> None:
    (workspace / "ghost.json").write_text(
        '{"task_id": "ghost", "steps": []}', encoding="utf-8"
    )
    code = main(["evaluate", "ghost.json"])
    captured = capsys.readouterr()
    assert code == 1
    assert "ghost" in captured.err
    assert captured.out == ""
    assert "Traceback" not in captured.err


def test_invalid_task_id_is_one_clear_line(workspace: Path, capsys) -> None:
    (workspace / "badid.json").write_text(
        '{"task_id": "Travel/One", "steps": []}', encoding="utf-8"
    )
    code = main(["evaluate", "badid.json"])
    captured = capsys.readouterr()
    assert code == 1
    assert "invalid task id" in captured.err
    assert captured.out == ""
    assert "Traceback" not in captured.err


@pytest.mark.parametrize("kind", ["missing", "not-json"])
def test_broken_trace_file_is_one_clear_line(
    workspace: Path, capsys, kind: str
) -> None:
    if kind == "missing":
        target = workspace / "nope.json"  # never created
    else:
        target = workspace / "broken.json"
        target.write_text("not json", encoding="utf-8")
    code = main(["evaluate", str(target)])
    captured = capsys.readouterr()
    assert code == 1
    assert target.name in captured.err
    assert captured.out == ""
    assert "Traceback" not in captured.err


def test_duplicate_stems_are_rejected_before_any_grading(
    workspace: Path, capsys
) -> None:
    # Two files with the same stem would be indistinguishable table rows;
    # the CLI rejects them as broken input, before calling compare().
    (workspace / "run1").mkdir()
    (workspace / "run2").mkdir()
    shutil.copy(workspace / "clean.json", workspace / "run1" / "x.json")
    shutil.copy(workspace / "wasteful.json", workspace / "run2" / "x.json")
    code = main(["evaluate", "run1/x.json", "run2/x.json"])
    captured = capsys.readouterr()
    assert code == 1
    assert "duplicate" in captured.err
    assert captured.out == ""
    assert "Traceback" not in captured.err


def test_missing_judge_key_names_the_variable_and_leaks_nothing(
    workspace: Path, monkeypatch, capsys
) -> None:
    # Real LLMJudge, real (offline) failure: with the key unset, constructing
    # the judge raises before any grading. The message must name the
    # environment variable -- and must never contain a key-like value; a
    # decoy env var proves nothing else leaks either.
    monkeypatch.delenv(KEY_ENV, raising=False)
    monkeypatch.setenv(f"{KEY_ENV}_OLD", "sk-SENTINEL-do-not-leak")
    code = main(["evaluate", "clean.json", "--judge"])
    captured = capsys.readouterr()
    assert code == 1
    assert KEY_ENV in captured.err
    assert "SENTINEL" not in captured.err + captured.out
    assert "Traceback" not in captured.err


# -- usage errors: argparse's own exit 2 --


def test_missing_trace_argument_is_a_usage_error(
    workspace: Path, capsys
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["evaluate"])
    assert excinfo.value.code == 2
    assert "usage:" in capsys.readouterr().err
