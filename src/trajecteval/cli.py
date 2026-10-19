# traject: the command-line entry point -- thin glue over evaluate(), compare(),
# run_agent(), and the existing renderers. No grading or agent logic lives
# here.
#
# Exit codes mean one thing only: 0 = the command ran, nonzero = it could not
# run. There is deliberately no pass/fail exit code (this project has no
# single score): a FAIL verdict is a *result*, printed in the report, not a
# command failure. The one exception is argparse's exit 2 for usage errors.
#
# Errors obey two lanes, same as the library:
# - broken input (a TrajEvalError: unreadable file, invalid JSON, unknown
#   task, invalid task id) -> ONE clear 'error: ...' line on stderr, exit 1,
#   no traceback;
# - a ValueError is caught ONLY where it means user-facing setup trouble
#   (judge or agent client configuration at construction). Anywhere else a
#   ValueError keeps its usual meaning: a bug, traceback and all.
# - `run` also aborts on a provider failure (LLMClientError) with one line
#   and exit 1: a crashed API call is not agent behavior, so nothing is
#   saved for that run.

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from trajecteval.agent import DEFAULT_MAX_STEPS, run_agent
from trajecteval.compare import compare, render_table
from trajecteval.envs import agent_instruction, make_environment
from trajecteval.errors import TrajEvalError
from trajecteval.judge.base import LLMClientError
from trajecteval.judge.llm import LLMJudge, OpenAIChatClient
from trajecteval.models import Trajectory
from trajecteval.report import Report, evaluate, render_text
from trajecteval.task_spec import TaskSpec, load_spec


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="traject",
        description=(
            "Grade agent trajectories against their own task specs, or run "
            "an LLM agent in a simulated world. One trace prints the "
            "detailed report; several print a comparison table. A model is "
            "only ever called with --judge (evaluate) or by run."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)
    evaluate_parser = sub.add_parser(
        "evaluate", help="grade one or more trajectory JSON files"
    )
    evaluate_parser.add_argument(
        "traces",
        nargs="+",
        metavar="TRACE",
        help="path to a trajectory JSON file",
    )
    evaluate_parser.add_argument(
        "--judge",
        action="store_true",
        help="also judge trajectory quality with an LLM (calls a paid model)",
    )
    run_parser = sub.add_parser(
        "run", help="run an LLM agent in a simulated world (calls a paid model)"
    )
    run_parser.add_argument(
        "task",
        metavar="TASK",
        help="task id to run, e.g. travel_booking",
    )
    run_parser.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="MODEL",
        help="model id for the agent (repeat: one sequential run per model)",
    )
    run_parser.add_argument(
        "--out",
        required=True,
        metavar="DIR",
        help="output folder for one trajectory JSON per model (created if missing)",
    )
    run_parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        metavar="N",
        help=f"step limit per run (default {DEFAULT_MAX_STEPS})",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    # argv=None means "use sys.argv[1:]" -- the console script's normal mode.
    # Tests call main(["evaluate", "a.json", ...]) directly and read the int.
    args = _build_parser().parse_args(argv)
    if args.command == "run":
        return _run(args)
    return _evaluate(args)


def _evaluate(args: argparse.Namespace) -> int:
    # The evaluate lane: grade what already exists. Nothing here can start a
    # run or write files.

    # Phase 1: everything that cannot spend money. Construct the judge object
    # first (it only reads configuration -- the key comes from the
    # environment; no network yet), then load EVERY trace and resolve EVERY
    # spec. Any broken input stops the command here, before the first judge
    # call -- a paid model is never asked to judge a run we cannot grade.
    judge = None
    if args.judge:
        try:
            judge = LLMJudge()
        except ValueError as exc:
            # Missing key or bad key-adjacent configuration (e.g. a
            # non-numeric TRAJECTEVAL_JUDGE_TIMEOUT). One line, no key, ever.
            print(f"error: cannot set up the judge: {exc}", file=sys.stderr)
            return 1

    # Each trace resolves its own task spec from its own task_id; a spec is
    # loaded by id from tasks/ in the working directory (load_spec's default).
    loaded: list[tuple[str, Trajectory, TaskSpec]] = []
    for path in args.traces:
        try:
            trajectory = Trajectory.from_json_file(path)
            spec = load_spec(trajectory.task_id)
        except TrajEvalError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        loaded.append((Path(path).stem, trajectory, spec))

    # Rows in the comparison table are labeled by file stem, so two inputs
    # sharing a file name (run1/clean.json, run2/clean.json) would be
    # indistinguishable. Checked HERE as broken input, one clear line --
    # compare()'s own ValueError stays the "caller bug" lane, untouched.
    labels = [label for label, _, _ in loaded]
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        print(
            f"error: duplicate trace file name(s): {', '.join(duplicates)}; "
            "rows must be tellable apart -- rename the files",
            file=sys.stderr,
        )
        return 1

    # Phase 2: grading. With --judge this is where the paid model is called,
    # once per trace, sequentially. A judge failure here is NOT a command
    # failure: it becomes an ERROR cell, which the report/table explains.
    reports: list[tuple[str, Report]] = [
        (label, evaluate(trajectory, spec, judge=judge))
        for label, trajectory, spec in loaded
    ]

    if len(reports) == 1:
        print(render_text(reports[0][1]))
    else:
        print(render_table(compare(reports)))
    return 0


def _run(args: argparse.Namespace) -> int:
    # The run lane: resolve the world + instruction, then one sequential run
    # per --model, one trajectory file per run. Every input/config problem is
    # checked BEFORE any model call; a provider failure mid-run is one line
    # and exit 1, saving nothing for that run.
    try:
        instruction = agent_instruction(args.task)
    except TrajEvalError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    duplicates = sorted({m for m in args.model if args.model.count(m) > 1})
    if duplicates:
        print(
            f"error: duplicate --model value(s): {', '.join(duplicates)}; "
            "each run needs its own model",
            file=sys.stderr,
        )
        return 1

    # Construct every client up front: a missing key or bad configuration
    # fails here, before the first run starts (one clear line, no key ever).
    clients: list[OpenAIChatClient] = []
    for model in args.model:
        try:
            clients.append(OpenAIChatClient(model=model))
        except ValueError as exc:
            print(f"error: cannot set up the agent client: {exc}", file=sys.stderr)
            return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for client in clients:
        try:
            trajectory = run_agent(
                instruction,
                make_environment(args.task),  # a fresh world per model
                client,
                model=client.model,
                max_steps=args.max_steps,
            )
        except LLMClientError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        # "/" is not a file-name character; the "__" already separates task
        # from model, so a lone "_" inside the model id stays readable. The
        # model id itself appears untouched in the trajectory's metadata.
        path = out_dir / f"{args.task}__{client.model.replace('/', '_')}.json"
        trajectory.to_json_file(path)
        print(
            f"wrote {path} ({len(trajectory.steps)} steps, "
            f"stop reason: {trajectory.metadata['stop_reason']})"
        )
    return 0
