# TrajEval

**AI agents don't just finish tasks — they take paths. TrajEval grades the path.**

An AI agent completes a task by taking actions one after another until it's
done. Most evaluations look only at the ending: did the final answer pass?
TrajEval records the whole episode — every action in order, with the state of
the world before and after each one — and grades it on four separate
questions, so a report tells you *why* a run was good or bad.

## A run that looks fine — and isn't

Here is a real recorded episode from this repository's test fixtures
(`tests/fixtures/travel_booking/critical_recovered.json`). An agent has to
book a flight and pay by card; the task's rules — a small JSON file — make
paying with points a forbidden move:

1. `search_flights` — fine
2. `book_flight` (economy) — fine
3. `pay` with **points** — forbidden move
4. `refund_payment` — undoes the mistake
5. `pay` with **card** — the required ending

The final state is exactly right: confirmed, paid by card. A checker that only
looks at the ending passes this run. TrajEval does not — the *critical
mistakes* column says FAIL at step 3, and it stays FAIL: recovery doesn't
erase what happened. The *final state* column says PASS. Same run, two
columns, and the disagreement is the most useful thing on the page.

One number can't tell you this story. Four columns can.

## Seven real runs, one table

This is real output — the repository's seven hand-made fixture trajectories,
graded by the shipped code (`evaluate()` → `compare()` → `render_table()`),
pasted verbatim. Reproduce it yourself with:

```bash
uv run python -c "
from pathlib import Path
from trajecteval import Trajectory, compare, evaluate, load_spec, render_table

F = Path('tests/fixtures/travel_booking')
spec = load_spec('travel_booking')
names = sorted(n.stem for n in F.glob('*.json') if n.stem != 'expected_verdicts')
pairs = [(n, evaluate(Trajectory.from_json_file(F / f'{n}.json'), spec)) for n in names]
print(render_table(compare(pairs)))"
```

```text
trace                task            discont  final_state            bounds                 critical               trajectory_quality
clean                travel_booking  -        PASS                   PASS                   PASS                   not evaluated
critical_recovered   travel_booking  -        PASS                   PASS                   FAIL (step 3)          not evaluated
empty                travel_booking  -        FAIL (no single step)  WARN (no single step)  WARN (no single step)  not evaluated
failed               travel_booking  -        FAIL (step 3)          PASS                   PASS                   not evaluated
first_class_overlap  travel_booking  -        PASS                   FAIL (step 2)          FAIL (step 2)          not evaluated
unlisted_action      travel_booking  -        PASS                   FAIL (step 3)          PASS                   not evaluated
wasteful             travel_booking  -        PASS                   PASS                   PASS                   not evaluated
legend: !ERROR = could not judge (setup problem); FAIL/WARN = judged -- details in the full report
```

Reading it:

- `critical_recovered` is the story above: PASS and FAIL side by side, on one run.
- `empty` warns in every column it could check — nothing was verified, so
  nothing gets a silent PASS.
- `first_class_overlap` shows one bad step breaking two columns at once.
- `trajectory_quality` says **not evaluated**: the LLM judge's contract is
  committed but its real client is Step 9. The table marks what it doesn't
  know — it never guesses and never drops a column.

## The four questions, in plain words

A **trajectory** is one recorded episode. Each is graded against a **task
spec** — a small JSON file holding that task's rules — on four separate
questions:

| # | Dimension | Type | Question it answers |
|---|-----------|------|---------------------|
| 1 | **Final state** | Deterministic code | Did the episode end in the correct state? |
| 2 | **Bounds** | Deterministic code | Did every action stay inside the allowed scope — allowed names, allowed argument values? |
| 3 | **Critical mistakes** | Deterministic code | Did the agent ever commit a forbidden action — *even if it later recovered*? |
| 4 | **Trajectory quality** | LLM judge | Was the path efficient and coherent, or full of detours, thrash, and redundant actions? |

Dimensions 1–3 are code that always gives the same answer. Dimension 4 is
fuzzy by nature — "was this step necessary?" resists hard rules — so an LLM
judges it; the two never mix inside one module. Mistakes are sticky: the
critical column scans the whole history, so a forbidden move at step 3 is
still FAIL at the end.

Every cell holds a verdict — **PASS**, **WARN**, **FAIL**, or **ERROR** —
with a one-sentence reason and evidence citing step numbers. ERROR means the
grader could not judge (a setup problem), never that the trajectory broke a
rule.

## What works today

**Steps 1–8 are done. Step 9 has started**: the judge's contract
(`src/trajecteval/judge/base.py`) is committed; the real client, the
`evaluate()` wiring, the CLI, and everything after are planned — the
[roadmap](#roadmap) says exactly what's next.

Works today, as a library:

- load a trajectory or a task spec from JSON, validated structurally;
- grade the three deterministic dimensions with `evaluate()` — one broken
  grader degrades its own column to ERROR, never the whole run;
- render a single report as readable text or machine JSON, and save/load
  reports to files;
- line many reports up in one comparison table, as plain text or Markdown.

Planned: the real LLM judge (Step 9), the `uv run traject` CLI (Step 10), a
recorder with a simulated environment (Step 11), an agent demo on real traces
(Step 12), a second toy task (Step 13).

## Install

Requires [uv](https://docs.astral.sh/uv/) and **Python 3.14+**. Everything
runs through `uv` — no pip, no virtualenv to activate, no Node.

```bash
git clone <repo-url> TrajEval
cd TrajEval
uv sync            # installs dependencies + the package itself (editable)
uv run pytest      # run the test suite
```

## Use

Everything in this section runs today.

**Evaluate one trajectory**

```python
from trajecteval import Trajectory, evaluate, load_spec, render_text

traj = Trajectory.from_json_file("trace.json")
report = evaluate(traj, load_spec("travel_booking"))
for result in report.results:
    print(result.dimension, result.verdict, result.reason)
print(render_text(report))     # or render_json(report) for the machine form
```

**Compare traces**

```python
from pathlib import Path

from trajecteval import (
    Report, Trajectory, compare, evaluate, load_spec, render_table,
)

spec = load_spec("travel_booking")
pairs = [
    (path.stem, evaluate(Trajectory.from_json_file(path), spec))
    for path in sorted(Path("reports").glob("*.json"))
]
print(render_table(compare(pairs)))          # plain aligned text
# print(render_table_markdown(comparison))   # the same cells as Markdown

report = evaluate(Trajectory.from_json_file("trace.json"), spec)
report.to_json_file("reports/clean.json")            # save
loaded = Report.from_json_file("reports/clean.json")  # load back, equal
```

Each row is one trace under a caller-supplied label; every dimension is a
column (a report missing one shows `not evaluated` — columns are never
dropped), `discont` counts recorder discontinuities, and `!ERROR` marks a
column that could not be judged.

**The CLI** *(planned — Step 10; does not exist yet)*

```bash
uv run traject eval trace.json
uv run traject compare reports/*.json
```

## Task specs

Graders know **how to check**, never **what to check**. All task-specific
rules live in one plain JSON file per task. This is
`tasks/travel_booking.json`, trimmed:

```json
{
  "task_id": "travel_booking",
  "expected_final_state": { "status": "confirmed", "passengers": 1 },
  "allowed_actions": {
    "search_flights": {},
    "book_flight": { "cabin": ["economy"] },
    "pay": { "method": ["card", "points"] },
    "refund_payment": {}
  },
  "critical_error_patterns": [
    { "action": "book_flight", "args": { "cabin": "first" } },
    { "action": "pay", "args": { "method": "points" } }
  ]
}
```

- **`expected_final_state`** — the listed fields must equal the listed values
  in the final state. Unlisted fields are not judged.
- **`allowed_actions`** — action name → per-argument lists of permitted values.
  `{}` means the action is allowed with any arguments; an unlisted action is
  out of bounds.
- **`critical_error_patterns`** — exact moves that count as critical mistakes:
  the action name plus, if given, arguments that must all match exactly.

Adding a new task means writing one JSON file — no new grader code.

## Design principles

1. **Per-dimension, not scalar.** No mega-score as the primary output; any
   aggregate is clearly secondary to the breakdown.
2. **Deterministic graders and the LLM judge never mix modules.** Same result
   type, separate implementations, judge injectable.
3. **Mistakes are sticky.** Recovery never downgrades a critical mistake.
4. **Evidence or it didn't happen.** Every verdict carries a reason; FAIL and
   WARN additionally cite typed evidence with step numbers. Evidence is
   neutral — the verdict, not the evidence, says whether it's good or bad.
5. **ERROR ≠ FAIL.** A grader that can't judge says so; it never guesses.
6. **Rules live in task specs, not grader code.** Graders stay domain-agnostic;
   every rule in a spec list is applied.
7. **Small and focused.** Single-environment, one task type at a time. No plugin
   registries, no over-engineering until a real need appears.
8. **Tested without the network.** Offline tests, fake judge, fixture
   trajectories as ground truth.
9. **Python/uv only.** Everything runs as `uv run ...`. No pip, no venv
   activation, no Node/npm/TypeScript.

## Development

Every command goes through `uv`:

```bash
uv run pytest                          # full suite (offline, always)
uv run pytest tests/test_models.py -v  # one file
uv run pytest -k discontinu            # by name
```

Conventions:

- One module = one responsibility; one test file per module.
- **No test may require network access or an API key** — the judge is faked.
- Hand-made fixtures are the ground truth; a recorder can generate traces, but
  tests never depend on one until Step 12, and never on an LLM until (optional)
  Step 13.

## The trajectory format

One episode per JSON file — plain data, no database:

```json
{
  "task_id": "<task-id>",
  "metadata": {"agent": "example", "attempt": 1},
  "steps": [
    {
      "index": 1,
      "state_before": {"balance": 100, "cart": []},
      "action": {"name": "add_to_cart", "args": {"item": "A1", "qty": 1}},
      "state_after": {"balance": 100, "cart": ["A1"]}
    },
    {
      "index": 2,
      "state_before": {"balance": 100, "cart": ["A1"]},
      "action": {"name": "checkout", "args": {"coupon": "NONE"}},
      "state_after": {"balance": 90, "cart": []}
    }
  ]
}
```

Loading rules (implemented in Step 2):

- Step indices must be sequential — gaps or duplicates are rejected with an
  error that names the step.
- Only structural problems **raise** at load time (missing `task_id`, missing
  step keys, missing action name, bad index sequence). Every such message names
  the step involved.
- A state discontinuity between one step's `state_after` and the next step's
  `state_before` does **not** prevent loading — it is *reported* by
  `find_discontinuities()` so it can appear in the report.
- Unknown top-level keys are preserved in `metadata`. On a collision, the
  explicit `metadata` entry wins.

## Project structure

```text
TrajEval/
├── pyproject.toml               # project metadata + dependencies (uv owns this)
├── uv.lock                      # pinned versions (reproducible installs)
├── README.md
├── tasks/                       # one JSON spec per task — the only place rules live
│   └── travel_booking.json      # task one spec (fixtures live in tests/fixtures/travel_booking/)
├── src/
│   └── trajecteval/
│       ├── __init__.py          # public API surface
│       ├── errors.py            # TrajEvalError base + TrajectoryError / SpecError / ReportError
│       ├── models.py            # Step, Action, Trajectory, Discontinuity + JSON I/O
│       ├── matching.py          # json_equal + the subset matcher -- shared, no imports from the rest
│       ├── results.py           # Verdict, Dimension, Evidence, GraderResult
│       ├── task_spec.py         # TaskSpec + load-by-id (no registry)
│       ├── graders/             # deterministic graders — one file per dimension
│       │   ├── base.py          # Grader protocol: grade(spec, trajectory)
│       │   ├── final_state.py
│       │   ├── bounds.py
│       │   └── critical.py
│       ├── report.py            # evaluate() pipeline + JSON/text/file renderers
│       ├── compare.py           # cross-trace comparison table (text + markdown)
│       ├── judge/
│       │   ├── base.py          # judge protocol + structured output types
│       │   ├── fake.py          # offline fake — every test uses this
│       │   └── llm.py           # real client (later)
│       ├── cli.py               # uv run traject ...
│       ├── environment.py       # simulated environment (later)
│       ├── recorder.py          # thin recorder wrapping the environment (later)
│       └── adapters/
│           └── debuggernaut.py  # optional, only if real logs exist (later)
├── tests/
│   ├── fixtures/
│   │   ├── minimal.json         # loader fixture (Step 2)
│   │   ├── discontinuous.json   # discontinuity fixture (Step 2)
│   │   ├── travel_booking/      # task one: seven trajectories
│   │   │   ├── clean.json
│   │   │   ├── wasteful.json
│   │   │   ├── critical_recovered.json
│   │   │   ├── first_class_overlap.json
│   │   │   ├── failed.json
│   │   │   ├── unlisted_action.json
│   │   │   ├── empty.json
│   │   │   └── expected_verdicts.json   # verdict per fixture per dimension
│   │   └── <task-two>/          # smaller, differently shaped (later)
│   ├── test_fixtures.py         # fixtures load, are continuous, match the table
│   ├── test_smoke.py
│   └── ...                      # one test file per module
└── examples/                    # LLM-agent demo (later)
```

**This is the target layout, not the current tree.** Today only
`pyproject.toml`, `README.md`, `tasks/travel_booking.json`,
`src/trajecteval/{__init__,errors,models,results,task_spec}.py`,
`src/trajecteval/graders/`, `tests/` and
`tests/fixtures/travel_booking/` exist — the roadmap's status column
says what's built.

**Why `src/` layout?** It prevents a classic trap: Python silently importing
your working directory instead of the installed package, so tests pass on your
machine and break elsewhere. With `src/`, you only ever test what is properly
installed — uv does that installation automatically.

## Toy tasks

Two toy tasks carry the tests: **task one is travel booking, task two is
support-ticket triage** (both confirmed). Neither is file-based. Task one gets
seven fixtures with an expected-verdict table beside them: clean, wasteful,
critical_recovered, first_class_overlap, failed, unlisted_action, empty.
Task two is smaller and shaped differently, and exists to prove the graders
work unchanged on a new domain.

## Roadmap

| # | Step | Status |
|---|------|--------|
| 1 | Project scaffold (`uv init`, src layout, first test) | ✅ done |
| 2 | Models: `Step`, `Trajectory`, JSON loading | ✅ done |
| 3 | Result types: `Verdict`, `Dimension`, `Evidence`, `GraderResult` | ✅ done |
| 4 | Task spec + grader contract: data-file rules, load by task id, `Grader` protocol | ✅ done |
| 5 | Fixtures for task one — seven trajectories (clean, wasteful, critical_recovered, first_class_overlap, failed, unlisted_action, empty) + the machine-readable expected-verdict table | ✅ done |
| 6 | Three deterministic graders (final state, bounds, critical) — reproduce the fixture answer key exactly | ✅ done |
| 7 | Report: `evaluate()` runs the three deterministic dimensions first (a crashed grader degrades its own column to ERROR, never the run), then renders JSON + readable text; the LLM judge is an optional addition once Step 9 exists | ✅ done |
| 8 | Comparison table across traces — a dimension with no result gets an explicit marker, the same honesty rule as `render_text`'s `not evaluated` lines, never a silently dropped column | ✅ done |
| 9 | LLM judge: contract + fake, fully offline | ⏳ planned |
| 10 | CLI (`uv run traject ...`) | ⏳ planned |
| 11 | Toy task two — same graders, new domain, unchanged code | ⏳ planned |
| 12 | Recorder + simulated environment (generate traces by running actions) | ⏳ planned |
| 13 | LLM-agent demo (agent → recorder → report; needs an API key) | ⏳ planned |
| 14 | Debuggernaut adapter — **optional**, only if real logs exist | ⏳ planned (optional) |

## Design notes & FAQ

**Why not one overall score?**
Because a single number destroys information. Two traces can average to 0.8 for
completely different reasons — one had a critical mistake it recovered from, the
other was just slow. The report must let a human see that difference quickly. If
you want a scalar, average the columns yourself and accept what you lose.

**Why is only one dimension LLM-judged?**
Efficiency and coherence are genuinely fuzzy; "was this necessary?" resists hard
rules. Everything else is a fact about recorded states and actions — code checks
facts better than a language model does. Using an LLM where rules suffice adds
cost, nondeterminism, and hallucination risk for no benefit.

**Why do rules live in task spec files?**
Because otherwise every new task means editing grader code, and grader code is
exactly what should stay boring and domain-agnostic. A spec file is reviewable,
diffable, and testable as data. It also forces rules to be explicit: if
something is graded, it's written down in the spec — not buried in a branch.

**Why two toy tasks?**
One task proves a grader works; two tasks prove the grader isn't secretly about
task one. Task two runs through the exact same graders with only the spec and
fixtures changed.

**Why is the LLM judge injectable?**
So tests never touch the network. `evaluate(..., judge=FakeJudge())` runs
offline and deterministically; the real client is a one-line swap at the call
site, and API keys stay out of core logic.

**Why dataclasses instead of Pydantic?**
The project parses its own well-defined JSON format rather than validating
arbitrary external input at an API boundary. Stdlib dataclasses keep the core
dependency-free and the mental model small. Revisiting later if needs grow is
normal.

**Where does Debuggernaut fit?**
Nowhere required. It's an optional final adapter (Step 14), included only if
real attempt logs exist. Everything through Step 13 runs on simulated tasks.

**What does a report look like?**
Two artifacts per trace: a `.json` file with verdicts, reasons, and evidence,
plus a human-readable rendering of the same data. The comparison table aligns
many reports side by side.

## License

TBD.
