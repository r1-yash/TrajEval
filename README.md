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
- `trajectory_quality` says **not evaluated**: this command passes no judge,
  and `evaluate()` only ever calls a model when you hand it one. The table
  marks what it doesn't know — it never guesses and never drops a column.

The same seven traces, this time through the real judge — one run, model
`openai/gpt-oss-120b` via Groq, 18 Oct 2026; judge output varies between
runs:

```text
trace                task            discont  final_state            bounds                 critical               trajectory_quality
clean                travel_booking  -        PASS                   PASS                   PASS                   PASS 0.96
critical_recovered   travel_booking  -        PASS                   PASS                   FAIL (step 3)          WARN 0.70 (step 4)
empty                travel_booking  -        FAIL (no single step)  WARN (no single step)  WARN (no single step)  WARN (no single step)
failed               travel_booking  -        FAIL (step 3)          PASS                   PASS                   PASS 0.92
first_class_overlap  travel_booking  -        PASS                   FAIL (step 2)          FAIL (step 2)          WARN 0.70 (step 3)
unlisted_action      travel_booking  -        PASS                   FAIL (step 3)          PASS                   WARN 0.70 (step 3)
wasteful             travel_booking  -        PASS                   PASS                   PASS                   FAIL 0.30 (step 3)
```

Only `trajectory_quality` is model-judged; the other three columns are
deterministic code. One corrected step is a `WARN`, and only a repeated
loop is a `FAIL`. The score is coarse — differences under about 0.1 mean
little — and the cited step can differ between runs.

## The four questions, in plain words

A **trajectory** is one recorded episode. Each is graded against a **task
spec** — a small JSON file holding that task's rules — on four separate
questions:

| # | Dimension | Type | Question it answers |
|---|-----------|------|---------------------|
| 1 | **Final state** | Deterministic code | Did the episode end in the correct state? |
| 2 | **Bounds** | Deterministic code | Did every action stay inside the allowed scope — allowed names, allowed argument values? |
| 3 | **Critical mistakes** | Deterministic code | Did the agent ever commit a forbidden action — *even if it later recovered*? |
| 4 | **Trajectory quality** | LLM judge | Was any effort wasted — identical actions repeated, steps undone or redone, steps that add nothing to the goal? |

Dimensions 1–3 are code that always gives the same answer. Dimension 4 is
fuzzy by nature — "was this step necessary?" resists hard rules — so an LLM
judges it; the two never mix inside one module. Mistakes are sticky: the
critical column scans the whole history, so a forbidden move at step 3 is
still FAIL at the end.

Every cell holds a verdict — **PASS**, **WARN**, **FAIL**, or **ERROR** —
with a one-sentence reason and evidence citing step numbers. ERROR means the
grader could not judge (a setup problem), never that the trajectory broke a
rule. Full verdict reference: [docs/DESIGN.md](docs/DESIGN.md).

## What works today

**Steps 1–9 are done, including the real LLM judge**: contract, strict
output validation, prompt builder, the quality-grader adapter,
`evaluate(judge=...)` wiring, and one real client (Groq over plain HTTP) —
verified live on Groq (`openai/gpt-oss-120b`) on 18 Oct 2026. The CLI and
everything after are planned — the [roadmap](#roadmap) says exactly what's
next.

Works today, as a library:

- load a trajectory or a task spec from JSON, validated structurally;
- grade the three deterministic dimensions with `evaluate()` — one broken
  grader degrades its own column to ERROR, never the whole run;
- judge trajectory quality with a real LLM by passing `judge=` — a model is
  only ever called when you pass one, and its key comes from the
  environment;
- render a single report as readable text or machine JSON, and save/load
  reports to files;
- line many reports up in one comparison table, as plain text or Markdown.

Planned: the `uv run traject` CLI (Step 10), a
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

**Judge path quality with a real LLM** *(needs a Groq API key; without a
judge the fourth column stays `not evaluated`, and the default test suite
uses a scripted fake instead)*

```python
from trajecteval import LLMJudge, Trajectory, evaluate, load_spec

report = evaluate(
    Trajectory.from_json_file("trace.json"),
    load_spec("travel_booking"),
    judge=LLMJudge(),  # reads TRAJECTEVAL_JUDGE_API_KEY; Groq model default
)
```

The judge only judges *path quality*, meaning wasted effort: identical
actions repeated, actions that were undone or reversed, and steps that add
nothing to reaching the goal. Severity is a rule, not a mood — nothing
wasted → `pass`; a single wasted, undone, or corrected step → `warn`,
never `fail`; `fail` only when waste dominates the path (the same action
three or more times, or several wasted steps forming a loop). Final state,
bounds, and critical mistakes stay with the deterministic graders: a broken
rule never lowers the quality verdict or score — only the effort it wasted
counts here. A model answer that fails validation becomes an
ERROR column with evidence, never a guessed verdict.

Alongside the verdict the judge returns a `score` from 0 to 1, shown
next to it (`WARN 0.72`). The score comes from the model, so it can vary
between runs — the verdict is the primary result, and there is still no
overall number across dimensions.

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

Adding a new task means writing one JSON file — no new grader code. Deeper
semantics — absent vs. empty sections, `null` handling, the full permit-list
rules — live in [docs/DESIGN.md](docs/DESIGN.md).

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

## Testing & development

Every command goes through `uv`:

```bash
uv run pytest                          # full suite (offline by default)
uv run pytest tests/test_models.py -v  # one file
uv run pytest -k discontinu            # by name
```

Conventions:

- One module = one responsibility; one test file per module.
- The default suite is **offline and deterministic**: hand-made fixtures are
  the ground truth, and unit tests use a small scripted fake judge
  (`ScriptedJudge`) — no network, no API key, no spend.
- Live tests exist behind a marker: `uv run pytest -m live` calls the real
  Groq client with two loose sanity checks (allowed verdicts only — never a
  fixed expected verdict) and needs `TRAJECTEVAL_JUDGE_API_KEY`. They are
  deselected by default, so the default suite never touches the network.

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

Loading rules are enforced by the loaders: step indices must be sequential
(gaps and duplicates are rejected, naming the step), only structural problems
raise, and a state discontinuity between steps is *reported* — it never
blocks loading. Full rules: [docs/DESIGN.md](docs/DESIGN.md).

## Project structure

What exists today:

```text
TrajEval/
├── pyproject.toml               # project metadata + dependencies (uv owns this)
├── uv.lock                      # pinned versions (reproducible installs)
├── README.md
├── docs/
│   └── DESIGN.md                # deep rules: verdicts, evidence, permits, loading
├── tasks/
│   └── travel_booking.json      # task one's rules — the only place rules live
├── src/trajecteval/
│   ├── __init__.py              # public API surface
│   ├── errors.py                # TrajEvalError + TrajectoryError / SpecError / ReportError
│   ├── models.py                # Step, Action, Trajectory, Discontinuity + JSON I/O
│   ├── matching.py              # json_equal + the subset matcher
│   ├── results.py               # Verdict, Dimension, Evidence, GraderResult
│   ├── task_spec.py             # TaskSpec + load-by-task-id (no registry)
│   ├── graders/                 # deterministic graders — one file per dimension
│   │   ├── base.py              # Grader protocol: grade(spec, trajectory)
│   │   ├── final_state.py
│   │   ├── bounds.py
│   │   └── critical.py
│   ├── report.py                # evaluate() + JSON/text/file renderers
│   ├── compare.py               # cross-trace comparison table (text + markdown)
│   └── judge/                    # judge code — never in graders/
│       ├── base.py               # contract (protocol + JudgeResult/JudgeNote)
│       ├── prompt.py             # exact system prompt + size cap (no truncation)
│       ├── validate.py           # strict parse of the model's answer
│       ├── quality.py            # TrajectoryQualityGrader: Judge → Grader
│       ├── llm.py                # OpenAI-compatible client (Groq) + LLMJudge
│       └── fake.py               # ScriptedJudge — unit tests pay nothing
└── tests/                       # one test file per module, plus fixtures:
    └── fixtures/
        ├── minimal.json         # loader fixture
        ├── discontinuous.json   # discontinuity fixture
        └── travel_booking/      # seven trajectories + expected_verdicts.json
```

Planned, in roadmap order (not built): the CLI (`cli.py`), the simulated
environment and recorder (`environment.py`, `recorder.py`), task two's
fixtures, and — only if real attempt logs ever exist — a Debuggernaut
adapter. The [roadmap](#roadmap) is the source of truth for what's next.

**Why `src/` layout?** It prevents a classic trap: Python silently importing
your working directory instead of the installed package, so tests pass on your
machine and break elsewhere. With `src/`, you only ever test what is properly
installed — uv does that installation automatically.

## Toy tasks

Task one is travel booking: seven fixture trajectories plus a
machine-readable expected-verdict table beside them. Task two (support-ticket
triage, Step 13) is smaller and shaped differently, and exists to prove the
graders run unchanged on a new domain — same code, only the spec and
fixtures change.

## Roadmap

| # | Step | Status |
|---|------|--------|
| 1 | Project scaffold (`uv init`, src layout, first test) | ✅ done |
| 2 | Models: `Step`, `Trajectory`, JSON loading | ✅ done |
| 3 | Result types: `Verdict`, `Dimension`, `Evidence`, `GraderResult` | ✅ done |
| 4 | Task spec + grader contract: data-file rules, load by task id, `Grader` protocol | ✅ done |
| 5 | Fixtures for task one — seven trajectories (clean, wasteful, critical_recovered, first_class_overlap, failed, unlisted_action, empty) + the machine-readable expected-verdict table | ✅ done |
| 6 | Three deterministic graders (final state, bounds, critical) — reproduce the fixture answer key exactly | ✅ done |
| 7 | Report: `evaluate()` runs the three deterministic dimensions first (a crashed grader degrades its own column to ERROR, never the run), then renders JSON + readable text; the LLM judge is an optional `judge=` addition (Step 9) | ✅ done |
| 8 | Comparison table across traces — a dimension with no result gets an explicit marker, the same honesty rule as `render_text`'s `not evaluated` lines, never a silently dropped column | ✅ done |
| 9 | Real LLM judge: contract + strict output validation + prompt builder + `TrajectoryQualityGrader` adapter + `evaluate(judge=...)` + one real client (Groq default, stdlib HTTP) + `ScriptedJudge` for offline tests; live tests behind `pytest -m live` | **✅ done** — verified live on Groq (openai/gpt-oss-120b), 18 Oct 2026 |
| 10 | CLI (`uv run traject ...`) | ⏳ planned |
| 11 | Recorder + simulated environment for travel_booking | ⏳ planned |
| 12 | LLM-agent demo: agent → recorder → report, real traces (needs an API key) | ⏳ planned |
| 13 | Toy task two — same graders, new domain, graders unchanged | ⏳ planned |

**Later / maybe** (not scheduled, one line each):

- Other judge backends (e.g. Jev — read about, not verified)
- Debuggernaut adapter (only if real attempt logs exist)

## Design notes & FAQ

**Why not one overall score?**
Because a single number destroys information. Two traces can average to 0.8 for
completely different reasons — one had a critical mistake it recovered from, the
other was just slow. The report must let a human see that difference quickly. If
you want a scalar, average the columns yourself and accept what you lose.

**Why is only one dimension LLM-judged?**
Whether effort was genuinely wasted is fuzzy; "was this necessary?" resists hard
rules. Everything else is a fact about recorded states and actions — code checks
facts better than a language model does. Using an LLM where rules suffice adds
cost, nondeterminism, and hallucination risk for no benefit.

**Why do rules live in task spec files?**
Because otherwise every new task means editing grader code, and grader code is
exactly what should stay boring and domain-agnostic. A spec file is reviewable,
diffable, and testable as data. It also forces rules to be explicit: if
something is graded, it's written down in the spec — not buried in a branch.

**Why is the judge injectable?**
So the default tests never touch the network — a small scripted fake judge
stands in — while the product runs a real one: `LLMJudge()` calls Groq over
plain standard-library HTTP behind the committed contract
(`src/trajecteval/judge/base.py`). Swapping judges is a one-line change at
the call site, and API keys stay out of core logic: `evaluate()` only ever
calls a model when you pass it a judge, and the key comes from
`TRAJECTEVAL_JUDGE_API_KEY` — never a file in the repo, never printed.

**Why dataclasses instead of Pydantic?**
The project parses its own well-defined JSON format rather than validating
arbitrary external input at an API boundary. Stdlib dataclasses keep the core
dependency-free and the mental model small. Revisiting later if needs grow is
normal.

## License

TBD.
