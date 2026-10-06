# TrajEval

**Evaluate AI agents on *how* they work, not just whether they finished.**

An AI agent completes a task by taking a series of actions: reading something,
calling a tool, changing a value — one after another, until the task is done.
Most evaluation setups look only at the end: did the final answer pass?

TrajEval records the whole episode instead. Every action, in order, with the
state of the world **before and after** each one — and grades that path on
four separate questions, so a report tells you *why* a run was good or bad,
not just whether it was.

> **Status:** under construction. Steps 1–3 (scaffold, models, result types)
> are done. Everything else in this README is a **plan** — roadmap rows are
> marked, and every usage example below says plainly whether it works yet.
> Nothing here is built unless its roadmap row says so.

## The four grading dimensions

| # | Dimension | Type | Question it answers |
|---|-----------|------|---------------------|
| 1 | **Final state** | Deterministic | Did the episode end in the correct state? |
| 2 | **Bounds** | Deterministic | Did every action stay inside the allowed scope — allowed action names, allowed argument values, and no actions outside the allowed set at all? |
| 3 | **Critical mistakes** | Deterministic | Did the agent ever commit a forbidden action — *even if it later recovered*? |
| 4 | **Trajectory quality** | LLM judge | Was the path efficient and coherent, or full of detours, thrash, and redundant actions? |

Dimensions 1–3 are code that always gives the same answer. Dimension 4 is
fuzzy by nature — "was this step necessary?" resists hard rules — so an LLM
judges it. The two never mix inside one module.

### The critical-mistake rule

Recovery does **not** erase mistakes:

```text
step 4:  agent does a forbidden action        ← CRITICAL MISTAKE (logged)
step 9:  agent undoes it                      ← recovery (noted at most)
step 12: episode ends, everything looks fine

❌ Wrong:  critical dimension → PASS  ("it fixed it in the end")
✅ TrajEval: critical dimension → FAIL ("it happened; here is the step")
```

The critical-mistake grader scans the **whole history**, so a mistake at step 4
is still there at step 12.

## Why TrajEval exists

An agent can **succeed badly** and **fail well**:

- **Succeed badly:** it reached the right result after a forbidden action it
  later recovered from, and twelve pointless detours. A pass/fail checker gives
  this a perfect score.
- **Fail well:** it worked cleanly but the final state was off on one field. A
  pass/fail checker gives this a zero and says nothing about what went wrong.

TrajEval separates these concerns. Per trace you get: did it reach the correct
final state, did it stay in bounds, did it commit a critical mistake, and how
efficient was the path — each with a verdict, a **reason**, and **typed
evidence citing step numbers**. The output is an inspectable report per trace,
plus a comparison table across traces. Not one scalar number: a breakdown of *why*.

## What a result looks like

```text
GraderResult
├── verdict   : PASS | FAIL | WARN | ERROR
├── reason    : str                       (the "why", human-readable)
├── evidence  : list[Evidence]            (facts cited: neutral, cites step numbers)
└── dimension : Dimension                 (closed set: final_state | bounds |
                                            critical | trajectory_quality)
```

Evidence is **neutral** — facts a result points at, not "proof of a
problem." The verdict carries the interpretation: a FAIL's smoking gun and
a PASS's recovery note look identical at the evidence level, and the report
renders them the same way.

**ERROR is not FAIL.** FAIL means the trajectory broke a rule. ERROR means the
*grader itself* could not judge — a missing spec field, malformed input. An
ERROR may cite evidence pointing at *what blocked it* (e.g. the broken spec
field) — a diagnostic, never an accusation. The two are never conflated,
because they need opposite responses from a human: fix the
trajectory vs. fix the setup. A grader **returns** a result — including ERROR —
rather than raising an exception.

## Task specs

Graders know **how to check**, never **what to check**. All task-specific rules
live in a plain JSON file per task:

```text
tasks/<task-id>.json
├── expected final state        # which fields must equal what
├── allowed actions             # action names + which argument values are permitted
└── critical-error patterns     # the forbidden moves, listed
```

- Graders receive `(spec, trajectory)` and work the same way for every task.
- Tasks are loaded **by task id** — no hardcoded registry.
- **Every rule in a spec is applied**, not just the first matching one. A
  trajectory violating two rules reports both.

This means adding a new task is writing one JSON file — no new grader code.

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

## Prior work

TrajEval is inspired by
[TraceEval](https://github.com/ayeangad/Trace-Eval), used as a **scope and
quality reference** — a bar for how clearly an evaluator of this kind can be
structured. TrajEval is built from scratch in Python; nothing is ported.

Differences in TrajEval's design (statements of fact, not judgments about
either project): per-step state recorded before and after each action; a
verdict vocabulary where ERROR (couldn't judge) is distinct from FAIL (broke a
rule); typed evidence that cites step numbers; an injectable judge whose tests
run fully offline; every rule in a task spec applied rather than only the first;
and two different toy tasks run through one unchanged pipeline.

## Installation

Requires [uv](https://docs.astral.sh/uv/) and **Python 3.14+**.

```bash
git clone <repo-url> TrajEval
cd TrajEval
uv sync            # installs dependencies + the package itself (editable)
uv run pytest      # run the test suite
```

Adding dependencies is always:

```bash
uv add <package>           # runtime dependency
uv add --dev <package>     # development/test dependency
```

## Usage

> **Not implemented yet.** The CLI does not exist until Step 10. This section
> shows the target interface; each example will be marked if it starts working.

**Evaluate one trajectory** *(not implemented yet — Step 10)*

```bash
uv run traject eval tests/fixtures/<task-one>/clean.json
```

**Compare traces** *(not implemented yet — Step 10)*

```bash
uv run traject compare reports/*.json
```

```text
trace               final state   bounds   critical err   trajectory qual
<task-one>/clean         PASS       PASS       PASS            PASS
<task-one>/critical      PASS       PASS       FAIL (step 4)   PASS
```

**As a library** *(not implemented yet — Step 7)*

```python
from trajecteval import Trajectory, evaluate

traj = Trajectory.from_json_file("trace.json")
report = evaluate(traj, spec=load_task_spec("<task-id>"))
for result in report.results:
    print(result.dimension, result.verdict, result.reason)
```

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
│   └── <task-one>.json
├── src/
│   └── trajecteval/
│       ├── __init__.py          # public API surface
│       ├── errors.py            # TrajectoryError (structural load errors only)
│       ├── models.py            # Step, Action, Trajectory, Discontinuity + JSON I/O
│       ├── results.py           # Verdict, Evidence, GraderResult — shared vocabulary
│       ├── task_spec.py         # TaskSpec + load-by-id (no registry)
│       ├── graders/             # deterministic graders — one file per dimension
│       │   ├── base.py          # Grader protocol: grade(spec, trajectory)
│       │   ├── final_state.py
│       │   ├── bounds.py
│       │   └── critical.py
│       ├── report.py            # evaluate() pipeline + JSON + readable text
│       ├── compare.py           # cross-trace comparison table
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
│   │   ├── <task-one>/          # four trajectories: clean, wasteful, critical, failed
│   │   └── <task-two>/          # smaller, differently shaped
│   ├── test_smoke.py
│   └── ...                      # one test file per module
└── examples/                    # LLM-agent demo (later)
```

**Why `src/` layout?** It prevents a classic trap: Python silently importing
your working directory instead of the installed package, so tests pass on your
machine and break elsewhere. With `src/`, you only ever test what is properly
installed — uv does that installation automatically.

## Toy tasks

Two toy tasks will carry the tests. **Both names/shapes are TBD** — candidates
are on the table and neither is file-based. Task one gets four fixtures (clean,
wasteful, critical mistake, failed). Task two is smaller and shaped differently,
and exists to prove the graders work unchanged on a new domain.

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

## Roadmap

| # | Step | Status |
|---|------|--------|
| 1 | Project scaffold (`uv init`, src layout, first test) | ✅ done |
| 2 | Models: `Step`, `Trajectory`, JSON loading | ✅ done |
| 3 | Result types: `Verdict`, `Evidence`, `GraderResult` | ✅ done |
| 4 | Task spec + grader contract: data-file rules, load by task id, `Grader` protocol | ⏳ planned |
| 5 | Toy task one + four fixtures (clean, wasteful, critical, failed) | ⏳ planned |
| 6 | Three deterministic graders (final state, bounds, critical) | ⏳ planned |
| 7 | Report: `evaluate()` pipeline, JSON + readable text | ⏳ planned |
| 8 | Comparison table across traces | ⏳ planned |
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

**Why is TraceEval referenced but not ported?**
TraceEval is a scope and quality reference — a bar for architectural clarity and
project focus. TrajEval is designed from first principles in Python; design
decisions are justified on their own terms (see Prior work).

**Where does Debuggernaut fit?**
Nowhere required. It's an optional final adapter (Step 14), included only if
real attempt logs exist. Everything through Step 13 runs on simulated tasks.

**What does a report look like?**
Two artifacts per trace: a `.json` file with verdicts, reasons, and evidence,
plus a human-readable rendering of the same data. The comparison table aligns
many reports side by side.

## License

TBD.
