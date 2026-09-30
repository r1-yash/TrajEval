# TrajEval

**Evaluate AI agents on *how* they work, not just whether they finished.**

Most evaluators look at the final answer and ask: *did it pass?* TrajEval records the
agent's entire episode — every action, in order, with the state before and after — and
grades that **trajectory** on four separate axes, producing an inspectable per-trace
report and a comparison table across traces. Not one scalar score: a breakdown of
*why*.

```text
                  ┌─────────────────── A trajectory (episode) ───────────────────┐
                  │  step 1 ──▶ step 2 ──▶ step 3 ──▶ ... ──▶ final state       │
                  │  (state before / action / state after, for every step)       │
                  └──────────────────────────┬───────────────────────────────────┘
                                             │
                      ┌──────────────────────┼──────────────────────┐
                      ▼                      ▼                      ▼
               DETERMINISTIC            DETERMINISTIC          DETERMINISTIC
                Final-state             Bounds/limits          Critical-mistake
                  grader                   grader                 grader
               "did it land            "did every step        "did it ever do
                in the right            stay inside             the forbidden
                  spot?"                 allowed space?"         thing?"
                      │                      │                 (sticky: a later
                      │                      │                  recovery does
                      │                      │                  NOT erase it)
                      └──────────┬───────────┴───────────┐
                                 ▼                       ▼
                          LLM TRAJECTORY JUDGE     per-dimension results
                        "efficient & coherent?"    (verdict + WHY + evidence)
                                 │                       │
                                 └───────────┬───────────┘
                                             ▼
                                  ┌─── Report (1 trace) ───┐   ┌─ Comparison table ─┐
                                  │ breakdown per dimension │──▶│ trace A vs B vs C  │
                                  └─────────────────────────┘   └────────────────────┘
```

---

## Table of contents

- [Why TrajEval exists](#why-trajeval-exists)
- [The four grading dimensions](#the-four-grading-dimensions)
- [How it works](#how-it-works)
- [Core design principles](#core-design-principles)
- [Installation](#installation)
- [Usage](#usage)
- [The trajectory format](#the-trajectory-format)
- [Project structure](#project-structure)
- [The toy environment](#the-toy-environment)
- [Real data: Debuggernaut](#real-data-debuggernaut)
- [Development](#development)
- [Roadmap](#roadmap)
- [Design notes & FAQ](#design-notes--faq)

---

## Why TrajEval exists

An agent can **succeed badly** and **fail well**:

- **Succeed badly:** it reached the right answer after deleting a file it shouldn't
  have, retrying five times, and wandering through twelve pointless steps. A
  pass/fail evaluator gives this a perfect score.
- **Fail well:** it did everything cleanly but the final environment state was off by
  one detail. A pass/fail evaluator gives this a zero and tells you nothing about
  *what* went wrong.

TrajEval separates these concerns. It tells you, per trace:

1. Did it reach the correct **final state**? *(deterministic)*
2. Did every action stay within **allowed bounds**? *(deterministic)*
3. Did it commit any **critical mistake**, even if it later recovered? *(deterministic)*
4. How **efficient and coherent** was the path? *(LLM-judged — "was this an unnecessary
   detour" is hard to encode as a hard rule)*

The output is an **inspectable report per trace**, plus a **comparison table across
multiple traces** — a breakdown of *why*, not just a number.

---

## The four grading dimensions

| # | Dimension | Type | Question it answers |
|---|-----------|------|---------------------|
| 1 | **Final state** | Deterministic | Did the episode end in the correct state? |
| 2 | **Bounds** | Deterministic | Did every action stay inside the allowed scope (files touched, tools called, values written)? |
| 3 | **Critical mistakes** | Deterministic | Did the agent ever commit a forbidden action — *even if it recovered afterwards*? |
| 4 | **Trajectory quality** | LLM judge | Was the path efficient and coherent, or full of detours, thrash, and redundant actions? |

Each dimension returns an independent result object — never blended into a single
number by default:

```text
GraderResult
├── verdict      : PASS | FAIL | WARN | ERROR     (the "what")
├── reason       : str                            (the "why", human-readable)
├── evidence     : list[Evidence]                 (the "prove it": step numbers, values)
└── dimension    : str                            (which axis produced this)
```

### The critical-mistake rule (the one everybody gets wrong)

Recovery does **not** erase mistakes.

```text
step 4:  agent deletes config.prod.json      ← CRITICAL MISTAKE (logged)
step 9:  agent restores config.prod.json     ← recovery (logged separately, at most)
step 12: agent finishes, everything looks fine

❌ Wrong:  critical-mistake dimension → PASS  ("it fixed it in the end")
✅ TrajEval: critical-mistake dimension → FAIL ("it happened; here is the step")
```

A later fix is interesting information — TrajEval can report it as a *recovery note* —
but the mistake dimension stays failed. This mirrors reality: in production, the
delete already happened.

---

## How it works

A **trajectory** (or *episode*) is an ordered list of steps. Each step records the
state before, the action taken, and the state after:

```text
Trajectory
 └── steps: [ Step, Step, Step, ... ]
        Step
        ├── index        : int          # 1-based position in the episode
        ├── state_before : dict         # environment snapshot before the action
        ├── action       : Action       # what the agent did (name + arguments)
        └── state_after  : dict         # environment snapshot after the action
```

Grading is a **pipeline**:

```text
Trajectory ──▶ [ grader 1 ──▶ GraderResult ]   deterministic, pure functions
           ──▶ [ grader 2 ──▶ GraderResult ]
           ──▶ [ grader 3 ──▶ GraderResult ]
           ──▶ [ LLM judge ─▶ GraderResult ]   injectable; faked in tests
           │
           ▼
      Report ──▶ JSON file (machine-readable) + rendered text (human-readable)
           │
           ▼ (across many traces)
      Comparison table (one row per trace, one column per dimension)
```

Two facts make this architecture hold up:

1. **Deterministic graders and the LLM judge implement the same interface but live in
   separate modules.** One is code that always gives the same answer; the other is
   probabilistic. A bug — or a hallucination — in one can never contaminate the other.
   The judge is *injected*, so tests run against a fake with zero API calls.
2. **Every result carries a reason, not just a boolean.** From the first grader onward,
   results have fields for the verdict, the explanation, and the evidence (which step
   numbers, what values). Reports are possible because the data was born rich.

---

## Core design principles

These are the rules the codebase is held to:

1. **Per-dimension, not scalar.** No mega-score as the primary output. Any aggregate
   is a convenience, clearly secondary to the breakdown.
2. **Deterministic ≠ LLM.** Separate modules, shared interface, injectable judge.
3. **Mistakes are sticky.** Recovery never downgrades a critical mistake to a pass.
4. **Evidence or it didn't happen.** Every verdict points at concrete steps/values.
5. **Small and focused.** Single environment, single task type. No plugin registries,
   no over-engineering until a real need appears.
6. **Tested without the network.** All tests run offline against fake judges and
   fixture trajectories.
7. **Python/uv only.** Everything runs as `uv run ...`. No pip, no venv activation,
   no Node/npm/TypeScript anywhere.

---

## Installation

Requires [uv](https://docs.astral.sh/uv/) and Python 3.10+ (developed on 3.13).

```bash
# clone the repo
git clone <repo-url> TrajEval
cd TrajEval

# install all dependencies (creates .venv + resolves uv.lock)
uv sync

# run the test suite
uv run pytest
```

Adding new dependencies is always:

```bash
uv add <package>          # runtime dependency
uv add --dev <package>    # development/test dependency (e.g. pytest)
```

---

## Usage

> **Status:** under construction — steps 1–12 of the build plan. Examples below show
> the target CLI surface; sections marked ⏳ are not implemented yet.

### Evaluate one trajectory ⏳

```bash
uv run traject eval traces/attempt-042.json
```

### Evaluate a directory of trajectories, with an LLM judge ⏳

```bash
export TRAJECT_JUDGE_API_KEY=sk-...
uv run traject eval traces/ --judge llm --out reports/
```

### Compare traces side by side ⏳

```bash
uv run traject compare reports/*.json
```

```text
┌───────────────┬──────────────┬────────┬──────────────┬────────────────┐
│ trace         │ final state  │ bounds │ critical err │ trajectory qual│
├───────────────┼──────────────┼────────┼──────────────┼────────────────┤
│ attempt-042   │ PASS         │ PASS   │ FAIL (step 4)│ WARN (3 detours)│
│ attempt-043   │ PASS         │ FAIL   │ PASS         │ PASS           │
│ attempt-044   │ FAIL         │ PASS   │ PASS         │ PASS           │
└───────────────┴──────────────┴────────┴──────────────┴────────────────┘
```

### As a library ⏳

```python
from trajecteval import Trajectory, evaluate

traj = Trajectory.from_json_file("traces/attempt-042.json")
report = evaluate(traj)

for result in report.results:
    print(f"{result.dimension:20} {result.verdict:6}  {result.reason}")
```

---

## The trajectory format

Trajectories are plain JSON — no database, no special tooling. One episode per file:

```json
{
  "task_id": "fix-null-check-42",
  "metadata": {"agent": "debuggernaut", "attempt": 42},
  "steps": [
    {
      "index": 1,
      "state_before": {"files": {"src/parser.py": "def parse(x): ..."}},
      "action": {"name": "read_file", "args": {"path": "src/parser.py"}},
      "state_after": {"files": {"src/parser.py": "def parse(x): ..."}}
    },
    {
      "index": 2,
      "state_before": {"files": {"src/parser.py": "def parse(x): ..."}},
      "action": {"name": "edit_file", "args": {"path": "src/parser.py", "old": "x[0]", "new": "x[0] if x else None"}},
      "state_after": {"files": {"src/parser.py": "def parse(x): ... x[0] if x else None ..."}}
    }
  ]
}
```

**Loading rules** (guaranteed by the dataclass layer):

- `index` values are sequential; a gap or duplicate raises a clear error.
- `state_after[i]` and `state_before[i+1]` should agree — a mismatch is *itself*
  reportable evidence that something is off about the recorder.
- Unknown top-level keys are preserved in `metadata`, never silently dropped.

---

## Project structure

```text
TrajEval/
├── pyproject.toml               # project metadata + dependencies (uv owns this)
├── uv.lock                      # exact dependency versions (reproducible installs)
├── README.md                    # this file
├── src/
│   └── trajecteval/
│       ├── __init__.py          # public API surface
│       ├── models.py            # Step, Action, Trajectory dataclasses + JSON I/O
│       ├── results.py           # GraderResult, Verdict, Evidence — shared vocabulary
│       ├── graders/             # deterministic graders (one file per dimension)
│       │   ├── __init__.py
│       │   ├── base.py          # Grader protocol (the shared interface)
│       │   ├── final_state.py
│       │   ├── bounds.py
│       │   └── critical.py
│       ├── judge/               # LLM trajectory judge (separate on purpose)
│       │   ├── __init__.py
│       │   ├── base.py          # Judge protocol
│       │   ├── fake.py          # offline fake used in tests
│       │   └── llm.py           # real client (provider decided at Step 8)
│       ├── report.py            # per-trace report assembly + rendering
│       └── compare.py           # cross-trace comparison table
├── tests/
│   ├── test_smoke.py
│   ├── fixtures/                # toy trajectories as JSON files
│   └── ...                      # one test file per module
└── examples/
    └── sandbox/                 # toy file-operation environment + demo traces
```

**Why `src/` layout?** It prevents a classic trap: Python silently importing your
local working directory instead of the installed package, so tests pass on your
machine and break everywhere else. With `src/`, you only ever test what is properly
installed — uv does that installation in editable mode automatically.

---

## The toy environment

Before touching real data, TrajEval is built and tested against a **file-operation
sandbox**: a small temp-directory world where the "agent" reads, edits, creates, and
deletes files. This gives us:

- **Obvious violations** to grade: editing a file outside the allowed set, touching
  `secrets/`, writing to a read-only path.
- **Obvious critical mistakes**: deleting a file, then "recovering" by recreating it.
- **Obvious detours** for the LLM judge: reading the same file six times, undoing and
  redoing the same edit.
- **Ground truth we control**: we author the fixtures, so we know exactly which
  dimension should pass or fail before running the grader.

The sandbox mirrors what Debuggernaut actually does (editing source files to fix
bugs), so every lesson transfers directly to the real data.

---

## Real data: Debuggernaut

Once the core is working, TrajEval is pointed at **attempt logs from
[Debuggernaut]**, an autonomous bug-fixing agent, as real trajectory data.

Debuggernaut's logs are not in TrajEval's format — they are in *Debuggernaut's*
format. The adapter lives in one place (an *anti-corruption layer*), and its whole
job is:

```text
Debuggernaut attempt log  ──▶  adapter  ──▶  Trajectory  ──▶  same pipeline as toys
      (their format)                       (our format)      (zero special cases)
```

Because the toy sandbox uses the same shape, the graders never learn whether the
input came from a fixture or production.

[Debuggernaut]: <repo-url>

---

## Development

Every command goes through `uv`:

```bash
uv run pytest                     # full test suite
uv run pytest tests/test_models.py -v   # one file, verbose
uv run pytest -k sticky           # run tests whose name contains "sticky"
uv run pytest --cov               # coverage (needs: uv add --dev pytest-cov)
```

**Conventions:**

- One module = one responsibility; one test file per module.
- Every new grader must ship with: a passing fixture, a failing fixture, and an
  edge-case fixture (e.g. empty trajectory).
- No test may require network access or an API key — the LLM judge is faked.

---

## Roadmap

| # | Step | Status |
|---|------|--------|
| 1 | Project scaffold: `uv init`, src layout, first test | ✅ done |
| 2 | Core domain models: `Step`, `Trajectory`, JSON loading | ⏳ |
| 3 | Grader protocol + shared `GraderResult` types | ⏳ |
| 4 | Final-state grader (deterministic) | ⏳ |
| 5 | Bounds grader (deterministic) | ⏳ |
| 6 | Critical-mistake grader — sticky errors | ⏳ |
| 7 | Per-trace report assembly (JSON + readable rendering) | ⏳ |
| 8 | LLM trajectory judge: fake first, real client later | ⏳ |
| 9 | Comparison table across traces | ⏳ |
| 10 | File-operation sandbox toy env + fixture trajectories | ⏳ |
| 11 | Debuggernaut adapter (real attempt logs → `Trajectory`) | ⏳ |
| 12 | CLI entry point (`uv run traject ...`) | ⏳ |

---

## Design notes & FAQ

**Why not one overall score?**
Because a single number destroys information. Two traces can score 0.8 for completely
different reasons — one had a critical mistake it recovered from, the other was just
slow. The report must let a human see that difference in under five seconds. If you
want a scalar, average the columns yourself and accept what you lose.

**Why is only one dimension LLM-judged?**
Efficiency and coherence are genuinely fuzzy: "was this step necessary?" resists
hard rules without becoming brittle. Everything else is a fact about the recorded
states and actions — code checks facts better than a language model does. Using an
LLM where rules suffice would add cost, nondeterminism, and hallucination risk for
zero benefit.

**Why dataclasses instead of Pydantic?**
The project needs to parse its own well-defined JSON format, not validate arbitrary
user input at an API boundary. Stdlib dataclasses keep the core dependency-free and
the mental model small. (This was a deliberate beginner-friendly choice — revisiting
it later if validation needs grow is fine and normal.)

**Why is the LLM judge injectable?**
So tests never touch the network. `evaluate(traj, judge=FakeJudge())` runs offline
and deterministically in CI; swapping in the real client is a one-line change at the
call site. It also keeps the API key entirely out of core logic.

**Why *not* port the TypeScript reference project (TraceEval)?**
TraceEval is used as a **quality and scope bar** — the level of architectural
clarity, the separation of deterministic graders from an LLM judge, per-dimension
reporting, and the *size* of the project (small, single-environment, not overbuilt).
TrajEval is designed and built from first principles in Python; nothing is translated
line-by-line.

**What does a report look like?**
Two artifacts per trace: a `.json` file (full structure, machine-readable — verdicts,
reasons, evidence) and a human-readable rendering of the same data. The comparison
table is just many reports, aligned side by side.

---

## License

TBD.
