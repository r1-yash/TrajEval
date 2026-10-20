# TrajEval

> Grade how an AI agent completes a task, not just whether it finishes.

TrajEval records an agent's whole episode — every action, with the state of
the world before and after it — and grades the episode on four separate
questions:

1. Did it end in the right state?
2. Did it stay within the allowed actions?
3. Did it ever make a critical mistake?
4. Was effort wasted along the way?

An agent can reach the right answer by a bad path: it makes a forbidden move,
undoes it, and still ends correctly. A checker that only looks at the ending
passes that run. TrajEval does not.

---

## Why TrajEval?

Task: *book one economy flight and pay by card.* The task's rules make paying
with **points** a forbidden move.

**Clean run**

```text
search_flights → book_flight (economy) → pay (card)
```

**Recovered run**

```text
search_flights → book_flight (economy) → pay (points)   ← forbidden
              → refund_payment → pay (card)
```

Both runs end in exactly the same state: confirmed, paid by card. A
final-state checker scores them identically.

TrajEval does not:

```text
trace                final_state  bounds  critical       trajectory_quality
clean                PASS         PASS    PASS           PASS 0.96
critical_recovered   PASS         PASS    FAIL (step 3)  WARN 0.70 (step 4)
```

The mistake at step 3 stays a FAIL even though the agent recovered.
Recovery does not erase what happened. One number can't show this; four
columns can.

---

## How it works

```text
Agent ──► Recorder ──► Environment
              │
              ▼
         Trajectory (JSON)          Task spec (JSON)
              │                          │
              └────────────┬─────────────┘
                           ▼
        ┌──────────────┬───────────┬──────────────┐
        │ Final state  │  Bounds   │   Critical   │   deterministic code
        └──────┬───────┴─────┬─────┴──────┬───────┘
               └─────────────┼────────────┘
                             ▼
                    LLM quality judge              one fuzzy question
                             ▼
                  Report  /  comparison table
```

| # | Dimension | Type | Question |
|---|-----------|------|----------|
| 1 | **Final state** | code | Did the episode end in the expected state? |
| 2 | **Bounds** | code | Did every action use an allowed name and allowed argument values? |
| 3 | **Critical mistakes** | code | Was a forbidden action ever taken — even if later undone? |
| 4 | **Trajectory quality** | LLM judge | Was effort wasted: repeated actions, undone steps, steps that add nothing? |

Every cell is a verdict — **PASS**, **WARN**, **FAIL**, or **ERROR** — with a
one-sentence reason and the step numbers it is based on. `ERROR` means "could
not judge", never "failed". There is no overall score, by design: one number
hides *why* a run was good or bad.

---

## Seven runs, one table

The repository ships seven hand-made trajectories for the travel-booking task.
Each one is designed to trip a different check.

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

*Quality column: judged by `openai/gpt-oss-120b` via Groq, 18 Oct 2026. It is
model-generated and can vary between runs; the other three columns are
deterministic.*

- `critical_recovered` — correct ending, critical mistake still flagged.
- `wasteful` — every rule satisfied, but the same search runs five times. Only the judge catches this.
- `failed` — wrong ending, yet an efficient path: the dimensions are independent.
- `empty` — nothing happened, so nothing gets a silent PASS.

---

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and Python 3.14+.

```bash
git clone https://github.com/r1-yash/TrajEval.git
cd TrajEval
uv sync
uv run pytest
```

Grade the seven trajectories (free, no model call):

```bash
uv run traject evaluate \
  tests/fixtures/travel_booking/clean.json \
  tests/fixtures/travel_booking/critical_recovered.json \
  tests/fixtures/travel_booking/empty.json \
  tests/fixtures/travel_booking/failed.json \
  tests/fixtures/travel_booking/first_class_overlap.json \
  tests/fixtures/travel_booking/unlisted_action.json \
  tests/fixtures/travel_booking/wasteful.json
```

The quality column reads `not evaluated` until you pass `--judge`.

### Add the LLM judge

Create a `.env` file (it is gitignored):

```bash
TRAJECTEVAL_JUDGE_API_KEY=<your Groq API key>
TRAJECTEVAL_JUDGE_MODEL=openai/gpt-oss-120b     # optional; this is the default
```

```bash
uv run --env-file .env traject evaluate \
  tests/fixtures/travel_booking/critical_recovered.json --judge
```

```text
task travel_booking (trajectory travel_booking)
  final_state PASS: final state at step 5 matches expected_final_state (2 field(s))
  bounds PASS: all 5 step(s) are within allowed_actions
  critical FAIL: 1 critical mistake(s): step 3: pay {'method': 'points'}
    critical_error (step 3, pay): {"action": "pay", "args": {"method": "points"}}
  trajectory_quality WARN 0.70: A refund undone the previous payment, causing unnecessary extra effort.
    undo_action (step 4): Refund payment reverted the confirmed status, requiring a second payment.
```

A model is only ever called when you pass `--judge` (or run an agent). Exit
code `0` means the command ran; nonzero means it could not run. A FAIL
verdict is a result, not a command failure.

---

## Run an LLM agent

TrajEval ships a small simulated travel-booking world. An LLM acts in it, a
recorder writes down every step, and the same graders grade the result.

The instruction the agent receives:

> Book a flight for one traveler: one economy seat on flight TR420 from SFO to NRT, and pay for it by card. When the flight is booked and paid for, you are done.

```bash
uv run --env-file .env traject run travel_booking \
  --model qwen/qwen3.8-27b \
  --model openai/gpt-oss-120b \
  --out runs/

uv run --env-file .env traject evaluate runs/*.json --judge
```

```text
trace                                task            discont  final_state  bounds  critical  trajectory_quality
travel_booking__openai_gpt-oss-120b  travel_booking  -        PASS         PASS    PASS      PASS 0.96
travel_booking__qwen_qwen3.8-27b     travel_booking  -        PASS         PASS    PASS      WARN 0.72 (step 2)
```

Both agents completed the task and broke no rule. Only the judge separates
them: qwen searched twice. *Run 19 Oct 2026, one run per model. The
`gpt-oss-120b` run was judged by the same model that produced it.*

These two recorded runs are committed, so you can grade them for free:

```bash
uv run traject evaluate examples/runs/*.json
```

The full log of the real runs is in [docs/RUNS.md](docs/RUNS.md).

### Record your own run from Python

```python
from trajecteval import Recorder, TravelBookingEnvironment

recorder = Recorder(TravelBookingEnvironment(), metadata={"agent": "demo"})
recorder.act("search_flights", {"origin": "SFO", "destination": "NRT"})
recorder.act("book_flight", {"flight": "TR420", "cabin": "economy", "passengers": 1})
recorder.act("pay", {"method": "card"})
recorder.trajectory().to_json_file("recorded_clean.json")
```

```bash
uv run traject evaluate recorded_clean.json
```

```text
task travel_booking (trajectory travel_booking)
  final_state PASS: final state at step 3 matches expected_final_state (2 field(s))
  bounds PASS: all 3 step(s) are within allowed_actions
  critical PASS: no critical_error_patterns matched in 3 step(s)
  not evaluated: trajectory_quality
```

The world models physics, not rules: booking first class or paying with points
*works*, because the graders are what judge those moves. An action the world
cannot perform (paying before booking) is still recorded as a step, then
raised to the agent.

---

## Use as a library

```python
from trajecteval import LLMJudge, Trajectory, compare, evaluate, load_spec, render_table, render_text

spec = load_spec("travel_booking")
traj = Trajectory.from_json_file("trace.json")

report = evaluate(traj, spec)                      # three deterministic dimensions
report = evaluate(traj, spec, judge=LLMJudge())    # + LLM quality judge (opt-in)
print(render_text(report))

pairs = [("run_a", evaluate(traj, spec)), ("run_b", report)]
print(render_table(compare(pairs)))                # comparison table
```

A grader that crashes becomes an `ERROR` in its own column; it never kills the
run. A judge answer that fails validation (bad JSON, a cited step that doesn't
exist) is an `ERROR` too, never a guessed verdict.

---

## Task specs

Graders know *how* to check, never *what* to check. All task rules live in one
JSON file per task — `tasks/travel_booking.json`, trimmed:

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

- `expected_final_state` — listed fields must match the final state.
- `allowed_actions` — action → permitted argument values. An unlisted action is out of bounds.
- `critical_error_patterns` — moves that are mistakes even if later undone.

A new task means a new JSON file, not new grader code.

---

## Trajectory format

One episode per JSON file:

```json
{
  "task_id": "travel_booking",
  "metadata": { "agent": "demo" },
  "steps": [
    {
      "index": 1,
      "state_before": { "status": "new" },
      "action": { "name": "search_flights", "args": { "origin": "SFO", "destination": "NRT" } },
      "state_after": { "status": "searching", "searches": 1 }
    }
  ]
}
```

Step indices must be sequential. A gap between one step's `state_after` and the
next step's `state_before` is reported as a discontinuity, not a crash.

---

## Project structure

```text
src/trajecteval/
├── models.py          Trajectory, Step, Action
├── results.py         Verdict, Dimension, Evidence, GraderResult
├── task_spec.py       load a task's rules from tasks/<id>.json
├── graders/           final_state.py · bounds.py · critical.py   (deterministic)
├── judge/             prompt · validate · quality · llm client   (LLM, kept separate)
├── report.py          evaluate() + text / JSON rendering
├── compare.py         comparison table
├── recorder.py        records an agent's actions into a Trajectory
├── agent.py           LLM agent loop
├── envs/              simulated worlds
└── cli.py             the `traject` command
tasks/                 task rules (JSON)
tests/fixtures/        seven travel-booking trajectories + expected verdicts
examples/runs/         two recorded LLM-agent runs
docs/                  DESIGN.md (rules in depth) · RUNS.md (log of real runs)
```

---

## Testing

```bash
uv run pytest             # offline: no network, no API key, no cost
uv run pytest -m live     # calls the real judge; needs TRAJECTEVAL_JUDGE_API_KEY
```

The default suite uses hand-made fixtures as ground truth and a scripted fake
judge. The seven fixtures replay byte-exact through the simulated world.

---

## Design principles

1. **Per dimension, not one score.** The breakdown is the result.
2. **Code checks facts; an LLM judges only what code can't.** The two never share a module.
3. **Mistakes are sticky.** Recovery never downgrades a critical mistake.
4. **Evidence or it didn't happen.** FAIL and WARN cite step numbers.
5. **ERROR is not FAIL.** A grader that can't judge says so.

---

## Limitations

- One task so far (travel booking), one simulated world.
- The quality score comes from an LLM: verdicts were stable across repeated runs, but scores moved by a few hundredths.
- Free-tier API rate limits can interrupt agent runs.
- Agent and judge models can overlap; the table notes when they do.

## Future directions

- A second task in a different domain, to test the graders beyond travel booking.
- Retry on provider rate limits.
- Comparing the judge's verdicts with human labels.

## License

TBD
