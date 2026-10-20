# TrajEval design reference

The deep rules that the [README](../README.md) summarizes in a sentence or
two. Everything here is implemented and covered by the test suite.

## Verdicts

Every graded dimension of a trajectory gets exactly one verdict:

| Verdict | Means | Who acts |
|---------|-------|----------|
| PASS | Judged; no rule broken. | Nobody. |
| WARN | Judged; no rule broken, but something noteworthy — most often that *nothing was checked* (an empty rules section, an empty trajectory). A clean PASS would oversell it. | Read the cited evidence. |
| FAIL | Judged; a rule was broken. | The trajectory is at fault. |
| ERROR | The grader **could not judge**: a missing spec section, a wrong spec/trajectory pairing, or a crash inside the grader. | The setup is at fault, not the run. |

- **ERROR is never FAIL.** They need opposite responses from a human — fix
  the setup vs. fix the run — so the two are never conflated. An ERROR cites
  what *blocked* it (e.g. the broken spec field), a diagnostic, never an
  accusation against the trajectory.
- **WARN is never FAIL.** If an actual rule was broken, that is FAIL.
  WARN is reserved for "no rule broken, but here's what you should know",
  and WARN must cite the noteworthy thing as evidence.
- Graders **return** results — including ERROR — rather than raising. A crash
  inside a grader is caught by `evaluate()` and becomes an ERROR in that one
  column (evidence kind `grader_crash`), never the whole run.
- A dimension with no result at all renders as **`not evaluated`** in
  `render_text` and in the comparison table — never a silently dropped
  column, never a guess.

## Evidence

- Evidence is **neutral**: facts a result points at, not "proof of a
  problem". A FAIL's smoking gun and a PASS's recovery note look identical at
  the evidence level; the verdict carries the interpretation.
- FAIL and WARN must cite evidence with step numbers. Spec-side findings
  (nothing to do with the run itself) have `step: null`.
- Each item has a typed `kind` (e.g. `unlisted_action`, `empty_section`,
  `grader_crash`), an optional `field_name`, and a JSON-native `value`.

## Absent ≠ empty, and `null`

- A **missing** spec section that a grader needs means the grader *cannot
  judge*: it returns ERROR with evidence naming the section (`step: null`).
- A **present but empty** section (`{}` or `[]`) carries no rules: the
  grader reports WARN with evidence `empty_section` — there was nothing to
  check, never a silent PASS.
- `null` means **omitted**, everywhere in a spec: a `null` section behaves
  exactly like an absent one, and `null` `metadata` becomes `{}`.

## Permit-list semantics (`allowed_actions`)

- An action name not listed in `allowed_actions` is out of bounds. `{}` means
  the action is allowed with any arguments.
- Permits police only arguments the agent *supplied*: an argument absent from
  the permit map is unconstrained; a listed-but-absent argument is never a
  violation — permits cannot *require* an argument (completeness is
  `expected_final_state`'s job); an empty permit list permits nothing.
- Values match **by kind**: a boolean never matches a number (`true` ≠ `1`),
  while other JSON numbers match by value (`1` matches `1.0`).
- **Every rule in a spec is applied**, not just the first matching one — a
  trajectory violating two rules reports both.

## Loading rules

**Trajectory files**

- Step indices are 1-based, sequential, and duplicate-free; a gap or
  duplicate raises `TrajectoryError` naming the step.
- Only structural problems raise at load time (missing `task_id`, missing
  step keys, missing/non-string action name, bad index sequence). Every
  message names the step involved.
- A state discontinuity (one step's `state_after` disagreeing with the next
  step's `state_before`) does **not** prevent loading — it is *reported* by
  `find_discontinuities()` and carried into the report, because it is
  evidence, not breakage.
- Unknown top-level keys are preserved in `metadata`; on a collision, the
  explicit `metadata` entry wins.

**Task specs**

- Loaded by task id: `load_spec("travel_booking")` reads
  `tasks/travel_booking.json`. An id is a file name, never a path.
- A spec file is hand-written, so an unknown key is a typo: it is rejected at
  load, not silently ignored.

**Saved reports**

- `task_id` and `trajectory_task_id` must be non-empty strings; at most one
  result per dimension (duplicates are a corrupt report); result order is
  preserved, not required.
- Any structural problem raises `ReportError` (a `TrajEvalError`), wrapping
  the nested error with the original chained as `__cause__`. One
  `except TrajEvalError` catches every broken-file case in the project.

## The trajectory-quality judge

Dimension 4 is LLM-judged on purpose; the other three stay deterministic.
The rules around it:

- **Injection, not construction.** `evaluate()` never builds a client or
  calls a model on its own: the column appears only when a `judge=` is
  passed, and `judge=` simply appends the quality grader to whichever
  grader list is in use — a second grader for the quality column is
  rejected by the existing duplicate-column check. Without a judge the
  column renders `not evaluated`, exactly as before Step 9.
- **One path, five steps:** `TrajectoryQualityGrader.grade()` → task-id
  check first (same pinned order as the deterministic graders) → empty
  trajectory short-circuit (WARN, evidence `empty_trajectory`, the judge is
  called **zero** times — a paid model never answers "no path") →
  `LLMJudge.judge()` = `build_prompt()` → `OpenAIChatClient.complete()` →
  `parse_judge_output()`.
- **The prompt judges wasted effort only:** identical actions repeated,
  actions that were undone or reversed, and steps that add nothing to
  reaching the goal. It fences off final state, bounds, critical mistakes,
  and other broken rules (already graded): they must never lower the
  quality verdict or score, and when the only problem looks like a broken
  rule, only the effort it wasted is judged (for example an action that
  had to be undone), nothing more. Severity is a stated rule the model
  follows: nothing wasted → `pass`; a single wasted, undone, or corrected
  step → `warn`, never `fail`; `fail` only when waste dominates the path —
  the same action repeated three or more times, or several wasted steps
  forming a loop. It declares the trajectory data to analyze —
  never instructions to follow. Steps travel as index + action + state
  before/after, untruncated; if the finished prompt (system + user) is over
  the size budget (default 24,000 characters) `build_prompt()` refuses with
  `TrajectoryTooLongError` — the model is not called at all, and the column
  becomes ERROR with evidence `trajectory_too_long` (`size`, `limit`).
- **The score is secondary and display-only.** The model returns one extra
  field, `score` (a number 0–1, anchored in the prompt to roughly 0.85–1.0
  for pass, 0.5–0.85 for warn, below 0.5 for fail, and required to agree
  with its verdict). The verdict stays the primary result; the score rides
  the quality result as one evidence item (`quality_score`, `step: null`,
  the number in `value`) — so `GraderResult` and `results.py` are unchanged
  (PASS may carry evidence; only FAIL/WARN are required to). Renderers show
  it beside the verdict (`WARN 0.72 (step 2)`), and nothing extra when
  there is no score (empty trace, ERROR, no judge). It comes from the
  model, so it can vary between runs; there is still no number across
  dimensions.
- **Validation is strict and lives apart from the client.** Not-valid JSON
  (fences are not stripped), wrong keys, an unknown or uppercase verdict, a
  missing or non-numeric or out-of-range `score` (a bool is not a number;
  no clamping), a note citing a step the trajectory does not have, WARN
  with no note, FAIL
  with no *citing* note → `JudgeOutputError` → the column becomes ERROR
  with evidence kind `judge_invalid_output` (the `problem`, plus `bad_step`
  and a short `output_excerpt` when they apply). Nothing is guessed,
  repaired, or retried.
- **Three can't-judge outcomes, and one crash lane, stay tellable apart:**
  an unusable *answer* → ERROR `judge_invalid_output`; an unreachable
  *judge* (network, bad key, rate limit, timeout) → the grader catches
  `LLMClientError` and returns ERROR `judge_unavailable` with the HTTP
  `status` and the capped provider `detail` as evidence (never the key);
  an *over-budget prompt* → ERROR `trajectory_too_long` with `size` and
  `limit`. None of the three is a crash, and none can become a pass. Only
  a genuine bug in judge or grader code still reaches `evaluate()`'s
  crash lane (ERROR `grader_crash`, exception type in the reason).
- **Configuration is environmental:** key `TRAJECTEVAL_JUDGE_API_KEY`
  (required — missing gives a plain message at construction; never in a
  repo file, never in a message), model `TRAJECTEVAL_JUDGE_MODEL` (default
  `openai/gpt-oss-120b`, a Groq-hosted model), base URL
  `TRAJECTEVAL_JUDGE_BASE_URL` (default `https://api.groq.com/openai/v1`),
  timeout `TRAJECTEVAL_JUDGE_TIMEOUT` seconds
  (default 60). Requests use temperature 0 and JSON output mode.
- **Tests:** the default suite drives the seam with `ScriptedJudge` — no
  network, no key, no spend — and never puts `trajectory_quality` in the
  fixture answer key (a real judge is not deterministic).
  `uv run pytest -m live` runs two loose sanity checks against the real API
  (verdicts from the closed set; never a fixed expected verdict).

## The environment and recorder

How a trajectory gets *made* instead of hand-written (Step 11). The rules:

- **The world models physics, not rules.** `TravelBookingEnvironment`
  never opens `tasks/` and never reads the spec: booking first class or
  paying with points works on purpose — the graders are what judge those
  moves. It rejects only what is physically impossible (an unknown action
  name; `pay` before anything is booked; `refund_payment` with nothing to
  undo), and its transitions mirror the seven travel_booking fixtures
  exactly — including `refund_payment` *clearing* `payment_method` to
  `null` (the key stays).
- **The recorder depends only on the `Environment` protocol.** The
  protocol (`envs/base.py`) is a `task_id`, a `current_state()` snapshot,
  a `run_action()` that returns the new state or raises `ActionRejected`,
  and an `actions()` description for agent prompts — the interface a
  second task's world would implement to reuse the recorder unchanged.
- **Every action is recorded — even a rejected one.** A rejected action
  becomes a step whose `state_after` equals its `state_before`, and
  `ActionRejected` is then re-raised: wasted attempts stay visible to the
  graders and the judge, never silently dropped. `ActionRejected` is
  deliberately not a `TrajEvalError` — that lane means "an input file is
  broken", and a rejection is normal gameplay.
- **Recorded states are independent copies.** The environment hands out
  copies and the recorder deep-copies both snapshots per step, so neither
  the world moving on nor a caller mutating a returned state can rewrite
  a recorded step.
- **The fixtures are the ground truth.** The replay test feeds every
  fixture's actions through a fresh environment + recorder and requires
  the recorded states to match the fixture's states exactly and the
  deterministic verdicts to match `expected_verdicts.json` — if the world
  and the fixtures ever disagree, the fixtures win and the test fails.

## The agent loop

How an LLM plays the agent instead of a fixture (Step 12). `run_agent()`
in `src/trajecteval/agent.py`:

- **One prompt per turn, rebuilt from state we hold.** The task
  instruction, the world's own `actions()` description, and the history
  so far (each action with the world's reply) — nothing is remembered
  between calls except that history; the model's own replies carry the
  conversation.
- **Replies are exactly one JSON object**, `{"action": name, "args":
  {...}}` or `{"done": true}`, parsed strictly. An unusable reply is fed
  back with the reason and never recorded as a step; three in a row stop
  the run (`stop_reason: bad_output`).
- **The loop never grades.** It only records through the Recorder —
  rejected actions included — and stops on `done`, the step limit
  (default 12; a looping model cannot run forever), or bad output; the
  reason lands in the trajectory's `stop_reason` metadata. Grading stays
  what `evaluate()` does afterwards, to the same trajectory a fixture
  would produce.
- **The same client, untouched judge.** The loop calls
  `complete(system, user)` on the judge's `OpenAIChatClient` (JSON mode,
  temperature 0); `judge/` was not changed for this. A provider failure
  propagates and the CLI aborts that run with one line, exit 1, saving
  nothing — a crashed API call is not agent behavior. A short pause
  between calls paces the model, and is injectable so tests never sleep.
