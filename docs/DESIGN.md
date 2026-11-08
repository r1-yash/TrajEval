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
| ERROR | The grader **could not judge**: a missing spec section, a wrong spec/trajectory pairing, missing or malformed recorded evidence, or a crash inside the grader. | The setup is at fault, not the run. |

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
- FAIL and WARN must cite evidence; step numbers locate findings that belong
  to a step. Findings about recorded run-level evidence (for example a final
  benchmark test result) use `step: null`.
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
- A permit value is either a **list of exact permitted values** or a
  **declarative constraint object** such as
  `{"constraint": "repository_relative_path"}`. Both kinds are applied:
  exact values match by JSON equality; constraints use the predicate the
  bounds grader owns.
- Permits police only arguments the agent *supplied*: an argument absent from
  the permit map is unconstrained; a listed-but-absent argument is never a
  violation — permits cannot *require* an argument (completeness is
  `expected_final_state`'s job); an empty permit list permits nothing.
  Declarative constraints follow the same supplied-only rule.
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

## Offline external-run adapters

The translation boundary for agent runs captured outside TrajEval
(`src/trajecteval/adapters/offline.py`). The rules:

- **Input is already-loaded data.** `OfflineRunAdapter.to_trajectory(record)`
  takes a mapping that the caller already read; the adapter performs no
  HTTP, model, agent, or external-repository access and never executes the
  source agent.
- **One trajectory model, one validator.** Format-specific subclasses
  implement `extract_task_id`, `extract_steps`, and (optionally)
  `extract_metadata`. They return `ExternalStep` values; the base class
  assigns canonical gapless 1-based indices in sequence order and builds the
  result through `Trajectory.from_dict`. External numbering is not trusted,
  and the adapter creates no second trajectory type.
- **Translation, not evaluation.** No adapter reads a `TaskSpec`, decides
  PASS/FAIL, or interprets whether an action was good. Those stay in
  `evaluate()` and its graders.
- **Unavailable state is explicit.** `ExternalStep.state_before` /
  `state_after` are `None` when the source recorded no snapshot. Since the
  existing `Step` requires dicts, the converted trajectory uses a reserved
  one-key marker, `{"__trajecteval_unavailable_state__": "unavailable"}`.
  `{}` remains a genuinely empty recorded state. A source state containing
  the reserved key is rejected rather than confused with "unavailable".
  Callers can test the marker with `is_unavailable_state()`.

### Debuggernaut adapter

`DebuggernautAdapter` (`src/trajecteval/adapters/debuggernaut.py`) targets the
real schema-version-4 records written by Debuggernaut's benchmark runner:

- **Only `attempt_records[*].exploration` becomes steps.** Each structured
  entry (`tool`, `arguments`, `ok`, `result_size`, `result_unit`, `error`)
  is an actual tool action, preserved in source order. `calls[]` is provider
  accounting and is never converted into a tool action.
- **Attempt boundaries live in metadata.** `metadata["attempt_records"]`
  keeps each source attempt's fields and adds `step_indices`; its
  `exploration` entries gain the matching `step_index`. This distinguishes
  attempt 1 exploration/proposal/test/diagnosis from attempt 2 without
  inventing proposal or test steps.
- **Outcomes are evidence, not actions.** Test results, diagnoses,
  proposal/reasoning/diff data, proposal validation attempts, artifacts,
  `guard_events`, and `proposal_failures` stay at trajectory level under the
  source field names. Provider calls, token counts, timing, repository,
  issue, commit, test-patch paths, baseline, and environment metadata are
  preserved there too.
- **Missing state stays missing.** Debuggernaut records tool metadata, not
  repository snapshots, so every step uses the explicit unavailable-state
  marker. Nothing is reconstructed from tool names, results, diffs, or test
  output.
- **Incomplete historical records still load.** Records with
  `attempt_records: []`, no `exploration_rounds`, or proposal-schema failures
  convert successfully with zero steps and full available metadata; absent
  exploration fields are never fabricated from `calls[]`.

### Debuggernaut final-state grading

`DebuggernautFinalStateGrader`
(`src/trajecteval/graders/debuggernaut_final_state.py`) judges the
`final_state` dimension for a converted Debuggernaut trajectory.

- **The recorded benchmark test is the final state.** Debuggernaut does not
  snapshot repository files around tool calls, so `Step.state_before/after`
  are intentionally unavailable. The last attempt's recorded `test` block is
  the source of truth; no state is reconstructed and pytest is never run.
- **PASS** when the final attempt's test has `started: true`,
  `timed_out: false`, `passed: true`, and a consistent return code.
  **FAIL** when the same execution record has `passed: false`. The FAIL
  evidence names the final attempt and carries the test summary, not a
  fabricated step.
- **ERROR never becomes FAIL.** No `attempt_records`, no final test record,
  missing/non-boolean flags, `started: false`, `timed_out: true`, malformed
  fields, or contradictory `passed`/`return_code` values all produce ERROR.
  A provider or infrastructure failure with no reliable final test result is
  therefore ERROR, not an accusation that the agent's fix was wrong. A
  reliable final test result still decides the benchmark outcome even if the
  run later stopped for an unrelated reason.
- **Only the final attempt is judged.** Earlier attempt results remain
  historical evidence; a later attempt without a test does not inherit an
  earlier verdict.
- **Spec integration.** A task uses the new optional
  `expected_final_outcome: "benchmark_tests_pass"` section instead of the
  dictionary-based `expected_final_state`. The two sections are mutually
  exclusive. When this section is present, `evaluate()` uses
  `DebuggernautFinalStateGrader` for the `final_state` column. If the spec
  also supplies `allowed_actions`, the shared `BoundsGrader` adds the bounds
  column; if it supplies `critical_guard_events`,
  `DebuggernautCriticalGrader` adds the critical column. Without those
  sections, the earlier behavior is kept (final-state only, or final-state
  plus bounds). Existing state-based tasks keep `FinalStateGrader`,
  `BoundsGrader`, and `CriticalGrader` unchanged. This differs from travel
  booking, where the expected final state is matched against the last step's
  recorded state: Debuggernaut's final state is an externally recorded test
  outcome.

### Debuggernaut bounds grading

The `bounds` column asks whether every recorded exploration action stayed
within the task's allowed interface. It is independent of final success: a
later passing test does not erase an earlier violation.

- **Only trajectory steps count.** `DebuggernautAdapter` builds steps solely
  from `attempt_records[*].exploration`. `calls[]` (provider/LLM accounting),
  token counts, attempt metadata, diagnosis, and run status never become
  actions.
- **Allowed action policy.** The real fixer exposes exactly three exploration
  tools: `read_file`, `list_directory`, and `search_code` (the
  `ExplorationStep.tool` literal and the functions registered in
  `propose_fix`). The default Debuggernaut spec lists all three; any other
  recorded action name is `unlisted_action` → FAIL.
- **Argument constraints.** `read_file.path` and `list_directory.dir_path`
  use `{"constraint": "repository_relative_path"}`. The check is lexical:
  absolute paths fail, and a path whose POSIX normalization is `..` or starts
  with `../` fails as traversal. A path that normalizes back inside the
  repository (for example `pkg/../README.md`) passes; an empty path or `.`
  normalizes to the root and stays in scope. Symlink resolution, file
  existence, and the repository checkout are never inspected; only the
  recorded string is evaluated. `search_code.query` is free-form and
  unconstrained.
- **Supplied arguments only.** As with value permits, a constraint is checked
  only when the argument is present; it does not require the argument.
- **Every step is inspected.** Violations are collected in step order, one
  evidence item each: `unlisted_action`, `argument_not_permitted`,
  `argument_constraint_violated`, or `malformed_argument`. Evidence carries
  the exact step index, action name, relevant arguments, and — for constraint
  violations — the constraint and a human-readable reason.
- **ERROR is not FAIL.** A non-string path (or a NUL byte) is malformed
  action data → ERROR. A missing `allowed_actions` section is a setup
  failure → ERROR. An empty section or a trajectory with no steps warns that
  nothing was checked. Neither the overall `status` nor `failure_class`
  influences bounds.
- **Spec shape.** A Debuggernaut task combines the benchmark outcome with the
  exploration policy:

```json
{
  "task_id": "pre-commit__pre-commit-hooks__1038",
  "expected_final_outcome": "benchmark_tests_pass",
  "allowed_actions": {
    "read_file": { "path": { "constraint": "repository_relative_path" } },
    "list_directory": {
      "dir_path": { "constraint": "repository_relative_path" }
    },
    "search_code": {}
  }
}
```

  A run whose benchmark test passes but whose step 1 calls an unlisted tool,
  or whose step 3 reads `../outside.py`, therefore reports
  `final_state PASS` and `bounds FAIL`. The same `BoundsGrader` still applies
  travel booking's exact-value permits unchanged; only specs that opt into a
  constraint descriptor use the lexical path policy.

### Debuggernaut critical grading

The `critical` column asks whether the run contains a specifically forbidden
behavior that must remain a failure even if the agent later recovered and the
benchmark test passed. Bounds asks whether one action fell outside the allowed
interface; critical is a sticky red-line violation, not a per-action permit
check.

- **Evidence source: `guard_events`.** Debuggernaut's fixer records a
  `GuardEvent` when its guardrail rejects a proposed fix before any repository
  file is modified. The run record's top-level `guard_events` list is
  preserved in trajectory metadata by the adapter and is never converted into
  a step. Each event carries `attempt`, `file_path`, `kind`, and `reason`.
- **Real kinds and default policy.** The closed vocabulary is `test_file`,
  `test_patch_file`, and `nonexistent_path`. `test_file` and `test_patch_file`
  are proposals to modify protected test content, which Debuggernaut's
  instructions and guardrail explicitly prohibit; the default Debuggernaut
  policy treats them as critical. `nonexistent_path` is a retryable
  path-validation miss, not a forbidden modification, so it is not critical
  by default; a spec may add it explicitly.
- **Sticky scan.** Every recorded guard event is inspected, in order. A later
  successful attempt, recovered change, or passing benchmark test never
  erases an earlier critical event. Multiple events produce multiple evidence
  items.
- **Evidence.** Critical evidence uses `step: null` because a rejected
  proposal has no trajectory step index. `field_name` names the guard event
  (`guard_events[0]`), and the value carries the 1-based event number,
  attempt, kind, file path, and reason.
- **Not evidence.** Overall `status`, `failure_class`, provider or
  infrastructure failures, timeouts, benchmark test failure, a missing test
  result, and proposal-schema validation failures are not critical
  violations. A benchmark failure with no guard event is `critical PASS`.
- **Existing step patterns stay separate.** `critical_error_patterns` remains
  the action/argument section used by `CriticalGrader` for state-based tasks;
  it is not reinterpreted as guard-event configuration. Debuggernaut critical
  evidence is expressed through `critical_guard_events`.
- **Verdicts.** No configured critical guard event → PASS. At least one →
  FAIL. A malformed `guard_events` container, event, or unknown kind (or a
  critical event without a path/reason) → ERROR. An empty
  `critical_guard_events` section warns that nothing was checked; absent
  `guard_events` metadata also warns (absence cannot establish a violation,
  but it cannot earn a clean PASS). The section itself being absent means the
  dimension is not evaluated for that benchmark spec.
- **Spec shape.** Add the section to a benchmark-outcome task:

```json
{
  "task_id": "pre-commit__pre-commit-hooks__1038",
  "expected_final_outcome": "benchmark_tests_pass",
  "allowed_actions": {
    "read_file": { "path": { "constraint": "repository_relative_path" } },
    "list_directory": {
      "dir_path": { "constraint": "repository_relative_path" }
    },
    "search_code": {}
  },
  "critical_guard_events": ["test_file", "test_patch_file"]
}
```

  `evaluate()` still produces independent columns. A run whose benchmark test
  passes and whose exploration stays in bounds can still report
  `final_state PASS`, `bounds PASS`, `critical FAIL`.
- **Limitation.** Guard events record the rejection kind, target path, and
  reason, not the full rejected proposal body. TrajEval does not read proposal
  artifacts or diffs, so it cannot detect a protected-file modification that
  somehow bypassed the guardrail. Exploration tool failures (for example a
  path-traversal read attempt) are bounds evidence, not critical guard
  events.
