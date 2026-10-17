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
