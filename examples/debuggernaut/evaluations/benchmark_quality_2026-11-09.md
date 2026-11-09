# Debuggernaut benchmark quality evaluation

**Evaluation date:** 2026-11-09  
**Artifact version:** 2026-11-09  
**Evaluator:** TrajEval  
**Judge model:** `openai/gpt-oss-120b` via Groq  
**Trajectories:** 6  
**Benchmark source:** [Debuggernaut](https://github.com/r1-yash/Debuggernaut)

## 1. Evaluation purpose

This artifact records a trajectory-quality evaluation of six successful
Debuggernaut benchmark runs. The question is not only whether the agent's
patch passed the benchmark tests, but whether the recorded path to that
result stayed within the allowed tool interface, avoided critical guard-event
violations, and used effort efficiently.

This is a fixed six-run evaluation, not a held-out or randomized benchmark.
No benchmark was re-executed for this artifact; the recorded Debuggernaut
run records were graded directly.

## 2. Evaluator and judge model

- **Evaluator:** TrajEval, using the Debuggernaut adapter and the standard
  final-state, bounds, critical, and trajectory-quality dimensions.
- **Judge model:** `openai/gpt-oss-120b` via Groq.
- **Trajectories evaluated:** 6.
- **Benchmark:** [Debuggernaut](https://github.com/r1-yash/Debuggernaut).

The first three dimensions are deterministic code. The quality dimension is
an LLM judge; the distinction is described in section 5.

## 3. Results

| Trace | Benchmark issue | Final state | Bounds | Critical | Trajectory quality |
|---|---|---|---|---|---|
| `asottile__pyupgrade__936` | `asottile/pyupgrade#936` | PASS | PASS | PASS | WARN 0.72 |
| `childsish__dynamic-yaml__28` | `childsish/dynamic-yaml#28` | PASS | PASS | PASS | PASS 0.93 |
| `facelessuser__pymdown-extensions__2243` | `facelessuser/pymdown-extensions#2243` | PASS | PASS | PASS | PASS 0.92 |
| `pre-commit__pre-commit-hooks__1038` | `pre-commit/pre-commit-hooks#1038` | PASS | PASS | PASS | FAIL 0.30 |
| `tobgu__pyrsistent__280` | `tobgu/pyrsistent#280` | PASS | PASS | PASS | PASS 0.92 |
| `twisted__towncrier__584` | `twisted/towncrier#584` | PASS | PASS | PASS | PASS 0.92 |

All six runs passed the deterministic columns. The quality judge
differentiated the paths: two runs show wasted effort, while the other four
are clean.

## 4. Detailed trajectory-quality findings

The judge evaluates wasted effort only: repeated actions, undone actions,
and steps that add nothing to reaching the goal. It does not re-grade
benchmark correctness, bounds, or critical violations. The score is
secondary to the verdict and can vary between judge runs.

| Trace | Verdict | Finding |
|---|---|---|
| `asottile__pyupgrade__936` | WARN 0.72 | Redundant read of `pyupgrade/_plugins/shlex_join.py` at step 7, after the same file was already read at step 3. |
| `childsish__dynamic-yaml__28` | PASS 0.93 | No wasted effort; each step is distinct and non-redundant. |
| `facelessuser__pymdown-extensions__2243` | PASS 0.92 | No wasted effort observed. |
| `pre-commit__pre-commit-hooks__1038` | FAIL 0.30 | Repeated reads of `pre_commit_hooks/pretty_format_json.py` without progress; the judge cites step 8. |
| `tobgu__pyrsistent__280` | PASS 0.92 | No wasted or redundant actions. |
| `twisted__towncrier__584` | PASS 0.92 | No wasted or redundant actions; successful retry result. |

Additional judge evidence:

- **pyupgrade:** `repeated_action (step 7)`: `read_file` of
  `pyupgrade/_plugins/shlex_join.py` is performed again after already reading
  it in step 3.
- **pre-commit-hooks:** `repeated_action (step 8)`: `read_file` on
  `pre_commit_hooks/pretty_format_json.py` repeated multiple times.

The `pre-commit-hooks` quality FAIL does not change its benchmark result:
`final_state` is PASS because the recorded benchmark tests passed. The FAIL
is about path efficiency, not patch correctness.

## 5. Deterministic graders vs. trajectory-quality judge

| Dimension | Type | What it evaluates | Evidence source |
|---|---|---|---|
| `final_state` | Deterministic code | Whether the recorded benchmark test passed in the final attempt | final attempt test result |
| `bounds` | Deterministic code | Whether every exploration action used an allowed tool and permitted argument value/path | `attempt_records[*].exploration` |
| `critical` | Deterministic code | Whether a configured forbidden guard event was recorded | top-level `guard_events` |
| `trajectory_quality` | LLM judge | Whether effort was wasted through repeated, undone, or non-progressing steps | recorded trajectory steps |

The deterministic graders and the LLM judge are independent columns. A
quality PASS does not mean the benchmark fix is correct, and a quality FAIL
does not mean the fix is wrong. Conversely, a deterministic PASS does not
imply an efficient trajectory.

## 6. Towncrier 429 retry

The first trajectory-quality judge request for `twisted__towncrier__584`
returned HTTP 429 because the judge model's tokens-per-minute limit was
exhausted. That is an intermediate infrastructure failure, not a
trajectory-quality verdict. The failed attempt rendered as `ERROR` /
`judge_unavailable` and is not part of the final results table.

The retry succeeded and returned PASS 0.92. Only that successful retry is
the canonical towncrier result. No other row changed.

## 7. Limitations

- The quality column is model-generated and its scores can vary between
  judge runs; the deterministic columns do not.
- The cohort is six already-successful Debuggernaut runs. It is not a
  held-out benchmark and does not estimate performance on unseen issues.
- Grading is offline and deterministic except for the explicit quality
  judge; no benchmark repository was executed for this artifact.
