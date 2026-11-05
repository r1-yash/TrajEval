# Debuggernaut adapter fixtures

These are verbatim copies of real Debuggernaut benchmark run records, used
offline by `tests/test_adapters_debuggernaut.py`. They are not hand-written
schemas.

| Fixture | Debuggernaut source |
| --- | --- |
| `success_pre_commit_hooks_1038.json` | `run_logs/runs/20261102T144640Z_pre-commit-pre-commit-hooks_1038.json` |
| `multi_attempt_mashumaro_184.json` | `run_logs/runs/20261102T140840Z_Fatal1ty-mashumaro_184.json` |
| `proposal_failure_sqlparse_763.json` | `run_logs/runs/20261102T140451Z_andialbrecht-sqlparse_763.json` |
| `toolcall_limit_sqlparse_763.json` | `run_logs/runs_attempt1_toolcall_limit/20261026T131109Z_andialbrecht-sqlparse_763.json` |

The first two are from the final schema-version-4 benchmark batch; the last
two cover earlier partially recorded failures. They contain agent-proposed
diffs and benchmark test-patch *paths*, but no gold patches or answer keys.
