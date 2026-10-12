# json_equal and the final-state subset matcher: TrajEval's one comparison rule, shared by specs, graders, and models.

from __future__ import annotations

from typing import Any


def json_equal(actual: Any, expected: Any) -> bool:
    # Exact JSON equality: same keys for objects, same kind for scalars (True never matches 1); contains_failures uses it for leaves.
    # bool first: bool is a subclass of int, so ``True`` would otherwise
    # sail through the number branch and match ``1``.
    if isinstance(actual, bool) or isinstance(expected, bool):
        return isinstance(actual, bool) and isinstance(expected, bool) and actual == expected
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return actual == expected
    if isinstance(actual, dict) and isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            json_equal(actual[key], expected[key]) for key in actual
        )
    if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        return len(actual) == len(expected) and all(
            json_equal(a, e) for a, e in zip(actual, expected)
        )
    # Strings, None, and anything else: same exact type, then value equality.
    return type(actual) is type(expected) and actual == expected


def json_contains(actual: Any, expected: Any) -> bool:
    # Subset match: every field `expected` names must be present and match; extras ignored, lists exact. Boolean form of contains_failures.
    return not contains_failures(actual, expected)


def contains_failures(
    actual: Any, expected: Any, path: str = ""
) -> list[tuple[str, str, Any, Any]]:
    # Records (path, kind, expected, actual), kind "missing" or "mismatch"; paths are plain '.'-joins (a dotted key looks like nesting); a list fails whole at its own path.
    if not isinstance(expected, dict):
        # Defensive: specs validate expected_final_state as an object, so a
        # non-dict expected only comes from direct calls. Fall back to
        # exact equality -- there is no subsetting to do.
        return [] if json_equal(actual, expected) else [(path, "mismatch", expected, actual)]
    if not isinstance(actual, dict):
        # An object expected where a non-object sits: nothing to walk down,
        # and the kinds differ -- one mismatch at this path.
        return [(path, "mismatch", expected, actual)]
    failures: list[tuple[str, str, Any, Any]] = []
    for key, wanted in expected.items():
        here = f"{path}.{key}" if path else key
        if key not in actual:
            failures.append((here, "missing", wanted, None))
        elif isinstance(wanted, dict) and isinstance(actual[key], dict):
            failures.extend(contains_failures(actual[key], wanted, here))
        elif not json_equal(actual[key], wanted):
            failures.append((here, "mismatch", wanted, actual[key]))
    return failures
