"""Custom exceptions for TrajEval.

Two lanes, kept apart on purpose:

- ``TrajectoryError`` / ``SpecError`` / ``ReportError`` / ``AdapterError`` --
  *the input files are broken* (structural problems found while loading a
  trajectory, a task spec, a saved report, or an already-loaded external run
  record). All subclass ``TrajEvalError`` so a caller can catch "some file I
  was given is malformed" with one ``except``.
- ``ValueError`` -- *the grader code is broken* (a GraderResult built with
  a blank reason, an unknown verdict, ...). Results never raise file errors,
  and file loaders never raise ValueError for data shape.
"""

from __future__ import annotations


class TrajEvalError(Exception):
    """Base class for 'an input file is structurally broken' errors."""


class TrajectoryError(TrajEvalError):
    """Raised when a trajectory fails *structural* validation at load time.

    Only structural problems raise this (missing fields, bad action, gaps or
    duplicates in step indices). Data-quality anomalies a loaded trace can
    still be *reported* from -- such as state discontinuities between steps --
    are deliberately not raised here.
    """


class SpecError(TrajEvalError):
    """Raised when a task-spec file fails *structural* validation at load time.

    Same job as ``TrajectoryError``, for task specs: unreadable file, invalid
    JSON, or a structure that does not match the pinned schema (wrong types,
    missing ``task_id``, unknown keys -- a rule file is hand-written, so an
    unknown key is a typo we reject at load rather than a rule we silently
    ignore).
    """


class ReportError(TrajEvalError):
    """Raised when a saved report file fails *structural* validation at load time."""

    # Same job as the other two, for the JSON form of a Report: a broken
    # results/discontinuities entry inside a saved report surfaces here too
    # (Report.from_dict wraps the nested ValueError/TrajectoryError).


class AdapterError(TrajEvalError):
    """Raised when an external run record cannot be translated into a trajectory.

    The record has already been loaded (this is the offline boundary), so an
    AdapterError means its structure does not match what the format-specific
    adapter promised -- not that the source agent did something wrong. The
    adapter never grades and never interprets the run.
    """
