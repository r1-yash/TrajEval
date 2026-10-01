"""Custom exceptions for TrajEval.

Keeping errors in their own module means every other file can do a single
clean import: ``from trajecteval.errors import TrajectoryError``.
"""


class TrajectoryError(Exception):
    """Raised when a trajectory fails *structural* validation at load time.

    Only structural problems raise this (missing fields, bad action, gaps or
    duplicates in step indices). Data-quality anomalies a loaded trace can
    still be *reported* from -- such as state discontinuities between steps --
    are deliberately not raised here.
    """
