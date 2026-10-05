"""TrajEval -- evaluate AI agents on *how* they work, not just whether they finished."""

from trajecteval.errors import TrajectoryError
from trajecteval.models import Action, Discontinuity, Step, Trajectory
from trajecteval.results import Evidence, GraderResult, Verdict

__all__ = [
    "Action",
    "Discontinuity",
    "Evidence",
    "GraderResult",
    "Step",
    "Trajectory",
    "TrajectoryError",
    "Verdict",
]
