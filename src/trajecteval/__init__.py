"""TrajEval -- evaluate AI agents on *how* they work, not just whether they finished."""

from trajecteval.errors import SpecError, TrajectoryError, TrajEvalError
from trajecteval.graders import Grader, missing_section_result
from trajecteval.models import Action, Discontinuity, Step, Trajectory
from trajecteval.results import Dimension, Evidence, GraderResult, Verdict
from trajecteval.task_spec import CriticalPattern, TaskSpec, load_spec

__all__ = [
    "Action",
    "CriticalPattern",
    "Dimension",
    "Discontinuity",
    "Evidence",
    "Grader",
    "GraderResult",
    "Step",
    "SpecError",
    "TaskSpec",
    "Trajectory",
    "TrajectoryError",
    "TrajEvalError",
    "Verdict",
    "load_spec",
    "missing_section_result",
]
