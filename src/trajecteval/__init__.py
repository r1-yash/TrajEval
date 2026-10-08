"""TrajEval -- evaluate AI agents on *how* they work, not just whether they finished."""

from trajecteval.errors import SpecError, TrajectoryError, TrajEvalError
from trajecteval.graders import (
    FinalStateGrader,
    Grader,
    missing_section_result,
    task_id_mismatch_result,
)
from trajecteval.models import Action, Discontinuity, Step, Trajectory
from trajecteval.results import Dimension, Evidence, GraderResult, Verdict
from trajecteval.task_spec import CriticalPattern, RULE_SECTIONS, TaskSpec, load_spec

__all__ = [
    "Action",
    "CriticalPattern",
    "Dimension",
    "Discontinuity",
    "Evidence",
    "FinalStateGrader",
    "Grader",
    "GraderResult",
    "RULE_SECTIONS",
    "Step",
    "SpecError",
    "TaskSpec",
    "Trajectory",
    "TrajectoryError",
    "TrajEvalError",
    "Verdict",
    "load_spec",
    "missing_section_result",
    "task_id_mismatch_result",
]
