"""TrajEval -- evaluate AI agents on *how* they work, not just whether they finished."""

from trajecteval.errors import SpecError, TrajectoryError, TrajEvalError
from trajecteval.graders import (
    BoundsGrader,
    CriticalGrader,
    FinalStateGrader,
    Grader,
    missing_section_result,
    task_id_mismatch_result,
)
from trajecteval.models import Action, Discontinuity, Step, Trajectory
from trajecteval.report import Report, evaluate, render_json
from trajecteval.results import Dimension, Evidence, GraderResult, Verdict
from trajecteval.task_spec import CriticalPattern, RULE_SECTIONS, TaskSpec, load_spec

__all__ = [
    "Action",
    "BoundsGrader",
    "CriticalGrader",
    "CriticalPattern",
    "Dimension",
    "Discontinuity",
    "Evidence",
    "FinalStateGrader",
    "Grader",
    "GraderResult",
    "RULE_SECTIONS",
    "Report",
    "Step",
    "SpecError",
    "TaskSpec",
    "Trajectory",
    "TrajectoryError",
    "TrajEvalError",
    "Verdict",
    "evaluate",
    "load_spec",
    "missing_section_result",
    "render_json",
    "task_id_mismatch_result",
]
