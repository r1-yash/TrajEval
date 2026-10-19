"""TrajEval -- evaluate AI agents on *how* they work, not just whether they finished."""

from trajecteval.compare import Comparison, compare, render_table, render_table_markdown
from trajecteval.envs import ActionRejected, Environment, TravelBookingEnvironment
from trajecteval.errors import ReportError, SpecError, TrajectoryError, TrajEvalError
from trajecteval.graders import (
    BoundsGrader,
    CriticalGrader,
    FinalStateGrader,
    Grader,
    missing_section_result,
    task_id_mismatch_result,
)
from trajecteval.judge import (
    Judge,
    JudgeNote,
    JudgeResult,
    LLMClientError,
    LLMJudge,
    OpenAIChatClient,
    ScriptedJudge,
    TrajectoryQualityGrader,
    TrajectoryTooLongError,
)
from trajecteval.models import Action, Discontinuity, Step, Trajectory
from trajecteval.recorder import Recorder
from trajecteval.report import Report, evaluate, render_json, render_text
from trajecteval.results import Dimension, Evidence, GraderResult, Verdict
from trajecteval.task_spec import CriticalPattern, RULE_SECTIONS, TaskSpec, load_spec

__all__ = [
    "Action",
    "ActionRejected",
    "BoundsGrader",
    "Comparison",
    "CriticalGrader",
    "CriticalPattern",
    "Dimension",
    "Discontinuity",
    "Environment",
    "Evidence",
    "FinalStateGrader",
    "Grader",
    "GraderResult",
    "Judge",
    "JudgeNote",
    "JudgeResult",
    "LLMClientError",
    "LLMJudge",
    "OpenAIChatClient",
    "RULE_SECTIONS",
    "Recorder",
    "Report",
    "ReportError",
    "ScriptedJudge",
    "Step",
    "SpecError",
    "TaskSpec",
    "Trajectory",
    "TrajectoryError",
    "TrajectoryQualityGrader",
    "TrajectoryTooLongError",
    "TrajEvalError",
    "TravelBookingEnvironment",
    "Verdict",
    "compare",
    "evaluate",
    "load_spec",
    "missing_section_result",
    "render_json",
    "render_table",
    "render_table_markdown",
    "render_text",
    "task_id_mismatch_result",
]
