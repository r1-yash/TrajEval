# The judge seam: the contract, strict output validation, the prompt
# builder, the quality-grader adapter, one real client (Grok over
# OpenAI-compatible HTTP), and a scripted fake for tests that must not
# spend API money.

from trajecteval.judge.base import Judge, JudgeNote, JudgeResult
from trajecteval.judge.fake import ScriptedJudge
from trajecteval.judge.llm import GrokChatClient, LLMClientError, LLMJudge
from trajecteval.judge.quality import TrajectoryQualityGrader

__all__ = [
    "GrokChatClient",
    "Judge",
    "JudgeNote",
    "JudgeResult",
    "LLMClientError",
    "LLMJudge",
    "ScriptedJudge",
    "TrajectoryQualityGrader",
]
