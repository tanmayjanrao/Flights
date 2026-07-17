"""
Schemas for the agent-chat QA tool.

Two shapes matter here:
- `QALLMOutput` is what we ask the local Qwen model to produce. It is
  deliberately small - no free-form reasoning fields, no arithmetic - so a
  4B model on CPU has as little to generate (and get wrong) as possible.
- `QAAnalysisResult` is what the API actually returns. `overall_score` is
  computed in Python from the sub-scores rather than trusted from the LLM,
  since asking a small model to also do the averaging correctly is an easy
  way to get inconsistent numbers for free.
"""
from enum import Enum

from pydantic import BaseModel, Field


class IssueCategory(str, Enum):
    """
    The full set of ticket categories this tool is meant to eventually cover.
    Few-shot examples currently only exist for CANCELLATION, REBOOKING, and
    BAGGAGE - the rest are here so the schema doesn't need to change when
    more few-shot examples are added later.
    """
    CANCELLATION = "cancellation"
    REBOOKING = "rebooking"
    BAGGAGE = "baggage"
    REFUND = "refund"
    SCHEDULE_CHANGE = "schedule_change"
    FORCE_MAJEURE = "force_majeure"
    ANCILLARY_OR_PET = "ancillary_or_pet"
    IRATE_CUSTOMER = "irate_customer"
    MISSED_CONNECTION = "missed_connection"
    OTHER = "other"


class ChatMessage(BaseModel):
    speaker: str = Field(..., description="'agent' or 'customer'")
    text: str


class ChatTranscript(BaseModel):
    transcript_id: str
    agent_id: str | None = None
    channel: str = "chat"
    messages: list[ChatMessage]


class QAScores(BaseModel):
    """Each dimension is scored 1 (poor) - 5 (excellent) by the LLM."""
    empathy: int = Field(..., ge=1, le=5)
    resolution_accuracy: int = Field(..., ge=1, le=5)
    policy_compliance: int = Field(..., ge=1, le=5)
    communication_clarity: int = Field(..., ge=1, le=5)
    efficiency: int = Field(..., ge=1, le=5)


class QALLMOutput(BaseModel):
    """Exact JSON shape requested from the model via Ollama structured outputs."""
    category: IssueCategory
    secondary_issues: list[str] = Field(default_factory=list, max_length=3)
    scores: QAScores
    resolved: bool
    escalation_needed: bool
    flags: list[str] = Field(default_factory=list, max_length=5)
    strengths: list[str] = Field(default_factory=list, max_length=3)
    improvements: list[str] = Field(default_factory=list, max_length=3)
    summary: str = Field(..., max_length=300)


class QAAnalysisResult(BaseModel):
    """Public response: the LLM output plus a deterministically computed overall score."""
    category: IssueCategory
    secondary_issues: list[str]
    scores: QAScores
    overall_score: int = Field(..., ge=0, le=100)
    resolved: bool
    escalation_needed: bool
    flags: list[str]
    strengths: list[str]
    improvements: list[str]
    summary: str


class QAAnalyzeResponse(BaseModel):
    transcript_id: str
    model: str
    thinking_disabled: bool
    latency_ms: int
    analysis: QAAnalysisResult


class QAHealthResponse(BaseModel):
    status: str
    ollama_reachable: bool
    model: str
    model_available: bool
    detail: str | None = None
