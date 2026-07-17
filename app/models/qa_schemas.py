"""
Schemas for the Airline Chat QA module.

Mirrors models/schemas.py used by the flight tracker. Regardless of which LLM
provider performs the evaluation (local Ollama today, another provider later),
the rest of the application always works with these normalized models.
"""

from enum import Enum

from pydantic import BaseModel, Field


class IssueCategory(str, Enum):
    SCHEDULE_CHANGE = "schedule_change"
    FORCE_MAJEURE = "force_majeure"
    BAGGAGE = "baggage"
    ANCILLARY = "ancillary"
    REBOOKING = "rebooking"
    CANCELLATION = "cancellation"
    REFUND = "refund"
    IRATE_CUSTOMER = "irate_customer"
    MISSED_CONNECTION = "missed_connection"
    OTHER = "other"


class Speaker(str, Enum):
    AGENT = "agent"
    CUSTOMER = "customer"


class ChatMessage(BaseModel):
    speaker: Speaker
    text: str
    timestamp: str | None = None


class ChatTranscript(BaseModel):
    chat_id: str
    agent_id: str | None = None
    category: IssueCategory = IssueCategory.OTHER
    channel: str = "chat"
    messages: list[ChatMessage]


class DimensionScore(BaseModel):
    score: int = Field(ge=0, le=10)
    comment: str


class DimensionScores(BaseModel):
    empathy_and_tone: DimensionScore
    policy_and_compliance: DimensionScore
    resolution_effectiveness: DimensionScore
    communication_clarity: DimensionScore
    de_escalation: DimensionScore | None = None


class FlaggedQuote(BaseModel):
    speaker: Speaker
    quote: str
    issue: str


class DSATRisk(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class QAAnalysis(BaseModel):
    chat_id: str
    category: IssueCategory

    overall_score: int = Field(
        ge=0,
        le=100,
        description="Overall quality score assigned by the LLM."
    )

    dimensions: DimensionScores

    strengths: list[str] = []

    areas_for_improvement: list[str] = []

    flagged_quotes: list[FlaggedQuote] = []

    compliance_flags: list[str] = []

    dsat_risk: DSATRisk

    better_agent_response: str

    summary: str


class QAAnalysisResponse(BaseModel):
    source: str
    analysis: QAAnalysis


class SampleChatSummary(BaseModel):
    chat_id: str
    category: IssueCategory
    agent_id: str | None = None
    preview: str