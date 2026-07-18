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
    elapsed_seconds: float | None = Field(
        default=None,
        description=(
            "Seconds since the start of the chat when this message was sent. "
            "Optional - only used by the deterministic (non-LLM) hold-time and "
            "idle-protocol checks. If any message in a transcript is missing "
            "this, both checks report evaluated=False rather than guessing."
        ),
    )


class ChatTranscript(BaseModel):
    transcript_id: str
    agent_id: str | None = None
    channel: str = "chat"
    messages: list[ChatMessage]


class HoldCheck(BaseModel):
    """One instance of the agent stating a hold/wait duration, checked against
    how long it actually took them to follow up with something substantive.

    This is a SOFT flag - exceeding a stated hold time is common in practice
    (see project notes) and isn't, by itself, a failure worth auto-flagging
    as a problem. `exceeded`/`overage_seconds` are informational.
    """
    agent_message_index: int
    stated_text: str = Field(..., max_length=300)
    stated_seconds: int
    actual_seconds: float
    exceeded: bool
    overage_seconds: float = Field(
        ..., description="actual_seconds - stated_seconds. Zero or negative means within the stated time."
    )


class HoldTimeCompliance(BaseModel):
    evaluated: bool
    holds: list[HoldCheck] = Field(default_factory=list)
    any_exceeded: bool = False
    note: str | None = None


class IdleWindowCheck(BaseModel):
    """One stretch where the passenger went quiet (no messages from them)
    while the agent was still working the ticket. Checked against the
    idle-passenger protocol: a first check-in at ~2 min idle, and - if the
    passenger is still quiet - a final message closing the chat at ~3 min
    idle (rather than the agent going silent or disappearing without
    closing the loop).
    """
    idle_start_index: int = Field(..., description="Index of the last customer message before this idle window")
    idle_duration_seconds: float
    customer_responded: bool = Field(..., description="Whether the customer sent another message before the transcript ends")
    first_checkin_seconds: float | None = None
    first_checkin_on_time: bool | None = None
    final_notice_sent: bool = False
    final_notice_seconds: float | None = None
    final_notice_on_time: bool | None = None
    outcome: str = Field(
        ..., description="customer_responded | closed_after_final_notice | no_final_notice_given"
    )
    violations: list[str] = Field(default_factory=list)


class IdleProtocolCompliance(BaseModel):
    evaluated: bool
    windows: list[IdleWindowCheck] = Field(default_factory=list)
    any_violation: bool = False
    note: str | None = None


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
    """Public response: the LLM output plus everything computed deterministically
    in Python - the overall score, and the hold-time / idle-protocol checks
    (see timing_checks.py; neither of these is asked of the LLM)."""
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
    hold_time_compliance: HoldTimeCompliance
    idle_protocol_compliance: IdleProtocolCompliance


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
