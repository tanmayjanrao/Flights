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
    """One instance of the agent placing the passenger on hold, checked
    against the Hold Time Policy's two independent rules:

    Rule 1 - Hold Duration (what the agent SAYS): the stated hold duration
    must always be exactly 5 minutes. Stating "1 minute", "2 minutes",
    "10 minutes", etc. is a violation of this rule regardless of how long
    the hold actually ends up taking - see `stated_duration_compliant`.

    Rule 2 - Hold Resumption (what the agent DOES): the agent must actually
    return within 5 minutes of announcing the hold. Coming back sooner is
    always fine; only taking longer than the fixed 5-minute benchmark is a
    violation - see `exceeded`. This is judged against `policy_seconds` (the
    fixed company benchmark), never against whatever duration the agent
    happened to state - stating the wrong number does not change how the
    actual-time rule is judged, and vice versa. The two rules are evaluated
    independently and either can fail on its own.
    """
    agent_message_index: int
    stated_text: str = Field(..., max_length=300)
    stated_seconds: int = Field(..., description="What the agent told the passenger, in seconds.")
    stated_duration_compliant: bool = Field(
        ..., description="Rule 1: True iff the agent stated exactly 5 minutes (300s). Any other stated duration is a violation."
    )
    actual_seconds: float
    policy_seconds: int = Field(..., description="Fixed company hold benchmark (300s / 5 min), regardless of what the agent stated.")
    exceeded: bool = Field(..., description="Rule 2: actual_seconds > policy_seconds")
    overage_seconds: float = Field(
        ..., description="actual_seconds - policy_seconds. Zero or negative means within the fixed policy benchmark."
    )
    violations: list[str] = Field(
        default_factory=list,
        description="Which of the two independent Hold Policy rules this hold broke: "
        "'stated_duration_not_5_minutes' (Rule 1) and/or 'resumption_exceeded_policy' (Rule 2). Empty means fully compliant.",
    )


class HoldTimeCompliance(BaseModel):
    evaluated: bool
    holds: list[HoldCheck] = Field(default_factory=list)
    any_exceeded: bool = False
    any_violation: bool = Field(
        default=False,
        description="True if any hold broke either Hold Policy rule (stated-duration or resumption-time). Superset of any_exceeded.",
    )
    note: str | None = None


class IdleWindowCheck(BaseModel):
    """One stretch where the passenger went quiet (no messages from them).
    Checked against the idle-passenger protocol: a first check-in at ~2 min
    idle, and - if the passenger is still quiet - a final message closing the
    chat at ~3 min idle (rather than the agent going silent or disappearing
    without closing the loop).

    Important distinction: if the agent had placed the passenger on hold to
    do their own work (e.g. "give me 5 minutes to check that"), that hold
    time is NOT passenger-idle time - the check-in clock only starts once the
    agent resumes with a real update and is now the one waiting on the
    passenger. A "still there?" ping is for when the *passenger* has gone
    quiet, not for the agent to announce their own return from a hold.
    """
    wait_start_index: int = Field(
        ...,
        description=(
            "Index of the message that starts the check-in clock: the agent's "
            "resumption/update message if this window followed a hold the agent "
            "was working (their own hold-work time doesn't count as passenger-idle "
            "time), otherwise the last customer message before the agent went quiet."
        ),
    )
    idle_duration_seconds: float
    customer_responded: bool = Field(..., description="Whether the customer sent another message before the transcript ends")
    first_checkin_seconds: float | None = None
    first_checkin_on_time: bool | None = None
    final_notice_sent: bool = False
    final_notice_seconds: float | None = None
    final_notice_on_time: bool | None = None
    outcome: str = Field(
        ...,
        description=(
            "customer_responded | closed_after_final_notice | no_final_notice_given | "
            "warning_sent_after_customer_reply (Rule 5 critical violation: the agent sent a "
            "check-in/final warning, or disconnected, after the customer had already replied)"
        ),
    )
    violations: list[str] = Field(default_factory=list)


class IdleProtocolCompliance(BaseModel):
    evaluated: bool
    windows: list[IdleWindowCheck] = Field(default_factory=list)
    any_violation: bool = False
    note: str | None = None


class ChatFlowStage(BaseModel):
    """One stage of the standard agent chat flow (see prompts.py for the
    full definition of what each stage requires). The LLM reports whether
    the agent actually followed that stage in this transcript, plus a short
    note on what was/wasn't done - not a 1-5 score, just followed or not.
    """
    stage: str = Field(
        ...,
        description=(
            "One of: opening_statement | acknowledgement_empathy | "
            "probing_verification | solution_discussion_agreement | "
            "resolution | further_assistance | closing_statement"
        ),
    )
    followed: bool
    note: str = Field(default="", max_length=200)


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
    chat_flow: list[ChatFlowStage] = Field(
        default_factory=list,
        max_length=7,
        description="One entry per chat-flow stage, in order - see ChatFlowStage.",
    )
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
    chat_flow: list[ChatFlowStage]
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
