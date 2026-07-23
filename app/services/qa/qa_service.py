"""
Orchestrates a single chat-transcript QA analysis: builds the prompt, calls
the local Ollama model, validates the structured output, and computes the
overall score deterministically (see qa_schemas.py for why).
"""
import time

from pydantic import ValidationError

from app.config import settings
from app.models.qa_schemas import ChatTranscript, QAAnalysisResult, QAAnalyzeResponse, QALLMOutput
from app.services.qa import ollama_client, timing_checks
from app.services.qa.prompts import SYSTEM_PROMPT, build_user_prompt

_SCHEMA = QALLMOutput.model_json_schema()

# Rule 1 of the Hold Time Policy ("stated duration must be exactly 5 minutes")
# is detected deterministically in timing_checks.py, independently of Rule 2
# (resumption time). But scores.policy_compliance is generated entirely by
# the LLM, whose system prompt never mentions this specific rule - so the
# model has no basis to penalize it, and a stated "3 minutes" can sail
# through with a perfect policy_compliance score even though the hold-time
# check elsewhere correctly flags it as non-compliant. This cap forces that
# already-detected violation to actually move the score, rather than sitting
# unused alongside it. It only ever lowers the score, never raises it.
HOLD_DURATION_VIOLATION_POLICY_SCORE_CAP = 2

# Same gap as above, but for the Idle Chat Handling policy: any detected idle
# violation is a real policy miss the LLM was never told the rule for.
# Rule 5 (continuing to warn/disconnect after the customer already replied)
# is the policy's own "major QA violation" language, so it's capped harder
# than a merely mistimed warning.
IDLE_VIOLATION_POLICY_SCORE_CAP = 2
IDLE_REPLY_VIOLATION_POLICY_SCORE_CAP = 1


def _format_transcript(transcript: ChatTranscript) -> str:
    return "\n".join(f"[{m.speaker}] {m.text}" for m in transcript.messages)


def _overall_score(scores) -> int:
    values = [scores.empathy, scores.resolution_accuracy, scores.policy_compliance,
              scores.communication_clarity, scores.efficiency]
    return round((sum(values) / len(values)) * 20)  # 1-5 scale -> 0-100


def _apply_hold_duration_violations(analysis: QAAnalysisResult) -> None:
    """Make a detected Rule 1 (stated hold duration != 5 min) violation
    actually affect policy_compliance and overall_score, and surface it as an
    explicit flag. Rule 2 (resumption time) is unaffected - it already drives
    `exceeded`/`any_exceeded` independently and is left alone here.
    """
    stated_violations = [
        h for h in analysis.hold_time_compliance.holds
        if "stated_duration_not_5_minutes" in h.violations
    ]
    if not stated_violations:
        return

    analysis.scores.policy_compliance = min(
        analysis.scores.policy_compliance, HOLD_DURATION_VIOLATION_POLICY_SCORE_CAP
    )
    analysis.overall_score = _overall_score(analysis.scores)

    for h in stated_violations:
        stated_minutes = h.stated_seconds // 60
        flag = (
            f"Hold Duration Violation \u2013 Agent stated {stated_minutes} minutes "
            "instead of the required 5 minutes"
        )
        if flag not in analysis.flags:
            analysis.flags.append(flag)


def _apply_idle_protocol_violations(analysis: QAAnalysisResult) -> None:
    """Same gap as _apply_hold_duration_violations, for the Idle Chat
    Handling policy: timing_checks.py already detects mistimed warnings and
    (per Rule 5) warnings/disconnects sent after the customer had already
    replied, but nothing previously made those detections move
    policy_compliance/overall_score or show up as a plain-language flag.
    """
    windows = analysis.idle_protocol_compliance.windows
    reply_violations = [w for w in windows if "warning_sent_after_customer_reply" in w.violations]
    other_violations = [w for w in windows if w.violations and w not in reply_violations]

    if not reply_violations and not other_violations:
        return

    if reply_violations:
        # Rule 5 - the policy's own "major QA violation" language. Continuing
        # to warn/disconnect after the customer replied is the single most
        # serious idle-handling mistake, so it gets the harder cap.
        analysis.scores.policy_compliance = min(
            analysis.scores.policy_compliance, IDLE_REPLY_VIOLATION_POLICY_SCORE_CAP
        )
    else:
        analysis.scores.policy_compliance = min(
            analysis.scores.policy_compliance, IDLE_VIOLATION_POLICY_SCORE_CAP
        )
    analysis.overall_score = _overall_score(analysis.scores)

    for w in reply_violations:
        flag = (
            "Idle Handling Violation \u2013 Agent sent a warning or disconnected the chat "
            "after the customer had already replied"
        )
        if flag not in analysis.flags:
            analysis.flags.append(flag)

    for w in other_violations:
        for v in w.violations:
            if v == "missing_final_notice":
                flag = "Idle Handling Violation \u2013 Chat went silent with no final warning before disconnect"
            elif v in ("checkin_early", "checkin_late"):
                flag = f"Idle Handling Violation \u2013 First warning was not sent at the required 2 minutes ({v.split('_')[1]})"
            elif v in ("final_notice_early", "final_notice_late"):
                flag = f"Idle Handling Violation \u2013 Final warning was not sent at the required 5 minutes total ({v.split('_')[-1]})"
            else:
                continue
            if flag not in analysis.flags:
                analysis.flags.append(flag)


async def analyze_transcript(transcript: ChatTranscript) -> QAAnalyzeResponse:
    user_prompt = build_user_prompt(_format_transcript(transcript))

    started = time.monotonic()
    raw = await ollama_client.chat_json(SYSTEM_PROMPT, user_prompt, _SCHEMA)
    latency_ms = round((time.monotonic() - started) * 1000)

    try:
        llm_output = QALLMOutput.model_validate(raw)
    except ValidationError as exc:
        raise ollama_client.OllamaGenerationError(
            f"Model output didn't match the expected schema: {exc}"
        ) from exc

    analysis = QAAnalysisResult(
        category=llm_output.category,
        secondary_issues=llm_output.secondary_issues,
        scores=llm_output.scores,
        overall_score=_overall_score(llm_output.scores),
        resolved=llm_output.resolved,
        escalation_needed=llm_output.escalation_needed,
        flags=llm_output.flags,
        strengths=llm_output.strengths,
        improvements=llm_output.improvements,
        summary=llm_output.summary,
        # Neither of these goes through the LLM - see timing_checks.py.
        hold_time_compliance=timing_checks.check_hold_time_compliance(transcript),
        idle_protocol_compliance=timing_checks.check_idle_protocol_compliance(transcript),
    )

    # The LLM's policy_compliance score doesn't know about the Hold Duration
    # rule or the Idle Handling rules - see the docstrings above. Correct it
    # here, deterministically, using what timing_checks.py already detected.
    _apply_hold_duration_violations(analysis)
    _apply_idle_protocol_violations(analysis)

    return QAAnalyzeResponse(
        transcript_id=transcript.transcript_id,
        model=settings.qa_model,
        thinking_disabled=settings.qa_disable_thinking,
        latency_ms=latency_ms,
        analysis=analysis,
    )


async def check_health():
    from app.models.qa_schemas import QAHealthResponse

    try:
        models = await ollama_client.list_models()
    except ollama_client.OllamaUnavailableError as exc:
        return QAHealthResponse(
            status="down",
            ollama_reachable=False,
            model=settings.qa_model,
            model_available=False,
            detail=str(exc),
        )

    model_available = any(m == settings.qa_model or m.startswith(f"{settings.qa_model.split(':')[0]}:")
                           for m in models)
    return QAHealthResponse(
        status="ok" if model_available else "model_missing",
        ollama_reachable=True,
        model=settings.qa_model,
        model_available=model_available,
        detail=None if model_available else f"Run `ollama pull {settings.qa_model}`",
    )
