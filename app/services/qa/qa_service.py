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


def _format_transcript(transcript: ChatTranscript) -> str:
    return "\n".join(f"[{m.speaker}] {m.text}" for m in transcript.messages)


def _overall_score(scores) -> int:
    values = [scores.empathy, scores.resolution_accuracy, scores.policy_compliance,
              scores.communication_clarity, scores.efficiency]
    return round((sum(values) / len(values)) * 20)  # 1-5 scale -> 0-100


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
