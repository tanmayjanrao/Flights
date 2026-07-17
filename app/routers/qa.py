import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.models.qa_schemas import ChatTranscript, QAAnalyzeResponse, QAHealthResponse
from app.services.qa import ollama_client, qa_service

router = APIRouter(prefix="/api/qa", tags=["qa"])

_SAMPLES_PATH = Path(__file__).resolve().parent.parent / "data" / "qa_sample_transcripts.json"


@router.get("/health", response_model=QAHealthResponse)
async def health():
    return await qa_service.check_health()


@router.get("/samples", response_model=list[ChatTranscript])
async def samples():
    with open(_SAMPLES_PATH) as f:
        raw = json.load(f)
    return [ChatTranscript.model_validate(t) for t in raw]


@router.post("/analyze", response_model=QAAnalyzeResponse)
async def analyze(transcript: ChatTranscript):
    if not transcript.messages:
        raise HTTPException(status_code=400, detail="Transcript has no messages")

    try:
        return await qa_service.analyze_transcript(transcript)
    except ollama_client.OllamaUnavailableError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Ollama isn't reachable - is it running? ({exc})",
        ) from exc
    except ollama_client.OllamaModelNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ollama_client.OllamaGenerationError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
