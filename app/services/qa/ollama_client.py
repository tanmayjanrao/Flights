"""
Thin async client around Ollama's /api/chat endpoint.

This module exists to isolate every CPU-inference bottleneck decision in one
place, since running qwen3:4b with no GPU is the whole constraint this tool
is designed around:

1. THINKING MODE - qwen3 will generate a long internal reasoning block
   before answering unless told not to. On CPU that reasoning is often
   slower than the actual answer. We turn it off two ways for redundancy:
   the native `think: false` request field (supported by modern Ollama for
   thinking-capable models), and the `/no_think` chat-template switch
   appended to the prompt itself, in case the installed Ollama version is
   older and ignores the `think` field. We also strip any stray
   <think>...</think> block from the response defensively.
2. STRUCTURED OUTPUT - `format` is set to the target JSON schema, so Ollama
   constrains decoding to that schema rather than letting the model free-
   write prose and hoping it's parseable JSON. This cuts down both wasted
   tokens and parsing failures.
3. TOKEN BUDGET - `num_predict` is capped. The answer is a small JSON
   object, not an essay, so there's no reason to let generation run long
   on a slow CPU.
4. TIMEOUT/RETRY - CPU generation can legitimately take 30-150+ seconds
   depending on hardware. We use a long httpx timeout and do NOT retry on
   timeout (that just doubles the wait) - we only retry once on a
   connection error (i.e. Ollama isn't running yet).
"""
import json
import re

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from app.config import settings

_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class OllamaUnavailableError(Exception):
    """Ollama isn't reachable at all (not running / wrong URL)."""


class OllamaModelNotFoundError(Exception):
    """Ollama is reachable but the configured model tag isn't pulled."""


class OllamaGenerationError(Exception):
    """Ollama responded but generation failed or timed out."""


async def list_models(client: httpx.AsyncClient | None = None) -> list[str]:
    owns_client = client is None
    client = client or httpx.AsyncClient(base_url=settings.ollama_base_url, timeout=10)
    try:
        resp = await client.get("/api/tags")
        resp.raise_for_status()
        data = resp.json()
        return [m["name"] for m in data.get("models", [])]
    except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
        raise OllamaUnavailableError(f"Could not reach Ollama at {settings.ollama_base_url}") from exc
    finally:
        if owns_client:
            await client.aclose()


def _strip_thinking(content: str) -> str:
    return _THINK_TAG_RE.sub("", content).strip()


@retry(
    retry=retry_if_exception_type(OllamaUnavailableError),
    stop=stop_after_attempt(2),
    wait=wait_fixed(1.5),
    reraise=True,
)
async def chat_json(
    system_prompt: str,
    user_prompt: str,
    json_schema: dict,
) -> dict:
    """
    Call Ollama chat with structured-output constraints and return the
    parsed JSON dict. Raises OllamaUnavailableError / OllamaGenerationError
    on failure - callers map these to HTTP errors.
    """
    if settings.qa_disable_thinking:
        # Belt-and-suspenders: native field first, template switch as fallback.
        user_prompt = f"{user_prompt}\n/no_think"

    payload = {
        "model": settings.qa_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "format": json_schema,
        "think": not settings.qa_disable_thinking,
        "options": {
            "temperature": settings.qa_temperature,
            "num_predict": settings.qa_num_predict,
            "num_ctx": settings.qa_num_ctx,
        },
    }

    timeout = httpx.Timeout(settings.qa_timeout_seconds, connect=5.0)
    async with httpx.AsyncClient(base_url=settings.ollama_base_url, timeout=timeout) as client:
        try:
            resp = await client.post("/api/chat", json=payload)
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise OllamaUnavailableError(f"Could not reach Ollama at {settings.ollama_base_url}") from exc
        except httpx.ReadTimeout as exc:
            raise OllamaGenerationError(
                f"Generation exceeded {settings.qa_timeout_seconds}s - this is expected "
                f"sometimes on CPU-only inference; consider raising qa_timeout_seconds "
                f"or qa_num_predict if it happens often."
            ) from exc

        if resp.status_code == 404:
            raise OllamaModelNotFoundError(
                f"Model '{settings.qa_model}' not found in Ollama - run "
                f"`ollama pull {settings.qa_model}` first."
            )
        resp.raise_for_status()
        data = resp.json()

    content = data.get("message", {}).get("content", "")
    content = _strip_thinking(content)

    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise OllamaGenerationError(
            f"Model did not return valid JSON even under schema constraint: {content[:300]!r}"
        ) from exc
