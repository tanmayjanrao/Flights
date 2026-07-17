"""
QA tool tests. The Ollama call is monkeypatched at the `chat_json` boundary
so these never hit a real (slow, CPU-only) local model - same principle as
the flight provider tests: mock at the abstraction boundary, not deeper.
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.qa import ollama_client

client = TestClient(app)

_FAKE_LLM_OUTPUT = {
    "category": "baggage",
    "secondary_issues": ["ancillary_or_pet"],
    "scores": {
        "empathy": 2,
        "resolution_accuracy": 3,
        "policy_compliance": 3,
        "communication_clarity": 2,
        "efficiency": 3,
    },
    "resolved": True,
    "escalation_needed": False,
    "flags": ["tone_issue"],
    "strengths": ["Got a baggage reference number"],
    "improvements": ["Should proactively give a delivery timeline"],
    "summary": "Bag delay handled but agent was terse and reactive rather than proactive.",
}

_SAMPLE_TRANSCRIPT = {
    "transcript_id": "TEST-001",
    "agent_id": "agent_test",
    "channel": "chat",
    "messages": [
        {"speaker": "customer", "text": "My bag never showed up."},
        {"speaker": "agent", "text": "tag number?"},
        {"speaker": "customer", "text": "1234"},
        {"speaker": "agent", "text": "filed, ref BAG-1"},
    ],
}


def test_samples_endpoint_returns_three_categories():
    resp = client.get("/api/qa/samples")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 3
    ids = {t["transcript_id"] for t in data}
    assert ids == {"SAMPLE-CXL-01", "SAMPLE-RBK-01", "SAMPLE-BAG-01"}


def test_qa_page_serves_html():
    resp = client.get("/qa")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


def test_analyze_rejects_empty_transcript():
    resp = client.post("/api/qa/analyze", json={"transcript_id": "EMPTY", "messages": []})
    assert resp.status_code == 400


def test_analyze_computes_overall_score_deterministically(monkeypatch):
    async def fake_chat_json(system_prompt, user_prompt, json_schema):
        return _FAKE_LLM_OUTPUT

    monkeypatch.setattr(ollama_client, "chat_json", fake_chat_json)

    resp = client.post("/api/qa/analyze", json=_SAMPLE_TRANSCRIPT)
    assert resp.status_code == 200
    data = resp.json()

    # (2 + 3 + 3 + 2 + 3) / 5 * 20 = 52
    assert data["analysis"]["overall_score"] == 52
    assert data["analysis"]["category"] == "baggage"
    assert data["analysis"]["flags"] == ["tone_issue"]
    assert data["model"] == "qwen3:4b"


def test_analyze_returns_503_when_ollama_unreachable(monkeypatch):
    async def fake_chat_json(system_prompt, user_prompt, json_schema):
        raise ollama_client.OllamaUnavailableError("connection refused")

    monkeypatch.setattr(ollama_client, "chat_json", fake_chat_json)

    resp = client.post("/api/qa/analyze", json=_SAMPLE_TRANSCRIPT)
    assert resp.status_code == 503


def test_analyze_returns_502_on_bad_model_output(monkeypatch):
    async def fake_chat_json(system_prompt, user_prompt, json_schema):
        return {"category": "not_a_real_category"}

    monkeypatch.setattr(ollama_client, "chat_json", fake_chat_json)

    resp = client.post("/api/qa/analyze", json=_SAMPLE_TRANSCRIPT)
    assert resp.status_code == 502


def test_health_endpoint_reports_down_when_ollama_unreachable(monkeypatch):
    async def fake_list_models(client=None):
        raise ollama_client.OllamaUnavailableError("connection refused")

    monkeypatch.setattr(ollama_client, "list_models", fake_list_models)

    resp = client.get("/api/qa/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "down"
    assert data["ollama_reachable"] is False
