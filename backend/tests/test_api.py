import pytest
import requests

import app as app_module
import llm
from prompts import build_recommendation_prompt

STATIONS = [
    {"id": "TN-03", "name": "Ather Grid", "location": "Tambaram",
     "utilization": 0.91, "risk": "critical"},
    {"id": "TN-01", "name": "Shell Recharge", "location": "Guduvanchery",
     "utilization": 0.75, "risk": "high"},
    {"id": "TN-05", "name": "Local EV Station", "location": "Urapakkam",
     "utilization": 0.20, "risk": "normal"},
]
PAYLOAD = {"stations": STATIONS, "avgUtil": 0.62, "peakHour": 19}


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")

    def json(self):
        return self._payload


def gemini_reply(text):
    return {"candidates": [{"content": {"parts": [{"text": text}]}}]}


@pytest.fixture
def client():
    return app_module.create_app().test_client()


def test_health(client):
    assert client.get("/api/health").get_json() == {"status": "ok"}


def test_prompt_lists_critical_and_high_stations():
    prompt = build_recommendation_prompt(STATIONS, 19, 0.62, "18:00:00")
    assert "Critical stations (>88%): TN-03(91%)" in prompt
    assert "High-risk stations (72-88%): TN-01(75%)" in prompt
    assert "Peak hour forecast: 19:00" in prompt


def test_gemini_call_uses_env_key_and_parses_fenced_json(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "secret-key")
    captured = {}

    def fake_post(url, params, json, timeout):
        captured["params"] = params
        return FakeResponse(gemini_reply('```json\n{"recommendations": [], "summary": "ok"}\n```'))

    monkeypatch.setattr(llm.requests, "post", fake_post)
    result = llm.get_recommendations("prompt")
    assert captured["params"] == {"key": "secret-key"}
    assert result == {"recommendations": [], "summary": "ok"}


def test_missing_key_is_an_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(llm.LLMError):
        llm.get_recommendations("prompt")


def test_recommendations_endpoint_returns_model_output(client, monkeypatch):
    expected = {"recommendations": [{"action": "x", "priority": "high",
                                     "impact": "y", "stations": ["TN-03"]}],
                "summary": "busy"}
    monkeypatch.setattr(llm, "get_recommendations", lambda prompt: expected)
    resp = client.post("/api/recommendations", json=PAYLOAD)
    assert resp.status_code == 200
    assert resp.get_json() == expected


def test_upstream_failure_returns_502_without_leaking_key(client, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "secret-key")

    def boom(url, params, json, timeout):
        raise requests.ConnectionError("down")

    monkeypatch.setattr(llm.requests, "post", boom)
    resp = client.post("/api/recommendations", json=PAYLOAD)
    assert resp.status_code == 502
    assert "secret-key" not in resp.get_data(as_text=True)


@pytest.mark.parametrize("bad", [
    None,
    {"avgUtil": 0.5},
    {"stations": [], "avgUtil": 0.5},
    {"stations": [{"id": "x", "name": "n", "location": "l", "utilization": 2.0, "risk": "normal"}],
     "avgUtil": 0.5},
    {"stations": [{"id": "x", "name": "n", "location": "l", "utilization": 0.5, "risk": "bogus"}],
     "avgUtil": 0.5},
])
def test_invalid_payload_is_rejected(client, bad):
    resp = client.post("/api/recommendations", json=bad)
    assert resp.status_code == 400
