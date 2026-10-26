"""Gemini client. The API key is read from the server environment only."""
import json
import os

import requests

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.0-flash:generateContent"
)
TIMEOUT_SECONDS = 15


class LLMError(Exception):
    pass


def _extract_json(text):
    cleaned = text.replace("```json", "").replace("```", "").strip()
    return json.loads(cleaned)


def get_recommendations(prompt, api_key=None, session=None):
    """Call Gemini and return the parsed JSON decision object.

    Raises LLMError if the key is missing, the request fails, or the reply is not valid JSON.
    """
    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise LLMError("GEMINI_API_KEY is not set")

    http = session or requests
    try:
        resp = http.post(
            GEMINI_URL,
            params={"key": api_key},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.4, "maxOutputTokens": 1000},
            },
            timeout=TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        raise LLMError(f"Gemini request failed: {exc}") from exc

    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        result = _extract_json(text)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise LLMError("Gemini returned an unparseable response") from exc

    if not isinstance(result.get("recommendations"), list):
        raise LLMError("Gemini response is missing 'recommendations'")
    return result
