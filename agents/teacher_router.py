"""
Unified teacher router: tries local Ollama first, falls back to
Groq/Gemini API teacher (agents/api_teacher.py) if Ollama fails
after retries. Used by planner, coder, and debugger agents.
"""

import os
import time
import requests
from agents.api_teacher import APITeacher

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = os.environ.get("TEACHER_MODEL", "qwen2.5:7b-instruct-q4_K_M")

_api_teacher = None


def _get_api_teacher() -> APITeacher:
    global _api_teacher
    if _api_teacher is None:
        _api_teacher = APITeacher()
    return _api_teacher


def call_teacher(prompt: str, max_ollama_retries: int = 3) -> str:
    """
    Tries local Ollama model first (with retries).
    Falls back to Groq/Gemini API teacher if Ollama fails entirely.
    Set SKIP_OLLAMA=1 in the environment to bypass Ollama entirely and go
    straight to the API teacher (useful for API-only trajectory runs).
    Returns the raw text response.
    """
    if os.environ.get("SKIP_OLLAMA"):
        return _get_api_teacher().generate(prompt)

    last_err = None
    for attempt in range(max_ollama_retries):
        try:
            response = requests.post(OLLAMA_URL, json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False
            }, timeout=60)
            response.raise_for_status()
            return response.json()["response"]
        except requests.exceptions.RequestException as e:
            last_err = e
            if attempt < max_ollama_retries - 1:
                time.sleep(5)

    print(f"[teacher_router] Ollama failed after {max_ollama_retries} attempts "
          f"({last_err}), falling back to API teacher (Groq/Gemini)...")
    return _get_api_teacher().generate(prompt)