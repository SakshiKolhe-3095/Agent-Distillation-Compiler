import requests

# OLLAMA_URL = "http://localhost:11434/api/generate"
# MODEL = "qwen2.5:7b-instruct-q4_K_M"

import os

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = os.environ.get("TEACHER_MODEL", "qwen2.5:7b-instruct-q4_K_M")

PLANNER_PROMPT = """You are a planning agent. Break the following coding problem
into a short numbered list of concrete implementation steps. Do not write code yet.

Problem:
{problem}
"""

def plan(problem: str) -> str:
    response = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "prompt": PLANNER_PROMPT.format(problem=problem),
        "stream": False
    })
    response.raise_for_status()
    return response.json()["response"]