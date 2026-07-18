import requests
from agents.coder import strip_code_fences

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:7b-instruct-q4_K_M"

DEBUGGER_PROMPT = """You are a debugging agent. The following code failed its tests.
Fix the code so all tests pass. Return only the corrected Python code, no explanation.

Problem:
{problem}

Code:
{code}

Test error:
{error}
"""

def debug(problem: str, code: str, error: str) -> str:
    response = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "prompt": DEBUGGER_PROMPT.format(problem=problem, code=code, error=error),
        "stream": False
    })
    response.raise_for_status()
    return strip_code_fences(response.json()["response"])