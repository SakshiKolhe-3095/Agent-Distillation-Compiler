"""
Compresses a full multi-agent trajectory (plan + code + retries) into a single
coherent chain-of-thought, using the teacher model to summarize its own trace.
"""
import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:7b-instruct-q4_K_M"

COMPRESS_PROMPT = """You are summarizing a multi-step reasoning process into a single,
coherent chain-of-thought that a student model can learn from in ONE pass — no
mention of retries, agents, or the multi-step process. Write as if you solved
it correctly the first time.

Problem:
{problem}

Original plan:
{plan}

Final working code:
{code}

Write a single, natural reasoning explanation (a few sentences) followed by the
final code. Format:

Reasoning: <your compressed reasoning>
Code:
{{code block}}
"""

def compress(problem: str, plan: str, code: str) -> str:
    response = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "prompt": COMPRESS_PROMPT.format(problem=problem, plan=plan, code=code),
        "stream": False
    })
    response.raise_for_status()
    return response.json()["response"]