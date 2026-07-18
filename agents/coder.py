# import requests

# OLLAMA_URL = "http://localhost:11434/api/generate"
# MODEL = "qwen2.5:7b-instruct-q4_K_M"

# CODER_PROMPT = """You are a coding agent. Given this problem and plan, write a single
# Python function that solves it. Return only code, no explanation.

# Problem:
# {problem}

# Plan:
# {plan}
# """

# def strip_code_fences(text: str) -> str:
#     text = text.strip()
#     if text.startswith("```"):
#         text = text.split("\n", 1)[1] if "\n" in text else text
#         text = text.rsplit("```", 1)[0]
#     return text.strip()

# def code(problem: str, plan: str) -> str:
#     response = requests.post(OLLAMA_URL, json={
#         "model": MODEL,
#         "prompt": CODER_PROMPT.format(problem=problem, plan=plan),
#         "stream": False
#     })
#     response.raise_for_status()
#     return strip_code_fences(response.json()["response"])


import requests

# OLLAMA_URL = "http://localhost:11434/api/generate"
# MODEL = "qwen2.5:7b-instruct-q4_K_M"

import os
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = os.environ.get("TEACHER_MODEL", "qwen2.5:7b-instruct-q4_K_M")

CODER_PROMPT = """You are a coding agent. Given this problem and plan, write a single
Python function that solves it. Return only code, no explanation.

Problem:
{problem}

Plan:
{plan}

The following test code will call your function — your function name and signature
MUST exactly match what this test code expects:
{test_code}
"""

def strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        text = text.rsplit("```", 1)[0]
    return text.strip()

def code(problem: str, plan: str, test_code: str) -> str:
    response = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "prompt": CODER_PROMPT.format(problem=problem, plan=plan, test_code=test_code),
        "stream": False
    })
    response.raise_for_status()
    return strip_code_fences(response.json()["response"])