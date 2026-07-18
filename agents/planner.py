# import requests

# # OLLAMA_URL = "http://localhost:11434/api/generate"
# # MODEL = "qwen2.5:7b-instruct-q4_K_M"

# import os

# OLLAMA_URL = "http://localhost:11434/api/generate"
# MODEL = os.environ.get("TEACHER_MODEL", "qwen2.5:7b-instruct-q4_K_M")

# PLANNER_PROMPT = """You are a planning agent. Break the following coding problem
# into a short numbered list of concrete implementation steps. Do not write code yet.

# Problem:
# {problem}
# """

# import time

# def plan(problem: str) -> str:
#     for attempt in range(3):
#         try:
#             response = requests.post(OLLAMA_URL, json={
#                 "model": MODEL,
#                 "prompt": PLANNER_PROMPT.format(problem=problem),
#                 "stream": False
#             })
#             response.raise_for_status()
#             return response.json()["response"]
#         except requests.exceptions.HTTPError:
#             if attempt == 2:
#                 raise
#             time.sleep(5)

from agents.teacher_router import call_teacher

PLANNER_PROMPT = """You are a planning agent. Break the following coding problem
into a short numbered list of concrete implementation steps. Do not write code yet.

Problem:
{problem}
"""

def plan(problem: str) -> str:
    return call_teacher(PLANNER_PROMPT.format(problem=problem))