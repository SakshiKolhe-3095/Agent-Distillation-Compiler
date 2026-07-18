# import requests

# OLLAMA_URL = "http://localhost:11434/api/generate"
# MODEL = "qwen2.5:7b-instruct-q4_K_M"

# DEBUGGER_PROMPT = """You are a debugging agent. The following code failed its tests.
# Fix the code so all tests pass. Return only the corrected Python code, no explanation.

# Problem:
# {problem}

# Code:
# {code}

# Test error:
# {error}
# """

# def debug(problem: str, code: str, error: str) -> str:
#     response = requests.post(OLLAMA_URL, json={
#         "model": MODEL,
#         "prompt": DEBUGGER_PROMPT.format(problem=problem, code=code, error=error),
#         "stream": False
#     })
#     response.raise_for_status()
#     return response.json()["response"]


# import requests
# from agents.utils import strip_code_fences

# OLLAMA_URL = "http://localhost:11434/api/generate"
# MODEL = "qwen2.5:7b-instruct-q4_K_M"

# DEBUGGER_PROMPT = """You are a debugging agent. The following code failed its tests.
# Fix the code so all tests pass. Return only the corrected Python code, no explanation.

# Problem:
# {problem}

# Code:
# {code}

# Test code:
# {test_code}

# Test error:
# {error}

# CRITICAL: If the error is a NameError about a missing function, the function name in
# your code does NOT match the test code exactly. Copy the exact function name used in
# the test code's assert statements — character for character.
# """

# def debug(problem: str, code: str, test_code: str, error: str) -> str:
#     response = requests.post(OLLAMA_URL, json={
#         "model": MODEL,
#         "prompt": DEBUGGER_PROMPT.format(problem=problem, code=code, test_code=test_code, error=error),
#         "stream": False
#     })
#     response.raise_for_status()
#     return strip_code_fences(response.json()["response"])


from agents.teacher_router import call_teacher
from agents.utils import strip_code_fences

DEBUGGER_PROMPT = """You are a debugging agent. The following code failed its tests.
Fix the code so all tests pass. Return only the corrected Python code, no explanation.

Problem:
{problem}

Code:
{code}

Test code:
{test_code}

Test error:
{error}

CRITICAL: If the error is a NameError about a missing function, the function name in
your code does NOT match the test code exactly. Copy the exact function name used in
the test code's assert statements — character for character.
"""

def debug(problem: str, code: str, test_code: str, error: str) -> str:
    response = call_teacher(DEBUGGER_PROMPT.format(problem=problem, code=code, test_code=test_code, error=error))
    return strip_code_fences(response)