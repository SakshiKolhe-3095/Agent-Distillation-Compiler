"""
Tester agent — runs generated code + test cases inside the Docker sandbox
(agents/sandbox_executor.py). Replaces the earlier subprocess stopgap.
"""

from agents.sandbox_executor import run_in_sandbox


def run_tests(code: str, test_code: str, timeout: int = 10) -> tuple[bool, str]:
    result = run_in_sandbox(code, test_code)
    passed = result["passed"]
    error = "" if passed else (result["stdout"] + result["stderr"])
    return passed, error