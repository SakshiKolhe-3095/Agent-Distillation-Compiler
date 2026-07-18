"""
Tester agent — runs generated code + test cases inside the Docker sandbox
(agents/sandbox_executor.py).
"""

import re
from agents.sandbox_executor import run_in_sandbox


def _ensure_check_invoked(code: str, test_code: str) -> str:
    """
    HumanEval-style tests define `check(candidate)` but never call it.
    If detected, append a call using the function name from the generated code.
    """
    if "def check(" in test_code and re.search(r"check\s*\(\s*\w+\s*\)\s*$", test_code.strip()) is None:
        match = re.search(r"def\s+(\w+)\s*\(", code)
        if match:
            fn_name = match.group(1)
            return test_code + f"\ncheck({fn_name})\n"
    return test_code


def run_tests(code: str, test_code: str, timeout: int = 10) -> tuple[bool, str]:
    full_test = _ensure_check_invoked(code, test_code)
    result = run_in_sandbox(code, full_test)
    passed = result["passed"]
    error = "" if passed else (result["stdout"] + result["stderr"])
    return passed, error