"""
Unit tests for core agent modules.
These test structure/logic without requiring live LLM calls where possible,
and use lightweight mocking for calls that would otherwise hit Ollama/Groq/Gemini.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from unittest.mock import patch


def test_coder_strip_code_fences_plain():
    from agents.coder import strip_code_fences
    assert strip_code_fences("def f(): pass") == "def f(): pass"


def test_coder_strip_code_fences_with_markdown():
    from agents.coder import strip_code_fences
    fenced = "```python\ndef f(): pass\n```"
    assert strip_code_fences(fenced) == "def f(): pass"


def test_planner_prompt_contains_problem():
    from agents.planner import PLANNER_PROMPT
    problem = "reverse a string"
    formatted = PLANNER_PROMPT.format(problem=problem)
    assert problem in formatted


def test_coder_prompt_contains_all_fields():
    from agents.coder import CODER_PROMPT
    formatted = CODER_PROMPT.format(problem="p", plan="pl", test_code="tc")
    assert "p" in formatted and "pl" in formatted and "tc" in formatted


@patch("agents.planner.call_teacher", return_value="1. step one\n2. step two")
def test_plan_calls_teacher_router(mock_call):
    from agents.planner import plan
    result = plan("some problem")
    assert "step" in result
    mock_call.assert_called_once()


@patch("agents.coder.call_teacher", return_value="```python\ndef f(): return 1\n```")
def test_code_calls_teacher_router_and_strips_fences(mock_call):
    from agents.coder import code
    result = code("problem", "plan", "test_code")
    assert result == "def f(): return 1"
    mock_call.assert_called_once()


def test_sandbox_pass_case():
    from agents.sandbox_executor import run_in_sandbox
    result = run_in_sandbox(
        "def add(a, b):\n    return a + b\n",
        "def test_add():\n    assert add(2, 3) == 5\n\ntest_add()\n",
    )
    assert result["passed"] is True


def test_sandbox_timeout_case():
    from agents.sandbox_executor import run_in_sandbox
    result = run_in_sandbox(
        "def add(a, b):\n    while True:\n        pass\n",
        "def test_add():\n    assert add(2, 3) == 5\n\ntest_add()\n",
        timeout_seconds=5,
    )
    assert result["passed"] is False