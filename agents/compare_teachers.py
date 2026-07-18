"""
Compares 7B vs 14B teacher on a small sample of problems through the full
pipeline. Logs pass rate + timing for each, for Week2 Thu comparison task.
"""

import os
import time
import json
from agents.graph import build_graph

PROBLEMS = [
    {
        "problem": "write a function that reverses a string",
        "test_code": "from solution import reverse_string\ndef test_rev(): assert reverse_string('abc') == 'cba'",
    },
    {
        "problem": "write a function that checks if a number is prime",
        "test_code": "from solution import is_prime\ndef test_prime(): assert is_prime(7) == True\ndef test_not_prime(): assert is_prime(8) == False",
    },
    {
        "problem": "write a function that returns the factorial of a number",
        "test_code": "from solution import factorial\ndef test_fact(): assert factorial(5) == 120",
    },
    {
        "problem": "write a function that finds the max value in a list",
        "test_code": "from solution import find_max\ndef test_max(): assert find_max([3,1,4,1,5,9,2,6]) == 9",
    },
    {
        "problem": "write a function that checks if a string is a palindrome",
        "test_code": "from solution import is_palindrome\ndef test_pal(): assert is_palindrome('racecar') == True\ndef test_not_pal(): assert is_palindrome('hello') == False",
    },
]


def run_comparison(model_name: str) -> dict:
    os.environ["TEACHER_MODEL"] = model_name
    graph = build_graph()

    passed_count = 0
    total_time = 0.0
    details = []

    for item in PROBLEMS:
        start = time.time()
        result = graph.invoke({
            "problem": item["problem"],
            "test_code": item["test_code"],
            "retries": 0,
        })
        elapsed = time.time() - start
        total_time += elapsed

        if result["passed"]:
            passed_count += 1

        details.append({
            "problem": item["problem"],
            "passed": result["passed"],
            "retries": result.get("retries", 0),
            "time_sec": round(elapsed, 2),
        })

    return {
        "model": model_name,
        "pass_rate": f"{passed_count}/{len(PROBLEMS)}",
        "total_time_sec": round(total_time, 2),
        "avg_time_sec": round(total_time / len(PROBLEMS), 2),
        "details": details,
    }


if __name__ == "__main__":
    results = {}
    for model in ["qwen2.5:7b-instruct-q4_K_M", "qwen2.5:14b-instruct-q4_K_M"]:
        print(f"\nRunning comparison for {model}...")
        results[model] = run_comparison(model)
        print(json.dumps(results[model], indent=2))

    os.makedirs("benchmarks", exist_ok=True)
    with open("benchmarks/teacher_comparison_7b_vs_14b.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved to benchmarks/teacher_comparison_7b_vs_14b.json")