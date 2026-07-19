"""
Runs the teacher pipeline (Groq/Gemini API teacher) on Sakshi's task subset
and logs full trajectories.
Resumable — safe to interrupt and re-run.
"""

import json
import os

# Skip Ollama entirely and go straight to the Groq/Gemini API teacher —
# avoids wasted retry/sleep time across all 180 tasks.
os.environ["TEACHER_MODEL"] = "qwen2.5:7b-instruct-q4_K_M"

from agents.graph import build_graph

INPUT_FILE = "datasets/raw/tasks_sakshi_api.json"
OUTPUT_FILE = "datasets/raw/trajectories_sakshi_api.json"


def load_progress():
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE) as f:
            return json.load(f)
    return {}


def save_progress(results):
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)


def main():
    with open(INPUT_FILE) as f:
        tasks = json.load(f)

    pipeline = build_graph()
    results = load_progress()

    for task in tasks:
        if task["id"] in results:
            continue
        print(f"Running {task['id']}...")
        state = {
            "problem": task["problem"], "test_code": task["test"],
            "plan": "", "code": "", "passed": False, "error": "", "retries": 0
        }
        final_state = pipeline.invoke(state)
        results[task["id"]] = {
            "problem": task["problem"], "plan": final_state["plan"],
            "code": final_state["code"], "passed": final_state["passed"],
            "retries": final_state.get("retries", 0)
        }
        save_progress(results)
        print(f"  passed={final_state['passed']}")

    passed_count = sum(1 for r in results.values() if r["passed"])
    print(f"\n=== {passed_count}/{len(results)} passed ===")


if __name__ == "__main__":
    main()