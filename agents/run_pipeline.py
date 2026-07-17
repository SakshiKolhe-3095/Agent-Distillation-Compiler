"""
End-to-end test: runs the 4-agent teacher pipeline on 10 sample HumanEval problems.
Resumable — if interrupted, re-running this script skips already-completed problems.
"""
import json
import os
from datasets import load_dataset
from agents.graph import build_graph

RESULTS_FILE = "results.json"


def load_progress():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return {}


def save_progress(results):
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)


def main():
    dataset = load_dataset("openai/openai_humaneval", split="test")
    samples = dataset.select(range(10))

    pipeline = build_graph()
    results = load_progress()

    for i, sample in enumerate(samples):
        key = str(i)
        if key in results:
            print(f"Skipping problem {i+1} (already done — passed: {results[key]})")
            continue

        print(f"\n=== Problem {i+1} ===")
        state = {
            "problem": sample["prompt"],
            "test_code": sample["test"],
            "plan": "", "code": "", "passed": False, "error": "", "retries": 0
        }
        final_state = pipeline.invoke(state)
        results[key] = final_state["passed"]
        save_progress(results)
        print(f"Passed: {final_state['passed']} (retries: {final_state.get('retries', 0)})")

    pass_rate = sum(results.values()) / len(results) * 100
    print(f"\n=== Pass@1 on {len(results)} samples: {pass_rate:.1f}% ===")


if __name__ == "__main__":
    main()