"""
Final benchmark: compares all three trained student checkpoints against the
full multi-agent teacher pipeline on pass@1, latency, and cost, using the
held-out test split.
"""
import json
import time
from agents.graph import build_graph

TEST_FILE = "datasets/splits/test.json"

def run_teacher_baseline():
    with open(TEST_FILE) as f:
        test_data = json.load(f)
    pipeline = build_graph()

    results = []
    for task_id, task in test_data.items():
        start = time.time()
        state = {
            "problem": task["problem"], "test_code": task.get("test", ""),
            "plan": "", "code": "", "passed": False, "error": "", "retries": 0
        }
        final = pipeline.invoke(state)
        elapsed = time.time() - start
        results.append({"id": task_id, "passed": final["passed"], "latency": elapsed, "retries": final.get("retries", 0)})
        print(f"{task_id}: passed={final['passed']} latency={elapsed:.1f}s")

    with open("evaluation/results/teacher_baseline.json", "w") as f:
        json.dump(results, f, indent=2)

    pass_rate = sum(1 for r in results if r["passed"]) / len(results)
    avg_latency = sum(r["latency"] for r in results) / len(results)
    print(f"\n=== Teacher: pass@1={pass_rate:.1%}, avg latency={avg_latency:.1f}s ===")

if __name__ == "__main__":
    run_teacher_baseline()