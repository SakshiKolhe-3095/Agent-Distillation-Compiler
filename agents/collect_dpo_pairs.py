"""
Re-runs tasks that previously required retries, capturing the full
code_history to build DPO preference pairs. Now builds a pair between EACH
consecutive attempt (not just first-vs-final), giving more training signal
per task.
"""
import json
import os

os.environ["TEACHER_MODEL"] = "qwen2.5:14b-instruct-q4_K_M"

from agents.graph import build_graph

TASKS_FILE = "datasets/raw/tasks_faiza_14b.json"
TRAJECTORIES_FILE = "datasets/raw/trajectories_faiza_14b.json"
OUTPUT_FILE = "datasets/raw/dpo_pairs_faiza_14b.json"


def load_progress():
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE) as f:
            return json.load(f)
    return {}


def save_progress(results):
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)


def main():
    with open(TASKS_FILE) as f:
        tasks = {t["id"]: t for t in json.load(f)}

    with open(TRAJECTORIES_FILE) as f:
        trajectories = json.load(f)

    retried_ids = [tid for tid, t in trajectories.items() if t["retries"] > 0]
    print(f"Re-running {len(retried_ids)} tasks that had retries...")

    pipeline = build_graph()
    results = load_progress()

    for tid in retried_ids:
        if tid in results:
            continue

        task = tasks[tid]
        print(f"Running {tid}...")
        state = {
            "problem": task["problem"], "test_code": task["test"],
            "plan": "", "code": "", "passed": False, "error": "",
            "retries": 0, "code_history": []
        }
        final_state = pipeline.invoke(state)

        history = final_state.get("code_history", [])
        full_chain = history + [final_state["code"]]

        if len(full_chain) < 2:
            print(f"  no retries this run, skipping pair")
            continue

        # build a pair for EACH consecutive attempt, not just first-vs-final
        pairs = []
        for i in range(len(full_chain) - 1):
            pairs.append({
                "rejected": full_chain[i],
                "preferred": full_chain[i + 1],
            })

        results[tid] = {
            "problem": task["problem"],
            "pairs": pairs,
            "passed": final_state["passed"],
            "retries": final_state.get("retries", 0),
        }
        save_progress(results)
        print(f"  passed={final_state['passed']}, {len(pairs)} pair(s) saved")

    total_pairs = sum(len(v["pairs"]) for v in results.values())
    print(f"\n=== {len(results)} tasks, {total_pairs} total DPO pairs ===")


if __name__ == "__main__":
    main()