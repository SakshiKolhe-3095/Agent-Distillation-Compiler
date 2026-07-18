"""
Re-runs tasks that previously required retries, this time capturing the full
code_history so we can build DPO preference pairs (rejected = early failing
attempt, preferred = final passing attempt).
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
        if len(history) == 0:
            # never actually retried this time around (model got it right first try)
            print(f"  no retries this run, skipping pair")
            continue

        results[tid] = {
            "problem": task["problem"],
            "rejected": history[0],          # first attempt (was failing)
            "preferred": final_state["code"],  # final attempt
            "passed": final_state["passed"],
            "retries": final_state.get("retries", 0),
        }
        save_progress(results)
        print(f"  passed={final_state['passed']}, pair saved")

    print(f"\n=== {len(results)} DPO pairs collected ===")


if __name__ == "__main__":
    main()