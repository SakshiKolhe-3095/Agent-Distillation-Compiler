"""
Builds labeled easy/hard training data for the router from trajectory results.
Label: 1 = hard (route to teacher), 0 = easy (route to student).
"""
import json
import os

TRAJECTORY_FILES = [
    "datasets/raw/trajectories_yeshita_7b.json",
    "datasets/raw/trajectories_faiza_14b.json",
    "datasets/raw/trajectories_sakshi_api.json",
]

OUTPUT_FILE = "datasets/raw/router_training_data.json"


def build_labels(files: list[str]) -> list[dict]:
    examples = []
    for f in files:
        if not os.path.exists(f):
            print(f"Skipping missing file: {f}")
            continue
        data = json.load(open(f))
        for task_id, entry in data.items():
            label = 1 if entry.get("retries", 0) >= 2 or not entry.get("passed") else 0
            examples.append({"id": task_id, "problem": entry["problem"], "label": label})
    return examples


if __name__ == "__main__":
    data = build_labels(TRAJECTORY_FILES)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(data, f, indent=2)
    easy = sum(1 for d in data if d["label"] == 0)
    hard = sum(1 for d in data if d["label"] == 1)
    print(f"Built {len(data)} examples ({easy} easy, {hard} hard)")