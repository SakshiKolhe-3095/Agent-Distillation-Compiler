"""
Splits all_tasks.json into three subsets — one per teacher instance.
"""
import json

with open("datasets/raw/all_tasks.json") as f:
    tasks = json.load(f)

n = len(tasks)
third = n // 3

splits = {
    "faiza_14b": tasks[:third],
    "yeshita_7b": tasks[third:2*third],
    "sakshi_api": tasks[2*third:]
}

for name, subset in splits.items():
    with open(f"datasets/raw/tasks_{name}.json", "w") as f:
        json.dump(subset, f, indent=2)
    print(f"{name}: {len(subset)} tasks")