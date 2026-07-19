"""
Filters trajectories_faiza_14b.json down to only passing examples —
these become the training data source for Week 4's train/val/test split.
"""
import json

INPUT_FILE = "datasets/raw/trajectories_faiza_14b.json"
OUTPUT_FILE = "datasets/raw/trajectories_faiza_14b_passing.json"


def main():
    with open(INPUT_FILE) as f:
        data = json.load(f)

    passing = {tid: t for tid, t in data.items() if t["passed"]}

    with open(OUTPUT_FILE, "w") as f:
        json.dump(passing, f, indent=2)

    print(f"Total: {len(data)}, Passing: {len(passing)}, Filtered out: {len(data) - len(passing)}")


if __name__ == "__main__":
    main()