"""
Builds train/val/test split (80/10/10) from the merged, passing-only
trajectories dataset (all 3 team members' sources combined via
Sakshi's collector.py).
"""
import json
import random
import os

INPUT_FILE = "datasets/raw/trajectories_merged.json"
TRAIN_OUT = "datasets/splits/train.json"
VAL_OUT = "datasets/splits/val.json"
TEST_OUT = "datasets/splits/test.json"

SEED = 42


def main():
    with open(INPUT_FILE) as f:
        data = json.load(f)

    passing = {k: v for k, v in data.items() if v.get("passed")}

    items = list(passing.items())
    random.Random(SEED).shuffle(items)

    n = len(items)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)

    train = dict(items[:n_train])
    val = dict(items[n_train:n_train + n_val])
    test = dict(items[n_train + n_val:])

    os.makedirs("datasets/splits", exist_ok=True)

    with open(TRAIN_OUT, "w") as f:
        json.dump(train, f, indent=2)
    with open(VAL_OUT, "w") as f:
        json.dump(val, f, indent=2)
    with open(TEST_OUT, "w") as f:
        json.dump(test, f, indent=2)

    print(f"Merged total: {len(data)}, Passing: {n}")
    print(f"Train: {len(train)} ({len(train)/n:.1%})")
    print(f"Val:   {len(val)} ({len(val)/n:.1%})")
    print(f"Test:  {len(test)} ({len(test)/n:.1%})")

    if n < 800:
        print(f"\nNOTE: {n} passing examples is below the 800-1500 target "
              f"range — flagging for team, may need more trajectory collection.")


if __name__ == "__main__":
    main()