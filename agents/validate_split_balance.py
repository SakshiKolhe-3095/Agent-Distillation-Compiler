"""
Validates that train/val/test splits are balanced across task difficulty
(source: humaneval vs mbpp, and retries as a difficulty proxy).
"""
import json


def load(path):
    with open(path) as f:
        return json.load(f)


def analyze(name, data):
    total = len(data)
    sources = {}
    retry_buckets = {"easy(0)": 0, "medium(1-2)": 0, "hard(3)": 0}

    for key, item in data.items():
        source = key.split("_")[0]
        sources[source] = sources.get(source, 0) + 1

        r = item.get("retries", 0)
        if r == 0:
            retry_buckets["easy(0)"] += 1
        elif r <= 2:
            retry_buckets["medium(1-2)"] += 1
        else:
            retry_buckets["hard(3)"] += 1

    print(f"\n{name} (n={total})")
    print("  Source split:", {k: f"{v} ({v/total:.1%})" for k, v in sources.items()})
    print("  Difficulty split:", {k: f"{v} ({v/total:.1%})" for k, v in retry_buckets.items()})


def main():
    train = load("datasets/splits/train.json")
    val = load("datasets/splits/val.json")
    test = load("datasets/splits/test.json")

    analyze("TRAIN", train)
    analyze("VAL", val)
    analyze("TEST", test)


if __name__ == "__main__":
    main()