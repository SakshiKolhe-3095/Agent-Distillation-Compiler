"""
Merges trajectory JSON logs from all three teacher sources (Yeshita's 7B,
Faiza's 14B, Sakshi's API teacher) into one deduplicated dataset.
"""

import json
import os

RAW_DIR = "datasets/raw"

SOURCE_FILES = {
    "yeshita_7b": "trajectories_yeshita_7b.json",
    "faiza_14b": "trajectories_faiza_14b.json",
    "sakshi_api": "trajectories_sakshi_api.json",
}

OUTPUT_FILE = OUTPUT_FILE = os.path.join(RAW_DIR, "trajectories_merged.json")


def load_source(name: str, filename: str) -> dict:
    path = os.path.join(RAW_DIR, filename)
    if not os.path.exists(path):
        print(f"[collector] {name}: file not found ({path}), skipping")
        return {}
    with open(path) as f:
        data = json.load(f)
    print(f"[collector] {name}: loaded {len(data)} trajectories from {filename}")
    return data


def merge_sources() -> dict:
    """
    Merges all source trajectory files. If the same task id appears in
    multiple sources, prefer the one that passed; if both passed (or both
    failed), keep the first one seen and record where it came from.
    """
    merged = {}
    for source_name, filename in SOURCE_FILES.items():
        data = load_source(source_name, filename)
        for task_id, trajectory in data.items():
            trajectory = dict(trajectory)
            trajectory["source"] = source_name

            if task_id not in merged:
                merged[task_id] = trajectory
            else:
                existing = merged[task_id]
                # prefer a passing trajectory over a failing one
                if trajectory.get("passed") and not existing.get("passed"):
                    merged[task_id] = trajectory
                # otherwise keep whichever came first (no-op)

    return merged


def main():
    merged = merge_sources()
    with open(OUTPUT_FILE, "w") as f:
        json.dump(merged, f, indent=2)

    total = len(merged)
    passed = sum(1 for t in merged.values() if t.get("passed"))
    print(f"\n=== Merged dataset: {total} unique tasks, {passed} passing ({passed/total*100:.1f}%) ===")
    print(f"Written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()