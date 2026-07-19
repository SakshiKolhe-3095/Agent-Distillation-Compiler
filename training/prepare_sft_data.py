"""
Converts compressed_dataset_v1.json into SFT-format {prompt, completion} pairs,
splits into train/val/test (80/10/10).
PLACEHOLDER: running on the 174-example compressed set (Yeshita + Faiza sources only).
Repoint INPUT_FILE once Sakshi's merged/deduped dataset lands, then re-run.
"""
import json
import random

INPUT_FILE = "datasets/raw/compressed_dataset_v1.json"
OUTPUT_DIR = "datasets/sft"

random.seed(42)

def main():
    with open(INPUT_FILE) as f:
        data = json.load(f)

    examples = []
    for key, entry in data.items():
        prompt = f"Problem:\n{entry['problem']}\n\nSolve this in Python."
        completion = f"{entry['compressed_cot']}"
        examples.append({"id": key, "prompt": prompt, "completion": completion})

    random.shuffle(examples)
    n = len(examples)
    train_end = int(n * 0.8)
    val_end = train_end + int(n * 0.1)

    splits = {
        "train": examples[:train_end],
        "val": examples[train_end:val_end],
        "test": examples[val_end:]
    }

    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for name, subset in splits.items():
        with open(f"{OUTPUT_DIR}/{name}.jsonl", "w") as f:
            for ex in subset:
                f.write(json.dumps(ex) + "\n")
        print(f"{name}: {len(subset)} examples")

if __name__ == "__main__":
    main()