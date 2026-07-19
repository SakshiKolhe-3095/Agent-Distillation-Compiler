"""
Converts Faiza's canonical datasets/splits/{train,val,test}.json (raw trajectory
schema: problem/plan/code) into SFT-format JSONL, compressing each entry via
agents.compressor along the way.
"""
import json
import os
from agents.compressor import compress, clean_final_code

SPLITS_DIR = "datasets/splits"
OUTPUT_DIR = "datasets/sft"


def load_progress(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return {json.loads(line)["id"]: json.loads(line) for line in f}
    return {}


def convert_split(name: str):
    with open(f"{SPLITS_DIR}/{name}.json") as f:
        data = json.load(f)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = f"{OUTPUT_DIR}/{name}.jsonl"
    done = load_progress(out_path)

    with open(out_path, "a") as f:
        for key, entry in data.items():
            if key in done:
                continue
            if not entry.get("passed"):
                continue
            print(f"[{name}] compressing {key}...")
            cot = compress(entry["problem"], entry.get("plan", ""), entry["code"])
            code = clean_final_code(entry["code"])
            prompt = f"Problem:\n{entry['problem']}\n\nSolve this in Python."
            completion = cot
            f.write(json.dumps({"id": key, "prompt": prompt, "completion": completion, "final_code": code}) + "\n")

    print(f"{name}: done")


if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        convert_split(split)