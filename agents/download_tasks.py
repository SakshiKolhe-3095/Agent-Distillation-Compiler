"""
Downloads full HumanEval (164) + MBPP subset (~400) task sets.
"""
from datasets import load_dataset
import json
import os

os.makedirs("datasets/raw", exist_ok=True)

def download_tasks():
    humaneval = load_dataset("openai/openai_humaneval", split="test")
    mbpp = load_dataset("google-research-datasets/mbpp", split="train")

    tasks = []
    for i, item in enumerate(humaneval):
        tasks.append({"id": f"humaneval_{i}", "source": "humaneval",
                       "problem": item["prompt"], "test": item["test"]})
    for i, item in enumerate(mbpp):
        tasks.append({"id": f"mbpp_{i}", "source": "mbpp",
                       "problem": item["text"], "test": item["test_list"]})

    with open("datasets/raw/all_tasks.json", "w") as f:
        json.dump(tasks, f, indent=2)
    print(f"Saved {len(tasks)} tasks to datasets/raw/all_tasks.json")

if __name__ == "__main__":
    download_tasks()