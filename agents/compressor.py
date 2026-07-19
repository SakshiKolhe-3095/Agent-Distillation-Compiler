"""
Compresses full multi-agent trajectories (plan + code) from all three teacher
sources into a single coherent chain-of-thought per task, using the teacher
model to summarize its own trace. Resumable — safe to interrupt and re-run.
"""
import requests
import json
import os

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:7b-instruct-q4_K_M"

COMPRESS_PROMPT = """You are summarizing a multi-step reasoning process into a single,
coherent chain-of-thought that a student model can learn from in ONE pass — no
mention of retries, agents, or the multi-step process. Write as if you solved
it correctly the first time.

Problem:
{problem}

Original plan:
{plan}

Final working code:
{code}

Write a single, natural reasoning explanation (a few sentences) followed by the
final code. Format:

Reasoning: <your compressed reasoning>
Code:
{{code block}}
"""

TRAJECTORY_FILES = [
    "datasets/raw/trajectories_yeshita_7b.json",
    "datasets/raw/trajectories_faiza_14b.json",
    "datasets/raw/trajectories_sakshi_api.json",   # add once Sakshi's lands
]

MERGED_FILE = "datasets/raw/trajectories_merged.json"
OUTPUT_FILE = "datasets/raw/compressed_dataset_v1.json"


def merge_trajectories(files: list[str], output_file: str) -> dict:
    merged = {}
    for f in files:
        if not os.path.exists(f):
            print(f"Skipping missing file: {f}")
            continue
        with open(f) as fp:
            merged.update(json.load(fp))
    with open(output_file, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"Merged {len(merged)} total tasks into {output_file}")
    return merged


def clean_final_code(code: str) -> str:
    """
    Keep only the function definition(s) — strip trailing example usage,
    assert statements, or __main__ blocks that leaked from debug/verification steps.
    """
    lines = code.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("if __name__") or stripped.startswith("assert ") or stripped.startswith("print("):
            break
        cleaned.append(line)
    return "\n".join(cleaned).rstrip()

def compress(problem: str, plan: str, code: str) -> str:
    response = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "prompt": COMPRESS_PROMPT.format(problem=problem, plan=plan, code=code),
        "stream": False
    })
    response.raise_for_status()
    return response.json()["response"]


def load_progress(output_file: str) -> dict:
    if os.path.exists(output_file):
        with open(output_file) as f:
            return json.load(f)
    return {}


def save_progress(output_file: str, compressed: dict):
    with open(output_file, "w") as f:
        json.dump(compressed, f, indent=2)


def compress_dataset(input_file: str, output_file: str):
    with open(input_file) as f:
        trajectories = json.load(f)

    # edge-case filter: only keep tasks that actually passed AND have non-empty code
    passing_only = {
        k: v for k, v in trajectories.items()
        if v.get("passed") and v.get("code", "").strip()
    }
    print(f"{len(passing_only)}/{len(trajectories)} trajectories are passing and non-empty")

    compressed = load_progress(output_file)

    for key, traj in passing_only.items():
        if key in compressed:
            continue
        print(f"Compressing {key}...")
        try:
            
            compressed[key] = {
                "problem": traj["problem"],
                "compressed_cot": compress(traj["problem"], traj.get("plan", ""), traj["code"]),
                "final_code": clean_final_code(traj["code"])
            }
        except Exception as e:
            print(f"  Skipping {key} due to error: {e}")
            continue
        save_progress(output_file, compressed)

    print(f"\n=== Compressed {len(compressed)} tasks total ===")


if __name__ == "__main__":
    merge_trajectories(TRAJECTORY_FILES, MERGED_FILE)
    compress_dataset(MERGED_FILE, OUTPUT_FILE)