"""
Live checkpoint evaluation hook.
Watches a training output directory for new checkpoint-N folders as they
appear (saved after each epoch by SFTTrainer), evaluates each one's pass@1
on the held-out val set as soon as it shows up, and logs the result to the
shared W&B project -- so training progress is visible without waiting for
the whole run to finish.

Reuses the single-shot generation + sandbox test pattern from
evaluation/eval_checkpoint.py, generalized to accept any checkpoint dir.
"""

import argparse
import json
import os
import re
import sys
import time

_cwd = sys.path[0]
if _cwd in sys.path:
    sys.path.remove(_cwd)  # avoid local datasets/ folder shadowing pip's datasets
from unsloth import FastLanguageModel
sys.path.insert(0, _cwd)

sys.path.insert(0, os.path.join(_cwd, "training"))
from agents.tester import run_tests
from wandb_config import init_wandb_run


def extract_code(generated_text: str) -> str:
    match = re.search(r"```python\n(.*?)```", generated_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return generated_text.strip()


def load_val_tasks(val_split_path: str, raw_tasks_path: str) -> list:
    """
    Combines the passing-only val split (for problem text) with the raw
    task file (for exact test code), same join eval_checkpoint.py does.
    """
    with open(val_split_path) as f:
        val_data = json.load(f)
    with open(raw_tasks_path) as f:
        raw_tasks = {t["id"]: t for t in json.load(f)}

    tasks = []
    for task_id, entry in val_data.items():
        if not entry.get("passed"):
            continue
        test_code = raw_tasks.get(task_id, {}).get("test", "")
        if not test_code:
            continue
        tasks.append({"id": task_id, "problem": entry["problem"], "test": test_code})
    return tasks


def eval_checkpoint(checkpoint_dir: str, tasks: list, max_seq_length: int = 1024) -> float:
    """
    Loads a single checkpoint, runs single-shot generation + sandbox test on
    every task, returns pass@1 as a float in [0, 1].
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=checkpoint_dir,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        device_map={"": 0},
    )
    FastLanguageModel.for_inference(model)

    passed_count = 0
    for task in tasks:
        prompt = f"Problem:\n{task['problem']}\n\nSolve this in Python."
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):]
        code = extract_code(generated)
        passed, _ = run_tests(code, task["test"])
        if passed:
            passed_count += 1

    del model
    return passed_count / len(tasks) if tasks else 0.0


def find_checkpoints(training_output_dir: str) -> list:
    """Returns sorted list of checkpoint-N subfolder paths, oldest first."""
    if not os.path.isdir(training_output_dir):
        return []
    subdirs = [
        d for d in os.listdir(training_output_dir)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(training_output_dir, d))
    ]
    subdirs.sort(key=lambda d: int(d.split("-")[1]))
    return [os.path.join(training_output_dir, d) for d in subdirs]


def main():
    parser = argparse.ArgumentParser(description="Live pass@1 tracking for checkpoints as they appear.")
    parser.add_argument("--training-dir", required=True,
                         help="Training output_dir to watch for checkpoint-N folders.")
    parser.add_argument("--val-split", default="datasets/splits/val.json")
    parser.add_argument("--raw-tasks", default="datasets/raw/tasks_faiza_14b.json",
                         help="Raw task file to pull exact test code from (adjust per source).")
    parser.add_argument("--poll-seconds", type=int, default=30,
                         help="How often to check for new checkpoints while training is ongoing.")
    parser.add_argument("--run-name", default="live-checkpoint-eval")
    args = parser.parse_args()

    tasks = load_val_tasks(args.val_split, args.raw_tasks)
    print(f"Loaded {len(tasks)} val tasks for evaluation.")

    run = init_wandb_run(
        run_name=args.run_name,
        config={"training_dir": args.training_dir, "num_val_tasks": len(tasks)},
        tags=["live-checkpoint-eval"],
    )

    seen = set()
    print(f"Watching {args.training_dir} for new checkpoints (Ctrl+C to stop)...")
    try:
        while True:
            checkpoints = find_checkpoints(args.training_dir)
            new_ones = [c for c in checkpoints if c not in seen]
            for ckpt in new_ones:
                print(f"\nFound new checkpoint: {ckpt}")
                pass_at_1 = eval_checkpoint(ckpt, tasks)
                step = int(os.path.basename(ckpt).split("-")[1])
                print(f"  pass@1 = {pass_at_1:.1%} at step {step}")
                run.log({"checkpoint_step": step, "checkpoint_pass_at_1": pass_at_1})
                seen.add(ckpt)
            time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        print("\nStopped watching.")
    finally:
        run.finish()


if __name__ == "__main__":
    main()