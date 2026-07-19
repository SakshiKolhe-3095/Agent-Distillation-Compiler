"""
Evaluates a trained QLoRA checkpoint's pass@1 on the held-out val set.
Loads the checkpoint, generates code directly (single-shot, no agent loop),
tests each generated solution in the sandbox.
"""
import json
import re

from agents.tester import run_tests  # local import first, needs cwd on path

import sys
_cwd = sys.path[0]
if _cwd in sys.path:
    sys.path.remove(_cwd)  # hide local datasets/ folder from shadowing pip's datasets

from unsloth import FastLanguageModel

sys.path.insert(0, _cwd)  # restore for any later local imports
CHECKPOINT_DIR = "training/checkpoints/qlora-primary"
VAL_TASKS_FILE = "datasets/splits/val.json"
MAX_SEQ_LENGTH = 2048


def extract_code(generated_text: str) -> str:
    match = re.search(r"```python\n(.*?)```", generated_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return generated_text.strip()


def main():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=CHECKPOINT_DIR,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    with open(VAL_TASKS_FILE) as f:
        val_data = json.load(f)

    results = []
    for task_id, task in val_data.items():
        if not task.get("passed"):
            continue

        prompt = f"Problem:\n{task['problem']}\n\nSolve this in Python."
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated = generated[len(prompt):]  # strip the prompt echo

        code = extract_code(generated)

        # need the original test format — pull from raw tasks file for test_code
        with open("datasets/raw/tasks_faiza_14b.json") as f:
            raw_tasks = {t["id"]: t for t in json.load(f)}
        test_code = raw_tasks.get(task_id, {}).get("test", "")

        passed, error = run_tests(code, test_code) if test_code else (False, "no test found")

        results.append({"id": task_id, "passed": passed})
        print(f"{task_id}: passed={passed}")

    pass_count = sum(1 for r in results if r["passed"])
    pass_at_1 = pass_count / len(results) if results else 0
    print(f"\n=== pass@1 = {pass_count}/{len(results)} = {pass_at_1:.1%} ===")

    import os
    os.makedirs("evaluation/results", exist_ok=True)
    with open("evaluation/results/checkpoint_eval_epoch1.json", "w") as f:
        json.dump({"pass_at_1": pass_at_1, "results": results}, f, indent=2)

if __name__ == "__main__":
    main()