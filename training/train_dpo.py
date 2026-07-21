"""
DPO alignment pass on Faiza's primary QLoRA checkpoint (rank 32), using
preference pairs collected from teacher retry trajectories.
Sized defensively for 6GB VRAM — DPO holds a reference model copy alongside
the policy model, so this is heavier than the original SFT run.
"""
import os
os.environ["UNSLOTH_DISABLE_FAST_CROSS_ENTROPY"] = "1"  # kept for safety, real fix below

from unsloth import FastLanguageModel

# --- Windows fused-CE bug fix (unslothai/unsloth#3827) ---
import unsloth_zoo.fused_losses.cross_entropy_loss as _ce_mod
def _patched_get_chunk_multiplier(vocab_size, target_gb=None):
    target_gb = 0.5
    multiplier = (vocab_size * 4 / 1024 / 1024 / 1024) / target_gb
    return multiplier / 4
_ce_mod._get_chunk_multiplier = _patched_get_chunk_multiplier
# --- end fix ---

import json
from datasets import Dataset
from trl import DPOTrainer, DPOConfig

# CHECKPOINT_DIR = "models/qlora-primary-final"
CHECKPOINT_DIR = "models/yeshita_ablation_rank16"
# CHECKPOINT_DIR = "models/yeshita_ablation_rank16_merged"
DPO_PAIRS_FILE = "datasets/raw/dpo_pairs_faiza_14b.json"
OUTPUT_DIR = "models/qlora-primary-dpo"
MAX_SEQ_LENGTH = 512  # reduced from Faiza's 2048 — DPO needs more headroom


def build_dpo_dataset(pairs_file: str) -> Dataset:
    with open(pairs_file) as f:
        raw = json.load(f)

    MAX_CHARS = 800  # defensive cap, avoids OOM spikes from long examples

    examples = []
    for task_id, entry in raw.items():
        problem = entry.get("problem", "")[:MAX_CHARS]
        for pair in entry.get("pairs", []):
            examples.append({
                "prompt": f"Problem:\n{problem}\n\nSolve this in Python.",
                "chosen": pair["preferred"][:MAX_CHARS],
                "rejected": pair["rejected"][:MAX_CHARS],
            })
    return Dataset.from_list(examples)


def main():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=CHECKPOINT_DIR,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
        device_map={"": 0},
    )
    # NOTE: no get_peft_model() call — this checkpoint already has a LoRA
    # adapter attached (rank 8, from the original SFT run). DPO continues
    # training that same adapter directly.

    dataset = build_dpo_dataset(DPO_PAIRS_FILE)
    print(f"DPO dataset: {len(dataset)} preference pairs")

    trainer = DPOTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=DPOConfig(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,
            num_train_epochs=1,
            learning_rate=5e-5,
            logging_steps=5,
            output_dir=OUTPUT_DIR,
            save_strategy="epoch",
            bf16=True,
            optim="adamw_8bit",
            max_length=512,
        ),
    )
    trainer.train()
    model.save_pretrained_merged(OUTPUT_DIR, tokenizer, save_method="lora")


if __name__ == "__main__":
    main()