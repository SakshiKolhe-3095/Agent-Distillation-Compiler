"""
QLoRA fine-tune, Sakshi's secondary experiment -- Llama3.1-8B-Instruct, rank 16.
Base model differs from Faiza's/Yeshita's Qwen2.5-Coder-7B runs, for a
base-model comparison (Week 5 Fri task).
Sized for 6GB VRAM (RTX 4050): batch 1, grad accum 16, gradient checkpointing on.
"""

import os
from unsloth import FastLanguageModel

# --- Same Windows fix as Yeshita's ablation run (unslothai/unsloth#3827) ---
import unsloth_zoo.fused_losses.cross_entropy_loss as _ce_mod

def _patched_get_chunk_multiplier(vocab_size, target_gb=None):
    target_gb = 0.5
    multiplier = (vocab_size * 4 / 1024 / 1024 / 1024) / target_gb
    return multiplier / 4

_ce_mod._get_chunk_multiplier = _patched_get_chunk_multiplier
# --- end fix ---

from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
import torch

from training.wandb_config import init_wandb_run

MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 1024
LORA_RANK = 16
OUTPUT_DIR = "models/sakshi_llama31_8b_rank16"


def formatting_func(example):
    return example["prompt"] + "\n\n" + example["completion"]


def main():
    run = init_wandb_run(
        run_name="sakshi-llama31-8b-rank16",
        config={
            "base_model": MODEL_NAME,
            "rank": LORA_RANK,
            "batch_size": 1,
            "grad_accum": 16,
            "max_seq_length": MAX_SEQ_LENGTH,
        },
        tags=["secondary-experiment", "llama3.1", "rank16"],
    )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
        device_map={"": 0},
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=42,
    )

    dataset = load_dataset("json", data_files={
        "train": "datasets/sft/train.jsonl",
        "validation": "datasets/sft/val.jsonl",
    })

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        formatting_func=formatting_func,
        max_seq_length=MAX_SEQ_LENGTH,
        args=SFTConfig(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,
            num_train_epochs=3,
            learning_rate=2e-4,
            logging_steps=5,
            output_dir=OUTPUT_DIR,
            save_strategy="epoch",
            eval_strategy="epoch",
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            optim="adamw_8bit",
            completion_only_loss=False,
            dataset_text_field=None,
            report_to="wandb",
        ),
    )

    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    run.finish()


if __name__ == "__main__":
    main()