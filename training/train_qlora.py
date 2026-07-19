"""
Primary QLoRA fine-tuning run — Qwen2.5-Coder-7B, 4-bit, rank 32.
Trains on datasets/sft/train.jsonl, evals against val.jsonl.
"""
import json
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

MODEL_NAME = "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 2048
LORA_RANK = 32
OUTPUT_DIR = "training/checkpoints/qlora-primary"

TRAIN_FILE = "datasets/sft/train.jsonl"
VAL_FILE = "datasets/sft/val.jsonl"


def load_jsonl_as_dataset(path):
    examples = []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            text = f"{item['prompt']}\n\n{item['completion']}"
            examples.append({"text": text})
    return Dataset.from_list(examples)


def main():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_alpha=LORA_RANK,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    train_dataset = load_jsonl_as_dataset(TRAIN_FILE)
    val_dataset = load_jsonl_as_dataset(VAL_FILE)

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        learning_rate=2e-4,
        logging_steps=1,
        save_strategy="epoch",
        eval_strategy="epoch",
        report_to="wandb",
        run_name="qlora-primary-14b-teacher",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    print(f"\nTraining complete. Checkpoint saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()