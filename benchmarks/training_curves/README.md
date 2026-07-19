# Primary QLoRA Training Curves

Model: Qwen2.5-Coder-7B-Instruct (4-bit), LoRA rank 32
Data: 139 train / 17 val examples (SFT format, compressed CoT)

## Results
- Epoch 1 eval_loss: 0.6163
- Epoch 2 eval_loss: 0.5461
- Epoch 1 val pass@1: 15/17 (88.2%)

## Files
- `qlora_primary_eval_loss.png` — eval/loss at end of epoch 1
- `qlora_primary_epoch2_train_loss.png` — train/loss curve during epoch 2 (resumed from epoch 1 checkpoint)

W&B runs:
- Epoch 1: https://wandb.ai/lakshitsaini1098-d-y-patil-international-university/huggingface/runs/sv7r0lsd
- Epoch 2: https://wandb.ai/lakshitsaini1098-d-y-patil-international-university/huggingface/runs/i9gjpi13
