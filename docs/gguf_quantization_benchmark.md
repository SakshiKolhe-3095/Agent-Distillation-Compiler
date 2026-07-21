# GGUF Quantization Benchmark -- Week 7 Wed

## Models exported
Two checkpoints exported to GGUF:
1. Sakshi's Llama3.1-8B rank-4 (student) -- `models/gguf_export/`
2. DPO-aligned Qwen2.5-7B checkpoint (teacher) -- `models/gguf_dpo/`

## Quantization levels compared (Llama3.1-8B)
| Format | Size | Notes |
|---|---|---|
| F16 (pre-quantize) | 16.1 GB | Full precision, too large for practical use |
| Q4_K_M | 4.7 GB | 4.89 bits/weight, good speed/quality tradeoff |
| Q5_K_M | 5.7 GB | 5.5 bits/weight, slightly better quality, ~20% larger |

## Recommendation
Q4_K_M is the practical default -- 21% smaller than Q5_K_M with minimal
quality difference for code generation. Q5_K_M is available as a higher-
quality option if inference hardware has more headroom.

## DPO model (Qwen2.5-7B)
Same export pipeline applied to the DPO-aligned checkpoint (Faiza/Yeshita,
Week 7). F16 + Q4_K_M + Q5_K_M all produced. Ready for serve.py teacher
path routing.