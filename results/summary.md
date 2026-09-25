# QuantCliff — summary

`t1` = single-shot tool calls (BFCL v4 subset). `t2` = multi-step agent trajectories. `grammar` = JSON-schema-constrained decoding.

| model | quant | grammar | fits 8GB | peak VRAM | t1 acc | t2 success | recovery | tok/s (median) |
|---|---|---|---|---|---|---|---|---|
| qwen3-1.7b | bf16 | off | yes | 4402 MiB | 89% | 35% | 60% | 72.9 |
| qwen3-1.7b | bf16 | on | yes | 4402 MiB | 88% | 35% | 65% | 72.8 |
| qwen3-1.7b | Q8_0 | off | yes | 2798 MiB | 89% | 42% | 86% | 115.6 |
| qwen3-1.7b | Q8_0 | on | yes | 2798 MiB | 88% | 45% | 87% | 113.6 |
| qwen3-1.7b | Q4_K_M | off | yes | 2106 MiB | 88% | 45% | 81% | 135.1 |
| qwen3-1.7b | Q4_K_M | on | yes | 2106 MiB | 87% | 45% | 76% | 131.2 |
| qwen3-1.7b | Q3_K_M | off | yes | 1948 MiB | 86% | 20% | 60% | 125.8 |
| qwen3-1.7b | Q3_K_M | on | yes | 1948 MiB | 86% | 30% | 61% | 125.2 |
| qwen3-4b | bf16 | off | no (8.05 GB weights, ngl=99) | 7822 MiB | 90% | 78% | 89% | 5.9 |
| qwen3-4b | bf16 | on | no (8.05 GB weights, ngl=99) | 7822 MiB | 90% | 78% | 89% | 5.9 |
| qwen3-4b | Q8_0 | off | yes | 5414 MiB | 90% | 78% | 89% | 57.8 |
| qwen3-4b | Q8_0 | on | yes | 5414 MiB | 90% | 78% | 89% | 57.7 |
| qwen3-4b | Q4_K_M | off | yes | 3714 MiB | 89% | 70% | 89% | 68.1 |
| qwen3-4b | Q4_K_M | on | yes | 3714 MiB | 89% | 70% | 89% | 67.8 |
| qwen3-4b | Q3_K_M | off | yes | 3312 MiB | 89% | 78% | 90% | 63.0 |
| qwen3-4b | Q3_K_M | on | yes | 3312 MiB | 89% | 78% | 90% | 62.4 |
| llama-3.1-8b | Q8_0 | off | no (8.54 GB weights, ngl=99) | 7818 MiB | 88% | 45% | 88% | 8.1 |
| llama-3.1-8b | Q8_0 | on | no (8.54 GB weights, ngl=99) | 7818 MiB | 89% | 45% | 88% | 8.1 |
| llama-3.1-8b | Q4_K_M | off | yes | 5648 MiB | 88% | 57% | 91% | 46.9 |
| llama-3.1-8b | Q4_K_M | on | yes | 5648 MiB | 89% | 57% | 94% | 45.3 |
| llama-3.1-8b | Q3_K_M | off | yes | 4901 MiB | 86% | 70% | 97% | 44.3 |
| llama-3.1-8b | Q3_K_M | on | yes | 4901 MiB | 89% | 70% | 91% | 43.9 |
