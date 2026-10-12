# Results

All runs: 1× NVIDIA L40S, vLLM v0.31.0 (V2 model runner) with this repo's patches, Triton attention and Mamba backends, prefix caching with `--mamba-cache-mode align --prefix-match-unit 16`. Each configuration was run once unless noted; repeat runs of the same configuration have differed by up to ~4%.

## Qwen/Qwen3.5-4B, Spec-Bench (public)

BF16 weights and KV. All 480 Spec-Bench prompts at c=32, 256 output tokens, temperature 0, after a 64-prompt warmup. Native MTP head.

| MTP K=5, c=32 | Output tok/s | TPOT (ms) | TTFT (ms) | Accept length |
|---|---|---|---|---|
| Stock kernels | 2527 | 11.61 | 205 | 3.940 |
| RecoverSSM | 2789 | 10.55 | 188 | 3.943 |

K ∈ {1, 3, 5, 7} × c ∈ {1, 8, 32} sweep: _in progress._

## Internal agent workload (anonymized)

Model: a Qwen3.5-4B-architecture agentic-RAG finetune with a TTT-finetuned MTP head.

Workload:
- Replay of real agent requests, c=30, 60 s.
- ~3.5k-token prompts sharing a ~1.9k-token system prompt.
- FP8 weights and FP8 KV.

Throughput is relative to the run without spec decode.

| Configuration | Relative throughput | Accept length | Prefix hit |
|---|---|---|---|
| No spec decode | 1.00× | – | 0.77 |
| MTP K=5, full drafter vocab | 1.62× | 4.63 | 0.76 |
| + 64k drafter vocab | 1.85× | 4.57 | 0.77 |
| + RecoverSSM | 2.08× | 4.58 | 0.81 |

### RecoverSSM by K (64k drafter vocab)

| K | Stock → RecoverSSM | Accept length (stock / RecoverSSM) |
|---|---|---|
| 1 | +13% | 1.93 / 1.93 |
| 3 | −6% (single run; not investigated) | 3.47 / 3.48 |
| 5 | +12% | 4.57 / 4.58 |
| 7 | +14% | 5.39 / 5.37 |

## MTP finetuning reproduction (Qwen3.5-4B, Alpaca self-distillation)

_In progress_ — native head vs. FastMTP-style finetune vs. TTT + POSTNORM finetune, Spec-Bench K=5.
