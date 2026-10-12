# Results

All runs: 1× NVIDIA L40S, vLLM v0.31.0 (V2 model runner) with this repo's patches, Triton attention and Mamba backends, prefix caching with `--mamba-cache-mode align --prefix-match-unit 16`. Each configuration was run once unless noted; repeat runs of the same configuration have differed by up to ~4%.

## Qwen/Qwen3.5-4B, Spec-Bench (public)

BF16 weights and KV. All 480 Spec-Bench prompts at c=32, 256 output tokens, temperature 0, after a 64-prompt warmup. Native MTP head.

| MTP K=5, c=32 | Output tok/s | TPOT (ms) | TTFT (ms) | Accept length |
|---|---|---|---|---|
| Stock kernels | 2527 | 11.61 | 205 | 3.940 |
| RecoverSSM | 2789 | 10.55 | 188 | 3.943 |

### K × concurrency sweep

Output tok/s, stock → RecoverSSM. Each server ran c=1 (80 prompts), then c=8 (240), then c=32 (480), with prefix caching on. Single run per cell.

| K | c=1 | c=8 | c=32 |
|---|---|---|---|
| 1 | 117 → 116 (−1%) | 760 → 748 (−2%) | 2062 → 2010 (−3%) |
| 3 | 151 → 150 (−1%) | 909 → 985 (+8%) | 2583 → **3004 (+16%)** |
| 5 | 152 → 151 (−1%) | 929 → 946 (+2%) | 2665 → 2847 (+7%) |
| 7 | 140 → 142 (+1%) | 891 → 921 (+3%) | 2436 → 2693 (+11%) |

Accept length is unchanged in every cell; for example, K=5 at c=32 is 3.936 → 3.927, with per-position accept .855 .692 .558 .456 .375 → .855 .692 .556 .453 .372.

- The gain grows with concurrency, because the per-position state write-back that RecoverSSM removes scales with batch size. At c=1 there is no gain.
- At K=1, only one draft position's write is saved, and the fixed commit cost makes it slightly slower. Do not enable RecoverSSM at K=1.
- The K=5, c=32 cell differs from the standalone run above (+7% here vs. +10%), because here the server had already served the c=1 and c=8 runs, which changes prefix-cache state for both arms. Expect about ±4% run-to-run.

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
