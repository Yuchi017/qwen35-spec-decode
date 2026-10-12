# Faster MTP speculative decoding for Qwen3.5 (GDN hybrid) on vLLM

This repo collects three optimizations for serving Qwen3.5-family models with MTP speculative decoding on vLLM. It also includes the measurements and findings behind them. The models in scope are Qwen3.5 and Qwen3-Next, which are hybrids of Gated DeltaNet (GDN) linear attention and full attention.

| | What | Where |
|---|---|---|
| 1 | **GDN RecoverSSM**: verify draft tokens without writing the recurrent state for every draft position, and commit only the accepted prefix after sampling | [`patches/vllm-v0.31.0-gdn-recoverssm.patch`](patches/vllm-v0.31.0-gdn-recoverssm.patch), [docs/recoverssm.md](docs/recoverssm.md) |
| 2 | **Reduced drafter vocabulary**: the MTP drafter's `lm_head` computes logits over the top-N target tokens only. Verification still uses the full vocabulary, so outputs are unchanged | [`drafter_vocab/`](drafter_vocab/), [docs/drafter-vocab.md](docs/drafter-vocab.md) |
| 3 | **vLLM-aligned MTP finetuning (TTT)**: train the MTP head with the same multi-step attention pattern and hidden states that vLLM's MTP proposer uses at inference | [`patches/speculators-mtp-ttt.patch`](patches/speculators-mtp-ttt.patch), [docs/mtp-ttt.md](docs/mtp-ttt.md) |

Along the way we also measured how vLLM 0.29 and 0.31 differ for MTP + prefix caching on hybrid models, and where GPU idle time comes from per attention backend. Those notes are in [docs/](docs/).

## Results

### Internal agent workload

Model: a Qwen3.5-4B-architecture agentic-RAG finetune with its MTP head.

Workload:
- Replay of real agent requests: ~3.5k-token prompts that share a ~1.9k-token system prompt, ~210 output tokens.
- 30 concurrent clients, 1× L40S.
- FP8 weights + FP8 KV, prefix caching, vLLM 0.31 V2 runner, Triton attention.

Throughput is shown relative to plain decoding.

| Configuration | Output throughput | Accept length (K=5) |
|---|---|---|
| No speculative decoding | 1.00× | – |
| MTP K=5 (TTT-finetuned head), full drafter vocab | 1.62× | 4.63 |
| + drafter vocab trimmed to 64k tokens | 1.85× | 4.57 |
| + GDN RecoverSSM | **2.08×** | 4.58 |

### Public model

Model: Qwen/Qwen3.5-4B with its native MTP head. Workload: Spec-Bench, all 480 prompts, 256 output tokens, 32 concurrent clients, 1× L40S, BF16.

| MTP K=5 | Output tok/s | TPOT | TTFT | Accept length |
|---|---|---|---|---|
| Stock spec-decode kernels | 2527 | 11.61 ms | 205 ms | 3.94 |
| GDN RecoverSSM | **2789 (+10.4%)** | 10.55 ms | 188 ms | 3.94 |

In a K × concurrency sweep at c=32, RecoverSSM gave +16% at K=3 (the fastest configuration, 3004 tok/s), +7% at K=5 and +11% at K=7. It gave no gain at c=1 and slightly negative results at K=1. Full table: [`results/`](results/README.md). The public MTP-finetuning reproduction is in progress.

## 1. GDN RecoverSSM

During verification, the stock GDN spec-decode kernel writes the recurrent state after every draft position. That is K+1 states per request per layer, plus one Mamba block reserved per draft position. RecoverSSM changes this in two steps:

- **Verify:** run from a read-only checkpoint and record a small per-token replay entry: the delta-rule correction (fp32), plus the raw key and gate input.
- **Commit:** after sampling, fold only the accepted tokens into the checkpoint with a single state write. In align mode, also write the block-boundary state:

  `S_n = (Π exp g_i) S_0 + Σ_i (Π_{j>i} exp g_j) c_i k_iᵀ`

This ports the protocol vLLM already uses for Kimi-K3's KDA layers to the Qwen GDN layers. On the internal workload, the per-step GDN cost dropped from 7.5 ms to 2.2 ms for verify plus 2.3 ms for commit. The decode step went from 32.3 ms to 28.4 ms, and steps that include a prefill went from 64.3 ms to 58.5 ms.

Two things in our first port hid the gain entirely. Both are described in [docs/recoverssm.md](docs/recoverssm.md):

1. **Replay records stored inside the Mamba page made every page ~14% larger.** Hybrid models align the attention block to the Mamba page, so the prefix-cache block grew from 576 to 656 tokens. Requests that share a system prompt ending mid-block reused 1,312 tokens instead of 1,728, and the prefix hit rate on a sequential probe fell from 0.489 to 0.381. The records only live between a step's verify and its commit, so we moved them to a per-layer buffer outside the KV pool.
2. **A debug helper evaluated `.tolist()` arguments unconditionally**, which forced a GPU→CPU sync every step. GPU idle time in prefill steps doubled, from 4.8 ms to 10.1 ms.

**Correctness.** Outputs are not bit-identical to the stock kernels, as expected for any kernel that reorders floating-point math. We checked with greedy decoding plus logprobs that every first divergence happens at a near-tie:
- At every first divergence, the top-1 − top-2 logprob gap is ≤ 0.875 nats (median 0.25), versus a median gap of 11.1 over all positions.
- This holds in single-turn runs and in follow-up turns whose prefix hits blocks written during decode.

See [docs/validation.md](docs/validation.md) and [`bench/correctness.sh`](bench/correctness.sh).

**Upstream.** vLLM [#58863](https://github.com/vllm-project/vllm/pull/58863) independently brings RecoverSSM to Qwen GDN, enabled for Qwen4Exp. It keeps the replay record inside the Mamba page. The exact-block-boundary bug we also hit was fixed upstream in [#59962](https://github.com/vllm-project/vllm/pull/59962).

## 2. Reduced drafter vocabulary

With full-vocabulary drafting, the drafter's `lm_head` costs 10 ms per decode step at K=5: 5 draft rounds × 248k vocabulary rows. A post-import hook copies the `lm_head` rows of the N most frequent target tokens into a small matrix. Greedy drafting then takes the argmax over those N rows and maps it back to the full vocabulary id. The target still verifies over the full vocabulary, so outputs do not change; tokens outside the set are simply never drafted.

| Drafter vocab | Drafter `lm_head` / step | Decode step | Accept length | Relative throughput |
|---|---|---|---|---|
| 248k (full) | 10.0 ms | 39.9 ms | 4.71 | 1.00× |
| 64k | 2.6 ms | 32.3 ms | 4.62 | 1.14× |
| 32k | 1.5 ms | 30.7 ms | 4.46 | 1.16× |

64k is the better default: throughput is within noise of 32k, accept length is higher, and coverage is more robust to distribution shift. Measured on the internal workload with vLLM 0.31 and Triton attention; the decode-step and accept-length columns come from nsys-profiled runs at c=30.

## 3. vLLM-aligned MTP finetuning

speculators' MTP finetune follows FastMTP: every draft step re-runs the MTP layer over the whole sequence and feeds it the pre-final-norm hidden states. vLLM's MTP proposer differs in two ways:

- **Attention:** draft step k sees the step-0 context KV plus only its own draft chain's KV, at position anchor + k.
- **Hidden states:** it feeds the verifier's output hidden states, which are taken after the final norm.

The patch adds a training-time-test (TTT) forward that matches vLLM on both counts. It also adds a vLLM-equivalent per-position accept metric, plus memory changes (logits only at supervised positions, recomputed in backward) so that 8 draft steps × 12k tokens fit on one 48 GB GPU.

On the internal model, finetuning with the upstream forward lowered online accept length from 4.37 (native head) to 3.92. The aligned forward raised it to ~4.6, and its offline per-position accept matches online vLLM within about 1 percentage point. A public reproduction on Qwen3.5-4B is in [`training/`](training/).

## Repo layout

```
patches/
  vllm-v0.31.0-gdn-recoverssm.patch   # apply to vLLM v0.31.0; enable with --use-replayssm
  speculators-mtp-ttt.patch           # apply to speculators 70fbcbd; SPECULATORS_MTP_TTT=1 SPECULATORS_MTP_POSTNORM=1
drafter_vocab/                        # SPEC_DRAFT_VOCAB=<top_ids.npy>:<N> + PYTHONPATH=drafter_vocab
bench/                                # Spec-Bench serving benchmark, sweep, greedy/logprob correctness checks
training/                             # public MTP finetune reproduction (Alpaca, Qwen3.5-4B)
docs/                                 # design notes and findings
results/                              # measured numbers
```

## Quick start

```bash
# vLLM with GDN RecoverSSM. The patch touches Python only, so a precompiled
# v0.31.0 build works: VLLM_USE_PRECOMPILED=1 pip install -e .
git clone https://github.com/vllm-project/vllm && cd vllm && git checkout v0.31.0
git apply ../qwen35-spec-decode/patches/vllm-v0.31.0-gdn-recoverssm.patch

vllm serve Qwen/Qwen3.5-4B \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 5}' --use-replayssm \
  --enable-prefix-caching --mamba-cache-mode align --mamba-backend triton \
  --attention-backend TRITON_ATTN --language-model-only

# benchmark: stock kernels vs RecoverSSM on Spec-Bench
cd bench && mkdir -p data && wget -O data/spec_bench_question.jsonl \
  https://raw.githubusercontent.com/hemingkx/Spec-Bench/refs/heads/main/data/spec_bench/question.jsonl
K=5 REPLAY=0 RUNS="32:480" bash bench_one.sh
K=5 REPLAY=1 RUNS="32:480" bash bench_one.sh
```

Requirements for RecoverSSM:
- vLLM's V2 model runner, with align mode for prefix caching.
- The Triton Mamba backend.
- Pipeline parallelism of 1. The per-row record buffer assumes one step in flight per batch row.

## Other findings

- [docs/prefix-cache-and-mtp.md](docs/prefix-cache-and-mtp.md): why MTP lowered the prefix hit rate on hybrid models in vLLM 0.29 (0.68 vs. 0.77–0.80 without MTP) and how 0.31 fixed it.
- [docs/gpu-idle.md](docs/gpu-idle.md): without speculative decoding, GPU idle time sits in CPU-bound prefill steps. With MTP, FlashInfer adds ~10 ms of idle per decode step at c=30 while Triton attention adds ~0.1 ms.
- [docs/dflash-notes.md](docs/dflash-notes.md): DFlash on a hybrid target. At K≠5 the drafter's sliding-window KV group no longer shares the target's block size, which effectively disables prefix caching. Also covers block size versus per-position accept.

## License

Apache-2.0, matching vLLM and speculators, whose code the patches modify.
