# GDN RecoverSSM

## Why

With MTP speculative decoding, each decode step verifies a window of K+1 tokens per request. vLLM's stock GDN kernel for this (`fused_sigmoid_gating_delta_rule_update`) writes the recurrent state after **every** window position, so the next step can resume from whichever prefix was accepted. For Qwen3.5-4B, that is 24 GDN layers × 32 value heads × 128×128 state per position. At K=5 it costs 7.5 ms of a 32 ms decode step at 29 concurrent requests on an L40S. It also reserves K extra Mamba blocks per request.

## Protocol

This is the same protocol vLLM uses for Kimi-K3's KDA layers (`--use-replayssm`).

**Verify** (in the forward pass, `_gdn_recoverssm_verify_kernel`):
- Load the checkpoint state S₀ once and run the window in registers.
- Write the outputs, but never write the state.
- Per token, record:
  - the beta-scaled delta-rule correction `c_i = β_i (v_i − S_{i−1} k̂_i)` (fp32, per value head);
  - the raw key and the raw gate input (activation dtype, lossless copies of the inputs).

**Commit** (after sampling, `GDNRecoverSSMCommitContext.commit`):
- **Plan kernel:** per request, compute how many tokens were accepted, the destination block, and, in align mode, whether the accepted prefix crossed a Mamba block boundary.
- **Conv compaction:** keep the accepted part of the conv window.
- **State kernel:** replay only the accepted tokens and write the state once. The decay is `g = −exp(A_log)·softplus(a + dt_bias)`, per value head.

  `S_n = (Π_{i≤n} e^{g_i}) S₀ + Σ_{i≤n} (Π_{i<j≤n} e^{g_j}) c_i k̂_iᵀ`

  If the prefix crossed a block boundary, also write the boundary state so prefix caching can reuse it.

Each request then needs a single Mamba state slot instead of K+1. At c=30, KV capacity rose from 1.40M to 1.67M tokens.

## Where the time goes

Internal workload, K=5, c=30, L40S, nsys, average ms per step:

| | Stock | RecoverSSM |
|---|---|---|
| GDN core (verify) | 7.51 | 2.25 |
| GDN commit | – | 2.35 |
| GPU idle | 0.16 | 0.22 |
| **Decode step** | **32.3** | **28.4** |
| Step with prefill | 64.3 | 58.5 |

After the fixes below, end-to-end throughput rose 12% at K=5. On the same workload it rose 13% at K=1 and 14% at K=7. A single K=3 run measured 6% slower and has not been investigated yet.

## Two bugs that hid the gain

### 1. Replay records inside the Mamba page

The first port appended the records to the GDN layer's KV-cache state, alongside the conv and SSM state, as the KDA implementation does:
- correction: `[HV, K+1, V]` fp32;
- key + gate: `[HV, K+1, K+1]`.

That made the Mamba page ~14% larger. On hybrid models vLLM sizes the attention block so that one attention page covers one Mamba page, so the attention block grew from 576 to 656 tokens.

Prefix hits on these models land on Mamba block multiples, because a GDN state exists only at block boundaries and at the prompt-end checkpoint. Requests that share a ~1.9k-token system prompt therefore reused:
- 1,728 tokens (3 × 576) with the stock page;
- 1,312 tokens (2 × 656) with the larger page.

A sequential probe of 40 requests went from a 0.489 hit rate to 0.381.

The records only live from a step's verify to the same step's commit. We now keep them in a per-layer buffer of `max_num_seqs` rows outside the KV pool, indexed by spec-decode row (verify and commit see the same rows). The page is back to the stock size and the hit rate matches.

> Row indexing assumes one batch in flight. With pipeline parallelism or deeper async overlap, key the buffer by the request-state slot instead.

### 2. A per-step GPU→CPU sync

The commit ended with:

```python
_debug_sync("commit_state", commit_lens=self.commit_lens[:batch].tolist(), ...)
```

`_debug_sync` returns immediately unless the debug env var is set. But Python evaluates the `.tolist()` arguments first, so every step synchronized on the GPU four times and the CPU could no longer schedule ahead. In steps with a prefill, GPU idle time rose from 4.8 ms to 10.1 ms. Those steps dominate wall time at c=30, so this alone erased the decode gain. The fix guards the call with `if _DEBUG:`.

### 3. (minor) Tile tuning

`benchmarks/kernels/bench_gdn_recoverssm.py` sweeps tile sizes at the decode shape. On an L40S:
- verify: BV=16 with 1 warp, 1.68 → 1.49 ms (24 layers);
- commit: BV=32 with 8 warps, 2.31 → 2.16 ms.

The commit moves about 1.4 GB per step (read and write the state once), so it already runs at ~70% of DRAM bandwidth.

## Not done: deferred commit

Fusing step *t*'s commit into step *t+1*'s verify would save one state read per step, about 1.3 ms of decode, roughly 3% end to end. The state would then be stale between steps, and three other readers would have to wait for it:
- the align-mode pre-copy at the start of a step;
- the align postprocess copy;
- flushes when a request finishes or is preempted while its blocks are already cached.

The workloads tested here (single-turn replay) would not catch a mistake there, so we left it out.

## Upstream

- vLLM [#58863](https://github.com/vllm-project/vllm/pull/58863): RecoverSSM for Qwen GDN (Qwen4Exp), with follow-ups #61017, #59982, #59983. That PR keeps the record in the page and notes the block growth.
- Exact-block-boundary commit bug: issue [#59933](https://github.com/vllm-project/vllm/issues/59933), fixed in PR [#59962](https://github.com/vllm-project/vllm/pull/59962). Our port hit the same bug; the fix is in the plan kernel here: `final_state_col = (max(num_computed, 1) − 1) // block_size`.
