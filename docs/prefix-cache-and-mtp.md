# MTP and prefix caching on hybrid models: vLLM 0.29 vs. 0.31

Prefix hit rate on the internal workload (c=30, `--mamba-cache-mode align`, `--prefix-match-unit 16`):

| | FlashInfer | Triton |
|---|---|---|
| 0.29, no speculative decoding | 0.80 | 0.77 |
| 0.29, MTP K=5 | **0.68** | **0.68** |
| 0.31, no speculative decoding | 0.79 | 0.77 |
| 0.31, MTP K=5 | 0.76 | 0.76 |

The attention backend does not matter: hits are decided by the scheduler's KV-cache manager before any attention kernel runs. On 0.29, turning on MTP cost about 10 points of hit rate, which shows up as extra prefill and higher TTFT. On 0.31 MTP matches no speculative decoding.

## Why

**EAGLE/MTP recompute the last matched unit.** The drafter needs the target's hidden states at the end of the matched prefix, and the prefix cache stores only KV. For multi-module MTP, the draft-layer KV near the end also depends on tokens after the prefix, which the block hash does not cover. So on a hit, vLLM drops the last match unit (`--prefix-match-unit`, here 16 tokens) and recomputes it. If B tokens match, attention reuses **B − 16**.

**GDN needs a stored state at exactly that position.** The recurrent state exists only at Mamba block boundaries (576 tokens here, because hybrid models stretch the attention block to the Mamba page size) and at the prompt-end checkpoint.

| | Prompt-end checkpoint stored at | Usable hit for a 3,800-token repeat (B = 3,792) |
|---|---|---|
| 0.29 | B | B − 16 = 3,776 has no state, so it falls back to 3,456 (6 × 576) |
| 0.31 | **B − 16** when EAGLE/MTP is on | 3,776 |

0.31 adds the shift in `get_mamba_prefill_checkpoint_position`, and the KV-cache coordinator sets `drop_eagle_checkpoint_block` on Mamba managers when EAGLE/MTP is active:

```python
checkpoint_position = (num_tokens - 1) // hash_block_size * hash_block_size
if drop_eagle_block:
    checkpoint_position -= hash_block_size
```

With 0.29 each affected hit loses on average half a Mamba block (~288 tokens).

## A related cost: page size

Anything that enlarges the Mamba page enlarges the attention block, because the two are aligned. Hits that fall back to a block boundary then lose more. A shared system prompt that ends mid-block is the common case. This is why the RecoverSSM replay records were moved out of the page; see [recoverssm.md](recoverssm.md).

MTP's boundary jumps (accepting several tokens across a block boundary) are not a problem in either version. The stock kernels write per-position states, and the align postprocess copies the boundary one. RecoverSSM's commit writes the boundary state explicitly.
