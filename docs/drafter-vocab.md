# Reduced drafter vocabulary

Qwen3.5 has a 248,320-token vocabulary. With greedy drafting, the drafter computes full logits once per draft round: 5 × `[batch, 2560] × [2560, 248320]` GEMMs per decode step at K=5. On the internal workload (c=30, L40S) that is **10.0 ms of a 39.9 ms decode step**, more than the rest of the drafter combined.

## How it works

[`drafter_vocab/spec_vocab.py`](../drafter_vocab/spec_vocab.py) hooks vLLM V2's `DraftModelSpeculator`:

- **`load_model`:** after the drafter loads, copy the `lm_head` rows of the N most frequent tokens into a contiguous `[N, hidden]` matrix.
- **`sample_draft`:** for greedy drafting, compute logits over those N rows, take the argmax, and map it back to the full vocabulary id.
- **Fallback:** probabilistic draft sampling and `use_local_argmax_reduction` keep the original path.

The target model still verifies over the full vocabulary, so outputs are unchanged. A token outside the set can never be drafted, which shows up only as a lower accept length.

The ranking comes from [`drafter_vocab/token_space.py`](../drafter_vocab/token_space.py). It counts target-token frequencies at supervised positions of the training data and reports two coverage numbers per candidate size N:
- single-token coverage: the share of target tokens inside the top-N set;
- chain coverage: the share of length-k windows whose tokens are all inside the set.

```bash
python drafter_vocab/token_space.py --data <prepared speculators dataset> --sizes 32768 65536
SPEC_DRAFT_VOCAB=token_space/top_ids_train.npy:65536 PYTHONPATH=drafter_vocab vllm serve ...
```

## Measurements

Internal workload, MTP K=5, vLLM 0.31 V2, Triton attention, c=30.

| Drafter vocab | Drafter `lm_head` | Drafter total | Decode step | Accept length | Per-position accept | Relative throughput |
|---|---|---|---|---|---|---|
| 248k | 10.02 ms | 13.09 ms | 39.9 ms | 4.71 | .918 .824 .733 .650 .584 | 1.00× |
| 64k | 2.59 ms | 5.63 ms | 32.3 ms | 4.62 | .910 .810 .713 .628 .562 | 1.14× |
| 32k | 1.52 ms | 4.48 ms | 30.7 ms | 4.46 | .891 .781 .677 .590 .522 | 1.16× |

The step times and accept lengths come from nsys-profiled runs. Relative throughput comes from separate unprofiled runs.

64k is the default we use:
- Throughput is within run-to-run noise of 32k.
- Accept length is 0.16 higher, and accept is higher at every position.
- A 32k set covers only the most frequent tokens of the training distribution, so it degrades faster on out-of-distribution traffic.

With FlashInfer attention the same 64k trim gave +12% (from 2011 to 2258 tok/s).
