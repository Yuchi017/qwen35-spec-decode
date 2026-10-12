# DFlash on a GDN-hybrid target: notes

These notes cover a DFlash block-diffusion drafter: 5 sliding-window layers plus 1 full-attention layer, converted from z-lab's Qwen3.5-4B DFlash and finetuned on the internal data. It was served with vLLM 0.31's V2 runner on the internal workload.

## Prefix caching breaks unless all KV groups share a block size

Hybrid targets size the attention block to the Mamba page, and the Mamba page grows with K (the conv window holds K extra tokens). The drafter's sliding-window layers form their own KV group, and only at K=5 did it get the same block size:

| DFlash K | Attention / Mamba block | Drafter SWA block | LCM of group block sizes | KV capacity | Prefix hit |
|---|---|---|---|---|---|
| 1 | 544 | 64 | 1088 | 0.43M tok | 0.07 |
| 3 | 560 | 64 | 2240 | 0.41M tok | 0.02 |
| 5 | 576 | 576 | 576 | 1.07M tok | 0.63 |
| 7 | 592 | 64 | 2368 | 0.39M tok | 0.01 |

A hit has to land on a multiple of the LCM **and** on a stored GDN state, which almost never happens once the LCM exceeds the Mamba block. KV capacity also drops 2.5× from padding. Every K≠5 run therefore looked slow (high TTFT and low throughput) for reasons unrelated to the drafter. vLLM also logs "no KV cache group could be identified as the draft model's"; see vLLM issue [#57032](https://github.com/vllm-project/vllm/issues/57032).

DFlash K=1 also showed an unexplained first-position accept of 0.28, against 0.83–0.87 at K≥3. It is probably a separate issue in the K=1 path.

## Inference block length vs. per-position accept

The same block-16-trained checkpoint, run at different K (c=10):

| K | Position 1 | Position 2 | Position 3 | Position 4 | Position 5 |
|---|---|---|---|---|---|
| 5 | .878 | .757 | .655 | .573 | .504 |
| 15 | .829 | .682 | .558 | .469 | .396 |

Early positions get worse with the longer window, even though K=15 matches the training block. In the training logs, block-8 and block-16 runs reach the same per-position accuracy (pos 1: .932 vs. .930 at the same step count). So training with a shorter block does not help early positions. A block-8 checkpoint gave the same early-position accept at K=5 and K=7 (pos 1: .868 / .868). The K=15 drop therefore looks like an inference-side effect of long query blocks, not something the model learned.

Note for speculators users: `--block-size` is ignored with `--from-pretrained`, because the whole pretrained config is reused. Set `block_size` in the init config to train a different block size.

## Bottom line on this workload

At K=5, DFlash reached 1742 tok/s (accept length 4.32) against 2520 for MTP K=5 with the 64k drafter vocab, and 2832 with RecoverSSM. MTP is the better choice on this hybrid target as long as the KV-group alignment issue stands.
