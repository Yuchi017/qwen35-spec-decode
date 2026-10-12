# Validating a kernel change that is not bit-exact

RecoverSSM computes the same math as the stock kernel in a different order:
- the commit uses the closed form instead of updating step by step;
- q/k are normalized with `x / sqrt(Σx² + ε)` instead of `x · rsqrt(Σx² + ε)`.

The fp32 rounding therefore differs before the state is stored as bf16. Greedy outputs cannot be expected to match token for token, so exact-match is the wrong test. These are the tests we used instead ([`bench/exact_match.py`](../bench/exact_match.py), [`bench/correctness.sh`](../bench/correctness.sh)).

## 1. Unit test

`tests/kernels/test_gdn_recoverssm.py` in the vLLM patch checks verify + commit against the stock kernel's per-position states:
- mixed window lengths and per-row commit lengths;
- fp32 and bf16 state;
- tolerance 1e-4 for fp32 state and 2e-2 for bf16.

## 2. The stock kernels are deterministic

On one build with `--use-replayssm` off, two runs on fresh servers (one request at a time, temperature 0, 30 prompts, two turns) gave identical outputs: 30/30 for each turn. Any divergence after that comes from the kernel change, not from noise.

## 3. Divergences happen only at near-ties

With top-5 logprobs recorded at every position, we look at where RecoverSSM first diverges from the stock kernels and measure the stock run's top-1 − top-2 gap at that position:

| | Divergent requests | Gap at the first divergence | All positions |
|---|---|---|---|
| Turn 1 | 21 / 30 | ≤ 0.875, median 0.25 | median 11.1; only 3.7% ≤ 0.5 |
| Turn 2 (own output appended) | 8 / 9 comparable | ≤ 0.5 (RecoverSSM side ≤ 0.625) | median 11.9; only 2.6% ≤ 0.5 |

Logits are bf16, so a gap of 0.125–0.25 is 1–2 ulps at this scale. A real bug, such as a wrong state, would diverge at confident positions too.

## 4. Blocks written during decode

Turn 2 above rarely reached a block whose state the commit wrote at a boundary, because outputs were short. To cover that path:
- **Phase a:** decode 600 forced tokens (`ignore_eos`), which crosses at least one block. Then send the follow-up on the same server; its prefix hits those blocks.
- **Phase b:** send the same follow-up on a fresh server, where the state comes from prefill.
- Do both with the stock kernels and with RecoverSSM. Both runs hit the same number of cached tokens.

| | Hit vs. cold, identical | Gap at the first divergence |
|---|---|---|
| Stock | 7 / 15 | 0.125 – 1.75 |
| RecoverSSM | 3 / 15 | 0 – 1.0 (mostly 0.125) |

Decode-written state differs from prefill-computed state by rounding even with the stock kernels. RecoverSSM's divergences are at near-ties as well.

## 5. Load test

At 30 concurrent clients, 90 requests:
- no degenerate or repetitive outputs (zlib compression-ratio check);
- accept length unchanged.

An earlier bug in the port (committing to block `n // block_size` instead of `(n − 1) // block_size` at exact block boundaries) showed up here as repeated-token outputs, with accept length falling to 3.1–3.8. That makes this a useful smoke test.
