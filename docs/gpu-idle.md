# Where GPU idle time comes from (vLLM 0.29 / 0.31 × FlashInfer / Triton)

Internal workload, c=30, L40S, nsys. Average ms per step. "Idle" is the wall time inside a step when no kernel is running. MTP runs use the TTT head with the full drafter vocabulary.

## Decode-only steps

| | 0.29 FI | 0.29 Triton | 0.31 FI | 0.31 Triton |
|---|---|---|---|---|
| No spec decode: kernels / idle / step | 13.6 / 1.2 / 14.8 | 14.1 / 0.4 / 14.6 | 13.7 / 0.4 / 14.1 | 13.9 / 0.3 / 14.1 |
| MTP K=5: kernels / idle / step | 38.5 / **9.7** / 48.2 | 39.4 / **0.2** / 39.6 | 38.1 / **10.0** / 48.1 | 39.8 / **0.1** / 39.9 |

## Steps that include a prefill

| | 0.29 FI | 0.29 Triton | 0.31 FI | 0.31 Triton |
|---|---|---|---|---|
| No spec decode: kernels / idle / step | 24.4 / **13.4** / 37.8 | 24.4 / **32.6** / 57.1 | 22.6 / **32.3** / 54.9 | 24.2 / **21.7** / 46.0 |
| MTP K=5: kernels / idle / step | 72.2 / 26.4 / 98.6 | 74.9 / 1.6 / 76.5 | 64.1 / 23.1 / 87.2 | 66.7 / 1.0 / 67.7 |

## End-to-end throughput, relative to 0.31 Triton without spec decode

| | 0.29 FI | 0.29 Triton | 0.31 FI | 0.31 Triton |
|---|---|---|---|---|
| No spec decode | 1.12× | 0.92× | 0.94× | 1.00× |
| MTP K=5 | 1.35× | 1.46× | 1.48× | **1.62×** |

## Reading

- **Without spec decode**, the differences are almost entirely idle time in prefill steps. Those run eagerly (~750 launches per step), so the CPU needs 41–64 ms per step for ~24 ms of GPU work. The allocator path (`allocate_slots`) is about 2.4× slower in three of the four builds, and those three are the slow ones.
- **With MTP**, decode steps on FlashInfer gain ~10 ms of idle at c=30, versus ~5.7 ms at c=20, in both 0.29 and 0.31. Triton stays near zero. The extra GPU work per step also hides most of the CPU launch time in Triton's prefill steps. So with MTP the backend matters more than the version, and the ranking flips compared with no spec decode.
- **Between versions**, 0.31's lower prefill kernel time with MTP (64–67 vs. 72–75 ms per step) matches its higher prefix hit rate; see [prefix-cache-and-mtp.md](prefix-cache-and-mtp.md).
- **Disabling async scheduling** made no-spec-decode throughput 28% worse, not better.
