#!/bin/bash
# Qwen3.5-4B MTP K x RecoverSSM on/off, Spec-Bench, c=1/8/32 per server.
cd "$(dirname "$0")"
for K in 1 3 5 7; do for R in 0 1; do
  echo "===== K=$K REPLAY=$R $(date +%H:%M)"
  K=$K REPLAY=$R RUNS="1:80 8:240 32:480" bash bench_one.sh | grep -E "^---|Output token|Mean TTFT|Mean TPOT|accept|server died"
done; done
echo SWEEP_DONE
