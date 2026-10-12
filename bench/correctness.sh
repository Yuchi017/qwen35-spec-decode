#!/bin/bash
# Greedy correctness check of RecoverSSM against the stock spec-decode kernels on one build.
# Each run gets a fresh server so no prefix cache carries over.
#   1. round 1 + round 2 (own output appended) with logprobs: base vs RecoverSSM, tie analysis
#   2. decode-written blocks: 600 forced decode tokens, follow-up hits those blocks (phase a),
#      vs the same follow-up on a fresh server where the state comes from prefill (phase b)
cd "$(dirname "$0")"
mkdir -p results logs
GPU=${GPU:-1}; PORT=8001
serve() {  # <replay 0|1> <log name>
  tmux kill-session -t q35 2>/dev/null
  for _ in $(seq 60); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU)" -lt 1000 ] && break; sleep 3; done
  tmux new -d -s q35 "GPU=$GPU PORT=$PORT K=${K:-5} REPLAY=$1 bash start_server.sh 2>&1 | tee logs/server_$2.log"
  for _ in $(seq 180); do curl -sf -m2 localhost:$PORT/health >/dev/null && return; sleep 5; done
}
for cfg in base:0 recover:1; do
  name=${cfg%:*}; serve ${cfg#*:} exact_$name
  python exact_match.py lp ${name}_lp 30
done
python exact_match.py cmp base_lp recover_lp
python exact_match.py margin base_lp recover_lp
for cfg in base:0 recover:1; do
  name=${cfg%:*}
  for phase in a b; do serve ${cfg#*:} selfhit_${name}_$phase; python exact_match.py selfhit $name 15 $phase; done
done
tmux kill-session -t q35
