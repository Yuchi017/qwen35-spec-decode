#!/bin/bash
# One server config x one Spec-Bench run. Starts a fresh server, benches, reads the
# spec-decode counters around the run, stops the server.
#   K=5 REPLAY=1 RUNS="1:80 8:240 32:480" bash bench_one.sh
cd "$(dirname "$0")"
K=${K:-5}; REPLAY=${REPLAY:-1}; GPU=${GPU:-1}; PORT=${PORT:-8001}
# "C:N" pairs, measured in order on one server
RUNS=${RUNS:-"32:480"}
SNAME=${NAME_PREFIX:+${NAME_PREFIX}_}k${K}_r${REPLAY}
tmux kill-session -t q35 2>/dev/null
for _ in $(seq 60); do
  [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU)" -lt 1000 ] && break; sleep 3
done
tmux new -d -s q35 "GPU=$GPU PORT=$PORT K=$K REPLAY=$REPLAY MODEL_PATH=${MODEL_PATH:-} bash start_server.sh 2>&1 | tee logs/server_$SNAME.log"
for _ in $(seq 180); do
  curl -sf -m2 localhost:$PORT/health >/dev/null && break
  tmux has-session -t q35 2>/dev/null || { echo "server died"; tail -20 logs/server_$SNAME.log; exit 1; }
  sleep 5
done
counters() {
  curl -s localhost:$PORT/metrics | awk '/^vllm:spec_decode_num_(drafts|draft_tokens|accepted_tokens|accepted_tokens_per_pos)_total/ {print $1"="$2}
                                          /^vllm:prefix_cache_(hits|queries)_total/ {print $1"="$2}'
}
# Warmup (not recorded): first-use Triton JITs (e.g. the RecoverSSM postprocess kernel) and graph paths.
${VLLM_CLIENT:-vllm} bench serve --backend openai-chat --endpoint /v1/chat/completions \
  --base-url http://localhost:$PORT --model qwen35-4b --tokenizer Qwen/Qwen3.5-4B \
  --dataset-name spec_bench --dataset-path data/spec_bench_question.jsonl \
  --num-prompts 64 --max-concurrency 32 --temperature 0 --seed 1 >/dev/null 2>&1
for run in $RUNS; do
C=${run%:*}; N=${run#*:}; NAME=${SNAME}_c$C
echo "--- c=$C n=$N"
counters > logs/metrics_before_$NAME.txt
${VLLM_CLIENT:-vllm} bench serve --backend openai-chat --endpoint /v1/chat/completions \
  --base-url http://localhost:$PORT --model qwen35-4b --tokenizer Qwen/Qwen3.5-4B \
  --dataset-name spec_bench --dataset-path data/spec_bench_question.jsonl \
  --num-prompts $N --max-concurrency $C --temperature 0 --seed 0 \
  --save-result --result-dir results --result-filename $NAME.json \
  2>&1 | tee logs/bench_$NAME.log | grep -E "throughput|TTFT|TPOT|Successful"
counters > logs/metrics_after_$NAME.txt
python3 - logs/metrics_before_$NAME.txt logs/metrics_after_$NAME.txt <<'EOF'
import re, sys
def load(p):
    d = {}
    for line in open(p):
        k, v = line.strip().rsplit("=", 1)
        pos = re.search(r'position="(\d+)"', k)
        key = re.sub(r"\{.*", "", k) + (f"@{pos.group(1)}" if pos else "")
        d[key] = d.get(key, 0) + float(v)
    return d
b, a = load(sys.argv[1]), load(sys.argv[2])
g = lambda k: a.get(k, 0) - b.get(k, 0)
drafts = g("vllm:spec_decode_num_drafts_total")
acc = g("vllm:spec_decode_num_accepted_tokens_total")
q, h = g("vllm:prefix_cache_queries_total"), g("vllm:prefix_cache_hits_total")
pp = [g(f"vllm:spec_decode_num_accepted_tokens_per_pos_total@{i}") / max(drafts, 1) for i in range(16)]
pp = [round(x, 3) for x in pp if x > 0]
print(f"accept len {1 + acc / max(drafts, 1):.3f}  prefix hit {h / max(q, 1):.3f}  per-pos {pp}")
EOF
done
grep -m1 -o "attention block size to [0-9]*" logs/server_$SNAME.log
tmux kill-session -t q35
