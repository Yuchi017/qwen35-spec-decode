#!/bin/bash
# Qwen3.5-4B + its own MTP head on vLLM v0.31.0 + patches/vllm-v0.31.0-gdn-recoverssm.patch.
# REPLAY=1 adds --use-replayssm; REPLAY=0 is the same build
# with the stock spec-decode kernels.
#   GPU=1 K=5 REPLAY=1 bash start_server.sh   (MODEL_PATH= to serve a stitched checkpoint)
export PATH=${VLLM_ENV:-$HOME/envs/vllm-gdn-recoverssm}/bin:$PATH  # env with the patched vLLM; its ninja is needed for JIT
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}  # 1 if the weights are cached and the box is offline
REPLAY_ARGS=()
[[ ${REPLAY:-1} == 1 ]] && REPLAY_ARGS=(--use-replayssm)
CUDA_VISIBLE_DEVICES=${GPU:-1} vllm serve ${MODEL_PATH:-Qwen/Qwen3.5-4B} \
    --speculative-config "{\"method\": \"mtp\", \"num_speculative_tokens\": ${K:-5}}" \
    "${REPLAY_ARGS[@]}" \
    --served-model-name qwen35-4b \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.9 \
    --max-num-seqs 100 \
    --max-num-batched-tokens 8192 \
    --max-model-len 32768 \
    --enable-prefix-caching \
    --mamba-cache-mode align \
    --prefix-match-unit 16 \
    --mamba-cache-dtype bfloat16 \
    --mamba-ssm-cache-dtype bfloat16 \
    --mamba-backend triton \
    --attention-backend ${ATTN_BACKEND:-TRITON_ATTN} \
    --language-model-only \
    --port ${PORT:-8001} ${EXTRA_ARGS:-}
