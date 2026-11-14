#!/usr/bin/env bash
# inkling2-tp4.sh <rank> — Inkling-2-Small TP4 (+EP4 experts) across 4 DGX Sparks.
#
# Launch order: workers first (3, 2, 1), then rank 0 on the head. Rank 0
# waits for API readiness and prints the bench hint.
#
# Expects .env next to this script's repo root (see config/cluster.env.example).
set -euo pipefail

RANK="${1:?usage: inkling2-tp4.sh <rank 0..3>}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/.env"

: "${MASTER_ADDR:?set in .env}" "${MASTER_PORT:=29500}"
MODEL="${MODEL:-thinkingmachines/Inkling-2-Small}"
SERVED="${SERVED_MODEL_NAME:-inkling-2-small}"
IMAGE="${INKLING_IMAGE:-vllm/vllm-openai:inkling-2-small}"
GMU="${GMU:-0.85}"
MAXLEN="${MAXLEN:-262144}"
SEQS="${SEQS:-4}"
MAX_BATCHED="${MAX_BATCHED:-8192}"
KV_DTYPE="${KV_DTYPE:-auto}"
MM_LIMIT="${MM_LIMIT:-2}"
PORT="${VLLM_PORT:-8000}"

ARGS=(
  "$MODEL"
  --served-model-name "$SERVED"
  --tensor-parallel-size 4
  --nnodes 4
  --node-rank "$RANK"
  --master-addr "$MASTER_ADDR"
  --master-port "$MASTER_PORT"
  --distributed-executor-backend mp
  --gpu-memory-utilization "$GMU"
  --max-model-len "$MAXLEN"
  --max-num-seqs "$SEQS"
  --max-num-batched-tokens "$MAX_BATCHED"
  --kv-cache-dtype "$KV_DTYPE"
  --enable-prefix-caching
  --enable-chunked-prefill
  --limit-mm-per-prompt "{\"image\":$MM_LIMIT}"
  --mm-processor-cache-gb 0
  --host 0.0.0.0
  --port "$PORT"
)
[ "${REASONING:-1}" = "1" ] && ARGS+=(--reasoning-parser inkling)
[ "${TOOLS:-0}" = "1" ] && ARGS+=(--tool-call-parser inkling --enable-auto-tool-choice)
if [ "${DSPARK:-1}" = "1" ]; then
  ARGS+=(--speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}')
fi
# shellcheck disable=SC2206
[ -n "${VLLM_EXTRA:-}" ] && ARGS+=($VLLM_EXTRA)

ENV_PAIRS=(
  -e "NCCL_IB_HCA=${NCCL_IB_HCA:?set in .env}"
  -e "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:?set in .env}"
  # The executor's gloo control-plane derives the rank's advertise address
  # from the default interface; each node's hostname maps to 127.0.0.1 in
  # /etc/hosts, so without this the full-mesh connect dies on localhost.
  -e "GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-wlP9s9}"
  -e "NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3}"
  -e "NCCL_IB_AUTO_DETECT=0"
  -e "NCCL_DEBUG=${NCCL_DEBUG:-WARN}"
  -e "VLLM_HOST_IP=${VLLM_HOST_IP:?set per-node in .env}"
  -e "HF_HUB_OFFLINE=0"
)
# shellcheck disable=SC2206
[ -n "${NCCL_EXTRA:-}" ] && ENV_PAIRS+=($NCCL_EXTRA)

echo "[rank $RANK] $SERVED <- $IMAGE  maxlen=$MAXLEN seqs=$SEQS gmu=$GMU kv=$KV_DTYPE"
exec docker run --rm --name "inkling2-tp4-rank$RANK" \
  --gpus all --privileged --ipc=host --network host \
  "${ENV_PAIRS[@]}" \
  -v "${HF_CACHE:?set in .env}:/root/.cache/huggingface" \
  "$IMAGE" "${ARGS[@]}"
