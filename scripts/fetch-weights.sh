#!/usr/bin/env bash
# fetch-weights.sh — stage the Inkling-2-Small checkpoint (~380 GB) into the
# HF cache on this node. Run on every node, or run once on the head and
# rsync to the workers (much friendlier to your uplink):
#
#   for h in $RANK1_HOST $RANK2_HOST $RANK3_HOST; do
#     rsync -aP --info=progress2 "$HF_CACHE/hub/models--thinkingmachines--Inkling-2-Small/" \
#       "$h:$HF_CACHE/hub/models--thinkingmachines--Inkling-2-Small/"
#   done
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/.env"

MODEL="${MODEL:-thinkingmachines/Inkling-2-Small}"
DEST="${HF_CACHE:?set in .env}"

command -v hf >/dev/null 2>&1 || {
  echo "hf CLI not found: pip install -U huggingface_hub[cli]" >&2; exit 1; }

echo "downloading $MODEL -> $DEST (~380 GB; resumable, safe to re-run)"
HF_HUB_ENABLE_HF_TRANSFER=1 hf download "$MODEL"
echo "done. snapshot is under $DEST/hub/models--${MODEL/\//--}"
