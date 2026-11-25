#!/usr/bin/env bash
# Wave-V16 Crown-Jewel Permanent Checkpoint Master SM121a Build Script
# Run on ONE quiet GB10 node during live qualification window, then distribute $OUT/*.so + SHA256SUMS to /home/<user>/inkling-tp8/rdma/ on 8/8.
# Strict -gencode arch=compute_121a,code=sm_121a --fmad=false --ptxas-options=-v & 0-spill hard gates across all final SM121a libraries.
set -euo pipefail
ROOT="${1:-$(cd "$(dirname "$0")" && pwd)}"
OUT="${2:-/tmp/v16_built}"
mkdir -p "$OUT"

echo "[1/7] Building 48L LPT + L2::evict_first MoE V16 (nvfp4 c1/c48 + mx7 c1/c48)..."
bash "$ROOT/moe_v16/build_moe_v16.sh" "$OUT"

echo "[2/7] Building Decode-Only BF11 V13..."
bash "$ROOT/bf11_v13/build_bf11_v13.sh" "$OUT"

echo "[3/7] Building Drafter U4 C18 V14 + CertHead/Router V14..."
bash "$ROOT/drafter_u4_v14/build_drafter_u4_v14.sh" "$OUT"
bash "$ROOT/certhead_router_v14/build_certhead_router_v14.sh" "$OUT"

echo "[4/7] Building Unbiased Zero-Sync temp>0 Rejection Sampler V16..."
bash "$ROOT/sync_free_temp_v16/build_sync_free_temp_v16.sh" "$OUT"

echo "[5/7] Building Register-Fused BF12 RDMA V16 + Dual-Regime SConv/AttnRes/RMSNorm V16..."
bash "$ROOT/rdma_sconv_v16/build_rdma_sconv_v16.sh" "$OUT"

echo "[6/7] Building Prefill Dense-FP8 / FP8-Attn / RHD-Comm V13 base libs..."
bash "$ROOT/pf_v13_base/build_pf_v13.sh" "$OUT"

echo "[7/7] Building Sub-60s Zero-Spin Prefill MoE V16 + 3-Slot cp.async Global Attn V16 (-j4)..."
ATTN_V13_DIR="$ROOT/pf_v13_base/attn_v13" \
V14_ATTN_PATCH="$ROOT/swa_v14_base/attn_v13_swa_c28.patch" \
V16_ATTN_PATCH="$ROOT/pf_attn_v16/attn_global_v16.patch" \
  bash "$ROOT/pf_attn_v16/build_pf_attn_v16.sh" "$OUT"

echo "=== Generating SHA256SUMS manifest ==="
(cd "$OUT" && sha256sum *.so > SHA256SUMS)
ls -lh "$OUT"/*.so "$OUT/SHA256SUMS"
echo "=== WAVE-V16 MASTER BUILD COMPLETE ==="
