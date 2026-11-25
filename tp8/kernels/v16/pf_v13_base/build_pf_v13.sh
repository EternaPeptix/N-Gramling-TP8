#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# =============================================================================
# Wave-V13 Offline-Ready SM121a Build Script (Part A Decode Attn + Part B Prefill)
# Builds ALL 7 shared libraries in parallel (-j4) with STRICT SM121a flags:
#   -gencode arch=compute_121a,code=sm_121a --fmad=false --ptxas-options=-v
# Hard gates:
#   - ZERO --use_fast_math, ZERO --ftz=true, ZERO -arch=sm_121a
#   - ZERO stack frame, ZERO spill stores, ZERO spill loads across ALL kernels
#   - Full nm -D C ABI symbol verification for every .so
#   - NEVER writes to /rdma (outputs strictly into OUT_DIR)
#
# Usage on GB10 build node:
#   ./build_pf_v13.sh [OUT_DIR]
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ATTN_DIR="${SCRIPT_DIR}/attn_v13"
PF_DIR="${SCRIPT_DIR}/pf_v13"
OUT_DIR="${1:-${SCRIPT_DIR}/build_v13_out}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
JOBS="${JOBS:-4}"

if [[ "${OUT_DIR}" == /rdma* ]]; then
    echo "ERROR: Refusing to write directly into /rdma (${OUT_DIR}). Stage locally first!" >&2
    exit 1
fi

mkdir -p "${OUT_DIR}" "${OUT_DIR}/logs"

NVCC_FLAGS=(
    -O3
    -std=c++17
    --shared
    -Xcompiler -fPIC
    -gencode arch=compute_121a,code=sm_121a
    --fmad=false
    --ptxas-options=-v
)

echo "========================================================================"
echo "Wave-V13 SM121a Parallel Build (Part A + Part B)"
echo "  NVCC    : ${NVCC}"
echo "  OUT_DIR : ${OUT_DIR}"
echo "  JOBS    : ${JOBS}"
echo "========================================================================"

# ---- PART A: Split-TU Fused Decode Attention v13 (fixes V11 cicc hang!) ----
echo "[Part A] Building libfused_decode_attn_exact_sm121_v13.so (5-TU parallel -j${JOBS})..."
NVCC="${NVCC}" JOBS="${JOBS}" "${ATTN_DIR}/build_attn_v13.sh" "${OUT_DIR}"

# ---- PART B: 6 Zero-Spin Prefill Libraries (parallel -j${JOBS}) ----
declare -a PF_TARGETS=(
    "pf_nvfp4_w13_fused_v13.cu:libpf_nvfp4_w13_fused_v13.so"
    "pf_nvfp4_w2_v13.cu:libpf_nvfp4_w2_v13.so"
    "pf_mx7_prefill_v13.cu:libpf_mx7_prefill_v13.so"
    "pf_dense_fp8_tc_v13.cu:libpf_dense_fp8_tc_v13.so"
    "pf_rhd_comm_v13.cu:libpf_rhd_comm_v13.so"
    "pf_fp8_attn_v13.cu:libpf_fp8_attn_v13.so"
)

build_one_pf() {
    local src="$1"
    local so="$2"
    local log="${OUT_DIR}/logs/${so}.ptxas.log"
    echo "[Part B] Compiling ${src} -> ${so} ..."
    "${NVCC}" "${NVCC_FLAGS[@]}" "${PF_DIR}/${src}" -o "${OUT_DIR}/${so}" 2>&1 | tee "${log}" >/dev/null
    echo "[Part B] Done: ${so}"
}

echo "[Part B] Launching parallel nvcc builds (max ${JOBS} concurrent)..."
declare -a PIDS=()
for item in "${PF_TARGETS[@]}"; do
    src="${item%%:*}"
    so="${item##*:}"
    build_one_pf "${src}" "${so}" &
    PIDS+=("$!")
    while (( $(jobs -pr | wc -l) >= JOBS )); do
        sleep 0.2
    done
done

FAIL=0
for pid in "${PIDS[@]}"; do
    if ! wait "${pid}"; then
        FAIL=1
    fi
done
if (( FAIL != 0 )); then
    echo "ERROR: One or more Part B nvcc compiles failed! Inspect ${OUT_DIR}/logs/" >&2
    exit 1
fi

# ---- HARD GATE 1: 0 B stack frame / 0 B spill stores / 0 B spill loads ----
echo "[Gate 1] Verifying 0 B stack / 0 B spills across all Part B ptxas logs..."
for item in "${PF_TARGETS[@]}"; do
    so="${item##*:}"
    log="${OUT_DIR}/logs/${so}.ptxas.log"
    if grep -E '([1-9][0-9]* bytes stack frame|[1-9][0-9]* bytes spill stores|[1-9][0-9]* bytes spill loads)' "${log}"; then
        echo "ERROR: Non-zero stack/spills detected in ${log}!" >&2
        exit 1
    fi
    echo "  [OK] ${so}: 0 B stack, 0 B spills"
done

# ---- HARD GATE 2: Exported C ABI symbol check via nm -D ----
check_syms() {
    local so_path="$1"
    shift
    local syms
    syms="$(nm -D --defined-only "${so_path}" | awk '{print $3}')"
    for sym in "$@"; do
        if ! grep -qx "${sym}" <<<"${syms}"; then
            echo "ERROR: Missing C ABI symbol '${sym}' in ${so_path}!" >&2
            exit 1
        fi
    done
    echo "  [OK] $(basename "${so_path}"): verified $# exported C ABI symbols"
}

echo "[Gate 2] Verifying exported C ABI symbols across all 6 Part B libraries..."
check_syms "${OUT_DIR}/libpf_nvfp4_w13_fused_v13.so" \
    pf_nvfp4_w13_v13_version pf_nvfp4_w13_v6_version \
    pf_nvfp4_w13_v6_slot_bytes pf_nvfp4_w13_v6_state_bytes \
    pf_nvfp4_build_w13_w2_plan_v6 pf_nvfp4_w13_situ_quant_v6 \
    pf_nvfp4_build_m128_worklist pf_nvfp4_w13_ws5_v4b_unfused \
    pf_nvfp4_w13_ws5_v4b_fused_situ pf_route_tokens_place_v6

check_syms "${OUT_DIR}/libpf_nvfp4_w2_v13.so" \
    pf_nvfp4_w2_v13_version pf_nvfp4_w2_v5_version \
    pf_nvfp4_w2_v5_set_reserve_sms pf_nvfp4_w2_v5_get_reserve_sms \
    pf_nvfp4_w2_build_plan_v5 pf_nvfp4_w2_ws5_v5_unfused \
    pf_nvfp4_w2_ws5_v5_combine pf_nvfp4_w2_panel_v5_combine

check_syms "${OUT_DIR}/libpf_mx7_prefill_v13.so" \
    pf_mx7_v13_version pf_mx7_v10_version \
    pf_mx7_v10_set_reserve_sms pf_mx7_v10_get_reserve_sms \
    pf_mx7_build_expert_plan pf_mx7_zero_slice_wb_bf16 \
    pf_mx7_w13_prefill_fused_situ pf_mx7_w13_prefill_unfused \
    pf_mx7_w2_prefill_persistent pf_mx7_build_plan_v10 \
    pf_mx7_w2_v10_unfused pf_mx7_w2_v10_combine

check_syms "${OUT_DIR}/libpf_dense_fp8_tc_v13.so" \
    pf_dense_fp8_tc_v13_version pf_dense_fp8_tc_v11_version \
    pf_quant_weight_f16safe_v11_fwd pf_dense_fp8_f16acc_v11_fwd \
    pf_wo_ud_bf16_to_bf16_v11_fwd pf_wo_ud_fused_a2a_int8_v11_fwd

check_syms "${OUT_DIR}/libpf_rhd_comm_v13.so" \
    pf_rhd_comm_v13_version pf_rhd_comm_v11_version \
    fused_sconv_ar_fp8_ag_v13_fwd fused_sconv_ar_fp8_ag_v11_fwd \
    a2a_quant_int8_g64_lowsm_fwd a2a_quant_fp8_g64_lowsm_fwd \
    a2a_dequant_sum8_int8_g64_lowsm_fwd a2a_dequant_sum8_fp8_g64_lowsm_fwd \
    a2a_quant_int8_g64_slice_lowsm_fwd a2a_quant_fp8_g64_slice_lowsm_fwd \
    a2a_dequant_sum8_int8_g64_slice_lowsm_fwd a2a_dequant_sum8_fp8_g64_slice_lowsm_fwd \
    rhd_ag_quant_fp8_rowwise_lowsm_fwd rhd_ag_unpack_fp8_rowwise_lowsm_fwd \
    rhd_ag_dequant_fp8_rowwise_lowsm_fwd rhd_ag_dequant_int8_g64_lowsm_fwd \
    rhd_ag_dequant_fp8_g64_lowsm_fwd

check_syms "${OUT_DIR}/libpf_fp8_attn_v13.so" \
    pf_fp8_attn_v13_version \
    local_swa_fp8_otf_rel_v6_sm121_fwd local_swa_otf_rel_bf16_sm121_fwd \
    global_tail_fp8_otf_rel_v6_sm121_fwd global_tail_otf_rel_bf16_sm121_fwd

echo "========================================================================"
echo "ALL 7 WAVE-V13 SM121a LIBRARIES BUILT & VERIFIED IN ${OUT_DIR}!"
echo "========================================================================"
