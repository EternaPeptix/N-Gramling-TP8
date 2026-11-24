#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Strict SM121a (NVIDIA GB10) build script for DSpark Drafter V10 CUDA libraries:
#   1. libdspark_markov_v10.so (100% drop-in ABI superset of libdspark_markov_v7.so
#      with 0 #206 PDL hazards across all 28 kernels, native TC48 HAS_SKIP + Batch-All ES
#      cross-step fusion, and TOP2 + HAS_ID_MAP for B==1 TREE_SPEC + VOCAB_HOT)
#   2. libdrafter_attn_v10.so  (100% drop-in ABI superset of libdrafter_attn_v5.so
#      with __launch_bounds__(192, 2) 2-CTA/SM occupancy and MAX_SPLITS=48 for B=1@128K Global)
#
# Strict SM121a Hygiene Gate:
#   - -O3 -std=c++17 -gencode arch=compute_121a,code=sm_121a --fmad=false --ptxas-options=-v
#   - NO --use_fast_math, NO --ftz=true
#   - Hard-fail on ANY stack frame > 0 B or spill store/load > 0 B
#   - Hard-fail on ANY missing extern "C" ABI export
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${1:-${SCRIPT_DIR}/build}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
mkdir -p "${OUT_DIR}"

if [[ ! -x "${NVCC}" ]]; then
    echo "ERROR: nvcc not found at ${NVCC} (override with NVCC=/path/to/nvcc)" >&2
    exit 1
fi

COMMON_FLAGS=(
    -O3
    -std=c++17
    -gencode arch=compute_121a,code=sm_121a
    --fmad=false
    --shared
    -Xcompiler -fPIC
    --ptxas-options=-v
)

check_ptxas_zero_spill() {
    local name="$1"
    local log_file="$2"
    echo "--- ptxas resource report: ${name} ---"
    grep -E "Compiling entry function|Used [0-9]+ registers|bytes stack frame|bytes spill" "${log_file}" || true

    if grep -Eq "([1-9][0-9]* bytes stack frame|[1-9][0-9]* bytes spill stores|[1-9][0-9]* bytes spill loads)" "${log_file}"; then
        echo "FAIL [${name}]: non-zero stack frame or register spill detected!" >&2
        grep -E "Compiling entry function|bytes stack frame|bytes spill" "${log_file}" >&2
        exit 1
    fi
    echo "PASS [${name}]: 0 B stack frame, 0 B spill stores, 0 B spill loads."
}

check_exports() {
    local so_file="$1"
    shift
    local missing=0
    local syms
    syms="$(nm -D --defined-only "${so_file}" | awk '{print $3}')"
    for sym in "$@"; do
        if ! grep -qx "${sym}" <<<"${syms}"; then
            echo "FAIL [${so_file}]: missing required extern \"C\" export: ${sym}" >&2
            missing=1
        fi
    done
    if [[ "${missing}" -ne 0 ]]; then
        exit 1
    fi
    echo "PASS [${so_file}]: all $# required extern \"C\" symbols exported."
}

# 1. Build libdspark_markov_v10.so
MARKOV_SRC="${SCRIPT_DIR}/dspark_markov_v10.cu"
MARKOV_SO="${OUT_DIR}/libdspark_markov_v10.so"
MARKOV_LOG="${OUT_DIR}/build_dspark_markov_v10.ptxas.log"

echo "=== Building libdspark_markov_v10.so ==="
"${NVCC}" "${COMMON_FLAGS[@]}" "${MARKOV_SRC}" -o "${MARKOV_SO}" 2>&1 | tee "${MARKOV_LOG}"
check_ptxas_zero_spill "libdspark_markov_v10.so" "${MARKOV_LOG}"
check_exports "${MARKOV_SO}" \
    dspark_markov_v10_version \
    dspark_markov_rnn_step_v10_fwd \
    dspark_fold_argmax_v10_fwd \
    dspark_select_and_rnn_step_v10_fwd \
    dspark_argmax_select_batch_es_v10_fwd \
    dspark_embed_block_ln_v7_fwd \
    dspark_mix_rmsnorm_v7_fwd \
    dspark_context_anchor_mix_rmsnorm_v7_fwd \
    dspark_mix_rmsnorm_5layer_v7_fwd \
    dspark_qk_norm_rope_write_kv_v7_fwd \
    dspark_silu_mul_v7_fwd \
    dspark_quant_fp8_rowwise_v7_fwd \
    dspark_silu_mul_quant_fp8_v7_fwd \
    dspark_add_softcap_v7_fwd \
    dspark_markov_gate_v7_fwd \
    dspark_markov_rnn_step_v7_fwd \
    dspark_fold_argmax_v7_fwd \
    dspark_argmax_select_v7_fwd \
    dspark_select_and_rnn_step_v7_fwd \
    dspark_select_last_and_vw_conf_v7_fwd \
    dspark_select_top2_pack_and_rnn_v7_fwd \
    dspark_select_beam2_and_rnn_v7_fwd \
    dspark_select_last_and_tree_pack_v7_fwd \
    sync_free_local_pairs_v7_fwd \
    sync_free_k0_tok_status_v7_fwd \
    sync_free_fused_reject_v7_fwd \
    sync_free_fused_tree_reject_v7_fwd \
    dspark_vw_update_dink_v7_fwd \
    dspark_fused_epilogue_prep_v7_fwd

# 2. Build libdrafter_attn_v10.so
ATTN_SRC="${SCRIPT_DIR}/drafter_attn_v10.cu"
ATTN_SO="${OUT_DIR}/libdrafter_attn_v10.so"
ATTN_LOG="${OUT_DIR}/build_drafter_attn_v10.ptxas.log"

echo "=== Building libdrafter_attn_v10.so ==="
"${NVCC}" "${COMMON_FLAGS[@]}" "${ATTN_SRC}" -o "${ATTN_SO}" 2>&1 | tee "${ATTN_LOG}"
check_ptxas_zero_spill "libdrafter_attn_v10.so" "${ATTN_LOG}"
check_exports "${ATTN_SO}" \
    drafter_attn_v10_version \
    drafter_attn_v5_fwd \
    drafter_attn_c1_v4_fwd

echo "=== BUILD COMPLETE ==="
ls -lh "${MARKOV_SO}" "${ATTN_SO}"
