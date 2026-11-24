#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Wave-V10 SM121a (GB10) Production KV-BF12 (BF12-G128-OVF-SAFE) Build Script
#
# Compiles BOTH production SM121a shared libraries:
#   1. libkv_bf12_paged_v10.so               (kv_bf12_paged_v10.cu)
#   2. libfused_decode_attn_exact_sm121_v10.so (fused_decode_attn_exact_sm121_v10.cu)
#
# Strict SM121a Hygiene Gates:
#   - Mandatory `-gencode arch=compute_121a,code=sm_121a --fmad=false --ptxas-options=-v`
#   - Strictly NO `--use_fast_math`, NO `--ftz=true`, NO `-arch=sm_121a`
#   - Hard-fail on ANY non-zero stack frame, spill store, or spill load in ptxas output
#   - Hard-fail if ANY expected C ABI symbol is missing from `nm -D`
#   - Never writes directly to read-only `/rdma`; outputs to `$OUT_DIR` (default: script dir)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
mkdir -p "${OUT_DIR}"

COMMON_FLAGS=(
    -O3
    -std=c++17
    -gencode arch=compute_121a,code=sm_121a
    --fmad=false
    --shared
    -Xcompiler -fPIC
    --ptxas-options=-v
    -I"${SCRIPT_DIR}"
)

check_ptxas_zero_spills() {
    local label="$1"
    local log_file="$2"
    echo "[AUDIT] Checking ptxas stack/spill hygiene for ${label} ..."
    if grep -E '([1-9][0-9]* bytes stack frame|[1-9][0-9]* bytes spill stores|[1-9][0-9]* bytes spill loads)' "${log_file}"; then
        echo "[FAIL] Non-zero stack frame or register spill detected in ${label}!" >&2
        exit 1
    fi
    local max_regs
    max_regs="$(grep -oE 'Used [0-9]+ registers' "${log_file}" | awk '{if($2>m)m=$2} END{print m+0}')"
    local n_kernels
    n_kernels="$(grep -c 'Compiling entry function' "${log_file}" || true)"
    echo "[PASS] ${label}: 0 B stack / 0 B spills across ${n_kernels} compiled entry points (max ${max_regs} regs/thr)."
}

check_nm_exports() {
    local so_path="$1"
    shift
    local syms
    syms="$(nm -D --defined-only "${so_path}")"
    local missing=0
    for sym in "$@"; do
        if ! grep -qw "${sym}" <<< "${syms}"; then
            echo "[FAIL] Missing exported symbol '${sym}' in ${so_path}!" >&2
            missing=1
        fi
    done
    if [[ "${missing}" -ne 0 ]]; then
        exit 1
    fi
    echo "[PASS] ${so_path}: all $# exported C ABI symbols verified via nm -D."
}

# =========================================================================
# Target 1: libkv_bf12_paged_v10.so
# =========================================================================
PAGED_SRC="${SCRIPT_DIR}/kv_bf12_paged_v10.cu"
PAGED_SO="${OUT_DIR}/libkv_bf12_paged_v10.so"
PAGED_LOG="${OUT_DIR}/build_kv_bf12_paged_v10.ptxas.log"

echo "[BUILD 1/2] Compiling ${PAGED_SRC} -> ${PAGED_SO} ..."
"${NVCC}" "${COMMON_FLAGS[@]}" "${PAGED_SRC}" -o "${PAGED_SO}" 2>&1 | tee "${PAGED_LOG}"
check_ptxas_zero_spills "libkv_bf12_paged_v10.so" "${PAGED_LOG}"
check_nm_exports "${PAGED_SO}" \
    kv_bf12_version \
    kv_bf12_reset_blocks_v10_fwd \
    kv_bf12_store_prefill_v10_fwd \
    kv_bf12_unpack_blocks_v10_fwd \
    kv_bf12_unpack_range_v10_fwd

# =========================================================================
# Target 2: libfused_decode_attn_exact_sm121_v10.so
# =========================================================================
DECODE_SRC="${SCRIPT_DIR}/fused_decode_attn_exact_sm121_v10.cu"
DECODE_SO="${OUT_DIR}/libfused_decode_attn_exact_sm121_v10.so"
DECODE_LOG="${OUT_DIR}/build_fused_decode_attn_exact_sm121_v10.ptxas.log"

echo "[BUILD 2/2] Compiling ${DECODE_SRC} -> ${DECODE_SO} ..."
"${NVCC}" "${COMMON_FLAGS[@]}" "${DECODE_SRC}" -o "${DECODE_SO}" 2>&1 | tee "${DECODE_LOG}"
check_ptxas_zero_spills "libfused_decode_attn_exact_sm121_v10.so" "${DECODE_LOG}"
check_nm_exports "${DECODE_SO}" \
    fused_decode_attn_v7_version \
    fused_decode_attn_v8_version \
    fused_decode_attn_v9_version \
    fused_decode_attn_v10_version \
    fused_decode_attn_v9_init_smem \
    fused_decode_attn_v10_init_smem \
    attn_prologue_kvq_exact_fwd \
    attn_prologue_kvq_tree_exact_fwd \
    attn_prologue_kvq_bf12_exact_v10_fwd \
    attn_prologue_kvq_tree_bf12_exact_v10_fwd \
    fused_decode_attn_otf_mgs_fwd \
    fused_decode_attn_otf_mgs_bf12_v10_fwd \
    fused_decode_attn_otf_mgs_c32_fwd \
    fused_decode_attn_otf_mgs_c32_bf12_v10_fwd \
    fused_decode_attn_otf_mgs_tree_fwd \
    fused_decode_attn_otf_mgs_tree_bf12_v10_fwd \
    fused_decode_attn_otf_mgs_v9_fwd \
    fused_decode_attn_otf_mgs_v10_bf12_paired_fwd

echo "[OK] Wave-V10 KV-BF12 SM121a build complete: ${PAGED_SO} + ${DECODE_SO}"
