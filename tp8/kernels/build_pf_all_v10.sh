#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Wave-V10 Unified Prefill CUDA Build Script for NVIDIA DGX Spark GB10 (SM121a)
#
# Enforces mandatory SM121a compiler rules across ALL 9 prefill libraries:
#   - Strictly `-gencode arch=compute_121a,code=sm_121a --fmad=false`
#   - ZERO `--use_fast_math`, ZERO `--ftz=true`, ZERO `-arch=sm_121a`
#   - Hard-fails on ANY > 0 bytes stack frame, spill stores, or spill loads (`--ptxas-options=-v`)
#   - Verifies ALL exported `extern "C"` C ABI symbols via `nm -D`
#   - Builds DBO-safe `pf_nvfp4_w13_fused_v6_dbo.cu`, `pf_nvfp4_w2_v5_dbo.cu`,
#     and `pf_mx7_prefill_v10.cu` (1-launch co-resident MX7 W2 with DBO SM reserve)!
#
# Usage on GB10 node:
#   OUT_DIR=/tmp/pf_v10_out bash build_pf_all_v10.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCR_ROOT="${SCR_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
OUT_DIR="${OUT_DIR:-/tmp/pf_v10_out}"
LOG_DIR="${OUT_DIR}/ptxas_logs"
BUILD_V4="${BUILD_V4:-1}"
BUILD_V5="${BUILD_V5:-1}"
BUILD_V6="${BUILD_V6:-1}"
BUILD_V10="${BUILD_V10:-1}"
INSTALL_SAGE2_DROPIN="${INSTALL_SAGE2_DROPIN:-0}"
INSTALL_W2_DBO_DROPIN="${INSTALL_W2_DBO_DROPIN:-1}"

if [[ ! -x "${NVCC}" ]]; then
    NVCC="$(command -v nvcc || true)"
fi
if [[ -z "${NVCC}" || ! -x "${NVCC}" ]]; then
    echo "FATAL: nvcc not found (set NVCC=/usr/local/cuda/bin/nvcc)" >&2
    exit 1
fi

mkdir -p "${OUT_DIR}" "${LOG_DIR}"

NVCC_STRICT_FLAGS=(
    -O3
    -std=c++17
    --fmad=false
    --shared
    -Xcompiler -fPIC
    -gencode arch=compute_121a,code=sm_121a
    --ptxas-options=-v
)

check_zero_spills_and_stack() {
    local label="$1"
    local log_file="$2"
    python3 - "${label}" "${log_file}" <<'PYEOF'
import re, sys

label, log_path = sys.argv[1], sys.argv[2]
txt = open(log_path, "r", encoding="utf-8", errors="replace").read()
funcs = re.findall(r"Compiling entry function '([^']+)'", txt)
spills = re.findall(
    r"(\d+)\s+bytes\s+stack\s+frame,\s*(\d+)\s+bytes\s+spill\s+stores,\s*(\d+)\s+bytes\s+spill\s+loads",
    txt,
)
regs = re.findall(r"Used\s+(\d+)\s+registers", txt)
bad = []
for i, (sf, ss, sl) in enumerate(spills):
    sf_i, ss_i, sl_i = int(sf), int(ss), int(sl)
    if sf_i > 0 or ss_i > 0 or sl_i > 0:
        fn = funcs[i] if i < len(funcs) else f"kernel#{i}"
        rg = regs[i] if i < len(regs) else "?"
        bad.append(f"  FAIL {fn}: regs={rg}, stack={sf_i}B, spill_st={ss_i}B, spill_ld={sl_i}B")
max_r = max((int(r) for r in regs), default=0)
print(f"[ptxas-gate] {label}: {len(funcs)} entrypoints, max_regs={max_r}, spill_violations={len(bad)}")
if len(funcs) == 0:
    print(f"  FAIL {label}: 0 entry functions parsed from ptxas log!", file=sys.stderr)
    sys.exit(1)
if bad:
    print("\n".join(bad), file=sys.stderr)
    sys.exit(1)
PYEOF
}

check_symbols() {
    local so_path="$1"
    shift
    local syms
    syms="$(nm -D --defined-only "${so_path}" | awk '{print $3}')"
    local missing=0
    for sym in "$@"; do
        if ! grep -qx "${sym}" <<< "${syms}"; then
            echo "FATAL: Missing exported C ABI symbol '${sym}' in ${so_path}" >&2
            missing=1
        fi
    done
    if [[ "${missing}" -ne 0 ]]; then
        exit 1
    fi
    echo "[nm-gate] $(basename "${so_path}"): all $# exported C ABI symbols verified!"
}

build_one() {
    local label="$1"
    local src_cu="$2"
    local out_so="$3"
    shift 3
    local log_file="${LOG_DIR}/${label}.ptxas.log"
    echo "=== Building $(basename "${out_so}") from ${src_cu} ==="
    "${NVCC}" "${NVCC_STRICT_FLAGS[@]}" "${src_cu}" -o "${out_so}" 2>&1 | tee "${log_file}"
    check_zero_spills_and_stack "${label}" "${log_file}"
    check_symbols "${out_so}" "$@"
}

if [[ "${BUILD_V4}" == "1" ]]; then
    build_one "pf_mx7_prefill_sm121" \
        "${SCR_ROOT}/v4/pf_1m_speed/pf_mx7_prefill_sm121.cu" \
        "${OUT_DIR}/libpf_mx7_prefill_sm121.so" \
        pf_mx7_build_expert_plan \
        pf_mx7_zero_slice_wb_bf16 \
        pf_mx7_w13_prefill_fused_situ \
        pf_mx7_w13_prefill_unfused \
        pf_mx7_w2_prefill_persistent

    build_one "pf_nvfp4_w13_v4b" \
        "${SCR_ROOT}/v4/pf_1m_speed/pf_nvfp4_w13_v4b.cu" \
        "${OUT_DIR}/libpf_nvfp4_w13_v4b.so" \
        pf_nvfp4_build_m128_worklist \
        pf_nvfp4_w13_ws5_v4b_unfused \
        pf_nvfp4_w13_ws5_v4b_fused_situ \
        pf_nvfp4_quant_288

    build_one "global_attn_sage2_sm121_v5_zspill" \
        "${SCR_ROOT}/v4/pf_1m_speed/global_attn_sage2_sm121_v5_zspill.cu" \
        "${OUT_DIR}/libglobal_attn_sage2_sm121_v5_zspill.so" \
        global_sage2_prefix_sm121_fwd \
        global_sage2_prefix_v3_sm121_fwd \
        global_sage2_prefix_v5_sm121_fwd \
        score_kmax_blocks_fp8_sm121_fwd

    if [[ "${INSTALL_SAGE2_DROPIN}" == "1" ]]; then
        cp -f "${OUT_DIR}/libglobal_attn_sage2_sm121_v5_zspill.so" \
              "${OUT_DIR}/libglobal_attn_sage2_sm121_v5.so"
    fi

    build_one "pf_attn_nonmoe_v4_sm121" \
        "${SCR_ROOT}/v4/pf_attn_fuzz/pf_attn_nonmoe_v4_sm121.cu" \
        "${OUT_DIR}/libpf_attn_nonmoe_v4_sm121.so" \
        quantize_k_shadow_v4_sm121_fwd \
        score_topk_sort_fp8_v4_sm121_fwd \
        kvq_prologue_pf_v4_sm121_fwd \
        quant_fp8_rowwise_v4_sm121_fwd
fi

if [[ "${BUILD_V5}" == "1" ]]; then
    build_one "pf_rhd_comm_v5" \
        "${SCR_ROOT}/v5/pf_w2_comm/pf_rhd_comm_v5.cu" \
        "${OUT_DIR}/libpf_rhd_comm_v5.so" \
        pf_rhd_comm_v5_version \
        rhd_quant_slice_int8_g64_fwd \
        rhd_quant_slice_fp8_g64_fwd \
        rhd_dequant_add_split_quant_int8_g64_fwd \
        rhd_dequant_add_split_quant_fp8_g64_fwd \
        rhd_dequant_add_quant_int8_g64_fwd \
        rhd_dequant_add_quant_fp8_g64_fwd \
        rhd_dequant_add_bf16_int8_g64_fwd \
        rhd_dequant_add_bf16_fp8_g64_fwd \
        rhd_ag_dequant_int8_g64_fwd \
        rhd_ag_dequant_fp8_g64_fwd \
        rhd_ag_quant_fp8_rowwise_fwd \
        rhd_ag_unpack_fp8_rowwise_fwd \
        rhd_quant_slice_int8_g64_lowsm_fwd \
        rhd_quant_slice_fp8_g64_lowsm_fwd \
        rhd_dequant_add_split_quant_int8_g64_lowsm_fwd \
        rhd_dequant_add_split_quant_fp8_g64_lowsm_fwd \
        rhd_dequant_add_quant_int8_g64_lowsm_fwd \
        rhd_dequant_add_quant_fp8_g64_lowsm_fwd \
        rhd_dequant_add_bf16_int8_g64_lowsm_fwd \
        rhd_dequant_add_bf16_fp8_g64_lowsm_fwd \
        a2a_quant_int8_g64_lowsm_fwd \
        a2a_quant_fp8_g64_lowsm_fwd \
        a2a_dequant_sum8_int8_g64_lowsm_fwd \
        a2a_dequant_sum8_fp8_g64_lowsm_fwd \
        rhd_ag_quant_fp8_rowwise_lowsm_fwd \
        rhd_ag_unpack_fp8_rowwise_lowsm_fwd \
        rhd_ag_dequant_fp8_rowwise_lowsm_fwd \
        rhd_ag_dequant_int8_g64_lowsm_fwd \
        rhd_ag_dequant_fp8_g64_lowsm_fwd
fi

if [[ "${BUILD_V6}" == "1" ]]; then
    build_one "pf_fp8_attn_v6" \
        "${SCR_ROOT}/v6/pf_fp8_attn_moe/pf_fp8_attn_v6.cu" \
        "${OUT_DIR}/libpf_fp8_attn_v6.so" \
        pf_fp8_attn_v6_version \
        global_sage2_prefix_v6_sm121_fwd \
        global_sage2_prefix_v5_sm121_fwd \
        global_sage2_prefix_v3_sm121_fwd \
        global_sage2_prefix_sm121_fwd \
        score_kmax_blocks_fp8_sm121_fwd \
        local_swa_fp8_otf_rel_v6_sm121_fwd \
        local_swa_otf_rel_bf16_sm121_fwd \
        global_tail_fp8_otf_rel_v6_sm121_fwd \
        global_tail_otf_rel_bf16_sm121_fwd

    build_one "pf_nvfp4_w13_fused_v6" \
        "${SCR_ROOT}/v9/unified/cuda/pf_nvfp4_w13_fused_v6_dbo.cu" \
        "${OUT_DIR}/libpf_nvfp4_w13_fused_v6.so" \
        pf_nvfp4_w13_v6_version \
        pf_nvfp4_w13_v6_slot_bytes \
        pf_nvfp4_w13_v6_state_bytes \
        pf_nvfp4_build_w13_w2_plan_v6 \
        pf_nvfp4_w13_situ_quant_v6 \
        pf_nvfp4_build_m128_worklist \
        pf_nvfp4_w13_ws5_v4b_unfused \
        pf_nvfp4_w13_ws5_v4b_fused_situ \
        pf_route_tokens_place_v6
fi

if [[ "${BUILD_V10}" == "1" ]]; then
    build_one "pf_nvfp4_w2_v5_dbo" \
        "${SCRIPT_DIR}/pf_nvfp4_w2_v5_dbo.cu" \
        "${OUT_DIR}/libpf_nvfp4_w2_v5_dbo.so" \
        pf_nvfp4_w2_v5_version \
        pf_nvfp4_w2_v5_set_reserve_sms \
        pf_nvfp4_w2_v5_get_reserve_sms \
        pf_nvfp4_w2_build_plan_v5 \
        pf_nvfp4_w2_ws5_v5_unfused \
        pf_nvfp4_w2_ws5_v5_combine \
        pf_nvfp4_w2_panel_v5_combine

    if [[ "${INSTALL_W2_DBO_DROPIN}" == "1" ]]; then
        cp -f "${OUT_DIR}/libpf_nvfp4_w2_v5_dbo.so" \
              "${OUT_DIR}/libpf_nvfp4_w2_v5.so"
    fi

    build_one "pf_mx7_prefill_v10" \
        "${SCRIPT_DIR}/pf_mx7_prefill_v10.cu" \
        "${OUT_DIR}/libpf_mx7_prefill_v10.so" \
        pf_mx7_build_expert_plan \
        pf_mx7_zero_slice_wb_bf16 \
        pf_mx7_w13_prefill_fused_situ \
        pf_mx7_w13_prefill_unfused \
        pf_mx7_w2_prefill_persistent \
        pf_mx7_v10_version \
        pf_mx7_v10_set_reserve_sms \
        pf_mx7_v10_get_reserve_sms \
        pf_mx7_build_plan_v10 \
        pf_mx7_w2_v10_unfused \
        pf_mx7_w2_v10_combine
fi

echo "=== ALL SELECTED WAVE-V10 SM121a PREFILL LIBRARIES PASSED ZERO-SPILL & ABI GATES -> ${OUT_DIR} ==="
ls -lh "${OUT_DIR}"/*.so
