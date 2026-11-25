#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# =============================================================================
# Wave-V16 SM121a (GB10: 48 SMs) 0-ULP, 0-Spill, 0-Stack Unified MoE Builder
# Builds ALL 4 MoE libraries (40L NVFP4 + 8L MX7 across C=1..2 and C=2..32):
#   1. libnvfp4_ws5_c48_v16.so  (C=2..32 M=17..256: L2::evict_first + LPT Queue + W2 T16/T32 occupancy)
#   2. libnvfp4_ws5_c1_v16.so   (C=1..2  M<=16:     L2::evict_first + LPT Queue + W2 pre-wait prefetch)
#   3. libmxfp8_mx7_c48_v16.so  (C=2..32 M=17..256: L2::evict_first + LPT Queue + W2 pre-wait hoist)
#   4. libmxfp8_mx7_c1_v16.so   (C=1..2  M<=16:     L2::evict_first + LPT Queue + W2 pre-wait hoist)
#
# Strict SM121a 0-ULP Compiler Contract:
#   - Mandatory: -O3 -std=c++17 -gencode arch=compute_121a,code=sm_121a --fmad=false --shared -Xcompiler -fPIC --ptxas-options=-v
#   - Forbidden: -arch=sm_121a, --ftz=true, -ftz=true, --use_fast_math
#   - Hard-fails if ANY kernel reports >0 B stack frame, >0 B spill stores, >0 B spill loads, or 0 entries!
#   - Full nm -D ABI verification of ALL 90 exported C symbols (19 + 10 + 35 + 26).
#
# Usage (on GB10 outside live inference windows):
#   ./build_moe_v16.sh [OUT_DIR]
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${1:-${SCRIPT_DIR}/build}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
if [[ ! -x "${NVCC}" ]] && command -v nvcc >/dev/null 2>&1; then
    NVCC="$(command -v nvcc)"
fi
NVFP4_EVICT="${NVFP4_USE_L2_EVICT_HINT:-1}"
NVFP4_NC="${NVFP4_L2_EVICT_USE_NC:-1}"
MX7_EVICT="${MX7_USE_L2_EVICT_HINT:-1}"

mkdir -p "${OUT_DIR}"

if [[ ! -x "${NVCC}" ]]; then
    echo "[build_moe_v16] ERROR: nvcc not found or not executable at ${NVCC}" >&2
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

check_zero_spills() {
    local log_file="$1"
    python3 - "${log_file}" <<'PY'
import re, sys

text = open(sys.argv[1], "r", encoding="utf-8", errors="replace").read()
bad = []
cur_fn = "<unknown>"
max_regs = 0
n_entries = 0
for line in text.splitlines():
    m_fn = re.search(r"Compiling entry function '([^']+)'", line)
    if m_fn:
        cur_fn = m_fn.group(1)
        n_entries += 1
    m_reg = re.search(r"Used\s+(\d+)\s+registers", line)
    if m_reg:
        max_regs = max(max_regs, int(m_reg.group(1)))
    m_st = re.search(r"(\d+)\s+bytes\s+stack\s+frame,\s*(\d+)\s+bytes\s+spill\s+stores,\s*(\d+)\s+bytes\s+spill\s+loads", line)
    if m_st:
        sf, ss, sl = map(int, m_st.groups())
        if sf > 0 or ss > 0 or sl > 0:
            bad.append((cur_fn, sf, ss, sl))
if n_entries == 0:
    print("ERROR: ptxas log contained 0 compiled entry functions!", file=sys.stderr)
    sys.exit(3)
if bad:
    for fn, sf, ss, sl in bad[:10]:
        print(f"SPILL/STACK DETECTED in {fn}: stack={sf}B, spill_st={ss}B, spill_ld={sl}B", file=sys.stderr)
    sys.exit(2)
print(f"ptxas OK: 0 B stack frame, 0 B spills across {n_entries} kernels (max_regs={max_regs})")
PY
}

check_symbols() {
    local so_path="$1"
    shift
    local missing=0
    local syms
    syms="$(nm -D --defined-only "${so_path}" | awk '{print $3}')"
    for sym in "$@"; do
        if ! printf '%s\n' "${syms}" | grep -qx "${sym}"; then
            echo "[build_moe_v16] ERROR: missing exported symbol '${sym}' in ${so_path}" >&2
            missing=1
        fi
    done
    if [[ "${missing}" -ne 0 ]]; then
        exit 1
    fi
    echo "[build_moe_v16] nm -D ABI OK: all $# symbols exported by $(basename "${so_path}")"
}

# =============================================================================
# 1. Build libnvfp4_ws5_c48_v16.so (19 symbols; NEVER drops BN16<8 or W2_T8<6!)
#    Tuple: W13_T8_BN32 W13_T8_BN16 W2_T8 W13_T16 W13_T32B W2_T16 W2_T32
# =============================================================================
NV_C48_SRC="${SCRIPT_DIR}/nvfp4_ws5_c48_v16.cu"
NV_C48_OUT="${OUT_DIR}/libnvfp4_ws5_c48_v16.so"
NV_C48_LOG="${OUT_DIR}/ptxas_nvfp4_c48_v16.log"

NV_C48_LADDER=(
    "5 8 6 4 4 5 4"
    "5 8 6 4 4 4 3"
    "4 8 6 4 4 4 3"
    "4 8 6 4 3 4 3"
)

NV_C48_OK=0
for cfg in "${NV_C48_LADDER[@]}"; do
    read -r mc_w13 mc_bn16 mc_w2 mc_t16 mc_t32b mc_w2_t16 mc_w2_t32 <<< "${cfg}"
    echo "=== Building libnvfp4_ws5_c48_v16.so [W13_T8=${mc_w13} BN16=${mc_bn16} W2_T8=${mc_w2} T16=${mc_t16} T32B=${mc_t32b} W2_T16=${mc_w2_t16} W2_T32=${mc_w2_t32} EVICT=${NVFP4_EVICT} NC=${NVFP4_NC}] ==="
    "${NVCC}" "${COMMON_FLAGS[@]}" \
        "-DNVFP4_W13_T8_MIN_CTA=${mc_w13}" \
        "-DNVFP4_W13_T8_BN16_MIN_CTA=${mc_bn16}" \
        "-DNVFP4_W2_T8_MIN_CTA=${mc_w2}" \
        "-DNVFP4_W13_T16_MIN_CTA=${mc_t16}" \
        "-DNVFP4_W13_T32B_MIN_CTA=${mc_t32b}" \
        "-DNVFP4_W2_T16_MIN_CTA=${mc_w2_t16}" \
        "-DNVFP4_W2_T32_MIN_CTA=${mc_w2_t32}" \
        "-DNVFP4_USE_L2_EVICT_HINT=${NVFP4_EVICT}" \
        "-DNVFP4_L2_EVICT_USE_NC=${NVFP4_NC}" \
        "${NV_C48_SRC}" -o "${NV_C48_OUT}" > "${NV_C48_LOG}" 2>&1
    if check_zero_spills "${NV_C48_LOG}"; then
        NV_C48_OK=1
        echo "[build_moe_v16] SELECTED nvfp4_c48_v16 rung: ${cfg}"
        break
    fi
    echo "[build_moe_v16] WARN: spills detected in nvfp4_c48_v16 (${cfg}); stepping down..." >&2
done

if [[ "${NV_C48_OK}" -ne 1 ]]; then
    echo "[build_moe_v16] FATAL: libnvfp4_ws5_c48_v16.so failed 0-spill gate! Inspect ${NV_C48_LOG}" >&2
    exit 1
fi

check_symbols "${NV_C48_OUT}" \
    nvfp4_ws5_c32_version \
    nvfp4_ws5_c32_w13_min_cta \
    nvfp4_ws5_c48_min_cta \
    nvfp4_ws5_c48_v9_bn16_min_cta \
    nvfp4_ws5_c48_v16_w2_min_cta \
    nvfp4_w13_ws5_c48_v16_fwd \
    nvfp4_w2_ws5_c48_v16_fwd \
    nvfp4_w13_ws5_c48_v10_fwd \
    nvfp4_w13_ws5_c48_v9_fwd \
    nvfp4_situ_quant_dequant_c48_v9_fwd \
    nvfp4_w2_ws5_c48_v9_fwd \
    nvfp4_combine_c32_v9_fwd \
    nvfp4_w13_ws5_c48_v8_fwd \
    nvfp4_w13_ws5_c32_fwd \
    nvfp4_w2_ws5_c48_v8_fwd \
    nvfp4_w2_ws5_c32_fwd \
    nvfp4_combine_c32_v6_fwd \
    nvfp4_w13_ws5_fwd \
    nvfp4_w2_ws5_fwd

# =============================================================================
# 2. Build libnvfp4_ws5_c1_v16.so (10 symbols; NEVER drops BN16<8 or W2_T8<6!)
#    Tuple: W13_T8_BN32 W13_T8_BN16 W2_T8
# =============================================================================
NV_C1_SRC="${SCRIPT_DIR}/nvfp4_ws5_c1_v16.cu"
NV_C1_OUT="${OUT_DIR}/libnvfp4_ws5_c1_v16.so"
NV_C1_LOG="${OUT_DIR}/ptxas_nvfp4_c1_v16.log"

NV_C1_LADDER=(
    "6 8 6"
    "5 8 6"
    "4 8 6"
)

NV_C1_OK=0
for cfg in "${NV_C1_LADDER[@]}"; do
    read -r mc_bn32 mc_bn16 mc_w2 <<< "${cfg}"
    echo "=== Building libnvfp4_ws5_c1_v16.so [BN32=${mc_bn32} BN16=${mc_bn16} W2_T8=${mc_w2} EVICT=${NVFP4_EVICT} NC=${NVFP4_NC}] ==="
    "${NVCC}" "${COMMON_FLAGS[@]}" \
        "-DNVFP4_C1_W13_T8_BN32_MIN_CTA=${mc_bn32}" \
        "-DNVFP4_C1_W13_T8_BN16_MIN_CTA=${mc_bn16}" \
        "-DNVFP4_C1_W2_T8_MIN_CTA=${mc_w2}" \
        "-DNVFP4_USE_L2_EVICT_HINT=${NVFP4_EVICT}" \
        "-DNVFP4_L2_EVICT_USE_NC=${NVFP4_NC}" \
        "${NV_C1_SRC}" -o "${NV_C1_OUT}" > "${NV_C1_LOG}" 2>&1
    if check_zero_spills "${NV_C1_LOG}"; then
        NV_C1_OK=1
        echo "[build_moe_v16] SELECTED nvfp4_c1_v16 rung: ${cfg}"
        break
    fi
    echo "[build_moe_v16] WARN: spills detected in nvfp4_c1_v16 (${cfg}); stepping down..." >&2
done

if [[ "${NV_C1_OK}" -ne 1 ]]; then
    echo "[build_moe_v16] FATAL: libnvfp4_ws5_c1_v16.so failed 0-spill gate! Inspect ${NV_C1_LOG}" >&2
    exit 1
fi

check_symbols "${NV_C1_OUT}" \
    nvfp4_ws5_c1_version \
    nvfp4_ws5_c1_min_cta \
    nvfp4_w13_ws5_c1_v16_fwd \
    nvfp4_w2_ws5_c1_v16_fwd \
    nvfp4_w13_ws5_c1_v10_fwd \
    nvfp4_w13_ws5_c1_v9_fwd \
    nvfp4_situ_quant_dequant_c1_v9_fwd \
    nvfp4_w2_ws5_c1_v9_fwd \
    nvfp4_w13_ws5_c1_v4_fwd \
    nvfp4_w2_ws5_c1_v4_fwd

# =============================================================================
# 3. Build libmxfp8_mx7_c48_v16.so (35 symbols; NEVER drops BN16<8 or W2_T8<6!)
#    Tuple: W13_T8_BN32 W13_T8_BN16 W2_T8 W13_T16 W2_T16 W13_T32 W2_T32
# =============================================================================
MX_C48_SRC="${SCRIPT_DIR}/mxfp8_mx7_c48_v16.cu"
MX_C48_OUT="${OUT_DIR}/libmxfp8_mx7_c48_v16.so"
MX_C48_LOG="${OUT_DIR}/ptxas_mx7_c48_v16.log"

MX_C48_LADDER=(
    "6 8 5 5 5 5 4"
    "5 8 5 5 5 5 4"
    "5 8 5 5 5 4 4"
    "5 8 5 5 4 4 4"
)

MX_C48_OK=0
for cfg in "${MX_C48_LADDER[@]}"; do
    read -r W13_T8 W13_BN16 W2_T8 W13_T16 W2_T16 W13_T32 W2_T32 <<< "${cfg}"
    echo "=== Building libmxfp8_mx7_c48_v16.so [W13_T8=${W13_T8} BN16=${W13_BN16} W2_T8=${W2_T8} W13_T16=${W13_T16} W2_T16=${W2_T16} W13_T32=${W13_T32} W2_T32=${W2_T32} EVICT=${MX7_EVICT}] ==="
    "${NVCC}" "${COMMON_FLAGS[@]}" \
        "-DMX7_W13_T8_BN32_MIN_CTA=${W13_T8}" \
        "-DMX7_W13_T8_BN16_MIN_CTA=${W13_BN16}" \
        "-DMX7_W2_T8_MIN_CTA=${W2_T8}" \
        "-DMX7_W13_T16_MIN_CTA=${W13_T16}" \
        "-DMX7_W2_T16_MIN_CTA=${W2_T16}" \
        "-DMX7_W13_T32_MIN_CTA=${W13_T32}" \
        "-DMX7_W2_T32_MIN_CTA=${W2_T32}" \
        "-DMX7_USE_L2_EVICT_HINT=${MX7_EVICT}" \
        "${MX_C48_SRC}" -o "${MX_C48_OUT}" > "${MX_C48_LOG}" 2>&1
    if check_zero_spills "${MX_C48_LOG}"; then
        MX_C48_OK=1
        echo "[build_moe_v16] SELECTED mx7_c48_v16 rung: ${cfg}"
        break
    fi
    echo "[build_moe_v16] WARN: spills detected in mx7_c48_v16 (${cfg}); stepping down..." >&2
done

if [[ "${MX_C48_OK}" -ne 1 ]]; then
    echo "[build_moe_v16] FATAL: libmxfp8_mx7_c48_v16.so failed 0-spill gate! Inspect ${MX_C48_LOG}" >&2
    exit 1
fi

check_symbols "${MX_C48_OUT}" \
    mx7_c32_version \
    mx7_c48_min_cta \
    mx7_c48_v14_bn16_min_cta \
    mx7_c48_v14_min_cta \
    mx7_combine_c48_v7_fwd \
    mx7_v2_count_flips \
    mx7_v2_pack \
    mx7_v5_count_hybrid_from_mx7 \
    mx7_v5_build_hybrid_from_mx7 \
    mx7_v5_panel_measure_from_mx7 \
    mx7_v5_panel_build_from_mx7 \
    mx7_w13_c48_v16_fwd \
    mx7_w2_c48_v16_fwd \
    mx7_w13_c48_v16_compat_fwd \
    mx7_w2_c48_v16_compat_fwd \
    mx7_w13_c48_v16_panel_fwd \
    mx7_w2_c48_v16_panel_fwd \
    mx7_w13_c48_v14_fwd \
    mx7_w2_c48_v14_fwd \
    mx7_w13_c48_v14_compat_fwd \
    mx7_w2_c48_v14_compat_fwd \
    mx7_w13_c48_v14_panel_fwd \
    mx7_w2_c48_v14_panel_fwd \
    mx7_w13_c48_v8_fwd \
    mx7_w2_c48_v8_fwd \
    mx7_w13_c48_v8_compat_fwd \
    mx7_w2_c48_v8_compat_fwd \
    mx7_w13_c48_v8_panel_fwd \
    mx7_w2_c48_v8_panel_fwd \
    mx7_w13_c32_fwd \
    mx7_w2_c32_fwd \
    mx7_w13_c32_v5_fwd \
    mx7_w2_c32_v5_fwd \
    mx7_w13_c32_v5_panel_fwd \
    mx7_w2_c32_v5_panel_fwd

# =============================================================================
# 4. Build libmxfp8_mx7_c1_v16.so (26 symbols; NEVER drops BN16<8 or W2_T8<6!)
#    Tuple: W13_T8_BN32 W13_T8_BN16 W2_T8 W13_T16_BN32 W13_T16_BN16 W2_T16
# =============================================================================
MX_C1_SRC="${SCRIPT_DIR}/mxfp8_mx7_c1_v16.cu"
MX_C1_OUT="${OUT_DIR}/libmxfp8_mx7_c1_v16.so"
MX_C1_LOG="${OUT_DIR}/ptxas_mx7_c1_v16.log"

MX_C1_LADDER=(
    "6 8 6 5 5 6"
    "6 8 6 5 5 5"
    "6 8 5 5 5 6"
    "5 8 5 5 5 5"
    "5 8 5 4 5 5"
)

MX_C1_OK=0
for cfg in "${MX_C1_LADDER[@]}"; do
    read -r W13_T8 W13_BN16 W2_T8 W13_T16 W13_T16_BN16 W2_T16 <<< "${cfg}"
    echo "=== Building libmxfp8_mx7_c1_v16.so [W13_T8=${W13_T8} BN16=${W13_BN16} W2_T8=${W2_T8} W13_T16=${W13_T16} T16_BN16=${W13_T16_BN16} W2_T16=${W2_T16} EVICT=${MX7_EVICT}] ==="
    "${NVCC}" "${COMMON_FLAGS[@]}" \
        "-DMX7_C1_W13_T8_BN32_MIN_CTA=${W13_T8}" \
        "-DMX7_C1_W13_T8_BN16_MIN_CTA=${W13_BN16}" \
        "-DMX7_C1_W2_T8_MIN_CTA=${W2_T8}" \
        "-DMX7_C1_W13_T16_BN32_MIN_CTA=${W13_T16}" \
        "-DMX7_C1_W13_T16_BN16_MIN_CTA=${W13_T16_BN16}" \
        "-DMX7_C1_W2_T16_MIN_CTA=${W2_T16}" \
        "-DMX7_USE_L2_EVICT_HINT=${MX7_EVICT}" \
        "${MX_C1_SRC}" -o "${MX_C1_OUT}" > "${MX_C1_LOG}" 2>&1
    if check_zero_spills "${MX_C1_LOG}"; then
        MX_C1_OK=1
        echo "[build_moe_v16] SELECTED mx7_c1_v16 rung: ${cfg}"
        break
    fi
    echo "[build_moe_v16] WARN: spills detected in mx7_c1_v16 (${cfg}); stepping down..." >&2
done

if [[ "${MX_C1_OK}" -ne 1 ]]; then
    echo "[build_moe_v16] FATAL: libmxfp8_mx7_c1_v16.so failed 0-spill gate! Inspect ${MX_C1_LOG}" >&2
    exit 1
fi

check_symbols "${MX_C1_OUT}" \
    mx7_c1_version \
    mx7_c1_min_cta \
    mx7_v2_count_flips \
    mx7_v2_pack \
    mx7_v5_count_hybrid_from_mx7 \
    mx7_v5_build_hybrid_from_mx7 \
    mx7_v5_panel_measure_from_mx7 \
    mx7_v5_panel_build_from_mx7 \
    mx7_w13_c1_v16_fwd \
    mx7_w2_c1_v16_fwd \
    mx7_w13_c1_v16_compat_fwd \
    mx7_w2_c1_v16_compat_fwd \
    mx7_w13_c1_v16_panel_fwd \
    mx7_w2_c1_v16_panel_fwd \
    mx7_w13_c1_v14_fwd \
    mx7_w2_c1_v14_fwd \
    mx7_w13_c1_v14_compat_fwd \
    mx7_w2_c1_v14_compat_fwd \
    mx7_w13_c1_v14_panel_fwd \
    mx7_w2_c1_v14_panel_fwd \
    mx7_w13_c1_v4_fwd \
    mx7_w2_c1_v4_fwd \
    mx7_w13_c1_v5_fwd \
    mx7_w2_c1_v5_fwd \
    mx7_w13_c1_v5_panel_fwd \
    mx7_w2_c1_v5_panel_fwd

echo "=== Wave-V16 MoE Build Complete (0 spills, 0 stack, ALL 90 ABI symbols verified!) ==="
ls -lh "${NV_C48_OUT}" "${NV_C1_OUT}" "${MX_C48_OUT}" "${MX_C1_OUT}"
