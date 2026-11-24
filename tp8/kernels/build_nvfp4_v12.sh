#!/usr/bin/env bash
# =============================================================================
# Wave-V12 SM121a NVFP4 MoE 0-ULP Build Script (GB10: 48 SMs, SM121a)
# Builds:
#   1. libnvfp4_ws5_c48_v12.so (M >= 17, C=2..32 + prefill)
#   2. libnvfp4_ws5_c1_v12.so  (M <= 16, C=1..2)
#
# Strict SM121a 0-ULP Compiler Contract:
#   - -gencode arch=compute_121a,code=sm_121a  (NEVER -arch=sm_121a!)
#   - --fmad=false                             (NO --use_fast_math, NO --ftz=true!)
#   - --ptxas-options=-v                       (HARD-FAIL on >0 B stack or >0 B spill!)
#   - Auto MIN_CTA probe: attempts max occupancy first, steps down only on spill.
#   - nm -D ABI verification of ALL 16 (c48) + 8 (c1) exported C symbols.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${1:-${SCRIPT_DIR}/build}"
NVCC="${NVCC:-nvcc}"
mkdir -p "${OUT_DIR}"

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
for line in text.splitlines():
    m_fn = re.search(r"Compiling entry function '([^']+)'", line)
    if m_fn:
        cur_fn = m_fn.group(1)
    m_reg = re.search(r"Used\s+(\d+)\s+registers", line)
    if m_reg:
        max_regs = max(max_regs, int(m_reg.group(1)))
    m_st = re.search(r"(\d+)\s+bytes\s+stack\s+frame,\s+(\d+)\s+bytes\s+spill\s+stores,\s+(\d+)\s+bytes\s+spill\s+loads", line)
    if m_st:
        sf, ss, sl = map(int, m_st.groups())
        if sf > 0 or ss > 0 or sl > 0:
            bad.append((cur_fn, sf, ss, sl))
if bad:
    for fn, sf, ss, sl in bad[:8]:
        print(f"SPILL/STACK DETECTED in {fn}: stack={sf}B, spill_st={ss}B, spill_ld={sl}B", file=sys.stderr)
    sys.exit(2)
print(f"ptxas OK: 0 B stack frame, 0 B spills across all kernels (max_regs={max_regs})")
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
            echo "ERROR: missing exported symbol ${sym} in ${so_path}" >&2
            missing=1
        fi
    done
    if [[ "${missing}" -ne 0 ]]; then
        exit 1
    fi
    echo "ABI OK: all $# expected symbols exported by $(basename "${so_path}")"
}

# -----------------------------------------------------------------------------
# 1. Build libnvfp4_ws5_c48_v12.so (auto MIN_CTA 0-spill probe)
# -----------------------------------------------------------------------------
C48_SRC="${SCRIPT_DIR}/nvfp4_ws5_c48_v12.cu"
C48_OUT="${OUT_DIR}/libnvfp4_ws5_c48_v12.so"
C48_LOG="${OUT_DIR}/build_c48_v12.log"

# (W13_T8, W13_T8_BN16, W2_T8, W13_T16, W13_T32B)
C48_CANDIDATES=(
    "6 8 6 5 4"
    "5 8 6 5 4"
    "5 8 5 4 4"
    "4 8 5 4 4"
    "4 6 4 4 3"
)

C48_OK=0
for cfg in "${C48_CANDIDATES[@]}"; do
    read -r mc_w13 mc_bn16 mc_w2 mc_t16 mc_t32b <<< "${cfg}"
    echo "=== Building libnvfp4_ws5_c48_v12.so [W13_T8=${mc_w13} BN16=${mc_bn16} W2_T8=${mc_w2} T16=${mc_t16} T32B=${mc_t32b}] ==="
    "${NVCC}" "${COMMON_FLAGS[@]}" \
        "-DNVFP4_W13_T8_MIN_CTA=${mc_w13}" \
        "-DNVFP4_W13_T8_BN16_MIN_CTA=${mc_bn16}" \
        "-DNVFP4_W2_T8_MIN_CTA=${mc_w2}" \
        "-DNVFP4_W13_T16_MIN_CTA=${mc_t16}" \
        "-DNVFP4_W13_T32B_MIN_CTA=${mc_t32b}" \
        "${C48_SRC}" -o "${C48_OUT}" 2>&1 | tee "${C48_LOG}"
    if check_zero_spills "${C48_LOG}"; then
        C48_OK=1
        echo "SELECTED c48_v12 MIN_CTA profile: ${cfg}"
        break
    fi
    echo "WARN: spills detected with profile (${cfg}); stepping down MIN_CTA..." >&2
done

if [[ "${C48_OK}" -ne 1 ]]; then
    echo "FATAL: libnvfp4_ws5_c48_v12.so failed 0-spill gate across all MIN_CTA profiles!" >&2
    exit 1
fi

check_symbols "${C48_OUT}" \
    nvfp4_ws5_c32_version \
    nvfp4_ws5_c32_w13_min_cta \
    nvfp4_ws5_c48_min_cta \
    nvfp4_ws5_c48_v9_bn16_min_cta \
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

# -----------------------------------------------------------------------------
# 2. Build libnvfp4_ws5_c1_v12.so (auto MIN_CTA 0-spill probe)
# -----------------------------------------------------------------------------
C1_SRC="${SCRIPT_DIR}/nvfp4_ws5_c1_v12.cu"
C1_OUT="${OUT_DIR}/libnvfp4_ws5_c1_v12.so"
C1_LOG="${OUT_DIR}/build_c1_v12.log"

# (W13_T8_BN32, W13_T8_BN16, W2_T8)
C1_CANDIDATES=(
    "6 8 6"
    "5 8 6"
    "5 8 5"
    "4 8 5"
    "4 6 4"
)

C1_OK=0
for cfg in "${C1_CANDIDATES[@]}"; do
    read -r mc_bn32 mc_bn16 mc_w2 <<< "${cfg}"
    echo "=== Building libnvfp4_ws5_c1_v12.so [BN32=${mc_bn32} BN16=${mc_bn16} W2_T8=${mc_w2}] ==="
    "${NVCC}" "${COMMON_FLAGS[@]}" \
        "-DNVFP4_C1_W13_T8_BN32_MIN_CTA=${mc_bn32}" \
        "-DNVFP4_C1_W13_T8_BN16_MIN_CTA=${mc_bn16}" \
        "-DNVFP4_C1_W2_T8_MIN_CTA=${mc_w2}" \
        "${C1_SRC}" -o "${C1_OUT}" 2>&1 | tee "${C1_LOG}"
    if check_zero_spills "${C1_LOG}"; then
        C1_OK=1
        echo "SELECTED c1_v12 MIN_CTA profile: ${cfg}"
        break
    fi
    echo "WARN: spills detected with profile (${cfg}); stepping down MIN_CTA..." >&2
done

if [[ "${C1_OK}" -ne 1 ]]; then
    echo "FATAL: libnvfp4_ws5_c1_v12.so failed 0-spill gate across all MIN_CTA profiles!" >&2
    exit 1
fi

check_symbols "${C1_OUT}" \
    nvfp4_ws5_c1_version \
    nvfp4_ws5_c1_min_cta \
    nvfp4_w13_ws5_c1_v10_fwd \
    nvfp4_w13_ws5_c1_v9_fwd \
    nvfp4_situ_quant_dequant_c1_v9_fwd \
    nvfp4_w2_ws5_c1_v9_fwd \
    nvfp4_w13_ws5_c1_v4_fwd \
    nvfp4_w2_ws5_c1_v4_fwd

echo "=== Wave-V12 Build Complete (0 spills, 0 stack, all 24 symbols verified) ==="
ls -lh "${C48_OUT}" "${C1_OUT}"
