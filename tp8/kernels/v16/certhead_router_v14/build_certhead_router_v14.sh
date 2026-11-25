#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Strict SM121a (NVIDIA DGX Spark GB10) Build Script for Wave-V14 CertHead & Router BF12
#
# Builds:
#   1. libcert_head_fast_v14.so  (0-ULP CertHead V14, M=2..128, 12 C ABI exports, drop-in V7/V9/V12 superset)
#   2. librouter_bf12_v14.so     (48x Router BF12/BF11 V14, M=1..128+, 14 C ABI exports, drop-in exact/c32/v10 superset)
#
# Mandatory invariants:
#   - -gencode arch=compute_121a,code=sm_121a --fmad=false --ptxas-options=-v
#   - ZERO fast-math / ftz / bare sm_121a shortcuts
#   - Hard-fail if ANY entry function reports > 0 B stack frame, spill stores, or spill loads
#   - Hard-fail if ANY required C ABI export is missing in nm -D
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${OUT_DIR:-${1:-/tmp/wave_v14_certhead_router_out}}"
NVCC="${NVCC:-$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)}"
mkdir -p "${OUT_DIR}"

COMMON_FLAGS=(
    -O3
    --shared
    -Xcompiler -fPIC
    -std=c++17
    -gencode arch=compute_121a,code=sm_121a
    --fmad=false
    --ptxas-options=-v
)

check_zero_spills() {
    local log_file="$1"
    local strict="${2:-1}"
    python3 - "${log_file}" "${strict}" <<'PY'
import pathlib, re, sys

log_path, strict = sys.argv[1], int(sys.argv[2])
txt = pathlib.Path(log_path).read_text(encoding="utf-8", errors="replace")

entries = re.findall(r"Compiling entry function '([^']+)'", txt)
regs = [int(m) for m in re.findall(r"Used\s+(\d+)\s+registers", txt)]
cur_fn = "<unknown>"
bad = []
for line in txt.splitlines():
    m_fn = re.search(r"Compiling entry function '([^']+)'", line)
    if m_fn:
        cur_fn = m_fn.group(1)
    m_st = re.search(
        r"(\d+)\s+bytes\s+stack\s+frame,\s+(\d+)\s+bytes\s+spill\s+stores,\s+(\d+)\s+bytes\s+spill\s+loads",
        line,
    )
    if m_st:
        st, ss, sl = int(m_st.group(1)), int(m_st.group(2)), int(m_st.group(3))
        if st > 0 or ss > 0 or sl > 0:
            bad.append((cur_fn, st, ss, sl))

if not entries:
    print("  [ERROR] No compiled entry functions found in ptxas log!", file=sys.stderr)
    sys.exit(1)

if bad:
    for fn, st, ss, sl in bad[:12]:
        print(f"  [SPILL/STACK] {fn}: stack={st}B spill_st={ss}B spill_ld={sl}B", file=sys.stderr)
    sys.exit(1 if strict else 2)

print(
    f"  [0-SPILL OK] entries={len(entries)} max_regs={max(regs) if regs else 0} "
    f"stack=0B spill_st=0B spill_ld=0B"
)
PY
}

verify_abi_symbols() {
    local so_path="$1"
    shift
    local syms
    syms="$(nm -D --defined-only "${so_path}")"
    for sym in "$@"; do
        if ! grep -Eq "[[:space:]]T[[:space:]]+${sym}$" <<< "${syms}"; then
            echo "[FAIL] Missing required exported C ABI symbol '${sym}' in ${so_path}!" >&2
            exit 1
        fi
    done
    echo "  [ABI OK] ${so_path} exports all $# required C ABI symbols."
}

# ==============================================================================
# [1/2] Build libcert_head_fast_v14.so
# ==============================================================================
CH_SRC="${SCRIPT_DIR}/cert_head_fast_v14.cu"
CH_SO="${OUT_DIR}/libcert_head_fast_v14.so"
CH_LOG="${OUT_DIR}/cert_head_fast_v14.ptxas.log"

echo "=== [1/2] Building ${CH_SO} (SM121a --fmad=false, 0-spill hard gate) ==="
"${NVCC}" "${COMMON_FLAGS[@]}" \
    "${CH_SRC}" -o "${CH_SO}" 2>&1 | tee "${CH_LOG}"

check_zero_spills "${CH_LOG}" 1
verify_abi_symbols "${CH_SO}" \
    cert_head_v14_version \
    cert_head_v14_pre_screen \
    cert_head_v14_post_refine \
    cert_head_v7_pre_screen \
    cert_head_v7_post_refine \
    cert_head_v7_post_refine_c48 \
    cert_head_v9_post_refine \
    cert_head_v7_exact_all \
    cert_head_v7_version \
    cert_head_v12_version \
    cert_head_v7_c48_max_m \
    cert_head_v9_max_m

# ==============================================================================
# [2/2] Build librouter_bf12_v14.so (auto MIN_CTA 0-spill probe)
# ==============================================================================
RT_SRC="${SCRIPT_DIR}/router_bf12_v14.cu"
RT_SO="${OUT_DIR}/librouter_bf12_v14.so"
RT_LOG="${OUT_DIR}/router_bf12_v14.ptxas.log"
RT_BUILT=0

echo "=== [2/2] Building ${RT_SO} (SM121a --fmad=false, auto MIN_CTA 0-spill probe) ==="
for cfg in \
    "2 2 3" \
    "2 2 2" \
    "2 1 3" \
    "2 1 2" \
    "1 1 2" \
    "1 1 1"; do
    read -r BM64 C128_GEMM K2 <<< "${cfg}"
    echo "  Probing Router V14: BM64_MIN_CTA=${BM64} C128_GEMM_MIN_CTA=${C128_GEMM} K2_MIN_CTA=${K2} ..."
    if "${NVCC}" "${COMMON_FLAGS[@]}" \
        -DROUTER_V14_BM64_MIN_CTA="${BM64}" \
        -DROUTER_V14_C128_GEMM_MIN_CTA="${C128_GEMM}" \
        -DROUTER_V14_K2_MIN_CTA="${K2}" \
        "${RT_SRC}" -o "${RT_SO}" 2> "${RT_LOG}"; then
        if check_zero_spills "${RT_LOG}" 0; then
            RT_BUILT=1
            echo "  SELECTED Router V14 config: BM64=${BM64} C128_GEMM=${C128_GEMM} K2=${K2}"
            break
        fi
    else
        cat "${RT_LOG}" >&2
        exit 1
    fi
done

if [[ "${RT_BUILT}" -ne 1 ]]; then
    echo "[FAIL] Could not build ${RT_SO} with 0 B stack and 0 B spills!" >&2
    check_zero_spills "${RT_LOG}" 1
    exit 1
fi

verify_abi_symbols "${RT_SO}" \
    router_bf12_v14_version \
    router_bf12_v10_version \
    router_bf12_c32_version \
    router_bf12_c48_min_cta \
    router_bf12_exact_fwd \
    router_bf12_c1_fused_v10_fwd \
    router_bf11_c1_fused_v10_fwd \
    router_bf12_c128_fused_v14_fwd \
    router_bf11_c128_fused_v14_fwd \
    router_bf12_c48_fused_v10_fwd \
    router_bf11_c48_fused_v10_fwd \
    router_bf12_c32_gemm_fwd \
    router_c32_quant_route_place_v9_fwd \
    router_c32_quant_route_place_fwd

echo "[PASS] Wave-V14 CertHead + Router BF12 libraries built & ABI-verified in ${OUT_DIR}!"
