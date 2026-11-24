#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Strict SM121a (GB10) Production Build for 0-ULP Lossless MX6-Huff7 Engine v12
#   - Strictly `-gencode arch=compute_121a,code=sm_121a --fmad=false --ptxas-options=-v`
#   - NO `--use_fast_math`, NO `--ftz=true`, NO `-arch=sm_121a`, NO writes to /rdma/lib
#   - Auto-probes W13 & W2 MIN_CTA across ALL tiles (T8, T16, T32) to guarantee:
#       0 B local stack frame, 0 B spill stores, 0 B spill loads on EVERY kernel!
#   - Hard-verifies all 13 C ABI exports via `nm -D`.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${SCRIPT_DIR}/mxfp8_mx6_v12.cu"
OUT="${1:-${SCRIPT_DIR}/libmxfp8_mx6_v12.so}"
NVCC="${NVCC:-nvcc}"
PTX_LOG="$(mktemp -t mx6_v12_ptxas.XXXXXX.log)"
trap 'rm -f "${PTX_LOG}"' EXIT

if [[ ! -f "${SRC}" ]]; then
    echo "ERROR: missing source ${SRC}" >&2
    exit 1
fi

COMMON_FLAGS=(
    -O3 -std=c++17
    -gencode arch=compute_121a,code=sm_121a
    --fmad=false
    --shared -Xcompiler -fPIC
    --ptxas-options=-v
)

has_spills_or_stack() {
    python3 - "$1" <<'PY'
import re, sys
txt = open(sys.argv[1], "r", encoding="utf-8", errors="replace").read()
bad = []
for m in re.finditer(r"(\d+)\s+bytes\s+(stack frame|spill stores|spill loads)", txt):
    if int(m.group(1)) > 0:
        bad.append(m.group(0))
if bad:
    print("FAIL:", ", ".join(bad[:8]))
    sys.exit(1)
sys.exit(0)
PY
}

# Candidate (W13_T8_BN32, W13_T8_BN16, W13_T16, W13_T32, W2_T8, W2_T16, W2_T32)
# Starts at peak SM121a occupancy (5/8/4/2/5/4/3) and steps down ONLY if CUDA 13.x spills.
CANDIDATES=(
    "5 8 4 2 5 4 3"
    "5 8 4 2 4 4 3"
    "5 8 4 2 4 3 3"
    "4 8 4 2 4 3 2"
    "4 6 3 2 4 3 2"
)

BUILT=0
for cfg in "${CANDIDATES[@]}"; do
    read -r W13_T8_32 W13_T8_16 W13_T16 W13_T32 W2_T8 W2_T16 W2_T32 <<< "${cfg}"
    echo "[build_mx6_v12] Trying MIN_CTA=(W13: T8_32=${W13_T8_32}, T8_16=${W13_T8_16}, T16=${W13_T16}, T32=${W13_T32} | W2: T8=${W2_T8}, T16=${W2_T16}, T32=${W2_T32})..."
    "${NVCC}" "${COMMON_FLAGS[@]}" \
        "-DMX6_W13_T8_BN32_MIN_CTA=${W13_T8_32}" \
        "-DMX6_W13_T8_BN16_MIN_CTA=${W13_T8_16}" \
        "-DMX6_W13_T16_MIN_CTA=${W13_T16}" \
        "-DMX6_W13_T32_MIN_CTA=${W13_T32}" \
        "-DMX6_W2_T8_MIN_CTA=${W2_T8}" \
        "-DMX6_W2_T16_MIN_CTA=${W2_T16}" \
        "-DMX6_W2_T32_MIN_CTA=${W2_T32}" \
        "${SRC}" -o "${OUT}" >"${PTX_LOG}" 2>&1
    if has_spills_or_stack "${PTX_LOG}"; then
        BUILT=1
        echo "[build_mx6_v12] PASS: 0 B stack / 0 B spill stores / 0 B spill loads with MIN_CTA=(${cfg})"
        break
    else
        echo "[build_mx6_v12] Spills/stack detected with MIN_CTA=(${cfg}); stepping down..."
    fi
done

if [[ "${BUILT}" -ne 1 ]]; then
    echo "ERROR: All MIN_CTA candidates had stack/spills! Last ptxas log:" >&2
    cat "${PTX_LOG}" >&2
    exit 1
fi

REQUIRED_SYMS=(
    mx6_v10_version
    mx6_v12_version
    mx6_v10_min_cta
    mx6_v10_hist_exp
    mx6_v10_build_huff7_table
    mx6_measure_v10
    mx6_pack_v10
    mx6_unpack_expert_batch_v10_fwd
    mx6_w13_c1_v10_fwd
    mx6_w2_c1_v10_fwd
    mx6_w13_c48_v10_fwd
    mx6_w2_c48_v10_fwd
    mx6_combine_c48_v10_fwd
)

NM_OUT="$(nm -D --defined-only "${OUT}")"
for sym in "${REQUIRED_SYMS[@]}"; do
    if ! grep -Eq " [TW] ${sym}$" <<< "${NM_OUT}"; then
        echo "ERROR: missing required export ${sym} in ${OUT}" >&2
        exit 1
    fi
done

echo "[build_mx6_v12] Verified all ${#REQUIRED_SYMS[@]} C ABI exports in ${OUT}"
