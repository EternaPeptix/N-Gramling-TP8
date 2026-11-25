#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Strict SM121a (GB10 Blackwell) build script for libdrafter_u4_c18_v14.so
# Enforces: -gencode arch=compute_121a,code=sm_121a, --fmad=false, 0 stack frame, 0 spills, exact nm -D exports.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${1:-/tmp/v14_drafter_u4_out}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
SRC="${SCRIPT_DIR}/drafter_u4_c18_v14.cu"
OUT_SO="${OUT_DIR}/libdrafter_u4_c18_v14.so"
LOG_FILE="${OUT_DIR}/drafter_u4_c18_v14.ptxas.log"

mkdir -p "${OUT_DIR}"

if [[ ! -f "${SRC}" ]]; then
    echo "ERROR: missing source file: ${SRC}" >&2
    exit 1
fi

echo "[build-v14] Compiling ${SRC} -> ${OUT_SO} (strict SM121a, --fmad=false)"
"${NVCC}" \
    -O3 \
    -std=c++17 \
    -gencode arch=compute_121a,code=sm_121a \
    --fmad=false \
    --shared \
    -Xcompiler -fPIC \
    --ptxas-options=-v \
    -o "${OUT_SO}" \
    "${SRC}" 2>&1 | tee "${LOG_FILE}"

echo "[build-v14] Checking 0-spill and 0-stack-frame gate..."
python3 - "${LOG_FILE}" <<'PY'
import re
import sys

log_path = sys.argv[1]
txt = open(log_path, "r", encoding="utf-8", errors="replace").read()
kernels = re.findall(r"Compiling entry function '([^']+)'", txt)
bad = []
cur = "<unknown>"
for line in txt.splitlines():
    m_k = re.search(r"Compiling entry function '([^']+)'", line)
    if m_k:
        cur = m_k.group(1)
    m_s = re.search(r"(\d+)\s+bytes\s+stack\s+frame,\s+(\d+)\s+bytes\s+spill\s+stores,\s+(\d+)\s+bytes\s+spill\s+loads", line)
    if m_s:
        sf, ss, sl = int(m_s.group(1)), int(m_s.group(2)), int(m_s.group(3))
        if sf > 0 or ss > 0 or sl > 0:
            bad.append((cur, sf, ss, sl))
if not kernels:
    print("ERROR: no compiled kernels found in ptxas log!", file=sys.stderr)
    sys.exit(2)
if bad:
    print(f"ERROR: {len(bad)} kernels violated 0-stack/0-spill gate:", file=sys.stderr)
    for k, sf, ss, sl in bad[:20]:
        print(f"  {k}: stack={sf}B, spill_stores={ss}B, spill_loads={sl}B", file=sys.stderr)
    sys.exit(3)
print(f"[build-v14] PASS: {len(kernels)} SM121a kernels compiled with 0B stack frame, 0B spill stores, 0B spill loads.")
PY

echo "[build-v14] Checking exported C symbols via nm -D..."
REQUIRED_SYMS=(
    "drafter_u4_c18_v14_version"
    "drafter_u4_c18_v14_fwd"
    "drafter_u4_tc_v9_version"
    "drafter_u4_tc_v9_fwd"
)
EXPORTS="$(nm -D --defined-only "${OUT_SO}" | awk '{print $3}')"
for sym in "${REQUIRED_SYMS[@]}"; do
    if ! grep -qx "${sym}" <<< "${EXPORTS}"; then
        echo "ERROR: missing required exported symbol: ${sym}" >&2
        exit 4
    fi
done
echo "[build-v14] PASS: all ${#REQUIRED_SYMS[@]} required symbols exported in ${OUT_SO}."
ls -lh "${OUT_SO}"
