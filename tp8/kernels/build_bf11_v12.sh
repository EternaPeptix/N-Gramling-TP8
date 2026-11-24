#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Strict SM121a (NVIDIA DGX Spark GB10) Builder for libbf11_flip_exact_v12.so
#
# Mandatory precision & zero-spill gates:
#   - Strictly `-gencode arch=compute_121a,code=sm_121a --fmad=false --ptxas-options=-v`
#   - ZERO `-arch=sm_121a`, ZERO `--use_fast_math`, ZERO `--ftz=true`
#   - Hard-fails if ANY compiled kernel reports > 0 B stack frame, spill stores, or spill loads
#   - Verifies all 24 exported `extern "C"` ABI symbols (full drop-in superset of s7/c32/v5/v9 + v12)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${OUT_DIR:-${1:-/tmp/bf11_v12_out}}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
PYTHON="${PYTHON:-python3}"

if [[ ! -x "${NVCC}" ]]; then
    NVCC="$(command -v nvcc || true)"
fi
if [[ -z "${NVCC}" || ! -x "${NVCC}" ]]; then
    echo "FATAL: nvcc not found (set NVCC=/usr/local/cuda/bin/nvcc)" >&2
    exit 1
fi

mkdir -p "${OUT_DIR}"
SRC="${SCRIPT_DIR}/bf11_flip_exact_v12.cu"
OUT_SO="${OUT_DIR}/libbf11_flip_exact_v12.so"
LOG_FILE="${OUT_DIR}/bf11_flip_exact_v12.ptxas.log"

echo "=== Building libbf11_flip_exact_v12.so (SM121a, --fmad=false, strict 0-spill) ==="
"${NVCC}" \
    -O3 \
    -std=c++17 \
    -gencode arch=compute_121a,code=sm_121a \
    --fmad=false \
    --shared \
    -Xcompiler -fPIC \
    --ptxas-options=-v \
    "${SRC}" \
    -o "${OUT_SO}" 2>&1 | tee "${LOG_FILE}"

echo "=== Auditing ptxas 0-stack / 0-spill gate ==="
"${PYTHON}" - "${LOG_FILE}" <<'PY'
import re, sys

log_path = sys.argv[1]
txt = open(log_path, "r", encoding="utf-8", errors="replace").read()
cur_fn = "<unknown>"
entries = 0
max_regs = 0
max_smem = 0
bad = []
for line in txt.splitlines():
    m_fn = re.search(r"Compiling entry function '([^']+)'", line)
    if m_fn:
        cur_fn = m_fn.group(1)
        entries += 1
        continue
    m_rg = re.search(r"Used\s+(\d+)\s+registers", line)
    if m_rg:
        max_regs = max(max_regs, int(m_rg.group(1)))
    m_sm = re.search(r"(\d+)\s+bytes\s+smem", line)
    if m_sm:
        max_smem = max(max_smem, int(m_sm.group(1)))
    m_st = re.search(
        r"(\d+)\s+bytes\s+stack\s+frame,\s*(\d+)\s+bytes\s+spill\s+stores,\s*(\d+)\s+bytes\s+spill\s+loads",
        line,
    )
    if m_st:
        sf, ss, sl = int(m_st.group(1)), int(m_st.group(2)), int(m_st.group(3))
        if sf > 0 or ss > 0 or sl > 0:
            bad.append((cur_fn, sf, ss, sl))

if entries == 0:
    print(f"FATAL: parsed 0 entry functions from {log_path}", file=sys.stderr)
    sys.exit(1)
if bad:
    for fn, sf, ss, sl in bad[:20]:
        print(f"FATAL SPILL/STACK: {fn}: stack={sf}B spill_st={ss}B spill_ld={sl}B", file=sys.stderr)
    sys.exit(1)
print(f"[0-SPILL OK] bf11_flip_exact_v12: {entries} kernels, max_regs={max_regs}, max_smem={max_smem}B, 0B stack/spills")
PY

echo "=== Verifying 24/24 exported extern \"C\" ABI symbols ==="
REQUIRED_SYMS=(
    bf11_flip_c32_version
    bf11_flip_fwd
    bf11_flip_dual_kv_fwd
    bf11_flip_c32_fwd
    bf11_flip_dual_kv_c32_fwd
    bf11_flip_v5_version
    bf11_v5_measure_sidecar_from_bf11
    bf11_v5_build_sidecar_from_bf11
    bf11_flip_v5_c1_fwd
    bf11_flip_dual_kv_v5_c1_fwd
    bf11_flip_v5_c32_fwd
    bf11_flip_dual_kv_v5_c32_fwd
    bf11_flip_v9_version
    bf11_dyn_probe_v9_fwd
    bf11_flip_dual_kv_dyn_v9_c1_fwd
    bf11_flip_dual_kv_dyn_v9_c32_fwd
    bf11_flip_v12_version
    bf11_flip_v12_c1_fwd
    bf11_flip_dual_kv_v12_c1_fwd
    bf11_flip_v12_c32_fwd
    bf11_flip_dual_kv_v12_c32_fwd
    bf11_flip_dual_kv_dyn_v12_c1_fwd
    bf11_flip_dual_kv_dyn_v12_c32_fwd
    bf11_unpack_bf16_v12_fwd
)

SYMS="$(nm -D --defined-only "${OUT_SO}" | awk '{print $3}')"
MISSING=0
for sym in "${REQUIRED_SYMS[@]}"; do
    if ! grep -qx "${sym}" <<< "${SYMS}"; then
        echo "FATAL: missing exported symbol '${sym}' in ${OUT_SO}" >&2
        MISSING=1
    fi
done
if [[ "${MISSING}" -ne 0 ]]; then
    exit 1
fi
echo "[ABI OK] ${OUT_SO}: verified all ${#REQUIRED_SYMS[@]} exported symbols."
