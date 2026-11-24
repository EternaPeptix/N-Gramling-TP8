#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Build rdma_ar_c32_v9.cu -> ${OUT_DIR}/rdma_ar_ext_c32_v9.so on GB10 (SM121a) inside vllm-node container.
# STRICT 0-ULP COMPILER RULES ENFORCED:
#   - Uses `-gencode arch=compute_121a,code=sm121a --fmad=false`
#   - NEVER uses `-arch=sm_121a`, `--ftz=true`, or `--use_fast_math`
#   - Default OUT_DIR=/tmp/c28_perf_v9_out (NEVER writes directly to :ro /rdma!)
#   - Audits `ptxas -v` output and HARD-FAILS on:
#       * ANY register spill (> 0 bytes spill stores/loads) in ANY kernel
#       * ANY non-zero stack frame (> 0 bytes stack frame) in ANY v9 `ar_kernel_c32_sconv_attnres_v9` kernel
#       * v9 kernel count != 36, static SMEM > 2048 B (protects 96 KiB dynamic SMEM budget <= 99 KiB),
#         or v9 max regs > 128!
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${1:-${SCRIPT_DIR}/rdma_ar_c32_v9.cu}"
OUT_DIR="${OUT_DIR:-/tmp/c28_perf_v9_out}"
OUT="${2:-${OUT_DIR}/rdma_ar_ext_c32_v9.so}"
MOD_NAME="${3:-rdma_ar_ext}"

export TRITON_PTXAS_BLACKWELL_PATH="${TRITON_PTXAS_BLACKWELL_PATH:-/usr/local/cuda/bin/ptxas}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
PYTHON="${PYTHON:-python3}"

if [[ ! -f "${SRC}" ]]; then
  echo "ERROR: Source file not found: ${SRC}" >&2
  exit 1
fi

readarray -t PY_FLAGS < <("${PYTHON}" - <<'PY'
import os
import sysconfig
import torch
from torch.utils import cpp_extension

for p in cpp_extension.include_paths(device_type="cuda"):
    print(f"-I{p}")
for extra_inc in (
    "/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include",
    "/usr/local/cuda/targets/sbsa-linux/include",
):
    if os.path.isdir(extra_inc):
        print(f"-I{extra_inc}")
for p in cpp_extension.library_paths(device_type="cuda"):
    print(f"-L{p}")
    print("-Xlinker")
    print(f"-rpath,{p}")
inc = sysconfig.get_path("include")
if inc:
    print(f"-I{inc}")
abi = int(bool(getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", False)))
print(f"-D_GLIBCXX_USE_CXX11_ABI={abi}")
PY
)

PTXAS_LOG="$(mktemp /tmp/rdma_ar_c32_v9_ptxas.XXXXXX.log)"
trap 'rm -f "${PTXAS_LOG}"' EXIT

mkdir -p "$(dirname "${OUT}")"

echo "[build_rdma_ar_v9] Compiling ${SRC} -> ${OUT} (SM121a, --fmad=false, 0-stack & 0-spill gate)..."
"${NVCC}" -O3 -std=c++20 --shared \
  -Xcompiler -fPIC,-Wall,-Wextra,-Wno-unused-parameter \
  -gencode arch=compute_121a,code=sm121a \
  --fmad=false \
  -Xptxas -v \
  -DTORCH_EXTENSION_NAME="${MOD_NAME}" \
  -DTORCH_API_INCLUDE_EXTENSION_H \
  "${PY_FLAGS[@]}" \
  "${SRC}" \
  -lc10 -lc10_cuda -ltorch -ltorch_cpu -ltorch_cuda -ltorch_python -libverbs -lpthread \
  -o "${OUT}" 2>&1 | tee "${PTXAS_LOG}"

echo "[build_rdma_ar_v9] Auditing ptxas register, stack frame, SMEM, & spill report..."
"${PYTHON}" - "${PTXAS_LOG}" <<'PY'
import re
import sys
from pathlib import Path

log = Path(sys.argv[1]).read_text(errors="replace")
cur_fn = None
v9_count = 0
v9_min_regs = 999
v9_max_regs = 0
v9_max_smem = 0
bad_spills = []
bad_v9_stack = []

for line in log.splitlines():
    m_fn = re.search(r"Compiling entry function '([^']+)'", line)
    if m_fn:
        cur_fn = m_fn.group(1)
        if "ar_kernel_c32_sconv_attnres_v9" in cur_fn:
            v9_count += 1
        continue
    m_sp = re.search(r"(\d+)\s+bytes stack frame,\s*(\d+)\s+bytes spill stores,\s*(\d+)\s+bytes spill loads", line)
    if m_sp:
        stk, st, ld = int(m_sp.group(1)), int(m_sp.group(2)), int(m_sp.group(3))
        if st > 0 or ld > 0:
            bad_spills.append((cur_fn, stk, st, ld))
        if cur_fn and "ar_kernel_c32_sconv_attnres_v9" in cur_fn and stk > 0:
            bad_v9_stack.append((cur_fn, stk))
        continue
    m_reg = re.search(r"Used\s+(\d+)\s+registers", line)
    if m_reg and cur_fn and "ar_kernel_c32_sconv_attnres_v9" in cur_fn:
        regs = int(m_reg.group(1))
        v9_min_regs = min(v9_min_regs, regs)
        v9_max_regs = max(v9_max_regs, regs)
        m_sm = re.search(r"(\d+)\s+bytes smem", line)
        if m_sm:
            v9_max_smem = max(v9_max_smem, int(m_sm.group(1)))

print(f"  v9 ar_kernel_c32_sconv_attnres_v9 kernels compiled: {v9_count} (expected 36)")
print(f"  v9 registers/thread: min={v9_min_regs}, max={v9_max_regs} (target <= 96, hard limit <= 128)")
print(f"  v9 max static SMEM/CTA: {v9_max_smem} B (limit <= 2048 B; + 92160 B dyn SMEM <= 99 KiB)")

if v9_count != 36:
    print(f"FATAL: Expected 36 ar_kernel_c32_sconv_attnres_v9 entry functions, found {v9_count}!", file=sys.stderr)
    sys.exit(2)
if bad_spills:
    print(f"FATAL: Non-zero register spills detected: {bad_spills[:8]}", file=sys.stderr)
    sys.exit(2)
if bad_v9_stack:
    print(f"FATAL: Non-zero stack frame detected in v9 kernels: {bad_v9_stack[:8]}", file=sys.stderr)
    sys.exit(2)
if v9_max_regs > 128:
    print(f"FATAL: v9 max registers {v9_max_regs} exceeds 128!", file=sys.stderr)
    sys.exit(2)
if v9_max_smem > 2048:
    print(f"FATAL: v9 static SMEM {v9_max_smem} B exceeds 2048 B budget!", file=sys.stderr)
    sys.exit(2)
if v9_max_regs > 104:
    print(f"WARNING: v9 max registers {v9_max_regs} > 104 (still <= 128 co-residency limit).", file=sys.stderr)
print("[build_rdma_ar_v9] ALL PTXAS GATES PASSED!")
PY

ls -lh "${OUT}"
