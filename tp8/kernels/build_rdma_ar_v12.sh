#!/usr/bin/env bash
# Wave-V12 SM121a Build Script for rdma_ar_ext_c32_v12.so
#   - Strict SM121a codegen (-gencode arch=compute_121a,code=sm_121a --fmad=false)
#   - Strict IEEE FP32 (no fast-math, no flush-to-zero)
#   - Hard ptxas gate: 0 B stack frame, 0 B spill stores, 0 B spill loads across ALL v12 kernels!
#   - Strict ABI superset of rdma_ar_ext_c32_v9.so (drop-in replacement via RDMA_AR_EXT=/rdma/rdma_ar_ext_c32_v12.so)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${OUT_DIR:-/rdma}"
LOG_DIR="${LOG_DIR:-/tmp/rdma_ar_v12_build}"
NVCC="${NVCC:-nvcc}"
PYTHON="${PYTHON:-python3}"
MOD_NAME="${RDMA_MOD_NAME:-rdma_ar_ext}"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"
RDMA_SO="${OUT_DIR}/rdma_ar_ext_c32_v12.so"
RDMA_LOG="${LOG_DIR}/rdma_ar_c32_v12.ptxas.log"

echo "=== Building ${RDMA_SO} (SM121a, --fmad=false, pybind11 + libibverbs) ==="

readarray -t PY_FLAGS < <("${PYTHON}" - <<'PY'
import os, sysconfig, torch
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
    print(f"-Xlinker")
    print(f"-rpath={p}")
inc = sysconfig.get_path("include")
if inc:
    print(f"-I{inc}")
abi = int(bool(getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", False)))
print(f"-D_GLIBCXX_USE_CXX11_ABI={abi}")
PY
)

"${NVCC}" -O3 -std=c++20 --shared \
    -Xcompiler -fPIC,-Wall,-Wextra,-Wno-unused-parameter \
    -gencode arch=compute_121a,code=sm_121a \
    --fmad=false \
    --ptxas-options=-v \
    -DTORCH_EXTENSION_NAME="${MOD_NAME}" \
    -DTORCH_API_INCLUDE_EXTENSION_H \
    "${PY_FLAGS[@]}" \
    "${SCRIPT_DIR}/rdma_ar_c32_v12.cu" \
    -lc10 -lc10_cuda -ltorch -ltorch_cpu -ltorch_cuda -ltorch_python -libverbs -lpthread \
    -o "${RDMA_SO}" 2>&1 | tee "${RDMA_LOG}"

"${PYTHON}" - "${RDMA_LOG}" <<'PY'
import re, sys
from pathlib import Path

log = Path(sys.argv[1]).read_text(errors="replace")
cur_fn = None
stats = {
    "v12_sconv": {"count": 0, "max_regs": 0, "max_smem": 0, "spills": 0, "stack": 0},
    "v12_draft": {"count": 0, "max_regs": 0, "max_smem": 0, "spills": 0, "stack": 0},
    "c32_ar":    {"count": 0, "max_regs": 0, "max_smem": 0, "spills": 0, "stack": 0},
    "v9_sconv":  {"count": 0, "max_regs": 0, "max_smem": 0, "spills": 0, "stack": 0},
}

def classify(name: str) -> str | None:
    if "ar_kernel_c28_sconv_attnres_v12" in name:
        return "v12_sconv"
    if "ar_kernel_draft_c18_oneshot_h64_q_norm" in name:
        return "v12_draft"
    if "ar_kernel_c32_sconv_attnres_v9" in name:
        return "v9_sconv"
    if "ar_kernel_c32<" in name or "ar_kernel_c32IL" in name:
        return "c32_ar"
    return None

for line in log.splitlines():
    m_fn = re.search(r"Compiling entry function '([^']+)'", line)
    if m_fn:
        cur_fn = classify(m_fn.group(1))
        if cur_fn:
            stats[cur_fn]["count"] += 1
        continue
    if cur_fn is None:
        continue
    m_stk = re.search(r"(\d+)\s+bytes stack frame,\s*(\d+)\s+bytes spill stores,\s*(\d+)\s+bytes spill loads", line)
    if m_stk:
        stk, sst, sld = int(m_stk.group(1)), int(m_stk.group(2)), int(m_stk.group(3))
        stats[cur_fn]["stack"] = max(stats[cur_fn]["stack"], stk)
        stats[cur_fn]["spills"] = max(stats[cur_fn]["spills"], sst, sld)
    m_reg = re.search(r"Used\s+(\d+)\s+registers(?:,\s*(\d+)\s+bytes smem)?", line)
    if m_reg:
        stats[cur_fn]["max_regs"] = max(stats[cur_fn]["max_regs"], int(m_reg.group(1)))
        if m_reg.group(2):
            stats[cur_fn]["max_smem"] = max(stats[cur_fn]["max_smem"], int(m_reg.group(2)))

for k, v in stats.items():
    print(f"[ptxas-gate] {k:10s}: count={v['count']:2d} max_regs={v['max_regs']:3d} "
          f"max_smem={v['max_smem']:5d}B stack={v['stack']}B spills={v['spills']}B")

assert stats["v12_sconv"]["count"] == 36, f"Expected 36 v12_sconv kernels, got {stats['v12_sconv']['count']}"
assert stats["v12_draft"]["count"] == 9,  f"Expected 9 v12_draft kernels, got {stats['v12_draft']['count']}"
assert stats["v9_sconv"]["count"] == 36,  f"Expected 36 v9_sconv kernels, got {stats['v9_sconv']['count']}"
for k in ("v12_sconv", "v12_draft", "c32_ar", "v9_sconv"):
    assert stats[k]["stack"] == 0 and stats[k]["spills"] == 0, f"Spill/stack failure in {k}: {stats[k]}"

print("[ptxas-gate] ALL GATES PASSED (0 spills, 0 stack, exact kernel counts)!")
PY

echo "=== Built and verified: ${RDMA_SO} ==="
