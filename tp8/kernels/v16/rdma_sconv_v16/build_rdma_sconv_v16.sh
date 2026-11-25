#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Wave-V16 Track 5 SM121a (GB10) Dual-Kernel Build & Hard PTXAS/ABI Gate
# Builds BOTH:
#   1. rdma_ar_c32_v16.cu          -> ${OUT_DIR}/rdma_ar_ext_c32_v16.so
#   2. fused_sconv_attnres_v16.cu  -> ${OUT_DIR}/libfused_sconv_attnres_v16.so
#
# Strict SM121a 0-ULP build rules:
#   - Mandatory: -gencode arch=compute_121a,code=sm_121a --fmad=false --ptxas-options=-v
#   - Forbidden: -arch=sm_121a, --ftz=true, --use_fast_math, -dc
#   - Hard gates: 0 B stack frame, 0 B spills, exact 36+36 c32/v9 template counts,
#     <= 128 regs on ar_kernel_c32 & ar_kernel_c32_sconv_attnres_v9,
#     17 C ABI symbols on libfused_sconv_attnres_v16.so, full pybind11 ABI on rdma_ar_ext_c32_v16.so.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${1:-/tmp/v16_rdma_sconv_build}"

# Safety guard: NEVER write directly to /rdma (stage to /tmp then atomic mv into /rdma!)
case "${OUT_DIR}" in
  /rdma|/rdma/*)
    echo "ERROR: Refusing to build directly into ${OUT_DIR} (/rdma). Stage in /tmp and atomic mv!" >&2
    exit 1
    ;;
esac

mkdir -p "${OUT_DIR}"

RDMA_SRC="${SCRIPT_DIR}/rdma_ar_c32_v16.cu"
SCONV_SRC="${SCRIPT_DIR}/fused_sconv_attnres_v16.cu"
RDMA_SO="${OUT_DIR}/rdma_ar_ext_c32_v16.so"
SCONV_SO="${OUT_DIR}/libfused_sconv_attnres_v16.so"
RDMA_LOG="${OUT_DIR}/rdma_v16_ptxas.log"
SCONV_LOG="${OUT_DIR}/sconv_v16_ptxas.log"

if [[ ! -f "${RDMA_SRC}" || ! -f "${SCONV_SRC}" ]]; then
  echo "ERROR: Missing source files in ${SCRIPT_DIR}" >&2
  exit 1
fi

NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
if [[ ! -x "${NVCC}" ]]; then
  NVCC="$(command -v nvcc)"
fi
PYTHON="${PYTHON:-$(command -v python3)}"

readarray -t TORCH_INFO < <("${PYTHON}" - <<'PY'
import os, sysconfig, torch
from torch.utils.cpp_extension import include_paths, library_paths
inc = include_paths() + [sysconfig.get_path("include")]
lib = library_paths()
abi = int(torch._C._GLIBCXX_USE_CXX11_ABI)
print(" ".join(f"-I{p}" for p in inc))
print(" ".join(f"-L{p}" for p in lib))
print(f"-D_GLIBCXX_USE_CXX11_ABI={abi}")
print(f"-rpath,{lib[0]}" if lib else "")
PY
)

TORCH_INCS="${TORCH_INFO[0]}"
TORCH_LIBS="${TORCH_INFO[1]}"
CXX11_ABI="${TORCH_INFO[2]}"
TORCH_RPATH="${TORCH_INFO[3]}"

STRICT_DEF=""
if [[ "${INKLING_PTXAS133_STRICT:-0}" == "1" ]]; then
  STRICT_DEF="-DINKLING_PTXAS133_STRICT"
fi

echo "[v16-build] NVCC=${NVCC}"
echo "[v16-build] OUT_DIR=${OUT_DIR}"
echo "[v16-build] Launching parallel SM121a --fmad=false builds..."

# Build 1: rdma_ar_c32_v16.cu -> rdma_ar_ext_c32_v16.so (C++20, pybind11 + ibverbs)
(
  # shellcheck disable=SC2086
  "${NVCC}" -O3 -std=c++20 --shared \
    -Xcompiler -fPIC -Xcompiler -Wall -Xcompiler -Wno-unused-function \
    -gencode arch=compute_121a,code=sm_121a \
    --fmad=false \
    --ptxas-options=-v \
    -DTORCH_EXTENSION_NAME=rdma_ar_ext \
    -DTORCH_API_INCLUDE_EXTENSION_H \
    ${CXX11_ABI} \
    ${TORCH_INCS} \
    "${RDMA_SRC}" \
    -o "${RDMA_SO}" \
    ${TORCH_LIBS} \
    -Xlinker "${TORCH_RPATH}" \
    -ltorch -ltorch_cpu -ltorch_cuda -ltorch_python -lc10 -lc10_cuda -libverbs -lpthread \
    2>&1 | tee "${RDMA_LOG}"
) &
PID_RDMA=$!

# Build 2: fused_sconv_attnres_v16.cu -> libfused_sconv_attnres_v16.so (C++17, pure C ABI)
(
  # shellcheck disable=SC2086
  "${NVCC}" -O3 -std=c++17 --shared \
    -Xcompiler -fPIC \
    -gencode arch=compute_121a,code=sm_121a \
    --fmad=false \
    --ptxas-options=-v \
    ${STRICT_DEF} \
    "${SCONV_SRC}" \
    -o "${SCONV_SO}" \
    2>&1 | tee "${SCONV_LOG}"
) &
PID_SCONV=$!

FAIL=0
wait "${PID_RDMA}"  || FAIL=1
wait "${PID_SCONV}" || FAIL=1
if [[ "${FAIL}" -ne 0 ]]; then
  echo "ERROR: One or more SM121a NVCC builds failed!" >&2
  exit 1
fi

echo "[v16-build] Running hard PTXAS 0-spill / 0-stack / register / template-count audit..."
"${PYTHON}" - "${RDMA_LOG}" "${SCONV_LOG}" <<'PY'
import re
import sys

rdma_log, sconv_log = sys.argv[1], sys.argv[2]

def parse_ptxas(path):
    entries = []
    cur = None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m_fn = re.search(r"Compiling entry function '([^']+)'", line)
            if m_fn:
                cur = {"name": m_fn.group(1), "regs": 0, "smem": 0, "stack": 0, "spill_st": 0, "spill_ld": 0}
                entries.append(cur)
                continue
            if cur is None:
                continue
            m_st = re.search(r"(\d+)\s+bytes stack frame,\s+(\d+)\s+bytes spill stores,\s+(\d+)\s+bytes spill loads", line)
            if m_st:
                cur["stack"] = int(m_st.group(1))
                cur["spill_st"] = int(m_st.group(2))
                cur["spill_ld"] = int(m_st.group(3))
            m_rg = re.search(r"Used\s+(\d+)\s+registers", line)
            if m_rg:
                cur["regs"] = int(m_rg.group(1))
            m_sm = re.search(r"(\d+)\s+bytes smem", line)
            if m_sm:
                cur["smem"] = int(m_sm.group(1))
    return entries

rdma = parse_ptxas(rdma_log)
sconv = parse_ptxas(sconv_log)
assert rdma, f"No ptxas entries found in {rdma_log}"
assert sconv, f"No ptxas entries found in {sconv_log}"

# 1. Universal 0-spill & 0-stack gate across BOTH .cu files
for tag, ents in (("rdma_v16", rdma), ("sconv_v16", sconv)):
    bad_stack = [e for e in ents if e["stack"] > 0]
    bad_spill = [e for e in ents if e["spill_st"] > 0 or e["spill_ld"] > 0]
    if bad_stack:
        raise SystemExit(f"FAIL [{tag}]: non-zero stack frame in {len(bad_stack)} kernels: {bad_stack[:5]}")
    if bad_spill:
        raise SystemExit(f"FAIL [{tag}]: non-zero register spills in {len(bad_spill)} kernels: {bad_spill[:5]}")

# 2. RDMA V16 template-count, register, & SMEM gates
c32 = [e for e in rdma if "ar_kernel_c32<" in e["name"] and "sconv" not in e["name"]]
v9  = [e for e in rdma if "ar_kernel_c32_sconv_attnres_v9" in e["name"]]
st_p = [e for e in rdma if "selftest_bf12_g128_pack_v16_kernel" in e["name"]]
st_u = [e for e in rdma if "selftest_bf12_g128_unpack_v16_kernel" in e["name"]]

assert len(c32) == 36, f"FAIL: expected 36 ar_kernel_c32 instantiations, got {len(c32)}"
assert len(v9) == 36, f"FAIL: expected 36 ar_kernel_c32_sconv_attnres_v9 instantiations, got {len(v9)}"
assert len(st_p) == 1 and len(st_u) == 1, f"FAIL: missing V16 BF12 selftest kernels ({len(st_p)}, {len(st_u)})"

max_c32_r = max(e["regs"] for e in c32)
max_c32_s = max(e["smem"] for e in c32)
max_v9_r  = max(e["regs"] for e in v9)
max_v9_s  = max(e["smem"] for e in v9)
assert max_c32_r <= 128, f"FAIL: ar_kernel_c32 max regs={max_c32_r} > 128"
assert max_c32_s <= 512, f"FAIL: ar_kernel_c32 static smem={max_c32_s} > 512"
assert max_v9_r <= 128, f"FAIL: ar_kernel_c32_sconv_attnres_v9 max regs={max_v9_r} > 128"
assert max_v9_s <= 2048, f"FAIL: ar_kernel_c32_sconv_attnres_v9 static smem={max_v9_s} > 2048"

# 3. SConv + AttnRes + RMSNorm V16 presence gates
c48 = [e for e in sconv if "fused_sconv_ar_c48_kernel_v14" in e["name"]]
sc14 = [e for e in sconv if "sconv_exact_v14_kernel" in e["name"]]
ar14 = [e for e in sconv if "attn_res_exact_v14_kernel" in e["name"]]
ar16b = [e for e in sconv if "fused_attnres_rmsnorm_v16_blockd_kernel" in e["name"]]
assert len(c48) > 0 and len(sc14) > 0 and len(ar14) > 0 and len(ar16b) > 0, (
    f"FAIL: missing expected sconv/attnres kernels: c48={len(c48)} sc14={len(sc14)} ar14={len(ar14)} ar16b={len(ar16b)}"
)

print(
    f"PASS PTXAS: rdma_v16={len(rdma)} kernels (c32={len(c32)} max_regs={max_c32_r} smem={max_c32_s}B, "
    f"v9={len(v9)} max_regs={max_v9_r} smem={max_v9_s}B), "
    f"sconv_v16={len(sconv)} kernels (c48={len(c48)} sc14={len(sc14)} ar14={len(ar14)} ar16_blockd={len(ar16b)}), "
    f"ALL 0 B stack frame & 0 B spills!"
)
PY

echo "[v16-build] Running nm -D C ABI gate on ${SCONV_SO}..."
NM_OUT="$(nm -D --defined-only "${SCONV_SO}")"
REQUIRED_C_SYMS=(
  fused_sconv_ar_v4_version
  fused_sconv_ar_v7_version
  fused_sconv_ar_v14_version
  fused_sconv_ar_v16_version
  fused_sconv_ar_c32_init_smem
  fused_sconv_ar_c48_init_smem
  fused_sconv_ar_c1_init_smem
  attn_res_exact_v14_init_smem
  fused_attnres_rmsnorm_v16_init_smem
  fused_sconv_ar_c32_row_fwd
  fused_sconv_ar_c32_blockd_fwd
  fused_sconv_ar_c48_fwd
  fused_sconv_ar_c1_fwd
  sconv_exact_v14_fwd
  sconv_exact_v16_fwd
  attn_res_exact_v14_fwd
  fused_attnres_rmsnorm_v16_fwd
)
for sym in "${REQUIRED_C_SYMS[@]}"; do
  if ! grep -Eq "[[:space:]]T[[:space:]]+${sym}$" <<< "${NM_OUT}"; then
    echo "ERROR: Missing required C ABI symbol '${sym}' in ${SCONV_SO}" >&2
    exit 1
  fi
done
echo "PASS nm -D: all ${#REQUIRED_C_SYMS[@]} C ABI symbols verified in ${SCONV_SO}"

echo "[v16-build] Running Python pybind11 ABI gate on ${RDMA_SO}..."
"${PYTHON}" - "${RDMA_SO}" <<'PY'
import importlib.util
import sys

so_path = sys.argv[1]
spec = importlib.util.spec_from_file_location("rdma_ar_ext_c32_v16", so_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

for attr in ("V13_C28_RDMA_VARIANT", "V15_RDMA_RESILIENCE_VARIANT", "V16_RDMA_REG_BF12_VARIANT"):
    val = getattr(mod, attr, None)
    assert isinstance(val, str) and len(val) > 0, f"Missing module attr {attr}"

for method in (
    "v13_info", "v15_info", "v16_info",
    "health_status", "last_error", "proxy_error", "is_closed", "abort", "close",
    "all_reduce_c32_rows", "all_reduce_c32_sconv_attnres",
):
    assert hasattr(mod.RdmaAR, method), f"Missing RdmaAR.{method}"

for method in (
    "bf12_pack_into", "bf12_unpack_into",
    "bf12_pack_v16_into", "bf12_unpack_v16_into",
    "all_reduce_c32_into", "all_reduce_c32_sconv_attnres_into",
    "set_seq_for_test", "trigger_abort_for_test", "clear_abort_for_test", "abort_flag",
):
    assert hasattr(mod.SelfTest, method), f"Missing SelfTest.{method}"

print(f"PASS pybind11 ABI: {mod.V16_RDMA_REG_BF12_VARIANT}")
PY

echo "[v16-build] ALL SM121a BUILD & ABI GATES PASSED!"
echo "  ${RDMA_SO}"
echo "  ${SCONV_SO}"
