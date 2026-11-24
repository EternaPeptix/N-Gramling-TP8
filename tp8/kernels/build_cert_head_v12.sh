#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Strict SM121a (NVIDIA DGX Spark GB10) Build Script for `cert_head_fast_v12.cu`
#
# Builds `libcert_head_fast_v12.so` (100% drop-in ABI superset of v9/c48/v7) with:
#   - Mandatory `-gencode arch=compute_121a,code=sm_121a --fmad=false --ptxas-options=-v`
#   - ZERO `--use_fast_math`, ZERO `--ftz=true`, ZERO `-arch=sm_121a`
#   - Hard-fail if ANY entry function reports > 0 B stack frame, spill stores, or spill loads
#   - Hard-fail if ANY exported C ABI symbol is missing in `nm -D`
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${OUT_DIR:-${1:-/tmp/wave_v12_drafter_spec_out}}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
SRC="${SCRIPT_DIR}/cert_head_fast_v12.cu"
OUT_SO="${OUT_DIR}/libcert_head_fast_v12.so"
PTX_LOG="${OUT_DIR}/cert_head_fast_v12.ptxas.log"

mkdir -p "${OUT_DIR}"

echo "[build-v12] Compiling ${SRC} -> ${OUT_SO} (SM121a, --fmad=false)"
"${NVCC}" \
    -O3 -std=c++17 \
    -gencode arch=compute_121a,code=sm_121a \
    --fmad=false \
    --shared -Xcompiler -fPIC \
    --ptxas-options=-v \
    "${SRC}" -o "${OUT_SO}" 2>&1 | tee "${PTX_LOG}"

echo "[build-v12] Verifying 0 B stack frames and 0 B spills in ${PTX_LOG}..."
python3 - "${PTX_LOG}" <<'PY'
import re, sys, pathlib
log = pathlib.Path(sys.argv[1]).read_text()
entries = re.findall(r"Compiling entry function '([^']+)'", log)
stacks = [int(m) for m in re.findall(r"(\d+)\s+bytes stack frame", log)]
sp_st  = [int(m) for m in re.findall(r"(\d+)\s+bytes spill stores", log)]
sp_ld  = [int(m) for m in re.findall(r"(\d+)\s+bytes spill loads", log)]
regs   = [int(m) for m in re.findall(r"Used\s+(\d+)\s+registers", log)]
print(f"  entries={len(entries)}, max_regs={max(regs) if regs else 0}, "
      f"max_stack={max(stacks) if stacks else 0}B, "
      f"max_spill_st={max(sp_st) if sp_st else 0}B, "
      f"max_spill_ld={max(sp_ld) if sp_ld else 0}B")
bad = []
if any(v > 0 for v in stacks):
    bad.append(f"stack_frame > 0: {max(stacks)}B")
if any(v > 0 for v in sp_st):
    bad.append(f"spill_stores > 0: {max(sp_st)}B")
if any(v > 0 for v in sp_ld):
    bad.append(f"spill_loads > 0: {max(sp_ld)}B")
if not entries:
    bad.append("no compiled entry functions found in ptxas log")
if bad:
    raise SystemExit("[FAIL] ptxas hygiene gate failed: " + "; ".join(bad))
print("  [PASS] 0 B stack frames, 0 B spill stores, 0 B spill loads across all entries!")
PY

echo "[build-v12] Verifying required C ABI exports via nm -D..."
SYMS=$(nm -D --defined-only "${OUT_SO}")
REQ_SYMS=(
    cert_head_v7_pre_screen
    cert_head_v7_post_refine
    cert_head_v7_post_refine_c48
    cert_head_v9_post_refine
    cert_head_v7_exact_all
    cert_head_v7_version
    cert_head_v12_version
    cert_head_v7_c48_max_m
    cert_head_v9_max_m
)
for s in "${REQ_SYMS[@]}"; do
    if ! echo "${SYMS}" | grep -qw "${s}"; then
        echo "[FAIL] Missing exported symbol: ${s}" >&2
        exit 1
    fi
done
echo "[PASS] Built ${OUT_SO} with all ${#REQ_SYMS[@]} required C ABI symbols!"
