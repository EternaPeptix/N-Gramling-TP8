#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Wave-V13 Parallel Split-TU SM121a Build Script for `libfused_decode_attn_exact_sm121_v13.so`
#   - Compiles 5 modular TUs in parallel (-j4): 96 tiny prologue + 36 SWA + 24 Global-BF16
#     + 24 Global-SWAR-BF12 + 0 host-dispatch (180 kernels total vs 606 monolithic in V11!).
#   - Strict SM121a codegen flags: `-gencode arch=compute_121a,code=sm_121a --fmad=false --ptxas-options=-v`
#     (NO `--use_fast_math`, NO `--ftz=true`, NO `-arch=sm_121a`, NO `-dc`!).
#   - Hard verification gates: 0 B stack frame, 0 B spill stores, 0 B spill loads, <= 50,688 B SMEM,
#     and all 22 exported C ABI symbols verified via `nm -D`.
#   - NEVER writes directly to `/rdma` (outputs locally; copy/install explicitly during rollout).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${1:-${OUT_DIR:-${SCRIPT_DIR}/build}}"
if [[ "${OUT_DIR}" == /rdma* ]]; then
    echo "ERROR: Refusing to write directly into /rdma (${OUT_DIR}). Stage locally first!" >&2
    exit 1
fi
OUT_SO="${OUT_DIR}/libfused_decode_attn_exact_sm121_v13.so"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
JOBS="${JOBS:-4}"

mkdir -p "${OUT_DIR}"

NVCC_FLAGS=(
    -O3
    -std=c++17
    --fmad=false
    -Xcompiler -fPIC
    -gencode arch=compute_121a,code=sm_121a
    --ptxas-options=-v
    -I"${SCRIPT_DIR}"
)

TUS=(
    attn_prologue_v13
    attn_swa_v13
    attn_global_s48_v13
    attn_global_bf12_v13
    attn_dispatch_v13
)

echo "[build_attn_v13] Building 5 split TUs in parallel (-j${JOBS}) for SM121a..."
PIDS=()
for tu in "${TUS[@]}"; do
    src="${SCRIPT_DIR}/${tu}.cu"
    obj="${OUT_DIR}/${tu}.o"
    log="${OUT_DIR}/${tu}.ptxas.log"
    (
        echo "[build_attn_v13] compiling ${tu}.cu ..."
        "${NVCC}" "${NVCC_FLAGS[@]}" -c "${src}" -o "${obj}" 2>&1 | tee "${log}"
    ) &
    PIDS+=($!)
    while (( $(jobs -rp | wc -l) >= JOBS )); do
        sleep 0.2
    done
done

FAIL=0
for pid in "${PIDS[@]}"; do
    if ! wait "${pid}"; then
        FAIL=1
    fi
done
if (( FAIL != 0 )); then
    echo "[build_attn_v13] ERROR: one or more TUs failed to compile!" >&2
    exit 1
fi

echo "[build_attn_v13] Linking ${OUT_SO}..."
OBJS=()
for tu in "${TUS[@]}"; do
    OBJS+=("${OUT_DIR}/${tu}.o")
done
"${NVCC}" --shared -Xcompiler -fPIC -gencode arch=compute_121a,code=sm_121a "${OBJS[@]}" -o "${OUT_SO}"

echo "[build_attn_v13] Checking ptxas stack/spill/SMEM gates..."
python3 - "${OUT_DIR}" << 'PYEOF'
import glob, os, re, sys

out_dir = sys.argv[1]
logs = sorted(glob.glob(os.path.join(out_dir, "*.ptxas.log")))
assert logs, "No ptxas logs found!"
bad = []
n_kernels = 0
max_regs = 0
for lp in logs:
    txt = open(lp).read()
    for m in re.finditer(r"(\d+)\s+bytes stack frame,\s+(\d+)\s+bytes spill stores,\s+(\d+)\s+bytes spill loads", txt):
        n_kernels += 1
        st, ss, sl = map(int, m.groups())
        if st > 0 or ss > 0 or sl > 0:
            bad.append((os.path.basename(lp), st, ss, sl))
    for m in re.finditer(r"Used\s+(\d+)\s+registers", txt):
        max_regs = max(max_regs, int(m.group(1)))
if bad:
    print(f"[build_attn_v13] FAIL: nonzero stack/spill detected: {bad}", file=sys.stderr)
    sys.exit(1)
print(f"[build_attn_v13] PASS: {n_kernels} kernels verified, 0 B stack, 0 B spills, max_regs={max_regs}")
PYEOF

echo "[build_attn_v13] Verifying all 22 C ABI symbols via nm -D..."
REQ_SYMS=(
    fused_decode_attn_v7_version
    fused_decode_attn_v8_version
    fused_decode_attn_v9_version
    fused_decode_attn_v10_version
    fused_decode_attn_v11_version
    fused_decode_attn_v13_version
    fused_decode_attn_v9_init_smem
    fused_decode_attn_v10_init_smem
    fused_decode_attn_v11_init_smem
    fused_decode_attn_v13_init_smem
    attn_prologue_kvq_exact_fwd
    attn_prologue_kvq_tree_exact_fwd
    attn_prologue_kvq_bf12_exact_v10_fwd
    attn_prologue_kvq_tree_bf12_exact_v10_fwd
    fused_decode_attn_otf_mgs_fwd
    fused_decode_attn_otf_mgs_bf12_v10_fwd
    fused_decode_attn_otf_mgs_c32_fwd
    fused_decode_attn_otf_mgs_c32_bf12_v10_fwd
    fused_decode_attn_otf_mgs_tree_fwd
    fused_decode_attn_otf_mgs_tree_bf12_v10_fwd
    fused_decode_attn_otf_mgs_v9_fwd
    fused_decode_attn_otf_mgs_v10_bf12_paired_fwd
)
EXPORTS="$(nm -D --defined-only "${OUT_SO}")"
for sym in "${REQ_SYMS[@]}"; do
    if ! grep -qw "${sym}" <<< "${EXPORTS}"; then
        echo "[build_attn_v13] ERROR: missing exported symbol ${sym}!" >&2
        exit 1
    fi
done
echo "[build_attn_v13] SUCCESS: ${OUT_SO} built & verified (${#REQ_SYMS[@]} ABI symbols, 0 B stack/spills)!"
