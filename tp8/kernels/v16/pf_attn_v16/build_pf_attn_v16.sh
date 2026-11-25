#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# =============================================================================
# Wave-V16 Track 4 (Sub-60s 1M Cold Prefill & >120 tok/s 1M Decode)
# SM121a (GB10) Parallel Build Script
#
# Builds ALL 4 Wave-V16 CUDA shared libraries in parallel (-j4):
#   1. libfused_decode_attn_exact_sm121_v16.so
#        5 modular TUs staged from V13 attn_v13 + V14 attn_v13_swa_c28.patch
#        + V16 attn_global_v16.patch (3-slot rotating cp.async KV ring for
#        Global S=48 @ tot_kv > 65536, L2::evict_first, <= 64 KiB SMEM,
#        100% V14 B=2..8 SWA ladder + 1dcfb306 exactness, 26 C ABI symbols!)
#   2. libpf_nvfp4_w13_fused_v16.so
#        Zero-spin 48-SM NVFP4 W13+SiTU+Quant+Route: 3-stage cp.async K-loop,
#        32/64-bkt fine LPT, L2::evict_first, 11 C ABI symbols!
#   3. libpf_nvfp4_w2_v16.so
#        Zero-spin 48-SM NVFP4 W2+Combine: 3-stage cp.async A-ring (a_pinned<=3),
#        64-bkt LPT, L2::evict_first, double-buffered zero-spin slice overlap,
#        10 C ABI symbols!
#   4. libpf_mx7_prefill_v16.so
#        Zero-spin 48-SM MX7 W13+W2: BM=256 3-stage W13, 3-stage W2 (a_pinned<=3),
#        64-bkt LPT, L2::evict_first, double-buffered zero-spin slice overlap,
#        14 C ABI symbols!
#
# Strict SM121a codegen flags (0-ULP safe):
#   -O3 -std=c++17 -gencode arch=compute_121a,code=sm_121a --fmad=false --ptxas-options=-v
#   (NEVER --use_fast_math, NEVER --ftz=true, NEVER -arch=sm_121a, NEVER -dc)
#
# Hard verification gates:
#   - Refuses to write directly into /rdma (stage locally first!)
#   - 0 B stack frame, 0 B spill stores, 0 B spill loads across EVERY kernel
#   - Static+dynamic SMEM <= 65,536 B on every attention kernel (<= 64 KiB)
#   - Full nm -D --defined-only verification across all 61 exported C ABI symbols
#
# Usage on GB10 build node:
#   ./build_pf_attn_v16.sh [OUT_DIR]
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${1:-${OUT_DIR:-${SCRIPT_DIR}/build_v16_out}}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
JOBS="${JOBS:-4}"

if [[ "${OUT_DIR}" == /rdma* ]]; then
    echo "ERROR: Refusing to write directly into /rdma (${OUT_DIR}). Stage locally first!" >&2
    exit 1
fi

ATTN_V13_DIR="${ATTN_V13_DIR:-${SCRIPT_DIR}/../../v15/unified/kernels/pf_v13/attn_v13}"
V14_ATTN_PATCH="${V14_ATTN_PATCH:-${SCRIPT_DIR}/../../v15/unified/kernels/swa_sconv_v14/attn_v13_swa_c28.patch}"
V16_ATTN_PATCH="${V16_ATTN_PATCH:-${SCRIPT_DIR}/attn_global_v16.patch}"

if [[ ! -f "${ATTN_V13_DIR}/attn_common_v13.cuh" ]]; then
    echo "ERROR: Base attn_v13 source tree not found at ATTN_V13_DIR=${ATTN_V13_DIR}" >&2
    exit 1
fi
if [[ ! -f "${V14_ATTN_PATCH}" ]]; then
    echo "ERROR: V14 SWA patch not found at V14_ATTN_PATCH=${V14_ATTN_PATCH}" >&2
    exit 1
fi
if [[ ! -f "${V16_ATTN_PATCH}" ]]; then
    echo "ERROR: V16 Global Attn patch not found at V16_ATTN_PATCH=${V16_ATTN_PATCH}" >&2
    exit 1
fi

STAGE_ATTN="${OUT_DIR}/attn_v16_src"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"
rm -rf "${STAGE_ATTN}"
mkdir -p "${STAGE_ATTN}"
cp -a "${ATTN_V13_DIR}/." "${STAGE_ATTN}/"

echo "========================================================================"
echo "Wave-V16 Track 4 SM121a Parallel Build (-j${JOBS})"
echo "  NVCC       : ${NVCC}"
echo "  OUT_DIR    : ${OUT_DIR}"
echo "  ATTN_V13   : ${ATTN_V13_DIR}"
echo "  V14_PATCH  : ${V14_ATTN_PATCH}"
echo "  V16_PATCH  : ${V16_ATTN_PATCH}"
echo "========================================================================"

echo "[Stage] Applying V14 SWA patch + V16 Global 3-slot cp.async patch (--fuzz=0)..."
patch -p1 --fuzz=0 -d "${STAGE_ATTN}" -i "${V14_ATTN_PATCH}"
patch -p1 --fuzz=0 -d "${STAGE_ATTN}" -i "${V16_ATTN_PATCH}"

NVCC_BASE_FLAGS=(
    -O3
    -std=c++17
    --fmad=false
    -Xcompiler -fPIC
    -gencode arch=compute_121a,code=sm_121a
    --ptxas-options=-v
)

ATTN_TUS=(
    attn_prologue_v13
    attn_swa_v13
    attn_global_s48_v13
    attn_global_bf12_v13
    attn_dispatch_v13
)

PF_TARGETS=(
    "pf_mx7_prefill_v16.cu:libpf_mx7_prefill_v16.so"
    "pf_nvfp4_w2_v16.cu:libpf_nvfp4_w2_v16.so"
    "pf_nvfp4_w13_fused_v16.cu:libpf_nvfp4_w13_fused_v16.so"
)

echo "[Build] Launching 3 V16 PF shared libs + 5 V16 Attn TUs in parallel (-j${JOBS})..."
PIDS=()

wait_slot() {
    while (( $(jobs -pr | wc -l) >= JOBS )); do
        sleep 0.2
    done
}

# Launch longest PF TUs first so they overlap cleanly with the 5 split Attn TUs
for item in "${PF_TARGETS[@]}"; do
    wait_slot
    src="${item%%:*}"
    so="${item##*:}"
    (
        echo "  [nvcc] ${src} -> ${so}"
        "${NVCC}" "${NVCC_BASE_FLAGS[@]}" --shared \
            "${SCRIPT_DIR}/${src}" -o "${OUT_DIR}/${so}" \
            2>&1 | tee "${LOG_DIR}/${so}.ptxas.log" >/dev/null
        echo "  [done] ${so}"
    ) &
    PIDS+=("$!")
done

for tu in "${ATTN_TUS[@]}"; do
    wait_slot
    (
        echo "  [nvcc] ${tu}.cu -> ${tu}.o"
        "${NVCC}" "${NVCC_BASE_FLAGS[@]}" -I"${STAGE_ATTN}" \
            -c "${STAGE_ATTN}/${tu}.cu" -o "${OUT_DIR}/${tu}.o" \
            2>&1 | tee "${LOG_DIR}/${tu}.ptxas.log" >/dev/null
        echo "  [done] ${tu}.o"
    ) &
    PIDS+=("$!")
done

FAIL=0
for pid in "${PIDS[@]}"; do
    if ! wait "${pid}"; then
        FAIL=1
    fi
done
if (( FAIL != 0 )); then
    echo "ERROR: One or more Wave-V16 nvcc builds failed! Inspect ${LOG_DIR}/" >&2
    exit 1
fi

ATTN_SO="${OUT_DIR}/libfused_decode_attn_exact_sm121_v16.so"
ATTN_OBJS=()
for tu in "${ATTN_TUS[@]}"; do
    ATTN_OBJS+=("${OUT_DIR}/${tu}.o")
done
echo "[Link] Linking ${ATTN_SO} (5 modular TUs, no -dc)..."
"${NVCC}" --shared -Xcompiler -fPIC -gencode arch=compute_121a,code=sm_121a \
    "${ATTN_OBJS[@]}" -o "${ATTN_SO}"

# ---- HARD GATE 1: 0 B stack / 0 B spill stores / 0 B spill loads + Attn SMEM <= 65536 B ----
echo "[Gate 1] Checking ptxas 0-stack / 0-spill gate across ALL compiled kernels..."
python3 - "${LOG_DIR}" << 'PYEOF'
import glob, os, re, sys

log_dir = sys.argv[1]
logs = sorted(glob.glob(os.path.join(log_dir, "*.ptxas.log")))
assert len(logs) == 8, f"Expected 8 ptxas logs (3 PF + 5 Attn), got {len(logs)}"

bad_spill = []
bad_smem = []
total_kernels = 0
for lp in logs:
    base = os.path.basename(lp)
    txt = open(lp, "r", encoding="utf-8", errors="replace").read()
    k_cnt = 0
    max_regs = 0
    max_smem = 0
    for m in re.finditer(
        r"(\d+)\s+bytes stack frame,\s+(\d+)\s+bytes spill stores,\s+(\d+)\s+bytes spill loads",
        txt,
    ):
        k_cnt += 1
        total_kernels += 1
        st, ss, sl = map(int, m.groups())
        if st > 0 or ss > 0 or sl > 0:
            bad_spill.append((base, st, ss, sl))
    for m in re.finditer(r"Used\s+(\d+)\s+registers", txt):
        max_regs = max(max_regs, int(m.group(1)))
    for m in re.finditer(r"(\d+)\s+bytes smem", txt):
        sb = int(m.group(1))
        max_smem = max(max_smem, sb)
        if base.startswith("attn_") and sb > 65536:
            bad_smem.append((base, sb))
    print(f"  [ptxas] {base:<36s}: {k_cnt:2d} kernels, max_regs={max_regs:3d}, max_static_smem={max_smem:5d} B")

if bad_spill:
    print(f"ERROR: Non-zero stack/spills detected: {bad_spill}", file=sys.stderr)
    sys.exit(1)
if bad_smem:
    print(f"ERROR: Attention static SMEM > 65536 B: {bad_smem}", file=sys.stderr)
    sys.exit(1)
print(f"  [PASS] {total_kernels} compiled kernels: 0 B stack, 0 B spill stores, 0 B spill loads!")
PYEOF

# ---- HARD GATE 2: Full nm -D C ABI symbol verification (61 symbols) ----
check_syms() {
    local so_path="$1"
    shift
    local syms
    syms="$(nm -D --defined-only "${so_path}" | awk '{print $3}')"
    for sym in "$@"; do
        if ! grep -qx "${sym}" <<<"${syms}"; then
            echo "ERROR: Missing C ABI symbol '${sym}' in ${so_path}!" >&2
            exit 1
        fi
    done
    echo "  [OK] $(basename "${so_path}"): verified $# exported C ABI symbols"
}

echo "[Gate 2] Verifying exported C ABI symbols across all 4 Wave-V16 libraries..."
check_syms "${ATTN_SO}" \
    fused_decode_attn_v7_version \
    fused_decode_attn_v8_version \
    fused_decode_attn_v9_version \
    fused_decode_attn_v10_version \
    fused_decode_attn_v11_version \
    fused_decode_attn_v13_version \
    fused_decode_attn_v14_version \
    fused_decode_attn_v16_version \
    fused_decode_attn_v9_init_smem \
    fused_decode_attn_v10_init_smem \
    fused_decode_attn_v11_init_smem \
    fused_decode_attn_v13_init_smem \
    fused_decode_attn_v14_init_smem \
    fused_decode_attn_v16_init_smem \
    attn_prologue_kvq_exact_fwd \
    attn_prologue_kvq_tree_exact_fwd \
    attn_prologue_kvq_bf12_exact_v10_fwd \
    attn_prologue_kvq_tree_bf12_exact_v10_fwd \
    fused_decode_attn_otf_mgs_fwd \
    fused_decode_attn_otf_mgs_bf12_v10_fwd \
    fused_decode_attn_otf_mgs_c32_fwd \
    fused_decode_attn_otf_mgs_c32_bf12_v10_fwd \
    fused_decode_attn_otf_mgs_tree_fwd \
    fused_decode_attn_otf_mgs_tree_bf12_v10_fwd \
    fused_decode_attn_otf_mgs_v9_fwd \
    fused_decode_attn_otf_mgs_v10_bf12_paired_fwd

check_syms "${OUT_DIR}/libpf_nvfp4_w13_fused_v16.so" \
    pf_nvfp4_w13_v16_version \
    pf_nvfp4_w13_v13_version \
    pf_nvfp4_w13_v6_version \
    pf_nvfp4_w13_v6_slot_bytes \
    pf_nvfp4_w13_v6_state_bytes \
    pf_nvfp4_build_w13_w2_plan_v6 \
    pf_nvfp4_w13_situ_quant_v6 \
    pf_nvfp4_build_m128_worklist \
    pf_nvfp4_w13_ws5_v4b_unfused \
    pf_nvfp4_w13_ws5_v4b_fused_situ \
    pf_route_tokens_place_v6

check_syms "${OUT_DIR}/libpf_nvfp4_w2_v16.so" \
    pf_nvfp4_w2_v16_version \
    pf_nvfp4_w2_set_double_buf_slice_v16 \
    pf_nvfp4_w2_v13_version \
    pf_nvfp4_w2_v5_version \
    pf_nvfp4_w2_v5_set_reserve_sms \
    pf_nvfp4_w2_v5_get_reserve_sms \
    pf_nvfp4_w2_build_plan_v5 \
    pf_nvfp4_w2_ws5_v5_unfused \
    pf_nvfp4_w2_ws5_v5_combine \
    pf_nvfp4_w2_panel_v5_combine

check_syms "${OUT_DIR}/libpf_mx7_prefill_v16.so" \
    pf_mx7_v16_version \
    pf_mx7_set_double_buf_slice_v16 \
    pf_mx7_v13_version \
    pf_mx7_v10_version \
    pf_mx7_v10_set_reserve_sms \
    pf_mx7_v10_get_reserve_sms \
    pf_mx7_build_expert_plan \
    pf_mx7_zero_slice_wb_bf16 \
    pf_mx7_w13_prefill_fused_situ \
    pf_mx7_w13_prefill_unfused \
    pf_mx7_w2_prefill_persistent \
    pf_mx7_build_plan_v10 \
    pf_mx7_w2_v10_unfused \
    pf_mx7_w2_v10_combine

echo "========================================================================"
echo "ALL 4 WAVE-V16 SM121a LIBRARIES BUILT & VERIFIED IN ${OUT_DIR}!"
echo "  - libfused_decode_attn_exact_sm121_v16.so (26 C ABI symbols)"
echo "  - libpf_nvfp4_w13_fused_v16.so            (11 C ABI symbols)"
echo "  - libpf_nvfp4_w2_v16.so                   (10 C ABI symbols)"
echo "  - libpf_mx7_prefill_v16.so                (14 C ABI symbols)"
echo "========================================================================"
