#!/usr/bin/env bash
set -euo pipefail

# Wave-V9 Build Script for libdrafter_u4_tc_v9.so & libdrafter_attn_v5_fixed.so on GB10 SM121a
# Mandatory rules:
#   - -gencode arch=compute_121a,code=sm_121a
#   - --fmad=false (strict IEEE, zero fast-math or flush-to-zero)
#   - 0 bytes stack frame, 0 spill stores, 0 spill loads (hard-enforced!)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${1:-${SCRIPT_DIR}}"
mkdir -p "${OUT_DIR}"

NVCC="${NVCC:-nvcc}"
if ! command -v "${NVCC}" >/dev/null 2>&1; then
    echo "ERROR: nvcc not found in PATH" >&2
    exit 1
fi

build_one() {
    local src="$1"
    local out="$2"
    local log="$3"
    local sym_re="$4"

    echo "[build_drafter_v9] Compiling ${src} -> ${out}"
    "${NVCC}" \
        -O3 -std=c++17 \
        -gencode arch=compute_121a,code=sm_121a \
        --fmad=false \
        --shared -Xcompiler -fPIC \
        --ptxas-options=-v \
        -o "${out}" \
        "${src}" 2>&1 | tee "${log}"

    echo "[build_drafter_v9] Enforcing SM121a 0-spill / 0-stack-frame gate on ${out}..."
    if grep -E "([1-9][0-9]* bytes stack frame|[1-9][0-9]* bytes spill stores|[1-9][0-9]* bytes spill loads)" "${log}"; then
        echo "ERROR: Register spill or stack frame detected in ${log}!" >&2
        rm -f "${out}"
        exit 2
    fi

    echo "[build_drafter_v9] Top register counts (${out}):"
    grep -E "Used [0-9]+ registers" "${log}" | sort -t' ' -k3 -rn | head -n 8 || true

    echo "[build_drafter_v9] Verifying exported C ABI symbols (${out})..."
    nm -D "${out}" | grep -E "${sym_re}"
    echo "[build_drafter_v9] SUCCESS: ${out} (0 spills, 0 stack frames)."
}

build_one \
    "${SCRIPT_DIR}/drafter_u4_tc_v9.cu" \
    "${OUT_DIR}/libdrafter_u4_tc_v9.so" \
    "${SCRIPT_DIR}/build_drafter_u4_tc_v9.ptxas.log" \
    " (drafter_u4_tc_v9_version|drafter_u4_tc_v9_fwd)$"

build_one \
    "${SCRIPT_DIR}/drafter_attn_v5_fixed.cu" \
    "${OUT_DIR}/libdrafter_attn_v5_fixed.so" \
    "${SCRIPT_DIR}/build_drafter_attn_v5_fixed.ptxas.log" \
    " (drafter_attn_v5_fwd|drafter_attn_c1_v4_fwd)$"
