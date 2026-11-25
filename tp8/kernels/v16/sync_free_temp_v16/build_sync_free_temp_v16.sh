#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Strict SM121a (NVIDIA DGX Spark GB10) Build Script for Wave-V16 SyncFree Temp Rejection Sampler
#
# Builds:
#   libsync_free_temp_v16.so (Unbiased Zero-Sync SM121a temp > 0 Speculative Rejection Sampler, 5 C ABI exports)
#
# Mandatory invariants:
#   - -gencode arch=compute_121a,code=sm_121a --fmad=false --ptxas-options=-v
#   - ZERO fast-math / ftz / bare sm_121a shortcuts
#   - Hard-fail if ANY entry function reports > 0 B stack frame, spill stores, or spill loads
#   - Hard-fail if ANY required C ABI export is missing in nm -D
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${OUT_DIR:-${1:-/tmp/wave_v16_temp_harvester_out}}"
NVCC="${NVCC:-$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)}"
mkdir -p "${OUT_DIR}"

COMMON_FLAGS=(
    -O3
    --shared
    -Xcompiler -fPIC
    -std=c++17
    -gencode arch=compute_121a,code=sm_121a
    --fmad=false
    --ptxas-options=-v
)

check_zero_spills() {
    local log_file="$1"
    python3 - "${log_file}" <<'PY'
import pathlib, re, sys

txt = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
entries = re.findall(r"Compiling entry function '([^']+)'", txt)
regs = [int(m) for m in re.findall(r"Used\s+(\d+)\s+registers", txt)]
cur_fn = "<unknown>"
bad = []
for line in txt.splitlines():
    m_fn = re.search(r"Compiling entry function '([^']+)'", line)
    if m_fn:
        cur_fn = m_fn.group(1)
    m_st = re.search(
        r"(\d+)\s+bytes\s+stack\s+frame,\s+(\d+)\s+bytes\s+spill\s+stores,\s+(\d+)\s+bytes\s+spill\s+loads",
        line,
    )
    if m_st:
        st, ss, sl = int(m_st.group(1)), int(m_st.group(2)), int(m_st.group(3))
        if st > 0 or ss > 0 or sl > 0:
            bad.append((cur_fn, st, ss, sl))

if not entries:
    print("  [ERROR] No compiled entry functions found in ptxas log!", file=sys.stderr)
    sys.exit(1)

if bad:
    for fn, st, ss, sl in bad[:12]:
        print(f"  [SPILL/STACK] {fn}: stack={st}B spill_st={ss}B spill_ld={sl}B", file=sys.stderr)
    sys.exit(1)

print(
    f"  [0-SPILL OK] entries={len(entries)} max_regs={max(regs) if regs else 0} "
    f"stack=0B spill_st=0B spill_ld=0B"
)
PY
}

verify_abi_symbols() {
    local so_path="$1"
    shift
    local syms
    syms="$(nm -D --defined-only "${so_path}")"
    for sym in "$@"; do
        if ! grep -Eq "[[:space:]]T[[:space:]]+${sym}$" <<< "${syms}"; then
            echo "[FAIL] Missing required exported C ABI symbol '${sym}' in ${so_path}!" >&2
            exit 1
        fi
    done
    echo "  [ABI OK] ${so_path} exports all $# required C ABI symbols."
}

SRC="${SCRIPT_DIR}/sync_free_temp_v16.cu"
SO="${OUT_DIR}/libsync_free_temp_v16.so"
LOG="${OUT_DIR}/sync_free_temp_v16.ptxas.log"

echo "=== Building ${SO} (SM121a --fmad=false, 0-spill hard gate) ==="
"${NVCC}" "${COMMON_FLAGS[@]}" \
    "${SRC}" -o "${SO}" 2>&1 | tee "${LOG}"

check_zero_spills "${LOG}"
verify_abi_symbols "${SO}" \
    sync_free_temp_v16_version \
    sync_free_temp_v16_smem_cap \
    sync_free_temp_reject_v16_fwd \
    sync_free_temp_sample_k0_v16_fwd \
    sync_free_temp_verify_rows_v16_fwd

echo "[PASS] Wave-V16 SyncFree Temp Rejection Sampler built & ABI-verified in ${OUT_DIR}!"
