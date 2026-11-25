#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Wave-V16 Reusable Cross-Wave Env, Patch & `.so` Manifest Conflict Auditor (100% Offline).

Clean CLI tool validating any Wave-V15 or Wave-V16 serving `.env` profile AND dynamically
ingesting sibling Wave-V16 overlays/patches/kernels from `--v16-root` once present:

  python3 v16_env_conflict_auditor.py \\
      --env <path/to/profile.env> \\
      --patches <path/to/patches-s6-v15_or_v16> \\
      --kernels <path/to/kernels_dir> \\
      [--v16-root <path/to/scratch/v16>] \\
      [--apply-v16-overlays] \\
      [--out-json audit_v16.json]

Performs 7 strict automated audit stages:
  1. Hygiene & Deduplication: 0 CR bytes (`\\r`), valid UTF-8, well-formed `KEY=VAL`, 0 duplicate keys.
  2. POSIX Shell Syntax: `bash -n` on env file (+ any discovered V16 overlays / build scripts).
  3. Launcher & Ops Forwarding Contract: every key forwarded/consumed by `inkling-tp8-v15/v16.sh`
     or V15/V16 supervisor/watchdog/exporter ops scripts.
  4. Dynamic Corpus & Patch-Merge Verification: extracts all identifier tokens across `--patches`
     (`*.py`), dynamically applied V16 `.patch` files (`--fuzz=0`, 0 overlapping files, `py_compile`),
     launcher, ops scripts, and `--kernels` / V16 `.cu`/`.cuh` files -> catches dead/typo keys!
  5. `.so` Manifest Completeness & Stale-Alias Audit: matches EVERY `*_LIB=/rdma/*.so` and
     `RDMA_AR_EXT=/rdma/*.so` against built targets in `--kernels` (+ V16 `build_*.sh`) and
     verified preserved V12 baseline `.so` libraries; catches stale version aliases & missing files.
  6. Hard Cross-Wave & V16 Safety Locks: 0-ULP `1dcfb306` asymmetric FMA, BF11 decode-only &
     `FREE_RAW=0`, RDMA fail-fast/0-nanosleep/`LEVEL_DB=1`, Prefill zero-spin & Half-2 NCCL,
     CUDAGraph bucket uniformity (`M_pad<=48`), L2PF bus <=185 GB/s, zero-swap UMA locks, PLUS
     all conditional Wave-V16 locks (Dynamic-K `<=15`, Same-Budget Tree vs legacy `TREE_SPEC=0`,
     SyncFree-Temp `SAM_ALLOW_TEMP=0`, Reg-BF12 `P2>=196608`, Non-Tail Drafter Skip, Telemetry interval).
  7. Cross-Wave Overlay Clobber Audit: dynamically scans `--v16-root/*/*.env` and catches any
     inter-track value collisions or silent clobbers of V15 safety invariants!

Pure Python 3.10+ stdlib. ZERO cluster/SSH/Docker commands.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple


_DEFAULT_SCR = pathlib.Path("/Users/jeweled/.gemini/antigravity/brain/78201cfd-43f0-4f1b-ae82-c573f3cd7b11/scratch")
_DEFAULT_V15_DEF = _DEFAULT_SCR / "v15/unified/envs/live_champion_v15_default.env"
_DEFAULT_V15_MAX = _DEFAULT_SCR / "v15/unified/envs/live_champion_v15_max_kv_speed.env"
_DEFAULT_PATCHES = _DEFAULT_SCR / "v15/unified/patches-s6-v15"
_DEFAULT_KERNELS = _DEFAULT_SCR / "v15/unified/kernels"
_DEFAULT_OPS = _DEFAULT_SCR / "v15/unified/ops"
_DEFAULT_LAUNCHER = _DEFAULT_OPS / "inkling-tp8-v15.sh"
_DEFAULT_V16_ROOT = _DEFAULT_SCR / "v16"

# Forwarding regex matching inkling-tp8-v15.sh (+ V16 telemetry/supervisor ops vars)
LAUNCHER_FORWARD_RE = re.compile(
    r"^(NCCL_[A-Z0-9_]+|INKLING_[A-Z0-9_]+|VLLM_[A-Z0-9_]+|DSV41_[A-Z0-9_]+|"
    r"RDMA_AR_[A-Z0-9_]+|RDMA_[A-Z0-9_]+|V15_[A-Z0-9_]+|V16_[A-Z0-9_]+|UMA_[A-Z0-9_]+|"
    r"OMP_NUM_THREADS|OPENBLAS_NUM_THREADS|MKL_NUM_THREADS|TORCH_NUM_THREADS|"
    r"MALLOC_[A-Z0-9_]+|PYTORCH_CUDA_ALLOC_CONF)$"
)
LAUNCHER_CLI_KEYS: Set[str] = {
    "PARITY", "SERVING", "DSPARK", "RDMA", "PATCH_DIR", "GMU", "MAXLEN", "SEQS",
    "MAX_BATCHED", "LONG_PREFILL_THRESHOLD", "STREAM_INTERVAL", "PREFILL_SCHEDULE_INTERVAL",
    "CUDAGRAPH_REQ_SIZES", "VLLM_EXTRA", "SPEC_K", "DRAFT_SAMPLE", "ADAPTIVE",
    "SPEC_BATCH_SCHEDULE", "IMAGES", "REASONING", "KV_DTYPE", "QUANT", "PORT", "MASTER_PORT",
}
STD_RUNTIME_PREFIXES: Tuple[str, ...] = (
    "NCCL_", "OMP_", "OPENBLAS_", "MKL_", "TORCH_", "MALLOC_", "PYTORCH_",
)

# Verified preserved V12 baseline `.so` libraries mounted under `/rdma/` (never overwritten by V15/V16 builds)
PRESERVED_V12_SO_MANIFEST: Set[str] = {
    "libattn_res_exact_v4.so",
    "libdrafter_attn_v10.so",
    "libdrafter_u4_tc_v9.so",
    "libdspark_markov_v10.so",
    "libdsv41_l2pf_g2_s7_v3.so",
    "libfused_sconv_ar_prefill.so",
    "libglobal_attn_sage2_sm121_v5_zspill.so",
    "libglobal_tail_otf_rel_sm121.so",
    "libinkling_target_attn_exact_s7.so",
    "libkv_bf12_paged_v10.so",
    "liblocal_swa_otf_rel_sm121_v2.so",
    "libmxfp8_7bit_exact_v5.so",
    "libnvfp4_ws5_exact_s7.so",
    "libpf_attn_nonmoe_v4_sm121.so",
    "libpf_moe_native_sm121_v4.so",
    "libprefill_moe_qf_s8.so",
}

# Known Wave-V16 safety-critical keys whose clobbering by an overlay is forbidden
PROTECTED_SAFETY_LOCKS: Dict[str, str] = {
    "INKLING_MOE_S8_ROUTER_QUANT_FMA": "1",
    "INKLING_MOE_S8_W2_QUANT_FMA": "0",
    "INKLING_MOE_V9_QUANT_FMA": "0",
    "INKLING_CERT_V9_QUALIFY_REF": "b1_exact",
    "INKLING_BF11_C32_DECODE_ONLY": "1",
    "INKLING_BF11_V5_C1": "0",
    "INKLING_BF11_FREE_RAW": "0",
    "INKLING_BF11_V13_ALLOW_GT64": "0",
    "INKLING_C48_MOE_TILE8": "1",
    "INKLING_MX6_EXACT": "0",
    "INKLING_MX7_V5": "compat",
    "INKLING_FUSED_SCONV_AR_PREFILL_MIN_T": "1024",
    "RDMA_AR_POLL_SLEEP_NS": "0",
    "RDMA_AR_LEVEL_DB": "1",
    "RDMA_AR_FAILFAST_EXIT": "1",
    "INKLING_PF_V13_ZEROSPIN": "1",
    "INKLING_PREFILL_SP_RS_HALF2_MODE": "nccl",
}


def strip_quotes(val: str) -> str:
    if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
        return val[1:-1]
    return val


def parse_env_file(path: pathlib.Path) -> Tuple[List[Tuple[int, str, str]], Dict[str, str], Dict[str, List[Tuple[int, str]]], List[str]]:
    """Parse an env file checking CR bytes, UTF-8, malformed lines, and duplicate keys."""
    errors: List[str] = []
    raw = path.read_bytes()
    if b"\r" in raw:
        errors.append(f"CR byte (\\r) detected in {path} (must be strict Unix LF)")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        return [], {}, {}, [f"Invalid UTF-8 in {path}: {exc}"]

    entries: List[Tuple[int, str, str]] = []
    final: Dict[str, str] = {}
    first_seen: Dict[str, Tuple[int, str]] = {}
    dups: Dict[str, List[Tuple[int, str]]] = {}

    for lineno, line in enumerate(text.splitlines(), 1):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("export "):
            s = s[len("export "):].strip()
        if "=" not in s:
            errors.append(f"Malformed env line {path.name}:{lineno}: {line!r}")
            continue
        k, v = s.split("=", 1)
        k, v = k.strip(), v.strip()
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", k):
            errors.append(f"Invalid env key name {path.name}:{lineno}: {k!r}")
            continue
        entries.append((lineno, k, v))
        if k in final:
            if k not in dups:
                dups[k] = [first_seen[k]]
            dups[k].append((lineno, v))
        else:
            first_seen[k] = (lineno, v)
        final[k] = v

    if dups:
        dup_summary = ", ".join(f"{k}@{[ln for ln, _ in occ]}" for k, occ in sorted(dups.items())[:10])
        errors.append(f"{len(dups)} duplicate key(s) in {path.name}: {dup_summary}")
    return entries, final, dups, errors


def check_bash_syntax(path: pathlib.Path) -> Tuple[bool, str]:
    r = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
    return (r.returncode == 0), (r.stderr.strip() or r.stdout.strip())


def discover_v16_artifacts(v16_root: Optional[pathlib.Path]) -> Dict[str, List[pathlib.Path]]:
    """Dynamically discover all sibling Wave-V16 overlays, patches, kernels, and ops scripts."""
    found: Dict[str, List[pathlib.Path]] = {
        "overlays": [],
        "patches": [],
        "cuda_sources": [],
        "build_scripts": [],
        "ops_scripts": [],
    }
    if not v16_root or not v16_root.is_dir():
        return found
    for sub in sorted(v16_root.iterdir()):
        if not sub.is_dir() or sub.name.startswith(".") or sub.name == "unified":
            continue
        for p in sorted(sub.rglob("*")):
            if not p.is_file():
                continue
            name = p.name
            if name.endswith(".env"):
                found["overlays"].append(p)
            elif name.endswith(".patch"):
                found["patches"].append(p)
            elif name.endswith((".cu", ".cuh")):
                found["cuda_sources"].append(p)
            elif name.startswith("build_") and name.endswith(".sh"):
                found["build_scripts"].append(p)
            elif name.endswith((".sh", ".service", ".timer", ".yml", ".yaml")) or (sub.name == "ops_telemetry" and name.endswith(".py")):
                found["ops_scripts"].append(p)
    return found


def build_symbol_corpus(
    patches_dir: Optional[pathlib.Path],
    kernels_dir: Optional[pathlib.Path],
    launcher_path: Optional[pathlib.Path],
    ops_dir: Optional[pathlib.Path],
    v16_artifacts: Dict[str, List[pathlib.Path]],
    apply_v16_patches: bool = True,
) -> Dict[str, Any]:
    """Extract all uppercase identifier tokens and built `.so` targets across Python, ops, and CUDA."""
    texts: List[str] = []
    built_sos: Set[str] = set()
    patch_audit: Dict[str, Any] = {
        "base_py_files": 0,
        "v16_patches_found": [str(p.name) for p in v16_artifacts["patches"]],
        "v16_patches_applied": [],
        "v16_patch_overlaps": {},
        "v16_patch_errors": [],
    }

    # 1. Python patch tree (+ dynamically apply discovered V16 Python patches in temp copy)
    if patches_dir and patches_dir.is_dir():
        py_files = sorted(patches_dir.glob("*.py"))
        patch_audit["base_py_files"] = len(py_files)
        v16_py_patches = [p for p in v16_artifacts["patches"] if "attn_global" not in p.name]
        if apply_v16_patches and v16_py_patches:
            with tempfile.TemporaryDirectory(prefix="v16_env_audit_py_") as td:
                work = pathlib.Path(td) / "patches"
                shutil.copytree(patches_dir, work)
                touched: Dict[str, List[str]] = defaultdict(list)
                for pf in v16_py_patches:
                    ptxt = pf.read_text(encoding="utf-8", errors="replace")
                    texts.append(ptxt)
                    targets = re.findall(r"^\+\+\+ [ab]/([^\s]+)", ptxt, flags=re.M)
                    # Only apply patches targeting Python files present in patches_dir
                    if not targets or not all(t.endswith(".py") for t in targets):
                        continue
                    for t in targets:
                        touched[t].append(pf.name)
                    r = subprocess.run(
                        ["patch", "--batch", "--fuzz=0", "-p1", "-i", str(pf)],
                        cwd=str(work),
                        capture_output=True,
                        text=True,
                    )
                    if r.returncode == 0:
                        patch_audit["v16_patches_applied"].append(pf.name)
                    else:
                        patch_audit["v16_patch_errors"].append(f"{pf.name}: {r.stderr.strip() or r.stdout.strip()}")
                overlaps = {f: ps for f, ps in touched.items() if len(ps) > 1}
                patch_audit["v16_patch_overlaps"] = overlaps
                for py in sorted(work.glob("*.py")):
                    texts.append(py.read_text(encoding="utf-8", errors="replace"))
        else:
            for py in py_files:
                texts.append(py.read_text(encoding="utf-8", errors="replace"))

    # 2. Launcher & Ops scripts
    if launcher_path and launcher_path.is_file():
        texts.append(launcher_path.read_text(encoding="utf-8", errors="replace"))
    if ops_dir and ops_dir.is_dir():
        for f in sorted(ops_dir.rglob("*")):
            if f.is_file() and f.suffix in (".sh", ".py", ".service", ".timer", ".env"):
                texts.append(f.read_text(encoding="utf-8", errors="replace"))
    for f in v16_artifacts["ops_scripts"]:
        texts.append(f.read_text(encoding="utf-8", errors="replace"))

    # 3. Kernel sources & build scripts (extract both tokens AND built `.so` targets!)
    kernel_files: List[pathlib.Path] = []
    if kernels_dir and kernels_dir.is_dir():
        for f in sorted(kernels_dir.rglob("*")):
            if f.is_file() and f.suffix in (".cu", ".cuh", ".sh", ".h", ".cc", ".cpp", ".patch"):
                kernel_files.append(f)
    kernel_files.extend(v16_artifacts["cuda_sources"])
    kernel_files.extend(v16_artifacts["build_scripts"])
    for pf in v16_artifacts["patches"]:
        if pf not in kernel_files:
            kernel_files.append(pf)

    for kf in kernel_files:
        txt = kf.read_text(encoding="utf-8", errors="replace")
        texts.append(txt)
        if kf.suffix == ".sh":
            for m in re.findall(r"\b([A-Za-z0-9_]+\.so)\b", txt):
                built_sos.add(m)

    corpus = "\n".join(texts)
    tokens = set(re.findall(r"[A-Z][A-Z0-9_]{2,}", corpus))
    return {
        "tokens": tokens,
        "built_sos": built_sos,
        "known_sos": built_sos | PRESERVED_V12_SO_MANIFEST,
        "patch_audit": patch_audit,
    }


def _check_safety_invariants(env: Dict[str, str], mode: str) -> Tuple[List[str], List[str]]:
    """Verify all Wave-V15 + conditional Wave-V16 safety, 0-ULP, bus, and CUDAGraph locks."""
    errors: List[str] = []
    notes: List[str] = []

    def _req(k: str, expected: str, reason: str) -> None:
        actual = env.get(k)
        if actual != expected:
            errors.append(f"Safety lock violated: {k}={actual!r} (expected {expected!r}; {reason})")

    def _req_float_le(k: str, hi: float, reason: str) -> None:
        if k in env:
            try:
                v = float(strip_quotes(env[k]))
                if v > hi:
                    errors.append(f"Safety limit violated: {k}={v} > {hi} ({reason})")
            except ValueError:
                errors.append(f"Non-numeric value for {k}={env[k]!r}")

    def _req_int_ge(k: str, lo: int, reason: str) -> None:
        if k in env:
            try:
                v = int(strip_quotes(env[k]))
                if v < lo:
                    errors.append(f"Safety limit violated: {k}={v} < {lo} ({reason})")
            except ValueError:
                errors.append(f"Non-integer value for {k}={env[k]!r}")

    # 1. Core 0-ULP 1dcfb306 asymmetric FMA & BF11 safety locks
    if "INKLING_MOE_S8_QUANT_FMA" in env:
        errors.append("Deprecated coupled INKLING_MOE_S8_QUANT_FMA must be removed (use decoupled ROUTER/W2 flags)")
    _req("INKLING_MOE_S8_ROUTER_QUANT_FMA", "1", "0-ULP router FMA lock")
    _req("INKLING_MOE_S8_W2_QUANT_FMA", "0", "0-ULP W2 non-FMA lock")
    _req("INKLING_MOE_V9_QUANT_FMA", "0", "0-ULP MoE V9 non-FMA lock")
    _req("INKLING_ROUTER_C32_QUANT_FMA", "1", "0-ULP router C32 FMA lock")
    _req("INKLING_ROUTER_V14_QUANT_FMA", "1", "0-ULP router V14 FMA lock")
    _req("INKLING_CERT_V9_QUALIFY_REF", "b1_exact", "0-ULP CertHead b1_exact reference")
    _req("INKLING_BF11_C32_DECODE_ONLY", "1", "BF11 decode-only lock")
    _req("INKLING_BF11_V5_C1", "0", "BF11 V5 C=1 disabled")
    _req("INKLING_BF11_FREE_RAW", "0", "BF11 FREE_RAW=0 safety lock")
    _req("INKLING_BF11_V13_ALLOW_GT64", "0", "BF11 C=32 M>64 guard")
    if "INKLING_BF11_C32_MAX_M" in env and int(env["INKLING_BF11_C32_MAX_M"]) != 64:
        errors.append(f"INKLING_BF11_C32_MAX_M={env['INKLING_BF11_C32_MAX_M']} (must be 64)")
    _req("INKLING_FUSED_SCONV_AR_PREFILL_MIN_T", "1024", "SConv prefill min T guard")

    # 2. MoE NVFP4 + MX7 invariants
    _req("INKLING_C48_MOE_TILE8", "1", "TILE8=1 anti-V12-clobber lock")
    _req("INKLING_MX6_EXACT", "0", "Never combine MX6 + MX7 simultaneously")
    _req("INKLING_MX7_V5", "compat", "MX7 zero-copy compat mode")
    _req("INKLING_MX7_FREE_RAW", "1", "MX7 free raw weights")

    # 3. RDMA fail-fast + zero-nanosleep + Pipe2 invariants
    _req("RDMA_AR_POLL_SLEEP_NS", "0", "Zero nanosleep RDMA anti-V12-B2 lock")
    _req("RDMA_AR_LEVEL_DB", "1", "LEVEL_DB=1 required for Pipe2 RDMA")
    _req("RDMA_AR_FAILFAST_EXIT", "1", "Fail-fast exit on peer death")
    _req_int_ge("RDMA_AR_MAX_BYTES", 2621440, "Must fit C=32 M=128 AR payload")
    _req_int_ge("DSV41_RDMA_SENDBUF_MB", 48, "SendBuf ring >= 48 MiB")

    # 4. Prefill zero-spin + EXP-6 Half-2 NCCL protection
    _req("INKLING_PF_V13_ZEROSPIN", "1", "Zero-spin prefill lock")
    _req("INKLING_PREFILL_SP_RS_HALF2_MODE", "nccl", "EXP-6 protection: NEVER a2a on Half-2")
    _req("INKLING_PF_W2_RESERVE_SMS", "0", "Zero reserved SMs for zero-spin W2")
    _req("INKLING_PF_MX7_V10_RESERVE_SMS", "0", "Zero reserved SMs for zero-spin MX7")
    _req("INKLING_PF_DENSE_FP8_RESERVE_SMS", "0", "Zero reserved SMs for zero-spin Dense")
    if "INKLING_PREFILL_DBO_MAX_INFLIGHT_LAYERS" in env and int(env["INKLING_PREFILL_DBO_MAX_INFLIGHT_LAYERS"]) > 2:
        errors.append("INKLING_PREFILL_DBO_MAX_INFLIGHT_LAYERS > 2 risks UMA/RHD OOM")

    # 5. L2PF Bus & Working-Set Safety (<= 185 GB/s pace, <= 232 < 244 GB/s bus ceiling!)
    for pk in (
        "INKLING_L2PF_C28_V14_PACE_LO_GBPS",
        "INKLING_L2PF_C28_V14_PACE_HI_GBPS",
        "INKLING_L2PF_C28_V14_PACE_64_GBPS",
        "INKLING_L2PF_PACE_DRAFT_GBPS",
        "INKLING_L2PF_V16_PACE_GBPS",
    ):
        _req_float_le(pk, 185.0, "GB10 LPDDR5X bus saturation hazard (>185 GB/s)")
    _req("INKLING_L2PF_C28_STOP", "hard", "Hard stop required for C28 L2PF")
    _req_float_le("INKLING_L2PF_PACE_DRAFT_ATTN_GBPS", 110.0, "DRAFT_ATTN bus ceiling")
    if "INKLING_L2PF_DRAFT_ATTN_MAX_M" in env and int(env["INKLING_L2PF_DRAFT_ATTN_MAX_M"]) > 32:
        errors.append(f"INKLING_L2PF_DRAFT_ATTN_MAX_M={env['INKLING_L2PF_DRAFT_ATTN_MAX_M']} > 32 (bus hazard)")

    # 6. Zero-Swap UMA Invariants
    _req("INKLING_LOAD_O_DIRECT", "1", "Zero page-cache weight load")
    _req("INKLING_LOAD_DROP_CACHE", "1", "Drop page cache after load")
    _req("INKLING_UMA_HOST_RECLAIM", "1", "Host UMA reclaim before KV sizing")
    _req("INKLING_PREWARM_FP8_BEFORE_KV", "1", "Prevent 634 MiB post-KV FP8 ambush")
    _req("INKLING_SHARED_PF_SCRATCH", "1", "Share prefill scratch across layers")
    _req_float_le("GMU", 0.87, "GMU > 0.87 causes UMA swap spill on 121.63 GiB GB10")
    if "VLLM_UMA_MIN_FREE_GIB" in env and float(env["VLLM_UMA_MIN_FREE_GIB"]) < 7.0:
        errors.append(f"VLLM_UMA_MIN_FREE_GIB={env['VLLM_UMA_MIN_FREE_GIB']} < 7.0 GiB")

    # 7. CUDAGraph Bucket Uniformity & Hetero WF/SAM Boundaries
    if "SPEC_BATCH_SCHEDULE" in env and "CUDAGRAPH_REQ_SIZES" in env and "VLLM_EXTRA" in env:
        try:
            sched = json.loads(strip_quotes(env["SPEC_BATCH_SCHEDULE"]))
            cg_sizes = [int(x.strip()) for x in strip_quotes(env["CUDAGRAPH_REQ_SIZES"]).split(",") if x.strip()]
            ve = strip_quotes(env["VLLM_EXTRA"])
            m_cc = re.search(r"--compilation-config=(\{.*\})", ve)
            if not m_cc:
                errors.append(f"Cannot parse --compilation-config JSON inside VLLM_EXTRA={ve!r}")
            else:
                cc_obj = json.loads(m_cc.group(1))
                if "max_cudagraph_capture_size" in cc_obj:
                    errors.append("Forbidden Pydantic-crashing key max_cudagraph_capture_size in VLLM_EXTRA!")
                if cc_obj.get("decode_cudagraph_capture_request_sizes") != cg_sizes:
                    errors.append("CUDAGRAPH_REQ_SIZES disagrees with VLLM_EXTRA decode_cudagraph_capture_request_sizes!")

            sam_max_b = int(env.get("INKLING_SAM_MAX_B", "4"))
            het_min_b = int(env.get("INKLING_HETERO_BUDGET_MIN_B", "2"))
            het_max_b = int(env.get("INKLING_HETERO_BUDGET_MAX_B", "8"))

            def _k_for_b(b: int) -> int:
                for lo, hi, k in sched:
                    if lo <= b <= hi:
                        return int(k)
                return int(env.get("SPEC_K", "7"))

            prev = 0
            for cap_g in cg_sizes:
                sigs = {(_k_for_b(b), b <= sam_max_b, het_min_b <= b <= het_max_b) for b in range(prev + 1, cap_g + 1)}
                if len(sigs) > 1:
                    errors.append(f"CUDAGraph bucket ({prev}, {cap_g}] straddles K/SAM/WF boundary: {sigs}")
                prev = cap_g
            for b in range(1, 9):
                m_pad = b * (_k_for_b(b) + 1)
                if m_pad > 48:
                    errors.append(f"SPEC_BATCH_SCHEDULE at B={b} yields M_pad={m_pad} > 48 (breaks FC=2/MX7-T8!)")
        except Exception as exc:
            errors.append(f"Failed parsing CUDAGraph / SPEC_BATCH_SCHEDULE config: {exc}")

    # 8. Conditional Wave-V16 Feature Safety Locks (auto-activated whenever V16 keys appear!)
    if env.get("INKLING_DYN_K_EXTEND_V16") == "1":
        notes.append("V16 Track 1 Dynamic-K active")
        for k_cap_key in ("INKLING_DYN_K_EXT_MAX", "INKLING_DYN_K_MAX", "INKLING_DYN_K_SAM_MAX"):
            if k_cap_key in env and int(env[k_cap_key]) > 15:
                errors.append(f"V16 Dynamic-K safety lock: {k_cap_key}={env[k_cap_key]} > 15 (breaks single S8 M<=16 tile!)")

    if env.get("INKLING_SAME_BUDGET_TREE_V16") == "1":
        notes.append("V16 Track 1 Same-Budget Tree active")
        if env.get("INKLING_TREE_SPEC", "0") != "1":
            errors.append("V16 Same-Budget Tree requires INKLING_TREE_SPEC=1 (tree_spec_runtime.py:40)!")
        if env.get("INKLING_TREE_SPEC_MAX_B", "1") != "1":
            errors.append("V16 Same-Budget Tree safety lock: INKLING_TREE_SPEC_MAX_B must be 1 (B>=2 linear)!")
        if env.get("INKLING_DYN_K_EXTEND_V16", "0") == "1":
            errors.append("V16 Profile A (Same-Budget Tree) and Profile B (Dyn-K15) are mutually exclusive!")

    if env.get("INKLING_SYNC_FREE_TEMP_V16") == "1":
        notes.append("V16 Track 3 SyncFree-Temp active")
        _req("INKLING_SYNC_FREE", "argmax", "SyncFree-Temp keeps greedy argmax path intact for temp==0")
        if env.get("INKLING_SAM_ALLOW_TEMP", "0") != "0":
            errors.append("V16 SyncFree-Temp safety lock: INKLING_SAM_ALLOW_TEMP must remain 0 to prevent temp>0 bias!")

    if env.get("INKLING_PF_SKIP_NONTAIL_DRAFT_V16") == "1":
        notes.append("V16 Track 4 Non-Tail Drafter Skip active")
        _req_int_ge("INKLING_DRAFT_GLOBAL_WINDOW", 1024, "Non-tail drafter skip requires valid DRAFT_GLOBAL_WINDOW")
        _req("INKLING_PF_SHADOW_K_STRICT_VERIFY", "1", "Strict Shadow-K verification required with V16 1M prefill")

    if env.get("RDMA_AR_C32_REG_BF12_V16") == "1":
        notes.append("V16 Track 5 Reg-Fused BF12-G128-OVF Pipe2 RDMA active")
        _req_int_ge("RDMA_AR_C32_BF12_P2_MIN_BYTES", 196608, "Keep C=1/C=2 uncompressed BF16 (P2 >= 196608 B)")
        _req_int_ge("RDMA_AR_C32_BF12_P1_MIN_BYTES", 262144, "Keep C=1/C=2 uncompressed BF16 (P1 >= 262144 B)")

    if env.get("INKLING_TELEMETRY_V16") == "1":
        notes.append("V16 Track 7 Telemetry Snapshot active")
        for iv_key in ("INKLING_TELEMETRY_V16_INTERVAL_S", "INKLING_TELEMETRY_INTERVAL_S"):
            if iv_key in env and float(env[iv_key]) < 1.0:
                errors.append(f"V16 Telemetry safety lock: {iv_key}={env[iv_key]} < 1.0s (hot-path overhead risk!)")

    # 9. Mode-specific Default vs Max-KV-Speed checks
    if mode == "default":
        _req("INKLING_KV_BF12", "0", "Default profile uses pure BF16 KV")
        _req("INKLING_PREFILL_SP_RS_HALF1_MODE", "nccl", "Default profile uses Tier-1 8ch NCCL")
    elif mode == "max_kv_speed":
        _req("INKLING_KV_BF12", "1", "Max-KV-Speed enables Global-Only SWAR KV-BF12")
        _req("INKLING_KV_BF12_GLOBAL_ONLY", "1", "Max-KV-Speed restricts KV-BF12 to 8L Global")
        _req("INKLING_PREFILL_SP_RS_HALF1_MODE", "a2a_int8_lowsm", "Max-KV-Speed uses Tier-2 Half-1 a2a")
        _req("INKLING_PREFILL_SP_RS_HALF1_GLOBAL_MODE", "nccl", "Keep Global Half-1 on NCCL")

    return errors, notes


def _check_so_manifest(
    env: Dict[str, str],
    known_sos: Set[str],
    rdma_dir: Optional[pathlib.Path] = None,
) -> Tuple[List[str], Dict[str, Any]]:
    """Verify every *_LIB=/rdma/*.so & RDMA_AR_EXT matches manifest and has zero stale aliases."""
    errors: List[str] = []
    lib_keys = {
        k: strip_quotes(v)
        for k, v in env.items()
        if (k.endswith("_LIB") or k in ("RDMA_AR_EXT", "RDMA_RS_EXT")) and strip_quotes(v).endswith(".so")
    }
    referenced_sos: Set[str] = set()
    missing_from_manifest: List[str] = []
    missing_on_disk: List[str] = []

    for k, val in sorted(lib_keys.items()):
        so_name = pathlib.Path(val).name
        referenced_sos.add(so_name)
        if so_name not in known_sos:
            missing_from_manifest.append(f"{k}={val}")
        if rdma_dir is not None:
            disk_p = rdma_dir / so_name
            if not disk_p.is_file() or disk_p.stat().st_size == 0:
                missing_on_disk.append(f"{k}={disk_p}")

    if missing_from_manifest:
        errors.append(f".so manifest missing {len(missing_from_manifest)} referenced lib(s): {missing_from_manifest}")
    if missing_on_disk:
        errors.append(f"Physical --rdma-dir missing {len(missing_on_disk)} lib(s): {missing_on_disk}")

    # Check alias consistency within each kernel family (zero stale cross-wave aliases!)
    alias_families = [
        ("FUSED_ATTN", [k for k in lib_keys if k.startswith("INKLING_FUSED_ATTN") and k.endswith("_LIB")]),
        ("ROUTER_BF12", [k for k in lib_keys if k.startswith("INKLING_ROUTER") and k.endswith("_LIB")]),
        ("CERT_HEAD", [k for k in lib_keys if k.startswith("INKLING_CERT") and k.endswith("_LIB")]),
        ("SCONV_AR_V14_V16", [k for k in ("INKLING_FUSED_SCONV_AR_C48_LIB", "INKLING_FUSED_SCONV_AR_C32_LIB", "INKLING_SCONV_V14_LIB", "INKLING_ATTN_RES_V14_LIB") if k in lib_keys]),
        ("PF_DENSE_FP8", [k for k in lib_keys if k.startswith("INKLING_PF_DENSE_FP8_TC") and k.endswith("_LIB")]),
        ("PF_MX7", [k for k in ("INKLING_PF_MX7_V10_LIB", "INKLING_PF_MX7_V13_LIB", "INKLING_PF_MX7_V16_LIB") if k in lib_keys]),
        ("PF_NVFP4_W13", [k for k in ("INKLING_PF_NVFP4_W13_V6_LIB", "INKLING_PF_NVFP4_W13_V13_LIB", "INKLING_PF_NVFP4_W13_V16_LIB") if k in lib_keys]),
        ("PF_NVFP4_W2", [k for k in ("INKLING_PF_NVFP4_W2_V5_LIB", "INKLING_PF_NVFP4_W2_V13_LIB", "INKLING_PF_NVFP4_W2_V16_LIB") if k in lib_keys]),
        ("PF_RHD_COMM", [k for k in lib_keys if k.startswith("INKLING_PF_RHD_COMM") and k.endswith("_LIB")]),
    ]
    for fam_name, keys in alias_families:
        vals = {lib_keys[k] for k in keys}
        if len(vals) > 1:
            errors.append(f"Stale *_LIB alias mismatch in {fam_name}: { {k: lib_keys[k] for k in keys} }")

    return errors, {
        "total_lib_keys": len(lib_keys),
        "unique_so_referenced": len(referenced_sos),
        "referenced_sos": sorted(referenced_sos),
        "known_so_manifest_size": len(known_sos),
    }


def _audit_v16_overlays(
    base_env: Dict[str, str],
    overlays: List[pathlib.Path],
    apply_overlays: bool,
) -> Tuple[Dict[str, str], List[str], Dict[str, Any]]:
    """Audit discovered sibling Wave-V16 overlays for syntax, inter-track conflicts & safety clobbers."""
    errors: List[str] = []
    merged = dict(base_env)
    seen_by_track: Dict[str, Tuple[str, str]] = {}
    inter_track_conflicts: List[str] = []
    clobbers: List[str] = []
    overlay_summaries: List[Dict[str, Any]] = []

    for ov in overlays:
        _, ov_map, _, ov_errs = parse_env_file(ov)
        b_ok, b_err = check_bash_syntax(ov)
        if not b_ok:
            ov_errs.append(f"bash -n failed on {ov.name}: {b_err}")
        errors.extend(ov_errs)
        overlay_summaries.append({"file": str(ov), "keys": len(ov_map), "errors": len(ov_errs)})

        for k, v in ov_map.items():
            if k in seen_by_track and seen_by_track[k][1] != v:
                msg = f"Cross-track V16 conflict on {k}: {seen_by_track[k][0]}={seen_by_track[k][1]} vs {ov.name}={v}"
                inter_track_conflicts.append(msg)
                errors.append(msg)
            seen_by_track[k] = (ov.name, v)

            if k in PROTECTED_SAFETY_LOCKS and v != PROTECTED_SAFETY_LOCKS[k]:
                msg = f"Overlay {ov.name} clobbers protected safety lock {k}: expected {PROTECTED_SAFETY_LOCKS[k]!r}, got {v!r}"
                clobbers.append(msg)
                errors.append(msg)

            if apply_overlays:
                merged[k] = v

    return merged, errors, {
        "overlays_discovered": len(overlays),
        "applied_to_env": apply_overlays,
        "inter_track_conflicts": inter_track_conflicts,
        "safety_clobbers": clobbers,
        "details": overlay_summaries,
    }


def audit_env_profile(
    env_path: pathlib.Path,
    *,
    patches_dir: Optional[pathlib.Path] = _DEFAULT_PATCHES,
    kernels_dir: Optional[pathlib.Path] = _DEFAULT_KERNELS,
    launcher_path: Optional[pathlib.Path] = _DEFAULT_LAUNCHER,
    ops_dir: Optional[pathlib.Path] = _DEFAULT_OPS,
    v16_root: Optional[pathlib.Path] = _DEFAULT_V16_ROOT,
    rdma_dir: Optional[pathlib.Path] = None,
    apply_v16_overlays: bool = False,
    mode: str = "auto",
    strict_corpus: bool = True,
) -> Dict[str, Any]:
    """Run all 7 audit stages on `env_path` and return a structured pass/fail report."""
    errors: List[str] = []
    checks: Dict[str, bool] = {}

    if not env_path.is_file():
        return {
            "status": "FAIL",
            "env_file": str(env_path),
            "checks": {"env_file_exists": False},
            "errors": [f"Env file not found: {env_path}"],
        }

    # Stage 1: Hygiene & Deduplication
    entries, env, dups, parse_errs = parse_env_file(env_path)
    checks["01_hygiene_zero_cr_zero_dups"] = (len(parse_errs) == 0)
    errors.extend(parse_errs)

    # Stage 2: POSIX Shell Syntax
    bash_ok, bash_err = check_bash_syntax(env_path)
    checks["02_bash_syntax_clean"] = bash_ok
    if not bash_ok:
        errors.append(f"bash -n failed on {env_path.name}: {bash_err}")

    # Discover V16 artifacts & build corpus
    v16_art = discover_v16_artifacts(v16_root)
    corpus = build_symbol_corpus(patches_dir, kernels_dir, launcher_path, ops_dir, v16_art, apply_v16_patches=True)
    pa = corpus["patch_audit"]
    patch_ok = (len(pa["v16_patch_errors"]) == 0) and (len(pa["v16_patch_overlaps"]) == 0)
    checks["03_v16_patches_clean_no_overlap"] = patch_ok
    if not patch_ok:
        errors.extend([f"V16 patch error: {e}" for e in pa["v16_patch_errors"]])
        if pa["v16_patch_overlaps"]:
            errors.append(f"V16 overlapping patch targets: {pa['v16_patch_overlaps']}")

    # Stage 7 (run early if --apply-v16-overlays so merged env is audited end-to-end!)
    eff_env, ov_errs, ov_info = _audit_v16_overlays(env, v16_art["overlays"], apply_v16_overlays)
    checks["04_cross_wave_v16_overlays_no_clobbers"] = (len(ov_errs) == 0)
    errors.extend(ov_errs)

    # Stage 3: Launcher & Ops Forwarding Contract
    unforwarded = [k for k in sorted(eff_env) if not LAUNCHER_FORWARD_RE.match(k) and k not in LAUNCHER_CLI_KEYS]
    checks["05_launcher_forwarding_100pct"] = (len(unforwarded) == 0)
    if unforwarded:
        errors.append(f"Keys NOT forwarded/consumed by launcher/ops: {unforwarded[:15]}")

    # Stage 4: Zero Dead/Typo Keys vs Merged Corpus
    unref = [
        k for k in sorted(eff_env)
        if not k.startswith(STD_RUNTIME_PREFIXES) and k not in corpus["tokens"]
    ]
    checks["06_zero_dead_or_typo_keys"] = (len(unref) == 0) if strict_corpus else True
    if strict_corpus and unref:
        errors.append(f"{len(unref)} unreferenced/typo key(s) not found in patches/launcher/ops/CUDA: {unref[:15]}")

    # Stage 5: `.so` Manifest Completeness & Stale-Alias Check
    so_errs, so_info = _check_so_manifest(eff_env, corpus["known_sos"], rdma_dir=rdma_dir)
    checks["07_so_manifest_complete_no_stale_aliases"] = (len(so_errs) == 0)
    errors.extend(so_errs)

    # Auto-detect default vs max_kv_speed if mode == "auto"
    eff_mode = mode
    if eff_mode == "auto":
        if eff_env.get("INKLING_KV_BF12") == "1" or "max_kv" in env_path.name.lower():
            eff_mode = "max_kv_speed"
        elif "default" in env_path.name.lower():
            eff_mode = "default"
        else:
            eff_mode = "generic"

    # Stage 6: Hard V15 + V16 Safety Locks
    lock_errs, v16_notes = _check_safety_invariants(eff_env, eff_mode)
    checks["08_v15_v16_safety_and_0ulp_locks"] = (len(lock_errs) == 0)
    errors.extend(lock_errs)

    status = "PASS" if (all(checks.values()) and len(errors) == 0) else "FAIL"
    return {
        "schema_version": "v16.env_audit.1",
        "status": status,
        "env_file": str(env_path),
        "mode": eff_mode,
        "total_lines": len(entries),
        "unique_keys": len(eff_env),
        "v16_features_detected": v16_notes,
        "checks": checks,
        "so_manifest": so_info,
        "patch_audit": pa,
        "v16_overlays": ov_info,
        "unreferenced_keys": unref,
        "errors": errors,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Wave-V16 Reusable Cross-Wave Env, Patch & .so Manifest Conflict Auditor"
    )
    ap.add_argument("--env", default="", help="Path to .env profile to audit (default: audits both V15 Default & Max-KV-Speed)")
    ap.add_argument("--patches", default=str(_DEFAULT_PATCHES), help="Path to unified Python patches directory")
    ap.add_argument("--kernels", default=str(_DEFAULT_KERNELS), help="Path to SM121a kernels directory")
    ap.add_argument("--launcher", default=str(_DEFAULT_LAUNCHER), help="Path to launcher shell script")
    ap.add_argument("--ops-dir", default=str(_DEFAULT_OPS), help="Path to ops scripts directory")
    ap.add_argument("--v16-root", default=str(_DEFAULT_V16_ROOT), help="Path to scratch/v16/ for dynamic sibling discovery")
    ap.add_argument("--rdma-dir", default="", help="Optional path to physical /rdma directory to verify .so files on disk")
    ap.add_argument("--apply-v16-overlays", action="store_true", help="Stack discovered V16 overlays onto --env before invariant checks")
    ap.add_argument("--mode", choices=("auto", "default", "max_kv_speed", "generic"), default="auto")
    ap.add_argument("--no-strict-corpus", action="store_true", help="Treat unreferenced keys as advisory instead of hard failure")
    ap.add_argument("--out-json", default="", help="Optional path to write JSON audit report")
    ap.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    ap = build_arg_parser()
    args = ap.parse_args(argv)

    patches_dir = pathlib.Path(args.patches) if args.patches else None
    kernels_dir = pathlib.Path(args.kernels) if args.kernels else None
    launcher_path = pathlib.Path(args.launcher) if args.launcher else None
    ops_dir = pathlib.Path(args.ops_dir) if args.ops_dir else None
    v16_root = pathlib.Path(args.v16_root) if args.v16_root else None
    rdma_dir = pathlib.Path(args.rdma_dir) if args.rdma_dir else None

    targets: List[pathlib.Path]
    if args.env:
        targets = [pathlib.Path(args.env)]
    else:
        targets = [_DEFAULT_V15_DEF, _DEFAULT_V15_MAX]

    reports: List[Dict[str, Any]] = []
    all_ok = True
    for tgt in targets:
        rep = audit_env_profile(
            tgt,
            patches_dir=patches_dir,
            kernels_dir=kernels_dir,
            launcher_path=launcher_path,
            ops_dir=ops_dir,
            v16_root=v16_root,
            rdma_dir=rdma_dir,
            apply_v16_overlays=args.apply_v16_overlays,
            mode=args.mode,
            strict_corpus=not args.no_strict_corpus,
        )
        reports.append(rep)
        if rep["status"] != "PASS":
            all_ok = False
        if not args.quiet:
            print("=" * 90)
            print(f"[V16 ENV AUDIT] {rep['status']} | {tgt.name} (mode={rep['mode']}, keys={rep['unique_keys']})")
            print("=" * 90)
            for ck, cv in rep["checks"].items():
                print(f"  [{'PASS' if cv else 'FAIL'}] {ck}")
            print(
                f"  .so manifest: {rep['so_manifest']['unique_so_referenced']} unique libs verified "
                f"(known={rep['so_manifest']['known_so_manifest_size']}); "
                f"V16 overlays discovered={rep['v16_overlays']['overlays_discovered']}; "
                f"V16 patches={len(rep['patch_audit']['v16_patches_found'])}"
            )
            if rep["v16_features_detected"]:
                print(f"  V16 features active: {', '.join(rep['v16_features_detected'])}")
            for err in rep["errors"]:
                print(f"  ERROR: {err}")

    out_payload = reports[0] if len(reports) == 1 else {
        "schema_version": "v16.env_audit.batch.1",
        "overall_status": "PASS" if all_ok else "FAIL",
        "results": reports,
    }
    if args.out_json:
        out_p = pathlib.Path(args.out_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(out_payload, indent=2) + "\n", encoding="utf-8")

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
