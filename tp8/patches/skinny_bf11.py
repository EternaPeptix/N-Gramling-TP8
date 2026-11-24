# SPDX-License-Identifier: Apache-2.0
"""SM12x (GB10) 0-ULP Lossless BF11_FLIP_EXACT Skinny GEMM (CUDA C++ mma.sync.m16n8k16).

Compresses BF16 dense projections from 2.00 B/w -> ~1.408-1.423 B/w (-29.4% vs BF16, -6.1% vs BF12):
  - 98.6% of weights fall in [base_exp - 7 .. base_exp]   -> 3-bit e3 (b3 = 0)
  -  1.4% of weights fall in [base_exp - 15 .. base_exp-8] -> 3-bit e3 + 1-bit sparse flip (b3 = 1)
  -  0.01% true outliers outside [base_exp - 15 .. base_exp] -> exact 64-bit inline patch list
  - SLABS: 1408 B/slab (1024 B sm + 384 B e3, 11x128B aligned), FLIPS: ~28 B/slab (__ldg cached)
  - Single-kernel last-CTA semaphore finalize (eliminates separate reduce kernel launch & gap!)
  - Dead-warp skip, optional NLIVE dynamic row skip, PDL (`::: "memory"`), TRIG on DIRECT & split-K,
    and Fused Dual-KV (k_proj + v_proj in ONE launch for global layers!).
  - LITERAL 0-ULP BIT-EXACT (ndiff == 0) with skinny.py across all M=1..16!

Enabled via INKLING_SKINNY_BF11=1. Pre-packed once before CUDA graph capture via prepack().
"""

import ctypes
import os
import torch

ENABLED = os.environ.get("INKLING_SKINNY_BF11", "0").strip().lower() not in ("0", "", "off", "false", "no")
FREE_RAW = os.environ.get("INKLING_BF11_FREE_RAW", "0").strip().lower() not in ("0", "", "off", "false", "no")
USE_NLIVE = os.environ.get("INKLING_BF11_NLIVE", "1").strip().lower() not in ("0", "", "off", "false", "no")
USE_DUAL_KV = os.environ.get("INKLING_BF11_DUAL_KV", "1").strip().lower() not in ("0", "", "off", "false", "no")
C32_ENABLED = os.environ.get("INKLING_BF11_C32", "0").strip().lower() not in ("0", "", "off", "false", "no")
C32_MAX_M = max(16, int(os.environ.get("INKLING_BF11_C32_MAX_M", "256") or 256))
C32_BM = int(os.environ.get("INKLING_BF11_C32_BM", "0") or 0)
C32_KORDER = int(os.environ.get("INKLING_BF11_C32_KORDER", "0") or 0)
C32_QUALIFY = os.environ.get("INKLING_BF11_C32_QUALIFY", "1").strip().lower() not in ("0", "", "off", "false", "no")
C32_EXACT_REF = os.environ.get("INKLING_BF11_C32_EXACT_REF", "torch").strip().lower()
C32_CAPTURE_ONLY = os.environ.get(
    "INKLING_BF11_C32_CAPTURE_ONLY",
    "1" if C32_EXACT_REF != "torch" else "0",
).strip().lower() not in ("0", "", "off", "false", "no")
C32_MIN_N = int(os.environ.get("INKLING_BF11_C32_MIN_N", "0") or 0)
C32_DUAL_KV = os.environ.get("INKLING_BF11_C32_DUAL_KV", "1").strip().lower() not in ("0", "", "off", "false", "no")
V5_ENABLED = os.environ.get("INKLING_BF11_V5", "0").strip().lower() not in ("0", "", "off", "false", "no")
V5_FLIP_THRESH = max(0, min(32, int(os.environ.get("INKLING_BF11_V5_FLIP_THRESH", "16") or 16)))
V5_SPLIT_CTA = int(os.environ.get("INKLING_BF11_V5_SPLIT_CTA", "0") or 0)
V5_W_RUN = int(os.environ.get("INKLING_BF11_V5_W_RUN", "0") or 0)
# Wave-V10 opt-in: w_run=2 on C=1..2 (M<=16) split==6 qkvr/q_side lifts wave eff 93.6% -> 100.0% (0-ULP!)
C1_SPLIT6_WRUN2 = os.environ.get("INKLING_BF11_C1_SPLIT6_WRUN2", "0").strip().lower() in ("1", "true", "on", "yes")
V5_C1 = os.environ.get("INKLING_BF11_V5_C1", "1").strip().lower() not in ("0", "", "off", "false", "no")
V5_C32 = os.environ.get("INKLING_BF11_V5_C32", "1").strip().lower() not in ("0", "", "off", "false", "no")
# Wave-V12 C=1..8 (M=1..48+) opt-in flags (ALL strictly default-OFF!)
V12_ENABLED = os.environ.get("INKLING_BF11_V12", "0").strip().lower() in ("1", "true", "on", "yes")
V12_BM48 = os.environ.get("INKLING_BF11_V12_BM48", "1" if V12_ENABLED else "0").strip().lower() in ("1", "true", "on", "yes")
V12_FAST_UNPACK = os.environ.get("INKLING_BF11_V12_FAST_UNPACK", "1" if V12_ENABLED else "0").strip().lower() in ("1", "true", "on", "yes")
FREE_RAW_SAFE = os.environ.get("INKLING_BF11_FREE_RAW_SAFE", "1" if V12_ENABLED else "0").strip().lower() in ("1", "true", "on", "yes")
FUSE_DYN_STRICT = os.environ.get("INKLING_BF11_FUSE_DYN_STRICT", "1" if V12_ENABLED else "0").strip().lower() in ("1", "true", "on", "yes")
_LIB_PATH = os.environ.get(
    "INKLING_BF11_LIB",
    "/rdma/libbf11_flip_exact_v12.so"
    if V12_ENABLED
    else (
        "/rdma/libbf11_flip_exact_v5.so"
        if V5_ENABLED
        else ("/rdma/libbf11_flip_exact_c32.so" if C32_ENABLED else "/rdma/libbf11_flip_exact.so")
    ),
)
_LIB = None
_LIB_TRIED = False
FUSE_DYN = os.environ.get("INKLING_BF11_FUSE_DYN", "0").strip().lower() in ("1", "true", "on", "yes")
FUSE_DYN_REF = os.environ.get("INKLING_BF11_FUSE_DYN_REF", "torch").strip().lower()
_HAS_C32 = False
_HAS_V5 = False
_HAS_V9_DYN = False
_HAS_V12 = False
_HAS_V12_UNPACK = False
# (N_dyn, K, M) -> (dyn_mode, dyn_split_k, torch_exact) or None (fail-closed)
_DYN_QUAL: dict[tuple[int, int, int], tuple[int, int, bool] | None] = {}
_DYN_QUAL_PREWARMED: set[tuple[int, int]] = set()
_C32_QUAL_OK: dict[tuple[int, int, int], bool] = {}  # (N, K, M) -> 0-ULP verified
_C32_TORCH_EXACT: dict[tuple[int, int, int], bool] = {}  # (N, K, M) -> literal torch.mm bit-identical
_C32_DUAL_QUAL_OK: dict[tuple[int, int, int], bool] = {}  # (N_each, K, M) -> 0-ULP verified
_C32_CANONICAL_MS = (24, 32, 36, 48, 60, 64, 72, 96, 112, 128, 160, 192, 224, 256)
_C32_CANONICAL_MS_EXACT = (
    (*range(17, 49), 52, 54, 56, 57, 60, 64, 66, 72, 75, 80, 84, 88, 96, 100, 104, 112, 120, 128, 160, 192, 224, 256)
    if V12_ENABLED
    else (18, 20, 21, 22, 24, 25, 26, 28, 30, 32, 33, 35, 36, 40, 42, 44, 45, 48, 49, 52, 54, 56, 57, 60, 64, 66, 72, 75, 80, 84, 88, 96, 100, 104, 112, 120, 128, 160, 192, 224, 256)
)
_C32_EXTRA_MS = tuple(
    int(tok.strip())
    for tok in os.environ.get("INKLING_BF11_C32_EXTRA_MS", "").split(",")
    if tok.strip().isdigit()
)


def _get_lib():
    global _LIB, _LIB_TRIED, _HAS_C32, _HAS_V5, _HAS_V9_DYN, _HAS_V12, _HAS_V12_UNPACK
    if not _LIB_TRIED:
        _LIB_TRIED = True
        if os.path.exists(_LIB_PATH):
            try:
                lib = ctypes.CDLL(_LIB_PATH)
                lib.bf11_flip_fwd.argtypes = [
                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                    ctypes.c_int, ctypes.c_int, ctypes.c_int,
                    ctypes.c_int, ctypes.c_int, ctypes.c_uint32,
                    ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
                ]
                lib.bf11_flip_fwd.restype = ctypes.c_int
                lib.bf11_flip_dual_kv_fwd.argtypes = [
                    ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_int, ctypes.c_int, ctypes.c_int,
                    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                    ctypes.c_int, ctypes.c_int, ctypes.c_uint32, ctypes.c_uint32,
                    ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
                ]
                lib.bf11_flip_dual_kv_fwd.restype = ctypes.c_int
                if hasattr(lib, "bf11_flip_c32_fwd") and hasattr(lib, "bf11_flip_dual_kv_c32_fwd"):
                    lib.bf11_flip_c32_fwd.argtypes = [
                        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                        ctypes.c_void_p, ctypes.c_void_p,
                        ctypes.c_void_p, ctypes.c_void_p,
                        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint32,
                        ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
                    ]
                    lib.bf11_flip_c32_fwd.restype = ctypes.c_int
                    lib.bf11_flip_dual_kv_c32_fwd.argtypes = [
                        ctypes.c_void_p, ctypes.c_void_p,
                        ctypes.c_void_p, ctypes.c_void_p,
                        ctypes.c_void_p, ctypes.c_void_p,
                        ctypes.c_void_p, ctypes.c_void_p,
                        ctypes.c_void_p, ctypes.c_void_p,
                        ctypes.c_void_p, ctypes.c_void_p,
                        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                        ctypes.c_int, ctypes.c_int, ctypes.c_int,
                        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                        ctypes.c_int, ctypes.c_int,
                        ctypes.c_int, ctypes.c_int, ctypes.c_uint32, ctypes.c_uint32,
                        ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
                    ]
                    lib.bf11_flip_dual_kv_c32_fwd.restype = ctypes.c_int
                    _HAS_C32 = True
                if (
                    hasattr(lib, "bf11_flip_v5_version")
                    and hasattr(lib, "bf11_v5_measure_sidecar_from_bf11")
                    and hasattr(lib, "bf11_v5_build_sidecar_from_bf11")
                    and hasattr(lib, "bf11_flip_v5_c1_fwd")
                    and hasattr(lib, "bf11_flip_dual_kv_v5_c1_fwd")
                    and hasattr(lib, "bf11_flip_v5_c32_fwd")
                    and hasattr(lib, "bf11_flip_dual_kv_v5_c32_fwd")
                ):
                    lib.bf11_flip_v5_version.argtypes = []
                    lib.bf11_flip_v5_version.restype = ctypes.c_int
                    lib.bf11_v5_measure_sidecar_from_bf11.argtypes = [
                        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
                    ]
                    lib.bf11_v5_measure_sidecar_from_bf11.restype = ctypes.c_int
                    lib.bf11_v5_build_sidecar_from_bf11.argtypes = [
                        *[ctypes.c_void_p] * 9,
                        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
                    ]
                    lib.bf11_v5_build_sidecar_from_bf11.restype = ctypes.c_int
                    lib.bf11_flip_v5_c1_fwd.argtypes = [
                        *[ctypes.c_void_p] * 11,
                        *[ctypes.c_int] * 10,
                        ctypes.c_uint32,
                        ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
                    ]
                    lib.bf11_flip_v5_c1_fwd.restype = ctypes.c_int
                    lib.bf11_flip_dual_kv_v5_c1_fwd.argtypes = [
                        *[ctypes.c_void_p] * 19,
                        *[ctypes.c_int] * 11,
                        ctypes.c_uint32, ctypes.c_uint32,
                        ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
                    ]
                    lib.bf11_flip_dual_kv_v5_c1_fwd.restype = ctypes.c_int
                    lib.bf11_flip_v5_c32_fwd.argtypes = [
                        *[ctypes.c_void_p] * 11,
                        *[ctypes.c_int] * 13,
                        ctypes.c_uint32,
                        ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
                    ]
                    lib.bf11_flip_v5_c32_fwd.restype = ctypes.c_int
                    lib.bf11_flip_dual_kv_v5_c32_fwd.argtypes = [
                        *[ctypes.c_void_p] * 19,
                        *[ctypes.c_int] * 14,
                        ctypes.c_uint32, ctypes.c_uint32,
                        ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
                    ]
                    lib.bf11_flip_dual_kv_v5_c32_fwd.restype = ctypes.c_int
                    _HAS_V5 = True
                if (
                    _HAS_V5
                    and hasattr(lib, "bf11_flip_v9_version")
                    and hasattr(lib, "bf11_dyn_probe_v9_fwd")
                    and hasattr(lib, "bf11_flip_dual_kv_dyn_v9_c1_fwd")
                    and hasattr(lib, "bf11_flip_dual_kv_dyn_v9_c32_fwd")
                ):
                    lib.bf11_flip_v9_version.argtypes = []
                    lib.bf11_flip_v9_version.restype = ctypes.c_int
                    lib.bf11_dyn_probe_v9_fwd.argtypes = [
                        *[ctypes.c_void_p] * 4,
                        *[ctypes.c_int] * 7,
                        ctypes.c_uint64,
                    ]
                    lib.bf11_dyn_probe_v9_fwd.restype = ctypes.c_int
                    lib.bf11_flip_dual_kv_dyn_v9_c1_fwd.argtypes = [
                        *[ctypes.c_void_p] * 23,
                        *[ctypes.c_int] * 14,
                        ctypes.c_uint32, ctypes.c_uint32,
                        *[ctypes.c_int] * 4,
                        ctypes.c_uint64,
                    ]
                    lib.bf11_flip_dual_kv_dyn_v9_c1_fwd.restype = ctypes.c_int
                    lib.bf11_flip_dual_kv_dyn_v9_c32_fwd.argtypes = [
                        *[ctypes.c_void_p] * 23,
                        *[ctypes.c_int] * 17,
                        ctypes.c_uint32, ctypes.c_uint32,
                        *[ctypes.c_int] * 4,
                        ctypes.c_uint64,
                    ]
                    lib.bf11_flip_dual_kv_dyn_v9_c32_fwd.restype = ctypes.c_int
                    _HAS_V9_DYN = True
                if (
                    _HAS_V5
                    and hasattr(lib, "bf11_flip_v12_version")
                    and hasattr(lib, "bf11_flip_v12_c1_fwd")
                    and hasattr(lib, "bf11_flip_dual_kv_v12_c1_fwd")
                    and hasattr(lib, "bf11_flip_v12_c32_fwd")
                    and hasattr(lib, "bf11_flip_dual_kv_v12_c32_fwd")
                    and hasattr(lib, "bf11_flip_dual_kv_dyn_v12_c1_fwd")
                    and hasattr(lib, "bf11_flip_dual_kv_dyn_v12_c32_fwd")
                ):
                    lib.bf11_flip_v12_version.argtypes = []
                    lib.bf11_flip_v12_version.restype = ctypes.c_int
                    lib.bf11_flip_v12_c1_fwd.argtypes = list(lib.bf11_flip_v5_c1_fwd.argtypes)
                    lib.bf11_flip_v12_c1_fwd.restype = ctypes.c_int
                    lib.bf11_flip_dual_kv_v12_c1_fwd.argtypes = list(lib.bf11_flip_dual_kv_v5_c1_fwd.argtypes)
                    lib.bf11_flip_dual_kv_v12_c1_fwd.restype = ctypes.c_int
                    lib.bf11_flip_v12_c32_fwd.argtypes = list(lib.bf11_flip_v5_c32_fwd.argtypes)
                    lib.bf11_flip_v12_c32_fwd.restype = ctypes.c_int
                    lib.bf11_flip_dual_kv_v12_c32_fwd.argtypes = list(lib.bf11_flip_dual_kv_v5_c32_fwd.argtypes)
                    lib.bf11_flip_dual_kv_v12_c32_fwd.restype = ctypes.c_int
                    lib.bf11_flip_dual_kv_dyn_v12_c1_fwd.argtypes = list(lib.bf11_flip_dual_kv_dyn_v9_c1_fwd.argtypes)
                    lib.bf11_flip_dual_kv_dyn_v12_c1_fwd.restype = ctypes.c_int
                    lib.bf11_flip_dual_kv_dyn_v12_c32_fwd.argtypes = list(lib.bf11_flip_dual_kv_dyn_v9_c32_fwd.argtypes)
                    lib.bf11_flip_dual_kv_dyn_v12_c32_fwd.restype = ctypes.c_int
                    _HAS_V12 = True
                if hasattr(lib, "bf11_unpack_bf16_v12_fwd"):
                    lib.bf11_unpack_bf16_v12_fwd.argtypes = [
                        *[ctypes.c_void_p] * 6,
                        *[ctypes.c_int] * 6,
                        ctypes.c_uint32,
                        ctypes.c_uint64,
                    ]
                    lib.bf11_unpack_bf16_v12_fwd.restype = ctypes.c_int
                    _HAS_V12_UNPACK = True
                _LIB = lib
            except Exception as e:
                print(f"[inkling-bf11] failed to load {_LIB_PATH}: {e!r}", flush=True)
                _LIB = None
    return _LIB


def active_max_m() -> int:
    if not ENABLED:
        return 0
    if C32_ENABLED:
        _get_lib()
        if _HAS_C32:
            return C32_MAX_M
    return 16


# cfg_id -> (warps, sub_n, bn)
_CFGS = {
    0: (4, 2, 64),
    1: (4, 1, 32),
    2: (2, 2, 32),
    4: (2, 1, 16),
    5: (1, 1, 8),
}

# Tuned (cfg_id, split) per (N, K) matching skinny.py split-K exactly (0-ULP!).
# cfg_id=1 (BN=32 W=4 SUB=1) achieves 237-245 GB/s DRAM (331-345 eff GB/s) on BOTH M=8 and M=16.
_QKVR_CFG = int(os.environ.get("INKLING_BF11_QKVR_CFG", "1"))      # 4 (BN=16 W=2: 714 CTAs = 14.875/SM) or 1 (360 CTAs)
_Q_SIDE_CFG = int(os.environ.get("INKLING_BF11_QSIDE_CFG", "1"))   # 1 (45.1/46.4us M=8/16) or 4 (43.3us M=8)
_KV_CFG = int(os.environ.get("INKLING_BF11_KV_CFG", "4"))          # 4 (BN=16 W=2: 144 CTAs = 3.0 CTAs/SM, zero wave tail!)
_WO_CFG = int(os.environ.get("INKLING_BF11_WO_CFG", "1"))          # 4 (BN=16 W=2: 288 CTAs = 6.0 CTAs/SM exact 1-wave, 0-ULP split=1!)
_SHAPES = {
    (1896, 4608): (_QKVR_CFG, 6),   # local qkvr:  cfg=4 eliminates 7.5 CTAs/SM wave tail (0-ULP identical split=6)
    (1640, 4608): (_Q_SIDE_CFG, 6), # global q_side: 45.13/46.41 us (334.9 eff GB/s; cfg=4: 43.31 us)
    (4608, 768):  (_WO_CFG, 1),     # wo_ud:       cfg=4 -> 288 CTAs = 6.0 CTAs/SM exact 1-wave (0-ULP split=1)
    (128, 4608):  (_KV_CFG, 9),     # global k/v:  cfg=4 eliminates 1.5-wave dual-KV tail (0-ULP, identical split=9)
}

# (data_ptr, N, K) -> (slabs, flips, warp_meta, outliers, o_off, base, ks, n_active, cfg_id, split, n_tiles, bn)
_CACHE: dict[tuple[int, int, int], tuple] = {}
# (data_ptr, N, K) -> (slab_meta_v5, sflips, bmasks, flip_thresh)
_V5_CACHE: dict[tuple[int, int, int], tuple] = {}
# 2x ping-pong persistent L2-hot split-K partial & self-resetting semaphore buffers per device.
# Preallocated in prepack() before CUDA graph capture; deterministic flip on every split-K call
# guarantees adjacent split-K calls never collide (including under PDL overlap & graph replay).
_PART_BUFS: dict[int, list[torch.Tensor]] = {}
_SEM_BUFS: dict[int, list[torch.Tensor]] = {}
_SCRATCH_IDX: dict[int, int] = {}
_MIN_PART_ELEMS = 262144  # 1.0 MiB/slot (>= 60*6*16*32 = 184,320 elems for max M=16 shape)
_MIN_SEM_TILES = 1024


def _ensure_scratch(device: torch.device, min_part_elems: int, min_sem_tiles: int) -> None:
    dev_idx = device.index if device.index is not None else 0
    need_p = max(_MIN_PART_ELEMS, int(min_part_elems))
    need_s = max(_MIN_SEM_TILES, int(min_sem_tiles))
    parts = _PART_BUFS.get(dev_idx)
    if parts is None or parts[0].numel() < need_p:
        _PART_BUFS[dev_idx] = [
            torch.empty(need_p, dtype=torch.float32, device=device),
            torch.empty(need_p, dtype=torch.float32, device=device),
        ]
    sems = _SEM_BUFS.get(dev_idx)
    if sems is None or sems[0].numel() < need_s:
        _SEM_BUFS[dev_idx] = [
            torch.zeros(need_s, dtype=torch.int32, device=device),
            torch.zeros(need_s, dtype=torch.int32, device=device),
        ]
    if dev_idx not in _SCRATCH_IDX:
        _SCRATCH_IDX[dev_idx] = 0


def _next_scratch(device: torch.device, min_part_elems: int, min_sem_tiles: int) -> tuple[torch.Tensor, torch.Tensor]:
    dev_idx = device.index if device.index is not None else 0
    parts = _PART_BUFS.get(dev_idx)
    sems = _SEM_BUFS.get(dev_idx)
    if parts is None or parts[0].numel() < min_part_elems or sems is None or sems[0].numel() < min_sem_tiles:
        if torch.cuda.is_current_stream_capturing():
            # Fallback should never trigger when prepack() ran prior to graph capture.
            part = torch.empty(min_part_elems, dtype=torch.float32, device=device)
            if sems is not None and sems[0].numel() >= min_sem_tiles:
                idx = _SCRATCH_IDX.get(dev_idx, 0)
                _SCRATCH_IDX[dev_idx] = idx ^ 1
                return part, sems[idx]
            return part, torch.zeros(max(_MIN_SEM_TILES, min_sem_tiles), dtype=torch.int32, device=device)
        _ensure_scratch(device, min_part_elems, min_sem_tiles)
        parts = _PART_BUFS[dev_idx]
        sems = _SEM_BUFS[dev_idx]
    idx = _SCRATCH_IDX.get(dev_idx, 0)
    _SCRATCH_IDX[dev_idx] = idx ^ 1
    return parts[idx], sems[idx]


def _get_sem(device: torch.device, min_tiles: int) -> torch.Tensor:
    _ensure_scratch(device, _MIN_PART_ELEMS, min_tiles)
    dev_idx = device.index if device.index is not None else 0
    return _SEM_BUFS[dev_idx][0]


def _rowar_nlive_ptr(M: int) -> int:
    if not USE_NLIVE:
        return 0
    try:
        from vllm.distributed.device_communicators import rdma_all_reduce as rar
        if rar.ROWAR.get("scope") == M:
            buf = rar.ROWAR.get("buf")
            if buf is not None:
                return int(buf.data_ptr())
    except Exception:
        pass
    return 0


@torch.no_grad()
def prepack(w: torch.Tensor) -> torch.Tensor | None:
    """Pack a bf16 [N, K] dense weight into 0-ULP BF11_FLIP_EXACT format (outside graph capture).
    Returns the packed slabs tensor (for L2 prefetch), or None if not eligible."""
    if not ENABLED or w is None or w.dtype != torch.bfloat16 or w.dim() != 2 or w.stride(1) != 1:
        return None
    if _get_lib() is None:
        return None
    N, K = int(w.shape[0]), int(w.shape[1])
    cfg = _SHAPES.get((N, K))
    if cfg is None:
        return None
    hit = get_packed(w)
    if hit is not None:
        if V5_ENABLED and get_v5_sidecar(w) is None and not torch.cuda.is_current_stream_capturing():
            build_v5_sidecar(w)
        return hit[0]
    if w.untyped_storage().size() == 0:
        return None
    key = (int(w.data_ptr()), N, K)
    if torch.cuda.is_current_stream_capturing():
        return None

    cfg_id, split = cfg
    warps, sub_n, bn = _CFGS[cfg_id]
    assert K % (split * 128) == 0
    ks = K // split
    n_chunks = ks // 128
    n_tiles = (N + bn - 1) // bn
    n_pad = n_tiles * bn
    spw = n_chunks * sub_n
    assert spw <= 12, f"BF11_FLIP WARP_META supports spw <= 12, got {spw}"

    nz_rows = (w != 0).any(dim=1).nonzero(as_tuple=False)
    n_active = int(nz_rows[-1].item()) + 1 if nz_rows.numel() > 0 else 0
    wp = torch.zeros((n_pad, K), dtype=torch.bfloat16, device=w.device)
    if n_active > 0:
        wp[:n_active].copy_(w[:n_active])
    u = wp.view(torch.uint16).to(torch.int32)
    exp = (u >> 7) & 0xFF

    if n_active > 0:
        cnt = torch.bincount(exp[:n_active].flatten(), minlength=256)
        tot_n = int(cnt.sum().item())
        best_b, best_cost = 122, 1 << 60
        for b in range(15, 255):
            c8 = int(cnt[b - 7 : b + 1].sum().item())
            c16 = int(cnt[b - 15 : b + 1].sum().item())
            cost = 2 * (c16 - c8) + 16 * (tot_n - c16)
            if cost < best_cost:
                best_cost, best_b = cost, b
        base = best_b
    else:
        base = 122

    delta = base - exp
    in_win16 = (delta >= 0) & (delta <= 15)
    if n_active < n_pad:
        in_win16[n_active:] = True
    is_flip = in_win16 & (delta >= 8)
    if n_active < n_pad:
        is_flip[n_active:] = False
    is_out = ~in_win16
    d3 = torch.where(in_win16, delta & 7, torch.zeros_like(delta)).to(torch.int64)
    sm = (((u >> 8) & 0x80) | (u & 0x7F)).to(torch.uint8)

    def to_mma(x: torch.Tensor) -> torch.Tensor:
        v = x.view(n_tiles, warps, sub_n, 8, split, n_chunks, 8, 2, 4, 2)
        v = v.permute(4, 0, 1, 5, 2, 6, 3, 8, 7, 9).contiguous()
        return v.view(split * n_tiles * warps, spw, 8, 32, 4)

    wtot = split * n_tiles * warps
    sm_m, d3_m, fl_m, out_m, u_m = to_mma(sm), to_mma(d3), to_mma(is_flip), to_mma(is_out), to_mma(u)

    sm_slab = sm_m.view(wtot, spw, 2, 4, 32, 4).permute(0, 1, 2, 4, 3, 5).contiguous().view(wtot, spw, 1024)
    d3_lane = d3_m.permute(0, 1, 3, 2, 4).reshape(wtot, spw, 32, 32)
    shifts_lo = torch.arange(0, 21, device=w.device, dtype=torch.int64) * 3
    lo63 = (d3_lane[..., :21] << shifts_lo).sum(dim=-1)
    v21 = d3_lane[..., 21]
    e0 = (lo63 & 0xFFFFFFFF).to(torch.int32)
    e1 = (((lo63 >> 32) & 0x7FFFFFFF) | ((v21 & 1) << 31)).to(torch.int32)
    shifts_hi = torch.arange(0, 10, device=w.device, dtype=torch.int64) * 3 + 2
    e2 = ((v21 >> 1) | (d3_lane[..., 22:] << shifts_hi).sum(dim=-1)).to(torch.int32)
    e3_slab = torch.stack([e0, e1, e2], dim=2).contiguous().view(torch.uint8).view(wtot, spw, 384)
    slabs = torch.cat([sm_slab, e3_slab], dim=2).contiguous().view(-1)

    f_idx = fl_m.nonzero(as_tuple=False)
    total_slabs_all = wtot * spw
    if f_idx.numel() > 0:
        widx, sl_i, s_i, lane_i, j_i = f_idx[:, 0], f_idx[:, 1], f_idx[:, 2], f_idx[:, 3], f_idx[:, 4]
        slab_uid = widx * spw + sl_i
        order = torch.argsort(slab_uid)
        widx, sl_i, s_i, lane_i, j_i, slab_uid = (
            widx[order], sl_i[order], s_i[order], lane_i[order], j_i[order], slab_uid[order]
        )
        counts_sl = torch.bincount(slab_uid, minlength=total_slabs_all)
        if int(counts_sl.max().item()) > 32:
            sl_starts = torch.zeros(total_slabs_all + 1, dtype=torch.int64, device=w.device)
            sl_starts[1:] = torch.cumsum(counts_sl, 0)
            rank_in_sl = torch.arange(slab_uid.numel(), device=w.device) - sl_starts[slab_uid]
            keep = rank_in_sl < 32
            exc = ~keep
            out_m[widx[exc], sl_i[exc], s_i[exc], lane_i[exc], j_i[exc]] = True
            widx, sl_i, s_i, lane_i, j_i, slab_uid = (
                widx[keep], sl_i[keep], s_i[keep], lane_i[keep], j_i[keep], slab_uid[keep]
            )
            counts_sl = torch.clamp(counts_sl, max=32)
        flips = ((lane_i << 5) | (s_i << 2) | j_i).to(torch.uint16).contiguous()
        meta = torch.zeros((wtot, 4), dtype=torch.int32, device=w.device)
        warp_counts = counts_sl.view(wtot, spw).sum(dim=1).to(torch.int32)
        warp_foff = torch.zeros(wtot + 1, dtype=torch.int32, device=w.device)
        warp_foff[1:] = torch.cumsum(warp_counts, 0, dtype=torch.int32)
        meta[:, 0] = warp_foff[:wtot]
        np_pad = torch.zeros((wtot, 12), dtype=torch.uint8, device=w.device)
        np_pad[:, :spw] = counts_sl.view(wtot, spw).to(torch.uint8)
        meta[:, 1:] = np_pad.view(torch.int32)
        warp_meta = meta.contiguous()
    else:
        flips = torch.zeros(1, dtype=torch.uint16, device=w.device)
        warp_meta = torch.zeros((wtot, 4), dtype=torch.int32, device=w.device)

    bad = out_m.nonzero(as_tuple=False)
    if bad.numel() > 0:
        widx, sl_i, s_i, lane_i, j_i = bad[:, 0], bad[:, 1], bad[:, 2], bad[:, 3], bad[:, 4]
        step_id = (sl_i << 3) | s_i
        lj = (lane_i << 2) | j_i
        uval = u_m[widx, sl_i, s_i, lane_i, j_i].to(torch.int64) & 0xFFFF
        order = torch.argsort((widx << 24) | (step_id << 8) | lj)
        widx, step_id, lj, uval = widx[order], step_id[order], lj[order], uval[order]
        outliers = ((step_id << 32) | (lj << 16) | uval).to(torch.uint64).contiguous()
        counts = torch.bincount(widx, minlength=wtot).to(torch.int32)
        o_off = torch.zeros(wtot + 1, dtype=torch.int32, device=w.device)
        o_off[1:] = torch.cumsum(counts, 0, dtype=torch.int32)
    else:
        outliers = torch.zeros(1, dtype=torch.uint64, device=w.device)
        o_off = torch.zeros(wtot + 1, dtype=torch.int32, device=w.device)

    min_part = max(_MIN_PART_ELEMS, 2 * n_tiles * split * 16 * bn)
    min_sem = max(_MIN_SEM_TILES, 2 * n_tiles)
    if V5_ENABLED and C32_ENABLED and split in (6, 9):
        max_m_sp = min(C32_MAX_M, 128) if N > 512 else C32_MAX_M
        m_pad_sp = (
            max(((max_m_sp + 127) // 128) * 128, ((max_m_sp + 47) // 48) * 48)
            if (V12_ENABLED or C32_BM == 48)
            else (((max_m_sp + 127) // 128) * 128)
        )
        n_pad_sp = ((N + 31) // 32) * 32
        dual_mult = 2 if N <= 256 else 1
        min_part = max(min_part, dual_mult * split * m_pad_sp * n_pad_sp)
        sem_n = (n_pad_sp // 8) if V12_ENABLED else ((N + 7) // 8)
        min_sem = max(min_sem, dual_mult * sem_n * ((C32_MAX_M + 31) // 32))
    _ensure_scratch(w.device, min_part, min_sem)
    entry = (slabs, flips, warp_meta, outliers, o_off, base, ks, n_active, cfg_id, split, n_tiles, bn)
    _CACHE[key] = entry
    try:
        n_flips = int(flips.numel()) if f_idx.numel() > 0 else 0
        n_outliers = int(bad.shape[0])
        w._inkling_bf11 = entry  # type: ignore[attr-defined]
        w._inkling_bf11_counts = (n_flips, n_outliers)  # type: ignore[attr-defined]
    except Exception:
        pass
    if V5_ENABLED:
        build_v5_sidecar(w)
    return slabs


def get_v5_sidecar(w: torch.Tensor | None):
    if not V5_ENABLED or w is None or w.dim() != 2:
        return None
    direct = getattr(w, "_inkling_bf11_v5", None)
    if direct is not None:
        return direct
    ptr = int(w.data_ptr())
    if ptr == 0:
        return None
    return _V5_CACHE.get((ptr, int(w.shape[0]), int(w.shape[1])))


@torch.no_grad()
def build_v5_sidecar(w: torch.Tensor, flip_thresh: int | None = None):
    """Build adaptive v5 flip sidecar `(slab_meta_v5, sflips, bmasks, thresh)` on GPU directly
    from existing packed BF11 tensors (NEVER touches raw BF16 storage -> 100% FREE_RAW=1 safe!)."""
    if not V5_ENABLED or w is None or w.dim() != 2 or torch.cuda.is_current_stream_capturing():
        return None
    lib = _get_lib()
    if lib is None or not _HAS_V5:
        return None
    entry = get_packed(w)
    if entry is None:
        return None
    thresh = V5_FLIP_THRESH if flip_thresh is None else max(0, min(32, int(flip_thresh)))
    existing = get_v5_sidecar(w)
    if existing is not None and int(existing[3]) == thresh:
        return existing
    slabs, flips, warp_meta, outliers, o_off, _base, ks, _n_act, cfg_id, split, n_tiles, _bn = entry
    warps, sub_n, _bn2 = _CFGS[cfg_id]
    spw = (ks // 128) * sub_n
    wtot = split * n_tiles * warps
    dev = slabs.device
    st = int(torch.cuda.current_stream().cuda_stream)

    w_s_cnt = torch.empty(wtot, dtype=torch.int32, device=dev)
    w_d_cnt = torch.empty(wtot, dtype=torch.int32, device=dev)
    rc = lib.bf11_v5_measure_sidecar_from_bf11(
        warp_meta.data_ptr(), w_s_cnt.data_ptr(), w_d_cnt.data_ptr(), wtot, spw, thresh, st
    )
    if rc != 0:
        raise RuntimeError(f"bf11_v5_measure_sidecar_from_bf11 failed with cudaError={rc}")

    w_s_off = torch.zeros(wtot + 1, dtype=torch.int32, device=dev)
    w_d_off = torch.zeros(wtot + 1, dtype=torch.int32, device=dev)
    w_s_off[1:] = torch.cumsum(w_s_cnt, 0, dtype=torch.int32)
    w_d_off[1:] = torch.cumsum(w_d_cnt, 0, dtype=torch.int32)
    total_s = int(w_s_off[-1].item())
    total_d = int(w_d_off[-1].item())

    slab_meta_v5 = torch.empty(wtot * spw, dtype=torch.int32, device=dev)
    sflips = torch.empty(max(1, total_s), dtype=torch.uint16, device=dev)
    bmasks = torch.empty(max(32, total_d * 32), dtype=torch.int32, device=dev)
    rc2 = lib.bf11_v5_build_sidecar_from_bf11(
        flips.data_ptr(), warp_meta.data_ptr(), outliers.data_ptr(), o_off.data_ptr(),
        w_s_off.data_ptr(), w_d_off.data_ptr(),
        slab_meta_v5.data_ptr(), sflips.data_ptr(), bmasks.data_ptr(),
        wtot, spw, thresh, st,
    )
    if rc2 != 0:
        raise RuntimeError(f"bf11_v5_build_sidecar_from_bf11 failed with cudaError={rc2}")

    sc = (slab_meta_v5, sflips, bmasks, thresh)
    ptr = int(w.data_ptr())
    if ptr != 0:
        _V5_CACHE[(ptr, int(w.shape[0]), int(w.shape[1]))] = sc
    try:
        w._inkling_bf11_v5 = sc  # type: ignore[attr-defined]
    except Exception:
        pass
    return sc


def _v5_c32_scratch_req(M: int, N: int, split: int, dual: bool = False) -> tuple[int, int]:
    m_pad = (((M + 47) // 48) * 48) if (C32_BM == 48) else (((M + 127) // 128) * 128)
    n_pad = ((N + 31) // 32) * 32
    mult = 2 if dual else 1
    need_p = mult * split * m_pad * n_pad
    need_s = mult * ((n_pad // 8) if V12_ENABLED else ((N + 7) // 8)) * ((M + 31) // 32)
    return need_p, need_s


@torch.no_grad()
def unpack_bf16(w: torch.Tensor) -> torch.Tensor | None:
    """Lossless 0-ULP reconstruction of `[N, K]` BF16 dense weight from BF11 slabs + flips + outliers.

    Zero host syncs (no `.item()`), 100% CUDA-graph-capture-safe fallback when raw BF16 storage was freed.
    """
    entry = get_packed(w)
    if entry is None:
        return None
    slabs, flips, warp_meta, outliers, o_off, base, ks, n_active, cfg_id, split, n_tiles, bn = entry
    N, K = int(w.shape[0]), int(w.shape[1])
    if V12_FAST_UNPACK:
        lib = _get_lib()
        if lib is not None and _HAS_V12_UNPACK:
            out = torch.empty((N, K), dtype=torch.bfloat16, device=slabs.device)
            st = int(torch.cuda.current_stream(slabs.device).cuda_stream)
            rc = lib.bf11_unpack_bf16_v12_fwd(
                slabs.data_ptr(), flips.data_ptr(), warp_meta.data_ptr(),
                outliers.data_ptr(), o_off.data_ptr(), out.data_ptr(),
                N, n_active, K, ks, cfg_id, split, base, st,
            )
            if rc == 0:
                return out
            raise RuntimeError(f"bf11_unpack_bf16_v12_fwd failed with cudaError={rc}")
    warps, sub_n, _bn = _CFGS[cfg_id]
    n_chunks = ks // 128
    n_pad = n_tiles * bn
    spw = n_chunks * sub_n
    wtot = split * n_tiles * warps
    dev = slabs.device

    slab_v = slabs.view(wtot, spw, 1408)
    sm_slab = slab_v[:, :, :1024].contiguous()
    e3_slab = slab_v[:, :, 1024:].contiguous()

    sm_m = (
        sm_slab.view(wtot, spw, 2, 32, 4, 4)
        .permute(0, 1, 2, 4, 3, 5)
        .contiguous()
        .view(wtot, spw, 8, 32, 4)
        .to(torch.int32)
    )
    e_i32 = e3_slab.view(torch.int32).view(wtot, spw, 3, 32)
    e0u = e_i32[:, :, 0, :].to(torch.int64) & 0xFFFFFFFF
    e1u = e_i32[:, :, 1, :].to(torch.int64) & 0xFFFFFFFF
    e2u = e_i32[:, :, 2, :].to(torch.int64) & 0xFFFFFFFF
    lo63 = e0u | ((e1u & 0x7FFFFFFF) << 32)
    shifts_lo = torch.arange(0, 21, device=dev, dtype=torch.int64) * 3
    lo21 = (lo63.unsqueeze(-1) >> shifts_lo) & 7
    v21 = (((e1u >> 31) & 1) | ((e2u & 3) << 1)).unsqueeze(-1)
    shifts_hi = torch.arange(0, 10, device=dev, dtype=torch.int64) * 3 + 2
    hi10 = (e2u.unsqueeze(-1) >> shifts_hi) & 7
    d3_lane = torch.cat([lo21, v21, hi10], dim=-1).to(torch.int32)
    delta_m = d3_lane.view(wtot, spw, 32, 8, 4).permute(0, 1, 3, 2, 4).contiguous()

    counts_pair = getattr(w, "_inkling_bf11_counts", None)
    if counts_pair is not None:
        n_flips, n_outliers = int(counts_pair[0]), int(counts_pair[1])
    elif not torch.cuda.is_current_stream_capturing():
        n_flips = int(warp_meta[:, 1:].contiguous().view(torch.uint8)[:, :spw].to(torch.int64).sum().item())
        n_outliers = int(o_off[-1].item())
    else:
        n_flips, n_outliers = 0, 0

    if n_flips > 0:
        counts_sl = warp_meta[:, 1:].contiguous().view(torch.uint8)[:, :spw].reshape(-1).to(torch.int64)
        slab_ends = torch.cumsum(counts_sl, 0)
        idx_f = torch.arange(n_flips, device=dev, dtype=torch.int64)
        slab_uid = torch.searchsorted(slab_ends, idx_f, right=True)
        widx_f = slab_uid // spw
        sl_f = slab_uid % spw
        fv = flips[:n_flips].to(torch.int64) & 0xFFFF
        lane_f = (fv >> 5) & 31
        s_f = (fv >> 2) & 7
        j_f = fv & 3
        flat_f = (((widx_f * spw + sl_f) * 8 + s_f) * 32 + lane_f) * 4 + j_f
        delta_m.reshape(-1).scatter_add_(
            0, flat_f, torch.full((n_flips,), 8, dtype=torch.int32, device=dev)
        )

    exp_m = int(base) - delta_m
    u_m = (((sm_m & 0x80) << 8) | (exp_m << 7) | (sm_m & 0x7F)).contiguous()

    if n_outliers > 0:
        ov = outliers[:n_outliers].view(torch.int64)
        idx_o = torch.arange(n_outliers, device=dev, dtype=torch.int32)
        widx_o = torch.searchsorted(o_off[1:], idx_o, right=True).to(torch.int64)
        step_id = (ov >> 32) & 0xFFFFFFFF
        sl_o = step_id >> 3
        s_o = step_id & 7
        lj = (ov >> 16) & 0xFFFF
        lane_o = (lj >> 2) & 31
        j_o = lj & 3
        uval = (ov & 0xFFFF).to(torch.int32)
        flat_o = (((widx_o * spw + sl_o) * 8 + s_o) * 32 + lane_o) * 4 + j_o
        u_m.reshape(-1).scatter_(0, flat_o, uval)

    u_pad = (
        u_m.view(split, n_tiles, warps, n_chunks, sub_n, 8, 8, 4, 2, 2)
        .permute(1, 2, 4, 6, 0, 3, 5, 8, 7, 9)
        .contiguous()
        .view(n_pad, K)
    )
    u_out = u_pad[:N].contiguous()
    if n_active < N:
        u_out[n_active:] = 0
    return u_out.to(torch.int16).view(torch.bfloat16)


@torch.no_grad()
def free_raw_weight(w: torch.Tensor) -> int:
    """Opt-in (`INKLING_BF11_FREE_RAW=1`): release full-ownership raw BF16 storage after BF11 packing."""
    if not FREE_RAW or w is None or get_packed(w) is None:
        return 0
    if w.storage_offset() != 0:
        return 0
    if FREE_RAW_SAFE:
        # Guard against v9 live 4.5x/16.7W unpack_bf16() capture storm:
        # NEVER free raw BF16 storage unless C=4/8/32 active buckets are actually qualified!
        _get_lib()
        if not C32_ENABLED or not _HAS_C32:
            return 0
        N_w, K_w = int(w.shape[0]), int(w.shape[1])
        req_ms = [m for m in (24, 32, 36, 48, 64) if m <= C32_MAX_M]
        if any(not _C32_QUAL_OK.get((N_w, K_w, m), False) for m in req_ms):
            return 0
    raw_bytes = int(w.numel() * w.element_size())
    stor = w.data.untyped_storage()
    if int(stor.size()) != raw_bytes or raw_bytes <= 0:
        return 0
    stor.resize_(0)
    return raw_bytes


_L2PF_MULTI = os.environ.get("INKLING_BF11_L2PF_MULTI", "0").strip().lower() not in ("0", "", "off", "false", "no")
_L2PF_FLIPS_BYTES = int(os.environ.get("INKLING_BF11_L2PF_FLIPS_KB", "128")) * 1024


def _is_full_decode_capturing() -> bool:
    """True ONLY inside FULL decode CUDA graph capture (never eager or PIECEWISE prefill)."""
    if not torch.cuda.is_current_stream_capturing():
        return False
    try:
        from . import l2pf
        if l2pf._is_piecewise_capturing():
            return False
    except Exception:
        pass
    try:
        from vllm.config.compilation import CUDAGraphMode
        from vllm.forward_context import get_forward_context, is_forward_context_available

        if is_forward_context_available():
            return get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.FULL
    except Exception:
        pass
    return True


def _bf16_ulp_diff(a: torch.Tensor, b: torch.Tensor) -> int:
    """Exact max ULP distance between two finite bf16 tensors (sign-magnitude ordered)."""
    ai = a.view(torch.uint16).to(torch.int32)
    bi = b.view(torch.uint16).to(torch.int32)
    ao = torch.where((ai & 0x8000) != 0, 0x8000 - ai, ai)
    bo = torch.where((bi & 0x8000) != 0, 0x8000 - bi, bi)
    return int((ao - bo).abs().max().item())


def _bf16_fp32_exact_ok(out: torch.Tensor, x: torch.Tensor, w: torch.Tensor) -> bool:
    """True iff `out` matches true FP32 `x @ w.T` within <= 2 BF16 ULPs (immune to zero-crossing
    ULP jumps and cuBLAS SM121a BF16-reduced split-K heuristics)."""
    ref32 = torch.mm(x.float(), w.float().t())
    err = (out.float() - ref32).abs()
    tol = ref32.abs() * (2.0 / 128.0) + 1e-3
    return bool((err <= tol).all().item())


def will_use_bf11(w: torch.Tensor | None, num_tokens: int | None = None) -> bool:
    """True iff bf11_linear will serve this weight for num_tokens (exact L2PF hygiene guard)."""
    if not ENABLED or w is None or w.dim() != 2:
        return False
    if get_packed(w) is None:
        return False
    if num_tokens is None or num_tokens <= 16:
        return True
    if not C32_ENABLED or num_tokens > C32_MAX_M:
        return False
    if C32_MIN_N > 0 and int(w.shape[0]) < C32_MIN_N:
        return False
    _get_lib()
    if not _HAS_C32:
        return False
    if not C32_QUALIFY:
        return True
    ok = _C32_QUAL_OK.get((int(w.shape[0]), int(w.shape[1]), int(num_tokens)))
    return bool(ok) if ok is not None else False


def l2pf_tensor(w: torch.Tensor | None, num_tokens: int | None = None):
    if not ENABLED or w is None or w.dim() != 2:
        return w
    hit = get_packed(w)
    if hit is None:
        return w
    if w.untyped_storage().size() > 0 and num_tokens is not None and num_tokens > 16 and not will_use_bf11(w, num_tokens):
        return w
    if _L2PF_MULTI:
        sc = get_v5_sidecar(w)
        if sc is not None:
            return (sc[0], (sc[2], _L2PF_FLIPS_BYTES), hit[4], hit[0])
        return (hit[2], (hit[1], _L2PF_FLIPS_BYTES), hit[4], hit[0])
    return hit[0]


def get_packed(w: torch.Tensor):
    if not ENABLED or w is None or w.dim() != 2:
        return None
    direct = getattr(w, "_inkling_bf11", None)
    if direct is not None:
        return direct
    ptr = int(w.data_ptr())
    if ptr == 0:
        return None
    return _CACHE.get((ptr, int(w.shape[0]), int(w.shape[1])))


def _qualify_c32_one(w: torch.Tensor, M: int) -> bool:
    """Fail-closed 0-ULP qualification of bf11_flip_c32_fwd on (N, K, M) outside graph capture."""
    N, K = int(w.shape[0]), int(w.shape[1])
    qkey = (N, K, M)
    cached = _C32_QUAL_OK.get(qkey)
    if cached is not None:
        return cached
    if C32_MIN_N > 0 and N < C32_MIN_N:
        _C32_QUAL_OK[qkey] = False
        _C32_TORCH_EXACT[qkey] = False
        return False
    if not C32_QUALIFY:
        _C32_QUAL_OK[qkey] = True
        return True
    if torch.cuda.is_current_stream_capturing():
        return False
    lib = _get_lib()
    entry = get_packed(w)
    if lib is None or not _HAS_C32 or entry is None:
        _C32_QUAL_OK[qkey] = False
        _C32_TORCH_EXACT[qkey] = False
        return False
    slabs, flips, warp_meta, outliers, o_off, base, ks, n_active, cfg_id, split, n_tiles, bn = entry
    v5_sc = build_v5_sidecar(w) if (V5_ENABLED and V5_C32 and _HAS_V5) else None
    w_ref = w if w.untyped_storage().size() > 0 else unpack_bf16(w)
    if w_ref is None:
        _C32_QUAL_OK[qkey] = False
        _C32_TORCH_EXACT[qkey] = False
        return False
    dev = slabs.device
    ok = True
    torch_exact = True
    g = torch.Generator(device=dev)
    g.manual_seed(7777 + N + M)
    skinny_mod = None
    if C32_EXACT_REF in ("exact", "skinny"):
        try:
            from . import skinny as skinny_mod
        except Exception:
            skinny_mod = None

    def _run_c32(x_in: torch.Tensor, o_out: torch.Tensor, bm_val: int, ko_val: int, split_req: int) -> int:
        st_l = int(torch.cuda.current_stream().cuda_stream)
        if v5_sc is not None:
            smeta_v5, sflips_v5, bmasks_v5, _th = v5_sc
            p_ptr, s_ptr = 0, 0
            if ko_val == 1 and split in (6, 9) and split_req != 1:
                need_p, need_s = _v5_c32_scratch_req(M, N, split, dual=False)
                part_b, sem_b = _next_scratch(dev, need_p, need_s)
                p_ptr, s_ptr = part_b.data_ptr(), sem_b.data_ptr()
            fn_c32 = lib.bf11_flip_v12_c32_fwd if (V12_ENABLED and V12_BM48 and _HAS_V12) else lib.bf11_flip_v5_c32_fwd
            return fn_c32(
                x_in.data_ptr(), slabs.data_ptr(),
                smeta_v5.data_ptr(), sflips_v5.data_ptr(), bmasks_v5.data_ptr(),
                outliers.data_ptr(), o_off.data_ptr(),
                p_ptr, s_ptr, o_out.data_ptr(), 0,
                M, N, n_active, K, int(x_in.stride(0)), ks, split, int(o_out.stride(0)),
                cfg_id, bm_val, V5_W_RUN, split_req, ko_val, base, 0, 0, st_l,
            )
        return lib.bf11_flip_c32_fwd(
            x_in.data_ptr(), slabs.data_ptr(), flips.data_ptr(), warp_meta.data_ptr(),
            outliers.data_ptr(), o_off.data_ptr(), o_out.data_ptr(), 0,
            M, N, n_active, K, int(x_in.stride(0)), ks, split, int(o_out.stride(0)),
            cfg_id, bm_val, ko_val, base, 0, 0, st_l,
        )

    for sc in (1.0, 0.25):
        x = (torch.randn((M, K), device=dev, generator=g) * sc).to(torch.bfloat16)
        out = torch.empty((M, N), dtype=torch.bfloat16, device=dev)
        rc = _run_c32(x, out, C32_BM, C32_KORDER, V5_SPLIT_CTA)
        if rc != 0:
            ok = False
            torch_exact = False
            break
        ref_torch = torch.mm(x, w_ref.t())
        if not bool(torch.equal(out.view(torch.int16), ref_torch.view(torch.int16))):
            torch_exact = False
        if C32_EXACT_REF == "torch":
            if not torch_exact:
                ok = False
                break
        elif C32_EXACT_REF in ("exact", "skinny"):
            # 1) End-to-end 0-ULP proof against RAW BF16 `w` across ALL M rows via chunked skinny_linear (k_order=1).
            if skinny_mod is None or not skinny_mod.supported(x[: min(16, M)], w_ref):
                ok = False
                break
            out_k1 = out if C32_KORDER == 1 else torch.empty_like(out)
            if C32_KORDER != 1:
                rc1 = _run_c32(x, out_k1, C32_BM, 1, V5_SPLIT_CTA)
                if rc1 != 0:
                    ok = False
                    break
            ref_skinny = torch.cat(
                [skinny_mod.skinny_linear(x[m0 : min(m0 + 16, M)], w_ref) for m0 in range(0, M, 16)],
                dim=0,
            )
            if not bool(torch.equal(out_k1.view(torch.int16), ref_skinny.view(torch.int16))):
                ok = False
                break
            # 2) Verify literal 0-ULP BM=32 & single-CTA invariance for active C32_KORDER.
            ref_bm32 = torch.empty_like(out)
            rc2 = _run_c32(x, ref_bm32, 32, C32_KORDER, 1)
            if rc2 != 0 or not bool(torch.equal(out.view(torch.int16), ref_bm32.view(torch.int16))):
                ok = False
                break
            # 3) Sanity-check against true FP32 matmul (<= 2 BF16 ULPs; avoids cuBLAS BF16-reduced split-K & 0-crossing ULP artifacts).
            if not _bf16_fp32_exact_ok(out, x, w_ref):
                ok = False
                break
        else:
            # Self-consistency mode: 0-ULP bit-identity across BM=32 vs configured C32_BM + bounded ULP vs torch.
            ref = torch.empty_like(out)
            rc2 = _run_c32(x, ref, 32, C32_KORDER, 1)
            if rc2 != 0 or not bool(torch.equal(out.view(torch.int16), ref.view(torch.int16))):
                ok = False
                break
            if not _bf16_fp32_exact_ok(out, x, w_ref):
                ok = False
                break
    _C32_QUAL_OK[qkey] = ok
    _C32_TORCH_EXACT[qkey] = ok and torch_exact
    return ok


def qualify_c32(weights: list[torch.Tensor]) -> dict[tuple[int, int, int], bool]:
    """Pre-qualify canonical C=32 M buckets on representative prepacked model weights."""
    if not ENABLED or not C32_ENABLED or torch.cuda.is_current_stream_capturing():
        return {}
    lib = _get_lib()
    if lib is None or not _HAS_C32:
        return {}
    seen_shapes: dict[tuple[int, int], torch.Tensor] = {}
    for w in weights:
        if w is not None and w.dim() == 2 and get_packed(w) is not None:
            sh = (int(w.shape[0]), int(w.shape[1]))
            if sh not in seen_shapes:
                seen_shapes[sh] = w
    canon = _C32_CANONICAL_MS if C32_EXACT_REF == "torch" else _C32_CANONICAL_MS_EXACT
    ms = sorted({m for m in (*canon, *_C32_EXTRA_MS) if 17 <= m <= C32_MAX_M})
    n_pass, n_fail, n_torch = 0, 0, 0
    for sh, w in seen_shapes.items():
        for m in ms:
            if _qualify_c32_one(w, m):
                n_pass += 1
                if _C32_TORCH_EXACT.get((sh[0], sh[1], m), False):
                    n_torch += 1
            else:
                n_fail += 1
    if seen_shapes:
        print(
            f"[inkling-bf11-c32] qualified {n_pass}/{n_pass + n_fail} (shape, M) pairs "
            f"(ref={C32_EXACT_REF}, torch_exact={n_torch}/{n_pass + n_fail}, "
            f"capture_only={int(C32_CAPTURE_ONLY)}, min_n={C32_MIN_N}, "
            f"k_order={C32_KORDER}, bm={C32_BM}, max_m={C32_MAX_M})",
            flush=True,
        )
    return dict(_C32_QUAL_OK)


def bf11_linear(
    x: torch.Tensor,
    w: torch.Tensor,
    out: torch.Tensor | None = None,
    pdl: bool = False,
    trig: bool = False,
) -> torch.Tensor | None:
    """Run 0-ULP BF11_FLIP_EXACT CUDA GEMM (M<=16 S7 path; opt-in M=17..C32_MAX_M C=32 path)."""
    if not ENABLED or x.dim() != 2 or x.dtype != torch.bfloat16 or x.stride(1) != 1:
        return None
    M, K = int(x.shape[0]), int(x.shape[1])
    if M > 16:
        if not C32_ENABLED or M > C32_MAX_M:
            return None
        N = int(w.shape[0])
        if C32_MIN_N > 0 and N < C32_MIN_N:
            return None
        lib = _get_lib()
        if lib is None or not _HAS_C32:
            return None
        entry = get_packed(w)
        if entry is None:
            return None
        if not _qualify_c32_one(w, M):
            return None
        if C32_CAPTURE_ONLY and not _C32_TORCH_EXACT.get((N, K, M), False) and not _is_full_decode_capturing():
            return None
        slabs, flips, warp_meta, outliers, o_off, base, ks, n_active, cfg_id, split, n_tiles, bn = entry
        if out is None:
            out = torch.empty((M, N), dtype=x.dtype, device=x.device)
        nlive_ptr = _rowar_nlive_ptr(M)
        st = int(torch.cuda.current_stream().cuda_stream)
        v5_sc = get_v5_sidecar(w) if (V5_ENABLED and V5_C32 and _HAS_V5) else None
        if v5_sc is not None:
            smeta_v5, sflips_v5, bmasks_v5, _th = v5_sc
            p_ptr, s_ptr = 0, 0
            if C32_KORDER == 1 and split in (6, 9) and V5_SPLIT_CTA != 1:
                need_p, need_s = _v5_c32_scratch_req(M, N, split, dual=False)
                part_b, sem_b = _next_scratch(x.device, need_p, need_s)
                p_ptr, s_ptr = part_b.data_ptr(), sem_b.data_ptr()
            fn_c32 = lib.bf11_flip_v12_c32_fwd if (V12_ENABLED and V12_BM48 and _HAS_V12) else lib.bf11_flip_v5_c32_fwd
            rc = fn_c32(
                x.data_ptr(), slabs.data_ptr(),
                smeta_v5.data_ptr(), sflips_v5.data_ptr(), bmasks_v5.data_ptr(),
                outliers.data_ptr(), o_off.data_ptr(),
                p_ptr, s_ptr, out.data_ptr(), nlive_ptr,
                M, N, n_active, K, int(x.stride(0)), ks, split, int(out.stride(0)),
                cfg_id, C32_BM, V5_W_RUN, V5_SPLIT_CTA, C32_KORDER, base,
                1 if pdl else 0, 1 if trig else 0, st,
            )
            if rc != 0:
                raise RuntimeError(f"bf11_flip_v5_c32_fwd failed with cudaError={rc}")
            return out
        rc = lib.bf11_flip_c32_fwd(
            x.data_ptr(), slabs.data_ptr(), flips.data_ptr(), warp_meta.data_ptr(),
            outliers.data_ptr(), o_off.data_ptr(), out.data_ptr(), nlive_ptr,
            M, N, n_active, K, int(x.stride(0)), ks, split, int(out.stride(0)),
            cfg_id, C32_BM, C32_KORDER, base, 1 if pdl else 0, 1 if trig else 0, st,
        )
        if rc != 0:
            raise RuntimeError(f"bf11_flip_c32_fwd failed with cudaError={rc}")
        return out
    lib = _get_lib()
    if lib is None:
        return None
    entry = get_packed(w)
    if entry is None:
        return None
    slabs, flips, warp_meta, outliers, o_off, base, ks, n_active, cfg_id, split, n_tiles, bn = entry
    N = int(w.shape[0])
    if out is None:
        out = torch.empty((M, N), dtype=x.dtype, device=x.device)
    if M == 0:
        return out
    v5_sc = get_v5_sidecar(w) if (V5_ENABLED and V5_C1 and _HAS_V5) else None
    if split == 1:
        part = None
        sem = None
    else:
        n_pad_c1 = ((N + 31) // 32) * 32 if v5_sc is not None else (n_tiles * bn)
        sem_c1 = max((N + 7) // 8, n_tiles * (2 if cfg_id == 4 else 4)) if v5_sc is not None else n_tiles
        part, sem = _next_scratch(x.device, split * M * n_pad_c1, sem_c1)
    nlive_ptr = _rowar_nlive_ptr(M)
    st = int(torch.cuda.current_stream().cuda_stream)
    if v5_sc is not None:
        smeta_v5, sflips_v5, bmasks_v5, _th = v5_sc
        fn_c1 = lib.bf11_flip_v12_c1_fwd if (V12_ENABLED and _HAS_V12) else lib.bf11_flip_v5_c1_fwd
        w_run_c1 = V5_W_RUN if V5_W_RUN != 0 else (0 if (V12_ENABLED and _HAS_V12) else (2 if (C1_SPLIT6_WRUN2 and split == 6 and cfg_id in (1, 4, 5)) else 0))
        rc = fn_c1(
            x.data_ptr(), slabs.data_ptr(),
            smeta_v5.data_ptr(), sflips_v5.data_ptr(), bmasks_v5.data_ptr(),
            outliers.data_ptr(), o_off.data_ptr(),
            0 if part is None else part.data_ptr(),
            0 if sem is None else sem.data_ptr(),
            out.data_ptr(), nlive_ptr,
            M, N, n_active, K, int(x.stride(0)), ks, int(out.stride(0)),
            cfg_id, w_run_c1, split, base, 1 if pdl else 0, 1 if trig else 0, st,
        )
        if rc != 0:
            raise RuntimeError(f"bf11_flip_v5_c1_fwd failed with cudaError={rc}")
        return out
    rc = lib.bf11_flip_fwd(
        x.data_ptr(), slabs.data_ptr(), flips.data_ptr(), warp_meta.data_ptr(),
        outliers.data_ptr(), o_off.data_ptr(),
        0 if part is None else part.data_ptr(),
        0 if sem is None else sem.data_ptr(),
        out.data_ptr(), nlive_ptr,
        M, N, n_active, K, int(x.stride(0)), ks, int(out.stride(0)),
        cfg_id, split, base, 1 if pdl else 0, 1 if trig else 0, st,
    )
    if rc != 0:
        raise RuntimeError(f"bf11_flip_fwd failed with cudaError={rc}")
    return out


def bf11_dual_kv(
    x_k: torch.Tensor,
    x_v: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    pdl: bool = False,
    trig: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Fused 0-ULP global-layer K+V projection in ONE single-kernel launch (M<=16; opt-in M=17..C32_MAX_M)."""
    if (
        not ENABLED or not USE_DUAL_KV
        or x_k.dim() != 2 or x_v.dim() != 2
        or x_k.shape != x_v.shape
        or x_k.dtype != torch.bfloat16 or x_v.dtype != torch.bfloat16
        or x_k.stride(1) != 1 or x_v.stride(1) != 1
    ):
        return None
    M, K = int(x_k.shape[0]), int(x_k.shape[1])
    if M > 16 and (not C32_ENABLED or not C32_DUAL_KV or M > C32_MAX_M):
        return None
    lib = _get_lib()
    if lib is None:
        return None
    ek = get_packed(w_k)
    ev = get_packed(w_v)
    if ek is None or ev is None:
        return None
    if (
        ek[8] != ev[8]
        or ek[9] != ev[9]
        or (M <= 16 and ek[9] != 9)
        or w_k.shape != w_v.shape
        or ek[7] != w_k.shape[0]
        or ev[7] != w_v.shape[0]
    ):
        return None
    sl0, fl0, wm0, ol0, oo0, b0, ks, _, cfg_id, split, n_tiles, bn = ek
    sl1, fl1, wm1, ol1, oo1, b1, _, _, _, _, _, _ = ev
    N_each = int(w_k.shape[0])
    if M > 16:
        if not _HAS_C32 or cfg_id not in (1, 4, 5):
            return None
        if C32_MIN_N > 0 and N_each < C32_MIN_N:
            return None
        if not (_qualify_c32_one(w_k, M) and _qualify_c32_one(w_v, M)):
            return None
        if (
            C32_CAPTURE_ONLY
            and not _C32_TORCH_EXACT.get((N_each, K, M), False)
            and not _is_full_decode_capturing()
        ):
            return None
        out_k = torch.empty((M, N_each), dtype=x_k.dtype, device=x_k.device)
        out_v = torch.empty((M, N_each), dtype=x_v.dtype, device=x_v.device)
        nlive_ptr = _rowar_nlive_ptr(M)
        st = int(torch.cuda.current_stream().cuda_stream)
        sc0 = get_v5_sidecar(w_k) if (V5_ENABLED and V5_C32 and _HAS_V5) else None
        sc1 = get_v5_sidecar(w_v) if (V5_ENABLED and V5_C32 and _HAS_V5) else None
        if sc0 is not None and sc1 is not None:
            p_ptr, s_ptr = 0, 0
            if C32_KORDER == 1 and split == 9 and V5_SPLIT_CTA != 1:
                need_p, need_s = _v5_c32_scratch_req(M, N_each, split, dual=True)
                part_b, sem_b = _next_scratch(x_k.device, need_p, need_s)
                p_ptr, s_ptr = part_b.data_ptr(), sem_b.data_ptr()
            fn_dkv_c32 = lib.bf11_flip_dual_kv_v12_c32_fwd if (V12_ENABLED and V12_BM48 and _HAS_V12) else lib.bf11_flip_dual_kv_v5_c32_fwd
            rc = fn_dkv_c32(
                x_k.data_ptr(), x_v.data_ptr(),
                sl0.data_ptr(), sl1.data_ptr(),
                sc0[0].data_ptr(), sc1[0].data_ptr(),
                sc0[1].data_ptr(), sc1[1].data_ptr(),
                sc0[2].data_ptr(), sc1[2].data_ptr(),
                ol0.data_ptr(), ol1.data_ptr(),
                oo0.data_ptr(), oo1.data_ptr(),
                p_ptr, s_ptr, out_k.data_ptr(), out_v.data_ptr(), nlive_ptr,
                M, N_each, K, int(x_k.stride(0)), int(x_v.stride(0)), ks, split,
                int(out_k.stride(0)), int(out_v.stride(0)),
                cfg_id, C32_BM, V5_W_RUN, V5_SPLIT_CTA, C32_KORDER, b0, b1,
                1 if pdl else 0, 1 if trig else 0, st,
            )
            if rc != 0:
                raise RuntimeError(f"bf11_flip_dual_kv_v5_c32_fwd failed with cudaError={rc}")
            return out_k, out_v
        rc = lib.bf11_flip_dual_kv_c32_fwd(
            x_k.data_ptr(), x_v.data_ptr(),
            sl0.data_ptr(), sl1.data_ptr(),
            fl0.data_ptr(), fl1.data_ptr(),
            wm0.data_ptr(), wm1.data_ptr(),
            ol0.data_ptr(), ol1.data_ptr(),
            oo0.data_ptr(), oo1.data_ptr(),
            out_k.data_ptr(), out_v.data_ptr(), nlive_ptr,
            M, N_each, K, int(x_k.stride(0)), int(x_v.stride(0)), ks, split,
            int(out_k.stride(0)), int(out_v.stride(0)),
            cfg_id, C32_KORDER, b0, b1, 1 if pdl else 0, 1 if trig else 0, st,
        )
        if rc != 0:
            raise RuntimeError(f"bf11_flip_dual_kv_c32_fwd failed with cudaError={rc}")
        return out_k, out_v
    out_k = torch.empty((M, N_each), dtype=x_k.dtype, device=x_k.device)
    out_v = torch.empty((M, N_each), dtype=x_v.dtype, device=x_v.device)
    if M == 0:
        return out_k, out_v
    sc0 = get_v5_sidecar(w_k) if (V5_ENABLED and V5_C1 and _HAS_V5) else None
    sc1 = get_v5_sidecar(w_v) if (V5_ENABLED and V5_C1 and _HAS_V5) else None
    n_pad_kv = ((N_each + 31) // 32) * 32 if (sc0 is not None and sc1 is not None) else (n_tiles * bn)
    sem_kv = (2 * ((N_each + 7) // 8)) if (sc0 is not None and sc1 is not None) else (2 * n_tiles)
    part, sem = _next_scratch(x_k.device, 2 * split * M * n_pad_kv, sem_kv)
    nlive_ptr = _rowar_nlive_ptr(M)
    st = int(torch.cuda.current_stream().cuda_stream)
    if sc0 is not None and sc1 is not None:
        fn_dkv_c1 = lib.bf11_flip_dual_kv_v12_c1_fwd if (V12_ENABLED and _HAS_V12) else lib.bf11_flip_dual_kv_v5_c1_fwd
        rc = fn_dkv_c1(
            x_k.data_ptr(), x_v.data_ptr(),
            sl0.data_ptr(), sl1.data_ptr(),
            sc0[0].data_ptr(), sc1[0].data_ptr(),
            sc0[1].data_ptr(), sc1[1].data_ptr(),
            sc0[2].data_ptr(), sc1[2].data_ptr(),
            ol0.data_ptr(), ol1.data_ptr(),
            oo0.data_ptr(), oo1.data_ptr(),
            part.data_ptr(), sem.data_ptr(), out_k.data_ptr(), out_v.data_ptr(), nlive_ptr,
            M, N_each, K, int(x_k.stride(0)), int(x_v.stride(0)), ks,
            int(out_k.stride(0)), int(out_v.stride(0)),
            cfg_id, V5_W_RUN, split, b0, b1, 1 if pdl else 0, 1 if trig else 0, st,
        )
        if rc != 0:
            raise RuntimeError(f"bf11_flip_dual_kv_v5_c1_fwd failed with cudaError={rc}")
        return out_k, out_v
    rc = lib.bf11_flip_dual_kv_fwd(
        x_k.data_ptr(), x_v.data_ptr(),
        sl0.data_ptr(), sl1.data_ptr(),
        fl0.data_ptr(), fl1.data_ptr(),
        wm0.data_ptr(), wm1.data_ptr(),
        ol0.data_ptr(), ol1.data_ptr(),
        oo0.data_ptr(), oo1.data_ptr(),
        part.data_ptr(), sem.data_ptr(), out_k.data_ptr(), out_v.data_ptr(), nlive_ptr,
        M, N_each, K, int(x_k.stride(0)), int(x_v.stride(0)), ks,
        int(out_k.stride(0)), int(out_v.stride(0)),
        cfg_id, split, b0, b1, 1 if pdl else 0, 1 if trig else 0, st,
    )
    if rc != 0:
        raise RuntimeError(f"bf11_flip_dual_kv_fwd failed with cudaError={rc}")
    return out_k, out_v


# ---- Wave-V9 Fused Global `k_dyn` / `v_dyn` ([M, N_dyn<=4]) Startup 0-ULP Qualifier & Launcher ----
_DYN_SPLIT_CANDS = (1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 32, 36)
_DYN_TREE_SPLITS = (2, 4, 8, 16, 32)


def _qualify_dyn_one(w_dyn: torch.Tensor, M: int) -> tuple[int, int, bool] | None:
    """Prove literal 0-ULP bit-equality of fused `dyn` Tensor Core path vs `F.linear(x, w_dyn)`."""
    N_dyn, K = int(w_dyn.shape[0]), int(w_dyn.shape[1])
    key = (N_dyn, K, M)
    if key in _DYN_QUAL:
        return _DYN_QUAL[key]
    if (
        not FUSE_DYN
        or N_dyn <= 0
        or N_dyn > 4
        or (K & 15) != 0
        or w_dyn.dtype != torch.bfloat16
        or not w_dyn.is_contiguous()
        or torch.cuda.is_current_stream_capturing()
    ):
        return None
    lib = _get_lib()
    if lib is None or not _HAS_V9_DYN:
        return None

    g = torch.Generator(device=w_dyn.device)
    g.manual_seed(7919 + M * 31 + N_dyn)
    if FUSE_DYN_STRICT:
        # Hardened adversarial qualification: constructive Big-Small-Cancel (+2^17 / -2^17 across
        # split boundaries & intra-tile K pairs) + multi-scale Gaussian trials.
        # Any K-accumulation order mismatch diverges by >10 BF16 ULPs (0% false-pass vs v9's ~97-100%!),
        # while a genuinely bit-exact matching order still passes 100.0% (nd == 0)!
        wf = w_dyn.float()
        xs = []
        # 1) Constructive Big-Small-Cancel probes spanning split boundaries & intra-tile K indices
        cancel_pairs = (
            (0, K - 1),
            (0, 15),
            (0, 7),
            (1, 14),
            (0, K // 2),
            (0, K // 4),
            (K // 3, K - 1),
            (128, K - 128),
            (256, K - 256),
            (512, K - 512),
        )
        for p_i, (k_lo, k_hi) in enumerate(cancel_pairs):
            if k_lo < K and k_hi < K and k_lo != k_hi:
                row_i = p_i % N_dyn
                w_lo = float(wf[row_i, k_lo].item())
                w_hi = float(wf[row_i, k_hi].item())
                if abs(w_lo) > 1e-4 and abs(w_hi) > 1e-4:
                    # Exact power-of-2 BF16 cross-cancellation:
                    #   x_lo = +2^s * w_hi (ZERO BF16 mantissa rounding!)
                    #   x_hi = -2^s * w_lo (ZERO BF16 mantissa rounding!)
                    #   => x_lo * w_lo + x_hi * w_hi == 0.0f EXACTLY in IEEE-754!
                    # Scaling 2^s so |x_lo * w_lo| ~ 2^25 wipes out small terms in ANY order that
                    # accumulates x_lo*w_lo before smalls & x_hi*w_hi after, exposing ANY K-order diff!
                    import math
                    prod = abs(w_lo * w_hi)
                    sh_exp = max(16, min(40, int(math.ceil(25.0 - math.log2(prod)))))
                    pow2 = math.ldexp(1.0, sh_exp)
                    x_lo = pow2 * w_hi
                    x_hi = -pow2 * w_lo
                    if abs(x_lo) < 1e18 and abs(x_hi) < 1e18:
                        xt = torch.randn((M, K), device=w_dyn.device, generator=g) * 0.5
                        xt[:, k_lo] = x_lo
                        xt[:, k_hi] = x_hi
                        xs.append(xt.to(torch.bfloat16))
        # 2) Multi-scale Gaussian + skewed-exponent trials (>= 32 total trials)
        for t_i in range(32):
            sc = (1.0, 0.3, 4.0, 0.0625, 16.0, 0.75, 2.5, 0.125)[t_i & 7]
            xt = torch.randn((M, K), device=w_dyn.device, generator=g) * sc
            if (t_i & 3) == 3:
                xt[:, : K // 4] *= 32.0
                xt[:, K // 4 :] *= 0.03125
            xs.append(xt.to(torch.bfloat16))
    else:
        xs = [(torch.randn((M, K), device=w_dyn.device, generator=g) * sc).to(torch.bfloat16) for sc in (1.0, 0.3, 4.0)]
    refs = [torch.nn.functional.linear(x, w_dyn) for x in xs]
    out = torch.empty((M, N_dyn), dtype=torch.bfloat16, device=w_dyn.device)
    st = int(torch.cuda.current_stream(w_dyn.device).cuda_stream)

    cands: list[tuple[int, int]] = []
    for spk in _DYN_SPLIT_CANDS:
        if K % spk == 0 and ((K // spk) & 15) == 0:
            cands.append((0, spk))  # mma.m16n8k16 linear
    for spk in _DYN_TREE_SPLITS:
        if K % spk == 0 and ((K // spk) & 15) == 0:
            cands.append((1, spk))  # mma.m16n8k16 balanced binary tree
    for spk in _DYN_SPLIT_CANDS:
        if K % spk == 0 and ((K // spk) & 15) == 0:
            cands.append((2, spk))  # mma.m16n8k8 linear
    for spk in _DYN_TREE_SPLITS:
        if K % spk == 0 and ((K // spk) & 15) == 0:
            cands.append((3, spk))  # mma.m16n8k8 balanced binary tree

    for mode, spk in cands:
        matched = True
        for x, ref in zip(xs, refs):
            rc = lib.bf11_dyn_probe_v9_fwd(
                x.data_ptr(), w_dyn.data_ptr(), out.data_ptr(), 0,
                M, N_dyn, K, int(x.stride(0)), int(out.stride(0)),
                mode, spk, st,
            )
            if rc != 0 or not bool(torch.equal(out.view(torch.int16), ref.view(torch.int16))):
                matched = False
                break
        if matched:
            res = (mode, spk, True)
            _DYN_QUAL[key] = res
            return res

    # Optional per-row B=1 exact invariance at M > 16 (mirrors INKLING_BF11_C32_EXACT_REF=exact)
    if M > 16 and FUSE_DYN_REF == "exact":
        b1_cfg = _DYN_QUAL.get((N_dyn, K, 8)) or _qualify_dyn_one(w_dyn, 8)
        if b1_cfg is not None:
            mode, spk, _ = b1_cfg
            res = (mode, spk, False)
            _DYN_QUAL[key] = res
            return res

    _DYN_QUAL[key] = None
    return None


def qualify_dyn(w_dyn: torch.Tensor, w_dyn_v: torch.Tensor | None = None) -> int:
    """Pre-qualify global dynamic-tap shapes outside CUDA graph capture."""
    if not FUSE_DYN or w_dyn is None or not w_dyn.is_cuda or torch.cuda.is_current_stream_capturing():
        return 0
    lib = _get_lib()
    if lib is None or not _HAS_V9_DYN:
        return 0
    N_dyn, K = int(w_dyn.shape[0]), int(w_dyn.shape[1])
    sh = (N_dyn, K)
    if sh in _DYN_QUAL_PREWARMED:
        return sum(1 for (n, k, _), v in _DYN_QUAL.items() if n == N_dyn and k == K and v is not None)
    _DYN_QUAL_PREWARMED.add(sh)
    ms_c1 = list(range(1, 17))
    canon = _C32_CANONICAL_MS if C32_EXACT_REF == "torch" else _C32_CANONICAL_MS_EXACT
    ms_c32 = sorted({m for m in (*canon, *_C32_EXTRA_MS) if 17 <= m <= C32_MAX_M}) if C32_ENABLED else []
    n_ok, n_torch = 0, 0
    st = int(torch.cuda.current_stream(w_dyn.device).cuda_stream)
    for m in (*ms_c1, *ms_c32):
        r = _qualify_dyn_one(w_dyn, m)
        if r is not None and FUSE_DYN_STRICT and w_dyn_v is not None and w_dyn_v.shape == w_dyn.shape:
            gv = torch.Generator(device=w_dyn_v.device)
            gv.manual_seed(15485863 + m * 17 + N_dyn)
            out_v = torch.empty((m, N_dyn), dtype=torch.bfloat16, device=w_dyn_v.device)
            mode_c, spk_c, is_tx = r
            if is_tx:
                for vi in range(32):
                    xv = (torch.randn((m, K), device=w_dyn_v.device, generator=gv) * (1.0 if (vi & 1) == 0 else 4.0)).to(torch.bfloat16)
                    rv = torch.nn.functional.linear(xv, w_dyn_v)
                    rc_v = lib.bf11_dyn_probe_v9_fwd(
                        xv.data_ptr(), w_dyn_v.data_ptr(), out_v.data_ptr(), 0,
                        m, N_dyn, K, int(xv.stride(0)), int(out_v.stride(0)),
                        mode_c, spk_c, st,
                    )
                    if rc_v != 0 or not bool(torch.equal(out_v.view(torch.int16), rv.view(torch.int16))):
                        _DYN_QUAL[(N_dyn, K, m)] = None
                        r = None
                        break
        if r is not None:
            n_ok += 1
            if r[2]:
                n_torch += 1
    tot = len(ms_c1) + len(ms_c32)
    print(
        f"[inkling-bf11-dyn-v9] qualified {n_ok}/{tot} M values for dyn [{N_dyn}, {K}] "
        f"(torch_exact={n_torch}/{tot}, ref={FUSE_DYN_REF}, capture_only={int(C32_CAPTURE_ONLY)})",
        flush=True,
    )
    return n_ok


def can_bf11_dual_kv_dyn(
    M: int,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_k_dyn: torch.Tensor | None,
    w_v_dyn: torch.Tensor | None,
) -> bool:
    if (
        not FUSE_DYN
        or not ENABLED
        or not USE_DUAL_KV
        or not V5_ENABLED
        or M <= 0
        or (M > 16 and (not C32_ENABLED or not C32_DUAL_KV or not V5_C32 or M > C32_MAX_M))
        or (M <= 16 and not V5_C1)
        or w_k_dyn is None
        or w_v_dyn is None
        or w_k_dyn.shape != w_v_dyn.shape
        or w_k_dyn.dtype != torch.bfloat16
        or w_v_dyn.dtype != torch.bfloat16
        or not w_k_dyn.is_contiguous()
        or not w_v_dyn.is_contiguous()
    ):
        return False
    lib = _get_lib()
    if lib is None or not _HAS_V9_DYN:
        return False
    ek = get_packed(w_k)
    ev = get_packed(w_v)
    if ek is None or ev is None:
        return False
    if get_v5_sidecar(w_k) is None or get_v5_sidecar(w_v) is None:
        return False
    if (
        ek[8] != ev[8]
        or ek[9] != ev[9]
        or (M <= 16 and ek[9] != 9)
        or w_k.shape != w_v.shape
        or ek[7] != w_k.shape[0]
        or ev[7] != w_v.shape[0]
        or ek[8] not in (1, 4, 5)
    ):
        return False
    N_dyn, K = int(w_k_dyn.shape[0]), int(w_k_dyn.shape[1])
    q = _DYN_QUAL.get((N_dyn, K, M))
    if q is None and (N_dyn, K, M) not in _DYN_QUAL and not torch.cuda.is_current_stream_capturing():
        q = _qualify_dyn_one(w_k_dyn, M)
    if q is None:
        return False
    if M > 16 and C32_CAPTURE_ONLY and not q[2] and not _is_full_decode_capturing():
        return False
    if M > 16:
        N_each = int(w_k.shape[0])
        if not (_qualify_c32_one(w_k, M) and _qualify_c32_one(w_v, M)):
            return False
        if (
            C32_CAPTURE_ONLY
            and not _C32_TORCH_EXACT.get((N_each, K, M), False)
            and not _is_full_decode_capturing()
        ):
            return False
    return True


def bf11_dual_kv_dyn(
    x_k: torch.Tensor,
    x_v: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_k_dyn: torch.Tensor,
    w_v_dyn: torch.Tensor,
    pdl: bool = False,
    trig: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Wave-V9 single-launch fused global K + V + k_dyn + v_dyn (0-ULP qualified, kills 16 cuBLAS calls & dyn_stream!)."""
    if (
        not FUSE_DYN
        or not ENABLED
        or not USE_DUAL_KV
        or x_k.dim() != 2
        or x_v.dim() != 2
        or x_k.shape != x_v.shape
        or x_k.dtype != torch.bfloat16
        or x_v.dtype != torch.bfloat16
        or x_k.stride(1) != 1
        or x_v.stride(1) != 1
        or w_k_dyn.shape != w_v_dyn.shape
        or w_k_dyn.dtype != torch.bfloat16
        or w_v_dyn.dtype != torch.bfloat16
        or not w_k_dyn.is_contiguous()
        or not w_v_dyn.is_contiguous()
    ):
        return None
    M, K = int(x_k.shape[0]), int(x_k.shape[1])
    N_dyn = int(w_k_dyn.shape[0])
    if M <= 0 or N_dyn <= 0 or N_dyn > 4 or int(w_k_dyn.shape[1]) != K:
        return None
    if not can_bf11_dual_kv_dyn(M, w_k, w_v, w_k_dyn, w_v_dyn):
        return None
    q = _DYN_QUAL.get((N_dyn, K, M))
    if q is None:
        return None
    dyn_mode, dyn_split_k, _ = q
    lib = _get_lib()
    ek = get_packed(w_k)
    ev = get_packed(w_v)
    sc0 = get_v5_sidecar(w_k)
    sc1 = get_v5_sidecar(w_v)
    if lib is None or ek is None or ev is None or sc0 is None or sc1 is None:
        return None
    if (
        ek[8] != ev[8]
        or ek[9] != ev[9]
        or (M <= 16 and ek[9] != 9)
        or w_k.shape != w_v.shape
        or ek[7] != w_k.shape[0]
        or ev[7] != w_v.shape[0]
    ):
        return None
    sl0, _, _, ol0, oo0, b0, ks, _, cfg_id, split, n_tiles, _ = ek
    sl1, _, _, ol1, oo1, b1, _, _, _, _, _, _ = ev
    N_each = int(w_k.shape[0])
    if cfg_id not in (1, 4, 5):
        return None

    if M > 16:
        if not (_qualify_c32_one(w_k, M) and _qualify_c32_one(w_v, M)):
            return None
        if (
            C32_CAPTURE_ONLY
            and not _C32_TORCH_EXACT.get((N_each, K, M), False)
            and not _is_full_decode_capturing()
        ):
            return None
        out_k = torch.empty((M, N_each), dtype=x_k.dtype, device=x_k.device)
        out_v = torch.empty((M, N_each), dtype=x_v.dtype, device=x_v.device)
        out_kd = torch.empty((M, N_dyn), dtype=x_k.dtype, device=x_k.device)
        out_vd = torch.empty((M, N_dyn), dtype=x_v.dtype, device=x_v.device)
        nlive_ptr = _rowar_nlive_ptr(M)
        st = int(torch.cuda.current_stream().cuda_stream)
        p_ptr, s_ptr = 0, 0
        if C32_KORDER == 1 and split == 9 and V5_SPLIT_CTA != 1:
            need_p, need_s = _v5_c32_scratch_req(M, N_each, split, dual=True)
            part_b, sem_b = _next_scratch(x_k.device, need_p, need_s)
            p_ptr, s_ptr = part_b.data_ptr(), sem_b.data_ptr()
        fn_dkvd_c32 = lib.bf11_flip_dual_kv_dyn_v12_c32_fwd if (V12_ENABLED and V12_BM48 and _HAS_V12) else lib.bf11_flip_dual_kv_dyn_v9_c32_fwd
        rc = fn_dkvd_c32(
            x_k.data_ptr(), x_v.data_ptr(),
            sl0.data_ptr(), sl1.data_ptr(),
            sc0[0].data_ptr(), sc1[0].data_ptr(),
            sc0[1].data_ptr(), sc1[1].data_ptr(),
            sc0[2].data_ptr(), sc1[2].data_ptr(),
            ol0.data_ptr(), ol1.data_ptr(),
            oo0.data_ptr(), oo1.data_ptr(),
            p_ptr, s_ptr, out_k.data_ptr(), out_v.data_ptr(),
            w_k_dyn.data_ptr(), w_v_dyn.data_ptr(), out_kd.data_ptr(), out_vd.data_ptr(),
            nlive_ptr,
            M, N_each, N_dyn, K, int(x_k.stride(0)), int(x_v.stride(0)), ks, split,
            int(out_k.stride(0)), int(out_v.stride(0)), int(out_kd.stride(0)), int(out_vd.stride(0)),
            cfg_id, C32_BM, V5_W_RUN, V5_SPLIT_CTA, C32_KORDER, b0, b1,
            dyn_mode, dyn_split_k, 1 if pdl else 0, 1 if trig else 0, st,
        )
        if rc != 0:
            raise RuntimeError(f"bf11_flip_dual_kv_dyn_v9_c32_fwd failed with cudaError={rc}")
        return out_k, out_v, out_kd, out_vd

    out_k = torch.empty((M, N_each), dtype=x_k.dtype, device=x_k.device)
    out_v = torch.empty((M, N_each), dtype=x_v.dtype, device=x_v.device)
    out_kd = torch.empty((M, N_dyn), dtype=x_k.dtype, device=x_k.device)
    out_vd = torch.empty((M, N_dyn), dtype=x_v.dtype, device=x_v.device)
    n_pad_kv = ((N_each + 31) // 32) * 32
    sem_kv = max(2 * ((N_each + 7) // 8), 2 * n_tiles * (2 if cfg_id == 4 else 4))
    part, sem = _next_scratch(x_k.device, 2 * split * M * n_pad_kv, sem_kv)
    nlive_ptr = _rowar_nlive_ptr(M)
    st = int(torch.cuda.current_stream().cuda_stream)
    fn_dkvd_c1 = lib.bf11_flip_dual_kv_dyn_v12_c1_fwd if (V12_ENABLED and _HAS_V12) else lib.bf11_flip_dual_kv_dyn_v9_c1_fwd
    rc = fn_dkvd_c1(
        x_k.data_ptr(), x_v.data_ptr(),
        sl0.data_ptr(), sl1.data_ptr(),
        sc0[0].data_ptr(), sc1[0].data_ptr(),
        sc0[1].data_ptr(), sc1[1].data_ptr(),
        sc0[2].data_ptr(), sc1[2].data_ptr(),
        ol0.data_ptr(), ol1.data_ptr(),
        oo0.data_ptr(), oo1.data_ptr(),
        part.data_ptr(), sem.data_ptr(), out_k.data_ptr(), out_v.data_ptr(),
        w_k_dyn.data_ptr(), w_v_dyn.data_ptr(), out_kd.data_ptr(), out_vd.data_ptr(),
        nlive_ptr,
        M, N_each, N_dyn, K, int(x_k.stride(0)), int(x_v.stride(0)), ks,
        int(out_k.stride(0)), int(out_v.stride(0)), int(out_kd.stride(0)), int(out_vd.stride(0)),
        cfg_id, V5_W_RUN, split, b0, b1,
        dyn_mode, dyn_split_k, 1 if pdl else 0, 1 if trig else 0, st,
    )
    if rc != 0:
        raise RuntimeError(f"bf11_flip_dual_kv_dyn_v9_c1_fwd failed with cudaError={rc}")
    return out_k, out_v, out_kd, out_vd
