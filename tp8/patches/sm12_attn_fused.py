# SPDX-License-Identifier: Apache-2.0
"""SM121a (GB10) 0-ULP Bit-Exact Fused Target Decode Attention Wrapper.

Wraps `fused_decode_attn_exact_sm121.cu`:
  1. `attn_prologue_kvq_exact_fwd`: 24-CTA 1-wave K-conv + V-conv + Q-clip ONLY
     (eliminates all 16 RelProj CTAs & 2.09 MB/layer sheared `rel_logits` round-trip!)
  2. `fused_decode_attn_otf_mgs_fwd`:
     - Pre-`gdc_wait()` async past-KV prefetch for splits `0..S-2`
     - On-the-fly exact `_l2_clip(R)` + `r_clip @ rel_proj_T` hidden behind `cp.async.cg` of K
     - Double-buffered XOR-swizzled SMEM + `ldmatrix.x4` / `ldmatrix.x4.trans`
     - Cooperative last-split `MergeGateSink` with persistent L2-hot ping-pong `OP`/`LP` scratch
       (eliminates 48 MGS launches/step AND 11.0 MB/step dirty per-layer `OP` allocations!)

Recommended paired L2PF env retuning (since fused attn cuts attn duration ~21 -> ~11 us/layer):
  INKLING_L2PF_PACE_ATTN_GBPS=200
  INKLING_L2PF_MAX_ATTN_US=10
  INKLING_L2PF_WO_MB=2.5
"""
from __future__ import annotations

import ctypes
import math
import os
import threading

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

ENABLED = os.environ.get("INKLING_FUSED_DECODE_ATTN", "0").strip().lower() in ("1", "true", "on", "yes")
ENABLED_C32 = os.environ.get("INKLING_FUSED_DECODE_ATTN_C32", "0").strip().lower() in ("1", "true", "on", "yes")
V7_ENABLED = os.environ.get("INKLING_FUSED_ATTN_V7", "0").strip().lower() in ("1", "true", "on", "yes")
V8_ENABLED = os.environ.get("INKLING_FUSED_ATTN_V8", "0").strip().lower() in ("1", "true", "on", "yes")
V9_ENABLED = os.environ.get("INKLING_FUSED_ATTN_V9", "0").strip().lower() in ("1", "true", "on", "yes")
KV_BF12_ENABLED = os.environ.get("INKLING_KV_BF12", "0").strip().lower() in ("1", "true", "on", "yes")
# Wave-V11 opt-in knobs (strictly default-OFF so stage10a/10b stay 100% invariant!):
#   INKLING_FUSED_ATTN_V11_S48=1         : S=48/3 ladder + B=1 Node-2 s_long=48 (Node-1 2K S=16 untouched!)
#   INKLING_FUSED_ATTN_V11_C1632_GLOBAL=1: extend paired L2-threshold FUSE_MT=1 to B=16..32 & G>=2
#   INKLING_FUSED_ATTN_V11_TARGET_CTAS   : 96 (default, 2 CTAs/SM) or 48 (1 full wave)
V11_S48 = os.environ.get("INKLING_FUSED_ATTN_V11_S48", "0").strip().lower() in ("1", "true", "on", "yes")
V11_C1632_GLOBAL = os.environ.get("INKLING_FUSED_ATTN_V11_C1632_GLOBAL", "0").strip().lower() in ("1", "true", "on", "yes")
_V11_TARGET_CTAS_RAW = os.environ.get("INKLING_FUSED_ATTN_V11_TARGET_CTAS", "").strip()
_V11_TARGET_CTAS = max(16, min(96, int(_V11_TARGET_CTAS_RAW))) if _V11_TARGET_CTAS_RAW else 96
_V11_C1_LONG_SPLITS = int(os.environ.get("INKLING_FUSED_ATTN_V11_C1_LONG_SPLITS", "48"))
KV_BF12_HALF_BYTES = 50688
KV_BF12_PAGE_BYTES = 101376
KV_BF12_EMERG_CAP = max(256, int(os.environ.get("INKLING_KV_BF12_EMERG_CAP", "4096")))
MAX_TOKENS_C32 = int(os.environ.get("INKLING_FUSED_ATTN_C32_MAX_T", "320"))
_LIB_PATH = os.environ.get("INKLING_FUSED_ATTN_LIB", "/rdma/libfused_decode_attn_exact_sm121.so")
_V7_LIB_PATH = os.environ.get("INKLING_FUSED_ATTN_V7_LIB", "/rdma/libfused_decode_attn_exact_sm121_v7.so")
_V8_LIB_PATH = os.environ.get("INKLING_FUSED_ATTN_V8_LIB", "/rdma/libfused_decode_attn_exact_sm121_v8.so")
_V9_LIB_PATH = os.environ.get("INKLING_FUSED_ATTN_V9_LIB", "/rdma/libfused_decode_attn_exact_sm121_v9.so")
_V10_LIB_PATH = os.environ.get("INKLING_FUSED_ATTN_V10_LIB", "/rdma/libfused_decode_attn_exact_sm121_v10.so")
_V11_LIB_PATH = os.environ.get("INKLING_FUSED_ATTN_V11_LIB", "/rdma/libfused_decode_attn_exact_sm121_v11.so")
_KV_BF12_PAGED_LIB_PATH = os.environ.get("INKLING_KV_BF12_PAGED_LIB", "/rdma/libkv_bf12_paged_v10.so")
_TREE_LIB_PATH = os.environ.get("INKLING_FUSED_ATTN_TREE_LIB", "/rdma/libfused_decode_attn_tree_exact_sm121.so")
_C32_LIB_PATH = os.environ.get("INKLING_FUSED_ATTN_C32_LIB", "/rdma/libfused_decode_attn_exact_sm121_c32.so")
_C32_FUSE_MT = os.environ.get("INKLING_FUSED_ATTN_C32_FUSE_MT", "auto").strip().lower()
_C32_EVICT_FIRST = os.environ.get("INKLING_FUSED_ATTN_C32_EVICT_FIRST", "1").strip().lower() not in ("0", "", "off", "false", "no")
_C32_SPLITS = int(os.environ.get("INKLING_FUSED_ATTN_C32_SPLITS", "1"))
_C28_SPLITS = os.environ.get("INKLING_FUSED_ATTN_C28_SPLITS", "0").strip().lower() in ("1", "true", "on", "yes")
_C28_SWA_B2_SPLITS = int(os.environ.get("INKLING_FUSED_ATTN_C28_SWA_B2_SPLITS", "8"))
_C28_SWA_B34_SPLITS = int(os.environ.get("INKLING_FUSED_ATTN_C28_SWA_B34_SPLITS", "1"))
_C28_SWA_B58_SPLITS = int(os.environ.get("INKLING_FUSED_ATTN_C28_SWA_B58_SPLITS", "1"))
_C28_GLOBAL_TARGET_CTAS = int(os.environ.get("INKLING_FUSED_ATTN_C28_GLOBAL_TARGET_CTAS", "96"))
_C28_PDL_WO = os.environ.get("INKLING_FUSED_ATTN_C28_PDL_WO", "0").strip().lower() in ("1", "true", "on", "yes")
_V8_C28_FUSE_MT = os.environ.get("INKLING_FUSED_ATTN_V8_C28_FUSE_MT", "0").strip().lower() in ("1", "true", "on", "yes")
_V8_C1_FUSE_MT = os.environ.get("INKLING_FUSED_ATTN_V8_C1_FUSE_MT", "0").strip().lower() in ("1", "true", "on", "yes")
_V8_C1_SWA_SPLITS = int(os.environ.get("INKLING_FUSED_ATTN_V8_C1_SWA_SPLITS", "8"))
_V8_C1_GLOBAL_SPLITS = int(os.environ.get("INKLING_FUSED_ATTN_V8_C1_GLOBAL_SPLITS", "16"))
_V8_C28_SWA_FUSE_MT = os.environ.get("INKLING_FUSED_ATTN_V8_C28_SWA_FUSE_MT", "1").strip().lower() not in ("0", "", "off", "false", "no")
_V8_SWA_SPLITS = os.environ.get("INKLING_FUSED_ATTN_V8_SWA_SPLITS", "auto").strip().lower()
_V8_SWA_B38_ONLY = os.environ.get("INKLING_FUSED_ATTN_V8_SWA_B38_ONLY", "0").strip().lower() not in ("0", "", "off", "false", "no")
_V8_SWA_MIN_B = max(2, min(9, int(os.environ.get("INKLING_FUSED_ATTN_V8_SWA_MIN_B", "3"))))
_V8_GLOBAL_TARGET_CTAS = int(os.environ.get("INKLING_FUSED_ATTN_V8_GLOBAL_TARGET_CTAS", "48"))
# Wave-V9 CUDAGraph-safe GPU-side L2-threshold dynamic FUSE_MT for Global layers (window < 0):
#   - batch_kv_toks <= THRESH (default 24,576 tok = 12 MiB = 50% of 24 MiB L2): Node 1 (FUSE_MT=0, 0% 2K regression)
#   - batch_kv_toks >  THRESH (16K..1M > L2): Node 2 (FUSE_MT=1 + L2::evict_first, 3.0x KV DRAM cut!)
#   - B=1 default S_long=16 keeps literal 0-ULP parity with S=16 Triton across 16..1,048,576!
_V9_GLOBAL_L2_THRESH_TOKS = int(os.environ.get("INKLING_FUSED_ATTN_V9_L2_THRESH_TOKS", "24576"))
_V9_C1_LONG_SPLITS = int(os.environ.get("INKLING_FUSED_ATTN_V9_C1_LONG_SPLITS", "16"))
_V9_GLOBAL_LONG_TARGET_CTAS = int(os.environ.get("INKLING_FUSED_ATTN_V9_GLOBAL_LONG_TARGET_CTAS", "96"))
_V9_GLOBAL_LONG_EVICT_FIRST = os.environ.get(
    "INKLING_FUSED_ATTN_V9_GLOBAL_EVICT_FIRST", "1"
).strip().lower() not in ("0", "", "off", "false", "no")
_V9_C915_GLOBAL = os.environ.get(
    "INKLING_FUSED_ATTN_V9_C915_GLOBAL", "1"
).strip().lower() not in ("0", "", "off", "false", "no")
_V9_SHORT_SPLIT_LADDER = (1, 2, 4, 6, 8, 12, 16)
_V9_LONG_SPLIT_LADDER = (2, 4, 6, 8, 12, 16, 24, 32)
_V11_SHORT_SPLIT_LADDER = (1, 2, 3, 4, 6, 8, 12, 16)
_V11_LONG_SPLIT_LADDER = (2, 3, 4, 6, 8, 12, 16, 24, 32, 48)
_V8_LOCAL_SPLIT_LADDER = (1, 2, 4, 6, 8)
_V8_GLOBAL_SPLIT_LADDER = (1, 2, 4, 6, 8, 12, 16, 24, 32)
_TREE_MODE = int(os.environ.get("INKLING_FUSED_ATTN_TREE_MODE", "0"))  # 0, 1, or 2 (auto-calibrated for nd==0)
_USE_PDL = os.environ.get("INKLING_FUSED_ATTN_PDL", "1").strip().lower() not in ("0", "", "off", "false", "no")
_TRIG_WO = os.environ.get("INKLING_FUSED_ATTN_TRIG_WO", "1").strip().lower() not in ("0", "", "off", "false", "no")
# Wave-V11 opt-in K=11/15 extended SAM decode attention support (default OFF -> 100% V10 invariant!):
# Allows eff_max_q * G <= 96 (qlen <= 16, mtiles <= 6) for small decode batches (num_reqs <= 2, T <= 32)
# routing cleanly via dispatch_mgs_impl on both BF16-KV and KV_BF12=1 without CUDAGraph fallback crash.
_SAM_EXT_ATTN_FIX = (
    os.environ.get("INKLING_SAM_EXT_ATTN_FIX", "0").strip().lower() in ("1", "true", "on", "yes")
    or os.environ.get("INKLING_SAM_EXT_K15", "0").strip().lower() in ("1", "true", "on", "yes")
)
_LOG2E = 1.4426950408889634

_LIB = None
_TREE_LIB = None
_TREE_LIB_TRIED = False
_HAS_C32_SYM = False
_HAS_V9_SYM = False
_IS_V7_LIB = False
_IS_V8_LIB = False
_IS_V9_LIB = False
_IS_V10_LIB = False
_IS_V11_LIB = False
_HAS_V10_BF12_SYM = False
_KV_BF12_PAGED_LIB = None
_KV_BF12_PAGED_LIB_TRIED = False
_LIB_LOCK = threading.RLock()

# 2x ping-pong persistent L2-hot scratch per device:
#   OP:  [16, 16, 6, 128] fp32 = 768 KiB/slot (1.5 MiB total, 100% L2-resident!)
#   LP:  [16, 16, 6]      fp32 =   6 KiB/slot
#   SEM: [64]            int32 = 256 B/slot (self-resetting on last split!)
_SCRATCH: dict[int, dict] = {}
_SCRATCH_V11: dict[int, dict] = {}
_REL_PROJ_T_CACHE: dict[int, object] = {}


def _load_lib():
    global _LIB, _HAS_C32_SYM, _HAS_V9_SYM, _IS_V7_LIB, _IS_V8_LIB, _IS_V9_LIB, _IS_V10_LIB, _IS_V11_LIB, _HAS_V10_BF12_SYM
    if _LIB is not None:
        return _LIB
    with _LIB_LOCK:
        if _LIB is None:
            want_v11 = V11_S48 or V11_C1632_GLOBAL
            want_v10 = want_v11 or KV_BF12_ENABLED
            want_v9 = want_v10 or V9_ENABLED
            want_v8 = want_v9 or V8_ENABLED or _V8_C28_FUSE_MT or _V8_C1_FUSE_MT
            if os.environ.get("INKLING_FUSED_ATTN_LIB"):
                if want_v11:
                    cands = [_LIB_PATH, _V11_LIB_PATH, _V10_LIB_PATH, _V9_LIB_PATH, _V8_LIB_PATH, _V7_LIB_PATH, _C32_LIB_PATH]
                elif want_v10:
                    cands = [_LIB_PATH, _V10_LIB_PATH, _V9_LIB_PATH, _V8_LIB_PATH, _V7_LIB_PATH, _C32_LIB_PATH]
                elif want_v9:
                    cands = [_LIB_PATH, _V9_LIB_PATH, _V8_LIB_PATH, _V7_LIB_PATH, _C32_LIB_PATH]
                elif want_v8:
                    cands = [_LIB_PATH, _V8_LIB_PATH, _V7_LIB_PATH, _C32_LIB_PATH]
                elif V7_ENABLED:
                    cands = [_LIB_PATH, _V7_LIB_PATH, _C32_LIB_PATH]
                else:
                    cands = [_LIB_PATH, _C32_LIB_PATH]
            elif want_v11:
                cands = [_V11_LIB_PATH, _V10_LIB_PATH, _V9_LIB_PATH, _V8_LIB_PATH, _V7_LIB_PATH, _C32_LIB_PATH, _LIB_PATH]
            elif want_v10:
                cands = [_V10_LIB_PATH, _V9_LIB_PATH, _V8_LIB_PATH, _V7_LIB_PATH, _C32_LIB_PATH, _LIB_PATH]
            elif want_v9:
                cands = [_V9_LIB_PATH, _V8_LIB_PATH, _V7_LIB_PATH, _C32_LIB_PATH, _LIB_PATH]
            elif want_v8:
                cands = [_V8_LIB_PATH, _V7_LIB_PATH, _C32_LIB_PATH, _LIB_PATH]
            elif V7_ENABLED:
                cands = [_V7_LIB_PATH, _C32_LIB_PATH, _LIB_PATH]
            elif ENABLED_C32:
                cands = [_C32_LIB_PATH, _LIB_PATH]
            else:
                cands = [_LIB_PATH]
            path = next((p for p in cands if os.path.exists(p)), None)
            if path is None:
                return None
            lib = ctypes.CDLL(path)
            lib.attn_prologue_kvq_exact_fwd.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
                ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
                ctypes.c_int64, ctypes.c_int64, ctypes.c_int,
                ctypes.c_int64, ctypes.c_int, ctypes.c_int, ctypes.c_int64,
                ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_uint64,
            ]
            lib.attn_prologue_kvq_exact_fwd.restype = ctypes.c_int

            lib.fused_decode_attn_otf_mgs_fwd.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
                ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
                ctypes.c_int64, ctypes.c_int64,
                ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
                ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
                ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
                ctypes.c_float, ctypes.c_float, ctypes.c_float,
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_int,
                ctypes.c_uint64,
            ]
            lib.fused_decode_attn_otf_mgs_fwd.restype = ctypes.c_int
            if hasattr(lib, "fused_decode_attn_otf_mgs_c32_fwd"):
                lib.fused_decode_attn_otf_mgs_c32_fwd.argtypes = (
                    list(lib.fused_decode_attn_otf_mgs_fwd.argtypes[:-1])
                    + [ctypes.c_int, ctypes.c_int, ctypes.c_uint64]
                )
                lib.fused_decode_attn_otf_mgs_c32_fwd.restype = ctypes.c_int
                _HAS_C32_SYM = True
            if hasattr(lib, "fused_decode_attn_v7_version"):
                lib.fused_decode_attn_v7_version.argtypes = []
                lib.fused_decode_attn_v7_version.restype = ctypes.c_int
                v7_ver = int(lib.fused_decode_attn_v7_version())
                _IS_V7_LIB = v7_ver >= 7
                _IS_V8_LIB = v7_ver >= 8
            if hasattr(lib, "fused_decode_attn_v8_version"):
                lib.fused_decode_attn_v8_version.argtypes = []
                lib.fused_decode_attn_v8_version.restype = ctypes.c_int
                _IS_V8_LIB = int(lib.fused_decode_attn_v8_version()) >= 8
            if hasattr(lib, "fused_decode_attn_otf_mgs_v9_fwd"):
                lib.fused_decode_attn_otf_mgs_v9_fwd.argtypes = (
                    list(lib.fused_decode_attn_otf_mgs_fwd.argtypes[:-1])
                    + [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint64]
                )
                lib.fused_decode_attn_otf_mgs_v9_fwd.restype = ctypes.c_int
                _HAS_V9_SYM = True
            if hasattr(lib, "fused_decode_attn_v9_version"):
                lib.fused_decode_attn_v9_version.argtypes = []
                lib.fused_decode_attn_v9_version.restype = ctypes.c_int
                _IS_V9_LIB = int(lib.fused_decode_attn_v9_version()) >= 9
            if hasattr(lib, "fused_decode_attn_v10_version"):
                lib.fused_decode_attn_v10_version.argtypes = []
                lib.fused_decode_attn_v10_version.restype = ctypes.c_int
                v10_ver = int(lib.fused_decode_attn_v10_version())
                _IS_V10_LIB = v10_ver >= 10
                _IS_V11_LIB = v10_ver >= 11
            if hasattr(lib, "fused_decode_attn_v11_version"):
                lib.fused_decode_attn_v11_version.argtypes = []
                lib.fused_decode_attn_v11_version.restype = ctypes.c_int
                _IS_V11_LIB = int(lib.fused_decode_attn_v11_version()) >= 11
            if (
                hasattr(lib, "attn_prologue_kvq_bf12_exact_v10_fwd")
                and hasattr(lib, "fused_decode_attn_otf_mgs_bf12_v10_fwd")
                and hasattr(lib, "fused_decode_attn_otf_mgs_c32_bf12_v10_fwd")
                and hasattr(lib, "fused_decode_attn_otf_mgs_v10_bf12_paired_fwd")
            ):
                lib.attn_prologue_kvq_bf12_exact_v10_fwd.argtypes = (
                    list(lib.attn_prologue_kvq_exact_fwd.argtypes)
                    + [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32]
                )
                lib.attn_prologue_kvq_bf12_exact_v10_fwd.restype = ctypes.c_int
                lib.fused_decode_attn_otf_mgs_bf12_v10_fwd.argtypes = (
                    list(lib.fused_decode_attn_otf_mgs_fwd.argtypes)
                    + [ctypes.c_void_p, ctypes.c_void_p]
                )
                lib.fused_decode_attn_otf_mgs_bf12_v10_fwd.restype = ctypes.c_int
                lib.fused_decode_attn_otf_mgs_c32_bf12_v10_fwd.argtypes = (
                    list(lib.fused_decode_attn_otf_mgs_c32_fwd.argtypes)
                    + [ctypes.c_void_p, ctypes.c_void_p]
                )
                lib.fused_decode_attn_otf_mgs_c32_bf12_v10_fwd.restype = ctypes.c_int
                lib.fused_decode_attn_otf_mgs_v10_bf12_paired_fwd.argtypes = (
                    list(lib.fused_decode_attn_otf_mgs_v9_fwd.argtypes)
                    + [ctypes.c_void_p, ctypes.c_void_p]
                )
                lib.fused_decode_attn_otf_mgs_v10_bf12_paired_fwd.restype = ctypes.c_int
                _HAS_V10_BF12_SYM = True
            if hasattr(lib, "fused_decode_attn_v11_init_smem"):
                lib.fused_decode_attn_v11_init_smem.argtypes = []
                lib.fused_decode_attn_v11_init_smem.restype = ctypes.c_int
                if torch is None or not torch.cuda.is_current_stream_capturing():
                    rc_smem = int(lib.fused_decode_attn_v11_init_smem())
                    if rc_smem != 0:
                        raise RuntimeError(f"fused_decode_attn_v11_init_smem failed: cudaError={rc_smem}")
            elif hasattr(lib, "fused_decode_attn_v10_init_smem"):
                lib.fused_decode_attn_v10_init_smem.argtypes = []
                lib.fused_decode_attn_v10_init_smem.restype = ctypes.c_int
                if torch is None or not torch.cuda.is_current_stream_capturing():
                    rc_smem = int(lib.fused_decode_attn_v10_init_smem())
                    if rc_smem != 0:
                        raise RuntimeError(f"fused_decode_attn_v10_init_smem failed: cudaError={rc_smem}")
            elif hasattr(lib, "fused_decode_attn_v9_init_smem"):
                lib.fused_decode_attn_v9_init_smem.argtypes = []
                lib.fused_decode_attn_v9_init_smem.restype = ctypes.c_int
                if torch is None or not torch.cuda.is_current_stream_capturing():
                    rc_smem = int(lib.fused_decode_attn_v9_init_smem())
                    if rc_smem != 0:
                        raise RuntimeError(f"fused_decode_attn_v9_init_smem failed: cudaError={rc_smem}")
            _LIB = lib
    return _LIB




def get_kv_bf12_paged_lib():
    """Load `libkv_bf12_paged_v10.so` (prefill store, coalesced block/range unpack, block reset)."""
    global _KV_BF12_PAGED_LIB, _KV_BF12_PAGED_LIB_TRIED
    if _KV_BF12_PAGED_LIB_TRIED:
        return _KV_BF12_PAGED_LIB
    with _LIB_LOCK:
        if not _KV_BF12_PAGED_LIB_TRIED:
            _KV_BF12_PAGED_LIB_TRIED = True
            if os.path.exists(_KV_BF12_PAGED_LIB_PATH):
                lib = ctypes.CDLL(_KV_BF12_PAGED_LIB_PATH)
                lib.kv_bf12_version.argtypes = []
                lib.kv_bf12_version.restype = ctypes.c_int
                lib.kv_bf12_reset_blocks_v10_fwd.argtypes = [
                    ctypes.c_uint64, ctypes.c_int64, ctypes.c_int64,
                    ctypes.c_uint64, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                    ctypes.c_uint64,
                ]
                lib.kv_bf12_reset_blocks_v10_fwd.restype = ctypes.c_int
                lib.kv_bf12_store_prefill_v10_fwd.argtypes = [
                    ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64,
                    ctypes.c_int64, ctypes.c_int64,
                    ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64,
                    ctypes.c_uint32, ctypes.c_int, ctypes.c_int,
                    ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
                    ctypes.c_int32, ctypes.c_uint64,
                ]
                lib.kv_bf12_store_prefill_v10_fwd.restype = ctypes.c_int
                lib.kv_bf12_store_prefill_devdiscard_v10_fwd.argtypes = [
                    ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64,
                    ctypes.c_int64, ctypes.c_int64,
                    ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64,
                    ctypes.c_uint32, ctypes.c_int, ctypes.c_int,
                    ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
                    ctypes.c_int32, ctypes.c_uint64, ctypes.c_uint64,
                ]
                lib.kv_bf12_store_prefill_devdiscard_v10_fwd.restype = ctypes.c_int
                lib.kv_bf12_unpack_blocks_v10_fwd.argtypes = [
                    ctypes.c_uint64, ctypes.c_int64, ctypes.c_int64,
                    ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64,
                    ctypes.c_uint64, ctypes.c_uint64,
                    ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
                    ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
                    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                    ctypes.c_uint64,
                ]
                lib.kv_bf12_unpack_blocks_v10_fwd.restype = ctypes.c_int
                lib.kv_bf12_unpack_range_v10_fwd.argtypes = [
                    ctypes.c_uint64, ctypes.c_int64, ctypes.c_int64,
                    ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64,
                    ctypes.c_int, ctypes.c_int, ctypes.c_int,
                    ctypes.c_uint64, ctypes.c_uint64,
                    ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
                    ctypes.c_int, ctypes.c_int, ctypes.c_int,
                    ctypes.c_uint64,
                ]
                lib.kv_bf12_unpack_range_v10_fwd.restype = ctypes.c_int
                lib.kv_bf12_unpack_bt_range_v10_fwd.argtypes = [
                    ctypes.c_uint64, ctypes.c_int64, ctypes.c_int64,
                    ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64,
                    ctypes.c_int64, ctypes.c_int, ctypes.c_int,
                    ctypes.c_int, ctypes.c_int,
                    ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int,
                    ctypes.c_uint64, ctypes.c_uint64,
                    ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
                    ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
                    ctypes.c_int, ctypes.c_int, ctypes.c_int,
                    ctypes.c_uint64,
                ]
                lib.kv_bf12_unpack_bt_range_v10_fwd.restype = ctypes.c_int
                _KV_BF12_PAGED_LIB = lib
    if _KV_BF12_PAGED_LIB is not None and not _KV_BF12_RESOLVER_REGISTERED:
        ensure_kv_bf12_prefill_resolver_registered()
    return _KV_BF12_PAGED_LIB


class KvBf12EmergPool:
    """Per-layer Tier-3 Emergency Raw BF16 fallback pool (2.0 MiB/layer @ cap=4096).
    Allocated once outside CUDA graph capture; holds separate K and V raw `[cap, 128]`
    BF16 buffers and atomic `[1]` uint32 allocation counters.
    """

    def __init__(self, device, capacity: int = KV_BF12_EMERG_CAP):
        assert torch is not None
        self.capacity = int(capacity)
        self.k_raw = torch.zeros((self.capacity, 128), dtype=torch.bfloat16, device=device)
        self.v_raw = torch.zeros((self.capacity, 128), dtype=torch.bfloat16, device=device)
        self.k_used = torch.zeros((1,), dtype=torch.int32, device=device)
        self.v_used = torch.zeros((1,), dtype=torch.int32, device=device)

    def reset_counters(self) -> None:
        self.k_used.zero_()
        self.v_used.zero_()

    def pointers(self) -> tuple[int, int, int, int, int]:
        return (
            int(self.k_raw.data_ptr()),
            int(self.k_used.data_ptr()),
            int(self.v_raw.data_ptr()),
            int(self.v_used.data_ptr()),
            self.capacity,
        )


def is_kv_bf12_cache(kv_cache) -> bool:
    """Return True if `kv_cache` is a 99 KiB KV-BF12 paged view (`shape[-1]==198` bf16 or `396` u8)."""
    if torch is None or kv_cache is None or kv_cache.numel() == 0 or kv_cache.ndim != 4:
        return False
    if kv_cache.dtype == torch.bfloat16 and int(kv_cache.shape[-1]) == 198 and int(kv_cache.shape[2]) == 256:
        return True
    if kv_cache.dtype == torch.uint8 and int(kv_cache.shape[-1]) == 396 and int(kv_cache.shape[2]) == 256:
        return True
    return False


def _kv_bf12_geom(kv_cache) -> tuple[int, int, int, int]:
    """Return `(base_ptr, blk_stride_bytes, head_stride_bytes, Hkv)` for a KV-BF12 cache view."""
    esize = int(kv_cache.element_size())
    return (
        int(kv_cache.data_ptr()),
        int(kv_cache.stride(0)) * esize,
        int(kv_cache.stride(1)) * esize,
        int(kv_cache.shape[1]),
    )


def kv_bf12_reset_blocks(kv_cache, block_ids, *, zero_meta: bool = False) -> None:
    """Reset `blk.ovf_used` (and optionally `meta[256]`) on GPU for allocated/recycled block IDs."""
    if torch is None or block_ids.numel() == 0:
        return
    lib = get_kv_bf12_paged_lib()
    if lib is None:
        raise RuntimeError("libkv_bf12_paged_v10.so required for kv_bf12_reset_blocks")
    ids = block_ids.to(dtype=torch.int32, device=kv_cache.device).reshape(-1).contiguous()
    base_ptr, blk_sb, head_sb, hkv = _kv_bf12_geom(kv_cache)
    stream = int(torch.cuda.current_stream(kv_cache.device).cuda_stream)
    rc = lib.kv_bf12_reset_blocks_v10_fwd(
        base_ptr, blk_sb, head_sb, int(ids.data_ptr()),
        int(ids.numel()), hkv, 1 if zero_meta else 0, stream,
    )
    if rc != 0:
        raise RuntimeError(f"kv_bf12_reset_blocks_v10_fwd failed: cudaError={rc}")


def kv_bf12_store_prefill(
    k_in,
    v_in,
    slot_mapping,
    kv_cache,
    emerg_pool: KvBf12EmergPool,
    *,
    discard_blk: int = -1,
    discard_blk_dev=None,
) -> None:
    """Encode BF16 `[T, Hkv, 128]` K/V tensors into paged KV-BF12 `kv_cache` (0-ULP exact).
    When `discard_blk_dev` (1-element int64 GPU tensor) is provided, reads `*discard_blk_dev`
    on GPU with ZERO host syncs and skips `_sparse_discard_block` cleanly (0 atomics, 0 emerg)."""
    assert torch is not None
    T = int(k_in.shape[0])
    if T <= 0:
        return
    lib = get_kv_bf12_paged_lib()
    if lib is None:
        raise RuntimeError("libkv_bf12_paged_v10.so required for kv_bf12_store_prefill")
    base_ptr, blk_sb, head_sb, hkv = _kv_bf12_geom(kv_cache)
    k_eraw, k_eused, v_eraw, v_eused, ecap = emerg_pool.pointers()
    sm = slot_mapping[:T].to(dtype=torch.int64).contiguous()
    stream = int(torch.cuda.current_stream(kv_cache.device).cuda_stream)
    if discard_blk_dev is not None:
        rc = lib.kv_bf12_store_prefill_devdiscard_v10_fwd(
            int(k_in.data_ptr()), int(v_in.data_ptr()), int(sm.data_ptr()), base_ptr,
            blk_sb, head_sb, k_eraw, k_eused, v_eraw, v_eused, ecap,
            T, hkv,
            int(k_in.stride(0)), int(k_in.stride(1)),
            int(v_in.stride(0)), int(v_in.stride(1)),
            int(discard_blk), int(discard_blk_dev.data_ptr()), stream,
        )
        if rc != 0:
            raise RuntimeError(f"kv_bf12_store_prefill_devdiscard_v10_fwd failed: cudaError={rc}")
    else:
        rc = lib.kv_bf12_store_prefill_v10_fwd(
            int(k_in.data_ptr()), int(v_in.data_ptr()), int(sm.data_ptr()), base_ptr,
            blk_sb, head_sb, k_eraw, k_eused, v_eraw, v_eused, ecap,
            T, hkv,
            int(k_in.stride(0)), int(k_in.stride(1)),
            int(v_in.stride(0)), int(v_in.stride(1)),
            int(discard_blk), stream,
        )
        if rc != 0:
            raise RuntimeError(f"kv_bf12_store_prefill_v10_fwd failed: cudaError={rc}")


def kv_bf12_unpack_blocks(
    kv_cache, emerg_pool: KvBf12EmergPool, block_ids, *, unpack_k: bool = True, unpack_v: bool = True
):
    """Unpack selected KV-BF12 physical blocks into `[N, 256, Hkv, 128]` BF16 K and/or V tensors."""
    assert torch is not None
    lib = get_kv_bf12_paged_lib()
    if lib is None:
        raise RuntimeError("libkv_bf12_paged_v10.so required for kv_bf12_unpack_blocks")
    ids = block_ids.to(dtype=torch.int32, device=kv_cache.device).reshape(-1).contiguous()
    n_blks = int(ids.numel())
    base_ptr, blk_sb, head_sb, hkv = _kv_bf12_geom(kv_cache)
    k_eraw, _, v_eraw, _, _ = emerg_pool.pointers()
    k_out = torch.empty((n_blks, 256, hkv, 128), dtype=torch.bfloat16, device=kv_cache.device) if unpack_k else None
    v_out = torch.empty((n_blks, 256, hkv, 128), dtype=torch.bfloat16, device=kv_cache.device) if unpack_v else None
    if n_blks > 0 and (unpack_k or unpack_v):
        stream = int(torch.cuda.current_stream(kv_cache.device).cuda_stream)
        rc = lib.kv_bf12_unpack_blocks_v10_fwd(
            base_ptr, blk_sb, head_sb, k_eraw, v_eraw, int(ids.data_ptr()),
            int(k_out.data_ptr()) if k_out is not None else 0,
            int(v_out.data_ptr()) if v_out is not None else 0,
            int(k_out.stride(0)) if k_out is not None else 0,
            int(k_out.stride(1)) if k_out is not None else 0,
            int(k_out.stride(2)) if k_out is not None else 0,
            int(v_out.stride(0)) if v_out is not None else 0,
            int(v_out.stride(1)) if v_out is not None else 0,
            int(v_out.stride(2)) if v_out is not None else 0,
            n_blks, hkv, 1 if unpack_k else 0, 1 if unpack_v else 0, stream,
        )
        if rc != 0:
            raise RuntimeError(f"kv_bf12_unpack_blocks_v10_fwd failed: cudaError={rc}")
    return k_out, v_out


def kv_bf12_unpack_range(
    kv_cache, emerg_pool: KvBf12EmergPool, block_table_row, start_pos: int, num_tokens: int,
    *, unpack_k: bool = True, unpack_v: bool = True
):
    """Unpack contiguous token slice `[start_pos, start_pos+num_tokens)` into `[num_tokens, Hkv, 128]` BF16."""
    assert torch is not None
    lib = get_kv_bf12_paged_lib()
    if lib is None:
        raise RuntimeError("libkv_bf12_paged_v10.so required for kv_bf12_unpack_range")
    bt = block_table_row.to(dtype=torch.int32, device=kv_cache.device).reshape(-1).contiguous()
    base_ptr, blk_sb, head_sb, hkv = _kv_bf12_geom(kv_cache)
    k_eraw, _, v_eraw, _, _ = emerg_pool.pointers()
    k_out = torch.empty((num_tokens, hkv, 128), dtype=torch.bfloat16, device=kv_cache.device) if unpack_k else None
    v_out = torch.empty((num_tokens, hkv, 128), dtype=torch.bfloat16, device=kv_cache.device) if unpack_v else None
    if num_tokens > 0 and (unpack_k or unpack_v):
        stream = int(torch.cuda.current_stream(kv_cache.device).cuda_stream)
        rc = lib.kv_bf12_unpack_range_v10_fwd(
            base_ptr, blk_sb, head_sb, k_eraw, v_eraw, int(bt.data_ptr()),
            int(start_pos), int(num_tokens), int(bt.numel()),
            int(k_out.data_ptr()) if k_out is not None else 0,
            int(v_out.data_ptr()) if v_out is not None else 0,
            int(k_out.stride(0)) if k_out is not None else 0,
            int(k_out.stride(1)) if k_out is not None else 0,
            int(v_out.stride(0)) if v_out is not None else 0,
            int(v_out.stride(1)) if v_out is not None else 0,
            hkv, 1 if unpack_k else 0, 1 if unpack_v else 0, stream,
        )
        if rc != 0:
            raise RuntimeError(f"kv_bf12_unpack_range_v10_fwd failed: cudaError={rc}")
    return k_out, v_out


_KV_BF12_ARANGE_BT_CACHE: dict[object, object] = {}


def _get_kv_bf12_arange_bt(device, n_cols: int):
    """Return a cached read-only `[1, n_cols]` int32 GPU tensor `[0, 1, ..., n_cols-1]` (0 sync)."""
    assert torch is not None
    t = _KV_BF12_ARANGE_BT_CACHE.get(device)
    if t is None or int(t.shape[1]) < n_cols:
        cap = max(4096, int(n_cols))
        t = torch.arange(cap, dtype=torch.int32, device=device).unsqueeze(0).contiguous()
        _KV_BF12_ARANGE_BT_CACHE[device] = t
    return t[:, :n_cols]


def kv_bf12_unpack_bt_range(
    kv_cache,
    emerg_pool: KvBf12EmergPool,
    block_table,
    *,
    num_reqs: int,
    max_bt_cols: int,
    b_beg: int,
    n_blks_per_req: int,
    req_b_beg=None,
    req_n_blks=None,
    guard_bt_state=None,
    guard_prev_idx: int = 0,
    k_out=None,
    v_out=None,
    unpack_k: bool = True,
    unpack_v: bool = True,
):
    """Zero-sync GPU unpack directly from `block_table[r, b_beg + j]` into `[num_reqs*n_blks_per_req, 256, Hkv, 128]`.
    Supports optional `guard_bt_state[2]` fast-exit for zero-sync Shadow-K prefix chunks (0 D2H syncs!)."""
    assert torch is not None
    lib = get_kv_bf12_paged_lib()
    if lib is None:
        raise RuntimeError("libkv_bf12_paged_v10.so required for kv_bf12_unpack_bt_range")
    dev = kv_cache.device
    base_ptr, blk_sb, head_sb, hkv = _kv_bf12_geom(kv_cache)
    k_eraw, _, v_eraw, _, _ = emerg_pool.pointers()
    total_blks = int(num_reqs) * int(n_blks_per_req)
    if unpack_k and k_out is None:
        k_out = torch.empty((total_blks, 256, hkv, 128), dtype=torch.bfloat16, device=dev)
    if unpack_v and v_out is None:
        v_out = torch.empty((total_blks, 256, hkv, 128), dtype=torch.bfloat16, device=dev)
    if total_blks > 0 and (unpack_k or unpack_v):
        bt_in = block_table if (block_table.dtype == torch.int32 and block_table.stride(-1) == 1) else block_table.to(dtype=torch.int32).contiguous()
        bt_stride = int(bt_in.stride(0)) if bt_in.ndim >= 2 else int(max_bt_cols)
        stream = int(torch.cuda.current_stream(dev).cuda_stream)
        rc = lib.kv_bf12_unpack_bt_range_v10_fwd(
            base_ptr, blk_sb, head_sb, k_eraw, v_eraw,
            int(bt_in.data_ptr()), bt_stride,
            int(num_reqs), int(max_bt_cols), int(b_beg), int(n_blks_per_req),
            int(req_b_beg.data_ptr()) if req_b_beg is not None else 0,
            int(req_n_blks.data_ptr()) if req_n_blks is not None else 0,
            int(guard_bt_state.data_ptr()) if guard_bt_state is not None else 0,
            int(guard_prev_idx),
            int(k_out.data_ptr()) if (unpack_k and k_out is not None) else 0,
            int(v_out.data_ptr()) if (unpack_v and v_out is not None) else 0,
            int(k_out.stride(0)) if (unpack_k and k_out is not None) else 0,
            int(k_out.stride(1)) if (unpack_k and k_out is not None) else 0,
            int(k_out.stride(2)) if (unpack_k and k_out is not None) else 0,
            int(v_out.stride(0)) if (unpack_v and v_out is not None) else 0,
            int(v_out.stride(1)) if (unpack_v and v_out is not None) else 0,
            int(v_out.stride(2)) if (unpack_v and v_out is not None) else 0,
            hkv, 1 if unpack_k else 0, 1 if unpack_v else 0, stream,
        )
        if rc != 0:
            raise RuntimeError(f"kv_bf12_unpack_bt_range_v10_fwd failed: cudaError={rc}")
    return (k_out if unpack_k else None), (v_out if unpack_v else None)


def kv_bf12_stage_active_blocks(
    kv_cache,
    emerg_pool: KvBf12EmergPool,
    block_table,
    cache_seqlens,
    *,
    cu_seqlens_q=None,
    window: int = -1,
    unpack_k: bool = True,
    unpack_v: bool = True,
    seq_lens_cpu=None,
    query_lens_cpu=None,
):
    """Unpack ONLY the active KV-BF12 blocks referenced by `block_table` (within `window` if >= 0)
    into compact BF16 `[N_stage, 256, Hkv, 128]` staging buffers + densely remapped `bt_remap`.
    Wave-V10 HIGH-3 Fix: 100% ZERO D2H syncs on prefill (`kv_bf12_unpack_bt_range_v10_fwd` reads
    `block_table` directly on GPU; NO `block_table.to('cpu').tolist()`!), and fail-fast guarded
    against CUDA graph capture (decode capture uses native `fused_decode_attn_exact_sm121_v10.so`).
    """
    assert torch is not None
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "kv_bf12_stage_active_blocks cannot execute inside CUDA graph capture; "
            "decode capture must dispatch native fused_decode_attn_exact_sm121_v10.so"
        )
    dev = kv_cache.device
    num_reqs = int(cache_seqlens.shape[0])
    max_bt_cols = int(block_table.shape[1])
    if num_reqs == 0 or max_bt_cols == 0:
        hkv = int(kv_cache.shape[1])
        z_k = torch.empty((1, 256, hkv, 128), dtype=torch.bfloat16, device=dev) if unpack_k else None
        z_v = torch.empty((1, 256, hkv, 128), dtype=torch.bfloat16, device=dev) if unpack_v else None
        return z_k, z_v, torch.zeros_like(block_table)

    if seq_lens_cpu is not None and len(seq_lens_cpu) >= num_reqs:
        sl_list = [int(seq_lens_cpu[r]) for r in range(num_reqs)]
    else:
        sl_list = [int(x) for x in cache_seqlens[:num_reqs].tolist()]
    if query_lens_cpu is not None and len(query_lens_cpu) >= num_reqs:
        ql_list = [int(query_lens_cpu[r]) for r in range(num_reqs)]
    elif cu_seqlens_q is not None and cu_seqlens_q.shape[0] >= num_reqs + 1:
        cu_list = [int(x) for x in cu_seqlens_q[: num_reqs + 1].tolist()]
        ql_list = [cu_list[r + 1] - cu_list[r] for r in range(num_reqs)]
    else:
        ql_list = [1] * num_reqs

    max_end_blk = 1
    ranges: list[tuple[int, int]] = []
    for r in range(num_reqs):
        sl = sl_list[r]
        ql = ql_list[r]
        if sl <= 0:
            ranges.append((0, 0))
            continue
        past = max(0, sl - ql)
        k_beg = max(0, past - window) if window >= 0 else 0
        b_beg = min(max_bt_cols, k_beg // 256)
        b_end = min(max_bt_cols, (sl + 255) // 256)
        ranges.append((b_beg, b_end))
        if b_end > max_end_blk:
            max_end_blk = b_end

    # Fast 1-request GPU-only path (100% of Cold-1M & Warm-1M prefill: 0 D2H sync, 0 H2D copy!)
    if num_reqs == 1:
        b_beg, b_end = ranges[0]
        n_act = max(0, b_end - b_beg)
        n_blks = max(1, n_act)
        k_stage, v_stage = kv_bf12_unpack_bt_range(
            kv_cache, emerg_pool, block_table[:1],
            num_reqs=1, max_bt_cols=max_bt_cols,
            b_beg=b_beg, n_blks_per_req=n_blks,
            unpack_k=unpack_k, unpack_v=unpack_v,
        )
        if b_beg == 0 and n_act == max_end_blk:
            bt_remap = _get_kv_bf12_arange_bt(dev, max_end_blk)
        else:
            bt_remap = torch.zeros((1, max_end_blk), dtype=torch.int32, device=dev)
            if n_act > 0:
                bt_remap[0, b_beg:b_end] = _get_kv_bf12_arange_bt(dev, n_act)[0]
        return k_stage, v_stage, bt_remap

    # Multi-request 0-D2H-sync GPU path: unpack ragged per-request `[b_beg_r, b_end_r)` directly on GPU!
    max_n = max(1, max((e - b) for b, e in ranges))
    b0, e0 = ranges[0]
    uniform = all((b == b0 and (e - b) == (e0 - b0)) for b, e in ranges)
    if uniform:
        req_b_beg = None
        req_n_blks = None
    else:
        req_b_beg = torch.tensor([b for b, _ in ranges], dtype=torch.int32, device=dev)
        req_n_blks = torch.tensor([max(0, e - b) for b, e in ranges], dtype=torch.int32, device=dev)
    k_stage, v_stage = kv_bf12_unpack_bt_range(
        kv_cache, emerg_pool, block_table[:num_reqs],
        num_reqs=num_reqs, max_bt_cols=max_bt_cols,
        b_beg=b0, n_blks_per_req=max_n,
        req_b_beg=req_b_beg, req_n_blks=req_n_blks,
        unpack_k=unpack_k, unpack_v=unpack_v,
    )
    remap_rows: list[list[int]] = []
    for r, (b_beg, b_end) in enumerate(ranges):
        row_out = [0] * max_end_blk
        base_lid = r * max_n
        for j in range(max(0, b_end - b_beg)):
            row_out[b_beg + j] = base_lid + j
        remap_rows.append(row_out)
    bt_remap = torch.tensor(remap_rows, dtype=torch.int32, device=dev)
    return k_stage, v_stage, bt_remap


_KV_BF12_EMERG_BY_PTR: dict[int, KvBf12EmergPool] = {}
_KV_BF12_FALLBACK_EMERG: dict[object, KvBf12EmergPool] = {}
_KV_BF12_RESOLVER_REGISTERED = False


def register_kv_bf12_emerg(kv_cache, emerg_pool: KvBf12EmergPool) -> None:
    """Associate a layer's `KvBf12EmergPool` with its KV-BF12 cache pointer for prefill resolvers."""
    if kv_cache is not None and emerg_pool is not None and kv_cache.numel() > 0:
        _KV_BF12_EMERG_BY_PTR[int(kv_cache.data_ptr())] = emerg_pool
        ensure_kv_bf12_prefill_resolver_registered()


def _kv_bf12_default_prefill_resolver(
    key_cache,
    value_cache,
    block_table,
    cache_seqlens,
    *,
    need_k: bool = True,
    need_v: bool = True,
    window: int = -1,
    cu_seqlens_q=None,
    max_seqlen_q: int = 0,
):
    """Default Wave-V10 KV-BF12 prefill unpacker (`libkv_bf12_paged_v10.so`) registered with
    `sm12_attn.register_kv_bf12_prefill_resolver(...)`. Handles both `(NBLK, Hkv, 256, 198)` bf16
    and `(NBLK, Hkv, 256, 396)` uint8 views cleanly."""
    if get_kv_bf12_paged_lib() is None:
        return None, None, None
    kv_src = key_cache if is_kv_bf12_cache(key_cache) else value_cache
    if not is_kv_bf12_cache(kv_src):
        return key_cache, value_cache, block_table
    emerg = getattr(kv_src, "_kv_bf12_emerg", None)
    if emerg is None:
        emerg = _KV_BF12_EMERG_BY_PTR.get(int(kv_src.data_ptr()))
    if emerg is None:
        emerg = _KV_BF12_FALLBACK_EMERG.get(kv_src.device)
        if emerg is None:
            emerg = KvBf12EmergPool(kv_src.device)
            _KV_BF12_FALLBACK_EMERG[kv_src.device] = emerg
    k_stage, v_stage, bt_remap = kv_bf12_stage_active_blocks(
        kv_src,
        emerg,
        block_table,
        cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        window=window,
        unpack_k=need_k,
        unpack_v=need_v,
    )
    out_k = k_stage if need_k else key_cache
    out_v = v_stage if need_v else value_cache
    return out_k, out_v, bt_remap


def ensure_kv_bf12_prefill_resolver_registered() -> bool:
    """Lazily register `_kv_bf12_default_prefill_resolver` with `sm12_attn` if available and unset."""
    global _KV_BF12_RESOLVER_REGISTERED
    if get_kv_bf12_paged_lib() is None:
        return False
    try:
        from . import sm12_attn as _sm12_attn
        if (
            hasattr(_sm12_attn, "register_kv_bf12_prefill_resolver")
            and _sm12_attn.get_kv_bf12_prefill_resolver() is None
        ):
            _sm12_attn.register_kv_bf12_prefill_resolver(_kv_bf12_default_prefill_resolver)
            _KV_BF12_RESOLVER_REGISTERED = True
        return _sm12_attn.get_kv_bf12_prefill_resolver() is not None
    except Exception:
        return False


def prepack_rel_proj_t(rel_proj_weight):
    """Cache contiguous `[E, 16]` BF16 transpose of `rel_logits_proj` (shape `[16, E]` or `[E, 16]`)."""
    assert torch is not None
    key = int(rel_proj_weight.data_ptr())
    hit = _REL_PROJ_T_CACHE.get(key)
    if hit is not None:
        return hit
    w = rel_proj_weight.detach()
    if w.shape[0] == 16 and w.shape[1] != 16:
        wt = w.transpose(0, 1).contiguous()
    else:
        wt = w.contiguous()
    _REL_PROJ_T_CACHE[key] = wt
    return wt


def ensure_scratch(device, max_splits: int = 16, max_t: int = 16, max_h: int = 6, max_sem: int = 64):
    assert torch is not None
    dev_idx = device.index if device.index is not None else 0
    s = _SCRATCH.get(dev_idx)
    if (
        s is None
        or s["op"][0].shape[0] < max_splits
        or s["op"][0].shape[1] < max_t
        or s["op"][0].shape[2] < max_h
        or s["sem"][0].shape[0] < max_sem
    ):
        _SCRATCH[dev_idx] = {
            "op": [
                torch.empty((max_splits, max_t, max_h, 128), dtype=torch.float32, device=device),
                torch.empty((max_splits, max_t, max_h, 128), dtype=torch.float32, device=device),
            ],
            "lp": [
                torch.empty((max_splits, max_t, max_h), dtype=torch.float32, device=device),
                torch.empty((max_splits, max_t, max_h), dtype=torch.float32, device=device),
            ],
            "sem": [
                torch.zeros((max_sem,), dtype=torch.int32, device=device),
                torch.zeros((max_sem,), dtype=torch.int32, device=device),
            ],
            "idx": 0,
        }
    return _SCRATCH[dev_idx]


def ensure_v11_scratch(device, max_st: int = 2048, max_h: int = 6, max_sem: int = 256):
    """Wave-V11 compact right-sized flat ping-pong scratch (`S * T <= max_st`).
    Pre-allocated once outside CUDAGraph capture (`6.0 MiB` per slot @ `max_st=2048, H=6`),
    avoiding `[48, 320, 6, 128]` stride holes & guaranteeing contiguous L2-hot active views
    (`1.125 MiB` @ B=1 T=8 S=48; `2.25 MiB` @ T=16 S=48; `1.125 MiB` @ B=32 T=128 S=3)."""
    assert torch is not None
    dev_idx = device.index if device.index is not None else 0
    need_st = max(2048, int(max_st))
    need_h = max(6, int(max_h))
    need_sem = max(256, int(max_sem))
    s = _SCRATCH_V11.get(dev_idx)
    if (
        s is None
        or s["max_st"] < need_st
        or s["max_h"] < need_h
        or s["sem"][0].shape[0] < need_sem
    ):
        _SCRATCH_V11[dev_idx] = {
            "op_flat": [
                torch.empty((need_st * need_h * 128,), dtype=torch.float32, device=device),
                torch.empty((need_st * need_h * 128,), dtype=torch.float32, device=device),
            ],
            "lp_flat": [
                torch.empty((need_st * need_h,), dtype=torch.float32, device=device),
                torch.empty((need_st * need_h,), dtype=torch.float32, device=device),
            ],
            "sem": [
                torch.zeros((need_sem,), dtype=torch.int32, device=device),
                torch.zeros((need_sem,), dtype=torch.int32, device=device),
            ],
            "max_st": need_st,
            "max_h": need_h,
            "idx": 0,
        }
    return _SCRATCH_V11[dev_idx]


def _next_scratch_v11(device, num_splits: int, T: int, H: int, need_sem: int):
    st = max(1, int(num_splits)) * max(1, int(T))
    h_eff = max(6, int(H))
    s = ensure_v11_scratch(device, max(2048, st), h_eff, max(256, int(need_sem)))
    idx = s["idx"]
    s["idx"] = idx ^ 1
    op_v = s["op_flat"][idx][: st * int(H) * 128].view(int(num_splits), int(T), int(H), 128)
    lp_v = s["lp_flat"][idx][: st * int(H)].view(int(num_splits), int(T), int(H))
    return op_v, lp_v, s["sem"][idx]


def _next_scratch(device, num_splits: int, T: int, H: int, need_sem: int):
    if V11_S48 or V11_C1632_GLOBAL:
        return _next_scratch_v11(device, num_splits, T, H, need_sem)
    v8_active = V9_ENABLED or _V8_C28_FUSE_MT or _V8_C1_FUSE_MT
    min_splits = 32 if v8_active else 16
    min_t = 64 if (_C28_SPLITS or _C28_PDL_WO or ENABLED_C32 or v8_active) else 16
    min_sem = 128 if (_C28_SPLITS or _C28_PDL_WO or ENABLED_C32 or v8_active) else 64
    s = ensure_scratch(device, max(min_splits, num_splits), max(min_t, T), max(6, H), max(min_sem, need_sem))
    idx = s["idx"]
    s["idx"] = idx ^ 1
    return s["op"][idx], s["lp"][idx], s["sem"][idx]


def _pick_v8_global_splits(base_ctas: int, target_ctas: int) -> int:
    """Choose S in _V8_GLOBAL_SPLIT_LADDER minimizing bottleneck WorkPerSM = ceil(base_ctas*S/48)/S
    subject to base_ctas*S <= target_ctas (avoiding wave-tail traps like B=2 S=32 -> 64 CTAs!)."""
    target = max(16, min(96, int(target_ctas)))
    b = max(1, int(base_ctas))
    cands = [s for s in _V8_GLOBAL_SPLIT_LADDER if b * s <= target]
    if not cands:
        return 1
    prefer_high = target > 48
    best_s = cands[0]
    best_num = (b * best_s + 47) // 48
    for s in cands[1:]:
        num = (b * s + 47) // 48
        lhs = num * best_s
        rhs = best_num * s
        if lhs < rhs or (lhs == rhs and ((s > best_s) if prefer_high else (s < best_s))):
            best_s, best_num = s, num
    return best_s


def _select_v8_fuse_mt_plan(
    num_reqs: int,
    mtiles: int,
    window: int,
    G: int,
    has_c32: bool,
    is_v8: bool,
) -> tuple[bool, int] | None:
    """Select `(use_fuse_mt, num_splits)` for Wave-V8 B=1..8 FUSE_MT=1 multi-split path.
    Returns None when v8 opt-in is OFF, lib < v8, or num_reqs >= 9 (strictly preserving v7!)."""
    if not (has_c32 and is_v8):
        return None
    can_fmt = 2 <= mtiles <= 3 and G >= 3
    if num_reqs == 1:
        if not _V8_C1_FUSE_MT:
            return None
        if window >= 0:
            if not _V8_C28_SWA_FUSE_MT:
                return None
            s = _V8_C1_SWA_SPLITS if _V8_C1_SWA_SPLITS in _V8_LOCAL_SPLIT_LADDER else 8
            return (can_fmt, s)
        s = _V8_C1_GLOBAL_SPLITS if _V8_C1_GLOBAL_SPLITS in _V8_GLOBAL_SPLIT_LADDER else 16
        return (can_fmt, s)
    if 2 <= num_reqs <= 8 and _V8_C28_FUSE_MT:
        if _V8_SWA_B38_ONLY and (window < 0 or num_reqs < _V8_SWA_MIN_B):
            return None
        if window >= 0:
            if not _V8_C28_SWA_FUSE_MT:
                return None
            if _V8_SWA_SPLITS == "auto":
                s = 6 if mtiles >= 2 else (8 if num_reqs <= 6 else 4)
            else:
                try:
                    req_s = int(_V8_SWA_SPLITS)
                except ValueError:
                    req_s = 6
                s = req_s if req_s in _V8_LOCAL_SPLIT_LADDER else 6
            return (can_fmt, s)
        base_ctas = num_reqs if (can_fmt or mtiles == 1) else (mtiles * num_reqs)
        s = _pick_v8_global_splits(base_ctas, _V8_GLOBAL_TARGET_CTAS)
        return (can_fmt, s)
    return None


def _pick_v11_global_splits(
    base_ctas: int,
    target_ctas: int,
    ladder: tuple[int, ...] = _V11_LONG_SPLIT_LADDER,
) -> int:
    """Wave-V11 split selector over `(2, 3, 4, 6, 8, 12, 16, 24, 32, 48)` minimizing
    bottleneck WorkPerSM = ceil(base_ctas * S / 48) / S subject to `base_ctas * S <= target`.
    If `base_ctas * min(ladder) > target` (e.g. B=26..32 under target=48), raises effective
    target to 96 (2 CTAs/SM) so B=26..32 cleanly select S=3 (78..96 CTAs, 0 wave-tail!)."""
    target = max(16, min(96, int(target_ctas)))
    b = max(1, int(base_ctas))
    cands = [s for s in ladder if b * s <= target]
    eff_target = target
    if not cands and target < 96:
        eff_target = 96
        cands = [s for s in ladder if b * s <= eff_target]
    if not cands:
        return 1
    prefer_high = eff_target > 48
    best_s = cands[0]
    best_num = (b * best_s + 47) // 48
    for s in cands[1:]:
        num = (b * s + 47) // 48
        lhs = num * best_s
        rhs = best_num * s
        if lhs < rhs or (lhs == rhs and ((s > best_s) if prefer_high else (s < best_s))):
            best_s, best_num = s, num
    return best_s


def _select_v9_global_paired_plan(
    num_reqs: int,
    mtiles: int,
    window: int,
    G: int,
    s_short: int,
    has_v9: bool,
    is_v9: bool,
    is_v11: bool | None = None,
) -> tuple[int, int, int, int] | None:
    """Select `(s_short, s_long, kv_tok_thresh, kv_evict_first_long)` for Wave-V9/V11
    CUDAGraph-safe GPU-side L2-threshold paired Global dispatch (`window < 0`).
    Returns None when V9 is OFF, lib < v9, Local SWA (`window >= 0`), `mtiles not in {2, 3}`,
    or `num_reqs` out of supported paired range (strictly preserving V10/V9/V8/V7 when V11 OFF!)."""
    if not (V9_ENABLED and has_v9 and is_v9):
        return None
    eff_v11 = _IS_V11_LIB if is_v11 is None else bool(is_v11)
    v11_s48_on = V11_S48 and eff_v11
    v11_c1632_on = V11_C1632_GLOBAL and eff_v11
    min_g = 2 if (v11_s48_on or v11_c1632_on) else 3
    if window >= 0 or G < min_g or mtiles not in (2, 3):
        return None
    short_ladder = _V11_SHORT_SPLIT_LADDER if (v11_s48_on or v11_c1632_on) else _V9_SHORT_SPLIT_LADDER
    if s_short not in short_ladder:
        return None
    thresh = _V9_GLOBAL_L2_THRESH_TOKS
    evict_long = 1 if _V9_GLOBAL_LONG_EVICT_FIRST else 0
    if num_reqs == 1:
        if v11_s48_on:
            s_long = _V11_C1_LONG_SPLITS if _V11_C1_LONG_SPLITS in _V11_LONG_SPLIT_LADDER else 48
        else:
            s_long = _V9_C1_LONG_SPLITS if _V9_C1_LONG_SPLITS in _V9_LONG_SPLIT_LADDER else 16
        return (s_short, s_long, thresh, evict_long)
    if 2 <= num_reqs <= 8:
        if v11_s48_on:
            s_long = _pick_v11_global_splits(num_reqs, _V11_TARGET_CTAS)
        else:
            s_long = _pick_v8_global_splits(num_reqs, _V9_GLOBAL_LONG_TARGET_CTAS)
            if s_long not in _V9_LONG_SPLIT_LADDER:
                s_long = max(2, min(32, s_long))
        return (s_short, s_long, thresh, evict_long)
    if _V9_C915_GLOBAL and 9 <= num_reqs <= 15:
        if v11_s48_on:
            s_long = _pick_v11_global_splits(num_reqs, _V11_TARGET_CTAS)
            if s_long in _V11_LONG_SPLIT_LADDER:
                return (s_short, s_long, thresh, evict_long)
        else:
            s_long = _pick_v8_global_splits(num_reqs, _V9_GLOBAL_LONG_TARGET_CTAS)
            if s_long in _V9_LONG_SPLIT_LADDER:
                return (s_short, s_long, thresh, evict_long)
    if v11_c1632_on and 16 <= num_reqs <= 32:
        s_long = _pick_v11_global_splits(num_reqs, _V11_TARGET_CTAS)
        if s_long in _V11_LONG_SPLIT_LADDER:
            return (s_short, s_long, thresh, evict_long)
    return None


def _select_num_splits(
    num_reqs: int,
    mtiles: int,
    window: int,
    has_c32: bool,
    has_global_s8: bool,
) -> int | None:
    """Select split-KV factor S. Strictly unchanged for B == 1, B >= 9, and default-OFF."""
    if _C28_SPLITS and 2 <= num_reqs <= 8:
        if window >= 0:
            if num_reqs == 2:
                req_s = _C28_SWA_B2_SPLITS
            elif num_reqs <= 4:
                req_s = _C28_SWA_B34_SPLITS
            else:
                req_s = _C28_SWA_B58_SPLITS
            return req_s if req_s in (1, 2, 4, 8) else 1
        avail = (16, 8, 4, 2, 1) if has_global_s8 else (16, 4, 2, 1)
        base_ctas = max(1, mtiles * num_reqs)
        target = max(16, _C28_GLOBAL_TARGET_CTAS)
        for cand in avail:
            if base_ctas * cand <= target:
                return cand
        return 1
    if num_reqs <= 4:
        return 8 if window >= 0 else 16
    if not has_c32:
        return None
    return _C32_SPLITS if _C32_SPLITS in (1, 2, 4) else 1


def fused_target_decode_attn(
    q, r, k, v,
    q_scale, r_scale, tau, rel_proj_t,
    k_conv_w, v_conv_w, k_dyn, v_dyn,
    conv_cache, key_cache, value_cache,
    kv_slot_mapping, positions, conv_block_table, conv_seq_idx, conv_slot_mapping, conv_query_start,
    attn_block_table, attn_seq_lens, attn_cu_seqlens_q,
    gate, sinks, output,
    softmax_scale: float,
    window: int,
    rel_threshold: float,
    clip_threshold: float,
    clip_eps: float = 1e-6,
    norm_eps: float = 1e-6,
    num_actual_tokens: int | None = None,
    n_live_buf=None,
    q_out_buf=None,
    tree_mode: int = _TREE_MODE,
    pdl: bool = _USE_PDL,
    trig_wo: bool = _TRIG_WO,
    pdl_prologue: bool | None = None,
    max_seqlen_q: int | None = None,
    kv_bf12_cache=None,
    kv_bf12_emerg: KvBf12EmergPool | None = None,
) -> bool:
    """Execute the complete 2-kernel 0-ULP fused target decode attention pipeline:
      1. `attn_prologue_kvq_exact_fwd` (24 CTAs for T<=16, wave-balanced 96 CTAs for T=17..320; NO RelProj!)
      2. `fused_decode_attn_otf_mgs[_c32]_fwd` (OTF rel-bias + XOR ldmatrix + direct S=1 Gate+Sink / last-split MGS!)
    Returns True if executed, False if unsupported shape/config (caller falls back cleanly).
    """
    lib = _load_lib()
    if lib is None or torch is None:
        return False
    T = int(q.shape[0])
    if T <= 0:
        return False
    if T <= 16:
        if not ENABLED:
            return False
    else:
        if not (ENABLED_C32 and _HAS_C32_SYM and T <= MAX_TOKENS_C32):
            return False
    use_kv_bf12 = kv_bf12_cache is not None
    if use_kv_bf12 and not (_HAS_V10_BF12_SYM and kv_bf12_emerg is not None):
        return False
    NT = T if num_actual_tokens is None else int(num_actual_tokens)
    num_reqs = int(attn_seq_lens.shape[0])
    H = int(q_scale.shape[0])
    Hkv = int(kv_bf12_cache.shape[1]) if use_kv_bf12 else int(key_cache.shape[2])
    G = H // Hkv
    eff_max_q = int(max_seqlen_q) if max_seqlen_q is not None else T
    max_qg = 96 if (_SAM_EXT_ATTN_FIX and num_reqs <= 2 and T <= 32 and eff_max_q <= 16) else 64
    if eff_max_q * G > max_qg:
        return False
    if window >= 0 and window != 1023:
        return False
    mtiles = (eff_max_q * G + 15) // 16
    v8_plan = _select_v8_fuse_mt_plan(num_reqs, mtiles, window, G, _HAS_C32_SYM, _IS_V8_LIB)
    if v8_plan is not None:
        use_fuse_mt, num_splits = v8_plan
    else:
        num_splits = _select_num_splits(num_reqs, mtiles, window, _HAS_C32_SYM, _IS_V7_LIB)
        if num_splits is None:
            return False
        use_fuse_mt = False
        if _HAS_C32_SYM and T > 16 and 1 <= mtiles <= 3 and num_splits in (1, 2, 4):
            if _C32_FUSE_MT == "1":
                use_fuse_mt = True
            elif _C32_FUSE_MT == "auto":
                use_fuse_mt = (num_reqs >= 16)
    v9_pair = None
    _try_v11_c1632_pair = (
        V11_C1632_GLOBAL
        and _IS_V11_LIB
        and window < 0
        and 16 <= num_reqs <= 32
        and v8_plan is None
    )
    if (not use_fuse_mt) or _try_v11_c1632_pair:
        v9_pair = _select_v9_global_paired_plan(
            num_reqs, mtiles, window, G, num_splits, _HAS_V9_SYM, _IS_V9_LIB, _IS_V11_LIB
        )
        if v9_pair is not None:
            use_fuse_mt = False
    allow_c28_pdl = _C28_PDL_WO and 17 <= T <= 64
    if T > 16 and not allow_c28_pdl:
        # At M >= 65 (or default-OFF M >= 17) wo_ud runs cuBLAS/non-PDL; never arm _WO_ARMED!
        pdl = False
        trig_wo = False
    elif NT < T:
        # output[NT:].zero_() executes between attn and wo_ud for ALL T (including T<=16 v8_plan=None);
        # never fire trig_wo or arm _WO_ARMED across the zero_() fill kernel! (#206 reconciliation)
        trig_wo = False
    eff_pdl_prologue = (pdl if pdl_prologue is None else bool(pdl_prologue)) and bool(pdl)

    q_out = q_out_buf if q_out_buf is not None else torch.empty((T, H, 128), dtype=torch.bfloat16, device=q.device)
    dynamic = (k_dyn is not None and v_dyn is not None)
    has_tau = (tau is not None)
    has_nlive = (n_live_buf is not None)
    # Exact Inkling2 W=4 short-conv scale matching patches-s6/sconv_ops.py:1191!
    width = 4
    sconv_scale = float((width * (2 if dynamic else 1) + 1) ** -0.5)
    # conv_cache shape is [num_blocks, 1, N_cache, C] -> stride(2) & shape[2] (matching sconv_ops.py:1182,1188)!
    c_n_dim = 2 if conv_cache.dim() == 4 else 1
    stream = int(torch.cuda.current_stream(q.device).cuda_stream)

    if use_kv_bf12:
        kv_base_ptr, kv_blk_sb, kv_head_sb, _ = _kv_bf12_geom(kv_bf12_cache)
        k_eraw, k_eused, v_eraw, v_eused, ecap = kv_bf12_emerg.pointers()
        rc = lib.attn_prologue_kvq_bf12_exact_v10_fwd(
            int(k.data_ptr()), int(v.data_ptr()),
            int(k_conv_w.data_ptr()), int(v_conv_w.data_ptr()),
            int(k_dyn.data_ptr()) if dynamic else 0,
            int(v_dyn.data_ptr()) if dynamic else 0,
            int(conv_cache.data_ptr()), kv_base_ptr, kv_base_ptr + KV_BF12_HALF_BYTES,
            int(kv_slot_mapping.data_ptr()), int(positions.data_ptr()),
            int(conv_seq_idx.data_ptr()), int(conv_slot_mapping.data_ptr()),
            int(conv_block_table.data_ptr()), int(conv_query_start.data_ptr()),
            int(q.data_ptr()), int(q_scale.data_ptr()),
            int(tau.data_ptr()) if has_tau else 0,
            int(q_out.data_ptr()),
            int(n_live_buf.data_ptr()) if has_nlive else 0,
            T, int(k.stride(0)), int(v.stride(0)),
            int(k_conv_w.stride(0)), int(k_conv_w.stride(1)),
            int(k_dyn.stride(0)) if dynamic else 0,
            int(v_dyn.stride(0)) if dynamic else 0,
            int(conv_cache.stride(0)), int(conv_cache.stride(c_n_dim)),
            kv_blk_sb, 0, 256,
            int(conv_block_table.stride(0)), int(conv_block_table.shape[1]), int(conv_cache.shape[c_n_dim]),
            int(q.stride(0)),
            sconv_scale, float(clip_threshold), float(clip_eps), float(norm_eps),
            1 if dynamic else 0, 1 if has_tau else 0, 1 if has_nlive else 0, 1 if eff_pdl_prologue else 0,
            stream,
            k_eraw, k_eused, v_eraw, v_eused, ecap,
        )
        if rc != 0:
            raise RuntimeError(f"attn_prologue_kvq_bf12_exact_v10_fwd failed: cudaError={rc}")
    else:
        rc = lib.attn_prologue_kvq_exact_fwd(
            int(k.data_ptr()), int(v.data_ptr()),
            int(k_conv_w.data_ptr()), int(v_conv_w.data_ptr()),
            int(k_dyn.data_ptr()) if dynamic else 0,
            int(v_dyn.data_ptr()) if dynamic else 0,
            int(conv_cache.data_ptr()), int(key_cache.data_ptr()), int(value_cache.data_ptr()),
            int(kv_slot_mapping.data_ptr()), int(positions.data_ptr()),
            int(conv_seq_idx.data_ptr()), int(conv_slot_mapping.data_ptr()),
            int(conv_block_table.data_ptr()), int(conv_query_start.data_ptr()),
            int(q.data_ptr()), int(q_scale.data_ptr()),
            int(tau.data_ptr()) if has_tau else 0,
            int(q_out.data_ptr()),
            int(n_live_buf.data_ptr()) if has_nlive else 0,
            T, int(k.stride(0)), int(v.stride(0)),
            int(k_conv_w.stride(0)), int(k_conv_w.stride(1)),
            int(k_dyn.stride(0)) if dynamic else 0,
            int(v_dyn.stride(0)) if dynamic else 0,
            int(conv_cache.stride(0)), int(conv_cache.stride(c_n_dim)),
            int(key_cache.stride(0)), int(key_cache.stride(1)), int(key_cache.shape[1]),
            int(conv_block_table.stride(0)), int(conv_block_table.shape[1]), int(conv_cache.shape[c_n_dim]),
            int(q.stride(0)),
            sconv_scale, float(clip_threshold), float(clip_eps), float(norm_eps),
            1 if dynamic else 0, 1 if has_tau else 0, 1 if has_nlive else 0, 1 if eff_pdl_prologue else 0,
            stream,
        )
        if rc != 0:
            raise RuntimeError(f"attn_prologue_kvq_exact_fwd failed: cudaError={rc}")

    eff_max_splits = max(num_splits, v9_pair[1]) if v9_pair is not None else num_splits
    if eff_max_splits == 1:
        op_ptr = lp_ptr = sem_ptr = 0
        so_s = so_t = so_h = sl_s = sl_t = sl_h = 0
    else:
        need_sem = num_reqs * (1 if (use_fuse_mt and v9_pair is None) else mtiles) * Hkv
        op_buf, lp_buf, sem_buf = _next_scratch(q.device, eff_max_splits, T, H, need_sem)
        op_ptr, lp_ptr, sem_ptr = int(op_buf.data_ptr()), int(lp_buf.data_ptr()), int(sem_buf.data_ptr())
        so_s, so_t, so_h = int(op_buf.stride(0)), int(op_buf.stride(1)), int(op_buf.stride(2))
        sl_s, sl_t, sl_h = int(lp_buf.stride(0)), int(lp_buf.stride(1)), int(lp_buf.stride(2))
    E = int(rel_proj_t.shape[0]) if rel_proj_t is not None else 0
    r_view = r.view(T, H, 16) if r.dim() == 2 else r

    if use_kv_bf12:
        k_ptr_arg = kv_base_ptr
        v_ptr_arg = kv_base_ptr + KV_BF12_HALF_BYTES
        sk_b_arg, sk_s_arg, sk_h_arg = kv_blk_sb, 0, kv_head_sb
        sv_b_arg, sv_s_arg, sv_h_arg = kv_blk_sb, 0, kv_head_sb
        bs_arg = 256
    else:
        k_ptr_arg = int(key_cache.data_ptr())
        v_ptr_arg = int(value_cache.data_ptr())
        sk_b_arg, sk_s_arg, sk_h_arg = int(key_cache.stride(0)), int(key_cache.stride(1)), int(key_cache.stride(2))
        sv_b_arg, sv_s_arg, sv_h_arg = int(value_cache.stride(0)), int(value_cache.stride(1)), int(value_cache.stride(2))
        bs_arg = int(key_cache.shape[1])

    common_args = (
        int(q_out.data_ptr()),
        int(r_view.data_ptr()),
        int(r_scale.data_ptr()),
        int(tau.data_ptr()) if has_tau else 0,
        int(rel_proj_t.data_ptr()) if E > 0 else 0,
        k_ptr_arg,
        v_ptr_arg,
        int(attn_block_table.data_ptr()),
        int(attn_seq_lens.data_ptr()),
        int(attn_cu_seqlens_q.data_ptr()),
        int(gate.data_ptr()),
        int(sinks.data_ptr()),
        op_ptr, lp_ptr, sem_ptr,
        int(output.data_ptr()),
        int(n_live_buf.data_ptr()) if has_nlive else 0,
        int(q_out.stride(0)), int(q_out.stride(1)),
        int(r_view.stride(0)), int(r_view.stride(1)),
        sk_b_arg, sk_s_arg, sk_h_arg,
        sv_b_arg, sv_s_arg, sv_h_arg,
        int(attn_block_table.stride(0)),
        int(gate.stride(0)), int(sinks.stride(0)), int(output.stride(0)),
        so_s, so_t, so_h, sl_s, sl_t, sl_h,
        float(softmax_scale * _LOG2E), float(rel_threshold), float(clip_eps),
        T, NT, mtiles, num_reqs, Hkv, bs_arg, E, G,
        int(window), int(num_splits),
        1 if has_tau else 0, 1 if has_nlive else 0,
        1 if pdl else 0, 1 if trig_wo else 0, int(tree_mode),
    )
    if use_kv_bf12:
        if v9_pair is not None:
            _, v9_s_long, v9_thresh, v9_evict_long = v9_pair
            rc = lib.fused_decode_attn_otf_mgs_v10_bf12_paired_fwd(
                *common_args, int(v9_s_long), int(v9_thresh), int(v9_evict_long), stream, k_eraw, v_eraw
            )
        elif _HAS_C32_SYM and (use_fuse_mt or T > 16):
            rc = lib.fused_decode_attn_otf_mgs_c32_bf12_v10_fwd(
                *common_args, 1 if use_fuse_mt else 0, 1 if _C32_EVICT_FIRST else 0, stream, k_eraw, v_eraw
            )
        else:
            rc = lib.fused_decode_attn_otf_mgs_bf12_v10_fwd(*common_args, stream, k_eraw, v_eraw)
    elif v9_pair is not None:
        _, v9_s_long, v9_thresh, v9_evict_long = v9_pair
        rc = lib.fused_decode_attn_otf_mgs_v9_fwd(
            *common_args, int(v9_s_long), int(v9_thresh), int(v9_evict_long), stream
        )
    elif _HAS_C32_SYM and (use_fuse_mt or T > 16):
        rc = lib.fused_decode_attn_otf_mgs_c32_fwd(
            *common_args, 1 if use_fuse_mt else 0, 1 if _C32_EVICT_FIRST else 0, stream
        )
    else:
        rc = lib.fused_decode_attn_otf_mgs_fwd(*common_args, stream)
    if rc != 0:
        raise RuntimeError(f"fused_decode_attn_otf_mgs_fwd failed: cudaError={rc}")
    if NT < T:
        output[NT:].zero_()
    if ((T <= 16) or allow_c28_pdl) and (NT == T) and trig_wo and pdl:
        from . import sm12_attn as _sm12_attn
        _sm12_attn._WO_ARMED = True
    return True


def _bind_tree_symbols(lib):
    if not (hasattr(lib, "attn_prologue_kvq_tree_exact_fwd") and hasattr(lib, "fused_decode_attn_otf_mgs_tree_fwd")):
        return None
    lib.attn_prologue_kvq_tree_exact_fwd.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int,
        ctypes.c_int64, ctypes.c_int, ctypes.c_int, ctypes.c_int64,
        ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_uint64,
    ]
    lib.attn_prologue_kvq_tree_exact_fwd.restype = ctypes.c_int

    lib.fused_decode_attn_otf_mgs_tree_fwd.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_int64, ctypes.c_int64,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_float, ctypes.c_float, ctypes.c_float,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_uint64,
    ]
    lib.fused_decode_attn_otf_mgs_tree_fwd.restype = ctypes.c_int
    if hasattr(lib, "attn_prologue_kvq_tree_bf12_exact_v10_fwd"):
        lib.attn_prologue_kvq_tree_bf12_exact_v10_fwd.argtypes = (
            list(lib.attn_prologue_kvq_tree_exact_fwd.argtypes)
            + [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32]
        )
        lib.attn_prologue_kvq_tree_bf12_exact_v10_fwd.restype = ctypes.c_int
    if hasattr(lib, "fused_decode_attn_otf_mgs_tree_bf12_v10_fwd"):
        lib.fused_decode_attn_otf_mgs_tree_bf12_v10_fwd.argtypes = (
            list(lib.fused_decode_attn_otf_mgs_tree_fwd.argtypes)
            + [ctypes.c_void_p, ctypes.c_void_p]
        )
        lib.fused_decode_attn_otf_mgs_tree_bf12_v10_fwd.restype = ctypes.c_int
    return lib


def _load_tree_lib():
    global _TREE_LIB, _TREE_LIB_TRIED
    if _TREE_LIB_TRIED:
        return _TREE_LIB
    with _LIB_LOCK:
        if not _TREE_LIB_TRIED:
            _TREE_LIB_TRIED = True
            if V11_S48 or V11_C1632_GLOBAL or KV_BF12_ENABLED or V7_ENABLED or V8_ENABLED or V9_ENABLED or _V8_C28_FUSE_MT or _V8_C1_FUSE_MT:
                base = _load_lib()
                if base is not None:
                    _TREE_LIB = _bind_tree_symbols(base)
            if _TREE_LIB is None and os.path.exists(_TREE_LIB_PATH):
                _TREE_LIB = _bind_tree_symbols(ctypes.CDLL(_TREE_LIB_PATH))
            if _TREE_LIB is None:
                base = _load_lib()
                if base is not None:
                    _TREE_LIB = _bind_tree_symbols(base)
    return _TREE_LIB


def has_tree_lib() -> bool:
    return _load_tree_lib() is not None


def fused_target_decode_attn_tree(
    q, r, k, v,
    q_scale, r_scale, tau, rel_proj_t,
    k_conv_w, v_conv_w, k_dyn, v_dyn,
    conv_cache, key_cache, value_cache,
    kv_slot_mapping, positions, conv_block_table, conv_seq_idx, conv_slot_mapping, conv_query_start,
    attn_block_table, attn_seq_lens, attn_cu_seqlens_q,
    gate, sinks, output,
    *,
    tree_sconv_anc,
    tree_meta,
    softmax_scale: float,
    window: int,
    rel_threshold: float,
    clip_threshold: float,
    clip_eps: float = 1e-6,
    norm_eps: float = 1e-6,
    num_actual_tokens: int | None = None,
    n_live_buf=None,
    q_out_buf=None,
    tree_mode: int = _TREE_MODE,
    pdl: bool = _USE_PDL,
    trig_wo: bool = _TRIG_WO,
    pdl_prologue: bool | None = None,
    kv_bf12_cache=None,
    kv_bf12_emerg: KvBf12EmergPool | None = None,
) -> bool:
    """Execute 0-ULP 2-kernel fused SM121a two-branch tree decode attention (M <= 8, ws <= 2)."""
    lib = _load_tree_lib()
    if lib is None or torch is None or tree_sconv_anc is None or tree_meta is None:
        return False
    use_kv_bf12 = kv_bf12_cache is not None
    if use_kv_bf12 and not (
        kv_bf12_emerg is not None
        and hasattr(lib, "attn_prologue_kvq_tree_bf12_exact_v10_fwd")
        and hasattr(lib, "fused_decode_attn_otf_mgs_tree_bf12_v10_fwd")
    ):
        return False
    T = int(q.shape[0])
    num_reqs = int(attn_seq_lens.shape[0])
    H = int(q_scale.shape[0])
    Hkv = int(kv_bf12_cache.shape[1]) if use_kv_bf12 else int(key_cache.shape[2])
    G = H // Hkv
    if T <= 0 or T > 8 or num_reqs != 1 or G != 6:
        return False
    if window >= 0 and window != 1023:
        return False
    NT = T if num_actual_tokens is None else int(num_actual_tokens)
    num_splits = 8 if window >= 0 else 16

    q_out = q_out_buf if q_out_buf is not None else torch.empty((T, H, 128), dtype=torch.bfloat16, device=q.device)
    dynamic = (k_dyn is not None and v_dyn is not None)
    has_tau = (tau is not None)
    has_nlive = (n_live_buf is not None)
    width = 4
    sconv_scale = float((width * (2 if dynamic else 1) + 1) ** -0.5)
    c_n_dim = 2 if conv_cache.dim() == 4 else 1
    stream = int(torch.cuda.current_stream(q.device).cuda_stream)
    eff_pdl_prologue = (pdl if pdl_prologue is None else bool(pdl_prologue)) and bool(pdl)

    if use_kv_bf12:
        kv_base_ptr, kv_blk_sb, kv_head_sb, _ = _kv_bf12_geom(kv_bf12_cache)
        k_eraw, k_eused, v_eraw, v_eused, ecap = kv_bf12_emerg.pointers()
        rc = lib.attn_prologue_kvq_tree_bf12_exact_v10_fwd(
            int(k.data_ptr()), int(v.data_ptr()),
            int(k_conv_w.data_ptr()), int(v_conv_w.data_ptr()),
            int(k_dyn.data_ptr()) if dynamic else 0,
            int(v_dyn.data_ptr()) if dynamic else 0,
            int(conv_cache.data_ptr()), kv_base_ptr, kv_base_ptr + KV_BF12_HALF_BYTES,
            int(kv_slot_mapping.data_ptr()), int(positions.data_ptr()),
            int(conv_seq_idx.data_ptr()), int(conv_slot_mapping.data_ptr()),
            int(conv_block_table.data_ptr()), int(conv_query_start.data_ptr()),
            int(q.data_ptr()), int(q_scale.data_ptr()),
            int(tau.data_ptr()) if has_tau else 0,
            int(q_out.data_ptr()),
            int(n_live_buf.data_ptr()) if has_nlive else 0,
            int(tree_sconv_anc.data_ptr()),
            T, int(k.stride(0)), int(v.stride(0)),
            int(k_conv_w.stride(0)), int(k_conv_w.stride(1)),
            int(k_dyn.stride(0)) if dynamic else 0,
            int(v_dyn.stride(0)) if dynamic else 0,
            int(conv_cache.stride(0)), int(conv_cache.stride(c_n_dim)),
            kv_blk_sb, 0, 256,
            int(conv_block_table.stride(0)), int(conv_block_table.shape[1]), int(conv_cache.shape[c_n_dim]),
            int(q.stride(0)),
            sconv_scale, float(clip_threshold), float(clip_eps), float(norm_eps),
            1 if dynamic else 0, 1 if has_tau else 0, 1 if has_nlive else 0, 1 if eff_pdl_prologue else 0,
            stream,
            k_eraw, k_eused, v_eraw, v_eused, ecap,
        )
        if rc != 0:
            raise RuntimeError(f"attn_prologue_kvq_tree_bf12_exact_v10_fwd failed: cudaError={rc}")
    else:
        rc = lib.attn_prologue_kvq_tree_exact_fwd(
            int(k.data_ptr()), int(v.data_ptr()),
            int(k_conv_w.data_ptr()), int(v_conv_w.data_ptr()),
            int(k_dyn.data_ptr()) if dynamic else 0,
            int(v_dyn.data_ptr()) if dynamic else 0,
            int(conv_cache.data_ptr()), int(key_cache.data_ptr()), int(value_cache.data_ptr()),
            int(kv_slot_mapping.data_ptr()), int(positions.data_ptr()),
            int(conv_seq_idx.data_ptr()), int(conv_slot_mapping.data_ptr()),
            int(conv_block_table.data_ptr()), int(conv_query_start.data_ptr()),
            int(q.data_ptr()), int(q_scale.data_ptr()),
            int(tau.data_ptr()) if has_tau else 0,
            int(q_out.data_ptr()),
            int(n_live_buf.data_ptr()) if has_nlive else 0,
            int(tree_sconv_anc.data_ptr()),
            T, int(k.stride(0)), int(v.stride(0)),
            int(k_conv_w.stride(0)), int(k_conv_w.stride(1)),
            int(k_dyn.stride(0)) if dynamic else 0,
            int(v_dyn.stride(0)) if dynamic else 0,
            int(conv_cache.stride(0)), int(conv_cache.stride(c_n_dim)),
            int(key_cache.stride(0)), int(key_cache.stride(1)), int(key_cache.shape[1]),
            int(conv_block_table.stride(0)), int(conv_block_table.shape[1]), int(conv_cache.shape[c_n_dim]),
            int(q.stride(0)),
            sconv_scale, float(clip_threshold), float(clip_eps), float(norm_eps),
            1 if dynamic else 0, 1 if has_tau else 0, 1 if has_nlive else 0, 1 if eff_pdl_prologue else 0,
            stream,
        )
        if rc != 0:
            raise RuntimeError(f"attn_prologue_kvq_tree_exact_fwd failed: cudaError={rc}")

    mtiles = 4
    op_buf, lp_buf, sem_buf = _next_scratch(q.device, num_splits, T, H, mtiles * Hkv)
    E = int(rel_proj_t.shape[0]) if rel_proj_t is not None else 0
    r_view = r.view(T, H, 16) if r.dim() == 2 else r

    if use_kv_bf12:
        rc = lib.fused_decode_attn_otf_mgs_tree_bf12_v10_fwd(
            int(q_out.data_ptr()),
            int(r_view.data_ptr()),
            int(r_scale.data_ptr()),
            int(tau.data_ptr()) if has_tau else 0,
            int(rel_proj_t.data_ptr()) if E > 0 else 0,
            kv_base_ptr,
            kv_base_ptr + KV_BF12_HALF_BYTES,
            int(attn_block_table.data_ptr()),
            int(attn_seq_lens.data_ptr()),
            int(attn_cu_seqlens_q.data_ptr()),
            int(gate.data_ptr()),
            int(sinks.data_ptr()),
            int(op_buf.data_ptr()),
            int(lp_buf.data_ptr()),
            int(sem_buf.data_ptr()),
            int(output.data_ptr()),
            int(n_live_buf.data_ptr()) if has_nlive else 0,
            int(tree_meta.data_ptr()),
            int(q_out.stride(0)), int(q_out.stride(1)),
            int(r_view.stride(0)), int(r_view.stride(1)),
            kv_blk_sb, 0, kv_head_sb,
            kv_blk_sb, 0, kv_head_sb,
            int(attn_block_table.stride(0)),
            int(gate.stride(0)), int(sinks.stride(0)), int(output.stride(0)),
            int(op_buf.stride(0)), int(op_buf.stride(1)), int(op_buf.stride(2)),
            int(lp_buf.stride(0)), int(lp_buf.stride(1)), int(lp_buf.stride(2)),
            float(softmax_scale * _LOG2E), float(rel_threshold), float(clip_eps),
            T, NT, mtiles, num_reqs, Hkv, 256, E, G,
            int(window), int(num_splits),
            1 if has_tau else 0, 1 if has_nlive else 0,
            1 if pdl else 0, 1 if trig_wo else 0, int(tree_mode),
            stream, k_eraw, v_eraw,
        )
    else:
        rc = lib.fused_decode_attn_otf_mgs_tree_fwd(
            int(q_out.data_ptr()),
            int(r_view.data_ptr()),
            int(r_scale.data_ptr()),
            int(tau.data_ptr()) if has_tau else 0,
            int(rel_proj_t.data_ptr()) if E > 0 else 0,
            int(key_cache.data_ptr()),
            int(value_cache.data_ptr()),
            int(attn_block_table.data_ptr()),
            int(attn_seq_lens.data_ptr()),
            int(attn_cu_seqlens_q.data_ptr()),
            int(gate.data_ptr()),
            int(sinks.data_ptr()),
            int(op_buf.data_ptr()),
            int(lp_buf.data_ptr()),
            int(sem_buf.data_ptr()),
            int(output.data_ptr()),
            int(n_live_buf.data_ptr()) if has_nlive else 0,
            int(tree_meta.data_ptr()),
            int(q_out.stride(0)), int(q_out.stride(1)),
            int(r_view.stride(0)), int(r_view.stride(1)),
            int(key_cache.stride(0)), int(key_cache.stride(1)), int(key_cache.stride(2)),
            int(value_cache.stride(0)), int(value_cache.stride(1)), int(value_cache.stride(2)),
            int(attn_block_table.stride(0)),
            int(gate.stride(0)), int(sinks.stride(0)), int(output.stride(0)),
            int(op_buf.stride(0)), int(op_buf.stride(1)), int(op_buf.stride(2)),
            int(lp_buf.stride(0)), int(lp_buf.stride(1)), int(lp_buf.stride(2)),
            float(softmax_scale * _LOG2E), float(rel_threshold), float(clip_eps),
            T, NT, mtiles, num_reqs, Hkv, int(key_cache.shape[1]), E, G,
            int(window), int(num_splits),
            1 if has_tau else 0, 1 if has_nlive else 0,
            1 if pdl else 0, 1 if trig_wo else 0, int(tree_mode),
            stream,
        )
    if rc != 0:
        raise RuntimeError(f"fused_decode_attn_otf_mgs_tree_fwd failed: cudaError={rc}")
    if trig_wo and pdl:
        from . import sm12_attn as _sm12_attn
        _sm12_attn._WO_ARMED = True
    return True
