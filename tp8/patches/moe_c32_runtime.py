# SPDX-License-Identifier: Apache-2.0
"""SM121a (GB10) C=32 (B=32, M=17..256) 0-ULP Bit-Exact Target 48L MoE Runtime.

Opt-in orchestration module for the C=32 3-library Target MoE stack (default OFF;
strictly guarded by `17 <= M <= 256` so C=1 `M <= 16` and `<120s` 1M prefill
`M > 256` are 100% untouched):

  1. `librouter_bf12_c32.so`:
     - `router_bf12_c32_gemm_fwd`: SMEM-tiled BF12 router GEMM reading existing
       `router_bf12` slabs (2.654 MB vs 3.539 MB BF16, -25.0% bytes, 0 extra VRAM).
     - `router_c32_quant_route_place_fwd`: 2-phase fused Zero-Scatter quantize +
       Top-8 + last-CTA 6-warp prefix scan + placement (`Row2Tok`, `TileCompactStart`,
       `Src2Compact`, pre-zeroed compact `RowAmaxW2`). Supports BOTH BF16 logits
       (from `k_group_partials` for literal `nd == 0` vs live `patches-s6` on
       `exact_ink.py` & `bench.py`) AND FP32 logits (from `router_bf12_c32_gemm_fwd`).
  2. `libnvfp4_ws5_c32.so`:
     - Native single-pass `BM=16` (`TILE=16`, `BK_DOT=256/32`) & `BM=32` (`TILE=32`,
       `BK_DOT=128/64` + tail zero-pad) 0-ULP WS5 NVFP4 W13/W2 kernels.
     - Dynamic `has_m8/has_m16/has_m24` row predication, `Row2Tok` Zero-Scatter,
       `WS_BASE` bit-7 `P_OFF` bypass, 0-way SMEM banking, Fused W13 SiTU -> compact
       `SiTU_Out` + W2 OTF strictly-unfused NVFP4 quant-dequant, and 6-panel-wave
       double-buffered compact L2-resident `DownPanel` combine -> `SendBuf`.
  3. `libmxfp8_mx7_c32.so`:
     - Native single-pass `BM=16` & `BM=32` 0-ULP BF16-MMA MX7 (7.28 bpw) W13/W2
       kernels (upgraded `BN=128, 4 warps` W2 with 1-to-1 zero-repack bijection
       onto existing MX7 slabs -> 2x less `sA` work, 4x more warps/SM than v4!).
     - Closes the live `INKLING_MX7_FREE_RAW=0` TILE=32 MX7 skip bug and shares the
       exact same 6-panel-wave L2-resident `DownPanel` combine -> `SendBuf`.

1-Knob Rollout Controls (ALL default OFF unless `INKLING_C32_MOE=1`):
  - `INKLING_C32_MOE=1`                      : master opt-in switch
  - `INKLING_C32_MOE_MIN_M=17`               : strict lower M guard (never touches M<=16)
  - `INKLING_C32_MOE_MAX_M=256`              : strict upper M guard (never touches prefill)
  - `INKLING_C32_ROUTER_MODE=route_only|bf12|off`
                                             : `route_only` (default, 0-ULP vs cuBLAS BF16)
                                               or `bf12` (-25% router weight bytes)
  - `INKLING_C32_ROUTER_BF12_CAPTURE_ONLY=1` : when `ROUTER_MODE=bf12`, keep eager prefill
                                               prompts on `route_only` (`exact_ink.py` 0-ULP!)
  - `INKLING_C32_FUSE_SITU=1`                : fuse W13 SiTU + W2 OTF quant (-48 launches)
  - `INKLING_C32_FUSE_COMBINE=1`             : 6-wave L2-resident fused combine (-48 launches)
  - `INKLING_C32_MX7_TILE32=1`               : enable MX7 at TILE=32 even when FREE_RAW=0
  - `INKLING_C32_PDL=2`                      : PDL mode (`0`, `1`, `2`)
"""

from __future__ import annotations

import ctypes
import os
import torch


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "", "off", "false", "no")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


C32_MASTER = _env_flag("INKLING_C32_MOE", False)
C32_MIN_M = max(17, _env_int("INKLING_C32_MOE_MIN_M", 17))
C32_MAX_M = min(256, _env_int("INKLING_C32_MOE_MAX_M", 256))

_raw_rmode = os.environ.get("INKLING_C32_ROUTER_MODE", "").strip().lower()
if _raw_rmode in ("route_only", "bf12", "off"):
    C32_ROUTER_MODE = _raw_rmode
else:
    C32_ROUTER_MODE = "route_only" if C32_MASTER else "off"

C32_ROUTER_BF12_CAPTURE_ONLY = _env_flag("INKLING_C32_ROUTER_BF12_CAPTURE_ONLY", True)
C32_FUSE_SITU = _env_flag("INKLING_C32_FUSE_SITU", C32_MASTER)
SHARED_MOE_SCRATCH = _env_flag(
    "INKLING_SHARED_MOE_SCRATCH",
    _env_flag("INKLING_FREE_PACKED_RAW_WEIGHTS", False),
)


def _parse_fuse_combine() -> int:
    raw = os.environ.get("INKLING_C32_FUSE_COMBINE")
    if raw is None or not raw.strip():
        return 1 if C32_MASTER else 0
    s = raw.strip().lower()
    if s in ("0", "off", "false", "no"):
        return 0
    if s in ("3", "slice", "slice_expert_major"):
        return 3
    if s in ("2", "panel", "panel_major"):
        return 2
    if s in ("1", "on", "true", "yes"):
        return 1
    try:
        v = int(s)
        return v if v in (0, 1, 2, 3) else (1 if v else 0)
    except ValueError:
        return 1 if C32_MASTER else 0


C32_FUSE_COMBINE: int = _parse_fuse_combine()
C32_FUSE_COMBINE_MAX_M: int = _env_int(
    "INKLING_C32_FUSE_COMBINE_MAX_M",
    48 if C32_FUSE_COMBINE == 2 else 256,
)
# Wave-V12 opt-in (default OFF): Direct MoE-W2 -> RDMA SendBuf (`FUSE_COMBINE=3` slice-expert-major)
# strictly scoped to C=3..8 (`M <= C32_FUSE_COMBINE_C18_MAX_M`, default 48) where 6 active panels stay
# 100% L2-resident (295..590 KiB), saving 48 combine launches + 85..170 MiB/step down_compact traffic
# while leaving C=16..32 (`M > 48`) untouched on FUSE_COMBINE=4 + combine_v6!
C32_FUSE_COMBINE_C18: int = max(0, min(3, _env_int("INKLING_C32_FUSE_COMBINE_C18", 0)))
C32_FUSE_COMBINE_C18_MAX_M: int = max(17, min(64, _env_int("INKLING_C32_FUSE_COMBINE_C18_MAX_M", 48)))
C32_MX7_TILE32 = _env_flag("INKLING_C32_MX7_TILE32", C32_MASTER)
C32_PDL = _env_int("INKLING_C32_PDL", 2)


def _parse_mx7_v5_mode() -> str:
    raw = os.environ.get("INKLING_MX7_V5", "").strip().lower()
    if raw in ("", "0", "off", "false", "no"):
        return "off"
    if raw in ("2", "panel", "panel_v5", "panel-v5"):
        return "panel"
    if raw in ("1", "compat", "hybrid", "compat_hybrid", "on", "true", "yes"):
        return "compat"
    return "off"


MX7_V5_MODE: str = _parse_mx7_v5_mode()
MX7_V5_THRESH: int = max(0, min(63, _env_int("INKLING_MX7_V5_THRESH", 32)))
NVFP4_V6: bool = _env_flag("INKLING_NVFP4_V6", False)
C32_COMBINE_V6: bool = _env_flag("INKLING_C32_COMBINE_V6", False)
C32_ROUTER_V6: bool = _env_flag("INKLING_C32_ROUTER_V6", False)
C48_MOE_V7: bool = _env_flag("INKLING_C48_MOE_V7", False)
C48_MOE_TILE8: bool = _env_flag("INKLING_C48_MOE_TILE8", False)
C48_MOE_MAX_M: int = max(17, min(64, _env_int("INKLING_C48_MOE_MAX_M", 64)))

# Wave-V8 C=2..8 MoE Overlap knobs (ALL default OFF -> 100% v7 invariant unless opted in!)
C48_MOE_V8: bool = _env_flag("INKLING_C48_MOE_V8", False)
C48_MOE_W13_W2_OVERLAP: bool = _env_flag("INKLING_C48_MOE_W13_W2_OVERLAP", C48_MOE_V8)
C48_ROUTER_PREQUANT_PDL: bool = _env_flag("INKLING_C48_ROUTER_PREQUANT_PDL", C48_MOE_V8)
C48_MOE_HOT_L2_HINT: bool = _env_flag("INKLING_C48_MOE_HOT_L2_HINT", C48_MOE_V8)
_C48_V8_ANY: bool = (
    C48_MOE_V8 or C48_MOE_W13_W2_OVERLAP or C48_ROUTER_PREQUANT_PDL or C48_MOE_HOT_L2_HINT
)

# Wave-V9 NVFP4 MoE & Router Roofline knobs (ALL default OFF -> 100% v8 invariant unless opted in!)
C48_MOE_V9: bool = _env_flag("INKLING_C48_MOE_V9", False)
NVFP4_V9_QDEQ: bool = _env_flag("INKLING_NVFP4_V9_QDEQ", C48_MOE_V9)
NVFP4_V9_W13_BN_MODE: int = max(0, min(2, _env_int("INKLING_NVFP4_V9_W13_BN_MODE", 2 if C48_MOE_V9 else 0)))
MX6_W13_BN_MODE: int = max(0, min(2, _env_int("INKLING_MX6_W13_BN_MODE", 0)))
NVFP4_V9_BN16_MARGIN: int = max(0, min(50, _env_int("INKLING_NVFP4_V9_BN16_MARGIN", 3)))
# Wave-V12 Forensic 0-ULP Default:
# - NVFP4 W2 OTF/QDeqA (`MOE_V9_QUANT_FMA`) defaults to 0 (`sub_rn`, matching SM121a cubin
#   `libnvfp4_ws5_c32.so` and exact Triton SiTU->FP4 quant!).
# - Router C32 (`ROUTER_C32_QUANT_FMA`) is DECOUPLED and defaults to 1 (`fmaf`, matching
#   `librouter_bf12_c32.so` & live `1dcfb306` exactness even when `MOE_V9_QUANT_FMA=0`!).
MOE_V9_QUANT_FMA: int = 1 if _env_flag("INKLING_MOE_V9_QUANT_FMA", False) else 0
_RAW_RT_QF = (
    os.environ.get("INKLING_ROUTER_C32_QUANT_FMA")
    if os.environ.get("INKLING_ROUTER_C32_QUANT_FMA") is not None
    else os.environ.get("INKLING_C32_ROUTER_QUANT_FMA")
)
ROUTER_C32_QUANT_FMA: int = (
    1
    if _RAW_RT_QF is None
    else (1 if _RAW_RT_QF.strip().lower() not in ("0", "", "off", "false", "no") else 0)
)
# Wave-V16 48L L2::evict_first + LPT Dynamic Tile Queue + W2 Pre-Wait Hoist (ALL default OFF -> 100% V15 invariant!)
NVFP4_C48_V16: bool = _env_flag("INKLING_NVFP4_C48_V16", False)
NVFP4_V16_EVICT: int = max(0, min(2, _env_int("INKLING_NVFP4_V16_EVICT", 1 if NVFP4_C48_V16 else 0)))
NVFP4_V16_LPT: int = max(0, min(2, _env_int("INKLING_NVFP4_V16_LPT", 2 if NVFP4_C48_V16 else 0)))
MX7_C48_V16: bool = _env_flag("INKLING_MX7_C48_V16", False)
MX7_V16_LPT: int = max(0, min(2, _env_int("INKLING_MX7_V16_LPT", 2 if MX7_C48_V16 else 0)))
# Wave-V12 0-ULP NVFP4 MoE knobs (ALL default OFF -> 100% V10 invariant unless opted in!)
C48_MOE_V12: bool = _env_flag("INKLING_C48_MOE_V12", NVFP4_C48_V16)
# Opt-in NVFP4-only direct W2->SendBuf fuse_combine override (default -1 = inherit C32_FUSE_COMBINE):
# Set INKLING_C32_NVFP4_FC_MODE=2 & INKLING_C32_NVFP4_FC_MAX_M=48 for zero-slice-tail Panel-Major
# direct W2->RDMA SendBuf at C=2..8 (M=17..48), leaving M>48 on FC=4 + combine_c32_v9 and MX7 untouched!
C32_NVFP4_FC_MODE: int = max(-1, min(3, _env_int("INKLING_C32_NVFP4_FC_MODE", -1)))
C32_NVFP4_FC_MAX_M: int = max(0, _env_int("INKLING_C32_NVFP4_FC_MAX_M", 48))
# Optional single-launch BN=16 cap for W13 TILE=8 (default 0 = disabled / use adaptive mode 2).
NVFP4_V12_BN16_SINGLE_MAX_M: int = max(0, _env_int("INKLING_NVFP4_V12_BN16_SINGLE_MAX_M", 0))
# Wave-V10 MoE + Single-Launch Fused Router knobs (ALL default OFF -> 100% V9 invariant!)
C48_MOE_V10: bool = _env_flag("INKLING_C48_MOE_V10", C48_MOE_V12)
ROUTER_V10_FUSED: bool = _env_flag("INKLING_ROUTER_V10_FUSED", C48_MOE_V10)
ROUTER_BF11: bool = _env_flag("INKLING_ROUTER_BF11", False)
NVFP4_V10_FUSE_QDEQ: bool = _env_flag("INKLING_NVFP4_V10_FUSE_QDEQ", C48_MOE_V10)
NVFP4_V10_BN16_MAX_M: int = _env_int("INKLING_NVFP4_V10_BN16_MAX_M", 48 if C48_MOE_V10 else 0)
_C48_V10_ANY: bool = C48_MOE_V12 or C48_MOE_V10 or ROUTER_V10_FUSED or NVFP4_V10_FUSE_QDEQ
_C48_V9_ANY: bool = C48_MOE_V9 or NVFP4_V9_QDEQ or _C48_V10_ANY
# Wave-V14 Global MXFP8 MX7 C=2..32 0-ULP Max-Occupancy knobs (ALL default OFF -> 100% V13 invariant!)
MX7_C48_V14: bool = _env_flag("INKLING_MX7_C48_V14", MX7_C48_V16)
MX7_V14_FUSE_QDEQ: bool = _env_flag("INKLING_MX7_V14_FUSE_QDEQ", MX7_C48_V14)
MX7_V14_W13_BN_MODE: int = max(0, min(2, _env_int("INKLING_MX7_V14_W13_BN_MODE", 2 if MX7_C48_V14 else 0)))
MX7_V14_BN16_MARGIN: int = max(0, min(50, _env_int("INKLING_MX7_V14_BN16_MARGIN", 3)))
MX7_V14_BN16_MAX_M: int = _env_int("INKLING_MX7_V14_BN16_MAX_M", 48 if MX7_C48_V14 else 0)
MX7_V14_EVICT: int = 1 if _env_flag("INKLING_MX7_V14_EVICT", MX7_C48_V14) else 0
# Opt-in MX7-only direct W2->SendBuf fuse_combine override (default -1 = inherit C32_FUSE_COMBINE):
C32_MX7_FC_MODE: int = max(-1, min(3, _env_int("INKLING_C32_MX7_FC_MODE", -1)))
C32_MX7_FC_MAX_M: int = max(0, _env_int("INKLING_C32_MX7_FC_MAX_M", 48))

_DIR = os.path.dirname(os.path.abspath(__file__))
_NVFP4_PATHS = (
    os.environ.get("INKLING_NVFP4_WS5_C32_LIB", "").strip(),
    *(
        (
            "/rdma/libnvfp4_ws5_c48_v16.so",
            os.path.join(_DIR, "libnvfp4_ws5_c48_v16.so"),
        )
        if NVFP4_C48_V16
        else ()
    ),
    *(
        (
            "/rdma/libnvfp4_ws5_c48_v12.so",
            os.path.join(_DIR, "libnvfp4_ws5_c48_v12.so"),
        )
        if C48_MOE_V12
        else ()
    ),
    *(
        (
            "/rdma/libnvfp4_ws5_c48_v10.so",
            os.path.join(_DIR, "libnvfp4_ws5_c48_v10.so"),
        )
        if (C48_MOE_V10 or NVFP4_V10_FUSE_QDEQ)
        else ()
    ),
    *(
        (
            "/rdma/libnvfp4_ws5_c48_v9.so",
            os.path.join(_DIR, "libnvfp4_ws5_c48_v9.so"),
        )
        if _C48_V9_ANY
        else ()
    ),
    *(
        (
            "/rdma/libnvfp4_ws5_c48_v8.so",
            os.path.join(_DIR, "libnvfp4_ws5_c48_v8.so"),
        )
        if _C48_V8_ANY
        else ()
    ),
    *(
        (
            "/rdma/libnvfp4_ws5_c48_v7.so",
            os.path.join(_DIR, "libnvfp4_ws5_c48_v7.so"),
        )
        if (C48_MOE_V7 or C48_MOE_TILE8 or _C48_V8_ANY)
        else ()
    ),
    *(
        (
            "/rdma/libnvfp4_ws5_c32_v6.so",
            os.path.join(_DIR, "libnvfp4_ws5_c32_v6.so"),
        )
        if (NVFP4_V6 or C32_COMBINE_V6 or C32_FUSE_COMBINE == 3)
        else ()
    ),
    "/rdma/libnvfp4_ws5_c32.so",
    os.path.join(_DIR, "libnvfp4_ws5_c32.so"),
)
_MX7_PATHS = (
    os.environ.get("INKLING_MX7_C32_LIB", "").strip(),
    *(
        (
            "/rdma/libmxfp8_mx7_c48_v16.so",
            os.path.join(_DIR, "libmxfp8_mx7_c48_v16.so"),
        )
        if MX7_C48_V16
        else ()
    ),
    *(
        (
            "/rdma/libmxfp8_mx7_c48_v14.so",
            os.path.join(_DIR, "libmxfp8_mx7_c48_v14.so"),
        )
        if (MX7_C48_V14 or MX7_V14_FUSE_QDEQ)
        else ()
    ),
    *(
        (
            "/rdma/libmxfp8_mx7_c48_v9.so",
            os.path.join(_DIR, "libmxfp8_mx7_c48_v9.so"),
        )
        if _C48_V9_ANY
        else ()
    ),
    *(
        (
            "/rdma/libmxfp8_mx7_c48_v8.so",
            os.path.join(_DIR, "libmxfp8_mx7_c48_v8.so"),
        )
        if _C48_V8_ANY
        else ()
    ),
    *(
        (
            "/rdma/libmxfp8_mx7_c48_v7.so",
            os.path.join(_DIR, "libmxfp8_mx7_c48_v7.so"),
        )
        if (C48_MOE_V7 or C48_MOE_TILE8 or _C48_V8_ANY)
        else ()
    ),
    *(
        (
            "/rdma/libmxfp8_mx7_c32_v5.so",
            os.path.join(_DIR, "libmxfp8_mx7_c32_v5.so"),
        )
        if MX7_V5_MODE != "off"
        else ()
    ),
    "/rdma/libmxfp8_mx7_c32.so",
    os.path.join(_DIR, "libmxfp8_mx7_c32.so"),
)
_ROUTER_PATHS = (
    os.environ.get("INKLING_ROUTER_BF12_C32_LIB", "").strip(),
    *(
        (
            "/rdma/librouter_bf12_fused_v10.so",
            os.path.join(_DIR, "librouter_bf12_fused_v10.so"),
        )
        if ROUTER_V10_FUSED
        else ()
    ),
    *(
        (
            "/rdma/librouter_bf12_c48_v9.so",
            os.path.join(_DIR, "librouter_bf12_c48_v9.so"),
        )
        if _C48_V9_ANY
        else ()
    ),
    *(
        (
            "/rdma/librouter_bf12_c48_v8.so",
            os.path.join(_DIR, "librouter_bf12_c48_v8.so"),
        )
        if _C48_V8_ANY
        else ()
    ),
    *(
        (
            "/rdma/librouter_bf12_c48_v7.so",
            os.path.join(_DIR, "librouter_bf12_c48_v7.so"),
        )
        if (C48_MOE_V7 or C48_MOE_TILE8 or _C48_V8_ANY)
        else ()
    ),
    *(
        (
            "/rdma/librouter_bf12_c32_v6.so",
            os.path.join(_DIR, "librouter_bf12_c32_v6.so"),
        )
        if (C32_ROUTER_V6 or NVFP4_V6)
        else ()
    ),
    "/rdma/librouter_bf12_c32.so",
    os.path.join(_DIR, "librouter_bf12_c32.so"),
)

_LIB_NVFP4: ctypes.CDLL | None = None
_LIB_MX7: ctypes.CDLL | None = None
_LIB_ROUTER: ctypes.CDLL | None = None
_LIB_NVFP4_VER: int = 0
_LIB_MX7_VER: int = 0
_LIB_ROUTER_VER: int = 0
_TRIED = {"nvfp4": False, "mx7": False, "router": False}

# Static per-device C32 workspace (7.62 MiB total across all 48L, CUDA-graph capture safe).
_WS_MAX_T = 256
_WS_MAX_ROUTED = _WS_MAX_T * 8   # 2048
_WS_MAX_CTAS = 512               # covers max_ctas <= 448 at M=256
_WS_MAX_ROWS = _WS_MAX_CTAS * 32  # 16384
_WS: dict[int, dict[str, torch.Tensor]] = {}


def _find_lib(paths: tuple[str, ...], symbol: str) -> ctypes.CDLL | None:
    for p in paths:
        if p and os.path.isfile(p):
            try:
                lib = ctypes.CDLL(p)
                if hasattr(lib, symbol):
                    return lib
            except OSError:
                continue
    return None


def get_nvfp4_lib(path: str | None = None) -> ctypes.CDLL | None:
    global _LIB_NVFP4
    if _LIB_NVFP4 is not None and path is None:
        return _LIB_NVFP4
    if _TRIED["nvfp4"] and path is None:
        return None
    if path is None:
        _TRIED["nvfp4"] = True
    paths = (path,) if path else _NVFP4_PATHS
    lib = _find_lib(paths, "nvfp4_w13_ws5_c32_fwd")
    if lib is None:
        return None
    # Match nvfp4_ws5_c32.cu extern "C" signatures 1-to-1
    lib.nvfp4_w13_ws5_c32_fwd.argtypes = [
        ctypes.c_void_p,  # A
        ctypes.c_void_p,  # AS
        ctypes.c_void_p,  # RS
        ctypes.c_void_p,  # Row2Tok
        ctypes.c_void_p,  # SLABS
        ctypes.c_void_p,  # WS_BASE
        ctypes.c_void_p,  # P_OFF
        ctypes.c_void_p,  # PATCHES
        ctypes.c_void_p,  # Ones
        ctypes.c_void_p,  # Tiles
        ctypes.c_void_p,  # TileCompactStart
        ctypes.c_void_p,  # Gammas
        ctypes.c_void_p,  # RowAmaxW2
        ctypes.c_void_p,  # Out
        ctypes.c_int,     # N
        ctypes.c_int,     # K
        ctypes.c_int,     # MAX_CTAS
        ctypes.c_int,     # TILE
        ctypes.c_int,     # bk_dot
        ctypes.c_int,     # zero_scatter
        ctypes.c_int,     # as_row_major
        ctypes.c_int,     # fuse_situ
        ctypes.c_int,     # pdl_wait_mode
        ctypes.c_int,     # trig_dep
        ctypes.c_uint64,  # stream_ptr
    ]
    lib.nvfp4_w13_ws5_c32_fwd.restype = ctypes.c_int

    lib.nvfp4_w2_ws5_c32_fwd.argtypes = [
        ctypes.c_void_p,  # A
        ctypes.c_void_p,  # AS
        ctypes.c_void_p,  # RS
        ctypes.c_void_p,  # RowAmaxW2
        ctypes.c_void_p,  # WAmax
        ctypes.c_void_p,  # WDenominator
        ctypes.c_void_p,  # W_PACK
        ctypes.c_void_p,  # WS5_PACK
        ctypes.c_void_p,  # WS_BASE
        ctypes.c_void_p,  # P_OFF
        ctypes.c_void_p,  # PATCHES
        ctypes.c_void_p,  # Ones
        ctypes.c_void_p,  # Tiles
        ctypes.c_void_p,  # TileCompactStart
        ctypes.c_void_p,  # Out
        ctypes.c_void_p,  # Src2Compact
        ctypes.c_void_p,  # NLivePtr
        ctypes.c_void_p,  # PanelDone
        ctypes.c_void_p,  # BufGen
        ctypes.c_void_p,  # SendBuf
        ctypes.c_int,     # N
        ctypes.c_int,     # K
        ctypes.c_int,     # MAX_CTAS
        ctypes.c_int,     # TILE
        ctypes.c_int,     # T
        ctypes.c_int,     # bk_dot
        ctypes.c_int,     # as_row_major
        ctypes.c_int,     # as_row_stride
        ctypes.c_int,     # down_row_stride
        ctypes.c_int,     # fuse_quant
        ctypes.c_int,     # fuse_combine
        ctypes.c_int,     # pdl_wait_mode
        ctypes.c_int,     # trig_dep
        ctypes.c_uint64,  # stream_ptr
    ]
    lib.nvfp4_w2_ws5_c32_fwd.restype = ctypes.c_int
    if hasattr(lib, "nvfp4_combine_c32_v6_fwd"):
        lib.nvfp4_combine_c32_v6_fwd.argtypes = [
            ctypes.c_void_p,  # Down
            ctypes.c_void_p,  # Src2Row
            ctypes.c_void_p,  # NLivePtr
            ctypes.c_void_p,  # Out
            ctypes.c_int,     # T
            ctypes.c_int,     # H
            ctypes.c_int,     # pdl_wait
            ctypes.c_int,     # trig_dep
            ctypes.c_uint64,  # stream_ptr
        ]
        lib.nvfp4_combine_c32_v6_fwd.restype = ctypes.c_int
    if hasattr(lib, "nvfp4_ws5_c32_w13_min_cta"):
        lib.nvfp4_ws5_c32_w13_min_cta.argtypes = []
        lib.nvfp4_ws5_c32_w13_min_cta.restype = ctypes.c_int
    if hasattr(lib, "nvfp4_ws5_c48_min_cta"):
        lib.nvfp4_ws5_c48_min_cta.argtypes = []
        lib.nvfp4_ws5_c48_min_cta.restype = ctypes.c_int
    if hasattr(lib, "nvfp4_w13_ws5_c48_v8_fwd"):
        lib.nvfp4_w13_ws5_c48_v8_fwd.argtypes = [
            ctypes.c_void_p,  # A
            ctypes.c_void_p,  # AS
            ctypes.c_void_p,  # RS
            ctypes.c_void_p,  # Row2Tok
            ctypes.c_void_p,  # SLABS
            ctypes.c_void_p,  # WS_BASE
            ctypes.c_void_p,  # P_OFF
            ctypes.c_void_p,  # PATCHES
            ctypes.c_void_p,  # Ones
            ctypes.c_void_p,  # Tiles
            ctypes.c_void_p,  # TileCompactStart
            ctypes.c_void_p,  # Gammas
            ctypes.c_void_p,  # RowAmaxW2
            ctypes.c_void_p,  # Out
            ctypes.c_int,     # N
            ctypes.c_int,     # K
            ctypes.c_int,     # MAX_CTAS
            ctypes.c_int,     # TILE
            ctypes.c_int,     # bk_dot
            ctypes.c_int,     # zero_scatter
            ctypes.c_int,     # as_row_major
            ctypes.c_int,     # fuse_situ
            ctypes.c_int,     # pdl_wait_mode
            ctypes.c_int,     # trig_dep
            ctypes.c_void_p,  # TileW13Done
            ctypes.c_void_p,  # TileReady
            ctypes.c_void_p,  # OverlapMeta
            ctypes.c_int,     # overlap_w13_w2
            ctypes.c_int,     # hot_l2_hint
            ctypes.c_uint64,  # stream_ptr
        ]
        lib.nvfp4_w13_ws5_c48_v8_fwd.restype = ctypes.c_int
    if hasattr(lib, "nvfp4_w2_ws5_c48_v8_fwd"):
        lib.nvfp4_w2_ws5_c48_v8_fwd.argtypes = [
            ctypes.c_void_p,  # A
            ctypes.c_void_p,  # AS
            ctypes.c_void_p,  # RS
            ctypes.c_void_p,  # RowAmaxW2
            ctypes.c_void_p,  # WAmax
            ctypes.c_void_p,  # WDenominator
            ctypes.c_void_p,  # W_PACK
            ctypes.c_void_p,  # WS5_PACK
            ctypes.c_void_p,  # WS_BASE
            ctypes.c_void_p,  # P_OFF
            ctypes.c_void_p,  # PATCHES
            ctypes.c_void_p,  # Ones
            ctypes.c_void_p,  # Tiles
            ctypes.c_void_p,  # TileCompactStart
            ctypes.c_void_p,  # Out
            ctypes.c_void_p,  # Src2Compact
            ctypes.c_void_p,  # NLivePtr
            ctypes.c_void_p,  # PanelDone
            ctypes.c_void_p,  # BufGen
            ctypes.c_void_p,  # SendBuf
            ctypes.c_int,     # N
            ctypes.c_int,     # K
            ctypes.c_int,     # MAX_CTAS
            ctypes.c_int,     # TILE
            ctypes.c_int,     # T
            ctypes.c_int,     # bk_dot
            ctypes.c_int,     # as_row_major
            ctypes.c_int,     # as_row_stride
            ctypes.c_int,     # down_row_stride
            ctypes.c_int,     # fuse_quant
            ctypes.c_int,     # fuse_combine
            ctypes.c_int,     # pdl_wait_mode
            ctypes.c_int,     # trig_dep
            ctypes.c_void_p,  # TileReady
            ctypes.c_void_p,  # OverlapMeta
            ctypes.c_int,     # overlap_w13_w2
            ctypes.c_int,     # hot_l2_hint
            ctypes.c_uint64,  # stream_ptr
        ]
        lib.nvfp4_w2_ws5_c48_v8_fwd.restype = ctypes.c_int
    if hasattr(lib, "nvfp4_ws5_c48_v9_bn16_min_cta"):
        lib.nvfp4_ws5_c48_v9_bn16_min_cta.argtypes = []
        lib.nvfp4_ws5_c48_v9_bn16_min_cta.restype = ctypes.c_int
    if hasattr(lib, "nvfp4_ws5_c48_v16_w2_min_cta"):
        lib.nvfp4_ws5_c48_v16_w2_min_cta.argtypes = []
        lib.nvfp4_ws5_c48_v16_w2_min_cta.restype = ctypes.c_int
    if hasattr(lib, "nvfp4_w13_ws5_c48_v16_fwd"):
        lib.nvfp4_w13_ws5_c48_v16_fwd.argtypes = [
            *[ctypes.c_void_p] * 15,
            *[ctypes.c_int] * 17,
            ctypes.c_void_p,
            ctypes.c_uint64,
        ]
        lib.nvfp4_w13_ws5_c48_v16_fwd.restype = ctypes.c_int
    if hasattr(lib, "nvfp4_w2_ws5_c48_v16_fwd"):
        lib.nvfp4_w2_ws5_c48_v16_fwd.argtypes = [
            *[ctypes.c_void_p] * 20,
            *[ctypes.c_int] * 16,
            ctypes.c_void_p,
            ctypes.c_uint64,
        ]
        lib.nvfp4_w2_ws5_c48_v16_fwd.restype = ctypes.c_int
    if hasattr(lib, "nvfp4_w13_ws5_c48_v10_fwd"):
        lib.nvfp4_w13_ws5_c48_v10_fwd.argtypes = [
            ctypes.c_void_p,  # A
            ctypes.c_void_p,  # AS
            ctypes.c_void_p,  # RS
            ctypes.c_void_p,  # Row2Tok
            ctypes.c_void_p,  # SLABS
            ctypes.c_void_p,  # WS_BASE
            ctypes.c_void_p,  # P_OFF
            ctypes.c_void_p,  # PATCHES
            ctypes.c_void_p,  # Ones
            ctypes.c_void_p,  # Tiles
            ctypes.c_void_p,  # TileCompactStart
            ctypes.c_void_p,  # Gammas
            ctypes.c_void_p,  # RowAmaxW2
            ctypes.c_void_p,  # Out
            ctypes.c_void_p,  # TileQdeqDone
            ctypes.c_int,     # N
            ctypes.c_int,     # K
            ctypes.c_int,     # MAX_CTAS
            ctypes.c_int,     # TILE
            ctypes.c_int,     # bk_dot
            ctypes.c_int,     # zero_scatter
            ctypes.c_int,     # as_row_major
            ctypes.c_int,     # fuse_situ
            ctypes.c_int,     # skip_amax
            ctypes.c_int,     # bn_mode
            ctypes.c_int,     # bn16_margin_pct
            ctypes.c_int,     # pdl_wait_mode
            ctypes.c_int,     # trig_dep
            ctypes.c_int,     # fuse_qdeq
            ctypes.c_int,     # quant_fma
            ctypes.c_uint64,  # stream_ptr
        ]
        lib.nvfp4_w13_ws5_c48_v10_fwd.restype = ctypes.c_int
    if hasattr(lib, "nvfp4_w13_ws5_c48_v9_fwd"):
        lib.nvfp4_w13_ws5_c48_v9_fwd.argtypes = [
            ctypes.c_void_p,  # A
            ctypes.c_void_p,  # AS
            ctypes.c_void_p,  # RS
            ctypes.c_void_p,  # Row2Tok
            ctypes.c_void_p,  # SLABS
            ctypes.c_void_p,  # WS_BASE
            ctypes.c_void_p,  # P_OFF
            ctypes.c_void_p,  # PATCHES
            ctypes.c_void_p,  # Ones
            ctypes.c_void_p,  # Tiles
            ctypes.c_void_p,  # TileCompactStart
            ctypes.c_void_p,  # Gammas
            ctypes.c_void_p,  # RowAmaxW2
            ctypes.c_void_p,  # Out
            ctypes.c_int,     # N
            ctypes.c_int,     # K
            ctypes.c_int,     # MAX_CTAS
            ctypes.c_int,     # TILE
            ctypes.c_int,     # bk_dot
            ctypes.c_int,     # zero_scatter
            ctypes.c_int,     # as_row_major
            ctypes.c_int,     # fuse_situ
            ctypes.c_int,     # skip_amax
            ctypes.c_int,     # bn_mode
            ctypes.c_int,     # bn16_margin_pct
            ctypes.c_int,     # pdl_wait_mode
            ctypes.c_int,     # trig_dep
            ctypes.c_uint64,  # stream_ptr
        ]
        lib.nvfp4_w13_ws5_c48_v9_fwd.restype = ctypes.c_int
    if hasattr(lib, "nvfp4_situ_quant_dequant_c48_v9_fwd"):
        lib.nvfp4_situ_quant_dequant_c48_v9_fwd.argtypes = [
            ctypes.c_void_p,  # SiTU_In
            ctypes.c_void_p,  # QDeqOut
            ctypes.c_void_p,  # RowAmaxW2
            ctypes.c_void_p,  # Tiles
            ctypes.c_void_p,  # TileCompactStart
            ctypes.c_int,     # MAX_CTAS
            ctypes.c_int,     # TILE
            ctypes.c_int,     # max_compact_rows
            ctypes.c_int,     # pdl_wait
            ctypes.c_int,     # trig_dep
            ctypes.c_int,     # quant_fma
            ctypes.c_uint64,  # stream_ptr
        ]
        lib.nvfp4_situ_quant_dequant_c48_v9_fwd.restype = ctypes.c_int
    if hasattr(lib, "nvfp4_w2_ws5_c48_v9_fwd"):
        lib.nvfp4_w2_ws5_c48_v9_fwd.argtypes = [
            ctypes.c_void_p,  # A
            ctypes.c_void_p,  # AS
            ctypes.c_void_p,  # RS
            ctypes.c_void_p,  # RowAmaxW2
            ctypes.c_void_p,  # WAmax
            ctypes.c_void_p,  # WDenominator
            ctypes.c_void_p,  # W_PACK
            ctypes.c_void_p,  # WS5_PACK
            ctypes.c_void_p,  # WS_BASE
            ctypes.c_void_p,  # P_OFF
            ctypes.c_void_p,  # PATCHES
            ctypes.c_void_p,  # Ones
            ctypes.c_void_p,  # Tiles
            ctypes.c_void_p,  # TileCompactStart
            ctypes.c_void_p,  # Out
            ctypes.c_void_p,  # Src2Compact
            ctypes.c_void_p,  # NLivePtr
            ctypes.c_void_p,  # PanelDone
            ctypes.c_void_p,  # BufGen
            ctypes.c_void_p,  # SendBuf
            ctypes.c_int,     # N
            ctypes.c_int,     # K
            ctypes.c_int,     # MAX_CTAS
            ctypes.c_int,     # TILE
            ctypes.c_int,     # T
            ctypes.c_int,     # bk_dot
            ctypes.c_int,     # as_row_major
            ctypes.c_int,     # as_row_stride
            ctypes.c_int,     # down_row_stride
            ctypes.c_int,     # fuse_quant_mode
            ctypes.c_int,     # fuse_combine
            ctypes.c_int,     # pdl_wait_mode
            ctypes.c_int,     # trig_dep
            ctypes.c_int,     # quant_fma
            ctypes.c_uint64,  # stream_ptr
        ]
        lib.nvfp4_w2_ws5_c48_v9_fwd.restype = ctypes.c_int
    if hasattr(lib, "nvfp4_combine_c32_v9_fwd"):
        lib.nvfp4_combine_c32_v9_fwd.argtypes = [
            ctypes.c_void_p,  # Down
            ctypes.c_void_p,  # Src2Row
            ctypes.c_void_p,  # NLivePtr
            ctypes.c_void_p,  # Out
            ctypes.c_int,     # T
            ctypes.c_int,     # H
            ctypes.c_int,     # pdl_wait
            ctypes.c_int,     # trig_dep
            ctypes.c_uint64,  # stream_ptr
        ]
        lib.nvfp4_combine_c32_v9_fwd.restype = ctypes.c_int
    ver = 3
    if hasattr(lib, "nvfp4_ws5_c32_version"):
        lib.nvfp4_ws5_c32_version.argtypes = []
        lib.nvfp4_ws5_c32_version.restype = ctypes.c_int
        ver = int(lib.nvfp4_ws5_c32_version())
    if path is None:
        global _LIB_NVFP4_VER
        _LIB_NVFP4 = lib
        _LIB_NVFP4_VER = ver
    return lib


def get_nvfp4_version() -> int:
    if _LIB_NVFP4 is None and not _TRIED["nvfp4"]:
        get_nvfp4_lib()
    return _LIB_NVFP4_VER


def supports_nvfp4_t32_bk256() -> bool:
    return C32_MASTER and get_nvfp4_version() >= 4


def get_mx6_lib() -> ctypes.CDLL | None:
    """Return shared 0-ULP MX6-Huff7 v10 CUDA library when INKLING_MX6_EXACT=1."""
    from . import trtllm_bmm
    if not getattr(trtllm_bmm, "_MX6_ENABLED", False):
        return None
    return trtllm_bmm._get_mx6_lib()


def supports_c48_tile8(nvfp4: bool = True) -> bool:
    if not (C32_MASTER and C48_MOE_TILE8):
        return False
    if nvfp4:
        return get_nvfp4_version() >= 7
    if get_mx6_lib() is not None:
        return True
    return get_mx7_version() >= 7


def get_mx7_lib(path: str | None = None) -> ctypes.CDLL | None:
    global _LIB_MX7
    if _LIB_MX7 is not None and path is None:
        return _LIB_MX7
    if _TRIED["mx7"] and path is None:
        return None
    if path is None:
        _TRIED["mx7"] = True
    paths = (path,) if path else _MX7_PATHS
    lib = _find_lib(paths, "mx7_w13_c32_fwd")
    if lib is None:
        return None
    # Match mxfp8_mx7_c32.cu extern "C" signatures 1-to-1
    if hasattr(lib, "mx7_v2_count_flips"):
        lib.mx7_v2_count_flips.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
        ]
        lib.mx7_v2_count_flips.restype = ctypes.c_int
    if hasattr(lib, "mx7_v2_pack"):
        lib.mx7_v2_pack.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_uint32, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
        ]
        lib.mx7_v2_pack.restype = ctypes.c_int

    lib.mx7_w13_c32_fwd.argtypes = [
        ctypes.c_void_p,  # A
        ctypes.c_void_p,  # AS
        ctypes.c_void_p,  # Row2Tok
        ctypes.c_void_p,  # SLABS
        ctypes.c_void_p,  # WS3
        ctypes.c_void_p,  # FLIPS
        ctypes.c_void_p,  # F_OFF
        ctypes.c_void_p,  # Ones
        ctypes.c_void_p,  # Tiles
        ctypes.c_void_p,  # TileCompactStart
        ctypes.c_void_p,  # Gammas
        ctypes.c_void_p,  # Out
        ctypes.c_int,     # N
        ctypes.c_int,     # K
        ctypes.c_int,     # MAX_CTAS
        ctypes.c_int,     # TILE
        ctypes.c_uint32,  # base_ws
        ctypes.c_int,     # zero_scatter
        ctypes.c_int,     # as_row_major
        ctypes.c_int,     # fuse_situ
        ctypes.c_int,     # pdl_wait_mode
        ctypes.c_int,     # trig_dep
        ctypes.c_uint64,  # stream_ptr
    ]
    lib.mx7_w13_c32_fwd.restype = ctypes.c_int

    lib.mx7_w2_c32_fwd.argtypes = [
        ctypes.c_void_p,  # A
        ctypes.c_void_p,  # AS
        ctypes.c_void_p,  # SLABS
        ctypes.c_void_p,  # WS3
        ctypes.c_void_p,  # FLIPS
        ctypes.c_void_p,  # F_OFF
        ctypes.c_void_p,  # Ones
        ctypes.c_void_p,  # Tiles
        ctypes.c_void_p,  # TileCompactStart
        ctypes.c_void_p,  # Out
        ctypes.c_void_p,  # Src2Compact
        ctypes.c_void_p,  # NLivePtr
        ctypes.c_void_p,  # PanelDone
        ctypes.c_void_p,  # BufGen
        ctypes.c_void_p,  # SendBuf
        ctypes.c_int,     # N
        ctypes.c_int,     # K
        ctypes.c_int,     # MAX_CTAS
        ctypes.c_int,     # TILE
        ctypes.c_int,     # T
        ctypes.c_uint32,  # base_ws
        ctypes.c_int,     # as_row_major
        ctypes.c_int,     # as_row_stride
        ctypes.c_int,     # down_row_stride
        ctypes.c_int,     # fuse_quant
        ctypes.c_int,     # fuse_combine
        ctypes.c_int,     # pdl_wait_mode
        ctypes.c_int,     # trig_dep
        ctypes.c_uint64,  # stream_ptr
    ]
    lib.mx7_w2_c32_fwd.restype = ctypes.c_int
    if hasattr(lib, "mx7_v5_count_hybrid_from_mx7"):
        lib.mx7_v5_count_hybrid_from_mx7.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
        ]
        lib.mx7_v5_count_hybrid_from_mx7.restype = ctypes.c_int
    if hasattr(lib, "mx7_v5_build_hybrid_from_mx7"):
        lib.mx7_v5_build_hybrid_from_mx7.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
        ]
        lib.mx7_v5_build_hybrid_from_mx7.restype = ctypes.c_int
    if hasattr(lib, "mx7_v5_panel_measure_from_mx7"):
        lib.mx7_v5_panel_measure_from_mx7.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
        ]
        lib.mx7_v5_panel_measure_from_mx7.restype = ctypes.c_int
    if hasattr(lib, "mx7_v5_panel_build_from_mx7"):
        lib.mx7_v5_panel_build_from_mx7.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
        ]
        lib.mx7_v5_panel_build_from_mx7.restype = ctypes.c_int
    if hasattr(lib, "mx7_w13_c32_v5_fwd"):
        lib.mx7_w13_c32_v5_fwd.argtypes = [
            ctypes.c_void_p,  # A
            ctypes.c_void_p,  # AS
            ctypes.c_void_p,  # Row2Tok
            ctypes.c_void_p,  # SLABS
            ctypes.c_void_p,  # WS3
            ctypes.c_void_p,  # SLAB_META
            ctypes.c_void_p,  # S_OFF
            ctypes.c_void_p,  # D_OFF
            ctypes.c_void_p,  # SFLIPS
            ctypes.c_void_p,  # BMASKS
            ctypes.c_void_p,  # Ones
            ctypes.c_void_p,  # Tiles
            ctypes.c_void_p,  # TileCompactStart
            ctypes.c_void_p,  # Gammas
            ctypes.c_void_p,  # Out
            ctypes.c_int,     # N
            ctypes.c_int,     # K
            ctypes.c_int,     # MAX_CTAS
            ctypes.c_int,     # TILE
            ctypes.c_uint32,  # base_ws
            ctypes.c_int,     # zero_scatter
            ctypes.c_int,     # as_row_major
            ctypes.c_int,     # fuse_situ
            ctypes.c_int,     # pdl_wait_mode
            ctypes.c_int,     # trig_dep
            ctypes.c_uint64,  # stream_ptr
        ]
        lib.mx7_w13_c32_v5_fwd.restype = ctypes.c_int
    if hasattr(lib, "mx7_w2_c32_v5_fwd"):
        lib.mx7_w2_c32_v5_fwd.argtypes = [
            ctypes.c_void_p,  # A
            ctypes.c_void_p,  # AS
            ctypes.c_void_p,  # SLABS
            ctypes.c_void_p,  # WS3
            ctypes.c_void_p,  # SLAB_META
            ctypes.c_void_p,  # S_OFF
            ctypes.c_void_p,  # D_OFF
            ctypes.c_void_p,  # SFLIPS
            ctypes.c_void_p,  # BMASKS
            ctypes.c_void_p,  # Ones
            ctypes.c_void_p,  # Tiles
            ctypes.c_void_p,  # TileCompactStart
            ctypes.c_void_p,  # Out
            ctypes.c_void_p,  # Src2Compact
            ctypes.c_void_p,  # NLivePtr
            ctypes.c_void_p,  # PanelDone
            ctypes.c_void_p,  # BufGen
            ctypes.c_void_p,  # SendBuf
            ctypes.c_int,     # N
            ctypes.c_int,     # K
            ctypes.c_int,     # MAX_CTAS
            ctypes.c_int,     # TILE
            ctypes.c_int,     # T
            ctypes.c_uint32,  # base_ws
            ctypes.c_int,     # as_row_major
            ctypes.c_int,     # as_row_stride
            ctypes.c_int,     # down_row_stride
            ctypes.c_int,     # fuse_quant
            ctypes.c_int,     # fuse_combine
            ctypes.c_int,     # pdl_wait_mode
            ctypes.c_int,     # trig_dep
            ctypes.c_uint64,  # stream_ptr
        ]
        lib.mx7_w2_c32_v5_fwd.restype = ctypes.c_int
    if hasattr(lib, "mx7_w13_c32_v5_panel_fwd"):
        lib.mx7_w13_c32_v5_panel_fwd.argtypes = [
            ctypes.c_void_p,  # A
            ctypes.c_void_p,  # AS
            ctypes.c_void_p,  # Row2Tok
            ctypes.c_void_p,  # PANEL_POOL
            ctypes.c_void_p,  # PANEL_OFF
            ctypes.c_void_p,  # SLAB_META
            ctypes.c_void_p,  # S_OFF
            ctypes.c_void_p,  # SFLIPS
            ctypes.c_void_p,  # Ones
            ctypes.c_void_p,  # Tiles
            ctypes.c_void_p,  # TileCompactStart
            ctypes.c_void_p,  # Gammas
            ctypes.c_void_p,  # Out
            ctypes.c_int,     # N
            ctypes.c_int,     # K
            ctypes.c_int,     # MAX_CTAS
            ctypes.c_int,     # TILE
            ctypes.c_uint32,  # base_ws
            ctypes.c_int,     # zero_scatter
            ctypes.c_int,     # as_row_major
            ctypes.c_int,     # fuse_situ
            ctypes.c_int,     # pdl_wait_mode
            ctypes.c_int,     # trig_dep
            ctypes.c_uint64,  # stream_ptr
        ]
        lib.mx7_w13_c32_v5_panel_fwd.restype = ctypes.c_int
    if hasattr(lib, "mx7_w2_c32_v5_panel_fwd"):
        lib.mx7_w2_c32_v5_panel_fwd.argtypes = [
            ctypes.c_void_p,  # A
            ctypes.c_void_p,  # AS
            ctypes.c_void_p,  # PANEL_POOL
            ctypes.c_void_p,  # PANEL_OFF
            ctypes.c_void_p,  # SLAB_META
            ctypes.c_void_p,  # S_OFF
            ctypes.c_void_p,  # SFLIPS
            ctypes.c_void_p,  # Ones
            ctypes.c_void_p,  # Tiles
            ctypes.c_void_p,  # TileCompactStart
            ctypes.c_void_p,  # Out
            ctypes.c_void_p,  # Src2Compact
            ctypes.c_void_p,  # NLivePtr
            ctypes.c_void_p,  # PanelDone
            ctypes.c_void_p,  # BufGen
            ctypes.c_void_p,  # SendBuf
            ctypes.c_int,     # N
            ctypes.c_int,     # K
            ctypes.c_int,     # MAX_CTAS
            ctypes.c_int,     # TILE
            ctypes.c_int,     # T
            ctypes.c_uint32,  # base_ws
            ctypes.c_int,     # as_row_major
            ctypes.c_int,     # as_row_stride
            ctypes.c_int,     # down_row_stride
            ctypes.c_int,     # fuse_quant
            ctypes.c_int,     # fuse_combine
            ctypes.c_int,     # pdl_wait_mode
            ctypes.c_int,     # trig_dep
            ctypes.c_uint64,  # stream_ptr
        ]
        lib.mx7_w2_c32_v5_panel_fwd.restype = ctypes.c_int
    if hasattr(lib, "mx7_c48_min_cta"):
        lib.mx7_c48_min_cta.argtypes = []
        lib.mx7_c48_min_cta.restype = ctypes.c_int
    if hasattr(lib, "mx7_combine_c48_v7_fwd"):
        lib.mx7_combine_c48_v7_fwd.argtypes = [
            ctypes.c_void_p,  # Down
            ctypes.c_void_p,  # Src2Row
            ctypes.c_void_p,  # NLivePtr
            ctypes.c_void_p,  # Out
            ctypes.c_int,     # T
            ctypes.c_int,     # H
            ctypes.c_int,     # pdl_wait
            ctypes.c_int,     # trig_dep
            ctypes.c_uint64,  # stream_ptr
        ]
        lib.mx7_combine_c48_v7_fwd.restype = ctypes.c_int
    if hasattr(lib, "mx7_w13_c48_v8_fwd"):
        lib.mx7_w13_c48_v8_fwd.argtypes = [
            ctypes.c_void_p,  # A
            ctypes.c_void_p,  # AS
            ctypes.c_void_p,  # Row2Tok
            ctypes.c_void_p,  # SLABS
            ctypes.c_void_p,  # WS3
            ctypes.c_void_p,  # FLIPS
            ctypes.c_void_p,  # F_OFF
            ctypes.c_void_p,  # Ones
            ctypes.c_void_p,  # Tiles
            ctypes.c_void_p,  # TileCompactStart
            ctypes.c_void_p,  # Gammas
            ctypes.c_void_p,  # Out
            ctypes.c_int,     # N
            ctypes.c_int,     # K
            ctypes.c_int,     # MAX_CTAS
            ctypes.c_int,     # TILE
            ctypes.c_uint32,  # base_ws
            ctypes.c_int,     # zero_scatter
            ctypes.c_int,     # as_row_major
            ctypes.c_int,     # fuse_situ
            ctypes.c_int,     # pdl_wait_mode
            ctypes.c_int,     # trig_dep
            ctypes.c_void_p,  # TileW13Done
            ctypes.c_void_p,  # TileReady
            ctypes.c_void_p,  # OverlapMeta
            ctypes.c_int,     # overlap_w13_w2
            ctypes.c_int,     # hot_l2_hint
            ctypes.c_uint64,  # stream_ptr
        ]
        lib.mx7_w13_c48_v8_fwd.restype = ctypes.c_int
    if hasattr(lib, "mx7_w2_c48_v8_fwd"):
        lib.mx7_w2_c48_v8_fwd.argtypes = [
            ctypes.c_void_p,  # A
            ctypes.c_void_p,  # AS
            ctypes.c_void_p,  # SLABS
            ctypes.c_void_p,  # WS3
            ctypes.c_void_p,  # FLIPS
            ctypes.c_void_p,  # F_OFF
            ctypes.c_void_p,  # Ones
            ctypes.c_void_p,  # Tiles
            ctypes.c_void_p,  # TileCompactStart
            ctypes.c_void_p,  # Out
            ctypes.c_void_p,  # Src2Compact
            ctypes.c_void_p,  # NLivePtr
            ctypes.c_void_p,  # PanelDone
            ctypes.c_void_p,  # BufGen
            ctypes.c_void_p,  # SendBuf
            ctypes.c_int,     # N
            ctypes.c_int,     # K
            ctypes.c_int,     # MAX_CTAS
            ctypes.c_int,     # TILE
            ctypes.c_int,     # T
            ctypes.c_uint32,  # base_ws
            ctypes.c_int,     # as_row_major
            ctypes.c_int,     # as_row_stride
            ctypes.c_int,     # down_row_stride
            ctypes.c_int,     # fuse_quant
            ctypes.c_int,     # fuse_combine
            ctypes.c_int,     # pdl_wait_mode
            ctypes.c_int,     # trig_dep
            ctypes.c_void_p,  # TileReady
            ctypes.c_void_p,  # OverlapMeta
            ctypes.c_int,     # overlap_w13_w2
            ctypes.c_int,     # hot_l2_hint
            ctypes.c_uint64,  # stream_ptr
        ]
        lib.mx7_w2_c48_v8_fwd.restype = ctypes.c_int
    if hasattr(lib, "mx7_w13_c48_v8_compat_fwd"):
        lib.mx7_w13_c48_v8_compat_fwd.argtypes = [
            ctypes.c_void_p,  # A
            ctypes.c_void_p,  # AS
            ctypes.c_void_p,  # Row2Tok
            ctypes.c_void_p,  # SLABS
            ctypes.c_void_p,  # WS3
            ctypes.c_void_p,  # SLAB_META
            ctypes.c_void_p,  # S_OFF
            ctypes.c_void_p,  # D_OFF
            ctypes.c_void_p,  # SFLIPS
            ctypes.c_void_p,  # BMASKS
            ctypes.c_void_p,  # Ones
            ctypes.c_void_p,  # Tiles
            ctypes.c_void_p,  # TileCompactStart
            ctypes.c_void_p,  # Gammas
            ctypes.c_void_p,  # Out
            ctypes.c_int,     # N
            ctypes.c_int,     # K
            ctypes.c_int,     # MAX_CTAS
            ctypes.c_int,     # TILE
            ctypes.c_uint32,  # base_ws
            ctypes.c_int,     # zero_scatter
            ctypes.c_int,     # as_row_major
            ctypes.c_int,     # fuse_situ
            ctypes.c_int,     # pdl_wait_mode
            ctypes.c_int,     # trig_dep
            ctypes.c_void_p,  # TileW13Done
            ctypes.c_void_p,  # TileReady
            ctypes.c_void_p,  # OverlapMeta
            ctypes.c_int,     # overlap_w13_w2
            ctypes.c_int,     # hot_l2_hint
            ctypes.c_uint64,  # stream_ptr
        ]
        lib.mx7_w13_c48_v8_compat_fwd.restype = ctypes.c_int
    if hasattr(lib, "mx7_w2_c48_v8_compat_fwd"):
        lib.mx7_w2_c48_v8_compat_fwd.argtypes = [
            ctypes.c_void_p,  # A
            ctypes.c_void_p,  # AS
            ctypes.c_void_p,  # SLABS
            ctypes.c_void_p,  # WS3
            ctypes.c_void_p,  # SLAB_META
            ctypes.c_void_p,  # S_OFF
            ctypes.c_void_p,  # D_OFF
            ctypes.c_void_p,  # SFLIPS
            ctypes.c_void_p,  # BMASKS
            ctypes.c_void_p,  # Ones
            ctypes.c_void_p,  # Tiles
            ctypes.c_void_p,  # TileCompactStart
            ctypes.c_void_p,  # Out
            ctypes.c_void_p,  # Src2Compact
            ctypes.c_void_p,  # NLivePtr
            ctypes.c_void_p,  # PanelDone
            ctypes.c_void_p,  # BufGen
            ctypes.c_void_p,  # SendBuf
            ctypes.c_int,     # N
            ctypes.c_int,     # K
            ctypes.c_int,     # MAX_CTAS
            ctypes.c_int,     # TILE
            ctypes.c_int,     # T
            ctypes.c_uint32,  # base_ws
            ctypes.c_int,     # as_row_major
            ctypes.c_int,     # as_row_stride
            ctypes.c_int,     # down_row_stride
            ctypes.c_int,     # fuse_quant
            ctypes.c_int,     # fuse_combine
            ctypes.c_int,     # pdl_wait_mode
            ctypes.c_int,     # trig_dep
            ctypes.c_void_p,  # TileReady
            ctypes.c_void_p,  # OverlapMeta
            ctypes.c_int,     # overlap_w13_w2
            ctypes.c_int,     # hot_l2_hint
            ctypes.c_uint64,  # stream_ptr
        ]
        lib.mx7_w2_c48_v8_compat_fwd.restype = ctypes.c_int
    if hasattr(lib, "mx7_w13_c48_v8_panel_fwd"):
        lib.mx7_w13_c48_v8_panel_fwd.argtypes = [
            ctypes.c_void_p,  # A
            ctypes.c_void_p,  # AS
            ctypes.c_void_p,  # Row2Tok
            ctypes.c_void_p,  # PANEL_POOL
            ctypes.c_void_p,  # PANEL_OFF
            ctypes.c_void_p,  # SLAB_META
            ctypes.c_void_p,  # S_OFF
            ctypes.c_void_p,  # SFLIPS
            ctypes.c_void_p,  # Ones
            ctypes.c_void_p,  # Tiles
            ctypes.c_void_p,  # TileCompactStart
            ctypes.c_void_p,  # Gammas
            ctypes.c_void_p,  # Out
            ctypes.c_int,     # N
            ctypes.c_int,     # K
            ctypes.c_int,     # MAX_CTAS
            ctypes.c_int,     # TILE
            ctypes.c_uint32,  # base_ws
            ctypes.c_int,     # zero_scatter
            ctypes.c_int,     # as_row_major
            ctypes.c_int,     # fuse_situ
            ctypes.c_int,     # pdl_wait_mode
            ctypes.c_int,     # trig_dep
            ctypes.c_void_p,  # TileW13Done
            ctypes.c_void_p,  # TileReady
            ctypes.c_void_p,  # OverlapMeta
            ctypes.c_int,     # overlap_w13_w2
            ctypes.c_int,     # hot_l2_hint
            ctypes.c_uint64,  # stream_ptr
        ]
        lib.mx7_w13_c48_v8_panel_fwd.restype = ctypes.c_int
    if hasattr(lib, "mx7_w2_c48_v8_panel_fwd"):
        lib.mx7_w2_c48_v8_panel_fwd.argtypes = [
            ctypes.c_void_p,  # A
            ctypes.c_void_p,  # AS
            ctypes.c_void_p,  # PANEL_POOL
            ctypes.c_void_p,  # PANEL_OFF
            ctypes.c_void_p,  # SLAB_META
            ctypes.c_void_p,  # S_OFF
            ctypes.c_void_p,  # SFLIPS
            ctypes.c_void_p,  # Ones
            ctypes.c_void_p,  # Tiles
            ctypes.c_void_p,  # TileCompactStart
            ctypes.c_void_p,  # Out
            ctypes.c_void_p,  # Src2Compact
            ctypes.c_void_p,  # NLivePtr
            ctypes.c_void_p,  # PanelDone
            ctypes.c_void_p,  # BufGen
            ctypes.c_void_p,  # SendBuf
            ctypes.c_int,     # N
            ctypes.c_int,     # K
            ctypes.c_int,     # MAX_CTAS
            ctypes.c_int,     # TILE
            ctypes.c_int,     # T
            ctypes.c_uint32,  # base_ws
            ctypes.c_int,     # as_row_major
            ctypes.c_int,     # as_row_stride
            ctypes.c_int,     # down_row_stride
            ctypes.c_int,     # fuse_quant
            ctypes.c_int,     # fuse_combine
            ctypes.c_int,     # pdl_wait_mode
            ctypes.c_int,     # trig_dep
            ctypes.c_void_p,  # TileReady
            ctypes.c_void_p,  # OverlapMeta
            ctypes.c_int,     # overlap_w13_w2
            ctypes.c_int,     # hot_l2_hint
            ctypes.c_uint64,  # stream_ptr
        ]
        lib.mx7_w2_c48_v8_panel_fwd.restype = ctypes.c_int
    if hasattr(lib, "mx7_c48_v14_bn16_min_cta"):
        lib.mx7_c48_v14_bn16_min_cta.argtypes = []
        lib.mx7_c48_v14_bn16_min_cta.restype = ctypes.c_int
    if hasattr(lib, "mx7_c48_v14_min_cta"):
        lib.mx7_c48_v14_min_cta.argtypes = []
        lib.mx7_c48_v14_min_cta.restype = ctypes.c_int
    if hasattr(lib, "mx7_w13_c48_v16_fwd"):
        lib.mx7_w13_c48_v16_fwd.argtypes = [
            *[ctypes.c_void_p] * 13,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint32,
            *[ctypes.c_int] * 8,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            *[ctypes.c_int] * 4,
            ctypes.c_void_p, ctypes.c_uint64,
        ]
        lib.mx7_w13_c48_v16_fwd.restype = ctypes.c_int
    if hasattr(lib, "mx7_w2_c48_v16_fwd"):
        lib.mx7_w2_c48_v16_fwd.argtypes = [
            *[ctypes.c_void_p] * 15,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint32,
            *[ctypes.c_int] * 7,
            ctypes.c_void_p, ctypes.c_void_p,
            *[ctypes.c_int] * 4,
            ctypes.c_void_p, ctypes.c_uint64,
        ]
        lib.mx7_w2_c48_v16_fwd.restype = ctypes.c_int
    if hasattr(lib, "mx7_w13_c48_v16_compat_fwd"):
        lib.mx7_w13_c48_v16_compat_fwd.argtypes = [
            *[ctypes.c_void_p] * 16,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint32,
            *[ctypes.c_int] * 8,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            *[ctypes.c_int] * 4,
            ctypes.c_void_p, ctypes.c_uint64,
        ]
        lib.mx7_w13_c48_v16_compat_fwd.restype = ctypes.c_int
    if hasattr(lib, "mx7_w2_c48_v16_compat_fwd"):
        lib.mx7_w2_c48_v16_compat_fwd.argtypes = [
            *[ctypes.c_void_p] * 18,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint32,
            *[ctypes.c_int] * 7,
            ctypes.c_void_p, ctypes.c_void_p,
            *[ctypes.c_int] * 4,
            ctypes.c_void_p, ctypes.c_uint64,
        ]
        lib.mx7_w2_c48_v16_compat_fwd.restype = ctypes.c_int
    if hasattr(lib, "mx7_w13_c48_v16_panel_fwd"):
        lib.mx7_w13_c48_v16_panel_fwd.argtypes = [
            *[ctypes.c_void_p] * 14,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint32,
            *[ctypes.c_int] * 8,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            *[ctypes.c_int] * 4,
            ctypes.c_void_p, ctypes.c_uint64,
        ]
        lib.mx7_w13_c48_v16_panel_fwd.restype = ctypes.c_int
    if hasattr(lib, "mx7_w2_c48_v16_panel_fwd"):
        lib.mx7_w2_c48_v16_panel_fwd.argtypes = [
            *[ctypes.c_void_p] * 16,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint32,
            *[ctypes.c_int] * 7,
            ctypes.c_void_p, ctypes.c_void_p,
            *[ctypes.c_int] * 4,
            ctypes.c_void_p, ctypes.c_uint64,
        ]
        lib.mx7_w2_c48_v16_panel_fwd.restype = ctypes.c_int
    if hasattr(lib, "mx7_w13_c48_v14_fwd"):
        lib.mx7_w13_c48_v14_fwd.argtypes = [
            *[ctypes.c_void_p] * 13,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint32,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
        ]
        lib.mx7_w13_c48_v14_fwd.restype = ctypes.c_int
    if hasattr(lib, "mx7_w2_c48_v14_fwd"):
        lib.mx7_w2_c48_v14_fwd.argtypes = [
            *[ctypes.c_void_p] * 15,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint32,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
        ]
        lib.mx7_w2_c48_v14_fwd.restype = ctypes.c_int
    if hasattr(lib, "mx7_w13_c48_v14_compat_fwd"):
        lib.mx7_w13_c48_v14_compat_fwd.argtypes = [
            *[ctypes.c_void_p] * 16,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint32,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
        ]
        lib.mx7_w13_c48_v14_compat_fwd.restype = ctypes.c_int
    if hasattr(lib, "mx7_w2_c48_v14_compat_fwd"):
        lib.mx7_w2_c48_v14_compat_fwd.argtypes = [
            *[ctypes.c_void_p] * 18,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint32,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
        ]
        lib.mx7_w2_c48_v14_compat_fwd.restype = ctypes.c_int
    if hasattr(lib, "mx7_w13_c48_v14_panel_fwd"):
        lib.mx7_w13_c48_v14_panel_fwd.argtypes = [
            *[ctypes.c_void_p] * 14,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint32,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
        ]
        lib.mx7_w13_c48_v14_panel_fwd.restype = ctypes.c_int
    if hasattr(lib, "mx7_w2_c48_v14_panel_fwd"):
        lib.mx7_w2_c48_v14_panel_fwd.argtypes = [
            *[ctypes.c_void_p] * 16,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint32,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
        ]
        lib.mx7_w2_c48_v14_panel_fwd.restype = ctypes.c_int
    ver = 3
    if hasattr(lib, "mx7_c32_version"):
        lib.mx7_c32_version.argtypes = []
        lib.mx7_c32_version.restype = ctypes.c_int
        ver = int(lib.mx7_c32_version())
    if path is None:
        global _LIB_MX7_VER
        _LIB_MX7 = lib
        _LIB_MX7_VER = ver
    return lib


def get_mx7_version() -> int:
    if _LIB_MX7 is None and not _TRIED["mx7"]:
        get_mx7_lib()
    return _LIB_MX7_VER


def get_router_lib(path: str | None = None) -> ctypes.CDLL | None:
    global _LIB_ROUTER
    if _LIB_ROUTER is not None and path is None:
        return _LIB_ROUTER
    if _TRIED["router"] and path is None:
        return None
    if path is None:
        _TRIED["router"] = True
    paths = (path,) if path else _ROUTER_PATHS
    lib = _find_lib(paths, "router_c32_quant_route_place_fwd")
    if lib is None:
        return None
    # Match router_bf12_c32.cu extern "C" signatures 1-to-1
    lib.router_bf12_c32_gemm_fwd.argtypes = [
        ctypes.c_void_p,  # X
        ctypes.c_void_p,  # SLABS
        ctypes.c_void_p,  # OUTLIERS
        ctypes.c_void_p,  # O_OFF
        ctypes.c_void_p,  # NLivePtr
        ctypes.c_void_p,  # Logits
        ctypes.c_int,     # T
        ctypes.c_uint32,  # base_exp
        ctypes.c_int,     # pdl
        ctypes.c_int,     # trig_dep
        ctypes.c_uint64,  # stream_ptr
    ]
    lib.router_bf12_c32_gemm_fwd.restype = ctypes.c_int

    lib.router_c32_quant_route_place_fwd.argtypes = [
        ctypes.c_void_p,  # X
        ctypes.c_void_p,  # Logits
        ctypes.c_void_p,  # Bias
        ctypes.c_void_p,  # ExpertBounds
        ctypes.c_void_p,  # Padding
        ctypes.c_void_p,  # NLivePtr
        ctypes.c_void_p,  # Counts
        ctypes.c_void_p,  # Counters
        ctypes.c_void_p,  # Ids
        ctypes.c_void_p,  # Ranks
        ctypes.c_void_p,  # Weights
        ctypes.c_void_p,  # Starts
        ctypes.c_void_p,  # StartsCompact
        ctypes.c_void_p,  # Src2Dst
        ctypes.c_void_p,  # Src2Compact
        ctypes.c_void_p,  # Gammas
        ctypes.c_void_p,  # Rows
        ctypes.c_void_p,  # Tiles
        ctypes.c_void_p,  # TileCompactStart
        ctypes.c_void_p,  # Row2Tok
        ctypes.c_void_p,  # RowAmaxW2
        ctypes.c_void_p,  # QValues
        ctypes.c_void_p,  # QScales
        ctypes.c_void_p,  # QRowScales
        ctypes.c_void_p,  # RowAmaxIn
        ctypes.c_void_p,  # WAmax
        ctypes.c_void_p,  # WDenominator
        ctypes.c_int,     # T
        ctypes.c_float,   # route_scale
        ctypes.c_int,     # num_local
        ctypes.c_int,     # tile_sz
        ctypes.c_int,     # max_ctas
        ctypes.c_int,     # nvfp4
        ctypes.c_int,     # logits_bf16
        ctypes.c_int,     # pdl
        ctypes.c_uint64,  # stream_ptr
    ]
    lib.router_c32_quant_route_place_fwd.restype = ctypes.c_int
    if hasattr(lib, "router_c32_quant_route_place_v9_fwd"):
        lib.router_c32_quant_route_place_v9_fwd.argtypes = [
            ctypes.c_void_p,  # X
            ctypes.c_void_p,  # Logits
            ctypes.c_void_p,  # Bias
            ctypes.c_void_p,  # ExpertBounds
            ctypes.c_void_p,  # Padding
            ctypes.c_void_p,  # NLivePtr
            ctypes.c_void_p,  # Counts
            ctypes.c_void_p,  # Counters
            ctypes.c_void_p,  # Ids
            ctypes.c_void_p,  # Ranks
            ctypes.c_void_p,  # Weights
            ctypes.c_void_p,  # Starts
            ctypes.c_void_p,  # StartsCompact
            ctypes.c_void_p,  # Src2Dst
            ctypes.c_void_p,  # Src2Compact
            ctypes.c_void_p,  # Gammas
            ctypes.c_void_p,  # Rows
            ctypes.c_void_p,  # Tiles
            ctypes.c_void_p,  # TileCompactStart
            ctypes.c_void_p,  # Row2Tok
            ctypes.c_void_p,  # RowAmaxW2
            ctypes.c_void_p,  # QValues
            ctypes.c_void_p,  # QScales
            ctypes.c_void_p,  # QRowScales
            ctypes.c_void_p,  # RowAmaxIn
            ctypes.c_void_p,  # WAmax
            ctypes.c_void_p,  # WDenominator
            ctypes.c_int,     # T
            ctypes.c_float,   # route_scale
            ctypes.c_int,     # num_local
            ctypes.c_int,     # tile_sz
            ctypes.c_int,     # max_ctas
            ctypes.c_int,     # nvfp4
            ctypes.c_int,     # logits_bf16
            ctypes.c_int,     # pdl
            ctypes.c_int,     # quant_fma
            ctypes.c_uint64,  # stream_ptr
        ]
        lib.router_c32_quant_route_place_v9_fwd.restype = ctypes.c_int
    if hasattr(lib, "router_bf12_v10_version"):
        lib.router_bf12_v10_version.argtypes = []
        lib.router_bf12_v10_version.restype = ctypes.c_int
    if hasattr(lib, "router_bf12_c48_fused_v10_fwd"):
        lib.router_bf12_c48_fused_v10_fwd.argtypes = [
            ctypes.c_void_p,  # X
            ctypes.c_void_p,  # SLABS
            ctypes.c_void_p,  # OUTLIERS
            ctypes.c_void_p,  # O_OFF
            ctypes.c_void_p,  # Bias
            ctypes.c_void_p,  # ExpertBounds
            ctypes.c_void_p,  # Padding
            ctypes.c_void_p,  # NLivePtr
            ctypes.c_void_p,  # LogitsWS
            ctypes.c_void_p,  # SelBitsWS
            ctypes.c_void_p,  # Counters
            ctypes.c_void_p,  # Starts
            ctypes.c_void_p,  # StartsCompact
            ctypes.c_void_p,  # Src2Dst
            ctypes.c_void_p,  # Src2Compact
            ctypes.c_void_p,  # Gammas
            ctypes.c_void_p,  # Rows
            ctypes.c_void_p,  # Tiles
            ctypes.c_void_p,  # TileCompactStart
            ctypes.c_void_p,  # Row2Tok
            ctypes.c_void_p,  # RowAmaxW2
            ctypes.c_void_p,  # QValues
            ctypes.c_void_p,  # QScales
            ctypes.c_void_p,  # QRowScales
            ctypes.c_void_p,  # RowAmaxIn
            ctypes.c_void_p,  # WAmax
            ctypes.c_void_p,  # WDenominator
            ctypes.c_int,     # T
            ctypes.c_float,   # route_scale
            ctypes.c_uint32,  # base_exp
            ctypes.c_int,     # num_local
            ctypes.c_int,     # tile_sz
            ctypes.c_int,     # max_ctas
            ctypes.c_int,     # nvfp4
            ctypes.c_int,     # quant_fma
            ctypes.c_int,     # pdl
            ctypes.c_uint64,  # stream_ptr
        ]
        lib.router_bf12_c48_fused_v10_fwd.restype = ctypes.c_int
    if hasattr(lib, "router_bf11_c48_fused_v10_fwd"):
        lib.router_bf11_c48_fused_v10_fwd.argtypes = [
            ctypes.c_void_p,  # X
            ctypes.c_void_p,  # SLABS
            ctypes.c_void_p,  # SLAB_META_V5
            ctypes.c_void_p,  # SFLIPS
            ctypes.c_void_p,  # BMASKS
            ctypes.c_void_p,  # OUTLIERS
            ctypes.c_void_p,  # O_OFF
            ctypes.c_void_p,  # Bias
            ctypes.c_void_p,  # ExpertBounds
            ctypes.c_void_p,  # Padding
            ctypes.c_void_p,  # NLivePtr
            ctypes.c_void_p,  # LogitsWS
            ctypes.c_void_p,  # SelBitsWS
            ctypes.c_void_p,  # Counters
            ctypes.c_void_p,  # Starts
            ctypes.c_void_p,  # StartsCompact
            ctypes.c_void_p,  # Src2Dst
            ctypes.c_void_p,  # Src2Compact
            ctypes.c_void_p,  # Gammas
            ctypes.c_void_p,  # Rows
            ctypes.c_void_p,  # Tiles
            ctypes.c_void_p,  # TileCompactStart
            ctypes.c_void_p,  # Row2Tok
            ctypes.c_void_p,  # RowAmaxW2
            ctypes.c_void_p,  # QValues
            ctypes.c_void_p,  # QScales
            ctypes.c_void_p,  # QRowScales
            ctypes.c_void_p,  # RowAmaxIn
            ctypes.c_void_p,  # WAmax
            ctypes.c_void_p,  # WDenominator
            ctypes.c_int,     # T
            ctypes.c_float,   # route_scale
            ctypes.c_uint32,  # base_exp
            ctypes.c_int,     # num_local
            ctypes.c_int,     # tile_sz
            ctypes.c_int,     # max_ctas
            ctypes.c_int,     # nvfp4
            ctypes.c_int,     # quant_fma
            ctypes.c_int,     # pdl
            ctypes.c_uint64,  # stream_ptr
        ]
        lib.router_bf11_c48_fused_v10_fwd.restype = ctypes.c_int
    if hasattr(lib, "router_bf12_c48_min_cta"):
        lib.router_bf12_c48_min_cta.argtypes = []
        lib.router_bf12_c48_min_cta.restype = ctypes.c_int
    ver = 3
    if hasattr(lib, "router_bf12_c32_version"):
        lib.router_bf12_c32_version.argtypes = []
        lib.router_bf12_c32_version.restype = ctypes.c_int
        ver = int(lib.router_bf12_c32_version())
    if path is None:
        global _LIB_ROUTER_VER
        _LIB_ROUTER = lib
        _LIB_ROUTER_VER = ver
    return lib


def get_router_version() -> int:
    if _LIB_ROUTER is None and not _TRIED["router"]:
        get_router_lib()
    return _LIB_ROUTER_VER


def get_workspace(device: torch.device) -> dict[str, torch.Tensor]:
    """Return persistent CUDA-graph-safe C32 scratch buffers (~7.62 MiB total; ~1.62 MiB when FC=0 + SHARED_MOE_SCRATCH)."""
    idx = int(device.index or 0)
    ws = _WS.get(idx)
    if ws is None:
        ws = {
            "situ_out": torch.empty((_WS_MAX_ROUTED, 288), dtype=torch.bfloat16, device=device),
            "row_amax_w2": torch.zeros((_WS_MAX_ROUTED,), dtype=torch.int32, device=device),
            "panel_done": torch.zeros((36,), dtype=torch.int32, device=device),
            "buf_gen": torch.zeros((12,), dtype=torch.int32, device=device),
            "logits_ws": torch.empty((_WS_MAX_T, 384), dtype=torch.float32, device=device),
            "sel_bits_ws": torch.empty((64, 384), dtype=torch.int32, device=device),
            "counts": torch.zeros((384,), dtype=torch.int32, device=device),
            "counters": torch.zeros((64,), dtype=torch.int32, device=device),
            "ids": torch.empty((_WS_MAX_ROUTED,), dtype=torch.int32, device=device),
            "ranks": torch.empty((_WS_MAX_ROUTED,), dtype=torch.int32, device=device),
            "weights": torch.empty((_WS_MAX_ROUTED,), dtype=torch.float32, device=device),
            "starts": torch.empty((384,), dtype=torch.int32, device=device),
            "starts_compact": torch.empty((384,), dtype=torch.int32, device=device),
            "tile_compact_start": torch.empty((_WS_MAX_CTAS,), dtype=torch.int32, device=device),
            "src2compact": torch.empty((_WS_MAX_ROUTED,), dtype=torch.int32, device=device),
            "row2tok": torch.empty((_WS_MAX_ROWS,), dtype=torch.int32, device=device),
            "row_amax_in": torch.empty((_WS_MAX_T,), dtype=torch.float32, device=device),
            "tile_w13_done": torch.zeros((_WS_MAX_CTAS,), dtype=torch.int32, device=device),
            "tile_ready": torch.zeros((_WS_MAX_CTAS,), dtype=torch.int32, device=device),
            "overlap_meta": torch.zeros((32,), dtype=torch.int32, device=device),
            "work_queue": torch.zeros((64,), dtype=torch.int32, device=device),
        }
        if not SHARED_MOE_SCRATCH or C32_FUSE_COMBINE == 1 or C32_FUSE_COMBINE_C18 == 1:
            ws["down_panel"] = torch.empty((2 * 6 * _WS_MAX_ROUTED, 128), dtype=torch.bfloat16, device=device)
        _WS[idx] = ws
    if (C32_FUSE_COMBINE == 1 or C32_FUSE_COMBINE_C18 == 1 or C32_NVFP4_FC_MODE == 1 or C32_MX7_FC_MODE == 1) and "down_panel" not in ws:
        ws["down_panel"] = torch.empty((2 * 6 * _WS_MAX_ROUTED, 128), dtype=torch.bfloat16, device=device)
    if (C32_FUSE_COMBINE in (2, 3) or C32_FUSE_COMBINE_C18 in (2, 3) or C32_NVFP4_FC_MODE in (2, 3) or C32_MX7_FC_MODE in (2, 3) or C32_COMBINE_V6 or C48_MOE_V7 or C48_MOE_V8 or C48_MOE_V9 or C48_MOE_V10 or C48_MOE_V12 or C48_MOE_TILE8 or MX7_C48_V14) and "down_compact" not in ws:
        # 36 panels * 2048 compact rows * 128 bf16 = 18.0 MiB (same size as [2048, 4608] bf16
        # for COMPACT_OUT + nvfp4_combine_c32_v6_fwd; 100% L2-hot <= 9.44 MB at M<=128!).
        ws["down_compact"] = torch.empty((36 * _WS_MAX_ROUTED, 128), dtype=torch.bfloat16, device=device)
    return ws


@torch.no_grad()
def get_ws_base_c32(meta: dict) -> torch.Tensor:
    """Return `ws_base` with bit 7 set (`| 0x80`) on experts with ZERO scale patches.

    Leaves `meta['ws_base']` untouched so Triton prefill kernels stay 100% unmodified.
    """
    tagged = meta.get("ws_base_c32")
    if tagged is None:
        tagged = meta.get("ws_base_s8")
    if tagged is not None:
        return tagged
    ws_base: torch.Tensor = meta["ws_base"]
    if torch.cuda.is_current_stream_capturing():
        return ws_base
    p_off: torch.Tensor = meta["p_off"]
    E = int(ws_base.shape[0])
    proj = int(meta["proj"])
    stride = (18 * 4) if proj == 0 else 36
    idx = torch.arange(E, device=ws_base.device, dtype=torch.long) * stride
    no_patch = p_off[idx + stride] == p_off[idx]
    assert int(ws_base.max().item()) < 128, "NVFP4 ws_base unexpectedly >= 128"
    tagged = torch.where(no_patch, ws_base | 0x80, ws_base).to(torch.uint8).contiguous()
    meta["ws_base_c32"] = tagged
    return tagged


@torch.no_grad()
def get_mx7_packed_c32(weight: torch.Tensor, weight_scale: torch.Tensor, projection: int):
    """Return `(slabs, ws3, flips, f_off, base_ws)` reusing `weight._inkling_mx7` if present."""
    cached = getattr(weight, "_inkling_mx7", None)
    if isinstance(cached, tuple):
        return cached
    if cached is False:
        return None
    # Prefer trtllm_bmm._mx7_get_packed so packing & FREE_RAW policy stay unified.
    try:
        from . import trtllm_bmm as _tbmm
        packed = _tbmm._mx7_get_packed(weight, weight_scale, projection, 0, 0)
        if isinstance(packed, tuple):
            return packed
    except Exception:
        pass
    if torch.cuda.is_current_stream_capturing():
        return None
    lib = get_mx7_lib()
    if lib is None or not (hasattr(lib, "mx7_v2_count_flips") and hasattr(lib, "mx7_v2_pack")):
        return None
    if weight.data.untyped_storage().size() == 0 or weight_scale.data.untyped_storage().size() == 0:
        return None
    e, n, k = int(weight.shape[0]), int(weight.shape[1]), int(weight.shape[2])
    if projection == 0 and (n % 32) == 0 and (k % 256) == 0:
        bn, warps, n_slabs = 32, 4, k // 128
    elif projection == 1 and (n % 64) == 0 and (k % 32) == 0:
        bn, warps, n_slabs = 64, 2, k // 32
    else:
        return None
    ws_u8 = weight_scale.data.view(torch.uint8)
    base_ws = int(ws_u8.min().item())
    max_ws = int(ws_u8.max().item())
    if base_ws == 0 or max_ws == 255 or (max_ws - base_ws) > 7:
        weight._inkling_mx7 = False
        return None
    w_u8 = weight.data.view(torch.uint8)
    dev = weight.device
    stream = int(torch.cuda.current_stream(dev).cuda_stream)
    total_warps = e * (n // bn) * warps
    panel_meta = getattr(weight, "_inkling_panel", None)
    if panel_meta is not None:
        from_panel, src_pb, src_kb = 1, int(panel_meta[0]), int(panel_meta[1])
    else:
        from_panel, src_pb, src_kb = 0, 0, 0
    counts = torch.empty(total_warps, dtype=torch.int32, device=dev)
    rc = lib.mx7_v2_count_flips(
        w_u8.data_ptr(), counts.data_ptr(), projection, e, n, k, from_panel, src_pb, src_kb, stream
    )
    if rc != 0:
        return None
    f_off = torch.zeros(total_warps + 1, dtype=torch.int32, device=dev)
    torch.cumsum(counts, dim=0, out=f_off[1:])
    del counts
    total_flips = int(f_off[-1].item())
    flips = torch.empty(max(1, total_flips), dtype=torch.uint16, device=dev)
    slabs = torch.empty((total_warps * n_slabs, 896), dtype=torch.uint8, device=dev)
    ws3 = torch.empty((total_warps * n_slabs, 4), dtype=torch.uint32, device=dev)
    rc = lib.mx7_v2_pack(
        w_u8.data_ptr(), ws_u8.data_ptr(), slabs.data_ptr(), ws3.data_ptr(),
        flips.data_ptr(), f_off.data_ptr(),
        projection, e, n, k, base_ws, from_panel, src_pb, src_kb, stream
    )
    if rc != 0:
        return None
    entry = (slabs, ws3, flips, f_off, base_ws)
    weight._inkling_mx7 = entry
    return entry


@torch.no_grad()
def get_mx7_v5_compat(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    projection: int,
    flip_thresh: int | None = None,
):
    """Return `(slabs, ws3, slab_meta, s_off, d_off, sflips, bmasks, base_ws, thr)`
    transcoded directly on GPU from `weight._inkling_mx7` (100% `FREE_RAW=1` & 1M-prefill safe).
    Shared across C=32 & C=1 via `weight._inkling_mx7_v5_compat` with zero duplicate VRAM."""
    thr = MX7_V5_THRESH if flip_thresh is None else max(0, min(63, int(flip_thresh)))
    cached = getattr(weight, "_inkling_mx7_v5_compat", None)
    if isinstance(cached, tuple) and len(cached) == 9 and cached[8] == thr:
        return cached
    if torch.cuda.is_current_stream_capturing():
        return None
    lib = get_mx7_lib()
    if lib is None or not (
        hasattr(lib, "mx7_v5_count_hybrid_from_mx7")
        and hasattr(lib, "mx7_v5_build_hybrid_from_mx7")
    ):
        return None
    packed = get_mx7_packed_c32(weight, weight_scale, projection)
    if packed is None:
        return None
    slabs, ws3, flips, f_off, base_ws = packed
    total_warps = int(f_off.shape[0]) - 1
    if total_warps <= 0 or (int(slabs.shape[0]) % total_warps) != 0:
        return None
    n_slabs = int(slabs.shape[0]) // total_warps
    dev = slabs.device
    stream = int(torch.cuda.current_stream(dev).cuda_stream)
    s_cnt = torch.empty(total_warps, dtype=torch.int32, device=dev)
    d_cnt = torch.empty(total_warps, dtype=torch.int32, device=dev)
    rc = lib.mx7_v5_count_hybrid_from_mx7(
        flips.data_ptr(), f_off.data_ptr(), s_cnt.data_ptr(), d_cnt.data_ptr(),
        total_warps, n_slabs, thr, stream,
    )
    if rc != 0:
        return None
    s_off = torch.zeros(total_warps + 1, dtype=torch.int32, device=dev)
    d_off = torch.zeros(total_warps + 1, dtype=torch.int32, device=dev)
    torch.cumsum(s_cnt, dim=0, out=s_off[1:])
    torch.cumsum(d_cnt, dim=0, out=d_off[1:])
    del s_cnt, d_cnt
    total_s = int(s_off[-1].item())
    total_d = int(d_off[-1].item())
    slab_meta = torch.empty(total_warps * n_slabs, dtype=torch.int32, device=dev)
    sflips = torch.empty(max(1, total_s), dtype=torch.uint16, device=dev)
    bmasks = torch.empty((max(1, total_d), 32), dtype=torch.int32, device=dev)
    rc = lib.mx7_v5_build_hybrid_from_mx7(
        slabs.data_ptr(), flips.data_ptr(), f_off.data_ptr(),
        s_off.data_ptr(), d_off.data_ptr(),
        slab_meta.data_ptr(), sflips.data_ptr(), bmasks.data_ptr(),
        total_warps, n_slabs, thr, stream,
    )
    if rc != 0:
        return None
    entry = (slabs, ws3, slab_meta, s_off, d_off, sflips, bmasks, int(base_ws), thr)
    weight._inkling_mx7_v5_compat = entry  # type: ignore[attr-defined]
    return entry


@torch.no_grad()
def get_mx7_v5_panel(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    projection: int,
    flip_thresh: int | None = None,
):
    """Return `(panel_pool, panel_off, slab_meta, s_off, sflips, base_ws, thr)`
    transcoded directly on GPU from `weight._inkling_mx7` into panel-contiguous super-slabs.
    Shared across C=32 & C=1 via `weight._inkling_mx7_v5_panel` with zero duplicate VRAM."""
    thr = MX7_V5_THRESH if flip_thresh is None else max(0, min(63, int(flip_thresh)))
    cached = getattr(weight, "_inkling_mx7_v5_panel", None)
    if isinstance(cached, tuple) and len(cached) == 7 and cached[6] == thr:
        return cached
    if torch.cuda.is_current_stream_capturing():
        return None
    lib = get_mx7_lib()
    if lib is None or not (
        hasattr(lib, "mx7_v5_panel_measure_from_mx7")
        and hasattr(lib, "mx7_v5_panel_build_from_mx7")
    ):
        return None
    packed = get_mx7_packed_c32(weight, weight_scale, projection)
    if packed is None:
        return None
    slabs, ws3, flips, f_off, base_ws = packed
    total_warps = int(f_off.shape[0]) - 1
    if total_warps <= 0 or (total_warps % 4) != 0 or (int(slabs.shape[0]) % total_warps) != 0:
        return None
    total_panels = total_warps // 4
    n_slabs = int(slabs.shape[0]) // total_warps
    dev = slabs.device
    stream = int(torch.cuda.current_stream(dev).cuda_stream)
    p_blks = torch.empty(total_panels, dtype=torch.int32, device=dev)
    s_cnt = torch.empty(total_warps, dtype=torch.int32, device=dev)
    rc = lib.mx7_v5_panel_measure_from_mx7(
        flips.data_ptr(), f_off.data_ptr(), p_blks.data_ptr(), s_cnt.data_ptr(),
        int(projection), total_panels, n_slabs, thr, stream,
    )
    if rc != 0:
        return None
    panel_off = torch.zeros(total_panels + 1, dtype=torch.int32, device=dev)
    s_off = torch.zeros(total_warps + 1, dtype=torch.int32, device=dev)
    torch.cumsum(p_blks, dim=0, out=panel_off[1:])
    torch.cumsum(s_cnt, dim=0, out=s_off[1:])
    del p_blks, s_cnt
    total_blocks = int(panel_off[-1].item())
    total_s = int(s_off[-1].item())
    panel_pool = torch.empty((max(1, total_blocks), 128), dtype=torch.uint8, device=dev)
    slab_meta = torch.empty(total_warps * n_slabs, dtype=torch.int32, device=dev)
    sflips = torch.empty(max(1, total_s), dtype=torch.uint16, device=dev)
    rc = lib.mx7_v5_panel_build_from_mx7(
        slabs.data_ptr(), ws3.data_ptr(), flips.data_ptr(), f_off.data_ptr(),
        panel_off.data_ptr(), s_off.data_ptr(),
        panel_pool.data_ptr(), slab_meta.data_ptr(), sflips.data_ptr(),
        int(projection), total_panels, n_slabs, thr, stream,
    )
    if rc != 0:
        return None
    entry = (panel_pool, panel_off, slab_meta, s_off, sflips, int(base_ws), thr)
    weight._inkling_mx7_v5_panel = entry  # type: ignore[attr-defined]
    return entry


@torch.no_grad()
def prewarm_mx7_v5(layers) -> int:
    """Prewarm v5 Compat-Hybrid or Panel-V5 buffers on all MXFP8 MoE layers before graph capture."""
    if MX7_V5_MODE == "off" or torch.cuda.is_current_stream_capturing():
        return 0
    lib = get_mx7_lib()
    if lib is None or get_mx7_version() < 5:
        return 0
    fn = get_mx7_v5_panel if MX7_V5_MODE == "panel" else get_mx7_v5_compat
    n_built = 0
    for layer in layers:
        exp = getattr(getattr(layer, "mlp", None), "experts", None)
        if exp is None or getattr(exp, "nvfp4", True):
            continue
        if fn(exp.w13_weight, exp.w13_weight_scale, 0) is not None:
            n_built += 1
        if fn(exp.w2_weight, exp.w2_weight_scale, 1) is not None:
            n_built += 1
    return n_built


def c32_eligible(
    tokens: int,
    hidden: int,
    topk: int,
    plan,
    gate_cap: float = 4.0,
    up_cap: float = 25.0,
) -> bool:
    tile_ok = (
        plan.tile in (16, 32)
        or (plan.tile == 8 and C48_MOE_TILE8 and tokens <= C48_MOE_MAX_M)
    )
    return (
        C32_MASTER
        and C32_MIN_M <= tokens <= C32_MAX_M
        and hidden == 4608
        and topk == 8
        and tile_ok
        and plan.max_ctas <= _WS_MAX_CTAS
        and plan.rows <= _WS_MAX_ROWS
        and gate_cap == 4.0
        and up_cap == 25.0
    )


def _is_full_decode_capturing() -> bool:
    """True ONLY inside FULL decode CUDA graph capture (never eager or PIECEWISE prefill!).

    vLLM runs in FULL_AND_PIECEWISE mode and captures PIECEWISE prefill graphs for M=32..256
    where `torch.cuda.is_current_stream_capturing()` is True. Running `router_bf12_c32_gemm_fwd`
    (FP32 5-K-group logits) inside PIECEWISE prefill diverges from stock cuBLAS BF16
    `k_group_partials` (deterministic `exact_ink` hash `1dcfb306` -> `9a860d12` when
    `INKLING_C32_MOE_MIN_M=32`). Restricting `bf12` to FULL decode graphs keeps all eager
    and PIECEWISE prefills 100% 0-ULP bit-exact while letting FULL decode graphs use `bf12`.
    """
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
    return False


def will_use_bf12_gate(num_tokens: int) -> bool:
    """True iff FULL decode graphs at `num_tokens` execute `router_bf12_c32_gemm_fwd`."""
    return (
        C32_MASTER
        and C32_ROUTER_MODE == "bf12"
        and C32_MIN_M <= int(num_tokens) <= C32_MAX_M
    )


def route_c32(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    route_scale: float,
    expert_bounds: torch.Tensor,
    plan,
    weight_amax: torch.Tensor | None,
    weight_denominator: torch.Tensor | None,
    num_local: int,
    is_padding: torch.Tensor | None = None,
    nlive_buf: torch.Tensor | None = None,
):
    """C=32 (M=17..256) fused router + Zero-Scatter quantizer + compact metadata builder.

    Returns `((src2dst, gammas, routed, tiles), quantized)` or `None` if ineligible/disabled.
    Attaches `_inkling_row2tok` onto `quantized[0]`, `_inkling_rowmaj = True` onto `quantized[1]`,
    and `_inkling_tile_compact_start`, `_inkling_src2compact`, `_inkling_row_amax_w2` onto `tiles`.
    """
    if C32_ROUTER_MODE == "off":
        return None
    if x.dtype != torch.bfloat16 or not x.is_contiguous():
        return None
    tokens, hidden = int(x.shape[0]), int(x.shape[1])
    if not c32_eligible(tokens, hidden, topk, plan):
        return None
    if int(gate_weight.shape[0]) != 384 or num_local > 384:
        return None

    lib = get_router_lib()
    if lib is None:
        return None

    device = x.device
    ws = get_workspace(device)
    st = int(torch.cuda.current_stream(device).cuda_stream)
    nvfp4 = weight_amax is not None
    n = tokens * topk
    rows = int(plan.rows)

    want_rtr_prequant = C48_ROUTER_PREQUANT_PDL and (C32_PDL > 0) and (get_router_version() >= 8)

    # Pre-allocate K2 output buffers BEFORE launching K1 when pre-gdc_wait overlap is enabled
    # so K2 enters the GPU stream immediately behind K1 with zero CPU allocation gap!
    src2dst = torch.empty(n, dtype=torch.int32, device=device)
    gammas = torch.empty(rows, dtype=torch.float32, device=device)
    routed = torch.empty(n, dtype=torch.int32, device=device)
    tiles = torch.empty(2 * plan.max_ctas + 3, dtype=torch.int32, device=device)

    # Zero-Scatter: QValues and row-major QScales are sized [tokens, ...], NEVER [plan.rows, ...]!
    if nvfp4:
        quantized = (
            torch.empty((tokens, hidden // 2), dtype=torch.uint8, device=device),
            torch.empty((tokens, hidden // 16), dtype=torch.float8_e4m3fn, device=device),
            torch.empty(rows, dtype=torch.float32, device=device),
        )
    else:
        quantized = (
            torch.empty((tokens, hidden), dtype=torch.float8_e4m3fn, device=device),
            torch.empty((tokens, hidden // 32), dtype=torch.uint8, device=device),
            None,
        )

    tile_compact_start = ws["tile_compact_start"][: plan.max_ctas]
    src2compact = ws["src2compact"][:n]
    row2tok = ws["row2tok"][:rows]
    row_amax_w2 = ws["row_amax_w2"][:n] if (nvfp4 and C32_FUSE_SITU) else None

    # Determine whether to run BF12 router GEMM or stock cuBLAS BF16 `k_group_partials`.
    use_bf12 = (C32_ROUTER_MODE == "bf12")
    if use_bf12 and C32_ROUTER_BF12_CAPTURE_ONLY and not _is_full_decode_capturing():
        use_bf12 = False

    # Wave-V10 Single-Launch 48-CTA Fused Router (T <= 64): -48 launches/step + 0 global Counts/Ids/Ranks!
    if (
        ROUTER_V10_FUSED
        and use_bf12
        and tokens <= 64
        and get_router_version() >= 10
        and hasattr(lib, "router_bf12_c48_fused_v10_fwd")
    ):
        try:
            from . import router_bf12 as _rb12
            _rb12.prepack(gate_weight)
            hit = _rb12.get_packed(gate_weight)
            if hit is not None:
                _, slabs, outliers, o_off, base_exp = hit
                skip_amax_prezero = (
                    nvfp4
                    and (NVFP4_V10_FUSE_QDEQ or NVFP4_V9_QDEQ)
                    and get_nvfp4_version() >= 9
                )
                amax_ptr = 0 if (row_amax_w2 is None or skip_amax_prezero) else row_amax_w2.data_ptr()
                pdl_v10 = 3 if (C32_PDL >= 2 and C48_ROUTER_PREQUANT_PDL) else C32_PDL
                bf11_entry = None
                if ROUTER_BF11 and hasattr(lib, "router_bf11_c48_fused_v10_fwd"):
                    bf11_entry = _rb12.get_packed_bf11(gate_weight)
                    if bf11_entry is None and not torch.cuda.is_current_stream_capturing():
                        bf11_entry = _rb12.prepack_bf11(gate_weight)
                if bf11_entry is not None:
                    s11, smeta11, sflips11, bmasks11, o11, oo11, be11 = bf11_entry
                    rc = lib.router_bf11_c48_fused_v10_fwd(
                        x.data_ptr(), s11.data_ptr(), smeta11.data_ptr(), sflips11.data_ptr(), bmasks11.data_ptr(),
                        o11.data_ptr(), oo11.data_ptr(),
                        bias.data_ptr(), expert_bounds.data_ptr(),
                        0 if is_padding is None else is_padding.data_ptr(),
                        0 if nlive_buf is None else nlive_buf.data_ptr(),
                        ws["logits_ws"].data_ptr(), ws["sel_bits_ws"].data_ptr(), ws["counters"].data_ptr(),
                        ws["starts"].data_ptr(), ws["starts_compact"].data_ptr(),
                        src2dst.data_ptr(), src2compact.data_ptr(), gammas.data_ptr(), routed.data_ptr(),
                        tiles.data_ptr(), tile_compact_start.data_ptr(), row2tok.data_ptr(), amax_ptr,
                        quantized[0].data_ptr(), quantized[1].data_ptr(),
                        0 if quantized[2] is None else quantized[2].data_ptr(),
                        ws["row_amax_in"].data_ptr(),
                        0 if weight_amax is None else weight_amax.data_ptr(),
                        0 if weight_denominator is None else weight_denominator.data_ptr(),
                        tokens, float(route_scale), int(be11), int(num_local),
                        int(plan.tile), int(plan.max_ctas), 1 if nvfp4 else 0,
                        ROUTER_C32_QUANT_FMA, pdl_v10, st,
                    )
                else:
                    rc = lib.router_bf12_c48_fused_v10_fwd(
                        x.data_ptr(), slabs.data_ptr(), outliers.data_ptr(), o_off.data_ptr(),
                        bias.data_ptr(), expert_bounds.data_ptr(),
                        0 if is_padding is None else is_padding.data_ptr(),
                        0 if nlive_buf is None else nlive_buf.data_ptr(),
                        ws["logits_ws"].data_ptr(), ws["sel_bits_ws"].data_ptr(), ws["counters"].data_ptr(),
                        ws["starts"].data_ptr(), ws["starts_compact"].data_ptr(),
                        src2dst.data_ptr(), src2compact.data_ptr(), gammas.data_ptr(), routed.data_ptr(),
                        tiles.data_ptr(), tile_compact_start.data_ptr(), row2tok.data_ptr(), amax_ptr,
                        quantized[0].data_ptr(), quantized[1].data_ptr(),
                        0 if quantized[2] is None else quantized[2].data_ptr(),
                        ws["row_amax_in"].data_ptr(),
                        0 if weight_amax is None else weight_amax.data_ptr(),
                        0 if weight_denominator is None else weight_denominator.data_ptr(),
                        tokens, float(route_scale), int(base_exp), int(num_local),
                        int(plan.tile), int(plan.max_ctas), 1 if nvfp4 else 0,
                        ROUTER_C32_QUANT_FMA, pdl_v10, st,
                    )
                if rc != 0:
                    raise RuntimeError(f"router_bf12_c48_fused_v10_fwd failed: rc={rc}")
                quantized[0]._inkling_row2tok = row2tok  # type: ignore[attr-defined]
                quantized[1]._inkling_rowmaj = True  # type: ignore[attr-defined]
                tiles._inkling_tile_compact_start = tile_compact_start  # type: ignore[attr-defined]
                tiles._inkling_src2compact = src2compact  # type: ignore[attr-defined]
                if row_amax_w2 is not None:
                    tiles._inkling_row_amax_w2 = row_amax_w2  # type: ignore[attr-defined]
                return (src2dst, gammas, routed, tiles), quantized
        except Exception:
            pass

    logits_bf16 = 1
    logits_ptr = 0
    if use_bf12:
        try:
            from . import router_bf12 as _rb12
            _rb12.prepack(gate_weight)
            hit = _rb12.get_packed(gate_weight)
            if hit is not None:
                _, slabs, outliers, o_off, base_exp = hit
                rc = lib.router_bf12_c32_gemm_fwd(
                    x.data_ptr(),
                    slabs.data_ptr(),
                    outliers.data_ptr(),
                    o_off.data_ptr(),
                    0 if nlive_buf is None else nlive_buf.data_ptr(),
                    ws["logits_ws"].data_ptr(),
                    tokens,
                    int(base_exp),
                    1 if C32_PDL else 0,
                    1 if (C32_PDL or want_rtr_prequant) else 0,
                    st,
                )
                if rc != 0:
                    raise RuntimeError(f"router_bf12_c32_gemm_fwd failed: rc={rc}")
                logits_bf16 = 0
                logits_ptr = ws["logits_ws"].data_ptr()
        except Exception:
            logits_bf16 = 1
            logits_ptr = 0

    if logits_ptr == 0:
        from .linear import k_group_partials
        partials = k_group_partials(x, gate_weight, out_dtype=torch.bfloat16)
        logits_bf16 = 1
        logits_ptr = partials.data_ptr()

    # Reconciled w/ #206: NEVER pass pdl_k2=3 on cuBLAS fallback (logits_bf16 != 0)!
    rtr_prequant = want_rtr_prequant and (logits_bf16 == 0) and (logits_ptr != 0)
    pdl_k2 = 3 if rtr_prequant else C32_PDL
    if _C48_V9_ANY and hasattr(lib, "router_c32_quant_route_place_v9_fwd"):
        rc = lib.router_c32_quant_route_place_v9_fwd(
            x.data_ptr(),
            logits_ptr,
            bias.data_ptr(),
            expert_bounds.data_ptr(),
            0 if is_padding is None else is_padding.data_ptr(),
            0 if nlive_buf is None else nlive_buf.data_ptr(),
            ws["counts"].data_ptr(),
            ws["counters"].data_ptr(),
            ws["ids"].data_ptr(),
            ws["ranks"].data_ptr(),
            ws["weights"].data_ptr(),
            ws["starts"].data_ptr(),
            ws["starts_compact"].data_ptr(),
            src2dst.data_ptr(),
            src2compact.data_ptr(),
            gammas.data_ptr(),
            routed.data_ptr(),
            tiles.data_ptr(),
            tile_compact_start.data_ptr(),
            row2tok.data_ptr(),
            0 if row_amax_w2 is None else row_amax_w2.data_ptr(),
            quantized[0].data_ptr(),
            quantized[1].data_ptr(),
            0 if quantized[2] is None else quantized[2].data_ptr(),
            ws["row_amax_in"].data_ptr(),
            0 if weight_amax is None else weight_amax.data_ptr(),
            0 if weight_denominator is None else weight_denominator.data_ptr(),
            tokens,
            float(route_scale),
            int(num_local),
            int(plan.tile),
            int(plan.max_ctas),
            1 if nvfp4 else 0,
            logits_bf16,
            pdl_k2,
            ROUTER_C32_QUANT_FMA,
            st,
        )
    else:
        rc = lib.router_c32_quant_route_place_fwd(
            x.data_ptr(),
            logits_ptr,
            bias.data_ptr(),
            expert_bounds.data_ptr(),
            0 if is_padding is None else is_padding.data_ptr(),
            0 if nlive_buf is None else nlive_buf.data_ptr(),
            ws["counts"].data_ptr(),
            ws["counters"].data_ptr(),
            ws["ids"].data_ptr(),
            ws["ranks"].data_ptr(),
            ws["weights"].data_ptr(),
            ws["starts"].data_ptr(),
            ws["starts_compact"].data_ptr(),
            src2dst.data_ptr(),
            src2compact.data_ptr(),
            gammas.data_ptr(),
            routed.data_ptr(),
            tiles.data_ptr(),
            tile_compact_start.data_ptr(),
            row2tok.data_ptr(),
            0 if row_amax_w2 is None else row_amax_w2.data_ptr(),
            quantized[0].data_ptr(),
            quantized[1].data_ptr(),
            0 if quantized[2] is None else quantized[2].data_ptr(),
            ws["row_amax_in"].data_ptr(),
            0 if weight_amax is None else weight_amax.data_ptr(),
            0 if weight_denominator is None else weight_denominator.data_ptr(),
            tokens,
            float(route_scale),
            int(num_local),
            int(plan.tile),
            int(plan.max_ctas),
            1 if nvfp4 else 0,
            logits_bf16,
            pdl_k2,
            st,
        )
    if rc != 0:
        raise RuntimeError(f"router_c32_quant_route_place_fwd failed: rc={rc}")

    quantized[0]._inkling_row2tok = row2tok  # type: ignore[attr-defined]
    quantized[1]._inkling_rowmaj = True  # type: ignore[attr-defined]
    tiles._inkling_tile_compact_start = tile_compact_start  # type: ignore[attr-defined]
    tiles._inkling_src2compact = src2compact  # type: ignore[attr-defined]
    if row_amax_w2 is not None:
        tiles._inkling_row_amax_w2 = row_amax_w2  # type: ignore[attr-defined]

    return (src2dst, gammas, routed, tiles), quantized


def bmm_c32_standalone(
    act: torch.Tensor,
    act_scale: torch.Tensor,
    row_scale: torch.Tensor | None,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    ones: torch.Tensor,
    tiles: torch.Tensor,
    plan,
    projection: int,
    *,
    row2tok: torch.Tensor | None = None,
    as_row_major: bool = False,
) -> torch.Tensor | None:
    """Standalone unfused C=4..8 / C=32 GEMM dispatch for `trtllm_bmm.bmm()` at `plan.tile in (8, 16, 32)`."""
    tile_ok = (
        (plan.tile in (16, 32) and plan.rows > 2048)
        or (plan.tile == 8 and C48_MOE_TILE8 and plan.max_ctas > 64)
    )
    if not C32_MASTER or not tile_ok or plan.max_ctas > _WS_MAX_CTAS:
        return None
    nvfp4 = weight.dtype == torch.uint8
    pack = 2 if nvfp4 else 1
    n = int(weight.shape[1])
    k = int(weight.shape[2]) * pack
    if not ((projection == 0 and n == 576 and k == 4608) or (projection == 1 and n == 4608 and k == 288)):
        return None

    device = act.device
    st = int(torch.cuda.current_stream(device).cuda_stream)
    out = torch.empty((plan.rows, n), dtype=torch.bfloat16, device=device)

    if nvfp4:
        lib = get_nvfp4_lib()
        if lib is None or (plan.tile == 8 and get_nvfp4_version() < 7):
            return None
        from . import nvfp4_ws5
        meta = (
            nvfp4_ws5.pack_w13_ws5_inplace(weight, weight_scale)
            if projection == 0
            else nvfp4_ws5.pack_w2_ws5_inplace(weight, weight_scale)
        )
        ws_base = get_ws_base_c32(meta)
        if projection == 0:
            zs = 1 if row2tok is not None else 0
            rm = 1 if (zs or as_row_major) else 0
            est_toks = int(act.shape[0]) if row2tok is not None else 0
            use_t32_bk256 = (plan.tile == 32 and 0 < est_toks <= 128 and get_nvfp4_version() >= 4)
            bk_dot = 256 if (plan.tile <= 16 or use_t32_bk256) else 128
            rc = lib.nvfp4_w13_ws5_c32_fwd(
                act.view(torch.uint8).data_ptr(),
                act_scale.view(torch.uint8).data_ptr(),
                row_scale.data_ptr(),
                0 if row2tok is None else row2tok.data_ptr(),
                meta["slabs"].data_ptr(),
                ws_base.data_ptr(),
                meta["p_off"].data_ptr(),
                meta["patches"].data_ptr(),
                ones.data_ptr(),
                tiles.data_ptr(),
                0, 0, 0,
                out.data_ptr(),
                n, k, int(plan.max_ctas), int(plan.tile),
                bk_dot, zs, rm, 0,
                1 if C32_PDL else 0, 1 if C32_PDL else 0, st,
            )
        else:
            use_t32_bk32 = (
                plan.tile == 32
                and plan.max_ctas <= 404
                and get_nvfp4_version() >= 4
                and _env_int("INKLING_C32_NVFP4_T32_MIN_M", 0) > 0
            )
            bk_dot = 32 if (plan.tile <= 16 or use_t32_bk32) else 64
            as_stride = int(act_scale.stride(0)) if (as_row_major and act_scale.ndim >= 2) else 18
            rc = lib.nvfp4_w2_ws5_c32_fwd(
                act.view(torch.uint8).data_ptr(),
                act_scale.view(torch.uint8).data_ptr(),
                row_scale.data_ptr(),
                0, 0, 0,
                meta["w_pack"].data_ptr(),
                meta["ws5_pack"].data_ptr(),
                ws_base.data_ptr(),
                meta["p_off"].data_ptr(),
                meta["patches"].data_ptr(),
                ones.data_ptr(),
                tiles.data_ptr(),
                0,
                out.data_ptr(),
                0, 0, 0, 0, 0,
                n, k, int(plan.max_ctas), int(plan.tile), 0,
                bk_dot, 1 if as_row_major else 0, as_stride, 0,
                0, 0,
                2 if C32_PDL else 0, 1 if C32_PDL else 0, st,
            )
        return out if rc == 0 else None

    # MXFP8 MX6-Huff7 v10 path (strictly opt-in via INKLING_MX6_EXACT=1)
    mx6_lib = get_mx6_lib()
    if mx6_lib is not None:
        from . import trtllm_bmm as _bmm_mx6
        pk = _bmm_mx6._mx6_get_packed(weight, weight_scale, projection)
        if pk is not None:
            sm4, ws3_pack, exp_stream, exp_off, warp_meta, huff_lut, base_ws = pk
            if projection == 0:
                zs = 1 if row2tok is not None else 0
                rm = 1 if (zs or as_row_major) else 0
                rc = mx6_lib.mx6_w13_c48_v10_fwd(
                    act.view(torch.uint8).data_ptr(),
                    act_scale.view(torch.uint8).data_ptr(),
                    0 if row2tok is None else row2tok.data_ptr(),
                    sm4.data_ptr(), ws3_pack.data_ptr(),
                    exp_stream.data_ptr(), exp_off.data_ptr(),
                    warp_meta.data_ptr(), huff_lut.data_ptr(),
                    ones.data_ptr(), tiles.data_ptr(),
                    0, 0, out.data_ptr(),
                    n, k, int(plan.max_ctas), int(plan.tile), int(base_ws),
                    zs, rm, 0, 0,
                    1 if C32_PDL else 0, 1 if C32_PDL else 0, st,
                )
            else:
                as_stride = int(act_scale.stride(0)) if (as_row_major and act_scale.ndim >= 2) else 9
                rc = mx6_lib.mx6_w2_c48_v10_fwd(
                    act.view(torch.uint8).data_ptr(),
                    act_scale.view(torch.uint8).data_ptr(),
                    sm4.data_ptr(), ws3_pack.data_ptr(),
                    exp_stream.data_ptr(), exp_off.data_ptr(),
                    warp_meta.data_ptr(), huff_lut.data_ptr(),
                    ones.data_ptr(), tiles.data_ptr(), 0, out.data_ptr(),
                    0, 0, 0, 0, 0,
                    n, k, int(plan.max_ctas), int(plan.tile), 0, int(base_ws),
                    1 if as_row_major else 0, as_stride, 0,
                    0, 0,
                    2 if C32_PDL else 0, 1 if C32_PDL else 0, st,
                )
            return out if rc == 0 else None

    # MXFP8 MX7 path
    if plan.tile == 32 and not C32_MX7_TILE32:
        return None
    lib = get_mx7_lib()
    if lib is None or (plan.tile == 8 and get_mx7_version() < 7):
        return None
    if MX7_V5_MODE == "panel" and get_mx7_version() >= 5 and hasattr(lib, "mx7_w13_c32_v5_panel_fwd"):
        pv5 = get_mx7_v5_panel(weight, weight_scale, projection)
        if pv5 is not None:
            panel_pool, panel_off, slab_meta, s_off, sflips, base_ws, _ = pv5
            if projection == 0:
                zs = 1 if row2tok is not None else 0
                rm = 1 if (zs or as_row_major) else 0
                rc = lib.mx7_w13_c32_v5_panel_fwd(
                    act.view(torch.uint8).data_ptr(),
                    act_scale.view(torch.uint8).data_ptr(),
                    0 if row2tok is None else row2tok.data_ptr(),
                    panel_pool.data_ptr(), panel_off.data_ptr(),
                    slab_meta.data_ptr(), s_off.data_ptr(), sflips.data_ptr(),
                    ones.data_ptr(), tiles.data_ptr(),
                    0, 0, out.data_ptr(),
                    n, k, int(plan.max_ctas), int(plan.tile), int(base_ws),
                    zs, rm, 0,
                    1 if C32_PDL else 0, 1 if C32_PDL else 0, st,
                )
            else:
                as_stride = int(act_scale.stride(0)) if (as_row_major and act_scale.ndim >= 2) else 9
                rc = lib.mx7_w2_c32_v5_panel_fwd(
                    act.view(torch.uint8).data_ptr(),
                    act_scale.view(torch.uint8).data_ptr(),
                    panel_pool.data_ptr(), panel_off.data_ptr(),
                    slab_meta.data_ptr(), s_off.data_ptr(), sflips.data_ptr(),
                    ones.data_ptr(), tiles.data_ptr(), 0, out.data_ptr(),
                    0, 0, 0, 0, 0,
                    n, k, int(plan.max_ctas), int(plan.tile), 0, int(base_ws),
                    1 if as_row_major else 0, as_stride, 0,
                    0, 0,
                    2 if C32_PDL else 0, 1 if C32_PDL else 0, st,
                )
            return out if rc == 0 else None
    elif MX7_V5_MODE == "compat" and get_mx7_version() >= 5 and hasattr(lib, "mx7_w13_c32_v5_fwd"):
        cv5 = get_mx7_v5_compat(weight, weight_scale, projection)
        if cv5 is not None:
            slabs, ws3, slab_meta, s_off, d_off, sflips, bmasks, base_ws, _ = cv5
            if projection == 0:
                zs = 1 if row2tok is not None else 0
                rm = 1 if (zs or as_row_major) else 0
                rc = lib.mx7_w13_c32_v5_fwd(
                    act.view(torch.uint8).data_ptr(),
                    act_scale.view(torch.uint8).data_ptr(),
                    0 if row2tok is None else row2tok.data_ptr(),
                    slabs.data_ptr(), ws3.data_ptr(),
                    slab_meta.data_ptr(), s_off.data_ptr(), d_off.data_ptr(),
                    sflips.data_ptr(), bmasks.data_ptr(),
                    ones.data_ptr(), tiles.data_ptr(),
                    0, 0, out.data_ptr(),
                    n, k, int(plan.max_ctas), int(plan.tile), int(base_ws),
                    zs, rm, 0,
                    1 if C32_PDL else 0, 1 if C32_PDL else 0, st,
                )
            else:
                as_stride = int(act_scale.stride(0)) if (as_row_major and act_scale.ndim >= 2) else 9
                rc = lib.mx7_w2_c32_v5_fwd(
                    act.view(torch.uint8).data_ptr(),
                    act_scale.view(torch.uint8).data_ptr(),
                    slabs.data_ptr(), ws3.data_ptr(),
                    slab_meta.data_ptr(), s_off.data_ptr(), d_off.data_ptr(),
                    sflips.data_ptr(), bmasks.data_ptr(),
                    ones.data_ptr(), tiles.data_ptr(), 0, out.data_ptr(),
                    0, 0, 0, 0, 0,
                    n, k, int(plan.max_ctas), int(plan.tile), 0, int(base_ws),
                    1 if as_row_major else 0, as_stride, 0,
                    0, 0,
                    2 if C32_PDL else 0, 1 if C32_PDL else 0, st,
                )
            return out if rc == 0 else None
    packed = get_mx7_packed_c32(weight, weight_scale, projection)
    if packed is None:
        return None
    slabs, ws3, flips, f_off, base_ws = packed
    if projection == 0:
        zs = 1 if row2tok is not None else 0
        rm = 1 if (zs or as_row_major) else 0
        rc = lib.mx7_w13_c32_fwd(
            act.view(torch.uint8).data_ptr(),
            act_scale.view(torch.uint8).data_ptr(),
            0 if row2tok is None else row2tok.data_ptr(),
            slabs.data_ptr(), ws3.data_ptr(), flips.data_ptr(), f_off.data_ptr(),
            ones.data_ptr(), tiles.data_ptr(),
            0, 0, out.data_ptr(),
            n, k, int(plan.max_ctas), int(plan.tile), int(base_ws),
            zs, rm, 0,
            1 if C32_PDL else 0, 1 if C32_PDL else 0, st,
        )
    else:
        as_stride = int(act_scale.stride(0)) if (as_row_major and act_scale.ndim >= 2) else 9
        rc = lib.mx7_w2_c32_fwd(
            act.view(torch.uint8).data_ptr(),
            act_scale.view(torch.uint8).data_ptr(),
            slabs.data_ptr(), ws3.data_ptr(), flips.data_ptr(), f_off.data_ptr(),
            ones.data_ptr(), tiles.data_ptr(), 0, out.data_ptr(),
            0, 0, 0, 0, 0,
            n, k, int(plan.max_ctas), int(plan.tile), 0, int(base_ws),
            1 if as_row_major else 0, as_stride, 0,
            0, 0,
            2 if C32_PDL else 0, 1 if C32_PDL else 0, st,
        )
    return out if rc == 0 else None


def forward_experts_c32(
    experts,
    x: torch.Tensor,
    routing: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    quantized: tuple[torch.Tensor, torch.Tensor, torch.Tensor | None],
    plan,
    *,
    out_buf: torch.Tensor | None = None,
    nlive_buf: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Execute full C=32 0-ULP fused W13 (+SiTU) -> W2 (+OTF Quant + 6-Wave L2 Combine).

    Returns unreduced combined `[T, 4608]` bf16 (in `out_buf` if provided), or `None`
    to cleanly fall back to stock `Inkling2Experts.forward`.
    """
    tokens = int(x.shape[0])
    if not c32_eligible(
        tokens, int(experts.hidden_size), int(experts.topk), plan,
        float(experts.gate_cap), float(experts.up_cap),
    ):
        return None
    if int(experts.w13_weight.shape[1]) != 576 or int(experts.w2_weight.shape[1]) != 4608:
        return None

    device = x.device
    out_dtype = torch.bfloat16 if x.dtype == torch.float8_e4m3fn else x.dtype
    if out_dtype != torch.bfloat16:
        return None

    src2dst, gammas, rows, tiles = routing
    act, act_scale, row_scale = quantized
    r2t = getattr(act, "_inkling_row2tok", None)
    as_rm = bool(getattr(act_scale, "_inkling_rowmaj", False))
    tile_compact_start = getattr(tiles, "_inkling_tile_compact_start", None)
    src2compact = getattr(tiles, "_inkling_src2compact", None)
    row_amax_w2 = getattr(tiles, "_inkling_row_amax_w2", None)

    has_compact = (tile_compact_start is not None) and (src2compact is not None)
    use_fuse_situ = C32_FUSE_SITU and has_compact and ((not experts.nvfp4) or (row_amax_w2 is not None))
    if C32_FUSE_COMBINE_C18 > 0 and has_compact and tokens <= C32_FUSE_COMBINE_C18_MAX_M:
        fc_mode = C32_FUSE_COMBINE_C18
    else:
        fc_mode = C32_FUSE_COMBINE if (has_compact and tokens <= C32_FUSE_COMBINE_MAX_M) else 0

    ws = get_workspace(device)
    st = int(torch.cuda.current_stream(device).cuda_stream)
    pdl_w13_wait = 1 if C32_PDL else 0
    pdl_w2_wait = 2 if C32_PDL else 0
    trig = 1 if C32_PDL else 0
    down_row_stride = tokens * 8  # compact rows in [0, 8*T)
    hot_l2_flag = 1 if C48_MOE_HOT_L2_HINT else 0
    tile_w13_done_ptr = ws["tile_w13_done"].data_ptr()
    tile_ready_ptr = ws["tile_ready"].data_ptr()
    overlap_meta_ptr = ws["overlap_meta"].data_ptr()
    work_queue_ptr = ws["work_queue"].data_ptr()
    # V17 fix: disjoint 16B-aligned WorkQueue[4..5] slice for W2 so C32_PDL=2 never races
    # with W13's WorkQueue[0..1] atomic counter under LPT=2 persistent scheduling!
    work_queue_w2_ptr = work_queue_ptr + 16

    if experts.nvfp4:
        lib = get_nvfp4_lib()
        if lib is None:
            return None
        from . import nvfp4_ws5
        m13 = nvfp4_ws5.pack_w13_ws5_inplace(experts.w13_weight, experts.w13_weight_scale)
        m2 = nvfp4_ws5.pack_w2_ws5_inplace(experts.w2_weight, experts.w2_weight_scale)
        ws_base_13 = get_ws_base_c32(m13)
        ws_base_2 = get_ws_base_c32(m2)
        nv_ver = get_nvfp4_version()
        if plan.tile == 8 and nv_ver < 7:
            return None
        # Keep BK_DOT=256/32 for tokens<=128 even under TILE=32 (v4 t32_bk256 + w2_bk32 -> literal nd == 0!)
        use_low_m_bk = (plan.tile <= 16) or (tokens <= 128 and nv_ver >= 4)
        bk_w13 = 256 if use_low_m_bk else 128
        bk_w2 = 32 if use_low_m_bk else 64
        nv_fc_req = (
            C32_NVFP4_FC_MODE
            if (C32_NVFP4_FC_MODE >= 0 and has_compact and tokens <= C32_NVFP4_FC_MAX_M)
            else fc_mode
        )
        if nv_fc_req == 3:
            nv_fc = 3 if nv_ver >= 6 else 0
        elif nv_fc_req == 2:
            nv_fc = 2 if nv_ver >= 4 else 0
        else:
            nv_fc = nv_fc_req

        use_nv_v16 = (
            NVFP4_C48_V16
            and nv_ver >= 16
            and hasattr(lib, "nvfp4_w13_ws5_c48_v16_fwd")
            and hasattr(lib, "nvfp4_w2_ws5_c48_v16_fwd")
        )
        use_nv_v9 = (
            _C48_V9_ANY
            and nv_ver >= 9
            and hasattr(lib, "nvfp4_w13_ws5_c48_v9_fwd")
            and hasattr(lib, "nvfp4_w2_ws5_c48_v9_fwd")
        )
        use_v10_fuse_qdeq = (
            use_nv_v9
            and NVFP4_V10_FUSE_QDEQ
            and use_fuse_situ
            and nv_ver >= 10
            and hasattr(lib, "nvfp4_w13_ws5_c48_v10_fwd")
            and int(plan.max_ctas) <= _WS_MAX_CTAS
        )
        use_v9_qdeq = (
            use_v10_fuse_qdeq
            or (
                use_nv_v9
                and NVFP4_V9_QDEQ
                and use_fuse_situ
                and hasattr(lib, "nvfp4_situ_quant_dequant_c48_v9_fwd")
            )
        )
        if (
            NVFP4_V12_BN16_SINGLE_MAX_M > 0
            and tokens <= NVFP4_V12_BN16_SINGLE_MAX_M
            and int(plan.tile) == 8
        ):
            eff_v9_bn_mode = 1
        elif NVFP4_V10_BN16_MAX_M > 0 and tokens > NVFP4_V10_BN16_MAX_M:
            eff_v9_bn_mode = 0
        else:
            eff_v9_bn_mode = NVFP4_V9_W13_BN_MODE if int(plan.tile) == 8 else 0
        use_nv_v8 = (
            (not use_nv_v9)
            and nv_ver >= 8
            and hasattr(lib, "nvfp4_w13_ws5_c48_v8_fwd")
            and hasattr(lib, "nvfp4_w2_ws5_c48_v8_fwd")
            and (C48_MOE_W13_W2_OVERLAP or C48_MOE_HOT_L2_HINT)
        )
        nv_overlap_flag = (
            1
            if (
                use_nv_v8
                and C48_MOE_W13_W2_OVERLAP
                and (C32_PDL > 0)
                and use_fuse_situ
                and int(plan.max_ctas) <= 512
            )
            else 0
        )

        def _call_nv_w13(
            out_ptr: int, tcs_ptr: int, gam_ptr: int, amax_ptr: int, fs_flag: int
        ) -> int:
            if use_nv_v16:
                fq_flag = 1 if (use_v10_fuse_qdeq and fs_flag) else 0
                return lib.nvfp4_w13_ws5_c48_v16_fwd(
                    act.view(torch.uint8).data_ptr(),
                    act_scale.view(torch.uint8).data_ptr(),
                    row_scale.data_ptr(),
                    0 if r2t is None else r2t.data_ptr(),
                    m13["slabs"].data_ptr(),
                    ws_base_13.data_ptr(),
                    m13["p_off"].data_ptr(),
                    m13["patches"].data_ptr(),
                    experts.ones.data_ptr(),
                    tiles.data_ptr(),
                    tcs_ptr,
                    gam_ptr,
                    amax_ptr,
                    out_ptr,
                    tile_w13_done_ptr if fq_flag else 0,
                    576, 4608, int(plan.max_ctas), int(plan.tile),
                    bk_w13, 1 if r2t is not None else 0, 1 if (r2t is not None or as_rm) else 0, fs_flag,
                    1 if (use_v9_qdeq and fs_flag) else 0,
                    eff_v9_bn_mode,
                    NVFP4_V9_BN16_MARGIN,
                    pdl_w13_wait, trig, fq_flag, MOE_V9_QUANT_FMA,
                    NVFP4_V16_EVICT, NVFP4_V16_LPT, work_queue_ptr, st,
                )
            if use_v10_fuse_qdeq and fs_flag:
                return lib.nvfp4_w13_ws5_c48_v10_fwd(
                    act.view(torch.uint8).data_ptr(),
                    act_scale.view(torch.uint8).data_ptr(),
                    row_scale.data_ptr(),
                    0 if r2t is None else r2t.data_ptr(),
                    m13["slabs"].data_ptr(),
                    ws_base_13.data_ptr(),
                    m13["p_off"].data_ptr(),
                    m13["patches"].data_ptr(),
                    experts.ones.data_ptr(),
                    tiles.data_ptr(),
                    tcs_ptr,
                    gam_ptr,
                    amax_ptr,
                    out_ptr,
                    tile_w13_done_ptr,
                    576, 4608, int(plan.max_ctas), int(plan.tile),
                    bk_w13, 1 if r2t is not None else 0, 1 if (r2t is not None or as_rm) else 0, fs_flag,
                    1,
                    eff_v9_bn_mode,
                    NVFP4_V9_BN16_MARGIN,
                    pdl_w13_wait, trig, 1, MOE_V9_QUANT_FMA, st,
                )
            if use_nv_v9:
                return lib.nvfp4_w13_ws5_c48_v9_fwd(
                    act.view(torch.uint8).data_ptr(),
                    act_scale.view(torch.uint8).data_ptr(),
                    row_scale.data_ptr(),
                    0 if r2t is None else r2t.data_ptr(),
                    m13["slabs"].data_ptr(),
                    ws_base_13.data_ptr(),
                    m13["p_off"].data_ptr(),
                    m13["patches"].data_ptr(),
                    experts.ones.data_ptr(),
                    tiles.data_ptr(),
                    tcs_ptr,
                    gam_ptr,
                    amax_ptr,
                    out_ptr,
                    576, 4608, int(plan.max_ctas), int(plan.tile),
                    bk_w13, 1 if r2t is not None else 0, 1 if (r2t is not None or as_rm) else 0, fs_flag,
                    1 if (use_v9_qdeq and fs_flag) else 0,
                    eff_v9_bn_mode,
                    NVFP4_V9_BN16_MARGIN,
                    pdl_w13_wait, trig, st,
                )
            if use_nv_v8:
                return lib.nvfp4_w13_ws5_c48_v8_fwd(
                    act.view(torch.uint8).data_ptr(),
                    act_scale.view(torch.uint8).data_ptr(),
                    row_scale.data_ptr(),
                    0 if r2t is None else r2t.data_ptr(),
                    m13["slabs"].data_ptr(),
                    ws_base_13.data_ptr(),
                    m13["p_off"].data_ptr(),
                    m13["patches"].data_ptr(),
                    experts.ones.data_ptr(),
                    tiles.data_ptr(),
                    tcs_ptr,
                    gam_ptr,
                    amax_ptr,
                    out_ptr,
                    576, 4608, int(plan.max_ctas), int(plan.tile),
                    bk_w13, 1 if r2t is not None else 0, 1 if (r2t is not None or as_rm) else 0, fs_flag,
                    pdl_w13_wait, trig,
                    tile_w13_done_ptr if nv_overlap_flag else 0,
                    tile_ready_ptr if nv_overlap_flag else 0,
                    overlap_meta_ptr if nv_overlap_flag else 0,
                    nv_overlap_flag if fs_flag else 0,
                    hot_l2_flag,
                    st,
                )
            return lib.nvfp4_w13_ws5_c32_fwd(
                act.view(torch.uint8).data_ptr(),
                act_scale.view(torch.uint8).data_ptr(),
                row_scale.data_ptr(),
                0 if r2t is None else r2t.data_ptr(),
                m13["slabs"].data_ptr(),
                ws_base_13.data_ptr(),
                m13["p_off"].data_ptr(),
                m13["patches"].data_ptr(),
                experts.ones.data_ptr(),
                tiles.data_ptr(),
                tcs_ptr,
                gam_ptr,
                amax_ptr,
                out_ptr,
                576, 4608, int(plan.max_ctas), int(plan.tile),
                bk_w13, 1 if r2t is not None else 0, 1 if (r2t is not None or as_rm) else 0, fs_flag,
                pdl_w13_wait, trig, st,
            )

        def _call_nv_w2(
            tcs_ptr: int, out_ptr: int, s2c_ptr: int, nl_ptr: int,
            pd_ptr: int, bg_ptr: int, sb_ptr: int, fc_flag: int,
        ) -> int:
            if use_nv_v16:
                fq_mode = 2 if use_v9_qdeq else (1 if use_fuse_situ else 0)
                return lib.nvfp4_w2_ws5_c48_v16_fwd(
                    w2_a_ptr, w2_as_ptr, w2_rs_ptr,
                    row_amax_w2.data_ptr() if use_fuse_situ else 0,
                    w2_amax.data_ptr() if use_fuse_situ else 0,
                    w2_denom.data_ptr() if use_fuse_situ else 0,
                    m2["w_pack"].data_ptr(), m2["ws5_pack"].data_ptr(), ws_base_2.data_ptr(),
                    m2["p_off"].data_ptr(), m2["patches"].data_ptr(),
                    experts.ones.data_ptr(), tiles.data_ptr(), tcs_ptr,
                    out_ptr, s2c_ptr, nl_ptr, pd_ptr, bg_ptr, sb_ptr,
                    4608, 288, int(plan.max_ctas), int(plan.tile), tokens,
                    bk_w2, w2_as_rm, w2_as_stride, down_row_stride,
                    fq_mode, fc_flag,
                    pdl_w2_wait, trig, MOE_V9_QUANT_FMA,
                    NVFP4_V16_EVICT, NVFP4_V16_LPT, work_queue_w2_ptr, st,
                )
            if use_nv_v9:
                fq_mode = 2 if use_v9_qdeq else (1 if use_fuse_situ else 0)
                return lib.nvfp4_w2_ws5_c48_v9_fwd(
                    w2_a_ptr, w2_as_ptr, w2_rs_ptr,
                    row_amax_w2.data_ptr() if use_fuse_situ else 0,
                    w2_amax.data_ptr() if use_fuse_situ else 0,
                    w2_denom.data_ptr() if use_fuse_situ else 0,
                    m2["w_pack"].data_ptr(), m2["ws5_pack"].data_ptr(), ws_base_2.data_ptr(),
                    m2["p_off"].data_ptr(), m2["patches"].data_ptr(),
                    experts.ones.data_ptr(), tiles.data_ptr(), tcs_ptr,
                    out_ptr, s2c_ptr, nl_ptr, pd_ptr, bg_ptr, sb_ptr,
                    4608, 288, int(plan.max_ctas), int(plan.tile), tokens,
                    bk_w2, w2_as_rm, w2_as_stride, down_row_stride,
                    fq_mode, fc_flag,
                    pdl_w2_wait, trig, MOE_V9_QUANT_FMA, st,
                )
            if use_nv_v8:
                return lib.nvfp4_w2_ws5_c48_v8_fwd(
                    w2_a_ptr, w2_as_ptr, w2_rs_ptr,
                    row_amax_w2.data_ptr() if use_fuse_situ else 0,
                    w2_amax.data_ptr() if use_fuse_situ else 0,
                    w2_denom.data_ptr() if use_fuse_situ else 0,
                    m2["w_pack"].data_ptr(), m2["ws5_pack"].data_ptr(), ws_base_2.data_ptr(),
                    m2["p_off"].data_ptr(), m2["patches"].data_ptr(),
                    experts.ones.data_ptr(), tiles.data_ptr(), tcs_ptr,
                    out_ptr, s2c_ptr, nl_ptr, pd_ptr, bg_ptr, sb_ptr,
                    4608, 288, int(plan.max_ctas), int(plan.tile), tokens,
                    bk_w2, w2_as_rm, w2_as_stride, down_row_stride,
                    1 if use_fuse_situ else 0, fc_flag,
                    pdl_w2_wait, trig,
                    tile_ready_ptr if nv_overlap_flag else 0,
                    overlap_meta_ptr if nv_overlap_flag else 0,
                    nv_overlap_flag,
                    hot_l2_flag,
                    st,
                )
            return lib.nvfp4_w2_ws5_c32_fwd(
                w2_a_ptr, w2_as_ptr, w2_rs_ptr,
                row_amax_w2.data_ptr() if use_fuse_situ else 0,
                w2_amax.data_ptr() if use_fuse_situ else 0,
                w2_denom.data_ptr() if use_fuse_situ else 0,
                m2["w_pack"].data_ptr(), m2["ws5_pack"].data_ptr(), ws_base_2.data_ptr(),
                m2["p_off"].data_ptr(), m2["patches"].data_ptr(),
                experts.ones.data_ptr(), tiles.data_ptr(), tcs_ptr,
                out_ptr, s2c_ptr, nl_ptr, pd_ptr, bg_ptr, sb_ptr,
                4608, 288, int(plan.max_ctas), int(plan.tile), tokens,
                bk_w2, w2_as_rm, w2_as_stride, down_row_stride,
                1 if use_fuse_situ else 0, fc_flag,
                pdl_w2_wait, trig, st,
            )

        if use_fuse_situ:
            situ_out = ws["situ_out"][:down_row_stride]
            rc = _call_nv_w13(
                situ_out.data_ptr(), tile_compact_start.data_ptr(),
                gammas.data_ptr(), row_amax_w2.data_ptr(), 1,
            )
            if rc != 0:
                raise RuntimeError(f"nvfp4_w13_ws5_c32_fwd (fuse_situ=1) failed: rc={rc}")
            if use_v9_qdeq and not use_v10_fuse_qdeq:
                rc = lib.nvfp4_situ_quant_dequant_c48_v9_fwd(
                    situ_out.data_ptr(),
                    situ_out.data_ptr(),
                    row_amax_w2.data_ptr(),
                    tiles.data_ptr(),
                    tile_compact_start.data_ptr(),
                    int(plan.max_ctas),
                    int(plan.tile),
                    down_row_stride,
                    1 if C32_PDL else 0,
                    trig,
                    MOE_V9_QUANT_FMA,
                    st,
                )
                if rc != 0:
                    raise RuntimeError(f"nvfp4_situ_quant_dequant_c48_v9_fwd failed: rc={rc}")
            w2_a_ptr = situ_out.data_ptr()
            w2_as_ptr = 0
            w2_rs_ptr = 0
            w2_as_rm = 0
            w2_as_stride = 18
        else:
            gate_up = torch.empty((plan.rows, 576), dtype=torch.bfloat16, device=device)
            rc = _call_nv_w13(gate_up.data_ptr(), 0, 0, 0, 0)
            if rc != 0:
                raise RuntimeError(f"nvfp4_w13_ws5_c32_fwd (fuse_situ=0) failed: rc={rc}")
            from .activation import swiglu_quantize
            m = plan.max_ctas
            act2_v, act2_s, act2_rs = swiglu_quantize(
                gate_up, gammas, experts.gate_cap, experts.up_cap,
                (rows, tiles[2 * m + 2 :], tiles[:m]), plan.tile, *experts.scales("w2"),
            )
            del gate_up
            w2_a_ptr = act2_v.view(torch.uint8).data_ptr()
            w2_as_ptr = act2_s.view(torch.uint8).data_ptr()
            w2_rs_ptr = act2_rs.data_ptr()
            w2_as_rm = 1 if bool(getattr(act2_s, "_inkling_rowmaj", False)) else 0
            w2_as_stride = int(act2_s.stride(0)) if (w2_as_rm and act2_s.ndim >= 2) else 18

        w2_amax, w2_denom = experts.scales("w2")
        if nv_fc > 0:
            out = out_buf if out_buf is not None else torch.empty(
                (tokens, 4608), dtype=torch.bfloat16, device=device
            )
            down_ws = ws["down_compact"] if nv_fc in (2, 3) else ws["down_panel"]
            rc = _call_nv_w2(
                tile_compact_start.data_ptr(),
                down_ws.data_ptr(),
                src2compact.data_ptr(),
                0 if nlive_buf is None else nlive_buf.data_ptr(),
                ws["panel_done"].data_ptr(),
                0 if nv_fc in (2, 3) else ws["buf_gen"].data_ptr(),
                out.data_ptr(),
                nv_fc,
            )
            if rc != 0:
                raise RuntimeError(f"nvfp4_w2_ws5_c32_fwd (fuse_combine={nv_fc}) failed: rc={rc}")
            return out

        if (
            (C32_COMBINE_V6 or C48_MOE_V8 or C48_MOE_V9 or C48_MOE_V10 or C48_MOE_V12)
            and has_compact
            and nv_ver >= 6
            and hasattr(lib, "nvfp4_combine_c32_v6_fwd")
        ):
            out = out_buf if out_buf is not None else torch.empty(
                (tokens, 4608), dtype=torch.bfloat16, device=device
            )
            down_compact_row = ws["down_compact"]
            rc = _call_nv_w2(
                tile_compact_start.data_ptr(),
                down_compact_row.data_ptr(),
                0, 0, 0, 0, 0, 4,
            )
            if rc != 0:
                raise RuntimeError(f"nvfp4_w2_ws5_c32_fwd (fuse_combine=4) failed: rc={rc}")
            rc = lib.nvfp4_combine_c32_v6_fwd(
                down_compact_row.data_ptr(),
                src2compact.data_ptr(),
                0 if nlive_buf is None else nlive_buf.data_ptr(),
                out.data_ptr(),
                tokens, 4608,
                1 if C32_PDL else 0, trig, st,
            )
            if rc != 0:
                raise RuntimeError(f"nvfp4_combine_c32_v6_fwd failed: rc={rc}")
            return out

        down = torch.empty((plan.rows, 4608), dtype=torch.bfloat16, device=device)
        rc = _call_nv_w2(
            tile_compact_start.data_ptr() if use_fuse_situ else 0,
            down.data_ptr(),
            0, 0, 0, 0, 0, 0,
        )
        if rc != 0:
            raise RuntimeError(f"nvfp4_w2_ws5_c32_fwd (fuse_combine=0) failed: rc={rc}")
        from .moe import combine
        return combine(down, src2dst, experts.topk, out_dtype, out=out_buf)

    # ---- MXFP8 MX6-Huff7 v10 (L40..47) C=2..8 & C=16..32 path ----
    mx6_lib = get_mx6_lib()
    if mx6_lib is not None:
        from . import trtllm_bmm as _bmm_mx6
        pk13 = _bmm_mx6._mx6_get_packed(experts.w13_weight, experts.w13_weight_scale, 0)
        pk2 = _bmm_mx6._mx6_get_packed(experts.w2_weight, experts.w2_weight_scale, 1)
        if pk13 is not None and pk2 is not None:
            sm4_13, ws3_13, es13, eo13, wm13, lut13, bws13 = pk13
            sm4_2, ws3_2, es2, eo2, wm2, lut2, bws2 = pk2
            mx6_bn_mode = int(MX6_W13_BN_MODE) if int(plan.tile) == 8 else 0
            if use_fuse_situ:
                situ_out = ws["situ_out"][:down_row_stride]
                rc = mx6_lib.mx6_w13_c48_v10_fwd(
                    act.view(torch.uint8).data_ptr(),
                    act_scale.view(torch.uint8).data_ptr(),
                    0 if r2t is None else r2t.data_ptr(),
                    sm4_13.data_ptr(), ws3_13.data_ptr(),
                    es13.data_ptr(), eo13.data_ptr(),
                    wm13.data_ptr(), lut13.data_ptr(),
                    experts.ones.data_ptr(), tiles.data_ptr(),
                    tile_compact_start.data_ptr(), gammas.data_ptr(), situ_out.data_ptr(),
                    576, 4608, int(plan.max_ctas), int(plan.tile), int(bws13),
                    1 if r2t is not None else 0, 1 if (r2t is not None or as_rm) else 0, 1,
                    mx6_bn_mode, pdl_w13_wait, trig, st,
                )
                if rc != 0:
                    raise RuntimeError(f"mx6_w13_c48_v10_fwd (fuse_situ=1) failed: rc={rc}")
                w2_a_ptr = situ_out.data_ptr()
                w2_as_ptr = 0
                w2_as_rm = 0
                w2_as_stride = 9
            else:
                gate_up = torch.empty((plan.rows, 576), dtype=torch.bfloat16, device=device)
                rc = mx6_lib.mx6_w13_c48_v10_fwd(
                    act.view(torch.uint8).data_ptr(),
                    act_scale.view(torch.uint8).data_ptr(),
                    0 if r2t is None else r2t.data_ptr(),
                    sm4_13.data_ptr(), ws3_13.data_ptr(),
                    es13.data_ptr(), eo13.data_ptr(),
                    wm13.data_ptr(), lut13.data_ptr(),
                    experts.ones.data_ptr(), tiles.data_ptr(),
                    0, 0, gate_up.data_ptr(),
                    576, 4608, int(plan.max_ctas), int(plan.tile), int(bws13),
                    1 if r2t is not None else 0, 1 if (r2t is not None or as_rm) else 0, 0,
                    mx6_bn_mode, pdl_w13_wait, trig, st,
                )
                if rc != 0:
                    raise RuntimeError(f"mx6_w13_c48_v10_fwd (fuse_situ=0) failed: rc={rc}")
                from .activation import swiglu_quantize
                m = plan.max_ctas
                act2_v, act2_s, _ = swiglu_quantize(
                    gate_up, gammas, experts.gate_cap, experts.up_cap,
                    (rows, tiles[2 * m + 2 :], tiles[:m]), plan.tile, None, None,
                )
                del gate_up
                w2_a_ptr = act2_v.view(torch.uint8).data_ptr()
                w2_as_ptr = act2_s.view(torch.uint8).data_ptr()
                w2_as_rm = 1 if bool(getattr(act2_s, "_inkling_rowmaj", False)) else 0
                w2_as_stride = int(act2_s.stride(0)) if (w2_as_rm and act2_s.ndim >= 2) else 9

            mx6_fc = fc_mode if fc_mode in (0, 1, 2, 3) else 0
            if mx6_fc > 0:
                out = out_buf if out_buf is not None else torch.empty(
                    (tokens, 4608), dtype=torch.bfloat16, device=device
                )
                down_ws = ws["down_compact"] if mx6_fc in (2, 3) else ws["down_panel"]
                rc = mx6_lib.mx6_w2_c48_v10_fwd(
                    w2_a_ptr, w2_as_ptr,
                    sm4_2.data_ptr(), ws3_2.data_ptr(),
                    es2.data_ptr(), eo2.data_ptr(),
                    wm2.data_ptr(), lut2.data_ptr(),
                    experts.ones.data_ptr(), tiles.data_ptr(), tile_compact_start.data_ptr(),
                    down_ws.data_ptr(),
                    src2compact.data_ptr(),
                    0 if nlive_buf is None else nlive_buf.data_ptr(),
                    ws["panel_done"].data_ptr(),
                    0 if mx6_fc in (2, 3) else ws["buf_gen"].data_ptr(),
                    out.data_ptr(),
                    4608, 288, int(plan.max_ctas), int(plan.tile), tokens, int(bws2),
                    w2_as_rm, w2_as_stride, down_row_stride,
                    1 if use_fuse_situ else 0, mx6_fc,
                    pdl_w2_wait, trig, st,
                )
                if rc != 0:
                    raise RuntimeError(f"mx6_w2_c48_v10_fwd (fuse_combine={mx6_fc}) failed: rc={rc}")
                return out

            if (C32_COMBINE_V6 or C48_MOE_V8 or C48_MOE_V9 or C48_MOE_V10) and has_compact:
                out = out_buf if out_buf is not None else torch.empty(
                    (tokens, 4608), dtype=torch.bfloat16, device=device
                )
                down_compact_row = ws["down_compact"]
                rc = mx6_lib.mx6_w2_c48_v10_fwd(
                    w2_a_ptr, w2_as_ptr,
                    sm4_2.data_ptr(), ws3_2.data_ptr(),
                    es2.data_ptr(), eo2.data_ptr(),
                    wm2.data_ptr(), lut2.data_ptr(),
                    experts.ones.data_ptr(), tiles.data_ptr(), tile_compact_start.data_ptr(),
                    down_compact_row.data_ptr(),
                    0, 0, 0, 0, 0,
                    4608, 288, int(plan.max_ctas), int(plan.tile), tokens, int(bws2),
                    w2_as_rm, w2_as_stride, down_row_stride,
                    1 if use_fuse_situ else 0, 4,
                    pdl_w2_wait, trig, st,
                )
                if rc != 0:
                    raise RuntimeError(f"mx6_w2_c48_v10_fwd (fuse_combine=4) failed: rc={rc}")
                rc = mx6_lib.mx6_combine_c48_v10_fwd(
                    down_compact_row.data_ptr(),
                    src2compact.data_ptr(),
                    0 if nlive_buf is None else nlive_buf.data_ptr(),
                    out.data_ptr(),
                    tokens, 4608,
                    1 if C32_PDL else 0, trig, st,
                )
                if rc != 0:
                    raise RuntimeError(f"mx6_combine_c48_v10_fwd failed: rc={rc}")
                return out

            down = torch.empty((plan.rows, 4608), dtype=torch.bfloat16, device=device)
            rc = mx6_lib.mx6_w2_c48_v10_fwd(
                w2_a_ptr, w2_as_ptr,
                sm4_2.data_ptr(), ws3_2.data_ptr(),
                es2.data_ptr(), eo2.data_ptr(),
                wm2.data_ptr(), lut2.data_ptr(),
                experts.ones.data_ptr(), tiles.data_ptr(),
                tile_compact_start.data_ptr() if use_fuse_situ else 0,
                down.data_ptr(),
                0, 0, 0, 0, 0,
                4608, 288, int(plan.max_ctas), int(plan.tile), tokens, int(bws2),
                w2_as_rm, w2_as_stride, down_row_stride,
                1 if use_fuse_situ else 0, 0,
                pdl_w2_wait, trig, st,
            )
            if rc != 0:
                raise RuntimeError(f"mx6_w2_c48_v10_fwd (fuse_combine=0) failed: rc={rc}")
            from .moe import combine
            return combine(down, src2dst, experts.topk, out_dtype, out=out_buf)

    # ---- MXFP8 MX7 (L40..47) C=32 path ----
    if plan.tile == 32 and not C32_MX7_TILE32:
        return None
    lib = get_mx7_lib()
    if lib is None:
        return None
    mx_ver = get_mx7_version()
    if plan.tile == 8 and mx_ver < 7:
        return None
    v5_mode = "legacy"
    v5_13 = None
    v5_2 = None
    if MX7_V5_MODE == "panel" and mx_ver >= 5 and hasattr(lib, "mx7_w13_c32_v5_panel_fwd"):
        v5_13 = get_mx7_v5_panel(experts.w13_weight, experts.w13_weight_scale, 0)
        v5_2 = get_mx7_v5_panel(experts.w2_weight, experts.w2_weight_scale, 1)
        if v5_13 is not None and v5_2 is not None:
            v5_mode = "panel"
    elif MX7_V5_MODE == "compat" and mx_ver >= 5 and hasattr(lib, "mx7_w13_c32_v5_fwd"):
        v5_13 = get_mx7_v5_compat(experts.w13_weight, experts.w13_weight_scale, 0)
        v5_2 = get_mx7_v5_compat(experts.w2_weight, experts.w2_weight_scale, 1)
        if v5_13 is not None and v5_2 is not None:
            v5_mode = "compat"

    if v5_mode == "legacy":
        p13 = get_mx7_packed_c32(experts.w13_weight, experts.w13_weight_scale, 0)
        p2 = get_mx7_packed_c32(experts.w2_weight, experts.w2_weight_scale, 1)
        if p13 is None or p2 is None:
            return None
        slabs13, ws3_13, flips13, f_off13, base_ws13 = p13
        slabs2, ws3_2, flips2, f_off2, base_ws2 = p2

    use_mx_v16 = (
        MX7_C48_V16
        and mx_ver >= 16
        and hasattr(lib, "mx7_w13_c48_v16_fwd")
        and hasattr(lib, "mx7_w2_c48_v16_fwd")
    )
    use_mx_v14 = (
        (MX7_C48_V14 or MX7_V14_FUSE_QDEQ)
        and mx_ver >= 14
        and hasattr(lib, "mx7_w13_c48_v14_fwd")
        and hasattr(lib, "mx7_w2_c48_v14_fwd")
    )
    use_mx_v14_qdeq = (
        use_mx_v14
        and MX7_V14_FUSE_QDEQ
        and use_fuse_situ
        and int(plan.max_ctas) <= _WS_MAX_CTAS
    )
    if MX7_V14_BN16_MAX_M > 0 and tokens > MX7_V14_BN16_MAX_M:
        eff_mx_bn_mode = 0
    else:
        eff_mx_bn_mode = MX7_V14_W13_BN_MODE if int(plan.tile) == 8 else 0
    mx_fq_w2 = (2 if use_mx_v14_qdeq else 1) if use_fuse_situ else 0

    use_mx_v8 = (
        (not use_mx_v14)
        and mx_ver >= 8
        and hasattr(lib, "mx7_w13_c48_v8_fwd")
        and hasattr(lib, "mx7_w2_c48_v8_fwd")
        and (C48_MOE_W13_W2_OVERLAP or C48_MOE_HOT_L2_HINT)
    )
    mx_overlap_flag = (
        1
        if (
            (use_mx_v14 or use_mx_v8)
            and C48_MOE_W13_W2_OVERLAP
            and (C32_PDL > 0)
            and use_fuse_situ
            and int(plan.max_ctas) <= 512
        )
        else 0
    )

    def _call_w13(out_ptr: int, tcs_ptr: int, gam_ptr: int, fs_flag: int) -> int:
        ov_flag = mx_overlap_flag if fs_flag else 0
        qdeq_flag = 1 if (use_mx_v14_qdeq and fs_flag) else 0
        qdeq_ptr = tile_w13_done_ptr if qdeq_flag else 0
        if v5_mode == "panel":
            pp13, po13, sm13, so13, sf13, bws13, _ = v5_13
            if use_mx_v16 and hasattr(lib, "mx7_w13_c48_v16_panel_fwd"):
                return lib.mx7_w13_c48_v16_panel_fwd(
                    act.view(torch.uint8).data_ptr(),
                    act_scale.view(torch.uint8).data_ptr(),
                    0 if r2t is None else r2t.data_ptr(),
                    pp13.data_ptr(), po13.data_ptr(),
                    sm13.data_ptr(), so13.data_ptr(), sf13.data_ptr(),
                    experts.ones.data_ptr(), tiles.data_ptr(),
                    tcs_ptr, gam_ptr, out_ptr, qdeq_ptr,
                    576, 4608, int(plan.max_ctas), int(plan.tile), int(bws13),
                    1 if r2t is not None else 0, 1 if (r2t is not None or as_rm) else 0, fs_flag, qdeq_flag,
                    eff_mx_bn_mode, MX7_V14_BN16_MARGIN,
                    pdl_w13_wait, trig,
                    tile_w13_done_ptr if ov_flag else 0,
                    tile_ready_ptr if ov_flag else 0,
                    overlap_meta_ptr if ov_flag else 0,
                    ov_flag, hot_l2_flag, MX7_V14_EVICT, MX7_V16_LPT, work_queue_ptr, st,
                )
            if use_mx_v14 and hasattr(lib, "mx7_w13_c48_v14_panel_fwd"):
                return lib.mx7_w13_c48_v14_panel_fwd(
                    act.view(torch.uint8).data_ptr(),
                    act_scale.view(torch.uint8).data_ptr(),
                    0 if r2t is None else r2t.data_ptr(),
                    pp13.data_ptr(), po13.data_ptr(),
                    sm13.data_ptr(), so13.data_ptr(), sf13.data_ptr(),
                    experts.ones.data_ptr(), tiles.data_ptr(),
                    tcs_ptr, gam_ptr, out_ptr, qdeq_ptr,
                    576, 4608, int(plan.max_ctas), int(plan.tile), int(bws13),
                    1 if r2t is not None else 0, 1 if (r2t is not None or as_rm) else 0, fs_flag, qdeq_flag,
                    eff_mx_bn_mode, MX7_V14_BN16_MARGIN,
                    pdl_w13_wait, trig,
                    tile_w13_done_ptr if ov_flag else 0,
                    tile_ready_ptr if ov_flag else 0,
                    overlap_meta_ptr if ov_flag else 0,
                    ov_flag, hot_l2_flag, MX7_V14_EVICT, st,
                )
            if use_mx_v8 and hasattr(lib, "mx7_w13_c48_v8_panel_fwd"):
                return lib.mx7_w13_c48_v8_panel_fwd(
                    act.view(torch.uint8).data_ptr(),
                    act_scale.view(torch.uint8).data_ptr(),
                    0 if r2t is None else r2t.data_ptr(),
                    pp13.data_ptr(), po13.data_ptr(),
                    sm13.data_ptr(), so13.data_ptr(), sf13.data_ptr(),
                    experts.ones.data_ptr(), tiles.data_ptr(),
                    tcs_ptr, gam_ptr, out_ptr,
                    576, 4608, int(plan.max_ctas), int(plan.tile), int(bws13),
                    1 if r2t is not None else 0, 1 if (r2t is not None or as_rm) else 0, fs_flag,
                    pdl_w13_wait, trig,
                    tile_w13_done_ptr if ov_flag else 0,
                    tile_ready_ptr if ov_flag else 0,
                    overlap_meta_ptr if ov_flag else 0,
                    ov_flag, hot_l2_flag, st,
                )
            return lib.mx7_w13_c32_v5_panel_fwd(
                act.view(torch.uint8).data_ptr(),
                act_scale.view(torch.uint8).data_ptr(),
                0 if r2t is None else r2t.data_ptr(),
                pp13.data_ptr(), po13.data_ptr(),
                sm13.data_ptr(), so13.data_ptr(), sf13.data_ptr(),
                experts.ones.data_ptr(), tiles.data_ptr(),
                tcs_ptr, gam_ptr, out_ptr,
                576, 4608, int(plan.max_ctas), int(plan.tile), int(bws13),
                1 if r2t is not None else 0, 1 if (r2t is not None or as_rm) else 0, fs_flag,
                pdl_w13_wait, trig, st,
            )
        if v5_mode == "compat":
            sl13, w3_13, sm13, so13, do13, sf13, bm13, bws13, _ = v5_13
            if use_mx_v16 and hasattr(lib, "mx7_w13_c48_v16_compat_fwd"):
                return lib.mx7_w13_c48_v16_compat_fwd(
                    act.view(torch.uint8).data_ptr(),
                    act_scale.view(torch.uint8).data_ptr(),
                    0 if r2t is None else r2t.data_ptr(),
                    sl13.data_ptr(), w3_13.data_ptr(),
                    sm13.data_ptr(), so13.data_ptr(), do13.data_ptr(),
                    sf13.data_ptr(), bm13.data_ptr(),
                    experts.ones.data_ptr(), tiles.data_ptr(),
                    tcs_ptr, gam_ptr, out_ptr, qdeq_ptr,
                    576, 4608, int(plan.max_ctas), int(plan.tile), int(bws13),
                    1 if r2t is not None else 0, 1 if (r2t is not None or as_rm) else 0, fs_flag, qdeq_flag,
                    eff_mx_bn_mode, MX7_V14_BN16_MARGIN,
                    pdl_w13_wait, trig,
                    tile_w13_done_ptr if ov_flag else 0,
                    tile_ready_ptr if ov_flag else 0,
                    overlap_meta_ptr if ov_flag else 0,
                    ov_flag, hot_l2_flag, MX7_V14_EVICT, MX7_V16_LPT, work_queue_ptr, st,
                )
            if use_mx_v14 and hasattr(lib, "mx7_w13_c48_v14_compat_fwd"):
                return lib.mx7_w13_c48_v14_compat_fwd(
                    act.view(torch.uint8).data_ptr(),
                    act_scale.view(torch.uint8).data_ptr(),
                    0 if r2t is None else r2t.data_ptr(),
                    sl13.data_ptr(), w3_13.data_ptr(),
                    sm13.data_ptr(), so13.data_ptr(), do13.data_ptr(),
                    sf13.data_ptr(), bm13.data_ptr(),
                    experts.ones.data_ptr(), tiles.data_ptr(),
                    tcs_ptr, gam_ptr, out_ptr, qdeq_ptr,
                    576, 4608, int(plan.max_ctas), int(plan.tile), int(bws13),
                    1 if r2t is not None else 0, 1 if (r2t is not None or as_rm) else 0, fs_flag, qdeq_flag,
                    eff_mx_bn_mode, MX7_V14_BN16_MARGIN,
                    pdl_w13_wait, trig,
                    tile_w13_done_ptr if ov_flag else 0,
                    tile_ready_ptr if ov_flag else 0,
                    overlap_meta_ptr if ov_flag else 0,
                    ov_flag, hot_l2_flag, MX7_V14_EVICT, st,
                )
            if use_mx_v8 and hasattr(lib, "mx7_w13_c48_v8_compat_fwd"):
                return lib.mx7_w13_c48_v8_compat_fwd(
                    act.view(torch.uint8).data_ptr(),
                    act_scale.view(torch.uint8).data_ptr(),
                    0 if r2t is None else r2t.data_ptr(),
                    sl13.data_ptr(), w3_13.data_ptr(),
                    sm13.data_ptr(), so13.data_ptr(), do13.data_ptr(),
                    sf13.data_ptr(), bm13.data_ptr(),
                    experts.ones.data_ptr(), tiles.data_ptr(),
                    tcs_ptr, gam_ptr, out_ptr,
                    576, 4608, int(plan.max_ctas), int(plan.tile), int(bws13),
                    1 if r2t is not None else 0, 1 if (r2t is not None or as_rm) else 0, fs_flag,
                    pdl_w13_wait, trig,
                    tile_w13_done_ptr if ov_flag else 0,
                    tile_ready_ptr if ov_flag else 0,
                    overlap_meta_ptr if ov_flag else 0,
                    ov_flag, hot_l2_flag, st,
                )
            return lib.mx7_w13_c32_v5_fwd(
                act.view(torch.uint8).data_ptr(),
                act_scale.view(torch.uint8).data_ptr(),
                0 if r2t is None else r2t.data_ptr(),
                sl13.data_ptr(), w3_13.data_ptr(),
                sm13.data_ptr(), so13.data_ptr(), do13.data_ptr(),
                sf13.data_ptr(), bm13.data_ptr(),
                experts.ones.data_ptr(), tiles.data_ptr(),
                tcs_ptr, gam_ptr, out_ptr,
                576, 4608, int(plan.max_ctas), int(plan.tile), int(bws13),
                1 if r2t is not None else 0, 1 if (r2t is not None or as_rm) else 0, fs_flag,
                pdl_w13_wait, trig, st,
            )
        if use_mx_v16:
            return lib.mx7_w13_c48_v16_fwd(
                act.view(torch.uint8).data_ptr(),
                act_scale.view(torch.uint8).data_ptr(),
                0 if r2t is None else r2t.data_ptr(),
                slabs13.data_ptr(), ws3_13.data_ptr(), flips13.data_ptr(), f_off13.data_ptr(),
                experts.ones.data_ptr(), tiles.data_ptr(),
                tcs_ptr, gam_ptr, out_ptr, qdeq_ptr,
                576, 4608, int(plan.max_ctas), int(plan.tile), int(base_ws13),
                1 if r2t is not None else 0, 1 if (r2t is not None or as_rm) else 0, fs_flag, qdeq_flag,
                eff_mx_bn_mode, MX7_V14_BN16_MARGIN,
                pdl_w13_wait, trig,
                tile_w13_done_ptr if ov_flag else 0,
                tile_ready_ptr if ov_flag else 0,
                overlap_meta_ptr if ov_flag else 0,
                ov_flag, hot_l2_flag, MX7_V14_EVICT, MX7_V16_LPT, work_queue_ptr, st,
            )
        if use_mx_v14:
            return lib.mx7_w13_c48_v14_fwd(
                act.view(torch.uint8).data_ptr(),
                act_scale.view(torch.uint8).data_ptr(),
                0 if r2t is None else r2t.data_ptr(),
                slabs13.data_ptr(), ws3_13.data_ptr(), flips13.data_ptr(), f_off13.data_ptr(),
                experts.ones.data_ptr(), tiles.data_ptr(),
                tcs_ptr, gam_ptr, out_ptr, qdeq_ptr,
                576, 4608, int(plan.max_ctas), int(plan.tile), int(base_ws13),
                1 if r2t is not None else 0, 1 if (r2t is not None or as_rm) else 0, fs_flag, qdeq_flag,
                eff_mx_bn_mode, MX7_V14_BN16_MARGIN,
                pdl_w13_wait, trig,
                tile_w13_done_ptr if ov_flag else 0,
                tile_ready_ptr if ov_flag else 0,
                overlap_meta_ptr if ov_flag else 0,
                ov_flag, hot_l2_flag, MX7_V14_EVICT, st,
            )
        if use_mx_v8:
            return lib.mx7_w13_c48_v8_fwd(
                act.view(torch.uint8).data_ptr(),
                act_scale.view(torch.uint8).data_ptr(),
                0 if r2t is None else r2t.data_ptr(),
                slabs13.data_ptr(), ws3_13.data_ptr(), flips13.data_ptr(), f_off13.data_ptr(),
                experts.ones.data_ptr(), tiles.data_ptr(),
                tcs_ptr, gam_ptr, out_ptr,
                576, 4608, int(plan.max_ctas), int(plan.tile), int(base_ws13),
                1 if r2t is not None else 0, 1 if (r2t is not None or as_rm) else 0, fs_flag,
                pdl_w13_wait, trig,
                tile_w13_done_ptr if ov_flag else 0,
                tile_ready_ptr if ov_flag else 0,
                overlap_meta_ptr if ov_flag else 0,
                ov_flag, hot_l2_flag, st,
            )
        return lib.mx7_w13_c32_fwd(
            act.view(torch.uint8).data_ptr(),
            act_scale.view(torch.uint8).data_ptr(),
            0 if r2t is None else r2t.data_ptr(),
            slabs13.data_ptr(), ws3_13.data_ptr(), flips13.data_ptr(), f_off13.data_ptr(),
            experts.ones.data_ptr(), tiles.data_ptr(),
            tcs_ptr, gam_ptr, out_ptr,
            576, 4608, int(plan.max_ctas), int(plan.tile), int(base_ws13),
            1 if r2t is not None else 0, 1 if (r2t is not None or as_rm) else 0, fs_flag,
            pdl_w13_wait, trig, st,
        )

    def _call_w2(
        tcs_ptr: int, out_ptr: int, s2c_ptr: int, nl_ptr: int,
        pd_ptr: int, bg_ptr: int, sb_ptr: int, fc_flag: int,
    ) -> int:
        if v5_mode == "panel":
            pp2, po2, sm2, so2, sf2, bws2, _ = v5_2
            if use_mx_v16 and hasattr(lib, "mx7_w2_c48_v16_panel_fwd"):
                return lib.mx7_w2_c48_v16_panel_fwd(
                    w2_a_ptr, w2_as_ptr,
                    pp2.data_ptr(), po2.data_ptr(),
                    sm2.data_ptr(), so2.data_ptr(), sf2.data_ptr(),
                    experts.ones.data_ptr(), tiles.data_ptr(), tcs_ptr,
                    out_ptr, s2c_ptr, nl_ptr, pd_ptr, bg_ptr, sb_ptr,
                    4608, 288, int(plan.max_ctas), int(plan.tile), tokens, int(bws2),
                    w2_as_rm, w2_as_stride, down_row_stride,
                    mx_fq_w2, fc_flag,
                    pdl_w2_wait, trig,
                    tile_ready_ptr if mx_overlap_flag else 0,
                    overlap_meta_ptr if mx_overlap_flag else 0,
                    mx_overlap_flag, hot_l2_flag, MX7_V14_EVICT, MX7_V16_LPT, work_queue_w2_ptr, st,
                )
            if use_mx_v14 and hasattr(lib, "mx7_w2_c48_v14_panel_fwd"):
                return lib.mx7_w2_c48_v14_panel_fwd(
                    w2_a_ptr, w2_as_ptr,
                    pp2.data_ptr(), po2.data_ptr(),
                    sm2.data_ptr(), so2.data_ptr(), sf2.data_ptr(),
                    experts.ones.data_ptr(), tiles.data_ptr(), tcs_ptr,
                    out_ptr, s2c_ptr, nl_ptr, pd_ptr, bg_ptr, sb_ptr,
                    4608, 288, int(plan.max_ctas), int(plan.tile), tokens, int(bws2),
                    w2_as_rm, w2_as_stride, down_row_stride,
                    mx_fq_w2, fc_flag,
                    pdl_w2_wait, trig,
                    tile_ready_ptr if mx_overlap_flag else 0,
                    overlap_meta_ptr if mx_overlap_flag else 0,
                    mx_overlap_flag, hot_l2_flag, MX7_V14_EVICT, st,
                )
            if use_mx_v8 and hasattr(lib, "mx7_w2_c48_v8_panel_fwd"):
                return lib.mx7_w2_c48_v8_panel_fwd(
                    w2_a_ptr, w2_as_ptr,
                    pp2.data_ptr(), po2.data_ptr(),
                    sm2.data_ptr(), so2.data_ptr(), sf2.data_ptr(),
                    experts.ones.data_ptr(), tiles.data_ptr(), tcs_ptr,
                    out_ptr, s2c_ptr, nl_ptr, pd_ptr, bg_ptr, sb_ptr,
                    4608, 288, int(plan.max_ctas), int(plan.tile), tokens, int(bws2),
                    w2_as_rm, w2_as_stride, down_row_stride,
                    1 if use_fuse_situ else 0, fc_flag,
                    pdl_w2_wait, trig,
                    tile_ready_ptr if mx_overlap_flag else 0,
                    overlap_meta_ptr if mx_overlap_flag else 0,
                    mx_overlap_flag, hot_l2_flag, st,
                )
            return lib.mx7_w2_c32_v5_panel_fwd(
                w2_a_ptr, w2_as_ptr,
                pp2.data_ptr(), po2.data_ptr(),
                sm2.data_ptr(), so2.data_ptr(), sf2.data_ptr(),
                experts.ones.data_ptr(), tiles.data_ptr(), tcs_ptr,
                out_ptr, s2c_ptr, nl_ptr, pd_ptr, bg_ptr, sb_ptr,
                4608, 288, int(plan.max_ctas), int(plan.tile), tokens, int(bws2),
                w2_as_rm, w2_as_stride, down_row_stride,
                1 if use_fuse_situ else 0, fc_flag,
                pdl_w2_wait, trig, st,
            )
        if v5_mode == "compat":
            sl2, w3_2, sm2, so2, do2, sf2, bm2, bws2, _ = v5_2
            if use_mx_v16 and hasattr(lib, "mx7_w2_c48_v16_compat_fwd"):
                return lib.mx7_w2_c48_v16_compat_fwd(
                    w2_a_ptr, w2_as_ptr,
                    sl2.data_ptr(), w3_2.data_ptr(),
                    sm2.data_ptr(), so2.data_ptr(), do2.data_ptr(),
                    sf2.data_ptr(), bm2.data_ptr(),
                    experts.ones.data_ptr(), tiles.data_ptr(), tcs_ptr,
                    out_ptr, s2c_ptr, nl_ptr, pd_ptr, bg_ptr, sb_ptr,
                    4608, 288, int(plan.max_ctas), int(plan.tile), tokens, int(bws2),
                    w2_as_rm, w2_as_stride, down_row_stride,
                    mx_fq_w2, fc_flag,
                    pdl_w2_wait, trig,
                    tile_ready_ptr if mx_overlap_flag else 0,
                    overlap_meta_ptr if mx_overlap_flag else 0,
                    mx_overlap_flag, hot_l2_flag, MX7_V14_EVICT, MX7_V16_LPT, work_queue_w2_ptr, st,
                )
            if use_mx_v14 and hasattr(lib, "mx7_w2_c48_v14_compat_fwd"):
                return lib.mx7_w2_c48_v14_compat_fwd(
                    w2_a_ptr, w2_as_ptr,
                    sl2.data_ptr(), w3_2.data_ptr(),
                    sm2.data_ptr(), so2.data_ptr(), do2.data_ptr(),
                    sf2.data_ptr(), bm2.data_ptr(),
                    experts.ones.data_ptr(), tiles.data_ptr(), tcs_ptr,
                    out_ptr, s2c_ptr, nl_ptr, pd_ptr, bg_ptr, sb_ptr,
                    4608, 288, int(plan.max_ctas), int(plan.tile), tokens, int(bws2),
                    w2_as_rm, w2_as_stride, down_row_stride,
                    mx_fq_w2, fc_flag,
                    pdl_w2_wait, trig,
                    tile_ready_ptr if mx_overlap_flag else 0,
                    overlap_meta_ptr if mx_overlap_flag else 0,
                    mx_overlap_flag, hot_l2_flag, MX7_V14_EVICT, st,
                )
            if use_mx_v8 and hasattr(lib, "mx7_w2_c48_v8_compat_fwd"):
                return lib.mx7_w2_c48_v8_compat_fwd(
                    w2_a_ptr, w2_as_ptr,
                    sl2.data_ptr(), w3_2.data_ptr(),
                    sm2.data_ptr(), so2.data_ptr(), do2.data_ptr(),
                    sf2.data_ptr(), bm2.data_ptr(),
                    experts.ones.data_ptr(), tiles.data_ptr(), tcs_ptr,
                    out_ptr, s2c_ptr, nl_ptr, pd_ptr, bg_ptr, sb_ptr,
                    4608, 288, int(plan.max_ctas), int(plan.tile), tokens, int(bws2),
                    w2_as_rm, w2_as_stride, down_row_stride,
                    1 if use_fuse_situ else 0, fc_flag,
                    pdl_w2_wait, trig,
                    tile_ready_ptr if mx_overlap_flag else 0,
                    overlap_meta_ptr if mx_overlap_flag else 0,
                    mx_overlap_flag, hot_l2_flag, st,
                )
            return lib.mx7_w2_c32_v5_fwd(
                w2_a_ptr, w2_as_ptr,
                sl2.data_ptr(), w3_2.data_ptr(),
                sm2.data_ptr(), so2.data_ptr(), do2.data_ptr(),
                sf2.data_ptr(), bm2.data_ptr(),
                experts.ones.data_ptr(), tiles.data_ptr(), tcs_ptr,
                out_ptr, s2c_ptr, nl_ptr, pd_ptr, bg_ptr, sb_ptr,
                4608, 288, int(plan.max_ctas), int(plan.tile), tokens, int(bws2),
                w2_as_rm, w2_as_stride, down_row_stride,
                1 if use_fuse_situ else 0, fc_flag,
                pdl_w2_wait, trig, st,
            )
        if use_mx_v16:
            return lib.mx7_w2_c48_v16_fwd(
                w2_a_ptr, w2_as_ptr,
                slabs2.data_ptr(), ws3_2.data_ptr(), flips2.data_ptr(), f_off2.data_ptr(),
                experts.ones.data_ptr(), tiles.data_ptr(), tcs_ptr,
                out_ptr, s2c_ptr, nl_ptr, pd_ptr, bg_ptr, sb_ptr,
                4608, 288, int(plan.max_ctas), int(plan.tile), tokens, int(base_ws2),
                w2_as_rm, w2_as_stride, down_row_stride,
                mx_fq_w2, fc_flag,
                pdl_w2_wait, trig,
                tile_ready_ptr if mx_overlap_flag else 0,
                overlap_meta_ptr if mx_overlap_flag else 0,
                mx_overlap_flag, hot_l2_flag, MX7_V14_EVICT, MX7_V16_LPT, work_queue_w2_ptr, st,
            )
        if use_mx_v14:
            return lib.mx7_w2_c48_v14_fwd(
                w2_a_ptr, w2_as_ptr,
                slabs2.data_ptr(), ws3_2.data_ptr(), flips2.data_ptr(), f_off2.data_ptr(),
                experts.ones.data_ptr(), tiles.data_ptr(), tcs_ptr,
                out_ptr, s2c_ptr, nl_ptr, pd_ptr, bg_ptr, sb_ptr,
                4608, 288, int(plan.max_ctas), int(plan.tile), tokens, int(base_ws2),
                w2_as_rm, w2_as_stride, down_row_stride,
                mx_fq_w2, fc_flag,
                pdl_w2_wait, trig,
                tile_ready_ptr if mx_overlap_flag else 0,
                overlap_meta_ptr if mx_overlap_flag else 0,
                mx_overlap_flag, hot_l2_flag, MX7_V14_EVICT, st,
            )
        if use_mx_v8:
            return lib.mx7_w2_c48_v8_fwd(
                w2_a_ptr, w2_as_ptr,
                slabs2.data_ptr(), ws3_2.data_ptr(), flips2.data_ptr(), f_off2.data_ptr(),
                experts.ones.data_ptr(), tiles.data_ptr(), tcs_ptr,
                out_ptr, s2c_ptr, nl_ptr, pd_ptr, bg_ptr, sb_ptr,
                4608, 288, int(plan.max_ctas), int(plan.tile), tokens, int(base_ws2),
                w2_as_rm, w2_as_stride, down_row_stride,
                1 if use_fuse_situ else 0, fc_flag,
                pdl_w2_wait, trig,
                tile_ready_ptr if mx_overlap_flag else 0,
                overlap_meta_ptr if mx_overlap_flag else 0,
                mx_overlap_flag, hot_l2_flag, st,
            )
        return lib.mx7_w2_c32_fwd(
            w2_a_ptr, w2_as_ptr,
            slabs2.data_ptr(), ws3_2.data_ptr(), flips2.data_ptr(), f_off2.data_ptr(),
            experts.ones.data_ptr(), tiles.data_ptr(), tcs_ptr,
            out_ptr, s2c_ptr, nl_ptr, pd_ptr, bg_ptr, sb_ptr,
            4608, 288, int(plan.max_ctas), int(plan.tile), tokens, int(base_ws2),
            w2_as_rm, w2_as_stride, down_row_stride,
            1 if use_fuse_situ else 0, fc_flag,
            pdl_w2_wait, trig, st,
        )

    if use_fuse_situ:
        situ_out = ws["situ_out"][:down_row_stride]
        rc = _call_w13(situ_out.data_ptr(), tile_compact_start.data_ptr(), gammas.data_ptr(), 1)
        if rc != 0:
            raise RuntimeError(f"mx7_w13_c32[{v5_mode}] (fuse_situ=1) failed: rc={rc}")
        w2_a_ptr = situ_out.data_ptr()
        w2_as_ptr = 0
        w2_as_rm = 0
        w2_as_stride = 9
    else:
        gate_up = torch.empty((plan.rows, 576), dtype=torch.bfloat16, device=device)
        rc = _call_w13(gate_up.data_ptr(), 0, 0, 0)
        if rc != 0:
            raise RuntimeError(f"mx7_w13_c32[{v5_mode}] (fuse_situ=0) failed: rc={rc}")
        from .activation import swiglu_quantize
        m = plan.max_ctas
        act2_v, act2_s, _ = swiglu_quantize(
            gate_up, gammas, experts.gate_cap, experts.up_cap,
            (rows, tiles[2 * m + 2 :], tiles[:m]), plan.tile, None, None,
        )
        del gate_up
        w2_a_ptr = act2_v.view(torch.uint8).data_ptr()
        w2_as_ptr = act2_s.view(torch.uint8).data_ptr()
        w2_as_rm = 1 if bool(getattr(act2_s, "_inkling_rowmaj", False)) else 0
        w2_as_stride = int(act2_s.stride(0)) if (w2_as_rm and act2_s.ndim >= 2) else 9

    eff_mx_fc_mode = fc_mode
    if (
        C32_MX7_FC_MODE >= 0
        and has_compact
        and (C32_MX7_FC_MAX_M <= 0 or tokens <= C32_MX7_FC_MAX_M)
    ):
        eff_mx_fc_mode = C32_MX7_FC_MODE
    if eff_mx_fc_mode == 3:
        mx_fc = 3 if mx_ver >= 7 else 0
    elif eff_mx_fc_mode == 2:
        mx_fc = 2 if mx_ver >= 4 else 0
    else:
        mx_fc = eff_mx_fc_mode if eff_mx_fc_mode in (0, 1) else 0
    if mx_fc > 0:
        out = out_buf if out_buf is not None else torch.empty(
            (tokens, 4608), dtype=torch.bfloat16, device=device
        )
        down_ws = ws["down_compact"] if mx_fc in (2, 3) else ws["down_panel"]
        rc = _call_w2(
            tile_compact_start.data_ptr(),
            down_ws.data_ptr(),
            src2compact.data_ptr(),
            0 if nlive_buf is None else nlive_buf.data_ptr(),
            ws["panel_done"].data_ptr(),
            0 if mx_fc in (2, 3) else ws["buf_gen"].data_ptr(),
            out.data_ptr(),
            mx_fc,
        )
        if rc != 0:
            raise RuntimeError(f"mx7_w2_c32[{v5_mode}] (fuse_combine={mx_fc}) failed: rc={rc}")
        return out

    if (
        (C32_COMBINE_V6 or C48_MOE_V8 or C48_MOE_V9 or C48_MOE_V10 or C48_MOE_V12 or MX7_C48_V14)
        and has_compact
        and mx_ver >= 7
        and hasattr(lib, "mx7_combine_c48_v7_fwd")
    ):
        out = out_buf if out_buf is not None else torch.empty(
            (tokens, 4608), dtype=torch.bfloat16, device=device
        )
        down_compact_row = ws["down_compact"]
        rc = _call_w2(
            tile_compact_start.data_ptr(),
            down_compact_row.data_ptr(),
            0, 0, 0, 0, 0, 4,
        )
        if rc != 0:
            raise RuntimeError(f"mx7_w2_c32[{v5_mode}] (fuse_combine=4) failed: rc={rc}")
        rc = lib.mx7_combine_c48_v7_fwd(
            down_compact_row.data_ptr(),
            src2compact.data_ptr(),
            0 if nlive_buf is None else nlive_buf.data_ptr(),
            out.data_ptr(),
            tokens, 4608,
            1 if C32_PDL else 0, trig, st,
        )
        if rc != 0:
            raise RuntimeError(f"mx7_combine_c48_v7_fwd failed: rc={rc}")
        return out

    down = torch.empty((plan.rows, 4608), dtype=torch.bfloat16, device=device)
    rc = _call_w2(
        tile_compact_start.data_ptr() if use_fuse_situ else 0,
        down.data_ptr(),
        0, 0, 0, 0, 0, 0,
    )
    if rc != 0:
        raise RuntimeError(f"mx7_w2_c32[{v5_mode}] (fuse_combine=0) failed: rc={rc}")
    from .moe import combine
    return combine(down, src2dst, experts.topk, out_dtype, out=out_buf)
