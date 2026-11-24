# SPDX-License-Identifier: Apache-2.0
"""SM121a (GB10) C=1 v4 0-ULP Bit-Exact Fused Target MoE Decode Runtime.

Drop-in orchestration module for the C=1 v4 3-library Target MoE decode stack (`M <= 16` ONLY,
leaving C=32 `M >= 17` and 1M prefill `/rdma/libmxfp8_7bit_exact_v5.so` 100% untouched!):
  1. `librouter_bf12_c1_v4.so`: 48-CTA zero-tail K1 + 1-CTA Zero-Scatter SMEM-Route K2
     (`TileMask` + `TileCompactStart` + `Src2Compact` + row-major `QValues[T]`/`QScales[T]`
     + pre-zeroed compact `RowAmaxW2`).
  2. `libnvfp4_ws5_c1_v4.so`: Zero-spill Adaptive `BN=16` (`U<=37`) / `BN=32` (`U>=38`) WS5 NVFP4
     W13 (`SA_STRIDE=144`, `fns.b32` Zero-Scatter, `WS_BASE` bit-7 `P_OFF` bypass, compact Fused
     SiTU -> `36 KiB` `SiTU_Out` + `RowAmaxW2`) + `BN=128` W2 (strictly-unfused OTF NVFP4
     quant-dequant + Zero-BufGen panel-major compact L2-resident `DownCompact[36, T*8, 128]`
     = `576 KiB` @ M=8 + `#104`-hardened `st.cg + __threadfence() + atom.acq_rel.gpu` combine
     directly into RDMA_AR `SendBuf` + PDL trigger).
  3. `libmxfp8_mx7_c1_v4.so`: Zero-spill Adaptive `BN=16` (`U<=37`) / `BN=32` (`U>=38`) MX7 W13
     (`SA_STRIDE=272`, zero-repack bijection onto existing MX7 slabs, `fns.b32` Zero-Scatter,
     compact Fused SiTU) + upgraded `BN=128` 4-warp MX7 W2 (zero-repack bijection, OTF MXFP8
     quant-dequant, exact same Zero-BufGen `DownCompact` combine -> `SendBuf`), reusing
     `weight._inkling_mx7` with 100% `INKLING_MX7_FREE_RAW=1` & 1M prefill compatibility!

Rollout Controls (all default OFF unless `INKLING_MOE_S8=1` or set individually):
  - `INKLING_MOE_S8=1`                  : master switch enabling all 4 knobs below
  - `INKLING_MOE_S8_ROUTER=1`           : 48-CTA zero-tail K1 + 1-CTA K2 router
  - `INKLING_MOE_S8_ZERO_SCATTER=1`     : TileMask + row-major per-token QValues/QScales
  - `INKLING_MOE_S8_FUSE_SITU=1`        : fuse SiTU into W13 + OTF quant into W2 (-48 launches/step)
  - `INKLING_MOE_S8_FUSE_COMBINE=1`     : fuse exact slot-0..7 combine into W2 -> SendBuf (-48 launches/step)
  - `INKLING_MOE_S8_BN_MODE=adaptive`   : `adaptive` (2, default), `bn16` (1), or `bn32` (0)
  - `INKLING_MOE_S8_U_THRESH=37`        : Adaptive BN=16 / BN=32 cutoff (`1 <= U <= u_thresh` -> BN=16)
  - `INKLING_MOE_S8_QUANT_FMA=1`        : Router K2 NVFP4 quant error FMA mode (1=fmaf, 0=unfused)
"""

from __future__ import annotations

import ctypes
import os
import warnings
import torch


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "", "off", "false", "no")


def _parse_bn_mode(raw: str | None) -> int:
    if not raw:
        return 2
    s = raw.strip().lower()
    if s in ("0", "bn32", "32"):
        return 0
    if s in ("1", "bn16", "16"):
        return 1
    if s in ("3", "wave", "wave_adapt"):
        return 3
    return 2


_S8_MASTER_RAW = os.environ.get("INKLING_MOE_S8")
S8_MASTER = _env_flag("INKLING_MOE_S8", False)
_S8_MASTER_FORCE_OFF = (_S8_MASTER_RAW is not None) and (not S8_MASTER)
S8_ROUTER = (not _S8_MASTER_FORCE_OFF) and _env_flag("INKLING_MOE_S8_ROUTER", S8_MASTER)
S8_ZERO_SCATTER = (not _S8_MASTER_FORCE_OFF) and _env_flag("INKLING_MOE_S8_ZERO_SCATTER", S8_MASTER)
S8_FUSE_SITU = (not _S8_MASTER_FORCE_OFF) and _env_flag("INKLING_MOE_S8_FUSE_SITU", S8_MASTER)
S8_FUSE_COMBINE = (not _S8_MASTER_FORCE_OFF) and _env_flag("INKLING_MOE_S8_FUSE_COMBINE", S8_MASTER)
S8_BN_MODE = _parse_bn_mode(os.environ.get("INKLING_MOE_S8_BN_MODE", "adaptive"))
S8_U_THRESH = int(os.environ.get("INKLING_MOE_S8_U_THRESH", "37") or "37")
_RAW_B2_BN_MODE = os.environ.get("INKLING_MOE_S8_B2_BN_MODE", "").strip()
S8_B2_BN_MODE: int | None = _parse_bn_mode(_RAW_B2_BN_MODE) if _RAW_B2_BN_MODE else None
_RAW_B2_U_THRESH = os.environ.get("INKLING_MOE_S8_B2_U_THRESH", "").strip()
S8_B2_U_THRESH: int | None = int(_RAW_B2_U_THRESH) if _RAW_B2_U_THRESH else None
# Wave-V12 Forensic Fix: decouple S8_QUANT_FMA from INKLING_MOE_V9_QUANT_FMA!
# S8 C=1..2 Router + W13->QDeqA + W2 matches live 1dcfb306 with fmaf (1), whereas
# C=2..32 NVFP4 W2 uses sub_rn (INKLING_MOE_V9_QUANT_FMA=0). Setting V9_QUANT_FMA=0
# must NOT silently break S8_QUANT_FMA!
S8_QUANT_FMA = 1 if _env_flag("INKLING_MOE_S8_QUANT_FMA", True) else 0


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
MX7_V5_THRESH: int = max(0, min(63, int(os.environ.get("INKLING_MX7_V5_THRESH", "32") or "32")))
NVFP4_V6: bool = _env_flag("INKLING_NVFP4_V6", False)
NVFP4_C1_V9: bool = _env_flag("INKLING_NVFP4_C1_V9", False)
NVFP4_C1_V9_QDEQ: bool = _env_flag("INKLING_NVFP4_C1_V9_QDEQ", NVFP4_C1_V9)
# Wave-V12 C=1..2 0-ULP NVFP4 MoE knob (default OFF -> 100% V10 invariant!)
NVFP4_C1_V12: bool = _env_flag("INKLING_NVFP4_C1_V12", False)
# Wave-V10 C=1..2 MoE + Single-Launch Fused Router knobs (ALL default OFF -> 100% V9 invariant!)
ROUTER_V10_FUSED: bool = _env_flag("INKLING_ROUTER_V10_FUSED", False)
ROUTER_BF11: bool = _env_flag("INKLING_ROUTER_BF11", False)
NVFP4_C1_V10: bool = _env_flag("INKLING_NVFP4_C1_V10", NVFP4_C1_V12)
NVFP4_C1_V10_FUSE_QDEQ: bool = _env_flag("INKLING_NVFP4_C1_V10_FUSE_QDEQ", NVFP4_C1_V10)

_DIR = os.path.dirname(os.path.abspath(__file__))
_ROUTER_PATHS = (
    os.environ.get("INKLING_ROUTER_BF12_S8_LIB", ""),
    *(
        (
            "/rdma/librouter_bf12_fused_v10.so",
            os.path.join(_DIR, "librouter_bf12_fused_v10.so"),
        )
        if ROUTER_V10_FUSED
        else ()
    ),
    "/rdma/librouter_bf12_c1_v4.so",
    os.path.join(_DIR, "librouter_bf12_c1_v4.so"),
    os.environ.get("INKLING_ROUTER_BF12_LIB", ""),
    "/rdma/librouter_bf12_s8.so",
    os.path.join(_DIR, "librouter_bf12_s8.so"),
)
_NVFP4_PATHS = (
    os.environ.get("INKLING_NVFP4_WS5_S8_LIB", "").strip(),
    *(
        (
            "/rdma/libnvfp4_ws5_c1_v12.so",
            os.path.join(_DIR, "libnvfp4_ws5_c1_v12.so"),
        )
        if NVFP4_C1_V12
        else ()
    ),
    *(
        (
            "/rdma/libnvfp4_ws5_c1_v10.so",
            os.path.join(_DIR, "libnvfp4_ws5_c1_v10.so"),
        )
        if (NVFP4_C1_V10 or NVFP4_C1_V10_FUSE_QDEQ)
        else ()
    ),
    *(
        (
            "/rdma/libnvfp4_ws5_c1_v9.so",
            os.path.join(_DIR, "libnvfp4_ws5_c1_v9.so"),
        )
        if (NVFP4_C1_V9 or NVFP4_C1_V10)
        else ()
    ),
    *(
        (
            "/rdma/libnvfp4_ws5_c1_v6.so",
            os.path.join(_DIR, "libnvfp4_ws5_c1_v6.so"),
        )
        if NVFP4_V6
        else ()
    ),
    "/rdma/libnvfp4_ws5_c1_v4.so",
    os.path.join(_DIR, "libnvfp4_ws5_c1_v4.so"),
    os.environ.get("INKLING_NVFP4_WS5_LIB", ""),
    "/rdma/libnvfp4_ws5_s8.so",
    os.path.join(_DIR, "libnvfp4_ws5_s8.so"),
)
_MX7_PATHS = (
    os.environ.get("INKLING_MX7_S8_LIB", ""),
    *(
        (
            "/rdma/libmxfp8_mx7_c1_v5.so",
            os.path.join(_DIR, "libmxfp8_mx7_c1_v5.so"),
        )
        if MX7_V5_MODE != "off"
        else ()
    ),
    "/rdma/libmxfp8_mx7_c1_v4.so",
    os.path.join(_DIR, "libmxfp8_mx7_c1_v4.so"),
    "/rdma/libmxfp8_mx7_s8.so",
    os.path.join(_DIR, "libmxfp8_mx7_s8.so"),
)

_LIB_ROUTER = None
_LIB_NVFP4 = None
_LIB_NVFP4_VER: int = 0
_LIB_MX7 = None
_LIB_MX7_VER: int = 0
_WARNED_W2_QUANT_FMA = False
_TRIED = {"router": False, "nvfp4": False, "mx7": False}

# Static per-device compact C=1 v4 workspace (shared across all 48 layers, CUDA-graph capture safe).
# Sized for T <= 16 decode (max compact rows = 16 * 8 = 128, max_ctas <= 256):
#   situ_out           [128, 288]      bf16  =  72 KiB (36 KiB active @ T=8)
#   row_amax_w2        [128]           int32 = 512 B
#   tile_mask          [256]           int32 =   1 KiB
#   tile_compact_start [256]           int32 =   1 KiB
#   src2compact        [128]           int32 = 512 B
#   down_compact       [36, 128, 128]  bf16  = 1.125 MiB (576 KiB = 16 KiB/panel active @ T=8!)
#   panel_done         [36]            int32 = 144 B (zeroed once; self-reset by last panel CTA!)
#   logits_ws          [16, 384]       fp32  =  24 KiB
# Total VRAM across all 48 layers = 1.22 MiB (15.7x smaller than legacy 19.25 MiB!).
_WS_MAX_ROWS = 2048
_WS_MAX_CTAS = 256
_WS_MAX_COMPACT = 128
_WS: dict[int, dict[str, torch.Tensor]] = {}


def _find_lib(paths: tuple[str, ...], symbols: tuple[str, ...]) -> tuple[ctypes.CDLL | None, str | None]:
    for sym in symbols:
        for p in paths:
            if p and os.path.isfile(p):
                try:
                    lib = ctypes.CDLL(p)
                    if hasattr(lib, sym):
                        return lib, sym
                except OSError:
                    continue
    return None, None


def get_router_lib(path: str | None = None) -> ctypes.CDLL | None:
    global _LIB_ROUTER
    if _LIB_ROUTER is not None and path is None:
        return _LIB_ROUTER
    if _TRIED["router"] and path is None:
        return None
    if path is None:
        _TRIED["router"] = True
    paths = (path,) if path else _ROUTER_PATHS
    _rtr_syms = (
        ("router_bf12_c1_fused_v10_fwd", "router_bf12_c1_v4_fwd", "router_bf12_s8_zero_scatter_fwd")
        if ROUTER_V10_FUSED
        else ("router_bf12_c1_v4_fwd", "router_bf12_s8_zero_scatter_fwd")
    )
    lib, sym = _find_lib(paths, _rtr_syms)
    if lib is None:
        return None
    if sym == "router_bf12_c1_fused_v10_fwd":
        if hasattr(lib, "router_bf12_v10_version"):
            lib.router_bf12_v10_version.argtypes = []
            lib.router_bf12_v10_version.restype = ctypes.c_int
        lib.router_bf12_c1_fused_v10_fwd.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_float, ctypes.c_uint32,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
        ]
        lib.router_bf12_c1_fused_v10_fwd.restype = ctypes.c_int
        if hasattr(lib, "router_bf11_c1_fused_v10_fwd"):
            lib.router_bf11_c1_fused_v10_fwd.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int, ctypes.c_float, ctypes.c_uint32,
                ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
            ]
            lib.router_bf11_c1_fused_v10_fwd.restype = ctypes.c_int
        lib._inkling_v4 = True
        lib._inkling_v10_fused = True
    elif sym == "router_bf12_c1_v4_fwd":
        lib.router_bf12_c1_v4_fwd.argtypes = [
            ctypes.c_void_p,  # X
            ctypes.c_void_p,  # SLABS
            ctypes.c_void_p,  # OUTLIERS
            ctypes.c_void_p,  # O_OFF
            ctypes.c_void_p,  # Bias
            ctypes.c_void_p,  # ExpertBounds
            ctypes.c_void_p,  # Padding
            ctypes.c_void_p,  # NLivePtr
            ctypes.c_void_p,  # LogitsWS
            ctypes.c_void_p,  # Src2Dst
            ctypes.c_void_p,  # Src2Compact
            ctypes.c_void_p,  # Gammas
            ctypes.c_void_p,  # Rows
            ctypes.c_void_p,  # Tiles
            ctypes.c_void_p,  # TileMask
            ctypes.c_void_p,  # TileCompactStart
            ctypes.c_void_p,  # RowAmaxW2
            ctypes.c_void_p,  # QValues
            ctypes.c_void_p,  # QScales
            ctypes.c_void_p,  # QRowScales
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
        lib.router_bf12_c1_v4_fwd.restype = ctypes.c_int
        lib._inkling_v4 = True
    else:
        lib.router_bf12_s8_zero_scatter_fwd.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_float, ctypes.c_uint32,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
        ]
        lib.router_bf12_s8_zero_scatter_fwd.restype = ctypes.c_int
        if hasattr(lib, "router_bf12_exact_fwd"):
            lib.router_bf12_exact_fwd.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int, ctypes.c_float, ctypes.c_uint32,
                ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
            ]
            lib.router_bf12_exact_fwd.restype = ctypes.c_int
        lib._inkling_v4 = False
    if path is None:
        _LIB_ROUTER = lib
    return lib


def get_nvfp4_lib(path: str | None = None) -> ctypes.CDLL | None:
    global _LIB_NVFP4
    if _LIB_NVFP4 is not None and path is None:
        return _LIB_NVFP4
    if _TRIED["nvfp4"] and path is None:
        return None
    if path is None:
        _TRIED["nvfp4"] = True
    paths = (path,) if path else _NVFP4_PATHS
    lib, sym = _find_lib(paths, ("nvfp4_w13_ws5_c1_v4_fwd", "nvfp4_w13_ws5_s8_fwd"))
    if lib is None:
        return None
    if sym == "nvfp4_w13_ws5_c1_v4_fwd":
        lib.nvfp4_w13_ws5_c1_v4_fwd.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
        ]
        lib.nvfp4_w13_ws5_c1_v4_fwd.restype = ctypes.c_int
        lib.nvfp4_w2_ws5_c1_v4_fwd.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
        ]
        lib.nvfp4_w2_ws5_c1_v4_fwd.restype = ctypes.c_int
        if hasattr(lib, "nvfp4_w2_ws5_c1_v6_fma_fwd"):
            lib.nvfp4_w2_ws5_c1_v6_fma_fwd.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
            ]
            lib.nvfp4_w2_ws5_c1_v6_fma_fwd.restype = ctypes.c_int
        if hasattr(lib, "nvfp4_w13_ws5_c1_v10_fwd"):
            lib.nvfp4_w13_ws5_c1_v10_fwd.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
            ]
            lib.nvfp4_w13_ws5_c1_v10_fwd.restype = ctypes.c_int
        if hasattr(lib, "nvfp4_w13_ws5_c1_v9_fwd"):
            lib.nvfp4_w13_ws5_c1_v9_fwd.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
            ]
            lib.nvfp4_w13_ws5_c1_v9_fwd.restype = ctypes.c_int
        if hasattr(lib, "nvfp4_situ_quant_dequant_c1_v9_fwd"):
            lib.nvfp4_situ_quant_dequant_c1_v9_fwd.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
            ]
            lib.nvfp4_situ_quant_dequant_c1_v9_fwd.restype = ctypes.c_int
        if hasattr(lib, "nvfp4_w2_ws5_c1_v9_fwd"):
            lib.nvfp4_w2_ws5_c1_v9_fwd.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
            ]
            lib.nvfp4_w2_ws5_c1_v9_fwd.restype = ctypes.c_int
        lib._inkling_v4 = True
        ver = 4
        if hasattr(lib, "nvfp4_ws5_c1_version"):
            lib.nvfp4_ws5_c1_version.argtypes = []
            lib.nvfp4_ws5_c1_version.restype = ctypes.c_int
            ver = int(lib.nvfp4_ws5_c1_version())
    else:
        lib.nvfp4_w13_ws5_s8_fwd.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
        ]
        lib.nvfp4_w13_ws5_s8_fwd.restype = ctypes.c_int
        lib.nvfp4_w2_ws5_s8_fwd.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
        ]
        lib.nvfp4_w2_ws5_s8_fwd.restype = ctypes.c_int
        lib._inkling_v4 = False
        ver = 3
    if path is None:
        global _LIB_NVFP4_VER
        _LIB_NVFP4 = lib
        _LIB_NVFP4_VER = ver
    return lib


def get_nvfp4_version() -> int:
    if _LIB_NVFP4 is None and not _TRIED["nvfp4"]:
        get_nvfp4_lib()
    return _LIB_NVFP4_VER


def get_mx6_lib() -> ctypes.CDLL | None:
    """Return shared 0-ULP MX6-Huff7 v10 CUDA library when INKLING_MX6_EXACT=1."""
    from . import trtllm_bmm
    if not getattr(trtllm_bmm, "_MX6_ENABLED", False):
        return None
    return trtllm_bmm._get_mx6_lib()


def get_mx7_lib(path: str | None = None) -> ctypes.CDLL | None:
    global _LIB_MX7
    if _LIB_MX7 is not None and path is None:
        return _LIB_MX7
    if _TRIED["mx7"] and path is None:
        return None
    if path is None:
        _TRIED["mx7"] = True
    paths = (path,) if path else _MX7_PATHS
    lib, sym = _find_lib(paths, ("mx7_w13_c1_v4_fwd", "mx7_w13_s8_fwd"))
    if lib is None:
        return None
    if sym == "mx7_w13_c1_v4_fwd":
        lib.mx7_w13_c1_v4_fwd.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint32,
            ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
        ]
        lib.mx7_w13_c1_v4_fwd.restype = ctypes.c_int
        lib.mx7_w2_c1_v4_fwd.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint32,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
        ]
        lib.mx7_w2_c1_v4_fwd.restype = ctypes.c_int
        lib._inkling_v4 = True
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
        if hasattr(lib, "mx7_w13_c1_v5_fwd"):
            lib.mx7_w13_c1_v5_fwd.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint32,
                ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
            ]
            lib.mx7_w13_c1_v5_fwd.restype = ctypes.c_int
        if hasattr(lib, "mx7_w2_c1_v5_fwd"):
            lib.mx7_w2_c1_v5_fwd.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint32,
                ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
            ]
            lib.mx7_w2_c1_v5_fwd.restype = ctypes.c_int
        if hasattr(lib, "mx7_w13_c1_v5_panel_fwd"):
            lib.mx7_w13_c1_v5_panel_fwd.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint32,
                ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
            ]
            lib.mx7_w13_c1_v5_panel_fwd.restype = ctypes.c_int
        if hasattr(lib, "mx7_w2_c1_v5_panel_fwd"):
            lib.mx7_w2_c1_v5_panel_fwd.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint32,
                ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
            ]
            lib.mx7_w2_c1_v5_panel_fwd.restype = ctypes.c_int
    else:
        lib.mx7_w13_s8_fwd.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint32,
            ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
        ]
        lib.mx7_w13_s8_fwd.restype = ctypes.c_int
        lib.mx7_w2_s8_fwd.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint32,
            ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
        ]
        lib.mx7_w2_s8_fwd.restype = ctypes.c_int
        lib._inkling_v4 = False
    ver = 4 if getattr(lib, "_inkling_v4", False) else 3
    if hasattr(lib, "mx7_c1_version"):
        lib.mx7_c1_version.argtypes = []
        lib.mx7_c1_version.restype = ctypes.c_int
        ver = int(lib.mx7_c1_version())
    if path is None:
        global _LIB_MX7_VER
        _LIB_MX7 = lib
        _LIB_MX7_VER = ver
    return lib


def get_mx7_version() -> int:
    if _LIB_MX7 is None and not _TRIED["mx7"]:
        get_mx7_lib()
    return _LIB_MX7_VER


def get_workspace(device: torch.device) -> dict[str, torch.Tensor]:
    """Return persistent CUDA-graph-safe compact C=1 v4 scratch buffers (1.22 MiB) for `device`."""
    idx = int(device.index or 0)
    ws = _WS.get(idx)
    if ws is None:
        ws = {
            "situ_out": torch.empty((_WS_MAX_COMPACT, 288), dtype=torch.bfloat16, device=device),
            "row_amax_w2": torch.zeros((_WS_MAX_COMPACT,), dtype=torch.int32, device=device),
            "tile_mask": torch.empty((_WS_MAX_CTAS,), dtype=torch.int32, device=device),
            "tile_compact_start": torch.empty((_WS_MAX_CTAS,), dtype=torch.int32, device=device),
            "src2compact": torch.empty((_WS_MAX_COMPACT,), dtype=torch.int32, device=device),
            "down_compact": torch.empty((36, _WS_MAX_COMPACT, 128), dtype=torch.bfloat16, device=device),
            "panel_done": torch.zeros((36,), dtype=torch.int32, device=device),
            "logits_ws": torch.empty((16, 384), dtype=torch.float32, device=device),
            "sel_bits_ws": torch.empty((16, 384), dtype=torch.int32, device=device),
            "row_amax_in": torch.empty((16,), dtype=torch.float32, device=device),
            "router_arrive": torch.zeros((1,), dtype=torch.int32, device=device),
            "tile_qdeq_done": torch.zeros((_WS_MAX_CTAS,), dtype=torch.int32, device=device),
        }
        _WS[idx] = ws
    return ws


@torch.no_grad()
def get_ws_base_s8(meta: dict) -> torch.Tensor:
    """Return `ws_base` with bit 7 set (`| 0x80`) on experts with ZERO scale patches.

    Leaves `meta['ws_base']` untouched so Triton prefill kernels stay 100% unmodified.
    Safe to call during CUDA graph capture once prewarmed during eager startup!
    """
    tagged = meta.get("ws_base_s8")
    if tagged is not None:
        return tagged
    ws_base: torch.Tensor = meta["ws_base"]
    if torch.cuda.is_current_stream_capturing():
        # Legacy untagged ws_base has bit 7 == 0 -> safe fallback (reads P_OFF normally).
        return ws_base
    p_off: torch.Tensor = meta["p_off"]
    E = int(ws_base.shape[0])
    proj = int(meta["proj"])
    stride = (18 * 4) if proj == 0 else 36  # warps/expert for W13, panels/expert for W2
    idx = torch.arange(E, device=ws_base.device, dtype=torch.long) * stride
    no_patch = p_off[idx + stride] == p_off[idx]
    # All valid NVFP4 e4m3 scales are non-negative (0..126 < 128), so bit 7 is guaranteed free!
    assert int(ws_base.max().item()) < 128, "NVFP4 ws_base unexpectedly >= 128"
    tagged = torch.where(no_patch, ws_base | 0x80, ws_base).to(torch.uint8).contiguous()
    meta["ws_base_s8"] = tagged
    return tagged


@torch.no_grad()
def get_mx7_packed_s8(weight: torch.Tensor, weight_scale: torch.Tensor, projection: int):
    """Return `(slabs, ws3, flips, f_off, base_ws)` safely reusing `weight._inkling_mx7`
    even when `INKLING_MX7_FREE_RAW=1` has already freed raw weight storage!"""
    cached = getattr(weight, "_inkling_mx7", None)
    if isinstance(cached, tuple):
        return cached
    if cached is False:
        return None
    try:
        from . import trtllm_bmm as _tbmm
        packed = _tbmm._mx7_get_packed(weight, weight_scale, projection, 0, 0)
        if isinstance(packed, tuple):
            return packed
    except Exception:
        pass
    return None


@torch.no_grad()
def get_mx7_v5_compat_s8(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    projection: int,
    flip_thresh: int | None = None,
):
    """Shared GPU-transcoded v5 Compat-Hybrid cache (reuses `weight._inkling_mx7_v5_compat`)."""
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
    packed = get_mx7_packed_s8(weight, weight_scale, projection)
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
def get_mx7_v5_panel_s8(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    projection: int,
    flip_thresh: int | None = None,
):
    """Shared GPU-transcoded v5 Panel-V5 cache (reuses `weight._inkling_mx7_v5_panel`)."""
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
    packed = get_mx7_packed_s8(weight, weight_scale, projection)
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
    fn = get_mx7_v5_panel_s8 if MX7_V5_MODE == "panel" else get_mx7_v5_compat_s8
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


def s8_decode_eligible(
    tokens: int,
    hidden: int,
    topk: int,
    plan,
    gate_cap: float = 4.0,
    up_cap: float = 25.0,
) -> bool:
    return (
        0 < tokens <= 16
        and hidden == 4608
        and topk == 8
        and plan.tile in (8, 16)
        and plan.tile >= tokens
        and plan.rows <= _WS_MAX_ROWS
        and plan.max_ctas <= _WS_MAX_CTAS
        and gate_cap == 4.0
        and up_cap == 25.0
    )


def route_bf12_s8(
    x: torch.Tensor,
    slabs: torch.Tensor,
    outliers: torch.Tensor,
    o_off: torch.Tensor,
    base_exp: int,
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
    pdl: int = 2,
    zero_scatter: bool | None = None,
    prezero_amax_w2: bool | None = None,
    gate_weight: torch.Tensor | None = None,
):
    """Execute C=1 v4 48-CTA zero-tail K1 + 1-CTA Zero-Scatter SMEM-Route K2.

    When `zero_scatter=True` (v4 default):
      - Returns `QValues` `[T, 2304|4608]` & row-major `QScales` `[T, 288|144]`.
      - Attaches `_inkling_tile_mask` (`[max_ctas]`), `_inkling_tile_compact_start` (`[max_ctas]`),
        `_inkling_src2compact` (`[T*8]`), and optional `_inkling_row_amax_w2` (`[T*8]`, pre-zeroed!)
        onto `tiles` so downstream W13/W2 consume them directly with zero extra launches.
    """
    lib = get_router_lib()
    if lib is None:
        return None
    tokens, hidden = int(x.shape[0]), int(x.shape[1])
    if not s8_decode_eligible(tokens, hidden, topk, plan):
        return None

    use_zs = S8_ZERO_SCATTER if zero_scatter is None else bool(zero_scatter)
    nvfp4 = weight_amax is not None
    if getattr(lib, "_inkling_v4", False):
        if not use_zs:
            # Unfused scatter ablation: let router_bf12.py fall back cleanly to librouter_bf12_exact.so
            return None
        # Only emit Zero-Scatter [T] QValues when the matching v4 expert library is loaded.
        expert_lib = get_nvfp4_lib() if nvfp4 else (get_mx6_lib() or get_mx7_lib())
        if expert_lib is None or not getattr(expert_lib, "_inkling_v4", False):
            return None

    use_amax = (S8_FUSE_SITU and weight_amax is not None) if prezero_amax_w2 is None else bool(prezero_amax_w2)

    device = x.device
    ws = get_workspace(device)
    n = tokens * topk
    rows = plan.rows

    src2dst = torch.empty(n, dtype=torch.int32, device=device)
    gammas = torch.empty(rows, dtype=torch.float32, device=device)
    routed = torch.empty(n, dtype=torch.int32, device=device)
    tiles = torch.empty(2 * plan.max_ctas + 3, dtype=torch.int32, device=device)

    qrows = tokens if use_zs else rows
    if nvfp4:
        quantized = (
            torch.empty((qrows, hidden // 2), dtype=torch.uint8, device=device),
            torch.empty((qrows, hidden // 16), dtype=torch.float8_e4m3fn, device=device),
            torch.empty(rows, dtype=torch.float32, device=device),
        )
    else:
        quantized = (
            torch.empty((qrows, hidden), dtype=torch.float8_e4m3fn, device=device),
            torch.empty((qrows, hidden // 32), dtype=torch.uint8, device=device),
            None,
        )

    st = int(torch.cuda.current_stream(device).cuda_stream)
    tile_mask = ws["tile_mask"][: plan.max_ctas]

    if getattr(lib, "_inkling_v10_fused", False):
        tile_compact_start = ws["tile_compact_start"][: plan.max_ctas]
        src2compact = ws["src2compact"][:n]
        row_amax_w2 = ws["row_amax_w2"][:n] if use_amax else None
        skip_amax_prezero = (
            nvfp4
            and (NVFP4_C1_V10_FUSE_QDEQ or NVFP4_C1_V9_QDEQ)
            and get_nvfp4_version() >= 9
        )
        amax_ptr = 0 if (row_amax_w2 is None or skip_amax_prezero) else row_amax_w2.data_ptr()
        bf11_entry = None
        if ROUTER_BF11 and gate_weight is not None and hasattr(lib, "router_bf11_c1_fused_v10_fwd"):
            from . import router_bf12 as _rb12
            bf11_entry = _rb12.get_packed_bf11(gate_weight)
            if bf11_entry is None and not torch.cuda.is_current_stream_capturing():
                bf11_entry = _rb12.prepack_bf11(gate_weight)
        if bf11_entry is not None:
            s11, smeta11, sflips11, bmasks11, o11, oo11, be11 = bf11_entry
            rc = lib.router_bf11_c1_fused_v10_fwd(
                x.data_ptr(), s11.data_ptr(), smeta11.data_ptr(), sflips11.data_ptr(), bmasks11.data_ptr(),
                o11.data_ptr(), oo11.data_ptr(),
                bias.data_ptr(), expert_bounds.data_ptr(),
                0 if is_padding is None else is_padding.data_ptr(),
                0 if nlive_buf is None else nlive_buf.data_ptr(),
                ws["logits_ws"].data_ptr(), ws["sel_bits_ws"].data_ptr(), ws["router_arrive"].data_ptr(),
                src2dst.data_ptr(), src2compact.data_ptr(),
                gammas.data_ptr(), routed.data_ptr(), tiles.data_ptr(),
                tile_mask.data_ptr(), tile_compact_start.data_ptr(), amax_ptr,
                quantized[0].data_ptr(), quantized[1].data_ptr(),
                0 if quantized[2] is None else quantized[2].data_ptr(),
                ws["row_amax_in"].data_ptr(),
                0 if weight_amax is None else weight_amax.data_ptr(),
                0 if weight_denominator is None else weight_denominator.data_ptr(),
                tokens, float(route_scale), int(be11), int(num_local),
                int(plan.tile), int(plan.max_ctas), 1 if nvfp4 else 0,
                int(S8_QUANT_FMA), int(pdl), st,
            )
            if rc != 0:
                raise RuntimeError(f"router_bf11_c1_fused_v10_fwd failed with cudaError={rc}")
        else:
            rc = lib.router_bf12_c1_fused_v10_fwd(
                x.data_ptr(), slabs.data_ptr(), outliers.data_ptr(), o_off.data_ptr(),
                bias.data_ptr(), expert_bounds.data_ptr(),
                0 if is_padding is None else is_padding.data_ptr(),
                0 if nlive_buf is None else nlive_buf.data_ptr(),
                ws["logits_ws"].data_ptr(), ws["sel_bits_ws"].data_ptr(), ws["router_arrive"].data_ptr(),
                src2dst.data_ptr(), src2compact.data_ptr(),
                gammas.data_ptr(), routed.data_ptr(), tiles.data_ptr(),
                tile_mask.data_ptr(), tile_compact_start.data_ptr(), amax_ptr,
                quantized[0].data_ptr(), quantized[1].data_ptr(),
                0 if quantized[2] is None else quantized[2].data_ptr(),
                ws["row_amax_in"].data_ptr(),
                0 if weight_amax is None else weight_amax.data_ptr(),
                0 if weight_denominator is None else weight_denominator.data_ptr(),
                tokens, float(route_scale), int(base_exp), int(num_local),
                int(plan.tile), int(plan.max_ctas), 1 if nvfp4 else 0,
                int(S8_QUANT_FMA), int(pdl), st,
            )
            if rc != 0:
                raise RuntimeError(f"router_bf12_c1_fused_v10_fwd failed with cudaError={rc}")
        tiles._inkling_tile_mask = tile_mask
        tiles._inkling_tile_compact_start = tile_compact_start
        tiles._inkling_src2compact = src2compact
        if row_amax_w2 is not None:
            tiles._inkling_row_amax_w2 = row_amax_w2
        return (src2dst, gammas, routed, tiles), quantized

    if getattr(lib, "_inkling_v4", False):
        tile_compact_start = ws["tile_compact_start"][: plan.max_ctas]
        src2compact = ws["src2compact"][:n]
        row_amax_w2 = ws["row_amax_w2"][:n] if use_amax else None
        rc = lib.router_bf12_c1_v4_fwd(
            x.data_ptr(), slabs.data_ptr(), outliers.data_ptr(), o_off.data_ptr(),
            bias.data_ptr(), expert_bounds.data_ptr(),
            0 if is_padding is None else is_padding.data_ptr(),
            0 if nlive_buf is None else nlive_buf.data_ptr(),
            ws["logits_ws"].data_ptr(),
            src2dst.data_ptr(), src2compact.data_ptr(),
            gammas.data_ptr(), routed.data_ptr(), tiles.data_ptr(),
            tile_mask.data_ptr(), tile_compact_start.data_ptr(),
            0 if row_amax_w2 is None else row_amax_w2.data_ptr(),
            quantized[0].data_ptr(), quantized[1].data_ptr(),
            0 if quantized[2] is None else quantized[2].data_ptr(),
            0 if weight_amax is None else weight_amax.data_ptr(),
            0 if weight_denominator is None else weight_denominator.data_ptr(),
            tokens, float(route_scale), int(base_exp), int(num_local),
            int(plan.tile), int(plan.max_ctas), 1 if nvfp4 else 0,
            int(S8_QUANT_FMA), int(pdl), st,
        )
        if rc != 0:
            raise RuntimeError(f"router_bf12_c1_v4_fwd failed with cudaError={rc}")
        tiles._inkling_tile_mask = tile_mask
        tiles._inkling_tile_compact_start = tile_compact_start
        tiles._inkling_src2compact = src2compact
        if row_amax_w2 is not None:
            tiles._inkling_row_amax_w2 = row_amax_w2
        return (src2dst, gammas, routed, tiles), quantized

    # Legacy S8 fallback path
    row_amax_w2 = torch.zeros((rows,), dtype=torch.int32, device=device) if use_amax else None
    logits_legacy = torch.empty((5, 16, 384), dtype=torch.float32, device=device)
    if use_zs:
        rc = lib.router_bf12_s8_zero_scatter_fwd(
            x.data_ptr(), slabs.data_ptr(), outliers.data_ptr(), o_off.data_ptr(),
            bias.data_ptr(), expert_bounds.data_ptr(),
            0 if is_padding is None else is_padding.data_ptr(),
            0 if nlive_buf is None else nlive_buf.data_ptr(),
            logits_legacy.data_ptr(),
            src2dst.data_ptr(), gammas.data_ptr(), routed.data_ptr(), tiles.data_ptr(),
            tile_mask.data_ptr(),
            0 if row_amax_w2 is None else row_amax_w2.data_ptr(),
            quantized[0].data_ptr(), quantized[1].data_ptr(),
            0 if quantized[2] is None else quantized[2].data_ptr(),
            0 if weight_amax is None else weight_amax.data_ptr(),
            0 if weight_denominator is None else weight_denominator.data_ptr(),
            tokens, float(route_scale), int(base_exp), int(num_local),
            int(plan.tile), int(plan.max_ctas), 1 if nvfp4 else 0, int(pdl), st,
        )
        if rc != 0:
            raise RuntimeError(f"router_bf12_s8_zero_scatter_fwd failed with cudaError={rc}")
        tiles._inkling_tile_mask = tile_mask
        if row_amax_w2 is not None:
            tiles._inkling_row_amax_w2 = row_amax_w2
    else:
        rc = lib.router_bf12_exact_fwd(
            x.data_ptr(), slabs.data_ptr(), outliers.data_ptr(), o_off.data_ptr(),
            bias.data_ptr(), expert_bounds.data_ptr(),
            0 if is_padding is None else is_padding.data_ptr(),
            0 if nlive_buf is None else nlive_buf.data_ptr(),
            0, 0, logits_legacy.data_ptr(), 0,
            src2dst.data_ptr(), gammas.data_ptr(), routed.data_ptr(), tiles.data_ptr(),
            quantized[0].data_ptr(), quantized[1].data_ptr(),
            0 if quantized[2] is None else quantized[2].data_ptr(),
            0 if weight_amax is None else weight_amax.data_ptr(),
            0 if weight_denominator is None else weight_denominator.data_ptr(),
            tokens, float(route_scale), int(base_exp), int(num_local),
            int(plan.tile), int(plan.max_ctas), 1 if nvfp4 else 0, int(pdl), st,
        )
        if rc != 0:
            raise RuntimeError(f"router_bf12_exact_fwd (s8) failed with cudaError={rc}")
        if row_amax_w2 is not None:
            tiles._inkling_row_amax_w2 = row_amax_w2

    return (src2dst, gammas, routed, tiles), quantized


def forward_experts_s8(
    experts,
    x: torch.Tensor,
    routing: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    quantized: tuple[torch.Tensor, torch.Tensor, torch.Tensor | None],
    plan,
    *,
    out_buf: torch.Tensor | None,
    nlive_buf: torch.Tensor | None = None,
    fuse_situ: bool | None = None,
    fuse_combine: bool | None = None,
    bn_mode: int | None = None,
    u_thresh: int | None = None,
    pdl: bool = True,
) -> torch.Tensor | None:
    """Execute C=1 v4 fused decode MoE experts (W13 [+SiTU] -> [swiglu_quantize] -> W2 [+Combine]).

    Returns combined `[T, 4608]` bf16 tensor (written directly into `out_buf` if provided),
    or `None` to cleanly fall back to stock S7 path if any library/condition isn't met.
    """
    tokens, hidden = int(x.shape[0]), int(x.shape[1])
    if not s8_decode_eligible(
        tokens, hidden, experts.topk, plan, experts.gate_cap, experts.up_cap
    ):
        return None
    if experts.intermediate_size != 288:
        return None

    use_fs = S8_FUSE_SITU if fuse_situ is None else bool(fuse_situ)
    use_fc = S8_FUSE_COMBINE if fuse_combine is None else bool(fuse_combine)
    if bn_mode is None:
        eff_bn_mode = S8_B2_BN_MODE if (tokens >= 9 and S8_B2_BN_MODE is not None) else S8_BN_MODE
    else:
        eff_bn_mode = int(bn_mode)
    if u_thresh is None:
        eff_u_thresh = S8_B2_U_THRESH if (tokens >= 9 and S8_B2_U_THRESH is not None) else S8_U_THRESH
    else:
        eff_u_thresh = int(u_thresh)

    src2dst, gammas, rows, tiles = routing
    tile_mask = getattr(tiles, "_inkling_tile_mask", None)
    tile_compact_start = getattr(tiles, "_inkling_tile_compact_start", None)
    src2compact = getattr(tiles, "_inkling_src2compact", None)
    use_zs = tile_mask is not None
    if not (use_zs or use_fs or use_fc):
        return None

    device = x.device
    ws = get_workspace(device)
    st = int(torch.cuda.current_stream(device).cuda_stream)
    use_pdl = pdl and plan.rows <= _WS_MAX_ROWS
    act, act_scale, row_scale = quantized
    n_compact = tokens * experts.topk  # <= 128

    if experts.nvfp4:
        lib = get_nvfp4_lib()
        if lib is None or not getattr(lib, "_inkling_v4", False):
            return None
        if (use_fs or use_fc) and tile_compact_start is None:
            return None
        if use_fc and src2compact is None:
            return None

        from . import nvfp4_ws5
        m13 = nvfp4_ws5.pack_w13_ws5_inplace(experts.w13_weight, experts.w13_weight_scale)
        m2 = nvfp4_ws5.pack_w2_ws5_inplace(experts.w2_weight, experts.w2_weight_scale)
        ws_base_13 = get_ws_base_s8(m13)
        ws_base_2 = get_ws_base_s8(m2)

        row_amax_w2 = getattr(tiles, "_inkling_row_amax_w2", None)
        if use_fs and row_amax_w2 is None:
            row_amax_w2 = ws["row_amax_w2"][:n_compact]
            row_amax_w2.zero_()

        w13_out = ws["situ_out"][:n_compact] if use_fs else torch.empty(
            (plan.rows, 576), dtype=torch.bfloat16, device=device
        )
        use_c1_v9 = (
            (NVFP4_C1_V9 or NVFP4_C1_V10 or NVFP4_C1_V10_FUSE_QDEQ)
            and get_nvfp4_version() >= 9
            and hasattr(lib, "nvfp4_w13_ws5_c1_v9_fwd")
            and hasattr(lib, "nvfp4_w2_ws5_c1_v9_fwd")
        )
        use_c1_v10_fuse_qdeq = (
            use_c1_v9
            and NVFP4_C1_V10_FUSE_QDEQ
            and use_fs
            and get_nvfp4_version() >= 10
            and hasattr(lib, "nvfp4_w13_ws5_c1_v10_fwd")
        )
        use_c1_qdeq = (
            use_c1_v10_fuse_qdeq
            or (
                use_c1_v9
                and NVFP4_C1_V9_QDEQ
                and use_fs
                and hasattr(lib, "nvfp4_situ_quant_dequant_c1_v9_fwd")
            )
        )
        w13_pw = 1 if (use_pdl and use_zs) else 0
        w13_td = 1 if (use_pdl and use_fs) else 0
        if use_c1_v10_fuse_qdeq:
            rc = lib.nvfp4_w13_ws5_c1_v10_fwd(
                act.data_ptr(), act_scale.data_ptr(), row_scale.data_ptr(),
                m13["slabs"].data_ptr(), ws_base_13.data_ptr(),
                m13["p_off"].data_ptr(), m13["patches"].data_ptr(),
                experts.ones.data_ptr(), tiles.data_ptr(),
                0 if not use_zs else tile_mask.data_ptr(),
                0 if tile_compact_start is None else tile_compact_start.data_ptr(),
                gammas.data_ptr() if use_fs else 0,
                row_amax_w2.data_ptr() if use_fs else 0,
                w13_out.data_ptr(),
                ws["tile_qdeq_done"].data_ptr(),
                576, 4608, plan.max_ctas, plan.tile,
                1 if use_zs else 0, 1 if use_fs else 0, 1,
                eff_bn_mode, 3 if eff_bn_mode == 3 else eff_u_thresh,
                w13_pw, w13_td, 1, S8_QUANT_FMA, st,
            )
            if rc != 0:
                raise RuntimeError(f"nvfp4_w13_ws5_c1_v10_fwd failed with cudaError={rc}")
        elif use_c1_v9:
            rc = lib.nvfp4_w13_ws5_c1_v9_fwd(
                act.data_ptr(), act_scale.data_ptr(), row_scale.data_ptr(),
                m13["slabs"].data_ptr(), ws_base_13.data_ptr(),
                m13["p_off"].data_ptr(), m13["patches"].data_ptr(),
                experts.ones.data_ptr(), tiles.data_ptr(),
                0 if not use_zs else tile_mask.data_ptr(),
                0 if tile_compact_start is None else tile_compact_start.data_ptr(),
                gammas.data_ptr() if use_fs else 0,
                row_amax_w2.data_ptr() if use_fs else 0,
                w13_out.data_ptr(),
                576, 4608, plan.max_ctas, plan.tile,
                1 if use_zs else 0, 1 if use_fs else 0,
                1 if use_c1_qdeq else 0,
                eff_bn_mode, 3 if eff_bn_mode == 3 else eff_u_thresh,
                w13_pw, w13_td, st,
            )
            if rc != 0:
                raise RuntimeError(f"nvfp4_w13_ws5_c1_v9_fwd failed with cudaError={rc}")
            if use_c1_qdeq:
                rc = lib.nvfp4_situ_quant_dequant_c1_v9_fwd(
                    w13_out.data_ptr(), w13_out.data_ptr(), row_amax_w2.data_ptr(),
                    tiles.data_ptr(), tile_compact_start.data_ptr(),
                    plan.max_ctas, plan.tile, n_compact,
                    1 if use_pdl else 0, 1 if use_pdl else 0, S8_QUANT_FMA, st,
                )
                if rc != 0:
                    raise RuntimeError(f"nvfp4_situ_quant_dequant_c1_v9_fwd failed with cudaError={rc}")
        else:
            rc = lib.nvfp4_w13_ws5_c1_v4_fwd(
                act.data_ptr(), act_scale.data_ptr(), row_scale.data_ptr(),
                m13["slabs"].data_ptr(), ws_base_13.data_ptr(),
                m13["p_off"].data_ptr(), m13["patches"].data_ptr(),
                experts.ones.data_ptr(), tiles.data_ptr(),
                0 if not use_zs else tile_mask.data_ptr(),
                0 if tile_compact_start is None else tile_compact_start.data_ptr(),
                gammas.data_ptr() if use_fs else 0,
                row_amax_w2.data_ptr() if use_fs else 0,
                w13_out.data_ptr(),
                576, 4608, plan.max_ctas, plan.tile,
                1 if use_zs else 0, 1 if use_fs else 0,
                2 if eff_bn_mode == 3 else eff_bn_mode, eff_u_thresh,
                w13_pw, w13_td, st,
            )
            if rc != 0:
                raise RuntimeError(f"nvfp4_w13_ws5_c1_v4_fwd failed with cudaError={rc}")

        if not use_fs:
            from .activation import swiglu_quantize
            m = plan.max_ctas
            act2, act2_s, act2_rs = swiglu_quantize(
                w13_out, gammas, experts.gate_cap, experts.up_cap,
                (rows, tiles[2 * m + 2 :], tiles[:m]), plan.tile, *experts.scales("w2"),
            )
        else:
            act2, act2_s, act2_rs = w13_out, None, None

        w2_amax, w2_denom = experts.scales("w2")
        down = ws["down_compact"] if use_fc else torch.empty(
            (plan.rows, 4608), dtype=torch.bfloat16, device=device
        )
        if use_fc and out_buf is None:
            out_buf = torch.empty((tokens, 4608), dtype=x.dtype, device=device)

        w2_pw = 2 if (use_pdl and use_fs) else 0
        w2_td = 1 if (use_pdl and use_fc) else 0
        if use_c1_v9:
            rc = lib.nvfp4_w2_ws5_c1_v9_fwd(
                act2.data_ptr(),
                0 if act2_s is None else act2_s.data_ptr(),
                0 if act2_rs is None else act2_rs.data_ptr(),
                row_amax_w2.data_ptr() if use_fs else 0,
                w2_amax.data_ptr() if use_fs else 0,
                w2_denom.data_ptr() if use_fs else 0,
                m2["w_pack"].data_ptr(), m2["ws5_pack"].data_ptr(), ws_base_2.data_ptr(),
                m2["p_off"].data_ptr(), m2["patches"].data_ptr(),
                experts.ones.data_ptr(), tiles.data_ptr(),
                0 if tile_compact_start is None else tile_compact_start.data_ptr(),
                down.data_ptr(),
                src2compact.data_ptr() if use_fc else 0,
                0 if (not use_fc or nlive_buf is None) else nlive_buf.data_ptr(),
                ws["panel_done"].data_ptr() if use_fc else 0,
                out_buf.data_ptr() if use_fc else 0,
                4608, 288, plan.max_ctas, plan.tile, tokens, n_compact,
                2 if use_c1_qdeq else (1 if use_fs else 0), 1 if use_fc else 0,
                w2_pw, w2_td, S8_QUANT_FMA, st,
            )
            if rc != 0:
                raise RuntimeError(f"nvfp4_w2_ws5_c1_v9_fwd failed with cudaError={rc}")
        elif hasattr(lib, "nvfp4_w2_ws5_c1_v6_fma_fwd"):
            rc = lib.nvfp4_w2_ws5_c1_v6_fma_fwd(
                act2.data_ptr(),
                0 if act2_s is None else act2_s.data_ptr(),
                0 if act2_rs is None else act2_rs.data_ptr(),
                row_amax_w2.data_ptr() if use_fs else 0,
                w2_amax.data_ptr() if use_fs else 0,
                w2_denom.data_ptr() if use_fs else 0,
                m2["w_pack"].data_ptr(), m2["ws5_pack"].data_ptr(), ws_base_2.data_ptr(),
                m2["p_off"].data_ptr(), m2["patches"].data_ptr(),
                experts.ones.data_ptr(), tiles.data_ptr(),
                0 if tile_compact_start is None else tile_compact_start.data_ptr(),
                down.data_ptr(),
                src2compact.data_ptr() if use_fc else 0,
                0 if (not use_fc or nlive_buf is None) else nlive_buf.data_ptr(),
                ws["panel_done"].data_ptr() if use_fc else 0,
                out_buf.data_ptr() if use_fc else 0,
                4608, 288, plan.max_ctas, plan.tile, tokens, n_compact,
                1 if use_fs else 0, 1 if use_fc else 0, int(S8_QUANT_FMA),
                w2_pw, w2_td, st,
            )
            if rc != 0:
                raise RuntimeError(f"nvfp4_w2_ws5_c1_v6_fma_fwd failed with cudaError={rc}")
        else:
            global _WARNED_W2_QUANT_FMA
            if use_fs and S8_QUANT_FMA and not _WARNED_W2_QUANT_FMA:
                _WARNED_W2_QUANT_FMA = True
                warnings.warn(
                    "Loaded NVFP4 C=1 library lacks nvfp4_w2_ws5_c1_v9_fwd / nvfp4_w2_ws5_c1_v6_fma_fwd; "
                    "W2 OTF NVFP4 quant will use unfused sub_rn despite INKLING_MOE_S8_QUANT_FMA=1.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            rc = lib.nvfp4_w2_ws5_c1_v4_fwd(
                act2.data_ptr(),
                0 if act2_s is None else act2_s.data_ptr(),
                0 if act2_rs is None else act2_rs.data_ptr(),
                row_amax_w2.data_ptr() if use_fs else 0,
                w2_amax.data_ptr() if use_fs else 0,
                w2_denom.data_ptr() if use_fs else 0,
                m2["w_pack"].data_ptr(), m2["ws5_pack"].data_ptr(), ws_base_2.data_ptr(),
                m2["p_off"].data_ptr(), m2["patches"].data_ptr(),
                experts.ones.data_ptr(), tiles.data_ptr(),
                0 if tile_compact_start is None else tile_compact_start.data_ptr(),
                down.data_ptr(),
                src2compact.data_ptr() if use_fc else 0,
                0 if (not use_fc or nlive_buf is None) else nlive_buf.data_ptr(),
                ws["panel_done"].data_ptr() if use_fc else 0,
                out_buf.data_ptr() if use_fc else 0,
                4608, 288, plan.max_ctas, plan.tile, tokens, n_compact,
                1 if use_fs else 0, 1 if use_fc else 0,
                w2_pw, w2_td, st,
            )
            if rc != 0:
                raise RuntimeError(f"nvfp4_w2_ws5_c1_v4_fwd failed with cudaError={rc}")
    else:
        mx6_lib = get_mx6_lib()
        if mx6_lib is not None:
            if (use_fs or use_fc) and tile_compact_start is None:
                return None
            if use_fc and src2compact is None:
                return None
            from . import trtllm_bmm as _bmm_mx6
            pk13 = _bmm_mx6._mx6_get_packed(experts.w13_weight, experts.w13_weight_scale, 0)
            pk2 = _bmm_mx6._mx6_get_packed(experts.w2_weight, experts.w2_weight_scale, 1)
            if pk13 is not None and pk2 is not None:
                sm4_13, ws3_13, es13, eo13, wm13, lut13, bws13 = pk13
                sm4_2, ws3_2, es2, eo2, wm2, lut2, bws2 = pk2
                w13_out = ws["situ_out"][:n_compact] if use_fs else torch.empty(
                    (plan.rows, 576), dtype=torch.bfloat16, device=device
                )
                w13_pw = 1 if (use_pdl and use_zs) else 0
                w13_td = 1 if (use_pdl and use_fs) else 0
                rc = mx6_lib.mx6_w13_c1_v10_fwd(
                    act.data_ptr(), act_scale.data_ptr(),
                    sm4_13.data_ptr(), ws3_13.data_ptr(),
                    es13.data_ptr(), eo13.data_ptr(),
                    wm13.data_ptr(), lut13.data_ptr(),
                    experts.ones.data_ptr(), tiles.data_ptr(),
                    0 if not use_zs else tile_mask.data_ptr(),
                    0 if tile_compact_start is None else tile_compact_start.data_ptr(),
                    gammas.data_ptr() if use_fs else 0,
                    w13_out.data_ptr(),
                    576, 4608, plan.max_ctas, plan.tile, int(bws13),
                    1 if use_zs else 0, 1 if use_fs else 0,
                    eff_bn_mode, 3 if eff_bn_mode == 3 else eff_u_thresh,
                    w13_pw, w13_td, st,
                )
                if rc != 0:
                    raise RuntimeError(f"mx6_w13_c1_v10_fwd failed with cudaError={rc}")
                if not use_fs:
                    from .activation import swiglu_quantize
                    m = plan.max_ctas
                    act2, act2_s, _ = swiglu_quantize(
                        w13_out, gammas, experts.gate_cap, experts.up_cap,
                        (rows, tiles[2 * m + 2 :], tiles[:m]), plan.tile, None, None,
                    )
                else:
                    act2, act2_s = w13_out, None
                down = ws["down_compact"] if use_fc else torch.empty(
                    (plan.rows, 4608), dtype=torch.bfloat16, device=device
                )
                if use_fc and out_buf is None:
                    out_buf = torch.empty((tokens, 4608), dtype=x.dtype, device=device)
                w2_pw = 2 if (use_pdl and use_fs) else 0
                w2_td = 1 if (use_pdl and use_fc) else 0
                rc = mx6_lib.mx6_w2_c1_v10_fwd(
                    act2.data_ptr(),
                    0 if act2_s is None else act2_s.data_ptr(),
                    sm4_2.data_ptr(), ws3_2.data_ptr(),
                    es2.data_ptr(), eo2.data_ptr(),
                    wm2.data_ptr(), lut2.data_ptr(),
                    experts.ones.data_ptr(), tiles.data_ptr(),
                    0 if tile_compact_start is None else tile_compact_start.data_ptr(),
                    down.data_ptr(),
                    src2compact.data_ptr() if use_fc else 0,
                    0 if (not use_fc or nlive_buf is None) else nlive_buf.data_ptr(),
                    ws["panel_done"].data_ptr() if use_fc else 0,
                    out_buf.data_ptr() if use_fc else 0,
                    4608, 288, plan.max_ctas, plan.tile, tokens, int(bws2),
                    0, 9, n_compact,
                    1 if use_fs else 0, 1 if use_fc else 0,
                    w2_pw, w2_td, st,
                )
                if rc != 0:
                    raise RuntimeError(f"mx6_w2_c1_v10_fwd failed with cudaError={rc}")
                if use_fc:
                    return out_buf
                from .moe import combine
                return combine(down, src2dst, experts.topk, x.dtype, out=out_buf)

        lib = get_mx7_lib()
        if lib is None or not getattr(lib, "_inkling_v4", False):
            return None
        if (use_fs or use_fc) and tile_compact_start is None:
            return None
        if use_fc and src2compact is None:
            return None

        mx_ver = get_mx7_version()
        v5_mode = "legacy"
        v5_13 = None
        v5_2 = None
        if MX7_V5_MODE == "panel" and mx_ver >= 5 and hasattr(lib, "mx7_w13_c1_v5_panel_fwd"):
            v5_13 = get_mx7_v5_panel_s8(experts.w13_weight, experts.w13_weight_scale, 0)
            v5_2 = get_mx7_v5_panel_s8(experts.w2_weight, experts.w2_weight_scale, 1)
            if v5_13 is not None and v5_2 is not None:
                v5_mode = "panel"
        elif MX7_V5_MODE == "compat" and mx_ver >= 5 and hasattr(lib, "mx7_w13_c1_v5_fwd"):
            v5_13 = get_mx7_v5_compat_s8(experts.w13_weight, experts.w13_weight_scale, 0)
            v5_2 = get_mx7_v5_compat_s8(experts.w2_weight, experts.w2_weight_scale, 1)
            if v5_13 is not None and v5_2 is not None:
                v5_mode = "compat"

        if v5_mode == "legacy":
            pk13 = get_mx7_packed_s8(experts.w13_weight, experts.w13_weight_scale, 0)
            pk2 = get_mx7_packed_s8(experts.w2_weight, experts.w2_weight_scale, 1)
            if pk13 is None or pk2 is None:
                return None
            slabs13, ws3_13, flips13, foff13, bws13 = pk13
            slabs2, ws3_2, flips2, foff2, bws2 = pk2

        w13_out = ws["situ_out"][:n_compact] if use_fs else torch.empty(
            (plan.rows, 576), dtype=torch.bfloat16, device=device
        )
        w13_pw = 1 if (use_pdl and use_zs) else 0
        w13_td = 1 if (use_pdl and use_fs) else 0
        if v5_mode == "panel":
            pp13, po13, sm13, so13, sf13, bws13, _ = v5_13
            rc = lib.mx7_w13_c1_v5_panel_fwd(
                act.data_ptr(), act_scale.data_ptr(),
                pp13.data_ptr(), po13.data_ptr(),
                sm13.data_ptr(), so13.data_ptr(), sf13.data_ptr(),
                experts.ones.data_ptr(), tiles.data_ptr(),
                0 if not use_zs else tile_mask.data_ptr(),
                0 if tile_compact_start is None else tile_compact_start.data_ptr(),
                gammas.data_ptr() if use_fs else 0,
                w13_out.data_ptr(),
                576, 4608, plan.max_ctas, plan.tile, int(bws13),
                1 if use_zs else 0, 1 if use_fs else 0,
                eff_bn_mode, eff_u_thresh,
                w13_pw, w13_td, st,
            )
        elif v5_mode == "compat":
            sl13, w3_13, sm13, so13, do13, sf13, bm13, bws13, _ = v5_13
            rc = lib.mx7_w13_c1_v5_fwd(
                act.data_ptr(), act_scale.data_ptr(),
                sl13.data_ptr(), w3_13.data_ptr(),
                sm13.data_ptr(), so13.data_ptr(), do13.data_ptr(),
                sf13.data_ptr(), bm13.data_ptr(),
                experts.ones.data_ptr(), tiles.data_ptr(),
                0 if not use_zs else tile_mask.data_ptr(),
                0 if tile_compact_start is None else tile_compact_start.data_ptr(),
                gammas.data_ptr() if use_fs else 0,
                w13_out.data_ptr(),
                576, 4608, plan.max_ctas, plan.tile, int(bws13),
                1 if use_zs else 0, 1 if use_fs else 0,
                eff_bn_mode, eff_u_thresh,
                w13_pw, w13_td, st,
            )
        else:
            rc = lib.mx7_w13_c1_v4_fwd(
                act.data_ptr(), act_scale.data_ptr(),
                slabs13.data_ptr(), ws3_13.data_ptr(), flips13.data_ptr(), foff13.data_ptr(),
                experts.ones.data_ptr(), tiles.data_ptr(),
                0 if not use_zs else tile_mask.data_ptr(),
                0 if tile_compact_start is None else tile_compact_start.data_ptr(),
                gammas.data_ptr() if use_fs else 0,
                w13_out.data_ptr(),
                576, 4608, plan.max_ctas, plan.tile, int(bws13),
                1 if use_zs else 0, 1 if use_fs else 0,
                eff_bn_mode, eff_u_thresh,
                w13_pw, w13_td, st,
            )
        if rc != 0:
            raise RuntimeError(f"mx7_w13_c1[{v5_mode}] failed with cudaError={rc}")

        if not use_fs:
            from .activation import swiglu_quantize
            m = plan.max_ctas
            act2, act2_s, _ = swiglu_quantize(
                w13_out, gammas, experts.gate_cap, experts.up_cap,
                (rows, tiles[2 * m + 2 :], tiles[:m]), plan.tile, None, None,
            )
        else:
            act2, act2_s = w13_out, None

        down = ws["down_compact"] if use_fc else torch.empty(
            (plan.rows, 4608), dtype=torch.bfloat16, device=device
        )
        if use_fc and out_buf is None:
            out_buf = torch.empty((tokens, 4608), dtype=x.dtype, device=device)

        w2_pw = 2 if (use_pdl and use_fs) else 0
        w2_td = 1 if (use_pdl and use_fc) else 0
        if v5_mode == "panel":
            pp2, po2, sm2, so2, sf2, bws2, _ = v5_2
            rc = lib.mx7_w2_c1_v5_panel_fwd(
                act2.data_ptr(),
                0 if act2_s is None else act2_s.data_ptr(),
                pp2.data_ptr(), po2.data_ptr(),
                sm2.data_ptr(), so2.data_ptr(), sf2.data_ptr(),
                experts.ones.data_ptr(), tiles.data_ptr(),
                0 if tile_compact_start is None else tile_compact_start.data_ptr(),
                down.data_ptr(),
                src2compact.data_ptr() if use_fc else 0,
                0 if (not use_fc or nlive_buf is None) else nlive_buf.data_ptr(),
                ws["panel_done"].data_ptr() if use_fc else 0,
                out_buf.data_ptr() if use_fc else 0,
                4608, 288, plan.max_ctas, plan.tile, tokens, int(bws2),
                0, 9, n_compact,
                1 if use_fs else 0, 1 if use_fc else 0,
                w2_pw, w2_td, st,
            )
        elif v5_mode == "compat":
            sl2, w3_2, sm2, so2, do2, sf2, bm2, bws2, _ = v5_2
            rc = lib.mx7_w2_c1_v5_fwd(
                act2.data_ptr(),
                0 if act2_s is None else act2_s.data_ptr(),
                sl2.data_ptr(), w3_2.data_ptr(),
                sm2.data_ptr(), so2.data_ptr(), do2.data_ptr(),
                sf2.data_ptr(), bm2.data_ptr(),
                experts.ones.data_ptr(), tiles.data_ptr(),
                0 if tile_compact_start is None else tile_compact_start.data_ptr(),
                down.data_ptr(),
                src2compact.data_ptr() if use_fc else 0,
                0 if (not use_fc or nlive_buf is None) else nlive_buf.data_ptr(),
                ws["panel_done"].data_ptr() if use_fc else 0,
                out_buf.data_ptr() if use_fc else 0,
                4608, 288, plan.max_ctas, plan.tile, tokens, int(bws2),
                0, 9, n_compact,
                1 if use_fs else 0, 1 if use_fc else 0,
                w2_pw, w2_td, st,
            )
        else:
            rc = lib.mx7_w2_c1_v4_fwd(
                act2.data_ptr(),
                0 if act2_s is None else act2_s.data_ptr(),
                slabs2.data_ptr(), ws3_2.data_ptr(), flips2.data_ptr(), foff2.data_ptr(),
                experts.ones.data_ptr(), tiles.data_ptr(),
                0 if tile_compact_start is None else tile_compact_start.data_ptr(),
                down.data_ptr(),
                src2compact.data_ptr() if use_fc else 0,
                0 if (not use_fc or nlive_buf is None) else nlive_buf.data_ptr(),
                ws["panel_done"].data_ptr() if use_fc else 0,
                out_buf.data_ptr() if use_fc else 0,
                4608, 288, plan.max_ctas, plan.tile, tokens, int(bws2),
                0, 9, n_compact,
                1 if use_fs else 0, 1 if use_fc else 0,
                w2_pw, w2_td, st,
            )
        if rc != 0:
            raise RuntimeError(f"mx7_w2_c1[{v5_mode}] failed with cudaError={rc}")

    if use_fc:
        return out_buf
    from .moe import combine
    return combine(down, src2dst, experts.topk, x.dtype, out=out_buf)
