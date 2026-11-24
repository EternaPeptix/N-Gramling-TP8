# SPDX-License-Identifier: Apache-2.0
"""SM121a (GB10) Native Block-Scaled FP4/FP8 MoE Prefill + Fused W2-Combine Loader.

Wave 2A integration module for Inkling-2-Small TP8 (E=384, H=4608, inter=288, topk=8):
  - Strictly gated to prefill chunks (``INKLING_PF_MOE_NATIVE=1``, ``plan.tile >= 64``,
    and ``tokens >= INKLING_PF_MOE_NATIVE_MIN_T`` [default 1024]), leaving C=1 decode
    (T <= 16) 100% untouched and 0-ULP bit-exact.
  - Zero extra weight VRAM: directly consumes live packed ``_inkling_nvfp4_ws5``
    (SLABS / W_PACK+WS5_PACK) OR ``_inkling_panel`` (WP + WSP / WS6) tensors!
  - Eliminates the ``w2_out [plan.rows, 4608]`` allocation (-1.96 GiB at T=16K) and
    7.42 TB of 1M-context DRAM round-trip via nb-outer L2-resident fused W2-Combine.

Environment variables:
  INKLING_PF_MOE_NATIVE=0|1           Enable native SM121a prefill MoE (default: 0)
  INKLING_PF_MOE_NATIVE_LIB=<path>    Path to libpf_moe_native_sm121.so
  INKLING_PF_MOE_NATIVE_MIN_T=1024    Minimum tokens to activate (default: 1024)
  INKLING_PF_MOE_COMBINE_MODE=0|1|2   0=unfused, 1=fused FP32 atomic+cast (default),
                                      2=fused BF16x2 atomic directly into out
  INKLING_PF_MOE_NATIVE_BM=0|64|128   0=auto (64 if tile==64 or max_ctas<=520 else 128)

Exact integration snippets (for Wave 2B wiring into patches-s6/):
  1) In ``trtllm_bmm.py:bmm(...)`` (right after ``nvfp4 = weight.dtype == torch.uint8``):
        if plan.tile >= 64:
            from . import pf_moe_native
            if pf_moe_native.is_active(plan):
                res = pf_moe_native.bmm_pf_native(
                    act, act_scale, row_scale, weight, weight_scale, ones, tiles,
                    plan, projection, row2tok=row2tok, as_row_major=as_row_major,
                )
                if res is not None:
                    return res

  2) In ``moe.py:Inkling2Experts.forward(...)`` (replacing ``down = self._gemm(...w2)`` +
     ``combine(down, ...)`` when ``reduce=True and push is None and plan.tile >= 64``):
        if reduce and push is None and plan.tile >= 64:
            from .ops import pf_moe_native
            if pf_moe_native.is_active(plan, tokens=x.shape[0]) and pf_moe_native.COMBINE_MODE != 0:
                sbuf = None
                if self.ep_size > 1:
                    from vllm.distributed.device_communicators.rdma_all_reduce import get_send_buffer
                    sbuf = get_send_buffer((x.shape[0], self.hidden_size), x.dtype)
                r2t_w2 = getattr(quantized[0], "_inkling_row2tok", None)
                act_v, act_s, row_s = activated
                out = pf_moe_native.w2_combine_pf_native(
                    act_v, act_s, row_s, self.w2_weight, self.w2_weight_scale,
                    self.ones, tiles, plan, src2dst, self.topk, x.dtype,
                    out=sbuf, row2tok=r2t_w2,
                    as_row_major=bool(getattr(act_s, "_inkling_rowmaj", False)),
                )
                if out is not None:
                    if self.ep_size == 1:
                        return out
                    return tensor_model_parallel_all_reduce(out)
"""

from __future__ import annotations

import ctypes
import os
import torch

ENABLED = os.environ.get("INKLING_PF_MOE_NATIVE", "0").strip().lower() in ("1", "true", "on")
MIN_TOKENS = int(os.environ.get("INKLING_PF_MOE_NATIVE_MIN_T", "1024"))
COMBINE_MODE = int(os.environ.get("INKLING_PF_MOE_COMBINE_MODE", "1"))
BM_OVERRIDE = int(os.environ.get("INKLING_PF_MOE_NATIVE_BM", "0"))
USE_MX7 = os.environ.get("INKLING_PF_MOE_NATIVE_MX7", "1").strip().lower() not in (
    "0", "", "off", "false", "no"
)
# Zeroing path for COMBINE_MODE=2 out [T, 4608] bf16 buffer:
#   "memset" (default): cudaMemsetAsync via pf_memset_zero (bypasses L2 WB RFO)
#   "cs":               zero_bf16_cs_kernel (streaming st.global.cs.v4.u32)
#   "torch":            legacy PyTorch out.zero_() (FillFunctor)
ZERO_MODE = os.environ.get("INKLING_PF_MOE_ZERO_MODE", "memset").strip().lower()
# Async pre-zero out [T, 4608] bf16 on a dedicated priority=0 stream BEFORE
# route() + W13 + situ_quantize so zeroing overlaps route/W13 and never evicts
# situ_quantize's w2_act from L2 right before W2! (opt-in, default 0)
ASYNC_ZERO = os.environ.get("INKLING_PF_MOE_ASYNC_ZERO", "0").strip().lower() in (
    "1", "true", "on", "yes",
)
# v3 weight L2 eviction policy: 0=ca (v2.1), 1=cs on last tile per expert (default), 2=cs all
W_EVICT_MODE = int(os.environ.get("INKLING_PF_MOE_W_EVICT", "1"))
# v3/v4 W2 panels/CTA for COMBINE_MODE=2: 2, 3, 4, 6; 0=auto
W2_PANELS = int(os.environ.get("INKLING_PF_MOE_W2_PANELS", "0"))
# v4 Expert-Persistent MoE Prefill (opt-in; eliminates 2-3x weight re-reads & Y_out zero/RFO!)
PERSISTENT = os.environ.get("INKLING_PF_MOE_PERSISTENT", "0").strip().lower() in (
    "1", "true", "on", "yes",
)
PERSISTENT_W13 = os.environ.get("INKLING_PF_MOE_PERSISTENT_W13", "1" if PERSISTENT else "0").strip().lower() in (
    "1", "true", "on", "yes",
)
# v4 W13 BM override: 0=auto (256 when tile>=256 else select_bm), 64, 128, or 256
V4_BM_OVERRIDE = int(os.environ.get("INKLING_PF_MOE_V4_BM", "0"))
# v4 weight eviction override: -1=auto (maps W_EVICT=2 -> 1 cs_last for persistent W13), 0=ca, 1=cs_last, 2=cs_all
V4_W_EVICT_OVERRIDE = int(os.environ.get("INKLING_PF_MOE_V4_W_EVICT", "-1"))
# v4 W2 per-slice in-L2 zeroing (COMBINE_MODE=2): eliminates 151 MB memset + 151 MB cold atomic R!
L2_ZERO_SLICE = os.environ.get("INKLING_PF_MOE_L2_ZERO_SLICE", "0").strip().lower() in (
    "1", "true", "on", "yes",
)
# v4 1-block LPT expert plan builder (shared across W13+W2, ~0.6 us, zero host sync)
LPT_PLAN = os.environ.get("INKLING_PF_MOE_LPT_PLAN", "1").strip().lower() not in (
    "0", "", "off", "false", "no",
)
# v4 optional mega-expert split threshold (multiple of 256; 0=no split, 1024=split rare >1024-row experts)
SPLIT_M = int(os.environ.get("INKLING_PF_MOE_SPLIT_M", "1024"))

# v4b Dedicated Prefill-Only Persistent Fused MX7 (libpf_mx7_prefill_sm121.so, opt-in default OFF!)
PF_MX7_V4 = os.environ.get("INKLING_PF_MX7_V4", "0").strip().lower() in ("1", "true", "on", "yes")
PF_MX7_V4_LIB_PATH = os.environ.get("INKLING_PF_MX7_V4_LIB", "/rdma/libpf_mx7_prefill_sm121.so").strip()
PF_MX7_V4_BM = int(os.environ.get("INKLING_PF_MX7_V4_BM", "256"))
PF_MX7_V4_FUSE_SITU = os.environ.get("INKLING_PF_MX7_V4_FUSE_SITU", "1").strip().lower() not in (
    "0", "", "off", "false", "no",
)
PF_MX7_V4_W2_PANELS = int(os.environ.get("INKLING_PF_MX7_V4_W2_PANELS", "4"))
PF_MX7_V4_W_EVICT = int(os.environ.get("INKLING_PF_MX7_V4_W_EVICT", "1"))
PF_MX7_V4_L2_ZERO_SLICE = os.environ.get(
    "INKLING_PF_MX7_V4_L2_ZERO_SLICE", "1" if L2_ZERO_SLICE else "0"
).strip().lower() in ("1", "true", "on", "yes")

# v4b Dedicated Prefill-Only LPT BM=128 NVFP4 W13 + Fused SiTU + 1-Warp/Row Quant (opt-in default OFF!)
PF_NVFP4_W13_V4B = os.environ.get("INKLING_PF_NVFP4_W13_V4B", "0").strip().lower() in (
    "1", "true", "on", "yes",
)
PF_NVFP4_W13_V4B_LIB_PATH = os.environ.get(
    "INKLING_PF_NVFP4_W13_V4B_LIB", "/rdma/libpf_nvfp4_w13_v4b.so"
).strip()
PF_NVFP4_W13_V4B_FUSE_SITU = os.environ.get("INKLING_PF_NVFP4_W13_V4B_FUSE_SITU", "1").strip().lower() not in (
    "0", "", "off", "false", "no",
)
PF_NVFP4_W13_V4B_W_EVICT = int(os.environ.get("INKLING_PF_NVFP4_W13_V4B_W_EVICT", "1"))
SHARED_PF_SCRATCH = os.environ.get(
    "INKLING_SHARED_PF_SCRATCH",
    os.environ.get("INKLING_FREE_PACKED_RAW_WEIGHTS", "0"),
).strip().lower() in ("1", "true", "on", "yes")

_LIB = None
_LIB_TRIED = False
_HERE = os.path.dirname(os.path.abspath(__file__))
_LIB_PATHS = (
    os.environ.get("INKLING_PF_MOE_NATIVE_LIB", ""),
    "/rdma/libpf_moe_native_sm121_v4.so",
    "/rdma/libpf_moe_native_sm121_v3.so",
    "/rdma/libpf_moe_native_sm121_v21.so",
    "/rdma/libpf_moe_native_sm121_v2.so",
    "/rdma/libpf_moe_native_sm121.so",
    os.path.join(_HERE, "libpf_moe_native_sm121_v4.so"),
    os.path.join(_HERE, "libpf_moe_native_sm121_v3.so"),
    os.path.join(_HERE, "libpf_moe_native_sm121_v21.so"),
    os.path.join(_HERE, "libpf_moe_native_sm121_v2.so"),
    os.path.join(_HERE, "libpf_moe_native_sm121.so"),
)

# Reusable per-device scratch buffers (Y_fp32 is maintained permanently zeroed via
# pf_fp32_to_bf16_zero after every layer so zero extra cudaMemsetAsync is needed).
_Y_FP32_CACHE: dict[torch.device, torch.Tensor] = {}
_R2T_CACHE: dict[torch.device, torch.Tensor] = {}
_PLAN_CACHE: dict[torch.device, torch.Tensor] = {}
_WORK_M128_CACHE: dict[torch.device, torch.Tensor] = {}
_Y_SITU_CACHE: dict[torch.device, torch.Tensor] = {}
_ZERO_STREAMS: dict[torch.device, torch.cuda.Stream] = {}
_MX7_V4_LIB: ctypes.CDLL | None = None
_MX7_V4_TRIED = False
_NVFP4_V4B_LIB: ctypes.CDLL | None = None
_NVFP4_V4B_TRIED = False
# v5 Dedicated Prefill-Only Co-Resident 1.000x WS5 NVFP4 W2 + Overlapped L2-Slice Combine (opt-in default OFF!)
PF_NVFP4_W2_V5 = os.environ.get("INKLING_PF_NVFP4_W2_V5", "0").strip().lower() in (
    "1", "true", "on", "yes",
)
PF_NVFP4_W2_V5_LIB_PATH = os.environ.get(
    "INKLING_PF_NVFP4_W2_V5_LIB", "/rdma/libpf_nvfp4_w2_v5.so"
).strip()
PF_NVFP4_W2_V5_PANELS = int(os.environ.get("INKLING_PF_NVFP4_W2_V5_PANELS", "0"))
PF_NVFP4_W2_V5_SPLIT_M = int(os.environ.get("INKLING_PF_NVFP4_W2_V5_SPLIT_M", "512"))
PF_NVFP4_W2_V5_W_EVICT = int(os.environ.get("INKLING_PF_NVFP4_W2_V5_W_EVICT", "1"))
_W2_V5_ZM_RAW = os.environ.get("INKLING_PF_NVFP4_W2_V5_ZERO_MODE", "auto").strip().lower()
_NVFP4_W2_V5_LIB: ctypes.CDLL | None = None
_NVFP4_W2_V5_TRIED = False
_W2_V5_PLAN_CACHE: dict[torch.device, torch.Tensor] = {}
_W2_V5_SLICE_STATE_CACHE: dict[torch.device, torch.Tensor] = {}
# Wave-V10 opt-in DBO-safe W2 SM reserve, V6 fused NVFP4 W13+SiTU+Quant & Route,
# V10 1-launch co-resident MX7 W2, and persistent prefill MoE scratch reuse (ALL default OFF!)
PF_W2_RESERVE_SMS = int(os.environ.get("INKLING_PF_W2_RESERVE_SMS", "8"))
PF_NVFP4_W13_V6 = os.environ.get("INKLING_PF_NVFP4_W13_V6", "0").strip().lower() in (
    "1", "true", "on", "yes",
)
PF_NVFP4_W13_V6_LIB_PATH = os.environ.get(
    "INKLING_PF_NVFP4_W13_V6_LIB", "/rdma/libpf_nvfp4_w13_fused_v6.so"
).strip()
PF_NVFP4_W13_V6_W_EVICT = int(os.environ.get("INKLING_PF_NVFP4_W13_V6_W_EVICT", "1"))
PF_ROUTE_V6 = os.environ.get("INKLING_PF_ROUTE_V6", "0").strip().lower() in (
    "1", "true", "on", "yes",
)
PF_MOE_SCRATCH_REUSE = os.environ.get("INKLING_PF_MOE_SCRATCH_REUSE", "0").strip().lower() in (
    "1", "true", "on", "yes",
)
PF_MX7_V10 = os.environ.get("INKLING_PF_MX7_V10", "0").strip().lower() in (
    "1", "true", "on", "yes",
)
PF_MX7_V10_LIB_PATH = os.environ.get(
    "INKLING_PF_MX7_V10_LIB", "/rdma/libpf_mx7_prefill_v10.so"
).strip()
PF_MX7_V10_PANELS = int(os.environ.get("INKLING_PF_MX7_V10_PANELS", "0"))
PF_MX7_V10_SPLIT_M = int(os.environ.get("INKLING_PF_MX7_V10_SPLIT_M", "512"))
PF_MX7_V10_W_EVICT = int(os.environ.get("INKLING_PF_MX7_V10_W_EVICT", "1"))
_MX7_V10_ZM_RAW = os.environ.get("INKLING_PF_MX7_V10_ZERO_MODE", "auto").strip().lower()
PF_MX7_V10_RESERVE_SMS = int(os.environ.get("INKLING_PF_MX7_V10_RESERVE_SMS", "8"))
_NVFP4_W13_V6_LIB: ctypes.CDLL | None = None
_NVFP4_W13_V6_TRIED = False
_V6_Y_SLOT_CACHE: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}
_MX7_V10_LIB: ctypes.CDLL | None = None
_MX7_V10_TRIED = False
_MX7_V10_PLAN_CACHE: dict[torch.device, torch.Tensor] = {}
_MX7_V10_SLICE_STATE_CACHE: dict[torch.device, torch.Tensor] = {}
_PF_MOE_SCRATCH_CACHE: dict[tuple[torch.device, str], torch.Tensor] = {}


def _bind_lib(lib: ctypes.CDLL) -> ctypes.CDLL:
    c_vp = ctypes.c_void_p
    c_i = ctypes.c_int
    c_i32 = ctypes.c_int32
    c_i64 = ctypes.c_int64
    c_u64 = ctypes.c_uint64

    lib.pf_nvfp4_w13_ws5_native.argtypes = [
        c_vp, c_vp, c_vp, c_vp,
        c_vp, c_vp, c_vp, c_vp,
        c_vp, c_vp, c_vp,
        c_i, c_i, c_i, c_i, c_i, c_i, c_u64,
    ]
    lib.pf_nvfp4_w13_ws5_native.restype = c_i

    lib.pf_nvfp4_w13_panel_native.argtypes = [
        c_vp, c_vp, c_vp, c_vp,
        c_vp, c_vp,
        c_vp, c_vp, c_vp,
        c_i, c_i, c_i, c_i, c_i, c_i, c_u64,
    ]
    lib.pf_nvfp4_w13_panel_native.restype = c_i

    lib.pf_nvfp4_w2_ws5_native.argtypes = [
        c_vp, c_vp, c_vp, c_vp,
        c_vp, c_vp, c_vp, c_vp, c_vp,
        c_vp, c_vp, c_vp,
        c_i, c_i, c_i, c_i, c_i, c_i, c_i, c_u64,
    ]
    lib.pf_nvfp4_w2_ws5_native.restype = c_i

    lib.pf_nvfp4_w2_panel_native.argtypes = [
        c_vp, c_vp, c_vp, c_vp,
        c_vp, c_vp,
        c_vp, c_vp, c_vp,
        c_i, c_i, c_i, c_i, c_i, c_i, c_u64,
    ]
    lib.pf_nvfp4_w2_panel_native.restype = c_i

    lib.pf_mxfp8_w13_native.argtypes = [
        c_vp, c_vp, c_vp,
        c_vp, c_vp,
        c_vp, c_vp, c_vp,
        c_i, c_i, c_i, c_i, c_i, c_u64,
    ]
    lib.pf_mxfp8_w13_native.restype = c_i

    lib.pf_mxfp8_w2_native.argtypes = [
        c_vp, c_vp, c_vp,
        c_vp, c_vp,
        c_vp, c_vp, c_vp,
        c_i, c_i, c_i, c_i, c_i, c_i, c_u64,
    ]
    lib.pf_mxfp8_w2_native.restype = c_i

    lib.pf_fp32_to_bf16_zero.argtypes = [c_vp, c_vp, c_i64, c_u64]
    lib.pf_fp32_to_bf16_zero.restype = c_i

    lib.pf_build_row2tok.argtypes = [c_vp, c_vp, c_i32, c_i32, c_u64]
    lib.pf_build_row2tok.restype = c_i

    # Optional v3 entrypoints (L2 weight eviction control, tunable W2 panels, memset/cs zeroing)
    if hasattr(lib, "pf_nvfp4_w13_ws5_native_v3"):
        lib.pf_nvfp4_w13_ws5_native_v3.argtypes = [c_vp] * 11 + [c_i] * 7 + [c_u64]
        lib.pf_nvfp4_w13_ws5_native_v3.restype = c_i
    if hasattr(lib, "pf_nvfp4_w2_ws5_native_v3"):
        lib.pf_nvfp4_w2_ws5_native_v3.argtypes = [c_vp] * 12 + [c_i] * 9 + [c_u64]
        lib.pf_nvfp4_w2_ws5_native_v3.restype = c_i
    if hasattr(lib, "pf_nvfp4_w2_panel_native_v3"):
        lib.pf_nvfp4_w2_panel_native_v3.argtypes = [c_vp] * 9 + [c_i] * 8 + [c_u64]
        lib.pf_nvfp4_w2_panel_native_v3.restype = c_i
    if hasattr(lib, "pf_mxfp8_w2_native_v3"):
        lib.pf_mxfp8_w2_native_v3.argtypes = [c_vp] * 8 + [c_i] * 8 + [c_u64]
        lib.pf_mxfp8_w2_native_v3.restype = c_i
    if hasattr(lib, "pf_memset_zero"):
        lib.pf_memset_zero.argtypes = [c_vp, c_i64, c_u64]
        lib.pf_memset_zero.restype = c_i
    if hasattr(lib, "pf_zero_cs_bf16"):
        lib.pf_zero_cs_bf16.argtypes = [c_vp, c_i64, c_u64]
        lib.pf_zero_cs_bf16.restype = c_i
    # Optional v4 Expert-Persistent entrypoints
    if hasattr(lib, "pf_build_expert_plan"):
        lib.pf_build_expert_plan.argtypes = [c_vp] * 2 + [c_i] * 5 + [c_u64]
        lib.pf_build_expert_plan.restype = c_i
    if hasattr(lib, "pf_nvfp4_w13_ws5_native_v4"):
        lib.pf_nvfp4_w13_ws5_native_v4.argtypes = [c_vp] * 12 + [c_i] * 9 + [c_u64]
        lib.pf_nvfp4_w13_ws5_native_v4.restype = c_i
    if hasattr(lib, "pf_nvfp4_w13_panel_native_v4"):
        lib.pf_nvfp4_w13_panel_native_v4.argtypes = [c_vp] * 10 + [c_i] * 9 + [c_u64]
        lib.pf_nvfp4_w13_panel_native_v4.restype = c_i
    if hasattr(lib, "pf_mxfp8_w13_native_v4"):
        lib.pf_mxfp8_w13_native_v4.argtypes = [c_vp] * 9 + [c_i] * 8 + [c_u64]
        lib.pf_mxfp8_w13_native_v4.restype = c_i
    if hasattr(lib, "pf_nvfp4_w2_ws5_native_v4"):
        lib.pf_nvfp4_w2_ws5_native_v4.argtypes = [c_vp] * 13 + [c_i] * 12 + [c_u64]
        lib.pf_nvfp4_w2_ws5_native_v4.restype = c_i
    if hasattr(lib, "pf_nvfp4_w2_panel_native_v4"):
        lib.pf_nvfp4_w2_panel_native_v4.argtypes = [c_vp] * 10 + [c_i] * 11 + [c_u64]
        lib.pf_nvfp4_w2_panel_native_v4.restype = c_i
    if hasattr(lib, "pf_mxfp8_w2_native_v4"):
        lib.pf_mxfp8_w2_native_v4.argtypes = [c_vp] * 9 + [c_i] * 11 + [c_u64]
        lib.pf_mxfp8_w2_native_v4.restype = c_i
    if hasattr(lib, "pf_zero_slice_wb_bf16"):
        lib.pf_zero_slice_wb_bf16.argtypes = [c_vp] + [c_i] * 3 + [c_u64]
        lib.pf_zero_slice_wb_bf16.restype = c_i
    return lib


def _get_mx7_v5_lib() -> ctypes.CDLL | None:
    if not USE_MX7:
        return None
    from . import trtllm_bmm
    lib = trtllm_bmm._get_mx7_lib()
    if lib is not None and getattr(trtllm_bmm, "_MX7_HAS_NATIVE_V5", False):
        return lib
    return None


def _get_mx7_v6_lib() -> ctypes.CDLL | None:
    if not USE_MX7:
        return None
    from . import trtllm_bmm
    lib = trtllm_bmm._get_mx7_lib()
    if lib is not None and getattr(trtllm_bmm, "_MX7_HAS_NATIVE_V6", False):
        return lib
    return None


def _get_pf_mx7_v10_lib() -> ctypes.CDLL | None:
    """Lazy loader for Wave-V10 `libpf_mx7_prefill_v10.so` (1-launch co-resident MX7 W2 + DBO reserve)."""
    global _MX7_V10_LIB, _MX7_V10_TRIED
    if not (PF_MX7_V10 and USE_MX7):
        return None
    if _MX7_V10_LIB is not None:
        return _MX7_V10_LIB
    if _MX7_V10_TRIED:
        return None
    _MX7_V10_TRIED = True
    c_vp, c_i, c_u64 = ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64
    for p in (
        PF_MX7_V10_LIB_PATH,
        "/rdma/libpf_mx7_prefill_v10.so",
        os.path.join(_HERE, "libpf_mx7_prefill_v10.so"),
    ):
        if p and os.path.isfile(p):
            lib = ctypes.CDLL(p)
            lib.pf_mx7_build_expert_plan.argtypes = [c_vp] * 2 + [c_i] * 5 + [c_u64]
            lib.pf_mx7_build_expert_plan.restype = c_i
            lib.pf_mx7_zero_slice_wb_bf16.argtypes = [c_vp] + [c_i] * 3 + [c_u64]
            lib.pf_mx7_zero_slice_wb_bf16.restype = c_i
            lib.pf_mx7_w13_prefill_fused_situ.argtypes = [c_vp] * 13 + [c_i] * 11 + [c_u64]
            lib.pf_mx7_w13_prefill_fused_situ.restype = c_i
            lib.pf_mx7_w13_prefill_unfused.argtypes = [c_vp] * 11 + [c_i] * 9 + [c_u64]
            lib.pf_mx7_w13_prefill_unfused.restype = c_i
            lib.pf_mx7_w2_prefill_persistent.argtypes = [c_vp] * 11 + [c_i] * 12 + [c_u64]
            lib.pf_mx7_w2_prefill_persistent.restype = c_i
            lib.pf_mx7_v10_version.argtypes = []
            lib.pf_mx7_v10_version.restype = c_i
            lib.pf_mx7_v10_set_reserve_sms.argtypes = [c_i]
            lib.pf_mx7_v10_set_reserve_sms.restype = c_i
            lib.pf_mx7_v10_get_reserve_sms.argtypes = []
            lib.pf_mx7_v10_get_reserve_sms.restype = c_i
            lib.pf_mx7_build_plan_v10.argtypes = [c_vp] * 2 + [c_i] * 5 + [c_u64]
            lib.pf_mx7_build_plan_v10.restype = c_i
            lib.pf_mx7_w2_v10_unfused.argtypes = [c_vp] * 11 + [c_i] * 8 + [c_u64]
            lib.pf_mx7_w2_v10_unfused.restype = c_i
            lib.pf_mx7_w2_v10_combine.argtypes = [c_vp] * 12 + [c_i] * 12 + [c_u64]
            lib.pf_mx7_w2_v10_combine.restype = c_i
            lib.pf_mx7_v10_set_reserve_sms(int(PF_MX7_V10_RESERVE_SMS))
            _MX7_V10_LIB = lib
            print(
                f"[inkling-pf-moe-native] loaded MX7 v10 prefill {p} "
                f"(ver={lib.pf_mx7_v10_version()}, reserve_sms={lib.pf_mx7_v10_get_reserve_sms()}, "
                f"panels={PF_MX7_V10_PANELS}, split_m={PF_MX7_V10_SPLIT_M}, zm={_MX7_V10_ZM_RAW})",
                flush=True,
            )
            return _MX7_V10_LIB
    return None


def _get_pf_mx7_v4_lib() -> ctypes.CDLL | None:
    """Lazy loader for dedicated prefill-only `libpf_mx7_prefill_sm121.so` (or V10 superset)."""
    global _MX7_V4_LIB, _MX7_V4_TRIED
    if PF_MX7_V10 and USE_MX7:
        v10 = _get_pf_mx7_v10_lib()
        if v10 is not None:
            return v10
    if not (PF_MX7_V4 and USE_MX7):
        return None
    if _MX7_V4_LIB is not None:
        return _MX7_V4_LIB
    if _MX7_V4_TRIED:
        return None
    _MX7_V4_TRIED = True
    c_vp, c_i, c_u64 = ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64
    for p in (
        PF_MX7_V4_LIB_PATH,
        "/rdma/libpf_mx7_prefill_sm121.so",
        os.path.join(_HERE, "libpf_mx7_prefill_sm121.so"),
    ):
        if p and os.path.isfile(p):
            lib = ctypes.CDLL(p)
            lib.pf_mx7_build_expert_plan.argtypes = [c_vp] * 2 + [c_i] * 5 + [c_u64]
            lib.pf_mx7_build_expert_plan.restype = c_i
            lib.pf_mx7_zero_slice_wb_bf16.argtypes = [c_vp] + [c_i] * 3 + [c_u64]
            lib.pf_mx7_zero_slice_wb_bf16.restype = c_i
            lib.pf_mx7_w13_prefill_fused_situ.argtypes = [c_vp] * 13 + [c_i] * 11 + [c_u64]
            lib.pf_mx7_w13_prefill_fused_situ.restype = c_i
            lib.pf_mx7_w13_prefill_unfused.argtypes = [c_vp] * 11 + [c_i] * 9 + [c_u64]
            lib.pf_mx7_w13_prefill_unfused.restype = c_i
            lib.pf_mx7_w2_prefill_persistent.argtypes = [c_vp] * 11 + [c_i] * 12 + [c_u64]
            lib.pf_mx7_w2_prefill_persistent.restype = c_i
            _MX7_V4_LIB = lib
            print(
                f"[inkling-pf-moe-native] loaded MX7 v4 prefill {p} "
                f"(bm={PF_MX7_V4_BM}, fuse_situ={int(PF_MX7_V4_FUSE_SITU)}, "
                f"w2_p64={PF_MX7_V4_W2_PANELS}, l2_zero={int(PF_MX7_V4_L2_ZERO_SLICE)})",
                flush=True,
            )
            return _MX7_V4_LIB
    return None


def _get_pf_nvfp4_w13_v4b_lib() -> ctypes.CDLL | None:
    """Lazy loader for dedicated prefill-only `libpf_nvfp4_w13_v4b.so` (strictly opt-in)."""
    global _NVFP4_V4B_LIB, _NVFP4_V4B_TRIED
    if not PF_NVFP4_W13_V4B:
        return None
    if _NVFP4_V4B_LIB is not None:
        return _NVFP4_V4B_LIB
    if _NVFP4_V4B_TRIED:
        return None
    _NVFP4_V4B_TRIED = True
    c_vp, c_i, c_u64 = ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64
    for p in (
        PF_NVFP4_W13_V4B_LIB_PATH,
        "/rdma/libpf_nvfp4_w13_v4b.so",
        os.path.join(_HERE, "libpf_nvfp4_w13_v4b.so"),
    ):
        if p and os.path.isfile(p):
            lib = ctypes.CDLL(p)
            lib.pf_nvfp4_build_m128_worklist.argtypes = [c_vp] * 2 + [c_i] * 4 + [c_u64]
            lib.pf_nvfp4_build_m128_worklist.restype = c_i
            lib.pf_nvfp4_w13_ws5_v4b_unfused.argtypes = [c_vp] * 11 + [c_i] * 6 + [c_u64]
            lib.pf_nvfp4_w13_ws5_v4b_unfused.restype = c_i
            lib.pf_nvfp4_w13_ws5_v4b_fused_situ.argtypes = [c_vp] * 12 + [c_i] * 6 + [c_u64]
            lib.pf_nvfp4_w13_ws5_v4b_fused_situ.restype = c_i
            lib.pf_nvfp4_quant_288.argtypes = [c_vp] * 9 + [c_i] * 3 + [c_u64]
            lib.pf_nvfp4_quant_288.restype = c_i
            _NVFP4_V4B_LIB = lib
            print(
                f"[inkling-pf-moe-native] loaded NVFP4 W13 v4b {p} "
                f"(fuse_situ={int(PF_NVFP4_W13_V4B_FUSE_SITU)}, w_evict={PF_NVFP4_W13_V4B_W_EVICT})",
                flush=True,
            )
            return _NVFP4_V4B_LIB
    return None


def _get_pf_nvfp4_w13_v6_lib() -> ctypes.CDLL | None:
    """Lazy loader for DBO-safe `libpf_nvfp4_w13_fused_v6.so` (W13+SiTU+Quant + Route v6)."""
    global _NVFP4_W13_V6_LIB, _NVFP4_W13_V6_TRIED
    if not (PF_NVFP4_W13_V6 or PF_ROUTE_V6):
        return None
    if _NVFP4_W13_V6_LIB is not None:
        return _NVFP4_W13_V6_LIB
    if _NVFP4_W13_V6_TRIED:
        return None
    _NVFP4_W13_V6_TRIED = True
    c_vp, c_i, c_f, c_u64 = ctypes.c_void_p, ctypes.c_int, ctypes.c_float, ctypes.c_uint64
    for p in (
        PF_NVFP4_W13_V6_LIB_PATH,
        "/rdma/libpf_nvfp4_w13_fused_v6.so",
        os.path.join(_HERE, "libpf_nvfp4_w13_fused_v6.so"),
    ):
        if p and os.path.isfile(p):
            lib = ctypes.CDLL(p)
            lib.pf_nvfp4_w13_v6_version.argtypes = []
            lib.pf_nvfp4_w13_v6_version.restype = c_i
            lib.pf_nvfp4_w13_v6_slot_bytes.argtypes = []
            lib.pf_nvfp4_w13_v6_slot_bytes.restype = c_i
            lib.pf_nvfp4_w13_v6_state_bytes.argtypes = []
            lib.pf_nvfp4_w13_v6_state_bytes.restype = c_i
            lib.pf_nvfp4_build_w13_w2_plan_v6.argtypes = [c_vp] * 4 + [c_i] * 6 + [c_u64]
            lib.pf_nvfp4_build_w13_w2_plan_v6.restype = c_i
            lib.pf_nvfp4_w13_situ_quant_v6.argtypes = [c_vp] * 18 + [c_i] * 6 + [c_u64]
            lib.pf_nvfp4_w13_situ_quant_v6.restype = c_i
            lib.pf_nvfp4_build_m128_worklist.argtypes = [c_vp] * 2 + [c_i] * 4 + [c_u64]
            lib.pf_nvfp4_build_m128_worklist.restype = c_i
            lib.pf_nvfp4_w13_ws5_v4b_unfused.argtypes = [c_vp] * 11 + [c_i] * 6 + [c_u64]
            lib.pf_nvfp4_w13_ws5_v4b_unfused.restype = c_i
            lib.pf_nvfp4_w13_ws5_v4b_fused_situ.argtypes = [c_vp] * 12 + [c_i] * 6 + [c_u64]
            lib.pf_nvfp4_w13_ws5_v4b_fused_situ.restype = c_i
            lib.pf_route_tokens_place_v6.argtypes = [c_vp] * 11 + [c_i, c_f] + [c_i] * 5 + [c_u64]
            lib.pf_route_tokens_place_v6.restype = c_i
            _NVFP4_W13_V6_LIB = lib
            print(
                f"[inkling-pf-moe-native] loaded NVFP4 W13+Route v6 {p} "
                f"(ver={lib.pf_nvfp4_w13_v6_version()}, w13_v6={int(PF_NVFP4_W13_V6)}, "
                f"route_v6={int(PF_ROUTE_V6)}, scratch_reuse={int(PF_MOE_SCRATCH_REUSE)})",
                flush=True,
            )
            return _NVFP4_W13_V6_LIB
    return None


def _get_pf_nvfp4_w2_v5_lib() -> ctypes.CDLL | None:
    """Lazy loader for DBO-safe `libpf_nvfp4_w2_v5_dbo.so` / `libpf_nvfp4_w2_v5.so` (strictly opt-in)."""
    global _NVFP4_W2_V5_LIB, _NVFP4_W2_V5_TRIED
    if not PF_NVFP4_W2_V5:
        return None
    if _NVFP4_W2_V5_LIB is not None:
        return _NVFP4_W2_V5_LIB
    if _NVFP4_W2_V5_TRIED:
        return None
    _NVFP4_W2_V5_TRIED = True
    c_vp, c_i, c_u64 = ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64
    for p in (
        PF_NVFP4_W2_V5_LIB_PATH,
        "/rdma/libpf_nvfp4_w2_v5_dbo.so",
        "/rdma/libpf_nvfp4_w2_v5.so",
        os.path.join(_HERE, "libpf_nvfp4_w2_v5_dbo.so"),
        os.path.join(_HERE, "libpf_nvfp4_w2_v5.so"),
    ):
        if p and os.path.isfile(p):
            lib = ctypes.CDLL(p)
            lib.pf_nvfp4_w2_v5_version.argtypes = []
            lib.pf_nvfp4_w2_v5_version.restype = c_i
            lib.pf_nvfp4_w2_build_plan_v5.argtypes = [c_vp] * 2 + [c_i] * 5 + [c_u64]
            lib.pf_nvfp4_w2_build_plan_v5.restype = c_i
            lib.pf_nvfp4_w2_ws5_v5_unfused.argtypes = [c_vp] * 14 + [c_i] * 8 + [c_u64]
            lib.pf_nvfp4_w2_ws5_v5_unfused.restype = c_i
            lib.pf_nvfp4_w2_ws5_v5_combine.argtypes = [c_vp] * 15 + [c_i] * 12 + [c_u64]
            lib.pf_nvfp4_w2_ws5_v5_combine.restype = c_i
            lib.pf_nvfp4_w2_panel_v5_combine.argtypes = [c_vp] * 12 + [c_i] * 11 + [c_u64]
            lib.pf_nvfp4_w2_panel_v5_combine.restype = c_i
            res_sms = 0
            if hasattr(lib, "pf_nvfp4_w2_v5_set_reserve_sms"):
                lib.pf_nvfp4_w2_v5_set_reserve_sms.argtypes = [c_i]
                lib.pf_nvfp4_w2_v5_set_reserve_sms.restype = c_i
                lib.pf_nvfp4_w2_v5_get_reserve_sms.argtypes = []
                lib.pf_nvfp4_w2_v5_get_reserve_sms.restype = c_i
                lib.pf_nvfp4_w2_v5_set_reserve_sms(int(PF_W2_RESERVE_SMS))
                res_sms = int(lib.pf_nvfp4_w2_v5_get_reserve_sms())
            _NVFP4_W2_V5_LIB = lib
            print(
                f"[inkling-pf-moe-native] loaded NVFP4 W2 v5 {p} "
                f"(ver={lib.pf_nvfp4_w2_v5_version()}, reserve_sms={res_sms}, "
                f"panels={PF_NVFP4_W2_V5_PANELS}, split_m={PF_NVFP4_W2_V5_SPLIT_M}, "
                f"zero_mode={_W2_V5_ZM_RAW}, w_evict={PF_NVFP4_W2_V5_W_EVICT})",
                flush=True,
            )
            return _NVFP4_W2_V5_LIB
    return None


def get_lib(path: str | None = None) -> ctypes.CDLL | None:
    global _LIB, _LIB_TRIED
    if path is not None:
        if os.path.isfile(path):
            _LIB = _bind_lib(ctypes.CDLL(path))
            _LIB_TRIED = True
            _v3 = int(hasattr(_LIB, "pf_nvfp4_w13_ws5_native_v3"))
            _v4 = int(hasattr(_LIB, "pf_nvfp4_w13_ws5_native_v4"))
            print(
                f"[inkling-pf-moe-native] loaded {path} "
                f"(v3={_v3}, v4={_v4}, persist={int(PERSISTENT)}, l2_zero={int(L2_ZERO_SLICE)}, "
                f"cm={COMBINE_MODE}, zero={ZERO_MODE}, async_zero={int(ASYNC_ZERO)}, "
                f"w_evict={W_EVICT_MODE}, w2_panels={W2_PANELS})",
                flush=True,
            )
            return _LIB
        return None
    if _LIB is not None:
        return _LIB
    if _LIB_TRIED:
        return None
    _LIB_TRIED = True
    for p in _LIB_PATHS:
        if p and os.path.isfile(p):
            _LIB = _bind_lib(ctypes.CDLL(p))
            _v3 = int(hasattr(_LIB, "pf_nvfp4_w13_ws5_native_v3"))
            _v4 = int(hasattr(_LIB, "pf_nvfp4_w13_ws5_native_v4"))
            print(
                f"[inkling-pf-moe-native] loaded {p} "
                f"(v3={_v3}, v4={_v4}, persist={int(PERSISTENT)}, l2_zero={int(L2_ZERO_SLICE)}, "
                f"cm={COMBINE_MODE}, zero={ZERO_MODE}, async_zero={int(ASYNC_ZERO)}, "
                f"w_evict={W_EVICT_MODE}, w2_panels={W2_PANELS})",
                flush=True,
            )
            return _LIB
    return None


def is_active(plan, tokens: int | None = None) -> bool:
    """Strictly gate native SM121a prefill MoE to long prefill chunks (never decode)."""
    if not ENABLED:
        return False
    if plan.tile < 64:
        return False
    if tokens is not None and tokens < MIN_TOKENS:
        return False
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return False
    # When tokens is omitted (inside trtllm_bmm.bmm), plan.tile >= 128 guarantees T >= 769;
    # enforce MIN_TOKENS via plan.max_ctas lower bound if MIN_TOKENS >= 1024.
    if tokens is None and MIN_TOKENS > 768 and plan.tile < 128:
        return False
    return get_lib() is not None


def select_bm(plan, bm: int | None = None) -> int:
    if bm is not None and bm in (64, 128) and plan.tile % bm == 0:
        return bm
    if BM_OVERRIDE in (64, 128) and plan.tile % BM_OVERRIDE == 0:
        return BM_OVERRIDE
    if plan.tile == 64 or plan.max_ctas <= 520:
        return 64
    return 128


def select_bm_v4(plan, bm: int | None = None) -> int:
    if bm is not None and bm in (64, 128, 256):
        return bm
    if V4_BM_OVERRIDE in (64, 128, 256):
        return V4_BM_OVERRIDE
    if BM_OVERRIDE in (64, 128):
        return BM_OVERRIDE
    if plan.tile >= 256:
        return 256
    if plan.tile == 64 or plan.max_ctas <= 520:
        return 64
    return 128


def select_v4_w_evict(w_evict_mode: int | None = None) -> int:
    if w_evict_mode is not None:
        return w_evict_mode
    if V4_W_EVICT_OVERRIDE >= 0:
        return V4_W_EVICT_OVERRIDE
    return 0 if W_EVICT_MODE == 0 else 1


def select_w2_panels(
    tokens: int,
    combine_mode: int,
    w2_panels: int | None = None,
    *,
    persistent: bool = False,
) -> int:
    if combine_mode != 2:
        return 2 if combine_mode == 1 else 6
    p = W2_PANELS if w2_panels is None else w2_panels
    if p in (2, 3, 4, 6):
        if p == 6 and not persistent:
            return 4
        return p
    if persistent:
        # Persistent W2: 4 panels @ T<=8192 (8 MiB Y_out + 10.6 MiB A = 18.6 MiB < 24 MiB L2),
        # 3 panels @ T>8192 (12 MiB Y_out + 5.3 MiB active A + serpentine = 17.3 MiB < 24 MiB L2).
        return 4 if tokens <= 8192 else 3
    # Auto (v3): 4 panels @ T<=8192 (8 MiB Y_out slice fits 24 MiB L2 cleanly),
    # 2 panels @ T>8192 (keeps Y_out slice <= 6 MiB @ T=12K inside 24 MiB L2).
    return 4 if tokens <= 8192 else 2


def select_w2_v5_panels(
    tokens: int,
    combine_mode: int,
    w2_panels: int | None = None,
) -> int:
    """L2-aware panel selector for `pf_nvfp4_w2_v5` (`PANELS in {1, 2, 3, 4, 6}`):
      - `tokens <= 8192`: 3 panels (`6.0 MiB` Out slice -> TWO adjacent slices `12 MiB` +
        `10.6 MiB` A = `22.6 MiB < 24 MiB`, enabling `zero_mode=1` overlapped slice zeroing!)
      - `8192 < tokens <= 12288`: 2 panels (`6.0 MiB` Out + `16.0 MiB` A = `22.0 MiB < 24 MiB`)
      - `tokens > 12288`: 1 panel (`4.0 MiB` Out @ T=16K, preventing 24 MiB L2 thrashing!)
    """
    if combine_mode != 2:
        return 2 if combine_mode == 1 else 6
    p = PF_NVFP4_W2_V5_PANELS if (w2_panels is None or w2_panels <= 0) else w2_panels
    if p in (1, 2, 3, 4, 6):
        return p
    if tokens <= 8192:
        return 3
    if tokens <= 12288:
        return 2
    return 1


def select_w2_v5_zero_mode(
    tokens: int,
    combine_mode: int,
    panels: int,
    out_prezeroed: bool,
) -> int:
    """Select `pf_nvfp4_w2_v5` in-L2 slice zeroing mode (`0`=prezeroed, `1`=overlapped, `2`=gated)."""
    if combine_mode != 2 or out_prezeroed or tokens <= 0:
        return 0
    if _W2_V5_ZM_RAW in ("-1", "off", "prezero", "none", "false"):
        return 0
    if _W2_V5_ZM_RAW == "1":
        return 1
    if _W2_V5_ZM_RAW == "2":
        return 2
    # Auto: overlapped `zero_mode=1` when 2 slices + A fit comfortably inside GB10's 24 MiB L2!
    if tokens <= 8192 and panels <= 3:
        return 1
    return 2


def select_mx7_v10_panels(tokens: int, combine_mode: int, w2_panels: int | None = None) -> int:
    """L2-aware `PANELS64_PER_CTA in {3,4,6,8,9,12}` selector for `pf_mx7_w2_v10_combine`."""
    if combine_mode != 2:
        return 4 if combine_mode == 1 else 12
    p = PF_MX7_V10_PANELS if (w2_panels is None or w2_panels <= 0) else w2_panels
    if p in (3, 4, 6, 8, 9, 12):
        return p
    if tokens <= 8192:
        return 6
    if tokens <= 12288:
        return 4
    return 3


def select_mx7_v10_zero_mode(
    tokens: int, combine_mode: int, panels64: int, out_prezeroed: bool
) -> int:
    """Select `pf_mx7_w2_v10_combine` in-L2 slice zeroing (`0`=prezeroed, `1`=overlapped, `2`=gated)."""
    if combine_mode != 2 or out_prezeroed or tokens <= 0:
        return 0
    if _MX7_V10_ZM_RAW in ("-1", "off", "prezero", "none", "false"):
        return 0
    if _MX7_V10_ZM_RAW == "1":
        return 1
    if _MX7_V10_ZM_RAW == "2":
        return 2
    if tokens <= 8192 and panels64 <= 6:
        return 1
    return 2


def uses_in_kernel_slice_zero(plan, tokens: int, nvfp4: bool) -> bool:
    """Return True when the active W2 prefill kernel zeroes `Out` slices directly inside L2,
    so `moe.py` skips redundant 75.5 MB `zero_stream` memsets (`INKLING_PF_MOE_ASYNC_ZERO=1`)."""
    if plan.tile < 64 or plan.tile % 64 != 0 or tokens < MIN_TOKENS:
        return False
    if COMBINE_MODE != 2:
        return False
    if nvfp4 and PF_NVFP4_W2_V5 and _get_pf_nvfp4_w2_v5_lib() is not None:
        p5 = select_w2_v5_panels(tokens, 2)
        return select_w2_v5_zero_mode(tokens, 2, p5, out_prezeroed=False) > 0
    if not nvfp4:
        from . import trtllm_bmm as _bmm_mx6
        if getattr(_bmm_mx6, "_MX6_ENABLED", False) and getattr(_bmm_mx6, "_MX6_FREE_RAW", False):
            return bool(PERSISTENT and L2_ZERO_SLICE)
    if (not nvfp4) and PF_MX7_V10 and _get_pf_mx7_v10_lib() is not None:
        p10 = select_mx7_v10_panels(tokens, 2)
        return select_mx7_v10_zero_mode(tokens, 2, p10, out_prezeroed=False) > 0
    if (not nvfp4) and PF_MX7_V4 and PF_MX7_V4_L2_ZERO_SLICE and _get_pf_mx7_v4_lib() is not None:
        return True
    return bool(PERSISTENT and L2_ZERO_SLICE)


def _get_w2_v5_slice_state(device: torch.device) -> torch.Tensor:
    st_buf = _W2_V5_SLICE_STATE_CACHE.get(device)
    if st_buf is None:
        st_buf = torch.zeros(512, dtype=torch.uint32, device=device)
        _W2_V5_SLICE_STATE_CACHE[device] = st_buf
    return st_buf


def _get_or_build_w2_v5_plan(
    tiles: torch.Tensor, plan, num_experts: int, w2_v5_lib: ctypes.CDLL, stream: int
) -> tuple[int, int]:
    if not LPT_PLAN or num_experts > 1024:
        return 0, num_experts
    cached = getattr(tiles, "_inkling_w2_v5_plan", None)
    if cached is not None and cached[2] == num_experts:
        return cached[0], cached[1]
    eff_split = ((PF_NVFP4_W2_V5_SPLIT_M + 63) // 64) * 64 if PF_NVFP4_W2_V5_SPLIT_M > 0 else 0
    max_work = (
        num_experts + (plan.max_ctas * plan.tile) // eff_split + 1
        if eff_split > 0
        else num_experts
    )
    need_i32 = 3 * max_work + 1
    buf = _W2_V5_PLAN_CACHE.get(tiles.device)
    if buf is None or buf.numel() < need_i32:
        buf = torch.empty(need_i32, dtype=torch.int32, device=tiles.device)
        _W2_V5_PLAN_CACHE[tiles.device] = buf
    sub = buf[:need_i32]
    rc = w2_v5_lib.pf_nvfp4_w2_build_plan_v5(
        tiles.data_ptr(), sub.data_ptr(),
        plan.max_ctas, plan.tile, num_experts, eff_split, max_work, stream,
    )
    if rc != 0:
        return 0, num_experts
    ptr = sub.data_ptr()
    try:
        tiles._inkling_w2_v5_plan = (ptr, max_work, num_experts)  # type: ignore[attr-defined]
    except Exception:
        pass
    return ptr, max_work


def _get_plan_buf(need_i32: int, device: torch.device) -> torch.Tensor:
    buf = _PLAN_CACHE.get(device)
    if buf is None or buf.numel() < need_i32:
        buf = torch.empty(need_i32, dtype=torch.int32, device=device)
        _PLAN_CACHE[device] = buf
    return buf[:need_i32]


def _get_m128_work_buf(need_i32: int, device: torch.device) -> torch.Tensor:
    buf = _WORK_M128_CACHE.get(device)
    if buf is None or buf.numel() < need_i32:
        buf = torch.empty(need_i32, dtype=torch.int32, device=device)
        _WORK_M128_CACHE[device] = buf
    return buf[:need_i32]


def _get_y_situ_buf(rows: int, cols: int, device: torch.device) -> torch.Tensor:
    if SHARED_PF_SCRATCH:
        return torch.empty((rows, cols), dtype=torch.bfloat16, device=device)
    need = rows * cols
    buf = _Y_SITU_CACHE.get(device)
    if buf is None or buf.numel() < need:
        buf = torch.empty(need, dtype=torch.bfloat16, device=device)
        _Y_SITU_CACHE[device] = buf
    return buf[:need].view(rows, cols)


def _get_pf_moe_scratch_view(
    device: torch.device, tag: str, shape: tuple[int, ...], dtype: torch.dtype
) -> torch.Tensor:
    """High-water-mark per-device scratch view (strictly compute_stream-local, fresh Tensor wrapper!)."""
    elems = 1
    for d in shape:
        elems *= int(d)
    nbytes = max(elems * torch.empty((), dtype=dtype).element_size(), 16)
    key = (device, tag)
    buf = _PF_MOE_SCRATCH_CACHE.get(key)
    if buf is None or buf.numel() < nbytes:
        buf = torch.empty(nbytes, dtype=torch.uint8, device=device)
        _PF_MOE_SCRATCH_CACHE[key] = buf
    return buf[: elems * torch.empty((), dtype=dtype).element_size()].view(dtype).view(shape)


def get_pf_route_scratch(
    device: torch.device,
    n: int,
    rows: int,
    max_ctas: int,
    qrows: int,
    hidden: int,
    nvfp4: bool,
):
    """Reusable persistent scratch tensors for `ops_moe.route()` (eliminates ~71.7K `torch.empty`s @ 1M)."""
    src2dst = _get_pf_moe_scratch_view(device, "rt_src2dst", (n,), torch.int32)
    gammas = _get_pf_moe_scratch_view(device, "rt_gammas", (rows,), torch.float32)
    routed = _get_pf_moe_scratch_view(device, "rt_routed", (n,), torch.int32)
    tiles = _get_pf_moe_scratch_view(device, "rt_tiles", (2 * max_ctas + 3,), torch.int32)
    ids = _get_pf_moe_scratch_view(device, "rt_ids", (n,), torch.int32)
    ranks = _get_pf_moe_scratch_view(device, "rt_ranks", (n,), torch.int32)
    weights = _get_pf_moe_scratch_view(device, "rt_weights", (n,), torch.float32)
    if nvfp4:
        quantized = (
            _get_pf_moe_scratch_view(device, "rt_qv_nv4", (qrows, hidden // 2), torch.uint8),
            _get_pf_moe_scratch_view(device, "rt_qs_nv4", (qrows, hidden // 16), torch.float8_e4m3fn),
            _get_pf_moe_scratch_view(device, "rt_qrs_nv4", (rows,), torch.float32),
        )
    else:
        quantized = (
            _get_pf_moe_scratch_view(device, "rt_qv_mx8", (qrows, hidden), torch.float8_e4m3fn),
            _get_pf_moe_scratch_view(device, "rt_qs_mx8", (qrows, hidden // 32), torch.uint8),
            None,
        )
    return src2dst, gammas, routed, tiles, ids, ranks, weights, quantized


def get_pf_route_starts_r2t(device: torch.device, local_pow2: int, rows: int, need_r2t: bool):
    starts = _get_pf_moe_scratch_view(device, "rt_starts", (3 * local_pow2,), torch.int32)
    r2t = _get_pf_moe_scratch_view(device, "rt_r2t", (rows,), torch.int32) if need_r2t else None
    return starts, r2t


def route_tokens_place_v6(
    partials: torch.Tensor,
    bias: torch.Tensor,
    expert_bounds: torch.Tensor,
    is_padding: torch.Tensor | None,
    counts: torch.Tensor,
    counters: torch.Tensor,
    ids: torch.Tensor,
    ranks: torch.Tensor,
    weights: torch.Tensor,
    starts: torch.Tensor,
    tiles: torch.Tensor,
    tokens: int,
    route_scale: float,
    num_local: int,
    local_pow2: int,
    tile: int,
    max_ctas: int,
) -> bool:
    """Native SM121a 8-tok/CTA fused route+place (`pf_route_tokens_place_v6`, strictly `T > 256`, `!capturing`)."""
    if not PF_ROUTE_V6 or tokens <= 256 or tile < 64 or (tile & (tile - 1)) != 0:
        return False
    if num_local <= 0 or num_local > 384 or local_pow2 < num_local or local_pow2 > 512:
        return False
    if partials.dtype not in (torch.bfloat16, torch.float32) or not partials.is_contiguous():
        return False
    if bias.dtype != torch.float32 or not bias.is_contiguous():
        return False
    if is_padding is not None and (is_padding.element_size() != 1 or not is_padding.is_contiguous()):
        return False
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return False
    v6_lib = _get_pf_nvfp4_w13_v6_lib()
    if v6_lib is None:
        return False
    st = int(torch.cuda.current_stream(partials.device).cuda_stream)
    rc = v6_lib.pf_route_tokens_place_v6(
        partials.data_ptr(),
        bias.data_ptr(),
        expert_bounds.data_ptr(),
        is_padding.data_ptr() if is_padding is not None else 0,
        counts.data_ptr(),
        counters.data_ptr(),
        ids.data_ptr(),
        ranks.data_ptr(),
        weights.data_ptr(),
        starts.data_ptr(),
        tiles.data_ptr(),
        int(tokens),
        float(route_scale),
        int(num_local),
        int(local_pow2),
        int(tile),
        int(max_ctas),
        1 if partials.dtype == torch.bfloat16 else 0,
        st,
    )
    return rc == 0


def _get_v6_slot_and_state(device: torch.device, v6_lib: ctypes.CDLL) -> tuple[torch.Tensor, torch.Tensor]:
    pair = _V6_Y_SLOT_CACHE.get(device)
    if pair is None:
        slot_b = int(v6_lib.pf_nvfp4_w13_v6_slot_bytes())
        state_b = int(v6_lib.pf_nvfp4_w13_v6_state_bytes())
        y_slot = torch.empty(slot_b, dtype=torch.uint8, device=device)
        s_state = torch.zeros(state_b, dtype=torch.uint8, device=device)
        pair = (y_slot, s_state)
        _V6_Y_SLOT_CACHE[device] = pair
    return pair


def _get_mx7_v10_slice_state(device: torch.device) -> torch.Tensor:
    st_buf = _MX7_V10_SLICE_STATE_CACHE.get(device)
    if st_buf is None:
        st_buf = torch.zeros(512, dtype=torch.uint32, device=device)
        _MX7_V10_SLICE_STATE_CACHE[device] = st_buf
    return st_buf


def _get_or_build_mx7_v10_plan(
    tiles: torch.Tensor, plan, num_experts: int, mx7_v10_lib: ctypes.CDLL, stream: int
) -> tuple[int, int]:
    if not LPT_PLAN or num_experts > 1024:
        return 0, num_experts
    cached = getattr(tiles, "_inkling_mx7_v10_plan", None)
    if cached is not None and cached[2] == num_experts:
        return cached[0], cached[1]
    eff_split = ((PF_MX7_V10_SPLIT_M + 63) // 64) * 64 if PF_MX7_V10_SPLIT_M > 0 else 0
    max_work = (
        num_experts + (plan.max_ctas * plan.tile) // eff_split + 1
        if eff_split > 0
        else num_experts
    )
    need_i32 = 3 * max_work + 1
    buf = _MX7_V10_PLAN_CACHE.get(tiles.device)
    if buf is None or buf.numel() < need_i32:
        buf = torch.empty(need_i32, dtype=torch.int32, device=tiles.device)
        _MX7_V10_PLAN_CACHE[tiles.device] = buf
    sub = buf[:need_i32]
    rc = mx7_v10_lib.pf_mx7_build_plan_v10(
        tiles.data_ptr(), sub.data_ptr(),
        plan.max_ctas, plan.tile, num_experts, eff_split, max_work, stream,
    )
    if rc != 0:
        return 0, num_experts
    ptr = sub.data_ptr()
    try:
        tiles._inkling_mx7_v10_plan = (ptr, max_work, num_experts)  # type: ignore[attr-defined]
    except Exception:
        pass
    return ptr, max_work


def release_unused_scratch() -> int:
    """Release cached `_Y_SITU_CACHE` and optional MoE scratch buffers."""
    freed = sum(int(t.numel() * t.element_size()) for t in _Y_SITU_CACHE.values())
    _Y_SITU_CACHE.clear()
    freed += sum(int(t.numel() * t.element_size()) for t in _PF_MOE_SCRATCH_CACHE.values())
    _PF_MOE_SCRATCH_CACHE.clear()
    return freed


def _get_or_build_m128_worklist(
    tiles: torch.Tensor, plan, num_experts: int, v4b_lib: ctypes.CDLL, stream: int
) -> tuple[int, int]:
    cached = getattr(tiles, "_inkling_m128_work", None)
    if cached is not None and cached[2] == num_experts:
        return cached[0], cached[1]
    max_m128 = num_experts + (plan.max_ctas * plan.tile) // 128 + 1
    buf = _get_m128_work_buf(4 * max_m128 + 1, tiles.device)
    rc = v4b_lib.pf_nvfp4_build_m128_worklist(
        tiles.data_ptr(), buf.data_ptr(),
        plan.max_ctas, plan.tile, num_experts, max_m128, stream,
    )
    assert rc == 0, f"pf_nvfp4_build_m128_worklist CUDA error {rc}"
    ptr = buf.data_ptr()
    try:
        tiles._inkling_m128_work = (ptr, max_m128, num_experts)  # type: ignore[attr-defined]
    except Exception:
        pass
    return ptr, max_m128


def _get_or_build_mx7_plan(
    tiles: torch.Tensor, plan, num_experts: int, mx7_v4_lib: ctypes.CDLL, stream: int
) -> tuple[int, int]:
    if not LPT_PLAN or num_experts > 1024:
        return 0, num_experts
    cached = getattr(tiles, "_inkling_exp_plan", None)
    if cached is not None and cached[2] == num_experts:
        return cached[0], cached[1]
    eff_split = ((SPLIT_M + 255) // 256) * 256 if SPLIT_M > 0 else 0
    max_work = (num_experts + (plan.max_ctas * plan.tile) // eff_split + 1) if eff_split > 0 else num_experts
    buf = _get_plan_buf(3 * max_work + 1, tiles.device)
    rc = mx7_v4_lib.pf_mx7_build_expert_plan(
        tiles.data_ptr(), buf.data_ptr(),
        plan.max_ctas, plan.tile, num_experts, eff_split, max_work, stream,
    )
    if rc != 0:
        return 0, num_experts
    ptr = buf.data_ptr()
    try:
        tiles._inkling_exp_plan = (ptr, max_work, num_experts)  # type: ignore[attr-defined]
    except Exception:
        pass
    return ptr, max_work


def is_w13_situ_v4_active(plan, tokens: int) -> bool:
    """Strictly gate fused W13+SiTU v4/v6/v10 fast paths to long eager prefill chunks (never decode!)."""
    if not ENABLED:
        return False
    if not (
        ((PF_MX7_V4 or PF_MX7_V10) and PF_MX7_V4_FUSE_SITU)
        or PF_NVFP4_W13_V6
        or (PF_NVFP4_W13_V4B and PF_NVFP4_W13_V4B_FUSE_SITU)
    ):
        return False
    if plan.tile < 64 or plan.tile % 64 != 0 or tokens < MIN_TOKENS:
        return False
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return False
    return True


def w13_situ_pf_v4(
    quantized,
    w13_weight: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    ones: torch.Tensor,
    gammas: torch.Tensor,
    tiles: torch.Tensor,
    plan,
    rows_tuple: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    gate_cap: float,
    up_cap: float,
    w2_weight_amax: torch.Tensor | None,
    w2_weight_denominator: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] | None:
    """Fused prefill-only W13 + exact SiTU + W2-input quantization (strictly plan.tile >= 64!).

    - MXFP8 (`PF_MX7_V4=1, PF_MX7_V4_FUSE_SITU=1`): 1 kernel (0 gate_up DRAM round-trip!),
      writes `(values [R,288] e4m3fn, scales [R,12] u8 rowmaj, None)`.
    - NVFP4 (`PF_NVFP4_W13_V4B=1, PF_NVFP4_W13_V4B_FUSE_SITU=1`): fused LPT BM=128 W13+SiTU
      (`Y_situ [R,288]` bf16, -75.5 MB DRAM) + 1-warp/row parallel `pf_nvfp4_quant_288`,
      writes `(values [R,144] u8, scales [R,20] e4m3fn rowmaj, row_scales [R] f32)`.
    Returns None cleanly if disabled or unsupported so caller falls back to standard path.
    """
    if gate_cap != 4.0 or up_cap != 25.0 or plan.tile < 64 or plan.tile % 64 != 0:
        return None
    act, act_scale, row_scale = quantized
    nvfp4 = w13_weight.dtype == torch.uint8
    pack = 2 if nvfp4 else 1
    num_experts = int(w13_weight.shape[0])
    n = int(w13_weight.shape[1])
    k = int(w13_weight.shape[2]) * pack
    if not (n == 576 and k == 4608):
        return None

    r2t = getattr(act, "_inkling_row2tok", None)
    use_r2t = 1 if r2t is not None else 0
    r2t_ptr = r2t.data_ptr() if use_r2t else tiles.data_ptr()
    as_rm = 1 if bool(getattr(act_scale, "_inkling_rowmaj", False)) else 0
    dev = act.device
    st = int(torch.cuda.current_stream(dev).cuda_stream)

    if not nvfp4:
        if not ((PF_MX7_V4 or PF_MX7_V10) and PF_MX7_V4_FUSE_SITU):
            return None
        mx7 = getattr(w13_weight, "_inkling_mx7", None)
        if not isinstance(mx7, tuple):
            return None
        mx7_v4_lib = _get_pf_mx7_v4_lib()
        if mx7_v4_lib is None:
            return None
        slabs, ws3, flips, f_off, base_ws = mx7
        p_ptr, max_w = _get_or_build_mx7_plan(tiles, plan, num_experts, mx7_v4_lib, st)
        if PF_MOE_SCRATCH_REUSE:
            values = _get_pf_moe_scratch_view(dev, "w13_val_mx8", (plan.rows, 288), torch.float8_e4m3fn)
            scales = _get_pf_moe_scratch_view(dev, "w13_sc_mx8", (plan.rows, 12), torch.uint8)
        else:
            values = torch.empty((plan.rows, 288), dtype=torch.float8_e4m3fn, device=dev)
            scales = torch.empty((plan.rows, 12), dtype=torch.uint8, device=dev)
        scales._inkling_rowmaj = True  # type: ignore[attr-defined]
        bm_mx = 256 if (PF_MX7_V4_BM == 256 and plan.tile >= 128) else 128
        rc = mx7_v4_lib.pf_mx7_w13_prefill_fused_situ(
            act.data_ptr(), act_scale.data_ptr(), r2t_ptr, gammas.data_ptr(),
            slabs.data_ptr(), ws3.data_ptr(), flips.data_ptr(), f_off.data_ptr(),
            ones.data_ptr(), tiles.data_ptr(), p_ptr,
            values.data_ptr(), scales.data_ptr(),
            plan.max_ctas, plan.tile, num_experts, max_w, bm_mx, int(base_ws),
            use_r2t, as_rm, 1, 12, PF_MX7_V4_W_EVICT, st,
        )
        assert rc == 0, f"pf_mx7_w13_prefill_fused_situ CUDA error {rc}"
        return values, scales, None

    if row_scale is None or w2_weight_amax is None or w2_weight_denominator is None:
        return None
    ws5 = getattr(w13_weight, "_inkling_nvfp4_ws5", None)
    if ws5 is None or ws5.get("proj") != 0:
        return None
    if PF_NVFP4_W13_V6:
        v6_lib = _get_pf_nvfp4_w13_v6_lib()
        if v6_lib is not None:
            y_slot, slot_state = _get_v6_slot_and_state(dev, v6_lib)
            max_m128 = num_experts + (plan.max_ctas * plan.tile) // 128 + 1
            work_buf = _get_m128_work_buf(4 * max_m128 + 1, dev)
            eff_split = ((PF_NVFP4_W2_V5_SPLIT_M + 63) // 64) * 64 if PF_NVFP4_W2_V5_SPLIT_M > 0 else 0
            max_w2_work = (
                num_experts + (plan.max_ctas * plan.tile) // eff_split + 1
                if eff_split > 0
                else num_experts
            )
            need_w2_i32 = 3 * max_w2_work + 1
            w2_buf = _W2_V5_PLAN_CACHE.get(dev)
            if w2_buf is None or w2_buf.numel() < need_w2_i32:
                w2_buf = torch.empty(need_w2_i32, dtype=torch.int32, device=dev)
                _W2_V5_PLAN_CACHE[dev] = w2_buf
            w2_sub = w2_buf[:need_w2_i32]
            rc_plan = v6_lib.pf_nvfp4_build_w13_w2_plan_v6(
                tiles.data_ptr(),
                work_buf.data_ptr(),
                w2_sub.data_ptr(),
                slot_state.data_ptr(),
                plan.max_ctas, plan.tile, num_experts, max_m128,
                eff_split, max_w2_work, st,
            )
            if rc_plan == 0:
                try:
                    tiles._inkling_m128_work = (work_buf.data_ptr(), max_m128, num_experts)  # type: ignore[attr-defined]
                    tiles._inkling_w2_v5_plan = (w2_sub.data_ptr(), max_w2_work, num_experts)  # type: ignore[attr-defined]
                except Exception:
                    pass
                if PF_MOE_SCRATCH_REUSE:
                    values = _get_pf_moe_scratch_view(dev, "w13_val_nv4", (plan.rows, 144), torch.uint8)
                    scales = _get_pf_moe_scratch_view(dev, "w13_sc_nv4", (plan.rows, 20), torch.float8_e4m3fn)
                    row_scales = _get_pf_moe_scratch_view(dev, "w13_rs_nv4", (plan.rows,), torch.float32)
                else:
                    values = torch.empty((plan.rows, 144), dtype=torch.uint8, device=dev)
                    scales = torch.empty((plan.rows, 20), dtype=torch.float8_e4m3fn, device=dev)
                    row_scales = torch.empty(plan.rows, dtype=torch.float32, device=dev)
                scales._inkling_rowmaj = True  # type: ignore[attr-defined]
                rc6 = v6_lib.pf_nvfp4_w13_situ_quant_v6(
                    act.data_ptr(), act_scale.data_ptr(), row_scale.data_ptr(), r2t_ptr,
                    ws5["slabs"].data_ptr(), ws5["ws_base"].data_ptr(),
                    ws5["p_off"].data_ptr(), ws5["patches"].data_ptr(),
                    ones.data_ptr(), gammas.data_ptr(), work_buf.data_ptr(),
                    y_slot.data_ptr(), slot_state.data_ptr(),
                    values.data_ptr(), scales.data_ptr(), row_scales.data_ptr(),
                    w2_weight_amax.data_ptr(), w2_weight_denominator.data_ptr(),
                    max_m128, plan.tile, use_r2t, as_rm,
                    1 if ws5.get("has_patches", True) else 0,
                    PF_NVFP4_W13_V6_W_EVICT, st,
                )
                if rc6 == 0:
                    return values, scales, row_scales
    if not (PF_NVFP4_W13_V4B and PF_NVFP4_W13_V4B_FUSE_SITU):
        return None
    v4b_lib = _get_pf_nvfp4_w13_v4b_lib()
    if v4b_lib is None:
        return None

    w_ptr, max_m128 = _get_or_build_m128_worklist(tiles, plan, num_experts, v4b_lib, st)
    y_situ = _get_y_situ_buf(plan.rows, 288, dev)
    rc1 = v4b_lib.pf_nvfp4_w13_ws5_v4b_fused_situ(
        act.data_ptr(), act_scale.data_ptr(), row_scale.data_ptr(), r2t_ptr,
        ws5["slabs"].data_ptr(), ws5["ws_base"].data_ptr(),
        ws5["p_off"].data_ptr(), ws5["patches"].data_ptr(),
        ones.data_ptr(), gammas.data_ptr(), w_ptr, y_situ.data_ptr(),
        max_m128, plan.tile, use_r2t, as_rm,
        1 if ws5.get("has_patches", True) else 0, PF_NVFP4_W13_V4B_W_EVICT, st,
    )
    assert rc1 == 0, f"pf_nvfp4_w13_ws5_v4b_fused_situ CUDA error {rc1}"

    routed, num_routed, cta_expert = rows_tuple
    values = torch.empty((plan.rows, 144), dtype=torch.uint8, device=dev)
    scales = torch.empty((plan.rows, 20), dtype=torch.float8_e4m3fn, device=dev)
    scales._inkling_rowmaj = True  # type: ignore[attr-defined]
    row_scales = torch.empty(plan.rows, dtype=torch.float32, device=dev)
    rc2 = v4b_lib.pf_nvfp4_quant_288(
        y_situ.data_ptr(), values.data_ptr(), scales.data_ptr(), row_scales.data_ptr(),
        routed.data_ptr(), num_routed.data_ptr(), cta_expert.data_ptr(),
        w2_weight_amax.data_ptr(), w2_weight_denominator.data_ptr(),
        int(routed.numel()), plan.tile, 0, st,
    )
    assert rc2 == 0, f"pf_nvfp4_quant_288 CUDA error {rc2}"
    return values, scales, row_scales


def get_or_build_expert_plan(
    tiles: torch.Tensor,
    plan,
    num_experts: int,
    lib: ctypes.CDLL | None,
    stream: int,
    *,
    split_m: int | None = None,
) -> tuple[int, int]:
    """Build (once per layer) the 1-block LPT expert work plan for persistent W13 & W2.

    Returns ``(plan_buf_ptr, max_work)``; returns ``(0, num_experts)`` if disabled or unavailable.
    """
    if not LPT_PLAN or lib is None or not hasattr(lib, "pf_build_expert_plan") or num_experts > 1024:
        return 0, num_experts
    cached = getattr(tiles, "_inkling_exp_plan", None)
    if cached is not None and cached[2] == num_experts:
        return cached[0], cached[1]
    sm = SPLIT_M if split_m is None else split_m
    eff_split = ((sm + 255) // 256) * 256 if sm > 0 else 0
    max_work = (num_experts + (plan.max_ctas * plan.tile) // eff_split + 1) if eff_split > 0 else num_experts
    buf = _get_plan_buf(3 * max_work + 1, tiles.device)
    rc = lib.pf_build_expert_plan(
        tiles.data_ptr(), buf.data_ptr(),
        plan.max_ctas, plan.tile, num_experts, eff_split, max_work, stream,
    )
    if rc != 0:
        return 0, num_experts
    ptr = buf.data_ptr()
    try:
        tiles._inkling_exp_plan = (ptr, max_work, num_experts)  # type: ignore[attr-defined]
    except Exception:
        pass
    return ptr, max_work


def _get_zero_stream(device: torch.device) -> torch.cuda.Stream:
    zs = _ZERO_STREAMS.get(device)
    if zs is None:
        zs = torch.cuda.Stream(device=device, priority=0)
        _ZERO_STREAMS[device] = zs
    return zs


def zero_bf16_out(
    out: torch.Tensor,
    *,
    mode: str | None = None,
    lib: ctypes.CDLL | None = None,
    stream: torch.cuda.Stream | None = None,
) -> None:
    """Zero a contiguous `[T, H]` bf16 output tensor without polluting L2 cache."""
    m = ZERO_MODE if mode is None else mode
    lib = lib or get_lib()
    st_obj = stream if stream is not None else torch.cuda.current_stream(out.device)
    st = int(st_obj.cuda_stream)
    if m == "memset" and lib is not None and hasattr(lib, "pf_memset_zero"):
        rc = lib.pf_memset_zero(out.data_ptr(), out.numel() * out.element_size(), st)
        if rc == 0:
            return
    elif m == "cs" and lib is not None and hasattr(lib, "pf_zero_cs_bf16"):
        rc = lib.pf_zero_cs_bf16(out.data_ptr(), out.numel(), st)
        if rc == 0:
            return
    with torch.cuda.stream(st_obj):
        out.zero_()


def alloc_and_async_zero_out(
    tokens: int,
    hidden: int,
    device: torch.device,
    compute_stream: torch.cuda.Stream | None = None,
    lib: ctypes.CDLL | None = None,
) -> tuple[torch.Tensor, torch.cuda.Event]:
    """Allocate `[tokens, hidden]` bf16 on `compute_stream` and asynchronously zero it on
    a dedicated priority=0 `zero_stream` so zeroing overlaps `route()` + `W13` and does
    not evict `situ_quantize_288`'s `w2_act` from L2 right before W2!"""
    cs = compute_stream if compute_stream is not None else torch.cuda.current_stream(device)
    zs = _get_zero_stream(device)
    out = torch.empty((tokens, hidden), dtype=torch.bfloat16, device=device)
    ev_alloc = torch.cuda.Event()
    ev_alloc.record(cs)
    zs.wait_event(ev_alloc)
    out.record_stream(zs)
    zero_bf16_out(out, lib=lib, stream=zs)
    ev_done = torch.cuda.Event()
    ev_done.record(zs)
    return out, ev_done


def _get_y_fp32(tokens: int, hidden: int, device: torch.device) -> torch.Tensor:
    need = tokens * hidden
    buf = _Y_FP32_CACHE.get(device)
    if buf is None or buf.numel() < need:
        buf = torch.zeros(need, dtype=torch.float32, device=device)
        _Y_FP32_CACHE[device] = buf
    return buf[:need].view(tokens, hidden)


def _get_r2t_buf(rows: int, device: torch.device) -> torch.Tensor:
    buf = _R2T_CACHE.get(device)
    if buf is None or buf.numel() < rows:
        buf = torch.empty(rows, dtype=torch.int32, device=device)
        _R2T_CACHE[device] = buf
    return buf[:rows]


def build_row2tok(
    src2dst: torch.Tensor,
    rows: int,
    topk: int = 8,
    out: torch.Tensor | None = None,
    lib: ctypes.CDLL | None = None,
) -> torch.Tensor:
    lib = lib or get_lib()
    assert lib is not None, "libpf_moe_native_sm121.so not loaded"
    tokens = src2dst.numel() // topk
    r2t = out if out is not None else _get_r2t_buf(rows, src2dst.device)
    st = int(torch.cuda.current_stream(src2dst.device).cuda_stream)
    rc = lib.pf_build_row2tok(src2dst.data_ptr(), r2t.data_ptr(), tokens, topk, st)
    assert rc == 0, f"pf_build_row2tok CUDA error {rc}"
    return r2t


def bmm_pf_native(
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
    topk_weights: torch.Tensor | None = None,
    as_row_major: bool = False,
    bm: int | None = None,
    w_evict_mode: int | None = None,
    persistent: bool | None = None,
    out: torch.Tensor | None = None,
    lib: ctypes.CDLL | None = None,
) -> torch.Tensor | None:
    """Drop-in native SM121a prefill GEMM (unfused output [plan.rows, N] bf16).

    Returns None (so caller falls back cleanly) if weight layout or shape is unsupported.
    """
    lib = lib or get_lib()
    if lib is None or plan.tile < 64 or plan.tile % 64 != 0:
        return None

    nvfp4 = weight.dtype == torch.uint8
    pack = 2 if nvfp4 else 1
    num_experts = int(weight.shape[0])
    n = weight.shape[1]
    k = weight.shape[2] * pack
    if not ((projection == 0 and n == 576 and k == 4608) or (projection == 1 and n == 4608 and k == 288)):
        return None

    use_persist = (PERSISTENT_W13 if projection == 0 else PERSISTENT) if persistent is None else persistent
    bm_sel = select_bm(plan, bm)
    bm_v4 = select_bm_v4(plan, bm)
    w_evict = W_EVICT_MODE if w_evict_mode is None else w_evict_mode
    w_evict_v4 = select_v4_w_evict(w_evict_mode)
    if out is None:
        out = torch.empty((plan.rows, n), dtype=torch.bfloat16, device=act.device)
    st = int(torch.cuda.current_stream(act.device).cuda_stream)

    if nvfp4:
        assert row_scale is not None
        ws5 = getattr(weight, "_inkling_nvfp4_ws5", None)
        if projection == 0:
            use_r2t = 1 if row2tok is not None else 0
            r2t_ptr = row2tok.data_ptr() if use_r2t else tiles.data_ptr()
            if ws5 is not None and ws5.get("proj") == 0:
                v4b_lib = _get_pf_nvfp4_w13_v4b_lib() if PF_NVFP4_W13_V4B else None
                if v4b_lib is not None:
                    w_ptr, max_m128 = _get_or_build_m128_worklist(tiles, plan, num_experts, v4b_lib, st)
                    rc = v4b_lib.pf_nvfp4_w13_ws5_v4b_unfused(
                        act.data_ptr(), act_scale.data_ptr(), row_scale.data_ptr(), r2t_ptr,
                        ws5["slabs"].data_ptr(), ws5["ws_base"].data_ptr(),
                        ws5["p_off"].data_ptr(), ws5["patches"].data_ptr(),
                        ones.data_ptr(), w_ptr, out.data_ptr(),
                        max_m128, plan.tile, use_r2t, 1 if as_row_major else 0,
                        1 if ws5.get("has_patches", True) else 0, PF_NVFP4_W13_V4B_W_EVICT, st,
                    )
                elif use_persist and hasattr(lib, "pf_nvfp4_w13_ws5_native_v4"):
                    p_ptr, max_w = get_or_build_expert_plan(tiles, plan, num_experts, lib, st)
                    rc = lib.pf_nvfp4_w13_ws5_native_v4(
                        act.data_ptr(), act_scale.data_ptr(), row_scale.data_ptr(), r2t_ptr,
                        ws5["slabs"].data_ptr(), ws5["ws_base"].data_ptr(),
                        ws5["p_off"].data_ptr(), ws5["patches"].data_ptr(),
                        ones.data_ptr(), tiles.data_ptr(), p_ptr, out.data_ptr(),
                        plan.max_ctas, plan.tile, num_experts, max_w, bm_v4, use_r2t,
                        1 if as_row_major else 0, 1 if ws5.get("has_patches", True) else 0,
                        w_evict_v4, st,
                    )
                elif hasattr(lib, "pf_nvfp4_w13_ws5_native_v3"):
                    rc = lib.pf_nvfp4_w13_ws5_native_v3(
                        act.data_ptr(), act_scale.data_ptr(), row_scale.data_ptr(), r2t_ptr,
                        ws5["slabs"].data_ptr(), ws5["ws_base"].data_ptr(),
                        ws5["p_off"].data_ptr(), ws5["patches"].data_ptr(),
                        ones.data_ptr(), tiles.data_ptr(), out.data_ptr(),
                        plan.max_ctas, plan.tile, bm_sel, use_r2t,
                        1 if as_row_major else 0, 1 if ws5.get("has_patches", True) else 0,
                        w_evict, st,
                    )
                else:
                    rc = lib.pf_nvfp4_w13_ws5_native(
                        act.data_ptr(), act_scale.data_ptr(), row_scale.data_ptr(), r2t_ptr,
                        ws5["slabs"].data_ptr(), ws5["ws_base"].data_ptr(),
                        ws5["p_off"].data_ptr(), ws5["patches"].data_ptr(),
                        ones.data_ptr(), tiles.data_ptr(), out.data_ptr(),
                        plan.max_ctas, plan.tile, bm_sel, use_r2t,
                        1 if as_row_major else 0, 1 if ws5.get("has_patches", True) else 0, st,
                    )
            elif (
                getattr(weight, "_inkling_panel", None) == (32, 128)
                and getattr(weight_scale, "_inkling_panel", None) == (32, 16)
                and weight.data.untyped_storage().size() > 0
                and weight_scale.data.untyped_storage().size() > 0
            ):
                ws6 = 1 if bool(getattr(weight_scale, "_inkling_ws6", False)) else 0
                if use_persist and hasattr(lib, "pf_nvfp4_w13_panel_native_v4"):
                    p_ptr, max_w = get_or_build_expert_plan(tiles, plan, num_experts, lib, st)
                    rc = lib.pf_nvfp4_w13_panel_native_v4(
                        act.data_ptr(), act_scale.data_ptr(), row_scale.data_ptr(), r2t_ptr,
                        weight.data_ptr(), weight_scale.data_ptr(),
                        ones.data_ptr(), tiles.data_ptr(), p_ptr, out.data_ptr(),
                        plan.max_ctas, plan.tile, num_experts, max_w, bm_v4, use_r2t,
                        1 if as_row_major else 0, ws6, w_evict_v4, st,
                    )
                else:
                    rc = lib.pf_nvfp4_w13_panel_native(
                        act.data_ptr(), act_scale.data_ptr(), row_scale.data_ptr(), r2t_ptr,
                        weight.data_ptr(), weight_scale.data_ptr(),
                        ones.data_ptr(), tiles.data_ptr(), out.data_ptr(),
                        plan.max_ctas, plan.tile, bm_sel, use_r2t,
                        1 if as_row_major else 0, ws6, st,
                    )
            else:
                return None
        else:
            as_stride = int(act_scale.stride(0)) if act_scale.ndim >= 2 else 18
            tw_ptr = topk_weights.data_ptr() if topk_weights is not None else 0
            w2_v5_lib = _get_pf_nvfp4_w2_v5_lib() if PF_NVFP4_W2_V5 else None
            if ws5 is not None and ws5.get("proj") == 1:
                if w2_v5_lib is not None:
                    p_ptr, max_w = _get_or_build_w2_v5_plan(tiles, plan, num_experts, w2_v5_lib, st)
                    s_state = _get_w2_v5_slice_state(act.device)
                    eff_ev = PF_NVFP4_W2_V5_W_EVICT if w_evict_mode is None else w_evict_mode
                    rc = w2_v5_lib.pf_nvfp4_w2_ws5_v5_unfused(
                        act.data_ptr(), act_scale.data_ptr(), row_scale.data_ptr(), tw_ptr,
                        ws5["w_pack"].data_ptr(), ws5["ws5_pack"].data_ptr(), ws5["ws_base"].data_ptr(),
                        ws5["p_off"].data_ptr(), ws5["patches"].data_ptr(),
                        ones.data_ptr(), tiles.data_ptr(), p_ptr, s_state.data_ptr(), out.data_ptr(),
                        plan.max_ctas, plan.tile, num_experts, max_w,
                        1 if as_row_major else 0, as_stride,
                        1 if ws5.get("has_patches", True) else 0, eff_ev, st,
                    )
                elif use_persist and hasattr(lib, "pf_nvfp4_w2_ws5_native_v4"):
                    p_ptr, max_w = get_or_build_expert_plan(tiles, plan, num_experts, lib, st)
                    rc = lib.pf_nvfp4_w2_ws5_native_v4(
                        act.data_ptr(), act_scale.data_ptr(), row_scale.data_ptr(), tiles.data_ptr(),
                        ws5["w_pack"].data_ptr(), ws5["ws5_pack"].data_ptr(), ws5["ws_base"].data_ptr(),
                        ws5["p_off"].data_ptr(), ws5["patches"].data_ptr(),
                        ones.data_ptr(), tiles.data_ptr(), p_ptr, out.data_ptr(),
                        plan.max_ctas, plan.tile, num_experts, max_w,
                        1 if as_row_major else 0, as_stride,
                        1 if ws5.get("has_patches", True) else 0, 0, w_evict_v4, 6, 0, 0, st,
                    )
                elif hasattr(lib, "pf_nvfp4_w2_ws5_native_v3"):
                    rc = lib.pf_nvfp4_w2_ws5_native_v3(
                        act.data_ptr(), act_scale.data_ptr(), row_scale.data_ptr(), tiles.data_ptr(),
                        ws5["w_pack"].data_ptr(), ws5["ws5_pack"].data_ptr(), ws5["ws_base"].data_ptr(),
                        ws5["p_off"].data_ptr(), ws5["patches"].data_ptr(),
                        ones.data_ptr(), tiles.data_ptr(), out.data_ptr(),
                        plan.max_ctas, plan.tile, bm_sel, 1 if as_row_major else 0, as_stride,
                        1 if ws5.get("has_patches", True) else 0, 0, w_evict, 6, st,
                    )
                else:
                    rc = lib.pf_nvfp4_w2_ws5_native(
                        act.data_ptr(), act_scale.data_ptr(), row_scale.data_ptr(), tiles.data_ptr(),
                        ws5["w_pack"].data_ptr(), ws5["ws5_pack"].data_ptr(), ws5["ws_base"].data_ptr(),
                        ws5["p_off"].data_ptr(), ws5["patches"].data_ptr(),
                        ones.data_ptr(), tiles.data_ptr(), out.data_ptr(),
                        plan.max_ctas, plan.tile, bm_sel, 1 if as_row_major else 0, as_stride,
                        1 if ws5.get("has_patches", True) else 0, 0, st,
                    )
            elif (
                getattr(weight, "_inkling_panel", None) == (256, 16)
                and getattr(weight_scale, "_inkling_panel", None) == (256, 2)
                and weight.data.untyped_storage().size() > 0
                and weight_scale.data.untyped_storage().size() > 0
            ):
                if w2_v5_lib is not None:
                    p_ptr, max_w = _get_or_build_w2_v5_plan(tiles, plan, num_experts, w2_v5_lib, st)
                    s_state = _get_w2_v5_slice_state(act.device)
                    eff_ev = PF_NVFP4_W2_V5_W_EVICT if w_evict_mode is None else w_evict_mode
                    rc = w2_v5_lib.pf_nvfp4_w2_panel_v5_combine(
                        act.data_ptr(), act_scale.data_ptr(), row_scale.data_ptr(), tw_ptr, tiles.data_ptr(),
                        weight.data_ptr(), weight_scale.data_ptr(),
                        ones.data_ptr(), tiles.data_ptr(), p_ptr, s_state.data_ptr(), out.data_ptr(),
                        plan.max_ctas, plan.tile, num_experts, max_w,
                        1 if as_row_major else 0, as_stride,
                        0, eff_ev, 6, 0, 0, st,
                    )
                elif use_persist and hasattr(lib, "pf_nvfp4_w2_panel_native_v4"):
                    p_ptr, max_w = get_or_build_expert_plan(tiles, plan, num_experts, lib, st)
                    rc = lib.pf_nvfp4_w2_panel_native_v4(
                        act.data_ptr(), act_scale.data_ptr(), row_scale.data_ptr(), tiles.data_ptr(),
                        weight.data_ptr(), weight_scale.data_ptr(),
                        ones.data_ptr(), tiles.data_ptr(), p_ptr, out.data_ptr(),
                        plan.max_ctas, plan.tile, num_experts, max_w,
                        1 if as_row_major else 0, as_stride,
                        0, w_evict_v4, 6, 0, 0, st,
                    )
                elif hasattr(lib, "pf_nvfp4_w2_panel_native_v3"):
                    rc = lib.pf_nvfp4_w2_panel_native_v3(
                        act.data_ptr(), act_scale.data_ptr(), row_scale.data_ptr(), tiles.data_ptr(),
                        weight.data_ptr(), weight_scale.data_ptr(),
                        ones.data_ptr(), tiles.data_ptr(), out.data_ptr(),
                        plan.max_ctas, plan.tile, bm_sel, 1 if as_row_major else 0, as_stride,
                        0, w_evict, 6, st,
                    )
                else:
                    rc = lib.pf_nvfp4_w2_panel_native(
                        act.data_ptr(), act_scale.data_ptr(), row_scale.data_ptr(), tiles.data_ptr(),
                        weight.data_ptr(), weight_scale.data_ptr(),
                        ones.data_ptr(), tiles.data_ptr(), out.data_ptr(),
                        plan.max_ctas, plan.tile, bm_sel, 1 if as_row_major else 0, as_stride,
                        0, st,
                    )
            else:
                return None
        assert rc == 0, f"pf_nvfp4 native CUDA error {rc} (proj={projection})"
        return out

    # MXFP8 path: prefer dedicated v4 prefill MX7 / v6 Expert-Persistent / v5 native on MX7 slabs
    mx7 = getattr(weight, "_inkling_mx7", None)
    if isinstance(mx7, tuple):
        mx7_v10_lib = _get_pf_mx7_v10_lib() if (PF_MX7_V10 and projection == 1) else None
        if mx7_v10_lib is not None:
            slabs, ws3, flips, f_off, base_ws = mx7
            p_ptr, max_w = _get_or_build_mx7_v10_plan(tiles, plan, num_experts, mx7_v10_lib, st)
            s_state = _get_mx7_v10_slice_state(act.device)
            as_stride = int(act_scale.stride(0)) if act_scale.ndim >= 2 else 9
            eff_ev = PF_MX7_V10_W_EVICT if w_evict_mode is None else w_evict_mode
            rc = mx7_v10_lib.pf_mx7_w2_v10_unfused(
                act.data_ptr(), act_scale.data_ptr(),
                slabs.data_ptr(), ws3.data_ptr(), flips.data_ptr(), f_off.data_ptr(),
                ones.data_ptr(), tiles.data_ptr(), p_ptr, s_state.data_ptr(), out.data_ptr(),
                plan.max_ctas, plan.tile, num_experts, max_w, int(base_ws),
                1 if as_row_major else 0, as_stride, eff_ev, st,
            )
            assert rc == 0, f"pf_mx7_w2_v10_unfused CUDA error {rc}"
            return out
        mx7_v4_lib = _get_pf_mx7_v4_lib()
        if mx7_v4_lib is not None:
            slabs, ws3, flips, f_off, base_ws = mx7
            p_ptr, max_w = _get_or_build_mx7_plan(tiles, plan, num_experts, mx7_v4_lib, st)
            if projection == 0:
                use_r2t = 1 if row2tok is not None else 0
                r2t_ptr = row2tok.data_ptr() if use_r2t else tiles.data_ptr()
                bm_mx = 256 if (PF_MX7_V4_BM == 256 and plan.tile >= 128) else 128
                rc = mx7_v4_lib.pf_mx7_w13_prefill_unfused(
                    act.data_ptr(), act_scale.data_ptr(), r2t_ptr,
                    slabs.data_ptr(), ws3.data_ptr(), flips.data_ptr(), f_off.data_ptr(),
                    ones.data_ptr(), tiles.data_ptr(), p_ptr, out.data_ptr(),
                    plan.max_ctas, plan.tile, num_experts, max_w, bm_mx, int(base_ws),
                    use_r2t, 1 if as_row_major else 0, PF_MX7_V4_W_EVICT, st,
                )
            else:
                as_stride = int(act_scale.stride(0)) if act_scale.ndim >= 2 else 9
                rc = mx7_v4_lib.pf_mx7_w2_prefill_persistent(
                    act.data_ptr(), act_scale.data_ptr(), tiles.data_ptr(),
                    slabs.data_ptr(), ws3.data_ptr(), flips.data_ptr(), f_off.data_ptr(),
                    ones.data_ptr(), tiles.data_ptr(), p_ptr, out.data_ptr(),
                    plan.max_ctas, plan.tile, num_experts, max_w, int(base_ws),
                    1 if as_row_major else 0, as_stride, 0, PF_MX7_V4_W_EVICT, 12, 0, 0, st,
                )
            assert rc == 0, f"pf_mx7_v4 prefill CUDA error {rc} (proj={projection})"
            return out
        mx7_v6_lib = _get_mx7_v6_lib() if use_persist else None
        if mx7_v6_lib is not None:
            slabs, ws3, flips, f_off, base_ws = mx7
            p_ptr, max_w = get_or_build_expert_plan(tiles, plan, num_experts, lib, st)
            if projection == 0:
                use_r2t = 1 if row2tok is not None else 0
                r2t_ptr = row2tok.data_ptr() if use_r2t else tiles.data_ptr()
                rc = mx7_v6_lib.mx7_v6_prefill_native_w13(
                    act.data_ptr(), act_scale.data_ptr(), r2t_ptr,
                    slabs.data_ptr(), ws3.data_ptr(), flips.data_ptr(), f_off.data_ptr(),
                    ones.data_ptr(), tiles.data_ptr(), p_ptr, out.data_ptr(),
                    plan.max_ctas, plan.tile, num_experts, max_w, bm_v4, base_ws,
                    use_r2t, 1 if as_row_major else 0, w_evict_v4, st,
                )
            else:
                as_stride = int(act_scale.stride(0)) if act_scale.ndim >= 2 else 9
                rc = mx7_v6_lib.mx7_v6_prefill_native_w2(
                    act.data_ptr(), act_scale.data_ptr(), tiles.data_ptr(),
                    slabs.data_ptr(), ws3.data_ptr(), flips.data_ptr(), f_off.data_ptr(),
                    ones.data_ptr(), tiles.data_ptr(), p_ptr, out.data_ptr(),
                    plan.max_ctas, plan.tile, num_experts, max_w, base_ws,
                    1 if as_row_major else 0, as_stride, 0, w_evict_v4, 6, 0, 0, st,
                )
            assert rc == 0, f"mx7_v6_prefill_native CUDA error {rc} (proj={projection})"
            return out
        mx7_lib = _get_mx7_v5_lib()
        if mx7_lib is not None:
            slabs, ws3, flips, f_off, base_ws = mx7
            if projection == 0:
                use_r2t = 1 if row2tok is not None else 0
                r2t_ptr = row2tok.data_ptr() if use_r2t else tiles.data_ptr()
                rc = mx7_lib.mx7_v5_prefill_native_w13(
                    act.data_ptr(), act_scale.data_ptr(), r2t_ptr,
                    slabs.data_ptr(), ws3.data_ptr(), flips.data_ptr(), f_off.data_ptr(),
                    ones.data_ptr(), tiles.data_ptr(), out.data_ptr(),
                    plan.max_ctas, plan.tile, bm_sel, base_ws,
                    use_r2t, 1 if as_row_major else 0, st,
                )
            else:
                as_stride = int(act_scale.stride(0)) if act_scale.ndim >= 2 else 9
                rc = mx7_lib.mx7_v5_prefill_native_w2(
                    act.data_ptr(), act_scale.data_ptr(), tiles.data_ptr(),
                    slabs.data_ptr(), ws3.data_ptr(), flips.data_ptr(), f_off.data_ptr(),
                    ones.data_ptr(), tiles.data_ptr(), out.data_ptr(),
                    plan.max_ctas, plan.tile, bm_sel, base_ws,
                    1 if as_row_major else 0, as_stride, 0, st,
                )
            assert rc == 0, f"mx7_v5_prefill_native CUDA error {rc} (proj={projection})"
            return out

    # Raw EXPERT_PANEL MXFP8 fallback (or transient MX6-Huff7 GPU panel unpack when MX6_FREE_RAW=1!)
    w_eff, ws_eff = weight, weight_scale
    _mx6_tmp = None
    if weight.data.untyped_storage().size() == 0 or weight_scale.data.untyped_storage().size() == 0:
        if isinstance(getattr(weight, "_inkling_mx6", None), tuple):
            from . import trtllm_bmm as _bmm_mx6
            _mx6_tmp = _bmm_mx6._mx6_unpack_to_panel_tmp(weight, projection)
            if _mx6_tmp is None:
                return None
            w_eff, ws_eff = _mx6_tmp
        else:
            return None
    try:
        if projection == 0:
            if getattr(w_eff, "_inkling_panel", None) != (32, 512) or getattr(ws_eff, "_inkling_panel", None) != (32, 16):
                return None
            use_r2t = 1 if row2tok is not None else 0
            r2t_ptr = row2tok.data_ptr() if use_r2t else tiles.data_ptr()
            if use_persist and hasattr(lib, "pf_mxfp8_w13_native_v4"):
                p_ptr, max_w = get_or_build_expert_plan(tiles, plan, num_experts, lib, st)
                rc = lib.pf_mxfp8_w13_native_v4(
                    act.data_ptr(), act_scale.data_ptr(), r2t_ptr,
                    w_eff.data_ptr(), ws_eff.data_ptr(),
                    ones.data_ptr(), tiles.data_ptr(), p_ptr, out.data_ptr(),
                    plan.max_ctas, plan.tile, num_experts, max_w, bm_v4, use_r2t,
                    1 if as_row_major else 0, w_evict_v4, st,
                )
            else:
                rc = lib.pf_mxfp8_w13_native(
                    act.data_ptr(), act_scale.data_ptr(), r2t_ptr,
                    w_eff.data_ptr(), ws_eff.data_ptr(),
                    ones.data_ptr(), tiles.data_ptr(), out.data_ptr(),
                    plan.max_ctas, plan.tile, bm_sel, use_r2t, 1 if as_row_major else 0, st,
                )
        else:
            if getattr(w_eff, "_inkling_panel", None) != (128, 32) or getattr(ws_eff, "_inkling_panel", None) != (128, 1):
                return None
            as_stride = int(act_scale.stride(0)) if act_scale.ndim >= 2 else 9
            if use_persist and hasattr(lib, "pf_mxfp8_w2_native_v4"):
                p_ptr, max_w = get_or_build_expert_plan(tiles, plan, num_experts, lib, st)
                rc = lib.pf_mxfp8_w2_native_v4(
                    act.data_ptr(), act_scale.data_ptr(), tiles.data_ptr(),
                    w_eff.data_ptr(), ws_eff.data_ptr(),
                    ones.data_ptr(), tiles.data_ptr(), p_ptr, out.data_ptr(),
                    plan.max_ctas, plan.tile, num_experts, max_w,
                    1 if as_row_major else 0, as_stride, 0, w_evict_v4, 6, 0, 0, st,
                )
            elif hasattr(lib, "pf_mxfp8_w2_native_v3"):
                rc = lib.pf_mxfp8_w2_native_v3(
                    act.data_ptr(), act_scale.data_ptr(), tiles.data_ptr(),
                    w_eff.data_ptr(), ws_eff.data_ptr(),
                    ones.data_ptr(), tiles.data_ptr(), out.data_ptr(),
                    plan.max_ctas, plan.tile, bm_sel, 1 if as_row_major else 0, as_stride,
                    0, w_evict, 6, st,
                )
            else:
                rc = lib.pf_mxfp8_w2_native(
                    act.data_ptr(), act_scale.data_ptr(), tiles.data_ptr(),
                    w_eff.data_ptr(), ws_eff.data_ptr(),
                    ones.data_ptr(), tiles.data_ptr(), out.data_ptr(),
                    plan.max_ctas, plan.tile, bm_sel, 1 if as_row_major else 0, as_stride,
                    0, st,
                )
    finally:
        if _mx6_tmp is not None:
            del _mx6_tmp, w_eff, ws_eff
    assert rc == 0, f"pf_mxfp8 native CUDA error {rc} (proj={projection})"
    return out


def w2_combine_pf_native(
    act: torch.Tensor,
    act_scale: torch.Tensor,
    row_scale: torch.Tensor | None,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    ones: torch.Tensor,
    tiles: torch.Tensor,
    plan,
    src2dst: torch.Tensor,
    topk: int = 8,
    out_dtype: torch.dtype = torch.bfloat16,
    *,
    out: torch.Tensor | None = None,
    out_prezeroed: bool = False,
    row2tok: torch.Tensor | None = None,
    topk_weights: torch.Tensor | None = None,
    as_row_major: bool = False,
    combine_mode: int | None = None,
    bm: int | None = None,
    w_evict_mode: int | None = None,
    w2_panels: int | None = None,
    persistent: bool | None = None,
    l2_zero_slice: bool | None = None,
    lib: ctypes.CDLL | None = None,
) -> torch.Tensor | None:
    """Fused SM121a W2 GEMM + L2-resident Combine epilogue -> [T, 4608] bf16.

    Eliminates ``w2_out [plan.rows, 4608]`` (-1.96 GiB VRAM at T=16K) and ``combine()``.
    - ``combine_mode == 1`` (default): nb-outer FP32 atomic into reusable pre-zeroed
      ``[T, 4608]`` fp32 scratch, followed by vectorized ``pf_fp32_to_bf16_zero``.
    - ``combine_mode == 2``: zeroes ``out [T, 4608]`` bf16 (or per-slice in-L2 zeroing
      when ``persistent=True`` and ``l2_zero_slice=True``) and runs nb-outer BF16x2
      atomic directly into ``out``.
    """
    lib = lib or get_lib()
    cm = COMBINE_MODE if combine_mode is None else combine_mode
    if lib is None or cm not in (1, 2) or out_dtype != torch.bfloat16:
        return None
    if plan.tile < 64 or plan.tile % 64 != 0:
        return None

    nvfp4 = weight.dtype == torch.uint8
    pack = 2 if nvfp4 else 1
    num_experts = int(weight.shape[0])
    n = weight.shape[1]
    k = weight.shape[2] * pack
    if not (n == 4608 and k == 288):
        return None

    use_persist = PERSISTENT if persistent is None else persistent
    use_l2_zero = L2_ZERO_SLICE if l2_zero_slice is None else l2_zero_slice
    tokens = src2dst.numel() // topk
    bm_sel = select_bm(plan, bm)
    w_evict = W_EVICT_MODE if w_evict_mode is None else w_evict_mode
    w_evict_v4 = select_v4_w_evict(w_evict_mode)
    device = act.device
    st = int(torch.cuda.current_stream(device).cuda_stream)

    # Determine upfront whether a v5/v4/v6 persistent handler will run (so l2_zero_slice only skips
    # pre-zeroing when the selected kernel actually performs per-slice in-L2 zeroing!).
    ws5 = getattr(weight, "_inkling_nvfp4_ws5", None) if nvfp4 else None
    w2_v5_lib = None
    if nvfp4 and PF_NVFP4_W2_V5:
        _has_ws5_w2 = ws5 is not None and ws5.get("proj") == 1
        _has_pan_w2 = (
            getattr(weight, "_inkling_panel", None) == (256, 16)
            and getattr(weight_scale, "_inkling_panel", None) == (256, 2)
            and weight.data.untyped_storage().size() > 0
            and weight_scale.data.untyped_storage().size() > 0
        )
        if _has_ws5_w2 or _has_pan_w2:
            w2_v5_lib = _get_pf_nvfp4_w2_v5_lib()
    mx6 = getattr(weight, "_inkling_mx6", None) if not nvfp4 else None
    mx7 = getattr(weight, "_inkling_mx7", None) if not nvfp4 else None
    mx7_v10_lib = (_get_pf_mx7_v10_lib() if (isinstance(mx7, tuple) and PF_MX7_V10) else None)
    mx7_v4_lib = (_get_pf_mx7_v4_lib() if (isinstance(mx7, tuple) and mx7_v10_lib is None) else None)
    mx7_v6_lib = (_get_mx7_v6_lib() if (isinstance(mx7, tuple) and use_persist and mx7_v10_lib is None and mx7_v4_lib is None) else None)
    can_persist_handler = False
    if w2_v5_lib is not None:
        can_persist_handler = True
    elif mx7_v10_lib is not None or mx7_v4_lib is not None:
        can_persist_handler = True
    elif use_persist:
        if nvfp4:
            if ws5 is not None and ws5.get("proj") == 1:
                can_persist_handler = hasattr(lib, "pf_nvfp4_w2_ws5_native_v4")
            elif getattr(weight, "_inkling_panel", None) == (256, 16):
                can_persist_handler = hasattr(lib, "pf_nvfp4_w2_panel_native_v4")
        else:
            if mx7_v6_lib is not None:
                can_persist_handler = True
            elif not isinstance(mx7, tuple) and (
                getattr(weight, "_inkling_panel", None) == (128, 32) or isinstance(mx6, tuple)
            ):
                can_persist_handler = hasattr(lib, "pf_mxfp8_w2_native_v4")

    if row2tok is None:
        row2tok = build_row2tok(src2dst, plan.rows, topk=topk, lib=lib)

    if out is None:
        out = torch.empty((tokens, n), dtype=torch.bfloat16, device=device)
        out_prezeroed = False

    mx7_v10_p64 = 4
    mx7_v10_zmode = 0
    if w2_v5_lib is not None:
        w2_p = select_w2_v5_panels(tokens, cm, w2_panels)
        v5_zmode = (
            0
            if l2_zero_slice is False
            else select_w2_v5_zero_mode(tokens, cm, w2_p, out_prezeroed)
        )
        do_l2_slice = int(v5_zmode > 0)
    elif mx7_v10_lib is not None:
        w2_p = 4
        v5_zmode = 0
        mx7_v10_p64 = select_mx7_v10_panels(tokens, cm, w2_panels)
        mx7_v10_zmode = (
            0
            if l2_zero_slice is False
            else select_mx7_v10_zero_mode(tokens, cm, mx7_v10_p64, out_prezeroed)
        )
        do_l2_slice = int(mx7_v10_zmode > 0)
    else:
        w2_p = select_w2_panels(tokens, cm, w2_panels, persistent=can_persist_handler)
        v5_zmode = 0
        eff_l2_zero = (PF_MX7_V4_L2_ZERO_SLICE or use_l2_zero) if mx7_v4_lib is not None else use_l2_zero
        do_l2_slice = int(cm == 2 and can_persist_handler and eff_l2_zero and not out_prezeroed and tokens > 0)

    if cm == 1:
        y_accum = _get_y_fp32(tokens, n, device)
        out_ptr = y_accum.data_ptr()
    else:
        if not out_prezeroed and not do_l2_slice:
            zero_bf16_out(out, lib=lib)
        y_accum = None
        out_ptr = out.data_ptr()

    as_stride = int(act_scale.stride(0)) if act_scale.ndim >= 2 else (18 if nvfp4 else 9)

    if nvfp4:
        assert row_scale is not None
        tw_ptr = topk_weights.data_ptr() if topk_weights is not None else 0
        if ws5 is not None and ws5.get("proj") == 1:
            if w2_v5_lib is not None:
                p_ptr, max_w = _get_or_build_w2_v5_plan(tiles, plan, num_experts, w2_v5_lib, st)
                s_state = _get_w2_v5_slice_state(device)
                eff_ev = PF_NVFP4_W2_V5_W_EVICT if w_evict_mode is None else w_evict_mode
                rc = w2_v5_lib.pf_nvfp4_w2_ws5_v5_combine(
                    act.data_ptr(), act_scale.data_ptr(), row_scale.data_ptr(), tw_ptr, row2tok.data_ptr(),
                    ws5["w_pack"].data_ptr(), ws5["ws5_pack"].data_ptr(), ws5["ws_base"].data_ptr(),
                    ws5["p_off"].data_ptr(), ws5["patches"].data_ptr(),
                    ones.data_ptr(), tiles.data_ptr(), p_ptr, s_state.data_ptr(), out_ptr,
                    plan.max_ctas, plan.tile, num_experts, max_w,
                    1 if as_row_major else 0, as_stride,
                    1 if ws5.get("has_patches", True) else 0, cm, eff_ev, w2_p,
                    v5_zmode, tokens, st,
                )
            elif can_persist_handler:
                p_ptr, max_w = get_or_build_expert_plan(tiles, plan, num_experts, lib, st)
                rc = lib.pf_nvfp4_w2_ws5_native_v4(
                    act.data_ptr(), act_scale.data_ptr(), row_scale.data_ptr(), row2tok.data_ptr(),
                    ws5["w_pack"].data_ptr(), ws5["ws5_pack"].data_ptr(), ws5["ws_base"].data_ptr(),
                    ws5["p_off"].data_ptr(), ws5["patches"].data_ptr(),
                    ones.data_ptr(), tiles.data_ptr(), p_ptr, out_ptr,
                    plan.max_ctas, plan.tile, num_experts, max_w,
                    1 if as_row_major else 0, as_stride,
                    1 if ws5.get("has_patches", True) else 0, cm, w_evict_v4, w2_p,
                    do_l2_slice, tokens, st,
                )
            elif hasattr(lib, "pf_nvfp4_w2_ws5_native_v3"):
                rc = lib.pf_nvfp4_w2_ws5_native_v3(
                    act.data_ptr(), act_scale.data_ptr(), row_scale.data_ptr(), row2tok.data_ptr(),
                    ws5["w_pack"].data_ptr(), ws5["ws5_pack"].data_ptr(), ws5["ws_base"].data_ptr(),
                    ws5["p_off"].data_ptr(), ws5["patches"].data_ptr(),
                    ones.data_ptr(), tiles.data_ptr(), out_ptr,
                    plan.max_ctas, plan.tile, bm_sel, 1 if as_row_major else 0, as_stride,
                    1 if ws5.get("has_patches", True) else 0, cm, w_evict, w2_p, st,
                )
            else:
                rc = lib.pf_nvfp4_w2_ws5_native(
                    act.data_ptr(), act_scale.data_ptr(), row_scale.data_ptr(), row2tok.data_ptr(),
                    ws5["w_pack"].data_ptr(), ws5["ws5_pack"].data_ptr(), ws5["ws_base"].data_ptr(),
                    ws5["p_off"].data_ptr(), ws5["patches"].data_ptr(),
                    ones.data_ptr(), tiles.data_ptr(), out_ptr,
                    plan.max_ctas, plan.tile, bm_sel, 1 if as_row_major else 0, as_stride,
                    1 if ws5.get("has_patches", True) else 0, cm, st,
                )
        elif (
            getattr(weight, "_inkling_panel", None) == (256, 16)
            and getattr(weight_scale, "_inkling_panel", None) == (256, 2)
            and weight.data.untyped_storage().size() > 0
            and weight_scale.data.untyped_storage().size() > 0
        ):
            if w2_v5_lib is not None:
                p_ptr, max_w = _get_or_build_w2_v5_plan(tiles, plan, num_experts, w2_v5_lib, st)
                s_state = _get_w2_v5_slice_state(device)
                eff_ev = PF_NVFP4_W2_V5_W_EVICT if w_evict_mode is None else w_evict_mode
                rc = w2_v5_lib.pf_nvfp4_w2_panel_v5_combine(
                    act.data_ptr(), act_scale.data_ptr(), row_scale.data_ptr(), tw_ptr, row2tok.data_ptr(),
                    weight.data_ptr(), weight_scale.data_ptr(),
                    ones.data_ptr(), tiles.data_ptr(), p_ptr, s_state.data_ptr(), out_ptr,
                    plan.max_ctas, plan.tile, num_experts, max_w,
                    1 if as_row_major else 0, as_stride,
                    cm, eff_ev, w2_p, v5_zmode, tokens, st,
                )
            elif can_persist_handler:
                p_ptr, max_w = get_or_build_expert_plan(tiles, plan, num_experts, lib, st)
                rc = lib.pf_nvfp4_w2_panel_native_v4(
                    act.data_ptr(), act_scale.data_ptr(), row_scale.data_ptr(), row2tok.data_ptr(),
                    weight.data_ptr(), weight_scale.data_ptr(),
                    ones.data_ptr(), tiles.data_ptr(), p_ptr, out_ptr,
                    plan.max_ctas, plan.tile, num_experts, max_w,
                    1 if as_row_major else 0, as_stride,
                    cm, w_evict_v4, w2_p, do_l2_slice, tokens, st,
                )
            elif hasattr(lib, "pf_nvfp4_w2_panel_native_v3"):
                rc = lib.pf_nvfp4_w2_panel_native_v3(
                    act.data_ptr(), act_scale.data_ptr(), row_scale.data_ptr(), row2tok.data_ptr(),
                    weight.data_ptr(), weight_scale.data_ptr(),
                    ones.data_ptr(), tiles.data_ptr(), out_ptr,
                    plan.max_ctas, plan.tile, bm_sel, 1 if as_row_major else 0, as_stride,
                    cm, w_evict, w2_p, st,
                )
            else:
                rc = lib.pf_nvfp4_w2_panel_native(
                    act.data_ptr(), act_scale.data_ptr(), row_scale.data_ptr(), row2tok.data_ptr(),
                    weight.data_ptr(), weight_scale.data_ptr(),
                    ones.data_ptr(), tiles.data_ptr(), out_ptr,
                    plan.max_ctas, plan.tile, bm_sel, 1 if as_row_major else 0, as_stride,
                    cm, st,
                )
        else:
            return None
    else:
        if mx7_v10_lib is not None and isinstance(mx7, tuple):
            slabs, ws3, flips, f_off, base_ws = mx7
            p_ptr, max_w = _get_or_build_mx7_v10_plan(tiles, plan, num_experts, mx7_v10_lib, st)
            s_state = _get_mx7_v10_slice_state(device)
            eff_ev = PF_MX7_V10_W_EVICT if w_evict_mode is None else w_evict_mode
            rc = mx7_v10_lib.pf_mx7_w2_v10_combine(
                act.data_ptr(), act_scale.data_ptr(), row2tok.data_ptr(),
                slabs.data_ptr(), ws3.data_ptr(), flips.data_ptr(), f_off.data_ptr(),
                ones.data_ptr(), tiles.data_ptr(), p_ptr, s_state.data_ptr(), out_ptr,
                plan.max_ctas, plan.tile, num_experts, max_w, int(base_ws),
                1 if as_row_major else 0, as_stride,
                cm, eff_ev, mx7_v10_p64, mx7_v10_zmode, tokens, st,
            )
        elif mx7_v4_lib is not None and isinstance(mx7, tuple):
            slabs, ws3, flips, f_off, base_ws = mx7
            p_ptr, max_w = _get_or_build_mx7_plan(tiles, plan, num_experts, mx7_v4_lib, st)
            p64 = PF_MX7_V4_W2_PANELS if PF_MX7_V4_W2_PANELS in (4, 6, 8, 12) else 4
            rc = mx7_v4_lib.pf_mx7_w2_prefill_persistent(
                act.data_ptr(), act_scale.data_ptr(), row2tok.data_ptr(),
                slabs.data_ptr(), ws3.data_ptr(), flips.data_ptr(), f_off.data_ptr(),
                ones.data_ptr(), tiles.data_ptr(), p_ptr, out_ptr,
                plan.max_ctas, plan.tile, num_experts, max_w, int(base_ws),
                1 if as_row_major else 0, as_stride,
                cm, PF_MX7_V4_W_EVICT, p64, do_l2_slice, tokens, st,
            )
        elif mx7_v6_lib is not None and isinstance(mx7, tuple):
            slabs, ws3, flips, f_off, base_ws = mx7
            p_ptr, max_w = get_or_build_expert_plan(tiles, plan, num_experts, lib, st)
            rc = mx7_v6_lib.mx7_v6_prefill_native_w2(
                act.data_ptr(), act_scale.data_ptr(), row2tok.data_ptr(),
                slabs.data_ptr(), ws3.data_ptr(), flips.data_ptr(), f_off.data_ptr(),
                ones.data_ptr(), tiles.data_ptr(), p_ptr, out_ptr,
                plan.max_ctas, plan.tile, num_experts, max_w, base_ws,
                1 if as_row_major else 0, as_stride,
                cm, w_evict_v4, w2_p, do_l2_slice, tokens, st,
            )
        else:
            mx7_lib = _get_mx7_v5_lib() if isinstance(mx7, tuple) else None
            if mx7_lib is not None:
                slabs, ws3, flips, f_off, base_ws = mx7
                rc = mx7_lib.mx7_v5_prefill_native_w2(
                    act.data_ptr(), act_scale.data_ptr(), row2tok.data_ptr(),
                    slabs.data_ptr(), ws3.data_ptr(), flips.data_ptr(), f_off.data_ptr(),
                    ones.data_ptr(), tiles.data_ptr(), out_ptr,
                    plan.max_ctas, plan.tile, bm_sel, base_ws,
                    1 if as_row_major else 0, as_stride, cm, st,
                )
            else:
                w_eff, ws_eff = weight, weight_scale
                _mx6_tmp = None
                if weight.data.untyped_storage().size() == 0 or weight_scale.data.untyped_storage().size() == 0:
                    if isinstance(mx6, tuple):
                        from . import trtllm_bmm as _bmm_mx6
                        _mx6_tmp = _bmm_mx6._mx6_unpack_to_panel_tmp(weight, 1)
                        if _mx6_tmp is None:
                            return None
                        w_eff, ws_eff = _mx6_tmp
                    else:
                        return None
                try:
                    if (
                        getattr(w_eff, "_inkling_panel", None) != (128, 32)
                        or getattr(ws_eff, "_inkling_panel", None) != (128, 1)
                    ):
                        return None
                    if can_persist_handler:
                        p_ptr, max_w = get_or_build_expert_plan(tiles, plan, num_experts, lib, st)
                        rc = lib.pf_mxfp8_w2_native_v4(
                            act.data_ptr(), act_scale.data_ptr(), row2tok.data_ptr(),
                            w_eff.data_ptr(), ws_eff.data_ptr(),
                            ones.data_ptr(), tiles.data_ptr(), p_ptr, out_ptr,
                            plan.max_ctas, plan.tile, num_experts, max_w,
                            1 if as_row_major else 0, as_stride,
                            cm, w_evict_v4, w2_p, do_l2_slice, tokens, st,
                        )
                    elif hasattr(lib, "pf_mxfp8_w2_native_v3"):
                        rc = lib.pf_mxfp8_w2_native_v3(
                            act.data_ptr(), act_scale.data_ptr(), row2tok.data_ptr(),
                            w_eff.data_ptr(), ws_eff.data_ptr(),
                            ones.data_ptr(), tiles.data_ptr(), out_ptr,
                            plan.max_ctas, plan.tile, bm_sel, 1 if as_row_major else 0, as_stride,
                            cm, w_evict, w2_p, st,
                        )
                    else:
                        rc = lib.pf_mxfp8_w2_native(
                            act.data_ptr(), act_scale.data_ptr(), row2tok.data_ptr(),
                            w_eff.data_ptr(), ws_eff.data_ptr(),
                            ones.data_ptr(), tiles.data_ptr(), out_ptr,
                            plan.max_ctas, plan.tile, bm_sel, 1 if as_row_major else 0, as_stride,
                            cm, st,
                        )
                finally:
                    if _mx6_tmp is not None:
                        del _mx6_tmp, w_eff, ws_eff
    assert rc == 0, f"w2_combine_pf_native CUDA error {rc} (nvfp4={nvfp4}, cm={cm})"

    if cm == 1:
        rc2 = lib.pf_fp32_to_bf16_zero(y_accum.data_ptr(), out.data_ptr(), tokens * n, st)
        assert rc2 == 0, f"pf_fp32_to_bf16_zero CUDA error {rc2}"

    return out
