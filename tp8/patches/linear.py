# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""bf16 linears: small batches on FlashInfer's CuTe DSL low-M kernels, optional
prefill-only dynamic FP8 dense path (INKLING_PF_DENSE_FP8=1, M >= 1024) on SM121,
and the reference ``linear`` with fp32 accumulation in ordered 1024-wide K groups."""

import ctypes
import os
import torch

from vllm.triton_utils import tl, triton
from vllm.utils.flashinfer import is_flashinfer_autotuning

# Up to this many rows, FlashInfer's CuTe DSL low-M kernels run near HBM
# bandwidth (cuBLAS is up to 2x slower); larger batches use cuBLAS.
LOW_M_MAX_TOKENS = 32
# sm121: FlashInfer's CuTe DSL mm_bf16 has no SM12x build.
_SM12 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12
# INKLING_SKINNY=1: decode-size bf16 GEMMs via a split-K Triton kernel that
# streams the weight at ~230-249 GB/s (cuBLAS picks 1-warp SM80 wmma, 207-230).
_SKINNY = _SM12 and os.environ.get("INKLING_SKINNY", "0") == "1"


# INKLING_PDL_SKINNY=1: qkvr (out is None, K=4608, N in _PDL_QKVR_N, behind attn_res_row)
# INKLING_WO_L2PF=1:    wo_ud (out given, weight [_PDL_WO_N, _PDL_WO_K], behind the
#                       sm12 attention + merge/gate_sink chain)
# launch their skinny GEMM with PDL + own-slice L2 prefetch (skinny.py).  Gated by
# shape so every other skinny call (drafter, k/v projections, lm_head) keeps the plain
# launch.  Both default OFF; the predecessors' early gdc_launch_dependents are behind
# the same flags (attn_res.py, sm12_attn.py).
_PDL_QKVR = _SM12 and os.environ.get("INKLING_PDL_SKINNY", "0") == "1"
_PDL_WO = (_SM12 and os.environ.get("INKLING_WO_L2PF", "0") == "1"
           and os.environ.get("INKLING_ATTN_FUSE_GS", "0") == "1")
# v12 opt-in (default OFF): decouple BF11 wo_ud PDL (attn->wo_ud wait + wo_ud->rdma_ar trig)
# from legacy Triton INKLING_WO_L2PF so all 48 attention layers overlap cleanly at C=1..8!
_PDL_WO_BF11 = (_SM12 and os.environ.get("INKLING_PDL_WO_BF11", "0").strip().lower() in (
    "1", "on", "true", "yes",
))
_PDL = _PDL_QKVR or _PDL_WO or _PDL_WO_BF11
_PDL_QKVR_N = tuple(int(v) for v in os.environ.get("INKLING_PDL_QKVR_N", "1896,1640,128").split(","))
_PDL_WO_N, _PDL_WO_K = (int(v) for v in os.environ.get("INKLING_PDL_WO_NK", "4608,768").split(","))

# ---------------------------------------------------------------------------
# Prefill-only Dynamic FP8 Dense Linear (INKLING_PF_DENSE_FP8=1, M >= 1024)
# ---------------------------------------------------------------------------
# Strictly gated to large prefill chunks outside CUDA graph capture so decode
# (M <= 16) and exact_ink.py (T <= 120) remain 100% untouched & 0-ULP bit-exact!
_PF_DENSE_FP8 = _SM12 and os.environ.get("INKLING_PF_DENSE_FP8", "0").strip().lower() in (
    "1", "true", "on", "yes",
)
_PF_DENSE_FP8_MIN_M = int(os.environ.get("INKLING_PF_DENSE_FP8_MIN_M", "1024"))
# Break-even N on GB10 (~660): quantizing x[M, K] costs ~3*M*K/BW, so N < 512
# (e.g. global k_proj/v_proj N=128, K=4608) stays faster in BF16 cuBLAS!
_PF_DENSE_FP8_MIN_N = int(os.environ.get("INKLING_PF_DENSE_FP8_MIN_N", "512"))
_PF_DENSE_FP8_MODE = os.environ.get("INKLING_PF_DENSE_FP8_MODE", "rowwise").strip().lower()
# Prefill-only FP8 Router GEMM (INKLING_PF_ROUTER_FP8=1, M >= 1024):
# Feeds pre-quantized (x_fp8, x_scale) directly into torch._scaled_mm inside
# k_group_partials (or dynamically quantizes BF16 x), bypassing BF16 cuBLAS
# router GEMM (~5.0s -> ~1.5s @ 1M). Strictly gated off for decode / CUDA graphs.
_PF_ROUTER_FP8 = _SM12 and os.environ.get("INKLING_PF_ROUTER_FP8", "0").strip().lower() in (
    "1", "true", "on", "yes",
)
_PF_ROUTER_FP8_MIN_M = int(os.environ.get("INKLING_PF_ROUTER_FP8_MIN_M", "1024"))
# Wave-V15 opt-in (default OFF): protect decode tokens from FP8 quantization on both
# pure eager decode (M >= MIN_M graph miss -> 100% BF16) and mixed prefill+decode chunks
# (decode prefix x[:num_decode_tokens] runs exact BF16, prefill suffix runs fast FP8 TC).
_PF_MIXED_DECODE_BF16 = os.environ.get("INKLING_PF_MIXED_DECODE_BF16", "0").strip().lower() in (
    "1", "true", "on", "yes",
)


def _classify_pf_decode_batch(m: int) -> tuple[bool, int]:
    """Inspect ForwardContext.attn_metadata for prefill vs decode token split.

    Returns ``(allow_fp8, n_decode_prefix)``:
      - ``(False, m)``: pure decode batch -> NEVER quantize to FP8!
      - ``(True, n_dec)`` with ``0 < n_dec < m``: mixed chunk -> BF16 on ``[:n_dec]``, FP8 on ``[n_dec:]``.
      - ``(True, 0)``: pure prefill (or no ForwardContext) -> fast FP8 path on all ``m`` rows.
    """
    if not _PF_MIXED_DECODE_BF16 or m <= 0:
        return True, 0
    try:
        from vllm.forward_context import get_forward_context, is_forward_context_available

        if not is_forward_context_available():
            return True, 0
        amd = getattr(get_forward_context(), "attn_metadata", None)
        if not isinstance(amd, dict) or not amd:
            return True, 0
        saw_attn_md = False
        max_pf_toks = 0
        max_pf_reqs = 0
        max_dec_reqs = 0
        max_dec_toks = 0
        for md in amd.values():
            if md is None:
                continue
            has_pf_tok = hasattr(md, "num_prefill_tokens")
            has_dec_req = hasattr(md, "num_decode_reqs")
            if not (has_pf_tok or has_dec_req):
                continue
            saw_attn_md = True
            if has_pf_tok:
                max_pf_toks = max(max_pf_toks, int(getattr(md, "num_prefill_tokens") or 0))
            if hasattr(md, "num_prefill_reqs"):
                max_pf_reqs = max(max_pf_reqs, int(getattr(md, "num_prefill_reqs") or 0))
            if has_dec_req:
                max_dec_reqs = max(max_dec_reqs, int(getattr(md, "num_decode_reqs") or 0))
            if hasattr(md, "num_decode_tokens"):
                max_dec_toks = max(max_dec_toks, int(getattr(md, "num_decode_tokens") or 0))
        if not saw_attn_md:
            return True, 0
        if max_pf_toks == 0 and max_pf_reqs == 0 and max_dec_reqs > 0:
            # Pure decode batch running eagerly (e.g. graph miss at M>=256): keep 100% BF16!
            return False, m
        if max_dec_reqs > 0 and max_dec_toks > 0:
            if max_dec_toks >= m:
                return False, m
            return True, max_dec_toks
        return True, 0
    except Exception:
        # Fail-closed against corrupting decode tokens if metadata inspection unexpectedly throws.
        return False, m
_FP8_MAX = 448.0
_FP8_EPS = 1e-12
_PF_ROWQUANT_V4 = _SM12 and os.environ.get("INKLING_PF_ROWQUANT_V4", "0").strip().lower() in (
    "1", "true", "on", "yes",
)
_PF_NONMOE_V4_LIB_PATH = os.environ.get(
    "INKLING_PF_NONMOE_V4_LIB", "/rdma/libpf_attn_nonmoe_v4_sm121.so"
).strip()
_PF_NONMOE_V4_LIB = None
_PF_NONMOE_V4_TRIED = False
# Wave-V11 314 TFLOPS SM121a FP8 `m16n8k32.f16`-inner / FP32-outer Dense GEMM + Fused wo_ud->A2A
# Strictly opt-in (default OFF) and gated to eager prefill (M >= 1024, !capturing -> 0 decode touch!)
_PF_DENSE_FP8_F16ACC = _SM12 and os.environ.get("INKLING_PF_DENSE_FP8_F16ACC", "0").strip().lower() in (
    "1", "true", "on", "yes",
)
_PF_WO_UD_FUSED_A2A = _SM12 and os.environ.get("INKLING_PF_WO_UD_FUSED_A2A", "0").strip().lower() in (
    "1", "true", "on", "yes",
)
_PF_DENSE_FP8_RESERVE_SMS = max(
    min(int(os.environ.get("INKLING_PF_DENSE_FP8_RESERVE_SMS", "8") or 8), 24), 0
)
_PF_DENSE_FP8_RESERVE_EXPLICIT = "INKLING_PF_DENSE_FP8_RESERVE_SMS" in os.environ
_PF_DENSE_FP8_TC_V13_LIB_PATH = os.environ.get(
    "INKLING_PF_DENSE_FP8_TC_V13_LIB", "/rdma/libpf_dense_fp8_tc_v13.so"
).strip()
_PF_DENSE_FP8_TC_V11_LIB_PATH = os.environ.get(
    "INKLING_PF_DENSE_FP8_TC_V11_LIB", "/rdma/libpf_dense_fp8_tc_v11.so"
).strip()
_PF_DENSE_FP8_TC_V11_LIB = None
_PF_DENSE_FP8_TC_V11_TRIED = False
_FP8_W_V11_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
_V11_W_MAX_PREQUANT = 2.2
_V11_W_MAX_WO_UD = 22.0
_V11_A_MAX_WO_UD = 22.0
_PF_ROWQUANT_SCRATCH: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
_SHARED_PF_SCRATCH = os.environ.get(
    "INKLING_SHARED_PF_SCRATCH",
    os.environ.get("INKLING_FREE_PACKED_RAW_WEIGHTS", "0"),
).strip().lower() in ("1", "true", "on", "yes")
_SHARED_PF_SCRATCH_BUF: dict[tuple, torch.Tensor] = {}


def get_shared_pf_scratch(
    device: torch.device,
    nbytes: int,
    stream_id: int | None = None,
) -> torch.Tensor:
    """Shared same-stream prefill scratch pool (`INKLING_SHARED_PF_SCRATCH=1`).

    Allows disjoint prefill phases (`_PF_ROWQUANT_SCRATCH` 72 MiB and
    `global_attn_1m._CUDA_SCRATCH` 48.75 MiB) to alias ONE high-water buffer.
    """
    if stream_id is None:
        stream_id = int(torch.cuda.current_stream(device).cuda_stream) if device.type == "cuda" else 0
    need = (int(nbytes) + 255) & ~255
    skey = (device, stream_id)
    buf = _SHARED_PF_SCRATCH_BUF.get(skey)
    if buf is None or int(buf.numel()) < need:
        buf = torch.empty((need,), dtype=torch.uint8, device=device)
        _SHARED_PF_SCRATCH_BUF[skey] = buf
    return buf[:need]


def release_prefill_scratch() -> int:
    """Release non-graphed prefill row-quant scratch pools (preserves _FP8_W_CACHE)."""
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return 0
    freed = 0
    for a, b in _PF_ROWQUANT_SCRATCH.values():
        freed += int(a.numel() * a.element_size()) + int(b.numel() * b.element_size())
    _PF_ROWQUANT_SCRATCH.clear()
    for buf in _SHARED_PF_SCRATCH_BUF.values():
        freed += int(buf.numel() * buf.element_size())
    _SHARED_PF_SCRATCH_BUF.clear()
    return freed


def _resolve_bf16_weight(weight: torch.Tensor) -> torch.Tensor:
    """Return `weight` if raw storage is live, else lossless 0-ULP unpack from BF12/BF11."""
    if weight.untyped_storage().size() > 0:
        return weight
    try:
        from . import router_bf12

        unpacked = router_bf12.unpack_bf16(weight)
        if unpacked is not None:
            return unpacked
    except Exception:
        pass
    try:
        from . import skinny_bf11

        unpacked = skinny_bf11.unpack_bf16(weight)
        if unpacked is not None:
            return unpacked
    except Exception:
        pass
    return weight


def _get_pf_nonmoe_v4_lib():
    global _PF_NONMOE_V4_LIB, _PF_NONMOE_V4_TRIED
    if _PF_NONMOE_V4_TRIED:
        return _PF_NONMOE_V4_LIB
    _PF_NONMOE_V4_TRIED = True
    if not _PF_NONMOE_V4_LIB_PATH or not os.path.exists(_PF_NONMOE_V4_LIB_PATH):
        return None
    try:
        lib = ctypes.CDLL(_PF_NONMOE_V4_LIB_PATH)
        lib.quant_fp8_rowwise_v4_sm121_fwd.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int64,
            ctypes.c_uint64,
        ]
        lib.quant_fp8_rowwise_v4_sm121_fwd.restype = ctypes.c_int
        _PF_NONMOE_V4_LIB = lib
    except Exception:
        _PF_NONMOE_V4_LIB = None
    return _PF_NONMOE_V4_LIB

# Keyed by (weight.data_ptr(), N, K, stride0, stride1, mode) so ephemeral slices
# like global attention q_side_w / w_k / w_v cache cleanly without re-quantizing!
_FP8_W_CACHE: dict[tuple[int, int, int, int, int, str], tuple[torch.Tensor, torch.Tensor, int]] = {}
# Resolved backend mode after one-time SM121a probe: ("rowwise" | "tensorwise" | "off", supports_out_kwarg)
_FP8_CAP: tuple[str, bool] | None = None


@triton.jit
def _quant_fp8_rowwise_ht_kernel(
    X,
    Y,
    S,
    M,
    stride_xm,
    stride_ym,
    HEAD: tl.constexpr,
    TAIL: tl.constexpr,
):
    """Single-pass fused BF16 -> FP8 E4M3FN per-row quantizer for K = HEAD + TAIL
    where HEAD and TAIL are powers of 2 (covers K=4608=4096+512, 768=512+256,
    6144=4096+2048, and all power-of-2 K with zero DRAM re-reads!)."""
    row = tl.program_id(0)
    if row >= M:
        # Zero-pad rows M..M_pad-1 for cuBLASLt 16-row alignment.
        oh = tl.arange(0, HEAD)
        z = tl.zeros((HEAD,), dtype=tl.float8e4nv)
        tl.store(Y + row.to(tl.int64) * stride_ym + oh, z)
        if TAIL > 0:
            ot = tl.arange(0, TAIL)
            zt = tl.zeros((TAIL,), dtype=tl.float8e4nv)
            tl.store(Y + row.to(tl.int64) * stride_ym + HEAD + ot, zt)
        tl.store(S + row, 1.0)
        return

    x_row = X + row.to(tl.int64) * stride_xm
    y_row = Y + row.to(tl.int64) * stride_ym
    oh = tl.arange(0, HEAD)
    xh = tl.load(x_row + oh).to(tl.float32)
    amax = tl.max(tl.abs(xh), axis=0)
    if TAIL > 0:
        ot = tl.arange(0, TAIL)
        xt = tl.load(x_row + HEAD + ot).to(tl.float32)
        amax = tl.maximum(amax, tl.max(tl.abs(xt), axis=0))
    scale = tl.maximum(amax * (1.0 / 448.0), 1e-12)
    inv_scale = 1.0 / scale
    tl.store(S + row, scale)
    yh = tl.clamp(xh * inv_scale, -448.0, 448.0).to(tl.float8e4nv)
    tl.store(y_row + oh, yh)
    if TAIL > 0:
        yt = tl.clamp(xt * inv_scale, -448.0, 448.0).to(tl.float8e4nv)
        tl.store(y_row + HEAD + ot, yt)


@triton.jit
def _quant_fp8_rowwise_generic_kernel(
    X,
    Y,
    S,
    M,
    K,
    stride_xm,
    stride_ym,
    BLOCK_K: tl.constexpr,
):
    """Generic per-row BF16 -> FP8 E4M3FN quantizer for arbitrary K % 16 == 0."""
    row = tl.program_id(0)
    y_row = Y + row.to(tl.int64) * stride_ym
    if row >= M:
        for k0 in range(0, K, BLOCK_K):
            offs = k0 + tl.arange(0, BLOCK_K)
            tl.store(y_row + offs, tl.zeros((BLOCK_K,), dtype=tl.float8e4nv), mask=offs < K)
        tl.store(S + row, 1.0)
        return

    x_row = X + row.to(tl.int64) * stride_xm
    amax = 0.0
    for k0 in range(0, K, BLOCK_K):
        offs = k0 + tl.arange(0, BLOCK_K)
        xv = tl.load(x_row + offs, mask=offs < K, other=0.0).to(tl.float32)
        amax = tl.maximum(amax, tl.max(tl.abs(xv), axis=0))
    scale = tl.maximum(amax * (1.0 / 448.0), 1e-12)
    inv_scale = 1.0 / scale
    tl.store(S + row, scale)
    for k0 in range(0, K, BLOCK_K):
        offs = k0 + tl.arange(0, BLOCK_K)
        xv = tl.load(x_row + offs, mask=offs < K, other=0.0).to(tl.float32)
        yv = tl.clamp(xv * inv_scale, -448.0, 448.0).to(tl.float8e4nv)
        tl.store(y_row + offs, yv, mask=offs < K)


def quantize_fp8_rowwise(
    x: torch.Tensor,
    pad_m16: bool = True,
    *,
    reuse_scratch: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamic per-row BF16 [M, K] -> (x_fp8 [M_pad, K] e4m3fn, x_scale [M_pad, 1] fp32)."""
    m, k = x.shape
    if m == 0:
        return (
            torch.empty((0, k), dtype=torch.float8_e4m3fn, device=x.device),
            torch.empty((0, 1), dtype=torch.float32, device=x.device),
        )
    m_pad = ((m + 15) & ~15) if pad_m16 else m
    can_v4 = (
        _PF_ROWQUANT_V4
        and m >= 512
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and x.stride(1) == 1
        and (k == 768 or (k & 127) == 0)
        and not torch.cuda.is_current_stream_capturing()
    )
    if can_v4 and reuse_scratch:
        stream_id = int(torch.cuda.current_stream(x.device).cuda_stream)
        cap_m = max(m_pad, 8192)
        if _SHARED_PF_SCRATCH:
            fp8_bytes = (cap_m * k + 255) & ~255
            sc_bytes = cap_m * 4
            buf = get_shared_pf_scratch(x.device, fp8_bytes + sc_bytes, stream_id)
            x_fp8 = buf[: m_pad * k].view(torch.float8_e4m3fn).view(m_pad, k)
            x_scale = buf[fp8_bytes : fp8_bytes + m_pad * 4].view(torch.float32).view(m_pad, 1)
        else:
            skey = (x.device, stream_id, k)
            pair = _PF_ROWQUANT_SCRATCH.get(skey)
            if pair is None or pair[0].shape[0] < m_pad:
                pair = (
                    torch.empty((cap_m, k), dtype=torch.float8_e4m3fn, device=x.device),
                    torch.empty((cap_m, 1), dtype=torch.float32, device=x.device),
                )
                _PF_ROWQUANT_SCRATCH[skey] = pair
            x_fp8 = pair[0][:m_pad]
            x_scale = pair[1][:m_pad]
    else:
        x_fp8 = torch.empty((m_pad, k), dtype=torch.float8_e4m3fn, device=x.device)
        x_scale = torch.empty((m_pad, 1), dtype=torch.float32, device=x.device)

    if can_v4:
        lib = _get_pf_nonmoe_v4_lib()
        if lib is not None:
            rc = lib.quant_fp8_rowwise_v4_sm121_fwd(
                x.data_ptr(),
                x_fp8.data_ptr(),
                x_scale.data_ptr(),
                int(m),
                int(m_pad),
                int(k),
                int(x.stride(0)),
                int(torch.cuda.current_stream(x.device).cuda_stream),
            )
            if rc == 0:
                return x_fp8, x_scale

    head = 1 << (k.bit_length() - 1)
    tail = k - head
    if head <= 8192 and (tail == 0 or (tail & (tail - 1)) == 0):
        warps = 8 if k >= 2048 else (4 if k >= 512 else 2)
        _quant_fp8_rowwise_ht_kernel[(m_pad,)](
            x, x_fp8, x_scale, m, x.stride(0), x_fp8.stride(0),
            HEAD=head, TAIL=tail, num_warps=warps,
        )
    else:
        _quant_fp8_rowwise_generic_kernel[(m_pad,)](
            x, x_fp8, x_scale, m, k, x.stride(0), x_fp8.stride(0),
            BLOCK_K=1024, num_warps=4,
        )
    return x_fp8, x_scale


@triton.jit
def _dequant_fp8_rowwise_ht_kernel(
    X,
    S,
    Y,
    stride_xm,
    stride_ym,
    HEAD: tl.constexpr,
    TAIL: tl.constexpr,
):
    """Single-pass fused FP8 E4M3FN + per-row fp32 scale -> BF16 dequantizer for K = HEAD + TAIL."""
    row = tl.program_id(0)
    scale = tl.load(S + row).to(tl.float32)
    x_row = X + row.to(tl.int64) * stride_xm
    y_row = Y + row.to(tl.int64) * stride_ym
    oh = tl.arange(0, HEAD)
    xh = tl.load(x_row + oh).to(tl.float32)
    tl.store(y_row + oh, (xh * scale).to(tl.bfloat16))
    if TAIL > 0:
        ot = tl.arange(0, TAIL)
        xt = tl.load(x_row + HEAD + ot).to(tl.float32)
        tl.store(y_row + HEAD + ot, (xt * scale).to(tl.bfloat16))


def dequantize_fp8_rowwise(x_fp8: torch.Tensor, x_scale: torch.Tensor) -> torch.Tensor:
    """Dequantize FP8 E4M3FN [M, K] + per-row FP32 scale [M, 1] -> contiguous BF16 [M, K]."""
    m, k = int(x_fp8.shape[0]), int(x_fp8.shape[1])
    if m == 0:
        return torch.empty((0, k), dtype=torch.bfloat16, device=x_fp8.device)
    head = 1 << (k.bit_length() - 1)
    tail = k - head
    if (
        x_fp8.is_cuda
        and x_fp8.stride(1) == 1
        and x_scale.is_cuda
        and x_scale.dtype == torch.float32
        and x_scale.is_contiguous()
        and head <= 8192
        and (tail == 0 or (tail & (tail - 1)) == 0)
    ):
        y = torch.empty((m, k), dtype=torch.bfloat16, device=x_fp8.device)
        warps = 8 if k >= 2048 else (4 if k >= 512 else 2)
        _dequant_fp8_rowwise_ht_kernel[(m,)](
            x_fp8, x_scale, y, x_fp8.stride(0), y.stride(0),
            HEAD=head, TAIL=tail, num_warps=warps,
        )
        return y
    return (x_fp8.to(torch.float32) * x_scale).to(torch.bfloat16)


def _get_pf_dense_fp8_tc_v11_lib():
    """Lazily load SM121a `libpf_dense_fp8_tc_v13.so` (or V11 fallback) outside CUDA graph capture."""
    global _PF_DENSE_FP8_TC_V11_LIB, _PF_DENSE_FP8_TC_V11_TRIED, _PF_DENSE_FP8_RESERVE_SMS
    if _PF_DENSE_FP8_TC_V11_TRIED:
        return _PF_DENSE_FP8_TC_V11_LIB
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return None
    _PF_DENSE_FP8_TC_V11_TRIED = True
    _here = os.path.dirname(os.path.abspath(__file__))
    _cands = (
        _PF_DENSE_FP8_TC_V13_LIB_PATH,
        "/rdma/libpf_dense_fp8_tc_v13.so",
        os.path.join(_here, "libpf_dense_fp8_tc_v13.so"),
        _PF_DENSE_FP8_TC_V11_LIB_PATH,
        "/rdma/libpf_dense_fp8_tc_v11.so",
        os.path.join(_here, "libpf_dense_fp8_tc_v11.so"),
    )
    _chosen = next((p for p in _cands if p and os.path.exists(p)), "")
    if _chosen:
        try:
            lib = ctypes.CDLL(_chosen)
            lib.pf_dense_fp8_tc_v11_version.argtypes = []
            lib.pf_dense_fp8_tc_v11_version.restype = ctypes.c_int
            lib.pf_quant_weight_f16safe_v11_fwd.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_uint64,
            ]
            lib.pf_quant_weight_f16safe_v11_fwd.restype = ctypes.c_int
            lib.pf_dense_fp8_f16acc_v11_fwd.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_uint64,
            ]
            lib.pf_dense_fp8_f16acc_v11_fwd.restype = ctypes.c_int
            lib.pf_wo_ud_bf16_to_bf16_v11_fwd.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_float, ctypes.c_int, ctypes.c_uint64,
            ]
            lib.pf_wo_ud_bf16_to_bf16_v11_fwd.restype = ctypes.c_int
            lib.pf_wo_ud_fused_a2a_int8_v11_fwd.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
                ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_int,
                ctypes.c_float, ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
            ]
            lib.pf_wo_ud_fused_a2a_int8_v11_fwd.restype = ctypes.c_int
            if (not _PF_DENSE_FP8_RESERVE_EXPLICIT) and int(lib.pf_dense_fp8_tc_v11_version()) >= 1300:
                _PF_DENSE_FP8_RESERVE_SMS = 0
            _PF_DENSE_FP8_TC_V11_LIB = lib
            print(
                f"[inkling-pf-dense-v11] loaded SM121a 314-TF FP8-f16acc dense TC kernels: "
                f"{_chosen} (ver={lib.pf_dense_fp8_tc_v11_version()}, reserve_sms={_PF_DENSE_FP8_RESERVE_SMS})",
                flush=True,
            )
        except Exception as _e:
            print(f"[inkling-pf-dense-v11] lib load failed ({_e!r}); falling back to cuBLASLt", flush=True)
            _PF_DENSE_FP8_TC_V11_LIB = None
    return _PF_DENSE_FP8_TC_V11_LIB


def _get_cached_w_fp8_v11_f16safe(
    weight: torch.Tensor, w_max: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lazily quantize & cache BF16 `[N, K]` -> `(w_fp8 [N, K] uint8 row-major, w_scale [N] fp32)`
    bounded to `[-w_max, w_max]` so `m16n8k32.f16` inner `K_CHUNK=64` has provably 0 FP16 overflow!"""
    w_key = int(round(w_max * 100.0))
    direct_entries = getattr(weight, "_inkling_fp8_v11_entries", None)
    if isinstance(direct_entries, dict):
        hit = direct_entries.get(w_key)
        if hit is not None:
            return hit
    n, k = int(weight.shape[0]), int(weight.shape[1])
    ptr = int(weight.data_ptr())
    ckey = (ptr, n, k, w_key)
    if ptr != 0:
        cached = _FP8_W_V11_CACHE.get(ckey)
        if cached is not None:
            return cached
    w_src = _resolve_bf16_weight(weight).contiguous()
    lib = _get_pf_dense_fp8_tc_v11_lib()
    if lib is not None and (k % 64) == 0 and w_src.is_cuda:
        w_fp8 = torch.empty((n, k), dtype=torch.uint8, device=w_src.device)
        w_sc = torch.empty((n,), dtype=torch.float32, device=w_src.device)
        rc = lib.pf_quant_weight_f16safe_v11_fwd(
            w_src.data_ptr(),
            w_fp8.data_ptr(),
            w_sc.data_ptr(),
            n,
            k,
            float(w_max),
            int(torch.cuda.current_stream(w_src.device).cuda_stream),
        )
        if rc == 0:
            entry = (w_fp8, w_sc)
            if ptr != 0:
                _FP8_W_V11_CACHE[ckey] = entry
            try:
                if not isinstance(direct_entries, dict):
                    direct_entries = {}
                    weight._inkling_fp8_v11_entries = direct_entries  # type: ignore[attr-defined]
                direct_entries[w_key] = entry
            except Exception:
                pass
            return entry
    w_f32 = w_src.float()
    amax = w_f32.abs().amax(dim=1)
    w_sc = torch.clamp(amax * (1.0 / float(w_max)), min=_FP8_EPS).contiguous()
    w_fp8 = (
        torch.clamp(w_f32 / w_sc[:, None], -float(w_max), float(w_max))
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
        .contiguous()
    )
    entry = (w_fp8, w_sc)
    if ptr != 0:
        _FP8_W_V11_CACHE[ckey] = entry
    try:
        if not isinstance(direct_entries, dict):
            direct_entries = {}
            weight._inkling_fp8_v11_entries = direct_entries  # type: ignore[attr-defined]
        direct_entries[w_key] = entry
    except Exception:
        pass
    return entry


def pf_wo_ud_fused_a2a_int8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    world_size: int,
    rank: int,
):
    """Wave-V11 fused BF16-in `wo_ud` (`314 TFLOPS` FP8-f16acc) -> direct register `A2A_INT8_g64`
    epilogue returning `PreQuantA2APartial` (eliminates `quant_k768`, `attn_partial` BF16 write,
    AND `a2a_quant_int8_g64_lowsm` on `comm_stream`!). Returns `None` on any guard miss."""
    if not (_PF_DENSE_FP8 and _PF_WO_UD_FUSED_A2A):
        return None
    if not x.is_cuda or x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        return None
    if x.ndim != 2 or weight.ndim != 2 or x.stride(1) != 1 or weight.stride(1) != 1:
        return None
    if torch.cuda.is_current_stream_capturing():
        return None
    m, k = int(x.shape[0]), int(x.shape[1])
    n = int(weight.shape[0])
    if m < max(_PF_DENSE_FP8_MIN_M, 1024) or world_size not in (2, 4, 8):
        return None
    if (m % world_size) != 0 or (k % 64) != 0 or (n % 128) != 0:
        return None
    s = m // world_size
    if (s % 128) != 0:
        return None
    lib = _get_pf_dense_fp8_tc_v11_lib()
    if lib is None:
        return None
    from .. import prefill_sp_rs as _sp_rs

    w_fp8, w_sc = _get_cached_w_fp8_v11_f16safe(weight, _V11_W_MAX_WO_UD)
    scale_off, peer_bytes = _sp_rs._rhd_g64_packed_bytes(s, n)
    x_c = x if (x.stride(0) & 7) == 0 and (x.data_ptr() & 15) == 0 else x.contiguous()
    send_buf = torch.empty((world_size, peer_bytes), dtype=torch.uint8, device=x.device)
    local_shard = torch.empty((s, n), dtype=torch.bfloat16, device=x.device)
    stream_u64 = int(torch.cuda.current_stream(x.device).cuda_stream)
    rc = lib.pf_wo_ud_fused_a2a_int8_v11_fwd(
        x_c.data_ptr(),
        w_fp8.data_ptr(),
        w_sc.data_ptr(),
        send_buf.data_ptr(),
        local_shard.data_ptr(),
        int(x_c.stride(0)),
        int(local_shard.stride(0)),
        int(peer_bytes),
        int(scale_off),
        s,
        n,
        k,
        world_size,
        rank,
        float(_V11_A_MAX_WO_UD),
        0,
        int(_PF_DENSE_FP8_RESERVE_SMS),
        stream_u64,
    )
    if rc != 0:
        return None
    return _sp_rs.PreQuantA2APartial(
        send_buf=send_buf,
        local_shard=local_shard,
        peer_bytes=peer_bytes,
        scale_off=scale_off,
        S=s,
        H=n,
        world_size=world_size,
        rank=rank,
        is_fp8=False,
    )


def pf_dense_fp8_linear_prequant(
    x_fp8: torch.Tensor,
    x_scale: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    min_n: int | None = None,
) -> torch.Tensor | None:
    """Direct `_scaled_mm` from pre-quantized rowwise FP8 `(x_fp8 [M, K], x_scale [M, 1])`
    (skips `_quant_fp8_rowwise_ht_kernel` completely when fed from FP8 AllGather!)."""
    if (
        not x_fp8.is_cuda
        or x_fp8.dtype != torch.float8_e4m3fn
        or weight.dtype != torch.bfloat16
        or x_fp8.ndim != 2
        or weight.ndim != 2
        or x_fp8.stride(1) != 1
        or weight.stride(1) != 1
    ):
        return None
    m, k = int(x_fp8.shape[0]), int(x_fp8.shape[1])
    n = int(weight.shape[0])
    n_floor = _PF_DENSE_FP8_MIN_N if min_n is None else min_n
    if m <= 0 or int(weight.shape[1]) != k or k < 256 or (k & 15) != 0 or n < n_floor:
        return None
    if out is not None and (out.dtype != torch.bfloat16 or out.ndim != 2 or out.shape != (m, n)):
        return None
    # Wave-V11 314 TFLOPS SM121a FP8 `m16n8k32.f16`-inner / FP32-outer (`W_MAX=2.2`, 0 overflow,
    # native `N % 8 == 0` -> kills `n_pad=1904` copy & doubles TC peak 157 -> 314 TFLOPS!)
    if (
        _PF_DENSE_FP8_F16ACC
        and m >= max(_PF_DENSE_FP8_MIN_M, 1024)
        and (k % 64) == 0
        and (n % 8) == 0
        and x_fp8.is_contiguous()
        and (out is None or (out.stride(1) == 1 and (out.stride(0) & 7) == 0 and (out.data_ptr() & 15) == 0))
        and not torch.cuda.is_current_stream_capturing()
    ):
        v11_lib = _get_pf_dense_fp8_tc_v11_lib()
        if v11_lib is not None:
            w_fp8_v11, w_sc_v11 = _get_cached_w_fp8_v11_f16safe(weight, _V11_W_MAX_PREQUANT)
            s_in_v11 = (
                x_scale
                if (x_scale.dtype == torch.float32 and x_scale.is_contiguous() and x_scale.numel() == m)
                else x_scale.to(torch.float32).reshape(m).contiguous()
            )
            out_buf = out if out is not None else torch.empty((m, n), dtype=torch.bfloat16, device=x_fp8.device)
            rc_v11 = v11_lib.pf_dense_fp8_f16acc_v11_fwd(
                x_fp8.data_ptr(),
                s_in_v11.data_ptr(),
                w_fp8_v11.data_ptr(),
                w_sc_v11.data_ptr(),
                out_buf.data_ptr(),
                m,
                n,
                k,
                int(out_buf.stride(0)),
                int(_PF_DENSE_FP8_RESERVE_SMS),
                int(torch.cuda.current_stream(x_fp8.device).cuda_stream),
            )
            if rc_v11 == 0:
                return out_buf
    mode, supports_out = _probe_fp8_cap(x_fp8.device)
    if mode != "rowwise":
        return None
    w_fp8_t, w_scale, n_pad = _get_cached_w_fp8(weight, "rowwise")
    m_pad = (m + 15) & ~15
    if m_pad != m:
        x_in = torch.zeros((m_pad, k), dtype=torch.float8_e4m3fn, device=x_fp8.device)
        x_in[:m].copy_(x_fp8)
        s_in = torch.ones((m_pad, 1), dtype=torch.float32, device=x_fp8.device)
        s_in[:m].copy_(x_scale.reshape(m, 1))
    else:
        x_in = x_fp8
        s_in = (
            x_scale
            if (
                x_scale.ndim == 2
                and x_scale.shape[1] == 1
                and x_scale.dtype == torch.float32
                and x_scale.is_contiguous()
            )
            else x_scale.to(torch.float32).reshape(m, 1).contiguous()
        )
    can_direct_out = (
        out is not None
        and supports_out
        and m_pad == m
        and n_pad == n
        and out.is_contiguous()
    )
    try:
        if can_direct_out:
            torch._scaled_mm(
                x_in, w_fp8_t, scale_a=s_in, scale_b=w_scale,
                out_dtype=torch.bfloat16, use_fast_accum=False, out=out,
            )
            return out
        res = torch._scaled_mm(
            x_in, w_fp8_t, scale_a=s_in, scale_b=w_scale,
            out_dtype=torch.bfloat16, use_fast_accum=False,
        )
    except Exception:
        return None
    if m_pad != m or n_pad != n:
        res = res[:m, :n]
    if out is not None:
        out.copy_(res)
        return out
    return res


def quantize_fp8_tensorwise(x: torch.Tensor, pad_m16: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamic per-tensor BF16 [M, K] -> (x_fp8 [M_pad, K] e4m3fn, x_scale [1] fp32)."""
    m, k = x.shape
    if m == 0:
        return (
            torch.empty((0, k), dtype=torch.float8_e4m3fn, device=x.device),
            torch.ones((1,), dtype=torch.float32, device=x.device),
        )
    m_pad = ((m + 15) & ~15) if pad_m16 else m
    # Wave-V15 NaN/Inf isolation: prevent a non-finite value in a padded row from poisoning global scale.
    xf = x.to(torch.float32)
    amax = torch.where(torch.isfinite(xf), xf.abs(), torch.zeros((), dtype=torch.float32, device=x.device)).amax()
    scale = torch.clamp(amax * (1.0 / _FP8_MAX), min=_FP8_EPS).reshape(1)
    inv = (1.0 / scale).to(x.dtype)
    q = torch.clamp(x * inv, -_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
    if m_pad == m:
        return q.contiguous(), scale
    out = torch.zeros((m_pad, k), dtype=torch.float8_e4m3fn, device=x.device)
    out[:m].copy_(q)
    return out, scale


def _probe_fp8_cap(device: torch.device) -> tuple[str, bool]:
    """One-time SM121a capability probe for torch._scaled_mm (rowwise vs tensorwise, out=)."""
    global _FP8_CAP
    if _FP8_CAP is not None:
        return _FP8_CAP
    a = torch.zeros((16, 64), dtype=torch.float8_e4m3fn, device=device)
    b = torch.zeros((64, 64), dtype=torch.float8_e4m3fn, device=device).t()
    out_probe = torch.empty((16, 64), dtype=torch.bfloat16, device=device)
    order = ("rowwise", "tensorwise") if _PF_DENSE_FP8_MODE != "tensorwise" else ("tensorwise", "rowwise")
    for mode in order:
        try:
            if mode == "rowwise":
                sa = torch.ones((16, 1), dtype=torch.float32, device=device)
                sb = torch.ones((1, 64), dtype=torch.float32, device=device)
                _ = torch._scaled_mm(a, b, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16, use_fast_accum=False)
                supports_out = False
                try:
                    torch._scaled_mm(a, b, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16, use_fast_accum=False, out=out_probe)
                    supports_out = True
                except Exception:
                    supports_out = False
                _FP8_CAP = ("rowwise", supports_out)
                return _FP8_CAP
            else:
                sa = torch.ones((1,), dtype=torch.float32, device=device)
                sb = torch.ones((1,), dtype=torch.float32, device=device)
                _ = torch._scaled_mm(a, b, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16, use_fast_accum=True)
                supports_out = False
                try:
                    torch._scaled_mm(a, b, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16, use_fast_accum=True, out=out_probe)
                    supports_out = True
                except Exception:
                    supports_out = False
                _FP8_CAP = ("tensorwise", supports_out)
                return _FP8_CAP
        except Exception:
            continue
    _FP8_CAP = ("off", False)
    return _FP8_CAP


def _get_cached_w_fp8(weight: torch.Tensor, mode: str) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Lazily quantize & cache BF16 weight [N, K] -> (w_fp8_T [K, N_pad], w_scale, N_pad)."""
    direct_entries = getattr(weight, "_inkling_fp8_entries", None)
    if isinstance(direct_entries, dict):
        hit = direct_entries.get(mode)
        if hit is not None:
            return hit
    n, k = int(weight.shape[0]), int(weight.shape[1])
    ptr = int(weight.data_ptr())
    key = (ptr, n, k, int(weight.stride(0)), int(weight.stride(1)), mode)
    if ptr != 0:
        cached = _FP8_W_CACHE.get(key)
        if cached is not None:
            return cached
    w_src = _resolve_bf16_weight(weight)
    n_pad = (n + 15) & ~15
    w_f32 = w_src.float()
    if mode == "rowwise":
        amax = w_f32.abs().amax(dim=1)  # [N]
        sc = torch.clamp(amax * (1.0 / _FP8_MAX), min=_FP8_EPS)  # [N]
        w_q = torch.clamp(w_f32 / sc[:, None], -_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
        if n_pad != n:
            w_pad = torch.zeros((n_pad, k), dtype=torch.float8_e4m3fn, device=w_src.device)
            w_pad[:n].copy_(w_q)
            sc_pad = torch.ones((1, n_pad), dtype=torch.float32, device=w_src.device)
            sc_pad[0, :n].copy_(sc)
            w_fp8, w_scale = w_pad, sc_pad
        else:
            w_fp8 = w_q.contiguous()
            w_scale = sc.reshape(1, n).contiguous()
    else:
        amax = w_f32.abs().amax()
        sc = torch.clamp(amax * (1.0 / _FP8_MAX), min=_FP8_EPS).reshape(1)
        w_q = torch.clamp(w_f32 / sc, -_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
        if n_pad != n:
            w_pad = torch.zeros((n_pad, k), dtype=torch.float8_e4m3fn, device=w_src.device)
            w_pad[:n].copy_(w_q)
            w_fp8 = w_pad
        else:
            w_fp8 = w_q.contiguous()
        w_scale = sc.contiguous()
    # Keep transpose view [K, N_pad] (col-major contiguous!) cached to save view overhead.
    entry = (w_fp8.t(), w_scale, n_pad)
    if ptr != 0:
        _FP8_W_CACHE[key] = entry
    try:
        if not isinstance(direct_entries, dict):
            direct_entries = {}
            weight._inkling_fp8_entries = direct_entries  # type: ignore[attr-defined]
        direct_entries[mode] = entry
    except Exception:
        pass
    return entry


def pf_dense_fp8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    mode_override: str | None = None,
    min_n: int | None = None,
) -> torch.Tensor | None:
    """Dynamic FP8 dense linear for prefill chunks (M >= 1024) on SM121a.

    Returns None to fall back cleanly to BF16 cuBLAS if unsupported.
    """
    if (
        not x.is_cuda
        or x.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or x.ndim != 2
        or weight.ndim != 2
        or x.stride(1) != 1
        or weight.stride(1) != 1
    ):
        return None
    m, k = int(x.shape[0]), int(x.shape[1])
    n = int(weight.shape[0])
    n_floor = _PF_DENSE_FP8_MIN_N if min_n is None else min_n
    if m <= 0 or int(weight.shape[1]) != k or k < 256 or (k & 15) != 0 or n < n_floor:
        return None
    if out is not None and (out.dtype != torch.bfloat16 or out.ndim != 2 or out.shape != (m, n)):
        return None

    # Wave-V11 314 TFLOPS SM121a BF16-in FP8-f16acc dense GEMM (`A_MAX=22, W_MAX=22`, 0 overflow,
    # eliminates `quantize_fp8_rowwise` `quant_k768` DRAM round-trip completely!)
    if (
        _PF_DENSE_FP8_F16ACC
        and mode_override is None
        and m >= max(_PF_DENSE_FP8_MIN_M, 1024)
        and (k % 64) == 0
        and (n % 8) == 0
        and (out is None or (out.stride(1) == 1 and (out.stride(0) & 7) == 0 and (out.data_ptr() & 15) == 0))
        and not torch.cuda.is_current_stream_capturing()
    ):
        v11_lib = _get_pf_dense_fp8_tc_v11_lib()
        if v11_lib is not None:
            w_fp8_v11, w_sc_v11 = _get_cached_w_fp8_v11_f16safe(weight, _V11_W_MAX_WO_UD)
            x_c = x if ((x.stride(0) & 7) == 0 and (x.data_ptr() & 15) == 0) else x.contiguous()
            out_buf = out if out is not None else torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
            rc_v11 = v11_lib.pf_wo_ud_bf16_to_bf16_v11_fwd(
                x_c.data_ptr(),
                w_fp8_v11.data_ptr(),
                w_sc_v11.data_ptr(),
                out_buf.data_ptr(),
                m,
                n,
                k,
                int(x_c.stride(0)),
                int(out_buf.stride(0)),
                float(_V11_A_MAX_WO_UD),
                int(_PF_DENSE_FP8_RESERVE_SMS),
                int(torch.cuda.current_stream(x.device).cuda_stream),
            )
            if rc_v11 == 0:
                return out_buf

    if mode_override is not None:
        mode = mode_override
        supports_out = True
    else:
        mode, supports_out = _probe_fp8_cap(x.device)
    if mode == "off":
        return None

    w_fp8_t, w_scale, n_pad = _get_cached_w_fp8(weight, mode)
    if mode == "rowwise":
        x_fp8, x_scale = quantize_fp8_rowwise(x, pad_m16=True, reuse_scratch=True)
        fast_accum = False
    else:
        x_fp8, x_scale = quantize_fp8_tensorwise(x, pad_m16=True)
        fast_accum = True

    m_pad = int(x_fp8.shape[0])
    can_direct_out = (
        out is not None
        and supports_out
        and m_pad == m
        and n_pad == n
        and out.is_contiguous()
    )
    try:
        if can_direct_out:
            torch._scaled_mm(
                x_fp8, w_fp8_t, scale_a=x_scale, scale_b=w_scale,
                out_dtype=torch.bfloat16, use_fast_accum=fast_accum, out=out,
            )
            return out
        res = torch._scaled_mm(
            x_fp8, w_fp8_t, scale_a=x_scale, scale_b=w_scale,
            out_dtype=torch.bfloat16, use_fast_accum=fast_accum,
        )
    except Exception:
        return None

    if m_pad != m or n_pad != n:
        res = res[:m, :n]
    if out is not None:
        out.copy_(res)
        return out
    return res


def _pdl_site(x, weight, out, *, for_bf11: bool = False) -> bool:
    if not _PDL:
        return False
    n, k = weight.shape
    if (_PDL_WO or (for_bf11 and _PDL_WO_BF11)) and n == _PDL_WO_N and k == _PDL_WO_K:
        # wo_ud in TP decode on GB10 (out=None or out=sbuf). Only when the preceding
        # merge_gate_sink was the PDL trigger (sm12_attn._WO_ARMED, set at trace/capture
        # time); the drafter o_proj has the same shape but no triggering predecessor.
        from . import sm12_attn
        return sm12_attn._WO_ARMED
    if out is None:
        return _PDL_QKVR and k == 4608 and n in _PDL_QKVR_N
    return False


def linear(
    x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor | None = None
) -> torch.Tensor:
    """bf16 ``x @ weight.T`` with fp32 accumulation (into ``out`` if given)."""
    m = x.shape[0]
    _is_wo_shape = weight.dim() == 2 and int(weight.shape[0]) == _PDL_WO_N and int(weight.shape[1]) == _PDL_WO_K
    if m == 0:
        if _is_wo_shape and (_PDL_WO or _PDL_WO_BF11):
            try:
                from . import sm12_attn
                sm12_attn._WO_ARMED = False
            except Exception:
                pass
        if out is not None:
            return out
        out_dt = torch.bfloat16 if x.dtype == torch.float8_e4m3fn else x.dtype
        return torch.empty((0, int(weight.shape[0])), dtype=out_dt, device=x.device)
    if x.dtype == torch.float8_e4m3fn:
        x_scale = getattr(x, "_inkling_fp8_scale", None)
        if x_scale is None:
            raise ValueError("FP8 tensor passed to ops.linear.linear without _inkling_fp8_scale")
        if _PF_DENSE_FP8 and not torch.cuda.is_current_stream_capturing():
            _fp8_ok, _n_dec = _classify_pf_decode_batch(m)
            if _fp8_ok and _n_dec == 0:
                res = pf_dense_fp8_linear_prequant(x, x_scale, weight, out)
                if res is not None:
                    if _is_wo_shape and (_PDL_WO or _PDL_WO_BF11):
                        from . import sm12_attn
                        sm12_attn._WO_ARMED = False
                    return res
        x = dequantize_fp8_rowwise(x, x_scale)
    if _SKINNY and m <= 16:
        from . import skinny_bf11, skinny_bf12
        from .skinny import skinny_linear, supported
        if supported(x, weight) and (
            out is None
            or (
                out.is_cuda
                and out.dim() == 2
                and out.shape == (m, int(weight.shape[0]))
                and out.dtype == x.dtype
                and out.stride(1) == 1
            )
        ):
            pdl_bf11 = _pdl_site(x, weight, out, for_bf11=True)
            pdl = _pdl_site(x, weight, out, for_bf11=False)
            if _is_wo_shape and (_PDL_WO or (_PDL_WO_BF11 and skinny_bf11.ENABLED) or skinny_bf11.V12_ENABLED):
                from . import sm12_attn
                sm12_attn._WO_ARMED = False  # one-shot: consumed/cleared by this wo_ud launch
            if skinny_bf11.ENABLED:
                trig = pdl_bf11 and (out is not None or weight.shape[1] == 4608)
                res = skinny_bf11.bf11_linear(x, weight, out, pdl=pdl_bf11, trig=trig)
                if res is not None:
                    return res
            if skinny_bf12.ENABLED:
                res = skinny_bf12.bf12_linear(x, weight, out, pdl=pdl, trig_reduce=pdl)
                if res is not None:
                    return res
            return skinny_linear(x, weight, out, pdl=pdl, trig_reduce=pdl)
    elif _SKINNY and m > 16:
        from . import skinny_bf11
        if (
            skinny_bf11.ENABLED
            and skinny_bf11.C32_ENABLED
            and m <= skinny_bf11.C32_MAX_M
            and x.is_cuda
            and weight.is_cuda
            and x.dim() == 2
            and weight.dim() == 2
            and int(x.shape[1]) == int(weight.shape[1])
            and x.dtype == torch.bfloat16
            and weight.dtype == torch.bfloat16
            and x.stride(1) == 1
            and weight.stride(1) == 1
            and (
                out is None
                or (
                    out.is_cuda
                    and out.dim() == 2
                    and out.shape == (m, int(weight.shape[0]))
                    and out.dtype == x.dtype
                    and out.stride(1) == 1
                )
            )
        ):
            pdl = _pdl_site(x, weight, out, for_bf11=True)
            if _is_wo_shape and (_PDL_WO or _PDL_WO_BF11 or skinny_bf11.V12_ENABLED):
                from . import sm12_attn
                sm12_attn._WO_ARMED = False
            trig = pdl and (out is not None or weight.shape[1] == 4608)
            res = skinny_bf11.bf11_linear(x, weight, out, pdl=pdl, trig=trig)
            if res is not None:
                return res
    # Wave-V15 fix: clear one-shot _WO_ARMED on ANY wo_ud fallback path (e.g. M>64 C=32 M=128
    # or eager prefill) so it can NEVER leak into a subsequent drafter o_proj call of shape (4608, 768)!
    if _is_wo_shape and (_PDL_WO or _PDL_WO_BF11):
        try:
            from . import sm12_attn
            sm12_attn._WO_ARMED = False
        except Exception:
            pass
    if (
        _PF_DENSE_FP8
        and m >= _PF_DENSE_FP8_MIN_M
        and not torch.cuda.is_current_stream_capturing()
    ):
        _fp8_ok, _n_dec = _classify_pf_decode_batch(m)
        if _fp8_ok and _n_dec == 0:
            res = pf_dense_fp8_linear(x, weight, out)
            if res is not None:
                return res
        elif _fp8_ok and 0 < _n_dec < m and (m - _n_dec) >= _PF_DENSE_FP8_MIN_M:
            w_bf16 = _resolve_bf16_weight(weight)
            out_buf = out if out is not None else torch.empty((m, int(weight.shape[0])), dtype=x.dtype, device=x.device)
            torch.mm(x[:_n_dec], w_bf16.t(), out=out_buf[:_n_dec])
            res_pf = pf_dense_fp8_linear(x[_n_dec:], weight, out=out_buf[_n_dec:])
            if res_pf is not None:
                return out_buf
            torch.mm(x[_n_dec:], w_bf16.t(), out=out_buf[_n_dec:])
            return out_buf
    weight = _resolve_bf16_weight(weight)
    if out is not None:
        return torch.mm(x, weight.t(), out=out)
    # Autotuning would profile the CuTe DSL kernels past their 32-row limit.
    if (x.shape[0] > LOW_M_MAX_TOKENS or is_flashinfer_autotuning()
            or _SM12):
        return torch.nn.functional.linear(x, weight)
    from flashinfer import mm_bf16

    return mm_bf16(x, weight.t(), pdl=True, backend="cute-dsl")


@triton.jit
def _k_group_tile(
    X,
    W,
    Y,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    SPLIT: tl.constexpr,
    pid_n,
    pid_m,
    pid_k,
):
    # Row tiles of at most 64 keep tl.dot on the same mma.sync instruction
    # sequence as the reference's 16x32x32 tiling. With SPLIT, each program computes one
    # K group's partial; otherwise it sums the groups in order.
    cols = pid_n * BN + tl.arange(0, BN)
    rows = pid_m * BM + tl.arange(0, BM)
    ks = tl.arange(0, 128)
    first = pid_k if SPLIT else 0
    NUM_GROUPS: tl.constexpr = 1 if SPLIT else (K + 1023) // 1024
    total = tl.zeros((BM, BN), tl.float32)
    for g in tl.static_range(NUM_GROUPS):
        group = first + g
        partial = tl.zeros((BM, BN), tl.float32)
        for tile in range(8):
            k = group * 1024 + tile * 128 + ks
            x = tl.load(
                X + rows[:, None] * K + k[None, :],
                (rows[:, None] < M) & (k[None, :] < K),
                other=0,
            )
            w = tl.load(
                W + cols[None, :] * K + k[:, None],
                (cols[None, :] < N) & (k[:, None] < K),
                other=0,
            )
            partial = tl.dot(x, w, partial)
        total = partial if g == 0 else total + partial
    tl.store(
        Y + (first * M + rows[:, None]) * N + cols[None, :],
        total.to(Y.dtype.element_ty),
        (rows[:, None] < M) & (cols[None, :] < N),
    )


@triton.jit
def _prefetch_k_group_tile(
    W, N: tl.constexpr, K: tl.constexpr, BN: tl.constexpr, pid_n, pid_k
):
    # Pulls one K group's weight tile into L2, one 128-byte line per element.
    cols = pid_n * BN + tl.arange(0, BN)
    k = pid_k * 1024 + tl.arange(0, 16) * 64
    mask = (cols[:, None] < N) & (k[None, :] < K)
    tl.inline_asm_elementwise(
        "{ .reg .pred p; setp.ne.b32 p, $2, 0; "
        "@p prefetch.global.L2 [$1]; mov.b32 $0, 0; }",
        "=r,l,r",
        [W + cols[:, None] * K + k[None, :], mask.to(tl.int32)],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _k_group_kernel(
    X,
    W,
    Y,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    SPLIT: tl.constexpr,
):
    if SPLIT and tl.program_id(1) == 0:
        _prefetch_k_group_tile(W, N, K, BN, tl.program_id(0), tl.program_id(2))
    tl.extra.cuda.gdc_wait()
    _k_group_tile(
        X,
        W,
        Y,
        M,
        N,
        K,
        BM,
        BN,
        SPLIT,
        tl.program_id(0),
        tl.program_id(1),
        tl.program_id(2),
    )


@triton.jit
def _k_group_sum_kernel(Partials, Y, MN, GROUPS: tl.constexpr, BLOCK: tl.constexpr):
    tl.extra.cuda.gdc_wait()
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < MN
    total = tl.load(Partials + offs, mask)
    for group in range(1, GROUPS):
        total = total + tl.load(Partials + group * MN + offs, mask)
    tl.store(Y + offs, total, mask)


@triton.jit
def _k_group_tma_kernel(
    XDesc,
    WDesc,
    Y,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    LAUNCH_DEPENDENTS: tl.constexpr,
):
    # One output tile over all of K: each group's partial accumulates on the
    # tensor cores and is added to the running sum in order.
    TILES: tl.constexpr = (K + BK - 1) // BK
    GROUP_TILES: tl.constexpr = 1024 // BK
    rows = tl.program_id(1) * BM
    cols = tl.program_id(0) * BN
    tl.extra.cuda.gdc_wait()
    if LAUNCH_DEPENDENTS:
        tl.extra.cuda.gdc_launch_dependents()
    total = tl.zeros((BM, BN), tl.float32)
    partial = tl.zeros((BM, BN), tl.float32)
    for tile in tl.range(TILES, warp_specialize=True):
        x = XDesc.load([rows, tile * BK])
        w = WDesc.load([cols, tile * BK])
        partial = tl.dot(x, w.T, partial)
        if tile % GROUP_TILES == GROUP_TILES - 1 or tile == TILES - 1:
            total = total + partial
            partial = tl.zeros((BM, BN), tl.float32)
    r = rows + tl.arange(0, BM)[:, None]
    c = cols + tl.arange(0, BN)[None, :]
    tl.store(Y + r * N + c, total.to(Y.dtype.element_ty), (r < M) & (c < N))


# From this many rows, 128-row TMA tiles over all of K fill the GPU.
TMA_MIN_TOKENS = 2 ** 31  # sm121: TMA tiles exceed the 99KB shared-memory limit


def k_group_partials(
    x: torch.Tensor,
    weight: torch.Tensor,
    out_dtype: torch.dtype = torch.float32,
    launch_dependents: bool = False,
) -> torch.Tensor:
    """sm121: the Triton k-group and TMA kernels both exceed the 99 KB
    shared-memory limit of GB10. Compute the same fp32 partials (G=1,
    fully summed) with a tensor-core matmul: bf16 inputs accumulate in
    fp32, matching the reference's precision up to summation order.

    When ``INKLING_PF_ROUTER_FP8=1`` and ``M >= INKLING_PF_ROUTER_FP8_MIN_M``
    (prefill only, never inside CUDA graph capture), routes through SM121a
    FP8 ``torch._scaled_mm`` — consuming pre-quantized ``(x_fp8, x_scale)``
    directly with ZERO dequant/re-quant when fed from FP8-Direct SP AllGather!"""
    m = int(x.shape[0])
    target_dtype = torch.bfloat16 if out_dtype == torch.float8_e4m3fn else out_dtype
    if m == 0:
        return torch.empty((1, 0, int(weight.shape[0])), dtype=target_dtype, device=x.device)
    if x.dtype == torch.float8_e4m3fn:
        x_scale = getattr(x, "_inkling_fp8_scale", None)
        if x_scale is None:
            raise ValueError("FP8 tensor passed to k_group_partials without _inkling_fp8_scale")
        if (
            _PF_ROUTER_FP8
            and m >= _PF_ROUTER_FP8_MIN_M
            and not torch.cuda.is_current_stream_capturing()
        ):
            _rfp8_ok, _rn_dec = _classify_pf_decode_batch(m)
            if _rfp8_ok and _rn_dec == 0:
                res = pf_dense_fp8_linear_prequant(x, x_scale, weight, min_n=16)
                if res is not None:
                    if res.dtype != target_dtype:
                        res = res.to(target_dtype)
                    return res.unsqueeze(0)
        x = dequantize_fp8_rowwise(x, x_scale)
    elif (
        _PF_ROUTER_FP8
        and x.dtype == torch.bfloat16
        and m >= _PF_ROUTER_FP8_MIN_M
        and not torch.cuda.is_current_stream_capturing()
    ):
        _rfp8_ok, _rn_dec = _classify_pf_decode_batch(m)
        if _rfp8_ok and _rn_dec == 0:
            res = pf_dense_fp8_linear(x, weight, min_n=16)
            if res is not None:
                if res.dtype != target_dtype:
                    res = res.to(target_dtype)
                return res.unsqueeze(0)
        elif _rfp8_ok and 0 < _rn_dec < m and (m - _rn_dec) >= _PF_ROUTER_FP8_MIN_M:
            w_bf16 = _resolve_bf16_weight(weight)
            out_r = torch.empty((m, int(weight.shape[0])), dtype=torch.bfloat16, device=x.device)
            torch.mm(x[:_rn_dec], w_bf16.t(), out=out_r[:_rn_dec])
            res_pf = pf_dense_fp8_linear(x[_rn_dec:], weight, out=out_r[_rn_dec:], min_n=16)
            if res_pf is None:
                torch.mm(x[_rn_dec:], w_bf16.t(), out=out_r[_rn_dec:])
            if out_r.dtype != target_dtype:
                out_r = out_r.to(target_dtype)
            return out_r.unsqueeze(0)
    weight = _resolve_bf16_weight(weight)
    out = torch.matmul(x, weight.t())
    if out.dtype != target_dtype:
        out = out.to(target_dtype)
    return out.unsqueeze(0)


def _k_group_partials_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    out_dtype: torch.dtype = torch.float32,
    launch_dependents: bool = False,
) -> torch.Tensor:
    """fp32 partials [G, T, N] of ``x @ weight.T`` whose in-order sum over G is
    the reference ``linear``. G is 1 (already summed, in ``out_dtype``) when
    there are enough rows to fill the GPU without splitting K. With
    ``launch_dependents``, the next kernel may start (and must wait) while
    the TMA kernel runs."""
    assert x.is_contiguous() and weight.is_contiguous()
    tokens, hidden = x.shape
    n = weight.shape[0]
    if tokens >= TMA_MIN_TOKENS:
        from triton.tools.tensor_descriptor import TensorDescriptor

        out = torch.empty((1, tokens, n), dtype=out_dtype, device=x.device)
        block_n = 64 if tokens < 4096 else 128
        _k_group_tma_kernel[(triton.cdiv(n, block_n), triton.cdiv(tokens, 128))](
            TensorDescriptor.from_tensor(x, [128, 128]),
            TensorDescriptor.from_tensor(weight, [block_n, 128]),
            out,
            tokens,
            n,
            hidden,
            128,
            block_n,
            128,
            launch_dependents,
            num_warps=4 if block_n == 64 else 8,
            num_stages=4 if block_n == 64 else 3,
            enable_fp_fusion=False,
            launch_pdl=True,
        )
        return out
    block_m = 16 if tokens <= 128 else 64
    block_n = 32 if tokens <= 16 else 64 if tokens <= 256 else 128
    groups = triton.cdiv(hidden, 1024)
    split = groups > 1 and tokens < 4096
    num_warps = 4 if block_n <= 64 else 8
    num_stages = 4 if not split else 3 if block_n <= 64 else 2
    out = torch.empty(
        (groups if split else 1, tokens, n),
        dtype=torch.float32 if split else out_dtype,
        device=x.device,
    )
    grid = (
        triton.cdiv(n, block_n),
        triton.cdiv(tokens, block_m),
        groups if split else 1,
    )
    _k_group_kernel[grid](
        x,
        weight,
        out,
        tokens,
        n,
        hidden,
        block_m,
        block_n,
        split,
        num_warps=num_warps,
        num_stages=num_stages,
        enable_fp_fusion=False,
        launch_pdl=True,
    )
    return out


def linear_k_groups(
    x: torch.Tensor, weight: torch.Tensor, out_dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """``x @ weight.T`` for bf16 ``x`` [T, K] and ``weight`` [N, K].

    Each 1024-wide K group is accumulated in fp32 and the partials are summed in
    order, as in the reference ``linear``.
    """
    partials = k_group_partials(x, weight, out_dtype)
    if partials.shape[0] == 1:
        return partials[0]
    out = torch.empty(partials.shape[1:], dtype=out_dtype, device=x.device)
    _k_group_sum_kernel[(triton.cdiv(out.numel(), 1024),)](
        partials, out, out.numel(), partials.shape[0], 1024, launch_pdl=True
    )
    return out
