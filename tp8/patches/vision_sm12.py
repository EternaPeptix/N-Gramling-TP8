# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 vision encoder: a PE-Lang-style ViT with a 2x2 merge adapter.

Module names mirror the checkpoint, so ``model.visual.<name>`` loads into
``Inkling2VisionEncoder.<name>`` unchanged, except that each layer's
``q_proj``/``k_proj``/``v_proj`` load into one fused ``qkv_proj``.
"""

from collections import defaultdict
from collections.abc import Iterable, Mapping
import os
from typing import Any, NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.models.inkling2.ops.vision import (
    apply_rotary_qk_,
    embed_tokens,
    gather_patches,
    scaled_residual_norm_,
)

try:
    from vllm.third_party.tml_fa4 import flash_attn_varlen_func
except ImportError:  # pragma: no cover - SM12x uses SDPA path
    flash_attn_varlen_func = None  # type: ignore[assignment]

_SM12 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12
_TRUE_VALS = ("1", "true", "on", "yes")
_FORCE_SDPA = os.environ.get("INKLING_VISION_FORCE_SDPA", "0").strip().lower() in _TRUE_VALS
# Avoid pinning variable-sized metadata buffers on GB10 UMA (prevents CachingHostAllocator
# pinned-page fragmentation across 1,000+ diverse image resolutions at GMU=0.85).
_PIN_META = os.environ.get(
    "INKLING_VISION_PIN_META", "0" if _SM12 else "1"
).strip().lower() in _TRUE_VALS
_SDPA_CHUNK_Q = max(64, int(os.environ.get("INKLING_VISION_SDPA_CHUNK_Q", "1024") or "1024"))
_SDPA_MAX_ATTN_BYTES = (
    max(4, int(os.environ.get("INKLING_VISION_SDPA_MAX_ATTN_MB", "64") or "64")) * (1 << 20)
)
_SDPA_KV_CHUNK = max(64, int(os.environ.get("INKLING_VISION_SDPA_KV_CHUNK", "1024") or "1024"))
_SDPA_FAST_ONLY_MIN_S = max(
    256, int(os.environ.get("INKLING_VISION_SDPA_FAST_MIN_S", "4096") or "4096")
)
# Token-chunk MLP and 2x2 merge projector on huge multi-image / 4K batches so peak
# intermediate activations stay <~80 MiB at GMU=0.85 (0-ULP identical elementwise).
_MLP_CHUNK = max(0, int(os.environ.get("INKLING_VISION_MLP_CHUNK", "4096") or "4096"))
# Release large transient caching-allocator segments after giant 4K/multi-image bursts
# (strictly skipped during CUDA graph capture; 0 disables).
_RECLAIM_MIN_PATCHES = max(
    0, int(os.environ.get("INKLING_VISION_RECLAIM_MIN_PATCHES", "4096") or "4096")
)

_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
# The image processor's CLIP std (float32-rounded constants): pixels are first
# CLIP-normalized with it and rounded to bf16, then mapped to [-1, 1].
_PROCESSOR_CLIP_STD = (0.26862954, 0.2613026, 0.2757771)


def _linear(in_features: int, out_features: int, bias: bool = True) -> nn.Linear:
    return nn.Linear(in_features, out_features, bias=bias, dtype=torch.bfloat16)


def _is_capturing() -> bool:
    return bool(
        torch.cuda.is_available()
        and hasattr(torch.cuda, "is_current_stream_capturing")
        and torch.cuda.is_current_stream_capturing()
    )


def _streaming_sdpa_fp32(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    kv_chunk: int = _SDPA_KV_CHUNK,
) -> torch.Tensor:
    """Memory-bounded online-softmax attention in FP32 (O(H * BQ * BK) scratch).

    Used as an SM121a safety fallback when fast SDPA kernels reject a shape (e.g.
    99 KiB SMEM limit or non-standard head_dim) or when S is too large for O(S^2)
    MATH attention under GMU=0.85 UMA limits. Guarantees zero BF16 softmax overflow
    and zero NaNs even for extreme resolutions or magnitudes.
    """
    q_f = q.float()
    seq_k = int(k.shape[2])
    step = max(64, int(kv_chunk))
    m_i: torch.Tensor | None = None
    l_i: torch.Tensor | None = None
    acc: torch.Tensor | None = None
    for k0 in range(0, seq_k, step):
        k1 = min(seq_k, k0 + step)
        k_blk = k[:, :, k0:k1, :].float()
        v_blk = v[:, :, k0:k1, :].float()
        scores = torch.matmul(q_f, k_blk.transpose(-2, -1)) * scale
        del k_blk
        blk_max = scores.amax(dim=-1, keepdim=True)
        if m_i is None:
            m_i = blk_max
            p = torch.exp(scores - m_i)
            l_i = p.sum(dim=-1, keepdim=True)
            acc = torch.matmul(p, v_blk)
            del scores, p, v_blk
        else:
            m_new = torch.maximum(m_i, blk_max)
            diff_m = torch.where(torch.isfinite(m_new), m_i - m_new, torch.zeros_like(m_new))
            alpha = torch.exp(diff_m)
            p = torch.exp(scores - m_new)
            assert l_i is not None and acc is not None
            l_i = l_i * alpha + p.sum(dim=-1, keepdim=True)
            acc = acc * alpha + torch.matmul(p, v_blk)
            m_i = m_new
            del scores, blk_max, diff_m, alpha, p, v_blk
    assert l_i is not None and acc is not None
    out = (acc / l_i.clamp(min=1e-30)).to(q.dtype)
    del q_f, m_i, l_i, acc
    return out


def _try_fast_sdpa(
    q_4d: torch.Tensor,
    k_4d: torch.Tensor,
    v_4d: torch.Tensor,
    scale: float,
) -> torch.Tensor | None:
    """Run O(S)-memory fast SDPA backends only (no O(S^2) MATH allocation)."""
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except ImportError:
        return None
    fast_backends = [
        b
        for name in ("EFFICIENT_ATTENTION", "CUDNN_ATTENTION", "FLASH_ATTENTION")
        if (b := getattr(SDPBackend, name, None)) is not None
    ]
    if not fast_backends:
        return None
    try:
        with sdpa_kernel(fast_backends):
            return F.scaled_dot_product_attention(q_4d, k_4d, v_4d, scale=scale)
    except RuntimeError:
        return None


def _sdpa_sm12_single(
    q_img: torch.Tensor,
    k_img: torch.Tensor,
    v_img: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """SM121a-safe 4D contiguous SDPA for one image slice ``[S, H, D]``.

    - Passes 4D contiguous ``[1, H, S, D]`` tensors so fast CUDA SDPA backends do
      not reject 3D or 3x-interleaved QKV views.
    - For large ``S > _SDPA_FAST_ONLY_MIN_S`` (2K/4K images), tries O(S)-memory
      fast backends first and falls back to O(BQ*BK) FP32 streaming online-softmax
      instead of letting MATH SDPA allocate multi-GB ``[H, S, S]`` matrices.
    - For ``S <= _SDPA_FAST_ONLY_MIN_S``, bounds query chunk size so even MATH
      fallback stays within ``_SDPA_MAX_ATTN_BYTES`` (default 64 MiB), and recovers
      from any SM121a SMEM/kernel ``RuntimeError`` via ``_streaming_sdpa_fp32``.
    """
    seq_len, num_heads, head_dim = q_img.shape
    if seq_len == 0:
        return torch.empty((0, num_heads, head_dim), dtype=q_img.dtype, device=q_img.device)

    k_4d = k_img.transpose(0, 1).unsqueeze(0).contiguous()
    v_4d = v_img.transpose(0, 1).unsqueeze(0).contiguous()
    try:
        if seq_len > _SDPA_FAST_ONLY_MIN_S:
            q_4d = q_img.transpose(0, 1).unsqueeze(0).contiguous()
            fast_out = _try_fast_sdpa(q_4d, k_4d, v_4d, scale)
            if fast_out is not None:
                del q_4d
                return fast_out.squeeze(0).transpose(0, 1).contiguous()
            del q_4d
            out_img = torch.empty(
                (seq_len, num_heads, head_dim), dtype=q_img.dtype, device=q_img.device
            )
            stream_bq = min(256, seq_len)
            for q0 in range(0, seq_len, stream_bq):
                q1 = min(seq_len, q0 + stream_bq)
                q_blk = q_img[q0:q1].transpose(0, 1).unsqueeze(0).contiguous()
                o_blk = _streaming_sdpa_fp32(q_blk, k_4d, v_4d, scale, _SDPA_KV_CHUNK)
                out_img[q0:q1] = o_blk.squeeze(0).transpose(0, 1)
                del q_blk, o_blk
            return out_img

        max_q_by_mem = max(64, _SDPA_MAX_ATTN_BYTES // max(1, num_heads * seq_len * 4))
        q_chunk = min(seq_len, _SDPA_CHUNK_Q, max_q_by_mem)
        if q_chunk >= seq_len:
            q_4d = q_img.transpose(0, 1).unsqueeze(0).contiguous()
            try:
                o_4d = F.scaled_dot_product_attention(q_4d, k_4d, v_4d, scale=scale)
            except RuntimeError:
                o_4d = _streaming_sdpa_fp32(q_4d, k_4d, v_4d, scale, _SDPA_KV_CHUNK)
            del q_4d
            return o_4d.squeeze(0).transpose(0, 1).contiguous()

        out_img = torch.empty(
            (seq_len, num_heads, head_dim), dtype=q_img.dtype, device=q_img.device
        )
        for q0 in range(0, seq_len, q_chunk):
            q1 = min(seq_len, q0 + q_chunk)
            q_blk = q_img[q0:q1].transpose(0, 1).unsqueeze(0).contiguous()
            try:
                o_blk = F.scaled_dot_product_attention(q_blk, k_4d, v_4d, scale=scale)
            except RuntimeError:
                o_blk = _streaming_sdpa_fp32(q_blk, k_4d, v_4d, scale, _SDPA_KV_CHUNK)
            out_img[q0:q1] = o_blk.squeeze(0).transpose(0, 1)
            del q_blk, o_blk
        return out_img
    finally:
        del k_4d, v_4d


class VisionMetadata(NamedTuple):
    # int32 [T, 4] (y, x, height, width) in native patches; CLS: y = x = -1.
    meta: torch.Tensor
    # Per token row, its row in the patch embeddings (unused on CLS rows).
    patch_rows: torch.Tensor
    # Per native patch, the offset of its top-left pixel.
    pixel_offsets: torch.Tensor
    # Token rows in 2x2-merge order, for the adapter.
    pool_rows: torch.Tensor
    cu_seqlens: torch.Tensor
    max_seqlen: int


class Inkling2VisionAttention(nn.Module):
    def __init__(self, width: int, num_heads: int) -> None:
        super().__init__()
        if num_heads <= 0 or width % num_heads != 0:
            raise ValueError(
                f"Vision width ({width}) must be divisible by num_heads ({num_heads})"
            )
        self.num_heads = num_heads
        self.head_dim = width // num_heads
        self.qkv_proj = _linear(width, 3 * width)
        self.out_proj = _linear(width, width)

    def forward(
        self,
        hidden: torch.Tensor,
        metadata: VisionMetadata,
        cu_seqlens_cpu: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden)
        apply_rotary_qk_(qkv, metadata.meta, self.num_heads)
        q, k, v = qkv.view(-1, 3, self.num_heads, self.head_dim).unbind(1)
        scale = float(self.head_dim) ** -0.5
        if _SM12 or _FORCE_SDPA:
            # tml_fa4 CuTe varlen does not compile on SM12x; use SM121a-safe
            # per-image 4D SDPA with pre-resolved CPU cu_seqlens (0 per-layer D2H syncs).
            if _is_capturing():
                raise RuntimeError(
                    "Inkling2VisionEncoder SM12x SDPA path does not support CUDA graph "
                    "capture; disable compilation_config.cudagraph_mm_encoder on SM12x."
                )
            total_tokens = int(q.shape[0])
            out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
            cu = (
                cu_seqlens_cpu
                if cu_seqlens_cpu is not None
                else tuple(int(x) for x in metadata.cu_seqlens.tolist())
            )
            covered_end = 0
            if len(cu) >= 2:
                first_start = cu[0]
                if first_start < 0 or first_start > total_tokens:
                    raise ValueError(
                        f"Invalid cu_seqlens start {first_start} for T={total_tokens}"
                    )
                if first_start > 0:
                    out[:first_start].zero_()
                covered_end = first_start
                for a, b in zip(cu[:-1], cu[1:]):
                    if a < covered_end or b < a or b > total_tokens:
                        raise ValueError(
                            f"Non-monotonic or out-of-bounds cu_seqlens segment "
                            f"[{a}, {b}) for T={total_tokens}"
                        )
                    if a > covered_end:
                        out[covered_end:a].zero_()
                    if a == b:
                        covered_end = b
                        continue
                    out[a:b] = _sdpa_sm12_single(q[a:b], k[a:b], v[a:b], scale)
                    covered_end = b
            if covered_end < total_tokens:
                out[covered_end:].zero_()
            del q, k, v, qkv
            proj = self.out_proj(out.view(hidden.shape))
            del out
            return proj
        if flash_attn_varlen_func is None:
            raise RuntimeError(
                "vllm.third_party.tml_fa4.flash_attn_varlen_func is unavailable; "
                "set INKLING_VISION_FORCE_SDPA=1 to use the SM12 SDPA path."
            )
        out, _ = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=metadata.cu_seqlens,
            cu_seqlens_k=metadata.cu_seqlens,
            max_seqlen_q=metadata.max_seqlen,
            max_seqlen_k=metadata.max_seqlen,
            softmax_scale=scale,
            causal=False,
        )
        del q, k, v, qkv
        proj = self.out_proj(out.view(hidden.shape))
        del out
        return proj


class Inkling2VisionMLP(nn.Module):
    def __init__(self, width: int, intermediate_size: int) -> None:
        super().__init__()
        self.c_fc = _linear(width, intermediate_size)
        self.c_proj = _linear(intermediate_size, width)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        total_tokens = int(hidden.shape[0])
        if _MLP_CHUNK > 0 and total_tokens > _MLP_CHUNK:
            out = torch.empty_like(hidden)
            for start in range(0, total_tokens, _MLP_CHUNK):
                end = min(total_tokens, start + _MLP_CHUNK)
                x = self.c_fc(hidden[start:end])
                x = F.gelu(x)
                out[start:end] = self.c_proj(x)
                del x
            return out
        x = self.c_fc(hidden)
        x = F.gelu(x)
        out = self.c_proj(x)
        del x
        return out


class Inkling2VisionLayerScale(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.empty(width, dtype=torch.bfloat16))


class Inkling2VisionLayer(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        width, eps = config["hidden_size"], config["layer_norm_eps"]
        self.ln_1 = nn.LayerNorm(width, eps=eps, dtype=torch.bfloat16)
        self.self_attn = Inkling2VisionAttention(width, config["num_attention_heads"])
        self.ln_2 = nn.LayerNorm(width, eps=eps, dtype=torch.bfloat16)
        self.mlp = Inkling2VisionMLP(width, int(width * config["mlp_ratio"]))
        self.ls_1 = Inkling2VisionLayerScale(width)
        self.ls_2 = Inkling2VisionLayerScale(width)

    def forward(
        self,
        hidden: torch.Tensor,
        normed: torch.Tensor,
        metadata: VisionMetadata,
        next_norm: nn.LayerNorm | None,
        cu_seqlens_cpu: tuple[int, ...] | None = None,
    ) -> torch.Tensor | None:
        """Updates the residual ``hidden`` in place; ``normed = ln_1(hidden)``.

        Returns ``next_norm(hidden)`` (the next layer's ``ln_1``), if any.
        """
        attn = self.self_attn(normed, metadata, cu_seqlens_cpu=cu_seqlens_cpu)
        normed = scaled_residual_norm_(hidden, attn, self.ls_1.gamma, self.ln_2)
        del attn
        mlp = self.mlp(normed)
        del normed
        out_norm = scaled_residual_norm_(hidden, mlp, self.ls_2.gamma, next_norm)
        del mlp
        return out_norm


class Inkling2VisionEmbeddings(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        width, grid = config["hidden_size"], config["posemb_grid_size"]
        self.patch_embedding = _linear(
            3 * config["native_patch_size"] ** 2, width, bias=False
        )
        self.position_embedding = nn.Parameter(
            torch.empty(grid * grid, width, dtype=torch.bfloat16)
        )
        self.class_embedding = nn.Parameter(torch.empty(width, dtype=torch.bfloat16))
        self.cls_position_embedding = nn.Parameter(
            torch.empty(width, dtype=torch.bfloat16)
        )


class Inkling2VisionBackbone(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.embeddings = Inkling2VisionEmbeddings(config)
        self.ln_pre = nn.LayerNorm(
            config["hidden_size"], eps=config["layer_norm_eps"], dtype=torch.bfloat16
        )
        self.layers = nn.ModuleList(
            Inkling2VisionLayer(config) for _ in range(config["num_hidden_layers"])
        )


class Inkling2VisionEncoder(nn.Module):
    """Maps raw ``[N, 40, 40, 3]`` bf16 pixel patches (-1 = padding) to decoder
    embeddings: a ViT with a CLS token per image, then a 2x2 merge projection to
    ``text_hidden_size``. Runs replicated on every TP rank. Images are stills,
    so the ``temporal_patch_size`` frames of a patch are equal and their mean is
    the frame itself.
    """

    def __init__(self, config: Mapping[str, Any], text_hidden_size: int) -> None:
        super().__init__()
        self.patch_size = config["native_patch_size"]
        self.merge_size = config["adapter_merge_size"]
        self.posemb_grid_size = config["posemb_grid_size"]
        self.text_hidden_size = text_hidden_size
        self.backbone = Inkling2VisionBackbone(config)
        self.proj_in = _linear(
            self.merge_size**2 * config["hidden_size"],
            config["projection_hidden_dim"],
            bias=config["projector_bias"],
        )
        self.proj_out = _linear(
            config["projection_hidden_dim"],
            text_hidden_size,
            bias=config["projector_bias"],
        )

    def _metadata_with_cu(
        self, grid: list[list[int]], device: torch.device
    ) -> tuple[VisionMetadata, tuple[int, ...]]:
        if not grid:
            empty_i32 = torch.empty((0,), dtype=torch.int32, device=device)
            cu_t = torch.zeros((1,), dtype=torch.int32, device=device)
            return (
                VisionMetadata(
                    empty_i32.view(0, 4), empty_i32, empty_i32, empty_i32, cu_t, 0
                ),
                (0,),
            )

        m, p, side = self.merge_size, self.patch_size, self.merge_size * self.patch_size
        frame = side * side * 3
        meta, patch_rows, offsets, pool_rows, lengths = [], [], [], [], []
        num_tokens = num_patches = num_pixels_rows = 0
        for item in grid:
            if len(item) != 3:
                raise ValueError(f"Expected (t, h, w) in grid_thw, got {item!r}")
            t, h, w = int(item[0]), int(item[1]), int(item[2])
            if t != 1 or h <= 0 or w <= 0:
                raise ValueError(
                    f"Invalid vision grid entry (t={t}, h={h}, w={w}); "
                    "expected t == 1, h > 0, w > 0"
                )
            height, width = h * m, w * m
            y, x = np.divmod(np.arange(height * width, dtype=np.int64), width)
            image_meta = np.empty((height * width + 1, 4), np.int64)
            image_meta[0, :2] = -1
            image_meta[1:, 0], image_meta[1:, 1] = y, x
            image_meta[:, 2], image_meta[:, 3] = height, width
            meta.append(image_meta)
            rows = np.arange(height * width + 1, dtype=np.int64) + num_pixels_rows - 1
            rows[0] = 0
            patch_rows.append(rows)
            patch = num_patches + (y // m) * w + x // m
            offsets.append(patch * frame + (y % m) * (p * side * 3) + (x % m) * (p * 3))
            # [h, w, m, m] merge order of the rows after the CLS row.
            merged = (y // m) * (w * m * m) + (x // m) * (m * m) + (y % m) * m + x % m
            pool = np.empty(height * width, np.int64)
            pool[merged] = num_tokens + 1 + np.arange(height * width)
            pool_rows.append(pool)
            lengths.append(height * width + 1)
            num_tokens += height * width + 1
            num_patches += h * w
            num_pixels_rows += height * width
        cu_seqlens = np.cumsum([0, *lengths], dtype=np.int64)
        cu_cpu = tuple(int(v) for v in cu_seqlens.tolist())
        parts = [
            np.concatenate(meta).ravel(),
            *map(np.concatenate, (patch_rows, offsets, pool_rows)),
            cu_seqlens,
        ]
        packed = np.concatenate(parts)
        if packed.size > 0 and (int(packed.max()) >= 2**31 or int(packed.min()) < -(2**31)):
            raise ValueError(
                f"Vision metadata int32 overflow: range=[{int(packed.min())}, {int(packed.max())}] "
                f"for {num_patches} patches across {len(lengths)} images"
            )
        sizes = [part.size for part in parts]
        packed_cpu = torch.from_numpy(packed.astype(np.int32))
        del packed, parts, meta, patch_rows, offsets, pool_rows
        if device.type == "cuda":
            if _PIN_META and not _is_capturing():
                packed_cpu = packed_cpu.pin_memory()
                packed_t = packed_cpu.to(device, non_blocking=True)
            else:
                packed_t = packed_cpu.to(device)
        else:
            packed_t = packed_cpu.to(device)
        del packed_cpu
        meta_t, patch_rows_t, offsets_t, pool_t, cu_t = packed_t.split(sizes)
        # Same max_seqlen as the reference FA4 launch.
        max_seqlen = max(257, num_tokens - len(lengths) + 1)
        return (
            VisionMetadata(
                meta_t.view(-1, 4), patch_rows_t, offsets_t, pool_t, cu_t, max_seqlen
            ),
            cu_cpu,
        )

    def metadata(self, grid: list[list[int]], device: torch.device) -> VisionMetadata:
        return self._metadata_with_cu(grid, device)[0]

    def forward(self, pixels: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        side = self.patch_size * self.merge_size
        if pixels.dtype != torch.bfloat16:
            raise ValueError(f"Expected bfloat16 vision pixels, got {pixels.dtype}")
        if pixels.ndim != 4 or tuple(pixels.shape[1:]) != (side, side, 3):
            raise ValueError(
                f"Expected pixels shape [N, {side}, {side}, 3], got {tuple(pixels.shape)}"
            )
        if grid_thw.ndim != 2 or grid_thw.shape[1] != 3:
            raise ValueError(
                f"Expected grid_thw shape [B, 3], got {tuple(grid_thw.shape)}"
            )
        grid = grid_thw.tolist()
        if not grid:
            if pixels.shape[0] != 0:
                raise ValueError(
                    f"Empty grid_thw received with non-empty pixels ({pixels.shape[0]} rows)"
                )
            return pixels.new_empty((0, self.text_hidden_size))
        expected_patches = 0
        for item in grid:
            t, h, w = int(item[0]), int(item[1]), int(item[2])
            if t != 1 or h <= 0 or w <= 0:
                raise ValueError(
                    f"Invalid vision grid_thw entry ({t}, {h}, {w}); "
                    "expected t == 1, h > 0, w > 0"
                )
            expected_patches += h * w
        if expected_patches != int(pixels.shape[0]):
            raise ValueError(
                f"Patch count mismatch: sum(h*w)={expected_patches} != pixels.shape[0]={pixels.shape[0]}"
            )

        metadata, cu_cpu = self._metadata_with_cu(grid, pixels.device)
        return self.encode(pixels, metadata, cu_seqlens_cpu=cu_cpu)

    def encode(
        self,
        pixels: torch.Tensor,
        metadata: VisionMetadata,
        cu_seqlens_cpu: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        """Encodes packed patches with fixed-shape (CUDA-graphable) kernels."""
        if int(metadata.meta.shape[0]) == 0 or int(metadata.pool_rows.numel()) == 0:
            return pixels.new_empty((0, self.text_hidden_size))

        if _SM12 or _FORCE_SDPA:
            if _is_capturing():
                raise RuntimeError(
                    "Inkling2VisionEncoder SM12x SDPA path does not support CUDA graph "
                    "capture; disable compilation_config.cudagraph_mm_encoder on SM12x."
                )
            if cu_seqlens_cpu is None:
                cu_seqlens_cpu = tuple(int(x) for x in metadata.cu_seqlens.tolist())

        backbone = self.backbone
        embeddings = backbone.embeddings
        patches = gather_patches(
            pixels.contiguous(),
            metadata.pixel_offsets,
            self.patch_size,
            _CLIP_MEAN,
            _PROCESSOR_CLIP_STD,
            _CLIP_STD,
        )
        patch_embeds = embeddings.patch_embedding(patches)
        del patches
        g = self.posemb_grid_size
        hidden, normed = embed_tokens(
            patch_embeds,
            metadata.patch_rows,
            embeddings.position_embedding.view(g, g, -1),
            metadata.meta,
            embeddings.class_embedding,
            embeddings.cls_position_embedding,
            backbone.ln_pre,
            backbone.layers[0].ln_1,
        )
        del patch_embeds
        layers = backbone.layers
        for i, layer in enumerate(layers):
            next_norm = layers[i + 1].ln_1 if i + 1 < len(layers) else None
            normed = layer(
                hidden, normed, metadata, next_norm, cu_seqlens_cpu=cu_seqlens_cpu
            )
        del normed
        pooled = hidden.index_select(0, metadata.pool_rows)
        del hidden
        pooled = pooled.view(-1, self.merge_size**2 * pooled.shape[-1])
        num_out = int(pooled.shape[0])
        if _MLP_CHUNK > 0 and num_out > _MLP_CHUNK:
            out = pooled.new_empty((num_out, self.text_hidden_size))
            for start in range(0, num_out, _MLP_CHUNK):
                end = min(num_out, start + _MLP_CHUNK)
                proj_h = self.proj_in(pooled[start:end])
                proj_h = F.gelu(proj_h)
                out[start:end] = self.proj_out(proj_h)
                del proj_h
            del pooled
        else:
            proj_h = self.proj_in(pooled)
            del pooled
            proj_h = F.gelu(proj_h)
            out = self.proj_out(proj_h)
            del proj_h
        if (
            _SM12
            and _RECLAIM_MIN_PATCHES > 0
            and int(pixels.shape[0]) >= _RECLAIM_MIN_PATCHES
            and getattr(pixels.device, "type", "") == "cuda"
            and not _is_capturing()
        ):
            torch.cuda.empty_cache()
        return out

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load ``model.visual.*`` checkpoint tensors with that prefix stripped."""
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        qkv_shards: dict[str, set[int]] = defaultdict(set)
        for name, weight in weights:
            for shard, proj in enumerate(("q_proj", "k_proj", "v_proj")):
                if f".self_attn.{proj}." in name:
                    fused_name = name.replace(proj, "qkv_proj")
                    if fused_name not in params:
                        raise ValueError(f"Unexpected vision checkpoint tensor: {name}")
                    param = params[fused_name]
                    size = param.shape[0] // 3
                    default_weight_loader(
                        param.data[shard * size : (shard + 1) * size], weight
                    )
                    qkv_shards[fused_name].add(shard)
                    break
            else:
                if name not in params:
                    raise ValueError(f"Unexpected vision checkpoint tensor: {name}")
                default_weight_loader(params[name], weight)
                loaded.add(name)
        return loaded | {name for name, shards in qkv_shards.items() if len(shards) == 3}
