# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM121 (GB10) replacement for the TRT-LLM expert batched GEMM.

The stock module loads precompiled cubins (sm100/sm103/sm107 only); on
SM121 the manifest has nothing to run. This replacement keeps the exact
``expert_plan`` / ``bmm`` / ``load_kernels`` interface of
``vllm.models.inkling2.ops.trtllm_bmm`` but computes the expert GEMM in
PyTorch: the quantized activations (NVFP4 or MXFP8, from the Triton
route kernel) and the quantized expert weights are dequantized in place
and multiplied in bf16.

Scale conventions (verified against ops/nvfp4.py and ops/moe.py):
  activations: fp4 * e4m3-block-scale * fp32-row-scale (row = amax/1536)
  weights:     fp4 * e4m3-block-scale, with the per-expert outer scale
               (amax / denominator) folded into ``row_scale`` by route()
MXFP8: fp8 value * e8m0-block-scale (2 ** (s - 127)), no row scale.

``tiles`` layout (set by the route kernel): tiles[:m] expert id per tile,
tiles[m:2m] row limit per tile, tiles[2m] live tiles, tiles[2m+1:] rows.
"""

import functools
from dataclasses import dataclass

import torch

# Batch-size -> token tile, identical to the cubin module's table (the
# route and swiglu kernels consume plan.tile, so the values must match).
_TILES = ((8, 8), (128, 16), (384, 32), (768, 64), (4096, 128), (1 << 31, 256))


@dataclass(frozen=True)
class ExpertPlan:
    """Token tile, launched tiles (an upper bound) and the kernel of each
    projection for a batch. ``kernels`` is unused by the torch path."""

    tile: int
    max_ctas: int
    kernels: tuple[int, int]

    @property
    def rows(self) -> int:
        return self.tile * self.max_ctas


@functools.cache
def expert_plan(
    tokens: int,
    topk: int,
    num_experts: int,
    nvfp4: bool,
    shapes: tuple[tuple[int, int], tuple[int, int]],
    tile: int | None = None,
) -> ExpertPlan:
    """Same arithmetic as the cubin module, minus the manifest lookups."""
    if tile is None:
        tile = next(t for max_tokens, t in _TILES if tokens <= max_tokens)
    rows = tokens * topk
    max_ctas = min(num_experts, rows) + (rows - min(num_experts, rows)) // tile
    return ExpertPlan(tile, max_ctas, (0, 0))


def _fp4_table(device, dtype=torch.float32) -> torch.Tensor:
    """E2M1 decode table indexed by nibble (s ee m: sign, exp, mantissa)."""
    t = torch.zeros(16, dtype=dtype)
    for i in range(8):
        e, m = (i >> 1) & 3, i & 1
        t[i] = m * 0.5 if e == 0 else (1.0 + m * 0.5) * float(1 << (e - 1))
        t[i + 8] = -t[i]
    return t.to(device)


def _dequant_nvfp4(
    packed: torch.Tensor, scale: torch.Tensor, out_dtype: torch.dtype
) -> torch.Tensor:
    """[R, K//2] uint8 + [R, K//16] fp8e4m3 -> [R, K] out_dtype."""
    r, kp2 = packed.shape
    table = _fp4_table(packed.device, torch.float32)
    lo = table[(packed & 0xF).long()]
    hi = table[(packed >> 4).long()]
    vals = torch.stack((lo, hi), dim=-1).reshape(r, kp2 * 2)
    return (vals * scale.float().repeat_interleave(16, dim=1)).to(out_dtype)


def _dequant_mxfp8(
    values: torch.Tensor, scale: torch.Tensor, out_dtype: torch.dtype
) -> torch.Tensor:
    """[R, K] fp8e4m3 + [R, K//32] uint8 (E8M0) -> [R, K] out_dtype."""
    scales = torch.ldexp(
        torch.ones((), dtype=torch.float32, device=scale.device),
        scale.int() - 127,
    )
    return (values.float() * scales.repeat_interleave(32, dim=1)).to(out_dtype)


def bmm(
    act: torch.Tensor,
    act_scale: torch.Tensor,
    row_scale: torch.Tensor | None,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    ones: torch.Tensor,
    tiles: torch.Tensor,
    plan: ExpertPlan,
    projection: int,
) -> torch.Tensor:
    """bf16 ``[plan.rows, N]`` expert GEMM of the tile-padded ``act`` rows.

    Torch port of the cubin bmm: dequantize per tile, bf16 matmul, apply
    the row scale (activation outer scale x weight outer scale, folded by
    the route kernel) and the per-expert output scale ``ones``.
    """
    nvfp4 = weight.dtype == torch.uint8
    n = weight.shape[1]
    m = plan.max_ctas
    out = torch.zeros((plan.rows, n), dtype=torch.bfloat16, device=act.device)
    num_tiles = int(tiles[2 * m].item())
    pos = 0
    for t in range(num_tiles):
        e = int(tiles[t].item())
        lim = int(tiles[m + t].item())
        if lim == 0:
            pos += plan.tile
            continue
        rows = act[pos : pos + lim]
        scales = act_scale[pos : pos + lim]
        if nvfp4:
            a = _dequant_nvfp4(rows, scales, torch.bfloat16)
            w = _dequant_nvfp4(weight[e], weight_scale[e], torch.bfloat16)
            out[pos : pos + lim] = (a @ w.T) * row_scale[pos : pos + lim, None] * ones[e]
        else:
            a = _dequant_mxfp8(rows, scales, torch.bfloat16)
            w = _dequant_mxfp8(weight[e], weight_scale[e], torch.bfloat16)
            out[pos : pos + lim] = (a @ w.T) * ones[e]
        pos += plan.tile
    return out


def load_kernels(
    weights: tuple[torch.Tensor, torch.Tensor],
    weight_scales: tuple[torch.Tensor, torch.Tensor],
    topk: int,
) -> None:
    """No cubins to load on the torch path; kept for interface parity."""
    return None
