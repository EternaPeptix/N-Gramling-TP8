# SPDX-License-Identifier: Apache-2.0
"""SM121a (GB10) 0-ULP Bit-Exact BF12 Fused MoE Router + Input Quantization.

Packs `gate.weight` [384, 4608] bf16 (3.54 MB) into lossless BF12 slabs
(2.65 MB, -25.0%) + tiny outlier list (~255 outliers/layer = 2 KB), and
executes a 1-wave 48-CTA fused CUDA kernel (`librouter_bf12_exact.so`) that
matches `ops_moe._fused_route_kernel` to LITERAL 0 DIFFS across every output
tensor (src2dst, gammas, routed, tiles, QValues, QScales, QRowScales) and
intermediate router logits.
"""

from __future__ import annotations

import ctypes
import os
import weakref
import torch

ENABLED = os.environ.get("INKLING_ROUTER_BF12", "1").strip().lower() not in (
    "0", "", "off", "false", "no"
)
ROUTER_BF11 = os.environ.get("INKLING_ROUTER_BF11", "0").strip().lower() in (
    "1", "on", "true", "yes"
)
ROUTER_BF11_FLIP_THRESH = max(0, min(32, int(os.environ.get("INKLING_ROUTER_BF11_FLIP_THRESH", "16") or "16")))
FREE_RAW = os.environ.get(
    "INKLING_ROUTER_BF12_FREE_RAW",
    os.environ.get("INKLING_FREE_PACKED_RAW_WEIGHTS", "0"),
).strip().lower() not in ("0", "", "off", "false", "no")
_LIB_PATH = os.environ.get("INKLING_ROUTER_BF12_LIB", "/rdma/librouter_bf12_exact.so")
_LIB = None
_LIB_TRIED = False

# (data_ptr, E, H) -> (weakref_w, slabs, outliers, o_off, base_exp)
_CACHE: dict[tuple[int, int, int], tuple] = {}
# Persistent per-device tiny workspace (partials [5, 16, 384] fp32 = 120 KB)
_WS: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}


def _get_lib(lib_path: str | None = None):
    global _LIB, _LIB_TRIED
    path = lib_path or _LIB_PATH
    if not os.path.exists(path) and lib_path is None:
        local_fallback = os.path.join(os.path.dirname(os.path.abspath(__file__)), "librouter_bf12_exact.so")
        if os.path.exists(local_fallback):
            path = local_fallback
    if _LIB is not None and lib_path is None:
        return _LIB
    if not _LIB_TRIED or lib_path is not None:
        _LIB_TRIED = True
        if os.path.exists(path):
            lib = ctypes.CDLL(path)
            lib.router_bf12_exact_fwd.argtypes = [
                ctypes.c_void_p,  # X
                ctypes.c_void_p,  # SLABS
                ctypes.c_void_p,  # OUTLIERS
                ctypes.c_void_p,  # O_OFF
                ctypes.c_void_p,  # Bias
                ctypes.c_void_p,  # ExpertBounds
                ctypes.c_void_p,  # Padding
                ctypes.c_void_p,  # NLivePtr
                ctypes.c_void_p,  # Counters
                ctypes.c_void_p,  # Masks
                ctypes.c_void_p,  # LogitsWS
                ctypes.c_void_p,  # KeysWS
                ctypes.c_void_p,  # Src2Dst
                ctypes.c_void_p,  # Gammas
                ctypes.c_void_p,  # Rows
                ctypes.c_void_p,  # Tiles
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
                ctypes.c_int,     # pdl
                ctypes.c_uint64,  # stream_ptr
            ]
            lib.router_bf12_exact_fwd.restype = ctypes.c_int
            _LIB = lib
    return _LIB


def _get_ws(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    idx = device.index or 0
    ws = _WS.get(idx)
    if ws is None:
        partials_ws = torch.empty((5, 16, 384), dtype=torch.float32, device=device)
        unused_ws = torch.empty((1,), dtype=torch.uint64, device=device)
        ws = (partials_ws, unused_ws)
        _WS[idx] = ws
    return ws


def get_packed(w: torch.Tensor | None) -> tuple | None:
    """Return packed BF12 entry `(weakref_w, slabs, outliers, o_off, base_exp)` surviving resize_(0)."""
    if not ENABLED or w is None or tuple(w.shape) != (384, 4608):
        return None
    direct = getattr(w, "_inkling_router_bf12", None)
    if direct is not None and direct[0]() is w:
        return direct
    ptr = int(w.data_ptr())
    if ptr == 0:
        return None
    hit = _CACHE.get((ptr, 384, 4608))
    if hit is not None and hit[0]() is w:
        return hit
    return None


@torch.no_grad()
def prepack(w: torch.Tensor) -> torch.Tensor | None:
    """Pack bf16 gate.weight [384, 4608] into 0-ULP BF12 router slabs (48 CTAs x 36 slabs).

    Returns the packed slabs tensor (2.654 MB, suitable for L2 prefetch), or None if ineligible.
    """
    if (
        not ENABLED
        or w is None
        or w.dtype != torch.bfloat16
        or tuple(w.shape) != (384, 4608)
        or not w.is_contiguous()
    ):
        return None
    hit = get_packed(w)
    if hit is not None:
        return hit[1]
    if w.untyped_storage().size() == 0:
        return None
    if torch.cuda.is_current_stream_capturing():
        return None
    key = (int(w.data_ptr()), 384, 4608)

    device = w.device
    u = w.view(torch.uint16).to(torch.int32)
    exp = (u >> 7) & 0xFF
    cnt = torch.bincount(exp.flatten(), minlength=256)
    base = max(range(15, 255), key=lambda b: int(cnt[b - 15 : b + 1].sum().item()))

    delta = base - exp
    in_win = (delta >= 0) & (delta <= 15)
    d4 = torch.where(in_win, delta, torch.zeros_like(delta)).to(torch.uint8)
    sm = (((u >> 8) & 0x80) | (u & 0x7F)).to(torch.uint8)

    # Reshape [384, 4608] = [48 CTAs, 8 experts, 36 slabs of 128 cols]:
    # Slab layout inside each CTA bid:
    #   warp 0: slabs 0..7   (k = 0..1023)
    #   warp 1: slabs 8..15  (k = 1024..2047)
    #   warp 2: slabs 16..23 (k = 2048..3071)
    #   warp 3: slabs 24..31 (k = 3072..4095)
    #   warp 4: slabs 32..35 (k = 4096..4607)
    # Within each slab (8 experts x 128 cols = 1024 elems):
    #   [8 steps of 16 cols] x [2 halves of 8 cols] x [4 t-groups of 2 cols]
    #   MMA lane = g * 4 + t (g = expert 0..7, t = 0..3), j = half * 2 + pair (0..3)
    def to_slab_mma(x: torch.Tensor) -> torch.Tensor:
        # [48, 8(g), 36(slab), 8(s), 2(half), 4(t), 2(pair)]
        v = x.view(48, 8, 36, 8, 2, 4, 2)
        # -> [48, 36, 8(s), 8(g), 4(t), 2(half), 2(pair)]
        v = v.permute(0, 2, 3, 1, 5, 4, 6).contiguous()
        return v.view(48, 36, 8, 32, 4)

    sm_m = to_slab_mma(sm)
    d4_m = to_slab_mma(d4)
    inw_m = to_slab_mma(in_win)
    u_m = to_slab_mma(u)

    # sm_slab: [48, 36, 2(chunk4s), 32(lane), 4(s_in_chunk), 4(j)] -> 1024 B/slab
    sm_slab = (
        sm_m.view(48, 36, 2, 4, 32, 4)
        .permute(0, 1, 2, 4, 3, 5)
        .contiguous()
        .view(48, 36, 1024)
    )
    # e4_slab: [48, 36, 32(lane), 8(s), 2(packed nibbles)] -> 512 B/slab
    d4_lane = d4_m.permute(0, 1, 3, 2, 4).contiguous()
    e4_slab = (
        torch.stack(
            [
                d4_lane[..., 0] | (d4_lane[..., 1] << 4),
                d4_lane[..., 2] | (d4_lane[..., 3] << 4),
            ],
            dim=-1,
        )
        .contiguous()
        .view(48, 36, 512)
    )
    slabs = torch.cat([sm_slab, e4_slab], dim=2).contiguous().view(-1)

    # Build per-warp outlier table (240 warps = 48 CTAs * 5 warps)
    # Slab index sl_global in [0..35] maps to:
    #   warp = sl_global // 8  (0..4; warp 4 gets 32..35)
    #   sl_local = sl_global % 8 (0..7 for warps 0..3, 0..3 for warp 4)
    bad = (~inw_m).nonzero(as_tuple=False)
    wtot = 48 * 5
    if bad.numel() > 0:
        bid_i = bad[:, 0]
        sl_g = bad[:, 1]
        s_i = bad[:, 2]
        lane_i = bad[:, 3]
        j_i = bad[:, 4]
        warp_i = sl_g >> 3
        sl_loc = sl_g & 7
        widx = bid_i * 5 + warp_i
        step_id = (sl_loc << 3) | s_i
        lj = (lane_i << 2) | j_i
        uval = u_m[bid_i, sl_g, s_i, lane_i, j_i].to(torch.int64) & 0xFFFF
        order = torch.argsort((widx << 24) | (step_id << 8) | lj)
        widx, step_id, lj, uval = widx[order], step_id[order], lj[order], uval[order]
        outliers = ((step_id << 32) | (lj << 16) | uval).to(torch.uint64).contiguous()
        counts = torch.bincount(widx, minlength=wtot).to(torch.int32)
        o_off = torch.zeros(wtot + 1, dtype=torch.int32, device=device)
        o_off[1:] = torch.cumsum(counts, 0, dtype=torch.int32)
    else:
        outliers = torch.zeros(1, dtype=torch.uint64, device=device)
        o_off = torch.zeros(wtot + 1, dtype=torch.int32, device=device)

    entry = (weakref.ref(w), slabs, outliers, o_off, base)
    _CACHE[key] = entry
    try:
        w._inkling_router_bf12 = entry  # type: ignore[attr-defined]
        w._inkling_router_bf12_nout = int(bad.shape[0])  # type: ignore[attr-defined]
    except Exception:
        pass
    return slabs


@torch.no_grad()
def unpack_bf16(w: torch.Tensor) -> torch.Tensor | None:
    """Lossless 0-ULP reconstruction of `[384, 4608]` BF16 gate weight from BF12 slabs + outliers.

    Zero host syncs (no `.item()`), 100% CUDA-graph-capture-safe fallback when raw BF16 storage was freed.
    """
    entry = get_packed(w)
    if entry is None:
        return None
    _, slabs, outliers, o_off, base_exp = entry
    dev = slabs.device
    slab_v = slabs.view(48, 36, 1536)
    sm_slab = slab_v[:, :, :1024].contiguous()
    e4_slab = slab_v[:, :, 1024:].contiguous()

    # Reverse sm_slab: [48, 36, 2(chunk4s), 32(lane), 4(s_in_chunk), 4(j)] -> [48, 36, 8(s), 32(lane), 4(j)]
    sm_m = (
        sm_slab.view(48, 36, 2, 32, 4, 4)
        .permute(0, 1, 2, 4, 3, 5)
        .contiguous()
        .view(48, 36, 8, 32, 4)
        .to(torch.int32)
    )
    # Reverse e4_slab: [48, 36, 32(lane), 8(s), 2(nibbles)] -> [48, 36, 8(s), 32(lane), 4(j)]
    e4_u8 = e4_slab.view(48, 36, 32, 8, 2).to(torch.int32)
    p0 = e4_u8[..., 0]
    p1 = e4_u8[..., 1]
    d4_lane = torch.stack([p0 & 0xF, (p0 >> 4) & 0xF, p1 & 0xF, (p1 >> 4) & 0xF], dim=-1)
    d4_m = d4_lane.permute(0, 1, 3, 2, 4).contiguous()

    exp_m = int(base_exp) - d4_m
    u_m = (((sm_m & 0x80) << 8) | (exp_m << 7) | (sm_m & 0x7F)).contiguous()

    n_out = getattr(w, "_inkling_router_bf12_nout", None)
    if n_out is None:
        n_out = int(o_off[-1].item()) if not torch.cuda.is_current_stream_capturing() else 0
    if n_out > 0:
        ov = outliers[:n_out].view(torch.int64)
        idx_out = torch.arange(n_out, device=dev, dtype=torch.int32)
        widx = torch.searchsorted(o_off[1:], idx_out, right=True).to(torch.int64)
        bid_i = widx // 5
        warp_i = widx % 5
        step_id = (ov >> 32) & 0xFFFFFFFF
        sl_loc = (step_id >> 3) & 7
        s_i = step_id & 7
        sl_g = (warp_i << 3) | sl_loc
        lj = (ov >> 16) & 0xFFFF
        lane_i = (lj >> 2) & 31
        j_i = lj & 3
        uval = (ov & 0xFFFF).to(torch.int32)
        flat = (((bid_i * 36 + sl_g) * 8 + s_i) * 32 + lane_i) * 4 + j_i
        u_m.reshape(-1).scatter_(0, flat, uval)

    # Inverse of to_slab_mma: [48, 36, 8(s), 8(g), 4(t), 2(half), 2(pair)] -> [48, 8(g), 36, 8(s), 2(half), 4(t), 2(pair)]
    u_orig = (
        u_m.view(48, 36, 8, 8, 4, 2, 2)
        .permute(0, 3, 1, 2, 5, 4, 6)
        .contiguous()
        .view(384, 4608)
    )
    return u_orig.to(torch.int16).view(torch.bfloat16)


@torch.no_grad()
def free_raw_weight(w: torch.Tensor) -> int:
    """Opt-in (`INKLING_ROUTER_BF12_FREE_RAW=1`): release raw BF16 storage after BF12 packing (-3.375 MiB/layer)."""
    if not FREE_RAW or w is None or get_packed(w) is None:
        return 0
    if w.storage_offset() != 0:
        return 0
    raw_bytes = int(w.numel() * w.element_size())
    stor = w.data.untyped_storage()
    if int(stor.size()) != raw_bytes or raw_bytes <= 0:
        return 0
    stor.resize_(0)
    return raw_bytes


@torch.no_grad()
def get_packed_bf11(w: torch.Tensor | None) -> tuple | None:
    """Return cached Wave-V10 lossless BF11-Flip router entry `(slabs11, slab_meta, sflips, bmasks, outliers, o_off, base_exp)`."""
    if not ROUTER_BF11 or w is None or tuple(w.shape) != (384, 4608):
        return None
    return getattr(w, "_inkling_router_bf11", None)


@torch.no_grad()
def prepack_bf11(w: torch.Tensor, flip_thresh: int | None = None) -> tuple | None:
    """Transcode packed BF12 router slabs into lossless V5 BF11-Flip slabs (1408 B/slab + adaptive sidecar).

    Works directly from BF12 slabs so it remains 100% safe even after `INKLING_ROUTER_BF12_FREE_RAW=1`.
    """
    if not ROUTER_BF11 or w is None or tuple(w.shape) != (384, 4608):
        return None
    cached = getattr(w, "_inkling_router_bf11", None)
    if cached is not None:
        return cached
    if torch.cuda.is_current_stream_capturing():
        return None
    hit = get_packed(w)
    if hit is None:
        if prepack(w) is None:
            return None
        hit = get_packed(w)
        if hit is None:
            return None
    _, slabs12, outliers, o_off, base_exp = hit
    dev = slabs12.device
    thr = ROUTER_BF11_FLIP_THRESH if flip_thresh is None else max(0, min(63, int(flip_thresh)))
    sv = slabs12.view(48, 36, 1536)
    sm_slab = sv[:, :, :1024].contiguous()
    e4_u8 = sv[:, :, 1024:].contiguous().view(48, 36, 32, 8, 2).to(torch.int32)
    p0, p1 = e4_u8[..., 0], e4_u8[..., 1]
    # d4_lane: [48, 36, 32(lane), 8(step), 4(j)] -> flatten [1728, 32, 32(bit=step*4+j)]
    d4_lane = torch.stack([p0 & 0xF, (p0 >> 4) & 0xF, p1 & 0xF, (p1 >> 4) & 0xF], dim=-1).view(1728, 32, 32)
    d3 = (d4_lane & 7).to(torch.int64)
    fl = ((d4_lane >> 3) & 1).to(torch.int64)

    # Pack 32 x 3-bit exponents per lane into 3 uint32 words [1728, 3, 32] (384 B/slab)
    e0 = torch.zeros((1728, 32), dtype=torch.int64, device=dev)
    e1 = torch.zeros((1728, 32), dtype=torch.int64, device=dev)
    e2 = torch.zeros((1728, 32), dtype=torch.int64, device=dev)
    for b in range(32):
        bit0 = 3 * b
        w_idx = bit0 >> 5
        sh = bit0 & 31
        val = d3[:, :, b]
        if w_idx == 0:
            e0 |= (val << sh) & 0xFFFFFFFF
            if sh > 29:
                e1 |= val >> (32 - sh)
        elif w_idx == 1:
            e1 |= (val << sh) & 0xFFFFFFFF
            if sh > 29:
                e2 |= val >> (32 - sh)
        else:
            e2 |= (val << sh) & 0xFFFFFFFF
    e3_u32 = torch.stack([e0, e1, e2], dim=1).to(torch.uint32).contiguous()
    e3_slab = e3_u32.view(torch.uint8).view(48, 36, 384)
    slabs11 = torch.cat([sm_slab, e3_slab], dim=2).contiguous().view(-1)

    # Build per-slab adaptive V5 flip sidecar + outlier hint bit
    bit_pos = torch.arange(32, dtype=torch.int64, device=dev).view(1, 1, 32)
    flip_masks = (fl << bit_pos).sum(dim=2).to(torch.uint32)  # [1728, 32]
    nf_per_slab = fl.sum(dim=(1, 2)).cpu().tolist()

    # Determine which slabs contain >= 1 outlier record
    has_out_slab = [False] * 1728
    n_out = int(o_off[-1].item())
    if n_out > 0:
        ov = outliers[:n_out].cpu().tolist()
        oo = o_off.cpu().tolist()
        for bid_i in range(48):
            for pk_i in range(5):
                widx = bid_i * 5 + pk_i
                for oi in range(oo[widx], oo[widx + 1]):
                    sl_loc = (int(ov[oi]) >> 35) & 7
                    sl_g = pk_i * 8 + sl_loc
                    if sl_g < 36:
                        has_out_slab[bid_i * 36 + sl_g] = True

    fl_cpu = fl.cpu()
    meta_list = [0] * 1728
    sflips_list: list[int] = []
    dense_indices: list[int] = []
    for s_idx in range(1728):
        nf = int(nf_per_slab[s_idx])
        out_bit = 0x01000000 if has_out_slab[s_idx] else 0
        if nf == 0:
            meta_list[s_idx] = out_bit
        elif nf <= thr:
            s_off = len(sflips_list)
            nz = fl_cpu[s_idx].nonzero(as_tuple=False)
            for r in range(nz.shape[0]):
                sflips_list.append((int(nz[r, 0]) << 5) | int(nz[r, 1]))
            meta_list[s_idx] = ((nf & 0x3F) << 25) | out_bit | (s_off & 0x00FFFFFF)
        else:
            d_idx = len(dense_indices)
            dense_indices.append(s_idx)
            meta_list[s_idx] = 0x80000000 | out_bit | (d_idx & 0x00FFFFFF)

    slab_meta = torch.tensor(meta_list, dtype=torch.int64, device=dev).to(torch.uint32).contiguous()
    sflips = (
        torch.tensor(sflips_list, dtype=torch.int32, device=dev).to(torch.uint16).contiguous()
        if sflips_list
        else torch.zeros((1,), dtype=torch.uint16, device=dev)
    )
    if dense_indices:
        idx_t = torch.tensor(dense_indices, dtype=torch.long, device=dev)
        bmasks = flip_masks.index_select(0, idx_t).contiguous()
    else:
        bmasks = torch.zeros((1, 32), dtype=torch.uint32, device=dev)

    entry11 = (slabs11, slab_meta, sflips, bmasks, outliers, o_off, int(base_exp))
    try:
        w._inkling_router_bf11 = entry11  # type: ignore[attr-defined]
    except Exception:
        pass
    return entry11


def l2pf_tensor(w: torch.Tensor | None, num_tokens: int | None = None) -> torch.Tensor | None:
    if not ENABLED or w is None or tuple(w.shape) != (384, 4608):
        return w
    if ROUTER_BF11:
        hit11 = get_packed_bf11(w)
        if hit11 is not None:
            return hit11[0]
    hit = get_packed(w)
    if hit is None:
        return w
    if w.untyped_storage().size() == 0:
        return hit[1]
    if num_tokens is not None and num_tokens > 16:
        return w
    return hit[1]


def route_bf12(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    bias: torch.Tensor,
    counters: torch.Tensor,
    counts: torch.Tensor,
    topk: int,
    route_scale: float,
    expert_bounds: torch.Tensor,
    plan,
    weight_amax: torch.Tensor | None,
    weight_denominator: torch.Tensor | None,
    is_padding: torch.Tensor | None = None,
    nlive_buf: torch.Tensor | None = None,
    pdl: int | bool = 2,
    lib_path: str | None = None,
):
    """0-ULP BF12 fused route + quantize for T <= 16, E = 384, H = 4608, topk = 8.

    pdl: 0 = off, 1 = internal PDL (K1 -> K2), 2 = internal PDL + K2 triggers next kernel (w13 PDL_WAIT_MODE=1).
    """
    tokens, hidden = int(x.shape[0]), int(x.shape[1])
    if (
        not ENABLED
        or tokens <= 0
        or tokens > 16
        or hidden != 4608
        or topk != 8
        or tuple(gate_weight.shape) != (384, 4608)
        or plan.tile > 64
        or plan.tile < tokens
    ):
        return None
    lib = _get_lib(lib_path)
    if lib is None:
        return None
    entry = get_packed(gate_weight)
    if entry is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        prepack(gate_weight)
        entry = get_packed(gate_weight)
        if entry is None:
            return None

    _, slabs, outliers, o_off, base_exp = entry
    device = x.device
    nvfp4 = weight_amax is not None
    num_local = counts.numel()
    # Dec160 S8 fast path: 48-CTA zero-tail K1 + 1-CTA Zero-Scatter SMEM-Route K2
    try:
        from . import moe_s8_runtime as _s8
        if _s8.S8_ROUTER or _s8.S8_ZERO_SCATTER or _s8.S8_FUSE_SITU:
            _pdl_i = int(pdl) if not isinstance(pdl, bool) else (2 if pdl else 0)
            _s8_res = _s8.route_bf12_s8(
                x, slabs, outliers, o_off, int(base_exp), bias, topk, float(route_scale),
                expert_bounds, plan, weight_amax, weight_denominator, int(num_local),
                is_padding=is_padding, nlive_buf=nlive_buf, pdl=_pdl_i,
                gate_weight=gate_weight if ROUTER_BF11 else None,
            )
            if _s8_res is not None:
                return _s8_res
    except Exception:
        pass
    n = tokens * topk
    rows = plan.rows

    src2dst = torch.empty(n, dtype=torch.int32, device=device)
    gammas = torch.empty(rows, dtype=torch.float32, device=device)
    routed = torch.empty(n, dtype=torch.int32, device=device)
    tiles = torch.empty(2 * plan.max_ctas + 3, dtype=torch.int32, device=device)
    if nvfp4:
        quantized = (
            torch.empty((rows, hidden // 2), dtype=torch.uint8, device=device),
            torch.empty((rows, hidden // 16), dtype=torch.float8_e4m3fn, device=device),
            torch.empty(rows, dtype=torch.float32, device=device),
        )
    else:
        quantized = (
            torch.empty((rows, hidden), dtype=torch.float8_e4m3fn, device=device),
            torch.empty((rows, hidden // 32), dtype=torch.uint8, device=device),
            None,
        )

    logits_ws, keys_ws = _get_ws(device)
    rc = lib.router_bf12_exact_fwd(
        x.data_ptr(),
        slabs.data_ptr(),
        outliers.data_ptr(),
        o_off.data_ptr(),
        bias.data_ptr(),
        expert_bounds.data_ptr(),
        0 if is_padding is None else is_padding.data_ptr(),
        0 if nlive_buf is None else nlive_buf.data_ptr(),
        counters.data_ptr(),
        counts.data_ptr(),
        logits_ws.data_ptr(),
        keys_ws.data_ptr(),
        src2dst.data_ptr(),
        gammas.data_ptr(),
        routed.data_ptr(),
        tiles.data_ptr(),
        quantized[0].data_ptr(),
        quantized[1].data_ptr(),
        0 if quantized[2] is None else quantized[2].data_ptr(),
        0 if weight_amax is None else weight_amax.data_ptr(),
        0 if weight_denominator is None else weight_denominator.data_ptr(),
        tokens,
        float(route_scale),
        int(base_exp),
        int(num_local),
        int(plan.tile),
        int(plan.max_ctas),
        1 if nvfp4 else 0,
        int(pdl) if not isinstance(pdl, bool) else (2 if pdl else 0),
        int(torch.cuda.current_stream().cuda_stream),
    )
    if rc != 0:
        raise RuntimeError(f"router_bf12_exact_fwd failed with cudaError={rc}")
    return (src2dst, gammas, routed, tiles), quantized
