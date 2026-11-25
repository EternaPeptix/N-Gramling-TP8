# SPDX-License-Identifier: Apache-2.0
"""SM121a (GB10) 0-ULP Certified LM Head v7:
2-Stage Progressive INT6_g512_ms8(2b) + 1bLSB + Full-Row L2-Prefetched Refine + Direct Logits Gather.

Key wins vs cert_head_fast_v6 (INT7_g64: 105.14 MB, 439 us screen + 20.3 us refine):
  1. Stage-1 screen (`INT6_g512_ms8(2b)`): 91.11 MB (-14.03 MB = -58.5 us vs INT7_g64!),
     100% uint4 aligned, e1 = 1.97%.
  2. Stage-2 1bLSB sparse filter: reads only 738 B/row (576 B LSB1 + 144 B MS + 18 B BIG_S)
     for Stage-1 survivors K1 (~400..800 rows = ~0.3-0.6 MB = 2-4 us), upgrading
     e1 = 1.97% -> e2 = 1.05% (TIGHTER than INT7_g64 1.20% & INT7_g32 1.09%!), pruning
     K1 -> K2 (~5..15 rows) with ZERO extra RDMA and ~identical total VRAM (105.58 MB)!
  3. Stage-3 exact 0-ULP BF16 refine: full-row parallel `prefetch.global.L2` fixes v6's
     20.3 us serial cold-DRAM latency bug -> ~4.5 us (-15.8 us/step, literal 0-ULP!).
  4. Direct `LOGITS_IDX` gather inside `cert_xnorm_pad_v7` skips PyTorch's 4.32 us
     `vectorized_gather_kernel` before CertHead!
  5. Includes `check_nw_heterogeneous(nw)` auto-probe to detect if live Inkling-2-Small
     `lm_head.weight` has heterogeneous row norms (`GM(|nw|)/RMS(nw) < 0.85`), which
     would unlock an extra -30 to -45 us via mixed INT6/INT5 water-filling.
"""
from __future__ import annotations

import ctypes
import hashlib
import math
import os
import threading
from dataclasses import dataclass

import numpy as np

try:
    import torch
except ImportError:  # Local Mac verification runs without PyTorch
    torch = None  # type: ignore[assignment]

V14_ENABLED = os.environ.get("INKLING_CERT_HEAD_V14", "0").strip().lower() in ("1", "true", "on", "yes")
_V14_MAX_M_RAW = os.environ.get("INKLING_CERT_HEAD_V14_MAX_M", "").strip()
V14_MAX_M = max(16, min(128, int(_V14_MAX_M_RAW or "64")))
ENABLED = (
    os.environ.get("INKLING_CERT_HEAD_V7", "0").strip().lower() in ("1", "true", "on", "yes")
    or V14_ENABLED
)
_LIB_PATH = (
    os.environ.get("INKLING_CERT_V14_LIB")
    or os.environ.get("INKLING_CERT_V7_LIB", "/rdma/libcert_head_fast_v14.so")
) if V14_ENABLED else os.environ.get("INKLING_CERT_V7_LIB", "/rdma/libcert_head_fast_v7.so")
MAX_M = (
    V14_MAX_M
    if (V14_ENABLED and _V14_MAX_M_RAW != "")
    else max(
        16,
        min(
            128,
            int(
                os.environ.get("INKLING_CERT_HEAD_V7_MAX_M", "64" if V14_ENABLED else "16")
                or ("64" if V14_ENABLED else "16")
            ),
        ),
    )
)
_QUALIFY_REF = os.environ.get("INKLING_CERT_V9_QUALIFY_REF", "torch").strip().lower()  # torch (default) | b1_exact
_CFG_ID = int(os.environ.get("INKLING_CERT_V7_CFG", "0"))  # 0: BN=64 W=4 SUB=2 (default); 1: BN=32 W=4; 2: BN=16 W=2
_REF_GRID = int(os.environ.get("INKLING_CERT_V7_REF_GRID", "96"))
_USE_PDL = os.environ.get("INKLING_CERT_V7_PDL", "1").strip().lower() not in ("0", "", "off", "false", "no")
_CACHE_DIR = os.environ.get("INKLING_CERT_HEAD_CACHE", "").strip()
_FREE_PACKED_DEFAULT = os.environ.get("INKLING_FREE_PACKED_RAW_WEIGHTS", "0").strip().lower() in (
    "1", "true", "on", "yes",
)
_DELEGATE_C32 = os.environ.get(
    "INKLING_CERT_V7_DELEGATE_C32",
    "0" if _FREE_PACKED_DEFAULT else "1",
).strip().lower() not in ("0", "", "off", "false", "no")
# Wave-V16 opt-in temp > 0 & n=4 shared-prefix direct-logit bypass (default OFF = 0, 100% 0-ULP safe).
TEMP_DIRECT_V16 = os.environ.get("INKLING_CERT_HEAD_TEMP_DIRECT_V16", "0").strip().lower() in (
    "1", "true", "on", "yes",
)
_TEMP_DIRECT_STEP_ACTIVE = False
_TEMP_DIRECT_L2PF_HINT = False
_TEMP_DIRECT_SHARED_PREFIX_N = 1
_TEMP_DIRECT_BYPASS_STEPS = 0
_TEMP_DIRECT_SHARED_PREFIX_STEPS = 0


def set_temp_direct_step(
    active: bool, shared_prefix_n: int = 1, update_l2pf_hint: bool = True
) -> None:
    """Mark current decode step as temp > 0 (zero host syncs): skips CertHead screening & redirects L2PF."""
    global _TEMP_DIRECT_STEP_ACTIVE, _TEMP_DIRECT_L2PF_HINT, _TEMP_DIRECT_SHARED_PREFIX_N
    global _TEMP_DIRECT_BYPASS_STEPS, _TEMP_DIRECT_SHARED_PREFIX_STEPS
    act = bool(active)
    _TEMP_DIRECT_STEP_ACTIVE = act
    if update_l2pf_hint:
        _TEMP_DIRECT_L2PF_HINT = act
    sp_n = max(1, int(shared_prefix_n))
    _TEMP_DIRECT_SHARED_PREFIX_N = sp_n
    if act and TEMP_DIRECT_V16:
        _TEMP_DIRECT_BYPASS_STEPS += 1
        if sp_n > 1:
            _TEMP_DIRECT_SHARED_PREFIX_STEPS += 1


def is_temp_direct_active() -> bool:
    return bool(TEMP_DIRECT_V16 and _TEMP_DIRECT_STEP_ACTIVE)


def is_temp_direct_l2pf_bypass() -> bool:
    return bool(TEMP_DIRECT_V16 and (_TEMP_DIRECT_STEP_ACTIVE or _TEMP_DIRECT_L2PF_HINT))


def get_temp_direct_stats() -> dict[str, int | bool]:
    return {
        "enabled": bool(TEMP_DIRECT_V16),
        "step_active": bool(_TEMP_DIRECT_STEP_ACTIVE),
        "l2pf_hint": bool(_TEMP_DIRECT_L2PF_HINT),
        "last_shared_prefix_n": int(_TEMP_DIRECT_SHARED_PREFIX_N),
        "bypass_steps": int(_TEMP_DIRECT_BYPASS_STEPS),
        "shared_prefix_steps": int(_TEMP_DIRECT_SHARED_PREFIX_STEPS),
    }

MS_MULTS = np.array([0.375, 0.5625, 0.75, 1.0], dtype=np.float32)

# Exact IEEE-754 soundness constants matching cert_head.py & cert_head_cuda.py
_EPS_F32 = 2.0 ** -24
_INFL_X = 1.0 + 2.0 ** -14
_X_ABS = 2.0 ** -59
_INFL_Q = 1.0 + 2.0 ** -14
_INFL_B = 1.0 + 2.0 ** -20
_SLOP_REL = 2.0 ** -21
_SLOP_ABS = 2.0 ** -126
CAP_DIP_30 = 1.9073486328125e-06
_SLACK = float(np.float32(2.0 * CAP_DIP_30 * (1.0 + 2.0 ** -10) + 2.0 ** -17))


def slack_of_v7(dip: float) -> float:
    return 0.0 if dip <= 0.0 else float(np.float32(2.0 * dip * (1.0 + 2.0 ** -10) + 2.0 ** -17))


def round_up_f32_np(x64: np.ndarray) -> np.ndarray:
    """float64 (>= 0 or inf) -> smallest IEEE-754 float32 >= x64."""
    x64 = np.asarray(x64, dtype=np.float64)
    f = x64.astype(np.float32)
    bump = f.astype(np.float64) < x64
    return np.where(bump, np.nextafter(f, np.float32(np.inf)), f).astype(np.float32)

_CFGS = {
    0: (4, 2, 64),
    1: (4, 1, 32),
    2: (2, 1, 16),
}

_LIB = None
_HAS_C48 = False
_HAS_V9 = False
_HAS_V14 = False
_LIB_VER = 0
_LIB_C48_MAX_M = 16
_LIB_LOCK = threading.Lock()
_ACTIVE_ADAPTER = None
_TP_ALL_GATHER_FN = None


def _get_tp_all_gather():
    global _TP_ALL_GATHER_FN
    fn = _TP_ALL_GATHER_FN
    if fn is None:
        from vllm.distributed import tensor_model_parallel_all_gather as fn
        _TP_ALL_GATHER_FN = fn
    return fn


def bf16_rn_np(x_f32: np.ndarray) -> np.ndarray:
    """Exact IEEE-754 round-to-nearest-even float32 -> BF16 -> float32 in pure NumPy."""
    x = np.asarray(x_f32, dtype=np.float32)
    u = x.view(np.uint32)
    # Handle NaN cleanly
    is_nan = (u & np.uint32(0x7FFFFFFF)) > np.uint32(0x7F800000)
    lsb = (u >> np.uint32(16)) & np.uint32(1)
    rounding_bias = np.uint32(0x7FFF) + lsb
    u_rn = (u + rounding_bias) & np.uint32(0xFFFF0000)
    u_out = np.where(is_nan, (u & np.uint32(0xFFFF0000)) | np.uint32(0x00400000), u_rn)
    return u_out.view(np.float32)


def cacc2_for_k(K: int) -> float:
    """Exact 2 * c_acc MMA accumulation error constant for K-dot products (matches cert_head.py)."""
    c_acc = (K / 16.0) * 17.0 * 2.0 ** -23
    return float(2.0 * c_acc)


def check_nw_heterogeneous(w_f32: np.ndarray) -> dict[str, float]:
    """Probe lm_head.weight row-norm spread: if GM(||w_j||)/RMS(||w_j||) < 0.85,
    mixed INT6/INT5 row-budgeted water-filling saves an additional 8-12% Stage-1 bytes."""
    nw = np.linalg.norm(w_f32.astype(np.float64), axis=1)
    nz = nw[nw > 0]
    gm = float(np.exp(np.mean(np.log(nz)))) if len(nz) else 0.0
    rms = float(np.sqrt(np.mean(nz * nz))) if len(nz) else 1.0
    ratio = gm / rms if rms > 0 else 1.0
    return {
        "nw_min": float(nw.min()),
        "nw_p50": float(np.median(nw)),
        "nw_p95": float(np.percentile(nw, 95)),
        "nw_max": float(nw.max()),
        "gm_over_rms": ratio,
        "recommend_hetero_int6_int5": float(ratio < 0.85),
    }


@dataclass
class PackedCertV7NumPy:
    slabs_u4: np.ndarray    # [wtot, n_qchunks * sub_n, 6, 32, 4] uint32 (view as uint4)
    ms_u4: np.ndarray       # [wtot, n_qchunks * sub_n, 8, 4] uint32 (view as uint4)
    big_s: np.ndarray       # [wtot, n_qchunks * sub_n, 8] float16
    lsb1_row: np.ndarray    # [V, n_qchunks * 16] uint32 (1b/elem row-contiguous)
    ms_row: np.ndarray      # [V, n_qchunks, 4] uint32 (view as uint4)
    big_s_row: np.ndarray   # [V, n_qchunks] float16
    rt1: np.ndarray         # [V] float32 (exact Stage-1 residual L2 norm, rounded UP)
    nv1: np.ndarray         # [V] float32 (max(||W||, ||W1_bf16||), rounded UP)
    rt2: np.ndarray         # [V] float32 (exact Stage-2 residual L2 norm, rounded UP)
    nv2: np.ndarray         # [V] float32 (NV1 + ||dW_bf16||, rounded UP)
    w1_bf16: np.ndarray     # [V, K] float32 (exact reconstructed Stage-1 BF16 weights, for verification)
    dw_bf16: np.ndarray     # [V, K] float32 (exact reconstructed Stage-2 delta BF16 weights, for verification)
    V: int
    K: int
    cfg_id: int

    @property
    def stage1_bytes(self) -> int:
        return (
            self.slabs_u4.nbytes
            + self.ms_u4.nbytes
            + self.big_s.nbytes
            + self.rt1.nbytes
            + self.nv1.nbytes
        )

    @property
    def stage2_bytes(self) -> int:
        return (
            self.lsb1_row.nbytes
            + self.ms_row.nbytes
            + self.big_s_row.nbytes
            + self.rt2.nbytes
            + self.nv2.nbytes
        )


def pack_cert_v7_np(w_bf16_as_f32: np.ndarray, cfg_id: int = 0) -> PackedCertV7NumPy:
    """Exact packer for `cert_head_fast_v7.cu` (Stage-1 INT6_g512_ms8(2b) + Stage-2 1bLSB).

    Input `w_bf16_as_f32` should hold BF16-representable float32 weights `[V, K]` (`K % 512 == 0`).
    """
    w = bf16_rn_np(np.asarray(w_bf16_as_f32, dtype=np.float32))
    V, K = int(w.shape[0]), int(w.shape[1])
    assert K % 512 == 0, f"K={K} must be a multiple of 512"
    warps, sub_n, bn = _CFGS[cfg_id]
    n_qchunks = K // 512
    n_tiles = (V + bn - 1) // bn
    v_pad = n_tiles * bn
    wtot = n_tiles * warps

    # 1. Quantize all [V, n_qchunks, 64, 8] blocks
    w_blk = w.reshape(V, n_qchunks, 64, 8)
    m_abs = np.max(np.abs(w_blk), axis=3)              # [V, n_qchunks, 64]
    row_max = np.max(m_abs, axis=2)                    # [V, n_qchunks]

    # Choose fp16 big_s rounded UP so 31 * fp16(big_s) >= row_max (zero clipping!)
    bs_raw = (row_max / np.float32(31.0)).astype(np.float16)
    bs_f32 = bs_raw.astype(np.float32)
    need_up = (row_max > 0.0) & (bs_f32 * np.float32(31.0) < row_max)
    while np.any(need_up):
        bs_raw = np.where(need_up, np.nextafter(bs_raw, np.float16(np.inf)), bs_raw)
        bs_f32 = bs_raw.astype(np.float32)
        need_up = (row_max > 0.0) & (bs_f32 * np.float32(31.0) < row_max)

    big_s_row = np.ascontiguousarray(bs_raw)           # [V, n_qchunks] float16
    bs_exp = bs_f32[:, :, None]                        # [V, n_qchunks, 1]

    # Select smallest 2-bit micro-scale code in {0, 1, 2, 3} covering m_abs <= 31 * sc4[code2]
    # Thresholds: 0.375, 0.5625, 0.75, 1.0
    t0 = bs_exp * np.float32(0.375 * 31.0)
    t1 = bs_exp * np.float32(0.5625 * 31.0)
    t2 = bs_exp * np.float32(0.75 * 31.0)
    code2 = np.where(
        m_abs <= t0,
        np.uint32(0),
        np.where(m_abs <= t1, np.uint32(1), np.where(m_abs <= t2, np.uint32(2), np.uint32(3))),
    ).astype(np.uint32)                                # [V, n_qchunks, 64]

    sc_mb = bs_exp * MS_MULTS[code2]                   # [V, n_qchunks, 64] float32
    inv_sc = np.zeros_like(sc_mb, dtype=np.float32)
    np.divide(np.float32(1.0), sc_mb, out=inv_sc, where=sc_mb > 0.0)
    q1 = np.clip(np.rint(w_blk * inv_sc[:, :, :, None]), -31.0, 31.0).astype(np.int32)
    u6 = (q1 + 32).astype(np.uint32)                   # [V, n_qchunks, 64, 8] in [1..63]

    # Exact Stage-1 BF16 weight seen by `mma_step6_ms8`!
    w1_blk = bf16_rn_np(q1.astype(np.float32) * sc_mb[:, :, :, None])
    w1_bf16 = w1_blk.reshape(V, K)

    # Exact Stage-2 1bLSB quantization matching `cert_stage2_lsb1_v7` (bs * 0.25f * mult[code2])!
    r1_blk = w_blk.astype(np.float64) - w1_blk.astype(np.float64)
    lsb_bit = (r1_blk >= 0.0).astype(np.uint32)        # [V, n_qchunks, 64, 8] in {0, 1}
    sc_s2 = (bs_exp * np.float32(0.25)) * MS_MULTS[code2]
    dw_blk = bf16_rn_np(np.where(lsb_bit == 1, sc_s2[:, :, :, None], -sc_s2[:, :, :, None]))
    dw_bf16 = dw_blk.reshape(V, K)

    # Exact conservative fp64 certificate norms (rounded UP into fp32!)
    r2_f64 = w.astype(np.float64) - (w1_bf16.astype(np.float64) + dw_bf16.astype(np.float64))
    r1_f64 = w.astype(np.float64) - w1_bf16.astype(np.float64)
    nw_f64 = np.sqrt(np.sum(w.astype(np.float64) ** 2, axis=1))
    nw1_f64 = np.sqrt(np.sum(w1_bf16.astype(np.float64) ** 2, axis=1))
    ndw_f64 = np.sqrt(np.sum(dw_bf16.astype(np.float64) ** 2, axis=1))

    rt1 = round_up_f32_np(np.sqrt(np.sum(r1_f64 ** 2, axis=1)))
    nv1 = round_up_f32_np(np.maximum(nw_f64, nw1_f64))
    rt2 = round_up_f32_np(np.sqrt(np.sum(r2_f64 ** 2, axis=1)))
    nv2 = round_up_f32_np(np.maximum(nw_f64, nw1_f64) + ndw_f64)

    # 2. Pack row-contiguous Stage-2 arrays: MS_ROW [V, n_qchunks, 4] uint32 & LSB1_ROW [V, n_qchunks*16] uint32
    c2_4 = code2.reshape(V, n_qchunks, 4, 16)
    shifts_ms = (np.arange(16, dtype=np.uint32) * np.uint32(2))[None, None, None, :]
    ms_row = np.ascontiguousarray(np.bitwise_or.reduce(c2_4 << shifts_ms, axis=3).astype(np.uint32))

    lsb_u32 = lsb_bit.reshape(V, n_qchunks * 16, 32)
    shifts_b = np.arange(32, dtype=np.uint32)[None, None, :]
    lsb1_row = np.ascontiguousarray(np.bitwise_or.reduce(lsb_u32 << shifts_b, axis=2).astype(np.uint32))

    # 3. Pad V -> v_pad and pack Stage-1 warp-tiled quad-slabs:
    #    SLABS_U4 [wtot, n_qchunks * sub_n, 6, 32, 4] uint32
    #    MS_U4    [wtot, n_qchunks * sub_n, 8, 4] uint32
    #    BIG_S    [wtot, n_qchunks * sub_n, 8] float16
    if v_pad != V:
        u6_p = np.full((v_pad, n_qchunks, 512), 32, dtype=np.uint32)
        u6_p[:V] = u6.reshape(V, n_qchunks, 512)
        ms_p = np.zeros((v_pad, n_qchunks, 4), dtype=np.uint32)
        ms_p[:V] = ms_row
        bs_p = np.zeros((v_pad, n_qchunks), dtype=np.float16)
        bs_p[:V] = big_s_row
    else:
        u6_p = u6.reshape(V, n_qchunks, 512)
        ms_p = ms_row
        bs_p = big_s_row

    # Reshape rows to [n_tiles, warps, sub_n, 8(g)] -> warp_uid = pn * warps + warp, q = qc * sub_n + sub!
    ms_t = ms_p.reshape(n_tiles, warps, sub_n, 8, n_qchunks, 4).transpose(0, 1, 4, 2, 3, 5)
    ms_u4 = np.ascontiguousarray(ms_t.reshape(wtot, n_qchunks * sub_n, 8, 4))

    bs_t = bs_p.reshape(n_tiles, warps, sub_n, 8, n_qchunks).transpose(0, 1, 4, 2, 3)
    big_s = np.ascontiguousarray(bs_t.reshape(wtot, n_qchunks * sub_n, 8))

    # Reshape 512 cols into: [8(half-chunks of 64), 4(MMA steps of 16), 2(hi/lo 8), 4(t), 2(pair)]
    u6_t = u6_p.reshape(n_tiles, warps, sub_n, 8, n_qchunks, 8, 4, 2, 4, 2)
    # Transpose to: [n_tiles, warps, n_qchunks, sub_n, 8(g), 4(t), 8(hc), 4(s), 2(hi), 2(pair)]
    u6_t = u6_t.transpose(0, 1, 4, 2, 3, 8, 5, 6, 7, 9)
    # Collapse -> [wtot, total_q, 32(lane), 8(hc), 4(s), 4(j=0..3)]
    u6_t = u6_t.reshape(wtot, n_qchunks * sub_n, 32, 8, 4, 4)

    # Pack 4 weights (j=0..3) into 24-bit word u24: [wtot, total_q, 32, 8, 4]
    u24 = (
        u6_t[..., 0]
        | (u6_t[..., 1] << np.uint32(6))
        | (u6_t[..., 2] << np.uint32(12))
        | (u6_t[..., 3] << np.uint32(18))
    )
    # Pack 4 x 24-bit words (96 bits) -> 3 x uint32_t per half-chunk!
    w0 = u24[..., 0] | ((u24[..., 1] & np.uint32(0xFF)) << np.uint32(24))
    w1 = (u24[..., 1] >> np.uint32(8)) | ((u24[..., 2] & np.uint32(0xFFFF)) << np.uint32(16))
    w2 = (u24[..., 2] >> np.uint32(16)) | (u24[..., 3] << np.uint32(8))
    # Stack -> [wtot, total_q, 32(lane), 8(hc), 3] -> [wtot, total_q, 32(lane), 6(v0..v5), 4(xyzw)]
    words24 = np.stack([w0, w1, w2], axis=-1).reshape(wtot, n_qchunks * sub_n, 32, 6, 4)
    # Transpose (32, 6) -> (6, 32) so ld_cs_u4(qp + v*32 + lane) is 100% coalesced!
    slabs_u4 = np.ascontiguousarray(words24.transpose(0, 1, 3, 2, 4))

    return PackedCertV7NumPy(
        slabs_u4=slabs_u4,
        ms_u4=ms_u4,
        big_s=big_s,
        lsb1_row=lsb1_row,
        ms_row=ms_row,
        big_s_row=big_s_row,
        rt1=rt1,
        nv1=nv1,
        rt2=rt2,
        nv2=nv2,
        w1_bf16=w1_bf16,
        dw_bf16=dw_bf16,
        V=V,
        K=K,
        cfg_id=cfg_id,
    )


def unpack_cert_v7_ref_np(pk: PackedCertV7NumPy) -> tuple[np.ndarray, np.ndarray]:
    """Bit-exact NumPy mirror of `cert_screen_int6_ms8_v7` AND `cert_stage2_lsb1_v7` dequantization.

    Returns (w1_unpacked [V, K], dw_unpacked [V, K]) matching `pk.w1_bf16` and `pk.dw_bf16` to 0 diffs!
    """
    warps, sub_n, bn = _CFGS[pk.cfg_id]
    n_qchunks = pk.K // 512
    n_tiles = (pk.V + bn - 1) // bn
    v_pad = n_tiles * bn
    wtot = n_tiles * warps

    # 1. Unpack Stage-1 SLABS_U4 + MS_U4 + BIG_S exactly like `cert_screen_int6_ms8_v7`
    words = pk.slabs_u4.transpose(0, 1, 3, 2, 4).reshape(wtot, n_qchunks * sub_n, 32, 8, 3)
    w0, w1, w2 = words[..., 0], words[..., 1], words[..., 2]
    u24_0 = w0 & np.uint32(0xFFFFFF)
    u24_1 = ((w0 >> np.uint32(24)) | (w1 << np.uint32(8))) & np.uint32(0xFFFFFF)
    u24_2 = ((w1 >> np.uint32(16)) | (w2 << np.uint32(16))) & np.uint32(0xFFFFFF)
    u24_3 = (w2 >> np.uint32(8)) & np.uint32(0xFFFFFF)
    u24 = np.stack([u24_0, u24_1, u24_2, u24_3], axis=-1)  # [wtot, total_q, 32, 8(hc), 4(s)]

    q_vals = np.stack([
        (u24 & np.uint32(0x3F)).astype(np.int32) - 32,
        ((u24 >> np.uint32(6)) & np.uint32(0x3F)).astype(np.int32) - 32,
        ((u24 >> np.uint32(12)) & np.uint32(0x3F)).astype(np.int32) - 32,
        ((u24 >> np.uint32(18)) & np.uint32(0x3F)).astype(np.int32) - 32,
    ], axis=-1).astype(np.float32)  # [wtot, total_q, 32, 8, 4, 4]

    # Decode MS_U4 [wtot, total_q, 8(g), 4(ch)] -> [wtot, total_q, 8(g), 8(hc), 4(s), 2(lo/hi)]
    ms_c = np.zeros((wtot, n_qchunks * sub_n, 8, 64), dtype=np.uint32)
    for i in range(16):
        ms_c[..., 0 + i] = (pk.ms_u4[..., 0] >> np.uint32(2 * i)) & np.uint32(3)
        ms_c[..., 16 + i] = (pk.ms_u4[..., 1] >> np.uint32(2 * i)) & np.uint32(3)
        ms_c[..., 32 + i] = (pk.ms_u4[..., 2] >> np.uint32(2 * i)) & np.uint32(3)
        ms_c[..., 48 + i] = (pk.ms_u4[..., 3] >> np.uint32(2 * i)) & np.uint32(3)
    bs_f32 = pk.big_s.astype(np.float32)[:, :, :, None]
    sc_all = bs_f32 * MS_MULTS[ms_c]                       # [wtot, total_q, 8(g), 64]
    sc_all = sc_all.reshape(wtot, n_qchunks * sub_n, 8, 1, 8, 4, 2)  # broadcast across t=4
    q_gt = q_vals.reshape(wtot, n_qchunks * sub_n, 8, 4, 8, 4, 2, 2)
    w1_gt = bf16_rn_np(q_gt * sc_all[..., None])
    # Transpose back to [n_tiles, warps, sub_n, 8(g), n_qchunks, 8(hc), 4(s), 2(hi), 4(t), 2(pair)]
    w1_gt = w1_gt.reshape(n_tiles, warps, n_qchunks, sub_n, 8, 4, 8, 4, 2, 2)
    w1_unpacked = w1_gt.transpose(0, 1, 3, 4, 2, 6, 7, 8, 5, 9).reshape(v_pad, pk.K)[: pk.V]

    # 2. Unpack Stage-2 LSB1_ROW + MS_ROW + BIG_S_ROW exactly like `cert_stage2_lsb1_v7`
    ms_r = np.zeros((pk.V, n_qchunks, 64), dtype=np.uint32)
    for i in range(16):
        ms_r[..., 0 + i] = (pk.ms_row[..., 0] >> np.uint32(2 * i)) & np.uint32(3)
        ms_r[..., 16 + i] = (pk.ms_row[..., 1] >> np.uint32(2 * i)) & np.uint32(3)
        ms_r[..., 32 + i] = (pk.ms_row[..., 2] >> np.uint32(2 * i)) & np.uint32(3)
        ms_r[..., 48 + i] = (pk.ms_row[..., 3] >> np.uint32(2 * i)) & np.uint32(3)
    sc_s2 = (pk.big_s_row.astype(np.float32)[:, :, None] * np.float32(0.25)) * MS_MULTS[ms_r]

    bits_k = ((pk.lsb1_row[:, :, None] >> np.arange(32, dtype=np.uint32)[None, None, :]) & np.uint32(1))
    bits_k = bits_k.reshape(pk.V, n_qchunks, 64, 8)
    dw_unpacked = bf16_rn_np(np.where(bits_k == 1, sc_s2[:, :, :, None], -sc_s2[:, :, :, None])).reshape(pk.V, pk.K)
    return w1_unpacked, dw_unpacked


# ================= PyTorch / GB10 Runtime Wrapper =================


@dataclass
class CudaCertTableV7:
    slabs_u4: object
    ms_u4: object
    big_s: object
    lsb1_row: object
    ms_row: object
    big_s_row: object
    rt1: object
    nv1: object
    rt2: object
    nv2: object
    V: int
    K: int
    cfg_id: int
    cacc2: float

    def l2pf_segments(self, num_tokens: int | None = None, g2: bool = True):
        """Multi-segment L2PF descriptor for Layer-47 MoE AR overlap (compatible with l2pf._cuda_head_segments).

        Returns 5-tuple `(nv1, rt1, (big_s, bs_cap), (ms_u4, ms_cap), slabs_u4)` when `g2=True` so
        `l2pf.py`'s C32/C48 scale-cap logic recognizes BOTH `(big_s, cap)` and `(ms_u4, cap)` cleanly.
        """
        if not g2:
            return self.slabs_u4
        if num_tokens is not None and int(num_tokens) > 16:
            return (self.nv1, self.rt1, (self.big_s, 262144), (self.ms_u4, 1048576), self.slabs_u4)
        return (self.nv1, self.rt1, (self.big_s, 131072), (self.ms_u4, 524288), self.slabs_u4)


@dataclass
class CudaCertStateV7:
    x_pad: object
    xn: object
    bad: object
    s1_out: object
    hi: object
    thr_u: object
    thr_ag: object
    t_glob: object
    idx1: object
    cnt1: object
    idx2: object
    cnt2: object
    best: object
    arrive: object
    out_pair: object
    nlive_full: object
    mask1: object = None
    mask1_128: object = None
    alive_bm: object = None


def _load_lib():
    global _LIB, _HAS_C48, _HAS_V9, _HAS_V14, _LIB_VER, _LIB_C48_MAX_M
    if _LIB is not None:
        return _LIB
    with _LIB_LOCK:
        if _LIB is None and os.path.exists(_LIB_PATH):
            lib = ctypes.CDLL(_LIB_PATH)
            lib.cert_head_v7_pre_screen.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_int,
                ctypes.c_float, ctypes.c_float, ctypes.c_float,
                ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
                ctypes.c_uint64,
            ]
            lib.cert_head_v7_pre_screen.restype = ctypes.c_int

            lib.cert_head_v7_post_refine.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
                ctypes.c_float, ctypes.c_float, ctypes.c_float,
                ctypes.c_uint64,
            ]
            lib.cert_head_v7_post_refine.restype = ctypes.c_int

            if hasattr(lib, "cert_head_v7_post_refine_c48"):
                lib.cert_head_v7_post_refine_c48.argtypes = [
                    ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                    ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
                    ctypes.c_float, ctypes.c_float, ctypes.c_float,
                    ctypes.c_uint64,
                ]
                lib.cert_head_v7_post_refine_c48.restype = ctypes.c_int
                _HAS_C48 = True

            if hasattr(lib, "cert_head_v9_post_refine"):
                lib.cert_head_v9_post_refine.argtypes = [
                    ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                    ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
                    ctypes.c_float, ctypes.c_float, ctypes.c_float,
                    ctypes.c_uint64,
                ]
                lib.cert_head_v9_post_refine.restype = ctypes.c_int
                _HAS_V9 = True

            if hasattr(lib, "cert_head_v14_pre_screen") and hasattr(lib, "cert_head_v14_post_refine"):
                lib.cert_head_v14_pre_screen.argtypes = [
                    ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                    ctypes.c_int, ctypes.c_int,
                    ctypes.c_float, ctypes.c_float, ctypes.c_float,
                    ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
                    ctypes.c_float, ctypes.c_float, ctypes.c_float,
                    ctypes.c_uint64,
                ]
                lib.cert_head_v14_pre_screen.restype = ctypes.c_int

                lib.cert_head_v14_post_refine.argtypes = [
                    ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                    ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
                    ctypes.c_float, ctypes.c_float, ctypes.c_float,
                    ctypes.c_uint64,
                ]
                lib.cert_head_v14_post_refine.restype = ctypes.c_int
                _HAS_V14 = True

            if hasattr(lib, "cert_head_v9_max_m"):
                lib.cert_head_v9_max_m.argtypes = []
                lib.cert_head_v9_max_m.restype = ctypes.c_int
                _LIB_C48_MAX_M = max(16, min(128, int(lib.cert_head_v9_max_m())))
            elif hasattr(lib, "cert_head_v7_c48_max_m"):
                lib.cert_head_v7_c48_max_m.argtypes = []
                lib.cert_head_v7_c48_max_m.restype = ctypes.c_int
                _LIB_C48_MAX_M = max(16, min(64, int(lib.cert_head_v7_c48_max_m())))
            elif _HAS_C48:
                _LIB_C48_MAX_M = 64
            else:
                _LIB_C48_MAX_M = 16

            lib.cert_head_v7_exact_all.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_uint64,
            ]
            lib.cert_head_v7_exact_all.restype = ctypes.c_int
            if hasattr(lib, "cert_head_v14_version"):
                lib.cert_head_v14_version.argtypes = []
                lib.cert_head_v14_version.restype = ctypes.c_int
                _LIB_VER = int(lib.cert_head_v14_version())
            elif hasattr(lib, "cert_head_v12_version"):
                lib.cert_head_v12_version.argtypes = []
                lib.cert_head_v12_version.restype = ctypes.c_int
                _LIB_VER = int(lib.cert_head_v12_version())
            elif hasattr(lib, "cert_head_v7_version"):
                lib.cert_head_v7_version.argtypes = []
                lib.cert_head_v7_version.restype = ctypes.c_int
                _LIB_VER = int(lib.cert_head_v7_version())
            else:
                _LIB_VER = 9 if _HAS_V9 else 7
            _LIB = lib
    return _LIB


def lib_max_m() -> int:
    """Return effective max_m supported by both env (`INKLING_CERT_HEAD_V7_MAX_M` / `V14_MAX_M`) and loaded `.so`."""
    if _load_lib() is None:
        return 16
    return min(MAX_M, _LIB_C48_MAX_M) if (_HAS_V14 or _HAS_V9 or _HAS_C48) else 16


def get_l2pf_head_slabs(num_tokens: int | None = None, g2: bool = True):
    """Clean L2PF slab query helper for `l2pf.py` across M=2..128 (supports V7/V12/V14 Stage-1 slabs).

    Returns the active multi-segment 5-tuple `(nv1, rt1, (big_s, cap), (ms_u4, cap), slabs_u4)` (or
    `slabs_u4` when `g2=False`) if the active CertHead table is qualified for `num_tokens`, else `None`.
    """
    if not ENABLED or is_temp_direct_l2pf_bypass():
        return None
    ad = _ACTIVE_ADAPTER
    if ad is None or getattr(ad, "tbl", None) is None:
        return None
    if num_tokens is not None:
        m = int(num_tokens)
        if m < 2 or m > int(ad.max_m) or not bool(ad.m_ok.get(m, False)):
            return None
    return ad.tbl.l2pf_segments(num_tokens=num_tokens, g2=g2)


def register_l2pf_head_slabs(tbl: CudaCertTableV7 | None = None, l2pf_mod=None) -> bool:
    """Register active CertHead V7/V12/V14 Stage-1 slabs with `l2pf` and prewarm M=8..active_max_m."""
    t = tbl if tbl is not None else (getattr(_ACTIVE_ADAPTER, "tbl", None) if _ACTIVE_ADAPTER is not None else None)
    if t is None:
        return False
    if torch is not None and torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return False
    if l2pf_mod is None:
        try:
            from . import l2pf as l2pf_mod
        except Exception:
            try:
                import l2pf as l2pf_mod  # type: ignore
            except Exception:
                return False
    if not hasattr(l2pf_mod, "set_cuda_head_slabs"):
        return False
    use_g2 = bool(getattr(l2pf_mod, "G2", True))
    segs = t.l2pf_segments(num_tokens=None, g2=use_g2)
    l2pf_mod.set_cuda_head_slabs(segs)
    if hasattr(l2pf_mod, "prewarm"):
        l2pf_mod.prewarm(segs, kind="head")
        max_m = active_max_m_global()
        if max_m > 16:
            segs_hi = t.l2pf_segments(num_tokens=max_m, g2=use_g2)
            for m_pw in (28, 32, 48, 64, 128):
                if m_pw <= max_m:
                    l2pf_mod.prewarm(segs_hi, kind="head", num_tokens=m_pw)
    return True


def active_max_m_global() -> int:
    """Global active max_m query for `l2pf.get_head_weight(num_tokens)`."""
    if not ENABLED or _load_lib() is None:
        return 0
    return lib_max_m()


def is_m_qualified_global(M: int) -> bool:
    """True iff `M` actually passed `m_ok[M]` qualification on the active CertHead v7/v9/v12 adapter.
    Prevents `l2pf.get_head_weight(M)` from prefetching unused v7 slabs when `M > 16` fell back
    or when a `temp > 0` direct-logit batch is active (`INKLING_CERT_HEAD_TEMP_DIRECT_V16=1`)!"""
    if is_temp_direct_l2pf_bypass():
        return False
    if not ENABLED or _load_lib() is None or M < 2:
        return False
    ad = _ACTIVE_ADAPTER
    if ad is None:
        return 2 <= M <= 16
    if M <= ad.max_m:
        return bool(ad.m_ok.get(int(M), False))
    c32 = getattr(ad, "_c32_state", None)
    if c32 is not None and hasattr(c32, "m_ok"):
        return bool(c32.m_ok.get(int(M), False))
    return False


def active_summary_global() -> dict | None:
    """Structured summary of the active CertHead v7/v9/v12 adapter for accurate startup logging."""
    ad = _ACTIVE_ADAPTER
    if ad is None:
        return None
    lim = int(ad.max_m)
    ok_le16 = [m for m in range(2, min(16, lim) + 1) if ad.m_ok.get(m, False)]
    bad_le16 = [m for m in range(2, min(16, lim) + 1) if not ad.m_ok.get(m, False)]
    ok_gt16 = [m for m in range(17, lim + 1) if ad.m_ok.get(m, False)]
    bad_gt16 = [m for m in range(17, lim + 1) if not ad.m_ok.get(m, False)]
    s1_mb = float(ad.tbl.slabs_u4.numel() * ad.tbl.slabs_u4.element_size() / 1e6)
    return {
        "lib_ver": int(_LIB_VER or (9 if _HAS_V9 else 7)),
        "max_m": lim,
        "qualify_ref": str(_QUALIFY_REF),
        "ok_le16": len(ok_le16),
        "bad_le16": bad_le16,
        "ok_gt16": len(ok_gt16),
        "bad_gt16": bad_gt16,
        "s1_mb": s1_mb,
        "v14_enabled": bool(V14_ENABLED and _HAS_V14),
        "v14_max_m": int(V14_MAX_M),
        "temp_direct_v16": bool(TEMP_DIRECT_V16),
    }


def _wkey_v7(w, cfg_id: int) -> str:
    h = hashlib.sha256()
    h.update(f"v7.2:{cfg_id}:{tuple(w.shape)}:{w.dtype}".encode())
    sample = w[:: max(1, w.shape[0] // 64), :: max(1, w.shape[1] // 64)]
    h.update(memoryview(sample.detach().cpu().contiguous().view(torch.uint8).numpy().reshape(-1)))
    return h.hexdigest()[:16]


def _round_up_f32_torch(x64):
    f = x64.float()
    bump = f.double() < x64
    return torch.where(bump, torch.nextafter(f, torch.full_like(f, float("inf"))), f)


def _to_u32_torch(x_i64):
    return (x_i64 & 0xFFFFFFFF).to(torch.int32).view(torch.uint32).contiguous()


def pack_cert_v7_torch(w, cfg_id: int = _CFG_ID) -> CudaCertTableV7:
    """Fast (~15-25 ms) GPU PyTorch packer bit-identical to `pack_cert_v7_np`."""
    assert torch is not None
    dev = w.device
    w_bf = w.detach().to(torch.bfloat16)
    V, K = int(w_bf.shape[0]), int(w_bf.shape[1])
    assert K % 512 == 0, f"K={K} must be a multiple of 512"
    warps, sub_n, bn = _CFGS[cfg_id]
    n_qchunks = K // 512
    n_tiles = (V + bn - 1) // bn
    v_pad = n_tiles * bn
    wtot = n_tiles * warps

    with torch.no_grad():
        wf = w_bf.float()
        w_blk = wf.view(V, n_qchunks, 64, 8)
        m_abs = w_blk.abs().amax(dim=3)
        row_max = m_abs.amax(dim=2)

        bs_raw = (row_max / 31.0).to(torch.float16)
        bs_f32 = bs_raw.float()
        need_up = (row_max > 0.0) & (bs_f32 * 31.0 < row_max)
        pos_inf16 = torch.full_like(bs_raw, float("inf"))
        while bool(need_up.any()):
            bs_raw = torch.where(need_up, torch.nextafter(bs_raw, pos_inf16), bs_raw)
            bs_f32 = bs_raw.float()
            need_up = (row_max > 0.0) & (bs_f32 * 31.0 < row_max)

        big_s_row = bs_raw.contiguous()
        bs_exp = bs_f32.unsqueeze(-1)

        t0 = bs_exp * float(np.float32(0.375 * 31.0))
        t1 = bs_exp * float(np.float32(0.5625 * 31.0))
        t2 = bs_exp * float(np.float32(0.75 * 31.0))
        code2 = torch.where(
            m_abs <= t0,
            0,
            torch.where(m_abs <= t1, 1, torch.where(m_abs <= t2, 2, 3)),
        ).to(torch.int64)

        mults = torch.tensor([0.375, 0.5625, 0.75, 1.0], dtype=torch.float32, device=dev)
        sc_mb = bs_exp * mults[code2]
        inv_sc = torch.where(sc_mb > 0.0, 1.0 / sc_mb, torch.zeros_like(sc_mb))
        q1 = torch.round(w_blk * inv_sc.unsqueeze(-1)).clamp(-31.0, 31.0).to(torch.int64)
        u6 = q1 + 32

        w1_blk = (q1.float() * sc_mb.unsqueeze(-1)).to(torch.bfloat16).float()
        w1_bf16 = w1_blk.view(V, K)

        r1_blk = w_blk.double() - w1_blk.double()
        lsb_bit = (r1_blk >= 0.0).to(torch.int64)
        sc_s2 = (bs_exp * 0.25) * mults[code2]
        dw_blk = torch.where(lsb_bit == 1, sc_s2.unsqueeze(-1), -sc_s2.unsqueeze(-1)).to(torch.bfloat16).float()
        dw_bf16 = dw_blk.view(V, K)

        w_d = wf.double()
        w1_d = w1_bf16.double()
        dw_d = dw_bf16.double()
        r1_d = w_d - w1_d
        r2_d = w_d - (w1_d + dw_d)
        nw_d = torch.sqrt(torch.sum(w_d * w_d, dim=1))
        nw1_d = torch.sqrt(torch.sum(w1_d * w1_d, dim=1))
        ndw_d = torch.sqrt(torch.sum(dw_d * dw_d, dim=1))

        rt1 = _round_up_f32_torch(torch.sqrt(torch.sum(r1_d * r1_d, dim=1))).contiguous()
        nv1 = _round_up_f32_torch(torch.maximum(nw_d, nw1_d)).contiguous()
        rt2 = _round_up_f32_torch(torch.sqrt(torch.sum(r2_d * r2_d, dim=1))).contiguous()
        nv2 = _round_up_f32_torch(torch.maximum(nw_d, nw1_d) + ndw_d).contiguous()

        c2_4 = code2.view(V, n_qchunks, 4, 16)
        shifts_ms = (torch.arange(16, dtype=torch.int64, device=dev) * 2).view(1, 1, 1, 16)
        ms_row_i64 = torch.sum(c2_4 << shifts_ms, dim=3)
        ms_row = _to_u32_torch(ms_row_i64)

        lsb_32 = lsb_bit.view(V, n_qchunks * 16, 32)
        shifts_b = torch.arange(32, dtype=torch.int64, device=dev).view(1, 1, 32)
        lsb1_row = _to_u32_torch(torch.sum(lsb_32 << shifts_b, dim=2))

        if v_pad != V:
            u6_p = torch.full((v_pad, n_qchunks, 512), 32, dtype=torch.int64, device=dev)
            u6_p[:V] = u6.view(V, n_qchunks, 512)
            ms_p = torch.zeros((v_pad, n_qchunks, 4), dtype=torch.int64, device=dev)
            ms_p[:V] = ms_row_i64
            bs_p = torch.zeros((v_pad, n_qchunks), dtype=torch.float16, device=dev)
            bs_p[:V] = big_s_row
        else:
            u6_p = u6.view(V, n_qchunks, 512)
            ms_p = ms_row_i64
            bs_p = big_s_row

        ms_t = ms_p.view(n_tiles, warps, sub_n, 8, n_qchunks, 4).permute(0, 1, 4, 2, 3, 5).contiguous()
        ms_u4 = _to_u32_torch(ms_t.view(wtot, n_qchunks * sub_n, 8, 4))

        bs_t = bs_p.view(n_tiles, warps, sub_n, 8, n_qchunks).permute(0, 1, 4, 2, 3).contiguous()
        big_s = bs_t.view(wtot, n_qchunks * sub_n, 8).contiguous()

        u6_t = u6_p.view(n_tiles, warps, sub_n, 8, n_qchunks, 8, 4, 2, 4, 2)
        u6_t = u6_t.permute(0, 1, 4, 2, 3, 8, 5, 6, 7, 9).contiguous()
        u6_t = u6_t.view(wtot, n_qchunks * sub_n, 32, 8, 4, 4)

        u24 = (
            u6_t[..., 0]
            | (u6_t[..., 1] << 6)
            | (u6_t[..., 2] << 12)
            | (u6_t[..., 3] << 18)
        )
        w0 = u24[..., 0] | ((u24[..., 1] & 0xFF) << 24)
        w1 = (u24[..., 1] >> 8) | ((u24[..., 2] & 0xFFFF) << 16)
        w2 = (u24[..., 2] >> 16) | (u24[..., 3] << 8)
        words24 = torch.stack([w0, w1, w2], dim=-1).view(wtot, n_qchunks * sub_n, 32, 6, 4)
        slabs_u4 = _to_u32_torch(words24.permute(0, 1, 3, 2, 4))

    return CudaCertTableV7(
        slabs_u4=slabs_u4,
        ms_u4=ms_u4,
        big_s=big_s,
        lsb1_row=lsb1_row,
        ms_row=ms_row,
        big_s_row=big_s_row,
        rt1=rt1,
        nv1=nv1,
        rt2=rt2,
        nv2=nv2,
        V=V,
        K=K,
        cfg_id=cfg_id,
        cacc2=cacc2_for_k(K),
    )


def build_cuda_table_v7(w, cfg_id: int = _CFG_ID) -> CudaCertTableV7:
    """Pack or disk-cache load a CUDA bf16 LM-head shard `w [V, K]` into `CudaCertTableV7`."""
    assert torch is not None
    dev = w.device
    V, K = int(w.shape[0]), int(w.shape[1])
    cache_path = None
    if _CACHE_DIR:
        try:
            wkey = _wkey_v7(w, cfg_id)
            cache_path = os.path.join(_CACHE_DIR, f"cert-v7-{wkey}-{V}x{K}-cfg{cfg_id}.pt")
            if os.path.exists(cache_path):
                d = torch.load(cache_path, map_location=dev, weights_only=True)
                tbl = CudaCertTableV7(
                    slabs_u4=d["slabs_u4"].view(torch.uint32).contiguous(),
                    ms_u4=d["ms_u4"].view(torch.uint32).contiguous(),
                    big_s=d["big_s"].contiguous(),
                    lsb1_row=d["lsb1_row"].view(torch.uint32).contiguous(),
                    ms_row=d["ms_row"].view(torch.uint32).contiguous(),
                    big_s_row=d["big_s_row"].contiguous(),
                    rt1=d["rt1"].contiguous(),
                    nv1=d["nv1"].contiguous(),
                    rt2=d["rt2"].contiguous(),
                    nv2=d["nv2"].contiguous(),
                    V=V,
                    K=K,
                    cfg_id=cfg_id,
                    cacc2=cacc2_for_k(K),
                )
                if os.environ.get("INKLING_CERT_VERBOSE", "1") != "0":
                    s1_mb = (tbl.slabs_u4.nbytes + tbl.ms_u4.nbytes + tbl.big_s.nbytes + tbl.rt1.nbytes + tbl.nv1.nbytes) / 1e6
                    print(f"[cert-v7] loaded cached V={V} K={K} cfg={cfg_id} S1={s1_mb:.2f}MB ({cache_path})", flush=True)
                return tbl
        except Exception:
            pass

    if w.is_cuda:
        tbl = pack_cert_v7_torch(w, cfg_id=cfg_id)
    else:
        w_f32 = w.detach().float().cpu().numpy()
        pk = pack_cert_v7_np(w_f32, cfg_id=cfg_id)
        tbl = CudaCertTableV7(
            slabs_u4=torch.from_numpy(pk.slabs_u4).to(dev).contiguous(),
            ms_u4=torch.from_numpy(pk.ms_u4).to(dev).contiguous(),
            big_s=torch.from_numpy(pk.big_s).to(dev).contiguous(),
            lsb1_row=torch.from_numpy(pk.lsb1_row).to(dev).contiguous(),
            ms_row=torch.from_numpy(pk.ms_row).to(dev).contiguous(),
            big_s_row=torch.from_numpy(pk.big_s_row).to(dev).contiguous(),
            rt1=torch.from_numpy(pk.rt1).to(dev).contiguous(),
            nv1=torch.from_numpy(pk.nv1).to(dev).contiguous(),
            rt2=torch.from_numpy(pk.rt2).to(dev).contiguous(),
            nv2=torch.from_numpy(pk.nv2).to(dev).contiguous(),
            V=pk.V,
            K=pk.K,
            cfg_id=cfg_id,
            cacc2=cacc2_for_k(pk.K),
        )

    if cache_path is not None:
        try:
            os.makedirs(_CACHE_DIR, exist_ok=True)
            tmp = f"{cache_path}.{os.getpid()}.tmp"
            torch.save(
                {
                    "slabs_u4": tbl.slabs_u4.view(torch.int32).cpu(),
                    "ms_u4": tbl.ms_u4.view(torch.int32).cpu(),
                    "big_s": tbl.big_s.cpu(),
                    "lsb1_row": tbl.lsb1_row.view(torch.int32).cpu(),
                    "ms_row": tbl.ms_row.view(torch.int32).cpu(),
                    "big_s_row": tbl.big_s_row.cpu(),
                    "rt1": tbl.rt1.cpu(),
                    "nv1": tbl.nv1.cpu(),
                    "rt2": tbl.rt2.cpu(),
                    "nv2": tbl.nv2.cpu(),
                },
                tmp,
            )
            os.replace(tmp, cache_path)
        except Exception:
            pass

    if os.environ.get("INKLING_CERT_VERBOSE", "1") != "0":
        s1_mb = (tbl.slabs_u4.nbytes + tbl.ms_u4.nbytes + tbl.big_s.nbytes + tbl.rt1.nbytes + tbl.nv1.nbytes) / 1e6
        s2_mb = (tbl.lsb1_row.nbytes + tbl.ms_row.nbytes + tbl.big_s_row.nbytes + tbl.rt2.nbytes + tbl.nv2.nbytes) / 1e6
        print(f"[cert-v7] packed V={V} K={K} cfg={cfg_id}: S1={s1_mb:.2f}MB S2={s2_mb:.2f}MB", flush=True)
    return tbl


def alloc_state_v7(M: int, V: int, K: int, world_size: int, device) -> CudaCertStateV7:
    assert torch is not None
    m_rows = 8 if M <= 8 else (((M + 15) // 16) * 16)
    m_cap = max(16, m_rows)
    return CudaCertStateV7(
        x_pad=torch.empty((m_rows, K), dtype=torch.bfloat16, device=device),
        xn=torch.empty((m_cap,), dtype=torch.float32, device=device),
        bad=torch.empty((m_cap,), dtype=torch.int32, device=device),
        s1_out=torch.empty((M, V), dtype=torch.float32, device=device),
        hi=torch.empty((M, V), dtype=torch.float16, device=device),
        thr_u=torch.full((m_cap,), 0x007FFFFF, dtype=torch.uint32, device=device),
        thr_ag=torch.empty((world_size, m_cap), dtype=torch.uint32, device=device),
        t_glob=torch.empty((m_cap,), dtype=torch.float32, device=device),
        idx1=torch.empty((V,), dtype=torch.int32, device=device),
        cnt1=torch.zeros((1,), dtype=torch.int32, device=device),
        idx2=torch.empty((V,), dtype=torch.int32, device=device),
        cnt2=torch.zeros((1,), dtype=torch.int32, device=device),
        best=torch.zeros((m_cap,), dtype=torch.uint64, device=device),
        arrive=torch.zeros((1,), dtype=torch.int32, device=device),
        out_pair=torch.empty((m_cap, 4), dtype=torch.float32, device=device),
        nlive_full=torch.tensor([M], dtype=torch.int32, device=device),
        mask1=torch.empty((V,), dtype=torch.int64, device=device),
        mask1_128=torch.empty((V, 2), dtype=torch.int64, device=device) if (M > 64 or _HAS_V9 or _HAS_V14) else None,
        alive_bm=torch.zeros(((V + 31) // 32,), dtype=torch.uint32, device=device) if (V14_ENABLED or _HAS_V14) else None,
    )


def _encode_logits_idx_ptr(logits_idx) -> int:
    if logits_idx is None:
        return 0
    ptr = int(logits_idx.data_ptr())
    if torch is not None and logits_idx.dtype == torch.int32:
        return ptr | 1
    return ptr


def _inv_cap_of_v7(cap: float) -> float:
    return float(np.float32(1.0) / np.float32(cap)) if cap > 0.0 else 0.0


def certified_local_pairs_v7(
    x,
    w_bf16,
    table: CudaCertTableV7,
    state: CudaCertStateV7,
    vreal: int,
    vstart: int,
    cap: float = 30.0,
    ag_thr_fn=None,
    world_size: int = 1,
    logits_idx=None,
    n_live_buf=None,
    pdl: bool = _USE_PDL,
    slack: float = _SLACK,
):
    """Run full 0-ULP CertHead-v7 pipeline:
      1. `cert_head_v7_pre_screen` (optional direct LOGITS_IDX gather + 91.11 MB INT6_g512_ms8 screen)
      2. Optional 32B RDMA AG1 (`ag_thr_fn(state.thr_u, state.thr_ag)`)
      3. `cert_head_v7_post_refine` (`cert_cand1` -> `cert_stage2_lsb1` -> L2-prefetched `cert_refine_argmax`)
    Returns `state.out_pair` `[M, 4]` fp32 (`(max_val, global_idx, 0, 0)`).
    """
    lib = _load_lib()
    assert lib is not None and torch is not None
    M = int(logits_idx.shape[0]) if logits_idx is not None else int(x.shape[0])
    nl_buf = n_live_buf if n_live_buf is not None else state.nlive_full
    stream = int(torch.cuda.current_stream(x.device).cuda_stream)

    inv_cap = _inv_cap_of_v7(cap)
    use_v14 = bool(
        V14_ENABLED
        and _HAS_V14
        and getattr(state, "alive_bm", None) is not None
        and getattr(state, "mask1_128", None) is not None
    )
    if use_v14:
        rc = lib.cert_head_v14_pre_screen(
            int(x.data_ptr()),
            _encode_logits_idx_ptr(logits_idx),
            int(table.slabs_u4.data_ptr()),
            int(table.ms_u4.data_ptr()),
            int(table.big_s.data_ptr()),
            int(table.rt1.data_ptr()),
            int(table.nv1.data_ptr()),
            int(state.x_pad.data_ptr()),
            int(state.xn.data_ptr()),
            int(state.bad.data_ptr()),
            int(state.s1_out.data_ptr()),
            int(state.hi.data_ptr()),
            int(state.thr_u.data_ptr()),
            int(state.cnt1.data_ptr()),
            int(state.cnt2.data_ptr()),
            int(state.best.data_ptr()),
            int(state.arrive.data_ptr()),
            int(state.alive_bm.data_ptr()),
            int(nl_buf.data_ptr()),
            M, table.V, vreal, table.K, int(x.stride(0)),
            table.cfg_id, 1 if pdl else 0,
            table.cacc2, _INFL_X, _X_ABS, _INFL_Q, _INFL_B, _SLOP_REL, _SLOP_ABS,
            inv_cap, float(cap), float(slack),
            stream,
        )
    else:
        rc = lib.cert_head_v7_pre_screen(
            int(x.data_ptr()),
            _encode_logits_idx_ptr(logits_idx),
            int(table.slabs_u4.data_ptr()),
            int(table.ms_u4.data_ptr()),
            int(table.big_s.data_ptr()),
            int(table.rt1.data_ptr()),
            int(table.nv1.data_ptr()),
            int(state.x_pad.data_ptr()),
            int(state.xn.data_ptr()),
            int(state.bad.data_ptr()),
            int(state.s1_out.data_ptr()),
            int(state.hi.data_ptr()),
            int(state.thr_u.data_ptr()),
            int(state.cnt1.data_ptr()),
            int(state.cnt2.data_ptr()),
            int(state.best.data_ptr()),
            int(state.arrive.data_ptr()),
            int(nl_buf.data_ptr()),
            M, table.V, vreal, table.K, int(x.stride(0)),
            table.cfg_id, 1 if pdl else 0,
            table.cacc2, _INFL_X, _X_ABS, _INFL_Q, _INFL_B, _SLOP_REL, _SLOP_ABS,
            stream,
        )
    if rc != 0:
        raise RuntimeError(f"cert_head_{'v14' if use_v14 else 'v7'}_pre_screen failed: cudaError={rc}")

    if ag_thr_fn is not None and world_size > 1:
        ag_thr_fn(state.thr_u, state.thr_ag)
        thr_ag_ptr = int(state.thr_ag.data_ptr())
        ws = world_size
        tstride = int(state.thr_ag.stride(0))
    else:
        thr_ag_ptr = int(state.thr_u.data_ptr())
        ws = 1
        tstride = M

    if use_v14:
        rc = lib.cert_head_v14_post_refine(
            int(state.x_pad.data_ptr()),
            int(w_bf16.data_ptr()),
            int(table.lsb1_row.data_ptr()),
            int(table.ms_row.data_ptr()),
            int(table.big_s_row.data_ptr()),
            int(state.s1_out.data_ptr()),
            int(table.rt2.data_ptr()),
            int(table.nv2.data_ptr()),
            int(state.xn.data_ptr()),
            int(state.hi.data_ptr()),
            thr_ag_ptr,
            int(state.bad.data_ptr()),
            int(state.alive_bm.data_ptr()),
            int(state.t_glob.data_ptr()),
            int(state.idx1.data_ptr()),
            int(state.mask1_128.data_ptr()),
            int(state.cnt1.data_ptr()),
            int(state.idx2.data_ptr()),
            int(state.cnt2.data_ptr()),
            int(state.best.data_ptr()),
            int(state.arrive.data_ptr()),
            int(state.out_pair.data_ptr()),
            int(nl_buf.data_ptr()),
            M, table.V, vreal, vstart, table.K, int(w_bf16.stride(0)),
            ws, tstride, _REF_GRID, 1 if pdl else 0,
            table.cacc2, _INFL_Q, _INFL_B, _SLOP_REL, _SLOP_ABS,
            inv_cap, cap, slack,
            stream,
        )
    elif _HAS_V9 and getattr(state, "mask1_128", None) is not None:
        rc = lib.cert_head_v9_post_refine(
            int(state.x_pad.data_ptr()),
            int(w_bf16.data_ptr()),
            int(table.lsb1_row.data_ptr()),
            int(table.ms_row.data_ptr()),
            int(table.big_s_row.data_ptr()),
            int(state.s1_out.data_ptr()),
            int(table.rt2.data_ptr()),
            int(table.nv2.data_ptr()),
            int(state.xn.data_ptr()),
            int(state.hi.data_ptr()),
            thr_ag_ptr,
            int(state.bad.data_ptr()),
            int(state.t_glob.data_ptr()),
            int(state.idx1.data_ptr()),
            int(state.mask1_128.data_ptr()),
            int(state.cnt1.data_ptr()),
            int(state.idx2.data_ptr()),
            int(state.cnt2.data_ptr()),
            int(state.best.data_ptr()),
            int(state.arrive.data_ptr()),
            int(state.out_pair.data_ptr()),
            int(nl_buf.data_ptr()),
            M, table.V, vreal, vstart, table.K, int(w_bf16.stride(0)),
            ws, tstride, _REF_GRID, 1 if pdl else 0,
            table.cacc2, _INFL_Q, _INFL_B, _SLOP_REL, _SLOP_ABS,
            inv_cap, cap, slack,
            stream,
        )
    elif _HAS_C48 and getattr(state, "mask1", None) is not None:
        rc = lib.cert_head_v7_post_refine_c48(
            int(state.x_pad.data_ptr()),
            int(w_bf16.data_ptr()),
            int(table.lsb1_row.data_ptr()),
            int(table.ms_row.data_ptr()),
            int(table.big_s_row.data_ptr()),
            int(state.s1_out.data_ptr()),
            int(table.rt2.data_ptr()),
            int(table.nv2.data_ptr()),
            int(state.xn.data_ptr()),
            int(state.hi.data_ptr()),
            thr_ag_ptr,
            int(state.bad.data_ptr()),
            int(state.t_glob.data_ptr()),
            int(state.idx1.data_ptr()),
            int(state.mask1.data_ptr()),
            int(state.cnt1.data_ptr()),
            int(state.idx2.data_ptr()),
            int(state.cnt2.data_ptr()),
            int(state.best.data_ptr()),
            int(state.arrive.data_ptr()),
            int(state.out_pair.data_ptr()),
            int(nl_buf.data_ptr()),
            M, table.V, vreal, vstart, table.K, int(w_bf16.stride(0)),
            ws, tstride, _REF_GRID, 1 if pdl else 0,
            table.cacc2, _INFL_Q, _INFL_B, _SLOP_REL, _SLOP_ABS,
            inv_cap, cap, slack,
            stream,
        )
    else:
        rc = lib.cert_head_v7_post_refine(
            int(state.x_pad.data_ptr()),
            int(w_bf16.data_ptr()),
            int(table.lsb1_row.data_ptr()),
            int(table.ms_row.data_ptr()),
            int(table.big_s_row.data_ptr()),
            int(state.s1_out.data_ptr()),
            int(table.rt2.data_ptr()),
            int(table.nv2.data_ptr()),
            int(state.xn.data_ptr()),
            int(state.hi.data_ptr()),
            thr_ag_ptr,
            int(state.bad.data_ptr()),
            int(state.t_glob.data_ptr()),
            int(state.idx1.data_ptr()),
            int(state.cnt1.data_ptr()),
            int(state.idx2.data_ptr()),
            int(state.cnt2.data_ptr()),
            int(state.best.data_ptr()),
            int(state.arrive.data_ptr()),
            int(state.out_pair.data_ptr()),
            int(nl_buf.data_ptr()),
            M, table.V, vreal, vstart, table.K, int(w_bf16.stride(0)),
            ws, tstride, _REF_GRID, 1 if pdl else 0,
            table.cacc2, _INFL_Q, _INFL_B, _SLOP_REL, _SLOP_ABS,
            inv_cap, cap, slack,
            stream,
        )
    if rc != 0:
        raise RuntimeError(f"cert_head_v7_post_refine failed: cudaError={rc}")
    return state.out_pair[:M]


def _get_tp_world_size() -> int:
    try:
        from vllm.distributed import get_tensor_model_parallel_world_size
        return int(get_tensor_model_parallel_world_size())
    except Exception:
        return 1


class CudaCertStateV7Adapter:
    """Drop-in adapter compatible with sync_free.py's CudaCertState interface
    (serves M=2..INKLING_CERT_HEAD_V7_MAX_M [up to 64] via CertHead-v7 over the same 91.20 MB
    Stage-1 table with 0 extra table VRAM, and optionally delegates M>max_m to CertHead-C32)."""

    def __init__(self, tbl: CudaCertTableV7, W, vreal: int, vstart: int, cap: float = 30.0, dip: float = 0.0):
        assert torch is not None
        self.tbl = tbl
        self.W = W
        self.vreal = int(vreal)
        self.vstart = int(vstart)
        self.cap = float(np.float32(cap)) if cap else 0.0
        self.inv_cap = _inv_cap_of_v7(self.cap)
        self.dip = float(dip)
        self.slack = slack_of_v7(self.dip) if self.dip > 0.0 else (_SLACK if self.cap > 0.0 else 0.0)
        self.world_size = _get_tp_world_size()
        self.max_m = lib_max_m()
        self.state = alloc_state_v7(self.max_m, tbl.V, tbl.K, max(1, self.world_size), W.device)
        self._nlive_full = {m: torch.tensor([m], dtype=torch.int32, device=W.device) for m in range(1, self.max_m + 1)}
        self.m_ok: dict[int, bool] = {}
        self.is_v7 = True
        self._c32_state = None
        if _DELEGATE_C32:
            try:
                from . import cert_head_cuda as chc
            except Exception:
                try:
                    import cert_head_cuda as chc  # type: ignore
                except Exception:
                    chc = None  # type: ignore
            if chc is not None and chc.ENABLED and chc.C32_ENABLED and chc._get_lib() is not None and chc._HAS_C32:
                try:
                    ct_c32 = chc.build_cuda_table(W, self.vreal)
                    cand_c32 = chc.CudaCertState(ct_c32, W, vstart=self.vstart, cap=cap, dip=dip)
                    cand_c32.qualify(cand_c32.active_max_m())
                    self._c32_state = cand_c32
                    print(
                        "[inkling-cert-v7] WARNING: delegating C=32 to v6 allocates +100.3 MiB duplicate "
                        "table (slower at M>=48; set INKLING_CERT_HEAD_C32=0 to reclaim)",
                        flush=True,
                    )
                except Exception:
                    self._c32_state = None

    def active_max_m(self) -> int:
        c32_m = int(self._c32_state.active_max_m()) if (
            self._c32_state is not None and hasattr(self._c32_state, "active_max_m")
        ) else 0
        return max(self.max_m, c32_m)

    def qualify_one(self, M: int) -> bool:
        """Prove literal 0-ULP bit-equality of CertHead-v7 refine MMA vs torch.mm(out_dtype=fp32)."""
        cached = self.m_ok.get(M)
        if cached is not None:
            return cached
        lib = _load_lib()
        if lib is None or M < 2 or M > self.max_m or torch.cuda.is_current_stream_capturing():
            return False
        V, K = self.tbl.V, self.tbl.K
        g = torch.Generator(device=self.W.device)
        g.manual_seed(1234 + M)
        ok = True
        m_pad = 8 if M <= 8 else (((M + 15) // 16) * 16)
        for sc in (1.0, 0.3, 4.0):
            x = (torch.randn((M, K), device=self.W.device, generator=g) * sc).to(torch.bfloat16)
            if _QUALIFY_REF == "b1_exact" and M > 16:
                # Wave-V12 fix: pad tail to uniform 16-row tiles so `M % 16 == 1` (17, 33, 49, 65, ...)
                # never dispatches cuBLASLt `(1, K)` GEMV instead of `m16n8k16`, proving exact 0-ULP
                # row-for-row bit-equality with `B=1` `exact_ink.py` across ALL `M = 2..128`!
                if M == m_pad:
                    x_q = x
                else:
                    x_q = torch.zeros((m_pad, K), dtype=x.dtype, device=x.device)
                    x_q[:M].copy_(x)
                ref = torch.cat(
                    [torch.mm(x_q[m0 : m0 + 16], self.W.t(), out_dtype=torch.float32) for m0 in range(0, m_pad, 16)],
                    dim=0,
                )[:M]
            else:
                ref = torch.mm(x, self.W.t(), out_dtype=torch.float32)
            self.state.x_pad[:M].copy_(x)
            if M < m_pad and m_pad <= self.state.x_pad.shape[0]:
                self.state.x_pad[M:m_pad].zero_()
            out = torch.empty_like(ref)
            rc = lib.cert_head_v7_exact_all(
                int(self.state.x_pad.data_ptr()), int(self.W.data_ptr()), int(out.data_ptr()),
                int(self._nlive_full[M].data_ptr()), M, V, K, int(self.W.stride(0)),
                _REF_GRID, int(torch.cuda.current_stream(self.W.device).cuda_stream),
            )
            if rc != 0 or not bool(torch.equal(out.view(torch.int32), ref.view(torch.int32))):
                ok = False
                break
        self.m_ok[M] = ok
        return ok

    def qualify(self, max_m: int | None = None) -> list[int]:
        lib = _load_lib()
        # sync_free.py:1043 calls cand_v7.qualify(16); when C48 is enabled (self.max_m > 16),
        # qualify all M=2..self.max_m up front (graph-capture safe!) and fail-closed any M>16
        # individually without disqualifying M=2..16.
        sync_free_compat = (max_m == 16 and self.max_m > 16)
        lim = self.max_m if (max_m is None or sync_free_compat) else min(self.max_m, int(max_m))
        if lib is None:
            return list(range(2, min(16, lim) + 1))
        bad = []
        for M in range(2, lim + 1):
            if not self.qualify_one(M):
                bad.append(M)
        global _ACTIVE_ADAPTER
        if sync_free_compat:
            bad_le16 = [m for m in bad if m <= 16]
            bad_gt16 = [m for m in bad if m > 16]
            if not bad_le16:
                _ACTIVE_ADAPTER = self
                if V14_ENABLED:
                    try:
                        register_l2pf_head_slabs(self.tbl)
                    except Exception:
                        pass
            if os.environ.get("INKLING_CERT_VERBOSE", "1") != "0":
                _ver_tag = _LIB_VER or (14 if _HAS_V14 else (9 if _HAS_V9 else 7))
                print(
                    f"[cert-v{_ver_tag}] C48 qualify (ref={_QUALIFY_REF}, v14={int(V14_ENABLED and _HAS_V14)}) M=2..{lim}: "
                    f"ok={lim - 1 - len(bad)}/{lim - 1} "
                    f"(bad_le16={bad_le16}, bad_gt16={bad_gt16})",
                    flush=True,
                )
            return bad_le16
        if not bad:
            _ACTIVE_ADAPTER = self
            if V14_ENABLED:
                try:
                    register_l2pf_head_slabs(self.tbl)
                except Exception:
                    pass
        return bad

    def direct_logits_handoff(
        self,
        x,
        out_dtype=None,
        shared_prefix_groups: tuple[int, ...] | list[int] | None = None,
    ):
        """Zero-sync direct BF16/FP32 lm_head handoff for `temp > 0` batches (0 CertHead screening,
        avoids redundant `.contiguous()` copies when `x.stride(1) == 1`, supports `n=4` shared-prefix dedup)."""
        assert torch is not None
        x_in = x if (x.dtype == torch.bfloat16 and x.stride(1) == 1) else x.to(torch.bfloat16).contiguous()
        if (
            TEMP_DIRECT_V16
            and shared_prefix_groups is not None
            and x_in.dim() == 2
            and len(shared_prefix_groups) == int(x_in.shape[0])
            and len(shared_prefix_groups) > 1
        ):
            first_idx: dict[int, int] = {}
            uniq_rows: list[int] = []
            gather_idx: list[int] = []
            for row_i, g_id in enumerate(shared_prefix_groups):
                u_pos = first_idx.get(g_id)
                if u_pos is None:
                    u_pos = len(uniq_rows)
                    first_idx[g_id] = u_pos
                    uniq_rows.append(row_i)
                gather_idx.append(u_pos)
            if 0 < len(uniq_rows) < len(shared_prefix_groups):
                idx_u = torch.tensor(uniq_rows, dtype=torch.int64, device=x_in.device)
                idx_g = torch.tensor(gather_idx, dtype=torch.int64, device=x_in.device)
                x_u = x_in.index_select(0, idx_u)
                out_u = (
                    torch.mm(x_u, self.W.t(), out_dtype=out_dtype)
                    if out_dtype is not None
                    else torch.mm(x_u, self.W.t())
                )
                return out_u.index_select(0, idx_g)
        if out_dtype is not None:
            return torch.mm(x_in, self.W.t(), out_dtype=out_dtype)
        return torch.mm(x_in, self.W.t())

    def local_pairs(self, x, nlive=None, logits_idx=None, pdl: bool = _USE_PDL):
        if TEMP_DIRECT_V16 and _TEMP_DIRECT_STEP_ACTIVE:
            return None
        M = int(logits_idx.shape[0]) if logits_idx is not None else int(x.shape[0])
        if M > self.max_m and self._c32_state is not None and logits_idx is None:
            try:
                from . import cert_head_cuda as chc
            except Exception:
                import cert_head_cuda as chc  # type: ignore
            return chc.certified_local_pairs(self._c32_state, x, nlive=nlive, pdl=pdl)
        if not ENABLED or M < 2 or M > self.max_m:
            return None
        if M not in self.m_ok and not torch.cuda.is_current_stream_capturing():
            self.qualify_one(M)
        if not self.m_ok.get(M, False):
            return None
        lib = _load_lib()
        if lib is None:
            return None
        if x.dtype != torch.bfloat16 or x.stride(1) != 1:
            x = x.contiguous()
        nl_buf = nlive if nlive is not None else self._nlive_full[M]
        stream = int(torch.cuda.current_stream(x.device).cuda_stream)
        t = self.tbl
        st = self.state
        use_v14 = bool(
            V14_ENABLED
            and _HAS_V14
            and getattr(st, "alive_bm", None) is not None
            and getattr(st, "mask1_128", None) is not None
        )
        if use_v14:
            rc = lib.cert_head_v14_pre_screen(
                int(x.data_ptr()),
                _encode_logits_idx_ptr(logits_idx),
                int(t.slabs_u4.data_ptr()),
                int(t.ms_u4.data_ptr()),
                int(t.big_s.data_ptr()),
                int(t.rt1.data_ptr()),
                int(t.nv1.data_ptr()),
                int(st.x_pad.data_ptr()),
                int(st.xn.data_ptr()),
                int(st.bad.data_ptr()),
                int(st.s1_out.data_ptr()),
                int(st.hi.data_ptr()),
                int(st.thr_u.data_ptr()),
                int(st.cnt1.data_ptr()),
                int(st.cnt2.data_ptr()),
                int(st.best.data_ptr()),
                int(st.arrive.data_ptr()),
                int(st.alive_bm.data_ptr()),
                int(nl_buf.data_ptr()),
                M, t.V, self.vreal, t.K, int(x.stride(0)),
                t.cfg_id, 1 if pdl else 0,
                t.cacc2, _INFL_X, _X_ABS, _INFL_Q, _INFL_B, _SLOP_REL, _SLOP_ABS,
                self.inv_cap, self.cap, self.slack,
                stream,
            )
        else:
            rc = lib.cert_head_v7_pre_screen(
                int(x.data_ptr()),
                _encode_logits_idx_ptr(logits_idx),
                int(t.slabs_u4.data_ptr()),
                int(t.ms_u4.data_ptr()),
                int(t.big_s.data_ptr()),
                int(t.rt1.data_ptr()),
                int(t.nv1.data_ptr()),
                int(st.x_pad.data_ptr()),
                int(st.xn.data_ptr()),
                int(st.bad.data_ptr()),
                int(st.s1_out.data_ptr()),
                int(st.hi.data_ptr()),
                int(st.thr_u.data_ptr()),
                int(st.cnt1.data_ptr()),
                int(st.cnt2.data_ptr()),
                int(st.best.data_ptr()),
                int(st.arrive.data_ptr()),
                int(nl_buf.data_ptr()),
                M, t.V, self.vreal, t.K, int(x.stride(0)),
                t.cfg_id, 1 if pdl else 0,
                t.cacc2, _INFL_X, _X_ABS, _INFL_Q, _INFL_B, _SLOP_REL, _SLOP_ABS,
                stream,
            )
        if rc != 0:
            raise RuntimeError(f"cert_head_{'v14' if use_v14 else 'v7'}_pre_screen failed: cudaError={rc}")

        if self.world_size > 1:
            thr_stride = ((M + 3) // 4) * 4
            thr_ag = _get_tp_all_gather()(st.thr_u[:thr_stride].view(torch.int32), dim=0)
            if not thr_ag.is_contiguous():
                thr_ag = thr_ag.contiguous()
            thr_ag_ptr = int(thr_ag.data_ptr())
            ws = self.world_size
        else:
            thr_stride = M
            thr_ag_ptr = int(st.thr_u.data_ptr())
            ws = 1

        if use_v14:
            rc = lib.cert_head_v14_post_refine(
                int(st.x_pad.data_ptr()),
                int(self.W.data_ptr()),
                int(t.lsb1_row.data_ptr()),
                int(t.ms_row.data_ptr()),
                int(t.big_s_row.data_ptr()),
                int(st.s1_out.data_ptr()),
                int(t.rt2.data_ptr()),
                int(t.nv2.data_ptr()),
                int(st.xn.data_ptr()),
                int(st.hi.data_ptr()),
                thr_ag_ptr,
                int(st.bad.data_ptr()),
                int(st.alive_bm.data_ptr()),
                int(st.t_glob.data_ptr()),
                int(st.idx1.data_ptr()),
                int(st.mask1_128.data_ptr()),
                int(st.cnt1.data_ptr()),
                int(st.idx2.data_ptr()),
                int(st.cnt2.data_ptr()),
                int(st.best.data_ptr()),
                int(st.arrive.data_ptr()),
                int(st.out_pair.data_ptr()),
                int(nl_buf.data_ptr()),
                M, t.V, self.vreal, self.vstart, t.K, int(self.W.stride(0)),
                ws, thr_stride, _REF_GRID, 1 if pdl else 0,
                t.cacc2, _INFL_Q, _INFL_B, _SLOP_REL, _SLOP_ABS,
                self.inv_cap, self.cap, self.slack,
                stream,
            )
        elif _HAS_V9 and getattr(st, "mask1_128", None) is not None:
            rc = lib.cert_head_v9_post_refine(
                int(st.x_pad.data_ptr()),
                int(self.W.data_ptr()),
                int(t.lsb1_row.data_ptr()),
                int(t.ms_row.data_ptr()),
                int(t.big_s_row.data_ptr()),
                int(st.s1_out.data_ptr()),
                int(t.rt2.data_ptr()),
                int(t.nv2.data_ptr()),
                int(st.xn.data_ptr()),
                int(st.hi.data_ptr()),
                thr_ag_ptr,
                int(st.bad.data_ptr()),
                int(st.t_glob.data_ptr()),
                int(st.idx1.data_ptr()),
                int(st.mask1_128.data_ptr()),
                int(st.cnt1.data_ptr()),
                int(st.idx2.data_ptr()),
                int(st.cnt2.data_ptr()),
                int(st.best.data_ptr()),
                int(st.arrive.data_ptr()),
                int(st.out_pair.data_ptr()),
                int(nl_buf.data_ptr()),
                M, t.V, self.vreal, self.vstart, t.K, int(self.W.stride(0)),
                ws, thr_stride, _REF_GRID, 1 if pdl else 0,
                t.cacc2, _INFL_Q, _INFL_B, _SLOP_REL, _SLOP_ABS,
                self.inv_cap, self.cap, self.slack,
                stream,
            )
        elif _HAS_C48 and getattr(st, "mask1", None) is not None:
            rc = lib.cert_head_v7_post_refine_c48(
                int(st.x_pad.data_ptr()),
                int(self.W.data_ptr()),
                int(t.lsb1_row.data_ptr()),
                int(t.ms_row.data_ptr()),
                int(t.big_s_row.data_ptr()),
                int(st.s1_out.data_ptr()),
                int(t.rt2.data_ptr()),
                int(t.nv2.data_ptr()),
                int(st.xn.data_ptr()),
                int(st.hi.data_ptr()),
                thr_ag_ptr,
                int(st.bad.data_ptr()),
                int(st.t_glob.data_ptr()),
                int(st.idx1.data_ptr()),
                int(st.mask1.data_ptr()),
                int(st.cnt1.data_ptr()),
                int(st.idx2.data_ptr()),
                int(st.cnt2.data_ptr()),
                int(st.best.data_ptr()),
                int(st.arrive.data_ptr()),
                int(st.out_pair.data_ptr()),
                int(nl_buf.data_ptr()),
                M, t.V, self.vreal, self.vstart, t.K, int(self.W.stride(0)),
                ws, thr_stride, _REF_GRID, 1 if pdl else 0,
                t.cacc2, _INFL_Q, _INFL_B, _SLOP_REL, _SLOP_ABS,
                self.inv_cap, self.cap, self.slack,
                stream,
            )
        else:
            rc = lib.cert_head_v7_post_refine(
                int(st.x_pad.data_ptr()),
                int(self.W.data_ptr()),
                int(t.lsb1_row.data_ptr()),
                int(t.ms_row.data_ptr()),
                int(t.big_s_row.data_ptr()),
                int(st.s1_out.data_ptr()),
                int(t.rt2.data_ptr()),
                int(t.nv2.data_ptr()),
                int(st.xn.data_ptr()),
                int(st.hi.data_ptr()),
                thr_ag_ptr,
                int(st.bad.data_ptr()),
                int(st.t_glob.data_ptr()),
                int(st.idx1.data_ptr()),
                int(st.cnt1.data_ptr()),
                int(st.idx2.data_ptr()),
                int(st.cnt2.data_ptr()),
                int(st.best.data_ptr()),
                int(st.arrive.data_ptr()),
                int(st.out_pair.data_ptr()),
                int(nl_buf.data_ptr()),
                M, t.V, self.vreal, self.vstart, t.K, int(self.W.stride(0)),
                ws, thr_stride, _REF_GRID, 1 if pdl else 0,
                t.cacc2, _INFL_Q, _INFL_B, _SLOP_REL, _SLOP_ABS,
                self.inv_cap, self.cap, self.slack,
                stream,
            )
        if rc != 0:
            raise RuntimeError(f"cert_head_v7_post_refine failed: cudaError={rc}")
        return st.out_pair[:M]
