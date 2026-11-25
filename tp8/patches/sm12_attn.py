# SPDX-License-Identifier: Apache-2.0
"""SM12x (GB10) paged relative-bias attention for Inkling2 (Triton, mma.sync).

Drop-in for Inkling2FA4RelAttentionKernel.kernel on devices where tml_fa4 has
no paged-KV / bias path. Semantics match ``_fa4_rel_attention_cute``:
  s = (q . k) * scale + bias,  bias = rel[t, h, qpos - kpos] if 0 <= d < E else 0
read straight from the sheared ``(T + 128, H, E + 256)`` table written by
``attention_prologue`` (no unshear gather). Causal, optional left window,
GQA heads packed into the M dimension, split-KV + merge. Writes fp32 ``out``
``(T, H, D)`` and natural-log ``lse`` ``(H, T)``. CUDA-graph safe (no host
reads of device metadata).
"""
import ctypes
import os

import torch

from vllm.triton_utils import tl, triton

LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453

# INKLING_ATTN_BM16=1 (default): decode rows <= 64 use BLOCK_M=16 tiles (3 tiles
# for 8 tokens x 6 heads) instead of one BLOCK_M=64 tile.  The 64-row tile spills
# 218 registers in the windowed layers.  Bit-identical to the BM=64 launch (see
# REQ_BOUNDS in _attn_kernel).  0 restores the previous launch exactly.
_BM16 = os.environ.get("INKLING_ATTN_BM16", "1") != "0"
# INKLING_ATTN_FUSE_GS=1: rel_attention defers the split merge; the following
# attention_ops.gate_sink() runs merge + gate/sink as ONE kernel (saves a launch and a
# graph-boundary gap per layer).  Bit-identical to merge -> gate_sink (same math,
# fp32 intermediates stay in registers).  Default OFF (structural coupling through
# a module slot); enabled by attention_ops.gate_sink when set.
_FUSE_GS = os.environ.get("INKLING_ATTN_FUSE_GS", "0") == "1"
# INKLING_WO_L2PF=1: decode attention chain for the wo_ud GEMM.  _attn_kernel is launched
# with PDL (it gdc_waits for the prologue, then triggers its dependent at once) and
# _merge_gate_sink_kernel triggers right after its own gdc_wait, so wo_ud's CTAs
# (linear.py, skinny.py pdl=True) are resident and L2-prefetching their W slice
# while the 7 + 3 us attention/merge kernels run.  Scheduling only: bit-identical.
# Needs INKLING_ATTN_FUSE_GS=1 (the unfused merge/gate_sink live in attention_ops.py and
# are not PDL triggers); without it this flag is a no-op here and in linear.py.
_PDL = os.environ.get("INKLING_WO_L2PF", "0") == "1" and _FUSE_GS
# INKLING_DRAFT_PDL=1: opt-in PDL on draft_block_attention (_attn_kernel + _merge_kernel
# when draft=True). ZERO change to target attention (draft=False).
_DRAFT_PDL = os.environ.get("INKLING_DRAFT_PDL", "0").strip().lower() not in (
    "0", "", "off", "false", "no"
)
_PENDING = None  # (out, lse, op, lp, num_splits) of a deferred merge
_PENDING_PREFILL = None  # deferred 0-ULP prefill+gate_sink launch args (num_splits == 1)
_PENDING_S1 = None  # deferred 0-ULP decode S=1 fused attn+gate_sink launch args
_PREFILL_OPT = os.environ.get("INKLING_ATTN_PREFILL_OPT", "1").strip().lower() not in ("0", "", "off", "false", "no")
# Host-side flag: True between consume_pending() having launched the PDL-triggering
# _merge_gate_sink_kernel and the very next wo_ud skinny launch (linear._pdl_site), which
# consumes (clears) it.  Set at capture/trace time only, so it is identical on every rank
# and safe under CUDA graphs.  The drafter's o_proj has the same [4608,768] shape but is
# never behind a triggering merge, so it never sees the flag set.
_WO_ARMED = False
_BM = int(os.environ.get("INKLING_ATTN_BM", "16"))  # tile rows for decode rows in (16, 64]
_BM16_BN = int(os.environ.get("INKLING_ATTN_BM16_BN", "64"))
_BM16_WARPS = int(os.environ.get("INKLING_ATTN_BM16_WARPS", "4"))
_BM16_STAGES = int(os.environ.get("INKLING_ATTN_BM16_STAGES", "2"))
_C32_STAGES2 = os.environ.get("INKLING_ATTN_C32_STAGES2", "0").strip().lower() in ("1", "true", "on", "yes")
_DRAFT_SPLIT_CONC = os.environ.get("INKLING_DRAFT_SPLIT_CONC", "0").strip().lower() in ("1", "true", "on", "yes")
_ATTN_NLIVE = os.environ.get("INKLING_ATTN_NLIVE", "1").strip().lower() not in ("0", "", "off", "false", "no")
# INKLING_MGS_CUDA=1: 0-ULP 1-wave 48-CTA CUDA C++ merge_gate_sink_exact_fwd
# (0 bar.sync barriers vs 64 in Triton, preloads GATE/SINK before gdc_wait).
_MGS_CUDA = os.environ.get("INKLING_MGS_CUDA", "0") == "1"
# INKLING_ATTN_T16_SPLITK=1: fix T=16 decode cliff (rows=96 > 64 drops to num_splits=1);
# default OFF since changing num_splits from 1->8/16 on T=16 changes fp32 split order.
_T16_SPLITK = os.environ.get("INKLING_ATTN_T16_SPLITK", "0") == "1"
_TREE_ATTN_FAST_EXIT = os.environ.get("INKLING_TREE_ATTN_FAST_EXIT", "0").strip().lower() in (
    "1", "true", "on", "yes"
)
# INKLING_SM12_FUSE_GS_S1=1: 0-ULP fused Gate+Sink epilogue inside `_attn_kernel` when
# `NUM_SPLITS == 1 and not DRAFT` (all C=32 decode batches B >= 5), eliminating 48 separate
# `_gate_sink_kernel` launches + up to 96 `zero_()` launches + 37.7 MiB/step FP32 out/lse traffic!
_FUSE_GS_S1 = os.environ.get("INKLING_SM12_FUSE_GS_S1", "0").strip().lower() not in (
    "0", "", "off", "false", "no"
) and _FUSE_GS
# Wave-V7 C=2..8 Target & Drafter Attention Split Policy (opt-in, default OFF; 0 change at B=1 and B>=9!):
#   - Eliminates B=3..4 SWA-1024 over-splitting (S=8 -> S=1 + _PENDING_S1 fused Gate+Sink + stages=2)
#     while keeping B=2 SWA at S=8 (48 CTAs exact + CUDA MGS) to prevent 6-CTA SM starvation.
#   - Eliminates B=5..8 Global Attention S=1 cliff (50-69% idle SMs -> 84..96 CTAs = 1.75-2.0 CTAs/SM + stages=2).
#   - Wave-balances drafter_attn_v5 across B=2..8 (42..48 CTAs = 0 double-loaded SM tail).
_C48_SPLIT_POLICY = os.environ.get("INKLING_ATTN_C48_SPLIT_POLICY", "0").strip().lower() in (
    "1", "true", "on", "yes"
)
# Wave-V15 Drafter-Only 48-SM Attention Split Policy (opt-in, default OFF):
#   - Lifts B=2..8 drafter attention from 33-67% SM util / 1.33-wave tails to 42..48 CTAs
#     (100% 48-SM saturation at B=2G,3,4,6G,8) while leaving target attention (draft=False)
#     and B=1 100% untouched (0-ULP 1dcfb306 invariant!).
_DRAFT_ATTN_C48_SPLIT_POLICY = os.environ.get(
    "INKLING_DRAFT_ATTN_C48_SPLIT_POLICY", "0"
).strip().lower() in ("1", "true", "on", "yes")
# Wave-V15 Decode Edge Guard (opt-in, default OFF):
#   - Enables GPU-only (0 host sync, CUDA-graph-safe) sequence-length clamping to
#     `block_table.shape[1] * BS` before external fixed-ABI CUDA kernels.
_DECODE_EDGE_GUARD_V15 = os.environ.get(
    "INKLING_DECODE_EDGE_GUARD_V15", "0"
).strip().lower() in ("1", "true", "on", "yes")
_C48_LOCAL_SPLITS = max(1, min(16, int(os.environ.get("INKLING_ATTN_C48_LOCAL_SPLITS", "1") or "1")))
_C48_LOCAL_SPLITS_B2 = max(1, min(16, int(os.environ.get("INKLING_ATTN_C48_LOCAL_SPLITS_B2", "8") or "8")))
_C48_GLOBAL_TARGET_CTAS = max(
    24, min(192, int(os.environ.get("INKLING_ATTN_C48_GLOBAL_TARGET_CTAS", "96") or "96"))
)


def _parse_c48_split_overrides(raw: str) -> dict[int, int]:
    out: dict[int, int] = {}
    for item in (raw or "").split(","):
        item = item.strip()
        if not item or ":" not in item:
            continue
        b_s, s_s = item.split(":", 1)
        out[int(b_s.strip())] = max(1, min(16, int(s_s.strip())))
    return out


_C48_GLOBAL_SPLITS_MAP = _parse_c48_split_overrides(
    os.environ.get("INKLING_ATTN_C48_GLOBAL_SPLITS", "")
)


def _plan_target_splits(
    num_reqs: int,
    rows: int,
    mtiles: int,
    window: int,
    split_row_cap: int,
    *,
    draft: bool = False,
) -> int:
    """CUDA-graph-safe static split planner (depends ONLY on static capture args).
    Guarantees 0 behavior change when INKLING_ATTN_C48_SPLIT_POLICY=0 (and when
    INKLING_DRAFT_ATTN_C48_SPLIT_POLICY=1, target draft=False remains 100% untouched),
    and 0 change at B == 1 (exact_ink 0-ULP invariant) and B >= 9 even when enabled."""
    if not _C48_SPLIT_POLICY:
        if (
            _DRAFT_ATTN_C48_SPLIT_POLICY
            and draft
            and rows <= split_row_cap
            and 2 <= num_reqs <= 8
        ):
            base = max(1, mtiles * num_reqs)
            return max(1, min(16, 48 // base))
        if rows <= split_row_cap and num_reqs <= 4:
            return 8 if window >= 0 else 16
        if _DRAFT_SPLIT_CONC and draft and rows <= split_row_cap and num_reqs > 4:
            return 2 if (num_reqs <= 16 or window < 0) else 1
        return 1
    if rows > split_row_cap:
        return 1
    if draft:
        if num_reqs <= 1:
            return 8 if window >= 0 else 16
        if num_reqs <= 8:
            return 1 if window >= 0 else 2
        if _DRAFT_SPLIT_CONC:
            return 2 if (num_reqs <= 16 or window < 0) else 1
        return 1
    # Target attention (draft == False):
    if num_reqs <= 1:
        return 8 if window >= 0 else 16
    if 2 <= num_reqs <= 8:
        if window >= 0:
            return _C48_LOCAL_SPLITS_B2 if num_reqs == 2 else _C48_LOCAL_SPLITS
        if num_reqs in _C48_GLOBAL_SPLITS_MAP:
            return _C48_GLOBAL_SPLITS_MAP[num_reqs]
        base = max(1, mtiles * num_reqs)
        return max(2, min(16, _C48_GLOBAL_TARGET_CTAS // base))
    return 1
_TARGET_ATTN_LIB_PATH = os.environ.get(
    "INKLING_TARGET_ATTN_LIB", "/rdma/libinkling_target_attn_exact.so"
)
_MGS_LIB = None
_MGS_LIB_TRIED = False


def _get_mgs_lib():
    global _MGS_LIB, _MGS_LIB_TRIED
    if _MGS_LIB_TRIED:
        return _MGS_LIB
    _MGS_LIB_TRIED = True
    if not os.path.exists(_TARGET_ATTN_LIB_PATH):
        return None
    lib = ctypes.CDLL(_TARGET_ATTN_LIB_PATH)
    lib.merge_gate_sink_exact_fwd.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_int64, ctypes.c_int64,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_uint64,
    ]
    lib.merge_gate_sink_exact_fwd.restype = ctypes.c_int
    _MGS_LIB = lib
    return lib


def _rowar_nlive_buf(num_tokens: int):
    if not _ATTN_NLIVE:
        return None
    try:
        from vllm.distributed.device_communicators import rdma_all_reduce as rar
        if rar.ROWAR.get("scope") == num_tokens:
            return rar.ROWAR.get("buf")
    except Exception:
        pass
    return None


@triton.jit
def _attn_kernel(
    Q, K, V, BT, SEQ, CUQ, BIAS, O, L,
    sq_t, sq_h,
    sk_b, sk_s, sk_h,
    sv_b, sv_s, sv_h,
    sbt_r,
    sb_t, sb_h,
    so_s, so_t, so_h,
    sl_s, sl_t, sl_h,
    scale_log2,
    WINDOW: tl.constexpr,  # left window (keys >= qpos - WINDOW); <0 = none
    E: tl.constexpr, EW: tl.constexpr,
    G: tl.constexpr, BS: tl.constexpr, D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    DRAFT: tl.constexpr = False,  # every query at the request's last key, no bias
    REQ_BOUNDS: tl.constexpr = False,  # key range / split partition from the whole request, not this M tile
    PDL: tl.constexpr = False,  # launched with PDL behind the prologue; trigger the wo chain early
    NLivePtr=None,
    HAS_NLIVE: tl.constexpr = False,
    TreeDepthPtr=None,
    TreePathPtr=None,
    TreeMetaPtr=None,
    HAS_TREE: tl.constexpr = False,
    HAS_TREE_2B: tl.constexpr = False,
    TREE_FAST_EXIT: tl.constexpr = False,
    TREE_MAX_M: tl.constexpr = 8,
    GATE=None,
    SINK=None,
    Y=None,
    sg_t=0,
    ssk_t=0,
    sy_t=0,
    FUSE_GS: tl.constexpr = False,
    max_bt_cols=0,
):
    if PDL:
        if DRAFT:
            # Strict PDL order for drafter chain: wait for qk_norm_rope[_write_kv]
            # before reading Q/K/V, then trigger _merge_kernel / o_proj early.
            tl.extra.cuda.gdc_wait()
            tl.extra.cuda.gdc_launch_dependents()
        elif FUSE_GS:
            # NUM_SPLITS == 1 fused Gate+Sink writes Y directly; downstream C=32 wo_ud
            # does not gdc_wait(), so NEVER launch dependents before Y is written!
            tl.extra.cuda.gdc_wait()
        else:
            # Trigger BEFORE the wait: the dependent (merge+gate_sink, then wo_ud behind it)
            # becomes resident while we run.  Legal because every dependent gdc_waits for
            # our full completion before reading anything we write; we wait for the
            # prologue (q / bias / new K,V) before reading anything it writes.
            tl.extra.cuda.gdc_launch_dependents()
            if HAS_TREE_2B and TREE_FAST_EXIT:
                if tl.program_id(0) == 3:
                    if (tl.load(TreeMetaPtr + 0) == 0) | (tl.load(TreeMetaPtr + 2) <= 0):
                        return
            tl.extra.cuda.gdc_wait()
    elif HAS_TREE_2B and TREE_FAST_EXIT and not DRAFT:
        if tl.program_id(0) == 3:
            if (tl.load(TreeMetaPtr + 0) == 0) | (tl.load(TreeMetaPtr + 2) <= 0):
                return
    mt = tl.program_id(0)
    r = tl.program_id(1)
    kvh = tl.program_id(2) // NUM_SPLITS
    split = tl.program_id(2) % NUM_SPLITS

    q0 = tl.load(CUQ + r)
    qlen = tl.maximum(tl.load(CUQ + r + 1) - q0, 0)
    seqlen = tl.load(SEQ + r)
    past = seqlen - qlen
    if HAS_NLIVE:
        nl = tl.load(NLivePtr)
        r_live = tl.maximum(0, tl.minimum(qlen, nl - q0))
    else:
        r_live = qlen

    if HAS_TREE_2B:
        # Fast 4-tile BM=16 two-branch tree mode (ws <= 2):
        #   mt = 0..2 -> primary chain 0..tree_wp (100% identical to linear BM=16!)
        #   mt = 3    -> secondary branch tree_wp+1..tree_wp+tree_ws (<= 12 <= 16 rows!)
        t_act = tl.load(TreeMetaPtr + 0)
        t_wp_raw = tl.load(TreeMetaPtr + 1)
        t_ws_raw = tl.load(TreeMetaPtr + 2)
        use_br = (t_act != 0) & (t_ws_raw > 0)
        tree_wp = tl.where(use_br, tl.minimum(tl.maximum(t_wp_raw, 0), qlen - 1), r_live - 1)
        tree_ws = tl.where(use_br, tl.minimum(t_ws_raw, tl.minimum(2, qlen - 1 - tree_wp)), 0)
        is_sec_tile = mt == 3
        if is_sec_tile:
            if tree_ws <= 0:
                return
        else:
            if mt * BLOCK_M // G > tree_wp:
                return
        lane_m = tl.arange(0, BLOCK_M)
        p_m = mt * BLOCK_M + lane_m
        p_iloc = p_m // G
        p_g = p_m % G
        p_ok = p_iloc <= tree_wp
        s_off = lane_m // G
        s_g = lane_m % G
        s_ok = s_off < tree_ws
        s_iloc = tl.where(s_ok, 1 + tree_wp + s_off, 1 + tree_wp)
        i_loc = tl.where(is_sec_tile, s_iloc, p_iloc)
        g = tl.where(is_sec_tile, s_g, p_g)
        row_ok = tl.where(is_sec_tile, s_ok, p_ok)
    elif HAS_TREE:
        # General canonical-lane 0-ULP tree mode: 1 tree node (G=6 heads) per CTA
        m = mt * G + tl.arange(0, BLOCK_M)
        i_loc = m // G
        g = m % G
        row_ok = (tl.arange(0, BLOCK_M) < G) & (mt < r_live)
        if mt >= r_live:
            return
    else:
        m = mt * BLOCK_M + tl.arange(0, BLOCK_M)
        i_loc = m // G
        g = m % G
        row_ok = i_loc < r_live
        if mt * BLOCK_M // G >= r_live:
            return
    h = kvh * G + g
    tok = q0 + i_loc
    if DRAFT:
        qpos = tl.zeros([BLOCK_M], tl.int32) + (seqlen - 1)
    elif HAS_TREE_2B:
        qpos = tl.where(is_sec_tile, past + 1 + tl.where(s_ok, s_off, tree_ws - 1), past + p_iloc)
    elif HAS_TREE:
        node_depth = tl.load(TreeDepthPtr + mt)
        qpos = tl.zeros([BLOCK_M], tl.int32) + (past + node_depth)
    else:
        qpos = past + i_loc
    # sheared-bias column offset per token (see _unshear_rel_logits)
    tile_end = tl.minimum((i_loc // 128 + 1) * 128, qlen)
    off = ((past + tile_end + 127) // 128 - 10) * 128

    dd = tl.arange(0, D)
    q = tl.load(Q + tok[:, None] * sq_t + h[:, None] * sq_h + dd[None, :],
                mask=row_ok[:, None], other=0.0)

    # key range for this M tile
    if DRAFT:
        k_end = seqlen
        if WINDOW >= 0:
            k_beg = tl.maximum(seqlen - 1 - WINDOW, 0)
        else:
            k_beg = 0
    elif REQ_BOUNDS:
        # Same bounds as the single BLOCK_M=64 tile (mt=0) that covers every
        # query row: identical split partition and block order for each row.
        # The per-row `valid` mask below enforces the true causal/window limits.
        k_end = past + qlen
        if WINDOW >= 0:
            k_beg = tl.maximum(past - WINDOW, 0)
        else:
            k_beg = 0
    else:
        i_hi = tl.minimum((mt * BLOCK_M + BLOCK_M - 1) // G, qlen - 1)
        k_end = past + i_hi + 1
        if WINDOW >= 0:
            k_beg = tl.maximum(past + mt * BLOCK_M // G - WINDOW, 0)
        else:
            k_beg = 0
    max_keys = tl.where(max_bt_cols > 0, max_bt_cols * BS, k_end)
    k_end = tl.maximum(tl.minimum(k_end, max_keys), 0)
    k_beg = tl.minimum(tl.maximum(k_beg, 0), k_end)
    span = k_end - k_beg
    per = tl.cdiv(tl.cdiv(span, NUM_SPLITS), BLOCK_N) * BLOCK_N
    s_beg = k_beg + split * per
    s_end = tl.minimum(s_beg + per, k_end)
    if HAS_TREE_2B:
        s_end_live = tl.maximum(tl.minimum(s_end, tl.where(is_sec_tile, past + 1 + tree_ws, past + 1 + tree_wp)), s_beg)
    elif HAS_NLIVE and not DRAFT:
        s_end_live = tl.maximum(tl.minimum(s_end, past + r_live), s_beg)
    else:
        s_end_live = tl.maximum(s_end, s_beg)

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)
    nn = tl.arange(0, BLOCK_N)
    for n0 in range(s_beg, s_end_live, BLOCK_N):
        kpos = n0 + nn
        k_ok = (kpos >= 0) & (kpos < s_end_live)
        if HAS_TREE_2B:
            phys_kpos = tl.where(is_sec_tile & (kpos > past), kpos + tree_wp, kpos)
            phys_blk = phys_kpos // BS
            k_ok = k_ok & (phys_kpos >= 0) & ((max_bt_cols <= 0) | (phys_blk < max_bt_cols))
            blk = tl.load(BT + r * sbt_r + phys_blk, mask=k_ok, other=0).to(tl.int64)
            slot = phys_kpos % BS
        elif HAS_TREE:
            is_tree_k = kpos >= past
            d_tree = tl.maximum(kpos - past, 0)
            in_anc = is_tree_k & (d_tree <= node_depth)
            anc_idx = tl.load(
                TreePathPtr + mt * TREE_MAX_M + tl.minimum(d_tree, TREE_MAX_M - 1),
                mask=in_anc,
                other=0,
            )
            phys_kpos = tl.where(is_tree_k, past + anc_idx, kpos)
            phys_blk = phys_kpos // BS
            k_ok = k_ok & ((~is_tree_k) | in_anc) & (phys_kpos >= 0) & ((max_bt_cols <= 0) | (phys_blk < max_bt_cols))
            blk = tl.load(BT + r * sbt_r + phys_blk, mask=k_ok, other=0).to(tl.int64)
            slot = phys_kpos % BS
        else:
            lblk = kpos // BS
            k_ok = k_ok & ((max_bt_cols <= 0) | (lblk < max_bt_cols))
            blk = tl.load(BT + r * sbt_r + lblk, mask=k_ok, other=0).to(tl.int64)
            slot = kpos % BS
        k = tl.load(K + blk[None, :] * sk_b + slot[None, :] * sk_s + kvh * sk_h + dd[:, None],
                    mask=k_ok[None, :], other=0.0)
        s = tl.dot(q, k)  # [M, N] fp32
        d = qpos[:, None] - kpos[None, :]
        valid = row_ok[:, None] & k_ok[None, :] & (d >= 0)
        if WINDOW >= 0:
            valid = valid & (d <= WINDOW)
        if E > 0:
            col = kpos[None, :] - off[:, None]
            b_ok = valid & (d < E) & (col >= 0) & (col < EW)
            bias = tl.load(BIAS + tok[:, None] * sb_t + h[:, None] * sb_h + col,
                           mask=b_ok, other=0.0).to(tl.float32)
            bias = tl.where(bias == bias, bias, 0.0)  # NaN padding -> 0
            s = s * scale_log2 + bias * 1.4426950408889634
        else:
            s = s * scale_log2
        s = tl.where(valid, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_safe)
        p = tl.exp2(s - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(V + blk[:, None] * sv_b + slot[:, None] * sv_s + kvh * sv_h + dd[None, :],
                    mask=k_ok[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    l_safe = tl.where(l_i > 0, l_i, 1.0)
    out = acc / l_safe[:, None]
    if FUSE_GS:
        lse = tl.where(l_i > 0, (m_i + tl.log2(l_safe)) * 0.6931471805599453, float("-inf"))
        col_g = h[:, None] * D + dd[None, :]
        g_val = tl.load(GATE + tok[:, None] * sg_t + col_g, mask=row_ok[:, None], other=0.0).to(tl.float32)
        sk_val = tl.load(SINK + tok * ssk_t + h, mask=row_ok, other=0.0).to(tl.float32)
        res = out * _sigmoid(g_val)
        res = res * _sigmoid(lse - sk_val)[:, None]
        tl.store(Y + tok[:, None] * sy_t + col_g, res, mask=row_ok[:, None])
        return
    tl.store(O + split * so_s + tok[:, None] * so_t + h[:, None] * so_h + dd[None, :],
             out, mask=row_ok[:, None])
    if not (DRAFT and NUM_SPLITS == 1):
        lse = tl.where(l_i > 0, (m_i + tl.log2(l_safe)) * 0.6931471805599453, float("-inf"))
        tl.store(L + split * sl_s + tok * sl_t + h * sl_h, lse, mask=row_ok)


@triton.jit
def _merge_kernel(OP, LP, O, L, so_s, so_t, so_h, sl_s, sl_t, sl_h,
                  O_t, O_h, L_h, L_t,
                  H: tl.constexpr, HP: tl.constexpr, D: tl.constexpr, S: tl.constexpr,
                  SP: tl.constexpr,
                  SKIP_LSE: tl.constexpr = False, PDL: tl.constexpr = False):
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    t = tl.program_id(0)
    hh = tl.arange(0, HP)
    hm = hh < H
    ss = tl.arange(0, SP)
    sm = ss < S
    dd = tl.arange(0, D)
    lse = tl.load(LP + ss[:, None] * sl_s + t * sl_t + hh[None, :] * sl_h, mask=sm[:, None] & hm[None, :], other=float("-inf"))  # [SP, HP]
    mx = tl.max(lse, 0)
    mx_safe = tl.where(mx == float("-inf"), 0.0, mx)
    w = tl.exp(lse - mx_safe[None, :])
    tot = tl.sum(w, 0)
    tot_safe = tl.where(tot > 0, tot, 1.0)
    acc = tl.zeros([HP, D], tl.float32)
    for s in tl.static_range(S):
        ws = tl.sum(tl.where(ss[:, None] == s, w, 0.0), 0)  # [H]
        o = tl.load(OP + s * so_s + t * so_t + hh[:, None] * so_h + dd[None, :], mask=hm[:, None], other=0.0)
        acc += o * ws[:, None]
    tl.store(O + t * O_t + hh[:, None] * O_h + dd[None, :], acc / tot_safe[:, None], mask=hm[:, None])
    if not SKIP_LSE:
        tl.store(L + hh * L_h + t * L_t,
                 tl.where(tot > 0, mx_safe + tl.log(tot_safe), float("-inf")), mask=hm)


@triton.jit
def _sigmoid(x):
    exp = tl.inline_asm_elementwise(
        "ex2.approx.f32 $0, $1;", constraints="=f,f", args=[-x * 1.4426950408889634],
        dtype=tl.float32, is_pure=True, pack=1)
    return tl.inline_asm_elementwise(
        "rcp.approx.f32 $0, $1;", constraints="=f,f", args=[exp + 1.0],
        dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _merge_gate_sink_kernel(OP, LP, GATE, SINK, Y, so_s, so_t, so_h, sl_s, sl_t, sl_h,
                            sg_t, ssk_t, NT,
                            H: tl.constexpr, HP: tl.constexpr, D: tl.constexpr, S: tl.constexpr,
                            SP: tl.constexpr,
                            TRIG: tl.constexpr = False,
                            NLivePtr=None, HAS_NLIVE: tl.constexpr = False):
    """_merge_kernel followed by attention_ops._gate_sink_kernel, one program per
    token.  Tokens >= NT reproduce the zero-filled out/lse tail of the unfused path."""
    if TRIG:
        # before the wait: wo_ud's CTAs go resident and L2-prefetch W while the attention
        # kernel (which we gdc_wait for) is still running; wo_ud gdc_waits for our end.
        tl.extra.cuda.gdc_launch_dependents()
    tl.extra.cuda.gdc_wait()
    t = tl.program_id(0)
    if HAS_NLIVE:
        if t >= tl.load(NLivePtr):
            return
    tv = t < NT
    hh = tl.arange(0, HP)
    hm = hh < H
    ss = tl.arange(0, SP)
    sm = ss < S
    dd = tl.arange(0, D)
    lse = tl.load(LP + ss[:, None] * sl_s + t * sl_t + hh[None, :] * sl_h, mask=sm[:, None] & hm[None, :] & tv, other=float("-inf"))
    mx = tl.max(lse, 0)
    mx_safe = tl.where(mx == float("-inf"), 0.0, mx)
    w = tl.exp(lse - mx_safe[None, :])
    tot = tl.sum(w, 0)
    tot_safe = tl.where(tot > 0, tot, 1.0)
    acc = tl.zeros([HP, D], tl.float32)
    for s in tl.static_range(S):
        ws = tl.sum(tl.where(ss[:, None] == s, w, 0.0), 0)
        o = tl.load(OP + s * so_s + t * so_t + hh[:, None] * so_h + dd[None, :], mask=hm[:, None] & tv, other=0.0)
        acc += o * ws[:, None]
    x = acc / tot_safe[:, None]
    lse_v = tl.where(tot > 0, mx_safe + tl.log(tot_safe), float("-inf"))
    x = tl.where(tv, x, 0.0)
    lse_v = tl.where(tv, lse_v, 0.0)
    col = hh[:, None] * D + dd[None, :]
    gate = tl.load(GATE + t * sg_t + col, mask=hm[:, None], other=0).to(tl.float32)
    sink = tl.load(SINK + t * ssk_t + hh, mask=hm, other=0).to(tl.float32)
    result = x * _sigmoid(gate)
    result = result * _sigmoid(lse_v - sink)[:, None]
    tl.store(Y + t * (H * D) + col, result, mask=hm[:, None])


@triton.jit
def _attn_prefill_local_kernel(
    Q, K, V, BT, SEQ, CUQ, BIAS, O, L, GATE, SINK, Y,
    sq_t, sq_h,
    sk_b, sk_s, sk_h,
    sv_b, sv_s, sv_h,
    sbt_r,
    sb_t, sb_h,
    so_t, so_h,
    sl_h, sl_t,
    sg_t, ssk_t,
    scale_log2,
    WINDOW: tl.constexpr,
    E: tl.constexpr, EW: tl.constexpr,
    G: tl.constexpr, BS: tl.constexpr, D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    ALIGN_KBEG_M64: tl.constexpr = True,
    FUSE_GS: tl.constexpr = True,
    max_bt_cols=0,
):
    """0-ULP prefill sliding-window attention (BLOCK_M=32, BLOCK_N=64, 4 warps).
    Eliminates the 218-register spill of the stock (64,64,4w) kernel while keeping
    k_beg aligned to the enclosing BLOCK_M=64 tile (literal 0-ULP match!)."""
    mt = tl.program_id(0)
    r = tl.program_id(1)
    kvh = tl.program_id(2)

    q0 = tl.load(CUQ + r)
    qlen = tl.maximum(tl.load(CUQ + r + 1) - q0, 0)
    seqlen = tl.load(SEQ + r)
    past = seqlen - qlen

    i_lo = (mt * BLOCK_M) // G
    if i_lo >= qlen:
        return
    i_hi = tl.minimum((mt * BLOCK_M + BLOCK_M - 1) // G, qlen - 1)

    m = mt * BLOCK_M + tl.arange(0, BLOCK_M)
    i_loc = m // G
    g = m % G
    row_ok = i_loc < qlen
    h = kvh * G + g
    tok = q0 + i_loc
    dd = tl.arange(0, D)
    q = tl.load(Q + tok[:, None] * sq_t + h[:, None] * sq_h + dd[None, :],
                mask=row_ok[:, None], other=0.0)
    tile_end = tl.minimum((i_loc // 128 + 1) * 128, qlen)
    off = ((past + tile_end + 127) // 128 - 10) * 128
    b_base = tok * sb_t + h * sb_h
    qpos = past + i_loc

    k_end = past + i_hi + 1
    max_keys = tl.where(max_bt_cols > 0, max_bt_cols * BS, k_end)
    k_end = tl.maximum(tl.minimum(k_end, max_keys), 0)
    if ALIGN_KBEG_M64:
        mt64 = (mt * BLOCK_M) // 64
        k_beg = tl.maximum(past + (mt64 * 64) // G - WINDOW, 0)
    else:
        k_beg = tl.maximum(past + i_lo - WINDOW, 0)
    k_beg = tl.minimum(k_beg, k_end)

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)
    nn = tl.arange(0, BLOCK_N)
    bt_row = BT + r * sbt_r

    for n0 in range(k_beg, k_end, BLOCK_N):
        kpos = n0 + nn
        k_ok = kpos < k_end
        blk = tl.load(bt_row + kpos // BS, mask=k_ok, other=0).to(tl.int64)
        slot = kpos % BS
        k = tl.load(K + blk[None, :] * sk_b + slot[None, :] * sk_s + kvh * sk_h + dd[:, None],
                    mask=k_ok[None, :], other=0.0)
        s = tl.dot(q, k)
        d = qpos[:, None] - kpos[None, :]
        valid = row_ok[:, None] & k_ok[None, :] & (d >= 0) & (d <= WINDOW)
        if E > 0:
            col = kpos[None, :] - off[:, None]
            if WINDOW == E - 1:
                b_ok = valid
            else:
                b_ok = valid & (d < E) & (col >= 0) & (col < EW)
            bias = tl.load(BIAS + b_base[:, None] + col,
                           mask=b_ok, other=0.0).to(tl.float32)
            bias = tl.where(bias == bias, bias, 0.0)
            s = s * scale_log2 + bias * 1.4426950408889634
        else:
            s = s * scale_log2
        s = tl.where(valid, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_safe)
        p = tl.exp2(s - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(V + blk[:, None] * sv_b + slot[:, None] * sv_s + kvh * sv_h + dd[None, :],
                    mask=k_ok[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    l_safe = tl.where(l_i > 0, l_i, 1.0)
    out = acc / l_safe[:, None]
    lse = tl.where(l_i > 0, (m_i + tl.log2(l_safe)) * 0.6931471805599453, float("-inf"))
    # Recompute indices in epilogue to keep inner-loop register pressure minimal
    m2 = mt * BLOCK_M + tl.arange(0, BLOCK_M)
    i2 = m2 // G
    h2 = kvh * G + (m2 % G)
    t2 = q0 + i2
    ok2 = i2 < qlen
    if FUSE_GS:
        col_g = h2[:, None] * D + dd[None, :]
        gate = tl.load(GATE + t2[:, None] * sg_t + col_g, mask=ok2[:, None], other=0).to(tl.float32)
        sink = tl.load(SINK + t2 * ssk_t + h2, mask=ok2, other=0).to(tl.float32)
        res = out * _sigmoid(gate)
        res = res * _sigmoid(lse - sink)[:, None]
        tl.store(Y + t2[:, None] * (G * D) + col_g, res, mask=ok2[:, None])
    else:
        tl.store(O + t2[:, None] * so_t + h2[:, None] * so_h + dd[None, :], out, mask=ok2[:, None])
        tl.store(L + h2 * sl_h + t2 * sl_t, lse, mask=ok2)


@triton.jit
def _attn_prefill_full_kernel(
    Q, K, V, BT, SEQ, CUQ, BIAS, O, L, GATE, SINK, Y,
    sq_t, sq_h,
    sk_b, sk_s, sk_h,
    sv_b, sv_s, sv_h,
    sbt_r,
    sb_t, sb_h,
    so_t, so_h,
    sl_h, sl_t,
    sg_t, ssk_t,
    scale_log2,
    E: tl.constexpr, EW: tl.constexpr,
    G: tl.constexpr, BS: tl.constexpr, D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    FUSE_GS: tl.constexpr = True,
    max_bt_cols=0,
):
    """0-ULP prefill full-attention kernel (unpacked 1 head/CTA, BM=64 or 128, BN=64).
    Since k_beg == 0 always in full attention, n0 = 0, 64, 128, ... is identical to
    stock _attn_kernel (literal 0-ULP match), while unlocking contiguous Q/BIAS/K/V,
    scalar off, scalar page-table load, and far-key bias skipping (d >= E)."""
    mt = tl.program_id(0)
    r = tl.program_id(1)
    h = tl.program_id(2)
    kvh = h // G

    q0 = tl.load(CUQ + r)
    qlen = tl.maximum(tl.load(CUQ + r + 1) - q0, 0)
    seqlen = tl.load(SEQ + r)
    past = seqlen - qlen

    i_lo = mt * BLOCK_M
    if i_lo >= qlen:
        return
    i_hi = tl.minimum(i_lo + BLOCK_M - 1, qlen - 1)

    i_loc = i_lo + tl.arange(0, BLOCK_M)
    row_ok = i_loc < qlen
    tok = q0 + i_loc
    qpos = past + i_loc
    tile_end = tl.minimum((i_lo // 128 + 1) * 128, qlen)
    off = ((past + tile_end + 127) // 128 - 10) * 128

    dd = tl.arange(0, D)
    q = tl.load(Q + tok[:, None] * sq_t + h * sq_h + dd[None, :],
                mask=row_ok[:, None], other=0.0)

    k_end = past + i_hi + 1
    max_keys = tl.where(max_bt_cols > 0, max_bt_cols * BS, k_end)
    k_end = tl.maximum(tl.minimum(k_end, max_keys), 0)
    q_min = past + i_lo
    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)
    nn = tl.arange(0, BLOCK_N)
    bt_row = BT + r * sbt_r
    b_base = tok * sb_t + h * sb_h

    for n0 in range(0, k_end, BLOCK_N):
        kpos = n0 + nn
        k_ok = kpos < k_end
        if (BS >= BLOCK_N) and (BS % BLOCK_N == 0):
            blk_s = tl.load(bt_row + n0 // BS).to(tl.int64)
            slot = (n0 % BS) + nn
            k = tl.load(K + blk_s * sk_b + slot[None, :] * sk_s + kvh * sk_h + dd[:, None],
                        mask=k_ok[None, :], other=0.0)
        else:
            blk = tl.load(bt_row + kpos // BS, mask=k_ok, other=0).to(tl.int64)
            slot = kpos % BS
            k = tl.load(K + blk[None, :] * sk_b + slot[None, :] * sk_s + kvh * sk_h + dd[:, None],
                        mask=k_ok[None, :], other=0.0)
        s = tl.dot(q, k)
        if E > 0 and (n0 + BLOCK_N > q_min - E + 1):
            d = qpos[:, None] - kpos[None, :]
            valid = row_ok[:, None] & k_ok[None, :] & (d >= 0)
            col = kpos[None, :] - off
            b_ok = valid & (d < E) & (col >= 0) & (col < EW)
            bias = tl.load(BIAS + b_base[:, None] + col,
                           mask=b_ok, other=0.0).to(tl.float32)
            bias = tl.where(bias == bias, bias, 0.0)
            s = s * scale_log2 + bias * 1.4426950408889634
            s = tl.where(valid, s, float("-inf"))
        else:
            s = s * scale_log2
            s = tl.where(row_ok[:, None], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_safe)
        p = tl.exp2(s - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        if (BS >= BLOCK_N) and (BS % BLOCK_N == 0):
            v = tl.load(V + blk_s * sv_b + slot[:, None] * sv_s + kvh * sv_h + dd[None, :],
                        mask=k_ok[:, None], other=0.0)
        else:
            v = tl.load(V + blk[:, None] * sv_b + slot[:, None] * sv_s + kvh * sv_h + dd[None, :],
                        mask=k_ok[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    l_safe = tl.where(l_i > 0, l_i, 1.0)
    out = acc / l_safe[:, None]
    lse = tl.where(l_i > 0, (m_i + tl.log2(l_safe)) * 0.6931471805599453, float("-inf"))
    if FUSE_GS:
        col_g = h * D + dd[None, :]
        gate = tl.load(GATE + tok[:, None] * sg_t + col_g, mask=row_ok[:, None], other=0).to(tl.float32)
        sink = tl.load(SINK + tok * ssk_t + h, mask=row_ok, other=0).to(tl.float32)
        res = out * _sigmoid(gate)
        res = res * _sigmoid(lse - sink)[:, None]
        tl.store(Y + tok[:, None] * (G * D) + col_g, res, mask=row_ok[:, None])
    else:
        tl.store(O + tok[:, None] * so_t + h * so_h + dd[None, :], out, mask=row_ok[:, None])
        tl.store(L + h * sl_h + tok * sl_t, lse, mask=row_ok)


# ---- Phase 2 (1M Context <= 120s) Global Prefill Hook & FP8 Shadow K Lifecycle ----
# Strictly opt-in (`INKLING_PF_GLOBAL_MODE` != "off"), gated ONLY to global layers
# (`window < 0`) on long prefill chunks (`max_seqlen_q >= MIN_QLEN` [default 1024],
# `past >= MIN_PAST` [default 4096], `!capturing`). Decode (C=1..C32), exact_ink.py
# (`T < 1024`), and local SWA layers NEVER enter this hook (100% bit-exact!).
_PF_GLOBAL_MODE = os.environ.get("INKLING_PF_GLOBAL_MODE", "off").strip().lower()
_PF_GLOBAL_ENABLED = _PF_GLOBAL_MODE not in ("", "0", "off", "false", "no")
_PF_GLOBAL_MIN_QLEN = int(os.environ.get("INKLING_PF_GLOBAL_MIN_QLEN", "1024") or 1024)
_PF_GLOBAL_MIN_PAST = int(os.environ.get("INKLING_PF_GLOBAL_MIN_PAST", "4096") or 4096)
# Warm long-context prefill gate (default 0 = disabled; e.g. 64 lets warm 1M Turn-2 queries with
# qlen=436 < 1024 use Sage2+TopK instead of 42-CTA dense BF16 over 1M tokens -> -1.42s Warm TTFT!
# Strictly >= 64 and !capturing so decode M <= 36 is 100% 0-ULP untouched).
_PF_GLOBAL_WARM_MIN_QLEN = int(os.environ.get("INKLING_PF_GLOBAL_WARM_MIN_QLEN", "0") or 0)
_PF_GLOBAL_WARM_MIN_PAST = int(os.environ.get("INKLING_PF_GLOBAL_WARM_MIN_PAST", "32768") or 32768)
_PF_GLOBAL_TAU = float(
    os.environ.get("INKLING_PF_GLOBAL_TAU", os.environ.get("INKLING_PF_PVSKIP_TAU", "24.0")) or 24.0
)
_PF_GLOBAL_TOPK = int(os.environ.get("INKLING_PF_GLOBAL_TOPK", "0") or 0)  # 0 = 100% DENSE default!
_PF_GLOBAL_BN = int(os.environ.get("INKLING_PF_GLOBAL_BN", "64") or 64)  # 64 -> fastest dense cuda_v3_bn64_2c!
_PF_GLOBAL_MIN_CTAS = int(os.environ.get("INKLING_PF_GLOBAL_MIN_CTAS", "2") or 2)
_PF_GLOBAL_LIB_ENV = os.environ.get("INKLING_PF_GLOBAL_LIB", "").strip()
_PF_SHADOW_K_MAX_GB = float(os.environ.get("INKLING_PF_SHADOW_K_MAX_GB", "10.0") or 10.0)
# Compact logical-block FP8 Shadow K (`INKLING_PF_SHADOW_K_COMPACT=1`, default OFF):
# Sizes Shadow K to `CAP_BLKS` (default 4096 = 1M tokens -> 1.016 GiB constant across 8 global layers!)
# instead of full KV pool `nblk` (7.25-9.57 GiB!), unlocking 5x-8x simultaneous 1M sessions without OOM!
_PF_SHADOW_K_COMPACT = os.environ.get("INKLING_PF_SHADOW_K_COMPACT", "0").strip().lower() in (
    "1", "on", "true", "yes",
)
_PF_SHADOW_K_CAP_BLKS = max(int(os.environ.get("INKLING_PF_SHADOW_K_CAP_BLKS", "4096") or 4096), 64)
_PF_GLOBAL_ZERO_SYNC = os.environ.get("INKLING_PF_GLOBAL_ZERO_SYNC", "0").strip().lower() in (
    "1", "on", "true", "yes",
)
_PF_SHADOW_K_CACHE: dict[int, dict] = {}


def release_shadow_k_cache() -> int:
    """Opt-in release of prefill FP8 shadow-K cache (1.02 GiB across 8 global layers; capture-guarded)."""
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return 0
    freed = 0
    for st in _PF_SHADOW_K_CACHE.values():
        if isinstance(st, dict):
            for k in ("k_fp8", "k_scale", "k_mid", "k_rad", "gpu_bt_state"):
                t_buf = st.get(k)
                if isinstance(t_buf, torch.Tensor):
                    freed += int(t_buf.numel() * t_buf.element_size())
    _PF_SHADOW_K_CACHE.clear()
    return freed
_PF_SHADOW_K_WARNED = False
_PF_COMPACT_ARANGE_BT: dict[tuple, torch.Tensor] = {}
_PREFILL_CPU_LENS: tuple = (None, None)
_PENDING_PREFILL_CPU_LENS: tuple = (None, None)


def _get_compact_arange_bt(device: torch.device, cap_blks: int) -> torch.Tensor:
    key = (device, cap_blks)
    t = _PF_COMPACT_ARANGE_BT.get(key)
    if t is None:
        t = torch.arange(cap_blks, dtype=torch.int32, device=device).unsqueeze(0).contiguous()
        _PF_COMPACT_ARANGE_BT[key] = t
    return t


def set_prefill_cpu_lens(seq_lens_cpu=None, query_lens_cpu=None) -> None:
    """Pass CPU-side `(seq_lens_cpu, query_lens_cpu)` into the upcoming `rel_attention`
    call so `_launch_prefill_opt` & `_prepare_fp8_shadow_k` never issue a DtoH sync."""
    global _PREFILL_CPU_LENS
    _PREFILL_CPU_LENS = (seq_lens_cpu, query_lens_cpu)


def _resolve_pf_global_lib() -> str | None:
    """Resolve Sage2 CUDA library preferring `INKLING_PF_GLOBAL_LIB` then v5 -> v4 -> v3 -> v2/v1."""
    for p in (
        _PF_GLOBAL_LIB_ENV,
        "/rdma/libglobal_attn_sage2_sm121_v5.so",
        "/rdma/libglobal_attn_sage2_sm121_v4.so",
        "/rdma/libglobal_attn_sage2_sm121_v3.so",
        "/rdma/libglobal_attn_sage2_sm121.so",
    ):
        if p and os.path.exists(p):
            return p
    return None


def _prepare_fp8_shadow_k_ex(
    key_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    num_reqs: int,
    bs: int,
    gmod,
    *,
    is_v3: bool = True,
    with_kmax: bool = False,
    seq_lens_cpu=None,
    query_lens_cpu=None,
    kv_bf12_cache=None,
    kv_bf12_emerg=None,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    int,
]:
    """Lazily allocate & maintain FP8 shadow K cache ONLY for the 8 global layers.

    Returns ``(k_fp8, k_scale, k_mid, k_rad, k_bt, k_bt_mode)``.
    When ``INKLING_PF_SHADOW_K_COMPACT=1`` and ``libglobal_attn_sage2_sm121_v5.so`` is loaded,
    allocates only ``min(nblk, CAP_BLKS)`` logical slots (1.016 GiB constant across 8 global layers
    instead of 7.25-9.57 GiB proportional to total KV pool `nblk`), saving 6.2-8.6 GiB of PyTorch
    VRAM under `set_per_process_memory_fraction` and unlocking 5x-8x simultaneous 1M KV sessions!
    """
    global _PF_SHADOW_K_WARNED
    use_kv_bf12 = kv_bf12_cache is not None
    ptr = int(kv_bf12_cache.data_ptr()) if use_kv_bf12 else int(key_cache.data_ptr())
    if use_kv_bf12:
        kc_shape = (int(kv_bf12_cache.shape[0]), 256, int(kv_bf12_cache.shape[1]), 128)
    else:
        kc_shape = tuple(key_cache.shape)
    st = _PF_SHADOW_K_CACHE.get(ptr)
    if st is None or st.get("shape") != kc_shape:
        nblk, bs_k, hkv, d = kc_shape
        use_compact = False
        cap_blks = nblk
        if _PF_SHADOW_K_COMPACT and is_v3:
            _lib = gmod.get_cuda_lib(_resolve_pf_global_lib())
            if _lib is not None and hasattr(_lib, "global_sage2_prefix_v5_sm121_fwd"):
                use_compact = True
                cap_blks = min(nblk, _PF_SHADOW_K_CAP_BLKS)
            elif not _PF_SHADOW_K_WARNED:
                _PF_SHADOW_K_WARNED = True
                print(
                    "[inkling-pf-global] INKLING_PF_SHADOW_K_COMPACT=1 requested but "
                    "libglobal_attn_sage2_sm121_v5.so (global_sage2_prefix_v5_sm121_fwd) "
                    "not found; using standard full-pool shadow K.",
                    flush=True,
                )
        alloc_blks = cap_blks if use_compact else nblk
        if is_v3:
            bytes_per_blk = bs_k * hkv * d + (bs_k // 64) * hkv * 4 + (2 * hkv * d * 2 if with_kmax else 0)
        else:
            bytes_per_blk = bs_k * hkv * (d + 4)
        per_layer_gb = (alloc_blks * bytes_per_blk) / (1024.0**3)
        est_8l_gb = 8.0 * per_layer_gb
        if est_8l_gb > _PF_SHADOW_K_MAX_GB:
            if not _PF_SHADOW_K_WARNED:
                _PF_SHADOW_K_WARNED = True
                print(
                    f"[inkling-pf-global] shadow K ({'v3' if is_v3 else 'v2'}, kmax={int(with_kmax)}, "
                    f"compact={int(use_compact)}) for 8 global layers needs {est_8l_gb:.2f} GiB "
                    f"(alloc_blks={alloc_blks}) > INKLING_PF_SHADOW_K_MAX_GB={_PF_SHADOW_K_MAX_GB:.1f} GiB; "
                    f"falling back to fp8_sage2_cuda v1 (online K quant, 0 shadow VRAM).",
                    flush=True,
                )
            _PF_SHADOW_K_CACHE[ptr] = {"disabled": True, "shape": kc_shape}
            return None, None, None, None, None, 0
        if is_v3:
            _alloc_ref = (
                torch.empty((nblk, bs_k, hkv, d), dtype=torch.bfloat16, device="meta")
                if use_kv_bf12
                else key_cache
            )
            k_fp8, k_scale, k_mid, k_rad = gmod.alloc_fp8_shadow_k_v3(
                _alloc_ref,
                with_kmax=with_kmax,
                cap_blks=cap_blks if use_compact else None,
                device=kv_bf12_cache.device if use_kv_bf12 else None,
            )
            if use_compact and len(_PF_SHADOW_K_CACHE) == 0:
                print(
                    f"[inkling-pf-global] COMPACT Shadow K ACTIVE: alloc_blks={alloc_blks}/{nblk} "
                    f"(est 8L={est_8l_gb:.3f} GiB vs full {(8.0 * nblk * bytes_per_blk)/(1024.0**3):.2f} GiB, "
                    f"saved {(8.0 * (nblk - alloc_blks) * bytes_per_blk)/(1024.0**3):.2f} GiB!)",
                    flush=True,
                )
        else:
            _alloc_ref = (
                torch.empty((nblk, bs_k, hkv, d), dtype=torch.bfloat16, device="meta")
                if use_kv_bf12
                else key_cache
            )
            k_fp8, k_scale = gmod.alloc_fp8_shadow_k(
                _alloc_ref, device=kv_bf12_cache.device if use_kv_bf12 else None
            )
            k_mid = k_rad = None
        st = {
            "disabled": False,
            "shape": kc_shape,
            "is_v3": is_v3,
            "with_kmax": with_kmax,
            "compact": use_compact,
            "cap_blks": alloc_blks,
            "k_fp8": k_fp8,
            "k_scale": k_scale,
            "k_mid": k_mid,
            "k_rad": k_rad,
            "last_bt0": -1,
            "last_bt_end": -1,
            "last_seqlen": -1,
            "last_nblks": 0,
            "gpu_bt_state": (
                torch.full((2,), -1, dtype=torch.int32, device=key_cache.device)
                if _PF_GLOBAL_ZERO_SYNC
                else None
            ),
        }
        _PF_SHADOW_K_CACHE[ptr] = st
    if st["disabled"]:
        return None, None, None, None, None, 0

    k_fp8, k_scale, k_mid, k_rad = st["k_fp8"], st["k_scale"], st["k_mid"], st["k_rad"]
    compact = bool(st.get("compact", False))
    cap_blks = int(st.get("cap_blks", kc_shape[0]))

    def _quant_blocks(blk_ids: torch.Tensor, out_offset: int = 0) -> None:
        if use_kv_bf12:
            from . import sm12_attn_fused as _sm12_f
            ids_flat = blk_ids.reshape(-1)
            total_b = int(ids_flat.numel())
            # Chunk unpack <= 512 blocks (32 MiB peak) even on Warm-1M full Shadow-K build!
            chunk_b = 512
            for off_i in range(0, total_b, chunk_b):
                sub_ids = ids_flat[off_i : off_i + chunk_b]
                n_sub = int(sub_ids.numel())
                k_stage, _ = _sm12_f.kv_bf12_unpack_blocks(
                    kv_bf12_cache, kv_bf12_emerg, sub_ids, unpack_k=True, unpack_v=False
                )
                local_ids = torch.arange(n_sub, dtype=torch.int32, device=sub_ids.device)
                if st["is_v3"]:
                    if compact:
                        gmod.quantize_k_to_fp8_blocks_v3(
                            k_stage, k_fp8, k_scale, k_mid, k_rad, local_ids,
                            with_kmax=st["with_kmax"], out_offset=out_offset + off_i, compact_out=True,
                        )
                    else:
                        t_fp8, t_sc, t_mid, t_rad = gmod.alloc_fp8_shadow_k_v3(
                            k_stage, with_kmax=st["with_kmax"], cap_blks=n_sub
                        )
                        gmod.quantize_k_to_fp8_blocks_v3(
                            k_stage, t_fp8, t_sc, t_mid, t_rad, local_ids,
                            with_kmax=st["with_kmax"], out_offset=0, compact_out=True,
                        )
                        dst_idx = sub_ids.to(dtype=torch.long)
                        k_fp8.index_copy_(0, dst_idx, t_fp8)
                        k_scale.index_copy_(0, dst_idx, t_sc)
                        if st["with_kmax"] and k_mid is not None and t_mid is not None:
                            k_mid.index_copy_(0, dst_idx, t_mid)
                            k_rad.index_copy_(0, dst_idx, t_rad)
                else:
                    t_fp8, t_sc = gmod.quantize_k_to_fp8_blocks(k_stage)
                    dst_idx = sub_ids.to(dtype=torch.long)
                    k_fp8.index_copy_(0, dst_idx, t_fp8)
                    k_scale.index_copy_(0, dst_idx, t_sc)
            return
        if st["is_v3"]:
            gmod.quantize_k_to_fp8_blocks_v3(
                key_cache, k_fp8, k_scale, k_mid, k_rad, blk_ids,
                with_kmax=st["with_kmax"], out_offset=out_offset, compact_out=compact,
            )
        else:
            gmod.quantize_k_to_fp8_blocks(key_cache, k_fp8, k_scale, blk_ids)

    if num_reqs == 1:
        if seq_lens_cpu is not None and len(seq_lens_cpu) >= 1:
            seqlen = int(seq_lens_cpu[0])
        else:
            seqlen = int(cache_seqlens[0].item())
        if query_lens_cpu is not None and len(query_lens_cpu) >= 1:
            qlen = int(query_lens_cpu[0])
        else:
            qlen = int((cu_seqlens_q[1] - cu_seqlens_q[0]).item())
        past = seqlen - qlen
        max_bt_cols = int(block_table.shape[1])
        nblks = min(max_bt_cols, (seqlen + bs - 1) // bs) if seqlen > 0 else 0
        if compact and nblks > cap_blks:
            st["last_bt0"] = -1
            st["last_seqlen"] = -1
            st["last_nblks"] = 0
            return None, None, None, None, None, 0
        k_bt_out = _get_compact_arange_bt(key_cache.device, cap_blks) if compact else None
        k_bt_mode = 1 if compact else 0
        if nblks <= 0:
            return k_fp8, k_scale, k_mid, k_rad, k_bt_out, k_bt_mode
        prev_nblks = st["last_nblks"]
        if _PF_GLOBAL_ZERO_SYNC and block_table.is_cuda and not use_kv_bf12:
            bt_state = st.get("gpu_bt_state")
            if bt_state is None:
                bt_state = torch.full((2,), -1, dtype=torch.int32, device=key_cache.device)
                st["gpu_bt_state"] = bt_state
            cpu_contig = (st["last_seqlen"] == past) and (0 < prev_nblks <= nblks)
            start_blk = (past // bs) if cpu_contig else 0
            prev_idx = (prev_nblks - 1) if cpu_contig else 0
            bt_row = block_table[0]
            if st["is_v3"]:
                gmod.quantize_k_to_fp8_blocks_v3_syncfree(
                    key_cache, k_fp8, k_scale, k_mid, k_rad, bt_row, bt_state,
                    nblks, start_blk, prev_idx, cpu_contig,
                    with_kmax=st["with_kmax"], compact_out=compact,
                )
            else:
                gmod.quantize_k_to_fp8_blocks_v2_syncfree(
                    key_cache, k_fp8, k_scale, bt_row, bt_state,
                    nblks, start_blk, prev_idx, cpu_contig,
                )
            st["last_seqlen"] = seqlen
            st["last_nblks"] = nblks
            return k_fp8, k_scale, k_mid, k_rad, k_bt_out, k_bt_mode
        if _PF_GLOBAL_ZERO_SYNC and block_table.is_cuda and use_kv_bf12 and st["is_v3"] and nblks <= int(block_table.shape[1]):
            from . import sm12_attn_fused as _sm12_f
            bt_state = st.get("gpu_bt_state")
            if bt_state is None:
                bt_state = torch.full((2,), -1, dtype=torch.int32, device=key_cache.device)
                st["gpu_bt_state"] = bt_state
            cpu_contig = (st["last_seqlen"] == past) and (0 < prev_nblks <= nblks)
            start_blk = (past // bs) if cpu_contig else 0
            prev_idx = (prev_nblks - 1) if cpu_contig else 0
            bt_row = block_table[0] if block_table[0].is_contiguous() else block_table[0].contiguous()
            bt_2d = block_table[:1]
            max_bt_cols = int(block_table.shape[1])
            hkv_bf12 = int(kv_bf12_cache.shape[1])
            chunk_b = min(512, nblks)
            k_stage_buf = torch.empty((chunk_b, bs, hkv_bf12, 128), dtype=torch.bfloat16, device=kv_bf12_cache.device)
            try:
                # Contiguous prefix chunks [0, start_blk): GPU-guarded 1-cycle no-op (0 D2H sync, 0 DRAM reads);
                # automatically unpacks & rebuilds Shadow-K on GPU if a different request reused `past`!
                if cpu_contig and start_blk > 0:
                    for off_i in range(0, start_blk, chunk_b):
                        n_sub = min(chunk_b, start_blk - off_i)
                        k_sub = k_stage_buf[:n_sub]
                        _sm12_f.kv_bf12_unpack_bt_range(
                            kv_bf12_cache, kv_bf12_emerg, bt_2d,
                            num_reqs=1, max_bt_cols=max_bt_cols,
                            b_beg=off_i, n_blks_per_req=n_sub,
                            guard_bt_state=bt_state, guard_prev_idx=prev_idx,
                            k_out=k_sub, unpack_k=True, unpack_v=False,
                        )
                        gmod.quantize_k_to_fp8_blocks_v3_local_syncfree(
                            k_sub, k_fp8, k_scale, k_mid, k_rad, bt_row, bt_state,
                            n_sub, off_i, prev_idx, check_contig=True,
                            with_kmax=st["with_kmax"], compact_out=compact,
                        )
                # Newly appended chunks [start_blk, nblks): unconditional unpack + FP8 quant (0 D2H sync!)
                for off_i in range(start_blk, nblks, chunk_b):
                    n_sub = min(chunk_b, nblks - off_i)
                    k_sub = k_stage_buf[:n_sub]
                    _sm12_f.kv_bf12_unpack_bt_range(
                        kv_bf12_cache, kv_bf12_emerg, bt_2d,
                        num_reqs=1, max_bt_cols=max_bt_cols,
                        b_beg=off_i, n_blks_per_req=n_sub,
                        k_out=k_sub, unpack_k=True, unpack_v=False,
                    )
                    gmod.quantize_k_to_fp8_blocks_v3_local_syncfree(
                        k_sub, k_fp8, k_scale, k_mid, k_rad, bt_row, bt_state,
                        n_sub, off_i, prev_idx, check_contig=False,
                        with_kmax=st["with_kmax"], compact_out=compact,
                    )
                gmod.update_bt_state_gpu(bt_row, bt_state, nblks - 1)
            finally:
                del k_stage_buf
            st["last_seqlen"] = seqlen
            st["last_nblks"] = nblks
            return k_fp8, k_scale, k_mid, k_rad, k_bt_out, k_bt_mode
        prev_idx = (prev_nblks - 1) if (0 < prev_nblks <= nblks) else 0
        # Batch 3 block_table scalars into a single tiny D2H sync (vs 5 separate .item() stalls!)
        bt_chk = torch.stack(
            (block_table[0, 0], block_table[0, prev_idx], block_table[0, nblks - 1])
        ).tolist()
        bt0, bt_prev, bt_end = int(bt_chk[0]), int(bt_chk[1]), int(bt_chk[2])
        is_contiguous = (
            st["last_bt0"] == bt0
            and st["last_seqlen"] == past
            and prev_nblks > 0
            and prev_nblks <= nblks
            and bt_prev == st["last_bt_end"]
        )
        if is_contiguous:
            start_blk = past // bs
            blk_ids = block_table[0, start_blk:nblks]
            _quant_blocks(blk_ids, out_offset=start_blk if compact else 0)
        else:
            blk_ids = block_table[0, :nblks]
            _quant_blocks(blk_ids, out_offset=0)
        st["last_bt0"] = bt0
        st["last_bt_end"] = bt_end
        st["last_seqlen"] = seqlen
        st["last_nblks"] = nblks
        return k_fp8, k_scale, k_mid, k_rad, k_bt_out, k_bt_mode

    # Multi-request prefill chunk: quantize all active blocks across requests & reset solo continuity.
    if seq_lens_cpu is not None and len(seq_lens_cpu) >= num_reqs:
        seqlens_list = [int(seq_lens_cpu[r]) for r in range(num_reqs)]
    else:
        seqlens_list = cache_seqlens[:num_reqs].tolist()
    max_bt_cols = int(block_table.shape[1])
    req_nblks = [
        min(max_bt_cols, (int(sl) + bs - 1) // bs) if int(sl) > 0 else 0
        for sl in seqlens_list
    ]
    total_blks = sum(req_nblks)
    st["last_bt0"] = -1
    st["last_seqlen"] = -1
    st["last_nblks"] = 0
    if compact:
        if total_blks > cap_blks:
            return None, None, None, None, None, 0
        max_nb = max(1, max(req_nblks)) if req_nblks else 1
        k_bt_rows = []
        off = 0
        for nb in req_nblks:
            row_vals = list(range(off, off + nb)) + [0] * (max_nb - nb)
            k_bt_rows.append(row_vals)
            off += nb
        chunks = [block_table[r, :nb] for r, nb in enumerate(req_nblks) if nb > 0]
        if chunks:
            _quant_blocks(torch.cat(chunks, dim=0), out_offset=0)
        k_bt_multi = torch.tensor(k_bt_rows, dtype=torch.int32, device=key_cache.device)
        return k_fp8, k_scale, k_mid, k_rad, k_bt_multi, 2
    chunks = [block_table[r, :nb] for r, nb in enumerate(req_nblks) if nb > 0]
    if chunks:
        _quant_blocks(torch.cat(chunks, dim=0), out_offset=0)
    return k_fp8, k_scale, k_mid, k_rad, None, 0


def _prepare_fp8_shadow_k(
    key_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    num_reqs: int,
    bs: int,
    gmod,
    *,
    is_v3: bool = True,
    with_kmax: bool = False,
    seq_lens_cpu=None,
    query_lens_cpu=None,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    k_fp8, k_scale, k_mid, k_rad, _, _ = _prepare_fp8_shadow_k_ex(
        key_cache, block_table, cache_seqlens, cu_seqlens_q, num_reqs, bs, gmod,
        is_v3=is_v3, with_kmax=with_kmax,
        seq_lens_cpu=seq_lens_cpu, query_lens_cpu=query_lens_cpu,
    )
    return k_fp8, k_scale, k_mid, k_rad


def _launch_prefill_opt(args, gate=None, sinks=None, result=None, seq_lens_cpu=None, query_lens_cpu=None):
    (q, key_cache, value_cache, block_table, cache_seqlens, cu_seqlens_q,
     sheared_bias, out, lse, max_seqlen_q, softmax_scale, window, E, EW, G, BS, H, Hkv, D, num_reqs) = args
    fuse = gate is not None
    dummy_g = gate if fuse else q
    dummy_s = sinks if fuse else q
    dummy_y = result if fuse else q
    sg_t = gate.stride(0) if fuse else 0
    ssk_t = sinks.stride(0) if fuse else 0
    if window >= 0:
        bm, bn, warps, stages = 32, 64, 4, 2
        grid = (triton.cdiv(max_seqlen_q * G, bm), num_reqs, Hkv)
        _attn_prefill_local_kernel[grid](
            q, key_cache, value_cache, block_table, cache_seqlens, cu_seqlens_q,
            sheared_bias, out, lse, dummy_g, dummy_s, dummy_y,
            q.stride(0), q.stride(1),
            key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
            value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
            block_table.stride(0), sheared_bias.stride(0), sheared_bias.stride(1),
            out.stride(0), out.stride(1), lse.stride(0), lse.stride(1),
            sg_t, ssk_t, softmax_scale * LOG2E,
            WINDOW=window, E=E, EW=EW, G=G, BS=BS, D=D,
            BLOCK_M=bm, BLOCK_N=bn, ALIGN_KBEG_M64=True, FUSE_GS=fuse,
            max_bt_cols=block_table.shape[1],
            num_warps=warps, num_stages=stages,
        )
    else:
        if (
            _PF_GLOBAL_ENABLED
            and not torch.cuda.is_current_stream_capturing()
            and (
                max_seqlen_q >= _PF_GLOBAL_MIN_QLEN
                or (_PF_GLOBAL_WARM_MIN_QLEN >= 64 and max_seqlen_q >= _PF_GLOBAL_WARM_MIN_QLEN)
            )
        ):
            if seq_lens_cpu is not None and len(seq_lens_cpu) >= num_reqs and num_reqs > 0:
                max_seq = int(max(seq_lens_cpu[:num_reqs]))
            else:
                max_seq = int(cache_seqlens[:num_reqs].max().item()) if num_reqs > 0 else 0
            eff_min_past = (
                _PF_GLOBAL_MIN_PAST
                if max_seqlen_q >= _PF_GLOBAL_MIN_QLEN
                else max(_PF_GLOBAL_MIN_PAST, _PF_GLOBAL_WARM_MIN_PAST)
            )
            if (max_seq - max_seqlen_q) >= eff_min_past:
                from . import global_attn_1m as _gmod
                k_fp8 = k_scale = k_scale64 = k_mid = k_rad = k_bt = None
                k_bt_mode = 0
                if _PF_GLOBAL_MODE == "fp8_sage2_cuda_v3":
                    k_fp8, k_scale64, k_mid, k_rad, k_bt, k_bt_mode = _prepare_fp8_shadow_k_ex(
                        key_cache, block_table, cache_seqlens, cu_seqlens_q, num_reqs, BS, _gmod,
                        is_v3=True, with_kmax=(_PF_GLOBAL_TOPK > 0),
                        seq_lens_cpu=seq_lens_cpu, query_lens_cpu=query_lens_cpu,
                    )
                elif _PF_GLOBAL_MODE == "fp8_sage2_cuda_v2":
                    k_fp8, k_scale, _, _ = _prepare_fp8_shadow_k(
                        key_cache, block_table, cache_seqlens, cu_seqlens_q, num_reqs, BS, _gmod,
                        is_v3=False, with_kmax=False,
                        seq_lens_cpu=seq_lens_cpu, query_lens_cpu=query_lens_cpu,
                    )
                _use_v6_lp = (
                    _PF_FP8_ATTN_V6
                    and _PF_GLOBAL_MODE == "fp8_sage2_cuda_v3"
                    and int(max_seqlen_q) >= _PF_FP8_ATTN_V6_MIN_Q
                    and os.path.exists(_PF_FP8_ATTN_V6_LIB_PATH)
                )
                if _gmod.launch_global_prefill(
                    args,
                    gate=gate,
                    sinks=sinks,
                    result=result,
                    mode=_PF_GLOBAL_MODE,
                    pv_skip_tau=_PF_GLOBAL_TAU,
                    bn=_PF_GLOBAL_BN if _PF_GLOBAL_MODE == "fp8_sage2_cuda_v3" else None,
                    min_ctas=_PF_GLOBAL_MIN_CTAS if _PF_GLOBAL_MODE == "fp8_sage2_cuda_v3" else None,
                    topk_tokens=_PF_GLOBAL_TOPK if _PF_GLOBAL_MODE == "fp8_sage2_cuda_v3" else None,
                    cuda_lib_path=_PF_FP8_ATTN_V6_LIB_PATH if _use_v6_lp else _resolve_pf_global_lib(),
                    use_v6_fp8=_use_v6_lp,
                    pv_f16_acc=_PF_FP8_PV_F16_ACC,
                    k_fp8_cache=k_fp8,
                    k_scale_cache=k_scale,
                    k_scale64_cache=k_scale64,
                    k_mid_cache=k_mid,
                    k_rad_cache=k_rad,
                    max_seq=max_seq,
                    k_bt=k_bt,
                    k_bt_mode=k_bt_mode,
                ):
                    return
        bm, bn, warps, stages = (128, 64, 8, 2) if max_seqlen_q >= 1024 else (64, 64, 4, 2)
        grid = (triton.cdiv(max_seqlen_q, bm), num_reqs, H)
        _attn_prefill_full_kernel[grid](
            q, key_cache, value_cache, block_table, cache_seqlens, cu_seqlens_q,
            sheared_bias, out, lse, dummy_g, dummy_s, dummy_y,
            q.stride(0), q.stride(1),
            key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
            value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
            block_table.stride(0), sheared_bias.stride(0), sheared_bias.stride(1),
            out.stride(0), out.stride(1), lse.stride(0), lse.stride(1),
            sg_t, ssk_t, softmax_scale * LOG2E,
            E=E, EW=EW, G=G, BS=BS, D=D,
            BLOCK_M=bm, BLOCK_N=bn, FUSE_GS=fuse,
            max_bt_cols=block_table.shape[1],
            num_warps=warps, num_stages=stages,
        )


def _merge(op, lp, out, lse, num_splits, H, D, skip_lse: bool = False, pdl: bool = False):
    T = op.shape[1]
    _merge_kernel[(T,)](
        op, lp, out, lse,
        op.stride(0), op.stride(1), op.stride(2),
        lp.stride(0), lp.stride(1), lp.stride(2),
        out.stride(0), out.stride(1), lse.stride(0), lse.stride(1),
        H=H, HP=triton.next_power_of_2(H),
        D=D, S=num_splits, SP=triton.next_power_of_2(num_splits),
        SKIP_LSE=skip_lse, PDL=pdl,
        num_warps=4, launch_pdl=pdl,
    )


def consume_pending(output, gate, lse_t, sinks, result) -> bool:
    """Called by attention_ops.gate_sink.  True if a deferred merge, S=1 decode
    attention, or prefill attention was fused into the gate/sink launch; otherwise
    any pending work is flushed into out/lse first and False is returned."""
    global _PENDING, _PENDING_PREFILL, _PENDING_PREFILL_CPU_LENS, _PENDING_S1, _WO_ARMED
    ps1, _PENDING_S1 = _PENDING_S1, None
    if ps1 is not None:
        _WO_ARMED = False
        (
            q_s, kc_s, vc_s, bt_s, seq_s, cuq_s, bias_s, out_s, lse_s,
            _max_q_s, scale_s, win_s, E_s, EW_s, G_s, BS_s, H_s, Hkv_s, D_s,
            nreqs_s, bm_s, bn_s, warps_s, stages_s, mtiles_s, nl_s,
        ) = ps1
        nt = out_s.shape[0]
        T = output.shape[0]
        if (
            out_s.data_ptr() == output.data_ptr()
            and lse_s.data_ptr() == lse_t.data_ptr()
            and output.shape[1:] == (H_s, D_s)
            and nt <= T
            and gate.stride(1) == 1
            and sinks.stride(1) == 1
            and result.is_contiguous()
            and result.shape == gate.shape
        ):
            y_sub = result[:nt]
            g_sub = gate[:nt]
            sk_sub = sinks[:nt]
            _attn_kernel[(mtiles_s, nreqs_s, Hkv_s)](
                q_s, kc_s, vc_s, bt_s, seq_s, cuq_s,
                bias_s, y_sub, y_sub,
                q_s.stride(0), q_s.stride(1),
                kc_s.stride(0), kc_s.stride(1), kc_s.stride(2),
                vc_s.stride(0), vc_s.stride(1), vc_s.stride(2),
                bt_s.stride(0),
                bias_s.stride(0), bias_s.stride(1),
                0, 0, 0,
                0, 0, 0,
                scale_s * LOG2E,
                WINDOW=win_s, E=E_s, EW=EW_s, G=G_s, BS=BS_s, D=D_s,
                BLOCK_M=bm_s, BLOCK_N=bn_s, NUM_SPLITS=1, DRAFT=False,
                REQ_BOUNDS=mtiles_s > 1,
                PDL=_PDL,
                NLivePtr=nl_s if nl_s is not None else cuq_s,
                HAS_NLIVE=nl_s is not None,
                TreeDepthPtr=cuq_s,
                TreePathPtr=cuq_s,
                TreeMetaPtr=cuq_s,
                HAS_TREE=False,
                HAS_TREE_2B=False,
                TREE_FAST_EXIT=False,
                TREE_MAX_M=8,
                GATE=g_sub,
                SINK=sk_sub,
                Y=y_sub,
                sg_t=g_sub.stride(0),
                ssk_t=sk_sub.stride(0),
                sy_t=y_sub.stride(0),
                FUSE_GS=True,
                max_bt_cols=bt_s.shape[1],
                num_warps=warps_s, num_stages=stages_s, launch_pdl=_PDL,
            )
            if nt < T:
                result[nt:].zero_()
            return True
        # Fallback flush into out_s / lse_s if caller tensors did not match
        op_f, lp_f = out_s.unsqueeze(0), lse_s.t().unsqueeze(0)
        _attn_kernel[(mtiles_s, nreqs_s, Hkv_s)](
            q_s, kc_s, vc_s, bt_s, seq_s, cuq_s,
            bias_s, op_f, lp_f,
            q_s.stride(0), q_s.stride(1),
            kc_s.stride(0), kc_s.stride(1), kc_s.stride(2),
            vc_s.stride(0), vc_s.stride(1), vc_s.stride(2),
            bt_s.stride(0),
            bias_s.stride(0), bias_s.stride(1),
            op_f.stride(0), op_f.stride(1), op_f.stride(2),
            lp_f.stride(0), lp_f.stride(1), lp_f.stride(2),
            scale_s * LOG2E,
            WINDOW=win_s, E=E_s, EW=EW_s, G=G_s, BS=BS_s, D=D_s,
            BLOCK_M=bm_s, BLOCK_N=bn_s, NUM_SPLITS=1, DRAFT=False,
            REQ_BOUNDS=mtiles_s > 1,
            PDL=_PDL,
            NLivePtr=nl_s if nl_s is not None else cuq_s,
            HAS_NLIVE=nl_s is not None,
            TreeDepthPtr=cuq_s,
            TreePathPtr=cuq_s,
            TreeMetaPtr=cuq_s,
            HAS_TREE=False,
            HAS_TREE_2B=False,
            TREE_FAST_EXIT=False,
            TREE_MAX_M=8,
            max_bt_cols=bt_s.shape[1],
            num_warps=warps_s, num_stages=stages_s, launch_pdl=_PDL,
        )
        return False
    pp, _PENDING_PREFILL = _PENDING_PREFILL, None
    cpu_lens, _PENDING_PREFILL_CPU_LENS = _PENDING_PREFILL_CPU_LENS, (None, None)
    if pp is not None:
        _WO_ARMED = False
        out_p, lse_p = pp[7], pp[8]
        nt, H, D = out_p.shape[0], out_p.shape[1], out_p.shape[2]
        T = output.shape[0]
        if (out_p.data_ptr() == output.data_ptr() and lse_p.data_ptr() == lse_t.data_ptr()
                and output.shape[1:] == (H, D) and nt <= T and gate.stride(1) == 1
                and sinks.stride(1) == 1 and result.is_contiguous() and result.shape == gate.shape):
            _launch_prefill_opt(
                pp, gate=gate, sinks=sinks, result=result,
                seq_lens_cpu=cpu_lens[0], query_lens_cpu=cpu_lens[1],
            )
            if nt < T:
                result[nt:].zero_()
            return True
        _launch_prefill_opt(pp, seq_lens_cpu=cpu_lens[0], query_lens_cpu=cpu_lens[1])
        return False
    p, _PENDING = _PENDING, None
    _WO_ARMED = False
    if p is None:
        return False
    out, lse, op, lp, S, nl_buf = p
    nt, H, D = op.shape[1], op.shape[2], op.shape[3]
    T = output.shape[0]
    if (out.data_ptr() == output.data_ptr() and lse.data_ptr() == lse_t.data_ptr()
            and output.shape[1:] == (H, D) and nt <= T and gate.stride(1) == 1
            and sinks.stride(1) == 1 and result.is_contiguous() and result.shape == gate.shape):
        if (
            _MGS_CUDA
            and S in (8, 16)
            and D == 128
            and H == 6
            and gate.dtype == torch.bfloat16
            and sinks.dtype == torch.bfloat16
            and result.dtype == torch.bfloat16
            and op.stride(3) == 1
        ):
            lib = _get_mgs_lib()
            if lib is not None:
                rc = lib.merge_gate_sink_exact_fwd(
                    op.data_ptr(), lp.data_ptr(), gate.data_ptr(), sinks.data_ptr(), result.data_ptr(),
                    nl_buf.data_ptr() if nl_buf is not None else 0,
                    op.stride(0), op.stride(1), op.stride(2),
                    lp.stride(0), lp.stride(1), lp.stride(2),
                    gate.stride(0), sinks.stride(0),
                    T, nt, H, S,
                    1 if nl_buf is not None else 0,
                    1 if _PDL else 0,
                    1,
                    int(torch.cuda.current_stream(result.device).cuda_stream),
                )
                if rc == 0:
                    _WO_ARMED = _PDL
                    return True
        _merge_gate_sink_kernel[(T,)](
            op, lp, gate, sinks, result,
            op.stride(0), op.stride(1), op.stride(2),
            lp.stride(0), lp.stride(1), lp.stride(2),
            gate.stride(0), sinks.stride(0), nt,
            H=H, HP=triton.next_power_of_2(H), D=D, S=S, SP=triton.next_power_of_2(S), TRIG=_PDL,
            NLivePtr=nl_buf if nl_buf is not None else op,
            HAS_NLIVE=nl_buf is not None,
            num_warps=4, launch_pdl=True,
        )
        _WO_ARMED = _PDL
        return True
    _merge(op, lp, out, lse, S, H, D)
    return False


def rel_attention(q, key_cache, value_cache, *, block_table, cache_seqlens,
                  cu_seqlens_q, max_seqlen_q, softmax_scale, window_size,
                  sheared_bias, out, lse, num_splits=None, draft=False):
    global _PENDING, _PENDING_PREFILL, _PENDING_PREFILL_CPU_LENS, _PENDING_S1, _PREFILL_CPU_LENS, _WO_ARMED
    from torch._subclasses.fake_tensor import is_fake
    if is_fake(q):
        return
    cpu_lens, _PREFILL_CPU_LENS = _PREFILL_CPU_LENS, (None, None)
    _PENDING = None
    _PENDING_PREFILL = None
    _PENDING_S1 = None
    _PENDING_PREFILL_CPU_LENS = (None, None)
    _WO_ARMED = False
    T, H, D = q.shape
    num_reqs = cu_seqlens_q.shape[0] - 1
    if T <= 0 or num_reqs <= 0:
        return
    Hkv = key_cache.shape[-2]
    G = H // Hkv
    BS = key_cache.shape[1]
    if draft:
        EW = E = 0
        sheared_bias = q
    else:
        EW = sheared_bias.shape[-1]
        E = EW - 256
    window = window_size[0] if window_size != (-1, -1) else -1
    rows = max_seqlen_q * G
    split_row_cap = 96 if _T16_SPLITK else 64
    if rows <= 16:
        bm, bn, warps, stages = 16, 64, 4, 2
    elif rows <= split_row_cap:
        if _BM16:
            bm, bn, warps, stages = _BM, _BM16_BN, _BM16_WARPS, _BM16_STAGES
        else:
            bm, bn, warps, stages = 64, 64, 4, 2
    else:
        bm, bn, warps, stages = 64, 64, 4, 2
    mtiles = triton.cdiv(rows, bm)
    if num_splits is None:
        num_splits = _plan_target_splits(
            num_reqs, rows, mtiles, window, split_row_cap, draft=draft
        )
    if rows <= split_row_cap and bm == 16 and stages < 2:
        if _C32_STAGES2 and num_splits == 1:
            # C=1 sets INKLING_ATTN_BM16_STAGES=1 (short 1-2 tile splits); S=1 runs
            # 16..64+ K-tiles/CTA and needs stages=2 cp.async pipelining (100% 0-ULP!).
            stages = 2
        elif _C48_SPLIT_POLICY and not draft and 2 <= num_reqs <= 8 and window < 0:
            # Wave-V7 C=2..8 Global Attention (S=4..16, 84..96 CTAs <= 2 CTAs/SM) runs
            # 8..4096 K-tiles/CTA and needs stages=2 cp.async pipelining!
            stages = 2
    if (_PREFILL_OPT and not draft and num_splits == 1 and rows > 64
            and E > 0 and (T + 128) * H * EW < (1 << 30)):
        prefill_args = (
            q, key_cache, value_cache, block_table, cache_seqlens, cu_seqlens_q,
            sheared_bias, out, lse, max_seqlen_q, softmax_scale, window, E, EW, G, BS, H, Hkv, D, num_reqs,
        )
        if _FUSE_GS:
            _PENDING_PREFILL = prefill_args
            _PENDING_PREFILL_CPU_LENS = cpu_lens
        else:
            _launch_prefill_opt(
                prefill_args, seq_lens_cpu=cpu_lens[0], query_lens_cpu=cpu_lens[1]
            )
        return
    if num_splits == 1:
        op, lp = out.unsqueeze(0), lse.t().unsqueeze(0)
    else:
        op = torch.empty((num_splits, T, H, D), dtype=torch.float32, device=q.device)
        lp = torch.empty((num_splits, T, H), dtype=torch.float32, device=q.device)
    tree_st = None
    if not draft and num_reqs == 1 and rows <= split_row_cap:
        try:
            from . import tree_spec_runtime as _tsr
            tree_st = _tsr.get_active_tree_state(T)
        except Exception:
            tree_st = None
    has_tree = tree_st is not None
    has_tree_2b = (
        has_tree
        and getattr(tree_st, "tree_meta", None) is not None
        and getattr(tree_st, "ws_max", 4) <= 2
        and max_seqlen_q <= 8
        and G == 6
        and bm == 16
    )
    grid_m = 4 if has_tree_2b else (max_seqlen_q if has_tree else mtiles)
    grid = (grid_m, num_reqs, Hkv * num_splits)
    nl_buf = _rowar_nlive_buf(T) if (not draft) else None
    if (
        _FUSE_GS_S1
        and _C48_SPLIT_POLICY
        and not draft
        and num_splits == 1
        and rows <= split_row_cap
        and not has_tree
        and D == 128
    ):
        # Defer S=1 decode launch to consume_pending() for true single-launch 0-ULP
        # fused Gate+Sink (bypasses attention.py:958 `not use_scheduler` dead-code guard!).
        _PENDING_S1 = (
            q, key_cache, value_cache, block_table, cache_seqlens, cu_seqlens_q,
            sheared_bias, out, lse, max_seqlen_q, softmax_scale, window, E, EW, G, BS,
            H, Hkv, D, num_reqs, bm, bn, warps, stages, mtiles, nl_buf,
        )
        return
    use_pdl = _DRAFT_PDL if draft else _PDL
    _attn_kernel[grid](
        q, key_cache, value_cache, block_table, cache_seqlens, cu_seqlens_q,
        sheared_bias, op, lp,
        q.stride(0), q.stride(1),
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
        value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
        block_table.stride(0),
        sheared_bias.stride(0) if not draft else 0,
        sheared_bias.stride(1) if not draft else 0,
        op.stride(0), op.stride(1), op.stride(2),
        lp.stride(0), lp.stride(1), lp.stride(2),
        softmax_scale * LOG2E,
        WINDOW=window, E=E, EW=EW, G=G, BS=BS, D=D,
        BLOCK_M=bm, BLOCK_N=bn, NUM_SPLITS=num_splits, DRAFT=draft,
        REQ_BOUNDS=(not draft) and mtiles > 1 and rows <= split_row_cap,  # decode BM16 tiles only; prefill keeps per-tile bounds
        PDL=use_pdl,
        NLivePtr=nl_buf if nl_buf is not None else cu_seqlens_q,
        HAS_NLIVE=nl_buf is not None,
        TreeDepthPtr=tree_st.tree_depth if has_tree else cu_seqlens_q,
        TreePathPtr=tree_st.tree_path if has_tree else cu_seqlens_q,
        TreeMetaPtr=tree_st.tree_meta if has_tree_2b else cu_seqlens_q,
        HAS_TREE=has_tree and (not has_tree_2b),
        HAS_TREE_2B=has_tree_2b,
        TREE_FAST_EXIT=has_tree_2b and _TREE_ATTN_FAST_EXIT,
        TREE_MAX_M=tree_st.max_m if has_tree else 8,
        max_bt_cols=block_table.shape[1],
        num_warps=warps, num_stages=stages, launch_pdl=use_pdl,
    )
    if num_splits > 1:
        if _FUSE_GS and not draft:
            _PENDING = (out, lse, op, lp, num_splits, nl_buf)
        else:
            _merge(
                op, lp, out, lse, num_splits, H, D,
                skip_lse=draft, pdl=draft and _DRAFT_PDL,
            )


def can_fuse_gs_s1(
    *,
    num_reqs: int,
    max_seqlen_q: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    num_splits: int | None,
) -> bool:
    """True when Target `_attention` will run `NUM_SPLITS == 1` `_attn_kernel` and can
    fuse `Gate + Sink` directly in registers at literal 0-ULP (`nd == 0`)."""
    if not _FUSE_GS_S1 or num_reqs <= 0 or num_kv_heads <= 0 or head_dim != 128:
        return False
    G = num_heads // num_kv_heads
    rows = max_seqlen_q * G
    split_row_cap = 96 if (_T16_SPLITK and num_reqs == 1) else 64
    if rows > split_row_cap:
        return False
    if _C48_SPLIT_POLICY and 2 <= num_reqs <= 8:
        # C=2..8 uses S=1 for local SWA and S>1 for global; defer via rel_attention->_PENDING_S1
        # where window_size is known instead of hardcoding S=1 across both layer types.
        return False
    if num_splits is None:
        eff_splits = 1 if num_reqs > 4 else 8
    else:
        eff_splits = int(num_splits)
    return eff_splits == 1


def rel_attention_fused_s1(
    q, key_cache, value_cache, *, block_table, cache_seqlens,
    cu_seqlens_q, max_seqlen_q, softmax_scale, window_size,
    sheared_bias, gate, sinks, y_out,
):
    """0-ULP single-launch Target Attention + Gate + Sink for `NUM_SPLITS == 1` (`B >= 5` C=32).
    Eliminates FP32 `out`/`lse` allocations, `zero_()` launches, and `_gate_sink_kernel`!"""
    T, H, D = q.shape
    num_reqs = cache_seqlens.shape[0]
    if T <= 0 or num_reqs <= 0:
        return
    Hkv = key_cache.shape[2]
    G = H // Hkv
    BS = key_cache.shape[1]
    window = window_size[0]
    EW = sheared_bias.shape[2]
    E = EW - 256
    bm = _BM if _BM16 else 64
    bn = _BM16_BN if _BM16 else 64
    warps = _BM16_WARPS if _BM16 else 4
    stages = 2 if (_C32_STAGES2 and bm == 16) else (_BM16_STAGES if _BM16 else 2)
    mtiles = triton.cdiv(max_seqlen_q * G, bm)
    nl_buf = _rowar_nlive_buf(T)
    use_pdl = _PDL
    _attn_kernel[(mtiles, num_reqs, Hkv)](
        q, key_cache, value_cache, block_table, cache_seqlens, cu_seqlens_q,
        sheared_bias, y_out, y_out,
        q.stride(0), q.stride(1),
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
        value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
        block_table.stride(0),
        sheared_bias.stride(0),
        sheared_bias.stride(1),
        0, 0, 0,
        0, 0, 0,
        softmax_scale * LOG2E,
        WINDOW=window, E=E, EW=EW, G=G, BS=BS, D=D,
        BLOCK_M=bm, BLOCK_N=bn, NUM_SPLITS=1, DRAFT=False,
        REQ_BOUNDS=mtiles > 1,
        PDL=use_pdl,
        NLivePtr=nl_buf if nl_buf is not None else cu_seqlens_q,
        HAS_NLIVE=nl_buf is not None,
        TreeDepthPtr=cu_seqlens_q,
        TreePathPtr=cu_seqlens_q,
        TreeMetaPtr=cu_seqlens_q,
        HAS_TREE=False,
        HAS_TREE_2B=False,
        TREE_FAST_EXIT=False,
        TREE_MAX_M=8,
        GATE=gate,
        SINK=sinks,
        Y=y_out,
        sg_t=gate.stride(0),
        ssk_t=sinks.stride(0),
        sy_t=y_out.stride(0),
        FUSE_GS=True,
        max_bt_cols=block_table.shape[1],
        num_warps=warps, num_stages=stages, launch_pdl=use_pdl,
    )


# ---------------------------------------------------------------------------
# Wave-V5 Generalized 6-Warp Fused Drafter Block-Attention (`B = 1..32, T_req <= 8`)
# Cuts C=32 (`B = 2..32`) drafter KV DRAM reads 3.0x -> 1.0x (`251.7 -> 83.9 MB` @ B=32),
# eliminates all 5 `_merge_kernel` launches across ALL `B = 1..32`, and overlaps tile-0
# historical KV fetches before `gdc_wait()`. Falls back cleanly to C=1 v4 or Triton.
# ---------------------------------------------------------------------------
_DRAFT_ATTN_V5 = os.environ.get("INKLING_DRAFT_ATTN_V5", "0").strip().lower() not in (
    "0", "", "off", "false", "no"
)
_DRAFT_ATTN_V5_LIB_PATH = os.environ.get(
    "INKLING_DRAFT_ATTN_V5_LIB", "/rdma/libdrafter_attn_v5.so"
)
_DRAFT_ATTN_V5_MAX_REQS = max(
    1, min(32, int(os.environ.get("INKLING_DRAFT_ATTN_V5_MAX_REQS", "32") or "32"))
)
_DRAFT_ATTN_V5_B32_SPLITS = max(
    1, min(4, int(os.environ.get("INKLING_DRAFT_ATTN_V5_B32_SPLITS", "2") or "2"))
)
_DRAFT_ATTN_V10_S48 = os.environ.get("INKLING_DRAFT_ATTN_V10_S48", "0").strip().lower() not in (
    "0", "", "off", "false", "no"
)
_DRAFT_ATTN_V10_VER = 0
_DRAFT_ATTN_V5_LIB = None
_DRAFT_ATTN_V5_TRIED = False
_DRAFT_ATTN_V5_WS: dict[int, tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]] = {}
_DRAFT_ATTN_V5_BANK: dict[int, int] = {}


def _get_draft_attn_v5_lib():
    global _DRAFT_ATTN_V5_LIB, _DRAFT_ATTN_V5_TRIED, _DRAFT_ATTN_V10_VER
    if not _DRAFT_ATTN_V5_TRIED:
        _DRAFT_ATTN_V5_TRIED = True
        if _DRAFT_ATTN_V5 and os.path.isfile(_DRAFT_ATTN_V5_LIB_PATH):
            try:
                lib = ctypes.CDLL(_DRAFT_ATTN_V5_LIB_PATH)
                lib.drafter_attn_v5_fwd.argtypes = [
                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_int64, ctypes.c_int64,
                    ctypes.c_int64, ctypes.c_int64,
                    ctypes.c_int64, ctypes.c_int64,
                    ctypes.c_int64,
                    ctypes.c_int64, ctypes.c_int64,
                    ctypes.c_float,
                    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                    ctypes.c_uint64,
                ]
                lib.drafter_attn_v5_fwd.restype = ctypes.c_int
                ver_fn = getattr(lib, "drafter_attn_v10_version", None)
                if ver_fn is not None:
                    ver_fn.argtypes = []
                    ver_fn.restype = ctypes.c_int
                    _DRAFT_ATTN_V10_VER = int(ver_fn())
                    print(
                        f"[inkling-draft-attn-v10] loaded {_DRAFT_ATTN_V5_LIB_PATH} "
                        f"(ver={_DRAFT_ATTN_V10_VER}, max_reqs={_DRAFT_ATTN_V5_MAX_REQS}, "
                        f"b32_splits={_DRAFT_ATTN_V5_B32_SPLITS}, s48={int(_DRAFT_ATTN_V10_S48)})",
                        flush=True,
                    )
                else:
                    _DRAFT_ATTN_V10_VER = 5
                    print(
                        f"[inkling-draft-attn-v5] loaded {_DRAFT_ATTN_V5_LIB_PATH} "
                        f"(max_reqs={_DRAFT_ATTN_V5_MAX_REQS}, b32_splits={_DRAFT_ATTN_V5_B32_SPLITS})",
                        flush=True,
                    )
                _DRAFT_ATTN_V5_LIB = lib
            except OSError:
                _DRAFT_ATTN_V5_LIB = None
                _DRAFT_ATTN_V10_VER = 0
    return _DRAFT_ATTN_V5_LIB


def prewarm_draft_attn_v5(device: torch.device) -> bool:
    if not _DRAFT_ATTN_V5:
        return False
    lib = _get_draft_attn_v5_lib()
    if lib is None:
        return False
    idx = int(device.index or 0)
    if idx not in _DRAFT_ATTN_V5_WS:
        if torch.cuda.is_current_stream_capturing():
            return False
        # 2-bank ping-pong [512, 6, 128] fp32 (1.50 MiB/bank): covers S*q_cap <= 512 across ALL B=1..32!
        _DRAFT_ATTN_V5_WS[idx] = (
            [
                torch.empty((512, 6, 128), dtype=torch.float32, device=device),
                torch.empty((512, 6, 128), dtype=torch.float32, device=device),
            ],
            [
                torch.empty((512, 6), dtype=torch.float32, device=device),
                torch.empty((512, 6), dtype=torch.float32, device=device),
            ],
            torch.zeros((32,), dtype=torch.int32, device=device),
        )
        _DRAFT_ATTN_V5_BANK[idx] = 0
    return True


_DRAFT_ATTN_V9_SPLITS = os.environ.get("INKLING_DRAFT_ATTN_V9_SPLITS", "0").strip().lower() not in (
    "0", "", "off", "false", "no"
)


def _plan_draft_attn_v5_splits(num_reqs: int, is_local: bool) -> tuple[int, int]:
    """Return `(num_splits, q_cap)` guaranteeing `num_splits * q_cap <= 512` and `num_reqs * 8 <= q_cap`."""
    if num_reqs <= 1:
        if _DRAFT_ATTN_V10_S48 and _DRAFT_ATTN_V10_VER >= 10 and not is_local:
            # Wave-V10: full 48/48 GB10 SM saturation for B=1 Global drafter attention (48*8 = 384 <= 512 scratch!)
            return 48, 8
        if _DRAFT_ATTN_V9_SPLITS and is_local and _DRAFT_ATTN_C1_V4_SPLITS_LOCAL == 16:
            # window=1032 has 17 KV tiles (64 tok/tile): S=17 gives 1 tile/CTA exact (eliminates S=16 2-tile straggler!)
            return 17, 8
        s = _DRAFT_ATTN_C1_V4_SPLITS_LOCAL if is_local else _DRAFT_ATTN_C1_V4_SPLITS_GLOBAL
        return s, 8
    if (_C48_SPLIT_POLICY or _DRAFT_ATTN_C48_SPLIT_POLICY) and 2 <= num_reqs <= 8:
        # Exact 48-SM wave-balanced C=2..8 drafter splits (42..48 CTAs = 0 double-loaded SM tail,
        # S * q_cap <= 384 <= 512 for every B=2..8!).
        q_cap = num_reqs * 8
        if num_reqs == 2:
            return ((17 if _DRAFT_ATTN_V9_SPLITS else 16) if is_local else 24), q_cap
        if num_reqs == 3:
            return 16, q_cap
        if num_reqs == 4:
            return 12, q_cap
        if num_reqs == 5:
            return 9, q_cap
        if num_reqs == 6:
            # Local (window=1032, 17 KV tiles): S=6 keeps max 3 tiles/CTA with 25% less merge traffic.
            # Global (window=4104+, 65+ KV tiles): under INKLING_DRAFT_ATTN_C48_SPLIT_POLICY=1,
            # S=8 achieves exact 48/48 SM saturation (9 vs 11 tiles/CTA, S*q_cap=384 <= 512).
            use_s6 = _DRAFT_ATTN_V9_SPLITS and (is_local or not _DRAFT_ATTN_C48_SPLIT_POLICY)
            return (6 if use_s6 else 8), q_cap
        # B=7 (42 CTAs) and B=8 (48 CTAs exact): eliminates v5 S=8 56/64-CTA 2-CTA/SM wave tail!
        return 6, q_cap
    if num_reqs == 2:
        return (8 if is_local else 16), 16
    if num_reqs <= 4:
        return (8 if is_local else 16), 32
    if num_reqs <= 8:
        return 8, 64
    if num_reqs <= 16:
        return 4, 128
    return _DRAFT_ATTN_V5_B32_SPLITS, 256


def _try_draft_block_attention_v5(
    q, key_cache, value_cache, out, *, block_table, cache_seqlens, cu_seqlens_q, block, window
) -> bool:
    lib = _get_draft_attn_v5_lib()
    if lib is None:
        return False
    num_reqs = int(cu_seqlens_q.shape[0] - 1)
    if num_reqs <= 0 or num_reqs > _DRAFT_ATTN_V5_MAX_REQS or int(cache_seqlens.shape[0]) < num_reqs:
        return False
    T, H, D = q.shape
    if (
        T <= 0 or T > num_reqs * 8 or block <= 0 or block > 8 or H != 6 or D != 128
        or key_cache.ndim != 4 or value_cache.ndim != 4
        or block_table.ndim != 2 or block_table.shape[0] < num_reqs or block_table.shape[1] <= 0
        or key_cache.shape[2] != 1 or value_cache.shape[2] != 1
        or q.dtype != torch.bfloat16 or key_cache.dtype != torch.bfloat16
        or value_cache.dtype != torch.bfloat16 or out.dtype != torch.bfloat16
        or block_table.dtype != torch.int32 or cache_seqlens.dtype != torch.int32
        or cu_seqlens_q.dtype != torch.int32
    ):
        return False
    out_3d = out.view(T, H, D)
    if (
        q.stride(2) != 1 or key_cache.stride(3) != 1
        or value_cache.stride(3) != 1 or out_3d.stride(2) != 1
    ):
        return False
    idx = int(q.device.index or 0)
    ws = _DRAFT_ATTN_V5_WS.get(idx)
    if ws is None:
        if not prewarm_draft_attn_v5(q.device):
            return False
        ws = _DRAFT_ATTN_V5_WS[idx]
    ops, lps, split_done = ws
    bank = _DRAFT_ATTN_V5_BANK.get(idx, 0)
    _DRAFT_ATTN_V5_BANK[idx] = bank ^ 1
    is_local = bool(window and 0 < int(window) <= 2048)
    n_splits, q_cap = _plan_draft_attn_v5_splits(num_reqs, is_local)
    if T > q_cap:
        return False
    seq_arg = (
        cache_seqlens.clamp(0, int(block_table.shape[1]) * int(key_cache.shape[1]))
        if (_DECODE_EDGE_GUARD_V15 and not torch.cuda.is_current_stream_capturing())
        else cache_seqlens
    )
    rc = lib.drafter_attn_v5_fwd(
        q.data_ptr(), key_cache.data_ptr(), value_cache.data_ptr(),
        block_table.data_ptr(), seq_arg.data_ptr(), cu_seqlens_q.data_ptr(),
        ops[bank].data_ptr(), lps[bank].data_ptr(), split_done.data_ptr(), out_3d.data_ptr(),
        q.stride(0), q.stride(1),
        key_cache.stride(0), key_cache.stride(1),
        value_cache.stride(0), value_cache.stride(1),
        block_table.stride(0),
        out_3d.stride(0), out_3d.stride(1),
        float(D ** -0.5),
        int(window or 0), int(key_cache.shape[1]), int(num_reqs), int(n_splits), int(q_cap),
        1 if _DRAFT_PDL else 0,
        int(torch.cuda.current_stream(q.device).cuda_stream),
    )
    return rc == 0


# ---------------------------------------------------------------------------
# C=1 v4 Single-Launch 6-Warp Fused Drafter Block-Attention (B == 1, T <= 8 ONLY)
# Eliminates 3x KV reads (BM=16 -> mtiles=3) and all 5 separate `_merge_kernel`s.
# Strictly opt-in via INKLING_DRAFT_ATTN_C1_V4=1; ZERO change to B >= 2 / C=32!
# ---------------------------------------------------------------------------
_DRAFT_ATTN_C1_V4 = os.environ.get("INKLING_DRAFT_ATTN_C1_V4", "0").strip().lower() not in (
    "0", "", "off", "false", "no"
)
_DRAFT_ATTN_C1_V4_LIB_PATH = os.environ.get(
    "INKLING_DRAFT_ATTN_C1_V4_LIB", "/rdma/libdrafter_attn_c1_v4.so"
)
_DRAFT_ATTN_C1_V4_SPLITS_LOCAL = max(
    1, min(32, int(os.environ.get("INKLING_DRAFT_ATTN_C1_V4_SPLITS_LOCAL", "16") or "16"))
)
_DRAFT_ATTN_C1_V4_SPLITS_GLOBAL = max(
    1, min(32, int(os.environ.get("INKLING_DRAFT_ATTN_C1_V4_SPLITS_GLOBAL", "32") or "32"))
)
_DRAFT_ATTN_C1_V4_LIB = None
_DRAFT_ATTN_C1_V4_TRIED = False
_DRAFT_ATTN_C1_V4_WS: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def _get_draft_attn_c1_v4_lib():
    global _DRAFT_ATTN_C1_V4_LIB, _DRAFT_ATTN_C1_V4_TRIED
    if not _DRAFT_ATTN_C1_V4_TRIED:
        _DRAFT_ATTN_C1_V4_TRIED = True
        if _DRAFT_ATTN_C1_V4 and os.path.isfile(_DRAFT_ATTN_C1_V4_LIB_PATH):
            try:
                lib = ctypes.CDLL(_DRAFT_ATTN_C1_V4_LIB_PATH)
                lib.drafter_attn_c1_v4_fwd.argtypes = [
                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_int64, ctypes.c_int64,
                    ctypes.c_int64, ctypes.c_int64,
                    ctypes.c_int64, ctypes.c_int64,
                    ctypes.c_int64, ctypes.c_int64,
                    ctypes.c_float,
                    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                    ctypes.c_uint64,
                ]
                lib.drafter_attn_c1_v4_fwd.restype = ctypes.c_int
                _DRAFT_ATTN_C1_V4_LIB = lib
                print(
                    f"[inkling-draft-attn-c1-v4] loaded {_DRAFT_ATTN_C1_V4_LIB_PATH} "
                    f"(splits local={_DRAFT_ATTN_C1_V4_SPLITS_LOCAL}, global={_DRAFT_ATTN_C1_V4_SPLITS_GLOBAL})",
                    flush=True,
                )
            except OSError:
                _DRAFT_ATTN_C1_V4_LIB = None
    return _DRAFT_ATTN_C1_V4_LIB


def prewarm_draft_attn_c1_v4(device: torch.device) -> bool:
    if not _DRAFT_ATTN_C1_V4:
        return False
    lib = _get_draft_attn_c1_v4_lib()
    if lib is None:
        return False
    idx = int(device.index or 0)
    if idx not in _DRAFT_ATTN_C1_V4_WS:
        if torch.cuda.is_current_stream_capturing():
            return False
        _DRAFT_ATTN_C1_V4_WS[idx] = (
            torch.empty((32, 8, 6, 128), dtype=torch.float32, device=device),
            torch.empty((32, 8, 6), dtype=torch.float32, device=device),
            torch.zeros((1,), dtype=torch.int32, device=device),
        )
    return True


def _try_draft_block_attention_c1_v4(
    q, key_cache, value_cache, out, *, block_table, cache_seqlens, cu_seqlens_q, block, window
) -> bool:
    lib = _get_draft_attn_c1_v4_lib()
    if lib is None:
        return False
    if int(cu_seqlens_q.shape[0] - 1) != 1:
        return False
    T, H, D = q.shape
    if (
        T <= 0 or T > 8 or block <= 0 or block > 8 or H != 6 or D != 128
        or key_cache.ndim != 4 or value_cache.ndim != 4
        or block_table.ndim != 2 or block_table.shape[0] < 1 or block_table.shape[1] <= 0
        or key_cache.shape[2] != 1 or value_cache.shape[2] != 1
        or q.dtype != torch.bfloat16 or key_cache.dtype != torch.bfloat16
        or value_cache.dtype != torch.bfloat16 or out.dtype != torch.bfloat16
        or block_table.dtype != torch.int32 or cache_seqlens.dtype != torch.int32
        or cu_seqlens_q.dtype != torch.int32
    ):
        return False
    out_3d = out.view(T, H, D)
    if (
        q.stride(2) != 1 or key_cache.stride(3) != 1
        or value_cache.stride(3) != 1 or out_3d.stride(2) != 1
    ):
        return False
    idx = int(q.device.index or 0)
    ws = _DRAFT_ATTN_C1_V4_WS.get(idx)
    if ws is None:
        if not prewarm_draft_attn_c1_v4(q.device):
            return False
        ws = _DRAFT_ATTN_C1_V4_WS[idx]
    op, lp, split_done = ws
    n_splits = (
        _DRAFT_ATTN_C1_V4_SPLITS_LOCAL
        if (window and 0 < int(window) <= 2048)
        else _DRAFT_ATTN_C1_V4_SPLITS_GLOBAL
    )
    seq_arg = (
        cache_seqlens.clamp(0, int(block_table.shape[1]) * int(key_cache.shape[1]))
        if (_DECODE_EDGE_GUARD_V15 and not torch.cuda.is_current_stream_capturing())
        else cache_seqlens
    )
    rc = lib.drafter_attn_c1_v4_fwd(
        q.data_ptr(), key_cache.data_ptr(), value_cache.data_ptr(),
        block_table.data_ptr(), seq_arg.data_ptr(), cu_seqlens_q.data_ptr(),
        op.data_ptr(), lp.data_ptr(), split_done.data_ptr(), out_3d.data_ptr(),
        q.stride(0), q.stride(1),
        key_cache.stride(0), key_cache.stride(1),
        value_cache.stride(0), value_cache.stride(1),
        out_3d.stride(0), out_3d.stride(1),
        float(D ** -0.5),
        int(window or 0), int(key_cache.shape[1]), int(n_splits),
        0,  # Legacy c1_v4 has pre-V9 #206 PDL launch-at-entry hazard; force pdl=0!
        int(torch.cuda.current_stream(q.device).cuda_stream),
    )
    return rc == 0


def draft_block_attention(q, key_cache, value_cache, out, *, block_table,
                          cache_seqlens, cu_seqlens_q, block, window, num_splits):
    """SM12x drop-in for ops.attention.block_decode_attention: bf16 ``out``
    ``(T, H * D)``; each ``block``-token request sees its keys up to the last
    one, or the last ``window`` of them if ``window``."""
    if q.shape[0] <= 0 or int(cu_seqlens_q.shape[0] - 1) <= 0:
        return
    if _DRAFT_ATTN_V5 and _try_draft_block_attention_v5(
        q, key_cache, value_cache, out,
        block_table=block_table, cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q, block=block, window=window,
    ):
        return
    if _DRAFT_ATTN_C1_V4 and _try_draft_block_attention_c1_v4(
        q, key_cache, value_cache, out,
        block_table=block_table, cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q, block=block, window=window,
    ):
        return
    T, H, D = q.shape
    lse = torch.empty((H, T), dtype=torch.float32, device=q.device)
    rel_attention(
        q, key_cache, value_cache, block_table=block_table,
        cache_seqlens=cache_seqlens, cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=block, softmax_scale=D ** -0.5,
        window_size=(window - 1, 0) if window else (-1, -1),
        sheared_bias=None, out=out.view(T, H, D), lse=lse,
        num_splits=None, draft=True,
    )


# ---------------------------------------------------------------------------
# Phase 2: Fused SM121a Local SWA + On-The-Fly `r @ rel_proj_t` (opt-in!)
# ---------------------------------------------------------------------------
_PF_LOCAL_OTF_REL = os.environ.get("INKLING_PF_LOCAL_OTF_REL", "0").strip().lower() not in (
    "0", "", "off", "false", "no"
)
_PF_LOCAL_OTF_REL_MIN_T = int(
    os.environ.get("INKLING_PF_LOCAL_OTF_REL_MIN_T", "1024") or "1024"
)
_PF_LOCAL_OTF_LIB_PATH = os.environ.get(
    "INKLING_PF_LOCAL_OTF_LIB", "/rdma/liblocal_swa_otf_rel_sm121.so"
)
_PF_LOCAL_OTF_LIB = None
_PF_LOCAL_OTF_TRIED = False

# Wave-V10 opt-in FP8 P@V attention v6 (`libpf_fp8_attn_v6.so`, default OFF!)
# Strictly gated to eager prefill `max_seqlen_q >= 1024` & `!capturing` so Warm-1M
# (`qlen=436`) stays on `v5_zspill` + BF16 OTF and decode (`M <= 128`) is 100% untouched.
_PF_FP8_ATTN_V13 = os.environ.get("INKLING_PF_FP8_ATTN_V13", "0").strip().lower() in (
    "1", "on", "true", "yes"
)
_PF_FP8_ATTN_V6 = _PF_FP8_ATTN_V13 or (
    os.environ.get("INKLING_PF_FP8_ATTN_V6", "0").strip().lower() in (
        "1", "on", "true", "yes"
    )
)
_PF_FP8_ATTN_V13_LIB_PATH = os.environ.get(
    "INKLING_PF_FP8_ATTN_V13_LIB", "/rdma/libpf_fp8_attn_v13.so"
).strip()
_PF_FP8_ATTN_V6_LIB_PATH = (
    _PF_FP8_ATTN_V13_LIB_PATH
    if (_PF_FP8_ATTN_V13 and _PF_FP8_ATTN_V13_LIB_PATH and os.path.exists(_PF_FP8_ATTN_V13_LIB_PATH))
    else os.environ.get("INKLING_PF_FP8_ATTN_V6_LIB", "/rdma/libpf_fp8_attn_v6.so").strip()
)
_PF_FP8_PV_F16_ACC = os.environ.get("INKLING_PF_FP8_PV_F16_ACC", "0").strip().lower() in (
    "1", "on", "true", "yes"
)
_PF_FP8_ATTN_V6_MIN_Q = max(
    int(os.environ.get("INKLING_PF_FP8_ATTN_V6_MIN_Q", "1024") or "1024"), 1024
)
_PF_FP8_ATTN_V6_LIB = None
_PF_FP8_ATTN_V6_TRIED = False
_KV_BF12_PREFILL_RESOLVER = None


def register_kv_bf12_prefill_resolver(fn) -> None:
    """Register a KV-BF12 prefill unpack hook (e.g. `libkv_bf12_paged_v10.so`):
    `fn(key_cache, value_cache, block_table, cache_seqlens, *, need_k, need_v, window, cu_seqlens_q, max_seqlen_q)`
    returning `(bf16_k_or_none, bf16_v_or_none, resolved_bt)`."""
    global _KV_BF12_PREFILL_RESOLVER
    _KV_BF12_PREFILL_RESOLVER = fn


def get_kv_bf12_prefill_resolver():
    return _KV_BF12_PREFILL_RESOLVER


def is_kv_bf12_cache(cache: torch.Tensor | None) -> bool:
    """Detect single-pool KV-BF12 cache tensors: `(NBLK, Hkv, 256, 198)` bf16 or `uint8` `shape[-1]==396`."""
    if cache is None:
        return False
    if bool(getattr(cache, "_inkling_kv_bf12", False)):
        return True
    if cache.dtype == torch.uint8:
        return True
    return cache.ndim >= 2 and int(cache.shape[-1]) in (198, 396)


def _ensure_kv_bf12_prefill_resolver() -> None:
    if _KV_BF12_PREFILL_RESOLVER is not None:
        return
    try:
        from . import sm12_attn_fused as _sm12_f
        if hasattr(_sm12_f, "ensure_kv_bf12_prefill_resolver_registered"):
            _sm12_f.ensure_kv_bf12_prefill_resolver_registered()
    except Exception:
        pass


def resolve_kv_bf12_for_prefill(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    *,
    need_k: bool = True,
    need_v: bool = True,
    window: int = -1,
    cu_seqlens_q: torch.Tensor | None = None,
    max_seqlen_q: int = 0,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    if not (is_kv_bf12_cache(key_cache) or is_kv_bf12_cache(value_cache)):
        return key_cache, value_cache, block_table
    _ensure_kv_bf12_prefill_resolver()
    resolver = _KV_BF12_PREFILL_RESOLVER
    if resolver is None:
        return None, None, None
    return resolver(
        key_cache,
        value_cache,
        block_table,
        cache_seqlens,
        need_k=need_k,
        need_v=need_v,
        window=window,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=max_seqlen_q,
    )


def get_pf_fp8_attn_v6_lib():
    """Lazily load compiled SM121a `libpf_fp8_attn_v6.so` (strictly opt-in via INKLING_PF_FP8_ATTN_V6=1)."""
    global _PF_FP8_ATTN_V6_LIB, _PF_FP8_ATTN_V6_TRIED
    if not _PF_FP8_ATTN_V6:
        return None
    if not _PF_FP8_ATTN_V6_TRIED:
        _PF_FP8_ATTN_V6_TRIED = True
        if _PF_FP8_ATTN_V6_LIB_PATH and os.path.exists(_PF_FP8_ATTN_V6_LIB_PATH):
            lib = ctypes.CDLL(_PF_FP8_ATTN_V6_LIB_PATH)
            lib.pf_fp8_attn_v6_version.argtypes = []
            lib.pf_fp8_attn_v6_version.restype = ctypes.c_int
            lib.local_swa_fp8_otf_rel_v6_sm121_fwd.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int64, ctypes.c_int64,
                ctypes.c_int64, ctypes.c_int64,
                ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
                ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
                ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_float, ctypes.c_float, ctypes.c_float,
                ctypes.c_uint64,
            ]
            lib.local_swa_fp8_otf_rel_v6_sm121_fwd.restype = ctypes.c_int
            lib.global_tail_fp8_otf_rel_v6_sm121_fwd.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int64, ctypes.c_int64,
                ctypes.c_int64, ctypes.c_int64,
                ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
                ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
                ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_float, ctypes.c_float, ctypes.c_float,
                ctypes.c_uint64,
            ]
            lib.global_tail_fp8_otf_rel_v6_sm121_fwd.restype = ctypes.c_int
            _PF_FP8_ATTN_V6_LIB = lib
            print(
                f"[inkling-pf-fp8-v6] loaded {_PF_FP8_ATTN_V6_LIB_PATH} "
                f"(ver={lib.pf_fp8_attn_v6_version()}, pv_f16={int(_PF_FP8_PV_F16_ACC)}, "
                f"min_q={_PF_FP8_ATTN_V6_MIN_Q})",
                flush=True,
            )
    return _PF_FP8_ATTN_V6_LIB


def get_local_otf_rel_lib():
    global _PF_LOCAL_OTF_LIB, _PF_LOCAL_OTF_TRIED
    if not _PF_LOCAL_OTF_TRIED:
        _PF_LOCAL_OTF_TRIED = True
        if _PF_LOCAL_OTF_REL and os.path.exists(_PF_LOCAL_OTF_LIB_PATH):
            import ctypes

            lib = ctypes.CDLL(_PF_LOCAL_OTF_LIB_PATH)
            lib.local_swa_otf_rel_bf16_sm121_fwd.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int64, ctypes.c_int64,
                ctypes.c_int64, ctypes.c_int64,
                ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
                ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
                ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_float, ctypes.c_float, ctypes.c_float,
                ctypes.c_uint64,
            ]
            lib.local_swa_otf_rel_bf16_sm121_fwd.restype = ctypes.c_int
            _PF_LOCAL_OTF_LIB = lib
            print(f"[inkling-pf-local-otf] loaded {_PF_LOCAL_OTF_LIB_PATH}", flush=True)
    return _PF_LOCAL_OTF_LIB


def local_swa_otf_rel_prefill(
    q: torch.Tensor,          # [nt, H, 128] bf16 (clipped_q[:nt])
    r: torch.Tensor,          # [nt, H, 16] bf16 (raw r[:nt], may be row-strided!)
    r_scale: torch.Tensor,    # [H, 16] bf16 contiguous
    rel_proj_t: torch.Tensor, # [E, 16] bf16 contiguous
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    gate: torch.Tensor,       # [nt, H*128] bf16
    sinks: torch.Tensor,      # [nt, H] bf16
    result: torch.Tensor,     # [nt, H*128] bf16
    *,
    max_seqlen_q: int,
    softmax_scale: float,
    window: int,
    rel_threshold: float,
    clip_eps: float = 1e-6,
) -> bool:
    use_v6 = (
        _PF_FP8_ATTN_V6
        and int(max_seqlen_q) >= _PF_FP8_ATTN_V6_MIN_Q
        and not torch.cuda.is_current_stream_capturing()
    )
    v6_lib = get_pf_fp8_attn_v6_lib() if use_v6 else None
    lib = get_local_otf_rel_lib()
    if v6_lib is None and lib is None:
        return False
    key_cache, value_cache, block_table = resolve_kv_bf12_for_prefill(
        key_cache, value_cache, block_table, cache_seqlens,
        need_k=True, need_v=True, window=window,
        cu_seqlens_q=cu_seqlens_q, max_seqlen_q=max_seqlen_q,
    )
    if key_cache is None or value_cache is None or block_table is None:
        return False
    T, H, D = q.shape
    num_reqs = int(cu_seqlens_q.shape[0] - 1)
    if T <= 0 or num_reqs <= 0:
        return True
    Hkv = key_cache.shape[2]
    BS = key_cache.shape[1]
    E = rel_proj_t.shape[0]
    if (
        D != 128 or r.shape[2] != 16 or BS < 64 or (BS % 64) != 0
        or q.dtype != torch.bfloat16 or r.dtype != torch.bfloat16
        or key_cache.dtype != torch.bfloat16 or value_cache.dtype != torch.bfloat16
        or q.stride(2) != 1 or r.stride(2) != 1 or gate.stride(1) != 1
        or sinks.stride(1) != 1 or result.stride(1) != 1
        or key_cache.stride(3) != 1 or value_cache.stride(3) != 1
    ):
        return False
    if _DECODE_EDGE_GUARD_V15 and block_table.ndim == 2 and block_table.shape[1] > 0:
        cache_seqlens = cache_seqlens.clamp(0, int(block_table.shape[1]) * int(BS))
    stream_u64 = int(torch.cuda.current_stream(q.device).cuda_stream)
    if v6_lib is not None:
        rc6 = v6_lib.local_swa_fp8_otf_rel_v6_sm121_fwd(
            q.data_ptr(), r.data_ptr(), r_scale.data_ptr(), rel_proj_t.data_ptr(),
            key_cache.data_ptr(), value_cache.data_ptr(),
            block_table.data_ptr(), cache_seqlens.data_ptr(), cu_seqlens_q.data_ptr(),
            gate.data_ptr(), sinks.data_ptr(), result.data_ptr(), 0, 0,
            q.stride(0), q.stride(1), r.stride(0), r.stride(1),
            key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
            value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
            block_table.stride(0), gate.stride(0), sinks.stride(0), result.stride(0),
            0, 0, 0, 0,
            num_reqs, int(max_seqlen_q), int(H), int(H // Hkv),
            int(BS), int(window), int(E), int(_PF_FP8_PV_F16_ACC),
            float(softmax_scale * LOG2E), float(rel_threshold), float(clip_eps),
            stream_u64,
        )
        if rc6 == 0:
            return True
    if lib is None:
        return False
    rc = lib.local_swa_otf_rel_bf16_sm121_fwd(
        q.data_ptr(), r.data_ptr(), r_scale.data_ptr(), rel_proj_t.data_ptr(),
        key_cache.data_ptr(), value_cache.data_ptr(),
        block_table.data_ptr(), cache_seqlens.data_ptr(), cu_seqlens_q.data_ptr(),
        gate.data_ptr(), sinks.data_ptr(), result.data_ptr(), 0, 0,
        q.stride(0), q.stride(1), r.stride(0), r.stride(1),
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
        value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
        block_table.stride(0), gate.stride(0), sinks.stride(0), result.stride(0),
        0, 0, 0, 0,
        num_reqs, int(max_seqlen_q), int(H), int(H // Hkv),
        int(BS), int(window), int(E),
        float(softmax_scale * LOG2E), float(rel_threshold), float(clip_eps),
        int(torch.cuda.current_stream(q.device).cuda_stream),
    )
    return rc == 0


# ---------------------------------------------------------------------------
# Phase 2 Strike: Fused SM121a Global Biased-Tail + On-The-Fly `r_clip_tau @ rel_proj_t`
# Eliminates 100% of `_rel_prefill_kernel` + `rel_logits` (126-249 MiB) on all 8 global layers!
# Strictly opt-in prefill-only (`INKLING_PF_GLOBAL_OTF_REL=1`, `T >= 1024`, `!capturing`).
# ---------------------------------------------------------------------------
_PF_GLOBAL_OTF_REL = os.environ.get("INKLING_PF_GLOBAL_OTF_REL", "0").strip().lower() not in (
    "0", "", "off", "false", "no"
)
_PF_GLOBAL_OTF_REL_MIN_T = int(
    os.environ.get("INKLING_PF_GLOBAL_OTF_REL_MIN_T", "1024") or "1024"
)
_PF_GLOBAL_TAIL_OTF_LIB_PATH = os.environ.get(
    "INKLING_PF_GLOBAL_TAIL_OTF_LIB", "/rdma/libglobal_tail_otf_rel_sm121.so"
)
_PF_GLOBAL_TAIL_OTF_BN = int(
    os.environ.get("INKLING_PF_GLOBAL_TAIL_OTF_BN", "64") or "64"
)
_PF_GLOBAL_TAIL_OTF_LIB = None
_PF_GLOBAL_TAIL_OTF_TRIED = False


def get_global_tail_otf_rel_lib():
    global _PF_GLOBAL_TAIL_OTF_LIB, _PF_GLOBAL_TAIL_OTF_TRIED
    if not _PF_GLOBAL_TAIL_OTF_TRIED:
        _PF_GLOBAL_TAIL_OTF_TRIED = True
        if _PF_GLOBAL_OTF_REL and os.path.exists(_PF_GLOBAL_TAIL_OTF_LIB_PATH):
            lib = ctypes.CDLL(_PF_GLOBAL_TAIL_OTF_LIB_PATH)
            lib.global_tail_otf_rel_bf16_sm121_fwd.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int64, ctypes.c_int64,
                ctypes.c_int64, ctypes.c_int64,
                ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
                ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
                ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_float, ctypes.c_float, ctypes.c_float,
                ctypes.c_uint64,
            ]
            lib.global_tail_otf_rel_bf16_sm121_fwd.restype = ctypes.c_int
            _PF_GLOBAL_TAIL_OTF_LIB = lib
            print(
                f"[inkling-pf-global-otf] loaded {_PF_GLOBAL_TAIL_OTF_LIB_PATH}",
                flush=True,
            )
    return _PF_GLOBAL_TAIL_OTF_LIB


def can_global_otf_rel_prefill(
    q: torch.Tensor,
    r: torch.Tensor,
    tau: torch.Tensor | None,
    rel_proj_t: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    gate: torch.Tensor,
    sinks: torch.Tensor,
    result: torch.Tensor,
    *, kv_bf12: bool = False,
) -> bool:
    """Fast shape/alignment check BEFORE skipping rel_logits in attention_prologue."""
    base_ok = (
        q.shape[2] == 128
        and r.shape[2] == 16
        and q.dtype == torch.bfloat16
        and r.dtype == torch.bfloat16
        and rel_proj_t.dtype == torch.bfloat16
        and rel_proj_t.is_contiguous()
        and gate.dtype == torch.bfloat16
        and sinks.dtype == torch.bfloat16
        and result.dtype == torch.bfloat16
        and (tau is None or (tau.dtype == torch.float32 and tau.is_contiguous()))
        and q.stride(2) == 1
        and r.stride(2) == 1
        and gate.stride(1) == 1
        and sinks.stride(1) == 1
        and result.stride(1) == 1
        and (q.data_ptr() & 15) == 0
    )
    if not base_ok:
        return False
    if kv_bf12 or is_kv_bf12_cache(key_cache) or is_kv_bf12_cache(value_cache):
        _ensure_kv_bf12_prefill_resolver()
        if _KV_BF12_PREFILL_RESOLVER is not None:
            return True
        from . import sm12_attn_fused as _sm12_f
        return (
            (_sm12_f.is_kv_bf12_cache(key_cache) or _sm12_f.is_kv_bf12_cache(value_cache))
            and _sm12_f.get_kv_bf12_paged_lib() is not None
        )
    BS = key_cache.shape[1]
    return (
        BS >= 64
        and (BS % 64) == 0
        and key_cache.dtype == torch.bfloat16
        and value_cache.dtype == torch.bfloat16
        and key_cache.stride(3) == 1
        and value_cache.stride(3) == 1
        and (key_cache.data_ptr() & 15) == 0
        and (value_cache.data_ptr() & 15) == 0
    )


def global_otf_rel_prefill(
    q: torch.Tensor,          # [nt, H, 128] bf16 (clipped_q[:nt])
    r: torch.Tensor,          # [nt, H, 16] bf16 (raw r[:nt], may be row-strided!)
    r_scale: torch.Tensor,    # [H, 16] bf16 contiguous
    tau: torch.Tensor | None, # [nt] fp32 contiguous, or None
    rel_proj_t: torch.Tensor, # [E, 16] bf16 contiguous
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    gate: torch.Tensor,       # [nt, H*128] bf16
    sinks: torch.Tensor,      # [nt, H] bf16
    result: torch.Tensor,     # [nt, H*128] bf16
    *,
    max_seqlen_q: int,
    softmax_scale: float,
    rel_threshold: float,
    clip_eps: float = 1e-6,
    seq_lens_cpu=None,
    query_lens_cpu=None,
    kv_bf12_cache=None,
    kv_bf12_emerg=None,
) -> bool:
    """Full global prefill path with ZERO rel_logits allocation and ZERO _rel_prefill_kernel:
      - When past >= MIN_PAST and Sage2 v3/v4/v6 is active: runs Sage2 prefix -> CUDA OTF tail merge.
      - When past < MIN_PAST (e.g. chunk 0) or Sage2 unavailable: runs CUDA OTF full causal (v6 FP8 or BF16)!
      - Composes seamlessly with KV-BF12 (both explicit `kv_bf12_cache` 3-stage minimal-VRAM staging
        and registered `resolve_kv_bf12_for_prefill` for 198-bf16 / 396-u8 views)."""
    use_v6 = (
        _PF_FP8_ATTN_V6
        and int(max_seqlen_q) >= _PF_FP8_ATTN_V6_MIN_Q
        and not torch.cuda.is_current_stream_capturing()
    )
    v6_lib = get_pf_fp8_attn_v6_lib() if use_v6 else None
    tail_lib = get_global_tail_otf_rel_lib()
    if v6_lib is None and tail_lib is None:
        return False
    use_kv_bf12 = kv_bf12_cache is not None and kv_bf12_emerg is not None
    if not use_kv_bf12 and (is_kv_bf12_cache(key_cache) or is_kv_bf12_cache(value_cache)):
        key_cache, value_cache, block_table = resolve_kv_bf12_for_prefill(
            key_cache, value_cache, block_table, cache_seqlens,
            need_k=True, need_v=True, window=-1,
            cu_seqlens_q=cu_seqlens_q, max_seqlen_q=max_seqlen_q,
        )
        if key_cache is None or value_cache is None or block_table is None:
            return False
    if not can_global_otf_rel_prefill(
        q, r, tau, rel_proj_t, key_cache, value_cache, gate, sinks, result,
        kv_bf12=use_kv_bf12,
    ):
        return False

    T, H, D = q.shape
    num_reqs = int(cu_seqlens_q.shape[0] - 1)
    if T <= 0 or num_reqs <= 0:
        return True
    Hkv = int(kv_bf12_cache.shape[1]) if use_kv_bf12 else int(key_cache.shape[2])
    G = H // Hkv
    BS = 256 if use_kv_bf12 else int(key_cache.shape[1])
    E = rel_proj_t.shape[0]
    if _DECODE_EDGE_GUARD_V15 and block_table.ndim == 2 and block_table.shape[1] > 0:
        cache_seqlens = cache_seqlens.clamp(0, int(block_table.shape[1]) * int(BS))
    if seq_lens_cpu is not None and len(seq_lens_cpu) >= num_reqs and num_reqs > 0:
        max_seq = int(max(seq_lens_cpu[:num_reqs]))
    else:
        max_seq = int(cache_seqlens[:num_reqs].max().item()) if num_reqs > 0 else 0

    acc_buf = m_buf = l_buf = None
    prefix_bm = 64
    prefix_bn = _PF_GLOBAL_TAIL_OTF_BN

    _warm_q_ok = (
        _PF_GLOBAL_WARM_MIN_QLEN >= 64
        and max_seqlen_q >= _PF_GLOBAL_WARM_MIN_QLEN
        and (max_seq - max_seqlen_q) >= max(_PF_GLOBAL_MIN_PAST, _PF_GLOBAL_WARM_MIN_PAST)
        and not torch.cuda.is_current_stream_capturing()
    )
    if (
        _PF_GLOBAL_ENABLED
        and (max_seqlen_q >= _PF_GLOBAL_MIN_QLEN or _warm_q_ok)
        and (max_seq - max_seqlen_q) >= _PF_GLOBAL_MIN_PAST
        and _PF_GLOBAL_MODE == "fp8_sage2_cuda_v3"
    ):
        from . import global_attn_1m as _gmod

        k_fp8, k_scale64, k_mid, k_rad, k_bt, k_bt_mode = _prepare_fp8_shadow_k_ex(
            key_cache, block_table, cache_seqlens, cu_seqlens_q, num_reqs, BS, _gmod,
            is_v3=True, with_kmax=(_PF_GLOBAL_TOPK > 0),
            seq_lens_cpu=seq_lens_cpu, query_lens_cpu=query_lens_cpu,
            kv_bf12_cache=kv_bf12_cache if use_kv_bf12 else None,
            kv_bf12_emerg=kv_bf12_emerg if use_kv_bf12 else None,
        )
        prefix_res = None
        if use_kv_bf12:
            from . import sm12_attn_fused as _sm12_f
            _, v_prefix_stage, bt_prefix_v = _sm12_f.kv_bf12_stage_active_blocks(
                kv_bf12_cache, kv_bf12_emerg, block_table, cache_seqlens,
                cu_seqlens_q=cu_seqlens_q, window=-1, unpack_k=False, unpack_v=True,
                seq_lens_cpu=seq_lens_cpu, query_lens_cpu=query_lens_cpu,
            )
            # If non-compact Shadow-K (k_bt_mode==0), pass original physical block_table via k_bt_mode=2
            # so v5/v6 reads K from physical block IDs and V from dense remapped `bt_prefix_v`!
            eff_k_bt = block_table if (k_bt_mode == 0 and k_fp8 is not None) else k_bt
            eff_k_bt_mode = 2 if (k_bt_mode == 0 and k_fp8 is not None) else k_bt_mode
            if v6_lib is not None:
                prefix_res = _gmod.launch_global_sage2_prefix_v3_only(
                    q, key_cache, v_prefix_stage, bt_prefix_v, cache_seqlens, cu_seqlens_q,
                    max_seqlen_q, softmax_scale, E, G, BS, H, D, num_reqs,
                    pv_skip_tau=_PF_GLOBAL_TAU,
                    bn=_PF_GLOBAL_BN,
                    min_ctas=_PF_GLOBAL_MIN_CTAS,
                    topk_tokens=_PF_GLOBAL_TOPK,
                    cuda_lib_path=_PF_FP8_ATTN_V6_LIB_PATH,
                    k_fp8_cache=k_fp8,
                    k_scale64_cache=k_scale64,
                    k_mid_cache=k_mid,
                    k_rad_cache=k_rad,
                    max_seq=max_seq,
                    k_bt=eff_k_bt,
                    k_bt_mode=eff_k_bt_mode,
                    use_v6_fp8=True,
                    pv_f16_acc=_PF_FP8_PV_F16_ACC,
                )
            if prefix_res is None:
                prefix_res = _gmod.launch_global_sage2_prefix_v3_only(
                    q, key_cache, v_prefix_stage, bt_prefix_v, cache_seqlens, cu_seqlens_q,
                    max_seqlen_q, softmax_scale, E, G, BS, H, D, num_reqs,
                    pv_skip_tau=_PF_GLOBAL_TAU,
                    bn=_PF_GLOBAL_BN,
                    min_ctas=_PF_GLOBAL_MIN_CTAS,
                    topk_tokens=_PF_GLOBAL_TOPK,
                    cuda_lib_path=_resolve_pf_global_lib(),
                    k_fp8_cache=k_fp8,
                    k_scale64_cache=k_scale64,
                    k_mid_cache=k_mid,
                    k_rad_cache=k_rad,
                    max_seq=max_seq,
                    k_bt=eff_k_bt,
                    k_bt_mode=eff_k_bt_mode,
                )
            del v_prefix_stage, bt_prefix_v
        else:
            if v6_lib is not None:
                prefix_res = _gmod.launch_global_sage2_prefix_v3_only(
                    q, key_cache, value_cache, block_table, cache_seqlens, cu_seqlens_q,
                    max_seqlen_q, softmax_scale, E, G, BS, H, D, num_reqs,
                    pv_skip_tau=_PF_GLOBAL_TAU,
                    bn=_PF_GLOBAL_BN,
                    min_ctas=_PF_GLOBAL_MIN_CTAS,
                    topk_tokens=_PF_GLOBAL_TOPK,
                    cuda_lib_path=_PF_FP8_ATTN_V6_LIB_PATH,
                    k_fp8_cache=k_fp8,
                    k_scale64_cache=k_scale64,
                    k_mid_cache=k_mid,
                    k_rad_cache=k_rad,
                    max_seq=max_seq,
                    k_bt=k_bt,
                    k_bt_mode=k_bt_mode,
                    use_v6_fp8=True,
                    pv_f16_acc=_PF_FP8_PV_F16_ACC,
                )
            if prefix_res is None:
                prefix_res = _gmod.launch_global_sage2_prefix_v3_only(
                    q, key_cache, value_cache, block_table, cache_seqlens, cu_seqlens_q,
                    max_seqlen_q, softmax_scale, E, G, BS, H, D, num_reqs,
                    pv_skip_tau=_PF_GLOBAL_TAU,
                    bn=_PF_GLOBAL_BN,
                    min_ctas=_PF_GLOBAL_MIN_CTAS,
                    topk_tokens=_PF_GLOBAL_TOPK,
                    cuda_lib_path=_resolve_pf_global_lib(),
                    k_fp8_cache=k_fp8,
                    k_scale64_cache=k_scale64,
                    k_mid_cache=k_mid,
                    k_rad_cache=k_rad,
                    max_seq=max_seq,
                    k_bt=k_bt,
                    k_bt_mode=k_bt_mode,
                )
        if prefix_res is not None:
            acc_buf, m_buf, l_buf, prefix_bm, prefix_bn = prefix_res

    tail_bn = _PF_GLOBAL_TAIL_OTF_BN
    if tail_bn not in (32, 64) or (prefix_bn % tail_bn) != 0:
        tail_bn = 64 if (prefix_bn % 64) == 0 else 32

    if use_kv_bf12:
        from . import sm12_attn_fused as _sm12_f
        tail_win = (int(E) + int(BS)) if acc_buf is not None else -1
        k_tail, v_tail, bt_tail = _sm12_f.kv_bf12_stage_active_blocks(
            kv_bf12_cache, kv_bf12_emerg, block_table, cache_seqlens,
            cu_seqlens_q=cu_seqlens_q, window=tail_win, unpack_k=True, unpack_v=True,
            seq_lens_cpu=seq_lens_cpu, query_lens_cpu=query_lens_cpu,
        )
    else:
        k_tail, v_tail, bt_tail = key_cache, value_cache, block_table

    stream_u64 = int(torch.cuda.current_stream(q.device).cuda_stream)
    if v6_lib is not None:
        rc6 = v6_lib.global_tail_fp8_otf_rel_v6_sm121_fwd(
            q.data_ptr(),
            r.data_ptr(),
            r_scale.data_ptr(),
            tau.data_ptr() if tau is not None else 0,
            rel_proj_t.data_ptr(),
            k_tail.data_ptr(),
            v_tail.data_ptr(),
            bt_tail.data_ptr(),
            cache_seqlens.data_ptr(),
            cu_seqlens_q.data_ptr(),
            acc_buf.data_ptr() if acc_buf is not None else 0,
            m_buf.data_ptr() if m_buf is not None else 0,
            l_buf.data_ptr() if l_buf is not None else 0,
            gate.data_ptr(),
            sinks.data_ptr(),
            result.data_ptr(),
            0,
            0,
            q.stride(0), q.stride(1),
            r.stride(0), r.stride(1),
            k_tail.stride(0), k_tail.stride(1), k_tail.stride(2),
            v_tail.stride(0), v_tail.stride(1), v_tail.stride(2),
            bt_tail.stride(0),
            gate.stride(0), sinks.stride(0), result.stride(0),
            0, 0, 0, 0,
            num_reqs, int(max_seqlen_q), int(H), int(G),
            int(BS), int(E),
            int(prefix_bm), int(prefix_bn), int(tail_bn), int(_PF_FP8_PV_F16_ACC),
            float(softmax_scale * LOG2E), float(rel_threshold), float(clip_eps),
            stream_u64,
        )
        if rc6 == 0:
            return True
    if tail_lib is None:
        return False
    rc = tail_lib.global_tail_otf_rel_bf16_sm121_fwd(
        q.data_ptr(),
        r.data_ptr(),
        r_scale.data_ptr(),
        tau.data_ptr() if tau is not None else 0,
        rel_proj_t.data_ptr(),
        k_tail.data_ptr(),
        v_tail.data_ptr(),
        bt_tail.data_ptr(),
        cache_seqlens.data_ptr(),
        cu_seqlens_q.data_ptr(),
        acc_buf.data_ptr() if acc_buf is not None else 0,
        m_buf.data_ptr() if m_buf is not None else 0,
        l_buf.data_ptr() if l_buf is not None else 0,
        gate.data_ptr(),
        sinks.data_ptr(),
        result.data_ptr(),
        0,
        0,
        q.stride(0), q.stride(1),
        r.stride(0), r.stride(1),
        k_tail.stride(0), k_tail.stride(1), k_tail.stride(2),
        v_tail.stride(0), v_tail.stride(1), v_tail.stride(2),
        bt_tail.stride(0),
        gate.stride(0), sinks.stride(0), result.stride(0),
        0, 0, 0, 0,
        num_reqs, int(max_seqlen_q), int(H), int(G),
        int(BS), int(E),
        int(prefix_bm), int(prefix_bn), int(tail_bn),
        float(softmax_scale * LOG2E), float(rel_threshold), float(clip_eps),
        stream_u64,
    )
    return rc == 0
