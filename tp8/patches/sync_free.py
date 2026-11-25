# SPDX-License-Identifier: Apache-2.0
"""Inkling-2-Small greedy "sync-free" target head (output-exact; port of DSv4.1 DSV41_SYNC_FREE).

conc-sched concurrency upgrades (100% bit-exact, zero B=1 change):
  1. INKLING_SYNC_FREE_MAX_M (default 256, previously hardcoded 128): prevents the
     catastrophic C24/C32 cliff where M=192..256 disarms sync_free and falls back
     to a 154-206 MB fp32 full-vocab all-gather over RoCEv2!
  2. INKLING_SYNC_FREE_K0=1 (default 1): hooks Sampler.__call__ in addition to
     RejectionSampler.__call__ so greedy K=0 steps (Dynamic SD at B>=16 and
     greedy prefill final chunks) stay armed with cert_head + 64 B/row RDMA AG
     instead of falling back to full [M, 201024] fp32 all-gather + stock head.
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch

try:  # CPU unit tests import this file without vllm
    from vllm.logger import init_logger

    logger = init_logger(__name__)
except Exception:  # noqa: BLE001
    import logging

    logger = logging.getLogger("inkling_sync_free")

_OFF = ("", "0", "off", "false", "no")


def parse_mode(raw: str) -> str:
    v = (raw or "").strip().lower()
    if v in _OFF:
        return "off"
    if v in ("1", "on", "all", "true", "argmax"):
        return "argmax"
    if v == "audit":
        return "audit"
    logger.warning("INKLING_SYNC_FREE=%r not recognised (expected off|argmax|audit); feature OFF", raw)
    return "off"


MODE = parse_mode(os.environ.get("INKLING_SYNC_FREE", "off"))
AUDIT_EVERY = int(os.environ.get("INKLING_SYNC_FREE_AUDIT_EVERY", "2000") or 2000)
SPLIT = int(os.environ.get("INKLING_SYNC_FREE_SPLIT", "4") or 4)
AGREE = os.environ.get("INKLING_SYNC_FREE_AGREE", "0").strip().lower() not in _OFF
FUSE = os.environ.get("INKLING_SYNC_FREE_FUSE", "1").strip().lower() not in _OFF
SF_BLOCK = 2048  # local kernel column block
OH_BLOCK = 4096  # one-hot kernel column block
MAX_M = max(1, int(os.environ.get("INKLING_SYNC_FREE_MAX_M", "256") or 256))
ALLOW_K0 = os.environ.get("INKLING_SYNC_FREE_K0", "1").strip().lower() not in _OFF
CERT = os.environ.get("INKLING_CERT_HEAD", "off").strip().lower()
CERT = "off" if CERT in _OFF else ("1" if CERT in ("1", "on", "true", "argmax") else ("check" if CERT in ("check", "audit") else "off"))
DIRECT = os.environ.get("INKLING_SYNC_FREE_DIRECT", "1").strip().lower() not in _OFF
FUSE_REJECT = os.environ.get("INKLING_SYNC_FREE_FUSE_REJECT", "1").strip().lower() not in _OFF
ZERO_COPY_SLICE = os.environ.get("INKLING_SYNC_FREE_SLICE", "1").strip().lower() not in _OFF
FP32_FUSE = os.environ.get("INKLING_SYNC_FREE_FP32_FUSE", "0").strip().lower() not in _OFF
TREE_SPEC = os.environ.get("INKLING_TREE_SPEC", "0").strip().lower() not in _OFF
# Wave-V7 Native SM121a SyncFree CUDA Stack (STRICT 0-ULP, opt-in default OFF = 0)
CUDA_V7 = os.environ.get("INKLING_SYNC_FREE_CUDA_V7", "0").strip().lower() not in _OFF
V7_PDL = os.environ.get("INKLING_SYNC_FREE_V7_PDL", "1").strip().lower() not in _OFF
V7_LIB_PATH = os.environ.get("INKLING_DSPARK_V7_LIB", "/rdma/libdspark_markov_v7.so")
_SF_V7_LIB = None
_SF_V7_TRIED = False
_SF_V7_QUAL: dict[int, bool] = {}
_SF_V7_WS: dict[int, dict] = {}

# Wave-V9 Phase-2: 4-slot static output ring buffer (default OFF = 0, 100% 0-ULP invariant).
# Eliminates 3 per-step torch.empty allocations (sampled, num_sampled, num_rejected) so dtgraph
# Phase 2B ("td") sees fixed device pointers per slot 0..3 and replays statically with 0 ExecUpdate!
STATIC_RING = os.environ.get("INKLING_DTGRAPH_STATIC_RING", "0").strip().lower() not in _OFF
STATIC_RING_SLOTS = 4
# Wave-V11 opt-in host-overhead micro-trims in sync_free (default OFF = 0, 100% 0-ULP exact!):
#   - Eliminates 4 per-step Python imports (dtgraph, NO_LOGPROBS, tensor_model_parallel_all_gather, l2pf).
#   - Caches (slot, num_reqs) static ring slice views + direct last-ring ref (eliminates 3 slice allocs/step).
#   - Caches static [M, 1] dummy logits placeholder (eliminates 1 CUDA tensor alloc/step).
#   - Defers Triton _kernels() lookup until Triton fallback + B=1 fast scalar check in step_armed.
FAST_V11 = os.environ.get("INKLING_SYNC_FREE_FAST_V11", "0").strip().lower() not in _OFF
# Wave-V14 opt-in zero-overhead sync_free host path (default OFF = 0, 100% 0-ULP exact!):
#   - Zero-alloc B=1..8 scalar loop in step_armed (eliminates 4 NumPy temp arrays + 2 ufunc reduces @ C=2..8).
#   - O(1) slot-indexed _StaticOutputRing.acquire_v14 with pre-cached data_ptr/stride/4-tuple (0 allocs/step).
#   - Zero str(device) MRU dummy-logits cache + dead CROSS_BOUNDARY callback elision.
#   - Localized tensor/dtype/pointer fast path in RejectionSampler._direct_call & target_logits.
FAST_V14 = os.environ.get("INKLING_SYNC_FREE_FAST_V14", "0").strip().lower() not in _OFF
# Wave-V16 Crown-Jewel Track 3: Unbiased Zero-Sync SM121a GPU temp > 0 Speculative Rejection Sampler
# (default OFF = 0; pure temp == 0.0 ALWAYS stays 100.000% untouched on the 1dcfb306 greedy path!).
TEMP_V16 = os.environ.get("INKLING_SYNC_FREE_TEMP_V16", "0").strip().lower() not in _OFF
TEMP_V16_LIB_PATH = os.environ.get("INKLING_SYNC_FREE_TEMP_V16_LIB", "/rdma/libsync_free_temp_v16.so")
TEMP_V16_PDL = os.environ.get("INKLING_SYNC_FREE_TEMP_V16_PDL", "1").strip().lower() not in _OFF
TEMP_V16_MAX_M = max(1, int(os.environ.get("INKLING_SYNC_FREE_TEMP_V16_MAX_M", str(MAX_M)) or MAX_M))
_SF_TEMP_V16_LIB = None
_SF_TEMP_V16_TRIED = False
_SF_TEMP_V16_QUAL: dict[int, bool] = {}
_CACHED_CERT_HEAD_MOD = None
_CACHED_CERT_V7_MOD = None
_CACHED_CERT_PROBED = False
_CACHED_DTG_REC_FN = None
_CACHED_NO_LOGPROBS = None
_CACHED_TP_AG = None
_CACHED_L2PF_FORK_CB = None
_CACHED_L2PF_PROBED = False
_DUMMY_LOGITS_CACHE: dict[tuple, torch.Tensor] = {}
_DUMMY_LAST: tuple | None = None
_SF_LAST_RING_OBJ = None
_SF_FAST_RING = None  # (d_idx, cols, ring) single-ring MRU fast reference
_SF_LAST_RING_4TUP: tuple[int, int, int, int] | None = None
_SF_V7_LIB_BY_DEV: dict[int, object] = {}
_NLIVE_HAS_NT: bool | None = None


def _fast_tp_all_gather():
    global _CACHED_TP_AG
    fn = _CACHED_TP_AG
    if fn is None:
        from vllm.distributed import tensor_model_parallel_all_gather as _tpag
        _CACHED_TP_AG = _tpag
        fn = _tpag
    return fn


def _fast_l2pf_fork_cb(tag: str) -> None:
    global _CACHED_L2PF_FORK_CB, _CACHED_L2PF_PROBED
    if not _CACHED_L2PF_PROBED:
        _CACHED_L2PF_PROBED = True
        try:
            from . import l2pf as _l2pf_cb
            if FAST_V14 and not getattr(_l2pf_cb, "CROSS_BOUNDARY", False):
                _CACHED_L2PF_FORK_CB = None
            else:
                _CACHED_L2PF_FORK_CB = getattr(_l2pf_cb, "fork_cross_boundary", None)
        except Exception:
            _CACHED_L2PF_FORK_CB = None
    if _CACHED_L2PF_FORK_CB is not None:
        _CACHED_L2PF_FORK_CB(tag)


def _fast_dummy_logits(m: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    global _DUMMY_LAST
    if FAST_V14:
        d_idx = device.index
        dl = _DUMMY_LAST
        if dl is not None and dl[0] == m and dl[1] == dtype and dl[2] == d_idx and dl[3].device == device:
            return dl[3]
        dk = (m, dtype, device.type, d_idx)
        dm = _DUMMY_LOGITS_CACHE.get(dk)
        if dm is None:
            dm = torch.empty((m, 1), dtype=dtype, device=device)
            _DUMMY_LOGITS_CACHE[dk] = dm
        _DUMMY_LAST = (m, dtype, d_idx, dm)
        return dm
    dk = (m, dtype, str(device))
    dm = _DUMMY_LOGITS_CACHE.get(dk)
    if dm is None:
        dm = torch.empty((m, 1), dtype=dtype, device=device)
        _DUMMY_LOGITS_CACHE[dk] = dm
    return dm


class _StaticOutputRing:
    """Per-device 4-slot static ring buffer for (sampled, num_sampled, num_rejected) with Event fences."""

    def __init__(self, device: torch.device, max_reqs: int, cols: int):
        self.device = device
        self.max_reqs = max(64, int(max_reqs))
        self.cols = int(cols)
        self.sampled = [
            torch.empty((self.max_reqs, self.cols), dtype=torch.int64, device=device)
            for _ in range(STATIC_RING_SLOTS)
        ]
        self.num_sampled = [
            torch.empty((self.max_reqs,), dtype=torch.int32, device=device)
            for _ in range(STATIC_RING_SLOTS)
        ]
        self.num_rejected = [
            torch.empty((self.max_reqs,), dtype=torch.int32, device=device)
            for _ in range(STATIC_RING_SLOTS)
        ]
        is_cuda = getattr(device, "type", "") == "cuda"
        self.events = [torch.cuda.Event() if is_cuda else None for _ in range(STATIC_RING_SLOTS)]
        self.recorded = [False] * STATIC_RING_SLOTS
        self.cursor = 0
        self.acquires = 0
        self.waits = 0
        self._slice_cache: dict[tuple[int, int], tuple] = {}
        self._is_cuda = is_cuda
        self._d_idx = _sf_dev_idx(device) if is_cuda else -1
        self._v14_by_nr: dict[int, tuple] = {}
        self._v14_last_nr: int = -1
        self._v14_last_slots: tuple | None = None

    def _build_v14_slots(self, num_reqs: int) -> tuple:
        slots = []
        for s in range(STATIC_RING_SLOTS):
            s_t = self.sampled[s][:num_reqs]
            ns_t = self.num_sampled[s][:num_reqs]
            nr_t = self.num_rejected[s][:num_reqs]
            p_s = int(s_t.data_ptr())
            p_ns = int(ns_t.data_ptr())
            p_nr = int(nr_t.data_ptr())
            slots.append((s, s_t, ns_t, nr_t, p_s, p_ns, p_nr, (s, p_s, p_ns, p_nr), int(s_t.stride(0))))
        res = tuple(slots)
        self._v14_by_nr[num_reqs] = res
        return res

    def acquire_v14(self, num_reqs: int):
        """Zero-alloc O(1) slot acquire returning pre-cached views, data_ptrs, 4-tuple state & stride."""
        slot = self.cursor
        self.cursor = (slot + 1) & 3
        if self.recorded[slot]:
            ev = self.events[slot]
            if ev is not None:
                if not ev.query():
                    ev.synchronize()
                    self.waits += 1
            self.recorded[slot] = False
        self.acquires += 1
        if num_reqs == self._v14_last_nr:
            return self._v14_last_slots[slot]
        slots = self._v14_by_nr.get(num_reqs)
        if slots is None:
            slots = self._build_v14_slots(num_reqs)
        self._v14_last_nr = num_reqs
        self._v14_last_slots = slots
        return slots[slot]

    def acquire(self, num_reqs: int):
        slot = self.cursor
        self.cursor = (slot + 1) % STATIC_RING_SLOTS
        ev = self.events[slot]
        if self.recorded[slot] and ev is not None:
            if not ev.query():
                ev.synchronize()
                self.waits += 1
            self.recorded[slot] = False
        self.acquires += 1
        if FAST_V11 or FAST_V14:
            ck = (slot, num_reqs)
            res = self._slice_cache.get(ck)
            if res is None:
                res = (
                    slot,
                    self.sampled[slot][:num_reqs],
                    self.num_sampled[slot][:num_reqs],
                    self.num_rejected[slot][:num_reqs],
                )
                self._slice_cache[ck] = res
            return res
        return (
            slot,
            self.sampled[slot][:num_reqs],
            self.num_sampled[slot][:num_reqs],
            self.num_rejected[slot][:num_reqs],
        )

    def fence(self, slot: int, stream=None) -> None:
        global _CACHED_DTG_REC_FN
        ev = self.events[slot]
        if ev is None:
            return
        if stream is None:
            stream = torch.cuda.current_stream(self.device)
        if FAST_V11 or FAST_V14:
            rec_fn = _CACHED_DTG_REC_FN
            if rec_fn is None:
                try:
                    from . import dtgraph as _dtg_m  # noqa: PLC0415
                    rec_fn = _dtg_m._ORIG.get("Event.record", torch.cuda.Event.record)
                except Exception:  # noqa: BLE001
                    rec_fn = torch.cuda.Event.record
                _CACHED_DTG_REC_FN = rec_fn
        else:
            try:
                from . import dtgraph as _dtg_m  # noqa: PLC0415
                rec_fn = _dtg_m._ORIG.get("Event.record", torch.cuda.Event.record)
            except Exception:  # noqa: BLE001
                rec_fn = torch.cuda.Event.record
        rec_fn(ev, stream)
        self.recorded[slot] = True


_SF_RINGS: dict[tuple[int, int], _StaticOutputRing] = {}
_SF_LAST_RING_STATE: tuple[int, int, int, int, int] | None = None  # (dev_idx, slot, p_s, p_ns, p_nr)


def _acquire_static_ring_outputs_v14(device: torch.device, num_reqs: int, cols: int):
    """Zero-alloc V14 acquire returning 9-tuple (slot, s_t, ns_t, nr_t, p_s, p_ns, p_nr, st4, stride0)."""
    global _SF_LAST_RING_STATE, _SF_LAST_RING_OBJ, _SF_FAST_RING, _SF_LAST_RING_4TUP
    if not STATIC_RING:
        _SF_LAST_RING_STATE = None
        _SF_LAST_RING_4TUP = None
        return None
    is_cuda = device.type == "cuda"
    if is_cuda and torch.cuda.is_current_stream_capturing():
        _SF_LAST_RING_STATE = None
        _SF_LAST_RING_4TUP = None
        return None
    d_idx = (device.index if device.index is not None else torch.cuda.current_device()) if is_cuda else -1
    fr = _SF_FAST_RING
    if fr is not None and fr[0] == d_idx and fr[1] == cols and fr[2].max_reqs >= num_reqs:
        ring = fr[2]
    else:
        rkey = (d_idx, cols)
        ring = _SF_RINGS.get(rkey)
        if ring is None or ring.max_reqs < num_reqs:
            ring = _StaticOutputRing(device, max(64, num_reqs), cols)
            _SF_RINGS[rkey] = ring
        _SF_FAST_RING = (d_idx, cols, ring)
        _SF_LAST_RING_OBJ = (d_idx, ring)
    entry = ring.acquire_v14(num_reqs)
    _SF_LAST_RING_4TUP = entry[7]
    _SF_LAST_RING_STATE = None
    return entry


def _acquire_static_ring_outputs(device: torch.device, num_reqs: int, cols: int):
    global _SF_LAST_RING_STATE, _SF_LAST_RING_OBJ, _SF_LAST_RING_4TUP
    if FAST_V14:
        entry = _acquire_static_ring_outputs_v14(device, int(num_reqs), int(cols))
        if entry is None:
            return None
        return entry[0], entry[1], entry[2], entry[3]
    if not STATIC_RING:
        _SF_LAST_RING_STATE = None
        _SF_LAST_RING_4TUP = None
        return None
    is_cuda = getattr(device, "type", "") == "cuda"
    if is_cuda and torch.cuda.is_current_stream_capturing():
        _SF_LAST_RING_STATE = None
        _SF_LAST_RING_4TUP = None
        return None
    d_idx = _sf_dev_idx(device) if is_cuda else -1
    rkey = (d_idx, int(cols))
    ring = _SF_RINGS.get(rkey)
    if ring is None or ring.max_reqs < num_reqs:
        ring = _StaticOutputRing(device, max(64, int(num_reqs)), int(cols))
        _SF_RINGS[rkey] = ring
    slot, s_t, ns_t, nr_t = ring.acquire(int(num_reqs))
    _SF_LAST_RING_OBJ = (d_idx, ring)
    _SF_LAST_RING_4TUP = None
    _SF_LAST_RING_STATE = (
        d_idx,
        slot,
        int(s_t.data_ptr()),
        int(ns_t.data_ptr()),
        int(nr_t.data_ptr()),
    )
    return slot, s_t, ns_t, nr_t


def clear_static_ring_state() -> None:
    """Clear active-step static ring pointer state so fallback steps never leak stale pointers."""
    global _SF_LAST_RING_STATE, _SF_LAST_RING_4TUP
    _SF_LAST_RING_STATE = None
    _SF_LAST_RING_4TUP = None


def get_static_ring_state() -> tuple[int, int, int, int] | None:
    """Return (slot, sampled_ptr, num_sampled_ptr, num_rejected_ptr) for the active step, or None."""
    if _SF_LAST_RING_4TUP is not None:
        return _SF_LAST_RING_4TUP
    st = _SF_LAST_RING_STATE
    if st is None:
        return None
    return st[1], st[2], st[3], st[4]


def fence_static_ring_slot(device: torch.device, slot: int, cols: int | None = None, stream=None) -> None:
    """Record completion event on ring slot `slot` strictly after AsyncOutput + Phase-2B td finish."""
    if FAST_V14 and cols is None and _SF_FAST_RING is not None and len(_SF_RINGS) == 1:
        d_idx = (device.index if device.index is not None else torch.cuda.current_device()) if device.type == "cuda" else -1
        if _SF_FAST_RING[0] == d_idx:
            _SF_FAST_RING[2].fence(slot, stream=stream)
            return
    is_cuda = getattr(device, "type", "") == "cuda"
    d_idx = _sf_dev_idx(device) if is_cuda else -1
    if (FAST_V11 or FAST_V14) and cols is None and _SF_LAST_RING_OBJ is not None and _SF_LAST_RING_OBJ[0] == d_idx and len(_SF_RINGS) == 1:
        _SF_LAST_RING_OBJ[1].fence(int(slot), stream=stream)
        return
    if cols is not None:
        ring = _SF_RINGS.get((d_idx, int(cols)))
        if ring is not None:
            ring.fence(int(slot), stream=stream)
        return
    for (rd, _rc), ring in _SF_RINGS.items():
        if rd == d_idx:
            ring.fence(int(slot), stream=stream)


def _sf_dev_idx(device: torch.device) -> int:
    return device.index if device.index is not None else torch.cuda.current_device()


def _get_sf_v7_lib():
    global _SF_V7_LIB, _SF_V7_TRIED
    if not _SF_V7_TRIED:
        _SF_V7_TRIED = True
        if CUDA_V7 and os.path.isfile(V7_LIB_PATH):
            try:
                import ctypes
                c_vp = ctypes.c_void_p
                c_i = ctypes.c_int
                c_i64 = ctypes.c_int64
                c_u32 = ctypes.c_uint32
                c_u64 = ctypes.c_uint64
                c_f = ctypes.c_float
                lib = ctypes.CDLL(V7_LIB_PATH)
                sigs = {
                    "sync_free_local_pairs_v7_fwd": [
                        c_vp, c_i, c_i64, c_vp, c_vp, c_vp,
                        c_i, c_i, c_i, c_i, c_i,
                        c_f, c_f, c_i, c_i, c_i, c_i, c_i, c_u64,
                    ],
                    "sync_free_k0_tok_status_v7_fwd": [
                        c_vp, c_i, c_i, c_i, c_vp, c_i,
                        c_vp, c_vp, c_vp, c_vp, c_i, c_vp, c_i, c_u64,
                    ],
                    "sync_free_fused_reject_v7_fwd": [
                        c_vp, c_i, c_i, c_i, c_i, c_vp, c_i64,
                        c_vp, c_vp, c_vp, c_vp, c_vp, c_vp, c_vp, c_vp, c_vp,
                        c_u32, c_i, c_u64,
                    ],
                    "sync_free_fused_tree_reject_v7_fwd": [
                        c_vp, c_i, c_i, c_i, c_i, c_i, c_vp, c_i64,
                        c_vp, c_vp, c_vp, c_vp, c_vp, c_vp, c_vp, c_vp, c_vp,
                        c_vp, c_vp, c_vp, c_vp, c_vp, c_u32, c_i, c_u64,
                    ],
                }
                for sym, argtypes in sigs.items():
                    fn = getattr(lib, sym)
                    fn.argtypes = argtypes
                    fn.restype = c_i
                _SF_V7_LIB = lib
                logger.info("INKLING_SYNC_FREE_CUDA_V7=1: loaded %s (16 B/row SPLIT=1 AG + direct reject/K0)", V7_LIB_PATH)
            except (OSError, AttributeError):
                _SF_V7_LIB = None
    return _SF_V7_LIB


def _prealloc_sf_v7_ws(device: torch.device, max_m: int = MAX_M) -> dict | None:
    if not CUDA_V7 or device.type != "cuda" or _capturing():
        return None
    d = _sf_dev_idx(device)
    mm = max(256, int(max_m))
    ws = _SF_V7_WS.get(d)
    if ws is not None and ws["max_m"] >= mm:
        return ws
    ws = {
        "max_m": mm,
        "rec_split": torch.empty((mm, 16, 4), dtype=torch.float32, device=device),
        "pair_out": [
            torch.empty((mm, 4), dtype=torch.float32, device=device),
            torch.empty((mm, 4), dtype=torch.float32, device=device),
        ],
        "sem": [
            torch.zeros((mm,), dtype=torch.int32, device=device),
            torch.zeros((mm,), dtype=torch.int32, device=device),
        ],
        "idx": 0,
    }
    _SF_V7_WS[d] = ws
    return ws


def _sf_v7_ok(device: torch.device) -> bool:
    if not CUDA_V7 or device.type != "cuda":
        return False
    d = _sf_dev_idx(device)
    return bool(_SF_V7_QUAL.get(d, False) and d in _SF_V7_WS and _get_sf_v7_lib() is not None)


@torch.no_grad()
def qualify_sf_v7(device: torch.device, cap: float | None, dt: torch.dtype, fp32_divmode: int) -> bool:
    """Fail-closed 0-ULP startup qualification for SyncFree CUDA v7 (local_pairs SPLIT=1, k0, fused_reject)."""
    d = _sf_dev_idx(device)
    if d in _SF_V7_QUAL:
        return _SF_V7_QUAL[d]
    lib = _get_sf_v7_lib()
    ws = _prealloc_sf_v7_ws(device, MAX_M) if lib is not None else None
    if lib is None or ws is None or _capturing():
        _SF_V7_QUAL[d] = False
        return False
    ok = False
    try:
        stream = int(torch.cuda.current_stream(device).cuda_stream)
        k = _kernels()
        gen = torch.Generator(device=device)
        gen.manual_seed(7091)
        M_q, V_q = 4, 4096
        lg = (torch.randn((M_q, V_q), device=device, generator=gen) * 6.0).to(dt)
        fp32_ch = bool(dt == torch.float32 and FP32_FUSE and cap is not None and fp32_divmode in (0, 1))
        has_cap = cap is not None and (dt == torch.bfloat16 or fp32_ch)
        inv_c = inv_cap_of(cap) if has_cap else 1.0
        cap_f = _f32(cap) if has_cap else 1.0
        # Reference Triton SPLIT=4 -> global winner across single rank
        seg_ref = SF_BLOCK * k["triton"].cdiv(k["triton"].cdiv(V_q, SF_BLOCK), 4)
        p_ref = torch.empty((M_q * 4, 4), dtype=torch.float32, device=device)
        k["local"][(M_q, 4)](
            lg, lg.stride(0), p_ref, V_q, 128, seg_ref, inv_c, cap_f,
            CAP=has_cap, SPLIT=4, BLOCK=SF_BLOCK,
            FP32_CHAIN=fp32_ch, DIVMODE=fp32_divmode if fp32_ch else 0, num_warps=8,
        )
        t_ref = torch.empty((M_q,), dtype=torch.int64, device=device)
        k["tok_only"][(M_q,)](p_ref, M_q, t_ref, TP=1, SPLIT=4, NP=4, PDL=False, num_warps=1)
        # V7 CUDA SPLIT=16 coherent collapse -> (M_q, 4)
        seg_v7 = ((V_q + 15) // 16 + 7) & ~7
        p_v7 = torch.empty((M_q, 4), dtype=torch.float32, device=device)
        sem_v7 = torch.zeros((M_q,), dtype=torch.int32, device=device)
        rc1 = lib.sync_free_local_pairs_v7_fwd(
            lg.data_ptr(), 1 if dt == torch.float32 else 0, int(lg.stride(0)),
            ws["rec_split"].data_ptr(), p_v7.data_ptr(), sem_v7.data_ptr(),
            M_q, 16, V_q, 128, seg_v7, inv_c, cap_f,
            1 if has_cap else 0, 1 if fp32_ch else 0,
            fp32_divmode if fp32_ch else 0, 1, 0, stream,
        )
        t_v7 = torch.empty((M_q,), dtype=torch.int64, device=device)
        rc2 = lib.sync_free_k0_tok_status_v7_fwd(
            p_v7.data_ptr(), M_q, 1, 1, t_v7.data_ptr(), 0,
            0, 0, 0, 0, 0, 0, 0, stream,
        )
        # Also verify exact tie-break + rejection kernel parity on dummy 1-req batch
        g_r = p_v7.repeat(2, 1).contiguous()
        s_ref = torch.full((1, 4), -9, dtype=torch.int64, device=device)
        s_v7 = s_ref.clone()
        ns_r, nr_r = torch.empty((1,), dtype=torch.int32, device=device), torch.empty((1,), dtype=torch.int32, device=device)
        ns_c, nr_c = torch.empty_like(ns_r), torch.empty_like(nr_r)
        sl_q = torch.tensor([10], dtype=torch.int32, device=device)
        ids_q = torch.tensor([0, int(t_v7[0].item()), int(t_v7[1].item()) + 1, 0], dtype=torch.int32, device=device)
        pos_q = torch.tensor([5, 6, 7, 8], dtype=torch.int64, device=device)
        lidx_q = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device=device)
        imap_q = torch.tensor([0], dtype=torch.int32, device=device)
        pfl_q = torch.tensor([4], dtype=torch.int32, device=device)
        cu_q = torch.tensor([0, 4], dtype=torch.int32, device=device)
        k["fused_reject"][(1,)](
            g_r, M_q, s_ref, 4, ns_r, nr_r, sl_q, ids_q, pos_q, lidx_q, imap_q, pfl_q, cu_q,
            TP=2, SPLIT=1, NP=2, PDL=False, num_warps=1,
        )
        rc3 = lib.sync_free_fused_reject_v7_fwd(
            g_r.data_ptr(), M_q, 1, 2, 1, s_v7.data_ptr(), 4, ns_c.data_ptr(), nr_c.data_ptr(),
            sl_q.data_ptr(), ids_q.data_ptr(), pos_q.data_ptr(), lidx_q.data_ptr(),
            imap_q.data_ptr(), pfl_q.data_ptr(), cu_q.data_ptr(),
            2, 0, stream,  # bit1: pos_i64
        )
        torch.cuda.synchronize(device)
        ok = (
            rc1 == 0 and rc2 == 0 and rc3 == 0
            and bool(torch.equal(t_v7, t_ref))
            and bool(torch.equal(ns_c, ns_r))
            and bool(torch.equal(nr_c, nr_r))
            and bool(torch.equal(s_v7[0, : int(ns_r[0].item())], s_ref[0, : int(ns_r[0].item())]))
        )
    except Exception:
        ok = False
    _SF_V7_QUAL[d] = ok
    logger.info("INKLING_SYNC_FREE_CUDA_V7 qualify dev=%d: ok=%d (SPLIT=1 16B/row AG)", d, int(ok))
    return ok


def _get_sf_temp_v16_lib():
    global _SF_TEMP_V16_LIB, _SF_TEMP_V16_TRIED
    if not _SF_TEMP_V16_TRIED:
        _SF_TEMP_V16_TRIED = True
        if TEMP_V16 and os.path.isfile(TEMP_V16_LIB_PATH):
            try:
                import ctypes
                c_vp = ctypes.c_void_p
                c_i = ctypes.c_int
                c_i64 = ctypes.c_int64
                c_u32 = ctypes.c_uint32
                c_u64 = ctypes.c_uint64
                lib = ctypes.CDLL(TEMP_V16_LIB_PATH)
                sigs = {
                    "sync_free_temp_v16_version": [],
                    "sync_free_temp_v16_smem_cap": [],
                    "sync_free_temp_reject_v16_fwd": [
                        c_vp, c_i, c_i64, c_i, c_i, c_i,
                        c_vp, c_vp, c_vp, c_vp, c_vp, c_u64,
                        c_vp, c_i, c_i, c_i64,
                        c_vp, c_vp, c_vp, c_vp, c_vp, c_vp, c_vp, c_u32,
                        c_vp, c_i64, c_vp, c_vp, c_i, c_u64,
                    ],
                    "sync_free_temp_sample_k0_v16_fwd": [
                        c_vp, c_i, c_i64, c_i, c_i,
                        c_vp, c_vp, c_vp, c_vp, c_vp, c_u64,
                        c_vp, c_vp, c_vp, c_u32,
                        c_vp, c_vp, c_vp, c_i, c_u64,
                    ],
                    "sync_free_temp_verify_rows_v16_fwd": [
                        c_vp, c_i, c_i64, c_i, c_i, c_i, c_i,
                        c_vp, c_vp, c_vp, c_vp, c_vp, c_vp, c_u64,
                        c_vp, c_vp, c_i, c_i, c_i64,
                        c_vp, c_vp, c_vp, c_i, c_u64,
                    ],
                }
                for sym, argtypes in sigs.items():
                    fn = getattr(lib, sym)
                    fn.argtypes = argtypes
                    fn.restype = c_i
                if int(lib.sync_free_temp_v16_version()) == 16:
                    _SF_TEMP_V16_LIB = lib
                    logger.info(
                        "INKLING_SYNC_FREE_TEMP_V16=1: loaded %s (smem_cap=%d, zero-sync SM121a temp>0 sampler)",
                        TEMP_V16_LIB_PATH, int(lib.sync_free_temp_v16_smem_cap()),
                    )
            except (OSError, AttributeError):
                _SF_TEMP_V16_LIB = None
    return _SF_TEMP_V16_LIB


@torch.no_grad()
def qualify_sf_temp_v16(device: torch.device) -> bool:
    """Fail-closed startup GPU qualification for Wave-V16 temp > 0 & mixed rejection sampler."""
    d = _sf_dev_idx(device)
    if d in _SF_TEMP_V16_QUAL:
        return _SF_TEMP_V16_QUAL[d]
    lib = _get_sf_temp_v16_lib()
    if lib is None or device.type != "cuda" or _capturing():
        _SF_TEMP_V16_QUAL[d] = False
        return False
    ok = False
    try:
        stream = int(torch.cuda.current_stream(device).cuda_stream)
        V_q = 256
        lg = torch.linspace(-2.0, 2.0, 4 * V_q, dtype=torch.float32, device=device).view(4, V_q).contiguous()
        lg[0, 42] = 10.0
        lg[1, 13] = 10.0
        temps = torch.tensor([0.0, 0.7], dtype=torch.float32, device=device)
        top_p = torch.tensor([1.0, 0.95], dtype=torch.float32, device=device)
        top_k = torch.tensor([0, 64], dtype=torch.int32, device=device)
        min_p = torch.tensor([0.0, 0.05], dtype=torch.float32, device=device)
        seeds = torch.tensor([12345, 67890], dtype=torch.int64, device=device)
        bpos = torch.tensor([10, 20], dtype=torch.int64, device=device)
        drafts = torch.tensor([[42], [100]], dtype=torch.int64, device=device)
        out1 = torch.zeros((2, 2), dtype=torch.int64, device=device)
        ns1 = torch.zeros((2,), dtype=torch.int32, device=device)
        nr1 = torch.zeros((2,), dtype=torch.int32, device=device)
        out2 = torch.zeros_like(out1)
        ns2 = torch.zeros_like(ns1)
        nr2 = torch.zeros_like(nr1)
        rc1 = lib.sync_free_temp_verify_rows_v16_fwd(
            lg.data_ptr(), 1, V_q, 4, V_q, 2, 1,
            temps.data_ptr(), top_p.data_ptr(), top_k.data_ptr(), min_p.data_ptr(),
            seeds.data_ptr(), bpos.data_ptr(), 1,
            drafts.data_ptr(), 0, 0, 0, 0,
            out1.data_ptr(), ns1.data_ptr(), nr1.data_ptr(), 0, stream,
        )
        rc2 = lib.sync_free_temp_verify_rows_v16_fwd(
            lg.data_ptr(), 1, V_q, 4, V_q, 2, 1,
            temps.data_ptr(), top_p.data_ptr(), top_k.data_ptr(), min_p.data_ptr(),
            seeds.data_ptr(), bpos.data_ptr(), 1,
            drafts.data_ptr(), 0, 0, 0, 0,
            out2.data_ptr(), ns2.data_ptr(), nr2.data_ptr(), 0, stream,
        )
        torch.cuda.synchronize(device)
        ok = (
            rc1 == 0
            and rc2 == 0
            and int(ns1[0].item()) == 2
            and int(nr1[0].item()) == 0
            and int(out1[0, 0].item()) == 42
            and int(out1[0, 1].item()) == 13
            and 0 <= int(out1[1, 0].item()) < V_q
            and bool(torch.equal(out1, out2))
            and bool(torch.equal(ns1, ns2))
            and bool(torch.equal(nr1, nr2))
        )
    except Exception:
        ok = False
    _SF_TEMP_V16_QUAL[d] = ok
    logger.info("INKLING_SYNC_FREE_TEMP_V16 qualify dev=%d: ok=%d", d, int(ok))
    return ok


def _sf_temp_v16_ok(device: torch.device) -> bool:
    if not TEMP_V16 or device.type != "cuda":
        return False
    d = _sf_dev_idx(device)
    q = _SF_TEMP_V16_QUAL.get(d)
    if q is None:
        if _capturing():
            return False
        q = qualify_sf_temp_v16(device)
    return bool(q and _get_sf_temp_v16_lib() is not None)


def _notify_cert_temp_direct(
    active: bool, shared_prefix_n: int = 1, update_l2pf_hint: bool = True
) -> None:
    """Zero-sync notification to cert_head & cert_head_cuda_v7 on temp > 0 / greedy transitions."""
    global _CACHED_CERT_HEAD_MOD, _CACHED_CERT_V7_MOD, _CACHED_CERT_PROBED
    if not _CACHED_CERT_PROBED:
        _CACHED_CERT_PROBED = True
        try:
            from . import cert_head as _ch_m
            _CACHED_CERT_HEAD_MOD = _ch_m
        except Exception:
            try:
                import cert_head as _ch_m  # type: ignore[no-redef]
                _CACHED_CERT_HEAD_MOD = _ch_m
            except Exception:
                _CACHED_CERT_HEAD_MOD = None
        try:
            from . import cert_head_cuda_v7 as _ch7_m
            _CACHED_CERT_V7_MOD = _ch7_m
        except Exception:
            try:
                import cert_head_cuda_v7 as _ch7_m  # type: ignore[no-redef]
                _CACHED_CERT_V7_MOD = _ch7_m
            except Exception:
                _CACHED_CERT_V7_MOD = None
    if _CACHED_CERT_HEAD_MOD is not None and hasattr(_CACHED_CERT_HEAD_MOD, "set_temp_direct_step"):
        _CACHED_CERT_HEAD_MOD.set_temp_direct_step(active, shared_prefix_n=shared_prefix_n)
    if _CACHED_CERT_V7_MOD is not None and hasattr(_CACHED_CERT_V7_MOD, "set_temp_direct_step"):
        _CACHED_CERT_V7_MOD.set_temp_direct_step(
            active, shared_prefix_n=shared_prefix_n, update_l2pf_hint=update_l2pf_hint
        )


def _detect_shared_prefix_n(input_batch) -> int:
    """Zero-sync CPU metadata probe for shared-prefix n>1 fanout (e.g. rollout_harvester --n 4)."""
    if input_batch is None:
        return 1
    sp_n = getattr(input_batch, "shared_prefix_group_size", None)
    if sp_n is not None:
        try:
            return max(1, int(sp_n))
        except Exception:
            return 1
    p_hashes = getattr(input_batch, "prompt_prefix_hash_np", None)
    if p_hashes is not None:
        try:
            n_tot = len(p_hashes)
            if n_tot > 1:
                counts: dict[int, int] = {}
                for h in p_hashes:
                    hi = int(h)
                    counts[hi] = counts.get(hi, 0) + 1
                return max(counts.values()) if counts else 1
        except Exception:
            return 1
    return 1


def _validate_temp_v16_sampling_states(ss, idx) -> bool:
    """Zero-sync CPU sanity validation of temp > 0 sampling parameters (fail-closed on NaN/invalid)."""
    if getattr(ss, "temperature", None) is None or getattr(ss.temperature, "gpu", None) is None:
        return False
    tp_obj = getattr(ss, "top_p", None)
    if tp_obj is not None and getattr(tp_obj, "np", None) is not None:
        tp_np = tp_obj.np
        for s in idx:
            v = float(tp_np[int(s)])
            if not np.isfinite(v) or v <= 0.0 or v > 1.0:
                return False
    mp_obj = getattr(ss, "min_p", None)
    if mp_obj is not None and getattr(mp_obj, "np", None) is not None:
        mp_np = mp_obj.np
        for s in idx:
            v = float(mp_np[int(s)])
            if not np.isfinite(v) or v < 0.0 or v >= 1.0:
                return False
    tk_obj = getattr(ss, "top_k", None)
    if tk_obj is not None and getattr(tk_obj, "np", None) is not None:
        tk_np = tk_obj.np
        for s in idx:
            if int(tk_np[int(s)]) < 0:
                return False
    return True


def _extract_temp_v16_ptrs(ss):
    """Extract GPU device pointers for per-slot sampling parameters (0 host syncs)."""
    t_gpu = ss.temperature.gpu
    if t_gpu is None or t_gpu.dtype != torch.float32:
        return None
    tp_gpu = getattr(getattr(ss, "top_p", None), "gpu", None)
    tk_gpu = getattr(getattr(ss, "top_k", None), "gpu", None)
    mp_gpu = getattr(getattr(ss, "min_p", None), "gpu", None)
    sd_gpu = getattr(getattr(ss, "seeds", None), "gpu", None)
    if tp_gpu is not None and tp_gpu.dtype != torch.float32:
        return None
    if tk_gpu is not None and tk_gpu.dtype != torch.int32:
        return None
    if mp_gpu is not None and mp_gpu.dtype != torch.float32:
        return None
    seeds_i64 = False
    if sd_gpu is not None:
        if sd_gpu.dtype == torch.int64:
            seeds_i64 = True
        elif sd_gpu.dtype == torch.int32:
            seeds_i64 = False
        else:
            return None
    return (
        int(t_gpu.data_ptr()),
        int(tp_gpu.data_ptr()) if tp_gpu is not None else 0,
        int(tk_gpu.data_ptr()) if tk_gpu is not None else 0,
        int(mp_gpu.data_ptr()) if mp_gpu is not None else 0,
        int(sd_gpu.data_ptr()) if sd_gpu is not None else 0,
        seeds_i64,
    )


def _try_temp_v16_sample_k0(self_sp, logits: torch.Tensor, input_batch):
    """Zero-sync SM121a K=0 GPU sampler for temp > 0 & mixed batches (returns SamplerOutput or None)."""
    if (
        not TEMP_V16
        or _capturing()
        or not isinstance(logits, torch.Tensor)
        or logits.dim() != 2
        or logits.stride(1) != 1
        or logits.dtype not in (torch.bfloat16, torch.float32)
    ):
        return None
    M, V = int(logits.shape[0]), int(logits.shape[1])
    num_reqs = int(getattr(input_batch, "num_reqs", 0) or 0)
    if num_reqs <= 0 or M != num_reqs or M > TEMP_V16_MAX_M or V <= 0:
        return None
    if not _sf_temp_v16_ok(logits.device):
        return None
    ib_sl = input_batch.seq_lens
    ib_idx = input_batch.idx_mapping
    pf_gpu = self_sp.req_states.prefill_len.gpu
    if (
        ib_sl.dtype != torch.int32
        or pf_gpu.dtype != torch.int32
        or ib_idx.dtype not in (torch.int32, torch.int64)
    ):
        return None
    ptrs = _extract_temp_v16_ptrs(self_sp.sampling_states)
    if ptrs is None:
        return None
    t_ptr, tp_ptr, tk_ptr, mp_ptr, sd_ptr, seeds_i64 = ptrs
    dflags = (8 if ib_idx.dtype == torch.int64 else 0) | (16 if seeds_i64 else 0)
    tok = torch.empty((num_reqs,), dtype=torch.int64, device=logits.device)
    ns = torch.empty((num_reqs,), dtype=torch.int32, device=logits.device)
    nr = torch.empty((num_reqs,), dtype=torch.int32, device=logits.device)
    lib = _get_sf_temp_v16_lib()
    rc = lib.sync_free_temp_sample_k0_v16_fwd(
        int(logits.data_ptr()),
        1 if logits.dtype == torch.float32 else 0,
        int(logits.stride(0)),
        num_reqs,
        V,
        t_ptr, tp_ptr, tk_ptr, mp_ptr, sd_ptr,
        int(_S.get("temp_step_id", 0)),
        int(ib_sl.data_ptr()),
        int(ib_idx.data_ptr()),
        int(pf_gpu.data_ptr()),
        int(dflags),
        int(tok.data_ptr()),
        int(ns.data_ptr()),
        int(nr.data_ptr()),
        1 if TEMP_V16_PDL else 0,
        int(torch.cuda.current_stream(logits.device).cuda_stream),
    )
    if rc != 0:
        return None
    from vllm.v1.worker.gpu.sample.output import SamplerOutput
    return SamplerOutput(
        sampled_token_ids=tok.view(-1, 1),
        logprobs_tensors=None,
        num_nans=None,
        num_sampled=ns,
        num_rejected=nr,
        sampling_mask_tensors=None,
    )


def _try_temp_v16_reject(self_rs, logits: torch.Tensor, input_batch, draft_logits=None):
    """Zero-sync SM121a speculative rejection sampler for temp > 0 & mixed batches (0 intermediate [M,V] allocs)."""
    if (
        not TEMP_V16
        or _capturing()
        or not isinstance(logits, torch.Tensor)
        or logits.dim() != 2
        or logits.stride(1) != 1
        or logits.dtype not in (torch.bfloat16, torch.float32)
    ):
        return None
    M, V = int(logits.shape[0]), int(logits.shape[1])
    num_reqs = int(getattr(input_batch, "num_reqs", 0) or 0)
    if num_reqs <= 0 or M <= 0 or M > TEMP_V16_MAX_M or V <= 0:
        return None
    if not _sf_temp_v16_ok(logits.device):
        return None
    ib_pos = input_batch.positions
    ib_li = input_batch.logits_indices
    ib_ids = input_batch.input_ids
    ib_sl = input_batch.seq_lens
    ib_cu = input_batch.cu_num_logits
    ib_idx = input_batch.idx_mapping
    pf_gpu = self_rs.sampler.req_states.prefill_len.gpu
    if (
        ib_pos.dim() != 1
        or ib_li.shape[0] != M
        or ib_ids.stride(0) != 1
        or ib_pos.stride(0) != 1
        or ib_li.stride(0) != 1
        or ib_sl.dtype != torch.int32
        or pf_gpu.dtype != torch.int32
        or ib_cu.dtype != torch.int32
        or ib_ids.dtype not in (torch.int32, torch.int64)
        or ib_pos.dtype not in (torch.int32, torch.int64)
        or ib_li.dtype not in (torch.int32, torch.int64)
        or ib_idx.dtype not in (torch.int32, torch.int64)
    ):
        return None
    ptrs = _extract_temp_v16_ptrs(self_rs.sampler.sampling_states)
    if ptrs is None:
        return None
    t_ptr, tp_ptr, tk_ptr, mp_ptr, sd_ptr, seeds_i64 = ptrs

    draft_mode = 0
    draft_q_ptr = 0
    draft_is_fp32 = 0
    draft_stride = 0
    if draft_logits is not None:
        if (
            isinstance(draft_logits, torch.Tensor)
            and draft_logits.dim() == 2
            and draft_logits.shape == logits.shape
            and draft_logits.stride(1) == 1
            and draft_logits.dtype in (torch.bfloat16, torch.float32)
        ):
            draft_mode = 2
            draft_q_ptr = int(draft_logits.data_ptr())
            draft_is_fp32 = 1 if draft_logits.dtype == torch.float32 else 0
            draft_stride = int(draft_logits.stride(0))
        elif (
            isinstance(draft_logits, torch.Tensor)
            and draft_logits.dim() == 1
            and draft_logits.shape[0] == M
            and draft_logits.dtype == torch.float32
            and draft_logits.is_contiguous()
        ):
            draft_mode = 3
            draft_q_ptr = int(draft_logits.data_ptr())
        else:
            return None

    cols = int(self_rs.num_speculative_steps) + 1
    _ring_out = _acquire_static_ring_outputs(logits.device, num_reqs, cols)
    if _ring_out is not None:
        _, sampled, num_sampled, num_rejected = _ring_out
    else:
        sampled = torch.empty((num_reqs, cols), dtype=torch.int64, device=logits.device)
        num_sampled = torch.empty((num_reqs,), dtype=torch.int32, device=logits.device)
        num_rejected = torch.empty((num_reqs,), dtype=torch.int32, device=logits.device)

    dflags = (
        (1 if ib_ids.dtype == torch.int64 else 0)
        | (2 if ib_pos.dtype == torch.int64 else 0)
        | (4 if ib_li.dtype == torch.int64 else 0)
        | (8 if ib_idx.dtype == torch.int64 else 0)
        | (16 if seeds_i64 else 0)
    )
    lib = _get_sf_temp_v16_lib()
    rc = lib.sync_free_temp_reject_v16_fwd(
        int(logits.data_ptr()),
        1 if logits.dtype == torch.float32 else 0,
        int(logits.stride(0)),
        M,
        V,
        num_reqs,
        t_ptr, tp_ptr, tk_ptr, mp_ptr, sd_ptr,
        int(_S.get("temp_step_id", 0)),
        draft_q_ptr,
        draft_mode,
        draft_is_fp32,
        draft_stride,
        int(ib_sl.data_ptr()),
        int(ib_ids.data_ptr()),
        int(ib_pos.data_ptr()),
        int(ib_li.data_ptr()),
        int(ib_idx.data_ptr()),
        int(pf_gpu.data_ptr()),
        int(ib_cu.data_ptr()),
        int(dflags),
        int(sampled.data_ptr()),
        int(sampled.stride(0)),
        int(num_sampled.data_ptr()),
        int(num_rejected.data_ptr()),
        1 if TEMP_V16_PDL else 0,
        int(torch.cuda.current_stream(logits.device).cuda_stream),
    )
    if rc != 0:
        clear_static_ring_state()
        return None
    from vllm.v1.worker.gpu.sample.output import SamplerOutput
    return SamplerOutput(
        sampled_token_ids=sampled,
        logprobs_tensors=None,
        num_nans=None,
        num_sampled=num_sampled,
        num_rejected=num_rejected,
        sampling_mask_tensors=None,
    )


_S: dict = {
    "armed": False,
    "armed_k0": False,
    "temp_mode": "off",
    "temp_step_id": 0,
    "temp_shared_prefix_n": 1,
    "num_tokens": 0,
    "installed": False,
    "runner_hooked": False,
    "tok": None,
    "gathered": None,
    "rs_hooked": False,
    "sp_hooked": False,
}


def _capturing() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


# =====================================================================================================
# Kernels
# =====================================================================================================
_K: dict = {}


def _kernels():
    if _K:
        return _K
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice as tldevice

    @triton.jit
    def _rn_bf16(x):
        bits = x.to(tl.uint32, bitcast=True)
        bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
        return bits.to(tl.float32, bitcast=True)

    @triton.jit
    def _chain(x, inv_cap, cap):
        y = _rn_bf16(x * inv_cap)
        t = tldevice.tanh(y)
        y = _rn_bf16(tl.where(tl.abs(y) < 1.1754943508222875e-38, y, t))
        return _rn_bf16(y * cap)

    @triton.jit(do_not_specialize=["n"])
    def _chain_kernel(X, Y, n, inv_cap, cap, BLOCK: tl.constexpr):
        cols = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = cols < n
        x = tl.load(X + cols, mask=m, other=0.0).to(tl.float32)
        tl.store(Y + cols, _chain(x, inv_cap, cap).to(tl.bfloat16), mask=m)

    @triton.jit(do_not_specialize=["sl", "valid", "vstart", "seg"])
    def _local_kernel(LG, sl, OUT, valid, vstart, seg, inv_cap, cap,
                      CAP: tl.constexpr, SPLIT: tl.constexpr, BLOCK: tl.constexpr,
                      FP32_CHAIN: tl.constexpr = False, DIVMODE: tl.constexpr = 0):
        row = tl.program_id(0)
        sp = tl.program_id(1)
        cols = tl.arange(0, BLOCK)
        lo = sp * seg
        hi = tl.minimum(lo + seg, valid)
        best_v = tl.full((), float("-inf"), tl.float32)
        best_i = lo
        for v0 in range(lo, hi, BLOCK):
            c = v0 + cols
            m = c < hi
            x = tl.load(LG + row.to(tl.int64) * sl + c, mask=m, other=0.0).to(tl.float32)
            bad = (x != x) | (~m)
            if CAP:
                if FP32_CHAIN:
                    y = x * inv_cap if DIVMODE == 0 else tldevice.div_rn(x, cap)
                    t = tldevice.tanh(y)
                    t = tl.where(tl.abs(y) < 1.1754943508222875e-38, y, t)
                    x = t * cap
                else:
                    x = _chain(x, inv_cap, cap)
            x = tl.where(bad, float("-inf"), x)
            v, i = tl.max(x, axis=0, return_indices=True, return_indices_tie_break_left=True)
            upd = v > best_v
            best_i = tl.where(upd, v0 + i, best_i)
            best_v = tl.where(upd, v, best_v)
        base = OUT + (row * SPLIT + sp) * 4
        tl.store(base, best_v)
        tl.store(base + 1, (vstart + best_i).to(tl.float32))
        tl.store(base + 2, 0.0)
        tl.store(base + 3, 0.0)

    @triton.jit(do_not_specialize=["M", "so", "V"])
    def _onehot_kernel(G, M, TOK, OUT, so, V, TP: tl.constexpr, SPLIT: tl.constexpr, NP: tl.constexpr,
                       BLOCK: tl.constexpr):
        row = tl.program_id(0)
        blk = tl.program_id(1)
        j = tl.arange(0, NP)
        r = j // SPLIT
        sidx = j - r * SPLIT
        ok = j < TP * SPLIT
        gi = ((r * M + row) * SPLIT + sidx) * 4
        v = tl.load(G + gi, mask=ok, other=float("-inf"))
        _, w = tl.max(v, axis=0, return_indices=True, return_indices_tie_break_left=True)
        best_t = tl.load(G + ((w // SPLIT) * M + row) * SPLIT * 4 + (w % SPLIT) * 4 + 1)
        tok = best_t.to(tl.int32)
        cols = blk * BLOCK + tl.arange(0, BLOCK)
        val = tl.where(cols == tok, 0.0, float("-inf")).to(OUT.dtype.element_ty)
        tl.store(OUT + row * so + cols, val, mask=cols < V)
        if blk == 0:
            tl.store(TOK + row, tok.to(tl.int64))

    @triton.jit(do_not_specialize=["M"])
    def _tok_only_kernel(
        G, M, TOK, TP: tl.constexpr, SPLIT: tl.constexpr, NP: tl.constexpr,
        PDL: tl.constexpr = False,
    ):
        if PDL:
            tl.extra.cuda.gdc_wait()
            tl.extra.cuda.gdc_launch_dependents()
        row = tl.program_id(0)
        j = tl.arange(0, NP)
        r = j // SPLIT
        sidx = j - r * SPLIT
        ok = j < TP * SPLIT
        gi = ((r * M + row) * SPLIT + sidx) * 4
        v = tl.load(G + gi, mask=ok, other=float("-inf"))
        _, w = tl.max(v, axis=0, return_indices=True, return_indices_tie_break_left=True)
        best_t = tl.load(G + ((w // SPLIT) * M + row) * SPLIT * 4 + (w % SPLIT) * 4 + 1)
        tl.store(TOK + row, best_t.to(tl.int32).to(tl.int64))

    @triton.jit
    def _direct_greedy_reject_kernel(
        sampled_ptr,
        sampled_stride,
        num_sampled_ptr,
        tok_ptr,
        draft_sampled_ptr,
        cu_num_logits_ptr,
    ):
        req_idx = tl.program_id(0)
        start_idx = tl.load(cu_num_logits_ptr + req_idx).to(tl.int64)
        end_idx = tl.load(cu_num_logits_ptr + req_idx + 1).to(tl.int64)
        num_draft_tokens = end_idx - start_idx - 1
        accepted_length = tl.zeros((), tl.int64)
        verifying = True
        for i in range(num_draft_tokens):
            logit_idx = start_idx + i
            draft_tok = tl.load(draft_sampled_ptr + logit_idx + 1).to(tl.int64)
            is_valid_draft = draft_tok >= 0
            if verifying:
                target_tok = tl.load(tok_ptr + logit_idx).to(tl.int64)
                accepted = (target_tok == draft_tok) & is_valid_draft
                verifying = accepted
                accepted_length += accepted
                tl.store(
                    sampled_ptr + req_idx * sampled_stride + i,
                    tl.where(accepted, draft_tok, target_tok),
                )
        if accepted_length == num_draft_tokens:
            bonus_tok = tl.load(tok_ptr + start_idx + num_draft_tokens).to(tl.int64)
            tl.store(sampled_ptr + req_idx * sampled_stride + num_draft_tokens, bonus_tok)
        tl.store(num_sampled_ptr + req_idx, (accepted_length + 1).to(tl.int32))

    @triton.jit(do_not_specialize=["M"])
    def _fused_tok_gather_reject_kernel(
        G,
        M,
        sampled_ptr,
        sampled_stride,
        num_sampled_ptr,
        num_rejected_ptr,
        seq_lens_ptr,
        input_ids_ptr,
        positions_ptr,
        logits_indices_ptr,
        idx_mapping_ptr,
        prefill_len_ptr,
        cu_num_logits_ptr,
        TP: tl.constexpr,
        SPLIT: tl.constexpr,
        NP: tl.constexpr,
        PDL: tl.constexpr = True,
    ):
        """Phase 2A 4->1 fusion: _tok_only + gather_draft_sampled + _direct_greedy_reject +
        _get_num_sampled_and_rejected. PDL-chained behind cert_head RDMA AG (gdc_wait before
        reading G), computes global argmax inline ONLY for verified rows (exact same
        tie-break-left as _tok_only_kernel), reads draft tokens directly via
        input_ids[logits_indices[logit_idx + 1]] with prefill_len guard, and writes sampled,
        num_sampled, and num_rejected directly (0-ULP bit-exact, eliminates 3 eager launches)."""
        if PDL:
            tl.extra.cuda.gdc_wait()
        req_idx = tl.program_id(0).to(tl.int64)
        start_idx = tl.load(cu_num_logits_ptr + req_idx).to(tl.int64)
        end_idx = tl.load(cu_num_logits_ptr + req_idx + 1).to(tl.int64)
        num_draft_tokens = end_idx - start_idx - 1
        req_state_idx = tl.load(idx_mapping_ptr + req_idx).to(tl.int64)
        req_prefill_len = tl.load(prefill_len_ptr + req_state_idx).to(tl.int64)
        seq_len = tl.load(seq_lens_ptr + req_idx).to(tl.int64)

        j = tl.arange(0, NP)
        r = j // SPLIT
        sidx = j - r * SPLIT
        ok = j < TP * SPLIT
        m64 = M.to(tl.int64)

        accepted_length = tl.zeros((), tl.int64)
        verifying = True
        for i in range(num_draft_tokens):
            logit_idx = start_idx + i
            tok_idx = tl.load(logits_indices_ptr + logit_idx + 1).to(tl.int64)
            draft_tok = tl.load(input_ids_ptr + tok_idx).to(tl.int64)
            draft_pos = tl.load(positions_ptr + tok_idx).to(tl.int64)
            is_valid_draft = (draft_tok >= 0) & (draft_pos >= req_prefill_len)
            if verifying:
                gi = ((r.to(tl.int64) * m64 + logit_idx) * SPLIT + sidx) * 4
                v = tl.load(G + gi, mask=ok, other=float("-inf"))
                _, w = tl.max(v, axis=0, return_indices=True, return_indices_tie_break_left=True)
                best_t = tl.load(
                    G + ((w // SPLIT).to(tl.int64) * m64 + logit_idx) * (SPLIT * 4) + (w % SPLIT) * 4 + 1
                )
                target_tok = best_t.to(tl.int32).to(tl.int64)
                accepted = (target_tok == draft_tok) & is_valid_draft
                verifying = accepted
                accepted_length += accepted
                tl.store(
                    sampled_ptr + req_idx * sampled_stride + i,
                    tl.where(accepted, draft_tok, target_tok),
                )
        if accepted_length == num_draft_tokens:
            bonus_row = start_idx + num_draft_tokens
            gi = ((r.to(tl.int64) * m64 + bonus_row) * SPLIT + sidx) * 4
            v = tl.load(G + gi, mask=ok, other=float("-inf"))
            _, w = tl.max(v, axis=0, return_indices=True, return_indices_tie_break_left=True)
            best_t = tl.load(
                G + ((w // SPLIT).to(tl.int64) * m64 + bonus_row) * (SPLIT * 4) + (w % SPLIT) * 4 + 1
            )
            bonus_tok = best_t.to(tl.int32).to(tl.int64)
            tl.store(sampled_ptr + req_idx * sampled_stride + num_draft_tokens, bonus_tok)
        incomplete = seq_len < req_prefill_len
        ns_out = tl.where(incomplete, 0, accepted_length + 1).to(tl.int32)
        nr_out = tl.where(incomplete, 0, num_draft_tokens - accepted_length).to(tl.int32)
        if PDL:
            tl.extra.cuda.gdc_launch_dependents()
        tl.store(num_sampled_ptr + req_idx, ns_out)
        tl.store(num_rejected_ptr + req_idx, nr_out)

    @triton.jit(do_not_specialize=["M"])
    def _fused_tree_reject_kernel(
        G,
        M,
        sampled_ptr,
        sampled_stride,
        num_sampled_ptr,
        num_rejected_ptr,
        seq_lens_ptr,
        input_ids_ptr,
        positions_ptr,
        logits_indices_ptr,
        idx_mapping_ptr,
        prefill_len_ptr,
        cu_num_logits_ptr,
        tree_parent_ptr,
        n_live_ptr,
        accepted_path_ptr,
        accepted_len_ptr,
        needs_compact_ptr,
        TP: tl.constexpr,
        SPLIT: tl.constexpr,
        NP: tl.constexpr,
        MAX_M: tl.constexpr = 8,
        PDL: tl.constexpr = True,
    ):
        """0-ULP Tree Greedy Rejection Sampler (C=1).
        Walks from root cur=0, computes global argmax ONLY for visited nodes along the accepted
        path, matches live children c with TreeParent[c]==cur against target_tok, and emits
        (sampled, num_sampled, num_rejected, accepted_path, accepted_len, needs_compact)
        directly into PDL-chained `tree_compact_caches_fwd`!"""
        if PDL:
            tl.extra.cuda.gdc_wait()
        req_idx = tl.program_id(0).to(tl.int64)
        start_idx = tl.load(cu_num_logits_ptr + req_idx).to(tl.int64)
        end_idx = tl.load(cu_num_logits_ptr + req_idx + 1).to(tl.int64)
        num_draft_tokens = end_idx - start_idx - 1
        req_state_idx = tl.load(idx_mapping_ptr + req_idx).to(tl.int64)
        req_prefill_len = tl.load(prefill_len_ptr + req_state_idx).to(tl.int64)
        seq_len = tl.load(seq_lens_ptr + req_idx).to(tl.int64)
        n_live = tl.minimum(tl.load(n_live_ptr).to(tl.int64), MAX_M)

        j = tl.arange(0, NP)
        r = j // SPLIT
        sidx = j - r * SPLIT
        ok = j < TP * SPLIT
        m64 = M.to(tl.int64)

        cur = tl.zeros((), tl.int64)
        depth = tl.zeros((), tl.int64)
        needs_cmp = tl.zeros((), tl.int32)
        tl.store(accepted_path_ptr + 0, 0)
        walking = True

        for _step in tl.static_range(MAX_M):
            if walking:
                logit_row = start_idx + cur
                gi = ((r.to(tl.int64) * m64 + logit_row) * SPLIT + sidx) * 4
                v = tl.load(G + gi, mask=ok, other=float("-inf"))
                _, w = tl.max(v, axis=0, return_indices=True, return_indices_tie_break_left=True)
                best_t = tl.load(
                    G + ((w // SPLIT).to(tl.int64) * m64 + logit_row) * (SPLIT * 4) + (w % SPLIT) * 4 + 1
                )
                target_tok = best_t.to(tl.int32).to(tl.int64)
                tl.store(sampled_ptr + req_idx * sampled_stride + depth, target_tok)

                matched_c = tl.full((), -1, tl.int64)
                for c in tl.static_range(1, MAX_M):
                    if (c < n_live) & (matched_c < 0):
                        par = tl.load(tree_parent_ptr + c).to(tl.int64)
                        if par == cur:
                            tok_idx = tl.load(logits_indices_ptr + start_idx + c).to(tl.int64)
                            dtok = tl.load(input_ids_ptr + tok_idx).to(tl.int64)
                            dpos = tl.load(positions_ptr + tok_idx).to(tl.int64)
                            if (dtok == target_tok) & (dtok >= 0) & (dpos >= req_prefill_len):
                                matched_c = tl.full((), c, tl.int64)
                if matched_c >= 0:
                    depth = depth + 1
                    tl.store(accepted_path_ptr + depth, matched_c.to(tl.int32))
                    needs_cmp = tl.where(matched_c != depth, 1, needs_cmp)
                    cur = matched_c
                else:
                    walking = False

        incomplete = seq_len < req_prefill_len
        ns_out = tl.where(incomplete, 0, depth + 1).to(tl.int32)
        nr_out = tl.where(incomplete, 0, num_draft_tokens - depth).to(tl.int32)
        if PDL:
            tl.extra.cuda.gdc_launch_dependents()
        tl.store(num_sampled_ptr + req_idx, ns_out)
        tl.store(num_rejected_ptr + req_idx, nr_out)
        tl.store(accepted_len_ptr, ns_out)
        tl.store(needs_compact_ptr, tl.where(incomplete, 0, needs_cmp))


    _K.update(triton=triton, chain=_chain_kernel, local=_local_kernel, onehot=_onehot_kernel,
              tok_only=_tok_only_kernel, direct_reject=_direct_greedy_reject_kernel,
              fused_reject=_fused_tok_gather_reject_kernel,
              fused_tree_reject=_fused_tree_reject_kernel)
    return _K


def _f32(x: float) -> float:
    return float(np.float32(x))


def inv_cap_of(cap: float) -> float:
    return float(np.float32(1.0) / np.float32(cap))


def chain_selftest(cap: float, device) -> tuple[bool, int]:
    k = _kernels()
    bits = torch.arange(0, 65536, dtype=torch.int32, device=device).to(torch.int16)
    x = bits.view(torch.bfloat16)
    ref = torch.tanh(x / cap) * cap
    y = torch.empty_like(x)
    k["chain"][(k["triton"].cdiv(x.numel(), 1024),)](x, y, x.numel(), inv_cap_of(cap), _f32(cap), BLOCK=1024)
    ok_mask = ~torch.isnan(x)
    diff = (ref.view(torch.int16) != y.view(torch.int16)) & ok_mask
    return int(diff.sum().item()) == 0, int(diff.sum().item())


def torch_chain(x: torch.Tensor, cap: float | None) -> torch.Tensor:
    if cap is None:
        return x
    return torch.tanh(x / cap) * cap


def sf_local_pairs(local: torch.Tensor, valid: int, vstart: int, cap: float | None = None,
                   split: int = 1, fuse: bool = True, fp32_divmode: int = -1) -> torch.Tensor:
    assert local.dim() == 2 and local.stride(1) == 1 and local.dtype in (torch.bfloat16, torch.float32)
    fp32_chain = False
    if local.dtype != torch.bfloat16:
        if FP32_FUSE and cap is not None and fp32_divmode in (0, 1):
            fp32_chain = True
            fuse = True
        else:
            fuse = False
    assert 0 < valid <= local.shape[1] and split >= 1
    k = _kernels()
    M = local.shape[0]
    if cap is not None and not fuse:
        local = torch_chain(local, cap).contiguous()
        cap = None
    if _sf_v7_ok(local.device):
        ws = _SF_V7_WS[_sf_dev_idx(local.device)]
        if M <= ws["max_m"]:
            lib = _get_sf_v7_lib()
            v7_splits = 16 if M <= 6 else (8 if M <= 127 else 4)
            seg_v7 = ((valid + v7_splits - 1) // v7_splits + 7) & ~7
            idx = ws["idx"]
            ws["idx"] = idx ^ 1
            pair_v7 = ws["pair_out"][idx][:M]
            sem_v7 = ws["sem"][idx]
            rc = lib.sync_free_local_pairs_v7_fwd(
                local.data_ptr(),
                1 if local.dtype == torch.float32 else 0,
                int(local.stride(0)),
                ws["rec_split"].data_ptr(),
                pair_v7.data_ptr(),
                sem_v7.data_ptr(),
                int(M),
                int(v7_splits),
                int(valid),
                int(vstart),
                int(seg_v7),
                inv_cap_of(cap) if cap is not None else 1.0,
                _f32(cap) if cap is not None else 1.0,
                1 if cap is not None else 0,
                1 if fp32_chain else 0,
                int(fp32_divmode if fp32_chain else 0),
                1,
                0,
                int(torch.cuda.current_stream(local.device).cuda_stream),
            )
            if rc == 0:
                return pair_v7
    seg = SF_BLOCK * k["triton"].cdiv(k["triton"].cdiv(valid, SF_BLOCK), split)
    pair = torch.empty((M * split, 4), dtype=torch.float32, device=local.device)
    k["local"][(M, split)](local, local.stride(0), pair, valid, vstart, seg,
                           inv_cap_of(cap) if cap is not None else 1.0, _f32(cap) if cap is not None else 1.0,
                           CAP=cap is not None, SPLIT=split, BLOCK=SF_BLOCK,
                           FP32_CHAIN=fp32_chain, DIVMODE=fp32_divmode if fp32_chain else 0,
                           num_warps=8)
    return pair


def triton_np2(n: int) -> int:
    return 1 << max(0, (int(n) - 1).bit_length())


def sf_tok_only(
    gathered: torch.Tensor, M: int, tp: int, split: int = 1, pdl: bool = False
) -> torch.Tensor:
    k = _kernels()
    assert gathered.shape == (tp * M * split, 4) and gathered.is_contiguous() and gathered.dtype == torch.float32
    tok = torch.empty((M,), dtype=torch.int64, device=gathered.device)
    if _sf_v7_ok(gathered.device):
        lib = _get_sf_v7_lib()
        rc = lib.sync_free_k0_tok_status_v7_fwd(
            gathered.data_ptr(), int(M), int(tp), int(split),
            tok.data_ptr(), 0, 0, 0, 0, 0, 0, 0,
            1 if pdl else 0,
            int(torch.cuda.current_stream(gathered.device).cuda_stream),
        )
        if rc == 0:
            return tok
    k["tok_only"][(M,)](
        gathered, M, tok, TP=tp, SPLIT=split, NP=triton_np2(tp * split),
        PDL=pdl, num_warps=1, launch_pdl=pdl,
    )
    return tok


def sf_onehot(gathered: torch.Tensor, M: int, tp: int, V: int, split: int = 1, dtype=torch.bfloat16):
    k = _kernels()
    assert gathered.shape == (tp * M * split, 4) and gathered.is_contiguous() and gathered.dtype == torch.float32
    out = torch.empty((M, V), dtype=dtype, device=gathered.device)
    tok = torch.empty((M,), dtype=torch.int64, device=gathered.device)
    k["onehot"][(M, k["triton"].cdiv(V, OH_BLOCK))](gathered, M, tok, out, out.stride(0), V,
                                                     TP=tp, SPLIT=split, NP=triton_np2(tp * split), BLOCK=OH_BLOCK, num_warps=4)
    return out, tok


# =====================================================================================================
# Engine integration
# =====================================================================================================
def _tp():
    from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size

    return get_tensor_model_parallel_rank(), get_tensor_model_parallel_world_size()


def _tp_min(vals: list[int]) -> list[int]:
    from vllm.distributed import get_tp_group

    grp = get_tp_group()
    if grp.world_size == 1:
        return vals
    t = torch.tensor(vals, dtype=torch.int64)
    torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MIN, group=grp.cpu_group)
    return [int(v) for v in t.tolist()]


def step_armed(runner, input_batch, grammar_output) -> tuple[bool, bool]:
    """Returns (armed, is_k0).
    - Pure `temp == 0.0` batches arm `_S['temp_mode'] = 'greedy'` (100.000% untouched 1dcfb306 path).
    - When `INKLING_SYNC_FREE_TEMP_V16=1`, pure `temp > 0` and mixed `temp == 0` + `temp > 0` batches
      arm `_S['temp_mode'] = 'temp' | 'mixed'` for zero-sync SM121a GPU rejection sampling, cleanly
      disarming on logprobs, penalties/logits_processing, grammar_output, watermark, and prompt_logprobs."""
    global _CACHED_NO_LOGPROBS
    _S["temp_mode"] = "off"
    _S["temp_shared_prefix_n"] = 1
    if FAST_V14 and not TEMP_V16 and grammar_output is None and input_batch is not None:
        try:
            sp = runner.sampler
            rs = runner.rejection_sampler
            num_draft = input_batch.num_draft_tokens
            if (
                sp is not None
                and rs is not None
                and num_draft > 0
                and getattr(runner, "batch_sharder", None) is None
                and getattr(rs, "watermark_key", None) is None
                and getattr(rs, "synthetic_conditional_rates", None) is None
                and not getattr(rs, "use_block_verification", False)
                and not getattr(rs, "fly_window_size", 0)
                and not getattr(rs, "enable_adaptive_verification", False)
                and not getattr(sp, "compute_nans", False)
                and getattr(sp, "trace_replay_state", None) is None
                and not getattr(sp, "return_sampling_mask", False)
            ):
                # V15 Track 8 LOW guard: disarm speculative prompt_logprobs / expanded logits
                cu_np = getattr(input_batch, "cu_num_logits_np", None)
                if cu_np is not None and len(cu_np) > 0 and int(cu_np[-1]) != int(input_batch.num_reqs) + int(num_draft):
                    return False, False
                idx = input_batch.idx_mapping_np
                if idx is not None:
                    n_idx = len(idx)
                    if 0 < n_idx <= 8:
                        ss = sp.sampling_states
                        t_np = ss.temperature.np
                        nlp_arr = sp.needs_logits_processing
                        if n_idx == 1:
                            i0 = int(idx[0])
                            if t_np[i0] != 0.0 or nlp_arr[i0]:
                                return False, False
                        else:
                            for _i in range(n_idx):
                                _s = int(idx[_i])
                                if t_np[_s] != 0.0 or nlp_arr[_s]:
                                    return False, False
                        no_lp = _CACHED_NO_LOGPROBS
                        if no_lp is None:
                            from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS as _nlp
                            _CACHED_NO_LOGPROBS = _nlp
                            no_lp = _nlp
                        if ss.max_num_logprobs(idx) != no_lp:
                            return False, False
                        lts = getattr(sp, "logprob_token_ids_state", None)
                        if lts is not None and lts.max_num_token_ids(idx) != 0:
                            return False, False
                        _S["temp_mode"] = "greedy"
                        return True, False
        except AttributeError:
            pass
    try:
        if getattr(runner, "batch_sharder", None) is not None:
            return False, False
        if grammar_output is not None or input_batch is None:
            return False, False
        rs, sp = getattr(runner, "rejection_sampler", None), getattr(runner, "sampler", None)
        if sp is None:
            return False, False
        num_draft = int(getattr(input_batch, "num_draft_tokens", 0))
        is_k0 = (num_draft <= 0) or (rs is None)
        cu_np = getattr(input_batch, "cu_num_logits_np", None)
        if is_k0:
            if not ALLOW_K0:
                return False, False
            # Require exactly 1 logit per request (excludes prompt_logprobs / expanded logits)
            if cu_np is None or len(cu_np) == 0 or int(cu_np[-1]) != int(input_batch.num_reqs):
                return False, False
            if getattr(sp, "watermark_key", None) is not None:
                return False, False
        else:
            # V15 Track 8 LOW guard: require exact num_reqs + num_draft logits (excludes speculative prompt_logprobs)
            if cu_np is not None and len(cu_np) > 0 and int(cu_np[-1]) != int(input_batch.num_reqs) + num_draft:
                return False, False
            if getattr(rs, "watermark_key", None) is not None or getattr(rs, "synthetic_conditional_rates", None) is not None:
                return False, False
            if getattr(rs, "use_block_verification", False) or getattr(rs, "fly_window_size", 0):
                return False, False
            if getattr(rs, "enable_adaptive_verification", False):
                return False, False
        if getattr(sp, "compute_nans", False) or getattr(sp, "trace_replay_state", None) is not None:
            return False, False
        if getattr(sp, "return_sampling_mask", False):
            return False, False
        idx = input_batch.idx_mapping_np
        if idx is None or len(idx) == 0:
            return False, False
        ss = sp.sampling_states
        n_idx = len(idx)
        t_np = ss.temperature.np
        nlp_arr = sp.needs_logits_processing
        has_zero = False
        has_pos = False
        if n_idx <= 8 or TEMP_V16:
            for _i in range(n_idx):
                _s = int(idx[_i])
                if nlp_arr[_s]:
                    return False, False
                _tv = float(t_np[_s])
                if not np.isfinite(_tv) or _tv < 0.0:
                    return False, False
                if _tv == 0.0:
                    has_zero = True
                else:
                    has_pos = True
                    if not TEMP_V16:
                        return False, False
        elif FAST_V11 and n_idx == 1:
            i0 = int(idx[0])
            if t_np[i0] != 0.0 or nlp_arr[i0]:
                return False, False
            has_zero = True
        else:
            if not bool(np.all(t_np[idx] == 0.0)):
                return False, False
            if bool(np.any(nlp_arr[idx])):
                return False, False
            has_zero = True

        # Defensive penalties check in addition to needs_logits_processing
        pen_st = getattr(sp, "penalties_state", None)
        if pen_st is not None:
            _hp_fn = getattr(pen_st, "has_penalties", None)
            if callable(_hp_fn) and bool(_hp_fn(idx)):
                return False, False

        if FAST_V11 or FAST_V14:
            no_lp = _CACHED_NO_LOGPROBS
            if no_lp is None:
                from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS as _nlp
                _CACHED_NO_LOGPROBS = _nlp
                no_lp = _nlp
        else:
            from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS as no_lp

        if ss.max_num_logprobs(idx) != no_lp:
            return False, False
        lts = getattr(sp, "logprob_token_ids_state", None)
        if lts is not None and lts.max_num_token_ids(idx) != 0:
            return False, False

        if has_pos:
            if not TEMP_V16 or MODE != "argmax":
                return False, False
            if not _validate_temp_v16_sampling_states(ss, idx):
                return False, False
            _S["temp_mode"] = "mixed" if has_zero else "temp"
            _S["temp_shared_prefix_n"] = _detect_shared_prefix_n(input_batch)
            return True, is_k0

        _S["temp_mode"] = "greedy"
        return True, is_k0
    except Exception as e:  # noqa: BLE001 - fail closed
        _S["temp_mode"] = "off"
        if not _S.get("arm_warned"):
            _S["arm_warned"] = True
            logger.warning("INKLING_SYNC_FREE: arming check failed (%r); stock head", e)
        return False, False


def _install_runner_hook() -> None:
    if _S["runner_hooked"]:
        return
    _S["runner_hooked"] = True
    try:
        from vllm.v1.worker.gpu import model_runner as mr

        cls = mr.GPUModelRunner
        if getattr(cls, "_inkling_sync_free", False):
            return
        orig = cls.sample

        def sample(self, *args, **kwargs):
            if FAST_V14 and not kwargs:
                n_a = len(args)
                hs = args[0] if n_a > 0 else None
                ib = args[1] if n_a > 1 else None
                go = args[2] if n_a > 2 else None
            else:
                hs = kwargs.get("hidden_states", args[0] if len(args) > 0 else None)
                ib = kwargs.get("input_batch", args[1] if len(args) > 1 else None)
                go = kwargs.get("grammar_output", args[2] if len(args) > 2 else None)
            armed, is_k0 = step_armed(self, ib, go)
            _s = _S
            _s["armed"] = armed
            _s["armed_k0"] = is_k0
            _temp_mode = _s.get("temp_mode", "off")
            if _temp_mode in ("temp", "mixed"):
                _s["temp_step_id"] = int(_s.get("temp_step_id", 0)) + 1
                _notify_cert_temp_direct(
                    True,
                    shared_prefix_n=int(_s.get("temp_shared_prefix_n", 1)),
                    update_l2pf_hint=True,
                )
            elif _temp_mode == "greedy":
                _notify_cert_temp_direct(False, shared_prefix_n=1, update_l2pf_hint=True)
            # Only pass num_tokens for pure-decode, unpadded-token-count batches so cert_head
            # nlive trimming safely falls back to M on prefill, mixed, or padded K=0 batches.
            nt = 0
            nt_raw = 0
            if FAST_V14 and ib is not None:
                try:
                    has_prefill = bool(ib.has_prefill)
                    if not has_prefill:
                        nt_pad = ib.num_tokens_after_padding or 0
                        nt_raw = ib.num_tokens or 0
                        if nt_pad > 0 and nt_raw == nt_pad:
                            nt = nt_pad
                except AttributeError:
                    has_prefill = bool(getattr(ib, "has_prefill", False))
                    if not has_prefill:
                        nt_pad = int(getattr(ib, "num_tokens_after_padding", 0) or 0)
                        nt_raw = int(getattr(ib, "num_tokens", 0) or 0)
                        if nt_pad > 0 and nt_raw == nt_pad:
                            nt = nt_pad
            else:
                has_prefill = True if ib is None else bool(getattr(ib, "has_prefill", False))
                if ib is not None and not has_prefill:
                    nt_pad = int(getattr(ib, "num_tokens_after_padding", 0) or 0)
                    nt_raw = int(getattr(ib, "num_tokens", 0) or 0)
                    if nt_pad > 0 and nt_raw == nt_pad:
                        nt = nt_pad
            _s["num_tokens"] = nt
            _s["tok"] = None
            _s["gathered"] = None
            try:
                # Zero-copy slice fast path on pure-decode batches where every token produces a logit
                # (sum(num_logits) == num_tokens -> logits_indices == [0, 1, ..., num_tokens - 1]):
                # eliminates the eager vectorized_gather_kernel (~6.2 us/step, 0-ULP bit-exact).
                if (
                    ZERO_COPY_SLICE
                    and armed
                    and go is None
                    and ib is not None
                    and not has_prefill
                    and (FAST_V14 or getattr(self, "batch_sharder", None) is None)
                    and getattr(self, "adaptive_verification", None) is None
                    and getattr(ib, "dsv41_dead_drafts", None) is None
                    and isinstance(hs, torch.Tensor)
                    and hs.dim() == 2
                    and nt_raw > 0
                    and nt_raw <= hs.shape[0]
                    and (ib.num_reqs if FAST_V14 else int(getattr(ib, "num_reqs", 0) or 0)) > 0
                ):
                    cu_np = ib.cu_num_logits_np if FAST_V14 else getattr(ib, "cu_num_logits_np", None)
                    nreqs = ib.num_reqs if FAST_V14 else int(ib.num_reqs)
                    if (
                        cu_np is not None
                        and len(cu_np) == nreqs + 1
                        and cu_np[-1] == nt_raw
                        and (
                            (is_k0 and getattr(self, "sampler", None) is not None)
                            or (
                                not is_k0
                                and getattr(self, "rejection_sampler", None) is not None
                                and getattr(self, "speculator", None) is not None
                            )
                        )
                    ):
                        sample_hs = hs if (FAST_V14 and hs.shape[0] == nt_raw) else hs[:nt_raw]
                        ms = self.model_state if FAST_V14 else getattr(self, "model_state", None)
                        if ms is not None and hasattr(ms, "compute_sample_logits"):
                            logits = ms.compute_sample_logits(sample_hs)
                        else:
                            logits = self.model.compute_logits(sample_hs)
                        if is_k0:
                            sampler_output = self.sampler(logits, ib)
                        else:
                            sampler_output = self.rejection_sampler(
                                logits, ib, self.speculator.draft_logits
                            )
                        return (
                            sampler_output,
                            sampler_output.num_sampled,
                            sampler_output.num_rejected,
                        )
                return orig(self, *args, **kwargs)
            finally:
                _s["armed"] = False
                _s["armed_k0"] = False
                _s["temp_mode"] = "off"
                _s["temp_shared_prefix_n"] = 1
                _s["num_tokens"] = 0
                _s["tok"] = None
                _s["gathered"] = None
                if _temp_mode in ("temp", "mixed"):
                    _notify_cert_temp_direct(False, shared_prefix_n=1, update_l2pf_hint=False)

        cls.sample = sample
        cls._inkling_sync_free = True
        logger.info("INKLING_SYNC_FREE: GPUModelRunner.sample arming hook installed (MAX_M=%d, K0=%d)", MAX_M, int(ALLOW_K0))
        if DIRECT and MODE == "argmax":
            if not _S.get("rs_hooked"):
                _install_rejection_hook()
            if ALLOW_K0 and not _S.get("sp_hooked"):
                _install_sampler_hook()
    except Exception as e:  # noqa: BLE001
        logger.warning("INKLING_SYNC_FREE: could not hook GPUModelRunner.sample (%r): never armed", e)


def _install_sampler_hook() -> None:
    """Direct greedy bypass for Sampler.__call__ (K=0 Dynamic SD & greedy prefill final chunks)."""
    if _S.get("sp_hooked"):
        return
    try:
        from vllm.v1.worker.gpu.input_batch import get_num_sampled_and_rejected
        from vllm.v1.worker.gpu.sample import sampler as sp_mod
        from vllm.v1.worker.gpu.sample.output import SamplerOutput

        cls = sp_mod.Sampler
        if getattr(cls, "_inkling_sf_direct", False):
            _S["sp_hooked"] = True
            return
        orig_call = cls.__call__

        def _direct_sampler_call(self, logits, input_batch):
            tok = _S.get("tok")
            g_info = _S.get("gathered")
            if tok is None and g_info is None and _S.get("temp_mode") in ("temp", "mixed") and TEMP_V16:
                out_v16 = _try_temp_v16_sample_k0(self, logits, input_batch)
                if out_v16 is not None:
                    return out_v16
                return orig_call(self, logits, input_batch)
            if tok is None and g_info is not None and _S.get("armed_k0", False):
                _S["gathered"] = None
                gathered, M_g, tp_g, split_g = g_info
                if M_g == logits.shape[0]:
                    # Wave-V7 3->1 K=0 fusion: argmax + num_sampled + num_rejected in 1 native 1-warp launch!
                    if (
                        M_g == input_batch.num_reqs
                        and input_batch.num_reqs > 0
                        and _sf_v7_ok(gathered.device)
                        and input_batch.seq_lens.dtype == torch.int32
                        and self.req_states.prefill_len.gpu.dtype == torch.int32
                        and input_batch.idx_mapping.dtype in (torch.int32, torch.int64)
                    ):
                        lib = _get_sf_v7_lib()
                        tok_v7 = torch.empty((M_g,), dtype=torch.int64, device=gathered.device)
                        ns_v7 = torch.empty((M_g,), dtype=torch.int32, device=gathered.device)
                        nr_v7 = torch.empty((M_g,), dtype=torch.int32, device=gathered.device)
                        rc_k0 = lib.sync_free_k0_tok_status_v7_fwd(
                            gathered.data_ptr(), int(M_g), int(tp_g), int(split_g),
                            tok_v7.data_ptr(), 1, ns_v7.data_ptr(), nr_v7.data_ptr(),
                            input_batch.seq_lens.data_ptr(),
                            input_batch.idx_mapping.data_ptr(),
                            1 if input_batch.idx_mapping.dtype == torch.int64 else 0,
                            self.req_states.prefill_len.gpu.data_ptr(),
                            1 if V7_PDL else 0,
                            int(torch.cuda.current_stream(gathered.device).cuda_stream),
                        )
                        if rc_k0 == 0:
                            return SamplerOutput(
                                sampled_token_ids=tok_v7.view(-1, 1),
                                logprobs_tensors=None,
                                num_nans=None,
                                num_sampled=ns_v7,
                                num_rejected=nr_v7,
                                sampling_mask_tensors=None,
                            )
                    tok = sf_tok_only(gathered, M_g, tp_g, split_g, pdl=True)
            if tok is not None and _S.get("armed_k0", False):
                _S["tok"] = None
                _S["gathered"] = None
                if (
                    tok.shape[0] == logits.shape[0]
                    and tok.shape[0] == input_batch.num_reqs
                    and input_batch.num_reqs > 0
                ):
                    num_sampled, num_rejected = get_num_sampled_and_rejected(
                        input_batch.seq_lens.new_ones(input_batch.num_reqs),
                        input_batch.seq_lens,
                        input_batch.cu_num_logits,
                        input_batch.idx_mapping,
                        self.req_states.prefill_len.gpu,
                    )
                    return SamplerOutput(
                        sampled_token_ids=tok.view(-1, 1),
                        logprobs_tensors=None,
                        num_nans=None,
                        num_sampled=num_sampled,
                        num_rejected=num_rejected,
                        sampling_mask_tensors=None,
                    )
            return orig_call(self, logits, input_batch)

        cls.__call__ = _direct_sampler_call
        cls._inkling_sf_direct = True
        _S["sp_hooked"] = True
        logger.info("INKLING_SYNC_FREE: direct greedy Sampler bypass installed (K=0 / non-spec zero [M,V] one-hot)")
    except Exception as e:  # noqa: BLE001
        logger.warning("INKLING_SYNC_FREE: direct Sampler hook failed (%r); K=0 falls back to one-hot", e)


def _install_rejection_hook() -> None:
    if _S.get("rs_hooked"):
        return
    try:
        from vllm.v1.worker.gpu.input_batch import get_num_sampled_and_rejected
        from vllm.v1.worker.gpu.sample.output import SamplerOutput
        from vllm.v1.worker.gpu.spec_decode import rejection_sampler as rs_mod

        gather_draft_sampled = rs_mod.gather_draft_sampled

        cls = rs_mod.RejectionSampler
        if getattr(cls, "_inkling_sf_direct", False):
            _S["rs_hooked"] = True
            return
        orig_call = cls.__call__

        def _direct_call(self, logits, input_batch, draft_logits=None):
            clear_static_ring_state()
            g_info = _S.get("gathered")
            tok = _S.get("tok")
            if g_info is None and tok is None and _S.get("temp_mode") in ("temp", "mixed") and TEMP_V16:
                out_v16 = _try_temp_v16_reject(self, logits, input_batch, draft_logits)
                if out_v16 is not None:
                    return out_v16
                return orig_call(self, logits, input_batch, draft_logits)
            if g_info is not None:
                _S["gathered"] = None
                _S["tok"] = None
                gathered, M_g, tp_g, split_g = g_info
                num_reqs = input_batch.num_reqs
                if M_g == logits.shape[0] and num_reqs > 0:
                    if FAST_V14 and FUSE_REJECT and not getattr(input_batch, "has_prefill", True):
                        ib_pos = input_batch.positions
                        ib_li = input_batch.logits_indices
                        ib_ids = input_batch.input_ids
                        if (
                            ib_pos.dim() == 1
                            and ib_li.shape[0] == M_g
                            and ib_ids.stride(0) == 1
                            and ib_pos.stride(0) == 1
                            and ib_li.stride(0) == 1
                            and not (TREE_SPEC and num_reqs == 1)
                        ):
                            g_dev = gathered.device
                            _ring_v14 = _acquire_static_ring_outputs_v14(
                                g_dev, num_reqs, self.num_speculative_steps + 1
                            )
                            if _ring_v14 is not None:
                                _, sampled, num_sampled, num_rejected, p_s, p_ns, p_nr, _, s_stride0 = _ring_v14
                            else:
                                sampled = torch.empty(
                                    (num_reqs, self.num_speculative_steps + 1),
                                    dtype=torch.int64,
                                    device=g_dev,
                                )
                                num_sampled = torch.empty((num_reqs,), dtype=torch.int32, device=g_dev)
                                num_rejected = torch.empty((num_reqs,), dtype=torch.int32, device=g_dev)
                                p_s = int(sampled.data_ptr())
                                p_ns = int(num_sampled.data_ptr())
                                p_nr = int(num_rejected.data_ptr())
                                s_stride0 = int(sampled.stride(0))
                            d_idx = (g_dev.index if g_dev.index is not None else torch.cuda.current_device()) if g_dev.type == "cuda" else -1
                            lib_r = _SF_V7_LIB_BY_DEV.get(d_idx)
                            if lib_r is None and _sf_v7_ok(g_dev):
                                lib_r = _get_sf_v7_lib()
                                if lib_r is not None:
                                    _SF_V7_LIB_BY_DEV[d_idx] = lib_r
                            if lib_r is not None:
                                i32 = torch.int32
                                i64 = torch.int64
                                ib_sl = input_batch.seq_lens
                                ib_cu = input_batch.cu_num_logits
                                ib_idx = input_batch.idx_mapping
                                pf_gpu = self.sampler.req_states.prefill_len.gpu
                                dt_ids = ib_ids.dtype
                                dt_pos = ib_pos.dtype
                                dt_li = ib_li.dtype
                                dt_idx = ib_idx.dtype
                                if (
                                    ib_sl.dtype is i32
                                    and pf_gpu.dtype is i32
                                    and ib_cu.dtype is i32
                                    and (dt_ids is i32 or dt_ids is i64)
                                    and (dt_pos is i32 or dt_pos is i64)
                                    and (dt_li is i32 or dt_li is i64)
                                    and (dt_idx is i32 or dt_idx is i64)
                                ):
                                    dflags = (
                                        (1 if dt_ids is i64 else 0)
                                        | (2 if dt_pos is i64 else 0)
                                        | (4 if dt_li is i64 else 0)
                                        | (8 if dt_idx is i64 else 0)
                                    )
                                    rc_fr = lib_r.sync_free_fused_reject_v7_fwd(
                                        gathered.data_ptr(), M_g, num_reqs, tp_g, split_g,
                                        p_s, s_stride0, p_ns, p_nr,
                                        ib_sl.data_ptr(), ib_ids.data_ptr(),
                                        ib_pos.data_ptr(), ib_li.data_ptr(),
                                        ib_idx.data_ptr(), pf_gpu.data_ptr(),
                                        ib_cu.data_ptr(),
                                        dflags, 1 if V7_PDL else 0,
                                        torch.cuda.current_stream(g_dev).cuda_stream,
                                    )
                                    if rc_fr == 0:
                                        return SamplerOutput(
                                            sampled_token_ids=sampled,
                                            logprobs_tensors=None,
                                            num_nans=None,
                                            num_sampled=num_sampled,
                                            num_rejected=num_rejected,
                                            sampling_mask_tensors=None,
                                        )
                            # Fallback to Triton fused_reject preserving the acquired ring tensors
                            k = _kernels()
                            k["fused_reject"][(num_reqs,)](
                                gathered,
                                M_g,
                                sampled,
                                s_stride0,
                                num_sampled,
                                num_rejected,
                                input_batch.seq_lens,
                                ib_ids,
                                ib_pos,
                                ib_li,
                                input_batch.idx_mapping,
                                self.sampler.req_states.prefill_len.gpu,
                                input_batch.cu_num_logits,
                                TP=tp_g,
                                SPLIT=split_g,
                                NP=triton_np2(tp_g * split_g),
                                PDL=True,
                                num_warps=1,
                                launch_pdl=True,
                            )
                            return SamplerOutput(
                                sampled_token_ids=sampled,
                                logprobs_tensors=None,
                                num_nans=None,
                                num_sampled=num_sampled,
                                num_rejected=num_rejected,
                                sampling_mask_tensors=None,
                            )
                    if (
                        FUSE_REJECT
                        and not getattr(input_batch, "has_prefill", True)
                        and input_batch.positions.dim() == 1
                        and input_batch.logits_indices.shape[0] == M_g
                        and input_batch.input_ids.stride(0) == 1
                        and input_batch.positions.stride(0) == 1
                        and input_batch.logits_indices.stride(0) == 1
                    ):
                        _ring_out = _acquire_static_ring_outputs(
                            gathered.device, num_reqs, self.num_speculative_steps + 1
                        )
                        if _ring_out is not None:
                            _, sampled, num_sampled, num_rejected = _ring_out
                        else:
                            sampled = torch.empty(
                                (num_reqs, self.num_speculative_steps + 1),
                                dtype=torch.int64,
                                device=gathered.device,
                            )
                            num_sampled = torch.empty(
                                (num_reqs,), dtype=torch.int32, device=gathered.device
                            )
                            num_rejected = torch.empty(
                                (num_reqs,), dtype=torch.int32, device=gathered.device
                            )
                        k = None if (FAST_V11 or FAST_V14) else _kernels()
                        tree_st = None
                        tsr_mod = None
                        if TREE_SPEC and num_reqs == 1:
                            try:
                                from . import tree_spec_runtime as tsr_mod
                            except ImportError:
                                import tree_spec_runtime as tsr_mod  # type: ignore[no-redef]
                            tree_st = tsr_mod.get_active_tree_state(M_g)
                        if tree_st is not None and tsr_mod is not None:
                            ran_v7_tr = False
                            if (
                                _sf_v7_ok(gathered.device)
                                and input_batch.seq_lens.dtype == torch.int32
                                and self.sampler.req_states.prefill_len.gpu.dtype == torch.int32
                                and input_batch.cu_num_logits.dtype == torch.int32
                                and input_batch.input_ids.dtype in (torch.int32, torch.int64)
                                and input_batch.positions.dtype in (torch.int32, torch.int64)
                                and input_batch.logits_indices.dtype in (torch.int32, torch.int64)
                                and input_batch.idx_mapping.dtype in (torch.int32, torch.int64)
                            ):
                                lib_r = _get_sf_v7_lib()
                                dflags = (
                                    (1 if input_batch.input_ids.dtype == torch.int64 else 0)
                                    | (2 if input_batch.positions.dtype == torch.int64 else 0)
                                    | (4 if input_batch.logits_indices.dtype == torch.int64 else 0)
                                    | (8 if input_batch.idx_mapping.dtype == torch.int64 else 0)
                                )
                                rc_tr = lib_r.sync_free_fused_tree_reject_v7_fwd(
                                    gathered.data_ptr(), int(M_g), int(num_reqs), int(tp_g), int(split_g),
                                    int(tree_st.max_m), sampled.data_ptr(), int(sampled.stride(0)),
                                    num_sampled.data_ptr(), num_rejected.data_ptr(),
                                    input_batch.seq_lens.data_ptr(), input_batch.input_ids.data_ptr(),
                                    input_batch.positions.data_ptr(), input_batch.logits_indices.data_ptr(),
                                    input_batch.idx_mapping.data_ptr(), self.sampler.req_states.prefill_len.gpu.data_ptr(),
                                    input_batch.cu_num_logits.data_ptr(), tree_st.tree_parent.data_ptr(),
                                    tree_st.n_live.data_ptr(), tree_st.accepted_path.data_ptr(),
                                    tree_st.accepted_len.data_ptr(), tree_st.needs_compact.data_ptr(),
                                    int(dflags), 1 if V7_PDL else 0,
                                    int(torch.cuda.current_stream(gathered.device).cuda_stream),
                                )
                                ran_v7_tr = (rc_tr == 0)
                            if not ran_v7_tr:
                                if k is None:
                                    k = _kernels()
                                k["fused_tree_reject"][(1,)](
                                    gathered,
                                    M_g,
                                    sampled,
                                    sampled.stride(0),
                                    num_sampled,
                                    num_rejected,
                                    input_batch.seq_lens,
                                    input_batch.input_ids,
                                    input_batch.positions,
                                    input_batch.logits_indices,
                                    input_batch.idx_mapping,
                                    self.sampler.req_states.prefill_len.gpu,
                                    input_batch.cu_num_logits,
                                    tree_st.tree_parent,
                                    tree_st.n_live,
                                    tree_st.accepted_path,
                                    tree_st.accepted_len,
                                    tree_st.needs_compact,
                                    TP=tp_g,
                                    SPLIT=split_g,
                                    NP=triton_np2(tp_g * split_g),
                                    MAX_M=tree_st.max_m,
                                    PDL=True,
                                    num_warps=1,
                                    launch_pdl=True,
                                )
                            tsr_mod.compact_tree_kv_conv(
                                tree_st,
                                input_batch.positions,
                                use_pdl=True,
                            )
                        else:
                            ran_v7_fr = False
                            if (
                                _sf_v7_ok(gathered.device)
                                and input_batch.seq_lens.dtype == torch.int32
                                and self.sampler.req_states.prefill_len.gpu.dtype == torch.int32
                                and input_batch.cu_num_logits.dtype == torch.int32
                                and input_batch.input_ids.dtype in (torch.int32, torch.int64)
                                and input_batch.positions.dtype in (torch.int32, torch.int64)
                                and input_batch.logits_indices.dtype in (torch.int32, torch.int64)
                                and input_batch.idx_mapping.dtype in (torch.int32, torch.int64)
                            ):
                                lib_r = _get_sf_v7_lib()
                                dflags = (
                                    (1 if input_batch.input_ids.dtype == torch.int64 else 0)
                                    | (2 if input_batch.positions.dtype == torch.int64 else 0)
                                    | (4 if input_batch.logits_indices.dtype == torch.int64 else 0)
                                    | (8 if input_batch.idx_mapping.dtype == torch.int64 else 0)
                                )
                                rc_fr = lib_r.sync_free_fused_reject_v7_fwd(
                                    gathered.data_ptr(), int(M_g), int(num_reqs), int(tp_g), int(split_g),
                                    sampled.data_ptr(), int(sampled.stride(0)),
                                    num_sampled.data_ptr(), num_rejected.data_ptr(),
                                    input_batch.seq_lens.data_ptr(), input_batch.input_ids.data_ptr(),
                                    input_batch.positions.data_ptr(), input_batch.logits_indices.data_ptr(),
                                    input_batch.idx_mapping.data_ptr(), self.sampler.req_states.prefill_len.gpu.data_ptr(),
                                    input_batch.cu_num_logits.data_ptr(),
                                    int(dflags), 1 if V7_PDL else 0,
                                    int(torch.cuda.current_stream(gathered.device).cuda_stream),
                                )
                                ran_v7_fr = (rc_fr == 0)
                            if not ran_v7_fr:
                                if k is None:
                                    k = _kernels()
                                k["fused_reject"][(num_reqs,)](
                                    gathered,
                                    M_g,
                                    sampled,
                                    sampled.stride(0),
                                    num_sampled,
                                    num_rejected,
                                    input_batch.seq_lens,
                                    input_batch.input_ids,
                                    input_batch.positions,
                                    input_batch.logits_indices,
                                    input_batch.idx_mapping,
                                    self.sampler.req_states.prefill_len.gpu,
                                    input_batch.cu_num_logits,
                                    TP=tp_g,
                                    SPLIT=split_g,
                                    NP=triton_np2(tp_g * split_g),
                                    PDL=True,
                                    num_warps=1,
                                    launch_pdl=True,
                                )
                        return SamplerOutput(
                            sampled_token_ids=sampled,
                            logprobs_tensors=None,
                            num_nans=None,
                            num_sampled=num_sampled,
                            num_rejected=num_rejected,
                            sampling_mask_tensors=None,
                        )
                    tok = sf_tok_only(gathered, M_g, tp_g, split_g, pdl=True)
            if tok is not None:
                _S["tok"] = None
                if tok.shape[0] == logits.shape[0] and input_batch.num_reqs > 0:
                    draft_sampled, _pos = gather_draft_sampled(
                        input_batch.input_ids,
                        input_batch.positions,
                        input_batch.logits_indices,
                        input_batch.expanded_idx_mapping,
                        input_batch.expanded_local_pos,
                        self.sampler.req_states.prefill_len.gpu,
                    )
                    num_reqs = input_batch.num_reqs
                    # Do NOT acquire static ring outputs on this fallback path: get_num_sampled_and_rejected()
                    # below replaces num_sampled & num_rejected with freshly allocated tensors.
                    sampled = draft_sampled.new_empty(
                        (num_reqs, self.num_speculative_steps + 1), dtype=torch.int64
                    )
                    num_sampled = sampled.new_empty((num_reqs,), dtype=torch.int32)
                    k = _kernels()
                    k["direct_reject"][(num_reqs,)](
                        sampled,
                        sampled.stride(0),
                        num_sampled,
                        tok,
                        draft_sampled,
                        input_batch.cu_num_logits,
                        num_warps=1,
                    )
                    num_sampled, num_rejected = get_num_sampled_and_rejected(
                        num_sampled,
                        input_batch.seq_lens,
                        input_batch.cu_num_logits,
                        input_batch.idx_mapping,
                        self.sampler.req_states.prefill_len.gpu,
                    )
                    return SamplerOutput(
                        sampled_token_ids=sampled,
                        logprobs_tensors=None,
                        num_nans=None,
                        num_sampled=num_sampled,
                        num_rejected=num_rejected,
                        sampling_mask_tensors=None,
                    )
            return orig_call(self, logits, input_batch, draft_logits)

        cls.__call__ = _direct_call
        cls._inkling_sf_direct = True
        _S["rs_hooked"] = True
        logger.info("INKLING_SYNC_FREE: direct greedy RejectionSampler bypass installed (zero [M,V] one-hot)")
    except Exception as e:  # noqa: BLE001
        logger.warning("INKLING_SYNC_FREE: direct RejectionSampler hook failed (%r); falling back to one-hot", e)


def _layout_ok(lp, head, V: int) -> tuple[bool, str]:
    try:
        si = head.shard_indices
        rank, tp = _tp()
        org = int(lp.org_vocab_size)
        if getattr(lp, "logits_as_input", False):
            return False, "logits_as_input"
        if lp.scale != 1.0:
            return False, "logits scale != 1"
        if int(head.tp_size) != tp or tp < 2:
            return False, f"head tp_size {head.tp_size} vs TP {tp}"
        if int(si.num_added_elements) != 0:
            return False, "added (LoRA) vocab"
        if int(si.num_org_vocab_padding) != 0 or int(si.num_org_elements) != V:
            return False, f"per-shard padding (num_org_elements {si.num_org_elements}, V {V})"
        if int(si.org_vocab_start_index) != rank * V:
            return False, f"org_vocab_start_index {si.org_vocab_start_index} != rank*V"
        if not ((tp - 1) * V < org <= tp * V):
            return False, f"org_vocab {org} not inside the last shard of {tp} x {V}"
        return True, ""
    except Exception as e:  # noqa: BLE001
        return False, f"layout probe failed {e!r}"


def _state(model, hidden_states: torch.Tensor):
    st = getattr(model, "_inkling_sf", None)
    if st is not None:
        return st or None
    if _capturing():
        return None
    lp, head = model.logits_processor, model.lm_head
    rank, tp = _tp()
    W = getattr(head, "weight", None)
    V = int(W.shape[0]) if W is not None and W.dim() == 2 else 0
    ok, why = _layout_ok(lp, head, V) if V else (False, "no 2-D lm_head.weight")
    cap = None if lp.soft_cap is None else float(lp.soft_cap)
    fuse = FUSE and cap is not None
    chain_ok = True
    nbad = 0
    if ok and fuse:
        chain_ok, nbad = chain_selftest(cap, hidden_states.device)
        if not chain_ok:
            logger.warning("INKLING_SYNC_FREE: fused soft-cap chain differs from torch on %d bf16 patterns; "
                           "using the torch-op chain on the shard", nbad)
    hd = getattr(lp, "head_dtype", None)
    dt = torch.float32 if hd == torch.float32 else torch.bfloat16
    if ok and (W.dtype != torch.bfloat16 or hidden_states.dtype != torch.bfloat16
               or hd not in (None, torch.bfloat16, torch.float32)):
        ok, why = False, f"head dtype {W.dtype} / head_dtype {hd}"
    agreed = _tp_min([int(ok)])[0] == 1
    st = False
    if agreed:
        org = int(lp.org_vocab_size)
        split = max(1, SPLIT)
        valid = max(1, min(V, org - rank * V))
        st = dict(V=V, tp=tp, rank=rank, org=org, cap=cap, valid=valid, vstart=rank * V,
                  fuse=fuse and chain_ok and dt == torch.bfloat16, split=split, dt=dt)
        st["cs"] = _cert_state(lp, head, W, valid, cap, dt, hidden_states.device) if CERT != "off" and MODE != "off" else None
        if FP32_FUSE and dt == torch.float32:
            if st["cs"] is not None:
                st["fp32_divmode"] = int(st["cs"].divmode)
            elif cap is not None:
                try:
                    from . import cert_head as _ch_dm
                except Exception:  # noqa: BLE001
                    import cert_head as _ch_dm  # type: ignore
                _dm, _, _ = _ch_dm.pick_chain_mode(cap, hidden_states.device)
                st["fp32_divmode"] = int(_dm) if _dm >= 0 else -1
            else:
                st["fp32_divmode"] = -1
        else:
            st["fp32_divmode"] = -1
        if CUDA_V7:
            qualify_sf_v7(hidden_states.device, cap, dt, st["fp32_divmode"])
        if TEMP_V16:
            qualify_sf_temp_v16(hidden_states.device)
        dummy = torch.zeros((2, V), dtype=dt, device=hidden_states.device)
        p = sf_local_pairs(dummy, valid, rank * V, cap, split, st["fuse"], fp32_divmode=st["fp32_divmode"])
        eff_sp_dum = max(1, int(p.shape[0]) // 2)
        g_dum = p.repeat(tp, 1).contiguous()
        sf_onehot(g_dum, 2, tp, org, eff_sp_dum, dt)
        sf_tok_only(g_dum, 2, tp, eff_sp_dum)
        sf_tok_only(g_dum, 2, tp, eff_sp_dum, pdl=True)
        if DIRECT and MODE == "argmax":
            k = _kernels()
            s_dum = torch.empty((1, 8), dtype=torch.int64, device=hidden_states.device)
            ns_dum = torch.empty((1,), dtype=torch.int32, device=hidden_states.device)
            t_dum = torch.zeros((2,), dtype=torch.int64, device=hidden_states.device)
            d_dum = torch.zeros((2,), dtype=torch.int32, device=hidden_states.device)
            c_dum = torch.tensor([0, 2], dtype=torch.int32, device=hidden_states.device)
            k["direct_reject"][(1,)](s_dum, 8, ns_dum, t_dum, d_dum, c_dum, num_warps=1)
            if FUSE_REJECT:
                nr_dum = torch.empty((1,), dtype=torch.int32, device=hidden_states.device)
                sl_dum = torch.tensor([2], dtype=torch.int32, device=hidden_states.device)
                idx_dum = torch.tensor([0, 1], dtype=torch.int32, device=hidden_states.device)
                pos_dum = torch.tensor([1, 2], dtype=torch.int64, device=hidden_states.device)
                map_dum = torch.zeros((1,), dtype=torch.int32, device=hidden_states.device)
                pfl_dum = torch.zeros((1,), dtype=torch.int32, device=hidden_states.device)
                for _sp in sorted({1, split}):
                    _g_sp = torch.zeros((tp * 2 * _sp, 4), dtype=torch.float32, device=hidden_states.device)
                    k["fused_reject"][(1,)](
                        _g_sp, 2, s_dum, 8, ns_dum, nr_dum, sl_dum, d_dum, pos_dum, idx_dum, map_dum, pfl_dum, c_dum,
                        TP=tp, SPLIT=_sp, NP=triton_np2(tp * _sp), PDL=True,
                        num_warps=1, launch_pdl=True,
                    )
                    if TREE_SPEC:
                        try:
                            from . import tree_spec_runtime as _tsr_w
                        except ImportError:
                            import tree_spec_runtime as _tsr_w  # type: ignore[no-redef]
                        _tsr_w.get_tree_target_lib()
                        _tp_dum = torch.tensor([-1, 0, 1, 2, 3, 4, 5, 6], dtype=torch.int32, device=hidden_states.device)
                        _nl_dum = torch.tensor([2], dtype=torch.int32, device=hidden_states.device)
                        _ap_dum = torch.zeros((8,), dtype=torch.int32, device=hidden_states.device)
                        _al_dum = torch.ones((1,), dtype=torch.int32, device=hidden_states.device)
                        _nc_dum = torch.zeros((1,), dtype=torch.int32, device=hidden_states.device)
                        k["fused_tree_reject"][(1,)](
                            _g_sp, 2, s_dum, 8, ns_dum, nr_dum, sl_dum, d_dum, pos_dum, idx_dum, map_dum, pfl_dum, c_dum,
                            _tp_dum, _nl_dum, _ap_dum, _al_dum, _nc_dum,
                            TP=tp, SPLIT=_sp, NP=triton_np2(tp * _sp), MAX_M=8, PDL=True,
                            num_warps=1, launch_pdl=True,
                        )
        torch.cuda.synchronize()
        logger.info("INKLING_SYNC_FREE=%s agreed on every rank: V/shard %d, org_vocab %d, rank %d valid %d, "
                    "cap %s, head dtype %s, fused chain %s (selftest: %d/65536 bf16 patterns differ), split %d, MAX_M %d, K0 %d",
                    MODE, V, org, rank, valid, cap, dt, st["fuse"], nbad, split, MAX_M, int(ALLOW_K0))
    else:
        logger.warning("INKLING_SYNC_FREE=%s DISABLED on every rank (this rank: %s)", MODE, why or "ok")
    model._inkling_sf = st
    return st or None


def _cert_state(lp, head, W: torch.Tensor, valid: int, cap, dt, device):
    try:
        from . import cert_head as ch
    except Exception:  # noqa: BLE001
        import cert_head as ch  # type: ignore
    try:
        if dt != torch.float32:
            raise ValueError("the certified head needs the fp32 head (head_dtype float32); this model runs a bf16 head")
        if cap is None:
            raise ValueError("no soft cap")
        mode, nd, dip = ch.pick_chain_mode(cap, device)
        if mode < 0:
            raise ValueError(f"fp32 soft-cap chain not bit-equal to torch / not near-monotone ({nd} values differ, dip {dip})")
        # INKLING_CERT_HEAD_CUDA_ONLY=1 (default ON when CUDA cert_head is enabled and not in check mode):
        # build ONLY the 105.3 MiB CUDA int7_g32 table and skip the redundant 117.4 MiB Triton MXINT8 table
        # (+0.115 GiB/rank saved and faster startup; M==1 or M>16 fall back to exact stock BF16 head).
        cuda_only = (
            os.environ.get("INKLING_CERT_HEAD_CUDA_ONLY", "1").strip().lower() not in ("0", "", "off", "false", "no")
            and CERT != "check"
        )
        cs_cu = None
        ct_cu = None
        try:
            from . import cert_head_cuda_v7 as ch7
            if ch7.ENABLED and ch7._load_lib() is not None:
                rank = _tp()[0]
                ct_v7 = ch7.build_cuda_table_v7(W)
                cand_v7 = ch7.CudaCertStateV7Adapter(ct_v7, W, vreal=valid, vstart=rank * W.shape[0], cap=cap, dip=dip)
                bad_v7 = cand_v7.qualify(16)
                if not bad_v7:
                    cs_cu = cand_v7
                    try:
                        from . import l2pf
                        _head_segs = (
                            (ct_v7.nv1, ct_v7.rt1, (ct_v7.big_s, 131072), (ct_v7.ms_u4, 524288), ct_v7.slabs_u4)
                            if getattr(l2pf, "G2", False)
                            else ct_v7.slabs_u4
                        )
                        l2pf.set_cuda_head_slabs(_head_segs)
                        l2pf.prewarm(_head_segs, kind="head")
                    except Exception:
                        pass
                    logger.info("INKLING_CERT_HEAD_V7=1 ready: S1=%.1f MB (0/15 M=2..16 diffs)",
                                ct_v7.slabs_u4.numel() * ct_v7.slabs_u4.element_size() / 1e6)
                else:
                    logger.warning("INKLING_CERT_HEAD_V7 qualify failed on M=%s; falling back to v6", bad_v7)
        except Exception as _ce7:
            logger.warning("INKLING_CERT_HEAD_V7 init failed (%r); falling back to v6", _ce7)

        if cs_cu is None:
            try:
                from . import cert_head_cuda as chc
                if chc.ENABLED and chc._get_lib() is not None:
                    rank = _tp()[0]
                    ct_cu = chc.build_cuda_table(W, valid)
                    cand_cu = chc.CudaCertState(ct_cu, W, vstart=rank * W.shape[0], cap=cap, dip=dip)
                    _qmax = cand_cu.active_max_m() if hasattr(cand_cu, "active_max_m") else 16
                    bad_cu = cand_cu.qualify(_qmax)
                    bad_le16 = [m for m in bad_cu if m <= 16]
                    if not bad_le16:
                        cs_cu = cand_cu
                        try:
                            from . import l2pf
                            # Wave 0-2 lockstep: nv (100KB) + rt (100KB) + scales[:640KB] (~142 CTAs) + slabs (~7-8MB)
                            _head_segs = (
                                (ct_cu.nv, ct_cu.rt, (ct_cu.scales, 655360), ct_cu.slabs)
                                if getattr(l2pf, "G2", False)
                                else ct_cu.slabs
                            )
                            l2pf.set_cuda_head_slabs(_head_segs)
                            l2pf.prewarm(_head_segs, kind="head")
                            if getattr(l2pf, "C32", False) and _qmax > 16:
                                _head_segs_c32 = (ct_cu.nv, ct_cu.rt, (ct_cu.scales, 1572864), ct_cu.slabs)
                                l2pf.prewarm(_head_segs_c32, kind="head", num_tokens=min(256, _qmax))
                        except Exception:
                            pass
                        logger.info("INKLING_CERT_HEAD_CUDA=1 ready: fmt=%s slabs=%.1f MB max_m=%d (bad_gt16=%s)",
                                    chc.DEFAULT_FMT, ct_cu.slabs.numel() * ct_cu.slabs.element_size() / 1e6,
                                    _qmax, [m for m in bad_cu if m > 16])
                    else:
                        logger.warning("INKLING_CERT_HEAD_CUDA qualify failed on M=%s; falling back to Triton cert_head", bad_le16)
            except Exception as _ce:
                logger.warning("INKLING_CERT_HEAD_CUDA init failed (%r); falling back to Triton cert_head", _ce)

        if cuda_only and cs_cu is not None:
            cs = ch.CertState(None, W, valid, cap, mode, dip)
            cs.ch = ch
            cs.cuda_state = cs_cu
            torch.cuda.empty_cache()
            _v7_sum = None
            try:
                from . import cert_head_cuda_v7 as _ch7_log
                if hasattr(_ch7_log, "active_summary_global") and isinstance(cs_cu, _ch7_log.CudaCertStateV7Adapter):
                    _v7_sum = _ch7_log.active_summary_global()
            except Exception:
                _v7_sum = None
            if _v7_sum is not None:
                _ok_tot = _v7_sum["ok_le16"] + _v7_sum["ok_gt16"]
                _max_m_s = _v7_sum["max_m"]
                _bad_gt16 = _v7_sum["bad_gt16"]
                _served_hi = _max_m_s if (_v7_sum["ok_gt16"] > 0 and not _bad_gt16) else 16
                logger.info(
                    "INKLING_CERT_HEAD_CUDA_ONLY=1: skipped 117.4 MiB Triton MXINT8 table; "
                    "CUDA V%d INT6_g512_ms8 (S1=%.1f MB, ref=%s) serves M=2..%d (%d/%d ok; bad_gt16=%s), "
                    "stock fp32 head serves M=1 & disqualified/M>%d",
                    _v7_sum["lib_ver"], _v7_sum["s1_mb"], _v7_sum["qualify_ref"],
                    _served_hi, _ok_tot, max(0, _max_m_s - 1),
                    _bad_gt16 if len(_bad_gt16) <= 8 else f"{len(_bad_gt16)} rows ({_bad_gt16[0]}..{_bad_gt16[-1]})",
                    _served_hi,
                )
            else:
                _cu_max_m = cs_cu.active_max_m() if hasattr(cs_cu, "active_max_m") else 16
                logger.info(
                    "INKLING_CERT_HEAD_CUDA_ONLY=1: skipped 117.4 MiB Triton MXINT8 table; "
                    "CUDA cert_head serves M=2..%d, stock fp32 head serves M=1 & M>%d",
                    _cu_max_m, _cu_max_m,
                )
            return cs

        t = ch.get_table(W, valid)
        if t is None:
            raise ValueError("table build failed")
        cs = ch.CertState(t, W, valid, cap, mode, dip)
        cs.ch = ch
        if cs_cu is not None:
            cs.cuda_state = cs_cu
        _prequalify(cs, lp, head, device)
        nq = sum(1 for v in cs.m_ok.values() if v)
        logger.info("INKLING_CERT_HEAD=%s: certified head ready (table %.1f MB, MAXM %d, chain mode %d, chain dip %.3g, "
                    "%d row counts bit-equal to the stock head)", CERT, t.bytes / 1e6, ch.MAXM, mode, dip, nq)
        return cs
    except Exception as e:  # noqa: BLE001
        logger.warning("INKLING_CERT_HEAD=%s DISABLED on this rank (%r): stock head", CERT, e)
        return None


def _winners(pairs: torch.Tensor, M: int, split: int, vlen: int) -> torch.Tensor:
    return sf_onehot(pairs.contiguous(), M, 1, vlen, pairs.shape[0] // M, torch.float32)[1]


def _bits(x: torch.Tensor) -> torch.Tensor:
    return x.contiguous().view(torch.int32 if x.dtype == torch.float32 else torch.int16)


def _qualify(cs, lp, head, M: int) -> bool:
    ch = cs.ch
    ok = 2 <= M <= ch.MAXM
    if ok:
        g = torch.Generator(device=cs.W.device)
        g.manual_seed(1234 + M)
        for sc in (1.0, 0.3, 4.0):
            x = (torch.randn((M, cs.W.shape[1]), device=cs.W.device, generator=g) * sc).to(torch.bfloat16)
            ref = lp._apply_head(head, x, None)
            if ref.dtype != torch.float32 or ref.shape != (M, cs.W.shape[0]) or \
                    not bool(torch.equal(_bits(ch.exact_all(cs, x)), _bits(ref))):
                ok = False
                break
    cs.m_ok[M] = ok
    return ok


def _prequalify(cs, lp, head, device) -> None:
    ch = cs.ch
    t0 = time.time()
    bad = [M for M in range(2, ch.MAXM + 1) if not _qualify(cs, lp, head, M)]
    torch.cuda.synchronize(device)
    logger.info("INKLING_CERT_HEAD: refine vs torch.mm(out_dtype=fp32) bit-equality for M=2..%d in %.1fs; "
                "NOT bit-equal (stock head): M=1 (always)%s", ch.MAXM, time.time() - t0,
                (", " + ",".join(map(str, bad))) if bad else "")


def _head_local(model, st, lp, head, hidden_states: torch.Tensor):
    cs = st.get("cs")
    M = int(hidden_states.shape[0])
    if cs is None or cs.disabled or M > cs.ch.MAXM or not cs.m_ok.get(M, False):
        return lp._apply_head(head, hidden_states, None)
    try:
        cert = cs.ch.certified_local(cs, hidden_states, num_tokens=_S.get("num_tokens", 0))
    except TypeError:
        cert = cs.ch.certified_local(cs, hidden_states)
    if cert is None:
        return lp._apply_head(head, hidden_states, None)
    if CERT == "check":
        stock = lp._apply_head(head, hidden_states, None)
        pc = sf_local_pairs(cert, st["valid"], st["vstart"], st["cap"], st["split"], st["fuse"])
        ps = sf_local_pairs(stock, st["valid"], st["vstart"], st["cap"], st["split"], st["fuse"])
        ctr = cs.check
        if ctr is None:
            ctr = cs.check = torch.zeros((4,), dtype=torch.int64, device=stock.device)
        comp = ~torch.isnan(cert)
        vlen = st["valid"] + st["vstart"]
        ctr.add_(torch.stack([
            torch.ones((), dtype=torch.int64, device=ctr.device),
            torch.full((), M, dtype=torch.int64, device=ctr.device),
            (_winners(pc, M, st["split"], vlen) != _winners(ps, M, st["split"], vlen)).sum(),
            (_bits(cert) != _bits(stock))[comp].sum()]))
        if (cs.calls + 1) % cs.ch.PERIOD == 0 and cs.ch.LOG_EVERY > 0:
            c = [int(v) for v in ctr.tolist()]
            msg = (f"INKLING_CERT_HEAD check rank {st['rank']}: calls {c[0]} rows {c[1]} winner mismatches {c[2]} "
                   f"computed-logit bit mismatches {c[3]}")
            (logger.warning if (c[2] or c[3]) else logger.info)(msg)
        cs.ch.tick(cs, st["rank"])
        return stock
    cs.ch.tick(cs, st["rank"])
    return cert


def _stats_tick(armed: bool) -> None:
    _S["calls"] = _S.get("calls", 0) + 1
    _S["armed_calls"] = _S.get("armed_calls", 0) + int(armed)
    if _S["calls"] % 20000 == 0:
        try:
            rank = _tp()[0]
        except Exception:  # noqa: BLE001
            rank = 0
        if rank == 0:
            logger.info("INKLING_SYNC_FREE: %d head calls, %d armed", _S["calls"], _S["armed_calls"])


def compact_logits(lp, head, st, local: torch.Tensor):
    from vllm.distributed import tensor_model_parallel_all_gather

    M = int(local.shape[0])
    pair = sf_local_pairs(
        local, st["valid"], st["vstart"], st["cap"], st["split"], st["fuse"],
        fp32_divmode=st.get("fp32_divmode", -1),
    )
    eff_split = max(1, int(pair.shape[0]) // M)
    gathered = tensor_model_parallel_all_gather(pair, dim=0)
    if not _S.get("logged"):
        _S["logged"] = True
        logger.info("INKLING_SYNC_FREE: first armed step (M=%d): %d B/row all-gather replaces the "
                    "%d x %d bf16 logits all-gather + 4 full-vocab elementwise kernels",
                    M, 16 * eff_split, M, local.shape[1])
    return sf_onehot(gathered.contiguous(), M, st["tp"], st["org"], eff_split, st.get("dt", torch.bfloat16))


def _audit(model, st, local: torch.Tensor, hidden_states: torch.Tensor):
    from vllm.v1.worker.gpu.sample.greedy_argmax import greedy_argmax

    full = model.logits_processor(model.lm_head, hidden_states)
    _out, tok = compact_logits(model.logits_processor, model.lm_head, st, local)
    ref = greedy_argmax(full)
    fnan = torch.isnan(full).any(dim=-1)
    ties = (full == full.max(dim=-1, keepdim=True).values).sum(dim=-1) > 1
    ctr = _S.get("ctr")
    if ctr is None:
        ctr = _S["ctr"] = torch.zeros((5,), dtype=torch.int64, device=local.device)
    n = torch.full((), ref.shape[0], dtype=torch.int64, device=ctr.device)
    bad = (ref != tok) & ~fnan
    ctr.add_(torch.stack([torch.ones((), dtype=torch.int64, device=ctr.device), n,
                          bad.sum(), fnan.sum(), ties.sum()]))
    _S["audit_calls"] = _S.get("audit_calls", 0) + 1
    if _S["audit_calls"] % AUDIT_EVERY == 0:
        c = [int(v) for v in ctr.tolist()]
        rank = st["rank"]
        msg = (f"INKLING_SYNC_FREE audit rank {rank}: calls {c[0]} rows {c[1]} winner mismatches {c[2]} "
               f"(NaN rows excluded: {c[3]}; rows with an exact max tie: {c[4]})")
        if c[2]:
            logger.warning(msg + "  MISMATCH: keep the stock gather")
        elif rank == 0:
            logger.info(msg)
    return full


def target_logits(model, hidden_states: torch.Tensor):
    """Hook for Inkling2ForCausalLM.compute_logits. None -> the caller runs the stock path."""
    global _NLIVE_HAS_NT
    if MODE == "off":
        return None
    _s = _S
    # Wave-V16: temp > 0 and mixed batches hand off straight to exact full-vocab BF16/FP32 logits
    # (skipping greedy-only CertHead screening) and execute zero-sync GPU rejection sampling in
    # RejectionSampler / Sampler! Pure temp == 0.0 stays 100.000% on the untouched greedy path below.
    if _s.get("temp_mode") in ("temp", "mixed"):
        return None
    if FAST_V14 and MODE == "argmax" and not AGREE and DIRECT and _s["installed"]:
        armed = _s["armed"]
        if armed and not _s.get("force") and not torch.cuda.is_current_stream_capturing():
            try:
                st = model._inkling_sf
                if st and isinstance(hidden_states, torch.Tensor) and hidden_states.dim() == 2 and hidden_states.dtype is torch.bfloat16:
                    M = hidden_states.shape[0]
                    is_k0 = _s["armed_k0"]
                    if 2 <= M <= MAX_M and (_s.get("sp_hooked") if is_k0 else _s.get("rs_hooked")):
                        cs = st.get("cs")
                        if cs is not None and not cs.disabled:
                            cs_cu = cs.cuda_state
                            if cs_cu is not None and cs_cu.is_v7 and cs_cu._c32_state is None and M <= cs_cu.max_m:
                                h_dev = hidden_states.device
                                if _NLIVE_HAS_NT is True:
                                    nlive = cs.ch._nlive_for(M, h_dev, num_tokens=_s["num_tokens"])
                                elif _NLIVE_HAS_NT is False:
                                    nlive = cs.ch._nlive_for(M, h_dev)
                                else:
                                    try:
                                        nlive = cs.ch._nlive_for(M, h_dev, num_tokens=_s["num_tokens"])
                                        _NLIVE_HAS_NT = True
                                    except TypeError:
                                        nlive = cs.ch._nlive_for(M, h_dev)
                                        _NLIVE_HAS_NT = False
                                pair = cs_cu.local_pairs(hidden_states, nlive=nlive)
                                if pair is not None:
                                    tp_ag = _CACHED_TP_AG if _CACHED_TP_AG is not None else _fast_tp_all_gather()
                                    calls = _s.get("calls", 0) + 1
                                    _s["calls"] = calls
                                    _s["armed_calls"] = _s.get("armed_calls", 0) + 1
                                    if calls % 20000 == 0:
                                        try:
                                            _r0 = _tp()[0]
                                        except Exception:
                                            _r0 = 0
                                        if _r0 == 0:
                                            logger.info("INKLING_SYNC_FREE: %d head calls, %d armed", calls, _s["armed_calls"])
                                    if not is_k0 and M <= 16 and (not _CACHED_L2PF_PROBED or _CACHED_L2PF_FORK_CB is not None):
                                        _fast_l2pf_fork_cb("draft_l0")
                                    gathered = tp_ag(pair, dim=0)
                                    if (FUSE_REJECT and not is_k0) or (is_k0 and _sf_v7_ok(h_dev)):
                                        _s["gathered"] = (gathered.contiguous(), M, st["tp"], 1)
                                        _s["tok"] = None
                                    else:
                                        _s["tok"] = sf_tok_only(gathered.contiguous(), M, st["tp"], split=1, pdl=True)
                                        _s["gathered"] = None
                                    if not _s.get("logged"):
                                        _s["logged"] = True
                                        logger.info("INKLING_SYNC_FREE: first armed CUDA cert_head step (M=%d, k0=%d, fuse_reject=%d): 16 B/row (SPLIT=1) AG + direct greedy",
                                                    M, int(is_k0), int(FUSE_REJECT and not is_k0))
                                    return _fast_dummy_logits(M, st["dt"], h_dev)
            except (AttributeError, KeyError):
                pass
    if not _S["installed"]:
        _S["installed"] = True
        _install_runner_hook()
    if _capturing() or not isinstance(hidden_states, torch.Tensor) or hidden_states.dim() != 2:
        return None
    M = int(hidden_states.shape[0])
    lp, head = model.logits_processor, model.lm_head
    armed = bool(_S["armed"])
    is_k0 = bool(_S.get("armed_k0", False))
    st = getattr(model, "_inkling_sf", None)
    if st is None:
        _state(model, hidden_states)
        return None
    if M == 0 or M > MAX_M or hidden_states.dtype != torch.bfloat16:
        return None
    if st is False or not (armed or _S.get("force")):
        _stats_tick(False)
        return None
    if MODE == "audit" or AGREE:
        armed = _tp_min([int(armed)])[0] == 1
        if not armed:
            _stats_tick(False)
            return None
    direct_ok = _S.get("sp_hooked") if is_k0 else _S.get("rs_hooked")
    cs = st.get("cs")
    cs_cu = getattr(cs, "cuda_state", None) if (cs is not None and not cs.disabled) else None
    if FAST_V14 and cs_cu is not None and getattr(cs_cu, "_c32_state", None) is None and hasattr(cs_cu, "max_m"):
        _cu_max_m = cs_cu.max_m
    else:
        _cu_max_m = cs_cu.active_max_m() if (cs_cu is not None and hasattr(cs_cu, "active_max_m")) else 16
    if (
        cs_cu is not None
        and DIRECT
        and direct_ok
        and armed
        and not _S.get("force")
        and MODE == "argmax"
        and 2 <= M <= _cu_max_m
    ):
        nlive = None
        if M <= _cu_max_m:
            if FAST_V14:
                if _NLIVE_HAS_NT is None:
                    try:
                        nlive = cs.ch._nlive_for(M, hidden_states.device, num_tokens=_S.get("num_tokens", 0))
                        _NLIVE_HAS_NT = True
                    except TypeError:
                        nlive = cs.ch._nlive_for(M, hidden_states.device)
                        _NLIVE_HAS_NT = False
                elif _NLIVE_HAS_NT:
                    nlive = cs.ch._nlive_for(M, hidden_states.device, num_tokens=_S.get("num_tokens", 0))
                else:
                    nlive = cs.ch._nlive_for(M, hidden_states.device)
            else:
                try:
                    nlive = cs.ch._nlive_for(M, hidden_states.device, num_tokens=_S.get("num_tokens", 0))
                except TypeError:
                    nlive = cs.ch._nlive_for(M, hidden_states.device)
        if getattr(cs_cu, "is_v7", False):
            pair = cs_cu.local_pairs(hidden_states, nlive=nlive)
        else:
            from . import cert_head_cuda as chc
            pair = chc.certified_local_pairs(cs_cu, hidden_states, nlive=nlive)
        if pair is not None:
            if FAST_V11 or FAST_V14:
                tp_ag = _CACHED_TP_AG if _CACHED_TP_AG is not None else _fast_tp_all_gather()
            else:
                from vllm.distributed import tensor_model_parallel_all_gather as tp_ag
            _stats_tick(True)
            if not is_k0 and M <= 16:
                if FAST_V14:
                    if not _CACHED_L2PF_PROBED or _CACHED_L2PF_FORK_CB is not None:
                        _fast_l2pf_fork_cb("draft_l0")
                elif FAST_V11:
                    _fast_l2pf_fork_cb("draft_l0")
                else:
                    try:
                        from . import l2pf as _l2pf_cb
                        if hasattr(_l2pf_cb, "fork_cross_boundary"):
                            _l2pf_cb.fork_cross_boundary("draft_l0")
                    except Exception:
                        pass
            gathered = tp_ag(pair, dim=0)
            if (FUSE_REJECT and not is_k0) or (is_k0 and _sf_v7_ok(hidden_states.device)):
                _S["gathered"] = (gathered.contiguous(), M, st["tp"], 1)
                _S["tok"] = None
            else:
                tok = sf_tok_only(gathered.contiguous(), M, st["tp"], split=1, pdl=True)
                _S["tok"] = tok
                _S["gathered"] = None
            if not _S.get("logged"):
                _S["logged"] = True
                logger.info("INKLING_SYNC_FREE: first armed CUDA cert_head step (M=%d, k0=%d, fuse_reject=%d): 16 B/row (SPLIT=1) AG + direct greedy",
                            M, int(is_k0), int(FUSE_REJECT and not is_k0))
            dt_out = st.get("dt", torch.bfloat16)
            if FAST_V11 or FAST_V14:
                return _fast_dummy_logits(M, dt_out, hidden_states.device)
            return torch.empty((M, 1), dtype=dt_out, device=hidden_states.device)
    local = _head_local(model, st, lp, head, hidden_states)
    if local.dtype != st.get("dt", torch.bfloat16) or local.stride(1) != 1 or local.shape[1] != st["V"]:
        return None
    _stats_tick(True)
    if MODE == "audit":
        return _audit(model, st, local, hidden_states)
    direct_ok = _S.get("sp_hooked") if is_k0 else _S.get("rs_hooked")
    if DIRECT and direct_ok and armed and not _S.get("force"):
        if FAST_V11 or FAST_V14:
            tp_ag = _CACHED_TP_AG if _CACHED_TP_AG is not None else _fast_tp_all_gather()
        else:
            from vllm.distributed import tensor_model_parallel_all_gather as tp_ag
        M_l = int(local.shape[0])
        pair = sf_local_pairs(
            local, st["valid"], st["vstart"], st["cap"], st["split"], st["fuse"],
            fp32_divmode=st.get("fp32_divmode", -1),
        )
        eff_split = max(1, int(pair.shape[0]) // M_l)
        if not is_k0 and M_l <= 16:
            if FAST_V14:
                if not _CACHED_L2PF_PROBED or _CACHED_L2PF_FORK_CB is not None:
                    _fast_l2pf_fork_cb("draft_l0")
            elif FAST_V11:
                _fast_l2pf_fork_cb("draft_l0")
            else:
                try:
                    from . import l2pf as _l2pf_cb2
                    if hasattr(_l2pf_cb2, "fork_cross_boundary"):
                        _l2pf_cb2.fork_cross_boundary("draft_l0")
                except Exception:
                    pass
        gathered = tp_ag(pair, dim=0)
        if (FUSE_REJECT and not is_k0) or (is_k0 and _sf_v7_ok(local.device)):
            _S["gathered"] = (gathered.contiguous(), M_l, st["tp"], eff_split)
            _S["tok"] = None
        else:
            tok = sf_tok_only(gathered.contiguous(), M_l, st["tp"], eff_split, pdl=True)
            _S["tok"] = tok
            _S["gathered"] = None
        if not _S.get("logged"):
            _S["logged"] = True
            logger.info("INKLING_SYNC_FREE: first armed direct step (M=%d, k0=%d, fuse_reject=%d): %d B/row all-gather + direct greedy %s (no [M,V] one-hot)",
                        M_l, int(is_k0), int(FUSE_REJECT and not is_k0), 16 * eff_split, "sample" if is_k0 else "reject")
        dt_out = st.get("dt", torch.bfloat16)
        if FAST_V11 or FAST_V14:
            return _fast_dummy_logits(M_l, dt_out, local.device)
        return torch.empty((M_l, 1), dtype=dt_out, device=local.device)
    out, _tok = compact_logits(lp, head, st, local)
    return out
