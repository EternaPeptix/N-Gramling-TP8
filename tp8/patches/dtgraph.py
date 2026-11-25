# SPDX-License-Identifier: Apache-2.0
"""Inkling-2-Small per-step CUDA graph capture for eager decode glue (INKLING_DTGRAPH).

Ported and upgraded from DSv4.1 climb-r1/r1-eager (`dsv41_dtgraph.py`) for Inkling-2-Small TP8:
  * Phase 4 (`dt` region, 4_dft2tgt_glue): captures GPUModelRunner.prepare_inputs +
    prepare_attn + model_state.prepare_attn + model_state.prepare_inputs_embeds (including
    _vw_mask, INKLING_REPL_EMBED lookup, embed_norm, and inputs_embeds DtoD copy) up to
    cudagraph_manager.run_fullgraph.
  * Phase 2B (`td` region, 2_tgt2dft_glue post-AsyncOutput): captures
    GPUModelRunner.postprocess_sampled + speculator.prepare_propose + speculator.propose
    pre-graph work (es_init, prepare_dflash_inputs, _store_hidden_states, _num_rejected copy)
    up to speculator.query_cudagraph_manager.run_fullgraph.
    NOTE: Phase 2A (cert_head + sync_free AG + direct_reject + AsyncOutput s20 copy) stays
    eager because cert_head is 595 us (few ops), contains the RDMA AG collective, and
    sync_free/cert_head gate on is_current_stream_capturing(). Phase 2B starts strictly AFTER
    AsyncOutput.__init__ returns to the main stream and has zero collectives and zero
    is_current_stream_capturing() branches.

Key upgrades over DSv4.1 r1-eager:
  1. Zero edits to vLLM's model_runner.py: installs cleanly via lazy hooks on GPUModelRunner
     and CUDAGraphManager / DFlashCudaGraphManager (same pattern as patches-s5/sync_free.py).
  2. `cudaGraphExecUpdate` execs are keyed by `(region_name, shape_key, seg, slot)` so
     alternating batch sizes (T=8, T=16, batch>1) hit ExecUpdate 100% without topology churn.
  3. Full support for PARITY=1 (IMAGES=16, SEQS=32, 1M ctx): multimodal prefill / encoder
     steps are skipped; decode steps' pinned `is_mm_embed` allocations are served from the
     persistent pinned Arena (zero CachingHostAllocator pinned-memory leak).
  4. 100% compatible with dynamic W (`INKLING_VERIFY_W=conf|conf+ema|ema|force:K`),
     `INKLING_REPL_EMBED=1` (or 0, via RDMA hazard rail), `INKLING_MARKOV_EARLY_STOP=1`.

Env:
  INKLING_DTGRAPH=off|0|dt|td|both|update|warm=N|seg=N|max_reqs=N|log=N
  Recommended: INKLING_DTGRAPH=both,update  (or INKLING_DTGRAPH=dt,update)
  Default: off (zero imports, zero hooks, zero overhead).
"""
from __future__ import annotations

import copy
import dataclasses
import operator
import os
import sys
import threading
import time
import warnings
from collections import Counter, OrderedDict, deque

import numpy as np
import torch

TAG = "inkling-dtgraph"


def parse_flag(raw: str | None) -> dict:
    cfg = {
        "regions": set(),
        "update": False,
        "warm": 2,
        "seg": 4,
        "max_reqs": 0,  # 0 = unlimited (all FULL-cudagraph batch sizes)
        "max_execs": max(0, int(os.environ.get("INKLING_DTGRAPH_MAX_EXECS", "128") or "128")),
        "log": 5000,
    }
    for tok in (raw or "").replace("+", ",").split(","):
        tok = tok.strip().lower()
        if tok in ("", "0", "off", "false", "no"):
            continue
        if tok in ("1", "on", "true", "dt"):
            cfg["regions"].add("dt")
        elif tok == "td":
            cfg["regions"].add("td")
        elif tok in ("both", "all"):
            cfg["regions"].update(("dt", "td"))
        elif tok == "update":
            cfg["update"] = True
        elif "=" in tok and tok.split("=", 1)[0] in ("warm", "seg", "max_reqs", "max_execs", "log"):
            k, v = tok.split("=", 1)
            cfg[k] = max(1 if k == "seg" else 0, int(v))
        else:
            raise ValueError(
                f"INKLING_DTGRAPH: unknown token {tok!r} "
                "(use dt, td, both, update, warm=N, seg=N, max_reqs=N, max_execs=N, log=N)"
            )
    if cfg["update"] and not cfg["regions"]:
        cfg["regions"].update(("dt", "td"))
    return cfg


CFG = parse_flag(os.environ.get("INKLING_DTGRAPH", ""))
ON = bool(CFG["regions"])

_OFF = frozenset(("", "0", "off", "false", "no"))
# S7 Phase-1 step-boundary zero-math-change accelerators (all 0-ULP bit-exact, default ON):
#   INKLING_FUSED_DT_PREP=1       : 1-CTA Triton prepare_inputs for B=1 pure-decode (zero H2D DMA, -18..21 us/step)
#   INKLING_FUSED_DT_PREP_MAX_B=1 : opt-in B=2..32 uniform pure-decode fused prepare_inputs (default 1 = B=1 only)
#   INKLING_FUSED_EMBED_DIRECT=1  : write _fused_vw_repl_embed_norm directly into inputs_embeds (zero 73KB DtoD, -2.5..6 us/step)
#   INKLING_FUSED_DRAFT_STORE=1   : fold req_states.draft_tokens scatter into _vw_conf + skip readback (-10.5..11.5 us/step)
#   INKLING_PREFILL_PREWARM=0     : opt-in post-capture warmup of lazy prefill .so libs, FP8 weight cache & multi-req JITs
# Wave-V8 opt-in host/step-overhead & ROWAR accelerators (ALL default OFF = 0):
#   INKLING_DTGRAPH_FASTPATH=0         : cached op_name/_VIEW_OPS bypass, ExecUpdate handles, InputBatch/NumPy/hook caches
#   INKLING_FUSED_DT_PREP_INLINE_B8=0  : zero-H2D-DMA inline-slot Triton prepare_inputs for B=2..8 pure-decode (-12..22 us/step)
#   INKLING_ROWAR_DESC_W_ORDER=0       : experimental TP8-lockstep tail-swap placing lowest-EMA request at B-1 (fail-closed)
FUSED_DT_PREP = os.environ.get("INKLING_FUSED_DT_PREP", "1").strip().lower() not in _OFF
FUSED_DT_PREP_MAX_B = max(1, int(os.environ.get("INKLING_FUSED_DT_PREP_MAX_B", "1") or "1"))
FUSED_EMBED_DIRECT = os.environ.get("INKLING_FUSED_EMBED_DIRECT", "1").strip().lower() not in _OFF
FUSED_DRAFT_STORE = os.environ.get("INKLING_FUSED_DRAFT_STORE", "1").strip().lower() not in _OFF
PREFILL_PREWARM = os.environ.get("INKLING_PREFILL_PREWARM", "0").strip().lower() not in _OFF
PF_SCRATCH_RELEASE_IDLE = os.environ.get("INKLING_PF_SCRATCH_RELEASE_IDLE", "0").strip().lower() not in _OFF
PF_RELEASE_SHADOW_K_IDLE = os.environ.get("INKLING_PF_RELEASE_SHADOW_K_IDLE", "0").strip().lower() not in _OFF
DTGRAPH_FASTPATH = os.environ.get("INKLING_DTGRAPH_FASTPATH", "0").strip().lower() not in _OFF
FUSED_DT_PREP_INLINE_B8 = os.environ.get("INKLING_FUSED_DT_PREP_INLINE_B8", "0").strip().lower() not in _OFF
# Wave-V9: Prewarm fused_dt_prep Triton kernels (bn & bn_inline) using real live runner buffer
# dtypes + vLLM V1 InputBatch index dtype combos (eliminates live inference-time JIT compile!).
FUSED_DT_PREP_PREWARM = os.environ.get("INKLING_FUSED_DT_PREP_PREWARM", "0").strip().lower() not in _OFF
ROWAR_DESC_W_ORDER = os.environ.get("INKLING_ROWAR_DESC_W_ORDER", "0").strip().lower() not in _OFF
ROWAR_DESC_W_MARGIN = float(os.environ.get("INKLING_ROWAR_DESC_W_MARGIN", "1.25") or "1.25")
ROWAR_DESC_W_FORCE = os.environ.get("INKLING_ROWAR_DESC_W_FORCE", "0").strip().lower() not in _OFF

PIN_ARENA = True  # unit tests on CPU-only torch set this False
# Wave-V9 Phase-2: 4-slot static ring alignment + zero-TorchDispatchMode / zero-ExecUpdate
# static replay for pure single-segment Phase-2B ("td") graphs (default OFF = 0, 0-ULP exact!).
STATIC_RING = os.environ.get("INKLING_DTGRAPH_STATIC_RING", "0").strip().lower() not in _OFF
STATIC_RING_REPLAY = os.environ.get("INKLING_DTGRAPH_STATIC_RING_REPLAY", "1").strip().lower() not in _OFF
# Wave-V10: Zero-TorchDispatchMode / zero-capture / zero-ExecUpdate static replay
# for pure single-segment Phase-4 ("dt", Drafter->Target) regions (default OFF = 0, 0-ULP exact!).
STATIC_DT_REPLAY = os.environ.get("INKLING_DTGRAPH_STATIC_DT_REPLAY", "0").strip().lower() not in _OFF
STATIC_DT_MAX_CERTS = max(1, int(os.environ.get("INKLING_DTGRAPH_STATIC_DT_MAX_CERTS", "64") or "64"))
STATIC_DT_FORCE_GPU_BT = os.environ.get("INKLING_DTGRAPH_STATIC_DT_FORCE_GPU_BT", "0").strip().lower() not in _OFF
# Wave-V11 opt-in host-overhead micro-trims on certified static "dt"+"td" replay (default OFF = 0, 0-ULP exact!):
#   - Skips redundant TorchBackend.wait_slot/fence + dict(cert) copy on certified pure "td" replay (+ exec_gen guard).
#   - Caches 10 static GPU slice views + pre-filtered InputBatch kwargs template on certified "dt" replay.
#   - Pre-compiles AttentionMetadata refresh plan once at "dt" cert (eliminates per-step hasattr/id/cloned dict).
#   - Defers B=2..8 _dt_prep_ptr_sig warmup probe until static "dt" cert misses & avoids duplicate idx_tuple.
FAST_REPLAY_V11 = os.environ.get("INKLING_DTGRAPH_FAST_REPLAY_V11", "0").strip().lower() not in _OFF
# Wave-V14 opt-in zero-overhead host replay loop on certified static "dt"+"td" replay (default OFF = 0, 0-ULP exact!):
#   - Direct pre-bound cudaGraphExec_t launch with O(1) MRU stream/exec handles & pre-interned stat keys.
#   - Compiled flat zero-alloc pointer/batch validators for "dt" and unified single-pass "td" (eliminates
#     per-step nested ptr_sig tuples, _collect_dt_* traversals, duplicate spec queries, & set() allocs).
#   - 4-slot pre-allocated InputBatch NumPy/Tensor ring + positional dataclass constructor (0 dict copy/kwargs).
#   - 4-slot pre-cloned AttentionMetadata ring (0 copy.copy(), 0 48-key dict rebuilds, 0 sl_t.numpy()/np.max).
#   - Cached inputs_embeds & draft_tokens slice views + zero-alloc B=2..8 eligibility checks.
FAST_REPLAY_V14 = os.environ.get("INKLING_DTGRAPH_FAST_REPLAY_V14", "0").strip().lower() not in _OFF
# Wave-V16 opt-in sub-25us zero-alloc static replay + O(1) 2D Dynamic-K LUT + fused step_meta_stamp
# + SyncFree-Temp (temp > 0) static replay compatibility (strictly default OFF = 0, 0-ULP exact!):
#   - Direct integer-keyed 2D LUT `dt_exec_lut[b][k]` (B=1..32, K=0..15) & `td_exec_lut[b][k][slot]`
#     (0 tuple creation, 0 dict hashing, instant O(1) Dynamic-K K=7/11/15 switching + C=32 support!).
#   - Fused single-call `step_meta_stamp` updating pre-built 4-slot InputBatch + AttentionMetadata ring
#     slots in-place (0 InputBatch alloc, 0 AttentionMetadata copy/seq_lens_cpu churn, 4-slot non-aliasing!).
#   - Compiled straight-line per-cert validator + direct inline cudaGraphLaunch (0 loop/attrgetter/wrapper frames).
#   - Pre-validated V16 graph certification supporting BOTH temp == 0 (SyncFree greedy) AND temp > 0
#     (SyncFree-Temp zero-sync GPU rejection sampling, including static GPU draft_logits) fail-closed!
FAST_REPLAY_V16 = os.environ.get("INKLING_DTGRAPH_FAST_REPLAY_V16", "0").strip().lower() not in _OFF
STATIC_TEMP_REPLAY_V16 = os.environ.get(
    "INKLING_DTGRAPH_STATIC_TEMP_V16", "1" if FAST_REPLAY_V16 else "0"
).strip().lower() not in _OFF
_LUT_MAX_B = 32
_LUT_MAX_K = 15
_STAT_STATIC_REPLAY_DT = "static_replay:dt"
_STAT_STATIC_REPLAY_TD = "static_replay:td"
SLOT_BYTES = 64 * 1024
NSLOT = 4
ALIGN = 256

# ----------------------------------------------------------------------------------------------
# Op classification (pure; unit-tested on CPU with "meta" standing in for CUDA)
# ----------------------------------------------------------------------------------------------
PASS, PASS_CPU, STAGE, HAZ, PIN = "pass", "pass_cpu", "stage", "haz", "pin"

_SYNC_OPS = frozenset({
    "_local_scalar_dense", "item", "nonzero", "nonzero_numpy", "argwhere", "masked_select",
    "_unique", "_unique2", "unique_dim", "unique_consecutive", "unique_dim_consecutive",
    "repeat_interleave", "bincount", "histc", "equal", "is_nonzero", "allclose", "isin_sync",
    "_assert_scalar", "_print", "segment_reduce", "_nested_tensor_from_mask",
})
_RNG_OPS = frozenset({
    "uniform", "uniform_", "normal", "normal_", "exponential", "exponential_", "bernoulli",
    "bernoulli_", "random", "random_", "multinomial", "rand", "rand_like", "randn", "randn_like",
    "randint", "randint_like", "randperm", "geometric", "geometric_", "cauchy", "cauchy_",
    "log_normal", "log_normal_", "rrelu", "rrelu_with_noise", "dropout", "native_dropout",
    "poisson", "_standard_gamma", "_sample_dirichlet", "binomial", "feature_dropout",
    "alpha_dropout", "_fused_dropout",
})
_COLL_WORDS = (
    "all_reduce", "allreduce", "all_gather", "allgather", "reduce_scatter", "broadcast",
    "all_to_all", "alltoall", "send", "recv", "barrier", "gather",
)
_COLL_NS = frozenset({"c10d", "_c10d_functional", "_c10d_functional_autograd", "_dtensor"})
_INDEX_OPS = frozenset({
    "index", "index_put", "index_put_", "_index_put_impl_", "_unsafe_index", "_unsafe_index_put",
})
_VIEW_OPS = frozenset({
    "view", "_unsafe_view", "slice", "select", "as_strided", "alias", "t", "transpose", "permute",
    "unsqueeze", "squeeze", "expand", "narrow", "detach", "reshape", "_reshape_alias", "flatten",
    "unflatten", "view_as", "split", "split_with_sizes", "chunk", "unbind", "diagonal", "numpy_T",
    "lift_fresh", "dim", "size", "stride", "sym_size", "sym_stride", "sym_numel", "is_pinned",
})
_META_FACTORY_OPS = frozenset({
    "new_empty", "new_empty_strided", "new_zeros", "new_ones", "new_full",
    "empty_like", "zeros_like", "ones_like", "full_like",
})
_INT_BOOL_DTYPES = frozenset({
    torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
})


def _default_is_dev(x) -> bool:
    return getattr(x, "is_cuda", False)


def _tensors(obj, out):
    if isinstance(obj, torch.Tensor):
        out.append(obj)
    elif isinstance(obj, (list, tuple)):
        for o in obj:
            _tensors(o, out)
    elif isinstance(obj, dict):
        for o in obj.values():
            _tensors(o, out)
    return out


def _is_dev_obj(d, is_dev) -> bool:
    if d is None:
        return False
    if isinstance(d, str):
        d = torch.device(d)
    if isinstance(d, torch.device):
        return is_dev(torch.empty(0, device="meta")) if d.type == "meta" else d.type == "cuda"
    return False


def _is_cpu_obj(d) -> bool:
    if d is None:
        return False
    if isinstance(d, str):
        d = torch.device(d)
    return isinstance(d, torch.device) and d.type == "cpu"


def _dev_kw(kwargs, is_dev) -> bool:
    return _is_dev_obj(kwargs.get("device"), is_dev)


def _is_cpu(t) -> bool:
    return isinstance(t, torch.Tensor) and t.device.type == "cpu"


def _dtype_size(dtype) -> int:
    sz = getattr(dtype, "itemsize", None)
    return int(sz) if sz is not None else torch.empty(0, dtype=dtype).element_size()


_OP_NAME_CACHE: dict = {}


def op_name(func) -> tuple[str, str]:
    if DTGRAPH_FASTPATH:
        cached = _OP_NAME_CACHE.get(func)
        if cached is not None:
            return cached
    try:
        full = func._schema.name
    except AttributeError:
        return "", str(func)
    ns, _, name = full.partition("::")
    res = (ns, name)
    if DTGRAPH_FASTPATH and len(_OP_NAME_CACHE) < 4096:
        try:
            _OP_NAME_CACHE[func] = res
        except TypeError:
            pass
    return res


def _op_overload(func) -> str:
    try:
        return getattr(func._schema, "overload_name", "") or ""
    except AttributeError:
        return ""


def _parse_to(func, args, kwargs, is_dev):
    """Parse all PyTorch aten::to overloads (dtype_layout, device, other, dtype, prim_Device).
    Returns (ov, src, to_dev, to_cpu, target_dtype, pin_memory, nb, bad_layout)."""
    src = args[0]
    ov = _op_overload(func)
    if not ov:
        if (len(args) >= 2 and isinstance(args[1], torch.Tensor)) or isinstance(
            kwargs.get("other"), torch.Tensor
        ):
            ov = "other"
        elif len(args) >= 2 and isinstance(args[1], (torch.device, str)):
            ov = "device"
        elif len(args) >= 2 and isinstance(args[1], torch.dtype):
            ov = "dtype"
        else:
            ov = "dtype_layout"

    layout = kwargs.get("layout")
    mem_fmt = kwargs.get("memory_format")
    pin_memory = bool(kwargs.get("pin_memory") or False)
    if ov == "other":
        other = args[1] if len(args) >= 2 else kwargs.get("other")
        to_dev = is_dev(other) if isinstance(other, torch.Tensor) else False
        to_cpu = _is_cpu(other)
        dt = other.dtype if isinstance(other, torch.Tensor) else None
        nb = bool(args[2]) if len(args) > 2 else bool(kwargs.get("non_blocking", False))
        if len(args) > 4 and mem_fmt is None:
            mem_fmt = args[4]
    elif ov in ("device", "prim_Device"):
        d = args[1] if len(args) >= 2 else kwargs.get("device")
        to_dev = _is_dev_obj(d, is_dev)
        to_cpu = _is_cpu_obj(d)
        dt = args[2] if len(args) > 2 else kwargs.get("dtype")
        nb = bool(args[3]) if len(args) > 3 else bool(kwargs.get("non_blocking", False))
        if len(args) > 5 and mem_fmt is None:
            mem_fmt = args[5]
    elif ov == "dtype":
        to_dev = False
        to_cpu = False
        dt = args[1] if len(args) >= 2 else kwargs.get("dtype")
        nb = bool(args[2]) if len(args) > 2 else bool(kwargs.get("non_blocking", False))
        if len(args) > 4 and mem_fmt is None:
            mem_fmt = args[4]
    else:  # dtype_layout
        d = kwargs.get("device")
        to_dev = _is_dev_obj(d, is_dev)
        to_cpu = _is_cpu_obj(d)
        dt = kwargs.get("dtype")
        nb = bool(kwargs.get("non_blocking", False))

    bad_layout = (
        layout not in (None, torch.strided)
        or mem_fmt not in (None, torch.preserve_format, torch.contiguous_format)
    )
    return ov, src, to_dev, to_cpu, dt, pin_memory, nb, bad_layout


def classify(func, args, kwargs, is_dev=_default_is_dev, owned=lambda t: False) -> tuple[str, str]:
    kwargs = kwargs or {}
    ns, name = op_name(func)
    if ns == "aten" and name in _VIEW_OPS:
        return PASS, ""
    if name in ("empty", "empty_strided", "zeros") and kwargs.get("pin_memory"):
        return PIN, "pin_alloc"
    if name == "_pin_memory":
        return PIN, "pin_copy"
    ts = _tensors(args, [])
    _tensors(kwargs, ts)
    dev = [t for t in ts if is_dev(t)]
    devkw = _dev_kw(kwargs, is_dev)
    if ns in _COLL_NS or (ns == "vllm" and any(w in name for w in _COLL_WORDS)):
        return HAZ, f"collective:{ns}::{name}"
    if name in _RNG_OPS:
        return (HAZ, f"rng:{name}") if (dev or devkw) else (PASS_CPU, "")
    if name == "record_stream":
        return HAZ, "stream:record_stream"
    if ns == "aten" and name in _META_FACTORY_OPS and not kwargs.get("pin_memory"):
        return (PASS, "") if (dev or devkw) else (PASS_CPU, "")
    if name == "copy_" and (len(args) >= 2 or (len(args) == 1 and "src" in kwargs)):
        dst = args[0]
        src = args[1] if len(args) >= 2 else kwargs["src"]
        nb = bool(args[2]) if len(args) > 2 else bool(kwargs.get("non_blocking", False))
        if is_dev(dst) and _is_cpu(src):
            if not nb:
                return HAZ, "h2d_blocking"
            if src.dtype != dst.dtype:
                return HAZ, "h2d_cast"
            if owned(src):
                return PASS, ""
            if not src.is_contiguous():
                return HAZ, "h2d_noncontig"
            return STAGE, "h2d"
        if _is_cpu(dst) and is_dev(src):
            return HAZ, "d2h"
        return (PASS, "") if dev else (PASS_CPU, "")
    if name == "to" and args and isinstance(args[0], torch.Tensor):
        ov, src, to_dev, to_cpu, dt, pin_mem, nb, bad_layout = _parse_to(func, args, kwargs, is_dev)
        if _is_cpu(src) and to_dev:
            if bad_layout:
                return HAZ, "h2d_to_layout"
            if dt is not None and dt != src.dtype:
                if not (
                    isinstance(dt, torch.dtype)
                    and src.dtype in _INT_BOOL_DTYPES
                    and dt in _INT_BOOL_DTYPES
                ):
                    return HAZ, f"h2d_to_cast:{src.dtype}->{dt}"
                return STAGE, "h2d_to"
            if src.numel() == 0 and nb:
                return PASS, ""
            if owned(src) and nb and src.is_contiguous():
                return PASS, ""
            # Stage into persistent pinned Arena and upgrade non_blocking=True (0-ULP raw copy,
            # safe against caller CPU mutation and capture-illegal cudaStreamSynchronize).
            return STAGE, "h2d_to"
        if is_dev(src) and to_cpu:
            return HAZ, "d2h"
        if _is_cpu(src) and pin_mem and not to_dev:
            if bad_layout:
                return HAZ, "pin_layout"
            if dt is not None and dt != src.dtype and not (
                isinstance(dt, torch.dtype)
                and src.dtype in _INT_BOOL_DTYPES
                and dt in _INT_BOOL_DTYPES
            ):
                return HAZ, "pin_cast"
            return PIN, "pin_copy"
        return (PASS, "") if (is_dev(src) or to_dev) else (PASS_CPU, "")
    if name == "_to_copy" and args:
        src = args[0]
        to_dev = devkw
        to_cpu = _is_cpu_obj(kwargs.get("device"))
        nb = bool(kwargs.get("non_blocking", False))
        if _is_cpu(src) and to_dev:
            if not nb:
                return HAZ, "h2d_blocking"
            dt = kwargs.get("dtype")
            if dt is not None and dt != src.dtype:
                return HAZ, "h2d_cast"
            if owned(src):
                return PASS, ""
            if not src.is_contiguous():
                return HAZ, "h2d_noncontig"
            return STAGE, "h2d"
        if is_dev(src) and to_cpu:
            return HAZ, "d2h"
        return (PASS, "") if (dev or to_dev) else (PASS_CPU, "")
    if not dev and not devkw:
        return PASS_CPU, ""
    if name in _SYNC_OPS or (name == "where" and len(args) == 1):
        return HAZ, f"sync:{name}"
    if name in _INDEX_OPS:
        idx = args[1] if len(args) > 1 else kwargs.get("indices")
        for t in _tensors(idx, []):
            if t.dtype in (torch.bool, torch.uint8):
                return HAZ, "mask_index"
    for t in ts:
        if _is_cpu(t) and t.dim() > 0:
            return HAZ, f"cpu_arg:{ns}::{name}"
    return PASS, ""


# ----------------------------------------------------------------------------------------------
# Pinned staging arena
# ----------------------------------------------------------------------------------------------
class Arena:
    def __init__(self, slot_bytes: int = SLOT_BYTES, nslot: int = NSLOT, pin: bool | None = None):
        self.slot_bytes, self.nslot = slot_bytes, nslot
        pin = PIN_ARENA if pin is None else pin
        self.buf = torch.zeros(slot_bytes * nslot, dtype=torch.uint8, pin_memory=pin)
        self.slot, self.off = 0, 0

    def start(self, slot: int) -> None:
        self.slot, self.off = slot, 0

    def owns(self, t) -> bool:
        if not isinstance(t, torch.Tensor) or t.device.type != "cpu":
            return False
        lo = self.buf.data_ptr()
        return lo <= t.data_ptr() < lo + self.buf.numel()

    def alloc(self, shape, dtype) -> torch.Tensor | None:
        n = 1
        for d in shape:
            n *= int(d)
        nb = n * _dtype_size(dtype)
        a = (self.off + ALIGN - 1) // ALIGN * ALIGN
        if a + max(nb, 1) > self.slot_bytes:
            return None
        base = self.slot * self.slot_bytes + a
        self.off = a + nb
        return self.buf[base : base + nb].view(dtype).view(tuple(int(d) for d in shape))

    def take_cast(self, src: torch.Tensor, dtype=None) -> torch.Tensor | None:
        dt = src.dtype if dtype is None else dtype
        n = src.numel() * _dtype_size(dt)
        a = (self.off + ALIGN - 1) // ALIGN * ALIGN
        if a + max(n, 1) > self.slot_bytes:
            return None
        base = self.slot * self.slot_bytes + a
        view = self.buf[base : base + n].view(dt).view(src.shape)
        if n > 0:
            view.copy_(src)
        self.off = a + n
        return view

    def take(self, src: torch.Tensor) -> torch.Tensor | None:
        return self.take_cast(src)


def _contig_strides(shape) -> list[int]:
    st, acc = [], 1
    for d in reversed([int(x) for x in shape]):
        st.append(acc)
        acc *= max(d, 1)
    return st[::-1]


def pin_result(func, args, kwargs, arena: Arena):
    name = op_name(func)[1]
    if name == "_pin_memory":
        src = args[0]
        if not src.is_contiguous():
            return None
        return arena.take(src)
    if name == "to":
        _, src, _, _, dt, _, _, bad_layout = _parse_to(func, args, kwargs, _default_is_dev)
        if bad_layout:
            return None
        return arena.take_cast(src, dt if isinstance(dt, torch.dtype) else None)
    dtype = kwargs.get("dtype") or torch.get_default_dtype()
    if kwargs.get("layout") not in (None, torch.strided):
        return None
    dev = kwargs.get("device")
    if dev is not None and torch.device(dev).type != "cpu":
        return None
    shape = args[0]
    if name == "empty_strided" and list(args[1]) != _contig_strides(shape):
        return None
    out = arena.alloc(shape, dtype)
    if out is not None and name == "zeros":
        out.zero_()
    return out


def stage_args(func, args, kwargs, arena: Arena):
    name = op_name(func)[1]
    if name == "copy_":
        if len(args) >= 2:
            v = arena.take(args[1])
            return None if v is None else ((args[0], v, *args[2:]), kwargs)
        if "src" in kwargs:
            v = arena.take(kwargs["src"])
            if v is None:
                return None
            kw = dict(kwargs)
            kw["src"] = v
            return (args, kw)
    if name == "to":
        ov, src, _, _, dt, _, _, _ = _parse_to(func, args, kwargs, _default_is_dev)
        v = arena.take_cast(src, dt if isinstance(dt, torch.dtype) else None)
        if v is None:
            return None
        kw = dict(kwargs)
        if ov == "other" and len(args) > 2:
            kw.pop("non_blocking", None)
            return ((v, args[1], True, *args[3:]), kw)
        if ov in ("device", "prim_Device") and len(args) > 3:
            kw.pop("non_blocking", None)
            return ((v, args[1], args[2], True, *args[4:]), kw)
        kw["non_blocking"] = True
        return ((v, *args[1:]), kw)
    v = arena.take(args[0])
    return None if v is None else ((v, *args[1:]), kwargs)


# ----------------------------------------------------------------------------------------------
# Eligibility (pure)
# ----------------------------------------------------------------------------------------------
def dt_ineligible(
    *,
    cg_full: bool,
    num_ubatches: int,
    has_prefill: bool,
    pcp: bool,
    dcp_size: int,
    lora: bool,
    encoder_inputs: bool,
    encoder_decoder: bool,
    pp_single: bool,
    num_reqs: int = 1,
    max_reqs: int = 0,
) -> str | None:
    if not cg_full:
        return "not_full_graph"
    if num_ubatches > 1:
        return "ubatch"
    if has_prefill:
        return "prefill"
    if pcp:
        return "pcp"
    if dcp_size > 1:
        return "dcp"
    if lora:
        return "lora"
    if encoder_inputs or encoder_decoder:
        return "encoder"
    if not pp_single:
        return "pp"
    if max_reqs > 0 and num_reqs > max_reqs:
        return "max_reqs"
    return None


def td_ineligible(
    *,
    has_speculator: bool,
    has_full_draft_cg: bool,
    has_prefill: bool,
    pcp: bool,
    dcp_size: int,
    mm_inputs: bool,
    num_spec_tokens: int,
    context_kv_only: bool,
    conf_log: bool,
    vw_stats: bool,
    num_reqs: int = 1,
    max_reqs: int = 0,
) -> str | None:
    if not has_speculator:
        return "no_spec"
    if not has_full_draft_cg:
        return "no_draft_cg"
    if has_prefill:
        return "prefill"
    if pcp:
        return "pcp"
    if dcp_size > 1:
        return "dcp"
    if mm_inputs:
        return "mm_inputs"
    if num_spec_tokens <= 0 or context_kv_only:
        return "no_spec_tokens"
    if conf_log or vw_stats:
        return "logging_sync"
    if max_reqs > 0 and num_reqs > max_reqs:
        return "max_reqs"
    return None


# ----------------------------------------------------------------------------------------------
# Graph backend (upgraded: exec cache keyed by (region_name, shape_key, seg, slot))
# ----------------------------------------------------------------------------------------------
class TorchBackend:
    """torch.cuda.CUDAGraph per segment, private pool, thread-local capture, ExecUpdate."""

    def __init__(self, update: bool, max_execs: int = 128):
        self.update = update
        self.max_execs = max(0, int(max_execs))
        self.pool = torch.cuda.graph_pool_handle()
        self.keep: deque = deque(maxlen=16)
        self.execs: OrderedDict = OrderedDict()  # (region_name, shape_key, seg, slot) -> owner CUDAGraph
        self.exec_gen: dict[tuple, int] = {}
        self.events = [torch.cuda.Event() for _ in range(NSLOT)]
        self.recorded = [False] * NSLOT
        self.cudart = None
        self.update_hits = 0
        self.update_fail = 0
        self.static_replay_hits = 0
        self.evictions = 0
        self._exec_handles: dict[int, object] = {}
        self._stream_handles: dict[int, object] = {}
        self._default_streams: dict[int, int] = {}
        self._cuda_ok = None
        self._rt_launch = None
        self._last_cs: int = -1
        self._last_st_h = None
        self._last_mru_key = None
        if update:
            try:
                from cuda.bindings import runtime as cudart  # noqa: PLC0415

                self.cudart = cudart
                self._cuda_ok = cudart.cudaError_t.cudaSuccess
                self._rt_launch = cudart.cudaGraphLaunch
            except Exception as e:  # noqa: BLE001
                warnings.warn(
                    f"INKLING_DTGRAPH update: cuda.bindings unavailable ({e}); instantiate mode"
                )
                self.update = False

    def stream(self):
        return torch.cuda.current_stream()

    def is_default(self, s) -> bool:
        cs = s.cuda_stream
        if cs in (0, 1, 2):
            return True
        if DTGRAPH_FASTPATH:
            dev = s.device
            dkey = dev.index if getattr(dev, "index", None) is not None else -1
            def_s = self._default_streams.get(dkey)
            if def_s is None:
                def_s = torch.cuda.default_stream(dev).cuda_stream
                self._default_streams[dkey] = def_s
            return cs == def_s
        return cs == torch.cuda.default_stream(s.device).cuda_stream

    def wait_slot(self, slot: int) -> bool:
        ev = self.events[slot]
        if self.recorded[slot] and not _ORIG["Event.query"](ev):
            _ORIG["Event.synchronize"](ev)
            return True
        return False

    def begin(self):
        g = torch.cuda.CUDAGraph(keep_graph=self.update)
        g.capture_begin(pool=self.pool, capture_error_mode="thread_local")
        return g

    def fence(self, slot: int, stream) -> None:
        _ORIG["Event.record"](self.events[slot], stream)
        self.recorded[slot] = True

    def end_and_launch(self, g, key: tuple, stream) -> None:
        g.capture_end()
        self.keep.append(g)
        replay = _ORIG.get("CUDAGraph.replay", torch.cuda.CUDAGraph.replay)
        if not self.update:
            replay(g)
            return
        if self._update_launch(g, key, stream):
            self.update_hits += 1
            return
        g.instantiate()
        replay(g)
        old_owner = self.execs.get(key)
        if old_owner is not None:
            self._exec_handles.pop(id(old_owner), None)
        self.execs[key] = g
        self.exec_gen[key] = self.exec_gen.get(key, 0) + 1
        self.execs.move_to_end(key)
        self._last_mru_key = key
        if self.max_execs > 0:
            while len(self.execs) > self.max_execs:
                ek, evicted = self.execs.popitem(last=False)
                self._exec_handles.pop(id(evicted), None)
                self.exec_gen.pop(ek, None)
                if self._last_mru_key == ek:
                    self._last_mru_key = None
                self.evictions += 1

    def _update_launch(self, g, key: tuple, stream) -> bool:
        owner = self.execs.get(key)
        if owner is None or self.cudart is None:
            return False
        rt = self.cudart
        ok_code = self._cuda_ok if self._cuda_ok is not None else rt.cudaError_t.cudaSuccess
        try:
            if DTGRAPH_FASTPATH:
                oid = id(owner)
                ex = self._exec_handles.get(oid)
                if ex is None:
                    ex = rt.cudaGraphExec_t(init_value=owner.raw_cuda_graph_exec())
                    self._exec_handles[oid] = ex
                cs = stream.cuda_stream
                st_h = self._stream_handles.get(cs)
                if st_h is None:
                    st_h = rt.cudaStream_t(init_value=cs)
                    if len(self._stream_handles) < 64:
                        self._stream_handles[cs] = st_h
            else:
                ex = rt.cudaGraphExec_t(init_value=owner.raw_cuda_graph_exec())
                st_h = rt.cudaStream_t(init_value=stream.cuda_stream)
            gr = rt.cudaGraph_t(init_value=g.raw_cuda_graph())
            res = rt.cudaGraphExecUpdate(ex, gr)
            err = res[0] if isinstance(res, tuple) else res
            if err != ok_code:
                self.update_fail += 1
                return False
            res = rt.cudaGraphLaunch(ex, st_h)
            err = res[0] if isinstance(res, tuple) else res
            if err != ok_code:
                self.update_fail += 1
                return False
            self.execs.move_to_end(key)
            self._last_mru_key = key
            self.exec_gen[key] = self.exec_gen.get(key, 0) + 1
            return True
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"INKLING_DTGRAPH update path disabled ({type(e).__name__}: {e})")
            self.cudart = None
            return False

    def replay_static(self, key: tuple, stream) -> bool:
        """Directly launch cached cudaGraphExec_t without capture_begin/end or cudaGraphExecUpdate."""
        owner = self.execs.get(key)
        if owner is None:
            return False
        if self.cudart is not None:
            rt = self.cudart
            ok_code = self._cuda_ok if self._cuda_ok is not None else rt.cudaError_t.cudaSuccess
            try:
                oid = id(owner)
                ex = self._exec_handles.get(oid)
                if ex is None:
                    ex = rt.cudaGraphExec_t(init_value=owner.raw_cuda_graph_exec())
                    self._exec_handles[oid] = ex
                cs = stream.cuda_stream
                st_h = self._stream_handles.get(cs)
                if st_h is None:
                    st_h = rt.cudaStream_t(init_value=cs)
                    if len(self._stream_handles) < 64:
                        self._stream_handles[cs] = st_h
                res = rt.cudaGraphLaunch(ex, st_h)
                err = res[0] if isinstance(res, tuple) else res
                if err != ok_code:
                    return False
                self.execs.move_to_end(key)
                self.static_replay_hits += 1
                return True
            except Exception:  # noqa: BLE001
                return False
        replay = _ORIG.get("CUDAGraph.replay", torch.cuda.CUDAGraph.replay)
        replay(owner)
        self.execs.move_to_end(key)
        self._last_mru_key = key
        self.static_replay_hits += 1
        return True

    def get_static_exec_bundle(self, key: tuple):
        """Resolve (owner, cudaGraphExec_t_or_None, exec_gen) once at certification time."""
        owner = self.execs.get(key)
        if owner is None:
            return None
        gen = self.exec_gen.get(key)
        ex = None
        rt = self.cudart
        if rt is not None:
            oid = id(owner)
            ex = self._exec_handles.get(oid)
            if ex is None:
                try:
                    ex = rt.cudaGraphExec_t(init_value=owner.raw_cuda_graph_exec())
                    self._exec_handles[oid] = ex
                except Exception:  # noqa: BLE001
                    ex = None
        return owner, ex, gen

    def replay_static_v14(
        self,
        key: tuple,
        owner,
        ex_handle,
        expected_gen: int | None,
        stream,
    ) -> bool:
        """Zero-alloc direct static launch with O(1) MRU stream/exec handle caching (100% fail-closed)."""
        if self.execs.get(key) is not owner:
            return False
        if expected_gen is not None and self.exec_gen.get(key) != expected_gen:
            return False
        rt = self.cudart
        if rt is not None:
            ex = ex_handle
            if ex is None:
                oid = id(owner)
                ex = self._exec_handles.get(oid)
                if ex is None:
                    try:
                        ex = rt.cudaGraphExec_t(init_value=owner.raw_cuda_graph_exec())
                        self._exec_handles[oid] = ex
                    except Exception:  # noqa: BLE001
                        return False
            cs = stream.cuda_stream
            if cs == self._last_cs:
                st_h = self._last_st_h
            else:
                st_h = self._stream_handles.get(cs)
                if st_h is None:
                    try:
                        st_h = rt.cudaStream_t(init_value=cs)
                    except Exception:  # noqa: BLE001
                        return False
                    if len(self._stream_handles) < 64:
                        self._stream_handles[cs] = st_h
                self._last_cs = cs
                self._last_st_h = st_h
            try:
                launch_fn = self._rt_launch if self._rt_launch is not None else rt.cudaGraphLaunch
                res = launch_fn(ex, st_h)
            except Exception:  # noqa: BLE001
                return False
            err = res[0] if type(res) is tuple else res
            ok_code = self._cuda_ok
            if err != (ok_code if ok_code is not None else rt.cudaError_t.cudaSuccess):
                return False
        else:
            replay = _ORIG.get("CUDAGraph.replay", torch.cuda.CUDAGraph.replay)
            replay(owner)
        if self._last_mru_key is not key:
            self.execs.move_to_end(key)
            self._last_mru_key = key
        self.static_replay_hits += 1
        return True


# ----------------------------------------------------------------------------------------------
# Region state machine
# ----------------------------------------------------------------------------------------------
_TLS = threading.local()
_ORIG: dict = {}


class Region:
    """Capture region state machine shared by DT (Phase 4) and TD (Phase 2B)."""

    def __init__(
        self, backend, *, max_seg: int = 4, arena: Arena | None = None, is_dev=_default_is_dev
    ):
        self.b = backend
        self.max_seg = max_seg
        self.arena = arena if arena is not None else Arena()
        self.is_dev = is_dev
        self.active = False
        self.capturing = False
        self.region_name = ""
        self.shape_key: tuple = ()
        self.g = None
        self.seg = 0
        self.latch = False
        self.want = False
        self.slot = 0
        self.stream = None
        self.n = 0
        self.stats: Counter = Counter()
        self.immutable_ptrs: set[int] = set()
        self.dt_hook_depth: int = 0
        self.dt_track_unhooked: bool = False
        self.dt_unhooked_dev_op: bool = False

    def _owns_cpu_src(self, t) -> bool:
        if self.arena.owns(t):
            return True
        if (
            self.immutable_ptrs
            and isinstance(t, torch.Tensor)
            and t.device.type == "cpu"
            and int(t.data_ptr()) in self.immutable_ptrs
        ):
            return True
        return False

    def begin(
        self,
        stream,
        region_name: str = "dt",
        shape_key: tuple = (),
        *,
        force_slot: int | None = None,
    ) -> bool:
        if self.active:
            raise RuntimeError(
                f"INKLING_DTGRAPH: region {self.region_name!r} still open when starting "
                f"{region_name!r}; refusing to continue"
            )
        self.slot = (int(force_slot) % NSLOT) if force_slot is not None else (self.n % NSLOT)
        if self.b.wait_slot(self.slot):
            self.stats["slot_wait"] += 1
        self.arena.start(self.slot)
        self.stream = stream
        self.region_name = region_name
        self.shape_key = shape_key
        self.seg, self.latch, self.want = 0, False, False
        self.dt_unhooked_dev_op = False
        self.g = self.b.begin()
        self.capturing = True
        self.active = True
        _TLS.region = self
        self.stats[f"regions:{region_name}"] += 1
        return True

    def end(self) -> None:
        if not self.active:
            return
        try:
            if self.capturing:
                self._close()
            self.b.fence(self.slot, self.stream)
        finally:
            self.active = False
            self.capturing = False
            self.want = False
            self.latch = False
            self.dt_track_unhooked = False
            self.region_name = ""
            _TLS.region = None
            self.n += 1

    def abort(self) -> None:
        if self.capturing:
            try:
                self.g.capture_end()
            except Exception:  # noqa: BLE001
                pass
        self.active = False
        self.capturing = False
        self.want = False
        self.latch = False
        self.dt_track_unhooked = False
        self.region_name = ""
        _TLS.region = None

    def replay_static_region(
        self,
        stream,
        region_name: str,
        shape_key: tuple,
        slot: int,
        *,
        wait_slot: bool = True,
        expected_exec_gen: int | None = None,
    ) -> bool:
        """Replay a certified single-segment pure static graph with zero TorchDispatchMode & zero ExecUpdate."""
        if self.active:
            return False
        s_idx = int(slot) % NSLOT
        key = (region_name, shape_key, 0, s_idx)
        if expected_exec_gen is not None:
            gen_map = getattr(self.b, "exec_gen", None)
            if gen_map is not None and gen_map.get(key) != expected_exec_gen:
                return False
        if wait_slot and self.b.wait_slot(s_idx):
            self.stats["slot_wait"] += 1
        if not self.b.replay_static(key, stream):
            return False
        self.stats[f"static_replay:{region_name}"] += 1
        self.n += 1
        return True

    def replay_static_region_v14(
        self,
        stream,
        stat_key: str,
        exec_key: tuple,
        exec_owner,
        exec_handle,
        expected_exec_gen: int | None,
    ) -> bool:
        """Zero-alloc static replay using pre-resolved exec bundle & pre-interned stat key."""
        if self.active:
            return False
        if not self.b.replay_static_v14(exec_key, exec_owner, exec_handle, expected_exec_gen, stream):
            return False
        self.stats[stat_key] += 1
        self.n += 1
        return True

    def _close(self) -> None:
        g, self.g = self.g, None
        self.capturing = False
        key = (self.region_name, self.shape_key, self.seg, self.slot)
        self.b.end_and_launch(g, key, self.stream)
        self.stats[f"segments:{self.region_name}"] += 1
        self.seg += 1

    def hazard(self, why: str, latch: bool = False) -> None:
        self.stats[f"haz:{self.region_name}:{why}"] += 1
        if latch:
            self.latch = True
        if self.capturing:
            self._close()
        self.want = not self.latch

    def maybe_resume(self) -> None:
        if (
            self.want
            and not self.latch
            and not self.capturing
            and self.seg < self.max_seg
            and self.b.stream().cuda_stream == self.stream.cuda_stream
        ):
            self.want = False
            self.g = self.b.begin()
            self.capturing = True

    def dispatch(self, func, args, kwargs):
        if not self.active:
            return func(*args, **kwargs)
        if not self.capturing:
            if self.want:
                self.maybe_resume()
            if not self.capturing:
                return func(*args, **kwargs)
        if DTGRAPH_FASTPATH:
            ns, name = _OP_NAME_CACHE.get(func) or op_name(func)
            if ns == "aten" and name in _VIEW_OPS:
                return func(*args, **kwargs)
        else:
            _, name = op_name(func)
        # Elide identical-buffer self-copy (e.g. inputs_embeds[:n].copy_(embeds) after direct embed write).
        if name == "copy_" and len(args) >= 2:
            dst, src = args[0], args[1]
            if (
                isinstance(dst, torch.Tensor)
                and isinstance(src, torch.Tensor)
                and dst.device == src.device
                and dst.dtype == src.dtype
                and dst.shape == src.shape
                and dst.stride() == src.stride()
                and dst.data_ptr() == src.data_ptr()
            ):
                self.stats["elided_self_copy"] += 1
                return dst
        owned_fn = self._owns_cpu_src if self.immutable_ptrs else self.arena.owns
        act, why = classify(func, args, kwargs, self.is_dev, owned_fn)
        if self.dt_track_unhooked and self.dt_hook_depth == 0 and act != PASS_CPU:
            self.dt_unhooked_dev_op = True
        if act in (PASS, PASS_CPU):
            return func(*args, **kwargs)
        if act == PIN:
            out = pin_result(func, args, kwargs, self.arena)
            if out is not None:
                self.stats["pinned"] += 1
                return out
            why = "arena_full"
        if act == STAGE:
            st = stage_args(func, args, kwargs, self.arena)
            if st is not None:
                self.stats["staged"] += 1
                return func(*st[0], **st[1])
            why = "arena_full"
        self.hazard(why)
        return func(*args, **kwargs)


def current_region():
    return getattr(_TLS, "region", None)


# ----------------------------------------------------------------------------------------------
# Python-level hazard hooks (installed once, only when INKLING_DTGRAPH is on)
# ----------------------------------------------------------------------------------------------
def _hz(orig, why, latch):
    def w(*a, **k):
        r = getattr(_TLS, "region", None)
        if r is not None:
            r.hazard(why, latch=latch)
        return orig(*a, **k)

    w.__wrapped__ = orig
    w._inkling_dtg = True
    return w


def _install_hooks() -> None:
    if _ORIG.get("_installed"):
        return
    E, S = torch.cuda.Event, torch.cuda.Stream
    for cls, cname, names in (
        (E, "Event", ("record", "wait", "synchronize", "query")),
        (S, "Stream", ("wait_stream", "wait_event", "record_event", "synchronize", "query")),
    ):
        for n in names:
            orig = getattr(cls, n)
            _ORIG[f"{cname}.{n}"] = orig
            setattr(cls, n, _hz(orig, f"{cname.lower()}:{n}", True))
    _ORIG["synchronize"] = torch.cuda.synchronize
    torch.cuda.synchronize = _hz(torch.cuda.synchronize, "sync:device", True)
    orig_set = torch.cuda.set_stream
    _ORIG["set_stream"] = orig_set

    def set_stream(stream):
        r = getattr(_TLS, "region", None)
        if r is not None and stream is not None and stream.cuda_stream != r.stream.cuda_stream:
            r.hazard("stream:set", latch=True)
        return orig_set(stream)

    set_stream._inkling_dtg = True
    torch.cuda.set_stream = set_stream
    _ORIG["CUDAGraph.replay"] = torch.cuda.CUDAGraph.replay
    torch.cuda.CUDAGraph.replay = _hz(torch.cuda.CUDAGraph.replay, "graph_replay", True)
    try:
        from vllm.distributed.parallel_state import GroupCoordinator  # noqa: PLC0415

        for n in (
            "all_reduce", "all_gather", "all_gatherv", "reduce_scatter", "reduce_scatterv",
            "reduce_scatterv_into_output", "gather", "broadcast", "broadcast_tensor_dict",
            "send", "recv", "send_tensor_dict", "recv_tensor_dict", "barrier",
        ):
            if hasattr(GroupCoordinator, n):
                setattr(GroupCoordinator, n, _hz(getattr(GroupCoordinator, n), "collective:" + n, False))
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"INKLING_DTGRAPH: GroupCoordinator hooks not installed ({e})")
    try:
        from triton import knobs  # noqa: PLC0415

        prev = knobs.runtime.jit_cache_hook

        def cache_hook(*a, **k):
            r = getattr(_TLS, "region", None)
            if r is not None:
                r.hazard("triton_compile", latch=False)
            return prev(*a, **k) if prev is not None else None

        knobs.runtime.jit_cache_hook = cache_hook

        def load_hook(*a, **k):
            r = getattr(_TLS, "region", None)
            if r is not None:
                r.hazard("triton_load", latch=False)

        hc = knobs.runtime.kernel_load_start_hook
        if hasattr(hc, "add"):
            hc.add(load_hook)
        elif hc is None:
            knobs.runtime.kernel_load_start_hook = load_hook
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"INKLING_DTGRAPH: Triton hooks not installed ({e})")
    warnings.filterwarnings("ignore", message=".*The CUDA Graph is empty.*")
    _ORIG["_installed"] = True


def _hook_rdma() -> None:
    if _ORIG.get("_rdma"):
        return
    import sys  # noqa: PLC0415

    m = sys.modules.get("vllm.distributed.device_communicators.rdma_all_reduce")
    if m is None:
        return
    cls = getattr(m, "RdmaAllReduce", None) or next(
        (v for v in vars(m).values() if isinstance(v, type) and hasattr(v, "all_gather_stacked")),
        None,
    )
    if cls is not None:
        for n in (
            "all_reduce",
            "all_reduce_add_rmsnorm",
            "all_gather_stacked",
            "all_gather_stacked_cond",
            "all_gather_cols",
        ):
            if hasattr(cls, n):
                setattr(cls, n, _hz(getattr(cls, n), "collective:rdma_" + n, False))
    _ORIG["_rdma"] = True


# ----------------------------------------------------------------------------------------------
# Lazy GPUModelRunner + CUDAGraphManager integration (ZERO edits to model_runner.py!)
# ----------------------------------------------------------------------------------------------
class _Mode:
    cls = None

    @classmethod
    def make(cls, region):
        if cls.cls is None:
            from torch.utils._python_dispatch import TorchDispatchMode  # noqa: PLC0415

            class _DTGMode(TorchDispatchMode):
                def __init__(self, r):
                    super().__init__()
                    self.r = r

                def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                    return self.r.dispatch(func, args, kwargs or {})

            cls.cls = _DTGMode
        return cls.cls(region)


class _RunnerState:
    def __init__(self):
        self.region: Region | None = None
        self.mode = None
        self.mode_obj = None
        self.seen: Counter = Counter()
        self.skips: Counter = Counter()
        self.disabled = False
        self.rank0 = True
        self.cur_sample_ctx: dict | None = None
        self.cur_ib = None
        self.cur_num_spec: int = 0
        # Lever A (INKLING_FUSED_DT_PREP=1): self-probed dtypes + persistent B=1 / B=2..N GPU buffers
        self.b1_dtypes: tuple | None = None
        self.b1_bufs: dict[str, torch.Tensor] | None = None
        self.b1_compiled: set[tuple] = set()
        self.bn_bufs: dict[str, torch.Tensor] | None = None
        self.bn_compiled: set[tuple] = set()
        self.bn_inline_compiled: set[tuple] = set()
        self.b1_failed = False
        self.b1_ib_cls = None
        self.b1_ib_fields: set[str] | None = None
        self.b1_ib_drop: tuple[str, ...] = ()
        # Lever B (INKLING_FUSED_EMBED_DIRECT=1): discovered static inputs_embeds buffer
        self.inputs_embeds_buf: torch.Tensor | None = None
        # Lever C (INKLING_FUSED_DRAFT_STORE=1): real req_states.draft_tokens saved across propose->set_draft_tokens
        self.draft_store_real: torch.Tensor | None = None
        self.draft_store_active = False
        self.draft_proxy = None
        # Wave-V8 opt-in host-overhead caches & ROWAR tail-swap state (all inactive when flags=0)
        self.exec_hooks_done = False
        self.sample_hooks_done = False
        self.write_pad: bool | None = None
        self.np_qsl_cache: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]] = {}
        self.rowar_tracker: _RowarAccTracker | None = None
        self.rowar_order_safe: bool | None = None
        # Wave-V9 Phase-2: 4-slot static ring alignment & certified static-replay state
        self.td_static_cert: dict[tuple, dict] = {}
        self.td_static_armed: dict | None = None
        self.td_static_stream = None
        self.td_cap_ctx: dict | None = None
        self.cur_ring_slot: int | None = None
        self.cur_ring_fenced: bool = True
        # Wave-V10 Phase-4: Certified zero-ExecUpdate static replay for "dt" (Drafter->Target)
        self.dt_static_cert: OrderedDict[tuple, dict] = OrderedDict()
        self.dt_static_armed: dict | None = None
        self.dt_cap_ctx: dict | None = None
        self.dt_immutable_idx_bufs: OrderedDict[tuple, torch.Tensor] = OrderedDict()
        self.dt_bt_mode: str | None = None
        self.dt_sconv_builders: list | None = None
        # Wave-V14 host fast-replay caches (strictly inactive unless INKLING_DTGRAPH_FAST_REPLAY_V14=1)
        self._last_nondef_cs: int = -1
        self._sf_ring_mod = None
        self._sf_ring_mod_probed: bool = False
        self._tree_spec_mod = None
        self._tree_spec_mod_probed: bool = False
        self._dt_sconv_shared: tuple = ()
        self.dt_last_cert: dict | None = None
        self.dt_last_sl: tuple | None = None
        self._orig_qrfg = None
        self._td_last_cert_by_slot: list = [None] * NSLOT
        self._td_spec_cache: tuple | None = None
        # Wave-V16 sub-25us host replay caches (strictly inactive unless INKLING_DTGRAPH_FAST_REPLAY_V16=1)
        self.dt_exec_lut: list[list] = [[None] * (_LUT_MAX_K + 1) for _ in range(_LUT_MAX_B + 1)]
        self.dt_fn_lut: list[list] = [[None] * (_LUT_MAX_K + 1) for _ in range(_LUT_MAX_B + 1)]
        self.td_exec_lut: list[list[list]] = [
            [[None] * NSLOT for _ in range(_LUT_MAX_K + 1)] for _ in range(_LUT_MAX_B + 1)
        ]
        self.td_fn_lut: list[list[list]] = [
            [[None] * NSLOT for _ in range(_LUT_MAX_K + 1)] for _ in range(_LUT_MAX_B + 1)
        ]
        self._dt_v16_pa_ready = None
        self._dt_v16_ie_ready = None
        self._dt_v16_ie_meta = None
        self._dt_v16_hit: bool = False
        self._td_v16_armed = None
        self._td_v16_hit: bool = False
        self._sf_clear_fn = None
        self._sf_get_ring_fn = None
        self._sf_fence_fn = None
        self._v16_imm_idx_by_ptr: dict[int, torch.Tensor] = {}
        try:
            from vllm.distributed import get_tensor_model_parallel_rank  # noqa: PLC0415

            self.rank0 = get_tensor_model_parallel_rank() == 0
        except Exception:  # noqa: BLE001
            pass


# ----------------------------------------------------------------------------------------------
# Wave-V8 Opt-In ROWAR Tail-Swap (INKLING_ROWAR_DESC_W_ORDER=1, fail-closed, TP8-lockstep)
# ----------------------------------------------------------------------------------------------
# Layout proof: in vLLM's target batch [M_cap], requests are strictly interleaved at fixed stride
# (K + 1), so _n_live_kernel computes n_live = (B - 1) * (K + 1) + 1 + W_{B-1}. Notice n_live
# depends ONLY on the LAST request's verify width W_{B-1} (MoE already skips all interior dead
# rows via is_padding |= vw_mask). Therefore a single O(B) swap placing the lowest-E[acc] slot
# at index B-1 achieves the EXACT same ROWAR ceiling as a full descending-W sort!
# To guarantee 100% TP8 lockstep with 0 GPU syncs, acceptance EMA is tracked strictly from CPU
# num_computed_tokens_np deltas using deterministic Q8 integer fixed-point arithmetic.
class _RowarAccTracker:
    __slots__ = ("last_comp", "ema_q8", "obs")

    def __init__(self):
        self.last_comp: dict[int, int] = {}
        self.ema_q8: dict[int, int] = {}
        self.obs: dict[int, int] = {}

    def update(self, slots_np, req_comp_np, max_step_delta: int) -> None:
        for i in range(len(slots_np)):
            s = int(slots_np[i])
            c = int(req_comp_np[s])
            prev = self.last_comp.get(s)
            if prev is not None:
                delta = c - prev
                if 1 <= delta <= max_step_delta:
                    acc = delta - 1
                    pe = self.ema_q8.get(s)
                    self.ema_q8[s] = (acc << 8) if pe is None else ((3 * pe + (acc << 8) + 2) >> 2)
                    self.obs[s] = self.obs.get(s, 0) + 1
                elif delta < 0 or delta > max_step_delta:
                    # Slot was reused by a new request (prefill jump or counter reset).
                    self.ema_q8.pop(s, None)
                    self.obs.pop(s, None)
            self.last_comp[s] = c


def _rowar_select_tail_swap(
    slots_np, tracker: _RowarAccTracker, margin_q8: int, min_obs: int = 2
) -> int:
    """Return index i in [0, B-2] to swap with tail B-1, or -1 if no swap improves ROWAR.
    100% deterministic integer math + lowest-index tie-break => identical on all TP8 ranks."""
    n = len(slots_np)
    if n < 2 or n > 8:
        return -1
    s_tail = int(slots_np[n - 1])
    if tracker.obs.get(s_tail, 0) < min_obs:
        return -1
    tail_ema = tracker.ema_q8.get(s_tail)
    if tail_ema is None:
        return -1
    best_i = -1
    best_ema = tail_ema
    for i in range(n - 1):
        si = int(slots_np[i])
        if tracker.obs.get(si, 0) < min_obs:
            continue
        ei = tracker.ema_q8.get(si)
        if ei is not None and ei < best_ema:
            best_ema = ei
            best_i = i
    if best_i >= 0 and (tail_ema - best_ema) >= margin_q8:
        return best_i
    return -1


def _probe_rowar_order_safe(runner) -> bool:
    """Fail-closed source probe verifying GPUModelRunner binds post-prepare_inputs outputs
    exclusively to input_batch (not batch_req_state) before permitting request-tail swaps."""
    if ROWAR_DESC_W_FORCE:
        return True
    try:
        import inspect  # noqa: PLC0415

        cls = type(runner)
        exec_src = inspect.getsource(cls.execute_model)
        samp_src = inspect.getsource(cls.sample_tokens)
        # Must construct ModelRunnerOutput / req_ids from input_batch.req_ids, and must NOT
        # retain batch_req_state past prepare_inputs into execute_model_state or sample_tokens.
        if "input_batch.req_ids" not in samp_src and "input_batch.req_ids" not in exec_src:
            return False
        if "batch_req_state" in samp_src:
            return False
        if "batch_req_state=" in exec_src and "ExecuteModelState" in exec_src:
            return False
        return True
    except Exception:  # noqa: BLE001
        return False


def _get_write_pad(st: _RunnerState) -> bool:
    if DTGRAPH_FASTPATH and st.write_pad is not None:
        return st.write_pad
    from vllm import envs  # noqa: PLC0415

    wp = bool(envs.VLLM_MOE_SKIP_PADDING)
    if DTGRAPH_FASTPATH:
        st.write_pad = wp
    return wp


def _get_cached_qsl_np(
    st: _RunnerState, num_reqs: int, q_len: int, num_reqs_padded: int
) -> tuple[np.ndarray, np.ndarray]:
    key = (num_reqs, q_len, num_reqs_padded)
    cached = st.np_qsl_cache.get(key)
    if cached is not None:
        return cached
    num_tokens = num_reqs * q_len
    cu_np = np.arange(num_reqs + 1, dtype=np.int32) * q_len
    qsl_np = np.full(num_reqs_padded + 1, num_tokens, dtype=np.int32)
    qsl_np[: num_reqs + 1] = cu_np
    if len(st.np_qsl_cache) < 256:
        st.np_qsl_cache[key] = (cu_np, qsl_np)
    return cu_np, qsl_np


# ----------------------------------------------------------------------------------------------
# Lever A: Fused 1-CTA Triton B=1 pure-decode prepare_inputs (zero H2D DMA, 0-ULP bit-exact)
# ----------------------------------------------------------------------------------------------
_FUSED_DT_PREP_KERNEL = None


def _get_fused_dt_prep_kernel():
    global _FUSED_DT_PREP_KERNEL
    if _FUSED_DT_PREP_KERNEL is not None:
        return _FUSED_DT_PREP_KERNEL
    from vllm.triton_utils import tl, triton  # noqa: PLC0415

    @triton.jit(
        do_not_specialize=[
            "slot0",
            "num_tokens",
            "num_tokens_padded",
            "qsl_len",
            "sl_len",
            "n_logits",
            "draft_stride",
            "last_sampled_stride",
        ]
    )
    def _fused_dt_prep_b1_kernel(
        is_padding_ptr,
        idx_mapping_ptr,
        cu_num_logits_ptr,
        query_start_loc_ptr,
        expanded_idx_ptr,
        expanded_local_pos_ptr,
        positions_ptr,
        seq_lens_ptr,
        input_ids_ptr,
        logits_indices_ptr,
        num_computed_ptr,
        last_sampled_ptr,
        draft_tokens_ptr,
        slot0,
        num_tokens,
        num_tokens_padded,
        qsl_len,
        sl_len,
        n_logits,
        draft_stride,
        last_sampled_stride,
        WRITE_IS_PADDING: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        """1-CTA fused replacement for is_padding.fill_ + 3x async_tensor_h2d +
        expand_idx_mapping + prepare_pos_seq_lens + combine_sampled_and_draft_tokens
        on B=1 pure-decode steps (zero DMA HtoD transitions, 100% bit-exact)."""
        s0 = slot0.to(tl.int64)
        nt = num_tokens.to(tl.int32)
        num_computed = tl.load(num_computed_ptr + s0).to(tl.int64)
        last_tok = tl.load(last_sampled_ptr + s0 * last_sampled_stride.to(tl.int64)).to(tl.int64)

        tl.store(idx_mapping_ptr, s0.to(idx_mapping_ptr.dtype.element_ty))
        tl.store(cu_num_logits_ptr, tl.zeros((), dtype=cu_num_logits_ptr.dtype.element_ty))
        tl.store(cu_num_logits_ptr + 1, n_logits.to(cu_num_logits_ptr.dtype.element_ty))

        seq_len0 = (num_computed + nt).to(seq_lens_ptr.dtype.element_ty)
        for off in range(0, qsl_len, BLOCK_R):
            r = off + tl.arange(0, BLOCK_R)
            tl.store(
                query_start_loc_ptr + r,
                tl.where(r == 0, 0, nt).to(query_start_loc_ptr.dtype.element_ty),
                mask=r < qsl_len,
            )
        for off in range(0, sl_len, BLOCK_R):
            r = off + tl.arange(0, BLOCK_R)
            tl.store(
                seq_lens_ptr + r,
                tl.where(r == 0, seq_len0, 0).to(seq_lens_ptr.dtype.element_ty),
                mask=r < sl_len,
            )

        t = tl.arange(0, BLOCK_T)
        if WRITE_IS_PADDING:
            tl.store(is_padding_ptr + t, t >= num_tokens, mask=t < num_tokens_padded)
        m_tok = t < num_tokens
        tl.store(
            positions_ptr + t,
            (num_computed + t).to(positions_ptr.dtype.element_ty),
            mask=m_tok,
        )
        tl.store(
            expanded_idx_ptr + t,
            (tl.zeros((BLOCK_T,), dtype=tl.int64) + s0).to(expanded_idx_ptr.dtype.element_ty),
            mask=m_tok,
        )
        tl.store(
            expanded_local_pos_ptr + t,
            t.to(expanded_local_pos_ptr.dtype.element_ty),
            mask=m_tok,
        )
        tl.store(
            logits_indices_ptr + t,
            t.to(logits_indices_ptr.dtype.element_ty),
            mask=m_tok,
        )
        d_idx = tl.maximum(t - 1, 0).to(tl.int64)
        draft_tok = tl.load(
            draft_tokens_ptr + s0 * draft_stride.to(tl.int64) + d_idx,
            mask=(t >= 1) & m_tok,
            other=0,
        ).to(tl.int64)
        inp_tok = tl.where(t == 0, last_tok, draft_tok)
        tl.store(
            input_ids_ptr + t,
            inp_tok.to(input_ids_ptr.dtype.element_ty),
            mask=m_tok,
        )

    _FUSED_DT_PREP_KERNEL = (triton, _fused_dt_prep_b1_kernel)
    return _FUSED_DT_PREP_KERNEL


def _b1_prep_eligible(runner, scheduler_output, batch_req_state, batch_desc) -> bool:
    if not FUSED_DT_PREP:
        return False
    if len(batch_req_state.req_ids) != 1 or bool(batch_req_state.has_prefill):
        return False
    if not getattr(scheduler_output, "scheduled_spec_decode_tokens", None):
        return False
    if getattr(runner.model_state, "num_new_sampled_tokens_per_step", 0) != 1:
        return False
    if getattr(runner, "adaptive_verification", None) is not None:
        return False
    if getattr(runner, "fast_prefill", None) is not None:
        return False
    if getattr(runner.model_config, "rswa_window", None) is not None:
        return False
    if getattr(runner, "pcp_manager", None) is not None:
        return False
    num_tokens = int(batch_req_state.num_tokens)
    if num_tokens <= 0:
        return False
    num_tokens_after_padding = max(num_tokens, int(batch_desc.num_tokens))
    if num_tokens_after_padding > 256:
        return False
    nst_np = batch_req_state.num_scheduled_tokens
    if int(nst_np[0]) != num_tokens:
        return False
    nd_np = batch_req_state.num_draft_tokens_np
    if nd_np is None or int(nd_np[0]) + 1 != num_tokens:
        return False
    # Guarantee decode past prefill boundary (so combine_sampled_and_draft_tokens always reads last_sampled + drafts).
    if int(batch_req_state.num_computed_prefill_tokens_np[0]) < int(batch_req_state.prefill_len_np[0]):
        return False
    return getattr(runner, "device", None) is not None and runner.device.type == "cuda"


def _ensure_b1_bufs(st: _RunnerState, device: torch.device, min_tok: int) -> bool:
    if st.b1_dtypes is None:
        return False
    need = max(64, int(min_tok))
    if st.b1_bufs is not None and st.b1_bufs["expanded_idx"].shape[0] >= need:
        return True
    if torch.cuda.is_current_stream_capturing():
        return False
    dt_idx, dt_cu, dt_eidx, dt_epos, dt_li = st.b1_dtypes
    st.b1_bufs = {
        "idx": torch.empty((1,), dtype=dt_idx, device=device),
        "cu": torch.empty((2,), dtype=dt_cu, device=device),
        "expanded_idx": torch.empty((need,), dtype=dt_eidx, device=device),
        "expanded_pos": torch.empty((need,), dtype=dt_epos, device=device),
        "logits_indices": torch.empty((need,), dtype=dt_li, device=device),
    }
    return True


def _probe_b1_from_ib(st: _RunnerState, ib) -> None:
    if st.b1_dtypes is not None or ib is None:
        return
    try:
        cls = type(ib)
        if not dataclasses.is_dataclass(cls):
            st.b1_failed = True
            return
        fields = {f.name for f in dataclasses.fields(cls)}
        required = {
            f.name
            for f in dataclasses.fields(cls)
            if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
        }
        known = {
            "req_ids", "num_reqs", "num_reqs_after_padding", "idx_mapping", "idx_mapping_np",
            "expanded_idx_mapping", "expanded_local_pos", "num_scheduled_tokens", "num_tokens",
            "num_tokens_after_padding", "num_draft_tokens", "num_draft_tokens_per_req",
            "num_admitted_draft_tokens", "query_start_loc", "query_start_loc_np", "seq_lens",
            "seq_lens_cpu_upper_bound", "dcp_local_seq_lens", "num_computed_tokens_np",
            "prefill_len_np", "num_computed_prefill_tokens_np", "is_prefilling_np", "has_prefill",
            "prefill_runs_as_decode_np", "decode_graph_eligible", "max_seq_len_np", "input_ids",
            "positions", "is_padding", "logits_indices", "cu_num_logits", "cu_num_logits_np",
            "has_structured_output_reqs", "prompt_lens", "fast_prefill", "max_query_len",
        }
        if not required.issubset(known):
            st.b1_failed = True
            return
        st.b1_ib_cls = cls
        st.b1_ib_fields = fields
        st.b1_ib_drop = tuple(sorted(known - fields))
        st.b1_dtypes = (
            ib.idx_mapping.dtype,
            ib.cu_num_logits.dtype,
            ib.expanded_idx_mapping.dtype,
            ib.expanded_local_pos.dtype,
            ib.logits_indices.dtype,
        )
    except Exception:  # noqa: BLE001
        st.b1_failed = True


def _try_fused_prepare_inputs_b1(
    runner,
    st: _RunnerState,
    scheduler_output,
    batch_req_state,
    batch_desc,
    *,
    skip_elig_check: bool = False,
):
    if st.b1_failed or st.b1_dtypes is None or st.b1_ib_cls is None or st.b1_ib_fields is None:
        return None
    if not skip_elig_check and not _b1_prep_eligible(
        runner, scheduler_output, batch_req_state, batch_desc
    ):
        return None
    num_tokens = int(batch_req_state.num_tokens)
    num_tokens_after_padding = max(num_tokens, int(batch_desc.num_tokens))
    if not _ensure_b1_bufs(st, runner.device, num_tokens_after_padding):
        return None
    write_pad = _get_write_pad(st)
    triton, kernel = _get_fused_dt_prep_kernel()
    block_r = 256
    block_t = max(16, triton.next_power_of_2(num_tokens_after_padding))
    ckey = (write_pad, block_r, block_t)
    if ckey not in st.b1_compiled and torch.cuda.is_current_stream_capturing():
        return None

    req_ids = batch_req_state.req_ids
    num_scheduled_tokens_np = batch_req_state.num_scheduled_tokens
    idx_mapping_np = batch_req_state.idx_mapping_np
    slot0 = int(idx_mapping_np[0])
    num_draft_tokens_per_req = batch_req_state.num_draft_tokens_np
    total_num_draft_tokens = int(num_draft_tokens_per_req[0])
    total_num_logits = total_num_draft_tokens + 1
    num_reqs_padded = int(batch_desc.num_reqs or 1)

    bufs = st.b1_bufs
    assert bufs is not None
    idx_mapping = bufs["idx"]
    cu_num_logits = bufs["cu"]
    expanded_idx_mapping = bufs["expanded_idx"][:total_num_logits]
    expanded_local_pos = bufs["expanded_pos"][:total_num_logits]
    logits_indices = bufs["logits_indices"][:total_num_logits]

    qsl_full = runner.input_buffers.query_start_loc
    sl_full = runner.input_buffers.seq_lens
    is_padding = runner.input_buffers.is_padding[:num_tokens_after_padding]
    last_sampled = runner.req_states.last_sampled_tokens
    draft_tokens_buf = runner.req_states.draft_tokens
    if (
        not isinstance(draft_tokens_buf, torch.Tensor)
        or draft_tokens_buf.dim() != 2
        or draft_tokens_buf.stride(1) != 1
    ):
        return None

    kernel[(1,)](
        is_padding,
        idx_mapping,
        cu_num_logits,
        qsl_full,
        expanded_idx_mapping,
        expanded_local_pos,
        runner.input_buffers.positions,
        sl_full,
        runner.input_buffers.input_ids,
        logits_indices,
        runner.req_states.num_computed_tokens.gpu,
        last_sampled,
        draft_tokens_buf,
        slot0,
        num_tokens,
        num_tokens_after_padding,
        int(qsl_full.shape[0]),
        int(sl_full.shape[0]),
        total_num_logits,
        int(draft_tokens_buf.stride(0)),
        int(last_sampled.stride(0)),
        WRITE_IS_PADDING=write_pad,
        BLOCK_R=block_r,
        BLOCK_T=block_t,
        num_warps=4,
    )
    st.b1_compiled.add(ckey)

    if DTGRAPH_FASTPATH:
        cu_num_logits_np, query_start_loc_np = _get_cached_qsl_np(
            st, 1, total_num_logits, num_reqs_padded
        )
    else:
        cu_num_logits_np = np.array([0, total_num_logits], dtype=np.int32)
        query_start_loc_np = np.full(num_reqs_padded + 1, num_tokens, dtype=np.int32)
        query_start_loc_np[0] = 0
    query_start_loc = qsl_full[: num_reqs_padded + 1]
    seq_lens = sl_full[:num_reqs_padded]

    num_computed_tokens_np = runner.req_states.num_computed_tokens_np[idx_mapping_np]
    seq_lens_cpu_upper_bound_np = np.zeros(num_reqs_padded, dtype=np.int32)
    seq_lens_cpu_upper_bound_np[0] = int(num_computed_tokens_np[0]) + num_tokens
    seq_lens_cpu_upper_bound = torch.from_numpy(seq_lens_cpu_upper_bound_np)

    all_kwargs = {
        "req_ids": req_ids,
        "num_reqs": 1,
        "num_reqs_after_padding": num_reqs_padded,
        "idx_mapping": idx_mapping,
        "idx_mapping_np": idx_mapping_np,
        "expanded_idx_mapping": expanded_idx_mapping,
        "expanded_local_pos": expanded_local_pos,
        "num_scheduled_tokens": num_scheduled_tokens_np,
        "num_tokens": num_tokens,
        "num_tokens_after_padding": num_tokens_after_padding,
        "num_draft_tokens": total_num_draft_tokens,
        "num_draft_tokens_per_req": num_draft_tokens_per_req,
        "num_admitted_draft_tokens": None,
        "query_start_loc": query_start_loc,
        "query_start_loc_np": query_start_loc_np,
        "seq_lens": seq_lens,
        "seq_lens_cpu_upper_bound": seq_lens_cpu_upper_bound,
        "dcp_local_seq_lens": None,
        "num_computed_tokens_np": num_computed_tokens_np,
        "prefill_len_np": batch_req_state.prefill_len_np,
        "num_computed_prefill_tokens_np": batch_req_state.num_computed_prefill_tokens_np,
        "is_prefilling_np": batch_req_state.is_prefilling_np,
        "has_prefill": False,
        "prefill_runs_as_decode_np": batch_req_state.prefill_runs_as_decode_np,
        "decode_graph_eligible": batch_req_state.decode_graph_eligible,
        "max_seq_len_np": batch_req_state.max_seq_len_np,
        "input_ids": runner.input_buffers.input_ids[:num_tokens_after_padding],
        "positions": runner.input_buffers.positions[:num_tokens_after_padding],
        "is_padding": is_padding,
        "logits_indices": logits_indices,
        "cu_num_logits": cu_num_logits,
        "cu_num_logits_np": cu_num_logits_np,
        "has_structured_output_reqs": scheduler_output.has_structured_output_requests,
        "prompt_lens": None,
        "fast_prefill": None,
        "max_query_len": None,
    }
    if DTGRAPH_FASTPATH:
        if st.b1_ib_drop:
            for k in st.b1_ib_drop:
                all_kwargs.pop(k, None)
        filtered = all_kwargs
    else:
        filtered = {k: v for k, v in all_kwargs.items() if k in st.b1_ib_fields}
    if st.region is not None:
        st.region.stats["b1_fused_prep"] += 1
    return st.b1_ib_cls(**filtered)


# ----------------------------------------------------------------------------------------------
# Lever A2 (opt-in C=32): Fused B=2..32 uniform pure-decode prepare_inputs
# (1 tiny B*8 B H2D of idx_mapping + 1 (B+1)-CTA Triton kernel replacing 2 H2Ds + 6 kernels!)
# ----------------------------------------------------------------------------------------------
_FUSED_DT_PREP_BN_KERNEL = None


def _get_fused_dt_prep_bn_kernel():
    global _FUSED_DT_PREP_BN_KERNEL
    if _FUSED_DT_PREP_BN_KERNEL is not None:
        return _FUSED_DT_PREP_BN_KERNEL
    from vllm.triton_utils import tl, triton  # noqa: PLC0415

    @triton.jit(
        do_not_specialize=[
            "num_reqs",
            "q_len",
            "num_tokens",
            "num_tokens_padded",
            "qsl_len",
            "sl_len",
            "draft_stride",
            "last_sampled_stride",
        ]
    )
    def _fused_dt_prep_bn_kernel(
        is_padding_ptr,
        idx_mapping_ptr,
        cu_num_logits_ptr,
        query_start_loc_ptr,
        expanded_idx_ptr,
        expanded_local_pos_ptr,
        positions_ptr,
        seq_lens_ptr,
        input_ids_ptr,
        logits_indices_ptr,
        num_computed_ptr,
        last_sampled_ptr,
        draft_tokens_ptr,
        num_reqs,
        q_len,
        num_tokens,
        num_tokens_padded,
        qsl_len,
        sl_len,
        draft_stride,
        last_sampled_stride,
        WRITE_IS_PADDING: tl.constexpr,
        HAS_DRAFTS: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_Q: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        """(B+1)-CTA fused replacement for 2x is_padding.fill_ + 2x H2D + expand_idx_mapping +
        prepare_pos_seq_lens + combine_sampled_and_draft_tokens on uniform pure-decode B=2..32."""
        pid = tl.program_id(0)
        if pid < num_reqs:
            r = pid.to(tl.int64)
            slot = tl.load(idx_mapping_ptr + r).to(tl.int64)
            num_computed = tl.load(num_computed_ptr + slot).to(tl.int64)
            last_tok = tl.load(
                last_sampled_ptr + slot * last_sampled_stride.to(tl.int64)
            ).to(tl.int64)
            ql = q_len.to(tl.int32)
            base = (pid * ql).to(tl.int32)

            tl.store(
                query_start_loc_ptr + r,
                base.to(query_start_loc_ptr.dtype.element_ty),
            )
            tl.store(
                cu_num_logits_ptr + r,
                base.to(cu_num_logits_ptr.dtype.element_ty),
            )
            tl.store(
                seq_lens_ptr + r,
                (num_computed + ql).to(seq_lens_ptr.dtype.element_ty),
            )

            q = tl.arange(0, BLOCK_Q)
            mq = q < ql
            pos_out = base + q
            tl.store(
                positions_ptr + pos_out,
                (num_computed + q).to(positions_ptr.dtype.element_ty),
                mask=mq,
            )
            tl.store(
                expanded_idx_ptr + pos_out,
                (tl.zeros((BLOCK_Q,), dtype=tl.int64) + slot).to(
                    expanded_idx_ptr.dtype.element_ty
                ),
                mask=mq,
            )
            tl.store(
                expanded_local_pos_ptr + pos_out,
                q.to(expanded_local_pos_ptr.dtype.element_ty),
                mask=mq,
            )
            tl.store(
                logits_indices_ptr + pos_out,
                pos_out.to(logits_indices_ptr.dtype.element_ty),
                mask=mq,
            )
            if HAS_DRAFTS:
                d_idx = tl.maximum(q - 1, 0).to(tl.int64)
                draft_tok = tl.load(
                    draft_tokens_ptr + slot * draft_stride.to(tl.int64) + d_idx,
                    mask=(q >= 1) & mq,
                    other=0,
                ).to(tl.int64)
                inp_tok = tl.where(q == 0, last_tok, draft_tok)
            else:
                inp_tok = tl.zeros((BLOCK_Q,), dtype=tl.int64) + last_tok
            tl.store(
                input_ids_ptr + pos_out,
                inp_tok.to(input_ids_ptr.dtype.element_ty),
                mask=mq,
            )
        else:
            nt = num_tokens.to(tl.int32)
            tl.store(
                cu_num_logits_ptr + num_reqs,
                nt.to(cu_num_logits_ptr.dtype.element_ty),
            )
            for off in range(0, qsl_len, BLOCK_R):
                rr = off + tl.arange(0, BLOCK_R)
                tl.store(
                    query_start_loc_ptr + rr,
                    nt.to(query_start_loc_ptr.dtype.element_ty),
                    mask=(rr >= num_reqs) & (rr < qsl_len),
                )
            for off in range(0, sl_len, BLOCK_R):
                rr = off + tl.arange(0, BLOCK_R)
                tl.store(
                    seq_lens_ptr + rr,
                    tl.zeros((BLOCK_R,), dtype=seq_lens_ptr.dtype.element_ty),
                    mask=(rr >= num_reqs) & (rr < sl_len),
                )
            if WRITE_IS_PADDING:
                for off in range(0, num_tokens_padded, BLOCK_T):
                    t = off + tl.arange(0, BLOCK_T)
                    tl.store(
                        is_padding_ptr + t,
                        t >= num_tokens,
                        mask=t < num_tokens_padded,
                    )

    _FUSED_DT_PREP_BN_KERNEL = (triton, _fused_dt_prep_bn_kernel)
    return _FUSED_DT_PREP_BN_KERNEL


_FUSED_DT_PREP_BN_INLINE_KERNEL = None


def _get_fused_dt_prep_bn_inline_kernel():
    global _FUSED_DT_PREP_BN_INLINE_KERNEL
    if _FUSED_DT_PREP_BN_INLINE_KERNEL is not None:
        return _FUSED_DT_PREP_BN_INLINE_KERNEL
    from vllm.triton_utils import tl, triton  # noqa: PLC0415

    @triton.jit(
        do_not_specialize=[
            "s0",
            "s1",
            "s2",
            "s3",
            "s4",
            "s5",
            "s6",
            "s7",
            "num_reqs",
            "q_len",
            "num_tokens",
            "num_tokens_padded",
            "qsl_len",
            "sl_len",
            "draft_stride",
            "last_sampled_stride",
        ]
    )
    def _fused_dt_prep_bn_inline_kernel(
        is_padding_ptr,
        idx_mapping_ptr,
        cu_num_logits_ptr,
        query_start_loc_ptr,
        expanded_idx_ptr,
        expanded_local_pos_ptr,
        positions_ptr,
        seq_lens_ptr,
        input_ids_ptr,
        logits_indices_ptr,
        num_computed_ptr,
        last_sampled_ptr,
        draft_tokens_ptr,
        s0,
        s1,
        s2,
        s3,
        s4,
        s5,
        s6,
        s7,
        num_reqs,
        q_len,
        num_tokens,
        num_tokens_padded,
        qsl_len,
        sl_len,
        draft_stride,
        last_sampled_stride,
        WRITE_IS_PADDING: tl.constexpr,
        HAS_DRAFTS: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_Q: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        """(B+1)-CTA zero-H2D-DMA inline-slot fused prepare_inputs for B=2..8 pure-decode.
        Passes slots s0..s7 in the kernel parameter buffer (eliminating Arena copy + cudaMemcpyAsync
        H2D + DMA-to-CE barrier), writes idx_mapping[0:B] identically, 100% 0-ULP bit-exact."""
        pid = tl.program_id(0)
        if pid < num_reqs:
            r = pid.to(tl.int64)
            slot = s7.to(tl.int64)
            slot = tl.where(pid == 6, s6.to(tl.int64), slot)
            slot = tl.where(pid == 5, s5.to(tl.int64), slot)
            slot = tl.where(pid == 4, s4.to(tl.int64), slot)
            slot = tl.where(pid == 3, s3.to(tl.int64), slot)
            slot = tl.where(pid == 2, s2.to(tl.int64), slot)
            slot = tl.where(pid == 1, s1.to(tl.int64), slot)
            slot = tl.where(pid == 0, s0.to(tl.int64), slot)
            tl.store(idx_mapping_ptr + r, slot.to(idx_mapping_ptr.dtype.element_ty))

            num_computed = tl.load(num_computed_ptr + slot).to(tl.int64)
            last_tok = tl.load(
                last_sampled_ptr + slot * last_sampled_stride.to(tl.int64)
            ).to(tl.int64)
            ql = q_len.to(tl.int32)
            base = (pid * ql).to(tl.int32)

            tl.store(
                query_start_loc_ptr + r,
                base.to(query_start_loc_ptr.dtype.element_ty),
            )
            tl.store(
                cu_num_logits_ptr + r,
                base.to(cu_num_logits_ptr.dtype.element_ty),
            )
            tl.store(
                seq_lens_ptr + r,
                (num_computed + ql).to(seq_lens_ptr.dtype.element_ty),
            )

            q = tl.arange(0, BLOCK_Q)
            mq = q < ql
            pos_out = base + q
            tl.store(
                positions_ptr + pos_out,
                (num_computed + q).to(positions_ptr.dtype.element_ty),
                mask=mq,
            )
            tl.store(
                expanded_idx_ptr + pos_out,
                (tl.zeros((BLOCK_Q,), dtype=tl.int64) + slot).to(
                    expanded_idx_ptr.dtype.element_ty
                ),
                mask=mq,
            )
            tl.store(
                expanded_local_pos_ptr + pos_out,
                q.to(expanded_local_pos_ptr.dtype.element_ty),
                mask=mq,
            )
            tl.store(
                logits_indices_ptr + pos_out,
                pos_out.to(logits_indices_ptr.dtype.element_ty),
                mask=mq,
            )
            if HAS_DRAFTS:
                d_idx = tl.maximum(q - 1, 0).to(tl.int64)
                draft_tok = tl.load(
                    draft_tokens_ptr + slot * draft_stride.to(tl.int64) + d_idx,
                    mask=(q >= 1) & mq,
                    other=0,
                ).to(tl.int64)
                inp_tok = tl.where(q == 0, last_tok, draft_tok)
            else:
                inp_tok = tl.zeros((BLOCK_Q,), dtype=tl.int64) + last_tok
            tl.store(
                input_ids_ptr + pos_out,
                inp_tok.to(input_ids_ptr.dtype.element_ty),
                mask=mq,
            )
        else:
            nt = num_tokens.to(tl.int32)
            tl.store(
                cu_num_logits_ptr + num_reqs,
                nt.to(cu_num_logits_ptr.dtype.element_ty),
            )
            for off in range(0, qsl_len, BLOCK_R):
                rr = off + tl.arange(0, BLOCK_R)
                tl.store(
                    query_start_loc_ptr + rr,
                    nt.to(query_start_loc_ptr.dtype.element_ty),
                    mask=(rr >= num_reqs) & (rr < qsl_len),
                )
            for off in range(0, sl_len, BLOCK_R):
                rr = off + tl.arange(0, BLOCK_R)
                tl.store(
                    seq_lens_ptr + rr,
                    tl.zeros((BLOCK_R,), dtype=seq_lens_ptr.dtype.element_ty),
                    mask=(rr >= num_reqs) & (rr < sl_len),
                )
            if WRITE_IS_PADDING:
                for off in range(0, num_tokens_padded, BLOCK_T):
                    t = off + tl.arange(0, BLOCK_T)
                    tl.store(
                        is_padding_ptr + t,
                        t >= num_tokens,
                        mask=t < num_tokens_padded,
                    )

    _FUSED_DT_PREP_BN_INLINE_KERNEL = (triton, _fused_dt_prep_bn_inline_kernel)
    return _FUSED_DT_PREP_BN_INLINE_KERNEL


def _bn_prep_eligible(runner, scheduler_output, batch_req_state, batch_desc) -> bool:
    if not FUSED_DT_PREP or FUSED_DT_PREP_MAX_B < 2:
        return False
    num_reqs = len(batch_req_state.req_ids)
    if num_reqs < 2 or num_reqs > FUSED_DT_PREP_MAX_B or bool(batch_req_state.has_prefill):
        return False
    if getattr(runner.model_state, "num_new_sampled_tokens_per_step", 0) != 1:
        return False
    if getattr(runner, "adaptive_verification", None) is not None:
        return False
    if getattr(runner, "fast_prefill", None) is not None:
        return False
    if getattr(runner.model_config, "rswa_window", None) is not None:
        return False
    if getattr(runner, "pcp_manager", None) is not None:
        return False
    num_tokens = int(batch_req_state.num_tokens)
    if num_tokens <= 0 or (num_tokens % num_reqs) != 0:
        return False
    q_len = num_tokens // num_reqs
    if q_len < 1 or q_len > 16:
        return False
    num_tokens_after_padding = max(num_tokens, int(batch_desc.num_tokens))
    if num_tokens_after_padding > 512:
        return False
    nst_np = batch_req_state.num_scheduled_tokens
    has_drafts = bool(getattr(scheduler_output, "scheduled_spec_decode_tokens", None))
    if FAST_REPLAY_V14 and num_reqs <= 8:
        ncpf = batch_req_state.num_computed_prefill_tokens_np
        pfl = batch_req_state.prefill_len_np
        if has_drafts:
            nd_np = batch_req_state.num_draft_tokens_np
            if nd_np is None:
                return False
            ql_m1 = q_len - 1
            for _i in range(num_reqs):
                if int(nst_np[_i]) != q_len or int(nd_np[_i]) != ql_m1 or int(ncpf[_i]) < int(pfl[_i]):
                    return False
        else:
            if q_len != 1:
                return False
            for _i in range(num_reqs):
                if int(nst_np[_i]) != 1 or int(ncpf[_i]) < int(pfl[_i]):
                    return False
        return getattr(runner, "device", None) is not None and runner.device.type == "cuda"
    if not bool(np.all(nst_np == q_len)):
        return False
    if has_drafts:
        nd_np = batch_req_state.num_draft_tokens_np
        if nd_np is None or not bool(np.all(nd_np == (q_len - 1))):
            return False
    elif q_len != 1:
        return False
    if not bool(
        np.all(
            batch_req_state.num_computed_prefill_tokens_np
            >= batch_req_state.prefill_len_np
        )
    ):
        return False
    return getattr(runner, "device", None) is not None and runner.device.type == "cuda"


def _ensure_bn_bufs(st: _RunnerState, device: torch.device, min_reqs: int, min_tok: int) -> bool:
    if st.b1_dtypes is None:
        return False
    need_r = max(64, int(min_reqs))
    need_t = max(256, int(min_tok))
    if (
        st.bn_bufs is not None
        and st.bn_bufs["idx"].shape[0] >= need_r
        and st.bn_bufs["expanded_idx"].shape[0] >= need_t
    ):
        return True
    if torch.cuda.is_current_stream_capturing():
        return False
    dt_idx, dt_cu, dt_eidx, dt_epos, dt_li = st.b1_dtypes
    use_pin = bool(PIN_ARENA and torch.cuda.is_available())
    try:
        idx_cpu_buf = torch.empty((need_r,), dtype=dt_idx, pin_memory=use_pin)
    except Exception:  # noqa: BLE001
        idx_cpu_buf = torch.empty((need_r,), dtype=dt_idx)
    st.bn_bufs = {
        "idx": torch.empty((need_r,), dtype=dt_idx, device=device),
        "idx_cpu": idx_cpu_buf,
        "cu": torch.empty((need_r + 1,), dtype=dt_cu, device=device),
        "expanded_idx": torch.empty((need_t,), dtype=dt_eidx, device=device),
        "expanded_pos": torch.empty((need_t,), dtype=dt_epos, device=device),
        "logits_indices": torch.empty((need_t,), dtype=dt_li, device=device),
    }
    return True


def _swap_np_tail(arr, i: int, tail: int):
    if arr is None or not isinstance(arr, np.ndarray) or arr.ndim < 1 or arr.shape[0] <= tail:
        return arr
    out = arr.copy()
    tmp = out[i].copy() if out.ndim > 1 else out[i]
    out[i] = out[tail]
    out[tail] = tmp
    return out


def _compute_rowar_swap(
    runner,
    st: _RunnerState,
    scheduler_output,
    batch_req_state,
    q_len: int,
    has_drafts: bool,
):
    """Apply deterministic Wave-V8 ROWAR tail-swap once per eligible B=2..8 step."""
    req_ids = batch_req_state.req_ids
    num_reqs = len(req_ids)
    idx_mapping_np = batch_req_state.idx_mapping_np
    prefill_len_np = batch_req_state.prefill_len_np
    num_computed_prefill_tokens_np = batch_req_state.num_computed_prefill_tokens_np
    is_prefilling_np = batch_req_state.is_prefilling_np
    prefill_runs_as_decode_np = batch_req_state.prefill_runs_as_decode_np
    max_seq_len_np = batch_req_state.max_seq_len_np
    if (
        ROWAR_DESC_W_ORDER
        and 2 <= num_reqs <= 8
        and has_drafts
        and not getattr(scheduler_output, "has_structured_output_requests", False)
    ):
        if st.rowar_order_safe is None:
            st.rowar_order_safe = _probe_rowar_order_safe(runner)
        if st.rowar_order_safe:
            if st.rowar_tracker is None:
                st.rowar_tracker = _RowarAccTracker()
            max_delta = max(q_len, int(getattr(runner, "num_speculative_steps", 7) or 7) + 1)
            st.rowar_tracker.update(
                idx_mapping_np, runner.req_states.num_computed_tokens_np, max_delta
            )
            margin_q8 = max(0, int(round(ROWAR_DESC_W_MARGIN * 256.0)))
            swap_i = _rowar_select_tail_swap(idx_mapping_np, st.rowar_tracker, margin_q8)
            if swap_i >= 0:
                tail = num_reqs - 1
                req_ids = list(req_ids)
                req_ids[swap_i], req_ids[tail] = req_ids[tail], req_ids[swap_i]
                idx_mapping_np = _swap_np_tail(idx_mapping_np, swap_i, tail)
                prefill_len_np = _swap_np_tail(prefill_len_np, swap_i, tail)
                num_computed_prefill_tokens_np = _swap_np_tail(
                    num_computed_prefill_tokens_np, swap_i, tail
                )
                is_prefilling_np = _swap_np_tail(is_prefilling_np, swap_i, tail)
                prefill_runs_as_decode_np = _swap_np_tail(
                    prefill_runs_as_decode_np, swap_i, tail
                )
                max_seq_len_np = _swap_np_tail(max_seq_len_np, swap_i, tail)
                if st.region is not None:
                    st.region.stats["rowar_tail_swap"] += 1
    return (
        req_ids,
        idx_mapping_np,
        prefill_len_np,
        num_computed_prefill_tokens_np,
        is_prefilling_np,
        prefill_runs_as_decode_np,
        max_seq_len_np,
    )


def _get_dt_immutable_idx_cpu(
    st: _RunnerState, dtype: torch.dtype, idx_mapping_np: np.ndarray
) -> torch.Tensor:
    """Return a strictly immutable pinned CPU tensor holding `idx_mapping_np` for B=9..32
    static-dt graph capture so H2D reads never alias the shared Region Arena."""
    key = (dtype, tuple(int(x) for x in idx_mapping_np))
    cached = st.dt_immutable_idx_bufs.get(key)
    if cached is not None:
        st.dt_immutable_idx_bufs.move_to_end(key)
        if st.region is not None:
            st.region.immutable_ptrs.add(int(cached.data_ptr()))
        return cached
    use_pin = bool(PIN_ARENA and torch.cuda.is_available())
    try:
        buf = torch.empty((len(idx_mapping_np),), dtype=dtype, pin_memory=use_pin)
    except Exception:  # noqa: BLE001
        buf = torch.empty((len(idx_mapping_np),), dtype=dtype)
    np.copyto(buf.numpy(), idx_mapping_np, casting="unsafe")
    st.dt_immutable_idx_bufs[key] = buf
    if FAST_REPLAY_V16:
        st._v16_imm_idx_by_ptr[int(buf.data_ptr())] = buf
    if st.region is not None:
        st.region.immutable_ptrs.add(int(buf.data_ptr()))
    max_imm = max(64, STATIC_DT_MAX_CERTS * 2)
    while len(st.dt_immutable_idx_bufs) > max_imm:
        _, old_buf = st.dt_immutable_idx_bufs.popitem(last=False)
        old_ptr = int(old_buf.data_ptr())
        if FAST_REPLAY_V16:
            st._v16_imm_idx_by_ptr.pop(old_ptr, None)
        if st.region is not None:
            st.region.immutable_ptrs.discard(old_ptr)
        stale_keys = [
            ck for ck, c in st.dt_static_cert.items() if c.get("imm_idx_ptr") == old_ptr
        ]
        for ck in stale_keys:
            _ev_c = st.dt_static_cert.pop(ck, None)
            if _ev_c is not None:
                if st.dt_last_cert is _ev_c:
                    st.dt_last_cert = None
                if FAST_REPLAY_V16:
                    _evict_dt_v16_lut(st, _ev_c)
    return buf


def _try_fused_prepare_inputs_bn(
    runner,
    st: _RunnerState,
    scheduler_output,
    batch_req_state,
    batch_desc,
    *,
    skip_elig_check: bool = False,
    precomputed_rowar: tuple | None = None,
    imm_idx_cpu: torch.Tensor | None = None,
):
    if st.b1_failed or st.b1_dtypes is None or st.b1_ib_cls is None or st.b1_ib_fields is None:
        return None
    if not skip_elig_check and not _bn_prep_eligible(
        runner, scheduler_output, batch_req_state, batch_desc
    ):
        return None
    req_ids = batch_req_state.req_ids
    num_reqs = len(req_ids)
    num_tokens = int(batch_req_state.num_tokens)
    q_len = num_tokens // num_reqs
    num_tokens_after_padding = max(num_tokens, int(batch_desc.num_tokens))
    if not _ensure_bn_bufs(st, runner.device, num_reqs, num_tokens_after_padding):
        return None
    write_pad = _get_write_pad(st)
    has_drafts = bool(getattr(scheduler_output, "scheduled_spec_decode_tokens", None))
    use_inline_b8 = FUSED_DT_PREP_INLINE_B8 and num_reqs <= 8
    if use_inline_b8:
        triton, kernel = _get_fused_dt_prep_bn_inline_kernel()
    else:
        triton, kernel = _get_fused_dt_prep_bn_kernel()
    block_r = 256
    block_q = max(16, triton.next_power_of_2(q_len))
    block_t = 256
    ib_buf = runner.input_buffers
    rs_buf = runner.req_states
    dr_t = getattr(rs_buf, "draft_tokens", None) if has_drafts else rs_buf.last_sampled_tokens
    ptr_sig = _dt_prep_ptr_sig(
        ib_buf.is_padding, st.bn_bufs["idx"], st.bn_bufs["cu"], ib_buf.query_start_loc,
        st.bn_bufs["expanded_idx"], st.bn_bufs["expanded_pos"], ib_buf.positions,
        ib_buf.seq_lens, ib_buf.input_ids, st.bn_bufs["logits_indices"],
        rs_buf.num_computed_tokens.gpu, rs_buf.last_sampled_tokens,
        dr_t if isinstance(dr_t, torch.Tensor) else rs_buf.last_sampled_tokens,
    )
    ckey = (write_pad, has_drafts, block_r, block_q, block_t, ptr_sig)
    compiled_set = st.bn_inline_compiled if use_inline_b8 else st.bn_compiled
    if ckey not in compiled_set and torch.cuda.is_current_stream_capturing():
        return None

    bufs = st.bn_bufs
    assert bufs is not None
    if precomputed_rowar is not None:
        (
            req_ids,
            idx_mapping_np,
            prefill_len_np,
            num_computed_prefill_tokens_np,
            is_prefilling_np,
            prefill_runs_as_decode_np,
            max_seq_len_np,
        ) = precomputed_rowar
    else:
        (
            req_ids,
            idx_mapping_np,
            prefill_len_np,
            num_computed_prefill_tokens_np,
            is_prefilling_np,
            prefill_runs_as_decode_np,
            max_seq_len_np,
        ) = _compute_rowar_swap(runner, st, scheduler_output, batch_req_state, q_len, has_drafts)

    idx_mapping = bufs["idx"][:num_reqs]
    cu_num_logits = bufs["cu"][: num_reqs + 1]
    expanded_idx_mapping = bufs["expanded_idx"][:num_tokens]
    expanded_local_pos = bufs["expanded_pos"][:num_tokens]
    logits_indices = bufs["logits_indices"][:num_tokens]

    qsl_full = runner.input_buffers.query_start_loc
    sl_full = runner.input_buffers.seq_lens
    is_padding = runner.input_buffers.is_padding[:num_tokens_after_padding]
    last_sampled = runner.req_states.last_sampled_tokens
    draft_tokens_buf = getattr(runner.req_states, "draft_tokens", None)
    if has_drafts:
        if (
            not isinstance(draft_tokens_buf, torch.Tensor)
            or draft_tokens_buf.dim() != 2
            or draft_tokens_buf.stride(1) != 1
        ):
            return None
        draft_ptr = draft_tokens_buf
        draft_stride = int(draft_tokens_buf.stride(0))
    else:
        draft_ptr = last_sampled
        draft_stride = int(last_sampled.stride(0))

    if use_inline_b8:
        s0 = int(idx_mapping_np[0])
        s1 = int(idx_mapping_np[1])
        s2 = int(idx_mapping_np[2]) if num_reqs > 2 else 0
        s3 = int(idx_mapping_np[3]) if num_reqs > 3 else 0
        s4 = int(idx_mapping_np[4]) if num_reqs > 4 else 0
        s5 = int(idx_mapping_np[5]) if num_reqs > 5 else 0
        s6 = int(idx_mapping_np[6]) if num_reqs > 6 else 0
        s7 = int(idx_mapping_np[7]) if num_reqs > 7 else 0
        kernel[(num_reqs + 1,)](
            is_padding,
            idx_mapping,
            cu_num_logits,
            qsl_full,
            expanded_idx_mapping,
            expanded_local_pos,
            runner.input_buffers.positions,
            sl_full,
            runner.input_buffers.input_ids,
            logits_indices,
            runner.req_states.num_computed_tokens.gpu,
            last_sampled,
            draft_ptr,
            s0,
            s1,
            s2,
            s3,
            s4,
            s5,
            s6,
            s7,
            num_reqs,
            q_len,
            num_tokens,
            num_tokens_after_padding,
            int(qsl_full.shape[0]),
            int(sl_full.shape[0]),
            draft_stride,
            int(last_sampled.stride(0)),
            WRITE_IS_PADDING=write_pad,
            HAS_DRAFTS=has_drafts,
            BLOCK_R=block_r,
            BLOCK_Q=block_q,
            BLOCK_T=block_t,
            num_warps=4,
        )
        st.bn_inline_compiled.add(ckey)
        if st.region is not None:
            st.region.stats["bn_inline_fused_prep"] += 1
    else:
        if imm_idx_cpu is not None:
            idx_mapping.copy_(imm_idx_cpu, non_blocking=True)
        else:
            staged = None
            if (
                DTGRAPH_FASTPATH
                and st.region is not None
                and st.region.active
                and st.region.capturing
            ):
                staged = st.region.arena.alloc((num_reqs,), idx_mapping.dtype)
            if staged is not None:
                np.copyto(staged.numpy(), idx_mapping_np, casting="unsafe")
                idx_mapping.copy_(staged, non_blocking=True)
            else:
                idx_cpu = bufs["idx_cpu"][:num_reqs]
                np.copyto(idx_cpu.numpy(), idx_mapping_np, casting="unsafe")
                idx_mapping.copy_(idx_cpu, non_blocking=True)

        kernel[(num_reqs + 1,)](
            is_padding,
            idx_mapping,
            cu_num_logits,
            qsl_full,
            expanded_idx_mapping,
            expanded_local_pos,
            runner.input_buffers.positions,
            sl_full,
            runner.input_buffers.input_ids,
            logits_indices,
            runner.req_states.num_computed_tokens.gpu,
            last_sampled,
            draft_ptr,
            num_reqs,
            q_len,
            num_tokens,
            num_tokens_after_padding,
            int(qsl_full.shape[0]),
            int(sl_full.shape[0]),
            draft_stride,
            int(last_sampled.stride(0)),
            WRITE_IS_PADDING=write_pad,
            HAS_DRAFTS=has_drafts,
            BLOCK_R=block_r,
            BLOCK_Q=block_q,
            BLOCK_T=block_t,
            num_warps=4,
        )
        st.bn_compiled.add(ckey)

    num_scheduled_tokens_np = batch_req_state.num_scheduled_tokens
    num_reqs_padded = int(batch_desc.num_reqs or num_reqs)
    if DTGRAPH_FASTPATH:
        cu_num_logits_np, query_start_loc_np = _get_cached_qsl_np(
            st, num_reqs, q_len, num_reqs_padded
        )
    else:
        cu_num_logits_np = np.arange(num_reqs + 1, dtype=np.int32) * q_len
        query_start_loc_np = np.full(num_reqs_padded + 1, num_tokens, dtype=np.int32)
        query_start_loc_np[: num_reqs + 1] = cu_num_logits_np
    query_start_loc = qsl_full[: num_reqs_padded + 1]
    seq_lens = sl_full[:num_reqs_padded]

    num_computed_tokens_np = runner.req_states.num_computed_tokens_np[idx_mapping_np]
    seq_lens_cpu_upper_bound_np = np.zeros(num_reqs_padded, dtype=np.int32)
    np.add(
        num_computed_tokens_np,
        num_scheduled_tokens_np,
        out=seq_lens_cpu_upper_bound_np[:num_reqs],
    )
    seq_lens_cpu_upper_bound = torch.from_numpy(seq_lens_cpu_upper_bound_np)

    num_draft_tokens_per_req = batch_req_state.num_draft_tokens_np if has_drafts else None
    total_num_draft_tokens = num_reqs * (q_len - 1) if has_drafts else 0

    all_kwargs = {
        "req_ids": req_ids,
        "num_reqs": num_reqs,
        "num_reqs_after_padding": num_reqs_padded,
        "idx_mapping": idx_mapping,
        "idx_mapping_np": idx_mapping_np,
        "expanded_idx_mapping": expanded_idx_mapping,
        "expanded_local_pos": expanded_local_pos,
        "num_scheduled_tokens": num_scheduled_tokens_np,
        "num_tokens": num_tokens,
        "num_tokens_after_padding": num_tokens_after_padding,
        "num_draft_tokens": total_num_draft_tokens,
        "num_draft_tokens_per_req": num_draft_tokens_per_req,
        "num_admitted_draft_tokens": None,
        "query_start_loc": query_start_loc,
        "query_start_loc_np": query_start_loc_np,
        "seq_lens": seq_lens,
        "seq_lens_cpu_upper_bound": seq_lens_cpu_upper_bound,
        "dcp_local_seq_lens": None,
        "num_computed_tokens_np": num_computed_tokens_np,
        "prefill_len_np": prefill_len_np,
        "num_computed_prefill_tokens_np": num_computed_prefill_tokens_np,
        "is_prefilling_np": is_prefilling_np,
        "has_prefill": False,
        "prefill_runs_as_decode_np": prefill_runs_as_decode_np,
        "decode_graph_eligible": batch_req_state.decode_graph_eligible,
        "max_seq_len_np": max_seq_len_np,
        "input_ids": runner.input_buffers.input_ids[:num_tokens_after_padding],
        "positions": runner.input_buffers.positions[:num_tokens_after_padding],
        "is_padding": is_padding,
        "logits_indices": logits_indices,
        "cu_num_logits": cu_num_logits,
        "cu_num_logits_np": cu_num_logits_np,
        "has_structured_output_reqs": scheduler_output.has_structured_output_requests,
        "prompt_lens": None,
        "fast_prefill": None,
        "max_query_len": None,
    }
    if DTGRAPH_FASTPATH:
        if st.b1_ib_drop:
            for k in st.b1_ib_drop:
                all_kwargs.pop(k, None)
        filtered = all_kwargs
    else:
        filtered = {k: v for k, v in all_kwargs.items() if k in st.b1_ib_fields}
    if st.region is not None:
        st.region.stats["bn_fused_prep"] += 1
    return st.b1_ib_cls(**filtered)


def _dt_prep_ptr_sig(*tensors) -> tuple:
    return tuple(getattr(t, "dtype", None) for t in tensors)


def _prewarm_fused_dt_prep_kernels(runner) -> int:
    """Pre-JIT fused prepare_inputs Triton kernels (bn & bn_inline) matching REAL live
    runner.input_buffers & runner.req_states dtypes across vLLM V1 InputBatch index dtype
    conventions, and tag compiled cache keys with exact 13-pointer dtype signatures."""
    if not (FUSED_DT_PREP and FUSED_DT_PREP_MAX_B >= 2):
        return 0
    device = getattr(runner, "device", None)
    if device is None or getattr(device, "type", "") != "cuda":
        return 0
    n_jits = 0
    try:
        from vllm import envs  # noqa: PLC0415

        w_pad = bool(envs.VLLM_MOE_SKIP_PADDING)
        st_pw = _get_state(runner)
        ib = getattr(runner, "input_buffers", None)
        rs = getattr(runner, "req_states", None)
        dt_pad = getattr(getattr(ib, "is_padding", None), "dtype", torch.bool)
        dt_qsl = getattr(getattr(ib, "query_start_loc", None), "dtype", torch.int32)
        dt_pos = getattr(getattr(ib, "positions", None), "dtype", torch.int64)
        dt_sl = getattr(getattr(ib, "seq_lens", None), "dtype", torch.int32)
        dt_inp = getattr(getattr(ib, "input_ids", None), "dtype", torch.int32)
        ncomp_obj = getattr(rs, "num_computed_tokens", None)
        dt_ncomp = getattr(getattr(ncomp_obj, "gpu", ncomp_obj), "dtype", torch.int32)
        dt_ls = getattr(getattr(rs, "last_sampled_tokens", None), "dtype", torch.int32)
        dt_dr = getattr(getattr(rs, "draft_tokens", None), "dtype", dt_ls)

        if st_pw.b1_dtypes is not None:
            ib_combos = [st_pw.b1_dtypes]
        else:
            ib_combos = [
                (torch.int32, torch.int32, torch.int32, torch.int32, torch.int64),
                (torch.int32, torch.int32, torch.int32, torch.int32, torch.int32),
                (torch.int64, torch.int32, torch.int64, torch.int32, torch.int32),
            ]

        d_pad = torch.zeros(128, dtype=dt_pad, device=device)
        d_qsl = torch.zeros(128, dtype=dt_qsl, device=device)
        d_pos = torch.zeros(128, dtype=dt_pos, device=device)
        d_sl = torch.zeros(128, dtype=dt_sl, device=device)
        d_inp = torch.zeros(128, dtype=dt_inp, device=device)
        d_ncomp = torch.zeros(128, dtype=dt_ncomp, device=device)
        d_ls = torch.zeros(128, dtype=dt_ls, device=device)
        d_dr = torch.zeros((32, 8), dtype=dt_dr, device=device)

        _, bn_k = _get_fused_dt_prep_bn_kernel()
        bni_k = _get_fused_dt_prep_bn_inline_kernel()[1] if FUSED_DT_PREP_INLINE_B8 else None
        for dt_idx, dt_cu, dt_eidx, dt_epos, dt_li in ib_combos:
            d_idx = torch.zeros(32, dtype=dt_idx, device=device)
            d_cu = torch.zeros(128, dtype=dt_cu, device=device)
            d_eidx = torch.zeros(128, dtype=dt_eidx, device=device)
            d_epos = torch.zeros(128, dtype=dt_epos, device=device)
            d_li = torch.zeros(128, dtype=dt_li, device=device)
            for has_d in (True, False):
                dr_arg = d_dr if has_d else d_ls
                dr_stride = 8 if has_d else 1
                psig = _dt_prep_ptr_sig(
                    d_pad, d_idx, d_cu, d_qsl, d_eidx, d_epos, d_pos, d_sl,
                    d_inp, d_li, d_ncomp, d_ls, dr_arg,
                )
                ck = (w_pad, has_d, 256, 16, 256, psig)
                if ck not in st_pw.bn_compiled:
                    bn_k[(3,)](
                        d_pad, d_idx, d_cu, d_qsl, d_eidx, d_epos, d_pos, d_sl,
                        d_inp, d_li, d_ncomp, d_ls, dr_arg,
                        2, 4 if has_d else 1, 8 if has_d else 2, 8, 33, 32, dr_stride, 1,
                        WRITE_IS_PADDING=w_pad, HAS_DRAFTS=has_d,
                        BLOCK_R=256, BLOCK_Q=16, BLOCK_T=256, num_warps=4,
                    )
                    st_pw.bn_compiled.add(ck)
                    n_jits += 1
                if bni_k is not None and ck not in st_pw.bn_inline_compiled:
                    bni_k[(3,)](
                        d_pad, d_idx, d_cu, d_qsl, d_eidx, d_epos, d_pos, d_sl,
                        d_inp, d_li, d_ncomp, d_ls, dr_arg,
                        0, 1, 0, 0, 0, 0, 0, 0,
                        2, 4 if has_d else 1, 8 if has_d else 2, 8, 33, 32, dr_stride, 1,
                        WRITE_IS_PADDING=w_pad, HAS_DRAFTS=has_d,
                        BLOCK_R=256, BLOCK_Q=16, BLOCK_T=256, num_warps=4,
                    )
                    st_pw.bn_inline_compiled.add(ck)
                    n_jits += 1
    except Exception:  # noqa: BLE001
        pass
    return n_jits


def prewarm_prefill_c32(runner) -> dict[str, int | float]:
    """Opt-in (`INKLING_PREFILL_PREWARM=1`): eagerly load lazy prefill .so libs, warm FP8 dense
    weight cache, and pre-JIT multi-req Triton glue kernels right after capture_model() so the
    first C=32 ramp-up prefill pays 0 ms of cold-load / JIT stall. 100% 0-ULP bit-exact."""
    t0 = time.perf_counter()
    summary: dict[str, int | float] = {"libs": 0, "fp8_weights": 0, "jits": 0, "ms": 0.0}
    device = getattr(runner, "device", None)
    if device is None or device.type != "cuda" or torch.cuda.is_current_stream_capturing():
        return summary

    # 1. Eagerly load lazy prefill CUDA .so libraries
    for mod_path, fn_name in (
        ("vllm.models.inkling2.sconv", "_get_fused_prefill_lib"),
        ("vllm.models.inkling2.ops.sm12_attn", "get_local_otf_rel_lib"),
        ("vllm.models.inkling2.ops.sm12_attn", "get_global_tail_otf_rel_lib"),
        ("vllm.models.inkling2.ops.pf_moe_native", "get_lib"),
        ("vllm.models.inkling2.ops.prefill_qf", "get_lib"),
    ):
        try:
            mod = __import__(mod_path, fromlist=[fn_name])
            fn = getattr(mod, fn_name, None)
            if fn is not None and fn() is not None:
                summary["libs"] += 1
        except Exception:  # noqa: BLE001
            pass

    # 2. Pre-warm FP8 dense & router weight caches (_FP8_W_CACHE) for all 48 target layers
    try:
        from vllm.models.inkling2.ops import linear as _lin  # noqa: PLC0415

        want_dense = bool(getattr(_lin, "_PF_DENSE_FP8", False))
        want_router = bool(getattr(_lin, "_PF_ROUTER_FP8", False))
        if want_dense or want_router:
            mode, _ = _lin._probe_fp8_cap(device)
            if mode != "off":
                model = getattr(runner, "model", None)
                lm = model.get_language_model() if hasattr(model, "get_language_model") else model
                inner = getattr(lm, "model", lm)
                layers = getattr(inner, "layers", None) or ()
                for layer in layers:
                    if want_dense:
                        attn = getattr(layer, "attn", None)
                        if attn is not None:
                            is_local = bool(getattr(layer, "is_local", True))
                            qkvr = getattr(attn, "qkvr", None)
                            wq = getattr(qkvr, "weight", None)
                            if (
                                isinstance(wq, torch.Tensor)
                                and wq.is_cuda
                                and wq.dtype == torch.bfloat16
                                and wq.ndim == 2
                                and wq.untyped_storage().size() > 0
                            ):
                                if not is_local and hasattr(attn, "q_side_splits"):
                                    wq_use = wq[: sum(attn.q_side_splits)]
                                else:
                                    wq_use = wq
                                _lin._get_cached_w_fp8(wq_use, mode)
                                summary["fp8_weights"] += 1
                            wo = getattr(attn, "wo_ud", None)
                            wwo = getattr(wo, "weight", None)
                            if (
                                isinstance(wwo, torch.Tensor)
                                and wwo.is_cuda
                                and wwo.dtype == torch.bfloat16
                                and wwo.ndim == 2
                                and wwo.untyped_storage().size() > 0
                            ):
                                _lin._get_cached_w_fp8(wwo, mode)
                                summary["fp8_weights"] += 1
                    if want_router:
                        mlp = getattr(layer, "mlp", None)
                        gate = getattr(mlp, "gate", None) if mlp is not None else None
                        wg = getattr(gate, "weight", None)
                        if (
                            isinstance(wg, torch.Tensor)
                            and wg.is_cuda
                            and wg.dtype == torch.bfloat16
                            and wg.ndim == 2
                            and (wg.untyped_storage().size() > 0 or getattr(wg, "_inkling_router_bf12", None) is not None)
                        ):
                            _lin._get_cached_w_fp8(wg, mode)
                            summary["fp8_weights"] += 1
    except Exception:  # noqa: BLE001
        pass

    # 3. Pre-JIT multi-req sconv_seq_metadata (skipped at B=1 by INKLING_SCONV_META_DEDUP=1!)
    try:
        from vllm.models.inkling.nvidia.ops.sconv import sconv_seq_metadata  # noqa: PLC0415

        qsl = torch.arange(0, 33, dtype=torch.int32, device=device) * 8
        seq = torch.zeros(256, dtype=torch.int32, device=device)
        qs = torch.zeros(256, dtype=torch.int32, device=device)
        sconv_seq_metadata(qsl, 32, 256, seq, qs, 256)
        summary["jits"] += 1
    except Exception:  # noqa: BLE001
        pass

    # 3b. Pre-JIT fused B=2..32 prepare_inputs kernels using REAL runner buffer dtypes
    # (fixes live r0 inference-time Triton JIT miss on _get_fused_dt_prep_bn_inline_kernel!)
    summary["jits"] += _prewarm_fused_dt_prep_kernels(runner)

    # 4. Sync warmup work cleanly before serving traffic
    try:
        torch.cuda.synchronize(device)
    except Exception:  # noqa: BLE001
        pass
    try:
        from . import host_pin as _hp_uma

        _hp_uma.reclaim_host_uma(empty_cuda=True, tag="post_prewarm_prefill")
    except Exception:  # noqa: BLE001
        pass
    summary["ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
    try:
        from vllm.distributed import get_tensor_model_parallel_rank  # noqa: PLC0415

        is_r0 = get_tensor_model_parallel_rank() == 0
    except Exception:  # noqa: BLE001
        is_r0 = True
    if is_r0:
        _logger().info(
            "INKLING_PREFILL_PREWARM: loaded %d prefill libs, warmed %d FP8 weights, "
            "JIT'd %d multi-req kernels in %.1f ms",
            summary["libs"],
            summary["fp8_weights"],
            summary["jits"],
            summary["ms"],
        )
    return summary


# ----------------------------------------------------------------------------------------------
# Lever C: Short-lived req_states.draft_tokens proxy across propose() -> set_draft_tokens()
# ----------------------------------------------------------------------------------------------
class _DraftSkipProxy:
    """Eliminates both eager index_elementwise_kernels in sample_tokens after _vw_conf_kernel
    already scatter-wrote req_states.draft_tokens[idx_mapping] (works for BOTH full-width B=1
    AND Dynamic-SD sliced B=2..32 where 0 < num_spec < num_speculative_steps!). Active ONLY
    between speculator.propose() return and draft_tokens_handler.set_draft_tokens() entry."""

    __slots__ = ("_real", "_idx", "_drafts", "_full", "_stats")

    def __init__(
        self,
        real: torch.Tensor,
        idx: torch.Tensor,
        drafts: torch.Tensor,
        stats=None,
        full_drafts: torch.Tensor | None = None,
    ):
        self._real = real
        self._idx = idx
        self._drafts = drafts
        self._full = full_drafts if full_drafts is not None else drafts
        self._stats = stats

    def reset(
        self,
        real: torch.Tensor,
        idx: torch.Tensor,
        drafts: torch.Tensor,
        stats=None,
        full_drafts: torch.Tensor | None = None,
    ):
        self._real = real
        self._idx = idx
        self._drafts = drafts
        self._full = full_drafts if full_drafts is not None else drafts
        self._stats = stats
        return self

    def __setitem__(self, key, value):
        if key is self._idx and (value is self._drafts or value is self._full):
            if self._stats is not None:
                self._stats["elided_draft_put"] += 1
            return
        if (
            isinstance(key, tuple)
            and len(key) == 2
            and key[0] is self._idx
            and isinstance(value, torch.Tensor)
            and value.data_ptr() == self._full.data_ptr()
            and value.stride() == self._full.stride()
        ):
            view = self._full[:, key[1]]
            if value.shape == view.shape:
                if self._stats is not None:
                    self._stats["elided_draft_put"] += 1
                return
        self._real[key] = value

    def __getitem__(self, key):
        if key is self._idx:
            if self._stats is not None:
                self._stats["elided_draft_get"] += 1
            return self._full
        if isinstance(key, tuple) and len(key) == 2 and key[0] is self._idx:
            if self._stats is not None:
                self._stats["elided_draft_get"] += 1
            return self._full[:, key[1]]
        return self._real[key]

    def __getattr__(self, name):
        return getattr(self._real, name)


def _get_model_set_direct_embed():
    try:
        from vllm.models.inkling2 import model as _m  # noqa: PLC0415

        return getattr(_m, "set_direct_embed_out", None)
    except Exception:  # noqa: BLE001
        return None


def _get_spec_draft_store_tls():
    try:
        from vllm.models.inkling2 import speculator as _sp  # noqa: PLC0415

        return getattr(_sp, "_DRAFT_STORE_TLS", None)
    except Exception:  # noqa: BLE001
        return None


def _logger():
    try:
        from vllm.logger import init_logger  # noqa: PLC0415

        return init_logger(__name__)
    except Exception:  # noqa: BLE001
        import logging

        return logging.getLogger(TAG)


def _get_state(runner) -> _RunnerState:
    st = getattr(runner, "_inkling_dtg", None)
    if st is None:
        st = runner._inkling_dtg = _RunnerState()
    return st


def _ensure_region(st: _RunnerState):
    if st.region is None:
        _install_hooks()
        st.region = Region(
            TorchBackend(CFG["update"], max_execs=CFG["max_execs"]),
            max_seg=CFG["seg"],
        )
        if st.rank0:
            _logger().info(
                "INKLING_DTGRAPH armed: regions=%s mode=%s warm=%d max_seg=%d max_reqs=%d max_execs=%d",
                sorted(CFG["regions"]),
                "update" if CFG["update"] else "instantiate",
                CFG["warm"],
                CFG["seg"],
                CFG["max_reqs"],
                CFG["max_execs"],
            )
    if FAST_REPLAY_V14 or FAST_REPLAY_V16:
        if not _ORIG.get("_rdma"):
            _hook_rdma()
        stream = torch.cuda.current_stream()
        cs = stream.cuda_stream
        if cs != st._last_nondef_cs:
            if st.region.b.is_default(stream):
                raise RuntimeError("current stream is the default stream")
            st._last_nondef_cs = cs
        return stream
    _hook_rdma()
    stream = torch.cuda.current_stream()
    if st.region.b.is_default(stream):
        raise RuntimeError("current stream is the default stream")
    return stream


def _get_sf_ring_mod_v14(st: _RunnerState):
    """Per-runner cached sync_free module lookup (zero sys.modules scan on steady-state steps)."""
    m = st._sf_ring_mod
    if m is not None:
        return m
    if st._sf_ring_mod_probed:
        return None
    m = _get_sync_free_ring_mod()
    if m is not None:
        st._sf_ring_mod = m
    else:
        st._sf_ring_mod_probed = True
    return m


_SF_RING_MOD_CACHE = None


def _get_sync_free_ring_mod():
    global _SF_RING_MOD_CACHE
    if not STATIC_RING:
        return None
    cached = _SF_RING_MOD_CACHE
    if cached is not None:
        mod_name = getattr(cached, "__name__", "")
        if sys.modules.get(mod_name) is cached:
            return cached
        _SF_RING_MOD_CACHE = None
    try:
        from vllm.models.inkling2.ops import sync_free as _sf_m  # noqa: PLC0415
        if hasattr(_sf_m, "get_static_ring_state"):
            _SF_RING_MOD_CACHE = _sf_m
            return _sf_m
    except Exception:  # noqa: BLE001
        pass
    try:
        import sync_free as _sf_m  # noqa: PLC0415
        if hasattr(_sf_m, "get_static_ring_state"):
            _SF_RING_MOD_CACHE = _sf_m
            return _sf_m
    except Exception:  # noqa: BLE001
        pass
    return None


def _opt_ptr(obj, attr: str) -> int:
    t = getattr(obj, attr, None)
    if t is not None and hasattr(t, "data_ptr"):
        return int(t.data_ptr())
    return 0


def _postprocess_args_match_ring(
    args: tuple, kwargs: dict, ring_state: tuple[int, int, int, int]
) -> bool:
    """Verify the actual tensors passed into postprocess_sampled match ring_state[1..3]."""
    want_s, want_ns, want_nr = ring_state[1], ring_state[2], ring_state[3]
    if want_s == 0 or want_ns == 0 or want_nr == 0:
        return False
    if FAST_REPLAY_V14:
        hit_s = False
        hit_ns = False
        hit_nr = False
        for val in args:
            if val is None:
                continue
            if isinstance(val, torch.Tensor):
                p = val.data_ptr()
                if p == want_s:
                    hit_s = True
                elif p == want_ns:
                    hit_ns = True
                elif p == want_nr:
                    hit_nr = True
                if hit_s and hit_ns and hit_nr:
                    return True
            else:
                t = getattr(val, "sampled_token_ids", None)
                if isinstance(t, torch.Tensor) and t.data_ptr() == want_s:
                    hit_s = True
                t = getattr(val, "num_sampled", None)
                if isinstance(t, torch.Tensor) and t.data_ptr() == want_ns:
                    hit_ns = True
                t = getattr(val, "num_rejected", None)
                if isinstance(t, torch.Tensor) and t.data_ptr() == want_nr:
                    hit_nr = True
                if hit_s and hit_ns and hit_nr:
                    return True
        if kwargs:
            for val in kwargs.values():
                if val is None:
                    continue
                if isinstance(val, torch.Tensor):
                    p = val.data_ptr()
                    if p == want_s:
                        hit_s = True
                    elif p == want_ns:
                        hit_ns = True
                    elif p == want_nr:
                        hit_nr = True
                    if hit_s and hit_ns and hit_nr:
                        return True
                else:
                    t = getattr(val, "sampled_token_ids", None)
                    if isinstance(t, torch.Tensor) and t.data_ptr() == want_s:
                        hit_s = True
                    t = getattr(val, "num_sampled", None)
                    if isinstance(t, torch.Tensor) and t.data_ptr() == want_ns:
                        hit_ns = True
                    t = getattr(val, "num_rejected", None)
                    if isinstance(t, torch.Tensor) and t.data_ptr() == want_nr:
                        hit_nr = True
                    if hit_s and hit_ns and hit_nr:
                        return True
        return hit_s and hit_ns and hit_nr
    ptrs: set[int] = set()
    for val in (*args, *kwargs.values()):
        if isinstance(val, torch.Tensor):
            ptrs.add(int(val.data_ptr()))
        elif val is not None:
            for attr in ("sampled_token_ids", "num_sampled", "num_rejected"):
                t = getattr(val, attr, None)
                if isinstance(t, torch.Tensor):
                    ptrs.add(int(t.data_ptr()))
    return (want_s in ptrs) and (want_ns in ptrs) and (want_nr in ptrs)


def _td_static_fingerprint(
    ib, num_spec_sched: int, ring_state: tuple[int, int, int, int], spec=None
) -> tuple[tuple, tuple]:
    req_ids = getattr(ib, "req_ids", ())
    num_reqs = int(getattr(ib, "num_reqs", 0) or 0)
    vw_sig: tuple = ()
    if spec is not None:
        draft_g = spec._next_draft_cap_g(num_reqs) if hasattr(spec, "_next_draft_cap_g") else None
        w_cap = (
            int(spec._effective_w_cap(num_reqs, num_spec_sched))
            if hasattr(spec, "_effective_w_cap")
            else int(num_spec_sched)
        )
        tau_ov = (
            float(spec._effective_tau_override(num_reqs))
            if hasattr(spec, "_effective_tau_override")
            else 0.0
        )
        hetero_active = (
            bool(spec._hetero_is_active_for_b(num_reqs))
            if hasattr(spec, "_hetero_is_active_for_b")
            else bool(getattr(spec, "_hetero_budget_on", False) and num_reqs >= 2)
        )
        sam_active = (
            bool(spec._sam_is_active_for_b(num_reqs))
            if hasattr(spec, "_sam_is_active_for_b")
            else bool(getattr(spec, "_sam_hybrid_on", False))
        )
        if (
            getattr(spec, "_hetero_budget_on", False)
            or getattr(spec, "_ctx_adaptive_tau_on", False)
            or getattr(spec, "_sam_hybrid_on", False)
        ):
            m_cap_b = (
                int(spec._hetero_m_cap_for_b(num_reqs))
                if hasattr(spec, "_hetero_m_cap_for_b")
                else int(getattr(spec, "_hetero_budget_m_cap", 0))
            )
            vw_sig = (
                draft_g,
                w_cap,
                tau_ov,
                hetero_active,
                m_cap_b if hetero_active else 0,
                bool(getattr(spec, "_ctx_adaptive_tau_on", False)),
                sam_active,
            )
        else:
            vw_sig = (draft_g, w_cap, tau_ov)
    batch_sig = (
        tuple(req_ids) if isinstance(req_ids, (list, tuple)) else (),
        num_reqs,
        int(getattr(ib, "num_tokens", 0) or 0),
        int(getattr(ib, "num_tokens_no_spec", 0) or 0),
        int(num_spec_sched),
        int(getattr(ib, "num_draft_tokens", 0) or 0),
        vw_sig,
    )
    ptr_sig = (
        _opt_ptr(ib, "idx_mapping"),
        _opt_ptr(ib, "query_start_loc"),
        _opt_ptr(ib, "positions"),
        _opt_ptr(ib, "input_ids"),
        _opt_ptr(ib, "seq_lens"),
        _opt_ptr(ib, "logits_indices"),
        _opt_ptr(ib, "cu_num_logits"),
        _opt_ptr(ib, "expanded_idx_mapping"),
        ring_state[1],
        ring_state[2],
        ring_state[3],
    )
    if FAST_REPLAY_V16 and STATIC_TEMP_REPLAY_V16 and spec is not None:
        _dl_t = getattr(spec, "draft_logits", None)
        if isinstance(_dl_t, torch.Tensor):
            ptr_sig = (*ptr_sig, int(_dl_t.data_ptr()), tuple(_dl_t.shape))
    return batch_sig, ptr_sig


def _td_static_spec_ok(spec, num_reqs: int, num_spec_sched: int | None = None) -> bool:
    if spec is None:
        return False
    if (
        getattr(spec, "_tree_spec_on", False)
        or getattr(spec, "_cl_on", False)
        or getattr(spec, "_vw_stats_on", False)
        or getattr(spec, "context_kv_only", False)
    ):
        return False
    _dl = getattr(spec, "draft_logits", None)
    if _dl is not None:
        if not (FAST_REPLAY_V16 and STATIC_TEMP_REPLAY_V16 and isinstance(_dl, torch.Tensor)):
            return False
    has_cap_lut = hasattr(spec, "_next_draft_cap_g")
    draft_g = spec._next_draft_cap_g(num_reqs) if has_cap_lut else None
    if getattr(spec, "_sam_hybrid_on", False):
        if getattr(spec, "_sam_mode", "") == "cpu":
            sam_max_b = int(getattr(spec, "_sam_max_b", 1))
            if num_reqs <= sam_max_b:
                return False
        else:
            sam_active = (
                bool(spec._sam_is_active_for_b(num_reqs))
                if hasattr(spec, "_sam_is_active_for_b")
                else bool(getattr(spec, "_sam_gpu_ready", False))
            )
            if sam_active:
                if not getattr(spec, "_sam_gpu_ready", False):
                    return False
                cap_sam = getattr(spec, "_sam_boost_in_graph_captured", ())
                if has_cap_lut:
                    if draft_g is None or draft_g not in cap_sam:
                        return False
                elif not any(g >= num_reqs for g in cap_sam):
                    return False
    if getattr(spec, "_vw_mode", "off") != "off":
        if not getattr(spec, "_vw_is_conf", False) or not getattr(spec, "_graph_vw_conf_captured", None):
            return False
        cap_vw = getattr(spec, "_graph_vw_conf_captured", {})
        if has_cap_lut:
            if draft_g is None or draft_g not in cap_vw:
                return False
            cap_entry = cap_vw[draft_g]
            if isinstance(cap_entry, tuple) and len(cap_entry) >= 2:
                want_w_cap = (
                    int(spec._effective_w_cap(num_reqs, num_spec_sched))
                    if hasattr(spec, "_effective_w_cap")
                    else int(cap_entry[0])
                )
                hetero_active = (
                    bool(spec._hetero_is_active_for_b(num_reqs))
                    if hasattr(spec, "_hetero_is_active_for_b")
                    else bool(getattr(spec, "_hetero_budget_on", False) and num_reqs >= 2)
                )
                want_tau_key = (
                    -2.0
                    if hetero_active
                    else (
                        float(spec._effective_tau_override(num_reqs))
                        if hasattr(spec, "_effective_tau_override")
                        else float(cap_entry[1])
                    )
                )
                if int(cap_entry[0]) != want_w_cap or float(cap_entry[1]) != want_tau_key:
                    return False
        elif not any(g >= num_reqs for g in cap_vw):
            return False
    return True


# ----------------------------------------------------------------------------------------------
# Wave-V10 Phase-4 ("dt", Drafter->Target) Fail-Closed Static Replay Helpers
# ----------------------------------------------------------------------------------------------
def _discover_sconv_builders(runner, st: _RunnerState) -> list:
    """Discover all InklingSconvMetadataBuilder instances attached to runner (cached once found)."""
    if st.dt_sconv_builders is not None:
        return st.dt_sconv_builders
    found: list = []
    seen_ids: set[int] = set()

    def _visit(obj, depth: int = 0) -> None:
        if obj is None or depth > 4:
            return
        oid = id(obj)
        if oid in seen_ids:
            return
        seen_ids.add(oid)
        if hasattr(obj, "seq_idx_buffer") and hasattr(obj, "query_start_buffer"):
            found.append(obj)
            return
        if isinstance(obj, (list, tuple)):
            for item in obj:
                _visit(item, depth + 1)
            return
        if isinstance(obj, dict):
            for item in obj.values():
                _visit(item, depth + 1)
            return
        for attr in ("attn_metadata_builder", "metadata_builder", "builder", "builders"):
            sub = getattr(obj, attr, None)
            if sub is not None:
                _visit(sub, depth + 1)

    for root_attr in (
        "attn_groups",
        "kv_cache_groups",
        "attn_metadata_builders",
        "sconv_builders",
    ):
        _visit(getattr(runner, root_attr, None), 0)
    if found:
        st.dt_sconv_builders = found
        _seen_sd: set[int] = set()
        _shared_list: list = []
        for _b in found:
            _sd = getattr(_b, "_inkling_shared", None)
            if isinstance(_sd, dict) and id(_sd) not in _seen_sd:
                _seen_sd.add(id(_sd))
                _shared_list.append(_sd)
        st._dt_sconv_shared = tuple(_shared_list)
    return found


def _sconv_any_dirty(builders: list, st: _RunnerState | None = None) -> bool:
    if FAST_REPLAY_V14 and st is not None and st._dt_sconv_shared:
        for sd in st._dt_sconv_shared:
            if sd.get("dirty", False):
                return True
        return False
    for b in builders:
        sd = getattr(b, "_inkling_shared", None)
        if isinstance(sd, dict) and bool(sd.get("dirty", False)):
            return True
    return False


def _sconv_mark_dirty(builders: list, num_padded_tokens: int, st: _RunnerState | None = None) -> None:
    if FAST_REPLAY_V14 and st is not None and st._dt_sconv_shared:
        for sd in st._dt_sconv_shared:
            sd["dirty"] = True
            if num_padded_tokens > (sd.get("dirty_len") or 0):
                sd["dirty_len"] = num_padded_tokens
        return
    seen: set[int] = set()
    for b in builders:
        sd = getattr(b, "_inkling_shared", None)
        if isinstance(sd, dict) and id(sd) not in seen:
            seen.add(id(sd))
            sd["dirty"] = True
            if num_padded_tokens > int(sd.get("dirty_len", 0) or 0):
                sd["dirty_len"] = int(num_padded_tokens)


def _is_tree_spec_active(runner, num_reqs: int | None = None, st: _RunnerState | None = None) -> bool:
    spec = getattr(runner, "speculator", None)
    if spec is not None and bool(getattr(spec, "_tree_spec_on", False)):
        if (
            num_reqs is not None
            and bool(getattr(spec, "_tree_spec_b1_only", False))
            and int(num_reqs) > int(getattr(spec, "_tree_spec_max_b", 1))
        ):
            return False
        return True
    if FAST_REPLAY_V14 and st is not None:
        m = st._tree_spec_mod
        if m is None and not st._tree_spec_mod_probed:
            for mod_name in ("vllm.models.inkling2.ops.tree_spec_runtime", "tree_spec_runtime"):
                m = sys.modules.get(mod_name)
                if m is not None:
                    st._tree_spec_mod = m
                    break
            st._tree_spec_mod_probed = True
        if m is not None and getattr(m, "TREE_SPEC_ENABLED", False):
            ts = getattr(m, "TREE_STATE", None)
            if isinstance(ts, dict) and (ts.get("scope") or 0) > 0 and not ts.get("suppressed", False):
                if (
                    num_reqs is not None
                    and getattr(m, "TREE_SPEC_B1_ONLY", False)
                    and num_reqs > getattr(m, "TREE_SPEC_MAX_B", 1)
                ):
                    return False
                return True
        return False
    for mod_name in ("vllm.models.inkling2.ops.tree_spec_runtime", "tree_spec_runtime"):
        m = sys.modules.get(mod_name)
        if m is not None and bool(getattr(m, "TREE_SPEC_ENABLED", False)):
            ts = getattr(m, "TREE_STATE", None)
            if isinstance(ts, dict) and int(ts.get("scope") or 0) > 0 and not ts.get("suppressed", False):
                if (
                    num_reqs is not None
                    and bool(getattr(m, "TREE_SPEC_B1_ONLY", False))
                    and int(num_reqs) > int(getattr(m, "TREE_SPEC_MAX_B", 1))
                ):
                    return False
                return True
    return False


def _probe_dt_block_table_mode(runner, st: _RunnerState) -> str:
    """Fail-closed probe: determine whether runner.block_tables.gather_block_tables /
    compute_slot_mappings depend on a dynamic CPU block count or max_seq_len scalar."""
    if st.dt_bt_mode is not None:
        return st.dt_bt_mode
    if STATIC_DT_FORCE_GPU_BT:
        st.dt_bt_mode = "gpu_fixed"
        return st.dt_bt_mode
    bt = getattr(runner, "block_tables", None)
    if bt is None:
        return "cpu_blocks"
    try:
        import inspect  # noqa: PLC0415

        gbt = getattr(bt, "gather_block_tables", None)
        csm = getattr(bt, "compute_slot_mappings", None)
        if gbt is None or csm is None:
            st.dt_bt_mode = "cpu_blocks"
            return st.dt_bt_mode
        src = inspect.getsource(gbt) + "\n" + inspect.getsource(csm)
        sig = inspect.signature(gbt)
        bad_param = any(
            p in sig.parameters
            for p in ("max_blocks", "max_num_blocks", "max_seq_len", "seq_lens_cpu")
        )
        bad_toks = (
            "seq_lens_cpu",
            "max_seq_len",
            "num_blocks_cpu",
            "num_blocks_np",
            "max_num_blocks_cpu",
            ".item()",
        )
        if bad_param or any(tok in src for tok in bad_toks):
            st.dt_bt_mode = "cpu_blocks"
        else:
            st.dt_bt_mode = "gpu_fixed"
    except Exception:  # noqa: BLE001
        st.dt_bt_mode = "cpu_blocks"
    return st.dt_bt_mode


def _tensor_seq_ptrs(seq) -> tuple:
    if seq is None:
        return ()
    if isinstance(seq, torch.Tensor):
        return (int(seq.data_ptr()), tuple(seq.shape), tuple(seq.stride()))
    if hasattr(seq, "gpu") and isinstance(seq.gpu, torch.Tensor):
        t = seq.gpu
        return (int(t.data_ptr()), tuple(t.shape), tuple(t.stride()))
    if isinstance(seq, (list, tuple)):
        out = []
        for item in seq:
            if isinstance(item, torch.Tensor):
                out.append((int(item.data_ptr()), tuple(item.shape), tuple(item.stride())))
            elif hasattr(item, "gpu") and isinstance(item.gpu, torch.Tensor):
                t = item.gpu
                out.append((int(t.data_ptr()), tuple(t.shape), tuple(t.stride())))
            else:
                out.append(0)
        return tuple(out)
    return ()


def _collect_dt_block_table_ptrs(runner) -> tuple:
    bt = getattr(runner, "block_tables", None)
    if bt is None:
        return ()
    return (
        _tensor_seq_ptrs(getattr(bt, "block_tables", None)),
        _tensor_seq_ptrs(getattr(bt, "input_block_tables", None)),
        _tensor_seq_ptrs(getattr(bt, "slot_mappings", None)),
        _tensor_seq_ptrs(getattr(bt, "num_blocks", None)),
    )


def _collect_dt_model_ptrs(runner) -> tuple:
    model = getattr(runner, "model", None)
    if model is None:
        return ()
    lm = model.get_language_model() if hasattr(model, "get_language_model") else model
    inner = getattr(lm, "model", lm)
    emb = getattr(inner, "embed_tokens", None)
    emb_w = getattr(emb, "_repl_weight", None)
    if not isinstance(emb_w, torch.Tensor):
        emb_w = getattr(emb, "weight", None)
    enorm = getattr(inner, "embed_norm", None)
    enorm_w = getattr(enorm, "weight", None) if enorm is not None else None
    vw_mask = getattr(inner, "_vw_mask", None)
    return (
        bool(getattr(inner, "_vw", False)),
        int(vw_mask.data_ptr()) if isinstance(vw_mask, torch.Tensor) else 0,
        int(emb_w.data_ptr()) if isinstance(emb_w, torch.Tensor) else 0,
        int(enorm_w.data_ptr()) if isinstance(enorm_w, torch.Tensor) else 0,
    )


def _collect_dt_sconv_ptrs(builders: list) -> tuple:
    out = []
    for b in builders:
        out.append(
            (
                _opt_ptr(b, "seq_idx_buffer"),
                _opt_ptr(b, "query_start_buffer"),
            )
        )
    return tuple(out)


def _dt_static_fingerprint(
    runner,
    st: _RunnerState,
    scheduler_output,
    batch_req_state,
    batch_desc,
    *,
    is_b1: bool,
    req_ids,
    idx_mapping_np,
    num_reqs: int,
    num_tokens: int,
    num_tokens_after_padding: int,
    num_reqs_padded: int,
    q_len: int,
    has_drafts: bool,
    write_pad: bool,
    imm_idx_ptr: int = 0,
    sconv_builders: list | None = None,
    idx_tuple: tuple[int, ...] | None = None,
) -> tuple[tuple, tuple] | None:
    """Compute complete fail-closed (batch_sig, ptr_sig) for Phase-4 ("dt") static replay."""
    ib_buf = getattr(runner, "input_buffers", None)
    rs_buf = getattr(runner, "req_states", None)
    if ib_buf is None or rs_buf is None:
        return None
    prep_bufs = st.b1_bufs if is_b1 else st.bn_bufs
    if prep_bufs is None:
        return None
    last_sampled = getattr(rs_buf, "last_sampled_tokens", None)
    if not isinstance(last_sampled, torch.Tensor):
        return None
    draft_tokens_buf = getattr(rs_buf, "draft_tokens", None)
    if isinstance(draft_tokens_buf, _DraftSkipProxy):
        return None
    if has_drafts:
        if (
            not isinstance(draft_tokens_buf, torch.Tensor)
            or draft_tokens_buf.dim() != 2
            or draft_tokens_buf.stride(1) != 1
        ):
            return None
        draft_ptr = draft_tokens_buf
        draft_stride = int(draft_tokens_buf.stride(0))
    else:
        draft_ptr = last_sampled
        draft_stride = int(last_sampled.stride(0))

    bt_mode = _probe_dt_block_table_mode(runner, st)
    if bt_mode == "cpu_blocks":
        nst_np = batch_req_state.num_scheduled_tokens
        ncomp_np = rs_buf.num_computed_tokens_np[idx_mapping_np]
        max_sl = int(np.max(ncomp_np + nst_np)) if num_reqs > 0 else 0
        bszs = getattr(getattr(runner, "block_tables", None), "block_sizes", None)
        if bszs:
            max_blocks_sig = tuple((max_sl + int(bs) - 1) // int(bs) for bs in bszs)
        else:
            max_blocks_sig = (max_sl,)
    else:
        max_blocks_sig = ()

    batch_sig = (
        bool(is_b1),
        int(num_reqs),
        int(num_tokens),
        int(num_tokens_after_padding),
        int(num_reqs_padded),
        int(q_len),
        bool(has_drafts),
        bool(write_pad),
        int(draft_stride),
        int(last_sampled.stride(0)),
        int(ib_buf.query_start_loc.shape[0]),
        int(ib_buf.seq_lens.shape[0]),
        idx_tuple if idx_tuple is not None else tuple(int(x) for x in idx_mapping_np),
        tuple(req_ids),
        bool(getattr(scheduler_output, "has_structured_output_requests", False)),
        bool(getattr(batch_req_state, "decode_graph_eligible", True)),
        (
            getattr(batch_desc.cg_mode, "name", str(batch_desc.cg_mode)),
            int(batch_desc.num_tokens),
            int(batch_desc.num_reqs or 0),
            bool(getattr(batch_desc, "uniform", False)),
        ),
        max_blocks_sig,
    )
    builders = sconv_builders if sconv_builders is not None else _discover_sconv_builders(runner, st)
    ncomp_obj = getattr(rs_buf, "num_computed_tokens", None)
    ncomp_gpu = getattr(ncomp_obj, "gpu", ncomp_obj)
    ie_buf = st.inputs_embeds_buf
    ptr_sig = (
        _opt_ptr(ib_buf, "input_ids"),
        _opt_ptr(ib_buf, "positions"),
        _opt_ptr(ib_buf, "seq_lens"),
        _opt_ptr(ib_buf, "query_start_loc"),
        _opt_ptr(ib_buf, "is_padding"),
        int(ncomp_gpu.data_ptr()) if isinstance(ncomp_gpu, torch.Tensor) else 0,
        int(last_sampled.data_ptr()),
        int(draft_ptr.data_ptr()),
        int(prep_bufs["idx"].data_ptr()),
        int(prep_bufs["cu"].data_ptr()),
        int(prep_bufs["expanded_idx"].data_ptr()),
        int(prep_bufs["expanded_pos"].data_ptr()),
        int(prep_bufs["logits_indices"].data_ptr()),
        int(imm_idx_ptr),
        int(ie_buf.data_ptr()) if isinstance(ie_buf, torch.Tensor) else 0,
        _collect_dt_model_ptrs(runner),
        _collect_dt_block_table_ptrs(runner),
        _collect_dt_sconv_ptrs(builders),
    )
    return batch_sig, ptr_sig


def _compile_dt_v14_plan(
    cert: dict,
    runner,
    st: _RunnerState,
    batch_desc,
    *,
    is_b1: bool,
    req_ids,
    idx_tuple: tuple[int, ...],
    num_reqs: int,
    num_tokens: int,
    num_tokens_after_padding: int,
    num_reqs_padded: int,
    q_len: int,
    has_drafts: bool,
    write_pad: bool,
    imm_idx_ptr: int,
    sconv_builders: list,
    has_so: bool,
    dge: bool,
    exec_key: tuple,
) -> None:
    """Compile zero-alloc flat pointer & batch validation plan + exec bundle on a certified 'dt' entry."""
    ib_buf = getattr(runner, "input_buffers", None)
    rs_buf = getattr(runner, "req_states", None)
    prep_bufs = st.b1_bufs if is_b1 else st.bn_bufs
    if ib_buf is None or rs_buf is None or prep_bufs is None:
        return
    last_sampled = getattr(rs_buf, "last_sampled_tokens", None)
    if not isinstance(last_sampled, torch.Tensor):
        return
    draft_tokens_buf = getattr(rs_buf, "draft_tokens", None)
    if has_drafts:
        if not isinstance(draft_tokens_buf, torch.Tensor):
            return
        draft_t = draft_tokens_buf
        draft_attr = "draft_tokens"
    else:
        draft_t = last_sampled
        draft_attr = "last_sampled_tokens"

    # Core static buffers verified directly via PEP-659 LOAD_ATTR (zero getattr/dict-hash overhead)
    t_ids = getattr(ib_buf, "input_ids", None)
    t_pos = getattr(ib_buf, "positions", None)
    t_sl = getattr(ib_buf, "seq_lens", None)
    t_qsl = getattr(ib_buf, "query_start_loc", None)
    t_ip = getattr(ib_buf, "is_padding", None)
    p_ids = int(t_ids.data_ptr()) if isinstance(t_ids, torch.Tensor) else 0
    p_pos = int(t_pos.data_ptr()) if isinstance(t_pos, torch.Tensor) else 0
    p_sl = int(t_sl.data_ptr()) if isinstance(t_sl, torch.Tensor) else 0
    p_qsl = int(t_qsl.data_ptr()) if isinstance(t_qsl, torch.Tensor) else 0
    p_ip = int(t_ip.data_ptr()) if isinstance(t_ip, torch.Tensor) else 0
    p_ls = int(last_sampled.data_ptr())
    p_dr = int(draft_t.data_ptr())
    ncomp_obj = getattr(rs_buf, "num_computed_tokens", None)
    ncomp_has_gpu = bool(ncomp_obj is not None and hasattr(ncomp_obj, "gpu"))
    t_nc = getattr(ncomp_obj, "gpu", None) if ncomp_has_gpu else ncomp_obj
    p_nc = int(t_nc.data_ptr()) if isinstance(t_nc, torch.Tensor) else 0
    ie_buf = st.inputs_embeds_buf
    p_ie = int(ie_buf.data_ptr()) if isinstance(ie_buf, torch.Tensor) else 0

    # Optional model & sconv pointers using C-level operator.attrgetter (0 getattr string overhead)
    extra_checks: list[tuple] = []
    inner = None
    exp_vw = False
    model = getattr(runner, "model", None)
    if model is not None:
        lm = model.get_language_model() if hasattr(model, "get_language_model") else model
        inner = getattr(lm, "model", lm)
        exp_vw = bool(getattr(inner, "_vw", False))
        vw_mask = getattr(inner, "_vw_mask", None)
        if hasattr(inner, "_vw_mask"):
            extra_checks.append((inner, operator.attrgetter("_vw_mask"), vw_mask, int(vw_mask.data_ptr()) if isinstance(vw_mask, torch.Tensor) else 0))
        emb = getattr(inner, "embed_tokens", None)
        if emb is not None:
            emb_rw = getattr(emb, "_repl_weight", None)
            if isinstance(emb_rw, torch.Tensor):
                extra_checks.append((emb, operator.attrgetter("_repl_weight"), emb_rw, int(emb_rw.data_ptr())))
            elif hasattr(emb, "weight"):
                emb_w = getattr(emb, "weight", None)
                extra_checks.append((emb, operator.attrgetter("weight"), emb_w, int(emb_w.data_ptr()) if isinstance(emb_w, torch.Tensor) else 0))
        enorm = getattr(inner, "embed_norm", None)
        if enorm is not None and hasattr(enorm, "weight"):
            en_w = getattr(enorm, "weight", None)
            extra_checks.append((enorm, operator.attrgetter("weight"), en_w, int(en_w.data_ptr()) if isinstance(en_w, torch.Tensor) else 0))

    _ag_seq_idx = operator.attrgetter("seq_idx_buffer")
    _ag_qstart = operator.attrgetter("query_start_buffer")
    for b in sconv_builders:
        t_si = getattr(b, "seq_idx_buffer", None)
        t_qs = getattr(b, "query_start_buffer", None)
        extra_checks.append((b, _ag_seq_idx, t_si, int(t_si.data_ptr()) if isinstance(t_si, torch.Tensor) else 0))
        extra_checks.append((b, _ag_qstart, t_qs, int(t_qs.data_ptr()) if isinstance(t_qs, torch.Tensor) else 0))

    # Prep buffer core tensors (unpacked directly, zero dict loop)
    pb_idx = prep_bufs["idx"]
    pb_cu = prep_bufs["cu"]
    pb_eidx = prep_bufs["expanded_idx"]
    pb_epos = prep_bufs["expanded_pos"]
    pb_li = prep_bufs["logits_indices"]
    prep_core = (
        pb_idx, int(pb_idx.data_ptr()),
        pb_cu, int(pb_cu.data_ptr()),
        pb_eidx, int(pb_eidx.data_ptr()),
        pb_epos, int(pb_epos.data_ptr()),
        pb_li, int(pb_li.data_ptr()),
    )

    # Flattened block-table checks using C-level attrgetter + zero-branch inner loops!
    bt = getattr(runner, "block_tables", None)
    bt_seq_hdrs: list[tuple] = []      # (attrgetter, exp_seq, exp_len_or_neg1)
    bt_flat_tensors: list[tuple] = []  # (seq, idx_i, exp_t, exp_ptr, exp_sh, exp_st)
    bt_flat_gpu: list[tuple] = []      # (seq, idx_i, exp_item, exp_gpu, exp_ptr, exp_sh, exp_st)
    bt_fallback: list[tuple] = []      # uncommon direct tensor / None / opaque entries
    if bt is not None:
        for attr in ("block_tables", "input_block_tables", "slot_mappings", "num_blocks"):
            ag_bt = operator.attrgetter(attr)
            if not hasattr(bt, attr):
                bt_fallback.append((bt, attr, 0, None, ()))
                continue
            seq = ag_bt(bt)
            if isinstance(seq, (list, tuple)):
                all_plain = True
                for idx_i, item in enumerate(seq):
                    if isinstance(item, torch.Tensor):
                        bt_flat_tensors.append((seq, idx_i, item, int(item.data_ptr()), item.shape, item.stride()))
                    elif hasattr(item, "gpu") and isinstance(item.gpu, torch.Tensor):
                        t = item.gpu
                        bt_flat_gpu.append((seq, idx_i, item, t, int(t.data_ptr()), t.shape, t.stride()))
                    else:
                        all_plain = False
                        break
                if all_plain:
                    bt_seq_hdrs.append((ag_bt, seq, len(seq)))
                else:
                    bt_fallback.append((bt, attr, 4, seq, ()))
            elif seq is None:
                bt_fallback.append((bt, attr, 0, None, ()))
            elif isinstance(seq, torch.Tensor):
                bt_fallback.append((bt, attr, 1, seq, (int(seq.data_ptr()), seq.shape, seq.stride())))
            elif hasattr(seq, "gpu") and isinstance(seq.gpu, torch.Tensor):
                t = seq.gpu
                bt_fallback.append((bt, attr, 2, seq, (t, int(t.data_ptr()), t.shape, t.stride())))
            else:
                bt_fallback.append((bt, attr, 4, seq, ()))

    # Exec bundle (pre-resolved owner + cudaGraphExec_t handle)
    exec_bundle = None
    if st.region is not None and hasattr(st.region.b, "get_static_exec_bundle"):
        exec_bundle = st.region.b.get_static_exec_bundle(exec_key)

    bt_mode = _probe_dt_block_table_mode(runner, st)
    bszs = tuple(int(bs) for bs in getattr(bt, "block_sizes", ()) or ()) if (bt is not None and bt_mode == "cpu_blocks") else ()
    exec_owner = exec_bundle[0] if exec_bundle is not None else None
    exec_handle = exec_bundle[1] if exec_bundle is not None else None
    ie_view = ie_buf[:num_tokens] if (isinstance(ie_buf, torch.Tensor) and ie_buf.dim() == 2 and ie_buf.is_contiguous() and ie_buf.shape[0] >= num_tokens) else None
    req_ids_t = tuple(req_ids)
    bd_cg_name = getattr(batch_desc.cg_mode, "name", str(batch_desc.cg_mode))
    bd_nt = int(batch_desc.num_tokens)
    bd_nr = int(batch_desc.num_reqs or 0)
    bd_uni = bool(getattr(batch_desc, "uniform", False))

    # Flat positional tuple unpacked in 1 UNPACK_SEQUENCE instruction inside _dt_v14_validate!
    cert["v14_dt_t"] = (
        exec_owner,
        bool(is_b1), num_reqs, num_tokens, num_tokens_after_padding, num_reqs_padded,
        q_len, bool(has_drafts), bool(write_pad), int(imm_idx_ptr),
        idx_tuple, idx_tuple[0] if num_reqs == 1 else -1,
        req_ids_t, req_ids_t[0] if num_reqs == 1 else None,
        bool(has_so), bool(dge),
        batch_desc, bd_cg_name, bd_nt, bd_nr, bd_uni,
        bt_mode == "cpu_blocks", bszs, cert["batch_sig"][-1],
        ib_buf, rs_buf, bt, prep_bufs, inner, exp_vw,
        t_ids, p_ids, t_pos, p_pos, t_sl, p_sl, t_qsl, p_qsl, t_ip, p_ip,
        last_sampled, p_ls, draft_t, p_dr,
        ncomp_obj, ncomp_has_gpu, t_nc, p_nc,
        ie_buf, p_ie,
        int(ib_buf.query_start_loc.shape[0]), int(ib_buf.seq_lens.shape[0]),
        int(draft_t.stride(0)), int(last_sampled.stride(0)),
        prep_core, tuple(extra_checks),
        tuple(bt_seq_hdrs), tuple(bt_flat_tensors), tuple(bt_flat_gpu), tuple(bt_fallback),
    )

    cert["v14"] = {
        "is_b1": bool(is_b1),
        "num_reqs": num_reqs,
        "num_tokens": num_tokens,
        "nt_pad": num_tokens_after_padding,
        "nr_pad": num_reqs_padded,
        "q_len": q_len,
        "has_drafts": bool(has_drafts),
        "write_pad": bool(write_pad),
        "exec_key": exec_key,
        "exec_owner": exec_owner,
        "exec_handle": exec_handle,
        "exec_gen": cert["exec_gen"],
        "ie_buf": ie_buf,
        "ie_ptr": p_ie,
        "ie_view": ie_view,
    }


def _dt_v14_validate(
    cert: dict,
    runner,
    st: _RunnerState,
    scheduler_output,
    batch_req_state,
    batch_desc,
    *,
    is_b1: bool,
    req_ids,
    eff_idx_np,
    num_reqs: int,
    nt_s: int,
    nt_pad_s: int,
    nr_pad_s: int,
    ql_s: int,
    has_d_s: bool,
    w_pad_s: bool,
    imm_ptr: int,
) -> bool:
    """Zero-alloc PEP-659 LOAD_ATTR fail-closed validator mathematically equivalent to _dt_static_fingerprint."""
    vt = cert.get("v14_dt_t")
    if vt is None:
        return False
    (
        exec_owner,
        exp_b1, exp_nr, exp_nt, exp_nt_pad, exp_nr_pad,
        exp_ql, exp_has_d, exp_wpad, exp_imm,
        exp_idx, exp_idx0, exp_rids, exp_rid0,
        exp_so, exp_dge,
        bd_ref, bd_cg_name, bd_nt, bd_nr, bd_uni,
        bt_mode_cpu, bt_bszs, exp_mbs,
        ib_buf, rs_buf, bt_ref, prep_bufs, inner, exp_vw,
        t_ids, p_ids, t_pos, p_pos, t_sl, p_sl, t_qsl, p_qsl, t_ip, p_ip,
        t_ls, p_ls, t_dr, p_dr,
        ncomp_obj, ncomp_has_gpu, t_nc, p_nc,
        ie_buf, p_ie,
        qsl_len, sl_len, dr_stride, ls_stride,
        prep_core, extra_checks,
        bt_seq_hdrs, bt_flat_tensors, bt_flat_gpu, bt_fallback,
    ) = vt
    if (
        exec_owner is None
        or exp_nr != num_reqs
        or exp_nt != nt_s
        or exp_nt_pad != nt_pad_s
        or exp_nr_pad != nr_pad_s
        or exp_ql != ql_s
        or exp_b1 != is_b1
        or exp_has_d != has_d_s
        or exp_wpad != w_pad_s
        or exp_imm != imm_ptr
    ):
        return False
    if num_reqs == 1:
        if int(eff_idx_np[0]) != exp_idx0 or req_ids[0] != exp_rid0:
            return False
    else:
        if len(eff_idx_np) != num_reqs or len(req_ids) != num_reqs:
            return False
        for i in range(num_reqs):
            if int(eff_idx_np[i]) != exp_idx[i] or req_ids[i] != exp_rids[i]:
                return False
    try:
        if bool(scheduler_output.has_structured_output_requests) != exp_so:
            return False
    except AttributeError:
        if exp_so:
            return False
    try:
        if bool(batch_req_state.decode_graph_eligible) != exp_dge:
            return False
    except AttributeError:
        if not exp_dge:
            return False
    if batch_desc is not bd_ref:
        if (
            getattr(batch_desc.cg_mode, "name", str(batch_desc.cg_mode)) != bd_cg_name
            or int(batch_desc.num_tokens) != bd_nt
            or int(batch_desc.num_reqs or 0) != bd_nr
            or bool(getattr(batch_desc, "uniform", False)) != bd_uni
        ):
            return False
    if bt_mode_cpu:
        nst_np = batch_req_state.num_scheduled_tokens
        rc_np = runner.req_states.num_computed_tokens_np
        max_sl = 0
        for i in range(num_reqs):
            sl_i = int(rc_np[int(eff_idx_np[i])]) + int(nst_np[i])
            if sl_i > max_sl:
                max_sl = sl_i
        if bt_bszs:
            for bi in range(len(bt_bszs)):
                bs = bt_bszs[bi]
                if ((max_sl + bs - 1) // bs) != exp_mbs[bi]:
                    return False
        elif (max_sl,) != exp_mbs:
            return False
    try:
        if runner.input_buffers is not ib_buf or runner.req_states is not rs_buf or runner.block_tables is not bt_ref:
            return False
        if (st.b1_bufs if is_b1 else st.bn_bufs) is not prep_bufs:
            return False
        if inner is not None and bool(inner._vw) != exp_vw:
            return False
        # Core input_buffers & req_states direct LOAD_ATTR checks (0 getattr!)
        if (
            ib_buf.input_ids is not t_ids or (p_ids and t_ids.data_ptr() != p_ids)
            or ib_buf.positions is not t_pos or (p_pos and t_pos.data_ptr() != p_pos)
            or ib_buf.seq_lens is not t_sl or (p_sl and t_sl.data_ptr() != p_sl)
            or ib_buf.query_start_loc is not t_qsl or (p_qsl and t_qsl.data_ptr() != p_qsl)
            or ib_buf.is_padding is not t_ip or (p_ip and t_ip.data_ptr() != p_ip)
            or rs_buf.last_sampled_tokens is not t_ls or t_ls.data_ptr() != p_ls
            or (rs_buf.draft_tokens if has_d_s else rs_buf.last_sampled_tokens) is not t_dr or t_dr.data_ptr() != p_dr
            or rs_buf.num_computed_tokens is not ncomp_obj
            or (ncomp_obj.gpu if ncomp_has_gpu else ncomp_obj) is not t_nc or (p_nc and t_nc.data_ptr() != p_nc)
            or st.inputs_embeds_buf is not ie_buf or (p_ie and ie_buf.data_ptr() != p_ie)
        ):
            return False
        if t_qsl.shape[0] != qsl_len or t_sl.shape[0] != sl_len:
            return False
        if t_dr.stride(0) != dr_stride or t_ls.stride(0) != ls_stride:
            return False
        if has_d_s and (t_dr.dim() != 2 or t_dr.stride(1) != 1):
            return False
        # Prep buffer dict direct checks (unpacked once)
        pb_idx, p_pidx, pb_cu, p_pcu, pb_eidx, p_peidx, pb_epos, p_pepos, pb_li, p_pli = prep_core
        if (
            prep_bufs["idx"] is not pb_idx or pb_idx.data_ptr() != p_pidx
            or prep_bufs["cu"] is not pb_cu or pb_cu.data_ptr() != p_pcu
            or prep_bufs["expanded_idx"] is not pb_eidx or pb_eidx.data_ptr() != p_peidx
            or prep_bufs["expanded_pos"] is not pb_epos or pb_epos.data_ptr() != p_pepos
            or prep_bufs["logits_indices"] is not pb_li or pb_li.data_ptr() != p_pli
        ):
            return False
        # Optional model & sconv pointers via C attrgetter
        for obj, ag, exp_t, exp_ptr in extra_checks:
            cur = ag(obj)
            if cur is not exp_t or (exp_ptr and cur.data_ptr() != exp_ptr):
                return False
        # Flattened block-table checks (0 branch per tensor!)
        if bt_ref is not None:
            for ag_bt, exp_seq, exp_len in bt_seq_hdrs:
                cur_seq = ag_bt(bt_ref)
                if cur_seq is not exp_seq or len(cur_seq) != exp_len:
                    return False
            for seq, idx_i, exp_t, exp_ptr, exp_sh, exp_st in bt_flat_tensors:
                cur_it = seq[idx_i]
                if cur_it is not exp_t or cur_it.data_ptr() != exp_ptr or cur_it.shape != exp_sh or cur_it.stride() != exp_st:
                    return False
            for seq, idx_i, exp_item, exp_gpu, exp_ptr, exp_sh, exp_st in bt_flat_gpu:
                cur_it = seq[idx_i]
                if cur_it is not exp_item:
                    return False
                t_g = cur_it.gpu
                if t_g is not exp_gpu or t_g.data_ptr() != exp_ptr or t_g.shape != exp_sh or t_g.stride() != exp_st:
                    return False
            if bt_fallback:
                for bt_obj, attr, kind, exp_seq, extra in bt_fallback:
                    cur_seq = getattr(bt_obj, attr, None)
                    if kind == 0:
                        if cur_seq is not None:
                            return False
                    elif kind == 1:
                        if cur_seq is not exp_seq or cur_seq.data_ptr() != extra[0] or cur_seq.shape != extra[1] or cur_seq.stride() != extra[2]:
                            return False
                    elif kind == 2:
                        if cur_seq is not exp_seq:
                            return False
                        t = getattr(cur_seq, "gpu", None)
                        if t is not extra[0] or t.data_ptr() != extra[1] or t.shape != extra[2] or t.stride() != extra[3]:
                            return False
                    else:
                        if cur_seq is not exp_seq:
                            return False
    except (AttributeError, KeyError, IndexError, TypeError):
        return False
    if ie_buf is None or ie_buf.shape[0] < nt_s:
        return False
    return True


def _compile_td_v14_plan(cert: dict, spec, st: _RunnerState, drafts_t: torch.Tensor | None) -> None:
    """Pre-resolve exec bundle, flat positional fingerprint, & cached draft slice views on a 'td' cert."""
    b_sig = cert["batch_sig"]
    p_sig = cert["ptr_sig"]
    exec_key = cert["exec_key"]
    exec_bundle = None
    if st.region is not None and hasattr(st.region.b, "get_static_exec_bundle"):
        exec_bundle = st.region.b.get_static_exec_bundle(exec_key)
    nr = int(cert["num_reqs"])
    dcols = int(cert["drafts_cols"])
    draft_tok_t = getattr(spec, "draft_tokens", None) if spec is not None else None
    draft_tok_ptr = int(draft_tok_t.data_ptr()) if isinstance(draft_tok_t, torch.Tensor) else 0
    drafts_view = draft_tok_t[:nr, :dcols] if (isinstance(draft_tok_t, torch.Tensor) and drafts_t is not None) else None
    full_drafts_view = draft_tok_t[:nr] if (isinstance(draft_tok_t, torch.Tensor) and drafts_t is not None) else None
    req_ids_t = b_sig[0]
    vw_sig = b_sig[6]
    exec_owner = exec_bundle[0] if exec_bundle is not None else None
    exec_handle = exec_bundle[1] if exec_bundle is not None else None
    cert["v14_td_t"] = (
        exec_owner,
        req_ids_t, req_ids_t[0] if len(req_ids_t) == 1 else None,
        int(b_sig[1]), int(b_sig[2]), int(b_sig[3]), int(b_sig[4]), int(b_sig[5]),
        vw_sig,
        p_sig[0], p_sig[1], p_sig[2], p_sig[3], p_sig[4], p_sig[5], p_sig[6], p_sig[7],
        p_sig[8], p_sig[9], p_sig[10],
    )
    cert["v14"] = {
        "exec_key": exec_key,
        "exec_owner": exec_owner,
        "exec_handle": exec_handle,
        "exec_gen": cert.get("exec_gen"),
        "draft_tok_t": draft_tok_t,
        "draft_tok_ptr": draft_tok_ptr,
        "drafts_view": drafts_view,
        "full_drafts_view": full_drafts_view,
    }


def _td_v14_validate(
    cert: dict,
    ib,
    num_spec_sched: int,
    ring_state: tuple[int, int, int, int],
    spec,
    st: _RunnerState,
) -> bool:
    """Unified single-pass PEP-659 LOAD_ATTR validator replacing _td_static_fingerprint + _td_static_spec_ok."""
    vt = cert.get("v14_td_t")
    if vt is None or spec is None:
        return False
    (
        exec_owner,
        exp_rids, exp_rid0,
        exp_nr, exp_nt, exp_nt_nospec, exp_nspec, exp_ndraft,
        exp_vw,
        p_idx, p_qsl, p_pos, p_ids, p_sl, p_li, p_cu, p_eidx,
        p_s, p_ns, p_nr,
    ) = vt
    if exec_owner is None:
        return False
    nr = ib.num_reqs
    if (
        nr != exp_nr
        or ib.num_tokens != exp_nt
        or num_spec_sched != exp_nspec
        or ring_state[1] != p_s
        or ring_state[2] != p_ns
        or ring_state[3] != p_nr
    ):
        return False
    try:
        if ib.num_tokens_no_spec != exp_nt_nospec or ib.num_draft_tokens != exp_ndraft:
            return False
    except AttributeError:
        if getattr(ib, "num_tokens_no_spec", 0) != exp_nt_nospec or getattr(ib, "num_draft_tokens", 0) != exp_ndraft:
            return False
    req_ids = ib.req_ids
    if nr == 1:
        if not req_ids or req_ids[0] != exp_rid0:
            return False
    else:
        if len(req_ids) != nr:
            return False
        for i in range(nr):
            if req_ids[i] != exp_rids[i]:
                return False
    # Direct LOAD_ATTR pointer checks on InputBatch (0 getattr, 0 tuple alloc!)
    try:
        t = ib.idx_mapping
        if (t.data_ptr() if t is not None else 0) != p_idx:
            return False
        t = ib.query_start_loc
        if (t.data_ptr() if t is not None else 0) != p_qsl:
            return False
        t = ib.positions
        if (t.data_ptr() if t is not None else 0) != p_pos:
            return False
        t = ib.input_ids
        if (t.data_ptr() if t is not None else 0) != p_ids:
            return False
        t = ib.seq_lens
        if (t.data_ptr() if t is not None else 0) != p_sl:
            return False
        t = ib.logits_indices
        if (t.data_ptr() if t is not None else 0) != p_li:
            return False
        t = ib.cu_num_logits
        if (t.data_ptr() if t is not None else 0) != p_cu:
            return False
        t = ib.expanded_idx_mapping
        if (t.data_ptr() if t is not None else 0) != p_eidx:
            return False
    except AttributeError:
        return False

    # Single-pass spec validation via direct LOAD_ATTR (falls back on missing optional attrs)
    try:
        if (
            spec._tree_spec_on
            or spec._cl_on
            or spec._vw_stats_on
            or spec.context_kv_only
            or spec.draft_logits is not None
        ):
            return False
        hetero_on = bool(spec._hetero_budget_on)
        ctx_tau_on = bool(spec._ctx_adaptive_tau_on)
        sam_on = bool(spec._sam_hybrid_on)
    except AttributeError:
        if (
            getattr(spec, "_tree_spec_on", False)
            or getattr(spec, "_cl_on", False)
            or getattr(spec, "_vw_stats_on", False)
            or getattr(spec, "context_kv_only", False)
            or getattr(spec, "draft_logits", None) is not None
        ):
            return False
        hetero_on = bool(getattr(spec, "_hetero_budget_on", False))
        ctx_tau_on = bool(getattr(spec, "_ctx_adaptive_tau_on", False))
        sam_on = bool(getattr(spec, "_sam_hybrid_on", False))
    sc = st._td_spec_cache
    if sc is None or sc[0] is not spec:
        sc = (
            spec,
            getattr(spec, "_next_draft_cap_g", None),
            getattr(spec, "_effective_w_cap", None),
            getattr(spec, "_effective_tau_override", None),
            getattr(spec, "_hetero_is_active_for_b", None),
            getattr(spec, "_sam_is_active_for_b", None),
            getattr(spec, "_hetero_m_cap_for_b", None),
        )
        st._td_spec_cache = sc
    _, fn_dg, fn_wc, fn_tau, fn_het, fn_sam, fn_mcap = sc
    draft_g = fn_dg(nr) if fn_dg is not None else None
    w_cap = int(fn_wc(nr, num_spec_sched)) if fn_wc is not None else int(num_spec_sched)
    tau_ov = float(fn_tau(nr)) if fn_tau is not None else 0.0
    if hetero_on or ctx_tau_on or sam_on:
        hetero_active = bool(fn_het(nr)) if fn_het is not None else (hetero_on and nr >= 2)
        sam_active = bool(fn_sam(nr)) if fn_sam is not None else sam_on
        m_cap_b = (int(fn_mcap(nr)) if fn_mcap is not None else int(getattr(spec, "_hetero_budget_m_cap", 0))) if hetero_active else 0
        if (
            len(exp_vw) != 7
            or exp_vw[0] != draft_g
            or exp_vw[1] != w_cap
            or exp_vw[2] != tau_ov
            or exp_vw[3] != hetero_active
            or exp_vw[4] != m_cap_b
            or exp_vw[5] != ctx_tau_on
            or exp_vw[6] != sam_active
        ):
            return False
    else:
        hetero_active = False
        sam_active = False
        if len(exp_vw) != 3 or exp_vw[0] != draft_g or exp_vw[1] != w_cap or exp_vw[2] != tau_ov:
            return False

    if sam_on:
        if getattr(spec, "_sam_mode", "") == "cpu":
            if nr <= int(getattr(spec, "_sam_max_b", 1)):
                return False
        elif sam_active:
            if not getattr(spec, "_sam_gpu_ready", False):
                return False
            cap_sam = getattr(spec, "_sam_boost_in_graph_captured", ())
            if fn_dg is not None:
                if draft_g is None or draft_g not in cap_sam:
                    return False
            elif not any(g >= nr for g in cap_sam):
                return False
    try:
        vw_mode = spec._vw_mode
    except AttributeError:
        vw_mode = "off"
    if vw_mode != "off":
        try:
            if not spec._vw_is_conf:
                return False
            cap_vw = spec._graph_vw_conf_captured
        except AttributeError:
            return False
        if not cap_vw:
            return False
        if fn_dg is not None:
            if draft_g is None or draft_g not in cap_vw:
                return False
            cap_entry = cap_vw[draft_g]
            if isinstance(cap_entry, tuple) and len(cap_entry) >= 2:
                want_w = w_cap if fn_wc is not None else int(cap_entry[0])
                want_tau = -2.0 if hetero_active else (tau_ov if fn_tau is not None else float(cap_entry[1]))
                if int(cap_entry[0]) != want_w or float(cap_entry[1]) != want_tau:
                    return False
        elif not any(g >= nr for g in cap_vw):
            return False
    return True


# ----------------------------------------------------------------------------------------------
# Wave-V16 Sub-25us Zero-Alloc Static Replay: O(1) 2D LUT + Fused step_meta_stamp + SyncFree-Temp
# ----------------------------------------------------------------------------------------------
def _bind_sf_fns_v16(st: _RunnerState) -> bool:
    """Bind sync_free static-ring functions directly on _RunnerState (0 module/attr lookup per step)."""
    if st._sf_get_ring_fn is not None:
        return True
    sf_m = _get_sf_ring_mod_v14(st)
    if sf_m is None:
        return False
    st._sf_clear_fn = getattr(sf_m, "clear_static_ring_state", None)
    st._sf_get_ring_fn = getattr(sf_m, "get_static_ring_state", None)
    st._sf_fence_fn = getattr(sf_m, "fence_static_ring_slot", None)
    return st._sf_get_ring_fn is not None


def _register_dt_v16_lut(st: _RunnerState, cert: dict, num_reqs: int, q_len: int) -> None:
    """Register certified 'dt' entry in direct O(1) 2D tables `dt_exec_lut[B][K]` & `dt_fn_lut[B][K]` (B=1..32, K=0..15)."""
    k_spec = q_len - 1
    if 1 <= num_reqs <= _LUT_MAX_B and 0 <= k_spec <= _LUT_MAX_K:
        st.dt_exec_lut[num_reqs][k_spec] = cert
        st.dt_fn_lut[num_reqs][k_spec] = cert.get("v16_dt_step")


def _evict_dt_v16_lut(st: _RunnerState, cert: dict) -> None:
    """Evict `cert` from `dt_exec_lut` and `dt_fn_lut` if currently installed."""
    nr = int(cert.get("num_reqs", 0) or 0)
    v14_d = cert.get("v14")
    ql = int(v14_d["q_len"]) if isinstance(v14_d, dict) and "q_len" in v14_d else 0
    k_spec = ql - 1
    if 1 <= nr <= _LUT_MAX_B and 0 <= k_spec <= _LUT_MAX_K:
        if st.dt_exec_lut[nr][k_spec] is cert:
            st.dt_exec_lut[nr][k_spec] = None
            st.dt_fn_lut[nr][k_spec] = None


def _register_td_v16_lut(st: _RunnerState, cert: dict, num_reqs: int, num_spec_sched: int, r_slot: int) -> None:
    """Register certified 'td' entry in direct O(1) 2D+slot tables `td_exec_lut` & `td_fn_lut`."""
    if 1 <= num_reqs <= _LUT_MAX_B and 0 <= num_spec_sched <= _LUT_MAX_K:
        s_idx = r_slot & 3
        st.td_exec_lut[num_reqs][num_spec_sched][s_idx] = cert
        st.td_fn_lut[num_reqs][num_spec_sched][s_idx] = cert.get("v16_td_step")


def _evict_td_v16_lut(st: _RunnerState, cert: dict) -> None:
    """Evict `cert` from `td_exec_lut` and `td_fn_lut` if currently installed."""
    nr = int(cert.get("num_reqs", 0) or 0)
    b_sig = cert.get("batch_sig")
    k_spec = int(b_sig[4]) if isinstance(b_sig, tuple) and len(b_sig) > 4 else -1
    r_slot = int(cert.get("ring_slot", -1))
    if 1 <= nr <= _LUT_MAX_B and 0 <= k_spec <= _LUT_MAX_K and r_slot >= 0:
        s_idx = r_slot & 3
        if st.td_exec_lut[nr][k_spec][s_idx] is cert:
            st.td_exec_lut[nr][k_spec][s_idx] = None
            st.td_fn_lut[nr][k_spec][s_idx] = None


def dt_lut_lookup_v16(st: _RunnerState, num_reqs: int, k_spec: int):
    """Direct O(1) integer-indexed 2D lookup into `dt_exec_lut[B][K]`."""
    if 1 <= num_reqs <= _LUT_MAX_B and 0 <= k_spec <= _LUT_MAX_K:
        return st.dt_exec_lut[num_reqs][k_spec]
    return None


def td_lut_lookup_v16(st: _RunnerState, num_reqs: int, k_spec: int, ring_slot: int):
    """Direct O(1) integer-indexed 2D+slot lookup into `td_exec_lut[B][K][slot]`."""
    if 1 <= num_reqs <= _LUT_MAX_B and 0 <= k_spec <= _LUT_MAX_K:
        return st.td_exec_lut[num_reqs][k_spec][ring_slot & 3]
    return None


def _build_pa_slot_v16(template, sl_np: np.ndarray) -> tuple[object, tuple]:
    """Pre-clone one AttentionMetadata ring slot with `seq_lens_cpu` permanently bound to `sl_np`.
    Returns `(pa_out, flat_unique_msl_objs)` so per-step stamping only writes `max_seq_len`!"""
    msl_seen: set[int] = set()
    msl_objs: list = []

    def _clone_dict(md_dict: dict) -> dict:
        cloned: dict[int, object] = {}
        out_d: dict = {}
        for k, v in md_dict.items():
            vid = id(v)
            c = cloned.get(vid)
            if c is None:
                has_sl = hasattr(v, "seq_lens_cpu")
                has_msl = hasattr(v, "max_seq_len")
                if has_sl or has_msl:
                    c = copy.copy(v)
                    if has_sl:
                        c.seq_lens_cpu = sl_np
                    if has_msl:
                        cid = id(c)
                        if cid not in msl_seen:
                            msl_seen.add(cid)
                            msl_objs.append(c)
                else:
                    c = v
                cloned[vid] = c
            out_d[k] = c
        return out_d

    if isinstance(template, dict):
        return _clone_dict(template), tuple(msl_objs)
    if isinstance(template, tuple):
        out_t = tuple(_clone_dict(item) if isinstance(item, dict) else item for item in template)
        return out_t, tuple(msl_objs)
    return template, ()


def _compile_step_meta_stamp_v16(
    cert: dict,
    runner,
    st: _RunnerState,
    scheduler_output,
    batch_req_state,
    *,
    is_b1: bool,
    req_ids,
    idx_tuple: tuple[int, ...],
    num_reqs: int,
    num_tokens: int,
    num_tokens_after_padding: int,
    num_reqs_padded: int,
    q_len: int,
    has_drafts: bool,
    ie_view,
    ie_meta,
):
    """Compile fused single-call `step_meta_stamp` updating pre-built 4-slot InputBatch +
    AttentionMetadata ring slots in-place (0 InputBatch alloc, 0 AttentionMetadata copy,
    0 intermediate Python wrapper frames, strict 4-slot non-aliasing)."""
    bufs = st.b1_bufs if is_b1 else st.bn_bufs
    ib_cls = st.b1_ib_cls
    fields = st.b1_ib_fields
    if bufs is None or ib_cls is None or fields is None:
        return None
    req_comp_all = getattr(runner.req_states, "num_computed_tokens_np", None)
    if req_comp_all is None:
        return None
    if is_b1:
        idx_mapping = bufs["idx"]
        cu_num_logits = bufs["cu"]
    else:
        idx_mapping = bufs["idx"][:num_reqs]
        cu_num_logits = bufs["cu"][: num_reqs + 1]
    ib_b = runner.input_buffers
    cu_num_logits_np, query_start_loc_np = _get_cached_qsl_np(st, num_reqs, q_len, num_reqs_padded)
    raw_base = {
        "num_reqs": num_reqs,
        "num_reqs_after_padding": num_reqs_padded,
        "idx_mapping": idx_mapping,
        "expanded_idx_mapping": bufs["expanded_idx"][:num_tokens],
        "expanded_local_pos": bufs["expanded_pos"][:num_tokens],
        "num_tokens": num_tokens,
        "num_tokens_after_padding": num_tokens_after_padding,
        "num_draft_tokens": num_reqs * (q_len - 1) if has_drafts else 0,
        "num_admitted_draft_tokens": None,
        "query_start_loc": ib_b.query_start_loc[: num_reqs_padded + 1],
        "query_start_loc_np": query_start_loc_np,
        "seq_lens": ib_b.seq_lens[:num_reqs_padded],
        "dcp_local_seq_lens": None,
        "has_prefill": False,
        "input_ids": ib_b.input_ids[:num_tokens_after_padding],
        "positions": ib_b.positions[:num_tokens_after_padding],
        "is_padding": ib_b.is_padding[:num_tokens_after_padding],
        "logits_indices": bufs["logits_indices"][:num_tokens],
        "cu_num_logits": cu_num_logits,
        "cu_num_logits_np": cu_num_logits_np,
        "prompt_lens": None,
        "fast_prefill": None,
        "max_query_len": None,
    }
    drop = set(st.b1_ib_drop) if st.b1_ib_drop else set()
    base_kw = {k: v for k, v in raw_base.items() if k in fields and k not in drop}
    dyn_all = (
        "req_ids",
        "idx_mapping_np",
        "num_scheduled_tokens",
        "num_draft_tokens_per_req",
        "seq_lens_cpu_upper_bound",
        "num_computed_tokens_np",
        "prefill_len_np",
        "num_computed_prefill_tokens_np",
        "is_prefilling_np",
        "prefill_runs_as_decode_np",
        "decode_graph_eligible",
        "max_seq_len_np",
        "has_structured_output_reqs",
    )
    dyn_set = frozenset(k for k in dyn_all if k in fields and k not in drop)

    pa_tmpl = cert.get("pa_template")
    ring: list[tuple] = []
    nc_dt = req_comp_all.dtype
    nd_init = batch_req_state.num_draft_tokens_np if has_drafts else None
    init_dyn = {
        "req_ids": req_ids,
        "idx_mapping_np": batch_req_state.idx_mapping_np,
        "num_scheduled_tokens": batch_req_state.num_scheduled_tokens,
        "num_draft_tokens_per_req": nd_init,
        "prefill_len_np": batch_req_state.prefill_len_np,
        "num_computed_prefill_tokens_np": batch_req_state.num_computed_prefill_tokens_np,
        "is_prefilling_np": batch_req_state.is_prefilling_np,
        "prefill_runs_as_decode_np": batch_req_state.prefill_runs_as_decode_np,
        "decode_graph_eligible": bool(getattr(batch_req_state, "decode_graph_eligible", True)),
        "max_seq_len_np": batch_req_state.max_seq_len_np,
        "has_structured_output_reqs": bool(getattr(scheduler_output, "has_structured_output_requests", False)),
    }
    try:
        for _ in range(NSLOT):
            nc_arr = np.empty(num_reqs, dtype=nc_dt)
            sl_arr = np.zeros(num_reqs_padded, dtype=np.int32)
            sl_t = torch.from_numpy(sl_arr)
            kw_s = dict(base_kw)
            for dk in dyn_set:
                if dk == "seq_lens_cpu_upper_bound":
                    kw_s[dk] = sl_t
                elif dk == "num_computed_tokens_np":
                    kw_s[dk] = nc_arr
                else:
                    kw_s[dk] = init_dyn[dk]
            ib_obj = ib_cls(**kw_s)
            # Probe mutability (fail-closed if ib_cls is frozen)
            if "req_ids" in dyn_set:
                ib_obj.req_ids = req_ids
            pa_out, pa_msl_objs = _build_pa_slot_v16(pa_tmpl, sl_arr)
            ring.append((ib_obj, nc_arr, sl_arr, sl_arr[:num_reqs], pa_out, pa_msl_objs))
    except Exception:  # noqa: BLE001
        return None

    ring_t = tuple(ring)
    ring_idx = [0]
    exp_idx_arr = np.array(idx_tuple, dtype=np.intp)
    is_contig_idx = (idx_tuple == tuple(range(num_reqs)))
    rc_orig_view = req_comp_all[:num_reqs] if is_contig_idx else None

    # Build reusable inline body snippet for `step_meta_stamp` (closure-bound LOAD_DEREF -> 0 default-arg copy!)
    field_assigns = (
        ("req_ids", "req_ids"),
        ("idx_mapping_np", "idx_np"),
        ("num_scheduled_tokens", "nst_np"),
        ("num_draft_tokens_per_req", "nd_pr"),
        ("prefill_len_np", "pfl_np"),
        ("num_computed_prefill_tokens_np", "ncpf_np"),
        ("is_prefilling_np", "ipf_np"),
        ("prefill_runs_as_decode_np", "prd_np"),
        ("decode_graph_eligible", "dge"),
        ("max_seq_len_np", "msl_np"),
        ("has_structured_output_reqs", "has_so"),
    )

    def _emit_stamp_body(indent: str) -> list[str]:
        b_lines = [
            f"{indent}s = _ridx[0]",
            f"{indent}_ridx[0] = (s + 1) & 3",
            f"{indent}ib_obj, nc_arr, sl_arr, sl_slc, pa_out, pa_msl = _ring[s]",
        ]
        if num_reqs == 1:
            i0 = int(idx_tuple[0])
            b_lines.extend([
                f"{indent}c0 = rc_np.item({i0})",
                f"{indent}nc_arr[0] = c0",
                f"{indent}max_sl = c0 + _ql",
                f"{indent}sl_arr[0] = max_sl",
            ])
        elif num_reqs <= 8:
            i0 = int(idx_tuple[0])
            b_lines.extend([
                f"{indent}c0 = rc_np.item({i0})",
                f"{indent}nc_arr[0] = c0",
                f"{indent}max_sl = c0 + _ql",
                f"{indent}sl_arr[0] = max_sl",
            ])
            for j in range(1, num_reqs):
                ij = int(idx_tuple[j])
                b_lines.extend([
                    f"{indent}c{j} = rc_np.item({ij})",
                    f"{indent}nc_arr[{j}] = c{j}",
                    f"{indent}s{j} = c{j} + _ql",
                    f"{indent}sl_arr[{j}] = s{j}",
                    f"{indent}if s{j} > max_sl: max_sl = s{j}",
                ])
        else:
            if is_contig_idx:
                b_lines.extend([
                    f"{indent}nc_arr[:] = _rc_view if rc_np is _rc_orig else rc_np[:_nr]",
                    f"{indent}_np_add(nc_arr, _ql, out=sl_slc)",
                    f"{indent}max_sl = sl_slc.max().item()",
                ])
            else:
                b_lines.extend([
                    f"{indent}_np_take(rc_np, _idx_arr, out=nc_arr)",
                    f"{indent}_np_add(nc_arr, _ql, out=sl_slc)",
                    f"{indent}max_sl = sl_slc.max().item()",
                ])
        for fname, argn in field_assigns:
            if fname in dyn_set:
                b_lines.append(f"{indent}ib_obj.{fname} = {argn}")
        if ring_t[0][5]:
            b_lines.extend([
                f"{indent}for md in pa_msl:",
                f"{indent}    md.max_seq_len = max_sl",
            ])
        b_lines.extend([
            f"{indent}_st._dt_v16_pa_ready = pa_out",
            f"{indent}_st._dt_v16_ie_ready = _ie_view",
            f"{indent}_st._dt_v16_ie_meta = _ie_meta",
            f"{indent}return ib_obj",
        ])
        return b_lines

    cdict: dict = {
        "_ring": ring_t,
        "_ridx": ring_idx,
        "_st": st,
        "_ie_view": ie_view,
        "_ie_meta": ie_meta,
        "_np_take": np.take,
        "_np_add": np.add,
        "_idx_arr": exp_idx_arr,
        "_rc_orig": req_comp_all,
        "_rc_view": rc_orig_view,
        "_ql": int(q_len),
        "_nr": int(num_reqs),
    }
    mk_lines = ["def _mk_stamp(_d):"]
    for k in cdict:
        mk_lines.append(f"    {k} = _d[{k!r}]")
    mk_lines.append(
        "    def step_meta_stamp(rc_np, req_ids, idx_np, nst_np, nd_pr, pfl_np, ncpf_np, ipf_np, prd_np, msl_np, dge, has_so):"
    )
    mk_lines.extend(_emit_stamp_body("        "))
    mk_lines.append("    return step_meta_stamp")
    ns: dict = {}
    exec("\n".join(mk_lines), ns)
    stamp_fn = ns["_mk_stamp"](cdict)
    cert["step_meta_stamp"] = stamp_fn
    cert["v16_ring"] = ring_t
    cert["v16_stamp_meta"] = (cdict, _emit_stamp_body)
    return stamp_fn


def _compile_dt_v16_plan(
    cert: dict,
    runner,
    st: _RunnerState,
    scheduler_output,
    batch_req_state,
    batch_desc,
    *,
    is_b1: bool,
    req_ids,
    idx_tuple: tuple[int, ...],
    num_reqs: int,
    num_tokens: int,
    num_tokens_after_padding: int,
    num_reqs_padded: int,
    q_len: int,
    has_drafts: bool,
    write_pad: bool,
    imm_idx_ptr: int,
    sconv_builders: list,
    has_so: bool,
    dge: bool,
    exec_key: tuple,
) -> None:
    """Compile Wave-V16 zero-loop validator + direct launch + fused `step_meta_stamp` and
    register into O(1) `st.dt_exec_lut[B][K]`."""
    if "v14_dt_t" not in cert or "v14" not in cert:
        _compile_dt_v14_plan(
            cert,
            runner,
            st,
            batch_desc,
            is_b1=is_b1,
            req_ids=req_ids,
            idx_tuple=idx_tuple,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            num_tokens_after_padding=num_tokens_after_padding,
            num_reqs_padded=num_reqs_padded,
            q_len=q_len,
            has_drafts=has_drafts,
            write_pad=write_pad,
            imm_idx_ptr=imm_idx_ptr,
            sconv_builders=sconv_builders,
            has_so=has_so,
            dge=dge,
            exec_key=exec_key,
        )
    vt = cert.get("v14_dt_t")
    v14_d = cert.get("v14")
    if vt is None or v14_d is None or st.region is None:
        return
    (
        exec_owner,
        _exp_b1, _exp_nr, _exp_nt, _exp_nt_pad, _exp_nr_pad,
        _exp_ql, _exp_has_d, _exp_wpad, _exp_imm,
        _exp_idx, _exp_idx0, _exp_rids, _exp_rid0,
        _exp_so, _exp_dge,
        bd_ref, bd_cg_name, bd_nt, bd_nr, bd_uni,
        bt_mode_cpu, bt_bszs, exp_mbs,
        ib_buf, rs_buf, bt_ref, prep_bufs, inner, exp_vw,
        t_ids, p_ids, t_pos, p_pos, t_sl, p_sl, t_qsl, p_qsl, t_ip, p_ip,
        t_ls, p_ls, t_dr, p_dr,
        ncomp_obj, ncomp_has_gpu, t_nc, p_nc,
        ie_buf, p_ie,
        qsl_len, sl_len, dr_stride, ls_stride,
        prep_core, extra_checks,
        bt_seq_hdrs, bt_flat_tensors, bt_flat_gpu, bt_fallback,
    ) = vt
    if exec_owner is None or ie_buf is None:
        return
    # If bt_fallback or bt_mode_cpu is non-empty, delegate cleanly to V14 validator path (still 100% safe!)
    if bt_fallback or bt_mode_cpu:
        return

    ie_view = v14_d.get("ie_view")
    ie_meta = (ie_buf, p_ie, num_tokens) if ie_view is not None else None
    stamp_fn = _compile_step_meta_stamp_v16(
        cert,
        runner,
        st,
        scheduler_output,
        batch_req_state,
        is_b1=is_b1,
        req_ids=req_ids,
        idx_tuple=idx_tuple,
        num_reqs=num_reqs,
        num_tokens=num_tokens,
        num_tokens_after_padding=num_tokens_after_padding,
        num_reqs_padded=num_reqs_padded,
        q_len=q_len,
        has_drafts=has_drafts,
        ie_view=ie_view,
        ie_meta=ie_meta,
    )
    if stamp_fn is None:
        return

    region = st.region
    backend = region.b
    exec_handle = v14_d.get("exec_handle")
    exp_gen = v14_d.get("exec_gen")
    imm_t = st._v16_imm_idx_by_ptr.get(imm_idx_ptr) if imm_idx_ptr else None
    if imm_idx_ptr and imm_t is None:
        return

    pb_idx, p_pidx, pb_cu, p_pcu, pb_eidx, p_peidx, pb_epos, p_pepos, pb_li, p_pli = prep_core
    idx_np_init = batch_req_state.idx_mapping_np
    nst_np_init = batch_req_state.num_scheduled_tokens
    nd_np_init = batch_req_state.num_draft_tokens_np
    ncpf_np_init = batch_req_state.num_computed_prefill_tokens_np
    pfl_np_init = batch_req_state.prefill_len_np
    exp_idx_b = idx_np_init.tobytes()
    exp_nst_b = nst_np_init.tobytes()
    exp_nd_b = nd_np_init.tobytes() if (has_drafts and nd_np_init is not None) else b""
    # Snapshot initial valid prefill-completed byte patterns (verified >= at compile time)
    prefill_done_init = all(int(ncpf_np_init[i]) >= int(pfl_np_init[i]) for i in range(num_reqs))
    exp_ncpf_b = ncpf_np_init.tobytes() if prefill_done_init else b""
    exp_pfl_b = pfl_np_init.tobytes() if prefill_done_init else b""

    stamp_cdict, emit_stamp_body = cert["v16_stamp_meta"]
    rt_launch = backend._rt_launch if backend._rt_launch is not None else (backend.cudart.cudaGraphLaunch if backend.cudart is not None else None)
    cuda_ok = backend._cuda_ok if backend._cuda_ok is not None else (backend.cudart.cudaError_t.cudaSuccess if backend.cudart is not None else 0)
    use_direct_rt = (rt_launch is not None and exec_handle is not None)
    spec_ref = getattr(runner, "speculator", None)
    has_so_dummy = hasattr(scheduler_output, "is_dummy")
    has_so_prof = hasattr(scheduler_output, "is_profile")
    has_runner_dummy = hasattr(runner, "_in_dummy_run")

    cdict: dict = dict(stamp_cdict)
    cdict.update({
        "_cert": cert,
        "_cfg": CFG,
        "_region": region,
        "_backend": backend,
        "_execs": backend.execs,
        "_execs_get": backend.execs.get,
        "_exec_gen_get": backend.exec_gen.get,
        "_exec_key": exec_key,
        "_exec_owner": exec_owner,
        "_exec_handle": exec_handle,
        "_exp_gen": exp_gen,
        "_rt_launch": rt_launch,
        "_cuda_ok": cuda_ok,
        "_cur_stream_fn": torch.cuda.current_stream,
        "_stats": region.stats,
        "_stat_key": _STAT_STATIC_REPLAY_DT,
        "_exp_nr": int(num_reqs),
        "_exp_nt": int(num_tokens),
        "_exp_nt_pad": int(num_tokens_after_padding),
        "_exp_nr_pad": int(num_reqs_padded),
        "_exp_has_d": bool(has_drafts),
        "_exp_so": bool(has_so),
        "_exp_dge": bool(dge),
        "_exp_rids": list(req_ids),
        "_exp_rid0": req_ids[0],
        "_exp_idx_b": exp_idx_b,
        "_exp_nst_b": exp_nst_b,
        "_exp_nd_b": exp_nd_b,
        "_exp_ncpf_b": exp_ncpf_b,
        "_exp_pfl_b": exp_pfl_b,
        "_bd_ref": bd_ref,
        "_bd_cg_name": bd_cg_name,
        "_bd_nt": bd_nt,
        "_bd_nr": bd_nr,
        "_bd_uni": bd_uni,
        "_ib_buf": ib_buf,
        "_rs_buf": rs_buf,
        "_bt_ref": bt_ref,
        "_prep_bufs": prep_bufs,
        "_inner": inner,
        "_exp_vw": bool(exp_vw),
        "_spec_ref": spec_ref,
        "_sys_mods_get": sys.modules.get,
        "_t_ids": t_ids, "_p_ids": p_ids,
        "_t_pos": t_pos, "_p_pos": p_pos,
        "_t_sl": t_sl, "_p_sl": p_sl,
        "_t_qsl": t_qsl, "_p_qsl": p_qsl,
        "_t_ip": t_ip, "_p_ip": p_ip,
        "_t_ls": t_ls, "_p_ls": p_ls,
        "_t_dr": t_dr, "_p_dr": p_dr,
        "_ncomp_obj": ncomp_obj, "_t_nc": t_nc, "_p_nc": p_nc,
        "_ie_buf": ie_buf, "_p_ie": p_ie,
        "_qsl_len": qsl_len, "_sl_len": sl_len,
        "_dr_stride": dr_stride, "_ls_stride": ls_stride,
        "_pb_idx": pb_idx, "_p_pidx": p_pidx,
        "_pb_cu": pb_cu, "_p_pcu": p_pcu,
        "_pb_eidx": pb_eidx, "_p_peidx": p_peidx,
        "_pb_epos": pb_epos, "_p_pepos": p_pepos,
        "_pb_li": pb_li, "_p_pli": p_pli,
        "_imm_t": imm_t, "_imm_ptr": imm_idx_ptr,
        "_sconv_shared": st._dt_sconv_shared,
        "_is_tree_fn": _is_tree_spec_active,
        "_type": type,
        "_tuple": tuple,
        "_len": len,
        "_int": int,
        "_bool": bool,
        "_getattr": getattr,
    })

    # Build unrolled checks for extra_checks (model + sconv tensors) and block tables
    extra_lines: list[str] = []
    model_obj = getattr(runner, "model", None)
    if model_obj is not None:
        lm_o = model_obj.get_language_model() if hasattr(model_obj, "get_language_model") else model_obj
        in_o = getattr(lm_o, "model", lm_o)
        if hasattr(in_o, "_vw_mask"):
            t_v = getattr(in_o, "_vw_mask", None)
            p_v = int(t_v.data_ptr()) if isinstance(t_v, torch.Tensor) else 0
            cdict["_ex_vwm_o"] = in_o; cdict["_ex_vwm_t"] = t_v; cdict["_ex_vwm_p"] = p_v
            if p_v:
                extra_lines.append("        if _ex_vwm_o._vw_mask is not _ex_vwm_t or _ex_vwm_t.data_ptr() != _ex_vwm_p: return None")
            else:
                extra_lines.append("        if _ex_vwm_o._vw_mask is not _ex_vwm_t: return None")
        emb_o = getattr(in_o, "embed_tokens", None)
        if emb_o is not None:
            rw_v = getattr(emb_o, "_repl_weight", None)
            if isinstance(rw_v, torch.Tensor):
                cdict["_ex_emb_o"] = emb_o; cdict["_ex_emb_t"] = rw_v; cdict["_ex_emb_p"] = int(rw_v.data_ptr())
                extra_lines.append("        if _ex_emb_o._repl_weight is not _ex_emb_t or _ex_emb_t.data_ptr() != _ex_emb_p: return None")
            elif hasattr(emb_o, "weight"):
                w_v = getattr(emb_o, "weight", None)
                p_v = int(w_v.data_ptr()) if isinstance(w_v, torch.Tensor) else 0
                cdict["_ex_emb_o"] = emb_o; cdict["_ex_emb_t"] = w_v; cdict["_ex_emb_p"] = p_v
                if p_v:
                    extra_lines.append("        if _ex_emb_o.weight is not _ex_emb_t or _ex_emb_t.data_ptr() != _ex_emb_p: return None")
                else:
                    extra_lines.append("        if _ex_emb_o.weight is not _ex_emb_t: return None")
        en_o = getattr(in_o, "embed_norm", None)
        if en_o is not None and hasattr(en_o, "weight"):
            w_v = getattr(en_o, "weight", None)
            p_v = int(w_v.data_ptr()) if isinstance(w_v, torch.Tensor) else 0
            cdict["_ex_en_o"] = en_o; cdict["_ex_en_t"] = w_v; cdict["_ex_en_p"] = p_v
            if p_v:
                extra_lines.append("        if _ex_en_o.weight is not _ex_en_t or _ex_en_t.data_ptr() != _ex_en_p: return None")
            else:
                extra_lines.append("        if _ex_en_o.weight is not _ex_en_t: return None")

    for si, sb in enumerate(sconv_builders):
        t_si = getattr(sb, "seq_idx_buffer", None)
        t_qs = getattr(sb, "query_start_buffer", None)
        p_si = int(t_si.data_ptr()) if isinstance(t_si, torch.Tensor) else 0
        p_qs = int(t_qs.data_ptr()) if isinstance(t_qs, torch.Tensor) else 0
        cdict[f"_sc_o{si}"] = sb
        cdict[f"_sc_si{si}"] = t_si; cdict[f"_sc_psi{si}"] = p_si
        cdict[f"_sc_qs{si}"] = t_qs; cdict[f"_sc_pqs{si}"] = p_qs
        c_si = f"_sc_o{si}.seq_idx_buffer is not _sc_si{si}" + (f" or _sc_si{si}.data_ptr() != _sc_psi{si}" if p_si else "")
        c_qs = f"_sc_o{si}.query_start_buffer is not _sc_qs{si}" + (f" or _sc_qs{si}.data_ptr() != _sc_pqs{si}" if p_qs else "")
        extra_lines.append(f"        if {c_si} or {c_qs}: return None")

    bt_lines: list[str] = []
    if bt_ref is not None:
        for attr in ("block_tables", "input_block_tables", "slot_mappings", "num_blocks"):
            seq = getattr(bt_ref, attr, None)
            if isinstance(seq, (list, tuple)):
                cdict[f"_bts_{attr}"] = seq
                bt_lines.append(f"        if _bt_ref.{attr} is not _bts_{attr} or _len(_bts_{attr}) != {len(seq)}: return None")
        for bi, (seq, idx_i, exp_t, exp_ptr, exp_sh, exp_st) in enumerate(bt_flat_tensors):
            cdict[f"_btq{bi}"] = seq
            cdict[f"_btt{bi}"] = exp_t
            cdict[f"_btp{bi}"] = exp_ptr
            cdict[f"_btsh{bi}"] = exp_sh
            cdict[f"_btst{bi}"] = exp_st
            bt_lines.append(
                f"        if _btq{bi}[{idx_i}] is not _btt{bi} or _btt{bi}.data_ptr() != _btp{bi}"
                f" or _btt{bi}.shape != _btsh{bi} or _btt{bi}.stride() != _btst{bi}: return None"
            )
        for bi, (seq, idx_i, exp_item, exp_gpu, exp_ptr, exp_sh, exp_st) in enumerate(bt_flat_gpu):
            cdict[f"_bgq{bi}"] = seq
            cdict[f"_bgi{bi}"] = exp_item
            cdict[f"_bgt{bi}"] = exp_gpu
            cdict[f"_bgp{bi}"] = exp_ptr
            cdict[f"_bgsh{bi}"] = exp_sh
            cdict[f"_bgst{bi}"] = exp_st
            bt_lines.append(
                f"        if _bgq{bi}[{idx_i}] is not _bgi{bi} or _bgi{bi}.gpu is not _bgt{bi}"
                f" or _bgt{bi}.data_ptr() != _bgp{bi} or _bgt{bi}.shape != _bgsh{bi}"
                f" or _bgt{bi}.stride() != _bgst{bi}: return None"
            )

    # Closure factory (LOAD_DEREF) -> ONLY 6 positional args on hot path, 0 default-arg copying!
    fn_lines = ["def _mk_v16_dt(_d):"]
    for k in cdict:
        fn_lines.append(f"    {k} = _d[{k!r}]")
    fn_lines.extend([
        "    def v16_dt_step(runner, scheduler_output, batch_req_state, batch_desc, req_ids, nt_s):",
        "        if nt_s != _exp_nt or _st.disabled or _st.b1_failed or _st.b1_dtypes is None or 'dt' not in _cfg['regions']: return None",
        "        max_r = _cfg['max_reqs']",
        "        if max_r and _exp_nr > max_r: return None",
        "        if batch_desc.cg_mode.name != _bd_cg_name or (batch_desc.num_ubatches or 1) > 1 or batch_desc.num_tokens != _bd_nt or (batch_desc.num_reqs or _exp_nr) != _exp_nr_pad: return None",
        "        if batch_desc is not _bd_ref and _bool(_getattr(batch_desc, 'uniform', False)) != _bd_uni: return None",
        "        if scheduler_output.scheduled_encoder_inputs: return None",
    ])
    if has_so_dummy and has_so_prof:
        fn_lines.append("        if scheduler_output.is_dummy or scheduler_output.is_profile: return None")
    else:
        fn_lines.append("        if _getattr(scheduler_output, 'is_dummy', False) or _getattr(scheduler_output, 'is_profile', False): return None")
    if has_runner_dummy:
        fn_lines.append("        if runner._in_dummy_run: return None")
    else:
        fn_lines.append("        if _getattr(runner, '_in_dummy_run', False): return None")
    fn_lines.extend([
        "        if runner.pcp_manager is not None or (runner.dcp_size or 1) > 1 or runner.is_encoder_decoder or not runner.is_first_pp_rank or not runner.is_last_pp_rank or runner.model_state.num_new_sampled_tokens_per_step != 1 or runner.adaptive_verification is not None or runner.fast_prefill is not None or runner.model_config.rswa_window is not None: return None",
        "        if _bool(scheduler_output.scheduled_spec_decode_tokens) != _exp_has_d: return None",
    ])
    if num_reqs == 1:
        fn_lines.append("        eff_idx_np = batch_req_state.idx_mapping_np")
        fn_lines.append("        if req_ids[0] != _exp_rid0 or eff_idx_np.tobytes() != _exp_idx_b: return None")
    else:
        fn_lines.append("        eff_idx_np = batch_req_state.idx_mapping_np")
        fn_lines.append("        if req_ids != _exp_rids or eff_idx_np.tobytes() != _exp_idx_b: return None")
    fn_lines.extend([
        "        nst_np = batch_req_state.num_scheduled_tokens",
        "        if nst_np.tobytes() != _exp_nst_b: return None",
    ])
    if has_drafts:
        fn_lines.extend([
            "        nd_np = batch_req_state.num_draft_tokens_np",
            "        if nd_np is None or nd_np.tobytes() != _exp_nd_b: return None",
        ])
    else:
        fn_lines.append("        nd_np = None")
    fn_lines.extend([
        "        ncpf_np = batch_req_state.num_computed_prefill_tokens_np",
        "        pfl_np = batch_req_state.prefill_len_np",
        "        if ncpf_np.tobytes() != _exp_ncpf_b or pfl_np.tobytes() != _exp_pfl_b:",
        f"            for _pi in range({num_reqs}):",
        "                if ncpf_np[_pi] < pfl_np[_pi]: return None",
        "        has_so_now = _bool(scheduler_output.has_structured_output_requests)",
        "        if has_so_now != _exp_so: return None",
        "        dge_now = _bool(batch_req_state.decode_graph_eligible)",
        "        if dge_now != _exp_dge: return None",
        "        if runner.input_buffers is not _ib_buf or runner.req_states is not _rs_buf or runner.block_tables is not _bt_ref: return None",
    ])
    if is_b1:
        fn_lines.append("        if _st.b1_bufs is not _prep_bufs: return None")
        if spec_ref is not None and hasattr(spec_ref, "_tree_spec_on"):
            fn_lines.extend([
                "        if runner.speculator is not _spec_ref or _spec_ref._tree_spec_on: return None",
                "        if _sys_mods_get('tree_spec_runtime') is not None or _sys_mods_get('vllm.models.inkling2.ops.tree_spec_runtime') is not None:",
                "            if _is_tree_fn(runner, 1, _st): return None",
            ])
        else:
            fn_lines.append("        if _is_tree_fn(runner, 1, _st): return None")
        for si in range(len(st._dt_sconv_shared)):
            fn_lines.append(f"        if _sconv_shared[{si}].get('dirty', False): return None")
    else:
        fn_lines.append("        if _st.bn_bufs is not _prep_bufs: return None")
    if inner is not None:
        fn_lines.append("        if _bool(_inner._vw) != _exp_vw: return None")

    # Core buffer identity + data_ptr + shape/stride checks (straight-line LOAD_DEREF!)
    dr_expr = "_rs_buf.draft_tokens" if has_drafts else "_rs_buf.last_sampled_tokens"
    nc_expr = "_ncomp_obj.gpu" if ncomp_has_gpu else "_ncomp_obj"
    fn_lines.extend([
        "        if (",
        "            _ib_buf.input_ids is not _t_ids or (_p_ids and _t_ids.data_ptr() != _p_ids)",
        "            or _ib_buf.positions is not _t_pos or (_p_pos and _t_pos.data_ptr() != _p_pos)",
        "            or _ib_buf.seq_lens is not _t_sl or (_p_sl and _t_sl.data_ptr() != _p_sl)",
        "            or _ib_buf.query_start_loc is not _t_qsl or (_p_qsl and _t_qsl.data_ptr() != _p_qsl)",
        "            or _ib_buf.is_padding is not _t_ip or (_p_ip and _t_ip.data_ptr() != _p_ip)",
        "            or _rs_buf.last_sampled_tokens is not _t_ls or _t_ls.data_ptr() != _p_ls",
        f"            or {dr_expr} is not _t_dr or _t_dr.data_ptr() != _p_dr",
        "            or _rs_buf.num_computed_tokens is not _ncomp_obj",
        f"            or {nc_expr} is not _t_nc or (_p_nc and _t_nc.data_ptr() != _p_nc)",
        "            or _st.inputs_embeds_buf is not _ie_buf or (_p_ie and _ie_buf.data_ptr() != _p_ie)",
        "            or _t_qsl.shape[0] != _qsl_len or _t_sl.shape[0] != _sl_len",
        "            or _t_dr.stride(0) != _dr_stride or _t_ls.stride(0) != _ls_stride",
        "            or _ie_buf.shape[0] < _exp_nt",
        "        ): return None",
    ])
    if has_drafts:
        fn_lines.append("        if _t_dr.dim() != 2 or _t_dr.stride(1) != 1: return None")
    fn_lines.extend([
        "        if (",
        "            _prep_bufs['idx'] is not _pb_idx or _pb_idx.data_ptr() != _p_pidx",
        "            or _prep_bufs['cu'] is not _pb_cu or _pb_cu.data_ptr() != _p_pcu",
        "            or _prep_bufs['expanded_idx'] is not _pb_eidx or _pb_eidx.data_ptr() != _p_peidx",
        "            or _prep_bufs['expanded_pos'] is not _pb_epos or _pb_epos.data_ptr() != _p_pepos",
        "            or _prep_bufs['logits_indices'] is not _pb_li or _pb_li.data_ptr() != _p_pli",
        "        ): return None",
    ])
    if imm_idx_ptr:
        fn_lines.append("        if _imm_t.data_ptr() != _imm_ptr: return None")
    fn_lines.extend(extra_lines)
    fn_lines.extend(bt_lines)

    # Direct inline cudaGraphLaunch (100% fail-closed on exec_owner, exec_gen, active, default stream)
    fn_lines.extend([
        "        if _region.active or _execs_get(_exec_key) is not _exec_owner or _exec_gen_get(_exec_key) != _exp_gen: return None",
        "        cur_s = _cur_stream_fn()",
        "        cs = cur_s.cuda_stream",
        "        if cs == _backend._last_cs:",
        "            st_h = _backend._last_st_h",
        "        else:",
        "            if _backend.is_default(cur_s): return None",
        "            _st._last_nondef_cs = cs",
        "            st_h = _backend._stream_handles.get(cs)",
        "            if st_h is None and _backend.cudart is not None:",
        "                st_h = _backend.cudart.cudaStream_t(init_value=cs)",
        "                if _len(_backend._stream_handles) < 64: _backend._stream_handles[cs] = st_h",
        "            _backend._last_cs = cs",
        "            _backend._last_st_h = st_h",
    ])
    if use_direct_rt:
        fn_lines.extend([
            "        res = _rt_launch(_exec_handle, st_h)",
            "        if (res[0] if _type(res) is _tuple else res) != _cuda_ok: return None",
        ])
    else:
        fn_lines.append("        if not _backend.replay_static_v14(_exec_key, _exec_owner, _exec_handle, _exp_gen, cur_s): return None")
    fn_lines.extend([
        "        if _backend.max_execs > 0 and _len(_execs) >= _backend.max_execs and _backend._last_mru_key is not _exec_key:",
        "            _execs.move_to_end(_exec_key)",
        "            _backend._last_mru_key = _exec_key",
        "        _backend.static_replay_hits += 1",
        "        _stats[_stat_key] += 1",
        "        _region.n += 1",
    ])
    if (not is_b1) and st._dt_sconv_shared:
        for si in range(len(st._dt_sconv_shared)):
            fn_lines.extend([
                f"        _sconv_shared[{si}]['dirty'] = True",
                f"        if _exp_nt_pad > (_sconv_shared[{si}].get('dirty_len') or 0): _sconv_shared[{si}]['dirty_len'] = _exp_nt_pad",
            ])
    # Inline fused `step_meta_stamp` (0 extra Python frame, 0 InputBatch alloc, 0 AttentionMetadata copy!)
    fn_lines.extend([
        "        _st.dt_static_armed = _cert",
        "        rc_np = _rs_buf.num_computed_tokens_np",
        "        idx_np = eff_idx_np",
        "        nd_pr = nd_np",
        "        ipf_np = batch_req_state.is_prefilling_np",
        "        prd_np = batch_req_state.prefill_runs_as_decode_np",
        "        msl_np = batch_req_state.max_seq_len_np",
        "        dge = dge_now",
        "        has_so = has_so_now",
    ])
    fn_lines.extend(emit_stamp_body("        "))
    fn_lines.append("    return v16_dt_step")
    ns: dict = {}
    try:
        exec("\n".join(fn_lines), ns)
        fn_compiled = ns["_mk_v16_dt"](cdict)
    except Exception:  # noqa: BLE001
        return
    cert["v16_dt_step"] = fn_compiled
    _register_dt_v16_lut(st, cert, num_reqs, q_len)


def _compile_td_v16_plan(cert: dict, spec, st: _RunnerState, drafts_t: torch.Tensor | None, qcgm=None) -> None:
    """Compile Wave-V16 zero-loop 'td' validator + direct launch (supporting BOTH temp==0 SyncFree
    and temp>0 SyncFree-Temp, incl. pinned static GPU draft_logits) and register in `td_exec_lut[B][K][slot]`."""
    if "v14_td_t" not in cert or "v14" not in cert:
        _compile_td_v14_plan(cert, spec, st, drafts_t)
    vt = cert.get("v14_td_t")
    v14_d = cert.get("v14")
    if vt is None or v14_d is None or spec is None or st.region is None:
        return
    (
        exec_owner,
        exp_rids, exp_rid0,
        exp_nr, exp_nt, exp_nt_nospec, exp_nspec, exp_ndraft,
        exp_vw,
        p_idx, p_qsl, p_pos, p_ids, p_sl, p_li, p_cu, p_eidx,
        p_s, p_ns, p_nr,
    ) = vt
    if exec_owner is None:
        return

    region = st.region
    backend = region.b
    exec_key = v14_d["exec_key"]
    exec_handle = v14_d.get("exec_handle")
    exp_gen = v14_d.get("exec_gen")
    r_slot = int(cert["ring_slot"])
    dl_t = getattr(spec, "draft_logits", None)
    if dl_t is not None and not (STATIC_TEMP_REPLAY_V16 and isinstance(dl_t, torch.Tensor)):
        return
    exp_dl_ptr = int(dl_t.data_ptr()) if isinstance(dl_t, torch.Tensor) else 0
    exp_dl_sh = tuple(dl_t.shape) if isinstance(dl_t, torch.Tensor) else None

    if qcgm is None:
        qcgm = getattr(spec, "query_cudagraph_manager", None)
    orig_qrfg = st._orig_qrfg
    if orig_qrfg is None and qcgm is not None:
        orig_qrfg = getattr(qcgm, "_inkling_dtg_orig_qrfg", getattr(qcgm, "run_fullgraph", None))
    draft_tok_t = v14_d.get("draft_tok_t")
    draft_tok_ptr = v14_d.get("draft_tok_ptr", 0)
    drafts_view = v14_d.get("drafts_view")
    full_drafts_view = v14_d.get("full_drafts_view")
    prop_bundle = (
        orig_qrfg,
        cert.get("draft_batch_desc"),
        r_slot,
        exp_nr,
        draft_tok_t,
        draft_tok_ptr,
        drafts_view,
        full_drafts_view,
        bool(cert.get("write_req_ok", False)),
        cert.get("idx_mapping"),
        int(cert.get("drafts_cols", exp_nspec)),
    )

    hetero_on_0 = bool(getattr(spec, "_hetero_budget_on", False))
    ctx_tau_on_0 = bool(getattr(spec, "_ctx_adaptive_tau_on", False))
    sam_on_0 = bool(getattr(spec, "_sam_hybrid_on", False))
    has_ext_vw = (len(exp_vw) == 7)
    has_dg = hasattr(spec, "_next_draft_cap_g")
    has_wc = hasattr(spec, "_effective_w_cap")
    has_tau = hasattr(spec, "_effective_tau_override")
    has_het = hasattr(spec, "_hetero_is_active_for_b")
    has_sam = hasattr(spec, "_sam_is_active_for_b")
    has_mcap = hasattr(spec, "_hetero_m_cap_for_b")

    rt_launch = backend._rt_launch if backend._rt_launch is not None else (backend.cudart.cudaGraphLaunch if backend.cudart is not None else None)
    cuda_ok = backend._cuda_ok if backend._cuda_ok is not None else (backend.cudart.cudaError_t.cudaSuccess if backend.cudart is not None else 0)
    use_direct_rt = (rt_launch is not None and exec_handle is not None)

    cdict: dict = {
        "_cert": cert,
        "_st": st,
        "_cfg": CFG,
        "_region": region,
        "_backend": backend,
        "_execs": backend.execs,
        "_execs_get": backend.execs.get,
        "_exec_gen_get": backend.exec_gen.get,
        "_exec_key": exec_key,
        "_exec_owner": exec_owner,
        "_exec_handle": exec_handle,
        "_exp_gen": exp_gen,
        "_r_slot": r_slot,
        "_rt_launch": rt_launch,
        "_cuda_ok": cuda_ok,
        "_cur_stream_fn": torch.cuda.current_stream,
        "_stats": region.stats,
        "_stat_key": _STAT_STATIC_REPLAY_TD,
        "_prop_bundle": prop_bundle,
        "_exp_rids": list(exp_rids),
        "_exp_rid0": exp_rid0,
        "_exp_nr": int(exp_nr),
        "_exp_nt": int(exp_nt),
        "_exp_nt_nospec": int(exp_nt_nospec),
        "_exp_nspec": int(exp_nspec),
        "_exp_ndraft": int(exp_ndraft),
        "_p_idx": p_idx, "_p_qsl": p_qsl, "_p_pos": p_pos, "_p_ids": p_ids,
        "_p_sl": p_sl, "_p_li": p_li, "_p_cu": p_cu, "_p_eidx": p_eidx,
        "_p_s": p_s, "_p_ns": p_ns, "_p_nr": p_nr,
        "_exp_dl_t": dl_t, "_exp_dl_ptr": exp_dl_ptr, "_exp_dl_sh": exp_dl_sh,
        "_exp_vw0": exp_vw[0] if len(exp_vw) >= 1 else None,
        "_exp_vw1": exp_vw[1] if len(exp_vw) >= 2 else 0,
        "_exp_vw2": exp_vw[2] if len(exp_vw) >= 3 else 0.0,
        "_exp_vw3": exp_vw[3] if len(exp_vw) >= 4 else False,
        "_exp_vw4": exp_vw[4] if len(exp_vw) >= 5 else 0,
        "_exp_vw5": exp_vw[5] if len(exp_vw) >= 6 else False,
        "_exp_vw6": exp_vw[6] if len(exp_vw) >= 7 else False,
        "_exp_ext_vw": has_ext_vw,
        "_type": type,
        "_tuple": tuple,
        "_len": len,
        "_int": int,
        "_float": float,
        "_bool": bool,
        "_getattr": getattr,
    }
    fn_lines = ["def _mk_v16_td(_d):"]
    for k in cdict:
        fn_lines.append(f"    {k} = _d[{k!r}]")
    fn_lines.extend([
        "    def v16_td_step(runner, ib, num_spec_sched, p_s, p_ns, p_nr):",
        "        if _st.disabled or 'td' not in _cfg['regions'] or ib.num_reqs != _exp_nr or ib.num_tokens != _exp_nt or num_spec_sched != _exp_nspec or p_s != _p_s or p_ns != _p_ns or p_nr != _p_nr: return False",
        "        max_r = _cfg['max_reqs']",
        "        if max_r and _exp_nr > max_r: return False",
        "        spec = runner.speculator",
        "        if spec is None or spec.query_cudagraph_manager is None or ib.has_prefill or runner.pcp_manager is not None or (runner.dcp_size or 1) > 1 or (spec.supports_mm_inputs and runner.model_state.supports_mm_inputs): return False",
        "        if ib.num_tokens_no_spec != _exp_nt_nospec or ib.num_draft_tokens != _exp_ndraft: return False",
    ])
    if exp_nr == 1:
        fn_lines.append("        rids = ib.req_ids")
        fn_lines.append("        if not rids or rids[0] != _exp_rid0: return False")
    else:
        fn_lines.append("        if ib.req_ids != _exp_rids: return False")
    fn_lines.extend([
        "        t = ib.idx_mapping",
        "        if (t.data_ptr() if t is not None else 0) != _p_idx: return False",
        "        t = ib.query_start_loc",
        "        if (t.data_ptr() if t is not None else 0) != _p_qsl: return False",
        "        t = ib.positions",
        "        if (t.data_ptr() if t is not None else 0) != _p_pos: return False",
        "        t = ib.input_ids",
        "        if (t.data_ptr() if t is not None else 0) != _p_ids: return False",
        "        t = ib.seq_lens",
        "        if (t.data_ptr() if t is not None else 0) != _p_sl: return False",
        "        t = ib.logits_indices",
        "        if (t.data_ptr() if t is not None else 0) != _p_li: return False",
        "        t = ib.cu_num_logits",
        "        if (t.data_ptr() if t is not None else 0) != _p_cu: return False",
        "        t = ib.expanded_idx_mapping",
        "        if (t.data_ptr() if t is not None else 0) != _p_eidx: return False",
        "        if spec._tree_spec_on or spec._cl_on or spec._vw_stats_on or spec.context_kv_only: return False",
    ])
    if exp_dl_ptr == 0:
        fn_lines.append("        if spec.draft_logits is not None: return False")
    else:
        fn_lines.append(
            "        if spec.draft_logits is not _exp_dl_t or _exp_dl_t.data_ptr() != _exp_dl_ptr or _exp_dl_t.shape != _exp_dl_sh: return False"
        )
    fn_lines.extend([
        "        hetero_on = _bool(spec._hetero_budget_on)",
        "        ctx_tau_on = _bool(spec._ctx_adaptive_tau_on)",
        "        sam_on = _bool(spec._sam_hybrid_on)",
        "        if (hetero_on or ctx_tau_on or sam_on) != _exp_ext_vw: return False",
    ])
    if has_dg:
        fn_lines.append("        draft_g = spec._next_draft_cap_g(_exp_nr)")
    else:
        fn_lines.append("        draft_g = None")
    if has_wc:
        fn_lines.append("        w_cap = _int(spec._effective_w_cap(_exp_nr, num_spec_sched))")
    else:
        fn_lines.append("        w_cap = num_spec_sched")
    if has_tau:
        fn_lines.append("        tau_ov = _float(spec._effective_tau_override(_exp_nr))")
    else:
        fn_lines.append("        tau_ov = 0.0")
    fn_lines.append("        if draft_g != _exp_vw0 or w_cap != _exp_vw1 or tau_ov != _exp_vw2: return False")
    if has_ext_vw:
        het_expr = "_bool(spec._hetero_is_active_for_b(_exp_nr))" if has_het else "(hetero_on and _exp_nr >= 2)"
        sam_expr = "_bool(spec._sam_is_active_for_b(_exp_nr))" if has_sam else "sam_on"
        mcap_expr = "_int(spec._hetero_m_cap_for_b(_exp_nr))" if has_mcap else "_int(_getattr(spec, '_hetero_budget_m_cap', 0))"
        fn_lines.extend([
            f"        hetero_active = {het_expr}",
            f"        sam_active = {sam_expr}",
            f"        m_cap_b = ({mcap_expr}) if hetero_active else 0",
            "        if hetero_active != _exp_vw3 or m_cap_b != _exp_vw4 or ctx_tau_on != _exp_vw5 or sam_active != _exp_vw6: return False",
            "        if sam_on:",
            "            if _getattr(spec, '_sam_mode', '') == 'cpu':",
            "                if _exp_nr <= _int(_getattr(spec, '_sam_max_b', 1)): return False",
            "            elif sam_active:",
            "                if not _getattr(spec, '_sam_gpu_ready', False): return False",
            "                cap_sam = _getattr(spec, '_sam_boost_in_graph_captured', ())",
            "                if draft_g is None or draft_g not in cap_sam: return False",
        ])
    else:
        fn_lines.append("        hetero_active = False")
    fn_lines.extend([
        "        if spec._vw_mode != 'off':",
        "            if not spec._vw_is_conf: return False",
        "            cap_vw = spec._graph_vw_conf_captured",
        "            if not cap_vw: return False",
    ])
    if has_dg:
        fn_lines.extend([
            "            if draft_g is None or draft_g not in cap_vw: return False",
            "            ce = cap_vw[draft_g]",
            "            if _type(ce) is _tuple and _len(ce) >= 2:",
            "                want_tau = -2.0 if hetero_active else tau_ov",
            "                if _int(ce[0]) != w_cap or _float(ce[1]) != want_tau: return False",
        ])
    else:
        fn_lines.append("            if not any(g >= _exp_nr for g in cap_vw): return False")

    # Direct inline cudaGraphLaunch
    fn_lines.extend([
        "        if _region.active or _execs_get(_exec_key) is not _exec_owner or _exec_gen_get(_exec_key) != _exp_gen: return False",
        "        cur_s = _cur_stream_fn()",
        "        cs = cur_s.cuda_stream",
        "        if cs == _backend._last_cs:",
        "            st_h = _backend._last_st_h",
        "        else:",
        "            if _backend.is_default(cur_s): return False",
        "            _st._last_nondef_cs = cs",
        "            st_h = _backend._stream_handles.get(cs)",
        "            if st_h is None and _backend.cudart is not None:",
        "                st_h = _backend.cudart.cudaStream_t(init_value=cs)",
        "                if _len(_backend._stream_handles) < 64: _backend._stream_handles[cs] = st_h",
        "            _backend._last_cs = cs",
        "            _backend._last_st_h = st_h",
    ])
    if use_direct_rt:
        fn_lines.extend([
            "        res = _rt_launch(_exec_handle, st_h)",
            "        if (res[0] if _type(res) is _tuple else res) != _cuda_ok: return False",
        ])
    else:
        fn_lines.append("        if not _backend.replay_static_v14(_exec_key, _exec_owner, _exec_handle, _exp_gen, cur_s): return False")
    fn_lines.extend([
        "        if _backend.max_execs > 0 and _len(_execs) >= _backend.max_execs and _backend._last_mru_key is not _exec_key:",
        "            _execs.move_to_end(_exec_key)",
        "            _backend._last_mru_key = _exec_key",
        "        _backend.static_replay_hits += 1",
        "        _stats[_stat_key] += 1",
        "        _region.n += 1",
        "        _st.cur_ring_slot = _r_slot",
        "        _st.cur_ring_fenced = False",
        "        _st.td_static_stream = cur_s",
        "        _st.td_static_armed = _cert",
        "        _st._td_v16_armed = _prop_bundle",
        "        return True",
        "    return v16_td_step",
    ])
    ns: dict = {}
    try:
        exec("\n".join(fn_lines), ns)
        fn_compiled = ns["_mk_v16_td"](cdict)
    except Exception:  # noqa: BLE001
        return
    cert["v16_td_step"] = fn_compiled
    cert["v16_td_prop_bundle"] = prop_bundle
    cert["v16_temp_compat"] = bool(STATIC_TEMP_REPLAY_V16)
    _register_td_v16_lut(st, cert, exp_nr, exp_nspec, r_slot)


def _build_dt_static_input_batch(
    runner,
    st: _RunnerState,
    scheduler_output,
    batch_req_state,
    *,
    is_b1: bool,
    req_ids,
    idx_mapping_np,
    prefill_len_np,
    num_computed_prefill_tokens_np,
    is_prefilling_np,
    prefill_runs_as_decode_np,
    max_seq_len_np,
    num_reqs: int,
    num_tokens: int,
    num_tokens_after_padding: int,
    num_reqs_padded: int,
    q_len: int,
    has_drafts: bool,
    cert: dict | None = None,
):
    """Construct a fresh CPU InputBatch wrapping the static GPU tensor slices for a
    certified static "dt" replay (0 GPU ops, fresh num_computed_tokens_np / seq_lens_cpu)."""
    bufs = st.b1_bufs if is_b1 else st.bn_bufs
    assert bufs is not None and st.b1_ib_cls is not None and st.b1_ib_fields is not None
    num_scheduled_tokens_np = batch_req_state.num_scheduled_tokens
    num_draft_tokens_per_req = batch_req_state.num_draft_tokens_np if has_drafts else None

    if FAST_REPLAY_V14 and cert is not None:
        ib_v14 = cert.get("ib_v14")
        req_comp_all = runner.req_states.num_computed_tokens_np
        if ib_v14 is None:
            if is_b1:
                idx_mapping = bufs["idx"]
                cu_num_logits = bufs["cu"]
            else:
                idx_mapping = bufs["idx"][:num_reqs]
                cu_num_logits = bufs["cu"][: num_reqs + 1]
            expanded_idx_mapping = bufs["expanded_idx"][:num_tokens]
            expanded_local_pos = bufs["expanded_pos"][:num_tokens]
            logits_indices = bufs["logits_indices"][:num_tokens]
            ib_b = runner.input_buffers
            cu_num_logits_np, query_start_loc_np = _get_cached_qsl_np(
                st, num_reqs, q_len, num_reqs_padded
            )
            raw_base = {
                "num_reqs": num_reqs,
                "num_reqs_after_padding": num_reqs_padded,
                "idx_mapping": idx_mapping,
                "expanded_idx_mapping": expanded_idx_mapping,
                "expanded_local_pos": expanded_local_pos,
                "num_tokens": num_tokens,
                "num_tokens_after_padding": num_tokens_after_padding,
                "num_draft_tokens": num_reqs * (q_len - 1) if has_drafts else 0,
                "num_admitted_draft_tokens": None,
                "query_start_loc": ib_b.query_start_loc[: num_reqs_padded + 1],
                "query_start_loc_np": query_start_loc_np,
                "seq_lens": ib_b.seq_lens[:num_reqs_padded],
                "dcp_local_seq_lens": None,
                "has_prefill": False,
                "input_ids": ib_b.input_ids[:num_tokens_after_padding],
                "positions": ib_b.positions[:num_tokens_after_padding],
                "is_padding": ib_b.is_padding[:num_tokens_after_padding],
                "logits_indices": logits_indices,
                "cu_num_logits": cu_num_logits,
                "cu_num_logits_np": cu_num_logits_np,
                "prompt_lens": None,
                "fast_prefill": None,
                "max_query_len": None,
            }
            drop = set(st.b1_ib_drop) if st.b1_ib_drop else set()
            fields = st.b1_ib_fields
            base_kw = {k: v for k, v in raw_base.items() if k in fields and k not in drop}
            dyn_names = (
                "req_ids",
                "idx_mapping_np",
                "num_scheduled_tokens",
                "num_draft_tokens_per_req",
                "seq_lens_cpu_upper_bound",
                "num_computed_tokens_np",
                "prefill_len_np",
                "num_computed_prefill_tokens_np",
                "is_prefilling_np",
                "prefill_runs_as_decode_np",
                "decode_graph_eligible",
                "max_seq_len_np",
                "has_structured_output_reqs",
            )
            dyn_set = frozenset(k for k in dyn_names if k in fields and k not in drop)
            # Compile positional constructor plan when all __init__ params are positional-compatible
            pos_template = None
            dyn_pos = None
            try:
                import inspect  # noqa: PLC0415

                sig = inspect.signature(st.b1_ib_cls.__init__)
                params = list(sig.parameters.values())[1:]  # skip self
                ok_pos = True
                tmpl: list = []
                dmap: list[int] = [-1] * len(dyn_names)
                dyn_index = {n: i for i, n in enumerate(dyn_names)}
                for p_idx, p in enumerate(params):
                    if p.kind not in (
                        inspect.Parameter.POSITIONAL_ONLY,
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    ):
                        ok_pos = False
                        break
                    pname = p.name
                    if pname in dyn_set:
                        dmap[dyn_index[pname]] = p_idx
                        tmpl.append(None)
                    elif pname in base_kw:
                        tmpl.append(base_kw[pname])
                    elif p.default is not inspect.Parameter.empty:
                        tmpl.append(p.default)
                    else:
                        ok_pos = False
                        break
                if ok_pos:
                    pos_template = tmpl
                    dyn_pos = tuple(dmap)
            except Exception:  # noqa: BLE001
                pos_template = None
                dyn_pos = None
            # 4-slot pre-allocated (ncomp_np, sl_np, sl_t) ring (zero per-step NumPy/Tensor alloc!)
            ring = []
            nc_dt = req_comp_all.dtype
            for _ in range(NSLOT):
                nc_arr = np.empty(num_reqs, dtype=nc_dt)
                sl_arr = np.zeros(num_reqs_padded, dtype=np.int32)
                ring.append((nc_arr, sl_arr, torch.from_numpy(sl_arr)))
            ib_v14 = [0, tuple(ring), st.b1_ib_cls, pos_template, dyn_pos, base_kw, dyn_set]
            cert["ib_v14"] = ib_v14

        r_cur = ib_v14[0]
        ib_v14[0] = (r_cur + 1) & 3
        ncomp_np, sl_np, sl_t = ib_v14[1][r_cur]
        if num_reqs == 1:
            c0 = int(req_comp_all[int(idx_mapping_np[0])])
            ncomp_np[0] = c0
            s0 = c0 + int(num_scheduled_tokens_np[0])
            sl_np[0] = s0
            max_sl = s0
        elif num_reqs <= 8:
            max_sl = 0
            for _i in range(num_reqs):
                ci = int(req_comp_all[int(idx_mapping_np[_i])])
                ncomp_np[_i] = ci
                si = ci + int(num_scheduled_tokens_np[_i])
                sl_np[_i] = si
                if si > max_sl:
                    max_sl = si
        else:
            np.take(req_comp_all, idx_mapping_np, out=ncomp_np)
            np.add(ncomp_np, num_scheduled_tokens_np, out=sl_np[:num_reqs])
            max_sl = int(np.max(sl_np[:num_reqs]))
        st.dt_last_sl = (sl_np, max_sl)
        ib_cls = ib_v14[2]
        pos_template = ib_v14[3]
        if pos_template is not None:
            args_list = pos_template.copy()
            dp = ib_v14[4]
            if dp[0] >= 0: args_list[dp[0]] = req_ids
            if dp[1] >= 0: args_list[dp[1]] = idx_mapping_np
            if dp[2] >= 0: args_list[dp[2]] = num_scheduled_tokens_np
            if dp[3] >= 0: args_list[dp[3]] = num_draft_tokens_per_req
            if dp[4] >= 0: args_list[dp[4]] = sl_t
            if dp[5] >= 0: args_list[dp[5]] = ncomp_np
            if dp[6] >= 0: args_list[dp[6]] = prefill_len_np
            if dp[7] >= 0: args_list[dp[7]] = num_computed_prefill_tokens_np
            if dp[8] >= 0: args_list[dp[8]] = is_prefilling_np
            if dp[9] >= 0: args_list[dp[9]] = prefill_runs_as_decode_np
            if dp[10] >= 0: args_list[dp[10]] = batch_req_state.decode_graph_eligible
            if dp[11] >= 0: args_list[dp[11]] = max_seq_len_np
            if dp[12] >= 0: args_list[dp[12]] = scheduler_output.has_structured_output_requests
            return ib_cls(*args_list)
        kw = ib_v14[5].copy()
        dk = ib_v14[6]
        if "req_ids" in dk: kw["req_ids"] = req_ids
        if "idx_mapping_np" in dk: kw["idx_mapping_np"] = idx_mapping_np
        if "num_scheduled_tokens" in dk: kw["num_scheduled_tokens"] = num_scheduled_tokens_np
        if "num_draft_tokens_per_req" in dk: kw["num_draft_tokens_per_req"] = num_draft_tokens_per_req
        if "seq_lens_cpu_upper_bound" in dk: kw["seq_lens_cpu_upper_bound"] = sl_t
        if "num_computed_tokens_np" in dk: kw["num_computed_tokens_np"] = ncomp_np
        if "prefill_len_np" in dk: kw["prefill_len_np"] = prefill_len_np
        if "num_computed_prefill_tokens_np" in dk: kw["num_computed_prefill_tokens_np"] = num_computed_prefill_tokens_np
        if "is_prefilling_np" in dk: kw["is_prefilling_np"] = is_prefilling_np
        if "prefill_runs_as_decode_np" in dk: kw["prefill_runs_as_decode_np"] = prefill_runs_as_decode_np
        if "decode_graph_eligible" in dk: kw["decode_graph_eligible"] = batch_req_state.decode_graph_eligible
        if "max_seq_len_np" in dk: kw["max_seq_len_np"] = max_seq_len_np
        if "has_structured_output_reqs" in dk: kw["has_structured_output_reqs"] = scheduler_output.has_structured_output_requests
        return ib_cls(**kw)

    num_computed_tokens_np = runner.req_states.num_computed_tokens_np[idx_mapping_np]
    seq_lens_cpu_upper_bound_np = np.zeros(num_reqs_padded, dtype=np.int32)
    np.add(
        num_computed_tokens_np,
        num_scheduled_tokens_np,
        out=seq_lens_cpu_upper_bound_np[:num_reqs],
    )
    seq_lens_cpu_upper_bound = torch.from_numpy(seq_lens_cpu_upper_bound_np)

    if FAST_REPLAY_V11 and cert is not None:
        base_kw = cert.get("ib_base_kwargs")
        if base_kw is None:
            if is_b1:
                idx_mapping = bufs["idx"]
                cu_num_logits = bufs["cu"]
            else:
                idx_mapping = bufs["idx"][:num_reqs]
                cu_num_logits = bufs["cu"][: num_reqs + 1]
            expanded_idx_mapping = bufs["expanded_idx"][:num_tokens]
            expanded_local_pos = bufs["expanded_pos"][:num_tokens]
            logits_indices = bufs["logits_indices"][:num_tokens]
            ib_b = runner.input_buffers
            cu_num_logits_np, query_start_loc_np = _get_cached_qsl_np(
                st, num_reqs, q_len, num_reqs_padded
            )
            raw_base = {
                "num_reqs": num_reqs,
                "num_reqs_after_padding": num_reqs_padded,
                "idx_mapping": idx_mapping,
                "expanded_idx_mapping": expanded_idx_mapping,
                "expanded_local_pos": expanded_local_pos,
                "num_tokens": num_tokens,
                "num_tokens_after_padding": num_tokens_after_padding,
                "num_draft_tokens": num_reqs * (q_len - 1) if has_drafts else 0,
                "num_admitted_draft_tokens": None,
                "query_start_loc": ib_b.query_start_loc[: num_reqs_padded + 1],
                "query_start_loc_np": query_start_loc_np,
                "seq_lens": ib_b.seq_lens[:num_reqs_padded],
                "dcp_local_seq_lens": None,
                "has_prefill": False,
                "input_ids": ib_b.input_ids[:num_tokens_after_padding],
                "positions": ib_b.positions[:num_tokens_after_padding],
                "is_padding": ib_b.is_padding[:num_tokens_after_padding],
                "logits_indices": logits_indices,
                "cu_num_logits": cu_num_logits,
                "cu_num_logits_np": cu_num_logits_np,
                "prompt_lens": None,
                "fast_prefill": None,
                "max_query_len": None,
            }
            drop = set(st.b1_ib_drop) if st.b1_ib_drop else set()
            fields = st.b1_ib_fields
            base_kw = {k: v for k, v in raw_base.items() if k in fields and k not in drop}
            dyn_keys = tuple(
                k for k in (
                    "req_ids",
                    "idx_mapping_np",
                    "num_scheduled_tokens",
                    "num_draft_tokens_per_req",
                    "seq_lens_cpu_upper_bound",
                    "num_computed_tokens_np",
                    "prefill_len_np",
                    "num_computed_prefill_tokens_np",
                    "is_prefilling_np",
                    "prefill_runs_as_decode_np",
                    "decode_graph_eligible",
                    "max_seq_len_np",
                    "has_structured_output_reqs",
                ) if k in fields and k not in drop
            )
            cert["ib_base_kwargs"] = base_kw
            cert["ib_dyn_keys"] = frozenset(dyn_keys)
        kw = base_kw.copy()
        dk = cert["ib_dyn_keys"]
        if "req_ids" in dk: kw["req_ids"] = req_ids
        if "idx_mapping_np" in dk: kw["idx_mapping_np"] = idx_mapping_np
        if "num_scheduled_tokens" in dk: kw["num_scheduled_tokens"] = num_scheduled_tokens_np
        if "num_draft_tokens_per_req" in dk: kw["num_draft_tokens_per_req"] = num_draft_tokens_per_req
        if "seq_lens_cpu_upper_bound" in dk: kw["seq_lens_cpu_upper_bound"] = seq_lens_cpu_upper_bound
        if "num_computed_tokens_np" in dk: kw["num_computed_tokens_np"] = num_computed_tokens_np
        if "prefill_len_np" in dk: kw["prefill_len_np"] = prefill_len_np
        if "num_computed_prefill_tokens_np" in dk: kw["num_computed_prefill_tokens_np"] = num_computed_prefill_tokens_np
        if "is_prefilling_np" in dk: kw["is_prefilling_np"] = is_prefilling_np
        if "prefill_runs_as_decode_np" in dk: kw["prefill_runs_as_decode_np"] = prefill_runs_as_decode_np
        if "decode_graph_eligible" in dk: kw["decode_graph_eligible"] = batch_req_state.decode_graph_eligible
        if "max_seq_len_np" in dk: kw["max_seq_len_np"] = max_seq_len_np
        if "has_structured_output_reqs" in dk: kw["has_structured_output_reqs"] = scheduler_output.has_structured_output_requests
        return st.b1_ib_cls(**kw)

    if is_b1:
        idx_mapping = bufs["idx"]
        cu_num_logits = bufs["cu"]
        expanded_idx_mapping = bufs["expanded_idx"][:num_tokens]
        expanded_local_pos = bufs["expanded_pos"][:num_tokens]
        logits_indices = bufs["logits_indices"][:num_tokens]
    else:
        idx_mapping = bufs["idx"][:num_reqs]
        cu_num_logits = bufs["cu"][: num_reqs + 1]
        expanded_idx_mapping = bufs["expanded_idx"][:num_tokens]
        expanded_local_pos = bufs["expanded_pos"][:num_tokens]
        logits_indices = bufs["logits_indices"][:num_tokens]

    qsl_full = runner.input_buffers.query_start_loc
    sl_full = runner.input_buffers.seq_lens
    is_padding = runner.input_buffers.is_padding[:num_tokens_after_padding]
    cu_num_logits_np, query_start_loc_np = _get_cached_qsl_np(
        st, num_reqs, q_len, num_reqs_padded
    )
    query_start_loc = qsl_full[: num_reqs_padded + 1]
    seq_lens = sl_full[:num_reqs_padded]
    total_num_draft_tokens = num_reqs * (q_len - 1) if has_drafts else 0

    all_kwargs = {
        "req_ids": req_ids,
        "num_reqs": num_reqs,
        "num_reqs_after_padding": num_reqs_padded,
        "idx_mapping": idx_mapping,
        "idx_mapping_np": idx_mapping_np,
        "expanded_idx_mapping": expanded_idx_mapping,
        "expanded_local_pos": expanded_local_pos,
        "num_scheduled_tokens": num_scheduled_tokens_np,
        "num_tokens": num_tokens,
        "num_tokens_after_padding": num_tokens_after_padding,
        "num_draft_tokens": total_num_draft_tokens,
        "num_draft_tokens_per_req": num_draft_tokens_per_req,
        "num_admitted_draft_tokens": None,
        "query_start_loc": query_start_loc,
        "query_start_loc_np": query_start_loc_np,
        "seq_lens": seq_lens,
        "seq_lens_cpu_upper_bound": seq_lens_cpu_upper_bound,
        "dcp_local_seq_lens": None,
        "num_computed_tokens_np": num_computed_tokens_np,
        "prefill_len_np": prefill_len_np,
        "num_computed_prefill_tokens_np": num_computed_prefill_tokens_np,
        "is_prefilling_np": is_prefilling_np,
        "has_prefill": False,
        "prefill_runs_as_decode_np": prefill_runs_as_decode_np,
        "decode_graph_eligible": batch_req_state.decode_graph_eligible,
        "max_seq_len_np": max_seq_len_np,
        "input_ids": runner.input_buffers.input_ids[:num_tokens_after_padding],
        "positions": runner.input_buffers.positions[:num_tokens_after_padding],
        "is_padding": is_padding,
        "logits_indices": logits_indices,
        "cu_num_logits": cu_num_logits,
        "cu_num_logits_np": cu_num_logits_np,
        "has_structured_output_reqs": scheduler_output.has_structured_output_requests,
        "prompt_lens": None,
        "fast_prefill": None,
        "max_query_len": None,
    }
    if st.b1_ib_drop:
        for k in st.b1_ib_drop:
            all_kwargs.pop(k, None)
    filtered = {k: v for k, v in all_kwargs.items() if k in st.b1_ib_fields}
    return st.b1_ib_cls(**filtered)


def _refresh_attn_meta_dict(
    md_dict: dict, sl_np: np.ndarray | None, max_sl: int | None
) -> dict:
    """Shallow-refresh seq_lens_cpu / max_seq_len on unique AttentionMetadata instances."""
    cloned: dict[int, object] = {}
    out: dict = {}
    for k, v in md_dict.items():
        vid = id(v)
        c = cloned.get(vid)
        if c is None:
            has_sl = hasattr(v, "seq_lens_cpu")
            has_msl = hasattr(v, "max_seq_len")
            if (has_sl and sl_np is not None) or (has_msl and max_sl is not None):
                c = copy.copy(v)
                if has_sl and sl_np is not None:
                    c.seq_lens_cpu = sl_np
                if has_msl and max_sl is not None:
                    c.max_seq_len = max_sl
            else:
                c = v
            cloned[vid] = c
        out[k] = c
    return out


def _compile_pa_dict_plan(md_dict: dict) -> tuple:
    id_to_idx: dict[int, int] = {}
    uniqs: list = []
    key_idx: list = []
    for k, v in md_dict.items():
        vid = id(v)
        idx = id_to_idx.get(vid)
        if idx is None:
            idx = len(uniqs)
            id_to_idx[vid] = idx
            uniqs.append((v, hasattr(v, "seq_lens_cpu"), hasattr(v, "max_seq_len")))
        key_idx.append((k, idx))
    return (tuple(uniqs), tuple(key_idx))


def _compile_pa_refresh_plan(template):
    if isinstance(template, dict):
        return ("dict", _compile_pa_dict_plan(template))
    if isinstance(template, tuple):
        return (
            "tuple",
            tuple(
                ("dict", _compile_pa_dict_plan(item)) if isinstance(item, dict) else ("raw", item)
                for item in template
            ),
        )
    return ("raw", template)


def _run_pa_dict_plan(dplan: tuple, sl_np: np.ndarray | None, max_sl: int | None) -> dict:
    uniqs, key_idx = dplan
    vals = []
    for v, has_sl, has_msl in uniqs:
        if (has_sl and sl_np is not None) or (has_msl and max_sl is not None):
            c = copy.copy(v)
            if has_sl and sl_np is not None:
                c.seq_lens_cpu = sl_np
            if has_msl and max_sl is not None:
                c.max_seq_len = max_sl
            vals.append(c)
        else:
            vals.append(v)
    return {k: vals[idx] for k, idx in key_idx}


def _compile_pa_dict_ring_v14(md_dict: dict) -> list:
    """Pre-clone 4-slot ring of unique mutable AttentionMetadata objects & 4 pre-built layer dicts.
    Eliminates per-step copy.copy() and 48-key dict construction on steady-state static 'dt' replay."""
    dplan = _compile_pa_dict_plan(md_dict)
    uniqs, key_idx = dplan
    slots = []
    for _ in range(NSLOT):
        vals = []
        muts = []
        for v, has_sl, has_msl in uniqs:
            if has_sl or has_msl:
                c = copy.copy(v)
                vals.append(c)
                muts.append((c, has_sl, has_msl))
            else:
                vals.append(v)
        out_d = {k: vals[idx] for k, idx in key_idx}
        slots.append((tuple(muts), out_d))
    return [0, tuple(slots)]


def _compile_pa_refresh_plan_v14(template):
    if isinstance(template, dict):
        return ("dict_ring", _compile_pa_dict_ring_v14(template))
    if isinstance(template, tuple):
        return (
            "tuple_ring",
            tuple(
                ("dict_ring", _compile_pa_dict_ring_v14(item)) if isinstance(item, dict) else ("raw", item)
                for item in template
            ),
        )
    return ("raw", template)


def _run_pa_dict_ring_v14(ring: list, sl_np: np.ndarray | None, max_sl: int | None) -> dict:
    s = ring[0]
    ring[0] = (s + 1) & 3
    muts, out_d = ring[1][s]
    for c, has_sl, has_msl in muts:
        if has_sl and sl_np is not None:
            c.seq_lens_cpu = sl_np
        if has_msl and max_sl is not None:
            c.max_seq_len = max_sl
    return out_d


def _refresh_cached_attn_metadata(template, input_batch, plan=None, precomputed_sl=None):
    if precomputed_sl is not None:
        sl_np, max_sl = precomputed_sl
    else:
        sl_t = getattr(input_batch, "seq_lens_cpu_upper_bound", None) if input_batch is not None else None
        sl_np = sl_t.numpy() if isinstance(sl_t, torch.Tensor) else None
        nr = int(getattr(input_batch, "num_reqs", 0) or 0) if input_batch is not None else 0
        if sl_np is not None and nr > 0:
            max_sl = int(sl_np[0]) if nr == 1 else int(np.max(sl_np[:nr]))
        else:
            max_sl = None
    if plan is not None:
        kind, body = plan
        if kind == "dict_ring":
            return _run_pa_dict_ring_v14(body, sl_np, max_sl)
        if kind == "tuple_ring":
            return tuple(
                _run_pa_dict_ring_v14(sub_b, sl_np, max_sl) if sub_k == "dict_ring" else sub_b
                for sub_k, sub_b in body
            )
        if kind == "dict":
            return _run_pa_dict_plan(body, sl_np, max_sl)
        if kind == "tuple":
            return tuple(
                _run_pa_dict_plan(sub_b, sl_np, max_sl) if sub_k == "dict" else sub_b
                for sub_k, sub_b in body
            )
        return body
    if isinstance(template, dict):
        return _refresh_attn_meta_dict(template, sl_np, max_sl)
    if isinstance(template, tuple):
        return tuple(
            _refresh_attn_meta_dict(item, sl_np, max_sl) if isinstance(item, dict) else item
            for item in template
        )
    return template


def _open_region(
    st: _RunnerState,
    region_name: str,
    shape_key: tuple,
    *,
    force_slot: int | None = None,
) -> bool:
    st.seen[(region_name, shape_key)] += 1
    if st.seen[(region_name, shape_key)] <= CFG["warm"]:
        st.skips[f"{region_name}:warm"] += 1
        return False
    try:
        stream = _ensure_region(st)
    except Exception as e:  # noqa: BLE001
        st.disabled = True
        _logger().warning("INKLING_DTGRAPH disabled before capture: %s", e)
        return False
    try:
        st.region.begin(stream, region_name=region_name, shape_key=shape_key, force_slot=force_slot)
    except Exception as e:  # noqa: BLE001
        st.region.active = False
        _TLS.region = None
        st.disabled = True
        _logger().warning("INKLING_DTGRAPH disabled (capture_begin failed: %s)", e)
        return False
    if DTGRAPH_FASTPATH:
        if st.mode_obj is None:
            st.mode_obj = _Mode.make(st.region)
        st.mode = st.mode_obj
    else:
        st.mode = _Mode.make(st.region)
    st.mode.__enter__()
    return True


def close_region(
    runner, expected_name: str | None = None, *, keep_mode: bool = False
) -> None:
    st = getattr(runner, "_inkling_dtg", None)
    if st is None:
        return
    r = st.region
    has_active = r is not None and r.active
    if has_active:
        if expected_name is not None and r.region_name != expected_name:
            return
        if not keep_mode:
            mode, st.mode = st.mode, None
            if mode is not None:
                mode.__exit__(None, None, None)
        r.end()
        every = CFG["log"]
        if every and r.n % every == 0 and st.rank0:
            s = dict(r.stats)
            _logger().info(
                "INKLING_DTGRAPH stats after %d regions: stats=%s skips=%s update_hits=%d update_fail=%d evictions=%d",
                r.n,
                s,
                dict(st.skips),
                getattr(r.b, "update_hits", 0),
                getattr(r.b, "update_fail", 0),
                getattr(r.b, "evictions", 0),
            )
    elif not keep_mode and st.mode is not None:
        # Finish deferred TorchDispatchMode.__exit__() strictly outside nested
        # @torch.inference_mode() guards (e.g. DFlashSpeculator.propose) so PyTorch's
        # ForceDispatchKeyGuard never restores DispatchKey::Python onto an empty mode stack!
        mode, st.mode = st.mode, None
        mode.__exit__(None, None, None)


def abort_region(runner) -> None:
    st = getattr(runner, "_inkling_dtg", None)
    if st is None:
        return
    has_active = st.region is not None and st.region.active
    mode, st.mode = st.mode, None
    if mode is not None:
        try:
            mode.__exit__(None, None, None)
        except Exception:  # noqa: BLE001
            pass
    if not has_active:
        return
    st.region.abort()
    st.disabled = True
    _logger().error("INKLING_DTGRAPH: exception inside glue region; capture dropped, lever disabled")


_INSTALLED = False


def install() -> bool:
    """Install lazy hooks on GPUModelRunner & graph managers when INKLING_DTGRAPH is enabled.
    Called once from Inkling2DSparkSpeculator.__init__ (or Inkling2Model). Safe & idempotent."""
    global _INSTALLED
    if not ON or _INSTALLED:
        return False
    _INSTALLED = True
    try:
        from vllm.v1.worker.gpu import model_runner as mr  # noqa: PLC0415

        cls = mr.GPUModelRunner
        if getattr(cls, "_inkling_dtg_installed", False):
            return True

        # 1. Wrap execute_model: abort on exception, ensure DT closes before non-FULL return,
        # and hook model_state.prepare_inputs_embeds for direct zero-copy embed output.
        orig_exec = cls.execute_model
        set_direct_embed = _get_model_set_direct_embed() if FUSED_EMBED_DIRECT else None
        draft_store_tls = _get_spec_draft_store_tls() if FUSED_DRAFT_STORE else None

        def _pa_has_sconv_md(pa_ret) -> bool:
            dicts = (
                [pa_ret]
                if isinstance(pa_ret, dict)
                else (
                    [x for x in pa_ret if isinstance(x, dict)]
                    if isinstance(pa_ret, tuple)
                    else []
                )
            )
            for d in dicts:
                for v in d.values():
                    if hasattr(v, "seq_idx") and hasattr(v, "query_start"):
                        return True
            return False

        def wrapped_execute_model(self, *args, **kwargs):
            if FAST_REPLAY_V16 and DTGRAPH_FASTPATH and not PF_SCRATCH_RELEASE_IDLE:
                try:
                    st = self._inkling_dtg
                except AttributeError:
                    st = _get_state(self)
                if st.exec_hooks_done:
                    st._dt_v16_hit = False
                    if st.dt_static_armed is not None:
                        st.dt_static_armed = None
                    if st.dt_cap_ctx is not None:
                        st.dt_cap_ctx = None
                    try:
                        out = orig_exec(self, *args, **kwargs)
                    except BaseException:
                        st.dt_static_armed = None
                        st.dt_cap_ctx = None
                        st._dt_v16_pa_ready = None
                        st._dt_v16_ie_ready = None
                        if st.region is not None:
                            st.region.dt_track_unhooked = False
                            st.region.dt_hook_depth = 0
                        abort_region(self)
                        raise
                    if st._dt_v16_hit:
                        st._dt_v16_hit = False
                        return out
                    if st.region is not None and (st.region.active or st.mode is not None):
                        close_region(self, expected_name="dt")
                    st.dt_static_armed = None
                    st.dt_cap_ctx = None
                    st._dt_v16_pa_ready = None
                    st._dt_v16_ie_ready = None
                    if st.region is not None:
                        st.region.dt_track_unhooked = False
                        st.region.dt_hook_depth = 0
                    return out
            st = _get_state(self)
            st.dt_static_armed = None
            st.dt_cap_ctx = None
            if FAST_REPLAY_V16:
                st._dt_v16_pa_ready = None
                st._dt_v16_ie_ready = None
                st._dt_v16_hit = False
            if not (DTGRAPH_FASTPATH and st.exec_hooks_done):
                # Ensure target cudagraph_manager.run_fullgraph closes DT right before replay.
                cgm = getattr(self, "cudagraph_manager", None)
                if cgm is not None and not getattr(cgm, "_inkling_dtg_cgm", False):
                    orig_rfg = cgm.run_fullgraph

                    def _rfg(batch_desc, _orig=orig_rfg, _runner=self, _st=st):
                        if _st.dt_static_armed is not None:
                            _st.dt_static_armed = None
                            if FAST_REPLAY_V16:
                                _st._dt_v16_hit = True
                            return _orig(batch_desc)
                        cap = _st.dt_cap_ctx
                        _st.dt_cap_ctx = None
                        r_pre = _st.region
                        dt_active_pre = (
                            cap is not None
                            and r_pre is not None
                            and r_pre.active
                            and r_pre.region_name == "dt"
                        )
                        arena_off_pre = int(r_pre.arena.off) if dt_active_pre else -1
                        unhooked_pre = bool(r_pre.dt_unhooked_dev_op) if dt_active_pre else True
                        close_region(_runner, expected_name="dt")
                        if dt_active_pre and r_pre is not None:
                            haz_now = sum(
                                v for k, v in r_pre.stats.items() if k.startswith("haz:dt:")
                            )
                            bd_cap = cap["batch_desc"]
                            bd_ok = (batch_desc is bd_cap) or (
                                getattr(batch_desc.cg_mode, "name", str(batch_desc.cg_mode))
                                == getattr(bd_cap.cg_mode, "name", str(bd_cap.cg_mode))
                                and int(batch_desc.num_tokens) == int(bd_cap.num_tokens)
                                and int(batch_desc.num_reqs or 0) == int(bd_cap.num_reqs or 0)
                            )
                            pure_ok = (
                                r_pre.seg == 1
                                and haz_now == cap["haz0"]
                                and r_pre.stats["staged"] == cap["staged0"]
                                and r_pre.stats["pinned"] == cap["pinned0"]
                                and arena_off_pre == 0
                                and not unhooked_pre
                                and cap["prep_fused_ok"]
                                and cap["pa_called"]
                                and cap["pa_ret_ok"]
                                and cap["ms_pa_ok"]
                                and cap["pie_called"]
                                and cap["pie_static_ok"]
                                and cap["sconv_ok"]
                                and bd_ok
                                and not _is_tree_spec_active(_runner, cap["num_reqs"])
                                and isinstance(_st.inputs_embeds_buf, torch.Tensor)
                            )
                            if pure_ok:
                                exec_key = ("dt", cap["exec_shape_key"], 0, cap["dt_slot"])
                                exec_gen = getattr(r_pre.b, "exec_gen", {}).get(exec_key)
                                fp_fin = _dt_static_fingerprint(
                                    _runner,
                                    _st,
                                    cap["scheduler_output"],
                                    cap["batch_req_state"],
                                    bd_cap,
                                    is_b1=cap["is_b1"],
                                    req_ids=cap["req_ids"],
                                    idx_mapping_np=cap["idx_mapping_np"],
                                    num_reqs=cap["num_reqs"],
                                    num_tokens=cap["num_tokens"],
                                    num_tokens_after_padding=cap["num_tokens_after_padding"],
                                    num_reqs_padded=cap["num_reqs_padded"],
                                    q_len=cap["q_len"],
                                    has_drafts=cap["has_drafts"],
                                    write_pad=cap["write_pad"],
                                    imm_idx_ptr=cap["imm_idx_ptr"],
                                    sconv_builders=cap["sconv_builders"],
                                )
                                if (
                                    fp_fin is not None
                                    and exec_key in getattr(r_pre.b, "execs", {})
                                    and exec_gen is not None
                                ):
                                    fin_batch_sig, fin_ptr_sig = fp_fin
                                    if fin_batch_sig == cap["batch_sig"]:
                                        _dt_cert_entry = {
                                            "exec_shape_key": cap["exec_shape_key"],
                                            "dt_slot": cap["dt_slot"],
                                            "exec_gen": exec_gen,
                                            "batch_sig": fin_batch_sig,
                                            "ptr_sig": fin_ptr_sig,
                                            "num_reqs": cap["num_reqs"],
                                            "num_tokens": cap["num_tokens"],
                                            "num_tokens_after_padding": cap[
                                                "num_tokens_after_padding"
                                            ],
                                            "pa_template": cap["pa_template"],
                                            "pa_refresh_plan": (
                                                _compile_pa_refresh_plan(cap["pa_template"])
                                                if (FAST_REPLAY_V11 or FAST_REPLAY_V14 or FAST_REPLAY_V16)
                                                else None
                                            ),
                                            "pa_refresh_plan_v14": (
                                                _compile_pa_refresh_plan_v14(cap["pa_template"])
                                                if (FAST_REPLAY_V14 or FAST_REPLAY_V16)
                                                else None
                                            ),
                                            "imm_idx_ptr": cap["imm_idx_ptr"],
                                        }
                                        if FAST_REPLAY_V14 or FAST_REPLAY_V16:
                                            _compile_dt_v14_plan(
                                                _dt_cert_entry,
                                                _runner,
                                                _st,
                                                bd_cap,
                                                is_b1=cap["is_b1"],
                                                req_ids=cap["req_ids"],
                                                idx_tuple=cap["exec_shape_key"][6],
                                                num_reqs=cap["num_reqs"],
                                                num_tokens=cap["num_tokens"],
                                                num_tokens_after_padding=cap["num_tokens_after_padding"],
                                                num_reqs_padded=cap["num_reqs_padded"],
                                                q_len=cap["q_len"],
                                                has_drafts=cap["has_drafts"],
                                                write_pad=cap["write_pad"],
                                                imm_idx_ptr=cap["imm_idx_ptr"],
                                                sconv_builders=cap["sconv_builders"],
                                                has_so=bool(getattr(cap["scheduler_output"], "has_structured_output_requests", False)),
                                                dge=bool(getattr(cap["batch_req_state"], "decode_graph_eligible", True)),
                                                exec_key=exec_key,
                                            )
                                            if FAST_REPLAY_V16:
                                                _compile_dt_v16_plan(
                                                    _dt_cert_entry,
                                                    _runner,
                                                    _st,
                                                    cap["scheduler_output"],
                                                    cap["batch_req_state"],
                                                    bd_cap,
                                                    is_b1=cap["is_b1"],
                                                    req_ids=cap["req_ids"],
                                                    idx_tuple=cap["exec_shape_key"][6],
                                                    num_reqs=cap["num_reqs"],
                                                    num_tokens=cap["num_tokens"],
                                                    num_tokens_after_padding=cap["num_tokens_after_padding"],
                                                    num_reqs_padded=cap["num_reqs_padded"],
                                                    q_len=cap["q_len"],
                                                    has_drafts=cap["has_drafts"],
                                                    write_pad=cap["write_pad"],
                                                    imm_idx_ptr=cap["imm_idx_ptr"],
                                                    sconv_builders=cap["sconv_builders"],
                                                    has_so=bool(getattr(cap["scheduler_output"], "has_structured_output_requests", False)),
                                                    dge=bool(getattr(cap["batch_req_state"], "decode_graph_eligible", True)),
                                                    exec_key=exec_key,
                                                )
                                            _st.dt_last_cert = _dt_cert_entry
                                        _st.dt_static_cert[cap["cert_key"]] = _dt_cert_entry
                                        _st.dt_static_cert.move_to_end(cap["cert_key"])
                                        while len(_st.dt_static_cert) > STATIC_DT_MAX_CERTS:
                                            _, _ev_c = _st.dt_static_cert.popitem(last=False)
                                            if _st.dt_last_cert is _ev_c:
                                                _st.dt_last_cert = None
                                            if FAST_REPLAY_V16:
                                                _evict_dt_v16_lut(_st, _ev_c)
                                        r_pre.stats["dt_static_certified"] += 1
                        return _orig(batch_desc)

                    cgm.run_fullgraph = _rfg
                    cgm._inkling_dtg_cgm = True
                # Hook model_state.prepare_attn (if present) for Wave-V10 static dt replay.
                ms = getattr(self, "model_state", None)
                if (
                    STATIC_DT_REPLAY
                    and ms is not None
                    and hasattr(ms, "prepare_attn")
                    and not getattr(ms, "_inkling_dtg_ms_pa", False)
                ):
                    orig_ms_pa = ms.prepare_attn

                    def _wrapped_ms_pa(*pa_args, _orig=orig_ms_pa, _st=st, **pa_kwargs):
                        if _st.dt_static_armed is not None:
                            return None
                        r = _st.region
                        tracked = r is not None and r.dt_track_unhooked
                        if tracked:
                            r.dt_hook_depth += 1
                        try:
                            res = _orig(*pa_args, **pa_kwargs)
                        finally:
                            if tracked:
                                r.dt_hook_depth -= 1
                        if _st.dt_cap_ctx is not None:
                            _st.dt_cap_ctx["ms_pa_ok"] = (res is None)
                        return res

                    ms.prepare_attn = _wrapped_ms_pa
                    ms._inkling_dtg_ms_pa = True
                # Lever B: Hook model_state.prepare_inputs_embeds once to write directly into inputs_embeds
                # (and short-circuit on Wave-V10 static dt replay hits).
                if (
                    (set_direct_embed is not None or STATIC_DT_REPLAY)
                    and ms is not None
                    and hasattr(ms, "prepare_inputs_embeds")
                    and not getattr(ms, "_inkling_dtg_pie", False)
                ):
                    orig_pie = ms.prepare_inputs_embeds

                    def _wrapped_pie(
                        scheduled_encoder_inputs,
                        input_batch,
                        req_states,
                        _orig=orig_pie,
                        _ms=ms,
                        _st=st,
                        _set_out=set_direct_embed,
                    ):
                        if FAST_REPLAY_V16 and not scheduled_encoder_inputs:
                            ie_r = _st._dt_v16_ie_ready
                            if ie_r is not None:
                                _st._dt_v16_ie_ready = None
                                ie_m = _st._dt_v16_ie_meta
                                if ie_m is not None and _st.inputs_embeds_buf is ie_m[0] and input_batch.num_tokens == ie_m[2]:
                                    return ie_r
                        armed = _st.dt_static_armed
                        if armed is not None:
                            if FAST_REPLAY_V14 and not scheduled_encoder_inputs:
                                v14_a = armed.get("v14")
                                if (
                                    v14_a is not None
                                    and v14_a["ie_view"] is not None
                                    and _st.inputs_embeds_buf is v14_a["ie_buf"]
                                    and _st.inputs_embeds_buf.data_ptr() == v14_a["ie_ptr"]
                                    and input_batch.num_tokens == v14_a["num_tokens"]
                                ):
                                    return v14_a["ie_view"]
                            nt_a = int(getattr(input_batch, "num_tokens", 0) or 0)
                            buf_a = _st.inputs_embeds_buf
                            if (
                                not scheduled_encoder_inputs
                                and nt_a == armed["num_tokens"]
                                and isinstance(buf_a, torch.Tensor)
                                and buf_a.dim() == 2
                                and buf_a.is_contiguous()
                                and buf_a.shape[0] >= nt_a
                            ):
                                return buf_a[:nt_a]
                            _st.dt_static_armed = None
                        direct_set = False
                        if (
                            not scheduled_encoder_inputs
                            and not getattr(input_batch, "has_prefill", True)
                            and int(getattr(input_batch, "num_tokens", 0) or 0) > 0
                        ):
                            buf = _st.inputs_embeds_buf
                            if buf is None:
                                cand = getattr(_ms, "inputs_embeds", None)
                                if cand is None:
                                    er = getattr(_ms, "encoder_runner", None)
                                    cand = getattr(er, "inputs_embeds", None) if er is not None else None
                                if (
                                    isinstance(cand, torch.Tensor)
                                    and cand.dim() == 2
                                    and cand.is_contiguous()
                                ):
                                    buf = _st.inputs_embeds_buf = cand
                            nt = int(input_batch.num_tokens)
                            if (
                                _set_out is not None
                                and buf is not None
                                and buf.dim() == 2
                                and buf.is_contiguous()
                                and buf.shape[0] >= nt
                            ):
                                _set_out(buf[:nt])
                                direct_set = True
                        r = _st.region
                        tracked = r is not None and r.dt_track_unhooked
                        if tracked:
                            r.dt_hook_depth += 1
                        try:
                            res = _orig(scheduled_encoder_inputs, input_batch, req_states)
                        finally:
                            if tracked:
                                r.dt_hook_depth -= 1
                            if direct_set:
                                _set_out(None)
                        if (
                            not scheduled_encoder_inputs
                            and not getattr(input_batch, "has_prefill", False)
                            and isinstance(res, torch.Tensor)
                            and res.dim() == 2
                            and res.is_contiguous()
                            and (
                                _st.inputs_embeds_buf is None
                                or res.data_ptr() != _st.inputs_embeds_buf.data_ptr()
                                or res.shape[0] > _st.inputs_embeds_buf.shape[0]
                            )
                        ):
                            _st.inputs_embeds_buf = res
                        if _st.dt_cap_ctx is not None:
                            _st.dt_cap_ctx["pie_called"] = True
                            nt_chk = int(getattr(input_batch, "num_tokens", -1) or -1)
                            _st.dt_cap_ctx["pie_static_ok"] = (
                                (not scheduled_encoder_inputs)
                                and isinstance(res, torch.Tensor)
                                and res.dim() == 2
                                and res.is_contiguous()
                                and _st.inputs_embeds_buf is not None
                                and res.data_ptr() == _st.inputs_embeds_buf.data_ptr()
                                and int(res.shape[0]) == nt_chk
                            )
                        return res

                    ms.prepare_inputs_embeds = _wrapped_pie
                    ms._inkling_dtg_pie = True
                if DTGRAPH_FASTPATH:
                    cgm_ok = cgm is not None and getattr(cgm, "_inkling_dtg_cgm", False)
                    pie_ok = (set_direct_embed is None and not STATIC_DT_REPLAY) or (
                        ms is not None
                        and (
                            not hasattr(ms, "prepare_inputs_embeds")
                            or getattr(ms, "_inkling_dtg_pie", False)
                        )
                    )
                    ms_pa_ok = (not STATIC_DT_REPLAY) or (
                        ms is None
                        or (not hasattr(ms, "prepare_attn"))
                        or getattr(ms, "_inkling_dtg_ms_pa", False)
                    )
                    if cgm_ok and pie_ok and ms_pa_ok:
                        st.exec_hooks_done = True
            try:
                out = orig_exec(self, *args, **kwargs)
                if not (FAST_REPLAY_V14 or FAST_REPLAY_V16) or (st.region is not None and (st.region.active or st.mode is not None)):
                    close_region(self, expected_name="dt")
                if PF_SCRATCH_RELEASE_IDLE:
                    _sched = args[0] if args else kwargs.get("scheduler_output")
                    _ntok = int(getattr(_sched, "total_num_scheduled_tokens", 0) or 0)
                    if _ntok >= 512:
                        st._pf_scratch_dirty = True
                        st._pf_idle_decode_steps = 0
                    elif getattr(st, "_pf_scratch_dirty", False) and _ntok > 0:
                        _cnt = int(getattr(st, "_pf_idle_decode_steps", 0)) + 1
                        st._pf_idle_decode_steps = _cnt
                        if _cnt >= 4 and not torch.cuda.is_current_stream_capturing():
                            st._pf_scratch_dirty = False
                            st._pf_idle_decode_steps = 0
                            try:
                                from . import host_pin as _hp_uma

                                _hp_uma.release_all_prefill_scratch(
                                    include_shadow_k=PF_RELEASE_SHADOW_K_IDLE,
                                    empty_cuda=True,
                                    logger=_logger(),
                                )
                            except Exception:  # noqa: BLE001
                                pass
                return out
            except BaseException:
                abort_region(self)
                raise
            finally:
                st.dt_static_armed = None
                st.dt_cap_ctx = None
                if st.region is not None:
                    st.region.dt_track_unhooked = False
                    st.region.dt_hook_depth = 0

        # 2. Wrap prepare_inputs (and prepare_attn when STATIC_DT_REPLAY=1): open Phase 4 ("dt")
        # region right before prepare_inputs, run Lever A fused prep, and replay certified static
        # "dt" graphs with zero TorchDispatchMode / zero capture / zero ExecUpdate when eligible.
        orig_prep = cls.prepare_inputs
        orig_prep_attn = getattr(cls, "prepare_attn", None)

        if STATIC_DT_REPLAY and orig_prep_attn is not None:
            def wrapped_prepare_attn(self, *pa_args, **pa_kwargs):
                if FAST_REPLAY_V16:
                    try:
                        st = self._inkling_dtg
                    except AttributeError:
                        st = _get_state(self)
                    pa_r = st._dt_v16_pa_ready
                    if pa_r is not None:
                        st._dt_v16_pa_ready = None
                        return pa_r
                else:
                    st = _get_state(self)
                armed = st.dt_static_armed
                if armed is not None:
                    ib_arg = pa_args[0] if pa_args else pa_kwargs.get("input_batch")
                    if FAST_REPLAY_V14:
                        sl_pre = st.dt_last_sl
                        st.dt_last_sl = None
                        p14 = armed.get("pa_refresh_plan_v14")
                        if p14 is None:
                            p14 = _compile_pa_refresh_plan_v14(armed["pa_template"])
                            armed["pa_refresh_plan_v14"] = p14
                        return _refresh_cached_attn_metadata(
                            armed["pa_template"], ib_arg, plan=p14, precomputed_sl=sl_pre
                        )
                    return _refresh_cached_attn_metadata(
                        armed["pa_template"], ib_arg, plan=armed.get("pa_refresh_plan")
                    )
                r = st.region
                tracked = r is not None and r.dt_track_unhooked
                if tracked:
                    r.dt_hook_depth += 1
                try:
                    res = orig_prep_attn(self, *pa_args, **pa_kwargs)
                finally:
                    if tracked:
                        r.dt_hook_depth -= 1
                cap = st.dt_cap_ctx
                if cap is not None:
                    cap["pa_called"] = True
                    cap["pa_ret_ok"] = isinstance(res, (dict, tuple))
                    cap["pa_template"] = res
                    if _pa_has_sconv_md(res) and not cap["sconv_builders"]:
                        cap["sconv_ok"] = False
                return res

        def wrapped_prepare_inputs(
            self, scheduler_output, batch_req_state, batch_desc, num_active_loras
        ):
            # Wave-V16 O(1) 2D LUT `dt_fn_lut[B][K]` + zero-loop validator + inlined step_meta_stamp!
            if (
                FAST_REPLAY_V16
                and STATIC_DT_REPLAY
                and FUSED_DT_PREP
                and not ROWAR_DESC_W_ORDER
                and orig_prep_attn is not None
                and num_active_loras == 0
            ):
                try:
                    st = self._inkling_dtg
                except AttributeError:
                    st = _get_state(self)
                if not batch_req_state.has_prefill:
                    eff_req_ids = batch_req_state.req_ids
                    nr_f = len(eff_req_ids)
                    if 0 < nr_f <= _LUT_MAX_B:
                        nt_f = batch_req_state.num_tokens
                        ql_f = nt_f if nr_f == 1 else (nt_f // nr_f)
                        if 0 < ql_f <= (_LUT_MAX_K + 1) and (nr_f == 1 or ql_f * nr_f == nt_f):
                            fn_v16 = st.dt_fn_lut[nr_f][ql_f - 1]
                            if fn_v16 is not None:
                                try:
                                    ib_v16 = fn_v16(
                                        self, scheduler_output, batch_req_state, batch_desc, eff_req_ids, nt_f
                                    )
                                    if ib_v16 is not None:
                                        return ib_v16
                                except Exception:  # noqa: BLE001
                                    pass
            else:
                st = _get_state(self)
            st.dt_static_armed = None
            st.dt_cap_ctx = None
            if FAST_REPLAY_V16:
                st._dt_v16_pa_ready = None
                st._dt_v16_ie_ready = None
            # Wave-V14 zero-alloc MRU fast-entry (avoids ~25 getattr + dt_ineligible kwdict on steady hits!)
            if (
                FAST_REPLAY_V14
                and STATIC_DT_REPLAY
                and FUSED_DT_PREP
                and not ROWAR_DESC_W_ORDER
                and orig_prep_attn is not None
                and st.dt_last_cert is not None
                and st.region is not None
                and not st.disabled
                and not st.b1_failed
                and st.b1_dtypes is not None
                and "dt" in CFG["regions"]
                and num_active_loras == 0
            ):
                try:
                    if (
                        not batch_req_state.has_prefill
                        and batch_desc.cg_mode.name == "FULL"
                        and (batch_desc.num_ubatches or 1) <= 1
                        and not scheduler_output.scheduled_encoder_inputs
                        and not getattr(self, "_in_dummy_run", False)
                        and not getattr(scheduler_output, "is_dummy", False)
                        and not getattr(scheduler_output, "is_profile", False)
                        and self.pcp_manager is None
                        and (self.dcp_size or 1) <= 1
                        and not self.is_encoder_decoder
                        and self.is_first_pp_rank
                        and self.is_last_pp_rank
                        and self.model_state.num_new_sampled_tokens_per_step == 1
                        and self.adaptive_verification is None
                        and self.fast_prefill is None
                        and self.model_config.rswa_window is None
                    ):
                        eff_req_ids = batch_req_state.req_ids
                        nr_f = len(eff_req_ids)
                        max_r_cfg = CFG["max_reqs"]
                        if (
                            0 < nr_f <= 8
                            and (nr_f == 1 or (FUSED_DT_PREP_MAX_B >= nr_f and FUSED_DT_PREP_INLINE_B8))
                            and (max_r_cfg == 0 or nr_f <= max_r_cfg)
                        ):
                            nt_f = batch_req_state.num_tokens
                            bd_nt_f = batch_desc.num_tokens
                            nt_pad_f = nt_f if nt_f >= bd_nt_f else bd_nt_f
                            nr_pad_f = batch_desc.num_reqs or nr_f
                            is_b1_f = (nr_f == 1)
                            has_d_f = bool(scheduler_output.scheduled_spec_decode_tokens)
                            if 0 < nt_f and nt_pad_f <= 256 and (not is_b1_f or has_d_f):
                                elig_ok = False
                                ql_f = nt_f if is_b1_f else (nt_f // nr_f)
                                nst_f = batch_req_state.num_scheduled_tokens
                                ncpf_f = batch_req_state.num_computed_prefill_tokens_np
                                pfl_f = batch_req_state.prefill_len_np
                                nd_f = batch_req_state.num_draft_tokens_np
                                if is_b1_f:
                                    if (
                                        nst_f[0] == nt_f
                                        and nd_f is not None
                                        and nd_f[0] + 1 == nt_f
                                        and ncpf_f[0] >= pfl_f[0]
                                        and not _is_tree_spec_active(self, 1, st)
                                    ):
                                        elig_ok = True
                                elif nt_f % nr_f == 0:
                                    elig_ok = True
                                    if has_d_f:
                                        if nd_f is None:
                                            elig_ok = False
                                        else:
                                            for _fi in range(nr_f):
                                                if nst_f[_fi] != ql_f or nd_f[_fi] + 1 != ql_f or ncpf_f[_fi] < pfl_f[_fi]:
                                                    elig_ok = False
                                                    break
                                    else:
                                        if ql_f != 1:
                                            elig_ok = False
                                        else:
                                            for _fi in range(nr_f):
                                                if nst_f[_fi] != 1 or ncpf_f[_fi] < pfl_f[_fi]:
                                                    elig_ok = False
                                                    break
                                if elig_ok:
                                    sconv_f = st.dt_sconv_builders if st.dt_sconv_builders is not None else _discover_sconv_builders(self, st)
                                    if not (is_b1_f and _sconv_any_dirty(sconv_f, st)):
                                        w_pad_f = st.write_pad if (DTGRAPH_FASTPATH and st.write_pad is not None) else _get_write_pad(st)
                                        last_c = st.dt_last_cert
                                        eff_idx_f = batch_req_state.idx_mapping_np
                                        if last_c is not None and _dt_v14_validate(
                                            last_c,
                                            self,
                                            st,
                                            scheduler_output,
                                            batch_req_state,
                                            batch_desc,
                                            is_b1=is_b1_f,
                                            req_ids=eff_req_ids,
                                            eff_idx_np=eff_idx_f,
                                            num_reqs=nr_f,
                                            nt_s=nt_f,
                                            nt_pad_s=nt_pad_f,
                                            nr_pad_s=nr_pad_f,
                                            ql_s=ql_f,
                                            has_d_s=has_d_f,
                                            w_pad_s=w_pad_f,
                                            imm_ptr=0,
                                        ):
                                            v14_c = last_c["v14"]
                                            stream = _ensure_region(st)
                                            if st.region.replay_static_region_v14(
                                                stream,
                                                _STAT_STATIC_REPLAY_DT,
                                                v14_c["exec_key"],
                                                v14_c["exec_owner"],
                                                v14_c["exec_handle"],
                                                v14_c["exec_gen"],
                                            ):
                                                if FAST_REPLAY_V16 and "v16_dt_step" not in last_c:
                                                    _compile_dt_v16_plan(
                                                        last_c,
                                                        self,
                                                        st,
                                                        scheduler_output,
                                                        batch_req_state,
                                                        batch_desc,
                                                        is_b1=is_b1_f,
                                                        req_ids=eff_req_ids,
                                                        idx_tuple=last_c["exec_shape_key"][6],
                                                        num_reqs=nr_f,
                                                        num_tokens=nt_f,
                                                        num_tokens_after_padding=nt_pad_f,
                                                        num_reqs_padded=nr_pad_f,
                                                        q_len=ql_f,
                                                        has_drafts=has_d_f,
                                                        write_pad=w_pad_f,
                                                        imm_idx_ptr=0,
                                                        sconv_builders=sconv_f,
                                                        has_so=bool(getattr(scheduler_output, "has_structured_output_requests", False)),
                                                        dge=bool(getattr(batch_req_state, "decode_graph_eligible", True)),
                                                        exec_key=v14_c["exec_key"],
                                                    )
                                                if not is_b1_f and sconv_f:
                                                    _sconv_mark_dirty(sconv_f, nt_pad_f, st)
                                                st.dt_static_armed = last_c
                                                return _build_dt_static_input_batch(
                                                    self,
                                                    st,
                                                    scheduler_output,
                                                    batch_req_state,
                                                    is_b1=is_b1_f,
                                                    req_ids=eff_req_ids,
                                                    idx_mapping_np=eff_idx_f,
                                                    prefill_len_np=pfl_f,
                                                    num_computed_prefill_tokens_np=ncpf_f,
                                                    is_prefilling_np=batch_req_state.is_prefilling_np,
                                                    prefill_runs_as_decode_np=batch_req_state.prefill_runs_as_decode_np,
                                                    max_seq_len_np=batch_req_state.max_seq_len_np,
                                                    num_reqs=nr_f,
                                                    num_tokens=nt_f,
                                                    num_tokens_after_padding=nt_pad_f,
                                                    num_reqs_padded=nr_pad_f,
                                                    q_len=ql_f,
                                                    has_drafts=has_d_f,
                                                    cert=last_c,
                                                )
                except Exception:  # noqa: BLE001
                    pass
            b1_elig = (
                FUSED_DT_PREP
                and not st.b1_failed
                and _b1_prep_eligible(self, scheduler_output, batch_req_state, batch_desc)
            )
            bn_elig = (
                (not b1_elig)
                and FUSED_DT_PREP
                and FUSED_DT_PREP_MAX_B >= 2
                and not st.b1_failed
                and _bn_prep_eligible(self, scheduler_output, batch_req_state, batch_desc)
            )
            need_b1_warm = False
            if b1_elig:
                nt_pad = max(int(batch_req_state.num_tokens), int(batch_desc.num_tokens))
                if st.b1_dtypes is None:
                    need_b1_warm = True
                else:
                    _ensure_b1_bufs(st, self.device, nt_pad)
                    w_pad = _get_write_pad(st)
                    bt = max(16, 1 << max(0, nt_pad - 1).bit_length())
                    if (w_pad, 256, bt) not in st.b1_compiled:
                        need_b1_warm = True
            elif bn_elig:
                nr = len(batch_req_state.req_ids)
                nt = int(batch_req_state.num_tokens)
                nt_pad = max(nt, int(batch_desc.num_tokens))
                ql = nt // nr
                if st.b1_dtypes is None:
                    need_b1_warm = True
                else:
                    _ensure_bn_bufs(st, self.device, nr, nt_pad)
                    w_pad = _get_write_pad(st)
                    has_d = bool(getattr(scheduler_output, "scheduled_spec_decode_tokens", None))
                    bq = max(16, 1 << max(0, ql - 1).bit_length())
                    cset = (
                        st.bn_inline_compiled
                        if (FUSED_DT_PREP_INLINE_B8 and nr <= 8)
                        else st.bn_compiled
                    )
                    _defer_bn_warm = bool(
                        (FAST_REPLAY_V11 or FAST_REPLAY_V14) and STATIC_DT_REPLAY and st.dt_static_cert and cset
                    )
                    if not _defer_bn_warm:
                        ib_b = self.input_buffers
                        rs_b = self.req_states
                        dr_b = getattr(rs_b, "draft_tokens", None) if has_d else rs_b.last_sampled_tokens
                        psig_b = _dt_prep_ptr_sig(
                            ib_b.is_padding, st.bn_bufs["idx"], st.bn_bufs["cu"], ib_b.query_start_loc,
                            st.bn_bufs["expanded_idx"], st.bn_bufs["expanded_pos"], ib_b.positions,
                            ib_b.seq_lens, ib_b.input_ids, st.bn_bufs["logits_indices"],
                            rs_b.num_computed_tokens.gpu, rs_b.last_sampled_tokens,
                            dr_b if isinstance(dr_b, torch.Tensor) else rs_b.last_sampled_tokens,
                        )
                        if (w_pad, has_d, 256, bq, 256, psig_b) not in cset:
                            need_b1_warm = True

            precomputed_rowar = None
            imm_idx_cpu = None
            if not st.disabled and "dt" in CFG["regions"] and not need_b1_warm:
                num_reqs = len(batch_req_state.req_ids)
                why = dt_ineligible(
                    cg_full=getattr(batch_desc.cg_mode, "name", str(batch_desc.cg_mode)) == "FULL",
                    num_ubatches=int(getattr(batch_desc, "num_ubatches", 1) or 1),
                    has_prefill=bool(batch_req_state.has_prefill),
                    pcp=getattr(self, "pcp_manager", None) is not None,
                    dcp_size=int(getattr(self, "dcp_size", 1) or 1),
                    lora=bool(getattr(self, "lora_config", None)) and num_active_loras > 0,
                    encoder_inputs=bool(getattr(scheduler_output, "scheduled_encoder_inputs", None)),
                    encoder_decoder=bool(getattr(self, "is_encoder_decoder", False)),
                    pp_single=bool(
                        getattr(self, "is_first_pp_rank", True)
                        and getattr(self, "is_last_pp_rank", True)
                    ),
                    num_reqs=num_reqs,
                    max_reqs=CFG["max_reqs"],
                )
                if why is not None:
                    st.skips[f"dt:{why}"] += 1
                else:
                    is_dummy = bool(
                        getattr(self, "_in_dummy_run", False)
                        or getattr(scheduler_output, "is_dummy", False)
                        or getattr(scheduler_output, "is_profile", False)
                    )
                    can_static_dt = (
                        STATIC_DT_REPLAY
                        and (b1_elig or bn_elig)
                        and orig_prep_attn is not None
                        and not is_dummy
                        and not _is_tree_spec_active(self, num_reqs, st)
                    )
                    if can_static_dt:
                        nt_s = int(batch_req_state.num_tokens)
                        nt_pad_s = max(nt_s, int(batch_desc.num_tokens))
                        nr_pad_s = int(batch_desc.num_reqs or num_reqs)
                        ql_s = nt_s // num_reqs
                        has_d_s = (
                            True
                            if b1_elig
                            else bool(getattr(scheduler_output, "scheduled_spec_decode_tokens", None))
                        )
                        w_pad_s = _get_write_pad(st)
                        if b1_elig:
                            eff_req_ids = batch_req_state.req_ids
                            eff_idx_np = batch_req_state.idx_mapping_np
                            eff_pfl_np = batch_req_state.prefill_len_np
                            eff_ncpf_np = batch_req_state.num_computed_prefill_tokens_np
                            eff_ipf_np = batch_req_state.is_prefilling_np
                            eff_prd_np = batch_req_state.prefill_runs_as_decode_np
                            eff_msl_np = batch_req_state.max_seq_len_np
                        else:
                            if FAST_REPLAY_V14 and not ROWAR_DESC_W_ORDER:
                                eff_req_ids = batch_req_state.req_ids
                                eff_idx_np = batch_req_state.idx_mapping_np
                                eff_pfl_np = batch_req_state.prefill_len_np
                                eff_ncpf_np = batch_req_state.num_computed_prefill_tokens_np
                                eff_ipf_np = batch_req_state.is_prefilling_np
                                eff_prd_np = batch_req_state.prefill_runs_as_decode_np
                                eff_msl_np = batch_req_state.max_seq_len_np
                                precomputed_rowar = (
                                    eff_req_ids,
                                    eff_idx_np,
                                    eff_pfl_np,
                                    eff_ncpf_np,
                                    eff_ipf_np,
                                    eff_prd_np,
                                    eff_msl_np,
                                )
                            else:
                                precomputed_rowar = _compute_rowar_swap(
                                    self, st, scheduler_output, batch_req_state, ql_s, has_d_s
                                )
                                (
                                    eff_req_ids,
                                    eff_idx_np,
                                    eff_pfl_np,
                                    eff_ncpf_np,
                                    eff_ipf_np,
                                    eff_prd_np,
                                    eff_msl_np,
                                ) = precomputed_rowar
                            if not (FUSED_DT_PREP_INLINE_B8 and num_reqs <= 8):
                                imm_idx_cpu = _get_dt_immutable_idx_cpu(
                                    st, st.bn_bufs["idx"].dtype, eff_idx_np
                                )
                        sconv_builders = _discover_sconv_builders(self, st)
                        sconv_b1_dirty = (num_reqs == 1) and _sconv_any_dirty(sconv_builders, st)
                        imm_ptr = int(imm_idx_cpu.data_ptr()) if imm_idx_cpu is not None else 0
                        # Wave-V14 zero-alloc MRU static "dt" replay fast path (0 tuple/dict/fingerprint allocs!)
                        if FAST_REPLAY_V14 and not sconv_b1_dirty and st.region is not None:
                            last_c = st.dt_last_cert
                            if last_c is not None and _dt_v14_validate(
                                last_c,
                                self,
                                st,
                                scheduler_output,
                                batch_req_state,
                                batch_desc,
                                is_b1=b1_elig,
                                req_ids=eff_req_ids,
                                eff_idx_np=eff_idx_np,
                                num_reqs=num_reqs,
                                nt_s=nt_s,
                                nt_pad_s=nt_pad_s,
                                nr_pad_s=nr_pad_s,
                                ql_s=ql_s,
                                has_d_s=has_d_s,
                                w_pad_s=w_pad_s,
                                imm_ptr=imm_ptr,
                            ):
                                try:
                                    v14_c = last_c["v14"]
                                    stream = _ensure_region(st)
                                    if st.region.replay_static_region_v14(
                                        stream,
                                        _STAT_STATIC_REPLAY_DT,
                                        v14_c["exec_key"],
                                        v14_c["exec_owner"],
                                        v14_c["exec_handle"],
                                        v14_c["exec_gen"],
                                    ):
                                        if num_reqs >= 2 and sconv_builders:
                                            _sconv_mark_dirty(sconv_builders, nt_pad_s, st)
                                        st.dt_static_armed = last_c
                                        return _build_dt_static_input_batch(
                                            self,
                                            st,
                                            scheduler_output,
                                            batch_req_state,
                                            is_b1=b1_elig,
                                            req_ids=eff_req_ids,
                                            idx_mapping_np=eff_idx_np,
                                            prefill_len_np=eff_pfl_np,
                                            num_computed_prefill_tokens_np=eff_ncpf_np,
                                            is_prefilling_np=eff_ipf_np,
                                            prefill_runs_as_decode_np=eff_prd_np,
                                            max_seq_len_np=eff_msl_np,
                                            num_reqs=num_reqs,
                                            num_tokens=nt_s,
                                            num_tokens_after_padding=nt_pad_s,
                                            num_reqs_padded=nr_pad_s,
                                            q_len=ql_s,
                                            has_drafts=has_d_s,
                                            cert=last_c,
                                        )
                                except Exception:  # noqa: BLE001
                                    pass
                                st.dt_last_cert = None
                        idx_tuple = (
                            ((int(eff_idx_np[0]),) if num_reqs == 1 else tuple(eff_idx_np.tolist()))
                            if FAST_REPLAY_V14
                            else tuple(int(x) for x in eff_idx_np)
                        )
                        exec_shape_key = (
                            "dt_s",
                            num_reqs,
                            nt_s,
                            nt_pad_s,
                            ql_s,
                            bool(has_d_s),
                            idx_tuple,
                        )
                        cert_key = exec_shape_key
                        # Wave-V14 non-MRU cert lookup (switched batch size, still skips _dt_static_fingerprint!)
                        if FAST_REPLAY_V14 and not sconv_b1_dirty and st.region is not None:
                            cert_lu = st.dt_static_cert.get(cert_key)
                            if (
                                cert_lu is not None
                                and cert_lu is not st.dt_last_cert
                                and _dt_v14_validate(
                                    cert_lu,
                                    self,
                                    st,
                                    scheduler_output,
                                    batch_req_state,
                                    batch_desc,
                                    is_b1=b1_elig,
                                    req_ids=eff_req_ids,
                                    eff_idx_np=eff_idx_np,
                                    num_reqs=num_reqs,
                                    nt_s=nt_s,
                                    nt_pad_s=nt_pad_s,
                                    nr_pad_s=nr_pad_s,
                                    ql_s=ql_s,
                                    has_d_s=has_d_s,
                                    w_pad_s=w_pad_s,
                                    imm_ptr=imm_ptr,
                                )
                            ):
                                try:
                                    v14_lu = cert_lu["v14"]
                                    stream = _ensure_region(st)
                                    if st.region.replay_static_region_v14(
                                        stream,
                                        _STAT_STATIC_REPLAY_DT,
                                        v14_lu["exec_key"],
                                        v14_lu["exec_owner"],
                                        v14_lu["exec_handle"],
                                        v14_lu["exec_gen"],
                                    ):
                                        if num_reqs >= 2 and sconv_builders:
                                            _sconv_mark_dirty(sconv_builders, nt_pad_s, st)
                                        st.dt_static_cert.move_to_end(cert_key)
                                        st.dt_last_cert = cert_lu
                                        st.dt_static_armed = cert_lu
                                        return _build_dt_static_input_batch(
                                            self,
                                            st,
                                            scheduler_output,
                                            batch_req_state,
                                            is_b1=b1_elig,
                                            req_ids=eff_req_ids,
                                            idx_mapping_np=eff_idx_np,
                                            prefill_len_np=eff_pfl_np,
                                            num_computed_prefill_tokens_np=eff_ncpf_np,
                                            is_prefilling_np=eff_ipf_np,
                                            prefill_runs_as_decode_np=eff_prd_np,
                                            max_seq_len_np=eff_msl_np,
                                            num_reqs=num_reqs,
                                            num_tokens=nt_s,
                                            num_tokens_after_padding=nt_pad_s,
                                            num_reqs_padded=nr_pad_s,
                                            q_len=ql_s,
                                            has_drafts=has_d_s,
                                            cert=cert_lu,
                                        )
                                except Exception:  # noqa: BLE001
                                    pass
                        fp = _dt_static_fingerprint(
                            self,
                            st,
                            scheduler_output,
                            batch_req_state,
                            batch_desc,
                            is_b1=b1_elig,
                            req_ids=eff_req_ids,
                            idx_mapping_np=eff_idx_np,
                            num_reqs=num_reqs,
                            num_tokens=nt_s,
                            num_tokens_after_padding=nt_pad_s,
                            num_reqs_padded=nr_pad_s,
                            q_len=ql_s,
                            has_drafts=has_d_s,
                            write_pad=w_pad_s,
                            imm_idx_ptr=imm_ptr,
                            sconv_builders=sconv_builders,
                            idx_tuple=idx_tuple,
                        )
                        if fp is not None and not sconv_b1_dirty:
                            batch_sig, ptr_sig = fp
                            cert = st.dt_static_cert.get(cert_key)
                            if (
                                cert is not None
                                and st.region is not None
                                and cert["batch_sig"] == batch_sig
                                and cert["ptr_sig"] == ptr_sig
                                and isinstance(st.inputs_embeds_buf, torch.Tensor)
                                and st.inputs_embeds_buf.shape[0] >= nt_s
                            ):
                                try:
                                    stream = _ensure_region(st)
                                    if st.region.replay_static_region(
                                        stream,
                                        "dt",
                                        cert["exec_shape_key"],
                                        cert["dt_slot"],
                                        wait_slot=False,
                                        expected_exec_gen=cert["exec_gen"],
                                    ):
                                        if (FAST_REPLAY_V14 or FAST_REPLAY_V16) and "v14" not in cert:
                                            _compile_dt_v14_plan(
                                                cert,
                                                self,
                                                st,
                                                batch_desc,
                                                is_b1=b1_elig,
                                                req_ids=eff_req_ids,
                                                idx_tuple=idx_tuple,
                                                num_reqs=num_reqs,
                                                num_tokens=nt_s,
                                                num_tokens_after_padding=nt_pad_s,
                                                num_reqs_padded=nr_pad_s,
                                                q_len=ql_s,
                                                has_drafts=has_d_s,
                                                write_pad=w_pad_s,
                                                imm_idx_ptr=imm_ptr,
                                                sconv_builders=sconv_builders,
                                                has_so=bool(getattr(scheduler_output, "has_structured_output_requests", False)),
                                                dge=bool(getattr(batch_req_state, "decode_graph_eligible", True)),
                                                exec_key=("dt", cert["exec_shape_key"], 0, cert["dt_slot"]),
                                            )
                                            cert["pa_refresh_plan_v14"] = _compile_pa_refresh_plan_v14(cert["pa_template"])
                                        if FAST_REPLAY_V16 and "v16_dt_step" not in cert:
                                            _compile_dt_v16_plan(
                                                cert,
                                                self,
                                                st,
                                                scheduler_output,
                                                batch_req_state,
                                                batch_desc,
                                                is_b1=b1_elig,
                                                req_ids=eff_req_ids,
                                                idx_tuple=idx_tuple,
                                                num_reqs=num_reqs,
                                                num_tokens=nt_s,
                                                num_tokens_after_padding=nt_pad_s,
                                                num_reqs_padded=nr_pad_s,
                                                q_len=ql_s,
                                                has_drafts=has_d_s,
                                                write_pad=w_pad_s,
                                                imm_idx_ptr=imm_ptr,
                                                sconv_builders=sconv_builders,
                                                has_so=bool(getattr(scheduler_output, "has_structured_output_requests", False)),
                                                dge=bool(getattr(batch_req_state, "decode_graph_eligible", True)),
                                                exec_key=("dt", cert["exec_shape_key"], 0, cert["dt_slot"]),
                                            )
                                        elif FAST_REPLAY_V16:
                                            _register_dt_v16_lut(st, cert, num_reqs, ql_s)
                                        if FAST_REPLAY_V14 or FAST_REPLAY_V16:
                                            st.dt_last_cert = cert
                                        if num_reqs >= 2 and sconv_builders:
                                            _sconv_mark_dirty(sconv_builders, nt_pad_s, st)
                                        st.dt_static_cert.move_to_end(cert_key)
                                        st.dt_static_armed = cert
                                        return _build_dt_static_input_batch(
                                            self,
                                            st,
                                            scheduler_output,
                                            batch_req_state,
                                            is_b1=b1_elig,
                                            req_ids=eff_req_ids,
                                            idx_mapping_np=eff_idx_np,
                                            prefill_len_np=eff_pfl_np,
                                            num_computed_prefill_tokens_np=eff_ncpf_np,
                                            is_prefilling_np=eff_ipf_np,
                                            prefill_runs_as_decode_np=eff_prd_np,
                                            max_seq_len_np=eff_msl_np,
                                            num_reqs=num_reqs,
                                            num_tokens=nt_s,
                                            num_tokens_after_padding=nt_pad_s,
                                            num_reqs_padded=nr_pad_s,
                                            q_len=ql_s,
                                            has_drafts=has_d_s,
                                            cert=cert,
                                        )
                                except Exception:  # noqa: BLE001
                                    pass
                                _popped = st.dt_static_cert.pop(cert_key, None)
                                if _popped is not None:
                                    if st.dt_last_cert is _popped:
                                        st.dt_last_cert = None
                                    if FAST_REPLAY_V16:
                                        _evict_dt_v16_lut(st, _popped)
                        if (FAST_REPLAY_V11 or FAST_REPLAY_V14) and bn_elig and st.bn_bufs is not None:
                            _bq_m = max(16, 1 << max(0, ql_s - 1).bit_length())
                            _cset_m = (
                                st.bn_inline_compiled
                                if (FUSED_DT_PREP_INLINE_B8 and num_reqs <= 8)
                                else st.bn_compiled
                            )
                            _ib_m = self.input_buffers
                            _rs_m = self.req_states
                            _dr_m = getattr(_rs_m, "draft_tokens", None) if has_d_s else _rs_m.last_sampled_tokens
                            _psig_m = _dt_prep_ptr_sig(
                                _ib_m.is_padding, st.bn_bufs["idx"], st.bn_bufs["cu"], _ib_m.query_start_loc,
                                st.bn_bufs["expanded_idx"], st.bn_bufs["expanded_pos"], _ib_m.positions,
                                _ib_m.seq_lens, _ib_m.input_ids, st.bn_bufs["logits_indices"],
                                _rs_m.num_computed_tokens.gpu, _rs_m.last_sampled_tokens,
                                _dr_m if isinstance(_dr_m, torch.Tensor) else _rs_m.last_sampled_tokens,
                            )
                            if (w_pad_s, has_d_s, 256, _bq_m, 256, _psig_m) not in _cset_m:
                                need_b1_warm = True
                        open_shape_key = (
                            ("dt_s_dirty", num_reqs, nt_s, nt_pad_s, ql_s, bool(has_d_s), idx_tuple)
                            if sconv_b1_dirty
                            else exec_shape_key
                        )
                        opened = False if need_b1_warm else _open_region(st, "dt", open_shape_key, force_slot=0)
                        if opened and st.region is not None:
                            r = st.region
                            if imm_idx_cpu is not None:
                                r.immutable_ptrs.add(int(imm_idx_cpu.data_ptr()))
                            if fp is not None and not sconv_b1_dirty:
                                r.dt_track_unhooked = True
                                r.dt_hook_depth = 0
                                r.dt_unhooked_dev_op = False
                                st.dt_cap_ctx = {
                                    "cert_key": cert_key,
                                    "exec_shape_key": exec_shape_key,
                                    "dt_slot": 0,
                                    "batch_sig": fp[0],
                                    "scheduler_output": scheduler_output,
                                    "batch_req_state": batch_req_state,
                                    "batch_desc": batch_desc,
                                    "is_b1": b1_elig,
                                    "req_ids": eff_req_ids,
                                    "idx_mapping_np": eff_idx_np,
                                    "num_reqs": num_reqs,
                                    "num_tokens": nt_s,
                                    "num_tokens_after_padding": nt_pad_s,
                                    "num_reqs_padded": nr_pad_s,
                                    "q_len": ql_s,
                                    "has_drafts": has_d_s,
                                    "write_pad": w_pad_s,
                                    "imm_idx_ptr": imm_ptr,
                                    "sconv_builders": sconv_builders,
                                    "sconv_ok": not sconv_b1_dirty,
                                    "haz0": sum(
                                        v for k, v in r.stats.items() if k.startswith("haz:dt:")
                                    ),
                                    "staged0": r.stats["staged"],
                                    "pinned0": r.stats["pinned"],
                                    "prep_fused_ok": False,
                                    "pa_called": False,
                                    "pa_ret_ok": False,
                                    "pa_template": None,
                                    "ms_pa_ok": True,
                                    "pie_called": False,
                                    "pie_static_ok": False,
                                }
                    else:
                        if (FAST_REPLAY_V11 or FAST_REPLAY_V14) and bn_elig and st.bn_bufs is not None:
                            _ql_e = int(batch_req_state.num_tokens) // num_reqs
                            _has_d_e = bool(getattr(scheduler_output, "scheduled_spec_decode_tokens", None))
                            _w_pad_e = _get_write_pad(st)
                            _bq_e = max(16, 1 << max(0, _ql_e - 1).bit_length())
                            _cset_e = (
                                st.bn_inline_compiled
                                if (FUSED_DT_PREP_INLINE_B8 and num_reqs <= 8)
                                else st.bn_compiled
                            )
                            _ib_e = self.input_buffers
                            _rs_e = self.req_states
                            _dr_e = getattr(_rs_e, "draft_tokens", None) if _has_d_e else _rs_e.last_sampled_tokens
                            _psig_e = _dt_prep_ptr_sig(
                                _ib_e.is_padding, st.bn_bufs["idx"], st.bn_bufs["cu"], _ib_e.query_start_loc,
                                st.bn_bufs["expanded_idx"], st.bn_bufs["expanded_pos"], _ib_e.positions,
                                _ib_e.seq_lens, _ib_e.input_ids, st.bn_bufs["logits_indices"],
                                _rs_e.num_computed_tokens.gpu, _rs_e.last_sampled_tokens,
                                _dr_e if isinstance(_dr_e, torch.Tensor) else _rs_e.last_sampled_tokens,
                            )
                            if (_w_pad_e, _has_d_e, 256, _bq_e, 256, _psig_e) not in _cset_e:
                                need_b1_warm = True
                        if not need_b1_warm:
                            shape_key = (
                                num_reqs,
                                int(batch_req_state.num_tokens),
                                int(batch_desc.num_tokens),
                            )
                            _open_region(st, "dt", shape_key)
            r_prep = st.region
            tracked_prep = r_prep is not None and r_prep.dt_track_unhooked
            if tracked_prep:
                r_prep.dt_hook_depth += 1
            try:
                if b1_elig:
                    if st.b1_dtypes is None:
                        ib = orig_prep(self, scheduler_output, batch_req_state, batch_desc, num_active_loras)
                        _probe_b1_from_ib(st, ib)
                        if st.b1_dtypes is not None and not st.b1_failed:
                            _ensure_b1_bufs(
                                st,
                                self.device,
                                max(int(batch_req_state.num_tokens), int(batch_desc.num_tokens)),
                            )
                        return ib
                    ib = _try_fused_prepare_inputs_b1(
                        self,
                        st,
                        scheduler_output,
                        batch_req_state,
                        batch_desc,
                        skip_elig_check=DTGRAPH_FASTPATH,
                    )
                    if ib is not None:
                        if st.dt_cap_ctx is not None:
                            st.dt_cap_ctx["prep_fused_ok"] = True
                        return ib
                elif bn_elig:
                    if st.b1_dtypes is None:
                        ib = orig_prep(self, scheduler_output, batch_req_state, batch_desc, num_active_loras)
                        _probe_b1_from_ib(st, ib)
                        if st.b1_dtypes is not None and not st.b1_failed:
                            _ensure_bn_bufs(
                                st,
                                self.device,
                                len(batch_req_state.req_ids),
                                max(int(batch_req_state.num_tokens), int(batch_desc.num_tokens)),
                            )
                        return ib
                    ib = _try_fused_prepare_inputs_bn(
                        self,
                        st,
                        scheduler_output,
                        batch_req_state,
                        batch_desc,
                        skip_elig_check=DTGRAPH_FASTPATH,
                        precomputed_rowar=precomputed_rowar,
                        imm_idx_cpu=imm_idx_cpu,
                    )
                    if ib is not None:
                        if st.dt_cap_ctx is not None:
                            st.dt_cap_ctx["prep_fused_ok"] = True
                        return ib
                return orig_prep(self, scheduler_output, batch_req_state, batch_desc, num_active_loras)
            finally:
                if tracked_prep:
                    r_prep.dt_hook_depth -= 1

        # 3. Wrap sample_tokens + postprocess_sampled for Phase 2B ("td", post-AsyncOutput)
        # and Lever C (direct req_states.draft_tokens scatter in _vw_conf + skip readback).
        orig_sample_tokens = cls.sample_tokens
        orig_postprocess = cls.postprocess_sampled

        def wrapped_sample_tokens(self, grammar_output):
            if FAST_REPLAY_V16 and DTGRAPH_FASTPATH and draft_store_tls is None and STATIC_RING:
                try:
                    st = self._inkling_dtg
                except AttributeError:
                    st = _get_state(self)
                if st.sample_hooks_done and (st._sf_get_ring_fn is not None or _bind_sf_fns_v16(st)):
                    ems = self.execute_model_state
                    if ems is not None and not st.disabled and "td" in CFG["regions"]:
                        st.cur_ib = ems.input_batch
                        st.cur_num_spec = ems.num_spec_tokens_to_schedule or 0
                    else:
                        st.cur_ib = None
                        st.cur_num_spec = 0
                    st._td_v16_hit = False
                    if st._td_v16_armed is not None:
                        st._td_v16_armed = None
                    clear_fn = st._sf_clear_fn
                    if clear_fn is not None:
                        clear_fn()
                    try:
                        out = orig_sample_tokens(self, grammar_output)
                    except BaseException:
                        st.td_static_armed = None
                        st._td_v16_armed = None
                        st.td_cap_ctx = None
                        st.cur_ring_slot = None
                        st.cur_ring_fenced = True
                        st.cur_ib = None
                        st.cur_num_spec = 0
                        abort_region(self)
                        raise
                    if st._td_v16_hit:
                        st._td_v16_hit = False
                        st.cur_ring_slot = None
                        st.cur_ib = None
                        st.cur_num_spec = 0
                        return out
                    if st.region is not None and (st.region.active or st.mode is not None):
                        close_region(self, expected_name="td")
                    if st.cur_ring_slot is not None and not st.cur_ring_fenced and st._sf_fence_fn is not None:
                        st._sf_fence_fn(self.device, st.cur_ring_slot)
                    st.td_static_armed = None
                    st._td_v16_armed = None
                    st.td_cap_ctx = None
                    st.cur_ring_slot = None
                    st.cur_ring_fenced = True
                    st.cur_ib = None
                    st.cur_num_spec = 0
                    return out
            st = _get_state(self)
            ems = getattr(self, "execute_model_state", None)
            spec = getattr(self, "speculator", None)
            dth = getattr(self, "draft_tokens_handler", None)
            if not (DTGRAPH_FASTPATH and st.sample_hooks_done):
                # Hook draft query_cudagraph_manager.run_fullgraph once to close TD capture right before
                # replay, and hook spec._generate_draft once for eager draft fallback.
                # CRITICAL PyTorch LIFO invariant: DFlashSpeculator.propose() is decorated with
                # @torch.inference_mode() (ForceDispatchKeyGuard snapshots TLS DispatchKey::Python on
                # entry and restores it on exit!). Therefore close_region inside propose() uses
                # keep_mode=True (ends CUDA graph capture immediately; r.dispatch becomes a 1-line no-op),
                # and _prop's finally block exits TorchDispatchMode strictly AFTER propose() returns!
                qcgm = getattr(spec, "query_cudagraph_manager", None) if spec is not None else None
                if qcgm is not None and not getattr(qcgm, "_inkling_dtg_qcgm", False):
                    orig_qrfg = qcgm.run_fullgraph

                    def _qrfg(batch_desc, _orig=orig_qrfg, _runner=self, _st=st):
                        if _st.td_cap_ctx is not None and _st.region is not None and _st.region.active:
                            _st.td_cap_ctx["draft_batch_desc"] = batch_desc
                        close_region(_runner, expected_name="td", keep_mode=True)
                        if _st.td_cap_ctx is not None and _st.region is not None:
                            r = _st.region
                            haz_now = sum(v for k, v in r.stats.items() if k.startswith("haz:td:"))
                            _st.td_cap_ctx["pure_ok"] = (
                                r.seg == 1
                                and haz_now == _st.td_cap_ctx["haz0"]
                                and r.stats["staged"] == _st.td_cap_ctx["staged0"]
                                and r.stats["pinned"] == _st.td_cap_ctx["pinned0"]
                            )
                        return _orig(batch_desc)

                    qcgm._inkling_dtg_orig_qrfg = orig_qrfg
                    qcgm.run_fullgraph = _qrfg
                    qcgm._inkling_dtg_qcgm = True
                if (
                    STATIC_RING
                    and spec is not None
                    and hasattr(spec, "prepare_propose")
                    and not getattr(spec, "_inkling_dtg_pp", False)
                ):
                    orig_pp = spec.prepare_propose

                    def _wrapped_pp(*a, _orig=orig_pp, _st=st, _sp=spec, **kw):
                        if _st.td_static_armed is not None:
                            _sp._deferred_dink_update = None
                            return None
                        ret_pp = _orig(*a, **kw)
                        if _st.td_cap_ctx is not None:
                            _st.td_cap_ctx["prep_ret_none"] = (ret_pp is None)
                        return ret_pp

                    spec.prepare_propose = _wrapped_pp
                    spec._inkling_dtg_pp = True
                if spec is not None and hasattr(spec, "_generate_draft") and not getattr(spec, "_inkling_dtg_gd", False):
                    orig_gd = spec._generate_draft

                    def _gd(*a, _orig=orig_gd, _runner=self, **kw):
                        close_region(_runner, expected_name="td", keep_mode=True)
                        return _orig(*a, **kw)

                    spec._generate_draft = _gd
                    spec._inkling_dtg_gd = True
                # Always hook spec.propose (even when FUSED_DRAFT_STORE=0) so deferred
                # TorchDispatchMode.__exit__() ALWAYS executes strictly outside propose()'s
                # @torch.inference_mode() guard (plus Lever C _DraftSkipProxy when active).
                if (
                    spec is not None
                    and hasattr(spec, "propose")
                    and not getattr(spec, "_inkling_dtg_prop", False)
                ):
                    orig_prop = spec.propose

                    def _prop(*a, _orig=orig_prop, _runner=self, _st=st, _tls=draft_store_tls, _sp=spec, _qcgm=qcgm, **kw):
                        if FAST_REPLAY_V16 and not kw:
                            td_b = _st._td_v16_armed
                            if td_b is not None and not _sp.context_kv_only:
                                _st._td_v16_armed = None
                                _st.td_static_armed = None
                                (
                                    _q_fn, _d_bd, _rs_c, _nr_c,
                                    _dtok_t, _dtok_p, _d_view, _fd_view,
                                    _w_ok, _idx_m, _d_cols,
                                ) = td_b
                                if _q_fn is not None:
                                    _q_fn(_d_bd)
                                elif _qcgm is not None:
                                    _q_fn = getattr(_qcgm, "_inkling_dtg_orig_qrfg", _qcgm.run_fullgraph)
                                    _q_fn(_d_bd)
                                _l2pf_s = getattr(_sp, "_l2pf", None)
                                if (
                                    _l2pf_s is not None
                                    and _nr_c <= getattr(_sp, "_l2pf_target_l0_max_reqs", 2)
                                    and hasattr(_l2pf_s, "fork_cross_boundary")
                                ):
                                    _l2pf_s.fork_cross_boundary("target_l0")
                                _fence_fn = _st._sf_fence_fn
                                if _fence_fn is not None:
                                    _fence_fn(_runner.device, _rs_c, stream=_st.td_static_stream)
                                    _st.cur_ring_fenced = True
                                _st._td_v16_hit = True
                                if _d_view is not None and _sp.draft_tokens is _dtok_t and _dtok_t.data_ptr() == _dtok_p:
                                    drafts = _d_view
                                    _full_d_cached = _fd_view
                                else:
                                    drafts = _sp.draft_tokens[:_nr_c, :_d_cols]
                                    _full_d_cached = None
                                if _tls is not None and _w_ok and _st.draft_store_active and _st.draft_store_real is not None:
                                    _full_d = _full_d_cached if _full_d_cached is not None else _sp.draft_tokens[:_nr_c]
                                    _tls.written_drafts = drafts
                                    _tls.written_full_drafts = _full_d
                                    _tls.written_idx = _idx_m
                                    stats = _st.region.stats if _st.region is not None else None
                                    proxy = _st.draft_proxy
                                    if proxy is None:
                                        proxy = _st.draft_proxy = _DraftSkipProxy(
                                            _st.draft_store_real, _idx_m, drafts, stats, _full_d
                                        )
                                    else:
                                        proxy.reset(_st.draft_store_real, _idx_m, drafts, stats, _full_d)
                                    _runner.req_states.draft_tokens = proxy
                                return drafts
                        if FAST_REPLAY_V14 and not kw:
                            _is_dummy_call = False
                            _skip_draft_call = bool(getattr(_sp, "context_kv_only", False))
                        else:
                            _is_dummy_call = bool(
                                kw.get("dummy_run", False)
                                or kw.get("skip_attn_for_dummy_run", False)
                                or kw.get("is_dummy", False)
                                or kw.get("is_profile", False)
                            )
                            _skip_draft_call = (
                                (kw.get("num_speculative_tokens") == 0)
                                or bool(kw.get("skip_draft", False))
                                or bool(getattr(_sp, "context_kv_only", False))
                            )
                        if _st.td_static_armed is not None and (_is_dummy_call or _skip_draft_call or _qcgm is None):
                            _st.td_static_armed = None
                        if (
                            _st.td_static_armed is not None
                            and not _is_dummy_call
                            and not _skip_draft_call
                            and _qcgm is not None
                        ):
                            cert = _st.td_static_armed
                            _st.td_static_armed = None
                            if FAST_REPLAY_V14:
                                _orig_qrfg = _st._orig_qrfg
                                if _orig_qrfg is None:
                                    _orig_qrfg = getattr(_qcgm, "_inkling_dtg_orig_qrfg", _qcgm.run_fullgraph)
                                    _st._orig_qrfg = _orig_qrfg
                            else:
                                _orig_qrfg = getattr(_qcgm, "_inkling_dtg_orig_qrfg", _qcgm.run_fullgraph)
                            _orig_qrfg(cert["draft_batch_desc"])
                            _nr_c = cert["num_reqs"]
                            _l2pf_s = getattr(_sp, "_l2pf", None)
                            if (
                                _l2pf_s is not None
                                and _nr_c <= getattr(_sp, "_l2pf_target_l0_max_reqs", 2)
                                and hasattr(_l2pf_s, "fork_cross_boundary")
                            ):
                                _l2pf_s.fork_cross_boundary("target_l0")
                            stream_c = _st.td_static_stream if (FAST_REPLAY_V11 or FAST_REPLAY_V14 or FAST_REPLAY_V16) else cert["stream"]
                            if (not (FAST_REPLAY_V11 or FAST_REPLAY_V14 or FAST_REPLAY_V16)) and _st.region is not None:
                                _st.region.b.fence(cert["ring_slot"], stream_c)
                            sf_m = _get_sf_ring_mod_v14(_st) if (FAST_REPLAY_V14 or FAST_REPLAY_V16) else _get_sync_free_ring_mod()
                            if sf_m is not None:
                                sf_m.fence_static_ring_slot(_runner.device, cert["ring_slot"], stream=stream_c)
                                _st.cur_ring_fenced = True
                            v14_p = cert.get("v14") if FAST_REPLAY_V14 else None
                            if (
                                v14_p is not None
                                and v14_p["drafts_view"] is not None
                                and _sp.draft_tokens is v14_p["draft_tok_t"]
                                and _sp.draft_tokens.data_ptr() == v14_p["draft_tok_ptr"]
                            ):
                                drafts = v14_p["drafts_view"]
                                _full_d_cached = v14_p["full_drafts_view"]
                            else:
                                drafts = _sp.draft_tokens[:_nr_c, :cert["drafts_cols"]]
                                _full_d_cached = None
                            if (
                                _tls is not None
                                and cert["write_req_ok"]
                                and _st.draft_store_active
                                and _st.draft_store_real is not None
                            ):
                                _full_d = _full_d_cached if _full_d_cached is not None else _sp.draft_tokens[:_nr_c]
                                _tls.written_drafts = drafts
                                _tls.written_full_drafts = _full_d
                                _tls.written_idx = cert["idx_mapping"]
                                stats = _st.region.stats if _st.region is not None else None
                                if DTGRAPH_FASTPATH or FAST_REPLAY_V14:
                                    proxy = _st.draft_proxy
                                    if proxy is None:
                                        proxy = _st.draft_proxy = _DraftSkipProxy(
                                            _st.draft_store_real, cert["idx_mapping"], drafts, stats, _full_d
                                        )
                                    else:
                                        proxy.reset(_st.draft_store_real, cert["idx_mapping"], drafts, stats, _full_d)
                                    _runner.req_states.draft_tokens = proxy
                                else:
                                    _runner.req_states.draft_tokens = _DraftSkipProxy(
                                        _st.draft_store_real, cert["idx_mapping"], drafts, stats, _full_d
                                    )
                            return drafts
                        try:
                            drafts = _orig(*a, **kw)
                        finally:
                            close_region(_runner, expected_name="td", keep_mode=False)
                        cap = _st.td_cap_ctx
                        _st.td_cap_ctx = None
                        if (
                            cap is not None
                            and cap.get("pure_ok", False)
                            and cap.get("pp_ret_none", False)
                            and cap.get("prep_ret_none", True)
                            and cap.get("draft_batch_desc") is not None
                            and not getattr(_sp, "_last_sample_seq_eager", True)
                            and not getattr(_sp, "_last_propose_post_graph_eager", True)
                            and _td_static_spec_ok(_sp, cap["num_reqs"], num_spec_sched=cap.get("num_spec_sched"))
                            and isinstance(drafts, torch.Tensor)
                            and drafts.data_ptr() == _sp.draft_tokens.data_ptr()
                        ):
                            _w_ok = bool(
                                _tls is not None
                                and _st.draft_store_active
                                and getattr(_tls, "written_drafts", None) is drafts
                                and getattr(_tls, "written_idx", None) is not None
                            )
                            _td_exec_gen = (
                                getattr(_st.region.b, "exec_gen", {}).get(cap["exec_key"])
                                if _st.region is not None
                                else None
                            )
                            _td_cert_entry = {
                                "exec_key": cap["exec_key"],
                                "exec_gen": _td_exec_gen,
                                "ring_slot": cap["ring_slot"],
                                "batch_sig": cap["batch_sig"],
                                "ptr_sig": cap["ptr_sig"],
                                "draft_batch_desc": cap["draft_batch_desc"],
                                "num_reqs": cap["num_reqs"],
                                "drafts_cols": int(drafts.shape[1]),
                                "write_req_ok": _w_ok,
                                "idx_mapping": _tls.written_idx if _w_ok else cap["idx_mapping"],
                            }
                            if FAST_REPLAY_V14 or FAST_REPLAY_V16:
                                _compile_td_v14_plan(_td_cert_entry, _sp, _st, drafts)
                                if FAST_REPLAY_V16:
                                    _compile_td_v16_plan(_td_cert_entry, _sp, _st, drafts, qcgm=_qcgm)
                                _st._td_last_cert_by_slot[cap["ring_slot"] % NSLOT] = _td_cert_entry
                            _st.td_static_cert[cap["cert_key"]] = _td_cert_entry
                        if (
                            _tls is not None
                            and _st.draft_store_active
                            and _st.draft_store_real is not None
                            and getattr(_tls, "written_drafts", None) is drafts
                            and getattr(_tls, "written_idx", None) is not None
                        ):
                            stats = _st.region.stats if _st.region is not None else None
                            full_d = getattr(_tls, "written_full_drafts", None)
                            if DTGRAPH_FASTPATH:
                                proxy = _st.draft_proxy
                                if proxy is None:
                                    proxy = _st.draft_proxy = _DraftSkipProxy(
                                        _st.draft_store_real,
                                        _tls.written_idx,
                                        drafts,
                                        stats,
                                        full_d,
                                    )
                                else:
                                    proxy.reset(
                                        _st.draft_store_real,
                                        _tls.written_idx,
                                        drafts,
                                        stats,
                                        full_d,
                                    )
                                _runner.req_states.draft_tokens = proxy
                            else:
                                _runner.req_states.draft_tokens = _DraftSkipProxy(
                                    _st.draft_store_real,
                                    _tls.written_idx,
                                    drafts,
                                    stats,
                                    full_d,
                                )
                        return drafts

                    spec.propose = _prop
                    spec._inkling_dtg_prop = True
                if (
                    draft_store_tls is not None
                    and dth is not None
                    and hasattr(dth, "set_draft_tokens")
                    and not getattr(dth, "_inkling_dtg_sdt", False)
                ):
                    orig_sdt = dth.set_draft_tokens

                    def _sdt(*a, _orig=orig_sdt, _runner=self, _st=st, **kw):
                        if _st.draft_store_real is not None and isinstance(
                            getattr(_runner.req_states, "draft_tokens", None), _DraftSkipProxy
                        ):
                            _runner.req_states.draft_tokens = _st.draft_store_real
                        return _orig(*a, **kw)

                    dth.set_draft_tokens = _sdt
                    dth._inkling_dtg_sdt = True
                if DTGRAPH_FASTPATH and spec is not None:
                    qcgm_ok = qcgm is not None and getattr(qcgm, "_inkling_dtg_qcgm", False)
                    gd_ok = (not hasattr(spec, "_generate_draft")) or getattr(
                        spec, "_inkling_dtg_gd", False
                    )
                    prop_ok = (not hasattr(spec, "propose")) or getattr(
                        spec, "_inkling_dtg_prop", False
                    )
                    sdt_ok = (draft_store_tls is None) or (
                        dth is not None
                        and (
                            not hasattr(dth, "set_draft_tokens")
                            or getattr(dth, "_inkling_dtg_sdt", False)
                        )
                    )
                    if qcgm_ok and gd_ok and prop_ok and sdt_ok:
                        st.sample_hooks_done = True

            if ems is not None and not st.disabled and "td" in CFG["regions"]:
                ib = ems.input_batch
                nspec_sched = int(getattr(ems, "num_spec_tokens_to_schedule", 0) or 0)
                if DTGRAPH_FASTPATH:
                    st.cur_ib = ib
                    st.cur_num_spec = nspec_sched
                else:
                    st.cur_sample_ctx = {"ib": ib, "num_spec": nspec_sched}
            else:
                st.cur_ib = None
                st.cur_num_spec = 0
                st.cur_sample_ctx = None

            st.draft_store_active = False
            st.draft_store_real = None
            if (
                draft_store_tls is not None
                and ems is not None
                and spec is not None
                and getattr(self, "pcp_manager", None) is None
                and getattr(self, "pp_handler", None) is None
                and getattr(self, "adaptive_verification", None) is None
                and getattr(self, "num_speculative_steps", 0) > 0
                and 0
                < int(getattr(ems, "num_spec_tokens_to_schedule", 0) or 0)
                <= int(self.num_speculative_steps)
                and not getattr(ems.input_batch, "has_structured_output_reqs", False)
                and hasattr(self, "req_states")
                and isinstance(getattr(self.req_states, "draft_tokens", None), torch.Tensor)
                and dth is not None
            ):
                st.draft_store_real = self.req_states.draft_tokens
                st.draft_store_active = True
                draft_store_tls.req_drafts = st.draft_store_real
                draft_store_tls.written_drafts = None
                draft_store_tls.written_full_drafts = None
                draft_store_tls.written_idx = None
            if STATIC_RING:
                _sf_clear_m = _get_sf_ring_mod_v14(st) if (FAST_REPLAY_V14 or FAST_REPLAY_V16) else _get_sync_free_ring_mod()
                if _sf_clear_m is not None and hasattr(_sf_clear_m, "clear_static_ring_state"):
                    _sf_clear_m.clear_static_ring_state()
            try:
                out = orig_sample_tokens(self, grammar_output)
                if not (FAST_REPLAY_V14 or FAST_REPLAY_V16) or (st.region is not None and (st.region.active or st.mode is not None)):
                    close_region(self, expected_name="td")
                return out
            except BaseException:
                abort_region(self)
                raise
            finally:
                if st.draft_store_real is not None and isinstance(
                    getattr(self.req_states, "draft_tokens", None), _DraftSkipProxy
                ):
                    self.req_states.draft_tokens = st.draft_store_real
                st.draft_store_active = False
                st.draft_store_real = None
                if draft_store_tls is not None:
                    draft_store_tls.req_drafts = None
                    draft_store_tls.written_drafts = None
                    draft_store_tls.written_full_drafts = None
                    draft_store_tls.written_idx = None
                if STATIC_RING and st.cur_ring_slot is not None and not st.cur_ring_fenced:
                    sf_m = _get_sf_ring_mod_v14(st) if (FAST_REPLAY_V14 or FAST_REPLAY_V16) else _get_sync_free_ring_mod()
                    if sf_m is not None:
                        sf_m.fence_static_ring_slot(self.device, st.cur_ring_slot)
                st.td_static_armed = None
                st.td_cap_ctx = None
                st.cur_ring_slot = None
                st.cur_ring_fenced = True
                st.cur_ib = None
                st.cur_num_spec = 0
                st.cur_sample_ctx = None

        def wrapped_postprocess(self, *args, **kwargs):
            # Wave-V16 O(1) 2D+slot LUT `td_fn_lut[B][K][slot]` + zero-loop validator + SyncFree-Temp compat!
            if FAST_REPLAY_V16 and STATIC_RING and STATIC_RING_REPLAY and DTGRAPH_FASTPATH:
                try:
                    st = self._inkling_dtg
                except AttributeError:
                    st = _get_state(self)
                ib = st.cur_ib
                num_spec_sched = st.cur_num_spec
                if ib is not None and 0 < num_spec_sched <= _LUT_MAX_K:
                    get_ring = st._sf_get_ring_fn
                    if get_ring is None and _bind_sf_fns_v16(st):
                        get_ring = st._sf_get_ring_fn
                    if get_ring is not None:
                        _cand_ring = get_ring()
                        if _cand_ring is not None:
                            r_slot, p_s, p_ns, p_nr = _cand_ring
                            if p_s and p_ns and p_nr and (
                                (
                                    len(args) == 3
                                    and not kwargs
                                    and args[0].data_ptr() == p_s
                                    and args[1].data_ptr() == p_ns
                                    and args[2].data_ptr() == p_nr
                                )
                                or _postprocess_args_match_ring(args, kwargs, _cand_ring)
                            ):
                                nr_td = ib.num_reqs
                                if 0 < nr_td <= _LUT_MAX_B:
                                    fn_td = st.td_fn_lut[nr_td][num_spec_sched][r_slot & 3]
                                    if fn_td is not None:
                                        try:
                                            if fn_td(self, ib, num_spec_sched, p_s, p_ns, p_nr):
                                                return None
                                        except Exception:  # noqa: BLE001
                                            pass
            else:
                st = _get_state(self)
                if DTGRAPH_FASTPATH:
                    ib = st.cur_ib
                    num_spec_sched = st.cur_num_spec
                else:
                    ctx = st.cur_sample_ctx
                    ib = ctx["ib"] if ctx is not None else None
                    num_spec_sched = ctx["num_spec"] if ctx is not None else 0
            if ib is not None and not st.disabled and "td" in CFG["regions"]:
                # Wave-V14 zero-alloc fast-entry (avoids 12 getattr + td_ineligible kwdict on steady hits!)
                if (
                    FAST_REPLAY_V14
                    and STATIC_RING
                    and STATIC_RING_REPLAY
                    and st.region is not None
                    and num_spec_sched > 0
                ):
                    try:
                        spec = self.speculator
                        if (
                            spec is not None
                            and spec.query_cudagraph_manager is not None
                            and not ib.has_prefill
                            and self.pcp_manager is None
                            and (self.dcp_size or 1) <= 1
                            and not (spec.supports_mm_inputs and self.model_state.supports_mm_inputs)
                            and not spec.context_kv_only
                            and not spec._cl_on
                            and not spec._vw_stats_on
                            and (CFG["max_reqs"] == 0 or ib.num_reqs <= CFG["max_reqs"])
                        ):
                            sf_m = _get_sf_ring_mod_v14(st)
                            if sf_m is not None:
                                _cand_ring = sf_m.get_static_ring_state()
                                if _cand_ring is not None and _postprocess_args_match_ring(args, kwargs, _cand_ring):
                                    r_slot = _cand_ring[0]
                                    st.cur_ring_slot = r_slot
                                    st.cur_ring_fenced = False
                                    _td_slot_idx = r_slot % NSLOT
                                    _td_mru = st._td_last_cert_by_slot[_td_slot_idx]
                                    if _td_mru is not None and _td_v14_validate(_td_mru, ib, num_spec_sched, _cand_ring, spec, st):
                                        v14_td = _td_mru["v14"]
                                        stream = _ensure_region(st)
                                        if st.region.replay_static_region_v14(
                                            stream,
                                            _STAT_STATIC_REPLAY_TD,
                                            v14_td["exec_key"],
                                            v14_td["exec_owner"],
                                            v14_td["exec_handle"],
                                            v14_td["exec_gen"],
                                        ):
                                            if FAST_REPLAY_V16 and "v16_td_step" not in _td_mru:
                                                _compile_td_v16_plan(
                                                    _td_mru,
                                                    spec,
                                                    st,
                                                    getattr(spec, "draft_tokens", None),
                                                    qcgm=getattr(spec, "query_cudagraph_manager", None),
                                                )
                                            st.td_static_stream = stream
                                            st.td_static_armed = _td_mru
                                            return None
                                        st._td_last_cert_by_slot[_td_slot_idx] = None
                    except Exception:  # noqa: BLE001
                        pass
                spec = getattr(self, "speculator", None)
                qcgm = getattr(spec, "query_cudagraph_manager", None) if spec is not None else None
                why = td_ineligible(
                    has_speculator=spec is not None,
                    has_full_draft_cg=qcgm is not None,
                    has_prefill=bool(getattr(ib, "has_prefill", False)),
                    pcp=getattr(self, "pcp_manager", None) is not None,
                    dcp_size=int(getattr(self, "dcp_size", 1) or 1),
                    mm_inputs=bool(
                        spec is not None
                        and getattr(spec, "supports_mm_inputs", False)
                        and getattr(self.model_state, "supports_mm_inputs", False)
                    ),
                    num_spec_tokens=num_spec_sched,
                    context_kv_only=bool(getattr(spec, "context_kv_only", False)) if spec else True,
                    conf_log=bool(getattr(spec, "_cl_on", False)) if spec else False,
                    vw_stats=bool(getattr(spec, "_vw_stats_on", False)) if spec else False,
                    num_reqs=int(getattr(ib, "num_reqs", 1)),
                    max_reqs=CFG["max_reqs"],
                )
                if why is not None:
                    st.skips[f"td:{why}"] += 1
                else:
                    ring_state = None
                    if STATIC_RING:
                        sf_m = _get_sf_ring_mod_v14(st) if (FAST_REPLAY_V14 or FAST_REPLAY_V16) else _get_sync_free_ring_mod()
                        if sf_m is not None:
                            _cand_ring = sf_m.get_static_ring_state()
                            if _cand_ring is not None and _postprocess_args_match_ring(args, kwargs, _cand_ring):
                                ring_state = _cand_ring
                    if ring_state is not None:
                        r_slot = ring_state[0]
                        st.cur_ring_slot = r_slot
                        st.cur_ring_fenced = False
                        # Wave-V14 O(1) slot-indexed single-pass "td" static replay (0 tuple/fingerprint/dup-spec allocs!)
                        if FAST_REPLAY_V14 and STATIC_RING_REPLAY and st.region is not None:
                            _td_slot_idx = r_slot % NSLOT
                            _td_mru = st._td_last_cert_by_slot[_td_slot_idx]
                            if _td_mru is not None and _td_v14_validate(_td_mru, ib, num_spec_sched, ring_state, spec, st):
                                try:
                                    v14_td = _td_mru["v14"]
                                    stream = _ensure_region(st)
                                    if st.region.replay_static_region_v14(
                                        stream,
                                        _STAT_STATIC_REPLAY_TD,
                                        v14_td["exec_key"],
                                        v14_td["exec_owner"],
                                        v14_td["exec_handle"],
                                        v14_td["exec_gen"],
                                    ):
                                        st.td_static_stream = stream
                                        st.td_static_armed = _td_mru
                                        return None
                                except Exception:  # noqa: BLE001
                                    pass
                                st._td_last_cert_by_slot[_td_slot_idx] = None
                        shape_key = (int(ib.num_reqs), int(ib.num_tokens), int(num_spec_sched))
                        cert_key = (shape_key, r_slot)
                        if FAST_REPLAY_V14 and STATIC_RING_REPLAY and st.region is not None:
                            _td_lu = st.td_static_cert.get(cert_key)
                            if (
                                _td_lu is not None
                                and _td_lu is not st._td_last_cert_by_slot[r_slot % NSLOT]
                                and _td_v14_validate(_td_lu, ib, num_spec_sched, ring_state, spec, st)
                            ):
                                try:
                                    v14_lu = _td_lu["v14"]
                                    stream = _ensure_region(st)
                                    if st.region.replay_static_region_v14(
                                        stream,
                                        _STAT_STATIC_REPLAY_TD,
                                        v14_lu["exec_key"],
                                        v14_lu["exec_owner"],
                                        v14_lu["exec_handle"],
                                        v14_lu["exec_gen"],
                                    ):
                                        st._td_last_cert_by_slot[r_slot % NSLOT] = _td_lu
                                        st.td_static_stream = stream
                                        st.td_static_armed = _td_lu
                                        return None
                                except Exception:  # noqa: BLE001
                                    pass
                        batch_sig, ptr_sig = _td_static_fingerprint(ib, num_spec_sched, ring_state, spec=spec)
                        cert = st.td_static_cert.get(cert_key) if STATIC_RING_REPLAY else None
                        if (
                            cert is not None
                            and st.region is not None
                            and cert["batch_sig"] == batch_sig
                            and cert["ptr_sig"] == ptr_sig
                            and _td_static_spec_ok(spec, int(ib.num_reqs), num_spec_sched=num_spec_sched)
                        ):
                            try:
                                stream = _ensure_region(st)
                                if FAST_REPLAY_V11 or FAST_REPLAY_V14:
                                    _td_ok = st.region.replay_static_region(
                                        stream,
                                        "td",
                                        shape_key,
                                        r_slot,
                                        wait_slot=False,
                                        expected_exec_gen=cert.get("exec_gen"),
                                    )
                                    if _td_ok:
                                        if (FAST_REPLAY_V14 or FAST_REPLAY_V16) and "v14" not in cert:
                                            _compile_td_v14_plan(cert, spec, st, getattr(spec, "draft_tokens", None))
                                        if FAST_REPLAY_V16 and "v16_td_step" not in cert:
                                            _compile_td_v16_plan(cert, spec, st, getattr(spec, "draft_tokens", None), qcgm=qcgm)
                                        elif FAST_REPLAY_V16:
                                            _register_td_v16_lut(st, cert, int(ib.num_reqs), num_spec_sched, r_slot)
                                        if FAST_REPLAY_V14 or FAST_REPLAY_V16:
                                            st._td_last_cert_by_slot[r_slot % NSLOT] = cert
                                        st.td_static_stream = stream
                                        st.td_static_armed = cert
                                        return None
                                elif st.region.replay_static_region(stream, "td", shape_key, r_slot):
                                    cert_armed = dict(cert)
                                    cert_armed["stream"] = stream
                                    st.td_static_armed = cert_armed
                                    return None
                            except Exception:  # noqa: BLE001
                                _pop_td = st.td_static_cert.pop(cert_key, None)
                                if _pop_td is not None:
                                    if st._td_last_cert_by_slot[r_slot % NSLOT] is _pop_td:
                                        st._td_last_cert_by_slot[r_slot % NSLOT] = None
                                    if FAST_REPLAY_V16:
                                        _evict_td_v16_lut(st, _pop_td)
                        opened = _open_region(st, "td", shape_key, force_slot=r_slot)
                        if opened and st.region is not None:
                            r = st.region
                            st.td_cap_ctx = {
                                "cert_key": cert_key,
                                "exec_key": ("td", shape_key, 0, r_slot),
                                "ring_slot": r_slot,
                                "batch_sig": batch_sig,
                                "ptr_sig": ptr_sig,
                                "num_reqs": int(ib.num_reqs),
                                "num_spec_sched": int(num_spec_sched),
                                "idx_mapping": ib.idx_mapping,
                                "haz0": sum(v for k, v in r.stats.items() if k.startswith("haz:td:")),
                                "staged0": r.stats["staged"],
                                "pinned0": r.stats["pinned"],
                                "pure_ok": False,
                                "pp_ret_none": False,
                                "prep_ret_none": True,
                                "draft_batch_desc": None,
                            }
                    else:
                        shape_key = (int(ib.num_reqs), int(ib.num_tokens), int(num_spec_sched))
                        _open_region(st, "td", shape_key)
            pp_res = orig_postprocess(self, *args, **kwargs)
            if st.td_cap_ctx is not None:
                st.td_cap_ctx["pp_ret_none"] = (pp_res is None)
            return pp_res

        # 4. Register static req_states.draft_tokens buffer before speculator.capture()
        #    so in-graph _vw_conf_kernel can write directly into req_states.draft_tokens,
        #    and optionally run opt-in INKLING_PREFILL_PREWARM right after capture completes.
        if (FUSED_DRAFT_STORE or PREFILL_PREWARM or FUSED_DT_PREP_PREWARM) and hasattr(cls, "capture_model"):
            orig_cap = cls.capture_model

            def wrapped_capture_model(self, *args, _orig=orig_cap, **kwargs):
                if FUSED_DRAFT_STORE:
                    spec = getattr(self, "speculator", None)
                    rs = getattr(self, "req_states", None)
                    if (
                        spec is not None
                        and rs is not None
                        and hasattr(spec, "set_req_drafts_buffer")
                        and isinstance(getattr(rs, "draft_tokens", None), torch.Tensor)
                    ):
                        spec.set_req_drafts_buffer(rs.draft_tokens)
                res = _orig(self, *args, **kwargs)
                if PREFILL_PREWARM:
                    prewarm_prefill_c32(self)
                elif FUSED_DT_PREP_PREWARM:
                    _prewarm_fused_dt_prep_kernels(self)
                return res

            cls.capture_model = wrapped_capture_model

        cls.execute_model = wrapped_execute_model
        cls.prepare_inputs = wrapped_prepare_inputs
        if STATIC_DT_REPLAY and orig_prep_attn is not None:
            cls.prepare_attn = wrapped_prepare_attn
        cls.sample_tokens = wrapped_sample_tokens
        cls.postprocess_sampled = wrapped_postprocess
        cls._inkling_dtg_installed = True
        _logger().info(
            "INKLING_DTGRAPH hooks installed on GPUModelRunner "
            "(regions=%s, update=%s, max_execs=%d, fused_dt_prep=%d[max_b=%d,inline_b8=%d], "
            "direct_embed=%d, fused_draft_store=%d, prefill_prewarm=%d, fastpath=%d, rowar_desc_w=%d, "
            "static_ring=%d, static_dt=%d)",
            sorted(CFG["regions"]),
            CFG["update"],
            CFG["max_execs"],
            int(FUSED_DT_PREP),
            FUSED_DT_PREP_MAX_B,
            int(FUSED_DT_PREP_INLINE_B8),
            int(FUSED_EMBED_DIRECT),
            int(FUSED_DRAFT_STORE),
            int(PREFILL_PREWARM),
            int(DTGRAPH_FASTPATH),
            int(ROWAR_DESC_W_ORDER),
            int(STATIC_RING),
            int(STATIC_DT_REPLAY),
        )
        return True
    except Exception as e:  # noqa: BLE001
        _logger().warning("INKLING_DTGRAPH install failed (%r); staying eager", e)
        return False
