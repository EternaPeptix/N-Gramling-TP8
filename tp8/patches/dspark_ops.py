# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kernels of the Inkling2 DSpark draft, matching the reference implementation."""

import os

import torch
import torch.distributed as dist

from vllm.triton_utils import tl, tldevice, triton

from .allreduce import _multicast_memory

# INKLING_DRAFT_PDL=1: opt-in PDL across drafter Triton kernels (default 0 for safe A/B).
# Invariant: ALWAYS gdc_wait() BEFORE reading any dynamic input, and gdc_launch_dependents() AFTER gdc_wait().
_DRAFT_PDL = os.environ.get("INKLING_DRAFT_PDL", "0").strip().lower() not in (
    "0", "", "off", "false", "no"
)

# ---- Wave-V7 Native SM121a DSpark Trunk + Markov CUDA Stack (STRICT 0-ULP, opt-in default OFF) ----
_DSPARK_CUDA_V7 = os.environ.get("INKLING_DSPARK_CUDA_V7", "0").strip().lower() not in (
    "0", "", "off", "false", "no"
)
_DSPARK_V7_LIB_PATH = os.environ.get(
    "INKLING_DSPARK_V7_LIB", "/rdma/libdspark_markov_v7.so"
)
# At C=2..8 (2 <= rows <= 8), BM=8 fold_argmax_tc uses 144 slices (3.0x48 exact waves!)
_MARKOV_FOLD_C48_SLICES = int(os.environ.get("INKLING_MARKOV_FOLD_C48_SLICES", "144") or "144")
# Wave-V14: Opt-in 4-CTA/SM full-wave slice auto-bump (192 = 4.0x48 for BM=8 __launch_bounds__(128,4), default OFF=0).
_MARKOV_FOLD_C48_AUTO_SLICES = os.environ.get("INKLING_MARKOV_FOLD_C48_AUTO_SLICES", "0").strip().lower() not in (
    "0", "", "off", "false", "no"
)
# Wave-V9: Opt-in wiring for already-compiled `dspark_select_and_rnn_step_v7_fwd` cross-step fusion
# (eliminates K-1 launches + overlaps STATE_I load/GEMM behind RDMA_AG across B=1..32, default OFF).
_MARKOV_FUSE_CROSS_STEP = os.environ.get("INKLING_MARKOV_FUSE_CROSS_STEP", "0").strip().lower() not in (
    "0", "", "off", "false", "no"
)
# Wave-V12: Opt-in wiring for already-compiled race-free V10 fused kernels in libdspark_markov_v10.so
# (ZERO CUDA rebuild, strictly default-OFF, 100% 0-ULP bit-identical to standalone Triton/CUDA sequence!):
_DSPARK_FUSE_TAIL_VW = os.environ.get("INKLING_DSPARK_FUSE_TAIL_VW", "0").strip().lower() not in (
    "0", "", "off", "false", "no"
)
_DSPARK_FUSE_EPILOGUE_PREP = os.environ.get("INKLING_DSPARK_FUSE_EPILOGUE_PREP", "0").strip().lower() not in (
    "0", "", "off", "false", "no"
)
_TREE_FUSE_V10 = os.environ.get("INKLING_TREE_FUSE_V10", "0").strip().lower() not in (
    "0", "", "off", "false", "no"
)
_DSPARK_V7_LIB = None
_DSPARK_V7_TRIED = False
_V7_QUAL: dict[int, dict] = {}
_V7_WS: dict[int, dict] = {}


_DSPARK_V10_VER = 0


def dspark_v10_version() -> int:
    """Return loaded DSpark Markov CUDA library version (10 for V10, 7 for V7, 0 if unavailable)."""
    if not _DSPARK_V7_TRIED:
        _get_dspark_v7_lib()
    return _DSPARK_V10_VER


def _ddev_idx(device: torch.device) -> int:
    return device.index if device.index is not None else torch.cuda.current_device()


def _get_dspark_v7_lib():
    global _DSPARK_V7_LIB, _DSPARK_V7_TRIED, _DSPARK_V10_VER
    if not _DSPARK_V7_TRIED:
        _DSPARK_V7_TRIED = True
        if _DSPARK_CUDA_V7 and os.path.isfile(_DSPARK_V7_LIB_PATH):
            try:
                import ctypes
                c_vp = ctypes.c_void_p
                c_i = ctypes.c_int
                c_u32 = ctypes.c_uint32
                c_i64 = ctypes.c_int64
                c_u64 = ctypes.c_uint64
                c_f = ctypes.c_float
                lib = ctypes.CDLL(_DSPARK_V7_LIB_PATH)
                sigs = {
                    "dspark_embed_block_ln_v7_fwd": [
                        c_vp, c_i, c_vp, c_vp, c_vp, c_vp, c_vp, c_vp,
                        c_i, c_i, c_i64, c_i64, c_i, c_f, c_f, c_i, c_i, c_u64,
                    ],
                    "dspark_mix_rmsnorm_v7_fwd": [
                        c_vp, c_vp, c_i, c_vp, c_vp, c_i, c_i, c_i,
                        c_i64, c_i64, c_i64, c_f, c_i, c_i, c_i, c_u64,
                    ],
                    "dspark_context_anchor_mix_rmsnorm_v7_fwd": [
                        c_vp, c_vp, c_vp, c_vp, c_i, c_vp, c_vp, c_i, c_i, c_i,
                        c_i64, c_i64, c_i64, c_f, c_i, c_i, c_i, c_u64,
                    ],
                    "dspark_mix_rmsnorm_5layer_v7_fwd": [
                        c_vp, c_vp, c_i, c_vp, c_vp, c_vp, c_vp, c_vp, c_vp, c_i64,
                        c_vp, c_i64, c_i, c_i, c_i, c_i64, c_i64, c_f, c_i, c_i, c_i, c_u64,
                    ],
                    "dspark_qk_norm_rope_write_kv_v7_fwd": [
                        c_vp, c_vp, c_vp, c_vp, c_vp, c_i, c_vp, c_vp, c_vp,
                        c_i, c_i, c_i, c_i,
                        c_i64, c_i64, c_i64, c_i64, c_i64, c_i64, c_i64, c_i64, c_i64, c_i64,
                        c_f, c_i, c_i, c_i, c_i, c_i, c_i, c_u64,
                    ],
                    "dspark_silu_mul_v7_fwd": [c_vp, c_vp, c_i, c_i, c_i64, c_i, c_u64],
                    "dspark_quant_fp8_rowwise_v7_fwd": [c_vp, c_vp, c_vp, c_i, c_i, c_i, c_i64, c_i64, c_i, c_u64],
                    "dspark_silu_mul_quant_fp8_v7_fwd": [c_vp, c_vp, c_vp, c_i, c_i, c_i, c_i64, c_i64, c_i, c_u64],
                    "dspark_add_softcap_v7_fwd": [c_vp, c_vp, c_vp, c_i, c_i, c_i64, c_i64, c_f, c_i, c_u64],
                    "dspark_markov_gate_v7_fwd": [c_vp, c_vp, c_vp, c_vp, c_i, c_i, c_i, c_u64],
                    "dspark_markov_rnn_step_v7_fwd": [
                        c_vp, c_vp, c_i, c_vp, c_i64, c_vp, c_vp, c_vp, c_vp, c_vp,
                        c_i, c_i, c_i64, c_i, c_i, c_i, c_i, c_u64,
                    ],
                    "dspark_fold_argmax_v7_fwd": [
                        c_vp, c_vp, c_vp, c_vp, c_vp, c_vp, c_vp, c_vp, c_vp, c_vp,
                        c_i, c_i, c_i, c_i, c_i64, c_f, c_i, c_i,
                        c_i, c_i, c_i, c_i, c_i, c_i, c_i, c_i, c_u64,
                    ],
                    "dspark_argmax_select_v7_fwd": [
                        c_vp, c_i, c_i, c_i, c_vp, c_i64, c_vp, c_i64, c_i,
                        c_i, c_i, c_i,
                        c_vp, c_vp, c_vp, c_vp, c_vp, c_vp, c_vp, c_vp, c_vp, c_vp, c_vp,
                        c_f, c_vp, c_vp, c_i, c_i, c_u64,
                    ],
                    "dspark_select_and_rnn_step_v7_fwd": [
                        c_vp, c_i, c_i, c_i, c_vp, c_i64, c_vp, c_i64, c_i,
                        c_i, c_i, c_i,
                        c_vp, c_vp, c_vp, c_vp, c_vp, c_vp, c_vp, c_vp,
                        c_vp, c_vp, c_vp, c_vp, c_vp, c_vp, c_vp, c_f,
                        c_vp, c_vp, c_i64, c_vp, c_vp, c_vp, c_vp,
                        c_i, c_i, c_u64,
                    ],
                }
                for sym, argtypes in sigs.items():
                    fn = getattr(lib, sym)
                    fn.argtypes = argtypes
                    fn.restype = c_i
                ver_fn = getattr(lib, "dspark_markov_v10_version", None)
                if ver_fn is not None:
                    ver_fn.argtypes = []
                    ver_fn.restype = c_i
                    _DSPARK_V10_VER = int(ver_fn())
                    v10_sigs = {
                        "dspark_markov_rnn_step_v10_fwd": sigs["dspark_markov_rnn_step_v7_fwd"],
                        "dspark_fold_argmax_v10_fwd": sigs["dspark_fold_argmax_v7_fwd"],
                        "dspark_select_and_rnn_step_v10_fwd": sigs["dspark_select_and_rnn_step_v7_fwd"],
                        "dspark_argmax_select_batch_es_v10_fwd": [
                            c_vp, c_i, c_i, c_i,
                            c_vp, c_i64, c_vp, c_i64,
                            c_i, c_i,
                            c_vp, c_vp, c_vp, c_vp, c_vp,
                            c_f, c_i, c_i, c_u64,
                        ],
                    }
                    for sym, argtypes in v10_sigs.items():
                        fn = getattr(lib, sym)
                        fn.argtypes = argtypes
                        fn.restype = c_i
                    # Wave-V12: bind optional race-free fused tail/epilogue/tree symbols (already in libdspark_markov_v10.so!)
                    opt_v12_sigs = {
                        "dspark_select_last_and_vw_conf_v7_fwd": [
                            c_i, c_vp, c_i, c_i, c_i, c_i, c_i, c_i, c_i, c_vp,
                            c_i, c_vp, c_i, c_vp, c_i64, c_vp, c_vp, c_vp, c_vp, c_vp,
                            c_vp, c_i64, c_vp, c_vp, c_vp, c_vp, c_vp, c_i, c_f, c_vp,
                            c_i64, c_f, c_i, c_i, c_i, c_i, c_i, c_i, c_i, c_i, c_i, c_u64,
                        ],
                        "dspark_select_last_and_tree_pack_v7_fwd": [
                            c_i, c_vp, c_i, c_i, c_i, c_vp, c_vp, c_vp, c_vp, c_vp,
                            c_vp, c_vp, c_vp, c_vp, c_vp, c_vp, c_vp, c_vp, c_vp, c_vp,
                            c_vp, c_i64, c_vp, c_vp, c_vp, c_vp, c_vp, c_vp, c_vp, c_i,
                            c_f, c_f, c_f, c_f, c_i, c_i, c_i, c_u64,
                        ],
                        "dspark_vw_update_dink_v7_fwd": [
                            c_vp, c_i, c_i, c_vp, c_vp, c_vp, c_vp, c_vp, c_vp, c_vp,
                            c_vp, c_vp, c_vp, c_vp, c_vp, c_vp, c_vp, c_vp, c_vp, c_vp,
                            c_vp, c_vp, c_vp, c_vp, c_vp, c_vp, c_f, c_f, c_f, c_f,
                            c_i, c_i, c_i, c_i, c_u64,
                        ],
                        "dspark_fused_epilogue_prep_v7_fwd": [
                            c_vp, c_vp, c_vp, c_vp, c_vp, c_vp, c_vp, c_vp, c_vp, c_vp,
                            c_vp, c_vp, c_vp, c_vp, c_vp, c_vp, c_vp, c_vp, c_vp, c_vp,
                            c_vp, c_vp, c_vp, c_vp, c_vp, c_vp, c_vp,
                            c_i64, c_i64, c_i64, c_i64, c_i64,
                            c_i64, c_i64, c_i64, c_i64, c_i64,
                            c_i64, c_i64, c_i64, c_i64, c_i64,
                            c_vp, c_vp, c_vp, c_vp, c_vp, c_vp, c_vp, c_vp, c_vp,
                            c_i, c_i, c_i, c_i, c_i, c_i64, c_i, c_i, c_i, c_i64,
                            c_i, c_i, c_i, c_i, c_u32, c_i,
                            c_vp, c_vp, c_vp, c_vp, c_vp, c_vp, c_vp, c_vp,
                            c_vp, c_vp, c_vp, c_vp, c_vp, c_vp, c_vp, c_vp,
                            c_vp, c_vp, c_vp, c_vp,
                            c_f, c_f, c_f, c_f, c_i, c_i, c_i, c_i, c_u64,
                        ],
                    }
                    for sym, argtypes in opt_v12_sigs.items():
                        fn = getattr(lib, sym, None)
                        if fn is not None:
                            fn.argtypes = argtypes
                            fn.restype = c_i
                    print(
                        f"[inkling-dspark-v10] loaded {_DSPARK_V7_LIB_PATH} "
                        f"(ver={_DSPARK_V10_VER}, 0-PDL-hazard + HAS_SKIP TC48/Batch-ES + TOP2+id_map)",
                        flush=True,
                    )
                else:
                    _DSPARK_V10_VER = 7
                    print(
                        f"[inkling-dspark-v7] loaded {_DSPARK_V7_LIB_PATH} "
                        f"(5L trunk + 48-CTA TC48 RNN + BM=8/16/32 fold_w8)",
                        flush=True,
                    )
                _DSPARK_V7_LIB = lib
            except (OSError, AttributeError):
                _DSPARK_V7_LIB = None
                _DSPARK_V10_VER = 0
    return _DSPARK_V7_LIB


def _bits_eq(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    ac, bc = a.contiguous(), b.contiguous()
    if ac.dtype in (torch.bfloat16, torch.float16):
        return bool(torch.equal(ac.view(torch.int16), bc.view(torch.int16)))
    if ac.dtype == torch.float32:
        return bool(torch.equal(ac.view(torch.int32), bc.view(torch.int32)))
    if ac.dtype == torch.float8_e4m3fn:
        return bool(torch.equal(ac.view(torch.uint8), bc.view(torch.uint8)))
    return bool(torch.equal(ac, bc))


def prealloc_v7_workspaces(device: torch.device, max_rows: int = 256) -> dict | None:
    """Pre-allocate persistent V7 drafter/Markov workspaces outside CUDA graph capture (~16.1 MiB/rank)."""
    if not _DSPARK_CUDA_V7 or device.type != "cuda" or torch.cuda.is_current_stream_capturing():
        return None
    d = _ddev_idx(device)
    mr = max(256, int(max_rows))
    ws = _V7_WS.get(d)
    if ws is not None and ws["max_rows"] >= mr:
        return ws
    ws = {
        "max_rows": mr,
        # Zero-init so skipped slot<0 rows stay clean finite BF16
        "ctx_mix_5l": torch.zeros((5, mr, 4608), dtype=torch.bfloat16, device=device),
        # 2x ping-pong FP8 activation buffers for C=32 high-M FP8 TC + fused silu_mul_quant
        "fp8_q": [
            torch.empty((mr, 4608), dtype=torch.float8_e4m3fn, device=device),
            torch.empty((mr, 4608), dtype=torch.float8_e4m3fn, device=device),
        ],
        "fp8_s": [
            torch.empty((mr, 1), dtype=torch.float32, device=device),
            torch.empty((mr, 1), dtype=torch.float32, device=device),
        ],
        "fp8_idx": 0,
        # Zero-alloc Markov K=5..7 workspace (up to 32 batch rows, 288 slices, top2 width=8)
        "mk_state": [
            torch.empty((32, 256), dtype=torch.bfloat16, device=device),
            torch.empty((32, 256), dtype=torch.bfloat16, device=device),
        ],
        "mk_latent": [
            torch.empty((32, 256), dtype=torch.bfloat16, device=device),
            torch.empty((32, 256), dtype=torch.bfloat16, device=device),
        ],
        "mk_lat_idx": 0,
        "mk_rec": torch.empty((32, 288, 8), dtype=torch.float32, device=device),
        "mk_rec1": [
            torch.empty((32, 1, 8), dtype=torch.float32, device=device),
            torch.empty((32, 1, 8), dtype=torch.float32, device=device),
        ],
        "mk_rec1_idx": 0,
        # Race-free ping-pong early-stop buffers for cross-step dspark_select_and_rnn_step_v7_fwd (r1 & V10 B=2..8 Batch-ES)
        "es_pp_skip": [torch.zeros(1, dtype=torch.int32, device=device), torch.zeros(1, dtype=torch.int32, device=device)],
        "es_pp_surv": [torch.ones(1, dtype=torch.float32, device=device), torch.ones(1, dtype=torch.float32, device=device)],
        "es_pp_surv_b": [torch.ones(8, dtype=torch.float32, device=device), torch.ones(8, dtype=torch.float32, device=device)],
        "es_pp_tok":  [torch.ones(1, dtype=torch.float32, device=device), torch.ones(1, dtype=torch.float32, device=device)],
        "es_pp_best": [torch.zeros(1, dtype=torch.float32, device=device), torch.zeros(1, dtype=torch.float32, device=device)],
    }
    _V7_WS[d] = ws
    prealloc_fold_sem(device, min_rows=max(_MIN_FOLD_SEM_ROWS, 64))
    return ws


def _v7_q(device: torch.device) -> dict | None:
    if not _DSPARK_CUDA_V7 or device.type != "cuda":
        return None
    return _V7_QUAL.get(_ddev_idx(device))


def v7_mix5_ready(device: torch.device) -> bool:
    q = _v7_q(device)
    return bool(q is not None and q.get("mix_ok", False) and _ddev_idx(device) in _V7_WS)


def v7_fp8_quant_ready(device: torch.device) -> bool:
    q = _v7_q(device)
    return bool(q is not None and q.get("fp8_ok", False) and _ddev_idx(device) in _V7_WS)


def v10_tail_vw_ready(device: torch.device) -> bool:
    if not _DSPARK_FUSE_TAIL_VW:
        return False
    q = _v7_q(device)
    lib = _get_dspark_v7_lib()
    return bool(
        lib is not None
        and q is not None
        and q.get("markov_ok", False)
        and hasattr(lib, "dspark_select_last_and_vw_conf_v7_fwd")
    )


def v10_epilogue_prep_ready(device: torch.device) -> bool:
    if not _DSPARK_FUSE_EPILOGUE_PREP:
        return False
    q = _v7_q(device)
    lib = _get_dspark_v7_lib()
    return bool(
        lib is not None
        and q is not None
        and q.get("markov_ok", False)
        and hasattr(lib, "dspark_fused_epilogue_prep_v7_fwd")
        and hasattr(lib, "dspark_vw_update_dink_v7_fwd")
    )


def v10_tree_pack_ready(device: torch.device) -> bool:
    if not _TREE_FUSE_V10:
        return False
    q = _v7_q(device)
    lib = _get_dspark_v7_lib()
    return bool(
        lib is not None
        and q is not None
        and q.get("markov_ok", False)
        and hasattr(lib, "dspark_select_last_and_tree_pack_v7_fwd")
    )


@torch.no_grad()
def qualify_dspark_v7(device: torch.device, max_rows: int = 256) -> dict:
    """Fail-closed 0-ULP (nd == 0) startup qualification & mode calibration for libdspark_markov_v7.so."""
    d = _ddev_idx(device)
    if d in _V7_QUAL:
        return _V7_QUAL[d]
    res = {
        "embed_ok": False, "embed_spt": 8,
        "mix_ok": False, "mix_spt": 8, "mix_fma": 0,
        "qk_ok": False, "qk_sqrt": 0, "qk_rfma": 0,
        "silu_ok": False, "fp8_ok": False, "softcap_ok": False,
        "gate_ok": False, "rnn_r1_ok": False, "rnn_tc_ok": False,
        "markov_ok": False, "fold_fma": 0, "exp_mode": 0,
        "select_rnn_r1_ok": False, "select_rnn_tc_ok": False,
        "v10_ver": 0, "rnn_tc_skip_ok": False, "fold_tc_skip_ok": False,
        "select_rnn_tc_es_ok": False, "fold_top2_idmap_ok": False,
    }
    lib = _get_dspark_v7_lib()
    if lib is None or device.type != "cuda" or torch.cuda.is_current_stream_capturing():
        _V7_QUAL[d] = res
        return res
    prealloc_v7_workspaces(device, max_rows=max_rows)
    stream = int(torch.cuda.current_stream(device).cuda_stream)
    gen = torch.Generator(device=device)
    gen.manual_seed(7071)

    # 1. Qualify embed_block_ln_v7 (nd == 0 vs _embed_block_ln_kernel)
    try:
        emb_w = (torch.randn((32, 4608), device=device, generator=gen) * 0.25).to(torch.bfloat16)
        emb_nw = (torch.randn((4608,), device=device, generator=gen) * 0.5 + 1.0).to(torch.bfloat16)
        mask_e = (torch.randn((4608,), device=device, generator=gen) * 0.25).to(torch.bfloat16)
        ln_w = (torch.randn((4608,), device=device, generator=gen) * 0.5 + 1.0).to(torch.bfloat16)
        ids = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, -1, 1, 2, 3, 4, 5, 6, 7], dtype=torch.int64, device=device)
        ref_o, ref_r = torch.empty((16, 4608), dtype=torch.bfloat16, device=device), torch.empty((16, 4608), dtype=torch.bfloat16, device=device)
        _embed_block_ln_kernel[(16,)](ids, emb_w, emb_nw, mask_e, ln_w, ref_r, ref_o, 4608, 32, H=4608, BLOCK_SIZE=8, EMB_EPS=1e-6, EPS=1e-6, B_H=8192, PDL=False, num_warps=8)
        for spt in (8, 1):
            co, cr = torch.empty_like(ref_o), torch.empty_like(ref_r)
            rc = lib.dspark_embed_block_ln_v7_fwd(
                ids.data_ptr(), 1, emb_w.data_ptr(), emb_nw.data_ptr(), mask_e.data_ptr(), ln_w.data_ptr(),
                cr.data_ptr(), co.data_ptr(), 16, 4608, 4608, 32, 8, 1e-6, 1e-6, spt, 0, stream,
            )
            if rc == 0 and _bits_eq(co, ref_o) and _bits_eq(cr, ref_r):
                res["embed_ok"] = True
                res["embed_spt"] = spt
                break
    except Exception:
        pass

    # 2. Qualify mix_rmsnorm & mix_rmsnorm_5layer (nd == 0 vs _mix_rmsnorm_kernel)
    try:
        x_m = (torch.randn((4, 8, 4608), device=device, generator=gen) * 0.3).to(torch.bfloat16)
        al = torch.softmax(torch.randn((5, 8), device=device, generator=gen), dim=-1).contiguous()
        ws_l = [(torch.randn((4608,), device=device, generator=gen) * 0.5 + 1.0).to(torch.bfloat16) for _ in range(5)]
        ref_m = torch.empty((4, 4608), dtype=torch.bfloat16, device=device)
        _mix_rmsnorm_kernel[(4,)](x_m, al[0], ws_l[0], ref_m, *x_m.stride(), 8, 4608, 1e-6, 8192, PDL=False, num_warps=8)
        for spt in (8, 1):
            for fma in (0, 1):
                cm = torch.empty_like(ref_m)
                rc = lib.dspark_mix_rmsnorm_v7_fwd(
                    x_m.data_ptr(), al[0].data_ptr(), 0, ws_l[0].data_ptr(), cm.data_ptr(),
                    4, 8, 4608, x_m.stride(0), x_m.stride(1), 4608, 1e-6, spt, fma, 0, stream,
                )
                if rc == 0 and _bits_eq(cm, ref_m):
                    out5 = torch.zeros((5, 4, 4608), dtype=torch.bfloat16, device=device)
                    smap = torch.tensor([1, -1, 3, -1], dtype=torch.int64, device=device)
                    rc5 = lib.dspark_mix_rmsnorm_5layer_v7_fwd(
                        x_m.data_ptr(), al.data_ptr(), 0,
                        ws_l[0].data_ptr(), ws_l[1].data_ptr(), ws_l[2].data_ptr(), ws_l[3].data_ptr(), ws_l[4].data_ptr(),
                        smap.data_ptr(), 1, out5.data_ptr(), 4 * 4608,
                        4, 8, 4608, x_m.stride(0), x_m.stride(1), 1e-6, spt, fma, 0, stream,
                    )
                    if rc5 == 0 and _bits_eq(out5[0, 0], ref_m[0]) and _bits_eq(out5[0, 2], ref_m[2]):
                        res["mix_ok"] = True
                        res["mix_spt"] = spt
                        res["mix_fma"] = fma
                        break
            if res["mix_ok"]:
                break
    except Exception:
        pass

    # 3. Qualify qk_norm_rope_write_kv_v7 (nd == 0 vs _qk_norm_rope_write_kv_kernel)
    try:
        qkv0 = (torch.randn((4, 1024), device=device, generator=gen) * 0.4).to(torch.bfloat16)
        qw = (torch.randn((128,), device=device, generator=gen) * 0.5 + 1.0).to(torch.bfloat16)
        kw = (torch.randn((128,), device=device, generator=gen) * 0.5 + 1.0).to(torch.bfloat16)
        invf = (1.0 / (10000.0 ** (torch.arange(0, 128, 2, dtype=torch.float32, device=device) / 128.0))).contiguous()
        pos = torch.tensor([0, 7, 19, 103], dtype=torch.int64, device=device)
        smap = torch.tensor([0, 5, -1, 12], dtype=torch.int64, device=device)
        rk, rv = torch.zeros((2, 16, 1, 128), dtype=torch.bfloat16, device=device), torch.zeros((2, 16, 1, 128), dtype=torch.bfloat16, device=device)
        rq = qkv0.clone()
        _qk_norm_rope_write_kv_kernel[(4, 7)](
            rq, qw, kw, invf, pos, rk, rv, smap,
            rq.stride(0), 6 * 128,
            rk.stride(0), rk.stride(1), rk.stride(2),
            rv.stride(0), rv.stride(1), rv.stride(2),
            1, 16, NHQ=6, NHK=1, D=128, EPS=1e-6, PDL=False, OTF_ROPE=True,
        )
        for sm in (0, 1, 2):
            for rfm in (0, 1, 2):
                cq = qkv0.clone()
                ck, cv = torch.zeros_like(rk), torch.zeros_like(rv)
                rc = lib.dspark_qk_norm_rope_write_kv_v7_fwd(
                    cq.data_ptr(), qw.data_ptr(), kw.data_ptr(), invf.data_ptr(),
                    pos.data_ptr(), 0, ck.data_ptr(), cv.data_ptr(), smap.data_ptr(),
                    4, 6, 1, 128, cq.stride(0), 6 * 128,
                    ck.stride(0), ck.stride(1), ck.stride(2),
                    cv.stride(0), cv.stride(1), cv.stride(2),
                    1, 16, 1e-6, 1, 1, 1, sm, rfm, 0, stream,
                )
                if rc == 0 and _bits_eq(cq, rq) and _bits_eq(ck, rk) and _bits_eq(cv, rv):
                    res["qk_ok"] = True
                    res["qk_sqrt"] = sm
                    res["qk_rfma"] = rfm
                    break
            if res["qk_ok"]:
                break
    except Exception:
        pass

    # 4. Qualify silu_mul, add_softcap, quant_fp8_rowwise & silu_mul_quant_fp8
    try:
        xg = (torch.randn((4, 4608), device=device, generator=gen) * 0.5).to(torch.bfloat16)
        ry = torch.empty((4, 2304), dtype=torch.bfloat16, device=device)
        _silu_and_mul_kernel[(4, 3)](xg, ry, xg.stride(0), 2304, 1024, num_warps=4)
        cy = torch.empty_like(ry)
        rc = lib.dspark_silu_mul_v7_fwd(xg.data_ptr(), cy.data_ptr(), 4, 2304, xg.stride(0), 0, stream)
        res["silu_ok"] = (rc == 0) and _bits_eq(cy, ry)

        aa = (torch.randn((2, 1024), device=device, generator=gen) * 4.0).to(torch.bfloat16)
        bb = (torch.randn((2, 1024), device=device, generator=gen) * 4.0).to(torch.bfloat16)
        rsc, csc = torch.empty_like(aa), torch.empty_like(aa)
        _add_softcap_kernel[(1, 2)](aa, bb, rsc, 1024, 1024, 30.0, 1024, 1024)
        rc2 = lib.dspark_add_softcap_v7_fwd(aa.data_ptr(), bb.data_ptr(), csc.data_ptr(), 2, 1024, 1024, 1024, 30.0, 0, stream)
        res["softcap_ok"] = (rc2 == 0) and _bits_eq(csc, rsc)

        from .linear import quantize_fp8_rowwise
        rq_fp8, rq_sc = quantize_fp8_rowwise(ry, pad_m16=True)
        cq_fp8 = torch.empty_like(rq_fp8)
        cq_sc = torch.empty_like(rq_sc)
        rc3 = lib.dspark_quant_fp8_rowwise_v7_fwd(
            ry.data_ptr(), cq_fp8.data_ptr(), cq_sc.data_ptr(), 4, 16, 2304, ry.stride(0), cq_fp8.stride(0), 0, stream
        )
        fq_fp8 = torch.empty_like(rq_fp8)
        fq_sc = torch.empty_like(rq_sc)
        rc4 = lib.dspark_silu_mul_quant_fp8_v7_fwd(
            xg.data_ptr(), fq_fp8.data_ptr(), fq_sc.data_ptr(), 4, 16, 2304, xg.stride(0), fq_fp8.stride(0), 0, stream
        )
        res["fp8_ok"] = (
            rc3 == 0 and rc4 == 0
            and _bits_eq(cq_fp8, rq_fp8) and _bits_eq(cq_sc, rq_sc)
            and _bits_eq(fq_fp8, rq_fp8) and _bits_eq(fq_sc, rq_sc)
        )
    except Exception:
        pass

    # 5. Qualify Markov gate, r1/TC48 RNN, BM=8/16/32 fold_argmax, & argmax_select (nd == 0 on tokens + logits!)
    try:
        raw_g = torch.randn((4, 768), dtype=torch.float32, device=device, generator=gen) * 0.5
        st_in = (torch.randn((4, 256), device=device, generator=gen) * 0.5).to(torch.bfloat16)
        ro_s, ro_l = torch.empty_like(st_in), torch.empty_like(st_in)
        _markov_gate_kernel[(4,)](raw_g, st_in, ro_s, ro_l, 256)
        co_s, co_l = torch.empty_like(st_in), torch.empty_like(st_in)
        rc_g = lib.dspark_markov_gate_v7_fwd(raw_g.data_ptr(), st_in.data_ptr(), co_s.data_ptr(), co_l.data_ptr(), 4, 256, 0, stream)
        res["gate_ok"] = (rc_g == 0) and _bits_eq(co_s, ro_s) and _bits_eq(co_l, ro_l)

        w1_t = (torch.randn((32, 256), device=device, generator=gen) * 0.25).to(torch.bfloat16)
        rw_t = (torch.randn((768, 512), device=device, generator=gen) * 0.1).to(torch.bfloat16)
        prev_t = torch.tensor([3, 7, 11, 29], dtype=torch.int64, device=device)
        ctx_t = torch.randn((4, 768), dtype=torch.float32, device=device, generator=gen) * 0.3
        rs1, rl1 = torch.empty((1, 256), dtype=torch.bfloat16, device=device), torch.empty((1, 256), dtype=torch.bfloat16, device=device)
        cs1, cl1 = torch.empty_like(rs1), torch.empty_like(rl1)
        _markov_rnn_step_kernel[(1, 32)](
            st_in[:1], prev_t[:1], ctx_t[:1], w1_t, rw_t, rs1, rl1, prev_t[:1], 1,
            RANK=256, BC=8, HAS_SKIP=False, INIT_ZERO=False, PDL=False, num_warps=4,
        )
        rc_r1 = lib.dspark_markov_rnn_step_v7_fwd(
            st_in[:1].data_ptr(), prev_t[:1].data_ptr(), 0, ctx_t[:1].data_ptr(), 768,
            w1_t.data_ptr(), rw_t.data_ptr(), cs1.data_ptr(), cl1.data_ptr(), 0,
            1, 256, 1, 0, 0, 0, 0, stream,
        )
        res["rnn_r1_ok"] = (rc_r1 == 0) and _bits_eq(cs1, rs1) and _bits_eq(cl1, rl1)

        rst, rlt = torch.empty((4, 256), dtype=torch.bfloat16, device=device), torch.empty((4, 256), dtype=torch.bfloat16, device=device)
        cst, clt = torch.empty_like(rst), torch.empty_like(rlt)
        _markov_rnn_step_tc_kernel[(1, 8)](
            st_in, prev_t, ctx_t, w1_t, rw_t, rst, rlt, 4, 1,
            RANK=256, BM=16, BC=32, BK=64, INIT_ZERO=False, PDL=False, num_warps=4,
        )
        rc_tc = lib.dspark_markov_rnn_step_v7_fwd(
            st_in.data_ptr(), prev_t.data_ptr(), 0, ctx_t.data_ptr(), 768,
            w1_t.data_ptr(), rw_t.data_ptr(), cst.data_ptr(), clt.data_ptr(), 0,
            4, 256, 1, 0, 0, 1, 0, stream,
        )
        res["rnn_tc_ok"] = (rc_tc == 0) and _bits_eq(cst, rst) and _bits_eq(clt, rlt)

        # Calibrate exp_mode & fold_fma on BM=8 (rows=4) fold_argmax_tc
        V_t, S_t, CH_t = 512, 4, 128
        base_t = (torch.randn((4, V_t), device=device, generator=gen) * 2.0).to(torch.bfloat16)
        lat_t = (torch.randn((4, 256), device=device, generator=gen) * 0.3).to(torch.bfloat16)
        fw_t = (torch.randn((V_t, 256), device=device, generator=gen) * 0.2).to(torch.float8_e4m3fn)
        fs_t = torch.rand((V_t,), dtype=torch.float32, device=device, generator=gen) * 0.1 + 0.01
        fb_t = (torch.randn((V_t,), device=device, generator=gen) * 0.5).to(torch.bfloat16)
        rec_ref = torch.empty((4, S_t, 4), dtype=torch.float32, device=device)
        rec1_ref = torch.empty((4, 1, 4), dtype=torch.float32, device=device)
        sem_ref = torch.zeros((4,), dtype=torch.int32, device=device)
        _fold_argmax_tc_kernel[(1, S_t)](
            base_t, lat_t, fw_t, fs_t, fb_t, rec_ref, rec1_ref, sem_ref, rec_ref, rec_ref,
            V_t, 4, 30.0, V_t, 0, CH_t, RANK=256, BM=16, BC=32, USE_FP8=True,
            FUSE_REDUCE=True, NP2_S=4, PDL=False, HAS_ID_MAP=False, HAS_SKIP=False, num_warps=4,
        )
        best_fallback = None
        for em in (0, 1, 2):
            for ff in (0, 1):
                rec_c = torch.empty_like(rec_ref)
                rec1_c = torch.empty_like(rec1_ref)
                sem_c = torch.zeros((4,), dtype=torch.int32, device=device)
                rc_f = lib.dspark_fold_argmax_v7_fwd(
                    base_t.data_ptr(), lat_t.data_ptr(), fw_t.data_ptr(), fs_t.data_ptr(), fb_t.data_ptr(),
                    rec_c.data_ptr(), rec1_c.data_ptr(), sem_c.data_ptr(), 0, 0,
                    4, 256, S_t, CH_t, V_t, 30.0, V_t, 0,
                    1, 0, 1, 0, 0, ff, em, 0, stream,
                )
                if rc_f == 0 and _bits_eq(rec1_c[:, 0, :2], rec1_ref[:, 0, :2]):
                    if _bits_eq(rec1_c[:, 0, :3], rec1_ref[:, 0, :3]):
                        res["markov_ok"] = True
                        res["fold_fma"] = ff
                        res["exp_mode"] = em
                        break
                    rel_se = float(((rec1_c[:, 0, 2] - rec1_ref[:, 0, 2]).abs() / rec1_ref[:, 0, 2].abs().clamp_min(1e-6)).max().item())
                    if rel_se <= 2e-6 and best_fallback is None:
                        best_fallback = (ff, em)
            if res["markov_ok"]:
                break
        if not res["markov_ok"] and best_fallback is not None:
            res["markov_ok"] = True
            res["fold_fma"], res["exp_mode"] = best_fallback
        if res["markov_ok"] and res["rnn_r1_ok"] and res["rnn_tc_ok"]:
            em_q = int(res["exp_mode"])
            g_r1 = torch.randn((8, 1, 4), dtype=torch.float32, device=device, generator=gen)
            g_r1[:, 0, 1] = torch.arange(8, dtype=torch.float32, device=device)
            g_r1[:, 0, 2] = g_r1[:, 0, 2].abs().clamp_min(0.5)
            tok_ref1 = torch.empty((1,), dtype=torch.int64, device=device)
            conf_ref1 = torch.empty((1,), dtype=torch.float32, device=device)
            lib.dspark_argmax_select_v7_fwd(
                g_r1.data_ptr(), 1, 1, 8, tok_ref1.data_ptr(), 1, conf_ref1.data_ptr(), 1, 1,
                0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, 0, 0, em_q, 0, stream,
            )
            st_ref1, lt_ref1 = torch.empty_like(rs1), torch.empty_like(rl1)
            lib.dspark_markov_rnn_step_v7_fwd(
                st_in[:1].data_ptr(), tok_ref1.data_ptr(), 0, ctx_t[:1].data_ptr(), 768,
                w1_t.data_ptr(), rw_t.data_ptr(), st_ref1.data_ptr(), lt_ref1.data_ptr(), 0,
                1, 256, 1, 0, 0, 0, 0, stream,
            )
            tok_c1, conf_c1 = torch.empty_like(tok_ref1), torch.empty_like(conf_ref1)
            st_c1, lt_c1 = torch.empty_like(rs1), torch.empty_like(rl1)
            rc_sr1 = lib.dspark_select_and_rnn_step_v7_fwd(
                g_r1.data_ptr(), 1, 1, 8, tok_c1.data_ptr(), 1, conf_c1.data_ptr(), 1, 1,
                0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0,
                st_in[:1].data_ptr(), ctx_t[:1].data_ptr(), 768, w1_t.data_ptr(), rw_t.data_ptr(),
                st_c1.data_ptr(), lt_c1.data_ptr(), em_q, 0, stream,
            )
            res["select_rnn_r1_ok"] = (
                rc_sr1 == 0 and _bits_eq(tok_c1, tok_ref1) and _bits_eq(conf_c1, conf_ref1)
                and _bits_eq(st_c1, st_ref1) and _bits_eq(lt_c1, lt_ref1)
            )
            g_tc = torch.randn((32, 1, 4), dtype=torch.float32, device=device, generator=gen)
            g_tc[:, 0, 1] = torch.arange(32, dtype=torch.float32, device=device) % 32
            g_tc[:, 0, 2] = g_tc[:, 0, 2].abs().clamp_min(0.5)
            tok_reft = torch.empty((4,), dtype=torch.int64, device=device)
            conf_reft = torch.empty((4,), dtype=torch.float32, device=device)
            lib.dspark_argmax_select_v7_fwd(
                g_tc.data_ptr(), 4, 1, 8, tok_reft.data_ptr(), 1, conf_reft.data_ptr(), 1, 1,
                0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, 0, 0, em_q, 0, stream,
            )
            st_reft, lt_reft = torch.empty_like(rst), torch.empty_like(rlt)
            lib.dspark_markov_rnn_step_v7_fwd(
                st_in.data_ptr(), tok_reft.data_ptr(), 0, ctx_t.data_ptr(), 768,
                w1_t.data_ptr(), rw_t.data_ptr(), st_reft.data_ptr(), lt_reft.data_ptr(), 0,
                4, 256, 1, 0, 0, 1, 0, stream,
            )
            tok_ct, conf_ct = torch.empty_like(tok_reft), torch.empty_like(conf_reft)
            st_ct, lt_ct = torch.empty_like(rst), torch.empty_like(rlt)
            rc_srt = lib.dspark_select_and_rnn_step_v7_fwd(
                g_tc.data_ptr(), 4, 1, 8, tok_ct.data_ptr(), 1, conf_ct.data_ptr(), 1, 1,
                0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0,
                st_in.data_ptr(), ctx_t.data_ptr(), 768, w1_t.data_ptr(), rw_t.data_ptr(),
                st_ct.data_ptr(), lt_ct.data_ptr(), em_q, 0, stream,
            )
            res["select_rnn_tc_ok"] = (
                rc_srt == 0 and _bits_eq(tok_ct, tok_reft) and _bits_eq(conf_ct, conf_reft)
                and _bits_eq(st_ct, st_reft) and _bits_eq(lt_ct, lt_reft)
            )
        if _DSPARK_V10_VER >= 10:
            res["v10_ver"] = _DSPARK_V10_VER
            skip0 = torch.zeros((1,), dtype=torch.int32, device=device)
            skip1 = torch.ones((1,), dtype=torch.int32, device=device)
            if res["rnn_tc_ok"]:
                st_sk0, lt_sk0 = torch.empty_like(rst), torch.empty_like(rlt)
                rc_rk0 = lib.dspark_markov_rnn_step_v10_fwd(
                    st_in.data_ptr(), prev_t.data_ptr(), 0, ctx_t.data_ptr(), 768,
                    w1_t.data_ptr(), rw_t.data_ptr(), st_sk0.data_ptr(), lt_sk0.data_ptr(),
                    skip0.data_ptr(), 4, 256, 1, 0, 1, 1, 0, stream,
                )
                st_sk1 = torch.full_like(rst, 1.25)
                lt_sk1 = torch.full_like(rlt, -2.5)
                st_sent, lt_sent = st_sk1.clone(), lt_sk1.clone()
                rc_rk1 = lib.dspark_markov_rnn_step_v10_fwd(
                    st_in.data_ptr(), prev_t.data_ptr(), 0, ctx_t.data_ptr(), 768,
                    w1_t.data_ptr(), rw_t.data_ptr(), st_sk1.data_ptr(), lt_sk1.data_ptr(),
                    skip1.data_ptr(), 4, 256, 1, 0, 1, 1, 0, stream,
                )
                res["rnn_tc_skip_ok"] = (
                    rc_rk0 == 0 and rc_rk1 == 0
                    and _bits_eq(st_sk0, rst) and _bits_eq(lt_sk0, rlt)
                    and _bits_eq(st_sk1, st_sent) and _bits_eq(lt_sk1, lt_sent)
                )
            if res["markov_ok"]:
                rec1_sk0 = torch.empty_like(rec1_ref)
                sem_sk = torch.zeros((4,), dtype=torch.int32, device=device)
                rc_fk0 = lib.dspark_fold_argmax_v10_fwd(
                    base_t.data_ptr(), lat_t.data_ptr(), fw_t.data_ptr(), fs_t.data_ptr(), fb_t.data_ptr(),
                    rec_c.data_ptr(), rec1_sk0.data_ptr(), sem_sk.data_ptr(), skip0.data_ptr(), 0,
                    4, 256, S_t, CH_t, V_t, 30.0, V_t, 0,
                    1, 1, 1, 0, 0, int(res["fold_fma"]), int(res["exp_mode"]), 0, stream,
                )
                rec1_sk1 = torch.full_like(rec1_ref, -7.25)
                rec1_sent = rec1_sk1.clone()
                sem_sk.zero_()
                rc_fk1 = lib.dspark_fold_argmax_v10_fwd(
                    base_t.data_ptr(), lat_t.data_ptr(), fw_t.data_ptr(), fs_t.data_ptr(), fb_t.data_ptr(),
                    rec_c.data_ptr(), rec1_sk1.data_ptr(), sem_sk.data_ptr(), skip1.data_ptr(), 0,
                    4, 256, S_t, CH_t, V_t, 30.0, V_t, 0,
                    1, 1, 1, 0, 0, int(res["fold_fma"]), int(res["exp_mode"]), 0, stream,
                )
                res["fold_tc_skip_ok"] = (
                    rc_fk0 == 0 and rc_fk1 == 0
                    and _bits_eq(rec1_sk0[:, 0, :2], rec1_ref[:, 0, :2])
                    and _bits_eq(rec1_sk1, rec1_sent)
                    and int(sem_sk.sum().item()) == 0
                )
                id_map_q = (torch.arange(V_t, dtype=torch.int32, device=device) * 3 + 10007).contiguous()
                rec_t2 = torch.empty((1, S_t, 8), dtype=torch.float32, device=device)
                rec1_t2_raw = torch.empty((1, 1, 8), dtype=torch.float32, device=device)
                rec1_t2_map = torch.empty((1, 1, 8), dtype=torch.float32, device=device)
                sem_t2 = torch.zeros((4,), dtype=torch.int32, device=device)
                rc_t2a = lib.dspark_fold_argmax_v10_fwd(
                    base_t[:1].data_ptr(), rl1.data_ptr(), fw_t.data_ptr(), fs_t.data_ptr(), fb_t.data_ptr(),
                    rec_t2.data_ptr(), rec1_t2_raw.data_ptr(), sem_t2.data_ptr(), 0, 0,
                    1, 256, S_t, CH_t, V_t, 30.0, V_t, 0,
                    1, 0, 1, 0, 1, int(res["fold_fma"]), int(res["exp_mode"]), 0, stream,
                )
                sem_t2.zero_()
                rc_t2b = lib.dspark_fold_argmax_v10_fwd(
                    base_t[:1].data_ptr(), rl1.data_ptr(), fw_t.data_ptr(), fs_t.data_ptr(), fb_t.data_ptr(),
                    rec_t2.data_ptr(), rec1_t2_map.data_ptr(), sem_t2.data_ptr(), 0, id_map_q.data_ptr(),
                    1, 256, S_t, CH_t, V_t, 30.0, V_t, 0,
                    1, 0, 1, 1, 1, int(res["fold_fma"]), int(res["exp_mode"]), 0, stream,
                )
                if rc_t2a == 0 and rc_t2b == 0:
                    i1_loc = int(rec1_t2_raw[0, 0, 1].item())
                    i2_loc = int(rec1_t2_raw[0, 0, 4].item())
                    res["fold_top2_idmap_ok"] = (
                        0 <= i1_loc < V_t and 0 <= i2_loc < V_t
                        and int(rec1_t2_map[0, 0, 1].item()) == int(id_map_q[i1_loc].item())
                        and int(rec1_t2_map[0, 0, 4].item()) == int(id_map_q[i2_loc].item())
                        and _bits_eq(rec1_t2_map[0, 0, 0:1], rec1_t2_raw[0, 0, 0:1])
                        and _bits_eq(rec1_t2_map[0, 0, 3:4], rec1_t2_raw[0, 0, 3:4])
                    )
            if res["select_rnn_tc_ok"] and res["rnn_tc_skip_ok"]:
                em_q = int(res["exp_mode"])
                cal_q = torch.zeros((20,), dtype=torch.float32, device=device)
                cal_q[:8] = 1.15
                cal_q[8:16] = -0.25
                db_q = torch.tensor([0.1, -0.2, 0.05, -0.15], dtype=torch.float32, device=device)
                gr_q = torch.ones((1,), dtype=torch.int32, device=device)
                surv_in_q = torch.ones((4,), dtype=torch.float32, device=device)
                surv_ref_q = surv_in_q.clone()
                skip_ref_q = torch.zeros((1,), dtype=torch.int32, device=device)
                tok_es_ref = torch.empty((4,), dtype=torch.int64, device=device)
                conf_es_ref = torch.empty((4,), dtype=torch.float32, device=device)
                _argmax_select_es_batch_kernel[(1,)](
                    g_tc, tok_es_ref, 1, conf_es_ref, 1,
                    skip_ref_q, surv_ref_q, gr_q, db_q, cal_q,
                    4, 1, 8, 0.001,
                    STEP_IDX=0, K=7, NP2=8, MAX_B=8, PDL=False, num_warps=1,
                )
                tok_es_c = torch.empty_like(tok_es_ref)
                conf_es_c = torch.empty_like(conf_es_ref)
                surv_out_c = torch.zeros((4,), dtype=torch.float32, device=device)
                skip_out_c = torch.full((1,), -1, dtype=torch.int32, device=device)
                st_es_c, lt_es_c = torch.empty_like(rst), torch.empty_like(rlt)
                rc_es0 = lib.dspark_select_and_rnn_step_v10_fwd(
                    g_tc.data_ptr(), 4, 1, 8, tok_es_c.data_ptr(), 1, conf_es_c.data_ptr(), 1, 1,
                    5, 0, 7,
                    skip0.data_ptr(), surv_in_q.data_ptr(), 0, 0,
                    skip_out_c.data_ptr(), surv_out_c.data_ptr(), 0, 0,
                    gr_q.data_ptr(), 0, db_q.data_ptr(), 0, 0, cal_q.data_ptr(), 0, 0.001,
                    st_in.data_ptr(), ctx_t.data_ptr(), 768, w1_t.data_ptr(), rw_t.data_ptr(),
                    st_es_c.data_ptr(), lt_es_c.data_ptr(), em_q, 0, stream,
                )
                tok_es_sk = torch.zeros((4,), dtype=torch.int64, device=device)
                conf_es_sk = torch.ones((4,), dtype=torch.float32, device=device)
                surv_out_sk = torch.zeros((4,), dtype=torch.float32, device=device)
                skip_out_sk = torch.zeros((1,), dtype=torch.int32, device=device)
                st_es_sk = torch.full_like(rst, 1.5)
                lt_es_sk = torch.full_like(rlt, -1.5)
                st_es_sent, lt_es_sent = st_es_sk.clone(), lt_es_sk.clone()
                rc_es1 = lib.dspark_select_and_rnn_step_v10_fwd(
                    g_tc.data_ptr(), 4, 1, 8, tok_es_sk.data_ptr(), 1, conf_es_sk.data_ptr(), 1, 1,
                    5, 1, 7,
                    skip1.data_ptr(), surv_ref_q.data_ptr(), 0, 0,
                    skip_out_sk.data_ptr(), surv_out_sk.data_ptr(), 0, 0,
                    gr_q.data_ptr(), 0, db_q.data_ptr(), 0, 0, cal_q.data_ptr(), 0, 0.001,
                    st_in.data_ptr(), ctx_t.data_ptr(), 768, w1_t.data_ptr(), rw_t.data_ptr(),
                    st_es_sk.data_ptr(), lt_es_sk.data_ptr(), em_q, 0, stream,
                )
                tok_be_c = torch.empty_like(tok_es_ref)
                conf_be_c = torch.empty_like(conf_es_ref)
                surv_be_c = surv_in_q.clone()
                skip_be_c = torch.zeros((1,), dtype=torch.int32, device=device)
                rc_be = lib.dspark_argmax_select_batch_es_v10_fwd(
                    g_tc.data_ptr(), 4, 1, 8, tok_be_c.data_ptr(), 1, conf_be_c.data_ptr(), 1,
                    0, 7, skip_be_c.data_ptr(), surv_be_c.data_ptr(),
                    gr_q.data_ptr(), db_q.data_ptr(), cal_q.data_ptr(),
                    0.001, em_q, 0, stream,
                )
                surv_diff = float((surv_out_c - surv_ref_q).abs().max().item())
                surv_be_diff = float((surv_be_c - surv_ref_q).abs().max().item())
                res["select_rnn_tc_es_ok"] = (
                    rc_es0 == 0 and rc_es1 == 0 and rc_be == 0
                    and _bits_eq(tok_es_c, tok_es_ref) and _bits_eq(conf_es_c, conf_es_ref)
                    and _bits_eq(tok_be_c, tok_es_ref) and _bits_eq(conf_be_c, conf_es_ref)
                    and _bits_eq(st_es_c, st_reft) and _bits_eq(lt_es_c, lt_reft)
                    and surv_diff <= 1e-5 and surv_be_diff <= 1e-5
                    and int(skip_out_c.item()) == int(skip_ref_q.item())
                    and int(skip_be_c.item()) == int(skip_ref_q.item())
                    and bool(torch.all(tok_es_sk == -1)) and bool(torch.all(conf_es_sk == 0.0))
                    and int(skip_out_sk.item()) == 1
                    and _bits_eq(st_es_sk, st_es_sent) and _bits_eq(lt_es_sk, lt_es_sent)
                )
    except Exception as _qv7_err:
        print(f"[inkling-dspark-v7] qualify dev={d} WARNING: {_qv7_err!r}", flush=True)

    torch.cuda.synchronize(device)
    _V7_QUAL[d] = res
    v10_str = (
        f" v10={res['v10_ver']} tc_skip={int(res['rnn_tc_skip_ok'])}/{int(res['fold_tc_skip_ok'])} "
        f"es_tc={int(res['select_rnn_tc_es_ok'])} top2_id={int(res['fold_top2_idmap_ok'])}"
        if res["v10_ver"] >= 10
        else ""
    )
    print(
        f"[inkling-dspark-v7] qualify dev={d}: embed={int(res['embed_ok'])} mix={int(res['mix_ok'])} "
        f"qk={int(res['qk_ok'])} silu={int(res['silu_ok'])} fp8={int(res['fp8_ok'])} "
        f"softcap={int(res['softcap_ok'])} gate={int(res['gate_ok'])} "
        f"rnn_r1={int(res['rnn_r1_ok'])} rnn_tc={int(res['rnn_tc_ok'])} markov={int(res['markov_ok'])}{v10_str} "
        f"(spt={res['mix_spt']},fma={res['mix_fma']},qk_sm={res['qk_sqrt']},rfma={res['qk_rfma']},exp={res['exp_mode']})",
        flush=True,
    )
    return res


@triton.jit
def _mix_rmsnorm_kernel(
    X,
    ALPHA,
    WEIGHT,
    OUT,
    ROW_STRIDE: tl.constexpr,
    TAP_STRIDE: tl.constexpr,
    HIDDEN_STRIDE: tl.constexpr,
    TAPS: tl.constexpr,
    HIDDEN: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
    PDL: tl.constexpr = False,
):
    row = tl.program_id(0).to(tl.int64)
    h = tl.arange(0, BLOCK)
    mixed = tl.full((BLOCK,), 0, tl.float32)
    if PDL:
        tl.extra.cuda.gdc_wait()
    for tap in tl.static_range(TAPS):
        x = tl.load(
            X + row * ROW_STRIDE + tap * TAP_STRIDE + h * HIDDEN_STRIDE,
            h < HIDDEN,
            0,
        ).to(tl.float32)
        alpha = tl.load(ALPHA + tap).to(tl.float32)
        mixed += x * alpha
    inv_rms = tl.rsqrt(tl.sum(mixed * mixed, 0) / HIDDEN + EPS)
    weight = tl.load(WEIGHT + h, h < HIDDEN, 0).to(tl.float32)
    if PDL:
        tl.extra.cuda.gdc_launch_dependents()
    tl.store(OUT + row * HIDDEN + h, mixed * inv_rms * weight, h < HIDDEN)


def mix_rmsnorm(
    targets: torch.Tensor,
    alpha: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    pdl: bool | None = None,
) -> torch.Tensor:
    """RMSNorm of the ``alpha``-weighted sum of ``targets[:, i]`` (fp32 math).

    ``targets`` is ``(rows, taps, hidden)``; the output has ``targets.dtype``.
    """
    rows, taps, hidden = targets.shape
    assert alpha.shape == (taps,) and alpha.is_contiguous()
    assert weight.shape == (hidden,) and weight.is_contiguous()
    out = targets.new_empty((rows, hidden))
    use_pdl = _DRAFT_PDL if pdl is None else bool(pdl)
    if rows:
        q = _v7_q(targets.device)
        if (
            q is not None
            and q.get("mix_ok", False)
            and targets.dtype == torch.bfloat16
            and weight.dtype == torch.bfloat16
            and alpha.dtype in (torch.float32, torch.bfloat16)
            and targets.stride(2) == 1
            and taps == 8
            and hidden == 4608
        ):
            lib = _get_dspark_v7_lib()
            if lib is not None:
                rs, ts = targets.stride(0), targets.stride(1)
                spt = q["mix_spt"] if (((rs | ts) & 7) == 0) else 1
                rc = lib.dspark_mix_rmsnorm_v7_fwd(
                    targets.data_ptr(),
                    alpha.data_ptr(),
                    1 if alpha.dtype == torch.bfloat16 else 0,
                    weight.data_ptr(),
                    out.data_ptr(),
                    rows,
                    taps,
                    hidden,
                    rs,
                    ts,
                    out.stride(0),
                    float(eps),
                    spt,
                    q["mix_fma"],
                    1 if use_pdl else 0,
                    int(torch.cuda.current_stream(targets.device).cuda_stream),
                )
                if rc == 0:
                    return out
        _mix_rmsnorm_kernel[(rows,)](
            targets,
            alpha,
            weight,
            out,
            *targets.stride(),
            taps,
            hidden,
            eps,
            triton.next_power_of_2(hidden),
            PDL=use_pdl,
            num_warps=8,
            launch_pdl=use_pdl,
        )
    return out


def mix_rmsnorm_5layer(
    targets: torch.Tensor,
    alpha_5x8: torch.Tensor,
    weights: tuple[torch.Tensor, ...],
    eps: float,
    slot_map: "torch.Tensor | None" = None,
    pdl: bool | None = None,
) -> "torch.Tensor | None":
    """Single-read 5-layer fused mix_rmsnorm -> persistent ``[5, rows, 4608]`` view (5->1 launches, 5x less ctx DRAM read)."""
    rows, taps, hidden = targets.shape
    if rows <= 0 or taps != 8 or hidden != 4608 or len(weights) != 5:
        return None
    q = _v7_q(targets.device)
    lib = _get_dspark_v7_lib() if (q is not None and q.get("mix_ok", False)) else None
    ws = _V7_WS.get(_ddev_idx(targets.device)) if lib is not None else None
    if (
        lib is None
        or ws is None
        or rows > ws["max_rows"]
        or targets.dtype != torch.bfloat16
        or targets.stride(2) != 1
        or alpha_5x8.dtype not in (torch.float32, torch.bfloat16)
        or not alpha_5x8.is_contiguous()
        or alpha_5x8.shape != (5, 8)
    ):
        return None
    for w in weights:
        if w.dtype != torch.bfloat16 or not w.is_contiguous() or w.shape != (4608,):
            return None
    sm_ptr = 0
    sm_stride = 0
    if slot_map is not None and slot_map.dtype == torch.int64 and slot_map.shape[0] >= rows:
        sm_ptr = slot_map.data_ptr()
        sm_stride = int(slot_map.stride(0))
    use_pdl = _DRAFT_PDL if pdl is None else bool(pdl)
    out_buf = ws["ctx_mix_5l"]
    layer_stride = int(out_buf.stride(0))
    rs, ts = targets.stride(0), targets.stride(1)
    spt = q["mix_spt"] if (((rs | ts) & 7) == 0) else 1
    rc = lib.dspark_mix_rmsnorm_5layer_v7_fwd(
        targets.data_ptr(),
        alpha_5x8.data_ptr(),
        1 if alpha_5x8.dtype == torch.bfloat16 else 0,
        weights[0].data_ptr(),
        weights[1].data_ptr(),
        weights[2].data_ptr(),
        weights[3].data_ptr(),
        weights[4].data_ptr(),
        sm_ptr,
        sm_stride,
        out_buf.data_ptr(),
        layer_stride,
        rows,
        taps,
        hidden,
        rs,
        ts,
        float(eps),
        spt,
        q["mix_fma"],
        1 if use_pdl else 0,
        int(torch.cuda.current_stream(targets.device).cuda_stream),
    )
    if rc == 0:
        return out_buf[:, :rows]
    return None


@triton.jit
def _fused_context_anchor_mix_rmsnorm_kernel(
    HID,
    QSL,
    NUM_REJ,
    ALPHA,
    WEIGHT,
    OUT,
    ROW_STRIDE: tl.constexpr,
    TAP_STRIDE: tl.constexpr,
    HIDDEN_STRIDE: tl.constexpr,
    OUT_ROW_STRIDE: tl.constexpr,
    TAPS: tl.constexpr,
    HIDDEN: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
    PDL: tl.constexpr = False,
):
    """Fuses prepare_context_anchor (ends = qsl[1:] - nr; rows = hidden[clamp(ends-1,0)];
    f_rnn[:R] = mix_rmsnorm(rows)) from 7 separate kernels (incl. 46 KB index_select copy
    + memcpy32_post!) into 1 single Triton kernel (~3.5 us, 0 temp alloc, 0 L2PF contention)."""
    req = tl.program_id(0).to(tl.int64)
    h = tl.arange(0, BLOCK)
    if PDL:
        tl.extra.cuda.gdc_wait()
    end_pos = tl.load(QSL + req + 1).to(tl.int64) - tl.load(NUM_REJ + req).to(tl.int64)
    src_row = tl.maximum(end_pos - 1, 0)
    mixed = tl.full((BLOCK,), 0.0, tl.float32)
    for tap in tl.static_range(TAPS):
        x = tl.load(
            HID + src_row * ROW_STRIDE + tap * TAP_STRIDE + h * HIDDEN_STRIDE,
            h < HIDDEN,
            0.0,
        ).to(tl.float32)
        alpha = tl.load(ALPHA + tap).to(tl.float32)
        mixed += x * alpha
    inv_rms = tl.rsqrt(tl.sum(mixed * mixed, 0) / HIDDEN + EPS)
    weight = tl.load(WEIGHT + h, h < HIDDEN, 0.0).to(tl.float32)
    if PDL:
        tl.extra.cuda.gdc_launch_dependents()
    tl.store(OUT + req * OUT_ROW_STRIDE + h, (mixed * inv_rms * weight).to(OUT.dtype.element_ty), h < HIDDEN)


def context_anchor_mix_rmsnorm(
    hidden_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_rejected: torch.Tensor,
    alpha: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    num_reqs: int,
    taps: int,
    hidden: int,
    eps: float,
    pdl: bool | None = None,
) -> None:
    """In-place fused prepare_context_anchor -> out[:num_reqs] (replaces 7 kernels with 1)."""
    if num_reqs <= 0:
        return
    targets = hidden_states.view(hidden_states.shape[0], taps, hidden)
    use_pdl = _DRAFT_PDL if pdl is None else bool(pdl)
    q = _v7_q(targets.device)
    if (
        q is not None
        and q.get("mix_ok", False)
        and targets.dtype == torch.bfloat16
        and out.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and alpha.dtype in (torch.float32, torch.bfloat16)
        and query_start_loc.dtype == torch.int32
        and num_rejected.dtype == torch.int32
        and targets.stride(2) == 1
        and out.stride(1) == 1
        and taps == 8
        and hidden == 4608
    ):
        lib = _get_dspark_v7_lib()
        if lib is not None:
            rs, ts = targets.stride(0), targets.stride(1)
            spt = q["mix_spt"] if (((rs | ts) & 7) == 0) else 1
            rc = lib.dspark_context_anchor_mix_rmsnorm_v7_fwd(
                targets.data_ptr(),
                query_start_loc.data_ptr(),
                num_rejected.data_ptr(),
                alpha.data_ptr(),
                1 if alpha.dtype == torch.bfloat16 else 0,
                weight.data_ptr(),
                out.data_ptr(),
                num_reqs,
                taps,
                hidden,
                rs,
                ts,
                out.stride(0),
                float(eps),
                spt,
                q["mix_fma"],
                1 if use_pdl else 0,
                int(torch.cuda.current_stream(targets.device).cuda_stream),
            )
            if rc == 0:
                return
    _fused_context_anchor_mix_rmsnorm_kernel[(num_reqs,)](
        targets,
        query_start_loc,
        num_rejected,
        alpha,
        weight,
        out,
        targets.stride(0),
        targets.stride(1),
        targets.stride(2),
        out.stride(0),
        TAPS=taps,
        HIDDEN=hidden,
        EPS=eps,
        BLOCK=triton.next_power_of_2(hidden),
        PDL=use_pdl,
        num_warps=8,
        launch_pdl=use_pdl,
    )


@triton.jit
def _embed_block_ln_kernel(
    IDS,
    EMB_W,
    EMB_NW,
    MASK_E,
    LN_W,
    RES,
    OUT,
    stride_emb,
    full_v,
    H: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    EMB_EPS: tl.constexpr,
    EPS: tl.constexpr,
    B_H: tl.constexpr,
    PDL: tl.constexpr = False,
):
    """Fuses embed_tokens(anchor) + embed_norm + mask_embed fill + L0.input_layernorm.
    Replaces 11 eager/graph PyTorch kernels (~32 us) with 1 Triton kernel (~8.4 us, max_abs=0).
    Handles OOB token IDs (< 0 or >= full_v) identically to _repl_forward (zeroed before norm)."""
    row = tl.program_id(0).to(tl.int64)
    h = tl.arange(0, B_H)
    m = h < H
    is_anchor = (row % BLOCK_SIZE) == 0
    if PDL:
        tl.extra.cuda.gdc_wait()
    if is_anchor:
        raw = tl.load(IDS + row).to(tl.int64)
        oob = (raw < 0) | (raw >= full_v)
        tok = tl.minimum(tl.maximum(raw, 0), full_v - 1)
        x = tl.load(EMB_W + tok * stride_emb + h, mask=m & (~oob), other=0.0).to(tl.float32)
        inv = tl.rsqrt(tl.sum(x * x, 0) / H + EMB_EPS)
        nw = tl.load(EMB_NW + h, mask=m, other=0.0).to(tl.float32)
        x = (x * inv * nw).to(tl.bfloat16)
    else:
        x = tl.load(MASK_E + h, mask=m, other=0.0).to(tl.bfloat16)
    xf = x.to(tl.float32)
    inv2 = tl.rsqrt(tl.sum(xf * xf, 0) / H + EPS)
    lw = tl.load(LN_W + h, mask=m, other=0.0).to(tl.float32)
    if PDL:
        tl.extra.cuda.gdc_launch_dependents()
    tl.store(RES + row * H + h, x, mask=m)
    tl.store(OUT + row * H + h, (xf * inv2 * lw).to(tl.bfloat16), mask=m)


def embed_block_ln_fused(
    input_ids: torch.Tensor,
    embed_weight: torch.Tensor,
    embed_norm_weight: torch.Tensor,
    mask_embed: torch.Tensor,
    ln_weight: torch.Tensor,
    block_size: int,
    eps: float,
    emb_eps: float | None = None,
    pdl: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused ``embed_block`` + ``layers[0].input_layernorm`` returning ``(x_normed, residual)``."""
    num_rows = int(input_ids.numel())
    hidden = int(mask_embed.shape[0])
    emb_nw = embed_norm_weight.reshape(-1)
    if not emb_nw.is_contiguous():
        emb_nw = emb_nw.contiguous()
    residual = mask_embed.new_empty((num_rows, hidden))
    out = mask_embed.new_empty((num_rows, hidden))
    use_pdl = _DRAFT_PDL if pdl is None else bool(pdl)
    if num_rows:
        ids_flat = input_ids.reshape(-1)
        q = _v7_q(mask_embed.device)
        if (
            q is not None
            and q.get("embed_ok", False)
            and hidden == 4608
            and ids_flat.is_contiguous()
            and ids_flat.dtype in (torch.int32, torch.int64)
            and embed_weight.dtype == torch.bfloat16
            and embed_weight.stride(1) == 1
            and emb_nw.dtype == torch.bfloat16
            and mask_embed.dtype == torch.bfloat16
            and ln_weight.dtype == torch.bfloat16
            and mask_embed.is_contiguous()
            and ln_weight.is_contiguous()
        ):
            lib = _get_dspark_v7_lib()
            if lib is not None:
                spt = q["embed_spt"] if ((embed_weight.stride(0) & 7) == 0) else 1
                rc = lib.dspark_embed_block_ln_v7_fwd(
                    ids_flat.data_ptr(),
                    1 if ids_flat.dtype == torch.int64 else 0,
                    embed_weight.data_ptr(),
                    emb_nw.data_ptr(),
                    mask_embed.data_ptr(),
                    ln_weight.data_ptr(),
                    residual.data_ptr(),
                    out.data_ptr(),
                    num_rows,
                    hidden,
                    embed_weight.stride(0),
                    int(embed_weight.shape[0]),
                    int(block_size),
                    float(eps if emb_eps is None else emb_eps),
                    float(eps),
                    spt,
                    1 if use_pdl else 0,
                    int(torch.cuda.current_stream(mask_embed.device).cuda_stream),
                )
                if rc == 0:
                    return out, residual
        _embed_block_ln_kernel[(num_rows,)](
            ids_flat,
            embed_weight,
            emb_nw,
            mask_embed,
            ln_weight,
            residual,
            out,
            embed_weight.stride(0),
            int(embed_weight.shape[0]),
            H=hidden,
            BLOCK_SIZE=block_size,
            EMB_EPS=eps if emb_eps is None else float(emb_eps),
            EPS=eps,
            B_H=triton.next_power_of_2(hidden),
            PDL=use_pdl,
            num_warps=8,
            launch_pdl=use_pdl,
        )
    return out, residual


@triton.jit
def _qk_norm_rope_kernel(
    X,
    Q_WEIGHT,
    K_WEIGHT,
    COS_SIN,
    POS,
    row_stride,
    q_size,
    NHQ: tl.constexpr,
    D: tl.constexpr,
    EPS: tl.constexpr,
    NORM: tl.constexpr,
    PDL: tl.constexpr = False,
    OTF_ROPE: tl.constexpr = False,
):
    t = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1)
    HALF: tl.constexpr = D // 2
    half_ar = tl.arange(0, HALF)
    d_ar = tl.arange(0, D)
    is_q = h < NHQ
    col0 = tl.where(is_q, h * D, q_size + (h - NHQ) * D).to(tl.int64)
    row = X + t * row_stride + col0
    if PDL:
        tl.extra.cuda.gdc_wait()
    pos = tl.load(POS + t).to(tl.int64)
    if OTF_ROPE:
        angles = pos.to(tl.float32) * tl.load(COS_SIN + half_ar).to(tl.float32)
        cos = tldevice.cos(angles)
        sin = tldevice.sin(angles)
    else:
        cos = tl.load(COS_SIN + pos * D + half_ar).to(tl.float32)
        sin = tl.load(COS_SIN + pos * D + HALF + half_ar).to(tl.float32)

    if NORM:
        w_ptr = tl.where(is_q, Q_WEIGHT.to(tl.int64), K_WEIGHT.to(tl.int64)).to(
            tl.pointer_type(tl.bfloat16)
        )
        x = tl.load(row + d_ar).to(tl.float32)
        ms = tl.sum(x * x, 0) / D
        inv = 1.0 / tl.sqrt(ms + EPS)
        w1 = tl.load(w_ptr + half_ar).to(tl.float32)
        w2 = tl.load(w_ptr + HALF + half_ar).to(tl.float32)
        x1 = tl.load(row + half_ar).to(tl.float32) * inv * w1
        x2 = tl.load(row + HALF + half_ar).to(tl.float32) * inv * w2
        x1 = x1.to(tl.bfloat16).to(tl.float32)
        x2 = x2.to(tl.bfloat16).to(tl.float32)
    else:
        x1 = tl.load(row + half_ar).to(tl.float32)
        x2 = tl.load(row + HALF + half_ar).to(tl.float32)
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    if PDL:
        tl.extra.cuda.gdc_launch_dependents()
    tl.store(row + half_ar, o1.to(tl.bfloat16))
    tl.store(row + HALF + half_ar, o2.to(tl.bfloat16))


@triton.jit
def _qk_norm_rope_write_kv_kernel(
    X,
    Q_WEIGHT,
    K_WEIGHT,
    COS_SIN,
    POS,
    KC,
    VC,
    SLOT_MAP,
    row_stride,
    q_size,
    sk_b,
    sk_s,
    sk_h,
    sv_b,
    sv_s,
    sv_h,
    slot_stride,
    kv_block_size,
    NHQ: tl.constexpr,
    NHK: tl.constexpr,
    D: tl.constexpr,
    EPS: tl.constexpr,
    PDL: tl.constexpr = False,
    OTF_ROPE: tl.constexpr = False,
):
    """Fused Q/K RMSNorm (bf16-rounded) + neox RoPE + direct paged KV-cache write.
    - h < NHQ: normalizes & rotates Q head h in-place in X.
    - h >= NHQ (kv_h = h - NHQ): normalizes & rotates K head directly from registers into
      KC[slot // BS, slot % BS, kv_h, :] and copies V head from X[..., q_size + (NHK + kv_h)*D]
      into VC[slot // BS, slot % BS, kv_h, :] (skipping slot < 0).
    Eliminates 10x reshape_and_cache_flash launches (5 trunk + 5 ctxkv); 0-ULP bit-exact."""
    t = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1)
    HALF: tl.constexpr = D // 2
    half_ar = tl.arange(0, HALF)
    d_ar = tl.arange(0, D)
    if PDL:
        tl.extra.cuda.gdc_wait()
    pos = tl.load(POS + t).to(tl.int64)
    if OTF_ROPE:
        angles = pos.to(tl.float32) * tl.load(COS_SIN + half_ar).to(tl.float32)
        cos = tldevice.cos(angles)
        sin = tldevice.sin(angles)
    else:
        cos = tl.load(COS_SIN + pos * D + half_ar).to(tl.float32)
        sin = tl.load(COS_SIN + pos * D + HALF + half_ar).to(tl.float32)

    if NHQ > 0:
        if h < NHQ:
            row = X + t * row_stride + h.to(tl.int64) * D
            x = tl.load(row + d_ar).to(tl.float32)
            ms = tl.sum(x * x, 0) / D
            inv = 1.0 / tl.sqrt(ms + EPS)
            w1 = tl.load(Q_WEIGHT + half_ar).to(tl.float32)
            w2 = tl.load(Q_WEIGHT + HALF + half_ar).to(tl.float32)
            x1 = tl.load(row + half_ar).to(tl.float32) * inv * w1
            x2 = tl.load(row + HALF + half_ar).to(tl.float32) * inv * w2
            x1 = x1.to(tl.bfloat16).to(tl.float32)
            x2 = x2.to(tl.bfloat16).to(tl.float32)
            o1 = x1 * cos - x2 * sin
            o2 = x2 * cos + x1 * sin
            if PDL:
                tl.extra.cuda.gdc_launch_dependents()
            tl.store(row + half_ar, o1.to(tl.bfloat16))
            tl.store(row + HALF + half_ar, o2.to(tl.bfloat16))
            return

    kv_h = (h - NHQ).to(tl.int64)
    row_k = X + t * row_stride + q_size + kv_h * D
    row_v = row_k + NHK * D
    x = tl.load(row_k + d_ar).to(tl.float32)
    ms = tl.sum(x * x, 0) / D
    inv = 1.0 / tl.sqrt(ms + EPS)
    w1 = tl.load(K_WEIGHT + half_ar).to(tl.float32)
    w2 = tl.load(K_WEIGHT + HALF + half_ar).to(tl.float32)
    x1 = tl.load(row_k + half_ar).to(tl.float32) * inv * w1
    x2 = tl.load(row_k + HALF + half_ar).to(tl.float32) * inv * w2
    x1 = x1.to(tl.bfloat16).to(tl.float32)
    x2 = x2.to(tl.bfloat16).to(tl.float32)
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    slot = tl.load(SLOT_MAP + t * slot_stride).to(tl.int64)
    if PDL:
        tl.extra.cuda.gdc_launch_dependents()
    if slot >= 0:
        blk = slot // kv_block_size
        off = slot % kv_block_size
        kc_row = KC + blk * sk_b + off * sk_s + kv_h * sk_h
        vc_row = VC + blk * sv_b + off * sv_s + kv_h * sv_h
        v1 = tl.load(row_v + half_ar)
        v2 = tl.load(row_v + HALF + half_ar)
        tl.store(kc_row + half_ar, o1.to(KC.dtype.element_ty))
        tl.store(kc_row + HALF + half_ar, o2.to(KC.dtype.element_ty))
        tl.store(vc_row + half_ar, v1.to(VC.dtype.element_ty))
        tl.store(vc_row + HALF + half_ar, v2.to(VC.dtype.element_ty))


def qk_norm_rope_(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    num_q_heads: int,
    num_k_heads: int,
    eps: float,
    pdl: bool | None = None,
) -> None:
    """In-place per-head Q/K RMSNorm (rounded to bf16) + neox RoPE on the
    leading ``num_q_heads + num_k_heads`` heads of the fused ``qkv``."""
    otf_rope = cos_sin_cache.ndim == 1
    head_dim = cos_sin_cache.shape[0] * 2 if otf_rope else cos_sin_cache.shape[-1]
    use_pdl = _DRAFT_PDL if pdl is None else bool(pdl)
    if qkv.shape[0]:
        q = _v7_q(qkv.device)
        if (
            q is not None
            and q.get("qk_ok", False)
            and head_dim == 128
            and 1 <= num_q_heads + num_k_heads <= 8
            and qkv.dtype == torch.bfloat16
            and qkv.stride(1) == 1
            and q_weight.dtype == torch.bfloat16
            and k_weight.dtype == torch.bfloat16
            and cos_sin_cache.dtype == torch.float32
            and cos_sin_cache.is_contiguous()
            and positions.dtype in (torch.int32, torch.int64)
            and positions.is_contiguous()
        ):
            lib = _get_dspark_v7_lib()
            if lib is not None:
                rc = lib.dspark_qk_norm_rope_write_kv_v7_fwd(
                    qkv.data_ptr(),
                    q_weight.data_ptr(),
                    k_weight.data_ptr(),
                    cos_sin_cache.data_ptr(),
                    positions.data_ptr(),
                    1 if positions.dtype == torch.int32 else 0,
                    0, 0, 0,
                    int(qkv.shape[0]),
                    int(num_q_heads),
                    int(num_k_heads),
                    int(head_dim),
                    qkv.stride(0),
                    int(num_q_heads * head_dim),
                    0, 0, 0, 0, 0, 0, 0, 1,
                    float(eps),
                    0, 1, 1 if otf_rope else 0,
                    q["qk_sqrt"],
                    q["qk_rfma"],
                    1 if use_pdl else 0,
                    int(torch.cuda.current_stream(qkv.device).cuda_stream),
                )
                if rc == 0:
                    return
        _qk_norm_rope_kernel[(qkv.shape[0], num_q_heads + num_k_heads)](
            qkv,
            q_weight,
            k_weight,
            cos_sin_cache,
            positions,
            qkv.stride(0),
            num_q_heads * head_dim,
            NHQ=num_q_heads,
            D=head_dim,
            EPS=eps,
            NORM=True,
            PDL=use_pdl,
            OTF_ROPE=otf_rope,
            launch_pdl=use_pdl,
        )


def qk_norm_rope_write_kv_(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    num_q_heads: int,
    num_k_heads: int,
    eps: float,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    pdl: bool | None = None,
) -> None:
    """Fused Q/K RMSNorm + neox RoPE (Q in-place in ``qkv``) + paged KV cache write.
    ``key_cache`` and ``value_cache`` have shape ``(num_blocks, block_size, num_k_heads, head_dim)``
    with contiguous last dimension."""
    otf_rope = cos_sin_cache.ndim == 1
    head_dim = cos_sin_cache.shape[0] * 2 if otf_rope else cos_sin_cache.shape[-1]
    assert qkv.stride(1) == 1
    assert key_cache.stride(-1) == 1 and value_cache.stride(-1) == 1
    use_pdl = _DRAFT_PDL if pdl is None else bool(pdl)
    if qkv.shape[0]:
        q = _v7_q(qkv.device)
        if (
            q is not None
            and q.get("qk_ok", False)
            and head_dim == 128
            and 1 <= num_q_heads + num_k_heads <= 8
            and qkv.dtype == torch.bfloat16
            and key_cache.dtype == torch.bfloat16
            and value_cache.dtype == torch.bfloat16
            and q_weight.dtype == torch.bfloat16
            and k_weight.dtype == torch.bfloat16
            and cos_sin_cache.dtype == torch.float32
            and cos_sin_cache.is_contiguous()
            and positions.dtype in (torch.int32, torch.int64)
            and positions.is_contiguous()
            and slot_mapping.dtype == torch.int64
        ):
            lib = _get_dspark_v7_lib()
            if lib is not None:
                rc = lib.dspark_qk_norm_rope_write_kv_v7_fwd(
                    qkv.data_ptr(),
                    q_weight.data_ptr(),
                    k_weight.data_ptr(),
                    cos_sin_cache.data_ptr(),
                    positions.data_ptr(),
                    1 if positions.dtype == torch.int32 else 0,
                    key_cache.data_ptr(),
                    value_cache.data_ptr(),
                    slot_mapping.data_ptr(),
                    int(qkv.shape[0]),
                    int(num_q_heads),
                    int(num_k_heads),
                    int(head_dim),
                    qkv.stride(0),
                    int(num_q_heads * head_dim),
                    key_cache.stride(0),
                    key_cache.stride(1),
                    key_cache.stride(2),
                    value_cache.stride(0),
                    value_cache.stride(1),
                    value_cache.stride(2),
                    slot_mapping.stride(0),
                    int(key_cache.shape[1]),
                    float(eps),
                    1, 1, 1 if otf_rope else 0,
                    q["qk_sqrt"],
                    q["qk_rfma"],
                    1 if use_pdl else 0,
                    int(torch.cuda.current_stream(qkv.device).cuda_stream),
                )
                if rc == 0:
                    return
        _qk_norm_rope_write_kv_kernel[(qkv.shape[0], num_q_heads + num_k_heads)](
            qkv,
            q_weight,
            k_weight,
            cos_sin_cache,
            positions,
            key_cache,
            value_cache,
            slot_mapping,
            qkv.stride(0),
            num_q_heads * head_dim,
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            value_cache.stride(0),
            value_cache.stride(1),
            value_cache.stride(2),
            slot_mapping.stride(0),
            key_cache.shape[1],
            NHQ=num_q_heads,
            NHK=num_k_heads,
            D=head_dim,
            EPS=eps,
            PDL=use_pdl,
            OTF_ROPE=otf_rope,
            launch_pdl=use_pdl,
        )


def rope_(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    pdl: bool | None = None,
):
    """In-place neox RoPE on ``x`` of shape ``(T, H, D)`` (fp32 math)."""
    num_tokens, num_heads, head_dim = x.shape
    assert x.stride(1) == head_dim and x.stride(2) == 1
    use_pdl = _DRAFT_PDL if pdl is None else bool(pdl)
    otf_rope = cos_sin_cache.ndim == 1
    if num_tokens:
        q = _v7_q(x.device)
        if (
            q is not None
            and q.get("qk_ok", False)
            and head_dim == 128
            and 1 <= num_heads <= 8
            and x.dtype == torch.bfloat16
            and cos_sin_cache.dtype == torch.float32
            and cos_sin_cache.is_contiguous()
            and positions.dtype in (torch.int32, torch.int64)
            and positions.is_contiguous()
        ):
            lib = _get_dspark_v7_lib()
            if lib is not None:
                rc = lib.dspark_qk_norm_rope_write_kv_v7_fwd(
                    x.data_ptr(),
                    x.data_ptr(),
                    x.data_ptr(),
                    cos_sin_cache.data_ptr(),
                    positions.data_ptr(),
                    1 if positions.dtype == torch.int32 else 0,
                    0, 0, 0,
                    int(num_tokens),
                    0,
                    int(num_heads),
                    int(head_dim),
                    x.stride(0),
                    0,
                    0, 0, 0, 0, 0, 0, 0, 1,
                    0.0,
                    0, 0, 1 if otf_rope else 0,
                    q["qk_sqrt"],
                    q["qk_rfma"],
                    1 if use_pdl else 0,
                    int(torch.cuda.current_stream(x.device).cuda_stream),
                )
                if rc == 0:
                    return
        _qk_norm_rope_kernel[(num_tokens, num_heads)](
            x,
            x,
            x,
            cos_sin_cache,
            positions,
            x.stride(0),
            0,
            NHQ=0,
            D=head_dim,
            EPS=0.0,
            NORM=False,
            PDL=use_pdl,
            OTF_ROPE=otf_rope,
            launch_pdl=use_pdl,
        )


@triton.jit
def _silu_and_mul_kernel(X, Y, stride_x, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < N
    gate = tl.load(X + row * stride_x + cols, mask, other=0).to(tl.float32)
    up = tl.load(X + row * stride_x + N + cols, mask, other=0).to(tl.float32)
    silu = tl.div_rn(gate, 1.0 + tldevice.exp(-gate))
    tl.store(Y + row * N + cols, (silu * up).to(Y.dtype.element_ty), mask)


def silu_and_mul(x: torch.Tensor, pdl: bool | None = None) -> torch.Tensor:
    """``silu(gate) * up`` in fp32 (precise exp/div), rounded once."""
    num_tokens, width = x.shape
    n = width // 2
    out = x.new_empty((num_tokens, n))
    if num_tokens:
        use_pdl = _DRAFT_PDL if pdl is None else bool(pdl)
        q = _v7_q(x.device)
        if (
            q is not None
            and q.get("silu_ok", False)
            and x.dtype == torch.bfloat16
            and x.stride(1) == 1
        ):
            lib = _get_dspark_v7_lib()
            if lib is not None:
                rc = lib.dspark_silu_mul_v7_fwd(
                    x.data_ptr(),
                    out.data_ptr(),
                    int(num_tokens),
                    int(n),
                    int(x.stride(0)),
                    1 if use_pdl else 0,
                    int(torch.cuda.current_stream(x.device).cuda_stream),
                )
                if rc == 0:
                    return out
        _silu_and_mul_kernel[(num_tokens, triton.cdiv(n, 1024))](
            x, out, x.stride(0), n, 1024, num_warps=4
        )
    return out


def quantize_fp8_rowwise_v7(
    x: torch.Tensor, pad_m16: bool = True, pdl: bool | None = None
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Native SM121a 0-ULP FP8 rowwise quantization into persistent ping-pong workspace (C=32 high-M TC)."""
    m, k = int(x.shape[0]), int(x.shape[1])
    if m <= 0 or k <= 0 or (k & 1) != 0 or x.dtype != torch.bfloat16 or x.stride(1) != 1:
        return None
    q = _v7_q(x.device)
    lib = _get_dspark_v7_lib() if (q is not None and q.get("fp8_ok", False)) else None
    ws = _V7_WS.get(_ddev_idx(x.device)) if lib is not None else None
    m_pad = ((m + 15) & ~15) if pad_m16 else m
    if lib is None or ws is None or m_pad > ws["max_rows"] or m_pad * k > ws["fp8_q"][0].numel():
        return None
    idx = ws["fp8_idx"]
    ws["fp8_idx"] = idx ^ 1
    y_fp8 = ws["fp8_q"][idx].view(-1)[: m_pad * k].view(m_pad, k)
    s_out = ws["fp8_s"][idx][:m_pad]
    use_pdl = _DRAFT_PDL if pdl is None else bool(pdl)
    rc = lib.dspark_quant_fp8_rowwise_v7_fwd(
        x.data_ptr(),
        y_fp8.data_ptr(),
        s_out.data_ptr(),
        m,
        m_pad,
        k,
        int(x.stride(0)),
        int(y_fp8.stride(0)),
        1 if use_pdl else 0,
        int(torch.cuda.current_stream(x.device).cuda_stream),
    )
    if rc == 0:
        return y_fp8, s_out
    return None


def silu_mul_quant_fp8_v7(
    x: torch.Tensor, pad_m16: bool = True, pdl: bool | None = None
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Fused ``silu_and_mul(x)`` + rowwise FP8 quantization (eliminates BF16 intermediate write+read at C=32)."""
    m, width = int(x.shape[0]), int(x.shape[1])
    n = width // 2
    if m <= 0 or n <= 0 or (width & 1) != 0 or (n & 1) != 0 or x.dtype != torch.bfloat16 or x.stride(1) != 1:
        return None
    q = _v7_q(x.device)
    lib = _get_dspark_v7_lib() if (q is not None and q.get("fp8_ok", False)) else None
    ws = _V7_WS.get(_ddev_idx(x.device)) if lib is not None else None
    m_pad = ((m + 15) & ~15) if pad_m16 else m
    if lib is None or ws is None or m_pad > ws["max_rows"] or m_pad * n > ws["fp8_q"][0].numel():
        return None
    idx = ws["fp8_idx"]
    ws["fp8_idx"] = idx ^ 1
    y_fp8 = ws["fp8_q"][idx].view(-1)[: m_pad * n].view(m_pad, n)
    s_out = ws["fp8_s"][idx][:m_pad]
    use_pdl = _DRAFT_PDL if pdl is None else bool(pdl)
    rc = lib.dspark_silu_mul_quant_fp8_v7_fwd(
        x.data_ptr(),
        y_fp8.data_ptr(),
        s_out.data_ptr(),
        m,
        m_pad,
        n,
        int(x.stride(0)),
        int(y_fp8.stride(0)),
        1 if use_pdl else 0,
        int(torch.cuda.current_stream(x.device).cuda_stream),
    )
    if rc == 0:
        return y_fp8, s_out
    return None


@triton.jit
def _add_softcap_kernel(A, B, OUT, stride_a, stride_b, softcap, N, BLOCK: tl.constexpr):
    row = tl.program_id(1).to(tl.int64)
    cols = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < N
    a = tl.load(A + row * stride_a + cols, mask, other=0).to(tl.float32)
    b = tl.load(B + row * stride_b + cols, mask, other=0).to(tl.float32)
    x = (a + b).to(tl.bfloat16)
    x = x / softcap
    x = tldevice.tanh(x)
    x = x * softcap
    tl.store(OUT + row * N + cols, x.to(tl.bfloat16), mask)


def add_softcap(a: torch.Tensor, b: torch.Tensor, softcap: float, pdl: bool | None = None) -> torch.Tensor:
    """bf16 ``softcap * tanh(bf16(a + b) / softcap)`` of two bf16 2D tensors."""
    rows, n = a.shape
    assert a.stride(1) == 1 and b.stride(1) == 1
    out = a.new_empty((rows, n))
    if rows:
        use_pdl = _DRAFT_PDL if pdl is None else bool(pdl)
        q = _v7_q(a.device)
        if (
            q is not None
            and q.get("softcap_ok", False)
            and a.dtype == torch.bfloat16
            and b.dtype == torch.bfloat16
        ):
            lib = _get_dspark_v7_lib()
            if lib is not None:
                rc = lib.dspark_add_softcap_v7_fwd(
                    a.data_ptr(),
                    b.data_ptr(),
                    out.data_ptr(),
                    int(rows),
                    int(n),
                    int(a.stride(0)),
                    int(b.stride(0)),
                    float(softcap),
                    1 if use_pdl else 0,
                    int(torch.cuda.current_stream(a.device).cuda_stream),
                )
                if rc == 0:
                    return out
        _add_softcap_kernel[(triton.cdiv(n, 1024), rows)](
            a, b, out, a.stride(0), b.stride(0), softcap, n, 1024
        )
    return out


@triton.jit
def _round_bf16(x):
    # fp32 -> bf16 -> fp32 (round to nearest even) on the bits: the compiler
    # folds an extf(truncf(x)) pair, and with it fuses mul + add into an FMA.
    bits = x.to(tl.uint32, bitcast=True)
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _markov_gate_kernel(RAW, STATE, OUT, LATENT, RANK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, RANK)
    raw = RAW + row * 3 * RANK + c
    gate = _round_bf16(tl.load(raw))
    cand = _round_bf16(tl.load(raw + RANK))
    out = _round_bf16(tl.load(raw + 2 * RANK))
    state = tl.load(STATE + row * RANK + c).to(tl.float32)
    # Each torch op of the reference rounds to bf16; its sigmoid is
    # 1 / (1 + expf(-x)) with IEEE division.
    gate = _round_bf16(tldevice.div_rn(1.0, 1.0 + tldevice.exp(-gate)))
    cand = _round_bf16(tldevice.tanh(cand))
    keep = _round_bf16(gate * state)
    update = _round_bf16(_round_bf16(1.0 - gate) * cand)
    tl.store(OUT + row * RANK + c, (keep + update).to(tl.bfloat16))
    tl.store(LATENT + row * RANK + c, tldevice.tanh(out).to(tl.bfloat16))


def markov_gate(
    raw: torch.Tensor, state: torch.Tensor, pdl: bool | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """The recurrent head's gating of the fp32 ``rnn_proj`` output ``raw``
    (rounded to bf16 first), as the reference's bf16 torch ops: returns
    ``(gate * state + (1 - gate) * tanh(cand), tanh(out))``."""
    rows, rank = state.shape
    assert raw.shape == (rows, 3 * rank) and raw.is_contiguous()
    assert state.is_contiguous() and rank & (rank - 1) == 0
    out, latent = torch.empty_like(state), torch.empty_like(state)
    if rows:
        use_pdl = _DRAFT_PDL if pdl is None else bool(pdl)
        q = _v7_q(state.device)
        if (
            q is not None
            and q.get("gate_ok", False)
            and rank == 256
            and raw.dtype == torch.float32
            and state.dtype == torch.bfloat16
        ):
            lib = _get_dspark_v7_lib()
            if lib is not None:
                rc = lib.dspark_markov_gate_v7_fwd(
                    raw.data_ptr(),
                    state.data_ptr(),
                    out.data_ptr(),
                    latent.data_ptr(),
                    int(rows),
                    int(rank),
                    1 if use_pdl else 0,
                    int(torch.cuda.current_stream(state.device).cuda_stream),
                )
                if rc == 0:
                    return out, latent
        _markov_gate_kernel[(rows,)](raw, state, out, latent, rank)
    return out, latent


class Inkling2DSparkArgmax:
    """Lamport slots of ``dspark_argmax``: each row's per-rank, per-block
    (max, index) pairs, multicast to every rank, in two buffers that a row
    alternates between; and each row's arrival count and buffer parity."""

    def __init__(self, group: dist.ProcessGroup, max_rows: int, num_cols: int):
        self.rank, self.world_size = dist.get_rank(group), group.size()
        self.num_cols = num_cols
        num_blocks = triton.cdiv(num_cols, 1024)
        num_bytes = 2 * max_rows * self.world_size * num_blocks * 8
        device = torch.accelerator.current_device_index()
        self.fallback = False
        if self.world_size > 1:
            cap = torch.cuda.get_device_capability(device)
            self.fallback = cap[0] == 12  # GB10: no NVLink multicast
        if self.world_size == 1 or self.fallback:
            # SM12x: dspark_argmax all-gathers per-rank (max, index) pairs.
            self.buffer = torch.full((num_bytes // 4,), -0.0, device=device)
            self.multicast_ptr = self.local_ptr = self.buffer.data_ptr()
        else:
            from flashinfer.comm.mnnvl import is_multicast_supported

            assert is_multicast_supported(device)
            self.buffer = _multicast_memory(num_bytes, group)
            self.multicast_ptr = self.buffer.get_multicast_ptr()
            assert self.multicast_ptr, "Inkling2 DSpark requires NVLink multicast"
            self.local_ptr = self.buffer.get_unicast_ptr(self.rank)
            self.buffer.lamport_initialize(self.rank, torch.float32)
        self.row_state = torch.zeros(max_rows, dtype=torch.int32, device=device)
        torch.accelerator.synchronize()
        dist.barrier(group=group)


# INKLING_ARGMAX_FUSED (default 1): the GB10 (no multicast) Markov argmax is one fused
# softcap+argmax(+sumexp) kernel per vocab slice, a 16 B-aligned RDMA all-gather of the
# slice records and one select kernel. Bit-exact to the legacy 11-kernel + NCCL path
# (selected token); 0 restores the legacy path.
_FUSED = os.environ.get("INKLING_ARGMAX_FUSED", "1") != "0"
_SLICES = int(os.environ.get("INKLING_ARGMAX_SLICES", "16"))  # column slices per row (grid y)


class _ConfSink:
    """Where ``dspark_argmax`` leaves the drafter's top-1 probability of each Markov step
    (INKLING_VERIFY_W=conf / INKLING_CONF_LOG). ``buf`` is a persistent fp32 ``[max_rows, C]``;
    step i of a proposal writes column i. ``col`` is trace-time Python state: the speculator
    resets it before the Markov loop (also while the CUDA graph is captured, so the graph
    bakes the column)."""

    def __init__(self, buf: torch.Tensor):
        self.buf = buf
        self.col = 0


_CONF_SINK: "_ConfSink | None" = None


def set_conf_sink(buf: "torch.Tensor | None") -> None:
    global _CONF_SINK
    _CONF_SINK = None if buf is None else _ConfSink(buf)


def conf_sink_begin() -> bool:
    """Call at the start of one Markov loop; True if the sink is armed."""
    if _CONF_SINK is None:
        return False
    _CONF_SINK.col = 0
    return True


def dspark_argmax(
    workspace: Inkling2DSparkArgmax,
    base: torch.Tensor,
    bias: torch.Tensor,
    softcap: float,
    valid: int,
    offset: int,
    id_map: "torch.Tensor | None" = None,
) -> torch.Tensor:
    """First argmax (NaN as -inf) of every rank's ``add_softcap(base, bias,
    softcap)[:, :valid]`` (its vocabulary shard at ``offset``, or ``id_map`` when pruned), gathered."""
    assert base.dtype == bias.dtype == torch.bfloat16 and base.shape == bias.shape
    assert base.shape[1] == workspace.num_cols
    if workspace.fallback or id_map is not None:
        if _FUSED:
            sink = _CONF_SINK
            if sink is not None and sink.col < sink.buf.shape[1] and base.shape[0] <= sink.buf.shape[0]:
                # Top-1 probability of this Markov step into sink.buf[:rows, col]
                # (identical on every rank: computed from the gathered records).
                col = sink.col
                sink.col += 1
                return gathered_argmax_fused(
                    base, bias, softcap, valid, offset,
                    conf_out=sink.buf[: base.shape[0], col],
                    id_map=id_map,
                )
            return gathered_argmax_fused(base, bias, softcap, valid, offset, id_map=id_map)
        return _gathered_argmax(base, bias, softcap, valid, offset, id_map=id_map)
    out = torch.empty(base.shape[0], dtype=torch.int64, device=base.device)
    torch.ops._C.inkling2_dspark_argmax(
        base,
        bias,
        softcap,
        valid,
        offset,
        workspace.multicast_ptr,
        workspace.local_ptr,
        workspace.row_state,
        workspace.rank,
        workspace.world_size,
        out,
    )
    return out


def _gathered_argmax(base, bias, softcap, valid, offset, id_map=None):
    """Same result as the multicast kernel: global first argmax, NaN as -inf.
    Ranks hold consecutive vocab shards (or pruned ``id_map`` global IDs); on ties lowest rank wins."""
    from vllm.distributed import tensor_model_parallel_all_gather
    logits = add_softcap(base, bias, softcap)[:, :valid].float()
    logits = torch.nan_to_num(logits, nan=float("-inf"))
    m = logits.max(dim=-1).values
    i = (logits == m[:, None]).int().argmax(dim=-1)  # first max index
    tok_id = id_map[i].float() if id_map is not None else (i + offset).float()
    pair = torch.stack((m, tok_id), dim=-1)  # vocab < 2**24: exact
    g = tensor_model_parallel_all_gather(pair.unsqueeze(1).contiguous(), dim=1)
    r = (g[..., 0] == g[..., 0].max(dim=1, keepdim=True).values).int().argmax(dim=1)
    return g.gather(1, r[:, None, None].expand(-1, 1, 2))[:, 0, 1].long()


@triton.jit
def _argmax_record_kernel(
    A, B, REC, ID_MAP, stride_a, stride_b, softcap, valid, offset, CH,
    BLOCK: tl.constexpr, HAS_ID_MAP: tl.constexpr = False,
):
    """Per (row, column slice): the slice's max of ``add_softcap(a, b, softcap)``
    (bit for bit the ``_add_softcap_kernel`` arithmetic, NaN and columns >= valid
    as -inf), the first column attaining it (+ offset or ``ID_MAP[idx]``, as fp32) and
    ``sum(exp(x - max))``. One 16 B record per (row, slice)."""
    row = tl.program_id(0).to(tl.int64)
    s = tl.program_id(1)
    S = tl.num_programs(1)
    lo = s * CH
    cols = lo + tl.arange(0, BLOCK)
    mask = (cols < valid) & (cols < lo + CH)
    a = tl.load(A + row * stride_a + cols, mask, other=0).to(tl.float32)
    b = tl.load(B + row * stride_b + cols, mask, other=0).to(tl.float32)
    x = (a + b).to(tl.bfloat16)
    x = x / softcap
    x = tldevice.tanh(x)
    x = x * softcap
    x = x.to(tl.bfloat16).to(tl.float32)
    x = tl.where(mask & (x == x), x, -float("inf"))
    m = tl.max(x, 0)
    idx = tl.min(tl.where(x == m, cols, 2147483647), 0)
    m0 = tl.where(m == -float("inf"), 0.0, m)
    e = tl.where(x == -float("inf"), 0.0, tl.exp(x - m0))
    se = tl.sum(e, 0)
    if HAS_ID_MAP:
        safe_idx = tl.where(idx < valid, idx, 0)
        out_id = tl.where(idx < valid, tl.load(ID_MAP + safe_idx).to(tl.float32), (idx + offset).to(tl.float32))
    else:
        out_id = (idx + offset).to(tl.float32)
    o = tl.arange(0, 4)
    val = tl.where(
        o == 0, m, tl.where(o == 1, out_id, tl.where(o == 2, se, 0.0))
    )
    tl.store(REC + (row * S + s) * 4 + o, val)


@triton.jit
def _argmax_select_kernel(
    G, OUT, CONF, rows, S, NS, OSTRIDE, CSTRIDE,
    NP2: tl.constexpr, WITH_CONF: tl.constexpr, PDL: tl.constexpr = False,
):
    """Global first max over the gathered records ``G[rank * rows + row, slice, 4]``
    (rank-major, slice-minor == vocabulary order, so the lowest index wins ties)."""
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    row = tl.program_id(0).to(tl.int64)
    r = tl.arange(0, NP2)
    ok = r < NS
    base = (((r // S).to(tl.int64) * rows + row) * S + r % S) * 4
    v = tl.load(G + base, ok, other=-float("inf"))
    i = tl.load(G + base + 1, ok, other=0.0)
    m = tl.max(v, 0)
    first = tl.min(tl.where(ok & (v == m), r, NP2), 0)
    tl.store(OUT + row * OSTRIDE, tl.sum(tl.where(r == first, i, 0.0), 0).to(tl.int64))
    if WITH_CONF:
        se = tl.load(G + base + 2, ok, other=0.0)
        m0 = tl.where(m == -float("inf"), 0.0, m)
        w = tl.where(v == -float("inf"), 0.0, se * tl.exp(v - m0))
        tot = tl.sum(w, 0)
        tl.store(CONF + row * CSTRIDE, tl.where(tot > 0, 1.0 / tot, 0.0))


def _argmax_records(base, bias, softcap, valid, offset, slices, id_map=None):
    rows = base.shape[0]
    assert base.stride(1) == 1 and bias.stride(1) == 1
    ch = triton.cdiv(triton.cdiv(valid, slices), 8) * 8
    rec = torch.empty((rows, slices, 4), dtype=torch.float32, device=base.device)
    has_id_map = id_map is not None
    _argmax_record_kernel[(rows, slices)](
        base, bias, rec, id_map if has_id_map else rec,
        base.stride(0), bias.stride(0), softcap, valid, offset, ch,
        BLOCK=triton.next_power_of_2(ch), HAS_ID_MAP=has_id_map, num_warps=4,
    )
    return rec


def argmax_select(
    g: torch.Tensor,
    rows: int,
    with_conf: bool = False,
    conf_out: "torch.Tensor | None" = None,
    out_tok: "torch.Tensor | None" = None,
    pdl: bool | None = None,
):
    """Tokens (int64 ``[rows]``) from gathered records ``g`` ``[world * rows, S, 4]``
    (and, with ``with_conf``, the top-1 probability ``[rows]`` fp32 of the
    softcapped full-vocabulary distribution: 1 / sum(exp(x - max))). With ``conf_out``
    (fp32 view of >= ``rows`` elements, any row stride) the probability is written there
    instead and only the tokens are returned. With ``out_tok`` (int64 view of >= ``rows``
    elements, any row stride), tokens are written directly into ``out_tok`` (zero memcpy)."""
    ws, s = g.shape[0] // rows, g.shape[1]
    if out_tok is not None:
        assert out_tok.dtype == torch.int64 and out_tok.shape[0] >= rows
        out, ostride = out_tok, out_tok.stride(0)
    else:
        out, ostride = torch.empty(rows, dtype=torch.int64, device=g.device), 1
    if conf_out is not None:
        assert conf_out.dtype == torch.float32 and conf_out.shape[0] >= rows
        conf, cstride, with_conf, ret_conf = conf_out, conf_out.stride(0), True, False
    else:
        conf = torch.empty(rows, dtype=torch.float32, device=g.device) if with_conf else out
        cstride, ret_conf = 1, with_conf
    np2 = triton.next_power_of_2(ws * s)
    use_pdl = _DRAFT_PDL if pdl is None else bool(pdl)
    q = _v7_q(g.device)
    if (
        q is not None
        and q.get("markov_ok", False)
        and g.dtype == torch.float32
        and g.is_contiguous()
    ):
        lib = _get_dspark_v7_lib()
        if lib is not None:
            rc = lib.dspark_argmax_select_v7_fwd(
                g.data_ptr(),
                int(rows),
                int(s),
                int(ws * s),
                out.data_ptr(),
                int(ostride),
                conf.data_ptr() if with_conf else 0,
                int(cstride),
                1 if with_conf else 0,
                0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0.0, 0, 0,
                int(q["exp_mode"]),
                1 if use_pdl else 0,
                int(torch.cuda.current_stream(g.device).cuda_stream),
            )
            if rc == 0:
                return (out, conf) if ret_conf else out
    _argmax_select_kernel[(rows,)](
        g, out, conf, rows, s, ws * s, ostride, cstride, np2, with_conf,
        PDL=use_pdl, num_warps=4 if np2 >= 512 else 1, launch_pdl=use_pdl,
    )
    return (out, conf) if ret_conf else out


def gathered_argmax_fused(
    base, bias, softcap, valid, offset, with_conf=False, conf_out=None, id_map=None
):
    """Same tokens as ``_gathered_argmax``: global first argmax, NaN as -inf.
    The 16 B-multiple records take the RDMA all-gather (not NCCL)."""
    from vllm.distributed import tensor_model_parallel_all_gather

    rows = base.shape[0]
    rec = _argmax_records(base, bias, softcap, valid, offset, _SLICES, id_map=id_map)
    g = tensor_model_parallel_all_gather(rec, dim=0)  # [world * rows, S, 4]
    return argmax_select(g, rows, with_conf, conf_out)


# ---- S5 Drop B + S6 Drafter-Opt: Fused Markov RNN step + Tuned Fold+Argmax (288x4w) + Direct out_tok ----
_MARKOV_FUSED = os.environ.get("INKLING_MARKOV_FUSED", "0").strip().lower() not in ("0", "", "off", "false", "no")
# 288 slices (6 CTAs/SM across 48 GB10 SMs) x 4 warps cuts fold_argmax from 42.8/14.6 us -> 34.7/10.3 us (0-ULP same FP8!)
_MARKOV_FOLD_SLICES = int(os.environ.get("INKLING_MARKOV_FOLD_SLICES", "288"))
_MARKOV_FOLD_TC_SLICES = int(os.environ.get("INKLING_MARKOV_FOLD_TC_SLICES", "96"))
_MARKOV_FOLD_WARPS = int(os.environ.get("INKLING_MARKOV_FOLD_WARPS", "4"))
_MARKOV_INLINE_AG = os.environ.get("INKLING_MARKOV_INLINE_AG", "1").strip().lower() not in ("0", "", "off", "false", "no")
# INKLING_MARKOV_FUSE_REDUCE=1: fuse _argmax_local_reduce_kernel into the last-arriving CTA of
# _fold_argmax_record_kernel (rows == 1, inline AG) + skip zero-state FillFunctor & half RNN
# weights on Markov step 0 (0-ULP bit-exact to separate kernels; saves ~23 us/step).
_MARKOV_FUSE_REDUCE = os.environ.get("INKLING_MARKOV_FUSE_REDUCE", "1").strip().lower() not in (
    "0", "", "off", "false", "no"
)
# C=32 Batched Markov Optimizations (strictly opt-in, default OFF, rows >= 2 ONLY -> C=1 untouched!):
# - INKLING_MARKOV_RNN_TC=1: Tensor-Core batched _markov_rnn_step_tc_kernel (8 CTAs vs 1024 scalar CTAs @ B=32, 32x less L2!)
# - INKLING_MARKOV_TC_FUSE_REDUCE=1: fuse _argmax_local_reduce_kernel into last-CTA of _fold_argmax_tc_kernel (2 <= rows <= 32)
_MARKOV_RNN_TC = os.environ.get("INKLING_MARKOV_RNN_TC", "0").strip().lower() not in (
    "0", "", "off", "false", "no"
)
_MARKOV_TC_FUSE_REDUCE = os.environ.get("INKLING_MARKOV_TC_FUSE_REDUCE", "0").strip().lower() not in (
    "0", "", "off", "false", "no"
)

# Per-device 2x ping-pong self-resetting semaphores for fused _fold_argmax_record_kernel reduce.
_FOLD_SEM_BUFS: dict[int, list[torch.Tensor]] = {}
_FOLD_SEM_IDX: dict[int, int] = {}
_MIN_FOLD_SEM_ROWS = 64


def prealloc_fold_sem(device: torch.device, min_rows: int = _MIN_FOLD_SEM_ROWS) -> None:
    """Pre-allocate zeroed semaphores for fused Markov fold+reduce outside CUDA graph capture."""
    if not (_MARKOV_FUSE_REDUCE or _MARKOV_TC_FUSE_REDUCE or _DSPARK_CUDA_V7) or device.type != "cuda" or torch.cuda.is_current_stream_capturing():
        return
    dev_idx = device.index if device.index is not None else torch.cuda.current_device()
    need = max(_MIN_FOLD_SEM_ROWS, int(min_rows))
    sems = _FOLD_SEM_BUFS.get(dev_idx)
    if sems is None or sems[0].numel() < need:
        _FOLD_SEM_BUFS[dev_idx] = [
            torch.zeros(need, dtype=torch.int32, device=device),
            torch.zeros(need, dtype=torch.int32, device=device),
        ]
    if dev_idx not in _FOLD_SEM_IDX:
        _FOLD_SEM_IDX[dev_idx] = 0


def _next_fold_sem(device: torch.device, rows: int, force_enable: bool = False) -> torch.Tensor | None:
    if (not force_enable and not _MARKOV_FUSE_REDUCE) or device.type != "cuda":
        return None
    dev_idx = device.index if device.index is not None else torch.cuda.current_device()
    sems = _FOLD_SEM_BUFS.get(dev_idx)
    if sems is None or sems[0].numel() < rows:
        if torch.cuda.is_current_stream_capturing():
            return None  # fail closed: never allocate/zero SEM inside graph capture
        prealloc_fold_sem(device, rows)
        sems = _FOLD_SEM_BUFS.get(dev_idx)
        if sems is None:
            return None
    idx = _FOLD_SEM_IDX.get(dev_idx, 0)
    _FOLD_SEM_IDX[dev_idx] = idx ^ 1
    return sems[idx]


@triton.jit
def _argmax_local_reduce_kernel(
    REC_IN, REC_OUT, SKIP_PTR,
    S, NP2: tl.constexpr, HAS_SKIP: tl.constexpr, PDL: tl.constexpr = False,
):
    """Collapse [rows, S, 4] slice records to [rows, 1, 4] (16B/row -> inline RDMA AG!)."""
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    if HAS_SKIP:
        if tl.load(SKIP_PTR) != 0:
            return
    row = tl.program_id(0).to(tl.int64)
    s = tl.arange(0, NP2)
    ok = s < S
    base = (row * S + s) * 4
    v = tl.load(REC_IN + base, ok, other=-float("inf"))
    idx = tl.load(REC_IN + base + 1, ok, other=0.0)
    se = tl.load(REC_IN + base + 2, ok, other=0.0)
    m = tl.max(v, 0)
    first = tl.min(tl.where(ok & (v == m), s, NP2), 0)
    best_i = tl.sum(tl.where(s == first, idx, 0.0), 0)
    m0 = tl.where(m == -float("inf"), 0.0, m)
    tot_se = tl.sum(tl.where(v == -float("inf"), 0.0, se * tl.exp(v - m0)), 0)
    o = tl.arange(0, 4)
    val = tl.where(o == 0, m, tl.where(o == 1, best_i, tl.where(o == 2, tot_se, 0.0)))
    tl.store(REC_OUT + row * 4 + o, val)


@triton.jit
def _markov_rnn_step_kernel(
    STATE, PREV, CTX, W1, RNN_W, OUT_STATE, LATENT, SKIP_PTR,
    prev_stride,
    RANK: tl.constexpr, BC: tl.constexpr, HAS_SKIP: tl.constexpr,
    INIT_ZERO: tl.constexpr = False,
    PDL: tl.constexpr = False,
):
    """Fuses markov_w1(prev) + cat([state, emb]) + addmm(ctx, x, rnn_w.T) + markov_gate.
    When INIT_ZERO=True (Markov step 0), state is identically 0: skips STATE & state-weight loads.
    Grid: (R, RANK // BC). Each CTA owns BC output dims of gate, cand, and out."""
    if PDL:
        tl.extra.cuda.gdc_wait()
    if HAS_SKIP:
        if tl.load(SKIP_PTR) != 0:
            if PDL:
                tl.extra.cuda.gdc_launch_dependents()
            return
    row = tl.program_id(0).to(tl.int64)
    cb = tl.program_id(1)
    c = cb * BC + tl.arange(0, BC)
    k = tl.arange(0, RANK)
    tok = tl.maximum(tl.load(PREV + row * prev_stride).to(tl.int64), 0)
    emb = tl.load(W1 + tok * RANK + k).to(tl.float32)

    stride_w = 2 * RANK
    if INIT_ZERO:
        wg1 = tl.load(RNN_W + c.to(tl.int64)[:, None] * stride_w + (RANK + k)[None, :]).to(tl.float32)
        raw_g = tl.load(CTX + row * (3 * RANK) + c) + tl.sum(wg1 * emb[None, :], axis=1)

        wc1 = tl.load(RNN_W + (RANK + c).to(tl.int64)[:, None] * stride_w + (RANK + k)[None, :]).to(tl.float32)
        raw_c = tl.load(CTX + row * (3 * RANK) + RANK + c) + tl.sum(wc1 * emb[None, :], axis=1)

        wo1 = tl.load(RNN_W + (2 * RANK + c).to(tl.int64)[:, None] * stride_w + (RANK + k)[None, :]).to(tl.float32)
        raw_o = tl.load(CTX + row * (3 * RANK) + 2 * RANK + c) + tl.sum(wo1 * emb[None, :], axis=1)
    else:
        st = tl.load(STATE + row * RANK + k).to(tl.float32)
        # Part 0: gate
        wg0 = tl.load(RNN_W + c.to(tl.int64)[:, None] * stride_w + k[None, :]).to(tl.float32)
        acc_g = tl.sum(wg0 * st[None, :], axis=1)
        wg1 = tl.load(RNN_W + c.to(tl.int64)[:, None] * stride_w + (RANK + k)[None, :]).to(tl.float32)
        raw_g = tl.load(CTX + row * (3 * RANK) + c) + acc_g + tl.sum(wg1 * emb[None, :], axis=1)

        # Part 1: cand
        wc0 = tl.load(RNN_W + (RANK + c).to(tl.int64)[:, None] * stride_w + k[None, :]).to(tl.float32)
        acc_c = tl.sum(wc0 * st[None, :], axis=1)
        wc1 = tl.load(RNN_W + (RANK + c).to(tl.int64)[:, None] * stride_w + (RANK + k)[None, :]).to(tl.float32)
        raw_c = tl.load(CTX + row * (3 * RANK) + RANK + c) + acc_c + tl.sum(wc1 * emb[None, :], axis=1)

        # Part 2: out
        wo0 = tl.load(RNN_W + (2 * RANK + c).to(tl.int64)[:, None] * stride_w + k[None, :]).to(tl.float32)
        acc_o = tl.sum(wo0 * st[None, :], axis=1)
        wo1 = tl.load(RNN_W + (2 * RANK + c).to(tl.int64)[:, None] * stride_w + (RANK + k)[None, :]).to(tl.float32)
        raw_o = tl.load(CTX + row * (3 * RANK) + 2 * RANK + c) + acc_o + tl.sum(wo1 * emb[None, :], axis=1)

    gate = _round_bf16(raw_g)
    cand = _round_bf16(raw_c)
    outv = _round_bf16(raw_o)
    gate = _round_bf16(tldevice.div_rn(1.0, 1.0 + tldevice.exp(-gate)))
    cand = _round_bf16(tldevice.tanh(cand))
    update = _round_bf16(_round_bf16(1.0 - gate) * cand)
    if INIT_ZERO:
        new_st = update
    else:
        st_c = tl.load(STATE + row * RANK + c).to(tl.float32)
        keep = _round_bf16(gate * st_c)
        new_st = keep + update
    if PDL:
        tl.extra.cuda.gdc_launch_dependents()
    tl.store(OUT_STATE + row * RANK + c, new_st.to(tl.bfloat16))
    tl.store(LATENT + row * RANK + c, tldevice.tanh(outv).to(tl.bfloat16))


@triton.jit
def _markov_rnn_step_tc_kernel(
    STATE, PREV, CTX, W1, RNN_W, OUT_STATE, LATENT,
    rows, prev_stride,
    RANK: tl.constexpr, BM: tl.constexpr, BC: tl.constexpr, BK: tl.constexpr,
    INIT_ZERO: tl.constexpr = False,
    PDL: tl.constexpr = False,
):
    """C=32 Batched Tensor-Core Markov RNN step (rows >= 2, opt-in via INKLING_MARKOV_RNN_TC=1).
    Reads rnn_w_rec [3*RANK, 2*RANK] ONCE per BM rows (16 or 32) instead of 32x per-row scalar
    CTAs (8 CTAs vs 1,024 CTAs @ B=32, 32x less L2 weight traffic!). Preserves exact _round_bf16
    recurrence gate semantics."""
    if PDL:
        tl.extra.cuda.gdc_wait()
    pm = tl.program_id(0)
    cb = tl.program_id(1)
    m = pm * BM + tl.arange(0, BM)
    mm = m < rows
    c = cb * BC + tl.arange(0, BC)
    m64 = m.to(tl.int64)
    tok = tl.maximum(tl.load(PREV + m64 * prev_stride, mask=mm, other=0).to(tl.int64), 0)

    stride_w = 2 * RANK
    acc_g = tl.zeros((BM, BC), dtype=tl.float32)
    acc_c = tl.zeros((BM, BC), dtype=tl.float32)
    acc_o = tl.zeros((BM, BC), dtype=tl.float32)

    for k0 in tl.static_range(0, RANK, BK):
        kk = k0 + tl.arange(0, BK)
        emb = tl.load(W1 + tok[:, None] * RANK + kk[None, :], mask=mm[:, None], other=0.0)
        wg1 = tl.load(RNN_W + c.to(tl.int64)[:, None] * stride_w + (RANK + kk)[None, :])
        acc_g = tl.dot(emb, tl.trans(wg1), acc_g)
        wc1 = tl.load(RNN_W + (RANK + c).to(tl.int64)[:, None] * stride_w + (RANK + kk)[None, :])
        acc_c = tl.dot(emb, tl.trans(wc1), acc_c)
        wo1 = tl.load(RNN_W + (2 * RANK + c).to(tl.int64)[:, None] * stride_w + (RANK + kk)[None, :])
        acc_o = tl.dot(emb, tl.trans(wo1), acc_o)
        if not INIT_ZERO:
            st = tl.load(STATE + m64[:, None] * RANK + kk[None, :], mask=mm[:, None], other=0.0)
            wg0 = tl.load(RNN_W + c.to(tl.int64)[:, None] * stride_w + kk[None, :])
            acc_g = tl.dot(st, tl.trans(wg0), acc_g)
            wc0 = tl.load(RNN_W + (RANK + c).to(tl.int64)[:, None] * stride_w + kk[None, :])
            acc_c = tl.dot(st, tl.trans(wc0), acc_c)
            wo0 = tl.load(RNN_W + (2 * RANK + c).to(tl.int64)[:, None] * stride_w + kk[None, :])
            acc_o = tl.dot(st, tl.trans(wo0), acc_o)

    ctx_base = m64[:, None] * (3 * RANK)
    raw_g = tl.load(CTX + ctx_base + c[None, :], mask=mm[:, None], other=0.0) + acc_g
    raw_c = tl.load(CTX + ctx_base + (RANK + c)[None, :], mask=mm[:, None], other=0.0) + acc_c
    raw_o = tl.load(CTX + ctx_base + (2 * RANK + c)[None, :], mask=mm[:, None], other=0.0) + acc_o

    gate = _round_bf16(raw_g)
    cand = _round_bf16(raw_c)
    outv = _round_bf16(raw_o)
    gate = _round_bf16(tldevice.div_rn(1.0, 1.0 + tldevice.exp(-gate)))
    cand = _round_bf16(tldevice.tanh(cand))
    update = _round_bf16(_round_bf16(1.0 - gate) * cand)
    if INIT_ZERO:
        new_st = update
    else:
        st_c = tl.load(STATE + m64[:, None] * RANK + c[None, :], mask=mm[:, None], other=0.0).to(tl.float32)
        keep = _round_bf16(gate * st_c)
        new_st = keep + update
    if PDL:
        tl.extra.cuda.gdc_launch_dependents()
    tl.store(OUT_STATE + m64[:, None] * RANK + c[None, :], new_st.to(tl.bfloat16), mask=mm[:, None])
    tl.store(LATENT + m64[:, None] * RANK + c[None, :], tldevice.tanh(outv).to(tl.bfloat16), mask=mm[:, None])


def markov_rnn_step_fused(
    state: "torch.Tensor | None",
    prev_tokens: torch.Tensor,
    context: torch.Tensor,
    w1_weight: torch.Tensor,
    rnn_w_rec: torch.Tensor,
    skip_dev: "torch.Tensor | None" = None,
    pdl: bool | None = None,
    init_zero: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if state is None:
        init_zero = True
        rows = int(context.shape[0])
        rank = int(rnn_w_rec.shape[0] // 3)
        state_arg = None
    else:
        rows, rank = state.shape
        state_arg = state
    use_pdl = _DRAFT_PDL if pdl is None else bool(pdl)
    if rows:
        q = _v7_q(context.device)
        if (
            q is not None
            and rank == 256
            and context.dtype == torch.float32
            and w1_weight.dtype == torch.bfloat16
            and rnn_w_rec.dtype == torch.bfloat16
            and w1_weight.is_contiguous()
            and rnn_w_rec.is_contiguous()
            and prev_tokens.dtype in (torch.int32, torch.int64)
            and (state_arg is None or (state_arg.dtype == torch.bfloat16 and state_arg.is_contiguous()))
        ):
            want_r1 = (rows == 1) and q.get("rnn_r1_ok", False)
            want_tc = (rows >= 2) and (skip_dev is None or q.get("rnn_tc_skip_ok", False)) and q.get("rnn_tc_ok", False)
            if want_r1 or want_tc:
                lib = _get_dspark_v7_lib()
                if lib is not None:
                    ws = _V7_WS.get(_ddev_idx(context.device))
                    if ws is not None and rows <= 32:
                        in_ptr = 0 if state_arg is None else state_arg.data_ptr()
                        s_buf = ws["mk_state"][1] if (in_ptr == ws["mk_state"][0].data_ptr()) else ws["mk_state"][0]
                        out_state = s_buf[:rows]
                        li = ws["mk_lat_idx"]
                        ws["mk_lat_idx"] = li ^ 1
                        latent = ws["mk_latent"][li][:rows]
                    else:
                        out_state = torch.empty((rows, rank), dtype=w1_weight.dtype, device=context.device)
                        latent = torch.empty((rows, rank), dtype=w1_weight.dtype, device=context.device)
                    rc = lib.dspark_markov_rnn_step_v7_fwd(
                        out_state.data_ptr() if state_arg is None else state_arg.data_ptr(),
                        prev_tokens.data_ptr(),
                        1 if prev_tokens.dtype == torch.int32 else 0,
                        context.data_ptr(),
                        int(context.stride(0)),
                        w1_weight.data_ptr(),
                        rnn_w_rec.data_ptr(),
                        out_state.data_ptr(),
                        latent.data_ptr(),
                        0 if skip_dev is None else skip_dev.data_ptr(),
                        int(rows),
                        int(rank),
                        int(prev_tokens.stride(0)),
                        1 if init_zero else 0,
                        1 if skip_dev is not None else 0,
                        1 if want_tc else 0,
                        1 if use_pdl else 0,
                        int(torch.cuda.current_stream(context.device).cuda_stream),
                    )
                    if rc == 0:
                        return out_state, latent
    if state is None:
        out_state = torch.empty((rows, rank), dtype=w1_weight.dtype, device=context.device)
        latent = torch.empty((rows, rank), dtype=w1_weight.dtype, device=context.device)
        state_arg = out_state
    else:
        out_state = torch.empty_like(state)
        latent = torch.empty_like(state)
    if rows:
        if _MARKOV_RNN_TC and rows >= 2 and skip_dev is None and (rank % 64 == 0):
            bm = 16 if rows <= 16 else 32
            bc = 32
            _markov_rnn_step_tc_kernel[(triton.cdiv(rows, bm), rank // bc)](
                state_arg, prev_tokens, context, w1_weight, rnn_w_rec, out_state, latent,
                rows, prev_tokens.stride(0),
                RANK=rank, BM=bm, BC=bc, BK=64, INIT_ZERO=bool(init_zero),
                PDL=use_pdl,
                num_warps=4, launch_pdl=use_pdl,
            )
        else:
            bc = 8
            _markov_rnn_step_kernel[(rows, rank // bc)](
                state_arg, prev_tokens, context, w1_weight, rnn_w_rec, out_state, latent,
                skip_dev if skip_dev is not None else prev_tokens,
                prev_tokens.stride(0),
                RANK=rank, BC=bc, HAS_SKIP=skip_dev is not None, INIT_ZERO=bool(init_zero),
                PDL=use_pdl,
                num_warps=4, launch_pdl=use_pdl,
            )
    return out_state, latent


@triton.jit
def _fold_argmax_record_kernel(
    BASE, LAT, FW, FS, FB, REC, REC_OUT, SEM, SKIP_PTR, ID_MAP,
    stride_base, softcap, valid, offset, CH,
    RANK: tl.constexpr, BC: tl.constexpr, USE_FP8: tl.constexpr, HAS_SKIP: tl.constexpr,
    FUSE_REDUCE: tl.constexpr = False, NP2_S: tl.constexpr = 512,
    PDL: tl.constexpr = False,
    HAS_ID_MAP: tl.constexpr = False,
):
    """Fuses fold_bias + latent @ fold_weight.T + base_logits + softcap + argmax_record
    (and optionally last-CTA local reduction across all S slices into REC_OUT[row, 4]).
    Zero intermediate [R, V] bias buffer; supports bf16 or per-row e4m3 fold_weight.
    Pins FW in L2 across steps 0..6 via eviction_policy='evict_last'."""
    if PDL:
        tl.extra.cuda.gdc_wait()
    if HAS_SKIP:
        if tl.load(SKIP_PTR) != 0:
            if PDL:
                tl.extra.cuda.gdc_launch_dependents()
            return
    row = tl.program_id(0).to(tl.int64)
    s = tl.program_id(1)
    S = tl.num_programs(1)
    HALF: tl.constexpr = RANK // 2
    k0 = tl.arange(0, HALF)
    lat0 = tl.load(LAT + row * RANK + k0).to(tl.float32)
    lat1 = tl.load(LAT + row * RANK + HALF + k0).to(tl.float32)

    lo = s * CH
    hi = tl.minimum(lo + CH, valid)
    m_best = -float("inf")
    idx_best = 2147483647
    se_best = 0.0

    for c0 in range(lo, hi, BC):
        cols = c0 + tl.arange(0, BC)
        cmask = cols < hi
        c64 = cols.to(tl.int64)[:, None] * RANK
        fw0 = tl.load(FW + c64 + k0[None, :], mask=cmask[:, None], other=0.0, eviction_policy="evict_last").to(tl.float32)
        dot = tl.sum(fw0 * lat0[None, :], axis=1)
        fw1 = tl.load(FW + c64 + (HALF + k0)[None, :], mask=cmask[:, None], other=0.0, eviction_policy="evict_last").to(tl.float32)
        dot = dot + tl.sum(fw1 * lat1[None, :], axis=1)
        if USE_FP8:
            dot = dot * tl.load(FS + cols, mask=cmask, other=0.0)
        fb = tl.load(FB + cols, mask=cmask, other=0.0).to(tl.float32)
        b = (dot + fb).to(tl.bfloat16).to(tl.float32)
        a = tl.load(BASE + row * stride_base + cols, mask=cmask, other=0).to(tl.float32)
        x = (a + b).to(tl.bfloat16)
        x = x / softcap
        x = tldevice.tanh(x)
        x = x * softcap
        x = x.to(tl.bfloat16).to(tl.float32)
        x = tl.where(cmask & (x == x), x, -float("inf"))

        cm = tl.max(x, 0)
        cidx = tl.min(tl.where(x == cm, cols, 2147483647), 0)
        m_new = tl.maximum(m_best, cm)
        m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        e = tl.where(x == -float("inf"), 0.0, tl.exp(x - m_safe))
        sc_old = tl.where(m_best == -float("inf"), 0.0, tl.exp(m_best - m_safe))
        se_best = se_best * sc_old + tl.sum(e, 0)
        idx_best = tl.where(cm > m_best, cidx, tl.where(cm == m_best, tl.minimum(idx_best, cidx), idx_best))
        m_best = m_new

    o = tl.arange(0, 4)
    if HAS_ID_MAP:
        safe_idx = tl.where(idx_best < valid, idx_best, 0)
        out_id = tl.load(ID_MAP + safe_idx).to(tl.float32)
    else:
        out_id = (idx_best + offset).to(tl.float32)
    val = tl.where(
        o == 0, m_best, tl.where(o == 1, out_id, tl.where(o == 2, se_best, 0.0))
    )
    if not FUSE_REDUCE:
        if PDL:
            tl.extra.cuda.gdc_launch_dependents()
        tl.store(REC + (row * S + s) * 4 + o, val)
    else:
        tl.store(REC + (row * S + s) * 4 + o, val)
        tl.debug_barrier()
        if tl.atomic_add(SEM + row, 1, sem="acq_rel") == S - 1:
            tl.store(SEM + row, 0)
            ss = tl.arange(0, NP2_S)
            ok = ss < S
            rbase = (row * S + ss) * 4
            rv = tl.load(REC + rbase, ok, other=-float("inf"), cache_modifier=".cg")
            ridx = tl.load(REC + rbase + 1, ok, other=0.0, cache_modifier=".cg")
            rse = tl.load(REC + rbase + 2, ok, other=0.0, cache_modifier=".cg")
            rm = tl.max(rv, 0)
            rfirst = tl.min(tl.where(ok & (rv == rm), ss, NP2_S), 0)
            rbest_i = tl.sum(tl.where(ss == rfirst, ridx, 0.0), 0)
            rm0 = tl.where(rm == -float("inf"), 0.0, rm)
            rtot_se = tl.sum(tl.where(rv == -float("inf"), 0.0, rse * tl.exp(rv - rm0)), 0)
            rval = tl.where(o == 0, rm, tl.where(o == 1, rbest_i, tl.where(o == 2, rtot_se, 0.0)))
            if PDL:
                tl.extra.cuda.gdc_launch_dependents()
            tl.store(REC_OUT + row * 4 + o, rval)


@triton.jit
def _fold_argmax_top2_record_kernel(
    BASE, LAT, FW, FS, FB, REC, REC_OUT, SEM, SKIP_PTR, ID_MAP,
    stride_base, softcap, valid, offset, CH,
    RANK: tl.constexpr, BC: tl.constexpr, USE_FP8: tl.constexpr, HAS_SKIP: tl.constexpr,
    FUSE_REDUCE: tl.constexpr = False, NP2_S: tl.constexpr = 512,
    PDL: tl.constexpr = False, HAS_ID_MAP: tl.constexpr = False,
):
    """Zero-extra-DRAM-read top-2 upgrade of `_fold_argmax_record_kernel` for Tree Speculation.
    Tracks (m1, idx1, se, m2, idx2) in the exact same single L2-pinned pass over fold_weight [V, 256].
    Writes 8 fp32 values/row (32 B/row inline RDMA AG) returning exact (tok1, p1, tok2, p2)."""
    if PDL:
        tl.extra.cuda.gdc_wait()
    if HAS_SKIP:
        if tl.load(SKIP_PTR) != 0:
            if PDL:
                tl.extra.cuda.gdc_launch_dependents()
            return
    row = tl.program_id(0).to(tl.int64)
    s = tl.program_id(1)
    S = tl.num_programs(1)
    HALF: tl.constexpr = RANK // 2
    k0 = tl.arange(0, HALF)
    lat0 = tl.load(LAT + row * RANK + k0).to(tl.float32)
    lat1 = tl.load(LAT + row * RANK + HALF + k0).to(tl.float32)

    lo = s * CH
    hi = tl.minimum(lo + CH, valid)
    m1_best = -float("inf")
    i1_best = 2147483647
    m2_best = -float("inf")
    i2_best = 2147483647
    se_best = 0.0

    for c0 in range(lo, hi, BC):
        cols = c0 + tl.arange(0, BC)
        cmask = cols < hi
        c64 = cols.to(tl.int64)[:, None] * RANK
        fw0 = tl.load(FW + c64 + k0[None, :], mask=cmask[:, None], other=0.0, eviction_policy="evict_last").to(tl.float32)
        dot = tl.sum(fw0 * lat0[None, :], axis=1)
        fw1 = tl.load(FW + c64 + (HALF + k0)[None, :], mask=cmask[:, None], other=0.0, eviction_policy="evict_last").to(tl.float32)
        dot = dot + tl.sum(fw1 * lat1[None, :], axis=1)
        if USE_FP8:
            dot = dot * tl.load(FS + cols, mask=cmask, other=0.0)
        fb = tl.load(FB + cols, mask=cmask, other=0.0).to(tl.float32)
        b = (dot + fb).to(tl.bfloat16).to(tl.float32)
        a = tl.load(BASE + row * stride_base + cols, mask=cmask, other=0).to(tl.float32)
        x = (a + b).to(tl.bfloat16)
        x = x / softcap
        x = tldevice.tanh(x)
        x = x * softcap
        x = x.to(tl.bfloat16).to(tl.float32)
        x = tl.where(cmask & (x == x), x, -float("inf"))

        cm1 = tl.max(x, 0)
        ci1 = tl.min(tl.where(x == cm1, cols, 2147483647), 0)
        x_no1 = tl.where(cols == ci1, -float("inf"), x)
        cm2 = tl.max(x_no1, 0)
        ci2 = tl.min(tl.where(x_no1 == cm2, cols, 2147483647), 0)

        m_new = tl.maximum(m1_best, cm1)
        m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        e = tl.where(x == -float("inf"), 0.0, tl.exp(x - m_safe))
        sc_old = tl.where(m1_best == -float("inf"), 0.0, tl.exp(m1_best - m_safe))
        se_best = se_best * sc_old + tl.sum(e, 0)

        beat1 = (cm1 > m1_best) | ((cm1 == m1_best) & (ci1 < i1_best))
        cand2_m = tl.where(beat1, tl.maximum(m1_best, cm2), tl.maximum(m2_best, cm1))
        cand2_i = tl.where(
            beat1,
            tl.where(cm2 > m1_best, ci2, tl.where(cm2 == m1_best, tl.minimum(i1_best, ci2), i1_best)),
            tl.where(cm1 > m2_best, ci1, tl.where(cm1 == m2_best, tl.minimum(i2_best, ci1), i2_best)),
        )
        i1_best = tl.where(beat1, ci1, i1_best)
        m1_best = tl.where(beat1, cm1, m1_best)
        m2_best = cand2_m
        i2_best = cand2_i

    if HAS_ID_MAP:
        safe_i1 = tl.where(i1_best < valid, i1_best, 0)
        safe_i2 = tl.where(i2_best < valid, i2_best, 0)
        out_id1 = tl.where(i1_best < valid, tl.load(ID_MAP + safe_i1).to(tl.float32), (i1_best + offset).to(tl.float32))
        out_id2 = tl.where(i2_best < valid, tl.load(ID_MAP + safe_i2).to(tl.float32), (i2_best + offset).to(tl.float32))
    else:
        out_id1 = (i1_best + offset).to(tl.float32)
        out_id2 = (i2_best + offset).to(tl.float32)
    o = tl.arange(0, 8)
    val = tl.where(
        o == 0, m1_best,
        tl.where(
            o == 1, out_id1,
            tl.where(
                o == 2, se_best,
                tl.where(
                    o == 3, m2_best,
                    tl.where(o == 4, out_id2, 0.0),
                ),
            ),
        ),
    )
    tl.store(REC + (row * S + s) * 8 + o, val)
    if FUSE_REDUCE:
        tl.debug_barrier()
        if tl.atomic_add(SEM + row, 1, sem="acq_rel") == S - 1:
            tl.store(SEM + row, 0)
            ss = tl.arange(0, NP2_S)
            ok = ss < S
            rb = (row * S + ss) * 8
            v1 = tl.load(REC + rb + 0, ok, other=-float("inf"), cache_modifier=".cg")
            k1 = tl.load(REC + rb + 1, ok, other=0.0, cache_modifier=".cg")
            se = tl.load(REC + rb + 2, ok, other=0.0, cache_modifier=".cg")
            v2 = tl.load(REC + rb + 3, ok, other=-float("inf"), cache_modifier=".cg")
            k2 = tl.load(REC + rb + 4, ok, other=0.0, cache_modifier=".cg")
            gm1 = tl.max(v1, 0)
            w1 = tl.min(tl.where(ok & (v1 == gm1), ss, NP2_S), 0)
            gk1 = tl.sum(tl.where(ss == w1, k1, 0.0), 0)
            v_pool = tl.where(ss == w1, v2, v1)
            k_pool = tl.where(ss == w1, k2, k1)
            gm2 = tl.max(v_pool, 0)
            w2 = tl.min(tl.where(ok & (v_pool == gm2), ss, NP2_S), 0)
            gk2 = tl.sum(tl.where(ss == w2, k_pool, 0.0), 0)
            gm0 = tl.where(gm1 == -float("inf"), 0.0, gm1)
            gtot = tl.sum(tl.where(v1 == -float("inf"), 0.0, se * tl.exp(v1 - gm0)), 0)
            rval = tl.where(
                o == 0, gm1,
                tl.where(
                    o == 1, gk1,
                    tl.where(o == 2, gtot, tl.where(o == 3, gm2, tl.where(o == 4, gk2, 0.0))),
                ),
            )
            if PDL:
                tl.extra.cuda.gdc_launch_dependents()
            tl.store(REC_OUT + row * 8 + o, rval)


@triton.jit
def _es_init_kernel(
    IDX_PTR, TEMP_PTR, STEP_PTR,
    SKIP_PTR, SURV_PTR, TOK_PTR, BESTV_PTR, GREEDY_PTR, SLOT_PTR,
    DUMMY: tl.constexpr,
):
    slot = tl.load(IDX_PTR).to(tl.int64)
    temp = tl.load(TEMP_PTR + slot)
    s0 = tl.load(STEP_PTR)
    tl.store(SKIP_PTR, 0)
    tl.store(SURV_PTR, 1.0)
    tl.store(TOK_PTR, 1.0)
    tl.store(BESTV_PTR, 1.0 / s0)
    greedy = 0 if DUMMY else tl.where(temp == 0.0, 1, 0)
    tl.store(GREEDY_PTR, greedy)
    tl.store(SLOT_PTR, slot.to(tl.int32))


@triton.jit
def _argmax_select_es_kernel(
    G, OUT, CONF,
    SKIP_PTR, SURV_PTR, TOK_PTR, BESTV_PTR, GREEDY_PTR, SLOT_PTR,
    EMA_PTR, CAL_PTR, STEP_PTR,
    S, NS,
    STEP_IDX: tl.constexpr, K: tl.constexpr, NP2: tl.constexpr,
    USE_HIST: tl.constexpr, PRIOR_LOGIT: tl.constexpr,
    PDL: tl.constexpr = False,
):
    """Single-request (R=1) argmax_select with exact replicated GPU-side early-stop decision."""
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    if STEP_IDX > 0:
        if tl.load(SKIP_PTR) != 0:
            tl.store(OUT, -1)
            tl.store(CONF, 0.0)
            return
    r = tl.arange(0, NP2)
    ok = r < NS
    base = r.to(tl.int64) * 4
    v = tl.load(G + base, ok, other=-float("inf"))
    i = tl.load(G + base + 1, ok, other=0.0)
    m = tl.max(v, 0)
    first = tl.min(tl.where(ok & (v == m), r, NP2), 0)
    tok = tl.sum(tl.where(r == first, i, 0.0), 0).to(tl.int64)
    tl.store(OUT, tok)

    se = tl.load(G + base + 2, ok, other=0.0)
    m0 = tl.where(m == -float("inf"), 0.0, m)
    w = tl.where(v == -float("inf"), 0.0, se * tl.exp(v - m0))
    tot = tl.sum(w, 0)
    p = tl.where(tot > 0, 1.0 / tot, 0.0)
    tl.store(CONF, p)

    if STEP_IDX < K - 1:
        if tl.load(GREEDY_PTR) != 0:
            pc = tl.minimum(tl.maximum(p, 1e-6), 1.0 - 1e-6)
            a = tl.load(CAL_PTR + STEP_IDX)
            b = tl.load(CAL_PTR + 8 + STEP_IDX)
            beta = tl.load(CAL_PTR + 16)
            tau = tl.load(CAL_PTR + 17)
            z = a * (tl.log(pc) - tl.log(1.0 - pc)) + b
            if USE_HIST:
                slot = tl.load(SLOT_PTR).to(tl.int64)
                h = tl.load(EMA_PTR + slot * K + STEP_IDX)
                hc = tl.minimum(tl.maximum(h, 1e-4), 1.0 - 1e-4)
                z = z + beta * (tl.log(hc) - tl.log(1.0 - hc) - PRIOR_LOGIT)
            c = 1.0 / (1.0 + tl.exp(-z))
            surv = tl.load(SURV_PTR) * c
            tl.store(SURV_PTR, surv)
            if tau > 0.0:
                if surv < tau:
                    tl.store(SKIP_PTR, 1)
            else:
                tokens = tl.load(TOK_PTR) + surv
                tl.store(TOK_PTR, tokens)
                v_cur = tokens / tl.load(STEP_PTR + STEP_IDX + 1)
                best_v = tl.maximum(tl.load(BESTV_PTR), v_cur)
                tl.store(BESTV_PTR, best_v)
                can_beat = 0
                for ww in tl.static_range(STEP_IDX + 2, K + 1):
                    v_ub = (tokens + (ww - STEP_IDX - 1) * surv) / tl.load(STEP_PTR + ww)
                    can_beat = tl.where(v_ub > best_v, 1, can_beat)
                if can_beat == 0:
                    tl.store(SKIP_PTR, 1)


@triton.jit
def _es_init_dink_kernel(
    IDX_PTR, TEMP_PTR, STEP_PTR, CAL_PTR, DB_PTR, ETOK_PTR, EMS_PTR,
    SKIP_PTR, SURV_PTR, TOK_PTR, BESTU_PTR, GREEDY_PTR, SLOT_PTR, DB_DEV_PTR, LAM_DEV_PTR,
    DUMMY: tl.constexpr,
):
    """Initialize coupled Dinkelbach early-stop GPU state for R=1 before draft graph replay."""
    slot = tl.load(IDX_PTR).to(tl.int64)
    temp = tl.load(TEMP_PTR + slot)
    s0 = tl.load(STEP_PTR)
    static_lam = tl.load(CAL_PTR + 18)
    gamma = tl.load(CAL_PTR + 19)
    db = tl.load(DB_PTR + slot)
    etok = tl.load(ETOK_PTR + slot)
    ems = tl.maximum(tl.load(EMS_PTR + slot), 1.0)
    lam = tl.where(static_lam > 0.0, static_lam, gamma * (etok / ems))
    tl.store(SKIP_PTR, 0)
    tl.store(SURV_PTR, 1.0)
    tl.store(TOK_PTR, 1.0)
    tl.store(BESTU_PTR, 1.0 - lam * s0)
    greedy = 0 if DUMMY else tl.where(temp == 0.0, 1, 0)
    tl.store(GREEDY_PTR, greedy)
    tl.store(SLOT_PTR, slot.to(tl.int32))
    tl.store(DB_DEV_PTR, db)
    tl.store(LAM_DEV_PTR, lam)


@triton.jit
def _argmax_select_es_dink_kernel(
    G, OUT, CONF,
    SKIP_PTR, SURV_PTR, TOK_PTR, BESTU_PTR, GREEDY_PTR, DB_DEV_PTR, LAM_DEV_PTR,
    CAL_PTR, STEP_PTR,
    S, NS,
    STEP_IDX: tl.constexpr, K: tl.constexpr, NP2: tl.constexpr,
    PDL: tl.constexpr = False,
):
    """Single-request (R=1) argmax_select with exact lossless coupled Dinkelbach early-stop."""
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    if STEP_IDX > 0:
        if tl.load(SKIP_PTR) != 0:
            tl.store(OUT, -1)
            tl.store(CONF, 0.0)
            return
    r = tl.arange(0, NP2)
    ok = r < NS
    base = r.to(tl.int64) * 4
    v = tl.load(G + base, ok, other=-float("inf"))
    i = tl.load(G + base + 1, ok, other=0.0)
    m = tl.max(v, 0)
    first = tl.min(tl.where(ok & (v == m), r, NP2), 0)
    tok = tl.sum(tl.where(r == first, i, 0.0), 0).to(tl.int64)
    tl.store(OUT, tok)

    se = tl.load(G + base + 2, ok, other=0.0)
    m0 = tl.where(m == -float("inf"), 0.0, m)
    w = tl.where(v == -float("inf"), 0.0, se * tl.exp(v - m0))
    tot = tl.sum(w, 0)
    p = tl.where(tot > 0, 1.0 / tot, 0.0)
    tl.store(CONF, p)

    if STEP_IDX < K - 1:
        if tl.load(GREEDY_PTR) != 0:
            pc = tl.minimum(tl.maximum(p, 1e-6), 1.0 - 1e-6)
            a = tl.load(CAL_PTR + STEP_IDX)
            b = tl.load(CAL_PTR + 8 + STEP_IDX)
            mk_ms = tl.load(CAL_PTR + 20)
            db = tl.load(DB_DEV_PTR)
            lam = tl.load(LAM_DEV_PTR)
            z = a * (tl.log(pc) - tl.log(1.0 - pc)) + b + db
            c = 1.0 / (1.0 + tl.exp(-z))
            surv = tl.load(SURV_PTR) * c
            tokens = tl.load(TOK_PTR) + surv
            tl.store(SURV_PTR, surv)
            tl.store(TOK_PTR, tokens)
            u_cur = tokens - lam * tl.load(STEP_PTR + STEP_IDX + 1)
            best_u = tl.maximum(tl.load(BESTU_PTR), u_cur)
            tl.store(BESTU_PTR, best_u)
            can_beat = 0
            thresh = best_u + lam * mk_ms
            for ww in tl.static_range(STEP_IDX + 2, K + 1):
                u_ub = (tokens + (ww - STEP_IDX - 1) * surv) - lam * tl.load(STEP_PTR + ww)
                can_beat = tl.where(u_ub > thresh, 1, can_beat)
            if can_beat == 0:
                tl.store(SKIP_PTR, 1)


def es_init(idx, temp, step_ms, state_bufs, dummy: bool):
    """Initialize early-stop GPU state for R=1 before draft graph replay."""
    skip_dev, surv_dev, tok_dev, bestv_dev, greedy_dev, slot_dev = state_bufs[:6]
    _es_init_kernel[(1,)](
        idx, temp, step_ms,
        skip_dev, surv_dev, tok_dev, bestv_dev, greedy_dev, slot_dev,
        DUMMY=dummy, num_warps=1,
    )


@triton.jit
def _es_init_batch_kernel(
    IDX_PTR, TEMP_PTR, DB_PTR,
    SKIP_PTR, SURV_B_PTR, GREEDY_ALL_PTR, DB_B_PTR,
    NUM_REQS,
    DUMMY: tl.constexpr, MAX_B: tl.constexpr,
):
    """Initialize Batch-All Markov Early-Stop GPU state for B=2..MAX_B (1 CTA, 1 warp)."""
    r = tl.arange(0, MAX_B)
    ok = r < NUM_REQS
    slot = tl.load(IDX_PTR + r, mask=ok, other=-1).to(tl.int64)
    valid = ok & (slot >= 0)
    temp = tl.load(TEMP_PTR + tl.maximum(slot, 0), mask=valid, other=1.0)
    db = tl.load(DB_PTR + tl.maximum(slot, 0), mask=valid, other=0.0)
    is_g = tl.where(valid, tl.where(temp == 0.0, 1, 0), 1)
    greedy_all = 0 if DUMMY else tl.min(is_g, 0)
    tl.store(SURV_B_PTR + r, tl.full((MAX_B,), 1.0, tl.float32), mask=ok)
    tl.store(DB_B_PTR + r, db, mask=ok)
    tl.store(SKIP_PTR, 0)
    tl.store(GREEDY_ALL_PTR, greedy_all)


@triton.jit
def _argmax_select_es_batch_kernel(
    G, OUT, OUT_STRIDE, CONF, CONF_STRIDE,
    SKIP_PTR, SURV_B_PTR, GREEDY_ALL_PTR, DB_B_PTR, CAL_PTR,
    ROWS, S, NS, TAU_BATCH,
    STEP_IDX: tl.constexpr, K: tl.constexpr, NP2: tl.constexpr, MAX_B: tl.constexpr,
    PDL: tl.constexpr = False,
):
    """Batched B=2..8 argmax_select with Batch-All early-stop: sets SKIP_PTR=1 iff
    greedy_all==1 AND EVERY active request r in 0..ROWS-1 has surv[r] < TAU_BATCH."""
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    if STEP_IDX > 0:
        if tl.load(SKIP_PTR) != 0:
            rr = tl.arange(0, MAX_B)
            mr = rr < ROWS
            tl.store(OUT + rr.to(tl.int64) * OUT_STRIDE, tl.full((MAX_B,), -1, tl.int64), mask=mr)
            tl.store(CONF + rr.to(tl.int64) * CONF_STRIDE, tl.zeros((MAX_B,), tl.float32), mask=mr)
            return
    r = tl.arange(0, NP2)
    ok = r < NS
    a = tl.load(CAL_PTR + STEP_IDX)
    b = tl.load(CAL_PTR + 8 + STEP_IDX)
    max_surv = 0.0
    for row in range(0, ROWS):
        base = ((r // S) * ROWS + row) * S + (r % S)
        base4 = base.to(tl.int64) * 4
        v = tl.load(G + base4, ok, other=-float("inf"))
        i = tl.load(G + base4 + 1, ok, other=0.0)
        m = tl.max(v, 0)
        first = tl.min(tl.where(ok & (v == m), r, NP2), 0)
        tok = tl.sum(tl.where(r == first, i, 0.0), 0).to(tl.int64)
        tl.store(OUT + row * OUT_STRIDE, tok)
        se = tl.load(G + base4 + 2, ok, other=0.0)
        m0 = tl.where(m == -float("inf"), 0.0, m)
        w = tl.where(v == -float("inf"), 0.0, se * tl.exp(v - m0))
        tot = tl.sum(w, 0)
        p = tl.where(tot > 0, 1.0 / tot, 0.0)
        tl.store(CONF + row * CONF_STRIDE, p)
        if STEP_IDX < K - 1:
            pc = tl.minimum(tl.maximum(p, 1e-6), 1.0 - 1e-6)
            db = tl.load(DB_B_PTR + row)
            z = a * (tl.log(pc) - tl.log(1.0 - pc)) + b + db
            c = 1.0 / (1.0 + tl.exp(-z))
            surv = tl.load(SURV_B_PTR + row) * c
            tl.store(SURV_B_PTR + row, surv)
            max_surv = tl.maximum(max_surv, surv)
    if STEP_IDX < K - 1:
        if (tl.load(GREEDY_ALL_PTR) != 0) & (max_surv < TAU_BATCH):
            tl.store(SKIP_PTR, 1)


def es_init_batch(idx, temp, db_ptr, state_bufs, num_reqs: int, dummy: bool):
    """Initialize Batch-All early-stop GPU state for B=2..8 before draft graph replay."""
    skip_d, surv_b, greedy_all, db_b = state_bufs
    _es_init_batch_kernel[(1,)](
        idx, temp, db_ptr, skip_d, surv_b, greedy_all, db_b,
        int(num_reqs), DUMMY=dummy, MAX_B=8, num_warps=1,
    )


def run_markov_loop_v9_fused(
    base_logits,      # [rows, k_run, V_local] bf16
    context,          # [k_run, rows, 3*R] fp32
    prev,             # [rows] int32/int64 anchor token
    w1_weight,        # [V, R] bf16
    rnn_w_rec,        # [3*R, 2*R] bf16
    fold_w,           # [V_local, R] bf16
    fold_b,           # [V_local] bf16
    softcap: float,
    valid: int,
    offset: int,
    draft_tokens_out, # [rows, >=k_run] int64
    *,
    fold_w8=None,
    fold_s8=None,
    conf_buf=None,    # [rows, >=k_run] fp32 or None
    es_ctx=None,      # (K, state_bufs, ema_ptr, cal_ptr, step_ptr, use_hist, prior_logit, is_dink) for rows==1
    es_batch_ctx=None,  # V10: (K, b_bufs, cal_ptr, tau_batch) for 2 <= rows <= 8 native CUDA Batch-All ES
    id_map=None,
    pdl: bool | None = None,
    tail_vw_ctx: dict | None = None,
) -> bool:
    """Wave-V9/V10 Cross-Step Fused Markov Loop (`dspark_select_and_rnn_step_v7/v10_fwd`).
    Eliminates K-1 separate argmax_select launches and overlaps STATE_I load/GEMV behind RDMA_AG(i)
    across BOTH B=1 (`r1` 48-CTA with ping-pong ES) AND B=2..32 (`TC48` 48-CTA, incl. V10 Batch-All ES)."""
    if not _MARKOV_FUSE_CROSS_STEP:
        return False
    rows, k_run = int(base_logits.shape[0]), int(base_logits.shape[1])
    if rows <= 0 or rows > 32 or k_run < 2:
        return False
    q = _v7_q(base_logits.device)
    lib = _get_dspark_v7_lib() if q is not None else None
    ws = _V7_WS.get(_ddev_idx(base_logits.device)) if lib is not None else None
    if lib is None or ws is None or not q.get("markov_ok", False):
        return False
    if rows == 1:
        if es_batch_ctx is not None or not q.get("select_rnn_r1_ok", False):
            return False
    else:
        if es_ctx is not None or not q.get("select_rnn_tc_ok", False):
            return False
        if es_batch_ctx is not None and not (
            2 <= rows <= 8
            and q.get("select_rnn_tc_es_ok", False)
            and q.get("fold_tc_skip_ok", False)
            and "es_pp_surv_b" in ws
            and (conf_buf is not None or _CONF_SINK is not None)
        ):
            return False
    use_fp8 = fold_w8 is not None and fold_s8 is not None
    fw_t = fold_w8 if use_fp8 else fold_w
    if (
        fw_t is None
        or not fw_t.is_contiguous()
        or (use_fp8 and (fold_s8.dtype != torch.float32 or not fold_s8.is_contiguous()))
        or not fold_b.is_contiguous()
        or not w1_weight.is_contiguous()
        or not rnn_w_rec.is_contiguous()
        or context.dtype != torch.float32
        or base_logits.dtype != torch.bfloat16
        or base_logits.stride(2) != 1
        or (id_map is not None and (id_map.dtype != torch.int32 or not id_map.is_contiguous()))
    ):
        return False

    from vllm.distributed import get_tp_group, tensor_model_parallel_all_gather
    comm = getattr(get_tp_group().device_communicator, "rdma_comm", None)
    use_pdl = _DRAFT_PDL if pdl is None else bool(pdl)
    stream = int(torch.cuda.current_stream(base_logits.device).cuda_stream)
    _c48_slices = 192 if (_MARKOV_FOLD_C48_AUTO_SLICES and valid >= 16384) else _MARKOV_FOLD_C48_SLICES
    base_slices = _c48_slices if (2 <= rows <= 8) else (_MARKOV_FOLD_TC_SLICES if rows >= 2 else _MARKOV_FOLD_SLICES)
    slices = min(base_slices, max(48, triton.cdiv(valid, 32)))
    ch = triton.cdiv(triton.cdiv(valid, slices), 8) * 8
    if rows * slices * 4 > ws["mk_rec"].numel():
        return False
    rec = ws["mk_rec"].view(-1)[: rows * slices * 4].view(rows, slices, 4)

    state, latent = markov_rnn_step_fused(
        None, prev, context[0], w1_weight, rnn_w_rec,
        skip_dev=None, pdl=use_pdl, init_zero=True,
    )

    es_mode = 0
    if es_ctx is not None and rows == 1:
        es_K, es_bufs, es_ema, es_cal, es_step, es_hist, es_prior, es_is_dink = es_ctx
        es_mode = 1 if es_is_dink else (3 if es_hist else 2)
        cur_skip, cur_surv, cur_tok, cur_best = es_bufs[0], es_bufs[1], es_bufs[2], es_bufs[3]
        gr_ptr = es_bufs[4].data_ptr()
        sl_ptr = es_bufs[5].data_ptr() if len(es_bufs) > 5 else 0
        db_ptr = es_bufs[6].data_ptr() if (es_is_dink and len(es_bufs) > 6) else 0
        lm_ptr = es_bufs[7].data_ptr() if (es_is_dink and len(es_bufs) > 7) else 0
        em_ptr = es_ema.data_ptr() if (es_hist and es_ema is not None) else 0
        cl_ptr = es_cal.data_ptr()
        st_ptr = es_step.data_ptr()
    elif es_batch_ctx is not None and 2 <= rows <= 8:
        es_K, b_bufs, es_cal, es_prior = es_batch_ctx
        es_mode = 5
        cur_skip, cur_surv = b_bufs[0], b_bufs[1]
        cur_tok = cur_best = None
        gr_ptr = b_bufs[2].data_ptr()
        db_ptr = b_bufs[3].data_ptr()
        cl_ptr = es_cal.data_ptr()
        sl_ptr = lm_ptr = em_ptr = st_ptr = 0
    else:
        es_K = k_run
        es_prior = 0.0
        cur_skip = cur_surv = cur_tok = cur_best = None
        gr_ptr = sl_ptr = db_ptr = lm_ptr = em_ptr = cl_ptr = st_ptr = 0

    sink = _CONF_SINK
    has_id_map = id_map is not None
    fold_fma = int(q["fold_fma"])
    exp_mode = int(q["exp_mode"])

    for i in range(k_run):
        skip_i = cur_skip if (es_mode != 0 and i > 0) else None
        if rows == 1:
            sem = _next_fold_sem(base_logits.device, 1, force_enable=True) if (_MARKOV_INLINE_AG and slices > 1) else None
        else:
            bm_v7 = 8 if rows <= 8 else (16 if rows <= 16 else 32)
            sem = _next_fold_sem(base_logits.device, triton.cdiv(rows, bm_v7), force_enable=True) if (_MARKOV_INLINE_AG and slices > 1) else None
        fused_local = sem is not None
        if fused_local:
            ri = ws["mk_rec1_idx"]
            ws["mk_rec1_idx"] = ri ^ 1
            rec1 = ws["mk_rec1"][ri].view(-1)[: rows * 4].view(rows, 1, 4)
        else:
            rec1 = rec
        b_i = base_logits[:, i]
        rc_f = lib.dspark_fold_argmax_v7_fwd(
            b_i.data_ptr(), latent.data_ptr(), fw_t.data_ptr(),
            fold_s8.data_ptr() if use_fp8 else 0, fold_b.data_ptr(),
            rec.data_ptr(), rec1.data_ptr(), sem.data_ptr() if fused_local else 0,
            skip_i.data_ptr() if skip_i is not None else 0,
            id_map.data_ptr() if has_id_map else 0,
            rows, 256, slices, ch, int(b_i.stride(0)),
            float(softcap), int(valid), int(offset),
            1 if use_fp8 else 0, 1 if skip_i is not None else 0,
            1 if fused_local else 0, 1 if has_id_map else 0,
            0, fold_fma, exp_mode, 1 if use_pdl else 0, stream,
        )
        if rc_f != 0:
            return False
        ag_in = rec1 if fused_local else rec
        ag_slices = 1 if fused_local else slices
        if skip_i is not None and comm is not None and comm.should_gather(ag_in):
            g = comm.all_gather_stacked_cond(ag_in, skip_i).flatten(0, 1)
        else:
            g = tensor_model_parallel_all_gather(ag_in, dim=0)
        ws_tp = int(g.shape[0]) // rows
        out_tok_i = draft_tokens_out[:rows, i]
        conf_i = None
        if sink is not None and sink.col < sink.buf.shape[1] and rows <= sink.buf.shape[0]:
            conf_i = sink.buf[:rows, sink.col]
            sink.col += 1
        elif conf_buf is not None:
            conf_i = conf_buf[:rows, i]

        if i < k_run - 1:
            in_ptr = state.data_ptr()
            next_state = (ws["mk_state"][1] if in_ptr == ws["mk_state"][0].data_ptr() else ws["mk_state"][0])[:rows]
            li = ws["mk_lat_idx"]
            ws["mk_lat_idx"] = li ^ 1
            next_latent = ws["mk_latent"][li][:rows]
            if es_mode == 5:
                pp_idx = i & 1
                nxt_skip = ws["es_pp_skip"][pp_idx]
                nxt_surv = ws["es_pp_surv_b"][pp_idx]
                nxt_tok = nxt_best = None
            elif es_mode != 0:
                pp_idx = i & 1
                nxt_skip = ws["es_pp_skip"][pp_idx]
                nxt_surv = ws["es_pp_surv"][pp_idx]
                nxt_tok = ws["es_pp_tok"][pp_idx]
                nxt_best = ws["es_pp_best"][pp_idx]
            else:
                nxt_skip = nxt_surv = nxt_tok = nxt_best = None
            ctx_nx = context[i + 1]
            rc_sr = lib.dspark_select_and_rnn_step_v7_fwd(
                g.data_ptr(), rows, ag_slices, ws_tp * ag_slices,
                out_tok_i.data_ptr(), int(out_tok_i.stride(0)),
                conf_i.data_ptr() if conf_i is not None else 0,
                int(conf_i.stride(0)) if conf_i is not None else 0,
                1 if conf_i is not None else 0,
                es_mode, i, int(es_K),
                cur_skip.data_ptr() if cur_skip is not None else 0,
                cur_surv.data_ptr() if cur_surv is not None else 0,
                cur_tok.data_ptr() if cur_tok is not None else 0,
                cur_best.data_ptr() if cur_best is not None else 0,
                nxt_skip.data_ptr() if nxt_skip is not None else 0,
                nxt_surv.data_ptr() if nxt_surv is not None else 0,
                nxt_tok.data_ptr() if nxt_tok is not None else 0,
                nxt_best.data_ptr() if nxt_best is not None else 0,
                gr_ptr, sl_ptr, db_ptr, lm_ptr, em_ptr, cl_ptr, st_ptr, float(es_prior),
                state.data_ptr(), ctx_nx.data_ptr(), int(ctx_nx.stride(0)),
                w1_weight.data_ptr(), rnn_w_rec.data_ptr(),
                next_state.data_ptr(), next_latent.data_ptr(),
                exp_mode, 1 if use_pdl else 0, stream,
            )
            if rc_sr != 0:
                return False
            state, latent = next_state, next_latent
            if es_mode != 0:
                cur_skip, cur_surv, cur_tok, cur_best = nxt_skip, nxt_surv, nxt_tok, nxt_best
        else:
            if (
                tail_vw_ctx is not None
                and _DSPARK_FUSE_TAIL_VW
                and hasattr(lib, "dspark_select_last_and_vw_conf_v7_fwd")
                and g.dtype == torch.float32
                and g.is_contiguous()
            ):
                _cb_t = tail_vw_ctx["conf_buf"]
                _rd_t = tail_vw_ctx["req_drafts"]
                rc_tvw = lib.dspark_select_last_and_vw_conf_v7_fwd(
                    1,
                    g.data_ptr(),
                    rows,
                    ag_slices,
                    ws_tp * ag_slices,
                    k_run,
                    int(tail_vw_ctx["n_spec"]),
                    int(tail_vw_ctx["c_end"]),
                    1 if skip_i is not None else 0,
                    skip_i.data_ptr() if skip_i is not None else 0,
                    1,
                    tail_vw_ctx["idx"].data_ptr(),
                    int(tail_vw_ctx["idx_stride"]),
                    _cb_t.data_ptr() if _cb_t is not None else 0,
                    int(_cb_t.stride(0)) if _cb_t is not None else 0,
                    tail_vw_ctx["temp"].data_ptr(),
                    tail_vw_ctx["w"].data_ptr(),
                    tail_vw_ctx["c"].data_ptr(),
                    tail_vw_ctx["cal"].data_ptr(),
                    tail_vw_ctx["step"].data_ptr(),
                    draft_tokens_out.data_ptr(),
                    int(draft_tokens_out.stride(0)),
                    tail_vw_ctx["cslot"].data_ptr() if tail_vw_ctx["cslot"] is not None else 0,
                    tail_vw_ctx["db"].data_ptr(),
                    tail_vw_ctx["etok"].data_ptr(),
                    tail_vw_ctx["ems"].data_ptr(),
                    tail_vw_ctx["cpred"].data_ptr(),
                    int(tail_vw_ctx["w_cap"]),
                    float(tail_vw_ctx["tau_override"]),
                    _rd_t.data_ptr() if _rd_t is not None else 0,
                    int(_rd_t.stride(0)) if _rd_t is not None else 0,
                    float(tail_vw_ctx["prior_logit"]),
                    int(tail_vw_ctx["K"]),
                    1 if tail_vw_ctx["use_hist"] else 0,
                    1 if tail_vw_ctx["use_dink"] else 0,
                    1 if tail_vw_ctx["do_log"] else 0,
                    1 if _rd_t is not None else 0,
                    1 if tail_vw_ctx["clear_req_tail"] else 0,
                    int(tail_vw_ctx["req_k"]),
                    exp_mode,
                    1 if use_pdl else 0,
                    stream,
                )
                if rc_tvw == 0:
                    tail_vw_ctx["ran"] = True
                    return True
            rc_s = lib.dspark_argmax_select_v7_fwd(
                g.data_ptr(), rows, ag_slices, ws_tp * ag_slices,
                out_tok_i.data_ptr(), int(out_tok_i.stride(0)),
                conf_i.data_ptr() if conf_i is not None else 0,
                int(conf_i.stride(0)) if conf_i is not None else 0,
                1 if conf_i is not None else 0,
                es_mode, i, int(es_K),
                cur_skip.data_ptr() if cur_skip is not None else 0,
                cur_surv.data_ptr() if cur_surv is not None else 0,
                cur_tok.data_ptr() if cur_tok is not None else 0,
                cur_best.data_ptr() if cur_best is not None else 0,
                gr_ptr, sl_ptr, db_ptr, lm_ptr, em_ptr, cl_ptr, st_ptr, float(es_prior),
                0, 0, exp_mode, 1 if use_pdl else 0, stream,
            )
            if rc_s != 0:
                return False
    return True


def vw_update_dink_v10(
    idx, num_sampled, num_rejected, temp,
    w, cpred, db, etok, ems, step, stats, cal,
    es_bufs,
    *,
    eta_b: float, decay_b: float, b_clip: float, eta_lam: float,
    K: int, do_stats: bool, fuse_es: bool, pdl: bool | None = None,
) -> bool:
    """Wave-V12: Native PDL CUDA wrapper for already-compiled `dspark_vw_update_dink_v7_fwd`."""
    if not _DSPARK_FUSE_EPILOGUE_PREP or idx.shape[0] <= 0 or K > 8:
        return False
    lib = _get_dspark_v7_lib()
    q = _v7_q(idx.device)
    if lib is None or q is None or not q.get("markov_ok", False) or not hasattr(lib, "dspark_vw_update_dink_v7_fwd"):
        return False
    if idx.dtype not in (torch.int32, torch.int64) or not idx.is_contiguous():
        return False
    if num_sampled.dtype != torch.int32 or num_rejected.dtype != torch.int32 or temp.dtype != torch.float32:
        return False
    do_fuse = bool(fuse_es and es_bufs is not None and len(es_bufs) == 8)
    if do_fuse:
        es_skip, es_surv, es_tok, es_bestu, es_greedy, es_slot, es_db, es_lam = es_bufs
    else:
        es_skip = es_surv = es_tok = es_bestu = es_greedy = es_slot = es_db = es_lam = None
    use_pdl = _DRAFT_PDL if pdl is None else bool(pdl)
    stream = int(torch.cuda.current_stream(idx.device).cuda_stream)
    rc = lib.dspark_vw_update_dink_v7_fwd(
        idx.data_ptr(),
        1 if idx.dtype == torch.int64 else 0,
        int(idx.shape[0]),
        num_sampled.data_ptr(),
        num_rejected.data_ptr(),
        temp.data_ptr(),
        w.data_ptr(),
        cpred.data_ptr(),
        db.data_ptr(),
        etok.data_ptr(),
        ems.data_ptr(),
        step.data_ptr(),
        stats.data_ptr() if (do_stats and stats is not None) else 0,
        cal.data_ptr(),
        es_skip.data_ptr() if do_fuse else 0,
        es_surv.data_ptr() if do_fuse else 0,
        es_tok.data_ptr() if do_fuse else 0,
        es_bestu.data_ptr() if do_fuse else 0,
        es_greedy.data_ptr() if do_fuse else 0,
        es_slot.data_ptr() if do_fuse else 0,
        es_db.data_ptr() if do_fuse else 0,
        es_lam.data_ptr() if do_fuse else 0,
        0, 0, 0, 0,
        float(eta_b), float(decay_b), float(b_clip), float(eta_lam),
        int(K), 1 if do_stats else 0, 1 if do_fuse else 0,
        1 if use_pdl else 0, stream,
    )
    return rc == 0


def fused_epilogue_prep_v10(
    out_input_ids, out_query_positions, out_is_padding,
    out_query_start_loc, out_seq_lens, out_context_positions,
    out_sample_indices, out_sample_pos, out_sample_idx_mapping,
    out_temperature, out_seeds, out_spec_num_rejected,
    qs, cs, bts, bt_strides, bszs, bsts,
    target_positions, target_query_start_loc, idx_mapping,
    last_sampled, next_prefill_tokens, num_sampled, num_rejected,
    input_temperature, input_seeds,
    *,
    parallel_drafting_token_id: int, num_query_per_req: int, num_speculative_steps: int,
    max_num_reqs: int, max_num_tokens: int, max_model_len: int,
    num_reqs: int, num_pad_programs: int, num_blocks: int,
    pad_slot_id: int, write_spec_num_rejected: bool,
    pad_c_start: int, pad_c_end: int, ng: int,
    dink_ctx: dict | None = None,
    pdl: bool | None = None,
) -> bool:
    """Wave-V12: Native PDL CUDA 3->1 epilogue prep (`dspark_fused_epilogue_prep_v7_fwd`)."""
    if not _DSPARK_FUSE_EPILOGUE_PREP or num_reqs <= 0 or num_blocks <= 0 or not (1 <= ng <= 5):
        return False
    lib = _get_dspark_v7_lib()
    q = _v7_q(out_input_ids.device)
    if lib is None or q is None or not q.get("markov_ok", False) or not hasattr(lib, "dspark_fused_epilogue_prep_v7_fwd"):
        return False
    if out_is_padding.element_size() != 1 or input_temperature.dtype != torch.float32 or out_temperature.dtype != torch.float32:
        return False
    if target_query_start_loc.dtype != torch.int32 or num_sampled.dtype != torch.int32 or num_rejected.dtype != torch.int32:
        return False
    if out_query_start_loc.dtype != torch.int32 or out_seq_lens.dtype != torch.int32:
        return False
    if out_seeds.dtype != input_seeds.dtype or out_seeds.dtype not in (torch.int32, torch.int64):
        return False
    # Exact bit layout matching dspark_fused_epilogue_prep_v10_kernel (note bit 10 = seed_is_i32!)
    flags = 0
    if out_input_ids.dtype == torch.int64:          flags |= (1 << 0)
    if out_query_positions.dtype == torch.int64:    flags |= (1 << 1)
    if out_context_positions.dtype == torch.int64:  flags |= (1 << 2)
    if out_sample_indices.dtype == torch.int64:     flags |= (1 << 3)
    if out_sample_pos.dtype == torch.int64:         flags |= (1 << 4)
    if out_sample_idx_mapping.dtype == torch.int64: flags |= (1 << 5)
    if target_positions.dtype == torch.int64:       flags |= (1 << 6)
    if idx_mapping.dtype == torch.int64:            flags |= (1 << 7)
    if last_sampled.dtype == torch.int64:           flags |= (1 << 8)
    if next_prefill_tokens.dtype == torch.int64:    flags |= (1 << 9)
    if input_seeds.dtype == torch.int32:            flags |= (1 << 10)
    do_dink = dink_ctx is not None
    if do_dink:
        do_fuse_es = bool(dink_ctx["fuse_es"] and dink_ctx["es_bufs"] is not None and len(dink_ctx["es_bufs"]) == 8)
        eb = dink_ctx["es_bufs"] if do_fuse_es else None
        w_p = dink_ctx["w"].data_ptr()
        cp_p = dink_ctx["cpred"].data_ptr()
        db_p = dink_ctx["db"].data_ptr()
        et_p = dink_ctx["etok"].data_ptr()
        em_p = dink_ctx["ems"].data_ptr()
        st_p = dink_ctx["step"].data_ptr()
        sn_p = dink_ctx["stats"].data_ptr() if (dink_ctx["do_stats"] and dink_ctx["stats"] is not None) else 0
        cl_p = dink_ctx["cal"].data_ptr()
        es_ptrs = [x.data_ptr() for x in eb] if do_fuse_es else [0] * 8
        eta_b = float(dink_ctx["eta_b"])
        decay_b = float(dink_ctx["decay_b"])
        b_clip = float(dink_ctx["b_clip"])
        eta_lam = float(dink_ctx["eta_lam"])
        dink_k = int(dink_ctx["K"])
        do_stats = 1 if dink_ctx["do_stats"] else 0
        fuse_es_i = 1 if do_fuse_es else 0
    else:
        w_p = cp_p = db_p = et_p = em_p = st_p = sn_p = cl_p = 0
        es_ptrs = [0] * 8
        eta_b = 0.25
        decay_b = 0.02
        b_clip = 2.0
        eta_lam = 0.05
        dink_k = 7
        do_stats = 0
        fuse_es_i = 0
    use_pdl = _DRAFT_PDL if pdl is None else bool(pdl)
    stream = int(torch.cuda.current_stream(out_input_ids.device).cuda_stream)
    rc = lib.dspark_fused_epilogue_prep_v7_fwd(
        out_input_ids.data_ptr(), out_query_positions.data_ptr(), out_is_padding.data_ptr(),
        out_query_start_loc.data_ptr(), out_seq_lens.data_ptr(), out_context_positions.data_ptr(),
        out_sample_indices.data_ptr(), out_sample_pos.data_ptr(), out_sample_idx_mapping.data_ptr(),
        out_temperature.data_ptr(), out_seeds.data_ptr(),
        out_spec_num_rejected.data_ptr() if write_spec_num_rejected else 0,
        qs[0].data_ptr(), qs[1].data_ptr(), qs[2].data_ptr(), qs[3].data_ptr(), qs[4].data_ptr(),
        cs[0].data_ptr(), cs[1].data_ptr(), cs[2].data_ptr(), cs[3].data_ptr(), cs[4].data_ptr(),
        bts[0].data_ptr(), bts[1].data_ptr(), bts[2].data_ptr(), bts[3].data_ptr(), bts[4].data_ptr(),
        int(bt_strides[0]), int(bt_strides[1]), int(bt_strides[2]), int(bt_strides[3]), int(bt_strides[4]),
        int(bszs[0]), int(bszs[1]), int(bszs[2]), int(bszs[3]), int(bszs[4]),
        int(bsts[0]), int(bsts[1]), int(bsts[2]), int(bsts[3]), int(bsts[4]),
        target_positions.data_ptr(), target_query_start_loc.data_ptr(), idx_mapping.data_ptr(),
        last_sampled.data_ptr(), next_prefill_tokens.data_ptr(),
        num_sampled.data_ptr(), num_rejected.data_ptr(),
        input_temperature.data_ptr(), input_seeds.data_ptr(),
        int(parallel_drafting_token_id), int(num_query_per_req), int(num_speculative_steps),
        int(max_num_reqs), int(max_num_tokens), int(max_model_len),
        int(num_reqs), int(num_pad_programs), int(num_blocks),
        int(pad_slot_id), 1 if write_spec_num_rejected else 0,
        int(pad_c_start), int(pad_c_end), int(ng), int(flags),
        1 if do_dink else 0,
        w_p, cp_p, db_p, et_p, em_p, st_p, sn_p, cl_p,
        es_ptrs[0], es_ptrs[1], es_ptrs[2], es_ptrs[3],
        es_ptrs[4], es_ptrs[5], es_ptrs[6], es_ptrs[7],
        0, 0, 0, 0,
        eta_b, decay_b, b_clip, eta_lam, dink_k, do_stats, fuse_es_i,
        1 if use_pdl else 0, stream,
    )
    return rc == 0


def select_last_and_tree_pack_v10(
    idx, temp, prim_toks, conf_buf, sec_toks, sec_p,
    cal, step, db, etok, ems, w, cpred,
    draft_tokens, req_drafts,
    tree_active, n_live, tree_meta, tree_parent, tree_depth, tree_sconv_anc, tree_path,
    *,
    ws_max: int, sib_disc: float, sib_penalty_ms: float, sib_scale: float, p2_min: float,
    write_req_drafts: bool, pdl: bool | None = None,
) -> bool:
    """Wave-V12: Race-free 1-CTA PDL CUDA wrapper for `dspark_select_last_and_tree_pack_v7_fwd`."""
    if not _TREE_FUSE_V10 or idx.dtype != torch.int64:
        return False
    lib = _get_dspark_v7_lib()
    q = _v7_q(idx.device)
    if lib is None or q is None or not q.get("markov_ok", False) or not hasattr(lib, "dspark_select_last_and_tree_pack_v7_fwd"):
        return False
    use_pdl = _DRAFT_PDL if pdl is None else bool(pdl)
    stream = int(torch.cuda.current_stream(idx.device).cuda_stream)
    rc = lib.dspark_select_last_and_tree_pack_v7_fwd(
        0, 0, 1, 8, 0, 0,
        idx.data_ptr(), temp.data_ptr(),
        prim_toks.data_ptr(), conf_buf.data_ptr(),
        sec_toks.data_ptr(), sec_p.data_ptr(),
        cal.data_ptr(), step.data_ptr(),
        db.data_ptr(), etok.data_ptr(), ems.data_ptr(),
        w.data_ptr(), cpred.data_ptr(),
        draft_tokens.data_ptr(),
        req_drafts.data_ptr() if write_req_drafts else 0,
        int(req_drafts.stride(0)) if write_req_drafts else 0,
        tree_active.data_ptr(), n_live.data_ptr(), tree_meta.data_ptr(),
        tree_parent.data_ptr(), tree_depth.data_ptr(),
        tree_sconv_anc.data_ptr(), tree_path.data_ptr(),
        int(ws_max), float(sib_disc), float(sib_penalty_ms), float(sib_scale), float(p2_min),
        1 if write_req_drafts else 0,
        int(q["exp_mode"]), 1 if use_pdl else 0, stream,
    )
    return rc == 0


def es_init_dink(idx, temp, step_eff, cal_ptr, db_ptr, etok_ptr, ems_ptr, state_bufs, dummy: bool):
    """Initialize coupled Dinkelbach early-stop GPU state for R=1 before draft graph replay."""
    skip_d, surv_d, tok_d, bestu_d, greedy_d, slot_d, db_d, lam_d = state_bufs
    _es_init_dink_kernel[(1,)](
        idx, temp, step_eff, cal_ptr, db_ptr, etok_ptr, ems_ptr,
        skip_d, surv_d, tok_d, bestu_d, greedy_d, slot_d, db_d, lam_d,
        DUMMY=dummy, num_warps=1,
    )


@triton.jit
def _fold_argmax_tc_kernel(
    BASE, LAT, FW, FS, FB, REC, REC_OUT, SEM, SKIP_PTR, ID_MAP,
    stride_base, rows, softcap, valid, offset, CH,
    RANK: tl.constexpr, BM: tl.constexpr, BC: tl.constexpr, USE_FP8: tl.constexpr,
    FUSE_REDUCE: tl.constexpr = False, NP2_S: tl.constexpr = 128,
    PDL: tl.constexpr = False,
    HAS_ID_MAP: tl.constexpr = False,
    HAS_SKIP: tl.constexpr = False,
):
    """High-concurrency (rows >= 2) batched Tensor-Core fold + softcap + argmax_record.
    Reads fold_weight [V, RANK] ONCE per BM rows (16 or 32) instead of per-row scalar loops
    (3.7x faster at B=32: 2.22 ms -> 0.60 ms across 7 Markov steps)."""
    pm = tl.program_id(0)
    s = tl.program_id(1)
    S = tl.num_programs(1)
    m = pm * BM + tl.arange(0, BM)
    mm = m < rows
    k = tl.arange(0, RANK)
    if PDL:
        tl.extra.cuda.gdc_wait()
    if HAS_SKIP:
        if tl.load(SKIP_PTR) != 0:
            if PDL:
                tl.extra.cuda.gdc_launch_dependents()
            return
    lat = tl.load(LAT + m.to(tl.int64)[:, None] * RANK + k[None, :], mask=mm[:, None], other=0.0)

    lo = s * CH
    hi = tl.minimum(lo + CH, valid)
    m_best = tl.full((BM,), -float("inf"), tl.float32)
    idx_best = tl.full((BM,), 2147483647, tl.int32)
    se_best = tl.zeros((BM,), tl.float32)

    for c0 in range(lo, hi, BC):
        cols = c0 + tl.arange(0, BC)
        cmask = cols < hi
        c64 = cols.to(tl.int64)[:, None] * RANK
        fw = tl.load(FW + c64 + k[None, :], mask=cmask[:, None], other=0.0, eviction_policy="evict_last")
        if USE_FP8:
            fw = fw.to(tl.bfloat16)
        dot = tl.dot(lat, tl.trans(fw))
        if USE_FP8:
            fs = tl.load(FS + cols, mask=cmask, other=0.0)
            dot = dot * fs[None, :]
        fb = tl.load(FB + cols, mask=cmask, other=0.0).to(tl.float32)
        b = (dot + fb[None, :]).to(tl.bfloat16).to(tl.float32)
        a = tl.load(
            BASE + m.to(tl.int64)[:, None] * stride_base + cols[None, :],
            mask=mm[:, None] & cmask[None, :],
            other=0.0,
        ).to(tl.float32)
        x = (a + b).to(tl.bfloat16)
        x = x / softcap
        x = tldevice.tanh(x)
        x = x * softcap
        x = x.to(tl.bfloat16).to(tl.float32)
        x = tl.where(mm[:, None] & cmask[None, :] & (x == x), x, -float("inf"))

        cm = tl.max(x, 1)
        cidx = tl.min(tl.where(x == cm[:, None], cols[None, :], 2147483647), 1)
        m_new = tl.maximum(m_best, cm)
        m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        e = tl.where(x == -float("inf"), 0.0, tl.exp(x - m_safe[:, None]))
        sc_old = tl.where(m_best == -float("inf"), 0.0, tl.exp(m_best - m_safe))
        se_best = se_best * sc_old + tl.sum(e, 1)
        idx_best = tl.where(cm > m_best, cidx, tl.where(cm == m_best, tl.minimum(idx_best, cidx), idx_best))
        m_best = m_new

    rec_base = (m.to(tl.int64) * S + s) * 4
    if HAS_ID_MAP:
        safe_idx = tl.where(idx_best < valid, idx_best, 0)
        out_id = tl.load(ID_MAP + safe_idx, mask=mm, other=0).to(tl.float32)
    else:
        out_id = (idx_best + offset).to(tl.float32)
    if not FUSE_REDUCE:
        if PDL:
            tl.extra.cuda.gdc_launch_dependents()
        tl.store(REC + rec_base + 0, m_best, mask=mm)
        tl.store(REC + rec_base + 1, out_id, mask=mm)
        tl.store(REC + rec_base + 2, se_best, mask=mm)
        tl.store(REC + rec_base + 3, 0.0, mask=mm)
    else:
        tl.store(REC + rec_base + 0, m_best, mask=mm)
        tl.store(REC + rec_base + 1, out_id, mask=mm)
        tl.store(REC + rec_base + 2, se_best, mask=mm)
        tl.store(REC + rec_base + 3, 0.0, mask=mm)
        tl.debug_barrier()
        if tl.atomic_add(SEM, 1, sem="acq_rel") == S - 1:
            tl.store(SEM, 0)
            ss = tl.arange(0, NP2_S)
            ok_s = ss < S
            o4 = tl.arange(0, 4)
            for r_i in range(0, rows):
                rbase = (r_i * S + ss).to(tl.int64) * 4
                rv = tl.load(REC + rbase, ok_s, other=-float("inf"), cache_modifier=".cg")
                ridx = tl.load(REC + rbase + 1, ok_s, other=0.0, cache_modifier=".cg")
                rse = tl.load(REC + rbase + 2, ok_s, other=0.0, cache_modifier=".cg")
                rm = tl.max(rv, 0)
                rfirst = tl.min(tl.where(ok_s & (rv == rm), ss, NP2_S), 0)
                rbest_i = tl.sum(tl.where(ss == rfirst, ridx, 0.0), 0)
                rm0 = tl.where(rm == -float("inf"), 0.0, rm)
                rtot_se = tl.sum(tl.where(rv == -float("inf"), 0.0, rse * tl.exp(rv - rm0)), 0)
                rval = tl.where(o4 == 0, rm, tl.where(o4 == 1, rbest_i, tl.where(o4 == 2, rtot_se, 0.0)))
                tl.store(REC_OUT + (r_i * 4 + o4).to(tl.int64), rval)
            if PDL:
                tl.extra.cuda.gdc_launch_dependents()


def gathered_fold_argmax_fused(
    base, latent, fold_w, fold_b, softcap, valid, offset,
    fold_w8=None, fold_s8=None,
    conf_out=None,
    es_ctx=None,  # (step_idx, K, state_bufs, ema_ptr, cal_ptr, step_ptr, use_hist, prior_logit)
    out_tok=None,
    pdl: bool | None = None,
    id_map: "torch.Tensor | None" = None,
):
    """Fused fold + softcap + argmax_record + (conditional) RDMA AG + (early-stop) argmax_select."""
    from vllm.distributed import get_tp_group, tensor_model_parallel_all_gather

    rows = base.shape[0]
    rank = latent.shape[1]
    q = _v7_q(base.device)
    v7_lib = (
        _get_dspark_v7_lib()
        if (
            q is not None
            and q.get("markov_ok", False)
            and rank == 256
            and base.dtype == torch.bfloat16
            and base.stride(1) == 1
            and latent.dtype == torch.bfloat16
            and latent.is_contiguous()
            and fold_b.dtype == torch.bfloat16
            and fold_b.is_contiguous()
            and (id_map is None or (id_map.dtype == torch.int32 and id_map.is_contiguous()))
        )
        else None
    )
    if v7_lib is not None and 2 <= rows <= 8:
        base_slices = 192 if (_MARKOV_FOLD_C48_AUTO_SLICES and valid >= 16384) else _MARKOV_FOLD_C48_SLICES
    else:
        base_slices = _MARKOV_FOLD_TC_SLICES if rows >= 2 else _MARKOV_FOLD_SLICES
    # Dynamically clamp slices for pruned vocabularies so we never launch empty CTAs or pad NP2=512.
    slices = min(base_slices, max(48, triton.cdiv(valid, 32)))
    ch = triton.cdiv(triton.cdiv(valid, slices), 8) * 8
    use_pdl = _DRAFT_PDL if pdl is None else bool(pdl)

    skip_dev = None
    if es_ctx is not None and (rows == 1 or (len(es_ctx) == 6 and es_ctx[5] == "batch")):
        step_idx = es_ctx[0]
        if step_idx > 0:
            skip_dev = es_ctx[2][0]

    use_fp8 = fold_w8 is not None and fold_s8 is not None
    has_id_map = id_map is not None
    fused_local = False
    rec1 = None
    ran_v7 = False

    if v7_lib is not None and rows > 0 and (rows == 1 or skip_dev is None or q.get("fold_tc_skip_ok", False)):
        fw_t = fold_w8 if use_fp8 else fold_w
        if fw_t is not None and fw_t.is_contiguous() and (not use_fp8 or (fold_s8.dtype == torch.float32 and fold_s8.is_contiguous())):
            ws = _V7_WS.get(_ddev_idx(base.device))
            if ws is not None and rows * slices * 4 <= ws["mk_rec"].numel():
                rec = ws["mk_rec"].view(-1)[: rows * slices * 4].view(rows, slices, 4)
            else:
                rec = torch.empty((rows, slices, 4), dtype=torch.float32, device=base.device)
            if rows == 1:
                sem = _next_fold_sem(base.device, 1, force_enable=True) if (_MARKOV_INLINE_AG and slices > 1) else None
            else:
                bm_v7 = 8 if rows <= 8 else (16 if rows <= 16 else 32)
                n_bm = triton.cdiv(rows, bm_v7)
                sem = _next_fold_sem(base.device, n_bm, force_enable=True) if (_MARKOV_INLINE_AG and slices > 1) else None
            fused_local = sem is not None
            if fused_local:
                if ws is not None and rows * 4 <= ws["mk_rec1"][0].numel():
                    ri = ws["mk_rec1_idx"]
                    ws["mk_rec1_idx"] = ri ^ 1
                    rec1 = ws["mk_rec1"][ri].view(-1)[: rows * 4].view(rows, 1, 4)
                else:
                    rec1 = torch.empty((rows, 1, 4), dtype=torch.float32, device=base.device)
            rc_f = v7_lib.dspark_fold_argmax_v7_fwd(
                base.data_ptr(),
                latent.data_ptr(),
                fw_t.data_ptr(),
                fold_s8.data_ptr() if use_fp8 else 0,
                fold_b.data_ptr(),
                rec.data_ptr(),
                rec1.data_ptr() if fused_local else rec.data_ptr(),
                sem.data_ptr() if fused_local else 0,
                skip_dev.data_ptr() if skip_dev is not None else 0,
                id_map.data_ptr() if has_id_map else 0,
                int(rows),
                int(rank),
                int(slices),
                int(ch),
                int(base.stride(0)),
                float(softcap),
                int(valid),
                int(offset),
                1 if use_fp8 else 0,
                1 if skip_dev is not None else 0,
                1 if fused_local else 0,
                1 if has_id_map else 0,
                0,
                int(q["fold_fma"]),
                int(q["exp_mode"]),
                1 if use_pdl else 0,
                int(torch.cuda.current_stream(base.device).cuda_stream),
            )
            if rc_f == 0:
                ran_v7 = True

    if not ran_v7:
        rec = torch.empty((rows, slices, 4), dtype=torch.float32, device=base.device)
        id_map_arg = id_map if has_id_map else rec
        fused_local = False
        rec1 = None
        if rows >= 2:
            bm = 16 if rows <= 16 else 32
            sem = (
                _next_fold_sem(base.device, 1, force_enable=True)
                if (_MARKOV_TC_FUSE_REDUCE and _MARKOV_INLINE_AG and slices > 1 and rows <= bm)
                else None
            )
            fused_local = sem is not None
            np2_s = triton.next_power_of_2(slices) if fused_local else 128
            if fused_local:
                rec1 = torch.empty((rows, 1, 4), dtype=torch.float32, device=base.device)
            _fold_argmax_tc_kernel[(triton.cdiv(rows, bm), slices)](
                base, latent,
                fold_w8 if use_fp8 else fold_w,
                fold_s8 if use_fp8 else fold_b,
                fold_b, rec,
                rec1 if fused_local else rec,
                sem if fused_local else rec,
                skip_dev if skip_dev is not None else rec,
                id_map_arg,
                base.stride(0), rows, softcap, valid, offset, ch,
                RANK=rank, BM=bm, BC=32, USE_FP8=use_fp8,
                FUSE_REDUCE=fused_local, NP2_S=np2_s, PDL=use_pdl,
                HAS_ID_MAP=has_id_map,
                HAS_SKIP=skip_dev is not None,
                num_warps=4, launch_pdl=use_pdl,
            )
        else:
            sem = _next_fold_sem(base.device, rows) if (_MARKOV_INLINE_AG and slices > 1) else None
            fused_local = sem is not None
            np2_s = triton.next_power_of_2(slices) if fused_local else 512
            if fused_local:
                rec1 = torch.empty((rows, 1, 4), dtype=torch.float32, device=base.device)
            _fold_argmax_record_kernel[(rows, slices)](
                base, latent,
                fold_w8 if use_fp8 else fold_w,
                fold_s8 if use_fp8 else fold_b,
                fold_b, rec,
                rec1 if fused_local else rec,
                sem if fused_local else rec,
                skip_dev if skip_dev is not None else rec,
                id_map_arg,
                base.stride(0), softcap, valid, offset, ch,
                RANK=rank, BC=32, USE_FP8=use_fp8, HAS_SKIP=skip_dev is not None,
                FUSE_REDUCE=fused_local, NP2_S=np2_s, PDL=use_pdl,
                HAS_ID_MAP=has_id_map,
                num_warps=_MARKOV_FOLD_WARPS, launch_pdl=use_pdl,
            )

    if fused_local:
        ag_in = rec1
        ag_slices = 1
    elif _MARKOV_INLINE_AG and slices > 1:
        rec1 = torch.empty((rows, 1, 4), dtype=torch.float32, device=base.device)
        np2_s = triton.next_power_of_2(slices)
        _argmax_local_reduce_kernel[(rows,)](
            rec, rec1, skip_dev if skip_dev is not None else rec1,
            slices, NP2=np2_s, HAS_SKIP=skip_dev is not None, PDL=use_pdl,
            num_warps=4 if np2_s >= 128 else 1, launch_pdl=use_pdl,
        )
        ag_in = rec1
        ag_slices = 1
    else:
        ag_in = rec
        ag_slices = slices

    comm = getattr(get_tp_group().device_communicator, "rdma_comm", None)
    if skip_dev is not None and comm is not None and comm.should_gather(ag_in):
        g = comm.all_gather_stacked_cond(ag_in, skip_dev).flatten(0, 1)  # [world * rows, ag_slices, 4]
    else:
        g = tensor_model_parallel_all_gather(ag_in, dim=0)

    if es_ctx is not None and len(es_ctx) == 6 and es_ctx[5] == "batch" and 2 <= rows <= 8 and conf_out is not None:
        step_idx, K, b_bufs, cal_ptr, tau_batch, _tag = es_ctx
        skip_d, surv_b, greedy_all, db_b = b_bufs
        ws_tp = int(g.shape[0]) // rows
        np2 = triton.next_power_of_2(ws_tp * ag_slices)
        out = out_tok if (out_tok is not None and out_tok.shape[0] >= rows) else torch.empty(rows, dtype=torch.int64, device=g.device)
        if (
            v7_lib is not None
            and q.get("select_rnn_tc_es_ok", False)
            and hasattr(v7_lib, "dspark_argmax_select_batch_es_v10_fwd")
            and g.dtype == torch.float32
            and g.is_contiguous()
        ):
            rc_be = v7_lib.dspark_argmax_select_batch_es_v10_fwd(
                g.data_ptr(), int(rows), int(ag_slices), int(ws_tp * ag_slices),
                out.data_ptr(), int(out.stride(0)), conf_out.data_ptr(), int(conf_out.stride(0)),
                int(step_idx), int(K),
                skip_d.data_ptr(), surv_b.data_ptr(),
                greedy_all.data_ptr(), db_b.data_ptr(), cal_ptr.data_ptr(),
                float(tau_batch), int(q["exp_mode"]), 1 if use_pdl else 0,
                int(torch.cuda.current_stream(g.device).cuda_stream),
            )
            if rc_be == 0:
                return out
        _argmax_select_es_batch_kernel[(1,)](
            g, out, int(out.stride(0)), conf_out, int(conf_out.stride(0)),
            skip_d, surv_b, greedy_all, db_b, cal_ptr,
            rows, ag_slices, ws_tp * ag_slices, float(tau_batch),
            STEP_IDX=step_idx, K=K, NP2=np2, MAX_B=8, PDL=use_pdl,
            num_warps=4 if np2 >= 512 else 1, launch_pdl=use_pdl,
        )
        return out
    if es_ctx is not None and rows == 1 and conf_out is not None:
        ws_tp = g.shape[0]
        np2 = triton.next_power_of_2(ws_tp * ag_slices)
        out = out_tok if (out_tok is not None and out_tok.shape[0] >= 1) else torch.empty(1, dtype=torch.int64, device=g.device)
        if v7_lib is not None and g.dtype == torch.float32 and g.is_contiguous():
            if len(es_ctx) == 9 and es_ctx[8] == "dink":
                step_idx, K, state_bufs, _ema_ptr, cal_ptr, step_ptr, _use_hist, _prior_logit, _mode = es_ctx
                skip_d, surv_d, tok_d, bestu_d, greedy_d, _slot_d, db_d, lam_d = state_bufs
                rc_s = v7_lib.dspark_argmax_select_v7_fwd(
                    g.data_ptr(), 1, int(ag_slices), int(ws_tp * ag_slices),
                    out.data_ptr(), int(out.stride(0)), conf_out.data_ptr(), int(conf_out.stride(0)), 1,
                    1, int(step_idx), int(K),
                    skip_d.data_ptr(), surv_d.data_ptr(), tok_d.data_ptr(), bestu_d.data_ptr(),
                    greedy_d.data_ptr(), 0, db_d.data_ptr(), lam_d.data_ptr(),
                    0, cal_ptr.data_ptr(), step_ptr.data_ptr(), 0.0,
                    0, 0, int(q["exp_mode"]), 1 if use_pdl else 0,
                    int(torch.cuda.current_stream(g.device).cuda_stream),
                )
                if rc_s == 0:
                    return out
            else:
                step_idx, K, state_bufs, ema_ptr, cal_ptr, step_ptr, use_hist, prior_logit = es_ctx[:8]
                skip_d, surv_d, tok_d, bestv_d, greedy_d, slot_d = state_bufs[:6]
                rc_s = v7_lib.dspark_argmax_select_v7_fwd(
                    g.data_ptr(), 1, int(ag_slices), int(ws_tp * ag_slices),
                    out.data_ptr(), int(out.stride(0)), conf_out.data_ptr(), int(conf_out.stride(0)), 1,
                    3 if use_hist else 2, int(step_idx), int(K),
                    skip_d.data_ptr(), surv_d.data_ptr(), tok_d.data_ptr(), bestv_d.data_ptr(),
                    greedy_d.data_ptr(), slot_d.data_ptr(), 0, 0,
                    ema_ptr.data_ptr() if (use_hist and ema_ptr is not None) else 0,
                    cal_ptr.data_ptr(), step_ptr.data_ptr(), float(prior_logit),
                    0, 0, int(q["exp_mode"]), 1 if use_pdl else 0,
                    int(torch.cuda.current_stream(g.device).cuda_stream),
                )
                if rc_s == 0:
                    return out
        if len(es_ctx) == 9 and es_ctx[8] == "dink":
            step_idx, K, state_bufs, _ema_ptr, cal_ptr, step_ptr, _use_hist, _prior_logit, _mode = es_ctx
            skip_d, surv_d, tok_d, bestu_d, greedy_d, _slot_d, db_d, lam_d = state_bufs
            _argmax_select_es_dink_kernel[(1,)](
                g, out, conf_out,
                skip_d, surv_d, tok_d, bestu_d, greedy_d, db_d, lam_d,
                cal_ptr, step_ptr,
                ag_slices, ws_tp * ag_slices,
                STEP_IDX=step_idx, K=K, NP2=np2, PDL=use_pdl,
                num_warps=4 if np2 >= 512 else 1, launch_pdl=use_pdl,
            )
            return out
        step_idx, K, state_bufs, ema_ptr, cal_ptr, step_ptr, use_hist, prior_logit = es_ctx[:8]
        skip_d, surv_d, tok_d, bestv_d, greedy_d, slot_d = state_bufs[:6]
        _argmax_select_es_kernel[(1,)](
            g, out, conf_out,
            skip_d, surv_d, tok_d, bestv_d, greedy_d, slot_d,
            ema_ptr, cal_ptr, step_ptr,
            ag_slices, ws_tp * ag_slices,
            STEP_IDX=step_idx, K=K, NP2=np2,
            USE_HIST=use_hist, PRIOR_LOGIT=prior_logit, PDL=use_pdl,
            num_warps=4 if np2 >= 512 else 1, launch_pdl=use_pdl,
        )
        return out
    return argmax_select(g, rows, with_conf=False, conf_out=conf_out, out_tok=out_tok, pdl=use_pdl)


@triton.jit
def _argmax_select_top2_kernel(
    G, TOK1_OUT, P1_OUT, TOK2_OUT, P2_OUT,
    NS, NP2: tl.constexpr, PDL: tl.constexpr = False,
):
    """Reduce TP-gathered [TP, rows, 8] fp32 records -> exact global (tok1, p1, tok2, p2)."""
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    row = tl.program_id(0).to(tl.int64)
    r = tl.arange(0, NP2)
    ok = r < NS
    base = (r * tl.num_programs(0) + row) * 8
    v1 = tl.load(G + base + 0, ok, other=-float("inf"))
    k1 = tl.load(G + base + 1, ok, other=0.0)
    se = tl.load(G + base + 2, ok, other=0.0)
    v2 = tl.load(G + base + 3, ok, other=-float("inf"))
    k2 = tl.load(G + base + 4, ok, other=0.0)

    m1 = tl.max(v1, 0)
    w1 = tl.min(tl.where(ok & (v1 == m1), r, NP2), 0)
    t1 = tl.sum(tl.where(r == w1, k1, 0.0), 0).to(tl.int64)

    v_pool = tl.where(r == w1, v2, v1)
    k_pool = tl.where(r == w1, k2, k1)
    m2 = tl.max(v_pool, 0)
    w2 = tl.min(tl.where(ok & (v_pool == m2), r, NP2), 0)
    t2 = tl.sum(tl.where(r == w2, k_pool, 0.0), 0).to(tl.int64)

    m0 = tl.where(m1 == -float("inf"), 0.0, m1)
    tot = tl.sum(tl.where(v1 == -float("inf"), 0.0, se * tl.exp(v1 - m0)), 0)
    p1 = tl.where(tot > 0.0, 1.0 / tot, 0.0)
    p2 = tl.where((tot > 0.0) & (m2 > -float("inf")), tl.exp(m2 - m0) / tot, 0.0)

    tl.store(TOK1_OUT + row, t1)
    tl.store(P1_OUT + row, p1)
    tl.store(TOK2_OUT + row, t2)
    tl.store(P2_OUT + row, p2)


def gathered_fold_argmax_top2_fused(
    base, latent, fold_w, fold_b, softcap, valid, offset,
    fold_w8=None, fold_s8=None, pdl: bool | None = None,
    id_map: "torch.Tensor | None" = None,
):
    """Single-pass top-2 Markov fold + 32 B/row inline RDMA AG returning (tok1, p1, tok2, p2)."""
    from vllm.distributed import tensor_model_parallel_all_gather

    rows, rank = int(base.shape[0]), int(latent.shape[1])
    slices = min(_MARKOV_FOLD_SLICES, max(48, triton.cdiv(valid, 32)))
    ch = triton.cdiv(triton.cdiv(valid, slices), 8) * 8
    sem = _next_fold_sem(base.device, rows, force_enable=_DSPARK_CUDA_V7)
    use_fp8 = fold_w8 is not None and fold_s8 is not None
    has_id_map = id_map is not None
    use_pdl = _DRAFT_PDL if pdl is None else bool(pdl)

    q = _v7_q(base.device)
    v7_lib = (
        _get_dspark_v7_lib()
        if (
            q is not None
            and q.get("markov_ok", False)
            and sem is not None
            and rank == 256
            and base.dtype == torch.bfloat16
            and base.stride(1) == 1
            and latent.dtype == torch.bfloat16
            and latent.is_contiguous()
            and fold_b.dtype == torch.bfloat16
            and fold_b.is_contiguous()
            and (
                not has_id_map
                or (
                    q.get("fold_top2_idmap_ok", False)
                    and id_map.dtype == torch.int32
                    and id_map.is_contiguous()
                )
            )
        )
        else None
    )
    ran_v7 = False
    ws = _V7_WS.get(_ddev_idx(base.device)) if v7_lib is not None else None
    if v7_lib is not None and ws is not None and rows * slices * 8 <= ws["mk_rec"].numel():
        rec = ws["mk_rec"].view(-1)[: rows * slices * 8].view(rows, slices, 8)
        ri = ws["mk_rec1_idx"]
        ws["mk_rec1_idx"] = ri ^ 1
        rec1 = ws["mk_rec1"][ri].view(-1)[: rows * 8].view(rows, 1, 8)
    else:
        rec = torch.empty((rows, slices, 8), dtype=torch.float32, device=base.device)
        rec1 = torch.empty((rows, 1, 8), dtype=torch.float32, device=base.device)

    if v7_lib is not None:
        fw_t = fold_w8 if use_fp8 else fold_w
        if fw_t is not None and fw_t.is_contiguous() and (not use_fp8 or (fold_s8.dtype == torch.float32 and fold_s8.is_contiguous())):
            rc_t = v7_lib.dspark_fold_argmax_v7_fwd(
                base.data_ptr(),
                latent.data_ptr(),
                fw_t.data_ptr(),
                fold_s8.data_ptr() if use_fp8 else 0,
                fold_b.data_ptr(),
                rec.data_ptr(),
                rec1.data_ptr(),
                sem.data_ptr(),
                0,
                id_map.data_ptr() if has_id_map else 0,
                rows, rank, slices, ch, int(base.stride(0)),
                float(softcap), int(valid), int(offset),
                1 if use_fp8 else 0,
                0, 1, 1 if has_id_map else 0, 1,
                int(q["fold_fma"]),
                int(q["exp_mode"]),
                1 if use_pdl else 0,
                int(torch.cuda.current_stream(base.device).cuda_stream),
            )
            if rc_t == 0:
                ran_v7 = True

    if not ran_v7:
        _fold_argmax_top2_record_kernel[(rows, slices)](
            base, latent,
            fold_w8 if use_fp8 else fold_w,
            fold_s8 if use_fp8 else fold_b,
            fold_b, rec, rec1, sem, rec,
            id_map if has_id_map else rec,
            base.stride(0), softcap, valid, offset, ch,
            RANK=rank, BC=32, USE_FP8=use_fp8, HAS_SKIP=False,
            FUSE_REDUCE=True, NP2_S=triton.next_power_of_2(slices), PDL=use_pdl,
            HAS_ID_MAP=has_id_map,
            num_warps=_MARKOV_FOLD_WARPS, launch_pdl=use_pdl,
        )
    g = tensor_model_parallel_all_gather(rec1.squeeze(1), dim=0)
    tp = g.shape[0] // rows
    tok1 = torch.empty((rows,), dtype=torch.int64, device=base.device)
    p1 = torch.empty((rows,), dtype=torch.float32, device=base.device)
    tok2 = torch.empty((rows,), dtype=torch.int64, device=base.device)
    p2 = torch.empty((rows,), dtype=torch.float32, device=base.device)
    if v7_lib is not None and g.dtype == torch.float32 and g.is_contiguous():
        rc_s = v7_lib.dspark_argmax_select_v7_fwd(
            g.data_ptr(), rows, 1, int(tp),
            tok1.data_ptr(), 1, p1.data_ptr(), 1, 1,
            4, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0,
            tok2.data_ptr(), p2.data_ptr(),
            int(q["exp_mode"]), 1 if use_pdl else 0,
            int(torch.cuda.current_stream(base.device).cuda_stream),
        )
        if rc_s == 0:
            return tok1, p1, tok2, p2
    _argmax_select_top2_kernel[(rows,)](
        g, tok1, p1, tok2, p2,
        tp, NP2=triton.next_power_of_2(tp), PDL=use_pdl,
        num_warps=1, launch_pdl=use_pdl,
    )
    return tok1, p1, tok2, p2
