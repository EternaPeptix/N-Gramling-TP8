# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark speculator for the Inkling2 draft: stores the raw target features
(each draft layer mixes them itself) and runs its recurrent Markov head.

S6 + conc-sched upgrades (100% bit-identical & zero-overhead at B=1):
  1. Preserves all S6 features: coupled Dinkelbach (`INKLING_VERIFY_W=dink`),
     zero-memcpy `out_tok`, `INKLING_HOST_PIN`, `INKLING_DTGRAPH`,
     `INKLING_ZERO_COPY_STORE`, and `INKLING_FUSE_DFLASH_PREP`.
  2. Guards `propose()` on `num_speculative_tokens == 0` or `context_kv_only`
     without clearing `_conf_armed` (keeps CUDA-graph-replayed `dink`/`conf` armed).
  3. Graph-capture-safe batch-aware Markov loop truncation (`k_run(B)`) when
     vLLM Dynamic SD (`num_speculative_tokens_per_batch_size`) caps `K < 7`
     for a batch size `B`, slicing `hidden` before `compute_base_logits` +
     `markov_head.context` and running only `k_run(B)` Markov steps.
  4. Batch-aware verify width cap `W_max(B)` (`INKLING_VERIFY_W_BATCH_CAP`) and
     confidence threshold `tau(B)` (`INKLING_CONF_TAU_BATCH`, non-dink) with
     `do_not_specialize` on `w_cap`/`tau_override` for zero runtime JIT stalls.
"""

import json
import math
import os
import threading
import time
from typing import Any

import torch
import torch.nn as nn
import triton
import triton.language as tl

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator
from vllm.v1.worker.gpu.spec_decode.dspark.utils import load_dspark_model

logger = init_logger(__name__)

_MARKOV_FUSE_REDUCE = os.environ.get(
    "INKLING_MARKOV_FUSE_REDUCE", "1"
).strip().lower() not in ("0", "", "off", "false", "no")

# Thread-local handoff from dtgraph's sample_tokens hook: when set, _vw_conf_kernel /
# _vw_mask_kernel scatter-writes the masked draft tokens directly into req_states.draft_tokens[slot],
# eliminating both eager index_elementwise_kernels (~10.5-11.5 us/step, 0-ULP bit-exact).
_DRAFT_STORE_TLS = threading.local()


def _install_hi_prio_capture(log) -> None:
    """Opt-in INKLING_CAPTURE_HI_PRIO=1: capture CUDA graphs on a highest-priority stream
    when caller omits stream=, so main-stream graph nodes outrank priority-0 side streams."""
    if getattr(torch.cuda.graph, "_inkling_hi_prio", False):
        return
    try:
        _, greatest = torch.cuda.Stream.priority_range()
        if greatest >= 0:
            return
        orig_init = torch.cuda.graph.__init__
        hi_streams: dict[int, torch.cuda.Stream] = {}

        def _hi_graph_init(self, cuda_graph, pool=None, stream=None, *args, **kwargs):
            if stream is None and "stream" not in kwargs:
                dev = torch.cuda.current_device()
                st = hi_streams.get(dev)
                if st is None:
                    st = torch.cuda.Stream(device=dev, priority=greatest)
                    hi_streams[dev] = st
                stream = st
            return orig_init(self, cuda_graph, pool, stream, *args, **kwargs)

        torch.cuda.graph.__init__ = _hi_graph_init
        torch.cuda.graph._inkling_hi_prio = True
        log.info(
            "INKLING_CAPTURE_HI_PRIO=1: torch.cuda.graph default capture stream -> priority %d",
            greatest,
        )
    except Exception as e:  # noqa: BLE001
        log.warning("INKLING_CAPTURE_HI_PRIO install failed (%r); using default priority 0", e)

# Measured GEMM-only step time (ms) by verified tokens T = W + 1 (index T-1),
# rank 0, 8-token verify, G1 prompts: the cost model of the "ema" policy.
_STEP_MS = (24.97, 28.48, 31.37, 33.76, 35.75, 37.41, 38.81, 40.0)
_EMA_ALPHA = 0.2  # critic: alpha >= 0.3 hurts code
_EMA_DECAY = 0.02  # unobserved positions drift to the prior, so they get probed
_EMA_PRIOR = 0.7
_PRIOR_LOGIT = 0.8472978603872037  # logit(0.7)
# Confidence policy defaults (no calibration file): c_k = p_k (the drafter's top-1 probability
# taken at face value), beta 0.5 on the centred history logit for conf+ema.
_CONF_BETA_DEFAULT = 0.5
_CF = 16  # floats per slot in the per-slot confidence record: p[0..6], W @7, hist[8..14]
_LOG_NF = 20  # floats per logged (step, request): see _vw_log_kernel


def _parse_verify_w() -> tuple[str, int]:
    """INKLING_VERIFY_W: unset/0/off -> ("off", 7); "ema"; "force:K"; "conf"; "conf+ema"; "dink"."""
    raw = os.environ.get("INKLING_VERIFY_W", "").strip().lower()
    if raw in ("", "0", "off"):
        return "off", 7
    if raw == "ema":
        return "ema", 7
    if raw in ("conf", "conf+ema", "dink"):
        return raw, 7
    if raw.startswith("force:"):
        return "force", max(0, min(7, int(raw[6:])))
    raise ValueError(f"INKLING_VERIFY_W={raw!r}: expected 0, ema, conf, conf+ema, dink or force:K")


def _parse_batch_schedule_int(
    raw: str, max_b: int, default_val: int, lo: int = 0, hi: int = 7
) -> list[int]:
    """Parse '1:7,2:5,4:3,8:1,16:0' into a 1-indexed dense lookup [0..max_b].
    Each entry 'B_max:val' applies up to B_max (ascending breakpoints; tail carries forward)."""
    out = [default_val] * (max_b + 1)
    s = (raw or "").strip()
    if not s:
        return out
    pts: list[tuple[int, int]] = []
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        b_str, v_str = item.split(":", 1)
        b = max(1, int(b_str.strip()))
        v = max(lo, min(hi, int(v_str.strip())))
        pts.append((b, v))
    if not pts:
        return out
    pts.sort(key=lambda kv: kv[0])
    prev_b = 1
    last_v = default_val
    for b_max, v in pts:
        for b in range(prev_b, min(b_max, max_b) + 1):
            out[b] = v
        prev_b = max(prev_b, b_max + 1)
        last_v = v
    for b in range(prev_b, max_b + 1):
        out[b] = last_v
    return out


def _parse_batch_schedule_float(
    raw: str, max_b: int, default_val: float, *, interp: bool = False
) -> list[float]:
    """Parse '1:-1,2:0.26,4:0.34,8:0.32' into a 1-indexed dense lookup [0..max_b].
    Default (interp=False): right-inclusive piecewise-constant breakpoints.
    Opt-in (interp=True): piecewise-linear interpolation between non-negative (>=0)
    breakpoints (never interpolates across negative sentinel entries such as 1:-1)."""
    out = [float(default_val)] * (max_b + 1)
    s = (raw or "").strip()
    if not s:
        return out
    pts: list[tuple[int, float]] = []
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        b_str, v_str = item.split(":", 1)
        pts.append((max(1, int(b_str.strip())), float(v_str.strip())))
    if not pts:
        return out
    pts.sort(key=lambda kv: kv[0])
    if not interp:
        prev_b = 1
        last_v = float(default_val)
        for b_max, v in pts:
            for b in range(prev_b, min(b_max, max_b) + 1):
                out[b] = v
            prev_b = max(prev_b, b_max + 1)
            last_v = v
        for b in range(prev_b, max_b + 1):
            out[b] = last_v
        return out
    # Piecewise-linear interpolation between non-negative breakpoints (dedup keeping last per B)
    dedup: dict[int, float] = {}
    for b_pt, v_pt in pts:
        dedup[b_pt] = v_pt
    uniq = sorted(dedup.items())
    first_b, first_v = uniq[0]
    for b in range(1, min(first_b, max_b) + 1):
        out[b] = first_v
    for i in range(1, len(uniq)):
        b0, v0 = uniq[i - 1]
        b1, v1 = uniq[i]
        for b in range(b0 + 1, min(b1, max_b) + 1):
            if v0 < 0.0 or v1 < 0.0 or b1 <= b0:
                out[b] = v1
            else:
                frac = (b - b0) / float(b1 - b0)
                out[b] = v0 + frac * (v1 - v0)
    last_b, last_v = uniq[-1]
    for b in range(last_b + 1, max_b + 1):
        out[b] = last_v
    return out


def _build_graph_k_table(
    dense_k: list[int], capture_sizes: list[int] | None, max_b: int, n_spec: int
) -> dict[int, int]:
    """For each draft CUDA graph capture size G, return the max K required by any
    runtime batch size b that replays graph G (i.e. prev_G < b <= G).
    Guarantees a replaying batch of size b <= G always gets at least dense_k[b] drafts."""
    if not capture_sizes:
        return {b: dense_k[b] for b in range(1, max_b + 1)}
    sizes = sorted(set(s for s in capture_sizes if 1 <= s <= max_b))
    table: dict[int, int] = {}
    prev = 0
    for g in sizes:
        lo = prev + 1
        hi = min(g, max_b)
        k_req = max((dense_k[b] for b in range(lo, hi + 1)), default=n_spec)
        table[g] = max(0, min(n_spec, k_req))
        prev = g
    return table


_WF_MAX_B = 32
_WF_COLS = 132  # [0..128]=T_step(B, M_pad, B+D, ctx=0), 129=glb_ms/tok, 130=swa_ms/tok, 131=T_2k_ref


def _wf_expected_unique_experts(b_active: int, w_live_per_req: float) -> float:
    """Calibrated GB10 Inkling-2-Small unique experts per layer (matches gb10_roofline_v10.py)."""
    b = max(1, int(b_active))
    w = max(1.0, float(w_live_per_req))
    w_eff = 1.0 + 0.80 * ((w - 1.0) ** 0.65)
    n_draws = b * 8.0 * w_eff
    # Power-law domain-core distribution: e_core=168 (alpha=0.45, mass=0.935) + 216 uniform tail
    core_w = [(i + 1.0) ** (-0.45) for i in range(168)]
    s_core = sum(core_w)
    u = 0.0
    for cw in core_w:
        pe = 0.935 * (cw / s_core)
        u += 1.0 - math.exp(-n_draws * pe)
    pe_tail = 0.065 / 216.0
    u += 216.0 * (1.0 - math.exp(-n_draws * pe_tail))
    return max(8.0, min(384.0, u))


def _build_gb10_waterfill_table(
    max_b: int,
    draft_k_dense: list[int],
    *,
    kv_bf12: bool = True,
    dspark_k: int = 7,
) -> list[list[float]]:
    """Precompute exact calibrated GB10 Wave-V10 step-time table `tbl[B, 0..131]` once at init.
    Cols 0..128: `T_step(B, M_pad(B), M_live=B+D, ctx=0)` (ms) for D = 0..128.
    Col 129: Global attention marginal ms per KV token (8 global layers, KV_BF12 or BF16).
    Col 130: SWA attention marginal ms per SWA-capped KV token (40 SWA layers).
    Col 131: Reference 2K step time (ms) for standalone `INKLING_CTX_ADAPTIVE_TAU=1` scaling.
    """
    n_rows = _WF_MAX_B + 1
    tbl = [[10.0] * _WF_COLS for _ in range(n_rows)]
    kv_comp = (0.750 * 1.045) if kv_bf12 else 1.00
    # Per-KV-token DRAM bytes across 1 KV head, head_dim=128, K+V bf16 (512 B/tok/layer)
    # 8 global layers @ ~218 GB/s; 40 SWA layers @ ~215 GB/s
    glb_ms_per_tok = (8.0 * 512.0 * kv_comp / 1e6) / 218.0
    swa_ms_per_tok = (40.0 * 512.0 * kv_comp / 1e6) / 215.0
    for b in range(1, n_rows):
        b_idx = min(b, max(1, max_b))
        k_sched = min(dspark_k, draft_k_dense[b_idx]) if b_idx < len(draft_k_dense) else dspark_k
        m_static = b * (k_sched + 1)
        tile_eff = 0.96 if m_static <= 16 else (1.00 if m_static < 64 else (1.06 if m_static < 160 else 1.09))
        nvfp4_bw = min(244.0 * 0.935, 244.0 * 0.905 * tile_eff)
        mx7_bw = min(244.0 * 0.935, 244.0 * 0.908 * min(1.06, tile_eff))
        l2pf_moe = (3.20 * min(2.0, m_static / 128.0)) if m_static >= 64 else (0.40 if m_static <= 16 else 0.0)
        l2_spill = 0.015 * max(0.0, m_static - 128.0)
        v10_save = (0.18 if m_static <= 16 else (0.24 if m_static <= 64 else 0.20)) + (
            0.065 if m_static <= 32 else (0.045 if m_static <= 64 else 0.030)
        ) + (0.055 if m_static > 48 else 0.0)
        # Non-MoE ctx=0 fixed per-bucket baseline matching gb10_roofline_v10.py Wave-V10B anchors
        router_ms = max(0.11, (119.0 / (244.0 * 0.87) + 0.144 + 0.004 * m_static - 0.45) - (
            0.185 if m_static <= 8 else (0.215 if m_static <= 16 else (0.175 if m_static <= 32 else (0.145 if m_static <= 64 else 0.0)))
        ))
        dense_cold = (3.72 + 0.0095 * m_static) if m_static <= 128 else (6.20 + 0.014 * m_static)
        dense_l2 = 1.65 if m_static <= 16 else (1.50 if m_static <= 64 else 2.30)
        dense_ms = max(1.50, dense_cold - dense_l2 - (0.10 if m_static <= 16 else 0.0)) + 0.010
        attn_fixed_ms = 40 * 0.008 + 8 * 0.010 + (0.14 + 0.004 * m_static)
        nbytes_ar = m_static * 4608.0 * 2.0
        if m_static <= 16:
            rdma_ms = max(1.90, 96.0 * (17.5 + (7.0 / 8.0) * 2.0 * nbytes_ar / 1000.0 / 16.5 + 3.2) / 1000.0 - 0.045)
        else:
            use_dqp = (nbytes_ar / 8.0) >= 16384.0
            wf_wire = 0.5 * (0.76 if m_static >= 42 else 1.0) + 0.5 * (0.76 if m_static >= 18 else 1.0)
            bw_r = 22.8 if use_dqp else 14.5
            w_us = (7.0 / 8.0) * 2.0 * (nbytes_ar * wf_wire) / 1000.0 / bw_r
            g_us = 0.0085 * (nbytes_ar / 1000.0)
            ovl = (0.45 * min(w_us, g_us + 0.45 * w_us) - 3.6) if m_static >= 42 else 0.0
            rdma_ms = max(1.90, 96.0 * (21.0 + w_us + g_us - ovl) / 1000.0 - 0.075)
        sconv_ms = 0.68 + (96.0 * 4.5 * nbytes_ar / 1e6) / 620.0 + 0.008 * m_static
        cert_ms = (
            max(0.32, (91.11 / 225.0) + 0.0042 * m_static - 0.005)
            if m_static <= 128
            else (197.0 / 220.0) + 0.0038 * m_static + 0.055
        )
        draft_ms = (
            0.05
            if k_sched <= 0
            else max(
                0.32,
                (85.27 / 208.0) * (1.0 + 0.008 * b)
                + (61.51 * 0.5 / 218.0) * (1.0 + 0.006 * b * k_sched / 8.0)
                + 0.055
                + 10.0 * (0.024 + 0.0014 * b)
                + k_sched * (20.0 + 4.8 * b) / 1000.0
                - 0.16,
            )
        )
        host_ms = 0.55 if b == 1 else (0.65 if b <= 8 else (0.82 if b <= 16 else 1.06))
        fixed_ms = router_ms + dense_ms + attn_fixed_ms + rdma_ms + sconv_ms + cert_ms + draft_ms + host_ms

        for d in range(129):
            m_live = float(min(m_static, b + d))
            u_exp = _wf_expected_unique_experts(b, m_live / float(b))
            # Exact TP8 per-expert shard bytes across 40L NVFP4 (85.847 MB) & 8L MX7/MX6 (29.447 MB * 0.9383)
            nv_mb = u_exp * 85.84704
            mx_mb = u_exp * 29.446784 * 0.938307
            nv_ms = max(0.5, nv_mb / nvfp4_bw + 0.022 * m_live + l2_spill * (40.0 / 48.0) + 0.22 - l2pf_moe * 0.80 - v10_save)
            mx_ms = max(0.2, mx_mb / mx7_bw + 0.006 * m_live + l2_spill * (8.0 / 48.0) + 0.052 - l2pf_moe * 0.20)
            tbl[b][d] = float(fixed_ms + nv_ms + mx_ms)
        # Global attn mult for B=1/2 small-B non-FUSE_MT vs FUSE_MT=1 (1.0x at long ctx B>=2!)
        tbl[b][129] = float(glb_ms_per_tok * (1.08 if b == 1 else 1.0))
        tbl[b][130] = float(swa_ms_per_tok)
        d_ref = min(128, int(round(b * min(3.5, float(k_sched)))))
        tbl[b][131] = float(tbl[b][d_ref] + b * 2048.0 * tbl[b][129] + b * 1024.0 * tbl[b][130])
    return tbl


def _load_cal(k: int) -> dict:
    """INKLING_CONF_CAL=<json>: {"a": [7], "b": [7], "beta": f, "tau": f, "step_ms": [8],
    "step_eff_ms": [8], "mk_step_ms": f, "dink": {...}} (written by calibrate_s5.py).
    Missing keys keep the defaults."""
    cal = {
        "a": [1.0] * k,
        "b": [0.0] * k,
        "beta": None,
        "tau": 0.0,
        "step_ms": list(_STEP_MS),
        "step_eff_ms": None,
        "mk_step_ms": 0.030,
        "dink": {
            "eta_b": 0.10,
            "decay_b": 0.002,
            "b_clip": 1.5,
            "eta_lam": 0.03,
            "gamma": 1.0,
            "init_tok": 2.8,
            "init_ms": None,
            "static_lam": 0.0,
        },
    }
    path = os.environ.get("INKLING_CONF_CAL", "").strip()
    if path:
        with open(path) as f:
            user = json.load(f)
        for key in ("a", "b", "beta", "tau", "step_ms", "step_eff_ms", "mk_step_ms"):
            if key in user and user[key] is not None:
                cal[key] = user[key]
        if isinstance(user.get("dink"), dict):
            cal["dink"].update({dk: dv for dk, dv in user["dink"].items() if dv is not None})
        for key in ("a", "b"):
            v = [float(x) for x in cal[key]][:k]
            cal[key] = v + [v[-1] if v else (1.0 if key == "a" else 0.0)] * (k - len(v))
        if len(cal["step_ms"]) != 8:
            raise ValueError("INKLING_CONF_CAL step_ms must have 8 entries (T=1..8 verified tokens)")
        if cal["step_eff_ms"] is not None and len(cal["step_eff_ms"]) != 8:
            raise ValueError("INKLING_CONF_CAL step_eff_ms must have 8 entries")
    if cal["step_eff_ms"] is None:
        mk = float(cal["mk_step_ms"])
        cal["step_eff_ms"] = [
            float(cal["step_ms"][w]) - mk * max(0, 7 - min(w + 1, 7)) for w in range(8)
        ]
    if cal["dink"]["init_ms"] is None:
        cal["dink"]["init_ms"] = float(cal["step_eff_ms"][2])
    if os.environ.get("INKLING_CONF_TAU", "").strip():
        cal["tau"] = float(os.environ["INKLING_CONF_TAU"])
    if os.environ.get("INKLING_DINK_LAMBDA", "").strip():
        cal["dink"]["static_lam"] = float(os.environ["INKLING_DINK_LAMBDA"])
    return cal


@triton.jit
def _vw_update_kernel(
    idx_ptr,  # [num_reqs] batch row -> request state slot
    num_sampled_ptr,  # [num_reqs] accepted drafts + 1 (0 on a chunked prefill)
    num_rejected_ptr,  # [num_reqs] verified rows - num_sampled
    temp_ptr,  # [max_num_reqs] sampling temperature by slot
    w_ptr,  # [max_num_reqs] int32 verify width W by slot (in/out)
    c_ptr,  # [max_num_reqs, K] fp32 conditional acceptance EMA (in/out)
    step_ptr,  # [8] fp32 step cost by verified tokens
    stats_ptr,  # [10] int32 (hist of W used, steps, accepted drafts) or dummy
    ALPHA: tl.constexpr,
    DECAY: tl.constexpr,
    PRIOR: tl.constexpr,
    K: tl.constexpr,
    STATS: tl.constexpr,
    WRITE_W: tl.constexpr = True,  # False: only the EMA is updated (conf / force / shadow)
    FIXED_USED: tl.constexpr = False,  # True: every draft was verified (shadow logging)
):
    r = tl.program_id(0)
    slot = tl.load(idx_ptr + r).to(tl.int64)
    ns = tl.load(num_sampled_ptr + r).to(tl.int32)
    nr = tl.load(num_rejected_ptr + r).to(tl.int32)
    temp = tl.load(temp_ptr + slot)
    w_used = tl.load(w_ptr + slot)
    if FIXED_USED:
        w_used = w_used * 0 + K
    nd = ns + nr - 1  # drafts that were in the verify batch
    acc = ns - 1  # accepted drafts
    active = (ns >= 1) & (nd >= 1) & (temp == 0.0)

    k = tl.arange(0, 8)
    kmask = k < K
    c = tl.load(c_ptr + slot * K + k, mask=kmask, other=PRIOR)
    # Draft k is observed when every earlier draft was accepted and it was
    # actually verified (k < min(W used, drafts present)).
    obs = (k <= acc) & (k < tl.minimum(w_used, nd))
    hit = (k < acc).to(tl.float32)
    c_new = tl.where(obs, c + ALPHA * (hit - c), c + DECAY * (PRIOR - c))
    tl.store(c_ptr + slot * K + k, c_new, mask=kmask & active)
    c_new = tl.where(active, c_new, c)

    # W = argmax_w (1 + sum_{k<w} prod_{j<=k} c_j) / step_ms[w]  (w verified drafts)
    surv = 1.0
    tokens = 1.0
    best_w = 0
    best_v = tokens / tl.load(step_ptr)
    for kk in tl.static_range(K):
        ck = tl.sum(tl.where(k == kk, c_new, 0.0), axis=0)
        surv = surv * ck
        tokens = tokens + surv
        v = tokens / tl.load(step_ptr + kk + 1)
        better = v > best_v
        best_v = tl.where(better, v, best_v)
        best_w = tl.where(better, kk + 1, best_w)
    # Sampling requests (temperature > 0) always verify everything: a masked
    # draft would be rejected without the target distribution being resampled
    # for it, so only greedy requests are width-limited.
    w_new = tl.where(temp == 0.0, tl.where(active, best_w, w_used), K)
    if WRITE_W:
        tl.store(w_ptr + slot, w_new.to(tl.int32))
    if STATS:
        if active:
            tl.atomic_add(stats_ptr + w_used, 1)
            tl.atomic_add(stats_ptr + 8, 1)
            tl.atomic_add(stats_ptr + 9, acc)


@triton.jit(do_not_specialize=["w_cap", "req_draft_stride"])
def _vw_mask_kernel(
    draft_ptr,  # [max_num_reqs, K] int64 draft tokens (in/out)
    draft_stride,
    idx_ptr,  # [num_reqs]
    temp_ptr,
    w_ptr,
    w_cap,  # int32 batch-aware upper bound on W (0..K)
    req_draft_ptr,  # [max_num_reqs, K] req_states.draft_tokens (optional direct write target)
    req_draft_stride,
    K: tl.constexpr,
    WRITE_REQ_DRAFTS: tl.constexpr = False,
    CLEAR_REQ_TAIL: tl.constexpr = False,
    REQ_K: tl.constexpr = 7,
):
    r = tl.program_id(0)
    slot = tl.load(idx_ptr + r).to(tl.int64)
    w = tl.minimum(tl.load(w_ptr + slot), w_cap)
    temp = tl.load(temp_ptr + slot)
    k = tl.arange(0, 8)
    kmask = k < K
    m = (k >= w) & kmask & (temp == 0.0)
    if WRITE_REQ_DRAFTS:
        tok = tl.load(draft_ptr + r * draft_stride + k, mask=kmask, other=-1)
        tok = tl.where(m, -1, tok)
        tl.store(draft_ptr + r * draft_stride + k, tok, mask=kmask)
        tl.store(req_draft_ptr + slot * req_draft_stride + k, tok, mask=kmask)
        if CLEAR_REQ_TAIL and REQ_K > K:
            for e in tl.static_range(REQ_K - K):
                tl.store(draft_ptr + r * draft_stride + K + e, -1)
                tl.store(req_draft_ptr + slot * req_draft_stride + K + e, -1)
    else:
        tl.store(draft_ptr + r * draft_stride + k, -1, mask=m)


@triton.jit
def _vw_update_dink_kernel(
    idx_ptr,  # [num_reqs]
    num_sampled_ptr,  # [num_reqs]
    num_rejected_ptr,  # [num_reqs]
    temp_ptr,  # [max_num_reqs]
    w_ptr,  # [max_num_reqs] int32 W used
    cpred_ptr,  # [max_num_reqs, K] fp32 predicted conditional acceptance c[0..6]
    db_ptr,  # [max_num_reqs] fp32 online logit bias Delta-b (in/out)
    etok_ptr,  # [max_num_reqs] fp32 EMA of tokens emitted per step (in/out)
    ems_ptr,  # [max_num_reqs] fp32 EMA of step cost ms (in/out)
    step_ptr,  # [8] fp32 effective step cost by W
    stats_ptr,  # [10] int32 stats or dummy
    cal_ptr,  # [24] fp32 calibration table (used when FUSE_ES=True)
    es_skip_ptr,
    es_surv_ptr,
    es_tok_ptr,
    es_bestu_ptr,
    es_greedy_ptr,
    es_slot_ptr,
    es_db_ptr,
    es_lam_ptr,
    ETA_B: tl.constexpr,
    DECAY_B: tl.constexpr,
    B_CLIP: tl.constexpr,
    ETA_LAM: tl.constexpr,
    K: tl.constexpr,
    STATS: tl.constexpr,
    FUSE_ES: tl.constexpr = False,
):
    """Exact zero-bias martingale Online Newton update for per-request Delta-b and
    EMA throughput (etok/ems) for adaptive Dinkelbach verify-W."""
    r = tl.program_id(0)
    slot = tl.load(idx_ptr + r).to(tl.int64)
    ns = tl.load(num_sampled_ptr + r).to(tl.int32)
    nr = tl.load(num_rejected_ptr + r).to(tl.int32)
    temp = tl.load(temp_ptr + slot)
    w_used = tl.load(w_ptr + slot)
    nd = ns + nr - 1
    acc = ns - 1
    active = (ns >= 1) & (nd >= 1) & (temp == 0.0)

    k = tl.arange(0, 8)
    kmask = k < K
    cpred = tl.load(cpred_ptr + slot * K + k, mask=kmask, other=0.7)
    obs = (k <= acc) & (k < tl.minimum(w_used, nd)) & kmask
    hit = (k < acc).to(tl.float32)
    n_obs = tl.sum(obs.to(tl.int32), axis=0)
    g = tl.sum(tl.where(obs, hit - cpred, 0.0), axis=0)
    h = tl.sum(tl.where(obs, cpred * (1.0 - cpred), 0.0), axis=0)

    db_old = tl.load(db_ptr + slot)
    db_obs = db_old * (1.0 - DECAY_B) + g / ((1.0 / ETA_B) + h)
    db_none = db_old * (1.0 - tl.maximum(DECAY_B, 0.02))
    db_new = tl.where(n_obs > 0, db_obs, db_none)
    db_new = tl.minimum(tl.maximum(db_new, -B_CLIP), B_CLIP)

    w_clamped = tl.minimum(tl.maximum(w_used, 0), K)
    step_c = tl.load(step_ptr + w_clamped)
    etok = tl.load(etok_ptr + slot)
    ems = tl.load(ems_ptr + slot)
    etok_new = etok + ETA_LAM * (ns.to(tl.float32) - etok)
    ems_new = ems + ETA_LAM * (step_c - ems)

    db_out = tl.where(active, db_new, db_old)
    etok_out = tl.where(active, etok_new, etok)
    ems_out = tl.where(active, ems_new, ems)
    tl.store(db_ptr + slot, db_out)
    tl.store(etok_ptr + slot, etok_out)
    tl.store(ems_ptr + slot, ems_out)
    if FUSE_ES:
        # 0-ULP bit-identical fusion of _es_init_dink_kernel for B=1 (zero extra launch inside TD).
        if r == 0:
            s0 = tl.load(step_ptr)
            static_lam = tl.load(cal_ptr + 18)
            gamma = tl.load(cal_ptr + 19)
            ems_cl = tl.maximum(ems_out, 1.0)
            lam = tl.where(static_lam > 0.0, static_lam, gamma * (etok_out / ems_cl))
            tl.store(es_skip_ptr, 0)
            tl.store(es_surv_ptr, 1.0)
            tl.store(es_tok_ptr, 1.0)
            tl.store(es_bestu_ptr, 1.0 - lam * s0)
            tl.store(es_greedy_ptr, tl.where(temp == 0.0, 1, 0))
            tl.store(es_slot_ptr, slot.to(tl.int32))
            tl.store(es_db_ptr, db_out)
            tl.store(es_lam_ptr, lam)
    if STATS:
        if active:
            tl.atomic_add(stats_ptr + w_clamped, 1)
            tl.atomic_add(stats_ptr + 8, 1)
            tl.atomic_add(stats_ptr + 9, acc)


@triton.jit(do_not_specialize=["w_cap", "tau_override", "req_draft_stride", "num_reqs_rt", "ctx_tau_floor"])
def _vw_conf_kernel(
    idx_ptr,  # [num_reqs] batch row -> request state slot
    conf_ptr,  # [rows, conf_stride] fp32 drafter top-1 probability by batch row and draft position
    conf_stride,
    temp_ptr,  # [max_num_reqs] sampling temperature by slot
    w_ptr,  # [max_num_reqs] int32 verify width by slot (out)
    c_ptr,  # [max_num_reqs, K] fp32 conditional acceptance EMA (history prior)
    cal_ptr,  # [24] fp32: a[0:8], b[8:16], beta @16, tau @17, static_lam @18, gamma @19, mk_ms @20
    step_ptr,  # [8] fp32 step cost by verified tokens
    draft_ptr,  # [max_num_reqs, K] int64 draft tokens (in/out)
    draft_stride,
    cslot_ptr,  # [max_num_reqs, 16] fp32 per-slot record for the logger (or dummy)
    db_ptr,  # [max_num_reqs] fp32 online logit bias Delta-b (dink mode) or dummy
    etok_ptr,  # [max_num_reqs] fp32 EMA tokens/step (dink mode) or dummy
    ems_ptr,  # [max_num_reqs] fp32 EMA ms/step (dink mode) or dummy
    cpred_ptr,  # [max_num_reqs, K] fp32 predicted c[0..6] output (dink mode) or dummy
    w_cap,  # int32 batch-aware max verify width (0..K)
    tau_override,  # fp32 batch-aware tau override (< 0 -> use cal_ptr[17] / dink step_ms; >= 0 -> dink/conf surv >= tau)
    req_draft_ptr,  # [max_num_reqs, K] req_states.draft_tokens (optional direct write target)
    req_draft_stride,
    seq_lens_ptr,  # [max_num_reqs] int32 live seq_lens (used only when CTX_ADAPTIVE=True)
    wf_tbl_ptr,  # [WF_MAX_B+1, WF_COLS] fp32 GB10 roofline table (used only when CTX_ADAPTIVE=True)
    num_reqs_rt,  # int32 batch rows in launch (used only when CTX_ADAPTIVE=True)
    ctx_tau_floor,  # fp32 min floor for context-scaled tau (used only when CTX_ADAPTIVE=True)
    PRIOR_LOGIT: tl.constexpr,
    K: tl.constexpr,
    USE_HIST: tl.constexpr,
    USE_DINK: tl.constexpr,
    LOG: tl.constexpr,
    WRITE_REQ_DRAFTS: tl.constexpr = False,
    IDX_STRIDE: tl.constexpr = 1,
    CLEAR_REQ_TAIL: tl.constexpr = False,
    REQ_K: tl.constexpr = 7,
    CTX_ADAPTIVE: tl.constexpr = False,
    WF_COLS: tl.constexpr = 132,
    WF_MAX_B: tl.constexpr = 32,
):
    """W per request from the drafter confidences, capped by batch-aware w_cap, then the -1 tail mask.
    When w_cap == K, tau_override < 0, and CTX_ADAPTIVE=False (B=1 default), arithmetic is bit-identical.
    When CTX_ADAPTIVE=True and tau_override >= 0 (INKLING_CTX_ADAPTIVE_TAU=1), dynamically scales
    `tau_eff = max(ctx_tau_floor, tau_override * T_2k(B) / T_ctx(B, sum_kv))` on GPU with 0 host sync."""
    r = tl.program_id(0)
    slot = tl.load(idx_ptr + r * IDX_STRIDE).to(tl.int64)
    if slot < 0:
        return
    temp = tl.load(temp_ptr + slot)
    k = tl.arange(0, 8)
    kmask = k < K
    p = tl.load(conf_ptr + r * conf_stride + k, mask=kmask, other=0.5)
    pc = tl.minimum(tl.maximum(p, 1e-6), 1.0 - 1e-6)
    a = tl.load(cal_ptr + k, mask=kmask, other=1.0)
    b = tl.load(cal_ptr + 8 + k, mask=kmask, other=0.0)
    beta = tl.load(cal_ptr + 16)
    tau_cal = tl.load(cal_ptr + 17)
    eff_tau_ov = tau_override
    if CTX_ADAPTIVE:
        if tau_override >= 0.0:
            rb = tl.arange(0, 32)
            rm = rb < tl.minimum(num_reqs_rt, WF_MAX_B)
            sl_b = tl.load(idx_ptr + rb * IDX_STRIDE, mask=rm, other=-1)
            v_b = rm & (sl_b >= 0)
            b_live = tl.maximum(1, tl.minimum(tl.sum(v_b.to(tl.int32), axis=0), WF_MAX_B))
            seq_b = tl.load(seq_lens_ptr + rb, mask=v_b, other=0).to(tl.float32)
            sum_kv = tl.sum(tl.where(v_b, tl.maximum(seq_b, 0.0), 0.0), axis=0)
            t_2k = tl.load(wf_tbl_ptr + b_live * WF_COLS + 131)
            kv_ms = tl.load(wf_tbl_ptr + b_live * WF_COLS + 129)
            extra_kv = tl.maximum(0.0, sum_kv - 2048.0 * b_live.to(tl.float32))
            t_ctx = t_2k + extra_kv * kv_ms
            eff_tau_ov = tl.maximum(ctx_tau_floor, tau_override * (t_2k / tl.maximum(t_ctx, 0.1)))
    tau = tl.where(eff_tau_ov >= 0.0, eff_tau_ov, tau_cal)
    z = a * (tl.log(pc) - tl.log(1.0 - pc)) + b
    h = tl.load(c_ptr + slot * K + k, mask=kmask, other=0.7)
    if USE_DINK:
        db = tl.load(db_ptr + slot)
        z = z + db
    elif USE_HIST:
        hc = tl.minimum(tl.maximum(h, 1e-4), 1.0 - 1e-4)
        z = z + beta * (tl.log(hc) - tl.log(1.0 - hc) - PRIOR_LOGIT)
    c = 1.0 / (1.0 + tl.exp(-z))
    if USE_DINK:
        tl.store(cpred_ptr + slot * K + k, c, mask=kmask)

    surv = 1.0
    tokens = 1.0
    best_w = 0
    if USE_DINK:
        static_lam = tl.load(cal_ptr + 18)
        gamma = tl.load(cal_ptr + 19)
        etok = tl.load(etok_ptr + slot)
        ems = tl.maximum(tl.load(ems_ptr + slot), 1.0)
        lam = tl.where(static_lam > 0.0, static_lam, gamma * (etok / ems))
        best_v = tokens - lam * tl.load(step_ptr)
        alive = 1
        w_tau = 0
        for kk in tl.static_range(K):
            ck = tl.sum(tl.where(k == kk, c, 0.0), axis=0)
            surv = surv * ck
            tokens = tokens + surv
            v = tokens - lam * tl.load(step_ptr + kk + 1)
            better = (v > best_v) & (kk < w_cap)
            best_v = tl.where(better, v, best_v)
            best_w = tl.where(better, kk + 1, best_w)
            alive = tl.where((surv >= tau) & (kk < w_cap), alive, 0)
            w_tau = w_tau + alive
        # C=32 batch-marginal Dinkelbach: when tau_override >= 0 (B > 1 via INKLING_DINK_BATCH_TAU
        # / INKLING_CONF_TAU_BATCH), preserve online Newton db_slot calibration while thresholding
        # surv >= tau_override; when tau_override < 0 (B=1 default), arithmetic is 100% bit-identical!
        w_pick = tl.where(tau_override >= 0.0, w_tau, best_w)
    else:
        best_v = tokens / tl.load(step_ptr)
        alive = 1
        w_tau = 0
        for kk in tl.static_range(K):
            ck = tl.sum(tl.where(k == kk, c, 0.0), axis=0)
            surv = surv * ck
            tokens = tokens + surv
            v = tokens / tl.load(step_ptr + kk + 1)
            better = (v > best_v) & (kk < w_cap)
            best_v = tl.where(better, v, best_v)
            best_w = tl.where(better, kk + 1, best_w)
            alive = tl.where((surv >= tau) & (kk < w_cap), alive, 0)
            w_tau = w_tau + alive
        w_pick = tl.where(tau > 0.0, w_tau, best_w)
    w_pick = tl.minimum(w_pick, w_cap)
    w_new = tl.where(temp == 0.0, w_pick, K)
    tl.store(w_ptr + slot, w_new.to(tl.int32))
    m = (k >= w_new) & kmask & (temp == 0.0)
    if WRITE_REQ_DRAFTS:
        tok = tl.load(draft_ptr + r * draft_stride + k, mask=kmask, other=-1)
        tok = tl.where(m, -1, tok)
        tl.store(draft_ptr + r * draft_stride + k, tok, mask=kmask)
        tl.store(req_draft_ptr + slot * req_draft_stride + k, tok, mask=kmask)
        if CLEAR_REQ_TAIL and REQ_K > K:
            for e in tl.static_range(REQ_K - K):
                tl.store(draft_ptr + r * draft_stride + K + e, -1)
                tl.store(req_draft_ptr + slot * req_draft_stride + K + e, -1)
    else:
        tl.store(draft_ptr + r * draft_stride + k, -1, mask=m)
    if LOG:
        tl.store(cslot_ptr + slot * 16 + k, p, mask=kmask)
        tl.store(cslot_ptr + slot * 16 + 8 + k, h, mask=kmask)
        tl.store(cslot_ptr + slot * 16 + 7, w_new.to(tl.float32))


@triton.jit(do_not_specialize=["num_reqs_rt", "w_cap", "m_live_cap", "tau_min", "req_draft_stride"])
def _vw_hetero_waterfill_kernel(
    idx_ptr,  # [num_reqs * IDX_STRIDE] batch row -> request state slot (-1 on padded/dummy)
    conf_ptr,  # [rows, conf_stride] fp32 drafter top-1 probability
    conf_stride,
    temp_ptr,  # [max_num_reqs] fp32 temperature by slot
    w_ptr,  # [max_num_reqs] int32 verify width by slot (out)
    cal_ptr,  # [24] fp32 calibration table a[0:8], b[8:16]
    draft_ptr,  # [max_num_reqs, K] int64 draft tokens (in/out)
    draft_stride,
    db_ptr,  # [max_num_reqs] fp32 online Newton logit bias Delta-b
    cpred_ptr,  # [max_num_reqs, K] fp32 predicted c[0..6] output
    seq_lens_ptr,  # [max_num_reqs] int32 live seq_lens (persistent inside draft graph!)
    wf_tbl_ptr,  # [WF_MAX_B+1, WF_COLS] fp32 calibrated GB10 roofline step-time table
    req_draft_ptr,  # [max_num_reqs, REQ_K] req_states.draft_tokens (optional direct write)
    req_draft_stride,
    num_reqs_rt,  # int32 runtime rows (<= B_TILE)
    w_cap,  # int32 max draft width per req (0..K)
    m_live_cap,  # int32 optional live-token budget cap (0 = unconstrained exact ratio max)
    tau_min,  # fp32 optional minimum survival floor (0.0 = unconstrained)
    K: tl.constexpr,
    B_TILE: tl.constexpr,
    K_TILE: tl.constexpr,
    N_LANES: tl.constexpr,
    WRITE_REQ_DRAFTS: tl.constexpr = False,
    IDX_STRIDE: tl.constexpr = 1,
    CLEAR_REQ_TAIL: tl.constexpr = False,
    REQ_K: tl.constexpr = 7,
    WF_COLS: tl.constexpr = 132,
    WF_MAX_B: tl.constexpr = 32,
):
    """1-CTA Zero-Sync Exact Batch-Coupled Water-Filling Kernel (INKLING_HETERO_BUDGET_SPEC=1).

    Solves the EXACT nonlinear batch-level fractional program across all B requests simultaneously:
        argmax_{W_0..W_{B-1} in [0..w_cap]}  (B + sum_r sum_{k<W_r} S_{r,k}) / T_step(B, M_pad, B + sum_r W_r, sum_kv)
    via deterministic int32 packed-key bitonic sort (`tl.sort`), prefix-closed tie-breaking
    `(7 - k) << 8 | (255 - r)` (100% unique keys -> zero-atomic cutoff threshold, 100% TP8 lockstep!),
    exact calibrated GB10 coupon-collector MoE curve `wf_tbl[B_live, D]`, and live GPU `seq_lens`
    global+SWA attention cost. Strictly preserves `temp > 0` rows unmasked (100% target-lossless!)."""
    r_vec = tl.arange(0, B_TILE)
    r_in = r_vec < num_reqs_rt
    slot_vec = tl.load(idx_ptr + r_vec * IDX_STRIDE, mask=r_in, other=-1).to(tl.int64)
    valid_r = r_in & (slot_vec >= 0)
    slot_safe = tl.where(valid_r, slot_vec, 0)
    b_live = tl.sum(valid_r.to(tl.int32), axis=0)
    if b_live <= 0:
        return
    b_safe = tl.maximum(1, tl.minimum(b_live, WF_MAX_B))
    temp_vec = tl.load(temp_ptr + slot_safe, mask=valid_r, other=0.0)
    is_greedy_r = valid_r & (temp_vec == 0.0)
    is_temp_r = valid_r & (temp_vec != 0.0)
    db_vec = tl.load(db_ptr + slot_safe, mask=valid_r, other=0.0)
    seq_vec = tl.load(seq_lens_ptr + r_vec, mask=valid_r, other=0).to(tl.float32)
    sum_kv = tl.sum(tl.where(valid_r, tl.maximum(seq_vec, 0.0), 0.0), axis=0)
    sum_swa = tl.sum(tl.where(valid_r, tl.minimum(tl.maximum(seq_vec, 0.0), 1024.0), 0.0), axis=0)
    glb_ms_tok = tl.load(wf_tbl_ptr + b_safe * WF_COLS + 129)
    swa_ms_tok = tl.load(wf_tbl_ptr + b_safe * WF_COLS + 130)
    attn_kv_ms = sum_kv * glb_ms_tok + sum_swa * swa_ms_tok

    eff_w_cap = tl.minimum(tl.maximum(w_cap, 0), K)
    surv_vec = tl.full([B_TILE], 1.0, tl.float32)
    tok_temp = tl.zeros((), tl.float32)
    keys = tl.full([B_TILE, K_TILE], -1, tl.int32)
    k_cols = tl.arange(0, K_TILE)[None, :]
    for kk in tl.static_range(K):
        p_kk = tl.load(conf_ptr + r_vec * conf_stride + kk, mask=valid_r, other=0.5)
        pc_kk = tl.minimum(tl.maximum(p_kk, 1e-6), 1.0 - 1e-6)
        ak = tl.load(cal_ptr + kk)
        bk = tl.load(cal_ptr + 8 + kk)
        ck = 1.0 / (1.0 + tl.exp(-(ak * (tl.log(pc_kk) - tl.log(1.0 - pc_kk)) + bk + db_vec)))
        tl.store(cpred_ptr + slot_safe * K + kk, ck, mask=valid_r)
        surv_vec = surv_vec * ck
        tok_temp = tok_temp + tl.sum(tl.where(is_temp_r & (kk < eff_w_cap), surv_vec, 0.0), axis=0)
        q_kk = tl.minimum(1048575, tl.maximum(0, (surv_vec * 1048575.0 + 0.5).to(tl.int32)))
        packed_kk = (q_kk << 11) | ((7 - kk) << 8) | (255 - r_vec)
        keep_cand = is_greedy_r & (kk < eff_w_cap) & (surv_vec >= tau_min)
        key_kk = tl.where(keep_cand, packed_kk, -1)
        keys = tl.where(k_cols == kk, key_kk[:, None], keys)

    keys_1d = tl.reshape(keys, [N_LANES])
    sorted_keys = tl.sort(keys_1d, descending=True)
    n_greedy_valid = tl.sum((sorted_keys >= 0).to(tl.int32), axis=0)
    s_sorted = tl.where(sorted_keys >= 0, (sorted_keys >> 11).to(tl.float32) * (1.0 / 1048575.0), 0.0)
    cum_s = tl.cumsum(s_sorted, axis=0)

    d_fixed = tl.sum(is_temp_r.to(tl.int32), axis=0) * eff_w_cap
    tok_0 = b_live.to(tl.float32) + tok_temp
    d_idx = tl.arange(0, N_LANES) + 1
    d_tot_0 = tl.minimum(d_fixed, 128)
    d_tot_d = tl.minimum(d_fixed + d_idx, 128)
    t_0 = tl.load(wf_tbl_ptr + b_safe * WF_COLS + d_tot_0) + attn_kv_ms
    t_d = tl.load(wf_tbl_ptr + b_safe * WF_COLS + d_tot_d) + attn_kv_ms

    d_cap_eff = tl.where(m_live_cap > 0, tl.maximum(0, m_live_cap - b_live - d_fixed), N_LANES)
    d_ok = (d_idx <= n_greedy_valid) & (d_idx <= d_cap_eff)
    v_0 = tok_0 / tl.maximum(t_0, 0.1)
    v_d = tl.where(d_ok, (tok_0 + cum_s) / tl.maximum(t_d, 0.1), -1.0e30)
    best_v_pos = tl.max(v_d, axis=0)
    best_d_pos = tl.min(tl.where(v_d == best_v_pos, d_idx, N_LANES + 1), axis=0)
    best_d = tl.where((best_v_pos > v_0) & (best_d_pos <= N_LANES), best_d_pos, 0)

    cutoff_key = tl.sum(tl.where(d_idx == best_d, sorted_keys, 0), axis=0)
    live_2d = (best_d > 0) & (keys >= 0) & (keys >= cutoff_key)
    w_greedy = tl.sum(live_2d.to(tl.int32), axis=1)
    w_new = tl.where(is_greedy_r, tl.minimum(w_greedy, eff_w_cap), eff_w_cap)
    tl.store(w_ptr + slot_safe, w_new, mask=valid_r)

    for kk in tl.static_range(K):
        mask_out = is_greedy_r & (kk >= w_new)
        if WRITE_REQ_DRAFTS:
            tok = tl.load(draft_ptr + r_vec * draft_stride + kk, mask=valid_r, other=-1)
            tok = tl.where(mask_out, -1, tok)
            tl.store(draft_ptr + r_vec * draft_stride + kk, tok, mask=valid_r)
            tl.store(req_draft_ptr + slot_safe * req_draft_stride + kk, tok, mask=valid_r)
        else:
            tl.store(draft_ptr + r_vec * draft_stride + kk, -1, mask=mask_out)
    if WRITE_REQ_DRAFTS and CLEAR_REQ_TAIL and REQ_K > K:
        for e in tl.static_range(REQ_K - K):
            tl.store(draft_ptr + r_vec * draft_stride + K + e, -1, mask=valid_r)
            tl.store(req_draft_ptr + slot_safe * req_draft_stride + K + e, -1, mask=valid_r)


@triton.jit
def _vw_store_conf_kernel(
    idx_ptr, conf_ptr, conf_stride, w_ptr, c_ptr, cslot_ptr, K: tl.constexpr, USE_W: tl.constexpr
):
    """Logging only (modes without the conf policy): copy this step's confidences by batch row
    into the per-slot record, so the next step's verification can be paired with them."""
    r = tl.program_id(0)
    slot = tl.load(idx_ptr + r).to(tl.int64)
    k = tl.arange(0, 8)
    kmask = k < K
    p = tl.load(conf_ptr + r * conf_stride + k, mask=kmask, other=0.0)
    h = tl.load(c_ptr + slot * K + k, mask=kmask, other=0.0)
    tl.store(cslot_ptr + slot * 16 + k, p, mask=kmask)
    tl.store(cslot_ptr + slot * 16 + 8 + k, h, mask=kmask)
    w = tl.load(w_ptr + slot).to(tl.float32)
    if not USE_W:
        w = w * 0.0 + K
    tl.store(cslot_ptr + slot * 16 + 7, w)


@triton.jit
def _vw_log_kernel(
    idx_ptr,  # [num_reqs]
    num_sampled_ptr,
    num_rejected_ptr,
    temp_ptr,
    cslot_ptr,  # [max_num_reqs, 16] per-slot record written at the last propose
    rid_ptr,  # [max_num_reqs] int32 request serial by slot
    ring_ptr,  # [cap_calls, R, 20] fp32 device ring
    base,  # offset (floats) of this call's [R, 20] block
    NF: tl.constexpr,
):
    """One record per (step, request): [rid+1 (0 = not a greedy verified step), slot, num_sampled,
    drafts verified-batch size, W used, p[0..6], ema[0..6], temp]. Run BEFORE the EMA update."""
    r = tl.program_id(0)
    slot = tl.load(idx_ptr + r).to(tl.int64)
    ns = tl.load(num_sampled_ptr + r).to(tl.int32)
    nr = tl.load(num_rejected_ptr + r).to(tl.int32)
    temp = tl.load(temp_ptr + slot)
    rid = tl.load(rid_ptr + slot).to(tl.float32)
    nd = ns + nr - 1
    active = (ns >= 1) & (nd >= 1) & (temp == 0.0)
    j = tl.arange(0, 32)
    wv = tl.load(cslot_ptr + slot * 16 + 7)
    pv = tl.load(cslot_ptr + slot * 16 + (j - 5), mask=(j >= 5) & (j < 12), other=0.0)
    hv = tl.load(cslot_ptr + slot * 16 + 8 + (j - 12), mask=(j >= 12) & (j < 19), other=0.0)
    val = tl.where(
        j == 0,
        tl.where(active, rid + 1.0, 0.0),
        tl.where(
            j == 1,
            slot.to(tl.float32),
            tl.where(
                j == 2,
                ns.to(tl.float32),
                tl.where(
                    j == 3,
                    nd.to(tl.float32),
                    tl.where(j == 4, wv, tl.where(j == 19, temp, pv + hv)),
                ),
            ),
        ),
    )
    tl.store(ring_ptr + base + r * NF + j, val, mask=j < NF)




# =============================================================================
# INKLING_SAM_HYBRID=1 (Option A) & INKLING_TREE_SPEC=1 (Option B) Kernels
# 100% CUDA-Graph-Capture-Safe (ZERO .item(), ZERO host syncs, static shapes!)
# =============================================================================

@triton.jit
def _sam_hash4(a, b, c, d):
    """Fast deterministic 4-gram integer hash (matched bit-for-bit in CPU unit tests)."""
    h = (a.to(tl.uint32) * tl.full((), 2654435761, tl.uint32)) ^ (b.to(tl.uint32) * tl.full((), 2246822519, tl.uint32))
    h = (h ^ (h >> 13)) * tl.full((), 3266489917, tl.uint32)
    h = h ^ (c.to(tl.uint32) * tl.full((), 668265263, tl.uint32)) ^ (d.to(tl.uint32) * tl.full((), 374761393, tl.uint32))
    h = h ^ (h >> 16)
    return h


@triton.jit
def _sam_hash2(a, b):
    """Fast deterministic 2-gram integer hash for compact short-suffix agreement fallback."""
    h = (a.to(tl.uint32) * tl.full((), 2654435761, tl.uint32)) ^ (b.to(tl.uint32) * tl.full((), 2246822519, tl.uint32))
    h = (h ^ (h >> 15)) * tl.full((), 2246822519, tl.uint32)
    h = h ^ (h >> 13)
    return h


@triton.jit
def _sam_hash4_tag32(a, b, c, d):
    """Independent secondary 32-bit hash for collision-free LPC (48b) & Shared-Pool (48b) tags."""
    h = ((d.to(tl.uint32) ^ tl.full((), 0x9E3779B9, tl.uint32)) * tl.full((), 2246822519, tl.uint32)) ^ (
        (c.to(tl.uint32) ^ tl.full((), 0x85EBCA6B, tl.uint32)) * tl.full((), 3266489917, tl.uint32)
    )
    h = (h ^ (h >> 15)) * tl.full((), 668265263, tl.uint32)
    h = h ^ ((b.to(tl.uint32) ^ tl.full((), 0xC2B2AE35, tl.uint32)) * tl.full((), 2654435761, tl.uint32)) ^ (
        (a.to(tl.uint32) ^ tl.full((), 0x27D4EB2F, tl.uint32)) * tl.full((), 374761393, tl.uint32)
    )
    h = h ^ (h >> 16)
    return h


@triton.jit
def _sam_lpc_pack_key(slot, gen, h1, h2):
    """Pack non-zero 64-bit slot-tagged LPC key: [16b (slot, gen) | 16b h1_hi | 32b h2]."""
    sg = (((slot.to(tl.int64) & 31) << 11) | ((gen.to(tl.int64) & 2047) | 1)) & 0xFFFF
    tag48 = (((h1.to(tl.int64) >> 16) & 0xFFFF) << 32) | (h2.to(tl.int64) & 0xFFFFFFFF)
    return (sg << 48) | tag48


@triton.jit
def _sam_lpc_ingest_chunk_kernel(
    input_ids_ptr,         # [T] int32/int64
    positions_ptr,         # [T] int64
    query_start_loc_ptr,   # [num_reqs + 1] int32
    idx_mapping_ptr,       # [num_reqs] int32/int64
    num_rejected_ptr,      # [num_reqs] int32
    slot_gen_ptr,          # [MAX_SLOTS] int32
    lpc_key_ptr,           # [LPC_CAP] int64
    lpc_cont_ptr,          # [LPC_CAP, 11] int32
    sp_tag_ptr,            # [SP_CAP] int32
    sp_cont_ptr,           # [SP_CAP, 7] int32
    MAX_SLOTS: tl.constexpr,
    LPC_CAP: tl.constexpr,
    LPC_STRIDE: tl.constexpr,
    SP_CAP: tl.constexpr,
    HAS_LPC: tl.constexpr,
    HAS_SHARED: tl.constexpr,
    BLOCK_J: tl.constexpr = 256,
):
    """Parallel multi-CTA prefill prompt indexer (grid=(num_reqs, ceil(max_chunk/BLOCK_J))).
    Indexes 100% of 16K/24K prefill chunks across 128K-1M prompts into the Strided-4 Slot-Tagged
    Long-Prompt Continuation Cache (`LPC`) and/or Shared Pool (`SP`) in < 0.08 ms/chunk, with
    ZERO MAX_INGEST=4096 cap and ZERO wiping of earlier chunks!"""
    r = tl.program_id(0)
    c_blk = tl.program_id(1)
    slot = tl.load(idx_mapping_ptr + r).to(tl.int64)
    if (slot < 0) | (slot >= MAX_SLOTS):
        return
    ctx_start = tl.load(query_start_loc_ptr + r).to(tl.int32)
    ctx_end = tl.load(query_start_loc_ptr + r + 1).to(tl.int32)
    nr = tl.load(num_rejected_ptr + r).to(tl.int32)
    num_valid = ctx_end - nr - ctx_start
    j = c_blk * BLOCK_J + tl.arange(0, BLOCK_J)
    in_rng = (j >= 3) & (j + 1 < num_valid)
    j_safe = tl.where(in_rng, j, 3)
    pos = tl.load(positions_ptr + ctx_start + j_safe, mask=in_rng, other=-1).to(tl.int32)
    t3 = tl.load(input_ids_ptr + ctx_start + j_safe - 3, mask=in_rng, other=-1).to(tl.int32)
    t2 = tl.load(input_ids_ptr + ctx_start + j_safe - 2, mask=in_rng, other=-1).to(tl.int32)
    t1 = tl.load(input_ids_ptr + ctx_start + j_safe - 1, mask=in_rng, other=-1).to(tl.int32)
    t0 = tl.load(input_ids_ptr + ctx_start + j_safe, mask=in_rng, other=-1).to(tl.int32)
    ok_tok = in_rng & (pos >= 3) & (t3 >= 0) & (t2 >= 0) & (t1 >= 0) & (t0 >= 0)
    h1 = _sam_hash4(t3, t2, t1, t0)
    h2 = _sam_hash4_tag32(t3, t2, t1, t0)

    if HAS_LPC:
        smask = LPC_STRIDE - 1
        do_lpc = ok_tok & ((pos & smask) == smask) & (j + 4 < num_valid)
        lmask = LPC_CAP - 1
        bkt = (h1 & lmask).to(tl.int64)
        gen = tl.load(slot_gen_ptr + slot).to(tl.int32)
        key64 = _sam_lpc_pack_key(slot, gen, h1, h2)
        for off in tl.static_range(11):
            c_ok = do_lpc & (j + 1 + off < num_valid)
            ctok = tl.load(input_ids_ptr + ctx_start + j_safe + 1 + off, mask=c_ok, other=-1).to(tl.int32)
            tl.store(lpc_cont_ptr + bkt * 11 + off, tl.where(c_ok, ctok, -1), mask=do_lpc)
        tl.store(lpc_key_ptr + bkt, key64, mask=do_lpc)

    if HAS_SHARED:
        do_sp = ok_tok & ((pos & 1) == 1) & (j + 4 < num_valid)
        pmask = SP_CAP - 1
        sbkt = (h1 & pmask).to(tl.int64)
        stag = (h2 | tl.full((), 1, tl.uint32)).to(tl.int32)
        for off in tl.static_range(7):
            c_ok = do_sp & (j + 1 + off < num_valid)
            ctok = tl.load(input_ids_ptr + ctx_start + j_safe + 1 + off, mask=c_ok, other=-1).to(tl.int32)
            tl.store(sp_cont_ptr + sbkt * 7 + off, tl.where(c_ok, ctok, -1), mask=do_sp)
        tl.store(sp_tag_ptr + sbkt, stag, mask=do_sp)


@triton.jit
def _sam_gpu_ingest_kernel(
    input_ids_ptr,         # [T] int32/int64 target input_ids
    positions_ptr,         # [T] int64 target positions
    query_start_loc_ptr,   # [num_reqs + 1] int32
    idx_mapping_ptr,       # [num_reqs] int32/int64
    num_sampled_ptr,       # [num_reqs] int32
    num_rejected_ptr,      # [num_reqs] int32
    last_sampled_ptr,      # [max_num_reqs] int64
    ring_ptr,              # [MAX_SLOTS, RING_SZ] int32
    ht_last_ptr,           # [MAX_SLOTS, HT_SZ] int32
    ht_prev_ptr,           # [MAX_SLOTS, HT_SZ] int32
    cur_len_ptr,           # [MAX_SLOTS] int32
    ring_lo_ptr,           # [MAX_SLOTS] int32 contiguous valid floor position
    tree_active_ptr,       # [1] int32 (used only when HAS_TREE=True)
    acc_path_ptr,          # [MAX_M] int32 accepted node indices (used only when HAS_TREE=True)
    ht2_last_ptr,          # [MAX_SLOTS, HT2_SZ] int32 (used only when HAS_HT2=True)
    sp_tag_ptr,            # [SP_CAP] int32 (used only when HAS_SHARED=True)
    sp_cont_ptr,           # [SP_CAP, 7] int32 (used only when HAS_SHARED=True)
    MAX_SLOTS: tl.constexpr,
    RING_SZ: tl.constexpr,
    HT_SZ: tl.constexpr,
    MAX_INGEST: tl.constexpr = 4096,
    HAS_TREE: tl.constexpr = False,
    TREE_MAX_M: tl.constexpr = 8,
    HAS_HT2: tl.constexpr = False,
    HT2_SZ: tl.constexpr = 4096,
    HAS_SHARED: tl.constexpr = False,
    SP_CAP: tl.constexpr = 65536,
):
    """1-CTA/req zero-sync GPU token-ring & 4-gram table ingest (B=1..MAX_SLOTS, capture & DTGRAPH safe!).
    Maintains per-slot contiguous valid floor `ring_lo[slot]` so chunked prefill, prefix-cache hits,
    MAX_INGEST truncation, and slot reuse NEVER hash or match stale tokens from prior requests."""
    r = tl.program_id(0)
    slot = tl.load(idx_mapping_ptr + r).to(tl.int64)
    if (slot < 0) | (slot >= MAX_SLOTS):
        return
    ctx_start = tl.load(query_start_loc_ptr + r).to(tl.int32)
    ctx_end = tl.load(query_start_loc_ptr + r + 1).to(tl.int32)
    nr = tl.load(num_rejected_ptr + r).to(tl.int32)
    ns = tl.load(num_sampled_ptr + r).to(tl.int32)
    valid_end = ctx_end - nr
    num_valid = valid_end - ctx_start
    if num_valid <= 0:
        return

    ring_base = slot * RING_SZ
    ht_base = slot * HT_SZ
    ht2_base = slot * HT2_SZ
    rmask = RING_SZ - 1
    hmask = HT_SZ - 1
    h2mask = HT2_SZ - 1
    spmask = SP_CAP - 1

    use_tree_path = tl.zeros((), tl.int32)
    if HAS_TREE:
        if (tl.load(tree_active_ptr) != 0) & (num_valid <= TREE_MAX_M) & ((ctx_end - ctx_start) == TREE_MAX_M):
            use_tree_path = tl.full((), 1, tl.int32)
    start_j = tl.maximum(0, num_valid - MAX_INGEST)
    first_pos = tl.load(positions_ptr + ctx_start + start_j).to(tl.int32)
    prev_len = tl.load(cur_len_ptr + slot).to(tl.int32)
    ring_lo = tl.load(ring_lo_ptr + slot).to(tl.int32)
    if (prev_len <= 0) | (first_pos < prev_len - 1) | (first_pos > prev_len):
        ring_lo = tl.maximum(first_pos, 0)
        tl.store(ring_lo_ptr + slot, ring_lo)

    last_pos = tl.full((), -1, tl.int32)
    for j in range(start_j, num_valid):
        src_j = j
        if HAS_TREE:
            if (use_tree_path != 0) & (j < TREE_MAX_M):
                src_j = tl.load(acc_path_ptr + j).to(tl.int32)
        tok = tl.load(input_ids_ptr + ctx_start + src_j).to(tl.int32)
        pos = tl.load(positions_ptr + ctx_start + j).to(tl.int32)
        if (tok >= 0) & (pos >= 0):
            tl.store(ring_ptr + ring_base + (pos & rmask), tok)
            last_pos = pos
            if HAS_HT2:
                if pos >= ring_lo + 1:
                    p1 = tl.load(ring_ptr + ring_base + ((pos - 1) & rmask))
                    h2i = (_sam_hash2(p1, tok) & h2mask).to(tl.int64)
                    tl.store(ht2_last_ptr + ht2_base + h2i, pos)
            if pos >= ring_lo + 3:
                t3 = tl.load(ring_ptr + ring_base + ((pos - 3) & rmask))
                t2 = tl.load(ring_ptr + ring_base + ((pos - 2) & rmask))
                t1 = tl.load(ring_ptr + ring_base + ((pos - 1) & rmask))
                h = (_sam_hash4(t3, t2, t1, tok) & hmask).to(tl.int64)
                old = tl.load(ht_last_ptr + ht_base + h)
                if old != pos:
                    prev_ok = tl.where((old >= ring_lo) & (old < pos), old, -1)
                    tl.store(ht_prev_ptr + ht_base + h, prev_ok)
                    tl.store(ht_last_ptr + ht_base + h, pos)
            if HAS_SHARED:
                if pos >= ring_lo + 10:
                    s3 = tl.load(ring_ptr + ring_base + ((pos - 10) & rmask))
                    s2 = tl.load(ring_ptr + ring_base + ((pos - 9) & rmask))
                    s1 = tl.load(ring_ptr + ring_base + ((pos - 8) & rmask))
                    s0 = tl.load(ring_ptr + ring_base + ((pos - 7) & rmask))
                    sh1 = _sam_hash4(s3, s2, s1, s0)
                    sh2 = _sam_hash4_tag32(s3, s2, s1, s0)
                    sbkt = (sh1 & spmask).to(tl.int64)
                    for spk in tl.static_range(7):
                        ct = tl.load(ring_ptr + ring_base + ((pos - 6 + spk) & rmask))
                        tl.store(sp_cont_ptr + sbkt * 7 + spk, ct)
                    tl.store(sp_tag_ptr + sbkt, (sh2 | tl.full((), 1, tl.uint32)).to(tl.int32))

    if (ns > 0) & (last_pos >= 0):
        bpos = last_pos + 1
        btok = tl.load(last_sampled_ptr + slot).to(tl.int32)
        if btok >= 0:
            tl.store(ring_ptr + ring_base + (bpos & rmask), btok)
            if HAS_HT2:
                if bpos >= ring_lo + 1:
                    p1 = tl.load(ring_ptr + ring_base + ((bpos - 1) & rmask))
                    h2i = (_sam_hash2(p1, btok) & h2mask).to(tl.int64)
                    tl.store(ht2_last_ptr + ht2_base + h2i, bpos)
            if bpos >= ring_lo + 3:
                t3 = tl.load(ring_ptr + ring_base + ((bpos - 3) & rmask))
                t2 = tl.load(ring_ptr + ring_base + ((bpos - 2) & rmask))
                t1 = tl.load(ring_ptr + ring_base + ((bpos - 1) & rmask))
                h = (_sam_hash4(t3, t2, t1, btok) & hmask).to(tl.int64)
                old = tl.load(ht_last_ptr + ht_base + h)
                if old != bpos:
                    prev_ok = tl.where((old >= ring_lo) & (old < bpos), old, -1)
                    tl.store(ht_prev_ptr + ht_base + h, prev_ok)
                    tl.store(ht_last_ptr + ht_base + h, bpos)
            if HAS_SHARED:
                if bpos >= ring_lo + 10:
                    s3 = tl.load(ring_ptr + ring_base + ((bpos - 10) & rmask))
                    s2 = tl.load(ring_ptr + ring_base + ((bpos - 9) & rmask))
                    s1 = tl.load(ring_ptr + ring_base + ((bpos - 8) & rmask))
                    s0 = tl.load(ring_ptr + ring_base + ((bpos - 7) & rmask))
                    sh1 = _sam_hash4(s3, s2, s1, s0)
                    sh2 = _sam_hash4_tag32(s3, s2, s1, s0)
                    sbkt = (sh1 & spmask).to(tl.int64)
                    for spk in tl.static_range(7):
                        ct = tl.load(ring_ptr + ring_base + ((bpos - 6 + spk) & rmask))
                        tl.store(sp_cont_ptr + sbkt * 7 + spk, ct)
                    tl.store(sp_tag_ptr + sbkt, (sh2 | tl.full((), 1, tl.uint32)).to(tl.int32))
            tl.store(cur_len_ptr + slot, bpos + 1)
    elif last_pos >= 0:
        tl.store(cur_len_ptr + slot, last_pos + 1)


@triton.jit
def _vw_dink_sam_ingest_fused_kernel(
    idx_ptr,               # [num_reqs]
    num_sampled_ptr,       # [num_reqs]
    num_rejected_ptr,      # [num_reqs]
    temp_ptr,              # [max_num_reqs]
    w_ptr,                 # [max_num_reqs] int32 W used
    cpred_ptr,             # [max_num_reqs, K] fp32 predicted c[0..6]
    db_ptr,                # [max_num_reqs] fp32 online logit bias Delta-b (in/out)
    etok_ptr,              # [max_num_reqs] fp32 EMA tokens/step (in/out)
    ems_ptr,               # [max_num_reqs] fp32 EMA step cost ms (in/out)
    step_ptr,              # [8] fp32 effective step cost by W
    cal_ptr,               # [24] fp32 calibration table
    es_skip_ptr,
    es_surv_ptr,
    es_tok_ptr,
    es_bestu_ptr,
    es_greedy_ptr,
    es_slot_ptr,
    es_db_ptr,
    es_lam_ptr,
    input_ids_ptr,         # [T] int32/int64 target input_ids
    positions_ptr,         # [T] int64 target positions
    query_start_loc_ptr,   # [num_reqs + 1] int32
    last_sampled_ptr,      # [max_num_reqs] int64
    ring_ptr,              # [MAX_SLOTS, RING_SZ] int32
    ht_last_ptr,           # [MAX_SLOTS, HT_SZ] int32
    ht_prev_ptr,           # [MAX_SLOTS, HT_SZ] int32
    cur_len_ptr,           # [MAX_SLOTS] int32
    ring_lo_ptr,           # [MAX_SLOTS] int32
    tree_active_ptr,       # [1] int32 (used only when HAS_TREE=True)
    acc_path_ptr,          # [MAX_M] int32 (used only when HAS_TREE=True)
    ht2_last_ptr,          # [MAX_SLOTS, HT2_SZ] int32 (used only when HAS_HT2=True)
    sp_tag_ptr,            # [SP_CAP] int32 (used only when HAS_SHARED=True)
    sp_cont_ptr,           # [SP_CAP, 7] int32 (used only when HAS_SHARED=True)
    ETA_B: tl.constexpr,
    DECAY_B: tl.constexpr,
    B_CLIP: tl.constexpr,
    ETA_LAM: tl.constexpr,
    K: tl.constexpr,
    FUSE_ES: tl.constexpr,
    MAX_SLOTS: tl.constexpr,
    RING_SZ: tl.constexpr,
    HT_SZ: tl.constexpr,
    MAX_INGEST: tl.constexpr = 4096,
    HAS_TREE: tl.constexpr = False,
    TREE_MAX_M: tl.constexpr = 8,
    HAS_HT2: tl.constexpr = False,
    HT2_SZ: tl.constexpr = 4096,
    HAS_SHARED: tl.constexpr = False,
    SP_CAP: tl.constexpr = 65536,
):
    """0-ULP bit-identical 2->1 fusion of `_vw_update_dink_kernel` + `_sam_gpu_ingest_kernel`
    inside the TD DTGRAPH region (eliminates 1 Triton launch + duplicate idx/ns/nr loads/step)."""
    r = tl.program_id(0)
    slot = tl.load(idx_ptr + r).to(tl.int64)
    ns = tl.load(num_sampled_ptr + r).to(tl.int32)
    nr = tl.load(num_rejected_ptr + r).to(tl.int32)

    # Part 1: exact _vw_update_dink_kernel math (bit-identical to standalone kernel)
    temp = tl.load(temp_ptr + slot)
    w_used = tl.load(w_ptr + slot)
    nd = ns + nr - 1
    acc = ns - 1
    active = (ns >= 1) & (nd >= 1) & (temp == 0.0)

    k = tl.arange(0, 8)
    kmask = k < K
    cpred = tl.load(cpred_ptr + slot * K + k, mask=kmask, other=0.7)
    obs = (k <= acc) & (k < tl.minimum(w_used, nd)) & kmask
    hit = (k < acc).to(tl.float32)
    n_obs = tl.sum(obs.to(tl.int32), axis=0)
    g = tl.sum(tl.where(obs, hit - cpred, 0.0), axis=0)
    h_info = tl.sum(tl.where(obs, cpred * (1.0 - cpred), 0.0), axis=0)

    db_old = tl.load(db_ptr + slot)
    db_obs = db_old * (1.0 - DECAY_B) + g / ((1.0 / ETA_B) + h_info)
    db_none = db_old * (1.0 - tl.maximum(DECAY_B, 0.02))
    db_new = tl.where(n_obs > 0, db_obs, db_none)
    db_new = tl.minimum(tl.maximum(db_new, -B_CLIP), B_CLIP)

    w_clamped = tl.minimum(tl.maximum(w_used, 0), K)
    step_c = tl.load(step_ptr + w_clamped)
    etok = tl.load(etok_ptr + slot)
    ems = tl.load(ems_ptr + slot)
    etok_new = etok + ETA_LAM * (ns.to(tl.float32) - etok)
    ems_new = ems + ETA_LAM * (step_c - ems)

    db_out = tl.where(active, db_new, db_old)
    etok_out = tl.where(active, etok_new, etok)
    ems_out = tl.where(active, ems_new, ems)
    tl.store(db_ptr + slot, db_out)
    tl.store(etok_ptr + slot, etok_out)
    tl.store(ems_ptr + slot, ems_out)
    if FUSE_ES:
        if r == 0:
            s0 = tl.load(step_ptr)
            static_lam = tl.load(cal_ptr + 18)
            gamma = tl.load(cal_ptr + 19)
            ems_cl = tl.maximum(ems_out, 1.0)
            lam = tl.where(static_lam > 0.0, static_lam, gamma * (etok_out / ems_cl))
            tl.store(es_skip_ptr, 0)
            tl.store(es_surv_ptr, 1.0)
            tl.store(es_tok_ptr, 1.0)
            tl.store(es_bestu_ptr, 1.0 - lam * s0)
            tl.store(es_greedy_ptr, tl.where(temp == 0.0, 1, 0))
            tl.store(es_slot_ptr, slot.to(tl.int32))
            tl.store(es_db_ptr, db_out)
            tl.store(es_lam_ptr, lam)

    # Part 2: exact _sam_gpu_ingest_kernel math (reusing slot, ns, nr)
    if (slot < 0) | (slot >= MAX_SLOTS):
        return
    ctx_start = tl.load(query_start_loc_ptr + r).to(tl.int32)
    ctx_end = tl.load(query_start_loc_ptr + r + 1).to(tl.int32)
    valid_end = ctx_end - nr
    num_valid = valid_end - ctx_start
    if num_valid <= 0:
        return

    ring_base = slot * RING_SZ
    ht_base = slot * HT_SZ
    ht2_base = slot * HT2_SZ
    rmask = RING_SZ - 1
    hmask = HT_SZ - 1
    h2mask = HT2_SZ - 1
    spmask = SP_CAP - 1

    use_tree_path = tl.zeros((), tl.int32)
    if HAS_TREE:
        if (tl.load(tree_active_ptr) != 0) & (num_valid <= TREE_MAX_M) & ((ctx_end - ctx_start) == TREE_MAX_M):
            use_tree_path = tl.full((), 1, tl.int32)
    start_j = tl.maximum(0, num_valid - MAX_INGEST)
    first_pos = tl.load(positions_ptr + ctx_start + start_j).to(tl.int32)
    prev_len = tl.load(cur_len_ptr + slot).to(tl.int32)
    ring_lo = tl.load(ring_lo_ptr + slot).to(tl.int32)
    if (prev_len <= 0) | (first_pos < prev_len - 1) | (first_pos > prev_len):
        ring_lo = tl.maximum(first_pos, 0)
        tl.store(ring_lo_ptr + slot, ring_lo)

    last_pos = tl.full((), -1, tl.int32)
    for j in range(start_j, num_valid):
        src_j = j
        if HAS_TREE:
            if (use_tree_path != 0) & (j < TREE_MAX_M):
                src_j = tl.load(acc_path_ptr + j).to(tl.int32)
        tok = tl.load(input_ids_ptr + ctx_start + src_j).to(tl.int32)
        pos = tl.load(positions_ptr + ctx_start + j).to(tl.int32)
        if (tok >= 0) & (pos >= 0):
            tl.store(ring_ptr + ring_base + (pos & rmask), tok)
            last_pos = pos
            if HAS_HT2:
                if pos >= ring_lo + 1:
                    p1 = tl.load(ring_ptr + ring_base + ((pos - 1) & rmask))
                    h2i = (_sam_hash2(p1, tok) & h2mask).to(tl.int64)
                    tl.store(ht2_last_ptr + ht2_base + h2i, pos)
            if pos >= ring_lo + 3:
                t3 = tl.load(ring_ptr + ring_base + ((pos - 3) & rmask))
                t2 = tl.load(ring_ptr + ring_base + ((pos - 2) & rmask))
                t1 = tl.load(ring_ptr + ring_base + ((pos - 1) & rmask))
                h = (_sam_hash4(t3, t2, t1, tok) & hmask).to(tl.int64)
                old = tl.load(ht_last_ptr + ht_base + h)
                if old != pos:
                    prev_ok = tl.where((old >= ring_lo) & (old < pos), old, -1)
                    tl.store(ht_prev_ptr + ht_base + h, prev_ok)
                    tl.store(ht_last_ptr + ht_base + h, pos)
            if HAS_SHARED:
                if pos >= ring_lo + 10:
                    s3 = tl.load(ring_ptr + ring_base + ((pos - 10) & rmask))
                    s2 = tl.load(ring_ptr + ring_base + ((pos - 9) & rmask))
                    s1 = tl.load(ring_ptr + ring_base + ((pos - 8) & rmask))
                    s0 = tl.load(ring_ptr + ring_base + ((pos - 7) & rmask))
                    sh1 = _sam_hash4(s3, s2, s1, s0)
                    sh2 = _sam_hash4_tag32(s3, s2, s1, s0)
                    sbkt = (sh1 & spmask).to(tl.int64)
                    for spk in tl.static_range(7):
                        ct = tl.load(ring_ptr + ring_base + ((pos - 6 + spk) & rmask))
                        tl.store(sp_cont_ptr + sbkt * 7 + spk, ct)
                    tl.store(sp_tag_ptr + sbkt, (sh2 | tl.full((), 1, tl.uint32)).to(tl.int32))

    if (ns > 0) & (last_pos >= 0):
        bpos = last_pos + 1
        btok = tl.load(last_sampled_ptr + slot).to(tl.int32)
        if btok >= 0:
            tl.store(ring_ptr + ring_base + (bpos & rmask), btok)
            if HAS_HT2:
                if bpos >= ring_lo + 1:
                    p1 = tl.load(ring_ptr + ring_base + ((bpos - 1) & rmask))
                    h2i = (_sam_hash2(p1, btok) & h2mask).to(tl.int64)
                    tl.store(ht2_last_ptr + ht2_base + h2i, bpos)
            if bpos >= ring_lo + 3:
                t3 = tl.load(ring_ptr + ring_base + ((bpos - 3) & rmask))
                t2 = tl.load(ring_ptr + ring_base + ((bpos - 2) & rmask))
                t1 = tl.load(ring_ptr + ring_base + ((bpos - 1) & rmask))
                h = (_sam_hash4(t3, t2, t1, btok) & hmask).to(tl.int64)
                old = tl.load(ht_last_ptr + ht_base + h)
                if old != bpos:
                    prev_ok = tl.where((old >= ring_lo) & (old < bpos), old, -1)
                    tl.store(ht_prev_ptr + ht_base + h, prev_ok)
                    tl.store(ht_last_ptr + ht_base + h, bpos)
            if HAS_SHARED:
                if bpos >= ring_lo + 10:
                    s3 = tl.load(ring_ptr + ring_base + ((bpos - 10) & rmask))
                    s2 = tl.load(ring_ptr + ring_base + ((bpos - 9) & rmask))
                    s1 = tl.load(ring_ptr + ring_base + ((bpos - 8) & rmask))
                    s0 = tl.load(ring_ptr + ring_base + ((bpos - 7) & rmask))
                    sh1 = _sam_hash4(s3, s2, s1, s0)
                    sh2 = _sam_hash4_tag32(s3, s2, s1, s0)
                    sbkt = (sh1 & spmask).to(tl.int64)
                    for spk in tl.static_range(7):
                        ct = tl.load(ring_ptr + ring_base + ((bpos - 6 + spk) & rmask))
                        tl.store(sp_cont_ptr + sbkt * 7 + spk, ct)
                    tl.store(sp_tag_ptr + sbkt, (sh2 | tl.full((), 1, tl.uint32)).to(tl.int32))
            tl.store(cur_len_ptr + slot, bpos + 1)
    elif last_pos >= 0:
        tl.store(cur_len_ptr + slot, last_pos + 1)


@triton.jit(do_not_specialize=["k_active", "k_ext_cap"])
def _sam_gpu_boost_extend_kernel(
    idx_ptr,
    temp_ptr,
    draft_ptr,
    draft_stride,
    conf_ptr,
    conf_stride,
    cal_ptr,
    db_ptr,
    ring_ptr,
    ht_last_ptr,
    ht_prev_ptr,
    cur_len_ptr,
    ring_lo_ptr,
    ht2_last_ptr,
    slot_gen_ptr,
    lpc_key_ptr,
    lpc_cont_ptr,
    sp_tag_ptr,
    sp_cont_ptr,
    k_active,
    k_ext_cap,
    MAX_SLOTS: tl.constexpr,
    RING_SZ: tl.constexpr,
    HT_SZ: tl.constexpr,
    K_DRAFT: tl.constexpr,
    K_MAX: tl.constexpr,
    MIN_MATCH: tl.constexpr,
    BOOST_BASE: tl.constexpr,
    BOOST_MAX: tl.constexpr,
    EXT_SURV_THRESH: tl.constexpr,
    FILL_EARLY_STOP: tl.constexpr = True,
    IDX_STRIDE: tl.constexpr = 1,
    OVERRIDE_LOW_CONF: tl.constexpr = False,
    OVERRIDE_CONF_THRESH: tl.constexpr = 0.25,
    OVERRIDE_MIN_MATCH: tl.constexpr = 5,
    OVERRIDE_CAL_INVERT: tl.constexpr = False,
    ALLOW_TEMP: tl.constexpr = False,
    BEST_OF_2: tl.constexpr = False,
    CONF_BOOST: tl.constexpr = False,
    CONF_BOOST_ML2: tl.constexpr = 0.76,
    CONF_BOOST_ML3: tl.constexpr = 0.85,
    CONF_BOOST_ML4: tl.constexpr = 0.92,
    HAS_HT2: tl.constexpr = False,
    HT2_SZ: tl.constexpr = 4096,
    HAS_LPC: tl.constexpr = False,
    LPC_CAP: tl.constexpr = 262144,
    LPC_STRIDE: tl.constexpr = 4,
    HAS_SHARED: tl.constexpr = False,
    SP_CAP: tl.constexpr = 65536,
):
    """1-CTA/req zero-sync GPU SAM Hybrid kernel (B=1..MAX_SLOTS, < 6 us @ B=32, 100% CUDA-Graph capturable!).
    Wave-V11 opt-in upgrades (all strictly default-OFF):
      - CONF_BOOST=True: calibration-inverted agreement confidence floor (`ml >= 2` via optional HT2),
        preventing false DINK_BATCH_TAU / Water-Fill truncation on verified DSpark+SAM agreements.
      - HAS_LPC=True: Strided-4 Slot-Tagged Long-Prompt Continuation Cache fallback (0 distance limit
        across 128K-1M prompts!).
      - HAS_SHARED=True: Lock-free 2.0 MiB Cross-Request Shared-Prefix SAM Pool fallback (C=4..32)."""
    r = tl.program_id(0)
    slot = tl.load(idx_ptr + r * IDX_STRIDE).to(tl.int64)
    if (slot < 0) | (slot >= MAX_SLOTS):
        return
    temp = tl.load(temp_ptr + slot)
    is_greedy = temp == 0.0
    if not ALLOW_TEMP:
        if not is_greedy:
            return
    d_row = draft_ptr + r.to(tl.int64) * draft_stride
    c_row = conf_ptr + r.to(tl.int64) * conf_stride
    n_tot = tl.load(cur_len_ptr + slot).to(tl.int32)
    ring_lo = tl.load(ring_lo_ptr + slot).to(tl.int32)
    cur_pos = n_tot - 1
    valid_lo = tl.maximum(ring_lo, cur_pos - (RING_SZ - 32))
    if (cur_pos < valid_lo + 7) | (k_active <= 0):
        if K_MAX > K_DRAFT:
            for e in tl.static_range(K_MAX - K_DRAFT):
                tl.store(d_row + K_DRAFT + e, -1)
        return

    ring_base = slot * RING_SZ
    ht_base = slot * HT_SZ
    rmask = RING_SZ - 1
    hmask = HT_SZ - 1

    t3 = tl.load(ring_ptr + ring_base + ((cur_pos - 3) & rmask))
    t2 = tl.load(ring_ptr + ring_base + ((cur_pos - 2) & rmask))
    t1 = tl.load(ring_ptr + ring_base + ((cur_pos - 1) & rmask))
    t0 = tl.load(ring_ptr + ring_base + (cur_pos & rmask))
    h = (_sam_hash4(t3, t2, t1, t0) & hmask).to(tl.int64)
    hp = tl.load(ht_last_ptr + ht_base + h)
    hp2 = tl.load(ht_prev_ptr + ht_base + h)
    if hp >= cur_pos:
        hp = hp2
        hp2 = -1

    ml = tl.zeros((), tl.int32)
    if (hp >= valid_lo + 3) & (hp < cur_pos) & ((cur_pos - hp) < (RING_SZ - 32)):
        still = tl.full((), 1, tl.int32)
        for d in tl.static_range(16):
            if (still != 0) & (hp - d >= valid_lo):
                a_t = tl.load(ring_ptr + ring_base + ((hp - d) & rmask))
                b_t = tl.load(ring_ptr + ring_base + ((cur_pos - d) & rmask))
                if a_t == b_t:
                    ml = ml + 1
                else:
                    still = 0
    if BEST_OF_2:
        if (hp2 >= valid_lo + 3) & (hp2 < cur_pos) & ((cur_pos - hp2) < (RING_SZ - 32)):
            ml2 = tl.zeros((), tl.int32)
            still2 = tl.full((), 1, tl.int32)
            for d in tl.static_range(16):
                if (still2 != 0) & (hp2 - d >= valid_lo):
                    a_t = tl.load(ring_ptr + ring_base + ((hp2 - d) & rmask))
                    b_t = tl.load(ring_ptr + ring_base + ((cur_pos - d) & rmask))
                    if a_t == b_t:
                        ml2 = ml2 + 1
                    else:
                        still2 = 0
            if ml2 > ml:
                ml = ml2
                hp = hp2

    # Fallback source selection: 0=per-slot ring, 1=128K/1M LPC, 2=cross-req Shared Pool
    ext_src = tl.zeros((), tl.int32)
    lpc_bkt = tl.zeros((), tl.int64)
    lpc_off = tl.zeros((), tl.int32)
    sp_bkt = tl.zeros((), tl.int64)

    if HAS_LPC:
        if ml < MIN_MATCH:
            gen = tl.load(slot_gen_ptr + slot).to(tl.int32)
            lmask = LPC_CAP - 1
            for off in tl.static_range(LPC_STRIDE):
                if (ext_src == 0) & (cur_pos - off - 3 >= valid_lo):
                    q3 = tl.load(ring_ptr + ring_base + ((cur_pos - off - 3) & rmask))
                    q2 = tl.load(ring_ptr + ring_base + ((cur_pos - off - 2) & rmask))
                    q1 = tl.load(ring_ptr + ring_base + ((cur_pos - off - 1) & rmask))
                    q0 = tl.load(ring_ptr + ring_base + ((cur_pos - off) & rmask))
                    lh1 = _sam_hash4(q3, q2, q1, q0)
                    lh2 = _sam_hash4_tag32(q3, q2, q1, q0)
                    lb = (lh1 & lmask).to(tl.int64)
                    want_k = _sam_lpc_pack_key(slot, gen, lh1, lh2)
                    got_k = tl.load(lpc_key_ptr + lb)
                    if got_k == want_k:
                        ver_ok = tl.full((), 1, tl.int32)
                        for vv in tl.static_range(off):
                            ct_v = tl.load(lpc_cont_ptr + lb * 11 + vv)
                            rt_v = tl.load(ring_ptr + ring_base + ((cur_pos - off + 1 + vv) & rmask))
                            if ct_v != rt_v:
                                ver_ok = 0
                        if ver_ok != 0:
                            ext_src = 1
                            lpc_bkt = lb
                            lpc_off = off
                            ml = 4 + off

    if HAS_SHARED:
        if (ml < MIN_MATCH) & (ext_src == 0):
            spmask = SP_CAP - 1
            sh1 = _sam_hash4(t3, t2, t1, t0)
            sh2 = _sam_hash4_tag32(t3, t2, t1, t0)
            sb = (sh1 & spmask).to(tl.int64)
            want_tag = (sh2 | tl.full((), 1, tl.uint32)).to(tl.int32)
            if tl.load(sp_tag_ptr + sb) == want_tag:
                ext_src = 2
                sp_bkt = sb
                ml = 4

    if CONF_BOOST and HAS_HT2:
        if (ml < 2) & (ext_src == 0):
            ht2_base = slot * HT2_SZ
            h2i = (_sam_hash2(t1, t0) & (HT2_SZ - 1)).to(tl.int64)
            hp_s = tl.load(ht2_last_ptr + ht2_base + h2i)
            if (hp_s >= valid_lo + 1) & (hp_s < cur_pos) & ((cur_pos - hp_s) < (RING_SZ - 32)):
                a0 = tl.load(ring_ptr + ring_base + (hp_s & rmask))
                a1 = tl.load(ring_ptr + ring_base + ((hp_s - 1) & rmask))
                if (a0 == t0) & (a1 == t1):
                    hp = hp_s
                    ml = 2
                    if hp_s >= valid_lo + 2:
                        a2 = tl.load(ring_ptr + ring_base + ((hp_s - 2) & rmask))
                        if a2 == t2:
                            ml = 3

    enter_ok = (ml >= 2) if CONF_BOOST else (ml >= MIN_MATCH)
    if enter_ok:
        db = tl.load(db_ptr + slot)
        agree = tl.full((), 1, tl.int32)
        overridden = tl.zeros((), tl.int32)
        ovr_decay = tl.full((), 1.0, tl.float32)
        boost_z = tl.minimum(BOOST_MAX, BOOST_BASE + 0.18 * (ml.to(tl.float32) - 2.0))
        for k in tl.static_range(K_DRAFT):
            if k < k_active:
                has_next = tl.zeros((), tl.int1)
                sam_t = tl.full((), -1, tl.int64)
                if ext_src == 0:
                    has_next = hp + 1 + k < cur_pos
                    if has_next:
                        sam_t = tl.load(ring_ptr + ring_base + ((hp + 1 + k) & rmask)).to(tl.int64)
                elif ext_src == 1:
                    if HAS_LPC:
                        sam_t = tl.load(lpc_cont_ptr + lpc_bkt * 11 + lpc_off + k).to(tl.int64)
                        has_next = sam_t >= 0
                else:
                    if HAS_SHARED:
                        sam_t = tl.load(sp_cont_ptr + sp_bkt * 7 + k).to(tl.int64)
                        has_next = sam_t >= 0
                if overridden != 0:
                    if has_next & (sam_t >= 0):
                        tl.store(d_row + k, sam_t)
                        eff_l = (ml + k).to(tl.float32)
                        p_sam = tl.minimum(0.88, 1.0 / (1.0 + tl.exp(-0.55 * (eff_l - 2.85)))) * ovr_decay
                        p_sam = tl.minimum(tl.maximum(p_sam, 0.10), 0.94)
                        p_out = p_sam
                        if OVERRIDE_CAL_INVERT:
                            ak = tl.maximum(tl.load(cal_ptr + k), 0.25)
                            bk = tl.load(cal_ptr + 8 + k)
                            z_inv = (tl.log(p_sam) - tl.log(1.0 - p_sam) - bk - db) / ak
                            p_out = tl.minimum(tl.maximum(1.0 / (1.0 + tl.exp(-z_inv)), 0.10), 0.995)
                        tl.store(c_row + k, p_out)
                        ovr_decay = ovr_decay * 0.96
                    else:
                        if is_greedy:
                            tl.store(d_row + k, -1)
                            tl.store(c_row + k, 0.0)
                        agree = 0
                else:
                    if (agree != 0) & has_next & (sam_t >= 0):
                        dt = tl.load(d_row + k)
                        if dt == sam_t:
                            pk = tl.load(c_row + k)
                            pc = tl.minimum(tl.maximum(pk, 1e-6), 1.0 - 1e-6)
                            if CONF_BOOST:
                                ml_c = tl.minimum(ml, 4).to(tl.float32)
                                p_fl4 = tl.minimum(0.985, CONF_BOOST_ML4 + 0.015 * ml_c)
                                p_fl = tl.where(
                                    ml >= 4,
                                    p_fl4,
                                    tl.where(ml == 3, CONF_BOOST_ML3, CONF_BOOST_ML2),
                                )
                                ak = tl.maximum(tl.load(cal_ptr + k), 0.25)
                                bk = tl.load(cal_ptr + 8 + k)
                                z_inv = (tl.log(p_fl) - tl.log(1.0 - p_fl) - bk - db) / ak
                                p_inv = tl.minimum(tl.maximum(1.0 / (1.0 + tl.exp(-z_inv)), 0.10), 0.999)
                                tl.store(c_row + k, tl.maximum(pk, p_inv))
                            else:
                                z = tl.log(pc) - tl.log(1.0 - pc) + boost_z
                                tl.store(c_row + k, 1.0 / (1.0 + tl.exp(-z)))
                        else:
                            handled = tl.zeros((), tl.int32)
                            if FILL_EARLY_STOP:
                                if (ml >= MIN_MATCH) & (dt < 0) & is_greedy & ((ml + k) >= (MIN_MATCH + 1)):
                                    tl.store(d_row + k, sam_t)
                                    eff_l = (ml + k).to(tl.float32)
                                    p_sam = tl.minimum(0.94, 1.0 / (1.0 + tl.exp(-0.55 * (eff_l - 2.85))))
                                    tl.store(c_row + k, p_sam)
                                    overridden = 1
                                    ovr_decay = 0.96
                                    handled = 1
                            if OVERRIDE_LOW_CONF:
                                if (ml >= MIN_MATCH) & (handled == 0) & (dt >= 0) & ((ml + k) >= OVERRIDE_MIN_MATCH):
                                    pk = tl.load(c_row + k)
                                    pc = tl.minimum(tl.maximum(pk, 1e-6), 1.0 - 1e-6)
                                    ak = tl.load(cal_ptr + k)
                                    bk = tl.load(cal_ptr + 8 + k)
                                    ck = 1.0 / (1.0 + tl.exp(-(ak * (tl.log(pc) - tl.log(1.0 - pc)) + bk + db)))
                                    if ck < OVERRIDE_CONF_THRESH:
                                        tl.store(d_row + k, sam_t)
                                        eff_l = (ml + k).to(tl.float32)
                                        p_sam = tl.minimum(0.88, 1.0 / (1.0 + tl.exp(-0.55 * (eff_l - 2.85))))
                                        p_out = p_sam
                                        if OVERRIDE_CAL_INVERT:
                                            ak_s = tl.maximum(ak, 0.25)
                                            z_inv = (tl.log(p_sam) - tl.log(1.0 - p_sam) - bk - db) / ak_s
                                            p_out = tl.minimum(tl.maximum(1.0 / (1.0 + tl.exp(-z_inv)), 0.10), 0.995)
                                        tl.store(c_row + k, p_out)
                                        overridden = 1
                                        ovr_decay = 0.96
                                        handled = 1
                            if handled == 0:
                                agree = 0
                    else:
                        agree = 0

    if K_MAX > K_DRAFT:
        db = tl.load(db_ptr + slot)
        surv = tl.full((), 1.0, tl.float32)
        all_valid = tl.full((), 1, tl.int32)
        for k in tl.static_range(K_DRAFT):
            dt = tl.load(d_row + k)
            if dt < 0:
                all_valid = 0
            pk = tl.load(c_row + k)
            pc = tl.minimum(tl.maximum(pk, 1e-6), 1.0 - 1e-6)
            ak = tl.load(cal_ptr + k)
            bk = tl.load(cal_ptr + 8 + k)
            ck = 1.0 / (1.0 + tl.exp(-(ak * (tl.log(pc) - tl.log(1.0 - pc)) + bk + db)))
            surv = surv * ck

        ext_ok = is_greedy & (k_active >= K_DRAFT) & (k_ext_cap > 0) & (all_valid != 0) & (surv >= EXT_SURV_THRESH)
        hp_e = tl.full((), -1, tl.int32)
        ml_e = tl.zeros((), tl.int32)
        ext_e_src = tl.zeros((), tl.int32)
        lpc_eb = tl.zeros((), tl.int64)
        lpc_eo = tl.zeros((), tl.int32)
        sp_eb = tl.zeros((), tl.int64)
        if ext_ok:
            d3 = tl.load(d_row + K_DRAFT - 4).to(tl.int32)
            d2 = tl.load(d_row + K_DRAFT - 3).to(tl.int32)
            d1 = tl.load(d_row + K_DRAFT - 2).to(tl.int32)
            d0 = tl.load(d_row + K_DRAFT - 1).to(tl.int32)
            he = (_sam_hash4(d3, d2, d1, d0) & hmask).to(tl.int64)
            hp_e = tl.load(ht_last_ptr + ht_base + he)
            if hp_e >= cur_pos:
                hp_e = tl.load(ht_prev_ptr + ht_base + he)
            if (hp_e >= valid_lo + 3) & (hp_e < cur_pos) & ((cur_pos - hp_e) < (RING_SZ - 32)):
                still_e = tl.full((), 1, tl.int32)
                for d in tl.static_range(16):
                    if (still_e != 0) & (hp_e - d >= valid_lo):
                        a_t = tl.load(ring_ptr + ring_base + ((hp_e - d) & rmask))
                        if d < K_DRAFT:
                            b_t = tl.load(d_row + (K_DRAFT - 1 - d)).to(tl.int32)
                        else:
                            b_t = tl.load(ring_ptr + ring_base + ((cur_pos - (d - K_DRAFT)) & rmask))
                        if a_t == b_t:
                            ml_e = ml_e + 1
                        else:
                            still_e = 0
            if HAS_LPC:
                if ml_e < MIN_MATCH:
                    gen_e = tl.load(slot_gen_ptr + slot).to(tl.int32)
                    lmask_e = LPC_CAP - 1
                    for off in tl.static_range(LPC_STRIDE):
                        if (ext_e_src == 0) & (K_DRAFT - 4 - off >= 0):
                            q3 = tl.load(d_row + K_DRAFT - 4 - off).to(tl.int32)
                            q2 = tl.load(d_row + K_DRAFT - 3 - off).to(tl.int32)
                            q1 = tl.load(d_row + K_DRAFT - 2 - off).to(tl.int32)
                            q0 = tl.load(d_row + K_DRAFT - 1 - off).to(tl.int32)
                            lh1 = _sam_hash4(q3, q2, q1, q0)
                            lh2 = _sam_hash4_tag32(q3, q2, q1, q0)
                            lb = (lh1 & lmask_e).to(tl.int64)
                            want_k = _sam_lpc_pack_key(slot, gen_e, lh1, lh2)
                            if tl.load(lpc_key_ptr + lb) == want_k:
                                ver_ok = tl.full((), 1, tl.int32)
                                for vv in tl.static_range(off):
                                    ct_v = tl.load(lpc_cont_ptr + lb * 11 + vv)
                                    dt_v = tl.load(d_row + K_DRAFT - off + vv).to(tl.int32)
                                    if ct_v != dt_v:
                                        ver_ok = 0
                                if ver_ok != 0:
                                    ext_e_src = 1
                                    lpc_eb = lb
                                    lpc_eo = off
                                    ml_e = 4 + off
            if HAS_SHARED:
                if (ml_e < MIN_MATCH) & (ext_e_src == 0):
                    spmask_e = SP_CAP - 1
                    sh1 = _sam_hash4(d3, d2, d1, d0)
                    sh2 = _sam_hash4_tag32(d3, d2, d1, d0)
                    sb = (sh1 & spmask_e).to(tl.int64)
                    if tl.load(sp_tag_ptr + sb) == (sh2 | tl.full((), 1, tl.uint32)).to(tl.int32):
                        ext_e_src = 2
                        sp_eb = sb
                        ml_e = 4

        for e in tl.static_range(K_MAX - K_DRAFT):
            etok = tl.full((), -1, tl.int64)
            if (e < k_ext_cap) & (ml_e >= MIN_MATCH):
                if ext_e_src == 0:
                    if hp_e + 1 + e < cur_pos:
                        etok = tl.load(ring_ptr + ring_base + ((hp_e + 1 + e) & rmask)).to(tl.int64)
                elif ext_e_src == 1:
                    if HAS_LPC:
                        if lpc_eo + e < 11:
                            etok = tl.load(lpc_cont_ptr + lpc_eb * 11 + lpc_eo + e).to(tl.int64)
                else:
                    if HAS_SHARED:
                        if e < 7:
                            etok = tl.load(sp_cont_ptr + sp_eb * 7 + e).to(tl.int64)
            tl.store(d_row + K_DRAFT + e, tl.where(etok >= 0, etok, -1))


@triton.jit(do_not_specialize=["k_ext_cap"])
def _sam_mask_ext_tail_kernel(
    idx_ptr, w_ptr, draft_ptr, draft_stride, req_draft_ptr, req_draft_stride,
    k_ext_cap,
    K_DRAFT: tl.constexpr, K_MAX: tl.constexpr,
    WRITE_REQ_DRAFTS: tl.constexpr = False,
    IDX_STRIDE: tl.constexpr = 1,
):
    """For K_MAX > K_DRAFT (K=11/15, B=1..2): keep SAM ext tokens [K_DRAFT..K_DRAFT+k_ext_cap-1]
    iff w_used >= K_DRAFT, otherwise clear to -1 (and mirror into req_drafts when WRITE_REQ_DRAFTS)."""
    r = tl.program_id(0)
    slot = tl.load(idx_ptr + r * IDX_STRIDE).to(tl.int64)
    if slot < 0:
        return
    w_used = tl.load(w_ptr + slot)
    keep = w_used >= K_DRAFT
    d_row = draft_ptr + r.to(tl.int64) * draft_stride
    for e in tl.static_range(K_MAX - K_DRAFT):
        dt = tl.load(d_row + K_DRAFT + e)
        out_t = tl.where(keep & (e < k_ext_cap), dt, -1)
        tl.store(d_row + K_DRAFT + e, out_t)
        if WRITE_REQ_DRAFTS:
            tl.store(req_draft_ptr + slot * req_draft_stride + K_DRAFT + e, out_t)


@triton.jit
def _sam_apply_staged_kernel(
    draft_ptr,
    conf_ptr,
    staged_tok_ptr,
    staged_boost_ptr,
    staged_ext_ok_ptr,
    K_DRAFT: tl.constexpr,
    K_MAX: tl.constexpr,
):
    """Apply CPU `/rdma/libsam_fast.so` staged Mode C boost & Mode A ext on GPU (0 sync)."""
    agree = tl.full((), 1, tl.int32)
    for k in tl.static_range(K_DRAFT):
        st = tl.load(staged_tok_ptr + k)
        bz = tl.load(staged_boost_ptr + k)
        dt = tl.load(draft_ptr + k)
        if (agree != 0) & (st >= 0) & (dt == st):
            if bz > 0.0:
                pk = tl.load(conf_ptr + k)
                pc = tl.minimum(tl.maximum(pk, 1e-6), 1.0 - 1e-6)
                z = tl.log(pc) - tl.log(1.0 - pc) + bz
                tl.store(conf_ptr + k, 1.0 / (1.0 + tl.exp(-z)))
        elif (agree != 0) & (st >= 0) & (dt < 0) & (bz >= 0.7):
            tl.store(draft_ptr + k, st)
            tl.store(conf_ptr + k, 0.80)
        else:
            agree = 0
    if K_MAX > K_DRAFT:
        ext_ok = tl.load(staged_ext_ok_ptr)
        for e in tl.static_range(K_MAX - K_DRAFT):
            et = tl.load(staged_tok_ptr + K_DRAFT + e)
            tl.store(draft_ptr + K_DRAFT + e, tl.where((ext_ok != 0) & (et >= 0), et, -1))


@triton.jit
def _pack_beam2_init_kernel(
    state0_ptr,
    t1_ptr, p1_ptr, t2_ptr, p2_ptr,
    beam_state_ptr,
    beam_prev_ptr,
    prim_tok_ptr,
    conf_ptr,
    sec_tok_ptr,
    sec_p_ptr,
    p2_0_ptr,
    skip_ptr, surv_ptr, tok_ptr, bestu_ptr, greedy_ptr, db_dev_ptr, lam_dev_ptr,
    cal_ptr, step_ptr,
    RANK: tl.constexpr,
    HAS_ES: tl.constexpr = False,
    CAN_SET_SKIP: tl.constexpr = False,
):
    """Zero-sync GPU pack of step-0 top-2 outputs + tail clear + optional Dinkelbach ES step-0 advance."""
    k = tl.arange(0, RANK)
    st = tl.load(state0_ptr + k)
    tl.store(beam_state_ptr + k, st)
    tl.store(beam_state_ptr + RANK + k, st)
    t1 = tl.load(t1_ptr)
    p1 = tl.load(p1_ptr)
    t2 = tl.load(t2_ptr)
    p2 = tl.load(p2_ptr)
    tl.store(beam_prev_ptr + 0, t1)
    tl.store(beam_prev_ptr + 1, t2)
    tl.store(prim_tok_ptr + 0, t1)
    tl.store(conf_ptr + 0, p1)
    tl.store(sec_tok_ptr + 0, t2)
    tl.store(sec_p_ptr + 0, p2)
    tl.store(p2_0_ptr + 0, p2)
    for j in tl.static_range(1, 7):
        tl.store(prim_tok_ptr + j, -1)
        tl.store(conf_ptr + j, 0.0)
    for j in tl.static_range(1, 4):
        tl.store(sec_tok_ptr + j, -1)
        tl.store(sec_p_ptr + j, 0.0)
    if HAS_ES:
        if tl.load(greedy_ptr) != 0:
            pc = tl.minimum(tl.maximum(p1, 1e-6), 1.0 - 1e-6)
            a0 = tl.load(cal_ptr + 0)
            b0 = tl.load(cal_ptr + 8)
            mk_ms = tl.load(cal_ptr + 20)
            db = tl.load(db_dev_ptr)
            lam = tl.load(lam_dev_ptr)
            c0 = 1.0 / (1.0 + tl.exp(-(a0 * (tl.log(pc) - tl.log(1.0 - pc)) + b0 + db)))
            surv = tl.load(surv_ptr) * c0
            tokens = tl.load(tok_ptr) + surv
            tl.store(surv_ptr, surv)
            tl.store(tok_ptr, tokens)
            u_cur = tokens - lam * tl.load(step_ptr + 1)
            best_u = tl.maximum(tl.load(bestu_ptr), u_cur)
            tl.store(bestu_ptr, best_u)
            if CAN_SET_SKIP:
                can_beat = 0
                thresh = best_u + lam * mk_ms
                for ww in tl.static_range(2, 8):
                    u_ub = (tokens + (ww - 1) * surv) - lam * tl.load(step_ptr + ww)
                    can_beat = tl.where(u_ub > thresh, 1, can_beat)
                if can_beat == 0:
                    tl.store(skip_ptr, 1)


@triton.jit
def _pack_beam2_step_kernel(
    bt1_ptr, bp1_ptr,
    prim_tok_ptr,
    conf_ptr,
    sec_tok_ptr,
    sec_p_ptr,
    skip_ptr, surv_ptr, tok_ptr, bestu_ptr, greedy_ptr, db_dev_ptr, lam_dev_ptr,
    cal_ptr, step_ptr,
    STEP: tl.constexpr,
    HAS_ES: tl.constexpr = False,
    CAN_SET_SKIP: tl.constexpr = False,
):
    p_tok = tl.load(bt1_ptr + 0)
    p_conf = tl.load(bp1_ptr + 0)
    tl.store(prim_tok_ptr + STEP, p_tok)
    tl.store(conf_ptr + STEP, p_conf)
    tl.store(sec_tok_ptr + STEP, tl.load(bt1_ptr + 1))
    tl.store(sec_p_ptr + STEP, tl.load(bp1_ptr + 1))
    if HAS_ES:
        if tl.load(greedy_ptr) != 0:
            pc = tl.minimum(tl.maximum(p_conf, 1e-6), 1.0 - 1e-6)
            a_s = tl.load(cal_ptr + STEP)
            b_s = tl.load(cal_ptr + 8 + STEP)
            mk_ms = tl.load(cal_ptr + 20)
            db = tl.load(db_dev_ptr)
            lam = tl.load(lam_dev_ptr)
            cs = 1.0 / (1.0 + tl.exp(-(a_s * (tl.log(pc) - tl.log(1.0 - pc)) + b_s + db)))
            surv = tl.load(surv_ptr) * cs
            tokens = tl.load(tok_ptr) + surv
            tl.store(surv_ptr, surv)
            tl.store(tok_ptr, tokens)
            u_cur = tokens - lam * tl.load(step_ptr + STEP + 1)
            best_u = tl.maximum(tl.load(bestu_ptr), u_cur)
            tl.store(bestu_ptr, best_u)
            if CAN_SET_SKIP:
                can_beat = 0
                thresh = best_u + lam * mk_ms
                for ww in tl.static_range(STEP + 2, 8):
                    u_ub = (tokens + (ww - STEP - 1) * surv) - lam * tl.load(step_ptr + ww)
                    can_beat = tl.where(u_ub > thresh, 1, can_beat)
                if can_beat == 0:
                    tl.store(skip_ptr, 1)


@triton.jit(do_not_specialize=["req_draft_stride"])
def _tree_dink_select_pack_kernel(
    idx_ptr,
    temp_ptr,
    prim_tok_ptr,
    conf_ptr,
    sec_tok_ptr,
    sec_p_ptr,
    cal_ptr,
    step_ptr,
    db_ptr,
    etok_ptr,
    ems_ptr,
    w_ptr,
    cpred_ptr,
    draft_ptr,
    req_draft_ptr,
    req_draft_stride,
    tree_active_ptr,
    n_live_ptr,
    tree_meta_ptr,
    tree_parent_ptr,
    tree_depth_ptr,
    tree_sconv_anc_ptr,
    tree_path_ptr,
    WS_MAX: tl.constexpr = 2,
    SIB_DISC: tl.constexpr = 0.0,
    SIB_PENALTY_MS: tl.constexpr = 0.10,
    SIB_SCALE: tl.constexpr = 0.85,
    P2_MIN: tl.constexpr = 0.10,
    WRITE_REQ_DRAFTS: tl.constexpr = False,
):
    """1-CTA GPU Exact Subtree Dinkelbach Tree Selector & Table Builder (< 2.5 us, 0 .item()).
    Evaluates rooted two-branch subtrees (w_p in 0..7, w_s in 0..min(WS_MAX, 7-w_p)),
    respects early-stopped primary tokens (prim_tok < 0), packs primary nodes first
    (so needs_compact==0 whenever primary wins!), and populates TreeSpecState + tree_meta[4]."""
    slot = tl.load(idx_ptr).to(tl.int64)
    safe_slot = tl.maximum(slot, 0)
    is_greedy = tl.where((slot >= 0) & (tl.load(temp_ptr + safe_slot) == 0.0), 1, 0)

    if is_greedy == 0:
        tl.store(tree_active_ptr, 0)
        tl.store(n_live_ptr, 8)
        tl.store(tree_meta_ptr + 0, 0)
        tl.store(tree_meta_ptr + 1, 7)
        tl.store(tree_meta_ptr + 2, 0)
        tl.store(tree_meta_ptr + 3, 8)
        for i in tl.static_range(8):
            tl.store(tree_parent_ptr + i, i - 1)
            tl.store(tree_depth_ptr + i, i)
            for l in tl.static_range(4):
                tl.store(tree_sconv_anc_ptr + i * 4 + l, i - l if i - l >= 0 else -1)
            for d in tl.static_range(8):
                tl.store(tree_path_ptr + i * 8 + d, d if d <= i else -1)
        if slot >= 0:
            for k in tl.static_range(7):
                pt = tl.load(prim_tok_ptr + k)
                tl.store(draft_ptr + k, pt)
                if WRITE_REQ_DRAFTS:
                    tl.store(req_draft_ptr + slot * req_draft_stride + k, pt)
        return

    db = tl.load(db_ptr + slot)
    static_lam = tl.load(cal_ptr + 18)
    gamma = tl.load(cal_ptr + 19)
    etok = tl.load(etok_ptr + slot)
    ems = tl.maximum(tl.load(ems_ptr + slot), 1.0)
    lam = tl.where(static_lam > 0.0, static_lam, gamma * (etok / ems))

    k_ar = tl.arange(0, 8)
    p_vec = tl.load(conf_ptr + k_ar, mask=k_ar < 7, other=0.5)
    pc_vec = tl.minimum(tl.maximum(p_vec, 1e-6), 1.0 - 1e-6)
    a_vec = tl.load(cal_ptr + k_ar, mask=k_ar < 7, other=0.5)
    b_vec = tl.load(cal_ptr + 8 + k_ar, mask=k_ar < 7, other=-0.4)
    c_vec = 1.0 / (1.0 + tl.exp(-(a_vec * (tl.log(pc_vec) - tl.log(1.0 - pc_vec)) + b_vec + db)))
    tl.store(cpred_ptr + slot * 7 + k_ar, c_vec, mask=k_ar < 7)

    c0 = tl.sum(tl.where(k_ar == 0, c_vec, 0.0))
    c1 = tl.sum(tl.where(k_ar == 1, c_vec, 0.0))
    c2 = tl.sum(tl.where(k_ar == 2, c_vec, 0.0))
    c3 = tl.sum(tl.where(k_ar == 3, c_vec, 0.0))
    c4 = tl.sum(tl.where(k_ar == 4, c_vec, 0.0))
    c5 = tl.sum(tl.where(k_ar == 5, c_vec, 0.0))
    c6 = tl.sum(tl.where(k_ar == 6, c_vec, 0.0))

    pt0 = tl.load(prim_tok_ptr + 0)
    pt1 = tl.load(prim_tok_ptr + 1)
    pt2 = tl.load(prim_tok_ptr + 2)
    pt3 = tl.load(prim_tok_ptr + 3)
    pt4 = tl.load(prim_tok_ptr + 4)
    pt5 = tl.load(prim_tok_ptr + 5)
    pt6 = tl.load(prim_tok_ptr + 6)

    p1_0 = tl.minimum(tl.maximum(tl.load(conf_ptr + 0), 1e-6), 1.0 - 1e-6)
    p2_0 = tl.maximum(tl.load(sec_p_ptr + 0), 0.0)
    q2_obs = tl.minimum(p2_0 / tl.maximum(1.0 - p1_0, 1e-4), 0.95)
    t2_0 = tl.load(sec_tok_ptr + 0)
    st1 = tl.load(sec_tok_ptr + 1)
    st2 = tl.load(sec_tok_ptr + 2)
    st3 = tl.load(sec_tok_ptr + 3)
    sec0_ok = (t2_0 >= 0) & (t2_0 != pt0) & (p2_0 >= P2_MIN)
    sec1_ok = sec0_ok & (st1 >= 0)
    sec2_ok = sec1_ok & (st2 >= 0)
    sec3_ok = sec2_ok & (st3 >= 0)
    s0 = tl.where(sec0_ok, (1.0 - c0) * q2_obs * SIB_SCALE, 0.0)

    a0 = tl.load(cal_ptr + 0)
    b0 = tl.load(cal_ptr + 8)
    sp1 = tl.minimum(tl.maximum(tl.load(sec_p_ptr + 1), 1e-6), 1.0 - 1e-6)
    sp2 = tl.minimum(tl.maximum(tl.load(sec_p_ptr + 2), 1e-6), 1.0 - 1e-6)
    sp3 = tl.minimum(tl.maximum(tl.load(sec_p_ptr + 3), 1e-6), 1.0 - 1e-6)
    cs1 = 1.0 / (1.0 + tl.exp(-(a0 * (tl.log(sp1) - tl.log(1.0 - sp1)) + b0 + db)))
    cs2 = 1.0 / (1.0 + tl.exp(-(a0 * (tl.log(sp2) - tl.log(1.0 - sp2)) + b0 + db)))
    cs3 = 1.0 / (1.0 + tl.exp(-(a0 * (tl.log(sp3) - tl.log(1.0 - sp3)) + b0 + db)))

    gs0 = 0.0
    gs1 = s0
    gs2 = gs1 + s0 * cs1
    gs3 = gs2 + s0 * cs1 * cs2
    gs4 = gs3 + s0 * cs1 * cs2 * cs3

    best_obj = -1e18
    best_wp = tl.zeros((), tl.int32)
    best_ws = tl.zeros((), tl.int32)

    surv_p = 1.0
    gp = 1.0
    prim_alive = tl.full((), 1, tl.int32)
    for wp in tl.static_range(8):
        if wp > 0:
            pk = (
                pt0 if wp == 1 else (
                pt1 if wp == 2 else (
                pt2 if wp == 3 else (
                pt3 if wp == 4 else (
                pt4 if wp == 5 else (
                pt5 if wp == 6 else pt6)))))
            )
            prim_alive = tl.where(pk >= 0, prim_alive, 0)
            ck = (
                c0 if wp == 1 else (
                c1 if wp == 2 else (
                c2 if wp == 3 else (
                c3 if wp == 4 else (
                c4 if wp == 5 else (
                c5 if wp == 6 else c6)))))
            )
            surv_p = surv_p * ck
            gp = gp + surv_p
        for ws in tl.static_range(WS_MAX + 1):
            if (wp + ws <= 7) and ((ws == 0) or (wp >= 1)):
                gs = gs0 if ws == 0 else (gs1 if ws == 1 else (gs2 if ws == 2 else (gs3 if ws == 3 else gs4)))
                ws_ok = (
                    True if ws == 0 else (
                    sec0_ok if ws == 1 else (
                    sec1_ok if ws == 2 else (
                    sec2_ok if ws == 3 else sec3_ok)))
                )
                tot_w = wp + ws
                if ws == 0:
                    lat = tl.load(step_ptr + tot_w)
                else:
                    lat_hi = tl.load(step_ptr + tot_w)
                    lat_lo = tl.load(step_ptr + tot_w - 1)
                    lat = lat_hi + SIB_PENALTY_MS - SIB_DISC * (lat_hi - lat_lo)
                obj = (gp + gs) - lam * lat
                better = (prim_alive != 0) & ws_ok & (obj > best_obj)
                best_obj = tl.where(better, obj, best_obj)
                best_wp = tl.where(better, wp, best_wp)
                best_ws = tl.where(better, ws, best_ws)

    n_live = 1 + best_wp + best_ws
    is_br = tl.where(best_ws > 0, 1, 0)
    tl.store(w_ptr + slot, best_wp)
    tl.store(tree_active_ptr, is_br)
    tl.store(n_live_ptr, n_live)
    tl.store(tree_meta_ptr + 0, is_br)
    tl.store(tree_meta_ptr + 1, best_wp)
    tl.store(tree_meta_ptr + 2, best_ws)
    tl.store(tree_meta_ptr + 3, n_live)

    for k in tl.static_range(7):
        is_p = k < best_wp
        is_s = (k >= best_wp) & (k < best_wp + best_ws)
        s_idx = tl.maximum(0, tl.minimum(3, k - best_wp))
        pt = tl.load(prim_tok_ptr + k)
        st = tl.load(sec_tok_ptr + s_idx)
        tok = tl.where(is_p, pt, tl.where(is_s, st, -1))
        tl.store(draft_ptr + k, tok)
        if WRITE_REQ_DRAFTS:
            tl.store(req_draft_ptr + slot * req_draft_stride + k, tok)

    tl.store(tree_parent_ptr + 0, -1)
    tl.store(tree_depth_ptr + 0, 0)
    tl.store(tree_sconv_anc_ptr + 0, 0)
    tl.store(tree_sconv_anc_ptr + 1, -1)
    tl.store(tree_sconv_anc_ptr + 2, -1)
    tl.store(tree_sconv_anc_ptr + 3, -1)
    tl.store(tree_path_ptr + 0, 0)
    for d in tl.static_range(1, 8):
        tl.store(tree_path_ptr + d, -1)

    for i in tl.static_range(1, 8):
        is_p = i <= best_wp
        is_s = (i > best_wp) & (i < n_live)
        s_pos = i - best_wp
        par = tl.where(is_p, i - 1, tl.where(is_s, tl.where(s_pos == 1, 0, i - 1), -1))
        dep = tl.where(is_p, i, tl.where(is_s, s_pos, 0))
        tl.store(tree_parent_ptr + i, par)
        tl.store(tree_depth_ptr + i, dep)
        for l in tl.static_range(4):
            anc_p = tl.where(l <= i, i - l, -1)
            anc_s = tl.where(l < s_pos, i - l, tl.where(l == s_pos, 0, -1))
            anc = tl.where(is_p, anc_p, tl.where(is_s, anc_s, tl.where(l == 0, i, -1)))
            tl.store(tree_sconv_anc_ptr + i * 4 + l, anc)
        for d in tl.static_range(8):
            pp = tl.where(d <= i, d, -1)
            ps = tl.where(d == 0, 0, tl.where(d <= s_pos, best_wp + d, -1))
            pv = tl.where(is_p, pp, tl.where(is_s, ps, tl.where(d == 0, 0, -1)))
            tl.store(tree_path_ptr + i * 8 + d, pv)

class Inkling2DSparkSpeculator(DSparkSpeculator):
    _speculator_name = "Inkling2 DSpark"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        config = self.draft_model_config.hf_config
        assert not self.sample_from_anchor
        assert self._draft_topk is None and not self.enable_adaptive_verification
        self.hidden_states = torch.zeros(
            self.max_num_tokens,
            len(config.target_layer_ids) * config.hidden_size,
            dtype=self.dtype,
            device=device,
        )
        # combine_hidden_states is the identity: store features eagerly.
        self._aux_staging = None
        # Read by prepare_context_anchor inside the FULL draft graph.
        self._num_rejected = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, device=device
        )
        self.f_rnn = torch.zeros(
            self.max_num_reqs, config.hidden_size, dtype=self.dtype, device=device
        )
        self._fuse_dink_es = os.environ.get("INKLING_FUSE_DINK_ES", "1").strip().lower() not in (
            "0", "", "off", "false", "no"
        )
        self._es_fused_step_idx_ptr: int = 0
        self._graph_vw_conf_on = os.environ.get("INKLING_GRAPH_VW_CONF", "1").strip().lower() not in (
            "0", "", "off", "false", "no"
        )
        self._graph_req_drafts: torch.Tensor | None = None
        self._graph_vw_conf_captured: dict[int, tuple[int, float, int]] = {}
        self._last_sample_seq_eager: bool = False
        self._last_tree_handled: bool = False
        # Option A: Suffix Automaton Hybrid (INKLING_SAM_HYBRID=1, mode=gpu|cpu)
        self._sam_hybrid_on = os.environ.get("INKLING_SAM_HYBRID", "0").strip().lower() not in (
            "0", "", "off", "false", "no"
        )
        self._sam_mode = os.environ.get("INKLING_SAM_MODE", "gpu").strip().lower()
        if self._sam_mode not in ("gpu", "cpu"):
            self._sam_mode = "gpu"
        self._sam_min_match = max(2, int(os.environ.get("INKLING_SAM_MIN_MATCH", "4") or "4"))
        self._sam_boost_base = float(os.environ.get("INKLING_SAM_BOOST_BASE", "0.35") or "0.35")
        self._sam_boost_max = float(os.environ.get("INKLING_SAM_BOOST_MAX", "1.40") or "1.40")
        self._sam_ext_surv = float(os.environ.get("INKLING_SAM_EXT_SURV", "0.25") or "0.25")
        self._sam_fill_es = os.environ.get("INKLING_SAM_FILL_ES", "1").strip().lower() not in (
            "0", "", "off", "false", "no"
        )
        self._sam_allow_sync = os.environ.get("INKLING_SAM_ALLOW_SYNC", "0").strip().lower() in (
            "1", "on", "true", "yes"
        )
        self._sam_batch_on = os.environ.get("INKLING_SAM_BATCH", "0").strip().lower() not in (
            "0", "", "off", "false", "no"
        )
        self._sam_max_slots = min(
            self.max_num_reqs,
            max(1, int(os.environ.get("INKLING_SAM_MAX_SLOTS", "32" if self._sam_batch_on else "4") or "4")),
        )
        _sam_ring_def = 16384 if self._sam_max_slots > 4 else 65536
        _sam_ring_raw = int(os.environ.get("INKLING_SAM_RING_SZ", str(_sam_ring_def)) or str(_sam_ring_def))
        # Ensure power-of-2 >= 1024 for bitmask indexing inside GPU SAM kernels.
        _sam_ring_sz = 1024
        while _sam_ring_sz < _sam_ring_raw:
            _sam_ring_sz <<= 1
        self._sam_ring_sz = _sam_ring_sz
        _sam_ht_raw = int(os.environ.get("INKLING_SAM_HT_SZ", str(_sam_ring_sz)) or str(_sam_ring_sz))
        _sam_ht_sz = _sam_ring_sz
        while _sam_ht_sz < _sam_ht_raw:
            _sam_ht_sz <<= 1
        self._sam_ht_sz = _sam_ht_sz
        self._sam_max_ingest = max(
            256,
            min(
                self._sam_ring_sz,
                int(os.environ.get("INKLING_SAM_MAX_INGEST", "4096") or "4096"),
            ),
        )
        self._sam_override_low_conf = os.environ.get(
            "INKLING_SAM_OVERRIDE_LOW_CONF", "0"
        ).strip().lower() in ("1", "on", "true", "yes")
        self._sam_override_conf_thresh = float(
            os.environ.get("INKLING_SAM_OVERRIDE_CONF_THRESH", "0.25") or "0.25"
        )
        self._sam_override_min_match = max(
            self._sam_min_match,
            int(os.environ.get("INKLING_SAM_OVERRIDE_MIN_MATCH", "5") or "5"),
        )
        self._sam_override_cal_invert = os.environ.get(
            "INKLING_SAM_OVERRIDE_CAL_INVERT", "1"
        ).strip().lower() not in ("0", "", "off", "false", "no")
        self._sam_allow_temp = os.environ.get(
            "INKLING_SAM_ALLOW_TEMP", "0"
        ).strip().lower() in ("1", "on", "true", "yes")
        # Wave-V11 GPU SAM Deep Upgrades (ALL strictly default-OFF -> 0 extra VRAM / 100% V10 invariant!):
        #   - INKLING_SAM_CONF_BOOST=1  : calibration-inverted DSpark+SAM agreement floor (ml >= 2)
        #   - INKLING_SAM_LONG_PROMPT=1 : 128K/1M Strided-4 Slot-Tagged Prompt Continuation Cache (13 MiB)
        #   - INKLING_SAM_SHARED_POOL=1 : 2.0 MiB lock-free Cross-Request Shared N-gram Pool (C=4..32)
        self._sam_conf_boost_on = os.environ.get(
            "INKLING_SAM_CONF_BOOST", "0"
        ).strip().lower() in ("1", "on", "true", "yes")
        self._sam_conf_boost_ml2 = float(
            os.environ.get("INKLING_SAM_CONF_BOOST_ML2", "0.76") or "0.76"
        )
        self._sam_conf_boost_ml3 = float(
            os.environ.get("INKLING_SAM_CONF_BOOST_ML3", "0.85") or "0.85"
        )
        self._sam_conf_boost_ml4 = float(
            os.environ.get("INKLING_SAM_CONF_BOOST_ML4", "0.92") or "0.92"
        )
        _ht2_raw = int(os.environ.get("INKLING_SAM_HT2_SZ", "4096") or "4096")
        _ht2_sz = 1024
        while _ht2_sz < _ht2_raw:
            _ht2_sz <<= 1
        self._sam_ht2_sz = _ht2_sz
        self._sam_long_prompt_on = os.environ.get(
            "INKLING_SAM_LONG_PROMPT", "0"
        ).strip().lower() in ("1", "on", "true", "yes")
        _lpc_raw = int(os.environ.get("INKLING_SAM_LPC_CAP", "262144") or "262144")
        _lpc_cap = 16384
        while _lpc_cap < _lpc_raw:
            _lpc_cap <<= 1
        self._sam_lpc_cap = _lpc_cap
        _lpc_s_raw = int(os.environ.get("INKLING_SAM_LPC_STRIDE", "4") or "4")
        self._sam_lpc_stride = 1 if _lpc_s_raw <= 1 else (2 if _lpc_s_raw == 2 else 4)
        self._sam_shared_pool_on = os.environ.get(
            "INKLING_SAM_SHARED_POOL", "0"
        ).strip().lower() in ("1", "on", "true", "yes")
        _sp_raw = int(os.environ.get("INKLING_SAM_SHARED_CAP", "65536") or "65536")
        _sp_cap = 4096
        while _sp_cap < _sp_raw:
            _sp_cap <<= 1
        self._sam_sp_cap = _sp_cap
        self._sam_ht2_last: torch.Tensor | None = None
        self._sam_slot_gen: torch.Tensor | None = None
        self._sam_lpc_key: torch.Tensor | None = None
        self._sam_lpc_cont: torch.Tensor | None = None
        self._sam_sp_tag: torch.Tensor | None = None
        self._sam_sp_cont: torch.Tensor | None = None
        self._sam_ext_k15_on = os.environ.get(
            "INKLING_SAM_EXT_K15", "0"
        ).strip().lower() in ("1", "on", "true", "yes")
        _ext_mb_def = "2" if self._sam_ext_k15_on else "1"
        self._sam_ext_max_b = max(
            1, min(2, int(os.environ.get("INKLING_SAM_EXT_MAX_B", _ext_mb_def) or _ext_mb_def))
        )
        self._fuse_dink_sam_ingest = os.environ.get(
            "INKLING_FUSE_DINK_SAM_INGEST", "0"
        ).strip().lower() in ("1", "on", "true", "yes")
        self._spec_fastpath_on = os.environ.get(
            "INKLING_SPEC_FASTPATH", "0"
        ).strip().lower() in ("1", "on", "true", "yes")
        self._pf_trim_draft_ctxkv = os.environ.get(
            "INKLING_PF_TRIM_DRAFT_CTXKV", "0"
        ).strip().lower() not in ("0", "", "off", "false", "no")
        self._deferred_dink_update: tuple | None = None
        self._sam_gpu_ready = False
        self._sam_advisors: dict[int, Any] = {}
        self._sam_boost_in_graph_captured: set[int] = set()
        self._sam_cap_g_lut: list[int | None] = [None] * (self.max_num_reqs + 1)
        self._vw_cap_g_lut: list[int | None] = [None] * (self.max_num_reqs + 1)
        self._draft_cap_g_lut: list[int | None] = [None] * (self.max_num_reqs + 1)
        self._fused_prep_cache: tuple | None = None
        # Option B: 0-ULP Beam-2 Tree Speculation (INKLING_TREE_SPEC=1, M=8, capture-safe)
        self._tree_spec_on = os.environ.get("INKLING_TREE_SPEC", "0").strip().lower() not in (
            "0", "", "off", "false", "no"
        )
        self._tree_spec_b1_only = os.environ.get("INKLING_TREE_SPEC_B1_ONLY", "0").strip().lower() not in (
            "0", "", "off", "false", "no"
        )
        _tree_max_b_raw = os.environ.get("INKLING_TREE_SPEC_MAX_B", "").strip()
        self._tree_spec_max_b = (
            1 if self._tree_spec_b1_only else (max(1, int(_tree_max_b_raw)) if _tree_max_b_raw else 1)
        )
        _tree_ws_raw = os.environ.get("INKLING_TREE_WS_MAX", "").strip()
        _tree_ws_def = "1" if self._tree_spec_b1_only else "2"
        self._tree_ws_max = max(1, min(4, int(_tree_ws_raw or _tree_ws_def)))
        self._tree_sib_disc = float(os.environ.get("INKLING_TREE_SIB_DISC", "0.0") or "0.0")
        self._tree_sib_penalty_ms = float(os.environ.get("INKLING_TREE_SIB_PENALTY_MS", "0.10") or "0.10")
        self._tree_sib_scale = float(os.environ.get("INKLING_TREE_SIB_SCALE", "0.85") or "0.85")
        self._tree_p2_min = float(os.environ.get("INKLING_TREE_P2_MIN", "0.10") or "0.10")
        self._tree_state = None
        self._tree_runtime_mod = None
        self._tree_graph_captured: dict[int, int] = {}
        # Decouple DSpark native 8-token query block (1 anchor + 7 masks) from wider target
        # verification windows (e.g. M=16 / SPEC_K=15 with INKLING_SAM_HYBRID=1):
        #   - num_query_per_req = 1 + _dspark_k = 8 (matches embed_block / draft_block_size = 8)
        #   - _anchor_idx = [0, 8, 16, ...]
        #   - num_speculative_steps stays 15 (draft_tokens width=15, _num_graph_context_tokens=16*B)
        draft_block_size = int(getattr(config, "draft_block_size", 8))
        self._dspark_k = min(self.num_speculative_steps, draft_block_size - 1)
        if self.num_speculative_steps > self._dspark_k and self._sam_hybrid_on:
            self.num_query_per_req = 1 + self._dspark_k
            self._anchor_idx = (
                torch.arange(self.max_num_reqs, dtype=torch.int64, device=device)
                * self.num_query_per_req
            )
            logger.info(
                "Inkling2 DSpark M=%d (SPEC_K=%d) decoupled from native draft_block_size=%d "
                "(num_query_per_req=%d, dspark_k=%d, sam_ext_cols=%d)",
                self.num_speculative_steps + 1,
                self.num_speculative_steps,
                draft_block_size,
                self.num_query_per_req,
                self._dspark_k,
                self.num_speculative_steps - self._dspark_k,
            )
        if self._sam_hybrid_on and self._sam_mode == "cpu":
            self._graph_vw_conf_on = False
        self._init_batch_spec_policy()
        self._init_verify_w(device)
        self._init_sam_and_tree(device)
        # Optional Big-Core CPU pinning (INKLING_HOST_PIN=1) and per-step glue CUDA graph (INKLING_DTGRAPH=...)
        if os.environ.get("INKLING_HOST_PIN", "0").strip().lower() in ("1", "on", "true", "yes"):
            from .ops import host_pin

            host_pin.pin_worker(logger)
        if os.environ.get("INKLING_DTGRAPH", "").strip().lower() not in ("", "0", "off", "false", "no"):
            from .ops import dtgraph

            dtgraph.install()
        if os.environ.get("INKLING_CAPTURE_HI_PRIO", "0").strip().lower() in ("1", "on", "true", "yes"):
            _install_hi_prio_capture(logger)
        self._zero_copy_store = os.environ.get("INKLING_ZERO_COPY_STORE", "1").strip().lower() not in (
            "0", "off", "false", "no"
        )
        self._fuse_dflash_prep = os.environ.get("INKLING_FUSE_DFLASH_PREP", "0").strip().lower() in (
            "1", "on", "true", "yes"
        )
        self._fuse_ctx_anchor = os.environ.get("INKLING_FUSE_CTX_ANCHOR", "0").strip().lower() in (
            "1", "on", "true", "yes"
        )
        if self._fuse_dflash_prep:
            _install_fused_dflash_prep()
        self._ctx_ovl_enabled = os.environ.get(
            "INKLING_DRAFT_CTXKV_OVERLAP", "1"
        ).strip().lower() not in ("0", "", "off", "false", "no")
        self._ctx_ovl_max_reqs = int(
            os.environ.get("INKLING_DRAFT_CTXKV_OVERLAP_MAX_REQS", "4") or "4"
        )
        self._ctx_ovl_stream: torch.cuda.Stream | None = None
        self._ctx_ovl_ev_in: list[torch.cuda.Event] | None = None
        self._ctx_ovl_ev_out: list[torch.cuda.Event] | None = None
        self._ctx_ovl_ready = False

    def _init_ctx_overlap(self) -> None:
        """Pre-allocate dedicated side stream & materialized events outside CUDA graph capture
        for staggered drafter ctx_kv (L0..L4) + f_rnn overlap."""
        if not self._ctx_ovl_enabled or self._ctx_ovl_ready:
            return
        if self.device.type != "cuda" or torch.cuda.is_current_stream_capturing():
            return
        try:
            decoder = getattr(self.model, "model", None)
            n_layers = len(decoder.layers) if decoder is not None and hasattr(decoder, "layers") else 0
            if n_layers <= 0:
                return
            st = torch.cuda.Stream(device=self.device)
            ev_in = [torch.cuda.Event() for _ in range(n_layers + 1)]
            ev_out = [torch.cuda.Event() for _ in range(n_layers + 1)]
            cur = torch.cuda.current_stream(self.device)
            for ev in ev_in:
                ev.record(cur)
            for ev in ev_out:
                ev.record(cur)
            # Skip 32 MiB per-stream cuBLASLt workspace ONLY when ctx_ovl_stream is guaranteed to
            # stay inside Triton u4_panel (no BF16 cuBLAS fallback and no high-M FP8 _scaled_mm).
            from .ops.skinny_int4 import _U4_PANEL_MAX_M as _SI4_MAX_M
            _max_ctx_toks = self._num_graph_context_tokens(min(self.max_num_reqs, self._ctx_ovl_max_reqs))
            _fp8_hi_m = any(
                getattr(layer.self_attn.kv_proj_ctx, "fp8_hi_wt", None) is not None
                for layer in decoder.layers
            )
            needs_cublas = (
                _max_ctx_toks > _SI4_MAX_M
                or _fp8_hi_m
                or not all(
                    getattr(layer.self_attn.kv_proj_ctx, "w4_meta", None) is not None
                    for layer in decoder.layers
                )
            )
            if needs_cublas:
                with torch.cuda.stream(st):
                    _z = torch.empty((16, 64), dtype=self.dtype, device=self.device)
                    _ = torch.mm(_z, _z.t())
                    if _fp8_hi_m:
                        try:
                            _a8 = torch.zeros((16, 64), dtype=torch.float8_e4m3fn, device=self.device)
                            _b8 = torch.zeros((64, 64), dtype=torch.float8_e4m3fn, device=self.device).t()
                            _sa = torch.ones((16, 1), dtype=torch.float32, device=self.device)
                            _sb = torch.ones((1, 64), dtype=torch.float32, device=self.device)
                            _ = torch._scaled_mm(
                                _a8, _b8, scale_a=_sa, scale_b=_sb,
                                out_dtype=torch.bfloat16, use_fast_accum=False,
                            )
                        except Exception:
                            pass
                st.synchronize()
            # Pre-allocate disjoint fused split-K scratch banks (bank 0 main/capture, bank 1 ctx_stream)
            # and fused Markov fold semaphores strictly outside graph capture.
            try:
                from .ops import skinny_int4 as _si4

                _si4.prealloc_stream_scratch(self.device)
                _si4.prealloc_stream_scratch(self.device, st)
            except Exception:
                pass
            try:
                from .ops import dspark as _dops_mod

                _dops_mod.prealloc_fold_sem(self.device)
            except Exception:
                pass
            self._ctx_ovl_stream = st
            self._ctx_ovl_ev_in = ev_in
            self._ctx_ovl_ev_out = ev_out
            self._ctx_ovl_ready = True
            logger.info(
                "INKLING_DRAFT_CTXKV_OVERLAP=1: staggered ctx_kv (L0..L%d) + f_rnn ready (max_reqs=%d)",
                n_layers - 1,
                self._ctx_ovl_max_reqs,
            )
        except Exception as e:  # noqa: BLE001
            self._ctx_ovl_ready = False
            logger.warning("INKLING_DRAFT_CTXKV_OVERLAP init failed (%r); using serial ctx_kv", e)

    def _ctx_ovl_can_defer(self, num_reqs: int) -> bool:
        return (
            self._ctx_ovl_ready
            and self._ctx_ovl_stream is not None
            and self._ctx_ovl_ev_in is not None
            and self._ctx_ovl_ev_out is not None
            and 0 < num_reqs <= self._ctx_ovl_max_reqs
            and self._aux_staging is None
            and getattr(self, "dcp_size", 1) == 1
            and hasattr(self.model, "model")
            and len(self.model.model.layers) > 0
            and getattr(self.model.model, "_ctx_sched", None) is None
        )

    def _prepare_graph_forward(
        self, num_reqs: int, attn_metadata: dict[str, Any] | None
    ) -> None:
        if not self._ctx_ovl_can_defer(num_reqs):
            super()._prepare_graph_forward(num_reqs, attn_metadata)
            return
        from .dspark import _CtxOverlapSchedule

        num_context = self._num_graph_context_tokens(num_reqs)
        decoder = self.model.model
        n_layers = len(decoder.layers)
        if self._layer_group_idx is not None:
            context_slots: list[torch.Tensor | None] = [
                self._context_slot_mappings[gidx][0:num_context]
                for gidx in self._layer_group_idx
            ]
        else:
            context_slots = [self._context_slot_mappings[0][0:num_context]] * n_layers

        def _frnn() -> None:
            self.prepare_context_anchor(
                num_reqs, self.target_input_buffers.query_start_loc, self._num_rejected
            )

        assert (
            self._ctx_ovl_stream is not None
            and self._ctx_ovl_ev_in is not None
            and self._ctx_ovl_ev_out is not None
        )
        decoder._ctx_sched = _CtxOverlapSchedule(
            decoder=decoder,
            context_states=self.hidden_states[0:num_context],
            context_positions=self.context_positions[0:num_context],
            context_slots=context_slots,
            frnn_fn=_frnn,
            stream=self._ctx_ovl_stream,
            ev_in=self._ctx_ovl_ev_in,
            ev_out=self._ctx_ovl_ev_out,
        )
        if attn_metadata is not None and self.supports_decode_metadata_update:
            self._update_draft_decode_metadata(attn_metadata, num_reqs)

    def _generate_draft(self, *args: Any, **kwargs: Any) -> None:
        try:
            super()._generate_draft(*args, **kwargs)
        finally:
            decoder = getattr(self.model, "model", None)
            sched = getattr(decoder, "_ctx_sched", None) if decoder is not None else None
            if sched is not None:
                decoder._ctx_sched = None
                sched.join_all()

    def set_req_drafts_buffer(self, req_drafts: torch.Tensor | None) -> None:
        """Register persistent runner.req_states.draft_tokens before capture() so
        in-graph _vw_conf_kernel can write masked drafts directly with zero eager gap."""
        if (
            isinstance(req_drafts, torch.Tensor)
            and req_drafts.dim() == 2
            and req_drafts.dtype == self.draft_tokens.dtype
            and req_drafts.device == self.draft_tokens.device
            and req_drafts.stride(1) == 1
            and req_drafts.shape[1] == self.num_speculative_steps
        ):
            self._graph_req_drafts = req_drafts

    def capture(self) -> None:
        self._init_ctx_overlap()
        super().capture()
        if os.environ.get("INKLING_HOST_PIN", "0").strip().lower() in ("1", "on", "true", "yes"):
            from .ops import host_pin

            host_pin.freeze_worker_gc(logger)

    def _prepare_graph_replay(
        self,
        input_batch: Any,
        num_graph_reqs: int,
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_rejected: torch.Tensor,
        dummy_run: bool,
    ) -> None:
        """Zero-copy num_rejected variant of DFlashSpeculator._prepare_graph_replay:
        when _prepare_dflash_inputs_5g_kernel already wrote self._num_rejected[:num_reqs],
        skip the redundant 4-byte copy_ (memcpy32_post, ~1.2-2.0 us/step)."""
        if (
            not dummy_run
            and getattr(_FUSED_PREP_TLS, "spec", None) is self
            and getattr(_FUSED_PREP_TLS, "num_rejected_written", False)
        ):
            num_target_tokens = input_batch.num_tokens
            num_context = self._num_graph_context_tokens(num_graph_reqs)
            self._store_hidden_states(
                num_target_tokens, last_hidden_states, aux_hidden_states, num_context
            )
            if num_target_tokens < num_context:
                _pad_rng = getattr(_FUSED_PREP_TLS, "ctx_pad_range", None)
                if not (
                    self._spec_fastpath_on
                    and _pad_rng is not None
                    and _pad_rng[0] == num_target_tokens
                    and _pad_rng[1] >= num_context
                ):
                    pad_id = getattr(_FUSED_PREP_TLS, "pad_slot_id", None)
                    if pad_id is None:
                        from vllm.v1.attention.backends.utils import PAD_SLOT_ID

                        pad_id = PAD_SLOT_ID
                        _FUSED_PREP_TLS.pad_slot_id = pad_id
                    self._context_slot_mappings[:, num_target_tokens:num_context].fill_(pad_id)
            elif num_target_tokens > num_context:
                self._precompute_context_kv(num_context, num_target_tokens)
            return
        super()._prepare_graph_replay(
            input_batch,
            num_graph_reqs,
            last_hidden_states,
            aux_hidden_states,
            num_rejected,
            dummy_run,
        )

    def _store_hidden_states(
        self,
        num_tokens: int,
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_graph_tokens: int = 0,
    ) -> None:
        """Zero-copy variant of DFlashSpeculator._store_hidden_states: since
        Inkling2DSparkModel.combine_hidden_states is the identity, cat directly into
        self.hidden_states[:num_tokens] (eliminates 590 KB temporary alloc + DtoD copy)."""
        if self._zero_copy_store and self._aux_staging is None:
            if aux_hidden_states:
                max_keep = getattr(self, "_pf_max_draft_keep", 0)
                if (
                    max_keep > 0
                    and bool(getattr(self.model, "_pf_single_req_chunk", False))
                    and num_tokens > max_keep + max(num_graph_tokens, 32)
                    and not torch.cuda.is_current_stream_capturing()
                ):
                    head_n = max(int(num_graph_tokens), 32)
                    tail_start = num_tokens - max_keep
                    torch.cat(
                        [h[:head_n] for h in aux_hidden_states],
                        dim=-1,
                        out=self.hidden_states[:head_n],
                    )
                    torch.cat(
                        [h[tail_start:num_tokens] for h in aux_hidden_states],
                        dim=-1,
                        out=self.hidden_states[tail_start:num_tokens],
                    )
                else:
                    torch.cat(
                        [h[:num_tokens] for h in aux_hidden_states],
                        dim=-1,
                        out=self.hidden_states[:num_tokens],
                    )
            else:
                self.hidden_states[:num_tokens].copy_(last_hidden_states[:num_tokens])
        else:
            super()._store_hidden_states(
                num_tokens, last_hidden_states, aux_hidden_states, num_graph_tokens
            )
        if (
            self._tree_spec_on
            and self._tree_state is not None
            and self._tree_runtime_mod is not None
            and not self._tree_runtime_mod.TREE_STATE.get("suppressed", False)
            and (
                not self._tree_spec_b1_only
                or int(self._tree_runtime_mod.TREE_STATE.get("active_b", 1)) <= self._tree_spec_max_b
            )
            and num_tokens == self._tree_state.max_m
            and aux_hidden_states
        ):
            self._tree_runtime_mod.compact_tree_packed_aux(
                self._tree_state,
                self.hidden_states,
                len(aux_hidden_states),
                int(aux_hidden_states[0].shape[-1]),
                use_pdl=False,
            )
            self._tree_state.needs_compact.zero_()

    def _init_batch_spec_policy(self) -> None:
        """Build batch-size -> K_cap / tau lookup tables from vLLM Dynamic SD config
        (num_speculative_tokens_per_batch_size) and/or env overrides.
        When all schedules are unset, B=1 and all B take zero-overhead fast paths."""
        n = self.max_num_reqs
        k = self.num_speculative_steps
        self._dyn_sd_k: list[int] | None = None
        spec_cfg = self.vllm_config.speculative_config
        if spec_cfg is not None and getattr(spec_cfg, "num_speculative_tokens_per_batch_size", None):
            from vllm.v1.spec_decode.dynamic.utils import build_dynamic_sd_schedule_lookup

            self._dyn_sd_k = build_dynamic_sd_schedule_lookup(
                spec_cfg.num_speculative_tokens_per_batch_size,
                vllm_max_batch_size=n,
                vllm_num_speculative_tokens=k,
            )
        else:
            raw_bs = os.environ.get("INKLING_BATCH_SPEC_SCHEDULE", "").strip()
            if raw_bs:
                if raw_bs.startswith("["):
                    import json as _json
                    sched = _json.loads(raw_bs)
                    tbl = [k] * (n + 1)
                    for item in sched:
                        if len(item) == 3:
                            lo_b, hi_b, kv = int(item[0]), int(item[1]), int(item[2])
                            for b in range(max(1, lo_b), min(n, hi_b) + 1):
                                tbl[b] = max(0, min(k, kv))
                        elif len(item) == 2:
                            b_idx, kv = int(item[0]), int(item[1])
                            if 1 <= b_idx <= n:
                                tbl[b_idx] = max(0, min(k, kv))
                    self._dyn_sd_k = tbl
                else:
                    self._dyn_sd_k = _parse_batch_schedule_int(raw_bs, n, k, lo=0, hi=k)
        # Optional env-only batch cap on verify width W(B) (e.g. "1:7,2:5,4:3,8:1,16:0,32:0").
        raw_cap = os.environ.get("INKLING_VERIFY_W_BATCH_CAP", "").strip()
        self._vw_batch_cap = _parse_batch_schedule_int(raw_cap, n, k, lo=0, hi=k)
        # Optional env-only batch-aware tau(B) (applies to conf, conf+ema, AND dink for B > 1!
        # Use -1 for B=1 so B=1 stays 100% bit-identical to live S8-P3 dink, e.g.:
        # INKLING_DINK_BATCH_TAU="1:-1,4:0.25,8:0.32,12:0.50,24:0.36,32:0.30").
        raw_tau = (
            os.environ.get("INKLING_DINK_BATCH_TAU", "").strip()
            or os.environ.get("INKLING_CONF_TAU_BATCH", "").strip()
        )
        self._vw_tau_batch_on = bool(raw_tau)
        self._vw_tau_batch_interp = os.environ.get(
            "INKLING_DINK_BATCH_TAU_INTERP", "0"
        ).strip().lower() in ("1", "true", "on", "yes")
        self._vw_tau_batch = _parse_batch_schedule_float(
            raw_tau, n, -1.0, interp=self._vw_tau_batch_interp
        )
        self._graph_vw_uniform_guard = os.environ.get(
            "INKLING_GRAPH_VW_CONF_UNIFORM_GUARD", "0"
        ).strip().lower() in ("1", "true", "on", "yes")
        # Wave-V11 opt-in algorithmic levers (ALL strictly default-OFF -> 100% V10 invariant!):
        #   - INKLING_HETERO_BUDGET_SPEC=1 : 1-CTA zero-sync GPU Exact Batch Water-Filling (B=2..32)
        #   - INKLING_CTX_ADAPTIVE_TAU=1   : live GPU seq_lens context-aware tau(B, ctx) scaling
        self._hetero_budget_on = os.environ.get(
            "INKLING_HETERO_BUDGET_SPEC", "0"
        ).strip().lower() in ("1", "true", "on", "yes")
        self._hetero_budget_m_cap = max(
            0, int(os.environ.get("INKLING_HETERO_BUDGET_M_CAP", "0") or "0")
        )
        self._hetero_budget_tau_min = max(
            0.0, float(os.environ.get("INKLING_HETERO_BUDGET_TAU_MIN", "0.0") or "0.0")
        )
        self._ctx_adaptive_tau_on = os.environ.get(
            "INKLING_CTX_ADAPTIVE_TAU", "0"
        ).strip().lower() in ("1", "true", "on", "yes")
        self._ctx_adaptive_tau_floor = max(
            0.01, float(os.environ.get("INKLING_CTX_ADAPTIVE_TAU_FLOOR", "0.04") or "0.04")
        )
        self._wf_step_tbl: torch.Tensor | None = None
        self._wf_dummy_seq: torch.Tensor | None = None
        self._l2pf_target_l0_max_reqs = max(
            1, int(os.environ.get("INKLING_L2PF_TARGET_L0_MAX_REQS", "2") or "2")
        )
        self._l2pf_fold_max_reqs = max(
            1, int(os.environ.get("INKLING_L2PF_FOLD_MAX_REQS", "1") or "1")
        )
        self._batch_policy_active = bool(self._dyn_sd_k is not None or raw_cap or raw_tau)
        # Effective per-batch Markov step budget for captured draft graphs:
        #   - If Dynamic SD is configured, respect dynamic_sd_lookup[b] (applies to greedy AND sampled!).
        #   - Never truncate below what sampling requests need unless Dynamic SD itself truncates.
        draft_k_dense = list(self._dyn_sd_k) if self._dyn_sd_k is not None else [k] * (n + 1)
        try:
            cap_sizes = self.vllm_config.get_draft_decode_cudagraph_request_sizes()
        except Exception:
            cap_sizes = self.vllm_config.compilation_config.decode_cudagraph_capture_request_sizes
        self._draft_graph_cap_sizes = (
            sorted({int(s) for s in cap_sizes if 1 <= int(s) <= n}) if cap_sizes else []
        )
        self._rebuild_cap_lut(self._draft_cap_g_lut, self._draft_graph_cap_sizes)
        self._draft_graph_k = _build_graph_k_table(draft_k_dense, cap_sizes, n, k)
        self._draft_k_dense = draft_k_dense
        self._last_draft_k = k
        if self._batch_policy_active:
            logger.info(
                "Inkling2 batch spec policy: dyn_sd=%s graph_k=%s batch_cap=%s tau_batch=%s",
                self._dyn_sd_k[1:] if self._dyn_sd_k is not None else "off",
                self._draft_graph_k,
                raw_cap or "off",
                raw_tau or "off",
            )

    def _draft_k_for_graph(self, num_reqs: int) -> int:
        """Number of Markov steps to capture/run for a draft pass of `num_reqs` rows."""
        if self._dyn_sd_k is None:
            return self.num_speculative_steps
        if num_reqs in self._draft_graph_k:
            return self._draft_graph_k[num_reqs]
        b = max(1, min(num_reqs, self.max_num_reqs))
        return self._draft_k_dense[b]

    def _effective_w_cap(self, num_reqs: int, num_spec_arg: int | None) -> int:
        if not self._batch_policy_active and num_spec_arg is None:
            return self.num_speculative_steps
        b = max(1, min(num_reqs, self.max_num_reqs))
        cap = self._vw_batch_cap[b]
        if self._dyn_sd_k is not None:
            cap = min(cap, self._dyn_sd_k[b])
        if num_spec_arg is not None:
            cap = min(cap, max(0, int(num_spec_arg)))
        return max(0, min(self.num_speculative_steps, cap))

    def _effective_tau_override(self, num_reqs: int) -> float:
        if not self._vw_tau_batch_on:
            return -1.0
        b = max(1, min(num_reqs, self.max_num_reqs))
        return float(self._vw_tau_batch[b])

    def _is_graph_vw_bucket_uniform(self, cap_g: int) -> bool:
        """True iff every runtime batch size b in (prev_G, cap_g] shares identical
        (w_cap(b), tau(b)) so in-graph _vw_conf_kernel can never destructively mask
        a smaller padded batch b with a stricter tau(cap_g)."""
        if not self._graph_vw_uniform_guard or cap_g <= 1:
            return True
        sizes = getattr(self, "_draft_graph_cap_sizes", ())
        if not sizes or cap_g not in sizes:
            return True
        idx = sizes.index(cap_g)
        lo = (sizes[idx - 1] + 1) if idx > 0 else 1
        hi = min(cap_g, self.max_num_reqs)
        ref = (
            self._effective_w_cap(hi, None),
            -2.0 if self._hetero_budget_on else self._effective_tau_override(hi),
        )
        for b in range(lo, hi):
            cur = (
                self._effective_w_cap(b, None),
                -2.0 if self._hetero_budget_on else self._effective_tau_override(b),
            )
            if cur != ref:
                return False
        return True

    @staticmethod
    def _rebuild_cap_lut(lut: list[int | None], keys) -> None:
        """Populate O(1) dense lookup lut[b] = min(g in keys if g >= b) for 1 <= b < len(lut)."""
        n = len(lut) - 1
        for i in range(n + 1):
            lut[i] = None
        if not keys:
            return
        sk = sorted(int(g) for g in keys if int(g) >= 1)
        prev = 0
        for g in sk:
            hi = min(g, n)
            for b in range(prev + 1, hi + 1):
                lut[b] = g
            prev = hi
            if prev >= n:
                break

    def _next_draft_cap_g(self, num_reqs: int) -> int | None:
        if 1 <= num_reqs < len(self._draft_cap_g_lut):
            return self._draft_cap_g_lut[num_reqs]
        return None

    # ---- INKLING_VERIFY_W: adaptive verify width (greedy requests) ----------
    # Verify only W of the 7 drafts. The draft tokens past W are overwritten
    # with -1 in req_states.draft_tokens, so the target step sees input id -1 on
    # those rows: model.embed_input_ids flags them, the first in-graph op of the
    # model turns the flag into is_padding (MoE gives them no expert rows), and
    # the stock rejection sampler rejects them (draft_sampled < 0) so the bonus
    # token is the target argmax of row W. W comes from replicated state only
    # (num_sampled / num_rejected, an on-device EMA and, for conf, the drafter's top-1
    # probability computed from the gathered argmax records), identical on every rank.
    #
    # Modes: ema (acceptance history), conf (drafter confidence), conf+ema (both),
    # force:K. INKLING_CONF_LOG=<path> additionally logs (confidence, accepted) per
    # request-step from rank 0 (a "shadow" mode when INKLING_VERIFY_W is off: nothing is masked).
    def _init_verify_w(self, device: torch.device) -> None:
        self._vw_mode, self._vw_k = _parse_verify_w()
        self._cl_path = os.environ.get("INKLING_CONF_LOG", "").strip()
        self._cl_on = False
        self._conf_sink_on = False
        self._conf_armed = False
        self._conf_warned = False
        self._dops = None
        self._l2pf = None
        self._markov_fused = False
        self._es_on = False
        self._vw_is_dink = False
        self._es_bufs = None
        self._es_dummy_temp = None
        self._es_dummy_idx = None
        self._markov_persistent = os.environ.get(
            "INKLING_MARKOV_PERSISTENT", "0"
        ).strip().lower() in ("1", "on", "true", "yes")
        self._mk_p_captured = False
        self._mk_p_state: torch.Tensor | None = None
        self._mk_p_latent: torch.Tensor | None = None
        self._mk_p_rec: torch.Tensor | None = None
        self._mk_p_ctrl: torch.Tensor | None = None
        try:
            from vllm.models.inkling2.ops import dspark as _dops_mod

            self._markov_fused = bool(getattr(_dops_mod, "_MARKOV_FUSED", False))
        except Exception:
            _dops_mod = None
        try:
            from vllm.models.inkling2.ops import l2pf as _l2pf_mod

            self._l2pf = _l2pf_mod
        except Exception:
            self._l2pf = None
        self._dspark_k = min(self.num_speculative_steps, 7)
        if self._vw_mode != "off" and self.num_speculative_steps != 7:
            if not (self._sam_hybrid_on and self.num_speculative_steps in (11, 15)):
                logger.warning("INKLING_VERIFY_W needs 7 draft tokens (or 11/15 with INKLING_SAM_HYBRID=1); disabled")
                self._vw_mode = "off"
        if self._vw_mode == "off" and self._cl_path and self.num_speculative_steps == 7:
            self._vw_mode = "shadow"  # logging only, no masking
        if self._vw_mode == "off":
            if self._markov_fused:
                logger.info("INKLING_MARKOV_FUSED=1 (verify_w=off, early_stop=0)")
            return
        n, k = self.max_num_reqs, self._dspark_k
        self._vw_is_dink = self._vw_mode == "dink"
        self._vw_is_conf = self._vw_mode in ("conf", "conf+ema", "dink")
        self._vw_apply = self._vw_mode != "shadow"
        init_w = self._vw_k if self._vw_mode == "force" else 7
        self._vw_init_w = init_w
        self._vw_w = torch.full((n,), init_w, dtype=torch.int32, device=device)
        self._vw_c = torch.full((n, k), _EMA_PRIOR, dtype=torch.float32, device=device)
        cal = _load_cal(k)
        self._vw_cal_cfg = cal
        dink_cfg = cal["dink"]
        self._vw_dink_eta_b = float(dink_cfg["eta_b"])
        self._vw_dink_decay_b = float(dink_cfg["decay_b"])
        self._vw_dink_b_clip = float(dink_cfg["b_clip"])
        self._vw_dink_eta_lam = float(dink_cfg["eta_lam"])
        self._vw_dink_init_tok = float(dink_cfg["init_tok"])
        self._vw_dink_init_ms = float(dink_cfg["init_ms"])
        self._vw_db = torch.zeros(n, dtype=torch.float32, device=device)
        self._vw_etok = torch.full((n,), self._vw_dink_init_tok, dtype=torch.float32, device=device)
        self._vw_ems = torch.full((n,), self._vw_dink_init_ms, dtype=torch.float32, device=device)
        self._vw_cpred = torch.full((n, k), _EMA_PRIOR, dtype=torch.float32, device=device)
        beta = cal["beta"]
        if beta is None:
            beta = _CONF_BETA_DEFAULT
        if self._vw_mode != "conf+ema":
            beta = 0.0
        calt = torch.zeros(24, dtype=torch.float32)
        calt[:k] = torch.tensor(cal["a"][:k])
        calt[8 : 8 + k] = torch.tensor(cal["b"][:k])
        calt[16] = float(beta)
        calt[17] = 0.0 if self._vw_is_dink else float(cal["tau"])
        calt[18] = float(dink_cfg["static_lam"])
        calt[19] = float(dink_cfg["gamma"])
        calt[20] = float(cal["mk_step_ms"])
        self._vw_cal = calt.to(device)
        self._vw_stats_on = os.environ.get("INKLING_VERIFY_W_STATS", "0") == "1"
        self._vw_stats = torch.zeros(10, dtype=torch.int32, device=device)
        self._vw_calls = 0
        # Confidence plumbing: the Markov loop writes each step's top-1 probability into
        # this persistent buffer (dspark_ops.dspark_argmax, in the draft graph).
        if self._vw_is_conf or self._cl_path:
            if _dops_mod is not None:
                self._dops = _dops_mod
            else:
                logger.warning("INKLING confidence unavailable; conf/log disabled")
        if self._dops is not None and not getattr(self._dops, "_FUSED", False):
            logger.warning("INKLING_ARGMAX_FUSED=0: no confidence record; conf/log disabled")
            self._dops = None
        if self._dops is None:
            if self._vw_is_conf:
                self._vw_mode, self._vw_is_conf, self._vw_is_dink = "off", False, False
                return
            self._cl_path = ""
        self._conf_sink_on = self._dops is not None
        if self._conf_sink_on:
            self._conf_buf = torch.zeros(n, 8, dtype=torch.float32, device=device)
            self._dops.set_conf_sink(self._conf_buf)
            self._conf_slot = torch.zeros(n, _CF, dtype=torch.float32, device=device)
        self._init_conf_log(device)
        self._es_on = (
            os.environ.get("INKLING_MARKOV_EARLY_STOP", "0").strip().lower() in ("1", "on", "true", "yes")
            and self._markov_fused
            and self._vw_is_conf
            and self._conf_sink_on
            and not self._cl_on
        )
        # Wave-V9: Opt-in Batch-All Markov Early-Stop for B=2..8 (default OFF)
        self._es_batch_on = (
            os.environ.get("INKLING_MARKOV_EARLY_STOP_BATCH", "0").strip().lower() in ("1", "on", "true", "yes")
            and self._markov_fused
            and self._vw_is_conf
            and self._conf_sink_on
            and not self._cl_on
        )
        self._es_batch_tau = float(os.environ.get("INKLING_MARKOV_EARLY_STOP_BATCH_TAU", "0.12") or "0.12")
        self._es_batch_max_b = min(8, max(2, int(os.environ.get("INKLING_MARKOV_EARLY_STOP_BATCH_MAX_B", "8") or "8")))
        self._es_batch_bufs = None
        self._sam_best_of_2 = os.environ.get("INKLING_SAM_BEST_OF_2", "0").strip().lower() in ("1", "on", "true", "yes")
        self._markov_fuse_cross_step = (
            os.environ.get("INKLING_MARKOV_FUSE_CROSS_STEP", "0").strip().lower() in ("1", "on", "true", "yes")
            and self._markov_fused
        )
        # Wave-V12 opt-in wiring for race-free V10 fused tail/epilogue/tree kernels (default OFF)
        self._dspark_fuse_tail_vw = (
            os.environ.get("INKLING_DSPARK_FUSE_TAIL_VW", "0").strip().lower() in ("1", "on", "true", "yes")
            and self._markov_fuse_cross_step
        )
        self._dspark_fuse_epilogue_prep = (
            os.environ.get("INKLING_DSPARK_FUSE_EPILOGUE_PREP", "0").strip().lower() in ("1", "on", "true", "yes")
        )
        self._tree_fuse_v10 = (
            os.environ.get("INKLING_TREE_FUSE_V10", "0").strip().lower() in ("1", "on", "true", "yes")
        )
        self._deferred_dink_prep = None
        step_tbl = cal["step_eff_ms"] if (self._vw_is_dink and self._es_on) else cal["step_ms"]
        self._vw_step = torch.tensor(step_tbl, dtype=torch.float32, device=device)
        if self._hetero_budget_on or self._ctx_adaptive_tau_on:
            _kv_bf12_on = os.environ.get("INKLING_KV_BF12", "0").strip().lower() not in (
                "0", "", "off", "false", "no"
            )
            _wf_rows = _build_gb10_waterfill_table(
                self.max_num_reqs,
                getattr(self, "_draft_k_dense", [k] * (n + 1)),
                kv_bf12=_kv_bf12_on,
                dspark_k=k,
            )
            self._wf_step_tbl = torch.tensor(_wf_rows, dtype=torch.float32, device=device)
            self._wf_dummy_seq = torch.zeros((max(n, _WF_MAX_B),), dtype=torch.int32, device=device)
            logger.info(
                "Wave-V11 batch spec active: hetero_budget=%d (m_cap=%d, tau_min=%.3f), "
                "ctx_adaptive_tau=%d (floor=%.3f, kv_bf12=%d, tbl=%.1fKiB)",
                int(self._hetero_budget_on),
                self._hetero_budget_m_cap,
                self._hetero_budget_tau_min,
                int(self._ctx_adaptive_tau_on),
                self._ctx_adaptive_tau_floor,
                int(_kv_bf12_on),
                float(self._wf_step_tbl.numel() * 4) / 1024.0,
            )
        # JIT the kernels now, not on the first request.
        idx = torch.zeros(1, dtype=torch.int64, device=device)
        ns = torch.full((1,), 2, dtype=torch.int32, device=device)
        nr = torch.full((1,), 6, dtype=torch.int32, device=device)
        temp = torch.zeros(n, dtype=torch.float32, device=device)
        self._vw_launch_update(idx, ns, nr, temp)
        if self._vw_is_dink:
            self._vw_launch_update_dink(idx, ns, nr, temp, fuse_es=False)
        self._vw_launch_mask(self.draft_tokens, idx, temp, w_cap=k)
        if self._conf_sink_on:
            self._vw_launch_conf(idx, temp, self.draft_tokens, w_cap=k, tau_override=-1.0)
            if self._ctx_adaptive_tau_on:
                _c_idx = torch.full((2,), -1, dtype=torch.int64, device=device)
                self._vw_launch_conf(
                    _c_idx, temp, self.draft_tokens, w_cap=k, tau_override=0.24,
                    num_reqs=2, idx_stride=1, seq_lens=self._wf_dummy_seq,
                )
            if self._graph_vw_conf_on:
                # Pre-JIT in-graph IDX_STRIDE=num_speculative_steps specializations outside graph capture.
                _stride_idx = self.num_speculative_steps
                _sidx = torch.full((_stride_idx,), -1, dtype=torch.int64, device=device)
                self._vw_launch_conf(
                    _sidx, temp, self.draft_tokens, w_cap=k, tau_override=-1.0,
                    num_reqs=1, idx_stride=_stride_idx,
                )
                self._vw_launch_conf(
                    _sidx, temp, self.draft_tokens, w_cap=k, tau_override=-1.0,
                    req_drafts=self.draft_tokens, num_reqs=1, idx_stride=_stride_idx,
                )
                if self._ctx_adaptive_tau_on:
                    _sidx2 = torch.full((2 * _stride_idx,), -1, dtype=torch.int64, device=device)
                    for _rd in (None, self.draft_tokens):
                        self._vw_launch_conf(
                            _sidx2, temp, self.draft_tokens, w_cap=k, tau_override=0.24,
                            req_drafts=_rd, num_reqs=2, idx_stride=_stride_idx,
                            seq_lens=self._wf_dummy_seq,
                        )
            if self._hetero_budget_on:
                _stride_idx = self.num_speculative_steps
                for _bt in (2, 4, 8, 16, 32):
                    if _bt > max(2, min(n, _WF_MAX_B)) and _bt > 2:
                        continue
                    _wf_idx1 = torch.full((_bt,), -1, dtype=torch.int64, device=device)
                    for _rd in (None, self.draft_tokens):
                        self._vw_launch_hetero_waterfill(
                            _wf_idx1, temp, self.draft_tokens,
                            w_cap=k, req_drafts=_rd, num_reqs=_bt, idx_stride=1,
                            seq_lens=self._wf_dummy_seq,
                        )
                    if self._graph_vw_conf_on:
                        _wf_idxs = torch.full((_bt * _stride_idx,), -1, dtype=torch.int64, device=device)
                        for _rd in (None, self.draft_tokens):
                            self._vw_launch_hetero_waterfill(
                                _wf_idxs, temp, self.draft_tokens,
                                w_cap=k, req_drafts=_rd, num_reqs=_bt, idx_stride=_stride_idx,
                                seq_lens=self._wf_dummy_seq,
                            )
            self._vw_launch_store(idx)
            if self._cl_on:
                self._vw_launch_log(idx, ns, nr, temp)
                self._cl_n = 0
        if self._es_on:
            if self._vw_is_dink:
                self._es_bufs = (
                    torch.zeros(1, dtype=torch.int32, device=device),    # skip
                    torch.ones(1, dtype=torch.float32, device=device),   # surv
                    torch.ones(1, dtype=torch.float32, device=device),   # tok
                    torch.zeros(1, dtype=torch.float32, device=device),  # best_u
                    torch.zeros(1, dtype=torch.int32, device=device),    # greedy (0 for capture!)
                    torch.zeros(1, dtype=torch.int32, device=device),    # slot
                    torch.zeros(1, dtype=torch.float32, device=device),  # db_dev
                    torch.full((1,), 0.12, dtype=torch.float32, device=device),  # lam_dev
                )
            else:
                self._es_bufs = (
                    torch.zeros(1, dtype=torch.int32, device=device),    # skip
                    torch.ones(1, dtype=torch.float32, device=device),   # surv
                    torch.ones(1, dtype=torch.float32, device=device),   # tok
                    torch.ones(1, dtype=torch.float32, device=device),   # best_v
                    torch.zeros(1, dtype=torch.int32, device=device),    # greedy (0 for capture!)
                    torch.zeros(1, dtype=torch.int32, device=device),    # slot
                )
            self._es_dummy_temp = torch.zeros(n, dtype=torch.float32, device=device)
            self._es_dummy_idx = torch.zeros(1, dtype=torch.int64, device=device)
            if self._vw_is_dink:
                if self._fuse_dink_es:
                    self._vw_launch_update_dink(idx, ns, nr, temp, fuse_es=True)
                self._dops.es_init_dink(
                    idx, temp, self._vw_step, self._vw_cal, self._vw_db, self._vw_etok, self._vw_ems,
                    self._es_bufs, dummy=False,
                )
                self._dops.es_init_dink(
                    idx, temp, self._vw_step, self._vw_cal, self._vw_db, self._vw_etok, self._vw_ems,
                    self._es_bufs, dummy=True,
                )
            else:
                self._dops.es_init(idx, temp, self._vw_step, self._es_bufs, dummy=False)
                self._dops.es_init(idx, temp, self._vw_step, self._es_bufs, dummy=True)
            # Leave greedy=0, skip=0 so CUDA graph warmup/capture never trips early-stop.
            self._es_bufs[0].zero_()
            self._es_bufs[4].zero_()
        if self._es_batch_on:
            self._es_batch_bufs = (
                torch.zeros(1, dtype=torch.int32, device=device),    # skip_all
                torch.ones(8, dtype=torch.float32, device=device),   # surv_b[8]
                torch.zeros(1, dtype=torch.int32, device=device),    # greedy_all
                torch.zeros(8, dtype=torch.float32, device=device),  # db_b[8]
            )
            _eb_idx = torch.zeros(2, dtype=torch.int64, device=device)
            self._dops.es_init_batch(_eb_idx, temp, self._vw_db, self._es_batch_bufs, 2, dummy=True)
            self._dops.es_init_batch(_eb_idx, temp, self._vw_db, self._es_batch_bufs, 2, dummy=False)
            self._es_batch_bufs[0].zero_()
            self._es_batch_bufs[2].zero_()
            if self._markov_persistent and self._vw_is_dink:
                mk_rank = int(getattr(self.draft_model_config.hf_config, "markov_rank", 256))
                self._mk_p_state = torch.zeros((2, mk_rank), dtype=self.dtype, device=device)
                self._mk_p_latent = torch.zeros((mk_rank,), dtype=self.dtype, device=device)
                self._mk_p_rec = torch.zeros((64, 4), dtype=torch.float32, device=device)
                self._mk_p_ctrl = torch.zeros((4,), dtype=torch.int64, device=device)
        self._vw_w.fill_(init_w)
        self._vw_c.fill_(_EMA_PRIOR)
        self._vw_db.zero_()
        self._vw_etok.fill_(self._vw_dink_init_tok)
        self._vw_ems.fill_(self._vw_dink_init_ms)
        self._vw_cpred.fill_(_EMA_PRIOR)
        self._vw_stats.zero_()
        self.draft_tokens.zero_()
        if self._conf_sink_on:
            self._conf_buf.zero_()
            self._conf_slot.zero_()
        logger.info(
            "INKLING_VERIFY_W=%s (initial W=%d) conf_log=%s markov_fused=%d early_stop=%d",
            self._vw_mode,
            init_w,
            self._cl_path or "off",
            int(self._markov_fused),
            int(self._es_on),
        )

    def _vw_launch_update(self, idx, num_sampled, num_rejected, temp) -> None:
        _vw_update_kernel[(idx.shape[0],)](
            idx,
            num_sampled,
            num_rejected,
            temp,
            self._vw_w,
            self._vw_c,
            self._vw_step,
            self._vw_stats,
            ALPHA=_EMA_ALPHA,
            DECAY=_EMA_DECAY,
            PRIOR=_EMA_PRIOR,
            K=self._dspark_k,
            STATS=self._vw_stats_on and not self._vw_is_dink,
            WRITE_W=getattr(self, "_vw_mode", "ema") == "ema",
            FIXED_USED=getattr(self, "_vw_mode", "ema") == "shadow",
        )

    def _vw_launch_update_dink(
        self, idx, num_sampled, num_rejected, temp, *, fuse_es: bool = False
    ) -> None:
        if (
            getattr(self, "_dspark_fuse_epilogue_prep", False)
            and self._dops is not None
            and hasattr(self._dops, "vw_update_dink_v10")
            and self._dops.vw_update_dink_v10(
                idx, num_sampled, num_rejected, temp,
                self._vw_w, self._vw_cpred, self._vw_db, self._vw_etok, self._vw_ems,
                self._vw_step, self._vw_stats, self._vw_cal, self._es_bufs,
                eta_b=self._vw_dink_eta_b, decay_b=self._vw_dink_decay_b,
                b_clip=self._vw_dink_b_clip, eta_lam=self._vw_dink_eta_lam,
                K=self._dspark_k, do_stats=self._vw_stats_on, fuse_es=fuse_es,
            )
        ):
            return
        do_fuse = bool(fuse_es and self._es_bufs is not None and len(self._es_bufs) == 8)
        if do_fuse:
            skip_d, surv_d, tok_d, bestu_d, greedy_d, slot_d, db_d, lam_d = self._es_bufs
        else:
            skip_d = slot_d = greedy_d = self._vw_w
            surv_d = tok_d = bestu_d = db_d = lam_d = self._vw_step
        _vw_update_dink_kernel[(idx.shape[0],)](
            idx,
            num_sampled,
            num_rejected,
            temp,
            self._vw_w,
            self._vw_cpred,
            self._vw_db,
            self._vw_etok,
            self._vw_ems,
            self._vw_step,
            self._vw_stats,
            self._vw_cal,
            skip_d,
            surv_d,
            tok_d,
            bestu_d,
            greedy_d,
            slot_d,
            db_d,
            lam_d,
            ETA_B=self._vw_dink_eta_b,
            DECAY_B=self._vw_dink_decay_b,
            B_CLIP=self._vw_dink_b_clip,
            ETA_LAM=self._vw_dink_eta_lam,
            K=self._dspark_k,
            STATS=self._vw_stats_on,
            FUSE_ES=do_fuse,
        )

    def _vw_launch_mask(
        self,
        draft_tokens,
        idx,
        temp,
        w_cap: int = 7,
        req_drafts: torch.Tensor | None = None,
        *,
        clear_req_tail: bool = False,
    ) -> None:
        write_req = req_drafts is not None
        _vw_mask_kernel[(idx.shape[0],)](
            draft_tokens,
            draft_tokens.stride(0),
            idx,
            temp,
            self._vw_w,
            int(w_cap),
            req_drafts if write_req else draft_tokens,
            req_drafts.stride(0) if write_req else draft_tokens.stride(0),
            K=self._dspark_k,
            WRITE_REQ_DRAFTS=write_req,
            CLEAR_REQ_TAIL=bool(write_req and clear_req_tail and self.num_speculative_steps > self._dspark_k),
            REQ_K=int(self.num_speculative_steps),
        )

    def _vw_launch_conf(
        self,
        idx,
        temp,
        draft_tokens,
        w_cap: int = 7,
        tau_override: float = -1.0,
        req_drafts: torch.Tensor | None = None,
        *,
        num_reqs: int | None = None,
        idx_stride: int = 1,
        clear_req_tail: bool = False,
        seq_lens: torch.Tensor | None = None,
    ) -> None:
        write_req = req_drafts is not None
        grid_n = int(idx.shape[0]) if num_reqs is None else int(num_reqs)
        ctx_adapt = bool(
            self._ctx_adaptive_tau_on
            and self._wf_step_tbl is not None
            and grid_n >= 2
            and float(tau_override) >= 0.0
            and seq_lens is not None
        )
        seq_arg = seq_lens if ctx_adapt else self._vw_w
        wf_arg = self._wf_step_tbl if ctx_adapt else self._vw_step
        _vw_conf_kernel[(grid_n,)](
            idx,
            self._conf_buf,
            self._conf_buf.stride(0),
            temp,
            self._vw_w,
            self._vw_c,
            self._vw_cal,
            self._vw_step,
            draft_tokens,
            draft_tokens.stride(0),
            self._conf_slot,
            self._vw_db,
            self._vw_etok,
            self._vw_ems,
            self._vw_cpred,
            int(w_cap),
            float(tau_override),
            req_drafts if write_req else draft_tokens,
            req_drafts.stride(0) if write_req else draft_tokens.stride(0),
            seq_arg,
            wf_arg,
            grid_n,
            float(self._ctx_adaptive_tau_floor),
            PRIOR_LOGIT=_PRIOR_LOGIT,
            K=self._dspark_k,
            USE_HIST=self._vw_mode == "conf+ema",
            USE_DINK=self._vw_is_dink,
            LOG=self._cl_on,
            WRITE_REQ_DRAFTS=write_req,
            IDX_STRIDE=int(idx_stride),
            CLEAR_REQ_TAIL=bool(write_req and clear_req_tail and self.num_speculative_steps > self._dspark_k),
            REQ_K=int(self.num_speculative_steps),
            CTX_ADAPTIVE=ctx_adapt,
            WF_COLS=_WF_COLS,
            WF_MAX_B=_WF_MAX_B,
        )

    def _vw_launch_hetero_waterfill(
        self,
        idx,
        temp,
        draft_tokens,
        w_cap: int = 7,
        req_drafts: torch.Tensor | None = None,
        *,
        num_reqs: int | None = None,
        idx_stride: int = 1,
        clear_req_tail: bool = False,
        seq_lens: torch.Tensor | None = None,
    ) -> None:
        write_req = req_drafts is not None
        grid_n = int(idx.shape[0]) if num_reqs is None else int(num_reqs)
        b_tile = 2
        while b_tile < grid_n and b_tile < _WF_MAX_B:
            b_tile <<= 1
        k_tile = 8
        n_lanes = b_tile * k_tile
        seq_arg = seq_lens if seq_lens is not None else self._wf_dummy_seq
        _vw_hetero_waterfill_kernel[(1,)](
            idx,
            self._conf_buf,
            self._conf_buf.stride(0),
            temp,
            self._vw_w,
            self._vw_cal,
            draft_tokens,
            draft_tokens.stride(0),
            self._vw_db,
            self._vw_cpred,
            seq_arg,
            self._wf_step_tbl,
            req_drafts if write_req else draft_tokens,
            req_drafts.stride(0) if write_req else draft_tokens.stride(0),
            int(grid_n),
            int(w_cap),
            int(self._hetero_budget_m_cap),
            float(self._hetero_budget_tau_min),
            K=self._dspark_k,
            B_TILE=b_tile,
            K_TILE=k_tile,
            N_LANES=n_lanes,
            WRITE_REQ_DRAFTS=write_req,
            IDX_STRIDE=int(idx_stride),
            CLEAR_REQ_TAIL=bool(write_req and clear_req_tail and self.num_speculative_steps > self._dspark_k),
            REQ_K=int(self.num_speculative_steps),
            WF_COLS=_WF_COLS,
            WF_MAX_B=_WF_MAX_B,
            num_warps=4,
        )

    def _vw_launch_store(self, idx) -> None:
        _vw_store_conf_kernel[(idx.shape[0],)](
            idx,
            self._conf_buf,
            self._conf_buf.stride(0),
            self._vw_w,
            self._vw_c,
            self._conf_slot,
            K=self._dspark_k,
            USE_W=self._vw_apply,
        )

    # ---- INKLING_CONF_LOG: (confidence, accepted) pairs from rank 0 ------------
    def _init_conf_log(self, device: torch.device) -> None:
        if not self._cl_path:
            return
        try:
            from vllm.distributed import get_tensor_model_parallel_rank

            rank = get_tensor_model_parallel_rank()
        except Exception:
            rank = 0
        allranks = os.environ.get("INKLING_CONF_LOG_ALLRANKS", "0") == "1"
        if rank != 0 and not allranks:
            return
        self._cl_every = max(1, int(os.environ.get("INKLING_CONF_LOG_EVERY", "128")))
        self._cl_ring = torch.zeros(
            self._cl_every, self.max_num_reqs, _LOG_NF, dtype=torch.float32, device=device
        )
        self._cl_rid = torch.zeros(self.max_num_reqs, dtype=torch.int32, device=device)
        self._cl_ctr = 0
        self._cl_n = 0
        self._cl_rows: list[int] = []
        self._cl_times: list[float] = []
        self._cl_pending: list = []
        self._cl_last = time.time()
        self._cl_file = None
        self._cl_rank = rank
        self._cl_name = self._cl_path + (f".r{rank}" if allranks else "")
        self._cl_on = True

    def _vw_launch_log(self, idx, num_sampled, num_rejected, temp) -> None:
        _vw_log_kernel[(idx.shape[0],)](
            idx,
            num_sampled,
            num_rejected,
            temp,
            self._conf_slot,
            self._cl_rid,
            self._cl_ring,
            self._cl_n * self.max_num_reqs * _LOG_NF,
            NF=_LOG_NF,
        )

    def _cl_record(self, input_batch, num_sampled, num_rejected, temp) -> None:
        num_reqs = input_batch.num_reqs
        self._vw_launch_log(
            input_batch.idx_mapping[:num_reqs],
            num_sampled[:num_reqs],
            num_rejected[:num_reqs],
            temp,
        )
        self._cl_rows.append(num_reqs)
        self._cl_times.append(time.time())
        self._cl_n += 1
        if self._cl_n >= self._cl_every or time.time() - self._cl_last > 3.0:
            self._cl_flush(sync=False)
        else:
            self._cl_drain(wait=False)

    def _cl_flush(self, sync: bool) -> None:
        """Queue a non-blocking device->pinned copy of the filled ring part; write finished
        copies to the JSONL file (sync=True waits for all of them)."""
        n = self._cl_n
        if n:
            buf = torch.empty(
                (n, self.max_num_reqs, _LOG_NF), dtype=torch.float32, pin_memory=True
            )
            buf.copy_(self._cl_ring[:n], non_blocking=True)
            ev = torch.cuda.Event()
            ev.record()
            self._cl_pending.append((buf, ev, self._cl_rows, self._cl_times))
            self._cl_n, self._cl_rows, self._cl_times = 0, [], []
        self._cl_last = time.time()
        self._cl_drain(wait=sync)

    def _cl_drain(self, wait: bool) -> None:
        while self._cl_pending:
            buf, ev, rows, times = self._cl_pending[0]
            if not wait and not ev.query():
                break
            ev.synchronize()
            self._cl_pending.pop(0)
            if self._cl_file is None:
                self._cl_file = open(self._cl_name, "a", buffering=1)
                meta = {
                    "meta": 1,
                    "K": self.num_speculative_steps,
                    "mode": self._vw_mode,
                    "fields": "rid,slot,ns,nd,w,p[7],hist[7]",
                    "t": time.time(),
                    "rank": self._cl_rank,
                }
                self._cl_file.write(json.dumps(meta) + "\n")
            a = buf.numpy()
            out = []
            for i, nrows in enumerate(rows):
                for r in range(nrows):
                    v = a[i, r]
                    if v[0] <= 0:
                        continue
                    out.append(
                        json.dumps(
                            {
                                "t": round(times[i], 4),
                                "rid": int(v[0]) - 1,
                                "slot": int(v[1]),
                                "ns": int(v[2]),
                                "nd": int(v[3]),
                                "w": int(v[4]),
                                "p": [float(f"{x:.6g}") for x in v[5:12]],
                                "h": [float(f"{x:.5g}") for x in v[12:19]],
                            },
                            separators=(",", ":"),
                        )
                    )
            if out:
                self._cl_file.write("\n".join(out) + "\n")

    def on_requests_finished(self, finished_req_ids) -> None:
        if self._cl_on:
            self._cl_flush(sync=True)
        if self._tree_state is not None:
            self._tree_state.reset_linear()
        super().on_requests_finished(finished_req_ids)

    def _init_sam_and_tree(self, device: torch.device) -> None:
        """Pre-allocate persistent GPU/pinned buffers for SAM Hybrid & Tree Speculation
        strictly OUTSIDE CUDA graph capture."""
        if self._tree_spec_on:
            try:
                try:
                    from .ops import tree_spec_runtime as _tsr
                except ImportError:
                    from . import tree_spec_runtime as _tsr
                from .ops import dspark as _dops_mod

                self._tree_runtime_mod = _tsr
                self._tree_state = _tsr.TreeSpecState(max_m=self.num_speculative_steps + 1, device=device)
                self._tree_state.ws_max = self._tree_ws_max
                self._tree_state.bind_global()
                _tsr.get_tree_target_lib()
                rank = int(getattr(self.draft_model_config.hf_config, "markov_rank", 256))
                self._tree_prim_toks = torch.full((8,), -1, dtype=torch.int64, device=device)
                self._tree_sec_toks = torch.full((4,), -1, dtype=torch.int64, device=device)
                self._tree_sec_p = torch.zeros((4,), dtype=torch.float32, device=device)
                self._tree_p2_0 = torch.zeros((1,), dtype=torch.float32, device=device)
                self._tree_beam_state = torch.zeros((2, rank), dtype=self.dtype, device=device)
                self._tree_beam_prev = torch.zeros((2,), dtype=torch.int64, device=device)
                self._tree_step_bt1 = torch.zeros((2,), dtype=torch.int64, device=device)
                self._tree_step_bp1 = torch.zeros((2,), dtype=torch.float32, device=device)
                if device.type == "cuda":
                    _dops_mod.prealloc_fold_sem(device, min_rows=4)
                    # Pre-JIT tree kernels strictly outside CUDA graph capture (using safe dummy slot=-1)
                    if self._vw_is_conf and self._conf_sink_on:
                        _d_idx = torch.full((1,), -1, dtype=torch.int64, device=device)
                        _d_tmp = torch.zeros((self.max_num_reqs,), dtype=torch.float32, device=device)
                        _d_i32 = torch.zeros((1,), dtype=torch.int32, device=device)
                        _d_i64 = torch.zeros((1,), dtype=torch.int64, device=device)
                        _d_f32 = torch.zeros((1,), dtype=torch.float32, device=device)
                        _has_es = bool(self._es_on and self._vw_is_dink and self._es_bufs is not None)
                        _es_skip = self._es_bufs[0] if _has_es else _d_i32
                        _es_surv = self._es_bufs[1] if _has_es else _d_f32
                        _es_tok = self._es_bufs[2] if _has_es else _d_f32
                        _es_bestu = self._es_bufs[3] if _has_es else _d_f32
                        _es_greedy = self._es_bufs[4] if _has_es else _d_i32
                        _es_db = self._es_bufs[6] if _has_es else _d_f32
                        _es_lam = self._es_bufs[7] if _has_es else _d_f32
                        _pack_beam2_init_kernel[(1,)](
                            self._tree_beam_state[:1], _d_i64, _d_f32, _d_i64, _d_f32,
                            self._tree_beam_state, self._tree_beam_prev,
                            self._tree_prim_toks, self._conf_buf,
                            self._tree_sec_toks, self._tree_sec_p, self._tree_p2_0,
                            _es_skip, _es_surv, _es_tok, _es_bestu, _es_greedy, _es_db, _es_lam,
                            self._vw_cal, self._vw_step,
                            RANK=rank, HAS_ES=_has_es, CAN_SET_SKIP=(self._tree_ws_max == 1),
                            num_warps=4,
                        )
                        for _st in range(1, self._tree_ws_max):
                            _pack_beam2_step_kernel[(1,)](
                                self._tree_step_bt1, self._tree_step_bp1,
                                self._tree_prim_toks, self._conf_buf,
                                self._tree_sec_toks, self._tree_sec_p,
                                _es_skip, _es_surv, _es_tok, _es_bestu, _es_greedy, _es_db, _es_lam,
                                self._vw_cal, self._vw_step,
                                STEP=_st, HAS_ES=_has_es, CAN_SET_SKIP=(_st == self._tree_ws_max - 1),
                                num_warps=1,
                            )
                        for _wr in (False, True):
                            _tree_dink_select_pack_kernel[(1,)](
                                _d_idx, _d_tmp,
                                self._tree_prim_toks, self._conf_buf,
                                self._tree_sec_toks, self._tree_sec_p,
                                self._vw_cal, self._vw_step,
                                self._vw_db, self._vw_etok, self._vw_ems,
                                self._vw_w, self._vw_cpred,
                                self.draft_tokens, self.draft_tokens, self.draft_tokens.stride(0),
                                self._tree_state.tree_active, self._tree_state.n_live,
                                self._tree_state.tree_meta,
                                self._tree_state.tree_parent, self._tree_state.tree_depth,
                                self._tree_state.tree_sconv_anc, self._tree_state.tree_path,
                                WS_MAX=self._tree_ws_max,
                                SIB_DISC=self._tree_sib_disc,
                                SIB_PENALTY_MS=self._tree_sib_penalty_ms,
                                SIB_SCALE=self._tree_sib_scale,
                                P2_MIN=self._tree_p2_min,
                                WRITE_REQ_DRAFTS=_wr,
                                num_warps=1,
                            )
                        self._tree_prim_toks.fill_(-1)
                        self._conf_buf.zero_()
                        self._tree_state.reset_linear()
                logger.info(
                    "INKLING_TREE_SPEC=1 (max_m=%d, ws_max=%d, b1_only=%d, max_b=%d, sib_disc=%.2f, "
                    "sib_pen=%.2fms, sib_scale=%.2f, p2_min=%.2f): Fast ES + TC Beam-2 Tree + persistent TreeSpecState ready",
                    self.num_speculative_steps + 1, self._tree_ws_max, int(self._tree_spec_b1_only),
                    self._tree_spec_max_b, self._tree_sib_disc,
                    self._tree_sib_penalty_ms, self._tree_sib_scale, self._tree_p2_min,
                )
            except Exception as e:  # noqa: BLE001
                self._tree_spec_on = False
                self._tree_state = None
                self._tree_runtime_mod = None
                logger.warning("INKLING_TREE_SPEC=1 init failed (%r); falling back to linear chain", e)
        if self._sam_hybrid_on:
            if self._sam_mode == "gpu":
                ms = self._sam_max_slots
                self._sam_ring = torch.zeros((ms, self._sam_ring_sz), dtype=torch.int32, device=device)
                self._sam_ht_last = torch.full((ms, self._sam_ht_sz), -1, dtype=torch.int32, device=device)
                self._sam_ht_prev = torch.full((ms, self._sam_ht_sz), -1, dtype=torch.int32, device=device)
                self._sam_cur_len = torch.zeros((ms,), dtype=torch.int32, device=device)
                self._sam_ring_lo = torch.zeros((ms,), dtype=torch.int32, device=device)
                if self._sam_conf_boost_on:
                    self._sam_ht2_last = torch.full((ms, self._sam_ht2_sz), -1, dtype=torch.int32, device=device)
                if self._sam_long_prompt_on:
                    self._sam_slot_gen = torch.ones((ms,), dtype=torch.int32, device=device)
                    self._sam_lpc_key = torch.zeros((self._sam_lpc_cap,), dtype=torch.int64, device=device)
                    self._sam_lpc_cont = torch.full((self._sam_lpc_cap, 11), -1, dtype=torch.int32, device=device)
                if self._sam_shared_pool_on:
                    self._sam_sp_tag = torch.zeros((self._sam_sp_cap,), dtype=torch.int32, device=device)
                    self._sam_sp_cont = torch.full((self._sam_sp_cap, 7), -1, dtype=torch.int32, device=device)
                self._sam_gpu_ready = True
                _ht2_arg = self._sam_ht2_last if self._sam_ht2_last is not None else self._sam_cur_len
                _sgen_arg = self._sam_slot_gen if self._sam_slot_gen is not None else self._sam_cur_len
                _lpck_arg = self._sam_lpc_key if self._sam_lpc_key is not None else self._sam_cur_len
                _lpcc_arg = self._sam_lpc_cont if self._sam_lpc_cont is not None else self._sam_cur_len
                _spt_arg = self._sam_sp_tag if self._sam_sp_tag is not None else self._sam_cur_len
                _spc_arg = self._sam_sp_cont if self._sam_sp_cont is not None else self._sam_cur_len
                if device.type == "cuda" and self._vw_is_conf and self._conf_sink_on:
                    # Pre-JIT ALL SAM GPU kernels strictly outside CUDA graph & TD DTGRAPH capture!
                    _d_idx = torch.full((1,), -1, dtype=torch.int64, device=device)
                    _d_idx32 = torch.full((1,), -1, dtype=torch.int32, device=device)
                    _d_tmp = torch.zeros((self.max_num_reqs,), dtype=torch.float32, device=device)
                    _d_ids32 = torch.zeros((1,), dtype=torch.int32, device=device)
                    _d_ids64 = torch.zeros((1,), dtype=torch.int64, device=device)
                    _d_pos64 = torch.zeros((1,), dtype=torch.int64, device=device)
                    _d_qsl = torch.zeros((2,), dtype=torch.int32, device=device)
                    _d_ns = torch.zeros((1,), dtype=torch.int32, device=device)
                    _d_nr = torch.zeros((1,), dtype=torch.int32, device=device)
                    _d_ls = torch.zeros((self.max_num_reqs,), dtype=torch.int64, device=device)
                    # Pre-JIT _sam_gpu_ingest_kernel for both int32/int64 input_ids & idx_mapping
                    for _ids_t in (_d_ids32, _d_ids64):
                        for _idx_t in (_d_idx32, _d_idx):
                            _sam_gpu_ingest_kernel[(1,)](
                                _ids_t, _d_pos64, _d_qsl, _idx_t, _d_ns, _d_nr, _d_ls,
                                self._sam_ring, self._sam_ht_last, self._sam_ht_prev,
                                self._sam_cur_len, self._sam_ring_lo,
                                self._sam_cur_len, self._sam_cur_len,
                                _ht2_arg, _spt_arg, _spc_arg,
                                MAX_SLOTS=self._sam_max_slots,
                                RING_SZ=self._sam_ring_sz,
                                HT_SZ=self._sam_ht_sz,
                                MAX_INGEST=self._sam_max_ingest,
                                HAS_TREE=False,
                                TREE_MAX_M=8,
                                HAS_HT2=self._sam_conf_boost_on,
                                HT2_SZ=self._sam_ht2_sz,
                                HAS_SHARED=self._sam_shared_pool_on,
                                SP_CAP=self._sam_sp_cap,
                                num_warps=1,
                            )
                            if self._sam_long_prompt_on or self._sam_shared_pool_on:
                                _sam_lpc_ingest_chunk_kernel[(1, 1)](
                                    _ids_t, _d_pos64, _d_qsl, _idx_t, _d_nr,
                                    _sgen_arg, _lpck_arg, _lpcc_arg, _spt_arg, _spc_arg,
                                    MAX_SLOTS=self._sam_max_slots,
                                    LPC_CAP=self._sam_lpc_cap,
                                    LPC_STRIDE=self._sam_lpc_stride,
                                    SP_CAP=self._sam_sp_cap,
                                    HAS_LPC=self._sam_long_prompt_on,
                                    HAS_SHARED=self._sam_shared_pool_on,
                                    num_warps=4,
                                )
                    if self._fuse_dink_sam_ingest and self._vw_is_dink:
                        _es_ok = self._es_bufs is not None and len(self._es_bufs) == 8
                        _fes_opts = (False, True) if (_es_ok and self._fuse_dink_es and self._es_on) else (False,)
                        for _ids_t in (_d_ids32, _d_ids64):
                            for _idx_t in (_d_idx32, _d_idx):
                                for _fes in _fes_opts:
                                    if _fes:
                                        _s0, _s1, _s2, _s3, _s4, _s5, _s6, _s7 = self._es_bufs
                                    else:
                                        _s0 = _s4 = _s5 = self._vw_w
                                        _s1 = _s2 = _s3 = _s6 = _s7 = self._vw_step
                                    # Use valid slot 0 with ns=0, nr=0, qsl=[0,0] so neither dink nor SAM writes state.
                                    _z_idx = torch.zeros((1,), dtype=_idx_t.dtype, device=device)
                                    _vw_dink_sam_ingest_fused_kernel[(1,)](
                                        _z_idx, _d_ns, _d_nr, _d_tmp,
                                        self._vw_w, self._vw_cpred,
                                        self._vw_db, self._vw_etok, self._vw_ems,
                                        self._vw_step, self._vw_cal,
                                        _s0, _s1, _s2, _s3, _s4, _s5, _s6, _s7,
                                        _ids_t, _d_pos64, _d_qsl, _d_ls,
                                        self._sam_ring, self._sam_ht_last, self._sam_ht_prev,
                                        self._sam_cur_len, self._sam_ring_lo,
                                        self._sam_cur_len, self._sam_cur_len,
                                        _ht2_arg, _spt_arg, _spc_arg,
                                        ETA_B=self._vw_dink_eta_b,
                                        DECAY_B=self._vw_dink_decay_b,
                                        B_CLIP=self._vw_dink_b_clip,
                                        ETA_LAM=self._vw_dink_eta_lam,
                                        K=self._dspark_k,
                                        FUSE_ES=_fes,
                                        MAX_SLOTS=self._sam_max_slots,
                                        RING_SZ=self._sam_ring_sz,
                                        HT_SZ=self._sam_ht_sz,
                                        MAX_INGEST=self._sam_max_ingest,
                                        HAS_TREE=False,
                                        TREE_MAX_M=8,
                                        HAS_HT2=self._sam_conf_boost_on,
                                        HT2_SZ=self._sam_ht2_sz,
                                        HAS_SHARED=self._sam_shared_pool_on,
                                        SP_CAP=self._sam_sp_cap,
                                        num_warps=1,
                                    )
                        if _es_ok:
                            self._es_bufs[0].zero_()
                            self._es_bufs[4].zero_()
                    for _idx_t in (_d_idx32, _d_idx):
                        self._sam_launch_gpu_boost_extend(
                            _idx_t, _d_tmp, num_reqs=1, idx_stride=1, k_active=self._dspark_k
                        )
                    if self._sam_batch_on and self.num_speculative_steps != 1:
                        _d_sidx = torch.full((self.num_speculative_steps,), -1, dtype=torch.int64, device=device)
                        _d_sidx32 = torch.full((self.num_speculative_steps,), -1, dtype=torch.int32, device=device)
                        for _sidx_t in (_d_sidx32, _d_sidx):
                            self._sam_launch_gpu_boost_extend(
                                _sidx_t, _d_tmp, num_reqs=1,
                                idx_stride=self.num_speculative_steps, k_active=self._dspark_k,
                            )
                    if self.num_speculative_steps > self._dspark_k:
                        _ext_cap_max = int(self.num_speculative_steps - self._dspark_k)
                        _d_sidx_ext = torch.full((self.num_speculative_steps,), -1, dtype=torch.int64, device=device)
                        for _wr in (False, True):
                            for _is_ext, _idx_ext in ((1, _d_idx), (self.num_speculative_steps, _d_sidx_ext)):
                                _sam_mask_ext_tail_kernel[(1,)](
                                    _idx_ext, self._vw_w,
                                    self.draft_tokens, self.draft_tokens.stride(0),
                                    self.draft_tokens, self.draft_tokens.stride(0),
                                    _ext_cap_max,
                                    K_DRAFT=self._dspark_k, K_MAX=self.num_speculative_steps,
                                    WRITE_REQ_DRAFTS=_wr, IDX_STRIDE=_is_ext, num_warps=1,
                                )
                vram_bytes = ms * (self._sam_ring_sz + 2 * self._sam_ht_sz + 2) * 4
                if self._sam_ht2_last is not None:
                    vram_bytes += ms * self._sam_ht2_sz * 4
                if self._sam_lpc_key is not None:
                    vram_bytes += self._sam_lpc_cap * (8 + 11 * 4) + ms * 4
                if self._sam_sp_tag is not None:
                    vram_bytes += self._sam_sp_cap * (4 + 7 * 4)
                vram_mib = float(vram_bytes) / float(1 << 20)
                logger.info(
                    "INKLING_SAM_HYBRID=1 (mode=gpu, batch=%d, K_draft=%d, K_max=%d, min_match=%d, "
                    "slots=%d, ring=%d, ht=%d, max_ingest=%d, ovr_low_conf=%d[thr=%.2f,ml>=%d], "
                    "conf_boost=%d, long_prompt=%d[cap=%d,s=%d], shared_pool=%d[cap=%d], "
                    "allow_temp=%d, fuse_dink_ingest=%d, vram=%.2fMiB): "
                    "zero-sync L2 4-gram ring+table ready (100%% CUDA-graph & DTGRAPH safe)",
                    int(self._sam_batch_on), self._dspark_k, self.num_speculative_steps,
                    self._sam_min_match, ms, self._sam_ring_sz, self._sam_ht_sz,
                    self._sam_max_ingest, int(self._sam_override_low_conf),
                    self._sam_override_conf_thresh, self._sam_override_min_match,
                    int(self._sam_conf_boost_on), int(self._sam_long_prompt_on),
                    self._sam_lpc_cap, self._sam_lpc_stride,
                    int(self._sam_shared_pool_on), self._sam_sp_cap,
                    int(self._sam_allow_temp), int(self._fuse_dink_sam_ingest), vram_mib,
                )
            else:
                k_max = self.num_speculative_steps
                self._sam_pin_tok = torch.full((k_max,), -1, dtype=torch.int64, pin_memory=(device.type == "cuda"))
                self._sam_pin_boost = torch.zeros((self._dspark_k,), dtype=torch.float32, pin_memory=(device.type == "cuda"))
                self._sam_pin_ext_ok = torch.zeros((1,), dtype=torch.int32, pin_memory=(device.type == "cuda"))
                self._sam_dev_tok = torch.full((k_max,), -1, dtype=torch.int64, device=device)
                self._sam_dev_boost = torch.zeros((self._dspark_k,), dtype=torch.float32, device=device)
                self._sam_dev_ext_ok = torch.zeros((1,), dtype=torch.int32, device=device)
                self._sam_cpu_last_pos: dict[int, int] = {}
                logger.info(
                    "INKLING_SAM_HYBRID=1 (mode=cpu, lib=%s, allow_sync=%d, K_max=%d): "
                    "host SuffixHybridAdvisor ready (strictly outside graph capture)",
                    os.environ.get("INKLING_SAM_LIB", "/rdma/libsam_fast.so"),
                    int(self._sam_allow_sync), k_max,
                )

    def _sam_reset_slot(self, req_index: int) -> None:
        if not self._sam_hybrid_on:
            return
        if self._sam_gpu_ready and 0 <= req_index < self._sam_max_slots:
            self._sam_ht_last[req_index].fill_(-1)
            self._sam_ht_prev[req_index].fill_(-1)
            self._sam_cur_len[req_index] = 0
            self._sam_ring_lo[req_index] = 0
            if self._sam_ht2_last is not None:
                self._sam_ht2_last[req_index].fill_(-1)
            if self._sam_slot_gen is not None:
                self._sam_slot_gen[req_index] += 1
        elif self._sam_mode == "cpu":
            adv = self._sam_advisors.get(req_index)
            if adv is not None:
                adv.sam.reset()
            self._sam_cpu_last_pos.pop(req_index, None)

    def _sam_get_cpu_advisor(self, req_index: int):
        adv = self._sam_advisors.get(req_index)
        if adv is None:
            try:
                from .ops.suffix_hybrid import SuffixHybridAdvisor
            except ImportError:
                from .suffix_hybrid import SuffixHybridAdvisor

            adv = SuffixHybridAdvisor(
                min_branch_match=self._sam_min_match,
                min_ext_match=self._sam_min_match,
                min_boost_match=self._sam_min_match,
                max_ext_tokens=max(1, self.num_speculative_steps - self._dspark_k),
                ext_surv_thresh=self._sam_ext_surv,
            )
            self._sam_advisors[req_index] = adv
        return adv

    def sam_ingest_prompt_cpu(self, req_index: int, prompt_tokens: Any) -> None:
        """Zero-sync host prompt ingestion for INKLING_SAM_MODE=cpu (called from add_requests)."""
        if self._sam_hybrid_on and self._sam_mode == "cpu":
            adv = self._sam_get_cpu_advisor(req_index)
            adv.ingest_prompt(prompt_tokens)
            self._sam_cpu_last_pos[req_index] = len(prompt_tokens)

    def _sam_launch_gpu_ingest(self, input_batch, num_sampled, num_rejected, last_sampled) -> None:
        n_r = int(input_batch.num_reqs)
        if not (
            self._sam_hybrid_on
            and self._sam_gpu_ready
            and (n_r == 1 or (self._sam_batch_on and 1 <= n_r <= self._sam_max_slots))
        ):
            return
        ts = self._tree_state if (self._tree_spec_on and 1 <= n_r <= self._tree_spec_max_b and n_r == 1) else None
        has_tree = ts is not None
        ls_flat = last_sampled.view(-1) if (last_sampled.dim() > 1 and last_sampled.is_contiguous()) else last_sampled
        _ht2_arg = self._sam_ht2_last if self._sam_ht2_last is not None else self._sam_cur_len
        _spt_arg = self._sam_sp_tag if self._sam_sp_tag is not None else self._sam_cur_len
        _spc_arg = self._sam_sp_cont if self._sam_sp_cont is not None else self._sam_cur_len
        _sam_gpu_ingest_kernel[(n_r,)](
            input_batch.input_ids,
            input_batch.positions,
            input_batch.query_start_loc,
            input_batch.idx_mapping,
            num_sampled,
            num_rejected,
            ls_flat,
            self._sam_ring,
            self._sam_ht_last,
            self._sam_ht_prev,
            self._sam_cur_len,
            self._sam_ring_lo,
            ts.tree_active if has_tree else self._sam_cur_len,
            ts.accepted_path if has_tree else self._sam_cur_len,
            _ht2_arg,
            _spt_arg,
            _spc_arg,
            MAX_SLOTS=self._sam_max_slots,
            RING_SZ=self._sam_ring_sz,
            HT_SZ=self._sam_ht_sz,
            MAX_INGEST=self._sam_max_ingest,
            HAS_TREE=has_tree,
            TREE_MAX_M=ts.max_m if has_tree else 8,
            HAS_HT2=self._sam_conf_boost_on,
            HT2_SZ=self._sam_ht2_sz,
            HAS_SHARED=self._sam_shared_pool_on,
            SP_CAP=self._sam_sp_cap,
            num_warps=1,
        )

    def _sam_launch_lpc_chunk_ingest(self, input_batch, num_rejected) -> None:
        """Parallel multi-CTA prompt indexer for prefill chunks (num_tokens > 64, strictly outside graph capture)."""
        if not (self._sam_long_prompt_on or self._sam_shared_pool_on) or not self._sam_gpu_ready:
            return
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            return
        n_r = int(getattr(input_batch, "num_reqs", 0))
        nt = int(getattr(input_batch, "num_tokens", 0))
        if n_r <= 0 or nt <= 64 or not (n_r == 1 or (self._sam_batch_on and n_r <= self._sam_max_slots)):
            return
        n_blks = (nt + 255) // 256
        _sgen_arg = self._sam_slot_gen if self._sam_slot_gen is not None else self._sam_cur_len
        _lpck_arg = self._sam_lpc_key if self._sam_lpc_key is not None else self._sam_cur_len
        _lpcc_arg = self._sam_lpc_cont if self._sam_lpc_cont is not None else self._sam_cur_len
        _spt_arg = self._sam_sp_tag if self._sam_sp_tag is not None else self._sam_cur_len
        _spc_arg = self._sam_sp_cont if self._sam_sp_cont is not None else self._sam_cur_len
        _sam_lpc_ingest_chunk_kernel[(n_r, n_blks)](
            input_batch.input_ids,
            input_batch.positions,
            input_batch.query_start_loc,
            input_batch.idx_mapping,
            num_rejected,
            _sgen_arg,
            _lpck_arg,
            _lpcc_arg,
            _spt_arg,
            _spc_arg,
            MAX_SLOTS=self._sam_max_slots,
            LPC_CAP=self._sam_lpc_cap,
            LPC_STRIDE=self._sam_lpc_stride,
            SP_CAP=self._sam_sp_cap,
            HAS_LPC=self._sam_long_prompt_on,
            HAS_SHARED=self._sam_shared_pool_on,
            num_warps=4,
        )

    def _sam_launch_fused_dink_ingest(
        self,
        input_batch,
        num_sampled,
        num_rejected,
        last_sampled,
        temp,
        *,
        fuse_es: bool = False,
    ) -> None:
        n_r = int(input_batch.num_reqs)
        do_fuse_es = bool(fuse_es and self._es_bufs is not None and len(self._es_bufs) == 8)
        if do_fuse_es:
            skip_d, surv_d, tok_d, bestu_d, greedy_d, slot_d, db_d, lam_d = self._es_bufs
        else:
            skip_d = slot_d = greedy_d = self._vw_w
            surv_d = tok_d = bestu_d = db_d = lam_d = self._vw_step
        ts = self._tree_state if (self._tree_spec_on and 1 <= n_r <= self._tree_spec_max_b and n_r == 1) else None
        has_tree = ts is not None
        ls_flat = last_sampled.view(-1) if (last_sampled.dim() > 1 and last_sampled.is_contiguous()) else last_sampled
        _ht2_arg = self._sam_ht2_last if self._sam_ht2_last is not None else self._sam_cur_len
        _spt_arg = self._sam_sp_tag if self._sam_sp_tag is not None else self._sam_cur_len
        _spc_arg = self._sam_sp_cont if self._sam_sp_cont is not None else self._sam_cur_len
        _vw_dink_sam_ingest_fused_kernel[(n_r,)](
            input_batch.idx_mapping,
            num_sampled,
            num_rejected,
            temp,
            self._vw_w,
            self._vw_cpred,
            self._vw_db,
            self._vw_etok,
            self._vw_ems,
            self._vw_step,
            self._vw_cal,
            skip_d,
            surv_d,
            tok_d,
            bestu_d,
            greedy_d,
            slot_d,
            db_d,
            lam_d,
            input_batch.input_ids,
            input_batch.positions,
            input_batch.query_start_loc,
            ls_flat,
            self._sam_ring,
            self._sam_ht_last,
            self._sam_ht_prev,
            self._sam_cur_len,
            self._sam_ring_lo,
            ts.tree_active if has_tree else self._sam_cur_len,
            ts.accepted_path if has_tree else self._sam_cur_len,
            _ht2_arg,
            _spt_arg,
            _spc_arg,
            ETA_B=self._vw_dink_eta_b,
            DECAY_B=self._vw_dink_decay_b,
            B_CLIP=self._vw_dink_b_clip,
            ETA_LAM=self._vw_dink_eta_lam,
            K=self._dspark_k,
            FUSE_ES=do_fuse_es,
            MAX_SLOTS=self._sam_max_slots,
            RING_SZ=self._sam_ring_sz,
            HT_SZ=self._sam_ht_sz,
            MAX_INGEST=self._sam_max_ingest,
            HAS_TREE=has_tree,
            TREE_MAX_M=ts.max_m if has_tree else 8,
            HAS_HT2=self._sam_conf_boost_on,
            HT2_SZ=self._sam_ht2_sz,
            HAS_SHARED=self._sam_shared_pool_on,
            SP_CAP=self._sam_sp_cap,
            num_warps=1,
        )

    def _sam_launch_gpu_boost_extend(
        self,
        idx_ptr: torch.Tensor,
        temp_ptr: torch.Tensor,
        *,
        num_reqs: int = 1,
        idx_stride: int = 1,
        k_active: int | None = None,
        k_ext_cap: int | None = None,
    ) -> None:
        if not (
            self._sam_hybrid_on
            and self._sam_gpu_ready
            and self._conf_sink_on
            and self._vw_is_conf
            and self.draft_logits is None
        ):
            return
        n_r = int(num_reqs)
        ka = int(self._dspark_k if k_active is None else min(int(k_active), self._dspark_k))
        if n_r <= 0 or ka <= 0:
            return
        max_ext = max(0, int(self.num_speculative_steps - self._dspark_k))
        if k_ext_cap is None:
            ke = max_ext if (n_r == 1 or (self._sam_ext_k15_on and n_r <= self._sam_ext_max_b)) else 0
        else:
            ke = max(0, min(max_ext, int(k_ext_cap)))
        _ht2_arg = self._sam_ht2_last if self._sam_ht2_last is not None else self._sam_cur_len
        _sgen_arg = self._sam_slot_gen if self._sam_slot_gen is not None else self._sam_cur_len
        _lpck_arg = self._sam_lpc_key if self._sam_lpc_key is not None else self._sam_cur_len
        _lpcc_arg = self._sam_lpc_cont if self._sam_lpc_cont is not None else self._sam_cur_len
        _spt_arg = self._sam_sp_tag if self._sam_sp_tag is not None else self._sam_cur_len
        _spc_arg = self._sam_sp_cont if self._sam_sp_cont is not None else self._sam_cur_len
        _sam_gpu_boost_extend_kernel[(n_r,)](
            idx_ptr,
            temp_ptr,
            self.draft_tokens,
            self.draft_tokens.stride(0),
            self._conf_buf,
            self._conf_buf.stride(0),
            self._vw_cal,
            self._vw_db,
            self._sam_ring,
            self._sam_ht_last,
            self._sam_ht_prev,
            self._sam_cur_len,
            self._sam_ring_lo,
            _ht2_arg,
            _sgen_arg,
            _lpck_arg,
            _lpcc_arg,
            _spt_arg,
            _spc_arg,
            ka,
            ke,
            MAX_SLOTS=self._sam_max_slots,
            RING_SZ=self._sam_ring_sz,
            HT_SZ=self._sam_ht_sz,
            K_DRAFT=self._dspark_k,
            K_MAX=self.num_speculative_steps,
            MIN_MATCH=self._sam_min_match,
            BOOST_BASE=self._sam_boost_base,
            BOOST_MAX=self._sam_boost_max,
            EXT_SURV_THRESH=self._sam_ext_surv,
            FILL_EARLY_STOP=self._sam_fill_es,
            IDX_STRIDE=int(idx_stride),
            OVERRIDE_LOW_CONF=self._sam_override_low_conf,
            OVERRIDE_CONF_THRESH=self._sam_override_conf_thresh,
            OVERRIDE_MIN_MATCH=self._sam_override_min_match,
            OVERRIDE_CAL_INVERT=bool(self._sam_override_low_conf and self._sam_override_cal_invert),
            ALLOW_TEMP=self._sam_allow_temp,
            BEST_OF_2=self._sam_best_of_2,
            CONF_BOOST=self._sam_conf_boost_on,
            CONF_BOOST_ML2=self._sam_conf_boost_ml2,
            CONF_BOOST_ML3=self._sam_conf_boost_ml3,
            CONF_BOOST_ML4=self._sam_conf_boost_ml4,
            HAS_HT2=self._sam_conf_boost_on,
            HT2_SZ=self._sam_ht2_sz,
            HAS_LPC=self._sam_long_prompt_on,
            LPC_CAP=self._sam_lpc_cap,
            LPC_STRIDE=self._sam_lpc_stride,
            HAS_SHARED=self._sam_shared_pool_on,
            SP_CAP=self._sam_sp_cap,
            num_warps=1,
        )

    def _sam_apply_cpu_eager(self, input_batch, num_sampled, num_rejected, last_sampled) -> None:
        """Host `/rdma/libsam_fast.so` SAM advisor — STRICTLY outside CUDA graph capture!"""
        if not (self._sam_hybrid_on and self._sam_mode == "cpu" and input_batch.num_reqs == 1):
            return
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            return  # fail-closed: never run host CPU sync/staging inside graph capture
        idx_np = getattr(input_batch, "idx_mapping_np", None)
        if idx_np is not None and len(idx_np) >= 1:
            slot = int(idx_np[0])
        elif self._sam_allow_sync:
            slot = int(input_batch.idx_mapping[0].item())
        else:
            return
        adv = self._sam_get_cpu_advisor(slot)
        if self._sam_allow_sync:
            ctx_s = int(input_batch.query_start_loc[0].item())
            ctx_e = int(input_batch.query_start_loc[1].item())
            nr = int(num_rejected[0].item())
            ns = int(num_sampled[0].item())
            valid_e = ctx_e - nr
            if valid_e > ctx_s:
                adv.ingest_accepted(input_batch.input_ids[ctx_s:valid_e].detach().cpu().tolist())
            if ns > 0:
                adv.ingest_accepted([int(last_sampled[slot].item())])
            dt_list = self.draft_tokens[0, : self._dspark_k].detach().cpu().tolist()
            conf_list = self._conf_buf[0, : self._dspark_k].detach().cpu().tolist() if self._conf_sink_on else [0.7] * self._dspark_k
            res = adv.advise(dt_list, conf_list, conf_boost_floor=self._sam_conf_boost_on)
            self._sam_pin_tok.fill_(-1)
            self._sam_pin_boost.zero_()
            for k in range(self._dspark_k):
                if res["logit_boost"][k] > 0.0 and dt_list[k] >= 0:
                    self._sam_pin_tok[k] = dt_list[k]
                    self._sam_pin_boost[k] = res["logit_boost"][k]
            ext_toks = res["ext_tokens"]
            k_ext_max = self.num_speculative_steps - self._dspark_k
            if k_ext_max > 0 and ext_toks:
                for e in range(min(k_ext_max, len(ext_toks))):
                    self._sam_pin_tok[self._dspark_k + e] = int(ext_toks[e])
                self._sam_pin_ext_ok[0] = 1
            else:
                self._sam_pin_ext_ok[0] = 0
        else:
            if len(adv.sam) < self._sam_min_match:
                return
            m_tail = adv.sam.query_tail_plus_ext(
                tail_win=32, ext=(), min_match_len=self._sam_min_match, max_draft=self._dspark_k
            )
            if not m_tail.tokens:
                return
            self._sam_pin_tok.fill_(-1)
            self._sam_pin_boost.zero_()
            self._sam_pin_ext_ok[0] = 0
            bz = min(self._sam_boost_max, self._sam_boost_base + 0.18 * (m_tail.match_len - 2))
            for k, tk in enumerate(m_tail.tokens[: self._dspark_k]):
                self._sam_pin_tok[k] = int(tk)
                self._sam_pin_boost[k] = bz
        self._sam_dev_tok.copy_(self._sam_pin_tok, non_blocking=True)
        self._sam_dev_boost.copy_(self._sam_pin_boost, non_blocking=True)
        self._sam_dev_ext_ok.copy_(self._sam_pin_ext_ok, non_blocking=True)
        _sam_apply_staged_kernel[(1,)](
            self.draft_tokens,
            self._conf_buf,
            self._sam_dev_tok,
            self._sam_dev_boost,
            self._sam_dev_ext_ok,
            K_DRAFT=self._dspark_k,
            K_MAX=self.num_speculative_steps,
            num_warps=1,
        )

    def add_request(self, req_index: int) -> None:
        self._sam_reset_slot(req_index)
        if self._tree_state is not None:
            self._tree_state.reset_linear()
        if self._vw_mode != "off":
            self._vw_w[req_index] = self._vw_init_w
            self._vw_c[req_index] = _EMA_PRIOR
            if self._vw_is_dink:
                self._vw_db[req_index] = 0.0
                self._vw_etok[req_index] = self._vw_dink_init_tok
                self._vw_ems[req_index] = self._vw_dink_init_ms
                self._vw_cpred[req_index] = _EMA_PRIOR
            if self._cl_on:
                self._cl_rid[req_index] = self._cl_ctr
                self._cl_ctr += 1
            if self._conf_sink_on:
                self._conf_slot[req_index] = 0.0

    def prepare_propose(self, input_batch, num_sampled, num_rejected, sampler) -> None:
        super().prepare_propose(input_batch, num_sampled, num_rejected, sampler)
        if self._deferred_dink_update is not None:
            _du = self._deferred_dink_update
            self._deferred_dink_update = None
            self._vw_launch_update_dink(_du[0], _du[1], _du[2], _du[3], fuse_es=_du[4])
        if getattr(self, "_deferred_dink_prep", None) is not None:
            _dp = self._deferred_dink_prep
            self._deferred_dink_prep = None
            self._vw_launch_update_dink(_dp[0], _dp[1], _dp[2], _dp[3], fuse_es=_dp[4])
        if self._vw_mode == "off":
            return
        num_reqs = input_batch.num_reqs
        temp = sampler.sampling_states.temperature.gpu
        if self._cl_on:
            # pair last step's confidences with this step's verification; before the EMA moves
            self._cl_record(input_batch, num_sampled, num_rejected, temp)
        self._es_fused_step_idx_ptr = 0
        if self._vw_is_dink:
            fuse_es_now = (
                self._fuse_dink_es
                and self._es_on
                and num_reqs == 1
                and min(self._draft_k_for_graph(1), self._dspark_k) == self._dspark_k
            )
            can_defer_sam = (
                self._fuse_dink_sam_ingest
                and self._sam_hybrid_on
                and self._sam_gpu_ready
                and not self._cl_on
                and not self._vw_stats_on
                and (num_reqs == 1 or (self._sam_batch_on and 1 <= num_reqs <= self._sam_max_slots))
            )
            can_defer_prep = (
                not can_defer_sam
                and getattr(self, "_dspark_fuse_epilogue_prep", False)
                and self._fuse_dflash_prep
                and not self._cl_on
                and self._dops is not None
                and hasattr(self._dops, "v10_epilogue_prep_ready")
                and self._dops.v10_epilogue_prep_ready(temp.device)
                # Preserve strict ordering: when Batch-ES is active at B>=2, es_init_batch reads _vw_db BEFORE _fused_prep!
                and not (self._es_batch_on and 2 <= num_reqs <= self._es_batch_max_b)
            )
            if can_defer_sam:
                self._deferred_dink_update = (
                    input_batch.idx_mapping[:num_reqs],
                    num_sampled[:num_reqs],
                    num_rejected[:num_reqs],
                    temp,
                    fuse_es_now,
                    int(input_batch.idx_mapping.data_ptr()),
                    int(num_reqs),
                )
            elif can_defer_prep:
                self._deferred_dink_prep = (
                    input_batch.idx_mapping[:num_reqs],
                    num_sampled[:num_reqs],
                    num_rejected[:num_reqs],
                    temp,
                    fuse_es_now,
                    int(input_batch.idx_mapping.data_ptr()),
                    int(num_reqs),
                )
            else:
                self._vw_launch_update_dink(
                    input_batch.idx_mapping[:num_reqs],
                    num_sampled[:num_reqs],
                    num_rejected[:num_reqs],
                    temp,
                    fuse_es=fuse_es_now,
                )
            if fuse_es_now:
                self._es_fused_step_idx_ptr = int(input_batch.idx_mapping.data_ptr())
        need_update = (
            self._vw_mode in ("ema", "conf+ema", "shadow")
            or self._cl_on
            or (self._vw_stats_on and not self._vw_is_dink)
        )
        if need_update:
            self._vw_launch_update(
                input_batch.idx_mapping[:num_reqs],
                num_sampled[:num_reqs],
                num_rejected[:num_reqs],
                temp,
            )
        if self._vw_stats_on:
            self._vw_calls += 1
            if self._vw_calls % 512 == 0:
                st = self._vw_stats.tolist()  # host sync, stats only
                steps = max(st[8], 1)
                logger.info(
                    "verify-W hist(W0..7)=%s steps=%d mean accepted=%.3f",
                    st[:8],
                    st[8],
                    st[9] / steps,
                )

    def propose(self, input_batch, *args: Any, **kwargs: Any) -> torch.Tensor:
        fused_idx_ptr = self._es_fused_step_idx_ptr
        self._es_fused_step_idx_ptr = 0
        self._last_sample_seq_eager = False
        self._last_tree_handled = False
        self._last_propose_post_graph_eager = False
        is_dummy = bool(kwargs.get("dummy_run", False))
        skip_attn_dummy = bool(kwargs.get("skip_attn_for_dummy_run", False))
        num_spec_arg = kwargs.get("num_speculative_tokens")
        skip_draft = (num_spec_arg == 0) or bool(getattr(self, "context_kv_only", False))
        _trim_ctxkv = (
            self._pf_trim_draft_ctxkv
            if self._spec_fastpath_on
            else (
                os.environ.get("INKLING_PF_TRIM_DRAFT_CTXKV", "0").strip().lower()
                not in ("0", "", "off", "false", "no")
            )
        )
        if _trim_ctxkv and not hasattr(self, "_pf_max_draft_keep"):
            layers = getattr(getattr(self.model, "model", None), "layers", ())
            if layers and all(
                getattr(ly.self_attn, "_effective_is_local", False) and getattr(ly.self_attn, "window", 0) > 0
                for ly in layers
            ):
                self._pf_max_draft_keep = max(int(ly.self_attn.window) + 512 for ly in layers)
            else:
                self._pf_max_draft_keep = 0
        _sam_nr = int(getattr(input_batch, "num_reqs", 0))
        _sam_ok_b = (_sam_nr == 1) or (self._sam_batch_on and 1 <= _sam_nr <= self._sam_max_slots)
        _sam_can_ingest = (
            self._sam_hybrid_on
            and self._sam_gpu_ready
            and not is_dummy
            and not skip_attn_dummy
            and _sam_ok_b
        )
        _deferred = self._deferred_dink_update
        self._deferred_dink_update = None
        _sam_ingested = False
        if _deferred is not None:
            _ns_d = kwargs.get("num_sampled", args[4] if len(args) > 4 else None)
            _nr_d = kwargs.get("num_rejected", args[5] if len(args) > 5 else None)
            _ls_d = kwargs.get("last_sampled", args[6] if len(args) > 6 else None)
            if (
                _sam_can_ingest
                and _ns_d is not None
                and _nr_d is not None
                and _ls_d is not None
                and int(input_batch.idx_mapping.data_ptr()) == _deferred[5]
                and _sam_nr == _deferred[6]
            ):
                self._sam_launch_fused_dink_ingest(
                    input_batch, _ns_d, _nr_d, _ls_d, _deferred[3], fuse_es=_deferred[4]
                )
                _sam_ingested = True
            else:
                self._vw_launch_update_dink(
                    _deferred[0], _deferred[1], _deferred[2], _deferred[3], fuse_es=_deferred[4]
                )
        if not _sam_ingested and _sam_can_ingest:
            _ns = kwargs.get("num_sampled", args[4] if len(args) > 4 else None)
            _nr = kwargs.get("num_rejected", args[5] if len(args) > 5 else None)
            _ls = kwargs.get("last_sampled", args[6] if len(args) > 6 else None)
            if _ns is not None and _nr is not None and _ls is not None:
                self._sam_launch_gpu_ingest(input_batch, _ns, _nr, _ls)
                _sam_ingested = True
        if _sam_can_ingest and (self._sam_long_prompt_on or self._sam_shared_pool_on):
            _nr_lpc = kwargs.get("num_rejected", args[5] if len(args) > 5 else None)
            if _nr_lpc is not None and int(getattr(input_batch, "num_tokens", 0)) > 64:
                self._sam_launch_lpc_chunk_ingest(input_batch, _nr_lpc)
        is_single_pf = (
            _trim_ctxkv
            and _sam_nr == 1
            and int(getattr(input_batch, "num_tokens", 0)) > 2048
            and not is_dummy
        )
        _def_prep = getattr(self, "_deferred_dink_prep", None)
        if _def_prep is not None and (
            skip_draft
            or is_dummy
            or skip_attn_dummy
            or not self._fuse_dflash_prep
            or int(input_batch.idx_mapping.data_ptr()) != _def_prep[5]
            or _sam_nr != _def_prep[6]
        ):
            self._deferred_dink_prep = None
            self._vw_launch_update_dink(_def_prep[0], _def_prep[1], _def_prep[2], _def_prep[3], fuse_es=_def_prep[4])
        if skip_draft:
            # Do NOT clear self._conf_armed: CUDA-graph replays do not re-run _sample_sequential.
            if is_single_pf:
                self.model._pf_single_req_chunk = True
                try:
                    return super().propose(input_batch, *args, **kwargs)
                finally:
                    self.model._pf_single_req_chunk = False
            return super().propose(input_batch, *args, **kwargs)
        if is_dummy and (self._graph_vw_conf_captured or self._markov_persistent):
            # Guarantee in-graph _vw_conf_kernel / persistent Markov sees slot=-1 and writes zero state on dummy_run.
            self.sample_idx_mapping[: self.max_num_reqs * self.num_speculative_steps].fill_(-1)
        if (
            self._es_on
            and _sam_nr == 1
            and min(self._draft_k_for_graph(1), self._dspark_k) == self._dspark_k
        ):
            if is_dummy:
                if self._vw_is_dink:
                    self._dops.es_init_dink(
                        self._es_dummy_idx, self._es_dummy_temp, self._vw_step, self._vw_cal,
                        self._vw_db, self._vw_etok, self._vw_ems, self._es_bufs, dummy=True,
                    )
                else:
                    self._dops.es_init(
                        self._es_dummy_idx, self._es_dummy_temp, self._vw_step, self._es_bufs, dummy=True
                    )
            elif not (
                self._vw_is_dink
                and fused_idx_ptr != 0
                and int(input_batch.idx_mapping.data_ptr()) == fused_idx_ptr
            ):
                temp = kwargs.get("temperature")
                if temp is None:
                    temp = args[8] if len(args) > 8 else self._es_dummy_temp
                if self._vw_is_dink:
                    self._dops.es_init_dink(
                        input_batch.idx_mapping[:1], temp, self._vw_step, self._vw_cal,
                        self._vw_db, self._vw_etok, self._vw_ems, self._es_bufs, dummy=False,
                    )
                else:
                    self._dops.es_init(
                        input_batch.idx_mapping[:1], temp, self._vw_step, self._es_bufs, dummy=False
                    )
        if (
            self._es_batch_on
            and self._es_batch_bufs is not None
            and 2 <= _sam_nr <= self._es_batch_max_b
        ):
            if is_dummy:
                _eb_dummy = torch.zeros(_sam_nr, dtype=torch.int64, device=input_batch.idx_mapping.device)
                self._dops.es_init_batch(
                    _eb_dummy, self._vw_step, self._vw_db, self._es_batch_bufs, _sam_nr, dummy=True
                )
            else:
                _temp_eb = kwargs.get("temperature")
                if _temp_eb is None:
                    _temp_eb = args[8] if len(args) > 8 else self._vw_step
                self._dops.es_init_batch(
                    input_batch.idx_mapping[:_sam_nr], _temp_eb, self._vw_db,
                    self._es_batch_bufs, _sam_nr, dummy=False,
                )
        n_draft_groups = len(getattr(self, "draft_kv_cache_group_ids", ()))
        use_fused_prep = (
            self._fuse_dflash_prep
            and not is_dummy
            and not skip_attn_dummy
            and 1 <= n_draft_groups <= 5
            and getattr(self, "dcp_size", 1) == 1
            and not self.sample_from_anchor
        )
        if use_fused_prep:
            _FUSED_PREP_TLS.spec = self
            _FUSED_PREP_TLS.done = False
            _FUSED_PREP_TLS.num_rejected_written = False
            _FUSED_PREP_TLS.ctx_pad_range = None
        if self._tree_runtime_mod is not None:
            self._tree_runtime_mod.set_active_batch_size(0 if is_dummy else _sam_nr)
        if self._tree_state is not None and (_sam_nr > self._tree_spec_max_b or _sam_nr != 1 or is_dummy):
            # Reset full linear tables so stale tree_active / n_live / tree_parent never leak into B>=2!
            self._tree_state.reset_linear()
        if not is_dummy and self._l2pf is not None and hasattr(self._l2pf, "join_cross_boundary"):
            self._l2pf.join_cross_boundary()
        if is_single_pf:
            self.model._pf_single_req_chunk = True
        try:
            drafts = super().propose(input_batch, *args, **kwargs)
            if getattr(self, "_deferred_dink_prep", None) is not None:
                _dp_rem = self._deferred_dink_prep
                self._deferred_dink_prep = None
                self._vw_launch_update_dink(_dp_rem[0], _dp_rem[1], _dp_rem[2], _dp_rem[3], fuse_es=_dp_rem[4])
        finally:
            if is_single_pf:
                self.model._pf_single_req_chunk = False
            if use_fused_prep:
                _FUSED_PREP_TLS.spec = None
                _FUSED_PREP_TLS.done = False
                _FUSED_PREP_TLS.num_rejected_written = False
                _FUSED_PREP_TLS.ctx_pad_range = None
        if (
            self._sam_hybrid_on
            and not is_dummy
            and not skip_attn_dummy
            and _sam_ok_b
            and not self._last_tree_handled
        ):
            if self._sam_mode == "cpu" and _sam_nr == 1:
                _ns = kwargs.get("num_sampled", args[4] if len(args) > 4 else None)
                _nr = kwargs.get("num_rejected", args[5] if len(args) > 5 else None)
                _ls = kwargs.get("last_sampled", args[6] if len(args) > 6 else None)
                if _ns is not None and _nr is not None and _ls is not None:
                    self._last_propose_post_graph_eager = True
                    self._sam_apply_cpu_eager(input_batch, _ns, _nr, _ls)
            elif self._sam_gpu_ready and self.draft_logits is None:
                _draft_g = self._next_draft_cap_g(_sam_nr)
                if _draft_g is not None:
                    _sam_in_g = _draft_g in self._sam_boost_in_graph_captured
                elif self._spec_fastpath_on and 1 <= _sam_nr < len(self._sam_cap_g_lut):
                    _sam_in_g = self._sam_cap_g_lut[_sam_nr] is not None
                else:
                    _sam_in_g = any(g >= _sam_nr for g in self._sam_boost_in_graph_captured)
                if self._last_sample_seq_eager or not _sam_in_g:
                    _temp_sam = kwargs.get("temperature", args[8] if len(args) > 8 else None)
                    _ka_eager = min(int(self._last_draft_k), self._dspark_k)
                    if _temp_sam is not None and _ka_eager > 0:
                        self._last_propose_post_graph_eager = True
                        _wcap_sam_e = int(self._effective_w_cap(_sam_nr, num_spec_arg))
                        _ke_eager = (
                            max(0, _wcap_sam_e - self._dspark_k)
                            if (_sam_nr == 1 or (self._sam_ext_k15_on and _sam_nr <= self._sam_ext_max_b))
                            else 0
                        )
                        self._sam_launch_gpu_boost_extend(
                            input_batch.idx_mapping[:_sam_nr],
                            _temp_sam,
                            num_reqs=_sam_nr,
                            idx_stride=1,
                            k_active=_ka_eager,
                            k_ext_cap=_ke_eager,
                        )
        if self._vw_mode != "off" and not is_dummy:
            # temperature: positional arg 9 of BaseSpeculator.propose (after
            # input_batch), keyword in the runner; handled below.
            temp = kwargs.get("temperature")
            if temp is None:
                temp = args[8]
            num_reqs = drafts.shape[0]
            w_cap = self._effective_w_cap(num_reqs, num_spec_arg)
            tau_ov = self._effective_tau_override(num_reqs)
            armed = self._conf_sink_on and self._conf_armed
            req_drafts = getattr(_DRAFT_STORE_TLS, "req_drafts", None)
            allow_ext_bn = (
                self._sam_hybrid_on
                and self.num_speculative_steps > self._dspark_k
                and (num_reqs == 1 or (self._sam_ext_k15_on and num_reqs <= self._sam_ext_max_b and w_cap > self._dspark_k))
            )
            clear_tail_bn = (
                num_reqs >= 2
                and self.num_speculative_steps > self._dspark_k
                and drafts.shape[1] <= self._dspark_k
                and not allow_ext_bn
            )
            if (
                self._spec_fastpath_on
                and req_drafts is not None
                and req_drafts is self._graph_req_drafts
                and self.num_speculative_steps == self._dspark_k
                and drafts.data_ptr() == self.draft_tokens.data_ptr()
            ):
                write_req_ok = True
            else:
                write_req_ok = (
                    req_drafts is not None
                    and isinstance(req_drafts, torch.Tensor)
                    and req_drafts.dim() == 2
                    and req_drafts.dtype == self.draft_tokens.dtype
                    and req_drafts.device == self.draft_tokens.device
                    and req_drafts.stride(1) == 1
                    and req_drafts.shape[1] == self.num_speculative_steps
                    and self.draft_tokens.stride(1) == 1
                    and 0 < drafts.shape[1] <= self.num_speculative_steps
                    and drafts.data_ptr() == self.draft_tokens.data_ptr()
                    and (
                        self.num_speculative_steps == self._dspark_k
                        or allow_ext_bn
                        or clear_tail_bn
                    )
                )
            if self._vw_is_conf:
                if armed:
                    want_req_ptr = int(req_drafts.data_ptr()) if write_req_ok else 0
                    if (
                        self._graph_vw_conf_captured
                        and getattr(self, "query_cudagraph_manager", None) is not None
                        and not kwargs.get("is_profile", False)
                    ):
                        if self._spec_fastpath_on and 1 <= num_reqs < len(self._vw_cap_g_lut):
                            cap_g = self._vw_cap_g_lut[num_reqs]
                        else:
                            cap_g = min(
                                (g for g in self._graph_vw_conf_captured if g >= num_reqs),
                                default=None,
                            )
                    else:
                        cap_g = None
                    tree_handled = (
                        self._tree_spec_on
                        and num_reqs == 1
                        and (
                            self._last_tree_handled
                            or (not self._last_sample_seq_eager and 1 in self._tree_graph_captured)
                        )
                    )
                    tree_wrote_req = (
                        tree_handled
                        and want_req_ptr != 0
                        and self._tree_graph_captured.get(1, 0) == want_req_ptr
                    )
                    use_wf_eager = (
                        self._hetero_budget_on
                        and self._wf_step_tbl is not None
                        and self._vw_is_dink
                        and not self._cl_on
                        and 2 <= num_reqs <= _WF_MAX_B
                    )
                    eff_tau_key = -2.0 if use_wf_eager else float(tau_ov)
                    already_in_graph = tree_handled or (
                        not self._last_sample_seq_eager
                        and cap_g is not None
                        and self._graph_vw_conf_captured[cap_g]
                        == (int(w_cap), eff_tau_key, want_req_ptr)
                    )
                    if not already_in_graph:
                        self._last_propose_post_graph_eager = True
                        idx = input_batch.idx_mapping[:num_reqs]
                        _ib_seq = getattr(input_batch, "seq_lens", None)
                        if (
                            self._mk_p_captured
                            and num_reqs == 1
                            and self._vw_is_dink
                            and not self._cl_on
                            and self._draft_k_for_graph(1) == self.num_speculative_steps
                        ):
                            # Persistent Markov kernel already wrote vw_w, vw_cpred, and masked draft_tokens[0].
                            if write_req_ok or w_cap < self.num_speculative_steps:
                                self._vw_launch_mask(
                                    self.draft_tokens,
                                    idx,
                                    temp,
                                    w_cap=w_cap,
                                    req_drafts=req_drafts if write_req_ok else None,
                                )
                        elif use_wf_eager:
                            self._vw_launch_hetero_waterfill(
                                idx,
                                temp,
                                self.draft_tokens,
                                w_cap=min(w_cap, self._dspark_k),
                                req_drafts=req_drafts if write_req_ok else None,
                                num_reqs=num_reqs,
                                idx_stride=1,
                                clear_req_tail=clear_tail_bn,
                                seq_lens=_ib_seq,
                            )
                        else:
                            self._vw_launch_conf(
                                idx,
                                temp,
                                self.draft_tokens,
                                w_cap=min(w_cap, self._dspark_k),
                                tau_override=tau_ov,
                                req_drafts=req_drafts if write_req_ok else None,
                                clear_req_tail=clear_tail_bn,
                                seq_lens=_ib_seq,
                            )
                            if allow_ext_bn:
                                _sam_mask_ext_tail_kernel[(num_reqs,)](
                                    idx,
                                    self._vw_w,
                                    self.draft_tokens,
                                    self.draft_tokens.stride(0),
                                    req_drafts if write_req_ok else self.draft_tokens,
                                    req_drafts.stride(0) if write_req_ok else self.draft_tokens.stride(0),
                                    max(0, int(w_cap) - self._dspark_k),
                                    K_DRAFT=self._dspark_k,
                                    K_MAX=self.num_speculative_steps,
                                    WRITE_REQ_DRAFTS=write_req_ok,
                                    IDX_STRIDE=1,
                                    num_warps=1,
                                )
                    if write_req_ok and (not tree_handled or tree_wrote_req):
                        _DRAFT_STORE_TLS.written_drafts = drafts
                        _DRAFT_STORE_TLS.written_full_drafts = (
                            drafts
                            if (self._spec_fastpath_on and drafts.shape[1] == self.num_speculative_steps)
                            else self.draft_tokens[:num_reqs]
                        )
                        _DRAFT_STORE_TLS.written_idx = input_batch.idx_mapping
                elif not self._conf_warned:
                    self._conf_warned = True
                    logger.warning(
                        "INKLING_VERIFY_W=%s: the draft loop did not produce confidences "
                        "(sampling draft path?); verifying all %d drafts",
                        self._vw_mode,
                        w_cap,
                    )
            else:
                idx = input_batch.idx_mapping[:num_reqs]
                if self._vw_apply:
                    self._last_propose_post_graph_eager = True
                    self._vw_launch_mask(
                        self.draft_tokens,
                        idx,
                        temp,
                        w_cap=min(w_cap, self._dspark_k),
                        req_drafts=req_drafts if write_req_ok else None,
                        clear_req_tail=clear_tail_bn,
                    )
                    if self._sam_hybrid_on and self.num_speculative_steps > self._dspark_k and num_reqs == 1:
                        _sam_mask_ext_tail_kernel[(1,)](
                            idx,
                            self._vw_w,
                            self.draft_tokens,
                            self.draft_tokens.stride(0),
                            req_drafts if write_req_ok else self.draft_tokens,
                            req_drafts.stride(0) if write_req_ok else self.draft_tokens.stride(0),
                            K_DRAFT=self._dspark_k,
                            K_MAX=self.num_speculative_steps,
                            WRITE_REQ_DRAFTS=write_req_ok,
                            num_warps=1,
                        )
                    if write_req_ok:
                        _DRAFT_STORE_TLS.written_drafts = drafts
                        _DRAFT_STORE_TLS.written_full_drafts = (
                            drafts
                            if (self._spec_fastpath_on and drafts.shape[1] == self.num_speculative_steps)
                            else self.draft_tokens[:num_reqs]
                        )
                        _DRAFT_STORE_TLS.written_idx = input_batch.idx_mapping
                if armed and self._cl_on:
                    self._vw_launch_store(idx)
        if (
            not is_dummy
            and self._l2pf is not None
            and _sam_nr <= getattr(self, "_l2pf_target_l0_max_reqs", 2)
            and hasattr(self._l2pf, "fork_cross_boundary")
        ):
            self._l2pf.fork_cross_boundary("target_l0")
        return drafts

    def _conf_end(self, expected_cols: int) -> None:
        """After the Markov loop: every executed step must have left its confidence column."""
        if self._conf_sink_on and self._conf_armed:
            sink = self._dops._CONF_SINK
            if sink is None or sink.col != expected_cols:
                self._conf_armed = False
                logger.warning(
                    "confidence sink incomplete (%s of %d columns); verifying all drafts",
                    getattr(sink, "col", None),
                    expected_cols,
                )

    def _conf_begin(self) -> None:
        """Reset the confidence column counter (trace time: also while the draft graph is captured)."""
        if self._conf_sink_on:
            self._conf_armed = self._dops.conf_sink_begin() and self.draft_logits is None

    def draft_logits_spec(self, vllm_config: VllmConfig) -> tuple[torch.dtype, float]:
        # Draft base logits come from a bf16 GEMM, unlike the target's fp32 head.
        return self.dtype, 0.0

    def load_draft_model(
        self, target_model: nn.Module, target_attn_layer_names: set[str]
    ) -> nn.Module:
        model = load_dspark_model(target_model, self.vllm_config)
        model.attach_target(target_model.get_language_model())
        return model

    def prepare_context_anchor(
        self,
        num_reqs: int,
        query_start_loc: torch.Tensor,
        num_rejected: torch.Tensor | None,
    ) -> None:
        assert num_rejected is not None
        if self._fuse_ctx_anchor and num_reqs > 0:
            dops = getattr(self, "_dops", None)
            if dops is None:
                try:
                    from .ops import dspark as dops
                except Exception:
                    dops = None
            fn = getattr(dops, "context_anchor_mix_rmsnorm", None) if dops is not None else None
            dec = getattr(self.model, "model", None)
            if (
                fn is not None
                and dec is not None
                and hasattr(dec, "fuse_alpha")
                and hasattr(dec, "markov_head")
                and self.hidden_states.dtype == self.f_rnn.dtype
                and self.hidden_states.is_contiguous()
            ):
                norm = dec.markov_head.dflare_norm
                fn(
                    self.hidden_states,
                    query_start_loc,
                    num_rejected,
                    dec.fuse_alpha[-1],
                    norm.weight,
                    self.f_rnn,
                    num_reqs,
                    len(self.model.config.target_layer_ids),
                    self.model.config.hidden_size,
                    norm.eps,
                )
                return
        ends = query_start_loc[1 : num_reqs + 1] - num_rejected[:num_reqs]
        rows = self.hidden_states[(ends - 1).clamp_min(0).long()]
        self.f_rnn[:num_reqs] = self.model.compute_f_rnn(rows)

    def _sample_sequential(self, num_reqs: int, head_hidden: torch.Tensor) -> None:
        n_spec = self.num_speculative_steps
        k_run = min(self._draft_k_for_graph(num_reqs), getattr(self, "_dspark_k", n_spec))
        self._last_draft_k = k_run
        dspark_k = getattr(self, "_dspark_k", n_spec)
        if k_run <= 0:
            # Dynamic SD K=0 for this graph bucket: mark all draft positions invalid (-1)
            # and skip base_logits, rnn_proj, and all 7 Markov steps + RDMA AGs.
            self.draft_tokens[:num_reqs, :n_spec].fill_(-1)
            if self._conf_sink_on:
                self._conf_buf[:num_reqs, : min(n_spec, self._conf_buf.shape[1])].zero_()
            return
        num_sample = num_reqs * n_spec
        fast_b1_slice = (
            num_reqs == 1
            and not self.sample_from_anchor
            and head_hidden.shape[0] >= 1 + k_run
        )
        if fast_b1_slice:
            # For B=1 & not sample_from_anchor, sample_indices[:k_run] == [1, ..., k_run]:
            # zero-copy contiguous view avoids index_select copy kernel (~4 us saved).
            hidden_in = head_hidden[1 : 1 + k_run]
        elif k_run < n_spec:
            # Only columns 0..k_run-1 of sample_indices are active (M=16 decouple or Dynamic SD).
            sidx_k = self.sample_indices[:num_sample].view(num_reqs, n_spec)[:, :k_run].reshape(-1)
            hidden_in = head_hidden[sidx_k]
        else:
            hidden_in = head_hidden[self.sample_indices[:num_sample]]
        hidden = hidden_in
        base_logits = self.model.compute_base_logits(hidden_in).view(num_reqs, k_run, -1)
        head = self.model.model.markov_head
        l2pf = self._l2pf
        fold_pf_forked = False
        if l2pf is not None and l2pf.DRAFT and num_reqs <= getattr(self, "_l2pf_fold_max_reqs", 1):
            fw = head.fold_w8 if head.fold_w8 is not None else head.fold_weight
            if fw is not None:
                l2pf.fork(fw, kind="draft", stop=False)
                fold_pf_forked = True
        sched = getattr(self.model.model, "_ctx_sched", None)
        if sched is not None:
            sched.join_frnn()
        context = head.context(hidden_in, self.f_rnn[:num_reqs])
        if fold_pf_forked:
            l2pf.join()  # no-op on G2 TMA (continues prefetching during rnn_step_0); joins legacy Triton
        idx_map = self.sample_idx_mapping[:num_sample].view(num_reqs, n_spec)
        sample_pos = self.sample_pos[:num_sample].view(num_reqs, n_spec)
        self._conf_begin()
        use_fused = self._markov_fused and self.draft_logits is None and self.model.argmax_workspace.fallback
        use_es = use_fused and self._es_on and num_reqs == 1 and k_run == dspark_k
        use_hist = self._vw_mode == "conf+ema"
        # Always skip the zero-state FillFunctor (~2 us) and anchor index_elementwise_kernel (~7.1 us)
        # on the fused Markov path (independent of INKLING_MARKOV_FUSE_REDUCE=0!).
        # markov_rnn_step_fused natively handles state=None (init_zero=True) and strided prev.
        state: torch.Tensor | None = (
            None
            if use_fused
            else hidden.new_zeros(num_reqs, self.model.config.markov_rank)
        )
        if (
            (use_fused or num_reqs == 1)
            and self.input_buffers.input_ids.dim() == 1
            and self.input_buffers.input_ids.shape[0] >= num_reqs * self.num_query_per_req
        ):
            # Zero-copy anchor view matching self._anchor_idx[:num_reqs] == [0, Q, 2Q, ...]
            # (eliminates 7.1 us index_elementwise_kernel; markov_rnn_step_fused uses prev.stride(0)).
            prev = (
                self.input_buffers.input_ids[:1]
                if num_reqs == 1
                else self.input_buffers.input_ids[
                    : num_reqs * self.num_query_per_req : self.num_query_per_req
                ]
            )
        else:
            prev = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]
        _is_cap = torch.cuda.is_current_stream_capturing()
        # Option B (INKLING_TREE_SPEC=1): 100% Capture-Safe GPU-Native Beam-2 Tree + Dinkelbach DP
        # Wave-V10: enforces strict B <= _tree_spec_max_b (B==1 only) AND supports VOCAB_HOT draft_id_map!
        _tree_id_map = getattr(self.model, "draft_id_map", None)
        if (
            self._tree_spec_on
            and self._tree_state is not None
            and 1 <= num_reqs <= self._tree_spec_max_b
            and num_reqs == 1
            and use_fused
            and k_run == 7
            and n_spec == 7
            and self._vw_is_conf
            and self._conf_sink_on
            and getattr(self._dops, "_MARKOV_FUSE_REDUCE", False)
            and (
                _tree_id_map is None
                or (_tree_id_map.dtype == torch.int32 and _tree_id_map.is_contiguous())
            )
        ):
            wrote_ptr = self._sample_tree_beam_graph_safe(
                head=head,
                base_logits=base_logits,
                context=context,
                anchor_tok=prev,
                use_es=use_es,
                use_hist=use_hist,
            )
            if fold_pf_forked:
                l2pf.join_end()
            if not _is_cap:
                self._last_sample_seq_eager = True
                self._last_tree_handled = True
            else:
                self._tree_graph_captured[1] = wrote_ptr
                if self._sam_hybrid_on and self._sam_gpu_ready:
                    self._sam_boost_in_graph_captured.add(1)
                    self._rebuild_cap_lut(self._sam_cap_g_lut, self._sam_boost_in_graph_captured)
            return
        persistent_ok = False
        persistent_did_graph_vw = False
        if (
            self._markov_persistent
            and not (self._sam_hybrid_on and self._sam_gpu_ready)
            and use_es
            and self._vw_is_dink
            and self._mk_p_ctrl is not None
            and head.fold_w8 is not None
            and head.fold_s8 is not None
            and head.fold_bias is not None
            and head.rnn_w_rec is not None
            and k_run == n_spec
            and num_reqs == 1
        ):
            from vllm.distributed import get_tp_group

            comm = getattr(get_tp_group().device_communicator, "rdma_comm", None)
            if comm is not None and comm.has_markov_persistent():
                if fold_pf_forked:
                    # Join `l2pf._side` BEFORE `cudaLaunchCooperativeKernel(48 CTAs)` on 48-SM GB10!
                    l2pf.join_end()
                    fold_pf_forked = False
                fuse_g_vw = (
                    self._graph_vw_conf_on
                    and getattr(self, "dp_size", 1) == 1
                    and self._vw_is_conf
                    and self._conf_sink_on
                    and self._conf_armed
                    and not self._cl_on
                    and not self._vw_stats_on
                    and self.draft_logits is None
                    and _is_cap
                )
                w_cap_g = int(self._effective_w_cap(1, None))
                req_d = self._graph_req_drafts if fuse_g_vw else None
                sidx0 = self.sample_idx_mapping[:1] if fuse_g_vw else None
                persistent_ok = comm.markov_loop_persistent(
                    base_logits[0],
                    context.view(k_run, -1),
                    head.markov_w1.weight,
                    head.rnn_w_rec,
                    head.fold_w8,
                    head.fold_s8,
                    head.fold_bias,
                    self.model.draft_id_map,
                    prev,
                    float(self.model.config.final_logit_softcapping),
                    int(self.model.valid_vocab),
                    int(self.model.vocab_offset),
                    int(k_run),
                    self._es_bufs[4],
                    self._es_bufs[5],
                    self._es_bufs[6],
                    self._es_bufs[7],
                    self._vw_step,
                    self._vw_cal,
                    self._vw_w,
                    self._vw_cpred,
                    self.draft_tokens[0],
                    self._conf_buf[0] if self._conf_sink_on else None,
                    w_cap_g,
                    self._mk_p_state,
                    self._mk_p_latent,
                    self._mk_p_rec,
                    self._mk_p_ctrl,
                    req_drafts=req_d,
                    sample_idx0=sidx0,
                )
                if persistent_ok:
                    self._mk_p_captured = True
                    if self._conf_sink_on and self._dops is not None and self._dops._CONF_SINK is not None:
                        self._dops._CONF_SINK.col = k_run
                    if fuse_g_vw:
                        persistent_did_graph_vw = True
                        self._graph_vw_conf_captured[1] = (
                            w_cap_g,
                            float(self._effective_tau_override(1)),
                            int(req_d.data_ptr()) if req_d is not None else 0,
                        )
                        self._rebuild_cap_lut(self._vw_cap_g_lut, self._graph_vw_conf_captured.keys())
        use_es_batch = (
            use_fused
            and self._es_batch_on
            and self._es_batch_bufs is not None
            and 2 <= num_reqs <= self._es_batch_max_b
            and k_run >= 2
        )
        cross_step_ok = False
        tail_vw_fused = False
        _tail_vw_ctx = None
        if (
            not persistent_ok
            and use_fused
            and self._markov_fuse_cross_step
            and k_run >= 2
            and self._dops is not None
            and hasattr(self._dops, "run_markov_loop_v9_fused")
        ):
            _es_v9 = (
                (dspark_k, self._es_bufs, self._vw_c, self._vw_cal, self._vw_step,
                 use_hist, _PRIOR_LOGIT, self._vw_is_dink)
                if use_es
                else None
            )
            _es_batch_v10 = (
                (k_run, self._es_batch_bufs, self._vw_cal, self._es_batch_tau)
                if use_es_batch
                else None
            )
            if (
                getattr(self, "_dspark_fuse_tail_vw", False)
                and _is_cap
                and self._graph_vw_conf_on
                and self._is_graph_vw_bucket_uniform(int(num_reqs))
                and getattr(self, "dp_size", 1) == 1
                and self._vw_is_conf
                and self._conf_sink_on
                and self._conf_armed
                and not self._cl_on
                and not self._vw_stats_on
                and not (self._sam_hybrid_on and self._sam_gpu_ready)
                and self.draft_logits is None
                and self.sample_idx_mapping.dtype == torch.int64
                and hasattr(self._dops, "v10_tail_vw_ready")
                and self._dops.v10_tail_vw_ready(base_logits.device)
            ):
                _w_cap_f = min(int(self._effective_w_cap(num_reqs, None)), k_run)
                _tau_ov_f = float(self._effective_tau_override(num_reqs))
                _clear_tail_f = bool(num_reqs >= 2 and n_spec > dspark_k and k_run <= dspark_k)
                _can_req_f = bool((n_spec == dspark_k) or _clear_tail_f)
                _req_d_f = self._graph_req_drafts if _can_req_f else None
                _tail_vw_ctx = {
                    "n_spec": int(n_spec),
                    "c_end": int(min(n_spec, self._conf_buf.shape[1])),
                    "idx": self.sample_idx_mapping,
                    "idx_stride": int(n_spec),
                    "conf_buf": self._conf_buf,
                    "temp": self.temperature,
                    "w": self._vw_w,
                    "c": self._vw_c,
                    "cal": self._vw_cal,
                    "step": self._vw_step,
                    "cslot": self._conf_slot,
                    "db": self._vw_db,
                    "etok": self._vw_etok,
                    "ems": self._vw_ems,
                    "cpred": self._vw_cpred,
                    "w_cap": int(min(_w_cap_f, dspark_k)),
                    "w_cap_full": int(_w_cap_f),
                    "tau_override": _tau_ov_f,
                    "req_drafts": _req_d_f,
                    "prior_logit": float(_PRIOR_LOGIT),
                    "K": int(dspark_k),
                    "use_hist": bool(use_hist),
                    "use_dink": bool(self._vw_is_dink),
                    "do_log": bool(self._cl_on),
                    "clear_req_tail": bool(_req_d_f is not None and _clear_tail_f and n_spec > dspark_k),
                    "req_k": int(n_spec),
                    "ran": False,
                }
            cross_step_ok = self._dops.run_markov_loop_v9_fused(
                base_logits,
                context,
                prev,
                head.markov_w1.weight,
                head.rnn_w_rec,
                head.fold_weight,
                head.fold_bias,
                float(self.model.config.final_logit_softcapping),
                int(self.model.valid_vocab),
                int(self.model.vocab_offset),
                self.draft_tokens,
                fold_w8=head.fold_w8,
                fold_s8=head.fold_s8,
                conf_buf=self._conf_buf if self._conf_sink_on else None,
                es_ctx=_es_v9,
                es_batch_ctx=_es_batch_v10,
                id_map=self.model.draft_id_map,
                pdl=getattr(self._dops, "_DRAFT_PDL", False),
                tail_vw_ctx=_tail_vw_ctx,
            )
            if cross_step_ok and _tail_vw_ctx is not None and _tail_vw_ctx.get("ran", False):
                tail_vw_fused = True
                persistent_did_graph_vw = True
                _req_d_f = _tail_vw_ctx["req_drafts"]
                self._graph_vw_conf_captured[int(num_reqs)] = (
                    int(_tail_vw_ctx["w_cap_full"]),
                    float(_tail_vw_ctx["tau_override"]),
                    int(_req_d_f.data_ptr()) if _req_d_f is not None else 0,
                )
                self._rebuild_cap_lut(self._vw_cap_g_lut, self._graph_vw_conf_captured.keys())
        for i in range(0 if (persistent_ok or cross_step_ok) else k_run):
            if use_fused:
                if use_es:
                    es_ctx = (
                        (i, dspark_k, self._es_bufs, self._vw_c, self._vw_cal, self._vw_step, use_hist, _PRIOR_LOGIT, "dink")
                        if self._vw_is_dink
                        else (i, dspark_k, self._es_bufs, self._vw_c, self._vw_cal, self._vw_step, use_hist, _PRIOR_LOGIT)
                    )
                elif use_es_batch:
                    es_ctx = (i, k_run, self._es_batch_bufs, self._vw_cal, self._es_batch_tau, "batch")
                else:
                    es_ctx = None
                state, prev = self.model.markov_step_and_argmax(
                    state, prev, context[i], base_logits[:, i], es_ctx=es_ctx,
                    out_tok=self.draft_tokens[:num_reqs, i],
                )
            else:
                state, bias = self.model.markov_step(state, prev, context[i])
                if self.draft_logits is None:
                    prev = self.model.markov_argmax(base_logits[:, i], bias)
                else:
                    logits = self.model.markov_logits(base_logits[:, i], bias)
                    prev = self._sample_logits(logits, idx_map[:, i], sample_pos[:, i], i)
                self.draft_tokens[:num_reqs, i] = prev
        if k_run < n_spec and not tail_vw_fused:
            self.draft_tokens[:num_reqs, k_run:n_spec].fill_(-1)
            if self._conf_sink_on:
                c_end = min(n_spec, self._conf_buf.shape[1])
                if k_run < c_end:
                    self._conf_buf[:num_reqs, k_run:c_end].zero_()
        self._conf_end(k_run)
        if fold_pf_forked:
            l2pf.join_end()
        if not _is_cap:
            self._last_sample_seq_eager = True
        if (
            not persistent_did_graph_vw
            and self._sam_hybrid_on
            and self._sam_gpu_ready
            and (num_reqs == 1 or (self._sam_batch_on and num_reqs <= self._sam_max_slots))
            and self._vw_is_conf
            and self._conf_sink_on
            and self._conf_armed
            and self.draft_logits is None
            and k_run > 0
            and _is_cap
        ):
            _wcap_sam_g = int(self._effective_w_cap(num_reqs, None))
            _ke_g = (
                max(0, _wcap_sam_g - dspark_k)
                if (num_reqs == 1 or (self._sam_ext_k15_on and num_reqs <= self._sam_ext_max_b))
                else 0
            )
            self._sam_launch_gpu_boost_extend(
                self.sample_idx_mapping,
                self.temperature,
                num_reqs=num_reqs,
                idx_stride=n_spec if num_reqs > 1 else 1,
                k_active=k_run,
                k_ext_cap=_ke_g,
            )
            self._sam_boost_in_graph_captured.add(int(num_reqs))
            self._rebuild_cap_lut(self._sam_cap_g_lut, self._sam_boost_in_graph_captured)
        if (
            not persistent_did_graph_vw
            and self._graph_vw_conf_on
            and self._is_graph_vw_bucket_uniform(int(num_reqs))
            and getattr(self, "dp_size", 1) == 1
            and self._vw_is_conf
            and self._conf_sink_on
            and self._conf_armed
            and not self._cl_on
            and not self._vw_stats_on
            and self.draft_logits is None
            and k_run > 0
            and _is_cap
        ):
            # Capture _vw_conf_kernel directly at the tail of the Draft FULL CUDA Graph
            # (now works for ALL Dynamic SD k_run in 1..dspark_k, eliminating eager tail gaps at B=2..32!).
            # sample_idx_mapping[r * n_spec] holds req_state_idx (or -1 on capture / padded rows).
            w_cap_g = int(self._effective_w_cap(num_reqs, None))
            w_cap_dspark_g = min(w_cap_g, k_run)
            tau_ov_g = float(self._effective_tau_override(num_reqs))
            allow_ext_g = (
                self._sam_hybrid_on
                and n_spec > dspark_k
                and (num_reqs == 1 or (self._sam_ext_k15_on and num_reqs <= self._sam_ext_max_b and w_cap_g > dspark_k))
            )
            clear_tail_g = (
                num_reqs >= 2 and n_spec > dspark_k and k_run <= dspark_k and not allow_ext_g
            )
            can_req_d = (
                (n_spec == dspark_k)
                or allow_ext_g
                or clear_tail_g
            )
            req_d = self._graph_req_drafts if can_req_d else None
            _g_seq = getattr(self.input_buffers, "seq_lens", None)
            use_wf_g = (
                self._hetero_budget_on
                and self._wf_step_tbl is not None
                and self._vw_is_dink
                and 2 <= num_reqs <= _WF_MAX_B
            )
            if use_wf_g:
                self._vw_launch_hetero_waterfill(
                    self.sample_idx_mapping,
                    self.temperature,
                    self.draft_tokens,
                    w_cap=min(w_cap_dspark_g, dspark_k),
                    req_drafts=req_d,
                    num_reqs=num_reqs,
                    idx_stride=n_spec,
                    clear_req_tail=clear_tail_g,
                    seq_lens=_g_seq,
                )
            else:
                self._vw_launch_conf(
                    self.sample_idx_mapping,
                    self.temperature,
                    self.draft_tokens,
                    w_cap=min(w_cap_dspark_g, dspark_k),
                    tau_override=tau_ov_g,
                    req_drafts=req_d,
                    num_reqs=num_reqs,
                    idx_stride=n_spec,
                    clear_req_tail=clear_tail_g,
                    seq_lens=_g_seq,
                )
            if allow_ext_g:
                _sam_mask_ext_tail_kernel[(num_reqs,)](
                    self.sample_idx_mapping,
                    self._vw_w,
                    self.draft_tokens,
                    self.draft_tokens.stride(0),
                    req_d if req_d is not None else self.draft_tokens,
                    req_d.stride(0) if req_d is not None else self.draft_tokens.stride(0),
                    max(0, w_cap_g - dspark_k),
                    K_DRAFT=dspark_k,
                    K_MAX=n_spec,
                    WRITE_REQ_DRAFTS=(req_d is not None),
                    IDX_STRIDE=n_spec,
                    num_warps=1,
                )
            self._graph_vw_conf_captured[int(num_reqs)] = (
                w_cap_g if allow_ext_g else w_cap_dspark_g,
                -2.0 if use_wf_g else tau_ov_g,
                int(req_d.data_ptr()) if req_d is not None else 0,
            )
            self._rebuild_cap_lut(self._vw_cap_g_lut, self._graph_vw_conf_captured.keys())

    def _sample_tree_beam_graph_safe(
        self,
        *,
        head,
        base_logits: torch.Tensor,  # [1, 7, V_local]
        context: torch.Tensor,      # [7, 1, 3*RANK]
        anchor_tok: torch.Tensor,   # [1] int64
        use_es: bool = False,
        use_hist: bool = False,
    ) -> int:
        """100% CUDA-Graph-Capture-Safe Fast GPU-Native Beam-2 Tree Speculation (M=8, 0 .item()!).
        Uses:
          - Step 0: 1-row top-2 fold (1 fold read, 32 B AG) -> (t1_0, p1_0, t2_0, p2_0)
          - Steps 1..ws_max-1 (empty for ws_max=1; 1 step for ws_max=2!): Tensor-Core 2-row
            `gathered_fold_argmax_fused` (reads fold_w8 ONCE instead of 2x scalar top2!)
          - Steps ws_max..6: 1-row `markov_step_and_argmax(..., es_ctx=es_ctx)` with FULL
            GPU Dinkelbach Early-Stop (`all_gather_stacked_cond`)!
          - 1-CTA GPU Dinkelbach Tree Selector writing `tree_meta[4]` for 4-tile BM=16 attn!"""
        dops = self._dops
        m = self.model
        softcap = m.config.final_logit_softcapping
        valid_v = m.valid_vocab
        v_off = m.vocab_offset
        w1_w = head.markov_w1.weight
        w_rec = head.rnn_w_rec
        rank = int(self._tree_beam_state.shape[1])
        ws_max = self._tree_ws_max

        has_es_dink = bool(use_es and self._vw_is_dink and self._es_bufs is not None and len(self._es_bufs) == 8)
        if has_es_dink:
            es_skip, es_surv, es_tok, es_bestu, es_greedy, _, es_db, es_lam = self._es_bufs
        else:
            es_skip = es_greedy = self._tree_state.n_live
            es_surv = es_tok = es_bestu = es_db = es_lam = self._tree_p2_0

        # Step 0 (1 row): top-2 Markov fold
        state0, lat0 = dops.markov_rnn_step_fused(
            None, anchor_tok, context[0], w1_w, w_rec, init_zero=True
        )
        t1_0, p1_0, t2_0, p2_0 = dops.gathered_fold_argmax_top2_fused(
            base_logits[:, 0], lat0, head.fold_weight, head.fold_bias,
            softcap, valid_v, v_off, fold_w8=head.fold_w8, fold_s8=head.fold_s8,
            id_map=m.draft_id_map,
        )
        _pack_beam2_init_kernel[(1,)](
            state0, t1_0, p1_0, t2_0, p2_0,
            self._tree_beam_state, self._tree_beam_prev,
            self._tree_prim_toks, self._conf_buf,
            self._tree_sec_toks, self._tree_sec_p, self._tree_p2_0,
            es_skip, es_surv, es_tok, es_bestu, es_greedy, es_db, es_lam,
            self._vw_cal, self._vw_step,
            RANK=rank,
            HAS_ES=has_es_dink,
            CAN_SET_SKIP=(ws_max == 1),
            num_warps=4,
        )

        # Steps 1..ws_max-1: batched 2-row Tensor-Core fold (reads fold_w8 ONCE!)
        b_state = self._tree_beam_state
        b_prev = self._tree_beam_prev
        for step in range(1, ws_max):
            b_ctx = context[step].expand(2, -1).contiguous()
            b_base = base_logits[:, step].expand(2, -1)
            b_state, b_lat = dops.markov_rnn_step_fused(b_state, b_prev, b_ctx, w1_w, w_rec)
            bt1 = dops.gathered_fold_argmax_fused(
                b_base, b_lat, head.fold_weight, head.fold_bias,
                softcap, valid_v, v_off,
                fold_w8=head.fold_w8, fold_s8=head.fold_s8,
                conf_out=self._tree_step_bp1,
                out_tok=self._tree_step_bt1,
                id_map=m.draft_id_map,
            )
            _pack_beam2_step_kernel[(1,)](
                bt1, self._tree_step_bp1,
                self._tree_prim_toks, self._conf_buf,
                self._tree_sec_toks, self._tree_sec_p,
                es_skip, es_surv, es_tok, es_bestu, es_greedy, es_db, es_lam,
                self._vw_cal, self._vw_step,
                STEP=step,
                HAS_ES=has_es_dink,
                CAN_SET_SKIP=(step == ws_max - 1),
                num_warps=1,
            )
            b_prev = bt1

        # Steps ws_max..6: 1-row primary continuation with FULL Dinkelbach GPU Early-Stop!
        if ws_max == 1:
            p_state = state0
            p_prev = t1_0
        else:
            p_state = b_state[:1].contiguous()
            p_prev = b_prev[:1]
        if dops._CONF_SINK is not None:
            dops._CONF_SINK.col = ws_max
        for step in range(ws_max, 7):
            es_ctx = (
                (step, 7, self._es_bufs, self._vw_c, self._vw_cal, self._vw_step, use_hist, _PRIOR_LOGIT, "dink")
                if has_es_dink
                else None
            )
            p_state, p_prev = self.model.markov_step_and_argmax(
                p_state, p_prev, context[step], base_logits[:, step],
                es_ctx=es_ctx,
                out_tok=self._tree_prim_toks[step : step + 1],
            )
        self._conf_end(7)

        if self._sam_hybrid_on and self._sam_gpu_ready:
            self.draft_tokens[0, :7].copy_(self._tree_prim_toks[:7])
            self._sam_launch_gpu_boost_extend(self.sample_idx_mapping, self.temperature)
            self._tree_prim_toks[:7].copy_(self.draft_tokens[0, :7])

        ts = self._tree_state
        assert ts is not None
        req_d = self._graph_req_drafts
        write_req = req_d is not None and req_d.shape[1] == 7
        if (
            getattr(self, "_tree_fuse_v10", False)
            and hasattr(dops, "select_last_and_tree_pack_v10")
            and dops.select_last_and_tree_pack_v10(
                self.sample_idx_mapping, self.temperature,
                self._tree_prim_toks, self._conf_buf,
                self._tree_sec_toks, self._tree_sec_p,
                self._vw_cal, self._vw_step,
                self._vw_db, self._vw_etok, self._vw_ems,
                self._vw_w, self._vw_cpred,
                self.draft_tokens, req_d,
                ts.tree_active, ts.n_live, ts.tree_meta,
                ts.tree_parent, ts.tree_depth, ts.tree_sconv_anc, ts.tree_path,
                ws_max=ws_max, sib_disc=self._tree_sib_disc,
                sib_penalty_ms=self._tree_sib_penalty_ms,
                sib_scale=self._tree_sib_scale, p2_min=self._tree_p2_min,
                write_req_drafts=write_req,
            )
        ):
            return int(req_d.data_ptr()) if write_req else 0
        _tree_dink_select_pack_kernel[(1,)](
            self.sample_idx_mapping,
            self.temperature,
            self._tree_prim_toks,
            self._conf_buf,
            self._tree_sec_toks,
            self._tree_sec_p,
            self._vw_cal,
            self._vw_step,
            self._vw_db,
            self._vw_etok,
            self._vw_ems,
            self._vw_w,
            self._vw_cpred,
            self.draft_tokens,
            req_d if write_req else self.draft_tokens,
            req_d.stride(0) if write_req else self.draft_tokens.stride(0),
            ts.tree_active,
            ts.n_live,
            ts.tree_meta,
            ts.tree_parent,
            ts.tree_depth,
            ts.tree_sconv_anc,
            ts.tree_path,
            WS_MAX=ws_max,
            SIB_DISC=self._tree_sib_disc,
            SIB_PENALTY_MS=self._tree_sib_penalty_ms,
            SIB_SCALE=self._tree_sib_scale,
            P2_MIN=self._tree_p2_min,
            WRITE_REQ_DRAFTS=write_req,
            num_warps=1,
        )
        return int(req_d.data_ptr()) if write_req else 0

# ---- INKLING_FUSE_DFLASH_PREP: fuse 5x _prepare_dflash_inputs_kernel -> 1 launch ----
_FUSED_PREP_TLS = threading.local()
_FUSED_PREP_INSTALLED = False


@triton.jit
def _dflash_slot_one_group(
    bt_ptr,
    bt_stride,
    out_ctx_slot_ptr,
    out_q_slot_ptr,
    req_idx,
    ctx_start,
    j,
    is_ctx,
    is_valid_ctx,
    ctx_pos,
    query_idx,
    is_query,
    query_pos,
    max_model_len,
    blk_size,
    blk_stride,
    PAD_SLOT_ID: tl.constexpr,
):
    ctx_bn = tl.minimum(ctx_pos // blk_size, bt_stride - 1)
    ctx_bid = tl.load(bt_ptr + req_idx * bt_stride + ctx_bn, mask=is_valid_ctx, other=0).to(tl.int64)
    ctx_slot = tl.where(
        is_valid_ctx & (ctx_bid != 0),
        ctx_bid * blk_stride + (ctx_pos % blk_size),
        PAD_SLOT_ID,
    )
    tl.store(out_ctx_slot_ptr + ctx_start + j, ctx_slot, mask=is_ctx)

    q_in_bounds = is_query & (query_pos < max_model_len)
    q_bn = tl.minimum(query_pos // blk_size, bt_stride - 1)
    q_bid = tl.load(bt_ptr + req_idx * bt_stride + q_bn, mask=q_in_bounds, other=0).to(tl.int64)
    q_slot = tl.where(
        q_in_bounds & (q_bid != 0),
        q_bid * blk_stride + (query_pos % blk_size),
        PAD_SLOT_ID,
    )
    tl.store(out_q_slot_ptr + query_idx, q_slot, mask=is_query)


@triton.jit(do_not_specialize=["num_reqs", "num_target_tokens", "ctx_pad_end"])
def _prepare_dflash_inputs_5g_kernel(
    # Shared outputs (written ONCE instead of 5x!)
    out_input_ids_ptr,
    out_query_positions_ptr,
    out_is_padding_ptr,
    out_query_start_loc_ptr,
    out_seq_lens_ptr,
    out_context_positions_ptr,
    out_sample_indices_ptr,
    out_sample_pos_ptr,
    out_sample_idx_mapping_ptr,
    out_temperature_ptr,
    out_seeds_ptr,
    out_spec_num_rejected_ptr,
    # Per-group outputs & block tables (5 draft KV cache groups)
    q_slot0, q_slot1, q_slot2, q_slot3, q_slot4,
    c_slot0, c_slot1, c_slot2, c_slot3, c_slot4,
    bt0, bt1, bt2, bt3, bt4,
    bts0, bts1, bts2, bts3, bts4,
    bsz0, bsz1, bsz2, bsz3, bsz4,
    bst0, bst1, bst2, bst3, bst4,
    # Inputs from target batch
    target_positions_ptr,
    target_query_start_loc_ptr,
    idx_mapping_ptr,
    last_sampled_ptr,
    next_prefill_tokens_ptr,
    num_sampled_ptr,
    num_rejected_ptr,
    temperature_ptr,
    seeds_ptr,
    # Scalars
    parallel_drafting_token_id,
    num_query_per_req,
    num_speculative_steps,
    max_num_reqs,
    max_num_tokens,
    max_model_len,
    num_reqs,
    num_target_tokens,
    ctx_pad_end,
    PAD_SLOT_ID: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_PAD_PROGRAMS: tl.constexpr,
    WRITE_SPEC_NUM_REJECTED: tl.constexpr = False,
    FUSE_CTX_SLOT_PAD: tl.constexpr = False,
    NG: tl.constexpr = 5,
):
    req_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    if req_idx >= num_reqs:
        if block_idx == 0:
            pad_idx = req_idx - num_reqs
            start = pad_idx * BLOCK_SIZE
            stride = NUM_PAD_PROGRAMS * BLOCK_SIZE
            last_query_end = num_reqs * num_query_per_req
            for i in range(num_reqs + start, max_num_reqs + 1, stride):
                block = i + tl.arange(0, BLOCK_SIZE)
                tl.store(out_query_start_loc_ptr + block, last_query_end, mask=block < max_num_reqs + 1)
            for i in range(num_reqs + start, max_num_reqs, stride):
                block = i + tl.arange(0, BLOCK_SIZE)
                tl.store(out_seq_lens_ptr + block, 0, mask=block < max_num_reqs)
            pad_end = max_num_reqs * num_speculative_steps
            for i in range(num_reqs * num_speculative_steps + start, pad_end, stride):
                block = i + tl.arange(0, BLOCK_SIZE)
                m = block < pad_end
                tl.store(out_sample_indices_ptr + block, 0, mask=m)
                tl.store(out_sample_pos_ptr + block, 0, mask=m)
                tl.store(out_sample_idx_mapping_ptr + block, -1, mask=m)
            for i in range(last_query_end + start, max_num_tokens, stride):
                block = i + tl.arange(0, BLOCK_SIZE)
                m = block < max_num_tokens
                tl.store(out_input_ids_ptr + block, 0, mask=m)
                tl.store(out_query_positions_ptr + block, 0, mask=m)
                tl.store(out_is_padding_ptr + block, True, mask=m)
                tl.store(q_slot0 + block, PAD_SLOT_ID, mask=m)
                if NG >= 2:
                    tl.store(q_slot1 + block, PAD_SLOT_ID, mask=m)
                if NG >= 3:
                    tl.store(q_slot2 + block, PAD_SLOT_ID, mask=m)
                if NG >= 4:
                    tl.store(q_slot3 + block, PAD_SLOT_ID, mask=m)
                if NG >= 5:
                    tl.store(q_slot4 + block, PAD_SLOT_ID, mask=m)
            if FUSE_CTX_SLOT_PAD:
                for i in range(num_target_tokens + start, ctx_pad_end, stride):
                    block = i + tl.arange(0, BLOCK_SIZE)
                    m = block < ctx_pad_end
                    tl.store(c_slot0 + block, PAD_SLOT_ID, mask=m)
                    if NG >= 2:
                        tl.store(c_slot1 + block, PAD_SLOT_ID, mask=m)
                    if NG >= 3:
                        tl.store(c_slot2 + block, PAD_SLOT_ID, mask=m)
                    if NG >= 4:
                        tl.store(c_slot3 + block, PAD_SLOT_ID, mask=m)
                    if NG >= 5:
                        tl.store(c_slot4 + block, PAD_SLOT_ID, mask=m)
        return

    req_state_idx = tl.load(idx_mapping_ptr + req_idx)
    ctx_start = tl.load(target_query_start_loc_ptr + req_idx)
    ctx_end = tl.load(target_query_start_loc_ptr + req_idx + 1)
    num_ctx = ctx_end - ctx_start
    num_rejected = tl.load(num_rejected_ptr + req_idx)
    valid_ctx_end = ctx_end - num_rejected
    num_valid_ctx = valid_ctx_end - ctx_start
    num_sampled = tl.load(num_sampled_ptr + req_idx)
    if num_sampled > 0:
        bonus_token = tl.load(last_sampled_ptr + req_state_idx).to(tl.int32)
    else:
        bonus_token = tl.load(next_prefill_tokens_ptr + req_state_idx).to(tl.int32)
    last_valid_pos = tl.load(target_positions_ptr + valid_ctx_end - 1)
    query_base = req_idx * num_query_per_req

    j = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    is_ctx = j < num_ctx
    is_valid_ctx = j < num_valid_ctx
    is_query = (j >= num_valid_ctx) & (j < num_valid_ctx + num_query_per_req)
    query_off = j - num_valid_ctx

    ctx_pos_idx = ctx_start + tl.where(is_ctx, j, 0)
    ctx_pos = tl.load(target_positions_ptr + ctx_pos_idx, mask=is_valid_ctx, other=0)
    tl.store(out_context_positions_ptr + ctx_start + j, ctx_pos, mask=is_ctx)

    query_pos = last_valid_pos + 1 + query_off
    query_idx = query_base + query_off
    input_id = tl.where(is_query & (query_off == 0), bonus_token, parallel_drafting_token_id)
    tl.store(out_input_ids_ptr + query_idx, input_id, mask=is_query)
    tl.store(out_is_padding_ptr + query_idx, False, mask=is_query)
    tl.store(out_query_positions_ptr + query_idx, tl.minimum(query_pos, max_model_len - 1), mask=is_query)

    # All NG (1..5) draft KV groups in one pass (compile-time pruned by Triton)!
    _dflash_slot_one_group(bt0, bts0, c_slot0, q_slot0, req_idx, ctx_start, j, is_ctx, is_valid_ctx, ctx_pos, query_idx, is_query, query_pos, max_model_len, bsz0, bst0, PAD_SLOT_ID)
    if NG >= 2:
        _dflash_slot_one_group(bt1, bts1, c_slot1, q_slot1, req_idx, ctx_start, j, is_ctx, is_valid_ctx, ctx_pos, query_idx, is_query, query_pos, max_model_len, bsz1, bst1, PAD_SLOT_ID)
    if NG >= 3:
        _dflash_slot_one_group(bt2, bts2, c_slot2, q_slot2, req_idx, ctx_start, j, is_ctx, is_valid_ctx, ctx_pos, query_idx, is_query, query_pos, max_model_len, bsz2, bst2, PAD_SLOT_ID)
    if NG >= 4:
        _dflash_slot_one_group(bt3, bts3, c_slot3, q_slot3, req_idx, ctx_start, j, is_ctx, is_valid_ctx, ctx_pos, query_idx, is_query, query_pos, max_model_len, bsz3, bst3, PAD_SLOT_ID)
    if NG >= 5:
        _dflash_slot_one_group(bt4, bts4, c_slot4, q_slot4, req_idx, ctx_start, j, is_ctx, is_valid_ctx, ctx_pos, query_idx, is_query, query_pos, max_model_len, bsz4, bst4, PAD_SLOT_ID)

    # sample_from_anchor=False: sample at query_off >= 1
    is_sample = is_query & (query_off >= 1)
    sample_idx = req_idx * num_speculative_steps + (query_off - 1)
    tl.store(out_sample_indices_ptr + sample_idx, query_idx, mask=is_sample)
    tl.store(out_sample_pos_ptr + sample_idx, query_pos, mask=is_sample)
    tl.store(out_sample_idx_mapping_ptr + sample_idx, req_state_idx, mask=is_sample)

    if block_idx == 0:
        tl.store(out_query_start_loc_ptr + req_idx, query_base)
        tl.store(
            out_seq_lens_ptr + req_idx,
            tl.minimum(last_valid_pos + 1 + num_query_per_req, max_model_len),
        )
        tl.store(out_temperature_ptr + req_state_idx, tl.load(temperature_ptr + req_state_idx))
        tl.store(out_seeds_ptr + req_state_idx, tl.load(seeds_ptr + req_state_idx))
        if WRITE_SPEC_NUM_REJECTED:
            tl.store(out_spec_num_rejected_ptr + req_idx, num_rejected)


def _install_fused_dflash_prep() -> None:
    global _FUSED_PREP_INSTALLED
    if _FUSED_PREP_INSTALLED:
        return
    _FUSED_PREP_INSTALLED = True
    try:
        from vllm.v1.attention.backends.utils import PAD_SLOT_ID
        from vllm.v1.worker.gpu.spec_decode.dflash import speculator as df_mod

        orig_prep = df_mod.prepare_dflash_inputs
        num_pad = df_mod._NUM_PAD_PROGRAMS

        def _fused_prep(
            input_buffers, query_slot_mapping, context_positions, context_slot_mapping,
            sample_indices, sample_pos, sample_idx_mapping, temperature, seeds,
            input_batch, num_sampled, num_rejected, last_sampled, next_prefill_tokens,
            input_temperature, input_seeds, block_table, block_size, block_stride,
            cp_rank, cp_size, cp_interleave, parallel_drafting_token_id,
            num_query_per_req, num_speculative_steps, max_num_reqs, max_num_tokens,
            max_model_len, sample_from_anchor=False, num_reqs_override=None,
        ):
            spec = getattr(_FUSED_PREP_TLS, "spec", None)
            if spec is not None and cp_size == 1 and not sample_from_anchor:
                if getattr(_FUSED_PREP_TLS, "done", False):
                    return
                if context_slot_mapping.data_ptr() == spec._context_slot_mappings[0].data_ptr():
                    gids = spec.draft_kv_cache_group_ids
                    ng = len(gids)
                    if not (1 <= ng <= 5):
                        return orig_prep(
                            input_buffers, query_slot_mapping, context_positions, context_slot_mapping,
                            sample_indices, sample_pos, sample_idx_mapping, temperature, seeds,
                            input_batch, num_sampled, num_rejected, last_sampled, next_prefill_tokens,
                            input_temperature, input_seeds, block_table, block_size, block_stride,
                            cp_rank, cp_size, cp_interleave, parallel_drafting_token_id,
                            num_query_per_req, num_speculative_steps, max_num_reqs, max_num_tokens,
                            max_model_len, sample_from_anchor, num_reqs_override,
                        )
                    fast_p = bool(getattr(spec, "_spec_fastpath_on", False))
                    pcache = getattr(spec, "_fused_prep_cache", None) if fast_p else None
                    if (
                        pcache is not None
                        and pcache[0] == ng
                        and pcache[1][0] is spec.block_tables.input_block_tables[gids[0]]
                        and pcache[2][0] is spec.block_tables.slot_mappings[gids[0]]
                        and pcache[3][0].data_ptr() == spec._context_slot_mappings[0].data_ptr()
                    ):
                        _, bts, qs, cs, bt_strides, bszs, bsts = pcache
                    else:
                        bts = [spec.block_tables.input_block_tables[g] for g in gids]
                        qs = [spec.block_tables.slot_mappings[g] for g in gids]
                        cs = [spec._context_slot_mappings[i] for i in range(ng)]
                        bszs = [spec.block_tables.block_sizes[g] for g in gids]
                        bsts = []
                        for g in gids:
                            grp = spec.attn_groups[g][0]
                            if grp.kernel_block_stride:
                                bsts.append(grp.kernel_block_stride * grp.kernel_block_size)
                            else:
                                bsts.append(spec.block_tables.block_sizes[g])
                        # Pad unused slots [ng..4] with slot 0 references (pruned at compile time by NG constexpr).
                        while len(bts) < 5:
                            bts.append(bts[0])
                            qs.append(qs[0])
                            cs.append(cs[0])
                            bszs.append(bszs[0])
                            bsts.append(bsts[0])
                        bt_strides = tuple(int(b.stride(0)) for b in bts)
                        if fast_p:
                            spec._fused_prep_cache = (ng, bts, qs, cs, bt_strides, bszs, bsts)
                    num_reqs = input_batch.num_reqs if num_reqs_override is None else num_reqs_override
                    max_tql = int(input_batch.num_scheduled_tokens[:num_reqs].max())
                    BLOCK_SIZE = 128
                    num_blocks = triton.cdiv(max_tql + num_query_per_req, BLOCK_SIZE)
                    spec_nr = getattr(spec, "_num_rejected", None)
                    write_nr = (
                        spec_nr is not None
                        and isinstance(spec_nr, torch.Tensor)
                        and spec_nr.dtype == num_rejected.dtype
                        and spec_nr.device == num_rejected.device
                        and spec_nr.shape[0] >= num_reqs
                    )
                    nt_tgt = int(input_batch.num_tokens)
                    fuse_ctx_pad = (
                        fast_p
                        and spec._context_slot_mappings.shape[0] == ng
                        and hasattr(spec, "_next_draft_cap_g")
                        and hasattr(spec, "_num_graph_context_tokens")
                    )
                    ctx_pad_end = 0
                    if fuse_ctx_pad:
                        g_cap = spec._next_draft_cap_g(int(num_reqs))
                        if g_cap is not None:
                            nc_est = int(spec._num_graph_context_tokens(g_cap))
                            if nt_tgt < nc_est <= int(cs[0].shape[0]):
                                ctx_pad_end = nc_est
                    _ep_ran = False
                    if (
                        getattr(spec, "_dspark_fuse_epilogue_prep", False)
                        and spec._dops is not None
                        and hasattr(spec._dops, "fused_epilogue_prep_v10")
                    ):
                        _dp = getattr(spec, "_deferred_dink_prep", None)
                        _dink_ctx = None
                        if (
                            _dp is not None
                            and int(input_batch.idx_mapping.data_ptr()) == _dp[5]
                            and int(num_reqs) == _dp[6]
                            and int(input_temperature.data_ptr()) == int(_dp[3].data_ptr())
                        ):
                            _dink_ctx = {
                                "w": spec._vw_w,
                                "cpred": spec._vw_cpred,
                                "db": spec._vw_db,
                                "etok": spec._vw_etok,
                                "ems": spec._vw_ems,
                                "step": spec._vw_step,
                                "stats": spec._vw_stats,
                                "cal": spec._vw_cal,
                                "es_bufs": spec._es_bufs,
                                "eta_b": spec._vw_dink_eta_b,
                                "decay_b": spec._vw_dink_decay_b,
                                "b_clip": spec._vw_dink_b_clip,
                                "eta_lam": spec._vw_dink_eta_lam,
                                "K": spec._dspark_k,
                                "do_stats": spec._vw_stats_on,
                                "fuse_es": bool(_dp[4]),
                            }
                        _ep_ran = spec._dops.fused_epilogue_prep_v10(
                            input_buffers.input_ids, input_buffers.positions, input_buffers.is_padding,
                            input_buffers.query_start_loc, input_buffers.seq_lens, context_positions,
                            sample_indices, sample_pos, sample_idx_mapping,
                            temperature, seeds, spec_nr if write_nr else num_rejected,
                            qs, cs, bts, bt_strides, bszs, bsts,
                            input_batch.positions, input_batch.query_start_loc, input_batch.idx_mapping,
                            last_sampled, next_prefill_tokens, num_sampled, num_rejected,
                            input_temperature, input_seeds,
                            parallel_drafting_token_id=parallel_drafting_token_id,
                            num_query_per_req=num_query_per_req,
                            num_speculative_steps=num_speculative_steps,
                            max_num_reqs=max_num_reqs, max_num_tokens=max_num_tokens,
                            max_model_len=max_model_len, num_reqs=num_reqs,
                            num_pad_programs=num_pad, num_blocks=num_blocks,
                            pad_slot_id=PAD_SLOT_ID, write_spec_num_rejected=write_nr,
                            pad_c_start=nt_tgt if (fuse_ctx_pad and ctx_pad_end > nt_tgt) else 0,
                            pad_c_end=ctx_pad_end if (fuse_ctx_pad and ctx_pad_end > nt_tgt) else 0,
                            ng=ng, dink_ctx=_dink_ctx,
                        )
                        if _ep_ran and _dink_ctx is not None:
                            spec._deferred_dink_prep = None
                    if not _ep_ran:
                        _dp_fb = getattr(spec, "_deferred_dink_prep", None)
                        if _dp_fb is not None:
                            spec._deferred_dink_prep = None
                            spec._vw_launch_update_dink(_dp_fb[0], _dp_fb[1], _dp_fb[2], _dp_fb[3], fuse_es=_dp_fb[4])
                        _prepare_dflash_inputs_5g_kernel[(num_reqs + num_pad, num_blocks)](
                            input_buffers.input_ids,
                        input_buffers.positions,
                        input_buffers.is_padding,
                        input_buffers.query_start_loc,
                        input_buffers.seq_lens,
                        context_positions,
                        sample_indices,
                        sample_pos,
                        sample_idx_mapping,
                        temperature,
                        seeds,
                        spec_nr if write_nr else num_rejected,
                        qs[0], qs[1], qs[2], qs[3], qs[4],
                        cs[0], cs[1], cs[2], cs[3], cs[4],
                        bts[0], bts[1], bts[2], bts[3], bts[4],
                        bt_strides[0], bt_strides[1], bt_strides[2], bt_strides[3], bt_strides[4],
                        bszs[0], bszs[1], bszs[2], bszs[3], bszs[4],
                        bsts[0], bsts[1], bsts[2], bsts[3], bsts[4],
                        input_batch.positions,
                        input_batch.query_start_loc,
                        input_batch.idx_mapping,
                        last_sampled,
                        next_prefill_tokens,
                        num_sampled,
                        num_rejected,
                        input_temperature,
                        input_seeds,
                        parallel_drafting_token_id,
                        num_query_per_req,
                        num_speculative_steps,
                        max_num_reqs,
                        max_num_tokens,
                        max_model_len,
                        num_reqs,
                        nt_tgt,
                        ctx_pad_end,
                        PAD_SLOT_ID=PAD_SLOT_ID,
                        BLOCK_SIZE=BLOCK_SIZE,
                        NUM_PAD_PROGRAMS=num_pad,
                        WRITE_SPEC_NUM_REJECTED=write_nr,
                        FUSE_CTX_SLOT_PAD=bool(fuse_ctx_pad),
                        NG=ng,
                    )
                    _FUSED_PREP_TLS.done = True
                    _FUSED_PREP_TLS.num_rejected_written = write_nr
                    _FUSED_PREP_TLS.pad_slot_id = PAD_SLOT_ID
                    _FUSED_PREP_TLS.ctx_pad_range = (nt_tgt, ctx_pad_end) if ctx_pad_end > nt_tgt else None
                    return
            return orig_prep(
                input_buffers, query_slot_mapping, context_positions, context_slot_mapping,
                sample_indices, sample_pos, sample_idx_mapping, temperature, seeds,
                input_batch, num_sampled, num_rejected, last_sampled, next_prefill_tokens,
                input_temperature, input_seeds, block_table, block_size, block_stride,
                cp_rank, cp_size, cp_interleave, parallel_drafting_token_id,
                num_query_per_req, num_speculative_steps, max_num_reqs, max_num_tokens,
                max_model_len, sample_from_anchor, num_reqs_override,
            )

        df_mod.prepare_dflash_inputs = _fused_prep
        logger.info("INKLING_FUSE_DFLASH_PREP=1: 1..5-group fused _prepare_dflash_inputs_5g_kernel(NG) installed")
    except Exception as e:  # noqa: BLE001
        logger.warning("INKLING_FUSE_DFLASH_PREP install failed (%r); using stock 5-launch loop", e)
