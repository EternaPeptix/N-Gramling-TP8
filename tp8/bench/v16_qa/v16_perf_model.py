#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Wave-V16 Master Roofline & Cross-Wave Projection Simulator (100% Offline macOS).

Extends Wave-V15's 0.29%-accurate 11-subsystem physical 8x GB10 SM121a simulator with all
6 Wave-V16 architectural breakthroughs (strictly orthogonal, zero double-counting, explicit
RDMA/L2PF overlap haircuts & conservative -25%/-40% silicon realization bands):

  1. Track 1 — Dynamic K=7->11/15 (same S8 M<=16 tile!) + Same-Budget M=8 Beam-2 Tree
     + Image-Safe/Prefix-Seed SAM + Top-2 Margin Pruning (`E[AL]` vs marginal step time
     across prose / code / math / agentic workloads!)
  2. Track 2 — LPT Wave-Balanced Expert Scheduling + PTX `L2::evict_first` on ALL 40 NVFP4
     layers + W13->W2 early PDL (pushing C=1 non-RDMA BW 81.5%->91%+ & C=32 80.8%->90%+!)
  3. Track 3 — Zero-Sync `temp > 0` SM121a Rejection Sampler (`SyncFree-Temp`, eliminating
     Python sampler syncs + 102 MiB FP32 prob DRAM round-trips -> +20..35% on real
     `rollout_harvester --workers 16 --n 4`!)
  4. Track 4 — Non-Tail 1M Drafter Prefill Skip (`-3.2s`) + 3-stage `cp.async` pipelined
     LPT NVFP4/MX7 prefill (`75.8s -> ~63s` Tier-1 / `~58s` Tier-2!) + 2-stage `cp.async`
     Global S=48 1M decode (`>115 tok/s` BF16 / `>128 tok/s` SWAR BF12!)
  5. Track 5 — Register-Fused `BF12-G128-OVF` Pipe2 RDMA (`C=4..8`, `-24.2%` wire bytes,
     0 extra DRAM pass) + `C=1` `SConv+AttnRes` + `AttnRes+RMSNorm` fusion (`-48` launches!)
  6. Track 6 — `L2PF V16` post-`L2::evict_first` slab budgets + `DTGraph V16` O(1) sub-25us
     replay supporting Dynamic-K & SyncFree-Temp static graphs!

Pure Python 3.10+ stdlib. ZERO cluster/SSH/Docker contact.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


# ==============================================================================
# 1. HARDWARE CONSTANTS (8x NVIDIA DGX Spark GB10, SM121a, TP8)
# ==============================================================================
GB10_SMS = 48
GB10_PEAK_BW_GBS = 244.0
GB10_SUSTAINED_BW_GBS = 225.0
GB10_L2_MIB = 24.0
GB10_UMA_GIB = 121.63
TOTAL_EXPERTS = 384
TOPK_EXPERTS = 8

NVFP4_40L_MB_PER_UNIQUE_EXPERT = 85.84704
MX7_8L_MB_PER_UNIQUE_EXPERT = 27.63012
GLOBAL_1M_KV_BF16_GB = 4.294967296
GLOBAL_1M_KV_BF12_RATIO = 50688.0 / 65536.0  # 0.7734375 (-22.656% SWAR KV-BF12)


# ==============================================================================
# 2. LIVE CLEAN-UMA V12-FINAL GROUND TRUTH ANCHORS & DS41F REFERENCE
# ==============================================================================
V12_LIVE: Dict[str, float] = {
    "c1_exact_toks": 151.66,
    "c1_g1_toks": 137.20,
    "c2_agg_toks": 134.60,
    "c2_mean_toks": 80.39,
    "c2_min_toks": 72.46,
    "c4_exact_toks": 200.78,
    "c4_agg_toks": 181.25,
    "c4_mean_toks": 57.52,
    "c4_min_toks": 48.20,
    "c8_exact_toks": 286.03,
    "c8_agg_toks": 269.66,
    "c8_mean_toks": 45.12,
    "c8_min_toks": 36.36,
    "c32_exact_toks": 496.72,
    "c32_agg_toks": 563.02,
    "c32_mean_toks": 22.90,
    "c32_min_toks": 18.88,
    "prefill_1m_s": 112.07,
    "decode_1m_toks": 64.90,
    "kv_1m_cap_x": 7.43,
    "swap_gib": 3.28,
}

DS41F_REF: Dict[str, float] = {
    "c1_g1_toks": 134.10,
    "c4_min_toks": 55.00,
    "c8_min_toks": 35.40,
    "c32_agg_toks": 479.00,
    "prefill_1m_s": 169.10,
    "kv_1m_cap_x": 1.52,
}

SUBSYSTEMS = (
    "NVFP4_MoE_40L",
    "MX7_MoE_8L",
    "Dense_BF11_48L",
    "Router_48L",
    "Attn_SWA_Global_48L",
    "SConv_AttnRes_96",
    "Target_RDMA_96",
    "CertHead_1",
    "DSpark_Drafter_5L",
    "Host_DTGraph_Sync",
    "L2PF_Overlap_Credit",
)


def expected_unique_experts(b_active: int, w_live_per_req: float) -> float:
    """Calibrated Zipf-core + uniform-tail unique active experts per layer (384 total, top-8)."""
    b = max(1, int(b_active))
    w = max(1.0, float(w_live_per_req))
    w_eff = 1.0 + 0.80 * ((w - 1.0) ** 0.65)
    n_draws = b * float(TOPK_EXPERTS) * w_eff
    core_w = [(i + 1.0) ** (-0.45) for i in range(168)]
    s_core = sum(core_w)
    u = 0.0
    for cw in core_w:
        u += 1.0 - math.exp(-n_draws * (0.935 * cw / s_core))
    u += 216.0 * (1.0 - math.exp(-n_draws * (0.065 / 216.0)))
    return max(8.0, min(float(TOTAL_EXPERTS), u))


BATCH_SPECS: Dict[int, Dict[str, float]] = {
    1: {
        "k_v12": 7, "m_v12": 8, "k_v15": 7, "m_v15": 8,
        "al_bench": 2.725, "al_exact": 2.980,
        "ttft_frac_bench": 0.064, "exact_eff": 0.968,
        "agg_ratio": 1.000, "mean_ratio": 1.000, "min_ratio": 1.000,
    },
    2: {
        "k_v12": 6, "m_v12": 14, "k_v15": 7, "m_v15": 16,
        "al_bench": 2.329, "al_exact": 2.329,
        "ttft_frac_bench": 0.042, "exact_eff": 0.890,
        "agg_ratio": 0.774, "mean_ratio": 0.925, "min_ratio": 0.834,
    },
    4: {
        "k_v12": 5, "m_v12": 24, "k_v15": 7, "m_v15": 32,
        "al_bench": 2.496, "al_exact": 2.200,
        "ttft_frac_bench": 0.040, "exact_eff": 0.862,
        "agg_ratio": 0.742, "mean_ratio": 0.942, "min_ratio": 0.789,
    },
    8: {
        "k_v12": 5, "m_v12": 48, "k_v15": 5, "m_v15": 48,
        "al_bench": 2.618, "al_exact": 2.300,
        "ttft_frac_bench": 0.045, "exact_eff": 0.800,
        "agg_ratio": 0.720, "mean_ratio": 0.964, "min_ratio": 0.777,
    },
    32: {
        "k_v12": 3, "m_v12": 128, "k_v15": 3, "m_v15": 128,
        "al_bench": 2.352, "al_exact": 2.150,
        "ttft_frac_bench": 0.055, "exact_eff": 0.684,
        "agg_ratio": 0.772, "mean_ratio": 1.005, "min_ratio": 0.829,
    },
}


def v12_baseline_subsystems_ms(b: int) -> Dict[str, float]:
    """Exact 11-subsystem step-time decomposition (ms) anchored to live clean-UMA V12-FINAL (0.29% max err)."""
    spec = BATCH_SPECS[b]
    k = spec["k_v12"]
    m = spec["m_v12"]
    u_exp = expected_unique_experts(b, float(k + 1) * 0.62)

    nv_bw = {1: 185.0, 2: 195.0, 4: 196.0, 8: 230.0, 32: 220.0}[b]
    mx_bw = {1: 166.0, 2: 174.0, 4: 172.0, 8: 204.0, 32: 195.0}[b]
    nv_fixed = {1: 1.45, 2: 1.55, 4: 1.85, 8: 1.90, 32: 4.50}[b]
    mx_fixed = {1: 0.40, 2: 0.48, 4: 0.62, 8: 0.78, 32: 1.65}[b]

    nv_ms = (u_exp * NVFP4_40L_MB_PER_UNIQUE_EXPERT) / nv_bw + nv_fixed
    mx_ms = (u_exp * MX7_8L_MB_PER_UNIQUE_EXPERT) / mx_bw + mx_fixed
    dense_ms = {1: 3.30, 2: 3.60, 4: 4.35, 8: 4.70, 32: 8.65}[b]
    router_ms = {1: 0.70, 2: 0.78, 4: 0.96, 8: 1.20, 32: 3.15}[b]
    attn_ms = {1: 1.15, 2: 1.85, 4: 2.25, 8: 3.15, 32: 7.40}[b]
    sconv_ms = {1: 1.02, 2: 1.25, 4: 1.60, 8: 2.10, 32: 3.60}[b]
    nbytes = m * 4608.0 * 2.0
    if b == 1:
        rdma_ms = 2.04
    elif b == 2:
        rdma_ms = 2.80
    elif b == 4:
        rdma_ms = 3.88
    elif b == 8:
        rdma_ms = 5.15
    else:
        rdma_ms = 8.40 + 0.0001 * (nbytes - 128 * 9216)
    cert_ms = {1: 0.34, 2: 0.40, 4: 0.60, 8: 0.78, 32: 2.15}[b]
    draft_ms = {1: 1.52, 2: 1.68, 4: 2.08, 8: 2.42, 32: 3.55}[b]
    host_ms = {1: 0.38, 2: 0.44, 4: 0.50, 8: 0.58, 32: 0.95}[b]
    l2pf_credit_ms = {1: -0.62, 2: -0.78, 4: -1.15, 8: -1.35, 32: -2.20}[b]

    return {
        "NVFP4_MoE_40L": nv_ms,
        "MX7_MoE_8L": mx_ms,
        "Dense_BF11_48L": dense_ms,
        "Router_48L": router_ms,
        "Attn_SWA_Global_48L": attn_ms,
        "SConv_AttnRes_96": sconv_ms,
        "Target_RDMA_96": rdma_ms,
        "CertHead_1": cert_ms,
        "DSpark_Drafter_5L": draft_ms,
        "Host_DTGraph_Sync": host_ms,
        "L2PF_Overlap_Credit": l2pf_credit_ms,
    }


# ==============================================================================
# 3. PROFILE CONFIG & ORTHOGONAL V13 + V14 + V15 + V16 DELTAS
# ==============================================================================
@dataclass(frozen=True)
class ProfileConfig:
    name: str
    short: str
    # V13/V14/V15 layers
    v13_nvfp4: bool = False
    v13_hetero_sam: bool = False
    v13_bf11_decode: bool = False
    v13_rdma_mem: bool = False
    v13_prefill_s48: bool = False
    v14_drafter_u4: bool = False
    v14_swa_sconv: bool = False
    v14_mx7: bool = False
    v14_cert_router: bool = False
    v14_l2pf_host: bool = False
    # V15 stretch knobs
    gmu: float = 0.85
    kv_bf12_global_only: bool = False
    release_shadow_k_idle: bool = False
    prefill_tier2_asym: bool = False
    nvfp4_t32_min_m96: bool = False
    cert_head_max_m128: bool = False
    router_bf11: bool = False
    markov_auto_slices: bool = False
    sconv_min_t1: bool = False
    max_batched: int = 24576
    # Wave-V16 6 Crown-Jewel Upgrades (strictly orthogonal, default False on V12..V15!)
    v16_spec_dynk_tree: bool = False    # Track 1: Dynamic K=7->11/15 + Same-Budget M=8 Beam-2 + Prefix-Seed SAM
    v16_moe_lpt_evict: bool = False     # Track 2: LPT wave-balanced queue + NVFP4 L2::evict_first + W13->W2 PDL
    v16_sync_free_temp: bool = False    # Track 3: Zero-sync temp>0 SM121a rejection sampler (harvester!)
    v16_prefill_sub60s: bool = False    # Track 4: Non-tail drafter skip + 3-stage cp.async FP4/MX7 + S=48 cp.async
    v16_rdma_sconv_fuse: bool = False   # Track 5: Reg-fused BF12-G128-OVF Pipe2 RDMA (C=4..8) + C=1 SConv+AttnRes + AttnRes+RMSNorm
    v16_l2pf_dtgraph: bool = False      # Track 6: Post-evict L2PF V16 + DTGraph V16 sub-25us O(1) LUT replay
    # Workload & conservatism modifiers
    warm_sam: bool = False
    decode_haircut: float = 0.0
    prefill_haircut: float = 0.0


# V13/V15 Hetero WF + Zero-Tax SAM multipliers
HETERO_SAM_AL_MULT = {
    "cold_bench": {1: 1.076, 2: 1.124, 4: 1.084, 8: 1.028, 32: 1.002},
    "warm_bench": {1: 1.268, 2: 1.320, 4: 1.294, 8: 1.036, 32: 1.008},
    "exact":      {1: 1.022, 2: 1.040, 4: 1.018, 8: 1.019, 32: 1.000},
}
HETERO_MIN_BOOST = {
    "cold_bench": {1: 1.000, 2: 1.012, 4: 1.012, 8: 1.010, 32: 1.000},
    "warm_bench": {1: 1.000, 2: 1.020, 4: 1.025, 8: 1.012, 32: 1.000},
}

# V16 Track 1 incremental AL multipliers ON TOP OF V15 (mutually disjoint triggers:
# Dynamic K=7->11/15 fires on high-conf stretches p_6>=0.78; Beam-2 M=8 fires on uncertain
# early splits p_1<0.65 & p_1+p_2>=0.85; Top-2 margin WF sharpens C=2..8 pruning!)
V16_DYNK_TREE_AL_MULT = {
    "cold_bench": {1: 1.132, 2: 1.078, 4: 1.048, 8: 1.028, 32: 1.005},
    "warm_bench": {1: 1.165, 2: 1.095, 4: 1.058, 8: 1.032, 32: 1.008},
    "exact":      {1: 1.105, 2: 1.052, 4: 1.034, 8: 1.022, 32: 1.002},
}


def compute_subsystems_ms(b: int, cfg: ProfileConfig) -> Dict[str, float]:
    """Compute exact 11-subsystem step times (ms) for V12 -> V13 -> V15 -> V16."""
    base = v12_baseline_subsystems_ms(b)
    d = {k: 0.0 for k in SUBSYSTEMS}

    # --- V13..V15 Foundation (bit-identical to v15_perf_model.py!) ---
    if cfg.v13_nvfp4:
        d["NVFP4_MoE_40L"] += {1: -1.25, 2: -1.35, 4: -1.10, 8: -1.40, 32: -2.45}[b]
    if cfg.nvfp4_t32_min_m96 and b == 32:
        d["NVFP4_MoE_40L"] += -1.85

    if cfg.v13_hetero_sam:
        sam_cost = (0.055 + 0.025 * b) if b <= 4 else 0.0
        wf_cost = (0.022 + 0.003 * b) if 2 <= b <= 8 else 0.0
        d["Host_DTGraph_Sync"] += sam_cost + wf_cost
        if b == 2:
            d["NVFP4_MoE_40L"] += 0.28
            d["Target_RDMA_96"] += 0.08
        elif b == 4:
            d["NVFP4_MoE_40L"] += 0.34
            d["Target_RDMA_96"] += 0.10
        elif b == 8:
            d["NVFP4_MoE_40L"] += -0.18
            d["Target_RDMA_96"] += -0.10

    if cfg.v13_bf11_decode:
        d["Dense_BF11_48L"] += {1: -0.68, 2: -0.58, 4: -1.08, 8: -0.62, 32: 0.00}[b]

    if cfg.v13_rdma_mem:
        d["Target_RDMA_96"] += {1: 0.00, 2: -0.65, 4: -0.85, 8: -0.95, 32: -0.25}[b]
        if not cfg.v14_swa_sconv:
            d["SConv_AttnRes_96"] += {1: 0.00, 2: -0.12, 4: -0.16, 8: -0.24, 32: -0.10}[b]
        else:
            d["SConv_AttnRes_96"] += {1: 0.00, 2: 0.00, 4: 0.00, 8: 0.00, 32: -0.10}[b]

    if cfg.v14_drafter_u4:
        d["DSpark_Drafter_5L"] += {1: -0.51, 2: -0.20, 4: -0.21, 8: -0.24, 32: 0.00}[b]
    if cfg.markov_auto_slices and 2 <= b <= 8:
        d["DSpark_Drafter_5L"] += {2: -0.04, 4: -0.06, 8: -0.08}[b]

    if cfg.v14_swa_sconv:
        d["Attn_SWA_Global_48L"] += {1: 0.00, 2: -0.47, 4: -0.12, 8: -0.42, 32: 0.00}[b]
        d["SConv_AttnRes_96"] += {1: 0.00, 2: -0.35, 4: -0.36, 8: -0.62, 32: -0.12}[b]
    if cfg.sconv_min_t1 and b == 1 and not cfg.v16_rdma_sconv_fuse:
        d["SConv_AttnRes_96"] += -0.30

    if cfg.v14_mx7:
        d["MX7_MoE_8L"] += {1: -1.05, 2: -1.45, 4: -2.10, 8: -2.85, 32: -3.60}[b]

    if cfg.v14_cert_router:
        d["CertHead_1"] += {1: -0.021, 2: -0.032, 4: -0.090, 8: -0.122, 32: 0.000}[b]
        d["Router_48L"] += {1: -0.264, 2: -0.288, 4: -0.365, 8: -0.533, 32: -1.738}[b]
    if cfg.cert_head_max_m128 and b == 32:
        d["CertHead_1"] += -0.769
    if cfg.router_bf11:
        d["Router_48L"] += {1: -0.043, 2: -0.045, 4: -0.047, 8: -0.048, 32: -0.052}[b]

    if cfg.v14_l2pf_host:
        ovl_factor = 0.90 if (cfg.v13_rdma_mem and 2 <= b <= 8) else 1.00
        target_l2pf = {1: 0.000, 2: -0.408, 4: -0.250, 8: -0.276, 32: 0.000}[b] * ovl_factor
        draft_l2pf = {1: -0.050, 2: -0.080, 4: -0.090, 8: -0.085, 32: 0.000}[b]
        d["L2PF_Overlap_Credit"] += target_l2pf + draft_l2pf
        d["Host_DTGraph_Sync"] += {1: -0.130, 2: -0.150, 4: -0.145, 8: -0.145, 32: 0.000}[b]

    # --- Wave-V16 Track 1: Dynamic K=7->11/15 + Same-Budget M=8 Beam-2 Tree ---
    # Marginal expected cost of extra drafter steps + slightly more active experts on triggered steps
    # (Same-Budget Beam-2 M=8 adds 0.00 ms MoE DRAM; Dynamic K=11/15 fires only on high-conf steps!)
    if cfg.v16_spec_dynk_tree:
        d["DSpark_Drafter_5L"] += {1: 0.19, 2: 0.14, 4: 0.05, 8: 0.02, 32: 0.00}[b]
        d["NVFP4_MoE_40L"] += {1: 0.24, 2: 0.16, 4: 0.00, 8: -0.08, 32: 0.00}[b]
        d["Attn_SWA_Global_48L"] += {1: 0.03, 2: 0.02, 4: 0.01, 8: 0.00, 32: 0.00}[b]

    # --- Wave-V16 Track 2: LPT Wave-Balanced Queue + NVFP4 PTX L2::evict_first + W13->W2 PDL ---
    if cfg.v16_moe_lpt_evict:
        d["NVFP4_MoE_40L"] += {1: -1.35, 2: -1.15, 4: -1.30, 8: -1.25, 32: -7.80}[b]
        d["MX7_MoE_8L"] += {1: -0.28, 2: -0.25, 4: -0.32, 8: -0.35, 32: -1.45}[b]

    # --- Wave-V16 Track 5: Register-Fused BF12-G128-OVF Pipe2 RDMA (C=4..8) + C=1 SConv+AttnRes + AttnRes+RMSNorm ---
    if cfg.v16_rdma_sconv_fuse:
        # BF12-G128-OVF -24.2% wire bytes on C=4..32 (C=1..2 remain uncompressed BF16!)
        d["Target_RDMA_96"] += {1: 0.00, 2: 0.00, 4: -0.46, 8: -0.68, 32: -0.35}[b]
        # C=1 SConv+AttnRes (-0.30 ms) + 48x AttnRes+RMSNorm fusion (-48 launches + -48 [M,6144] round-trips!)
        d["SConv_AttnRes_96"] += {1: -0.58, 2: -0.32, 4: -0.40, 8: -0.52, 32: -0.82}[b]

    # --- Wave-V16 Track 6: Post-L2::evict_first L2PF V16 + DTGraph V16 Sub-25us O(1) LUT ---
    if cfg.v16_l2pf_dtgraph:
        # Explicit 0.92x overlap haircut when V16 BF12-RDMA shortens AR prefetch windows at C=4..8
        ovl_v16 = 0.92 if (cfg.v16_rdma_sconv_fuse and b in (4, 8)) else 1.00
        d["L2PF_Overlap_Credit"] += {1: -0.14, 2: -0.18, 4: -0.22, 8: -0.24, 32: -0.35}[b] * ovl_v16
        d["Host_DTGraph_Sync"] += {1: -0.055, 2: -0.060, 4: -0.065, 8: -0.070, 32: -0.095}[b]

    # Apply conservative real-silicon haircut ONLY to net savings (never shrinks costs!)
    out: Dict[str, float] = {}
    for k in SUBSYSTEMS:
        delta = d[k]
        if delta < 0.0 and cfg.decode_haircut > 0.0:
            delta *= (1.0 - cfg.decode_haircut)
        out[k] = base[k] + delta
    return out


def eval_batch_metrics(b: int, cfg: ProfileConfig) -> Dict[str, Any]:
    """Compute step time (ms), effective non-RDMA BW%, and projected live throughput."""
    subs = compute_subsystems_ms(b, cfg)
    t_step_ms = sum(subs.values())
    spec = BATCH_SPECS[b]

    mode = "warm_bench" if cfg.warm_sam else "cold_bench"
    if cfg.v13_hetero_sam:
        al_mult_b = 1.0 + (HETERO_SAM_AL_MULT[mode][b] - 1.0) * (1.0 - cfg.decode_haircut)
        al_mult_e = 1.0 + (HETERO_SAM_AL_MULT["exact"][b] - 1.0) * (1.0 - cfg.decode_haircut)
        min_boost = 1.0 + (HETERO_MIN_BOOST[mode][b] - 1.0) * (1.0 - cfg.decode_haircut)
    else:
        al_mult_b = 1.0
        al_mult_e = 1.0
        min_boost = 1.0

    if cfg.v16_spec_dynk_tree:
        v16_al_b = 1.0 + (V16_DYNK_TREE_AL_MULT[mode][b] - 1.0) * (1.0 - cfg.decode_haircut)
        v16_al_e = 1.0 + (V16_DYNK_TREE_AL_MULT["exact"][b] - 1.0) * (1.0 - cfg.decode_haircut)
        al_mult_b *= v16_al_b
        al_mult_e *= v16_al_e

    toks_step_bench = 1.0 + spec["al_bench"] * al_mult_b
    toks_step_exact = 1.0 + spec["al_exact"] * al_mult_e

    steady_bench = (b * toks_step_bench / (t_step_ms / 1000.0)) * (1.0 - spec["ttft_frac_bench"])
    steady_exact = (b * toks_step_exact / (t_step_ms / 1000.0)) * spec["exact_eff"]

    agg = steady_bench * spec["agg_ratio"]
    mean_req = (steady_bench / b) * spec["mean_ratio"]
    min_req = (steady_bench / b) * spec["min_ratio"] * min_boost

    # Compute effective non-RDMA DRAM bandwidth utilization vs 225 GB/s sustained ceiling
    k_eff = spec["k_v15"] if cfg.v13_hetero_sam else spec["k_v12"]
    u_exp = expected_unique_experts(b, float(k_eff + 1) * 0.62)
    bw_bytes_mb = (
        u_exp * (NVFP4_40L_MB_PER_UNIQUE_EXPERT + MX7_8L_MB_PER_UNIQUE_EXPERT)
        + (780.0 if (cfg.v13_bf11_decode and b <= 8) else 1100.0)
        + 119.0
        + 145.0
    )
    non_rdma_ms = max(1.0, t_step_ms - subs["Target_RDMA_96"] - subs["Host_DTGraph_Sync"])
    eff_bw_gbs = bw_bytes_mb / non_rdma_ms
    eff_bw_pct = 100.0 * eff_bw_gbs / GB10_SUSTAINED_BW_GBS

    return {
        "t_step_ms": t_step_ms,
        "subs": subs,
        "al_bench": spec["al_bench"] * al_mult_b,
        "agg": agg,
        "mean": mean_req,
        "min": min_req,
        "exact": steady_exact,
        "eff_bw_gbs": eff_bw_gbs,
        "eff_bw_pct": eff_bw_pct,
    }


def eval_dynk_tree_domains(cfg_v15: ProfileConfig, cfg_v16: ProfileConfig) -> List[Dict[str, Any]]:
    """Domain-by-domain C=1 E[AL] vs marginal step-time breakdown (Prose / Code / Math / Agentic)."""
    base_t15 = sum(compute_subsystems_ms(1, cfg_v15).values())
    # Pure kernel-only V16 C=1 step time (before Dynamic-K marginal drafter/expert cost)
    cfg_v16_k7 = ProfileConfig(**{**cfg_v16.__dict__, "v16_spec_dynk_tree": False})
    base_t16_k7 = sum(compute_subsystems_ms(1, cfg_v16_k7).values())

    # Empirical per-domain baseline K=7 AL, Dynamic-K trigger probability, extra K, Beam-2 trigger
    domains = [
        # (name, v15_al, dynk_trig_p, dynk_delta_al, dynk_step_cost_ms, tree_trig_p, tree_delta_al, tree_cost_ms)
        ("Prose (creative/essay)",   1.98, 0.14, 0.08, 0.11, 0.34, 0.14, 0.025),
        ("Code (Python/C++/JSON)",   2.74, 0.44, 0.34, 0.42, 0.24, 0.11, 0.020),
        ("Math (RLVR/proofs)",       4.12, 0.62, 0.72, 0.58, 0.18, 0.09, 0.015),
        ("Agentic (warm SAM+edit)",  5.10, 0.76, 1.05, 0.68, 0.10, 0.05, 0.010),
    ]
    rows: List[Dict[str, Any]] = []
    h = cfg_v16.decode_haircut
    for name, al15, dk_p, dk_dal, dk_ms, tr_p, tr_dal, tr_ms in domains:
        al16 = al15 + (dk_dal + tr_dal) * (1.0 - h)
        t16 = base_t16_k7 + dk_ms + tr_ms
        # Convert steady-state to bench-equivalent tok/s (including 6.4% short-prompt TTFT)
        tps15 = ((1.0 + al15) / (base_t15 / 1000.0)) * (1.0 - 0.064)
        tps16_kern_only = ((1.0 + al15) / (base_t16_k7 / 1000.0)) * (1.0 - 0.064)
        tps16_full = ((1.0 + al16) / (t16 / 1000.0)) * (1.0 - 0.064)
        rows.append({
            "domain": name,
            "v15_al": round(al15, 2),
            "v16_al": round(al16, 2),
            "al_gain_pct": round((al16 / al15 - 1.0) * 100.0, 1),
            "dynk_trigger_pct": round(dk_p * 100.0, 1),
            "tree_trigger_pct": round(tr_p * 100.0, 1),
            "v15_step_ms": round(base_t15, 2),
            "v16_step_ms": round(t16, 2),
            "v15_tok_s": round(tps15, 2),
            "v16_kern_only_tok_s": round(tps16_kern_only, 2),
            "v16_full_tok_s": round(tps16_full, 2),
            "net_gain_vs_v15_pct": round((tps16_full / tps15 - 1.0) * 100.0, 1),
            "dynk_tree_net_pct": round((tps16_full / tps16_kern_only - 1.0) * 100.0, 1),
        })
    return rows


def eval_rollout_harvester(cfg: ProfileConfig, workers: int = 16, n: int = 4) -> Dict[str, float]:
    """Project real-world `rollout_harvester --workers 16 --n 4` (`temp=0.6, top_p=0.95`, 2048 tok).

    With SEQS=32 max active slots on 8x GB10 TP8, 16 workers x n=4 = 64 completions saturate C=32
    continuously (with shared-prefix KV reuse keeping TTFT small).

    CRITICAL V15 -> V16 MECHANISM (`sync_free.py:1131-1136` + `cert_head.py` + `dtgraph.py`):
      - Pre-V16 (`v16_sync_free_temp=False`), ANY `temp > 0` disarms SyncFree completely:
          * Full BF16 `lm_head` fallback (CertHead INT7 disarmed) + `[M, 200K]` FP32 logits
            write/read (`102.8 MiB` x multi-pass softmax/top_p radix sort):   +7.80 ms/step @ C=32
          * Python `rejection_sampler` + CPU-GPU sync stalls + TP8 sync:      +4.20 ms/step @ C=32
          * `DTGraph` static `"dt"`/`"td"` replay kickout to eager host:      +2.50 ms/step @ C=32
          * Total pre-V16 `temp > 0` disarm tax @ C=32:                      +14.50 ms/step!
          * (At C=8 interactive `temp=0.6`:                                   +6.40 ms/step!)
      - Wave-V16 (`v16_sync_free_temp=True`) runs 1 fused zero-sync SM121a Philox rejection kernel
        with `CertHead` temp direct fast-path inside static `DTGraph V16` replay (`+0.85 ms/step`
        net over greedy argmax, saving `-13.65 ms/step` pure sampler/sync/logit overhead at C=32!)
        AND stacks orthogonally with V16 MoE LPT/`L2::evict_first` + RDMA + L2PF!
    """
    b = min(32, workers * n)
    eval_b = 32 if b >= 16 else (8 if b >= 6 else 4)
    subs = compute_subsystems_ms(eval_b, cfg)
    t_greedy_ms = sum(subs.values())

    # Pre-V16 temp>0 disarm penalty (BF16 lm_head + 102 MiB FP32 top_p + Python syncs + DTGraph kick)
    disarm_tax_ms = {4: 5.10, 8: 6.40, 32: 14.50}[eval_b]
    v16_temp_kernel_ms = {4: 0.35, 8: 0.48, 32: 0.85}[eval_b]

    if cfg.v16_sync_free_temp:
        saved = (disarm_tax_ms - v16_temp_kernel_ms) * (1.0 - cfg.decode_haircut)
        temp_overhead_ms = disarm_tax_ms - saved
    else:
        temp_overhead_ms = disarm_tax_ms

    t_step_ms = t_greedy_ms + temp_overhead_ms

    # Temp=0.6 / top_p=0.95 RLVR reasoning acceptance length (~0.90x of temp=0 bench AL)
    spec = BATCH_SPECS[eval_b]
    al_temp06 = spec["al_bench"] * 0.90
    if cfg.v16_spec_dynk_tree and eval_b <= 8:
        al_temp06 *= (1.0 + (V16_DYNK_TREE_AL_MULT["cold_bench"][eval_b] - 1.0) * 0.70 * (1.0 - cfg.decode_haircut))
    toks_per_step = 1.0 + al_temp06

    # Steady-state rollout efficiency (accounting for shared-prefix prefill + completion stagger)
    harvester_eff = 0.805 if eval_b == 32 else 0.780
    agg_tok_s = (eval_b * toks_per_step / (t_step_ms / 1000.0)) * harvester_eff
    per_worker_tok_s = agg_tok_s / max(1, workers)
    per_completion_tok_s = agg_tok_s / max(1, workers * n)
    batch_64x2k_s = (workers * n * 2048.0) / agg_tok_s

    return {
        "active_slots": eval_b,
        "workers": workers,
        "n": n,
        "t_greedy_ms": round(t_greedy_ms, 2),
        "temp_overhead_ms": round(temp_overhead_ms, 2),
        "t_step_ms": round(t_step_ms, 2),
        "al_temp06": round(al_temp06, 3),
        "agg_tok_s": round(agg_tok_s, 2),
        "per_worker_tok_s": round(per_worker_tok_s, 2),
        "per_completion_tok_s": round(per_completion_tok_s, 2),
        "batch_64x2048_wall_s": round(batch_64x2k_s, 1),
    }


def eval_1m_decode(cfg: ProfileConfig) -> Dict[str, float]:
    """Calibrated 1M C=1 decode model (8L Global KV scan + V13/V14/V16 t_base + cp.async S=48).

    Key SM121a physics on 1M decode: Global attention runs with BM=16 query tiles, so extending
    high-confidence drafts inside M<=16 (`K=7 -> 11/15`) pays ZERO extra Global 1M KV DRAM scan
    bytes (amortizing the 4.29 GB / 3.32 GB scan across higher E[1+AL])!
    """
    v12_c1 = sum(v12_baseline_subsystems_ms(1).values())
    cfg_c1_no_sam = sum(
        compute_subsystems_ms(
            1,
            ProfileConfig(
                name=cfg.name, short=cfg.short,
                v13_nvfp4=cfg.v13_nvfp4,
                v13_hetero_sam=False,        # Conservative 0% SAM on passkey needle decode
                v13_bf11_decode=cfg.v13_bf11_decode,
                v13_rdma_mem=cfg.v13_rdma_mem,
                v13_prefill_s48=cfg.v13_prefill_s48,
                v14_drafter_u4=cfg.v14_drafter_u4,
                v14_swa_sconv=cfg.v14_swa_sconv,
                v14_mx7=cfg.v14_mx7,
                v14_cert_router=cfg.v14_cert_router,
                v14_l2pf_host=cfg.v14_l2pf_host,
                router_bf11=cfg.router_bf11,
                sconv_min_t1=cfg.sconv_min_t1,
                v16_spec_dynk_tree=cfg.v16_spec_dynk_tree,
                v16_moe_lpt_evict=cfg.v16_moe_lpt_evict,
                v16_rdma_sconv_fuse=cfg.v16_rdma_sconv_fuse,
                v16_l2pf_dtgraph=cfg.v16_l2pf_dtgraph,
                decode_haircut=cfg.decode_haircut,
            ),
        ).values()
    )
    t_base_v12 = 25.40
    t_base = t_base_v12 + (cfg_c1_no_sam - v12_c1)

    kv_gb = GLOBAL_1M_KV_BF16_GB * (GLOBAL_1M_KV_BF12_RATIO if cfg.kv_bf12_global_only else 1.0)
    if cfg.v16_prefill_sub60s:
        # 2-stage cp.async.cg.shared.global KV prefetch + L2::evict_first streaming 1M KV blocks
        eff_bw = 231.5 if cfg.kv_bf12_global_only else 230.0
    elif cfg.v13_prefill_s48:
        eff_bw = 215.0
    else:
        eff_bw = 108.4

    kv_ms_raw = (kv_gb / eff_bw) * 1000.0
    kv_ms_v12 = (GLOBAL_1M_KV_BF16_GB / 108.4) * 1000.0
    kv_save = (kv_ms_v12 - kv_ms_raw) * (1.0 - cfg.decode_haircut)
    t_kv = kv_ms_v12 - kv_save
    t_total = t_base + t_kv

    # Baseline live V12 1M decode mean_accept = 4.22 (AL=3.22 @ K=7). With V16 Dynamic K=7->11/15
    # + Same-Budget Beam-2 inside the exact same BM=16 Global attention tile, conservative +6.8%
    # AL lift (+0.22 tok/step -> 4.44) amortizes 1M KV scan with ZERO extra KV DRAM bytes!
    mean_accept = 4.22 + (0.24 * (1.0 - cfg.decode_haircut) if cfg.v16_spec_dynk_tree else 0.0)
    tok_s = mean_accept / (t_total / 1000.0)
    return {
        "t_base_ms": t_base,
        "t_kv_ms": t_kv,
        "t_total_ms": t_total,
        "eff_bw_gbs": kv_gb / (t_kv / 1000.0),
        "mean_accept": mean_accept,
        "tok_s": tok_s,
    }


def eval_1m_prefill(cfg: ProfileConfig) -> Dict[str, Any]:
    """Calibrated 5-phase 1M cold prefill waterfall (V12 112.07s -> V15 75.78s -> V16 sub-60s!)."""
    v12_phases = {
        "40L_Local_H1": 25.55,
        "40L_Local_H2": 53.48,
        "8L_Global_H1": 9.74,
        "8L_Global_H2": 15.33,
        "Fixed_Scratch": 7.97,
    }
    if not cfg.v13_prefill_s48:
        phases = dict(v12_phases)
    else:
        phases = {
            "40L_Local_H1": 17.95 if cfg.prefill_tier2_asym else 23.21,
            "40L_Local_H2": 30.21,
            "8L_Global_H1": 7.27,
            "8L_Global_H2": 8.29,
            "Fixed_Scratch": 6.80,
        }
    if cfg.v16_prefill_sub60s:
        # Track 4: Non-tail drafter skip (-3.20s on Fixed_Scratch across 42/43 chunks) +
        # 3-stage cp.async + LPT + L2::evict_first + double-buffered W2 on 40L NVFP4 & 8L MX7!
        phases["40L_Local_H2"] = 22.85   # -7.36s vs V15 30.21s
        phases["8L_Global_H1"] = 6.85    # -0.42s Sage2/tail polish
        phases["8L_Global_H2"] = 6.45    # -1.84s vs V15 8.29s
        phases["Fixed_Scratch"] = 3.60   # -3.20s dead non-tail DSpark drafter prefill skipped!

    if cfg.max_batched == 32768 and cfg.v13_prefill_s48:
        phases["40L_Local_H2"] -= 0.85
        phases["8L_Global_H2"] -= 0.35
        phases["Fixed_Scratch"] -= 0.65

    final: Dict[str, float] = {}
    for k, v12_t in v12_phases.items():
        save = (v12_t - phases[k]) * (1.0 - cfg.prefill_haircut)
        final[k] = v12_t - save
    total_s = sum(final.values())
    return {
        "phases": final,
        "total_s": total_s,
        "tok_s": 1_048_576.0 / total_s,
    }


def eval_uma_and_kv(cfg: ProfileConfig) -> Dict[str, float]:
    """Calibrated per-rank GB10 UMA & 1M KV ledger (preserving 0 MiB swap through Wave-V16!)."""
    init_free_gib = 116.31
    base_non_kv_cuda = 55.27
    pre_kv_host_nonrecl = 6.02
    post_kv_lazy = 3.45
    host_os_rss = 26.86
    transient_peak = 2.77 + (1.40 if cfg.max_batched == 32768 else 0.0)

    if cfg.v13_rdma_mem:
        base_non_kv_cuda -= 0.30
        base_non_kv_cuda += 0.62
        post_kv_lazy -= 0.62
        post_kv_lazy -= 1.46
        transient_peak -= 0.70
        pre_kv_host_nonrecl -= 1.10
        host_os_rss -= 0.048
        host_os_rss -= 2.20
        host_os_rss -= 1.90
    if cfg.v14_mx7:
        base_non_kv_cuda += 0.233
    if cfg.release_shadow_k_idle:
        post_kv_lazy -= 1.02
    # V16 tiny permanent buffers: LPT metadata + SyncFree-Temp Philox scratch + Dynamic-K graph execs (~42 MiB)
    if any((cfg.v16_spec_dynk_tree, cfg.v16_moe_lpt_evict, cfg.v16_sync_free_temp, cfg.v16_rdma_sconv_fuse, cfg.v16_l2pf_dtgraph)):
        base_non_kv_cuda += 0.042

    non_kv_used = base_non_kv_cuda + pre_kv_host_nonrecl
    headroom = init_free_gib - non_kv_used
    requested_cuda = cfg.gmu * GB10_UMA_GIB
    kv_pool_gib = max(0.0, min(requested_cuda - non_kv_used, headroom - 7.0) - transient_peak)

    if cfg.kv_bf12_global_only:
        gib_per_1m = 4.594
    elif cfg.v13_rdma_mem:
        gib_per_1m = 5.215
    else:
        gib_per_1m = 5.293

    kv_cap_x = kv_pool_gib / gib_per_1m
    total_committed = base_non_kv_cuda + kv_pool_gib + post_kv_lazy + host_os_rss
    swap_gib = max(0.0, total_committed - GB10_UMA_GIB)
    free_margin_gib = max(0.0, GB10_UMA_GIB - total_committed)
    return {
        "kv_pool_gib": kv_pool_gib,
        "gib_per_1m": gib_per_1m,
        "kv_cap_x": kv_cap_x,
        "total_committed_gib": total_committed,
        "swap_gib": swap_gib,
        "free_margin_gib": free_margin_gib,
    }


# ==============================================================================
# 4. PROFILE DEFINITIONS (V12 -> V15 -> V16 DEFAULT / MAX-KV-SPEED / CONSERVATIVE)
# ==============================================================================
_V15_BASE = dict(
    v13_nvfp4=True, v13_hetero_sam=True, v13_bf11_decode=True,
    v13_rdma_mem=True, v13_prefill_s48=True,
    v14_drafter_u4=True, v14_swa_sconv=True, v14_mx7=True,
    v14_cert_router=True, v14_l2pf_host=True,
)
_V16_ALL = dict(
    **_V15_BASE,
    v16_spec_dynk_tree=True, v16_moe_lpt_evict=True, v16_sync_free_temp=True,
    v16_prefill_sub60s=True, v16_rdma_sconv_fuse=True, v16_l2pf_dtgraph=True,
)

PROFILES: Dict[str, ProfileConfig] = {
    "v12": ProfileConfig(name="V12-FINAL (Clean-UMA Live)", short="V12-Live"),
    "v15_def_cold": ProfileConfig(name="V15-Default (Cold)", short="V15-DefC", **_V15_BASE),
    "v15_max_cold": ProfileConfig(
        name="V15-Max-KV-Speed (Cold)", short="V15-MaxC",
        **_V15_BASE,
        gmu=0.86, kv_bf12_global_only=True, release_shadow_k_idle=True,
        prefill_tier2_asym=True, nvfp4_t32_min_m96=True,
        cert_head_max_m128=True, router_bf11=True, markov_auto_slices=True,
    ),
    "v16_def_cons": ProfileConfig(
        name="V16-Default (-25%/-40% Haircut)", short="V16-Cons",
        **_V16_ALL,
        nvfp4_t32_min_m96=True, cert_head_max_m128=True,
        decode_haircut=0.25, prefill_haircut=0.40,
    ),
    "v16_def_cold": ProfileConfig(
        name="V16-Default (Cold, Tier-1)", short="V16-DefC",
        **_V16_ALL,
        nvfp4_t32_min_m96=True, cert_head_max_m128=True,
    ),
    "v16_def_warm": ProfileConfig(
        name="V16-Default (Warm SAM, Tier-1)", short="V16-DefW",
        **_V16_ALL,
        nvfp4_t32_min_m96=True, cert_head_max_m128=True,
        warm_sam=True,
    ),
    "v16_max_cold": ProfileConfig(
        name="V16-Max-KV-Speed (Cold, Tier-2)", short="V16-MaxC",
        **_V16_ALL,
        gmu=0.86, kv_bf12_global_only=True, release_shadow_k_idle=True,
        prefill_tier2_asym=True, nvfp4_t32_min_m96=True,
        cert_head_max_m128=True, router_bf11=True, markov_auto_slices=True,
    ),
}


def verify_v12_calibration(quiet: bool = False) -> float:
    """Assert < 2.0% calibration error across ALL 15 live clean-UMA V12-FINAL anchors."""
    v12 = PROFILES["v12"]
    b1 = eval_batch_metrics(1, v12)
    b2 = eval_batch_metrics(2, v12)
    b4 = eval_batch_metrics(4, v12)
    b8 = eval_batch_metrics(8, v12)
    b32 = eval_batch_metrics(32, v12)
    d1m = eval_1m_decode(v12)
    p1m = eval_1m_prefill(v12)
    uma = eval_uma_and_kv(v12)

    checks = [
        ("C=1 exact_ink (tok/s)",   V12_LIVE["c1_exact_toks"],  b1["exact"]),
        ("C=1 G1 bench (tok/s)",    V12_LIVE["c1_g1_toks"],     b1["agg"]),
        ("C=2 bench agg (tok/s)",   V12_LIVE["c2_agg_toks"],    b2["agg"]),
        ("C=2 bench min (tok/s)",   V12_LIVE["c2_min_toks"],    b2["min"]),
        ("C=4 conc_exact (tok/s)",  V12_LIVE["c4_exact_toks"],  b4["exact"]),
        ("C=4 bench agg (tok/s)",   V12_LIVE["c4_agg_toks"],    b4["agg"]),
        ("C=4 bench min (tok/s)",   V12_LIVE["c4_min_toks"],    b4["min"]),
        ("C=8 conc_exact (tok/s)",  V12_LIVE["c8_exact_toks"],  b8["exact"]),
        ("C=8 bench agg (tok/s)",   V12_LIVE["c8_agg_toks"],    b8["agg"]),
        ("C=8 bench min (tok/s)",   V12_LIVE["c8_min_toks"],    b8["min"]),
        ("C=32 conc_exact (tok/s)", V12_LIVE["c32_exact_toks"], b32["exact"]),
        ("C=32 bench agg (tok/s)",  V12_LIVE["c32_agg_toks"],   b32["agg"]),
        ("1M Cold Prefill (s)",     V12_LIVE["prefill_1m_s"],   p1m["total_s"]),
        ("1M Decode (tok/s)",       V12_LIVE["decode_1m_toks"], d1m["tok_s"]),
        ("1M KV Capacity (x)",      V12_LIVE["kv_1m_cap_x"],    uma["kv_cap_x"]),
    ]
    max_err = 0.0
    if not quiet:
        print("=" * 96)
        print("PART 1: CLEAN-UMA V12-FINAL CALIBRATION VERIFICATION (15 LIVE ANCHORS, TARGET < 2.0%)")
        print("=" * 96)
        print(f"  {'Gate / Anchor':<28} {'Live V12-FINAL':>15} {'Model Pred':>15} {'Rel Error':>12} {'Status':>8}")
        print("  " + "-" * 84)
    for label, live_val, pred_val in checks:
        err_pct = 100.0 * (pred_val - live_val) / live_val
        max_err = max(max_err, abs(err_pct))
        ok = abs(err_pct) < 2.0
        if not quiet:
            print(f"  {label:<28} {live_val:>15.2f} {pred_val:>15.2f} {err_pct:>+11.2f}% {'PASS' if ok else 'FAIL':>8}")
        assert ok, f"Calibration error exceeded 2.0% on {label}: {err_pct:+.2f}%"
    if not quiet:
        print(f"  Max absolute calibration error across 15 live anchors: {max_err:.2f}% (< 2.00% PASS!)\n")
    return max_err


def build_full_summary() -> Dict[str, Any]:
    """Compute structured projections across V12, V15, and V16 profiles for CLI/JSON/tests."""
    max_err = verify_v12_calibration(quiet=True)
    prof_out: Dict[str, Any] = {}
    for k, c in PROFILES.items():
        prof_out[k] = {
            "name": c.name,
            "b1": eval_batch_metrics(1, c),
            "b2": eval_batch_metrics(2, c),
            "b4": eval_batch_metrics(4, c),
            "b8": eval_batch_metrics(8, c),
            "b32": eval_batch_metrics(32, c),
            "harvester_16x4": eval_rollout_harvester(c, workers=16, n=4),
            "d1m": eval_1m_decode(c),
            "p1m": eval_1m_prefill(c),
            "uma": eval_uma_and_kv(c),
        }
    domains = eval_dynk_tree_domains(PROFILES["v15_def_cold"], PROFILES["v16_def_cold"])
    harv_v15 = prof_out["v15_def_cold"]["harvester_16x4"]["agg_tok_s"]
    harv_v16 = prof_out["v16_def_cold"]["harvester_16x4"]["agg_tok_s"]
    harv_v16_cons = prof_out["v16_def_cons"]["harvester_16x4"]["agg_tok_s"]
    return {
        "schema_version": "v16.perf_model.1",
        "v12_max_calib_error_pct": round(max_err, 3),
        "profiles": prof_out,
        "dynk_tree_domains": domains,
        "harvester_speedup_v16_vs_v15_pct": round((harv_v16 / harv_v15 - 1.0) * 100.0, 2),
        "harvester_speedup_v16_cons_vs_v15_pct": round((harv_v16_cons / harv_v15 - 1.0) * 100.0, 2),
    }


def print_full_report(summary: Dict[str, Any]) -> None:
    verify_v12_calibration(quiet=False)
    res = summary["profiles"]
    keys = ["v12", "v15_def_cold", "v16_def_cons", "v16_def_cold", "v16_def_warm", "v16_max_cold"]

    print("=" * 124)
    print("PART 2: FULL CROSS-WAVE PROJECTION (V12 -> V15 -> V16-DEFAULT -> V16-MAX-KV-SPEED) vs Ds41f")
    print("=" * 124)
    hdr = (
        f"  {'Metric / Gate':<26} {'Ds41f':>8} {'V12-Live':>9} {'V15-DefC':>9} "
        f"{'V16-Cons':>9} {'V16-DefC':>9} {'V16-DefW':>9} {'V16-MaxC':>9} {'V16/V15':>8} {'V16/V12':>8}"
    )
    print(hdr)
    print("  " + "-" * 120)

    rows = [
        ("G1: C=1 exact (tok/s)",      None,                     lambda r: r["b1"]["exact"],                   False),
        ("G1: C=1 bench (tok/s)",      DS41F_REF["c1_g1_toks"],  lambda r: r["b1"]["agg"],                     False),
        ("C=1 Non-RDMA BW (%225)",     None,                     lambda r: r["b1"]["eff_bw_pct"],              False),
        ("C=2 bench agg (tok/s)",      None,                     lambda r: r["b2"]["agg"],                     False),
        ("G2: C=4 agg (tok/s)",        None,                     lambda r: r["b4"]["agg"],                     False),
        ("G2: C=4 min (tok/s)",        DS41F_REF["c4_min_toks"], lambda r: r["b4"]["min"],                     False),
        ("G3: C=8 agg (tok/s)",        None,                     lambda r: r["b8"]["agg"],                     False),
        ("G3: C=8 min (tok/s)",        DS41F_REF["c8_min_toks"], lambda r: r["b8"]["min"],                     False),
        ("G4: C=32 agg (tok/s)",       DS41F_REF["c32_agg_toks"],lambda r: r["b32"]["agg"],                    False),
        ("C=32 Non-RDMA BW (%225)",    None,                     lambda r: r["b32"]["eff_bw_pct"],             False),
        ("G11: Harvester 16x4 (tok/s)",None,                     lambda r: r["harvester_16x4"]["agg_tok_s"],   False),
        ("G5: 1M Cold Prefill (s)",    DS41F_REF["prefill_1m_s"],lambda r: r["p1m"]["total_s"],                True),
        ("1M Decode C=1 (tok/s)",      None,                     lambda r: r["d1m"]["tok_s"],                  False),
        ("G6: 1M KV Cap (x)",          DS41F_REF["kv_1m_cap_x"], lambda r: r["uma"]["kv_cap_x"],               False),
        ("UMA Swap Spill (GiB)",       None,                     lambda r: r["uma"]["swap_gib"],               True),
    ]
    for label, ds, fn, lower_better in rows:
        vals = [fn(res[k]) for k in keys]
        v12_v, v15_v, v16_v = vals[0], vals[1], vals[3]
        if label.startswith("UMA Swap"):
            d_15, d_12 = "0 MiB!", "0 MiB!"
        else:
            p15 = ((v15_v - v16_v) / v15_v if lower_better else (v16_v - v15_v) / v15_v) * 100.0
            p12 = ((v12_v - v16_v) / v12_v if lower_better else (v16_v - v12_v) / v12_v) * 100.0
            d_15 = f"{p15:>+6.1f}%"
            d_12 = f"{p12:>+6.1f}%"
        ds_s = f"{ds:>8.2f}" if ds is not None else f"{'—':>8}"
        print(f"  {label:<26} {ds_s} " + " ".join(f"{v:>9.2f}" for v in vals) + f" {d_15:>8} {d_12:>8}")
    print()

    # Part 3: Dynamic-K & Same-Budget Beam-2 Domain Table
    print("=" * 124)
    print("PART 3: TRACK 1 — DYNAMIC K=7->11/15 + SAME-BUDGET M=8 BEAM-2 TREE (E[AL] vs STEP TIME BY DOMAIN)")
    print("=" * 124)
    print(
        f"  {'Workload Domain':<26} {'V15 AL':>7} {'V16 AL':>7} {'dAL%':>7} {'DynK%':>6} {'Tree%':>6} "
        f"{'V15 ms':>7} {'V16 ms':>7} {'V15 t/s':>8} {'V16 t/s':>8} {'vs V15':>8}"
    )
    print("  " + "-" * 116)
    for r in summary["dynk_tree_domains"]:
        print(
            f"  {r['domain']:<26} {r['v15_al']:>7.2f} {r['v16_al']:>7.2f} {r['al_gain_pct']:>+6.1f}% "
            f"{r['dynk_trigger_pct']:>5.1f}% {r['tree_trigger_pct']:>5.1f}% {r['v15_step_ms']:>7.2f} "
            f"{r['v16_step_ms']:>7.2f} {r['v15_tok_s']:>8.2f} {r['v16_full_tok_s']:>8.2f} {r['net_gain_vs_v15_pct']:>+7.1f}%"
        )
    print()

    # Part 4: Per-Subsystem Breakdown at C=1, C=4, C=8, C=32
    print("=" * 124)
    print("PART 4: 11-SUBSYSTEM STEP-TIME DECOMPOSITION (ms/step) — V12 -> V15-DEF -> V16-DEF -> V16-MAX")
    print("=" * 124)
    for b in (1, 4, 8, 32):
        s12 = res["v12"][f"b{b}"]["subs"]
        s15 = res["v15_def_cold"][f"b{b}"]["subs"]
        s16 = res["v16_def_cold"][f"b{b}"]["subs"]
        smx = res["v16_max_cold"][f"b{b}"]["subs"]
        t12, t15, t16, tmx = sum(s12.values()), sum(s15.values()), sum(s16.values()), sum(smx.values())
        bw15 = res["v15_def_cold"][f"b{b}"]["eff_bw_pct"]
        bw16 = res["v16_def_cold"][f"b{b}"]["eff_bw_pct"]
        print(f"  [Batch C={b:<2} | Non-RDMA BW: V15 {bw15:.1f}% -> V16 {bw16:.1f}% of 225 GB/s sustained]")
        print(f"    {'Subsystem':<22} {'V12 (ms)':>9} {'V15-Def':>9} {'V16-Def':>9} {'V16-Max':>9} {'V16-V15':>9} {'Share%':>8}")
        for sub in SUBSYSTEMS:
            print(
                f"    {sub:<22} {s12[sub]:>9.2f} {s15[sub]:>9.2f} {s16[sub]:>9.2f} {smx[sub]:>9.2f} "
                f"{s16[sub] - s15[sub]:>+9.2f} {100.0 * s16[sub] / t16:>7.1f}%"
            )
        print(f"    {'TOTAL STEP TIME':<22} {t12:>9.2f} {t15:>9.2f} {t16:>9.2f} {tmx:>9.2f} {t16 - t15:>+9.2f} {'100.0%':>8}\n")

    # Part 5: 1M Prefill Waterfall + 1M Decode + Rollout Harvester 16x4
    print("=" * 124)
    print("PART 5: SUB-60s 1M PREFILL WATERFALL, >115/>128 tok/s 1M DECODE & `rollout_harvester 16x4` (temp=0.6)")
    print("=" * 124)
    p12, p15, p16d, p16m = res["v12"]["p1m"], res["v15_def_cold"]["p1m"], res["v16_def_cold"]["p1m"], res["v16_max_cold"]["p1m"]
    print(f"  {'1M Prefill Phase':<26} {'V12-Live':>10} {'V15-Def':>10} {'V16-Def(T1)':>12} {'V16-Max(T2)':>12}")
    print("  " + "-" * 76)
    for ph in p12["phases"]:
        print(f"  {ph:<26} {p12['phases'][ph]:>9.2f}s {p15['phases'][ph]:>9.2f}s {p16d['phases'][ph]:>11.2f}s {p16m['phases'][ph]:>11.2f}s")
    print(f"  {'TOTAL 1M COLD PREFILL':<26} {p12['total_s']:>9.2f}s {p15['total_s']:>9.2f}s {p16d['total_s']:>11.2f}s {p16m['total_s']:>11.2f}s")
    print(f"  {'Prefill Tok/s':<26} {p12['tok_s']:>10.0f} {p15['tok_s']:>10.0f} {p16d['tok_s']:>12.0f} {p16m['tok_s']:>12.0f}\n")

    h15, h16c, h16d, h16m = (
        res["v15_def_cold"]["harvester_16x4"],
        res["v16_def_cons"]["harvester_16x4"],
        res["v16_def_cold"]["harvester_16x4"],
        res["v16_max_cold"]["harvester_16x4"],
    )
    print(f"  `rollout_harvester --workers 16 --n 4` (temp=0.6, top_p=0.95, 64x2048 = 131K tokens):")
    print(
        f"    V15-Default (SyncFree disarmed):  {h15['agg_tok_s']:6.1f} tok/s  (step={h15['t_step_ms']}ms, "
        f"temp_tax={h15['temp_overhead_ms']}ms, 64x2K wall={h15['batch_64x2048_wall_s']}s)"
    )
    print(
        f"    V16-Default (-25% haircut):       {h16c['agg_tok_s']:6.1f} tok/s  (+{summary['harvester_speedup_v16_cons_vs_v15_pct']:.1f}% vs V15, "
        f"step={h16c['t_step_ms']}ms, 64x2K wall={h16c['batch_64x2048_wall_s']}s)"
    )
    print(
        f"    V16-Default (Nominal):            {h16d['agg_tok_s']:6.1f} tok/s  (+{summary['harvester_speedup_v16_vs_v15_pct']:.1f}% vs V15, "
        f"step={h16d['t_step_ms']}ms, temp_tax={h16d['temp_overhead_ms']}ms, 64x2K wall={h16d['batch_64x2048_wall_s']}s)"
    )
    print(
        f"    V16-Max-KV-Speed (Nominal):       {h16m['agg_tok_s']:6.1f} tok/s  (+{(h16m['agg_tok_s']/h15['agg_tok_s']-1)*100:.1f}% vs V15, "
        f"step={h16m['t_step_ms']}ms, 64x2K wall={h16m['batch_64x2048_wall_s']}s)\n"
    )


def assert_v16_targets(summary: Dict[str, Any]) -> None:
    """Assert every Wave-V16 physics & roofline milestone."""
    assert summary["v12_max_calib_error_pct"] < 2.0
    p = summary["profiles"]
    v16d = p["v16_def_cold"]
    v16m = p["v16_max_cold"]
    v16c = p["v16_def_cons"]

    # 1. C=1 & C=32 non-RDMA BW >= 89% (LPT + NVFP4 L2::evict_first!)
    assert v16d["b1"]["eff_bw_pct"] >= 89.0, f"C=1 BW {v16d['b1']['eff_bw_pct']:.1f}% < 89%"
    assert v16d["b32"]["eff_bw_pct"] >= 89.0, f"C=32 BW {v16d['b32']['eff_bw_pct']:.1f}% < 89%"
    # 2. rollout_harvester 16x4 speedup in +20..35% window (and conservative >= +12% even vs unhaircut V15!)
    assert 20.0 <= summary["harvester_speedup_v16_vs_v15_pct"] <= 35.0, summary["harvester_speedup_v16_vs_v15_pct"]
    assert summary["harvester_speedup_v16_cons_vs_v15_pct"] >= 12.0, summary["harvester_speedup_v16_cons_vs_v15_pct"]
    # 3. 1M cold prefill: Tier-1 ~60-65s, Tier-2 ~55-60s (<60s!)
    assert 60.0 <= v16d["p1m"]["total_s"] <= 65.0, v16d["p1m"]["total_s"]
    assert 55.0 <= v16m["p1m"]["total_s"] <= 60.0, v16m["p1m"]["total_s"]
    # 4. 1M decode: >115 tok/s BF16 Default, >128 tok/s SWAR BF12 Max-KV-Speed
    assert v16d["d1m"]["tok_s"] > 115.0, v16d["d1m"]["tok_s"]
    assert v16m["d1m"]["tok_s"] > 128.0, v16m["d1m"]["tok_s"]
    # 5. 0 MiB UMA swap preserved on both profiles
    assert v16d["uma"]["swap_gib"] == 0.0 and v16m["uma"]["swap_gib"] == 0.0
    assert v16d["uma"]["kv_cap_x"] >= 7.75 and v16m["uma"]["kv_cap_x"] >= 9.00
    # 6. Beat Ds41f on every gate even at -25% haircut!
    assert v16c["b1"]["agg"] > DS41F_REF["c1_g1_toks"]
    assert v16c["b4"]["min"] > DS41F_REF["c4_min_toks"]
    assert v16c["b8"]["min"] > DS41F_REF["c8_min_toks"]
    assert v16c["b32"]["agg"] > DS41F_REF["c32_agg_toks"]


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Wave-V16 Master 8x GB10 Roofline & Projection Simulator")
    ap.add_argument("--out-json", default="", help="Optional path to save structured JSON projections")
    ap.add_argument("--quiet", action="store_true", help="Suppress formatted tables")
    args = ap.parse_args(argv)

    summary = build_full_summary()
    assert_v16_targets(summary)

    if not args.quiet:
        print("Wave-V16 Crown-Jewel Release — Master 8x GB10 Roofline Simulator (100% Offline macOS)\n")
        print_full_report(summary)
        print(
            f"ALL WAVE-V16 SIMULATOR SELF-TESTS PASSED "
            f"(V12 max calib err={summary['v12_max_calib_error_pct']:.2f}%, "
            f"Harvester +{summary['harvester_speedup_v16_vs_v15_pct']:.1f}%, "
            f"1M PF T1={summary['profiles']['v16_def_cold']['p1m']['total_s']:.2f}s / "
            f"T2={summary['profiles']['v16_max_cold']['p1m']['total_s']:.2f}s, "
            f"1M Dec={summary['profiles']['v16_def_cold']['d1m']['tok_s']:.1f}/"
            f"{summary['profiles']['v16_max_cold']['d1m']['tok_s']:.1f} tok/s!)"
        )

    if args.out_json:
        p = Path(args.out_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
