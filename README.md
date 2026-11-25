# Inkling-2-Small on 8× NVIDIA DGX Spark (TP8, SM121a)

Production-grade, **100% 0-ULP bit-exact** inference stack for
[`thinkingmachines/Inkling-2-Small`](https://huggingface.co/thinkingmachines/Inkling-2-Small)
(594B total / 17B active MoE, 384 routed experts top-8, 40 NVFP4 local SWA-1024 layers + 8 MXFP8 global layers, 1M context, 5-layer DSpark speculator)
on **8× NVIDIA DGX Spark (GB10, SM121a)** over **4× 200GbE ConnectX-7 RoCEv2**.

> **Status (2026-11-25): PERMANENT LIVE PRODUCTION CHECKPOINT on 8/8 nodes (`0.0.0.0:8010`, `tp8/config/live_champion_v16.env`, `V16-FINAL`).**
> Empirically qualified and pruned across Wave-V13→V16 on live 8-node GB10 silicon, `V16-FINAL` delivers **10/10 0-ULP deterministic decode (`1dcf…f8c3`)** across cold, warm, post-chaos/harvester/multimodal, and post-1M runs, **`139.07` GM / `144.88` mean `C=1`** (`bench.py single 1024t`, `202.30 tok/s` math), **`136.74 tok/s` (`74.85 min`) `C=2`**, **`187.64 tok/s` (`51.28 min`) `C=4`**, **`567.68 tok/s` (`23.10 mean / 18.90 min`) `C=32`**, **`90.1 tok/s` `1M` decode (`+38.8%` over `V12-FINAL`!)** at **`113.02 s` cold / `2.34 s` warm (`3/3` needles)**, **`417.78 tok/s`** zero-sync `temp=0.6` `16×4` RLVR harvesting (`1.00` unique ratio), **`7.11×` simultaneous 1M KV residency**, **`0` KV leaks** under 20%-abort chaos soak, and **`0 MiB` new swap**.

---

## Qualified Permanent Checkpoint Results (`V16-FINAL` → `tp8/config/live_champion_v16.env`)

All `V16-FINAL` numbers were measured on **one unified live 8-Spark server** (`GMU=0.85`, `MAX_BATCHED=24576`, port `8010`, stock 2000 MHz GB10 clocks, `live_champion_v16.env` sha256 `413f1ae7275d…`).

| Metric / Gate | `EXP-6B` (`best.env`) | `V12-FINAL` (`live_champion_v12.env`) | **Current (`V16-FINAL`)** | Delta vs `V12-FINAL` | How Measured |
|---|---:|---:|---:|---:|---|
| **1. Bit-Exact Decode (0-ULP)** | `1dcf…f8c3` (3/3) | `1dcf…f8c3` (10/10) | **`1dcf…f8c3` (10/10)** | **0-ULP LOCK** | `exact_ink.py` cold, warm, post-stress, post-1M & live `0.0.0.0:8010` (`1dcfb30680d0252c9c259e5cb2bd02d402b2f6123261f904a0074d3b5b46f8c3`) |
| **2a. `C=1` `bench.py single` (`1024t`)** | `139.5` | `137.20` GM (`131.0 / 106.2 / 174.4`) | **`139.07` GM (`144.88` mean: `130.3 / 102.1 / 202.3`)** | **`+1.87` GM (`+27.9` math)** | `bench.py single --max-tokens 1024 --temp 0` (`code / prose / math`) |
| **2b. `C=1` `exact_ink.py`** | — | `151.66` *(today `150.27`)* | **`148.58` warm** (`147.72` prod cold) | `-1.7` | `exact_ink.py` (10 prompts, 384 tok, serial) |
| **3a. `C=2` `bench.py` (`256t`)** | — | `134.60` (`80.39 / 72.46 min`) | **`136.74` (`81.08 / 74.85 min`)** | **`+2.14` agg / `+2.39` min** | `bench.py conc --conc 2 --max-tokens 256` (`AL 2.44`, **ALL-TIME RECORD**) |
| **3b. `C=4` `bench.py` (`256t`)** | — | `181.25` (`57.52 / 48.20 min`) | **`187.64` (`58.23 / 51.28 min`)** | **`+6.39` agg / `+3.08` min** | `bench.py conc --conc 4 --max-tokens 256` (`AL 2.72`, **ALL-TIME RECORD**) |
| **3c. `C=8` `bench.py` / `conc_exact`** | — | `269.66` / `286.03` | **`259.49` (`44.14 / 35.08 min`)** / **`276.01`** | `-10.2` / `-10.0` | `bench.py conc --conc 8` (`256t`) & `conc_exact` (`128t`) |
| **4. `C=32` `bench.py` (`256t`)** | `518.71` *(512t)* | `563.02` (`22.90 / 18.88 min`) | **`567.68` (`23.10 / 18.90 min`)** | **`+4.66` agg (`+88.7` vs Ds41f)** | `bench.py conc --conc 32 --max-tokens 256` (`AL 2.41`, **ALL-TIME RECORD**) |
| **5a. `32K` Prefill & Decode** | — | `4.30 s` / `0.32 s` (`158.4` dec) | **`4.30 s` / `0.30 s` (`166.3 tok/s` dec)** | **`+7.9 tok/s` dec (`3/3`)** | `bench_1m_e2e.py --ctx 32768 --max-tokens 384 --warm` |
| **5b. `128K` Prefill & Decode** | — | `14.45 s` / `0.44 s` (`134.0` dec) | **`14.49 s` / `0.44 s` (`132.0 tok/s` dec)** | `parity` (`3/3`) | `bench_1m_e2e.py --ctx 131072 --max-tokens 384 --warm` |
| **5c. `1M` Cold / Warm / Decode** | `120.90 s` / `2.95 s` | `112.07 s` / `2.36 s` (`64.9` dec) | **`113.02 s` / `2.34 s` (`90.1 tok/s` dec!)** | **`+25.2 tok/s` (`+38.8%` dec!)** | `bench_1m_e2e.py --ctx 1048500 --max-tokens 384 --warm` (`48.3×` warm, `3/3` needles) |
| **6. RLVR Harvester (`16×4`, `temp=0.6`)** | — | Synced PyTorch sampler | **`417.78 tok/s`** (`64/64`, `1.00` uniq, `0` collapsed) | **Zero-Sync SM121a** | `v16_harvester_stress_test.py --workers 16 --n 4 --max-tokens 128` |
| **7. Chaos Soak & Multimodal API** | — | — | **`96/96` OK, `24/24` aborts, `0` leaks, `7/7` API PASS** | **Production Hardened** | `v15_chaos_soak.py` (120 req, C=16, 20% abort) + `test_api_multimodal_live_v15.py` |
| **8. 1M KV Residency & Telemetry** | `4.8×` (`5.07M` tok) | `7.43×` (`7.79M` tok) | **`7.11×` (`7,455,548` tok, `0 MiB` new swap)** | **`/dev/shm` telemetry** | `43.58 GiB` KV (`4.68×` Ds41f's `1.52×`), `HEALTH_200_OK`, `0` KV audit anomalies |

- Full machine-readable archives: [`tp8/results/checkpoint_2026-11-25_v16_final.json`](tp8/results/checkpoint_2026-11-25_v16_final.json) · [`tp8/results/checkpoint_2026-11-24_v12_final.json`](tp8/results/checkpoint_2026-11-24_v12_final.json) · [`tp8/results/wave_v12_live_report.md`](tp8/results/wave_v12_live_report.md).

---

## Hardware Topology (8× DGX Spark)

- **GPU:** NVIDIA GB10 (`SM121a`), 48 SMs, 99 KiB SMEM/SM, **244 GB/s** sustained UMA LPDDR5X, 24 MiB L2, 121.63 GiB usable UMA/node.
- **CPU:** 20-core Grace ARM (`0..4, 10..14` Cortex-A725 LITTLE; `5..9, 15..19` Cortex-X925 BIG). Container pinned to `--cpuset-cpus=5-9,15-19` with `SCHED_FIFO` RDMA proxies on CPUs `9` & `19` (`tp8/patches/host_pin.py`).
- **Fabric:** 4× 200GbE ConnectX-7 RoCEv2 (`~25.8 GB/s` aggregate wire, `NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0,rocep1s0f1,roceP2p1s0f1`, `NCCL_IB_QPS_PER_CONNECTION=2`, `NCCL_IB_PCI_RELAXED_ORDERING=1`).

---

## What Shipped in `V16-FINAL` (Live-Verified Wins)

All decode kernels compile with **`-gencode arch=compute_121a,code=sm_121a --fmad=false --ptxas-options=-v`** (`0` register spills, `0` stack frames, no `--ftz=true`, no `--use_fast_math`) and are strictly gated by `exact_ink.py` (`1dcf…f8c3`, 10/10):

1. **RDMA V16 (`tp8/kernels/v16/rdma_sconv_v16/rdma_ar_c32_v16.cu`):** Register-fused BF12 AllReduce for `C=4..32` (`P1=262144`, `P2=196608`, `MIN_NVEC=18432`, `ACTIVE_RED=1`, `EXACT_GRID=1`, `EXACT_ROWS=1`, `PAR_QP2_POLL=1`, `TIMEOUT_MS=25000`). Head-to-head beat RDMA V9 by `+14.6 tok/s` `C=4`, `+4.6` `C=8`, `+9.2` `C=32` at `10/10 1dcfb306`.
2. **Max-Occupancy 48L `C48 MoE V16` (`tp8/kernels/v16/moe_v16/{nvfp4_ws5_c48_v16.cu, mxfp8_mx7_c48_v16.cu}`):** Compiled at `BN16=8, W2=6` (`0` spills) with `L2::evict_first` (`EVICT=1`), `LPT=0`, and V12 default `FC_MODE=-1`.
3. **`CertHead V14 M<=128` (`tp8/kernels/v16/certhead_router_v14/cert_head_fast_v14.cu`):** Qualified `127/127` exact (`ref=b1_exact`) across `M=2..128`, eliminating the `C=32` (`M=128`) FP32 `lm_head` fallback and pushing `bench C=32` to a new **`567.68 tok/s`** all-time record.
4. **`Attn V16` Long-Context `S=48` (`tp8/kernels/v16/pf_v13_base/attn_v13/` + `pf_attn_v16/attn_global_v16.patch`):** Enabled strictly for long-context global decode (`S=48`, preserving V12 SWA & short-context kernels untouched), surging `1M` decode from **`64.9` → `90.1 tok/s` (`+38.8%`)** and `32K` decode from `158.4` → **`166.3 tok/s`** at `3/3` needles.
5. **Zero-Sync `temp>0` Sampling & `DTGraph V14/V16` (`tp8/kernels/v16/sync_free_temp_v16/sync_free_temp_v16.cu`):** SM121a zero-sync stochastic sampler (`ok=1`) + `CERT_HEAD_TEMP_DIRECT_V16=1`, delivering **`417.78 tok/s`** on `16×4` (`64` concurrent completions) `temp=0.6` RLVR rollout harvesting with `1.00` unique ratio and `0` collapsed groups.
6. **Non-Tail Draft Prefill Skip V16 (`INKLING_PF_SKIP_NONTAIL_DRAFT_V16=1`) + Markov Auto-Slices (`INKLING_MARKOV_FOLD_C48_AUTO_SLICES=1`).**
7. **V15/V16 Production Hardening, Mixed-Decode Guards & Zero-Overhead Telemetry:** Prefill mixed-decode guards (`PREFILL_SP_GUARD_MIXED_DECODE`, `PF_GLOBAL_GUARD_MIXED_DECODE`, `PF_MOE_MIXED_FP32_COMBINE`, `PF_MIXED_DECODE_BF16`), `PF_SHADOW_K_STRICT_VERIFY=1`, and background telemetry thread writing `/dev/shm/inkling_v16_rank0_metrics.json` every `5 s` + complete ops stack under `tp8/ops/`.

---

## Empirically Pruned Candidates (Do Not Re-Enable Without Redesign)

| Candidate Tested Live | Measured Outcome | Why Pruned |
|---|---|---|
| **`TREE_SPEC=1` / `SAME_BUDGET_TREE_V16`** | `2/10 f7e65036`, `-2.4 tok/s` `C=1` | Violated 0-ULP `1dcfb306` hard gate |
| **`MOE_S8=1` (`v10/v5/c1_v16`)** | `0/10 ab56fa68`, `-9.6 tok/s` `C=2`; `c1_v16` IMA | Violated 0-ULP gate / warmup IMA |
| **`BF11_V13=1` / `ROUTER_V14=1`** | `-6.5` to `-8.5 tok/s` `C=1` | Regressed vs V12 BF11/Router |
| **`L2PF_V16=1`** (`185 GB/s`, `17.25 MB`) | `-17 tok/s` `C=32`, `-7 tok/s` `C=4` | Thrashed 24 MiB GB10 L2; kept V12 L2PF |
| **`DRAFT_SWA_SPARSE=1` / `L2PF_DRAFT_ATTN=1`** | Step-3/4 RDMA hang + `700%` KV warn | Deadlocked in live multi-node decode |
| **`SPEC_K=15` / `DYN_K_EXTEND_V16=1`** | `-11.7 tok/s` `C=1` (`148.5 → 136.8`) | Drafted `K=15` unconditionally even on prose |
| **`HETERO_BUDGET_SPEC=1` / `K=7@C2..4`** | `C=4` `188.6 → 174.4`, `C=8` `259.5 → 243.3` | Kept V12 schedule `[[1,1,7],[2,3,6],[4,8,5],[9,24,3],[25,32,3]]` |
| **`Prefill V16` (`W13_V6/W2_V5/MX7_V10/FP8_ATTN_V13`)** | `128K` `18.56 s`, `1M` `142.3 s` | Slower than V12 Prefill (`14.49 s` / `113.02 s`) |
| **`KV_BF12_GLOBAL_ONLY=1`** | Prompt-0 `CUDA_ERROR_LAUNCH_FAILED`, KV `6.81×` | Crashed AND shrank KV via pool padding |
| **`C48 MoE V16 LPT=2`** | Prompt-0 IMA | PDL `launch_dependents` raced shared `ws["work_queue"]` between W13 & W2; clean at `LPT=0` |
| **`DECODE_EDGE_GUARD_V15=1`** | Prompt-0 decode IMA | Allocated temp tensors inside CUDA graph capture + wrong KV axis (`shape[0]=2`); patched in source & kept OFF on hot decode path |

---

## Repository Layout

```text
tp8/
  config/
    live_champion_v16.env                  # Promoted V16-FINAL permanent production default (GMU=0.85)
    live_champion_v12.env                  # Preserved V12-FINAL rollback baseline (GMU=0.85)
    live_champion_v12_sam.env              # Optional GPU-SAM agentic variant (GMU=0.76)
    best.env                               # Preserved EXP-6B rollback baseline (GMU=0.72)
    inkling-cal*.json                      # Confidence calibration tables
  launch/inkling-tp8-s6.sh                 # Per-rank Docker launcher (CPU pinning + RoCEv2 + RDMA)
  ops/                                     # V15/V16 supervisor, watchdog, UMA guard, Prometheus exporter, Grafana, systemd
  patches/                                 # V16-FINAL vLLM Inkling-2 Python overlays (py_compile green)
  kernels/
    *.cu, *.cuh, build_*.sh                # Preserved V12 SM121a CUDA kernels & build scripts
    v16/                                   # Wave-V13..V16 SM121a CUDA sources (0 spills) + build_wave_v16_all.sh
  bench/
    bench_multi_1m.py, bench_rhd_8rank.py, verify_c1_attn_v2_gb10.py
    v16_qa/                                # Harvester 16x4, multimodal API, chaos soak, env auditor, qualification suites
  results/
    checkpoint_2026-11-25_v16_final.json   # Live-verified V16-FINAL permanent checkpoint archive
    checkpoint_2026-11-24_v12_final.json   # Prior V12-FINAL archive
    wave_v12_live_report.md                # Wave-V12 live A/B report
    checkpoint_2026-11-22.json             # Prior EXP-6B archive

# Preserved 4-Spark TP4 archive (2026-11-16):
launch/inkling2-tp4.sh · config/cluster.env.example · patches/ · repro/ · nvidia-bug-report.md
```

## Rollback & Operations

All prior baselines (`live_champion_v12.env`, `best.env`, `patches-s6-v12/`, `patches-s6/`, and V12 `.so`s) remain untouched on all 8 Sparks for instant 1-command rollback:

- **Live Health & Telemetry:** `curl -sf http://spark-one:8010/health` · `cat /dev/shm/inkling_v16_rank0_metrics.json`
- **0-ULP Certification:** `python3 /home/jeweled/inkling-tp8/exact_ink.py` must return `10/10` SHA-256 `1dcfb30680d0252c9c259e5cb2bd02d402b2f6123261f904a0074d3b5b46f8c3`.
- **Instant Rollback to `V12-FINAL`:** Source `/home/jeweled/inkling-tp8/live_champion_v12.env` across ranks `0..7` and relaunch `inkling-tp8-s6.sh`.

See [`STATUS.md`](STATUS.md) for the complete wave-by-wave engineering log and GB10 operational rules.
