# Inkling-2-Small on 8× NVIDIA DGX Spark (TP8, SM121a)

Production-grade, **100% 0-ULP bit-exact** inference stack for
[`thinkingmachines/Inkling-2-Small`](https://huggingface.co/thinkingmachines/Inkling-2-Small)
(594B total / 17B active MoE, 384 routed experts top-8, 40 NVFP4 local SWA-1024 layers + 8 MXFP8 global layers, 1M context, 5-layer DSpark speculator)
on **8× NVIDIA DGX Spark (GB10, SM121a)** over **4× 200GbE ConnectX-7 RoCEv2**.

> **Status (2026-11-24): LIVE & 6-FOR-6 ALL-TIME RECORD on 8/8 nodes (`tp8/config/live_champion_v12.env`, `V12-FINAL`).**
> This repository started as a fork of `msuiche/dgx-inkling-small` (which documented the 4-Spark TP4 driver livelock — preserved under `launch/inkling2-tp4.sh`, `repro/`, and `nvidia-bug-report.md`). On 8× DGX Spark (`TP8`), `V12-FINAL` achieves **10/10 0-ULP deterministic decode (`1dcf…f8c3`)**, **`151.66 tok/s` C=1**, **`200.78 tok/s` C=4**, **`286.03 tok/s` C=8** (`269.66` `bench.py`), **`563.02 tok/s` C=32** (`496.72` `conc_exact`), **`112.07 s` 1M cold prefill (`9,354 tok/s`)**, and **`7.43×` simultaneous 1M-token KV residency (`7.79M` tokens, `1.84 GiB` CUDA graphs)**.

---

## Qualified Checkpoint Results (`V12-FINAL` → `tp8/config/live_champion_v12.env`)

All numbers below were measured on **one unified live server** on clean UMA (`GMU=0.85`, `MAX_BATCHED=24576`, port `8010`, stock 2000 MHz GB10 clocks, `live_champion_v12.env` sha256 `f028df17b550…`).

| Metric / Gate | `EXP-6B` (`best.env`) | `EXP-7C` / `Inc P7` | **Current (`V12-FINAL`)** | How Measured |
|---|---:|---:|---:|---|
| **1. Bit-Exact Decode (0-ULP)** | `1dcf…f8c3` (3/3) | `1dcf…f8c3` (10/10) | **`1dcf…f8c3` (10/10 cold+warm)** | `exact_ink.py` (10 prompts, `temp=0`, 384 tok, serial). Full SHA-256: `1dcfb30680d0252c9c259e5cb2bd02d402b2f6123261f904a0074d3b5b46f8c3` |
| **2a. `C=1` `exact_ink.py`** | — | `147.12` / `147.58` | **`151.66 tok/s`** (`21.74 s`) | `exact_ink.py` cold & warm serial wall speed (`+4.54 tok/s` vs `EXP-7C`) |
| **2b. `C=1` `bench.py single` (`G1`)** | `139.5` *(1024t)* | `134.42` *(256t)* | **`137.20 tok/s`** (`131.0 / 106.2 / 174.4`) | `bench.py single` (`code / prose / math` GM, beats Ds41f `132.0–134.1`) |
| **3a. `C=2` `bench.py` (`256t`)** | — | — | **`134.60 agg`** (`80.39 mean / 72.46 min`) | `bench.py conc --conc 2 --max-tokens 256` (`AL 2.33`) |
| **3b. `C=4` `conc_exact` (`128t`)** | — | `186.86` / `194.68` | **`200.78 tok/s`** (`2.55 s` warm) | `conc_exact` (`C=4, 128t`, `+13.92 tok/s` / `+7.5%` vs `EXP-7C`) |
| **3c. `C=4` `bench.py` (`256t`)** | — | `176.55` (`56.1 / 48.2`) | **`181.25 agg`** (`57.52 mean / 48.20 min`) | `bench.py conc --conc 4 --max-tokens 256` (`AL 2.50`) |
| **3d. `C=8` `conc_exact` (`128t`)** | — | `265.97` / `274.53` | **`286.03` cold / `279.78` warm** (`3.58 s`) | `conc_exact` (`C=8, 128t`, `+20.06 tok/s` / `+7.5%` vs `EXP-7C`) |
| **3e. `C=8` `bench.py` (`256t`)** | — | `215.95` (`38.8 / 30.6`) | **`269.66 agg`** (**`45.12 mean / 36.36 min`**) | `bench.py conc --conc 8 --max-tokens 256` (`AL 2.62`, **`+24.9%`** vs `Inc P7`) |
| **4a. `C=32` `conc_exact` (`128t`)** | — | `463.87` / `487.62` | **`496.72 tok/s`** (`8.24 s` warm) | `conc_exact` (`C=32, 128t`, `+32.85 tok/s` / `+7.1%` vs `EXP-7C`) |
| **4b. `C=32` `bench.py` (`256t`)** | `518.71` *(512t)* | `494.44` (`20.5 / 16.9`) | **`563.02 agg`** (**`22.90 mean / 18.88 min`**) | `bench.py conc --conc 32 --max-tokens 256` (`AL 2.35`, **`+13.9%`** vs `Inc P7`) |
| **5a. `32K` / `128K` Prefill (`256t`)** | — | `4.34 s` / `15.05 s` | **`4.30 s` (`158.4` dec)** / **`14.45 s` (`134.0` dec)** | `bench_1m_e2e.py --warm --max-tokens 256`, **3/3 needles** (warm `0.32 s / 0.44 s`) |
| **5b. `1M` Cold / Warm Prefill** | `120.90 s` / `2.95 s` | `118.86 s` / `112.83 s` | **`112.07 s` (`9,354 tok/s`) / `2.36 s` warm** | `bench_1m_e2e.py --ctx 1048500 --warm --max-tokens 256`, `64.9 tok/s` dec, **3/3 needles** |
| **6. 1M KV Residency & Graphs** | `4.8×` (`5.07M` tok) | `5.11×` / `7.50×` | **`7.43×` (`7,791,866` tok, `1.84 GiB` graphs)** | Simultaneous 1M KV pool (`5 s` graph capture, `HEALTH_200_OK`) |
| **WikiText-2 Quality** (`lm-eval`) | `5.8052` / `0.4745` | `5.8052` / `0.4745` | **`word_ppl = 5.8052` · `BPB = 0.4745`** | `wikitext-2-raw-v1`, test, 62 docs, `max_length=2048` (0-ULP invariant) |

- **Optional Agentic/Shared-Prefix Variant (`tp8/config/live_champion_v12_sam.env`, `V12-B11`):** Enables GPU Suffix-Automaton Hybrid Speculation (`INKLING_SAM_HYBRID=1`, `GMU=0.76`, zero swap, **10/10 `1dcf…f8c3`**) for **`155.69 tok/s`** warm `C=1` and **`184.17 tok/s`** `C=4` on repetitive/code workloads.
- Full machine-readable archives: [`tp8/results/checkpoint_2026-11-24_v12_final.json`](tp8/results/checkpoint_2026-11-24_v12_final.json) · [`tp8/results/wave_v12_live_report.md`](tp8/results/wave_v12_live_report.md) · [`tp8/results/checkpoint_2026-11-22.json`](tp8/results/checkpoint_2026-11-22.json).

---

## Hardware Topology (8× DGX Spark)

- **GPU:** NVIDIA GB10 (`SM121a`), 48 SMs, 99 KiB SMEM/SM, **244 GB/s** sustained UMA LPDDR5X, 24 MiB L2, 121.63 GiB usable UMA/node (maintain clean contiguous 2 MiB–32 MiB UMA blocks; post-OOM fragmentation degrades decode by `-7%` to `-10%`).
- **CPU:** 20-core Grace ARM (`0..4, 10..14` Cortex-A725 LITTLE; `5..9, 15..19` Cortex-X925 BIG). Container pinned to `--cpuset-cpus=5-9,15-19` with `SCHED_FIFO` RDMA proxies on CPUs `9` & `19` (`tp8/patches/host_pin.py`).
- **Fabric:** 4× 200GbE ConnectX-7 RoCEv2 (`~25.8 GB/s` aggregate wire, `NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0,rocep1s0f1,roceP2p1s0f1`, `NCCL_IB_QPS_PER_CONNECTION=2`, `NCCL_IB_PCI_RELAXED_ORDERING=1`).

---

## What We Built & Proven Wins

### 1. Wave-V9 through Wave-V12 `C=1..8` & Unified Serving Upgrades (Live in `V12-FINAL`)
All target decode kernels are compiled with **`-gencode arch=compute_121a,code=sm_121a --fmad=false`** (no `--ftz=true`, no `--use_fast_math`, verified `0` register spills) and gated by `exact_ink.py` (`1dcf…f8c3`, 10/10):
- **`CertHead V12` (`tp8/kernels/cert_head_fast_v12.cu`):** Verified `63/63` exact (`ref=b1_exact`) across `M=2..64`, eliminating the `-717 us/step` silent FP32 `lm_head` fallback at `C=4..8` at 10/10 `1dcfb306`.
- **`Drafter U4 TC V9` (`tp8/kernels/drafter_u4_tc_v9.cu`, `MIN_M=9`) + `DSpark V10` (`dspark_markov_v10.cu`, `drafter_attn_v10.cu`):** Saves `-260 us/step` at `C=2..8` while leaving `B=1` untouched, and fixes 4 runtime wiring typos in `dspark_ops.py` (`tc_skip=1/1, es_tc=1, top2_id=1`), unlocking native Batch-All Early-Stop and fused tail/epilogue kernels.
- **`L2PF v11` Adaptive `C=2..8` QKVR+Tail + `PDL_WO_BF11=1` + `DTGraph/SyncFree` Fast-Replay v11:** Directly drove `+8.7%` `C=4` and `+8.6%` `C=8` gains in isolated same-hour A/B (`V12-B3` vs `P7_CTRL`).
- **`C=4` 48-CTA SWA (`fused_decode_attn_exact_sm121_v10.cu`, `V8_SWA_MIN_B=5`, `C28_SWA_B34_SPLITS=4`):** Reaches 100% SM utilization (48 CTAs across 48 SMs) on `B=3..4`, cutting critical-path SWA passes/SM by `-44%`.
- **Context-Aware Confidence Pruning (`INKLING_CTX_ADAPTIVE_TAU=1, FLOOR=0.04`):** Dynamically tightens pruning at `512K–1M` contexts (`+1.5–3.2%` decode) with zero overhead at short prompts.
- **0-ULP RDMA AllReduce V9 (`rdma_ar_c32_v9.cu`), MoE & Router (`nvfp4_ws5_c32.cu`, `mxfp8_mx7_c48_v9.cu`, `mxfp8_mx7_c1_v5.cu`, `router_bf12_fused_v10.cu`), Fused ShortConv+AR (`fused_sconv_ar_c32_exact.cu`), and BF11 Dense GEMMs (`bf11_flip_exact_c32.cu`).**

### 2. `112.07 s` 1M Cold Prefill & `7.43×` Simultaneous Multi-1M KV Residency
- **Compact Logical-Block FP8 Shadow K (`global_attn_sage2_sm121_v5_zspill.cu`):** Decouples Shadow K to a constant **`1.016 GiB`**, enabling `GMU=0.85` with **`7.43×` simultaneous 1M KV capacity** (`7.79M` tokens) and **`1.84 GiB`** CUDA graphs (`4.9×` more 1M KV capacity than DeepSeek-V4.1 on the same 8-Spark cluster).
- **Persistent NVFP4 W2 + V10 Non-MoE + 8-Channel NCCL Ring (`pf_moe_native_sm121_v4.cu`, `pf_attn_nonmoe_v4_sm121.cu`, `pf_mx7_prefill_v10.cu`, `MAX_BATCHED=24576`):** Cuts 1M cold prefill from `120.90 s` (`EXP-6B`) → **`112.07 s` (`9,354 tok/s`, warm `2.36 s`, 3/3 needles)** — `1.51×` faster than Ds41f (`169.1 s`).
- **Bounded Depth-2 DBO Allocator Ring (`INKLING_PREFILL_DBO_MAX_INFLIGHT_LAYERS=2`):** Caps cross-stream `record_stream` limbo at `0.90 GB` (preventing 21.75 GB UMA page-fault thrash).

### 3. Verified R&D Kernels Included (Non-Default / Wave-V13 Offline Targets)
- **`nvfp4_ws5_c48_v12.cu` & `nvfp4_ws5_c1_v12.cu`:** `c48_v12` Section A passed **10/10 `1dcf…f8c3`** on live silicon (`V12-B6`), closing the Wave-V10 `dadc285c` SiTU bug; queued for max-occupancy `4 8 6 4 3` / `5 8 6 5 4` rebuild and `S8=1` 1-ULP isolation.
- **`bf11_flip_exact_v12.cu`:** `+17.8` `C=4` / `+23.1` `C=32` standalone potential; queued for decode-only CUDA graph capture gating (fixing `M=17..128` prefill graph leak).
- **`mxfp8_mx6_v12.cu` (`MX6-Huff7`) & `pf_rhd_comm_sm121.cu` (`rhd_int8`):** Both 0-ULP / standalone verified; kept non-default as `MX7 v5` and 8ch NCCL BF16 Ring won end-to-end on GB10.

---

## Dead Ends Proven & Closed

| Idea Tested | Measured Outcome | Why |
|---|---|---|
| **`MX6-Huff7 V12` (`mxfp8_mx6_v12.cu`)** | `125.6 / 169.0 / 231.7 / 374.8` vs `MX7 v5` (`137.0 / 186.9 / 268.1 / 453.1`) | 0-ULP verified (`1dcfb306` 10/10 with `MX7=0`), but Huffman decode ALU overhead outweighs memory savings on GB10 |
| **`RDMA v12` full overlay (`rdma_ar_c32_v12.cu`)** | `-15%` across `C=1..32` | `h64_q8` drafter AR / `P2_SLEEP_NS=32` / `ADAPTIVE_BLOCKS=1` regressed vs `rdma_ar_c32_v9.cu` |
| **7-peer P2P INT8/FP8 All-to-All RS** | `130.4–138.6 s` vs `112.1 s` | 7-to-1 RoCEv2 incast vs 8-channel Ring |
| **C=32 `K=4 / 5 / 7` & Tier-2 Live Compaction** | `K=4` `-2.57%`, `K=7` `-47 tok/s` | Extra verify rows & prose stragglers outweigh AL gain; `K=3` is optimal |
| **BF11 Dense GEMMs above `M=64`** | `+1.7 to +7.4 ms/step` slower | Above `M=64`, cuBLAS BF16 Tensor Cores beat custom BF11 unpack on GB10 |
| **`RDMA_AR_WIRE_BF12=1` / `RDMA_AR_ROWS_P2=3` on C=1** | `-1.3` to `-3.2 tok/s` | Extra pack/wire latency hurts C=1 (`M=8`) |

---

## Repository Layout

```text
tp8/
  config/
    live_champion_v12.env                  # Promoted V12-FINAL 6-for-6 TP8 default (GMU=0.85)
    live_champion_v12_sam.env              # Optional GPU-SAM agentic variant (GMU=0.76, zero swap)
    best.env                               # Preserved EXP-6B rollback baseline (GMU=0.72)
    inkling-cal*.json                      # Confidence calibration tables
  launch/inkling-tp8-s6.sh                 # Per-rank Docker launcher (CPU pinning + RoCEv2 + RDMA)
  patches/                                 # 54 vLLM Inkling-2 Python overlays + MANIFEST (py_compile green)
  kernels/                                 # SM121a (GB10) CUDA kernels (.cu/.cuh) + build scripts
  bench/                                   # Multi-1M, 8-rank RHD comm, and 0-ULP attention verifiers
  results/
    checkpoint_2026-11-24_v12_final.json   # Clean-UMA V12-FINAL verified metrics archive
    wave_v12_live_report.md                # Full Wave-V12 live A/B progression & verdicts
    checkpoint_2026-11-22.json             # Prior EXP-6B baseline archive

# Preserved 4-Spark TP4 archive (2026-11-16):
launch/inkling2-tp4.sh · config/cluster.env.example · patches/ · repro/ · nvidia-bug-report.md
```

## Quickstart (8× DGX Spark TP8)

1. Build the SM121a `.so` libraries on one quiet GB10 node using `-gencode arch=compute_121a,code=sm_121a` (using `--fmad=false` on all decode kernels, 0 spills) and distribute to `/home/<user>/inkling-tp8/rdma/` on all 8 nodes.
2. Sync `tp8/patches/` to `/home/<user>/inkling-tp8/patches-s6-v12/` and `tp8/config/live_champion_v12.env` + `tp8/launch/inkling-tp8-s6.sh` to all 8 nodes.
3. Launch ranks `7..0` via `tp8/launch/inkling-tp8-s6.sh` (`vllm/vllm-openai:inkling-2-small` image) with `live_champion_v12.env` sourced and verify:
   - Health: `curl -sf http://127.0.0.1:8010/health`
   - Bit-exactness: `exact_ink.py` must match `1dcfb30680d0252c9c259e5cb2bd02d402b2f6123261f904a0074d3b5b46f8c3` (`10/10`).
   - Long-context & KV residency: `bench_1m_e2e.py --ctx 1048500 --warm --max-tokens 256` and `python3 tp8/bench/bench_multi_1m.py --sessions 4`.

See [`STATUS.md`](STATUS.md) for the full wave-by-wave engineering log, GB10 operational rules, and Wave-V13 offline optimization roadmap.
