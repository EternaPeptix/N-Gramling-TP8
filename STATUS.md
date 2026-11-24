# Status & Engineering Log — Inkling-2-Small on DGX Spark

**Current Status (2026-11-24): LIVE, HEALTHY & 6-FOR-6 ALL-TIME RECORD on 8× DGX Spark (TP8, port `8010`)**  
**Promoted Default Config:** `tp8/config/live_champion_v12.env` (`V12-FINAL`, sha256 `f028df17b5507f0b678214de1f8cd9048f95b44e57676b3698e8c7223d698d81`)  
**Optional GPU-SAM Variant:** `tp8/config/live_champion_v12_sam.env` (`V12-B11`, sha256 `8f2e30861c168a2347385016f59280f32cb80bdebb7e8e1f02e2b5f7f8853d28`)  
**Preserved Rollback Baseline:** `tp8/config/best.env` (`EXP-6B`, sha256 `a4a8aa6235d000b0a6fa4eb38e4b3d34f7acc8a3413b0496ed80788ee7692cb5`)

---

## 1. Qualified Checkpoint Snapshot (`V12-FINAL`, Clean UMA)

Measured after clean 8-node boot (`3,720–3,724` contiguous 32 MiB UMA blocks/node, `4/4` IB Active, `GMU=0.85`, `MAX_BATCHED=24576`):

- **Bit-exact decode (`exact_ink.py`, 10 prompts, `temp=0`, 384 tok, serial):**
  `1dcfb30680d0252c9c259e5cb2bd02d402b2f6123261f904a0074d3b5b46f8c3` (**10/10 PASS cold & warm**).
- **`C=1` decode:**
  - `exact_ink.py`: **`151.66 tok/s`** (`21.74 s`, `+4.54 tok/s` vs `EXP-7C`, `+4.08` vs `Inc P7`).
  - `bench.py single`: **`137.20 tok/s` G1 GM** (`code 131.0`, `prose 106.2`, `math 174.4`; beats Ds41f `132.0–134.1`).
- **`C=2..8` middle-zone decode:**
  - `C=2` `bench.py` (`256t`): **`134.60 agg`** (`80.39 mean / 72.46 min`, `AL 2.33`).
  - `C=4` `conc_exact` (`128t`): **`200.78 tok/s`** (`2.55 s` warm, `196.92` cold; `+13.92 tok/s` vs `EXP-7C`).
  - `C=4` `bench.py` (`256t`): **`181.25 agg`** (`57.52 mean / 48.20 min`, `AL 2.50`).
  - `C=8` `conc_exact` (`128t`): **`286.03` cold / `279.78` warm** (`3.58 s`; `+20.06 tok/s` vs `EXP-7C`).
  - `C=8` `bench.py` (`256t`): **`269.66 agg`** (**`45.12 mean / 36.36 min`**, `AL 2.62`; **`+53.71 tok/s` / `+24.9%`** vs `Inc P7`, beats Ds41f `pinx` min `35.4`).
- **`C=32` throughput:**
  - `conc_exact` (`128t`): **`496.72 tok/s`** (`8.24 s` warm, `492.90` cold; `+32.85 tok/s` vs `EXP-7C`).
  - `bench.py` (`256t`): **`563.02 agg`** (**`22.90 mean / 18.88 min`**, `AL 2.35`; **`+68.58 tok/s` / `+13.9%`** vs `Inc P7`, `+84 tok/s` vs Ds41f `479`).
- **Long-context prefill (`bench_1m_e2e.py --warm --max-tokens 256`):**
  - `32K`: **`4.30 s`** cold / **`0.32 s`** warm (**`158.4 tok/s`** decode, **3/3 needles**).
  - `128K`: **`14.45 s`** cold / **`0.44 s`** warm (**`134.0 tok/s`** decode, **3/3 needles**).
  - `1M`: **`112.07 s`** cold (`9,354 tok/s`) / **`2.36 s`** warm (`64.9 tok/s` decode, **3/3 needles**; `1.51×` faster than Ds41f `169.1 s`).
- **1M KV residency & CUDA graphs:**
  - **`7.43×` simultaneous 1M KV capacity** (`7,791,866` tokens, `1.84 GiB` CUDA graphs, `5 s` capture, `HEALTH_200_OK`; `4.9×` Ds41f's `1.52×`).
- **WikiText-2 perplexity (`lm-eval 0.4.13`, `wikitext-2-raw-v1`, test, `max_length=2048`):**
  - `word_perplexity = 5.8052`, `byte_perplexity = 1.3894`, `bits_per_byte = 0.4745` (`0.4936` detok).

---

## 2. Wave-by-Wave Progression (8× DGX Spark TP8)

| Milestone | `exact_ink` | `C=1` (`exact_ink` / `G1`) | `C=4` (`conc` / `bench`) | `C=8` (`conc` / `bench`) | `C=32` (`conc` / `bench`) | `1M` Cold Prefill | `1M` KV Pool | Key Changes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Starting TP8 baseline | PASS | `~110.9–126.1` | — | — | `~360–420.9` | `~250+ s` | `~2.5×` | Stock vLLM TP8 + initial RoCE bring-up |
| `EXP-6B` (`best.env`) | PASS (3/3) | `—` / `139.5` *(1024t)* | — | — | `—` / `518.7` *(512t)* | `120.90 s` | `4.8×` | Compact Shadow K (`sage2_v5`), `v4` W2 persistent, `RDMA_AR_LEVEL_DB=1`, `BF11_C32` |
| `EXP-7C` | PASS (10/10) | `147.12` / `—` | `186.86` / `—` | `265.97` / `—` | `463.87` / `—` | `118.86 s` | `5.11×` | Wave-V4..V8 MoE/RDMA/Attention refinements |
| `Inc P7` (Wave-V10) | PASS (10/10) | `147.58` / `134.42` | `194.68` / `176.55` | `274.53` / `215.95` | `487.62` / `494.44` | `112.83 s` | `7.50×` | `V10` Drafter/Router/Attn/MX7-prefill + 8ch NCCL Ring + `MAX_BATCHED=24576` + `GMU=0.85` |
| **`V12-FINAL` (LIVE)** | **PASS (10/10)** | **`151.66` / `137.20`** | **`200.78` / `181.25`** | **`286.03` / `269.66`** | **`496.72` / `563.02`** | **`112.07 s`** | **`7.43×`** | **`CertHead V12` (`63/63`), `U4_V9` (`MIN_M=9`), `DSpark 4-typo`, `L2PF v11`, `PDL_WO_BF11`, `FastReplay v11`, `C=4` 48-CTA SWA, `Ctx-Tau`, clean UMA** |

---

## 3. Hard-Won Operational Rules (GB10 / SM121a)

1. **Always compile SM121a kernels with `-gencode arch=compute_121a,code=sm_121a`** (not `-arch=sm_121a`). On CUDA 13.0, `-arch=sm_121a` also emits virtual `compute_121`, doubling compile time and rejecting `cvt.e2m1x2` PTX instructions.
2. **Always keep `--fmad=false` (and never `--use_fast_math` / `--ftz=true`)** on all decode kernels to preserve strict 0-ULP `1dcf…f8c3` determinism.
3. **Never let an OOM shatter UMA buddy orders (`orders 9–12`):** Post-OOM UMA fragmentation costs `-7.3%` to `-10.1%` across every concurrency even when GPUs report `P0` and `0x0` throttle. Rebooting all 8 Sparks (`15 s`) restores `3,720+` 32 MiB blocks/node immediately.
4. **Never co-enable `INKLING_MX6_EXACT=1` (`FREE_RAW=0`) and `INKLING_MX7_EXACT=1` (`FREE_RAW=1`) together:** Packing both `MX6` (`10 GiB`) and `MX7` (`12 GiB`) + raw MXFP8 triggers simultaneous 8-node UMA OOM.
5. **Always set `TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas` before `import torch`** in standalone 0-ULP tests.
6. **Never run `nvcc` or standalone GPU tests on any sibling node while benchmarking.**
7. **Guard `PF_GLOBAL_ZERO_SYNC=1` with bounded inflight DBO depth (`INKLING_PREFILL_DBO_MAX_INFLIGHT_LAYERS=2`).**

---

## 4. Wave-V13 Offline Optimization Roadmap (Targeting `C=1..8` & `<95 s` 1M Prefill)

1. **Forensic 0-ULP Fix for `INKLING_MOE_S8=1` + `c1_v12` & Max-Occupancy `c48_v12`:**
   - Isolate the last 1-ULP difference in `S8` (`a5dc798c` `2/10` → `1dcfb306` `10/10`) to unlock `+8–10 tok/s` `C=1` (`>160`) and `+15–20 tok/s` `C=2`.
   - Rebuild already-0-ULP-verified `c48_v12` (`V12-B6` `10/10`) at max 0-spill occupancy (`4 8 6 4 3` / `5 8 6 5 4`, restoring `BN16=8` and `W2=6`) to unlock `+10–20 tok/s` at `C=4..32`.
2. **Fix `INKLING_HETERO_BUDGET_SPEC=1` Drain Deadlock + Zero-Tax GPU SAM:**
   - Eliminate the `K=5→7` schedule transition rank-divergence / graph-key hang in `speculator.py` + `dtgraph.py` so `bench.py C=4` captures the `+11.1%` mean / `+8.7%` min lift (`min > 55 tok/s`, beating Ds41f `4a04b175`) with zero hangs.
   - Fuse/gate GPU SAM (`INKLING_SAM_MAX_B=4` or single-kernel launch) to keep `+17.5 tok/s` warm `C=1` (`155.7+`) with zero `C=8/32` per-step overhead.
3. **Decode-Only CUDA Graph Gate for `BF11 V12 BM=48`:**
   - Strictly separate decode CUDA graph capture from prefill CUDA graph capture (`M=17..128`) so `BF11 V12` never alters `exact_ink` prefill (`10/10 1dcfb306`) while reclaiming `+15–20 tok/s` at `C=4`.
4. **Surgical `C=2..8` RDMA/SConv & Zero-Swap `GMU=0.85`:**
   - Isolate winning `C=2..8` RDMA micro-opts cleanly without `V12-B2` regressions, and free dead init-time host copies so `GMU=0.85` runs at `0 MiB` swap.
5. **Sub-95s 1M Cold Prefill & `>85 tok/s` 1M Decode:**
   - Fix `v11/attn_s48` split-compile (`cicc` hang) and optimize `128K–1M` global attention + FP8 RS overlap.

---

## 5. Historical Note (2026-11-16 — 4-Spark TP4 Bring-Up)

The earlier 4-Spark TP4 bring-up (`launch/inkling2-tp4.sh`, `repro/gb10-livelock-repro.py`, `nvidia-bug-report.md`) hit a GB10 CUDA driver livelock near `94.5 GiB` per node when fitting the ~380 GB mixed NVFP4/MXFP8 checkpoint on 4× 121.6 GiB Sparks. Moving to **8× DGX Spark (TP8)** halves per-rank weight residency, avoids the high-UMA livelock regime, and delivers `7.43×` simultaneous 1M contexts alongside 6-for-6 0-ULP performance records.
