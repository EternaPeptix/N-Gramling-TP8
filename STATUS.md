# Status & Engineering Log — Inkling-2-Small on DGX Spark

**Current Status (2026-11-25): PERMANENT LIVE PRODUCTION CHECKPOINT (`V16-FINAL`) on 8× DGX Spark (TP8, `0.0.0.0:8010`)**  
**Promoted Production Config:** `tp8/config/live_champion_v16.env` (`V16-FINAL`, sha256 `413f1ae7275d7d9992e4823793f285d6798c5e12e607a90d78b4a94eb6f2acfa`)  
**Preserved Rollback Baselines:**
- `V12-FINAL`: `tp8/config/live_champion_v12.env` (sha256 `f028df17b5507f0b678214de1f8cd9048f95b44e57676b3698e8c7223d698d81`)
- `V12-SAM`: `tp8/config/live_champion_v12_sam.env` (sha256 `8f2e30861c168a2347385016f59280f32cb80bdebb7e8e1f02e2b5f7f8853d28`)
- `EXP-6B`: `tp8/config/best.env` (sha256 `a4a8aa6235d000b0a6fa4eb38e4b3d34f7acc8a3413b0496ed80788ee7692cb5`)

---

## 1. Live-Verified Permanent Checkpoint Snapshot (`V16-FINAL`)

Measured on live 8-node GB10 silicon (`GMU=0.85`, `MAX_BATCHED=24576`, `0 MiB` new swap, `0.0.0.0:8010` healthy across 8/8 nodes):

- **Bit-exact decode (`exact_ink.py`, 10 prompts, `temp=0`, 384 tok, serial):**
  `1dcfb30680d0252c9c259e5cb2bd02d402b2f6123261f904a0074d3b5b46f8c3` (**10/10 PASS** cold, warm, post-stress, post-1M, and post-production-deploy).
- **`C=1` decode:**
  - `bench.py single` (`1024t`): **`139.07` GM / `144.88` mean** (`code 130.25`, `prose 102.08`, `math 202.30`; beats `V12-FINAL` `137.20` GM by `+1.87`, math `+27.9 tok/s`).
  - `exact_ink.py`: **`148.58 tok/s` warm** / **`148.25` cold** (`147.72` post-deploy live prod).
- **`C=2..8` middle-zone decode:**
  - `C=2` `bench.py` (`256t`): **`136.74 agg`** (**`81.08 mean / 74.85 min`**, `AL 2.44`; **ALL-TIME RECORD**, `+2.14 agg / +2.39 min` vs `V12-FINAL`).
  - `C=4` `bench.py` (`256t`): **`187.64 agg`** (**`58.23 mean / 51.28 min`**, `AL 2.72`; **ALL-TIME RECORD**, `+6.39 agg / +3.08 min` vs `V12-FINAL`).
  - `C=8` `bench.py` (`256t`) / `conc_exact` (`128t`): **`259.49 agg`** (`44.14 mean / 35.08 min`, `AL 2.61`) / **`276.01`** `conc_exact`.
- **`C=32` throughput:**
  - `bench.py` (`256t`): **`567.68 agg`** (**`23.10 mean / 18.90 min`**, `AL 2.41`; **ALL-TIME RECORD**, `+4.66` vs `V12-FINAL`, `+88.7` vs Ds41f `479`).
  - `conc_exact` (`128t`): **`479.63 tok/s`** (`8.54 s`).
- **Long-context prefill & decode (`bench_1m_e2e.py --warm --max-tokens 384`):**
  - `32K`: **`4.30 s`** cold / **`0.30 s`** warm (`14.4×`), **`166.3 tok/s`** decode (`+7.9` vs `V12-FINAL`), **3/3 needles**.
  - `128K`: **`14.49 s`** cold / **`0.44 s`** warm (`32.6×`), **`132.0 tok/s`** decode, **3/3 needles**.
  - `1M` (`1,048,166` tok): **`113.02 s`** cold (`9,274 tok/s`) / **`2.34 s`** warm (`48.3×`), **`90.1 tok/s` decode (`+25.2 tok/s` / `+38.8%` vs `V12-FINAL`'s `64.9`!)**, **3/3 needles**.
- **RLVR Harvester, Multimodal API & Chaos Soak:**
  - `v16_harvester_stress_test.py` (`16×4`, `temp=0.6`, `SyncFree Temp V16`): **`417.78 tok/s`** agg, `64/64` completions, `0` drops, **`1.00`** unique ratio (`0` collapsed), `24.8%` prefix hit.
  - `test_api_multimodal_live_v15.py`: **`7/7` supported API gates PASS** (health, 3/3 temp=0 determinism, SSE streaming + cancel at `93.3 ms` TTFT, `inkling` tool calling + JSON schema, 4-image multi-image + extreme shapes, multimodal soak, post-storm exactness).
  - `v15_chaos_soak.py` (120 mixed requests, `C=16`, 20% forced aborts): **`96/96` OK**, **`24/24` clean aborts**, `leak_ok=True`, `0` KV audit anomalies, `0.354×` drift.
- **1M KV residency, UMA & Telemetry:**
  - **`7.11×` simultaneous 1M KV capacity** (`7,455,548` tokens, `43.58 GiB` KV; `4.68×` Ds41f's `1.52×`), **`0 MiB` new swap** (`63 MiB` baseline on `spark-one`, `13.73 GiB` available UMA, `10.09 GiB` order-13 contiguous blocks), live background telemetry at `/dev/shm/inkling_v16_rank0_metrics.json`.

---

## 2. Wave-by-Wave Progression (8× DGX Spark TP8)

| Milestone | `exact_ink` | `C=1` (`G1` GM) | `C=2` (`bench`) | `C=4` (`bench`) | `C=8` (`conc` / `bench`) | `C=32` (`bench`) | `1M` Cold / Decode | `1M` KV Pool | Key Changes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Starting TP8 baseline | PASS | `~110.9–126.1` | — | — | — | `~360–420.9` | `~250+ s` / — | `~2.5×` | Stock vLLM TP8 + initial RoCE bring-up |
| `EXP-6B` (`best.env`) | PASS (3/3) | `139.5` *(1024t)* | — | — | — | `518.7` *(512t)* | `120.90 s` / — | `4.8×` | Compact Shadow K (`sage2_v5`), `v4` W2 persistent, `RDMA_AR_LEVEL_DB=1`, `BF11_C32` |
| `Inc P7` (Wave-V10) | PASS (10/10) | `134.42` *(256t)* | — | `176.55` | `274.53` / `215.95` | `494.44` | `112.83 s` / — | `7.50×` | `V10` Drafter/Router/Attn/MX7-prefill + 8ch NCCL Ring + `MAX_BATCHED=24576` + `GMU=0.85` |
| `V12-FINAL` | PASS (10/10) | `137.20` | `134.60 [72.5m]` | `181.25 [48.2m]` | `286.03` / `269.66` | `563.02` | `112.07 s` / `64.9` | `7.43×` | `CertHead V12`, `U4_V9`, `DSpark 4-typo`, `L2PF v11`, `PDL_WO_BF11`, `C=4` 48-CTA SWA, `Ctx-Tau` |
| **`V16-FINAL` (LIVE)** | **PASS (10/10)** | **`139.07` (`144.88` mean)** | **`136.74 [74.9m]`** | **`187.64 [51.3m]`** | **`276.01` / `259.49`** | **`567.68`** | **`113.02 s` / `90.1` (+38.8%!)** | **`7.11×`** | **`RDMA V16`, `C48 MoE V16 (LPT=0)`, `CertHead V14 M=128`, `Attn V16 S=48`, `SyncFree Temp V16`, `NonTailSkip`, `V15/V16` guards & `/dev/shm` telemetry** |

---

## 3. Live-Window Fixes Applied During V16 Qualification

1. **Zero-spill SM121a compilation across 6 CUDA sources:** Eliminated all `ptxas` register spills and stack frames in `cert_head_fast_v14.cu`, `pf_nvfp4_w13_fused_v16.cu`, `attn_global_v16.patch`, `pf_dense_fp8_tc_v13.cu`, `pf_fp8_attn_v13.cu`, and `build_moe_v16.sh` (`19/19` `.so` libraries at `0` spills / `0` stack).
2. **`rdma_ar_ext_c32_v16.so` export & loader fix:** Fixed `PyInit_rdma_ar_ext` module symbol resolution in `build_rdma_sconv_v16.sh` and `rdma_all_reduce.py`.
3. **Per-kernel occupancy cache in prefill W2/MX7:** Replaced shared `static int cached_bpm` with a 128-entry per-`kfn` table in `pf_nvfp4_w2_v16.cu` and `pf_mx7_prefill_v16.cu` (closing `CUDA error -1`).
4. **`INKLING_SYNC_FREE_TEMP_V16` qualification init:** Changed `torch.empty` → `torch.zeros` in `sync_free.py` (`ok=0` → `ok=1`).
5. **`release_all_prefill_scratch()` stream sync:** Added `torch.cuda.synchronize()` in `host_pin.py` to prevent async release races.
6. **`attention.py` & `sm12_attn.py` metadata/guard fixes:** Fixed `getattr(md, "num_reqs", None)` fallback, fixed `_kv_slots_d` to use `key_cache.shape[0]` instead of `self.kv_cache.shape[0]` (`2`), and gated `_DECODE_EDGE_GUARD_V15` outside CUDA graph capture (kept OFF on hot decode path).

---

## 4. Hard-Won Operational Rules (GB10 / SM121a)

1. **Always compile SM121a kernels with `-gencode arch=compute_121a,code=sm_121a`** (never `-arch=sm_121a`).
2. **Always keep `--fmad=false` (never `--use_fast_math` / `--ftz=true`)** on all decode kernels for strict 0-ULP `1dcf…f8c3` determinism.
3. **Never let an OOM shatter UMA buddy orders (`orders 9–12`):** Post-OOM fragmentation degrades decode by `-7%` to `-10%`. Monitor `/dev/shm/inkling_v16_rank0_metrics.json` (`buddy.order_13_gib` & `uma.swap_used_mib`).
4. **Never run `nvcc` or standalone GPU tests on any node while benchmarking or serving.**
5. **Keep `INKLING_NVFP4_V16_LPT=0` and `INKLING_MX7_V16_LPT=0`** while `C32_PDL=2` is active (W13/W2 share `ws["work_queue"]` across PDL overlap).
6. **Preserve rollback files untouched:** Never overwrite `/home/jeweled/inkling-tp8/{best.env, live_champion_v12.env, patches-s6/, patches-s6-v12/}` or V12 `.so` libraries.
