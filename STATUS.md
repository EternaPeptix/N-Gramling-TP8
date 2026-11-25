# Status & Engineering Log — Inkling-2-Small on DGX Spark

**Current Status (2026-11-25): PERMANENT LIVE PRODUCTION CHECKPOINT (`V17.2` live on 8× DGX Spark TP8 `0.0.0.0:8010`; `V17.3` GPU SAM + `dsh-sptc` compat committed & staged for next restart)**  
**Promoted Production Config:** `tp8/config/live_champion_v17.env` (`V17.2` live: `INKLING_SYNC_FREE_TEMP_V16_MIN_B=64`, `INKLING_VERIFY_W_TEMP=1`, `RDMA_AR_EXT=/rdma/rdma_ar_ext_c32_v17.so`; `V17.3` staged: GPU SAM `INKLING_SAM_HYBRID=1` for `B=1..8` + `inkling_renderer.py` `thinking`/`enable_thinking` compat for `dsh-sptc`)  
**Launcher & Live Ops:** `tp8/ops/inkling-tp8-v17.sh` · `tp8/ops/spark-top.py` · `v16_metrics_exporter.py (:9108)`  
**Preserved Rollback Baselines:**
- `V16-FINAL`: `tp8/config/live_champion_v16.env` (sha256 `413f1ae7275d7d9992e4823793f285d6798c5e12e607a90d78b4a94eb6f2acfa`)
- `V12-FINAL`: `tp8/config/live_champion_v12.env` (sha256 `f028df17b5507f0b678214de1f8cd9048f95b44e57676b3698e8c7223d698d81`)
- `V12-SAM`: `tp8/config/live_champion_v12_sam.env` (sha256 `8f2e30861c168a2347385016f59280f32cb80bdebb7e8e1f02e2b5f7f8853d28`)
- `EXP-6B`: `tp8/config/best.env` (sha256 `a4a8aa6235d000b0a6fa4eb38e4b3d34f7acc8a3413b0496ed80788ee7692cb5`)

---

## 1. Live-Verified Permanent Checkpoint Snapshot (`V17-FINAL`)

Measured on live 8-node GB10 silicon (`GMU=0.85`, `MAX_BATCHED=24576`, `0 MiB` new swap, `0.0.0.0:8010` healthy across 8/8 nodes, `88 s` cold cluster boot):

- **Bit-exact decode (`exact_ink.py`, 10 prompts, `temp=0`, 384 tok, serial):**
  `1dcfb30680d0252c9c259e5cb2bd02d402b2f6123261f904a0074d3b5b46f8c3` (**10/10 PASS** cold, warm, post-stress, post-chaos, and post-production-deploy).
- **Batching & Decode Curve (`C=1..32`):**
  - `C=1` `bench.py single` (`1024t`, `temp=0.0`): **`138.24` GM / `143.94` mean** (`code 129.65`, `prose 101.66`, `math 200.66`).
  - `C=1` `temp > 0` streaming (`V17.1` fix: `INKLING_SYNC_FREE_TEMP_V16_MIN_B=64` + `INKLING_VERIFY_W_TEMP=1`, 512t):
    - `code`: `t=0.6 p=0.95` **`126.0` (`129.1` dec, `21.90 ms/step`)** *(was `100.9`, `+25.2%`)* | `t=0.6 p=0.95 min_p=0.05` **`129.3` (`132.9` dec)** | `t=0.6 p=1.0` **`127.9` (`131.8` dec)** *(was `77.9`, `+66.2%`)* | `t=1.0 p=1.0` **`122.2` (`125.1` dec)** *(was `74.6`, `+63.8%`)*.
    - `prose`: `t=0.6 p=0.95` **`92.1` (`17.79 ms/step`)** *(was `65.1`, `+41.5%`)* | `t=1.0 p=1.0` **`90.3`** *(was `58.2`, `+55.2%`)*.
    - `math`: `t=0.6 p=0.95` **`181.2` dec** *(was `175.1`)* | `t=0.6 p=1.0` **`176.2` dec** *(was `136.9`, `+28.7%`)* | `t=1.0 p=1.0` **`170.3` dec** *(was `126.1`, `+35.1%`)*.
  - `C=2` `bench.py` (`256t`): **`131.33 agg`** (`78.59 mean / 70.74 min`).
  - `C=4` `bench.py` (`256t`) / `conc_exact` (`128t`): **`188.66 agg`** (`59.92 mean / 51.58 min`) / **`199.22` `conc_exact`** (**ALL-TIME RECORDS**).
  - `C=8` `bench.py` (`256t`) / `conc_exact` (`128t`): **`258.95 agg`** (`43.20 mean / 35.22 min`) / **`277.51` `conc_exact`** (**ALL-TIME `conc_exact` RECORD**).
  - `C=9 / 12 / 14 / 16` (`256t`, cliff eliminated): **`266.33` / `311.52` / `320.64` / `380.23 tok/s`**.
  - `C=32` `bench.py` (`256t`) / `conc_exact` (`128t`): **`550.01 agg`** (`22.68 mean / 18.56 min`) / **`479.06` `conc_exact`** (`8.55 s`).
- **Mixed Prefill+Decode & Short-Skip (`InklingScheduler` + `--scheduling-policy priority`):**
  - `bench_v17_mixed_sched.py` (`C=8` active decoders + cold `128K` prefill + urgent short chat):
    - Urgent short-chat TTFT: **`0.837 s`** (`2.67 s` total; `short_skip=1`, down from `19.01 s` without short-skip-first and `~14–113 s` stock).
    - `C=8` decode throughput during `128K` cold prefill: **`133.41 tok/s agg`** (`16.68 mean / 14.50 min`; `pure_decode=103`, `pure_prefill=25`, `pure_fallback=0`).
- **Armed `SYNC_FREE_TEMP_V16` + `min_p` + RLVR Harvester (`16×4`, `temp=0.6`):**
  - `v16_harvester_stress_test.py` (`16×4`, `max_tokens=512`): **`563.77 tok/s`** (**`+35.0%`** over V16's `417.78 tok/s`; peak **`650.6 tok/s`**), `64/64` completions, `0` drops, `1.00` unique ratio, `0.66` Jaccard distance, `0` collapsed groups.
  - `min_p=0.05` under DSPARK spec decode: **HTTP 200** (`74.44 tok/s`), **`12/12` PASS** on `temp=0.6 + top_p=0.95 + min_p=0.05` math accuracy gate.
- **Long-Context Prefill & Decode (`bench_1m_e2e.py --warm --max-tokens 384`):**
  - `32K`: **`4.42 s`** cold / **`0.29 s`** warm (`15.3×`), **`134.2 tok/s`** decode, **3/3 needles**.
  - `128K`: **`14.70 s`** cold / **`0.43 s`** warm (`34.2×`), **`124.8 tok/s`** decode, **3/3 needles**.
  - `1M` (`1,048,144` tok): **`111.76 s`** cold (**`-1.26 s` faster than V16**; `9,378.8 tok/s`) / **`2.36 s`** warm (`47.4×`), **`75.7 tok/s`** decode, **3/3 needles**.
- **APIServer CPU, Chaos Soak & Cluster Ops:**
  - `_inkling_arg_value_extender` (`tp8/patches/inkling_parser.py`): **`57.1×` faster** (`1551.1 ms → 27.15 ms` on 6 KB streaming tool arg, `4/4` 100% identical deltas).
  - `InklingRenderer` (`tp8/patches/inkling_renderer.py`): `INKLING_INCR_TOK=1` (`128 MiB` 0-ULP LRU) + `V17.3` `chat_template_kwargs.thinking` / `enable_thinking` → `reasoning_effort` compat for `dsh-sptc` `llm_query` (`thinking: false → "none"`, verified 8/8 unit tests; explicit `reasoning_effort` wins; 0-ULP neutral when unset).
  - `V17.3` GPU SAM (`tp8/config/live_champion_v17.env`): `INKLING_SAM_HYBRID=1` (`gpu`, `MAX_B=8`, `MAX_SLOTS=32`, `ALLOW_TEMP=1`, `IMAGE_SAFE_V16=1`, `PREFIX_SEED_V16=1`, `LPC_CAP=262144`, `SHARED_POOL=0`) paired with `dsh-sptc` (`jit: true`, `layer3: true`; live SSE replay verified 2/2 mid-stream launches `660 ms` / `229 ms` early, `0` wasted).
  - `v15_chaos_soak.py` (`120` reqs, `C=16`, `20%` forced aborts): **`96/96` OK**, **`24/24` clean aborts**, **`7.2 req/s`**, `drift=0.543×`, `leak_ok=True`, `0` KV anomalies.
  - True **`7.0 GiB`** OS UMA reserve (`6.86×` 1M KV pool, `7,196,613` tokens, `37.88 GiB`), `0 MiB` new swap, `32/32` CX-7 HCAs ACTIVE (`0` IB errors), `sconv=1.0` phantom telemetry fixed (`FAST_KV_FREE=1`), `:9108` Prometheus exporter & `spark-top.py` live.

---

## 2. Wave-by-Wave Progression (8× DGX Spark TP8)

| Milestone | `exact_ink` | `C=1` (`G1` GM) | `C=4` (`bench` / `conc`) | `C=8` (`bench` / `conc`) | `C=32` (`bench` / `conc`) | `16×4` Harvester (`temp=0.6`) | `1M` Cold / Warm | Key Changes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Starting TP8 baseline | PASS | `~110.9–126.1` | — | — | `~360–420.9` / — | — | `~250+ s` / — | Stock vLLM TP8 + initial RoCE bring-up |
| `EXP-6B` (`best.env`) | PASS (3/3) | `139.5` | — | — | `518.7` / — | — | `120.90 s` / `2.95 s` | Compact Shadow K (`sage2_v5`), `v4` W2, `RDMA_AR_LEVEL_DB=1` |
| `V12-FINAL` | PASS (10/10) | `137.20` | `181.25` / — | `269.66` / `286.03` | `563.02` / — | Synced PyTorch | `112.07 s` / `2.36 s` | `CertHead V12`, `U4_V9`, `DSpark`, `L2PF v11`, `Ctx-Tau` |
| `V16-FINAL` | PASS (10/10) | `139.07` | `187.64` / `191.76` | `259.49` / `276.01` | `567.68` / `479.63` | `417.78` *(disarmed)* | `113.02 s` / `2.34 s` | `RDMA V16`, `C48 MoE V16`, `CertHead V14`, `Attn V16 S=48`, `NonTailSkip` |
| **`V17-FINAL` (LIVE)** | **PASS (10/10)** | **`138.24`** | **`188.66` / `199.22`** | **`258.95` / `277.51`** | **`550.01` / `479.06`** | **`563.77` (+35.0%!)** | **`111.76 s` / `2.36 s`** | **`InklingScheduler` (`0.84 s` short TTFT / `133 tok/s` mixed `C=8`), cliff-free `C=9..16`, armed `TEMP_V16` + `min_p`, `57.1×` parser, `INCR_TOK`, true `7 GiB` UMA, `88 s` boot, auto GID, `:9108` + `spark-top`** |

---

## 3. Hard-Won Operational Rules (GB10 / SM121a)

1. **Always compile SM121a kernels with `-gencode arch=compute_121a,code=sm_121a --fmad=false`** (never `--use_fast_math` / `--ftz=true`) for strict 0-ULP `1dcf…f8c3` determinism.
2. **Keep `INKLING_C48_MOE_TILE8=0`, `INKLING_NVFP4/MX7_V16_LPT=0`, `INKLING_C32_NVFP4/MX7_T32_MIN_M=0`, `RDMA_AR_C32_MIN_NVEC=18432`, `INKLING_ROWAR_DESC_W_ORDER=0`, and `INKLING_SYNC_FREE_TEMP_V16_MIXED=0`** — all empirically verified on live 8-Spark silicon.
3. **Always launch via `tp8/ops/inkling-tp8-v17.sh`** so sysfs IPv4 RoCEv2 GID auto-discovery overrides any stale shell `NCCL_IB_GID_INDEX=3` (as seen on `spark-three`).
4. **Never let an OOM shatter UMA buddy orders (`orders 9–12`):** Monitor `python3 /home/jeweled/inkling-tp8/spark-top.py` and `http://spark-one:9108/metrics`.
5. **Never run `nvcc` or standalone GPU tests on any node while benchmarking or serving.**
6. **Preserve rollback files untouched:** Never overwrite `/home/jeweled/inkling-tp8/{best.env, live_champion_v12.env, live_champion_v16.env, patches-s6/, patches-s6-v12/, patches-s6-v16/}` or prior `.so` libraries.
