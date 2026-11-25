# Inkling-2-Small on 8× NVIDIA DGX Spark (TP8, SM121a)

Production-grade, **100% 0-ULP bit-exact** inference stack for
[`thinkingmachines/Inkling-2-Small`](https://huggingface.co/thinkingmachines/Inkling-2-Small)
(594B total / 17B active MoE, 384 routed experts top-8, 40 NVFP4 local SWA-1024 layers + 8 MXFP8 global layers, 1M context, 5-layer DSpark speculator)
on **8× NVIDIA DGX Spark (GB10, SM121a)** over **4× 200GbE ConnectX-7 RoCEv2**.

> **Status (2026-11-25): PERMANENT LIVE PRODUCTION CHECKPOINT on 8/8 nodes (`0.0.0.0:8010`, `tp8/config/live_champion_v17.env`, `V17-FINAL`).**
> Empirically qualified and pruned on live 8-node GB10 silicon with **zero CUDA recompiles**, `V17-FINAL` delivers **10/10 0-ULP deterministic decode (`1dcf…f8c3`)**, **88 s cold boot** (down from `115 s`), **`+35.0%` faster `16×4` `temp=0.6` RLVR harvesting (`563.77` vs `417.78 tok/s`, peak `650.6 tok/s`, `1.00` unique, `12/12` math quality PASS, `min_p=0.05` unblocked)**, **`22.7×` faster urgent short-chat TTFT during active `128K` prefills (`0.837 s` via `InklingScheduler` short-skip + pure-prefill/pure-decode interleaving holding `C=8` at `133.4 tok/s`)**, **new all-time `C=4` (`188.66` / `199.22 conc_exact`) and `C=8` (`277.51 conc_exact`) records** with the `B=8→9..16` batching cliff eliminated (`C=9..16`: `266.3 → 380.2 tok/s`), **`57.1×` faster $O(1)$ streaming JSON tool-call parsing**, **`111.76 s` cold / `2.36 s` warm `1M` context (`3/3` needles)**, **`6.86×` simultaneous 1M KV residency** with true `7.0 GiB` OS UMA reserve, **`0` KV leaks**, **`0 MiB` new swap**, and live `:9108` Prometheus + `spark-top.py` cluster HUD.

---

## Qualified Permanent Checkpoint Results (`V17-FINAL` → `tp8/config/live_champion_v17.env`)

All `V17-FINAL` numbers were measured on **one unified live 8-Spark server** (`GMU=0.85`, `MAX_BATCHED=24576`, port `8010`, stock 2000 MHz GB10 clocks, `live_champion_v17.env` sha256 `d49f98882104…`).

| Metric / Gate | `V12-FINAL` | `V16-FINAL` | **Current (`V17-FINAL`)** | Delta vs `V16-FINAL` | How Measured |
|---|---:|---:|---:|---:|---|
| **1. Bit-Exact Decode (0-ULP)** | `1dcf…f8c3` (10/10) | `1dcf…f8c3` (10/10) | **`1dcf…f8c3` (10/10)** | **0-ULP LOCK** | `exact_ink.py` cold, warm, post-stress, post-chaos & live `0.0.0.0:8010` |
| **2. Cold Cluster Boot Time** | `~120 s` | `115 s` | **`88 s`** | **`-27 s` (`-23.5%`)** | `LOAD_O_DIRECT=1` + persistent `~/.nv` 4 GiB PTX cache + 17-kernel prewarm |
| **3. `C=1` `bench.py single` (`1024t`, `t=0`)** | `137.20` GM | `139.07` GM | **`138.24` GM (`143.94` mean)** | `parity` | `code 129.7 / prose 101.7 / math 200.7` |
| **3b. `C=1` `temp > 0` (`V17.1`, `512t`)** | — | — | **`code 126–129` · `prose 90–92` · `math 170–181`** | **`+25%` to `+66%` vs V17.0!** | `INKLING_SYNC_FREE_TEMP_V16_MIN_B=64` + `INKLING_VERIFY_W_TEMP=1` (`-9..-15 ms/step`, 100% unbiased) |
| **4a. `C=4` `bench` (`256t`) / `conc_exact` (`128t`)** | `181.25` / — | `187.64` / `191.76` | **`188.66` / `199.22`** | **`+1.02` / `+7.46` (RECORD!)** | `bench.py conc --conc 4` (`59.9 mean / 51.6 min`) & `conc_exact` |
| **4b. `C=8` `bench` (`256t`) / `conc_exact` (`128t`)** | `269.66` / `286.03` | `259.49` / `276.01` | **`258.95` / `277.51`** | **`+1.50` `conc_exact` (RECORD!)** | `bench.py conc --conc 8` (`43.2 mean / 35.2 min`) & `conc_exact` |
| **4c. `C=9 / 12 / 14 / 16` (`256t`, Cliff Fix)** | cliff @ `B=9` | cliff @ `B=9` | **`266.3 / 311.5 / 320.6 / 380.2`** | **Cliff Eliminated** | Exact graph buckets `9,10,13,14,15,18` + `K=4@B9..12` + monotonic `DINK_TAU` |
| **4d. `C=32` `bench` (`256t`) / `conc_exact` (`128t`)** | `563.02` / — | `567.68` / `479.63` | **`550.01` / `479.06`** | `conc_exact` parity | `bench.py conc --conc 32` (`+71` vs Ds41f) & `conc_exact` (`8.55 s`) |
| **5. Mixed `C=8` Decode + Cold `128K` + Short Chat** | — | ~14–113 s short TTFT | **`0.837 s` short TTFT · `133.4 tok/s` `C=8`** | **`~17–135×` faster TTFT!** | `bench_v17_mixed_sched.py` (`InklingScheduler`: `pure_dec=103`, `pure_pf=25`, `short_skip=1`) |
| **6. RLVR Harvester (`16×4`, `temp=0.6`, `512t`)** | Synced PyTorch | `417.78 tok/s` *(disarmed)* | **`563.77 tok/s`** *(peak `650.6`)* | **`+146.0 tok/s` (`+35.0%`!)** | `SYNC_FREE_TEMP_V16` fixed (`may_mask_logits`) + `min_p=0.05` unblocked (`12/12` math PASS) |
| **7. `32K` / `128K` / `1M` Long-Ctx (`3/3` needles)** | `4.30 / 14.45 / 112.07 s` | `4.30 / 14.49 / 113.02 s` | **`4.42 / 14.70 / 111.76 s`** (`0.29 / 0.43 / 2.36 s` warm) | **`-1.26 s` `1M` cold!** | `bench_1m_e2e.py --max-tokens 384 --warm` (`47.4×` 1M warm speedup) |
| **8. Streaming Tool-Call Parser (`6 KB` arg)** | `1551.1 ms` | `1551.1 ms` | **`27.15 ms`** | **`57.1×` faster ($O(1)$)** | `_inkling_arg_value_extender` (`4/4` 100% bit-identical deltas) |
| **9. Chaos Soak (`120` req, `C=16`, `20%` abort)** | — | `96/96` OK, `0` leaks | **`96/96` OK, `24/24` aborts, `0` leaks** | **`7.2 req/s` (`+22%`)** | `v15_chaos_soak.py` (`drift=0.543×`, `0` KV anomalies, `0 MiB` new swap) |
| **10. 1M KV Residency, Idle CPU & 24/7 Ops** | `7.43×` | `7.11×` (`2.79 GiB` OS floor) | **`6.86×` (`7.20M` tok, `7.0 GiB` OS floor, `0.33%` idle RT CPU)** | **`130×` lower idle CPU + HUD** | `rdma_ar_ext_c32_v17.so` (`43.2% → 0.33%` idle, `99.4%` sys idle), `FAST_KV_FREE=1`, `spark-top.py` + `:9108` live |

- Full machine-readable archives: [`tp8/results/checkpoint_2026-11-25_v17_final.json`](tp8/results/checkpoint_2026-11-25_v17_final.json) · [`tp8/results/checkpoint_2026-11-25_v16_final.json`](tp8/results/checkpoint_2026-11-25_v16_final.json) · [`tp8/results/checkpoint_2026-11-24_v12_final.json`](tp8/results/checkpoint_2026-11-24_v12_final.json).

---

## What Shipped in `V17-FINAL` (All Zero-CUDA-Recompile, Live-Verified)

1. **Standalone `InklingScheduler` (`tp8/patches/inkling_sched.py`, `--scheduler-cls inkling_sched.InklingScheduler`) + `--scheduling-policy priority`:**
   - Ports `ds41f`'s prefix-cache-probed `short-skip` + `hold=1` async lookahead gate (checked *first* so urgent chats/warm multi-turn requests preempt running `128K–1M` prefills in **`0.837 s`** instead of `19–113 s`).
   - Adds **Pure-Prefill (`4096t`) / Pure-Decode (`4`-step burst) step separation** whenever active decoders coexist with a long prefill (`>=2048` fresh tokens), keeping decode CUDA graphs + DTGraph + SyncFree ON during decode bursts and SP + DBO + Sage2 + OTF-Rel + cm=2 ON during prefill chunks (`C=8` holds **`133.4 tok/s`** through a cold `128K` prefill with `0` fallbacks), while leaving solo `1M` prefills at full `24576`-token chunks (`111.76 s`).
2. **Batching Cliff Elimination (`B=8→9..16`):** Added exact decode graph buckets `9,10,13,14,15,18`, smooth `K=4` bridge at `B=9..12`, and monotonic `INKLING_DINK_BATCH_TAU` + `INTERP=1` → new records at `C=4` (`188.66 / 199.22`) and `C=8` (`277.51`) and smooth ramp `C=9..16` (`266.3 → 380.2 tok/s`).
3. **`SYNC_FREE_TEMP_V16` Fixed & Unblocked (`sync_free.py` + `sampling_params.py`):** Fixed the `needs_logits_processing` check that had disarmed `TEMP_V16` on 99% of sampled traffic (now checks `may_mask_logits` for `temp>0`), isolated mixed `temp=0 + temp>0` batches (`TEMP_V16_MIXED=0`) for strict 0-ULP determinism, and relaxed `_validate_spec_decode()` to permit `min_p` when `TEMP_V16=1` → **`+35.0%` Harvester (`563.77 tok/s`)** and `12/12` `temp=0.6 + min_p=0.05` math quality PASS.
4. **APIServer CPU & Parser Upgrades (`inkling_parser.py` + `inkling_renderer.py`):** Added $O(1)$ open-JSON-string `_inkling_arg_value_extender` (**`57.1×` faster** on 6 KB tool args, 100% streaming parity) and thread-safe 128 MiB bit-exact LRU memo on `_HfBackedTmlTokenizer.encode_text()` (`INKLING_INCR_TOK=1`).
5. **UMA, KV Telemetry, CPU Pinning & Warmup Hardening:**
   - Fixed `extensible_kv_cache.py` UMA OS floor math (`max(KV_CACHE_NON_TORCH_RESERVE_BYTES, UMA_MIN_FREE_BYTES)` → true `7.0 GiB` OS reserve).
   - Enabled `INKLING_FAST_KV_FREE=1` + non-null fallback in `kv_cache_manager.py` (eliminates `sconv=1.0` phantom telemetry).
   - Fixed `host_pin.py`: `--node-rank` parsing for ranks `2–7`, post-capture re-sweep moving late worker threads off CPU `5` onto `[6, 7, 8]`, `EngineCore` sibling threads moved off CPU `15` onto `[16, 17, 18]`, eliminated duplicate `rank0_metrics.json` writer, and added 180 s `faulthandler` capture watchdog.
   - Extended `prewarm_prefill_c32()` in `dtgraph.py` (`17` kernels pre-JIT'd before serving) and split `WorkQueue` pointers (`+16B`) in `moe_c32_runtime.py`.
6. **Production Launcher & 24/7 Observability (`tp8/ops/inkling-tp8-v17.sh`, `tp8/ops/spark-top.py`, `v16_metrics_exporter.py`):**
   - Auto-discovers each node's IPv4-mapped RoCEv2 GID index from sysfs (immune to stale `~/.bashrc` `NCCL_IB_GID_INDEX=3`), verifies 4/4 active CX-7 HCAs + 92 `.so` `SHA256SUMS`, serves dual model aliases (`inkling-2-small` + `Inkling-2-Small`), mounts persistent `~/.nv` with `CUDA_CACHE_MAXSIZE=4GiB` + `LOAD_O_DIRECT=1` (**88 s boot**), and enforces `--oom-score-adj 500`, `--ulimit nofile=524288:524288`, `200m × 5` logs, and `30s` SIGTERM drain.
   - Live 1-command 8-node cluster HUD (`python3 /home/jeweled/inkling-tp8/spark-top.py`) + live Prometheus exporter on `:9108/metrics`.

---

## Empirically Pruned Candidates (V16 & V17)

| Candidate Tested Live | Measured Outcome | Why Pruned |
|---|---|---|
| **`C48_MOE_TILE8=1`** *(V17)* | `C=4` `-7.6%`, `C=8` `-5.9%` | V16 `BN16` beats `TILE8` at `B=2..8`; pruned to `0` |
| **`NVFP4/MX7_V16_LPT=2`** *(V17)* | `10/10 1dcfb306` with `+16B` split `WorkQueue`, slightly slower | Atomic queue overhead vs `LPT=0`; pruned to `0` |
| **`C32_NVFP4/MX7_T32_MIN_M=96`** *(V17)* | `C=32` `-3.2%` | `TILE=16` faster on GB10; pruned to `0` |
| **`RDMA_AR_C32_MIN_NVEC=10368`** *(V17)* | Displaced `ROW_AR_C28` at `C=4` | Restored `18432` |
| **`ROWAR_DESC_W_ORDER=1`** *(V17)* | Disables `DTGraph FAST_REPLAY_V14/V16` | Kept `0` |
| **`TREE_SPEC` / `MOE_S8` / `BF11_V13` / `ROUTER_V14` / `L2PF_V16` / `Prefill_V16`** *(V16)* | Broke `1dcfb306` or regressed throughput | Kept pruned (see `STATUS.md`) |

---

## Repository Layout & Operations

```text
tp8/
  config/
    live_champion_v17.env                  # Promoted V17-FINAL permanent production default (GMU=0.85)
    live_champion_v16.env                  # Preserved V16-FINAL rollback baseline (GMU=0.85)
    live_champion_v12.env                  # Preserved V12-FINAL rollback baseline (GMU=0.85)
    live_champion_v12_sam.env              # Optional GPU-SAM agentic variant (GMU=0.76)
    best.env                               # Preserved EXP-6B rollback baseline (GMU=0.72)
  ops/
    inkling-tp8-v17.sh                     # V17 production launcher (auto GID, 4/4 HCA, SHA256SUMS, dual alias)
    spark-top.py                           # 1-command 8-node live cluster HUD
    v16_metrics_exporter.py                # Live Prometheus exporter (:9108)
    v15_health_watchdog.py                 # Health watchdog + crash bundler + fallback
    inkling-tp8-v15.sh, v16_drain_and_deploy.sh, v16_grafana_dashboard.json, v15_systemd/
  patches/                                 # V17-FINAL vLLM Inkling-2 overlays (58 files, py_compile green)
  kernels/                                 # Preserved V12 & V16 SM121a CUDA sources (0 spills)
  bench/v16_qa/                            # Harvester 16x4, mixed-sched V17 suite, multimodal API, chaos soak
  results/
    checkpoint_2026-11-25_v17_final.json   # Live-verified V17-FINAL permanent checkpoint archive
    checkpoint_2026-11-25_v16_final.json   # Prior V16-FINAL archive
    checkpoint_2026-11-24_v12_final.json   # Prior V12-FINAL archive
```

- **Live HUD:** `python3 /home/jeweled/inkling-tp8/spark-top.py` (or `--once`)
- **Live Health & Metrics:** `curl -sf http://spark-one:8010/health` · `curl -sf http://spark-one:9108/metrics`
- **0-ULP Certification:** `python3 /home/jeweled/inkling-tp8/exact_ink.py` (`10/10` SHA-256 `1dcfb30680d0252c9c259e5cb2bd02d402b2f6123261f904a0074d3b5b46f8c3`)
- **Instant Rollback to `V16-FINAL` or `V12-FINAL`:** Source `/home/jeweled/inkling-tp8/live_champion_v16.env` (or `live_champion_v12.env`) across ranks `0..7` and relaunch `inkling-tp8-s6.sh`.

See [`STATUS.md`](STATUS.md) for the complete wave-by-wave engineering log and GB10 operational rules.
