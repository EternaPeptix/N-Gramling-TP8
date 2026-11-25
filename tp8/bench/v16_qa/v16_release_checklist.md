# Wave-V16 Crown-Jewel Permanent Production Release, 12-Gate Qualification & Rollback Checklist

**Target Cluster:** 8× NVIDIA DGX Spark (`spark-one`..`spark-eight`, GB10 SM121a, 121.63 GiB UMA, 4× 200GbE ConnectX-7 RoCEv2, TP=8, port `8010`, Prometheus Exporter `:9108`)  
**Model:** `thinkingmachines/Inkling-2-Small` (594B total / 17B active MoE, 40 NVFP4 SWA-1024 + 8 MXFP8 Global 1M + 5-layer DSpark)  
**Mandatory 0-ULP Lock (`temp=0`):** `1dcfb30680d0252c9c259e5cb2bd02d402b2f6123261f904a0074d3b5b46f8c3` (`exact_ink.py`, `10/10` cold, warm, post-chaos, post-harvester, and post-recovery)

---

## 0. Rollback Preservation Guarantee (Verify Before Touching Anything!)

Before syncing ANY Wave-V16 artifact, confirm all three fallback baselines remain **100% untouched** on all 8 nodes:

- [ ] **Tier-1 Rollback (`V15-FINAL` permanent baseline):** `/home/jeweled/inkling-tp8/live_champion_v15_default.env` + `patches-s6-v15/`
- [ ] **Tier-2 Rollback (`V12-FINAL` live champion):** `/home/jeweled/inkling-tp8/live_champion_v12.env` + `patches-s6-v12/`
- [ ] **Tier-3 Nuclear Rollback (`EXP-6B`):** `/home/jeweled/inkling-tp8/best.env` + `patches-s6/`
- [ ] **Never overwrite** `best.env`, `live_champion_v12.env`, `live_champion_v15_default.env`, `patches-s6/`, `patches-s6-v12/`, or `patches-s6-v15/`. Wave-V16 deploys into dedicated `patches-s6-v16/` and `live_champion_v16_*.env` paths.

---

## 1. Pre-Deployment Graceful Drain, Clean UMA Guard & Static Env Audit

- [ ] **Run static Wave-V16 Cross-Wave Env, Patch & `.so` Manifest Audit locally BEFORE touching cluster:**
  ```bash
  python3 v16_env_conflict_auditor.py \
    --env /path/to/live_champion_v16_default.env \
    --patches /path/to/patches-s6-v16 \
    --kernels /path/to/kernels_v16 \
    --out-json /tmp/audit_v16_default.json
  ```
  - Pass criteria: `0` duplicate keys, `0` CR bytes, `bash -n` CLEAN, `100%` launcher/ops forwarding, `0` dead/typo keys, `100%` `.so` manifest coverage (0 stale aliases), ALL V15+V16 safety & 0-ULP locks PASS (`S8_W2_QUANT_FMA=0`, `BF11_FREE_RAW=0`, `TILE8=1`, `DYN_K<=15`, `SAM_ALLOW_TEMP=0`, `REG_BF12 P2>=196608`, `L2PF pace<=185 GB/s`, `PREFILL_SP_RS_HALF2_MODE=nccl`).
- [ ] **Gracefully drain active traffic (e.g., running `rollout_harvester`) with ZERO dropped completions:**
  ```bash
  ./v16_drain_and_deploy.sh drain --drain-timeout 120
  ```
- [ ] **Run 8-Node Pre-Boot UMA Buddy & Swap Guard (`0 MiB` swap, `order-13 >= 2500`, `4/4` CX-7 Active, `0` Xids):**
  ```bash
  ./v15_preboot_uma_guard.sh --json
  ```
  - If any node shows post-OOM buddy fragmentation (`order-13 < 2500`), run `./v15_preboot_uma_guard.sh --auto-reboot-defrag`.

---

## 2. Build & Atomic 8-Node Deployment of Wave-V16 Artifacts

- [ ] **Build all SM121a `.so` libraries on ONE quiet GB10 node** (NEVER compile while benchmarking!):
  - Mandatory compiler flags on every 0-ULP decode/sampler kernel:
    `-gencode arch=compute_121a,code=sm_121a --fmad=false --ptxas-options=-v`
    (**0 register spills / 0 stack frame** enforced!).
  - Export `TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas` BEFORE `import torch` for any standalone Triton check.
- [ ] **Verify `/rdma/SHA256SUMS` and deploy atomically (`.tmp` + `mv -f`) to `/home/jeweled/inkling-tp8/rdma/` on 8/8 nodes** (`spark-five -> 10.77.0.51`, `spark-six -> 10.77.0.61` via `-J spark-one`).
- [ ] **Sync unified `patches-s6-v16/`, `live_champion_v16_{default,max_kv_speed}.env`, Track 7 ops/exporter, and Track 8 QA suite (`v16_qualify_all.py`, `v16_harvester_stress_test.py`, `v16_env_conflict_auditor.py`) to 8/8 nodes.**
- [ ] **Start `inkling-tp8-v16-exporter.service` on head node (`:9108`) and verify:**
  ```bash
  curl -sf http://spark-one:9108/healthz
  curl -sf http://spark-one:9108/metrics | grep -c '^inkling_v16_'
  ```
- [ ] **Launch 8/8 ranks (`7..0`) atomically via `v16_cluster_supervisor.sh` (includes automatic 0-ULP boot cert & auto-rollback on boot failure):**
  ```bash
  ./v16_cluster_supervisor.sh start /home/jeweled/inkling-tp8/live_champion_v16_default.env
  ```

---

## 3. Automated 12-Gate Wave-V16 Permanent Release Qualification

### Step 3A — Fast Smoke Pre-Check (`--mode quick`, ~3 min)
```bash
python3 /home/jeweled/dsv41-bench/v16_qualify_all.py \
  --base-url http://spark-one:8010 \
  --exporter-url http://spark-one:9108 \
  --mode quick \
  --env-file /home/jeweled/inkling-tp8/live_champion_v16_default.env \
  --out-json /tmp/results_v16_quick.json
```

### Step 3B — Dedicated Production `rollout_harvester` 16×4 Stress & Diversity Benchmark
```bash
python3 /home/jeweled/dsv41-bench/v16_harvester_stress_test.py \
  --base-url http://spark-one:8010 \
  --workers 16 --n 4 --think-budget 2048 --max-tokens 2048 \
  --temperature 0.6 --top-p 0.95 \
  --out-json /tmp/results_v16_harvester_16x4.json
```

### Step 3C — Full 12-Gate Permanent Release Certification (`--mode full`)
```bash
python3 /home/jeweled/dsv41-bench/v16_qualify_all.py \
  --base-url http://spark-one:8010 \
  --exporter-url http://spark-one:9108 \
  --mode full \
  --profile-name live_champion_v16_default.env \
  --env-file /home/jeweled/inkling-tp8/live_champion_v16_default.env \
  --test-crash-recovery \
  --enforce-perf-floors \
  --out-json /tmp/results_v16_permanent.json
```

### Mandatory 12-Gate Wave-V16 Acceptance Matrix

| Gate | Criterion | `V12-FINAL` Live | `V15-Def` Proj | **Wave-V16 Target & Hard Gate** |
| :--- | :--- | :---: | :---: | :--- |
| **G1. 0-ULP Bit-Exact** | `exact_ink.py` Cold + Warm + Post-Chaos | `1dcfb306` `10/10` | `10/10` | **MANDATORY `10/10` (`1dcfb306...b46f8c3`) all 3 runs** |
| **G2. `C=1` Decode** | `exact_ink` + `bench single` G1 GM | `151.7 / 137.2` | `181.9 / 171.0` | **G1 GM `>= 134.1`** (proj **`206.5` cold / `240.2` warm**, `90.4%` BW) |
| **G3. `C=2,4,8` Mid-Zone** | `conc_exact` + `bench` agg/min vs Ds41f | `C4m 48.2 / C8m 36.4` | `C4m 58.0 / C8m 42.4` | **`C4 agg >= 175`, `C8 min >= 35.4`** (proj **`C4m 63.5 / C8m 45.4`**) |
| **G4. `C=32` Throughput** | `conc_exact` (`128t`) + `bench` (`512t`) | `496.7 / 563.0` | `599.4` | **`agg >= 479.0`** (proj **`670.6`**, `90.4%` BW) |
| **G5. `32K/128K/1M` Context** | Cold & Warm TTFT + Decode + Needles | `112.1s / 64.9` | `75.8s / 101.9` | **`3/3` needles all tiers**, proj **`63.0s` T1 / `57.7s` T2**, **`116.8 / 131.9 tok/s`** |
| **G6. Multi-1M Residency** | `8x` warm + `9th/10th` overflow + S0 Turn-3 | `7.43×` KV | `7.78×` KV | **`8/8` warm**, `9th/10th` clean w/ **0 stale leaks** (`7.77×` / `9.09×` Max) |
| **G7. Mixed `1M + C=4/C=8`** | Cold 1M + continuous interactive decode | — | PASS | Cold 1M `3/3`, true overlap, `0` errors/deadlocks, ITL p99 `<= 4000ms` |
| **G8. 500-Req Chaos Storm** | 500+ mixed + 20% aborts + post-chaos 0-ULP | — | PASS | `0` crashes/5xx, `0` slot/KV leaks, `<1.5x` drift, **post-chaos `10/10 1dcfb306`** |
| **G9. 8/8 Cluster & UMA** | `meminfo`, `buddyinfo`, `ibstat`, `dmesg` | `3.28 GiB` swap | `0 MiB` swap | **8/8 `HEALTH_200_OK`**, **`0 MiB` swap** (`<256`), `4/4` RoCEv2, `0` Xids, healthy buddy |
| **G10. Crash Recovery** | `--test-crash-recovery` (kill `r3`) | — | `<180s` | Detect, capture bundle, clean 8/8, restart + **`10/10 1dcfb306` `< 180s`** |
| **G11. `rollout_harvester` `16×4`** | `temp=0.6`, `top_p=0.95`, 64 conc + canaries | `~523` proj | `~553` proj | **`0` dropped**, **`0` collapsed `n=4` groups**, unique `>= 0.75`, Jaccard `>= 0.10`, concurrent `temp=0` canaries exact, **post-harvest `10/10 1dcfb306`** (proj **`679.8 tok/s`, `+23.0%` vs V15**) |
| **G12. V16 Features & Telemetry** | Dynamic-K/Tree + Multimodal + `:9108` + Drain | — | — | High-conf code/math `temp=0` exact + speedup, multimodal PNG zero `<|image_pad|>` pollution & SAM isolated, **`10/10` Prometheus `:9108` metrics present & `0` Xids**, zero-drop graceful drain to `0` running/waiting! |

- [ ] **Confirm `results_v16_permanent.json` reports `"overall_status": "PASS"` and `"gates_passed": 12, "gates_failed": 0`!**

---

## 4. Rollback Procedures (If Any Gate Fails!)

### Immediate Rollback -> `V15-FINAL` (or `V12-FINAL`, ~70s)
```bash
./v16_drain_and_deploy.sh deploy /home/jeweled/inkling-tp8/live_champion_v15_default.env
# Or direct V12-FINAL nuclear fallback:
./v16_cluster_supervisor.sh stop
bash v8_cluster.sh launch /home/jeweled/inkling-tp8/live_champion_v12.env 0
curl -sf http://spark-one:8010/health
ssh spark-one 'python3 /home/jeweled/inkling-tp8/exact_ink.py /tmp/rollback_exact.json rollback'
```

---

## 5. Final Sign-Off & Git Tagging (Post Live 12/12 PASS Only)

- [ ] Archive signed `results_v16_permanent.json`, `results_v16_harvester_16x4.json`, `audit_v16_default.json`, and `/rdma/SHA256SUMS`.
- [ ] Verify offline self-test passes cleanly (`python3 test_v16_qa_offline.py` -> `EXIT 0`).
- [ ] Verify `systemctl is-active inkling-tp8-v15-supervisor.service inkling-tp8-v16-exporter.service` are both `active`.
- [ ] Import `v16_grafana_dashboard.json` and load `v16_prometheus_alerts.yml` into Prometheus/Grafana.
- [ ] Commit & tag `v16-crown-jewel-permanent` with ZERO hardcoded secrets and strict Unix LF line endings.
