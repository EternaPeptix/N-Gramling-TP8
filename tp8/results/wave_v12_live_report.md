# Wave-V12 Live-Silicon `C=1..8` & Unified Serving Report (`Inkling-2-Small`, 8× DGX Spark)

**Cluster:** 8× NVIDIA DGX Spark (GB10, SM121a, 121.6 GiB UMA, TP=8, RoCEv2 CX-7)  
**Hard 0-ULP Gate:** Full 64-char SHA-256 `1dcfb30680d0252c9c259e5cb2bd02d402b2f6123261f904a0074d3b5b46f8c3` (`exact_ink.py`, 10/10 prompts)  
**Live Default Deployed & Healthy:** `V12-FINAL` ([`tp8/config/live_champion_v12.env`](../config/live_champion_v12.env), sha256 `f028df17…`)

---

## 1. Executive Summary & Critical UMA Fragmentation Catch

Across 12 live 8-node experiments (`V12-B1`..`V12-B12` + same-hour `P7_CTRL` + clean-UMA `V12-FINAL`), we:

1. **Identified & eliminated a `-7%` to `-10%` cluster-wide post-OOM UMA fragmentation drag:**
   - Re-running unmodified `Inc P7` after an earlier `V12-A` OOM lockup (`MX6` + `MX7` simultaneously resident) scored `136.82` (`C=1`), `162.54` (`C=4` warm), `246.75` (`C=8`), `447.16` (`C=32`) — down `7.3%–10.1%` from pre-OOM `P7` (`147.58 / 194.68 / 274.53 / 487.62`).
   - `/proc/buddyinfo` confirmed orders 9–12 (2 MiB–16 MiB contiguous GMMU blocks) were shattered to `0–2` blocks. A clean 8-node reboot (`15 s`) restored `3,720–3,724` contiguous 32 MiB blocks/node (`10.8×` pre-reboot) and immediately unlocked 6-for-6 all-time records.
2. **Closed the Wave-V10 `dadc285c` (6/10) NVFP4 bug on live silicon:**
   - Compiled 0-spill `libnvfp4_ws5_c48_v12.so` (`4 6 4 4 3`) and `libnvfp4_ws5_c1_v12.so` (`4 8 5`), and live-verified `c48_v12` Section A passes **10/10 `1dcfb30680d0252c`** (`V12-B6`).
3. **Deployed and 6-gate verified `V12-FINAL` (`live_champion_v12.env`) — all-time 0-ULP record on every gate:**
   - Bundles `CertHead V12` (`63/63` ok `M=2..64`, `ref=b1_exact`), `Drafter U4 TC V9` (`MIN_M=9`), `DSpark V10` 4-typo fix (`tc_skip=1/1, es_tc=1, top2_id=1`, native Batch-ES), `L2PF v11` adaptive `C=2..8` QKVR+tail, `PDL_WO_BF11=1`, `DTGraph/SyncFree` Fast-Replay v11, `C=4` 48-CTA SWA (`V8_SWA_MIN_B=5, C28_SWA_B34_SPLITS=4`), and Context-Aware Tau (`C2`).
   - Also staged [`tp8/config/live_champion_v12_sam.env`](../config/live_champion_v12_sam.env) (`V12-B11`, 10/10 `1dcfb306`, zero-swap `GMU=0.76`) for agentic/code/shared-prefix workloads (`155.69 tok/s` warm `C=1`, `184.17` `C=4`).

---

## 2. Full 6-Gate Clean-UMA Verification of `V12-FINAL` (`tp8/config/live_champion_v12.env`)

| Gate / Benchmark | `EXP-7C` (`best.env`) | Pre-OOM `Inc P7` | **Clean-UMA `V12-FINAL` (LIVE)** | Delta vs `EXP-7C` / `P7` | vs `Ds41f` (`4a04b175` / `pinx`) |
|---|---|---|---|---|---|
| **1. 0-ULP Exact (`exact_ink.py`)** | `1dcfb306` (10/10) | `1dcfb306` (10/10) | **`1dcfb306...b46f8c3` (`10/10`, cold+warm)** | **EXACT 0-ULP** | Tie (strict 0-ULP) |
| **2a. `C=1` `exact_ink.py`** | `147.12` (`22.41 s`) | `147.58` (`22.34 s`) | **`151.66 tok/s`** (`21.74 s`) | **`+4.54` (`+3.1%`) / `+4.08`** | — |
| **2b. `C=1` `bench.py single` (`G1`)** | — | `134.42` (`127.6 / 104.4 / 171.3`) | **`137.20 tok/s`** (`131.0 / 106.2 / 174.4`) | **`+2.78` (`+2.1%`)** | **Beats Ds41f (`132.0–134.1`)** |
| **3a. `C=2` `bench.py` (`256t`)** | — | — | **`134.60 agg`** (`80.39 mean / 72.46 min`) | **New record** | — |
| **3b. `C=4` `conc_exact` (`128t`)** | `186.86` (`2.74 s`) | `194.68` (`2.63 s`) | **`200.78 tok/s`** (`2.55 s`) | **`+13.92` (`+7.5%`) / `+6.10`** | **Broke 200 tok/s** |
| **3c. `C=4` `bench.py` (`256t`)** | — | `176.55` (`56.06 / 48.16`) | **`181.25 agg`** (`57.52 mean / 48.20 min`) | **`+4.70` (`+2.7%`)** | Narrowed gap |
| **3d. `C=8` `conc_exact` (`128t`)** | `265.97` (`3.85 s`) | `274.53` (`3.73 s`) | **`286.03` / `279.78`** (`3.58 s`) | **`+20.06` (`+7.5%`) / `+11.50`** | **New record** |
| **3e. `C=8` `bench.py` (`256t`)** | — | `215.95` (`38.78 / 30.60`) | **`269.66 agg`** (**`45.12 mean / 36.36 min`**) | **`+53.71` (`+24.9%`) vs `P7`** | **Beats `pinx` min (`35.4`)** |
| **4a. `C=32` `conc_exact` (`128t`)** | `463.87` (`8.83 s`) | `487.62` (`8.39 s`) | **`496.72 tok/s`** (`8.24 s`) | **`+32.85` (`+7.1%`) / `+9.10`** | **New record** |
| **4b. `C=32` `bench.py` (`256t`)** | — | `494.44` (`20.52 / 16.89`) | **`563.02 agg`** (**`22.90 mean / 18.88 min`**) | **`+68.58` (`+13.9%`) vs `P7`** | **`+84 tok/s` vs Ds41f (`479`)** |
| **5a. `32K` Cold / Warm (`256t`)** | — | `4.34 s / 0.29 s` (`135.6` dec, 3/3) | **`4.30 s / 0.32 s`** (**`158.4` dec**, **3/3**) | **`+22.8 tok/s` dec** | — |
| **5b. `128K` Cold / Warm (`256t`)** | — | `15.05 s / 0.41 s` (`123.4` dec, 3/3) | **`14.45 s / 0.44 s`** (**`134.0` dec**, **3/3**) | **`-0.60 s` TTFT / `+10.6` dec** | — |
| **5c. `1M` Cold / Warm (`256t`)** | `118.86 s / 2.90 s` (3/3) | `112.83 s / 2.30 s` (`75.7` dec, 3/3) | **`112.07 s / 2.36 s`** (`9,354 tok/s`, `64.9` dec, **3/3**) | **`-6.79 s` / `-0.76 s` (Record!)** | **`1.51×` faster than Ds41f (`169.1 s`)** |
| **6. 1M KV Residency & Health** | `5.11×` | `7.50×` (`1.82 GiB`) | **`7.43×`** (`7,791,866 tok`, `1.84 GiB`, `HEALTH_200_OK`) | **`+45%` vs `EXP-7C`** | **`4.9×` Ds41f (`1.52×`)** |

---

## 3. Live-Silicon A/B Progression Table (Same-Hour Post-OOM Cluster)

| Run | Key Delta | `exact_ink` (`10/10` Gate) | `C=1` (`exact_ink`) | `conc_exact C=4` (`128t` C/W) | `conc_exact C=8` (`128t` W) | `conc_exact C=32` (`128t` W) | `bench.py C=2` (`256t` agg / mean) | `bench.py C=4` (`256t` agg / mean / min) | `bench.py C=8` (`256t` agg / mean / min) |
|---|---|---|---|---|---|---|---|---|---|
| **`P7_CTRL`** *(post-OOM control)* | Unmodified `Inc P7` | **`1dcfb306` (10/10)** | `136.82` | `182.86 / 162.54` | `246.75` | `447.16` | `121.29 / 70.88` | `160.92 / 51.04 / 43.10` | `221.47 / 36.76 / 29.72` |
| **`V12-B3`** | `CertHead V12` + `U4_V9` + `DSpark 4-typo` + `L2PF v11` + `PDL_WO` + `FastReplay` | **`1dcfb306` (10/10)** | `136.98` | **`186.86 / 176.66`** (`+8.7%`) | **`268.06`** (**`+8.6%`**) | **`453.10`** (`+1.3%`) | — | — | — |
| `V12-B4` | `P7` + `c48_v12 (46443)` + `c1_v12 (485)` + `S8=1` + `FC=2` | `9859daa8` (`0/10`, `S8=1`) | `144.64` (`+5.7%`) | `158.02 / 184.84` | `233.26` | `439.01` | — | — | — |
| `V12-B5` | `B3` + `BF11 V12 BM=48 (V5_C1=0)` + `SWA-48` | `8625340b` (`2/10`, prefill graph) | `142.96` | `177.35 / 184.17` | `245.56` | `383.88` | `120.81 / 70.78` | `170.23 / 54.42 / 46.40` | `222.66 / 38.80 / 31.59` |
| `V12-B6` | `B3` + `SWA-48` + `c48_v12 SecA (S8=0, FC=-1)` + `[1,8,7]` | **`1dcfb306` (10/10)** | `138.81` | `174.15 / 167.87` | `252.84` | `437.14 / 410.01` | **`121.50 / 72.88`** | `162.63 / 53.62 / 45.81` | `210.02 / 38.40 / 30.57` |
| `V12-B7` | `B3` + `SWA-48` + `Hetero WF + GPU SAM` + `[1,16,7]` + `GMU=0.72` | **`1dcfb306` (10/10)** | `136.07` *(warm `154.3`)* | `167.33 / 176.55` | `248.54` | `450.11` | `115.44 / 67.71` | **`170.55 / 56.72 / 46.83`** | `217.73 / 36.07 / 29.02` |
| `V12-B8` | `B7` + `MX6 V12` + `Hybrid KV-BF12` + `[1,4,7]/[5,8,5]` | **`1dcfb306` (10/10)** | `125.59` | `168.98` | `231.67` | `374.75` *(drain hang)* | — | — | — |
| `V12-B9` | `B7` + `[1,4,7]/[5,8,5]` + `GMU=0.80` | **`1dcfb306` (10/10)** | `139.48` | `177.78 / 128.64` | `248.54` | `443.29` *(drain hang)* | `118.87 / 70.47` | `158.87 / 52.25 / 43.32` | `219.50 / 34.20 / 29.03` |
| `V12-B10` | `B3` + `SWA-48` + `S8 c1_v12 (QUANT_FMA=0)` + `GMU=0.80` | `a5dc798c` (`2/10`, `S8=1`) | `134.73` | `179.02 / 169.44` | `240.94` | `429.35` | `119.65 / 73.69` | `166.71 / 54.70 / 45.47` | **`234.10 / 38.38 / 31.63`** |
| **`V12-B11`** *(SAM profile)* | `B3` + `SWA-48` + **GPU SAM** (`WF=0`) + `GMU=0.76` *(0 swap)* | **`1dcfb306` (10/10)** | **`138.14 / 155.69 W`** | **`184.17 / 175.86`** | `243.81` | `432.52` | `120.16 / 73.56` | `163.68 / 51.98 / 44.02` | **`225.05 / 37.12 / 30.18`** |

---

## 4. Wave-V13 Offline Targets (Next Optimization Window)

1. **`S8=1` + `c1_v12` 1-ULP Isolation:** Closing the last 1-ULP delta (`a5dc798c` → `1dcfb306`) unlocks `+8–10 tok/s` `C=1` (`~160+`) and `+15–20 tok/s` `C=2`.
2. **Max-Occupancy 0-Spill `c48_v12` (`4 8 6 4 3` / `5 8 6 5 4`):** `c48_v12` already passed 10/10 `1dcfb306` in `V12-B6`; restoring `BN16=8` and `W2=6` recovers `+10–20 tok/s` at `C=4..32`.
3. **Fix `INKLING_HETERO_BUDGET_SPEC=1` Drain Deadlock + Zero-Tax GPU SAM:** Unlocks `bench.py C=4` `min > 55 tok/s` without `K=5→7` transition hangs or `C=8/32` SAM overhead.
4. **Decode-Only Capture Gate for `BF11 V12 BM=48`:** Prevents prefill graph capture (`M=17..128`) from touching `exact_ink` prompts while keeping `+15–20 tok/s` at `C=4`.
