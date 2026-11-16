# Driver bug report: libcuda livelock during model load on GB10 (SM121), 4× DGX Spark

**Audience:** Thinking Machines (please route to NVIDIA driver team)
**Date:** 2026-11-16
**Reporter:** Matt Suiche

## Summary

Loading a 594B NVFP4 MoE checkpoint (Inkling-2-Small, 380 GB) via vLLM's
safetensors loader **deterministically livelocks** on 2 of 4 identical DGX
Spark (GB10) nodes. The loader's main thread spins at 100% CPU **inside
`libcuda.so`** with GPU memory frozen at exactly **94,498 MiB** for 5+
hours. The remaining 2 nodes load the same model to completion in 30-80
minutes on identical hardware, driver, VBIOS, kernel, image, and config.
A gdb all-threads backtrace shows the main thread spinning entirely within
libcuda; ~110 other threads idle in futex. No Xid, no OOM error, no
exception — the driver never returns from the copy/allocation call.

## Environment (identical on all 4 nodes)

| | |
|---|---|
| GPU | NVIDIA GB10 (SM121a), Blackwell, 128 GiB unified memory |
| Driver | 580.178.04 |
| VBIOS | 9A.0B.2D.00.00 |
| Kernel | 7.0.0-1019-nvidia-64k (aarch64) |
| CUDA | 13.0 (driver) / image ships 13.3 |
| torch | 2.14.0+cu130 |
| Runtime | vLLM nightly (`vllm/vllm-openai:inkling-2-small`) |
| Memory | all four nodes: 128 GB, dmidecode identical |

## The wedge (exact, reproducible ~8 boots)

- **Site:** `Inkling2Experts._load_full → param.data.copy_(loaded)`
  (per-tensor safetensors load, ~18k small CPU→GPU copies)
- **Frozen at:** 94,498 MiB GPU-allocated (rank-consistent), zero MiB
  movement for hours
- **CPU:** main thread 100-102% (busy-spin, not I/O wait)
- **python frame (py-spy, stable for hours):**
  `moe.py:177 param.data.copy_ ← moe.py:162 _load_expert_shard ← moe.py:165 _load_weight`
- **native frames (py-spy --native):** `[vdso]` only
- **gdb thread 1:** all frames inside `libcuda.so.1`
- **strace:** futex / epoll_pwait / ppoll storm, no read/write progress

## The node-specificity (the part we can't explain)

| node | role | result |
|---|---|---|
| spark-4687 | rank 0 | **loads fine** (50-81 min) |
| spark-7275 | rank 3 | **loads fine** (24-50 min) |
| spark-7425 | rank 2 | **livelocks at 94,498 MiB every time** |
| spark-5bc3 | rank 1 | **livelocks at 94,498 MiB every time** |

Same image, same checkpoint bytes (rsync-verified), same driver/VBIOS/
kernel/memory config, no residual GPU memory at idle, no thermals, no
throttling, healthy disks (no I/O errors). Two nodes always pass; the
other two always wedge. Whatever differentiates them is below our
visibility.

## Ruled out (each tested on the failing nodes)

- Simple standalone copies: 20k small pageable/mmap→GPU copies at
  ~90 GiB occupancy — complete in <1 s. (No minimal repro yet; the
  production-path repro below is 100% deterministic.)
- 9k small device allocations (expert-tensor sizes) — fine.
- One embedding-sized 1.85 GB single copy at ~47 GiB — fine.
- fp8_e4m3 / uint8 / fp32 copy dtypes individually — fine.
- GMU 0.80 vs 0.85 (model is 97.65 GB, fits both) — same wedge.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False` — same wedge.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False` + clean restart —
  same wedge.
- Checkpoint bytes (md5 of sampled shards across nodes) — identical.
- GPUs: clocks ~2093 MHz, 43-44 °C, no throttle reasons, no Xid in dmesg.

## Production repro (deterministic)

Repo `msuiche/dgx-inkling-small` (private; we can grant access):
`launch/inkling2-tp4.sh {0..3}` on the four nodes after
`scripts/ring-routes.sh`. The two failing nodes wedge in the loader at
94,498 MiB every boot within ~30-90 min.

## What works (so the model is not implicated)

- The model fully builds and initializes on GB10 with our torch
  NVFP4/MXFP8 expert GEMM substitute (`patches/trtllm_bmm.py`) — the
  stock image's TRT-LLM cubins simply don't exist for sm121
  (`sm100/sm103/sm107 only`), which is the separate known issue.
- The same hardware serves DeepSeek-V4.1-Flash (552B MoE, also NVFP4,
  TP4) at ~115 tok/s via a different (B12X) kernel/loader stack, so the
  pattern is specific to this loader path, not the silicon.

## Asks

1. Driver-level diagnosis of the libcuda spin: what in the copy/alloc
   path can loop forever at ~95 GB occupancy on unified memory without
   returning an error? (Happy to collect `CUDA_LOG`/driver debug logs or
   a core dump on request.)
2. Any guidance on a loader pattern on GB10/unified-memory that is
   guaranteed not to hit this (e.g., bulk cudaMemcpyAsync vs many small
   pageable copies).
3. Confirmation whether sm121 has known migration/allocation issues in
   580.178, and whether a newer/earlier driver is worth testing.

## Contact / artifacts

- `STATUS.md` — full handoff (all 14 fixes that got us here).
- `repro/gb10-livelock-repro.py` — the (non-triggering) standalone
  patterns tested so far.
- gdb / py-spy / strace captures embedded above; more available on
  request, including live access to the wedging nodes.
