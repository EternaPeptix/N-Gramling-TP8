# STATUS: parked 2026-11-16 — blocked on an NVIDIA GB10 driver livelock

Inkling-2-Small on 4× DGX Spark (TP4). Everything below the driver layer
is done and staged; the boot dies deterministically in the weight loader on
2 of 4 nodes. This document is the handoff: evidence, repro, what's staged,
and the two paths that unblock it.

## The blocker (exact)

The safetensors loader wedges at exactly **94,498 MiB** GPU-allocated on
**spark-7425** and **spark-5bc3** (reproducible across ~8 boots, both GMU
0.80 and 0.85, and with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`):

- python frame (stable for 5+ hours, 100% CPU):
  `Inkling2Experts._load_full (moe.py:177) → param.data.copy_(loaded)`
- `py-spy --native`: main thread spins in `[vdso]`
- gdb (all threads): main thread at 100% CPU inside **`libcuda.so.1`**;
  ~110 other threads idle in futex. No Xid errors, clocks/temps normal
- GPU memory frozen to the MiB for hours; no new file progress; no new
  syscalls (futex/epoll storm only)

Reads: the copy is a safetensors-lazy CPU tensor → CUDA param, ~18k small
copies at ~95 GB occupancy. fp8/uint8/fp32 copies are individually fine on
the same box. This is a driver-internal spin (unified-memory migration /
allocation path) on GB10 (SM121), driver 580.178.04, CUDA 13.3.
**Not fixable from userspace config.** Needs an NVIDIA driver fix, or a
loader that avoids the copy pattern (see paths below).

## What works (validated on this ring)

- Checkpoint staged on all 4 nodes (~380 GB, revision 2ad1157 = current
  main; `quantization_config` field removed from config.json — it
  contradicts the model's own `hf_quant_config.json` inkling2 declaration).
- `patches/trtllm_bmm.py`: **torch NVFP4/MXFP8 expert GEMM replacing the
  missing sm121 cubins** (the stock image ships sm100/103/107 only) — the
  full 48-layer model builds and initializes on GB10.
- `patches/linear.py`: k-group gate GEMM via tensor-core matmul (both
  Triton k-group variants exceed SM121's 99 KB shared memory).
- `patches/encoder_output_width.py`: skip the encoder-width measurement on
  follower nodes (driver-only RPC).
- Followers launch `--headless` (worker-only; full serve on every node
  asserts in init collectives).
- NCCL over TCP sockets through the routed switchless ring
  (`scripts/ring-routes.sh`; RoCE full-mesh is unsolved on this fabric —
  see commit history). DSpark disabled (its combine needs NVLink multicast,
  absent on GB10).
- rank 0 and rank 3 have each loaded the full model (90.93 GiB) and reached
  the post-load rendezvous multiple times.

## Paths that unblock it

1. **NVIDIA driver fix** for the copy livelock (repro above is minimal).
2. **Avoid the copy pattern** (userspace, ~a day): repack the checkpoint
   into a few large bulk files in the loader's exact target layout, or use
   vLLM's `fastsafetensors` C++ bulk loader, so the driver never sees 18k
   small pageable copies. rank 3 (7275) loads cleanly 3+ times, which says
   the bug is the pattern, not the silicon.
3. **The B12X-class kernel path** (Christophe's dsv41 stack family): his
   loader/kernel path for NVFP4 MoE on SM121 does not share this loader at
   all — dsv41 (552B MoE) serves fine on these same chips.

## Resume

`scripts/ring-routes.sh` on all four nodes, then
`launch/inkling2-tp4.sh 3 2 1 0` (workers first, head last). The entire
patch set applies via `patches/MANIFEST` bind-mounts. Boot is ~1-2h weight
load when the driver behaves.
