#!/usr/bin/env python3
"""Minimal repro: GB10 (SM121) unified-memory driver livelock.

Mimics the vLLM safetensors loader's pattern on a 594B MoE checkpoint:
allocate ~90 GiB on the GPU in many small chunks (like ~18k expert params),
then run many small CPU(pageable)->GPU copies. On affected GB10 + driver
580.178 the copy phase spins forever inside libcuda at 100% CPU with GPU
memory frozen; on healthy setups it completes in seconds.

Usage:
  python3 gb10-livelock-repro.py [occupy_gib=90] [copies=20000]
Watch:
  - healthy: prints "copy N" progress and exits 0
  - livelock: stops at some N, process pegs one core, GPU memory frozen,
    main thread inside libcuda.so (gdb: bt shows only libcuda frames)
"""
import os
import sys
import time

import torch

occupy_gib = int(sys.argv[1]) if len(sys.argv) > 1 else 90
copies = int(sys.argv[2]) if len(sys.argv) > 2 else 20000


def occupy(gib: int) -> list[torch.Tensor]:
    """Allocate gib on the GPU in loader-like small chunks."""
    keep = []
    chunk = 32 * 1024 * 1024  # 32 MiB per allocation, like expert tensors
    per_chunk = chunk // 2  # bf16
    n = (gib * 1024**3) // chunk
    for i in range(n):
        keep.append(torch.empty(per_chunk, dtype=torch.bfloat16, device="cuda"))
        if i % 50 == 0:
            torch.cuda.synchronize()
    return keep


def main() -> None:
    print(f"torch {torch.__version__}, cuda {torch.version.cuda}, "
          f"device {torch.cuda.get_device_name(0)}")
    print(f"occupying ~{occupy_gib} GiB in small chunks ...", flush=True)
    keep = occupy(occupy_gib)
    torch.cuda.synchronize()
    print(f"allocated {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)

    # loader's access pattern: many small pageable CPU -> GPU copies
    src = torch.empty(288 * 16, dtype=torch.uint8)  # ~4.6 KB, pageable
    t0 = time.time()
    for i in range(copies):
        dst = src.cuda(non_blocking=False)
        del dst
        if i % 1000 == 0:
            print(f"copy {i}/{copies}  ({time.time() - t0:.1f}s)", flush=True)
    torch.cuda.synchronize()
    print(f"OK: {copies} copies in {time.time() - t0:.1f}s", flush=True)
    del keep


if __name__ == "__main__":
    main()
