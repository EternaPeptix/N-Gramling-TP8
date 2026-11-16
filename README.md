# dgx-inkling-small

Serving [thinkingmachines/Inkling-2-Small](https://huggingface.co/thinkingmachines/Inkling-2-Small)
on **4× NVIDIA DGX Spark (GB10)**, TP=4 across the switchless RoCE ring.

> **STATUS 2026-11-16: parked — blocked on an NVIDIA GB10 driver livelock
> in the weight loader.** See [STATUS.md](STATUS.md) for the full handoff
> (evidence, repro, staged state, and the three paths that unblock it).
> Everything below the driver layer works: the torch NVFP4 expert GEMM
> substitute is in `patches/`, all four nodes have the checkpoint, and the
> model builds. The boot currently wedges at 94498 MiB in the loader on
> 2 of 4 nodes (driver spin inside `libcuda.so` — not userspace-fixable).

594B total / 17B active MoE (384 routed experts, top-8), multimodal (23-layer
ViT), bundled DSpark speculator (7 tokens/step), relative-bias hybrid
attention: 40 sliding-window layers (1024) + 8 full-attention layers.

## What fits, honestly

The vendor card says the NVFP4 checkpoint needs "at least 500 GB aggregated
VRAM". Four Sparks have **484 GiB of unified memory in total** — under the
vendor number, and the unified memory is shared with the OS. The checkpoint is
**mixed precision**: NVFP4 experts on layers 0–39, **MXFP8 experts on layers
40–47**, so the weight footprint lands ~370–390 GB, not the ~300 GB an
all-NVFP4 read suggests.

| item | size |
|---|---|
| weights (mixed NVFP4/MXFP8 + ViT + embeddings) | ~370–390 GB |
| KV for one 1M-token sequence (vLLM's measured figure) | 64.47 GiB |
| KV for one 262144-token sequence | ~16 GiB |
| ring budget at `--gpu-memory-utilization 0.85` | ~411 GiB |

Consequence: the **full 1M context does not fit** on 4 Sparks. The working
envelope is:

| `--max-model-len` | KV/seq | concurrent seqs | notes |
|---|---|---|---|
| 131072 | ~8 GiB | 4–6 | roomy, default dev profile |
| **262144** (recipe default) | ~16 GiB | 2–3 | **recommended default** |
| 524288 | ~32 GiB | 1 | needs GMU 0.90, single user |
| 1048576 | 64.47 GiB | — | does not fit; needs TP8 (2 trays) |

The local layers are why even 256K is cheap: 40 of 48 layers pay a flat
~160 MiB per sequence regardless of length; only the 8 global layers scale
(32 KiB/token bf16). FP8 KV (`KV_DTYPE=fp8`) halves the scaling term.

## Layout

```
config/cluster.env.example  all knobs, documented — copy to .env and fill in
launch/inkling2-tp4.sh      per-rank launcher (run on each node: rank 3,2,1 then 0)
scripts/fetch-weights.sh    stage the ~380 GB checkpoint to all 4 nodes
bench/bench.py              C1 decode + TTFT smoke benchmark
```

## Quickstart

```bash
# every node
cp config/cluster.env.example .env && $EDITOR .env
scripts/fetch-weights.sh            # ~380 GB per node, or once + rsync

# rank 3, 2, 1 (workers), then rank 0 (head) last
launch/inkling2-tp4.sh 3            # on node 4
launch/inkling2-tp4.sh 2            # on node 3
launch/inkling2-tp4.sh 1            # on node 2
launch/inkling2-tp4.sh 0            # on the head — waits for the API
```

Then: `python3 bench/bench.py --host http://<head>:8000`.

## Runtime facts that matter

- **Image**: `vllm/vllm-openai:inkling-2-small` until Inkling 2 support lands
  in a vLLM release (currently nightly-only).
- **TP4 here means TP4 attention + EP4 experts**: the MoE runs expert-parallel
  across the TP group, so the RoCE fabric carries expert all-to-all, not just
  all-reduce. Keep the NCCL single-HCA pin from `.env` — the dual-rail pin +
  EP combination already cost a rig-day on another lane.
- **DSpark** drafts from the bundled speculator: k=7, probabilistic sampling
  (`greedy` loses 15–25% acceptance at temperature 1.0).
- **`--reasoning-parser inkling`** is required for the thinking split; tool
  calling uses the `inkling` tool-call parser (opt-in, unit-tested only).
- Vision: keep `--limit-mm-per-prompt` small; the ViT shares the same memory.

## References

- [Model card](https://huggingface.co/thinkingmachines/Inkling-2-Small) ·
  [vLLM recipe](https://recipes.vllm.ai/thinkingmachines/Inkling-2-Small) ·
  [vLLM blog](https://vllm.ai/blog/2026-11-12-inkling2-small) ·
  [announcement](https://thinkingmachines.ai/news/inkling-2-small/)
- Ring topology and fabric notes: sibling repos `spark-ds41f`,
  `weightless/lanes/inkling` (the 2-node Inkling-Small lane).
