# Notes: End-to-End 4-Bit Training (Thinking Machines, 2026-11-13)

Source: <https://thinkingmachines.ai/blog/end-to-end-4-bit-training/>
Why we care: Inkling 2 Small (the model this repo serves) is the artifact
this recipe produced. The recipe explains the checkpoint's mixed precision,
its weight footprint, and the EP traffic our ring has to carry.

## The claim

Largest known model trained end-to-end with a native 4-bit forward pass:
same numerics from pretraining start through final RL rollout. No post-hoc
quantization, no calibration. NVFP4 is the model's native precision, which
is why the model card recommends it over MXFP8 (1-2 pp eval gain, not
meaningful in practice).

## The recipe by stage

| stage | format | reason |
|---|---|---|
| MoE forward | NVFP4 + dedicated kernel | 15 vs 5 vs 2.5 PFLOPS (NVFP4/MXFP8/BF16 on GB300); MoE layers = most FLOPs, least accuracy harm |
| backward | MXFP8, re-quantized with stochastic rounding | backward tolerates variance (averages out) but not bias (compounds); SR keeps re-quantized tensors equal in expectation |
| last 1/6 of layers | MXFP8 | higher precision where it matters most |
| RL sampling | NVFP4 | no discernible reward difference; 20% faster RL |

Full NVFP4 everywhere (not just MoE) destroyed compute efficiency in their
experiments. MoE-only NVFP4 was the tradeoff that held.

## What this explains about the checkpoint

- **Layers 40-47 are MXFP8 = the "last 1/6 in higher precision" rule**
  (8/48 = 16.6%). The mixed-precision checkpoint is the recipe, not a
  packaging accident. It is also why the weights land ~370-390 GB on our
  ring instead of the ~300 GB an all-NVFP4 read suggests.
- NVFP4 costs 4.5 bits/value effective ((4*16+8)/16: FP4 value + one E4M3
  scale per 16). Plus an FP32 outer scale per tensor (or per-expert /
  per-token, see below) to shift the representable range.

## The transpose problem (the real crux)

NVFP4 block scales are 1x16 and not transpose-invariant; the backward pass
needs the transposed grouping, so it cannot reuse the forward's quantized
tensors. NVIDIA's fix (16x16 tiles, transpose-invariant by construction)
flushes small values to zero around outliers — too coarse, hurt CE past
what they accepted. Their fix: de-quantize, transpose in high precision,
re-quantize with **stochastic rounding** to MXFP8. Unbiased gradients; SR
is hardware-accelerated on Blackwell. (Inspiration: Quartet II, which uses
randomized rotations + bias correction instead.)

## Quantized EP communications — the part our ring should watch

After the GEMM speedup, expert-parallel all-to-all became the bottleneck:
15 PFLOPS against 900 GB/s NVLink = 16,000 FLOPs per received byte, so
Tensor Cores idle waiting on tokens.

- **Dispatch in NVFP4.** Backward consumes quantized (not high-precision)
  activations from the forward pass, which covers the WGRAD input too.
- **Combine via 32-dim random Hadamard transform + block-scaled S2F6**
  (8-bit fixed-point). The RHT smooths outliers; post-rotation an integer
  format beats a float format at this precision. Combine drops 16 -> 8.25
  bits/value; fused into the w2 GEMM epilogue with Blackwell's native
  `cvt.*.s2f6x2` PTX instructions. No measurable accuracy degradation.

Our EP4 all-to-all rides 200 Gb/s RoCE — ~4.5x slower than the NVLink this
optimization targets, so the same imbalance hits us harder. Whether vLLM's
Inkling-2 EP path inherits any comm quantization is an open question worth
checking before we tune the lane.

## Kernels

BBBK ("Big Beautiful Blackwell Kernel"): saturates the GB300 2SM UMMA
(K=96) with a 6-stage multiplicand + 7-stage scale-factor pipeline; fuses
RHT+S2F6 into GEMM epilogues so ALU-heavy steps (scale factors, SR,
re-quantization) never stall the Tensor Cores. Fully deterministic.
GB300-shaped: GB10 has NVFP4 Tensor Cores but not that pipeline — expect
different speed ratios on our hardware.

## The ledger

- ~2% compute-efficiency cost in pretraining for the NVFP4 recipe
- 8% wall-clock pretraining speedup, 20% RL speedup vs full MXFP8
- Inkling 2 Small pretrained in 68 hours on GB300 NVL72

## Techniques worth stealing

- **Four Over Six (4/6)**: try two candidate scale factors per activation
  block, keep the better one (weights skip it).
- **Per-expert outer scales** for weights, **per-token outer scales** for
  activations (from Composer 2): batch-invariant, no causality leakage.
- Keep the last 1/6 of layers in higher precision (NVIDIA uses 15%; they
  use 1/6 because their ladder scales layer count by 6).

## Open questions for the lane

- Does vLLM's Inkling-2 EP implementation quantize dispatch/combine at
  all, and in what format? (Grep the image's EP path before tuning.)
- NVFP4 GEMM throughput on GB10 vs the GB300 figures quoted here — needs
  a measurement on our ring once the lane is up.
- The MXFP8 layers 40-47 are the natural first target if a future
  repack ever wants the last ~40 GB back.
