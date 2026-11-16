#!/usr/bin/env python3
"""Does b12x (fused Trellis MoE, W4A16) compile and run on GB10 (sm121)?

Credit: the fused Trellis MoE kernels are Local Inference Lab's b12x
(github.com/local-inference-lab/b12x, now in FlashInfer
flashinfer/experimental/b12x), with the Inkling-2 integration recipe from
0xSero's inkling2-exl3 (github.com/0xSero/inkling2-exl3). Result on
2026-11-16: COMPILED for sm121 in ~5 s, run OK in ~6 ms.
"""
import sys
import time

sys.path.insert(0, "/tmp/flashinfer/flashinfer/experimental/b12x")
sys.path.insert(0, "/tmp/flashinfer/flashinfer/experimental/b12x/_compat")

import torch

from b12x.moe._shared.kernels.w4a16.host import make_w4a16_packed_buffers
from b12x.moe._shared.kernels.w4a16.kernel import (
    compile_w4a16_fused_moe,
    compile_w4a16_topk_sum,
    run_w4a16_moe,
)
from b12x.moe._shared.kernels.w4a16.route_pack import compile_w4a16_route_pack_launches
from b12x.moe._shared.kernels.w4a16.prepare import prepare_trellis256_moe_weights

props = torch.cuda.get_device_properties(0)
print(f"device: {props.name} sm{props.major}{props.minor}, sms={props.multi_processor_count}")

E, H, I, K, T = 8, 4608, 2304, 2, 8
act = "silu"

print("prepare_trellis256_moe_weights ...", flush=True)
t0 = time.time()
prepared = prepare_trellis256_moe_weights(
    hidden_size=H,
    intermediate_size=I,
    num_experts=E,
    activation=act,
    fc1_tile_n=128,
    fc2_tile_n=512,
    device="cuda",
    seed=1,
)
print(f"  prepared in {time.time() - t0:.1f}s", flush=True)

print("compile_w4a16_fused_moe (the arch gate) ...", flush=True)
t0 = time.time()
sms = props.multi_processor_count
max_shared = props.shared_memory_per_block_optin
rp = compile_w4a16_route_pack_launches(
    tokens=T, topk=K, block_size=32, num_experts=E, ordinal=0,
)
topk_sum = compile_w4a16_topk_sum(
    m=T, topk=K, hidden_size=H, num_experts=0, route_num_experts=0,
    route_ids_dtype=torch.int32,
)
fused = compile_w4a16_fused_moe(
    size_m=T,
    hidden_size=H,
    intermediate_size=I,
    num_experts=E,
    top_k=K,
    activation=act,
    apply_router_weight_on_input=False,
    zero_fc2_output=False,
    moe_block_size=32,
    max_m_blocks=16,
    sms=sms,
    max_shared_mem=max_shared,
    weight_layout="trellis_t256",
    w13_layout="packed",
    scale_format="e4m3_k32",
    trellis_codebook="mcg",
)
print(f"  COMPILED for sm{props.major}{props.minor} in {time.time() - t0:.1f}s", flush=True)

x = torch.randn(T, H, dtype=torch.bfloat16, device="cuda")
tw = torch.rand(T, K, dtype=torch.float32, device="cuda")
tw = tw / tw.sum(-1, keepdim=True)
ids = torch.stack(
    [torch.randperm(E, device="cuda")[:K] for _ in range(T)]
).to(torch.int32)
out = torch.zeros(T, H, dtype=torch.bfloat16, device="cuda")
bufs = make_w4a16_packed_buffers(prepared, m=T, topk=K, dtype=torch.bfloat16, device=x.device)
c13, c2 = bufs.intermediate_cache13, bufs.intermediate_cache2

t0 = time.time()
run_w4a16_moe(
    x,
    prepared,
    tw,
    ids,
    activation=act,
    intermediate_cache13=c13,
    intermediate_cache2=c2,
    output=out,
    fused_launch=fused,
    topk_sum_launch=topk_sum,
    route_pack_launches=rp,
)
torch.cuda.synchronize()
print(f"run_w4a16_moe OK in {(time.time() - t0) * 1000:.1f} ms; "
      f"out std={out.float().std().item():.4f} nonzero={int((out != 0).sum())}", flush=True)
print("SM121 GATE: PASS", flush=True)
