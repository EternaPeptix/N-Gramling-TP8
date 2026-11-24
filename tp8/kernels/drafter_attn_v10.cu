// SPDX-License-Identifier: Apache-2.0
// SM121a (GB10) Generalized 6-Warp Fused Drafter Block-Attention v5 (B = 1..32, T_req <= 8, H = 6, Hkv = 1, D = 128)
//
// Upgrades Wave-V4 `drafter_attn_c1_v4.cu` (`B == 1` only) to ALL batch sizes `B = 1..32`:
//   1. Eliminates 3.0x redundant KV DRAM reads across ALL C=32 steps (`B = 2..32`, steady-state & tail drain):
//      stock Triton `_attn_kernel` (`BM=16, G=6 -> mtiles=3`) streamed every request's KV cache 3x across all
//      5 DSpark layers (`251.7 MB -> 83.9 MB` @ B=32, saving ~167.8 MB/step = 0.65-0.85 ms/step).
//   2. Eliminates all 5 separate `_merge_kernel` launches across ALL `B = 1..32` via per-request self-resetting
//      `SplitDone[req]` + hardware-coherent `st.global.cg` + `__threadfence()` + `atom.acq_rel.gpu` + `ld.global.cg`.
//   3. Pre-`gdc_wait()` Historical-KV Tile-0 Preload (`PDL=true`):
//      `CUQ`, `SEQ`, `BT`, and all past KV slots `< seqlen - qlen` are immutable during `qk_norm_rope_write_kv_`.
//      Splits whose initial 64-key tile is strictly historical (`k_beg + (t_beg + 1)*64 <= past`) issue tile-0 K
//      global streaming loads into `sKV[0]` BEFORE `griddepcontrol.wait`, hiding DRAM latency behind RoPE+KV-write
//      with ZERO extra live registers across the main loop!
//   4. Exports BOTH `drafter_attn_v5_fwd` (batched B=1..32 ABI) AND `drafter_attn_c1_v4_fwd` (100% drop-in v4 ABI).
//
// Strict SM121a Build Rules:
//   /usr/local/cuda/bin/nvcc -O3 -std=c++17 -gencode arch=compute_121a,code=sm_121a \
//       --fmad=false --shared -Xcompiler -fPIC --ptxas-options=-v drafter_attn_v5.cu -o libdrafter_attn_v5.so

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>

namespace {

static constexpr int MAX_T_DRAFT = 8;
static constexpr int H_DRAFT     = 6;
static constexpr int D_DRAFT     = 128;
static constexpr int BN_TILE     = 64;
static constexpr int SKV_STRIDE  = 136;  // 136 bf16 = 272 B = 68 words === 4 mod 32 banks -> 0-way K & V!
// V10: MAX_SPLITS=48 enables full 48-SM parallel scan on B=1@128K Global layer (48*8=384 <= 512 OP/LP slots!)
static constexpr int MAX_SPLITS  = 48;
static constexpr float LOG2E_F   = 1.4426950408889634f;
static constexpr float LN2_F     = 0.6931471805599453f;

static __device__ __forceinline__ void gdc_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

static __device__ __forceinline__ void gdc_launch_dependents() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

static __device__ __forceinline__ uint4 ld_cs_u4(const void* p) {
    uint4 v;
#if defined(__CUDA_ARCH__)
    asm volatile("ld.global.cs.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
#else
    v = *reinterpret_cast<const uint4*>(p);
#endif
    return v;
}

static __device__ __forceinline__ float ld_cg_f32(const float* p) {
    float v;
#if defined(__CUDA_ARCH__)
    asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(v) : "l"(p));
#else
    v = *p;
#endif
    return v;
}

static __device__ __forceinline__ float4 ld_cg_f4(const float* p) {
    float4 v;
#if defined(__CUDA_ARCH__)
    asm volatile("ld.global.cg.v4.f32 {%0,%1,%2,%3}, [%4];"
                 : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
                 : "l"(p));
#else
    v = make_float4(p[0], p[1], p[2], p[3]);
#endif
    return v;
}

static __device__ __forceinline__ void st_cg_f32(float* p, float v) {
#if defined(__CUDA_ARCH__)
    asm volatile("st.global.cg.f32 [%0], %1;" :: "l"(p), "f"(v) : "memory");
#else
    *p = v;
#endif
}

static __device__ __forceinline__ void st_cg_f2(float* p, float a, float b) {
#if defined(__CUDA_ARCH__)
    asm volatile("st.global.cg.v2.f32 [%0], {%1,%2};" :: "l"(p), "f"(a), "f"(b) : "memory");
#else
    p[0] = a; p[1] = b;
#endif
}

static __device__ __forceinline__ uint32_t atom_acq_rel_gpu_add(uint32_t* p, uint32_t v) {
    uint32_t old;
#if defined(__CUDA_ARCH__)
    asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                 : "=r"(old) : "l"(p), "r"(v) : "memory");
#else
    old = *p; *p = old + v;
#endif
    return old;
}

static __device__ __forceinline__ void st_rel_gpu_u32(uint32_t* p, uint32_t v) {
#if defined(__CUDA_ARCH__)
    asm volatile("st.release.gpu.global.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
#else
    *p = v;
#endif
}

static __device__ __forceinline__ float ex2_approx(float x) {
    float r;
#if defined(__CUDA_ARCH__)
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
#else
    r = exp2f(x);
#endif
    return r;
}

static __device__ __forceinline__ float div_rn(float a, float b) {
    float r;
#if defined(__CUDA_ARCH__)
    asm("div.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
#else
    r = a / b;
#endif
    return r;
}

static __device__ __forceinline__ uint32_t pack_bf16x2(float f0, float f1) {
    uint32_t out;
#if defined(__CUDA_ARCH__)
    asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(out) : "f"(f0), "f"(f1));
#else
    union { float f; uint32_t u; } x0, x1;
    x0.f = f0; x1.f = f1;
    uint32_t b0 = (x0.u + 0x7FFFu + ((x0.u >> 16) & 1u)) >> 16;
    uint32_t b1 = (x1.u + 0x7FFFu + ((x1.u >> 16) & 1u)) >> 16;
    out = (b0 & 0xFFFFu) | ((b1 & 0xFFFFu) << 16);
#endif
    return out;
}

static __device__ __forceinline__ void mma_m16n8k16_bf16(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)
    );
#else
    (void)d0; (void)d1; (void)d2; (void)d3;
    (void)a0; (void)a1; (void)a2; (void)a3; (void)b0; (void)b1;
#endif
}

static __device__ __forceinline__ void load_kv_tile_to_regs(
    const __nv_bfloat16* __restrict__ KV,
    const int32_t* __restrict__ bt_row,
    int64_t skv_b, int64_t skv_s,
    int BS, int bs_shift, int bs_mask,
    int n0, int k_end,
    int row_base, int d8,
    uint4 (&regs)[6]
) {
    const int d_off = d8 << 3;
    #pragma unroll
    for (int it = 0; it < 6; ++it) {
        const int n = row_base + it * 12;
        if (n < BN_TILE) {
            const int kpos = n0 + n;
            if (kpos < k_end) {
                const int blk_idx = (bs_shift >= 0) ? (kpos >> bs_shift) : (kpos / BS);
                const int slot    = (bs_shift >= 0) ? (kpos & bs_mask)   : (kpos - blk_idx * BS);
                const int64_t blk = static_cast<int64_t>(__ldg(bt_row + blk_idx));
                regs[it] = ld_cs_u4(KV + blk * skv_b + static_cast<int64_t>(slot) * skv_s + d_off);
            } else {
                regs[it] = make_uint4(0u, 0u, 0u, 0u);
            }
        }
    }
}

static __device__ __forceinline__ void store_regs_to_smem(
    const uint4 (&regs)[6],
    __nv_bfloat16* dst,
    int row_base, int d8
) {
    const int d_off = d8 << 3;
    #pragma unroll
    for (int it = 0; it < 6; ++it) {
        const int n = row_base + it * 12;
        if (n < BN_TILE) {
            *reinterpret_cast<uint4*>(dst + n * SKV_STRIDE + d_off) = regs[it];
        }
    }
}

// =========================================================================
// Generalized 6-Warp Fused Drafter Block-Attention (B = 1..32, T_req <= 8, H = 6, Hkv = 1, D = 128)
//   - Grid: (num_splits, num_reqs, 1), Block: (192, 1, 1), __launch_bounds__(192, 2)
//   - SMEM: sKV[2][64 * 136] = 34.0 KiB -> 2 CTAs/SM -> <= 168 regs/thr -> 0 spills!
// =========================================================================
template <bool PDL>
__global__ __launch_bounds__(192, 2)
void drafter_attn_v10_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const int32_t* __restrict__ BT,
    const int32_t* __restrict__ SEQ,
    const int32_t* __restrict__ CUQ,
    float* __restrict__ OP,              // [num_splits, q_cap, 6, 128] fp32 (unused when num_splits == 1)
    float* __restrict__ LP,              // [num_splits, q_cap, 6] fp32 (unused when num_splits == 1)
    uint32_t* __restrict__ SplitDone,    // [num_reqs] self-resetting (unused when num_splits == 1)
    __nv_bfloat16* __restrict__ Out,     // [total_q, 6, 128] bf16
    int64_t sq_t, int64_t sq_h,
    int64_t sk_b, int64_t sk_s,
    int64_t sv_b, int64_t sv_s,
    int64_t sbt_r,
    int64_t so_t, int64_t so_h,
    float scale_log2,
    int window,        // <= 0 = global (no window), > 0 = last `window` keys [max(seqlen-window,0), seqlen)
    int BS, int bs_shift, int bs_mask,
    int num_splits,
    int q_cap          // stride in tokens per split in OP/LP (>= total_q)
) {
    const int split    = static_cast<int>(blockIdx.x);
    const int req      = static_cast<int>(blockIdx.y);
    const int tid      = threadIdx.x;   // 0..191
    const int warp     = tid >> 5;      // 0..5 == query head h
    const int lane     = tid & 31;      // 0..31
    const int g        = lane >> 2;     // 0..7 == intra-req token index
    const int lt       = lane & 3;      // 0..3
    const int row_base = tid >> 4;      // 0..11
    const int d8       = tid & 15;      // 0..15

    // Metadata (CUQ, SEQ, BT) is prepared before layer 0 and immutable across qk_norm_rope_write_kv_.
    const int q0     = __ldg(CUQ + req);
    const int qlen_r = __ldg(CUQ + req + 1) - q0;
    const int qlen   = (qlen_r < 0) ? 0 : ((qlen_r > MAX_T_DRAFT) ? MAX_T_DRAFT : qlen_r);
    if (qlen == 0) {
        if constexpr (PDL) {
            gdc_wait();
            __syncthreads();
            if (tid == 0) gdc_launch_dependents();
        }
        return;
    }

    const int seqlen = __ldg(SEQ + req);
    const int k_end  = (seqlen > 0) ? seqlen : 0;
    const int past   = (k_end > qlen) ? (k_end - qlen) : 0;
    int k_beg = 0;
    if (window > 0) {
        k_beg = k_end - window;
        if (k_beg < 0) k_beg = 0;
    }
    const int span    = k_end - k_beg;
    const int n_tiles = (span > 0) ? ((span + BN_TILE - 1) / BN_TILE) : 0;
    const int t_beg   = (split * n_tiles) / num_splits;
    const int t_end   = ((split + 1) * n_tiles) / num_splits;
    const int32_t* bt_row = BT + static_cast<int64_t>(req) * sbt_r;

    alignas(16) __shared__ __nv_bfloat16 sKV[2][BN_TILE * SKV_STRIDE];
    __shared__ int32_t sIsLastSplit;

    // Pre-gdc_wait() Historical Tile-0 Preload:
    // When tile t_beg lies strictly in past KV (< past), K[t_beg] is already finalized in DRAM/L2!
    bool preloaded_k0 = false;
    if constexpr (PDL) {
        if (t_beg < t_end && (k_beg + (t_beg + 1) * BN_TILE) <= past) {
            uint4 kv_pre[6];
            load_kv_tile_to_regs(K, bt_row, sk_b, sk_s, BS, bs_shift, bs_mask,
                                 k_beg + t_beg * BN_TILE, k_end, row_base, d8, kv_pre);
            store_regs_to_smem(kv_pre, sKV[0], row_base, d8);
            preloaded_k0 = true;
        }
        gdc_wait();
        // Wave-V9 PDL Fix: NEVER call griddepcontrol.launch_dependents at entry before Out is written!
        // Dependents are launched strictly at epilogue after Out stores complete (0 SM starvation / 0 3-grid overlap).
    }

    float m_i = -INFINITY;
    float l_i = 0.0f;
    float o_acc[16][2];
    #pragma unroll
    for (int nd = 0; nd < 16; ++nd) {
        o_acc[nd][0] = 0.0f;
        o_acc[nd][1] = 0.0f;
    }

    if (t_beg < t_end) {
        uint32_t q_reg[8][2];
        if (g < qlen) {
            const __nv_bfloat16* qp = Q + static_cast<int64_t>(q0 + g) * sq_t + static_cast<int64_t>(warp) * sq_h;
            #pragma unroll
            for (int kb = 0; kb < 8; ++kb) {
                const int k0 = kb * 16;
                q_reg[kb][0] = *reinterpret_cast<const uint32_t*>(qp + k0 + 2 * lt);
                q_reg[kb][1] = *reinterpret_cast<const uint32_t*>(qp + k0 + 8 + 2 * lt);
            }
        } else {
            #pragma unroll
            for (int kb = 0; kb < 8; ++kb) {
                q_reg[kb][0] = 0u;
                q_reg[kb][1] = 0u;
            }
        }

        uint4 kv_regs[6];
        if (!preloaded_k0) {
            load_kv_tile_to_regs(K, bt_row, sk_b, sk_s, BS, bs_shift, bs_mask,
                                 k_beg + t_beg * BN_TILE, k_end, row_base, d8, kv_regs);
            store_regs_to_smem(kv_regs, sKV[0], row_base, d8);
        }
        __syncthreads();

        for (int ti = t_beg; ti < t_end; ++ti) {
            const int n0 = k_beg + ti * BN_TILE;

            // Issue V_ti global loads BEFORE Q @ K_ti^T so V latency hides behind 64 MMAs + softmax!
            load_kv_tile_to_regs(V, bt_row, sv_b, sv_s, BS, bs_shift, bs_mask,
                                 n0, k_end, row_base, d8, kv_regs);

            // 1. Q @ K_ti^T (M=8, N=64, K=128) -> s_val[8][2]
            float s_val[8][2];
            #pragma unroll
            for (int ns = 0; ns < 8; ++ns) {
                float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
                const __nv_bfloat16* krow = sKV[0] + (ns * 8 + g) * SKV_STRIDE + 2 * lt;
                #pragma unroll
                for (int kb = 0; kb < 8; ++kb) {
                    const int k0 = kb * 16;
                    const uint32_t b0 = *reinterpret_cast<const uint32_t*>(krow + k0);
                    const uint32_t b1 = *reinterpret_cast<const uint32_t*>(krow + k0 + 8);
                    mma_m16n8k16_bf16(d0, d1, d2, d3,
                                      q_reg[kb][0], 0u, q_reg[kb][1], 0u,
                                      b0, b1);
                }
                s_val[ns][0] = d0;
                s_val[ns][1] = d1;
            }

            // 2. Scale, boundary mask, and in-register quad online softmax (2 shuffles, 0 SMEM!)
            float m_tile = -INFINITY;
            const bool row_ok = (g < qlen);
            #pragma unroll
            for (int ns = 0; ns < 8; ++ns) {
                const int kp0 = n0 + ns * 8 + 2 * lt;
                const int kp1 = kp0 + 1;
                s_val[ns][0] = (row_ok && kp0 < k_end) ? (s_val[ns][0] * scale_log2) : -INFINITY;
                s_val[ns][1] = (row_ok && kp1 < k_end) ? (s_val[ns][1] * scale_log2) : -INFINITY;
                m_tile = fmaxf(m_tile, fmaxf(s_val[ns][0], s_val[ns][1]));
            }
            m_tile = fmaxf(m_tile, __shfl_xor_sync(0xFFFFFFFFu, m_tile, 1));
            m_tile = fmaxf(m_tile, __shfl_xor_sync(0xFFFFFFFFu, m_tile, 2));

            const float m_new  = fmaxf(m_i, m_tile);
            const float m_safe = (m_new == -INFINITY) ? 0.0f : m_new;
            const float alpha  = (m_i == -INFINITY) ? 0.0f : ex2_approx(m_i - m_safe);

            float l_tile = 0.0f;
            #pragma unroll
            for (int ns = 0; ns < 8; ++ns) {
                const float p0 = (s_val[ns][0] == -INFINITY) ? 0.0f : ex2_approx(s_val[ns][0] - m_safe);
                const float p1 = (s_val[ns][1] == -INFINITY) ? 0.0f : ex2_approx(s_val[ns][1] - m_safe);
                s_val[ns][0] = p0;
                s_val[ns][1] = p1;
                l_tile += (p0 + p1);
            }
            l_tile += __shfl_xor_sync(0xFFFFFFFFu, l_tile, 1);
            l_tile += __shfl_xor_sync(0xFFFFFFFFu, l_tile, 2);
            l_i = l_i * alpha + l_tile;
            m_i = m_new;

            #pragma unroll
            for (int nd = 0; nd < 16; ++nd) {
                o_acc[nd][0] *= alpha;
                o_acc[nd][1] *= alpha;
            }

            uint32_t p_reg[4][2];
            #pragma unroll
            for (int kp = 0; kp < 4; ++kp) {
                p_reg[kp][0] = pack_bf16x2(s_val[2 * kp + 0][0], s_val[2 * kp + 0][1]);
                p_reg[kp][1] = pack_bf16x2(s_val[2 * kp + 1][0], s_val[2 * kp + 1][1]);
            }

            store_regs_to_smem(kv_regs, sKV[1], row_base, d8);
            __syncthreads();

            if (ti + 1 < t_end) {
                load_kv_tile_to_regs(K, bt_row, sk_b, sk_s, BS, bs_shift, bs_mask,
                                     n0 + BN_TILE, k_end, row_base, d8, kv_regs);
            }

            const uint16_t* sV16 = reinterpret_cast<const uint16_t*>(sKV[1]);
            #pragma unroll
            for (int kp = 0; kp < 4; ++kp) {
                const int r0 = kp * 16 + 2 * lt;
                const int r1 = r0 + 1;
                const int r2 = r0 + 8;
                const int r3 = r2 + 1;
                const uint32_t pa0 = p_reg[kp][0];
                const uint32_t pa2 = p_reg[kp][1];
                const uint16_t* vr0 = sV16 + r0 * SKV_STRIDE + g;
                const uint16_t* vr1 = sV16 + r1 * SKV_STRIDE + g;
                const uint16_t* vr2 = sV16 + r2 * SKV_STRIDE + g;
                const uint16_t* vr3 = sV16 + r3 * SKV_STRIDE + g;

                #pragma unroll
                for (int nd = 0; nd < 16; ++nd) {
                    const int d0 = nd * 8;
                    const uint32_t b0 = static_cast<uint32_t>(vr0[d0]) | (static_cast<uint32_t>(vr1[d0]) << 16);
                    const uint32_t b1 = static_cast<uint32_t>(vr2[d0]) | (static_cast<uint32_t>(vr3[d0]) << 16);
                    float d2_tmp = 0.0f, d3_tmp = 0.0f;
                    mma_m16n8k16_bf16(o_acc[nd][0], o_acc[nd][1], d2_tmp, d3_tmp,
                                      pa0, 0u, pa2, 0u,
                                      b0, b1);
                }
            }

            if (ti + 1 < t_end) {
                store_regs_to_smem(kv_regs, sKV[0], row_base, d8);
                __syncthreads();
            }
        }
    }

    // ---- Fast path: num_splits == 1 writes directly to Out in bf16 (zero scratch/atomics!) ----
    if (num_splits == 1) {
        if (g < qlen) {
            const float inv_l = (l_i > 0.0f) ? div_rn(1.0f, l_i) : 0.0f;
            __nv_bfloat16* op = Out + static_cast<int64_t>(q0 + g) * so_t + static_cast<int64_t>(warp) * so_h;
            #pragma unroll
            for (int nd = 0; nd < 16; ++nd) {
                *reinterpret_cast<uint32_t*>(op + nd * 8 + 2 * lt) =
                    pack_bf16x2(o_acc[nd][0] * inv_l, o_acc[nd][1] * inv_l);
            }
        }
        if constexpr (PDL) {
            __syncthreads();
            if (tid == 0) gdc_launch_dependents();
        }
        return;
    }

    // ---- Multi-split path: write partial OP/LP via st.global.cg -> __threadfence() -> per-req last-split merge ----
    const int64_t tok_idx = static_cast<int64_t>(split) * q_cap + (q0 + g);
    if (g < qlen) {
        if (t_beg < t_end && l_i > 0.0f) {
            const float inv_l = div_rn(1.0f, l_i);
            float* op_row = OP + (tok_idx * H_DRAFT + warp) * D_DRAFT;
            #pragma unroll
            for (int nd = 0; nd < 16; ++nd) {
                st_cg_f2(op_row + nd * 8 + 2 * lt, o_acc[nd][0] * inv_l, o_acc[nd][1] * inv_l);
            }
            if (lt == 0) {
                const float lse = (m_i + log2f(l_i)) * LN2_F;
                st_cg_f32(LP + tok_idx * H_DRAFT + warp, lse);
            }
        } else if (lt == 0) {
            st_cg_f32(LP + tok_idx * H_DRAFT + warp, -INFINITY);
        }
    }

    __threadfence();
    __syncthreads();
    if (tid == 0) {
        uint32_t* sem_req = SplitDone + req;
        const uint32_t prev = atom_acq_rel_gpu_add(sem_req, 1u);
        if (prev == static_cast<uint32_t>(num_splits - 1)) {
            st_rel_gpu_u32(sem_req, 0u);
            sIsLastSplit = 1;
        } else {
            sIsLastSplit = 0;
        }
    }
    __syncthreads();

    if (sIsLastSplit) {
        const int d_col = lane * 4;  // 32 lanes * 4 = 128 == D_DRAFT
        const int lane_hi = lane + 32;
        for (int tok = 0; tok < qlen; ++tok) {
            const int64_t g_tok = static_cast<int64_t>(q0 + tok);
            const float lse_s0 = (lane < num_splits)
                ? ld_cg_f32(LP + ((static_cast<int64_t>(lane) * q_cap + g_tok) * H_DRAFT + warp))
                : -INFINITY;
            const float lse_s1 = (lane_hi < num_splits)
                ? ld_cg_f32(LP + ((static_cast<int64_t>(lane_hi) * q_cap + g_tok) * H_DRAFT + warp))
                : -INFINITY;
            float mx = fmaxf(lse_s0, lse_s1);
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, off));
            }
            const float mx_safe = (mx == -INFINITY) ? 0.0f : mx;
            const bool is_live0 = (lane < num_splits) && (lse_s0 > -1e30f);
            const bool is_live1 = (lane_hi < num_splits) && (lse_s1 > -1e30f);
            const float w_s0 = is_live0 ? expf(lse_s0 - mx_safe) : 0.0f;
            const float w_s1 = is_live1 ? expf(lse_s1 - mx_safe) : 0.0f;
            float tot = (num_splits > 32) ? (w_s0 + w_s1) : w_s0;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                tot += __shfl_xor_sync(0xFFFFFFFFu, tot, off);
            }
            const float inv_tot = (tot > 0.0f) ? div_rn(1.0f, tot) : 0.0f;

            float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            uint32_t am0 = __ballot_sync(0xFFFFFFFFu, is_live0);
            while (am0 != 0u) {
                const int s = __ffs(static_cast<int>(am0)) - 1;
                am0 &= (am0 - 1u);
                const float ws = __shfl_sync(0xFFFFFFFFu, w_s0, s);
                const float4 ov = ld_cg_f4(
                    OP + (((static_cast<int64_t>(s) * q_cap + g_tok) * H_DRAFT + warp) * D_DRAFT + d_col)
                );
                acc.x += ov.x * ws;
                acc.y += ov.y * ws;
                acc.z += ov.z * ws;
                acc.w += ov.w * ws;
            }
            if (num_splits > 32) {
                uint32_t am1 = __ballot_sync(0xFFFFFFFFu, is_live1);
                while (am1 != 0u) {
                    const int s_lo = __ffs(static_cast<int>(am1)) - 1;
                    am1 &= (am1 - 1u);
                    const float ws = __shfl_sync(0xFFFFFFFFu, w_s1, s_lo);
                    const float4 ov = ld_cg_f4(
                        OP + (((static_cast<int64_t>(32 + s_lo) * q_cap + g_tok) * H_DRAFT + warp) * D_DRAFT + d_col)
                    );
                    acc.x += ov.x * ws;
                    acc.y += ov.y * ws;
                    acc.z += ov.z * ws;
                    acc.w += ov.w * ws;
                }
            }
            const uint32_t b01 = pack_bf16x2(acc.x * inv_tot, acc.y * inv_tot);
            const uint32_t b23 = pack_bf16x2(acc.z * inv_tot, acc.w * inv_tot);
            *reinterpret_cast<uint2*>(
                Out + g_tok * so_t + static_cast<int64_t>(warp) * so_h + d_col
            ) = make_uint2(b01, b23);
        }
        if constexpr (PDL) {
            __syncthreads();
            if (tid == 0) gdc_launch_dependents();
        }
    }
}

template <typename KernelT, typename... Args>
static inline cudaError_t launch_draft_attn_v5(
    KernelT kernel, dim3 grid, dim3 block, cudaStream_t stream, bool pdl, Args... args
) {
    if (!pdl) {
        kernel<<<grid, block, 0, stream>>>(args...);
        return cudaGetLastError();
    }
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.stream = stream;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    return cudaLaunchKernelEx(&cfg, kernel, args...);
}

}  // namespace

extern "C" int drafter_attn_v10_version() {
    return 10;
}

extern "C" int drafter_attn_v5_fwd(
    const void* Q,
    const void* K,
    const void* V,
    const void* BT,
    const void* SEQ,
    const void* CUQ,
    void* OP,
    void* LP,
    void* SplitDone,
    void* Out,
    int64_t sq_t, int64_t sq_h,
    int64_t sk_b, int64_t sk_s,
    int64_t sv_b, int64_t sv_s,
    int64_t sbt_r,
    int64_t so_t, int64_t so_h,
    float softmax_scale,
    int window,
    int BS,
    int num_reqs,
    int num_splits,
    int q_cap,
    int pdl,
    uintptr_t stream_ptr
) {
    if (BS <= 0 || num_reqs <= 0 || num_splits <= 0 || num_splits > MAX_SPLITS) return -1;
    if (num_splits > 1 && (OP == nullptr || LP == nullptr || SplitDone == nullptr || q_cap <= 0)) return -2;

    int bs_shift = -1;
    int bs_mask = 0;
    if ((BS & (BS - 1)) == 0) {
        bs_shift = __builtin_ctz(static_cast<unsigned int>(BS));
        bs_mask = BS - 1;
    }

    const float scale_log2 = softmax_scale * LOG2E_F;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    dim3 grid(static_cast<unsigned int>(num_splits), static_cast<unsigned int>(num_reqs), 1u);
    dim3 block(192u, 1u, 1u);

    cudaError_t err;
    if (pdl != 0) {
        err = launch_draft_attn_v5(
            drafter_attn_v10_kernel<true>, grid, block, stream, true,
            static_cast<const __nv_bfloat16*>(Q), static_cast<const __nv_bfloat16*>(K),
            static_cast<const __nv_bfloat16*>(V), static_cast<const int32_t*>(BT),
            static_cast<const int32_t*>(SEQ), static_cast<const int32_t*>(CUQ),
            static_cast<float*>(OP), static_cast<float*>(LP), static_cast<uint32_t*>(SplitDone),
            static_cast<__nv_bfloat16*>(Out),
            sq_t, sq_h, sk_b, sk_s, sv_b, sv_s, sbt_r, so_t, so_h,
            scale_log2, window, BS, bs_shift, bs_mask, num_splits, q_cap
        );
    } else {
        err = launch_draft_attn_v5(
            drafter_attn_v10_kernel<false>, grid, block, stream, false,
            static_cast<const __nv_bfloat16*>(Q), static_cast<const __nv_bfloat16*>(K),
            static_cast<const __nv_bfloat16*>(V), static_cast<const int32_t*>(BT),
            static_cast<const int32_t*>(SEQ), static_cast<const int32_t*>(CUQ),
            static_cast<float*>(OP), static_cast<float*>(LP), static_cast<uint32_t*>(SplitDone),
            static_cast<__nv_bfloat16*>(Out),
            sq_t, sq_h, sk_b, sk_s, sv_b, sv_s, sbt_r, so_t, so_h,
            scale_log2, window, BS, bs_shift, bs_mask, num_splits, q_cap
        );
    }
    return static_cast<int>(err);
}

// 100% backwards-compatible C=1 v4 ABI wrapper (num_reqs = 1, sbt_r = 0, q_cap = 8)
extern "C" int drafter_attn_c1_v4_fwd(
    const void* Q,
    const void* K,
    const void* V,
    const void* BT,
    const void* SEQ,
    const void* CUQ,
    void* OP,
    void* LP,
    void* SplitDone,
    void* Out,
    int64_t sq_t, int64_t sq_h,
    int64_t sk_b, int64_t sk_s,
    int64_t sv_b, int64_t sv_s,
    int64_t so_t, int64_t so_h,
    float softmax_scale,
    int window,
    int BS,
    int num_splits,
    int pdl,
    uintptr_t stream_ptr
) {
    return drafter_attn_v5_fwd(
        Q, K, V, BT, SEQ, CUQ, OP, LP, SplitDone, Out,
        sq_t, sq_h, sk_b, sk_s, sv_b, sv_s, 0, so_t, so_h,
        softmax_scale, window, BS, 1, num_splits, MAX_T_DRAFT, pdl, stream_ptr
    );
}
