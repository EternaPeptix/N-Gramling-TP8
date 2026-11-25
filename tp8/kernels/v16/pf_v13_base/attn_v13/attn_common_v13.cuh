// SPDX-License-Identifier: Apache-2.0
// ============================================================================
// Wave-V13 SM121a (GB10) Modular Split-TU 0-ULP Fused Decode Attention Header
//   (`attn_common_v13.cuh` -> `libfused_decode_attn_exact_sm121_v13.so`)
//
// Fixes the Wave-V11 `v11/attn_s48` `cicc` compile hang via 3 orthogonal cuts:
//   1. Split into 5 parallel TUs (`attn_prologue_v13.cu`, `attn_swa_v13.cu`,
//      `attn_global_s48_v13.cu`, `attn_global_bf12_v13.cu`, `attn_dispatch_v13.cu`)
//      compiled with `-j4` (standard `nvcc -c` -> `nvcc --shared`, no `-dc`!).
//   2. Collapse dead/redundant template axes:
//      - `NUM_SPLITS` (11 values) -> `S_MODE in {0: S==1, 1: 2<=S<=32, 2: 33<=S<=64}`
//        with 0-ULP identical `reduce_splits_sum_warp_rt(my_w, num_splits)`!
//      - `GUARD_MODE` (3 values)  -> 1 uniform pre-loop runtime `guard_mode`!
//      - `TREE_MODE == -1` (dead Wave-S8 diagnostic sweep) pruned (-202 kernels!).
//   3. Factor Global split-phase `fusedmt` loop (ZERO loop-body duplication:
//      2 `unpack` + 2 `issue` copies per loop instead of 5 `unpack` + 9 `issue`!).
//   4. Upgrade KV-BF12 codec to Wave-V12 branchless 2-wide SWAR (`-41%` ALU).
//
// 0-ULP Bit-Exactness Invariants (vs Triton `sm12_attn._attn_kernel`, 1dcfb306):
//   - Order-A `fma.rn.f32(scale_log2, s, __fmul_rn(b, LOG2E))`
//   - `ex2.approx.ftz.f32` softmax, `BL1_2x2` (Local) / `B0` (Global) sum(p)
//   - `div.full.f32`, `ex2.approx.f32` MGS weights, exact SP=next_pow2(S) tree
//   - Strictly 0 B stack frame, 0 B spills, `<= 50,336 B <= 50,688 B` SMEM (2 CTAs/SM),
//     0/7 `#206` PDL hazards!
// ============================================================================
#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

#include "kv_bf12_swar_v13.cuh"

namespace attn_v13 {

static constexpr float LOG2E = 1.4426950408889634f;
static constexpr float LN2   = 0.6931471805599453f;
static constexpr int   D_DIM = 128;
static constexpr int   D_REL = 16;
static constexpr int   BN    = 64;
static constexpr int   BM    = 16;
static constexpr int   B_COLS     = 80;
static constexpr int   B_COLS_FMT = 72;
static constexpr int   P_COLS     = 72;
static constexpr int   kFusedMt3SmemBytes = 49664;

// Split-TU internal launch parameter bundle (POD, host-only pass-through)
struct AttnLaunchParams {
    const void* Q;
    const void* R;
    const void* RScale;
    const void* Tau;
    const void* RelProjT;
    const void* K;
    const void* V;
    const void* BT;
    const void* SEQ;
    const void* CUQ;
    const void* GATE;
    const void* SINK;
    void*       OP;
    void*       LP;
    void*       SEM;
    void*       Y;
    const void* NLivePtr;
    const void* TreeMeta;
    int64_t sq_t, sq_h, sr_t, sr_h;
    int64_t sk_b, sk_s, sk_h, sv_b, sv_s, sv_h;
    int64_t sbt_r, sg_t, ssk_t, sy_t;
    int64_t so_s, so_t, so_h, sl_s, sl_t, sl_h;
    float scale_log2, rel_threshold, clip_eps;
    int T, NT, mtiles, num_reqs, Hkv, BS, E, G;
    int window, num_splits, has_tau, has_nlive, use_pdl, trig_wo;
    int fuse_mt, kv_evict_first, guard_mode, kv_tok_thresh;
    uint64_t stream_u64;
    const void* KEmergRaw;
    const void* VEmergRaw;
};

// Cross-TU host launch entrypoints (defined in their respective device TUs)
int launch_prologue_v13(
    bool kv_bf12,
    const void* K, const void* V, const void* WK, const void* WV, const void* DK, const void* DV,
    void* ConvCache, void* KeyCache, void* ValueCache,
    const void* KVSlot, const void* Pos, const void* SeqIdx, const void* Slot, const void* BlockTable, const void* QStart,
    const void* Q, const void* QScale, const void* Tau, void* QOut, const void* NLivePtr,
    const void* TreeSconvAnc,
    int T, int64_t stride_k, int64_t stride_v, int64_t stride_w0, int64_t stride_w1,
    int64_t stride_dk, int64_t stride_dv, int64_t stride_c_blk, int64_t stride_c_n,
    int64_t stride_kv_blk, int64_t stride_kv_n, int kv_bs,
    int64_t stride_bt, int max_blocks, int N_cache, int64_t stride_q,
    float scale, float clip_threshold, float clip_eps, float norm_eps,
    int dynamic, int has_tau, int has_nlive, int use_pdl, bool has_tree, uint64_t stream_u64,
    void* KEmergRaw, void* KEmergUsed, void* VEmergRaw, void* VEmergUsed, uint32_t emerg_cap
);

int init_smem_swa_v13();
int init_smem_global_bf16_v13();
int init_smem_global_bf12_v13();

int launch_swa_mgs_v13(const AttnLaunchParams& p, bool kv_bf12, bool has_tree);
int launch_swa_fusedmt_v13(const AttnLaunchParams& p, bool kv_bf12, bool smem_inited);

int launch_global_bf16_mgs_v13(const AttnLaunchParams& p, bool has_tree);
int launch_global_bf16_fusedmt_v13(const AttnLaunchParams& p, bool smem_inited);

int launch_global_bf12_mgs_v13(const AttnLaunchParams& p, bool has_tree);
int launch_global_bf12_fusedmt_v13(const AttnLaunchParams& p, bool smem_inited);

// =========================================================================
// Shared SM121a PTX Primitives (Device-Only)
// =========================================================================
#if defined(__CUDACC__) || defined(ATTN_V13_DEVICE_TU)

static __device__ __forceinline__ void gdc_launch_dependents() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

static __device__ __forceinline__ void gdc_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

static __device__ __forceinline__ int atomic_add_acq_rel_gpu(int32_t* ptr, int32_t val) {
    int32_t old;
    asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;"
                 : "=r"(old) : "l"(ptr), "r"(val) : "memory");
    return old;
}

static __device__ __forceinline__ float ld_cg_f32(const float* p) {
    float v;
    asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(v) : "l"(p));
    return v;
}

static __device__ __forceinline__ float4 ld_cg_f4(const float* p) {
    float4 v;
    asm volatile("ld.global.cg.v4.f32 {%0,%1,%2,%3}, [%4];"
                 : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(p));
    return v;
}

static __device__ __forceinline__ float fma_rn_f32(float a, float b, float c) {
    float y;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(y) : "f"(a), "f"(b), "f"(c));
    return y;
}

static __device__ __forceinline__ float ex2_approx_f32(float x) {
    float y;
    asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

static __device__ __forceinline__ float ex2_approx_ftz_f32(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

static __device__ __forceinline__ float rcp_approx_f32(float x) {
    float y;
    asm("rcp.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

static __device__ __forceinline__ float rsqrt_approx_f32(float x) {
    float y;
    asm("rsqrt.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

static __device__ __forceinline__ float div_full_f32(float a, float b) {
    float y;
    asm("div.full.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
    return y;
}

static __device__ __forceinline__ float triton_sigmoid(float x) {
    const float neg_x_log2e = __fmul_rn(x, -LOG2E);
    const float e = ex2_approx_f32(neg_x_log2e);
    return rcp_approx_f32(__fadd_rn(e, 1.0f));
}

static __device__ __forceinline__ float bf16_to_f32(uint16_t b) {
    float f;
    asm("cvt.f32.bf16 %0, %1;" : "=f"(f) : "h"(b));
    return f;
}

static __device__ __forceinline__ uint16_t f32_to_bf16(float f) {
    __nv_bfloat16 b = __float2bfloat16_rn(f);
    return *reinterpret_cast<const uint16_t*>(&b);
}

static __device__ __forceinline__ uint32_t pack_bf16x2(uint16_t a, uint16_t b) {
    return static_cast<uint32_t>(a) | (static_cast<uint32_t>(b) << 16);
}

static __device__ __forceinline__ void divmod_p2(int64_t val, int n_mod, int n_shift, int64_t& q, int64_t& r) {
    if (n_shift >= 0) {
        q = val >> n_shift;
        r = val & ((1LL << n_shift) - 1);
    } else {
        q = val / n_mod;
        r = val % n_mod;
    }
}

static __device__ __forceinline__ uint64_t make_l2_evict_first_policy() {
    uint64_t policy = 0;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    float frac = 1.0f;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, %1;"
                 : "=l"(policy) : "f"(frac));
#endif
    return policy;
}

static __device__ __forceinline__ void cp_async_cg_16(uint32_t smem_addr, const void* gmem_ptr) {
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n"
                 :: "r"(smem_addr), "l"(gmem_ptr) : "memory");
}

static __device__ __forceinline__ void cp_async_cg_16_evict_first(
    uint32_t smem_addr, const void* gmem_ptr, uint64_t policy
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;\n"
                 :: "r"(smem_addr), "l"(gmem_ptr), "l"(policy) : "memory");
#else
    (void)policy;
    cp_async_cg_16(smem_addr, gmem_ptr);
#endif
}

static __device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template <int N>
static __device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}

static __device__ __forceinline__ void ldmatrix_x4(
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t smem_addr
) {
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_addr));
}

static __device__ __forceinline__ void ldmatrix_x4_trans(
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t smem_addr
) {
    asm volatile("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_addr));
}

static __device__ __forceinline__ void mma_m16n8k16(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)
    );
}

// =========================================================================
// Exact IEEE-754 Runtime Split-KV Weight Sum Tree (0-ULP vs V11 `<S>`!)
//   Covers ALL S in {2, 3, 4, 6, 8, 12, 16, 24, 32, 48} with ZERO template bloat.
// =========================================================================
static __device__ __forceinline__ float reduce_splits_sum_warp_rt(float my_w, int num_splits) {
    float v = my_w;
    if (num_splits <= 2) {
        v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, 1));
    } else if (num_splits <= 4) {
        v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, 2));
        v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, 1));
    } else if (num_splits <= 8) {
        v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, 2));
        v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, 1));
        v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, 4));
    } else if (num_splits <= 16) {
        v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, 2));
        v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, 1));
        v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, 8));
        v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, 4));
    } else {
        v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, 16));
        v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, 2));
        v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, 1));
        v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, 8));
        v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, 4));
    }
    return __shfl_sync(0xFFFFFFFFu, v, 0);
}

static __device__ __forceinline__ void issue_kv64_swiz_async(
    const uint16_t* __restrict__ KV_base,
    int64_t stride_b, int64_t stride_s,
    const int2* __restrict__ blk_slot,
    uint32_t kv_sptr,
    uint4* __restrict__ smem_kv4,
    bool evict_first,
    uint64_t l2_policy,
    int tid
) {
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int idx = i * 128 + tid;
        const int row = idx >> 4;
        const int c = idx & 15;
        const int swiz_c = c ^ (row & 7);
        const int2 bs = blk_slot[row];
        if (bs.x >= 0) {
            const int64_t roff = static_cast<int64_t>(bs.x) * stride_b + static_cast<int64_t>(bs.y) * stride_s;
            const uint32_t s_addr = kv_sptr + static_cast<uint32_t>(row * 256 + swiz_c * 16);
            if (evict_first) {
                cp_async_cg_16_evict_first(s_addr, KV_base + roff + c * 8, l2_policy);
            } else {
                cp_async_cg_16(s_addr, KV_base + roff + c * 8);
            }
        } else {
            smem_kv4[row * 16 + swiz_c] = make_uint4(0u, 0u, 0u, 0u);
        }
    }
}

template <bool KV_BF12>
static __device__ __forceinline__ void issue_kv64_maybe_bf12_async(
    const uint16_t* __restrict__ KV_base,
    const uint8_t*  __restrict__ KV_bf12_base,
    int64_t stride_b, int64_t stride_s,
    const int2* __restrict__ blk_slot,
    uint32_t kv_sptr,
    uint4* __restrict__ smem_kv4,
    bool evict_first,
    uint64_t l2_policy,
    int tid
) {
    if constexpr (KV_BF12) {
        (void)KV_base; (void)stride_s;
        kv_bf12_issue_tile64_async(
            KV_bf12_base, stride_b, blk_slot, kv_sptr,
            reinterpret_cast<uint8_t*>(smem_kv4), evict_first, l2_policy, tid
        );
    } else {
        (void)KV_bf12_base;
        issue_kv64_swiz_async(
            KV_base, stride_b, stride_s, blk_slot, kv_sptr, smem_kv4, evict_first, l2_policy, tid
        );
    }
}

static __device__ __forceinline__ void clip_r_row_exact_8lane(
    const uint16_t* __restrict__ r_ptr,
    const uint16_t* __restrict__ r_scale_row,
    bool valid, bool has_tau,
    float tau, float rel_threshold, float clip_eps,
    int sub_lane,
    uint16_t* dst_row16
) {
    const int d = sub_lane * 2;
    float x0 = 0.0f, x1 = 0.0f;
    if (valid) {
        uint32_t rv = *reinterpret_cast<const uint32_t*>(r_ptr + d);
        uint32_t sv = *reinterpret_cast<const uint32_t*>(r_scale_row + d);
        x0 = __fmul_rn(bf16_to_f32(static_cast<uint16_t>(rv & 0xFFFFu)), bf16_to_f32(static_cast<uint16_t>(sv & 0xFFFFu)));
        x1 = __fmul_rn(bf16_to_f32(static_cast<uint16_t>(rv >> 16)),      bf16_to_f32(static_cast<uint16_t>(sv >> 16)));
    }
    float s = __fadd_rn(__fmul_rn(x0, x0), __fmul_rn(x1, x1));
    s = __fadd_rn(s, __shfl_xor_sync(0xFFFFFFFFu, s, 4));
    s = __fadd_rn(s, __shfl_xor_sync(0xFFFFFFFFu, s, 2));
    s = __fadd_rn(s, __shfl_xor_sync(0xFFFFFFFFu, s, 1));
    if (valid) {
        float norm = (s > 0.0f) ? __fmul_rn(s, rsqrt_approx_f32(s)) : 0.0f;
        float mult = fminf(__fdiv_rn(rel_threshold, __fadd_rn(norm, clip_eps)), 1.0f);
        uint16_t o0 = f32_to_bf16(__fmul_rn(x0, mult));
        uint16_t o1 = f32_to_bf16(__fmul_rn(x1, mult));
        if (has_tau) {
            o0 = f32_to_bf16(__fmul_rn(bf16_to_f32(o0), tau));
            o1 = f32_to_bf16(__fmul_rn(bf16_to_f32(o1), tau));
        }
        *reinterpret_cast<uint32_t*>(dst_row16 + d) = pack_bf16x2(o0, o1);
    } else {
        *reinterpret_cast<uint32_t*>(dst_row16 + d) = 0u;
    }
}

// Order-A exact score + bias accumulation (`TREE_MODE == 0`)
static __device__ __forceinline__ float apply_score_bias_exact(
    float s, float scale_log2, float b
) {
    return fma_rn_f32(scale_log2, s, __fmul_rn(b, LOG2E));
}

// Exact B0 8-element contiguous chunk sum (Global `WINDOW < 0`)
static __device__ __forceinline__ float reduce_chunk8_b0(float u, float v) {
    u = __fadd_rn(u, __shfl_xor_sync(0xFFFFFFFFu, u, 1));
    v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, 1));
    u = __fadd_rn(u, __shfl_xor_sync(0xFFFFFFFFu, u, 2));
    v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, 2));
    return __fadd_rn(u, v);
}

// Exact BL1_2x2 32-lane butterfly (Local `WINDOW >= 0`)
template <int STRIDE>
static __device__ __forceinline__ float bfly32_from_64(
    const float (*Red_smem)[STRIDE], int off, int row, int tid
) {
    float e = __fadd_rn(__fadd_rn(Red_smem[off + 0 + tid][row],  Red_smem[off + 16 + tid][row]),
                        __fadd_rn(Red_smem[off + 8 + tid][row],  Red_smem[off + 24 + tid][row]));
    float o = __fadd_rn(__fadd_rn(Red_smem[off + 4 + tid][row],  Red_smem[off + 20 + tid][row]),
                        __fadd_rn(Red_smem[off + 12 + tid][row], Red_smem[off + 28 + tid][row]));
    e = __fadd_rn(e, __shfl_xor_sync(0xFFFFFFFFu, e, 2));
    o = __fadd_rn(o, __shfl_xor_sync(0xFFFFFFFFu, o, 2));
    e = __fadd_rn(e, __shfl_xor_sync(0xFFFFFFFFu, e, 1));
    o = __fadd_rn(o, __shfl_xor_sync(0xFFFFFFFFu, o, 1));
    return __fadd_rn(e, o);
}

template <int WINDOW>
static __device__ __forceinline__ void reduce_p_phase2_write_64x17(
    float (*Red_smem)[17],
    float p0_0, float p1_0, float p0_1, float p1_1,
    float p2_0, float p3_0, float p2_1, float p3_1,
    int wid, int grp, int tid
) {
    if constexpr (WINDOW < 0) {
        const float a0 = __fadd_rn(reduce_chunk8_b0(p0_0, p1_0), reduce_chunk8_b0(p0_1, p1_1));
        const float a1 = __fadd_rn(reduce_chunk8_b0(p2_0, p3_0), reduce_chunk8_b0(p2_1, p3_1));
        if (tid == 0) {
            Red_smem[wid][grp]     = a0;
            Red_smem[wid][grp + 8] = a1;
        }
    } else {
        Red_smem[wid * 8 + tid][grp]              = p0_0;
        Red_smem[wid * 8 + 4 + tid][grp]          = p1_0;
        Red_smem[32 + wid * 8 + tid][grp]         = p0_1;
        Red_smem[32 + wid * 8 + 4 + tid][grp]     = p1_1;
        Red_smem[wid * 8 + tid][grp + 8]          = p2_0;
        Red_smem[wid * 8 + 4 + tid][grp + 8]      = p3_0;
        Red_smem[32 + wid * 8 + tid][grp + 8]     = p2_1;
        Red_smem[32 + wid * 8 + 4 + tid][grp + 8] = p3_1;
    }
}

template <int WINDOW>
static __device__ __forceinline__ void reduce_p_phase2_read_64x17(
    const float (*Red_smem)[17],
    int grp, int tid,
    float& sum_p0, float& sum_p1
) {
    if constexpr (WINDOW < 0) {
        (void)tid;
        sum_p0 = __fadd_rn(__fadd_rn(Red_smem[0][grp],     Red_smem[2][grp]),
                           __fadd_rn(Red_smem[1][grp],     Red_smem[3][grp]));
        sum_p1 = __fadd_rn(__fadd_rn(Red_smem[0][grp + 8], Red_smem[2][grp + 8]),
                           __fadd_rn(Red_smem[1][grp + 8], Red_smem[3][grp + 8]));
    } else {
        sum_p0 = __fadd_rn(bfly32_from_64<17>(Red_smem, 0, grp, tid),
                           bfly32_from_64<17>(Red_smem, 32, grp, tid));
        sum_p1 = __fadd_rn(bfly32_from_64<17>(Red_smem, 0, grp + 8, tid),
                           bfly32_from_64<17>(Red_smem, 32, grp + 8, tid));
    }
}

template <int WINDOW>
static __device__ __forceinline__ void reduce_p_phase2_write_64x8(
    float (*Red_smem)[8],
    float p0_0, float p1_0, float p0_1, float p1_1,
    float p2_0, float p3_0, float p2_1, float p3_1,
    int wid, int grp, int tid
) {
    if constexpr (WINDOW < 0) {
        const float a0 = __fadd_rn(reduce_chunk8_b0(p0_0, p1_0), reduce_chunk8_b0(p0_1, p1_1));
        const float a1 = __fadd_rn(reduce_chunk8_b0(p2_0, p3_0), reduce_chunk8_b0(p2_1, p3_1));
        if (tid == 0) {
            Red_smem[wid][grp]     = a0;
            Red_smem[wid + 4][grp] = a1;
        }
    } else {
        (void)p2_0; (void)p3_0; (void)p2_1; (void)p3_1;
        Red_smem[wid * 8 + tid][grp]          = p0_0;
        Red_smem[wid * 8 + 4 + tid][grp]      = p1_0;
        Red_smem[32 + wid * 8 + tid][grp]     = p0_1;
        Red_smem[32 + wid * 8 + 4 + tid][grp] = p1_1;
    }
}

template <int WINDOW>
static __device__ __forceinline__ void reduce_p_phase2_read_64x8(
    float (*Red_smem)[8],
    float p2_0, float p3_0, float p2_1, float p3_1,
    int wid, int grp, int tid,
    float& sum_p0, float& sum_p1
) {
    if constexpr (WINDOW < 0) {
        (void)p2_0; (void)p3_0; (void)p2_1; (void)p3_1; (void)wid; (void)tid;
        sum_p0 = __fadd_rn(__fadd_rn(Red_smem[0][grp], Red_smem[2][grp]),
                           __fadd_rn(Red_smem[1][grp], Red_smem[3][grp]));
        sum_p1 = __fadd_rn(__fadd_rn(Red_smem[4][grp], Red_smem[6][grp]),
                           __fadd_rn(Red_smem[5][grp], Red_smem[7][grp]));
    } else {
        sum_p0 = __fadd_rn(bfly32_from_64<8>(Red_smem, 0, grp, tid),
                           bfly32_from_64<8>(Red_smem, 32, grp, tid));
        __syncthreads();
        Red_smem[wid * 8 + tid][grp]          = p2_0;
        Red_smem[wid * 8 + 4 + tid][grp]      = p3_0;
        Red_smem[32 + wid * 8 + tid][grp]     = p2_1;
        Red_smem[32 + wid * 8 + 4 + tid][grp] = p3_1;
        __syncthreads();
        sum_p1 = __fadd_rn(bfly32_from_64<8>(Red_smem, 0, grp, tid),
                           bfly32_from_64<8>(Red_smem, 32, grp, tid));
    }
}

template <int BIAS_COLS>
static __device__ __forceinline__ void compute_otf_bias_tile(
    const uint16_t* __restrict__ RelProjT,
    uint16_t* __restrict__ smem_b,
    uint32_t r_a0, uint32_t r_a1, uint32_t r_a2, uint32_t r_a3,
    int d_base, int d_hi, int E,
    int wid, int grp, int tid
) {
    const int n_d = min(E, d_hi + 1) - d_base;
    const int n_nt = (n_d + 7) >> 3;
    for (int nt = wid; nt < n_nt; nt += 4) {
        const int d_col = d_base + nt * 8 + grp;
        uint32_t b0 = 0u, b1 = 0u;
        if (d_col < E) {
            const uint16_t* rp = RelProjT + static_cast<int64_t>(d_col) * D_REL;
            b0 = *reinterpret_cast<const uint32_t*>(rp + 2 * tid);
            b1 = *reinterpret_cast<const uint32_t*>(rp + 8 + 2 * tid);
        }
        float bd0 = 0.f, bd1 = 0.f, bd2 = 0.f, bd3 = 0.f;
        mma_m16n8k16(bd0, bd1, bd2, bd3, r_a0, r_a1, r_a2, r_a3, b0, b1);
        const int c_off = nt * 8 + 2 * tid;
        *reinterpret_cast<uint32_t*>(smem_b + grp * BIAS_COLS + c_off) =
            pack_bf16x2(f32_to_bf16(__fadd_rn(bd0, 0.f)), f32_to_bf16(__fadd_rn(bd1, 0.f)));
        *reinterpret_cast<uint32_t*>(smem_b + (grp + 8) * BIAS_COLS + c_off) =
            pack_bf16x2(f32_to_bf16(__fadd_rn(bd2, 0.f)), f32_to_bf16(__fadd_rn(bd3, 0.f)));
    }
}

static __device__ __forceinline__ void store_direct_gate_sink_row(
    bool row_ok, int tok, int h, int NT,
    float m_i, float l_i,
    float a0, float a1, float a2, float a3,
    float a4, float a5, float a6, float a7,
    const uint16_t* __restrict__ GATE, const uint16_t* __restrict__ SINK,
    uint16_t* __restrict__ Y,
    int64_t sg_t, int64_t ssk_t, int64_t sy_t,
    int wid, int tid
) {
    if (!row_ok) return;
    const bool tv = (tok < NT);
    const float l_safe = (l_i > 0.0f) ? l_i : 1.0f;
    const float lse = tv
        ? ((l_i > 0.0f) ? __fmul_rn(__fadd_rn(m_i, log2f(l_safe)), LN2) : -INFINITY)
        : 0.0f;
    const float sink_f = bf16_to_f32(SINK[static_cast<int64_t>(tok) * ssk_t + h]);
    const float sig_s = triton_sigmoid(__fsub_rn(lse, sink_f));

    const uint16_t* gp = GATE + static_cast<int64_t>(tok) * sg_t + h * D_DIM + wid * 32;
    uint16_t* yp       = Y    + static_cast<int64_t>(tok) * sy_t + h * D_DIM + wid * 32;

    auto emit_pair = [&](int nt, float va, float vb) {
        const int c = nt * 8 + 2 * tid;
        float g0, g1;
        if ((reinterpret_cast<uintptr_t>(gp + c) & 3u) == 0u) {
            uint32_t gv = *reinterpret_cast<const uint32_t*>(gp + c);
            g0 = bf16_to_f32(static_cast<uint16_t>(gv & 0xFFFFu));
            g1 = bf16_to_f32(static_cast<uint16_t>(gv >> 16));
        } else {
            g0 = bf16_to_f32(gp[c]);
            g1 = bf16_to_f32(gp[c + 1]);
        }
        const float x0 = tv ? div_full_f32(va, l_safe) : 0.0f;
        const float x1 = tv ? div_full_f32(vb, l_safe) : 0.0f;
        const uint16_t y0 = f32_to_bf16(__fmul_rn(__fmul_rn(x0, triton_sigmoid(g0)), sig_s));
        const uint16_t y1 = f32_to_bf16(__fmul_rn(__fmul_rn(x1, triton_sigmoid(g1)), sig_s));
        *reinterpret_cast<uint32_t*>(yp + c) = pack_bf16x2(y0, y1);
    };

    emit_pair(0, a0, a1);
    emit_pair(1, a2, a3);
    emit_pair(2, a4, a5);
    emit_pair(3, a6, a7);
}

// S_MODE == 1: 2 <= num_splits <= 32; S_MODE == 2: 33 <= num_splits <= 64 (e.g. S=48)
template <int S_MODE, bool COMPACT_ROWS>
static __device__ __forceinline__ void cooperative_merge_gate_sink_v13(
    int num_splits,
    int n_rows,
    const uint8_t* Row_ok,
    const int32_t* Row_tok, const int32_t* Row_h,
    const uint8_t* Row_iloc, const uint8_t* Row_g,
    int q0_base, int h_base,
    const float* __restrict__ OP, const float* __restrict__ LP,
    const uint16_t* __restrict__ GATE, const uint16_t* __restrict__ SINK,
    uint16_t* __restrict__ Y,
    int64_t so_s, int64_t so_t, int64_t so_h,
    int64_t sl_s, int64_t sl_t, int64_t sl_h,
    int64_t sg_t, int64_t ssk_t, int64_t sy_t,
    int NT, int wid, int lane
) {
    if constexpr (COMPACT_ROWS) { (void)Row_tok; (void)Row_h; }
    else                        { (void)Row_iloc; (void)Row_g; (void)q0_base; (void)h_base; }

    const int d0 = lane * 4;
    for (int ri = wid; ri < n_rows; ri += 4) {
        if (!Row_ok[ri]) continue;
        const int tok = COMPACT_ROWS ? (q0_base + static_cast<int>(Row_iloc[ri])) : Row_tok[ri];
        const int h   = COMPACT_ROWS ? (h_base  + static_cast<int>(Row_g[ri]))    : Row_h[ri];
        const int col = h * D_DIM + d0;

        float g0, g1, g2, g3;
        {
            const uint16_t* gp = GATE + static_cast<int64_t>(tok) * sg_t + col;
            if ((reinterpret_cast<uintptr_t>(gp) & 7u) == 0u) {
                uint2 v = *reinterpret_cast<const uint2*>(gp);
                g0 = bf16_to_f32(static_cast<uint16_t>(v.x & 0xFFFFu));
                g1 = bf16_to_f32(static_cast<uint16_t>(v.x >> 16));
                g2 = bf16_to_f32(static_cast<uint16_t>(v.y & 0xFFFFu));
                g3 = bf16_to_f32(static_cast<uint16_t>(v.y >> 16));
            } else {
                g0 = bf16_to_f32(gp[0]); g1 = bf16_to_f32(gp[1]);
                g2 = bf16_to_f32(gp[2]); g3 = bf16_to_f32(gp[3]);
            }
        }
        const float sink_f = bf16_to_f32(SINK[static_cast<int64_t>(tok) * ssk_t + h]);
        const float sg0 = triton_sigmoid(g0);
        const float sg1 = triton_sigmoid(g1);
        const float sg2 = triton_sigmoid(g2);
        const float sg3 = triton_sigmoid(g3);

        const bool tv = (tok < NT);
        float x0 = 0.f, x1 = 0.f, x2 = 0.f, x3 = 0.f;
        float lse_v = 0.0f;
        if (tv) {
            const float* lp_base = LP + static_cast<int64_t>(tok) * sl_t + static_cast<int64_t>(h) * sl_h;
            const float* op_base = OP + static_cast<int64_t>(tok) * so_t + static_cast<int64_t>(h) * so_h + d0;
            if constexpr (S_MODE == 1) {
                float my_lse = -INFINITY;
                if (lane < num_splits) {
                    my_lse = ld_cg_f32(lp_base + static_cast<int64_t>(lane) * sl_s);
                }
                float mx = my_lse;
                mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, 1));
                mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, 2));
                mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, 4));
                mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, 8));
                mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, 16));
                mx = __shfl_sync(0xFFFFFFFFu, mx, 0);

                const float mx_safe = (mx == -INFINITY) ? 0.0f : mx;
                float my_w = 0.0f;
                if (lane < num_splits) {
                    const float arg = __fmul_rn(__fsub_rn(my_lse, mx_safe), LOG2E);
                    my_w = ex2_approx_f32(arg);
                }
                const float tot = reduce_splits_sum_warp_rt(my_w, num_splits);
                const float tot_safe = (tot > 0.0f) ? tot : 1.0f;
                const uint32_t valid_mask = __ballot_sync(0xFFFFFFFFu, (lane < num_splits) && (my_lse > -INFINITY));

                float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
                #pragma unroll 4
                for (int s = 0; s < num_splits; ++s) {
                    const float ws = __shfl_sync(0xFFFFFFFFu, my_w, s);
                    if ((valid_mask >> s) & 1u) {
                        float4 ov = ld_cg_f4(op_base + static_cast<int64_t>(s) * so_s);
                        acc0 = fma_rn_f32(ws, ov.x, acc0);
                        acc1 = fma_rn_f32(ws, ov.y, acc1);
                        acc2 = fma_rn_f32(ws, ov.z, acc2);
                        acc3 = fma_rn_f32(ws, ov.w, acc3);
                    }
                }
                x0 = div_full_f32(acc0, tot_safe);
                x1 = div_full_f32(acc1, tot_safe);
                x2 = div_full_f32(acc2, tot_safe);
                x3 = div_full_f32(acc3, tot_safe);
                lse_v = (tot > 0.0f) ? __fadd_rn(mx_safe, logf(tot_safe)) : -INFINITY;
            } else {
                // Dual-slot 32-lane warp merge for 32 < num_splits <= 64 (S = 48)
                const int hi_cnt = num_splits - 32;
                const float my_lse0 = ld_cg_f32(lp_base + static_cast<int64_t>(lane) * sl_s);
                float my_lse1 = -INFINITY;
                if (lane < hi_cnt) {
                    my_lse1 = ld_cg_f32(lp_base + static_cast<int64_t>(lane + 32) * sl_s);
                }
                float mx = fmaxf(my_lse0, my_lse1);
                mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, 1));
                mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, 2));
                mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, 4));
                mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, 8));
                mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, 16));
                mx = __shfl_sync(0xFFFFFFFFu, mx, 0);

                const float mx_safe = (mx == -INFINITY) ? 0.0f : mx;
                const float my_w0 = ex2_approx_f32(__fmul_rn(__fsub_rn(my_lse0, mx_safe), LOG2E));
                float my_w1 = 0.0f;
                if (lane < hi_cnt) {
                    my_w1 = ex2_approx_f32(__fmul_rn(__fsub_rn(my_lse1, mx_safe), LOG2E));
                }
                const float tot = reduce_splits_sum_warp_rt(__fadd_rn(my_w0, my_w1), num_splits);
                const float tot_safe = (tot > 0.0f) ? tot : 1.0f;
                const uint32_t valid_mask_lo = __ballot_sync(0xFFFFFFFFu, my_lse0 > -INFINITY);
                const uint32_t valid_mask_hi = __ballot_sync(0xFFFFFFFFu, (lane < hi_cnt) && (my_lse1 > -INFINITY));

                float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
                #pragma unroll 4
                for (int s = 0; s < 32; ++s) {
                    const float ws = __shfl_sync(0xFFFFFFFFu, my_w0, s);
                    if ((valid_mask_lo >> s) & 1u) {
                        float4 ov = ld_cg_f4(op_base + static_cast<int64_t>(s) * so_s);
                        acc0 = fma_rn_f32(ws, ov.x, acc0);
                        acc1 = fma_rn_f32(ws, ov.y, acc1);
                        acc2 = fma_rn_f32(ws, ov.z, acc2);
                        acc3 = fma_rn_f32(ws, ov.w, acc3);
                    }
                }
                #pragma unroll 4
                for (int sh = 0; sh < hi_cnt; ++sh) {
                    const float ws = __shfl_sync(0xFFFFFFFFu, my_w1, sh);
                    if ((valid_mask_hi >> sh) & 1u) {
                        float4 ov = ld_cg_f4(op_base + static_cast<int64_t>(sh + 32) * so_s);
                        acc0 = fma_rn_f32(ws, ov.x, acc0);
                        acc1 = fma_rn_f32(ws, ov.y, acc1);
                        acc2 = fma_rn_f32(ws, ov.z, acc2);
                        acc3 = fma_rn_f32(ws, ov.w, acc3);
                    }
                }
                x0 = div_full_f32(acc0, tot_safe);
                x1 = div_full_f32(acc1, tot_safe);
                x2 = div_full_f32(acc2, tot_safe);
                x3 = div_full_f32(acc3, tot_safe);
                lse_v = (tot > 0.0f) ? __fadd_rn(mx_safe, logf(tot_safe)) : -INFINITY;
            }
        }
        const float sig_s = triton_sigmoid(__fsub_rn(lse_v, sink_f));
        const uint16_t b0 = f32_to_bf16(__fmul_rn(__fmul_rn(x0, sg0), sig_s));
        const uint16_t b1 = f32_to_bf16(__fmul_rn(__fmul_rn(x1, sg1), sig_s));
        const uint16_t b2 = f32_to_bf16(__fmul_rn(__fmul_rn(x2, sg2), sig_s));
        const uint16_t b3 = f32_to_bf16(__fmul_rn(__fmul_rn(x3, sg3), sig_s));
        uint2 out_v;
        out_v.x = pack_bf16x2(b0, b1);
        out_v.y = pack_bf16x2(b2, b3);
        *reinterpret_cast<uint2*>(Y + static_cast<int64_t>(tok) * sy_t + col) = out_v;
    }
}

// =========================================================================
// KERNEL 2A (FUSE_MT = 0): S_MODE in {0: S==1, 1: 2<=S<=32, 2: 33<=S<=64}
//   Runtime `guard_mode` (0=none, 1=short-ctx Node 1, 2=long-ctx Node 2).
// =========================================================================
template <int WINDOW, int S_MODE, bool USE_PDL, bool HAS_TREE_2B, bool KV_BF12>
__global__ void __launch_bounds__(128, 2)
fused_decode_attn_otf_mgs_kernel_v13(
    const uint16_t* __restrict__ Q,
    const uint16_t* __restrict__ R,
    const uint16_t* __restrict__ RScale,
    const float*    __restrict__ Tau,
    const uint16_t* __restrict__ RelProjT,
    const uint16_t* __restrict__ K,
    const uint16_t* __restrict__ V,
    const int32_t*  __restrict__ BT,
    const int32_t*  __restrict__ SEQ,
    const int32_t*  __restrict__ CUQ,
    const uint16_t* __restrict__ GATE,
    const uint16_t* __restrict__ SINK,
    float*          __restrict__ OP,
    float*          __restrict__ LP,
    int32_t*        __restrict__ SEM,
    uint16_t*       __restrict__ Y,
    const int32_t*  __restrict__ NLivePtr,
    const int32_t*  __restrict__ TreeMeta,
    int64_t sq_t, int64_t sq_h,
    int64_t sr_t, int64_t sr_h,
    int64_t sk_b, int64_t sk_s, int64_t sk_h,
    int64_t sv_b, int64_t sv_s, int64_t sv_h,
    int64_t sbt_r,
    int64_t sg_t, int64_t ssk_t, int64_t sy_t,
    int64_t so_s, int64_t so_t, int64_t so_h,
    int64_t sl_s, int64_t sl_t, int64_t sl_h,
    float scale_log2, float rel_threshold, float clip_eps,
    int T, int NT, int BS, int BS_shift, int E, int G,
    int num_splits, int has_tau_i, int has_nlive_i, int trig_wo_i,
    int guard_mode, int kv_tok_thresh,
    const uint16_t* __restrict__ KEmergRaw,
    const uint16_t* __restrict__ VEmergRaw
) {
    (void)T;
    if constexpr (!KV_BF12) { (void)KEmergRaw; (void)VEmergRaw; }
    if constexpr (S_MODE == 0) { (void)OP; (void)LP; (void)SEM; (void)so_s; (void)so_t; (void)so_h; (void)sl_s; (void)sl_t; (void)sl_h; }
    if (guard_mode != 0) {
        int64_t tot_kv = 0;
        const int nr = static_cast<int>(gridDim.y);
        #pragma unroll 4
        for (int rr = 0; rr < nr; ++rr) {
            const int sl = SEQ[rr];
            if (sl > 0) tot_kv += static_cast<int64_t>(sl);
        }
        const bool skip_guard = (guard_mode == 1)
            ? (tot_kv > static_cast<int64_t>(kv_tok_thresh))
            : (tot_kv <= static_cast<int64_t>(kv_tok_thresh));
        if (skip_guard) {
            if constexpr (USE_PDL) {
                gdc_wait();
                __syncthreads();
                if (threadIdx.x == 0) gdc_launch_dependents();
            }
            return;
        }
    }
    const bool has_tau   = (has_tau_i != 0);
    const bool has_nlive = (has_nlive_i != 0);
    const bool trig_wo   = (trig_wo_i != 0);
    if constexpr (!USE_PDL) { (void)trig_wo; }

    const int mt = blockIdx.x;
    const int r  = blockIdx.y;
    const int kvh   = (S_MODE == 0) ? static_cast<int>(blockIdx.z) : (static_cast<int>(blockIdx.z) / num_splits);
    const int split = (S_MODE == 0) ? 0 : (static_cast<int>(blockIdx.z) % num_splits);
    const int sem_idx = (S_MODE == 0) ? 0 : ((r * static_cast<int>(gridDim.x) + mt) * (static_cast<int>(gridDim.z) / num_splits) + kvh);

    const int tid_all = threadIdx.x;
    const int wid  = tid_all >> 5;
    const int lane = tid_all & 31;
    const int grp  = lane >> 2;
    const int tid  = lane & 3;
    const int sub  = lane >> 3;
    const int r_in = lane & 7;

    const int q0 = CUQ[r];
    const int qlen = CUQ[r + 1] - q0;
    const int seqlen = SEQ[r];
    const int past = seqlen - qlen;
    int r_live = qlen;
    if (has_nlive) {
        int nl = *NLivePtr;
        int d = nl - q0;
        if (d < 0) d = 0;
        if (d < r_live) r_live = d;
    }

    int tree_wp = r_live - 1;
    int tree_ws = 0;
    bool is_sec_tile = false;
    if constexpr (HAS_TREE_2B) {
        const int t_act = TreeMeta[0];
        const int t_wp  = TreeMeta[1];
        const int t_ws  = TreeMeta[2];
        const int t_nl  = TreeMeta[3];
        if (t_act != 0 && t_ws > 0) {
            tree_wp = min(max(t_wp, 0), qlen - 1);
            tree_ws = min(t_ws, min(2, qlen - 1 - tree_wp));
            r_live  = min(max(t_nl, 1), qlen);
            is_sec_tile = (mt == 3);
        }
        const bool skip_tile = is_sec_tile ? (tree_ws <= 0) : (mt >= 3 || (mt * BM) / G > tree_wp);
        if (skip_tile) {
            if constexpr (USE_PDL) {
                gdc_wait();
                __syncthreads();
                if (trig_wo && tid_all == 0) gdc_launch_dependents();
            }
            return;
        }
    } else {
        (void)TreeMeta; (void)tree_wp; (void)tree_ws; (void)is_sec_tile;
        if ((mt * BM) / G >= r_live) {
            if constexpr (USE_PDL) {
                gdc_wait();
                __syncthreads();
                if (trig_wo && tid_all == 0) gdc_launch_dependents();
            }
            return;
        }
    }

    const int k_end = max(past + qlen, 0);
    const int k_beg = (WINDOW >= 0) ? max(past - WINDOW, 0) : 0;
    const int span  = max(k_end - k_beg, 0);
    const int per   = (S_MODE == 0) ? (((span + 63) / 64) * 64) : ((((span + num_splits - 1) / num_splits) + 63) / 64 * 64);
    const int s_beg = k_beg + split * per;
    const int s_end = min(s_beg + per, k_end);
    int s_end_live  = (seqlen > 0 && q0 < NT) ? s_end : s_beg;
    if constexpr (HAS_TREE_2B) {
        const int max_k = is_sec_tile ? (past + 1 + tree_ws) : (past + 1 + tree_wp);
        if (max_k < s_end_live) s_end_live = max_k;
    } else if (has_nlive) {
        if (past + r_live < s_end_live) s_end_live = past + r_live;
    }

    extern __shared__ uint4 smem_raw_u4[];
    uint8_t*  smem_raw  = reinterpret_cast<uint8_t*>(smem_raw_u4);
    uint16_t* smem_kv0  = reinterpret_cast<uint16_t*>(smem_raw);
    uint16_t* smem_kv1  = reinterpret_cast<uint16_t*>(smem_raw + 16384);
    uint8_t*  smem_work = smem_raw + 32768;
    uint16_t* smem_b    = reinterpret_cast<uint16_t*>(smem_work);
    float (*Red_smem)[17] = reinterpret_cast<float (*)[17]>(smem_work);
    uint4*    smem_q_4  = reinterpret_cast<uint4*>(smem_work);
    uint16_t* smem_p    = reinterpret_cast<uint16_t*>(smem_raw + 37120);
    uint16_t* smem_r    = reinterpret_cast<uint16_t*>(smem_raw + 39424);

    alignas(16) __shared__ float    Max_smem[4][17];
    alignas(16) __shared__ int32_t  Row_tok[16];
    alignas(16) __shared__ int32_t  Row_h[16];
    alignas(16) __shared__ int32_t  Row_qpos[16];
    alignas(16) __shared__ uint8_t  Row_ok[16];
    alignas(16) __shared__ int2     KV_blk_slot[64];

    if (tid_all < 16) {
        if constexpr (HAS_TREE_2B) {
            if (is_sec_tile) {
                int sec_off = tid_all / G;
                int g = tid_all % G;
                bool ok = (sec_off < tree_ws);
                Row_ok[tid_all]   = ok ? 1 : 0;
                Row_tok[tid_all]  = ok ? (q0 + 1 + tree_wp + sec_off) : (q0 + 1 + tree_wp);
                Row_h[tid_all]    = kvh * G + g;
                Row_qpos[tid_all] = past + 1 + (ok ? sec_off : (tree_ws - 1));
            } else {
                int m = mt * BM + tid_all;
                int i_loc = m / G;
                int g = m % G;
                bool ok = (i_loc <= tree_wp);
                Row_ok[tid_all]   = ok ? 1 : 0;
                Row_tok[tid_all]  = q0 + i_loc;
                Row_h[tid_all]    = kvh * G + g;
                Row_qpos[tid_all] = past + i_loc;
            }
        } else {
            int m = mt * BM + tid_all;
            int i_loc = m / G;
            int g = m % G;
            bool ok = (i_loc < r_live);
            Row_ok[tid_all]   = ok ? 1 : 0;
            Row_tok[tid_all]  = q0 + i_loc;
            Row_h[tid_all]    = kvh * G + g;
            Row_qpos[tid_all] = past + i_loc;
        }
    }

    const int32_t* bt_row = BT + static_cast<int64_t>(r) * sbt_r;
    const uint16_t* K_base = K + static_cast<int64_t>(kvh) * sk_h;
    const uint16_t* V_base = V + static_cast<int64_t>(kvh) * sv_h;
    const uint8_t* K_bf12_base = reinterpret_cast<const uint8_t*>(K) + static_cast<int64_t>(kvh) * sk_h;
    const uint8_t* V_bf12_base = reinterpret_cast<const uint8_t*>(V) + static_cast<int64_t>(kvh) * sv_h;
    const uint32_t kv0_sptr  = static_cast<uint32_t>(__cvta_generic_to_shared(smem_kv0));
    const uint32_t kv1_sptr  = static_cast<uint32_t>(__cvta_generic_to_shared(smem_kv1));
    const uint32_t work_sptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_work));
    uint4* smem_kv0_4 = reinterpret_cast<uint4*>(smem_kv0);
    uint4* smem_kv1_4 = reinterpret_cast<uint4*>(smem_kv1);

    const bool tile0_is_past = (!USE_PDL) && (s_beg < s_end_live) && ((s_beg + BN) <= past);
    if (tile0_is_past) {
        if (tid_all < 64) {
            int kpos = s_beg + tid_all;
            int lblk = (BS_shift >= 0) ? (kpos >> BS_shift) : (kpos / BS);
            int slot = (BS_shift >= 0) ? (kpos & ((1 << BS_shift) - 1)) : (kpos % BS);
            KV_blk_slot[tid_all] = make_int2(bt_row[lblk], slot);
        }
        __syncthreads();
        issue_kv64_maybe_bf12_async<KV_BF12>(K_base, K_bf12_base, sk_b, sk_s, KV_blk_slot, kv0_sptr, smem_kv0_4, false, 0ULL, tid_all);
        cp_async_commit();
        issue_kv64_maybe_bf12_async<KV_BF12>(V_base, V_bf12_base, sv_b, sv_s, KV_blk_slot, kv1_sptr, smem_kv1_4, false, 0ULL, tid_all);
        cp_async_commit();
    } else {
        __syncthreads();
    }

    if constexpr (USE_PDL) { gdc_wait(); }

    const bool ok0 = Row_ok[grp] != 0;
    const bool ok1 = Row_ok[grp + 8] != 0;
    float m_i0 = -INFINITY, m_i1 = -INFINITY;
    float l_i0 = 0.0f, l_i1 = 0.0f;
    float acc[4][4] = {{0.f}};

    if (s_beg >= s_end_live) {
        if constexpr (S_MODE > 0) {
            if (tid_all < 16 && Row_ok[tid_all]) {
                LP[split * sl_s + Row_tok[tid_all] * sl_t + Row_h[tid_all] * sl_h] = -INFINITY;
            }
        }
    } else {
        #pragma unroll
        for (int pass = 0; pass < 2; ++pass) {
            int idx = pass * 128 + tid_all;
            int row = idx >> 4;
            int c = idx & 15;
            int swiz_c = c ^ (row & 7);
            uint4 qv = make_uint4(0u, 0u, 0u, 0u);
            if (Row_ok[row]) {
                qv = *reinterpret_cast<const uint4*>(
                    Q + static_cast<int64_t>(Row_tok[row]) * sq_t
                      + static_cast<int64_t>(Row_h[row]) * sq_h + c * 8
                );
            }
            smem_q_4[row * 16 + swiz_c] = qv;
        }
        {
            const int sub_grp = lane >> 3;
            const int sub_lane = lane & 7;
            const int row = wid * 4 + sub_grp;
            const bool r_ok = (Row_ok[row] != 0);
            const int tok = r_ok ? Row_tok[row] : 0;
            const int h = r_ok ? Row_h[row] : 0;
            const float tau = (has_tau && r_ok) ? Tau[tok] : 1.0f;
            clip_r_row_exact_8lane(
                R + static_cast<int64_t>(tok) * sr_t + static_cast<int64_t>(h) * sr_h,
                RScale + h * D_REL,
                r_ok, has_tau, tau, rel_threshold, clip_eps, sub_lane, smem_r + row * D_REL
            );
        }
        if (!tile0_is_past && tid_all < 64) {
            int kpos = s_beg + tid_all;
            if (kpos < s_end_live) {
                int phys_kpos = kpos;
                if constexpr (HAS_TREE_2B) {
                    if (is_sec_tile && kpos > past) phys_kpos = kpos + tree_wp;
                }
                int lblk = (BS_shift >= 0) ? (phys_kpos >> BS_shift) : (phys_kpos / BS);
                int slot = (BS_shift >= 0) ? (phys_kpos & ((1 << BS_shift) - 1)) : (phys_kpos % BS);
                KV_blk_slot[tid_all] = make_int2(bt_row[lblk], slot);
            } else {
                KV_blk_slot[tid_all] = make_int2(-1, 0);
            }
        }
        __syncthreads();

        if (!tile0_is_past) {
            issue_kv64_maybe_bf12_async<KV_BF12>(K_base, K_bf12_base, sk_b, sk_s, KV_blk_slot, kv0_sptr, smem_kv0_4, false, 0ULL, tid_all);
            cp_async_commit();
            issue_kv64_maybe_bf12_async<KV_BF12>(V_base, V_bf12_base, sv_b, sv_s, KV_blk_slot, kv1_sptr, smem_kv1_4, false, 0ULL, tid_all);
            cp_async_commit();
        }

        uint32_t q_frag[8][4];
        #pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
            const int m_row = lane & 15;
            const int k_col8 = ks * 2 + (lane >> 4);
            const int swiz_c = k_col8 ^ (m_row & 7);
            const uint32_t s_addr = work_sptr + static_cast<uint32_t>(m_row * 256 + swiz_c * 16);
            ldmatrix_x4(q_frag[ks][0], q_frag[ks][1], q_frag[ks][2], q_frag[ks][3], s_addr);
        }

        const uint32_t r_a0 = *reinterpret_cast<const uint32_t*>(smem_r + grp * D_REL + 2 * tid);
        const uint32_t r_a1 = *reinterpret_cast<const uint32_t*>(smem_r + (grp + 8) * D_REL + 2 * tid);
        const uint32_t r_a2 = *reinterpret_cast<const uint32_t*>(smem_r + grp * D_REL + 8 + 2 * tid);
        const uint32_t r_a3 = *reinterpret_cast<const uint32_t*>(smem_r + (grp + 8) * D_REL + 8 + 2 * tid);

        const int qpos0 = Row_qpos[grp];
        const int qpos1 = Row_qpos[grp + 8];
        const int qpos_min = Row_qpos[0];
        const int qpos_max = Row_qpos[15];
        __syncthreads();

        for (int n0 = s_beg; n0 < s_end_live; n0 += BN) {
            const int n_next = n0 + BN;
            const int d_lo = qpos_min - (n0 + BN - 1);
            const int d_hi = qpos_max - n0;
            const bool tile_has_bias = (E > 0) && (d_hi >= 0) && (d_lo < E);
            const int d_base = max(0, d_lo);
            if (tile_has_bias) {
                compute_otf_bias_tile<B_COLS>(RelProjT, smem_b, r_a0, r_a1, r_a2, r_a3, d_base, d_hi, E, wid, grp, tid);
            }

            cp_async_wait_group<1>();
            __syncthreads();
            if constexpr (KV_BF12) {
                kv_bf12_unpack_tile64_inplace_swiz(smem_kv0, K_bf12_base, sk_b, KEmergRaw, tid_all);
            }

            float s0[2] = {0.f, 0.f}, s1[2] = {0.f, 0.f}, s2[2] = {0.f, 0.f}, s3[2] = {0.f, 0.f};
            const int k_row = (wid + (sub >> 1) * 4) * 8 + r_in;
            #pragma unroll
            for (int ks = 0; ks < 8; ++ks) {
                const int k_col8 = ks * 2 + (sub & 1);
                const int swiz_c = k_col8 ^ r_in;
                const uint32_t s_addr = kv0_sptr + static_cast<uint32_t>(k_row * 256 + swiz_c * 16);
                uint32_t kb0_0, kb1_0, kb0_1, kb1_1;
                ldmatrix_x4(kb0_0, kb1_0, kb0_1, kb1_1, s_addr);
                mma_m16n8k16(s0[0], s1[0], s2[0], s3[0], q_frag[ks][0], q_frag[ks][1], q_frag[ks][2], q_frag[ks][3], kb0_0, kb1_0);
                mma_m16n8k16(s0[1], s1[1], s2[1], s3[1], q_frag[ks][0], q_frag[ks][1], q_frag[ks][2], q_frag[ks][3], kb0_1, kb1_1);
            }

            float m_w0 = -INFINITY, m_w1 = -INFINITY;
            #pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                const int kpos_a = n0 + nt * 32 + wid * 8 + 2 * tid;
                const int kpos_b = kpos_a + 1;
                const bool ka_ok = (kpos_a < s_end_live);
                const bool kb_ok = (kpos_b < s_end_live);

                const int da0 = qpos0 - kpos_a;
                const int db0 = qpos0 - kpos_b;
                const bool va0 = ok0 && ka_ok && (da0 >= 0) && (WINDOW < 0 || da0 <= WINDOW);
                const bool vb0 = ok0 && kb_ok && (db0 >= 0) && (WINDOW < 0 || db0 <= WINDOW);
                if (tile_has_bias) {
                    const float ba0 = (va0 && da0 < E) ? bf16_to_f32(smem_b[grp * B_COLS + (da0 - d_base)]) : 0.f;
                    const float bb0 = (vb0 && db0 < E) ? bf16_to_f32(smem_b[grp * B_COLS + (db0 - d_base)]) : 0.f;
                    s0[nt] = apply_score_bias_exact(s0[nt], scale_log2, ba0);
                    s1[nt] = apply_score_bias_exact(s1[nt], scale_log2, bb0);
                } else {
                    s0[nt] = __fmul_rn(s0[nt], scale_log2);
                    s1[nt] = __fmul_rn(s1[nt], scale_log2);
                }
                s0[nt] = va0 ? s0[nt] : -INFINITY;
                s1[nt] = vb0 ? s1[nt] : -INFINITY;
                m_w0 = fmaxf(m_w0, fmaxf(s0[nt], s1[nt]));

                const int da1 = qpos1 - kpos_a;
                const int db1 = qpos1 - kpos_b;
                const bool va1 = ok1 && ka_ok && (da1 >= 0) && (WINDOW < 0 || da1 <= WINDOW);
                const bool vb1 = ok1 && kb_ok && (db1 >= 0) && (WINDOW < 0 || db1 <= WINDOW);
                if (tile_has_bias) {
                    const float ba1 = (va1 && da1 < E) ? bf16_to_f32(smem_b[(grp + 8) * B_COLS + (da1 - d_base)]) : 0.f;
                    const float bb1 = (vb1 && db1 < E) ? bf16_to_f32(smem_b[(grp + 8) * B_COLS + (db1 - d_base)]) : 0.f;
                    s2[nt] = apply_score_bias_exact(s2[nt], scale_log2, ba1);
                    s3[nt] = apply_score_bias_exact(s3[nt], scale_log2, bb1);
                } else {
                    s2[nt] = __fmul_rn(s2[nt], scale_log2);
                    s3[nt] = __fmul_rn(s3[nt], scale_log2);
                }
                s2[nt] = va1 ? s2[nt] : -INFINITY;
                s3[nt] = vb1 ? s3[nt] : -INFINITY;
                m_w1 = fmaxf(m_w1, fmaxf(s2[nt], s3[nt]));
            }

            m_w0 = fmaxf(m_w0, __shfl_xor_sync(0xFFFFFFFFu, m_w0, 2));
            m_w0 = fmaxf(m_w0, __shfl_xor_sync(0xFFFFFFFFu, m_w0, 1));
            m_w1 = fmaxf(m_w1, __shfl_xor_sync(0xFFFFFFFFu, m_w1, 2));
            m_w1 = fmaxf(m_w1, __shfl_xor_sync(0xFFFFFFFFu, m_w1, 1));
            if (tid == 0) {
                Max_smem[wid][grp]     = m_w0;
                Max_smem[wid][grp + 8] = m_w1;
            }

            if (n_next < s_end_live && tid_all < 64) {
                int kpos = n_next + tid_all;
                if (kpos < s_end_live) {
                    int phys_kpos = kpos;
                    if constexpr (HAS_TREE_2B) {
                        if (is_sec_tile && kpos > past) phys_kpos = kpos + tree_wp;
                    }
                    int lblk = (BS_shift >= 0) ? (phys_kpos >> BS_shift) : (phys_kpos / BS);
                    int slot = (BS_shift >= 0) ? (phys_kpos & ((1 << BS_shift) - 1)) : (phys_kpos % BS);
                    KV_blk_slot[tid_all] = make_int2(bt_row[lblk], slot);
                } else {
                    KV_blk_slot[tid_all] = make_int2(-1, 0);
                }
            }
            __syncthreads();

            if (n_next < s_end_live) {
                issue_kv64_maybe_bf12_async<KV_BF12>(K_base, K_bf12_base, sk_b, sk_s, KV_blk_slot, kv0_sptr, smem_kv0_4, false, 0ULL, tid_all);
                cp_async_commit();
            }

            const float m_blk0 = fmaxf(fmaxf(Max_smem[0][grp],     Max_smem[1][grp]),
                                       fmaxf(Max_smem[2][grp],     Max_smem[3][grp]));
            const float m_blk1 = fmaxf(fmaxf(Max_smem[0][grp + 8], Max_smem[1][grp + 8]),
                                       fmaxf(Max_smem[2][grp + 8], Max_smem[3][grp + 8]));
            const float m_new0  = fmaxf(m_i0, m_blk0);
            const float m_new1  = fmaxf(m_i1, m_blk1);
            const float m_safe0 = (m_new0 == -INFINITY) ? 0.0f : m_new0;
            const float m_safe1 = (m_new1 == -INFINITY) ? 0.0f : m_new1;
            const float alpha0  = ex2_approx_ftz_f32(__fsub_rn(m_i0, m_safe0));
            const float alpha1  = ex2_approx_ftz_f32(__fsub_rn(m_i1, m_safe1));

            const float p0_0 = ex2_approx_ftz_f32(__fsub_rn(s0[0], m_safe0));
            const float p1_0 = ex2_approx_ftz_f32(__fsub_rn(s1[0], m_safe0));
            const float p0_1 = ex2_approx_ftz_f32(__fsub_rn(s0[1], m_safe0));
            const float p1_1 = ex2_approx_ftz_f32(__fsub_rn(s1[1], m_safe0));
            const float p2_0 = ex2_approx_ftz_f32(__fsub_rn(s2[0], m_safe1));
            const float p3_0 = ex2_approx_ftz_f32(__fsub_rn(s3[0], m_safe1));
            const float p2_1 = ex2_approx_ftz_f32(__fsub_rn(s2[1], m_safe1));
            const float p3_1 = ex2_approx_ftz_f32(__fsub_rn(s3[1], m_safe1));

            *reinterpret_cast<uint32_t*>(smem_p + grp * P_COLS + wid * 8 + 2 * tid)            = pack_bf16x2(f32_to_bf16(p0_0), f32_to_bf16(p1_0));
            *reinterpret_cast<uint32_t*>(smem_p + grp * P_COLS + 32 + wid * 8 + 2 * tid)       = pack_bf16x2(f32_to_bf16(p0_1), f32_to_bf16(p1_1));
            *reinterpret_cast<uint32_t*>(smem_p + (grp + 8) * P_COLS + wid * 8 + 2 * tid)      = pack_bf16x2(f32_to_bf16(p2_0), f32_to_bf16(p3_0));
            *reinterpret_cast<uint32_t*>(smem_p + (grp + 8) * P_COLS + 32 + wid * 8 + 2 * tid) = pack_bf16x2(f32_to_bf16(p2_1), f32_to_bf16(p3_1));

            reduce_p_phase2_write_64x17<WINDOW>(Red_smem, p0_0, p1_0, p0_1, p1_1, p2_0, p3_0, p2_1, p3_1, wid, grp, tid);

            if (n_next < s_end_live) cp_async_wait_group<1>();
            else                     cp_async_wait_group<0>();
            __syncthreads();
            if constexpr (KV_BF12) {
                kv_bf12_unpack_tile64_inplace_swiz(smem_kv1, V_bf12_base, sv_b, VEmergRaw, tid_all);
            }

            float sum_p0, sum_p1;
            reduce_p_phase2_read_64x17<WINDOW>(Red_smem, grp, tid, sum_p0, sum_p1);
            l_i0 = fma_rn_f32(l_i0, alpha0, sum_p0);
            l_i1 = fma_rn_f32(l_i1, alpha1, sum_p1);
            m_i0 = m_new0;
            m_i1 = m_new1;

            #pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                acc[nt][0] = __fmul_rn(acc[nt][0], alpha0);
                acc[nt][1] = __fmul_rn(acc[nt][1], alpha0);
                acc[nt][2] = __fmul_rn(acc[nt][2], alpha1);
                acc[nt][3] = __fmul_rn(acc[nt][3], alpha1);
            }

            #pragma unroll
            for (int ks = 0; ks < 4; ++ks) {
                const int k0 = ks * 16 + 2 * tid;
                const int k1 = k0 + 8;
                const uint32_t a0 = *reinterpret_cast<const uint32_t*>(smem_p + grp * P_COLS + k0);
                const uint32_t a1 = *reinterpret_cast<const uint32_t*>(smem_p + (grp + 8) * P_COLS + k0);
                const uint32_t a2 = *reinterpret_cast<const uint32_t*>(smem_p + grp * P_COLS + k1);
                const uint32_t a3 = *reinterpret_cast<const uint32_t*>(smem_p + (grp + 8) * P_COLS + k1);
                const int v_row = ks * 16 + (sub & 1) * 8 + r_in;
                #pragma unroll
                for (int j = 0; j < 2; ++j) {
                    const int nt0 = 2 * j;
                    const int d_col8 = wid * 4 + nt0 + (sub >> 1);
                    const int swiz_c = d_col8 ^ r_in;
                    const uint32_t s_addr = kv1_sptr + static_cast<uint32_t>(v_row * 256 + swiz_c * 16);
                    uint32_t vb0_0, vb1_0, vb0_1, vb1_1;
                    ldmatrix_x4_trans(vb0_0, vb1_0, vb0_1, vb1_1, s_addr);
                    mma_m16n8k16(acc[nt0][0],     acc[nt0][1],     acc[nt0][2],     acc[nt0][3],     a0, a1, a2, a3, vb0_0, vb1_0);
                    mma_m16n8k16(acc[nt0 + 1][0], acc[nt0 + 1][1], acc[nt0 + 1][2], acc[nt0 + 1][3], a0, a1, a2, a3, vb0_1, vb1_1);
                }
            }
            if (n_next < s_end_live) {
                __syncthreads();
                issue_kv64_maybe_bf12_async<KV_BF12>(V_base, V_bf12_base, sv_b, sv_s, KV_blk_slot, kv1_sptr, smem_kv1_4, false, 0ULL, tid_all);
                cp_async_commit();
            }
        }
    }

    if constexpr (S_MODE == 0) {
        if constexpr (USE_PDL) {
            __syncthreads();
            if (trig_wo && tid_all == 0) gdc_launch_dependents();
        }
        store_direct_gate_sink_row(
            ok0, Row_tok[grp], Row_h[grp], NT, m_i0, l_i0,
            acc[0][0], acc[0][1], acc[1][0], acc[1][1], acc[2][0], acc[2][1], acc[3][0], acc[3][1],
            GATE, SINK, Y, sg_t, ssk_t, sy_t, wid, tid
        );
        store_direct_gate_sink_row(
            ok1, Row_tok[grp + 8], Row_h[grp + 8], NT, m_i1, l_i1,
            acc[0][2], acc[0][3], acc[1][2], acc[1][3], acc[2][2], acc[2][3], acc[3][2], acc[3][3],
            GATE, SINK, Y, sg_t, ssk_t, sy_t, wid, tid
        );
    } else {
        if (s_beg < s_end_live) {
            const float l_safe0 = (l_i0 > 0.0f) ? l_i0 : 1.0f;
            const float l_safe1 = (l_i1 > 0.0f) ? l_i1 : 1.0f;
            if (ok0) {
                float* o0 = OP + split * so_s + Row_tok[grp] * so_t + Row_h[grp] * so_h + wid * 32;
                #pragma unroll
                for (int nt = 0; nt < 4; ++nt) {
                    o0[nt * 8 + 2 * tid]     = div_full_f32(acc[nt][0], l_safe0);
                    o0[nt * 8 + 2 * tid + 1] = div_full_f32(acc[nt][1], l_safe0);
                }
                if (wid == 0 && tid == 0) {
                    LP[split * sl_s + Row_tok[grp] * sl_t + Row_h[grp] * sl_h] =
                        (l_i0 > 0.0f) ? __fmul_rn(__fadd_rn(m_i0, log2f(l_safe0)), LN2) : -INFINITY;
                }
            }
            if (ok1) {
                float* o1 = OP + split * so_s + Row_tok[grp + 8] * so_t + Row_h[grp + 8] * so_h + wid * 32;
                #pragma unroll
                for (int nt = 0; nt < 4; ++nt) {
                    o1[nt * 8 + 2 * tid]     = div_full_f32(acc[nt][2], l_safe1);
                    o1[nt * 8 + 2 * tid + 1] = div_full_f32(acc[nt][3], l_safe1);
                }
                if (wid == 0 && tid == 0) {
                    LP[split * sl_s + Row_tok[grp + 8] * sl_t + Row_h[grp + 8] * sl_h] =
                        (l_i1 > 0.0f) ? __fmul_rn(__fadd_rn(m_i1, log2f(l_safe1)), LN2) : -INFINITY;
                }
            }
        }
        __threadfence();
        __syncthreads();
        alignas(16) __shared__ int s_done;
        if (tid_all == 0) s_done = atomic_add_acq_rel_gpu(&SEM[sem_idx], 1);
        __syncthreads();
        if constexpr (USE_PDL) {
            if (trig_wo && tid_all == 0) gdc_launch_dependents();
        }
        if (s_done == num_splits - 1) {
            if (tid_all == 0) SEM[sem_idx] = 0;
            cooperative_merge_gate_sink_v13<S_MODE, false>(
                num_splits, BM, Row_ok, Row_tok, Row_h, nullptr, nullptr, 0, 0,
                OP, LP, GATE, SINK, Y,
                so_s, so_t, so_h, sl_s, sl_t, sl_h, sg_t, ssk_t, sy_t, NT, wid, lane
            );
        }
    }
}

// =========================================================================
// KERNEL 2B (FUSE_MT = 1) HELPERS — ZERO-DUPLICATION FACTORED PIPELINES!
// =========================================================================

// Local SWA (or MT_MAX==1) step helper — runtime `is_last_mt` removes 3x branch duplication!
template <int WINDOW, int MT_MAX, bool IS_FIRST_MT, bool KV_BF12>
static __device__ __forceinline__ void fusedmt_step_tile_v13(
    int mt, bool is_last_mt, int past,
    int n0, int n_next, int s_end_live, int E, int BS, int BS_shift,
    float scale_log2,
    const uint16_t* __restrict__ RelProjT,
    const uint16_t* __restrict__ K_base,
    const uint16_t* __restrict__ V_base,
    const uint8_t*  __restrict__ K_bf12_base,
    const uint8_t*  __restrict__ V_bf12_base,
    const uint16_t* __restrict__ VEmergRaw,
    int64_t sk_b, int64_t sk_s, int64_t sv_b, int64_t sv_s,
    const int32_t* __restrict__ bt_row,
    uint32_t q_sptr_mt, uint32_t kv0_sptr, uint32_t kv1_sptr,
    uint4* __restrict__ smem_kv0_4,
    uint4* __restrict__ smem_kv1_4,
    uint16_t* __restrict__ smem_bp,
    void* red_smem_raw,
    float (*Max_smem)[8],
    int2 KV_blk_slot[64],
    const uint8_t* Row_iloc,
    const uint8_t* Row_ok,
    uint32_t r_a0, uint32_t r_a1, uint32_t r_a2, uint32_t r_a3,
    float& m_i0, float& m_i1, float& l_i0, float& l_i1,
    float (&acc)[4][4],
    bool evict_first, uint64_t l2_policy,
    int tid_all, int wid, int lane, int grp, int tid, int sub, int r_in
) {
    if constexpr (!KV_BF12) { (void)K_bf12_base; (void)V_bf12_base; (void)VEmergRaw; }
    const int base = mt * BM;
    const bool ok0 = (Row_ok[base + grp] != 0);
    const bool ok1 = (Row_ok[base + grp + 8] != 0);
    const int qpos0 = past + static_cast<int>(Row_iloc[base + grp]);
    const int qpos1 = past + static_cast<int>(Row_iloc[base + grp + 8]);
    const int qpos_min = past + static_cast<int>(Row_iloc[base]);
    const int qpos_max = past + static_cast<int>(Row_iloc[base + 15]);

    const int d_lo = qpos_min - (n0 + BN - 1);
    const int d_hi = qpos_max - n0;
    const bool tile_has_bias = (E > 0) && (d_hi >= 0) && (d_lo < E);
    const int d_base = max(0, d_lo);

    if constexpr (!IS_FIRST_MT) {
        if (tile_has_bias) {
            compute_otf_bias_tile<B_COLS_FMT>(RelProjT, smem_bp, r_a0, r_a1, r_a2, r_a3, d_base, d_hi, E, wid, grp, tid);
            __syncthreads();
        }
    } else {
        (void)RelProjT; (void)r_a0; (void)r_a1; (void)r_a2; (void)r_a3;
    }

    float s0[2] = {0.f, 0.f}, s1[2] = {0.f, 0.f}, s2[2] = {0.f, 0.f}, s3[2] = {0.f, 0.f};
    const int k_row = (wid + (sub >> 1) * 4) * 8 + r_in;
    const int q_row = lane & 15;
    #pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
        const int q_col8 = ks * 2 + (lane >> 4);
        const int q_swiz = q_col8 ^ (q_row & 7);
        const uint32_t q_addr = q_sptr_mt + static_cast<uint32_t>(q_row * 256 + q_swiz * 16);
        uint32_t qa0, qa1, qa2, qa3;
        ldmatrix_x4(qa0, qa1, qa2, qa3, q_addr);

        const int k_col8 = ks * 2 + (sub & 1);
        const int k_swiz = k_col8 ^ r_in;
        const uint32_t k_addr = kv0_sptr + static_cast<uint32_t>(k_row * 256 + k_swiz * 16);
        uint32_t kb0_0, kb1_0, kb0_1, kb1_1;
        ldmatrix_x4(kb0_0, kb1_0, kb0_1, kb1_1, k_addr);

        mma_m16n8k16(s0[0], s1[0], s2[0], s3[0], qa0, qa1, qa2, qa3, kb0_0, kb1_0);
        mma_m16n8k16(s0[1], s1[1], s2[1], s3[1], qa0, qa1, qa2, qa3, kb0_1, kb1_1);
    }

    float m_w0 = -INFINITY, m_w1 = -INFINITY;
    #pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
        const int kpos_a = n0 + nt * 32 + wid * 8 + 2 * tid;
        const int kpos_b = kpos_a + 1;
        const bool ka_ok = (kpos_a < s_end_live);
        const bool kb_ok = (kpos_b < s_end_live);

        const int da0 = qpos0 - kpos_a;
        const int db0 = qpos0 - kpos_b;
        const bool va0 = ok0 && ka_ok && (da0 >= 0) && (WINDOW < 0 || da0 <= WINDOW);
        const bool vb0 = ok0 && kb_ok && (db0 >= 0) && (WINDOW < 0 || db0 <= WINDOW);
        if (tile_has_bias) {
            const float ba0 = (va0 && da0 < E) ? bf16_to_f32(smem_bp[grp * B_COLS_FMT + (da0 - d_base)]) : 0.f;
            const float bb0 = (vb0 && db0 < E) ? bf16_to_f32(smem_bp[grp * B_COLS_FMT + (db0 - d_base)]) : 0.f;
            s0[nt] = apply_score_bias_exact(s0[nt], scale_log2, ba0);
            s1[nt] = apply_score_bias_exact(s1[nt], scale_log2, bb0);
        } else {
            s0[nt] = __fmul_rn(s0[nt], scale_log2);
            s1[nt] = __fmul_rn(s1[nt], scale_log2);
        }
        s0[nt] = va0 ? s0[nt] : -INFINITY;
        s1[nt] = vb0 ? s1[nt] : -INFINITY;
        m_w0 = fmaxf(m_w0, fmaxf(s0[nt], s1[nt]));

        const int da1 = qpos1 - kpos_a;
        const int db1 = qpos1 - kpos_b;
        const bool va1 = ok1 && ka_ok && (da1 >= 0) && (WINDOW < 0 || da1 <= WINDOW);
        const bool vb1 = ok1 && kb_ok && (db1 >= 0) && (WINDOW < 0 || db1 <= WINDOW);
        if (tile_has_bias) {
            const float ba1 = (va1 && da1 < E) ? bf16_to_f32(smem_bp[(grp + 8) * B_COLS_FMT + (da1 - d_base)]) : 0.f;
            const float bb1 = (vb1 && db1 < E) ? bf16_to_f32(smem_bp[(grp + 8) * B_COLS_FMT + (db1 - d_base)]) : 0.f;
            s2[nt] = apply_score_bias_exact(s2[nt], scale_log2, ba1);
            s3[nt] = apply_score_bias_exact(s3[nt], scale_log2, bb1);
        } else {
            s2[nt] = __fmul_rn(s2[nt], scale_log2);
            s3[nt] = __fmul_rn(s3[nt], scale_log2);
        }
        s2[nt] = va1 ? s2[nt] : -INFINITY;
        s3[nt] = vb1 ? s3[nt] : -INFINITY;
        m_w1 = fmaxf(m_w1, fmaxf(s2[nt], s3[nt]));
    }

    m_w0 = fmaxf(m_w0, __shfl_xor_sync(0xFFFFFFFFu, m_w0, 2));
    m_w0 = fmaxf(m_w0, __shfl_xor_sync(0xFFFFFFFFu, m_w0, 1));
    m_w1 = fmaxf(m_w1, __shfl_xor_sync(0xFFFFFFFFu, m_w1, 2));
    m_w1 = fmaxf(m_w1, __shfl_xor_sync(0xFFFFFFFFu, m_w1, 1));
    if (tid == 0) {
        Max_smem[wid][grp]     = m_w0;
        Max_smem[wid + 4][grp] = m_w1;
    }

    if (is_last_mt && n_next < s_end_live && tid_all < 64) {
        int kpos = n_next + tid_all;
        if (kpos < s_end_live) {
            int lblk = (BS_shift >= 0) ? (kpos >> BS_shift) : (kpos / BS);
            int slot = (BS_shift >= 0) ? (kpos & ((1 << BS_shift) - 1)) : (kpos % BS);
            KV_blk_slot[tid_all] = make_int2(bt_row[lblk], slot);
        } else {
            KV_blk_slot[tid_all] = make_int2(-1, 0);
        }
    }
    __syncthreads();

    if (is_last_mt && n_next < s_end_live) {
        issue_kv64_maybe_bf12_async<KV_BF12>(K_base, K_bf12_base, sk_b, sk_s, KV_blk_slot, kv0_sptr, smem_kv0_4, evict_first, l2_policy, tid_all);
        cp_async_commit();
    }

    const float m_blk0  = fmaxf(fmaxf(Max_smem[0][grp], Max_smem[1][grp]), fmaxf(Max_smem[2][grp], Max_smem[3][grp]));
    const float m_blk1  = fmaxf(fmaxf(Max_smem[4][grp], Max_smem[5][grp]), fmaxf(Max_smem[6][grp], Max_smem[7][grp]));
    const float m_new0  = fmaxf(m_i0, m_blk0);
    const float m_new1  = fmaxf(m_i1, m_blk1);
    const float m_safe0 = (m_new0 == -INFINITY) ? 0.0f : m_new0;
    const float m_safe1 = (m_new1 == -INFINITY) ? 0.0f : m_new1;
    const float alpha0  = ex2_approx_ftz_f32(__fsub_rn(m_i0, m_safe0));
    const float alpha1  = ex2_approx_ftz_f32(__fsub_rn(m_i1, m_safe1));

    const float p0_0 = ex2_approx_ftz_f32(__fsub_rn(s0[0], m_safe0));
    const float p1_0 = ex2_approx_ftz_f32(__fsub_rn(s1[0], m_safe0));
    const float p0_1 = ex2_approx_ftz_f32(__fsub_rn(s0[1], m_safe0));
    const float p1_1 = ex2_approx_ftz_f32(__fsub_rn(s1[1], m_safe0));
    const float p2_0 = ex2_approx_ftz_f32(__fsub_rn(s2[0], m_safe1));
    const float p3_0 = ex2_approx_ftz_f32(__fsub_rn(s3[0], m_safe1));
    const float p2_1 = ex2_approx_ftz_f32(__fsub_rn(s2[1], m_safe1));
    const float p3_1 = ex2_approx_ftz_f32(__fsub_rn(s3[1], m_safe1));

    *reinterpret_cast<uint32_t*>(smem_bp + grp * P_COLS + wid * 8 + 2 * tid)            = pack_bf16x2(f32_to_bf16(p0_0), f32_to_bf16(p1_0));
    *reinterpret_cast<uint32_t*>(smem_bp + grp * P_COLS + 32 + wid * 8 + 2 * tid)       = pack_bf16x2(f32_to_bf16(p0_1), f32_to_bf16(p1_1));
    *reinterpret_cast<uint32_t*>(smem_bp + (grp + 8) * P_COLS + wid * 8 + 2 * tid)      = pack_bf16x2(f32_to_bf16(p2_0), f32_to_bf16(p3_0));
    *reinterpret_cast<uint32_t*>(smem_bp + (grp + 8) * P_COLS + 32 + wid * 8 + 2 * tid) = pack_bf16x2(f32_to_bf16(p2_1), f32_to_bf16(p3_1));

    if constexpr (MT_MAX <= 2) {
        auto Red17 = reinterpret_cast<float (*)[17]>(red_smem_raw);
        reduce_p_phase2_write_64x17<WINDOW>(Red17, p0_0, p1_0, p0_1, p1_1, p2_0, p3_0, p2_1, p3_1, wid, grp, tid);
    } else {
        auto Red8 = reinterpret_cast<float (*)[8]>(red_smem_raw);
        reduce_p_phase2_write_64x8<WINDOW>(Red8, p0_0, p1_0, p0_1, p1_1, p2_0, p3_0, p2_1, p3_1, wid, grp, tid);
    }

    if constexpr (IS_FIRST_MT) {
        if (is_last_mt && n_next < s_end_live) cp_async_wait_group<1>();
        else                                   cp_async_wait_group<0>();
    }
    __syncthreads();
    if constexpr (KV_BF12 && IS_FIRST_MT) {
        kv_bf12_unpack_tile64_inplace_swiz(
            reinterpret_cast<uint16_t*>(smem_kv1_4), V_bf12_base, sv_b, VEmergRaw, tid_all
        );
    }

    float sum_p0, sum_p1;
    if constexpr (MT_MAX <= 2) {
        auto Red17 = reinterpret_cast<const float (*)[17]>(red_smem_raw);
        reduce_p_phase2_read_64x17<WINDOW>(Red17, grp, tid, sum_p0, sum_p1);
    } else {
        auto Red8 = reinterpret_cast<float (*)[8]>(red_smem_raw);
        reduce_p_phase2_read_64x8<WINDOW>(Red8, p2_0, p3_0, p2_1, p3_1, wid, grp, tid, sum_p0, sum_p1);
    }
    l_i0 = fma_rn_f32(l_i0, alpha0, sum_p0);
    l_i1 = fma_rn_f32(l_i1, alpha1, sum_p1);
    m_i0 = m_new0;
    m_i1 = m_new1;

    #pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
        acc[nt][0] = __fmul_rn(acc[nt][0], alpha0);
        acc[nt][1] = __fmul_rn(acc[nt][1], alpha0);
        acc[nt][2] = __fmul_rn(acc[nt][2], alpha1);
        acc[nt][3] = __fmul_rn(acc[nt][3], alpha1);
    }

    #pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
        const int k0 = ks * 16 + 2 * tid;
        const int k1 = k0 + 8;
        const uint32_t a0 = *reinterpret_cast<const uint32_t*>(smem_bp + grp * P_COLS + k0);
        const uint32_t a1 = *reinterpret_cast<const uint32_t*>(smem_bp + (grp + 8) * P_COLS + k0);
        const uint32_t a2 = *reinterpret_cast<const uint32_t*>(smem_bp + grp * P_COLS + k1);
        const uint32_t a3 = *reinterpret_cast<const uint32_t*>(smem_bp + (grp + 8) * P_COLS + k1);
        const int v_row = ks * 16 + (sub & 1) * 8 + r_in;
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int nt0 = 2 * j;
            const int d_col8 = wid * 4 + nt0 + (sub >> 1);
            const int swiz_c = d_col8 ^ r_in;
            const uint32_t s_addr = kv1_sptr + static_cast<uint32_t>(v_row * 256 + swiz_c * 16);
            uint32_t vb0_0, vb1_0, vb0_1, vb1_1;
            ldmatrix_x4_trans(vb0_0, vb1_0, vb0_1, vb1_1, s_addr);
            mma_m16n8k16(acc[nt0][0],     acc[nt0][1],     acc[nt0][2],     acc[nt0][3],     a0, a1, a2, a3, vb0_0, vb1_0);
            mma_m16n8k16(acc[nt0 + 1][0], acc[nt0 + 1][1], acc[nt0 + 1][2], acc[nt0 + 1][3], a0, a1, a2, a3, vb0_1, vb1_1);
        }
    }

    if (!is_last_mt) {
        __syncthreads();
    } else if (n_next < s_end_live) {
        __syncthreads();
        issue_kv64_maybe_bf12_async<KV_BF12>(V_base, V_bf12_base, sv_b, sv_s, KV_blk_slot, kv1_sptr, smem_kv1_4, evict_first, l2_policy, tid_all);
        cp_async_commit();
    }
}

// =========================================================================
// Factored Global Split-Phase Dual-Inflight Helpers (`WINDOW < 0, MT_MAX >= 2`)
//   ZERO duplication of `issue_K`, `issue_V`, or `unpack(V)`!
//   Every `acc[0/1/2]` uses compile-time constant indexing -> 0 B stack / 0 B spills!
// =========================================================================
template <int MT_MAX, int MT_IDX>
static __device__ __forceinline__ float (*fusedmt_global_max_slice(
    void* red_smem_raw, float (*Max_smem)[8]
))[8] {
    if constexpr (MT_IDX == 0) {
        (void)red_smem_raw;
        return Max_smem;
    } else if constexpr (MT_MAX <= 2) {
        (void)Max_smem;
        auto Red17 = reinterpret_cast<float (*)[17]>(red_smem_raw);
        return reinterpret_cast<float (*)[8]>(&Red17[32][0]);
    } else {
        (void)Max_smem;
        auto Red8 = reinterpret_cast<float (*)[8]>(red_smem_raw);
        return &Red8[24 + MT_IDX * 8];
    }
}

template <int MT_MAX, int MT_IDX>
static __device__ __forceinline__ void fusedmt_global_red_write(
    void* red_smem_raw, float a0, float a1, int wid, int grp, int tid
) {
    if (tid == 0) {
        if constexpr (MT_MAX <= 2) {
            auto Red17 = reinterpret_cast<float (*)[17]>(red_smem_raw);
            Red17[MT_IDX * 4 + wid][grp]     = a0;
            Red17[MT_IDX * 4 + wid][grp + 8] = a1;
        } else {
            auto Red8 = reinterpret_cast<float (*)[8]>(red_smem_raw);
            Red8[MT_IDX * 8 + wid][grp]     = a0;
            Red8[MT_IDX * 8 + wid + 4][grp] = a1;
        }
    }
}

template <int MT_MAX, int MT_IDX>
static __device__ __forceinline__ void fusedmt_global_red_read(
    const void* red_smem_raw, int grp, float& sum_p0, float& sum_p1
) {
    if constexpr (MT_MAX <= 2) {
        auto Red17 = reinterpret_cast<const float (*)[17]>(red_smem_raw);
        constexpr int r = MT_IDX * 4;
        sum_p0 = __fadd_rn(__fadd_rn(Red17[r + 0][grp],     Red17[r + 2][grp]),
                           __fadd_rn(Red17[r + 1][grp],     Red17[r + 3][grp]));
        sum_p1 = __fadd_rn(__fadd_rn(Red17[r + 0][grp + 8], Red17[r + 2][grp + 8]),
                           __fadd_rn(Red17[r + 1][grp + 8], Red17[r + 3][grp + 8]));
    } else {
        auto Red8 = reinterpret_cast<const float (*)[8]>(red_smem_raw);
        constexpr int r = MT_IDX * 8;
        sum_p0 = __fadd_rn(__fadd_rn(Red8[r + 0][grp], Red8[r + 2][grp]),
                           __fadd_rn(Red8[r + 1][grp], Red8[r + 3][grp]));
        sum_p1 = __fadd_rn(__fadd_rn(Red8[r + 4][grp], Red8[r + 6][grp]),
                           __fadd_rn(Red8[r + 5][grp], Red8[r + 7][grp]));
    }
}

static __device__ __forceinline__ void fusedmt_store_p_pack(
    uint16_t* __restrict__ smem_bp,
    uint32_t pk0, uint32_t pk1, uint32_t pk2, uint32_t pk3,
    int wid, int grp, int tid
) {
    *reinterpret_cast<uint32_t*>(smem_bp + grp * P_COLS + wid * 8 + 2 * tid)            = pk0;
    *reinterpret_cast<uint32_t*>(smem_bp + grp * P_COLS + 32 + wid * 8 + 2 * tid)       = pk1;
    *reinterpret_cast<uint32_t*>(smem_bp + (grp + 8) * P_COLS + wid * 8 + 2 * tid)      = pk2;
    *reinterpret_cast<uint32_t*>(smem_bp + (grp + 8) * P_COLS + 32 + wid * 8 + 2 * tid) = pk3;
}

static __device__ __forceinline__ void fusedmt_global_v_mma(
    const uint16_t* __restrict__ smem_bp,
    uint32_t kv1_sptr,
    float (&acc)[4][4],
    int wid, int grp, int tid, int sub, int r_in
) {
    #pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
        const int k0 = ks * 16 + 2 * tid;
        const int k1 = k0 + 8;
        const uint32_t a0 = *reinterpret_cast<const uint32_t*>(smem_bp + grp * P_COLS + k0);
        const uint32_t a1 = *reinterpret_cast<const uint32_t*>(smem_bp + (grp + 8) * P_COLS + k0);
        const uint32_t a2 = *reinterpret_cast<const uint32_t*>(smem_bp + grp * P_COLS + k1);
        const uint32_t a3 = *reinterpret_cast<const uint32_t*>(smem_bp + (grp + 8) * P_COLS + k1);
        const int v_row = ks * 16 + (sub & 1) * 8 + r_in;
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int nt0 = 2 * j;
            const int d_col8 = wid * 4 + nt0 + (sub >> 1);
            const int swiz_c = d_col8 ^ r_in;
            const uint32_t s_addr = kv1_sptr + static_cast<uint32_t>(v_row * 256 + swiz_c * 16);
            uint32_t vb0_0, vb1_0, vb0_1, vb1_1;
            ldmatrix_x4_trans(vb0_0, vb1_0, vb0_1, vb1_1, s_addr);
            mma_m16n8k16(acc[nt0][0],     acc[nt0][1],     acc[nt0][2],     acc[nt0][3],     a0, a1, a2, a3, vb0_0, vb1_0);
            mma_m16n8k16(acc[nt0 + 1][0], acc[nt0 + 1][1], acc[nt0 + 1][2], acc[nt0 + 1][3], a0, a1, a2, a3, vb0_1, vb1_1);
        }
    }
}

// Pure K-compute phase (NO `issue_K`, NO `wait_group`, NO `unpack(V)` -> 0 code duplication!)
template <int MT_MAX, int MT_IDX>
static __device__ __forceinline__ void fusedmt_global_k_compute_v13(
    bool is_last_mt, int past,
    int n0, int n_next, int s_end_live, int E, int BS, int BS_shift,
    float scale_log2,
    const uint16_t* __restrict__ RelProjT,
    const int32_t* __restrict__ bt_row,
    uint32_t q_sptr_mt, uint32_t kv0_sptr,
    uint16_t* __restrict__ smem_bp,
    void* red_smem_raw,
    float (*Max_smem)[8],
    int2 KV_blk_slot[64],
    const uint8_t* Row_iloc,
    const uint8_t* Row_ok,
    uint32_t r_a0, uint32_t r_a1, uint32_t r_a2, uint32_t r_a3,
    float& m_i0, float& m_i1,
    float (&acc)[4][4],
    float& prev_l0, float& prev_l1, float prev_a0, float prev_a1,
    float& out_a0, float& out_a1,
    uint32_t& pk0_out, uint32_t& pk1_out, uint32_t& pk2_out, uint32_t& pk3_out,
    int tid_all, int wid, int lane, int grp, int tid, int sub, int r_in
) {
    constexpr bool IS_FIRST_MT = (MT_IDX == 0);
    constexpr int base = MT_IDX * BM;
    const bool ok0 = (Row_ok[base + grp] != 0);
    const bool ok1 = (Row_ok[base + grp + 8] != 0);
    const int qpos0 = past + static_cast<int>(Row_iloc[base + grp]);
    const int qpos1 = past + static_cast<int>(Row_iloc[base + grp + 8]);
    const int qpos_min = past + static_cast<int>(Row_iloc[base]);
    const int qpos_max = past + static_cast<int>(Row_iloc[base + 15]);

    const int d_lo = qpos_min - (n0 + BN - 1);
    const int d_hi = qpos_max - n0;
    const bool tile_has_bias = (E > 0) && (d_hi >= 0) && (d_lo < E);
    const int d_base = max(0, d_lo);

    if constexpr (!IS_FIRST_MT) {
        if (tile_has_bias) {
            compute_otf_bias_tile<B_COLS_FMT>(RelProjT, smem_bp, r_a0, r_a1, r_a2, r_a3, d_base, d_hi, E, wid, grp, tid);
            __syncthreads();
        }
    } else {
        (void)RelProjT; (void)r_a0; (void)r_a1; (void)r_a2; (void)r_a3;
        (void)prev_l0; (void)prev_l1; (void)prev_a0; (void)prev_a1;
    }

    float s0[2] = {0.f, 0.f}, s1[2] = {0.f, 0.f}, s2[2] = {0.f, 0.f}, s3[2] = {0.f, 0.f};
    const int k_row = (wid + (sub >> 1) * 4) * 8 + r_in;
    const int q_row = lane & 15;
    #pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
        const int q_col8 = ks * 2 + (lane >> 4);
        const int q_swiz = q_col8 ^ (q_row & 7);
        const uint32_t q_addr = q_sptr_mt + static_cast<uint32_t>(q_row * 256 + q_swiz * 16);
        uint32_t qa0, qa1, qa2, qa3;
        ldmatrix_x4(qa0, qa1, qa2, qa3, q_addr);

        const int k_col8 = ks * 2 + (sub & 1);
        const int k_swiz = k_col8 ^ r_in;
        const uint32_t k_addr = kv0_sptr + static_cast<uint32_t>(k_row * 256 + k_swiz * 16);
        uint32_t kb0_0, kb1_0, kb0_1, kb1_1;
        ldmatrix_x4(kb0_0, kb1_0, kb0_1, kb1_1, k_addr);

        mma_m16n8k16(s0[0], s1[0], s2[0], s3[0], qa0, qa1, qa2, qa3, kb0_0, kb1_0);
        mma_m16n8k16(s0[1], s1[1], s2[1], s3[1], qa0, qa1, qa2, qa3, kb0_1, kb1_1);
    }

    float m_w0 = -INFINITY, m_w1 = -INFINITY;
    #pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
        const int kpos_a = n0 + nt * 32 + wid * 8 + 2 * tid;
        const int kpos_b = kpos_a + 1;
        const bool ka_ok = (kpos_a < s_end_live);
        const bool kb_ok = (kpos_b < s_end_live);

        const int da0 = qpos0 - kpos_a;
        const int db0 = qpos0 - kpos_b;
        const bool va0 = ok0 && ka_ok && (da0 >= 0);
        const bool vb0 = ok0 && kb_ok && (db0 >= 0);
        if (tile_has_bias) {
            const float ba0 = (va0 && da0 < E) ? bf16_to_f32(smem_bp[grp * B_COLS_FMT + (da0 - d_base)]) : 0.f;
            const float bb0 = (vb0 && db0 < E) ? bf16_to_f32(smem_bp[grp * B_COLS_FMT + (db0 - d_base)]) : 0.f;
            s0[nt] = apply_score_bias_exact(s0[nt], scale_log2, ba0);
            s1[nt] = apply_score_bias_exact(s1[nt], scale_log2, bb0);
        } else {
            s0[nt] = __fmul_rn(s0[nt], scale_log2);
            s1[nt] = __fmul_rn(s1[nt], scale_log2);
        }
        s0[nt] = va0 ? s0[nt] : -INFINITY;
        s1[nt] = vb0 ? s1[nt] : -INFINITY;
        m_w0 = fmaxf(m_w0, fmaxf(s0[nt], s1[nt]));

        const int da1 = qpos1 - kpos_a;
        const int db1 = qpos1 - kpos_b;
        const bool va1 = ok1 && ka_ok && (da1 >= 0);
        const bool vb1 = ok1 && kb_ok && (db1 >= 0);
        if (tile_has_bias) {
            const float ba1 = (va1 && da1 < E) ? bf16_to_f32(smem_bp[(grp + 8) * B_COLS_FMT + (da1 - d_base)]) : 0.f;
            const float bb1 = (vb1 && db1 < E) ? bf16_to_f32(smem_bp[(grp + 8) * B_COLS_FMT + (db1 - d_base)]) : 0.f;
            s2[nt] = apply_score_bias_exact(s2[nt], scale_log2, ba1);
            s3[nt] = apply_score_bias_exact(s3[nt], scale_log2, bb1);
        } else {
            s2[nt] = __fmul_rn(s2[nt], scale_log2);
            s3[nt] = __fmul_rn(s3[nt], scale_log2);
        }
        s2[nt] = va1 ? s2[nt] : -INFINITY;
        s3[nt] = vb1 ? s3[nt] : -INFINITY;
        m_w1 = fmaxf(m_w1, fmaxf(s2[nt], s3[nt]));
    }

    m_w0 = fmaxf(m_w0, __shfl_xor_sync(0xFFFFFFFFu, m_w0, 2));
    m_w0 = fmaxf(m_w0, __shfl_xor_sync(0xFFFFFFFFu, m_w0, 1));
    m_w1 = fmaxf(m_w1, __shfl_xor_sync(0xFFFFFFFFu, m_w1, 2));
    m_w1 = fmaxf(m_w1, __shfl_xor_sync(0xFFFFFFFFu, m_w1, 1));
    float (*Max_sl)[8] = fusedmt_global_max_slice<MT_MAX, MT_IDX>(red_smem_raw, Max_smem);
    if (tid == 0) {
        Max_sl[wid][grp]     = m_w0;
        Max_sl[wid + 4][grp] = m_w1;
    }

    if (is_last_mt && n_next < s_end_live && tid_all < 64) {
        int kpos = n_next + tid_all;
        if (kpos < s_end_live) {
            int lblk = (BS_shift >= 0) ? (kpos >> BS_shift) : (kpos / BS);
            int slot = (BS_shift >= 0) ? (kpos & ((1 << BS_shift) - 1)) : (kpos % BS);
            KV_blk_slot[tid_all] = make_int2(bt_row[lblk], slot);
        } else {
            KV_blk_slot[tid_all] = make_int2(-1, 0);
        }
    }
    __syncthreads();  // Max_sl ready; smem_bp dead; if !FIRST: prev Red ready; if LAST: smem_kv0 dead & KV_blk_slot ready!

    if constexpr (!IS_FIRST_MT) {
        float p_sp0, p_sp1;
        fusedmt_global_red_read<MT_MAX, MT_IDX - 1>(red_smem_raw, grp, p_sp0, p_sp1);
        prev_l0 = fma_rn_f32(prev_l0, prev_a0, p_sp0);
        prev_l1 = fma_rn_f32(prev_l1, prev_a1, p_sp1);
    }

    const float m_blk0  = fmaxf(fmaxf(Max_sl[0][grp], Max_sl[1][grp]), fmaxf(Max_sl[2][grp], Max_sl[3][grp]));
    const float m_blk1  = fmaxf(fmaxf(Max_sl[4][grp], Max_sl[5][grp]), fmaxf(Max_sl[6][grp], Max_sl[7][grp]));
    const float m_new0  = fmaxf(m_i0, m_blk0);
    const float m_new1  = fmaxf(m_i1, m_blk1);
    const float m_safe0 = (m_new0 == -INFINITY) ? 0.0f : m_new0;
    const float m_safe1 = (m_new1 == -INFINITY) ? 0.0f : m_new1;
    const float alpha0  = ex2_approx_ftz_f32(__fsub_rn(m_i0, m_safe0));
    const float alpha1  = ex2_approx_ftz_f32(__fsub_rn(m_i1, m_safe1));

    const float p0_0 = ex2_approx_ftz_f32(__fsub_rn(s0[0], m_safe0));
    const float p1_0 = ex2_approx_ftz_f32(__fsub_rn(s1[0], m_safe0));
    const float p0_1 = ex2_approx_ftz_f32(__fsub_rn(s0[1], m_safe0));
    const float p1_1 = ex2_approx_ftz_f32(__fsub_rn(s1[1], m_safe0));
    const float p2_0 = ex2_approx_ftz_f32(__fsub_rn(s2[0], m_safe1));
    const float p3_0 = ex2_approx_ftz_f32(__fsub_rn(s3[0], m_safe1));
    const float p2_1 = ex2_approx_ftz_f32(__fsub_rn(s2[1], m_safe1));
    const float p3_1 = ex2_approx_ftz_f32(__fsub_rn(s3[1], m_safe1));

    pk0_out = pack_bf16x2(f32_to_bf16(p0_0), f32_to_bf16(p1_0));
    pk1_out = pack_bf16x2(f32_to_bf16(p0_1), f32_to_bf16(p1_1));
    pk2_out = pack_bf16x2(f32_to_bf16(p2_0), f32_to_bf16(p3_0));
    pk3_out = pack_bf16x2(f32_to_bf16(p2_1), f32_to_bf16(p3_1));

    const float ra0 = __fadd_rn(reduce_chunk8_b0(p0_0, p1_0), reduce_chunk8_b0(p0_1, p1_1));
    const float ra1 = __fadd_rn(reduce_chunk8_b0(p2_0, p3_0), reduce_chunk8_b0(p2_1, p3_1));
    fusedmt_global_red_write<MT_MAX, MT_IDX>(red_smem_raw, ra0, ra1, wid, grp, tid);

    m_i0 = m_new0;
    m_i1 = m_new1;
    out_a0 = alpha0;
    out_a1 = alpha1;
    #pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
        acc[nt][0] = __fmul_rn(acc[nt][0], alpha0);
        acc[nt][1] = __fmul_rn(acc[nt][1], alpha0);
        acc[nt][2] = __fmul_rn(acc[nt][2], alpha1);
        acc[nt][3] = __fmul_rn(acc[nt][3], alpha1);
    }
}

// =========================================================================
// KERNEL 2B (FUSE_MT = 1): 1 CTA per (req, split) covering MT_MAX in {1, 2, 3}
//   - Factored zero-duplication split-phase pipeline (Global) & step-tile (Local)
//   - SMEM <= 50,336 B <= 50,688 B (2 CTAs/SM across ALL MT_MAX in {1, 2, 3}!)
//   - Supports full S=48 via S_MODE=2!
// =========================================================================
template <int WINDOW, int S_MODE, int MT_MAX, bool USE_PDL, bool KV_BF12>
__global__ void __launch_bounds__(128, 2)
fused_decode_attn_otf_fusedmt_kernel_v13(
    const uint16_t* __restrict__ Q,
    const uint16_t* __restrict__ R,
    const uint16_t* __restrict__ RScale,
    const float*    __restrict__ Tau,
    const uint16_t* __restrict__ RelProjT,
    const uint16_t* __restrict__ K,
    const uint16_t* __restrict__ V,
    const int32_t*  __restrict__ BT,
    const int32_t*  __restrict__ SEQ,
    const int32_t*  __restrict__ CUQ,
    const uint16_t* __restrict__ GATE,
    const uint16_t* __restrict__ SINK,
    float*          __restrict__ OP,
    float*          __restrict__ LP,
    int32_t*        __restrict__ SEM,
    uint16_t*       __restrict__ Y,
    const int32_t*  __restrict__ NLivePtr,
    int64_t sq_t, int64_t sq_h,
    int64_t sr_t, int64_t sr_h,
    int64_t sk_b, int64_t sk_s, int64_t sk_h,
    int64_t sv_b, int64_t sv_s, int64_t sv_h,
    int64_t sbt_r,
    int64_t sg_t, int64_t ssk_t, int64_t sy_t,
    int64_t so_s, int64_t so_t, int64_t so_h,
    int64_t sl_s, int64_t sl_t, int64_t sl_h,
    float scale_log2, float rel_threshold, float clip_eps,
    int T, int NT, int BS, int BS_shift, int E, int G,
    int num_splits, int has_tau_i, int has_nlive_i, int evict_first_i, int trig_wo_i,
    int guard_mode, int kv_tok_thresh,
    const uint16_t* __restrict__ KEmergRaw,
    const uint16_t* __restrict__ VEmergRaw
) {
    (void)T;
    if constexpr (!KV_BF12) { (void)KEmergRaw; (void)VEmergRaw; }
    if constexpr (S_MODE == 0) { (void)OP; (void)LP; (void)SEM; (void)so_s; (void)so_t; (void)so_h; (void)sl_s; (void)sl_t; (void)sl_h; }
    if (guard_mode != 0) {
        int64_t tot_kv = 0;
        const int nr = static_cast<int>(gridDim.y);
        #pragma unroll 4
        for (int rr = 0; rr < nr; ++rr) {
            const int sl = SEQ[rr];
            if (sl > 0) tot_kv += static_cast<int64_t>(sl);
        }
        const bool skip_guard = (guard_mode == 1)
            ? (tot_kv > static_cast<int64_t>(kv_tok_thresh))
            : (tot_kv <= static_cast<int64_t>(kv_tok_thresh));
        if (skip_guard) {
            if constexpr (USE_PDL) {
                gdc_wait();
                __syncthreads();
                if (trig_wo_i != 0 && threadIdx.x == 0) gdc_launch_dependents();
            }
            return;
        }
    }
    const bool has_tau     = (has_tau_i != 0);
    const bool has_nlive   = (has_nlive_i != 0);
    const bool evict_first = (evict_first_i != 0);
    const bool trig_wo     = (trig_wo_i != 0);
    if constexpr (!USE_PDL) { (void)trig_wo; }

    constexpr int MAX_ROWS = MT_MAX * BM;
    const int r = blockIdx.y;
    const int kvh   = (S_MODE == 0) ? static_cast<int>(blockIdx.z) : (static_cast<int>(blockIdx.z) / num_splits);
    const int split = (S_MODE == 0) ? 0 : (static_cast<int>(blockIdx.z) % num_splits);
    const int sem_idx = (S_MODE == 0) ? 0 : (r * (static_cast<int>(gridDim.z) / num_splits) + kvh);
    const int h_base = kvh * G;

    const int tid_all = threadIdx.x;
    const int wid  = tid_all >> 5;
    const int lane = tid_all & 31;
    const int grp  = lane >> 2;
    const int tid  = lane & 3;
    const int sub  = lane >> 3;
    const int r_in = lane & 7;

    const int q0 = CUQ[r];
    const int qlen = CUQ[r + 1] - q0;
    const int seqlen = SEQ[r];
    const int past = seqlen - qlen;
    int r_live = qlen;
    if (has_nlive) {
        int nl = *NLivePtr;
        int d = nl - q0;
        if (d < 0) d = 0;
        if (d < r_live) r_live = d;
    }

    const int live_rows = r_live * G;
    const int actual_mt = min(MT_MAX, (live_rows + BM - 1) / BM);
    if (actual_mt <= 0) {
        if constexpr (USE_PDL) {
            gdc_wait();
            __syncthreads();
            if (trig_wo && tid_all == 0) gdc_launch_dependents();
        }
        return;
    }

    const int k_end = max(past + qlen, 0);
    const int k_beg = (WINDOW >= 0) ? max(past - WINDOW, 0) : 0;
    const int span  = max(k_end - k_beg, 0);
    const int per   = (S_MODE == 0) ? (((span + 63) / 64) * 64) : ((((span + num_splits - 1) / num_splits) + 63) / 64 * 64);
    const int s_beg = k_beg + split * per;
    const int s_end = min(s_beg + per, k_end);
    int s_end_live  = (seqlen > 0 && q0 < NT) ? s_end : s_beg;
    if (has_nlive) {
        if (past + r_live < s_end_live) s_end_live = past + r_live;
    }

    constexpr size_t RED_BYTES = (MT_MAX <= 2) ? (64 * 17 * sizeof(float)) : (64 * 8 * sizeof(float));
    extern __shared__ uint4 smem_raw_u4[];
    uint8_t*  smem_raw     = reinterpret_cast<uint8_t*>(smem_raw_u4);
    uint16_t* smem_kv0     = reinterpret_cast<uint16_t*>(smem_raw);
    uint16_t* smem_kv1     = reinterpret_cast<uint16_t*>(smem_raw + 16384);
    uint16_t* smem_q       = reinterpret_cast<uint16_t*>(smem_raw + 32768);
    uint16_t* smem_bp      = reinterpret_cast<uint16_t*>(smem_raw + 32768 + MT_MAX * 4096);
    void*     red_smem_raw = reinterpret_cast<void*>(smem_raw + 32768 + MT_MAX * 4096 + 2304);
    float (*Max_smem)[8]   = reinterpret_cast<float (*)[8]>(smem_raw + 32768 + MT_MAX * 4096 + 2304 + RED_BYTES);

    alignas(16) __shared__ int2     KV_blk_slot[64];
    alignas(16) __shared__ uint8_t  Row_iloc[MAX_ROWS];
    alignas(16) __shared__ uint8_t  Row_g[MAX_ROWS];
    alignas(16) __shared__ uint8_t  Row_ok[MAX_ROWS];

    if (tid_all < MAX_ROWS) {
        int i_loc = tid_all / G;
        int g = tid_all % G;
        bool ok = (i_loc < r_live);
        Row_ok[tid_all]   = ok ? 1 : 0;
        Row_iloc[tid_all] = static_cast<uint8_t>(i_loc);
        Row_g[tid_all]    = static_cast<uint8_t>(g);
    }

    const uint64_t l2_policy = evict_first ? make_l2_evict_first_policy() : 0ULL;
    const int32_t* bt_row = BT + static_cast<int64_t>(r) * sbt_r;
    const uint16_t* K_base = K + static_cast<int64_t>(kvh) * sk_h;
    const uint16_t* V_base = V + static_cast<int64_t>(kvh) * sv_h;
    const uint8_t* K_bf12_base = reinterpret_cast<const uint8_t*>(K) + static_cast<int64_t>(kvh) * sk_h;
    const uint8_t* V_bf12_base = reinterpret_cast<const uint8_t*>(V) + static_cast<int64_t>(kvh) * sv_h;
    const uint32_t kv0_sptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_kv0));
    const uint32_t kv1_sptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_kv1));
    const uint32_t q_sptr   = static_cast<uint32_t>(__cvta_generic_to_shared(smem_q));
    uint4* smem_kv0_4 = reinterpret_cast<uint4*>(smem_kv0);
    uint4* smem_kv1_4 = reinterpret_cast<uint4*>(smem_kv1);
    uint4* smem_q_4   = reinterpret_cast<uint4*>(smem_q);

    const bool tile0_is_past = (!USE_PDL) && (s_beg < s_end_live) && ((s_beg + BN) <= past);
    if (tile0_is_past) {
        if (tid_all < 64) {
            int kpos = s_beg + tid_all;
            int lblk = (BS_shift >= 0) ? (kpos >> BS_shift) : (kpos / BS);
            int slot = (BS_shift >= 0) ? (kpos & ((1 << BS_shift) - 1)) : (kpos % BS);
            KV_blk_slot[tid_all] = make_int2(bt_row[lblk], slot);
        }
        __syncthreads();
        issue_kv64_maybe_bf12_async<KV_BF12>(K_base, K_bf12_base, sk_b, sk_s, KV_blk_slot, kv0_sptr, smem_kv0_4, evict_first, l2_policy, tid_all);
        cp_async_commit();
        issue_kv64_maybe_bf12_async<KV_BF12>(V_base, V_bf12_base, sv_b, sv_s, KV_blk_slot, kv1_sptr, smem_kv1_4, evict_first, l2_policy, tid_all);
        cp_async_commit();
    } else {
        __syncthreads();
    }

    if constexpr (USE_PDL) { gdc_wait(); }

    float m_i0[MT_MAX], m_i1[MT_MAX], l_i0[MT_MAX], l_i1[MT_MAX];
    float acc[MT_MAX][4][4];
    #pragma unroll
    for (int m = 0; m < MT_MAX; ++m) {
        m_i0[m] = -INFINITY; m_i1[m] = -INFINITY;
        l_i0[m] = 0.0f;      l_i1[m] = 0.0f;
        #pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            acc[m][nt][0] = 0.f; acc[m][nt][1] = 0.f;
            acc[m][nt][2] = 0.f; acc[m][nt][3] = 0.f;
        }
    }

    if (s_beg >= s_end_live) {
        if constexpr (S_MODE > 0) {
            if (tid_all < MAX_ROWS && Row_ok[tid_all]) {
                int tok = q0 + static_cast<int>(Row_iloc[tid_all]);
                int h   = h_base + static_cast<int>(Row_g[tid_all]);
                LP[split * sl_s + tok * sl_t + h * sl_h] = -INFINITY;
            }
        }
    } else {
        if (!tile0_is_past) {
            if (tid_all < 64) {
                int kpos = s_beg + tid_all;
                if (kpos < s_end_live) {
                    int lblk = (BS_shift >= 0) ? (kpos >> BS_shift) : (kpos / BS);
                    int slot = (BS_shift >= 0) ? (kpos & ((1 << BS_shift) - 1)) : (kpos % BS);
                    KV_blk_slot[tid_all] = make_int2(bt_row[lblk], slot);
                } else {
                    KV_blk_slot[tid_all] = make_int2(-1, 0);
                }
            }
            __syncthreads();
            issue_kv64_maybe_bf12_async<KV_BF12>(K_base, K_bf12_base, sk_b, sk_s, KV_blk_slot, kv0_sptr, smem_kv0_4, evict_first, l2_policy, tid_all);
            cp_async_commit();
            issue_kv64_maybe_bf12_async<KV_BF12>(V_base, V_bf12_base, sv_b, sv_s, KV_blk_slot, kv1_sptr, smem_kv1_4, evict_first, l2_policy, tid_all);
            cp_async_commit();
        }

        #pragma unroll
        for (int mt = 0; mt < MT_MAX; ++mt) {
            if (mt < actual_mt) {
                #pragma unroll
                for (int pass = 0; pass < 2; ++pass) {
                    int idx = pass * 128 + tid_all;
                    int lrow = idx >> 4;
                    int c = idx & 15;
                    int swiz_c = c ^ (lrow & 7);
                    int grow = mt * BM + lrow;
                    uint4 qv = make_uint4(0u, 0u, 0u, 0u);
                    if (Row_ok[grow]) {
                        int tok = q0 + static_cast<int>(Row_iloc[grow]);
                        int h   = h_base + static_cast<int>(Row_g[grow]);
                        qv = *reinterpret_cast<const uint4*>(
                            Q + static_cast<int64_t>(tok) * sq_t
                              + static_cast<int64_t>(h) * sq_h + c * 8
                        );
                    }
                    smem_q_4[(mt * BM + lrow) * 16 + swiz_c] = qv;
                }
            }
        }

        uint32_t r_a[MT_MAX][4];
        {
            const int sub_grp = lane >> 3;
            const int sub_lane = lane & 7;
            #pragma unroll
            for (int mt = 0; mt < MT_MAX; ++mt) {
                if (mt < actual_mt) {
                    const int row = mt * BM + wid * 4 + sub_grp;
                    const bool r_ok = (Row_ok[row] != 0);
                    const int tok = r_ok ? (q0 + static_cast<int>(Row_iloc[row])) : 0;
                    const int h   = r_ok ? (h_base + static_cast<int>(Row_g[row])) : 0;
                    const float tau = (has_tau && r_ok) ? Tau[tok] : 1.0f;
                    clip_r_row_exact_8lane(
                        R + static_cast<int64_t>(tok) * sr_t + static_cast<int64_t>(h) * sr_h,
                        RScale + h * D_REL,
                        r_ok, has_tau, tau, rel_threshold, clip_eps, sub_lane, smem_bp + (wid * 4 + sub_grp) * D_REL
                    );
                    __syncthreads();
                    r_a[mt][0] = *reinterpret_cast<const uint32_t*>(smem_bp + grp * D_REL + 2 * tid);
                    r_a[mt][1] = *reinterpret_cast<const uint32_t*>(smem_bp + (grp + 8) * D_REL + 2 * tid);
                    r_a[mt][2] = *reinterpret_cast<const uint32_t*>(smem_bp + grp * D_REL + 8 + 2 * tid);
                    r_a[mt][3] = *reinterpret_cast<const uint32_t*>(smem_bp + (grp + 8) * D_REL + 8 + 2 * tid);
                    __syncthreads();
                } else {
                    r_a[mt][0] = 0u; r_a[mt][1] = 0u; r_a[mt][2] = 0u; r_a[mt][3] = 0u;
                }
            }
        }

        const int qpos0_min = past + static_cast<int>(Row_iloc[0]);
        const int qpos0_max = past + static_cast<int>(Row_iloc[15]);

        for (int n0 = s_beg; n0 < s_end_live; n0 += BN) {
            const int d0_lo = qpos0_min - (n0 + BN - 1);
            const int d0_hi = qpos0_max - n0;
            if ((E > 0) && (d0_hi >= 0) && (d0_lo < E)) {
                compute_otf_bias_tile<B_COLS_FMT>(
                    RelProjT, smem_bp, r_a[0][0], r_a[0][1], r_a[0][2], r_a[0][3],
                    max(0, d0_lo), d0_hi, E, wid, grp, tid
                );
            }

            cp_async_wait_group<1>();
            __syncthreads();
            if constexpr (KV_BF12) {
                kv_bf12_unpack_tile64_inplace_swiz(smem_kv0, K_bf12_base, sk_b, KEmergRaw, tid_all);
            }

            const int n_next = n0 + BN;

            if constexpr (WINDOW < 0 && MT_MAX >= 2) {
                // Wave-V13 Factored Zero-Duplication Split-Phase Dual-Inflight Pipeline:
                //   ONLY 1 `issue_K`, 1 `unpack(V)`, 1 `issue_V`, and 1 `k_compute`/`v_mma` per mt!
                float dummy_l0 = 0.f, dummy_l1 = 0.f;
                float a0_0 = 0.f, a1_0 = 0.f, a0_1 = 0.f, a1_1 = 0.f, a0_2 = 0.f, a1_2 = 0.f;
                uint32_t pk0_0 = 0u, pk0_1 = 0u, pk0_2 = 0u, pk0_3 = 0u;
                uint32_t pk1_0 = 0u, pk1_1 = 0u, pk1_2 = 0u, pk1_3 = 0u;
                uint32_t pk2_0 = 0u, pk2_1 = 0u, pk2_2 = 0u, pk2_3 = 0u;

                fusedmt_global_k_compute_v13<MT_MAX, 0>(
                    actual_mt == 1, past, n0, n_next, s_end_live, E, BS, BS_shift, scale_log2,
                    RelProjT, bt_row, q_sptr, kv0_sptr, smem_bp, red_smem_raw, Max_smem,
                    KV_blk_slot, Row_iloc, Row_ok,
                    r_a[0][0], r_a[0][1], r_a[0][2], r_a[0][3],
                    m_i0[0], m_i1[0], acc[0],
                    dummy_l0, dummy_l1, 0.f, 0.f, a0_0, a1_0,
                    pk0_0, pk0_1, pk0_2, pk0_3,
                    tid_all, wid, lane, grp, tid, sub, r_in
                );
                if (actual_mt >= 2) {
                    fusedmt_global_k_compute_v13<MT_MAX, 1>(
                        actual_mt == 2, past, n0, n_next, s_end_live, E, BS, BS_shift, scale_log2,
                        RelProjT, bt_row, q_sptr + 4096u, kv0_sptr, smem_bp, red_smem_raw, Max_smem,
                        KV_blk_slot, Row_iloc, Row_ok,
                        r_a[1][0], r_a[1][1], r_a[1][2], r_a[1][3],
                        m_i0[1], m_i1[1], acc[1],
                        l_i0[0], l_i1[0], a0_0, a1_0, a0_1, a1_1,
                        pk1_0, pk1_1, pk1_2, pk1_3,
                        tid_all, wid, lane, grp, tid, sub, r_in
                    );
                }
                if constexpr (MT_MAX == 3) {
                    if (actual_mt == 3) {
                        fusedmt_global_k_compute_v13<MT_MAX, 2>(
                            true, past, n0, n_next, s_end_live, E, BS, BS_shift, scale_log2,
                            RelProjT, bt_row, q_sptr + 8192u, kv0_sptr, smem_bp, red_smem_raw, Max_smem,
                            KV_blk_slot, Row_iloc, Row_ok,
                            r_a[2][0], r_a[2][1], r_a[2][2], r_a[2][3],
                            m_i0[2], m_i1[2], acc[2],
                            l_i0[1], l_i1[1], a0_1, a1_1, a0_2, a1_2,
                            pk2_0, pk2_1, pk2_2, pk2_3,
                            tid_all, wid, lane, grp, tid, sub, r_in
                        );
                    }
                }

                // Single shared K(n_next) issue (smem_kv0 dead & KV_blk_slot visible after last Max barrier!)
                if (n_next < s_end_live) {
                    issue_kv64_maybe_bf12_async<KV_BF12>(
                        K_base, K_bf12_base, sk_b, sk_s, KV_blk_slot, kv0_sptr, smem_kv0_4,
                        evict_first, l2_policy, tid_all
                    );
                    cp_async_commit();
                }

                // Stage last active mt's P pack into smem_bp before V-wait barrier
                if constexpr (MT_MAX == 3) {
                    if (actual_mt == 3)      fusedmt_store_p_pack(smem_bp, pk2_0, pk2_1, pk2_2, pk2_3, wid, grp, tid);
                    else if (actual_mt == 2) fusedmt_store_p_pack(smem_bp, pk1_0, pk1_1, pk1_2, pk1_3, wid, grp, tid);
                    else                     fusedmt_store_p_pack(smem_bp, pk0_0, pk0_1, pk0_2, pk0_3, wid, grp, tid);
                } else {
                    if (actual_mt == 2)      fusedmt_store_p_pack(smem_bp, pk1_0, pk1_1, pk1_2, pk1_3, wid, grp, tid);
                    else                     fusedmt_store_p_pack(smem_bp, pk0_0, pk0_1, pk0_2, pk0_3, wid, grp, tid);
                }

                if (n_next < s_end_live) cp_async_wait_group<1>();
                else                     cp_async_wait_group<0>();
                __syncthreads();  // V(n0), smem_bp, and Red[last_mt] visible!

                if constexpr (KV_BF12) {
                    kv_bf12_unpack_tile64_inplace_swiz(smem_kv1, V_bf12_base, sv_b, VEmergRaw, tid_all);
                }

                // Finalize last active mt's l_i0/l_i1 from its Red slice
                float last_sp0, last_sp1;
                if constexpr (MT_MAX == 3) {
                    if (actual_mt == 3) {
                        fusedmt_global_red_read<MT_MAX, 2>(red_smem_raw, grp, last_sp0, last_sp1);
                        l_i0[2] = fma_rn_f32(l_i0[2], a0_2, last_sp0);
                        l_i1[2] = fma_rn_f32(l_i1[2], a1_2, last_sp1);
                    } else if (actual_mt == 2) {
                        fusedmt_global_red_read<MT_MAX, 1>(red_smem_raw, grp, last_sp0, last_sp1);
                        l_i0[1] = fma_rn_f32(l_i0[1], a0_1, last_sp0);
                        l_i1[1] = fma_rn_f32(l_i1[1], a1_1, last_sp1);
                    } else {
                        fusedmt_global_red_read<MT_MAX, 0>(red_smem_raw, grp, last_sp0, last_sp1);
                        l_i0[0] = fma_rn_f32(l_i0[0], a0_0, last_sp0);
                        l_i1[0] = fma_rn_f32(l_i1[0], a1_0, last_sp1);
                    }
                } else {
                    if (actual_mt == 2) {
                        fusedmt_global_red_read<MT_MAX, 1>(red_smem_raw, grp, last_sp0, last_sp1);
                        l_i0[1] = fma_rn_f32(l_i0[1], a0_1, last_sp0);
                        l_i1[1] = fma_rn_f32(l_i1[1], a1_1, last_sp1);
                    } else {
                        fusedmt_global_red_read<MT_MAX, 0>(red_smem_raw, grp, last_sp0, last_sp1);
                        l_i0[0] = fma_rn_f32(l_i0[0], a0_0, last_sp0);
                        l_i1[0] = fma_rn_f32(l_i1[0], a1_0, last_sp1);
                    }
                }

                // Single-copy reverse V-MMA chain (exact same barrier count as V11!)
                if constexpr (MT_MAX == 3) {
                    if (actual_mt == 3) {
                        fusedmt_global_v_mma(smem_bp, kv1_sptr, acc[2], wid, grp, tid, sub, r_in);
                        __syncthreads();
                        fusedmt_store_p_pack(smem_bp, pk1_0, pk1_1, pk1_2, pk1_3, wid, grp, tid);
                        __syncthreads();
                    }
                }
                if (actual_mt >= 2) {
                    fusedmt_global_v_mma(smem_bp, kv1_sptr, acc[1], wid, grp, tid, sub, r_in);
                    __syncthreads();
                    fusedmt_store_p_pack(smem_bp, pk0_0, pk0_1, pk0_2, pk0_3, wid, grp, tid);
                    __syncthreads();
                }
                fusedmt_global_v_mma(smem_bp, kv1_sptr, acc[0], wid, grp, tid, sub, r_in);

                // Single shared V(n_next) issue!
                if (n_next < s_end_live) {
                    __syncthreads();
                    issue_kv64_maybe_bf12_async<KV_BF12>(
                        V_base, V_bf12_base, sv_b, sv_s, KV_blk_slot, kv1_sptr, smem_kv1_4,
                        evict_first, l2_policy, tid_all
                    );
                    cp_async_commit();
                }
            } else {
                // Local SWA (WINDOW >= 0) or MT_MAX == 1 — single factored pass, 0 duplication!
                fusedmt_step_tile_v13<WINDOW, MT_MAX, true, KV_BF12>(
                    0, actual_mt == 1, past, n0, n_next, s_end_live, E, BS, BS_shift, scale_log2,
                    RelProjT, K_base, V_base, K_bf12_base, V_bf12_base, VEmergRaw,
                    sk_b, sk_s, sv_b, sv_s, bt_row,
                    q_sptr, kv0_sptr, kv1_sptr, smem_kv0_4, smem_kv1_4,
                    smem_bp, red_smem_raw, Max_smem, KV_blk_slot, Row_iloc, Row_ok,
                    r_a[0][0], r_a[0][1], r_a[0][2], r_a[0][3],
                    m_i0[0], m_i1[0], l_i0[0], l_i1[0], acc[0],
                    evict_first, l2_policy, tid_all, wid, lane, grp, tid, sub, r_in
                );
                if constexpr (MT_MAX >= 2) {
                    if (actual_mt >= 2) {
                        fusedmt_step_tile_v13<WINDOW, MT_MAX, false, KV_BF12>(
                            1, actual_mt == 2, past, n0, n_next, s_end_live, E, BS, BS_shift, scale_log2,
                            RelProjT, K_base, V_base, K_bf12_base, V_bf12_base, VEmergRaw,
                            sk_b, sk_s, sv_b, sv_s, bt_row,
                            q_sptr + 4096u, kv0_sptr, kv1_sptr, smem_kv0_4, smem_kv1_4,
                            smem_bp, red_smem_raw, Max_smem, KV_blk_slot, Row_iloc, Row_ok,
                            r_a[1][0], r_a[1][1], r_a[1][2], r_a[1][3],
                            m_i0[1], m_i1[1], l_i0[1], l_i1[1], acc[1],
                            evict_first, l2_policy, tid_all, wid, lane, grp, tid, sub, r_in
                        );
                    }
                }
                if constexpr (MT_MAX == 3) {
                    if (actual_mt == 3) {
                        fusedmt_step_tile_v13<WINDOW, MT_MAX, false, KV_BF12>(
                            2, true, past, n0, n_next, s_end_live, E, BS, BS_shift, scale_log2,
                            RelProjT, K_base, V_base, K_bf12_base, V_bf12_base, VEmergRaw,
                            sk_b, sk_s, sv_b, sv_s, bt_row,
                            q_sptr + 8192u, kv0_sptr, kv1_sptr, smem_kv0_4, smem_kv1_4,
                            smem_bp, red_smem_raw, Max_smem, KV_blk_slot, Row_iloc, Row_ok,
                            r_a[2][0], r_a[2][1], r_a[2][2], r_a[2][3],
                            m_i0[2], m_i1[2], l_i0[2], l_i1[2], acc[2],
                            evict_first, l2_policy, tid_all, wid, lane, grp, tid, sub, r_in
                        );
                    }
                }
            }
        }
    }

    if constexpr (S_MODE == 0) {
        if constexpr (USE_PDL) {
            __syncthreads();
            if (trig_wo && tid_all == 0) gdc_launch_dependents();
        }
        #pragma unroll
        for (int mt = 0; mt < MT_MAX; ++mt) {
            if (mt < actual_mt) {
                const int r0 = mt * BM + grp;
                const int r1 = r0 + 8;
                store_direct_gate_sink_row(
                    Row_ok[r0] != 0, q0 + static_cast<int>(Row_iloc[r0]), h_base + static_cast<int>(Row_g[r0]),
                    NT, m_i0[mt], l_i0[mt],
                    acc[mt][0][0], acc[mt][0][1], acc[mt][1][0], acc[mt][1][1],
                    acc[mt][2][0], acc[mt][2][1], acc[mt][3][0], acc[mt][3][1],
                    GATE, SINK, Y, sg_t, ssk_t, sy_t, wid, tid
                );
                store_direct_gate_sink_row(
                    Row_ok[r1] != 0, q0 + static_cast<int>(Row_iloc[r1]), h_base + static_cast<int>(Row_g[r1]),
                    NT, m_i1[mt], l_i1[mt],
                    acc[mt][0][2], acc[mt][0][3], acc[mt][1][2], acc[mt][1][3],
                    acc[mt][2][2], acc[mt][2][3], acc[mt][3][2], acc[mt][3][3],
                    GATE, SINK, Y, sg_t, ssk_t, sy_t, wid, tid
                );
            }
        }
    } else {
        if (s_beg < s_end_live) {
            #pragma unroll
            for (int mt = 0; mt < MT_MAX; ++mt) {
                if (mt < actual_mt) {
                    const int r0 = mt * BM + grp;
                    const int r1 = r0 + 8;
                    const float l_safe0 = (l_i0[mt] > 0.0f) ? l_i0[mt] : 1.0f;
                    const float l_safe1 = (l_i1[mt] > 0.0f) ? l_i1[mt] : 1.0f;
                    if (Row_ok[r0]) {
                        const int tok0 = q0 + static_cast<int>(Row_iloc[r0]);
                        const int h0   = h_base + static_cast<int>(Row_g[r0]);
                        float* o0 = OP + split * so_s + tok0 * so_t + h0 * so_h + wid * 32;
                        #pragma unroll
                        for (int nt = 0; nt < 4; ++nt) {
                            o0[nt * 8 + 2 * tid]     = div_full_f32(acc[mt][nt][0], l_safe0);
                            o0[nt * 8 + 2 * tid + 1] = div_full_f32(acc[mt][nt][1], l_safe0);
                        }
                        if (wid == 0 && tid == 0) {
                            LP[split * sl_s + tok0 * sl_t + h0 * sl_h] =
                                (l_i0[mt] > 0.0f) ? __fmul_rn(__fadd_rn(m_i0[mt], log2f(l_safe0)), LN2) : -INFINITY;
                        }
                    }
                    if (Row_ok[r1]) {
                        const int tok1 = q0 + static_cast<int>(Row_iloc[r1]);
                        const int h1   = h_base + static_cast<int>(Row_g[r1]);
                        float* o1 = OP + split * so_s + tok1 * so_t + h1 * so_h + wid * 32;
                        #pragma unroll
                        for (int nt = 0; nt < 4; ++nt) {
                            o1[nt * 8 + 2 * tid]     = div_full_f32(acc[mt][nt][2], l_safe1);
                            o1[nt * 8 + 2 * tid + 1] = div_full_f32(acc[mt][nt][3], l_safe1);
                        }
                        if (wid == 0 && tid == 0) {
                            LP[split * sl_s + tok1 * sl_t + h1 * sl_h] =
                                (l_i1[mt] > 0.0f) ? __fmul_rn(__fadd_rn(m_i1[mt], log2f(l_safe1)), LN2) : -INFINITY;
                        }
                    }
                }
            }
        }
        __threadfence();
        __syncthreads();
        alignas(16) __shared__ int s_done;
        if (tid_all == 0) s_done = atomic_add_acq_rel_gpu(&SEM[sem_idx], 1);
        __syncthreads();
        if constexpr (USE_PDL) {
            if (trig_wo && tid_all == 0) gdc_launch_dependents();
        }
        if (s_done == num_splits - 1) {
            if (tid_all == 0) SEM[sem_idx] = 0;
            cooperative_merge_gate_sink_v13<S_MODE, true>(
                num_splits, actual_mt * BM, Row_ok, nullptr, nullptr, Row_iloc, Row_g, q0, h_base,
                OP, LP, GATE, SINK, Y,
                so_s, so_t, so_h, sl_s, sl_t, sl_h, sg_t, ssk_t, sy_t, NT, wid, lane
            );
        }
    }
}

template <typename KernelT, typename... Args>
static inline cudaError_t launch_maybe_pdl(
    KernelT kernel, dim3 grid, dim3 block, size_t smem, cudaStream_t stream, bool pdl, Args... args
) {
    if (!pdl) {
        kernel<<<grid, block, smem, stream>>>(args...);
        return cudaGetLastError();
    }
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = smem;
    cfg.stream = stream;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    return cudaLaunchKernelEx(&cfg, kernel, args...);
}

#endif  // __CUDACC__ || ATTN_V13_DEVICE_TU

}  // namespace attn_v13
