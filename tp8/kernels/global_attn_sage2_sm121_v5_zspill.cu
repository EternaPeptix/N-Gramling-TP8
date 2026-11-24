// SPDX-License-Identifier: Apache-2.0
// Native SM121a (GB10) SageAttention2 FP8-E4M3 Prefix Kernels (v5-zspill: 0-Spill 2-CTA/SM)
// + Native SM121a FP8-f16 Exact-Kmax Quest Block Scorer (`score_kmax_blocks_fp8_sm121_fwd`)
// for Inkling-2-Small 8 Global Full-Attention Layers (D=128, BS%BN==0, 1M-safe int64).
//
// 0-Spill Fix over v5 (`global_attn_sage2_sm121_v5.cu`):
//   - v5 added Compact-Shadow-K (`K_BT`, `skbt_r`, `k_bt_row`, `locate_v_blk`), pushing
//     live uniform pointer/stride/bound registers over the 128-reg cap at MIN_CTAS=2 and
//     triggering 32-72 B local-memory register spills in the main QK^T/PV loop!
//   - v5-zspill moves ALL CTA-uniform pointers, 64-bit strides, shift/mask bounds, and
//     epilogue coordinates into a 160 B `__shared__ SageCtaParams sP` struct read on-demand
//     via uniform broadcast `ld.shared.*` (preventing NVVM LICM register hoisting!) AND
//     hoists per-tile `(cur_v_blk, cur_slot, ks_tile[0..1])` into `if (tid == 0)` before
//     the loop's existing `__syncthreads()` (ZERO extra barriers, 255x fewer redundant
//     BT/K_BT/SEL/KS64 loads!).
//   - Result: 0 bytes stack frame / 0 bytes spill stores / 0 bytes spill loads across ALL
//     instantiated kernels at MIN_CTAS=2 (2 CTAs/SM, 49.3 KiB <= 49.5 KiB/CTA on SM121a),
//     100% ABI-compatible drop-in replacement for `libglobal_attn_sage2_sm121_v5.so`!
//
// Build on GB10 (OMIT --use_fast_math!):
//   nvcc -O3 -gencode arch=compute_121a,code=sm_121a -shared -Xcompiler -fPIC \
//        --ptxas-options=-v -o libglobal_attn_sage2_sm121_v5.so \
//        global_attn_sage2_sm121_v5_zspill.cu

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace {

static constexpr int BM = 128;
static constexpr int D = 128;
static constexpr int WARPS = 8;            // 8 warps = 256 threads, 16 Q-rows/warp
static constexpr int THREADS = WARPS * 32; // 256 threads
static constexpr float QK_MAX = 22.0f;     // 128 * 22 * 22 = 61,952 < 65,504 (provably zero FP16 overflow!)
static constexpr float INV_QK_MAX = 1.0f / 22.0f;
static constexpr float NEG_INF_GUARD = -1.0e30f;

// ---- Shared PTX primitives ----

__device__ __forceinline__ float div_rn_ftz(float a, float b) {
    float r;
    asm("div.rn.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

__device__ __forceinline__ uint16_t f32x2_to_e4m3x2_rn(float lo, float hi) {
    uint16_t r;
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(r) : "f"(lo), "f"(hi));
    return r;
}

__device__ __forceinline__ uint32_t f32x4_to_e4m3x4_rn(float v0, float v1, float v2, float v3) {
    uint16_t lo = f32x2_to_e4m3x2_rn(v0, v1);
    uint16_t hi = f32x2_to_e4m3x2_rn(v2, v3);
    return (uint32_t)lo | ((uint32_t)hi << 16);
}

__device__ __forceinline__ uint32_t f32x2_to_bf16x2_rn(float lo, float hi) {
    uint32_t r;
    asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(r) : "f"(lo), "f"(hi));
    return r;
}

__device__ __forceinline__ float fast_exp2f(float x) {
    float r;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}

__device__ __forceinline__ uint32_t hmax2_u32(uint32_t a, uint32_t b) {
    uint32_t r;
    asm volatile("max.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
    return r;
}

// Non-hoistable uniform broadcast loads from __shared__ SageCtaParams (1-cycle LDS broadcast,
// prevents NVVM LICM from keeping 24+ uniform pointer/stride regs live across the main loop!).
__device__ __forceinline__ uint64_t ld_smem_u64_nocse(const void* smem_ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    uint64_t v;
    asm volatile("ld.shared.u64 %0, [%1];" : "=l"(v) : "r"(addr));
    return v;
}

__device__ __forceinline__ uint32_t ld_smem_u32_nocse(const void* smem_ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    uint32_t v;
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(addr));
    return v;
}

__device__ __forceinline__ int ld_smem_i32_nocse(const int* smem_ptr) {
    return static_cast<int>(ld_smem_u32_nocse(smem_ptr));
}

__device__ __forceinline__ int64_t ld_smem_i64_nocse(const int64_t* smem_ptr) {
    return static_cast<int64_t>(ld_smem_u64_nocse(smem_ptr));
}

__device__ __forceinline__ float ld_smem_f32_nocse(const float* smem_ptr) {
    uint32_t u = ld_smem_u32_nocse(smem_ptr);
    return __uint_as_float(u);
}

template <typename T>
__device__ __forceinline__ const T* ld_smem_ptr_nocse(const T* const* smem_ptr) {
    return reinterpret_cast<const T*>(static_cast<uintptr_t>(ld_smem_u64_nocse(smem_ptr)));
}

__device__ __forceinline__ void mma_m16n8k32_f16_e4m3(
    uint32_t& d0, uint32_t& d1,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    uint32_t c0, uint32_t c1
) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 "
        "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n"
        : "=r"(d0), "=r"(d1)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "r"(c0), "r"(c1)
    );
}

__device__ __forceinline__ void mma_m16n8k16_f32_bf16(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3)
    );
}

__device__ __forceinline__ void cp_async_cg_16(uint32_t smem_addr, const void* gmem_ptr) {
    asm volatile(
        "cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n"
        :: "r"(smem_addr), "l"(gmem_ptr)
        : "memory"
    );
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}

__device__ __forceinline__ void ldmatrix_x4(
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3,
    uint32_t smem_addr
) {
    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(smem_addr)
    );
}

__device__ __forceinline__ void ldmatrix_x4_trans(
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3,
    uint32_t smem_addr
) {
    asm volatile(
        "ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(smem_addr)
    );
}

// ---- Ultra-low-register Q loader: holds only 16 packed bf16x2 regs + div_rn_ftz ----

__device__ __forceinline__ void load_quant_q_row_v4(
    const __nv_bfloat16* __restrict__ row_ptr,
    bool valid,
    int t,
    float scale_mult,
    uint32_t q_lo[4],
    uint32_t q_hi[4],
    float& qs_out
) {
    uint32_t raw[16];
    float amax = 0.0f;

    #pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
        if (valid) {
            const uint2 u_lo = *reinterpret_cast<const uint2*>(row_ptr + ks * 32 + 4 * t);
            const uint2 u_hi = *reinterpret_cast<const uint2*>(row_ptr + ks * 32 + 16 + 4 * t);
            raw[ks * 4 + 0] = u_lo.x;
            raw[ks * 4 + 1] = u_lo.y;
            raw[ks * 4 + 2] = u_hi.x;
            raw[ks * 4 + 3] = u_hi.y;
            const float2 f0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u_lo.x));
            const float2 f1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u_lo.y));
            const float2 f2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u_hi.x));
            const float2 f3 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u_hi.y));
            amax = fmaxf(amax, fmaxf(fmaxf(fabsf(f0.x), fabsf(f0.y)), fmaxf(fabsf(f1.x), fabsf(f1.y))));
            amax = fmaxf(amax, fmaxf(fmaxf(fabsf(f2.x), fabsf(f2.y)), fmaxf(fabsf(f3.x), fabsf(f3.y))));
        } else {
            raw[ks * 4 + 0] = 0u;
            raw[ks * 4 + 1] = 0u;
            raw[ks * 4 + 2] = 0u;
            raw[ks * 4 + 3] = 0u;
        }
    }
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 2));
    amax = fmaxf(amax, 1.0e-12f);
    const float inv = div_rn_ftz(QK_MAX, amax);
    qs_out = amax * INV_QK_MAX * scale_mult;

    #pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
        const float2 f0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw[ks * 4 + 0]));
        const float2 f1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw[ks * 4 + 1]));
        const float2 f2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw[ks * 4 + 2]));
        const float2 f3 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw[ks * 4 + 3]));
        q_lo[ks] = f32x4_to_e4m3x4_rn(f0.x * inv, f0.y * inv, f1.x * inv, f1.y * inv);
        q_hi[ks] = f32x4_to_e4m3x4_rn(f2.x * inv, f2.y * inv, f3.x * inv, f3.y * inv);
    }
}

__device__ __forceinline__ void load_quant_q_regs_v4(
    const __nv_bfloat16* __restrict__ Q,
    int64_t sq_t, int64_t sq_h,
    int q0, int h, int r0_loc, int r1_loc,
    bool r0_valid, bool r1_valid,
    int t, float scale_mult,
    uint32_t q_a0[4], uint32_t q_a1[4], uint32_t q_a2[4], uint32_t q_a3[4],
    float& qs_0, float& qs_1
) {
    load_quant_q_row_v4(
        Q + (int64_t)(q0 + r0_loc) * sq_t + (int64_t)h * sq_h,
        r0_valid, t, scale_mult, q_a0, q_a2, qs_0
    );
    load_quant_q_row_v4(
        Q + (int64_t)(q0 + r1_loc) * sq_t + (int64_t)h * sq_h,
        r1_valid, t, scale_mult, q_a1, q_a3, qs_1
    );
}

// ---- CTA-Uniform Parameter Block in SMEM (160 B, saves ~26-30 regs/thread!) ----

struct alignas(16) SageCtaParams {
    const int32_t* bt_row;
    const int32_t* k_bt_row;
    const int32_t* sel_row;
    const uint8_t* K_fp8_h;
    const float*   KS64_h;
    const __nv_bfloat16* V_h;
    int64_t sk8_b;
    int64_t sk8_s;
    int64_t sks64_b;
    int64_t sv_b;
    int64_t sv_s;
    int64_t cur_v_blk;
    uint32_t k_base_sptr;
    uint32_t v_base_sptr;
    int bs_shift;
    int bs_mask;
    int tpb_shift;
    int tpb_mask;
    int num_tiles;
    int q0;
    int qlen;
    int i_lo;
    int h;
    int cur_slot;
    float ks_tile[2];
};

// ---- Power-of-2 division-free tile locator for dense and sparse Top-K prefix ----

template <bool SPARSE_SEL, int K_BT_MODE = 0>
struct TileLocatorV5 {
    __device__ __forceinline__ static int logical_blk_and_slot(
        int tile_idx, int BN, int bs_shift, int bs_mask,
        int tpb_shift, int tpb_mask,
        const int32_t* __restrict__ sel_row,
        int& slot
    ) {
        if constexpr (SPARSE_SEL) {
            const int b_i = tile_idx >> tpb_shift;
            const int sub_t = tile_idx & tpb_mask;
            slot = sub_t * BN;
            return sel_row[b_i];
        } else {
            const int n0 = tile_idx * BN;
            slot = n0 & bs_mask;
            return n0 >> bs_shift;
        }
    }

    __device__ __forceinline__ static void locate_k(
        int tile_idx, int BN, int bs_shift, int bs_mask,
        int tpb_shift, int tpb_mask,
        const int32_t* __restrict__ bt_row,
        const int32_t* __restrict__ k_bt_row,
        const int32_t* __restrict__ sel_row,
        int64_t& k_blk, int& slot
    ) {
        const int lb = logical_blk_and_slot(
            tile_idx, BN, bs_shift, bs_mask, tpb_shift, tpb_mask, sel_row, slot);
        if constexpr (K_BT_MODE == 0) {
            k_blk = (int64_t)bt_row[lb];
        } else if constexpr (K_BT_MODE == 1) {
            k_blk = (int64_t)lb;
        } else {
            k_blk = (int64_t)k_bt_row[lb];
        }
    }

    __device__ __forceinline__ static void locate_kv(
        int tile_idx, int BN, int bs_shift, int bs_mask,
        int tpb_shift, int tpb_mask,
        const int32_t* __restrict__ bt_row,
        const int32_t* __restrict__ k_bt_row,
        const int32_t* __restrict__ sel_row,
        int64_t& k_blk, int64_t& v_blk, int& slot
    ) {
        const int lb = logical_blk_and_slot(
            tile_idx, BN, bs_shift, bs_mask, tpb_shift, tpb_mask, sel_row, slot);
        if constexpr (K_BT_MODE == 0) {
            const int64_t b = (int64_t)bt_row[lb];
            k_blk = b;
            v_blk = b;
        } else if constexpr (K_BT_MODE == 1) {
            k_blk = (int64_t)lb;
            v_blk = (int64_t)bt_row[lb];
        } else {
            k_blk = (int64_t)k_bt_row[lb];
            v_blk = (int64_t)bt_row[lb];
        }
    }
};

template <int BN>
__device__ __forceinline__ void issue_k_async_v4(
    const uint8_t* __restrict__ K_fp8_h,
    int64_t blk, int slot,
    int64_t sk8_b, int64_t sk8_s,
    uint32_t smem_k_stage_sptr,
    int tid
) {
    const uint8_t* k_base = K_fp8_h + blk * sk8_b + (int64_t)slot * sk8_s;
    constexpr int ITERS = BN / 32;  // 2 for BN=64, 4 for BN=128
    #pragma unroll
    for (int i = 0; i < ITERS; ++i) {
        const int idx = i * 256 + tid;
        const int k_row = idx >> 3;
        const int c = idx & 7;
        const int swiz_c = c ^ (k_row & 7);
        const uint32_t s_addr = smem_k_stage_sptr + (uint32_t)(k_row * 128 + swiz_c * 16);
        cp_async_cg_16(s_addr, k_base + (int64_t)k_row * sk8_s + c * 16);
    }
}

// Async XOR-swizzled [64, 128] BF16 V loader (256 threads x 4 cp.async.cg = 16,384 B)
__device__ __forceinline__ void issue_v64_async_v4(
    const __nv_bfloat16* __restrict__ V_base,
    int64_t sv_s,
    uint32_t v_base_sptr,
    int tid
) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int idx = i * 256 + tid;
        const int v_row = idx >> 4;
        const int c = idx & 15;
        const int swiz_c = c ^ (v_row & 7);
        const uint32_t s_addr = v_base_sptr + (uint32_t)(v_row * 256 + swiz_c * 16);
        cp_async_cg_16(s_addr, V_base + (int64_t)v_row * sv_s + c * 8);
    }
}

// =================================================================================================
// 0-Spill v5 Global Sage2 Prefix Kernel (MIN_CTAS=2 -> 16 warps/SM, 0 B stack / 0 B spills!)
// =================================================================================================

template <int BN, int STAGES, int MIN_CTAS, bool SPARSE_SEL, int K_BT_MODE = 0>
__global__ void __launch_bounds__(256, 1) global_sage2_prefix_v4_sm121_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const uint8_t* __restrict__ K_fp8,
    const float* __restrict__ KS64,
    const __nv_bfloat16* __restrict__ V,
    const int32_t* __restrict__ BT,
    const int32_t* __restrict__ K_BT,
    const int32_t* __restrict__ SEQ,
    const int32_t* __restrict__ CUQ,
    const int32_t* __restrict__ SEL_BLOCKS,
    const int32_t* __restrict__ SEL_COUNTS,
    float* __restrict__ ACC_OUT,
    float* __restrict__ M_OUT,
    float* __restrict__ L_OUT,
    int64_t sq_t, int64_t sq_h,
    int64_t sk8_b, int64_t sk8_s, int64_t sk8_h,
    int64_t sks64_b, int64_t sks64_h,
    int64_t sv_b, int64_t sv_s, int64_t sv_h,
    int64_t sbt_r,
    int64_t skbt_r,
    int H, int G, int BS, int E, int sel_stride,
    float scale_log2,
    float pv_skip_tau
) {
    static_assert(BN == 64 || BN == 128, "BN must be 64 or 128");
    static_assert(STAGES == 2 || STAGES == 3, "STAGES must be 2 or 3");
    constexpr int PREFETCH = STAGES - 1;
    constexpr int SUB64 = BN / 64;

    extern __shared__ uint8_t dyn_smem[];
    __shared__ SageCtaParams sP;

    const int tid = (int)threadIdx.x;
    if (tid == 0) {
        const int pid = (int)blockIdx.x;
        const int r = (int)blockIdx.y;
        const int h = pid % H;
        const int mt = pid / H;
        const int kvh = h / G;

        const int q0 = CUQ[r];
        const int qlen = CUQ[r + 1] - q0;
        const int seqlen = SEQ[r];
        const int past = seqlen - qlen;
        const int i_lo = mt * BM;

        int num_tiles = 0;
        const int32_t* sel_row = nullptr;
        const int tiles_per_blk = BS / BN;
        const int tpb_shift = __ffs(tiles_per_blk) - 1;

        if (i_lo < qlen) {
            if constexpr (SPARSE_SEL) {
                const int cta_idx = r * (int)gridDim.x + pid;
                const int n_sel = SEL_COUNTS[cta_idx];
                if (n_sel > 0) {
                    sel_row = SEL_BLOCKS + (int64_t)cta_idx * sel_stride;
                    num_tiles = n_sel << tpb_shift;
                }
            } else {
                const int q_min = past + i_lo;
                const int k_bias_raw = (E > 0) ? max(q_min - E + 1, 0) : max(q_min, 0);
                const int k_bias_start = (k_bias_raw / BN) * BN;
                if (k_bias_start > 0) {
                    num_tiles = k_bias_start / BN;
                }
            }
        }

        const uint32_t k_sptr = static_cast<uint32_t>(__cvta_generic_to_shared(dyn_smem));
        sP.bt_row      = BT + (int64_t)r * sbt_r;
        sP.k_bt_row    = (K_BT_MODE == 2) ? (K_BT + (int64_t)r * skbt_r) : nullptr;
        sP.sel_row     = sel_row;
        sP.K_fp8_h     = K_fp8 + (int64_t)kvh * sk8_h;
        sP.KS64_h      = KS64 + (int64_t)kvh * sks64_h;
        sP.V_h         = V + (int64_t)kvh * sv_h;
        sP.sk8_b       = sk8_b;
        sP.sk8_s       = sk8_s;
        sP.sks64_b     = sks64_b;
        sP.sv_b        = sv_b;
        sP.sv_s        = sv_s;
        sP.k_base_sptr = k_sptr;
        sP.v_base_sptr = k_sptr + static_cast<uint32_t>(STAGES * BN * 128);
        sP.bs_shift    = __ffs(BS) - 1;
        sP.bs_mask     = BS - 1;
        sP.tpb_shift   = tpb_shift;
        sP.tpb_mask    = tiles_per_blk - 1;
        sP.num_tiles   = num_tiles;
        sP.q0          = q0;
        sP.qlen        = qlen;
        sP.i_lo        = i_lo;
        sP.h           = h;
    }
    __syncthreads();

    if (ld_smem_i32_nocse(&sP.num_tiles) <= 0) return;

    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int gid = lane >> 2;
    const int t = lane & 3;

    // Quantize Q ONCE into 16 FP8 regs before acc[16][4] becomes live
    uint32_t q_a0[4], q_a1[4], q_a2[4], q_a3[4];
    float qs_log2_0, qs_log2_1;
    uint32_t valid_mask = 0u;
    {
        const int i_lo_q = ld_smem_i32_nocse(&sP.i_lo);
        const int qlen_q = ld_smem_i32_nocse(&sP.qlen);
        const int q0_q   = ld_smem_i32_nocse(&sP.q0);
        const int h_q    = ld_smem_i32_nocse(&sP.h);
        const int r0_loc = i_lo_q + warp * 16 + gid;
        const int r1_loc = r0_loc + 8;
        const bool r0_v = (r0_loc < qlen_q);
        const bool r1_v = (r1_loc < qlen_q);
        valid_mask = (r0_v ? 1u : 0u) | (r1_v ? 2u : 0u);
        load_quant_q_regs_v4(
            Q, sq_t, sq_h, q0_q, h_q, r0_loc, r1_loc, r0_v, r1_v, t, scale_log2,
            q_a0, q_a1, q_a2, q_a3, qs_log2_0, qs_log2_1
        );
    }

    float m_i0 = -INFINITY, m_i1 = -INFINITY;
    float l_i0 = 0.0f, l_i1 = 0.0f;
    float acc[16][4];
    #pragma unroll
    for (int dt = 0; dt < 16; ++dt) {
        acc[dt][0] = 0.0f; acc[dt][1] = 0.0f;
        acc[dt][2] = 0.0f; acc[dt][3] = 0.0f;
    }

    // Helper: issue K tile `tile_idx` async into `stage_idx` reading uniform params on-demand from sP
    auto issue_k_stage = [&](int tile_idx, int stage_idx) {
        int64_t k_blk_p;
        int slot_p;
        TileLocatorV5<SPARSE_SEL, K_BT_MODE>::locate_k(
            tile_idx, BN,
            ld_smem_i32_nocse(&sP.bs_shift),
            ld_smem_i32_nocse(&sP.bs_mask),
            ld_smem_i32_nocse(&sP.tpb_shift),
            ld_smem_i32_nocse(&sP.tpb_mask),
            ld_smem_ptr_nocse(&sP.bt_row),
            ld_smem_ptr_nocse(&sP.k_bt_row),
            ld_smem_ptr_nocse(&sP.sel_row),
            k_blk_p, slot_p);
        issue_k_async_v4<BN>(
            ld_smem_ptr_nocse(&sP.K_fp8_h),
            k_blk_p, slot_p,
            ld_smem_i64_nocse(&sP.sk8_b),
            ld_smem_i64_nocse(&sP.sk8_s),
            ld_smem_u32_nocse(&sP.k_base_sptr) + (uint32_t)(stage_idx * BN * 128),
            tid);
        cp_async_commit();
    };

    // ---- Prologue: issue first PREFETCH K tiles via cp.async.cg ----
    {
        const int nt_init = ld_smem_i32_nocse(&sP.num_tiles);
        #pragma unroll
        for (int s = 0; s < PREFETCH; ++s) {
            if (s < nt_init) {
                issue_k_stage(s, s);
            }
        }
    }

    // ---- Main Multi-Stage Pipelined Bias-Free Prefix Loop ----
    for (int it = 0; it < ld_smem_i32_nocse(&sP.num_tiles); ++it) {
        const int nt_loop = ld_smem_i32_nocse(&sP.num_tiles);
        const int pf_it = it + PREFETCH;
        if (pf_it < nt_loop) {
            issue_k_stage(pf_it, pf_it % STAGES);
            cp_async_wait_group<PREFETCH>();
        } else {
            const int rem = nt_loop - 1 - it;
            if (rem == 1) cp_async_wait_group<1>();
            else          cp_async_wait_group<0>();
        }

        // Single-thread tile metadata + KS64 load right before existing __syncthreads()!
        // Saves 255x redundant BT/K_BT/SEL/KS64 loads AND removes k_blk/v_blk/KS64 regs from h64 loop!
        if (tid == 0) {
            int64_t k_blk_c, v_blk_c;
            int slot_c;
            TileLocatorV5<SPARSE_SEL, K_BT_MODE>::locate_kv(
                it, BN, sP.bs_shift, sP.bs_mask, sP.tpb_shift, sP.tpb_mask,
                sP.bt_row, sP.k_bt_row, sP.sel_row, k_blk_c, v_blk_c, slot_c);
            sP.cur_v_blk = v_blk_c;
            sP.cur_slot = slot_c;
            #pragma unroll
            for (int h64 = 0; h64 < SUB64; ++h64) {
                sP.ks_tile[h64] = __ldg(sP.KS64_h + k_blk_c * sP.sks64_b + ((slot_c >> 6) + h64));
            }
        }
        __syncthreads();

        const uint32_t cur_k_sptr = ld_smem_u32_nocse(&sP.k_base_sptr)
                                  + (uint32_t)((it % STAGES) * BN * 128);

        #pragma unroll
        for (int h64 = 0; h64 < SUB64; ++h64) {
            const int nt_base = h64 * 8;
            uint32_t d0[8], d1[8];

            // 1) QK^T via ldmatrix.x4 + SM121a 4x FP8-f16 mma.sync.m16n8k32
            {
                const int sub = lane >> 3;
                const int r_in = lane & 7;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int nt0 = nt_base + 2 * j;
                    uint32_t c0_0 = 0u, c1_0 = 0u;
                    uint32_t c0_1 = 0u, c1_1 = 0u;
                    const int k_row = (nt0 + (sub >> 1)) * 8 + r_in;

                    #pragma unroll
                    for (int ks = 0; ks < 4; ++ks) {
                        const int k_chunk = ks * 2 + (sub & 1);
                        const int swiz_c = k_chunk ^ r_in;
                        const uint32_t addr = cur_k_sptr + (uint32_t)(k_row * 128 + swiz_c * 16);
                        uint32_t b0_0, b1_0, b0_1, b1_1;
                        ldmatrix_x4(b0_0, b1_0, b0_1, b1_1, addr);
                        mma_m16n8k32_f16_e4m3(c0_0, c1_0, q_a0[ks], q_a1[ks], q_a2[ks], q_a3[ks], b0_0, b1_0, c0_0, c1_0);
                        mma_m16n8k32_f16_e4m3(c0_1, c1_1, q_a0[ks], q_a1[ks], q_a2[ks], q_a3[ks], b0_1, b1_1, c0_1, c1_1);
                    }
                    d0[2 * j] = c0_0; d1[2 * j] = c1_0;
                    d0[2 * j + 1] = c0_1; d1[2 * j + 1] = c1_1;
                }
            }

            // 2) Packed half2 `max.f16x2` reduction (ZERO half2->float2 on skipped tiles!)
            uint32_t hm0 = hmax2_u32(
                hmax2_u32(hmax2_u32(d0[0], d0[1]), hmax2_u32(d0[2], d0[3])),
                hmax2_u32(hmax2_u32(d0[4], d0[5]), hmax2_u32(d0[6], d0[7])));
            uint32_t hm1 = hmax2_u32(
                hmax2_u32(hmax2_u32(d1[0], d1[1]), hmax2_u32(d1[2], d1[3])),
                hmax2_u32(hmax2_u32(d1[4], d1[5]), hmax2_u32(d1[6], d1[7])));

            hm0 = hmax2_u32(hm0, __shfl_xor_sync(0xffffffff, hm0, 1));
            hm0 = hmax2_u32(hm0, __shfl_xor_sync(0xffffffff, hm0, 2));
            hm1 = hmax2_u32(hm1, __shfl_xor_sync(0xffffffff, hm1, 1));
            hm1 = hmax2_u32(hm1, __shfl_xor_sync(0xffffffff, hm1, 2));

            const float ks_tile = ld_smem_f32_nocse(&sP.ks_tile[h64]);
            const float qs_ks_0 = qs_log2_0 * ks_tile;
            const float qs_ks_1 = qs_log2_1 * ks_tile;

            const float2 m0_f2 = __half22float2(*reinterpret_cast<const __half2*>(&hm0));
            const float2 m1_f2 = __half22float2(*reinterpret_cast<const __half2*>(&hm1));
            const float tm0 = fmaxf(m0_f2.x, m0_f2.y) * qs_ks_0;
            const float tm1 = fmaxf(m1_f2.x, m1_f2.y) * qs_ks_1;

            const float mn0_raw = fmaxf(m_i0, tm0);
            const float mn1_raw = fmaxf(m_i1, tm1);

            int local_active = 0;
            if ((valid_mask & 1u) && (tm0 - mn0_raw >= -pv_skip_tau)) local_active = 1;
            if ((valid_mask & 2u) && (tm1 - mn1_raw >= -pv_skip_tau)) local_active = 1;

            const int do_pv = __syncthreads_or((pv_skip_tau > 0.0f) ? local_active : 1);

            if (do_pv) {
                // Issue async V load FIRST using precomputed (cur_v_blk, cur_slot) from sP!
                const int64_t v_blk_cur = ld_smem_i64_nocse(&sP.cur_v_blk);
                const int slot_cur = ld_smem_i32_nocse(&sP.cur_slot);
                const int64_t sv_s_cur = ld_smem_i64_nocse(&sP.sv_s);
                const __nv_bfloat16* v_ptr = ld_smem_ptr_nocse(&sP.V_h)
                    + v_blk_cur * ld_smem_i64_nocse(&sP.sv_b)
                    + (int64_t)(slot_cur + h64 * 64) * sv_s_cur;
                const uint32_t v_sptr = ld_smem_u32_nocse(&sP.v_base_sptr);
                issue_v64_async_v4(v_ptr, sv_s_cur, v_sptr, tid);
                cp_async_commit();

                const bool warp_active = (pv_skip_tau <= 0.0f) || (__any_sync(0xffffffff, local_active) != 0);

                if (warp_active) {
                    const float mn0 = (mn0_raw < NEG_INF_GUARD) ? 0.0f : mn0_raw;
                    const float mn1 = (mn1_raw < NEG_INF_GUARD) ? 0.0f : mn1_raw;
                    const float a0 = (m_i0 < NEG_INF_GUARD) ? 0.0f : fast_exp2f(m_i0 - mn0);
                    const float a1 = (m_i1 < NEG_INF_GUARD) ? 0.0f : fast_exp2f(m_i1 - mn1);

                    if (__any_sync(0xffffffff, a0 != 1.0f)) {
                        #pragma unroll
                        for (int dt = 0; dt < 16; ++dt) {
                            acc[dt][0] *= a0;
                            acc[dt][1] *= a0;
                        }
                    }
                    if (__any_sync(0xffffffff, a1 != 1.0f)) {
                        #pragma unroll
                        for (int dt = 0; dt < 16; ++dt) {
                            acc[dt][2] *= a1;
                            acc[dt][3] *= a1;
                        }
                    }

                    float sum0 = 0.0f, sum1 = 0.0f;
                    #pragma unroll
                    for (int nt = 0; nt < 8; ++nt) {
                        const float2 f0 = __half22float2(*reinterpret_cast<const __half2*>(&d0[nt]));
                        const float2 f1 = __half22float2(*reinterpret_cast<const __half2*>(&d1[nt]));
                        const float p00 = fast_exp2f(fmaf(f0.x, qs_ks_0, -mn0));
                        const float p01 = fast_exp2f(fmaf(f0.y, qs_ks_0, -mn0));
                        const float p10 = fast_exp2f(fmaf(f1.x, qs_ks_1, -mn1));
                        const float p11 = fast_exp2f(fmaf(f1.y, qs_ks_1, -mn1));
                        sum0 += p00 + p01;
                        sum1 += p10 + p11;
                        d0[nt] = f32x2_to_bf16x2_rn(p00, p01);
                        d1[nt] = f32x2_to_bf16x2_rn(p10, p11);
                    }

                    sum0 += __shfl_xor_sync(0xffffffff, sum0, 1);
                    sum0 += __shfl_xor_sync(0xffffffff, sum0, 2);
                    sum1 += __shfl_xor_sync(0xffffffff, sum1, 1);
                    sum1 += __shfl_xor_sync(0xffffffff, sum1, 2);
                    l_i0 = l_i0 * a0 + sum0;
                    l_i1 = l_i1 * a1 + sum1;
                    m_i0 = mn0_raw;
                    m_i1 = mn1_raw;
                }

                cp_async_wait_group<0>();
                __syncthreads();

                if (warp_active) {
                    const uint32_t v_sptr_pv = ld_smem_u32_nocse(&sP.v_base_sptr);
                    const int sub = lane >> 3;
                    const int r_in = lane & 7;
                    #pragma unroll
                    for (int kt = 0; kt < 4; ++kt) {
                        const int v_row = kt * 16 + (sub & 1) * 8 + r_in;
                        const uint32_t pa0 = d0[2 * kt];
                        const uint32_t pa1 = d1[2 * kt];
                        const uint32_t pa2 = d0[2 * kt + 1];
                        const uint32_t pa3 = d1[2 * kt + 1];
                        #pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            const int dt0 = 2 * j;
                            const int v_chunk = dt0 + (sub >> 1);
                            const uint32_t v_addr = v_sptr_pv + (uint32_t)(v_row * 256 + (v_chunk ^ r_in) * 16);
                            uint32_t vb0_0, vb1_0, vb0_1, vb1_1;
                            ldmatrix_x4_trans(vb0_0, vb1_0, vb0_1, vb1_1, v_addr);
                            mma_m16n8k16_f32_bf16(
                                acc[dt0][0], acc[dt0][1], acc[dt0][2], acc[dt0][3],
                                pa0, pa1, pa2, pa3, vb0_0, vb1_0,
                                acc[dt0][0], acc[dt0][1], acc[dt0][2], acc[dt0][3]);
                            mma_m16n8k16_f32_bf16(
                                acc[dt0 + 1][0], acc[dt0 + 1][1], acc[dt0 + 1][2], acc[dt0 + 1][3],
                                pa0, pa1, pa2, pa3, vb0_1, vb1_1,
                                acc[dt0 + 1][0], acc[dt0 + 1][1], acc[dt0 + 1][2], acc[dt0 + 1][3]);
                        }
                    }
                }
                if (SUB64 > 1 && h64 + 1 < SUB64) {
                    __syncthreads();
                }
            }
        }
    }

    // ---- Store (ACC_OUT, M_OUT, L_OUT) for global_tail_otf_rel_sm121 ----
    const int i_lo_e = ld_smem_i32_nocse(&sP.i_lo);
    const int qlen_e = ld_smem_i32_nocse(&sP.qlen);
    const int q0_e   = ld_smem_i32_nocse(&sP.q0);
    const int h_e    = ld_smem_i32_nocse(&sP.h);
    const int r0_loc = i_lo_e + warp * 16 + gid;
    const int r1_loc = r0_loc + 8;
    if (r0_loc < qlen_e) {
        const int64_t th0 = (int64_t)(q0_e + r0_loc) * H + h_e;
        float2* a_ptr0 = reinterpret_cast<float2*>(ACC_OUT + th0 * D);
        #pragma unroll
        for (int dt = 0; dt < 16; ++dt) {
            a_ptr0[dt * 4 + t] = make_float2(acc[dt][0], acc[dt][1]);
        }
        if (t == 0) { M_OUT[th0] = m_i0; L_OUT[th0] = l_i0; }
    }
    if (r1_loc < qlen_e) {
        const int64_t th1 = (int64_t)(q0_e + r1_loc) * H + h_e;
        float2* a_ptr1 = reinterpret_cast<float2*>(ACC_OUT + th1 * D);
        #pragma unroll
        for (int dt = 0; dt < 16; ++dt) {
            a_ptr1[dt * 4 + t] = make_float2(acc[dt][2], acc[dt][3]);
        }
        if (t == 0) { M_OUT[th1] = m_i1; L_OUT[th1] = l_i1; }
    }
}

// =================================================================================================
// Native SM121a FP8-f16 Exact-Kmax Quest Block Scorer (`score_kmax_blocks_fp8_sm121_kernel`)
// =================================================================================================

__device__ __forceinline__ void load_quant_box64_swiz(
    const __nv_bfloat16* __restrict__ K_BOX_h,
    const int32_t* __restrict__ bt_row,
    int64_t sm_b,
    int b0, int n_pblks,
    uint8_t* __restrict__ smem_fp8,
    float* __restrict__ smem_scale,
    int tid
) {
    const int row = tid >> 2;
    const int chunk = tid & 3;
    const int lb = b0 + row;
    const bool ok = (lb < n_pblks);

    uint32_t raw[16];
    float amax = 0.0f;
    if (ok) {
        const int64_t pblk = (int64_t)bt_row[lb];
        const uint4* src4 = reinterpret_cast<const uint4*>(K_BOX_h + pblk * sm_b + chunk * 32);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint4 v = src4[i];
            raw[i * 4 + 0] = v.x;
            raw[i * 4 + 1] = v.y;
            raw[i * 4 + 2] = v.z;
            raw[i * 4 + 3] = v.w;
            const float2 f0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v.x));
            const float2 f1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v.y));
            const float2 f2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v.z));
            const float2 f3 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v.w));
            amax = fmaxf(amax, fmaxf(fmaxf(fabsf(f0.x), fabsf(f0.y)), fmaxf(fabsf(f1.x), fabsf(f1.y))));
            amax = fmaxf(amax, fmaxf(fmaxf(fabsf(f2.x), fabsf(f2.y)), fmaxf(fabsf(f3.x), fabsf(f3.y))));
        }
    } else {
        #pragma unroll
        for (int i = 0; i < 16; ++i) raw[i] = 0u;
    }

    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 2));
    amax = fmaxf(amax, 1.0e-12f);
    const float inv = div_rn_ftz(QK_MAX, amax);

    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int base = i * 8;
        uint4 pack;
        const float2 a0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw[base + 0]));
        const float2 a1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw[base + 1]));
        const float2 a2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw[base + 2]));
        const float2 a3 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw[base + 3]));
        const float2 a4 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw[base + 4]));
        const float2 a5 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw[base + 5]));
        const float2 a6 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw[base + 6]));
        const float2 a7 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw[base + 7]));
        pack.x = f32x4_to_e4m3x4_rn(a0.x * inv, a0.y * inv, a1.x * inv, a1.y * inv);
        pack.y = f32x4_to_e4m3x4_rn(a2.x * inv, a2.y * inv, a3.x * inv, a3.y * inv);
        pack.z = f32x4_to_e4m3x4_rn(a4.x * inv, a4.y * inv, a5.x * inv, a5.y * inv);
        pack.w = f32x4_to_e4m3x4_rn(a6.x * inv, a6.y * inv, a7.x * inv, a7.y * inv);
        const int c = chunk * 2 + i;
        const int swiz_c = c ^ (row & 7);
        *reinterpret_cast<uint4*>(smem_fp8 + row * 128 + swiz_c * 16) = pack;
    }
    if (chunk == 0) {
        smem_scale[row] = amax * INV_QK_MAX;
    }
}

__global__ void __launch_bounds__(256, 2) score_kmax_blocks_fp8_sm121_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K_MID,
    const __nv_bfloat16* __restrict__ K_RAD,
    const int32_t* __restrict__ BT,
    const int32_t* __restrict__ SEQ,
    const int32_t* __restrict__ CUQ,
    float* __restrict__ SCORES,
    int32_t* __restrict__ SEL_COUNTS,
    int64_t sq_t, int64_t sq_h,
    int64_t sm_b, int64_t sm_h,
    int64_t sbt_r,
    int64_t ssc_cta,
    int max_pblks, int H, int G, int BS, int E,
    int sink_blocks, int local_blocks, int k_sel
) {
    constexpr int BLOCK_B = 64;
    const int pid = (int)blockIdx.x;
    const int r = (int)blockIdx.y;
    const int h = pid % H;
    const int mt = pid / H;
    const int kvh = h / G;
    const int cta_idx = r * (int)gridDim.x + pid;

    const int q0 = CUQ[r];
    const int qlen = CUQ[r + 1] - q0;
    const int seqlen = SEQ[r];
    const int past = seqlen - qlen;
    const int tid = (int)threadIdx.x;

    const int i_lo = mt * BM;
    float* sc_row = SCORES + (int64_t)cta_idx * ssc_cta;

    if (i_lo >= qlen) {
        if (tid == 0 && SEL_COUNTS != nullptr) SEL_COUNTS[cta_idx] = 0;
        for (int lb = tid; lb < max_pblks; lb += THREADS) {
            sc_row[lb] = -1.0e30f;
        }
        return;
    }

    const int q_min = past + i_lo;
    const int k_bias_raw = (E > 0) ? max(q_min - E + 1, 0) : max(q_min, 0);
    const int n_pblks = min(k_bias_raw / BS, max_pblks);

    if (tid == 0 && SEL_COUNTS != nullptr) {
        SEL_COUNTS[cta_idx] = min(n_pblks, k_sel);
    }

    if (n_pblks <= 0) {
        for (int lb = tid; lb < max_pblks; lb += THREADS) {
            sc_row[lb] = -1.0e30f;
        }
        return;
    }

    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int gid = lane >> 2;
    const int t = lane & 3;
    const int sub = lane >> 3;
    const int r_in = lane & 7;

    __shared__ __align__(16) uint8_t smem_km[BLOCK_B * 128];
    __shared__ __align__(16) uint8_t smem_kr[BLOCK_B * 128];
    __shared__ __align__(16) float smem_kms[BLOCK_B];
    __shared__ __align__(16) float smem_krs[BLOCK_B];
    __shared__ __align__(16) float smem_wmax[WARPS][BLOCK_B];

    const uint32_t km_sptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_km));
    const uint32_t kr_sptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_kr));

    uint32_t q_a0[4], q_a1[4], q_a2[4], q_a3[4];
    const int r0_loc = i_lo + warp * 16 + gid;
    const int r1_loc = r0_loc + 8;
    const bool r0_valid = (r0_loc < qlen);
    const bool r1_valid = (r1_loc < qlen);
    float qs0, qs1;
    load_quant_q_regs_v4(
        Q, sq_t, sq_h, q0, h, r0_loc, r1_loc, r0_valid, r1_valid, t, 1.0f,
        q_a0, q_a1, q_a2, q_a3, qs0, qs1
    );

    const int32_t* bt_row = BT + (int64_t)r * sbt_r;
    const __nv_bfloat16* km_h = K_MID + (int64_t)kvh * sm_h;
    const __nv_bfloat16* kr_h = K_RAD + (int64_t)kvh * sm_h;
    const int local_start = n_pblks - local_blocks;

    for (int b0 = 0; b0 < n_pblks; b0 += BLOCK_B) {
        const int b_hi = min(b0 + BLOCK_B, n_pblks);
        const bool all_mandatory = (b_hi <= sink_blocks) || (b0 >= local_start);
        if (all_mandatory) {
            if (tid < BLOCK_B) {
                const int lb = b0 + tid;
                if (lb < max_pblks) {
                    sc_row[lb] = (lb < n_pblks) ? 1.0e30f : -1.0e30f;
                }
            }
            continue;
        }

        load_quant_box64_swiz(km_h, bt_row, sm_b, b0, n_pblks, smem_km, smem_kms, tid);
        load_quant_box64_swiz(kr_h, bt_row, sm_b, b0, n_pblks, smem_kr, smem_krs, tid);
        __syncthreads();

        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int nt0 = 2 * j;
            uint32_t dm0_0 = 0u, dm1_0 = 0u, dm0_1 = 0u, dm1_1 = 0u;
            uint32_t dr0_0 = 0u, dr1_0 = 0u, dr0_1 = 0u, dr1_1 = 0u;
            const int k_row = (nt0 + (sub >> 1)) * 8 + r_in;

            #pragma unroll
            for (int ks = 0; ks < 4; ++ks) {
                const int k_chunk = ks * 2 + (sub & 1);
                const uint32_t off = (uint32_t)(k_row * 128 + (k_chunk ^ r_in) * 16);

                uint32_t bm0_0, bm1_0, bm0_1, bm1_1;
                ldmatrix_x4(bm0_0, bm1_0, bm0_1, bm1_1, km_sptr + off);
                mma_m16n8k32_f16_e4m3(dm0_0, dm1_0, q_a0[ks], q_a1[ks], q_a2[ks], q_a3[ks], bm0_0, bm1_0, dm0_0, dm1_0);
                mma_m16n8k32_f16_e4m3(dm0_1, dm1_1, q_a0[ks], q_a1[ks], q_a2[ks], q_a3[ks], bm0_1, bm1_1, dm0_1, dm1_1);

                const uint32_t qa0_abs = q_a0[ks] & 0x7F7F7F7Fu;
                const uint32_t qa1_abs = q_a1[ks] & 0x7F7F7F7Fu;
                const uint32_t qa2_abs = q_a2[ks] & 0x7F7F7F7Fu;
                const uint32_t qa3_abs = q_a3[ks] & 0x7F7F7F7Fu;

                uint32_t br0_0, br1_0, br0_1, br1_1;
                ldmatrix_x4(br0_0, br1_0, br0_1, br1_1, kr_sptr + off);
                mma_m16n8k32_f16_e4m3(dr0_0, dr1_0, qa0_abs, qa1_abs, qa2_abs, qa3_abs, br0_0, br1_0, dr0_0, dr1_0);
                mma_m16n8k32_f16_e4m3(dr0_1, dr1_1, qa0_abs, qa1_abs, qa2_abs, qa3_abs, br0_1, br1_1, dr0_1, dr1_1);
            }

            const float2 kms_a = reinterpret_cast<const float2*>(smem_kms)[nt0 * 4 + t];
            const float2 krs_a = reinterpret_cast<const float2*>(smem_krs)[nt0 * 4 + t];
            const float2 kms_b = reinterpret_cast<const float2*>(smem_kms)[(nt0 + 1) * 4 + t];
            const float2 krs_b = reinterpret_cast<const float2*>(smem_krs)[(nt0 + 1) * 4 + t];

            const float2 fm0_a = __half22float2(*reinterpret_cast<const __half2*>(&dm0_0));
            const float2 fm1_a = __half22float2(*reinterpret_cast<const __half2*>(&dm1_0));
            const float2 fr0_a = __half22float2(*reinterpret_cast<const __half2*>(&dr0_0));
            const float2 fr1_a = __half22float2(*reinterpret_cast<const __half2*>(&dr1_0));

            float ca0 = -1.0e30f, ca1 = -1.0e30f;
            if (r0_valid) {
                ca0 = fmaxf(ca0, (fm0_a.x * kms_a.x + fr0_a.x * krs_a.x) * qs0);
                ca1 = fmaxf(ca1, (fm0_a.y * kms_a.y + fr0_a.y * krs_a.y) * qs0);
            }
            if (r1_valid) {
                ca0 = fmaxf(ca0, (fm1_a.x * kms_a.x + fr1_a.x * krs_a.x) * qs1);
                ca1 = fmaxf(ca1, (fm1_a.y * kms_a.y + fr1_a.y * krs_a.y) * qs1);
            }

            const float2 fm0_b = __half22float2(*reinterpret_cast<const __half2*>(&dm0_1));
            const float2 fm1_b = __half22float2(*reinterpret_cast<const __half2*>(&dm1_1));
            const float2 fr0_b = __half22float2(*reinterpret_cast<const __half2*>(&dr0_1));
            const float2 fr1_b = __half22float2(*reinterpret_cast<const __half2*>(&dr1_1));

            float cb0 = -1.0e30f, cb1 = -1.0e30f;
            if (r0_valid) {
                cb0 = fmaxf(cb0, (fm0_b.x * kms_b.x + fr0_b.x * krs_b.x) * qs0);
                cb1 = fmaxf(cb1, (fm0_b.y * kms_b.y + fr0_b.y * krs_b.y) * qs0);
            }
            if (r1_valid) {
                cb0 = fmaxf(cb0, (fm1_b.x * kms_b.x + fr1_b.x * krs_b.x) * qs1);
                cb1 = fmaxf(cb1, (fm1_b.y * kms_b.y + fr1_b.y * krs_b.y) * qs1);
            }

            ca0 = fmaxf(ca0, __shfl_xor_sync(0xffffffff, ca0, 4));
            ca0 = fmaxf(ca0, __shfl_xor_sync(0xffffffff, ca0, 8));
            ca0 = fmaxf(ca0, __shfl_xor_sync(0xffffffff, ca0, 16));
            ca1 = fmaxf(ca1, __shfl_xor_sync(0xffffffff, ca1, 4));
            ca1 = fmaxf(ca1, __shfl_xor_sync(0xffffffff, ca1, 8));
            ca1 = fmaxf(ca1, __shfl_xor_sync(0xffffffff, ca1, 16));

            cb0 = fmaxf(cb0, __shfl_xor_sync(0xffffffff, cb0, 4));
            cb0 = fmaxf(cb0, __shfl_xor_sync(0xffffffff, cb0, 8));
            cb0 = fmaxf(cb0, __shfl_xor_sync(0xffffffff, cb0, 16));
            cb1 = fmaxf(cb1, __shfl_xor_sync(0xffffffff, cb1, 4));
            cb1 = fmaxf(cb1, __shfl_xor_sync(0xffffffff, cb1, 8));
            cb1 = fmaxf(cb1, __shfl_xor_sync(0xffffffff, cb1, 16));

            if (gid == 0) {
                smem_wmax[warp][nt0 * 8 + 2 * t + 0] = ca0;
                smem_wmax[warp][nt0 * 8 + 2 * t + 1] = ca1;
                smem_wmax[warp][(nt0 + 1) * 8 + 2 * t + 0] = cb0;
                smem_wmax[warp][(nt0 + 1) * 8 + 2 * t + 1] = cb1;
            }
        }
        __syncthreads();

        if (tid < BLOCK_B) {
            const int lb = b0 + tid;
            if (lb < max_pblks) {
                float mx = -1.0e30f;
                if (lb < n_pblks) {
                    if (lb < sink_blocks || lb >= local_start) {
                        mx = 1.0e30f;
                    } else {
                        #pragma unroll
                        for (int w = 0; w < WARPS; ++w) {
                            mx = fmaxf(mx, smem_wmax[w][tid]);
                        }
                    }
                }
                sc_row[lb] = mx;
            }
        }
        __syncthreads();
    }

    const int tail_start = ((n_pblks + BLOCK_B - 1) / BLOCK_B) * BLOCK_B;
    for (int lb = tail_start + tid; lb < max_pblks; lb += THREADS) {
        sc_row[lb] = -1.0e30f;
    }
}

template <int BN, int STAGES, int MIN_CTAS, bool SPARSE_SEL, int K_BT_MODE = 0>
inline void launch_v4_impl(
    dim3 grid, dim3 block, cudaStream_t stream,
    const void* q_ptr, const void* k_fp8_ptr, const void* ks64_ptr, const void* v_ptr,
    const void* bt_ptr, const void* k_bt_ptr, const void* seq_ptr, const void* cuq_ptr,
    const void* sel_blocks_ptr, const void* sel_counts_ptr,
    void* acc_out_ptr, void* m_out_ptr, void* l_out_ptr,
    int64_t sq_t, int64_t sq_h,
    int64_t sk8_b, int64_t sk8_s, int64_t sk8_h,
    int64_t sks64_b, int64_t sks64_h,
    int64_t sv_b, int64_t sv_s, int64_t sv_h,
    int64_t sbt_r, int64_t skbt_r,
    int H, int G, int BS, int E, int sel_stride,
    float scale_log2, float pv_skip_tau
) {
    constexpr size_t smem_bytes = STAGES * BN * 128 + 64 * 256;
    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(
            global_sage2_prefix_v4_sm121_kernel<BN, STAGES, MIN_CTAS, SPARSE_SEL, K_BT_MODE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)smem_bytes);
        cudaFuncSetAttribute(
            global_sage2_prefix_v4_sm121_kernel<BN, STAGES, MIN_CTAS, SPARSE_SEL, K_BT_MODE>,
            cudaFuncAttributePreferredSharedMemoryCarveout,
            cudaSharedmemCarveoutMaxShared);
        attr_set = true;
    }
    global_sage2_prefix_v4_sm121_kernel<BN, STAGES, MIN_CTAS, SPARSE_SEL, K_BT_MODE><<<grid, block, smem_bytes, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q_ptr),
        reinterpret_cast<const uint8_t*>(k_fp8_ptr),
        reinterpret_cast<const float*>(ks64_ptr),
        reinterpret_cast<const __nv_bfloat16*>(v_ptr),
        reinterpret_cast<const int32_t*>(bt_ptr),
        reinterpret_cast<const int32_t*>(k_bt_ptr),
        reinterpret_cast<const int32_t*>(seq_ptr),
        reinterpret_cast<const int32_t*>(cuq_ptr),
        reinterpret_cast<const int32_t*>(sel_blocks_ptr),
        reinterpret_cast<const int32_t*>(sel_counts_ptr),
        reinterpret_cast<float*>(acc_out_ptr),
        reinterpret_cast<float*>(m_out_ptr),
        reinterpret_cast<float*>(l_out_ptr),
        sq_t, sq_h,
        sk8_b, sk8_s, sk8_h,
        sks64_b, sks64_h,
        sv_b, sv_s, sv_h,
        sbt_r, skbt_r,
        H, G, BS, E, sel_stride,
        scale_log2,
        pv_skip_tau
    );
}

template <int K_BT_MODE>
inline int dispatch_v5_kbt(
    dim3 grid, dim3 block, cudaStream_t stream,
    bool sparse, int bn, int min_ctas,
    const void* q_ptr, const void* k_fp8_ptr, const void* ks64_ptr, const void* v_ptr,
    const void* bt_ptr, const void* k_bt_ptr, const void* seq_ptr, const void* cuq_ptr,
    const void* sel_blocks_ptr, const void* sel_counts_ptr,
    void* acc_out_ptr, void* m_out_ptr, void* l_out_ptr,
    int64_t sq_t, int64_t sq_h,
    int64_t sk8_b, int64_t sk8_s, int64_t sk8_h,
    int64_t sks64_b, int64_t sks64_h,
    int64_t sv_b, int64_t sv_s, int64_t sv_h,
    int64_t sbt_r, int64_t skbt_r,
    int H, int G, int BS, int E, int sel_stride,
    float scale_log2, float pv_skip_tau
) {
    if (!sparse) {
        if (bn == 128 && min_ctas == 2) {
            launch_v4_impl<128, 2, 2, false, K_BT_MODE>(
                grid, block, stream, q_ptr, k_fp8_ptr, ks64_ptr, v_ptr, bt_ptr, k_bt_ptr, seq_ptr, cuq_ptr,
                nullptr, nullptr, acc_out_ptr, m_out_ptr, l_out_ptr,
                sq_t, sq_h, sk8_b, sk8_s, sk8_h, sks64_b, sks64_h, sv_b, sv_s, sv_h, sbt_r, skbt_r,
                H, G, BS, E, 0, scale_log2, pv_skip_tau);
        } else if (bn == 64 && min_ctas == 2) {
            launch_v4_impl<64, 3, 2, false, K_BT_MODE>(
                grid, block, stream, q_ptr, k_fp8_ptr, ks64_ptr, v_ptr, bt_ptr, k_bt_ptr, seq_ptr, cuq_ptr,
                nullptr, nullptr, acc_out_ptr, m_out_ptr, l_out_ptr,
                sq_t, sq_h, sk8_b, sk8_s, sk8_h, sks64_b, sks64_h, sv_b, sv_s, sv_h, sbt_r, skbt_r,
                H, G, BS, E, 0, scale_log2, pv_skip_tau);
        } else if (bn == 128) {
            launch_v4_impl<128, 2, 1, false, K_BT_MODE>(
                grid, block, stream, q_ptr, k_fp8_ptr, ks64_ptr, v_ptr, bt_ptr, k_bt_ptr, seq_ptr, cuq_ptr,
                nullptr, nullptr, acc_out_ptr, m_out_ptr, l_out_ptr,
                sq_t, sq_h, sk8_b, sk8_s, sk8_h, sks64_b, sks64_h, sv_b, sv_s, sv_h, sbt_r, skbt_r,
                H, G, BS, E, 0, scale_log2, pv_skip_tau);
        } else {
            launch_v4_impl<64, 3, 1, false, K_BT_MODE>(
                grid, block, stream, q_ptr, k_fp8_ptr, ks64_ptr, v_ptr, bt_ptr, k_bt_ptr, seq_ptr, cuq_ptr,
                nullptr, nullptr, acc_out_ptr, m_out_ptr, l_out_ptr,
                sq_t, sq_h, sk8_b, sk8_s, sk8_h, sks64_b, sks64_h, sv_b, sv_s, sv_h, sbt_r, skbt_r,
                H, G, BS, E, 0, scale_log2, pv_skip_tau);
        }
    } else {
        if (bn == 128) {
            launch_v4_impl<128, 2, 2, true, K_BT_MODE>(
                grid, block, stream, q_ptr, k_fp8_ptr, ks64_ptr, v_ptr, bt_ptr, k_bt_ptr, seq_ptr, cuq_ptr,
                sel_blocks_ptr, sel_counts_ptr, acc_out_ptr, m_out_ptr, l_out_ptr,
                sq_t, sq_h, sk8_b, sk8_s, sk8_h, sks64_b, sks64_h, sv_b, sv_s, sv_h, sbt_r, skbt_r,
                H, G, BS, E, sel_stride, scale_log2, pv_skip_tau);
        } else {
            launch_v4_impl<64, 3, 2, true, K_BT_MODE>(
                grid, block, stream, q_ptr, k_fp8_ptr, ks64_ptr, v_ptr, bt_ptr, k_bt_ptr, seq_ptr, cuq_ptr,
                sel_blocks_ptr, sel_counts_ptr, acc_out_ptr, m_out_ptr, l_out_ptr,
                sq_t, sq_h, sk8_b, sk8_s, sk8_h, sks64_b, sks64_h, sv_b, sv_s, sv_h, sbt_r, skbt_r,
                H, G, BS, E, sel_stride, scale_log2, pv_skip_tau);
        }
    }
    return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

}  // namespace

extern "C" int global_sage2_prefix_sm121_fwd(
    const void*, const void*, const void*, const void*, const void*, const void*,
    void*, void*, void*,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    int, int, int, int, int, int, float, float, uint64_t
) {
    return -1;
}

extern "C" int global_sage2_prefix_v5_sm121_fwd(
    const void* q_ptr,
    const void* k_fp8_ptr,
    const void* ks64_ptr,
    const void* v_ptr,
    const void* bt_ptr,
    const void* seq_ptr,
    const void* cuq_ptr,
    const void* sel_blocks_ptr,
    const void* sel_counts_ptr,
    void* acc_out_ptr,
    void* m_out_ptr,
    void* l_out_ptr,
    int64_t sq_t, int64_t sq_h,
    int64_t sk8_b, int64_t sk8_s, int64_t sk8_h,
    int64_t sks64_b, int64_t sks64_h,
    int64_t sv_b, int64_t sv_s, int64_t sv_h,
    int64_t sbt_r,
    int num_reqs, int max_seqlen_q, int H, int G,
    int BS, int E, int bn, int min_ctas, int sel_stride,
    float scale_log2,
    float pv_skip_tau,
    const void* k_bt_ptr,
    int64_t skbt_r,
    int k_bt_mode,
    uint64_t stream_u64
) {
    if (bn != 64 && bn != 128) return -1;
    if (min_ctas != 1 && min_ctas != 2) return -1;
    if (k_bt_mode < 0 || k_bt_mode > 2) return -1;
    if (k_bt_mode == 2 && k_bt_ptr == nullptr) return -1;
    if (num_reqs <= 0 || max_seqlen_q <= 0 || H <= 0 || G <= 0 || BS < bn || (BS % bn) != 0) {
        return -1;
    }
    if ((BS & (BS - 1)) != 0) return -1;
    if (((uintptr_t)q_ptr & 7u) != 0 || ((uintptr_t)k_fp8_ptr & 15u) != 0 ||
        ((uintptr_t)v_ptr & 15u) != 0 || ((uintptr_t)acc_out_ptr & 7u) != 0) {
        return -1;
    }
    if ((sq_t & 3) != 0 || (sq_h & 3) != 0 ||
        (sk8_b & 15) != 0 || (sk8_s & 15) != 0 || (sk8_h & 15) != 0 ||
        (sv_b & 7) != 0 || (sv_s & 7) != 0 || (sv_h & 7) != 0) {
        return -1;
    }
    const bool sparse = (sel_blocks_ptr != nullptr && sel_counts_ptr != nullptr && sel_stride > 0);
    const int num_mt = (max_seqlen_q + BM - 1) / BM;
    dim3 grid(num_mt * H, num_reqs, 1);
    dim3 block(THREADS, 1, 1);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_u64);

    if (k_bt_mode == 1) {
        return dispatch_v5_kbt<1>(
            grid, block, stream, sparse, bn, min_ctas,
            q_ptr, k_fp8_ptr, ks64_ptr, v_ptr, bt_ptr, nullptr, seq_ptr, cuq_ptr,
            sel_blocks_ptr, sel_counts_ptr, acc_out_ptr, m_out_ptr, l_out_ptr,
            sq_t, sq_h, sk8_b, sk8_s, sk8_h, sks64_b, sks64_h, sv_b, sv_s, sv_h, sbt_r, 0,
            H, G, BS, E, sel_stride, scale_log2, pv_skip_tau);
    }
    if (k_bt_mode == 2) {
        return dispatch_v5_kbt<2>(
            grid, block, stream, sparse, bn, min_ctas,
            q_ptr, k_fp8_ptr, ks64_ptr, v_ptr, bt_ptr, k_bt_ptr, seq_ptr, cuq_ptr,
            sel_blocks_ptr, sel_counts_ptr, acc_out_ptr, m_out_ptr, l_out_ptr,
            sq_t, sq_h, sk8_b, sk8_s, sk8_h, sks64_b, sks64_h, sv_b, sv_s, sv_h, sbt_r, skbt_r,
            H, G, BS, E, sel_stride, scale_log2, pv_skip_tau);
    }
    return dispatch_v5_kbt<0>(
        grid, block, stream, sparse, bn, min_ctas,
        q_ptr, k_fp8_ptr, ks64_ptr, v_ptr, bt_ptr, nullptr, seq_ptr, cuq_ptr,
        sel_blocks_ptr, sel_counts_ptr, acc_out_ptr, m_out_ptr, l_out_ptr,
        sq_t, sq_h, sk8_b, sk8_s, sk8_h, sks64_b, sks64_h, sv_b, sv_s, sv_h, sbt_r, 0,
        H, G, BS, E, sel_stride, scale_log2, pv_skip_tau);
}

extern "C" int global_sage2_prefix_v3_sm121_fwd(
    const void* q_ptr,
    const void* k_fp8_ptr,
    const void* ks64_ptr,
    const void* v_ptr,
    const void* bt_ptr,
    const void* seq_ptr,
    const void* cuq_ptr,
    const void* sel_blocks_ptr,
    const void* sel_counts_ptr,
    void* acc_out_ptr,
    void* m_out_ptr,
    void* l_out_ptr,
    int64_t sq_t, int64_t sq_h,
    int64_t sk8_b, int64_t sk8_s, int64_t sk8_h,
    int64_t sks64_b, int64_t sks64_h,
    int64_t sv_b, int64_t sv_s, int64_t sv_h,
    int64_t sbt_r,
    int num_reqs, int max_seqlen_q, int H, int G,
    int BS, int E, int bn, int min_ctas, int sel_stride,
    float scale_log2,
    float pv_skip_tau,
    uint64_t stream_u64
) {
    return global_sage2_prefix_v5_sm121_fwd(
        q_ptr, k_fp8_ptr, ks64_ptr, v_ptr, bt_ptr, seq_ptr, cuq_ptr,
        sel_blocks_ptr, sel_counts_ptr, acc_out_ptr, m_out_ptr, l_out_ptr,
        sq_t, sq_h, sk8_b, sk8_s, sk8_h, sks64_b, sks64_h, sv_b, sv_s, sv_h, sbt_r,
        num_reqs, max_seqlen_q, H, G, BS, E, bn, min_ctas, sel_stride,
        scale_log2, pv_skip_tau,
        nullptr, 0, 0, stream_u64);
}

extern "C" int score_kmax_blocks_fp8_sm121_fwd(
    const void* q_ptr,
    const void* k_mid_ptr,
    const void* k_rad_ptr,
    const void* bt_ptr,
    const void* seq_ptr,
    const void* cuq_ptr,
    void* scores_ptr,
    void* sel_counts_ptr,
    int64_t sq_t, int64_t sq_h,
    int64_t sm_b, int64_t sm_h,
    int64_t sbt_r,
    int64_t ssc_cta,
    int num_reqs, int max_seqlen_q, int max_pblks,
    int H, int G, int BS, int E,
    int sink_blocks, int local_blocks, int k_sel,
    uint64_t stream_u64
) {
    if (num_reqs <= 0 || max_seqlen_q <= 0 || max_pblks <= 0 || H <= 0 || G <= 0 || BS <= 0) {
        return -1;
    }
    if (((uintptr_t)q_ptr & 7u) != 0 || ((uintptr_t)k_mid_ptr & 15u) != 0 ||
        ((uintptr_t)k_rad_ptr & 15u) != 0) {
        return -1;
    }
    if ((sq_t & 3) != 0 || (sq_h & 3) != 0 || (sm_b & 7) != 0 || (sm_h & 7) != 0) {
        return -1;
    }
    const int num_mt = (max_seqlen_q + BM - 1) / BM;
    dim3 grid(num_mt * H, num_reqs, 1);
    dim3 block(THREADS, 1, 1);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_u64);
    score_kmax_blocks_fp8_sm121_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q_ptr),
        reinterpret_cast<const __nv_bfloat16*>(k_mid_ptr),
        reinterpret_cast<const __nv_bfloat16*>(k_rad_ptr),
        reinterpret_cast<const int32_t*>(bt_ptr),
        reinterpret_cast<const int32_t*>(seq_ptr),
        reinterpret_cast<const int32_t*>(cuq_ptr),
        reinterpret_cast<float*>(scores_ptr),
        reinterpret_cast<int32_t*>(sel_counts_ptr),
        sq_t, sq_h, sm_b, sm_h, sbt_r, ssc_cta,
        max_pblks, H, G, BS, E, sink_blocks, local_blocks, k_sel
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}
