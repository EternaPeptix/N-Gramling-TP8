// SPDX-License-Identifier: Apache-2.0
// =================================================================================================
// SM121a (NVIDIA DGX Spark GB10) Non-MoE 1M Cold & Warm Prefill Accelerator Sidecar
//   Library: /rdma/libpf_attn_nonmoe_v4_sm121.so
//
// Strictly prefill-only (decode M < 1024 & CUDA-graph capture paths stay 100% 0-ULP untouched).
//
// Exported C ABI Kernels:
//   1. quantize_k_shadow_v4_sm121_fwd:
//        Zero-sync GPU paged key_cache -> K_FP8 + KS64 + BF16 K_MID/K_RAD + pre-quantized E4M3
//        K_MID_FP8 / K_RAD_FP8 / K_BOX_SCALE (eliminates 384x redundant per-CTA BF16->FP8 box
//        quantization inside TopK scoring).
//   2. score_topk_sort_fp8_v4_sm121_fwd:
//        Single-launch 2-Q-head/CTA (GQA Hkv=1 shared!) SM121a FP8 Tensor Core
//        (mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16) Quest exact-Kmax block scorer +
//        in-SMEM streaming Top-K + Sink/Local merge + ascending int32 bitonic sort.
//        Cuts 5 kernel launches -> 1 launch, eliminates 100% of SCORES DRAM traffic, 0 spills!
//   3. kvq_prologue_pf_v4_sm121_fwd:
//        Wave-balanced SM121a 0-ULP prefill K/V 4-tap ShortConv + paged KV write + Q L2-clip
//        with automatic zero-arg sparse_prefill_discard write skip (eliminates 25.1M L2 write
//        collisions per 1M prompt and 62.5% wasted HQ lanes).
//   4. quant_fp8_rowwise_v4_sm121_fwd:
//        Multi-row vectorized 128-bit BF16 -> rowwise FP8 E4M3 ([-448, 448]) quantizer with
//        specialized 0-SMEM K=768 fast path for O_proj (wo_ud) and general K%128==0 path.
//
// Build on GB10 (MUST use -gencode arch=compute_121a,code=sm_121a, OMIT --use_fast_math):
//   /usr/local/cuda/bin/nvcc -O3 -std=c++17 -gencode arch=compute_121a,code=sm_121a \
//       --shared -Xcompiler -fPIC --ptxas-options=-v \
//       pf_attn_nonmoe_v4_sm121.cu -o /rdma/libpf_attn_nonmoe_v4_sm121.so
// =================================================================================================

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>

namespace {

static constexpr int BM_SCORE = 128;
static constexpr int D_HEAD = 128;
static constexpr int WARPS_SCORE = 8;
static constexpr int THREADS_SCORE = WARPS_SCORE * 32;  // 256
static constexpr int BLOCK_B = 64;
static constexpr int K_MID_CAP = 128;                   // supports k_need_mid <= 128 (TOPK <= 32,768)
static constexpr float QK_MAX = 22.0f;                  // 128 * 22 * 22 = 61,952 < 65,504 (0 FP16 overflow)
static constexpr float INV_QK_MAX = 1.0f / 22.0f;
static constexpr float FP8_E4M3_MAX = 448.0f;
static constexpr float INV_FP8_E4M3_MAX = 1.0f / 448.0f;

// =================================================================================================
// SM121a PTX Math, Conversion, Async Copy, ldmatrix, and Tensor Core MMA Primitives
// =================================================================================================

__device__ __forceinline__ float div_rn_ftz(float a, float b) {
    float r;
    asm("div.rn.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

__device__ __forceinline__ float rsqrt_approx_f32(float x) {
    float y;
    asm("rsqrt.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float bf16_to_f32_u16(uint16_t b) {
    float f;
    asm("cvt.f32.bf16 %0, %1;" : "=f"(f) : "h"(b));
    return f;
}

__device__ __forceinline__ uint16_t f32_to_bf16_u16(float f) {
    __nv_bfloat16 b = __float2bfloat16_rn(f);
    return *reinterpret_cast<const uint16_t*>(&b);
}

__device__ __forceinline__ uint32_t pack_bf16x2_u32(uint16_t a, uint16_t b) {
    return static_cast<uint32_t>(a) | (static_cast<uint32_t>(b) << 16);
}

__device__ __forceinline__ uint16_t f32x2_to_e4m3x2_rn(float lo, float hi) {
    uint16_t r;
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(r) : "f"(lo), "f"(hi));
    return r;
}

__device__ __forceinline__ uint32_t f32x4_to_e4m3x4_rn(float v0, float v1, float v2, float v3) {
    uint16_t lo = f32x2_to_e4m3x2_rn(v0, v1);
    uint16_t hi = f32x2_to_e4m3x2_rn(v2, v3);
    return static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
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

__device__ __forceinline__ void divmod_p2(
    int64_t val, int n_mod, int n_shift, int64_t& q, int64_t& r
) {
    if (n_shift >= 0) {
        q = val >> n_shift;
        r = val & ((1LL << n_shift) - 1);
    } else {
        q = val / n_mod;
        r = val % n_mod;
    }
}

// Order-preserving float -> uint32 key + tie-break on smaller logical block index `lb`
// (strictly > 0ULL for all finite floats >= -1e35f so 0ULL is reserved for empty slots).
__device__ __forceinline__ uint64_t pack_score_idx_u64(float score, int lb) {
    uint32_t u = __float_as_uint(score);
    u = (u & 0x80000000u) ? (~u) : (u | 0x80000000u);
    if (u == 0u) u = 1u;
    return (static_cast<uint64_t>(u) << 32) | static_cast<uint64_t>(0xFFFFFFFFu - static_cast<uint32_t>(lb));
}

__device__ __forceinline__ int unpack_idx_from_u64(uint64_t key) {
    return static_cast<int>(0xFFFFFFFFu - static_cast<uint32_t>(key & 0xFFFFFFFFULL));
}

// =================================================================================================
// KERNEL 1: Sync-Free Shadow-K Quantizer + Pre-Quantized FP8 Bounding Boxes
//   128 threads/CTA (4 warps), 16.05 KiB SMEM -> up to 6 CTAs/SM, ~32 regs/thread, 0 spills!
// =================================================================================================

__global__ void __launch_bounds__(32, 1) update_bt_state_v4_kernel(
    const int32_t* __restrict__ BT_ROW,
    int32_t* __restrict__ BT_STATE,
    int end_idx
) {
    if (threadIdx.x == 0) {
        BT_STATE[0] = BT_ROW[0];
        BT_STATE[1] = BT_ROW[end_idx];
    }
}

template <bool CHECK_CONTIG, bool COMPUTE_KMAX, bool COMPACT_OUT>
__global__ void __launch_bounds__(128, 4) quantize_k_shadow_v4_kernel(
    const __nv_bfloat16* __restrict__ K_IN,
    uint8_t* __restrict__ K_FP8_OUT,
    float* __restrict__ KS64_OUT,
    __nv_bfloat16* __restrict__ K_MID_BF16_OUT,
    __nv_bfloat16* __restrict__ K_RAD_BF16_OUT,
    uint8_t* __restrict__ K_MID_FP8_OUT,
    uint8_t* __restrict__ K_RAD_FP8_OUT,
    float2* __restrict__ K_BOX_SC_OUT,
    const int32_t* __restrict__ BT_ROW,
    const int32_t* __restrict__ BT_STATE,
    int64_t sk_b, int64_t sk_s, int64_t sk_h,
    int64_t sk8_b, int64_t sk8_s, int64_t sk8_h,
    int64_t sks_b, int64_t sks_h,
    int64_t smid_b, int64_t smid_h,
    int64_t sbox8_b, int64_t sbox8_h,
    int64_t sboxsc_b, int64_t sboxsc_h,
    int nblks, int start_blk, int prev_idx,
    int BS
) {
    const int b_idx = static_cast<int>(blockIdx.x);
    const int kvh = static_cast<int>(blockIdx.y);
    if (b_idx >= nblks) return;

    if constexpr (CHECK_CONTIG) {
        if (b_idx < start_blk) {
            if (BT_ROW[0] == BT_STATE[0] && BT_ROW[prev_idx] == BT_STATE[1]) {
                return;
            }
        }
    }

    const int64_t phys_blk = static_cast<int64_t>(BT_ROW[b_idx]);
    const int64_t out_blk = COMPACT_OUT ? static_cast<int64_t>(b_idx) : phys_blk;
    const int tid = static_cast<int>(threadIdx.x);  // 0..127
    const int warp = tid >> 5;                      // 0..3
    const int lane = tid & 31;

    __shared__ __align__(16) __nv_bfloat16 smem_k[64 * D_HEAD];
    __shared__ float smem_wmax[4];

    float b_min = 1.0e30f;
    float b_max = -1.0e30f;

    const int n_sub = BS >> 6;  // BS / 64
    for (int g = 0; g < n_sub; ++g) {
        float local_amax = 0.0f;
        const __nv_bfloat16* k_sub_base = K_IN + phys_blk * sk_b + static_cast<int64_t>(g * 64) * sk_s + kvh * sk_h;

        // 128 threads x 8 uint4 (8 BF16 each) = 64 x 128 BF16 coalesced load
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int idx = i * 128 + tid;
            const int r = idx >> 4;
            const int c8 = (idx & 15) * 8;
            const uint4 v = *reinterpret_cast<const uint4*>(k_sub_base + static_cast<int64_t>(r) * sk_s + c8);
            *reinterpret_cast<uint4*>(smem_k + r * D_HEAD + c8) = v;

            const float2 f0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v.x));
            const float2 f1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v.y));
            const float2 f2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v.z));
            const float2 f3 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v.w));
            local_amax = fmaxf(local_amax, fmaxf(fmaxf(fabsf(f0.x), fabsf(f0.y)), fmaxf(fabsf(f1.x), fabsf(f1.y))));
            local_amax = fmaxf(local_amax, fmaxf(fmaxf(fabsf(f2.x), fabsf(f2.y)), fmaxf(fabsf(f3.x), fabsf(f3.y))));
        }

        local_amax = fmaxf(local_amax, __shfl_xor_sync(0xffffffffu, local_amax, 16));
        local_amax = fmaxf(local_amax, __shfl_xor_sync(0xffffffffu, local_amax, 8));
        local_amax = fmaxf(local_amax, __shfl_xor_sync(0xffffffffu, local_amax, 4));
        local_amax = fmaxf(local_amax, __shfl_xor_sync(0xffffffffu, local_amax, 2));
        local_amax = fmaxf(local_amax, __shfl_xor_sync(0xffffffffu, local_amax, 1));
        if (lane == 0) smem_wmax[warp] = local_amax;
        __syncthreads();

        const float amax64 = fmaxf(
            fmaxf(fmaxf(smem_wmax[0], smem_wmax[1]), fmaxf(smem_wmax[2], smem_wmax[3])),
            1.0e-12f
        );
        const float inv64 = div_rn_ftz(QK_MAX, amax64);
        if (tid == 0) {
            KS64_OUT[out_blk * sks_b + kvh * sks_h + g] = amax64 * INV_QK_MAX;
        }

        // Quantize & store [64, 128] FP8 (128 threads x 4 uint4 = 8,192 B coalesced)
        uint8_t* k8_sub_base = K_FP8_OUT + out_blk * sk8_b + static_cast<int64_t>(g * 64) * sk8_s + kvh * sk8_h;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int idx = i * 128 + tid;
            const int r = idx >> 3;
            const int c16 = (idx & 7) * 16;
            const uint4 v0 = *reinterpret_cast<const uint4*>(smem_k + r * D_HEAD + c16);
            const uint4 v1 = *reinterpret_cast<const uint4*>(smem_k + r * D_HEAD + c16 + 8);
            const float2 a0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v0.x));
            const float2 a1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v0.y));
            const float2 a2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v0.z));
            const float2 a3 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v0.w));
            const float2 a4 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v1.x));
            const float2 a5 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v1.y));
            const float2 a6 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v1.z));
            const float2 a7 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v1.w));
            uint4 pack;
            pack.x = f32x4_to_e4m3x4_rn(a0.x * inv64, a0.y * inv64, a1.x * inv64, a1.y * inv64);
            pack.y = f32x4_to_e4m3x4_rn(a2.x * inv64, a2.y * inv64, a3.x * inv64, a3.y * inv64);
            pack.z = f32x4_to_e4m3x4_rn(a4.x * inv64, a4.y * inv64, a5.x * inv64, a5.y * inv64);
            pack.w = f32x4_to_e4m3x4_rn(a6.x * inv64, a6.y * inv64, a7.x * inv64, a7.y * inv64);
            *reinterpret_cast<uint4*>(k8_sub_base + static_cast<int64_t>(r) * sk8_s + c16) = pack;
        }

        if constexpr (COMPUTE_KMAX) {
            #pragma unroll 8
            for (int r = 0; r < 64; ++r) {
                const float val = __bfloat162float(smem_k[r * D_HEAD + tid]);
                b_min = fminf(b_min, val);
                b_max = fmaxf(b_max, val);
            }
        }
        __syncthreads();
    }

    if constexpr (COMPUTE_KMAX) {
        // Round through BF16 first so pre-quantized FP8 boxes are 0-ULP identical to v5's BF16 boxes!
        const __nv_bfloat16 mid_bf = __float2bfloat16_rn(0.5f * (b_max + b_min));
        const __nv_bfloat16 rad_bf = __float2bfloat16_rn(0.5f * (b_max - b_min));
        if (K_MID_BF16_OUT != nullptr && K_RAD_BF16_OUT != nullptr) {
            K_MID_BF16_OUT[out_blk * smid_b + kvh * smid_h + tid] = mid_bf;
            K_RAD_BF16_OUT[out_blk * smid_b + kvh * smid_h + tid] = rad_bf;
        }
        if (K_MID_FP8_OUT != nullptr && K_RAD_FP8_OUT != nullptr && K_BOX_SC_OUT != nullptr) {
            const float mf = __bfloat162float(mid_bf);
            const float rf = __bfloat162float(rad_bf);
            float wm = fabsf(mf);
            float wr = fabsf(rf);
            wm = fmaxf(wm, __shfl_xor_sync(0xffffffffu, wm, 16));
            wm = fmaxf(wm, __shfl_xor_sync(0xffffffffu, wm, 8));
            wm = fmaxf(wm, __shfl_xor_sync(0xffffffffu, wm, 4));
            wm = fmaxf(wm, __shfl_xor_sync(0xffffffffu, wm, 2));
            wm = fmaxf(wm, __shfl_xor_sync(0xffffffffu, wm, 1));
            wr = fmaxf(wr, __shfl_xor_sync(0xffffffffu, wr, 16));
            wr = fmaxf(wr, __shfl_xor_sync(0xffffffffu, wr, 8));
            wr = fmaxf(wr, __shfl_xor_sync(0xffffffffu, wr, 4));
            wr = fmaxf(wr, __shfl_xor_sync(0xffffffffu, wr, 2));
            wr = fmaxf(wr, __shfl_xor_sync(0xffffffffu, wr, 1));

            float* smem_f = reinterpret_cast<float*>(smem_k);
            smem_f[tid] = mf;
            smem_f[128 + tid] = rf;
            if (lane == 0) {
                smem_f[256 + warp] = wm;
                smem_f[260 + warp] = wr;
            }
            __syncthreads();

            const float amax_m = fmaxf(
                fmaxf(fmaxf(smem_f[256], smem_f[257]), fmaxf(smem_f[258], smem_f[259])),
                1.0e-12f
            );
            const float amax_r = fmaxf(
                fmaxf(fmaxf(smem_f[260], smem_f[261]), fmaxf(smem_f[262], smem_f[263])),
                1.0e-12f
            );
            const float inv_m = div_rn_ftz(QK_MAX, amax_m);
            const float inv_r = div_rn_ftz(QK_MAX, amax_r);

            if (tid < 8) {
                const int c16 = tid * 16;
                uint4 pm, pr;
                pm.x = f32x4_to_e4m3x4_rn(smem_f[c16 + 0] * inv_m, smem_f[c16 + 1] * inv_m, smem_f[c16 + 2] * inv_m, smem_f[c16 + 3] * inv_m);
                pm.y = f32x4_to_e4m3x4_rn(smem_f[c16 + 4] * inv_m, smem_f[c16 + 5] * inv_m, smem_f[c16 + 6] * inv_m, smem_f[c16 + 7] * inv_m);
                pm.z = f32x4_to_e4m3x4_rn(smem_f[c16 + 8] * inv_m, smem_f[c16 + 9] * inv_m, smem_f[c16 + 10] * inv_m, smem_f[c16 + 11] * inv_m);
                pm.w = f32x4_to_e4m3x4_rn(smem_f[c16 + 12] * inv_m, smem_f[c16 + 13] * inv_m, smem_f[c16 + 14] * inv_m, smem_f[c16 + 15] * inv_m);
                *reinterpret_cast<uint4*>(K_MID_FP8_OUT + out_blk * sbox8_b + kvh * sbox8_h + c16) = pm;

                const float* rf_base = smem_f + 128 + c16;
                pr.x = f32x4_to_e4m3x4_rn(rf_base[0] * inv_r, rf_base[1] * inv_r, rf_base[2] * inv_r, rf_base[3] * inv_r);
                pr.y = f32x4_to_e4m3x4_rn(rf_base[4] * inv_r, rf_base[5] * inv_r, rf_base[6] * inv_r, rf_base[7] * inv_r);
                pr.z = f32x4_to_e4m3x4_rn(rf_base[8] * inv_r, rf_base[9] * inv_r, rf_base[10] * inv_r, rf_base[11] * inv_r);
                pr.w = f32x4_to_e4m3x4_rn(rf_base[12] * inv_r, rf_base[13] * inv_r, rf_base[14] * inv_r, rf_base[15] * inv_r);
                *reinterpret_cast<uint4*>(K_RAD_FP8_OUT + out_blk * sbox8_b + kvh * sbox8_h + c16) = pr;
            }
            if (tid == 0) {
                K_BOX_SC_OUT[out_blk * sboxsc_b + kvh * sboxsc_h] = make_float2(
                    amax_m * INV_QK_MAX, amax_r * INV_QK_MAX
                );
            }
        }
    }
}

// =================================================================================================
// KERNEL 2: Single-Launch Multi-Head Pre-Quantized Quest Scorer + In-SMEM Top-K + Bitonic Sort
//   256 threads/CTA, 24.53 KiB SMEM -> 2-3 CTAs/SM, ~76 regs/thread, 0 spills!
// =================================================================================================

__device__ __forceinline__ void load_quant_q_row_regs(
    const __nv_bfloat16* __restrict__ row_ptr,
    bool valid,
    int t,
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
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
    amax = fmaxf(amax, 1.0e-12f);
    const float inv = div_rn_ftz(QK_MAX, amax);
    qs_out = amax * INV_QK_MAX;

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

template <int H_GROUP>
__device__ __forceinline__ void bitonic_sort_desc_256(
    uint64_t smem_top[H_GROUP][256],
    int tid
) {
    const int u = (H_GROUP == 2) ? (tid >> 7) : 0;
    const int idx = tid & 127;
    const bool active = (H_GROUP == 2) ? true : (tid < 128);

    for (int k = 2; k <= 256; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            if (active) {
                const int low = idx & (j - 1);
                const int i = ((idx - low) << 1) + low;
                const int ij = i ^ j;
                const bool desc_dir = ((i & k) == 0);
                const uint64_t a = smem_top[u][i];
                const uint64_t b = smem_top[u][ij];
                if (desc_dir ? (a < b) : (a > b)) {
                    smem_top[u][i] = b;
                    smem_top[u][ij] = a;
                }
            }
            __syncthreads();
        }
    }
}

template <int H_GROUP>
__device__ __forceinline__ void bitonic_sort_asc_i32_128(
    int32_t smem_idx[H_GROUP][128],
    int tid
) {
    const int u = (H_GROUP == 2) ? (tid >> 6) : 0;
    const int idx = tid & 63;
    const bool active = (H_GROUP == 2) ? (tid < 128) : (tid < 64);

    for (int k = 2; k <= 128; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            if (active) {
                const int low = idx & (j - 1);
                const int i = ((idx - low) << 1) + low;
                const int ij = i ^ j;
                const bool asc_dir = ((i & k) == 0);
                const int32_t a = smem_idx[u][i];
                const int32_t b = smem_idx[u][ij];
                if (asc_dir ? (a > b) : (a < b)) {
                    smem_idx[u][i] = b;
                    smem_idx[u][ij] = a;
                }
            }
            __syncthreads();
        }
    }
}

template <int H_GROUP, int K_BT_MODE>
__global__ void __launch_bounds__(256, 2) score_topk_sort_fp8_v4_sm121_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const uint8_t* __restrict__ K_MID_FP8,
    const uint8_t* __restrict__ K_RAD_FP8,
    const float2* __restrict__ K_BOX_SC,
    const int32_t* __restrict__ BT,
    const int32_t* __restrict__ SEQ,
    const int32_t* __restrict__ CUQ,
    int32_t* __restrict__ SEL_BLOCKS,
    int32_t* __restrict__ SEL_COUNTS,
    int64_t sq_t, int64_t sq_h,
    int64_t sbox8_b, int64_t sbox8_h,
    int64_t sboxsc_b, int64_t sboxsc_h,
    int64_t sbt_r,
    int64_t ssel_row,
    int max_pblks, int H, int G, int BS, int E,
    int sink_blocks, int local_blocks, int k_sel
) {
    static_assert(H_GROUP == 1 || H_GROUP == 2, "H_GROUP must be 1 or 2");
    const int pid = static_cast<int>(blockIdx.x);
    const int r = static_cast<int>(blockIdx.y);
    const int h_groups = H / H_GROUP;
    const int hg = pid % h_groups;
    const int mt = pid / h_groups;
    const int h0 = hg * H_GROUP;
    const int kvh = h0 / G;
    const int n_mt = static_cast<int>(gridDim.x) / h_groups;
    const int base_cta_idx = r * (n_mt * H) + mt * H + h0;

    const int q0 = CUQ[r];
    const int qlen = CUQ[r + 1] - q0;
    const int seqlen = SEQ[r];
    const int past = seqlen - qlen;
    const int tid = static_cast<int>(threadIdx.x);

    const int i_lo = mt * BM_SCORE;
    if (i_lo >= qlen) {
        if (tid < H_GROUP) {
            SEL_COUNTS[base_cta_idx + tid] = 0;
        }
        #pragma unroll
        for (int u = 0; u < H_GROUP; ++u) {
            int32_t* dst = SEL_BLOCKS + static_cast<int64_t>(base_cta_idx + u) * ssel_row;
            for (int c = tid; c < k_sel; c += THREADS_SCORE) dst[c] = 0;
        }
        return;
    }

    const int q_min = past + i_lo;
    const int k_bias_raw = (E > 0) ? max(q_min - E + 1, 0) : max(q_min, 0);
    const int bs_shift = __ffs(BS) - 1;
    const int n_pblks = min(k_bias_raw >> bs_shift, max_pblks);
    const int n_sel_out = min(n_pblks, k_sel);

    if (tid < H_GROUP) {
        SEL_COUNTS[base_cta_idx + tid] = n_sel_out;
    }

    // Fast 0-GEMM exit when ALL prefix blocks fit inside k_sel (or n_pblks <= 0)!
    if (n_pblks <= k_sel) {
        #pragma unroll
        for (int u = 0; u < H_GROUP; ++u) {
            int32_t* dst = SEL_BLOCKS + static_cast<int64_t>(base_cta_idx + u) * ssel_row;
            for (int c = tid; c < k_sel; c += THREADS_SCORE) {
                dst[c] = (c < n_pblks) ? c : 0;
            }
        }
        return;
    }

    // Fast 0-GEMM exit if mandatory sink + local already fill k_sel!
    const int s_cnt = min(sink_blocks, n_pblks);
    const int local_start = max(s_cnt, n_pblks - local_blocks);
    const int l_cnt = n_pblks - local_start;
    const int k_need_mid = k_sel - (s_cnt + l_cnt);
    if (k_need_mid <= 0) {
        #pragma unroll
        for (int u = 0; u < H_GROUP; ++u) {
            int32_t* dst = SEL_BLOCKS + static_cast<int64_t>(base_cta_idx + u) * ssel_row;
            for (int c = tid; c < k_sel; c += THREADS_SCORE) {
                dst[c] = (c < s_cnt) ? c : (local_start + (c - s_cnt));
            }
        }
        return;
    }

    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int gid = lane >> 2;
    const int t = lane & 3;
    const int sub = lane >> 3;
    const int r_in = lane & 7;

    // 24.53 KiB static SMEM -> 2-3 CTAs/SM on SM121a (99 KiB/SM)!
    __shared__ __align__(16) uint8_t smem_km[BLOCK_B * D_HEAD];
    __shared__ __align__(16) uint8_t smem_kr[BLOCK_B * D_HEAD];
    __shared__ __align__(16) float2 smem_ksc[BLOCK_B];
    __shared__ __align__(16) float smem_wmax[H_GROUP][WARPS_SCORE][BLOCK_B];
    __shared__ __align__(16) uint64_t smem_top[H_GROUP][256];
    __shared__ uint64_t smem_kth[H_GROUP];
    __shared__ int smem_beat[H_GROUP];

    const uint32_t km_sptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_km));
    const uint32_t kr_sptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_kr));

    // Initialize streaming Top-128 state
    #pragma unroll
    for (int u = 0; u < H_GROUP; ++u) {
        smem_top[u][tid] = 0ULL;
    }
    if (tid < H_GROUP) {
        smem_kth[tid] = 0ULL;
    }

    // Quantize Q for H_GROUP heads directly into registers (18 regs/head, 0 SMEM!)
    uint32_t q0_a0[4], q0_a1[4], q0_a2[4], q0_a3[4];
    float qs0_0, qs0_1;
    uint32_t q1_a0[4], q1_a1[4], q1_a2[4], q1_a3[4];
    float qs1_0 = 0.0f, qs1_1 = 0.0f;

    const int r0_loc = i_lo + warp * 16 + gid;
    const int r1_loc = r0_loc + 8;
    const bool r0_valid = (r0_loc < qlen);
    const bool r1_valid = (r1_loc < qlen);

    load_quant_q_row_regs(
        Q + static_cast<int64_t>(q0 + r0_loc) * sq_t + static_cast<int64_t>(h0) * sq_h,
        r0_valid, t, q0_a0, q0_a2, qs0_0
    );
    load_quant_q_row_regs(
        Q + static_cast<int64_t>(q0 + r1_loc) * sq_t + static_cast<int64_t>(h0) * sq_h,
        r1_valid, t, q0_a1, q0_a3, qs0_1
    );
    if constexpr (H_GROUP == 2) {
        load_quant_q_row_regs(
            Q + static_cast<int64_t>(q0 + r0_loc) * sq_t + static_cast<int64_t>(h0 + 1) * sq_h,
            r0_valid, t, q1_a0, q1_a2, qs1_0
        );
        load_quant_q_row_regs(
            Q + static_cast<int64_t>(q0 + r1_loc) * sq_t + static_cast<int64_t>(h0 + 1) * sq_h,
            r1_valid, t, q1_a1, q1_a3, qs1_1
        );
    } else {
        (void)q1_a0; (void)q1_a1; (void)q1_a2; (void)q1_a3;
        (void)qs1_0; (void)qs1_1;
    }

    const int32_t* bt_row = BT + static_cast<int64_t>(r) * sbt_r;
    const uint8_t* km_h = K_MID_FP8 + static_cast<int64_t>(kvh) * sbox8_h;
    const uint8_t* kr_h = K_RAD_FP8 + static_cast<int64_t>(kvh) * sbox8_h;
    const float2* ksc_h = K_BOX_SC + static_cast<int64_t>(kvh) * sboxsc_h;

    const int b0_begin = (s_cnt / BLOCK_B) * BLOCK_B;
    const int b0_end = ((local_start + BLOCK_B - 1) / BLOCK_B) * BLOCK_B;

    __syncthreads();

    for (int b0 = b0_begin; b0 < b0_end; b0 += BLOCK_B) {
        // Async XOR-swizzled load of pre-quantized [64, 128] FP8 K_MID & K_RAD (0 quant math!)
        {
            const int row = tid >> 2;
            const int chunk = tid & 3;
            const int lb = b0 + row;
            const bool ok = (lb >= s_cnt) && (lb < local_start);
            int64_t pblk = 0;
            if (ok) {
                pblk = (K_BT_MODE == 1) ? static_cast<int64_t>(lb) : static_cast<int64_t>(bt_row[lb]);
            }
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                const int c = chunk * 2 + i;
                const int swiz_c = c ^ (row & 7);
                const uint32_t off = static_cast<uint32_t>(row * D_HEAD + swiz_c * 16);
                if (ok) {
                    cp_async_cg_16(km_sptr + off, km_h + pblk * sbox8_b + c * 16);
                    cp_async_cg_16(kr_sptr + off, kr_h + pblk * sbox8_b + c * 16);
                } else {
                    *reinterpret_cast<uint4*>(smem_km + off) = make_uint4(0u, 0u, 0u, 0u);
                    *reinterpret_cast<uint4*>(smem_kr + off) = make_uint4(0u, 0u, 0u, 0u);
                }
            }
            if (chunk == 0) {
                smem_ksc[row] = ok ? ksc_h[pblk * sboxsc_b] : make_float2(0.0f, 0.0f);
            }
            if (tid < H_GROUP) {
                smem_beat[tid] = 0;
            }
        }
        cp_async_commit();
        cp_async_wait_group<0>();
        __syncthreads();

        // Shared ldmatrix_x4 feeding FP8-f16 MMAs for H_GROUP heads simultaneously!
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int nt0 = 2 * j;
            uint32_t dm0_0 = 0u, dm1_0 = 0u, dm0_1 = 0u, dm1_1 = 0u;
            uint32_t dr0_0 = 0u, dr1_0 = 0u, dr0_1 = 0u, dr1_1 = 0u;
            uint32_t em0_0 = 0u, em1_0 = 0u, em0_1 = 0u, em1_1 = 0u;
            uint32_t er0_0 = 0u, er1_0 = 0u, er0_1 = 0u, er1_1 = 0u;
            const int k_row = (nt0 + (sub >> 1)) * 8 + r_in;

            #pragma unroll
            for (int ks = 0; ks < 4; ++ks) {
                const int k_chunk = ks * 2 + (sub & 1);
                const uint32_t off = static_cast<uint32_t>(k_row * D_HEAD + (k_chunk ^ r_in) * 16);

                uint32_t bm0_0, bm1_0, bm0_1, bm1_1;
                ldmatrix_x4(bm0_0, bm1_0, bm0_1, bm1_1, km_sptr + off);
                mma_m16n8k32_f16_e4m3(dm0_0, dm1_0, q0_a0[ks], q0_a1[ks], q0_a2[ks], q0_a3[ks], bm0_0, bm1_0, dm0_0, dm1_0);
                mma_m16n8k32_f16_e4m3(dm0_1, dm1_1, q0_a0[ks], q0_a1[ks], q0_a2[ks], q0_a3[ks], bm0_1, bm1_1, dm0_1, dm1_1);
                if constexpr (H_GROUP == 2) {
                    mma_m16n8k32_f16_e4m3(em0_0, em1_0, q1_a0[ks], q1_a1[ks], q1_a2[ks], q1_a3[ks], bm0_0, bm1_0, em0_0, em1_0);
                    mma_m16n8k32_f16_e4m3(em0_1, em1_1, q1_a0[ks], q1_a1[ks], q1_a2[ks], q1_a3[ks], bm0_1, bm1_1, em0_1, em1_1);
                }

                uint32_t br0_0, br1_0, br0_1, br1_1;
                ldmatrix_x4(br0_0, br1_0, br0_1, br1_1, kr_sptr + off);
                mma_m16n8k32_f16_e4m3(
                    dr0_0, dr1_0,
                    q0_a0[ks] & 0x7F7F7F7Fu, q0_a1[ks] & 0x7F7F7F7Fu,
                    q0_a2[ks] & 0x7F7F7F7Fu, q0_a3[ks] & 0x7F7F7F7Fu,
                    br0_0, br1_0, dr0_0, dr1_0
                );
                mma_m16n8k32_f16_e4m3(
                    dr0_1, dr1_1,
                    q0_a0[ks] & 0x7F7F7F7Fu, q0_a1[ks] & 0x7F7F7F7Fu,
                    q0_a2[ks] & 0x7F7F7F7Fu, q0_a3[ks] & 0x7F7F7F7Fu,
                    br0_1, br1_1, dr0_1, dr1_1
                );
                if constexpr (H_GROUP == 2) {
                    mma_m16n8k32_f16_e4m3(
                        er0_0, er1_0,
                        q1_a0[ks] & 0x7F7F7F7Fu, q1_a1[ks] & 0x7F7F7F7Fu,
                        q1_a2[ks] & 0x7F7F7F7Fu, q1_a3[ks] & 0x7F7F7F7Fu,
                        br0_0, br1_0, er0_0, er1_0
                    );
                    mma_m16n8k32_f16_e4m3(
                        er0_1, er1_1,
                        q1_a0[ks] & 0x7F7F7F7Fu, q1_a1[ks] & 0x7F7F7F7Fu,
                        q1_a2[ks] & 0x7F7F7F7Fu, q1_a3[ks] & 0x7F7F7F7Fu,
                        br0_1, br1_1, er0_1, er1_1
                    );
                }
            }

            const float2 sc_a0 = smem_ksc[nt0 * 8 + 2 * t + 0];
            const float2 sc_a1 = smem_ksc[nt0 * 8 + 2 * t + 1];
            const float2 sc_b0 = smem_ksc[(nt0 + 1) * 8 + 2 * t + 0];
            const float2 sc_b1 = smem_ksc[(nt0 + 1) * 8 + 2 * t + 1];

            // Head 0 dequant + warp row-max
            {
                const float2 fm0_a = __half22float2(*reinterpret_cast<const __half2*>(&dm0_0));
                const float2 fm1_a = __half22float2(*reinterpret_cast<const __half2*>(&dm1_0));
                const float2 fr0_a = __half22float2(*reinterpret_cast<const __half2*>(&dr0_0));
                const float2 fr1_a = __half22float2(*reinterpret_cast<const __half2*>(&dr1_0));
                const float2 fm0_b = __half22float2(*reinterpret_cast<const __half2*>(&dm0_1));
                const float2 fm1_b = __half22float2(*reinterpret_cast<const __half2*>(&dm1_1));
                const float2 fr0_b = __half22float2(*reinterpret_cast<const __half2*>(&dr0_1));
                const float2 fr1_b = __half22float2(*reinterpret_cast<const __half2*>(&dr1_1));

                float ca0 = -1.0e30f, ca1 = -1.0e30f, cb0 = -1.0e30f, cb1 = -1.0e30f;
                if (r0_valid) {
                    ca0 = fmaxf(ca0, (fm0_a.x * sc_a0.x + fr0_a.x * sc_a0.y) * qs0_0);
                    ca1 = fmaxf(ca1, (fm0_a.y * sc_a1.x + fr0_a.y * sc_a1.y) * qs0_0);
                    cb0 = fmaxf(cb0, (fm0_b.x * sc_b0.x + fr0_b.x * sc_b0.y) * qs0_0);
                    cb1 = fmaxf(cb1, (fm0_b.y * sc_b1.x + fr0_b.y * sc_b1.y) * qs0_0);
                }
                if (r1_valid) {
                    ca0 = fmaxf(ca0, (fm1_a.x * sc_a0.x + fr1_a.x * sc_a0.y) * qs0_1);
                    ca1 = fmaxf(ca1, (fm1_a.y * sc_a1.x + fr1_a.y * sc_a1.y) * qs0_1);
                    cb0 = fmaxf(cb0, (fm1_b.x * sc_b0.x + fr1_b.x * sc_b0.y) * qs0_1);
                    cb1 = fmaxf(cb1, (fm1_b.y * sc_b1.x + fr1_b.y * sc_b1.y) * qs0_1);
                }
                ca0 = fmaxf(ca0, __shfl_xor_sync(0xffffffffu, ca0, 4));
                ca0 = fmaxf(ca0, __shfl_xor_sync(0xffffffffu, ca0, 8));
                ca0 = fmaxf(ca0, __shfl_xor_sync(0xffffffffu, ca0, 16));
                ca1 = fmaxf(ca1, __shfl_xor_sync(0xffffffffu, ca1, 4));
                ca1 = fmaxf(ca1, __shfl_xor_sync(0xffffffffu, ca1, 8));
                ca1 = fmaxf(ca1, __shfl_xor_sync(0xffffffffu, ca1, 16));
                cb0 = fmaxf(cb0, __shfl_xor_sync(0xffffffffu, cb0, 4));
                cb0 = fmaxf(cb0, __shfl_xor_sync(0xffffffffu, cb0, 8));
                cb0 = fmaxf(cb0, __shfl_xor_sync(0xffffffffu, cb0, 16));
                cb1 = fmaxf(cb1, __shfl_xor_sync(0xffffffffu, cb1, 4));
                cb1 = fmaxf(cb1, __shfl_xor_sync(0xffffffffu, cb1, 8));
                cb1 = fmaxf(cb1, __shfl_xor_sync(0xffffffffu, cb1, 16));
                if (gid == 0) {
                    smem_wmax[0][warp][nt0 * 8 + 2 * t + 0] = ca0;
                    smem_wmax[0][warp][nt0 * 8 + 2 * t + 1] = ca1;
                    smem_wmax[0][warp][(nt0 + 1) * 8 + 2 * t + 0] = cb0;
                    smem_wmax[0][warp][(nt0 + 1) * 8 + 2 * t + 1] = cb1;
                }
            }

            if constexpr (H_GROUP == 2) {
                const float2 fm0_a = __half22float2(*reinterpret_cast<const __half2*>(&em0_0));
                const float2 fm1_a = __half22float2(*reinterpret_cast<const __half2*>(&em1_0));
                const float2 fr0_a = __half22float2(*reinterpret_cast<const __half2*>(&er0_0));
                const float2 fr1_a = __half22float2(*reinterpret_cast<const __half2*>(&er1_0));
                const float2 fm0_b = __half22float2(*reinterpret_cast<const __half2*>(&em0_1));
                const float2 fm1_b = __half22float2(*reinterpret_cast<const __half2*>(&em1_1));
                const float2 fr0_b = __half22float2(*reinterpret_cast<const __half2*>(&er0_1));
                const float2 fr1_b = __half22float2(*reinterpret_cast<const __half2*>(&er1_1));

                float ca0 = -1.0e30f, ca1 = -1.0e30f, cb0 = -1.0e30f, cb1 = -1.0e30f;
                if (r0_valid) {
                    ca0 = fmaxf(ca0, (fm0_a.x * sc_a0.x + fr0_a.x * sc_a0.y) * qs1_0);
                    ca1 = fmaxf(ca1, (fm0_a.y * sc_a1.x + fr0_a.y * sc_a1.y) * qs1_0);
                    cb0 = fmaxf(cb0, (fm0_b.x * sc_b0.x + fr0_b.x * sc_b0.y) * qs1_0);
                    cb1 = fmaxf(cb1, (fm0_b.y * sc_b1.x + fr0_b.y * sc_b1.y) * qs1_0);
                }
                if (r1_valid) {
                    ca0 = fmaxf(ca0, (fm1_a.x * sc_a0.x + fr1_a.x * sc_a0.y) * qs1_1);
                    ca1 = fmaxf(ca1, (fm1_a.y * sc_a1.x + fr1_a.y * sc_a1.y) * qs1_1);
                    cb0 = fmaxf(cb0, (fm1_b.x * sc_b0.x + fr1_b.x * sc_b0.y) * qs1_1);
                    cb1 = fmaxf(cb1, (fm1_b.y * sc_b1.x + fr1_b.y * sc_b1.y) * qs1_1);
                }
                ca0 = fmaxf(ca0, __shfl_xor_sync(0xffffffffu, ca0, 4));
                ca0 = fmaxf(ca0, __shfl_xor_sync(0xffffffffu, ca0, 8));
                ca0 = fmaxf(ca0, __shfl_xor_sync(0xffffffffu, ca0, 16));
                ca1 = fmaxf(ca1, __shfl_xor_sync(0xffffffffu, ca1, 4));
                ca1 = fmaxf(ca1, __shfl_xor_sync(0xffffffffu, ca1, 8));
                ca1 = fmaxf(ca1, __shfl_xor_sync(0xffffffffu, ca1, 16));
                cb0 = fmaxf(cb0, __shfl_xor_sync(0xffffffffu, cb0, 4));
                cb0 = fmaxf(cb0, __shfl_xor_sync(0xffffffffu, cb0, 8));
                cb0 = fmaxf(cb0, __shfl_xor_sync(0xffffffffu, cb0, 16));
                cb1 = fmaxf(cb1, __shfl_xor_sync(0xffffffffu, cb1, 4));
                cb1 = fmaxf(cb1, __shfl_xor_sync(0xffffffffu, cb1, 8));
                cb1 = fmaxf(cb1, __shfl_xor_sync(0xffffffffu, cb1, 16));
                if (gid == 0) {
                    smem_wmax[1][warp][nt0 * 8 + 2 * t + 0] = ca0;
                    smem_wmax[1][warp][nt0 * 8 + 2 * t + 1] = ca1;
                    smem_wmax[1][warp][(nt0 + 1) * 8 + 2 * t + 0] = cb0;
                    smem_wmax[1][warp][(nt0 + 1) * 8 + 2 * t + 1] = cb1;
                }
            } else {
                (void)em0_0; (void)em1_0; (void)em0_1; (void)em1_1;
                (void)er0_0; (void)er1_0; (void)er0_1; (void)er1_1;
            }
        }
        __syncthreads();

        // Cross-warp max + threshold check against current k_need_mid-th best score!
        if (tid < H_GROUP * BLOCK_B) {
            const int u = tid >> 6;
            const int col = tid & 63;
            const int lb = b0 + col;
            uint64_t key = 0ULL;
            if (lb >= s_cnt && lb < local_start) {
                float mx = -1.0e30f;
                #pragma unroll
                for (int w = 0; w < WARPS_SCORE; ++w) {
                    mx = fmaxf(mx, smem_wmax[u][w][col]);
                }
                key = pack_score_idx_u64(mx, lb);
            }
            smem_top[u][128 + col] = key;
            smem_top[u][192 + col] = 0ULL;
            if (key > smem_kth[u]) {
                smem_beat[u] = 1;
            }
        }
        __syncthreads();

        bool any_beat = (smem_beat[0] != 0);
        if constexpr (H_GROUP == 2) {
            any_beat = any_beat || (smem_beat[1] != 0);
        }
        if (any_beat) {
            bitonic_sort_desc_256<H_GROUP>(smem_top, tid);
            if (tid < H_GROUP) {
                smem_kth[tid] = smem_top[tid][k_need_mid - 1];
            }
            __syncthreads();
        }
    }

    // Extract Top-k_need_mid middle block indices and sort ascending in-SMEM (reuse smem_wmax as int32[H_GROUP][128])
    int32_t (*smem_idx)[128] = reinterpret_cast<int32_t (*)[128]>(smem_wmax);
    if (tid < H_GROUP * 128) {
        const int u = tid >> 7;
        const int idx = tid & 127;
        smem_idx[u][idx] = (idx < k_need_mid) ? unpack_idx_from_u64(smem_top[u][idx]) : INT_MAX;
    }
    __syncthreads();

    bitonic_sort_asc_i32_128<H_GROUP>(smem_idx, tid);

    // Coalesced write of [0..s_cnt-1, sorted_mid[0..k_need_mid-1], local_start..n_pblks-1]!
    #pragma unroll
    for (int u = 0; u < H_GROUP; ++u) {
        int32_t* dst = SEL_BLOCKS + static_cast<int64_t>(base_cta_idx + u) * ssel_row;
        for (int c = tid; c < k_sel; c += THREADS_SCORE) {
            int32_t val;
            if (c < s_cnt) {
                val = c;
            } else if (c < s_cnt + k_need_mid) {
                val = smem_idx[u][c - s_cnt];
            } else {
                val = local_start + (c - s_cnt - k_need_mid);
            }
            dst[c] = val;
        }
    }
}

// =================================================================================================
// KERNEL 3: 0-ULP Prefill KVQ Prologue with Automatic Sparse-Discard Block Skip
//   - K & V CTAs: 128 threads (4 warps), 32 tokens/CTA (8 tokens/warp), W_conv cached in regs,
//                 skips ConvCache writes when >N_cache tokens of same seq share a physical block!
//   - Q CTAs:     96 active threads (6 heads x 16 lanes), 16 tokens/CTA, QScale cached in regs!
// =================================================================================================

template <bool IS_V, bool DYNAMIC>
__device__ __forceinline__ void run_kv_prefill_cta(
    int cta_tile,
    const uint16_t* __restrict__ X,
    const uint16_t* __restrict__ W,
    const uint16_t* __restrict__ Dyn,
    uint16_t* __restrict__ ConvCache,
    uint16_t* __restrict__ KVCache,
    const int64_t* __restrict__ KVSlot,
    const int64_t* __restrict__ Pos,
    const int32_t* __restrict__ SeqIdx,
    const int64_t* __restrict__ Slot,
    const int32_t* __restrict__ BlockTable,
    const int32_t* __restrict__ QStart,
    int T, int64_t stride_x, int64_t stride_w0, int64_t stride_w1, int64_t stride_dyn,
    int64_t stride_c_blk, int64_t stride_c_n, int64_t stride_kv_blk, int64_t stride_kv_n,
    int kv_bs, int kv_shift, int64_t stride_bt, int max_blocks, int N_cache, int N_shift,
    float scale, float clip_threshold, float clip_eps, float norm_eps
) {
    constexpr int TOK_PER_WARP = 8;
    constexpr int WARPS_KV = 4;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int t_begin = (cta_tile * WARPS_KV + warp) * TOK_PER_WARP;
    if (t_begin >= T) return;
    const int t_end = min(t_begin + TOK_PER_WARP, T);

    const int c0 = lane * 4, c1 = c0 + 1, c2 = c0 + 2, c3 = c0 + 3;
    const int64_t cache_off = IS_V ? 128 : 0;

    float w_t[4][4];
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        if constexpr (DYNAMIC) {
            w_t[k][0] = bf16_to_f32_u16(W[k * stride_w0 + c0 * stride_w1]);
            w_t[k][1] = bf16_to_f32_u16(W[k * stride_w0 + c1 * stride_w1]);
            w_t[k][2] = bf16_to_f32_u16(W[k * stride_w0 + c2 * stride_w1]);
            w_t[k][3] = bf16_to_f32_u16(W[k * stride_w0 + c3 * stride_w1]);
        } else {
            w_t[k][0] = bf16_to_f32_u16(W[c0 * stride_w0 + k * stride_w1]);
            w_t[k][1] = bf16_to_f32_u16(W[c1 * stride_w0 + k * stride_w1]);
            w_t[k][2] = bf16_to_f32_u16(W[c2 * stride_w0 + k * stride_w1]);
            w_t[k][3] = bf16_to_f32_u16(W[c3 * stride_w0 + k * stride_w1]);
        }
    }

    for (int t = t_begin; t < t_end; ++t) {
        const int64_t pos = Pos[t];
        const int64_t sl = Slot[t];
        const int64_t kvs = KVSlot[t];
        const int32_t req = SeqIdx[t];
        const int32_t qs = QStart[t];
        const bool valid = (sl >= 0);

        const uint16_t* xp = X + static_cast<int64_t>(t) * stride_x;
        const uint32_t cur01 = *reinterpret_cast<const uint32_t*>(xp + c0);
        const uint32_t cur23 = *reinterpret_cast<const uint32_t*>(xp + c2);
        const float cur[4] = {
            bf16_to_f32_u16(static_cast<uint16_t>(cur01 & 0xFFFFu)),
            bf16_to_f32_u16(static_cast<uint16_t>(cur01 >> 16)),
            bf16_to_f32_u16(static_cast<uint16_t>(cur23 & 0xFFFFu)),
            bf16_to_f32_u16(static_cast<uint16_t>(cur23 >> 16)),
        };

        if (valid) {
            int64_t dblk, drem;
            divmod_p2(sl, N_cache, N_shift, dblk, drem);
            // Zero-arg sparse_prefill_discard detection: within one sequence, tokens N_cache
            // positions apart NEVER legitimately occupy the same N_cache-slot physical block!
            bool is_discard = false;
            if (N_shift >= 0) {
                if (t + N_cache < T && SeqIdx[t + N_cache] == req) {
                    const int64_t sl_next = Slot[t + N_cache];
                    if (sl_next >= 0 && (sl_next >> N_shift) == dblk) is_discard = true;
                }
                if (!is_discard && t >= N_cache && SeqIdx[t - N_cache] == req) {
                    const int64_t sl_prev = Slot[t - N_cache];
                    if (sl_prev >= 0 && (sl_prev >> N_shift) == dblk) is_discard = true;
                }
            }
            if (!is_discard) {
                uint16_t* dst_c = ConvCache + dblk * stride_c_blk + drem * stride_c_n + cache_off;
                *reinterpret_cast<uint32_t*>(dst_c + c0) = cur01;
                *reinterpret_cast<uint32_t*>(dst_c + c2) = cur23;
            }
        }

        float dyn_w[4] = {0.f, 0.f, 0.f, 0.f};
        if constexpr (DYNAMIC) {
            const uint16_t* dp = Dyn + static_cast<int64_t>(t) * stride_dyn;
            #pragma unroll
            for (int k = 0; k < 4; ++k) dyn_w[k] = bf16_to_f32_u16(dp[k]);
        }

        float acc[4] = {0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int lag = DYNAMIC ? k : (3 - k);
            const int64_t src = pos - lag;
            const bool in_win = valid && (src >= 0);
            const int row = t - lag;
            const bool intra = in_win && (row >= qs);
            const bool cached = in_win && (row < qs);

            float xt[4] = {0.f, 0.f, 0.f, 0.f};
            if (intra) {
                if (lag == 0) {
                    xt[0] = __fadd_rn(cur[0], 0.f);
                    xt[1] = __fadd_rn(cur[1], 0.f);
                    xt[2] = __fadd_rn(cur[2], 0.f);
                    xt[3] = __fadd_rn(cur[3], 0.f);
                } else {
                    const uint16_t* rx = X + static_cast<int64_t>(row) * stride_x;
                    const uint32_t r01 = *reinterpret_cast<const uint32_t*>(rx + c0);
                    const uint32_t r23 = *reinterpret_cast<const uint32_t*>(rx + c2);
                    xt[0] = __fadd_rn(bf16_to_f32_u16(static_cast<uint16_t>(r01 & 0xFFFFu)), 0.f);
                    xt[1] = __fadd_rn(bf16_to_f32_u16(static_cast<uint16_t>(r01 >> 16)), 0.f);
                    xt[2] = __fadd_rn(bf16_to_f32_u16(static_cast<uint16_t>(r23 & 0xFFFFu)), 0.f);
                    xt[3] = __fadd_rn(bf16_to_f32_u16(static_cast<uint16_t>(r23 >> 16)), 0.f);
                }
            } else if (cached) {
                int64_t lblk, rem;
                divmod_p2(src, N_cache, N_shift, lblk, rem);
                if (lblk >= max_blocks) lblk = max_blocks - 1;
                const int64_t blk = static_cast<int64_t>(BlockTable[static_cast<int64_t>(req) * stride_bt + lblk]);
                const uint16_t* cp = ConvCache + blk * stride_c_blk + rem * stride_c_n + cache_off;
                const uint32_t r01 = *reinterpret_cast<const uint32_t*>(cp + c0);
                const uint32_t r23 = *reinterpret_cast<const uint32_t*>(cp + c2);
                xt[0] = __fadd_rn(0.f, bf16_to_f32_u16(static_cast<uint16_t>(r01 & 0xFFFFu)));
                xt[1] = __fadd_rn(0.f, bf16_to_f32_u16(static_cast<uint16_t>(r01 >> 16)));
                xt[2] = __fadd_rn(0.f, bf16_to_f32_u16(static_cast<uint16_t>(r23 & 0xFFFFu)));
                xt[3] = __fadd_rn(0.f, bf16_to_f32_u16(static_cast<uint16_t>(r23 >> 16)));
            }

            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float wk = DYNAMIC ? __fadd_rn(w_t[k][i], dyn_w[k]) : w_t[k][i];
                if (DYNAMIC && k == 0) acc[i] = __fmul_rn(xt[i], wk);
                else acc[i] = fmaf(xt[i], wk, acc[i]);
            }
        }

        float y[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            y[i] = bf16_to_f32_u16(f32_to_bf16_u16(__fmul_rn(__fadd_rn(acc[i], cur[i]), scale)));
        }
        const float s0 = fmaf(y[0], y[0], __fmul_rn(y[2], y[2]));
        const float s1 = fmaf(y[1], y[1], __fmul_rn(y[3], y[3]));
        float sum_sq = __fadd_rn(s0, s1);
        sum_sq = __fadd_rn(sum_sq, __shfl_xor_sync(0xffffffffu, sum_sq, 16));
        sum_sq = __fadd_rn(sum_sq, __shfl_xor_sync(0xffffffffu, sum_sq, 8));
        sum_sq = __fadd_rn(sum_sq, __shfl_xor_sync(0xffffffffu, sum_sq, 4));
        sum_sq = __fadd_rn(sum_sq, __shfl_xor_sync(0xffffffffu, sum_sq, 2));
        sum_sq = __fadd_rn(sum_sq, __shfl_xor_sync(0xffffffffu, sum_sq, 1));

        float mult;
        if constexpr (IS_V) {
            mult = rsqrt_approx_f32(__fadd_rn(__fdiv_rn(sum_sq, 128.0f), norm_eps));
        } else {
            const float norm = (sum_sq > 0.0f) ? __fmul_rn(sum_sq, rsqrt_approx_f32(sum_sq)) : 0.0f;
            mult = fminf(__fdiv_rn(clip_threshold, __fadd_rn(norm, clip_eps)), 1.0f);
        }

        if (kvs >= 0) {
            int64_t kv_blk, kv_rem;
            divmod_p2(kvs, kv_bs, kv_shift, kv_blk, kv_rem);
            uint16_t* kv_dst = KVCache + kv_blk * stride_kv_blk + kv_rem * stride_kv_n;
            *reinterpret_cast<uint32_t*>(kv_dst + c0) = pack_bf16x2_u32(
                f32_to_bf16_u16(__fmul_rn(y[0], mult)),
                f32_to_bf16_u16(__fmul_rn(y[1], mult))
            );
            *reinterpret_cast<uint32_t*>(kv_dst + c2) = pack_bf16x2_u32(
                f32_to_bf16_u16(__fmul_rn(y[2], mult)),
                f32_to_bf16_u16(__fmul_rn(y[3], mult))
            );
        }
    }
}

template <bool HAS_TAU>
__device__ __forceinline__ void run_q_prefill_cta(
    int q_tile,
    const uint16_t* __restrict__ Q,
    const uint16_t* __restrict__ QScale,
    const float* __restrict__ Tau,
    uint16_t* __restrict__ QOut,
    int T, int64_t stride_q, float clip_threshold, float clip_eps
) {
    constexpr int TOK_PER_CTA_Q = 16;
    if (threadIdx.x >= 96) return;
    const int t_begin = q_tile * TOK_PER_CTA_Q;
    if (t_begin >= T) return;
    const int t_end = min(t_begin + TOK_PER_CTA_Q, T);

    const int h = threadIdx.x >> 4;   // 0..5 (HQ = 6)
    const int l = threadIdx.x & 15;   // 0..15
    const int d0 = l * 8;
    float sc[8];
    {
        const uint4 sv = *reinterpret_cast<const uint4*>(QScale + h * D_HEAD + d0);
        sc[0] = bf16_to_f32_u16(static_cast<uint16_t>(sv.x & 0xFFFFu));
        sc[1] = bf16_to_f32_u16(static_cast<uint16_t>(sv.x >> 16));
        sc[2] = bf16_to_f32_u16(static_cast<uint16_t>(sv.y & 0xFFFFu));
        sc[3] = bf16_to_f32_u16(static_cast<uint16_t>(sv.y >> 16));
        sc[4] = bf16_to_f32_u16(static_cast<uint16_t>(sv.z & 0xFFFFu));
        sc[5] = bf16_to_f32_u16(static_cast<uint16_t>(sv.z >> 16));
        sc[6] = bf16_to_f32_u16(static_cast<uint16_t>(sv.w & 0xFFFFu));
        sc[7] = bf16_to_f32_u16(static_cast<uint16_t>(sv.w >> 16));
    }

    const bool q_aligned16 = ((stride_q & 7) == 0) && ((reinterpret_cast<uintptr_t>(Q) & 15) == 0);
    for (int t = t_begin; t < t_end; ++t) {
        const uint16_t* qp = Q + static_cast<int64_t>(t) * stride_q + h * D_HEAD + d0;
        uint4 qv;
        if (q_aligned16) {
            qv = *reinterpret_cast<const uint4*>(qp);
        } else {
            qv.x = *reinterpret_cast<const uint32_t*>(qp + 0);
            qv.y = *reinterpret_cast<const uint32_t*>(qp + 2);
            qv.z = *reinterpret_cast<const uint32_t*>(qp + 4);
            qv.w = *reinterpret_cast<const uint32_t*>(qp + 6);
        }
        float x[8];
        x[0] = __fmul_rn(bf16_to_f32_u16(static_cast<uint16_t>(qv.x & 0xFFFFu)), sc[0]);
        x[1] = __fmul_rn(bf16_to_f32_u16(static_cast<uint16_t>(qv.x >> 16)),      sc[1]);
        x[2] = __fmul_rn(bf16_to_f32_u16(static_cast<uint16_t>(qv.y & 0xFFFFu)), sc[2]);
        x[3] = __fmul_rn(bf16_to_f32_u16(static_cast<uint16_t>(qv.y >> 16)),      sc[3]);
        x[4] = __fmul_rn(bf16_to_f32_u16(static_cast<uint16_t>(qv.z & 0xFFFFu)), sc[4]);
        x[5] = __fmul_rn(bf16_to_f32_u16(static_cast<uint16_t>(qv.z >> 16)),      sc[5]);
        x[6] = __fmul_rn(bf16_to_f32_u16(static_cast<uint16_t>(qv.w & 0xFFFFu)), sc[6]);
        x[7] = __fmul_rn(bf16_to_f32_u16(static_cast<uint16_t>(qv.w >> 16)),      sc[7]);

        const float e01 = fmaf(x[0], x[0], __fmul_rn(x[2], x[2]));
        const float o01 = fmaf(x[1], x[1], __fmul_rn(x[3], x[3]));
        const float e23 = fmaf(x[4], x[4], __fmul_rn(x[6], x[6]));
        const float o23 = fmaf(x[5], x[5], __fmul_rn(x[7], x[7]));
        float sum_sq = __fadd_rn(__fadd_rn(e01, e23), __fadd_rn(o01, o23));
        sum_sq = __fadd_rn(sum_sq, __shfl_xor_sync(0xffffffffu, sum_sq, 8));
        sum_sq = __fadd_rn(sum_sq, __shfl_xor_sync(0xffffffffu, sum_sq, 4));
        sum_sq = __fadd_rn(sum_sq, __shfl_xor_sync(0xffffffffu, sum_sq, 2));
        sum_sq = __fadd_rn(sum_sq, __shfl_xor_sync(0xffffffffu, sum_sq, 1));

        const float norm = (sum_sq > 0.0f) ? __fmul_rn(sum_sq, rsqrt_approx_f32(sum_sq)) : 0.0f;
        const float mult = fminf(__fdiv_rn(clip_threshold, __fadd_rn(norm, clip_eps)), 1.0f);
        const float tau = HAS_TAU ? Tau[t] : 1.0f;

        uint16_t b[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            uint16_t bi = f32_to_bf16_u16(__fmul_rn(x[i], mult));
            if constexpr (HAS_TAU) bi = f32_to_bf16_u16(__fmul_rn(bf16_to_f32_u16(bi), tau));
            b[i] = bi;
        }
        uint4 ov;
        ov.x = pack_bf16x2_u32(b[0], b[1]);
        ov.y = pack_bf16x2_u32(b[2], b[3]);
        ov.z = pack_bf16x2_u32(b[4], b[5]);
        ov.w = pack_bf16x2_u32(b[6], b[7]);
        *reinterpret_cast<uint4*>(QOut + static_cast<int64_t>(t) * (6 * D_HEAD) + h * D_HEAD + d0) = ov;
    }
}

template <bool DYNAMIC, bool HAS_TAU>
__global__ void __launch_bounds__(128, 4) kvq_prologue_pf_v4_sm121_kernel(
    const uint16_t* __restrict__ K,
    const uint16_t* __restrict__ V,
    const uint16_t* __restrict__ WK,
    const uint16_t* __restrict__ WV,
    const uint16_t* __restrict__ DK,
    const uint16_t* __restrict__ DV,
    uint16_t* __restrict__ ConvCache,
    uint16_t* __restrict__ KeyCache,
    uint16_t* __restrict__ ValCache,
    const int64_t* __restrict__ KVSlot,
    const int64_t* __restrict__ Pos,
    const int32_t* __restrict__ SeqIdx,
    const int64_t* __restrict__ Slot,
    const int32_t* __restrict__ BlockTable,
    const int32_t* __restrict__ QStart,
    const uint16_t* __restrict__ Q,
    const uint16_t* __restrict__ QScale,
    const float* __restrict__ Tau,
    uint16_t* __restrict__ QOut,
    int T, int n_kv_ctas,
    int64_t stride_k, int64_t stride_v, int64_t stride_w0, int64_t stride_w1,
    int64_t stride_dk, int64_t stride_dv,
    int64_t stride_c_blk, int64_t stride_c_n,
    int64_t stride_kv_blk, int64_t stride_kv_n,
    int kv_bs, int kv_shift, int64_t stride_bt, int max_blocks,
    int N_cache, int N_shift, int64_t stride_q,
    float scale, float clip_threshold, float clip_eps, float norm_eps
) {
    const int bx = static_cast<int>(blockIdx.x);
    if (bx < n_kv_ctas) {
        run_kv_prefill_cta<false, DYNAMIC>(
            bx, K, WK, DK, ConvCache, KeyCache, KVSlot, Pos, SeqIdx, Slot, BlockTable, QStart,
            T, stride_k, stride_w0, stride_w1, stride_dk, stride_c_blk, stride_c_n,
            stride_kv_blk, stride_kv_n, kv_bs, kv_shift, stride_bt, max_blocks, N_cache, N_shift,
            scale, clip_threshold, clip_eps, norm_eps
        );
    } else if (bx < 2 * n_kv_ctas) {
        run_kv_prefill_cta<true, DYNAMIC>(
            bx - n_kv_ctas, V, WV, DV, ConvCache, ValCache, KVSlot, Pos, SeqIdx, Slot, BlockTable, QStart,
            T, stride_v, stride_w0, stride_w1, stride_dv, stride_c_blk, stride_c_n,
            stride_kv_blk, stride_kv_n, kv_bs, kv_shift, stride_bt, max_blocks, N_cache, N_shift,
            scale, clip_threshold, clip_eps, norm_eps
        );
    } else {
        run_q_prefill_cta<HAS_TAU>(
            bx - 2 * n_kv_ctas, Q, QScale, Tau, QOut, T, stride_q, clip_threshold, clip_eps
        );
    }
}

// =================================================================================================
// KERNEL 4: Multi-Row Vectorized BF16 -> Rowwise FP8 E4M3 Quantizer
//   - Specialized K=768 (O_proj / wo_ud): 16 rows/CTA (16 threads/row), 6x uint4 in regs,
//     0 SMEM, 0 __syncthreads(), 0 masked lanes, 3x uint4 coalesced FP8 stores!
//   - General K%128==0 (e.g. K=4608): 2 rows/CTA (128 threads/row), L1-cached 2-pass uint4!
// =================================================================================================

__global__ void __launch_bounds__(256, 4) quant_fp8_rowwise_k768_v4_kernel(
    const __nv_bfloat16* __restrict__ X,
    uint8_t* __restrict__ Y_FP8,
    float* __restrict__ Y_SCALE,
    int M, int M_pad, int64_t sx_row
) {
    const int row = static_cast<int>(blockIdx.x) * 16 + (static_cast<int>(threadIdx.x) >> 4);
    const int lane16 = static_cast<int>(threadIdx.x) & 15;
    if (row >= M_pad) return;

    if (row >= M) {
        uint4 zero4 = make_uint4(0u, 0u, 0u, 0u);
        uint4* dst4 = reinterpret_cast<uint4*>(Y_FP8 + static_cast<int64_t>(row) * 768 + lane16 * 48);
        dst4[0] = zero4;
        dst4[1] = zero4;
        dst4[2] = zero4;
        if (lane16 == 0) Y_SCALE[row] = 1.0f;
        return;
    }

    const uint4* src4 = reinterpret_cast<const uint4*>(X + static_cast<int64_t>(row) * sx_row + lane16 * 48);
    uint4 v[6];
    float amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < 6; ++i) {
        v[i] = src4[i];
        const float2 f0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v[i].x));
        const float2 f1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v[i].y));
        const float2 f2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v[i].z));
        const float2 f3 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v[i].w));
        amax = fmaxf(amax, fmaxf(fmaxf(fabsf(f0.x), fabsf(f0.y)), fmaxf(fabsf(f1.x), fabsf(f1.y))));
        amax = fmaxf(amax, fmaxf(fmaxf(fabsf(f2.x), fabsf(f2.y)), fmaxf(fabsf(f3.x), fabsf(f3.y))));
    }

    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 8));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 4));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));

    const float sc = fmaxf(amax * INV_FP8_E4M3_MAX, 1.0e-12f);
    const float inv = div_rn_ftz(1.0f, sc);

    uint4* dst4 = reinterpret_cast<uint4*>(Y_FP8 + static_cast<int64_t>(row) * 768 + lane16 * 48);
    #pragma unroll
    for (int g = 0; g < 3; ++g) {
        const uint4 a = v[2 * g + 0];
        const uint4 b = v[2 * g + 1];
        const float2 f0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&a.x));
        const float2 f1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&a.y));
        const float2 f2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&a.z));
        const float2 f3 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&a.w));
        const float2 f4 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&b.x));
        const float2 f5 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&b.y));
        const float2 f6 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&b.z));
        const float2 f7 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&b.w));
        uint4 pack;
        pack.x = f32x4_to_e4m3x4_rn(f0.x * inv, f0.y * inv, f1.x * inv, f1.y * inv);
        pack.y = f32x4_to_e4m3x4_rn(f2.x * inv, f2.y * inv, f3.x * inv, f3.y * inv);
        pack.z = f32x4_to_e4m3x4_rn(f4.x * inv, f4.y * inv, f5.x * inv, f5.y * inv);
        pack.w = f32x4_to_e4m3x4_rn(f6.x * inv, f6.y * inv, f7.x * inv, f7.y * inv);
        dst4[g] = pack;
    }
    if (lane16 == 0) {
        Y_SCALE[row] = sc;
    }
}

__global__ void __launch_bounds__(256, 4) quant_fp8_rowwise_gen128_v4_kernel(
    const __nv_bfloat16* __restrict__ X,
    uint8_t* __restrict__ Y_FP8,
    float* __restrict__ Y_SCALE,
    int M, int M_pad, int K, int64_t sx_row
) {
    const int tid = static_cast<int>(threadIdx.x);
    const int sub_r = tid >> 7;      // 0..1 (2 rows per CTA, 128 threads/row)
    const int r_tid = tid & 127;     // 0..127
    const int warp = tid >> 5;       // 0..7
    const int lane = tid & 31;
    const int row = static_cast<int>(blockIdx.x) * 2 + sub_r;

    __shared__ float smem_wmax[2][4];

    if (row < M) {
        float amax = 0.0f;
        const __nv_bfloat16* x_row = X + static_cast<int64_t>(row) * sx_row;
        for (int c = r_tid * 8; c < K; c += 1024) {
            const uint4 v = *reinterpret_cast<const uint4*>(x_row + c);
            const float2 f0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v.x));
            const float2 f1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v.y));
            const float2 f2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v.z));
            const float2 f3 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v.w));
            amax = fmaxf(amax, fmaxf(fmaxf(fabsf(f0.x), fabsf(f0.y)), fmaxf(fabsf(f1.x), fabsf(f1.y))));
            amax = fmaxf(amax, fmaxf(fmaxf(fabsf(f2.x), fabsf(f2.y)), fmaxf(fabsf(f3.x), fabsf(f3.y))));
        }
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 16));
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 8));
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 4));
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
        if (lane == 0) smem_wmax[sub_r][warp & 3] = amax;
    }
    __syncthreads();

    if (row >= M_pad) return;
    if (row >= M) {
        uint8_t* y_row = Y_FP8 + static_cast<int64_t>(row) * K;
        uint4 zero4 = make_uint4(0u, 0u, 0u, 0u);
        for (int c = r_tid * 16; c < K; c += 2048) {
            *reinterpret_cast<uint4*>(y_row + c) = zero4;
        }
        if (r_tid == 0) Y_SCALE[row] = 1.0f;
        return;
    }

    const float row_amax = fmaxf(
        fmaxf(smem_wmax[sub_r][0], smem_wmax[sub_r][1]),
        fmaxf(smem_wmax[sub_r][2], smem_wmax[sub_r][3])
    );
    const float sc = fmaxf(row_amax * INV_FP8_E4M3_MAX, 1.0e-12f);
    const float inv = div_rn_ftz(1.0f, sc);
    if (r_tid == 0) Y_SCALE[row] = sc;

    const __nv_bfloat16* x_row = X + static_cast<int64_t>(row) * sx_row;
    uint8_t* y_row = Y_FP8 + static_cast<int64_t>(row) * K;
    for (int c = r_tid * 16; c < K; c += 2048) {
        const uint4 a = *reinterpret_cast<const uint4*>(x_row + c);
        const uint4 b = *reinterpret_cast<const uint4*>(x_row + c + 8);
        const float2 f0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&a.x));
        const float2 f1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&a.y));
        const float2 f2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&a.z));
        const float2 f3 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&a.w));
        const float2 f4 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&b.x));
        const float2 f5 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&b.y));
        const float2 f6 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&b.z));
        const float2 f7 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&b.w));
        uint4 pack;
        pack.x = f32x4_to_e4m3x4_rn(f0.x * inv, f0.y * inv, f1.x * inv, f1.y * inv);
        pack.y = f32x4_to_e4m3x4_rn(f2.x * inv, f2.y * inv, f3.x * inv, f3.y * inv);
        pack.z = f32x4_to_e4m3x4_rn(f4.x * inv, f4.y * inv, f5.x * inv, f5.y * inv);
        pack.w = f32x4_to_e4m3x4_rn(f6.x * inv, f6.y * inv, f7.x * inv, f7.y * inv);
        *reinterpret_cast<uint4*>(y_row + c) = pack;
    }
}

}  // namespace

// =================================================================================================
// Exported C ABI Entrypoints
// =================================================================================================

extern "C" int quantize_k_shadow_v4_sm121_fwd(
    const void* k_in_ptr,
    void* k_fp8_out_ptr,
    void* ks64_out_ptr,
    void* k_mid_bf16_out_ptr,
    void* k_rad_bf16_out_ptr,
    void* k_mid_fp8_out_ptr,
    void* k_rad_fp8_out_ptr,
    void* k_box_sc_out_ptr,
    const void* bt_row_ptr,
    void* bt_state_ptr,
    int64_t sk_b, int64_t sk_s, int64_t sk_h,
    int64_t sk8_b, int64_t sk8_s, int64_t sk8_h,
    int64_t sks_b, int64_t sks_h,
    int64_t smid_b, int64_t smid_h,
    int64_t sbox8_b, int64_t sbox8_h,
    int64_t sboxsc_b, int64_t sboxsc_h,
    int nblks, int start_blk, int prev_idx,
    int check_contig, int compute_kmax, int compact_out,
    int BS, int Hkv,
    uint64_t stream_raw
) {
    if (nblks <= 0 || Hkv <= 0) return 0;
    if ((BS & 63) != 0 || BS <= 0) return -1;

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_raw);
    dim3 grid(static_cast<unsigned>(nblks), static_cast<unsigned>(Hkv));
    dim3 block(128);

    const auto* k_in = static_cast<const __nv_bfloat16*>(k_in_ptr);
    auto* k_fp8 = static_cast<uint8_t*>(k_fp8_out_ptr);
    auto* ks64 = static_cast<float*>(ks64_out_ptr);
    auto* km_bf = static_cast<__nv_bfloat16*>(k_mid_bf16_out_ptr);
    auto* kr_bf = static_cast<__nv_bfloat16*>(k_rad_bf16_out_ptr);
    auto* km_8 = static_cast<uint8_t*>(k_mid_fp8_out_ptr);
    auto* kr_8 = static_cast<uint8_t*>(k_rad_fp8_out_ptr);
    auto* kb_sc = static_cast<float2*>(k_box_sc_out_ptr);
    const auto* bt_row = static_cast<const int32_t*>(bt_row_ptr);
    auto* bt_state = static_cast<int32_t*>(bt_state_ptr);

    const bool cc = (check_contig != 0) && (bt_state != nullptr);
    const bool km = (compute_kmax != 0);
    const bool co = (compact_out != 0);

    #define LAUNCH_QK_V4(CC_B, KM_B, CO_B) \
        quantize_k_shadow_v4_kernel<CC_B, KM_B, CO_B><<<grid, block, 0, stream>>>( \
            k_in, k_fp8, ks64, km_bf, kr_bf, km_8, kr_8, kb_sc, bt_row, bt_state, \
            sk_b, sk_s, sk_h, sk8_b, sk8_s, sk8_h, sks_b, sks_h, \
            smid_b, smid_h, sbox8_b, sbox8_h, sboxsc_b, sboxsc_h, \
            nblks, start_blk, prev_idx, BS)

    if (cc) {
        if (km) {
            if (co) LAUNCH_QK_V4(true, true, true);
            else    LAUNCH_QK_V4(true, true, false);
        } else {
            if (co) LAUNCH_QK_V4(true, false, true);
            else    LAUNCH_QK_V4(true, false, false);
        }
    } else {
        if (km) {
            if (co) LAUNCH_QK_V4(false, true, true);
            else    LAUNCH_QK_V4(false, true, false);
        } else {
            if (co) LAUNCH_QK_V4(false, false, true);
            else    LAUNCH_QK_V4(false, false, false);
        }
    }
    #undef LAUNCH_QK_V4

    if (bt_state != nullptr) {
        update_bt_state_v4_kernel<<<1, 32, 0, stream>>>(bt_row, bt_state, nblks - 1);
    }
    return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

extern "C" int score_topk_sort_fp8_v4_sm121_fwd(
    const void* q_ptr,
    const void* k_mid_fp8_ptr,
    const void* k_rad_fp8_ptr,
    const void* k_box_sc_ptr,
    const void* bt_ptr,
    const void* seq_ptr,
    const void* cuq_ptr,
    void* sel_blocks_ptr,
    void* sel_counts_ptr,
    int64_t sq_t, int64_t sq_h,
    int64_t sbox8_b, int64_t sbox8_h,
    int64_t sboxsc_b, int64_t sboxsc_h,
    int64_t sbt_r,
    int64_t ssel_row,
    int num_reqs, int max_seqlen_q, int max_pblks,
    int H, int G, int BS, int E,
    int sink_blocks, int local_blocks, int k_sel,
    int k_bt_mode,
    uint64_t stream_raw
) {
    if (num_reqs <= 0 || max_seqlen_q <= 0 || max_pblks <= 0 || H <= 0 || G <= 0) return -1;
    if (BS <= 0 || (BS & (BS - 1)) != 0 || k_sel <= 0) return -2;
    const int k_need_max = k_sel - (sink_blocks + local_blocks);
    if (k_need_max > K_MID_CAP) return -3;  // clean fallback if TOPK > 32,768 tokens

    const int n_mt = (max_seqlen_q + BM_SCORE - 1) / BM_SCORE;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_raw);
    dim3 block(THREADS_SCORE);

    const bool can_hg2 = ((H & 1) == 0) && ((G & 1) == 0);
    if (can_hg2) {
        dim3 grid(static_cast<unsigned>(n_mt * (H / 2)), static_cast<unsigned>(num_reqs));
        if (k_bt_mode == 1) {
            score_topk_sort_fp8_v4_sm121_kernel<2, 1><<<grid, block, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(q_ptr),
                static_cast<const uint8_t*>(k_mid_fp8_ptr),
                static_cast<const uint8_t*>(k_rad_fp8_ptr),
                static_cast<const float2*>(k_box_sc_ptr),
                static_cast<const int32_t*>(bt_ptr),
                static_cast<const int32_t*>(seq_ptr),
                static_cast<const int32_t*>(cuq_ptr),
                static_cast<int32_t*>(sel_blocks_ptr),
                static_cast<int32_t*>(sel_counts_ptr),
                sq_t, sq_h, sbox8_b, sbox8_h, sboxsc_b, sboxsc_h, sbt_r, ssel_row,
                max_pblks, H, G, BS, E, sink_blocks, local_blocks, k_sel
            );
        } else {
            score_topk_sort_fp8_v4_sm121_kernel<2, 0><<<grid, block, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(q_ptr),
                static_cast<const uint8_t*>(k_mid_fp8_ptr),
                static_cast<const uint8_t*>(k_rad_fp8_ptr),
                static_cast<const float2*>(k_box_sc_ptr),
                static_cast<const int32_t*>(bt_ptr),
                static_cast<const int32_t*>(seq_ptr),
                static_cast<const int32_t*>(cuq_ptr),
                static_cast<int32_t*>(sel_blocks_ptr),
                static_cast<int32_t*>(sel_counts_ptr),
                sq_t, sq_h, sbox8_b, sbox8_h, sboxsc_b, sboxsc_h, sbt_r, ssel_row,
                max_pblks, H, G, BS, E, sink_blocks, local_blocks, k_sel
            );
        }
    } else {
        dim3 grid(static_cast<unsigned>(n_mt * H), static_cast<unsigned>(num_reqs));
        if (k_bt_mode == 1) {
            score_topk_sort_fp8_v4_sm121_kernel<1, 1><<<grid, block, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(q_ptr),
                static_cast<const uint8_t*>(k_mid_fp8_ptr),
                static_cast<const uint8_t*>(k_rad_fp8_ptr),
                static_cast<const float2*>(k_box_sc_ptr),
                static_cast<const int32_t*>(bt_ptr),
                static_cast<const int32_t*>(seq_ptr),
                static_cast<const int32_t*>(cuq_ptr),
                static_cast<int32_t*>(sel_blocks_ptr),
                static_cast<int32_t*>(sel_counts_ptr),
                sq_t, sq_h, sbox8_b, sbox8_h, sboxsc_b, sboxsc_h, sbt_r, ssel_row,
                max_pblks, H, G, BS, E, sink_blocks, local_blocks, k_sel
            );
        } else {
            score_topk_sort_fp8_v4_sm121_kernel<1, 0><<<grid, block, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(q_ptr),
                static_cast<const uint8_t*>(k_mid_fp8_ptr),
                static_cast<const uint8_t*>(k_rad_fp8_ptr),
                static_cast<const float2*>(k_box_sc_ptr),
                static_cast<const int32_t*>(bt_ptr),
                static_cast<const int32_t*>(seq_ptr),
                static_cast<const int32_t*>(cuq_ptr),
                static_cast<int32_t*>(sel_blocks_ptr),
                static_cast<int32_t*>(sel_counts_ptr),
                sq_t, sq_h, sbox8_b, sbox8_h, sboxsc_b, sboxsc_h, sbt_r, ssel_row,
                max_pblks, H, G, BS, E, sink_blocks, local_blocks, k_sel
            );
        }
    }
    return (cudaGetLastError() == cudaSuccess) ? 0 : -4;
}

extern "C" int kvq_prologue_pf_v4_sm121_fwd(
    const void* k_ptr,
    const void* v_ptr,
    const void* wk_ptr,
    const void* wv_ptr,
    const void* dk_ptr,
    const void* dv_ptr,
    void* conv_cache_ptr,
    void* key_cache_ptr,
    void* val_cache_ptr,
    const void* kv_slot_ptr,
    const void* pos_ptr,
    const void* seq_idx_ptr,
    const void* slot_ptr,
    const void* block_table_ptr,
    const void* qstart_ptr,
    const void* q_ptr,
    const void* q_scale_ptr,
    const void* tau_ptr,
    void* q_out_ptr,
    int T,
    int64_t stride_k, int64_t stride_v, int64_t stride_w0, int64_t stride_w1,
    int64_t stride_dk, int64_t stride_dv,
    int64_t stride_c_blk, int64_t stride_c_n,
    int64_t stride_kv_blk, int64_t stride_kv_n,
    int kv_bs, int64_t stride_bt, int max_blocks,
    int N_cache, int64_t stride_q,
    float scale, float clip_threshold, float clip_eps, float norm_eps,
    int dynamic, int has_tau,
    uint64_t stream_raw
) {
    if (T <= 0) return 0;
    if (kv_bs <= 0 || N_cache <= 0) return -1;

    const int kv_shift = ((kv_bs & (kv_bs - 1)) == 0) ? (__builtin_ctz(kv_bs)) : -1;
    const int N_shift = ((N_cache & (N_cache - 1)) == 0) ? (__builtin_ctz(N_cache)) : -1;

    const int n_kv_ctas = (T + 31) / 32;
    const int n_q_ctas = (T + 15) / 16;
    dim3 grid(static_cast<unsigned>(2 * n_kv_ctas + n_q_ctas));
    dim3 block(128);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_raw);

    #define LAUNCH_KVQ_V4(DYN_B, TAU_B) \
        kvq_prologue_pf_v4_sm121_kernel<DYN_B, TAU_B><<<grid, block, 0, stream>>>( \
            static_cast<const uint16_t*>(k_ptr), \
            static_cast<const uint16_t*>(v_ptr), \
            static_cast<const uint16_t*>(wk_ptr), \
            static_cast<const uint16_t*>(wv_ptr), \
            static_cast<const uint16_t*>(dk_ptr), \
            static_cast<const uint16_t*>(dv_ptr), \
            static_cast<uint16_t*>(conv_cache_ptr), \
            static_cast<uint16_t*>(key_cache_ptr), \
            static_cast<uint16_t*>(val_cache_ptr), \
            static_cast<const int64_t*>(kv_slot_ptr), \
            static_cast<const int64_t*>(pos_ptr), \
            static_cast<const int32_t*>(seq_idx_ptr), \
            static_cast<const int64_t*>(slot_ptr), \
            static_cast<const int32_t*>(block_table_ptr), \
            static_cast<const int32_t*>(qstart_ptr), \
            static_cast<const uint16_t*>(q_ptr), \
            static_cast<const uint16_t*>(q_scale_ptr), \
            static_cast<const float*>(tau_ptr), \
            static_cast<uint16_t*>(q_out_ptr), \
            T, n_kv_ctas, stride_k, stride_v, stride_w0, stride_w1, stride_dk, stride_dv, \
            stride_c_blk, stride_c_n, stride_kv_blk, stride_kv_n, kv_bs, kv_shift, \
            stride_bt, max_blocks, N_cache, N_shift, stride_q, \
            scale, clip_threshold, clip_eps, norm_eps)

    if (dynamic != 0) {
        if (has_tau != 0) LAUNCH_KVQ_V4(true, true);
        else              LAUNCH_KVQ_V4(true, false);
    } else {
        if (has_tau != 0) LAUNCH_KVQ_V4(false, true);
        else              LAUNCH_KVQ_V4(false, false);
    }
    #undef LAUNCH_KVQ_V4

    return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

extern "C" int quant_fp8_rowwise_v4_sm121_fwd(
    const void* x_ptr,
    void* y_fp8_ptr,
    void* y_scale_ptr,
    int M, int M_pad, int K, int64_t sx_row,
    uint64_t stream_raw
) {
    if (M_pad <= 0 || K <= 0) return 0;
    if ((sx_row & 7) != 0 || (reinterpret_cast<uintptr_t>(x_ptr) & 15) != 0) return -1;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_raw);

    if (K == 768) {
        dim3 grid(static_cast<unsigned>((M_pad + 15) / 16));
        dim3 block(256);
        quant_fp8_rowwise_k768_v4_kernel<<<grid, block, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x_ptr),
            static_cast<uint8_t*>(y_fp8_ptr),
            static_cast<float*>(y_scale_ptr),
            M, M_pad, sx_row
        );
    } else if ((K & 127) == 0) {
        dim3 grid(static_cast<unsigned>((M_pad + 1) / 2));
        dim3 block(256);
        quant_fp8_rowwise_gen128_v4_kernel<<<grid, block, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x_ptr),
            static_cast<uint8_t*>(y_fp8_ptr),
            static_cast<float*>(y_scale_ptr),
            M, M_pad, K, sx_row
        );
    } else {
        return -2;
    }
    return (cudaGetLastError() == cudaSuccess) ? 0 : -3;
}
