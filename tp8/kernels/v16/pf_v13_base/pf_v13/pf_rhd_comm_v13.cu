// SPDX-License-Identifier: Apache-2.0
// =============================================================================
// SM121a (NVIDIA DGX Spark GB10) Wave-V11 Prefill-Only Compressed Comm &
// Fused ShortConv+AttnRes+FP8-AG-Quant CUDA Kernels (`libpf_rhd_comm_v11.so`)
// =============================================================================
//
// 100% ABI-compatible superset of `libpf_rhd_comm_v5.so` & `libpf_rhd_comm_sm121.so`.
// Strictly prefill-only (guarded by `!is_stream_capturing(stream)` returning -4).
// Zero effect on C=1 / C=32 decode (`exact_ink.py` 1dcfb306 100% untouched).
//
// Wave-V11 Additions:
//   1. `fused_sconv_ar_fp8_ag_v11_fwd`:
//      - 0-ULP bit-exact BLOCK_D=1024 Order-A ShortConv (W=4) + AttnRes + RMSNorm
//      - Fused `DiscardBlkPtr` skip (eliminates 14.16 MB/call dead ConvCache writes!)
//      - Fused Direct FP8-E4M3FN rowwise quant for query 0 (`OutFp8`, `OutScale`),
//        rounding through BF16 in registers -> 0-ULP identical to unfused
//        `fused_sconv_ar_prefill` + `rhd_ag_quant_fp8_rowwise_lowsm_fwd` while
//        eliminating 28.3 MB/call BF16 round-trip & 8,256 quant launches!
//      - Persistent grid-stride `max_ctas` throttle (6-8 CTAs Half-2, 24-48 Half-1).
//   2. `a2a_quant_{int8,fp8}_g64_slice_lowsm_fwd` &
//      `a2a_dequant_sum8_{int8,fp8}_g64_slice_lowsm_fwd`:
//      - Row-sliced `[row_beg, row_beg + row_cnt)` + flexible `self_row_base`
//        (supports 2-stage overlapped MoE A2A pipelining, pre-quantized `wo_ud`
//        local-shard `[S, H]` Attention A2A, and compute-stream NIC-Pure Half-1!).
//
// Build on GB10 (SM121a — strict flags, 0 B stack / 0 B spills):
//   /usr/local/cuda/bin/nvcc -O3 -std=c++17 --fmad=false --shared -Xcompiler -fPIC \
//       -gencode arch=compute_121a,code=sm_121a --ptxas-options=-v \
//       pf_rhd_comm_v11.cu -o libpf_rhd_comm_v11.so
// =============================================================================

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <math.h>

namespace {

constexpr int kGroupSize = 64;
constexpr int kMaxSmemGroups = 256;  // Supports up to H = 16,384 in SMEM for legacy 1-CTA/row kernels

// ---------------------------------------------------------------------------
// Low-level SM121a PTX Load / Store & Quantization Helpers
// ---------------------------------------------------------------------------
static __device__ __forceinline__ uint2 ld_global_cs_u2(const void* p) {
    uint2 v;
    asm volatile("ld.global.cs.v2.u32 {%0,%1}, [%2];"
                 : "=r"(v.x), "=r"(v.y) : "l"(p));
    return v;
}

static __device__ __forceinline__ uint2 ld_global_ca_u2(const void* p) {
    uint2 v;
    asm volatile("ld.global.ca.v2.u32 {%0,%1}, [%2];"
                 : "=r"(v.x), "=r"(v.y) : "l"(p));
    return v;
}

static __device__ __forceinline__ uint32_t ld_global_cs_u32(const void* p) {
    uint32_t v;
    asm volatile("ld.global.cs.u32 %0, [%1];"
                 : "=r"(v) : "l"(p));
    return v;
}

static __device__ __forceinline__ uint4 ld_global_cs_u4(const void* p) {
    uint4 v;
    asm volatile("ld.global.cs.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}

static __device__ __forceinline__ void st_global_cs_u32(void* p, uint32_t v) {
    asm volatile("st.global.cs.u32 [%0], %1;"
                 :: "l"(p), "r"(v) : "memory");
}

static __device__ __forceinline__ void st_global_cs_u2(void* p, uint2 v) {
    asm volatile("st.global.cs.v2.u32 [%0], {%1,%2};"
                 :: "l"(p), "r"(v.x), "r"(v.y) : "memory");
}

static __device__ __forceinline__ void st_global_cs_u4(void* p, uint4 v) {
    asm volatile("st.global.cs.v4.u32 [%0], {%1,%2,%3,%4};"
                 :: "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}

static __device__ __forceinline__ void st_global_wb_u2(void* p, uint2 v) {
    asm volatile("st.global.v2.u32 [%0], {%1,%2};"
                 :: "l"(p), "r"(v.x), "r"(v.y) : "memory");
}

static __device__ __forceinline__ void st_global_wb_u4(void* p, uint4 v) {
    asm volatile("st.global.v4.u32 [%0], {%1,%2,%3,%4};"
                 :: "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}

static __device__ __forceinline__ uint16_t bf16_to_u16_bits(__nv_bfloat16 v) {
    return *reinterpret_cast<const uint16_t*>(&v);
}

static __device__ __forceinline__ float u16_bits_to_bf16_f32(uint16_t bits) {
    __nv_bfloat16 b = *reinterpret_cast<const __nv_bfloat16*>(&bits);
    return __bfloat162float(b);
}

static __device__ __forceinline__ float halfwarp_max_f32(float v) {
    v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, 8));
    v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, 4));
    v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, 2));
    v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, 1));
    return v;
}

static __device__ __forceinline__ float warp_max_f32(float v) {
    v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, 16));
    return halfwarp_max_f32(v);
}

static __device__ __forceinline__ uint16_t f32x2_to_e4m3x2(float a, float b) {
    uint16_t out;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
        : "=h"(out) : "f"(b), "f"(a));
    return out;
}

static __device__ __forceinline__ float2 e4m3x2_to_f32x2(uint16_t packed) {
    uint32_t h2_bits;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;"
        : "=r"(h2_bits) : "h"(packed));
    __half2 h2 = *reinterpret_cast<const __half2*>(&h2_bits);
    return __half22float2(h2);
}

static __device__ __forceinline__ uint32_t f32x4_to_e4m3x4(
    float v0, float v1, float v2, float v3
) {
    uint16_t lo = f32x2_to_e4m3x2(v0, v1);
    uint16_t hi = f32x2_to_e4m3x2(v2, v3);
    return (uint32_t)lo | ((uint32_t)hi << 16);
}

static __device__ __forceinline__ void e4m3x4_to_f32x4(
    uint32_t q4, float& o0, float& o1, float& o2, float& o3
) {
    float2 f01 = e4m3x2_to_f32x2((uint16_t)(q4 & 0xFFFFu));
    float2 f23 = e4m3x2_to_f32x2((uint16_t)(q4 >> 16));
    o0 = f01.x; o1 = f01.y; o2 = f23.x; o3 = f23.y;
}

static __device__ __forceinline__ uint32_t f32x4_to_int8x4(
    float v0, float v1, float v2, float v3
) {
    int i0 = __float2int_rn(fminf(fmaxf(v0, -127.0f), 127.0f));
    int i1 = __float2int_rn(fminf(fmaxf(v1, -127.0f), 127.0f));
    int i2 = __float2int_rn(fminf(fmaxf(v2, -127.0f), 127.0f));
    int i3 = __float2int_rn(fminf(fmaxf(v3, -127.0f), 127.0f));
    uint32_t b0 = (uint32_t)(uint8_t)(int8_t)i0;
    uint32_t b1 = (uint32_t)(uint8_t)(int8_t)i1;
    uint32_t b2 = (uint32_t)(uint8_t)(int8_t)i2;
    uint32_t b3 = (uint32_t)(uint8_t)(int8_t)i3;
    return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
}

static __device__ __forceinline__ void int8x4_to_f32x4(
    uint32_t q4, float& o0, float& o1, float& o2, float& o3
) {
    o0 = (float)(int8_t)(q4 & 0xFFu);
    o1 = (float)(int8_t)((q4 >> 8) & 0xFFu);
    o2 = (float)(int8_t)((q4 >> 16) & 0xFFu);
    o3 = (float)(int8_t)((q4 >> 24) & 0xFFu);
}

static __device__ __forceinline__ void bf16x4_to_f32x4(
    uint2 raw, float& o0, float& o1, float& o2, float& o3
) {
    __nv_bfloat162 h0 = *reinterpret_cast<const __nv_bfloat162*>(&raw.x);
    __nv_bfloat162 h1 = *reinterpret_cast<const __nv_bfloat162*>(&raw.y);
    float2 f01 = __bfloat1622float2(h0);
    float2 f23 = __bfloat1622float2(h1);
    o0 = f01.x; o1 = f01.y; o2 = f23.x; o3 = f23.y;
}

static __device__ __forceinline__ uint2 f32x4_to_bf16_u2(
    float v0, float v1, float v2, float v3
) {
    __nv_bfloat162 h0 = __halves2bfloat162(__float2bfloat16_rn(v0), __float2bfloat16_rn(v1));
    __nv_bfloat162 h1 = __halves2bfloat162(__float2bfloat16_rn(v2), __float2bfloat16_rn(v3));
    uint2 out;
    out.x = *reinterpret_cast<const uint32_t*>(&h0);
    out.y = *reinterpret_cast<const uint32_t*>(&h1);
    return out;
}

template <bool IS_FP8>
static __device__ __forceinline__ uint32_t quant_group64_4vals(
    float v0, float v1, float v2, float v3,
    __nv_bfloat16& sc_bf_out
) {
    constexpr float inv_qmax = IS_FP8 ? (1.0f / 448.0f) : (1.0f / 127.0f);
    float a = fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3)));
    float amax = halfwarp_max_f32(a);
    float raw_sc = fmaxf(amax * inv_qmax, 1.0e-12f);
    __nv_bfloat16 sc_bf = __float2bfloat16_rn(raw_sc);
    sc_bf_out = sc_bf;
    float inv_sc = 1.0f / __bfloat162float(sc_bf);
    if constexpr (IS_FP8) {
        return f32x4_to_e4m3x4(v0 * inv_sc, v1 * inv_sc, v2 * inv_sc, v3 * inv_sc);
    } else {
        return f32x4_to_int8x4(v0 * inv_sc, v1 * inv_sc, v2 * inv_sc, v3 * inv_sc);
    }
}

template <bool IS_FP8>
static __device__ __forceinline__ void dequant_4vals(
    uint32_t q4, float sc, float& o0, float& o1, float& o2, float& o3
) {
    float q0, q1, q2, q3;
    if constexpr (IS_FP8) {
        e4m3x4_to_f32x4(q4, q0, q1, q2, q3);
    } else {
        int8x4_to_f32x4(q4, q0, q1, q2, q3);
    }
    o0 = q0 * sc;
    o1 = q1 * sc;
    o2 = q2 * sc;
    o3 = q3 * sc;
}

// Zero-SMEM, zero-barrier warp-pair BF16 scale store & load helpers:
// Requires `g_pair` even and 4-byte aligned `sc_row + g_pair` (guaranteed whenever `H % 128 == 0`
// and `scale_off % 4 == 0` and buffer stride `% 4 == 0`).
static __device__ __forceinline__ void store_scale_pair_cs(
    __nv_bfloat16* sc_row_g_pair, __nv_bfloat16 sc_bf, int lane
) {
    const uint32_t my_u16 = (uint32_t)bf16_to_u16_bits(sc_bf);
    const uint32_t hi_u16 = __shfl_sync(0xFFFFFFFFu, my_u16, 16);
    if (lane == 0) {
        st_global_cs_u32(sc_row_g_pair, my_u16 | (hi_u16 << 16));
    }
}

static __device__ __forceinline__ float load_scale_pair_cs(
    const __nv_bfloat16* sc_row_g_pair, int lane, int halfw
) {
    uint32_t pair_bits = 0u;
    if (lane == 0) {
        pair_bits = ld_global_cs_u32(sc_row_g_pair);
    }
    pair_bits = __shfl_sync(0xFFFFFFFFu, pair_bits, 0);
    const uint16_t my_u16 = halfw
        ? (uint16_t)(pair_bits >> 16)
        : (uint16_t)(pair_bits & 0xFFFFu);
    return u16_bits_to_bf16_f32(my_u16);
}

static inline unsigned clamp_lowsm_ctas(int total_rows, int max_ctas) {
    if (max_ctas > 0 && max_ctas < total_rows) {
        return static_cast<unsigned>(max_ctas);
    }
    return static_cast<unsigned>(total_rows);
}

// ===========================================================================
// PART 1: Backwards-Compatible Full-Grid Kernels (Half-1 Comm-Bound Path)
// ===========================================================================

template <bool IS_FP8>
__global__ __launch_bounds__(128)
void rhd_quant_slice_g64_kernel(
    const __nv_bfloat16* __restrict__ X,
    uint8_t* __restrict__ OUT,
    int64_t stride_xm,
    int64_t row_off,
    int64_t scale_off,
    int H
) {
    const int row      = (int)blockIdx.x;
    const int tid      = (int)threadIdx.x;
    const int warp     = tid >> 5;
    const int lane     = tid & 31;
    const int halfw    = (lane >> 4) & 1;
    const int sublane  = lane & 15;
    const int G        = H / kGroupSize;

    const __nv_bfloat16* x_row = X + (row_off + (int64_t)row) * stride_xm;
    uint8_t* q_row             = OUT + (int64_t)row * H;
    __nv_bfloat16* sc_row      = reinterpret_cast<__nv_bfloat16*>(
        OUT + scale_off + (int64_t)row * G * 2
    );

    __shared__ __nv_bfloat16 sSc[kMaxSmemGroups];
    const bool use_smem = (G <= kMaxSmemGroups);

    for (int g_pair = warp * 2; g_pair < G; g_pair += 8) {
        const int g = g_pair + halfw;
        const int d = (g * kGroupSize) + (sublane << 2);
        uint2 raw = ld_global_cs_u2(x_row + d);
        float v0, v1, v2, v3;
        bf16x4_to_f32x4(raw, v0, v1, v2, v3);

        __nv_bfloat16 sc_bf;
        uint32_t q4 = quant_group64_4vals<IS_FP8>(v0, v1, v2, v3, sc_bf);
        st_global_cs_u32(q_row + d, q4);

        if (sublane == 0) {
            if (use_smem) sSc[g] = sc_bf;
            else sc_row[g] = sc_bf;
        }
    }

    if (use_smem) {
        __syncthreads();
        for (int i = tid; i < G; i += 128) {
            sc_row[i] = sSc[i];
        }
    }
}

template <bool IS_FP8>
__global__ __launch_bounds__(128)
void rhd_dequant_add_split_quant_g64_kernel(
    const uint8_t* __restrict__ RECV,
    const __nv_bfloat16* __restrict__ LOCAL,
    uint8_t* __restrict__ NEXT_SEND,
    __nv_bfloat16* __restrict__ NEXT_KEEP,
    int64_t stride_local,
    int64_t local_row_off,
    int64_t recv_scale_off,
    int64_t next_scale_off,
    int64_t stride_keep,
    int N,
    int H,
    int send_upper,
    int wb_keep
) {
    const int row      = (int)blockIdx.x;
    const int tid      = (int)threadIdx.x;
    const int warp     = tid >> 5;
    const int lane     = tid & 31;
    const int halfw    = (lane >> 4) & 1;
    const int sublane  = lane & 15;
    const int G        = H / kGroupSize;

    const bool is_send = send_upper ? (row >= N) : (row < N);
    const int out_row  = is_send
        ? (send_upper ? (row - N) : row)
        : (send_upper ? row : (row - N));

    const uint8_t* recv_q_row      = RECV + (int64_t)row * H;
    const __nv_bfloat16* recv_sc_p = reinterpret_cast<const __nv_bfloat16*>(
        RECV + recv_scale_off + (int64_t)row * G * 2
    );
    const __nv_bfloat16* loc_row   = LOCAL + (local_row_off + (int64_t)row) * stride_local;

    __shared__ float sRecvSc[kMaxSmemGroups];
    __shared__ __nv_bfloat16 sSendSc[kMaxSmemGroups];
    const bool use_smem = (G <= kMaxSmemGroups);
    if (use_smem) {
        for (int i = tid; i < G; i += 128) {
            sRecvSc[i] = __bfloat162float(recv_sc_p[i]);
        }
        __syncthreads();
    }

    const bool ca_local = ((int64_t)N * (int64_t)H * 4 <= (20LL << 20));

    if (is_send) {
        uint8_t* send_q_row      = NEXT_SEND + (int64_t)out_row * H;
        __nv_bfloat16* send_sc_p = reinterpret_cast<__nv_bfloat16*>(
            NEXT_SEND + next_scale_off + (int64_t)out_row * G * 2
        );
        for (int g_pair = warp * 2; g_pair < G; g_pair += 8) {
            const int g = g_pair + halfw;
            const int d = (g * kGroupSize) + (sublane << 2);
            float rsc = use_smem ? sRecvSc[g] : __bfloat162float(recv_sc_p[g]);

            uint2 raw_loc = ca_local ? ld_global_ca_u2(loc_row + d) : ld_global_cs_u2(loc_row + d);
            uint32_t q_in = ld_global_cs_u32(recv_q_row + d);

            float l0, l1, l2, l3, r0, r1, r2, r3;
            bf16x4_to_f32x4(raw_loc, l0, l1, l2, l3);
            dequant_4vals<IS_FP8>(q_in, rsc, r0, r1, r2, r3);

            __nv_bfloat16 sc_out;
            uint32_t q_out = quant_group64_4vals<IS_FP8>(
                l0 + r0, l1 + r1, l2 + r2, l3 + r3, sc_out);
            st_global_cs_u32(send_q_row + d, q_out);

            if (sublane == 0) {
                if (use_smem) sSendSc[g] = sc_out;
                else send_sc_p[g] = sc_out;
            }
        }
        if (use_smem) {
            __syncthreads();
            for (int i = tid; i < G; i += 128) {
                send_sc_p[i] = sSendSc[i];
            }
        }
    } else {
        __nv_bfloat16* keep_row = NEXT_KEEP + (int64_t)out_row * stride_keep;
        for (int g_pair = warp * 2; g_pair < G; g_pair += 8) {
            const int g = g_pair + halfw;
            const int d = (g * kGroupSize) + (sublane << 2);
            float rsc = use_smem ? sRecvSc[g] : __bfloat162float(recv_sc_p[g]);

            uint2 raw_loc = ca_local ? ld_global_ca_u2(loc_row + d) : ld_global_cs_u2(loc_row + d);
            uint32_t q_in = ld_global_cs_u32(recv_q_row + d);

            float l0, l1, l2, l3, r0, r1, r2, r3;
            bf16x4_to_f32x4(raw_loc, l0, l1, l2, l3);
            dequant_4vals<IS_FP8>(q_in, rsc, r0, r1, r2, r3);

            uint2 out_bf = f32x4_to_bf16_u2(l0 + r0, l1 + r1, l2 + r2, l3 + r3);
            if (wb_keep) st_global_wb_u2(keep_row + d, out_bf);
            else         st_global_cs_u2(keep_row + d, out_bf);
        }
    }
}

template <bool IS_FP8>
__global__ __launch_bounds__(128)
void rhd_dequant_add_quant_g64_kernel(
    const uint8_t* __restrict__ RECV,
    const __nv_bfloat16* __restrict__ LOCAL,
    uint8_t* __restrict__ NEXT_SEND,
    int64_t stride_local,
    int64_t local_row_off,
    int64_t recv_scale_off,
    int64_t next_scale_off,
    int H
) {
    const int row      = (int)blockIdx.x;
    const int tid      = (int)threadIdx.x;
    const int warp     = tid >> 5;
    const int lane     = tid & 31;
    const int halfw    = (lane >> 4) & 1;
    const int sublane  = lane & 15;
    const int G        = H / kGroupSize;

    const uint8_t* recv_q_row      = RECV + (int64_t)row * H;
    const __nv_bfloat16* recv_sc_p = reinterpret_cast<const __nv_bfloat16*>(
        RECV + recv_scale_off + (int64_t)row * G * 2
    );
    const __nv_bfloat16* loc_row   = LOCAL + (local_row_off + (int64_t)row) * stride_local;
    uint8_t* send_q_row            = NEXT_SEND + (int64_t)row * H;
    __nv_bfloat16* send_sc_p       = reinterpret_cast<__nv_bfloat16*>(
        NEXT_SEND + next_scale_off + (int64_t)row * G * 2
    );

    __shared__ float sRecvSc[kMaxSmemGroups];
    __shared__ __nv_bfloat16 sSendSc[kMaxSmemGroups];
    const bool use_smem = (G <= kMaxSmemGroups);
    if (use_smem) {
        for (int i = tid; i < G; i += 128) {
            sRecvSc[i] = __bfloat162float(recv_sc_p[i]);
        }
        __syncthreads();
    }

    for (int g_pair = warp * 2; g_pair < G; g_pair += 8) {
        const int g = g_pair + halfw;
        const int d = (g * kGroupSize) + (sublane << 2);
        float rsc = use_smem ? sRecvSc[g] : __bfloat162float(recv_sc_p[g]);

        uint2 raw_loc = ld_global_cs_u2(loc_row + d);
        uint32_t q_in = ld_global_cs_u32(recv_q_row + d);

        float l0, l1, l2, l3, r0, r1, r2, r3;
        bf16x4_to_f32x4(raw_loc, l0, l1, l2, l3);
        dequant_4vals<IS_FP8>(q_in, rsc, r0, r1, r2, r3);

        __nv_bfloat16 sc_out;
        uint32_t q_out = quant_group64_4vals<IS_FP8>(l0 + r0, l1 + r1, l2 + r2, l3 + r3, sc_out);
        st_global_cs_u32(send_q_row + d, q_out);

        if (sublane == 0) {
            if (use_smem) sSendSc[g] = sc_out;
            else send_sc_p[g] = sc_out;
        }
    }

    if (use_smem) {
        __syncthreads();
        for (int i = tid; i < G; i += 128) {
            send_sc_p[i] = sSendSc[i];
        }
    }
}

template <bool IS_FP8>
__global__ __launch_bounds__(128)
void rhd_dequant_add_bf16_g64_kernel(
    const uint8_t* __restrict__ RECV,
    const __nv_bfloat16* __restrict__ LOCAL,
    __nv_bfloat16* __restrict__ OUT,
    int64_t stride_local,
    int64_t local_row_off,
    int64_t recv_scale_off,
    int64_t stride_out,
    int H,
    int wb_out
) {
    const int row      = (int)blockIdx.x;
    const int tid      = (int)threadIdx.x;
    const int warp     = tid >> 5;
    const int lane     = tid & 31;
    const int halfw    = (lane >> 4) & 1;
    const int sublane  = lane & 15;
    const int G        = H / kGroupSize;

    const uint8_t* recv_q_row      = RECV + (int64_t)row * H;
    const __nv_bfloat16* recv_sc_p = reinterpret_cast<const __nv_bfloat16*>(
        RECV + recv_scale_off + (int64_t)row * G * 2
    );
    const __nv_bfloat16* loc_row   = LOCAL + (local_row_off + (int64_t)row) * stride_local;
    __nv_bfloat16* out_row         = OUT + (int64_t)row * stride_out;

    __shared__ float sRecvSc[kMaxSmemGroups];
    const bool use_smem = (G <= kMaxSmemGroups);
    if (use_smem) {
        for (int i = tid; i < G; i += 128) {
            sRecvSc[i] = __bfloat162float(recv_sc_p[i]);
        }
        __syncthreads();
    }

    for (int g_pair = warp * 2; g_pair < G; g_pair += 8) {
        const int g = g_pair + halfw;
        const int d = (g * kGroupSize) + (sublane << 2);
        float rsc = use_smem ? sRecvSc[g] : __bfloat162float(recv_sc_p[g]);

        uint2 raw_loc = wb_out ? ld_global_ca_u2(loc_row + d) : ld_global_cs_u2(loc_row + d);
        uint32_t q_in = ld_global_cs_u32(recv_q_row + d);

        float l0, l1, l2, l3, r0, r1, r2, r3;
        bf16x4_to_f32x4(raw_loc, l0, l1, l2, l3);
        dequant_4vals<IS_FP8>(q_in, rsc, r0, r1, r2, r3);

        uint2 out_bf = f32x4_to_bf16_u2(l0 + r0, l1 + r1, l2 + r2, l3 + r3);
        if (wb_out) st_global_wb_u2(out_row + d, out_bf);
        else        st_global_cs_u2(out_row + d, out_bf);
    }
}

template <bool IS_FP8>
__global__ __launch_bounds__(128)
void rhd_ag_dequant_g64_kernel(
    const uint8_t* __restrict__ AG_BUF,
    __nv_bfloat16* __restrict__ OUT,
    int64_t shard_bytes,
    int64_t scale_off,
    int64_t stride_out,
    int S,
    int H
) {
    const int local_row = (int)blockIdx.x;
    const int r         = (int)blockIdx.y;
    const int tid       = (int)threadIdx.x;
    const int warp      = tid >> 5;
    const int lane      = tid & 31;
    const int halfw     = (lane >> 4) & 1;
    const int sublane   = lane & 15;
    const int G         = H / kGroupSize;

    const uint8_t* slot_base = AG_BUF + (int64_t)r * shard_bytes;
    const uint8_t* q_row     = slot_base + (int64_t)local_row * H;
    const __nv_bfloat16* sc_p = reinterpret_cast<const __nv_bfloat16*>(
        slot_base + scale_off + (int64_t)local_row * G * 2
    );
    __nv_bfloat16* out_row   = OUT + ((int64_t)r * S + local_row) * stride_out;

    __shared__ float sSc[kMaxSmemGroups];
    const bool use_smem = (G <= kMaxSmemGroups);
    if (use_smem) {
        for (int i = tid; i < G; i += 128) {
            sSc[i] = __bfloat162float(sc_p[i]);
        }
        __syncthreads();
    }

    for (int g_pair = warp * 2; g_pair < G; g_pair += 8) {
        const int g = g_pair + halfw;
        const int d = (g * kGroupSize) + (sublane << 2);
        float sc = use_smem ? sSc[g] : __bfloat162float(sc_p[g]);

        uint32_t q4 = ld_global_cs_u32(q_row + d);
        float v0, v1, v2, v3;
        dequant_4vals<IS_FP8>(q4, sc, v0, v1, v2, v3);
        st_global_wb_u2(out_row + d, f32x4_to_bf16_u2(v0, v1, v2, v3));
    }
}

__global__ __launch_bounds__(128)
void rhd_ag_quant_fp8_rowwise_kernel(
    const __nv_bfloat16* __restrict__ X,
    uint8_t* __restrict__ OUT_SHARD,
    int64_t stride_xm,
    int64_t scale_off,
    int H
) {
    const int row    = (int)blockIdx.x;
    const int tid    = (int)threadIdx.x;
    const int warp   = tid >> 5;
    const int lane   = tid & 31;
    const int n_vec4 = H >> 2;

    const __nv_bfloat16* x_row = X + (int64_t)row * stride_xm;
    uint8_t* q_row             = OUT_SHARD + (int64_t)row * H;
    float* sc_out              = reinterpret_cast<float*>(OUT_SHARD + scale_off);

    __shared__ float sWarpMax[4];

    if (n_vec4 <= 1280) {
        // Explicit scalar register cache (ZERO local arrays -> guaranteed 0 B stack frame!)
        uint2 c0 = make_uint2(0u, 0u), c1 = make_uint2(0u, 0u);
        uint2 c2 = make_uint2(0u, 0u), c3 = make_uint2(0u, 0u);
        uint2 c4 = make_uint2(0u, 0u), c5 = make_uint2(0u, 0u);
        uint2 c6 = make_uint2(0u, 0u), c7 = make_uint2(0u, 0u);
        uint2 c8 = make_uint2(0u, 0u), c9 = make_uint2(0u, 0u);
        float local_amax = 0.0f;

        #define LOAD_ROW_VEC4_128(IT, CREG) \
            do { \
                const int idx = (IT) * 128 + tid; \
                if (idx < n_vec4) { \
                    const uint2 raw = ld_global_cs_u2(x_row + (idx << 2)); \
                    (CREG) = raw; \
                    float v0, v1, v2, v3; \
                    bf16x4_to_f32x4(raw, v0, v1, v2, v3); \
                    float a = fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3))); \
                    local_amax = fmaxf(local_amax, a); \
                } \
            } while (0)

        LOAD_ROW_VEC4_128(0, c0); LOAD_ROW_VEC4_128(1, c1);
        LOAD_ROW_VEC4_128(2, c2); LOAD_ROW_VEC4_128(3, c3);
        LOAD_ROW_VEC4_128(4, c4); LOAD_ROW_VEC4_128(5, c5);
        LOAD_ROW_VEC4_128(6, c6); LOAD_ROW_VEC4_128(7, c7);
        LOAD_ROW_VEC4_128(8, c8); LOAD_ROW_VEC4_128(9, c9);
        #undef LOAD_ROW_VEC4_128

        float wmax = warp_max_f32(local_amax);
        if (lane == 0) sWarpMax[warp] = wmax;
        __syncthreads();
        float row_amax = fmaxf(fmaxf(sWarpMax[0], sWarpMax[1]), fmaxf(sWarpMax[2], sWarpMax[3]));
        float scale = fmaxf(row_amax * (1.0f / 448.0f), 1.0e-12f);
        float inv_scale = 1.0f / scale;
        if (tid == 0) sc_out[row] = scale;

        #define STORE_ROW_VEC4_128(IT, CREG) \
            do { \
                const int idx = (IT) * 128 + tid; \
                if (idx < n_vec4) { \
                    float v0, v1, v2, v3; \
                    bf16x4_to_f32x4((CREG), v0, v1, v2, v3); \
                    uint32_t q4 = f32x4_to_e4m3x4( \
                        v0 * inv_scale, v1 * inv_scale, v2 * inv_scale, v3 * inv_scale); \
                    st_global_cs_u32(q_row + (idx << 2), q4); \
                } \
            } while (0)

        STORE_ROW_VEC4_128(0, c0); STORE_ROW_VEC4_128(1, c1);
        STORE_ROW_VEC4_128(2, c2); STORE_ROW_VEC4_128(3, c3);
        STORE_ROW_VEC4_128(4, c4); STORE_ROW_VEC4_128(5, c5);
        STORE_ROW_VEC4_128(6, c6); STORE_ROW_VEC4_128(7, c7);
        STORE_ROW_VEC4_128(8, c8); STORE_ROW_VEC4_128(9, c9);
        #undef STORE_ROW_VEC4_128
    } else {
        float local_amax = 0.0f;
        for (int idx = tid; idx < n_vec4; idx += 128) {
            uint2 raw = ld_global_ca_u2(x_row + (idx << 2));
            float v0, v1, v2, v3;
            bf16x4_to_f32x4(raw, v0, v1, v2, v3);
            float a = fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3)));
            local_amax = fmaxf(local_amax, a);
        }
        float wmax = warp_max_f32(local_amax);
        if (lane == 0) sWarpMax[warp] = wmax;
        __syncthreads();
        float row_amax = fmaxf(fmaxf(sWarpMax[0], sWarpMax[1]), fmaxf(sWarpMax[2], sWarpMax[3]));
        float scale = fmaxf(row_amax * (1.0f / 448.0f), 1.0e-12f);
        float inv_scale = 1.0f / scale;
        if (tid == 0) sc_out[row] = scale;

        for (int idx = tid; idx < n_vec4; idx += 128) {
            uint2 raw = ld_global_ca_u2(x_row + (idx << 2));
            float v0, v1, v2, v3;
            bf16x4_to_f32x4(raw, v0, v1, v2, v3);
            uint32_t q4 = f32x4_to_e4m3x4(
                v0 * inv_scale, v1 * inv_scale, v2 * inv_scale, v3 * inv_scale);
            st_global_cs_u32(q_row + (idx << 2), q4);
        }
    }
}

__global__ __launch_bounds__(128)
void rhd_ag_unpack_fp8_rowwise_kernel(
    const uint8_t* __restrict__ AG_BUF,
    uint8_t* __restrict__ OUT_FP8,
    float* __restrict__ OUT_SCALE,
    int64_t shard_bytes,
    int64_t scale_off,
    int64_t stride_out,
    int S,
    int H
) {
    const int local_row = (int)blockIdx.x;
    const int r         = (int)blockIdx.y;
    const int tid       = (int)threadIdx.x;
    const int n_vec16   = H >> 4;

    const uint8_t* slot_base = AG_BUF + (int64_t)r * shard_bytes;
    const uint8_t* src_row   = slot_base + (int64_t)local_row * H;
    const int64_t dst_r      = (int64_t)r * S + local_row;
    uint8_t* dst_row         = OUT_FP8 + dst_r * stride_out;

    for (int idx = tid; idx < n_vec16; idx += 128) {
        uint4 v = ld_global_cs_u4(src_row + (idx << 4));
        st_global_wb_u4(dst_row + (idx << 4), v);
    }
    if (tid == 0) {
        const float* src_sc = reinterpret_cast<const float*>(slot_base + scale_off);
        OUT_SCALE[dst_r] = src_sc[local_row];
    }
}

// ===========================================================================
// PART 2: v5 Low-SM (`4..8` CTAs), Zero-SMEM/Zero-Barrier, L2-Streaming Kernels
//   Specifically engineered for Half-2 (`comm_stream` overlapped with `MoE`):
//     - `__launch_bounds__(256, 2)`: <= 8 SMs occupied, leaving >= 40..44 SMs on MoE!
//     - `100% ld.global.cs` + `st.global.cs` when `wb == 0`: < 0.5 MiB L2 footprint!
// ===========================================================================

template <bool IS_FP8>
__global__ __launch_bounds__(256, 2)
void rhd_quant_slice_g64_lowsm_kernel(
    const __nv_bfloat16* __restrict__ X,
    uint8_t* __restrict__ OUT,
    int64_t stride_xm,
    int64_t row_off,
    int64_t scale_off,
    int M,
    int H
) {
    const int tid     = (int)threadIdx.x;
    const int warp    = tid >> 5;           // 0..7
    const int lane    = tid & 31;           // 0..31
    const int halfw   = (lane >> 4) & 1;    // 0 or 1
    const int sublane = lane & 15;          // 0..15
    const int G       = H / kGroupSize;     // even because H % 128 == 0

    for (int row = (int)blockIdx.x; row < M; row += (int)gridDim.x) {
        const __nv_bfloat16* x_row = X + (row_off + (int64_t)row) * stride_xm;
        uint8_t* q_row             = OUT + (int64_t)row * H;
        __nv_bfloat16* sc_row      = reinterpret_cast<__nv_bfloat16*>(
            OUT + scale_off + (int64_t)row * G * 2
        );

        for (int g_pair = warp * 2; g_pair < G; g_pair += 16) {
            const int g = g_pair + halfw;
            const int d = (g * kGroupSize) + (sublane << 2);
            uint2 raw = ld_global_cs_u2(x_row + d);
            float v0, v1, v2, v3;
            bf16x4_to_f32x4(raw, v0, v1, v2, v3);

            __nv_bfloat16 sc_bf;
            uint32_t q4 = quant_group64_4vals<IS_FP8>(v0, v1, v2, v3, sc_bf);
            st_global_cs_u32(q_row + d, q4);
            store_scale_pair_cs(sc_row + g_pair, sc_bf, lane);
        }
    }
}

template <bool IS_FP8>
__global__ __launch_bounds__(256, 2)
void rhd_dequant_add_split_quant_g64_lowsm_kernel(
    const uint8_t* __restrict__ RECV,
    const __nv_bfloat16* __restrict__ LOCAL,
    uint8_t* __restrict__ NEXT_SEND,
    __nv_bfloat16* __restrict__ NEXT_KEEP,
    int64_t stride_local,
    int64_t local_row_off,
    int64_t recv_scale_off,
    int64_t next_scale_off,
    int64_t stride_keep,
    int N,
    int H,
    int send_upper,
    int wb_keep
) {
    const int tid      = (int)threadIdx.x;
    const int warp     = tid >> 5;
    const int lane     = tid & 31;
    const int halfw    = (lane >> 4) & 1;
    const int sublane  = lane & 15;
    const int G        = H / kGroupSize;
    const int tot_rows = 2 * N;
    // Strict L2 streaming when wb_keep == 0: never cache LOCAL or NEXT_KEEP in GB10's 24 MiB L2!
    const bool ca_local = (wb_keep != 0) && ((int64_t)N * (int64_t)H * 4 <= (20LL << 20));

    for (int row = (int)blockIdx.x; row < tot_rows; row += (int)gridDim.x) {
        const bool is_send = send_upper ? (row >= N) : (row < N);
        const int out_row  = is_send
            ? (send_upper ? (row - N) : row)
            : (send_upper ? row : (row - N));

        const uint8_t* recv_q_row      = RECV + (int64_t)row * H;
        const __nv_bfloat16* recv_sc_p = reinterpret_cast<const __nv_bfloat16*>(
            RECV + recv_scale_off + (int64_t)row * G * 2
        );
        const __nv_bfloat16* loc_row   = LOCAL + (local_row_off + (int64_t)row) * stride_local;

        if (is_send) {
            uint8_t* send_q_row      = NEXT_SEND + (int64_t)out_row * H;
            __nv_bfloat16* send_sc_p = reinterpret_cast<__nv_bfloat16*>(
                NEXT_SEND + next_scale_off + (int64_t)out_row * G * 2
            );
            for (int g_pair = warp * 2; g_pair < G; g_pair += 16) {
                const int g = g_pair + halfw;
                const int d = (g * kGroupSize) + (sublane << 2);
                const float rsc = load_scale_pair_cs(recv_sc_p + g_pair, lane, halfw);

                uint2 raw_loc = ca_local ? ld_global_ca_u2(loc_row + d) : ld_global_cs_u2(loc_row + d);
                uint32_t q_in = ld_global_cs_u32(recv_q_row + d);

                float l0, l1, l2, l3, r0, r1, r2, r3;
                bf16x4_to_f32x4(raw_loc, l0, l1, l2, l3);
                dequant_4vals<IS_FP8>(q_in, rsc, r0, r1, r2, r3);

                __nv_bfloat16 sc_out;
                uint32_t q_out = quant_group64_4vals<IS_FP8>(
                    l0 + r0, l1 + r1, l2 + r2, l3 + r3, sc_out);
                st_global_cs_u32(send_q_row + d, q_out);
                store_scale_pair_cs(send_sc_p + g_pair, sc_out, lane);
            }
        } else {
            __nv_bfloat16* keep_row = NEXT_KEEP + (int64_t)out_row * stride_keep;
            for (int g_pair = warp * 2; g_pair < G; g_pair += 16) {
                const int g = g_pair + halfw;
                const int d = (g * kGroupSize) + (sublane << 2);
                const float rsc = load_scale_pair_cs(recv_sc_p + g_pair, lane, halfw);

                uint2 raw_loc = ca_local ? ld_global_ca_u2(loc_row + d) : ld_global_cs_u2(loc_row + d);
                uint32_t q_in = ld_global_cs_u32(recv_q_row + d);

                float l0, l1, l2, l3, r0, r1, r2, r3;
                bf16x4_to_f32x4(raw_loc, l0, l1, l2, l3);
                dequant_4vals<IS_FP8>(q_in, rsc, r0, r1, r2, r3);

                uint2 out_bf = f32x4_to_bf16_u2(l0 + r0, l1 + r1, l2 + r2, l3 + r3);
                if (wb_keep) st_global_wb_u2(keep_row + d, out_bf);
                else         st_global_cs_u2(keep_row + d, out_bf);
            }
        }
    }
}

template <bool IS_FP8>
__global__ __launch_bounds__(256, 2)
void rhd_dequant_add_quant_g64_lowsm_kernel(
    const uint8_t* __restrict__ RECV,
    const __nv_bfloat16* __restrict__ LOCAL,
    uint8_t* __restrict__ NEXT_SEND,
    int64_t stride_local,
    int64_t local_row_off,
    int64_t recv_scale_off,
    int64_t next_scale_off,
    int M,
    int H
) {
    const int tid     = (int)threadIdx.x;
    const int warp    = tid >> 5;
    const int lane    = tid & 31;
    const int halfw   = (lane >> 4) & 1;
    const int sublane = lane & 15;
    const int G       = H / kGroupSize;

    for (int row = (int)blockIdx.x; row < M; row += (int)gridDim.x) {
        const uint8_t* recv_q_row      = RECV + (int64_t)row * H;
        const __nv_bfloat16* recv_sc_p = reinterpret_cast<const __nv_bfloat16*>(
            RECV + recv_scale_off + (int64_t)row * G * 2
        );
        const __nv_bfloat16* loc_row   = LOCAL + (local_row_off + (int64_t)row) * stride_local;
        uint8_t* send_q_row            = NEXT_SEND + (int64_t)row * H;
        __nv_bfloat16* send_sc_p       = reinterpret_cast<__nv_bfloat16*>(
            NEXT_SEND + next_scale_off + (int64_t)row * G * 2
        );

        for (int g_pair = warp * 2; g_pair < G; g_pair += 16) {
            const int g = g_pair + halfw;
            const int d = (g * kGroupSize) + (sublane << 2);
            const float rsc = load_scale_pair_cs(recv_sc_p + g_pair, lane, halfw);

            uint2 raw_loc = ld_global_cs_u2(loc_row + d);
            uint32_t q_in = ld_global_cs_u32(recv_q_row + d);

            float l0, l1, l2, l3, r0, r1, r2, r3;
            bf16x4_to_f32x4(raw_loc, l0, l1, l2, l3);
            dequant_4vals<IS_FP8>(q_in, rsc, r0, r1, r2, r3);

            __nv_bfloat16 sc_out;
            uint32_t q_out = quant_group64_4vals<IS_FP8>(
                l0 + r0, l1 + r1, l2 + r2, l3 + r3, sc_out);
            st_global_cs_u32(send_q_row + d, q_out);
            store_scale_pair_cs(send_sc_p + g_pair, sc_out, lane);
        }
    }
}

template <bool IS_FP8>
__global__ __launch_bounds__(256, 2)
void rhd_dequant_add_bf16_g64_lowsm_kernel(
    const uint8_t* __restrict__ RECV,
    const __nv_bfloat16* __restrict__ LOCAL,
    __nv_bfloat16* __restrict__ OUT,
    int64_t stride_local,
    int64_t local_row_off,
    int64_t recv_scale_off,
    int64_t stride_out,
    int M,
    int H,
    int wb_out
) {
    const int tid     = (int)threadIdx.x;
    const int warp    = tid >> 5;
    const int lane    = tid & 31;
    const int halfw   = (lane >> 4) & 1;
    const int sublane = lane & 15;
    const int G       = H / kGroupSize;

    for (int row = (int)blockIdx.x; row < M; row += (int)gridDim.x) {
        const uint8_t* recv_q_row      = RECV + (int64_t)row * H;
        const __nv_bfloat16* recv_sc_p = reinterpret_cast<const __nv_bfloat16*>(
            RECV + recv_scale_off + (int64_t)row * G * 2
        );
        const __nv_bfloat16* loc_row   = LOCAL + (local_row_off + (int64_t)row) * stride_local;
        __nv_bfloat16* out_row         = OUT + (int64_t)row * stride_out;

        for (int g_pair = warp * 2; g_pair < G; g_pair += 16) {
            const int g = g_pair + halfw;
            const int d = (g * kGroupSize) + (sublane << 2);
            const float rsc = load_scale_pair_cs(recv_sc_p + g_pair, lane, halfw);

            uint2 raw_loc = wb_out ? ld_global_ca_u2(loc_row + d) : ld_global_cs_u2(loc_row + d);
            uint32_t q_in = ld_global_cs_u32(recv_q_row + d);

            float l0, l1, l2, l3, r0, r1, r2, r3;
            bf16x4_to_f32x4(raw_loc, l0, l1, l2, l3);
            dequant_4vals<IS_FP8>(q_in, rsc, r0, r1, r2, r3);

            uint2 out_bf = f32x4_to_bf16_u2(l0 + r0, l1 + r1, l2 + r2, l3 + r3);
            if (wb_out) st_global_wb_u2(out_row + d, out_bf);
            else        st_global_cs_u2(out_row + d, out_bf);
        }
    }
}

// ---------------------------------------------------------------------------
// Low-SM 1-Hop Shift-Rotated AllToAll ReduceScatter (`a2a_{int8,fp8}_lowsm`)
//   - Quantizes ONLY remote peers (`dst != rank`), keeping local `1/8` shard exact BF16!
//   - Single quantization stage (`+3.5 dB` higher SQNR than 3-hop RHD)
//   - Deterministic ascending FP32 accumulation `r = 0 .. world_size - 1`
// ---------------------------------------------------------------------------
template <bool IS_FP8>
__global__ __launch_bounds__(256, 2)
void a2a_quant_g64_lowsm_kernel(
    const __nv_bfloat16* __restrict__ X,
    uint8_t* __restrict__ SEND_BUF,
    int64_t stride_xm,
    int64_t peer_bytes,
    int64_t scale_off,
    int S,
    int H,
    int world_size,
    int rank
) {
    const int tid        = (int)threadIdx.x;
    const int warp       = tid >> 5;
    const int lane       = tid & 31;
    const int halfw      = (lane >> 4) & 1;
    const int sublane    = lane & 15;
    const int G          = H / kGroupSize;
    const int total_work = (world_size - 1) * S;

    for (int widx = (int)blockIdx.x; widx < total_work; widx += (int)gridDim.x) {
        const int p       = widx / S;
        const int s_row   = widx - p * S;
        const int dst     = (p < rank) ? p : (p + 1);
        const int64_t src = (int64_t)dst * S + s_row;

        const __nv_bfloat16* x_row = X + src * stride_xm;
        uint8_t* peer_base         = SEND_BUF + (int64_t)dst * peer_bytes;
        uint8_t* q_row             = peer_base + (int64_t)s_row * H;
        __nv_bfloat16* sc_row      = reinterpret_cast<__nv_bfloat16*>(
            peer_base + scale_off + (int64_t)s_row * G * 2
        );

        for (int g_pair = warp * 2; g_pair < G; g_pair += 16) {
            const int g = g_pair + halfw;
            const int d = (g * kGroupSize) + (sublane << 2);
            uint2 raw = ld_global_cs_u2(x_row + d);
            float v0, v1, v2, v3;
            bf16x4_to_f32x4(raw, v0, v1, v2, v3);

            __nv_bfloat16 sc_bf;
            uint32_t q4 = quant_group64_4vals<IS_FP8>(v0, v1, v2, v3, sc_bf);
            st_global_cs_u32(q_row + d, q4);
            store_scale_pair_cs(sc_row + g_pair, sc_bf, lane);
        }
    }
}

template <bool IS_FP8>
__global__ __launch_bounds__(256, 2)
void a2a_dequant_sum8_g64_lowsm_kernel(
    const uint8_t* __restrict__ RECV_BUF,
    const __nv_bfloat16* __restrict__ SELF_X,
    __nv_bfloat16* __restrict__ OUT,
    int64_t stride_xm,
    int64_t stride_om,
    int64_t peer_bytes,
    int64_t scale_off,
    int S,
    int H,
    int world_size,
    int rank,
    int wb_out
) {
    const int tid     = (int)threadIdx.x;
    const int warp    = tid >> 5;
    const int lane    = tid & 31;
    const int halfw   = (lane >> 4) & 1;
    const int sublane = lane & 15;
    const int G       = H / kGroupSize;

    for (int s_row = (int)blockIdx.x; s_row < S; s_row += (int)gridDim.x) {
        const __nv_bfloat16* self_row = SELF_X + ((int64_t)rank * S + s_row) * stride_xm;
        __nv_bfloat16* out_row        = OUT + (int64_t)s_row * stride_om;

        for (int g_pair = warp * 2; g_pair < G; g_pair += 16) {
            const int g = g_pair + halfw;
            const int d = (g * kGroupSize) + (sublane << 2);

            float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
            for (int r = 0; r < world_size; ++r) {
                if (r == rank) {
                    uint2 raw_self = ld_global_cs_u2(self_row + d);
                    float s0, s1, s2, s3;
                    bf16x4_to_f32x4(raw_self, s0, s1, s2, s3);
                    acc0 += s0; acc1 += s1; acc2 += s2; acc3 += s3;
                } else {
                    const uint8_t* p_base = RECV_BUF + (int64_t)r * peer_bytes;
                    const __nv_bfloat16* sc_p = reinterpret_cast<const __nv_bfloat16*>(
                        p_base + scale_off + (int64_t)s_row * G * 2
                    );
                    const float sc = load_scale_pair_cs(sc_p + g_pair, lane, halfw);
                    const uint32_t q4 = ld_global_cs_u32(p_base + (int64_t)s_row * H + d);
                    float r0, r1, r2, r3;
                    dequant_4vals<IS_FP8>(q4, sc, r0, r1, r2, r3);
                    acc0 += r0; acc1 += r1; acc2 += r2; acc3 += r3;
                }
            }

            uint2 out_bf = f32x4_to_bf16_u2(acc0, acc1, acc2, acc3);
            if (wb_out) st_global_wb_u2(out_row + d, out_bf);
            else        st_global_cs_u2(out_row + d, out_bf);
        }
    }
}

// ---------------------------------------------------------------------------
// Low-SM 1-Collective Packed Rowwise FP8 & g64 AllGather Kernels
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256, 2)
void rhd_ag_quant_fp8_rowwise_lowsm_kernel(
    const __nv_bfloat16* __restrict__ X,
    uint8_t* __restrict__ OUT_SHARD,
    int64_t stride_xm,
    int64_t scale_off,
    int S,
    int H
) {
    const int tid    = (int)threadIdx.x;
    const int warp   = tid >> 5;   // 0..7
    const int lane   = tid & 31;
    const int n_vec4 = H >> 2;     // 1152 for H = 4608 -> 4.5 iters/thread at 256 threads!

    float* sc_out = reinterpret_cast<float*>(OUT_SHARD + scale_off);
    __shared__ float sWarpMax[8];

    for (int row = (int)blockIdx.x; row < S; row += (int)gridDim.x) {
        const __nv_bfloat16* x_row = X + (int64_t)row * stride_xm;
        uint8_t* q_row             = OUT_SHARD + (int64_t)row * H;

        if (n_vec4 <= 1280) {
            // Fast 1-pass scalar register cache (5 uint2 = 10 regs, ZERO local arrays!)
            uint2 c0 = make_uint2(0u, 0u), c1 = make_uint2(0u, 0u);
            uint2 c2 = make_uint2(0u, 0u), c3 = make_uint2(0u, 0u);
            uint2 c4 = make_uint2(0u, 0u);
            float local_amax = 0.0f;

            #define LOAD_ROW_VEC4_256(IT, CREG) \
                do { \
                    const int idx = (IT) * 256 + tid; \
                    if (idx < n_vec4) { \
                        const uint2 raw = ld_global_cs_u2(x_row + (idx << 2)); \
                        (CREG) = raw; \
                        float v0, v1, v2, v3; \
                        bf16x4_to_f32x4(raw, v0, v1, v2, v3); \
                        float a = fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3))); \
                        local_amax = fmaxf(local_amax, a); \
                    } \
                } while (0)

            LOAD_ROW_VEC4_256(0, c0);
            LOAD_ROW_VEC4_256(1, c1);
            LOAD_ROW_VEC4_256(2, c2);
            LOAD_ROW_VEC4_256(3, c3);
            LOAD_ROW_VEC4_256(4, c4);
            #undef LOAD_ROW_VEC4_256

            float wmax = warp_max_f32(local_amax);
            if (lane == 0) sWarpMax[warp] = wmax;
            __syncthreads();

            float row_amax = fmaxf(
                fmaxf(fmaxf(sWarpMax[0], sWarpMax[1]), fmaxf(sWarpMax[2], sWarpMax[3])),
                fmaxf(fmaxf(sWarpMax[4], sWarpMax[5]), fmaxf(sWarpMax[6], sWarpMax[7]))
            );
            float scale = fmaxf(row_amax * (1.0f / 448.0f), 1.0e-12f);
            float inv_scale = 1.0f / scale;
            if (tid == 0) st_global_cs_u32(sc_out + row, __float_as_uint(scale));

            #define STORE_ROW_VEC4_256(IT, CREG) \
                do { \
                    const int idx = (IT) * 256 + tid; \
                    if (idx < n_vec4) { \
                        float v0, v1, v2, v3; \
                        bf16x4_to_f32x4((CREG), v0, v1, v2, v3); \
                        uint32_t q4 = f32x4_to_e4m3x4( \
                            v0 * inv_scale, v1 * inv_scale, v2 * inv_scale, v3 * inv_scale); \
                        st_global_cs_u32(q_row + (idx << 2), q4); \
                    } \
                } while (0)

            STORE_ROW_VEC4_256(0, c0);
            STORE_ROW_VEC4_256(1, c1);
            STORE_ROW_VEC4_256(2, c2);
            STORE_ROW_VEC4_256(3, c3);
            STORE_ROW_VEC4_256(4, c4);
            #undef STORE_ROW_VEC4_256
            __syncthreads();
        } else {
            float local_amax = 0.0f;
            for (int idx = tid; idx < n_vec4; idx += 256) {
                uint2 raw = ld_global_cs_u2(x_row + (idx << 2));
                float v0, v1, v2, v3;
                bf16x4_to_f32x4(raw, v0, v1, v2, v3);
                float a = fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3)));
                local_amax = fmaxf(local_amax, a);
            }
            float wmax = warp_max_f32(local_amax);
            if (lane == 0) sWarpMax[warp] = wmax;
            __syncthreads();

            float row_amax = fmaxf(
                fmaxf(fmaxf(sWarpMax[0], sWarpMax[1]), fmaxf(sWarpMax[2], sWarpMax[3])),
                fmaxf(fmaxf(sWarpMax[4], sWarpMax[5]), fmaxf(sWarpMax[6], sWarpMax[7]))
            );
            float scale = fmaxf(row_amax * (1.0f / 448.0f), 1.0e-12f);
            float inv_scale = 1.0f / scale;
            if (tid == 0) st_global_cs_u32(sc_out + row, __float_as_uint(scale));

            for (int idx = tid; idx < n_vec4; idx += 256) {
                uint2 raw = ld_global_cs_u2(x_row + (idx << 2));
                float v0, v1, v2, v3;
                bf16x4_to_f32x4(raw, v0, v1, v2, v3);
                uint32_t q4 = f32x4_to_e4m3x4(
                    v0 * inv_scale, v1 * inv_scale, v2 * inv_scale, v3 * inv_scale);
                st_global_cs_u32(q_row + (idx << 2), q4);
            }
            __syncthreads();
        }
    }
}

__global__ __launch_bounds__(256, 2)
void rhd_ag_unpack_fp8_rowwise_lowsm_kernel(
    const uint8_t* __restrict__ AG_BUF,
    uint8_t* __restrict__ OUT_FP8,
    float* __restrict__ OUT_SCALE,
    int64_t shard_bytes,
    int64_t scale_off,
    int64_t stride_out,
    int S,
    int H,
    int W,
    int wb_out
) {
    const int tid        = (int)threadIdx.x;
    const int n_vec16    = H >> 4;
    const int total_rows = W * S;

    for (int dst_r = (int)blockIdx.x; dst_r < total_rows; dst_r += (int)gridDim.x) {
        const int r              = dst_r / S;
        const int local_row      = dst_r - r * S;
        const uint8_t* slot_base = AG_BUF + (int64_t)r * shard_bytes;
        const uint8_t* src_row   = slot_base + (int64_t)local_row * H;
        uint8_t* dst_row         = OUT_FP8 + (int64_t)dst_r * stride_out;

        for (int idx = tid; idx < n_vec16; idx += 256) {
            uint4 v = ld_global_cs_u4(src_row + (idx << 4));
            if (wb_out) st_global_wb_u4(dst_row + (idx << 4), v);
            else        st_global_cs_u4(dst_row + (idx << 4), v);
        }
        if (tid == 0) {
            const uint32_t* src_sc = reinterpret_cast<const uint32_t*>(slot_base + scale_off);
            const uint32_t sc_bits = ld_global_cs_u32(src_sc + local_row);
            if (wb_out) OUT_SCALE[dst_r] = __uint_as_float(sc_bits);
            else        st_global_cs_u32(OUT_SCALE + dst_r, sc_bits);
        }
    }
}

__global__ __launch_bounds__(256, 2)
void rhd_ag_dequant_fp8_rowwise_lowsm_kernel(
    const uint8_t* __restrict__ AG_BUF,
    __nv_bfloat16* __restrict__ OUT_BF16,
    int64_t shard_bytes,
    int64_t scale_off,
    int64_t stride_out,
    int S,
    int H,
    int W,
    int wb_out
) {
    const int tid        = (int)threadIdx.x;
    const int lane       = tid & 31;
    const int n_vec4     = H >> 2;
    const int total_rows = W * S;

    for (int dst_r = (int)blockIdx.x; dst_r < total_rows; dst_r += (int)gridDim.x) {
        const int r              = dst_r / S;
        const int local_row      = dst_r - r * S;
        const uint8_t* slot_base = AG_BUF + (int64_t)r * shard_bytes;
        const uint8_t* src_row   = slot_base + (int64_t)local_row * H;
        __nv_bfloat16* dst_row   = OUT_BF16 + (int64_t)dst_r * stride_out;

        uint32_t sc_bits = 0u;
        if (lane == 0) {
            const uint32_t* src_sc = reinterpret_cast<const uint32_t*>(slot_base + scale_off);
            sc_bits = ld_global_cs_u32(src_sc + local_row);
        }
        const float sc = __uint_as_float(__shfl_sync(0xFFFFFFFFu, sc_bits, 0));

        for (int idx = tid; idx < n_vec4; idx += 256) {
            const int d = idx << 2;
            const uint32_t q4 = ld_global_cs_u32(src_row + d);
            float v0, v1, v2, v3;
            dequant_4vals<true>(q4, sc, v0, v1, v2, v3);
            uint2 out_bf = f32x4_to_bf16_u2(v0, v1, v2, v3);
            if (wb_out) st_global_wb_u2(dst_row + d, out_bf);
            else        st_global_cs_u2(dst_row + d, out_bf);
        }
    }
}

template <bool IS_FP8>
__global__ __launch_bounds__(256, 2)
void rhd_ag_dequant_g64_lowsm_kernel(
    const uint8_t* __restrict__ AG_BUF,
    __nv_bfloat16* __restrict__ OUT,
    int64_t shard_bytes,
    int64_t scale_off,
    int64_t stride_out,
    int S,
    int H,
    int W,
    int wb_out
) {
    const int tid        = (int)threadIdx.x;
    const int warp       = tid >> 5;
    const int lane       = tid & 31;
    const int halfw      = (lane >> 4) & 1;
    const int sublane    = lane & 15;
    const int G          = H / kGroupSize;
    const int total_rows = W * S;

    for (int dst_r = (int)blockIdx.x; dst_r < total_rows; dst_r += (int)gridDim.x) {
        const int r              = dst_r / S;
        const int local_row      = dst_r - r * S;
        const uint8_t* slot_base = AG_BUF + (int64_t)r * shard_bytes;
        const uint8_t* q_row     = slot_base + (int64_t)local_row * H;
        const __nv_bfloat16* sc_p = reinterpret_cast<const __nv_bfloat16*>(
            slot_base + scale_off + (int64_t)local_row * G * 2
        );
        __nv_bfloat16* out_row   = OUT + (int64_t)dst_r * stride_out;

        for (int g_pair = warp * 2; g_pair < G; g_pair += 16) {
            const int g = g_pair + halfw;
            const int d = (g * kGroupSize) + (sublane << 2);
            const float sc = load_scale_pair_cs(sc_p + g_pair, lane, halfw);

            uint32_t q4 = ld_global_cs_u32(q_row + d);
            float v0, v1, v2, v3;
            dequant_4vals<IS_FP8>(q4, sc, v0, v1, v2, v3);
            uint2 out_bf = f32x4_to_bf16_u2(v0, v1, v2, v3);
            if (wb_out) st_global_wb_u2(out_row + d, out_bf);
            else        st_global_cs_u2(out_row + d, out_bf);
        }
    }
}


// =============================================================================
// WAVE-V11 ADDITIONS: Capture Guard, Sliced A2A Low-SM, and 0-ULP Fused
// ShortConv + AttnRes + RMSNorm + DiscardBlkPtr Skip + Direct FP8-AG Quant
// =============================================================================

static inline bool v11_is_stream_capturing(cudaStream_t stream) {
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    cudaError_t err = cudaStreamIsCapturing(stream, &status);
    if (err != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return status != cudaStreamCaptureStatusNone;
}

static __device__ __forceinline__ float v11_rsqrt_approx(float x) {
    float y;
    asm volatile("rsqrt.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

static __device__ __forceinline__ float v11_tldevice_exp_sub(float a, float b) {
    float y;
    asm volatile(
        "{\n\t"
        ".reg .f32 diff, t0, t1, t2, t3, t4, t5, t6, t7, f0;\n\t"
        ".reg .b32 b2, s0;\n\t"
        "sub.rn.f32 diff, %1, %2;\n\t"
        "fma.rn.ftz.f32 t0, diff, 0f3BBB989D, 0f3F000000;\n\t"
        "cvt.ftz.sat.f32.f32 t1, t0;\n\t"
        "fma.rm.ftz.f32 t2, t1, 0f437C0000, 0f4B400001;\n\t"
        "add.rn.f32 t3, t2, 0fCB40007F;\n\t"
        "neg.f32 t4, t3;\n\t"
        "fma.rn.ftz.f32 t5, diff, 0f3FB8AA3B, t4;\n\t"
        "fma.rn.ftz.f32 t6, diff, 0f32A57060, t5;\n\t"
        "mov.b32 b2, t2;\n\t"
        "shl.b32 s0, b2, 23;\n\t"
        "mov.b32 f0, s0;\n\t"
        "ex2.approx.ftz.f32 t7, t6;\n\t"
        "mul.rn.f32 %0, t7, f0;\n\t"
        "}\n\t"
        : "=f"(y) : "f"(a), "f"(b)
    );
    return y;
}

static __device__ __forceinline__ void v11_divmod_cache(
    int64_t val, int n_cache, int n_shift, int64_t& q, int64_t& r
) {
    if (n_shift >= 0) {
        q = val >> n_shift;
        r = val & ((1LL << n_shift) - 1);
    } else {
        q = val / n_cache;
        r = val % n_cache;
    }
}

static __device__ __forceinline__ uint4 v11_ld_global_ca_u4(const void* p) {
    uint4 v;
    asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}

static __device__ __forceinline__ uint4 v11_ld_const_u4(const void* p) {
    uint4 v;
    asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
        : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}

static __device__ __forceinline__ float4 v11_ld_const_f4(const float* p) {
    float4 v;
    asm("ld.global.nc.v4.f32 {%0,%1,%2,%3}, [%4];"
        : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(p));
    return v;
}

static __device__ __forceinline__ void v11_unpack_bf16x8(uint4 raw, float out[8]) {
    const __nv_bfloat162* h2 = reinterpret_cast<const __nv_bfloat162*>(&raw);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        float2 f = __bfloat1622float2(h2[i]);
        out[2 * i]     = f.x;
        out[2 * i + 1] = f.y;
    }
}

static __device__ __forceinline__ uint4 v11_pack_f32x8_to_bf16_u4(const float in[8]) {
    uint4 raw;
    __nv_bfloat162* h2 = reinterpret_cast<__nv_bfloat162*>(&raw);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        h2[i] = __halves2bfloat162(__float2bfloat16_rn(in[2 * i]), __float2bfloat16_rn(in[2 * i + 1]));
    }
    return raw;
}

static __device__ __forceinline__ void v11_load_f32x8_const(const float* p, float out[8]) {
    float4 v0 = v11_ld_const_f4(p);
    float4 v1 = v11_ld_const_f4(p + 4);
    out[0] = v0.x; out[1] = v0.y; out[2] = v0.z; out[3] = v0.w;
    out[4] = v1.x; out[5] = v1.y; out[6] = v1.z; out[7] = v1.w;
}

static __device__ __forceinline__ void v11_load_cw4x8_const(
    const __nv_bfloat16* cw_base, float w[4][8]
) {
    uint4 r0 = v11_ld_const_u4(cw_base);
    uint4 r1 = v11_ld_const_u4(cw_base + 8);
    uint4 r2 = v11_ld_const_u4(cw_base + 16);
    uint4 r3 = v11_ld_const_u4(cw_base + 24);
    auto unpack2 = [&](uint4 r, int c0) {
        const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&r);
        float2 a0 = __bfloat1622float2(h[0]);
        float2 a1 = __bfloat1622float2(h[1]);
        float2 b0 = __bfloat1622float2(h[2]);
        float2 b1 = __bfloat1622float2(h[3]);
        w[0][c0] = a0.x; w[1][c0] = a0.y; w[2][c0] = a1.x; w[3][c0] = a1.y;
        w[0][c0 + 1] = b0.x; w[1][c0 + 1] = b0.y; w[2][c0 + 1] = b1.x; w[3][c0 + 1] = b1.y;
    };
    unpack2(r0, 0);
    unpack2(r1, 2);
    unpack2(r2, 4);
    unpack2(r3, 6);
}

static __device__ __forceinline__ float v11_tree8_sum(const float v[8]) {
    float e0 = __fadd_rn(__fadd_rn(v[0], v[2]), __fadd_rn(v[4], v[6]));
    float o0 = __fadd_rn(__fadd_rn(v[1], v[3]), __fadd_rn(v[5], v[7]));
    return __fadd_rn(e0, o0);
}

static __device__ __forceinline__ float v11_warp_sum_bfly(float v) {
    v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, 16));
    v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, 8));
    v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, 4));
    v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, 2));
    v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, 1));
    return v;
}

static __device__ __forceinline__ float v11_warp4_sum_bfly(float v) {
    v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, 2));
    v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, 1));
    return v;
}

// -----------------------------------------------------------------------------
// V11 Kernel 1: Row-Sliced Low-SM 1-Hop A2A Quant (`[row_beg, row_beg+row_cnt)`)
// -----------------------------------------------------------------------------
template <bool IS_FP8>
__global__ __launch_bounds__(256, 2)
void a2a_quant_g64_slice_lowsm_kernel(
    const __nv_bfloat16* __restrict__ X,
    uint8_t* __restrict__ SEND_BUF,
    int64_t stride_xm,
    int64_t peer_bytes,
    int64_t scale_off,
    int S,
    int row_beg,
    int row_cnt,
    int H,
    int world_size,
    int rank
) {
    const int tid        = (int)threadIdx.x;
    const int warp       = tid >> 5;
    const int lane       = tid & 31;
    const int halfw      = (lane >> 4) & 1;
    const int sublane    = lane & 15;
    const int G          = H / kGroupSize;
    const int n_peers    = world_size - 1;
    const int total_rows = n_peers * row_cnt;

    for (int task = (int)blockIdx.x; task < total_rows; task += (int)gridDim.x) {
        const int p_idx = task / row_cnt;
        const int r_out = task - p_idx * row_cnt;
        const int r_loc = row_beg + r_out;
        const int dst   = (p_idx < rank) ? p_idx : (p_idx + 1);

        const __nv_bfloat16* x_row = X + ((int64_t)dst * S + r_loc) * stride_xm;
        uint8_t* dst_base          = SEND_BUF + (int64_t)dst * peer_bytes;
        uint8_t* q_row             = dst_base + (int64_t)r_out * H;
        __nv_bfloat16* sc_row      = reinterpret_cast<__nv_bfloat16*>(
            dst_base + scale_off + (int64_t)r_out * G * 2
        );

        for (int g_pair = warp * 2; g_pair < G; g_pair += 16) {
            const int g = g_pair + halfw;
            const int d = (g * kGroupSize) + (sublane << 2);

            uint2 raw = ld_global_cs_u2(x_row + d);
            float v0, v1, v2, v3;
            bf16x4_to_f32x4(raw, v0, v1, v2, v3);

            __nv_bfloat16 sc_bf;
            uint32_t q4 = quant_group64_4vals<IS_FP8>(v0, v1, v2, v3, sc_bf);
            st_global_cs_u32(q_row + d, q4);
            store_scale_pair_cs(sc_row + g_pair, sc_bf, lane);
        }
    }
}

// -----------------------------------------------------------------------------
// V11 Kernel 2: Row-Sliced Low-SM 1-Hop A2A Dequant+Sum8 (flexible `self_row_base`)
// -----------------------------------------------------------------------------
template <bool IS_FP8>
__global__ __launch_bounds__(256, 2)
void a2a_dequant_sum8_g64_slice_lowsm_kernel(
    const uint8_t* __restrict__ RECV_BUF,
    const __nv_bfloat16* __restrict__ SELF_X,
    __nv_bfloat16* __restrict__ OUT,
    int64_t stride_xm,
    int64_t stride_om,
    int64_t self_row_base,
    int64_t peer_bytes,
    int64_t scale_off,
    int row_beg,
    int recv_row_beg,
    int row_cnt,
    int H,
    int world_size,
    int rank,
    int wb_out
) {
    const int tid     = (int)threadIdx.x;
    const int warp    = tid >> 5;
    const int lane    = tid & 31;
    const int halfw   = (lane >> 4) & 1;
    const int sublane = lane & 15;
    const int G       = H / kGroupSize;

    for (int r_in = (int)blockIdx.x; r_in < row_cnt; r_in += (int)gridDim.x) {
        const int s_row = row_beg + r_in;
        const int r_buf = recv_row_beg + r_in;
        const __nv_bfloat16* self_row = SELF_X + (self_row_base + (int64_t)s_row) * stride_xm;
        __nv_bfloat16* out_row        = OUT + (int64_t)s_row * stride_om;

        for (int g_pair = warp * 2; g_pair < G; g_pair += 16) {
            const int g = g_pair + halfw;
            const int d = (g * kGroupSize) + (sublane << 2);

            float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
            for (int r = 0; r < world_size; ++r) {
                if (r == rank) {
                    uint2 raw_self = ld_global_cs_u2(self_row + d);
                    float s0, s1, s2, s3;
                    bf16x4_to_f32x4(raw_self, s0, s1, s2, s3);
                    acc0 += s0; acc1 += s1; acc2 += s2; acc3 += s3;
                } else {
                    const uint8_t* p_base = RECV_BUF + (int64_t)r * peer_bytes;
                    const __nv_bfloat16* sc_p = reinterpret_cast<const __nv_bfloat16*>(
                        p_base + scale_off + (int64_t)r_buf * G * 2
                    );
                    const float sc = load_scale_pair_cs(sc_p + g_pair, lane, halfw);
                    const uint32_t q4 = ld_global_cs_u32(p_base + (int64_t)r_buf * H + d);
                    float r0, r1, r2, r3;
                    dequant_4vals<IS_FP8>(q4, sc, r0, r1, r2, r3);
                    acc0 += r0; acc1 += r1; acc2 += r2; acc3 += r3;
                }
            }

            uint2 out_bf = f32x4_to_bf16_u2(acc0, acc1, acc2, acc3);
            if (wb_out) st_global_wb_u2(out_row + d, out_bf);
            else        st_global_cs_u2(out_row + d, out_bf);
        }
    }
}

// -----------------------------------------------------------------------------
// V11 Kernel 3: 0-ULP Fused ShortConv (W=4) + AttnRes (BLOCK_D=1024 Order-A) +
//               RMSNorm + DiscardBlkPtr Skip + Direct FP8-E4M3FN Rowwise AG Quant
// -----------------------------------------------------------------------------
struct SconvArFp8V11SmemCtrl {
    const __nv_bfloat16* tap_ptrs[3];
    __nv_bfloat16* cache_dst;
    int64_t discard_blk;
    int tap_valid[3];
    int tap_is_cache[3];
    int write_cache;
    int lag0_ok;
    float rstd_arr[9];
    float probs[9];
    float sRed[9][4][4];
    float sSq2[5][4];
    float sAmax[4];
};

template <int N_STACK, int NQ, bool HAS_CONV_RES, bool HAS_CONV_ADDEND, bool HAS_MIX_LAST, bool WRITE_FP8_Q0>
__global__ void __launch_bounds__(128, 1)
fused_sconv_ar_fp8_ag_v11_kernel(
    const __nv_bfloat16* __restrict__ X,
    const __nv_bfloat16* __restrict__ CW,
    __nv_bfloat16* __restrict__ Cache,
    __nv_bfloat16* ConvOut,
    const __nv_bfloat16* ConvRes,
    const __nv_bfloat16* __restrict__ ConvAdd,
    const int64_t* __restrict__ Pos,
    const int32_t* __restrict__ SeqIdx,
    const int64_t* __restrict__ Slot,
    const int32_t* __restrict__ BT,
    const int32_t* __restrict__ QStart,
    const __nv_bfloat16* __restrict__ Stack,
    const __nv_bfloat16* __restrict__ MixLast,
    const float* __restrict__ Queries,
    const __nv_bfloat16* __restrict__ NormW,
    uint8_t* __restrict__ OutFp8,
    float* __restrict__ OutScale,
    __nv_bfloat16* __restrict__ OutAux,
    const int64_t* __restrict__ DiscardBlkPtr,
    int64_t stride_x,
    int64_t stride_c_blk,
    int64_t stride_c_n,
    int64_t stride_bt,
    int max_blocks,
    int N_CACHE,
    int N_SHIFT,
    int stream_offset,
    int64_t stride_s,
    int T,
    float scale,
    float eps,
    float norm_eps
) {
    constexpr int D = 4608;
    constexpr int NS = N_STACK + 1;
    constexpr int U4_PER_ROW = D / 8;
    constexpr float INV_D = 1.0f / (float)D;
    constexpr int BLOCK_N = (NS <= 2) ? 2 : (NS <= 4) ? 4 : (NS <= 8) ? 8 : 16;

    const int tid  = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    extern __shared__ uint4 dyn_smem[];
    uint4* sStack   = dyn_smem;
    uint4* sLastU4  = sStack + N_STACK * U4_PER_ROW;
    float* sLastF32 = reinterpret_cast<float*>(sLastU4);
    __shared__ SconvArFp8V11SmemCtrl sC;

    if (tid == 0) {
        sC.discard_blk = (DiscardBlkPtr != nullptr) ? (*DiscardBlkPtr) : -1LL;
    }
    __syncthreads();

    for (int t = (int)blockIdx.x; t < T; t += (int)gridDim.x) {
        const int64_t row_off = static_cast<int64_t>(t) * D;

        if (tid == 0) {
            const int64_t slot = Slot[t];
            const bool valid = (slot >= 0);
            const int64_t pos = Pos[t];
            const int32_t req = SeqIdx[t];
            const int32_t qstart = QStart[t];
            sC.lag0_ok = (valid && (pos >= 0)) ? 1 : 0;

            #pragma unroll
            for (int k = 0; k < 3; ++k) {
                const int lag = 3 - k;
                const int64_t src = pos - lag;
                const int row = t - lag;
                const bool in_win = valid && (src >= 0);
                sC.tap_valid[k] = in_win ? 1 : 0;
                if (in_win) {
                    if (row >= qstart) {
                        sC.tap_ptrs[k] = X + static_cast<int64_t>(row) * stride_x;
                        sC.tap_is_cache[k] = 0;
                    } else {
                        int64_t lblk = 0, rem = 0;
                        v11_divmod_cache(src, N_CACHE, N_SHIFT, lblk, rem);
                        if (lblk >= max_blocks) lblk = max_blocks - 1;
                        const int64_t blk = static_cast<int64_t>(BT[static_cast<int64_t>(req) * stride_bt + lblk]);
                        sC.tap_ptrs[k] = Cache + blk * stride_c_blk + rem * stride_c_n + stream_offset;
                        sC.tap_is_cache[k] = 1;
                    }
                } else {
                    sC.tap_ptrs[k] = X;
                    sC.tap_is_cache[k] = 0;
                }
            }
            if (valid) {
                int64_t dblk = 0, drem = 0;
                v11_divmod_cache(slot, N_CACHE, N_SHIFT, dblk, drem);
                const bool skip_discard = (sC.discard_blk >= 0 && dblk == sC.discard_blk);
                sC.write_cache = skip_discard ? 0 : 1;
                sC.cache_dst = Cache + dblk * stride_c_blk + drem * stride_c_n + stream_offset;
            } else {
                sC.write_cache = 0;
                sC.cache_dst = nullptr;
            }
        }
        __syncthreads();

        const bool tv0 = (sC.tap_valid[0] != 0);
        const bool tv1 = (sC.tap_valid[1] != 0);
        const bool tv2 = (sC.tap_valid[2] != 0);
        const bool tc0 = (sC.tap_is_cache[0] != 0);
        const bool tc1 = (sC.tap_is_cache[1] != 0);
        const bool tc2 = (sC.tap_is_cache[2] != 0);
        const bool lag0_ok = (sC.lag0_ok != 0);
        const bool write_cache = (sC.write_cache != 0);
        const __nv_bfloat16* tp0 = sC.tap_ptrs[0];
        const __nv_bfloat16* tp1 = sC.tap_ptrs[1];
        const __nv_bfloat16* tp2 = sC.tap_ptrs[2];
        __nv_bfloat16* cache_dst = sC.cache_dst;

        // Step 1: ShortConv + Pass 1 on Last
        {
            const __nv_bfloat16* x_cur_p = X + static_cast<int64_t>(t) * stride_x;
            float sum_sq[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
            float dots[NQ][8];
            #pragma unroll
            for (int q = 0; q < NQ; ++q) {
                #pragma unroll
                for (int i = 0; i < 8; ++i) dots[q][i] = 0.0f;
            }

            #pragma unroll
            for (int c = 0; c < 5; ++c) {
                const int d_base = c * 1024 + tid * 8;
                if (c < 4 || tid < 64) {
                    uint4 raw_cur = v11_ld_global_ca_u4(x_cur_p + d_base);
                    uint4 raw_t0 = tv0 ? (tc0 ? ld_global_cs_u4(tp0 + d_base) : v11_ld_global_ca_u4(tp0 + d_base)) : make_uint4(0u, 0u, 0u, 0u);
                    uint4 raw_t1 = tv1 ? (tc1 ? ld_global_cs_u4(tp1 + d_base) : v11_ld_global_ca_u4(tp1 + d_base)) : make_uint4(0u, 0u, 0u, 0u);
                    uint4 raw_t2 = tv2 ? (tc2 ? ld_global_cs_u4(tp2 + d_base) : v11_ld_global_ca_u4(tp2 + d_base)) : make_uint4(0u, 0u, 0u, 0u);
                    if (write_cache) st_global_cs_u4(cache_dst + d_base, raw_cur);

                    float w[4][8];
                    v11_load_cw4x8_const(CW + d_base * 4, w);
                    float cur_f[8], t0_f[8], t1_f[8], t2_f[8];
                    v11_unpack_bf16x8(raw_cur, cur_f);
                    v11_unpack_bf16x8(raw_t0, t0_f);
                    v11_unpack_bf16x8(raw_t1, t1_f);
                    v11_unpack_bf16x8(raw_t2, t2_f);

                    float acc_c[8];
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        float a = 0.0f;
                        a = __fmaf_rn(tv0     ? t0_f[i]  : 0.0f, w[0][i], a);
                        a = __fmaf_rn(tv1     ? t1_f[i]  : 0.0f, w[1][i], a);
                        a = __fmaf_rn(tv2     ? t2_f[i]  : 0.0f, w[2][i], a);
                        a = __fmaf_rn(lag0_ok ? cur_f[i] : 0.0f, w[3][i], a);
                        acc_c[i] = __fmul_rn(__fadd_rn(a, cur_f[i]), scale);
                    }
                    uint4 raw_y = v11_pack_f32x8_to_bf16_u4(acc_c);

                    if constexpr (HAS_CONV_ADDEND) {
                        uint4 raw_ca = ld_global_cs_u4(ConvAdd + row_off + d_base);
                        float ca[8], y_f[8];
                        v11_unpack_bf16x8(raw_ca, ca);
                        v11_unpack_bf16x8(raw_y, y_f);
                        if constexpr (HAS_CONV_RES) {
                            uint4 raw_cr = ld_global_cs_u4(ConvRes + row_off + d_base);
                            float cr[8];
                            v11_unpack_bf16x8(raw_cr, cr);
                            #pragma unroll
                            for (int i = 0; i < 8; ++i) ca[i] = __fadd_rn(cr[i], ca[i]);
                        }
                        #pragma unroll
                        for (int i = 0; i < 8; ++i) acc_c[i] = __fadd_rn(ca[i], y_f[i]);
                        raw_y = v11_pack_f32x8_to_bf16_u4(acc_c);
                    }
                    st_global_cs_u4(ConvOut + row_off + d_base, raw_y);

                    float xv[8];
                    v11_unpack_bf16x8(raw_y, xv);
                    if constexpr (HAS_MIX_LAST) {
                        uint4 raw_ml = ld_global_cs_u4(MixLast + row_off + d_base);
                        float ml[8];
                        v11_unpack_bf16x8(raw_ml, ml);
                        #pragma unroll
                        for (int i = 0; i < 8; ++i) xv[i] = __fadd_rn(ml[i], xv[i]);
                        *reinterpret_cast<float4*>(sLastF32 + d_base)     = make_float4(xv[0], xv[1], xv[2], xv[3]);
                        *reinterpret_cast<float4*>(sLastF32 + d_base + 4) = make_float4(xv[4], xv[5], xv[6], xv[7]);
                    } else {
                        sLastU4[d_base >> 3] = raw_y;
                    }
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) sum_sq[i] = __fmaf_rn(xv[i], xv[i], sum_sq[i]);
                    #pragma unroll
                    for (int q = 0; q < NQ; ++q) {
                        float qv[8];
                        v11_load_f32x8_const(Queries + static_cast<int64_t>(q) * D + d_base, qv);
                        #pragma unroll
                        for (int i = 0; i < 8; ++i) dots[q][i] = __fmaf_rn(qv[i], xv[i], dots[q][i]);
                    }
                }
            }
            float w_sq = v11_warp_sum_bfly(v11_tree8_sum(sum_sq));
            if (lane == 0) sC.sRed[N_STACK][0][warp] = w_sq;
            #pragma unroll
            for (int q = 0; q < NQ; ++q) {
                float w_dt = v11_warp_sum_bfly(v11_tree8_sum(dots[q]));
                if (lane == 0) sC.sRed[N_STACK][1 + q][warp] = w_dt;
            }
        }

        // Step 2: Stream Stack[0..N_STACK-1] ONCE into SMEM (5x pipelined cs loads)
        #pragma unroll
        for (int n = 0; n < N_STACK; ++n) {
            const __nv_bfloat16* sp = Stack + static_cast<int64_t>(n) * stride_s + row_off;
            uint4* dst_s = sStack + n * U4_PER_ROW;
            uint4 raw0 = ld_global_cs_u4(sp + 0 * 1024 + tid * 8);
            uint4 raw1 = ld_global_cs_u4(sp + 1 * 1024 + tid * 8);
            uint4 raw2 = ld_global_cs_u4(sp + 2 * 1024 + tid * 8);
            uint4 raw3 = ld_global_cs_u4(sp + 3 * 1024 + tid * 8);
            uint4 raw4 = (tid < 64) ? ld_global_cs_u4(sp + 4 * 1024 + tid * 8) : make_uint4(0u, 0u, 0u, 0u);

            float sum_sq[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
            float dots[NQ][8];
            #pragma unroll
            for (int q = 0; q < NQ; ++q) {
                #pragma unroll
                for (int i = 0; i < 8; ++i) dots[q][i] = 0.0f;
            }

            auto proc_stack_chunk = [&](int c, uint4 raw_c) {
                const int d_base = c * 1024 + tid * 8;
                dst_s[d_base >> 3] = raw_c;
                float xv[8];
                v11_unpack_bf16x8(raw_c, xv);
                #pragma unroll
                for (int i = 0; i < 8; ++i) sum_sq[i] = __fmaf_rn(xv[i], xv[i], sum_sq[i]);
                #pragma unroll
                for (int q = 0; q < NQ; ++q) {
                    float qv[8];
                    v11_load_f32x8_const(Queries + static_cast<int64_t>(q) * D + d_base, qv);
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) dots[q][i] = __fmaf_rn(qv[i], xv[i], dots[q][i]);
                }
            };
            proc_stack_chunk(0, raw0);
            proc_stack_chunk(1, raw1);
            proc_stack_chunk(2, raw2);
            proc_stack_chunk(3, raw3);
            if (tid < 64) proc_stack_chunk(4, raw4);

            float w_sq = v11_warp_sum_bfly(v11_tree8_sum(sum_sq));
            if (lane == 0) sC.sRed[n][0][warp] = w_sq;
            #pragma unroll
            for (int q = 0; q < NQ; ++q) {
                float w_dt = v11_warp_sum_bfly(v11_tree8_sum(dots[q]));
                if (lane == 0) sC.sRed[n][1 + q][warp] = w_dt;
            }
        }

        __syncthreads();

        // Step 3: Exact Pass 2 + Pass 3 from SMEM (Order-A FFMA Pass 2!)
        const int l4 = lane & 3;
        if (warp == 0) {
            #pragma unroll
            for (int n = 0; n < NS; ++n) {
                float tot_sq = v11_warp4_sum_bfly(sC.sRed[n][0][l4]);
                if (lane == 0) {
                    sC.rstd_arr[n] = v11_rsqrt_approx(__fmaf_rn(tot_sq, INV_D, eps));
                }
            }
        }
        __syncthreads();

        #pragma unroll
        for (int q = 0; q < NQ; ++q) {
            if (warp == 0) {
                float my_lg = -INFINITY;
                float my_ir = 0.0f;
                #pragma unroll
                for (int n = 0; n < NS; ++n) {
                    float tot_dt = v11_warp4_sum_bfly(sC.sRed[n][1 + q][l4]);
                    if (lane == n) {
                        float rn = sC.rstd_arr[n];
                        my_lg = __fmul_rn(tot_dt, rn);
                        my_ir = rn;
                    }
                }
                float mx = my_lg;
                if constexpr (BLOCK_N >= 16) mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, 8));
                if constexpr (BLOCK_N >= 8)  mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, 4));
                if constexpr (BLOCK_N >= 4)  mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, 2));
                if constexpr (BLOCK_N >= 2)  mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, 1));
                float p = (lane < NS) ? v11_tldevice_exp_sub(my_lg, mx) : 0.0f;
                float sp = p;
                if constexpr (BLOCK_N >= 16) sp = __fadd_rn(sp, __shfl_xor_sync(0xFFFFFFFFu, sp, 8));
                if constexpr (BLOCK_N >= 8)  sp = __fadd_rn(sp, __shfl_xor_sync(0xFFFFFFFFu, sp, 4));
                if constexpr (BLOCK_N >= 4)  sp = __fadd_rn(sp, __shfl_xor_sync(0xFFFFFFFFu, sp, 2));
                if constexpr (BLOCK_N >= 2)  sp = __fadd_rn(sp, __shfl_xor_sync(0xFFFFFFFFu, sp, 1));
                float inv_sum = __fdiv_rn(1.0f, __shfl_sync(0xFFFFFFFFu, sp, 0));
                p = __fmul_rn(p, my_ir);
                if (lane < NS) sC.probs[lane] = p;
                if (lane == 0) sC.sAmax[0] = inv_sum;
            }
            __syncthreads();
            const float inv_sum = sC.sAmax[0];

            float acc[5][8];
            #pragma unroll
            for (int c = 0; c < 5; ++c) {
                const int d_base = c * 1024 + tid * 8;
                float sq_sum = 0.0f;
                if (c < 4 || tid < 64) {
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) acc[c][i] = 0.0f;
                    const int u4_idx = d_base >> 3;
                    #pragma unroll
                    for (int n = 0; n < N_STACK; ++n) {
                        const float pn = sC.probs[n];
                        float xv[8];
                        v11_unpack_bf16x8(sStack[n * U4_PER_ROW + u4_idx], xv);
                        #pragma unroll
                        for (int i = 0; i < 8; ++i) acc[c][i] = __fmaf_rn(pn, xv[i], acc[c][i]);
                    }
                    {
                        const float pn = sC.probs[N_STACK];
                        float xv[8];
                        if constexpr (HAS_MIX_LAST) {
                            float4 v0 = *reinterpret_cast<const float4*>(sLastF32 + d_base);
                            float4 v1 = *reinterpret_cast<const float4*>(sLastF32 + d_base + 4);
                            xv[0] = v0.x; xv[1] = v0.y; xv[2] = v0.z; xv[3] = v0.w;
                            xv[4] = v1.x; xv[5] = v1.y; xv[6] = v1.z; xv[7] = v1.w;
                        } else {
                            v11_unpack_bf16x8(sLastU4[u4_idx], xv);
                        }
                        #pragma unroll
                        for (int i = 0; i < 8; ++i) acc[c][i] = __fmaf_rn(pn, xv[i], acc[c][i]);
                    }
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) acc[c][i] = __fmul_rn(acc[c][i], inv_sum);

                    float e01 = __fmaf_rn(acc[c][0], acc[c][0], __fmul_rn(acc[c][2], acc[c][2]));
                    float o01 = __fmaf_rn(acc[c][1], acc[c][1], __fmul_rn(acc[c][3], acc[c][3]));
                    float e23 = __fmaf_rn(acc[c][4], acc[c][4], __fmul_rn(acc[c][6], acc[c][6]));
                    float o23 = __fmaf_rn(acc[c][5], acc[c][5], __fmul_rn(acc[c][7], acc[c][7]));
                    sq_sum = __fadd_rn(__fadd_rn(e01, e23), __fadd_rn(o01, o23));
                }
                float w_sq = v11_warp_sum_bfly(sq_sum);
                if (lane == 0) sC.sSq2[c][warp] = w_sq;
            }

            __syncthreads();

            float tot_sum_sq = 0.0f;
            #pragma unroll
            for (int c = 0; c < 5; ++c) {
                tot_sum_sq = __fadd_rn(tot_sum_sq, v11_warp4_sum_bfly(sC.sSq2[c][l4]));
            }
            const float out_rstd = v11_rsqrt_approx(__fadd_rn(__fdiv_rn(tot_sum_sq, (float)D), norm_eps));
            const __nv_bfloat16* nw_p = NormW + static_cast<int64_t>(q) * D;

            if constexpr (WRITE_FP8_Q0) {
                if (q == 0) {
                    float local_amax = 0.0f;
                    #pragma unroll
                    for (int c = 0; c < 5; ++c) {
                        const int d_base = c * 1024 + tid * 8;
                        if (c < 4 || tid < 64) {
                            float w[8];
                            v11_unpack_bf16x8(v11_ld_const_u4(nw_p + d_base), w);
                            float res[8];
                            #pragma unroll
                            for (int i = 0; i < 8; ++i) {
                                res[i] = __fmul_rn(__fmul_rn(acc[c][i], out_rstd), w[i]);
                            }
                            uint4 bf_u4 = v11_pack_f32x8_to_bf16_u4(res);
                            v11_unpack_bf16x8(bf_u4, acc[c]);
                            #pragma unroll
                            for (int i = 0; i < 8; ++i) {
                                local_amax = fmaxf(local_amax, fabsf(acc[c][i]));
                            }
                        }
                    }
                    float wmax = warp_max_f32(local_amax);
                    if (lane == 0) sC.sAmax[warp] = wmax;
                    __syncthreads();

                    const float row_amax = fmaxf(fmaxf(sC.sAmax[0], sC.sAmax[1]), fmaxf(sC.sAmax[2], sC.sAmax[3]));
                    const float fp8_scale = fmaxf(__fmul_rn(row_amax, (1.0f / 448.0f)), 1.0e-12f);
                    const float inv_fp8_scale = __fdiv_rn(1.0f, fp8_scale);
                    if (tid == 0) {
                        st_global_cs_u32(OutScale + t, __float_as_uint(fp8_scale));
                    }
                    uint8_t* out_fp8_row = OutFp8 + row_off;
                    #pragma unroll
                    for (int c = 0; c < 5; ++c) {
                        const int d_base = c * 1024 + tid * 8;
                        if (c < 4 || tid < 64) {
                            uint32_t q_lo = f32x4_to_e4m3x4(
                                __fmul_rn(acc[c][0], inv_fp8_scale),
                                __fmul_rn(acc[c][1], inv_fp8_scale),
                                __fmul_rn(acc[c][2], inv_fp8_scale),
                                __fmul_rn(acc[c][3], inv_fp8_scale)
                            );
                            uint32_t q_hi = f32x4_to_e4m3x4(
                                __fmul_rn(acc[c][4], inv_fp8_scale),
                                __fmul_rn(acc[c][5], inv_fp8_scale),
                                __fmul_rn(acc[c][6], inv_fp8_scale),
                                __fmul_rn(acc[c][7], inv_fp8_scale)
                            );
                            st_global_cs_u2(out_fp8_row + d_base, make_uint2(q_lo, q_hi));
                        }
                    }
                } else {
                    __nv_bfloat16* out_p = OutAux + static_cast<int64_t>(q - 1) * T * D + row_off;
                    #pragma unroll
                    for (int c = 0; c < 5; ++c) {
                        const int d_base = c * 1024 + tid * 8;
                        if (c < 4 || tid < 64) {
                            float w[8];
                            v11_unpack_bf16x8(v11_ld_const_u4(nw_p + d_base), w);
                            float res[8];
                            #pragma unroll
                            for (int i = 0; i < 8; ++i) {
                                res[i] = __fmul_rn(__fmul_rn(acc[c][i], out_rstd), w[i]);
                            }
                            st_global_cs_u4(out_p + d_base, v11_pack_f32x8_to_bf16_u4(res));
                        }
                    }
                }
            } else {
                __nv_bfloat16* out_p = OutAux + static_cast<int64_t>(q) * T * D + row_off;
                #pragma unroll
                for (int c = 0; c < 5; ++c) {
                    const int d_base = c * 1024 + tid * 8;
                    if (c < 4 || tid < 64) {
                        float w[8];
                        v11_unpack_bf16x8(v11_ld_const_u4(nw_p + d_base), w);
                        float res[8];
                        #pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            res[i] = __fmul_rn(__fmul_rn(acc[c][i], out_rstd), w[i]);
                        }
                        st_global_cs_u4(out_p + d_base, v11_pack_f32x8_to_bf16_u4(res));
                    }
                }
            }

            if constexpr (NQ > 1) {
                if (q + 1 < NQ) __syncthreads();
            }
        }
        __syncthreads();
    }
}

template <int N_STACK, int NQ, bool HCR, bool HCA, bool HML, bool FP8_Q0>
static inline cudaError_t v11_launch_sconv_ar_one(
    int T, int max_ctas, cudaStream_t stream,
    const __nv_bfloat16* X, const __nv_bfloat16* CW, __nv_bfloat16* Cache,
    __nv_bfloat16* ConvOut, const __nv_bfloat16* ConvRes, const __nv_bfloat16* ConvAdd,
    const int64_t* Pos, const int32_t* SeqIdx, const int64_t* Slot, const int32_t* BT, const int32_t* QStart,
    const __nv_bfloat16* Stack, const __nv_bfloat16* MixLast,
    const float* Queries, const __nv_bfloat16* NormW,
    uint8_t* OutFp8, float* OutScale, __nv_bfloat16* OutAux,
    const int64_t* DiscardBlkPtr,
    int64_t stride_x, int64_t stride_c_blk, int64_t stride_c_n, int64_t stride_bt,
    int max_blocks, int N_CACHE, int N_SHIFT, int stream_offset, int64_t stride_s,
    float scale, float eps, float norm_eps
) {
    constexpr int D = 4608;
    constexpr size_t smem_bytes =
        static_cast<size_t>(N_STACK) * D * 2 + (HML ? static_cast<size_t>(D) * 4 : static_cast<size_t>(D) * 2);
    auto k = fused_sconv_ar_fp8_ag_v11_kernel<N_STACK, NQ, HCR, HCA, HML, FP8_Q0>;
    static bool optin_done = false;
    if (!optin_done && smem_bytes > 49152) {
        cudaError_t e = cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
        if (e != cudaSuccess) return e;
        optin_done = true;
    }
    int cap = (max_ctas > 0) ? max_ctas : 48;
    if (cap > 96) cap = 96;
    unsigned grid = (unsigned)((T < cap) ? T : cap);
    if (grid == 0u) grid = 1u;
    k<<<dim3(grid, 1, 1), dim3(128, 1, 1), smem_bytes, stream>>>(
        X, CW, Cache, ConvOut, ConvRes, ConvAdd,
        Pos, SeqIdx, Slot, BT, QStart,
        Stack, MixLast, Queries, NormW,
        OutFp8, OutScale, OutAux, DiscardBlkPtr,
        stride_x, stride_c_blk, stride_c_n, stride_bt,
        max_blocks, N_CACHE, N_SHIFT, stream_offset, stride_s,
        T, scale, eps, norm_eps
    );
    return cudaGetLastError();
}

template <int N_STACK, bool FP8_Q0>
static cudaError_t v11_dispatch_sconv_ar_ns(
    int nq, int T, int max_ctas, cudaStream_t stream,
    const __nv_bfloat16* X, const __nv_bfloat16* CW, __nv_bfloat16* Cache,
    __nv_bfloat16* ConvOut, const __nv_bfloat16* ConvRes, const __nv_bfloat16* ConvAdd,
    const int64_t* Pos, const int32_t* SeqIdx, const int64_t* Slot, const int32_t* BT, const int32_t* QStart,
    const __nv_bfloat16* Stack, const __nv_bfloat16* MixLast,
    const float* Queries, const __nv_bfloat16* NormW,
    uint8_t* OutFp8, float* OutScale, __nv_bfloat16* OutAux,
    const int64_t* DiscardBlkPtr,
    int64_t stride_x, int64_t stride_c_blk, int64_t stride_c_n, int64_t stride_bt,
    int max_blocks, int N_CACHE, int N_SHIFT, int stream_offset, int64_t stride_s,
    float scale, float eps, float norm_eps,
    bool hcr, bool hca, bool hml
) {
    #define V11_GO(NQ_V, HCR_V, HCA_V, HML_V) \
        return v11_launch_sconv_ar_one<N_STACK, NQ_V, HCR_V, HCA_V, HML_V, FP8_Q0>( \
            T, max_ctas, stream, X, CW, Cache, ConvOut, ConvRes, ConvAdd, \
            Pos, SeqIdx, Slot, BT, QStart, Stack, MixLast, Queries, NormW, \
            OutFp8, OutScale, OutAux, DiscardBlkPtr, \
            stride_x, stride_c_blk, stride_c_n, stride_bt, \
            max_blocks, N_CACHE, N_SHIFT, stream_offset, stride_s, scale, eps, norm_eps)
    if (nq == 1) {
        if (!hca && !hcr && !hml) { V11_GO(1, false, false, false); }
        if (!hca && !hcr &&  hml) { V11_GO(1, false, false, true);  }
        if ( hca && !hcr && !hml) { V11_GO(1, false, true,  false); }
        if ( hca &&  hcr && !hml) { V11_GO(1, true,  true,  false); }
    } else if (nq == 3) {
        if ( hca && !hcr && !hml) { V11_GO(3, false, true,  false); }
        if ( hca &&  hcr && !hml) { V11_GO(3, true,  true,  false); }
    }
    #undef V11_GO
    return cudaErrorUnknown;
}

}  // namespace

// ===========================================================================
// Public C ABI Entrypoints (12 Legacy + Version + 15 v5 Low-SM Entrypoints)
// ===========================================================================
extern "C" {

int pf_rhd_comm_v5_version() {
    return 5;
}

// ---- Legacy 1-CTA/row Entrypoints (100% Backwards-Compatible) ----

int rhd_quant_slice_int8_g64_fwd(
    const void* x_bf16,
    void* out_packed,
    int64_t stride_xm,
    int64_t row_off,
    int64_t scale_off,
    int M,
    int H,
    uint64_t stream_u64
) {
    if (M <= 0 || H <= 0 || (H % 128) != 0) return -1;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_u64);
    if (v11_is_stream_capturing(stream)) return -4;
    rhd_quant_slice_g64_kernel<false><<<dim3((unsigned)M, 1, 1), dim3(128, 1, 1), 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x_bf16),
        reinterpret_cast<uint8_t*>(out_packed),
        stride_xm, row_off, scale_off, H
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

int rhd_quant_slice_fp8_g64_fwd(
    const void* x_bf16,
    void* out_packed,
    int64_t stride_xm,
    int64_t row_off,
    int64_t scale_off,
    int M,
    int H,
    uint64_t stream_u64
) {
    if (M <= 0 || H <= 0 || (H % 128) != 0) return -1;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_u64);
    if (v11_is_stream_capturing(stream)) return -4;
    rhd_quant_slice_g64_kernel<true><<<dim3((unsigned)M, 1, 1), dim3(128, 1, 1), 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x_bf16),
        reinterpret_cast<uint8_t*>(out_packed),
        stride_xm, row_off, scale_off, H
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

int rhd_dequant_add_split_quant_int8_g64_fwd(
    const void* recv_packed,
    const void* local_bf16,
    void* next_send_packed,
    void* next_keep_bf16,
    int64_t stride_local,
    int64_t local_row_off,
    int64_t recv_scale_off,
    int64_t next_scale_off,
    int64_t stride_keep,
    int N,
    int H,
    int send_upper,
    int wb_keep,
    uint64_t stream_u64
) {
    if (N <= 0 || H <= 0 || (H % 128) != 0) return -1;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_u64);
    if (v11_is_stream_capturing(stream)) return -4;
    rhd_dequant_add_split_quant_g64_kernel<false><<<dim3((unsigned)(2 * N), 1, 1), dim3(128, 1, 1), 0, stream>>>(
        reinterpret_cast<const uint8_t*>(recv_packed),
        reinterpret_cast<const __nv_bfloat16*>(local_bf16),
        reinterpret_cast<uint8_t*>(next_send_packed),
        reinterpret_cast<__nv_bfloat16*>(next_keep_bf16),
        stride_local, local_row_off, recv_scale_off, next_scale_off, stride_keep,
        N, H, send_upper, wb_keep
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

int rhd_dequant_add_split_quant_fp8_g64_fwd(
    const void* recv_packed,
    const void* local_bf16,
    void* next_send_packed,
    void* next_keep_bf16,
    int64_t stride_local,
    int64_t local_row_off,
    int64_t recv_scale_off,
    int64_t next_scale_off,
    int64_t stride_keep,
    int N,
    int H,
    int send_upper,
    int wb_keep,
    uint64_t stream_u64
) {
    if (N <= 0 || H <= 0 || (H % 128) != 0) return -1;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_u64);
    if (v11_is_stream_capturing(stream)) return -4;
    rhd_dequant_add_split_quant_g64_kernel<true><<<dim3((unsigned)(2 * N), 1, 1), dim3(128, 1, 1), 0, stream>>>(
        reinterpret_cast<const uint8_t*>(recv_packed),
        reinterpret_cast<const __nv_bfloat16*>(local_bf16),
        reinterpret_cast<uint8_t*>(next_send_packed),
        reinterpret_cast<__nv_bfloat16*>(next_keep_bf16),
        stride_local, local_row_off, recv_scale_off, next_scale_off, stride_keep,
        N, H, send_upper, wb_keep
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

int rhd_dequant_add_quant_int8_g64_fwd(
    const void* recv_packed,
    const void* local_bf16,
    void* next_send_packed,
    int64_t stride_local,
    int64_t local_row_off,
    int64_t recv_scale_off,
    int64_t next_scale_off,
    int M,
    int H,
    uint64_t stream_u64
) {
    if (M <= 0 || H <= 0 || (H % 128) != 0) return -1;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_u64);
    if (v11_is_stream_capturing(stream)) return -4;
    rhd_dequant_add_quant_g64_kernel<false><<<dim3((unsigned)M, 1, 1), dim3(128, 1, 1), 0, stream>>>(
        reinterpret_cast<const uint8_t*>(recv_packed),
        reinterpret_cast<const __nv_bfloat16*>(local_bf16),
        reinterpret_cast<uint8_t*>(next_send_packed),
        stride_local, local_row_off, recv_scale_off, next_scale_off, H
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

int rhd_dequant_add_quant_fp8_g64_fwd(
    const void* recv_packed,
    const void* local_bf16,
    void* next_send_packed,
    int64_t stride_local,
    int64_t local_row_off,
    int64_t recv_scale_off,
    int64_t next_scale_off,
    int M,
    int H,
    uint64_t stream_u64
) {
    if (M <= 0 || H <= 0 || (H % 128) != 0) return -1;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_u64);
    if (v11_is_stream_capturing(stream)) return -4;
    rhd_dequant_add_quant_g64_kernel<true><<<dim3((unsigned)M, 1, 1), dim3(128, 1, 1), 0, stream>>>(
        reinterpret_cast<const uint8_t*>(recv_packed),
        reinterpret_cast<const __nv_bfloat16*>(local_bf16),
        reinterpret_cast<uint8_t*>(next_send_packed),
        stride_local, local_row_off, recv_scale_off, next_scale_off, H
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

int rhd_dequant_add_bf16_int8_g64_fwd(
    const void* recv_packed,
    const void* local_bf16,
    void* out_bf16,
    int64_t stride_local,
    int64_t local_row_off,
    int64_t recv_scale_off,
    int64_t stride_out,
    int M,
    int H,
    int wb_out,
    uint64_t stream_u64
) {
    if (M <= 0 || H <= 0 || (H % 128) != 0) return -1;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_u64);
    if (v11_is_stream_capturing(stream)) return -4;
    rhd_dequant_add_bf16_g64_kernel<false><<<dim3((unsigned)M, 1, 1), dim3(128, 1, 1), 0, stream>>>(
        reinterpret_cast<const uint8_t*>(recv_packed),
        reinterpret_cast<const __nv_bfloat16*>(local_bf16),
        reinterpret_cast<__nv_bfloat16*>(out_bf16),
        stride_local, local_row_off, recv_scale_off, stride_out, H, wb_out
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

int rhd_dequant_add_bf16_fp8_g64_fwd(
    const void* recv_packed,
    const void* local_bf16,
    void* out_bf16,
    int64_t stride_local,
    int64_t local_row_off,
    int64_t recv_scale_off,
    int64_t stride_out,
    int M,
    int H,
    int wb_out,
    uint64_t stream_u64
) {
    if (M <= 0 || H <= 0 || (H % 128) != 0) return -1;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_u64);
    if (v11_is_stream_capturing(stream)) return -4;
    rhd_dequant_add_bf16_g64_kernel<true><<<dim3((unsigned)M, 1, 1), dim3(128, 1, 1), 0, stream>>>(
        reinterpret_cast<const uint8_t*>(recv_packed),
        reinterpret_cast<const __nv_bfloat16*>(local_bf16),
        reinterpret_cast<__nv_bfloat16*>(out_bf16),
        stride_local, local_row_off, recv_scale_off, stride_out, H, wb_out
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

int rhd_ag_dequant_int8_g64_fwd(
    const void* ag_buf,
    void* out_bf16,
    int64_t shard_bytes,
    int64_t scale_off,
    int64_t stride_out,
    int S,
    int H,
    int W,
    uint64_t stream_u64
) {
    if (S <= 0 || H <= 0 || W <= 0 || (H % 128) != 0) return -1;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_u64);
    if (v11_is_stream_capturing(stream)) return -4;
    rhd_ag_dequant_g64_kernel<false><<<dim3((unsigned)S, (unsigned)W, 1), dim3(128, 1, 1), 0, stream>>>(
        reinterpret_cast<const uint8_t*>(ag_buf),
        reinterpret_cast<__nv_bfloat16*>(out_bf16),
        shard_bytes, scale_off, stride_out, S, H
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

int rhd_ag_dequant_fp8_g64_fwd(
    const void* ag_buf,
    void* out_bf16,
    int64_t shard_bytes,
    int64_t scale_off,
    int64_t stride_out,
    int S,
    int H,
    int W,
    uint64_t stream_u64
) {
    if (S <= 0 || H <= 0 || W <= 0 || (H % 128) != 0) return -1;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_u64);
    if (v11_is_stream_capturing(stream)) return -4;
    rhd_ag_dequant_g64_kernel<true><<<dim3((unsigned)S, (unsigned)W, 1), dim3(128, 1, 1), 0, stream>>>(
        reinterpret_cast<const uint8_t*>(ag_buf),
        reinterpret_cast<__nv_bfloat16*>(out_bf16),
        shard_bytes, scale_off, stride_out, S, H
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

int rhd_ag_quant_fp8_rowwise_fwd(
    const void* x_bf16,
    void* out_shard_packed,
    int64_t stride_xm,
    int64_t scale_off,
    int S,
    int H,
    uint64_t stream_u64
) {
    if (S <= 0 || H <= 0 || (H % 16) != 0) return -1;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_u64);
    if (v11_is_stream_capturing(stream)) return -4;
    rhd_ag_quant_fp8_rowwise_kernel<<<dim3((unsigned)S, 1, 1), dim3(128, 1, 1), 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x_bf16),
        reinterpret_cast<uint8_t*>(out_shard_packed),
        stride_xm, scale_off, H
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

int rhd_ag_unpack_fp8_rowwise_fwd(
    const void* ag_buf,
    void* out_fp8,
    void* out_scale,
    int64_t shard_bytes,
    int64_t scale_off,
    int64_t stride_out,
    int S,
    int H,
    int W,
    uint64_t stream_u64
) {
    if (S <= 0 || H <= 0 || W <= 0 || (H % 16) != 0) return -1;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_u64);
    if (v11_is_stream_capturing(stream)) return -4;
    rhd_ag_unpack_fp8_rowwise_kernel<<<dim3((unsigned)S, (unsigned)W, 1), dim3(128, 1, 1), 0, stream>>>(
        reinterpret_cast<const uint8_t*>(ag_buf),
        reinterpret_cast<uint8_t*>(out_fp8),
        reinterpret_cast<float*>(out_scale),
        shard_bytes, scale_off, stride_out, S, H
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

// ---- v5 Low-SM (`max_ctas = 4..8`) L2-Streaming Entrypoints ----

int rhd_quant_slice_int8_g64_lowsm_fwd(
    const void* x_bf16,
    void* out_packed,
    int64_t stride_xm,
    int64_t row_off,
    int64_t scale_off,
    int M,
    int H,
    int max_ctas,
    uint64_t stream_u64
) {
    if (M <= 0 || H <= 0 || (H % 128) != 0 || (scale_off & 3) != 0) return -1;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_u64);
    if (v11_is_stream_capturing(stream)) return -4;
    unsigned grid = clamp_lowsm_ctas(M, max_ctas);
    rhd_quant_slice_g64_lowsm_kernel<false><<<dim3(grid, 1, 1), dim3(256, 1, 1), 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x_bf16),
        reinterpret_cast<uint8_t*>(out_packed),
        stride_xm, row_off, scale_off, M, H
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

int rhd_quant_slice_fp8_g64_lowsm_fwd(
    const void* x_bf16,
    void* out_packed,
    int64_t stride_xm,
    int64_t row_off,
    int64_t scale_off,
    int M,
    int H,
    int max_ctas,
    uint64_t stream_u64
) {
    if (M <= 0 || H <= 0 || (H % 128) != 0 || (scale_off & 3) != 0) return -1;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_u64);
    if (v11_is_stream_capturing(stream)) return -4;
    unsigned grid = clamp_lowsm_ctas(M, max_ctas);
    rhd_quant_slice_g64_lowsm_kernel<true><<<dim3(grid, 1, 1), dim3(256, 1, 1), 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x_bf16),
        reinterpret_cast<uint8_t*>(out_packed),
        stride_xm, row_off, scale_off, M, H
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

int rhd_dequant_add_split_quant_int8_g64_lowsm_fwd(
    const void* recv_packed,
    const void* local_bf16,
    void* next_send_packed,
    void* next_keep_bf16,
    int64_t stride_local,
    int64_t local_row_off,
    int64_t recv_scale_off,
    int64_t next_scale_off,
    int64_t stride_keep,
    int N,
    int H,
    int send_upper,
    int wb_keep,
    int max_ctas,
    uint64_t stream_u64
) {
    if (N <= 0 || H <= 0 || (H % 128) != 0 || (recv_scale_off & 3) != 0 || (next_scale_off & 3) != 0) {
        return -1;
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_u64);
    if (v11_is_stream_capturing(stream)) return -4;
    unsigned grid = clamp_lowsm_ctas(2 * N, max_ctas);
    rhd_dequant_add_split_quant_g64_lowsm_kernel<false><<<dim3(grid, 1, 1), dim3(256, 1, 1), 0, stream>>>(
        reinterpret_cast<const uint8_t*>(recv_packed),
        reinterpret_cast<const __nv_bfloat16*>(local_bf16),
        reinterpret_cast<uint8_t*>(next_send_packed),
        reinterpret_cast<__nv_bfloat16*>(next_keep_bf16),
        stride_local, local_row_off, recv_scale_off, next_scale_off, stride_keep,
        N, H, send_upper, wb_keep
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

int rhd_dequant_add_split_quant_fp8_g64_lowsm_fwd(
    const void* recv_packed,
    const void* local_bf16,
    void* next_send_packed,
    void* next_keep_bf16,
    int64_t stride_local,
    int64_t local_row_off,
    int64_t recv_scale_off,
    int64_t next_scale_off,
    int64_t stride_keep,
    int N,
    int H,
    int send_upper,
    int wb_keep,
    int max_ctas,
    uint64_t stream_u64
) {
    if (N <= 0 || H <= 0 || (H % 128) != 0 || (recv_scale_off & 3) != 0 || (next_scale_off & 3) != 0) {
        return -1;
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_u64);
    if (v11_is_stream_capturing(stream)) return -4;
    unsigned grid = clamp_lowsm_ctas(2 * N, max_ctas);
    rhd_dequant_add_split_quant_g64_lowsm_kernel<true><<<dim3(grid, 1, 1), dim3(256, 1, 1), 0, stream>>>(
        reinterpret_cast<const uint8_t*>(recv_packed),
        reinterpret_cast<const __nv_bfloat16*>(local_bf16),
        reinterpret_cast<uint8_t*>(next_send_packed),
        reinterpret_cast<__nv_bfloat16*>(next_keep_bf16),
        stride_local, local_row_off, recv_scale_off, next_scale_off, stride_keep,
        N, H, send_upper, wb_keep
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

int rhd_dequant_add_quant_int8_g64_lowsm_fwd(
    const void* recv_packed,
    const void* local_bf16,
    void* next_send_packed,
    int64_t stride_local,
    int64_t local_row_off,
    int64_t recv_scale_off,
    int64_t next_scale_off,
    int M,
    int H,
    int max_ctas,
    uint64_t stream_u64
) {
    if (M <= 0 || H <= 0 || (H % 128) != 0 || (recv_scale_off & 3) != 0 || (next_scale_off & 3) != 0) {
        return -1;
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_u64);
    if (v11_is_stream_capturing(stream)) return -4;
    unsigned grid = clamp_lowsm_ctas(M, max_ctas);
    rhd_dequant_add_quant_g64_lowsm_kernel<false><<<dim3(grid, 1, 1), dim3(256, 1, 1), 0, stream>>>(
        reinterpret_cast<const uint8_t*>(recv_packed),
        reinterpret_cast<const __nv_bfloat16*>(local_bf16),
        reinterpret_cast<uint8_t*>(next_send_packed),
        stride_local, local_row_off, recv_scale_off, next_scale_off, M, H
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

int rhd_dequant_add_quant_fp8_g64_lowsm_fwd(
    const void* recv_packed,
    const void* local_bf16,
    void* next_send_packed,
    int64_t stride_local,
    int64_t local_row_off,
    int64_t recv_scale_off,
    int64_t next_scale_off,
    int M,
    int H,
    int max_ctas,
    uint64_t stream_u64
) {
    if (M <= 0 || H <= 0 || (H % 128) != 0 || (recv_scale_off & 3) != 0 || (next_scale_off & 3) != 0) {
        return -1;
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_u64);
    if (v11_is_stream_capturing(stream)) return -4;
    unsigned grid = clamp_lowsm_ctas(M, max_ctas);
    rhd_dequant_add_quant_g64_lowsm_kernel<true><<<dim3(grid, 1, 1), dim3(256, 1, 1), 0, stream>>>(
        reinterpret_cast<const uint8_t*>(recv_packed),
        reinterpret_cast<const __nv_bfloat16*>(local_bf16),
        reinterpret_cast<uint8_t*>(next_send_packed),
        stride_local, local_row_off, recv_scale_off, next_scale_off, M, H
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

int rhd_dequant_add_bf16_int8_g64_lowsm_fwd(
    const void* recv_packed,
    const void* local_bf16,
    void* out_bf16,
    int64_t stride_local,
    int64_t local_row_off,
    int64_t recv_scale_off,
    int64_t stride_out,
    int M,
    int H,
    int wb_out,
    int max_ctas,
    uint64_t stream_u64
) {
    if (M <= 0 || H <= 0 || (H % 128) != 0 || (recv_scale_off & 3) != 0) return -1;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_u64);
    if (v11_is_stream_capturing(stream)) return -4;
    unsigned grid = clamp_lowsm_ctas(M, max_ctas);
    rhd_dequant_add_bf16_g64_lowsm_kernel<false><<<dim3(grid, 1, 1), dim3(256, 1, 1), 0, stream>>>(
        reinterpret_cast<const uint8_t*>(recv_packed),
        reinterpret_cast<const __nv_bfloat16*>(local_bf16),
        reinterpret_cast<__nv_bfloat16*>(out_bf16),
        stride_local, local_row_off, recv_scale_off, stride_out, M, H, wb_out
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

int rhd_dequant_add_bf16_fp8_g64_lowsm_fwd(
    const void* recv_packed,
    const void* local_bf16,
    void* out_bf16,
    int64_t stride_local,
    int64_t local_row_off,
    int64_t recv_scale_off,
    int64_t stride_out,
    int M,
    int H,
    int wb_out,
    int max_ctas,
    uint64_t stream_u64
) {
    if (M <= 0 || H <= 0 || (H % 128) != 0 || (recv_scale_off & 3) != 0) return -1;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_u64);
    if (v11_is_stream_capturing(stream)) return -4;
    unsigned grid = clamp_lowsm_ctas(M, max_ctas);
    rhd_dequant_add_bf16_g64_lowsm_kernel<true><<<dim3(grid, 1, 1), dim3(256, 1, 1), 0, stream>>>(
        reinterpret_cast<const uint8_t*>(recv_packed),
        reinterpret_cast<const __nv_bfloat16*>(local_bf16),
        reinterpret_cast<__nv_bfloat16*>(out_bf16),
        stride_local, local_row_off, recv_scale_off, stride_out, M, H, wb_out
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

int a2a_quant_int8_g64_lowsm_fwd(
    const void* x_bf16,
    void* send_buf,
    int64_t stride_xm,
    int64_t peer_bytes,
    int64_t scale_off,
    int S,
    int H,
    int world_size,
    int rank,
    int max_ctas,
    uint64_t stream_u64
) {
    if (S <= 0 || H <= 0 || world_size <= 1 || rank < 0 || rank >= world_size ||
        (H % 128) != 0 || (peer_bytes & 3) != 0 || (scale_off & 3) != 0) {
        return -1;
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_u64);
    if (v11_is_stream_capturing(stream)) return -4;
    unsigned grid = clamp_lowsm_ctas((world_size - 1) * S, max_ctas);
    a2a_quant_g64_lowsm_kernel<false><<<dim3(grid, 1, 1), dim3(256, 1, 1), 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x_bf16),
        reinterpret_cast<uint8_t*>(send_buf),
        stride_xm, peer_bytes, scale_off, S, H, world_size, rank
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

int a2a_quant_fp8_g64_lowsm_fwd(
    const void* x_bf16,
    void* send_buf,
    int64_t stride_xm,
    int64_t peer_bytes,
    int64_t scale_off,
    int S,
    int H,
    int world_size,
    int rank,
    int max_ctas,
    uint64_t stream_u64
) {
    if (S <= 0 || H <= 0 || world_size <= 1 || rank < 0 || rank >= world_size ||
        (H % 128) != 0 || (peer_bytes & 3) != 0 || (scale_off & 3) != 0) {
        return -1;
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_u64);
    if (v11_is_stream_capturing(stream)) return -4;
    unsigned grid = clamp_lowsm_ctas((world_size - 1) * S, max_ctas);
    a2a_quant_g64_lowsm_kernel<true><<<dim3(grid, 1, 1), dim3(256, 1, 1), 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x_bf16),
        reinterpret_cast<uint8_t*>(send_buf),
        stride_xm, peer_bytes, scale_off, S, H, world_size, rank
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

int a2a_dequant_sum8_int8_g64_lowsm_fwd(
    const void* recv_buf,
    const void* self_x_bf16,
    void* out_shard_bf16,
    int64_t stride_xm,
    int64_t stride_om,
    int64_t peer_bytes,
    int64_t scale_off,
    int S,
    int H,
    int world_size,
    int rank,
    int wb_out,
    int max_ctas,
    uint64_t stream_u64
) {
    if (S <= 0 || H <= 0 || world_size <= 1 || rank < 0 || rank >= world_size ||
        (H % 128) != 0 || (peer_bytes & 3) != 0 || (scale_off & 3) != 0) {
        return -1;
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_u64);
    if (v11_is_stream_capturing(stream)) return -4;
    unsigned grid = clamp_lowsm_ctas(S, max_ctas);
    a2a_dequant_sum8_g64_lowsm_kernel<false><<<dim3(grid, 1, 1), dim3(256, 1, 1), 0, stream>>>(
        reinterpret_cast<const uint8_t*>(recv_buf),
        reinterpret_cast<const __nv_bfloat16*>(self_x_bf16),
        reinterpret_cast<__nv_bfloat16*>(out_shard_bf16),
        stride_xm, stride_om, peer_bytes, scale_off, S, H, world_size, rank, wb_out
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

int a2a_dequant_sum8_fp8_g64_lowsm_fwd(
    const void* recv_buf,
    const void* self_x_bf16,
    void* out_shard_bf16,
    int64_t stride_xm,
    int64_t stride_om,
    int64_t peer_bytes,
    int64_t scale_off,
    int S,
    int H,
    int world_size,
    int rank,
    int wb_out,
    int max_ctas,
    uint64_t stream_u64
) {
    if (S <= 0 || H <= 0 || world_size <= 1 || rank < 0 || rank >= world_size ||
        (H % 128) != 0 || (peer_bytes & 3) != 0 || (scale_off & 3) != 0) {
        return -1;
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_u64);
    if (v11_is_stream_capturing(stream)) return -4;
    unsigned grid = clamp_lowsm_ctas(S, max_ctas);
    a2a_dequant_sum8_g64_lowsm_kernel<true><<<dim3(grid, 1, 1), dim3(256, 1, 1), 0, stream>>>(
        reinterpret_cast<const uint8_t*>(recv_buf),
        reinterpret_cast<const __nv_bfloat16*>(self_x_bf16),
        reinterpret_cast<__nv_bfloat16*>(out_shard_bf16),
        stride_xm, stride_om, peer_bytes, scale_off, S, H, world_size, rank, wb_out
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

int rhd_ag_quant_fp8_rowwise_lowsm_fwd(
    const void* x_bf16,
    void* out_shard_packed,
    int64_t stride_xm,
    int64_t scale_off,
    int S,
    int H,
    int max_ctas,
    uint64_t stream_u64
) {
    if (S <= 0 || H <= 0 || (H % 16) != 0 || (scale_off & 3) != 0) return -1;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_u64);
    if (v11_is_stream_capturing(stream)) return -4;
    unsigned grid = clamp_lowsm_ctas(S, max_ctas);
    rhd_ag_quant_fp8_rowwise_lowsm_kernel<<<dim3(grid, 1, 1), dim3(256, 1, 1), 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x_bf16),
        reinterpret_cast<uint8_t*>(out_shard_packed),
        stride_xm, scale_off, S, H
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

int rhd_ag_unpack_fp8_rowwise_lowsm_fwd(
    const void* ag_buf,
    void* out_fp8,
    void* out_scale,
    int64_t shard_bytes,
    int64_t scale_off,
    int64_t stride_out,
    int S,
    int H,
    int W,
    int wb_out,
    int max_ctas,
    uint64_t stream_u64
) {
    if (S <= 0 || H <= 0 || W <= 0 || (H % 16) != 0 || (shard_bytes & 15) != 0 || (scale_off & 3) != 0) {
        return -1;
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_u64);
    if (v11_is_stream_capturing(stream)) return -4;
    unsigned grid = clamp_lowsm_ctas(W * S, max_ctas);
    rhd_ag_unpack_fp8_rowwise_lowsm_kernel<<<dim3(grid, 1, 1), dim3(256, 1, 1), 0, stream>>>(
        reinterpret_cast<const uint8_t*>(ag_buf),
        reinterpret_cast<uint8_t*>(out_fp8),
        reinterpret_cast<float*>(out_scale),
        shard_bytes, scale_off, stride_out, S, H, W, wb_out
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

int rhd_ag_dequant_fp8_rowwise_lowsm_fwd(
    const void* ag_buf,
    void* out_bf16,
    int64_t shard_bytes,
    int64_t scale_off,
    int64_t stride_out,
    int S,
    int H,
    int W,
    int wb_out,
    int max_ctas,
    uint64_t stream_u64
) {
    if (S <= 0 || H <= 0 || W <= 0 || (H % 16) != 0 || (shard_bytes & 3) != 0 || (scale_off & 3) != 0) {
        return -1;
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_u64);
    if (v11_is_stream_capturing(stream)) return -4;
    unsigned grid = clamp_lowsm_ctas(W * S, max_ctas);
    rhd_ag_dequant_fp8_rowwise_lowsm_kernel<<<dim3(grid, 1, 1), dim3(256, 1, 1), 0, stream>>>(
        reinterpret_cast<const uint8_t*>(ag_buf),
        reinterpret_cast<__nv_bfloat16*>(out_bf16),
        shard_bytes, scale_off, stride_out, S, H, W, wb_out
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

int rhd_ag_dequant_int8_g64_lowsm_fwd(
    const void* ag_buf,
    void* out_bf16,
    int64_t shard_bytes,
    int64_t scale_off,
    int64_t stride_out,
    int S,
    int H,
    int W,
    int wb_out,
    int max_ctas,
    uint64_t stream_u64
) {
    if (S <= 0 || H <= 0 || W <= 0 || (H % 128) != 0 || (shard_bytes & 3) != 0 || (scale_off & 3) != 0) {
        return -1;
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_u64);
    if (v11_is_stream_capturing(stream)) return -4;
    unsigned grid = clamp_lowsm_ctas(W * S, max_ctas);
    rhd_ag_dequant_g64_lowsm_kernel<false><<<dim3(grid, 1, 1), dim3(256, 1, 1), 0, stream>>>(
        reinterpret_cast<const uint8_t*>(ag_buf),
        reinterpret_cast<__nv_bfloat16*>(out_bf16),
        shard_bytes, scale_off, stride_out, S, H, W, wb_out
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

int rhd_ag_dequant_fp8_g64_lowsm_fwd(
    const void* ag_buf,
    void* out_bf16,
    int64_t shard_bytes,
    int64_t scale_off,
    int64_t stride_out,
    int S,
    int H,
    int W,
    int wb_out,
    int max_ctas,
    uint64_t stream_u64
) {
    if (S <= 0 || H <= 0 || W <= 0 || (H % 128) != 0 || (shard_bytes & 3) != 0 || (scale_off & 3) != 0) {
        return -1;
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_u64);
    if (v11_is_stream_capturing(stream)) return -4;
    unsigned grid = clamp_lowsm_ctas(W * S, max_ctas);
    rhd_ag_dequant_g64_lowsm_kernel<true><<<dim3(grid, 1, 1), dim3(256, 1, 1), 0, stream>>>(
        reinterpret_cast<const uint8_t*>(ag_buf),
        reinterpret_cast<__nv_bfloat16*>(out_bf16),
        shard_bytes, scale_off, stride_out, S, H, W, wb_out
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}


// ---- Wave-V11 Entrypoints ----

int pf_rhd_comm_v13_version() {
    return 1300;  // Wave-V13 Asymmetric Half-1 A2A_INT8_pipe + DBO-Throttled Fused SConv+AR+FP8_AG!
}

int pf_rhd_comm_v11_version() {
    return 1300;  // Back-compat V11 symbol reports 1300
}

int a2a_quant_int8_g64_slice_lowsm_fwd(
    const void* x_bf16,
    void* send_buf,
    int64_t stride_xm,
    int64_t peer_bytes,
    int64_t scale_off,
    int S,
    int row_beg,
    int row_cnt,
    int H,
    int world_size,
    int rank,
    int max_ctas,
    uint64_t stream_u64
) {
    if (S <= 0 || row_beg < 0 || row_cnt <= 0 || row_beg + row_cnt > S ||
        H <= 0 || world_size <= 1 || rank < 0 || rank >= world_size ||
        (H % 128) != 0 || (peer_bytes & 3) != 0 || (scale_off & 3) != 0) {
        return -1;
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_u64);
    if (v11_is_stream_capturing(stream)) return -4;
    unsigned grid = clamp_lowsm_ctas((world_size - 1) * row_cnt, max_ctas);
    a2a_quant_g64_slice_lowsm_kernel<false><<<dim3(grid, 1, 1), dim3(256, 1, 1), 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x_bf16),
        reinterpret_cast<uint8_t*>(send_buf),
        stride_xm, peer_bytes, scale_off, S, row_beg, row_cnt, H, world_size, rank
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

int a2a_quant_fp8_g64_slice_lowsm_fwd(
    const void* x_bf16,
    void* send_buf,
    int64_t stride_xm,
    int64_t peer_bytes,
    int64_t scale_off,
    int S,
    int row_beg,
    int row_cnt,
    int H,
    int world_size,
    int rank,
    int max_ctas,
    uint64_t stream_u64
) {
    if (S <= 0 || row_beg < 0 || row_cnt <= 0 || row_beg + row_cnt > S ||
        H <= 0 || world_size <= 1 || rank < 0 || rank >= world_size ||
        (H % 128) != 0 || (peer_bytes & 3) != 0 || (scale_off & 3) != 0) {
        return -1;
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_u64);
    if (v11_is_stream_capturing(stream)) return -4;
    unsigned grid = clamp_lowsm_ctas((world_size - 1) * row_cnt, max_ctas);
    a2a_quant_g64_slice_lowsm_kernel<true><<<dim3(grid, 1, 1), dim3(256, 1, 1), 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x_bf16),
        reinterpret_cast<uint8_t*>(send_buf),
        stride_xm, peer_bytes, scale_off, S, row_beg, row_cnt, H, world_size, rank
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

int a2a_dequant_sum8_int8_g64_slice_lowsm_fwd(
    const void* recv_buf,
    const void* self_x_bf16,
    void* out_shard_bf16,
    int64_t stride_xm,
    int64_t stride_om,
    int64_t self_row_base,
    int64_t peer_bytes,
    int64_t scale_off,
    int S,
    int row_beg,
    int recv_row_beg,
    int row_cnt,
    int H,
    int world_size,
    int rank,
    int wb_out,
    int max_ctas,
    uint64_t stream_u64
) {
    if (S <= 0 || row_beg < 0 || recv_row_beg < 0 || row_cnt <= 0 ||
        row_beg + row_cnt > S || recv_row_beg + row_cnt > S ||
        self_row_base < 0 || H <= 0 || world_size <= 1 || rank < 0 || rank >= world_size ||
        (H % 128) != 0 || (peer_bytes & 3) != 0 || (scale_off & 3) != 0) {
        return -1;
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_u64);
    if (v11_is_stream_capturing(stream)) return -4;
    unsigned grid = clamp_lowsm_ctas(row_cnt, max_ctas);
    a2a_dequant_sum8_g64_slice_lowsm_kernel<false><<<dim3(grid, 1, 1), dim3(256, 1, 1), 0, stream>>>(
        reinterpret_cast<const uint8_t*>(recv_buf),
        reinterpret_cast<const __nv_bfloat16*>(self_x_bf16),
        reinterpret_cast<__nv_bfloat16*>(out_shard_bf16),
        stride_xm, stride_om, self_row_base, peer_bytes, scale_off,
        row_beg, recv_row_beg, row_cnt, H, world_size, rank, wb_out
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

int a2a_dequant_sum8_fp8_g64_slice_lowsm_fwd(
    const void* recv_buf,
    const void* self_x_bf16,
    void* out_shard_bf16,
    int64_t stride_xm,
    int64_t stride_om,
    int64_t self_row_base,
    int64_t peer_bytes,
    int64_t scale_off,
    int S,
    int row_beg,
    int recv_row_beg,
    int row_cnt,
    int H,
    int world_size,
    int rank,
    int wb_out,
    int max_ctas,
    uint64_t stream_u64
) {
    if (S <= 0 || row_beg < 0 || recv_row_beg < 0 || row_cnt <= 0 ||
        row_beg + row_cnt > S || recv_row_beg + row_cnt > S ||
        self_row_base < 0 || H <= 0 || world_size <= 1 || rank < 0 || rank >= world_size ||
        (H % 128) != 0 || (peer_bytes & 3) != 0 || (scale_off & 3) != 0) {
        return -1;
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_u64);
    if (v11_is_stream_capturing(stream)) return -4;
    unsigned grid = clamp_lowsm_ctas(row_cnt, max_ctas);
    a2a_dequant_sum8_g64_slice_lowsm_kernel<true><<<dim3(grid, 1, 1), dim3(256, 1, 1), 0, stream>>>(
        reinterpret_cast<const uint8_t*>(recv_buf),
        reinterpret_cast<const __nv_bfloat16*>(self_x_bf16),
        reinterpret_cast<__nv_bfloat16*>(out_shard_bf16),
        stride_xm, stride_om, self_row_base, peer_bytes, scale_off,
        row_beg, recv_row_beg, row_cnt, H, world_size, rank, wb_out
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

int fused_sconv_ar_fp8_ag_v11_fwd(
    const void* x_ptr,
    const void* cw_ptr,
    void* cache_ptr,
    void* conv_out_ptr,
    const void* conv_res_ptr,
    const void* conv_add_ptr,
    const void* pos_ptr,
    const void* seq_idx_ptr,
    const void* slot_ptr,
    const void* bt_ptr,
    const void* qstart_ptr,
    const void* stack_ptr,
    const void* mix_last_ptr,
    const void* queries_ptr,
    const void* norm_w_ptr,
    void* out_fp8_ptr,
    void* out_scale_ptr,
    void* out_aux_bf16_ptr,
    const void* discard_blk_ptr,
    int64_t stride_x,
    int64_t stride_c_blk,
    int64_t stride_c_n,
    int64_t stride_bt,
    int max_blocks,
    int n_cache,
    int stream_offset,
    int64_t stride_s,
    int nq,
    int T,
    int n_stack,
    float scale,
    float eps,
    float norm_eps,
    int has_conv_res,
    int has_conv_addend,
    int has_mix_last,
    int write_fp8_q0,
    int max_ctas,
    uint64_t stream_u64
) {
    if (T <= 0 || (nq != 1 && nq != 3) || n_stack < 1 || n_stack > 8 || n_cache <= 0 || max_blocks <= 0) {
        return -1;
    }
    if (write_fp8_q0 != 0 && (out_fp8_ptr == nullptr || out_scale_ptr == nullptr)) {
        return -1;
    }
    if ((write_fp8_q0 == 0 || nq == 3) && out_aux_bf16_ptr == nullptr) {
        return -1;
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_u64);
    if (v11_is_stream_capturing(stream)) return -4;

    int n_shift = -1;
    if ((n_cache & (n_cache - 1)) == 0) {
        int s = 0;
        int tmp = n_cache;
        while (tmp > 1) { tmp >>= 1; ++s; }
        n_shift = s;
    }

    const auto* X       = reinterpret_cast<const __nv_bfloat16*>(x_ptr);
    const auto* CW      = reinterpret_cast<const __nv_bfloat16*>(cw_ptr);
    auto*       Cache   = reinterpret_cast<__nv_bfloat16*>(cache_ptr);
    auto*       ConvOut = reinterpret_cast<__nv_bfloat16*>(conv_out_ptr);
    const auto* ConvRes = reinterpret_cast<const __nv_bfloat16*>(conv_res_ptr);
    const auto* ConvAdd = reinterpret_cast<const __nv_bfloat16*>(conv_add_ptr);
    const auto* Pos     = reinterpret_cast<const int64_t*>(pos_ptr);
    const auto* SeqIdx  = reinterpret_cast<const int32_t*>(seq_idx_ptr);
    const auto* Slot    = reinterpret_cast<const int64_t*>(slot_ptr);
    const auto* BT      = reinterpret_cast<const int32_t*>(bt_ptr);
    const auto* QStart  = reinterpret_cast<const int32_t*>(qstart_ptr);
    const auto* Stack   = reinterpret_cast<const __nv_bfloat16*>(stack_ptr);
    const auto* MixLast = reinterpret_cast<const __nv_bfloat16*>(mix_last_ptr);
    const auto* Queries = reinterpret_cast<const float*>(queries_ptr);
    const auto* NormW   = reinterpret_cast<const __nv_bfloat16*>(norm_w_ptr);
    auto*       OutFp8  = reinterpret_cast<uint8_t*>(out_fp8_ptr);
    auto*       OutSc   = reinterpret_cast<float*>(out_scale_ptr);
    auto*       OutAux  = reinterpret_cast<__nv_bfloat16*>(out_aux_bf16_ptr);
    const auto* DiscPtr = reinterpret_cast<const int64_t*>(discard_blk_ptr);

    const bool hcr = (has_conv_res != 0);
    const bool hca = (has_conv_addend != 0);
    const bool hml = (has_mix_last != 0);

    cudaError_t e = cudaErrorUnknown;
    #define V11_CASE_NS(NS_VAL) \
        case NS_VAL: \
            e = (write_fp8_q0 != 0) \
                ? v11_dispatch_sconv_ar_ns<NS_VAL, true>( \
                    nq, T, max_ctas, stream, X, CW, Cache, ConvOut, ConvRes, ConvAdd, \
                    Pos, SeqIdx, Slot, BT, QStart, Stack, MixLast, Queries, NormW, \
                    OutFp8, OutSc, OutAux, DiscPtr, \
                    stride_x, stride_c_blk, stride_c_n, stride_bt, \
                    max_blocks, n_cache, n_shift, stream_offset, stride_s, \
                    scale, eps, norm_eps, hcr, hca, hml) \
                : v11_dispatch_sconv_ar_ns<NS_VAL, false>( \
                    nq, T, max_ctas, stream, X, CW, Cache, ConvOut, ConvRes, ConvAdd, \
                    Pos, SeqIdx, Slot, BT, QStart, Stack, MixLast, Queries, NormW, \
                    OutFp8, OutSc, OutAux, DiscPtr, \
                    stride_x, stride_c_blk, stride_c_n, stride_bt, \
                    max_blocks, n_cache, n_shift, stream_offset, stride_s, \
                    scale, eps, norm_eps, hcr, hca, hml); \
            break
    switch (n_stack) {
        V11_CASE_NS(1);
        V11_CASE_NS(2);
        V11_CASE_NS(3);
        V11_CASE_NS(4);
        V11_CASE_NS(5);
        V11_CASE_NS(6);
        V11_CASE_NS(7);
        V11_CASE_NS(8);
        default: return -1;
    }
    #undef V11_CASE_NS
    return (e == cudaSuccess) ? 0 : -2;
}

int fused_sconv_ar_fp8_ag_v13_fwd(
    const void* x_ptr, const void* cw_ptr, void* cache_ptr, void* conv_out_ptr,
    const void* conv_res_ptr, const void* conv_add_ptr, const void* pos_ptr,
    const void* seq_idx_ptr, const void* slot_ptr, const void* bt_ptr,
    const void* qstart_ptr, const void* stack_ptr, const void* mix_last_ptr,
    const void* queries_ptr, const void* norm_w_ptr, void* out_fp8_ptr,
    void* out_scale_ptr, void* out_aux_bf16_ptr, const void* discard_blk_ptr,
    int64_t stride_x, int64_t stride_c_blk, int64_t stride_c_n, int64_t stride_bt,
    int max_blocks, int n_cache, int stream_offset, int64_t stride_s,
    int nq, int T, int n_stack, float scale, float eps, float norm_eps,
    int has_conv_res, int has_conv_addend, int has_mix_last,
    int write_fp8_q0, int max_ctas, uint64_t stream_u64
) {
    return fused_sconv_ar_fp8_ag_v11_fwd(
        x_ptr, cw_ptr, cache_ptr, conv_out_ptr, conv_res_ptr, conv_add_ptr,
        pos_ptr, seq_idx_ptr, slot_ptr, bt_ptr, qstart_ptr, stack_ptr, mix_last_ptr,
        queries_ptr, norm_w_ptr, out_fp8_ptr, out_scale_ptr, out_aux_bf16_ptr, discard_blk_ptr,
        stride_x, stride_c_blk, stride_c_n, stride_bt, max_blocks, n_cache, stream_offset,
        stride_s, nq, T, n_stack, scale, eps, norm_eps, has_conv_res, has_conv_addend,
        has_mix_last, write_fp8_q0, max_ctas, stream_u64);
}

}  // extern "C"
