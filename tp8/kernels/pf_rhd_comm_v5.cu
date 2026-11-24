// SPDX-License-Identifier: Apache-2.0
// =============================================================================
// SM121a (NVIDIA DGX Spark GB10) Prefill-Only Compressed ReduceScatter &
// AllGather CUDA Kernels v5 (`libpf_rhd_comm_v5.so`)
// =============================================================================
//
// Strictly prefill-only (guarded by `T >= 2048` & `!capturing` in Python caller).
// Zero effect on C=1 / C=32 decode (exact_ink SHA-256 unchanged).
//
// Why Half-2 Regressed `+3.6 s` with Legacy All-SM `rhd_int8` — And How v5 Fixes It:
//   1. Legacy `rhd_quant_slice_g64_kernel` / `rhd_dequant_add_split_quant_g64_kernel`
//      launched `1 CTA/row` (`2,048..4,096` CTAs) on `comm_stream` (`priority = -1`),
//      preempting ALL 48 SMs away from Half-2 `MoE(A)` / `MoE(B)` on `compute_stream`
//      4 times per sublayer (8 times per layer = 384 full-GPU preemptions per chunk!).
//   2. Legacy `ca_local` + `wb_keep=1` / `wb_out=1` wrote `18.87 MiB + 9.44 MiB + 9.44 MiB`
//      into GB10's 24 MiB L2, evicting active `W2_v5` `Out` slices and `A` activations!
//   3. Legacy `SP_FP8_AG=1` on Half-2 launched 1,024-CTA `quantize_fp8_rowwise` (48-SM
//      preemption) + TWO separate NCCL Ring `all_gather` collectives (`x_fp8` + `x_scale`).
//
// `pf_rhd_comm_v5.cu` Additions (100% Backwards-Compatible Supersets of `pf_rhd_comm_sm121.cu`):
//   - Low-SM Grid-Stride Persistent Codec Kernels (`max_ctas = 4..8`, default `6`,
//     `__launch_bounds__(256, 2)`, `0 B` stack frame, `0 B` spills):
//       * Caps `comm_stream` occupancy to `<= 6/48` SMs (`12.5%`), leaving `>= 42/48` SMs
//         (`87.5%`) continuously executing `pf_nvfp4_w13_v4b` + `pf_nvfp4_w2_v5`!
//       * Zero-SMEM / Zero-Barrier `g64 + BF16 scale` warp-pair scale exchange (`lane 0` +
//         `lane 16` `__shfl_sync` -> single aligned 4B `st.global.cs.u32` / `ld.global.cs.u32`).
//       * Pure L2-Streaming (`100% ld.global.cs` + `st.global.cs` when `wb == 0`), bounding
//         Half-2 L2 footprint to `< 0.5 MiB` (`<= 4 MiB` cap with 8x margin, vs 16 MiB NCCL!).
//   - Three Low-SM Half-2 Collective Families:
//       (A) 3-Step Zero-Incast Butterfly RHD RS (`rhd_*_g64_lowsm_fwd`)
//       (B) 1-Hop Shift-Rotated AllToAll RS (`a2a_quant_*_g64_lowsm_fwd`,
//           `a2a_dequant_sum8_*_g64_lowsm_fwd` — single quant stage, exact BF16 self shard,
//           deterministic `r = 0..W-1` FP32 sum, `+3.5 dB` SQNR!)
//       (C) 1-Collective Packed Rowwise FP8 & g64 AllGather (`rhd_ag_quant_fp8_rowwise_lowsm_fwd`,
//           `rhd_ag_unpack_fp8_rowwise_lowsm_fwd`, `rhd_ag_dequant_fp8_rowwise_lowsm_fwd`,
//           `rhd_ag_dequant_*_g64_lowsm_fwd` — works with BOTH 4-HCA NCCL Ring `nccl_fp8_lowsm`
//           AND 3-step RHD `rhd_fp8_lowsm`, eliminating 48-SM `quantize_fp8_rowwise` AND the
//           2nd scale `all_gather`!).
//
// Build on GB10 (SM121a):
//   /usr/local/cuda/bin/nvcc -O3 -std=c++17 --shared -Xcompiler -fPIC \
//       -gencode arch=compute_121a,code=sm_121a -Xptxas -v \
//       pf_rhd_comm_v5.cu -o libpf_rhd_comm_v5.so
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
    unsigned grid = clamp_lowsm_ctas(W * S, max_ctas);
    rhd_ag_dequant_g64_lowsm_kernel<true><<<dim3(grid, 1, 1), dim3(256, 1, 1), 0, stream>>>(
        reinterpret_cast<const uint8_t*>(ag_buf),
        reinterpret_cast<__nv_bfloat16*>(out_bf16),
        shard_bytes, scale_off, stride_out, S, H, W, wb_out
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

}  // extern "C"
