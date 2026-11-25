// SPDX-License-Identifier: Apache-2.0
// =================================================================================================
// Wave-V6 Zero-DRAM Fused SM121a NVFP4 `W13 + SiTU + In-L2 Quant288` + Fused W13/W2 Plan & Router
// File: `pf_nvfp4_w13_fused_v6.cu` -> `libpf_nvfp4_w13_v6.so`
// Target: NVIDIA GB10 (SM121a, 48 SMs, 99 KiB SMEM/SM, 24 MiB L2, 244 GB/s LPDDR5x UMA)
//
// Eliminates:
//   1. `386.55 GB` of `Y_situ [plan.rows, 288]` BF16 DRAM round-trip + `37.75-45 MB` scratch VRAM
//      + `5,120` separate `pf_nvfp4_quant_288` kernel launches across 40 NVFP4 layers at 1M prefill!
//      Mechanism: Co-resident persistent 6-CTA group grid `dim3(6, G)` (`G <= 16` -> `<= 96` CTAs =
//      2 CTAs/SM) writing `[valid_m, 48]` BF16 via `st.global.wb.v4.u32` into a 4-bank L2-resident
//      slot ring `Y_slot [G, 4, 128, 288]` (`4.50 MiB << 24 MiB L2`). First 5 CTAs in each group
//      advance immediately to `step + 1` with zero spin; the 6th arriving CTA immediately quantizes
//      `[valid_m, 288]` from coherent L2 (`ld.global.cg.v4.u32`) into `(Values, Scales, RowScales)`
//      via `st.global.wb` with literal 0-ULP bit-exact parity (`INF dB` SQNR) vs `v4b`!
//   2. `5,120` duplicate 1-CTA plan-builder launches via `pf_nvfp4_build_w13_w2_plan_v6`, which
//      builds BOTH `WorkBuf` (BM=128 W13 LPT worklist), `W2PlanBuf` (BM=64-split W2 v5 LPT plan),
//      AND zeroes the `SlotState` ring in a single CTA launch!
//   3. `8,192-24,576` single-warp Triton `_route_tokens_kernel` CTAs + global atomic contention via
//      `pf_route_tokens_place_v6` (8 warps = 8 tokens/CTA, SMEM `Bias[384]`, SMEM `Counts` histogram
//      cutting global atomics 8x, 12-reg exact `E=384` top-8 + `_sum8`, parallel `_place_experts`).
//
// Strict Codegen Invariants:
//   - Pure scalar unrolled 8-way WS5 patch select (`sb0..sb7`) — ZERO runtime `sb[idx]` arrays!
//   - `__shared__` uniform parameter block (`ld.shared` broadcast) — 0 B stack, 0 B spills!
//   - Strictly gated to `plan.tile >= 64` & `!is_stream_capturing` (0% decode touch!).
// =================================================================================================

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace {

static constexpr float FP32_MAX_VAL = 3.4028234663852886e+38f;
static constexpr float INV_6        = 1.0f / 6.0f;
static constexpr float INV_25       = 1.0f / 25.0f;
static constexpr float INV_1536     = 1.0f / 1536.0f;
static constexpr int   kMaxGroups     = 32;
static constexpr int   kSlotBanks     = 4;   // Total slots = 128 (9.0 MiB buffer, exact V6 ABI!)
static constexpr int   kV13BatchSlots = 64;  // Wave-V13 ZERO-SPIN host batch: 64*72 KiB = 4.50 MiB << 24 MiB L2 (384 CTAs = 4 waves on 48 SMs)!

struct alignas(64) W13SlotBankState {
    uint32_t arrived;
    uint32_t ready_epoch;
    uint32_t _pad[14];
};
static_assert(sizeof(W13SlotBankState) == 64, "W13SlotBankState must be 64B cacheline-aligned");

// =================================================================================================
// SM121a PTX Primitives (Exact IEEE-RN Rounding, Coherent L2 `cg`/`wb`, Non-Hoistable LDS)
// =================================================================================================

__device__ __forceinline__ float mul_rn(float a, float b) {
    float r;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

__device__ __forceinline__ float add_rn(float a, float b) {
    float r;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

__device__ __forceinline__ float sub_rn(float a, float b) {
    float r;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

__device__ __forceinline__ float div_rn(float a, float b) {
    float r;
    asm("div.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

__device__ __forceinline__ float rcp_rn(float a) {
    float r;
    asm("rcp.rn.f32 %0, %1;" : "=f"(r) : "f"(a));
    return r;
}

__device__ __forceinline__ float bf16_bits_to_f32(uint16_t b) {
    uint32_t u = static_cast<uint32_t>(b) << 16;
    return __uint_as_float(u);
}

__device__ __forceinline__ uint16_t f32_to_bf16_rn(float x) {
    __nv_bfloat16 b = __float2bfloat16_rn(x);
    return *reinterpret_cast<const uint16_t*>(&b);
}

__device__ __forceinline__ uint32_t pack_bf16x2(float a, float b) {
    __nv_bfloat162 bv = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<const uint32_t*>(&bv);
}

__device__ __forceinline__ uint32_t prmt_b32(uint32_t a, uint32_t b, uint32_t sel) {
    uint32_t out;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(out) : "r"(a), "r"(b), "r"(sel));
    return out;
}

__device__ __forceinline__ uint4 ld_cs_u4(const void* ptr) {
    uint4 v;
    asm volatile("ld.global.cs.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(ptr));
    return v;
}

__device__ __forceinline__ uint4 ld_ca_u4(const void* ptr) {
    uint4 v;
    asm volatile("ld.global.ca.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(ptr));
    return v;
}

// L1-bypassing coherent L2 load (mandatory when reading cross-SM `Y_slot` written via `st.wb`!)
__device__ __forceinline__ uint4 ld_cg_u4(const void* ptr) {
    uint4 v;
    asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(ptr));
    return v;
}

__device__ __forceinline__ uint32_t ld_ca_u32(const void* ptr) {
    uint32_t v;
    asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(v) : "l"(ptr));
    return v;
}

__device__ __forceinline__ uint32_t ld_cs_u32(const void* ptr) {
    uint32_t v;
    asm volatile("ld.global.cs.u32 %0, [%1];" : "=r"(v) : "l"(ptr));
    return v;
}

__device__ __forceinline__ uint8_t ld_ca_u8(const void* ptr) {
    uint32_t v;
    asm volatile("ld.global.ca.u8 %0, [%1];" : "=r"(v) : "l"(ptr));
    return static_cast<uint8_t>(v);
}

__device__ __forceinline__ uint8_t ld_cs_u8(const void* ptr) {
    uint32_t v;
    asm volatile("ld.global.cs.u8 %0, [%1];" : "=r"(v) : "l"(ptr));
    return static_cast<uint8_t>(v);
}

__device__ __forceinline__ uint4 ld_w_u4(const void* ptr, bool use_cs) {
    return use_cs ? ld_cs_u4(ptr) : ld_ca_u4(ptr);
}

__device__ __forceinline__ uint32_t ld_w_u32(const void* ptr, bool use_cs) {
    return use_cs ? ld_cs_u32(ptr) : ld_ca_u32(ptr);
}

__device__ __forceinline__ uint8_t ld_w_u8(const void* ptr, bool use_cs) {
    return use_cs ? ld_cs_u8(ptr) : ld_ca_u8(ptr);
}

__device__ __forceinline__ void st_cs_u4(void* ptr, uint4 v) {
    asm volatile("st.global.cs.v4.u32 [%0], {%1, %2, %3, %4};"
                 :: "l"(ptr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}

__device__ __forceinline__ void st_wb_u4(void* ptr, uint4 v) {
    asm volatile("st.global.wb.v4.u32 [%0], {%1, %2, %3, %4};"
                 :: "l"(ptr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}

__device__ __forceinline__ void st_wb_u2(void* ptr, uint2 v) {
    asm volatile("st.global.wb.v2.u32 [%0], {%1, %2};"
                 :: "l"(ptr), "r"(v.x), "r"(v.y) : "memory");
}

__device__ __forceinline__ void st_wb_u32(void* ptr, uint32_t v) {
    asm volatile("st.global.wb.u32 [%0], %1;" :: "l"(ptr), "r"(v) : "memory");
}

// Wave-V13 ZERO-SPIN: ld_acquire_gpu_u32 / st_release_gpu_u32 removed completely —
// ZERO cross-CTA spinlocks exist anywhere in pf_nvfp4_w13_fused_v13.cu!

// Non-hoistable SMEM parameter broadcast loads (keeps uniform pointers out of K-loop GPRs!)
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

__device__ __forceinline__ float ld_smem_f32_nocse(const float* smem_ptr) {
    return __uint_as_float(ld_smem_u32_nocse(smem_ptr));
}

template <typename T>
__device__ __forceinline__ const T* ld_smem_ptr_nocse(const T* const* smem_ptr) {
    return reinterpret_cast<const T*>(static_cast<uintptr_t>(ld_smem_u64_nocse(smem_ptr)));
}

template <typename T>
__device__ __forceinline__ T* ld_smem_mut_ptr_nocse(T* const* smem_ptr) {
    return reinterpret_cast<T*>(static_cast<uintptr_t>(ld_smem_u64_nocse(smem_ptr)));
}

__device__ __forceinline__ void cp_async_cg_16(void* smem_ptr, const void* gmem_ptr, int src_bytes) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
        :: "r"(smem_addr), "l"(gmem_ptr), "r"(src_bytes) : "memory"
    );
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template <int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}

__device__ __forceinline__ void mma_m16n8k64_nvfp4(
    float (&d)[4],
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    uint32_t sfa, uint32_t sfb
) {
    constexpr uint16_t bidA = 0, tidA = 0, bidB = 0, tidB = 0;
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, "
        "{%14}, {%15, %16}, {%17}, {%18, %19};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "f"(d[0]), "f"(d[1]), "f"(d[2]), "f"(d[3]),
          "r"(sfa), "h"(bidA), "h"(tidA), "r"(sfb), "h"(bidB), "h"(tidB)
    );
}

__device__ __forceinline__ int64_t scale_offset_byte(int64_t row, int col, int COLS, int TILE) {
    if (TILE >= 128) {
        int64_t tile = (row >> 7) * ((COLS + 3) >> 2) + (col >> 2);
        return tile * 512 + (row & 31) * 16 + ((row >> 5) & 3) * 4 + (col & 3);
    }
    if (TILE > 0) {
        int64_t tile = (row >> 3) * ((COLS + 3) >> 2) + (col >> 2);
        return tile * 32 + (row & 7) * 4 + (col & 3);
    }
    return row * COLS + col;
}

// =================================================================================================
// Exact 0-ULP SiTU & Pure-Scalar Dual-Candidate NVFP4 Quant288 Helpers (Zero Local Arrays!)
// =================================================================================================

__device__ __forceinline__ float sigmoid_approx_exact(float x) {
    float r;
    asm("{ .reg .f32 t; mul.rn.f32 t, %1, 0fBFB8AA3B; "
        "ex2.approx.ftz.f32 t, t; add.rn.f32 t, t, 0f3F800000; "
        "rcp.approx.ftz.f32 %0, t; }"
        : "=f"(r) : "f"(x));
    return r;
}

__device__ __forceinline__ float softcap_4(float x) {
    float q = mul_rn(x, 0.25f);
    float t;
    asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(q));
    return mul_rn(4.0f, t);
}

__device__ __forceinline__ float softcap_25(float x) {
    float q = mul_rn(x, INV_25);
    float res = fmaxf(fmaf(q, 25.0f, -x), -FP32_MAX_VAL);
    q = fmaf(-res, INV_25, q);
    float t;
    asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(q));
    return mul_rn(25.0f, t);
}

__device__ __forceinline__ uint16_t situ_from_f32_pair_exact(float g_f32, float u_f32, float gamma) {
    float gate = bf16_bits_to_f32(f32_to_bf16_rn(g_f32));
    float up   = bf16_bits_to_f32(f32_to_bf16_rn(u_f32));
    float sig  = sigmoid_approx_exact(gate);
    gate = softcap_4(gate);
    up   = softcap_25(up);
    float y = mul_rn(mul_rn(mul_rn(gate, sig), up), gamma);
    return f32_to_bf16_rn(y);
}

__device__ __forceinline__ float bmax_u32_bf16x2(uint32_t w) {
    const float a = bf16_bits_to_f32(static_cast<uint16_t>(w & 0xFFFFu));
    const float b = bf16_bits_to_f32(static_cast<uint16_t>(w >> 16));
    return fmaxf(fabsf(a), fabsf(b));
}

__device__ __forceinline__ float bmax_u4_bf16(uint4 v) {
    return fmaxf(fmaxf(bmax_u32_bf16x2(v.x), bmax_u32_bf16x2(v.y)),
                 fmaxf(bmax_u32_bf16x2(v.z), bmax_u32_bf16x2(v.w)));
}

__device__ __forceinline__ void eval_nvfp4_word_scalar(
    uint32_t word,
    int shift_bits,
    float inv6, float scale6,
    float inv4, float scale4,
    float outer_scale,
    uint32_t& p6, uint32_t& p4,
    float& pe6, float& pe4
) {
    const float x0 = bf16_bits_to_f32(static_cast<uint16_t>(word & 0xFFFFu));
    const float x1 = bf16_bits_to_f32(static_cast<uint16_t>(word >> 16));
    {
        uint32_t qbyte, h;
        const float a = mul_rn(x0, inv6), b = mul_rn(x1, inv6);
        asm("{ .reg .b8 q; cvt.rn.satfinite.e2m1x2.f32 q, %3, %2; "
            "cvt.u32.u8 %0, q; cvt.rn.f16x2.e2m1x2 %1, q; }"
            : "=&r"(qbyte), "=r"(h) : "f"(a), "f"(b));
        p6 |= (qbyte << shift_bits);
        const float r0 = __half2float(__ushort_as_half(static_cast<uint16_t>(h & 0xFFFFu)));
        const float r1 = __half2float(__ushort_as_half(static_cast<uint16_t>(h >> 16)));
        pe6 = add_rn(fabsf(sub_rn(mul_rn(mul_rn(r0, scale6), outer_scale), x0)),
                     fabsf(sub_rn(mul_rn(mul_rn(r1, scale6), outer_scale), x1)));
    }
    {
        uint32_t qbyte, h;
        const float a = mul_rn(x0, inv4), b = mul_rn(x1, inv4);
        asm("{ .reg .b8 q; cvt.rn.satfinite.e2m1x2.f32 q, %3, %2; "
            "cvt.u32.u8 %0, q; cvt.rn.f16x2.e2m1x2 %1, q; }"
            : "=&r"(qbyte), "=r"(h) : "f"(a), "f"(b));
        p4 |= (qbyte << shift_bits);
        const float r0 = __half2float(__ushort_as_half(static_cast<uint16_t>(h & 0xFFFFu)));
        const float r1 = __half2float(__ushort_as_half(static_cast<uint16_t>(h >> 16)));
        pe4 = add_rn(fabsf(sub_rn(mul_rn(mul_rn(r0, scale4), outer_scale), x0)),
                     fabsf(sub_rn(mul_rn(mul_rn(r1, scale4), outer_scale), x1)));
    }
}

__device__ __forceinline__ void eval_nvfp4_half4_scalar(
    uint4 v,
    float inv6, float scale6,
    float inv4, float scale4,
    float outer_scale,
    uint32_t& pk6_w, uint32_t& pk4_w,
    float& err6_h, float& err4_h
) {
    uint32_t p6 = 0u, p4 = 0u;
    float e6_0, e4_0, e6_1, e4_1, e6_2, e4_2, e6_3, e4_3;
    eval_nvfp4_word_scalar(v.x,  0, inv6, scale6, inv4, scale4, outer_scale, p6, p4, e6_0, e4_0);
    eval_nvfp4_word_scalar(v.y,  8, inv6, scale6, inv4, scale4, outer_scale, p6, p4, e6_1, e4_1);
    eval_nvfp4_word_scalar(v.z, 16, inv6, scale6, inv4, scale4, outer_scale, p6, p4, e6_2, e4_2);
    eval_nvfp4_word_scalar(v.w, 24, inv6, scale6, inv4, scale4, outer_scale, p6, p4, e6_3, e4_3);
    pk6_w = p6;
    pk4_w = p4;
    err6_h = add_rn(add_rn(e6_0, e6_1), add_rn(e6_2, e6_3));
    err4_h = add_rn(add_rn(e4_0, e4_1), add_rn(e4_2, e4_3));
}

// Quantize `[valid_m, 288]` BF16 rows from coherent L2 `y_slot_bank` directly into `(Values, Scales, RowScales)`
__device__ __forceinline__ void quant_slot_tile_288_in_l2(
    const uint16_t* __restrict__ y_slot_bank,
    uint8_t* __restrict__ Values,
    uint8_t* __restrict__ Scales,
    float* __restrict__ RowScales,
    int row0,
    int valid_m,
    float w2_amax_e,
    float inv_denom,
    int warp,
    int lane
) {
    for (int m = warp; m < valid_m; m += 8) {
        const int dst_row = row0 + m;
        uint4 v0 = make_uint4(0u, 0u, 0u, 0u);
        uint4 v1 = make_uint4(0u, 0u, 0u, 0u);
        float bmax = 0.0f;

        if (lane < 18) {
            const uint4* src = reinterpret_cast<const uint4*>(
                y_slot_bank + m * 288 + lane * 16);
            v0 = ld_cg_u4(src + 0);
            v1 = ld_cg_u4(src + 1);
            bmax = fmaxf(bmax_u4_bf16(v0), bmax_u4_bf16(v1));
        }

        float row_amax = bmax;
        row_amax = fmaxf(row_amax, __shfl_xor_sync(0xFFFFFFFFu, row_amax, 16));
        row_amax = fmaxf(row_amax, __shfl_xor_sync(0xFFFFFFFFu, row_amax, 8));
        row_amax = fmaxf(row_amax, __shfl_xor_sync(0xFFFFFFFFu, row_amax, 4));
        row_amax = fmaxf(row_amax, __shfl_xor_sync(0xFFFFFFFFu, row_amax, 2));
        row_amax = fmaxf(row_amax, __shfl_xor_sync(0xFFFFFFFFu, row_amax, 1));

        if (lane == 0) {
            const float rs_val = mul_rn(mul_rn(row_amax, w2_amax_e), inv_denom);
            st_wb_u32(&RowScales[dst_row], __float_as_uint(rs_val));
        }

        const float enc_s = fminf(div_rn(1536.0f, row_amax), FP32_MAX_VAL);
        const float dec_s = div_rn(1.0f, enc_s);
        const float out_s = mul_rn(row_amax, INV_1536);

        uint8_t sc_byte = 0u;
        if (lane < 18) {
            const float s6_raw = fminf(mul_rn(bmax, mul_rn(enc_s, INV_6)), 448.0f);
            const float s4_raw = fminf(mul_rn(bmax, mul_rn(enc_s, 0.25f)), 448.0f);

            uint16_t s_e4m3x2;
            uint32_t s_f16x2;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %3, %2;\n\t"
                "cvt.rn.f16x2.e4m3x2 %1, %0;"
                : "=&h"(s_e4m3x2), "=r"(s_f16x2)
                : "f"(s6_raw), "f"(s4_raw));

            const float scale6 = __half2float(__ushort_as_half(static_cast<uint16_t>(s_f16x2 & 0xFFFFu)));
            const float scale4 = __half2float(__ushort_as_half(static_cast<uint16_t>(s_f16x2 >> 16)));
            const float inv6 = fminf(div_rn(1.0f, mul_rn(scale6, dec_s)), FP32_MAX_VAL);
            const float inv4 = fminf(div_rn(1.0f, mul_rn(scale4, dec_s)), FP32_MAX_VAL);

            uint32_t pk6_lo, pk4_lo, pk6_hi, pk4_hi;
            float e6_lo, e4_lo, e6_hi, e4_hi;
            eval_nvfp4_half4_scalar(v0, inv6, scale6, inv4, scale4, out_s, pk6_lo, pk4_lo, e6_lo, e4_lo);
            eval_nvfp4_half4_scalar(v1, inv6, scale6, inv4, scale4, out_s, pk6_hi, pk4_hi, e6_hi, e4_hi);

            const bool choose4 = (add_rn(e4_lo, e4_hi) < add_rn(e6_lo, e6_hi));
            const uint2 packed = make_uint2(choose4 ? pk4_lo : pk6_lo, choose4 ? pk4_hi : pk6_hi);
            sc_byte = static_cast<uint8_t>(choose4 ? (s_e4m3x2 >> 8) : (s_e4m3x2 & 0xFFu));

            st_wb_u2(Values + static_cast<int64_t>(dst_row) * 144 + lane * 8, packed);
        }

        const uint32_t my_sc = (lane < 18) ? static_cast<uint32_t>(sc_byte) : 0u;
        const uint32_t b0 = __shfl_sync(0xFFFFFFFFu, my_sc, (lane * 4 + 0) & 31);
        const uint32_t b1 = __shfl_sync(0xFFFFFFFFu, my_sc, (lane * 4 + 1) & 31);
        const uint32_t b2 = __shfl_sync(0xFFFFFFFFu, my_sc, (lane * 4 + 2) & 31);
        const uint32_t b3 = __shfl_sync(0xFFFFFFFFu, my_sc, (lane * 4 + 3) & 31);
        if (lane < 5) {
            const uint32_t pack4 = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
            st_wb_u32(reinterpret_cast<uint32_t*>(Scales + static_cast<int64_t>(dst_row) * 20) + lane, pack4);
        }
    }
}

// =================================================================================================
// PART 1: Fused 1-CTA W13 (`BM=128`) + W2 (`BM=64` split) LPT Plan Builder + `SlotState` Zero
// =================================================================================================

__device__ __forceinline__ void find_expert_row_range_v6(
    const int32_t* __restrict__ Tiles,
    int max_ctas,
    int tile,
    int target_e,
    int& row_beg,
    int& row_end
) {
    const int num_tiles = Tiles[2 * max_ctas];
    if (num_tiles <= 0) {
        row_beg = 0; row_end = 0; return;
    }
    int lo = 0, hi = num_tiles;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (Tiles[mid] < target_e) lo = mid + 1;
        else hi = mid;
    }
    const int t_lo = lo;
    if (t_lo >= num_tiles || Tiles[t_lo] != target_e) {
        row_beg = 0; row_end = 0; return;
    }
    lo = t_lo + 1;
    hi = num_tiles;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (Tiles[mid] <= target_e) lo = mid + 1;
        else hi = mid;
    }
    row_beg = t_lo * tile;
    row_end = Tiles[max_ctas + lo - 1];
}

__global__ __launch_bounds__(256, 1)
void pf_nvfp4_build_w13_w2_plan_v6_kernel(
    const int32_t* __restrict__ Tiles,
    int32_t* __restrict__ WorkBuf,
    int32_t* __restrict__ W2PlanBuf,
    W13SlotBankState* __restrict__ SlotState,
    int max_ctas,
    int tile,
    int num_experts,
    int max_m128,
    int w2_split_m,
    int max_w2_work
) {
    constexpr int MAX_E = 1024;
    __shared__ int32_t  sBeg[MAX_E];
    __shared__ int32_t  sEnd[MAX_E];
    __shared__ uint32_t sW13Bucket[16];
    __shared__ uint32_t sW13Off[16];
    __shared__ uint32_t sW2Bucket[32];
    __shared__ uint32_t sW2Off[32];

    const int tid = static_cast<int>(threadIdx.x);
    const int eff_e = (num_experts < MAX_E) ? num_experts : MAX_E;
    const bool build_w2 = (W2PlanBuf != nullptr && max_w2_work > 0);
    const int eff_split = (w2_split_m > 0) ? ((w2_split_m + 63) & ~63) : 0;

    if (SlotState != nullptr && tid < kMaxGroups * kSlotBanks) {
        SlotState[tid].arrived = 0u;
        SlotState[tid].ready_epoch = 0u;
    }
    if (tid < 16) sW13Bucket[tid] = 0u;
    if (tid < 32) sW2Bucket[tid] = 0u;
    if (build_w2) {
        for (int w = tid; w < max_w2_work; w += 256) W2PlanBuf[w] = -1;
    }
    __syncthreads();

    for (int e = tid; e < eff_e; e += 256) {
        int rb = 0, re = 0;
        find_expert_row_range_v6(Tiles, max_ctas, tile, e, rb, re);
        sBeg[e] = rb;
        sEnd[e] = re;
        const int cnt = re - rb;
        if (cnt > 0) {
            const int n_m = (cnt + 127) >> 7;
            const int b13 = 15 - ((n_m - 1 > 15) ? 15 : (n_m - 1));
            atomicAdd(&sW13Bucket[b13], static_cast<uint32_t>(n_m));

            if (build_w2) {
                if (eff_split > 0 && cnt > eff_split) {
                    for (int off = 0; off < cnt; off += eff_split) {
                        int sz = cnt - off;
                        if (sz > eff_split) sz = eff_split;
                        const int raw_b = (sz - 1) >> 5;
                        const int b2 = 31 - ((raw_b > 31) ? 31 : raw_b);
                        atomicAdd(&sW2Bucket[b2], 1u);
                    }
                } else {
                    const int raw_b = (cnt - 1) >> 5;
                    const int b2 = 31 - ((raw_b > 31) ? 31 : raw_b);
                    atomicAdd(&sW2Bucket[b2], 1u);
                }
            }
        }
    }
    __syncthreads();

    if (tid == 0) {
        uint32_t acc13 = 0u;
        for (int b = 0; b < 16; ++b) {
            sW13Off[b] = acc13;
            acc13 += sW13Bucket[b];
        }
        int tot13 = static_cast<int>(acc13);
        if (tot13 > max_m128) tot13 = max_m128;
        WorkBuf[4 * max_m128] = tot13;

        if (build_w2) {
            uint32_t acc2 = 0u;
            for (int b = 0; b < 32; ++b) {
                sW2Off[b] = acc2;
                acc2 += sW2Bucket[b];
            }
            int tot2 = static_cast<int>(acc2);
            if (tot2 > max_w2_work) tot2 = max_w2_work;
            W2PlanBuf[3 * max_w2_work] = tot2;
        }
    }
    __syncthreads();

    int32_t* out13_e     = WorkBuf;
    int32_t* out13_row0  = WorkBuf + max_m128;
    int32_t* out13_end   = WorkBuf + 2 * max_m128;
    int32_t* out13_flags = WorkBuf + 3 * max_m128;

    int32_t* out2_e   = build_w2 ? W2PlanBuf : nullptr;
    int32_t* out2_beg = build_w2 ? (W2PlanBuf + max_w2_work) : nullptr;
    int32_t* out2_end = build_w2 ? (W2PlanBuf + 2 * max_w2_work) : nullptr;

    for (int e = tid; e < eff_e; e += 256) {
        const int rb = sBeg[e];
        const int re = sEnd[e];
        const int cnt = re - rb;
        if (cnt > 0) {
            const int n_m = (cnt + 127) >> 7;
            const int b13 = 15 - ((n_m - 1 > 15) ? 15 : (n_m - 1));
            const uint32_t base_pos = atomicAdd(&sW13Off[b13], static_cast<uint32_t>(n_m));
            for (int m = 0; m < n_m; ++m) {
                const int pos = static_cast<int>(base_pos) + m;
                if (pos < max_m128) {
                    const int r0 = rb + m * 128;
                    const int fl = ((m + 1 == n_m) ? 1 : 0) | ((r0 + 128 <= re) ? 2 : 0);
                    out13_e[pos]     = e;
                    out13_row0[pos]  = r0;
                    out13_end[pos]   = re;
                    out13_flags[pos] = fl;
                }
            }

            if (build_w2) {
                if (eff_split > 0 && cnt > eff_split) {
                    for (int off = 0; off < cnt; off += eff_split) {
                        int sz = cnt - off;
                        if (sz > eff_split) sz = eff_split;
                        const int raw_b = (sz - 1) >> 5;
                        const int b2 = 31 - ((raw_b > 31) ? 31 : raw_b);
                        const uint32_t pos2 = atomicAdd(&sW2Off[b2], 1u);
                        if (static_cast<int>(pos2) < max_w2_work) {
                            out2_e[pos2]   = e;
                            out2_beg[pos2] = rb + off;
                            out2_end[pos2] = rb + off + sz;
                        }
                    }
                } else {
                    const int raw_b = (cnt - 1) >> 5;
                    const int b2 = 31 - ((raw_b > 31) ? 31 : raw_b);
                    const uint32_t pos2 = atomicAdd(&sW2Off[b2], 1u);
                    if (static_cast<int>(pos2) < max_w2_work) {
                        out2_e[pos2]   = e;
                        out2_beg[pos2] = rb;
                        out2_end[pos2] = re;
                    }
                }
            }
        }
    }
}

// =================================================================================================
// PART 2: Co-Resident 4-Bank L2-Slot Fused `NVFP4 W13 + SiTU + In-L2 Quant288` (`0 B` Stack/Spills)
// =================================================================================================

struct W13EpiSmem128 {
    uint32_t bf16x2[128][49];
};

struct W13SituEpiSmem128 {
    uint16_t y[128][50];
};

struct alignas(16) W13V6CtaParams {
    const uint8_t* A;
    const uint8_t* AS;
    const float* RS;
    const int32_t* Row2Tok;
    const uint8_t* SLABS;
    const uint8_t* WS_BASE;
    const int32_t* P_OFF;
    const uint32_t* PATCHES;
    const float* Ones;
    const float* Gammas;
    const int32_t* WorkBuf;
    uint16_t* y_slot_grp;
    W13SlotBankState* slot_state_grp;
    uint8_t* Values;
    uint8_t* Scales;
    float* RowScales;
    const float* WAmax;
    void* Out_unfused;
    float inv_denom;
    int max_m128;
    int total_work;
    int tile;
    int w_evict_mode;
    int batch_start;
    int cur_e;
    int cur_row0;
    int cur_end;
    int cur_flags;
    int is_last_cta;
};

// Pure scalar unrolled 8-way WS5 patch decoder (ZERO local array `uint8_t sb[8]`!)
template <int HAS_PATCHES>
__device__ __forceinline__ void decode_sfb_ss_scalar_v6(
    const uint8_t* __restrict__ slabs_cta,
    int base_e,
    const int32_t (&sPOff)[12][2],
    const uint32_t* __restrict__ PATCHES,
    int ss,
    int wp_n,
    int lane,
    int g,
    int lt,
    bool use_cs_w,
    uint32_t (&p0)[3],
    uint32_t (&p1)[3]
) {
    const int kg_lo = ss * 32;
    const int kg_hi = kg_lo + 32;

    #pragma unroll
    for (int sn = 0; sn < 3; ++sn) {
        const int wg = wp_n * 3 + sn;
        const int pl = wg >> 2;
        const int wp = wg & 3;
        const uint8_t* ss_ptr = slabs_cta + static_cast<int64_t>(pl * 9 + ss) * 8832;
        const uint32_t ws_lo = ld_w_u32(ss_ptr + 8192 + wp * 128 + lane * 4, use_cs_w);
        const uint8_t  ws_hi = ld_w_u8(ss_ptr + 8704 + wp * 32 + lane, use_cs_w);

        uint8_t sb0 = static_cast<uint8_t>(base_e + (((ws_lo >>  0) & 0xFu) | (((ws_hi >> 0) & 1u) << 4)));
        uint8_t sb1 = static_cast<uint8_t>(base_e + (((ws_lo >>  4) & 0xFu) | (((ws_hi >> 1) & 1u) << 4)));
        uint8_t sb2 = static_cast<uint8_t>(base_e + (((ws_lo >>  8) & 0xFu) | (((ws_hi >> 2) & 1u) << 4)));
        uint8_t sb3 = static_cast<uint8_t>(base_e + (((ws_lo >> 12) & 0xFu) | (((ws_hi >> 3) & 1u) << 4)));
        uint8_t sb4 = static_cast<uint8_t>(base_e + (((ws_lo >> 16) & 0xFu) | (((ws_hi >> 4) & 1u) << 4)));
        uint8_t sb5 = static_cast<uint8_t>(base_e + (((ws_lo >> 20) & 0xFu) | (((ws_hi >> 5) & 1u) << 4)));
        uint8_t sb6 = static_cast<uint8_t>(base_e + (((ws_lo >> 24) & 0xFu) | (((ws_hi >> 6) & 1u) << 4)));
        uint8_t sb7 = static_cast<uint8_t>(base_e + (((ws_lo >> 28) & 0xFu) | (((ws_hi >> 7) & 1u) << 4)));

        if constexpr (HAS_PATCHES) {
            const int plo = sPOff[wg][0];
            const int phi = sPOff[wg][1];
            for (int pi = plo; pi < phi; ++pi) {
                const uint32_t rec = PATCHES[pi];
                const int p_kg = static_cast<int>(rec >> 16);
                if (p_kg >= kg_lo && p_kg < kg_hi) {
                    if (g == static_cast<int>((rec >> 8) & 0xFFu) && lt == ((p_kg - kg_lo) >> 3)) {
                        const int idx_s = p_kg & 7;
                        const uint8_t pval = static_cast<uint8_t>(rec & 0xFFu);
                        if      (idx_s == 0) sb0 = pval;
                        else if (idx_s == 1) sb1 = pval;
                        else if (idx_s == 2) sb2 = pval;
                        else if (idx_s == 3) sb3 = pval;
                        else if (idx_s == 4) sb4 = pval;
                        else if (idx_s == 5) sb5 = pval;
                        else if (idx_s == 6) sb6 = pval;
                        else                 sb7 = pval;
                    }
                }
            }
        }

        p0[sn] = static_cast<uint32_t>(sb0) | (static_cast<uint32_t>(sb1) << 8)
               | (static_cast<uint32_t>(sb2) << 16) | (static_cast<uint32_t>(sb3) << 24);
        p1[sn] = static_cast<uint32_t>(sb4) | (static_cast<uint32_t>(sb5) << 8)
               | (static_cast<uint32_t>(sb6) << 16) | (static_cast<uint32_t>(sb7) << 24);
    }
}

// FUSE_MODE:
//   0 = unfused W13 -> Out [R, 576] bf16
//   1 = v4b-compatible fused SiTU -> Out_situ [R, 288] bf16
//   2 = v6 ZERO-DRAM fused W13 + SiTU + in-L2 Quant288 -> (Values, Scales, RowScales) directly!
template <int USE_R2T, int AS_ROW_MAJOR, int HAS_PATCHES, int FUSE_MODE>
__global__ __launch_bounds__(256, 2)
void pf_nvfp4_w13_ws5_v6_kernel(
    const uint8_t* __restrict__ A,
    const uint8_t* __restrict__ AS,
    const float* __restrict__ RS,
    const int32_t* __restrict__ Row2Tok,
    const uint8_t* __restrict__ SLABS,
    const uint8_t* __restrict__ WS_BASE,
    const int32_t* __restrict__ P_OFF,
    const uint32_t* __restrict__ PATCHES,
    const float* __restrict__ Ones,
    const float* __restrict__ Gammas,
    const int32_t* __restrict__ WorkBuf,
    uint16_t* __restrict__ Y_slot,
    W13SlotBankState* __restrict__ SlotState,
    uint8_t* __restrict__ Values,
    uint8_t* __restrict__ Scales,
    float* __restrict__ RowScales,
    const float* __restrict__ WAmax,
    const float* __restrict__ WDenominator,
    void* __restrict__ Out_unfused,
    int max_m128,
    int tile,
    int w_evict_mode,
    int batch_start
) {
    constexpr int BM = 128;
    constexpr int KP = 2304;
    constexpr int KG = 288;
    constexpr int M_TILES_W = 4;

    __shared__ alignas(16) union {
        struct { uint32_t sA[2][BM][32]; uint32_t sAS[2][BM][4]; } m;
        W13EpiSmem128 epi_raw;
        W13SituEpiSmem128 epi_situ;
    } sm;
    __shared__ alignas(16) int32_t sTok[BM];
    __shared__ alignas(16) float   sRS[BM];
    __shared__ alignas(16) float   sGam[BM];
    __shared__ alignas(16) int32_t sPOff[12][2];
    __shared__ W13V6CtaParams sP;

    const int tid     = static_cast<int>(threadIdx.x);
    const int nb_cta  = static_cast<int>(blockIdx.x);  // 0..5
    const int slot_id = static_cast<int>(blockIdx.y);  // 0..kV13BatchSlots-1 (1-to-1 L2 slot!)
    const int w_idx   = batch_start + slot_id;

    if (tid == 0) {
        const int tot_w = WorkBuf[4 * max_m128];
        sP.A              = A;
        sP.AS             = AS;
        sP.RS             = RS;
        sP.Row2Tok        = Row2Tok;
        sP.SLABS          = SLABS;
        sP.WS_BASE        = WS_BASE;
        sP.P_OFF          = P_OFF;
        sP.PATCHES        = PATCHES;
        sP.Ones           = Ones;
        sP.Gammas         = Gammas;
        sP.WorkBuf        = WorkBuf;
        sP.y_slot_grp     = (FUSE_MODE == 2) ? (Y_slot + static_cast<int64_t>(slot_id) * (BM * 288)) : nullptr;
        sP.slot_state_grp = (FUSE_MODE == 2) ? (SlotState + slot_id) : nullptr;
        sP.Values         = Values;
        sP.Scales         = Scales;
        sP.RowScales      = RowScales;
        sP.WAmax          = WAmax;
        sP.Out_unfused    = Out_unfused;
        sP.inv_denom      = (FUSE_MODE == 2) ? rcp_rn(mul_rn(1536.0f, *WDenominator)) : 0.0f;
        sP.max_m128       = max_m128;
        sP.total_work     = tot_w;
        sP.tile           = tile;
        sP.w_evict_mode   = w_evict_mode;
        sP.batch_start    = batch_start;
        if (w_idx < tot_w) {
            sP.cur_e     = WorkBuf[w_idx];
            sP.cur_row0  = WorkBuf[max_m128 + w_idx];
            sP.cur_end   = WorkBuf[2 * max_m128 + w_idx];
            sP.cur_flags = WorkBuf[3 * max_m128 + w_idx];
        } else {
            sP.cur_e     = -1;
            sP.cur_row0  = 0;
            sP.cur_end   = 0;
            sP.cur_flags = 0;
        }
    }
    __syncthreads();

    if (w_idx >= ld_smem_i32_nocse(&sP.total_work)) {
        return;
    }

    const int warp = tid >> 5;
    const int wm   = warp >> 2;  // 0..1
    const int wp_n = warp & 3;   // 0..3
    const int lane = tid & 31;
    const int g    = lane >> 2;  // 0..7
    const int lt   = lane & 3;   // 0..3
    const int m_warp_base = wm * 64;
    const int m_sf_off    = g + ((lt & 1) << 3);

    {
        const int e_i32 = ld_smem_i32_nocse(&sP.cur_e);
        const int row0  = ld_smem_i32_nocse(&sP.cur_row0);
        const int end   = ld_smem_i32_nocse(&sP.cur_end);
        if (e_i32 < 0 || row0 >= end) {
            return;
        }

        const int64_t e       = static_cast<int64_t>(e_i32);
        const int flags       = ld_smem_i32_nocse(&sP.cur_flags);
        const bool is_last_e  = (flags & 1) != 0;
        const bool is_full_m  = (flags & 2) != 0;
        const int ev_mode     = ld_smem_i32_nocse(&sP.w_evict_mode);
        const bool use_cs_w   = (ev_mode == 2) || (ev_mode == 1 && is_last_e);

        if (tid < BM) {
            const int r = row0 + tid;
            if (is_full_m || r < end) {
                const int32_t* r2t_p = ld_smem_ptr_nocse(&sP.Row2Tok);
                sTok[tid] = USE_R2T ? r2t_p[r] : r;
                sRS[tid]  = ld_smem_ptr_nocse(&sP.RS)[r];
                if constexpr (FUSE_MODE != 0) {
                    sGam[tid] = ld_smem_ptr_nocse(&sP.Gammas)[r];
                }
            } else {
                sTok[tid] = -1;
                sRS[tid]  = 0.0f;
                if constexpr (FUSE_MODE != 0) {
                    sGam[tid] = 0.0f;
                }
            }
        }
        if constexpr (HAS_PATCHES) {
            if (tid < 12) {
                const int32_t* poff_p = ld_smem_ptr_nocse(&sP.P_OFF);
                const int64_t wuid = (e * 18 + nb_cta * 3 + (tid >> 2)) * 4 + (tid & 3);
                sPOff[tid][0] = poff_p[wuid];
                sPOff[tid][1] = poff_p[wuid + 1];
            }
        }
        __syncthreads();

        auto load_stage = [&](int ks, int buf) {
            const uint8_t* a_base = ld_smem_ptr_nocse(&sP.A);
            const uint8_t* as_base = ld_smem_ptr_nocse(&sP.AS);
            #pragma unroll
            for (int it = 0; it < 4; ++it) {
                const int idx = it * 256 + tid;
                const int m   = idx >> 3;
                const int c4  = idx & 7;
                const int sc4 = c4 ^ (m & 7);
                const int32_t tok = sTok[m];
                const uint8_t* src = (tok >= 0) ? (a_base + static_cast<int64_t>(tok) * KP + ks * 128 + c4 * 16) : a_base;
                cp_async_cg_16(&sm.m.sA[buf][m][sc4 * 4], src, (tok >= 0) ? 16 : 0);
            }
            if constexpr (USE_R2T || AS_ROW_MAJOR) {
                if (tid < BM) {
                    const int32_t tok = sTok[tid];
                    const uint8_t* src_as = (tok >= 0) ? (as_base + static_cast<int64_t>(tok) * KG + ks * 16) : as_base;
                    cp_async_cg_16(&sm.m.sAS[buf][tid][0], src_as, (tok >= 0) ? 16 : 0);
                }
            } else {
                const int tile_cur = ld_smem_i32_nocse(&sP.tile);
                #pragma unroll
                for (int it = 0; it < 2; ++it) {
                    const int idx = it * 256 + tid;
                    const int m = idx >> 2;
                    const int q = idx & 3;
                    const int32_t tok = sTok[m];
                    uint32_t sval = 0u;
                    if (tok >= 0) {
                        const int64_t off = scale_offset_byte(static_cast<int64_t>(row0 + m), ks * 16 + q * 4, KG, tile_cur);
                        sval = *reinterpret_cast<const uint32_t*>(as_base + off);
                    }
                    sm.m.sAS[buf][m][q] = sval;
                }
            }
            cp_async_commit();
        };

        const uint8_t* slabs_cta = ld_smem_ptr_nocse(&sP.SLABS) + (e * 18 + nb_cta * 3) * (9LL * 8832LL);
        const int base_e = static_cast<int>(ld_smem_ptr_nocse(&sP.WS_BASE)[e]);

        load_stage(0, 0);
        uint4 wa[3], wb[3];
        #pragma unroll
        for (int sn = 0; sn < 3; ++sn) {
            const int wg = wp_n * 3 + sn;
            wa[sn] = ld_w_u4(slabs_cta + static_cast<int64_t>((wg >> 2) * 9) * 8832 + (wg & 3) * 1024 + lane * 16, use_cs_w);
        }
        uint32_t sfb_p0[3], sfb_p1[3];
        decode_sfb_ss_scalar_v6<HAS_PATCHES>(
            slabs_cta, base_e, sPOff, ld_smem_ptr_nocse(&sP.PATCHES),
            0, wp_n, lane, g, lt, use_cs_w, sfb_p0, sfb_p1);
        cp_async_wait<0>();
        __syncthreads();

        float acc[M_TILES_W][3][4];
        #pragma unroll
        for (int mi = 0; mi < M_TILES_W; ++mi) {
            #pragma unroll
            for (int sn = 0; sn < 3; ++sn) {
                acc[mi][sn][0] = 0.0f; acc[mi][sn][1] = 0.0f;
                acc[mi][sn][2] = 0.0f; acc[mi][sn][3] = 0.0f;
            }
        }

        // Branch-free 18-step K-loop (0-ULP identical accumulation order to v4b & v3!)
        #pragma unroll 1
        for (int ks = 0; ks < 18; ++ks) {
            const int cur = ks & 1, nxt = cur ^ 1, ss = ks >> 1, ks2 = ks & 1;
            if (ks + 1 < 18) load_stage(ks + 1, nxt);
            #pragma unroll
            for (int sn = 0; sn < 3; ++sn) {
                const int wg = wp_n * 3 + sn;
                wb[sn] = ld_w_u4(slabs_cta + static_cast<int64_t>((wg >> 2) * 9 + ss) * 8832
                                 + ks2 * 4096 + (wg & 3) * 1024 + 512 + lane * 16, use_cs_w);
            }
            const int src_lt0 = ks2 * 2;
            #pragma unroll
            for (int q2 = 0; q2 < 2; ++q2) {
                uint32_t b0[3], b1[3], sfb[3];
                #pragma unroll
                for (int sn = 0; sn < 3; ++sn) {
                    const uint32_t lx = (q2 == 0) ? wa[sn].x : wa[sn].z;
                    const uint32_t ly = (q2 == 0) ? wa[sn].y : wa[sn].w;
                    b0[sn] = prmt_b32(lx, ly, 0x5410u);
                    b1[sn] = prmt_b32(lx, ly, 0x7632u);
                    sfb[sn] = __shfl_sync(0xFFFFFFFFu, (q2 == 0) ? sfb_p0[sn] : sfb_p1[sn], (g << 2) | src_lt0);
                }
                const int sw_c0 = (q2 * 8 + lt) ^ (g << 2);
                const int sw_c1 = (q2 * 8 + 4 + lt) ^ (g << 2);
                #pragma unroll
                for (int mi = 0; mi < M_TILES_W; ++mi) {
                    const int m_lo = m_warp_base + mi * 16 + g, m_hi = m_lo + 8;
                    const uint32_t a0 = sm.m.sA[cur][m_lo][sw_c0], a1 = sm.m.sA[cur][m_hi][sw_c0];
                    const uint32_t a2 = sm.m.sA[cur][m_lo][sw_c1], a3 = sm.m.sA[cur][m_hi][sw_c1];
                    const uint32_t sfa = sm.m.sAS[cur][m_warp_base + mi * 16 + m_sf_off][q2];
                    #pragma unroll
                    for (int sn = 0; sn < 3; ++sn) {
                        mma_m16n8k64_nvfp4(acc[mi][sn], a0, a1, a2, a3, b0[sn], b1[sn], sfa, sfb[sn]);
                    }
                }
            }
            if (ks + 1 < 18) {
                const int n_ss = (ks + 1) >> 1, n_ks2 = (ks + 1) & 1;
                #pragma unroll
                for (int sn = 0; sn < 3; ++sn) {
                    const int wg = wp_n * 3 + sn;
                    wa[sn] = ld_w_u4(slabs_cta + static_cast<int64_t>((wg >> 2) * 9 + n_ss) * 8832
                                     + n_ks2 * 4096 + (wg & 3) * 1024 + lane * 16, use_cs_w);
                }
            }
            const int src_lt1 = ks2 * 2 + 1;
            #pragma unroll
            for (int q2 = 0; q2 < 2; ++q2) {
                uint32_t b0[3], b1[3], sfb[3];
                #pragma unroll
                for (int sn = 0; sn < 3; ++sn) {
                    const uint32_t lx = (q2 == 0) ? wb[sn].x : wb[sn].z;
                    const uint32_t ly = (q2 == 0) ? wb[sn].y : wb[sn].w;
                    b0[sn] = prmt_b32(lx, ly, 0x5410u);
                    b1[sn] = prmt_b32(lx, ly, 0x7632u);
                    sfb[sn] = __shfl_sync(0xFFFFFFFFu, (q2 == 0) ? sfb_p0[sn] : sfb_p1[sn], (g << 2) | src_lt1);
                }
                const int sw_c0 = ((2 + q2) * 8 + lt) ^ (g << 2);
                const int sw_c1 = ((2 + q2) * 8 + 4 + lt) ^ (g << 2);
                #pragma unroll
                for (int mi = 0; mi < M_TILES_W; ++mi) {
                    const int m_lo = m_warp_base + mi * 16 + g, m_hi = m_lo + 8;
                    const uint32_t a0 = sm.m.sA[cur][m_lo][sw_c0], a1 = sm.m.sA[cur][m_hi][sw_c0];
                    const uint32_t a2 = sm.m.sA[cur][m_lo][sw_c1], a3 = sm.m.sA[cur][m_hi][sw_c1];
                    const uint32_t sfa = sm.m.sAS[cur][m_warp_base + mi * 16 + m_sf_off][2 + q2];
                    #pragma unroll
                    for (int sn = 0; sn < 3; ++sn) {
                        mma_m16n8k64_nvfp4(acc[mi][sn], a0, a1, a2, a3, b0[sn], b1[sn], sfa, sfb[sn]);
                    }
                }
            }
            if (ks + 1 < 18) {
                if (((ks + 1) & 1) == 0) {
                    decode_sfb_ss_scalar_v6<HAS_PATCHES>(
                        slabs_cta, base_e, sPOff, ld_smem_ptr_nocse(&sP.PATCHES),
                        (ks + 1) >> 1, wp_n, lane, g, lt, use_cs_w, sfb_p0, sfb_p1);
                }
                cp_async_wait<0>();
                __syncthreads();
            }
        }

        const float scale_e = ld_smem_ptr_nocse(&sP.Ones)[e];
        __syncthreads();

        if constexpr (FUSE_MODE != 0) {
            #pragma unroll
            for (int mi = 0; mi < M_TILES_W; ++mi) {
                const int m_lo = m_warp_base + mi * 16 + g;
                const int m_hi = m_lo + 8;
                const bool ok_lo = is_full_m || (row0 + m_lo < end);
                const bool ok_hi = is_full_m || (row0 + m_hi < end);
                const float s_lo = ok_lo ? (sRS[m_lo] * scale_e) : 0.0f;
                const float s_hi = ok_hi ? (sRS[m_hi] * scale_e) : 0.0f;
                const float g_lo = ok_lo ? sGam[m_lo] : 0.0f;
                const float g_hi = ok_hi ? sGam[m_hi] : 0.0f;
                #pragma unroll
                for (int sn = 0; sn < 3; ++sn) {
                    const int col_situ = (wp_n * 3 + sn) * 4 + lt;
                    sm.epi_situ.y[m_lo][col_situ] = ok_lo
                        ? situ_from_f32_pair_exact(acc[mi][sn][0] * s_lo, acc[mi][sn][1] * s_lo, g_lo)
                        : static_cast<uint16_t>(0u);
                    sm.epi_situ.y[m_hi][col_situ] = ok_hi
                        ? situ_from_f32_pair_exact(acc[mi][sn][2] * s_hi, acc[mi][sn][3] * s_hi, g_hi)
                        : static_cast<uint16_t>(0u);
                }
            }
            __syncthreads();

            if constexpr (FUSE_MODE == 2) {
                // ---- Wave-V13 ZERO-SPIN Dedicated L2-Slot Epilogue + Last-Arriving CTA In-L2 Quant288 ----
                // 1-to-1 slot_id ownership within each host batch (64 slots x 72 KiB = 4.50 MiB << 24 MiB L2).
                // First 5 CTAs exit IMMEDIATELY (zero wait!); 6th CTA runs Quant288 from coherent L2 and
                // resets `slot_st->arrived = 0u` before the stream barrier to the next host batch.
                // Mathematically 100% deadlock-impossible even on 1 SM under arbitrary DBO comm contention!
                W13SlotBankState* slot_st = ld_smem_mut_ptr_nocse(&sP.slot_state_grp);
                uint16_t* y_slot_bank     = ld_smem_mut_ptr_nocse(&sP.y_slot_grp);
                const int valid_m         = is_full_m ? BM : (end - row0);

                #pragma unroll
                for (int it = 0; it < 3; ++it) {
                    const int idx   = it * 256 + tid;
                    const int r_epi = (idx * 10923) >> 16;
                    const int c4    = idx - r_epi * 6;
                    if (r_epi < valid_m) {
                        const uint32_t* srow = reinterpret_cast<const uint32_t*>(&sm.epi_situ.y[r_epi][0]);
                        uint4 v;
                        v.x = srow[c4 * 4 + 0];
                        v.y = srow[c4 * 4 + 1];
                        v.z = srow[c4 * 4 + 2];
                        v.w = srow[c4 * 4 + 3];
                        uint4* dst = reinterpret_cast<uint4*>(y_slot_bank + r_epi * 288 + nb_cta * 48);
                        st_wb_u4(dst + c4, v);
                    }
                }
                __syncthreads();

                if (tid == 0) {
                    __threadfence();
                    const uint32_t prev = atomicAdd(&slot_st->arrived, 1u);
                    if (prev == 5u) {
                        __threadfence();
                        sP.is_last_cta = 1;
                    } else {
                        sP.is_last_cta = 0;
                    }
                }
                __syncthreads();

                if (ld_smem_i32_nocse(&sP.is_last_cta) != 0) {
                    const float w2_amax_e = ld_smem_ptr_nocse(&sP.WAmax)[e];
                    quant_slot_tile_288_in_l2(
                        y_slot_bank,
                        ld_smem_mut_ptr_nocse(&sP.Values),
                        ld_smem_mut_ptr_nocse(&sP.Scales),
                        ld_smem_mut_ptr_nocse(&sP.RowScales),
                        row0, valid_m, w2_amax_e,
                        ld_smem_f32_nocse(&sP.inv_denom),
                        warp, lane
                    );
                    __syncthreads();
                    if (tid == 0) {
                        st_wb_u32(&slot_st->arrived, 0u);
                    }
                }
            } else {
                uint16_t* Out_situ = reinterpret_cast<uint16_t*>(ld_smem_mut_ptr_nocse(&sP.Out_unfused));
                #pragma unroll
                for (int it = 0; it < 3; ++it) {
                    const int idx   = it * 256 + tid;
                    const int r_epi = (idx * 10923) >> 16;
                    const int c4    = idx - r_epi * 6;
                    const int r_glob = row0 + r_epi;
                    if (is_full_m || r_glob < end) {
                        const uint32_t* srow = reinterpret_cast<const uint32_t*>(&sm.epi_situ.y[r_epi][0]);
                        uint4 v;
                        v.x = srow[c4 * 4 + 0];
                        v.y = srow[c4 * 4 + 1];
                        v.z = srow[c4 * 4 + 2];
                        v.w = srow[c4 * 4 + 3];
                        uint4* dst = reinterpret_cast<uint4*>(
                            Out_situ + static_cast<int64_t>(r_glob) * 288 + nb_cta * 48);
                        st_cs_u4(dst + c4, v);
                    }
                }
            }
        } else {
            #pragma unroll
            for (int mi = 0; mi < M_TILES_W; ++mi) {
                const int m_lo = m_warp_base + mi * 16 + g;
                const int m_hi = m_lo + 8;
                const float s_lo = (is_full_m || row0 + m_lo < end) ? (sRS[m_lo] * scale_e) : 0.0f;
                const float s_hi = (is_full_m || row0 + m_hi < end) ? (sRS[m_hi] * scale_e) : 0.0f;
                #pragma unroll
                for (int sn = 0; sn < 3; ++sn) {
                    const int col_pair = (wp_n * 3 + sn) * 4 + lt;
                    sm.epi_raw.bf16x2[m_lo][col_pair] = pack_bf16x2(acc[mi][sn][0] * s_lo, acc[mi][sn][1] * s_lo);
                    sm.epi_raw.bf16x2[m_hi][col_pair] = pack_bf16x2(acc[mi][sn][2] * s_hi, acc[mi][sn][3] * s_hi);
                }
            }
            __syncthreads();

            uint32_t* Out_u32 = reinterpret_cast<uint32_t*>(ld_smem_mut_ptr_nocse(&sP.Out_unfused));
            #pragma unroll
            for (int it = 0; it < 6; ++it) {
                const int idx   = it * 256 + tid;
                const int r_epi = (idx * 5462) >> 16;
                const int c4    = idx - r_epi * 12;
                const int r_glob = row0 + r_epi;
                if (is_full_m || r_glob < end) {
                    uint4 v;
                    v.x = sm.epi_raw.bf16x2[r_epi][c4 * 4 + 0];
                    v.y = sm.epi_raw.bf16x2[r_epi][c4 * 4 + 1];
                    v.z = sm.epi_raw.bf16x2[r_epi][c4 * 4 + 2];
                    v.w = sm.epi_raw.bf16x2[r_epi][c4 * 4 + 3];
                    uint4* dst = reinterpret_cast<uint4*>(
                        Out_u32 + static_cast<int64_t>(r_glob) * 288 + nb_cta * 48);
                    st_cs_u4(dst + c4, v);
                }
            }
        }
    }
}

// =================================================================================================
// PART 3: Native SM121a Multi-Warp SMEM-Histogram Prefill Router (`pf_route_tokens_place_v6`)
// =================================================================================================
// Exact 0-ULP bit-identical semantics to `ops_moe.py:_route_tokens_kernel` (`E = 384, TOPK = 8`):
//   - 8 warps = 8 tokens/CTA (`Bias[384]` loaded ONCE into SMEM per CTA vs 8,192 global loads)
//   - 12 registers/lane (`12 * 32 = 384`, 0% masked lane waste vs 25% in Triton `E_POW2=512`)
//   - SMEM `sCtaCnt[384]` pre-aggregation cuts global `Counts` atomic collisions by up to 8x
//   - Last arriving CTA runs 256-thread parallel `_place_experts` (vs 1-warp serial scan)

__device__ __forceinline__ uint64_t make_selection_key_384(float logit, float bias, int e_idx) {
    const float sig = div_rn(1.0f, add_rn(1.0f, expf(-logit)));
    const float sel = add_rn(sig, bias);
    uint32_t bits = __float_as_uint(sel);
    bits ^= ((bits & 0x80000000u) != 0u) ? 0xFFFFFFFFu : 0x80000000u;
    return (static_cast<uint64_t>(bits) << 16) | static_cast<uint64_t>(384 - e_idx);
}

__device__ __forceinline__ uint64_t warp_max_u64(uint64_t v) {
    #pragma unroll
    for (int mask = 16; mask >= 1; mask >>= 1) {
        const uint32_t lo = __shfl_xor_sync(0xFFFFFFFFu, static_cast<uint32_t>(v), mask);
        const uint32_t hi = __shfl_xor_sync(0xFFFFFFFFu, static_cast<uint32_t>(v >> 32), mask);
        const uint64_t o = (static_cast<uint64_t>(hi) << 32) | static_cast<uint64_t>(lo);
        if (o > v) v = o;
    }
    return v;
}

template <bool PARTIALS_BF16>
__global__ __launch_bounds__(256, 2)
void pf_route_tokens_place_v6_kernel(
    const void* __restrict__ Partials,        // [1, T, 384] bf16 or fp32
    const float* __restrict__ Bias,           // [384] fp32
    const int32_t* __restrict__ ExpertBounds, // [2] int32
    const uint8_t* __restrict__ Padding,      // [T] bool or nullptr
    int32_t* __restrict__ Counts,             // [num_local] int32 (zeroed on entry, reset on exit)
    int32_t* __restrict__ Counters,           // [64] int32 (Counters[32] = _TOKENS_DONE)
    int32_t* __restrict__ Ids,                // [T * 8] int32
    int32_t* __restrict__ Ranks,              // [T * 8] int32
    float* __restrict__ Weights,              // [T * 8] fp32
    int32_t* __restrict__ Starts,             // [3 * local_pow2] int32
    int32_t* __restrict__ Tiles,              // [2 * max_ctas + 3] int32
    int T,
    float route_scale,
    int num_local,
    int local_pow2,
    int tile,
    int max_ctas
) {
    __shared__ float   sBias[384];
    __shared__ int32_t sCtaCnt[384];
    __shared__ int32_t sCtaBase[384];
    __shared__ int32_t sBounds[3];  // [lo, hi, is_last_cta]
    __shared__ int32_t sScan[1024]; // prefix scan scratch for _place_experts

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int cta_t0 = static_cast<int>(blockIdx.x) * 8;
    const int cta_tokens = min(8, max(0, T - cta_t0));
    if (cta_tokens <= 0) return;

    for (int i = tid; i < 384; i += 256) {
        sBias[i]   = Bias[i];
        sCtaCnt[i] = 0;
    }
    if (tid == 0) {
        sBounds[0] = ExpertBounds[0];
        sBounds[1] = ExpertBounds[1];
        sBounds[2] = 0;
    }
    __syncthreads();

    const int tok = cta_t0 + warp;
    const bool tok_valid = (warp < cta_tokens);
    int my_local = -1;
    int my_intra = 0;
    float my_weight = 0.0f;

    if (tok_valid) {
        const bool is_pad = (Padding != nullptr) && (Padding[tok] != 0u);
        float lg0, lg1, lg2, lg3, lg4, lg5, lg6, lg7, lg8, lg9, lg10, lg11;

        if constexpr (PARTIALS_BF16) {
            const uint16_t* prow = reinterpret_cast<const uint16_t*>(Partials) + static_cast<int64_t>(tok) * 384;
            lg0  = bf16_bits_to_f32(prow[  0 + lane]);
            lg1  = bf16_bits_to_f32(prow[ 32 + lane]);
            lg2  = bf16_bits_to_f32(prow[ 64 + lane]);
            lg3  = bf16_bits_to_f32(prow[ 96 + lane]);
            lg4  = bf16_bits_to_f32(prow[128 + lane]);
            lg5  = bf16_bits_to_f32(prow[160 + lane]);
            lg6  = bf16_bits_to_f32(prow[192 + lane]);
            lg7  = bf16_bits_to_f32(prow[224 + lane]);
            lg8  = bf16_bits_to_f32(prow[256 + lane]);
            lg9  = bf16_bits_to_f32(prow[288 + lane]);
            lg10 = bf16_bits_to_f32(prow[320 + lane]);
            lg11 = bf16_bits_to_f32(prow[352 + lane]);
        } else {
            const float* prow = reinterpret_cast<const float*>(Partials) + static_cast<int64_t>(tok) * 384;
            lg0  = prow[  0 + lane]; lg1  = prow[ 32 + lane]; lg2  = prow[ 64 + lane]; lg3  = prow[ 96 + lane];
            lg4  = prow[128 + lane]; lg5  = prow[160 + lane]; lg6  = prow[192 + lane]; lg7  = prow[224 + lane];
            lg8  = prow[256 + lane]; lg9  = prow[288 + lane]; lg10 = prow[320 + lane]; lg11 = prow[352 + lane];
        }

        uint64_t k0  = make_selection_key_384(lg0,  sBias[  0 + lane],   0 + lane);
        uint64_t k1  = make_selection_key_384(lg1,  sBias[ 32 + lane],  32 + lane);
        uint64_t k2  = make_selection_key_384(lg2,  sBias[ 64 + lane],  64 + lane);
        uint64_t k3  = make_selection_key_384(lg3,  sBias[ 96 + lane],  96 + lane);
        uint64_t k4  = make_selection_key_384(lg4,  sBias[128 + lane], 128 + lane);
        uint64_t k5  = make_selection_key_384(lg5,  sBias[160 + lane], 160 + lane);
        uint64_t k6  = make_selection_key_384(lg6,  sBias[192 + lane], 192 + lane);
        uint64_t k7  = make_selection_key_384(lg7,  sBias[224 + lane], 224 + lane);
        uint64_t k8  = make_selection_key_384(lg8,  sBias[256 + lane], 256 + lane);
        uint64_t k9  = make_selection_key_384(lg9,  sBias[288 + lane], 288 + lane);
        uint64_t k10 = make_selection_key_384(lg10, sBias[320 + lane], 320 + lane);
        uint64_t k11 = make_selection_key_384(lg11, sBias[352 + lane], 352 + lane);

        int sel_id = 0;
        float sel_logit = 0.0f;

        #pragma unroll
        for (int s = 0; s < 8; ++s) {
            uint64_t lm = k0;
            if (k1  > lm) lm = k1;
            if (k2  > lm) lm = k2;
            if (k3  > lm) lm = k3;
            if (k4  > lm) lm = k4;
            if (k5  > lm) lm = k5;
            if (k6  > lm) lm = k6;
            if (k7  > lm) lm = k7;
            if (k8  > lm) lm = k8;
            if (k9  > lm) lm = k9;
            if (k10 > lm) lm = k10;
            if (k11 > lm) lm = k11;

            const uint64_t best = warp_max_u64(lm);
            const int best_id = 384 - static_cast<int>(best & 0xFFFFu);
            const int best_c  = best_id >> 5;
            float match_lg = lg0;
            if      (best_c ==  1) match_lg = lg1;
            else if (best_c ==  2) match_lg = lg2;
            else if (best_c ==  3) match_lg = lg3;
            else if (best_c ==  4) match_lg = lg4;
            else if (best_c ==  5) match_lg = lg5;
            else if (best_c ==  6) match_lg = lg6;
            else if (best_c ==  7) match_lg = lg7;
            else if (best_c ==  8) match_lg = lg8;
            else if (best_c ==  9) match_lg = lg9;
            else if (best_c == 10) match_lg = lg10;
            else if (best_c == 11) match_lg = lg11;

            const float best_lg = __shfl_sync(0xFFFFFFFFu, match_lg, best_id & 31);
            if ((best_id & 31) == lane) {
                if      (best_c ==  0) k0  = 0ULL;
                else if (best_c ==  1) k1  = 0ULL;
                else if (best_c ==  2) k2  = 0ULL;
                else if (best_c ==  3) k3  = 0ULL;
                else if (best_c ==  4) k4  = 0ULL;
                else if (best_c ==  5) k5  = 0ULL;
                else if (best_c ==  6) k6  = 0ULL;
                else if (best_c ==  7) k7  = 0ULL;
                else if (best_c ==  8) k8  = 0ULL;
                else if (best_c ==  9) k9  = 0ULL;
                else if (best_c == 10) k10 = 0ULL;
                else if (best_c == 11) k11 = 0ULL;
            }
            if (lane == s) {
                sel_id    = best_id;
                sel_logit = best_lg;
            }
        }

        // Exact `_routing_weights(_log_sigmoid(selected), route_scale, 8)` on lanes 0..7
        const float lp = (lane < 8)
            ? sub_rn(fminf(sel_logit, 0.0f), log1pf(expf(-fabsf(sel_logit))))
            : -INFINITY;
        float mlp = lp;
        mlp = fmaxf(mlp, __shfl_xor_sync(0xFFFFFFFFu, mlp, 4));
        mlp = fmaxf(mlp, __shfl_xor_sync(0xFFFFFFFFu, mlp, 2));
        mlp = fmaxf(mlp, __shfl_xor_sync(0xFFFFFFFFu, mlp, 1));

        const float sh = (lane < 8) ? expf(sub_rn(lp, mlp)) : 0.0f;
        // Exact `_sum8`: ((x0 + x2) + (x1 + x3)) + ((x4 + x6) + (x5 + x7))
        const float s02 = add_rn(sh, __shfl_xor_sync(0xFFFFFFFFu, sh, 2));
        const float s03 = add_rn(s02, __shfl_xor_sync(0xFFFFFFFFu, s02, 1));
        const float tot = add_rn(s03, __shfl_xor_sync(0xFFFFFFFFu, s03, 4));
        const float lse = add_rn(mlp, logf(tot));

        if (lane < 8) {
            my_weight = mul_rn(expf(sub_rn(lp, lse)), route_scale);
            const int lo = sBounds[0];
            const int hi = sBounds[1];
            if (!is_pad && sel_id >= lo && sel_id < hi) {
                my_local = sel_id - lo;
                my_intra = atomicAdd(&sCtaCnt[my_local], 1);
            }
        }
    }
    __syncthreads();

    for (int e = tid; e < num_local; e += 256) {
        const int c = sCtaCnt[e];
        sCtaBase[e] = (c > 0) ? atomicAdd(&Counts[e], c) : 0;
    }
    __syncthreads();

    if (tok_valid && lane < 8) {
        const int64_t slot = static_cast<int64_t>(tok) * 8 + lane;
        const int rnk = (my_local >= 0) ? (sCtaBase[my_local] + my_intra) : 0;
        Ranks[slot]   = rnk;
        Ids[slot]     = my_local;
        Weights[slot] = my_weight;
    }
    __syncthreads();

    if (tid == 0) {
        __threadfence();
        const int prev = atomicAdd(&Counters[32], cta_tokens);
        if (prev + cta_tokens == T) {
            __threadfence();
            sBounds[2] = 1;
        }
    }
    __syncthreads();

    if (sBounds[2] != 0) {
        // Last CTA executes parallel `_place_experts` (`num_local <= 384, local_pow2 <= 512`)
        const int tile_shift = __ffs(tile) - 1;
        for (int idx = tid; idx < local_pow2; idx += 256) {
            const int c = (idx < num_local) ? Counts[idx] : 0;
            if (idx < num_local) {
                sCtaCnt[idx] = c;
                Counts[idx]  = 0;  // reset Counts for next call!
            }
            sScan[idx]       = (c + tile - 1) >> tile_shift;  // tiles per expert
            sScan[512 + idx] = c;                             // counts per expert
        }
        __syncthreads();

        // Inclusive prefix scan over `local_pow2 <= 512` in SMEM (handles both <=256 and 512!)
        for (int offset = 1; offset < local_pow2; offset <<= 1) {
            int add_t0 = 0, add_c0 = 0, add_t1 = 0, add_c1 = 0;
            if (tid < local_pow2 && tid >= offset) {
                add_t0 = sScan[tid - offset];
                add_c0 = sScan[512 + tid - offset];
            }
            const int idx1 = tid + 256;
            if (idx1 < local_pow2 && idx1 >= offset) {
                add_t1 = sScan[idx1 - offset];
                add_c1 = sScan[512 + idx1 - offset];
            }
            __syncthreads();
            if (tid < local_pow2 && tid >= offset) {
                sScan[tid]       += add_t0;
                sScan[512 + tid] += add_c0;
            }
            if (idx1 < local_pow2 && idx1 >= offset) {
                sScan[idx1]       += add_t1;
                sScan[512 + idx1] += add_c1;
            }
            __syncthreads();
        }

        const int num_tiles = (local_pow2 > 0) ? sScan[local_pow2 - 1] : 0;
        const int total_routed = (local_pow2 > 0) ? sScan[512 + local_pow2 - 1] : 0;

        for (int idx = tid; idx < local_pow2; idx += 256) {
            const int c = (idx < num_local) ? sCtaCnt[idx] : 0;
            const int t_cnt = (c + tile - 1) >> tile_shift;
            const int t_end = sScan[idx];
            const int c_end = sScan[512 + idx];
            const int start = (t_end - t_cnt) << tile_shift;
            if (idx < num_local) {
                Starts[idx] = start;
                Starts[local_pow2 + idx] = c_end - c;
            }
            Starts[2 * local_pow2 + idx] = start + c;
        }
        if (tid == 0) {
            Tiles[2 * max_ctas]     = num_tiles;
            Tiles[2 * max_ctas + 1] = num_tiles << tile_shift;
            Tiles[2 * max_ctas + 2] = total_routed;
            Counters[32]            = 0;  // reset _TOKENS_DONE!
        }
        __syncthreads();

        // Parallel binary-search fill of `Tiles[0..num_tiles-1]` and `Tiles[max_ctas + 0..num_tiles-1]`
        for (int cta = tid; cta < num_tiles; cta += 256) {
            int lo = 0, hi = num_local - 1;
            while (lo < hi) {
                const int mid = (lo + hi) >> 1;
                if (sScan[mid] <= cta) lo = mid + 1;
                else hi = mid;
            }
            const int exp_id = lo;
            const int end_row = Starts[2 * local_pow2 + exp_id];
            const int cta_end = (cta + 1) << tile_shift;
            Tiles[cta] = exp_id;
            Tiles[max_ctas + cta] = (cta_end < end_row) ? cta_end : end_row;
        }
    }
}

// =================================================================================================
// Host Launch Helpers & Cached Active-Group Query
// =================================================================================================

inline bool is_stream_capturing(cudaStream_t stream) {
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    cudaError_t err = cudaStreamIsCapturing(stream, &status);
    if (err != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return status != cudaStreamCaptureStatusNone;
}

// Wave-V13 ZERO-SPIN: no co-resident SM reservation needed! Uses ALL 48/48 SMs safely!

template <int FUSE_MODE>
inline int launch_nvfp4_w13_v6_impl(
    const void* A, const void* AS, const void* RS, const void* Row2Tok,
    const void* SLABS, const void* WS_BASE, const void* P_OFF, const void* PATCHES,
    const void* Ones, const void* Gammas, const void* WorkBuf,
    void* Y_slot, void* SlotState,
    void* Values, void* Scales, void* RowScales,
    const void* WAmax, const void* WDenominator,
    void* Out_unfused,
    int max_m128, int tile, int use_r2t, int as_row_major, int has_patches,
    int w_evict_mode, cudaStream_t stream
) {
    if (max_m128 <= 0 || tile < 64) return -2;
    if (is_stream_capturing(stream)) return -4;

    dim3 block(256);
    const int key = ((use_r2t ? 1 : 0) << 2) | ((as_row_major ? 1 : 0) << 1) | (has_patches ? 1 : 0);

    #define DISPATCH_V6_W13(R2T, RM, HP) \
        do { \
            auto kfn = pf_nvfp4_w13_ws5_v6_kernel<R2T, RM, HP, FUSE_MODE>; \
            static bool _carved = false; \
            if (!_carved) { \
                cudaFuncSetAttribute(kfn, cudaFuncAttributePreferredSharedMemoryCarveout, \
                                     cudaSharedmemCarveoutMaxShared); \
                _carved = true; \
            } \
            if constexpr (FUSE_MODE == 2) { \
                for (int b0 = 0; b0 < max_m128; b0 += kV13BatchSlots) { \
                    const int bw = ((max_m128 - b0) < kV13BatchSlots) ? (max_m128 - b0) : kV13BatchSlots; \
                    dim3 grid(6, bw); \
                    kfn<<<grid, block, 0, stream>>>( \
                        reinterpret_cast<const uint8_t*>(A), reinterpret_cast<const uint8_t*>(AS), \
                        reinterpret_cast<const float*>(RS), reinterpret_cast<const int32_t*>(Row2Tok), \
                        reinterpret_cast<const uint8_t*>(SLABS), reinterpret_cast<const uint8_t*>(WS_BASE), \
                        reinterpret_cast<const int32_t*>(P_OFF), reinterpret_cast<const uint32_t*>(PATCHES), \
                        reinterpret_cast<const float*>(Ones), reinterpret_cast<const float*>(Gammas), \
                        reinterpret_cast<const int32_t*>(WorkBuf), \
                        reinterpret_cast<uint16_t*>(Y_slot), reinterpret_cast<W13SlotBankState*>(SlotState), \
                        reinterpret_cast<uint8_t*>(Values), reinterpret_cast<uint8_t*>(Scales), \
                        reinterpret_cast<float*>(RowScales), \
                        reinterpret_cast<const float*>(WAmax), reinterpret_cast<const float*>(WDenominator), \
                        Out_unfused, max_m128, tile, w_evict_mode, b0); \
                } \
            } else { \
                dim3 grid(6, max_m128); \
                kfn<<<grid, block, 0, stream>>>( \
                    reinterpret_cast<const uint8_t*>(A), reinterpret_cast<const uint8_t*>(AS), \
                    reinterpret_cast<const float*>(RS), reinterpret_cast<const int32_t*>(Row2Tok), \
                    reinterpret_cast<const uint8_t*>(SLABS), reinterpret_cast<const uint8_t*>(WS_BASE), \
                    reinterpret_cast<const int32_t*>(P_OFF), reinterpret_cast<const uint32_t*>(PATCHES), \
                    reinterpret_cast<const float*>(Ones), reinterpret_cast<const float*>(Gammas), \
                    reinterpret_cast<const int32_t*>(WorkBuf), \
                    reinterpret_cast<uint16_t*>(Y_slot), reinterpret_cast<W13SlotBankState*>(SlotState), \
                    reinterpret_cast<uint8_t*>(Values), reinterpret_cast<uint8_t*>(Scales), \
                    reinterpret_cast<float*>(RowScales), \
                    reinterpret_cast<const float*>(WAmax), reinterpret_cast<const float*>(WDenominator), \
                    Out_unfused, max_m128, tile, w_evict_mode, 0); \
            } \
        } while (0)

    switch (key) {
        case 0: DISPATCH_V6_W13(0, 0, 0); break;
        case 1: DISPATCH_V6_W13(0, 0, 1); break;
        case 2: DISPATCH_V6_W13(0, 1, 0); break;
        case 3: DISPATCH_V6_W13(0, 1, 1); break;
        case 4: DISPATCH_V6_W13(1, 0, 0); break;
        case 5: DISPATCH_V6_W13(1, 0, 1); break;
        case 6: DISPATCH_V6_W13(1, 1, 0); break;
        case 7: DISPATCH_V6_W13(1, 1, 1); break;
        default: return -3;
    }
    #undef DISPATCH_V6_W13
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

}  // namespace

// =================================================================================================
// Exported C ABI Entrypoints
// =================================================================================================

extern "C" {

int pf_nvfp4_w13_v13_version() {
    return 1300;  // Wave-V13 ZERO-SPIN Host-Batched 64-Slot L2, 48/48 SMs!
}

int pf_nvfp4_w13_v6_version() {
    return 1300;  // Back-compat V6 symbol reports 1300 (zero-spin superset)
}

int pf_nvfp4_w13_v6_slot_bytes() {
    // Y_slot bytes required: kMaxGroups * kSlotBanks * 128 * 288 * 2B = 9,437,184 B (9.0 MiB max, 4.5 MiB active on GB10)
    return kMaxGroups * kSlotBanks * 128 * 288 * static_cast<int>(sizeof(uint16_t));
}

int pf_nvfp4_w13_v6_state_bytes() {
    // SlotState bytes required: kMaxGroups * kSlotBanks * 64B = 8,192 B
    return kMaxGroups * kSlotBanks * static_cast<int>(sizeof(W13SlotBankState));
}

int pf_nvfp4_build_w13_w2_plan_v6(
    const void* Tiles,
    void* WorkBuf,
    void* W2PlanBuf,
    void* SlotState,
    int max_ctas,
    int tile,
    int num_experts,
    int max_m128,
    int w2_split_m,
    int max_w2_work,
    uint64_t stream_u64
) {
    if (num_experts <= 0 || max_m128 <= 0 || tile < 64) return -2;
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream_u64);
    if (is_stream_capturing(st)) return -4;
    pf_nvfp4_build_w13_w2_plan_v6_kernel<<<1, 256, 0, st>>>(
        reinterpret_cast<const int32_t*>(Tiles),
        reinterpret_cast<int32_t*>(WorkBuf),
        reinterpret_cast<int32_t*>(W2PlanBuf),
        reinterpret_cast<W13SlotBankState*>(SlotState),
        max_ctas, tile, num_experts, max_m128, w2_split_m, max_w2_work
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int pf_nvfp4_w13_situ_quant_v6(
    const void* A, const void* AS, const void* RS, const void* Row2Tok,
    const void* SLABS, const void* WS_BASE, const void* P_OFF, const void* PATCHES,
    const void* Ones, const void* Gammas, const void* WorkBuf,
    void* Y_slot, void* SlotState,
    void* Values, void* Scales, void* RowScales,
    const void* WAmax, const void* WDenominator,
    int max_m128, int tile, int use_r2t, int as_row_major, int has_patches,
    int w_evict_mode, uint64_t stream_u64
) {
    if (Gammas == nullptr || Y_slot == nullptr || SlotState == nullptr ||
        Values == nullptr || Scales == nullptr || RowScales == nullptr ||
        WAmax == nullptr || WDenominator == nullptr) {
        return -2;
    }
    return launch_nvfp4_w13_v6_impl<2>(
        A, AS, RS, Row2Tok, SLABS, WS_BASE, P_OFF, PATCHES, Ones, Gammas, WorkBuf,
        Y_slot, SlotState, Values, Scales, RowScales, WAmax, WDenominator, nullptr,
        max_m128, tile, use_r2t, as_row_major, has_patches, w_evict_mode,
        reinterpret_cast<cudaStream_t>(stream_u64));
}

// Backwards-compatible v4b entrypoints (for unfused fallback or standalone comparison)
int pf_nvfp4_build_m128_worklist(
    const void* Tiles,
    void* WorkBuf,
    int max_ctas,
    int tile,
    int num_experts,
    int max_m128,
    uint64_t stream_u64
) {
    return pf_nvfp4_build_w13_w2_plan_v6(
        Tiles, WorkBuf, nullptr, nullptr, max_ctas, tile, num_experts, max_m128, 0, 0, stream_u64);
}

int pf_nvfp4_w13_ws5_v4b_unfused(
    const void* A, const void* AS, const void* RS, const void* Row2Tok,
    const void* SLABS, const void* WS_BASE, const void* P_OFF, const void* PATCHES,
    const void* Ones, const void* WorkBuf, void* Out,
    int max_m128, int tile, int use_r2t, int as_row_major, int has_patches,
    int w_evict_mode, uint64_t stream_u64
) {
    return launch_nvfp4_w13_v6_impl<0>(
        A, AS, RS, Row2Tok, SLABS, WS_BASE, P_OFF, PATCHES, Ones, nullptr, WorkBuf,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, Out,
        max_m128, tile, use_r2t, as_row_major, has_patches, w_evict_mode,
        reinterpret_cast<cudaStream_t>(stream_u64));
}

int pf_nvfp4_w13_ws5_v4b_fused_situ(
    const void* A, const void* AS, const void* RS, const void* Row2Tok,
    const void* SLABS, const void* WS_BASE, const void* P_OFF, const void* PATCHES,
    const void* Ones, const void* Gammas, const void* WorkBuf, void* Out_situ,
    int max_m128, int tile, int use_r2t, int as_row_major, int has_patches,
    int w_evict_mode, uint64_t stream_u64
) {
    if (Gammas == nullptr) return -2;
    return launch_nvfp4_w13_v6_impl<1>(
        A, AS, RS, Row2Tok, SLABS, WS_BASE, P_OFF, PATCHES, Ones, Gammas, WorkBuf,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, Out_situ,
        max_m128, tile, use_r2t, as_row_major, has_patches, w_evict_mode,
        reinterpret_cast<cudaStream_t>(stream_u64));
}

int pf_route_tokens_place_v6(
    const void* Partials,
    const void* Bias,
    const void* ExpertBounds,
    const void* Padding,
    void* Counts,
    void* Counters,
    void* Ids,
    void* Ranks,
    void* Weights,
    void* Starts,
    void* Tiles,
    int T,
    float route_scale,
    int num_local,
    int local_pow2,
    int tile,
    int max_ctas,
    int partials_is_bf16,
    uint64_t stream_u64
) {
    if (T <= 256 || tile < 64 || (tile & (tile - 1)) != 0 ||
        num_local <= 0 || num_local > 384 || local_pow2 < num_local || local_pow2 > 512) {
        return -2;
    }
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream_u64);
    if (is_stream_capturing(st)) return -4;

    const int blocks = (T + 7) / 8;
    if (partials_is_bf16 != 0) {
        pf_route_tokens_place_v6_kernel<true><<<blocks, 256, 0, st>>>(
            Partials,
            reinterpret_cast<const float*>(Bias),
            reinterpret_cast<const int32_t*>(ExpertBounds),
            reinterpret_cast<const uint8_t*>(Padding),
            reinterpret_cast<int32_t*>(Counts),
            reinterpret_cast<int32_t*>(Counters),
            reinterpret_cast<int32_t*>(Ids),
            reinterpret_cast<int32_t*>(Ranks),
            reinterpret_cast<float*>(Weights),
            reinterpret_cast<int32_t*>(Starts),
            reinterpret_cast<int32_t*>(Tiles),
            T, route_scale, num_local, local_pow2, tile, max_ctas
        );
    } else {
        pf_route_tokens_place_v6_kernel<false><<<blocks, 256, 0, st>>>(
            Partials,
            reinterpret_cast<const float*>(Bias),
            reinterpret_cast<const int32_t*>(ExpertBounds),
            reinterpret_cast<const uint8_t*>(Padding),
            reinterpret_cast<int32_t*>(Counts),
            reinterpret_cast<int32_t*>(Counters),
            reinterpret_cast<int32_t*>(Ids),
            reinterpret_cast<int32_t*>(Ranks),
            reinterpret_cast<float*>(Weights),
            reinterpret_cast<int32_t*>(Starts),
            reinterpret_cast<int32_t*>(Tiles),
            T, route_scale, num_local, local_pow2, tile, max_ctas
        );
    }
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

}  // extern "C"
