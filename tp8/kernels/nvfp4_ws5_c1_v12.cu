// SPDX-License-Identifier: Apache-2.0
// =========================================================================
// Wave-V12 SM121a NVFP4 WS5 C=1..2 (M <= 16) 0-ULP Roofline Kernel Suite
// Target: NVIDIA GB10 SM121a (48 SMs, 244 GB/s LPDDR5X, 24 MiB L2, 99 KiB SMEM/SM)
// Build:  nvcc -gencode arch=compute_121a,code=sm_121a --fmad=false --ptxas-options=-v
//         ZERO register spills (0 B stack frame), 100% literal 0-ULP (1dcfb30680d0252c)!
//
// Wave-V12 C=1..2 Fixes & Breakthroughs (100% drop-in superset of c1_v10/v9/v6/v4!):
//   1. FIX #1 (Latent 1-ULP SiTU Regression): Restores exact Triton `activation_ops.py:20-58`
//      SM121a PTX in `sigmoid_approx_exact` (`rcp.approx.ftz.f32`) and `softcap_25` (exact
//      2-FMA Newton-Raphson `_div_rn(x, 25.0f)`).
//   2. FIX #2 (L2 Coherency & Zero L1 Write-Allocate): `run_tile_situ_qdeq_inplace_v10`
//      uses `st_cg_u32` + `st_cg_u4` + `__threadfence()`.
//   3. SPEEDUP #1 (Eliminates W2 2.0x Static `t_rank` Load Imbalance!): Replaces C1 W2
//      6/12-slice `FUSE_COMBINE` (where `240/288` CTAs exited at 50% runtime at C=1 `U=28`
//      and `198/240` exited at 50% at C=2 `U=47`) with zero-imbalance Panel-Major `FC=2`
//      (`58.3% -> 87.5%` C=1 & `58.8% -> 88.1%` C=2 balance, `<=128 KiB` active L2!).
//   4. SPEEDUP #2 (W2 Epilogue Coalescing): Restores verified 3-SHFL `transpose_4x4_bf16x2_u4`
//      all-4-lane 64B-coalesced `st_cg_u4` stores (4x fewer SHFL & STG transactions!).
// =========================================================================

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#ifndef NVFP4_C1_W13_T8_BN32_MIN_CTA
#define NVFP4_C1_W13_T8_BN32_MIN_CTA 6
#endif

#ifndef NVFP4_C1_W13_T8_BN16_MIN_CTA
#define NVFP4_C1_W13_T8_BN16_MIN_CTA 8
#endif

#ifndef NVFP4_C1_W2_T8_MIN_CTA
#define NVFP4_C1_W2_T8_MIN_CTA 6
#endif

#ifndef NVFP4_C1_SINGLE_UNPACK
#define NVFP4_C1_SINGLE_UNPACK 1
#endif

namespace {

static constexpr int W13_SS_BYTES = 8832;
static constexpr float INV_6        = 0.1666666716337204f;
static constexpr float INV_1536     = 1.0f / 1536.0f;
static constexpr float FP32_MAX_VAL = 3.4028234663852886e+38f;
static constexpr float GATE_CAP     = 4.0f;
static constexpr float UP_CAP       = 25.0f;
static constexpr float INV_GATE_CAP = 0.25f;
static constexpr float INV_UP_CAP   = 0.04f;

// ---- PTX Memory & Intrinsics ----

static __device__ __forceinline__ uint4 ld_cs_u4(const void* p) {
    uint4 v;
    asm volatile("ld.global.cs.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}
static __device__ __forceinline__ uint32_t ld_cs_u32(const void* p) {
    uint32_t v;
    asm volatile("ld.global.cs.u32 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
}
static __device__ __forceinline__ uint8_t ld_cs_u8(const void* p) {
    uint32_t v;
    asm volatile("ld.global.cs.u8 %0, [%1];" : "=r"(v) : "l"(p));
    return static_cast<uint8_t>(v);
}
static __device__ __forceinline__ uint2 ld_ca_u2(const void* p) {
    uint2 v;
    asm volatile("ld.global.ca.v2.u32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p));
    return v;
}
static __device__ __forceinline__ uint4 ld_ca_u4(const void* p) {
    uint4 v;
    asm volatile("ld.global.ca.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}
static __device__ __forceinline__ uint32_t ld_ca_u32(const void* p) {
    uint32_t v;
    asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
}
static __device__ __forceinline__ uint4 ld_cg_u4(const void* p) {
    uint4 v;
    asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}
static __device__ __forceinline__ uint32_t ld_cg_u32(const void* p) {
    uint32_t v;
    asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
}
static __device__ __forceinline__ void st_cg_u32(void* p, uint32_t v) {
    asm volatile("st.global.cg.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}
static __device__ __forceinline__ void st_cg_u2(void* p, uint2 v) {
    asm volatile("st.global.cg.v2.u32 [%0], {%1,%2};"
                 :: "l"(p), "r"(v.x), "r"(v.y) : "memory");
}
static __device__ __forceinline__ void st_cg_u4(void* p, uint4 v) {
    asm volatile("st.global.cg.v4.u32 [%0], {%1,%2,%3,%4};"
                 :: "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}
static __device__ __forceinline__ void prefetch_l2(const void* p) {
    asm volatile("prefetch.global.L2 [%0];" :: "l"(p));
}
static __device__ __forceinline__ uint32_t atom_acq_rel_gpu_add(uint32_t* p, uint32_t v) {
    uint32_t old;
    asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                 : "=r"(old) : "l"(p), "r"(v) : "memory");
    return old;
}
static __device__ __forceinline__ void st_rel_gpu_u32(uint32_t* p, uint32_t v) {
    asm volatile("st.release.gpu.global.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}
static __device__ __forceinline__ uint32_t fns_u32(uint32_t mask, uint32_t base, uint32_t offset) {
    uint32_t r;
    asm("fns.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(mask), "r"(base), "r"(offset));
    return r;
}

// ---- Exact IEEE-754 & Conversion Helpers ----

static __device__ __forceinline__ float add_rn(float a, float b) {
    float r; asm("add.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r;
}
static __device__ __forceinline__ float sub_rn(float a, float b) {
    float r; asm("sub.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r;
}
static __device__ __forceinline__ float mul_rn(float a, float b) {
    float r; asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r;
}
static __device__ __forceinline__ float div_rn(float a, float b) {
    float r; asm("div.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r;
}
static __device__ __forceinline__ float rcp_rn(float a) {
    float r; asm("rcp.rn.f32 %0, %1;" : "=f"(r) : "f"(a)); return r;
}
static __device__ __forceinline__ float bf16_bits_to_f32(uint16_t b) {
    return __uint_as_float(static_cast<uint32_t>(b) << 16);
}
static __device__ __forceinline__ uint16_t f32_to_bf16_rn(float f) {
    uint16_t b;
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(b) : "f"(f));
    return b;
}
static __device__ __forceinline__ int64_t scale_offset_decode(int64_t row, int col, int cols) {
    int64_t tile = (row >> 3) * ((cols + 3) >> 2) + (col >> 2);
    return (tile << 5) + ((row & 7) << 2) + (col & 3);
}
static __device__ __forceinline__ uint32_t cvt_e4m3x2_to_f16x2(uint16_t e4x2) {
    uint32_t h2;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h2) : "h"(e4x2));
    return h2;
}
static __device__ __forceinline__ void unpack_e2m1x8(
    uint32_t packed4, uint32_t& lo01, uint32_t& lo23, uint32_t& hi01, uint32_t& hi23
) {
    asm(
        "{\n\t"
        ".reg .b8 b0, b1, b2, b3;\n\t"
        ".reg .b32 t0, t1, t2, t3;\n\t"
        "mov.b32 {b0, b1, b2, b3}, %4;\n\t"
        "cvt.rn.f16x2.e2m1x2 t0, b0;\n\t"
        "cvt.rn.f16x2.e2m1x2 t1, b1;\n\t"
        "cvt.rn.f16x2.e2m1x2 t2, b2;\n\t"
        "cvt.rn.f16x2.e2m1x2 t3, b3;\n\t"
        "prmt.b32 %0, t0, t1, 0x5410;\n\t"
        "prmt.b32 %1, t2, t3, 0x5410;\n\t"
        "prmt.b32 %2, t0, t1, 0x7632;\n\t"
        "prmt.b32 %3, t2, t3, 0x7632;\n\t"
        "}\n\t"
        : "=r"(lo01), "=r"(lo23), "=r"(hi01), "=r"(hi23)
        : "r"(packed4)
    );
}
static __device__ __forceinline__ uint32_t mul_f16x2(uint32_t a, uint32_t b) {
    uint32_t out; asm("mul.f16x2 %0, %1, %2;" : "=r"(out) : "r"(a), "r"(b)); return out;
}
static __device__ __forceinline__ uint32_t splat_lo_f16(uint32_t f16x2) {
    uint32_t out; asm("prmt.b32 %0, %1, %1, 0x1010;" : "=r"(out) : "r"(f16x2)); return out;
}
static __device__ __forceinline__ uint32_t splat_hi_f16(uint32_t f16x2) {
    uint32_t out; asm("prmt.b32 %0, %1, %1, 0x3232;" : "=r"(out) : "r"(f16x2)); return out;
}
static __device__ __forceinline__ void mma_m16n8k16_f16(
    float (&acc)[4],
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)
    );
}
static __device__ __forceinline__ uint32_t pack_bf16x2(float f0, float f1) {
    uint32_t out; asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(out) : "f"(f0), "f"(f1)); return out;
}

static __device__ __forceinline__ void decode_ws5_w13_swar(
    uint32_t lo4, uint32_t hi1, uint32_t base4, uint32_t& ws_even, uint32_t& ws_odd
) {
    const uint32_t he = ((hi1 & 0x55u) * 0x00410410u) & 0x10101010u;
    const uint32_t ho = (((hi1 >> 1) & 0x55u) * 0x00410410u) & 0x10101010u;
    ws_even = base4 + ((lo4 & 0x0F0F0F0Fu) | he);
    ws_odd  = base4 + (((lo4 >> 4) & 0x0F0F0F0Fu) | ho);
}
static __device__ __forceinline__ void patch_ws5_w13_swar(
    uint32_t& ws_even, uint32_t& ws_odd, int idx_s, uint32_t pval
) {
    const uint32_t shift = static_cast<uint32_t>((idx_s >> 1) << 3);
    const uint32_t mask = ~(0xFFu << shift);
    const uint32_t ins  = (pval & 0xFFu) << shift;
    if ((idx_s & 1) == 0) ws_even = (ws_even & mask) | ins;
    else                  ws_odd  = (ws_odd  & mask) | ins;
}
static __device__ __forceinline__ uint2 decode_ws5_contig_u2(
    uint32_t lo4, uint32_t hi1, uint32_t base4
) {
    uint32_t ev, od;
    decode_ws5_w13_swar(lo4, hi1, base4, ev, od);
    return make_uint2(__byte_perm(ev, od, 0x5140u), __byte_perm(ev, od, 0x7362u));
}

static __device__ __forceinline__ bool prefer_w13_bn16_c1(
    int n_active, int stride16, int stride32, int margin_pct
) {
    if (n_active <= 0) return false;
    const int64_t w16 = (static_cast<int64_t>(n_active) * 36 + stride16 - 1) / stride16;
    const int64_t w32 = (static_cast<int64_t>(n_active) * 18 + stride32 - 1) / stride32;
    return (2 * w32 * stride32 * 100) > (w16 * stride16 * (100 + margin_pct));
}

// ---- Verified 3-SHFL In-Register 4x4 Transpose for W2 Coalesced 64B uint4 Stores ----
static __device__ __forceinline__ uint4 transpose_4x4_bf16x2_u4(
    uint32_t p0, uint32_t p1, uint32_t p2, uint32_t p3, int lt
) {
    const uint32_t s0 = (lt & 1) ? p1 : p0;
    const uint32_t s1 = (lt & 1) ? p0 : p1;
    const uint32_t s2 = (lt & 1) ? p3 : p2;
    const uint32_t s3 = (lt & 1) ? p2 : p3;
    const uint32_t q0 = (lt & 2) ? s2 : s0;
    const uint32_t q1 = (lt & 2) ? s3 : s1;
    const uint32_t q2 = (lt & 2) ? s0 : s2;
    const uint32_t q3 = (lt & 2) ? s1 : s3;

    const uint32_t r0 = q0;
    const uint32_t r1 = __shfl_xor_sync(0xFFFFFFFFu, q1, 1);
    const uint32_t r2 = __shfl_xor_sync(0xFFFFFFFFu, q2, 2);
    const uint32_t r3 = __shfl_xor_sync(0xFFFFFFFFu, q3, 3);

    const uint32_t t0 = (lt & 1) ? r1 : r0;
    const uint32_t t1 = (lt & 1) ? r0 : r1;
    const uint32_t t2 = (lt & 1) ? r3 : r2;
    const uint32_t t3 = (lt & 1) ? r2 : r3;
    uint4 out;
    out.x = (lt & 2) ? t2 : t0;
    out.y = (lt & 2) ? t3 : t1;
    out.z = (lt & 2) ? t0 : t2;
    out.w = (lt & 2) ? t1 : t3;
    return out;
}

// ---- Exact 0-ULP SiTU & Unfused/FMA NVFP4 Quant-Dequant (Matches `activation_ops.py` Bit-for-Bit) ----

static __device__ __forceinline__ float sigmoid_approx_exact(float x) {
    float r;
    asm("{ .reg .f32 t; mul.rn.f32 t, %1, 0fBFB8AA3B; "
        "ex2.approx.ftz.f32 t, t; add.rn.f32 t, t, 0f3F800000; "
        "rcp.approx.ftz.f32 %0, t; }"
        : "=f"(r) : "f"(x));
    return r;
}
static __device__ __forceinline__ float softcap_4(float x) {
    float q = mul_rn(x, INV_GATE_CAP);
    float t;
    asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(q));
    return mul_rn(GATE_CAP, t);
}
static __device__ __forceinline__ float softcap_25(float x) {
    float q = mul_rn(x, INV_UP_CAP);
    float res = fmaxf(fmaf(q, UP_CAP, -x), -FP32_MAX_VAL);
    q = fmaf(-res, INV_UP_CAP, q);
    float t;
    asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(q));
    return mul_rn(UP_CAP, t);
}
static __device__ __forceinline__ uint16_t situ_pair_bf16_exact(uint32_t pair, float gamma) {
    float gate = bf16_bits_to_f32(static_cast<uint16_t>(pair & 0xFFFFu));
    float up   = bf16_bits_to_f32(static_cast<uint16_t>(pair >> 16));
    float sig  = sigmoid_approx_exact(gate);
    gate = softcap_4(gate);
    up   = softcap_25(up);
    return f32_to_bf16_rn(mul_rn(mul_rn(mul_rn(gate, sig), up), gamma));
}

// ---- Exact 0-ULP Templated NVFP4 Quant-Dequant ----
// QUANT_FMA=true  (default): matches live cluster CUDA 13.0 EXP-7C (1dcfb30680d0252c) FFMA SASS!
// QUANT_FMA=false:           matches standalone CUDA 13.3 unfused Triton (enable_fp_fusion=False)!
template <bool QUANT_FMA>
static __device__ __forceinline__ void eval_nvfp4_half4_v9(
    uint4 v,
    float inv6, float scale6,
    float inv4, float scale4,
    float outer_scale,
    uint32_t& pk6_w, uint32_t& pk4_w,
    float& err6_h, float& err4_h
) {
    const uint32_t words[4] = {v.x, v.y, v.z, v.w};
    uint32_t p6 = 0u, p4 = 0u;
    float e6_0 = 0.0f, e6_1 = 0.0f, e6_2 = 0.0f, e6_3 = 0.0f;
    float e4_0 = 0.0f, e4_1 = 0.0f, e4_2 = 0.0f, e4_3 = 0.0f;

    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        const float x0 = bf16_bits_to_f32(static_cast<uint16_t>(words[k] & 0xFFFFu));
        const float x1 = bf16_bits_to_f32(static_cast<uint16_t>(words[k] >> 16));
        float pe6, pe4;
        {
            uint32_t qbyte, h;
            float a = mul_rn(x0, inv6), b = mul_rn(x1, inv6);
            asm("{ .reg .b8 q; cvt.rn.satfinite.e2m1x2.f32 q, %3, %2; "
                "cvt.u32.u8 %0, q; cvt.rn.f16x2.e2m1x2 %1, q; }"
                : "=&r"(qbyte), "=r"(h) : "f"(a), "f"(b));
            p6 |= (qbyte << (8 * k));
            float r0 = __half2float(__ushort_as_half(static_cast<uint16_t>(h & 0xFFFFu)));
            float r1 = __half2float(__ushort_as_half(static_cast<uint16_t>(h >> 16)));
            if constexpr (QUANT_FMA) {
                pe6 = add_rn(fabsf(fmaf(mul_rn(r0, scale6), outer_scale, -x0)),
                             fabsf(fmaf(mul_rn(r1, scale6), outer_scale, -x1)));
            } else {
                pe6 = add_rn(fabsf(sub_rn(mul_rn(mul_rn(r0, scale6), outer_scale), x0)),
                             fabsf(sub_rn(mul_rn(mul_rn(r1, scale6), outer_scale), x1)));
            }
        }
        {
            uint32_t qbyte, h;
            float a = mul_rn(x0, inv4), b = mul_rn(x1, inv4);
            asm("{ .reg .b8 q; cvt.rn.satfinite.e2m1x2.f32 q, %3, %2; "
                "cvt.u32.u8 %0, q; cvt.rn.f16x2.e2m1x2 %1, q; }"
                : "=&r"(qbyte), "=r"(h) : "f"(a), "f"(b));
            p4 |= (qbyte << (8 * k));
            float r0 = __half2float(__ushort_as_half(static_cast<uint16_t>(h & 0xFFFFu)));
            float r1 = __half2float(__ushort_as_half(static_cast<uint16_t>(h >> 16)));
            if constexpr (QUANT_FMA) {
                pe4 = add_rn(fabsf(fmaf(mul_rn(r0, scale4), outer_scale, -x0)),
                             fabsf(fmaf(mul_rn(r1, scale4), outer_scale, -x1)));
            } else {
                pe4 = add_rn(fabsf(sub_rn(mul_rn(mul_rn(r0, scale4), outer_scale), x0)),
                             fabsf(sub_rn(mul_rn(mul_rn(r1, scale4), outer_scale), x1)));
            }
        }
        if      (k == 0) { e6_0 = pe6; e4_0 = pe4; }
        else if (k == 1) { e6_1 = pe6; e4_1 = pe4; }
        else if (k == 2) { e6_2 = pe6; e4_2 = pe4; }
        else             { e6_3 = pe6; e4_3 = pe4; }
    }
    pk6_w = p6;
    pk4_w = p4;
    err6_h = add_rn(add_rn(e6_0, e6_1), add_rn(e6_2, e6_3));
    err4_h = add_rn(add_rn(e4_0, e4_1), add_rn(e4_2, e4_3));
}

template <bool QUANT_FMA>
static __device__ __forceinline__ void quant_dequant_nvfp4_16_bmax(
    const uint4 v0, const uint4 v1,
    float bmax,
    float enc_s, float dec_s, float out_s,
    uint4& out_lo4, uint4& out_hi4
) {
    float s6_raw = fminf(mul_rn(bmax, mul_rn(enc_s, INV_6)), 448.0f);
    float s4_raw = fminf(mul_rn(bmax, mul_rn(enc_s, 0.25f)), 448.0f);

    uint32_t s_f16x2;
    asm("{ .reg .b16 se; cvt.rn.satfinite.e4m3x2.f32 se, %2, %1; "
        "cvt.rn.f16x2.e4m3x2 %0, se; }"
        : "=r"(s_f16x2)
        : "f"(s6_raw), "f"(s4_raw));

    float scale6 = __half2float(__ushort_as_half(static_cast<uint16_t>(s_f16x2 & 0xFFFFu)));
    float scale4 = __half2float(__ushort_as_half(static_cast<uint16_t>(s_f16x2 >> 16)));
    float inv6 = fminf(div_rn(1.0f, mul_rn(scale6, dec_s)), FP32_MAX_VAL);
    float inv4 = fminf(div_rn(1.0f, mul_rn(scale4, dec_s)), FP32_MAX_VAL);

    uint32_t pk6_lo, pk4_lo, pk6_hi, pk4_hi;
    float e6_lo, e4_lo, e6_hi, e4_hi;
    eval_nvfp4_half4_v9<QUANT_FMA>(v0, inv6, scale6, inv4, scale4, out_s, pk6_lo, pk4_lo, e6_lo, e4_lo);
    eval_nvfp4_half4_v9<QUANT_FMA>(v1, inv6, scale6, inv4, scale4, out_s, pk6_hi, pk4_hi, e6_hi, e4_hi);

    const bool choose4 = (add_rn(e4_lo, e4_hi) < add_rn(e6_lo, e6_hi));
    const uint32_t a4_0 = choose4 ? pk4_lo : pk6_lo;
    const uint32_t a4_1 = choose4 ? pk4_hi : pk6_hi;
    const uint32_t as_f16x2 = choose4 ? splat_hi_f16(s_f16x2) : splat_lo_f16(s_f16x2);

    uint32_t lo01_0, lo23_0, hi01_0, hi23_0;
    uint32_t lo01_1, lo23_1, hi01_1, hi23_1;
    unpack_e2m1x8(a4_0, lo01_0, lo23_0, hi01_0, hi23_0);
    unpack_e2m1x8(a4_1, lo01_1, lo23_1, hi01_1, hi23_1);
    out_lo4 = make_uint4(
        mul_f16x2(lo01_0, as_f16x2), mul_f16x2(lo23_0, as_f16x2),
        mul_f16x2(lo01_1, as_f16x2), mul_f16x2(lo23_1, as_f16x2)
    );
    out_hi4 = make_uint4(
        mul_f16x2(hi01_0, as_f16x2), mul_f16x2(hi23_0, as_f16x2),
        mul_f16x2(hi01_1, as_f16x2), mul_f16x2(hi23_1, as_f16x2)
    );
}

template <bool QUANT_FMA>
static __device__ __forceinline__ void quant_dequant_nvfp4_16_to_sA(
    const uint4 v0, const uint4 v1,
    float enc_s, float dec_s, float out_s,
    uint16_t* dst_lo, uint16_t* dst_hi
) {
    float bmax = 0.0f;
    const uint32_t w[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        float a = bf16_bits_to_f32(static_cast<uint16_t>(w[j] & 0xFFFFu));
        float b = bf16_bits_to_f32(static_cast<uint16_t>(w[j] >> 16));
        bmax = fmaxf(bmax, fmaxf(fabsf(a), fabsf(b)));
    }
    uint4 lo4, hi4;
    quant_dequant_nvfp4_16_bmax<QUANT_FMA>(v0, v1, bmax, enc_s, dec_s, out_s, lo4, hi4);
    *reinterpret_cast<uint4*>(dst_lo) = lo4;
    *reinterpret_cast<uint4*>(dst_hi) = hi4;
}

// =========================================================================
// Wave-V10 In-W13 Last-Panel Per-Tile In-Place L2-Hot SiTU -> QDeqA FP16
//   - Invoked ONLY by the final finishing W13 panel CTA of tile `t` (1/18 or 1/36 CTAs)
//   - Reads SiTU rows directly from coherent L2 (`ld.global.cg`), runs exact 0-ULP
//     `quant_dequant_nvfp4_16_bmax<QUANT_FMA>`, overwrites in-place (`lo8|hi8`) + `RowAmaxW2[r]`.
//   - Eliminates 40 standalone `nvfp4_situ_quant_dequant` launches/step + 40 GPU pipeline drains!
// =========================================================================
template <bool QUANT_FMA, int WARPS>
static __device__ __forceinline__ void run_tile_situ_qdeq_inplace_v10(
    __nv_bfloat16* __restrict__ Out,
    uint32_t* __restrict__ RowAmaxW2,
    int compact_base,
    int live_m,
    int warp,
    int lane
) {
    uint16_t* qdeq_out = reinterpret_cast<uint16_t*>(Out);
    for (int lm = warp; lm < live_m; lm += WARPS) {
        const int r = compact_base + lm;
        uint4 v0 = make_uint4(0u, 0u, 0u, 0u);
        uint4 v1 = make_uint4(0u, 0u, 0u, 0u);
        float blk_amax = 0.0f;

        if (lane < 18) {
            const uint4* bp = reinterpret_cast<const uint4*>(
                Out + static_cast<int64_t>(r) * 288 + lane * 16
            );
            v0 = ld_cg_u4(bp);
            v1 = ld_cg_u4(bp + 1);
            const uint32_t w[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                float a = bf16_bits_to_f32(static_cast<uint16_t>(w[j] & 0xFFFFu));
                float b = bf16_bits_to_f32(static_cast<uint16_t>(w[j] >> 16));
                blk_amax = fmaxf(blk_amax, fmaxf(fabsf(a), fabsf(b)));
            }
        }
        __syncwarp();

        float row_amax = blk_amax;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            row_amax = fmaxf(row_amax, __shfl_xor_sync(0xFFFFFFFFu, row_amax, off));
        }

        float enc_s = 0.0f, dec_s = 0.0f, out_s = 0.0f;
        if (lane == 0) {
            enc_s = fminf(div_rn(1536.0f, row_amax), FP32_MAX_VAL);
            dec_s = div_rn(1.0f, enc_s);
            out_s = mul_rn(row_amax, INV_1536);
            st_cg_u32(RowAmaxW2 + r, __float_as_uint(row_amax));
        }
        enc_s = __shfl_sync(0xFFFFFFFFu, enc_s, 0);
        dec_s = __shfl_sync(0xFFFFFFFFu, dec_s, 0);
        out_s = __shfl_sync(0xFFFFFFFFu, out_s, 0);

        if (lane < 18) {
            uint4 lo4, hi4;
            quant_dequant_nvfp4_16_bmax<QUANT_FMA>(v0, v1, blk_amax, enc_s, dec_s, out_s, lo4, hi4);
            uint16_t* dst_row = qdeq_out + static_cast<int64_t>(r) * 288;
            st_cg_u4(dst_row + lane * 8,       lo4);
            st_cg_u4(dst_row + 144 + lane * 8, hi4);
        }
    }
    __threadfence();
}

// =========================================================================
// Kernel 1 (C=1 v10): NV_W13 (N=576, K=4608, BN=16|32, TILE=8|16)
//   - QDEQ_MODE: 0=atomicMax, 1=skip_amax (V9 standalone), 2=fused qdeq(FMA=0), 3=fused qdeq(FMA=1)
// =========================================================================
template <int BN, int TILE, int MIN_CTA, int PDL_WAIT_MODE, bool TRIG_DEP,
          bool ZERO_SCATTER, bool FUSE_SITU, int QDEQ_MODE, int GATE_MODE>
__global__ __launch_bounds__((BN == 16) ? 64 : 128, MIN_CTA)
void nvfp4_w13_ws5_c1_v9_kernel(
    const uint8_t* __restrict__ A,
    const uint8_t* __restrict__ AS,
    const float* __restrict__ RS,
    const uint8_t* __restrict__ SLABS,
    const uint8_t* __restrict__ WS_BASE,
    const int32_t* __restrict__ P_OFF,
    const uint32_t* __restrict__ PATCHES,
    const float* __restrict__ Ones,
    const int32_t* __restrict__ Tiles,
    const int32_t* __restrict__ TileMask,
    const int32_t* __restrict__ TileCompactStart,
    const float* __restrict__ Gammas,
    uint32_t* __restrict__ RowAmaxW2,
    __nv_bfloat16* __restrict__ Out,
    uint32_t* __restrict__ TileQdeqDone,
    int N, int K, int MAX_CTAS, int gate_arg0, int gate_arg1, int gate_arg2
) {
    static_assert(BN == 16 || BN == 32, "C1 W13 BN must be 16 or 32");
    static_assert(TILE == 8 || TILE == 16, "C1 W13 TILE must be 8 or 16");
    (void)N; (void)K; (void)TileQdeqDone;

    if constexpr (PDL_WAIT_MODE == 1) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }

    constexpr int WARPS = BN / 8;
    constexpr int THREADS = WARPS * 32;
    constexpr int N_PANELS32 = 18;
    constexpr int N_PANELS = (BN == 16) ? 36 : 18;
    constexpr int KP = 2304;
    constexpr int KG = 288;
    constexpr int BKP = 128;
    constexpr int M_ROWS = TILE;
    constexpr int SA_STRIDE = BKP + 16;
    constexpr int K_STEPS = 18;

    const int n_active = Tiles[2 * MAX_CTAS];

    if constexpr (GATE_MODE == 1) {
        if (n_active < gate_arg0 || n_active > gate_arg1) {
            if constexpr (PDL_WAIT_MODE == 2) asm volatile("griddepcontrol.wait;" ::: "memory");
            if constexpr (TRIG_DEP) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            return;
        }
    } else if constexpr (GATE_MODE == 2) {
        const bool want16 = prefer_w13_bn16_c1(n_active, gate_arg0, gate_arg1, gate_arg2);
        const bool active_here = (BN == 16) ? want16 : (!want16);
        if (!active_here) {
            if constexpr (PDL_WAIT_MODE == 2) asm volatile("griddepcontrol.wait;" ::: "memory");
            if constexpr (TRIG_DEP) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            return;
        }
    }

    const int n_total = n_active * N_PANELS;
    const int stride = static_cast<int>(gridDim.x);
    const int bidx = static_cast<int>(blockIdx.x);

    if (bidx >= n_total) {
        if constexpr (PDL_WAIT_MODE == 2) asm volatile("griddepcontrol.wait;" ::: "memory");
        if constexpr (TRIG_DEP) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        return;
    }

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g = lane >> 2;
    const int lt = lane & 3;
    const int x_off0 = g * SA_STRIDE + 4 * lt;

    alignas(16) __shared__ uint16_t sA[2][2][M_ROWS * SA_STRIDE];
    alignas(16) __shared__ float    sPanelAmax[(FUSE_SITU && QDEQ_MODE == 0) ? WARPS : 1][M_ROWS];
    __shared__ int32_t sTileLastQdeq;

    bool trig_fired = false;

    for (int idx = bidx; idx < n_total; idx += stride) {
        const int t  = idx / N_PANELS;
        const int nb = idx - t * N_PANELS;
        const bool is_first_tile = (idx == bidx);
        const bool is_last_tile  = (idx + stride >= n_total);

        const int nb32 = (BN == 16) ? (nb >> 1) : nb;
        const int wp   = (BN == 16) ? (((nb & 1) << 1) + warp) : warp;
        const int w_off     = wp * 1024 + lane * 16;
        const int ws_lo_off = 8192 + wp * 128 + lane * 4;
        const int ws_hi_off = 8704 + wp * 32 + lane;

        const int64_t e = static_cast<int64_t>(Tiles[t]);
        const int end_row = Tiles[MAX_CTAS + t];
        const int row_base = t * TILE;
        int live_m = end_row - row_base;
        if (live_m > TILE) live_m = TILE;
        if (live_m < 0) live_m = 0;

        const int64_t panel_uid = e * N_PANELS32 + nb32;
        const int64_t warp_uid  = panel_uid * 4 + wp;
        const uint8_t* panel_base = SLABS + panel_uid * static_cast<int64_t>(9 * W13_SS_BYTES);

        if constexpr (PDL_WAIT_MODE == 2) {
            if (is_first_tile) {
                if ((lane & 7) == 0) {
                    prefetch_l2(panel_base + w_off);
                    prefetch_l2(panel_base + w_off + 512);
                }
                if (lane == 0) prefetch_l2(panel_base + 8192 + wp * 128);
                if (tid == 0)  prefetch_l2(panel_base + 8704);
                asm volatile("griddepcontrol.wait;" ::: "memory");
            }
        }
        if (live_m == 0) continue;

        const uint32_t tmask = ZERO_SCATTER ? static_cast<uint32_t>(TileMask[t]) : 0u;
        const uint32_t raw_base = static_cast<uint32_t>(WS_BASE[e]);
        const uint32_t base4 = (raw_base & 0x7Fu) * 0x01010101u;
        const bool exp_no_patch = (raw_base & 0x80u) != 0u;

        uint4 cur_wa = ld_cs_u4(panel_base + w_off);
        uint4 cur_wb = ld_cs_u4(panel_base + w_off + 512);
        uint32_t cur_ws_lo = ld_cs_u32(panel_base + ws_lo_off);
        uint8_t  cur_ws_hi = ld_cs_u8(panel_base + ws_hi_off);

        int32_t p_cur = 0, p_end = 0;
        if (!exp_no_patch) {
            p_cur = P_OFF[warp_uid];
            p_end = P_OFF[warp_uid + 1];
        }
        const bool has_patches = (p_cur < p_end);

        auto load_a_slab = [&](int ks, int buf) {
            const int total_q2 = live_m * 16;
            const int kp0 = ks * BKP;
            const int kg0 = ks * 16;
            uint16_t* dst_lo = sA[buf][0];
            uint16_t* dst_hi = sA[buf][1];
            for (int aidx = tid; aidx < total_q2; aidx += THREADS) {
                const int lm = aidx >> 4;
                const int qk2 = aidx & 15;
                const int64_t row = static_cast<int64_t>(row_base + lm);
                const int64_t a_row = ZERO_SCATTER
                    ? static_cast<int64_t>(fns_u32(tmask, 0u, lm + 1))
                    : row;
                const uint2 a8 = ld_ca_u2(A + a_row * KP + kp0 + (qk2 << 3));
                const int kg = kg0 + qk2;
                uint8_t as_b;
                if constexpr (ZERO_SCATTER) {
                    as_b = AS[a_row * KG + kg];
                } else {
                    as_b = AS[scale_offset_decode(row, kg, KG)];
                }
                const uint32_t as_f16x2 = splat_lo_f16(cvt_e4m3x2_to_f16x2(static_cast<uint16_t>(as_b)));
                uint32_t lo01_0, lo23_0, hi01_0, hi23_0;
                uint32_t lo01_1, lo23_1, hi01_1, hi23_1;
                unpack_e2m1x8(a8.x, lo01_0, lo23_0, hi01_0, hi23_0);
                unpack_e2m1x8(a8.y, lo01_1, lo23_1, hi01_1, hi23_1);
                const int off = lm * SA_STRIDE + (qk2 << 3);
                *reinterpret_cast<uint4*>(dst_lo + off) = make_uint4(
                    mul_f16x2(lo01_0, as_f16x2), mul_f16x2(lo23_0, as_f16x2),
                    mul_f16x2(lo01_1, as_f16x2), mul_f16x2(lo23_1, as_f16x2)
                );
                *reinterpret_cast<uint4*>(dst_hi + off) = make_uint4(
                    mul_f16x2(hi01_0, as_f16x2), mul_f16x2(hi23_0, as_f16x2),
                    mul_f16x2(hi01_1, as_f16x2), mul_f16x2(hi23_1, as_f16x2)
                );
            }
        };

        load_a_slab(0, 0);
        __syncthreads();

        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        uint32_t ws_even = 0u, ws_odd = 0u;
        const bool row0_live = (g < live_m);
        [[maybe_unused]] const bool row8_live = (TILE > 8) && (g + 8 < live_m);

        for (int ks = 0; ks < K_STEPS; ++ks) {
            const int cur_buf = ks & 1;
            const int nxt_buf = cur_buf ^ 1;
            const int ss = ks >> 1;

            uint4 nxt_wa, nxt_wb;
            uint32_t nxt_ws_lo = cur_ws_lo;
            uint8_t  nxt_ws_hi = cur_ws_hi;
            if (ks + 1 < K_STEPS) {
                const int nxt_ks = ks + 1;
                const uint8_t* nxt_ss = panel_base + static_cast<int64_t>(nxt_ks >> 1) * W13_SS_BYTES;
                const int ks_off_b = (nxt_ks & 1) * 4096;
                nxt_wa = ld_cs_u4(nxt_ss + ks_off_b + w_off);
                nxt_wb = ld_cs_u4(nxt_ss + ks_off_b + w_off + 512);
                if ((nxt_ks & 1) == 0) {
                    nxt_ws_lo = ld_cs_u32(nxt_ss + ws_lo_off);
                    nxt_ws_hi = ld_cs_u8(nxt_ss + ws_hi_off);
                }
            }

            if ((ks & 1) == 0) {
                decode_ws5_w13_swar(cur_ws_lo, static_cast<uint32_t>(cur_ws_hi), base4, ws_even, ws_odd);
                if (__builtin_expect(has_patches, 0)) {
                    const int kg_lo = ss * 32 + lt * 8;
                    const int kg_hi = kg_lo + 8;
                    for (int32_t pi = p_cur; pi < p_end; ++pi) {
                        const uint32_t rec = PATCHES[pi];
                        const int p_kg = static_cast<int>(rec >> 16);
                        const int p_g  = static_cast<int>((rec >> 8) & 0xFFu);
                        if (p_g == g && p_kg >= kg_lo && p_kg < kg_hi) {
                            patch_ws5_w13_swar(ws_even, ws_odd, p_kg - kg_lo, rec);
                        }
                    }
                }
            }

            if (ks + 1 < K_STEPS) load_a_slab(ks + 1, nxt_buf);
            if (!is_last_tile && ks == K_STEPS - 2) {
                const int nidx = idx + stride;
                const int nt = nidx / N_PANELS;
                const int nnb = nidx - nt * N_PANELS;
                const int nnb32 = (BN == 16) ? (nnb >> 1) : nnb;
                const int nwp   = (BN == 16) ? (((nnb & 1) << 1) + warp) : warp;
                const int64_t ne = static_cast<int64_t>(Tiles[nt]);
                const uint8_t* nxt_pbase = SLABS + (ne * N_PANELS32 + nnb32) * static_cast<int64_t>(9 * W13_SS_BYTES);
                const int nw_off = nwp * 1024 + lane * 16;
                if ((lane & 7) == 0) {
                    prefetch_l2(nxt_pbase + nw_off);
                    prefetch_l2(nxt_pbase + nw_off + 512);
                }
                if (lane == 0) prefetch_l2(nxt_pbase + 8192 + nwp * 128);
                if (tid == 0)  prefetch_l2(nxt_pbase + 8704);
            }

            const int src0 = (g << 2) | ((ks & 1) * 2);
            const int src1 = src0 | 1;
            const uint32_t e0 = __shfl_sync(0xFFFFFFFFu, ws_even, src0);
            const uint32_t o0 = __shfl_sync(0xFFFFFFFFu, ws_odd,  src0);
            const uint32_t e1 = __shfl_sync(0xFFFFFFFFu, ws_even, src1);
            const uint32_t o1 = __shfl_sync(0xFFFFFFFFu, ws_odd,  src1);
            const uint32_t p01 = (lt < 2) ? e0 : o0;
            const uint32_t p23 = (lt < 2) ? e1 : o1;

            const uint16_t* sA_lo = sA[cur_buf][0] + x_off0;
            const uint16_t* sA_hi = sA[cur_buf][1] + x_off0;

            uint32_t h00, h01, h02, h03;
            uint32_t h10, h11, h12, h13;
            uint32_t h20, h21, h22, h23;
            uint32_t h30, h31, h32, h33;

            #define LO_C1(Q, WA, WB, PSRC, SHIFT, H0, H1, H2, H3) do { \
                const uint16_t e4_pair = static_cast<uint16_t>((PSRC) >> (SHIFT)); \
                const uint32_t sc_f16x2 = cvt_e4m3x2_to_f16x2(e4_pair); \
                const uint32_t sc0 = splat_lo_f16(sc_f16x2); \
                const uint32_t sc1 = splat_hi_f16(sc_f16x2); \
                const int q_off = (Q) * 32; \
                const uint2 a0_01 = row0_live ? *reinterpret_cast<const uint2*>(sA_lo + q_off)      : make_uint2(0u, 0u); \
                const uint2 a2_01 = row0_live ? *reinterpret_cast<const uint2*>(sA_lo + q_off + 16) : make_uint2(0u, 0u); \
                uint2 a1_01 = make_uint2(0u, 0u), a3_01 = make_uint2(0u, 0u); \
                if constexpr (TILE > 8) { \
                    if (row8_live) { \
                        a1_01 = *reinterpret_cast<const uint2*>(sA_lo + 8 * SA_STRIDE + q_off); \
                        a3_01 = *reinterpret_cast<const uint2*>(sA_lo + 8 * SA_STRIDE + q_off + 16); \
                    } \
                } \
                uint32_t lo01, lo23, hi01, hi23; \
                unpack_e2m1x8((WA), lo01, lo23, hi01, hi23); \
                (H0) = mul_f16x2(hi01, sc0); (H1) = mul_f16x2(hi23, sc1); \
                mma_m16n8k16_f16(acc, a0_01.x, a1_01.x, a2_01.x, a3_01.x, mul_f16x2(lo01, sc0), mul_f16x2(lo23, sc1)); \
                unpack_e2m1x8((WB), lo01, lo23, hi01, hi23); \
                (H2) = mul_f16x2(hi01, sc0); (H3) = mul_f16x2(hi23, sc1); \
                mma_m16n8k16_f16(acc, a0_01.y, a1_01.y, a2_01.y, a3_01.y, mul_f16x2(lo01, sc0), mul_f16x2(lo23, sc1)); \
            } while (0)

            LO_C1(0, cur_wa.x, cur_wa.y, p01,  0, h00, h01, h02, h03);
            LO_C1(1, cur_wa.z, cur_wa.w, p01, 16, h10, h11, h12, h13);
            LO_C1(2, cur_wb.x, cur_wb.y, p23,  0, h20, h21, h22, h23);
            LO_C1(3, cur_wb.z, cur_wb.w, p23, 16, h30, h31, h32, h33);
            #undef LO_C1

            #define HI_C1(Q, H0, H1, H2, H3) do { \
                const int q_off = (Q) * 32; \
                const uint2 a0_01 = row0_live ? *reinterpret_cast<const uint2*>(sA_hi + q_off)      : make_uint2(0u, 0u); \
                const uint2 a2_01 = row0_live ? *reinterpret_cast<const uint2*>(sA_hi + q_off + 16) : make_uint2(0u, 0u); \
                uint2 a1_01 = make_uint2(0u, 0u), a3_01 = make_uint2(0u, 0u); \
                if constexpr (TILE > 8) { \
                    if (row8_live) { \
                        a1_01 = *reinterpret_cast<const uint2*>(sA_hi + 8 * SA_STRIDE + q_off); \
                        a3_01 = *reinterpret_cast<const uint2*>(sA_hi + 8 * SA_STRIDE + q_off + 16); \
                    } \
                } \
                mma_m16n8k16_f16(acc, a0_01.x, a1_01.x, a2_01.x, a3_01.x, (H0), (H1)); \
                mma_m16n8k16_f16(acc, a0_01.y, a1_01.y, a2_01.y, a3_01.y, (H2), (H3)); \
            } while (0)

            HI_C1(0, h00, h01, h02, h03);
            HI_C1(1, h10, h11, h12, h13);
            HI_C1(2, h20, h21, h22, h23);
            HI_C1(3, h30, h31, h32, h33);
            #undef HI_C1

            if constexpr (TRIG_DEP) {
                if (is_last_tile && ks == K_STEPS - 2 && !trig_fired) {
                    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
                    trig_fired = true;
                }
            }

            if (ks + 1 < K_STEPS) {
                cur_wa = nxt_wa;
                cur_wb = nxt_wb;
                cur_ws_lo = nxt_ws_lo;
                cur_ws_hi = nxt_ws_hi;
                __syncthreads();
            }
        }

        const float scale = Ones[e];
        const int r0 = row_base + g;
        const int r1 = row_base + g + 8;
        const float rs0 = row0_live ? RS[r0] : 0.0f;
        const float o0 = mul_rn(mul_rn(acc[0], rs0), scale);
        const float o1 = mul_rn(mul_rn(acc[1], rs0), scale);
        float o2 = 0.0f, o3 = 0.0f;
        if constexpr (TILE > 8) {
            const float rs1 = row8_live ? RS[r1] : 0.0f;
            o2 = mul_rn(mul_rn(acc[2], rs1), scale);
            o3 = mul_rn(mul_rn(acc[3], rs1), scale);
        }

        if constexpr (FUSE_SITU) {
            constexpr int INTER = 288;
            const int c_warp = nb * (BN >> 1) + warp * 4;
            const int compact_base = TileCompactStart[t];
            const int c0 = compact_base + g;
            const int c1 = compact_base + g + 8;

            uint16_t y0_b = 0u;
            if (row0_live) y0_b = situ_pair_bf16_exact(pack_bf16x2(o0, o1), __ldg(Gammas + r0));
            const uint32_t p0 = static_cast<uint32_t>(y0_b)
                | (__shfl_xor_sync(0xFFFFFFFFu, static_cast<uint32_t>(y0_b), 1) << 16);
            const uint32_t p1 = __shfl_xor_sync(0xFFFFFFFFu, p0, 2);
            if (lt == 0 && row0_live) {
                if constexpr (QDEQ_MODE >= 2) {
                    st_cg_u2(Out + static_cast<int64_t>(c0) * INTER + c_warp, make_uint2(p0, p1));
                } else {
                    *reinterpret_cast<uint2*>(Out + static_cast<int64_t>(c0) * INTER + c_warp) = make_uint2(p0, p1);
                }
            }

            uint16_t y1_b = 0u;
            if constexpr (TILE > 8) {
                if (row8_live) y1_b = situ_pair_bf16_exact(pack_bf16x2(o2, o3), __ldg(Gammas + r1));
                const uint32_t q0 = static_cast<uint32_t>(y1_b)
                    | (__shfl_xor_sync(0xFFFFFFFFu, static_cast<uint32_t>(y1_b), 1) << 16);
                const uint32_t q1 = __shfl_xor_sync(0xFFFFFFFFu, q0, 2);
                if (lt == 0 && row8_live) {
                    if constexpr (QDEQ_MODE >= 2) {
                        st_cg_u2(Out + static_cast<int64_t>(c1) * INTER + c_warp, make_uint2(q0, q1));
                    } else {
                        *reinterpret_cast<uint2*>(Out + static_cast<int64_t>(c1) * INTER + c_warp) = make_uint2(q0, q1);
                    }
                }
            }

            if constexpr (QDEQ_MODE == 0) {
                float a0 = row0_live ? fabsf(bf16_bits_to_f32(y0_b)) : 0.0f;
                a0 = fmaxf(a0, __shfl_xor_sync(0xFFFFFFFFu, a0, 1));
                a0 = fmaxf(a0, __shfl_xor_sync(0xFFFFFFFFu, a0, 2));
                if (lt == 0) sPanelAmax[warp][g] = a0;
                if constexpr (TILE > 8) {
                    float a1 = row8_live ? fabsf(bf16_bits_to_f32(y1_b)) : 0.0f;
                    a1 = fmaxf(a1, __shfl_xor_sync(0xFFFFFFFFu, a1, 1));
                    a1 = fmaxf(a1, __shfl_xor_sync(0xFFFFFFFFu, a1, 2));
                    if (lt == 0) sPanelAmax[warp][g + 8] = a1;
                }
                __syncthreads();
                if (tid < live_m) {
                    float rmax = fmaxf(sPanelAmax[0][tid], sPanelAmax[1][tid]);
                    if constexpr (WARPS == 4) {
                        rmax = fmaxf(rmax, fmaxf(sPanelAmax[2][tid], sPanelAmax[3][tid]));
                    }
                    if (rmax > 0.0f) {
                        atomicMax(&RowAmaxW2[compact_base + tid], __float_as_uint(rmax));
                    }
                }
            } else if constexpr (QDEQ_MODE >= 2) {
                __threadfence();
                __syncthreads();
                if (tid == 0) {
                    const uint32_t prev = atom_acq_rel_gpu_add(&TileQdeqDone[t], 1u);
                    if (prev == static_cast<uint32_t>(N_PANELS - 1)) {
                        st_rel_gpu_u32(&TileQdeqDone[t], 0u);
                        sTileLastQdeq = 1;
                    } else {
                        sTileLastQdeq = 0;
                    }
                }
                __syncthreads();
                if (sTileLastQdeq) {
                    run_tile_situ_qdeq_inplace_v10<(QDEQ_MODE == 3), WARPS>(
                        Out, RowAmaxW2, compact_base, live_m, warp, lane
                    );
                }
            }
        } else {
            const int n0 = nb * BN + warp * 8 + 2 * lt;
            if (row0_live) {
                *reinterpret_cast<uint32_t*>(Out + static_cast<int64_t>(r0) * 576 + n0) = pack_bf16x2(o0, o1);
            }
            if constexpr (TILE > 8) {
                if (row8_live) {
                    *reinterpret_cast<uint32_t*>(Out + static_cast<int64_t>(r1) * 576 + n0) = pack_bf16x2(o2, o3);
                }
            }
        }

        if (!is_last_tile) __syncthreads();
    }

    if constexpr (TRIG_DEP) {
        if (!trig_fired) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }
}

// =========================================================================
// Kernel 1.5 (C=1 v9): Single-Pass In-Place L2-Hot SiTU -> QDeqA FP16
// =========================================================================
template <bool PDL_WAIT, bool TRIG_DEP, bool QUANT_FMA>
__global__ __launch_bounds__(128, 6)
void nvfp4_situ_quant_dequant_c1_v9_kernel(
    const __nv_bfloat16* __restrict__ SiTU_In,
    uint16_t* __restrict__ QDeqOut,
    uint32_t* __restrict__ RowAmaxW2,
    const int32_t* __restrict__ Tiles,
    const int32_t* __restrict__ TileCompactStart,
    int MAX_CTAS,
    int TILE,
    int max_compact_rows
) {
    if constexpr (PDL_WAIT) asm volatile("griddepcontrol.wait;" ::: "memory");
    if constexpr (TRIG_DEP) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

    const int n_active = Tiles[2 * MAX_CTAS];
    if (n_active <= 0) return;

    const int last_t = n_active - 1;
    int last_lm = Tiles[MAX_CTAS + last_t] - last_t * TILE;
    if (last_lm > TILE) last_lm = TILE;
    if (last_lm < 0) last_lm = 0;
    int r_live = TileCompactStart[last_t] + last_lm;
    if (r_live > max_compact_rows) r_live = max_compact_rows;
    if (r_live <= 0) return;

    const int warp_in_cta = threadIdx.x >> 5;
    const int lane        = threadIdx.x & 31;
    const int warp_global = static_cast<int>(blockIdx.x) * 4 + warp_in_cta;
    const int warp_stride = static_cast<int>(gridDim.x) * 4;

    for (int r = warp_global; r < r_live; r += warp_stride) {
        uint4 v0 = make_uint4(0u, 0u, 0u, 0u);
        uint4 v1 = make_uint4(0u, 0u, 0u, 0u);
        float blk_amax = 0.0f;

        if (lane < 18) {
            const uint4* bp = reinterpret_cast<const uint4*>(
                SiTU_In + static_cast<int64_t>(r) * 288 + lane * 16
            );
            v0 = ld_ca_u4(bp);
            v1 = ld_ca_u4(bp + 1);
            const uint32_t w[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                float a = bf16_bits_to_f32(static_cast<uint16_t>(w[j] & 0xFFFFu));
                float b = bf16_bits_to_f32(static_cast<uint16_t>(w[j] >> 16));
                blk_amax = fmaxf(blk_amax, fmaxf(fabsf(a), fabsf(b)));
            }
        }
        __syncwarp();

        float row_amax = blk_amax;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            row_amax = fmaxf(row_amax, __shfl_xor_sync(0xFFFFFFFFu, row_amax, off));
        }

        float enc_s = 0.0f, dec_s = 0.0f, out_s = 0.0f;
        if (lane == 0) {
            enc_s = fminf(div_rn(1536.0f, row_amax), FP32_MAX_VAL);
            dec_s = div_rn(1.0f, enc_s);
            out_s = mul_rn(row_amax, INV_1536);
            RowAmaxW2[r] = __float_as_uint(row_amax);
        }
        enc_s = __shfl_sync(0xFFFFFFFFu, enc_s, 0);
        dec_s = __shfl_sync(0xFFFFFFFFu, dec_s, 0);
        out_s = __shfl_sync(0xFFFFFFFFu, out_s, 0);

        if (lane < 18) {
            uint4 lo4, hi4;
            quant_dequant_nvfp4_16_bmax<QUANT_FMA>(v0, v1, blk_amax, enc_s, dec_s, out_s, lo4, hi4);
            uint16_t* dst_row = QDeqOut + static_cast<int64_t>(r) * 288;
            *reinterpret_cast<uint4*>(dst_row + lane * 8)       = lo4;
            *reinterpret_cast<uint4*>(dst_row + 144 + lane * 8) = hi4;
        }
    }
}

// =========================================================================
// Kernel 2 (C=1 v9): NV_W2 (N=4608, K=288, BN=128, 4 warps, TILE=8|16)
//   - FUSE_QUANT_MODE == 2 (QDeqA): issues `cur_w0` + `WS5_PACK` BEFORE `gdc_wait`,
//     eliminates 36x OTF quant + zero-fill + dead LDS -> MIN_CTA=6 (288 CTAs) & 0 spills!
// =========================================================================
template <int TILE, int MIN_CTA, int PDL_WAIT_MODE, bool TRIG_DEP, int FUSE_QUANT_MODE, bool FUSE_COMBINE, bool QUANT_FMA>
__global__ __launch_bounds__(128, MIN_CTA)
void nvfp4_w2_ws5_c1_v9_kernel(
    const uint8_t* __restrict__ A,
    const uint8_t* __restrict__ AS,
    const float* __restrict__ RS,
    const uint32_t* __restrict__ RowAmaxW2,
    const float* __restrict__ WAmax,
    const float* __restrict__ WDenominator,
    const uint8_t* __restrict__ W_PACK,
    const uint32_t* __restrict__ WS5_PACK,
    const uint8_t* __restrict__ WS_BASE,
    const int32_t* __restrict__ P_OFF,
    const uint32_t* __restrict__ PATCHES,
    const float* __restrict__ Ones,
    const int32_t* __restrict__ Tiles,
    const int32_t* __restrict__ TileCompactStart,
    __nv_bfloat16* __restrict__ Out,
    const int32_t* __restrict__ Src2Compact,
    const int32_t* __restrict__ NLivePtr,
    uint32_t* __restrict__ PanelDone,
    __nv_bfloat16* __restrict__ SendBuf,
    int N, int K, int MAX_CTAS, int T, int down_row_stride
) {
    static_assert(TILE == 8 || TILE == 16, "C1 NV_W2 TILE must be 8 or 16");
    (void)N; (void)K;

    if constexpr (PDL_WAIT_MODE == 1) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }

    constexpr int WARPS = 4;
    constexpr int SUB_N = 4;
    constexpr int BN = 128;
    constexpr int H_DIM = 4608;
    constexpr int KP = 144;
    constexpr int KG = 18;
    constexpr int K_STEPS = 9;
    constexpr int M_ROWS = TILE;
    constexpr int SA_STRIDE = KP + 8;
    constexpr int N_PANELS = 36;
    constexpr bool USE_COMPACT_IN = (FUSE_QUANT_MODE != 0);

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g = lane >> 2;
    const int lt = lane & 3;
    const int ws_row_base = (warp * SUB_N) * 8 + g;

    const int n_active = Tiles[2 * MAX_CTAS];
    const int bidx = static_cast<int>(blockIdx.x);
    const int stride = static_cast<int>(gridDim.x);

    if (n_active == 0) {
        if constexpr (PDL_WAIT_MODE == 2) asm volatile("griddepcontrol.wait;" ::: "memory");
        if constexpr (FUSE_COMBINE) {
            if (bidx < N_PANELS) {
                int n_live_tok = T;
                if (NLivePtr != nullptr) {
                    int nl = *NLivePtr;
                    if (nl < 0) nl = 0;
                    if (nl < n_live_tok) n_live_tok = nl;
                }
                const int col0 = bidx * BN + (tid & 15) * 8;
                const uint4 z4 = make_uint4(0u, 0u, 0u, 0u);
                for (int m = (tid >> 4); m < n_live_tok; m += 8) {
                    *reinterpret_cast<uint4*>(SendBuf + static_cast<int64_t>(m) * H_DIM + col0) = z4;
                }
                __threadfence();
                __syncthreads();
            }
        }
        if constexpr (TRIG_DEP) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        return;
    }

    const int n_total = n_active * N_PANELS;
    if (bidx >= n_total) {
        if constexpr (PDL_WAIT_MODE == 2) asm volatile("griddepcontrol.wait;" ::: "memory");
        if constexpr (TRIG_DEP) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        return;
    }

    alignas(16) __shared__ uint16_t sA[2][M_ROWS * SA_STRIDE];
    alignas(16) __shared__ uint8_t  sWS[2304];
    alignas(16) __shared__ float    sRowEnc[(FUSE_QUANT_MODE == 1) ? M_ROWS : 1];
    alignas(16) __shared__ float    sRowDec[(FUSE_QUANT_MODE == 1) ? M_ROWS : 1];
    alignas(16) __shared__ float    sRowOut[(FUSE_QUANT_MODE == 1) ? M_ROWS : 1];
    alignas(16) __shared__ float    sRowAmax[USE_COMPACT_IN ? M_ROWS : 1];
    __shared__ int32_t sIsLastPanel;

    bool waited_pdl = false;
    bool trig_fired = false;

    const int x0_0 = g * SA_STRIDE + 2 * lt;
    const uint16_t* sA0_base = sA[0] + x0_0;
    const uint16_t* sA1_base = sA[1] + x0_0;

    auto run_tile = [&](int t, int nb, bool is_very_first, bool is_very_last,
                        int next_t, int next_nb, bool has_next_tile) {
        const int64_t e = static_cast<int64_t>(Tiles[t]);
        const int end_row = Tiles[MAX_CTAS + t];
        const int row_base = t * TILE;
        int live_m = end_row - row_base;
        if (live_m > TILE) live_m = TILE;
        if (live_m < 0) live_m = 0;

        const int64_t panel_uid = e * N_PANELS + nb;
        const int64_t warp_uid = panel_uid * WARPS + warp;
        const uint8_t* wp_base = W_PACK + warp_uid * static_cast<int64_t>(K_STEPS) * 512;

        uint4 cur_w0 = make_uint4(0u, 0u, 0u, 0u);
        if constexpr (FUSE_QUANT_MODE == 2) {
            if (live_m > 0) cur_w0 = ld_cs_u4(wp_base + lane * 16);
        } else if constexpr (PDL_WAIT_MODE == 2) {
            if (is_very_first && (lane & 7) == 0) {
                prefetch_l2(wp_base + lane * 16);
                prefetch_l2(wp_base + 512 + lane * 16);
            }
        }

        const uint32_t raw_base = static_cast<uint32_t>(WS_BASE[e]);
        const uint32_t base4 = (raw_base & 0x7Fu) * 0x01010101u;
        const bool exp_no_patch = (raw_base & 0x80u) != 0u;

        if (tid < 72) {
            const uint32_t* ws_lo = WS5_PACK + panel_uid * 360;
            const uint32_t* ws_hi = ws_lo + 288;
            const uint4 lo4 = ld_cs_u4(ws_lo + tid * 4);
            const uint32_t hi4 = ld_cs_u32(ws_hi + tid);
            const uint2 d0 = decode_ws5_contig_u2(lo4.x, hi4 & 0xFFu, base4);
            const uint2 d1 = decode_ws5_contig_u2(lo4.y, (hi4 >> 8) & 0xFFu, base4);
            const uint2 d2 = decode_ws5_contig_u2(lo4.z, (hi4 >> 16) & 0xFFu, base4);
            const uint2 d3 = decode_ws5_contig_u2(lo4.w, hi4 >> 24, base4);
            uint8_t* dst = sWS + tid * 32;
            *reinterpret_cast<uint4*>(dst)      = make_uint4(d0.x, d0.y, d1.x, d1.y);
            *reinterpret_cast<uint4*>(dst + 16) = make_uint4(d2.x, d2.y, d3.x, d3.y);
        }

        int32_t p_cur = 0, p_end = 0;
        if (!exp_no_patch) {
            p_cur = P_OFF[panel_uid];
            p_end = P_OFF[panel_uid + 1];
            if constexpr (PDL_WAIT_MODE == 2) {
                if (is_very_first && tid == 0 && p_cur < p_end) prefetch_l2(PATCHES + p_cur);
            }
        }

        if constexpr (PDL_WAIT_MODE == 2) {
            if (!waited_pdl) {
                asm volatile("griddepcontrol.wait;" ::: "memory");
                waited_pdl = true;
            }
        }
        if (live_m == 0) return;

        const int compact_tile_base = (USE_COMPACT_IN || FUSE_COMBINE) ? TileCompactStart[t] : 0;

        if constexpr (FUSE_QUANT_MODE == 1) {
            if ((lane & 7) == 0) {
                prefetch_l2(wp_base + lane * 16);
                prefetch_l2(wp_base + 512 + lane * 16);
            }
        }

        {
            if constexpr (FUSE_QUANT_MODE == 2) {
                if (tid < live_m) {
                    const float amax = __uint_as_float(ld_ca_u32(RowAmaxW2 + compact_tile_base + tid));
                    sRowAmax[tid] = amax;
                }
                const uint16_t* qdeq_in = reinterpret_cast<const uint16_t*>(A);
                const int total_blks = live_m * KG;
                for (int bq = tid; bq < total_blks; bq += 128) {
                    const int lm = bq / KG;
                    const int blk = bq - lm * KG;
                    const int crow = compact_tile_base + lm;
                    const uint16_t* src_row = qdeq_in + static_cast<int64_t>(crow) * 288;
                    const int off0 = lm * SA_STRIDE + (blk << 3);
                    *reinterpret_cast<uint4*>(sA[0] + off0) = ld_ca_u4(src_row + (blk << 3));
                    *reinterpret_cast<uint4*>(sA[1] + off0) = ld_ca_u4(src_row + 144 + (blk << 3));
                }
            } else if constexpr (FUSE_QUANT_MODE == 1) {
                if (tid < live_m) {
                    const float amax = __uint_as_float(ld_cg_u32(RowAmaxW2 + compact_tile_base + tid));
                    const float enc_s = fminf(div_rn(1536.0f, amax), FP32_MAX_VAL);
                    sRowAmax[tid] = amax;
                    sRowEnc[tid]  = enc_s;
                    sRowDec[tid]  = div_rn(1.0f, enc_s);
                    sRowOut[tid]  = mul_rn(amax, INV_1536);
                }
                __syncthreads();

                const __nv_bfloat16* situ_in = reinterpret_cast<const __nv_bfloat16*>(A);
                const int total_blks = live_m * KG;
                for (int bq = tid; bq < total_blks; bq += 128) {
                    const int lm = bq / KG;
                    const int blk = bq - lm * KG;
                    const int crow = compact_tile_base + lm;
                    const uint4* bp = reinterpret_cast<const uint4*>(situ_in + static_cast<int64_t>(crow) * 288 + blk * 16);
                    const int off0 = lm * SA_STRIDE + (blk << 3);
                    quant_dequant_nvfp4_16_to_sA<QUANT_FMA>(
                        ld_ca_u4(bp), ld_ca_u4(bp + 1),
                        sRowEnc[lm], sRowDec[lm], sRowOut[lm],
                        sA[0] + off0, sA[1] + off0
                    );
                }
            } else {
                const int total_q2 = live_m * KG;
                for (int aidx = tid; aidx < total_q2; aidx += 128) {
                    const int lm = aidx / KG;
                    const int qk2 = aidx - lm * KG;
                    const int64_t row = static_cast<int64_t>(row_base + lm);
                    const uint2 a8 = ld_ca_u2(A + row * KP + (qk2 << 3));
                    const uint8_t as_b = AS[scale_offset_decode(row, qk2, KG)];
                    const uint32_t as_f16x2 = splat_lo_f16(cvt_e4m3x2_to_f16x2(static_cast<uint16_t>(as_b)));
                    uint32_t lo01_0, lo23_0, hi01_0, hi23_0;
                    uint32_t lo01_1, lo23_1, hi01_1, hi23_1;
                    unpack_e2m1x8(a8.x, lo01_0, lo23_0, hi01_0, hi23_0);
                    unpack_e2m1x8(a8.y, lo01_1, lo23_1, hi01_1, hi23_1);
                    const int off = lm * SA_STRIDE + (qk2 << 3);
                    *reinterpret_cast<uint4*>(sA[0] + off) = make_uint4(
                        mul_f16x2(lo01_0, as_f16x2), mul_f16x2(lo23_0, as_f16x2),
                        mul_f16x2(lo01_1, as_f16x2), mul_f16x2(lo23_1, as_f16x2)
                    );
                    *reinterpret_cast<uint4*>(sA[1] + off) = make_uint4(
                        mul_f16x2(hi01_0, as_f16x2), mul_f16x2(hi23_0, as_f16x2),
                        mul_f16x2(hi01_1, as_f16x2), mul_f16x2(hi23_1, as_f16x2)
                    );
                }
            }
        }

        if constexpr (FUSE_QUANT_MODE != 2) {
            cur_w0 = ld_cs_u4(wp_base + lane * 16);
        }
        __syncthreads();

        if (__builtin_expect(p_cur < p_end, 0)) {
            for (int32_t pi = p_cur + tid; pi < p_end; pi += 128) {
                const uint32_t rec = PATCHES[pi];
                sWS[rec >> 8] = static_cast<uint8_t>(rec & 0xFFu);
            }
            __syncthreads();
        }

        float acc[SUB_N][4];
        #pragma unroll
        for (int sub = 0; sub < SUB_N; ++sub) {
            acc[sub][0] = 0.0f; acc[sub][1] = 0.0f;
            acc[sub][2] = 0.0f; acc[sub][3] = 0.0f;
        }

        const bool row0_live = (g < live_m);
        [[maybe_unused]] const bool row8_live = (TILE > 8) && (g + 8 < live_m);

        #pragma unroll
        for (int ks = 0; ks < K_STEPS; ++ks) {
            uint4 nxt_w;
            if (ks + 1 < K_STEPS) {
                nxt_w = ld_cs_u4(wp_base + static_cast<int64_t>(ks + 1) * 512 + lane * 16);
            }
            if (has_next_tile && ks == K_STEPS - 2) {
                const int64_t ne = static_cast<int64_t>(Tiles[next_t]);
                const int64_t npuid = ne * N_PANELS + next_nb;
                const uint8_t* nwp = W_PACK + (npuid * WARPS + warp) * static_cast<int64_t>(K_STEPS) * 512;
                if ((lane & 7) == 0) {
                    prefetch_l2(nwp + lane * 16);
                    prefetch_l2(nwp + 512 + lane * 16);
                }
                if (tid < 12) prefetch_l2(WS5_PACK + npuid * 360 + tid * 32);
            }

            const int ks_off = ks * 16;
            const uint32_t w_sub[4] = {cur_w0.x, cur_w0.y, cur_w0.z, cur_w0.w};

            #pragma unroll
            for (int sub = 0; sub < SUB_N; ++sub) {
                const int s_idx = (ws_row_base + sub * 8) * 18 + ks * 2;
                const uint16_t ws_pair = *reinterpret_cast<const uint16_t*>(sWS + s_idx);
                const uint32_t ws_f16x2 = cvt_e4m3x2_to_f16x2(ws_pair);
                const uint32_t sc0 = splat_lo_f16(ws_f16x2);
                const uint32_t sc1 = splat_hi_f16(ws_f16x2);

                uint32_t lo01, lo23, hi01, hi23;
                unpack_e2m1x8(w_sub[sub], lo01, lo23, hi01, hi23);
                const uint32_t bl0 = mul_f16x2(lo01, sc0), bl1 = mul_f16x2(lo23, sc1);
                const uint32_t bh0 = mul_f16x2(hi01, sc0), bh1 = mul_f16x2(hi23, sc1);

                {
                    const uint32_t a0 = row0_live ? *reinterpret_cast<const uint32_t*>(sA0_base + ks_off)     : 0u;
                    const uint32_t a2 = row0_live ? *reinterpret_cast<const uint32_t*>(sA0_base + ks_off + 8) : 0u;
                    uint32_t a1 = 0u, a3 = 0u;
                    if constexpr (TILE > 8) {
                        if (row8_live) {
                            a1 = *reinterpret_cast<const uint32_t*>(sA0_base + 8 * SA_STRIDE + ks_off);
                            a3 = *reinterpret_cast<const uint32_t*>(sA0_base + 8 * SA_STRIDE + ks_off + 8);
                        }
                    }
                    mma_m16n8k16_f16(acc[sub], a0, a1, a2, a3, bl0, bl1);
                }
                {
                    const uint32_t a0 = row0_live ? *reinterpret_cast<const uint32_t*>(sA1_base + ks_off)     : 0u;
                    const uint32_t a2 = row0_live ? *reinterpret_cast<const uint32_t*>(sA1_base + ks_off + 8) : 0u;
                    uint32_t a1 = 0u, a3 = 0u;
                    if constexpr (TILE > 8) {
                        if (row8_live) {
                            a1 = *reinterpret_cast<const uint32_t*>(sA1_base + 8 * SA_STRIDE + ks_off);
                            a3 = *reinterpret_cast<const uint32_t*>(sA1_base + 8 * SA_STRIDE + ks_off + 8);
                        }
                    }
                    mma_m16n8k16_f16(acc[sub], a0, a1, a2, a3, bh0, bh1);
                }
            }

            if constexpr (TRIG_DEP) {
                if (is_very_last && ks == K_STEPS - 2 && !trig_fired) {
                    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
                    trig_fired = true;
                }
            }
            cur_w0 = nxt_w;
        }

        const float scale = Ones[e];
        const float w_amax_e  = USE_COMPACT_IN ? __ldg(WAmax + e) : 0.0f;
        const float rcp_denom = USE_COMPACT_IN ? rcp_rn(mul_rn(1536.0f, __ldg(WDenominator))) : 1.0f;

        auto emit_row_w2 = [&](int lm, int acc_pair_idx) {
            const bool active = (lm < live_m);
            float row_scale = 0.0f;
            int64_t r = 0;
            if (active) {
                r = static_cast<int64_t>(row_base + lm);
                const float rs = USE_COMPACT_IN ? mul_rn(mul_rn(sRowAmax[lm], w_amax_e), rcp_denom) : RS[r];
                row_scale = mul_rn(rs, scale);
            }

            if constexpr (FUSE_COMBINE) {
                const int64_t crow = active ? static_cast<int64_t>(compact_tile_base + lm) : 0;
                __nv_bfloat16* row_ptr = Out + (static_cast<int64_t>(nb) * down_row_stride + crow) * BN + warp * 32 + lt * 8;
                uint32_t pk[4];
                #pragma unroll
                for (int sub = 0; sub < SUB_N; ++sub) {
                    pk[sub] = active ? pack_bf16x2(
                        mul_rn(acc[sub][acc_pair_idx + 0], row_scale),
                        mul_rn(acc[sub][acc_pair_idx + 1], row_scale)
                    ) : 0u;
                }
                const uint4 v4 = transpose_4x4_bf16x2_u4(pk[0], pk[1], pk[2], pk[3], lt);
                if (active) st_cg_u4(row_ptr, v4);
            } else {
                if (!active) return;
                const int col_base = nb * BN + warp * 32 + 2 * lt;
                #pragma unroll
                for (int sub = 0; sub < SUB_N; ++sub) {
                    *reinterpret_cast<uint32_t*>(Out + r * H_DIM + col_base + sub * 8) = pack_bf16x2(
                        mul_rn(acc[sub][acc_pair_idx + 0], row_scale),
                        mul_rn(acc[sub][acc_pair_idx + 1], row_scale)
                    );
                }
            }
        };

        emit_row_w2(g, 0);
        if constexpr (TILE > 8) emit_row_w2(g + 8, 2);
    };

    auto combine_panel_to_sendbuf_u4 = [&](int nb) {
        int n_live_tok = T;
        if (NLivePtr != nullptr) {
            int nl = *NLivePtr;
            if (nl < 0) nl = 0;
            if (nl < n_live_tok) n_live_tok = nl;
        }
        const __nv_bfloat16* down_panel_buf = Out + static_cast<int64_t>(nb) * down_row_stride * BN;
        const int col_in_panel = (tid & 15) * 8;
        const int col_global = nb * BN + col_in_panel;
        const uint4 z4 = make_uint4(0u, 0u, 0u, 0u);

        for (int m = (tid >> 4); m < n_live_tok; m += 8) {
            const int32_t* s2c = Src2Compact + static_cast<int64_t>(m) * 8;
            const int32_t r0 = __ldg(s2c + 0), r1 = __ldg(s2c + 1);
            const int32_t r2 = __ldg(s2c + 2), r3 = __ldg(s2c + 3);
            const uint4 v0 = (r0 >= 0) ? ld_cg_u4(down_panel_buf + static_cast<int64_t>(r0) * BN + col_in_panel) : z4;
            const uint4 v1 = (r1 >= 0) ? ld_cg_u4(down_panel_buf + static_cast<int64_t>(r1) * BN + col_in_panel) : z4;
            const uint4 v2 = (r2 >= 0) ? ld_cg_u4(down_panel_buf + static_cast<int64_t>(r2) * BN + col_in_panel) : z4;
            const uint4 v3 = (r3 >= 0) ? ld_cg_u4(down_panel_buf + static_cast<int64_t>(r3) * BN + col_in_panel) : z4;
            const int32_t r4 = __ldg(s2c + 4), r5 = __ldg(s2c + 5);
            const int32_t r6 = __ldg(s2c + 6), r7 = __ldg(s2c + 7);
            const uint4 v4 = (r4 >= 0) ? ld_cg_u4(down_panel_buf + static_cast<int64_t>(r4) * BN + col_in_panel) : z4;

            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
            float a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;
            auto acc_u4 = [&](uint4 v) {
                a0 = add_rn(a0, bf16_bits_to_f32(static_cast<uint16_t>(v.x & 0xFFFFu)));
                a1 = add_rn(a1, bf16_bits_to_f32(static_cast<uint16_t>(v.x >> 16)));
                a2 = add_rn(a2, bf16_bits_to_f32(static_cast<uint16_t>(v.y & 0xFFFFu)));
                a3 = add_rn(a3, bf16_bits_to_f32(static_cast<uint16_t>(v.y >> 16)));
                a4 = add_rn(a4, bf16_bits_to_f32(static_cast<uint16_t>(v.z & 0xFFFFu)));
                a5 = add_rn(a5, bf16_bits_to_f32(static_cast<uint16_t>(v.z >> 16)));
                a6 = add_rn(a6, bf16_bits_to_f32(static_cast<uint16_t>(v.w & 0xFFFFu)));
                a7 = add_rn(a7, bf16_bits_to_f32(static_cast<uint16_t>(v.w >> 16)));
            };
            acc_u4(v0);
            const uint4 v5 = (r5 >= 0) ? ld_cg_u4(down_panel_buf + static_cast<int64_t>(r5) * BN + col_in_panel) : z4;
            acc_u4(v1);
            const uint4 v6 = (r6 >= 0) ? ld_cg_u4(down_panel_buf + static_cast<int64_t>(r6) * BN + col_in_panel) : z4;
            acc_u4(v2);
            const uint4 v7 = (r7 >= 0) ? ld_cg_u4(down_panel_buf + static_cast<int64_t>(r7) * BN + col_in_panel) : z4;
            acc_u4(v3); acc_u4(v4); acc_u4(v5); acc_u4(v6); acc_u4(v7);

            *reinterpret_cast<uint4*>(SendBuf + static_cast<int64_t>(m) * H_DIM + col_global) = make_uint4(
                pack_bf16x2(a0, a1), pack_bf16x2(a2, a3),
                pack_bf16x2(a4, a5), pack_bf16x2(a6, a7)
            );
        }
    };

    if constexpr (FUSE_COMBINE) {
        // Wave-V12 Zero-Imbalance Panel-Major Schedule:
        // Eliminates v4..v10's 2.0x static `t_rank` work imbalance at C=1 (U=25..36) & C=2 (U=41..72),
        // lifting load balance from 58.3% -> 87.5..100% while keeping <=128 KiB active L2 footprint!
        uint32_t cur_nb_tiles = 0u;
        for (int idx = bidx; idx < n_total; idx += stride) {
            const int nb = idx / n_active;
            const int t  = idx - nb * n_active;
            const bool is_very_first = (idx == bidx);
            const int nidx = idx + stride;
            const bool has_next = (nidx < n_total);
            const int next_nb = has_next ? (nidx / n_active) : 0;
            const int next_t  = has_next ? (nidx - next_nb * n_active) : 0;

            run_tile(t, nb, is_very_first, !has_next, next_t, next_nb, has_next);
            cur_nb_tiles += 1u;

            if (!has_next || next_nb != nb) {
                __threadfence();
                __syncthreads();
                if (tid == 0) {
                    if (cur_nb_tiles == static_cast<uint32_t>(n_active)) {
                        sIsLastPanel = 1;
                    } else {
                        const uint32_t prev = atom_acq_rel_gpu_add(&PanelDone[nb], cur_nb_tiles);
                        if (prev + cur_nb_tiles == static_cast<uint32_t>(n_active)) {
                            st_rel_gpu_u32(&PanelDone[nb], 0u);
                            sIsLastPanel = 1;
                        } else {
                            sIsLastPanel = 0;
                        }
                    }
                }
                __syncthreads();
                if (sIsLastPanel) {
                    combine_panel_to_sendbuf_u4(nb);
                    __threadfence();
                    __syncthreads();
                } else if (has_next) {
                    __syncthreads();
                }
                cur_nb_tiles = 0u;
            } else {
                __syncthreads();
            }
        }
    } else {
        for (int idx = bidx; idx < n_total; idx += stride) {
            const int t  = idx / N_PANELS;
            const int nb = idx - t * N_PANELS;
            const bool is_very_first = (idx == bidx);
            const bool is_very_last  = (idx + stride >= n_total);
            const int nidx = idx + stride;
            const int next_t  = is_very_last ? 0 : (nidx / N_PANELS);
            const int next_nb = is_very_last ? 0 : (nidx - next_t * N_PANELS);
            run_tile(t, nb, is_very_first, is_very_last, next_t, next_nb, !is_very_last);
            if (!is_very_last) __syncthreads();
        }
    }

    if constexpr (TRIG_DEP) {
        if (!trig_fired) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }
}

template <typename KernelT, typename... Args>
static inline cudaError_t launch_nvfp4_c1(
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

template <int BN, int TILE, int MIN_CTA, bool ZS, bool FS, int QM, int GM>
static int dispatch_w13_c1_v10(
    dim3 grid, dim3 block, cudaStream_t stream, int pdl_wait_mode, int trig_dep,
    const uint8_t* A, const uint8_t* AS, const float* RS,
    const uint8_t* SLABS, const uint8_t* WS_BASE,
    const int32_t* P_OFF, const uint32_t* PATCHES,
    const float* Ones, const int32_t* Tiles,
    const int32_t* TileMask, const int32_t* TileCompactStart,
    const float* Gammas, uint32_t* RowAmaxW2, __nv_bfloat16* Out,
    uint32_t* TileQdeqDone,
    int N, int K, int MAX_CTAS, int g0, int g1, int g2
) {
    const bool use_pdl = (pdl_wait_mode != 0) || (trig_dep != 0);
    cudaError_t err = cudaSuccess;
    #define RUN13(PW, TR) err = launch_nvfp4_c1( \
        nvfp4_w13_ws5_c1_v9_kernel<BN, TILE, MIN_CTA, PW, TR, ZS, FS, QM, GM>, \
        grid, block, stream, use_pdl, \
        A, AS, RS, SLABS, WS_BASE, P_OFF, PATCHES, Ones, Tiles, \
        TileMask, TileCompactStart, Gammas, RowAmaxW2, Out, TileQdeqDone, N, K, MAX_CTAS, g0, g1, g2)
    if      (pdl_wait_mode == 2 && trig_dep)  { RUN13(2, true);  }
    else if (pdl_wait_mode == 1 && trig_dep)  { RUN13(1, true);  }
    else if (pdl_wait_mode == 0 && trig_dep)  { RUN13(0, true);  }
    else if (pdl_wait_mode == 2 && !trig_dep) { RUN13(2, false); }
    else if (pdl_wait_mode == 1 && !trig_dep) { RUN13(1, false); }
    else                                      { RUN13(0, false); }
    #undef RUN13
    return static_cast<int>(err);
}

template <int TILE, int MIN_CTA, int FQ_MODE, bool FC, bool QF>
static int dispatch_w2_c1_v9(
    dim3 grid, dim3 block, cudaStream_t stream, int pdl_wait_mode, int trig_dep,
    const uint8_t* A, const uint8_t* AS, const float* RS,
    const uint32_t* RowAmaxW2, const float* WAmax, const float* WDenominator,
    const uint8_t* W_PACK, const uint32_t* WS5_PACK, const uint8_t* WS_BASE,
    const int32_t* P_OFF, const uint32_t* PATCHES,
    const float* Ones, const int32_t* Tiles, const int32_t* TileCompactStart,
    __nv_bfloat16* Out, const int32_t* Src2Compact, const int32_t* NLivePtr,
    uint32_t* PanelDone, __nv_bfloat16* SendBuf,
    int N, int K, int MAX_CTAS, int T, int down_row_stride
) {
    const bool use_pdl = (pdl_wait_mode != 0) || (trig_dep != 0);
    cudaError_t err = cudaSuccess;
    #define RUN2(PW, TR) err = launch_nvfp4_c1( \
        nvfp4_w2_ws5_c1_v9_kernel<TILE, MIN_CTA, PW, TR, FQ_MODE, FC, QF>, \
        grid, block, stream, use_pdl, \
        A, AS, RS, RowAmaxW2, WAmax, WDenominator, \
        W_PACK, WS5_PACK, WS_BASE, P_OFF, PATCHES, Ones, Tiles, TileCompactStart, \
        Out, Src2Compact, NLivePtr, PanelDone, SendBuf, N, K, MAX_CTAS, T, down_row_stride)
    if      (pdl_wait_mode == 2 && trig_dep)  { RUN2(2, true);  }
    else if (pdl_wait_mode == 1 && trig_dep)  { RUN2(1, true);  }
    else if (pdl_wait_mode == 0 && trig_dep)  { RUN2(0, true);  }
    else if (pdl_wait_mode == 2 && !trig_dep) { RUN2(2, false); }
    else if (pdl_wait_mode == 1 && !trig_dep) { RUN2(1, false); }
    else                                      { RUN2(0, false); }
    #undef RUN2
    return static_cast<int>(err);
}

}  // namespace

// =========================================================================
// Extern "C" C=1 Wave-V10 Exports (100% Drop-In Superset of c1_v9/v6/v4 ABI!)
// =========================================================================

extern "C" int nvfp4_ws5_c1_version() {
    return 12;
}

extern "C" int nvfp4_ws5_c1_min_cta() {
    return NVFP4_C1_W13_T8_BN32_MIN_CTA * 10 + NVFP4_C1_W2_T8_MIN_CTA;
}

extern "C" int nvfp4_w13_ws5_c1_v10_fwd(
    const void* A, const void* AS, const void* RS,
    const void* SLABS, const void* WS_BASE,
    const void* P_OFF, const void* PATCHES,
    const void* Ones, const void* Tiles,
    const void* TileMask, const void* TileCompactStart,
    const void* Gammas, void* RowAmaxW2, void* Out, void* TileQdeqDone,
    int N, int K, int MAX_CTAS, int TILE,
    int zero_scatter, int fuse_situ, int skip_amax,
    int bn_mode, int u_thresh_or_margin,
    int pdl_wait_mode, int trig_dep,
    int fuse_qdeq, int quant_fma, uintptr_t stream_ptr
) {
    if (N != 576 || K != 4608 || (TILE != 8 && TILE != 16)) return -2;
    if (MAX_CTAS <= 0) return 0;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    const bool zs = (zero_scatter != 0) && (TileMask != nullptr);
    const bool fs = (fuse_situ != 0);
    const bool fq = fs && (fuse_qdeq != 0) && (TileQdeqDone != nullptr) && (RowAmaxW2 != nullptr);
    const int qm = fq ? ((quant_fma != 0) ? 3 : 2) : ((fs && skip_amax != 0) ? 1 : 0);
    const int eff_pdl_wait = (zs && pdl_wait_mode == 2) ? 1 : pdl_wait_mode;

    const int cap16 = (TILE == 8) ? (48 * NVFP4_C1_W13_T8_BN16_MIN_CTA) : 240;
    const int cap32 = (TILE == 8) ? (48 * NVFP4_C1_W13_T8_BN32_MIN_CTA) : 192;

    auto launch_one = [&](int bn, int gm, int g0, int g1, int g2, int pw, int td) -> int {
        const int panels = (bn == 16) ? 36 : 18;
        const int total_tiles = panels * MAX_CTAS;
        const int cap = (bn == 16) ? cap16 : cap32;
        const int gx = (gm != 0) ? cap : ((total_tiles < cap) ? total_tiles : cap);
        dim3 grid(gx, 1, 1);
        dim3 block((bn / 8) * 32, 1, 1);
        #define D13(BNV, TV, MC, Z, F, QMV, GMV) return dispatch_w13_c1_v10<BNV, TV, MC, Z, F, QMV, GMV>( \
            grid, block, stream, pw, td, \
            static_cast<const uint8_t*>(A), static_cast<const uint8_t*>(AS), \
            static_cast<const float*>(RS), static_cast<const uint8_t*>(SLABS), \
            static_cast<const uint8_t*>(WS_BASE), static_cast<const int32_t*>(P_OFF), \
            static_cast<const uint32_t*>(PATCHES), static_cast<const float*>(Ones), \
            static_cast<const int32_t*>(Tiles), static_cast<const int32_t*>(TileMask), \
            static_cast<const int32_t*>(TileCompactStart), static_cast<const float*>(Gammas), \
            static_cast<uint32_t*>(RowAmaxW2), static_cast<__nv_bfloat16*>(Out), \
            static_cast<uint32_t*>(TileQdeqDone), \
            N, K, MAX_CTAS, g0, g1, g2)
        #define D13_QM(BNV, TV, MC, GMV) do { \
            if (qm == 3) { \
                if (zs) { D13(BNV, TV, MC, true,  true, 3, GMV); } \
                else    { D13(BNV, TV, MC, false, true, 3, GMV); } \
            } else if (qm == 2) { \
                if (zs) { D13(BNV, TV, MC, true,  true, 2, GMV); } \
                else    { D13(BNV, TV, MC, false, true, 2, GMV); } \
            } else if (qm == 1) { \
                if (zs) { D13(BNV, TV, MC, true,  true, 1, GMV); } \
                else    { D13(BNV, TV, MC, false, true, 1, GMV); } \
            } else { \
                if      (zs && fs)  { D13(BNV, TV, MC, true,  true,  0, GMV); } \
                else if (zs && !fs) { D13(BNV, TV, MC, true,  false, 0, GMV); } \
                else if (!zs && fs) { D13(BNV, TV, MC, false, true,  0, GMV); } \
                else                { D13(BNV, TV, MC, false, false, 0, GMV); } \
            } \
        } while (0)
        if (bn == 16) {
            if (TILE == 8) {
                if      (gm == 0) { D13_QM(16, 8, NVFP4_C1_W13_T8_BN16_MIN_CTA, 0); }
                else if (gm == 1) { D13_QM(16, 8, NVFP4_C1_W13_T8_BN16_MIN_CTA, 1); }
                else              { D13_QM(16, 8, NVFP4_C1_W13_T8_BN16_MIN_CTA, 2); }
            } else {
                if      (gm == 0) { D13_QM(16, 16, 5, 0); }
                else if (gm == 1) { D13_QM(16, 16, 5, 1); }
                else              { D13_QM(16, 16, 5, 2); }
            }
        } else {
            if (TILE == 8) {
                if      (gm == 0) { D13_QM(32, 8, NVFP4_C1_W13_T8_BN32_MIN_CTA, 0); }
                else if (gm == 1) { D13_QM(32, 8, NVFP4_C1_W13_T8_BN32_MIN_CTA, 1); }
                else              { D13_QM(32, 8, NVFP4_C1_W13_T8_BN32_MIN_CTA, 2); }
            } else {
                if      (gm == 0) { D13_QM(32, 16, 4, 0); }
                else if (gm == 1) { D13_QM(32, 16, 4, 1); }
                else              { D13_QM(32, 16, 4, 2); }
            }
        }
        #undef D13_QM
        #undef D13
        return -1;
    };

    if (bn_mode == 0) return launch_one(32, 0, 0, 0, 0, eff_pdl_wait, trig_dep);
    if (bn_mode == 1) return launch_one(16, 0, 0, 0, 0, eff_pdl_wait, trig_dep);
    const int chain_pdl = (eff_pdl_wait != 0 || trig_dep != 0) ? 1 : 0;
    if (bn_mode == 3) {
        const int margin = (u_thresh_or_margin >= 0) ? u_thresh_or_margin : 3;
        int rc = launch_one(16, 2, cap16, cap32, margin, eff_pdl_wait, chain_pdl);
        if (rc != 0) return rc;
        return launch_one(32, 2, cap16, cap32, margin, chain_pdl, trig_dep);
    }
    const int eff_thresh = (u_thresh_or_margin > 0) ? u_thresh_or_margin : 37;
    int rc = launch_one(16, 1, 1, eff_thresh, 0, eff_pdl_wait, chain_pdl);
    if (rc != 0) return rc;
    return launch_one(32, 1, eff_thresh + 1, 1000000, 0, chain_pdl, trig_dep);
}

extern "C" int nvfp4_w13_ws5_c1_v9_fwd(
    const void* A, const void* AS, const void* RS,
    const void* SLABS, const void* WS_BASE,
    const void* P_OFF, const void* PATCHES,
    const void* Ones, const void* Tiles,
    const void* TileMask, const void* TileCompactStart,
    const void* Gammas, void* RowAmaxW2, void* Out,
    int N, int K, int MAX_CTAS, int TILE,
    int zero_scatter, int fuse_situ, int skip_amax,
    int bn_mode, int u_thresh_or_margin,
    int pdl_wait_mode, int trig_dep, uintptr_t stream_ptr
) {
    return nvfp4_w13_ws5_c1_v10_fwd(
        A, AS, RS, SLABS, WS_BASE, P_OFF, PATCHES, Ones, Tiles,
        TileMask, TileCompactStart, Gammas, RowAmaxW2, Out, nullptr,
        N, K, MAX_CTAS, TILE, zero_scatter, fuse_situ, skip_amax,
        bn_mode, u_thresh_or_margin, pdl_wait_mode, trig_dep, 0, 1, stream_ptr
    );
}

extern "C" int nvfp4_situ_quant_dequant_c1_v9_fwd(
    const void* SiTU_In,
    void* QDeqOut,
    void* RowAmaxW2,
    const void* Tiles,
    const void* TileCompactStart,
    int MAX_CTAS,
    int TILE,
    int max_compact_rows,
    int pdl_wait,
    int trig_dep,
    int quant_fma,
    uintptr_t stream_ptr
) {
    if (MAX_CTAS <= 0 || max_compact_rows <= 0) return 0;
    if (TILE != 8 && TILE != 16) return -2;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    const int need_ctas = (max_compact_rows + 3) / 4;
    const int grid_x = (need_ctas < 48) ? need_ctas : 48;
    dim3 grid(grid_x, 1, 1);
    dim3 block(128, 1, 1);
    const bool pw = (pdl_wait != 0);
    const bool td = (trig_dep != 0);
    const bool qf = (quant_fma != 0);
    const bool use_pdl = pw || td;
    cudaError_t err = cudaSuccess;
    #define RUN_Q(PW, TD, QF) err = launch_nvfp4_c1( \
        nvfp4_situ_quant_dequant_c1_v9_kernel<PW, TD, QF>, \
        grid, block, stream, use_pdl, \
        static_cast<const __nv_bfloat16*>(SiTU_In), static_cast<uint16_t*>(QDeqOut), \
        static_cast<uint32_t*>(RowAmaxW2), static_cast<const int32_t*>(Tiles), \
        static_cast<const int32_t*>(TileCompactStart), MAX_CTAS, TILE, max_compact_rows)
    if (qf) {
        if      (pw && td)  { RUN_Q(true,  true,  true); }
        else if (pw && !td) { RUN_Q(true,  false, true); }
        else if (!pw && td) { RUN_Q(false, true,  true); }
        else                { RUN_Q(false, false, true); }
    } else {
        if      (pw && td)  { RUN_Q(true,  true,  false); }
        else if (pw && !td) { RUN_Q(true,  false, false); }
        else if (!pw && td) { RUN_Q(false, true,  false); }
        else                { RUN_Q(false, false, false); }
    }
    #undef RUN_Q
    return static_cast<int>(err);
}

extern "C" int nvfp4_w2_ws5_c1_v9_fwd(
    const void* A, const void* AS, const void* RS,
    const void* RowAmaxW2, const void* WAmax, const void* WDenominator,
    const void* W_PACK, const void* WS5_PACK, const void* WS_BASE,
    const void* P_OFF, const void* PATCHES,
    const void* Ones, const void* Tiles, const void* TileCompactStart,
    void* Out, const void* Src2Compact, const void* NLivePtr,
    void* PanelDone, void* SendBuf,
    int N, int K, int MAX_CTAS, int TILE, int T, int down_row_stride,
    int fuse_quant_mode, int fuse_combine,
    int pdl_wait_mode, int trig_dep, int quant_fma, uintptr_t stream_ptr
) {
    if (N != 4608 || K != 288 || (TILE != 8 && TILE != 16)) return -2;
    if (fuse_quant_mode < 0 || fuse_quant_mode > 2) return -3;
    const bool fc = (fuse_combine != 0);
    const int total_tiles = 36 * MAX_CTAS;
    if (total_tiles <= 0 && !fc) return 0;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    const int eff_down_stride = (down_row_stride > 0) ? down_row_stride : (T * 8);
    const int persist_cap = (TILE == 8) ? (48 * NVFP4_C1_W2_T8_MIN_CTA) : 240;
    int grid_x = (total_tiles < persist_cap) ? total_tiles : persist_cap;
    if (fc && grid_x < 36) grid_x = 36;
    dim3 grid(grid_x, 1, 1);
    dim3 block(128, 1, 1);

    const bool qf = (quant_fma != 0);
    #define C2(TV, MC, QM, C, QF) return dispatch_w2_c1_v9<TV, MC, QM, C, QF>( \
        grid, block, stream, pdl_wait_mode, trig_dep, \
        static_cast<const uint8_t*>(A), static_cast<const uint8_t*>(AS), \
        static_cast<const float*>(RS), static_cast<const uint32_t*>(RowAmaxW2), \
        static_cast<const float*>(WAmax), static_cast<const float*>(WDenominator), \
        static_cast<const uint8_t*>(W_PACK), static_cast<const uint32_t*>(WS5_PACK), \
        static_cast<const uint8_t*>(WS_BASE), static_cast<const int32_t*>(P_OFF), \
        static_cast<const uint32_t*>(PATCHES), static_cast<const float*>(Ones), \
        static_cast<const int32_t*>(Tiles), static_cast<const int32_t*>(TileCompactStart), \
        static_cast<__nv_bfloat16*>(Out), static_cast<const int32_t*>(Src2Compact), \
        static_cast<const int32_t*>(NLivePtr), static_cast<uint32_t*>(PanelDone), \
        static_cast<__nv_bfloat16*>(SendBuf), N, K, MAX_CTAS, T, eff_down_stride)

    #define DISPATCH_C1_W2(TV, MC) do { \
        if      (fuse_quant_mode == 2 && fc)        { C2(TV, MC, 2, true,  false); } \
        else if (fuse_quant_mode == 2 && !fc)       { C2(TV, MC, 2, false, false); } \
        else if (fuse_quant_mode == 1 && fc && qf)  { C2(TV, MC, 1, true,  true);  } \
        else if (fuse_quant_mode == 1 && !fc && qf) { C2(TV, MC, 1, false, true);  } \
        else if (fuse_quant_mode == 1 && fc)        { C2(TV, MC, 1, true,  false); } \
        else if (fuse_quant_mode == 1 && !fc)       { C2(TV, MC, 1, false, false); } \
        else if (fc)                                { C2(TV, MC, 0, true,  false); } \
        else                                        { C2(TV, MC, 0, false, false); } \
    } while (0)
    if (TILE == 8) { DISPATCH_C1_W2(8, NVFP4_C1_W2_T8_MIN_CTA); }
    else           { DISPATCH_C1_W2(16, 5); }
    #undef DISPATCH_C1_W2
    #undef C2
    return -1;
}

// Drop-in v4/v6 backward-compatible ABI wrappers
extern "C" int nvfp4_w13_ws5_c1_v4_fwd(
    const void* A, const void* AS, const void* RS,
    const void* SLABS, const void* WS_BASE,
    const void* P_OFF, const void* PATCHES,
    const void* Ones, const void* Tiles,
    const void* TileMask, const void* TileCompactStart,
    const void* Gammas, void* RowAmaxW2, void* Out,
    int N, int K, int MAX_CTAS, int TILE,
    int zero_scatter, int fuse_situ,
    int bn_mode, int u_thresh,
    int pdl_wait_mode, int trig_dep, uintptr_t stream_ptr
) {
    return nvfp4_w13_ws5_c1_v9_fwd(
        A, AS, RS, SLABS, WS_BASE, P_OFF, PATCHES, Ones, Tiles,
        TileMask, TileCompactStart, Gammas, RowAmaxW2, Out,
        N, K, MAX_CTAS, TILE, zero_scatter, fuse_situ, 0,
        bn_mode, u_thresh, pdl_wait_mode, trig_dep, stream_ptr
    );
}

extern "C" int nvfp4_w2_ws5_c1_v4_fwd(
    const void* A, const void* AS, const void* RS,
    const void* RowAmaxW2, const void* WAmax, const void* WDenominator,
    const void* W_PACK, const void* WS5_PACK, const void* WS_BASE,
    const void* P_OFF, const void* PATCHES,
    const void* Ones, const void* Tiles, const void* TileCompactStart,
    void* Out, const void* Src2Compact, const void* NLivePtr,
    void* PanelDone, void* SendBuf,
    int N, int K, int MAX_CTAS, int TILE, int T, int down_row_stride,
    int fuse_quant, int fuse_combine,
    int pdl_wait_mode, int trig_dep, uintptr_t stream_ptr
) {
    return nvfp4_w2_ws5_c1_v9_fwd(
        A, AS, RS, RowAmaxW2, WAmax, WDenominator,
        W_PACK, WS5_PACK, WS_BASE, P_OFF, PATCHES, Ones, Tiles, TileCompactStart,
        Out, Src2Compact, NLivePtr, PanelDone, SendBuf,
        N, K, MAX_CTAS, TILE, T, down_row_stride,
        fuse_quant, fuse_combine, pdl_wait_mode, trig_dep, 1, stream_ptr
    );
}
