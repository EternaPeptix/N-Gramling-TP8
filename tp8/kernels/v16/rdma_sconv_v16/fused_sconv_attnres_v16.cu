// SPDX-License-Identifier: Apache-2.0
// Wave-V16 SM121a (GB10) 0-ULP Bit-Exact Fused ShortConv (W=4) + AttnRes + RMSNorm (C=1..8, T=1..64)
//   + Standalone Upgraded ShortConv V16 (T=1..64) & Dual-Regime Fused AttnRes + RMSNorm V16
// (`fused_sconv_attnres_v16.cu` -> `libfused_sconv_attnres_v16.so`)
//
// 100% ABI-compatible 17-symbol drop-in superset of V14/V9/V7/V4:
//   - `/rdma/libfused_sconv_attnres_v14.so`         (all 13 V14/C48/C1/C32 symbols preserved verbatim)
//   - PLUS `fused_sconv_ar_v16_version`             (returns 16)
//   - PLUS `sconv_exact_v16_fwd`                    (T=1..64 standalone ShortConv with T==1 full CW prefetch)
//   - PLUS `fused_attnres_rmsnorm_v16_{init_smem, fwd}` (Dual-Regime: Regime A T*nq<=256 256-thr SMEM Stack
//     + Regime B T*nq>256 128-thr BLOCK_D=1024 SMEM-pinned Stack [<=90 KiB for N_STACK=1..10] + L1-hot
//     Last/Addend, NLivePtr, normalize_values in {0,1}, #206-safe PDL -> QKVR / Router, -48 launches/step!)
//
// Wave-V16 Upgrades (ALL 100% literal 0-ULP `nd == 0` / `1dcfb306`):
//   1. `C=1 (M=8)` Unlock (`INKLING_FUSED_SCONV_AR_C48_MIN_T=1`, T=1..64):
//      - Fixed dead-row `ConvOut` NaN/garbage bug at `t >= nl` (strictly post-`gdc_wait_v14()`).
//      - Full 576-line (36.864 KiB) `CW` L2 prefetch when `T == 1` (vs half in V14).
//      - L2-caching writeback on `ConvOut` for `T <= 64` (keeps 72..576 KiB `ConvOut` L2-hot for next sublayer).
//      - In-place `NQ == 1` Phase-3 `lx0/lx1` dot-product reuse (-16 live regs in Phase 3 -> 0 B spill!).
//   2. Dual-Regime `fused_attnres_rmsnorm_v16_fwd` (Regime A `T*nq <= 256` + Regime B `T*nq > 256`):
//      - Eliminates -48 standalone RMSNorm launches + -48 intermediate `[M, 4608]` BF16 round-trips/step
//        across post-Attn AttnRes->RMSNorm->Router and post-MoE AttnRes->RMSNorm->QKVR, with `#206`-safe
//        PDL trigger directly into `QKVR`/`Router` and `N_STACK = 1..10` safe SMEM (< 99 KiB on GB10).
//
// Build on GB10 (SM121a — MUST pass `-gencode arch=compute_121a,code=sm_121a --fmad=false`,
// OMIT `--use_fast_math` & `--ftz=true`, hard-fail on any stack frame > 0 or spills > 0):
//   /usr/local/cuda/bin/nvcc -O3 -std=c++17 -gencode arch=compute_121a,code=sm_121a \
//       --fmad=false --shared -Xcompiler -fPIC --ptxas-options=-v \
//       fused_sconv_attnres_v16.cu -o libfused_sconv_attnres_v16.so

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace {

static __device__ __forceinline__ float rsqrt_approx(float x) {
    float y;
    asm volatile("rsqrt.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

static __device__ __forceinline__ float tldevice_exp_sub(float a, float b) {
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

static __device__ __forceinline__ void divmod_cache(
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

static __device__ __forceinline__ uint4 ld_global_cs_u4(const void* p) {
    uint4 v;
    asm volatile("ld.global.cs.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

static __device__ __forceinline__ uint32_t ld_global_cs_u32(const void* p) {
    uint32_t v;
    asm volatile("ld.global.cs.u32 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
}

static __device__ __forceinline__ uint4 ld_global_ca_u4(const void* p) {
    uint4 v;
    asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

static __device__ __forceinline__ uint32_t ld_global_ca_u32(const void* p) {
    uint32_t v;
    asm volatile("ld.global.u32 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
}

static __device__ __forceinline__ uint4 ld_const_u4(const void* p) {
    uint4 v;
    asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
        : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
        : "l"(p));
    return v;
}

static __device__ __forceinline__ uint32_t ld_const_u32(const void* p) {
    uint32_t v;
    asm("ld.global.nc.u32 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
}

static __device__ __forceinline__ float4 ld_const_f4(const float* p) {
    float4 v;
    asm("ld.global.nc.v4.f32 {%0,%1,%2,%3}, [%4];"
        : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
        : "l"(p));
    return v;
}

static __device__ __forceinline__ float2 ld_const_f2(const float* p) {
    float2 v;
    asm("ld.global.nc.v2.f32 {%0,%1}, [%2];"
        : "=f"(v.x), "=f"(v.y)
        : "l"(p));
    return v;
}

static __device__ __forceinline__ void st_global_cs_u4(void* p, uint4 v) {
    asm volatile("st.global.cs.v4.u32 [%0], {%1,%2,%3,%4};"
                 :: "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}

static __device__ __forceinline__ void st_global_cs_u32(void* p, uint32_t v) {
    asm volatile("st.global.cs.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}

static __device__ __forceinline__ void prefetch_l2(const void* p) {
    asm volatile("prefetch.global.L2 [%0];" :: "l"(p));
}

static __device__ __forceinline__ void gdc_wait_v14() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

static __device__ __forceinline__ void gdc_launch_dependents_v14() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

static __device__ __forceinline__ void unpack_bf16x8(uint4 raw, float out[8]) {
    const __nv_bfloat162* h2 = reinterpret_cast<const __nv_bfloat162*>(&raw);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        float2 f = __bfloat1622float2(h2[i]);
        out[2 * i]     = f.x;
        out[2 * i + 1] = f.y;
    }
}

static __device__ __forceinline__ void unpack_bf16x2(uint32_t raw, float out[2]) {
    float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw));
    out[0] = f.x;
    out[1] = f.y;
}

static __device__ __forceinline__ uint4 pack_f32x8_to_bf16_u4(const float in[8]) {
    uint4 raw;
    __nv_bfloat162* h2 = reinterpret_cast<__nv_bfloat162*>(&raw);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        h2[i] = __halves2bfloat162(__float2bfloat16_rn(in[2 * i]), __float2bfloat16_rn(in[2 * i + 1]));
    }
    return raw;
}

static __device__ __forceinline__ uint32_t pack_f32x2_to_bf16_u32(const float in[2]) {
    __nv_bfloat162 h = __halves2bfloat162(__float2bfloat16_rn(in[0]), __float2bfloat16_rn(in[1]));
    return *reinterpret_cast<const uint32_t*>(&h);
}

static __device__ __forceinline__ void load_f32x8_const(const float* p, float out[8]) {
    float4 v0 = ld_const_f4(p);
    float4 v1 = ld_const_f4(p + 4);
    out[0] = v0.x; out[1] = v0.y; out[2] = v0.z; out[3] = v0.w;
    out[4] = v1.x; out[5] = v1.y; out[6] = v1.z; out[7] = v1.w;
}

// Low-register pairwise W=4 ShortConv for 2 channels (exact 0-ULP IEEE-754 FMA order).
static __device__ __forceinline__ uint32_t sconv_w4_pair_exact(
    uint32_t raw_cur, uint32_t raw_t0, uint32_t raw_t1, uint32_t raw_t2,
    uint4 cw_u4,
    bool tv0, bool tv1, bool tv2, bool lag0_ok, float scale
) {
    float2 fc  = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw_cur));
    float2 ft0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw_t0));
    float2 ft1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw_t1));
    float2 ft2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw_t2));
    const __nv_bfloat162* wh = reinterpret_cast<const __nv_bfloat162*>(&cw_u4);
    float2 wa0 = __bfloat1622float2(wh[0]);
    float2 wa1 = __bfloat1622float2(wh[1]);
    float2 wb0 = __bfloat1622float2(wh[2]);
    float2 wb1 = __bfloat1622float2(wh[3]);

    float a0 = 0.0f;
    a0 = fmaf(tv0     ? ft0.x : 0.0f, wa0.x, a0);
    a0 = fmaf(tv1     ? ft1.x : 0.0f, wa0.y, a0);
    a0 = fmaf(tv2     ? ft2.x : 0.0f, wa1.x, a0);
    a0 = fmaf(lag0_ok ? fc.x  : 0.0f, wa1.y, a0);
    float y0 = __fmul_rn(__fadd_rn(a0, fc.x), scale);

    float a1 = 0.0f;
    a1 = fmaf(tv0     ? ft0.y : 0.0f, wb0.x, a1);
    a1 = fmaf(tv1     ? ft1.y : 0.0f, wb0.y, a1);
    a1 = fmaf(tv2     ? ft2.y : 0.0f, wb1.x, a1);
    a1 = fmaf(lag0_ok ? fc.y  : 0.0f, wb1.y, a1);
    float y1 = __fmul_rn(__fadd_rn(a1, fc.y), scale);

    __nv_bfloat162 out_h = __halves2bfloat162(__float2bfloat16_rn(y0), __float2bfloat16_rn(y1));
    return *reinterpret_cast<const uint32_t*>(&out_h);
}

// Low-register W=4 ShortConv for 8 consecutive channels (issues all loads upfront, unpacks 2 ch at a time).
static __device__ __forceinline__ uint4 sconv_w4_seg8_exact(
    uint4 raw_cur, uint4 raw_t0, uint4 raw_t1, uint4 raw_t2,
    const __nv_bfloat16* cw_base,
    bool tv0, bool tv1, bool tv2, bool lag0_ok, float scale
) {
    uint4 cw0 = ld_const_u4(cw_base);
    uint4 cw1 = ld_const_u4(cw_base + 8);
    uint4 cw2 = ld_const_u4(cw_base + 16);
    uint4 cw3 = ld_const_u4(cw_base + 24);
    uint4 out;
    out.x = sconv_w4_pair_exact(raw_cur.x, raw_t0.x, raw_t1.x, raw_t2.x, cw0, tv0, tv1, tv2, lag0_ok, scale);
    out.y = sconv_w4_pair_exact(raw_cur.y, raw_t0.y, raw_t1.y, raw_t2.y, cw1, tv0, tv1, tv2, lag0_ok, scale);
    out.z = sconv_w4_pair_exact(raw_cur.z, raw_t0.z, raw_t1.z, raw_t2.z, cw2, tv0, tv1, tv2, lag0_ok, scale);
    out.w = sconv_w4_pair_exact(raw_cur.w, raw_t0.w, raw_t1.w, raw_t2.w, cw3, tv0, tv1, tv2, lag0_ok, scale);
    return out;
}

template <bool HAS_CONV_RES>
static __device__ __forceinline__ uint4 apply_conv_addend_seg8_exact(
    uint4 raw_y, uint4 raw_ca, uint4 raw_cr
) {
    float ca[8], yf[8];
    unpack_bf16x8(raw_ca, ca);
    unpack_bf16x8(raw_y, yf);
    if constexpr (HAS_CONV_RES) {
        float cr[8];
        unpack_bf16x8(raw_cr, cr);
        #pragma unroll
        for (int i = 0; i < 8; ++i) ca[i] = __fadd_rn(cr[i], ca[i]);
    } else {
        (void)raw_cr;
    }
    float acc[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] = __fadd_rn(ca[i], yf[i]);
    return pack_f32x8_to_bf16_u4(acc);
}

template <bool HAS_CONV_RES>
static __device__ __forceinline__ uint32_t apply_conv_addend_seg2_exact(
    uint32_t raw_y, uint32_t raw_ca, uint32_t raw_cr
) {
    float ca[2], yf[2];
    unpack_bf16x2(raw_ca, ca);
    unpack_bf16x2(raw_y, yf);
    if constexpr (HAS_CONV_RES) {
        float cr[2];
        unpack_bf16x2(raw_cr, cr);
        ca[0] = __fadd_rn(cr[0], ca[0]);
        ca[1] = __fadd_rn(cr[1], ca[1]);
    } else {
        (void)raw_cr;
    }
    float acc[2];
    acc[0] = __fadd_rn(ca[0], yf[0]);
    acc[1] = __fadd_rn(ca[1], yf[1]);
    return pack_f32x2_to_bf16_u32(acc);
}

// =========================================================================
// Exact reduction trees matching BOTH Triton `_attn_res_row_kernel` (Regime A)
// AND Triton `_attn_res_kernel` (Regime B) under live CUDA 13.0 ptxas!
// =========================================================================
static __device__ __forceinline__ float tree16_sum(const float a0[8], const float a1[8]) {
    float r0_e = __fadd_rn(__fadd_rn(a0[0], a0[2]), __fadd_rn(a0[4], a0[6]));
    float r0_o = __fadd_rn(__fadd_rn(a0[1], a0[3]), __fadd_rn(a0[5], a0[7]));
    float r1_e = __fadd_rn(__fadd_rn(a1[0], a1[2]), __fadd_rn(a1[4], a1[6]));
    float r1_o = __fadd_rn(__fadd_rn(a1[1], a1[3]), __fadd_rn(a1[5], a1[7]));
    return __fadd_rn(__fadd_rn(r0_e, r1_e), __fadd_rn(r0_o, r1_o));
}

template <int MODE>
static __device__ __forceinline__ float tree16_sum_sq(const float a0[8], const float a1[8]) {
    if constexpr (MODE == 0) {
        float r0_e = __fadd_rn(__fadd_rn(__fmul_rn(a0[0], a0[0]), __fmul_rn(a0[2], a0[2])),
                               __fadd_rn(__fmul_rn(a0[4], a0[4]), __fmul_rn(a0[6], a0[6])));
        float r0_o = __fadd_rn(__fadd_rn(__fmul_rn(a0[1], a0[1]), __fmul_rn(a0[3], a0[3])),
                               __fadd_rn(__fmul_rn(a0[5], a0[5]), __fmul_rn(a0[7], a0[7])));
        float r1_e = __fadd_rn(__fadd_rn(__fmul_rn(a1[0], a1[0]), __fmul_rn(a1[2], a1[2])),
                               __fadd_rn(__fmul_rn(a1[4], a1[4]), __fmul_rn(a1[6], a1[6])));
        float r1_o = __fadd_rn(__fadd_rn(__fmul_rn(a1[1], a1[1]), __fmul_rn(a1[3], a1[3])),
                               __fadd_rn(__fmul_rn(a1[5], a1[5]), __fmul_rn(a1[7], a1[7])));
        return __fadd_rn(__fadd_rn(r0_e, r1_e), __fadd_rn(r0_o, r1_o));
    } else if constexpr (MODE == 1) {
        float r0_e = __fadd_rn(fmaf(a0[0], a0[0], __fmul_rn(a0[2], a0[2])),
                               fmaf(a0[4], a0[4], __fmul_rn(a0[6], a0[6])));
        float r0_o = __fadd_rn(fmaf(a0[1], a0[1], __fmul_rn(a0[3], a0[3])),
                               fmaf(a0[5], a0[5], __fmul_rn(a0[7], a0[7])));
        float r1_e = __fadd_rn(fmaf(a1[0], a1[0], __fmul_rn(a1[2], a1[2])),
                               fmaf(a1[4], a1[4], __fmul_rn(a1[6], a1[6])));
        float r1_o = __fadd_rn(fmaf(a1[1], a1[1], __fmul_rn(a1[3], a1[3])),
                               fmaf(a1[5], a1[5], __fmul_rn(a1[7], a1[7])));
        return __fadd_rn(__fadd_rn(r0_e, r1_e), __fadd_rn(r0_o, r1_o));
    } else {
        // MODE == 3 (N_STACK == 8 Pass 2 quirk in CUDA 13.0 ptxas)
        float r0_e = __fadd_rn(fmaf(a0[2], a0[2], __fmul_rn(a0[0], a0[0])),
                               fmaf(a0[4], a0[4], __fmul_rn(a0[6], a0[6])));
        float r0_o = __fadd_rn(fmaf(a0[3], a0[3], __fmul_rn(a0[1], a0[1])),
                               fmaf(a0[5], a0[5], __fmul_rn(a0[7], a0[7])));
        float r1_e = __fadd_rn(fmaf(a1[0], a1[0], __fmul_rn(a1[2], a1[2])),
                               fmaf(a1[4], a1[4], __fmul_rn(a1[6], a1[6])));
        float r1_o = __fadd_rn(fmaf(a1[1], a1[1], __fmul_rn(a1[3], a1[3])),
                               fmaf(a1[5], a1[5], __fmul_rn(a1[7], a1[7])));
        return __fadd_rn(__fadd_rn(r0_e, r1_e), __fadd_rn(r0_o, r1_o));
    }
}

static __device__ __forceinline__ float tree8_sum(const float v[8]) {
    float e0 = __fadd_rn(__fadd_rn(v[0], v[2]), __fadd_rn(v[4], v[6]));
    float o0 = __fadd_rn(__fadd_rn(v[1], v[3]), __fadd_rn(v[5], v[7]));
    return __fadd_rn(e0, o0);
}

static __device__ __forceinline__ float warp_sum_bfly(float v) {
    v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, 16));
    v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, 8));
    v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, 4));
    v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, 2));
    v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, 1));
    return v;
}

static __device__ __forceinline__ void warp_sum_bfly2(float& a, float& b) {
    #pragma unroll
    for (int off = 16; off >= 1; off >>= 1) {
        a = __fadd_rn(a, __shfl_xor_sync(0xFFFFFFFFu, a, off));
        b = __fadd_rn(b, __shfl_xor_sync(0xFFFFFFFFu, b, off));
    }
}

static __device__ __forceinline__ float warp8_sum_bfly(float v) {
    v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, 4));
    v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, 2));
    v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, 1));
    return v;
}

static __device__ __forceinline__ float warp4_sum_bfly(float v) {
    v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, 2));
    v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, 1));
    return v;
}

template <typename KernelT>
static inline cudaError_t ensure_smem_optin(
    KernelT k, size_t smem_bytes, bool global_inited, bool& local_inited, cudaStream_t stream
) {
    if (global_inited || local_inited || (smem_bytes + 4096u) <= 49152u) {
        return cudaSuccess;
    }
    cudaStreamCaptureStatus cap_st = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &cap_st) == cudaSuccess && cap_st != cudaStreamCaptureStatusNone) {
        return cudaErrorInvalidValue;
    }
    cudaError_t e = cudaFuncSetAttribute(
        k, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes)
    );
    if (e == cudaSuccess) local_inited = true;
    return e;
}

// =========================================================================
// KERNEL 1 (HERO C=1..8 V14): `fused_sconv_ar_c48_kernel_v14`
//   T = 1..64, N_STACK = 1..10, NQ in {1, 3}, normalize_values in {0, 1},
//   NLivePtr, TreeAnc, AuxOut, ConvRes==ConvOut aliasing, hoisted NQ=1 Queries,
//   pre-wait MixLast/ConvAdd/ConvRes/Stack overlap, and #206-safe PDL!
// =========================================================================
template <int N_STACK, int NQ, bool HAS_CONV_RES, bool HAS_CONV_ADDEND, bool HAS_MIX_LAST, bool PDL>
__global__ void __launch_bounds__(256, (N_STACK <= 5) ? 2 : 1)
fused_sconv_ar_c48_kernel_v14(
    const __nv_bfloat16* __restrict__ X,
    const __nv_bfloat16* __restrict__ CW,       // [4608, 4] contiguous
    __nv_bfloat16* __restrict__ Cache,          // [num_blocks, 1, N_CACHE, cache_D]
    __nv_bfloat16* ConvOut,                     // [T, 4608] (may alias ConvRes!)
    const __nv_bfloat16* ConvRes,               // [T, 4608] optional
    const __nv_bfloat16* __restrict__ ConvAdd,  // [T, 4608] optional
    __nv_bfloat16* __restrict__ AuxOut,         // [T, 4608] optional
    const int64_t* __restrict__ Pos,
    const int32_t* __restrict__ SeqIdx,
    const int64_t* __restrict__ Slot,
    const int32_t* __restrict__ BT,
    const int32_t* __restrict__ QStart,
    const __nv_bfloat16* __restrict__ Stack,    // [N_STACK, stride_s]
    const __nv_bfloat16* __restrict__ MixLast,  // [T, 4608] optional
    const float* __restrict__ Queries,          // [NQ, 4608]
    const __nv_bfloat16* __restrict__ NormW,    // [NQ, 4608]
    __nv_bfloat16* __restrict__ Out,            // [NQ, T, 4608]
    const int32_t* __restrict__ NLivePtr,       // optional ROWAR["buf"] [1] int32
    const int32_t* __restrict__ TreeAnc,        // optional tree_sconv_anc [T, 4] int32
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
    float norm_eps,
    int normalize_values,
    int early_pdl
) {
    constexpr int D = 4608;
    constexpr int NS = N_STACK + 1;
    constexpr float INV_D = 1.0f / static_cast<float>(D);
    constexpr int BLOCK_N = (NS <= 2) ? 2 : (NS <= 4) ? 4 : (NS <= 8) ? 8 : 16;

    const int t = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int64_t row_off = static_cast<int64_t>(t) * D;
    const int off_a0 = tid * 8;
    const int off_a1 = 2048 + tid * 8;
    const int off_b  = 4096 + tid * 2;

    // Step-start live-row check (100% safe before gdc_wait()).
    int nl = T;
    if (NLivePtr != nullptr) {
        const int raw_nl = *NLivePtr;
        if (raw_nl >= 1 && raw_nl <= T) nl = raw_nl;
    }
    if (t >= nl) {
        // #206 PDL safety: wait strictly BEFORE writing ConvOut/AuxOut/Out and BEFORE launch_dependents.
        if constexpr (PDL) {
            gdc_wait_v14();
        }
        const uint4 z4 = make_uint4(0u, 0u, 0u, 0u);
        // V16 fix: zero ConvOut on dead rows too (prevents NaN/garbage in empty_like ConvOut / stack[num_blocks]!).
        st_global_cs_u4(ConvOut + row_off + off_a0, z4);
        st_global_cs_u4(ConvOut + row_off + off_a1, z4);
        st_global_cs_u32(ConvOut + row_off + off_b, 0u);
        if (AuxOut != nullptr) {
            st_global_cs_u4(AuxOut + row_off + off_a0, z4);
            st_global_cs_u4(AuxOut + row_off + off_a1, z4);
            st_global_cs_u32(AuxOut + row_off + off_b, 0u);
        }
        #pragma unroll
        for (int q = 0; q < NQ; ++q) {
            __nv_bfloat16* out_p = Out + static_cast<int64_t>(q) * T * D + row_off;
            *reinterpret_cast<uint4*>(out_p + off_a0)   = z4;
            *reinterpret_cast<uint4*>(out_p + off_a1)   = z4;
            *reinterpret_cast<uint32_t*>(out_p + off_b) = 0u;
        }
        if constexpr (PDL) {
            __syncthreads();
            if (tid == 0) gdc_launch_dependents_v14();
        }
        return;
    }

    extern __shared__ uint4 dyn_raw_u4[];
    uint8_t*  dyn_raw   = reinterpret_cast<uint8_t*>(dyn_raw_u4);
    uint4*    sStack_a0 = dyn_raw_u4;
    uint4*    sStack_a1 = reinterpret_cast<uint4*>(dyn_raw + static_cast<size_t>(N_STACK) * 4096);
    uint32_t* sStack_b  = reinterpret_cast<uint32_t*>(dyn_raw + static_cast<size_t>(N_STACK) * 8192);

    alignas(16) __shared__ float sRedSq[NS][2][8];
    alignas(16) __shared__ float sRedDt[NQ][NS][2][8];
    alignas(16) __shared__ float sSq2[2][8];

    // ------------------------------------------------------------------
    // Phase 1 (PRE-WAIT): Hide metadata, weight prefetches, MixLast/ConvAdd/ConvRes
    // loads, AND all N_STACK Stack SMEM loads + Pass-1 reductions behind `ar_kernel_c32`!
    // ------------------------------------------------------------------
    if (t < 4 && tid < 72) {
        #pragma unroll
        for (int q = 0; q < NQ; ++q) {
            prefetch_l2(NormW + static_cast<int64_t>(q) * D + tid * 64);
        }
    }
    if (T == 1) {
        // V16 fix: when T == 1, CTA 0 prefetches ALL 576 CW lines (36.864 KiB) instead of half!
        prefetch_l2(CW + tid * 32);
        prefetch_l2(CW + (256 + tid) * 32);
        if (tid < 64) prefetch_l2(CW + (512 + tid) * 32);
    } else if (t < 2) {
        // Prefetch CW [4608, 4] BF16 (36.864 KiB = 576 lines -> 288 lines/CTA for t=0,1).
        const int cw_line_base = (t * 288) + tid;
        if (cw_line_base < 576) {
            prefetch_l2(CW + cw_line_base * 32);
        }
        if (tid < 32) {
            const int cw_line_hi = (t * 288) + 256 + tid;
            if (cw_line_hi < 576) prefetch_l2(CW + cw_line_hi * 32);
        }
    }

    const int64_t slot = Slot[t];
    const bool valid = (slot >= 0);
    const int64_t pos = Pos[t];
    const int32_t req = SeqIdx[t];
    const int32_t qstart = QStart[t];
    const bool lag0_ok = valid && (pos >= 0);

    __nv_bfloat16* cache_dst = nullptr;
    if (valid) {
        int64_t dblk, drem;
        divmod_cache(slot, N_CACHE, N_SHIFT, dblk, drem);
        cache_dst = Cache + dblk * stride_c_blk + drem * stride_c_n + stream_offset;
    }

    const __nv_bfloat16* tap_ptrs[3];
    uint32_t tap_flags = 0u;  // bits 0..2: valid[0..2], bits 3..5: is_cache[0..2]
    #pragma unroll
    for (int k = 0; k < 3; ++k) {
        const int lag = 3 - k;
        const int64_t src = pos - lag;
        const bool in_win = valid && (src >= 0);
        if (in_win) {
            tap_flags |= (1u << k);
            int row = t - lag;
            bool is_intra = (row >= qstart);
            if (TreeAnc != nullptr) {
                const int anc = TreeAnc[t * 4 + lag];
                row = anc;
                is_intra = (anc >= 0);
            }
            if (is_intra) {
                tap_ptrs[k] = X + static_cast<int64_t>(row) * stride_x;
            } else {
                int64_t lblk, rem;
                divmod_cache(src, N_CACHE, N_SHIFT, lblk, rem);
                if (lblk >= max_blocks) lblk = max_blocks - 1;
                const int64_t blk = static_cast<int64_t>(BT[static_cast<int64_t>(req) * stride_bt + lblk]);
                tap_ptrs[k] = Cache + blk * stride_c_blk + rem * stride_c_n + stream_offset;
                tap_flags |= (1u << (k + 3));
                if (tid < 72) prefetch_l2(tap_ptrs[k] + tid * 64);
            }
        } else {
            tap_ptrs[k] = X;
        }
    }

    // Pre-wait loads of prior-sublayer tensors (100% ready before predecessor `ar_kernel_c32`!).
    uint4 ml_r0 = make_uint4(0u, 0u, 0u, 0u), ml_r1 = make_uint4(0u, 0u, 0u, 0u);
    uint32_t ml_rb = 0u;
    if constexpr (HAS_MIX_LAST) {
        ml_r0 = ld_global_cs_u4(MixLast + row_off + off_a0);
        ml_r1 = ld_global_cs_u4(MixLast + row_off + off_a1);
        ml_rb = ld_global_cs_u32(MixLast + row_off + off_b);
    }

    uint4 ca_r0 = make_uint4(0u, 0u, 0u, 0u), ca_r1 = make_uint4(0u, 0u, 0u, 0u);
    uint32_t ca_rb = 0u;
    uint4 cr_r0 = make_uint4(0u, 0u, 0u, 0u), cr_r1 = make_uint4(0u, 0u, 0u, 0u);
    uint32_t cr_rb = 0u;
    if constexpr (HAS_CONV_ADDEND) {
        ca_r0 = ld_global_cs_u4(ConvAdd + row_off + off_a0);
        ca_r1 = ld_global_cs_u4(ConvAdd + row_off + off_a1);
        ca_rb = ld_global_cs_u32(ConvAdd + row_off + off_b);
        if constexpr (HAS_CONV_RES) {
            // Loaded strictly BEFORE ConvOut writeback -> 100% safe when ConvRes == ConvOut!
            cr_r0 = ld_global_cs_u4(ConvRes + row_off + off_a0);
            cr_r1 = ld_global_cs_u4(ConvRes + row_off + off_a1);
            cr_rb = ld_global_cs_u32(ConvRes + row_off + off_b);
        }
    }

    // Hoist NQ==1 Queries ONCE before N_STACK loop (cuts N_STACK=8 loads 8x -> 1x!).
    float q0_a0[8], q0_a1[8];
    float2 q0_bf = make_float2(0.0f, 0.0f);
    if constexpr (NQ == 1) {
        load_f32x8_const(Queries + off_a0, q0_a0);
        load_f32x8_const(Queries + off_a1, q0_a1);
        q0_bf = ld_const_f2(Queries + off_b);
    }

    #pragma unroll
    for (int n = 0; n < N_STACK; ++n) {
        const __nv_bfloat16* sp = Stack + static_cast<int64_t>(n) * stride_s + row_off;
        uint4 r0 = ld_global_cs_u4(sp + off_a0);
        uint4 r1 = ld_global_cs_u4(sp + off_a1);
        uint32_t rb = ld_global_cs_u32(sp + off_b);
        sStack_a0[n * 256 + tid] = r0;
        sStack_a1[n * 256 + tid] = r1;
        sStack_b[n * 256 + tid]  = rb;

        float xa0[8], xa1[8], xb[2];
        unpack_bf16x8(r0, xa0);
        unpack_bf16x8(r1, xa1);
        unpack_bf16x2(rb, xb);

        float sqa = tree16_sum_sq<0>(xa0, xa1);
        float sqb = __fadd_rn(__fmul_rn(xb[0], xb[0]), __fmul_rn(xb[1], xb[1]));
        warp_sum_bfly2(sqa, sqb);
        if (lane == 0) {
            sRedSq[n][0][warp] = sqa;
            sRedSq[n][1][warp] = sqb;
        }
        if constexpr (NQ == 1) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                xa0[i] = __fmul_rn(q0_a0[i], xa0[i]);
                xa1[i] = __fmul_rn(q0_a1[i], xa1[i]);
            }
            float dta = tree16_sum(xa0, xa1);
            float dtb = __fadd_rn(__fmul_rn(q0_bf.x, xb[0]), __fmul_rn(q0_bf.y, xb[1]));
            warp_sum_bfly2(dta, dtb);
            if (lane == 0) {
                sRedDt[0][n][0][warp] = dta;
                sRedDt[0][n][1][warp] = dtb;
            }
        } else {
            #pragma unroll
            for (int q = 0; q < NQ; ++q) {
                const float* qp = Queries + static_cast<int64_t>(q) * D;
                float qa0[8], qa1[8];
                load_f32x8_const(qp + off_a0, qa0);
                load_f32x8_const(qp + off_a1, qa1);
                float2 qbf = ld_const_f2(qp + off_b);
                float dt0[8], dt1[8];
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    dt0[i] = __fmul_rn(qa0[i], xa0[i]);
                    dt1[i] = __fmul_rn(qa1[i], xa1[i]);
                }
                float dta = tree16_sum(dt0, dt1);
                float dtb = __fadd_rn(__fmul_rn(qbf.x, xb[0]), __fmul_rn(qbf.y, xb[1]));
                warp_sum_bfly2(dta, dtb);
                if (lane == 0) {
                    sRedDt[q][n][0][warp] = dta;
                    sRedDt[q][n][1][warp] = dtb;
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // #206-Safe PDL Synchronization Point:
    //   1. ALL 256 threads complete `gdc_wait_v14()` for predecessor `ar_kernel_c32`.
    //   2. When `early_pdl != 0`, `__syncthreads()` ensures ALL 8 warps passed `gdc_wait`
    //      before thread 0 launches dependent BF11 `qkvr` / `router_bf12` weight preloads!
    // ------------------------------------------------------------------
    if constexpr (PDL) {
        gdc_wait_v14();
        if (early_pdl != 0) {
            __syncthreads();
            if (tid == 0) gdc_launch_dependents_v14();
        }
    }

    // ------------------------------------------------------------------
    // Phase 2 (POST-WAIT): Low-register ShortConv on L2-hot X -> Cache, ConvOut, AuxOut
    // ------------------------------------------------------------------
    const __nv_bfloat16* x_cur_p = X + static_cast<int64_t>(t) * stride_x;
    const bool tv0 = (tap_flags & 1u) != 0u;
    const bool tv1 = (tap_flags & 2u) != 0u;
    const bool tv2 = (tap_flags & 4u) != 0u;
    const bool tc0 = (tap_flags & 8u) != 0u;
    const bool tc1 = (tap_flags & 16u) != 0u;
    const bool tc2 = (tap_flags & 32u) != 0u;

    auto run_sconv_seg8 = [&](int off, uint4 pre_ca, uint4 pre_cr) -> uint4 {
        uint4 raw_cur = ld_global_ca_u4(x_cur_p + off);
        uint4 raw_t0 = tv0 ? (tc0 ? ld_global_cs_u4(tap_ptrs[0] + off) : ld_global_ca_u4(tap_ptrs[0] + off)) : make_uint4(0u, 0u, 0u, 0u);
        uint4 raw_t1 = tv1 ? (tc1 ? ld_global_cs_u4(tap_ptrs[1] + off) : ld_global_ca_u4(tap_ptrs[1] + off)) : make_uint4(0u, 0u, 0u, 0u);
        uint4 raw_t2 = tv2 ? (tc2 ? ld_global_cs_u4(tap_ptrs[2] + off) : ld_global_ca_u4(tap_ptrs[2] + off)) : make_uint4(0u, 0u, 0u, 0u);
        if (valid) st_global_cs_u4(cache_dst + off, raw_cur);

        uint4 raw_y = sconv_w4_seg8_exact(
            raw_cur, raw_t0, raw_t1, raw_t2, CW + off * 4, tv0, tv1, tv2, lag0_ok, scale
        );
        if constexpr (HAS_CONV_ADDEND) {
            raw_y = apply_conv_addend_seg8_exact<HAS_CONV_RES>(raw_y, pre_ca, pre_cr);
        } else {
            (void)pre_ca; (void)pre_cr;
        }
        // V16: keep ConvOut L2-hot for T<=64 (72..576 KiB << 24 MiB L2, consumed immediately next sublayer!).
        *reinterpret_cast<uint4*>(ConvOut + row_off + off) = raw_y;
        if (AuxOut != nullptr) st_global_cs_u4(AuxOut + row_off + off, raw_y);
        return raw_y;
    };

    auto run_sconv_seg2 = [&](int off, uint32_t pre_ca, uint32_t pre_cr) -> uint32_t {
        uint32_t raw_cur = ld_global_ca_u32(x_cur_p + off);
        uint32_t raw_t0 = tv0 ? (tc0 ? ld_global_cs_u32(tap_ptrs[0] + off) : ld_global_ca_u32(tap_ptrs[0] + off)) : 0u;
        uint32_t raw_t1 = tv1 ? (tc1 ? ld_global_cs_u32(tap_ptrs[1] + off) : ld_global_ca_u32(tap_ptrs[1] + off)) : 0u;
        uint32_t raw_t2 = tv2 ? (tc2 ? ld_global_cs_u32(tap_ptrs[2] + off) : ld_global_ca_u32(tap_ptrs[2] + off)) : 0u;
        if (valid) st_global_cs_u32(cache_dst + off, raw_cur);

        uint4 cw_u4 = ld_const_u4(CW + off * 4);
        uint32_t raw_y = sconv_w4_pair_exact(
            raw_cur, raw_t0, raw_t1, raw_t2, cw_u4, tv0, tv1, tv2, lag0_ok, scale
        );
        if constexpr (HAS_CONV_ADDEND) {
            raw_y = apply_conv_addend_seg2_exact<HAS_CONV_RES>(raw_y, pre_ca, pre_cr);
        } else {
            (void)pre_ca; (void)pre_cr;
        }
        *reinterpret_cast<uint32_t*>(ConvOut + row_off + off) = raw_y;
        if (AuxOut != nullptr) st_global_cs_u32(AuxOut + row_off + off, raw_y);
        return raw_y;
    };

    uint4 last_r0 = run_sconv_seg8(off_a0, ca_r0, cr_r0);
    uint4 last_r1 = run_sconv_seg8(off_a1, ca_r1, cr_r1);
    uint32_t last_rb = run_sconv_seg2(off_b, ca_rb, cr_rb);

    // ------------------------------------------------------------------
    // Phase 3: AttnRes Pass 1 on Last (+ pre-loaded MixLast) + Softmax + SMEM Pass 2 + RMSNorm
    // ------------------------------------------------------------------
    float last_f0[8], last_f1[8], last_fb[2];
    {
        float lx0[8], lx1[8], lxb[2];
        unpack_bf16x8(last_r0, lx0);
        unpack_bf16x8(last_r1, lx1);
        unpack_bf16x2(last_rb, lxb);
        if constexpr (HAS_MIX_LAST) {
            float m0[8], m1[8], mb[2];
            unpack_bf16x8(ml_r0, m0);
            unpack_bf16x8(ml_r1, m1);
            unpack_bf16x2(ml_rb, mb);
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                lx0[i] = __fadd_rn(m0[i], lx0[i]);
                lx1[i] = __fadd_rn(m1[i], lx1[i]);
                last_f0[i] = lx0[i];
                last_f1[i] = lx1[i];
            }
            lxb[0] = __fadd_rn(mb[0], lxb[0]);
            lxb[1] = __fadd_rn(mb[1], lxb[1]);
            last_fb[0] = lxb[0];
            last_fb[1] = lxb[1];
        }
#ifdef INKLING_PTXAS133_STRICT
        constexpr int P1_LAST_MODE = 0;
#else
        constexpr int P1_LAST_MODE = HAS_MIX_LAST ? 1 : 0;
#endif
        float sqa = tree16_sum_sq<P1_LAST_MODE>(lx0, lx1);
        float sqb = __fadd_rn(__fmul_rn(lxb[0], lxb[0]), __fmul_rn(lxb[1], lxb[1]));
        warp_sum_bfly2(sqa, sqb);
        if (lane == 0) {
            sRedSq[N_STACK][0][warp] = sqa;
            sRedSq[N_STACK][1][warp] = sqb;
        }
        if constexpr (NQ == 1) {
            float qa0[8], qa1[8];
            load_f32x8_const(Queries + off_a0, qa0);
            load_f32x8_const(Queries + off_a1, qa1);
            float2 qbf = ld_const_f2(Queries + off_b);
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                lx0[i] = __fmul_rn(qa0[i], lx0[i]);
                lx1[i] = __fmul_rn(qa1[i], lx1[i]);
            }
            float dta = tree16_sum(lx0, lx1);
            float dtb = __fadd_rn(__fmul_rn(qbf.x, lxb[0]), __fmul_rn(qbf.y, lxb[1]));
            warp_sum_bfly2(dta, dtb);
            if (lane == 0) {
                sRedDt[0][N_STACK][0][warp] = dta;
                sRedDt[0][N_STACK][1][warp] = dtb;
            }
        } else {
            #pragma unroll
            for (int q = 0; q < NQ; ++q) {
                const float* qp = Queries + static_cast<int64_t>(q) * D;
                float qa0[8], qa1[8];
                load_f32x8_const(qp + off_a0, qa0);
                load_f32x8_const(qp + off_a1, qa1);
                float2 qbf = ld_const_f2(qp + off_b);
                float dt0[8], dt1[8];
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    dt0[i] = __fmul_rn(qa0[i], lx0[i]);
                    dt1[i] = __fmul_rn(qa1[i], lx1[i]);
                }
                float dta = tree16_sum(dt0, dt1);
                float dtb = __fadd_rn(__fmul_rn(qbf.x, lxb[0]), __fmul_rn(qbf.y, lxb[1]));
                warp_sum_bfly2(dta, dtb);
                if (lane == 0) {
                    sRedDt[q][N_STACK][0][warp] = dta;
                    sRedDt[q][N_STACK][1][warp] = dtb;
                }
            }
        }
    }

    __syncthreads();

    const int l8 = lane & 7;
    const int half = (lane >> 3) & 1;
    float rstd_arr[NS];
    #pragma unroll
    for (int n = 0; n < NS; ++n) {
        float v = warp8_sum_bfly(sRedSq[n][half][l8]);
        float sa = __shfl_sync(0xFFFFFFFFu, v, 0);
        float sb = __shfl_sync(0xFFFFFFFFu, v, 8);
        rstd_arr[n] = rsqrt_approx(fmaf(__fadd_rn(sa, sb), INV_D, eps));
    }

    #pragma unroll
    for (int q = 0; q < NQ; ++q) {
        float my_lg = -INFINITY;
        float my_ir = 0.0f;
        #pragma unroll
        for (int n = 0; n < NS; ++n) {
            float v = warp8_sum_bfly(sRedDt[q][n][half][l8]);
            float da = __shfl_sync(0xFFFFFFFFu, v, 0);
            float db = __shfl_sync(0xFFFFFFFFu, v, 8);
            if (lane == n) {
                my_lg = __fmul_rn(__fadd_rn(da, db), rstd_arr[n]);
                my_ir = rstd_arr[n];
            }
        }
        float mx = my_lg;
        if constexpr (BLOCK_N >= 16) mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, 8));
        if constexpr (BLOCK_N >= 8)  mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, 4));
        if constexpr (BLOCK_N >= 4)  mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, 2));
        if constexpr (BLOCK_N >= 2)  mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, 1));
        float p = (lane < NS) ? tldevice_exp_sub(my_lg, mx) : 0.0f;
        float sp = p;
        if constexpr (BLOCK_N >= 16) sp = __fadd_rn(sp, __shfl_xor_sync(0xFFFFFFFFu, sp, 8));
        if constexpr (BLOCK_N >= 8)  sp = __fadd_rn(sp, __shfl_xor_sync(0xFFFFFFFFu, sp, 4));
        if constexpr (BLOCK_N >= 4)  sp = __fadd_rn(sp, __shfl_xor_sync(0xFFFFFFFFu, sp, 2));
        if constexpr (BLOCK_N >= 2)  sp = __fadd_rn(sp, __shfl_xor_sync(0xFFFFFFFFu, sp, 1));
        float inv_sum = __fdiv_rn(1.0f, __shfl_sync(0xFFFFFFFFu, sp, 0));
        if (normalize_values != 0) p = __fmul_rn(p, my_ir);
        float probs[NS];
        #pragma unroll
        for (int n = 0; n < NS; ++n) probs[n] = __shfl_sync(0xFFFFFFFFu, p, n);

        const int64_t q_off = static_cast<int64_t>(q) * D;
        uint4 nw_r0 = ld_const_u4(NormW + q_off + off_a0);
        uint4 nw_r1 = ld_const_u4(NormW + q_off + off_a1);
        uint32_t nw_rb = ld_const_u32(NormW + q_off + off_b);

        float acc0[8] = {0,0,0,0,0,0,0,0};
        float acc1[8] = {0,0,0,0,0,0,0,0};
        float accb[2] = {0,0};

        #pragma unroll
        for (int n = 0; n < N_STACK; ++n) {
            const float pn = probs[n];
            uint4 r0 = sStack_a0[n * 256 + tid];
            uint4 r1 = sStack_a1[n * 256 + tid];
            uint32_t rb = sStack_b[n * 256 + tid];
            const __nv_bfloat162* h0 = reinterpret_cast<const __nv_bfloat162*>(&r0);
            const __nv_bfloat162* h1 = reinterpret_cast<const __nv_bfloat162*>(&r1);
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                float2 f0 = __bfloat1622float2(h0[i]);
                acc0[2*i]   = fmaf(pn, f0.x, acc0[2*i]);
                acc0[2*i+1] = fmaf(pn, f0.y, acc0[2*i+1]);
                float2 f1 = __bfloat1622float2(h1[i]);
                acc1[2*i]   = fmaf(pn, f1.x, acc1[2*i]);
                acc1[2*i+1] = fmaf(pn, f1.y, acc1[2*i+1]);
            }
            float2 fb = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&rb));
            accb[0] = fmaf(pn, fb.x, accb[0]);
            accb[1] = fmaf(pn, fb.y, accb[1]);
        }
        {
            const float pn = probs[N_STACK];
            if constexpr (HAS_MIX_LAST) {
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    acc0[i] = fmaf(pn, last_f0[i], acc0[i]);
                    acc1[i] = fmaf(pn, last_f1[i], acc1[i]);
                }
                accb[0] = fmaf(pn, last_fb[0], accb[0]);
                accb[1] = fmaf(pn, last_fb[1], accb[1]);
            } else {
                const __nv_bfloat162* h0 = reinterpret_cast<const __nv_bfloat162*>(&last_r0);
                const __nv_bfloat162* h1 = reinterpret_cast<const __nv_bfloat162*>(&last_r1);
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    float2 f0 = __bfloat1622float2(h0[i]);
                    acc0[2*i]   = fmaf(pn, f0.x, acc0[2*i]);
                    acc0[2*i+1] = fmaf(pn, f0.y, acc0[2*i+1]);
                    float2 f1 = __bfloat1622float2(h1[i]);
                    acc1[2*i]   = fmaf(pn, f1.x, acc1[2*i]);
                    acc1[2*i+1] = fmaf(pn, f1.y, acc1[2*i+1]);
                }
                float2 fb = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&last_rb));
                accb[0] = fmaf(pn, fb.x, accb[0]);
                accb[1] = fmaf(pn, fb.y, accb[1]);
            }
        }

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            acc0[i] = __fmul_rn(acc0[i], inv_sum);
            acc1[i] = __fmul_rn(acc1[i], inv_sum);
        }
        accb[0] = __fmul_rn(accb[0], inv_sum);
        accb[1] = __fmul_rn(accb[1], inv_sum);

#ifdef INKLING_PTXAS133_STRICT
        constexpr int P2_MODE = 0;
#else
        constexpr int P2_MODE = (N_STACK == 8) ? 3 : (HAS_MIX_LAST ? 1 : 0);
#endif
        float sqa = tree16_sum_sq<P2_MODE>(acc0, acc1);
        float sqb = __fadd_rn(__fmul_rn(accb[0], accb[0]), __fmul_rn(accb[1], accb[1]));
        warp_sum_bfly2(sqa, sqb);
        if (lane == 0) {
            sSq2[0][warp] = sqa;
            sSq2[1][warp] = sqb;
        }
        __syncthreads();

        float v2 = warp8_sum_bfly(sSq2[half][l8]);
        float sa = __shfl_sync(0xFFFFFFFFu, v2, 0);
        float sb = __shfl_sync(0xFFFFFFFFu, v2, 8);
        float out_rstd = rsqrt_approx(__fadd_rn(__fdiv_rn(__fadd_rn(sa, sb), static_cast<float>(D)), norm_eps));

        float wa0[8], wa1[8], wb[2];
        unpack_bf16x8(nw_r0, wa0);
        unpack_bf16x8(nw_r1, wa1);
        unpack_bf16x2(nw_rb, wb);

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            acc0[i] = __fmul_rn(__fmul_rn(acc0[i], out_rstd), wa0[i]);
            acc1[i] = __fmul_rn(__fmul_rn(acc1[i], out_rstd), wa1[i]);
        }
        accb[0] = __fmul_rn(__fmul_rn(accb[0], out_rstd), wb[0]);
        accb[1] = __fmul_rn(__fmul_rn(accb[1], out_rstd), wb[1]);

        __nv_bfloat16* out_p = Out + static_cast<int64_t>(q) * T * D + row_off;
        *reinterpret_cast<uint4*>(out_p + off_a0)   = pack_f32x8_to_bf16_u4(acc0);
        *reinterpret_cast<uint4*>(out_p + off_a1)   = pack_f32x8_to_bf16_u4(acc1);
        *reinterpret_cast<uint32_t*>(out_p + off_b) = pack_f32x2_to_bf16_u32(accb);

        if constexpr (NQ > 1) {
            if (q + 1 < NQ) __syncthreads();
        }
    }

    if constexpr (PDL) {
        if (early_pdl == 0) {
            __syncthreads();
            if (tid == 0) gdc_launch_dependents_v14();
        }
    }
}

// =========================================================================
// KERNEL 2 (C=32 REGIME A, T*nq <= 256): `fused_sconv_ar_c32_row_kernel`
//   Upgraded with V14 NQ==1 Queries hoisting + low-reg pairwise ShortConv.
// =========================================================================
template <int N_STACK, int NQ, bool HAS_CONV_RES, bool HAS_CONV_ADDEND, bool HAS_MIX_LAST>
__global__ void __launch_bounds__(256, (N_STACK <= 5) ? 2 : 1)
fused_sconv_ar_c32_row_kernel(
    const __nv_bfloat16* __restrict__ X,
    const __nv_bfloat16* __restrict__ CW,
    __nv_bfloat16* __restrict__ Cache,
    __nv_bfloat16* ConvOut,
    const __nv_bfloat16* ConvRes,
    const __nv_bfloat16* __restrict__ ConvAdd,
    __nv_bfloat16* __restrict__ AuxOut,
    const int64_t* __restrict__ Pos,
    const int32_t* __restrict__ SeqIdx,
    const int64_t* __restrict__ Slot,
    const int32_t* __restrict__ BT,
    const int32_t* __restrict__ QStart,
    const __nv_bfloat16* __restrict__ Stack,
    const __nv_bfloat16* __restrict__ MixLast,
    const float* __restrict__ Queries,
    const __nv_bfloat16* __restrict__ NormW,
    __nv_bfloat16* __restrict__ Out,
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
    constexpr float INV_D = 1.0f / static_cast<float>(D);
    constexpr int BLOCK_N = (NS <= 2) ? 2 : (NS <= 4) ? 4 : (NS <= 8) ? 8 : 16;

    const int t = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int64_t row_off = static_cast<int64_t>(t) * D;
    const int off_a0 = tid * 8;
    const int off_a1 = 2048 + tid * 8;
    const int off_b  = 4096 + tid * 2;

    extern __shared__ uint4 dyn_raw_u4[];
    uint8_t*  dyn_raw   = reinterpret_cast<uint8_t*>(dyn_raw_u4);
    uint4*    sStack_a0 = dyn_raw_u4;
    uint4*    sStack_a1 = reinterpret_cast<uint4*>(dyn_raw + static_cast<size_t>(N_STACK) * 4096);
    uint32_t* sStack_b  = reinterpret_cast<uint32_t*>(dyn_raw + static_cast<size_t>(N_STACK) * 8192);

    alignas(16) __shared__ float sRedSq[NS][2][8];
    alignas(16) __shared__ float sRedDt[NQ][NS][2][8];
    alignas(16) __shared__ float sSq2[2][8];

    const int64_t slot = Slot[t];
    const bool valid = (slot >= 0);
    const int64_t pos = Pos[t];
    const int32_t req = SeqIdx[t];
    const int32_t qstart = QStart[t];
    const bool lag0_ok = valid && (pos >= 0);

    __nv_bfloat16* cache_dst = nullptr;
    if (valid) {
        int64_t dblk, drem;
        divmod_cache(slot, N_CACHE, N_SHIFT, dblk, drem);
        cache_dst = Cache + dblk * stride_c_blk + drem * stride_c_n + stream_offset;
    }

    const __nv_bfloat16* tap_ptrs[3];
    uint32_t tap_flags = 0u;
    #pragma unroll
    for (int k = 0; k < 3; ++k) {
        const int lag = 3 - k;
        const int64_t src = pos - lag;
        const bool in_win = valid && (src >= 0);
        if (in_win) {
            tap_flags |= (1u << k);
            const int row = t - lag;
            if (row >= qstart) {
                tap_ptrs[k] = X + static_cast<int64_t>(row) * stride_x;
            } else {
                int64_t lblk, rem;
                divmod_cache(src, N_CACHE, N_SHIFT, lblk, rem);
                if (lblk >= max_blocks) lblk = max_blocks - 1;
                const int64_t blk = static_cast<int64_t>(BT[static_cast<int64_t>(req) * stride_bt + lblk]);
                tap_ptrs[k] = Cache + blk * stride_c_blk + rem * stride_c_n + stream_offset;
                tap_flags |= (1u << (k + 3));
            }
        } else {
            tap_ptrs[k] = X;
        }
    }

    const __nv_bfloat16* x_cur_p = X + static_cast<int64_t>(t) * stride_x;
    const bool tv0 = (tap_flags & 1u) != 0u;
    const bool tv1 = (tap_flags & 2u) != 0u;
    const bool tv2 = (tap_flags & 4u) != 0u;
    const bool tc0 = (tap_flags & 8u) != 0u;
    const bool tc1 = (tap_flags & 16u) != 0u;
    const bool tc2 = (tap_flags & 32u) != 0u;

    auto run_seg8 = [&](int off) -> uint4 {
        uint4 raw_cur = ld_global_ca_u4(x_cur_p + off);
        uint4 raw_t0 = tv0 ? (tc0 ? ld_global_cs_u4(tap_ptrs[0] + off) : ld_global_ca_u4(tap_ptrs[0] + off)) : make_uint4(0u, 0u, 0u, 0u);
        uint4 raw_t1 = tv1 ? (tc1 ? ld_global_cs_u4(tap_ptrs[1] + off) : ld_global_ca_u4(tap_ptrs[1] + off)) : make_uint4(0u, 0u, 0u, 0u);
        uint4 raw_t2 = tv2 ? (tc2 ? ld_global_cs_u4(tap_ptrs[2] + off) : ld_global_ca_u4(tap_ptrs[2] + off)) : make_uint4(0u, 0u, 0u, 0u);
        if (valid) st_global_cs_u4(cache_dst + off, raw_cur);

        uint4 raw_y = sconv_w4_seg8_exact(
            raw_cur, raw_t0, raw_t1, raw_t2, CW + off * 4, tv0, tv1, tv2, lag0_ok, scale
        );
        if constexpr (HAS_CONV_ADDEND) {
            uint4 raw_ca = ld_global_cs_u4(ConvAdd + row_off + off);
            uint4 raw_cr = make_uint4(0u, 0u, 0u, 0u);
            if constexpr (HAS_CONV_RES) raw_cr = ld_global_cs_u4(ConvRes + row_off + off);
            raw_y = apply_conv_addend_seg8_exact<HAS_CONV_RES>(raw_y, raw_ca, raw_cr);
        }
        st_global_cs_u4(ConvOut + row_off + off, raw_y);
        if (AuxOut != nullptr) st_global_cs_u4(AuxOut + row_off + off, raw_y);
        return raw_y;
    };

    auto run_seg2 = [&](int off) -> uint32_t {
        uint32_t raw_cur = ld_global_ca_u32(x_cur_p + off);
        uint32_t raw_t0 = tv0 ? (tc0 ? ld_global_cs_u32(tap_ptrs[0] + off) : ld_global_ca_u32(tap_ptrs[0] + off)) : 0u;
        uint32_t raw_t1 = tv1 ? (tc1 ? ld_global_cs_u32(tap_ptrs[1] + off) : ld_global_ca_u32(tap_ptrs[1] + off)) : 0u;
        uint32_t raw_t2 = tv2 ? (tc2 ? ld_global_cs_u32(tap_ptrs[2] + off) : ld_global_ca_u32(tap_ptrs[2] + off)) : 0u;
        if (valid) st_global_cs_u32(cache_dst + off, raw_cur);

        uint4 cw_u4 = ld_const_u4(CW + off * 4);
        uint32_t raw_y = sconv_w4_pair_exact(
            raw_cur, raw_t0, raw_t1, raw_t2, cw_u4, tv0, tv1, tv2, lag0_ok, scale
        );
        if constexpr (HAS_CONV_ADDEND) {
            uint32_t raw_ca = ld_global_cs_u32(ConvAdd + row_off + off);
            uint32_t raw_cr = 0u;
            if constexpr (HAS_CONV_RES) raw_cr = ld_global_cs_u32(ConvRes + row_off + off);
            raw_y = apply_conv_addend_seg2_exact<HAS_CONV_RES>(raw_y, raw_ca, raw_cr);
        }
        st_global_cs_u32(ConvOut + row_off + off, raw_y);
        if (AuxOut != nullptr) st_global_cs_u32(AuxOut + row_off + off, raw_y);
        return raw_y;
    };

    uint4 last_r0 = run_seg8(off_a0);
    uint4 last_r1 = run_seg8(off_a1);
    uint32_t last_rb = run_seg2(off_b);

    float q0_a0[8], q0_a1[8];
    float2 q0_bf = make_float2(0.0f, 0.0f);
    if constexpr (NQ == 1) {
        load_f32x8_const(Queries + off_a0, q0_a0);
        load_f32x8_const(Queries + off_a1, q0_a1);
        q0_bf = ld_const_f2(Queries + off_b);
    }

    float last_f0[8], last_f1[8], last_fb[2];
    {
        float lx0[8], lx1[8], lxb[2];
        unpack_bf16x8(last_r0, lx0);
        unpack_bf16x8(last_r1, lx1);
        unpack_bf16x2(last_rb, lxb);
        if constexpr (HAS_MIX_LAST) {
            uint4 ml_r0 = ld_global_cs_u4(MixLast + row_off + off_a0);
            uint4 ml_r1 = ld_global_cs_u4(MixLast + row_off + off_a1);
            uint32_t ml_rb = ld_global_cs_u32(MixLast + row_off + off_b);
            float m0[8], m1[8], mb[2];
            unpack_bf16x8(ml_r0, m0);
            unpack_bf16x8(ml_r1, m1);
            unpack_bf16x2(ml_rb, mb);
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                lx0[i] = __fadd_rn(m0[i], lx0[i]);
                lx1[i] = __fadd_rn(m1[i], lx1[i]);
                last_f0[i] = lx0[i];
                last_f1[i] = lx1[i];
            }
            lxb[0] = __fadd_rn(mb[0], lxb[0]);
            lxb[1] = __fadd_rn(mb[1], lxb[1]);
            last_fb[0] = lxb[0];
            last_fb[1] = lxb[1];
        }
#ifdef INKLING_PTXAS133_STRICT
        constexpr int P1_LAST_MODE = 0;
#else
        constexpr int P1_LAST_MODE = HAS_MIX_LAST ? 1 : 0;
#endif
        float sqa = tree16_sum_sq<P1_LAST_MODE>(lx0, lx1);
        float sqb = __fadd_rn(__fmul_rn(lxb[0], lxb[0]), __fmul_rn(lxb[1], lxb[1]));
        warp_sum_bfly2(sqa, sqb);
        if (lane == 0) {
            sRedSq[N_STACK][0][warp] = sqa;
            sRedSq[N_STACK][1][warp] = sqb;
        }
        if constexpr (NQ == 1) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                lx0[i] = __fmul_rn(q0_a0[i], lx0[i]);
                lx1[i] = __fmul_rn(q0_a1[i], lx1[i]);
            }
            float dta = tree16_sum(lx0, lx1);
            float dtb = __fadd_rn(__fmul_rn(q0_bf.x, lxb[0]), __fmul_rn(q0_bf.y, lxb[1]));
            warp_sum_bfly2(dta, dtb);
            if (lane == 0) {
                sRedDt[0][N_STACK][0][warp] = dta;
                sRedDt[0][N_STACK][1][warp] = dtb;
            }
        } else {
            #pragma unroll
            for (int q = 0; q < NQ; ++q) {
                const float* qp = Queries + static_cast<int64_t>(q) * D;
                float qa0[8], qa1[8];
                load_f32x8_const(qp + off_a0, qa0);
                load_f32x8_const(qp + off_a1, qa1);
                float2 qbf = ld_const_f2(qp + off_b);
                float dt0[8], dt1[8];
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    dt0[i] = __fmul_rn(qa0[i], lx0[i]);
                    dt1[i] = __fmul_rn(qa1[i], lx1[i]);
                }
                float dta = tree16_sum(dt0, dt1);
                float dtb = __fadd_rn(__fmul_rn(qbf.x, lxb[0]), __fmul_rn(qbf.y, lxb[1]));
                warp_sum_bfly2(dta, dtb);
                if (lane == 0) {
                    sRedDt[q][N_STACK][0][warp] = dta;
                    sRedDt[q][N_STACK][1][warp] = dtb;
                }
            }
        }
    }

    #pragma unroll
    for (int n = 0; n < N_STACK; ++n) {
        const __nv_bfloat16* sp = Stack + static_cast<int64_t>(n) * stride_s + row_off;
        uint4 r0 = ld_global_cs_u4(sp + off_a0);
        uint4 r1 = ld_global_cs_u4(sp + off_a1);
        uint32_t rb = ld_global_cs_u32(sp + off_b);
        sStack_a0[n * 256 + tid] = r0;
        sStack_a1[n * 256 + tid] = r1;
        sStack_b[n * 256 + tid]  = rb;

        float xa0[8], xa1[8], xb[2];
        unpack_bf16x8(r0, xa0);
        unpack_bf16x8(r1, xa1);
        unpack_bf16x2(rb, xb);

        float sqa = tree16_sum_sq<0>(xa0, xa1);
        float sqb = __fadd_rn(__fmul_rn(xb[0], xb[0]), __fmul_rn(xb[1], xb[1]));
        warp_sum_bfly2(sqa, sqb);
        if (lane == 0) {
            sRedSq[n][0][warp] = sqa;
            sRedSq[n][1][warp] = sqb;
        }
        if constexpr (NQ == 1) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                xa0[i] = __fmul_rn(q0_a0[i], xa0[i]);
                xa1[i] = __fmul_rn(q0_a1[i], xa1[i]);
            }
            float dta = tree16_sum(xa0, xa1);
            float dtb = __fadd_rn(__fmul_rn(q0_bf.x, xb[0]), __fmul_rn(q0_bf.y, xb[1]));
            warp_sum_bfly2(dta, dtb);
            if (lane == 0) {
                sRedDt[0][n][0][warp] = dta;
                sRedDt[0][n][1][warp] = dtb;
            }
        } else {
            #pragma unroll
            for (int q = 0; q < NQ; ++q) {
                const float* qp = Queries + static_cast<int64_t>(q) * D;
                float qa0[8], qa1[8];
                load_f32x8_const(qp + off_a0, qa0);
                load_f32x8_const(qp + off_a1, qa1);
                float2 qbf = ld_const_f2(qp + off_b);
                float dt0[8], dt1[8];
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    dt0[i] = __fmul_rn(qa0[i], xa0[i]);
                    dt1[i] = __fmul_rn(qa1[i], xa1[i]);
                }
                float dta = tree16_sum(dt0, dt1);
                float dtb = __fadd_rn(__fmul_rn(qbf.x, xb[0]), __fmul_rn(qbf.y, xb[1]));
                warp_sum_bfly2(dta, dtb);
                if (lane == 0) {
                    sRedDt[q][n][0][warp] = dta;
                    sRedDt[q][n][1][warp] = dtb;
                }
            }
        }
    }

    __syncthreads();

    const int l8 = lane & 7;
    const int half = (lane >> 3) & 1;
    float rstd_arr[NS];
    #pragma unroll
    for (int n = 0; n < NS; ++n) {
        float v = warp8_sum_bfly(sRedSq[n][half][l8]);
        float sa = __shfl_sync(0xFFFFFFFFu, v, 0);
        float sb = __shfl_sync(0xFFFFFFFFu, v, 8);
        rstd_arr[n] = rsqrt_approx(fmaf(__fadd_rn(sa, sb), INV_D, eps));
    }

    const bool keep_out_l2 = (T <= 512);
    #pragma unroll
    for (int q = 0; q < NQ; ++q) {
        float my_lg = -INFINITY;
        float my_ir = 0.0f;
        #pragma unroll
        for (int n = 0; n < NS; ++n) {
            float v = warp8_sum_bfly(sRedDt[q][n][half][l8]);
            float da = __shfl_sync(0xFFFFFFFFu, v, 0);
            float db = __shfl_sync(0xFFFFFFFFu, v, 8);
            if (lane == n) {
                my_lg = __fmul_rn(__fadd_rn(da, db), rstd_arr[n]);
                my_ir = rstd_arr[n];
            }
        }
        float mx = my_lg;
        if constexpr (BLOCK_N >= 16) mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, 8));
        if constexpr (BLOCK_N >= 8)  mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, 4));
        if constexpr (BLOCK_N >= 4)  mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, 2));
        if constexpr (BLOCK_N >= 2)  mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, 1));
        float p = (lane < NS) ? tldevice_exp_sub(my_lg, mx) : 0.0f;
        float sp = p;
        if constexpr (BLOCK_N >= 16) sp = __fadd_rn(sp, __shfl_xor_sync(0xFFFFFFFFu, sp, 8));
        if constexpr (BLOCK_N >= 8)  sp = __fadd_rn(sp, __shfl_xor_sync(0xFFFFFFFFu, sp, 4));
        if constexpr (BLOCK_N >= 4)  sp = __fadd_rn(sp, __shfl_xor_sync(0xFFFFFFFFu, sp, 2));
        if constexpr (BLOCK_N >= 2)  sp = __fadd_rn(sp, __shfl_xor_sync(0xFFFFFFFFu, sp, 1));
        float inv_sum = __fdiv_rn(1.0f, __shfl_sync(0xFFFFFFFFu, sp, 0));
        p = __fmul_rn(p, my_ir);
        float probs[NS];
        #pragma unroll
        for (int n = 0; n < NS; ++n) probs[n] = __shfl_sync(0xFFFFFFFFu, p, n);

        const int64_t q_off = static_cast<int64_t>(q) * D;
        uint4 nw_r0 = ld_const_u4(NormW + q_off + off_a0);
        uint4 nw_r1 = ld_const_u4(NormW + q_off + off_a1);
        uint32_t nw_rb = ld_const_u32(NormW + q_off + off_b);

        float acc0[8] = {0,0,0,0,0,0,0,0};
        float acc1[8] = {0,0,0,0,0,0,0,0};
        float accb[2] = {0,0};

        #pragma unroll
        for (int n = 0; n < N_STACK; ++n) {
            const float pn = probs[n];
            uint4 r0 = sStack_a0[n * 256 + tid];
            uint4 r1 = sStack_a1[n * 256 + tid];
            uint32_t rb = sStack_b[n * 256 + tid];
            const __nv_bfloat162* h0 = reinterpret_cast<const __nv_bfloat162*>(&r0);
            const __nv_bfloat162* h1 = reinterpret_cast<const __nv_bfloat162*>(&r1);
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                float2 f0 = __bfloat1622float2(h0[i]);
                acc0[2*i]   = fmaf(pn, f0.x, acc0[2*i]);
                acc0[2*i+1] = fmaf(pn, f0.y, acc0[2*i+1]);
                float2 f1 = __bfloat1622float2(h1[i]);
                acc1[2*i]   = fmaf(pn, f1.x, acc1[2*i]);
                acc1[2*i+1] = fmaf(pn, f1.y, acc1[2*i+1]);
            }
            float2 fb = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&rb));
            accb[0] = fmaf(pn, fb.x, accb[0]);
            accb[1] = fmaf(pn, fb.y, accb[1]);
        }
        {
            const float pn = probs[N_STACK];
            if constexpr (HAS_MIX_LAST) {
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    acc0[i] = fmaf(pn, last_f0[i], acc0[i]);
                    acc1[i] = fmaf(pn, last_f1[i], acc1[i]);
                }
                accb[0] = fmaf(pn, last_fb[0], accb[0]);
                accb[1] = fmaf(pn, last_fb[1], accb[1]);
            } else {
                const __nv_bfloat162* h0 = reinterpret_cast<const __nv_bfloat162*>(&last_r0);
                const __nv_bfloat162* h1 = reinterpret_cast<const __nv_bfloat162*>(&last_r1);
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    float2 f0 = __bfloat1622float2(h0[i]);
                    acc0[2*i]   = fmaf(pn, f0.x, acc0[2*i]);
                    acc0[2*i+1] = fmaf(pn, f0.y, acc0[2*i+1]);
                    float2 f1 = __bfloat1622float2(h1[i]);
                    acc1[2*i]   = fmaf(pn, f1.x, acc1[2*i]);
                    acc1[2*i+1] = fmaf(pn, f1.y, acc1[2*i+1]);
                }
                float2 fb = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&last_rb));
                accb[0] = fmaf(pn, fb.x, accb[0]);
                accb[1] = fmaf(pn, fb.y, accb[1]);
            }
        }

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            acc0[i] = __fmul_rn(acc0[i], inv_sum);
            acc1[i] = __fmul_rn(acc1[i], inv_sum);
        }
        accb[0] = __fmul_rn(accb[0], inv_sum);
        accb[1] = __fmul_rn(accb[1], inv_sum);

#ifdef INKLING_PTXAS133_STRICT
        constexpr int P2_MODE = 0;
#else
        constexpr int P2_MODE = (N_STACK == 8) ? 3 : (HAS_MIX_LAST ? 1 : 0);
#endif
        float sqa = tree16_sum_sq<P2_MODE>(acc0, acc1);
        float sqb = __fadd_rn(__fmul_rn(accb[0], accb[0]), __fmul_rn(accb[1], accb[1]));
        warp_sum_bfly2(sqa, sqb);
        if (lane == 0) {
            sSq2[0][warp] = sqa;
            sSq2[1][warp] = sqb;
        }
        __syncthreads();

        float v2 = warp8_sum_bfly(sSq2[half][l8]);
        float sa = __shfl_sync(0xFFFFFFFFu, v2, 0);
        float sb = __shfl_sync(0xFFFFFFFFu, v2, 8);
        float out_rstd = rsqrt_approx(__fadd_rn(__fdiv_rn(__fadd_rn(sa, sb), static_cast<float>(D)), norm_eps));

        float wa0[8], wa1[8], wb[2];
        unpack_bf16x8(nw_r0, wa0);
        unpack_bf16x8(nw_r1, wa1);
        unpack_bf16x2(nw_rb, wb);

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            acc0[i] = __fmul_rn(__fmul_rn(acc0[i], out_rstd), wa0[i]);
            acc1[i] = __fmul_rn(__fmul_rn(acc1[i], out_rstd), wa1[i]);
        }
        accb[0] = __fmul_rn(__fmul_rn(accb[0], out_rstd), wb[0]);
        accb[1] = __fmul_rn(__fmul_rn(accb[1], out_rstd), wb[1]);

        __nv_bfloat16* out_p = Out + static_cast<int64_t>(q) * T * D + row_off;
        uint4 o0 = pack_f32x8_to_bf16_u4(acc0);
        uint4 o1 = pack_f32x8_to_bf16_u4(acc1);
        uint32_t ob = pack_f32x2_to_bf16_u32(accb);
        if (keep_out_l2) {
            *reinterpret_cast<uint4*>(out_p + off_a0)   = o0;
            *reinterpret_cast<uint4*>(out_p + off_a1)   = o1;
            *reinterpret_cast<uint32_t*>(out_p + off_b) = ob;
        } else {
            st_global_cs_u4(out_p + off_a0, o0);
            st_global_cs_u4(out_p + off_a1, o1);
            st_global_cs_u32(out_p + off_b, ob);
        }

        if constexpr (NQ > 1) {
            if (q + 1 < NQ) __syncthreads();
        }
    }
}

// =========================================================================
// KERNEL 3 (C=32 REGIME B, T*nq > 256): `fused_sconv_ar_c32_blockd_kernel`
//   Exact 0-ULP match to short_conv + Triton `_attn_res_kernel` (BLOCK_D=1024, 128 thr).
// =========================================================================
template <int N_STACK, int NQ, bool HAS_CONV_RES, bool HAS_CONV_ADDEND, bool HAS_MIX_LAST>
__global__ void __launch_bounds__(128, 1)
fused_sconv_ar_c32_blockd_kernel(
    const __nv_bfloat16* __restrict__ X,
    const __nv_bfloat16* __restrict__ CW,
    __nv_bfloat16* __restrict__ Cache,
    __nv_bfloat16* ConvOut,
    const __nv_bfloat16* ConvRes,
    const __nv_bfloat16* __restrict__ ConvAdd,
    __nv_bfloat16* __restrict__ AuxOut,
    const int64_t* __restrict__ Pos,
    const int32_t* __restrict__ SeqIdx,
    const int64_t* __restrict__ Slot,
    const int32_t* __restrict__ BT,
    const int32_t* __restrict__ QStart,
    const __nv_bfloat16* __restrict__ Stack,
    const __nv_bfloat16* __restrict__ MixLast,
    const float* __restrict__ Queries,
    const __nv_bfloat16* __restrict__ NormW,
    __nv_bfloat16* __restrict__ Out,
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
    constexpr int U4_PER_ROW = D / 8;
    constexpr int NS = N_STACK + 1;
    constexpr float INV_D = 1.0f / static_cast<float>(D);
    constexpr int BLOCK_N = (NS <= 2) ? 2 : (NS <= 4) ? 4 : (NS <= 8) ? 8 : 16;

    const int t = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int64_t row_off = static_cast<int64_t>(t) * D;

    extern __shared__ uint4 dyn_raw_u4[];
    uint4* sStack = dyn_raw_u4;
    uint4* sLastU4 = dyn_raw_u4 + N_STACK * U4_PER_ROW;
    float* sLastF32 = reinterpret_cast<float*>(sLastU4);

    alignas(16) __shared__ float sRed[NS][1 + NQ][4];
    alignas(16) __shared__ float sSq2[5][4];

    const int64_t slot = Slot[t];
    const bool valid = (slot >= 0);
    const int64_t pos = Pos[t];
    const int32_t req = SeqIdx[t];
    const int32_t qstart = QStart[t];
    const bool lag0_ok = valid && (pos >= 0);

    __nv_bfloat16* cache_dst = nullptr;
    if (valid) {
        int64_t dblk, drem;
        divmod_cache(slot, N_CACHE, N_SHIFT, dblk, drem);
        cache_dst = Cache + dblk * stride_c_blk + drem * stride_c_n + stream_offset;
    }

    const __nv_bfloat16* tap_ptrs[3];
    uint32_t tap_flags = 0u;
    #pragma unroll
    for (int k = 0; k < 3; ++k) {
        const int lag = 3 - k;
        const int64_t src = pos - lag;
        const bool in_win = valid && (src >= 0);
        if (in_win) {
            tap_flags |= (1u << k);
            const int row = t - lag;
            if (row >= qstart) {
                tap_ptrs[k] = X + static_cast<int64_t>(row) * stride_x;
            } else {
                int64_t lblk, rem;
                divmod_cache(src, N_CACHE, N_SHIFT, lblk, rem);
                if (lblk >= max_blocks) lblk = max_blocks - 1;
                const int64_t blk = static_cast<int64_t>(BT[static_cast<int64_t>(req) * stride_bt + lblk]);
                tap_ptrs[k] = Cache + blk * stride_c_blk + rem * stride_c_n + stream_offset;
                tap_flags |= (1u << (k + 3));
            }
        } else {
            tap_ptrs[k] = X;
        }
    }

    const __nv_bfloat16* x_cur_p = X + static_cast<int64_t>(t) * stride_x;
    const bool tv0 = (tap_flags & 1u) != 0u;
    const bool tv1 = (tap_flags & 2u) != 0u;
    const bool tv2 = (tap_flags & 4u) != 0u;
    const bool tc0 = (tap_flags & 8u) != 0u;
    const bool tc1 = (tap_flags & 16u) != 0u;
    const bool tc2 = (tap_flags & 32u) != 0u;

    {
        float sum_sq[8] = {0,0,0,0,0,0,0,0};
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
                uint4 raw_cur = ld_global_ca_u4(x_cur_p + d_base);
                uint4 raw_t0 = tv0 ? (tc0 ? ld_global_cs_u4(tap_ptrs[0] + d_base) : ld_global_ca_u4(tap_ptrs[0] + d_base)) : make_uint4(0u,0u,0u,0u);
                uint4 raw_t1 = tv1 ? (tc1 ? ld_global_cs_u4(tap_ptrs[1] + d_base) : ld_global_ca_u4(tap_ptrs[1] + d_base)) : make_uint4(0u,0u,0u,0u);
                uint4 raw_t2 = tv2 ? (tc2 ? ld_global_cs_u4(tap_ptrs[2] + d_base) : ld_global_ca_u4(tap_ptrs[2] + d_base)) : make_uint4(0u,0u,0u,0u);
                if (valid) st_global_cs_u4(cache_dst + d_base, raw_cur);

                uint4 raw_y = sconv_w4_seg8_exact(
                    raw_cur, raw_t0, raw_t1, raw_t2, CW + d_base * 4, tv0, tv1, tv2, lag0_ok, scale
                );
                if constexpr (HAS_CONV_ADDEND) {
                    uint4 raw_ca = ld_global_cs_u4(ConvAdd + row_off + d_base);
                    uint4 raw_cr = make_uint4(0u,0u,0u,0u);
                    if constexpr (HAS_CONV_RES) raw_cr = ld_global_cs_u4(ConvRes + row_off + d_base);
                    raw_y = apply_conv_addend_seg8_exact<HAS_CONV_RES>(raw_y, raw_ca, raw_cr);
                }
                st_global_cs_u4(ConvOut + row_off + d_base, raw_y);
                if (AuxOut != nullptr) st_global_cs_u4(AuxOut + row_off + d_base, raw_y);

                float x[8];
                unpack_bf16x8(raw_y, x);
                if constexpr (HAS_MIX_LAST) {
                    uint4 raw_ml = ld_global_cs_u4(MixLast + row_off + d_base);
                    float ml[8];
                    unpack_bf16x8(raw_ml, ml);
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) x[i] = __fadd_rn(ml[i], x[i]);
                    *reinterpret_cast<float4*>(sLastF32 + d_base)     = make_float4(x[0], x[1], x[2], x[3]);
                    *reinterpret_cast<float4*>(sLastF32 + d_base + 4) = make_float4(x[4], x[5], x[6], x[7]);
                } else {
                    sLastU4[d_base >> 3] = raw_y;
                }
                #pragma unroll
                for (int i = 0; i < 8; ++i) sum_sq[i] = fmaf(x[i], x[i], sum_sq[i]);
                #pragma unroll
                for (int q = 0; q < NQ; ++q) {
                    float qv[8];
                    load_f32x8_const(Queries + static_cast<int64_t>(q) * D + d_base, qv);
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) dots[q][i] = fmaf(qv[i], x[i], dots[q][i]);
                }
            }
        }
        float w_sq = warp_sum_bfly(tree8_sum(sum_sq));
        if (lane == 0) sRed[N_STACK][0][warp] = w_sq;
        #pragma unroll
        for (int q = 0; q < NQ; ++q) {
            float w_dt = warp_sum_bfly(tree8_sum(dots[q]));
            if (lane == 0) sRed[N_STACK][1 + q][warp] = w_dt;
        }
    }

    #pragma unroll
    for (int n = 0; n < N_STACK; ++n) {
        const __nv_bfloat16* sp = Stack + static_cast<int64_t>(n) * stride_s + row_off;
        uint4* dst_s = sStack + n * U4_PER_ROW;
        uint4 raws[5];
        #pragma unroll
        for (int c = 0; c < 5; ++c) {
            if (c < 4 || tid < 64) raws[c] = ld_global_cs_u4(sp + c * 1024 + tid * 8);
        }
        float sum_sq[8] = {0,0,0,0,0,0,0,0};
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
                dst_s[d_base >> 3] = raws[c];
                float x[8];
                unpack_bf16x8(raws[c], x);
                #pragma unroll
                for (int i = 0; i < 8; ++i) sum_sq[i] = fmaf(x[i], x[i], sum_sq[i]);
                #pragma unroll
                for (int q = 0; q < NQ; ++q) {
                    float qv[8];
                    load_f32x8_const(Queries + static_cast<int64_t>(q) * D + d_base, qv);
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) dots[q][i] = fmaf(qv[i], x[i], dots[q][i]);
                }
            }
        }
        float w_sq = warp_sum_bfly(tree8_sum(sum_sq));
        if (lane == 0) sRed[n][0][warp] = w_sq;
        #pragma unroll
        for (int q = 0; q < NQ; ++q) {
            float w_dt = warp_sum_bfly(tree8_sum(dots[q]));
            if (lane == 0) sRed[n][1 + q][warp] = w_dt;
        }
    }

    __syncthreads();

    const int l4 = lane & 3;
    float rstd_arr[NS];
    #pragma unroll
    for (int n = 0; n < NS; ++n) {
        float tot_sq = warp4_sum_bfly(sRed[n][0][l4]);
        rstd_arr[n] = rsqrt_approx(fmaf(tot_sq, INV_D, eps));
    }

    const bool keep_out_l2 = (T <= 512);
    #pragma unroll
    for (int q = 0; q < NQ; ++q) {
        float my_lg = -INFINITY;
        float my_ir = 0.0f;
        #pragma unroll
        for (int n = 0; n < NS; ++n) {
            float tot_dt = warp4_sum_bfly(sRed[n][1 + q][l4]);
            if (lane == n) {
                my_lg = __fmul_rn(tot_dt, rstd_arr[n]);
                my_ir = rstd_arr[n];
            }
        }
        float mx = my_lg;
        if constexpr (BLOCK_N >= 16) mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, 8));
        if constexpr (BLOCK_N >= 8)  mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, 4));
        if constexpr (BLOCK_N >= 4)  mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, 2));
        if constexpr (BLOCK_N >= 2)  mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, 1));
        float p = (lane < NS) ? tldevice_exp_sub(my_lg, mx) : 0.0f;
        float sp = p;
        if constexpr (BLOCK_N >= 16) sp = __fadd_rn(sp, __shfl_xor_sync(0xFFFFFFFFu, sp, 8));
        if constexpr (BLOCK_N >= 8)  sp = __fadd_rn(sp, __shfl_xor_sync(0xFFFFFFFFu, sp, 4));
        if constexpr (BLOCK_N >= 4)  sp = __fadd_rn(sp, __shfl_xor_sync(0xFFFFFFFFu, sp, 2));
        if constexpr (BLOCK_N >= 2)  sp = __fadd_rn(sp, __shfl_xor_sync(0xFFFFFFFFu, sp, 1));
        float inv_sum = __fdiv_rn(1.0f, __shfl_sync(0xFFFFFFFFu, sp, 0));
        p = __fmul_rn(p, my_ir);
        float probs[NS];
        #pragma unroll
        for (int n = 0; n < NS; ++n) probs[n] = __shfl_sync(0xFFFFFFFFu, p, n);

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
                    const float pn = probs[n];
                    float x[8];
                    unpack_bf16x8(sStack[n * U4_PER_ROW + u4_idx], x);
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) acc[c][i] = fmaf(pn, x[i], acc[c][i]);
                }
                {
                    const float pn = probs[N_STACK];
                    float x[8];
                    if constexpr (HAS_MIX_LAST) {
                        float4 v0 = *reinterpret_cast<const float4*>(sLastF32 + d_base);
                        float4 v1 = *reinterpret_cast<const float4*>(sLastF32 + d_base + 4);
                        x[0]=v0.x; x[1]=v0.y; x[2]=v0.z; x[3]=v0.w;
                        x[4]=v1.x; x[5]=v1.y; x[6]=v1.z; x[7]=v1.w;
                    } else {
                        unpack_bf16x8(sLastU4[u4_idx], x);
                    }
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) acc[c][i] = fmaf(pn, x[i], acc[c][i]);
                }
                #pragma unroll
                for (int i = 0; i < 8; ++i) acc[c][i] = __fmul_rn(acc[c][i], inv_sum);
#ifdef INKLING_PTXAS133_STRICT
                float sq_v[8];
                #pragma unroll
                for (int i = 0; i < 8; ++i) sq_v[i] = __fmul_rn(acc[c][i], acc[c][i]);
                sq_sum = tree8_sum(sq_v);
#else
                float e01 = fmaf(acc[c][0], acc[c][0], __fmul_rn(acc[c][2], acc[c][2]));
                float o01 = fmaf(acc[c][1], acc[c][1], __fmul_rn(acc[c][3], acc[c][3]));
                float e23 = fmaf(acc[c][4], acc[c][4], __fmul_rn(acc[c][6], acc[c][6]));
                float o23 = fmaf(acc[c][5], acc[c][5], __fmul_rn(acc[c][7], acc[c][7]));
                sq_sum = __fadd_rn(__fadd_rn(e01, e23), __fadd_rn(o01, o23));
#endif
            }
            float w_sq = warp_sum_bfly(sq_sum);
            if (lane == 0) sSq2[c][warp] = w_sq;
        }

        __syncthreads();

        float tot_sum_sq = 0.0f;
        #pragma unroll
        for (int c = 0; c < 5; ++c) {
            tot_sum_sq = __fadd_rn(tot_sum_sq, warp4_sum_bfly(sSq2[c][l4]));
        }
        float out_rstd = rsqrt_approx(__fadd_rn(__fdiv_rn(tot_sum_sq, static_cast<float>(D)), norm_eps));

        __nv_bfloat16* out_p = Out + static_cast<int64_t>(q) * T * D + row_off;
        const __nv_bfloat16* nw_p = NormW + static_cast<int64_t>(q) * D;
        #pragma unroll
        for (int c = 0; c < 5; ++c) {
            const int d_base = c * 1024 + tid * 8;
            if (c < 4 || tid < 64) {
                float w[8];
                unpack_bf16x8(ld_const_u4(nw_p + d_base), w);
                float res[8];
                #pragma unroll
                for (int i = 0; i < 8; ++i) res[i] = __fmul_rn(__fmul_rn(acc[c][i], out_rstd), w[i]);
                uint4 out_u4 = pack_f32x8_to_bf16_u4(res);
                if (keep_out_l2) *reinterpret_cast<uint4*>(out_p + d_base) = out_u4;
                else             st_global_cs_u4(out_p + d_base, out_u4);
            }
        }
        if constexpr (NQ > 1) {
            if (q + 1 < NQ) __syncthreads();
        }
    }
}

// =========================================================================
// KERNEL 4 (STANDALONE SHORTCONV V14, T = 1..64): `sconv_exact_v14_kernel`
//   Closes the 33..64 token gap in `sconv_ops.py`, adding NLivePtr, TreeAnc,
//   fused AuxOut (all combos including layer 40 addend=1,res=0,aux=1),
//   pre-wait ConvAdd/ConvRes/cached-tap overlap, and #206-safe PDL!
// =========================================================================
template <bool HAS_CONV_RES, bool HAS_CONV_ADDEND, bool PDL>
__global__ void __launch_bounds__(256, 2)
sconv_exact_v14_kernel(
    const __nv_bfloat16* __restrict__ X,
    __nv_bfloat16* __restrict__ Cache,
    const __nv_bfloat16* __restrict__ CW,
    __nv_bfloat16* Out,
    __nv_bfloat16* __restrict__ AuxOut,
    const __nv_bfloat16* ConvRes,
    const __nv_bfloat16* __restrict__ ConvAdd,
    const int64_t* __restrict__ Pos,
    const int32_t* __restrict__ SeqIdx,
    const int64_t* __restrict__ Slot,
    const int32_t* __restrict__ BT,
    const int32_t* __restrict__ QStart,
    const int32_t* __restrict__ NLivePtr,
    const int32_t* __restrict__ TreeAnc,
    int T,
    int64_t stride_x,
    int64_t stride_c_blk,
    int64_t stride_c_n,
    int64_t stride_bt,
    int max_blocks,
    int N_CACHE,
    int N_SHIFT,
    int stream_offset,
    float scale,
    int pdl_mode
) {
    constexpr int D = 4608;
    const int t = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    const int64_t row_off = static_cast<int64_t>(t) * D;
    const int off_a0 = tid * 8;
    const int off_a1 = 2048 + tid * 8;
    const int off_b  = 4096 + tid * 2;

    int nl = T;
    if (NLivePtr != nullptr) {
        const int raw_nl = *NLivePtr;
        if (raw_nl >= 1 && raw_nl <= T) nl = raw_nl;
    }
    if (t >= nl) {
        if constexpr (PDL) {
            gdc_wait_v14();
        }
        const uint4 z4 = make_uint4(0u, 0u, 0u, 0u);
        *reinterpret_cast<uint4*>(Out + row_off + off_a0)   = z4;
        *reinterpret_cast<uint4*>(Out + row_off + off_a1)   = z4;
        *reinterpret_cast<uint32_t*>(Out + row_off + off_b) = 0u;
        if (AuxOut != nullptr) {
            st_global_cs_u4(AuxOut + row_off + off_a0, z4);
            st_global_cs_u4(AuxOut + row_off + off_a1, z4);
            st_global_cs_u32(AuxOut + row_off + off_b, 0u);
        }
        if constexpr (PDL) {
            __syncthreads();
            if (pdl_mode >= 2 && tid == 0) gdc_launch_dependents_v14();
        }
        return;
    }

    if (T == 1) {
        // V16 fix: full 576-line CW prefetch when T == 1!
        prefetch_l2(CW + tid * 32);
        prefetch_l2(CW + (256 + tid) * 32);
        if (tid < 64) prefetch_l2(CW + (512 + tid) * 32);
    } else if (t < 2) {
        const int cw_line_base = (t * 288) + tid;
        if (cw_line_base < 576) prefetch_l2(CW + cw_line_base * 32);
        if (tid < 32) {
            const int cw_line_hi = (t * 288) + 256 + tid;
            if (cw_line_hi < 576) prefetch_l2(CW + cw_line_hi * 32);
        }
    }

    const int64_t slot = Slot[t];
    const bool valid = (slot >= 0);
    const int64_t pos = Pos[t];
    const int32_t req = SeqIdx[t];
    const int32_t qstart = QStart[t];
    const bool lag0_ok = valid && (pos >= 0);

    __nv_bfloat16* cache_dst = nullptr;
    if (valid) {
        int64_t dblk, drem;
        divmod_cache(slot, N_CACHE, N_SHIFT, dblk, drem);
        cache_dst = Cache + dblk * stride_c_blk + drem * stride_c_n + stream_offset;
    }

    const __nv_bfloat16* tap_ptrs[3];
    uint32_t tap_flags = 0u;
    #pragma unroll
    for (int k = 0; k < 3; ++k) {
        const int lag = 3 - k;
        const int64_t src = pos - lag;
        const bool in_win = valid && (src >= 0);
        if (in_win) {
            tap_flags |= (1u << k);
            int row = t - lag;
            bool is_intra = (row >= qstart);
            if (TreeAnc != nullptr) {
                const int anc = TreeAnc[t * 4 + lag];
                row = anc;
                is_intra = (anc >= 0);
            }
            if (is_intra) {
                tap_ptrs[k] = X + static_cast<int64_t>(row) * stride_x;
            } else {
                int64_t lblk, rem;
                divmod_cache(src, N_CACHE, N_SHIFT, lblk, rem);
                if (lblk >= max_blocks) lblk = max_blocks - 1;
                const int64_t blk = static_cast<int64_t>(BT[static_cast<int64_t>(req) * stride_bt + lblk]);
                tap_ptrs[k] = Cache + blk * stride_c_blk + rem * stride_c_n + stream_offset;
                tap_flags |= (1u << (k + 3));
                if (tid < 72) prefetch_l2(tap_ptrs[k] + tid * 64);
            }
        } else {
            tap_ptrs[k] = X;
        }
    }

    uint4 ca_r0 = make_uint4(0u, 0u, 0u, 0u), ca_r1 = make_uint4(0u, 0u, 0u, 0u);
    uint32_t ca_rb = 0u;
    uint4 cr_r0 = make_uint4(0u, 0u, 0u, 0u), cr_r1 = make_uint4(0u, 0u, 0u, 0u);
    uint32_t cr_rb = 0u;
    if constexpr (HAS_CONV_ADDEND) {
        ca_r0 = ld_global_cs_u4(ConvAdd + row_off + off_a0);
        ca_r1 = ld_global_cs_u4(ConvAdd + row_off + off_a1);
        ca_rb = ld_global_cs_u32(ConvAdd + row_off + off_b);
        if constexpr (HAS_CONV_RES) {
            cr_r0 = ld_global_cs_u4(ConvRes + row_off + off_a0);
            cr_r1 = ld_global_cs_u4(ConvRes + row_off + off_a1);
            cr_rb = ld_global_cs_u32(ConvRes + row_off + off_b);
        }
    }

    if constexpr (PDL) {
        gdc_wait_v14();
        if (pdl_mode >= 3) {
            __syncthreads();
            if (tid == 0) gdc_launch_dependents_v14();
        }
    }

    const __nv_bfloat16* x_cur_p = X + static_cast<int64_t>(t) * stride_x;
    const bool tv0 = (tap_flags & 1u) != 0u;
    const bool tv1 = (tap_flags & 2u) != 0u;
    const bool tv2 = (tap_flags & 4u) != 0u;
    const bool tc0 = (tap_flags & 8u) != 0u;
    const bool tc1 = (tap_flags & 16u) != 0u;
    const bool tc2 = (tap_flags & 32u) != 0u;

    auto do_seg8 = [&](int off, uint4 pre_ca, uint4 pre_cr) {
        uint4 raw_cur = ld_global_ca_u4(x_cur_p + off);
        uint4 raw_t0 = tv0 ? (tc0 ? ld_global_cs_u4(tap_ptrs[0] + off) : ld_global_ca_u4(tap_ptrs[0] + off)) : make_uint4(0u, 0u, 0u, 0u);
        uint4 raw_t1 = tv1 ? (tc1 ? ld_global_cs_u4(tap_ptrs[1] + off) : ld_global_ca_u4(tap_ptrs[1] + off)) : make_uint4(0u, 0u, 0u, 0u);
        uint4 raw_t2 = tv2 ? (tc2 ? ld_global_cs_u4(tap_ptrs[2] + off) : ld_global_ca_u4(tap_ptrs[2] + off)) : make_uint4(0u, 0u, 0u, 0u);
        if (valid) st_global_cs_u4(cache_dst + off, raw_cur);

        uint4 raw_y = sconv_w4_seg8_exact(
            raw_cur, raw_t0, raw_t1, raw_t2, CW + off * 4, tv0, tv1, tv2, lag0_ok, scale
        );
        if constexpr (HAS_CONV_ADDEND) {
            raw_y = apply_conv_addend_seg8_exact<HAS_CONV_RES>(raw_y, pre_ca, pre_cr);
        } else {
            (void)pre_ca; (void)pre_cr;
        }
        *reinterpret_cast<uint4*>(Out + row_off + off) = raw_y;
        if (AuxOut != nullptr) st_global_cs_u4(AuxOut + row_off + off, raw_y);
    };

    do_seg8(off_a0, ca_r0, cr_r0);
    do_seg8(off_a1, ca_r1, cr_r1);

    {
        uint32_t raw_cur = ld_global_ca_u32(x_cur_p + off_b);
        uint32_t raw_t0 = tv0 ? (tc0 ? ld_global_cs_u32(tap_ptrs[0] + off_b) : ld_global_ca_u32(tap_ptrs[0] + off_b)) : 0u;
        uint32_t raw_t1 = tv1 ? (tc1 ? ld_global_cs_u32(tap_ptrs[1] + off_b) : ld_global_ca_u32(tap_ptrs[1] + off_b)) : 0u;
        uint32_t raw_t2 = tv2 ? (tc2 ? ld_global_cs_u32(tap_ptrs[2] + off_b) : ld_global_ca_u32(tap_ptrs[2] + off_b)) : 0u;
        if (valid) st_global_cs_u32(cache_dst + off_b, raw_cur);

        uint4 cw_u4 = ld_const_u4(CW + off_b * 4);
        uint32_t raw_y = sconv_w4_pair_exact(
            raw_cur, raw_t0, raw_t1, raw_t2, cw_u4, tv0, tv1, tv2, lag0_ok, scale
        );
        if constexpr (HAS_CONV_ADDEND) {
            raw_y = apply_conv_addend_seg2_exact<HAS_CONV_RES>(raw_y, ca_rb, cr_rb);
        }
        *reinterpret_cast<uint32_t*>(Out + row_off + off_b) = raw_y;
        if (AuxOut != nullptr) st_global_cs_u32(AuxOut + row_off + off_b, raw_y);
    }

    if constexpr (PDL) {
        if (pdl_mode == 2) {
            __syncthreads();
            if (tid == 0) gdc_launch_dependents_v14();
        }
    }
}

// =========================================================================
// KERNEL 5 (STANDALONE MULTI-QUERY SMEM ATTNRES V14, T*nq <= 256):
//   Replaces `grid=(nq, T)` (3x DRAM Stack reads on 8 Global layers) with
//   `grid=(T, 1, 1)` 1x SMEM-cached Stack + pre-`gdc_wait()` Stack preload
//   + post-`gdc_wait()` Last/Addend read (#206-safe after short_conv!) +
//   NQ=1 Queries hoisting + N_STACK=1..10 + normalize_values in {0, 1}!
// =========================================================================
template <int N_STACK, int NQ, bool HAS_ADDEND, bool PDL>
__global__ void __launch_bounds__(256, (N_STACK <= 5) ? 2 : 1)
attn_res_exact_v14_kernel(
    const __nv_bfloat16* __restrict__ Stack,
    const __nv_bfloat16* __restrict__ Last,
    const __nv_bfloat16* __restrict__ Addend,
    const float* __restrict__ Queries,
    const __nv_bfloat16* __restrict__ NormW,
    __nv_bfloat16* __restrict__ Out,
    const int32_t* __restrict__ NLivePtr,
    int64_t stride_s,
    int T,
    float eps,
    float norm_eps,
    int normalize_values,
    int early_pdl
) {
    constexpr int D = 4608;
    constexpr int NS = N_STACK + 1;
    constexpr float INV_D = 1.0f / static_cast<float>(D);
    constexpr int BLOCK_N = (NS <= 2) ? 2 : (NS <= 4) ? 4 : (NS <= 8) ? 8 : 16;

    const int t = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int64_t row_off = static_cast<int64_t>(t) * D;
    const int off_a0 = tid * 8;
    const int off_a1 = 2048 + tid * 8;
    const int off_b  = 4096 + tid * 2;

    int nl = T;
    if (NLivePtr != nullptr) {
        const int raw_nl = *NLivePtr;
        if (raw_nl >= 1 && raw_nl <= T) nl = raw_nl;
    }
    if (t >= nl) {
        if constexpr (PDL) {
            gdc_wait_v14();
        }
        const uint4 z4 = make_uint4(0u, 0u, 0u, 0u);
        #pragma unroll
        for (int q = 0; q < NQ; ++q) {
            __nv_bfloat16* out_p = Out + static_cast<int64_t>(q) * T * D + row_off;
            *reinterpret_cast<uint4*>(out_p + off_a0)   = z4;
            *reinterpret_cast<uint4*>(out_p + off_a1)   = z4;
            *reinterpret_cast<uint32_t*>(out_p + off_b) = 0u;
        }
        if constexpr (PDL) {
            __syncthreads();
            if (tid == 0) gdc_launch_dependents_v14();
        }
        return;
    }

    extern __shared__ uint4 dyn_raw_u4[];
    uint8_t*  dyn_raw   = reinterpret_cast<uint8_t*>(dyn_raw_u4);
    uint4*    sStack_a0 = dyn_raw_u4;
    uint4*    sStack_a1 = reinterpret_cast<uint4*>(dyn_raw + static_cast<size_t>(N_STACK) * 4096);
    uint32_t* sStack_b  = reinterpret_cast<uint32_t*>(dyn_raw + static_cast<size_t>(N_STACK) * 8192);

    alignas(16) __shared__ float sRedSq[NS][2][8];
    alignas(16) __shared__ float sRedDt[NQ][NS][2][8];
    alignas(16) __shared__ float sSq2[2][8];

    // PRE-WAIT: Prefetch NormW and stream Stack[0..N_STACK-1] into SMEM + Pass-1 reductions.
    // (Last & Addend may be written by immediately preceding `short_conv` -> strictly POST-WAIT!)
    if (t < 4 && tid < 72) {
        #pragma unroll
        for (int q = 0; q < NQ; ++q) {
            prefetch_l2(NormW + static_cast<int64_t>(q) * D + tid * 64);
        }
    }

    float q0_a0[8], q0_a1[8];
    float2 q0_bf = make_float2(0.0f, 0.0f);
    if constexpr (NQ == 1) {
        load_f32x8_const(Queries + off_a0, q0_a0);
        load_f32x8_const(Queries + off_a1, q0_a1);
        q0_bf = ld_const_f2(Queries + off_b);
    }

    #pragma unroll
    for (int n = 0; n < N_STACK; ++n) {
        const __nv_bfloat16* sp = Stack + static_cast<int64_t>(n) * stride_s + row_off;
        uint4 r0 = ld_global_cs_u4(sp + off_a0);
        uint4 r1 = ld_global_cs_u4(sp + off_a1);
        uint32_t rb = ld_global_cs_u32(sp + off_b);
        sStack_a0[n * 256 + tid] = r0;
        sStack_a1[n * 256 + tid] = r1;
        sStack_b[n * 256 + tid]  = rb;

        float xa0[8], xa1[8], xb[2];
        unpack_bf16x8(r0, xa0);
        unpack_bf16x8(r1, xa1);
        unpack_bf16x2(rb, xb);

        float sqa = tree16_sum_sq<0>(xa0, xa1);
        float sqb = __fadd_rn(__fmul_rn(xb[0], xb[0]), __fmul_rn(xb[1], xb[1]));
        warp_sum_bfly2(sqa, sqb);
        if (lane == 0) {
            sRedSq[n][0][warp] = sqa;
            sRedSq[n][1][warp] = sqb;
        }
        if constexpr (NQ == 1) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                xa0[i] = __fmul_rn(q0_a0[i], xa0[i]);
                xa1[i] = __fmul_rn(q0_a1[i], xa1[i]);
            }
            float dta = tree16_sum(xa0, xa1);
            float dtb = __fadd_rn(__fmul_rn(q0_bf.x, xb[0]), __fmul_rn(q0_bf.y, xb[1]));
            warp_sum_bfly2(dta, dtb);
            if (lane == 0) {
                sRedDt[0][n][0][warp] = dta;
                sRedDt[0][n][1][warp] = dtb;
            }
        } else {
            #pragma unroll
            for (int q = 0; q < NQ; ++q) {
                const float* qp = Queries + static_cast<int64_t>(q) * D;
                float qa0[8], qa1[8];
                load_f32x8_const(qp + off_a0, qa0);
                load_f32x8_const(qp + off_a1, qa1);
                float2 qbf = ld_const_f2(qp + off_b);
                float dt0[8], dt1[8];
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    dt0[i] = __fmul_rn(qa0[i], xa0[i]);
                    dt1[i] = __fmul_rn(qa1[i], xa1[i]);
                }
                float dta = tree16_sum(dt0, dt1);
                float dtb = __fadd_rn(__fmul_rn(qbf.x, xb[0]), __fmul_rn(qbf.y, xb[1]));
                warp_sum_bfly2(dta, dtb);
                if (lane == 0) {
                    sRedDt[q][n][0][warp] = dta;
                    sRedDt[q][n][1][warp] = dtb;
                }
            }
        }
    }

    if constexpr (PDL) {
        gdc_wait_v14();
        if (early_pdl != 0) {
            __syncthreads();
            if (tid == 0) gdc_launch_dependents_v14();
        }
    }

    uint4 last_r0 = ld_global_ca_u4(Last + row_off + off_a0);
    uint4 last_r1 = ld_global_ca_u4(Last + row_off + off_a1);
    uint32_t last_rb = ld_global_ca_u32(Last + row_off + off_b);

    float last_f0[8], last_f1[8], last_fb[2];
    {
        float lx0[8], lx1[8], lxb[2];
        unpack_bf16x8(last_r0, lx0);
        unpack_bf16x8(last_r1, lx1);
        unpack_bf16x2(last_rb, lxb);
        if constexpr (HAS_ADDEND) {
            uint4 ad_r0 = ld_global_ca_u4(Addend + row_off + off_a0);
            uint4 ad_r1 = ld_global_ca_u4(Addend + row_off + off_a1);
            uint32_t ad_rb = ld_global_ca_u32(Addend + row_off + off_b);
            float a0[8], a1[8], ab[2];
            unpack_bf16x8(ad_r0, a0);
            unpack_bf16x8(ad_r1, a1);
            unpack_bf16x2(ad_rb, ab);
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                lx0[i] = __fadd_rn(lx0[i], a0[i]);
                lx1[i] = __fadd_rn(lx1[i], a1[i]);
                last_f0[i] = lx0[i];
                last_f1[i] = lx1[i];
            }
            lxb[0] = __fadd_rn(lxb[0], ab[0]);
            lxb[1] = __fadd_rn(lxb[1], ab[1]);
            last_fb[0] = lxb[0];
            last_fb[1] = lxb[1];
        }
#ifdef INKLING_PTXAS133_STRICT
        constexpr int P1_LAST_MODE = 0;
#else
        constexpr int P1_LAST_MODE = HAS_ADDEND ? 1 : 0;
#endif
        float sqa = tree16_sum_sq<P1_LAST_MODE>(lx0, lx1);
        float sqb = __fadd_rn(__fmul_rn(lxb[0], lxb[0]), __fmul_rn(lxb[1], lxb[1]));
        warp_sum_bfly2(sqa, sqb);
        if (lane == 0) {
            sRedSq[N_STACK][0][warp] = sqa;
            sRedSq[N_STACK][1][warp] = sqb;
        }
        if constexpr (NQ == 1) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                lx0[i] = __fmul_rn(q0_a0[i], lx0[i]);
                lx1[i] = __fmul_rn(q0_a1[i], lx1[i]);
            }
            float dta = tree16_sum(lx0, lx1);
            float dtb = __fadd_rn(__fmul_rn(q0_bf.x, lxb[0]), __fmul_rn(q0_bf.y, lxb[1]));
            warp_sum_bfly2(dta, dtb);
            if (lane == 0) {
                sRedDt[0][N_STACK][0][warp] = dta;
                sRedDt[0][N_STACK][1][warp] = dtb;
            }
        } else {
            #pragma unroll
            for (int q = 0; q < NQ; ++q) {
                const float* qp = Queries + static_cast<int64_t>(q) * D;
                float qa0[8], qa1[8];
                load_f32x8_const(qp + off_a0, qa0);
                load_f32x8_const(qp + off_a1, qa1);
                float2 qbf = ld_const_f2(qp + off_b);
                float dt0[8], dt1[8];
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    dt0[i] = __fmul_rn(qa0[i], lx0[i]);
                    dt1[i] = __fmul_rn(qa1[i], lx1[i]);
                }
                float dta = tree16_sum(dt0, dt1);
                float dtb = __fadd_rn(__fmul_rn(qbf.x, lxb[0]), __fmul_rn(qbf.y, lxb[1]));
                warp_sum_bfly2(dta, dtb);
                if (lane == 0) {
                    sRedDt[q][N_STACK][0][warp] = dta;
                    sRedDt[q][N_STACK][1][warp] = dtb;
                }
            }
        }
    }

    __syncthreads();

    const int l8 = lane & 7;
    const int half = (lane >> 3) & 1;
    float rstd_arr[NS];
    #pragma unroll
    for (int n = 0; n < NS; ++n) {
        float v = warp8_sum_bfly(sRedSq[n][half][l8]);
        float sa = __shfl_sync(0xFFFFFFFFu, v, 0);
        float sb = __shfl_sync(0xFFFFFFFFu, v, 8);
        rstd_arr[n] = rsqrt_approx(fmaf(__fadd_rn(sa, sb), INV_D, eps));
    }

    #pragma unroll
    for (int q = 0; q < NQ; ++q) {
        float my_lg = -INFINITY;
        float my_ir = 0.0f;
        #pragma unroll
        for (int n = 0; n < NS; ++n) {
            float v = warp8_sum_bfly(sRedDt[q][n][half][l8]);
            float da = __shfl_sync(0xFFFFFFFFu, v, 0);
            float db = __shfl_sync(0xFFFFFFFFu, v, 8);
            if (lane == n) {
                my_lg = __fmul_rn(__fadd_rn(da, db), rstd_arr[n]);
                my_ir = rstd_arr[n];
            }
        }
        float mx = my_lg;
        if constexpr (BLOCK_N >= 16) mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, 8));
        if constexpr (BLOCK_N >= 8)  mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, 4));
        if constexpr (BLOCK_N >= 4)  mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, 2));
        if constexpr (BLOCK_N >= 2)  mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, 1));
        float p = (lane < NS) ? tldevice_exp_sub(my_lg, mx) : 0.0f;
        float sp = p;
        if constexpr (BLOCK_N >= 16) sp = __fadd_rn(sp, __shfl_xor_sync(0xFFFFFFFFu, sp, 8));
        if constexpr (BLOCK_N >= 8)  sp = __fadd_rn(sp, __shfl_xor_sync(0xFFFFFFFFu, sp, 4));
        if constexpr (BLOCK_N >= 4)  sp = __fadd_rn(sp, __shfl_xor_sync(0xFFFFFFFFu, sp, 2));
        if constexpr (BLOCK_N >= 2)  sp = __fadd_rn(sp, __shfl_xor_sync(0xFFFFFFFFu, sp, 1));
        float inv_sum = __fdiv_rn(1.0f, __shfl_sync(0xFFFFFFFFu, sp, 0));
        if (normalize_values != 0) p = __fmul_rn(p, my_ir);
        float probs[NS];
        #pragma unroll
        for (int n = 0; n < NS; ++n) probs[n] = __shfl_sync(0xFFFFFFFFu, p, n);

        const int64_t q_off = static_cast<int64_t>(q) * D;
        uint4 nw_r0 = ld_const_u4(NormW + q_off + off_a0);
        uint4 nw_r1 = ld_const_u4(NormW + q_off + off_a1);
        uint32_t nw_rb = ld_const_u32(NormW + q_off + off_b);

        float acc0[8] = {0,0,0,0,0,0,0,0};
        float acc1[8] = {0,0,0,0,0,0,0,0};
        float accb[2] = {0,0};

        #pragma unroll
        for (int n = 0; n < N_STACK; ++n) {
            const float pn = probs[n];
            uint4 r0 = sStack_a0[n * 256 + tid];
            uint4 r1 = sStack_a1[n * 256 + tid];
            uint32_t rb = sStack_b[n * 256 + tid];
            const __nv_bfloat162* h0 = reinterpret_cast<const __nv_bfloat162*>(&r0);
            const __nv_bfloat162* h1 = reinterpret_cast<const __nv_bfloat162*>(&r1);
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                float2 f0 = __bfloat1622float2(h0[i]);
                acc0[2*i]   = fmaf(pn, f0.x, acc0[2*i]);
                acc0[2*i+1] = fmaf(pn, f0.y, acc0[2*i+1]);
                float2 f1 = __bfloat1622float2(h1[i]);
                acc1[2*i]   = fmaf(pn, f1.x, acc1[2*i]);
                acc1[2*i+1] = fmaf(pn, f1.y, acc1[2*i+1]);
            }
            float2 fb = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&rb));
            accb[0] = fmaf(pn, fb.x, accb[0]);
            accb[1] = fmaf(pn, fb.y, accb[1]);
        }
        {
            const float pn = probs[N_STACK];
            if constexpr (HAS_ADDEND) {
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    acc0[i] = fmaf(pn, last_f0[i], acc0[i]);
                    acc1[i] = fmaf(pn, last_f1[i], acc1[i]);
                }
                accb[0] = fmaf(pn, last_fb[0], accb[0]);
                accb[1] = fmaf(pn, last_fb[1], accb[1]);
            } else {
                const __nv_bfloat162* h0 = reinterpret_cast<const __nv_bfloat162*>(&last_r0);
                const __nv_bfloat162* h1 = reinterpret_cast<const __nv_bfloat162*>(&last_r1);
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    float2 f0 = __bfloat1622float2(h0[i]);
                    acc0[2*i]   = fmaf(pn, f0.x, acc0[2*i]);
                    acc0[2*i+1] = fmaf(pn, f0.y, acc0[2*i+1]);
                    float2 f1 = __bfloat1622float2(h1[i]);
                    acc1[2*i]   = fmaf(pn, f1.x, acc1[2*i]);
                    acc1[2*i+1] = fmaf(pn, f1.y, acc1[2*i+1]);
                }
                float2 fb = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&last_rb));
                accb[0] = fmaf(pn, fb.x, accb[0]);
                accb[1] = fmaf(pn, fb.y, accb[1]);
            }
        }

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            acc0[i] = __fmul_rn(acc0[i], inv_sum);
            acc1[i] = __fmul_rn(acc1[i], inv_sum);
        }
        accb[0] = __fmul_rn(accb[0], inv_sum);
        accb[1] = __fmul_rn(accb[1], inv_sum);

#ifdef INKLING_PTXAS133_STRICT
        constexpr int P2_MODE = 0;
#else
        constexpr int P2_MODE = (N_STACK == 8) ? 3 : (HAS_ADDEND ? 1 : 0);
#endif
        float sqa = tree16_sum_sq<P2_MODE>(acc0, acc1);
        float sqb = __fadd_rn(__fmul_rn(accb[0], accb[0]), __fmul_rn(accb[1], accb[1]));
        warp_sum_bfly2(sqa, sqb);
        if (lane == 0) {
            sSq2[0][warp] = sqa;
            sSq2[1][warp] = sqb;
        }
        __syncthreads();

        float v2 = warp8_sum_bfly(sSq2[half][l8]);
        float sa = __shfl_sync(0xFFFFFFFFu, v2, 0);
        float sb = __shfl_sync(0xFFFFFFFFu, v2, 8);
        float out_rstd = rsqrt_approx(__fadd_rn(__fdiv_rn(__fadd_rn(sa, sb), static_cast<float>(D)), norm_eps));

        float wa0[8], wa1[8], wb[2];
        unpack_bf16x8(nw_r0, wa0);
        unpack_bf16x8(nw_r1, wa1);
        unpack_bf16x2(nw_rb, wb);

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            acc0[i] = __fmul_rn(__fmul_rn(acc0[i], out_rstd), wa0[i]);
            acc1[i] = __fmul_rn(__fmul_rn(acc1[i], out_rstd), wa1[i]);
        }
        accb[0] = __fmul_rn(__fmul_rn(accb[0], out_rstd), wb[0]);
        accb[1] = __fmul_rn(__fmul_rn(accb[1], out_rstd), wb[1]);

        __nv_bfloat16* out_p = Out + static_cast<int64_t>(q) * T * D + row_off;
        *reinterpret_cast<uint4*>(out_p + off_a0)   = pack_f32x8_to_bf16_u4(acc0);
        *reinterpret_cast<uint4*>(out_p + off_a1)   = pack_f32x8_to_bf16_u4(acc1);
        *reinterpret_cast<uint32_t*>(out_p + off_b) = pack_f32x2_to_bf16_u32(accb);

        if constexpr (NQ > 1) {
            if (q + 1 < NQ) __syncthreads();
        }
    }

    if constexpr (PDL) {
        if (early_pdl == 0) {
            __syncthreads();
            if (tid == 0) gdc_launch_dependents_v14();
        }
    }
}

// =========================================================================
// KERNEL 6 (DUAL-REGIME V16 REGIME B, T*nq > 256): `fused_attnres_rmsnorm_v16_blockd_kernel`
//   Exact 0-ULP match to Triton `_attn_res_kernel` (BLOCK_D=1024, 128 thr, Order-A FFMA tree)
//   for C=32 M=128 NQ=3 (T*nq=384) and prefill (T>256), supporting N_STACK=1..10, NQ in {1,3},
//   HAS_ADDEND in {0,1}, normalize_values in {0,1}, NLivePtr, and #206-safe PDL!
//   SMEM budget: pins ONLY sStack (N_STACK * 9216 B <= 92,160 B = 90.0 KiB < 99 KiB on GB10
//   even at N_STACK=10!) while keeping Last (+ Addend) L1-hot across Pass 1 -> Pass 2.
// =========================================================================
template <int N_STACK, int NQ, bool HAS_ADDEND, bool PDL>
__global__ void __launch_bounds__(128, 1)
fused_attnres_rmsnorm_v16_blockd_kernel(
    const __nv_bfloat16* __restrict__ Stack,
    const __nv_bfloat16* __restrict__ Last,
    const __nv_bfloat16* __restrict__ Addend,
    const float* __restrict__ Queries,
    const __nv_bfloat16* __restrict__ NormW,
    __nv_bfloat16* __restrict__ Out,
    const int32_t* __restrict__ NLivePtr,
    int64_t stride_s,
    int T,
    float eps,
    float norm_eps,
    int normalize_values,
    int early_pdl
) {
    constexpr int D = 4608;
    constexpr int U4_PER_ROW = D / 8;  // 576
    constexpr int NS = N_STACK + 1;
    constexpr float INV_D = 1.0f / static_cast<float>(D);
    constexpr int BLOCK_N = (NS <= 2) ? 2 : (NS <= 4) ? 4 : (NS <= 8) ? 8 : 16;

    const int t = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int64_t row_off = static_cast<int64_t>(t) * D;

    int nl = T;
    if (NLivePtr != nullptr) {
        const int raw_nl = *NLivePtr;
        if (raw_nl >= 1 && raw_nl <= T) nl = raw_nl;
    }
    if (t >= nl) {
        if constexpr (PDL) {
            gdc_wait_v14();
        }
        const uint4 z4 = make_uint4(0u, 0u, 0u, 0u);
        #pragma unroll
        for (int q = 0; q < NQ; ++q) {
            __nv_bfloat16* out_p = Out + static_cast<int64_t>(q) * T * D + row_off;
            #pragma unroll
            for (int c = 0; c < 5; ++c) {
                const int d_base = c * 1024 + tid * 8;
                if (c < 4 || tid < 64) {
                    *reinterpret_cast<uint4*>(out_p + d_base) = z4;
                }
            }
        }
        if constexpr (PDL) {
            __syncthreads();
            if (tid == 0) gdc_launch_dependents_v14();
        }
        return;
    }

    extern __shared__ uint4 dyn_raw_u4[];
    uint4* sStack = dyn_raw_u4;  // N_STACK * 576 * 16 B <= 92,160 B (< 99 KiB for N_STACK=1..10!)

    alignas(16) __shared__ float sRed[NS][1 + NQ][4];
    alignas(16) __shared__ float sSq2[5][4];

    // PRE-WAIT: Prefetch NormW and stream Stack[0..N_STACK-1] into SMEM + Pass-1 reductions.
    // (Last & Addend may be produced by immediately preceding `short_conv` -> strictly POST-WAIT!)
    if (t < 4 && tid < 72) {
        #pragma unroll
        for (int q = 0; q < NQ; ++q) {
            prefetch_l2(NormW + static_cast<int64_t>(q) * D + tid * 64);
        }
    }

    #pragma unroll
    for (int n = 0; n < N_STACK; ++n) {
        const __nv_bfloat16* sp = Stack + static_cast<int64_t>(n) * stride_s + row_off;
        uint4* dst_s = sStack + n * U4_PER_ROW;
        uint4 raws[5];
        #pragma unroll
        for (int c = 0; c < 5; ++c) {
            if (c < 4 || tid < 64) raws[c] = ld_global_cs_u4(sp + c * 1024 + tid * 8);
        }
        float sum_sq[8] = {0,0,0,0,0,0,0,0};
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
                dst_s[d_base >> 3] = raws[c];
                float x[8];
                unpack_bf16x8(raws[c], x);
                #pragma unroll
                for (int i = 0; i < 8; ++i) sum_sq[i] = fmaf(x[i], x[i], sum_sq[i]);
                #pragma unroll
                for (int q = 0; q < NQ; ++q) {
                    float qv[8];
                    load_f32x8_const(Queries + static_cast<int64_t>(q) * D + d_base, qv);
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) dots[q][i] = fmaf(qv[i], x[i], dots[q][i]);
                }
            }
        }
        float w_sq = warp_sum_bfly(tree8_sum(sum_sq));
        if (lane == 0) sRed[n][0][warp] = w_sq;
        #pragma unroll
        for (int q = 0; q < NQ; ++q) {
            float w_dt = warp_sum_bfly(tree8_sum(dots[q]));
            if (lane == 0) sRed[n][1 + q][warp] = w_dt;
        }
    }

    // #206-Safe PDL Synchronization Point before reading Last/Addend!
    if constexpr (PDL) {
        gdc_wait_v14();
        if (early_pdl != 0) {
            __syncthreads();
            if (tid == 0) gdc_launch_dependents_v14();
        }
    }

    // POST-WAIT: Pass-1 on Last (+ Addend), L1-cached (.ca) for instant Pass-2 reuse (0 SMEM overflow!)
    {
        float sum_sq[8] = {0,0,0,0,0,0,0,0};
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
                float x[8];
                unpack_bf16x8(ld_global_ca_u4(Last + row_off + d_base), x);
                if constexpr (HAS_ADDEND) {
                    float a[8];
                    unpack_bf16x8(ld_global_ca_u4(Addend + row_off + d_base), a);
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) x[i] = __fadd_rn(x[i], a[i]);
                }
                #pragma unroll
                for (int i = 0; i < 8; ++i) sum_sq[i] = fmaf(x[i], x[i], sum_sq[i]);
                #pragma unroll
                for (int q = 0; q < NQ; ++q) {
                    float qv[8];
                    load_f32x8_const(Queries + static_cast<int64_t>(q) * D + d_base, qv);
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) dots[q][i] = fmaf(qv[i], x[i], dots[q][i]);
                }
            }
        }
        float w_sq = warp_sum_bfly(tree8_sum(sum_sq));
        if (lane == 0) sRed[N_STACK][0][warp] = w_sq;
        #pragma unroll
        for (int q = 0; q < NQ; ++q) {
            float w_dt = warp_sum_bfly(tree8_sum(dots[q]));
            if (lane == 0) sRed[N_STACK][1 + q][warp] = w_dt;
        }
    }

    __syncthreads();

    const int l4 = lane & 3;
    float rstd_arr[NS];
    #pragma unroll
    for (int n = 0; n < NS; ++n) {
        float tot_sq = warp4_sum_bfly(sRed[n][0][l4]);
        rstd_arr[n] = rsqrt_approx(fmaf(tot_sq, INV_D, eps));
    }

    const bool keep_out_l2 = (T <= 512);
    #pragma unroll
    for (int q = 0; q < NQ; ++q) {
        float my_lg = -INFINITY;
        float my_ir = 0.0f;
        #pragma unroll
        for (int n = 0; n < NS; ++n) {
            float tot_dt = warp4_sum_bfly(sRed[n][1 + q][l4]);
            if (lane == n) {
                my_lg = __fmul_rn(tot_dt, rstd_arr[n]);
                my_ir = rstd_arr[n];
            }
        }
        float mx = my_lg;
        if constexpr (BLOCK_N >= 16) mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, 8));
        if constexpr (BLOCK_N >= 8)  mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, 4));
        if constexpr (BLOCK_N >= 4)  mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, 2));
        if constexpr (BLOCK_N >= 2)  mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, 1));
        float p = (lane < NS) ? tldevice_exp_sub(my_lg, mx) : 0.0f;
        float sp = p;
        if constexpr (BLOCK_N >= 16) sp = __fadd_rn(sp, __shfl_xor_sync(0xFFFFFFFFu, sp, 8));
        if constexpr (BLOCK_N >= 8)  sp = __fadd_rn(sp, __shfl_xor_sync(0xFFFFFFFFu, sp, 4));
        if constexpr (BLOCK_N >= 4)  sp = __fadd_rn(sp, __shfl_xor_sync(0xFFFFFFFFu, sp, 2));
        if constexpr (BLOCK_N >= 2)  sp = __fadd_rn(sp, __shfl_xor_sync(0xFFFFFFFFu, sp, 1));
        float inv_sum = __fdiv_rn(1.0f, __shfl_sync(0xFFFFFFFFu, sp, 0));
        if (normalize_values != 0) p = __fmul_rn(p, my_ir);
        float probs[NS];
        #pragma unroll
        for (int n = 0; n < NS; ++n) probs[n] = __shfl_sync(0xFFFFFFFFu, p, n);

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
                    const float pn = probs[n];
                    float x[8];
                    unpack_bf16x8(sStack[n * U4_PER_ROW + u4_idx], x);
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) acc[c][i] = fmaf(pn, x[i], acc[c][i]);
                }
                {
                    const float pn = probs[N_STACK];
                    float x[8];
                    unpack_bf16x8(ld_global_ca_u4(Last + row_off + d_base), x);
                    if constexpr (HAS_ADDEND) {
                        float a[8];
                        unpack_bf16x8(ld_global_ca_u4(Addend + row_off + d_base), a);
                        #pragma unroll
                        for (int i = 0; i < 8; ++i) x[i] = __fadd_rn(x[i], a[i]);
                    }
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) acc[c][i] = fmaf(pn, x[i], acc[c][i]);
                }
                #pragma unroll
                for (int i = 0; i < 8; ++i) acc[c][i] = __fmul_rn(acc[c][i], inv_sum);
#ifdef INKLING_PTXAS133_STRICT
                float sq_v[8];
                #pragma unroll
                for (int i = 0; i < 8; ++i) sq_v[i] = __fmul_rn(acc[c][i], acc[c][i]);
                sq_sum = tree8_sum(sq_v);
#else
                float e01 = fmaf(acc[c][0], acc[c][0], __fmul_rn(acc[c][2], acc[c][2]));
                float o01 = fmaf(acc[c][1], acc[c][1], __fmul_rn(acc[c][3], acc[c][3]));
                float e23 = fmaf(acc[c][4], acc[c][4], __fmul_rn(acc[c][6], acc[c][6]));
                float o23 = fmaf(acc[c][5], acc[c][5], __fmul_rn(acc[c][7], acc[c][7]));
                sq_sum = __fadd_rn(__fadd_rn(e01, e23), __fadd_rn(o01, o23));
#endif
            }
            float w_sq = warp_sum_bfly(sq_sum);
            if (lane == 0) sSq2[c][warp] = w_sq;
        }

        __syncthreads();

        float tot_sum_sq = 0.0f;
        #pragma unroll
        for (int c = 0; c < 5; ++c) {
            tot_sum_sq = __fadd_rn(tot_sum_sq, warp4_sum_bfly(sSq2[c][l4]));
        }
        float out_rstd = rsqrt_approx(__fadd_rn(__fdiv_rn(tot_sum_sq, static_cast<float>(D)), norm_eps));

        __nv_bfloat16* out_p = Out + static_cast<int64_t>(q) * T * D + row_off;
        const __nv_bfloat16* nw_p = NormW + static_cast<int64_t>(q) * D;
        #pragma unroll
        for (int c = 0; c < 5; ++c) {
            const int d_base = c * 1024 + tid * 8;
            if (c < 4 || tid < 64) {
                float w[8];
                unpack_bf16x8(ld_const_u4(nw_p + d_base), w);
                float res[8];
                #pragma unroll
                for (int i = 0; i < 8; ++i) res[i] = __fmul_rn(__fmul_rn(acc[c][i], out_rstd), w[i]);
                uint4 out_u4 = pack_f32x8_to_bf16_u4(res);
                if (keep_out_l2) *reinterpret_cast<uint4*>(out_p + d_base) = out_u4;
                else             st_global_cs_u4(out_p + d_base, out_u4);
            }
        }
        if constexpr (NQ > 1) {
            if (q + 1 < NQ) __syncthreads();
        }
    }

    if constexpr (PDL) {
        if (early_pdl == 0) {
            __syncthreads();
            if (tid == 0) gdc_launch_dependents_v14();
        }
    }
}

// =========================================================================
// SMEM OPT-IN & DISPATCH HELPERS
// =========================================================================

static bool g_c32_smem_inited = false;
static bool g_c48_smem_inited = false;
static bool g_ar14_smem_inited = false;
static bool g_ar16_blockd_smem_inited = false;

template <int N_STACK, int NQ, bool HCR, bool HCA, bool HML>
static inline cudaError_t init_row_smem_one() {
    constexpr size_t smem_bytes = static_cast<size_t>(N_STACK) * 9216;
    if constexpr (smem_bytes + 4096u > 49152u) {
        return cudaFuncSetAttribute(
            fused_sconv_ar_c32_row_kernel<N_STACK, NQ, HCR, HCA, HML>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem_bytes)
        );
    }
    return cudaSuccess;
}

template <int N_STACK, int NQ, bool HCR, bool HCA, bool HML>
static inline cudaError_t init_blockd_smem_one() {
    constexpr int D = 4608;
    constexpr size_t smem_bytes = static_cast<size_t>(N_STACK) * D * 2 + (HML ? static_cast<size_t>(D) * 4 : static_cast<size_t>(D) * 2);
    if constexpr (smem_bytes + 4096u > 49152u) {
        return cudaFuncSetAttribute(
            fused_sconv_ar_c32_blockd_kernel<N_STACK, NQ, HCR, HCA, HML>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem_bytes)
        );
    }
    return cudaSuccess;
}

template <int N_STACK>
static inline cudaError_t init_c32_smem_ns() {
    cudaError_t e = cudaSuccess;
    #define TRY_SMEM(fn) do { e = (fn); if (e != cudaSuccess) return e; } while (0)
    TRY_SMEM((init_row_smem_one<N_STACK, 1, false, false, false>()));
    TRY_SMEM((init_row_smem_one<N_STACK, 1, false, false, true>()));
    TRY_SMEM((init_row_smem_one<N_STACK, 1, false, true,  false>()));
    TRY_SMEM((init_row_smem_one<N_STACK, 1, true,  true,  false>()));
    TRY_SMEM((init_row_smem_one<N_STACK, 3, false, true,  false>()));
    TRY_SMEM((init_row_smem_one<N_STACK, 3, true,  true,  false>()));
    TRY_SMEM((init_blockd_smem_one<N_STACK, 1, false, false, false>()));
    TRY_SMEM((init_blockd_smem_one<N_STACK, 1, false, false, true>()));
    TRY_SMEM((init_blockd_smem_one<N_STACK, 1, false, true,  false>()));
    TRY_SMEM((init_blockd_smem_one<N_STACK, 1, true,  true,  false>()));
    TRY_SMEM((init_blockd_smem_one<N_STACK, 3, false, true,  false>()));
    TRY_SMEM((init_blockd_smem_one<N_STACK, 3, true,  true,  false>()));
    #undef TRY_SMEM
    return cudaSuccess;
}

template <int N_STACK, int NQ, bool HCR, bool HCA, bool HML>
static inline cudaError_t init_c48_smem_one() {
    constexpr size_t smem_bytes = static_cast<size_t>(N_STACK) * 9216;
    if constexpr (smem_bytes + 4096u > 49152u) {
        cudaError_t e = cudaFuncSetAttribute(
            fused_sconv_ar_c48_kernel_v14<N_STACK, NQ, HCR, HCA, HML, false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem_bytes)
        );
        if (e != cudaSuccess) return e;
        e = cudaFuncSetAttribute(
            fused_sconv_ar_c48_kernel_v14<N_STACK, NQ, HCR, HCA, HML, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem_bytes)
        );
        if (e != cudaSuccess) return e;
    }
    return cudaSuccess;
}

template <int N_STACK>
static inline cudaError_t init_c48_smem_ns() {
    cudaError_t e = cudaSuccess;
    #define TRY_C48_SMEM(fn) do { e = (fn); if (e != cudaSuccess) return e; } while (0)
    TRY_C48_SMEM((init_c48_smem_one<N_STACK, 1, false, false, false>()));
    TRY_C48_SMEM((init_c48_smem_one<N_STACK, 1, false, false, true>()));
    TRY_C48_SMEM((init_c48_smem_one<N_STACK, 1, false, true,  false>()));
    TRY_C48_SMEM((init_c48_smem_one<N_STACK, 1, true,  true,  false>()));
    TRY_C48_SMEM((init_c48_smem_one<N_STACK, 3, false, false, false>()));
    TRY_C48_SMEM((init_c48_smem_one<N_STACK, 3, false, false, true>()));
    TRY_C48_SMEM((init_c48_smem_one<N_STACK, 3, false, true,  false>()));
    TRY_C48_SMEM((init_c48_smem_one<N_STACK, 3, true,  true,  false>()));
    #undef TRY_C48_SMEM
    return cudaSuccess;
}

template <int N_STACK, int NQ, bool HA>
static inline cudaError_t init_ar14_smem_one() {
    constexpr size_t smem_bytes = static_cast<size_t>(N_STACK) * 9216;
    if constexpr (smem_bytes + 4096u > 49152u) {
        cudaError_t e = cudaFuncSetAttribute(
            attn_res_exact_v14_kernel<N_STACK, NQ, HA, false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem_bytes)
        );
        if (e != cudaSuccess) return e;
        e = cudaFuncSetAttribute(
            attn_res_exact_v14_kernel<N_STACK, NQ, HA, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem_bytes)
        );
        if (e != cudaSuccess) return e;
    }
    return cudaSuccess;
}

template <int N_STACK>
static inline cudaError_t init_ar14_smem_ns() {
    cudaError_t e = cudaSuccess;
    #define TRY_AR14_SMEM(fn) do { e = (fn); if (e != cudaSuccess) return e; } while (0)
    TRY_AR14_SMEM((init_ar14_smem_one<N_STACK, 1, false>()));
    TRY_AR14_SMEM((init_ar14_smem_one<N_STACK, 1, true>()));
    TRY_AR14_SMEM((init_ar14_smem_one<N_STACK, 3, false>()));
    TRY_AR14_SMEM((init_ar14_smem_one<N_STACK, 3, true>()));
    #undef TRY_AR14_SMEM
    return cudaSuccess;
}

template <int N_STACK, int NQ, bool HA>
static inline cudaError_t init_ar16_blockd_smem_one() {
    constexpr size_t smem_bytes = static_cast<size_t>(N_STACK) * 9216;
    if constexpr (smem_bytes + 4096u > 49152u) {
        cudaError_t e = cudaFuncSetAttribute(
            fused_attnres_rmsnorm_v16_blockd_kernel<N_STACK, NQ, HA, false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem_bytes)
        );
        if (e != cudaSuccess) return e;
        e = cudaFuncSetAttribute(
            fused_attnres_rmsnorm_v16_blockd_kernel<N_STACK, NQ, HA, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem_bytes)
        );
        if (e != cudaSuccess) return e;
    }
    return cudaSuccess;
}

template <int N_STACK>
static inline cudaError_t init_ar16_blockd_smem_ns() {
    cudaError_t e = cudaSuccess;
    #define TRY_AR16B_SMEM(fn) do { e = (fn); if (e != cudaSuccess) return e; } while (0)
    TRY_AR16B_SMEM((init_ar16_blockd_smem_one<N_STACK, 1, false>()));
    TRY_AR16B_SMEM((init_ar16_blockd_smem_one<N_STACK, 1, true>()));
    TRY_AR16B_SMEM((init_ar16_blockd_smem_one<N_STACK, 3, false>()));
    TRY_AR16B_SMEM((init_ar16_blockd_smem_one<N_STACK, 3, true>()));
    #undef TRY_AR16B_SMEM
    return cudaSuccess;
}

template <int N_STACK, int NQ, bool HCR, bool HCA, bool HML>
static inline cudaError_t launch_c32_row_one(
    int T, cudaStream_t stream,
    const __nv_bfloat16* X, const __nv_bfloat16* CW, __nv_bfloat16* Cache,
    __nv_bfloat16* ConvOut, const __nv_bfloat16* ConvRes, const __nv_bfloat16* ConvAdd, __nv_bfloat16* AuxOut,
    const int64_t* Pos, const int32_t* SeqIdx, const int64_t* Slot, const int32_t* BT, const int32_t* QStart,
    const __nv_bfloat16* Stack, const __nv_bfloat16* MixLast,
    const float* Queries, const __nv_bfloat16* NormW, __nv_bfloat16* Out,
    int64_t stride_x, int64_t stride_c_blk, int64_t stride_c_n, int64_t stride_bt,
    int max_blocks, int N_CACHE, int N_SHIFT, int stream_offset, int64_t stride_s,
    float scale, float eps, float norm_eps
) {
    constexpr size_t smem_bytes = static_cast<size_t>(N_STACK) * 9216;
    static bool optin_done = false;
    auto k = fused_sconv_ar_c32_row_kernel<N_STACK, NQ, HCR, HCA, HML>;
    cudaError_t se = ensure_smem_optin(k, smem_bytes, g_c32_smem_inited, optin_done, stream);
    if (se != cudaSuccess) return se;
    k<<<dim3(static_cast<unsigned>(T), 1, 1), dim3(256, 1, 1), smem_bytes, stream>>>(
        X, CW, Cache, ConvOut, ConvRes, ConvAdd, AuxOut,
        Pos, SeqIdx, Slot, BT, QStart,
        Stack, MixLast, Queries, NormW, Out,
        stride_x, stride_c_blk, stride_c_n, stride_bt,
        max_blocks, N_CACHE, N_SHIFT, stream_offset, stride_s,
        T, scale, eps, norm_eps
    );
    return cudaGetLastError();
}

template <int N_STACK, int NQ, bool HCR, bool HCA, bool HML>
static inline cudaError_t launch_c32_blockd_one(
    int T, cudaStream_t stream,
    const __nv_bfloat16* X, const __nv_bfloat16* CW, __nv_bfloat16* Cache,
    __nv_bfloat16* ConvOut, const __nv_bfloat16* ConvRes, const __nv_bfloat16* ConvAdd, __nv_bfloat16* AuxOut,
    const int64_t* Pos, const int32_t* SeqIdx, const int64_t* Slot, const int32_t* BT, const int32_t* QStart,
    const __nv_bfloat16* Stack, const __nv_bfloat16* MixLast,
    const float* Queries, const __nv_bfloat16* NormW, __nv_bfloat16* Out,
    int64_t stride_x, int64_t stride_c_blk, int64_t stride_c_n, int64_t stride_bt,
    int max_blocks, int N_CACHE, int N_SHIFT, int stream_offset, int64_t stride_s,
    float scale, float eps, float norm_eps
) {
    constexpr int D = 4608;
    constexpr size_t smem_bytes = static_cast<size_t>(N_STACK) * D * 2 + (HML ? static_cast<size_t>(D) * 4 : static_cast<size_t>(D) * 2);
    static bool optin_done = false;
    auto k = fused_sconv_ar_c32_blockd_kernel<N_STACK, NQ, HCR, HCA, HML>;
    cudaError_t se = ensure_smem_optin(k, smem_bytes, g_c32_smem_inited, optin_done, stream);
    if (se != cudaSuccess) return se;
    k<<<dim3(static_cast<unsigned>(T), 1, 1), dim3(128, 1, 1), smem_bytes, stream>>>(
        X, CW, Cache, ConvOut, ConvRes, ConvAdd, AuxOut,
        Pos, SeqIdx, Slot, BT, QStart,
        Stack, MixLast, Queries, NormW, Out,
        stride_x, stride_c_blk, stride_c_n, stride_bt,
        max_blocks, N_CACHE, N_SHIFT, stream_offset, stride_s,
        T, scale, eps, norm_eps
    );
    return cudaGetLastError();
}

template <int N_STACK, bool ROW_REGIME>
static cudaError_t dispatch_c32_ns(
    int nq, int T, cudaStream_t stream,
    const __nv_bfloat16* X, const __nv_bfloat16* CW, __nv_bfloat16* Cache,
    __nv_bfloat16* ConvOut, const __nv_bfloat16* ConvRes, const __nv_bfloat16* ConvAdd, __nv_bfloat16* AuxOut,
    const int64_t* Pos, const int32_t* SeqIdx, const int64_t* Slot, const int32_t* BT, const int32_t* QStart,
    const __nv_bfloat16* Stack, const __nv_bfloat16* MixLast,
    const float* Queries, const __nv_bfloat16* NormW, __nv_bfloat16* Out,
    int64_t stride_x, int64_t stride_c_blk, int64_t stride_c_n, int64_t stride_bt,
    int max_blocks, int N_CACHE, int N_SHIFT, int stream_offset, int64_t stride_s,
    float scale, float eps, float norm_eps,
    bool hcr, bool hca, bool hml
) {
    #define GO_C32(NQ_V, HCR_V, HCA_V, HML_V) \
        if constexpr (ROW_REGIME) { \
            return launch_c32_row_one<N_STACK, NQ_V, HCR_V, HCA_V, HML_V>( \
                T, stream, X, CW, Cache, ConvOut, ConvRes, ConvAdd, AuxOut, \
                Pos, SeqIdx, Slot, BT, QStart, Stack, MixLast, Queries, NormW, Out, \
                stride_x, stride_c_blk, stride_c_n, stride_bt, max_blocks, N_CACHE, N_SHIFT, stream_offset, stride_s, \
                scale, eps, norm_eps); \
        } else { \
            return launch_c32_blockd_one<N_STACK, NQ_V, HCR_V, HCA_V, HML_V>( \
                T, stream, X, CW, Cache, ConvOut, ConvRes, ConvAdd, AuxOut, \
                Pos, SeqIdx, Slot, BT, QStart, Stack, MixLast, Queries, NormW, Out, \
                stride_x, stride_c_blk, stride_c_n, stride_bt, max_blocks, N_CACHE, N_SHIFT, stream_offset, stride_s, \
                scale, eps, norm_eps); \
        }
    if (nq == 1) {
        if (!hca && !hcr && !hml) { GO_C32(1, false, false, false); }
        if (!hca && !hcr &&  hml) { GO_C32(1, false, false, true);  }
        if ( hca && !hcr && !hml) { GO_C32(1, false, true,  false); }
        if ( hca &&  hcr && !hml) { GO_C32(1, true,  true,  false); }
    } else if (nq == 3) {
        if ( hca && !hcr && !hml) { GO_C32(3, false, true,  false); }
        if ( hca &&  hcr && !hml) { GO_C32(3, true,  true,  false); }
    }
    #undef GO_C32
    return cudaErrorInvalidValue;
}

template <int N_STACK, int NQ, bool HCR, bool HCA, bool HML, bool PDL>
static inline cudaError_t launch_c48_one_pdl(
    int T, cudaStream_t stream,
    const __nv_bfloat16* X, const __nv_bfloat16* CW, __nv_bfloat16* Cache,
    __nv_bfloat16* ConvOut, const __nv_bfloat16* ConvRes, const __nv_bfloat16* ConvAdd, __nv_bfloat16* AuxOut,
    const int64_t* Pos, const int32_t* SeqIdx, const int64_t* Slot, const int32_t* BT, const int32_t* QStart,
    const __nv_bfloat16* Stack, const __nv_bfloat16* MixLast,
    const float* Queries, const __nv_bfloat16* NormW, __nv_bfloat16* Out,
    const int32_t* NLivePtr, const int32_t* TreeAnc,
    int64_t stride_x, int64_t stride_c_blk, int64_t stride_c_n, int64_t stride_bt,
    int max_blocks, int N_CACHE, int N_SHIFT, int stream_offset, int64_t stride_s,
    float scale, float eps, float norm_eps, int normalize_values, int early_pdl
) {
    constexpr size_t smem_bytes = static_cast<size_t>(N_STACK) * 9216;
    static bool optin_done = false;
    auto k = fused_sconv_ar_c48_kernel_v14<N_STACK, NQ, HCR, HCA, HML, PDL>;
    cudaError_t se = ensure_smem_optin(k, smem_bytes, g_c48_smem_inited, optin_done, stream);
    if (se != cudaSuccess) return se;
    const dim3 grid(static_cast<unsigned>(T), 1, 1);
    const dim3 block(256, 1, 1);
    if constexpr (!PDL) {
        k<<<grid, block, smem_bytes, stream>>>(
            X, CW, Cache, ConvOut, ConvRes, ConvAdd, AuxOut,
            Pos, SeqIdx, Slot, BT, QStart,
            Stack, MixLast, Queries, NormW, Out,
            NLivePtr, TreeAnc,
            stride_x, stride_c_blk, stride_c_n, stride_bt,
            max_blocks, N_CACHE, N_SHIFT, stream_offset, stride_s,
            T, scale, eps, norm_eps, normalize_values, early_pdl
        );
        return cudaGetLastError();
    } else {
        cudaLaunchAttribute attr[1];
        attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr[0].val.programmaticStreamSerializationAllowed = 1;
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = grid;
        cfg.blockDim = block;
        cfg.dynamicSmemBytes = smem_bytes;
        cfg.stream = stream;
        cfg.attrs = attr;
        cfg.numAttrs = 1;
        return cudaLaunchKernelEx(
            &cfg, k,
            X, CW, Cache, ConvOut, ConvRes, ConvAdd, AuxOut,
            Pos, SeqIdx, Slot, BT, QStart,
            Stack, MixLast, Queries, NormW, Out,
            NLivePtr, TreeAnc,
            stride_x, stride_c_blk, stride_c_n, stride_bt,
            max_blocks, N_CACHE, N_SHIFT, stream_offset, stride_s,
            T, scale, eps, norm_eps, normalize_values, early_pdl
        );
    }
}

template <int N_STACK>
static cudaError_t dispatch_c48_ns(
    int nq, int T, cudaStream_t stream,
    const __nv_bfloat16* X, const __nv_bfloat16* CW, __nv_bfloat16* Cache,
    __nv_bfloat16* ConvOut, const __nv_bfloat16* ConvRes, const __nv_bfloat16* ConvAdd, __nv_bfloat16* AuxOut,
    const int64_t* Pos, const int32_t* SeqIdx, const int64_t* Slot, const int32_t* BT, const int32_t* QStart,
    const __nv_bfloat16* Stack, const __nv_bfloat16* MixLast,
    const float* Queries, const __nv_bfloat16* NormW, __nv_bfloat16* Out,
    const int32_t* NLivePtr, const int32_t* TreeAnc,
    int64_t stride_x, int64_t stride_c_blk, int64_t stride_c_n, int64_t stride_bt,
    int max_blocks, int N_CACHE, int N_SHIFT, int stream_offset, int64_t stride_s,
    float scale, float eps, float norm_eps,
    int normalize_values,
    bool hcr, bool hca, bool hml, bool pdl, int early_pdl
) {
    #define GO_C48(NQ_V, HCR_V, HCA_V, HML_V) \
        if (pdl) { \
            return launch_c48_one_pdl<N_STACK, NQ_V, HCR_V, HCA_V, HML_V, true>( \
                T, stream, X, CW, Cache, ConvOut, ConvRes, ConvAdd, AuxOut, \
                Pos, SeqIdx, Slot, BT, QStart, Stack, MixLast, Queries, NormW, Out, NLivePtr, TreeAnc, \
                stride_x, stride_c_blk, stride_c_n, stride_bt, max_blocks, N_CACHE, N_SHIFT, stream_offset, stride_s, \
                scale, eps, norm_eps, normalize_values, early_pdl); \
        } else { \
            return launch_c48_one_pdl<N_STACK, NQ_V, HCR_V, HCA_V, HML_V, false>( \
                T, stream, X, CW, Cache, ConvOut, ConvRes, ConvAdd, AuxOut, \
                Pos, SeqIdx, Slot, BT, QStart, Stack, MixLast, Queries, NormW, Out, NLivePtr, TreeAnc, \
                stride_x, stride_c_blk, stride_c_n, stride_bt, max_blocks, N_CACHE, N_SHIFT, stream_offset, stride_s, \
                scale, eps, norm_eps, normalize_values, 0); \
        }
    if (nq == 1) {
        if (!hca && !hcr && !hml) { GO_C48(1, false, false, false); }
        if (!hca && !hcr &&  hml) { GO_C48(1, false, false, true);  }
        if ( hca && !hcr && !hml) { GO_C48(1, false, true,  false); }
        if ( hca &&  hcr && !hml) { GO_C48(1, true,  true,  false); }
    } else if (nq == 3) {
        if (!hca && !hcr && !hml) { GO_C48(3, false, false, false); }
        if (!hca && !hcr &&  hml) { GO_C48(3, false, false, true);  }
        if ( hca && !hcr && !hml) { GO_C48(3, false, true,  false); }
        if ( hca &&  hcr && !hml) { GO_C48(3, true,  true,  false); }
    }
    #undef GO_C48
    return cudaErrorInvalidValue;
}

static int fused_sconv_ar_c48_impl(
    const void* X, const void* CW, void* Cache,
    void* ConvOut, const void* ConvRes, const void* ConvAdd, void* AuxOut,
    const void* Pos, const void* SeqIdx, const void* Slot, const void* BT, const void* QStart,
    const void* Stack, const void* MixLast,
    const void* Queries, const void* NormW, void* Out,
    const void* NLivePtr, const void* TreeSconvAnc,
    int64_t stride_x, int64_t stride_c_blk, int64_t stride_c_n, int64_t stride_bt,
    int max_blocks, int N_CACHE, int stream_offset, int64_t stride_s,
    int nq, int T, int n_stack,
    float scale, float eps, float norm_eps,
    int normalize_values,
    int has_conv_res, int has_conv_add, int has_mix_last,
    int pdl, int early_pdl,
    uint64_t stream_ptr
) {
    if (T <= 0 || T > 64 || (nq != 1 && nq != 3)) return -1;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    const int n_shift = (N_CACHE > 0 && (N_CACHE & (N_CACHE - 1)) == 0) ? __builtin_ctz(N_CACHE) : -1;
    #define CASE_C48(N) case N: return static_cast<int>(dispatch_c48_ns<N>(nq, T, stream, \
        static_cast<const __nv_bfloat16*>(X), static_cast<const __nv_bfloat16*>(CW), static_cast<__nv_bfloat16*>(Cache), \
        static_cast<__nv_bfloat16*>(ConvOut), static_cast<const __nv_bfloat16*>(ConvRes), static_cast<const __nv_bfloat16*>(ConvAdd), static_cast<__nv_bfloat16*>(AuxOut), \
        static_cast<const int64_t*>(Pos), static_cast<const int32_t*>(SeqIdx), static_cast<const int64_t*>(Slot), static_cast<const int32_t*>(BT), static_cast<const int32_t*>(QStart), \
        static_cast<const __nv_bfloat16*>(Stack), static_cast<const __nv_bfloat16*>(MixLast), \
        static_cast<const float*>(Queries), static_cast<const __nv_bfloat16*>(NormW), static_cast<__nv_bfloat16*>(Out), \
        static_cast<const int32_t*>(NLivePtr), static_cast<const int32_t*>(TreeSconvAnc), \
        stride_x, stride_c_blk, stride_c_n, stride_bt, max_blocks, N_CACHE, n_shift, stream_offset, stride_s, \
        scale, eps, norm_eps, normalize_values, has_conv_res != 0, has_conv_add != 0, has_mix_last != 0, pdl != 0, early_pdl))
    switch (n_stack) {
        CASE_C48(1); CASE_C48(2); CASE_C48(3); CASE_C48(4); CASE_C48(5);
        CASE_C48(6); CASE_C48(7); CASE_C48(8); CASE_C48(9); CASE_C48(10);
        default: return -1;
    }
    #undef CASE_C48
}

template <bool HCR, bool HCA, bool PDL>
static inline cudaError_t launch_sconv_v14_one(
    int T, cudaStream_t stream,
    const __nv_bfloat16* X, __nv_bfloat16* Cache, const __nv_bfloat16* CW,
    __nv_bfloat16* Out, __nv_bfloat16* AuxOut, const __nv_bfloat16* Res, const __nv_bfloat16* Add,
    const int64_t* Pos, const int32_t* SeqIdx, const int64_t* Slot, const int32_t* BT, const int32_t* QStart,
    const int32_t* NLivePtr, const int32_t* TreeAnc,
    int64_t stride_x, int64_t stride_c_blk, int64_t stride_c_n, int64_t stride_bt,
    int max_blocks, int N_CACHE, int N_SHIFT, int stream_offset,
    float scale, int pdl_mode
) {
    auto k = sconv_exact_v14_kernel<HCR, HCA, PDL>;
    const dim3 grid(static_cast<unsigned>(T), 1, 1);
    const dim3 block(256, 1, 1);
    if constexpr (!PDL) {
        k<<<grid, block, 0, stream>>>(
            X, Cache, CW, Out, AuxOut, Res, Add,
            Pos, SeqIdx, Slot, BT, QStart, NLivePtr, TreeAnc,
            T, stride_x, stride_c_blk, stride_c_n, stride_bt,
            max_blocks, N_CACHE, N_SHIFT, stream_offset, scale, 0
        );
        return cudaGetLastError();
    } else {
        cudaLaunchAttribute attr[1];
        attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr[0].val.programmaticStreamSerializationAllowed = 1;
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = grid;
        cfg.blockDim = block;
        cfg.dynamicSmemBytes = 0;
        cfg.stream = stream;
        cfg.attrs = attr;
        cfg.numAttrs = 1;
        return cudaLaunchKernelEx(
            &cfg, k,
            X, Cache, CW, Out, AuxOut, Res, Add,
            Pos, SeqIdx, Slot, BT, QStart, NLivePtr, TreeAnc,
            T, stride_x, stride_c_blk, stride_c_n, stride_bt,
            max_blocks, N_CACHE, N_SHIFT, stream_offset, scale, pdl_mode
        );
    }
}

template <int N_STACK, int NQ, bool HA, bool PDL>
static inline cudaError_t launch_ar14_one(
    int T, cudaStream_t stream,
    const __nv_bfloat16* Stack, const __nv_bfloat16* Last, const __nv_bfloat16* Addend,
    const float* Queries, const __nv_bfloat16* NormW, __nv_bfloat16* Out,
    const int32_t* NLivePtr,
    int64_t stride_s, float eps, float norm_eps, int normalize_values, int early_pdl
) {
    constexpr size_t smem_bytes = static_cast<size_t>(N_STACK) * 9216;
    static bool optin_done = false;
    auto k = attn_res_exact_v14_kernel<N_STACK, NQ, HA, PDL>;
    cudaError_t se = ensure_smem_optin(k, smem_bytes, g_ar14_smem_inited, optin_done, stream);
    if (se != cudaSuccess) return se;
    const dim3 grid(static_cast<unsigned>(T), 1, 1);
    const dim3 block(256, 1, 1);
    if constexpr (!PDL) {
        k<<<grid, block, smem_bytes, stream>>>(
            Stack, Last, Addend, Queries, NormW, Out, NLivePtr,
            stride_s, T, eps, norm_eps, normalize_values, 0
        );
        return cudaGetLastError();
    } else {
        cudaLaunchAttribute attr[1];
        attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr[0].val.programmaticStreamSerializationAllowed = 1;
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = grid;
        cfg.blockDim = block;
        cfg.dynamicSmemBytes = smem_bytes;
        cfg.stream = stream;
        cfg.attrs = attr;
        cfg.numAttrs = 1;
        return cudaLaunchKernelEx(
            &cfg, k,
            Stack, Last, Addend, Queries, NormW, Out, NLivePtr,
            stride_s, T, eps, norm_eps, normalize_values, early_pdl
        );
    }
}

template <int N_STACK>
static cudaError_t dispatch_ar14_ns(
    int nq, int T, cudaStream_t stream,
    const __nv_bfloat16* Stack, const __nv_bfloat16* Last, const __nv_bfloat16* Addend,
    const float* Queries, const __nv_bfloat16* NormW, __nv_bfloat16* Out,
    const int32_t* NLivePtr,
    int64_t stride_s, float eps, float norm_eps,
    bool has_addend, int normalize_values, int early_pdl, bool use_pdl
) {
    #define GO_AR14(NQ_V, HA_V) \
        if (use_pdl) { \
            return launch_ar14_one<N_STACK, NQ_V, HA_V, true>( \
                T, stream, Stack, Last, Addend, Queries, NormW, Out, NLivePtr, \
                stride_s, eps, norm_eps, normalize_values, early_pdl); \
        } else { \
            return launch_ar14_one<N_STACK, NQ_V, HA_V, false>( \
                T, stream, Stack, Last, Addend, Queries, NormW, Out, NLivePtr, \
                stride_s, eps, norm_eps, normalize_values, 0); \
        }
    if (nq == 1) {
        if (!has_addend) { GO_AR14(1, false); }
        else             { GO_AR14(1, true);  }
    } else if (nq == 3) {
        if (!has_addend) { GO_AR14(3, false); }
        else             { GO_AR14(3, true);  }
    }
    #undef GO_AR14
    return cudaErrorInvalidValue;
}

template <int N_STACK, int NQ, bool HA, bool PDL>
static inline cudaError_t launch_ar16_blockd_one(
    int T, cudaStream_t stream,
    const __nv_bfloat16* Stack, const __nv_bfloat16* Last, const __nv_bfloat16* Addend,
    const float* Queries, const __nv_bfloat16* NormW, __nv_bfloat16* Out,
    const int32_t* NLivePtr,
    int64_t stride_s, float eps, float norm_eps, int normalize_values, int early_pdl
) {
    constexpr size_t smem_bytes = static_cast<size_t>(N_STACK) * 9216;
    static bool optin_done = false;
    auto k = fused_attnres_rmsnorm_v16_blockd_kernel<N_STACK, NQ, HA, PDL>;
    cudaError_t se = ensure_smem_optin(k, smem_bytes, g_ar16_blockd_smem_inited, optin_done, stream);
    if (se != cudaSuccess) return se;
    const dim3 grid(static_cast<unsigned>(T), 1, 1);
    const dim3 block(128, 1, 1);
    if constexpr (!PDL) {
        k<<<grid, block, smem_bytes, stream>>>(
            Stack, Last, Addend, Queries, NormW, Out, NLivePtr,
            stride_s, T, eps, norm_eps, normalize_values, 0
        );
        return cudaGetLastError();
    } else {
        cudaLaunchAttribute attr[1];
        attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr[0].val.programmaticStreamSerializationAllowed = 1;
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = grid;
        cfg.blockDim = block;
        cfg.dynamicSmemBytes = smem_bytes;
        cfg.stream = stream;
        cfg.attrs = attr;
        cfg.numAttrs = 1;
        return cudaLaunchKernelEx(
            &cfg, k,
            Stack, Last, Addend, Queries, NormW, Out, NLivePtr,
            stride_s, T, eps, norm_eps, normalize_values, early_pdl
        );
    }
}

template <int N_STACK>
static cudaError_t dispatch_ar16_blockd_ns(
    int nq, int T, cudaStream_t stream,
    const __nv_bfloat16* Stack, const __nv_bfloat16* Last, const __nv_bfloat16* Addend,
    const float* Queries, const __nv_bfloat16* NormW, __nv_bfloat16* Out,
    const int32_t* NLivePtr,
    int64_t stride_s, float eps, float norm_eps,
    bool has_addend, int normalize_values, int early_pdl, bool use_pdl
) {
    #define GO_AR16B(NQ_V, HA_V) \
        if (use_pdl) { \
            return launch_ar16_blockd_one<N_STACK, NQ_V, HA_V, true>( \
                T, stream, Stack, Last, Addend, Queries, NormW, Out, NLivePtr, \
                stride_s, eps, norm_eps, normalize_values, early_pdl); \
        } else { \
            return launch_ar16_blockd_one<N_STACK, NQ_V, HA_V, false>( \
                T, stream, Stack, Last, Addend, Queries, NormW, Out, NLivePtr, \
                stride_s, eps, norm_eps, normalize_values, 0); \
        }
    if (nq == 1) {
        if (!has_addend) { GO_AR16B(1, false); }
        else             { GO_AR16B(1, true);  }
    } else if (nq == 3) {
        if (!has_addend) { GO_AR16B(3, false); }
        else             { GO_AR16B(3, true);  }
    }
    #undef GO_AR16B
    return cudaErrorInvalidValue;
}

}  // namespace

// =========================================================================
// EXPORTED C ABI SYMBOLS (17 symbols: 100% superset of v4 + c32 + c48 + v14 + v16!)
// =========================================================================

extern "C" int fused_sconv_ar_v4_version(void)  { return 4; }
extern "C" int fused_sconv_ar_v7_version(void)  { return 7; }
extern "C" int fused_sconv_ar_v14_version(void) { return 14; }
extern "C" int fused_sconv_ar_v16_version(void) { return 16; }

extern "C" int fused_sconv_ar_c32_init_smem(void) {
    if (g_c32_smem_inited) return 0;
    cudaError_t e = cudaSuccess;
    if ((e = init_c32_smem_ns<4>()) != cudaSuccess) return static_cast<int>(e);
    if ((e = init_c32_smem_ns<5>()) != cudaSuccess) return static_cast<int>(e);
    if ((e = init_c32_smem_ns<6>()) != cudaSuccess) return static_cast<int>(e);
    if ((e = init_c32_smem_ns<7>()) != cudaSuccess) return static_cast<int>(e);
    if ((e = init_c32_smem_ns<8>()) != cudaSuccess) return static_cast<int>(e);
    g_c32_smem_inited = true;
    return 0;
}

extern "C" int fused_sconv_ar_c48_init_smem(void) {
    if (g_c48_smem_inited) return 0;
    cudaError_t e = cudaSuccess;
    if ((e = init_c48_smem_ns<5>())  != cudaSuccess) return static_cast<int>(e);
    if ((e = init_c48_smem_ns<6>())  != cudaSuccess) return static_cast<int>(e);
    if ((e = init_c48_smem_ns<7>())  != cudaSuccess) return static_cast<int>(e);
    if ((e = init_c48_smem_ns<8>())  != cudaSuccess) return static_cast<int>(e);
    if ((e = init_c48_smem_ns<9>())  != cudaSuccess) return static_cast<int>(e);
    if ((e = init_c48_smem_ns<10>()) != cudaSuccess) return static_cast<int>(e);
    g_c48_smem_inited = true;
    return 0;
}

extern "C" int fused_sconv_ar_c1_init_smem(void) {
    return fused_sconv_ar_c48_init_smem();
}

extern "C" int attn_res_exact_v14_init_smem(void) {
    if (g_ar14_smem_inited) return 0;
    cudaError_t e = cudaSuccess;
    if ((e = init_ar14_smem_ns<5>())  != cudaSuccess) return static_cast<int>(e);
    if ((e = init_ar14_smem_ns<6>())  != cudaSuccess) return static_cast<int>(e);
    if ((e = init_ar14_smem_ns<7>())  != cudaSuccess) return static_cast<int>(e);
    if ((e = init_ar14_smem_ns<8>())  != cudaSuccess) return static_cast<int>(e);
    if ((e = init_ar14_smem_ns<9>())  != cudaSuccess) return static_cast<int>(e);
    if ((e = init_ar14_smem_ns<10>()) != cudaSuccess) return static_cast<int>(e);
    g_ar14_smem_inited = true;
    return 0;
}

extern "C" int fused_attnres_rmsnorm_v16_init_smem(void) {
    int rc = attn_res_exact_v14_init_smem();
    if (rc != 0) return rc;
    if (g_ar16_blockd_smem_inited) return 0;
    cudaError_t e = cudaSuccess;
    if ((e = init_ar16_blockd_smem_ns<5>())  != cudaSuccess) return static_cast<int>(e);
    if ((e = init_ar16_blockd_smem_ns<6>())  != cudaSuccess) return static_cast<int>(e);
    if ((e = init_ar16_blockd_smem_ns<7>())  != cudaSuccess) return static_cast<int>(e);
    if ((e = init_ar16_blockd_smem_ns<8>())  != cudaSuccess) return static_cast<int>(e);
    if ((e = init_ar16_blockd_smem_ns<9>())  != cudaSuccess) return static_cast<int>(e);
    if ((e = init_ar16_blockd_smem_ns<10>()) != cudaSuccess) return static_cast<int>(e);
    g_ar16_blockd_smem_inited = true;
    return 0;
}

extern "C" int fused_sconv_ar_c32_row_fwd(
    const void* X, const void* CW, void* Cache,
    void* ConvOut, const void* ConvRes, const void* ConvAdd, void* AuxOut,
    const void* Pos, const void* SeqIdx, const void* Slot, const void* BT, const void* QStart,
    const void* Stack, const void* MixLast,
    const void* Queries, const void* NormW, void* Out,
    int64_t stride_x, int64_t stride_c_blk, int64_t stride_c_n, int64_t stride_bt,
    int max_blocks, int N_CACHE, int stream_offset, int64_t stride_s,
    int nq, int T, int n_stack,
    float scale, float eps, float norm_eps,
    int has_conv_res, int has_conv_add, int has_mix_last,
    uint64_t stream_ptr
) {
    if (T <= 0 || T * nq > 256) return -1;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    const int n_shift = (N_CACHE > 0 && (N_CACHE & (N_CACHE - 1)) == 0) ? __builtin_ctz(N_CACHE) : -1;
    #define CASE_R(N) case N: return static_cast<int>(dispatch_c32_ns<N, true>(nq, T, stream, \
        static_cast<const __nv_bfloat16*>(X), static_cast<const __nv_bfloat16*>(CW), static_cast<__nv_bfloat16*>(Cache), \
        static_cast<__nv_bfloat16*>(ConvOut), static_cast<const __nv_bfloat16*>(ConvRes), static_cast<const __nv_bfloat16*>(ConvAdd), static_cast<__nv_bfloat16*>(AuxOut), \
        static_cast<const int64_t*>(Pos), static_cast<const int32_t*>(SeqIdx), static_cast<const int64_t*>(Slot), static_cast<const int32_t*>(BT), static_cast<const int32_t*>(QStart), \
        static_cast<const __nv_bfloat16*>(Stack), static_cast<const __nv_bfloat16*>(MixLast), \
        static_cast<const float*>(Queries), static_cast<const __nv_bfloat16*>(NormW), static_cast<__nv_bfloat16*>(Out), \
        stride_x, stride_c_blk, stride_c_n, stride_bt, max_blocks, N_CACHE, n_shift, stream_offset, stride_s, \
        scale, eps, norm_eps, has_conv_res != 0, has_conv_add != 0, has_mix_last != 0))
    switch (n_stack) {
        CASE_R(1); CASE_R(2); CASE_R(3); CASE_R(4);
        CASE_R(5); CASE_R(6); CASE_R(7); CASE_R(8);
        default: return -1;
    }
    #undef CASE_R
}

extern "C" int fused_sconv_ar_c32_blockd_fwd(
    const void* X, const void* CW, void* Cache,
    void* ConvOut, const void* ConvRes, const void* ConvAdd, void* AuxOut,
    const void* Pos, const void* SeqIdx, const void* Slot, const void* BT, const void* QStart,
    const void* Stack, const void* MixLast,
    const void* Queries, const void* NormW, void* Out,
    int64_t stride_x, int64_t stride_c_blk, int64_t stride_c_n, int64_t stride_bt,
    int max_blocks, int N_CACHE, int stream_offset, int64_t stride_s,
    int nq, int T, int n_stack,
    float scale, float eps, float norm_eps,
    int has_conv_res, int has_conv_add, int has_mix_last,
    uint64_t stream_ptr
) {
    if (T <= 0 || T * nq <= 256) return -1;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    const int n_shift = (N_CACHE > 0 && (N_CACHE & (N_CACHE - 1)) == 0) ? __builtin_ctz(N_CACHE) : -1;
    #define CASE_B(N) case N: return static_cast<int>(dispatch_c32_ns<N, false>(nq, T, stream, \
        static_cast<const __nv_bfloat16*>(X), static_cast<const __nv_bfloat16*>(CW), static_cast<__nv_bfloat16*>(Cache), \
        static_cast<__nv_bfloat16*>(ConvOut), static_cast<const __nv_bfloat16*>(ConvRes), static_cast<const __nv_bfloat16*>(ConvAdd), static_cast<__nv_bfloat16*>(AuxOut), \
        static_cast<const int64_t*>(Pos), static_cast<const int32_t*>(SeqIdx), static_cast<const int64_t*>(Slot), static_cast<const int32_t*>(BT), static_cast<const int32_t*>(QStart), \
        static_cast<const __nv_bfloat16*>(Stack), static_cast<const __nv_bfloat16*>(MixLast), \
        static_cast<const float*>(Queries), static_cast<const __nv_bfloat16*>(NormW), static_cast<__nv_bfloat16*>(Out), \
        stride_x, stride_c_blk, stride_c_n, stride_bt, max_blocks, N_CACHE, n_shift, stream_offset, stride_s, \
        scale, eps, norm_eps, has_conv_res != 0, has_conv_add != 0, has_mix_last != 0))
    switch (n_stack) {
        CASE_B(1); CASE_B(2); CASE_B(3); CASE_B(4);
        CASE_B(5); CASE_B(6); CASE_B(7); CASE_B(8);
        default: return -1;
    }
    #undef CASE_B
}

extern "C" int fused_sconv_ar_c48_fwd(
    const void* X, const void* CW, void* Cache,
    void* ConvOut, const void* ConvRes, const void* ConvAdd, void* AuxOut,
    const void* Pos, const void* SeqIdx, const void* Slot, const void* BT, const void* QStart,
    const void* Stack, const void* MixLast,
    const void* Queries, const void* NormW, void* Out,
    const void* NLivePtr, const void* TreeSconvAnc,
    int64_t stride_x, int64_t stride_c_blk, int64_t stride_c_n, int64_t stride_bt,
    int max_blocks, int N_CACHE, int stream_offset, int64_t stride_s,
    int nq, int T, int n_stack,
    float scale, float eps, float norm_eps,
    int normalize_values,
    int has_conv_res, int has_conv_add, int has_mix_last,
    int pdl, int early_pdl,
    uint64_t stream_ptr
) {
    return fused_sconv_ar_c48_impl(
        X, CW, Cache, ConvOut, ConvRes, ConvAdd, AuxOut,
        Pos, SeqIdx, Slot, BT, QStart, Stack, MixLast,
        Queries, NormW, Out, NLivePtr, TreeSconvAnc,
        stride_x, stride_c_blk, stride_c_n, stride_bt,
        max_blocks, N_CACHE, stream_offset, stride_s,
        nq, T, n_stack, scale, eps, norm_eps,
        normalize_values, has_conv_res, has_conv_add, has_mix_last,
        pdl, early_pdl, stream_ptr
    );
}

extern "C" int fused_sconv_ar_c1_fwd(
    const void* X, const void* CW, void* Cache,
    void* ConvOut, const void* ConvRes, const void* ConvAdd, void* AuxOut,
    const void* Pos, const void* SeqIdx, const void* Slot, const void* BT, const void* QStart,
    const void* Stack, const void* MixLast,
    const void* Queries, const void* NormW, void* Out,
    const void* NLivePtr, const void* TreeSconvAnc,
    int64_t stride_x, int64_t stride_c_blk, int64_t stride_c_n, int64_t stride_bt,
    int max_blocks, int N_CACHE, int stream_offset, int64_t stride_s,
    int nq, int T, int n_stack,
    float scale, float eps, float norm_eps,
    int has_conv_res, int has_conv_add, int has_mix_last,
    int pdl, int early_pdl,
    uint64_t stream_ptr
) {
    return fused_sconv_ar_c48_impl(
        X, CW, Cache, ConvOut, ConvRes, ConvAdd, AuxOut,
        Pos, SeqIdx, Slot, BT, QStart, Stack, MixLast,
        Queries, NormW, Out, NLivePtr, TreeSconvAnc,
        stride_x, stride_c_blk, stride_c_n, stride_bt,
        max_blocks, N_CACHE, stream_offset, stride_s,
        nq, T, n_stack, scale, eps, norm_eps,
        1, has_conv_res, has_conv_add, has_mix_last,
        pdl, early_pdl, stream_ptr
    );
}

extern "C" int sconv_exact_v14_fwd(
    const void* X, void* Cache, const void* CW,
    void* Out, void* AuxOut, const void* Res, const void* Add,
    const void* Pos, const void* SeqIdx, const void* Slot, const void* BT, const void* QStart,
    const void* NLivePtr, const void* TreeAnc,
    int T, int D,
    int64_t stride_x, int64_t stride_c_blk, int64_t stride_c_n, int64_t stride_bt,
    int max_blocks, int N_CACHE, int stream_offset,
    float scale,
    int has_addend, int has_res, int has_aux, int has_nlive, int pdl_mode,
    uint64_t stream_ptr
) {
    if (T <= 0 || T > 64 || D != 4608) return -1;
    if (has_res && !has_addend) return -1;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    const int n_shift = (N_CACHE > 0 && (N_CACHE & (N_CACHE - 1)) == 0) ? __builtin_ctz(N_CACHE) : -1;
    const bool pdl = (pdl_mode > 0);
    __nv_bfloat16* aux_p = has_aux ? static_cast<__nv_bfloat16*>(AuxOut) : nullptr;
    const int32_t* nl_p  = has_nlive ? static_cast<const int32_t*>(NLivePtr) : nullptr;
    #define GO_SC14(HCR_V, HCA_V) \
        if (pdl) { \
            return static_cast<int>(launch_sconv_v14_one<HCR_V, HCA_V, true>( \
                T, stream, static_cast<const __nv_bfloat16*>(X), static_cast<__nv_bfloat16*>(Cache), \
                static_cast<const __nv_bfloat16*>(CW), static_cast<__nv_bfloat16*>(Out), aux_p, \
                static_cast<const __nv_bfloat16*>(Res), static_cast<const __nv_bfloat16*>(Add), \
                static_cast<const int64_t*>(Pos), static_cast<const int32_t*>(SeqIdx), \
                static_cast<const int64_t*>(Slot), static_cast<const int32_t*>(BT), \
                static_cast<const int32_t*>(QStart), nl_p, static_cast<const int32_t*>(TreeAnc), \
                stride_x, stride_c_blk, stride_c_n, stride_bt, max_blocks, N_CACHE, n_shift, stream_offset, scale, pdl_mode)); \
        } else { \
            return static_cast<int>(launch_sconv_v14_one<HCR_V, HCA_V, false>( \
                T, stream, static_cast<const __nv_bfloat16*>(X), static_cast<__nv_bfloat16*>(Cache), \
                static_cast<const __nv_bfloat16*>(CW), static_cast<__nv_bfloat16*>(Out), aux_p, \
                static_cast<const __nv_bfloat16*>(Res), static_cast<const __nv_bfloat16*>(Add), \
                static_cast<const int64_t*>(Pos), static_cast<const int32_t*>(SeqIdx), \
                static_cast<const int64_t*>(Slot), static_cast<const int32_t*>(BT), \
                static_cast<const int32_t*>(QStart), nl_p, static_cast<const int32_t*>(TreeAnc), \
                stride_x, stride_c_blk, stride_c_n, stride_bt, max_blocks, N_CACHE, n_shift, stream_offset, scale, 0)); \
        }
    if (!has_addend)      { GO_SC14(false, false); }
    else if (!has_res)    { GO_SC14(false, true);  }
    else                  { GO_SC14(true,  true);  }
    #undef GO_SC14
}

extern "C" int attn_res_exact_v14_fwd(
    const void* Stack, const void* Last, const void* Addend,
    const void* Queries, const void* NormW, void* Out, const void* NLivePtr,
    int64_t stride_s,
    int nq, int T, int n_stack,
    float eps, float norm_eps,
    int has_addend, int normalize_values, int early_pdl, int has_nlive, int use_pdl,
    uint64_t stream_ptr
) {
    if (T <= 0 || (nq != 1 && nq != 3) || T * nq > 256) return -1;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    const int32_t* nl_p = has_nlive ? static_cast<const int32_t*>(NLivePtr) : nullptr;
    #define CASE_AR14(N) case N: return static_cast<int>(dispatch_ar14_ns<N>(nq, T, stream, \
        static_cast<const __nv_bfloat16*>(Stack), static_cast<const __nv_bfloat16*>(Last), \
        static_cast<const __nv_bfloat16*>(Addend), static_cast<const float*>(Queries), \
        static_cast<const __nv_bfloat16*>(NormW), static_cast<__nv_bfloat16*>(Out), nl_p, \
        stride_s, eps, norm_eps, has_addend != 0, normalize_values, early_pdl, use_pdl != 0))
    switch (n_stack) {
        CASE_AR14(1); CASE_AR14(2); CASE_AR14(3); CASE_AR14(4); CASE_AR14(5);
        CASE_AR14(6); CASE_AR14(7); CASE_AR14(8); CASE_AR14(9); CASE_AR14(10);
        default: return -1;
    }
    #undef CASE_AR14
}

extern "C" int sconv_exact_v16_fwd(
    const void* X, void* Cache, const void* CW,
    void* Out, void* AuxOut, const void* Res, const void* Add,
    const void* Pos, const void* SeqIdx, const void* Slot, const void* BT, const void* QStart,
    const void* NLivePtr, const void* TreeAnc,
    int T, int D,
    int64_t stride_x, int64_t stride_c_blk, int64_t stride_c_n, int64_t stride_bt,
    int max_blocks, int N_CACHE, int stream_offset,
    float scale,
    int has_addend, int has_res, int has_aux, int has_nlive, int pdl_mode,
    uint64_t stream_ptr
) {
    return sconv_exact_v14_fwd(
        X, Cache, CW, Out, AuxOut, Res, Add,
        Pos, SeqIdx, Slot, BT, QStart, NLivePtr, TreeAnc,
        T, D, stride_x, stride_c_blk, stride_c_n, stride_bt,
        max_blocks, N_CACHE, stream_offset, scale,
        has_addend, has_res, has_aux, has_nlive, pdl_mode, stream_ptr
    );
}

extern "C" int fused_attnres_rmsnorm_v16_fwd(
    const void* Stack, const void* Last, const void* Addend,
    const void* Queries, const void* NormW, void* Out, const void* NLivePtr,
    int64_t stride_s,
    int nq, int T, int n_stack,
    float eps, float norm_eps,
    int has_addend, int normalize_values, int early_pdl, int has_nlive, int use_pdl,
    uint64_t stream_ptr
) {
    if (T <= 0 || (nq != 1 && nq != 3)) return -1;
    if (T * nq <= 256) {
        return attn_res_exact_v14_fwd(
            Stack, Last, Addend, Queries, NormW, Out, NLivePtr,
            stride_s, nq, T, n_stack, eps, norm_eps,
            has_addend, normalize_values, early_pdl, has_nlive, use_pdl, stream_ptr
        );
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    const int32_t* nl_p = has_nlive ? static_cast<const int32_t*>(NLivePtr) : nullptr;
    #define CASE_AR16B(N) case N: return static_cast<int>(dispatch_ar16_blockd_ns<N>(nq, T, stream, \
        static_cast<const __nv_bfloat16*>(Stack), static_cast<const __nv_bfloat16*>(Last), \
        static_cast<const __nv_bfloat16*>(Addend), static_cast<const float*>(Queries), \
        static_cast<const __nv_bfloat16*>(NormW), static_cast<__nv_bfloat16*>(Out), nl_p, \
        stride_s, eps, norm_eps, has_addend != 0, normalize_values, early_pdl, use_pdl != 0))
    switch (n_stack) {
        CASE_AR16B(1); CASE_AR16B(2); CASE_AR16B(3); CASE_AR16B(4); CASE_AR16B(5);
        CASE_AR16B(6); CASE_AR16B(7); CASE_AR16B(8); CASE_AR16B(9); CASE_AR16B(10);
        default: return -1;
    }
    #undef CASE_AR16B
}
