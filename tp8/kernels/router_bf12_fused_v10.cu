// SPDX-License-Identifier: Apache-2.0
// SM121a (GB10) Wave-V10 Single-Launch 48-CTA Fused BF12/BF11 Router + Zero-Scatter Quant
//   + 48x Parallel Selection Keys + Lock-Free Last-CTA Exact TopK-8 + Softmax + Placement
//
// Drop-in ABI superset of BOTH `librouter_bf12_c1_v4.so` (C=1..2, T<=16) AND
// `librouter_bf12_c48_v9.so` (C=3..32, T=1..256), PLUS Wave-V10 single-launch entry points:
//   - `router_bf12_c1_fused_v10_fwd`  (T=1..16, 1 launch -> eliminates 48 K2 launches/step!)
//   - `router_bf11_c1_fused_v10_fwd`  (T=1..16, lossless BF11-Flip slabs + 1 launch, opt-in)
//   - `router_bf12_c48_fused_v10_fwd` (T=1..64, 1 launch -> eliminates 48 K2 launches + global Counts/Ids/Ranks!)
//   - `router_bf11_c48_fused_v10_fwd` (T=1..64, lossless BF11-Flip slabs + 1 launch, opt-in)
//   - `router_bf12_v10_version()`     -> returns 10
//
// Key Wave-V10 Architectural Breakthroughs (100% 0-ULP `nd == 0`, 0 B stack / 0 B spills, 0 #206 PDL hazards):
//   1. Single-Launch 48-CTA Fusion (Grid=(48,1,1), Block=(256,1,1), `__launch_bounds__(256, 1)`):
//      - Eliminates 48 separate K2 launches/step across C=1..8 (T=1..64) -> -140..-260 us/step!
//      - Exact 1 CTA/SM across GB10's 48 SMs -> full 255 regs/thr budget -> GUARANTEED 0 B spills!
//      - Static SMEM <= 46.7 KiB (< 48 KiB default) -> ZERO cudaFuncSetAttribute needed (graph-capture safe!).
//   2. 48-CTA Parallel Zero-Scatter Input Quantization (Phase 1A):
//      - Distributes tokens across CTAs (`tok = bid, bid+48`): 1 token/CTA for T<=48, <=2 for T=49..64!
//      - Upgrades C=1..2 from legacy v4 1-CTA serialized `float x[16]/x[32]` + 288x `div_rn` to
//        register-compact 0-stack quant with 1 `div_rn`/token and exact `<bool QUANT_FMA>` toggle
//        (matching BOTH `QUANT_FMA=1` EXP-7C FFMA AND `QUANT_FMA=0` CUDA 13.3 unfused Triton `nd == 0`!).
//      - Runs right after `gdc_wait` while X[tok] is maximally L2-hot from SConv+AttnRes, overlapped with DRAM GEMM.
//   3. 48x Parallel `selection_bits(logit, bias)` Epilogue (Phase 1C):
//      - Each CTA owns 8 experts (`bid*8..bid*8+7`) and computes `selection_bits` in parallel right after
//        exact ascending `pk=0..4` FP32 `add_rn` reduction -> removes 100% of 384*T `expf`/`div_rn` from Last CTA!
//   4. Lock-Free Last-CTA Election (Phase 2) + Pure-ALU 8-Warp Top-8 + 64-Bit SMEM Bitmask Prefix Scan (Phase 3):
//      - ZERO spin-wait, 100% deadlock-free even if L2PF or another stream shares SMs!
//      - Self-resetting `Arrive[0]` atomic counter (`atom.acq_rel.gpu.global.add` == 47).
//      - Last CTA runs pure integer `redux_max_u32`/`redux_min_u32` Top-8 across 8 warps (1 pass for T<=8,
//        2 for T<=16, 4 for T<=32, 8 for T<=64), exact `_log_sigmoid` + `_sum8` Softmax, deterministic
//        64-bit SMEM token bitmask (`sMaskLo[384], sMaskHi[384]`), 6-warp ballot-accelerated prefix scan,
//        and zero-global-round-trip placement (eliminates global Counts/Ids/Ranks/Weights atomics!).
//   5. Optional Lossless `router_bf11` Slab Support (`USE_BF11=true`):
//      - Exact V5 `Slab11` (1408 B/slab, 3-bit `e3` + adaptive `SLAB_META_V5` flip sidecar) cleanly stacked
//        into the same 1-launch fused kernel (-8.4 MB/48L net DRAM, -8.0 MiB VRAM, 0-ULP!).
//
// Build on GB10 (SM121a, STRICT):
//   /usr/local/cuda/bin/nvcc -O3 -std=c++17 -gencode arch=compute_121a,code=sm_121a \
//       --fmad=false --ptxas-options=-v --shared -Xcompiler -fPIC \
//       router_bf12_fused_v10.cu -o librouter_bf12_fused_v10.so

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <math.h>

#ifndef ROUTER_BM64_MIN_CTA
#define ROUTER_BM64_MIN_CTA 2
#endif
#ifndef ROUTER_K2_MIN_CTA
#define ROUTER_K2_MIN_CTA 3
#endif

namespace {

static constexpr int E_FIXED = 384;
static constexpr int K_FIXED = 4608;
static constexpr int TOPK_FIXED = 8;
static constexpr int MAX_T_C1 = 16;
static constexpr int SLAB12_BYTES = 1536;  // 1024 B sm + 512 B e4 (12 x 128 B)
static constexpr int SLAB11_BYTES = 1408;  // 1024 B sm + 384 B e3 (11 x 128 B)
static constexpr float FP32_MAX_VAL = 3.4028234663852886e38f;
static constexpr float INV_6 = 1.0f / 6.0f;
static constexpr float INV_448 = 1.0f / 448.0f;
static constexpr float INV_1536 = 1.0f / 1536.0f;

static constexpr uint32_t BF11_V5_META_BITMASK_BIT = 0x80000000u;
static constexpr uint32_t BF11_V5_META_NF_SHIFT    = 25u;
static constexpr uint32_t BF11_V5_META_NF_MASK     = 0x3Fu;
static constexpr uint32_t BF11_V5_META_OUTLIER_BIT = 0x01000000u;
static constexpr uint32_t BF11_V5_META_OFF_MASK    = 0x00FFFFFFu;

// =========================================================================
// PTX Memory, Atomics, Exact IEEE-754 & Reduction Primitives
// =========================================================================

static __device__ __forceinline__ uint4 ld_cs_u4(const void* p) {
    uint4 v;
    asm volatile("ld.global.cs.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

static __device__ __forceinline__ uint4 ld_ca_u4(const void* p) {
    uint4 v;
    asm volatile("ld.global.ca.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

static __device__ __forceinline__ uint32_t ld_cs_u32(const void* p) {
    uint32_t v;
    asm volatile("ld.global.cs.u32 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
}

static __device__ __forceinline__ uint32_t ld_cs_u16(const void* p) {
    uint16_t v;
    asm volatile("ld.global.cs.u16 %0, [%1];" : "=h"(v) : "l"(p));
    return static_cast<uint32_t>(v);
}

static __device__ __forceinline__ float ld_cg_f32(const float* p) {
    float v;
    asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(v) : "l"(p));
    return v;
}

static __device__ __forceinline__ uint32_t ld_cg_u32(const void* p) {
    uint32_t v;
    asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
}

static __device__ __forceinline__ uint16_t ld_cg_u16(const uint16_t* p) {
    uint16_t v;
    asm volatile("ld.global.cg.u16 %0, [%1];" : "=h"(v) : "l"(p));
    return v;
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

static __device__ __forceinline__ void prefetch_l2(const void* p) {
    asm volatile("prefetch.global.L2 [%0];" :: "l"(p));
}

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
static __device__ __forceinline__ uint32_t redux_max_u32(uint32_t v) {
    uint32_t r; asm("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(r) : "r"(v)); return r;
}
static __device__ __forceinline__ uint32_t redux_min_u32(uint32_t v) {
    uint32_t r; asm("redux.sync.min.u32 %0, %1, 0xffffffff;" : "=r"(r) : "r"(v)); return r;
}

static __device__ __forceinline__ void mma_bf16_r(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)
    );
}

// =========================================================================
// BF12 & BF11-Flip Slab Load, Unpack, & Outlier Patch Helpers (0-Stack!)
// =========================================================================

struct Slab12 { uint4 sm0, sm1, e4; };

template <bool CACHE_W>
static __device__ __forceinline__ Slab12 load_slab12(const uint8_t* sp, int lane) {
    const uint4* p = reinterpret_cast<const uint4*>(sp);
    Slab12 r;
    if constexpr (CACHE_W) {
        r.sm0 = ld_ca_u4(p + lane);
        r.sm1 = ld_ca_u4(p + 32 + lane);
        r.e4  = ld_ca_u4(p + 64 + lane);
    } else {
        r.sm0 = ld_cs_u4(p + lane);
        r.sm1 = ld_cs_u4(p + 32 + lane);
        r.e4  = ld_cs_u4(p + 64 + lane);
    }
    return r;
}

static __device__ __forceinline__ void unpack4_bf12(
    uint32_t sm4, uint32_t e16, uint32_t base_pair,
    uint32_t& b0, uint32_t& b1
) {
    uint32_t d0 = (e16 & 0x000Fu) | ((e16 & 0x00F0u) << 12);
    uint32_t d1 = ((e16 >> 8) & 0x000Fu) | ((e16 & 0xF000u) << 4);
    uint32_t exp0 = base_pair - d0;
    uint32_t exp1 = base_pair - d1;
    uint32_t s0 = __byte_perm(sm4, 0u, 0x4140u);
    uint32_t s1 = __byte_perm(sm4, 0u, 0x4342u);
    b0 = ((s0 & 0x00800080u) << 8) | (exp0 << 7) | (s0 & 0x007F007Fu);
    b1 = ((s1 & 0x00800080u) << 8) | (exp1 << 7) | (s1 & 0x007F007Fu);
}

struct Slab11 {
    uint4 sm0, sm1;
    uint32_t e0, e1, e2;
};

static __device__ __forceinline__ Slab11 load_slab11_cs(const uint8_t* sp, int lane) {
    const uint4* sm_p = reinterpret_cast<const uint4*>(sp);
    const uint32_t* e3_p = reinterpret_cast<const uint32_t*>(sp + 1024);
    Slab11 r;
    r.sm0 = ld_cs_u4(sm_p + lane);
    r.sm1 = ld_cs_u4(sm_p + 32 + lane);
    r.e0  = ld_cs_u32(e3_p + lane);
    r.e1  = ld_cs_u32(e3_p + 32 + lane);
    r.e2  = ld_cs_u32(e3_p + 64 + lane);
    return r;
}

template <int S>
static __device__ __forceinline__ uint32_t slab11_sm_word(const Slab11& r) {
    if constexpr (S == 0) return r.sm0.x;
    else if constexpr (S == 1) return r.sm0.y;
    else if constexpr (S == 2) return r.sm0.z;
    else if constexpr (S == 3) return r.sm0.w;
    else if constexpr (S == 4) return r.sm1.x;
    else if constexpr (S == 5) return r.sm1.y;
    else if constexpr (S == 6) return r.sm1.z;
    else return r.sm1.w;
}

template <int S>
static __device__ __forceinline__ uint32_t slab11_c12_raw(const Slab11& r) {
    if constexpr (S == 0) return r.e0;
    else if constexpr (S == 1) return r.e0 >> 12;
    else if constexpr (S == 2) return __funnelshift_r(r.e0, r.e1, 24);
    else if constexpr (S == 3) return r.e1 >> 4;
    else if constexpr (S == 4) return r.e1 >> 16;
    else if constexpr (S == 5) return __funnelshift_r(r.e1, r.e2, 28);
    else if constexpr (S == 6) return r.e2 >> 8;
    else return r.e2 >> 20;
}

static __device__ __forceinline__ void unpack4_bf11_flip_v5(
    uint32_t sm4, uint32_t c12_raw, uint32_t f4, uint32_t base_sh7,
    uint32_t& b0, uint32_t& b1
) {
    const uint32_t fm0 = ((f4 & 0x3u) * 0x02000400u) & 0x04000400u;
    const uint32_t fm1 = ((f4 & 0xCu) * 0x00800100u) & 0x04000400u;
    const uint32_t d0_sh7 = ((c12_raw << 7) & 0x00000380u)
                          | ((c12_raw << 20) & 0x03800000u)
                          | fm0;
    const uint32_t d1_sh7 = ((c12_raw << 1) & 0x00000380u)
                          | ((c12_raw << 14) & 0x03800000u)
                          | fm1;
    const uint32_t exp0_sh7 = base_sh7 - d0_sh7;
    const uint32_t exp1_sh7 = base_sh7 - d1_sh7;
    const uint32_t s0 = __byte_perm(sm4, 0u, 0x4140u);
    const uint32_t s1 = __byte_perm(sm4, 0u, 0x4342u);
    b0 = exp0_sh7 | ((s0 << 8) & 0x80008000u) | (s0 & 0x007F007Fu);
    b1 = exp1_sh7 | ((s1 << 8) & 0x80008000u) | (s1 & 0x007F007Fu);
}

static __device__ __forceinline__ uint32_t issue_flip_load_bf11(
    uint32_t smeta,
    const uint16_t* __restrict__ SFLIPS,
    const uint32_t* __restrict__ BMASKS,
    int lane
) {
    if (smeta & BF11_V5_META_BITMASK_BIT) {
        const int64_t d_idx = static_cast<int64_t>(smeta & BF11_V5_META_OFF_MASK);
        return ld_cs_u32(BMASKS + (d_idx << 5) + lane);
    }
    const uint32_t nf = (smeta >> BF11_V5_META_NF_SHIFT) & BF11_V5_META_NF_MASK;
    if (nf == 0u) return 0u;
    const int64_t s_off = static_cast<int64_t>(smeta & BF11_V5_META_OFF_MASK);
    return (static_cast<uint32_t>(lane) < nf) ? ld_cs_u16(SFLIPS + s_off + lane) : 0u;
}

static __device__ __forceinline__ uint32_t finalize_flip_mask_bf11(
    uint32_t smeta, uint32_t raw_fl, uint32_t* s_flip_warp, int lane
) {
    if (smeta & BF11_V5_META_BITMASK_BIT) return raw_fl;
    const uint32_t nf = (smeta >> BF11_V5_META_NF_SHIFT) & BF11_V5_META_NF_MASK;
    if (nf == 0u) return 0u;
    if (nf <= 4u) {
        uint32_t mask = 0u;
        const uint32_t u_lane = static_cast<uint32_t>(lane);
        #pragma unroll
        for (uint32_t i = 0u; i < 4u; ++i) {
            const uint32_t rec = __shfl_sync(0xFFFFFFFFu, raw_fl, static_cast<int>(i));
            if (i < nf && (rec >> 5) == u_lane) {
                mask |= (1u << (rec & 31u));
            }
        }
        return mask;
    }
    s_flip_warp[lane] = 0u;
    __syncwarp();
    if (static_cast<uint32_t>(lane) < nf) {
        atomicOr(&s_flip_warp[raw_fl >> 5], 1u << (raw_fl & 31u));
    }
    __syncwarp();
    return s_flip_warp[lane];
}

static __device__ __forceinline__ void patch_outliers(
    uint32_t cur_step, int lane,
    const uint64_t* __restrict__ OUTLIERS,
    int32_t& oc, int32_t oe,
    uint64_t& o_rec, uint32_t& o_next_step,
    uint32_t& b0, uint32_t& b1
) {
    if (__builtin_expect(o_next_step == cur_step, 0)) {
        while (oc < oe && o_next_step == cur_step) {
            const uint32_t lj = static_cast<uint32_t>(o_rec >> 16) & 0x7Fu;
            if ((lj >> 2) == static_cast<uint32_t>(lane)) {
                const uint32_t val = static_cast<uint32_t>(o_rec & 0xFFFFu);
                const uint32_t j = lj & 3u;
                if      (j == 0u) b0 = (b0 & 0xFFFF0000u) | val;
                else if (j == 1u) b0 = (b0 & 0x0000FFFFu) | (val << 16);
                else if (j == 2u) b1 = (b1 & 0xFFFF0000u) | val;
                else              b1 = (b1 & 0x0000FFFFu) | (val << 16);
            }
            ++oc;
            o_rec = (oc < oe) ? __ldg(OUTLIERS + oc) : 0xFFFFFFFFFFFFFFFFULL;
            o_next_step = static_cast<uint32_t>(o_rec >> 32);
        }
    }
}

// =========================================================================
// Exact 0-ULP Register-Compact Input Quantization Helpers (0 B Stack, 0 B Spills!)
// =========================================================================

template <bool QUANT_FMA>
static __device__ __forceinline__ void eval_nvfp4_half4_v10(
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
static __device__ __forceinline__ void quant_nvfp4_block16_compact_v10(
    const uint4 v0, const uint4 v1,
    float bmax,
    float encoder_scale, float decoder_scale, float outer_scale,
    uint64_t& out_packed, uint8_t& out_scale
) {
    float s6_raw = fminf(mul_rn(bmax, mul_rn(encoder_scale, INV_6)), 448.0f);
    float s4_raw = fminf(mul_rn(bmax, mul_rn(encoder_scale, 0.25f)), 448.0f);

    uint16_t s_e4m3x2;
    uint32_t s_f16x2;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %3, %2;\n\t"
        "cvt.rn.f16x2.e4m3x2 %1, %0;"
        : "=&h"(s_e4m3x2), "=r"(s_f16x2)
        : "f"(s6_raw), "f"(s4_raw));

    float scale6 = __half2float(__ushort_as_half(static_cast<uint16_t>(s_f16x2 & 0xFFFFu)));
    float scale4 = __half2float(__ushort_as_half(static_cast<uint16_t>(s_f16x2 >> 16)));
    float inv6 = fminf(div_rn(1.0f, mul_rn(scale6, decoder_scale)), FP32_MAX_VAL);
    float inv4 = fminf(div_rn(1.0f, mul_rn(scale4, decoder_scale)), FP32_MAX_VAL);

    uint32_t pk6_lo, pk4_lo, pk6_hi, pk4_hi;
    float e6_lo, e4_lo, e6_hi, e4_hi;
    eval_nvfp4_half4_v10<QUANT_FMA>(v0, inv6, scale6, inv4, scale4, outer_scale, pk6_lo, pk4_lo, e6_lo, e4_lo);
    eval_nvfp4_half4_v10<QUANT_FMA>(v1, inv6, scale6, inv4, scale4, outer_scale, pk6_hi, pk4_hi, e6_hi, e4_hi);

    const bool choose4 = (add_rn(e4_lo, e4_hi) < add_rn(e6_lo, e6_hi));
    const uint32_t lo_w = choose4 ? pk4_lo : pk6_lo;
    const uint32_t hi_w = choose4 ? pk4_hi : pk6_hi;
    out_packed = static_cast<uint64_t>(lo_w) | (static_cast<uint64_t>(hi_w) << 32);
    out_scale  = choose4 ? static_cast<uint8_t>(s_e4m3x2 >> 8) : static_cast<uint8_t>(s_e4m3x2 & 0xFFu);
}

static __device__ __forceinline__ float block16_amax(uint4 v0, uint4 v1) {
    float bm = 0.0f;
    const uint32_t w[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        float a = bf16_bits_to_f32(static_cast<uint16_t>(w[j] & 0xFFFFu));
        float b = bf16_bits_to_f32(static_cast<uint16_t>(w[j] >> 16));
        bm = fmaxf(bm, fmaxf(fabsf(a), fabsf(b)));
    }
    return bm;
}

static __device__ __forceinline__ void quant_mxfp8_block32_compact(
    const uint4 v[4], uint8_t* dst_vals, uint8_t& out_exp
) {
    float amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t w[4] = {v[i].x, v[i].y, v[i].z, v[i].w};
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float a = bf16_bits_to_f32(static_cast<uint16_t>(w[j] & 0xFFFFu));
            float b = bf16_bits_to_f32(static_cast<uint16_t>(w[j] >> 16));
            amax = fmaxf(amax, fmaxf(fabsf(a), fabsf(b)));
        }
    }

    float raw_scale = mul_rn(amax, INV_448);
    uint32_t rs_u = __float_as_uint(raw_scale);
    uint32_t exp_u;
    if (raw_scale != raw_scale) exp_u = 255u;
    else if (rs_u <= 0x00400000u) exp_u = 0u;
    else {
        exp_u = ((rs_u + 0x007FFFFFu) & 0x7F800000u) >> 23;
        if (exp_u > 254u) exp_u = 254u;
    }

    float inv;
    if (exp_u == 255u) inv = 0.0f;
    else if (exp_u == 0u) inv = __uint_as_float(0x7F000000u);
    else inv = rcp_rn(__uint_as_float(exp_u << 23));

    uint16_t* dst16 = reinterpret_cast<uint16_t*>(dst_vals);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t w[4] = {v[i].x, v[i].y, v[i].z, v[i].w};
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float a = mul_rn(bf16_bits_to_f32(static_cast<uint16_t>(w[j] & 0xFFFFu)), inv);
            float b = mul_rn(bf16_bits_to_f32(static_cast<uint16_t>(w[j] >> 16)), inv);
            if (a == a) { if (a < -448.0f) a = -448.0f; if (a > 448.0f) a = 448.0f; }
            if (b == b) { if (b < -448.0f) b = -448.0f; if (b > 448.0f) b = 448.0f; }
            uint16_t e4;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(e4) : "f"(a), "f"(b));
            dst16[i * 4 + j] = e4;
        }
    }
    out_exp = static_cast<uint8_t>(exp_u);
}

static __device__ __forceinline__ uint32_t selection_bits(float logit, float bias) {
    float sig = div_rn(1.0f, add_rn(1.0f, expf(-logit)));
    float sel = add_rn(sig, bias);
    uint32_t bits = __float_as_uint(sel);
    bits ^= (bits & 0x80000000u) ? 0xFFFFFFFFu : 0x80000000u;
    return bits;
}

// =========================================================================
// WAVE-V10 SINGLE-LAUNCH 48-CTA FUSED KERNEL (T = 1..64, BF12 or BF11-Flip)
//   - Grid  = (48, 1, 1) -> exact 1 CTA/SM on GB10 (48 SMs)!
//   - Block = (256, 1, 1) (8 warps), `__launch_bounds__(256, 1)` -> 255 regs/thr, 0 spills!
//   - BM_CTA in {16, 32, 64}:
//       BM_CTA == 16: C=1..2 (T <= 16), full K=128 slab load (21.25 KiB SMEM)
//       BM_CTA == 32: C=3..4 (T <= 32), K_HALF=64 slab load (22.50 KiB SMEM)
//       BM_CTA == 64: C=5..8 (T <= 64), K_HALF=64 slab load (45.00 KiB SMEM)
//   - C1_MODE == true : emits TileMask[U] (T <= 16 <= tile_sz, 1 tile/expert), omits Starts/Row2Tok
//   - C1_MODE == false: emits Starts[E], StartsCompact[E], multi-tile Tiles/TileCompactStart, Row2Tok
//   - TRIG_W13_MODE == 0: no launch_dependents
//   - TRIG_W13_MODE == 1: late W13 trigger at Last-CTA exit (after Arrive reset)
//   - TRIG_W13_MODE == 2: early W13 trigger at Last-CTA election (safe with W13 pdl_wait==1!)
// =========================================================================
template <
    int BM_CTA,
    bool C1_MODE,
    bool USE_BF11,
    bool NVFP4,
    bool QUANT_FMA,
    bool PDL_WAIT,
    int TRIG_W13_MODE
>
__global__ __launch_bounds__(256, 1)
void router_fused_v10_kernel(
    const __nv_bfloat16* __restrict__ X,
    const uint8_t* __restrict__ SLABS,          // BF12 [48,36,1536] or BF11 [48,36,1408]
    const uint32_t* __restrict__ SLAB_META_V5,  // BF11 [48*36] or nullptr
    const uint16_t* __restrict__ SFLIPS,        // BF11 sparse flips or nullptr
    const uint32_t* __restrict__ BMASKS,        // BF11 bitmasks or nullptr
    const uint64_t* __restrict__ OUTLIERS,
    const int32_t* __restrict__ O_OFF,
    const float* __restrict__ Bias,             // [384]
    const int32_t* __restrict__ ExpertBounds,   // [2]
    const bool* __restrict__ Padding,           // [T] or nullptr
    const int32_t* __restrict__ NLivePtr,       // [1] or nullptr
    float* __restrict__ LogitsWS,               // [T, 384] fp32
    uint32_t* __restrict__ SelBitsWS,           // [T, 384] u32
    uint32_t* __restrict__ Arrive,              // [1] u32 (0-init, self-resetting!)
    int32_t* __restrict__ Starts,               // [384] (C48) or nullptr (C1)
    int32_t* __restrict__ StartsCompact,        // [384] (C48) or nullptr (C1)
    int32_t* __restrict__ Src2Dst,              // [T * 8]
    int32_t* __restrict__ Src2Compact,          // [T * 8]
    float* __restrict__ Gammas,                 // [plan.rows]
    int32_t* __restrict__ Rows,                 // [T * 8]
    int32_t* __restrict__ Tiles,                // [2 * max_ctas + 3]
    int32_t* __restrict__ TileMask,             // [max_ctas] (C1) or nullptr (C48)
    int32_t* __restrict__ TileCompactStart,     // [max_ctas]
    int32_t* __restrict__ Row2Tok,              // [plan.rows] (C48) or nullptr (C1)
    uint32_t* __restrict__ RowAmaxW2,           // [T * 8] or nullptr (skip when V9/V10 qdeq!)
    uint8_t* __restrict__ QValues,              // [T, 2304] NVFP4 or [T, 4608] MXFP8
    uint8_t* __restrict__ QScales,              // [T, 288]  NVFP4 or [T, 144]  MXFP8
    float* __restrict__ QRowScales,             // [plan.rows] or nullptr
    float* __restrict__ RowAmaxIn,              // [T] fp32 (NVFP4)
    const float* __restrict__ WAmax,            // [num_local] or nullptr
    const float* __restrict__ WDenominator,     // [1] or nullptr
    int T,
    float route_scale,
    uint32_t base_exp,
    int num_local,
    int tile_sz,
    int max_ctas
) {
    static_assert(BM_CTA == 16 || BM_CTA == 32 || BM_CTA == 64, "BM_CTA must be 16, 32, or 64");
    constexpr bool FULL_SLAB_X = (BM_CTA == 16);
    constexpr int SX_STRIDE = FULL_SLAB_X ? 136 : 72;
    constexpr int SLAB_BYTES = USE_BF11 ? SLAB11_BYTES : SLAB12_BYTES;

    const int bid  = static_cast<int>(blockIdx.x);  // 0..47
    const int tid  = static_cast<int>(threadIdx.x); // 0..255
    const int warp = tid >> 5;                      // 0..7
    const int lane = tid & 31;                      // 0..31

    struct Phase3Smem {
        uint32_t sMaskLo[E_FIXED];
        uint32_t sMaskHi[E_FIXED];
        int32_t  sStarts[E_FIXED];
        int32_t  sStartsCompact[E_FIXED];
        int32_t  sTokLoc[BM_CTA][TOPK_FIXED];
        float    sTokWt[BM_CTA][TOPK_FIXED];
        uint32_t sSelBuf[8][E_FIXED];
        int32_t  sWarpT[8];
        int32_t  sWarpC[8];
        uint8_t  sIsPad[BM_CTA];
    };

    union alignas(16) SmemV10 {
        struct {
            __nv_bfloat16 sX[5][BM_CTA * SX_STRIDE];
            uint32_t      sFlip[5][USE_BF11 ? 32 : 1];
        } gemm;
        float      sP[5][BM_CTA][9];
        float      sWarpMax[8];
        Phase3Smem p3;
    };
    static_assert(sizeof(SmemV10) <= 49152, "SmemV10 must fit 48 KiB default static SMEM");
    __shared__ SmemV10 smem;
    __shared__ float sBiasLocal[8];
    __shared__ int32_t sIsLastCta;

    // ---- Phase 0 (Pre-gdc_wait): Read-Only Slab Prefetch + Local Bias + Outlier Bounds ----
    const int e_col_base = bid * 8;
    const uint8_t* slab_base = SLABS + static_cast<int64_t>(bid * 36 + warp * 8) * SLAB_BYTES;
    int32_t oc = 0, oe = 0;
    if (warp < 5) {
        const int wuid = bid * 5 + warp;
        oc = __ldg(O_OFF + wuid);
        oe = __ldg(O_OFF + wuid + 1);
        if constexpr (PDL_WAIT) {
            constexpr int PF_LINES = USE_BF11 ? 11 : 12;
            if (lane < PF_LINES) prefetch_l2(slab_base + lane * 128);
        }
    }
    if (tid < 8) {
        sBiasLocal[tid] = __ldg(Bias + e_col_base + tid);
    }

    if constexpr (PDL_WAIT) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }

    int n_live = T;
    if (NLivePtr != nullptr) {
        int nl = *NLivePtr;
        if (nl < 0) nl = 0;
        if (nl < n_live) n_live = nl;
    }
    if (n_live > BM_CTA) n_live = BM_CTA;

    // ---- Phase 1A: 48-CTA Parallel Zero-Scatter Input Quantization (X is maximally L2-hot!) ----
    // For T <= 48: at most 1 token/CTA; for T = 49..64: CTAs 0..15 quantize 2 tokens!
    for (int tok = bid; tok < n_live; tok += 48) {
        const bool pad = (Padding != nullptr) ? Padding[tok] : false;
        if (pad) {
            if constexpr (NVFP4) {
                if (tid == 0) RowAmaxIn[tok] = 0.0f;
            }
            continue;
        }
        if constexpr (NVFP4) {
            // 256 threads cover 288 blocks of 16 BF16: threads 0..31 handle 2 blocks, 32..255 handle 1!
            const __nv_bfloat16* x_row = X + static_cast<int64_t>(tok) * K_FIXED;
            const uint4* bp0 = reinterpret_cast<const uint4*>(x_row + tid * 16);
            const uint4 v0_a = ld_ca_u4(bp0);
            const uint4 v1_a = ld_ca_u4(bp0 + 1);
            const float bm_a = block16_amax(v0_a, v1_a);

            uint4 v0_b = make_uint4(0u, 0u, 0u, 0u);
            uint4 v1_b = make_uint4(0u, 0u, 0u, 0u);
            float bm_b = 0.0f;
            if (tid < 32) {
                const uint4* bp1 = reinterpret_cast<const uint4*>(x_row + (tid + 256) * 16);
                v0_b = ld_ca_u4(bp1);
                v1_b = ld_ca_u4(bp1 + 1);
                bm_b = block16_amax(v0_b, v1_b);
            }

            float wmax = fmaxf(bm_a, bm_b);
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                wmax = fmaxf(wmax, __shfl_xor_sync(0xFFFFFFFFu, wmax, off));
            }
            if (lane == 0) smem.sWarpMax[warp] = wmax;
            __syncthreads();

            float enc_s = 0.0f, dec_s = 0.0f, out_s = 0.0f;
            if (lane == 0) {
                float tok_amax = smem.sWarpMax[0];
                #pragma unroll
                for (int wi = 1; wi < 8; ++wi) tok_amax = fmaxf(tok_amax, smem.sWarpMax[wi]);
                if (tid == 0) RowAmaxIn[tok] = tok_amax;
                enc_s = fminf(div_rn(1536.0f, tok_amax), FP32_MAX_VAL);
                dec_s = div_rn(1.0f, enc_s);
                out_s = mul_rn(tok_amax, INV_1536);
            }
            enc_s = __shfl_sync(0xFFFFFFFFu, enc_s, 0);
            dec_s = __shfl_sync(0xFFFFFFFFu, dec_s, 0);
            out_s = __shfl_sync(0xFFFFFFFFu, out_s, 0);

            uint64_t pk_a;
            uint8_t sc_a;
            quant_nvfp4_block16_compact_v10<QUANT_FMA>(v0_a, v1_a, bm_a, enc_s, dec_s, out_s, pk_a, sc_a);
            reinterpret_cast<uint64_t*>(QValues)[static_cast<int64_t>(tok) * 288 + tid] = pk_a;
            QScales[static_cast<int64_t>(tok) * 288 + tid] = sc_a;

            if (tid < 32) {
                uint64_t pk_b;
                uint8_t sc_b;
                quant_nvfp4_block16_compact_v10<QUANT_FMA>(v0_b, v1_b, bm_b, enc_s, dec_s, out_s, pk_b, sc_b);
                reinterpret_cast<uint64_t*>(QValues)[static_cast<int64_t>(tok) * 288 + (tid + 256)] = pk_b;
                QScales[static_cast<int64_t>(tok) * 288 + (tid + 256)] = sc_b;
            }
            __syncthreads();
        } else {
            if (tid < 144) {
                const uint4* bp = reinterpret_cast<const uint4*>(
                    X + static_cast<int64_t>(tok) * K_FIXED + tid * 32
                );
                uint4 v[4] = {ld_ca_u4(bp), ld_ca_u4(bp + 1), ld_ca_u4(bp + 2), ld_ca_u4(bp + 3)};
                uint8_t sc;
                quant_mxfp8_block32_compact(v, QValues + static_cast<int64_t>(tok) * K_FIXED + tid * 32, sc);
                QScales[static_cast<int64_t>(tok) * 144 + tid] = sc;
            }
            __syncthreads();
        }
    }

    // ---- Phase 1B & 1C: 5-Warp Exact BF12/BF11 GEMM + 48x Parallel `selection_bits` Epilogue ----
    if (n_live > 0) {
        if (warp < 5) {
            const int pk = warp;
            const int g  = lane >> 2;  // 0..7
            const int lt = lane & 3;   // 0..3
            const int n_slabs = (pk < 4) ? 8 : 4;
            const int k_lo = pk * 1024;
            const uint32_t base_pair = base_exp | (base_exp << 16);
            const uint32_t base_sh7  = (base_exp << 7) | (base_exp << 23);

            __nv_bfloat16* sX_warp = smem.gemm.sX[pk];
            const __nv_bfloat16* sX_base = sX_warp + g * SX_STRIDE + 2 * lt;

            const bool has_m8  = (n_live > 8);
            const bool has_m16 = (BM_CTA >= 32) && (n_live > 16);
            const bool has_m24 = (BM_CTA >= 32) && (n_live > 24);
            const bool has_m32 = (BM_CTA >= 64) && (n_live > 32);
            const bool has_m40 = (BM_CTA >= 64) && (n_live > 40);
            const bool has_m48 = (BM_CTA >= 64) && (n_live > 48);
            const bool has_m56 = (BM_CTA >= 64) && (n_live > 56);

            float a0_0 = 0.0f, a0_1 = 0.0f, a0_2 = 0.0f, a0_3 = 0.0f;
            float a1_0 = 0.0f, a1_1 = 0.0f, a1_2 = 0.0f, a1_3 = 0.0f;
            float a2_0 = 0.0f, a2_1 = 0.0f, a2_2 = 0.0f, a2_3 = 0.0f;
            float a3_0 = 0.0f, a3_1 = 0.0f, a3_2 = 0.0f, a3_3 = 0.0f;

            uint64_t o_rec = (oc < oe) ? __ldg(OUTLIERS + oc) : 0xFFFFFFFFFFFFFFFFULL;
            uint32_t o_next_step = static_cast<uint32_t>(o_rec >> 32);

            const int pad_rows = (n_live + 7) & ~7;
            const int total_u4 = FULL_SLAB_X ? (pad_rows * 16) : (pad_rows * 8);

            auto load_x_chunk = [&](int k_base) {
                const uint4 z4 = make_uint4(0u, 0u, 0u, 0u);
                for (int qi = lane; qi < total_u4; qi += 32) {
                    const int lm = FULL_SLAB_X ? (qi >> 4) : (qi >> 3);
                    const int qk = FULL_SLAB_X ? (qi & 15) : (qi & 7);
                    uint4 v = z4;
                    if (lm < n_live) {
                        v = ld_ca_u4(X + static_cast<int64_t>(lm) * K_FIXED + k_base + (qk << 3));
                    }
                    *reinterpret_cast<uint4*>(sX_warp + lm * SX_STRIDE + (qk << 3)) = v;
                }
                __syncwarp();
            };

            auto run_mma = [&](int s_loc, uint32_t b0, uint32_t b1) {
                const int s_off = s_loc * 16;
                {
                    const uint32_t x0 = *reinterpret_cast<const uint32_t*>(sX_base + s_off);
                    const uint32_t x2 = *reinterpret_cast<const uint32_t*>(sX_base + s_off + 8);
                    const uint32_t x1 = has_m8 ? *reinterpret_cast<const uint32_t*>(sX_base + 8 * SX_STRIDE + s_off) : 0u;
                    const uint32_t x3 = has_m8 ? *reinterpret_cast<const uint32_t*>(sX_base + 8 * SX_STRIDE + s_off + 8) : 0u;
                    mma_bf16_r(a0_0, a0_1, a0_2, a0_3, x0, x1, x2, x3, b0, b1);
                }
                if constexpr (BM_CTA >= 32) {
                    if (has_m16) {
                        const uint32_t x0 = *reinterpret_cast<const uint32_t*>(sX_base + 16 * SX_STRIDE + s_off);
                        const uint32_t x2 = *reinterpret_cast<const uint32_t*>(sX_base + 16 * SX_STRIDE + s_off + 8);
                        const uint32_t x1 = has_m24 ? *reinterpret_cast<const uint32_t*>(sX_base + 24 * SX_STRIDE + s_off) : 0u;
                        const uint32_t x3 = has_m24 ? *reinterpret_cast<const uint32_t*>(sX_base + 24 * SX_STRIDE + s_off + 8) : 0u;
                        mma_bf16_r(a1_0, a1_1, a1_2, a1_3, x0, x1, x2, x3, b0, b1);
                    }
                }
                if constexpr (BM_CTA >= 64) {
                    if (has_m32) {
                        const uint32_t x0 = *reinterpret_cast<const uint32_t*>(sX_base + 32 * SX_STRIDE + s_off);
                        const uint32_t x2 = *reinterpret_cast<const uint32_t*>(sX_base + 32 * SX_STRIDE + s_off + 8);
                        const uint32_t x1 = has_m40 ? *reinterpret_cast<const uint32_t*>(sX_base + 40 * SX_STRIDE + s_off) : 0u;
                        const uint32_t x3 = has_m40 ? *reinterpret_cast<const uint32_t*>(sX_base + 40 * SX_STRIDE + s_off + 8) : 0u;
                        mma_bf16_r(a2_0, a2_1, a2_2, a2_3, x0, x1, x2, x3, b0, b1);
                    }
                    if (has_m48) {
                        const uint32_t x0 = *reinterpret_cast<const uint32_t*>(sX_base + 48 * SX_STRIDE + s_off);
                        const uint32_t x2 = *reinterpret_cast<const uint32_t*>(sX_base + 48 * SX_STRIDE + s_off + 8);
                        const uint32_t x1 = has_m56 ? *reinterpret_cast<const uint32_t*>(sX_base + 56 * SX_STRIDE + s_off) : 0u;
                        const uint32_t x3 = has_m56 ? *reinterpret_cast<const uint32_t*>(sX_base + 56 * SX_STRIDE + s_off + 8) : 0u;
                        mma_bf16_r(a3_0, a3_1, a3_2, a3_3, x0, x1, x2, x3, b0, b1);
                    }
                }
            };

            if constexpr (!USE_BF11) {
                Slab12 cur_slab = load_slab12<false>(slab_base, lane);
                for (int sl = 0; sl < n_slabs; ++sl) {
                    const int k_sl = k_lo + sl * 128;
                    load_x_chunk(k_sl);

                    Slab12 nxt_slab;
                    if (sl + 1 < n_slabs) {
                        nxt_slab = load_slab12<false>(slab_base + static_cast<int64_t>(sl + 1) * SLAB12_BYTES, lane);
                    }

                    if constexpr (FULL_SLAB_X) {
                        const uint32_t sm_w[8] = {
                            cur_slab.sm0.x, cur_slab.sm0.y, cur_slab.sm0.z, cur_slab.sm0.w,
                            cur_slab.sm1.x, cur_slab.sm1.y, cur_slab.sm1.z, cur_slab.sm1.w
                        };
                        const uint32_t e_w[8] = {
                            cur_slab.e4.x & 0xFFFFu, cur_slab.e4.x >> 16,
                            cur_slab.e4.y & 0xFFFFu, cur_slab.e4.y >> 16,
                            cur_slab.e4.z & 0xFFFFu, cur_slab.e4.z >> 16,
                            cur_slab.e4.w & 0xFFFFu, cur_slab.e4.w >> 16
                        };
                        #pragma unroll
                        for (int s = 0; s < 8; ++s) {
                            uint32_t b0, b1;
                            unpack4_bf12(sm_w[s], e_w[s], base_pair, b0, b1);
                            patch_outliers(static_cast<uint32_t>((sl << 3) | s), lane,
                                           OUTLIERS, oc, oe, o_rec, o_next_step, b0, b1);
                            run_mma(s, b0, b1);
                        }
                    } else {
                        {
                            const uint32_t sm_h0[4] = {cur_slab.sm0.x, cur_slab.sm0.y, cur_slab.sm0.z, cur_slab.sm0.w};
                            const uint32_t e_h0[4]  = {
                                cur_slab.e4.x & 0xFFFFu, cur_slab.e4.x >> 16,
                                cur_slab.e4.y & 0xFFFFu, cur_slab.e4.y >> 16
                            };
                            #pragma unroll
                            for (int s = 0; s < 4; ++s) {
                                uint32_t b0, b1;
                                unpack4_bf12(sm_h0[s], e_h0[s], base_pair, b0, b1);
                                patch_outliers(static_cast<uint32_t>((sl << 3) | s), lane,
                                               OUTLIERS, oc, oe, o_rec, o_next_step, b0, b1);
                                run_mma(s, b0, b1);
                            }
                        }
                        __syncwarp();
                        load_x_chunk(k_sl + 64);
                        {
                            const uint32_t sm_h1[4] = {cur_slab.sm1.x, cur_slab.sm1.y, cur_slab.sm1.z, cur_slab.sm1.w};
                            const uint32_t e_h1[4]  = {
                                cur_slab.e4.z & 0xFFFFu, cur_slab.e4.z >> 16,
                                cur_slab.e4.w & 0xFFFFu, cur_slab.e4.w >> 16
                            };
                            #pragma unroll
                            for (int s = 0; s < 4; ++s) {
                                uint32_t b0, b1;
                                unpack4_bf12(sm_h1[s], e_h1[s], base_pair, b0, b1);
                                patch_outliers(static_cast<uint32_t>((sl << 3) | (4 + s)), lane,
                                               OUTLIERS, oc, oe, o_rec, o_next_step, b0, b1);
                                run_mma(s, b0, b1);
                            }
                        }
                    }
                    if (sl + 1 < n_slabs) cur_slab = nxt_slab;
                    __syncwarp();
                }
            } else {
                const int64_t meta_base = static_cast<int64_t>(bid * 36 + pk * 8);
                uint32_t cur_meta = __ldg(SLAB_META_V5 + meta_base);
                Slab11 cur_slab = load_slab11_cs(slab_base, lane);
                uint32_t cur_raw_fl = issue_flip_load_bf11(cur_meta, SFLIPS, BMASKS, lane);

                for (int sl = 0; sl < n_slabs; ++sl) {
                    const int k_sl = k_lo + sl * 128;
                    load_x_chunk(k_sl);

                    uint32_t nxt_meta = 0u;
                    Slab11 nxt_slab;
                    uint32_t nxt_raw_fl = 0u;
                    if (sl + 1 < n_slabs) {
                        nxt_meta = __ldg(SLAB_META_V5 + meta_base + sl + 1);
                        nxt_slab = load_slab11_cs(slab_base + static_cast<int64_t>(sl + 1) * SLAB11_BYTES, lane);
                        nxt_raw_fl = issue_flip_load_bf11(nxt_meta, SFLIPS, BMASKS, lane);
                    }

                    const uint32_t flip_mask = finalize_flip_mask_bf11(cur_meta, cur_raw_fl, smem.gemm.sFlip[pk], lane);
                    const bool has_out = (cur_meta & BF11_V5_META_OUTLIER_BIT) != 0u;

                    #define BF11_STEP(S_IDX, S_LOC) do { \
                        uint32_t b0, b1; \
                        unpack4_bf11_flip_v5( \
                            slab11_sm_word<(S_IDX)>(cur_slab), \
                            slab11_c12_raw<(S_IDX)>(cur_slab), \
                            (flip_mask >> ((S_IDX) * 4)) & 0xFu, \
                            base_sh7, b0, b1); \
                        if (__builtin_expect(has_out, 0)) { \
                            patch_outliers(static_cast<uint32_t>((sl << 3) | (S_IDX)), lane, \
                                           OUTLIERS, oc, oe, o_rec, o_next_step, b0, b1); \
                        } \
                        run_mma((S_LOC), b0, b1); \
                    } while (0)

                    if constexpr (FULL_SLAB_X) {
                        BF11_STEP(0, 0); BF11_STEP(1, 1); BF11_STEP(2, 2); BF11_STEP(3, 3);
                        BF11_STEP(4, 4); BF11_STEP(5, 5); BF11_STEP(6, 6); BF11_STEP(7, 7);
                    } else {
                        BF11_STEP(0, 0); BF11_STEP(1, 1); BF11_STEP(2, 2); BF11_STEP(3, 3);
                        __syncwarp();
                        load_x_chunk(k_sl + 64);
                        BF11_STEP(4, 0); BF11_STEP(5, 1); BF11_STEP(6, 2); BF11_STEP(7, 3);
                    }
                    #undef BF11_STEP

                    if (sl + 1 < n_slabs) {
                        cur_meta = nxt_meta;
                        cur_slab = nxt_slab;
                        cur_raw_fl = nxt_raw_fl;
                    }
                    __syncwarp();
                }
            }

            // Wait for all 8 warps before aliasing `smem.gemm` -> `smem.sP`!
            __syncthreads();

            smem.sP[pk][g][2 * lt]     = a0_0;
            smem.sP[pk][g][2 * lt + 1] = a0_1;
            if (has_m8) {
                smem.sP[pk][g + 8][2 * lt]     = a0_2;
                smem.sP[pk][g + 8][2 * lt + 1] = a0_3;
            }
            if constexpr (BM_CTA >= 32) {
                if (has_m16) {
                    smem.sP[pk][g + 16][2 * lt]     = a1_0;
                    smem.sP[pk][g + 16][2 * lt + 1] = a1_1;
                }
                if (has_m24) {
                    smem.sP[pk][g + 24][2 * lt]     = a1_2;
                    smem.sP[pk][g + 24][2 * lt + 1] = a1_3;
                }
            }
            if constexpr (BM_CTA >= 64) {
                if (has_m32) {
                    smem.sP[pk][g + 32][2 * lt]     = a2_0;
                    smem.sP[pk][g + 32][2 * lt + 1] = a2_1;
                }
                if (has_m40) {
                    smem.sP[pk][g + 40][2 * lt]     = a2_2;
                    smem.sP[pk][g + 40][2 * lt + 1] = a2_3;
                }
                if (has_m48) {
                    smem.sP[pk][g + 48][2 * lt]     = a3_0;
                    smem.sP[pk][g + 48][2 * lt + 1] = a3_1;
                }
                if (has_m56) {
                    smem.sP[pk][g + 56][2 * lt]     = a3_2;
                    smem.sP[pk][g + 56][2 * lt + 1] = a3_3;
                }
            }
        } else {
            // Warps 5..7 match the `__syncthreads()` before `smem.sP` write!
            __syncthreads();
        }
        __syncthreads();

        // Phase 1C: All 256 threads reduce exact ascending `pk=0..4` FP32 `add_rn` AND compute
        // `selection_bits(l, bias)` in parallel across ALL 48 SMs!
        const int n_out = n_live * 8;
        for (int oidx = tid; oidx < n_out; oidx += 256) {
            const int lm = oidx >> 3;
            const int e_loc = oidx & 7;
            const float l = add_rn(add_rn(add_rn(add_rn(
                        smem.sP[0][lm][e_loc],
                        smem.sP[1][lm][e_loc]),
                        smem.sP[2][lm][e_loc]),
                        smem.sP[3][lm][e_loc]),
                        smem.sP[4][lm][e_loc]);
            const int64_t g_off = static_cast<int64_t>(lm) * E_FIXED + e_col_base + e_loc;
            LogitsWS[g_off]  = l;
            SelBitsWS[g_off] = selection_bits(l, sBiasLocal[e_loc]);
        }
    }

    // ---- Phase 2: Lock-Free Last-CTA Election (0 Spin-Wait, 100% Deadlock-Free!) ----
    __threadfence();
    __syncthreads();
    if (tid == 0) {
        const uint32_t prev = atom_acq_rel_gpu_add(&Arrive[0], 1u);
        sIsLastCta = (prev == 47u) ? 1 : 0;
    }
    __syncthreads();

    if (!sIsLastCta) {
        if constexpr (TRIG_W13_MODE > 0) {
            if (tid == 0) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
        return;
    }

    // Early W13 trigger (pdl == 3): wake W13 onto the 47 freed SMs NOW while Last CTA runs Phase 3!
    if constexpr (TRIG_W13_MODE == 2) {
        if (tid == 0) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }

    // ---- Phase 3 (Last CTA Only): Pure-ALU 8-Warp Top-8 + Softmax + 64-Bit SMEM Bitmask + Placement ----
    const int lo = __ldg(ExpertBounds);
    const int hi = __ldg(ExpertBounds + 1);

    for (int i = tid; i < E_FIXED; i += 256) {
        smem.p3.sMaskLo[i] = 0u;
        smem.p3.sMaskHi[i] = 0u;
    }
    if (tid < BM_CTA) {
        const bool pad = (tid >= n_live) || ((Padding != nullptr && tid < T) ? Padding[tid] : false);
        smem.p3.sIsPad[tid] = pad ? 1u : 0u;
    }
    __syncthreads();

    // Step 3A: 8 warps run pure-ALU `redux_max_u32`/`redux_min_u32` Top-8 (ZERO `expf`/`div_rn` on 384!)
    for (int tok = warp; tok < T; tok += 8) {
        if (smem.p3.sIsPad[tok]) {
            if (lane < TOPK_FIXED) {
                smem.p3.sTokLoc[tok][lane] = -1;
                smem.p3.sTokWt[tok][lane]  = 0.0f;
            }
            continue;
        }

        const uint32_t* row_sel = SelBitsWS + static_cast<int64_t>(tok) * E_FIXED;
        uint32_t* warp_sel = smem.p3.sSelBuf[warp];
        uint32_t alive_mask = 0xFFFu;
        #pragma unroll
        for (int k = 0; k < 12; ++k) {
            warp_sel[k * 32 + lane] = ld_cg_u32(row_sel + k * 32 + lane);
        }
        __syncwarp();

        int32_t my_id = 0;
        #pragma unroll
        for (int r = 0; r < TOPK_FIXED; ++r) {
            uint32_t best_b = 0u;
            int best_k = -1;
            #pragma unroll
            for (int k = 0; k < 12; ++k) {
                const uint32_t bv = warp_sel[k * 32 + lane];
                if ((alive_mask & (1u << k)) && (best_k < 0 || bv > best_b)) {
                    best_b = bv;
                    best_k = k;
                }
            }
            const uint32_t mb = redux_max_u32(best_k >= 0 ? best_b : 0u);
            const uint32_t cand_id = (best_k >= 0 && best_b == mb)
                ? static_cast<uint32_t>(best_k * 32 + lane) : 0xFFFFFFFFu;
            const uint32_t win_id = redux_min_u32(cand_id);
            if (lane == r) my_id = static_cast<int32_t>(win_id);
            if (static_cast<uint32_t>(lane) == (win_id & 31u)) {
                alive_mask &= ~(1u << (win_id >> 5));
            }
        }

        // Exact `_log_sigmoid` + `_sum8` Softmax on ONLY the 8 winning logits!
        float lp = -INFINITY;
        if (lane < TOPK_FIXED) {
            const float win_logit = ld_cg_f32(LogitsWS + static_cast<int64_t>(tok) * E_FIXED + my_id);
            lp = sub_rn(fminf(win_logit, 0.0f), log1pf(expf(-fabsf(win_logit))));
        }
        const float lp0 = __shfl_sync(0xFFFFFFFFu, lp, 0), lp1 = __shfl_sync(0xFFFFFFFFu, lp, 1);
        const float lp2 = __shfl_sync(0xFFFFFFFFu, lp, 2), lp3 = __shfl_sync(0xFFFFFFFFu, lp, 3);
        const float lp4 = __shfl_sync(0xFFFFFFFFu, lp, 4), lp5 = __shfl_sync(0xFFFFFFFFu, lp, 5);
        const float lp6 = __shfl_sync(0xFFFFFFFFu, lp, 6), lp7 = __shfl_sync(0xFFFFFFFFu, lp, 7);
        const float max_lp = fmaxf(fmaxf(fmaxf(lp0, lp1), fmaxf(lp2, lp3)),
                                   fmaxf(fmaxf(lp4, lp5), fmaxf(lp6, lp7)));

        const float sh = (lane < TOPK_FIXED) ? expf(sub_rn(lp, max_lp)) : 0.0f;
        const float sh0 = __shfl_sync(0xFFFFFFFFu, sh, 0), sh1 = __shfl_sync(0xFFFFFFFFu, sh, 1);
        const float sh2 = __shfl_sync(0xFFFFFFFFu, sh, 2), sh3 = __shfl_sync(0xFFFFFFFFu, sh, 3);
        const float sh4 = __shfl_sync(0xFFFFFFFFu, sh, 4), sh5 = __shfl_sync(0xFFFFFFFFu, sh, 5);
        const float sh6 = __shfl_sync(0xFFFFFFFFu, sh, 6), sh7 = __shfl_sync(0xFFFFFFFFu, sh, 7);

        const float total = add_rn(add_rn(add_rn(sh0, sh2), add_rn(sh1, sh3)),
                                   add_rn(add_rn(sh4, sh6), add_rn(sh5, sh7)));
        const float lse = add_rn(max_lp, logf(total));

        if (lane < TOPK_FIXED) {
            const int32_t my_loc = (my_id >= lo && my_id < hi) ? (my_id - lo) : -1;
            const float my_wt = mul_rn(expf(sub_rn(lp, lse)), route_scale);
            smem.p3.sTokLoc[tok][lane] = my_loc;
            smem.p3.sTokWt[tok][lane]  = my_wt;
            if (my_loc >= 0) {
                if (tok < 32) {
                    atomicOr(&smem.p3.sMaskLo[my_loc], 1u << tok);
                } else {
                    atomicOr(&smem.p3.sMaskHi[my_loc], 1u << (tok - 32));
                }
            }
        }
    }
    __syncthreads();

    // Step 3B: 6-Warp SMEM Prefix Scan over all num_local <= 384 Experts (exact C1 & multi-tile C48!)
    const int tile_shift = (tile_sz == 8) ? 3 : ((tile_sz == 16) ? 4 : ((tile_sz == 32) ? 5 : -1));
    const int tile_mask_i = tile_sz - 1;
    const int e0 = warp * 64 + lane;
    const int e1 = e0 + 32;
    uint32_t m0_lo = 0u, m1_lo = 0u;
    int c0 = 0, c1 = 0, t0 = 0, t1 = 0;
    int exc_t0 = 0, exc_c0 = 0, exc_t1 = 0, exc_c1 = 0;

    if (warp < 6) {
        m0_lo = (e0 < num_local) ? smem.p3.sMaskLo[e0] : 0u;
        m1_lo = (e1 < num_local) ? smem.p3.sMaskLo[e1] : 0u;
        const uint32_t m0_hi = (BM_CTA > 32 && e0 < num_local) ? smem.p3.sMaskHi[e0] : 0u;
        const uint32_t m1_hi = (BM_CTA > 32 && e1 < num_local) ? smem.p3.sMaskHi[e1] : 0u;

        c0 = __popc(m0_lo) + ((BM_CTA > 32) ? __popc(m0_hi) : 0);
        c1 = __popc(m1_lo) + ((BM_CTA > 32) ? __popc(m1_hi) : 0);
        if constexpr (C1_MODE) {
            t0 = (c0 > 0) ? 1 : 0;
            t1 = (c1 > 0) ? 1 : 0;
        } else {
            if (tile_shift >= 0) {
                t0 = (c0 + tile_mask_i) >> tile_shift;
                t1 = (c1 + tile_mask_i) >> tile_shift;
            } else {
                t0 = (c0 + tile_mask_i) / tile_sz;
                t1 = (c1 + tile_mask_i) / tile_sz;
            }
        }

        int it0 = t0, ic0 = c0, it1 = t1, ic1 = c1;
        #pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            int nt0 = __shfl_up_sync(0xFFFFFFFFu, it0, off), nc0 = __shfl_up_sync(0xFFFFFFFFu, ic0, off);
            int nt1 = __shfl_up_sync(0xFFFFFFFFu, it1, off), nc1 = __shfl_up_sync(0xFFFFFFFFu, ic1, off);
            if (lane >= off) { it0 += nt0; ic0 += nc0; it1 += nt1; ic1 += nc1; }
        }
        const int sum0_t = __shfl_sync(0xFFFFFFFFu, it0, 31);
        const int sum0_c = __shfl_sync(0xFFFFFFFFu, ic0, 31);
        if (lane == 31) {
            smem.p3.sWarpT[warp] = sum0_t + it1;
            smem.p3.sWarpC[warp] = sum0_c + ic1;
        }
        exc_t0 = it0 - t0;
        exc_c0 = ic0 - c0;
        exc_t1 = sum0_t + it1 - t1;
        exc_c1 = sum0_c + ic1 - c1;
    }
    __syncthreads();

    if (warp == 0) {
        const int wt = (lane < 6) ? smem.p3.sWarpT[lane] : 0;
        const int wc = (lane < 6) ? smem.p3.sWarpC[lane] : 0;
        int iwt = wt, iwc = wc;
        #pragma unroll
        for (int off = 1; off < 8; off <<= 1) {
            int nt = __shfl_up_sync(0xFFFFFFFFu, iwt, off);
            int nc = __shfl_up_sync(0xFFFFFFFFu, iwc, off);
            if (lane >= off) { iwt += nt; iwc += nc; }
        }
        if (lane < 6) {
            smem.p3.sWarpT[lane] = iwt - wt;
            smem.p3.sWarpC[lane] = iwc - wc;
        }
        if (lane == 5) {
            Tiles[2 * max_ctas]     = iwt;
            Tiles[2 * max_ctas + 1] = iwt * tile_sz;
            Tiles[2 * max_ctas + 2] = iwc;
        }
    }
    __syncthreads();

    if (warp < 6) {
        const int base_t = smem.p3.sWarpT[warp];
        const int base_c = smem.p3.sWarpC[warp];
        auto emit_exp = [&](int e_idx, int c_cnt, int t_cnt, int et, int ec, uint32_t m_lo_val) {
            if (e_idx >= num_local) return;
            const int ts = base_t + et;
            const int cs = base_c + ec;
            const int st = ts * tile_sz;
            smem.p3.sStarts[e_idx]        = st;
            smem.p3.sStartsCompact[e_idx] = cs;
            if constexpr (!C1_MODE) {
                if (Starts != nullptr)        Starts[e_idx] = st;
                if (StartsCompact != nullptr) StartsCompact[e_idx] = cs;
            }
            if constexpr (C1_MODE) {
                if (c_cnt > 0) {
                    Tiles[ts]            = e_idx;
                    Tiles[max_ctas + ts] = st + c_cnt;
                    TileMask[ts]         = static_cast<int32_t>(m_lo_val);
                    TileCompactStart[ts] = cs;
                }
            } else {
                for (int k = 0; k < t_cnt; ++k) {
                    const int ti = ts + k;
                    const int end_r = (k + 1 < t_cnt) ? ((ti + 1) * tile_sz) : (st + c_cnt);
                    Tiles[ti]            = e_idx;
                    Tiles[max_ctas + ti] = end_r;
                    TileCompactStart[ti] = cs + k * tile_sz;
                }
            }
        };
        emit_exp(e0, c0, t0, exc_t0, exc_c0, m0_lo);
        emit_exp(e1, c1, t1, exc_t1, exc_c1, m1_lo);
    }
    __syncthreads();

    // Step 3C: In-SMEM Zero-Global-Round-Trip Placement (256 threads: 1 pass for T<=32, 2 passes for T<=64!)
    const int total_slots = T * TOPK_FIXED;
    float rcp_denom = 0.0f;
    if constexpr (NVFP4) {
        rcp_denom = rcp_rn(mul_rn(1536.0f, __ldg(WDenominator)));
    }
    for (int slot = tid; slot < total_slots; slot += 256) {
        const int tok = slot >> 3;
        const int k   = slot & 7;
        const int32_t loc = smem.p3.sTokLoc[tok][k];
        if (loc >= 0) {
            int32_t rank;
            if (tok < 32) {
                rank = __popc(smem.p3.sMaskLo[loc] & ((1u << tok) - 1u));
            } else {
                rank = __popc(smem.p3.sMaskLo[loc])
                     + __popc(smem.p3.sMaskHi[loc] & ((1u << (tok - 32)) - 1u));
            }
            const int32_t dst  = smem.p3.sStarts[loc] + rank;
            const int32_t cdst = smem.p3.sStartsCompact[loc] + rank;
            Src2Dst[slot]     = dst;
            Src2Compact[slot] = cdst;
            Gammas[dst]       = smem.p3.sTokWt[tok][k];
            Rows[cdst]        = dst;
            if constexpr (!C1_MODE) {
                if (Row2Tok != nullptr) Row2Tok[dst] = tok;
            }
            if constexpr (NVFP4) {
                const float amax_in = ld_cg_f32(RowAmaxIn + tok);
                QRowScales[dst] = mul_rn(mul_rn(amax_in, __ldg(WAmax + loc)), rcp_denom);
            }
            if (RowAmaxW2 != nullptr) {
                RowAmaxW2[cdst] = 0u;
            }
        } else {
            Src2Dst[slot]     = -1;
            Src2Compact[slot] = -1;
        }
    }

    // Flush ALL Last-CTA metadata & placement writes before self-resetting Arrive & exiting!
    __threadfence();
    __syncthreads();
    if (tid == 0) {
        st_rel_gpu_u32(&Arrive[0], 0u);
        if constexpr (TRIG_W13_MODE == 1) {
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
    }
}

// =========================================================================
// Fallback V9 Kernels Preserved 100% Intact (for T = 65..256 & cuBLAS route_only!)
// =========================================================================

template <bool PDL_WAIT, bool TRIG>
__global__ __launch_bounds__(160, ROUTER_BM64_MIN_CTA)
void router_bf12_gemm_c48_v9_bm64_kernel(
    const __nv_bfloat16* __restrict__ X,
    const uint8_t* __restrict__ SLABS,
    const uint64_t* __restrict__ OUTLIERS,
    const int32_t* __restrict__ O_OFF,
    const int32_t* __restrict__ NLivePtr,
    float* __restrict__ Logits,
    int T,
    uint32_t base_exp
) {
    constexpr int BM_CTA = 64;
    constexpr int SX_STRIDE = 72;

    const int e_warp = static_cast<int>(blockIdx.x);
    const int tid = threadIdx.x;
    const int pk = tid >> 5;
    const int lane = tid & 31;
    const int g = lane >> 2;
    const int lt = lane & 3;

    const int n_slabs = (pk < 4) ? 8 : 4;
    const int k_lo = pk * 1024;
    const uint32_t base_pair = base_exp | (base_exp << 16);

    const int wuid = e_warp * 5 + pk;
    int32_t oc = __ldg(O_OFF + wuid);
    const int32_t oe = __ldg(O_OFF + wuid + 1);

    const uint8_t* slab0 = SLABS + static_cast<int64_t>(e_warp * 36 + pk * 8) * SLAB12_BYTES;
    if constexpr (PDL_WAIT) {
        if (lane < 12) prefetch_l2(slab0 + lane * 128);
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }
    if constexpr (TRIG) {
        asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }

    int m_live = T;
    if (NLivePtr != nullptr) {
        int nl = *NLivePtr;
        if (nl < 0) nl = 0;
        if (nl < m_live) m_live = nl;
    }
    if (m_live > BM_CTA) m_live = BM_CTA;
    if (m_live <= 0) return;

    const bool has_m8  = (m_live > 8);
    const bool has_m16 = (m_live > 16);
    const bool has_m24 = (m_live > 24);
    const bool has_m32 = (m_live > 32);
    const bool has_m40 = (m_live > 40);
    const bool has_m48 = (m_live > 48);
    const bool has_m56 = (m_live > 56);

    union alignas(16) SmemUnion64 {
        __nv_bfloat16 sX[5][BM_CTA * SX_STRIDE];
        float         sP[5][BM_CTA][9];
    };
    __shared__ SmemUnion64 smem;
    __nv_bfloat16* sX_warp = smem.sX[pk];

    float a0_0 = 0.0f, a0_1 = 0.0f, a0_2 = 0.0f, a0_3 = 0.0f;
    float a1_0 = 0.0f, a1_1 = 0.0f, a1_2 = 0.0f, a1_3 = 0.0f;
    float a2_0 = 0.0f, a2_1 = 0.0f, a2_2 = 0.0f, a2_3 = 0.0f;
    float a3_0 = 0.0f, a3_1 = 0.0f, a3_2 = 0.0f, a3_3 = 0.0f;

    const __nv_bfloat16* sX_base = sX_warp + g * SX_STRIDE + 2 * lt;
    const int pad_rows = (m_live + 7) & ~7;
    const int total_u4 = pad_rows * 8;

    uint64_t o_rec = (oc < oe) ? __ldg(OUTLIERS + oc) : 0xFFFFFFFFFFFFFFFFULL;
    uint32_t o_next_step = static_cast<uint32_t>(o_rec >> 32);

    Slab12 cur_slab = load_slab12<false>(slab0, lane);

    auto load_x_half = [&](int k_base) {
        const uint4 z4 = make_uint4(0u, 0u, 0u, 0u);
        for (int qi = lane; qi < total_u4; qi += 32) {
            const int lm = qi >> 3;
            const int qk = qi & 7;
            uint4 v = z4;
            if (lm < m_live) {
                v = ld_ca_u4(X + static_cast<int64_t>(lm) * K_FIXED + k_base + (qk << 3));
            }
            *reinterpret_cast<uint4*>(sX_warp + lm * SX_STRIDE + (qk << 3)) = v;
        }
        __syncwarp();
    };

    auto run_mma_step = [&](int s_loc, uint32_t b0, uint32_t b1) {
        const int s_off = s_loc * 16;
        {
            const uint32_t x0 = *reinterpret_cast<const uint32_t*>(sX_base + s_off);
            const uint32_t x2 = *reinterpret_cast<const uint32_t*>(sX_base + s_off + 8);
            const uint32_t x1 = has_m8 ? *reinterpret_cast<const uint32_t*>(sX_base + 8 * SX_STRIDE + s_off) : 0u;
            const uint32_t x3 = has_m8 ? *reinterpret_cast<const uint32_t*>(sX_base + 8 * SX_STRIDE + s_off + 8) : 0u;
            mma_bf16_r(a0_0, a0_1, a0_2, a0_3, x0, x1, x2, x3, b0, b1);
        }
        if (has_m16) {
            const uint32_t x0 = *reinterpret_cast<const uint32_t*>(sX_base + 16 * SX_STRIDE + s_off);
            const uint32_t x2 = *reinterpret_cast<const uint32_t*>(sX_base + 16 * SX_STRIDE + s_off + 8);
            const uint32_t x1 = has_m24 ? *reinterpret_cast<const uint32_t*>(sX_base + 24 * SX_STRIDE + s_off) : 0u;
            const uint32_t x3 = has_m24 ? *reinterpret_cast<const uint32_t*>(sX_base + 24 * SX_STRIDE + s_off + 8) : 0u;
            mma_bf16_r(a1_0, a1_1, a1_2, a1_3, x0, x1, x2, x3, b0, b1);
        }
        if (has_m32) {
            const uint32_t x0 = *reinterpret_cast<const uint32_t*>(sX_base + 32 * SX_STRIDE + s_off);
            const uint32_t x2 = *reinterpret_cast<const uint32_t*>(sX_base + 32 * SX_STRIDE + s_off + 8);
            const uint32_t x1 = has_m40 ? *reinterpret_cast<const uint32_t*>(sX_base + 40 * SX_STRIDE + s_off) : 0u;
            const uint32_t x3 = has_m40 ? *reinterpret_cast<const uint32_t*>(sX_base + 40 * SX_STRIDE + s_off + 8) : 0u;
            mma_bf16_r(a2_0, a2_1, a2_2, a2_3, x0, x1, x2, x3, b0, b1);
        }
        if (has_m48) {
            const uint32_t x0 = *reinterpret_cast<const uint32_t*>(sX_base + 48 * SX_STRIDE + s_off);
            const uint32_t x2 = *reinterpret_cast<const uint32_t*>(sX_base + 48 * SX_STRIDE + s_off + 8);
            const uint32_t x1 = has_m56 ? *reinterpret_cast<const uint32_t*>(sX_base + 56 * SX_STRIDE + s_off) : 0u;
            const uint32_t x3 = has_m56 ? *reinterpret_cast<const uint32_t*>(sX_base + 56 * SX_STRIDE + s_off + 8) : 0u;
            mma_bf16_r(a3_0, a3_1, a3_2, a3_3, x0, x1, x2, x3, b0, b1);
        }
    };

    for (int sl = 0; sl < n_slabs; ++sl) {
        const int k_sl = k_lo + sl * 128;
        load_x_half(k_sl);

        Slab12 nxt_slab;
        if (sl + 1 < n_slabs) {
            nxt_slab = load_slab12<false>(slab0 + static_cast<int64_t>(sl + 1) * SLAB12_BYTES, lane);
        }

        {
            const uint32_t sm_h0[4] = {cur_slab.sm0.x, cur_slab.sm0.y, cur_slab.sm0.z, cur_slab.sm0.w};
            const uint32_t e_h0[4]  = {
                cur_slab.e4.x & 0xFFFFu, cur_slab.e4.x >> 16,
                cur_slab.e4.y & 0xFFFFu, cur_slab.e4.y >> 16
            };
            #pragma unroll
            for (int s_loc = 0; s_loc < 4; ++s_loc) {
                uint32_t b0, b1;
                unpack4_bf12(sm_h0[s_loc], e_h0[s_loc], base_pair, b0, b1);
                patch_outliers(static_cast<uint32_t>((sl << 3) | s_loc), lane,
                               OUTLIERS, oc, oe, o_rec, o_next_step, b0, b1);
                run_mma_step(s_loc, b0, b1);
            }
        }
        __syncwarp();
        load_x_half(k_sl + 64);
        {
            const uint32_t sm_h1[4] = {cur_slab.sm1.x, cur_slab.sm1.y, cur_slab.sm1.z, cur_slab.sm1.w};
            const uint32_t e_h1[4]  = {
                cur_slab.e4.z & 0xFFFFu, cur_slab.e4.z >> 16,
                cur_slab.e4.w & 0xFFFFu, cur_slab.e4.w >> 16
            };
            #pragma unroll
            for (int s_loc = 0; s_loc < 4; ++s_loc) {
                uint32_t b0, b1;
                unpack4_bf12(sm_h1[s_loc], e_h1[s_loc], base_pair, b0, b1);
                patch_outliers(static_cast<uint32_t>((sl << 3) | (4 + s_loc)), lane,
                               OUTLIERS, oc, oe, o_rec, o_next_step, b0, b1);
                run_mma_step(s_loc, b0, b1);
            }
        }
        __syncwarp();
        if (sl + 1 < n_slabs) cur_slab = nxt_slab;
    }

    __syncthreads();
    smem.sP[pk][g][2 * lt]     = a0_0; smem.sP[pk][g][2 * lt + 1]     = a0_1;
    if (has_m8)  { smem.sP[pk][g + 8][2 * lt]  = a0_2; smem.sP[pk][g + 8][2 * lt + 1]  = a0_3; }
    if (has_m16) { smem.sP[pk][g + 16][2 * lt] = a1_0; smem.sP[pk][g + 16][2 * lt + 1] = a1_1; }
    if (has_m24) { smem.sP[pk][g + 24][2 * lt] = a1_2; smem.sP[pk][g + 24][2 * lt + 1] = a1_3; }
    if (has_m32) { smem.sP[pk][g + 32][2 * lt] = a2_0; smem.sP[pk][g + 32][2 * lt + 1] = a2_1; }
    if (has_m40) { smem.sP[pk][g + 40][2 * lt] = a2_2; smem.sP[pk][g + 40][2 * lt + 1] = a2_3; }
    if (has_m48) { smem.sP[pk][g + 48][2 * lt] = a3_0; smem.sP[pk][g + 48][2 * lt + 1] = a3_1; }
    if (has_m56) { smem.sP[pk][g + 56][2 * lt] = a3_2; smem.sP[pk][g + 56][2 * lt + 1] = a3_3; }
    __syncthreads();

    const int n_out = m_live * 8;
    const int e_col_base = e_warp * 8;
    for (int oidx = tid; oidx < n_out; oidx += 160) {
        const int lm = oidx >> 3;
        const int e_loc = oidx & 7;
        float l = add_rn(add_rn(add_rn(add_rn(
                    smem.sP[0][lm][e_loc], smem.sP[1][lm][e_loc]),
                    smem.sP[2][lm][e_loc]), smem.sP[3][lm][e_loc]),
                    smem.sP[4][lm][e_loc]);
        Logits[static_cast<int64_t>(lm) * E_FIXED + e_col_base + e_loc] = l;
    }
}

template <int EW_PER_CTA, bool CACHE_W, bool PDL_WAIT, bool TRIG>
__global__ __launch_bounds__(160, 2)
void router_bf12_gemm_c32_v9_kernel(
    const __nv_bfloat16* __restrict__ X,
    const uint8_t* __restrict__ SLABS,
    const uint64_t* __restrict__ OUTLIERS,
    const int32_t* __restrict__ O_OFF,
    const int32_t* __restrict__ NLivePtr,
    float* __restrict__ Logits,
    int T,
    uint32_t base_exp
) {
    constexpr int BM_CTA = 32;
    constexpr int SX_STRIDE = 72;

    const int e_blk = static_cast<int>(blockIdx.x);
    const int m_blk = static_cast<int>(blockIdx.y);
    const int m_start = m_blk * BM_CTA;
    const int tid = threadIdx.x;
    const int pk = tid >> 5;
    const int lane = tid & 31;
    const int g = lane >> 2;
    const int lt = lane & 3;

    const int n_slabs = (pk < 4) ? 8 : 4;
    const int k_lo = pk * 1024;
    const uint32_t base_pair = base_exp | (base_exp << 16);

    const int ew0 = e_blk * EW_PER_CTA;
    const uint8_t* slab0 = SLABS + static_cast<int64_t>(ew0 * 36 + pk * 8) * SLAB12_BYTES;
    if constexpr (PDL_WAIT) {
        if (lane < 12) prefetch_l2(slab0 + lane * 128);
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }
    if constexpr (TRIG) {
        asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }

    int n_live = T;
    if (NLivePtr != nullptr) {
        int nl = *NLivePtr;
        if (nl < 0) nl = 0;
        if (nl < n_live) n_live = nl;
    }
    int m_live = n_live - m_start;
    if (m_live > BM_CTA) m_live = BM_CTA;
    if (m_live <= 0) return;

    const bool has_m8  = (m_live > 8);
    const bool has_m16 = (m_live > 16);
    const bool has_m24 = (m_live > 24);

    union alignas(16) SmemUnion32 {
        __nv_bfloat16 sX[5][BM_CTA * SX_STRIDE];
        float         sP[5][BM_CTA][9];
    };
    __shared__ SmemUnion32 smem;
    __nv_bfloat16* sX_warp = smem.sX[pk];

    const __nv_bfloat16* sX_base = sX_warp + g * SX_STRIDE + 2 * lt;
    const int pad_rows = (m_live + 7) & ~7;
    const int total_u4 = pad_rows * 8;

    auto load_x_half = [&](int k_base) {
        const uint4 z4 = make_uint4(0u, 0u, 0u, 0u);
        for (int qi = lane; qi < total_u4; qi += 32) {
            const int lm = qi >> 3;
            const int qk = qi & 7;
            uint4 v = z4;
            if (lm < m_live) {
                v = ld_ca_u4(X + static_cast<int64_t>(m_start + lm) * K_FIXED + k_base + (qk << 3));
            }
            *reinterpret_cast<uint4*>(sX_warp + lm * SX_STRIDE + (qk << 3)) = v;
        }
        __syncwarp();
    };

    float acc_lo[EW_PER_CTA][4];
    float acc_hi[EW_PER_CTA][4];
    int32_t o_cur[EW_PER_CTA];
    int32_t o_end_arr[EW_PER_CTA];
    uint64_t o_rec[EW_PER_CTA];
    uint32_t o_next_step[EW_PER_CTA];

    #pragma unroll
    for (int ew = 0; ew < EW_PER_CTA; ++ew) {
        acc_lo[ew][0] = 0.0f; acc_lo[ew][1] = 0.0f; acc_lo[ew][2] = 0.0f; acc_lo[ew][3] = 0.0f;
        acc_hi[ew][0] = 0.0f; acc_hi[ew][1] = 0.0f; acc_hi[ew][2] = 0.0f; acc_hi[ew][3] = 0.0f;
        const int wuid = (ew0 + ew) * 5 + pk;
        const int32_t oc = __ldg(O_OFF + wuid);
        const int32_t oe = __ldg(O_OFF + wuid + 1);
        o_cur[ew] = oc;
        o_end_arr[ew] = oe;
        const uint64_t rec = (oc < oe) ? __ldg(OUTLIERS + oc) : 0xFFFFFFFFFFFFFFFFULL;
        o_rec[ew] = rec;
        o_next_step[ew] = static_cast<uint32_t>(rec >> 32);
    }

    auto run_mma_ew = [&](int ew, int s_loc, uint32_t b0, uint32_t b1) {
        const int s_off = s_loc * 16;
        {
            const uint32_t x0 = *reinterpret_cast<const uint32_t*>(sX_base + s_off);
            const uint32_t x2 = *reinterpret_cast<const uint32_t*>(sX_base + s_off + 8);
            const uint32_t x1 = has_m8 ? *reinterpret_cast<const uint32_t*>(sX_base + 8 * SX_STRIDE + s_off) : 0u;
            const uint32_t x3 = has_m8 ? *reinterpret_cast<const uint32_t*>(sX_base + 8 * SX_STRIDE + s_off + 8) : 0u;
            mma_bf16_r(acc_lo[ew][0], acc_lo[ew][1], acc_lo[ew][2], acc_lo[ew][3], x0, x1, x2, x3, b0, b1);
        }
        if (has_m16) {
            const uint32_t x0 = *reinterpret_cast<const uint32_t*>(sX_base + 16 * SX_STRIDE + s_off);
            const uint32_t x2 = *reinterpret_cast<const uint32_t*>(sX_base + 16 * SX_STRIDE + s_off + 8);
            const uint32_t x1 = has_m24 ? *reinterpret_cast<const uint32_t*>(sX_base + 24 * SX_STRIDE + s_off) : 0u;
            const uint32_t x3 = has_m24 ? *reinterpret_cast<const uint32_t*>(sX_base + 24 * SX_STRIDE + s_off + 8) : 0u;
            mma_bf16_r(acc_hi[ew][0], acc_hi[ew][1], acc_hi[ew][2], acc_hi[ew][3], x0, x1, x2, x3, b0, b1);
        }
    };

    for (int sl = 0; sl < n_slabs; ++sl) {
        const int k_sl = k_lo + sl * 128;
        load_x_half(k_sl);

        Slab12 cur_slab = load_slab12<CACHE_W>(
            SLABS + static_cast<int64_t>(ew0 * 36 + pk * 8 + sl) * SLAB12_BYTES, lane
        );
        #pragma unroll
        for (int ew = 0; ew < EW_PER_CTA; ++ew) {
            Slab12 nxt_slab;
            if (ew + 1 < EW_PER_CTA) {
                nxt_slab = load_slab12<CACHE_W>(
                    SLABS + static_cast<int64_t>((ew0 + ew + 1) * 36 + pk * 8 + sl) * SLAB12_BYTES, lane
                );
            }
            const uint32_t sm_h0[4] = {cur_slab.sm0.x, cur_slab.sm0.y, cur_slab.sm0.z, cur_slab.sm0.w};
            const uint32_t e_h0[4]  = {
                cur_slab.e4.x & 0xFFFFu, cur_slab.e4.x >> 16,
                cur_slab.e4.y & 0xFFFFu, cur_slab.e4.y >> 16
            };
            #pragma unroll
            for (int s_loc = 0; s_loc < 4; ++s_loc) {
                uint32_t b0, b1;
                unpack4_bf12(sm_h0[s_loc], e_h0[s_loc], base_pair, b0, b1);
                patch_outliers(static_cast<uint32_t>((sl << 3) | s_loc), lane,
                               OUTLIERS, o_cur[ew], o_end_arr[ew], o_rec[ew], o_next_step[ew], b0, b1);
                run_mma_ew(ew, s_loc, b0, b1);
            }
            if (ew + 1 < EW_PER_CTA) cur_slab = nxt_slab;
        }
        __syncwarp();
        load_x_half(k_sl + 64);

        cur_slab = load_slab12<CACHE_W>(
            SLABS + static_cast<int64_t>(ew0 * 36 + pk * 8 + sl) * SLAB12_BYTES, lane
        );
        #pragma unroll
        for (int ew = 0; ew < EW_PER_CTA; ++ew) {
            Slab12 nxt_slab;
            if (ew + 1 < EW_PER_CTA) {
                nxt_slab = load_slab12<CACHE_W>(
                    SLABS + static_cast<int64_t>((ew0 + ew + 1) * 36 + pk * 8 + sl) * SLAB12_BYTES, lane
                );
            }
            const uint32_t sm_h1[4] = {cur_slab.sm1.x, cur_slab.sm1.y, cur_slab.sm1.z, cur_slab.sm1.w};
            const uint32_t e_h1[4]  = {
                cur_slab.e4.z & 0xFFFFu, cur_slab.e4.z >> 16,
                cur_slab.e4.w & 0xFFFFu, cur_slab.e4.w >> 16
            };
            #pragma unroll
            for (int s_loc = 0; s_loc < 4; ++s_loc) {
                uint32_t b0, b1;
                unpack4_bf12(sm_h1[s_loc], e_h1[s_loc], base_pair, b0, b1);
                patch_outliers(static_cast<uint32_t>((sl << 3) | (4 + s_loc)), lane,
                               OUTLIERS, o_cur[ew], o_end_arr[ew], o_rec[ew], o_next_step[ew], b0, b1);
                run_mma_ew(ew, s_loc, b0, b1);
            }
            if (ew + 1 < EW_PER_CTA) cur_slab = nxt_slab;
        }
        __syncwarp();
    }

    __syncthreads();
    const int n_out = m_live * 8;
    #pragma unroll
    for (int ew = 0; ew < EW_PER_CTA; ++ew) {
        smem.sP[pk][g][2 * lt]     = acc_lo[ew][0];
        smem.sP[pk][g][2 * lt + 1] = acc_lo[ew][1];
        if (has_m8)  { smem.sP[pk][g + 8][2 * lt]  = acc_lo[ew][2]; smem.sP[pk][g + 8][2 * lt + 1]  = acc_lo[ew][3]; }
        if (has_m16) { smem.sP[pk][g + 16][2 * lt] = acc_hi[ew][0]; smem.sP[pk][g + 16][2 * lt + 1] = acc_hi[ew][1]; }
        if (has_m24) { smem.sP[pk][g + 24][2 * lt] = acc_hi[ew][2]; smem.sP[pk][g + 24][2 * lt + 1] = acc_hi[ew][3]; }
        __syncthreads();

        const int e_col_base = (ew0 + ew) * 8;
        for (int oidx = tid; oidx < n_out; oidx += 160) {
            const int lm = oidx >> 3;
            const int e_loc = oidx & 7;
            float l = add_rn(add_rn(add_rn(add_rn(
                        smem.sP[0][lm][e_loc], smem.sP[1][lm][e_loc]),
                        smem.sP[2][lm][e_loc]), smem.sP[3][lm][e_loc]),
                        smem.sP[4][lm][e_loc]);
            Logits[static_cast<int64_t>(m_start + lm) * E_FIXED + e_col_base + e_loc] = l;
        }
        if (ew + 1 < EW_PER_CTA) __syncthreads();
    }
}

template <bool NVFP4, bool LOGITS_BF16, bool PDL_WAIT, int TRIG_W13_MODE, bool QUANT_FMA>
__global__ __launch_bounds__(384, ROUTER_K2_MIN_CTA)
void c32_quant_topk_place_v9_kernel(
    const __nv_bfloat16* __restrict__ X,
    const void* __restrict__ Logits,
    const float* __restrict__ Bias,
    const int32_t* __restrict__ ExpertBounds,
    const bool* __restrict__ Padding,
    const int32_t* __restrict__ NLivePtr,
    int32_t* __restrict__ Counts,
    uint32_t* __restrict__ Counters,
    int32_t* __restrict__ Ids,
    int32_t* __restrict__ Ranks,
    float* __restrict__ Weights,
    int32_t* __restrict__ Starts,
    int32_t* __restrict__ StartsCompact,
    int32_t* __restrict__ Src2Dst,
    int32_t* __restrict__ Src2Compact,
    float* __restrict__ Gammas,
    int32_t* __restrict__ Rows,
    int32_t* __restrict__ Tiles,
    int32_t* __restrict__ TileCompactStart,
    int32_t* __restrict__ Row2Tok,
    uint32_t* __restrict__ RowAmaxW2,
    uint8_t* __restrict__ QValues,
    uint8_t* __restrict__ QScales,
    float* __restrict__ QRowScales,
    float* __restrict__ RowAmaxIn,
    const float* __restrict__ WAmax,
    const float* __restrict__ WDenominator,
    int T,
    float route_scale,
    int num_local,
    int tile_sz,
    int max_ctas
) {
    const int tok  = static_cast<int>(blockIdx.x);
    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    alignas(16) __shared__ int32_t  sStarts[E_FIXED];
    alignas(16) __shared__ int32_t  sStartsCompact[E_FIXED];
    alignas(16) __shared__ uint32_t sSelBits[E_FIXED];
    alignas(16) __shared__ float    sLogits[E_FIXED];
    __shared__ float    sWarpMax[9];
    __shared__ int32_t  sWarpT[12];
    __shared__ int32_t  sWarpC[12];
    __shared__ int32_t  sIsLastCta;

    int n_live = T;
    if (NLivePtr != nullptr) {
        int nl = *NLivePtr;
        if (nl < 0) nl = 0;
        if (nl < n_live) n_live = nl;
    }
    const bool is_pad = (tok >= n_live) || ((Padding != nullptr) ? Padding[tok] : false);
    const int lo = ExpertBounds[0];
    const int hi = ExpertBounds[1];

    if (!is_pad) {
        if constexpr (NVFP4) {
            uint4 v0 = make_uint4(0u, 0u, 0u, 0u);
            uint4 v1 = make_uint4(0u, 0u, 0u, 0u);
            float blk_amax = 0.0f;
            if (tid < 288) {
                const uint4* bp = reinterpret_cast<const uint4*>(
                    X + static_cast<int64_t>(tok) * K_FIXED + tid * 16
                );
                v0 = ld_ca_u4(bp);
                v1 = ld_ca_u4(bp + 1);
                blk_amax = block16_amax(v0, v1);
                float wmax = blk_amax;
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
                    wmax = fmaxf(wmax, __shfl_xor_sync(0xFFFFFFFFu, wmax, off));
                }
                if (lane == 0) sWarpMax[warp] = wmax;
            }
            __syncthreads();
            if (tid < 288) {
                float enc_s = 0.0f, dec_s = 0.0f, out_s = 0.0f;
                if (lane == 0) {
                    float tok_amax = sWarpMax[0];
                    #pragma unroll
                    for (int wi = 1; wi < 9; ++wi) tok_amax = fmaxf(tok_amax, sWarpMax[wi]);
                    if (tid == 0) RowAmaxIn[tok] = tok_amax;
                    enc_s = fminf(div_rn(1536.0f, tok_amax), FP32_MAX_VAL);
                    dec_s = div_rn(1.0f, enc_s);
                    out_s = mul_rn(tok_amax, INV_1536);
                }
                enc_s = __shfl_sync(0xFFFFFFFFu, enc_s, 0);
                dec_s = __shfl_sync(0xFFFFFFFFu, dec_s, 0);
                out_s = __shfl_sync(0xFFFFFFFFu, out_s, 0);
                uint64_t packed;
                uint8_t sc;
                quant_nvfp4_block16_compact_v10<QUANT_FMA>(v0, v1, blk_amax, enc_s, dec_s, out_s, packed, sc);
                reinterpret_cast<uint64_t*>(QValues)[static_cast<int64_t>(tok) * 288 + tid] = packed;
                QScales[static_cast<int64_t>(tok) * 288 + tid] = sc;
            }
        } else {
            if (tid < 144) {
                const uint4* bp = reinterpret_cast<const uint4*>(
                    X + static_cast<int64_t>(tok) * K_FIXED + tid * 32
                );
                uint4 v[4] = {ld_ca_u4(bp), ld_ca_u4(bp + 1), ld_ca_u4(bp + 2), ld_ca_u4(bp + 3)};
                uint8_t sc;
                quant_mxfp8_block32_compact(v, QValues + static_cast<int64_t>(tok) * K_FIXED + tid * 32, sc);
                QScales[static_cast<int64_t>(tok) * 144 + tid] = sc;
            }
        }
    }

    const float my_bias = (!is_pad) ? __ldg(Bias + tid) : 0.0f;
    if constexpr (PDL_WAIT) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }

    if (!is_pad) {
        const int64_t row_off = static_cast<int64_t>(tok) * E_FIXED;
        float my_logit;
        if constexpr (LOGITS_BF16) {
            my_logit = bf16_bits_to_f32(ld_cg_u16(static_cast<const uint16_t*>(Logits) + row_off + tid));
        } else {
            my_logit = ld_cg_f32(static_cast<const float*>(Logits) + row_off + tid);
        }
        sLogits[tid]  = my_logit;
        sSelBits[tid] = selection_bits(my_logit, my_bias);
    }
    __syncthreads();

    if (warp == 0) {
        if (!is_pad) {
            uint32_t alive_mask = 0xFFFu;
            int32_t my_id = 0;
            #pragma unroll
            for (int r = 0; r < TOPK_FIXED; ++r) {
                uint32_t best_b = 0u;
                int best_k = -1;
                #pragma unroll
                for (int k = 0; k < 12; ++k) {
                    const uint32_t bv = sSelBits[k * 32 + lane];
                    if ((alive_mask & (1u << k)) && (best_k < 0 || bv > best_b)) {
                        best_b = bv;
                        best_k = k;
                    }
                }
                uint32_t mb = redux_max_u32(best_k >= 0 ? best_b : 0u);
                uint32_t cand_id = (best_k >= 0 && best_b == mb)
                    ? static_cast<uint32_t>(best_k * 32 + lane) : 0xFFFFFFFFu;
                uint32_t win_id = redux_min_u32(cand_id);
                if (lane == r) my_id = static_cast<int32_t>(win_id);
                if (static_cast<uint32_t>(lane) == (win_id & 31u)) {
                    alive_mask &= ~(1u << (win_id >> 5));
                }
            }

            float lp = -INFINITY;
            if (lane < TOPK_FIXED) {
                const float win_logit = sLogits[my_id];
                lp = sub_rn(fminf(win_logit, 0.0f), log1pf(expf(-fabsf(win_logit))));
            }
            float lp0 = __shfl_sync(0xFFFFFFFFu, lp, 0), lp1 = __shfl_sync(0xFFFFFFFFu, lp, 1);
            float lp2 = __shfl_sync(0xFFFFFFFFu, lp, 2), lp3 = __shfl_sync(0xFFFFFFFFu, lp, 3);
            float lp4 = __shfl_sync(0xFFFFFFFFu, lp, 4), lp5 = __shfl_sync(0xFFFFFFFFu, lp, 5);
            float lp6 = __shfl_sync(0xFFFFFFFFu, lp, 6), lp7 = __shfl_sync(0xFFFFFFFFu, lp, 7);
            float max_lp = fmaxf(fmaxf(fmaxf(lp0, lp1), fmaxf(lp2, lp3)),
                                 fmaxf(fmaxf(lp4, lp5), fmaxf(lp6, lp7)));
            float sh = (lane < TOPK_FIXED) ? expf(sub_rn(lp, max_lp)) : 0.0f;
            float sh0 = __shfl_sync(0xFFFFFFFFu, sh, 0), sh1 = __shfl_sync(0xFFFFFFFFu, sh, 1);
            float sh2 = __shfl_sync(0xFFFFFFFFu, sh, 2), sh3 = __shfl_sync(0xFFFFFFFFu, sh, 3);
            float sh4 = __shfl_sync(0xFFFFFFFFu, sh, 4), sh5 = __shfl_sync(0xFFFFFFFFu, sh, 5);
            float sh6 = __shfl_sync(0xFFFFFFFFu, sh, 6), sh7 = __shfl_sync(0xFFFFFFFFu, sh, 7);
            float total = add_rn(add_rn(add_rn(sh0, sh2), add_rn(sh1, sh3)),
                                 add_rn(add_rn(sh4, sh6), add_rn(sh5, sh7)));
            float lse = add_rn(max_lp, logf(total));

            if (lane < TOPK_FIXED) {
                const int32_t my_loc = (my_id >= lo && my_id < hi) ? (my_id - lo) : -1;
                const int32_t my_rank = (my_loc >= 0) ? atomicAdd(&Counts[my_loc], 1) : 0;
                const float my_wt = mul_rn(expf(sub_rn(lp, lse)), route_scale);
                const int slot = tok * TOPK_FIXED + lane;
                Ids[slot] = my_loc;
                Ranks[slot] = my_rank;
                Weights[slot] = my_wt;
            }
        } else {
            if (lane < TOPK_FIXED) {
                const int slot = tok * TOPK_FIXED + lane;
                Ids[slot] = -1;
                Ranks[slot] = 0;
                Weights[slot] = 0.0f;
            }
        }
    }

    __threadfence();
    __syncthreads();
    if (tid == 0) {
        uint32_t prev = atom_acq_rel_gpu_add(&Counters[0], 1u);
        sIsLastCta = (prev == static_cast<uint32_t>(T - 1)) ? 1 : 0;
    }
    __syncthreads();

    if (!sIsLastCta) {
        if constexpr (TRIG_W13_MODE > 0) {
            if (tid == 0) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
        return;
    }
    if constexpr (TRIG_W13_MODE == 2) {
        if (tid == 0) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }

    const int tile_shift = (tile_sz == 8) ? 3 : ((tile_sz == 16) ? 4 : ((tile_sz == 32) ? 5 : -1));
    const int tile_mask  = tile_sz - 1;
    const int e = tid;
    int c = (e < num_local) ? static_cast<int>(ld_cg_u32(Counts + e)) : 0;
    if (e < num_local) Counts[e] = 0;

    int t = (tile_shift >= 0) ? ((c + tile_mask) >> tile_shift) : ((c + tile_mask) / tile_sz);
    int exc_t = 0, exc_c = 0;
    const uint32_t nz_mask = __ballot_sync(0xFFFFFFFFu, c > 0);
    if (nz_mask == 0u) {
        if (lane == 31) { sWarpT[warp] = 0; sWarpC[warp] = 0; }
    } else {
        const uint32_t low_mask = (1u << lane) - 1u;
        const uint32_t multi_t = __ballot_sync(0xFFFFFFFFu, t > 1);
        const uint32_t multi_c = __ballot_sync(0xFFFFFFFFu, c > 1);
        if (multi_t == 0u) {
            exc_t = __popc(nz_mask & low_mask);
            if (lane == 31) sWarpT[warp] = __popc(nz_mask);
        } else {
            int it = t;
            #pragma unroll
            for (int off = 1; off < 32; off <<= 1) {
                int nt = __shfl_up_sync(0xFFFFFFFFu, it, off);
                if (lane >= off) it += nt;
            }
            exc_t = it - t;
            if (lane == 31) sWarpT[warp] = it;
        }
        if (multi_c == 0u) {
            exc_c = exc_t;
            if (lane == 31) sWarpC[warp] = __popc(nz_mask);
        } else {
            int ic = c;
            #pragma unroll
            for (int off = 1; off < 32; off <<= 1) {
                int nc = __shfl_up_sync(0xFFFFFFFFu, ic, off);
                if (lane >= off) ic += nc;
            }
            exc_c = ic - c;
            if (lane == 31) sWarpC[warp] = ic;
        }
    }
    __syncthreads();

    if (warp == 0) {
        int wt = (lane < 12) ? sWarpT[lane] : 0;
        int wc = (lane < 12) ? sWarpC[lane] : 0;
        int iwt = wt, iwc = wc;
        #pragma unroll
        for (int off = 1; off < 16; off <<= 1) {
            int nt = __shfl_up_sync(0xFFFFFFFFu, iwt, off);
            int nc = __shfl_up_sync(0xFFFFFFFFu, iwc, off);
            if (lane >= off) { iwt += nt; iwc += nc; }
        }
        if (lane < 12) { sWarpT[lane] = iwt - wt; sWarpC[lane] = iwc - wc; }
        if (lane == 11) {
            Tiles[2 * max_ctas]     = iwt;
            Tiles[2 * max_ctas + 1] = iwt * tile_sz;
            Tiles[2 * max_ctas + 2] = iwc;
        }
    }
    __syncthreads();

    if (e < num_local) {
        const int ts = sWarpT[warp] + exc_t;
        const int cs = sWarpC[warp] + exc_c;
        const int st = ts * tile_sz;
        Starts[e]         = st;
        StartsCompact[e]  = cs;
        sStarts[e]        = st;
        sStartsCompact[e] = cs;
        for (int k = 0; k < t; ++k) {
            const int ti = ts + k;
            const int end_r = (k + 1 < t) ? ((ti + 1) * tile_sz) : (st + c);
            Tiles[ti]            = e;
            Tiles[max_ctas + ti] = end_r;
            TileCompactStart[ti] = cs + k * tile_sz;
        }
    }
    __syncthreads();

    const int total_slots = T * TOPK_FIXED;
    float rcp_denom = 0.0f;
    if constexpr (NVFP4) {
        rcp_denom = rcp_rn(mul_rn(1536.0f, __ldg(WDenominator)));
    }
    for (int slot = tid; slot < total_slots; slot += 384) {
        const int s_tok = slot >> 3;
        const int32_t loc = static_cast<int32_t>(ld_cg_u32(Ids + slot));
        if (loc >= 0) {
            const int32_t rank = static_cast<int32_t>(ld_cg_u32(Ranks + slot));
            const int32_t dst  = sStarts[loc] + rank;
            const int32_t cdst = sStartsCompact[loc] + rank;
            Src2Dst[slot]     = dst;
            Src2Compact[slot] = cdst;
            Gammas[dst]       = ld_cg_f32(Weights + slot);
            Rows[cdst]        = dst;
            Row2Tok[dst]      = s_tok;
            if constexpr (NVFP4) {
                const float amax_in = ld_cg_f32(RowAmaxIn + s_tok);
                QRowScales[dst] = mul_rn(mul_rn(amax_in, __ldg(WAmax + loc)), rcp_denom);
            }
            if (RowAmaxW2 != nullptr) {
                RowAmaxW2[cdst] = 0u;
            }
        } else {
            Src2Dst[slot]     = -1;
            Src2Compact[slot] = -1;
        }
    }

    __threadfence();
    __syncthreads();
    if (tid == 0) {
        st_rel_gpu_u32(&Counters[0], 0u);
        if constexpr (TRIG_W13_MODE == 1) {
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
    }
}

template <typename KernelT, typename... Args>
static inline cudaError_t launch_router_v10(
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

template <int BM_CTA, bool C1_MODE, bool USE_BF11>
static int dispatch_router_fused_v10(
    const void* X, const void* SLABS,
    const void* SLAB_META_V5, const void* SFLIPS, const void* BMASKS,
    const void* OUTLIERS, const void* O_OFF,
    const void* Bias, const void* ExpertBounds,
    const void* Padding, const void* NLivePtr,
    void* LogitsWS, void* SelBitsWS, void* Arrive,
    void* Starts, void* StartsCompact,
    void* Src2Dst, void* Src2Compact, void* Gammas, void* Rows,
    void* Tiles, void* TileMask, void* TileCompactStart, void* Row2Tok,
    void* RowAmaxW2, void* QValues, void* QScales, void* QRowScales, void* RowAmaxIn,
    const void* WAmax, const void* WDenominator,
    int T, float route_scale, uint32_t base_exp,
    int num_local, int tile_sz, int max_ctas,
    int nvfp4, int quant_fma, int pdl, uintptr_t stream_ptr
) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    const bool use_pdl  = (pdl != 0);
    const int trig_mode = (pdl >= 3) ? 2 : ((pdl == 2) ? 1 : 0);
    const bool is_nv    = (nvfp4 != 0);
    const bool use_fma  = (quant_fma != 0);
    const dim3 grid(48, 1, 1);
    const dim3 block(256, 1, 1);
    cudaError_t err = cudaSuccess;

    #define LAUNCH_RF10(NV, QF, PW, TM) err = launch_router_v10( \
        router_fused_v10_kernel<BM_CTA, C1_MODE, USE_BF11, NV, QF, PW, TM>, \
        grid, block, stream, (PW || (TM > 0)), \
        static_cast<const __nv_bfloat16*>(X), static_cast<const uint8_t*>(SLABS), \
        static_cast<const uint32_t*>(SLAB_META_V5), static_cast<const uint16_t*>(SFLIPS), \
        static_cast<const uint32_t*>(BMASKS), static_cast<const uint64_t*>(OUTLIERS), \
        static_cast<const int32_t*>(O_OFF), static_cast<const float*>(Bias), \
        static_cast<const int32_t*>(ExpertBounds), static_cast<const bool*>(Padding), \
        static_cast<const int32_t*>(NLivePtr), static_cast<float*>(LogitsWS), \
        static_cast<uint32_t*>(SelBitsWS), static_cast<uint32_t*>(Arrive), \
        static_cast<int32_t*>(Starts), static_cast<int32_t*>(StartsCompact), \
        static_cast<int32_t*>(Src2Dst), static_cast<int32_t*>(Src2Compact), \
        static_cast<float*>(Gammas), static_cast<int32_t*>(Rows), \
        static_cast<int32_t*>(Tiles), static_cast<int32_t*>(TileMask), \
        static_cast<int32_t*>(TileCompactStart), static_cast<int32_t*>(Row2Tok), \
        static_cast<uint32_t*>(RowAmaxW2), static_cast<uint8_t*>(QValues), \
        static_cast<uint8_t*>(QScales), static_cast<float*>(QRowScales), \
        static_cast<float*>(RowAmaxIn), static_cast<const float*>(WAmax), \
        static_cast<const float*>(WDenominator), \
        T, route_scale, base_exp, num_local, tile_sz, max_ctas)

    #define DISP_PDL(NV, QF) do { \
        if      (use_pdl && trig_mode == 2) { LAUNCH_RF10(NV, QF, true,  2); } \
        else if (use_pdl && trig_mode == 1) { LAUNCH_RF10(NV, QF, true,  1); } \
        else if (use_pdl)                   { LAUNCH_RF10(NV, QF, true,  0); } \
        else                                { LAUNCH_RF10(NV, QF, false, 0); } \
    } while (0)

    if (is_nv) {
        if (use_fma) { DISP_PDL(true, true);  }
        else         { DISP_PDL(true, false); }
    } else {
        DISP_PDL(false, false);
    }
    #undef DISP_PDL
    #undef LAUNCH_RF10
    return static_cast<int>(err);
}

}  // namespace

// =========================================================================
// Extern "C" Entry Points (Wave-V10 Single-Launch + 100% Drop-In V9/V4 ABI!)
// =========================================================================

extern "C" int router_bf12_v10_version() {
    return 10;
}

extern "C" int router_bf12_c32_version() {
    return 10;
}

extern "C" int router_bf12_c48_min_cta() {
    return ROUTER_BM64_MIN_CTA * 10 + ROUTER_K2_MIN_CTA;
}

// ---- NEW Wave-V10 Single-Launch C=1..2 (T <= 16) BF12 Entry Point (35 args) ----
extern "C" int router_bf12_c1_fused_v10_fwd(
    const void* X,
    const void* SLABS,
    const void* OUTLIERS,
    const void* O_OFF,
    const void* Bias,
    const void* ExpertBounds,
    const void* Padding,
    const void* NLivePtr,
    void* LogitsWS,
    void* SelBitsWS,
    void* Arrive,
    void* Src2Dst,
    void* Src2Compact,
    void* Gammas,
    void* Rows,
    void* Tiles,
    void* TileMask,
    void* TileCompactStart,
    void* RowAmaxW2,
    void* QValues,
    void* QScales,
    void* QRowScales,
    void* RowAmaxIn,
    const void* WAmax,
    const void* WDenominator,
    int T,
    float route_scale,
    uint32_t base_exp,
    int num_local,
    int tile_sz,
    int max_ctas,
    int nvfp4,
    int quant_fma,
    int pdl,
    uintptr_t stream_ptr
) {
    if (T <= 0 || T > MAX_T_C1 || num_local <= 0 || num_local > E_FIXED || tile_sz < T || tile_sz > MAX_T_C1) {
        return -1;
    }
    return dispatch_router_fused_v10<16, true, false>(
        X, SLABS, nullptr, nullptr, nullptr, OUTLIERS, O_OFF,
        Bias, ExpertBounds, Padding, NLivePtr,
        LogitsWS, SelBitsWS, Arrive, nullptr, nullptr,
        Src2Dst, Src2Compact, Gammas, Rows,
        Tiles, TileMask, TileCompactStart, nullptr,
        RowAmaxW2, QValues, QScales, QRowScales, RowAmaxIn, WAmax, WDenominator,
        T, route_scale, base_exp, num_local, tile_sz, max_ctas,
        nvfp4, quant_fma, pdl, stream_ptr
    );
}

// ---- NEW Wave-V10 Single-Launch C=1..2 (T <= 16) BF11-Flip Entry Point (38 args) ----
extern "C" int router_bf11_c1_fused_v10_fwd(
    const void* X,
    const void* SLABS,
    const void* SLAB_META_V5,
    const void* SFLIPS,
    const void* BMASKS,
    const void* OUTLIERS,
    const void* O_OFF,
    const void* Bias,
    const void* ExpertBounds,
    const void* Padding,
    const void* NLivePtr,
    void* LogitsWS,
    void* SelBitsWS,
    void* Arrive,
    void* Src2Dst,
    void* Src2Compact,
    void* Gammas,
    void* Rows,
    void* Tiles,
    void* TileMask,
    void* TileCompactStart,
    void* RowAmaxW2,
    void* QValues,
    void* QScales,
    void* QRowScales,
    void* RowAmaxIn,
    const void* WAmax,
    const void* WDenominator,
    int T,
    float route_scale,
    uint32_t base_exp,
    int num_local,
    int tile_sz,
    int max_ctas,
    int nvfp4,
    int quant_fma,
    int pdl,
    uintptr_t stream_ptr
) {
    if (T <= 0 || T > MAX_T_C1 || num_local <= 0 || num_local > E_FIXED || tile_sz < T || tile_sz > MAX_T_C1) {
        return -1;
    }
    return dispatch_router_fused_v10<16, true, true>(
        X, SLABS, SLAB_META_V5, SFLIPS, BMASKS, OUTLIERS, O_OFF,
        Bias, ExpertBounds, Padding, NLivePtr,
        LogitsWS, SelBitsWS, Arrive, nullptr, nullptr,
        Src2Dst, Src2Compact, Gammas, Rows,
        Tiles, TileMask, TileCompactStart, nullptr,
        RowAmaxW2, QValues, QScales, QRowScales, RowAmaxIn, WAmax, WDenominator,
        T, route_scale, base_exp, num_local, tile_sz, max_ctas,
        nvfp4, quant_fma, pdl, stream_ptr
    );
}

// ---- NEW Wave-V10 Single-Launch C=1..8 (T = 1..64) BF12 Entry Point (37 args) ----
extern "C" int router_bf12_c48_fused_v10_fwd(
    const void* X,
    const void* SLABS,
    const void* OUTLIERS,
    const void* O_OFF,
    const void* Bias,
    const void* ExpertBounds,
    const void* Padding,
    const void* NLivePtr,
    void* LogitsWS,
    void* SelBitsWS,
    void* Counters,
    void* Starts,
    void* StartsCompact,
    void* Src2Dst,
    void* Src2Compact,
    void* Gammas,
    void* Rows,
    void* Tiles,
    void* TileCompactStart,
    void* Row2Tok,
    void* RowAmaxW2,
    void* QValues,
    void* QScales,
    void* QRowScales,
    void* RowAmaxIn,
    const void* WAmax,
    const void* WDenominator,
    int T,
    float route_scale,
    uint32_t base_exp,
    int num_local,
    int tile_sz,
    int max_ctas,
    int nvfp4,
    int quant_fma,
    int pdl,
    uintptr_t stream_ptr
) {
    if (T <= 0 || T > 64 || num_local <= 0 || num_local > E_FIXED || tile_sz <= 0 || max_ctas <= 0) {
        return -1;
    }
    if (T <= 16) {
        return dispatch_router_fused_v10<16, false, false>(
            X, SLABS, nullptr, nullptr, nullptr, OUTLIERS, O_OFF,
            Bias, ExpertBounds, Padding, NLivePtr,
            LogitsWS, SelBitsWS, Counters, Starts, StartsCompact,
            Src2Dst, Src2Compact, Gammas, Rows,
            Tiles, nullptr, TileCompactStart, Row2Tok,
            RowAmaxW2, QValues, QScales, QRowScales, RowAmaxIn, WAmax, WDenominator,
            T, route_scale, base_exp, num_local, tile_sz, max_ctas,
            nvfp4, quant_fma, pdl, stream_ptr
        );
    } else if (T <= 32) {
        return dispatch_router_fused_v10<32, false, false>(
            X, SLABS, nullptr, nullptr, nullptr, OUTLIERS, O_OFF,
            Bias, ExpertBounds, Padding, NLivePtr,
            LogitsWS, SelBitsWS, Counters, Starts, StartsCompact,
            Src2Dst, Src2Compact, Gammas, Rows,
            Tiles, nullptr, TileCompactStart, Row2Tok,
            RowAmaxW2, QValues, QScales, QRowScales, RowAmaxIn, WAmax, WDenominator,
            T, route_scale, base_exp, num_local, tile_sz, max_ctas,
            nvfp4, quant_fma, pdl, stream_ptr
        );
    } else {
        return dispatch_router_fused_v10<64, false, false>(
            X, SLABS, nullptr, nullptr, nullptr, OUTLIERS, O_OFF,
            Bias, ExpertBounds, Padding, NLivePtr,
            LogitsWS, SelBitsWS, Counters, Starts, StartsCompact,
            Src2Dst, Src2Compact, Gammas, Rows,
            Tiles, nullptr, TileCompactStart, Row2Tok,
            RowAmaxW2, QValues, QScales, QRowScales, RowAmaxIn, WAmax, WDenominator,
            T, route_scale, base_exp, num_local, tile_sz, max_ctas,
            nvfp4, quant_fma, pdl, stream_ptr
        );
    }
}

// ---- NEW Wave-V10 Single-Launch C=1..8 (T = 1..64) BF11-Flip Entry Point (40 args) ----
extern "C" int router_bf11_c48_fused_v10_fwd(
    const void* X,
    const void* SLABS,
    const void* SLAB_META_V5,
    const void* SFLIPS,
    const void* BMASKS,
    const void* OUTLIERS,
    const void* O_OFF,
    const void* Bias,
    const void* ExpertBounds,
    const void* Padding,
    const void* NLivePtr,
    void* LogitsWS,
    void* SelBitsWS,
    void* Counters,
    void* Starts,
    void* StartsCompact,
    void* Src2Dst,
    void* Src2Compact,
    void* Gammas,
    void* Rows,
    void* Tiles,
    void* TileCompactStart,
    void* Row2Tok,
    void* RowAmaxW2,
    void* QValues,
    void* QScales,
    void* QRowScales,
    void* RowAmaxIn,
    const void* WAmax,
    const void* WDenominator,
    int T,
    float route_scale,
    uint32_t base_exp,
    int num_local,
    int tile_sz,
    int max_ctas,
    int nvfp4,
    int quant_fma,
    int pdl,
    uintptr_t stream_ptr
) {
    if (T <= 0 || T > 64 || num_local <= 0 || num_local > E_FIXED || tile_sz <= 0 || max_ctas <= 0) {
        return -1;
    }
    if (T <= 16) {
        return dispatch_router_fused_v10<16, false, true>(
            X, SLABS, SLAB_META_V5, SFLIPS, BMASKS, OUTLIERS, O_OFF,
            Bias, ExpertBounds, Padding, NLivePtr,
            LogitsWS, SelBitsWS, Counters, Starts, StartsCompact,
            Src2Dst, Src2Compact, Gammas, Rows,
            Tiles, nullptr, TileCompactStart, Row2Tok,
            RowAmaxW2, QValues, QScales, QRowScales, RowAmaxIn, WAmax, WDenominator,
            T, route_scale, base_exp, num_local, tile_sz, max_ctas,
            nvfp4, quant_fma, pdl, stream_ptr
        );
    } else if (T <= 32) {
        return dispatch_router_fused_v10<32, false, true>(
            X, SLABS, SLAB_META_V5, SFLIPS, BMASKS, OUTLIERS, O_OFF,
            Bias, ExpertBounds, Padding, NLivePtr,
            LogitsWS, SelBitsWS, Counters, Starts, StartsCompact,
            Src2Dst, Src2Compact, Gammas, Rows,
            Tiles, nullptr, TileCompactStart, Row2Tok,
            RowAmaxW2, QValues, QScales, QRowScales, RowAmaxIn, WAmax, WDenominator,
            T, route_scale, base_exp, num_local, tile_sz, max_ctas,
            nvfp4, quant_fma, pdl, stream_ptr
        );
    } else {
        return dispatch_router_fused_v10<64, false, true>(
            X, SLABS, SLAB_META_V5, SFLIPS, BMASKS, OUTLIERS, O_OFF,
            Bias, ExpertBounds, Padding, NLivePtr,
            LogitsWS, SelBitsWS, Counters, Starts, StartsCompact,
            Src2Dst, Src2Compact, Gammas, Rows,
            Tiles, nullptr, TileCompactStart, Row2Tok,
            RowAmaxW2, QValues, QScales, QRowScales, RowAmaxIn, WAmax, WDenominator,
            T, route_scale, base_exp, num_local, tile_sz, max_ctas,
            nvfp4, quant_fma, pdl, stream_ptr
        );
    }
}

// ---- Drop-In V9/V8 Backward-Compatible 2-Launch Entry Points (T = 1..256 & route_only) ----

extern "C" int router_bf12_c32_gemm_fwd(
    const void* X, const void* SLABS, const void* OUTLIERS, const void* O_OFF,
    const void* NLivePtr, void* Logits, int T, uint32_t base_exp,
    int pdl, int trig_dep, uintptr_t stream_ptr
) {
    if (T <= 0 || T > 256) return -1;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    const int m_blks = (T + 31) / 32;
    const bool pdl_wait = (pdl != 0);
    const bool trig = (trig_dep != 0);
    const bool use_pdl_attr = pdl_wait || trig;
    cudaError_t err = cudaSuccess;

    #define LAUNCH_G(EW, CW, PW, TR) err = launch_router_v10( \
        router_bf12_gemm_c32_v9_kernel<EW, CW, PW, TR>, \
        dim3(48 / (EW), m_blks, 1), dim3(160, 1, 1), stream, use_pdl_attr, \
        static_cast<const __nv_bfloat16*>(X), static_cast<const uint8_t*>(SLABS), \
        static_cast<const uint64_t*>(OUTLIERS), static_cast<const int32_t*>(O_OFF), \
        static_cast<const int32_t*>(NLivePtr), static_cast<float*>(Logits), T, base_exp)

    #define LAUNCH_BM64(PW, TR) err = launch_router_v10( \
        router_bf12_gemm_c48_v9_bm64_kernel<PW, TR>, \
        dim3(48, 1, 1), dim3(160, 1, 1), stream, use_pdl_attr, \
        static_cast<const __nv_bfloat16*>(X), static_cast<const uint8_t*>(SLABS), \
        static_cast<const uint64_t*>(OUTLIERS), static_cast<const int32_t*>(O_OFF), \
        static_cast<const int32_t*>(NLivePtr), static_cast<float*>(Logits), T, base_exp)

    #define DISPATCH_G(EW, CW) do { \
        if      (pdl_wait && trig)  { LAUNCH_G(EW, CW, true,  true);  } \
        else if (pdl_wait && !trig) { LAUNCH_G(EW, CW, true,  false); } \
        else if (!pdl_wait && trig) { LAUNCH_G(EW, CW, false, true);  } \
        else                        { LAUNCH_G(EW, CW, false, false); } \
    } while (0)

    if (T <= 32) {
        DISPATCH_G(1, false);
    } else if (T <= 64) {
        if      (pdl_wait && trig)  { LAUNCH_BM64(true,  true);  }
        else if (pdl_wait && !trig) { LAUNCH_BM64(true,  false); }
        else if (!pdl_wait && trig) { LAUNCH_BM64(false, true);  }
        else                        { LAUNCH_BM64(false, false); }
    } else if (T <= 96) {
        DISPATCH_G(3, true);
    } else if (T <= 128) {
        DISPATCH_G(4, true);
    } else if (T <= 192) {
        DISPATCH_G(6, true);
    } else {
        DISPATCH_G(4, true);
    }
    #undef DISPATCH_G
    #undef LAUNCH_BM64
    #undef LAUNCH_G
    return static_cast<int>(err);
}

extern "C" int router_c32_quant_route_place_v9_fwd(
    const void* X, const void* Logits, const void* Bias, const void* ExpertBounds,
    const void* Padding, const void* NLivePtr,
    void* Counts, void* Counters, void* Ids, void* Ranks, void* Weights,
    void* Starts, void* StartsCompact, void* Src2Dst, void* Src2Compact,
    void* Gammas, void* Rows, void* Tiles, void* TileCompactStart, void* Row2Tok,
    void* RowAmaxW2, void* QValues, void* QScales, void* QRowScales, void* RowAmaxIn,
    const void* WAmax, const void* WDenominator,
    int T, float route_scale, int num_local, int tile_sz, int max_ctas,
    int nvfp4, int logits_bf16, int pdl, int quant_fma, uintptr_t stream_ptr
) {
    if (T <= 0 || T > 256 || num_local <= 0 || num_local > E_FIXED) return -1;
    if (tile_sz <= 0 || max_ctas <= 0) return -2;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    const bool use_pdl = (pdl != 0);
    const int trig_mode = (pdl >= 3) ? 2 : ((pdl == 2) ? 1 : 0);
    const bool is_nv = (nvfp4 != 0);
    const bool is_bf16 = (logits_bf16 != 0);
    const bool use_fma = (quant_fma != 0);
    cudaError_t err = cudaSuccess;

    dim3 grid(T, 1, 1);
    dim3 block(384, 1, 1);
    #define LAUNCH_FUSED(NV, LB, PW, TM, QF) err = launch_router_v10( \
        c32_quant_topk_place_v9_kernel<NV, LB, PW, TM, QF>, grid, block, stream, (PW), \
        static_cast<const __nv_bfloat16*>(X), Logits, \
        static_cast<const float*>(Bias), static_cast<const int32_t*>(ExpertBounds), \
        static_cast<const bool*>(Padding), static_cast<const int32_t*>(NLivePtr), \
        static_cast<int32_t*>(Counts), static_cast<uint32_t*>(Counters), \
        static_cast<int32_t*>(Ids), static_cast<int32_t*>(Ranks), static_cast<float*>(Weights), \
        static_cast<int32_t*>(Starts), static_cast<int32_t*>(StartsCompact), \
        static_cast<int32_t*>(Src2Dst), static_cast<int32_t*>(Src2Compact), \
        static_cast<float*>(Gammas), static_cast<int32_t*>(Rows), \
        static_cast<int32_t*>(Tiles), static_cast<int32_t*>(TileCompactStart), \
        static_cast<int32_t*>(Row2Tok), static_cast<uint32_t*>(RowAmaxW2), \
        static_cast<uint8_t*>(QValues), static_cast<uint8_t*>(QScales), \
        static_cast<float*>(QRowScales), static_cast<float*>(RowAmaxIn), \
        static_cast<const float*>(WAmax), static_cast<const float*>(WDenominator), \
        T, route_scale, num_local, tile_sz, max_ctas)

    #define DISPATCH_FUSED(NV, LB, QF) do { \
        if      (use_pdl && trig_mode == 2) { LAUNCH_FUSED(NV, LB, true,  2, QF); } \
        else if (use_pdl && trig_mode == 1) { LAUNCH_FUSED(NV, LB, true,  1, QF); } \
        else if (use_pdl)                   { LAUNCH_FUSED(NV, LB, true,  0, QF); } \
        else                                { LAUNCH_FUSED(NV, LB, false, 0, QF); } \
    } while (0)

    if (is_nv) {
        if (use_fma) {
            if (is_bf16) { DISPATCH_FUSED(true, true,  true);  }
            else         { DISPATCH_FUSED(true, false, true);  }
        } else {
            if (is_bf16) { DISPATCH_FUSED(true, true,  false); }
            else         { DISPATCH_FUSED(true, false, false); }
        }
    } else {
        if (is_bf16) { DISPATCH_FUSED(false, true,  false); }
        else         { DISPATCH_FUSED(false, false, false); }
    }
    #undef DISPATCH_FUSED
    #undef LAUNCH_FUSED
    return static_cast<int>(err);
}

extern "C" int router_c32_quant_route_place_fwd(
    const void* X, const void* Logits, const void* Bias, const void* ExpertBounds,
    const void* Padding, const void* NLivePtr,
    void* Counts, void* Counters, void* Ids, void* Ranks, void* Weights,
    void* Starts, void* StartsCompact, void* Src2Dst, void* Src2Compact,
    void* Gammas, void* Rows, void* Tiles, void* TileCompactStart, void* Row2Tok,
    void* RowAmaxW2, void* QValues, void* QScales, void* QRowScales, void* RowAmaxIn,
    const void* WAmax, const void* WDenominator,
    int T, float route_scale, int num_local, int tile_sz, int max_ctas,
    int nvfp4, int logits_bf16, int pdl, uintptr_t stream_ptr
) {
    return router_c32_quant_route_place_v9_fwd(
        X, Logits, Bias, ExpertBounds, Padding, NLivePtr,
        Counts, Counters, Ids, Ranks, Weights,
        Starts, StartsCompact, Src2Dst, Src2Compact, Gammas, Rows,
        Tiles, TileCompactStart, Row2Tok, RowAmaxW2,
        QValues, QScales, QRowScales, RowAmaxIn, WAmax, WDenominator,
        T, route_scale, num_local, tile_sz, max_ctas,
        nvfp4, logits_bf16, pdl, 1, stream_ptr
    );
}
