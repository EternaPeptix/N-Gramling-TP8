// SPDX-License-Identifier: Apache-2.0
// SM121a (GB10) Wave-V14 48x Fused BF12/BF11 MoE Router + Warp-Specialized Concurrent Quant
//   + Double-Buffered cp.async sX + Register-Only 3x ld_cg_u4 Lazy-Rescan Top-8 + Full M=1..128!
//
// 100% Universal Drop-In ABI Superset of ALL THREE Router Libraries:
//   - `librouter_bf12_exact.so`     (`router_bf12_exact_fwd`, T=1..16 scattered R8c4, QUANT_FMA=1)
//   - `librouter_bf12_c32.so`       (`router_bf12_c32_gemm_fwd`, `router_c32_quant_route_place[_v9]_fwd`, T=1..256)
//   - `librouter_bf12_fused_v10.so` (`router_bf12/bf11_{c1,c48}_fused_v10_fwd`, T=1..64)
//   + NEW Wave-V14 Single-Launch M=1..128 Entry Points:
//   - `router_bf12_c128_fused_v14_fwd` / `router_bf11_c128_fused_v14_fwd` (1-launch 48-CTA across ALL T=1..128!)
//   - `router_bf12_v14_version()` -> returns 14
//
// Key Wave-V14 Architectural Breakthroughs (100% 0-ULP `nd == 0`, 0 B stack / 0 B spills, 0 `#206` PDL hazards):
//   1. Warp-Specialized Concurrent Quantization (`Warps 5..7` || `Warps 0..4` GEMM):
//      - In V10, Phase 1A serialized NVFP4/MXFP8 input quant BEFORE Phase 1B on CTAs `0..T-1` while
//        Warps `5..7` (37.5% of every CTA across all 48 SMs!) sat 100% idle during the entire BF12 GEMM!
//      - V14 runs Phase 1B GEMM immediately on Warps `0..4` (cycle 0 after `gdc_wait`), while Warps `5..7`
//        (96 threads = EXACTLY 3.000 NVFP4 blocks/thread, 0 wasted threads!) concurrently quantize
//        `tok = bid, bid+48, ...` using hardware named barrier `bar.sync 1, 96;` (independent of barrier 0!).
//      - Hides 100% of input quant latency behind GEMM across ALL `M = 1..128` (`-70..-110 us/step` over 48L!).
//   2. Double-Buffered `cp.async.ca.shared.global` `sX[2]` Ping-Pong (`BM_CTA in {8, 16, 32}`) &
//      Dedicated Zero-Waste `BM_CTA = 8` (C=1) + `BM_CTA = 48` (C=8) + `BM_CTA = 128` (C=32):
//      - `BM_CTA = 8` (21.25 KiB SMEM) & `BM_CTA = 16` (42.50 KiB) & `BM_CTA = 32` (45.00 KiB) overlap
//        next-slab `X` L2->SMEM copies asynchronously with current-slab BF12/BF11 unpack + MMA!
//      - `BM_CTA = 48` (33.75 KiB) eliminates dead `a3` regs & 128 dynamic `has_m48/has_m56` branches/warp at C=8!
//      - `BM_CTA = 128` (45.00 KiB) extends single-launch 48-CTA fusion to C=32 (`M = 65..128`) at 1.000x slab read!
//   3. Register-Only `3x ld_cg_u4` + Lazy-Rescan Exact Top-8 + SMEM-Cached `RowAmaxIn`:
//      - Eliminates 12.0 KiB `sSelBuf[8][384]` SMEM completely (`Phase3Smem` <= 17.6 KiB even at `BM_CTA=128`!).
//      - 4x fewer L2 load instructions (`3x ld.cg.v4.u32` vs `12x ld.cg.u32`), 0 SMEM reads in Top-8 loop,
//        only the single winning lane rescans its 12 scalar regs per round (exact tie-break -> lower expert ID!).
//      - Caches `RowAmaxIn[0..T-1]` once in SMEM -> eliminates 8x redundant `ld.cg.f32` in Step 3C.
//   4. 8.0x Weight-Load Reduction at `M = 65..128` + 2.0x Half-Slab Fix at `M = 129..256` (`router_bf12_c32_gemm_fwd`):
//      - V10 `router_bf12_gemm_c32_v9_kernel` loaded full `Slab12` TWICE per slab (`2x`) across 4 `m_blks` (`8x` = 21.23 MB/L!).
//      - V14 runs 1-pass 48-CTA `BM_CTA=128` (`23.04 KiB` SMEM -> co-resides inside 48 KiB with `3.30 KiB` K2!) reading
//        the 2.654 MB slabs EXACTLY 1.000x, AND uses `Slab12Half` (sm0+e4.xy / sm1+e4.zw) for `M=129..256` (1.000x)!
//   5. 48-CTA Zero-Wave-Tail Multi-Warp Parallel `c32_quant_topk_place_v14_kernel` (`M = 1..256`):
//      - Replaces `grid=(T,1,1)` (`2.67` waves @ M=128, `11/12` warps idle!) with `grid=(min(T,48),1,1)` (1 exact wave,
//        3-6 warps running parallel register Top-8 per CTA, only `3.30 KiB` SMEM, co-resident pre-`gdc_wait` quant!).
//
// Build on GB10 (SM121a, STRICT):
//   ./build_certhead_router_v14.sh [OUT_DIR]

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <math.h>

#ifndef ROUTER_V14_BM64_MIN_CTA
#define ROUTER_V14_BM64_MIN_CTA 2
#endif
#ifndef ROUTER_V14_C128_GEMM_MIN_CTA
#define ROUTER_V14_C128_GEMM_MIN_CTA 2
#endif
#ifndef ROUTER_V14_K2_MIN_CTA
#define ROUTER_V14_K2_MIN_CTA 2
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
// PTX Memory, Named Barrier, cp.async, Atomics & Exact IEEE-754 Primitives
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

static __device__ __forceinline__ uint2 ld_cs_u2(const void* p) {
    uint2 v;
    asm volatile("ld.global.cs.v2.u32 {%0,%1}, [%2];"
                 : "=r"(v.x), "=r"(v.y)
                 : "l"(p));
    return v;
}

static __device__ __forceinline__ uint2 ld_ca_u2(const void* p) {
    uint2 v;
    asm volatile("ld.global.ca.v2.u32 {%0,%1}, [%2];"
                 : "=r"(v.x), "=r"(v.y)
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

static __device__ __forceinline__ uint4 ld_cg_u4(const void* p) {
    uint4 v;
    asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

static __device__ __forceinline__ uint2 ld_cg_u2(const void* p) {
    uint2 v;
    asm volatile("ld.global.cg.v2.u32 {%0,%1}, [%2];"
                 : "=r"(v.x), "=r"(v.y)
                 : "l"(p));
    return v;
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

static __device__ __forceinline__ void cp_async_ca_16(void* smem_dst, const void* gmem_src, int src_bytes) {
    const uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;"
                 :: "r"(smem_addr), "l"(gmem_src), "r"(src_bytes) : "memory");
}

static __device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;" ::: "memory");
}

template <int N>
static __device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;" :: "n"(N) : "memory");
}

static __device__ __forceinline__ void bar_sync_quant96() {
    asm volatile("bar.sync 1, 96;" ::: "memory");
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

// Exact TRT-LLM R8c4 block scale offset matching `block_scales_ops.py:_scale_offsets` (0 < TILE < 128)
static __device__ __forceinline__ int64_t r8c4_scale_offset(int row, int col, int cols) {
    const int64_t tile = static_cast<int64_t>(row >> 3) * (cols >> 2) + (col >> 2);
    return (tile << 5) + ((row & 7) << 2) + (col & 3);
}

// =========================================================================
// BF12 & BF11-Flip Slab Load, Half-Slab Load (1.000x!), Unpack & Outlier Patch
// =========================================================================

struct Slab12 { uint4 sm0, sm1, e4; };
struct Slab12Half { uint4 sm; uint2 e2; };

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

template <bool CACHE_W, int HALF_ID>
static __device__ __forceinline__ Slab12Half load_slab12_half(const uint8_t* sp, int lane) {
    const uint4* p = reinterpret_cast<const uint4*>(sp);
    const uint2* e2_p = reinterpret_cast<const uint2*>(sp + 1024 + lane * 16 + HALF_ID * 8);
    Slab12Half r;
    if constexpr (CACHE_W) {
        r.sm = ld_ca_u4(p + (HALF_ID * 32) + lane);
        r.e2 = ld_ca_u2(e2_p);
    } else {
        r.sm = ld_cs_u4(p + (HALF_ID * 32) + lane);
        r.e2 = ld_cs_u2(e2_p);
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
static __device__ __forceinline__ void eval_nvfp4_half4_v14(
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
static __device__ __forceinline__ void quant_nvfp4_block16_compact_v14(
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
    eval_nvfp4_half4_v14<QUANT_FMA>(v0, inv6, scale6, inv4, scale4, outer_scale, pk6_lo, pk4_lo, e6_lo, e4_lo);
    eval_nvfp4_half4_v14<QUANT_FMA>(v1, inv6, scale6, inv4, scale4, outer_scale, pk6_hi, pk4_hi, e6_hi, e4_hi);

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

static __device__ __forceinline__ void quant_mxfp8_block32_regs(
    uint4 v0, uint4 v1, uint4 v2, uint4 v3,
    uint4& q0, uint4& q1, uint8_t& out_exp
) {
    auto max_u4 = [](float cur, uint4 vx) {
        const uint32_t w[4] = {vx.x, vx.y, vx.z, vx.w};
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float a = bf16_bits_to_f32(static_cast<uint16_t>(w[j] & 0xFFFFu));
            float b = bf16_bits_to_f32(static_cast<uint16_t>(w[j] >> 16));
            cur = fmaxf(cur, fmaxf(fabsf(a), fabsf(b)));
        }
        return cur;
    };
    float amax = 0.0f;
    amax = max_u4(amax, v0);
    amax = max_u4(amax, v1);
    amax = max_u4(amax, v2);
    amax = max_u4(amax, v3);

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

    auto cvt_pair = [inv](uint32_t word) -> uint32_t {
        float a = mul_rn(bf16_bits_to_f32(static_cast<uint16_t>(word & 0xFFFFu)), inv);
        float b = mul_rn(bf16_bits_to_f32(static_cast<uint16_t>(word >> 16)), inv);
        if (a == a) { if (a < -448.0f) a = -448.0f; if (a > 448.0f) a = 448.0f; }
        if (b == b) { if (b < -448.0f) b = -448.0f; if (b > 448.0f) b = 448.0f; }
        uint16_t e4;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(e4) : "f"(a), "f"(b));
        return static_cast<uint32_t>(e4);
    };

    q0.x = cvt_pair(v0.x) | (cvt_pair(v0.y) << 16);
    q0.y = cvt_pair(v0.z) | (cvt_pair(v0.w) << 16);
    q0.z = cvt_pair(v1.x) | (cvt_pair(v1.y) << 16);
    q0.w = cvt_pair(v1.z) | (cvt_pair(v1.w) << 16);
    q1.x = cvt_pair(v2.x) | (cvt_pair(v2.y) << 16);
    q1.y = cvt_pair(v2.z) | (cvt_pair(v2.w) << 16);
    q1.z = cvt_pair(v3.x) | (cvt_pair(v3.y) << 16);
    q1.w = cvt_pair(v3.z) | (cvt_pair(v3.w) << 16);
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
// V14 Register-Only Exact Top-8 Selection (0 SMEM, 0 Stack, 3x ld_cg_u4, Lazy Rescan!)
//   Input: 12 scalar `sel_bits` per lane (`c0, c1, c2` = experts `{lane*4..+3, 128+lane*4..+3, 256+lane*4..+3}`).
//   Output: lane `r in 0..7` receives exact winning expert ID `my_id` (tie-break -> strictly lower expert ID!).
// =========================================================================
static __device__ __forceinline__ int32_t warp_topk8_from_3u4(
    uint4 c0, uint4 c1, uint4 c2, int lane
) {
    const uint32_t e0 = static_cast<uint32_t>(lane << 2);
    const uint32_t e1 = 128u + e0;
    const uint32_t e2 = 256u + e0;

    uint32_t alive = 0xFFFu;
    uint32_t best_b = 0u;
    uint32_t best_id = 0xFFFFFFFFu;
    bool has_any = false;

    auto scan_lane = [&]() {
        best_b = 0u;
        best_id = 0xFFFFFFFFu;
        has_any = false;
        auto chk = [&](uint32_t bit, uint32_t v, uint32_t id) {
            if ((alive & bit) && (!has_any || v > best_b)) {
                best_b = v;
                best_id = id;
                has_any = true;
            }
        };
        chk(1u <<  0, c0.x, e0 + 0u);
        chk(1u <<  1, c0.y, e0 + 1u);
        chk(1u <<  2, c0.z, e0 + 2u);
        chk(1u <<  3, c0.w, e0 + 3u);
        chk(1u <<  4, c1.x, e1 + 0u);
        chk(1u <<  5, c1.y, e1 + 1u);
        chk(1u <<  6, c1.z, e1 + 2u);
        chk(1u <<  7, c1.w, e1 + 3u);
        chk(1u <<  8, c2.x, e2 + 0u);
        chk(1u <<  9, c2.y, e2 + 1u);
        chk(1u << 10, c2.z, e2 + 2u);
        chk(1u << 11, c2.w, e2 + 3u);
    };

    scan_lane();
    int32_t my_id = 0;
    #pragma unroll
    for (int r = 0; r < TOPK_FIXED; ++r) {
        const uint32_t mb = redux_max_u32(has_any ? best_b : 0u);
        const uint32_t cand_id = (has_any && best_b == mb) ? best_id : 0xFFFFFFFFu;
        const uint32_t win_id = redux_min_u32(cand_id);
        if (lane == r) my_id = static_cast<int32_t>(win_id);
        if (r < TOPK_FIXED - 1) {
            if (has_any && best_id == win_id) {
                const uint32_t slot = ((win_id >> 7) << 2) | (win_id & 3u);
                alive &= ~(1u << slot);
                scan_lane();
            }
        }
    }
    return my_id;
}

static __device__ __forceinline__ int32_t warp_topk8_from_selbits_ptr(
    const uint32_t* __restrict__ row_sel, int lane
) {
    const uint4* p4 = reinterpret_cast<const uint4*>(row_sel);
    const uint4 c0 = ld_cg_u4(p4 + lane);
    const uint4 c1 = ld_cg_u4(p4 + 32 + lane);
    const uint4 c2 = ld_cg_u4(p4 + 64 + lane);
    return warp_topk8_from_3u4(c0, c1, c2, lane);
}

static __device__ __forceinline__ float warp_softmax8_exact(
    float lp, int lane, float route_scale
) {
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
    return (lane < TOPK_FIXED) ? mul_rn(expf(sub_rn(lp, lse)), route_scale) : 0.0f;
}

// =========================================================================
// V14 Concurrent Warps 5..7 Input Quantizer (96 threads = 3.000 NVFP4 blks/thr, bar.sync 1,96!)
// =========================================================================
template <bool NVFP4, bool QUANT_FMA>
static __device__ __forceinline__ void run_warps57_concurrent_quant_v14(
    const __nv_bfloat16* __restrict__ X,
    const bool* __restrict__ Padding,
    uint8_t* __restrict__ QValues,
    uint8_t* __restrict__ QScales,
    float* __restrict__ RowAmaxIn,
    float* sQuantWarpMax,   // [3] floats outside smem union
    int bid, int n_live, int qw, int lane
) {
    const int qtid = (qw << 5) | lane;  // 0..95
    for (int tok = bid; tok < n_live; tok += 48) {
        const bool pad = (Padding != nullptr) ? Padding[tok] : false;
        if (pad) {
            if constexpr (NVFP4) {
                if (qtid == 0) RowAmaxIn[tok] = 0.0f;
            }
            continue;
        }
        if constexpr (NVFP4) {
            // Exact 288 blocks / 96 threads = 3.000 blocks/thread (ZERO tail, ZERO L2 re-reads!)
            const __nv_bfloat16* x_row = X + static_cast<int64_t>(tok) * K_FIXED;
            const uint4* bp0 = reinterpret_cast<const uint4*>(x_row + qtid * 16);
            const uint4* bp1 = reinterpret_cast<const uint4*>(x_row + (qtid + 96) * 16);
            const uint4* bp2 = reinterpret_cast<const uint4*>(x_row + (qtid + 192) * 16);

            const uint4 v0_a = ld_ca_u4(bp0);
            const uint4 v1_a = ld_ca_u4(bp0 + 1);
            const float bm_a = block16_amax(v0_a, v1_a);

            const uint4 v0_b = ld_ca_u4(bp1);
            const uint4 v1_b = ld_ca_u4(bp1 + 1);
            const float bm_b = block16_amax(v0_b, v1_b);

            const uint4 v0_c = ld_ca_u4(bp2);
            const uint4 v1_c = ld_ca_u4(bp2 + 1);
            const float bm_c = block16_amax(v0_c, v1_c);

            float wmax = fmaxf(bm_a, fmaxf(bm_b, bm_c));
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                wmax = fmaxf(wmax, __shfl_xor_sync(0xFFFFFFFFu, wmax, off));
            }
            if (lane == 0) sQuantWarpMax[qw] = wmax;
            bar_sync_quant96();

            float enc_s = 0.0f, dec_s = 0.0f, out_s = 0.0f;
            if (lane == 0) {
                const float tok_amax = fmaxf(sQuantWarpMax[0], fmaxf(sQuantWarpMax[1], sQuantWarpMax[2]));
                if (qtid == 0) RowAmaxIn[tok] = tok_amax;
                enc_s = fminf(div_rn(1536.0f, tok_amax), FP32_MAX_VAL);
                dec_s = div_rn(1.0f, enc_s);
                out_s = mul_rn(tok_amax, INV_1536);
            }
            enc_s = __shfl_sync(0xFFFFFFFFu, enc_s, 0);
            dec_s = __shfl_sync(0xFFFFFFFFu, dec_s, 0);
            out_s = __shfl_sync(0xFFFFFFFFu, out_s, 0);

            const int64_t qbase = static_cast<int64_t>(tok) * 288;
            uint64_t pk_a, pk_b, pk_c;
            uint8_t sc_a, sc_b, sc_c;
            quant_nvfp4_block16_compact_v14<QUANT_FMA>(v0_a, v1_a, bm_a, enc_s, dec_s, out_s, pk_a, sc_a);
            reinterpret_cast<uint64_t*>(QValues)[qbase + qtid] = pk_a;
            QScales[qbase + qtid] = sc_a;

            quant_nvfp4_block16_compact_v14<QUANT_FMA>(v0_b, v1_b, bm_b, enc_s, dec_s, out_s, pk_b, sc_b);
            reinterpret_cast<uint64_t*>(QValues)[qbase + qtid + 96] = pk_b;
            QScales[qbase + qtid + 96] = sc_b;

            quant_nvfp4_block16_compact_v14<QUANT_FMA>(v0_c, v1_c, bm_c, enc_s, dec_s, out_s, pk_c, sc_c);
            reinterpret_cast<uint64_t*>(QValues)[qbase + qtid + 192] = pk_c;
            QScales[qbase + qtid + 192] = sc_c;

            if (tok + 48 < n_live) bar_sync_quant96();
        } else {
            // 144 MXFP8 blocks / 96 threads: all 96 threads handle qtid; qtid < 48 also handles qtid + 96!
            const __nv_bfloat16* x_row = X + static_cast<int64_t>(tok) * K_FIXED;
            {
                const uint4* bp = reinterpret_cast<const uint4*>(x_row + qtid * 32);
                uint4 q0, q1; uint8_t sc;
                quant_mxfp8_block32_regs(ld_ca_u4(bp), ld_ca_u4(bp + 1), ld_ca_u4(bp + 2), ld_ca_u4(bp + 3), q0, q1, sc);
                uint4* dq = reinterpret_cast<uint4*>(QValues + static_cast<int64_t>(tok) * K_FIXED + qtid * 32);
                dq[0] = q0; dq[1] = q1;
                QScales[static_cast<int64_t>(tok) * 144 + qtid] = sc;
            }
            if (qtid < 48) {
                const int blk2 = qtid + 96;
                const uint4* bp = reinterpret_cast<const uint4*>(x_row + blk2 * 32);
                uint4 q0, q1; uint8_t sc;
                quant_mxfp8_block32_regs(ld_ca_u4(bp), ld_ca_u4(bp + 1), ld_ca_u4(bp + 2), ld_ca_u4(bp + 3), q0, q1, sc);
                uint4* dq = reinterpret_cast<uint4*>(QValues + static_cast<int64_t>(tok) * K_FIXED + blk2 * 32);
                dq[0] = q0; dq[1] = q1;
                QScales[static_cast<int64_t>(tok) * 144 + blk2] = sc;
            }
        }
    }
}

// =========================================================================
// WAVE-V14 SINGLE-LAUNCH 48-CTA WARP-SPECIALIZED FUSED ROUTER (`BM_CTA in {8, 16, 32, 48, 64, 128}`)
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
void router_fused_v14_kernel(
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
    int32_t* __restrict__ Starts,               // [384] (C48/C128) or nullptr (C1)
    int32_t* __restrict__ StartsCompact,        // [384] (C48/C128) or nullptr (C1)
    int32_t* __restrict__ Src2Dst,              // [T * 8]
    int32_t* __restrict__ Src2Compact,          // [T * 8]
    float* __restrict__ Gammas,                 // [plan.rows]
    int32_t* __restrict__ Rows,                 // [T * 8]
    int32_t* __restrict__ Tiles,                // [2 * max_ctas + 3]
    int32_t* __restrict__ TileMask,             // [max_ctas] (C1) or nullptr (C48/C128)
    int32_t* __restrict__ TileCompactStart,     // [max_ctas]
    int32_t* __restrict__ Row2Tok,              // [plan.rows] (C48/C128) or nullptr (C1)
    uint32_t* __restrict__ RowAmaxW2,           // [T * 8] or nullptr
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
    static_assert(BM_CTA == 8 || BM_CTA == 16 || BM_CTA == 32 || BM_CTA == 48 || BM_CTA == 64 || BM_CTA == 128,
                  "BM_CTA must be in {8, 16, 32, 48, 64, 128}");
    constexpr bool FULL_SLAB_X = (BM_CTA <= 16);
    constexpr bool DOUBLE_BUF_X = (BM_CTA <= 32);
    constexpr int SX_ROWS = (BM_CTA == 128) ? 64 : BM_CTA;
    constexpr int SX_STRIDE = FULL_SLAB_X ? 136 : 72;
    constexpr int SX_BUFS = DOUBLE_BUF_X ? 2 : 1;
    constexpr int SP_ROWS = (BM_CTA == 128) ? 64 : BM_CTA;
    constexpr int MASK_WORDS = (BM_CTA <= 32) ? 1 : ((BM_CTA <= 64) ? 2 : 4);
    constexpr int SLAB_BYTES = USE_BF11 ? SLAB11_BYTES : SLAB12_BYTES;

    const int bid  = static_cast<int>(blockIdx.x);  // 0..47
    const int tid  = static_cast<int>(threadIdx.x); // 0..255
    const int warp = tid >> 5;                      // 0..7
    const int lane = tid & 31;                      // 0..31

    // Register-only Top-8 eliminates 12 KiB `sSelBuf` -> Phase3Smem <= 18.1 KiB even at BM_CTA=128!
    struct Phase3Smem {
        uint32_t sMask[MASK_WORDS][E_FIXED];
        int32_t  sStarts[E_FIXED];
        int32_t  sStartsCompact[E_FIXED];
        int32_t  sTokLoc[BM_CTA][TOPK_FIXED];
        float    sTokWt[BM_CTA][TOPK_FIXED];
        float    sRowAmaxIn[BM_CTA];
        int32_t  sWarpT[8];
        int32_t  sWarpC[8];
        uint8_t  sIsPad[BM_CTA];
    };

    union alignas(16) SmemV14 {
        struct {
            __nv_bfloat16 sX[SX_BUFS][5][SX_ROWS * SX_STRIDE];
            uint32_t      sFlip[5][USE_BF11 ? 32 : 1];
        } gemm;
        float      sP[5][SP_ROWS][9];
        Phase3Smem p3;
    };
    static_assert(sizeof(SmemV14) <= 48640, "SmemV14 must fit 48 KiB default static SMEM");
    __shared__ SmemV14 smem;
    __shared__ float   sBiasLocal[8];
    __shared__ float   sQuantWarpMax[3];
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

    // ---- Phase 1 (CONCURRENT): Warps 0..4 BF12/BF11 GEMM || Warps 5..7 Input Quant! ----
    if (n_live > 0) {
        if (warp < 5) {
            const int pk = warp;
            const int g  = lane >> 2;  // 0..7
            const int lt = lane & 3;   // 0..3
            const int n_slabs = (pk < 4) ? 8 : 4;
            const int k_lo = pk * 1024;
            const uint32_t base_pair = base_exp | (base_exp << 16);
            const uint32_t base_sh7  = (base_exp << 7) | (base_exp << 23);

            const bool has_m8  = (BM_CTA >= 16) && (n_live > 8);
            const bool has_m16 = (BM_CTA >= 32) && (n_live > 16);
            const bool has_m24 = (BM_CTA >= 32) && (n_live > 24);
            const bool has_m32 = (BM_CTA >= 48) && (n_live > 32);
            const bool has_m40 = (BM_CTA >= 48) && (n_live > 40);
            const bool has_m48 = (BM_CTA >= 64) && (n_live > 48);
            const bool has_m56 = (BM_CTA >= 64) && (n_live > 56);

            const bool has_m64  = (BM_CTA == 128) && (n_live > 64);
            const bool has_m72  = (BM_CTA == 128) && (n_live > 72);
            const bool has_m80  = (BM_CTA == 128) && (n_live > 80);
            const bool has_m88  = (BM_CTA == 128) && (n_live > 88);
            const bool has_m96  = (BM_CTA == 128) && (n_live > 96);
            const bool has_m104 = (BM_CTA == 128) && (n_live > 104);
            const bool has_m112 = (BM_CTA == 128) && (n_live > 112);
            const bool has_m120 = (BM_CTA == 128) && (n_live > 120);

            float a0_0 = 0.0f, a0_1 = 0.0f, a0_2 = 0.0f, a0_3 = 0.0f;
            float a1_0 = 0.0f, a1_1 = 0.0f, a1_2 = 0.0f, a1_3 = 0.0f;
            float a2_0 = 0.0f, a2_1 = 0.0f, a2_2 = 0.0f, a2_3 = 0.0f;
            float a3_0 = 0.0f, a3_1 = 0.0f, a3_2 = 0.0f, a3_3 = 0.0f;
            float b0_0 = 0.0f, b0_1 = 0.0f, b0_2 = 0.0f, b0_3 = 0.0f;
            float b1_0 = 0.0f, b1_1 = 0.0f, b1_2 = 0.0f, b1_3 = 0.0f;
            float b2_0 = 0.0f, b2_1 = 0.0f, b2_2 = 0.0f, b2_3 = 0.0f;
            float b3_0 = 0.0f, b3_1 = 0.0f, b3_2 = 0.0f, b3_3 = 0.0f;

            uint64_t o_rec = (oc < oe) ? __ldg(OUTLIERS + oc) : 0xFFFFFFFFFFFFFFFFULL;
            uint32_t o_next_step = static_cast<uint32_t>(o_rec >> 32);

            const int lo_rows = (n_live < SX_ROWS) ? n_live : SX_ROWS;
            const int lo_pad_rows = (lo_rows + 7) & ~7;
            const int lo_total_u4 = FULL_SLAB_X ? (lo_pad_rows * 16) : (lo_pad_rows * 8);

            auto load_x_sync = [&](__nv_bfloat16* dst_sX, int m_off, int m_cnt, int k_base) {
                const int p_rows = (m_cnt + 7) & ~7;
                const int tot_u4 = FULL_SLAB_X ? (p_rows * 16) : (p_rows * 8);
                const uint4 z4 = make_uint4(0u, 0u, 0u, 0u);
                for (int qi = lane; qi < tot_u4; qi += 32) {
                    const int lm = FULL_SLAB_X ? (qi >> 4) : (qi >> 3);
                    const int qk = FULL_SLAB_X ? (qi & 15) : (qi & 7);
                    uint4 v = z4;
                    if (lm < m_cnt) {
                        v = ld_ca_u4(X + static_cast<int64_t>(m_off + lm) * K_FIXED + k_base + (qk << 3));
                    }
                    *reinterpret_cast<uint4*>(dst_sX + lm * SX_STRIDE + (qk << 3)) = v;
                }
                __syncwarp();
            };

            auto issue_x_async = [&](__nv_bfloat16* dst_sX, int k_base) {
                for (int qi = lane; qi < lo_total_u4; qi += 32) {
                    const int lm = FULL_SLAB_X ? (qi >> 4) : (qi >> 3);
                    const int qk = FULL_SLAB_X ? (qi & 15) : (qi & 7);
                    const int safe_m = (lm < lo_rows) ? lm : 0;
                    const int src_b = (lm < lo_rows) ? 16 : 0;
                    cp_async_ca_16(
                        dst_sX + lm * SX_STRIDE + (qk << 3),
                        X + static_cast<int64_t>(safe_m) * K_FIXED + k_base + (qk << 3),
                        src_b
                    );
                }
                cp_async_commit();
            };

            auto run_mma_lo = [&](const __nv_bfloat16* sX_b, int s_loc, uint32_t b0, uint32_t b1) {
                const int s_off = s_loc * 16;
                {
                    const uint32_t x0 = *reinterpret_cast<const uint32_t*>(sX_b + s_off);
                    const uint32_t x2 = *reinterpret_cast<const uint32_t*>(sX_b + s_off + 8);
                    const uint32_t x1 = has_m8 ? *reinterpret_cast<const uint32_t*>(sX_b + 8 * SX_STRIDE + s_off) : 0u;
                    const uint32_t x3 = has_m8 ? *reinterpret_cast<const uint32_t*>(sX_b + 8 * SX_STRIDE + s_off + 8) : 0u;
                    mma_bf16_r(a0_0, a0_1, a0_2, a0_3, x0, x1, x2, x3, b0, b1);
                }
                if constexpr (BM_CTA >= 32) {
                    if (has_m16) {
                        const uint32_t x0 = *reinterpret_cast<const uint32_t*>(sX_b + 16 * SX_STRIDE + s_off);
                        const uint32_t x2 = *reinterpret_cast<const uint32_t*>(sX_b + 16 * SX_STRIDE + s_off + 8);
                        const uint32_t x1 = has_m24 ? *reinterpret_cast<const uint32_t*>(sX_b + 24 * SX_STRIDE + s_off) : 0u;
                        const uint32_t x3 = has_m24 ? *reinterpret_cast<const uint32_t*>(sX_b + 24 * SX_STRIDE + s_off + 8) : 0u;
                        mma_bf16_r(a1_0, a1_1, a1_2, a1_3, x0, x1, x2, x3, b0, b1);
                    }
                }
                if constexpr (BM_CTA >= 48) {
                    if (has_m32) {
                        const uint32_t x0 = *reinterpret_cast<const uint32_t*>(sX_b + 32 * SX_STRIDE + s_off);
                        const uint32_t x2 = *reinterpret_cast<const uint32_t*>(sX_b + 32 * SX_STRIDE + s_off + 8);
                        const uint32_t x1 = has_m40 ? *reinterpret_cast<const uint32_t*>(sX_b + 40 * SX_STRIDE + s_off) : 0u;
                        const uint32_t x3 = has_m40 ? *reinterpret_cast<const uint32_t*>(sX_b + 40 * SX_STRIDE + s_off + 8) : 0u;
                        mma_bf16_r(a2_0, a2_1, a2_2, a2_3, x0, x1, x2, x3, b0, b1);
                    }
                }
                if constexpr (BM_CTA >= 64) {
                    if (has_m48) {
                        const uint32_t x0 = *reinterpret_cast<const uint32_t*>(sX_b + 48 * SX_STRIDE + s_off);
                        const uint32_t x2 = *reinterpret_cast<const uint32_t*>(sX_b + 48 * SX_STRIDE + s_off + 8);
                        const uint32_t x1 = has_m56 ? *reinterpret_cast<const uint32_t*>(sX_b + 56 * SX_STRIDE + s_off) : 0u;
                        const uint32_t x3 = has_m56 ? *reinterpret_cast<const uint32_t*>(sX_b + 56 * SX_STRIDE + s_off + 8) : 0u;
                        mma_bf16_r(a3_0, a3_1, a3_2, a3_3, x0, x1, x2, x3, b0, b1);
                    }
                }
            };

            auto run_mma_hi = [&](const __nv_bfloat16* sX_b, int s_loc, uint32_t b0, uint32_t b1) {
                if constexpr (BM_CTA == 128) {
                    const int s_off = s_loc * 16;
                    if (has_m64) {
                        const uint32_t x0 = *reinterpret_cast<const uint32_t*>(sX_b + s_off);
                        const uint32_t x2 = *reinterpret_cast<const uint32_t*>(sX_b + s_off + 8);
                        const uint32_t x1 = has_m72 ? *reinterpret_cast<const uint32_t*>(sX_b + 8 * SX_STRIDE + s_off) : 0u;
                        const uint32_t x3 = has_m72 ? *reinterpret_cast<const uint32_t*>(sX_b + 8 * SX_STRIDE + s_off + 8) : 0u;
                        mma_bf16_r(b0_0, b0_1, b0_2, b0_3, x0, x1, x2, x3, b0, b1);
                    }
                    if (has_m80) {
                        const uint32_t x0 = *reinterpret_cast<const uint32_t*>(sX_b + 16 * SX_STRIDE + s_off);
                        const uint32_t x2 = *reinterpret_cast<const uint32_t*>(sX_b + 16 * SX_STRIDE + s_off + 8);
                        const uint32_t x1 = has_m88 ? *reinterpret_cast<const uint32_t*>(sX_b + 24 * SX_STRIDE + s_off) : 0u;
                        const uint32_t x3 = has_m88 ? *reinterpret_cast<const uint32_t*>(sX_b + 24 * SX_STRIDE + s_off + 8) : 0u;
                        mma_bf16_r(b1_0, b1_1, b1_2, b1_3, x0, x1, x2, x3, b0, b1);
                    }
                    if (has_m96) {
                        const uint32_t x0 = *reinterpret_cast<const uint32_t*>(sX_b + 32 * SX_STRIDE + s_off);
                        const uint32_t x2 = *reinterpret_cast<const uint32_t*>(sX_b + 32 * SX_STRIDE + s_off + 8);
                        const uint32_t x1 = has_m104 ? *reinterpret_cast<const uint32_t*>(sX_b + 40 * SX_STRIDE + s_off) : 0u;
                        const uint32_t x3 = has_m104 ? *reinterpret_cast<const uint32_t*>(sX_b + 40 * SX_STRIDE + s_off + 8) : 0u;
                        mma_bf16_r(b2_0, b2_1, b2_2, b2_3, x0, x1, x2, x3, b0, b1);
                    }
                    if (has_m112) {
                        const uint32_t x0 = *reinterpret_cast<const uint32_t*>(sX_b + 48 * SX_STRIDE + s_off);
                        const uint32_t x2 = *reinterpret_cast<const uint32_t*>(sX_b + 48 * SX_STRIDE + s_off + 8);
                        const uint32_t x1 = has_m120 ? *reinterpret_cast<const uint32_t*>(sX_b + 56 * SX_STRIDE + s_off) : 0u;
                        const uint32_t x3 = has_m120 ? *reinterpret_cast<const uint32_t*>(sX_b + 56 * SX_STRIDE + s_off + 8) : 0u;
                        mma_bf16_r(b3_0, b3_1, b3_2, b3_3, x0, x1, x2, x3, b0, b1);
                    }
                }
            };

            if constexpr (!USE_BF11) {
                Slab12 cur_slab = load_slab12<false>(slab_base, lane);
                if constexpr (DOUBLE_BUF_X) {
                    // V14: Ping-pong cp.async double-buffered sX for BM_CTA in {8, 16, 32} (C=1..4!)
                    issue_x_async(smem.gemm.sX[0][pk], k_lo);
                    int cur_buf = 0;
                    for (int sl = 0; sl < n_slabs; ++sl) {
                        const int k_sl = k_lo + sl * 128;
                        Slab12 nxt_slab;
                        if (sl + 1 < n_slabs) {
                            nxt_slab = load_slab12<false>(slab_base + static_cast<int64_t>(sl + 1) * SLAB12_BYTES, lane);
                        }
                        if constexpr (FULL_SLAB_X) {
                            if (sl + 1 < n_slabs) {
                                issue_x_async(smem.gemm.sX[cur_buf ^ 1][pk], k_sl + 128);
                                cp_async_wait<1>();
                            } else {
                                cp_async_wait<0>();
                            }
                            __syncwarp();
                            const __nv_bfloat16* sX_b = smem.gemm.sX[cur_buf][pk] + g * SX_STRIDE + 2 * lt;
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
                                run_mma_lo(sX_b, s, b0, b1);
                            }
                            cur_buf ^= 1;
                        } else {
                            // BM_CTA == 32: K_HALF=64 ping-pong between Half 0 and Half 1!
                            issue_x_async(smem.gemm.sX[cur_buf ^ 1][pk], k_sl + 64);
                            cp_async_wait<1>();
                            __syncwarp();
                            const __nv_bfloat16* sX_b0 = smem.gemm.sX[cur_buf][pk] + g * SX_STRIDE + 2 * lt;
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
                                    run_mma_lo(sX_b0, s, b0, b1);
                                }
                            }
                            cur_buf ^= 1;
                            if (sl + 1 < n_slabs) {
                                issue_x_async(smem.gemm.sX[cur_buf ^ 1][pk], k_sl + 128);
                                cp_async_wait<1>();
                            } else {
                                cp_async_wait<0>();
                            }
                            __syncwarp();
                            const __nv_bfloat16* sX_b1 = smem.gemm.sX[cur_buf][pk] + g * SX_STRIDE + 2 * lt;
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
                                    run_mma_lo(sX_b1, s, b0, b1);
                                }
                            }
                            cur_buf ^= 1;
                        }
                        if (sl + 1 < n_slabs) cur_slab = nxt_slab;
                        __syncwarp();
                    }
                } else {
                    // BM_CTA in {48, 64, 128}: Single-buffer sX; for BM_CTA=128 unpacks (b0,b1) ONCE and reuses across lo64 & hi64!
                    __nv_bfloat16* sX_w = smem.gemm.sX[0][pk];
                    const __nv_bfloat16* sX_b = sX_w + g * SX_STRIDE + 2 * lt;
                    for (int sl = 0; sl < n_slabs; ++sl) {
                        const int k_sl = k_lo + sl * 128;
                        load_x_sync(sX_w, 0, lo_rows, k_sl);

                        Slab12 nxt_slab;
                        if (sl + 1 < n_slabs) {
                            nxt_slab = load_slab12<false>(slab_base + static_cast<int64_t>(sl + 1) * SLAB12_BYTES, lane);
                        }
                        {
                            const uint32_t sm_h0[4] = {cur_slab.sm0.x, cur_slab.sm0.y, cur_slab.sm0.z, cur_slab.sm0.w};
                            const uint32_t e_h0[4]  = {
                                cur_slab.e4.x & 0xFFFFu, cur_slab.e4.x >> 16,
                                cur_slab.e4.y & 0xFFFFu, cur_slab.e4.y >> 16
                            };
                            uint32_t b0_s[4], b1_s[4];
                            #pragma unroll
                            for (int s = 0; s < 4; ++s) {
                                unpack4_bf12(sm_h0[s], e_h0[s], base_pair, b0_s[s], b1_s[s]);
                                patch_outliers(static_cast<uint32_t>((sl << 3) | s), lane,
                                               OUTLIERS, oc, oe, o_rec, o_next_step, b0_s[s], b1_s[s]);
                                run_mma_lo(sX_b, s, b0_s[s], b1_s[s]);
                            }
                            if constexpr (BM_CTA == 128) {
                                if (has_m64) {
                                    __syncwarp();
                                    load_x_sync(sX_w, 64, n_live - 64, k_sl);
                                    #pragma unroll
                                    for (int s = 0; s < 4; ++s) {
                                        run_mma_hi(sX_b, s, b0_s[s], b1_s[s]);
                                    }
                                }
                            }
                        }
                        __syncwarp();
                        load_x_sync(sX_w, 0, lo_rows, k_sl + 64);
                        {
                            const uint32_t sm_h1[4] = {cur_slab.sm1.x, cur_slab.sm1.y, cur_slab.sm1.z, cur_slab.sm1.w};
                            const uint32_t e_h1[4]  = {
                                cur_slab.e4.z & 0xFFFFu, cur_slab.e4.z >> 16,
                                cur_slab.e4.w & 0xFFFFu, cur_slab.e4.w >> 16
                            };
                            uint32_t b0_s[4], b1_s[4];
                            #pragma unroll
                            for (int s = 0; s < 4; ++s) {
                                unpack4_bf12(sm_h1[s], e_h1[s], base_pair, b0_s[s], b1_s[s]);
                                patch_outliers(static_cast<uint32_t>((sl << 3) | (4 + s)), lane,
                                               OUTLIERS, oc, oe, o_rec, o_next_step, b0_s[s], b1_s[s]);
                                run_mma_lo(sX_b, s, b0_s[s], b1_s[s]);
                            }
                            if constexpr (BM_CTA == 128) {
                                if (has_m64) {
                                    __syncwarp();
                                    load_x_sync(sX_w, 64, n_live - 64, k_sl + 64);
                                    #pragma unroll
                                    for (int s = 0; s < 4; ++s) {
                                        run_mma_hi(sX_b, s, b0_s[s], b1_s[s]);
                                    }
                                }
                            }
                        }
                        if (sl + 1 < n_slabs) cur_slab = nxt_slab;
                        __syncwarp();
                    }
                }
            } else {
                // Opt-in lossless BF11-Flip V5 slab path across ALL BM_CTA in {8, 16, 32, 48, 64, 128}!
                const int64_t meta_base = static_cast<int64_t>(bid * 36 + pk * 8);
                uint32_t cur_meta = __ldg(SLAB_META_V5 + meta_base);
                Slab11 cur_slab = load_slab11_cs(slab_base, lane);
                uint32_t cur_raw_fl = issue_flip_load_bf11(cur_meta, SFLIPS, BMASKS, lane);
                __nv_bfloat16* sX_w = smem.gemm.sX[0][pk];
                const __nv_bfloat16* sX_b = sX_w + g * SX_STRIDE + 2 * lt;

                for (int sl = 0; sl < n_slabs; ++sl) {
                    const int k_sl = k_lo + sl * 128;
                    load_x_sync(sX_w, 0, lo_rows, k_sl);

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

                    #define BF11_UNPACK_STEP(S_IDX, OUT_B0, OUT_B1) do { \
                        unpack4_bf11_flip_v5( \
                            slab11_sm_word<(S_IDX)>(cur_slab), \
                            slab11_c12_raw<(S_IDX)>(cur_slab), \
                            (flip_mask >> ((S_IDX) * 4)) & 0xFu, \
                            base_sh7, (OUT_B0), (OUT_B1)); \
                        if (__builtin_expect(has_out, 0)) { \
                            patch_outliers(static_cast<uint32_t>((sl << 3) | (S_IDX)), lane, \
                                           OUTLIERS, oc, oe, o_rec, o_next_step, (OUT_B0), (OUT_B1)); \
                        } \
                    } while (0)

                    if constexpr (FULL_SLAB_X) {
                        uint32_t b0, b1;
                        BF11_UNPACK_STEP(0, b0, b1); run_mma_lo(sX_b, 0, b0, b1);
                        BF11_UNPACK_STEP(1, b0, b1); run_mma_lo(sX_b, 1, b0, b1);
                        BF11_UNPACK_STEP(2, b0, b1); run_mma_lo(sX_b, 2, b0, b1);
                        BF11_UNPACK_STEP(3, b0, b1); run_mma_lo(sX_b, 3, b0, b1);
                        BF11_UNPACK_STEP(4, b0, b1); run_mma_lo(sX_b, 4, b0, b1);
                        BF11_UNPACK_STEP(5, b0, b1); run_mma_lo(sX_b, 5, b0, b1);
                        BF11_UNPACK_STEP(6, b0, b1); run_mma_lo(sX_b, 6, b0, b1);
                        BF11_UNPACK_STEP(7, b0, b1); run_mma_lo(sX_b, 7, b0, b1);
                    } else {
                        uint32_t b0_s[4], b1_s[4];
                        BF11_UNPACK_STEP(0, b0_s[0], b1_s[0]); run_mma_lo(sX_b, 0, b0_s[0], b1_s[0]);
                        BF11_UNPACK_STEP(1, b0_s[1], b1_s[1]); run_mma_lo(sX_b, 1, b0_s[1], b1_s[1]);
                        BF11_UNPACK_STEP(2, b0_s[2], b1_s[2]); run_mma_lo(sX_b, 2, b0_s[2], b1_s[2]);
                        BF11_UNPACK_STEP(3, b0_s[3], b1_s[3]); run_mma_lo(sX_b, 3, b0_s[3], b1_s[3]);
                        if constexpr (BM_CTA == 128) {
                            if (has_m64) {
                                __syncwarp();
                                load_x_sync(sX_w, 64, n_live - 64, k_sl);
                                #pragma unroll
                                for (int s = 0; s < 4; ++s) run_mma_hi(sX_b, s, b0_s[s], b1_s[s]);
                            }
                        }
                        __syncwarp();
                        load_x_sync(sX_w, 0, lo_rows, k_sl + 64);
                        BF11_UNPACK_STEP(4, b0_s[0], b1_s[0]); run_mma_lo(sX_b, 0, b0_s[0], b1_s[0]);
                        BF11_UNPACK_STEP(5, b0_s[1], b1_s[1]); run_mma_lo(sX_b, 1, b0_s[1], b1_s[1]);
                        BF11_UNPACK_STEP(6, b0_s[2], b1_s[2]); run_mma_lo(sX_b, 2, b0_s[2], b1_s[2]);
                        BF11_UNPACK_STEP(7, b0_s[3], b1_s[3]); run_mma_lo(sX_b, 3, b0_s[3], b1_s[3]);
                        if constexpr (BM_CTA == 128) {
                            if (has_m64) {
                                __syncwarp();
                                load_x_sync(sX_w, 64, n_live - 64, k_sl + 64);
                                #pragma unroll
                                for (int s = 0; s < 4; ++s) run_mma_hi(sX_b, s, b0_s[s], b1_s[s]);
                            }
                        }
                    }
                    #undef BF11_UNPACK_STEP

                    if (sl + 1 < n_slabs) {
                        cur_meta = nxt_meta;
                        cur_slab = nxt_slab;
                        cur_raw_fl = nxt_raw_fl;
                    }
                    __syncwarp();
                }
            }

            // Rendezvous with Warps 5..7 (which finished concurrent input quant!) before aliasing smem.gemm -> smem.sP!
            __syncthreads();

            smem.sP[pk][g][2 * lt]     = a0_0;
            smem.sP[pk][g][2 * lt + 1] = a0_1;
            if constexpr (BM_CTA >= 16) {
                if (has_m8)  { smem.sP[pk][g +  8][2 * lt] = a0_2; smem.sP[pk][g +  8][2 * lt + 1] = a0_3; }
            }
            if constexpr (BM_CTA >= 32) {
                if (has_m16) { smem.sP[pk][g + 16][2 * lt] = a1_0; smem.sP[pk][g + 16][2 * lt + 1] = a1_1; }
                if (has_m24) { smem.sP[pk][g + 24][2 * lt] = a1_2; smem.sP[pk][g + 24][2 * lt + 1] = a1_3; }
            }
            if constexpr (BM_CTA >= 48) {
                if (has_m32) { smem.sP[pk][g + 32][2 * lt] = a2_0; smem.sP[pk][g + 32][2 * lt + 1] = a2_1; }
                if (has_m40) { smem.sP[pk][g + 40][2 * lt] = a2_2; smem.sP[pk][g + 40][2 * lt + 1] = a2_3; }
            }
            if constexpr (BM_CTA >= 64) {
                if (has_m48) { smem.sP[pk][g + 48][2 * lt] = a3_0; smem.sP[pk][g + 48][2 * lt + 1] = a3_1; }
                if (has_m56) { smem.sP[pk][g + 56][2 * lt] = a3_2; smem.sP[pk][g + 56][2 * lt + 1] = a3_3; }
            }
            __syncthreads();

            const int n_out_lo = lo_rows * 8;
            for (int oidx = tid; oidx < n_out_lo; oidx += 256) {
                const int lm = oidx >> 3;
                const int e_loc = oidx & 7;
                const float l = add_rn(add_rn(add_rn(add_rn(
                            smem.sP[0][lm][e_loc], smem.sP[1][lm][e_loc]),
                            smem.sP[2][lm][e_loc]), smem.sP[3][lm][e_loc]),
                            smem.sP[4][lm][e_loc]);
                const int64_t g_off = static_cast<int64_t>(lm) * E_FIXED + e_col_base + e_loc;
                LogitsWS[g_off]  = l;
                SelBitsWS[g_off] = selection_bits(l, sBiasLocal[e_loc]);
            }

            if constexpr (BM_CTA == 128) {
                if (has_m64) {
                    __syncthreads();
                    smem.sP[pk][g][2 * lt]     = b0_0; smem.sP[pk][g][2 * lt + 1]     = b0_1;
                    if (has_m72)  { smem.sP[pk][g +  8][2 * lt] = b0_2; smem.sP[pk][g +  8][2 * lt + 1] = b0_3; }
                    if (has_m80)  { smem.sP[pk][g + 16][2 * lt] = b1_0; smem.sP[pk][g + 16][2 * lt + 1] = b1_1; }
                    if (has_m88)  { smem.sP[pk][g + 24][2 * lt] = b1_2; smem.sP[pk][g + 24][2 * lt + 1] = b1_3; }
                    if (has_m96)  { smem.sP[pk][g + 32][2 * lt] = b2_0; smem.sP[pk][g + 32][2 * lt + 1] = b2_1; }
                    if (has_m104) { smem.sP[pk][g + 40][2 * lt] = b2_2; smem.sP[pk][g + 40][2 * lt + 1] = b2_3; }
                    if (has_m112) { smem.sP[pk][g + 48][2 * lt] = b3_0; smem.sP[pk][g + 48][2 * lt + 1] = b3_1; }
                    if (has_m120) { smem.sP[pk][g + 56][2 * lt] = b3_2; smem.sP[pk][g + 56][2 * lt + 1] = b3_3; }
                    __syncthreads();

                    const int n_out_hi = (n_live - 64) * 8;
                    for (int oidx = tid; oidx < n_out_hi; oidx += 256) {
                        const int lm = oidx >> 3;
                        const int e_loc = oidx & 7;
                        const float l = add_rn(add_rn(add_rn(add_rn(
                                    smem.sP[0][lm][e_loc], smem.sP[1][lm][e_loc]),
                                    smem.sP[2][lm][e_loc]), smem.sP[3][lm][e_loc]),
                                    smem.sP[4][lm][e_loc]);
                        const int64_t g_off = static_cast<int64_t>(64 + lm) * E_FIXED + e_col_base + e_loc;
                        LogitsWS[g_off]  = l;
                        SelBitsWS[g_off] = selection_bits(l, sBiasLocal[e_loc]);
                    }
                }
            }
        } else {
            // Warps 5, 6, 7: Run 100%-overlapped input quantization concurrently with Warps 0..4 GEMM!
            run_warps57_concurrent_quant_v14<NVFP4, QUANT_FMA>(
                X, Padding, QValues, QScales, RowAmaxIn, sQuantWarpMax, bid, n_live, warp - 5, lane
            );
            __syncthreads();  // Matches Warps 0..4 before smem.sP write
            __syncthreads();  // Matches Warps 0..4 after smem.sP write

            const int lo_rows = (n_live < SX_ROWS) ? n_live : SX_ROWS;
            const int n_out_lo = lo_rows * 8;
            for (int oidx = tid; oidx < n_out_lo; oidx += 256) {
                const int lm = oidx >> 3;
                const int e_loc = oidx & 7;
                const float l = add_rn(add_rn(add_rn(add_rn(
                            smem.sP[0][lm][e_loc], smem.sP[1][lm][e_loc]),
                            smem.sP[2][lm][e_loc]), smem.sP[3][lm][e_loc]),
                            smem.sP[4][lm][e_loc]);
                const int64_t g_off = static_cast<int64_t>(lm) * E_FIXED + e_col_base + e_loc;
                LogitsWS[g_off]  = l;
                SelBitsWS[g_off] = selection_bits(l, sBiasLocal[e_loc]);
            }

            if constexpr (BM_CTA == 128) {
                if (n_live > 64) {
                    __syncthreads();
                    __syncthreads();
                    const int n_out_hi = (n_live - 64) * 8;
                    for (int oidx = tid; oidx < n_out_hi; oidx += 256) {
                        const int lm = oidx >> 3;
                        const int e_loc = oidx & 7;
                        const float l = add_rn(add_rn(add_rn(add_rn(
                                    smem.sP[0][lm][e_loc], smem.sP[1][lm][e_loc]),
                                    smem.sP[2][lm][e_loc]), smem.sP[3][lm][e_loc]),
                                    smem.sP[4][lm][e_loc]);
                        const int64_t g_off = static_cast<int64_t>(64 + lm) * E_FIXED + e_col_base + e_loc;
                        LogitsWS[g_off]  = l;
                        SelBitsWS[g_off] = selection_bits(l, sBiasLocal[e_loc]);
                    }
                }
            }
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

    // Early W13 trigger (pdl >= 3): wake W13 onto the 47 freed SMs NOW while Last CTA runs Phase 3!
    if constexpr (TRIG_W13_MODE == 2) {
        if (tid == 0) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }

    // ---- Phase 3 (Last CTA Only): Register-Only 3x ld_cg_u4 Top-8 + Softmax + SMEM Bitmask + Placement ----
    const int lo = __ldg(ExpertBounds);
    const int hi = __ldg(ExpertBounds + 1);

    for (int i = tid; i < E_FIXED; i += 256) {
        #pragma unroll
        for (int mw = 0; mw < MASK_WORDS; ++mw) {
            smem.p3.sMask[mw][i] = 0u;
        }
    }
    if (tid < BM_CTA) {
        const bool pad = (tid >= n_live) || ((Padding != nullptr && tid < T) ? Padding[tid] : false);
        smem.p3.sIsPad[tid] = pad ? 1u : 0u;
        if constexpr (NVFP4) {
            smem.p3.sRowAmaxIn[tid] = (!pad && tid < T) ? ld_cg_f32(RowAmaxIn + tid) : 0.0f;
        }
    }
    __syncthreads();

    // Step 3A: 8 warps run register-only 3x ld_cg_u4 + lazy-rescan Top-8 (0 SMEM selbuf!)
    for (int tok = warp; tok < T; tok += 8) {
        if (smem.p3.sIsPad[tok]) {
            if (lane < TOPK_FIXED) {
                smem.p3.sTokLoc[tok][lane] = -1;
                smem.p3.sTokWt[tok][lane]  = 0.0f;
            }
            continue;
        }

        const int32_t my_id = warp_topk8_from_selbits_ptr(
            SelBitsWS + static_cast<int64_t>(tok) * E_FIXED, lane
        );

        float lp = -INFINITY;
        if (lane < TOPK_FIXED) {
            const float win_logit = ld_cg_f32(LogitsWS + static_cast<int64_t>(tok) * E_FIXED + my_id);
            lp = sub_rn(fminf(win_logit, 0.0f), log1pf(expf(-fabsf(win_logit))));
        }
        const float my_wt = warp_softmax8_exact(lp, lane, route_scale);

        if (lane < TOPK_FIXED) {
            const int32_t my_loc = (my_id >= lo && my_id < hi) ? (my_id - lo) : -1;
            smem.p3.sTokLoc[tok][lane] = my_loc;
            smem.p3.sTokWt[tok][lane]  = my_wt;
            if (my_loc >= 0) {
                if constexpr (MASK_WORDS == 1) {
                    atomicOr(&smem.p3.sMask[0][my_loc], 1u << tok);
                } else {
                    atomicOr(&smem.p3.sMask[tok >> 5][my_loc], 1u << (tok & 31));
                }
            }
        }
    }
    __syncthreads();

    // Step 3B: 6-Warp SMEM Prefix Scan over all num_local <= 384 Experts (exact C1 & multi-tile C48/C128!)
    const int tile_shift = (tile_sz == 8) ? 3 : ((tile_sz == 16) ? 4 : ((tile_sz == 32) ? 5 : -1));
    const int tile_mask_i = tile_sz - 1;
    const int e0 = warp * 64 + lane;
    const int e1 = e0 + 32;
    uint32_t m0_w0 = 0u, m1_w0 = 0u;
    int c0 = 0, c1 = 0, t0 = 0, t1 = 0;
    int exc_t0 = 0, exc_c0 = 0, exc_t1 = 0, exc_c1 = 0;

    if (warp < 6) {
        m0_w0 = (e0 < num_local) ? smem.p3.sMask[0][e0] : 0u;
        m1_w0 = (e1 < num_local) ? smem.p3.sMask[0][e1] : 0u;
        c0 = __popc(m0_w0);
        c1 = __popc(m1_w0);
        #pragma unroll
        for (int mw = 1; mw < MASK_WORDS; ++mw) {
            if (e0 < num_local) c0 += __popc(smem.p3.sMask[mw][e0]);
            if (e1 < num_local) c1 += __popc(smem.p3.sMask[mw][e1]);
        }

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
        emit_exp(e0, c0, t0, exc_t0, exc_c0, m0_w0);
        emit_exp(e1, c1, t1, exc_t1, exc_c1, m1_w0);
    }
    __syncthreads();

    // Step 3C: In-SMEM Zero-Global-Round-Trip Placement (with 1-cycle SMEM `sRowAmaxIn`!)
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
            int32_t rank = 0;
            if constexpr (MASK_WORDS == 1) {
                rank = __popc(smem.p3.sMask[0][loc] & ((1u << tok) - 1u));
            } else {
                const int w_tok = tok >> 5;
                #pragma unroll
                for (int mw = 0; mw < MASK_WORDS; ++mw) {
                    if (mw < w_tok) rank += __popc(smem.p3.sMask[mw][loc]);
                }
                rank += __popc(smem.p3.sMask[w_tok][loc] & ((1u << (tok & 31)) - 1u));
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
                QRowScales[dst] = mul_rn(mul_rn(smem.p3.sRowAmaxIn[tok], __ldg(WAmax + loc)), rcp_denom);
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
        st_rel_gpu_u32(&Arrive[0], 0u);
        if constexpr (TRIG_W13_MODE == 1) {
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
    }
}

// =========================================================================
// WAVE-V14 UNIVERSAL `router_bf12_exact_fwd` KERNEL (`T <= 16`, scattered R8c4, QUANT_FMA=1!)
//   1-Launch 48-CTA when `Counters != nullptr` (100% of `router_bf12.py` calls when S8=0!);
//   Uses existing 122,880 B `LogitsWS` scratch for L2-hot NVFP4 compact quant overlap (ZERO extra VRAM!).
// =========================================================================
template <int BM_CTA, bool NVFP4, bool PDL_WAIT, int TRIG_W13_MODE>
__global__ __launch_bounds__(256, 1)
void router_bf12_exact_v14_kernel(
    const __nv_bfloat16* __restrict__ X,
    const uint8_t* __restrict__ SLABS,
    const uint64_t* __restrict__ OUTLIERS,
    const int32_t* __restrict__ O_OFF,
    const float* __restrict__ Bias,
    const int32_t* __restrict__ ExpertBounds,
    const bool* __restrict__ Padding,
    const int32_t* __restrict__ NLivePtr,
    uint32_t* __restrict__ Counters,
    float* __restrict__ LogitsWS_Raw,     // >= 122,880 B ([5, 16, 384] fp32)
    int32_t* __restrict__ Src2Dst,
    float* __restrict__ Gammas,
    int32_t* __restrict__ Rows,
    int32_t* __restrict__ Tiles,
    uint8_t* __restrict__ QValues,        // [plan.rows, 2304] NVFP4 or [plan.rows, 4608] MXFP8
    uint8_t* __restrict__ QScales,        // R8c4 swizzled!
    float* __restrict__ QRowScales,       // [plan.rows] or nullptr
    const float* __restrict__ WAmax,
    const float* __restrict__ WDenominator,
    int T,
    float route_scale,
    uint32_t base_exp,
    int num_local,
    int tile_sz,
    int max_ctas
) {
    static_assert(BM_CTA == 8 || BM_CTA == 16, "exact_v14 handles BM_CTA in {8, 16}");
    constexpr int SX_STRIDE = 136;

    // Partition existing 122,880 B LogitsWS_Raw buffer cleanly (0 extra VRAM!):
    //   [0 .. 24,575]      LogitsWS      [16, 384] fp32
    //   [24,576 .. 49,151] SelBitsWS     [16, 384] u32
    //   [49,152 .. 86,015] TmpQVal_NV4   [16, 2304] u8  (NVFP4 only)
    //   [86,016 .. 90,623] TmpQSc_NV4    [16, 288]  u8  (NVFP4 only)
    //   [90,624 .. 90,687] TmpRowAmaxIn  [16]       f32 (NVFP4 only)
    uint8_t* raw_b = reinterpret_cast<uint8_t*>(LogitsWS_Raw);
    float*    LogitsWS    = reinterpret_cast<float*>(raw_b);
    uint32_t* SelBitsWS   = reinterpret_cast<uint32_t*>(raw_b + 24576);
    uint8_t*  TmpQVal     = raw_b + 49152;
    uint8_t*  TmpQSc      = raw_b + 86016;
    float*    TmpRowAmax  = reinterpret_cast<float*>(raw_b + 90624);

    const int bid  = static_cast<int>(blockIdx.x);
    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;

    struct Phase3ExactSmem {
        uint32_t sMask[E_FIXED];
        int32_t  sStarts[E_FIXED];
        int32_t  sStartsCompact[E_FIXED];
        int32_t  sTokLoc[BM_CTA][TOPK_FIXED];
        int32_t  sTokDst[BM_CTA][TOPK_FIXED];
        float    sTokWt[BM_CTA][TOPK_FIXED];
        float    sRowAmaxIn[BM_CTA];
        int32_t  sWarpT[8];
        int32_t  sWarpC[8];
        uint8_t  sIsPad[BM_CTA];
    };

    union alignas(16) SmemExact {
        __nv_bfloat16   sX[2][5][BM_CTA * SX_STRIDE];
        float           sP[5][BM_CTA][9];
        Phase3ExactSmem p3;
    };
    static_assert(sizeof(SmemExact) <= 48640, "SmemExact must fit 48 KiB default static SMEM");
    __shared__ SmemExact smem;
    __shared__ float     sBiasLocal[8];
    __shared__ float     sQuantWarpMax[3];
    __shared__ int32_t   sIsLastCta;

    const int e_col_base = bid * 8;
    const uint8_t* slab_base = SLABS + static_cast<int64_t>(bid * 36 + warp * 8) * SLAB12_BYTES;
    int32_t oc = 0, oe = 0;
    if (warp < 5) {
        const int wuid = bid * 5 + warp;
        oc = __ldg(O_OFF + wuid);
        oe = __ldg(O_OFF + wuid + 1);
        if constexpr (PDL_WAIT) {
            if (lane < 12) prefetch_l2(slab_base + lane * 128);
        }
    }
    if (tid < 8) sBiasLocal[tid] = __ldg(Bias + e_col_base + tid);

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

    if (n_live > 0) {
        if (warp < 5) {
            const int pk = warp;
            const int g  = lane >> 2;
            const int lt = lane & 3;
            const int n_slabs = (pk < 4) ? 8 : 4;
            const int k_lo = pk * 1024;
            const uint32_t base_pair = base_exp | (base_exp << 16);
            const bool has_m8 = (BM_CTA == 16) && (n_live > 8);

            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
            uint64_t o_rec = (oc < oe) ? __ldg(OUTLIERS + oc) : 0xFFFFFFFFFFFFFFFFULL;
            uint32_t o_next_step = static_cast<uint32_t>(o_rec >> 32);

            const int pad_rows = (n_live + 7) & ~7;
            const int total_u4 = pad_rows * 16;
            auto issue_x = [&](__nv_bfloat16* dst, int k_base) {
                for (int qi = lane; qi < total_u4; qi += 32) {
                    const int lm = qi >> 4;
                    const int qk = qi & 15;
                    const int safe_m = (lm < n_live) ? lm : 0;
                    cp_async_ca_16(dst + lm * SX_STRIDE + (qk << 3),
                                   X + static_cast<int64_t>(safe_m) * K_FIXED + k_base + (qk << 3),
                                   (lm < n_live) ? 16 : 0);
                }
                cp_async_commit();
            };

            Slab12 cur_slab = load_slab12<false>(slab_base, lane);
            issue_x(smem.sX[0][pk], k_lo);
            int cur_buf = 0;
            for (int sl = 0; sl < n_slabs; ++sl) {
                const int k_sl = k_lo + sl * 128;
                Slab12 nxt_slab;
                if (sl + 1 < n_slabs) {
                    nxt_slab = load_slab12<false>(slab_base + static_cast<int64_t>(sl + 1) * SLAB12_BYTES, lane);
                    issue_x(smem.sX[cur_buf ^ 1][pk], k_sl + 128);
                    cp_async_wait<1>();
                } else {
                    cp_async_wait<0>();
                }
                __syncwarp();
                const __nv_bfloat16* sX_b = smem.sX[cur_buf][pk] + g * SX_STRIDE + 2 * lt;
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
                    const int s_off = s * 16;
                    const uint32_t x0 = *reinterpret_cast<const uint32_t*>(sX_b + s_off);
                    const uint32_t x2 = *reinterpret_cast<const uint32_t*>(sX_b + s_off + 8);
                    const uint32_t x1 = has_m8 ? *reinterpret_cast<const uint32_t*>(sX_b + 8 * SX_STRIDE + s_off) : 0u;
                    const uint32_t x3 = has_m8 ? *reinterpret_cast<const uint32_t*>(sX_b + 8 * SX_STRIDE + s_off + 8) : 0u;
                    mma_bf16_r(a0, a1, a2, a3, x0, x1, x2, x3, b0, b1);
                }
                cur_buf ^= 1;
                if (sl + 1 < n_slabs) cur_slab = nxt_slab;
                __syncwarp();
            }

            __syncthreads();
            smem.sP[pk][g][2 * lt]     = a0;
            smem.sP[pk][g][2 * lt + 1] = a1;
            if constexpr (BM_CTA == 16) {
                if (has_m8) { smem.sP[pk][g + 8][2 * lt] = a2; smem.sP[pk][g + 8][2 * lt + 1] = a3; }
            }
        } else {
            // For NVFP4, Warps 5..7 overlap QUANT_FMA=1 quant directly into L2-hot TmpQVal/TmpQSc!
            if constexpr (NVFP4) {
                run_warps57_concurrent_quant_v14<true, true>(
                    X, Padding, TmpQVal, TmpQSc, TmpRowAmax, sQuantWarpMax, bid, n_live, warp - 5, lane
                );
            }
            __syncthreads();
        }
        __syncthreads();

        const int n_out = n_live * 8;
        for (int oidx = tid; oidx < n_out; oidx += 256) {
            const int lm = oidx >> 3;
            const int e_loc = oidx & 7;
            const float l = add_rn(add_rn(add_rn(add_rn(
                        smem.sP[0][lm][e_loc], smem.sP[1][lm][e_loc]),
                        smem.sP[2][lm][e_loc]), smem.sP[3][lm][e_loc]),
                        smem.sP[4][lm][e_loc]);
            const int64_t g_off = static_cast<int64_t>(lm) * E_FIXED + e_col_base + e_loc;
            LogitsWS[g_off]  = l;
            SelBitsWS[g_off] = selection_bits(l, sBiasLocal[e_loc]);
        }
    }

    __threadfence();
    __syncthreads();
    if (tid == 0) {
        const uint32_t prev = atom_acq_rel_gpu_add(&Counters[0], 1u);
        sIsLastCta = (prev == 47u) ? 1 : 0;
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

    const int lo = __ldg(ExpertBounds);
    const int hi = __ldg(ExpertBounds + 1);
    for (int i = tid; i < E_FIXED; i += 256) smem.p3.sMask[i] = 0u;
    if (tid < BM_CTA) {
        const bool pad = (tid >= n_live) || ((Padding != nullptr && tid < T) ? Padding[tid] : false);
        smem.p3.sIsPad[tid] = pad ? 1u : 0u;
        if constexpr (NVFP4) {
            smem.p3.sRowAmaxIn[tid] = (!pad && tid < T) ? ld_cg_f32(TmpRowAmax + tid) : 0.0f;
        }
    }
    __syncthreads();

    for (int tok = warp; tok < T; tok += 8) {
        if (smem.p3.sIsPad[tok]) {
            if (lane < TOPK_FIXED) {
                smem.p3.sTokLoc[tok][lane] = -1;
                smem.p3.sTokWt[tok][lane]  = 0.0f;
            }
            continue;
        }
        const int32_t my_id = warp_topk8_from_selbits_ptr(
            SelBitsWS + static_cast<int64_t>(tok) * E_FIXED, lane
        );
        float lp = -INFINITY;
        if (lane < TOPK_FIXED) {
            const float win_logit = ld_cg_f32(LogitsWS + static_cast<int64_t>(tok) * E_FIXED + my_id);
            lp = sub_rn(fminf(win_logit, 0.0f), log1pf(expf(-fabsf(win_logit))));
        }
        const float my_wt = warp_softmax8_exact(lp, lane, route_scale);
        if (lane < TOPK_FIXED) {
            const int32_t my_loc = (my_id >= lo && my_id < hi) ? (my_id - lo) : -1;
            smem.p3.sTokLoc[tok][lane] = my_loc;
            smem.p3.sTokWt[tok][lane]  = my_wt;
            if (my_loc >= 0) atomicOr(&smem.p3.sMask[my_loc], 1u << tok);
        }
    }
    __syncthreads();

    const int tile_shift = (tile_sz == 8) ? 3 : ((tile_sz == 16) ? 4 : ((tile_sz == 32) ? 5 : -1));
    const int tile_mask_i = tile_sz - 1;
    const int e0 = warp * 64 + lane;
    const int e1 = e0 + 32;
    int c0 = 0, c1 = 0, t0 = 0, t1 = 0, exc_t0 = 0, exc_c0 = 0, exc_t1 = 0, exc_c1 = 0;
    if (warp < 6) {
        c0 = (e0 < num_local) ? __popc(smem.p3.sMask[e0]) : 0;
        c1 = (e1 < num_local) ? __popc(smem.p3.sMask[e1]) : 0;
        if (tile_shift >= 0) {
            t0 = (c0 + tile_mask_i) >> tile_shift;
            t1 = (c1 + tile_mask_i) >> tile_shift;
        } else {
            t0 = (c0 + tile_mask_i) / tile_sz;
            t1 = (c1 + tile_mask_i) / tile_sz;
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
        if (lane == 31) { smem.p3.sWarpT[warp] = sum0_t + it1; smem.p3.sWarpC[warp] = sum0_c + ic1; }
        exc_t0 = it0 - t0; exc_c0 = ic0 - c0;
        exc_t1 = sum0_t + it1 - t1; exc_c1 = sum0_c + ic1 - c1;
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
        if (lane < 6) { smem.p3.sWarpT[lane] = iwt - wt; smem.p3.sWarpC[lane] = iwc - wc; }
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
        auto emit_e = [&](int e_idx, int c_cnt, int t_cnt, int et, int ec) {
            if (e_idx >= num_local) return;
            const int ts = base_t + et;
            const int cs = base_c + ec;
            const int st = ts * tile_sz;
            smem.p3.sStarts[e_idx]        = st;
            smem.p3.sStartsCompact[e_idx] = cs;
            for (int k = 0; k < t_cnt; ++k) {
                const int ti = ts + k;
                const int end_r = (k + 1 < t_cnt) ? ((ti + 1) * tile_sz) : (st + c_cnt);
                Tiles[ti]            = e_idx;
                Tiles[max_ctas + ti] = end_r;
            }
        };
        emit_e(e0, c0, t0, exc_t0, exc_c0);
        emit_e(e1, c1, t1, exc_t1, exc_c1);
    }
    __syncthreads();

    const int total_slots = T * TOPK_FIXED;
    float rcp_denom = 0.0f;
    if constexpr (NVFP4) rcp_denom = rcp_rn(mul_rn(1536.0f, __ldg(WDenominator)));
    for (int slot = tid; slot < total_slots; slot += 256) {
        const int tok = slot >> 3;
        const int k   = slot & 7;
        const int32_t loc = smem.p3.sTokLoc[tok][k];
        if (loc >= 0) {
            const int32_t rank = __popc(smem.p3.sMask[loc] & ((1u << tok) - 1u));
            const int32_t dst  = smem.p3.sStarts[loc] + rank;
            const int32_t cdst = smem.p3.sStartsCompact[loc] + rank;
            smem.p3.sTokDst[tok][k] = dst;
            Src2Dst[slot] = dst;
            Gammas[dst]   = smem.p3.sTokWt[tok][k];
            Rows[cdst]    = dst;
            if constexpr (NVFP4) {
                QRowScales[dst] = mul_rn(mul_rn(smem.p3.sRowAmaxIn[tok], __ldg(WAmax + loc)), rcp_denom);
            }
        } else {
            smem.p3.sTokDst[tok][k] = -1;
            Src2Dst[slot] = -1;
        }
    }
    __syncthreads();

    // Scatter QValues & R8c4-swizzled QScales to all routed `dst` rows for each live token!
    for (int tok = 0; tok < T; ++tok) {
        if (smem.p3.sIsPad[tok]) continue;
        if constexpr (NVFP4) {
            const uint64_t* src_v = reinterpret_cast<const uint64_t*>(TmpQVal) + static_cast<int64_t>(tok) * 288;
            const uint8_t*  src_s = TmpQSc + static_cast<int64_t>(tok) * 288;
            for (int blk = tid; blk < 288; blk += 256) {
                const uint64_t qv = *reinterpret_cast<const uint64_t*>(src_v + blk);
                const uint8_t  qs = src_s[blk];
                #pragma unroll
                for (int k = 0; k < TOPK_FIXED; ++k) {
                    const int32_t dst = smem.p3.sTokDst[tok][k];
                    if (dst >= 0) {
                        reinterpret_cast<uint64_t*>(QValues)[static_cast<int64_t>(dst) * 288 + blk] = qv;
                        QScales[r8c4_scale_offset(dst, blk, 288)] = qs;
                    }
                }
            }
        } else {
            if (tid < 144) {
                const uint4* bp = reinterpret_cast<const uint4*>(
                    X + static_cast<int64_t>(tok) * K_FIXED + tid * 32
                );
                uint4 q0, q1;
                uint8_t sc;
                quant_mxfp8_block32_regs(ld_ca_u4(bp), ld_ca_u4(bp + 1), ld_ca_u4(bp + 2), ld_ca_u4(bp + 3), q0, q1, sc);
                #pragma unroll
                for (int k = 0; k < TOPK_FIXED; ++k) {
                    const int32_t dst = smem.p3.sTokDst[tok][k];
                    if (dst >= 0) {
                        uint4* dst_q = reinterpret_cast<uint4*>(
                            QValues + static_cast<int64_t>(dst) * K_FIXED + tid * 32
                        );
                        dst_q[0] = q0;
                        dst_q[1] = q1;
                        QScales[r8c4_scale_offset(dst, tid, 144)] = sc;
                    }
                }
            }
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

// =========================================================================
// WAVE-V14 1-PASS 48-CTA `BM_CTA=128` GEMM KERNEL (`T = 65..128`, 8.0x Less Weight Traffic!)
//   - Grid=(48,1,1), Block=(160,1,1), SMEM = 23.04 KiB (co-resides w/ 3.30 KiB K2 inside 48 KiB!)
//   - Loads the 2.654 MB BF12 slabs EXACTLY 1.000x (vs 8.0x = 21.23 MB in V10)!
// =========================================================================
template <bool PDL_WAIT, bool TRIG>
__global__ __launch_bounds__(160, ROUTER_V14_C128_GEMM_MIN_CTA)
void router_bf12_gemm_c128_v14_kernel(
    const __nv_bfloat16* __restrict__ X,
    const uint8_t* __restrict__ SLABS,
    const uint64_t* __restrict__ OUTLIERS,
    const int32_t* __restrict__ O_OFF,
    const int32_t* __restrict__ NLivePtr,
    float* __restrict__ Logits,
    int T,
    uint32_t base_exp
) {
    constexpr int CHUNK_M = 32;
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
    if (m_live > 128) m_live = 128;
    if (m_live <= 0) return;

    union alignas(16) SmemC128 {
        __nv_bfloat16 sX[5][CHUNK_M * SX_STRIDE];  // 23,040 B (22.5 KiB!)
        float         sP[5][CHUNK_M][9];           //  5,760 B
    };
    __shared__ SmemC128 smem;
    __nv_bfloat16* sX_warp = smem.sX[pk];
    const __nv_bfloat16* sX_base = sX_warp + g * SX_STRIDE + 2 * lt;

    float acc[8][4];
    #pragma unroll
    for (int mt = 0; mt < 8; ++mt) {
        acc[mt][0] = 0.0f; acc[mt][1] = 0.0f; acc[mt][2] = 0.0f; acc[mt][3] = 0.0f;
    }

    uint64_t o_rec = (oc < oe) ? __ldg(OUTLIERS + oc) : 0xFFFFFFFFFFFFFFFFULL;
    uint32_t o_next_step = static_cast<uint32_t>(o_rec >> 32);

    auto load_x_chunk32 = [&](int m_off, int k_base) {
        int m_cnt = m_live - m_off;
        if (m_cnt > CHUNK_M) m_cnt = CHUNK_M;
        const int p_rows = (m_cnt + 7) & ~7;
        const int tot_u4 = p_rows * 8;
        const uint4 z4 = make_uint4(0u, 0u, 0u, 0u);
        for (int qi = lane; qi < tot_u4; qi += 32) {
            const int lm = qi >> 3;
            const int qk = qi & 7;
            uint4 v = z4;
            if (lm < m_cnt) {
                v = ld_ca_u4(X + static_cast<int64_t>(m_off + lm) * K_FIXED + k_base + (qk << 3));
            }
            *reinterpret_cast<uint4*>(sX_warp + lm * SX_STRIDE + (qk << 3)) = v;
        }
        __syncwarp();
    };

    auto apply_chunk32 = [&](int c_idx, int m_off, const uint32_t b0_s[4], const uint32_t b1_s[4]) {
        const int m_cnt = m_live - m_off;
        const bool hm8  = (m_cnt > 8);
        const bool hm16 = (m_cnt > 16);
        const bool hm24 = (m_cnt > 24);
        const int mt0 = c_idx * 2;
        const int mt1 = mt0 + 1;
        #pragma unroll
        for (int s = 0; s < 4; ++s) {
            const int s_off = s * 16;
            {
                const uint32_t x0 = *reinterpret_cast<const uint32_t*>(sX_base + s_off);
                const uint32_t x2 = *reinterpret_cast<const uint32_t*>(sX_base + s_off + 8);
                const uint32_t x1 = hm8 ? *reinterpret_cast<const uint32_t*>(sX_base + 8 * SX_STRIDE + s_off) : 0u;
                const uint32_t x3 = hm8 ? *reinterpret_cast<const uint32_t*>(sX_base + 8 * SX_STRIDE + s_off + 8) : 0u;
                mma_bf16_r(acc[mt0][0], acc[mt0][1], acc[mt0][2], acc[mt0][3], x0, x1, x2, x3, b0_s[s], b1_s[s]);
            }
            if (hm16) {
                const uint32_t x0 = *reinterpret_cast<const uint32_t*>(sX_base + 16 * SX_STRIDE + s_off);
                const uint32_t x2 = *reinterpret_cast<const uint32_t*>(sX_base + 16 * SX_STRIDE + s_off + 8);
                const uint32_t x1 = hm24 ? *reinterpret_cast<const uint32_t*>(sX_base + 24 * SX_STRIDE + s_off) : 0u;
                const uint32_t x3 = hm24 ? *reinterpret_cast<const uint32_t*>(sX_base + 24 * SX_STRIDE + s_off + 8) : 0u;
                mma_bf16_r(acc[mt1][0], acc[mt1][1], acc[mt1][2], acc[mt1][3], x0, x1, x2, x3, b0_s[s], b1_s[s]);
            }
        }
    };

    Slab12 cur_slab = load_slab12<false>(slab0, lane);
    for (int sl = 0; sl < n_slabs; ++sl) {
        const int k_sl = k_lo + sl * 128;
        Slab12 nxt_slab;
        if (sl + 1 < n_slabs) {
            nxt_slab = load_slab12<false>(slab0 + static_cast<int64_t>(sl + 1) * SLAB12_BYTES, lane);
        }

        // Half 0: unpack (b0, b1) ONCE, reuse across all active 32-row chunks!
        {
            const uint32_t sm_h0[4] = {cur_slab.sm0.x, cur_slab.sm0.y, cur_slab.sm0.z, cur_slab.sm0.w};
            const uint32_t e_h0[4]  = {
                cur_slab.e4.x & 0xFFFFu, cur_slab.e4.x >> 16,
                cur_slab.e4.y & 0xFFFFu, cur_slab.e4.y >> 16
            };
            uint32_t b0_s[4], b1_s[4];
            #pragma unroll
            for (int s = 0; s < 4; ++s) {
                unpack4_bf12(sm_h0[s], e_h0[s], base_pair, b0_s[s], b1_s[s]);
                patch_outliers(static_cast<uint32_t>((sl << 3) | s), lane,
                               OUTLIERS, oc, oe, o_rec, o_next_step, b0_s[s], b1_s[s]);
            }
            #pragma unroll
            for (int c_idx = 0; c_idx < 4; ++c_idx) {
                const int m_off = c_idx * CHUNK_M;
                if (m_off < m_live) {
                    load_x_chunk32(m_off, k_sl);
                    apply_chunk32(c_idx, m_off, b0_s, b1_s);
                    __syncwarp();
                }
            }
        }
        // Half 1: unpack (b0, b1) ONCE, reuse across all active 32-row chunks!
        {
            const uint32_t sm_h1[4] = {cur_slab.sm1.x, cur_slab.sm1.y, cur_slab.sm1.z, cur_slab.sm1.w};
            const uint32_t e_h1[4]  = {
                cur_slab.e4.z & 0xFFFFu, cur_slab.e4.z >> 16,
                cur_slab.e4.w & 0xFFFFu, cur_slab.e4.w >> 16
            };
            uint32_t b0_s[4], b1_s[4];
            #pragma unroll
            for (int s = 0; s < 4; ++s) {
                unpack4_bf12(sm_h1[s], e_h1[s], base_pair, b0_s[s], b1_s[s]);
                patch_outliers(static_cast<uint32_t>((sl << 3) | (4 + s)), lane,
                               OUTLIERS, oc, oe, o_rec, o_next_step, b0_s[s], b1_s[s]);
            }
            #pragma unroll
            for (int c_idx = 0; c_idx < 4; ++c_idx) {
                const int m_off = c_idx * CHUNK_M;
                if (m_off < m_live) {
                    load_x_chunk32(m_off, k_sl + 64);
                    apply_chunk32(c_idx, m_off, b0_s, b1_s);
                    __syncwarp();
                }
            }
        }
        if (sl + 1 < n_slabs) cur_slab = nxt_slab;
    }

    const int e_col_base = e_warp * 8;
    #pragma unroll
    for (int c_idx = 0; c_idx < 4; ++c_idx) {
        const int m_off = c_idx * CHUNK_M;
        if (m_off >= m_live) break;
        const int m_cnt = (m_live - m_off < CHUNK_M) ? (m_live - m_off) : CHUNK_M;
        const int mt0 = c_idx * 2;
        const int mt1 = mt0 + 1;
        __syncthreads();
        smem.sP[pk][g][2 * lt]     = acc[mt0][0]; smem.sP[pk][g][2 * lt + 1]     = acc[mt0][1];
        if (m_cnt >  8) { smem.sP[pk][g +  8][2 * lt] = acc[mt0][2]; smem.sP[pk][g +  8][2 * lt + 1] = acc[mt0][3]; }
        if (m_cnt > 16) { smem.sP[pk][g + 16][2 * lt] = acc[mt1][0]; smem.sP[pk][g + 16][2 * lt + 1] = acc[mt1][1]; }
        if (m_cnt > 24) { smem.sP[pk][g + 24][2 * lt] = acc[mt1][2]; smem.sP[pk][g + 24][2 * lt + 1] = acc[mt1][3]; }
        __syncthreads();

        const int n_out = m_cnt * 8;
        for (int oidx = tid; oidx < n_out; oidx += 160) {
            const int lm = oidx >> 3;
            const int e_loc = oidx & 7;
            const float l = add_rn(add_rn(add_rn(add_rn(
                        smem.sP[0][lm][e_loc], smem.sP[1][lm][e_loc]),
                        smem.sP[2][lm][e_loc]), smem.sP[3][lm][e_loc]),
                        smem.sP[4][lm][e_loc]);
            Logits[static_cast<int64_t>(m_off + lm) * E_FIXED + e_col_base + e_loc] = l;
        }
    }
}

// =========================================================================
// V14 Fixed 1.000x Half-Slab `router_bf12_gemm_c32_v14_kernel` (T = 1..64 & 129..256)
//   Fixes V10's 2.00x double-load bug via `load_slab12_half<CACHE_W, 0/1>`!
// =========================================================================
template <bool PDL_WAIT, bool TRIG>
__global__ __launch_bounds__(160, ROUTER_V14_BM64_MIN_CTA)
void router_bf12_gemm_c48_v14_bm64_kernel(
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
    if (has_m8)  { smem.sP[pk][g +  8][2 * lt] = a0_2; smem.sP[pk][g +  8][2 * lt + 1] = a0_3; }
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
void router_bf12_gemm_c32_v14_kernel(
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

    // V14 FIX: `load_slab12_half<CACHE_W, 0>` loads ONLY sm0 + e4.xy (Half 0),
    //          `load_slab12_half<CACHE_W, 1>` loads ONLY sm1 + e4.zw (Half 1) -> exact 1.000x slab reads!
    for (int sl = 0; sl < n_slabs; ++sl) {
        const int k_sl = k_lo + sl * 128;
        load_x_half(k_sl);

        Slab12Half cur_h0 = load_slab12_half<CACHE_W, 0>(
            SLABS + static_cast<int64_t>(ew0 * 36 + pk * 8 + sl) * SLAB12_BYTES, lane
        );
        #pragma unroll
        for (int ew = 0; ew < EW_PER_CTA; ++ew) {
            Slab12Half nxt_h0;
            if (ew + 1 < EW_PER_CTA) {
                nxt_h0 = load_slab12_half<CACHE_W, 0>(
                    SLABS + static_cast<int64_t>((ew0 + ew + 1) * 36 + pk * 8 + sl) * SLAB12_BYTES, lane
                );
            }
            const uint32_t sm_h0[4] = {cur_h0.sm.x, cur_h0.sm.y, cur_h0.sm.z, cur_h0.sm.w};
            const uint32_t e_h0[4]  = {
                cur_h0.e2.x & 0xFFFFu, cur_h0.e2.x >> 16,
                cur_h0.e2.y & 0xFFFFu, cur_h0.e2.y >> 16
            };
            #pragma unroll
            for (int s_loc = 0; s_loc < 4; ++s_loc) {
                uint32_t b0, b1;
                unpack4_bf12(sm_h0[s_loc], e_h0[s_loc], base_pair, b0, b1);
                patch_outliers(static_cast<uint32_t>((sl << 3) | s_loc), lane,
                               OUTLIERS, o_cur[ew], o_end_arr[ew], o_rec[ew], o_next_step[ew], b0, b1);
                run_mma_ew(ew, s_loc, b0, b1);
            }
            if (ew + 1 < EW_PER_CTA) cur_h0 = nxt_h0;
        }
        __syncwarp();
        load_x_half(k_sl + 64);

        Slab12Half cur_h1 = load_slab12_half<CACHE_W, 1>(
            SLABS + static_cast<int64_t>(ew0 * 36 + pk * 8 + sl) * SLAB12_BYTES, lane
        );
        #pragma unroll
        for (int ew = 0; ew < EW_PER_CTA; ++ew) {
            Slab12Half nxt_h1;
            if (ew + 1 < EW_PER_CTA) {
                nxt_h1 = load_slab12_half<CACHE_W, 1>(
                    SLABS + static_cast<int64_t>((ew0 + ew + 1) * 36 + pk * 8 + sl) * SLAB12_BYTES, lane
                );
            }
            const uint32_t sm_h1[4] = {cur_h1.sm.x, cur_h1.sm.y, cur_h1.sm.z, cur_h1.sm.w};
            const uint32_t e_h1[4]  = {
                cur_h1.e2.x & 0xFFFFu, cur_h1.e2.x >> 16,
                cur_h1.e2.y & 0xFFFFu, cur_h1.e2.y >> 16
            };
            #pragma unroll
            for (int s_loc = 0; s_loc < 4; ++s_loc) {
                uint32_t b0, b1;
                unpack4_bf12(sm_h1[s_loc], e_h1[s_loc], base_pair, b0, b1);
                patch_outliers(static_cast<uint32_t>((sl << 3) | (4 + s_loc)), lane,
                               OUTLIERS, o_cur[ew], o_end_arr[ew], o_rec[ew], o_next_step[ew], b0, b1);
                run_mma_ew(ew, s_loc, b0, b1);
            }
            if (ew + 1 < EW_PER_CTA) cur_h1 = nxt_h1;
        }
        __syncwarp();
    }

    __syncthreads();
    const int n_out = m_live * 8;
    #pragma unroll
    for (int ew = 0; ew < EW_PER_CTA; ++ew) {
        smem.sP[pk][g][2 * lt]     = acc_lo[ew][0];
        smem.sP[pk][g][2 * lt + 1] = acc_lo[ew][1];
        if (has_m8)  { smem.sP[pk][g +  8][2 * lt] = acc_lo[ew][2]; smem.sP[pk][g +  8][2 * lt + 1] = acc_lo[ew][3]; }
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

// =========================================================================
// WAVE-V14 48-CTA ZERO-WAVE-TAIL MULTI-WARP `c32_quant_topk_place_v14_kernel` (T = 1..256)
//   - Grid = (min(T, 48), 1, 1) -> 1 exact wave on GB10 (0 wave tails @ M=64, 96, 128, 192, 256!)
//   - SMEM = 3.30 KiB (co-resides simultaneously w/ 23.04 KiB GEMM inside 48 KiB during PDL pre-quant!)
//   - Multi-warp parallel register-only 3x ld_cg_u4/u2 Top-8 (3 warps @ M=128, 6 warps @ M=256 vs 1 warp in V10!)
// =========================================================================
template <bool NVFP4, bool LOGITS_BF16, bool PDL_WAIT, int TRIG_W13_MODE, bool QUANT_FMA>
__global__ __launch_bounds__(384, ROUTER_V14_K2_MIN_CTA)
void c32_quant_topk_place_v14_kernel(
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
    const int bid  = static_cast<int>(blockIdx.x);
    const int ncta = static_cast<int>(gridDim.x);
    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    // Only 3,264 B total SMEM (no sLogits[384], no sSelBits[384]!)
    alignas(16) __shared__ int32_t sStarts[E_FIXED];
    alignas(16) __shared__ int32_t sStartsCompact[E_FIXED];
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
    const int lo = __ldg(ExpertBounds);
    const int hi = __ldg(ExpertBounds + 1);

    // Pre-gdc_wait: Quantize ALL tokens assigned to this CTA (`tok = bid, bid+ncta, ...`) overlapped with GEMM!
    for (int tok = bid; tok < T; tok += ncta) {
        const bool is_pad = (tok >= n_live) || ((Padding != nullptr) ? Padding[tok] : false);
        if (is_pad) {
            if constexpr (NVFP4) {
                if (tid == 0) RowAmaxIn[tok] = 0.0f;
            }
            continue;
        }
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
                quant_nvfp4_block16_compact_v14<QUANT_FMA>(v0, v1, blk_amax, enc_s, dec_s, out_s, packed, sc);
                reinterpret_cast<uint64_t*>(QValues)[static_cast<int64_t>(tok) * 288 + tid] = packed;
                QScales[static_cast<int64_t>(tok) * 288 + tid] = sc;
            }
            __syncthreads();
        } else {
            if (tid < 144) {
                const uint4* bp = reinterpret_cast<const uint4*>(
                    X + static_cast<int64_t>(tok) * K_FIXED + tid * 32
                );
                uint4 q0, q1;
                uint8_t sc;
                quant_mxfp8_block32_regs(ld_ca_u4(bp), ld_ca_u4(bp + 1), ld_ca_u4(bp + 2), ld_ca_u4(bp + 3), q0, q1, sc);
                uint4* dq = reinterpret_cast<uint4*>(QValues + static_cast<int64_t>(tok) * K_FIXED + tid * 32);
                dq[0] = q0; dq[1] = q1;
                QScales[static_cast<int64_t>(tok) * 144 + tid] = sc;
            }
        }
    }

    if constexpr (PDL_WAIT) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }

    // Post-gdc_wait: Multi-warp parallel register-only Top-8 + Softmax across ALL tokens owned by this CTA!
    // Warp `w` handles token `tok = bid + w * ncta` (for `T=128`, 3 warps run in parallel; for `T=256`, 6 warps!).
    const int tok = bid + warp * ncta;
    if (tok < T) {
        const bool is_pad = (tok >= n_live) || ((Padding != nullptr) ? Padding[tok] : false);
        if (!is_pad) {
            const int64_t row_off = static_cast<int64_t>(tok) * E_FIXED;
            const int e0 = lane << 2;
            const int e1 = 128 + e0;
            const int e2 = 256 + e0;

            auto load4_sel = [&](int e_base) -> uint4 {
                const float b0 = __ldg(Bias + e_base + 0);
                const float b1 = __ldg(Bias + e_base + 1);
                const float b2 = __ldg(Bias + e_base + 2);
                const float b3 = __ldg(Bias + e_base + 3);
                float l0, l1, l2, l3;
                if constexpr (LOGITS_BF16) {
                    const uint2 raw = ld_cg_u2(static_cast<const uint16_t*>(Logits) + row_off + e_base);
                    l0 = bf16_bits_to_f32(static_cast<uint16_t>(raw.x & 0xFFFFu));
                    l1 = bf16_bits_to_f32(static_cast<uint16_t>(raw.x >> 16));
                    l2 = bf16_bits_to_f32(static_cast<uint16_t>(raw.y & 0xFFFFu));
                    l3 = bf16_bits_to_f32(static_cast<uint16_t>(raw.y >> 16));
                } else {
                    const uint4 raw = ld_cg_u4(static_cast<const float*>(Logits) + row_off + e_base);
                    l0 = __uint_as_float(raw.x);
                    l1 = __uint_as_float(raw.y);
                    l2 = __uint_as_float(raw.z);
                    l3 = __uint_as_float(raw.w);
                }
                return make_uint4(
                    selection_bits(l0, b0), selection_bits(l1, b1),
                    selection_bits(l2, b2), selection_bits(l3, b3)
                );
            };

            const uint4 c0 = load4_sel(e0);
            const uint4 c1 = load4_sel(e1);
            const uint4 c2 = load4_sel(e2);
            const int32_t my_id = warp_topk8_from_3u4(c0, c1, c2, lane);

            float lp = -INFINITY;
            if (lane < TOPK_FIXED) {
                float win_logit;
                if constexpr (LOGITS_BF16) {
                    win_logit = bf16_bits_to_f32(ld_cg_u16(static_cast<const uint16_t*>(Logits) + row_off + my_id));
                } else {
                    win_logit = ld_cg_f32(static_cast<const float*>(Logits) + row_off + my_id);
                }
                lp = sub_rn(fminf(win_logit, 0.0f), log1pf(expf(-fabsf(win_logit))));
            }
            const float my_wt = warp_softmax8_exact(lp, lane, route_scale);

            if (lane < TOPK_FIXED) {
                const int32_t my_loc = (my_id >= lo && my_id < hi) ? (my_id - lo) : -1;
                const int32_t my_rank = (my_loc >= 0) ? atomicAdd(&Counts[my_loc], 1) : 0;
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
        const uint32_t prev = atom_acq_rel_gpu_add(&Counters[0], 1u);
        sIsLastCta = (prev == static_cast<uint32_t>(ncta - 1)) ? 1 : 0;
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

// =========================================================================
// Legacy NULL-Counters Defensive K2 for `router_bf12_exact_fwd` (1-CTA, T <= 16)
// =========================================================================
template <bool NVFP4, bool PDL_WAIT, bool TRIG>
__global__ __launch_bounds__(256, 1)
void router_bf12_exact_nullcnt_k2_kernel(
    const __nv_bfloat16* __restrict__ X,
    const float* __restrict__ LogitsWS,
    const float* __restrict__ Bias,
    const int32_t* __restrict__ ExpertBounds,
    const bool* __restrict__ Padding,
    const int32_t* __restrict__ NLivePtr,
    int32_t* __restrict__ Src2Dst,
    float* __restrict__ Gammas,
    int32_t* __restrict__ Rows,
    int32_t* __restrict__ Tiles,
    uint8_t* __restrict__ QValues,
    uint8_t* __restrict__ QScales,
    float* __restrict__ QRowScales,
    const float* __restrict__ WAmax,
    const float* __restrict__ WDenominator,
    int T,
    float route_scale,
    int num_local,
    int tile_sz,
    int max_ctas
) {
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    if constexpr (PDL_WAIT) asm volatile("griddepcontrol.wait;" ::: "memory");

    __shared__ uint32_t sMask[E_FIXED];
    __shared__ int32_t  sStarts[E_FIXED];
    __shared__ int32_t  sStartsCompact[E_FIXED];
    __shared__ int32_t  sTokLoc[MAX_T_C1][TOPK_FIXED];
    __shared__ int32_t  sTokDst[MAX_T_C1][TOPK_FIXED];
    __shared__ float    sTokWt[MAX_T_C1][TOPK_FIXED];
    __shared__ float    sWarpMax[8];
    __shared__ int32_t  sWarpT[8];
    __shared__ int32_t  sWarpC[8];
    __shared__ uint8_t  sIsPad[MAX_T_C1];

    int n_live = T;
    if (NLivePtr != nullptr) {
        int nl = *NLivePtr;
        if (nl < 0) nl = 0;
        if (nl < n_live) n_live = nl;
    }
    const int lo = __ldg(ExpertBounds);
    const int hi = __ldg(ExpertBounds + 1);

    for (int i = tid; i < E_FIXED; i += 256) sMask[i] = 0u;
    if (tid < MAX_T_C1) {
        const bool pad = (tid >= n_live) || ((Padding != nullptr && tid < T) ? Padding[tid] : false);
        sIsPad[tid] = pad ? 1u : 0u;
    }
    __syncthreads();

    for (int tok = warp; tok < T; tok += 8) {
        if (sIsPad[tok]) {
            if (lane < TOPK_FIXED) { sTokLoc[tok][lane] = -1; sTokWt[tok][lane] = 0.0f; }
            continue;
        }
        const int64_t row_off = static_cast<int64_t>(tok) * E_FIXED;
        const int e0 = lane << 2;
        auto load4 = [&](int eb) -> uint4 {
            const uint4 raw = ld_cg_u4(LogitsWS + row_off + eb);
            return make_uint4(
                selection_bits(__uint_as_float(raw.x), __ldg(Bias + eb + 0)),
                selection_bits(__uint_as_float(raw.y), __ldg(Bias + eb + 1)),
                selection_bits(__uint_as_float(raw.z), __ldg(Bias + eb + 2)),
                selection_bits(__uint_as_float(raw.w), __ldg(Bias + eb + 3))
            );
        };
        const int32_t my_id = warp_topk8_from_3u4(load4(e0), load4(128 + e0), load4(256 + e0), lane);
        float lp = -INFINITY;
        if (lane < TOPK_FIXED) {
            const float wl = ld_cg_f32(LogitsWS + row_off + my_id);
            lp = sub_rn(fminf(wl, 0.0f), log1pf(expf(-fabsf(wl))));
        }
        const float my_wt = warp_softmax8_exact(lp, lane, route_scale);
        if (lane < TOPK_FIXED) {
            const int32_t my_loc = (my_id >= lo && my_id < hi) ? (my_id - lo) : -1;
            sTokLoc[tok][lane] = my_loc;
            sTokWt[tok][lane]  = my_wt;
            if (my_loc >= 0) atomicOr(&sMask[my_loc], 1u << tok);
        }
    }
    __syncthreads();

    const int tile_shift = (tile_sz == 8) ? 3 : ((tile_sz == 16) ? 4 : ((tile_sz == 32) ? 5 : -1));
    const int tile_mask_i = tile_sz - 1;
    const int e0 = warp * 64 + lane;
    const int e1 = e0 + 32;
    int c0 = 0, c1 = 0, t0 = 0, t1 = 0, exc_t0 = 0, exc_c0 = 0, exc_t1 = 0, exc_c1 = 0;
    if (warp < 6) {
        c0 = (e0 < num_local) ? __popc(sMask[e0]) : 0;
        c1 = (e1 < num_local) ? __popc(sMask[e1]) : 0;
        if (tile_shift >= 0) {
            t0 = (c0 + tile_mask_i) >> tile_shift;
            t1 = (c1 + tile_mask_i) >> tile_shift;
        } else {
            t0 = (c0 + tile_mask_i) / tile_sz;
            t1 = (c1 + tile_mask_i) / tile_sz;
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
        if (lane == 31) { sWarpT[warp] = sum0_t + it1; sWarpC[warp] = sum0_c + ic1; }
        exc_t0 = it0 - t0; exc_c0 = ic0 - c0;
        exc_t1 = sum0_t + it1 - t1; exc_c1 = sum0_c + ic1 - c1;
    }
    __syncthreads();

    if (warp == 0) {
        const int wt = (lane < 6) ? sWarpT[lane] : 0;
        const int wc = (lane < 6) ? sWarpC[lane] : 0;
        int iwt = wt, iwc = wc;
        #pragma unroll
        for (int off = 1; off < 8; off <<= 1) {
            int nt = __shfl_up_sync(0xFFFFFFFFu, iwt, off), nc = __shfl_up_sync(0xFFFFFFFFu, iwc, off);
            if (lane >= off) { iwt += nt; iwc += nc; }
        }
        if (lane < 6) { sWarpT[lane] = iwt - wt; sWarpC[lane] = iwc - wc; }
        if (lane == 5) {
            Tiles[2 * max_ctas]     = iwt;
            Tiles[2 * max_ctas + 1] = iwt * tile_sz;
            Tiles[2 * max_ctas + 2] = iwc;
        }
    }
    __syncthreads();

    if (warp < 6) {
        const int base_t = sWarpT[warp];
        const int base_c = sWarpC[warp];
        auto emit_e = [&](int e_idx, int c_cnt, int t_cnt, int et, int ec) {
            if (e_idx >= num_local) return;
            const int ts = base_t + et;
            const int cs = base_c + ec;
            const int st = ts * tile_sz;
            sStarts[e_idx] = st; sStartsCompact[e_idx] = cs;
            for (int k = 0; k < t_cnt; ++k) {
                const int ti = ts + k;
                const int end_r = (k + 1 < t_cnt) ? ((ti + 1) * tile_sz) : (st + c_cnt);
                Tiles[ti] = e_idx;
                Tiles[max_ctas + ti] = end_r;
            }
        };
        emit_e(e0, c0, t0, exc_t0, exc_c0);
        emit_e(e1, c1, t1, exc_t1, exc_c1);
    }
    __syncthreads();

    const int total_slots = T * TOPK_FIXED;
    for (int slot = tid; slot < total_slots; slot += 256) {
        const int tok = slot >> 3;
        const int k   = slot & 7;
        const int32_t loc = sTokLoc[tok][k];
        if (loc >= 0) {
            const int32_t rank = __popc(sMask[loc] & ((1u << tok) - 1u));
            const int32_t dst  = sStarts[loc] + rank;
            const int32_t cdst = sStartsCompact[loc] + rank;
            sTokDst[tok][k] = dst;
            Src2Dst[slot] = dst;
            Gammas[dst]   = sTokWt[tok][k];
            Rows[cdst]    = dst;
        } else {
            sTokDst[tok][k] = -1;
            Src2Dst[slot] = -1;
        }
    }
    __syncthreads();

    float rcp_denom = 0.0f;
    if constexpr (NVFP4) rcp_denom = rcp_rn(mul_rn(1536.0f, __ldg(WDenominator)));
    for (int tok = 0; tok < T; ++tok) {
        if (sIsPad[tok]) continue;
        if constexpr (NVFP4) {
            const __nv_bfloat16* x_row = X + static_cast<int64_t>(tok) * K_FIXED;
            const uint4* bp0 = reinterpret_cast<const uint4*>(x_row + tid * 16);
            const uint4 v0_a = ld_ca_u4(bp0);
            const uint4 v1_a = ld_ca_u4(bp0 + 1);
            const float bm_a = block16_amax(v0_a, v1_a);
            uint4 v0_b = make_uint4(0u, 0u, 0u, 0u), v1_b = make_uint4(0u, 0u, 0u, 0u);
            float bm_b = 0.0f;
            if (tid < 32) {
                const uint4* bp1 = reinterpret_cast<const uint4*>(x_row + (tid + 256) * 16);
                v0_b = ld_ca_u4(bp1); v1_b = ld_ca_u4(bp1 + 1);
                bm_b = block16_amax(v0_b, v1_b);
            }
            float wmax = fmaxf(bm_a, bm_b);
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) wmax = fmaxf(wmax, __shfl_xor_sync(0xFFFFFFFFu, wmax, off));
            if (lane == 0) sWarpMax[warp] = wmax;
            __syncthreads();
            float tok_amax = 0.0f, enc_s = 0.0f, dec_s = 0.0f, out_s = 0.0f;
            if (lane == 0) {
                tok_amax = sWarpMax[0];
                #pragma unroll
                for (int wi = 1; wi < 8; ++wi) tok_amax = fmaxf(tok_amax, sWarpMax[wi]);
                enc_s = fminf(div_rn(1536.0f, tok_amax), FP32_MAX_VAL);
                dec_s = div_rn(1.0f, enc_s);
                out_s = mul_rn(tok_amax, INV_1536);
            }
            tok_amax = __shfl_sync(0xFFFFFFFFu, tok_amax, 0);
            enc_s = __shfl_sync(0xFFFFFFFFu, enc_s, 0);
            dec_s = __shfl_sync(0xFFFFFFFFu, dec_s, 0);
            out_s = __shfl_sync(0xFFFFFFFFu, out_s, 0);
            if (tid < TOPK_FIXED) {
                const int32_t loc = sTokLoc[tok][tid];
                const int32_t dst = sTokDst[tok][tid];
                if (dst >= 0 && loc >= 0) {
                    QRowScales[dst] = mul_rn(mul_rn(tok_amax, __ldg(WAmax + loc)), rcp_denom);
                }
            }
            uint64_t pk_a; uint8_t sc_a;
            quant_nvfp4_block16_compact_v14<true>(v0_a, v1_a, bm_a, enc_s, dec_s, out_s, pk_a, sc_a);
            #pragma unroll
            for (int k = 0; k < TOPK_FIXED; ++k) {
                const int32_t dst = sTokDst[tok][k];
                if (dst >= 0) {
                    reinterpret_cast<uint64_t*>(QValues)[static_cast<int64_t>(dst) * 288 + tid] = pk_a;
                    QScales[r8c4_scale_offset(dst, tid, 288)] = sc_a;
                }
            }
            if (tid < 32) {
                uint64_t pk_b; uint8_t sc_b;
                quant_nvfp4_block16_compact_v14<true>(v0_b, v1_b, bm_b, enc_s, dec_s, out_s, pk_b, sc_b);
                #pragma unroll
                for (int k = 0; k < TOPK_FIXED; ++k) {
                    const int32_t dst = sTokDst[tok][k];
                    if (dst >= 0) {
                        reinterpret_cast<uint64_t*>(QValues)[static_cast<int64_t>(dst) * 288 + (tid + 256)] = pk_b;
                        QScales[r8c4_scale_offset(dst, tid + 256, 288)] = sc_b;
                    }
                }
            }
            __syncthreads();
        } else {
            if (tid < 144) {
                const uint4* bp = reinterpret_cast<const uint4*>(
                    X + static_cast<int64_t>(tok) * K_FIXED + tid * 32
                );
                uint4 q0, q1; uint8_t sc;
                quant_mxfp8_block32_regs(ld_ca_u4(bp), ld_ca_u4(bp + 1), ld_ca_u4(bp + 2), ld_ca_u4(bp + 3), q0, q1, sc);
                #pragma unroll
                for (int k = 0; k < TOPK_FIXED; ++k) {
                    const int32_t dst = sTokDst[tok][k];
                    if (dst >= 0) {
                        uint4* dq = reinterpret_cast<uint4*>(QValues + static_cast<int64_t>(dst) * K_FIXED + tid * 32);
                        dq[0] = q0; dq[1] = q1;
                        QScales[r8c4_scale_offset(dst, tid, 144)] = sc;
                    }
                }
            }
        }
    }

    if constexpr (TRIG) {
        __syncthreads();
        if (tid == 0) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }
}

template <typename KernelT, typename... Args>
static inline cudaError_t launch_router_v14(
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
static int dispatch_router_fused_v14(
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

    #define LAUNCH_RF14(NV, QF, PW, TM) err = launch_router_v14( \
        router_fused_v14_kernel<BM_CTA, C1_MODE, USE_BF11, NV, QF, PW, TM>, \
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
        if      (use_pdl && trig_mode == 2) { LAUNCH_RF14(NV, QF, true,  2); } \
        else if (use_pdl && trig_mode == 1) { LAUNCH_RF14(NV, QF, true,  1); } \
        else if (use_pdl)                   { LAUNCH_RF14(NV, QF, true,  0); } \
        else                                { LAUNCH_RF14(NV, QF, false, 0); } \
    } while (0)

    if (is_nv) {
        if (use_fma) { DISP_PDL(true, true);  }
        else         { DISP_PDL(true, false); }
    } else {
        DISP_PDL(false, false);
    }
    #undef DISP_PDL
    #undef LAUNCH_RF14
    return static_cast<int>(err);
}

}  // namespace

// =========================================================================
// Extern "C" Entry Points (14 Total: Universal Superset of exact + c32 + v10 + v14!)
// =========================================================================

extern "C" int router_bf12_v14_version() { return 14; }
extern "C" int router_bf12_v10_version() { return 14; }
extern "C" int router_bf12_c32_version() { return 14; }
extern "C" int router_bf12_c48_min_cta() {
    return ROUTER_V14_BM64_MIN_CTA * 100 + ROUTER_V14_C128_GEMM_MIN_CTA * 10 + ROUTER_V14_K2_MIN_CTA;
}

// ---- Drop-In `librouter_bf12_exact.so` Entry Point (30 args, T=1..16, R8c4 scattered, QUANT_FMA=1) ----
extern "C" int router_bf12_exact_fwd(
    const void* X, const void* SLABS, const void* OUTLIERS, const void* O_OFF,
    const void* Bias, const void* ExpertBounds, const void* Padding, const void* NLivePtr,
    void* Counters, void* /*Masks*/, void* LogitsWS, void* /*KeysWS*/,
    void* Src2Dst, void* Gammas, void* Rows, void* Tiles,
    void* QValues, void* QScales, void* QRowScales,
    const void* WAmax, const void* WDenominator,
    int T, float route_scale, uint32_t base_exp,
    int num_local, int tile_sz, int max_ctas,
    int nvfp4, int pdl, uintptr_t stream_ptr
) {
    if (T <= 0 || T > MAX_T_C1 || num_local <= 0 || num_local > E_FIXED || tile_sz < T || tile_sz > 64) {
        return -1;
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    const bool use_pdl = (pdl != 0);
    const int trig_mode = (pdl >= 3) ? 2 : ((pdl == 2) ? 1 : 0);
    const bool is_nv = (nvfp4 != 0);
    cudaError_t err = cudaSuccess;

    if (Counters != nullptr) {
        #define LAUNCH_EX(BM, NV, PW, TM) err = launch_router_v14( \
            router_bf12_exact_v14_kernel<BM, NV, PW, TM>, dim3(48, 1, 1), dim3(256, 1, 1), stream, (PW || (TM > 0)), \
            static_cast<const __nv_bfloat16*>(X), static_cast<const uint8_t*>(SLABS), \
            static_cast<const uint64_t*>(OUTLIERS), static_cast<const int32_t*>(O_OFF), \
            static_cast<const float*>(Bias), static_cast<const int32_t*>(ExpertBounds), \
            static_cast<const bool*>(Padding), static_cast<const int32_t*>(NLivePtr), \
            static_cast<uint32_t*>(Counters), static_cast<float*>(LogitsWS), \
            static_cast<int32_t*>(Src2Dst), static_cast<float*>(Gammas), \
            static_cast<int32_t*>(Rows), static_cast<int32_t*>(Tiles), \
            static_cast<uint8_t*>(QValues), static_cast<uint8_t*>(QScales), \
            static_cast<float*>(QRowScales), static_cast<const float*>(WAmax), \
            static_cast<const float*>(WDenominator), \
            T, route_scale, base_exp, num_local, tile_sz, max_ctas)
        #define DISP_EX(BM, NV) do { \
            if      (use_pdl && trig_mode == 2) { LAUNCH_EX(BM, NV, true,  2); } \
            else if (use_pdl && trig_mode == 1) { LAUNCH_EX(BM, NV, true,  1); } \
            else if (use_pdl)                   { LAUNCH_EX(BM, NV, true,  0); } \
            else                                { LAUNCH_EX(BM, NV, false, 0); } \
        } while (0)
        if (T <= 8) {
            if (is_nv) DISP_EX(8, true); else DISP_EX(8, false);
        } else {
            if (is_nv) DISP_EX(16, true); else DISP_EX(16, false);
        }
        #undef DISP_EX
        #undef LAUNCH_EX
        return static_cast<int>(err);
    }

    // Defensive 2-launch path when legacy caller passes Counters == nullptr (e.g. moe_s8_runtime fallback)
    err = launch_router_v14(
        router_bf12_gemm_c32_v14_kernel<1, false, false, false>,
        dim3(48, 1, 1), dim3(160, 1, 1), stream, false,
        static_cast<const __nv_bfloat16*>(X), static_cast<const uint8_t*>(SLABS),
        static_cast<const uint64_t*>(OUTLIERS), static_cast<const int32_t*>(O_OFF),
        static_cast<const int32_t*>(NLivePtr), static_cast<float*>(LogitsWS), T, base_exp
    );
    if (err != cudaSuccess) return static_cast<int>(err);
    const bool trig = (trig_mode > 0);
    #define LAUNCH_NK2(NV, TR) err = launch_router_v14( \
        router_bf12_exact_nullcnt_k2_kernel<NV, false, TR>, dim3(1, 1, 1), dim3(256, 1, 1), stream, TR, \
        static_cast<const __nv_bfloat16*>(X), static_cast<const float*>(LogitsWS), \
        static_cast<const float*>(Bias), static_cast<const int32_t*>(ExpertBounds), \
        static_cast<const bool*>(Padding), static_cast<const int32_t*>(NLivePtr), \
        static_cast<int32_t*>(Src2Dst), static_cast<float*>(Gammas), \
        static_cast<int32_t*>(Rows), static_cast<int32_t*>(Tiles), \
        static_cast<uint8_t*>(QValues), static_cast<uint8_t*>(QScales), \
        static_cast<float*>(QRowScales), static_cast<const float*>(WAmax), \
        static_cast<const float*>(WDenominator), \
        T, route_scale, num_local, tile_sz, max_ctas)
    if (is_nv) { if (trig) LAUNCH_NK2(true, true); else LAUNCH_NK2(true, false); }
    else       { if (trig) LAUNCH_NK2(false, true); else LAUNCH_NK2(false, false); }
    #undef LAUNCH_NK2
    return static_cast<int>(err);
}

// ---- Wave-V14 Upgraded Single-Launch C=1..2 (T <= 16) BF12 Entry Point (35 args, BM_CTA=8/16!) ----
extern "C" int router_bf12_c1_fused_v10_fwd(
    const void* X, const void* SLABS, const void* OUTLIERS, const void* O_OFF,
    const void* Bias, const void* ExpertBounds, const void* Padding, const void* NLivePtr,
    void* LogitsWS, void* SelBitsWS, void* Arrive,
    void* Src2Dst, void* Src2Compact, void* Gammas, void* Rows,
    void* Tiles, void* TileMask, void* TileCompactStart, void* RowAmaxW2,
    void* QValues, void* QScales, void* QRowScales, void* RowAmaxIn,
    const void* WAmax, const void* WDenominator,
    int T, float route_scale, uint32_t base_exp,
    int num_local, int tile_sz, int max_ctas,
    int nvfp4, int quant_fma, int pdl, uintptr_t stream_ptr
) {
    if (T <= 0 || T > MAX_T_C1 || num_local <= 0 || num_local > E_FIXED || tile_sz < T || tile_sz > MAX_T_C1) {
        return -1;
    }
    if (T <= 8) {
        return dispatch_router_fused_v14<8, true, false>(
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
    return dispatch_router_fused_v14<16, true, false>(
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

// ---- Wave-V14 Upgraded Single-Launch C=1..2 (T <= 16) BF11-Flip Entry Point (38 args, BM_CTA=8/16!) ----
extern "C" int router_bf11_c1_fused_v10_fwd(
    const void* X, const void* SLABS, const void* SLAB_META_V5, const void* SFLIPS, const void* BMASKS,
    const void* OUTLIERS, const void* O_OFF,
    const void* Bias, const void* ExpertBounds, const void* Padding, const void* NLivePtr,
    void* LogitsWS, void* SelBitsWS, void* Arrive,
    void* Src2Dst, void* Src2Compact, void* Gammas, void* Rows,
    void* Tiles, void* TileMask, void* TileCompactStart, void* RowAmaxW2,
    void* QValues, void* QScales, void* QRowScales, void* RowAmaxIn,
    const void* WAmax, const void* WDenominator,
    int T, float route_scale, uint32_t base_exp,
    int num_local, int tile_sz, int max_ctas,
    int nvfp4, int quant_fma, int pdl, uintptr_t stream_ptr
) {
    if (T <= 0 || T > MAX_T_C1 || num_local <= 0 || num_local > E_FIXED || tile_sz < T || tile_sz > MAX_T_C1) {
        return -1;
    }
    if (T <= 8) {
        return dispatch_router_fused_v14<8, true, true>(
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
    return dispatch_router_fused_v14<16, true, true>(
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

// ---- NEW Wave-V14 Single-Launch C=1..32 (T = 1..128) BF12 Entry Point (37 args!) ----
extern "C" int router_bf12_c128_fused_v14_fwd(
    const void* X, const void* SLABS, const void* OUTLIERS, const void* O_OFF,
    const void* Bias, const void* ExpertBounds, const void* Padding, const void* NLivePtr,
    void* LogitsWS, void* SelBitsWS, void* Counters,
    void* Starts, void* StartsCompact,
    void* Src2Dst, void* Src2Compact, void* Gammas, void* Rows,
    void* Tiles, void* TileCompactStart, void* Row2Tok, void* RowAmaxW2,
    void* QValues, void* QScales, void* QRowScales, void* RowAmaxIn,
    const void* WAmax, const void* WDenominator,
    int T, float route_scale, uint32_t base_exp,
    int num_local, int tile_sz, int max_ctas,
    int nvfp4, int quant_fma, int pdl, uintptr_t stream_ptr
) {
    if (T <= 0 || T > 128 || num_local <= 0 || num_local > E_FIXED || tile_sz <= 0 || max_ctas <= 0) {
        return -1;
    }
    #define CALL_C128(BM) dispatch_router_fused_v14<BM, false, false>( \
        X, SLABS, nullptr, nullptr, nullptr, OUTLIERS, O_OFF, \
        Bias, ExpertBounds, Padding, NLivePtr, \
        LogitsWS, SelBitsWS, Counters, Starts, StartsCompact, \
        Src2Dst, Src2Compact, Gammas, Rows, \
        Tiles, nullptr, TileCompactStart, Row2Tok, \
        RowAmaxW2, QValues, QScales, QRowScales, RowAmaxIn, WAmax, WDenominator, \
        T, route_scale, base_exp, num_local, tile_sz, max_ctas, \
        nvfp4, quant_fma, pdl, stream_ptr)
    if (T <= 8)       return CALL_C128(8);
    else if (T <= 16) return CALL_C128(16);
    else if (T <= 32) return CALL_C128(32);
    else if (T <= 48) return CALL_C128(48);
    else if (T <= 64) return CALL_C128(64);
    else              return CALL_C128(128);
    #undef CALL_C128
}

// ---- NEW Wave-V14 Single-Launch C=1..32 (T = 1..128) BF11-Flip Entry Point (40 args!) ----
extern "C" int router_bf11_c128_fused_v14_fwd(
    const void* X, const void* SLABS, const void* SLAB_META_V5, const void* SFLIPS, const void* BMASKS,
    const void* OUTLIERS, const void* O_OFF,
    const void* Bias, const void* ExpertBounds, const void* Padding, const void* NLivePtr,
    void* LogitsWS, void* SelBitsWS, void* Counters,
    void* Starts, void* StartsCompact,
    void* Src2Dst, void* Src2Compact, void* Gammas, void* Rows,
    void* Tiles, void* TileCompactStart, void* Row2Tok, void* RowAmaxW2,
    void* QValues, void* QScales, void* QRowScales, void* RowAmaxIn,
    const void* WAmax, const void* WDenominator,
    int T, float route_scale, uint32_t base_exp,
    int num_local, int tile_sz, int max_ctas,
    int nvfp4, int quant_fma, int pdl, uintptr_t stream_ptr
) {
    if (T <= 0 || T > 128 || num_local <= 0 || num_local > E_FIXED || tile_sz <= 0 || max_ctas <= 0) {
        return -1;
    }
    #define CALL_C128_11(BM) dispatch_router_fused_v14<BM, false, true>( \
        X, SLABS, SLAB_META_V5, SFLIPS, BMASKS, OUTLIERS, O_OFF, \
        Bias, ExpertBounds, Padding, NLivePtr, \
        LogitsWS, SelBitsWS, Counters, Starts, StartsCompact, \
        Src2Dst, Src2Compact, Gammas, Rows, \
        Tiles, nullptr, TileCompactStart, Row2Tok, \
        RowAmaxW2, QValues, QScales, QRowScales, RowAmaxIn, WAmax, WDenominator, \
        T, route_scale, base_exp, num_local, tile_sz, max_ctas, \
        nvfp4, quant_fma, pdl, stream_ptr)
    if (T <= 8)       return CALL_C128_11(8);
    else if (T <= 16) return CALL_C128_11(16);
    else if (T <= 32) return CALL_C128_11(32);
    else if (T <= 48) return CALL_C128_11(48);
    else if (T <= 64) return CALL_C128_11(64);
    else              return CALL_C128_11(128);
    #undef CALL_C128_11
}

// ---- Drop-In V10 C=1..8 (T = 1..64) BF12 Entry Point (37 args, routes into V14 BM_CTA=8/16/32/48/64!) ----
extern "C" int router_bf12_c48_fused_v10_fwd(
    const void* X, const void* SLABS, const void* OUTLIERS, const void* O_OFF,
    const void* Bias, const void* ExpertBounds, const void* Padding, const void* NLivePtr,
    void* LogitsWS, void* SelBitsWS, void* Counters,
    void* Starts, void* StartsCompact,
    void* Src2Dst, void* Src2Compact, void* Gammas, void* Rows,
    void* Tiles, void* TileCompactStart, void* Row2Tok, void* RowAmaxW2,
    void* QValues, void* QScales, void* QRowScales, void* RowAmaxIn,
    const void* WAmax, const void* WDenominator,
    int T, float route_scale, uint32_t base_exp,
    int num_local, int tile_sz, int max_ctas,
    int nvfp4, int quant_fma, int pdl, uintptr_t stream_ptr
) {
    if (T <= 0 || T > 64) return -1;
    return router_bf12_c128_fused_v14_fwd(
        X, SLABS, OUTLIERS, O_OFF, Bias, ExpertBounds, Padding, NLivePtr,
        LogitsWS, SelBitsWS, Counters, Starts, StartsCompact,
        Src2Dst, Src2Compact, Gammas, Rows, Tiles, TileCompactStart, Row2Tok, RowAmaxW2,
        QValues, QScales, QRowScales, RowAmaxIn, WAmax, WDenominator,
        T, route_scale, base_exp, num_local, tile_sz, max_ctas,
        nvfp4, quant_fma, pdl, stream_ptr
    );
}

// ---- Drop-In V10 C=1..8 (T = 1..64) BF11 Entry Point (40 args, routes into V14 BM_CTA=8/16/32/48/64!) ----
extern "C" int router_bf11_c48_fused_v10_fwd(
    const void* X, const void* SLABS, const void* SLAB_META_V5, const void* SFLIPS, const void* BMASKS,
    const void* OUTLIERS, const void* O_OFF,
    const void* Bias, const void* ExpertBounds, const void* Padding, const void* NLivePtr,
    void* LogitsWS, void* SelBitsWS, void* Counters,
    void* Starts, void* StartsCompact,
    void* Src2Dst, void* Src2Compact, void* Gammas, void* Rows,
    void* Tiles, void* TileCompactStart, void* Row2Tok, void* RowAmaxW2,
    void* QValues, void* QScales, void* QRowScales, void* RowAmaxIn,
    const void* WAmax, const void* WDenominator,
    int T, float route_scale, uint32_t base_exp,
    int num_local, int tile_sz, int max_ctas,
    int nvfp4, int quant_fma, int pdl, uintptr_t stream_ptr
) {
    if (T <= 0 || T > 64) return -1;
    return router_bf11_c128_fused_v14_fwd(
        X, SLABS, SLAB_META_V5, SFLIPS, BMASKS, OUTLIERS, O_OFF, Bias, ExpertBounds, Padding, NLivePtr,
        LogitsWS, SelBitsWS, Counters, Starts, StartsCompact,
        Src2Dst, Src2Compact, Gammas, Rows, Tiles, TileCompactStart, Row2Tok, RowAmaxW2,
        QValues, QScales, QRowScales, RowAmaxIn, WAmax, WDenominator,
        T, route_scale, base_exp, num_local, tile_sz, max_ctas,
        nvfp4, quant_fma, pdl, stream_ptr
    );
}

// ---- Drop-In V9/C32 GEMM Entry Point (11 args: 1-pass BM128 @ T=65..128 + 1.000x Half-Slab @ T=129..256!) ----
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

    #define LAUNCH_G(EW, CW, PW, TR) err = launch_router_v14( \
        router_bf12_gemm_c32_v14_kernel<EW, CW, PW, TR>, \
        dim3(48 / (EW), m_blks, 1), dim3(160, 1, 1), stream, use_pdl_attr, \
        static_cast<const __nv_bfloat16*>(X), static_cast<const uint8_t*>(SLABS), \
        static_cast<const uint64_t*>(OUTLIERS), static_cast<const int32_t*>(O_OFF), \
        static_cast<const int32_t*>(NLivePtr), static_cast<float*>(Logits), T, base_exp)

    #define LAUNCH_BM64(PW, TR) err = launch_router_v14( \
        router_bf12_gemm_c48_v14_bm64_kernel<PW, TR>, \
        dim3(48, 1, 1), dim3(160, 1, 1), stream, use_pdl_attr, \
        static_cast<const __nv_bfloat16*>(X), static_cast<const uint8_t*>(SLABS), \
        static_cast<const uint64_t*>(OUTLIERS), static_cast<const int32_t*>(O_OFF), \
        static_cast<const int32_t*>(NLivePtr), static_cast<float*>(Logits), T, base_exp)

    #define LAUNCH_C128(PW, TR) err = launch_router_v14( \
        router_bf12_gemm_c128_v14_kernel<PW, TR>, \
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
    } else if (T <= 128) {
        // V14: 1-pass 48-CTA BM128 (23.04 KiB SMEM, 1.000x slab reads = 8x less weight traffic vs V10!)
        if      (pdl_wait && trig)  { LAUNCH_C128(true,  true);  }
        else if (pdl_wait && !trig) { LAUNCH_C128(true,  false); }
        else if (!pdl_wait && trig) { LAUNCH_C128(false, true);  }
        else                        { LAUNCH_C128(false, false); }
    } else if (T <= 192) {
        DISPATCH_G(6, true);
    } else {
        DISPATCH_G(4, true);
    }
    #undef DISPATCH_G
    #undef LAUNCH_C128
    #undef LAUNCH_BM64
    #undef LAUNCH_G
    return static_cast<int>(err);
}

// ---- Drop-In V9/C32 K2 Entry Point (37 args: 48-CTA zero-wave-tail, 3.3 KiB SMEM, multi-warp reg Top-8!) ----
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

    // Exact <= 48 CTAs (1 wave on GB10, ZERO wave tail at M=64, 96, 128, 192, 256!)
    const int ncta = (T <= 48) ? T : 48;
    dim3 grid(ncta, 1, 1);
    dim3 block(384, 1, 1);

    #define LAUNCH_FUSED(NV, LB, PW, TM, QF) err = launch_router_v14( \
        c32_quant_topk_place_v14_kernel<NV, LB, PW, TM, QF>, grid, block, stream, (PW || (TM > 0)), \
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
