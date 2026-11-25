// SPDX-License-Identifier: Apache-2.0
// SM121a (GB10) Wave-V16 C=1..2 (T <= 16) L2::evict_first + LPT Dynamic-Queue + W2 Pre-Wait Hoist 0-ULP MX7 Kernels
// Targets Inkling-2-Small TP8 Global MXFP8 MoE layers (8L: L5,11,17,23,29,35,41,47; E=384, H=4608, inter=288, topk=8):
//   - MX_W13 (N=576,  K=4608): BN=16 (2 warps, 36 panels, MIN_CTA=8 -> 384 CTAs = 8/SM)
//                              / BN=32 (4 warps, 18 panels, MIN_CTA=6 -> 288 CTAs = 6/SM)
//                              with exact GPU Wave-Adaptive (`bn_mode=3`, `prefer_w13_bn16`) +
//                              in-W13 last-panel in-place L2-hot MXFP8 quant-dequant (`FUSE_QDEQ`)!
//   - MX_W2  (N=4608, K=288):  BN=128 (4 warps, 36 panels, SUB_N=4, hot FQ=2/0 MIN_CTA=6 -> 288 CTAs = 6/SM,
//                              80 regs/thr, 0 spills!), 3-SHFL 4x4 coalesced 64B `st_cg_u4`, V13 3-wide
//                              L2-hot combine ring (`va, vb, vc`, -25 regs, exact slot 0..7 FP32 `add_rn`!).
//
// Wave-V14 Upgrades over mxfp8_mx7_c1_v5.cu (literal 0-ULP nd == 0, 0 B stack / 0 B spills):
//   1. In-W13 Last-Panel In-Place MXFP8 Quant-Dequant (`FUSE_QDEQ` via self-resetting `TileQdeqDone[t]`):
//        - Eliminates 35/36 (97.2%) of redundant MXFP8 activation quant-dequant work in W2!
//        - Cuts ~16 live registers + ~1,000 instructions from hot W2 (`FQ=2` vectorized `ld_cg_u4` copy),
//          unlocking `MX7_C1_W2_T8_MIN_CTA=6` (288 CTAs = 6/SM) with 0 spills!
//        - Cold legacy OTF quant (`FQ=1`) and legacy v4 (`V5_MODE=0`) decoupled to lower `MIN_CTA` so hot
//          Compat-Hybrid (`V5_MODE=1`) & Panel-V5 (`V5_MODE=2`) ALWAYS compile at max occupancy.
//   2. Zero-Math Inner-Loop Register Liveness Trims:
//        - Eliminates upfront 16-reg `sm_arr[8]` / `e21_arr[8]` expansion (on-the-fly extraction per q/sub).
//        - Drops live `int64_t e` & `row_base` across K-loop (reloads `Tiles[t]` at epilogue).
//        - Eliminates `sA` / `sA_all` zero-fill loops & dead-row `LDS` via `row0_live`/`row8_live` predication.
//   3. W2 Epilogue & Fused Combine Upgrade:
//        - Upgrades W2 from 4B `st_cg_u32` + `ld_cg_u2` (4 BF16/thr) to 3-SHFL 4x4 transpose + coalesced
//          64B `st_cg_u4` + V13 3-wide `ld_cg_u4` combine ring (8 BF16/thr, 8 tokens/iter, exact slot 0..7!).
//   4. PTX `L2::evict_first` Weight Streaming Hints (`createpolicy.fractional.L2::evict_first.b64` +
//      `ld.global.cs.L2::cache_hint`): protects L2-resident activations, SiTU_Out, DownCompact & L2PF!
//   5. Native Wave-Adaptive `bn_mode=3` (`prefer_w13_bn16`) + Defense-in-Depth PDL Guard
//      (`if (pdl_wait_mode == 0 && trig_dep != 0) pdl_wait_mode = 1;`) across all launchers.
//   6. 100% Backward Compatible: exports all 13 v4/v5 symbols + `mx7_c1_version() == 14`,
//      `mx7_c1_min_cta()`, and 6 v14 entrypoints (`mx7_w13_c1_v14{,_compat,_panel}_fwd`,
//      `mx7_w2_c1_v14{,_compat,_panel}_fwd`).
//
// Build on GB10 (SM121a):
//   /usr/local/cuda/bin/nvcc -O3 -std=c++17 -gencode arch=compute_121a,code=sm_121a \
//       --fmad=false --shared -Xcompiler -fPIC --ptxas-options=-v \
//       mxfp8_mx7_c1_v14.cu -o libmxfp8_mx7_c1_v14.so
//   (MANDATORY: --fmad=false, NO -arch=sm_121a, NO --ftz=true, NO --use_fast_math!)

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

#ifndef MX7_C1_W13_T8_BN32_MIN_CTA
#define MX7_C1_W13_T8_BN32_MIN_CTA 6
#endif

#ifndef MX7_C1_W13_T8_BN16_MIN_CTA
#define MX7_C1_W13_T8_BN16_MIN_CTA 8
#endif

#ifndef MX7_C1_W13_T16_BN32_MIN_CTA
#define MX7_C1_W13_T16_BN32_MIN_CTA 5
#endif

#ifndef MX7_C1_W13_T16_BN16_MIN_CTA
#define MX7_C1_W13_T16_BN16_MIN_CTA 5
#endif

#ifndef MX7_C1_W2_T8_MIN_CTA
#define MX7_C1_W2_T8_MIN_CTA 6
#endif

#ifndef MX7_C1_W2_T16_MIN_CTA
#define MX7_C1_W2_T16_MIN_CTA 5
#endif

#ifndef MX7_USE_L2_EVICT_HINT
#define MX7_USE_L2_EVICT_HINT 1
#endif

namespace {

static constexpr int MX7_SLAB_BYTES      = 896;
static constexpr int MX7_RAW8_SLAB_BYTES = 1024;
static constexpr int MX7_PANEL_MX7_BYTES = 3584;
static constexpr int MX7_PANEL_R8_BYTES  = 4096;
static constexpr int W13_WS3_HDR_BLOCKS  = 20;
static constexpr int W2_WS3_HDR_BLOCKS   = 5;
static constexpr int W13_WS3_WARP_U32    = 160;
static constexpr int W2_WS3_WARP_U32     = 40;

static constexpr uint32_t META_DENSE_BIT      = 0x80000000u;
static constexpr uint32_t META_ALL_NORMAL_BIT = 0x40000000u;
static constexpr uint32_t META_HAS_NAN_BIT    = 0x20000000u;
static constexpr uint32_t META_NF_SHIFT       = 23u;
static constexpr uint32_t META_NF_MASK        = 0x3Fu;
static constexpr uint32_t META_OFF_MASK       = 0x007FFFFFu;

static constexpr float FP32_MAX_VAL = 3.4028234663852886e38f;
static constexpr float INV_25       = 1.0f / 25.0f;
static constexpr float INV_448      = 1.0f / 448.0f;

// =========================================================================
// PTX Memory, L2::evict_first Cache Policy, Sync & IEEE-754 Exact Math
// =========================================================================

static __device__ __forceinline__ uint64_t make_l2_evict_first_policy() {
    uint64_t policy = 0ULL;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 && MX7_USE_L2_EVICT_HINT
    float frac = 1.0f;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, %1;"
                 : "=l"(policy) : "f"(frac));
#endif
    return policy;
}

static __device__ __forceinline__ uint4 ld_cs_u4(const void* p) {
    uint4 v;
    asm volatile("ld.global.cs.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

#if MX7_USE_L2_EVICT_HINT
static __device__ __forceinline__ uint4 ld_cs_ef_u4(const void* p, uint64_t pol) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    uint4 v;
    asm volatile("ld.global.cs.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], %5;"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p), "l"(pol));
    return v;
#else
    (void)pol;
    return ld_cs_u4(p);
#endif
}
#endif

static __device__ __forceinline__ uint4 ld_w_u4(const void* p, bool use_ef, uint64_t pol) {
#if MX7_USE_L2_EVICT_HINT
    return use_ef ? ld_cs_ef_u4(p, pol) : ld_cs_u4(p);
#else
    (void)use_ef; (void)pol;
    return ld_cs_u4(p);
#endif
}

static __device__ __forceinline__ uint2 ld_cs_u2(const void* p) {
    uint2 v;
    asm volatile("ld.global.cs.v2.u32 {%0,%1}, [%2];"
                 : "=r"(v.x), "=r"(v.y)
                 : "l"(p));
    return v;
}

#if MX7_USE_L2_EVICT_HINT
static __device__ __forceinline__ uint2 ld_cs_ef_u2(const void* p, uint64_t pol) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    uint2 v;
    asm volatile("ld.global.cs.L2::cache_hint.v2.u32 {%0,%1}, [%2], %3;"
                 : "=r"(v.x), "=r"(v.y)
                 : "l"(p), "l"(pol));
    return v;
#else
    (void)pol;
    return ld_cs_u2(p);
#endif
}
#endif

static __device__ __forceinline__ uint2 ld_w_u2(const void* p, bool use_ef, uint64_t pol) {
#if MX7_USE_L2_EVICT_HINT
    return use_ef ? ld_cs_ef_u2(p, pol) : ld_cs_u2(p);
#else
    (void)use_ef; (void)pol;
    return ld_cs_u2(p);
#endif
}

static __device__ __forceinline__ uint32_t ld_cs_u32(const void* p) {
    uint32_t v;
    asm volatile("ld.global.cs.u32 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
}

#if MX7_USE_L2_EVICT_HINT
static __device__ __forceinline__ uint32_t ld_cs_ef_u32(const void* p, uint64_t pol) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    uint32_t v;
    asm volatile("ld.global.cs.L2::cache_hint.u32 %0, [%1], %2;"
                 : "=r"(v) : "l"(p), "l"(pol));
    return v;
#else
    (void)pol;
    return ld_cs_u32(p);
#endif
}
#endif

static __device__ __forceinline__ uint32_t ld_w_u32(const void* p, bool use_ef, uint64_t pol) {
#if MX7_USE_L2_EVICT_HINT
    return use_ef ? ld_cs_ef_u32(p, pol) : ld_cs_u32(p);
#else
    (void)use_ef; (void)pol;
    return ld_cs_u32(p);
#endif
}

static __device__ __forceinline__ uint16_t ld_cs_u16(const void* p) {
    uint16_t v;
    asm volatile("ld.global.cs.u16 %0, [%1];" : "=h"(v) : "l"(p));
    return v;
}

#if MX7_USE_L2_EVICT_HINT
static __device__ __forceinline__ uint16_t ld_cs_ef_u16(const void* p, uint64_t pol) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    uint16_t v;
    asm volatile("ld.global.cs.L2::cache_hint.u16 %0, [%1], %2;"
                 : "=h"(v) : "l"(p), "l"(pol));
    return v;
#else
    (void)pol;
    return ld_cs_u16(p);
#endif
}
#endif

static __device__ __forceinline__ uint16_t ld_w_u16(const void* p, bool use_ef, uint64_t pol) {
#if MX7_USE_L2_EVICT_HINT
    return use_ef ? ld_cs_ef_u16(p, pol) : ld_cs_u16(p);
#else
    (void)use_ef; (void)pol;
    return ld_cs_u16(p);
#endif
}

static __device__ __forceinline__ uint4 ld_cg_u4(const void* p) {
    uint4 v;
    asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
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

static __device__ __forceinline__ int fns_u32(uint32_t mask, uint32_t base, int offset) {
    uint32_t r;
    asm("fns.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(mask), "r"(base), "r"(offset));
    return static_cast<int>(r);
}

static __device__ __forceinline__ uint32_t prmt_b32(uint32_t a, uint32_t b, uint32_t sel) {
    uint32_t out;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(out) : "r"(a), "r"(b), "r"(sel));
    return out;
}

static __device__ __forceinline__ float add_rn(float a, float b) {
    float r; asm("add.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r;
}
static __device__ __forceinline__ float mul_rn(float a, float b) {
    float r; asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r;
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
static __device__ __forceinline__ uint32_t pack_bf16x2(float f0, float f1) {
    uint32_t out;
    asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(out) : "f"(f0), "f"(f1));
    return out;
}

static __device__ __forceinline__ float e8m0_to_f32(uint8_t s) {
    if (s == 0) return 5.877471754111438e-39f;
    if (s == 255) return __int_as_float(0x7FC00000);
    return __uint_as_float(static_cast<uint32_t>(s) << 23);
}

static __device__ __forceinline__ int64_t scale_offset_decode(int64_t row, int col, int cols) {
    int64_t tile = (row >> 3) * ((cols + 3) >> 2) + (col >> 2);
    return (tile << 5) + ((row & 7) << 2) + (col & 3);
}

// ---- Exact 0-ULP GPU Wave-Efficiency Predicate for Adaptive BN=16 vs BN=32 ----

// ---- Wave-V16 O(1)-per-tile Deterministic 3-Bucket LPT Tile Permutation ----
template <int TILE>
static __device__ __forceinline__ int classify_tile_lpt(
    const int32_t* __restrict__ Tiles, int MAX_CTAS, int n_active, int t
) {
    const int e = Tiles[t];
    const int end_row = Tiles[MAX_CTAS + t];
    int live_m = end_row - t * TILE;
    if (live_m > TILE) live_m = TILE;
    if (live_m < 0) live_m = 0;
    const bool is_split =
        ((t + 1 < n_active) && (Tiles[t + 1] == e)) ||
        ((t > 0) && (Tiles[t - 1] == e));
    constexpr int HEAVY_THR = (TILE >= 32) ? 16 : ((TILE >= 16) ? 8 : 3);
    constexpr int MED_THR   = (TILE >= 32) ? 4  : 2;
    if (is_split || live_m > HEAVY_THR) return 0;
    if (live_m >= MED_THR) return 1;
    return 2;
}

template <int TILE, int MAX_LPT_TILES>
static __device__ __forceinline__ void build_lpt_tile_order(
    const int32_t* __restrict__ Tiles,
    int MAX_CTAS,
    int n_active,
    int16_t* __restrict__ sTileOrder,
    int warp,
    int lane
) {
    static_assert(MAX_LPT_TILES <= 512, "32-bit tags register covers up to 16*32=512 tiles");
    if (warp == 0) {
        uint32_t tags = 0u;
        int cnt0 = 0;
        int cnt1 = 0;
        const int n_chunks = (n_active + 31) >> 5;
        for (int ch = 0; ch < n_chunks; ++ch) {
            const int t = (ch << 5) + lane;
            uint32_t b = 3u;
            if (t < n_active) {
                b = static_cast<uint32_t>(classify_tile_lpt<TILE>(Tiles, MAX_CTAS, n_active, t));
            }
            tags |= (b << (ch << 1));
            cnt0 += __popc(__ballot_sync(0xFFFFFFFFu, b == 0u));
            cnt1 += __popc(__ballot_sync(0xFFFFFFFFu, b == 1u));
        }
        int pos0 = 0;
        int pos1 = cnt0;
        int pos2 = cnt0 + cnt1;
        const uint32_t lane_mask = (1u << lane) - 1u;
        for (int ch = 0; ch < n_chunks; ++ch) {
            const int t = (ch << 5) + lane;
            const uint32_t b = (tags >> (ch << 1)) & 3u;
            const uint32_t m0 = __ballot_sync(0xFFFFFFFFu, b == 0u);
            const uint32_t m1 = __ballot_sync(0xFFFFFFFFu, b == 1u);
            const uint32_t m2 = __ballot_sync(0xFFFFFFFFu, b == 2u);
            if      (b == 0u) sTileOrder[pos0 + __popc(m0 & lane_mask)] = static_cast<int16_t>(t);
            else if (b == 1u) sTileOrder[pos1 + __popc(m1 & lane_mask)] = static_cast<int16_t>(t);
            else if (b == 2u) sTileOrder[pos2 + __popc(m2 & lane_mask)] = static_cast<int16_t>(t);
            pos0 += __popc(m0);
            pos1 += __popc(m1);
            pos2 += __popc(m2);
        }
    }
}

static __device__ __forceinline__ bool prefer_w13_bn16(
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

// =========================================================================
// Compact Slab Structs & Fast SWAR Direct-BF16 Decoders (Zero-Math Exact)
// =========================================================================

struct SlabMX7Lite {
    uint4    sm;
    uint2    e21;
    uint32_t e0;
};

struct SlabPanelV5 {
    uint4 a;
    uint4 b;
};

static __device__ __forceinline__ SlabMX7Lite load_slab_mx7_lite(
    const uint8_t* sp, int lane, bool use_ef, uint64_t pol
) {
    SlabMX7Lite r;
    r.sm  = ld_w_u4(sp + lane * 16, use_ef, pol);
    r.e21 = ld_w_u2(sp + 512 + lane * 8, use_ef, pol);
    r.e0  = ld_w_u32(sp + 768 + lane * 4, use_ef, pol);
    return r;
}

static __device__ __forceinline__ SlabPanelV5 load_slab_panel_v5(
    const uint8_t* sp, int lane, bool is_raw8, bool use_ef, uint64_t pol
) {
    SlabPanelV5 r;
    if (is_raw8) {
        r.a = ld_w_u4(sp + lane * 16, use_ef, pol);
        r.b = ld_w_u4(sp + 512 + lane * 16, use_ef, pol);
    } else {
        r.a = ld_w_u4(sp + lane * 16, use_ef, pol);
        uint2 e21 = ld_w_u2(sp + 512 + lane * 8, use_ef, pol);
        uint32_t e0 = ld_w_u32(sp + 768 + lane * 4, use_ef, pol);
        r.b = make_uint4(e21.x, e21.y, e0, 0u);
    }
    return r;
}

static __device__ __forceinline__ uint32_t build_e4_pack_compat_fast(
    uint32_t sm16, uint32_t e21_8, uint32_t e0_4, uint32_t flip4
) {
    uint32_t t = prmt_b32(sm16, 0u, 0x4140u);
    uint32_t s_lo = (t & 0x000F000Fu) | ((t << 4) & 0x0F000F00u);
    uint32_t sm_b = ((s_lo & 0x08080808u) << 4) | (s_lo & 0x07070707u);
    uint32_t u = (e21_8 | (e21_8 << 12)) & 0x000F000Fu;
    uint32_t e21_b = ((u | (u << 6)) & 0x03030303u) << 4;
    uint32_t e0_b = (e0_4 * 0x01020408u) & 0x08080808u;
    uint32_t e3_b = 0x40404040u ^ ((flip4 * 0x08102040u) & 0x40404040u);
    return sm_b | e21_b | e0_b | e3_b;
}

static __device__ __forceinline__ void e4m3x4_scale_to_bf16x4(
    uint32_t e4_pack, float sc, uint32_t& b0, uint32_t& b1
) {
    uint32_t h0, h1;
    asm volatile(
        "{\n\t"
        ".reg .b16 p0, p1;\n\t"
        "mov.b32 {p0, p1}, %2;\n\t"
        "cvt.rn.f16x2.e4m3x2 %0, p0;\n\t"
        "cvt.rn.f16x2.e4m3x2 %1, p1;\n\t"
        "}\n\t"
        : "=r"(h0), "=r"(h1) : "r"(e4_pack)
    );
    float2 f0 = __half22float2(*reinterpret_cast<const __half2*>(&h0));
    float2 f1 = __half22float2(*reinterpret_cast<const __half2*>(&h1));
    float x0 = mul_rn(f0.x, sc), y0 = mul_rn(f0.y, sc);
    float x1 = mul_rn(f1.x, sc), y1 = mul_rn(f1.y, sc);
    asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(b0) : "f"(x0), "f"(y0));
    asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(b1) : "f"(x1), "f"(y1));
}

static __device__ __forceinline__ void direct_compat_normal_to_bf16x4(
    uint32_t sm16, uint32_t e21_8, uint32_t e0_4, uint32_t flip4,
    uint32_t ws_off2, uint32_t& b0, uint32_t& b1
) {
    const uint32_t q0    = (sm16 & 0xFFu) * 0x00010010u;
    const uint32_t m0    = q0 & 0x00700070u;
    const uint32_t s0    = (q0 << 8) & 0x80008000u;
    const uint32_t e21_0 = ((e21_8 & 0x0Fu) * 0x00400100u) & 0x03000300u;
    const uint32_t e0_0  = ((e0_4 & 0x03u) * 0x00400080u) & 0x00800080u;
    const uint32_t e3_0  = 0x04000400u ^ (((flip4 & 0x03u) * 0x02000400u) & 0x04000400u);
    b0 = ((m0 | e21_0 | e0_0 | e3_0) + ws_off2) | s0;

    const uint32_t q1    = (sm16 >> 8) * 0x00010010u;
    const uint32_t m1    = q1 & 0x00700070u;
    const uint32_t s1    = (q1 << 8) & 0x80008000u;
    const uint32_t e21_1 = ((e21_8 >> 4) * 0x00400100u) & 0x03000300u;
    const uint32_t e0_1  = ((e0_4 >> 2) * 0x00400080u) & 0x00800080u;
    const uint32_t e3_1  = 0x04000400u ^ (((flip4 >> 2) * 0x02000400u) & 0x04000400u);
    b1 = ((m1 | e21_1 | e0_1 | e3_1) + ws_off2) | s1;
}

static __device__ __forceinline__ void e4_pack_normal_to_bf16x4(
    uint32_t e4_pack, uint32_t ws_off2, uint32_t& b0, uint32_t& b1
) {
    const uint32_t u0 = prmt_b32(e4_pack, 0u, 0x4140u);
    const uint32_t u1 = prmt_b32(e4_pack, 0u, 0x4342u);
    b0 = (((u0 << 4) & 0x07F007F0u) + ws_off2) | ((u0 << 8) & 0x80008000u);
    b1 = (((u1 << 4) & 0x07F007F0u) + ws_off2) | ((u1 << 8) & 0x80008000u);
}

static __device__ __forceinline__ void decode_mx7_step_to_bf16x4(
    uint32_t sm16, uint32_t e21_8, uint32_t e0_4, uint32_t flip4,
    uint32_t ws_off2, float w_sc, bool slab_all_normal, bool slab_no_nan,
    uint32_t& b0, uint32_t& b1
) {
    bool fast_ok = slab_all_normal;
    if (__builtin_expect(!fast_ok && slab_no_nan, 0)) {
        const uint32_t c4 = flip4 & (~e0_4);
        if (c4 == 0u) {
            fast_ok = true;
        } else {
            const uint32_t nz21 = (e21_8 | (e21_8 >> 1)) & 0x55u;
            const uint32_t c_sp = (c4 & 1u) | ((c4 & 2u) << 1) | ((c4 & 4u) << 2) | ((c4 & 8u) << 3);
            fast_ok = ((c_sp & (~nz21)) == 0u);
        }
    }
    if (__builtin_expect(fast_ok, 1)) {
        direct_compat_normal_to_bf16x4(sm16, e21_8, e0_4, flip4, ws_off2, b0, b1);
    } else {
        e4m3x4_scale_to_bf16x4(
            build_e4_pack_compat_fast(sm16, e21_8, e0_4, flip4), w_sc, b0, b1
        );
    }
}

static __device__ __forceinline__ void decode_raw8_step_to_bf16x4(
    uint32_t e4_pack, uint32_t ws_off2, float w_sc,
    bool slab_all_normal, bool slab_no_nan,
    uint32_t& b0, uint32_t& b1
) {
    bool fast_ok = slab_all_normal;
    if (__builtin_expect(!fast_ok && slab_no_nan, 0)) {
        const uint32_t e_bits = e4_pack & 0x78787878u;
        fast_ok = (((e_bits - 0x08080808u) & 0x80808080u) == 0u);
    }
    if (__builtin_expect(fast_ok, 1)) {
        e4_pack_normal_to_bf16x4(e4_pack, ws_off2, b0, b1);
    } else {
        e4m3x4_scale_to_bf16x4(e4_pack, w_sc, b0, b1);
    }
}

static __device__ __forceinline__ void mma_m16n8k16_bf16(
    float (&acc)[4],
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)
    );
}

// =========================================================================
// Flip Decoders: Legacy vs v5/v14 Adaptive Hybrid (L2::evict_first aware)
// =========================================================================

static __device__ __forceinline__ uint32_t finish_slab_flip_mask_legacy(
    uint32_t rec, const uint16_t* __restrict__ FLIPS, int32_t& f_cur, int32_t f_end,
    uint32_t sl, int lane, uint32_t* sFlipWarp, bool use_ef, uint64_t pol
) {
    bool in_slab = (rec != 0xFFFFu) && ((rec >> 10) == sl);
    uint32_t m = __ballot_sync(0xFFFFFFFFu, in_slab);
    if (__builtin_expect(m == 0u, 0)) return 0u;
    sFlipWarp[lane] = 0u;
    __syncwarp();
    if (in_slab) {
        const uint32_t s  = (rec >> 7) & 7u;
        const uint32_t tl = (rec >> 2) & 31u;
        const uint32_t j  = rec & 3u;
        atomicOr(&sFlipWarp[tl], 1u << ((s << 2) | j));
    }
    int cnt = __popc(m);
    f_cur += cnt;
    while (__builtin_expect(cnt == 32 && f_cur < f_end, 0)) {
        const int32_t idx = f_cur + lane;
        const uint32_t r2 = (idx < f_end) ? static_cast<uint32_t>(ld_w_u16(FLIPS + idx, use_ef, pol)) : 0xFFFFu;
        const bool in2 = (r2 != 0xFFFFu) && ((r2 >> 10) == sl);
        const uint32_t m2 = __ballot_sync(0xFFFFFFFFu, in2);
        if (m2 == 0u) break;
        if (in2) {
            const uint32_t s  = (r2 >> 7) & 7u;
            const uint32_t tl = (r2 >> 2) & 31u;
            const uint32_t j  = r2 & 3u;
            atomicOr(&sFlipWarp[tl], 1u << ((s << 2) | j));
        }
        cnt = __popc(m2);
        f_cur += cnt;
    }
    __syncwarp();
    return sFlipWarp[lane];
}

static __device__ __forceinline__ uint32_t issue_hybrid_flip_load(
    uint32_t meta,
    const uint16_t* __restrict__ SFLIPS, int32_t s_base,
    const uint32_t* __restrict__ BMASKS, int32_t d_base,
    int lane, bool is_panel_mode, bool use_ef, uint64_t pol
) {
    if (meta & META_DENSE_BIT) {
        if (is_panel_mode) return 0u;
        const int64_t d_idx = static_cast<int64_t>(d_base) + static_cast<int64_t>(meta & META_OFF_MASK);
        return ld_w_u32(BMASKS + (d_idx << 5) + lane, use_ef, pol);
    }
    const uint32_t nf = (meta >> META_NF_SHIFT) & META_NF_MASK;
    if (nf == 0u) return 0u;
    const int64_t s_idx = static_cast<int64_t>(s_base) + static_cast<int64_t>(meta & META_OFF_MASK);
    return (static_cast<uint32_t>(lane) < nf)
        ? static_cast<uint32_t>(ld_w_u16(SFLIPS + s_idx + lane, use_ef, pol))
        : 0u;
}

static __device__ __forceinline__ uint32_t finish_hybrid_flip_mask(
    uint32_t raw_val, uint32_t meta,
    const uint16_t* __restrict__ SFLIPS, int32_t s_base,
    int lane, uint32_t* sFlipWarp, bool is_panel_mode, bool use_ef, uint64_t pol
) {
    if (meta & META_DENSE_BIT) {
        return is_panel_mode ? 0u : raw_val;
    }
    const uint32_t nf = (meta >> META_NF_SHIFT) & META_NF_MASK;
    if (nf == 0u) return 0u;

    sFlipWarp[lane] = 0u;
    __syncwarp();
    if (static_cast<uint32_t>(lane) < nf) {
        atomicOr(&sFlipWarp[raw_val >> 5], 1u << (raw_val & 31u));
    }
    if (__builtin_expect(nf > 32u, 0)) {
        const uint32_t idx2 = static_cast<uint32_t>(lane) + 32u;
        if (idx2 < nf) {
            const int64_t s_idx = static_cast<int64_t>(s_base) + static_cast<int64_t>(meta & META_OFF_MASK);
            const uint32_t r2 = static_cast<uint32_t>(ld_w_u16(SFLIPS + s_idx + idx2, use_ef, pol));
            atomicOr(&sFlipWarp[r2 >> 5], 1u << (r2 & 31u));
        }
    }
    __syncwarp();
    return sFlipWarp[lane];
}

// =========================================================================
// Exact 0-ULP SiTU & MXFP8 Activation Quant-Dequant Helpers
// (Matches Triton activation_ops.py & libprefill_moe_qf_s8.so bit-for-bit!)
// =========================================================================

static __device__ __forceinline__ float sigmoid_approx_exact(float x) {
    float r;
    asm("{ .reg .f32 t; mul.rn.f32 t, %1, 0fBFB8AA3B; "
        "ex2.approx.ftz.f32 t, t; add.rn.f32 t, t, 0f3F800000; "
        "rcp.approx.ftz.f32 %0, t; }"
        : "=f"(r) : "f"(x));
    return r;
}

static __device__ __forceinline__ float softcap_4(float x) {
    float q = mul_rn(x, 0.25f);
    float t;
    asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(q));
    return mul_rn(4.0f, t);
}

static __device__ __forceinline__ float softcap_25(float x) {
    float q = mul_rn(x, INV_25);
    float res = fmaxf(fmaf(q, 25.0f, -x), -FP32_MAX_VAL);
    q = fmaf(-res, INV_25, q);
    float t;
    asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(q));
    return mul_rn(25.0f, t);
}

static __device__ __forceinline__ uint16_t situ_pair_bf16_exact(uint32_t pair, float gamma) {
    float gate = bf16_bits_to_f32(static_cast<uint16_t>(pair & 0xFFFFu));
    float up   = bf16_bits_to_f32(static_cast<uint16_t>(pair >> 16));
    float sig  = sigmoid_approx_exact(gate);
    gate = softcap_4(gate);
    up   = softcap_25(up);
    float y = mul_rn(mul_rn(mul_rn(gate, sig), up), gamma);
    return f32_to_bf16_rn(y);
}

static __device__ __forceinline__ void mxfp8_block32_params(
    const uint4 v[4], float& inv_out, float& a_sc_out
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

    if (exp_u == 255u) inv_out = 0.0f;
    else if (exp_u == 0u) inv_out = __uint_as_float(0x7F000000u);
    else inv_out = rcp_rn(__uint_as_float(exp_u << 23));

    a_sc_out = e8m0_to_f32(static_cast<uint8_t>(exp_u));
}

static __device__ __forceinline__ uint4 mxfp8_quant_dequant_8bf16(
    uint4 vx, float inv, float a_sc
) {
    const uint32_t u[4] = {vx.x, vx.y, vx.z, vx.w};
    uint32_t out_w[4];
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
        const uint32_t u0 = u[2 * h + 0];
        const uint32_t u1 = u[2 * h + 1];
        float f0 = mul_rn(bf16_bits_to_f32(static_cast<uint16_t>(u0 & 0xFFFFu)), inv);
        float f1 = mul_rn(bf16_bits_to_f32(static_cast<uint16_t>(u0 >> 16)),      inv);
        float f2 = mul_rn(bf16_bits_to_f32(static_cast<uint16_t>(u1 & 0xFFFFu)), inv);
        float f3 = mul_rn(bf16_bits_to_f32(static_cast<uint16_t>(u1 >> 16)),      inv);
        uint16_t e_lo, e_hi;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(e_lo) : "f"(f0), "f"(f1));
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(e_hi) : "f"(f2), "f"(f3));
        uint32_t a4 = static_cast<uint32_t>(e_lo) | (static_cast<uint32_t>(e_hi) << 16);
        uint32_t b0, b1;
        e4m3x4_scale_to_bf16x4(a4, a_sc, b0, b1);
        out_w[2 * h + 0] = b0;
        out_w[2 * h + 1] = b1;
    }
    return make_uint4(out_w[0], out_w[1], out_w[2], out_w[3]);
}

static __device__ __forceinline__ void quant_dequant_mxfp8_32_to_sA(
    const uint4 v[4], __nv_bfloat16* dst
) {
    float inv, a_sc;
    mxfp8_block32_params(v, inv, a_sc);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        *reinterpret_cast<uint4*>(dst + i * 8) = mxfp8_quant_dequant_8bf16(v[i], inv, a_sc);
    }
}

// Wave-V14 In-W13 Last-Panel In-Place L2-Hot MXFP8 Block-32 Quant-Dequant:
//   - Executed ONLY by the final finishing panel CTA of tile `t` (1/18 or 1/36 CTAs).
//   - Every 32-element block is 100% independent -> zero shuffles, zero SMEM, 1 single pass at TILE=8!
//   - Strictly L2-coherent (`ld_cg_u4` -> `st_cg_u4` + `__threadfence()`), zero stale L1 risk!
template <int THREADS>
static __device__ __forceinline__ void run_tile_situ_mxfp8_qdeq_inplace_v14(
    __nv_bfloat16* __restrict__ Out,
    int compact_base,
    int live_m,
    int tid
) {
    const int total_blks = live_m * 9;
    for (int blk = tid; blk < total_blks; blk += THREADS) {
        const int lm = blk / 9;
        const int b  = blk - lm * 9;
        __nv_bfloat16* p = Out + static_cast<int64_t>(compact_base + lm) * 288 + b * 32;
        const uint4 v[4] = {
            ld_cg_u4(p + 0), ld_cg_u4(p + 8), ld_cg_u4(p + 16), ld_cg_u4(p + 24)
        };
        float inv, a_sc;
        mxfp8_block32_params(v, inv, a_sc);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            st_cg_u4(p + i * 8, mxfp8_quant_dequant_8bf16(v[i], inv, a_sc));
        }
    }
    __threadfence();
}

// =========================================================================
// Self-Contained GPU Packer & Zero-Raw-Weight v5 Transcoders (0 B Spills!)
// =========================================================================

template <bool FROM_PANEL>
static __device__ __forceinline__ uint8_t read_w_elem(
    const uint8_t* W, int e, int row, int kcol, int N, int K, int pb, int kb
) {
    if constexpr (!FROM_PANEL) {
        return W[(static_cast<int64_t>(e) * N + row) * K + kcol];
    } else {
        int n_panels = N / pb;
        int k_steps  = K / kb;
        int64_t p_idx = (static_cast<int64_t>(e) * n_panels + (row / pb)) * k_steps + (kcol / kb);
        return W[(p_idx * pb + (row % pb)) * kb + (kcol % kb)];
    }
}

template <bool FROM_PANEL>
static __device__ __forceinline__ uint8_t read_ws_elem(
    const uint8_t* WS, int e, int row, int kg_idx, int N, int KG, int pb, int kbg
) {
    if constexpr (!FROM_PANEL) {
        return WS[(static_cast<int64_t>(e) * N + row) * KG + kg_idx];
    } else {
        int n_panels = N / pb;
        int k_steps  = KG / kbg;
        int64_t p_idx = (static_cast<int64_t>(e) * n_panels + (row / pb)) * k_steps + (kg_idx / kbg);
        return WS[(p_idx * kbg + (kg_idx % kbg)) * pb + (row % pb)];
    }
}

template <int PROJ, bool FROM_PANEL>
__global__ __launch_bounds__(32, 1)
void mx7_c1_count_flips_kernel(
    const uint8_t* __restrict__ W,
    int32_t* __restrict__ counts,
    int E, int N, int K, int src_pb, int src_kb
) {
    constexpr int WARPS  = (PROJ == 0) ? 4 : 2;
    constexpr int SUB_N  = (PROJ == 0) ? 1 : 4;
    constexpr int BN     = WARPS * SUB_N * 8;
    constexpr int SLAB_K = (PROJ == 0) ? 128 : 32;

    const int n_panels = N / BN;
    const int64_t warp_uid = static_cast<int64_t>(blockIdx.x);
    if (warp_uid >= static_cast<int64_t>(E) * n_panels * WARPS) return;

    const int lane = threadIdx.x;
    const int g = lane >> 2;
    const int lt = lane & 3;
    const int warp = static_cast<int>(warp_uid % WARPS);
    const int64_t panel_uid = warp_uid / WARPS;
    const int nb = static_cast<int>(panel_uid % n_panels);
    const int e = static_cast<int>(panel_uid / n_panels);
    const int n_slabs = K / SLAB_K;

    int c = 0;
    for (int sl = 0; sl < n_slabs; ++sl) {
        #pragma unroll
        for (int s = 0; s < 8; ++s) {
            const int ks2 = s & 1;
            const int n_base = (PROJ == 0)
                ? (nb * BN + warp * 8)
                : (nb * BN + (warp * SUB_N + (s >> 1)) * 8);
            const int k32_base = (PROJ == 0) ? (sl * 128 + (s >> 1) * 32) : (sl * 32);
            const int row = n_base + g;
            const int k0 = k32_base + 4 * lt + ks2 * 2;
            uint8_t v0 = read_w_elem<FROM_PANEL>(W, e, row, k0,      N, K, src_pb, src_kb);
            uint8_t v1 = read_w_elem<FROM_PANEL>(W, e, row, k0 + 1,  N, K, src_pb, src_kb);
            uint8_t v2 = read_w_elem<FROM_PANEL>(W, e, row, k0 + 16, N, K, src_pb, src_kb);
            uint8_t v3 = read_w_elem<FROM_PANEL>(W, e, row, k0 + 17, N, K, src_pb, src_kb);
            c += ((v0 & 0x40) == 0) + ((v1 & 0x40) == 0) + ((v2 & 0x40) == 0) + ((v3 & 0x40) == 0);
        }
    }
    for (int off = 16; off > 0; off >>= 1) {
        c += __shfl_down_sync(0xFFFFFFFFu, c, off);
    }
    if (lane == 0) counts[warp_uid] = c;
}

template <int PROJ, bool FROM_PANEL>
__global__ __launch_bounds__(32, 1)
void mx7_c1_pack_kernel(
    const uint8_t* __restrict__ W,
    const uint8_t* __restrict__ WS,
    uint8_t* __restrict__ SLABS,
    uint4* __restrict__ WS3,
    uint16_t* __restrict__ FLIPS,
    const int32_t* __restrict__ F_OFF,
    int E, int N, int K, uint32_t base_ws, int src_pb, int src_kb
) {
    constexpr int WARPS  = (PROJ == 0) ? 4 : 2;
    constexpr int SUB_N  = (PROJ == 0) ? 1 : 4;
    constexpr int BN     = WARPS * SUB_N * 8;
    constexpr int SLAB_K = (PROJ == 0) ? 128 : 32;

    const int n_panels = N / BN;
    const int64_t warp_uid = static_cast<int64_t>(blockIdx.x);
    if (warp_uid >= static_cast<int64_t>(E) * n_panels * WARPS) return;

    const int lane = threadIdx.x;
    const int g = lane >> 2;
    const int lt = lane & 3;
    const int warp = static_cast<int>(warp_uid % WARPS);
    const int64_t panel_uid = warp_uid / WARPS;
    const int nb = static_cast<int>(panel_uid % n_panels);
    const int e = static_cast<int>(panel_uid / n_panels);
    const int KG = K >> 5;
    const int n_slabs = K / SLAB_K;
    const int src_kbg = src_kb >> 5;

    int32_t f_pos = F_OFF[warp_uid];

    for (int sl = 0; sl < n_slabs; ++sl) {
        uint16_t sm_arr[8];
        uint8_t  e21_arr[8];
        uint32_t e0_reg = 0u;
        uint32_t ws_words[4] = {0u, 0u, 0u, 0u};

        #pragma unroll
        for (int s = 0; s < 8; ++s) {
            const int ks2 = s & 1;
            const int n_base = (PROJ == 0)
                ? (nb * BN + warp * 8)
                : (nb * BN + (warp * SUB_N + (s >> 1)) * 8);
            const int k32_base = (PROJ == 0) ? (sl * 128 + (s >> 1) * 32) : (sl * 32);

            if (ks2 == 0 && lane == 0) {
                const int kg_idx = k32_base >> 5;
                uint32_t w_word = 0u;
                #pragma unroll
                for (int gg = 0; gg < 8; ++gg) {
                    uint32_t v = static_cast<uint32_t>(
                        read_ws_elem<FROM_PANEL>(WS, e, n_base + gg, kg_idx, N, KG, src_pb, src_kbg)
                    ) - base_ws;
                    w_word |= (v & 7u) << (gg * 3);
                }
                ws_words[s >> 1] = w_word;
            }

            const int row = n_base + g;
            const int k0 = k32_base + 4 * lt + ks2 * 2;
            const uint8_t vals[4] = {
                read_w_elem<FROM_PANEL>(W, e, row, k0,      N, K, src_pb, src_kb),
                read_w_elem<FROM_PANEL>(W, e, row, k0 + 1,  N, K, src_pb, src_kb),
                read_w_elem<FROM_PANEL>(W, e, row, k0 + 16, N, K, src_pb, src_kb),
                read_w_elem<FROM_PANEL>(W, e, row, k0 + 17, N, K, src_pb, src_kb),
            };

            uint32_t sm16 = 0u, e21_8 = 0u, e0_4 = 0u;
            uint32_t flip_mask = 0u;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                uint32_t b = vals[j];
                if ((b & 0x40u) == 0u) flip_mask |= (1u << j);
                uint32_t sm4 = (((b >> 7) & 1u) << 3) | (b & 7u);
                sm16  |= sm4 << (j * 4);
                e21_8 |= ((b >> 4) & 3u) << (j * 2);
                e0_4  |= ((b >> 3) & 1u) << j;
            }
            sm_arr[s] = static_cast<uint16_t>(sm16);
            e21_arr[s] = static_cast<uint8_t>(e21_8);
            e0_reg |= (e0_4 << (s * 4));

            const uint32_t step_id = static_cast<uint32_t>((sl << 3) | s);
            const uint32_t any_flips = __ballot_sync(0xFFFFFFFFu, flip_mask != 0u);
            if (any_flips != 0u) {
                int my_cnt = __popc(flip_mask);
                int prefix = my_cnt;
                #pragma unroll
                for (int off = 1; off < 32; off <<= 1) {
                    int n_val = __shfl_up_sync(0xFFFFFFFFu, prefix, off);
                    if (lane >= off) prefix += n_val;
                }
                int total_step_flips = __shfl_sync(0xFFFFFFFFu, prefix, 31);
                int my_off = f_pos + (prefix - my_cnt);
                if (flip_mask != 0u) {
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        if (flip_mask & (1u << j)) {
                            FLIPS[my_off++] = static_cast<uint16_t>((step_id << 7) | (lane << 2) | j);
                        }
                    }
                }
                f_pos += total_step_flips;
            }
        }

        const int64_t slab_idx = warp_uid * n_slabs + sl;
        uint8_t* sp = SLABS + slab_idx * MX7_SLAB_BYTES;
        uint4 sm_v;
        sm_v.x = static_cast<uint32_t>(sm_arr[0]) | (static_cast<uint32_t>(sm_arr[1]) << 16);
        sm_v.y = static_cast<uint32_t>(sm_arr[2]) | (static_cast<uint32_t>(sm_arr[3]) << 16);
        sm_v.z = static_cast<uint32_t>(sm_arr[4]) | (static_cast<uint32_t>(sm_arr[5]) << 16);
        sm_v.w = static_cast<uint32_t>(sm_arr[6]) | (static_cast<uint32_t>(sm_arr[7]) << 16);
        *reinterpret_cast<uint4*>(sp + lane * 16) = sm_v;

        uint2 e21_v;
        e21_v.x = static_cast<uint32_t>(e21_arr[0]) | (static_cast<uint32_t>(e21_arr[1]) << 8)
                | (static_cast<uint32_t>(e21_arr[2]) << 16) | (static_cast<uint32_t>(e21_arr[3]) << 24);
        e21_v.y = static_cast<uint32_t>(e21_arr[4]) | (static_cast<uint32_t>(e21_arr[5]) << 8)
                | (static_cast<uint32_t>(e21_arr[6]) << 16) | (static_cast<uint32_t>(e21_arr[7]) << 24);
        *reinterpret_cast<uint2*>(sp + 512 + lane * 8) = e21_v;
        *reinterpret_cast<uint32_t*>(sp + 768 + lane * 4) = e0_reg;

        if (lane == 0) {
            WS3[slab_idx] = make_uint4(ws_words[0], ws_words[1], ws_words[2], ws_words[3]);
        }
    }
}

static __device__ __forceinline__ uint32_t scan_legacy_slab_flips(
    const uint16_t* __restrict__ FLIPS, int32_t& f_cur, int32_t f_end,
    uint32_t sl, int lane, uint32_t* sFlipWarp, uint32_t& fmask_out
) {
    sFlipWarp[lane] = 0u;
    __syncwarp();
    uint32_t total = 0u;
    while (f_cur < f_end) {
        const int32_t idx = f_cur + lane;
        const uint32_t r = (idx < f_end) ? static_cast<uint32_t>(FLIPS[idx]) : 0xFFFFu;
        const bool in_sl = (r != 0xFFFFu) && ((r >> 10) == sl);
        const uint32_t m = __ballot_sync(0xFFFFFFFFu, in_sl);
        if (m == 0u) break;
        if (in_sl) {
            const uint32_t s  = (r >> 7) & 7u;
            const uint32_t tl = (r >> 2) & 31u;
            const uint32_t j  = r & 3u;
            atomicOr(&sFlipWarp[tl], 1u << ((s << 2) | j));
        }
        const uint32_t cnt = static_cast<uint32_t>(__popc(m));
        total += cnt;
        f_cur += static_cast<int32_t>(cnt);
        if (cnt < 32u) break;
    }
    __syncwarp();
    fmask_out = sFlipWarp[lane];
    return total;
}

static __device__ __forceinline__ void classify_and_reconstruct_slab(
    const SlabMX7Lite& slb, uint32_t fmask,
    bool& slab_all_normal, bool& slab_has_nan,
    uint4& raw0, uint4& raw1
) {
    uint32_t e4[8];
    bool any_e0 = false;
    bool any_nan = false;
    #pragma unroll
    for (int s = 0; s < 8; ++s) {
        const uint32_t sm_w = (s < 2) ? slb.sm.x : ((s < 4) ? slb.sm.y : ((s < 6) ? slb.sm.z : slb.sm.w));
        const uint32_t sm16 = (s & 1) ? (sm_w >> 16) : (sm_w & 0xFFFFu);
        const uint32_t e21_w = (s < 4) ? slb.e21.x : slb.e21.y;
        const uint32_t e21_8 = (e21_w >> ((s & 3) * 8)) & 0xFFu;
        const uint32_t p = build_e4_pack_compat_fast(
            sm16, e21_8, (slb.e0 >> (s * 4)) & 0xFu, (fmask >> (s * 4)) & 0xFu
        );
        e4[s] = p;
        const uint32_t eb = p & 0x78787878u;
        if (((eb - 0x08080808u) & 0x80808080u) != 0u) any_e0 = true;
        if ((((p & 0x7F7F7F7Fu) + 0x01010101u) & 0x80808080u) != 0u) any_nan = true;
    }
    raw0 = make_uint4(e4[0], e4[1], e4[2], e4[3]);
    raw1 = make_uint4(e4[4], e4[5], e4[6], e4[7]);
    const bool warp_e0  = (__ballot_sync(0xFFFFFFFFu, any_e0) != 0u);
    const bool warp_nan = (__ballot_sync(0xFFFFFFFFu, any_nan) != 0u);
    slab_has_nan    = warp_nan;
    slab_all_normal = (!warp_e0) && (!warp_nan);
}

__global__ __launch_bounds__(32, 1)
void mx7_c1_v5_count_hybrid_from_mx7_kernel(
    const uint16_t* __restrict__ FLIPS,
    const int32_t* __restrict__ F_OFF,
    int32_t* __restrict__ sparse_counts,
    int32_t* __restrict__ dense_counts,
    int total_warps, int n_slabs, int flip_thresh
) {
    const int64_t warp_uid = static_cast<int64_t>(blockIdx.x);
    if (warp_uid >= total_warps) return;
    const int lane = threadIdx.x;
    int32_t f_cur = F_OFF[warp_uid];
    const int32_t f_end = F_OFF[warp_uid + 1];
    const uint32_t thr = static_cast<uint32_t>(flip_thresh);

    int32_t s_cnt = 0, d_cnt = 0;
    for (int sl = 0; sl < n_slabs; ++sl) {
        uint32_t nf = 0u;
        while (f_cur < f_end) {
            const int32_t idx = f_cur + lane;
            const uint32_t r = (idx < f_end) ? static_cast<uint32_t>(FLIPS[idx]) : 0xFFFFu;
            const bool in_sl = (r != 0xFFFFu) && ((r >> 10) == static_cast<uint32_t>(sl));
            const uint32_t m = __ballot_sync(0xFFFFFFFFu, in_sl);
            if (m == 0u) break;
            const uint32_t cnt = static_cast<uint32_t>(__popc(m));
            nf += cnt;
            f_cur += static_cast<int32_t>(cnt);
            if (cnt < 32u) break;
        }
        if (nf > thr) d_cnt += 1;
        else s_cnt += static_cast<int32_t>(nf);
    }
    if (lane == 0) {
        sparse_counts[warp_uid] = s_cnt;
        dense_counts[warp_uid]  = d_cnt;
    }
}

__global__ __launch_bounds__(32, 1)
void mx7_c1_v5_build_hybrid_from_mx7_kernel(
    const uint8_t* __restrict__ SLABS,
    const uint16_t* __restrict__ FLIPS,
    const int32_t* __restrict__ F_OFF,
    const int32_t* __restrict__ S_OFF,
    const int32_t* __restrict__ D_OFF,
    uint32_t* __restrict__ SLAB_META,
    uint16_t* __restrict__ SFLIPS,
    uint32_t* __restrict__ BMASKS,
    int total_warps, int n_slabs, int flip_thresh
) {
    const int64_t warp_uid = static_cast<int64_t>(blockIdx.x);
    if (warp_uid >= total_warps) return;
    const int lane = threadIdx.x;
    __shared__ uint32_t sFlip[32];

    int32_t f_cur = F_OFF[warp_uid];
    const int32_t f_end = F_OFF[warp_uid + 1];
    const int64_t s_base = static_cast<int64_t>(S_OFF[warp_uid]);
    const int64_t d_base = static_cast<int64_t>(D_OFF[warp_uid]);
    const uint32_t thr = static_cast<uint32_t>(flip_thresh);

    uint32_t s_rel = 0u, d_rel = 0u;
    for (int sl = 0; sl < n_slabs; ++sl) {
        const int32_t f_sl_start = f_cur;
        uint32_t fmask = 0u;
        const uint32_t nf = scan_legacy_slab_flips(
            FLIPS, f_cur, f_end, static_cast<uint32_t>(sl), lane, sFlip, fmask
        );
        const int64_t slab_idx = warp_uid * n_slabs + sl;
        const SlabMX7Lite slb = load_slab_mx7_lite(SLABS + slab_idx * MX7_SLAB_BYTES, lane, false, 0ULL);
        bool all_normal = false, has_nan = false;
        uint4 raw0, raw1;
        classify_and_reconstruct_slab(slb, fmask, all_normal, has_nan, raw0, raw1);
        (void)raw0; (void)raw1;

        const uint32_t norm_flags = (all_normal ? META_ALL_NORMAL_BIT : 0u)
                                  | (has_nan    ? META_HAS_NAN_BIT    : 0u);
        if (nf > thr) {
            BMASKS[((d_base + static_cast<int64_t>(d_rel)) << 5) + lane] = fmask;
            if (lane == 0) {
                SLAB_META[slab_idx] = META_DENSE_BIT | norm_flags | (d_rel & META_OFF_MASK);
            }
            d_rel += 1u;
        } else {
            for (uint32_t k = static_cast<uint32_t>(lane); k < nf; k += 32u) {
                const uint32_t r = static_cast<uint32_t>(FLIPS[f_sl_start + static_cast<int32_t>(k)]);
                const uint32_t s  = (r >> 7) & 7u;
                const uint32_t tl = (r >> 2) & 31u;
                const uint32_t j  = r & 3u;
                SFLIPS[s_base + static_cast<int64_t>(s_rel + k)] =
                    static_cast<uint16_t>((tl << 5) | (s << 2) | j);
            }
            if (lane == 0) {
                SLAB_META[slab_idx] = norm_flags | ((nf & META_NF_MASK) << META_NF_SHIFT) | (s_rel & META_OFF_MASK);
            }
            s_rel += nf;
        }
    }
}

__global__ __launch_bounds__(128, 1)
void mx7_c1_v5_panel_measure_from_mx7_kernel(
    const uint16_t* __restrict__ FLIPS,
    const int32_t* __restrict__ F_OFF,
    int32_t* __restrict__ panel_blocks_out,
    int32_t* __restrict__ sparse_counts,
    int total_panels, int n_slabs, int ws3_hdr_blocks, int flip_thresh
) {
    const int64_t panel_uid = static_cast<int64_t>(blockIdx.x);
    if (panel_uid >= total_panels) return;
    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int64_t warp_uid = panel_uid * 4 + warp;

    __shared__ uint32_t sNf[4];

    int32_t f_cur = F_OFF[warp_uid];
    const int32_t f_end = F_OFF[warp_uid + 1];
    const uint32_t thr = static_cast<uint32_t>(flip_thresh);

    int32_t s_cnt = 0, body_blocks = 0;
    for (int sl = 0; sl < n_slabs; ++sl) {
        uint32_t nf = 0u;
        while (f_cur < f_end) {
            const int32_t idx = f_cur + lane;
            const uint32_t r = (idx < f_end) ? static_cast<uint32_t>(FLIPS[idx]) : 0xFFFFu;
            const bool in_sl = (r != 0xFFFFu) && ((r >> 10) == static_cast<uint32_t>(sl));
            const uint32_t m = __ballot_sync(0xFFFFFFFFu, in_sl);
            if (m == 0u) break;
            const uint32_t cnt = static_cast<uint32_t>(__popc(m));
            nf += cnt;
            f_cur += static_cast<int32_t>(cnt);
            if (cnt < 32u) break;
        }
        if (lane == 0) sNf[warp] = nf;
        __syncthreads();
        const bool is_raw8 = (sNf[0] > thr) || (sNf[1] > thr) || (sNf[2] > thr) || (sNf[3] > thr);
        if (is_raw8) {
            body_blocks += 32;
        } else {
            body_blocks += 28;
            s_cnt += static_cast<int32_t>(nf);
        }
        __syncthreads();
    }
    if (tid == 0) panel_blocks_out[panel_uid] = ws3_hdr_blocks + body_blocks;
    if (lane == 0) sparse_counts[warp_uid] = s_cnt;
}

__global__ __launch_bounds__(128, 1)
void mx7_c1_v5_panel_build_from_mx7_kernel(
    const uint8_t* __restrict__ SLABS,
    const uint4* __restrict__ WS3,
    const uint16_t* __restrict__ FLIPS,
    const int32_t* __restrict__ F_OFF,
    const int32_t* __restrict__ PANEL_OFF,
    const int32_t* __restrict__ S_OFF,
    uint8_t* __restrict__ PANEL_POOL,
    uint32_t* __restrict__ SLAB_META,
    uint16_t* __restrict__ SFLIPS,
    int total_panels, int n_slabs, int ws3_hdr_blocks, int ws3_warp_u32, int flip_thresh
) {
    const int64_t panel_uid = static_cast<int64_t>(blockIdx.x);
    if (panel_uid >= total_panels) return;
    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int64_t warp_uid = panel_uid * 4 + warp;

    __shared__ uint32_t sFlip[4][32];
    __shared__ uint32_t sNf[4];

    uint8_t* panel_base = PANEL_POOL + static_cast<int64_t>(PANEL_OFF[panel_uid]) * 128;
    uint32_t* hdr_w = reinterpret_cast<uint32_t*>(panel_base) + warp * ws3_warp_u32;
    const uint32_t* src_ws = reinterpret_cast<const uint32_t*>(WS3 + warp_uid * n_slabs);
    const int ws_words_live = n_slabs * 4;
    for (int i = lane; i < ws3_warp_u32; i += 32) {
        hdr_w[i] = (i < ws_words_live) ? src_ws[i] : 0u;
    }

    int32_t f_cur = F_OFF[warp_uid];
    const int32_t f_end = F_OFF[warp_uid + 1];
    const int64_t s_base = static_cast<int64_t>(S_OFF[warp_uid]);
    const uint32_t thr = static_cast<uint32_t>(flip_thresh);

    uint8_t* step_ptr = panel_base + static_cast<int64_t>(ws3_hdr_blocks) * 128;
    uint32_t s_rel = 0u;

    for (int sl = 0; sl < n_slabs; ++sl) {
        const int32_t f_sl_start = f_cur;
        uint32_t fmask = 0u;
        const uint32_t nf = scan_legacy_slab_flips(
            FLIPS, f_cur, f_end, static_cast<uint32_t>(sl), lane, sFlip[warp], fmask
        );
        if (lane == 0) sNf[warp] = nf;
        __syncthreads();
        const bool is_raw8 = (sNf[0] > thr) || (sNf[1] > thr) || (sNf[2] > thr) || (sNf[3] > thr);

        const int64_t slab_idx = warp_uid * n_slabs + sl;
        const SlabMX7Lite slb = load_slab_mx7_lite(SLABS + slab_idx * MX7_SLAB_BYTES, lane, false, 0ULL);
        bool all_normal = false, has_nan = false;
        uint4 raw0, raw1;
        classify_and_reconstruct_slab(slb, fmask, all_normal, has_nan, raw0, raw1);

        const uint32_t norm_flags = (all_normal ? META_ALL_NORMAL_BIT : 0u)
                                  | (has_nan    ? META_HAS_NAN_BIT    : 0u);
        if (is_raw8) {
            uint8_t* dst = step_ptr + warp * MX7_RAW8_SLAB_BYTES;
            *reinterpret_cast<uint4*>(dst + lane * 16)       = raw0;
            *reinterpret_cast<uint4*>(dst + 512 + lane * 16) = raw1;
            if (lane == 0) {
                SLAB_META[slab_idx] = META_DENSE_BIT | norm_flags;
            }
            step_ptr += MX7_PANEL_R8_BYTES;
        } else {
            uint8_t* dst = step_ptr + warp * MX7_SLAB_BYTES;
            *reinterpret_cast<uint4*>(dst + lane * 16)         = slb.sm;
            *reinterpret_cast<uint2*>(dst + 512 + lane * 8)    = slb.e21;
            *reinterpret_cast<uint32_t*>(dst + 768 + lane * 4) = slb.e0;
            for (uint32_t k = static_cast<uint32_t>(lane); k < nf; k += 32u) {
                const uint32_t r = static_cast<uint32_t>(FLIPS[f_sl_start + static_cast<int32_t>(k)]);
                const uint32_t s  = (r >> 7) & 7u;
                const uint32_t tl = (r >> 2) & 31u;
                const uint32_t j  = r & 3u;
                SFLIPS[s_base + static_cast<int64_t>(s_rel + k)] =
                    static_cast<uint16_t>((tl << 5) | (s << 2) | j);
            }
            if (lane == 0) {
                SLAB_META[slab_idx] = norm_flags | ((nf & META_NF_MASK) << META_NF_SHIFT) | (s_rel & META_OFF_MASK);
            }
            s_rel += nf;
            step_ptr += MX7_PANEL_MX7_BYTES;
        }
        __syncthreads();
    }
}

// =========================================================================
// Kernel 1 (C=1..2 v14): Adaptive BN=16 / BN=32 MX_W13 (N=576, K=4608, TILE=8|16)
//   - Zero-math register liveness trims (on-the-fly sm/e21 extraction, reload Tiles[t] at epilogue)
//   - Zero-fill & dead-LDS elimination (`row0_live`/`row8_live` predication -> literal 0-ULP!)
//   - In-W13 last-panel in-place L2-hot MXFP8 quant-dequant (`FUSE_QDEQ` via `TileQdeqDone[t]`)
//   - Wave-adaptive GPU gating (`gate_mode == 2` -> `prefer_w13_bn16`) + `L2::evict_first` weights!
// =========================================================================
template <
    int V5_MODE, int BN, int TILE, int MIN_CTA, bool USE_PDL,
    bool ZERO_SCATTER, bool FUSE_SITU, bool FUSE_QDEQ
>
__global__ __launch_bounds__((BN / 8) * 32, MIN_CTA)
void mx7_w13_c1_v16_kernel(
    const uint8_t* __restrict__ A,
    const uint8_t* __restrict__ AS,
    const uint8_t* __restrict__ SLABS,
    const uint4* __restrict__ WS3,
    const uint16_t* __restrict__ FLIPS,
    const int32_t* __restrict__ F_OFF,
    const uint32_t* __restrict__ SLAB_META,
    const uint32_t* __restrict__ BMASKS,
    const int32_t* __restrict__ AUX_OFF,
    const float* __restrict__ Ones,
    const int32_t* __restrict__ Tiles,
    const int32_t* __restrict__ TileMask,
    const int32_t* __restrict__ TileCompactStart,
    const float* __restrict__ Gammas,
    __nv_bfloat16* __restrict__ Out,
    uint32_t* __restrict__ TileQdeqDone,
    uint32_t* __restrict__ WorkQueue,
    int N, int K, int MAX_CTAS, uint32_t base_ws,
    int gate_mode, int u_min_or_stride16, int u_max_or_stride32, int margin_pct,
    int pdl_wait_mode, int trig_dep, int evict_first, int lpt_mode
) {
    static_assert(BN == 16 || BN == 32, "MX_W13 BN must be 16 or 32");
    static_assert(TILE == 8 || TILE == 16, "C1 MX_W13 TILE must be 8 or 16");
    (void)TileQdeqDone;

    if constexpr (USE_PDL) {
        if (pdl_wait_mode == 1) {
            asm volatile("griddepcontrol.wait;" ::: "memory");
        }
    }

    constexpr int WARPS = BN / 8;
    constexpr int THREADS = WARPS * 32;
    constexpr int SLAB_K = 128;
    constexpr int BK_SMEM = 256;
    constexpr int SLABS_PER_BLK = 2;
    constexpr int M_ROWS = TILE;
    constexpr int SA_STRIDE = BK_SMEM + 16;  // 272 bf16 -> 0-way LDS.64!
    constexpr int N_PANELS32 = 18;
    constexpr int N_PANELS = (BN == 16) ? 36 : 18;

    const int n_active = Tiles[2 * MAX_CTAS];
    bool gated_out = false;
    if (gate_mode == 1) {
        gated_out = (n_active < u_min_or_stride16 || n_active > u_max_or_stride32);
    } else if (gate_mode == 2) {
        const bool want16 = prefer_w13_bn16(n_active, u_min_or_stride16, u_max_or_stride32, margin_pct);
        gated_out = (BN == 16) ? (!want16) : want16;
    }
    if (gated_out) {
        if constexpr (USE_PDL) {
            if (pdl_wait_mode == 2) asm volatile("griddepcontrol.wait;" ::: "memory");
            if (trig_dep) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
        return;
    }

    const int n_total = n_active * N_PANELS;
    const int stride = static_cast<int>(gridDim.x);
    const int bidx   = static_cast<int>(blockIdx.x);

    if (n_active <= 0) {
        if constexpr (USE_PDL) {
            if (pdl_wait_mode == 2) asm volatile("griddepcontrol.wait;" ::: "memory");
            if (trig_dep) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
        return;
    }

    const bool use_dyn_q = (lpt_mode >= 2) && (WorkQueue != nullptr) && (n_total > stride);
    if (!use_dyn_q && bidx >= n_total) {
        if constexpr (USE_PDL) {
            if (pdl_wait_mode == 2) asm volatile("griddepcontrol.wait;" ::: "memory");
            if (trig_dep) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
        return;
    }

    const int n_slabs = K / SLAB_K;   // 36
    const int n_blks  = K / BK_SMEM;  // 18
    const int KG      = K >> 5;       // 144
    const bool fast_ws = (base_ws >= 8u && base_ws <= 238u);
    const bool use_ef = (evict_first != 0);
    const uint64_t l2_pol = use_ef ? make_l2_evict_first_policy() : 0ULL;

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g    = lane >> 2;
    const int lt   = lane & 3;
    const int x0_0 = g * SA_STRIDE + 4 * lt;
    const int x0_1 = (g + 8) * SA_STRIDE + 4 * lt;

    constexpr int MAX_LPT_TILES = 256;
    alignas(16) __shared__ uint32_t sFlip[WARPS][32];
    alignas(16) __shared__ __nv_bfloat16 sA[2][M_ROWS * SA_STRIDE];
    alignas(16) __shared__ int16_t sTileOrder[MAX_LPT_TILES];
    __shared__ int32_t sTileLastQdeq;
    __shared__ int32_t sDynIdx;

    const bool use_lpt = (lpt_mode != 0) && (n_active > 1) && (n_active <= MAX_LPT_TILES);
    if (use_lpt) {
        build_lpt_tile_order<TILE, MAX_LPT_TILES>(Tiles, MAX_CTAS, n_active, sTileOrder, warp, lane);
        __syncthreads();
    }
    auto map_tile = [&](int lt_idx) -> int {
        return use_lpt ? static_cast<int>(sTileOrder[lt_idx]) : lt_idx;
    };

    bool trig_fired = false;

    int idx = bidx;
    bool is_first_tile = true;
    while (idx < n_total) {
        const int log_t = idx / N_PANELS;
        const int nb = idx - log_t * N_PANELS;
        const int t  = map_tile(log_t);
        bool next_claimed = false;
        int nidx = idx + stride;
        auto ensure_next = [&]() {
            if (!next_claimed) {
                if (use_dyn_q) {
                    if (tid == 0) {
                        sDynIdx = stride + static_cast<int>(atom_acq_rel_gpu_add(&WorkQueue[0], 1u));
                    }
                    __syncthreads();
                    nidx = sDynIdx;
                } else {
                    nidx = idx + stride;
                }
                next_claimed = true;
            }
        };

        const int e_i32 = Tiles[t];
        const int end_row = Tiles[MAX_CTAS + t];
        int live_m = end_row - t * TILE;
        if (live_m > TILE) live_m = TILE;
        if (live_m < 0) live_m = 0;

        const int nb32 = (BN == 16) ? (nb >> 1) : nb;
        const int wp   = (BN == 16) ? (((nb & 1) << 1) + warp) : warp;
        const int64_t panel_uid = static_cast<int64_t>(e_i32) * N_PANELS32 + nb32;
        const int64_t warp_uid  = panel_uid * 4 + wp;

        const uint8_t* slab_base = nullptr;
        const uint32_t* ws_u32_base = nullptr;
        if constexpr (V5_MODE == 2) {
            const uint8_t* pbase = SLABS + static_cast<int64_t>(AUX_OFF[panel_uid]) * 128;
            ws_u32_base = reinterpret_cast<const uint32_t*>(pbase) + wp * W13_WS3_WARP_U32;
            slab_base   = pbase + static_cast<int64_t>(W13_WS3_HDR_BLOCKS) * 128;
        } else {
            slab_base   = SLABS + warp_uid * n_slabs * MX7_SLAB_BYTES;
            ws_u32_base = reinterpret_cast<const uint32_t*>(WS3 + warp_uid * n_slabs);
        }

        int32_t f_cur = F_OFF[warp_uid];
        const int32_t f_end  = (V5_MODE == 0) ? F_OFF[warp_uid + 1] : 0;
        const int32_t d_base = (V5_MODE == 1) ? AUX_OFF[warp_uid] : 0;

        if constexpr (USE_PDL) {
            if (pdl_wait_mode == 2 && is_first_tile) {
                if constexpr (V5_MODE == 2) {
                    if (lane < 7) prefetch_l2(slab_base + wp * MX7_SLAB_BYTES + lane * 128);
                    else if (lane < 12) prefetch_l2(ws_u32_base + (lane - 7) * 32);
                    else if (lane < 14) prefetch_l2(SLAB_META + warp_uid * n_slabs + (lane - 12) * 32);
                } else {
                    if (lane < 7) prefetch_l2(slab_base + lane * 128);
                    else if (lane < 14) prefetch_l2(slab_base + MX7_SLAB_BYTES + (lane - 7) * 128);
                    else if (lane == 14) prefetch_l2(ws_u32_base);
                    else if (lane == 15) {
                        if constexpr (V5_MODE == 0) {
                            if (f_cur < f_end) prefetch_l2(FLIPS + f_cur);
                        } else {
                            prefetch_l2(SLAB_META + warp_uid * n_slabs);
                        }
                    }
                }
                asm volatile("griddepcontrol.wait;" ::: "memory");
            }
        }

        if (live_m > 0) {
            const uint32_t tmask = ZERO_SCATTER ? static_cast<uint32_t>(TileMask[t]) : 0u;

            uint32_t meta0 = 0u, meta1 = 0u;
            if constexpr (V5_MODE != 0) {
                const uint32_t* mp = SLAB_META + warp_uid * n_slabs;
                meta0 = (lane < n_slabs) ? ld_w_u32(mp + lane, use_ef, l2_pol) : 0u;
                meta1 = (lane + 32 < n_slabs) ? ld_w_u32(mp + 32 + lane, use_ef, l2_pol) : 0u;
            }
            const int ws_tot_u32 = n_slabs * 4;
            uint32_t ws_cur = (lane < ws_tot_u32) ? ld_w_u32(ws_u32_base + lane, use_ef, l2_pol) : 0u;
            uint32_t ws_nxt = 0u;

            auto load_a_blk = [&](int blk, __nv_bfloat16* dst) {
                constexpr int Q_PER_ROW = BK_SMEM >> 2;  // 64
                const int total_q = live_m * Q_PER_ROW;
                const int k0 = blk * BK_SMEM;
                const int rbase = t * TILE;
                for (int aidx = tid; aidx < total_q; aidx += THREADS) {
                    const int lm = aidx >> 6;
                    const int qk = aidx & 63;
                    const int64_t row = static_cast<int64_t>(rbase + lm);
                    const int64_t a_row = ZERO_SCATTER
                        ? static_cast<int64_t>(fns_u32(tmask, 0u, lm + 1))
                        : row;
                    const int kcol = k0 + (qk << 2);
                    const int kg_col = kcol >> 5;
                    const uint32_t a4 = *reinterpret_cast<const uint32_t*>(A + a_row * K + kcol);
                    uint8_t as_b;
                    if constexpr (ZERO_SCATTER) {
                        as_b = AS[a_row * KG + kg_col];
                    } else {
                        as_b = AS[scale_offset_decode(row, kg_col, KG)];
                    }
                    const float a_sc = e8m0_to_f32(as_b);
                    uint32_t b0, b1;
                    e4m3x4_scale_to_bf16x4(a4, a_sc, b0, b1);
                    *reinterpret_cast<uint2*>(dst + lm * SA_STRIDE + (qk << 2)) = make_uint2(b0, b1);
                }
            };

            uint32_t cur_meta = (V5_MODE != 0) ? __shfl_sync(0xFFFFFFFFu, meta0, 0) : 0u;
            const uint8_t* panel_step_ptr = slab_base;
            SlabMX7Lite cur_slab_lite = {};
            SlabPanelV5 cur_slab_panel = {};
            uint32_t rec0 = 0u;

            if constexpr (V5_MODE == 0) {
                rec0 = (f_cur + lane < f_end) ? static_cast<uint32_t>(ld_w_u16(FLIPS + f_cur + lane, use_ef, l2_pol)) : 0xFFFFu;
                cur_slab_lite = load_slab_mx7_lite(slab_base, lane, use_ef, l2_pol);
            } else if constexpr (V5_MODE == 1) {
                cur_slab_lite = load_slab_mx7_lite(slab_base, lane, use_ef, l2_pol);
                rec0 = issue_hybrid_flip_load(cur_meta, FLIPS, f_cur, BMASKS, d_base, lane, false, use_ef, l2_pol);
            } else {
                const bool is_r8_0 = (cur_meta & META_DENSE_BIT) != 0u;
                cur_slab_panel = load_slab_panel_v5(
                    panel_step_ptr + wp * (is_r8_0 ? MX7_RAW8_SLAB_BYTES : MX7_SLAB_BYTES),
                    lane, is_r8_0, use_ef, l2_pol
                );
                rec0 = issue_hybrid_flip_load(cur_meta, FLIPS, f_cur, nullptr, 0, lane, true, use_ef, l2_pol);
            }

            load_a_blk(0, sA[0]);

            uint32_t cur_fmask = 0u;
            if constexpr (V5_MODE == 0) {
                cur_fmask = finish_slab_flip_mask_legacy(rec0, FLIPS, f_cur, f_end, 0u, lane, sFlip[warp], use_ef, l2_pol);
            } else {
                cur_fmask = finish_hybrid_flip_mask(rec0, cur_meta, FLIPS, f_cur, lane, sFlip[warp], V5_MODE == 2, use_ef, l2_pol);
            }
            __syncthreads();

            float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            const bool row0_live = (g < live_m);
            const bool row8_live = (TILE > 8) && (g + 8 < live_m);

            for (int blk = 0; blk < n_blks; ++blk) {
                const __nv_bfloat16* sAp = sA[blk & 1];
                __nv_bfloat16* sAnext    = sA[(blk & 1) ^ 1];

                #pragma unroll 1
                for (int is = 0; is < SLABS_PER_BLK; ++is) {
                    const int sl = blk * SLABS_PER_BLK + is;
                    const bool cur_is_raw8 = (V5_MODE == 2) && ((cur_meta & META_DENSE_BIT) != 0u);
                    if constexpr (V5_MODE == 2) {
                        panel_step_ptr += cur_is_raw8 ? MX7_PANEL_R8_BYTES : MX7_PANEL_MX7_BYTES;
                    }

                    SlabMX7Lite nxt_slab_lite = {};
                    SlabPanelV5 nxt_slab_panel = {};
                    uint32_t nxt_meta = 0u;
                    uint32_t nxt_rec  = 0u;

                    if (sl + 1 < n_slabs) {
                        const int nsl = sl + 1;
                        if constexpr (V5_MODE == 0) {
                            nxt_slab_lite = load_slab_mx7_lite(slab_base + static_cast<int64_t>(nsl) * MX7_SLAB_BYTES, lane, use_ef, l2_pol);
                            nxt_rec = (f_cur + lane < f_end) ? static_cast<uint32_t>(ld_w_u16(FLIPS + f_cur + lane, use_ef, l2_pol)) : 0xFFFFu;
                        } else if constexpr (V5_MODE == 1) {
                            nxt_meta = (nsl < 32)
                                ? __shfl_sync(0xFFFFFFFFu, meta0, nsl)
                                : __shfl_sync(0xFFFFFFFFu, meta1, nsl - 32);
                            nxt_slab_lite = load_slab_mx7_lite(slab_base + static_cast<int64_t>(nsl) * MX7_SLAB_BYTES, lane, use_ef, l2_pol);
                            nxt_rec = issue_hybrid_flip_load(nxt_meta, FLIPS, f_cur, BMASKS, d_base, lane, false, use_ef, l2_pol);
                        } else {
                            nxt_meta = (nsl < 32)
                                ? __shfl_sync(0xFFFFFFFFu, meta0, nsl)
                                : __shfl_sync(0xFFFFFFFFu, meta1, nsl - 32);
                            const bool nxt_r8 = (nxt_meta & META_DENSE_BIT) != 0u;
                            nxt_slab_panel = load_slab_panel_v5(
                                panel_step_ptr + wp * (nxt_r8 ? MX7_RAW8_SLAB_BYTES : MX7_SLAB_BYTES),
                                lane, nxt_r8, use_ef, l2_pol
                            );
                            nxt_rec = issue_hybrid_flip_load(nxt_meta, FLIPS, f_cur, nullptr, 0, lane, true, use_ef, l2_pol);
                        }
                        if ((sl & 7) == 6 && sl + 2 < n_slabs) {
                            const int off32 = ((sl + 2) >> 3) * 32 + lane;
                            ws_nxt = (off32 < ws_tot_u32) ? ld_w_u32(ws_u32_base + off32, use_ef, l2_pol) : 0u;
                        }
                    } else {
                        ensure_next();
                        if (nidx < n_total) {
                        const int nlog_t = nidx / N_PANELS;
                        const int nnb    = nidx - nlog_t * N_PANELS;
                        const int nt     = map_tile(nlog_t);
                        const int nnb32 = (BN == 16) ? (nnb >> 1) : nnb;
                        const int nwp   = (BN == 16) ? (((nnb & 1) << 1) + warp) : warp;
                        const int64_t ne    = static_cast<int64_t>(Tiles[nt]);
                        const int64_t npuid = ne * N_PANELS32 + nnb32;
                        const int64_t nwuid = npuid * 4 + nwp;
                        if constexpr (V5_MODE == 2) {
                            const uint8_t* npb   = SLABS + static_cast<int64_t>(AUX_OFF[npuid]) * 128;
                            const uint8_t* nbody = npb + static_cast<int64_t>(W13_WS3_HDR_BLOCKS) * 128;
                            if (lane < 7) prefetch_l2(nbody + nwp * MX7_SLAB_BYTES + lane * 128);
                            else if (lane < 14) prefetch_l2(nbody + MX7_PANEL_MX7_BYTES + nwp * MX7_SLAB_BYTES + (lane - 7) * 128);
                            else if (lane < 19) prefetch_l2(reinterpret_cast<const uint32_t*>(npb) + nwp * W13_WS3_WARP_U32 + (lane - 14) * 32);
                            else if (lane < 21) prefetch_l2(SLAB_META + nwuid * n_slabs + (lane - 19) * 32);
                        } else {
                            const uint8_t* nsp = SLABS + nwuid * n_slabs * MX7_SLAB_BYTES;
                            if (lane < 7) prefetch_l2(nsp + lane * 128);
                            else if (lane < 14) prefetch_l2(nsp + MX7_SLAB_BYTES + (lane - 7) * 128);
                            else if (lane < 21) prefetch_l2(nsp + 2 * MX7_SLAB_BYTES + (lane - 14) * 128);
                            else if (lane < 23) prefetch_l2(reinterpret_cast<const uint32_t*>(WS3 + nwuid * n_slabs) + (lane - 21) * 32);
                            else if (lane < 25) {
                                if constexpr (V5_MODE != 0) prefetch_l2(SLAB_META + nwuid * n_slabs + (lane - 23) * 32);
                            }
                        }
                        }
                    }

                    if (is == 0 && blk + 1 < n_blks) {
                        load_a_blk(blk + 1, sAnext);
                    }

                    const bool slab_all_normal = (V5_MODE != 0) && fast_ws && ((cur_meta & META_ALL_NORMAL_BIT) != 0u);
                    const bool slab_no_nan     = (V5_MODE != 0) && fast_ws && ((cur_meta & META_HAS_NAN_BIT) == 0u);
                    const int blk_k_off  = is * SLAB_K;
                    const int ws_sl_base = (sl & 7) << 2;

                    auto run_w13_mma_pair = [&](int kq_off, uint32_t b0_0, uint32_t b1_0, uint32_t b0_1, uint32_t b1_1) {
                        const uint2 a0 = row0_live ? *reinterpret_cast<const uint2*>(sAp + x0_0 + kq_off)      : make_uint2(0u, 0u);
                        const uint2 a2 = row0_live ? *reinterpret_cast<const uint2*>(sAp + x0_0 + kq_off + 16) : make_uint2(0u, 0u);
                        uint2 a1 = make_uint2(0u, 0u), a3 = make_uint2(0u, 0u);
                        if constexpr (TILE > 8) {
                            if (row8_live) {
                                a1 = *reinterpret_cast<const uint2*>(sAp + x0_1 + kq_off);
                                a3 = *reinterpret_cast<const uint2*>(sAp + x0_1 + kq_off + 16);
                            }
                        }
                        mma_m16n8k16_bf16(acc, a0.x, a1.x, a2.x, a3.x, b0_0, b1_0);
                        mma_m16n8k16_bf16(acc, a0.y, a1.y, a2.y, a3.y, b0_1, b1_1);
                    };

                    if constexpr (V5_MODE == 2) {
                        if (cur_is_raw8) {
                            #pragma unroll
                            for (int q = 0; q < 4; ++q) {
                                const int kq_off = blk_k_off + q * 32;
                                const uint32_t ws_word = __shfl_sync(0xFFFFFFFFu, ws_cur, ws_sl_base | q);
                                const uint32_t ws_byte = base_ws + ((ws_word >> (g * 3)) & 7u);
                                const uint32_t ws_off2 = ((ws_byte - 7u) << 7) * 0x00010001u;
                                const float    w_sc    = __uint_as_float(ws_byte << 23);

                                const uint32_t r0 = (q == 0) ? cur_slab_panel.a.x : ((q == 1) ? cur_slab_panel.a.z : ((q == 2) ? cur_slab_panel.b.x : cur_slab_panel.b.z));
                                const uint32_t r1 = (q == 0) ? cur_slab_panel.a.y : ((q == 1) ? cur_slab_panel.a.w : ((q == 2) ? cur_slab_panel.b.y : cur_slab_panel.b.w));
                                uint32_t b0_0, b1_0, b0_1, b1_1;
                                decode_raw8_step_to_bf16x4(r0, ws_off2, w_sc, slab_all_normal, slab_no_nan, b0_0, b1_0);
                                decode_raw8_step_to_bf16x4(r1, ws_off2, w_sc, slab_all_normal, slab_no_nan, b0_1, b1_1);
                                run_w13_mma_pair(kq_off, b0_0, b1_0, b0_1, b1_1);
                            }
                        } else {
                            const uint32_t e0_reg = cur_slab_panel.b.z;
                            #pragma unroll
                            for (int q = 0; q < 4; ++q) {
                                const int kq_off = blk_k_off + q * 32;
                                const uint32_t ws_word = __shfl_sync(0xFFFFFFFFu, ws_cur, ws_sl_base | q);
                                const uint32_t ws_byte = base_ws + ((ws_word >> (g * 3)) & 7u);
                                const uint32_t ws_off2 = ((ws_byte - 7u) << 7) * 0x00010001u;
                                const float    w_sc    = __uint_as_float(ws_byte << 23);

                                const uint32_t sm_w  = (q == 0) ? cur_slab_panel.a.x : ((q == 1) ? cur_slab_panel.a.y : ((q == 2) ? cur_slab_panel.a.z : cur_slab_panel.a.w));
                                const uint32_t e21_p = ((q < 2) ? cur_slab_panel.b.x : cur_slab_panel.b.y) >> ((q & 1) * 16);
                                const int s0 = q << 1;
                                const int s1 = s0 | 1;
                                uint32_t b0_0, b1_0, b0_1, b1_1;
                                decode_mx7_step_to_bf16x4(
                                    sm_w & 0xFFFFu, e21_p & 0xFFu, (e0_reg >> (s0 * 4)) & 0xFu, (cur_fmask >> (s0 * 4)) & 0xFu,
                                    ws_off2, w_sc, slab_all_normal, slab_no_nan, b0_0, b1_0);
                                decode_mx7_step_to_bf16x4(
                                    sm_w >> 16, (e21_p >> 8) & 0xFFu, (e0_reg >> (s1 * 4)) & 0xFu, (cur_fmask >> (s1 * 4)) & 0xFu,
                                    ws_off2, w_sc, slab_all_normal, slab_no_nan, b0_1, b1_1);
                                run_w13_mma_pair(kq_off, b0_0, b1_0, b0_1, b1_1);
                            }
                        }
                    } else {
                        #pragma unroll
                        for (int q = 0; q < 4; ++q) {
                            const int kq_off = blk_k_off + q * 32;
                            const uint32_t ws_word = __shfl_sync(0xFFFFFFFFu, ws_cur, ws_sl_base | q);
                            const uint32_t ws_byte = base_ws + ((ws_word >> (g * 3)) & 7u);
                            const uint32_t ws_off2 = ((ws_byte - 7u) << 7) * 0x00010001u;
                            const float    w_sc    = __uint_as_float(ws_byte << 23);

                            const uint32_t sm_w  = (q == 0) ? cur_slab_lite.sm.x : ((q == 1) ? cur_slab_lite.sm.y : ((q == 2) ? cur_slab_lite.sm.z : cur_slab_lite.sm.w));
                            const uint32_t e21_p = ((q < 2) ? cur_slab_lite.e21.x : cur_slab_lite.e21.y) >> ((q & 1) * 16);
                            const int s0 = q << 1;
                            const int s1 = s0 | 1;
                            uint32_t b0_0, b1_0, b0_1, b1_1;
                            if constexpr (V5_MODE == 0) {
                                e4m3x4_scale_to_bf16x4(
                                    build_e4_pack_compat_fast(sm_w & 0xFFFFu, e21_p & 0xFFu, (cur_slab_lite.e0 >> (s0 * 4)) & 0xFu, (cur_fmask >> (s0 * 4)) & 0xFu),
                                    w_sc, b0_0, b1_0);
                                e4m3x4_scale_to_bf16x4(
                                    build_e4_pack_compat_fast(sm_w >> 16, (e21_p >> 8) & 0xFFu, (cur_slab_lite.e0 >> (s1 * 4)) & 0xFu, (cur_fmask >> (s1 * 4)) & 0xFu),
                                    w_sc, b0_1, b1_1);
                            } else {
                                decode_mx7_step_to_bf16x4(
                                    sm_w & 0xFFFFu, e21_p & 0xFFu, (cur_slab_lite.e0 >> (s0 * 4)) & 0xFu, (cur_fmask >> (s0 * 4)) & 0xFu,
                                    ws_off2, w_sc, slab_all_normal, slab_no_nan, b0_0, b1_0);
                                decode_mx7_step_to_bf16x4(
                                    sm_w >> 16, (e21_p >> 8) & 0xFFu, (cur_slab_lite.e0 >> (s1 * 4)) & 0xFu, (cur_fmask >> (s1 * 4)) & 0xFu,
                                    ws_off2, w_sc, slab_all_normal, slab_no_nan, b0_1, b1_1);
                            }
                            run_w13_mma_pair(kq_off, b0_0, b1_0, b0_1, b1_1);
                        }
                    }

                    if (sl == n_slabs - 2) {
                        ensure_next();
                        if constexpr (USE_PDL) {
                            if (trig_dep && nidx >= n_total && !trig_fired) {
                                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
                                trig_fired = true;
                            }
                        }
                    }

                    if (sl + 1 < n_slabs) {
                        if ((sl & 7) == 7) ws_cur = ws_nxt;
                        if constexpr (V5_MODE == 2) {
                            cur_slab_panel = nxt_slab_panel;
                            cur_meta       = nxt_meta;
                            cur_fmask      = finish_hybrid_flip_mask(nxt_rec, cur_meta, FLIPS, f_cur, lane, sFlip[warp], true, use_ef, l2_pol);
                        } else if constexpr (V5_MODE == 1) {
                            cur_slab_lite  = nxt_slab_lite;
                            cur_meta       = nxt_meta;
                            cur_fmask      = finish_hybrid_flip_mask(nxt_rec, cur_meta, FLIPS, f_cur, lane, sFlip[warp], false, use_ef, l2_pol);
                        } else {
                            cur_slab_lite  = nxt_slab_lite;
                            cur_fmask      = finish_slab_flip_mask_legacy(nxt_rec, FLIPS, f_cur, f_end, static_cast<uint32_t>(sl + 1), lane, sFlip[warp], use_ef, l2_pol);
                        }
                    }
                }
                if (blk + 1 < n_blks) __syncthreads();
            }

            const float scale = Ones[Tiles[t]];
            const int rbase = t * TILE;
            const int r0 = rbase + g;
            const int r1 = rbase + g + 8;
            const float o0 = mul_rn(acc[0], scale);
            const float o1 = mul_rn(acc[1], scale);
            const float o2 = mul_rn(acc[2], scale);
            const float o3 = mul_rn(acc[3], scale);

            if constexpr (FUSE_SITU) {
                const int inter = N >> 1;  // 288
                const int c_warp = nb * (BN >> 1) + warp * 4;
                const int compact_base = TileCompactStart[t];
                const int c0 = compact_base + g;
                const int c1 = compact_base + g + 8;

                uint16_t y0_b = 0u;
                if (g < live_m) y0_b = situ_pair_bf16_exact(pack_bf16x2(o0, o1), __ldg(Gammas + r0));
                const uint32_t p0 = static_cast<uint32_t>(y0_b)
                    | (__shfl_xor_sync(0xFFFFFFFFu, static_cast<uint32_t>(y0_b), 1) << 16);
                const uint32_t p1 = __shfl_xor_sync(0xFFFFFFFFu, p0, 2);
                if (lt == 0 && g < live_m) {
                    if constexpr (FUSE_QDEQ) {
                        st_cg_u2(Out + static_cast<int64_t>(c0) * inter + c_warp, make_uint2(p0, p1));
                    } else {
                        *reinterpret_cast<uint2*>(Out + static_cast<int64_t>(c0) * inter + c_warp) = make_uint2(p0, p1);
                    }
                }
                if constexpr (TILE > 8) {
                    uint16_t y1_b = 0u;
                    if (g + 8 < live_m) y1_b = situ_pair_bf16_exact(pack_bf16x2(o2, o3), __ldg(Gammas + r1));
                    const uint32_t q0 = static_cast<uint32_t>(y1_b)
                        | (__shfl_xor_sync(0xFFFFFFFFu, static_cast<uint32_t>(y1_b), 1) << 16);
                    const uint32_t q1 = __shfl_xor_sync(0xFFFFFFFFu, q0, 2);
                    if (lt == 0 && g + 8 < live_m) {
                        if constexpr (FUSE_QDEQ) {
                            st_cg_u2(Out + static_cast<int64_t>(c1) * inter + c_warp, make_uint2(q0, q1));
                        } else {
                            *reinterpret_cast<uint2*>(Out + static_cast<int64_t>(c1) * inter + c_warp) = make_uint2(q0, q1);
                        }
                    }
                }
            } else {
                const int n0 = nb * BN + warp * 8 + 2 * lt;
                if (g < live_m) {
                    *reinterpret_cast<uint32_t*>(Out + static_cast<int64_t>(r0) * N + n0) = pack_bf16x2(o0, o1);
                }
                if constexpr (TILE > 8) {
                    if (g + 8 < live_m) {
                        *reinterpret_cast<uint32_t*>(Out + static_cast<int64_t>(r1) * N + n0) = pack_bf16x2(o2, o3);
                    }
                }
            }
        }

        if constexpr (FUSE_SITU && FUSE_QDEQ) {
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
            if (sTileLastQdeq && live_m > 0) {
                run_tile_situ_mxfp8_qdeq_inplace_v14<THREADS>(
                    Out, TileCompactStart[t], live_m, tid
                );
            }
            ensure_next();
            __syncthreads();
        } else {
            ensure_next();
            if (nidx < n_total) __syncthreads();
        }
        idx = nidx;
        is_first_tile = false;
    }

    if (use_dyn_q) {
        __syncthreads();
        if (tid == 0) {
            const uint32_t prev_done = atom_acq_rel_gpu_add(&WorkQueue[1], 1u);
            if (prev_done == static_cast<uint32_t>(stride - 1)) {
                st_rel_gpu_u32(&WorkQueue[0], 0u);
                st_rel_gpu_u32(&WorkQueue[1], 0u);
                __threadfence();
            }
        }
    }

    if constexpr (USE_PDL) {
        if (trig_dep && !trig_fired) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }
}

// =========================================================================
// Kernel 2 (C=1..2 v14): BN=128 (4-Warp) MX_W2 (N=4608, K=288, TILE=8|16)
//   - FUSE_QUANT_MODE:
//       0: Pre-quantized MXFP8 `A[row, 288]` + `AS` (legacy)
//       1: OTF BF16 MXFP8 quant-dequant `SiTU_Out[compact, 288]` (cold decoupled fallback)
//       2: V14 FAST PATH! L2-hot pre-QDeq BF16 `SiTU_Out[compact, 288]` (vectorized `ld_cg_u4` -> `sA_all`)
//   - 3-SHFL 4x4 coalesced 64B `st_cg_u4` epilogue + V13 3-wide `ld_cg_u4` combine ring (`va, vb, vc`)
//   - On-the-fly sm/e21 extraction + dead-row `LDS` elimination + `L2::evict_first` weights -> MIN_CTA=6!
// =========================================================================
template <
    int V5_MODE, int TILE, int MIN_CTA, bool USE_PDL,
    bool AS_ROW_MAJOR, int FUSE_QUANT_MODE, bool FUSE_COMBINE
>
__global__ __launch_bounds__(128, MIN_CTA)
void mx7_w2_c1_v16_kernel(
    const uint8_t* __restrict__ A,
    const uint8_t* __restrict__ AS,
    const uint8_t* __restrict__ SLABS,
    const uint4* __restrict__ WS3,
    const uint16_t* __restrict__ FLIPS,
    const int32_t* __restrict__ F_OFF,
    const uint32_t* __restrict__ SLAB_META,
    const uint32_t* __restrict__ BMASKS,
    const int32_t* __restrict__ AUX_OFF,
    const float* __restrict__ Ones,
    const int32_t* __restrict__ Tiles,
    const int32_t* __restrict__ TileCompactStart,
    __nv_bfloat16* __restrict__ Out,
    const int32_t* __restrict__ Src2Compact,
    const int32_t* __restrict__ NLivePtr,
    uint32_t* __restrict__ PanelDone,
    __nv_bfloat16* __restrict__ SendBuf,
    uint32_t* __restrict__ WorkQueue,
    int N, int K, int MAX_CTAS, int T, uint32_t base_ws, int as_row_stride, int down_row_stride,
    int pdl_wait_mode, int trig_dep, int evict_first, int lpt_mode
) {
    static_assert(TILE == 8 || TILE == 16, "C1 MX_W2 TILE must be 8 or 16");
    static_assert(FUSE_QUANT_MODE >= 0 && FUSE_QUANT_MODE <= 2, "FUSE_QUANT_MODE must be 0..2");
    (void)K;

    if constexpr (USE_PDL) {
        if (pdl_wait_mode == 1) {
            asm volatile("griddepcontrol.wait;" ::: "memory");
        }
    }

    constexpr int WARPS = 4;
    constexpr int THREADS = 128;
    constexpr int SUB_N = 4;
    constexpr int BN = 128;
    constexpr int N_PANELS = 36;
    constexpr int M_ROWS = TILE;
    constexpr int KG = 9;
    constexpr int SA_STRIDE = 304;  // 0-way LDS.64!
    constexpr bool USE_COMPACT_IN = (FUSE_QUANT_MODE != 0);

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g    = lane >> 2;
    const int lt   = lane & 3;
    const int x0_0 = g * SA_STRIDE + 4 * lt;
    const int x0_1 = (g + 8) * SA_STRIDE + 4 * lt;
    const bool fast_ws = (base_ws >= 8u && base_ws <= 238u);
    const bool use_ef = (evict_first != 0);
    const uint64_t l2_pol = use_ef ? make_l2_evict_first_policy() : 0ULL;

    const int n_active = Tiles[2 * MAX_CTAS];
    const int bidx   = static_cast<int>(blockIdx.x);
    const int stride = static_cast<int>(gridDim.x);

    if (n_active == 0) {
        if constexpr (USE_PDL) {
            if (pdl_wait_mode == 2) asm volatile("griddepcontrol.wait;" ::: "memory");
        }
        if constexpr (FUSE_COMBINE) {
            if (bidx < N_PANELS) {
                int n_live_tok = T;
                if (NLivePtr != nullptr) {
                    int nl = *NLivePtr;
                    if (nl < 0) nl = 0;
                    if (nl < n_live_tok) n_live_tok = nl;
                }
                const int col_glob = bidx * BN + (tid & 15) * 8;
                const uint4 z4 = make_uint4(0u, 0u, 0u, 0u);
                for (int m = (tid >> 4); m < n_live_tok; m += 8) {
                    *reinterpret_cast<uint4*>(SendBuf + static_cast<int64_t>(m) * N + col_glob) = z4;
                }
                __threadfence();
                __syncthreads();
            }
        }
        if constexpr (USE_PDL) {
            if (trig_dep) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
        return;
    }

    const int n_total = n_active * N_PANELS;
    const bool use_dyn_q = (lpt_mode >= 2) && (WorkQueue != nullptr) && (n_total > stride);
    if (!use_dyn_q && bidx >= n_total) {
        if constexpr (USE_PDL) {
            if (pdl_wait_mode == 2) asm volatile("griddepcontrol.wait;" ::: "memory");
            if (trig_dep) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
        return;
    }

    constexpr int MAX_LPT_TILES = 256;
    alignas(16) __shared__ uint32_t sFlip[WARPS][32];
    alignas(16) __shared__ __nv_bfloat16 sA_all[M_ROWS * SA_STRIDE];
    alignas(16) __shared__ int16_t sTileOrder[MAX_LPT_TILES];
    __shared__ int32_t sIsLastPanel;
    __shared__ int32_t sDynIdx;

    // Built BEFORE griddepcontrol.wait (Tiles is router output -> 100% hidden in W13->W2 PDL window!)
    const bool use_lpt = (lpt_mode != 0) && (n_active > 1) && (n_active <= MAX_LPT_TILES);
    if (use_lpt) {
        build_lpt_tile_order<TILE, MAX_LPT_TILES>(Tiles, MAX_CTAS, n_active, sTileOrder, warp, lane);
        __syncthreads();
    }
    auto map_tile = [&](int lt_idx) -> int {
        return use_lpt ? static_cast<int>(sTileOrder[lt_idx]) : lt_idx;
    };

    bool waited_pdl = false;
    bool trig_fired = false;
    uint32_t cur_nb_tiles = 0u;

    int idx = bidx;
    bool is_very_first = true;
    while (idx < n_total) {
        const int nb    = FUSE_COMBINE ? (idx / n_active) : (idx % N_PANELS);
        const int log_t = FUSE_COMBINE ? (idx - nb * n_active) : (idx / N_PANELS);
        const int t     = map_tile(log_t);
        bool next_claimed = false;
        int nidx = idx + stride;
        bool is_very_last = false;
        int next_nb = 0;
        int next_t  = 0;
        auto ensure_next = [&]() {
            if (!next_claimed) {
                if (use_dyn_q) {
                    if (tid == 0) {
                        sDynIdx = stride + static_cast<int>(atom_acq_rel_gpu_add(&WorkQueue[0], 1u));
                    }
                    __syncthreads();
                    nidx = sDynIdx;
                } else {
                    nidx = idx + stride;
                }
                is_very_last = (nidx >= n_total);
                if (!is_very_last) {
                    next_nb = FUSE_COMBINE ? (nidx / n_active) : (nidx % N_PANELS);
                    const int nlt = FUSE_COMBINE ? (nidx - next_nb * n_active) : (nidx / N_PANELS);
                    next_t  = map_tile(nlt);
                }
                next_claimed = true;
            }
        };

        const int e_i32 = Tiles[t];
        const int end_row = Tiles[MAX_CTAS + t];
        int live_m = end_row - t * TILE;
        if (live_m > TILE) live_m = TILE;
        if (live_m < 0) live_m = 0;

        const int64_t panel_uid = static_cast<int64_t>(e_i32) * N_PANELS + nb;
        const int64_t warp_uid  = panel_uid * WARPS + warp;

        const uint8_t* slab_base = nullptr;
        const uint32_t* ws_u32_base = nullptr;
        if constexpr (V5_MODE == 2) {
            const uint8_t* pbase = SLABS + static_cast<int64_t>(AUX_OFF[panel_uid]) * 128;
            ws_u32_base = reinterpret_cast<const uint32_t*>(pbase) + warp * W2_WS3_WARP_U32;
            slab_base   = pbase + static_cast<int64_t>(W2_WS3_HDR_BLOCKS) * 128;
        } else {
            slab_base   = SLABS + warp_uid * KG * MX7_SLAB_BYTES;
            ws_u32_base = reinterpret_cast<const uint32_t*>(WS3 + warp_uid * KG);
        }

        int32_t f_cur = F_OFF[warp_uid];
        const int32_t f_end  = (V5_MODE == 0) ? F_OFF[warp_uid + 1] : 0;
        const int32_t d_base = (V5_MODE == 1) ? AUX_OFF[warp_uid] : 0;

        auto prefetch_w2_slabs = [&]() {
            if constexpr (V5_MODE == 2) {
                if (lane < 7) prefetch_l2(slab_base + warp * MX7_SLAB_BYTES + lane * 128);
                else if (lane < 14) prefetch_l2(slab_base + MX7_PANEL_MX7_BYTES + warp * MX7_SLAB_BYTES + (lane - 7) * 128);
                else if (lane < 21) prefetch_l2(slab_base + 2 * MX7_PANEL_MX7_BYTES + warp * MX7_SLAB_BYTES + (lane - 14) * 128);
                else if (lane < 23) prefetch_l2(ws_u32_base + (lane - 21) * 32);
                else if (lane == 23) prefetch_l2(SLAB_META + warp_uid * KG);
            } else {
                if (lane < 7) prefetch_l2(slab_base + lane * 128);
                else if (lane < 14) prefetch_l2(slab_base + MX7_SLAB_BYTES + (lane - 7) * 128);
                else if (lane < 21) prefetch_l2(slab_base + 2 * MX7_SLAB_BYTES + (lane - 14) * 128);
                else if (lane < 28) prefetch_l2(slab_base + 3 * MX7_SLAB_BYTES + (lane - 21) * 128);
                else if (lane == 28) prefetch_l2(ws_u32_base);
                else if (lane == 29) prefetch_l2(ws_u32_base + 32);
                else if (lane == 30) {
                    if constexpr (V5_MODE == 0) {
                        if (f_cur < f_end) prefetch_l2(FLIPS + f_cur);
                    } else {
                        prefetch_l2(SLAB_META + warp_uid * KG);
                    }
                }
            }
        };

        // Wave-V16 Zero-Gap W13->W2 Overlap: hoist immutable weight/meta/ws/slab-0/rec0 loads
        // BEFORE griddepcontrol.wait (matching NVFP4 W2 FQ=2!) with 0 extra live registers!
        if constexpr (USE_PDL) {
            if (pdl_wait_mode == 2 && is_very_first) {
                prefetch_w2_slabs();
            }
        }
        uint32_t meta0 = 0u, ws0 = 0u, ws1 = 0u, cur_meta = 0u, rec0 = 0u;
        const uint8_t* panel_step_ptr = slab_base;
        SlabMX7Lite cur_slab_lite = {};
        SlabPanelV5 cur_slab_panel = {};
        if (live_m > 0) {
            meta0 = (V5_MODE != 0 && lane < KG) ? ld_w_u32(SLAB_META + warp_uid * KG + lane, use_ef, l2_pol) : 0u;
            ws0   = ld_w_u32(ws_u32_base + lane, use_ef, l2_pol);
            ws1   = (lane < 4) ? ld_w_u32(ws_u32_base + 32 + lane, use_ef, l2_pol) : 0u;
            cur_meta = (V5_MODE != 0) ? __shfl_sync(0xFFFFFFFFu, meta0, 0) : 0u;
            if constexpr (V5_MODE == 0) {
                rec0 = (f_cur + lane < f_end) ? static_cast<uint32_t>(ld_w_u16(FLIPS + f_cur + lane, use_ef, l2_pol)) : 0xFFFFu;
                cur_slab_lite = load_slab_mx7_lite(slab_base, lane, use_ef, l2_pol);
            } else if constexpr (V5_MODE == 1) {
                cur_slab_lite = load_slab_mx7_lite(slab_base, lane, use_ef, l2_pol);
                rec0 = issue_hybrid_flip_load(cur_meta, FLIPS, f_cur, BMASKS, d_base, lane, false, use_ef, l2_pol);
            } else {
                const bool is_r8_0 = (cur_meta & META_DENSE_BIT) != 0u;
                cur_slab_panel = load_slab_panel_v5(
                    panel_step_ptr + warp * (is_r8_0 ? MX7_RAW8_SLAB_BYTES : MX7_SLAB_BYTES),
                    lane, is_r8_0, use_ef, l2_pol
                );
                rec0 = issue_hybrid_flip_load(cur_meta, FLIPS, f_cur, nullptr, 0, lane, true, use_ef, l2_pol);
            }
        }
        if constexpr (USE_PDL) {
            if (pdl_wait_mode == 2 && !waited_pdl) {
                asm volatile("griddepcontrol.wait;" ::: "memory");
                waited_pdl = true;
            }
        }

        const int compact_tile_base = (USE_COMPACT_IN || FUSE_COMBINE) ? TileCompactStart[t] : 0;

        if (live_m > 0) {
            if constexpr (FUSE_QUANT_MODE == 1) {
                prefetch_w2_slabs();
            }

            {
                if constexpr (FUSE_QUANT_MODE == 2) {
                    // V14 FAST PATH: 100% L2-coherent vectorized copy of pre-QDeq BF16 [live_m, 288] -> sA_all
                    const __nv_bfloat16* situ_in = reinterpret_cast<const __nv_bfloat16*>(A);
                    const int total_u4 = live_m * 36;
                    for (int aidx = tid; aidx < total_u4; aidx += THREADS) {
                        const int lm = aidx / 36;
                        const int v  = aidx - lm * 36;
                        const uint4 d = ld_cg_u4(
                            situ_in + static_cast<int64_t>(compact_tile_base + lm) * 288 + v * 8
                        );
                        *reinterpret_cast<uint4*>(sA_all + lm * SA_STRIDE + v * 8) = d;
                    }
                } else if constexpr (FUSE_QUANT_MODE == 1) {
                    const __nv_bfloat16* situ_in = reinterpret_cast<const __nv_bfloat16*>(A);
                    const int total_blks = live_m * KG;
                    for (int bq = tid; bq < total_blks; bq += THREADS) {
                        const int lm = bq / KG;
                        const int blk = bq - lm * KG;
                        const int crow = compact_tile_base + lm;
                        const uint4* bp = reinterpret_cast<const uint4*>(
                            situ_in + static_cast<int64_t>(crow) * 288 + blk * 32
                        );
                        uint4 v[4] = {ld_cg_u4(bp), ld_cg_u4(bp + 1), ld_cg_u4(bp + 2), ld_cg_u4(bp + 3)};
                        quant_dequant_mxfp8_32_to_sA(v, sA_all + lm * SA_STRIDE + blk * 32);
                    }
                } else {
                    constexpr int Q_PER_ROW = 72;
                    const int total_q = live_m * Q_PER_ROW;
                    const int rbase = t * TILE;
                    for (int aidx = tid; aidx < total_q; aidx += THREADS) {
                        const int lm = aidx / Q_PER_ROW;
                        const int qk = aidx - lm * Q_PER_ROW;
                        const int64_t row = static_cast<int64_t>(rbase + lm);
                        const int kcol = qk << 2;
                        const int kg_col = kcol >> 5;
                        const uint32_t a4 = *reinterpret_cast<const uint32_t*>(A + row * 288 + kcol);
                        uint8_t as_b;
                        if constexpr (AS_ROW_MAJOR) {
                            as_b = AS[row * as_row_stride + kg_col];
                        } else {
                            as_b = AS[scale_offset_decode(row, kg_col, KG)];
                        }
                        const float a_sc = e8m0_to_f32(as_b);
                        uint32_t b0, b1;
                        e4m3x4_scale_to_bf16x4(a4, a_sc, b0, b1);
                        *reinterpret_cast<uint2*>(sA_all + lm * SA_STRIDE + kcol) = make_uint2(b0, b1);
                    }
                }
            }

            uint32_t cur_fmask = 0u;
            if constexpr (V5_MODE == 0) {
                cur_fmask = finish_slab_flip_mask_legacy(rec0, FLIPS, f_cur, f_end, 0u, lane, sFlip[warp], use_ef, l2_pol);
            } else {
                cur_fmask = finish_hybrid_flip_mask(rec0, cur_meta, FLIPS, f_cur, lane, sFlip[warp], V5_MODE == 2, use_ef, l2_pol);
            }
            __syncthreads();

            float acc[SUB_N][4];
            #pragma unroll
            for (int sub = 0; sub < SUB_N; ++sub) {
                acc[sub][0] = 0.0f; acc[sub][1] = 0.0f;
                acc[sub][2] = 0.0f; acc[sub][3] = 0.0f;
            }
            const bool row0_live = (g < live_m);
            const bool row8_live = (TILE > 8) && (g + 8 < live_m);

            #pragma unroll 1
            for (int kg = 0; kg < KG; ++kg) {
                const bool cur_is_raw8 = (V5_MODE == 2) && ((cur_meta & META_DENSE_BIT) != 0u);
                if constexpr (V5_MODE == 2) {
                    panel_step_ptr += cur_is_raw8 ? MX7_PANEL_R8_BYTES : MX7_PANEL_MX7_BYTES;
                }

                SlabMX7Lite nxt_slab_lite = {};
                SlabPanelV5 nxt_slab_panel = {};
                uint32_t nxt_meta = 0u;
                uint32_t nxt_rec  = 0u;

                if (kg + 1 < KG) {
                    const int nkg = kg + 1;
                    if constexpr (V5_MODE == 0) {
                        nxt_slab_lite = load_slab_mx7_lite(slab_base + static_cast<int64_t>(nkg) * MX7_SLAB_BYTES, lane, use_ef, l2_pol);
                        nxt_rec = (f_cur + lane < f_end) ? static_cast<uint32_t>(ld_w_u16(FLIPS + f_cur + lane, use_ef, l2_pol)) : 0xFFFFu;
                    } else if constexpr (V5_MODE == 1) {
                        nxt_meta = __shfl_sync(0xFFFFFFFFu, meta0, nkg);
                        nxt_slab_lite = load_slab_mx7_lite(slab_base + static_cast<int64_t>(nkg) * MX7_SLAB_BYTES, lane, use_ef, l2_pol);
                        nxt_rec = issue_hybrid_flip_load(nxt_meta, FLIPS, f_cur, BMASKS, d_base, lane, false, use_ef, l2_pol);
                    } else {
                        nxt_meta = __shfl_sync(0xFFFFFFFFu, meta0, nkg);
                        const bool nxt_r8 = (nxt_meta & META_DENSE_BIT) != 0u;
                        nxt_slab_panel = load_slab_panel_v5(
                            panel_step_ptr + warp * (nxt_r8 ? MX7_RAW8_SLAB_BYTES : MX7_SLAB_BYTES),
                            lane, nxt_r8, use_ef, l2_pol
                        );
                        nxt_rec = issue_hybrid_flip_load(nxt_meta, FLIPS, f_cur, nullptr, 0, lane, true, use_ef, l2_pol);
                    }
                } else {
                    ensure_next();
                    if (!is_very_last) {
                    const int64_t ne    = static_cast<int64_t>(Tiles[next_t]);
                    const int64_t npuid = ne * N_PANELS + next_nb;
                    const int64_t nwuid = npuid * WARPS + warp;
                    if constexpr (V5_MODE == 2) {
                        const uint8_t* npb   = SLABS + static_cast<int64_t>(AUX_OFF[npuid]) * 128;
                        const uint8_t* nbody = npb + static_cast<int64_t>(W2_WS3_HDR_BLOCKS) * 128;
                        if (lane < 7) prefetch_l2(nbody + warp * MX7_SLAB_BYTES + lane * 128);
                        else if (lane < 14) prefetch_l2(nbody + MX7_PANEL_MX7_BYTES + warp * MX7_SLAB_BYTES + (lane - 7) * 128);
                        else if (lane < 16) prefetch_l2(reinterpret_cast<const uint32_t*>(npb) + warp * W2_WS3_WARP_U32 + (lane - 14) * 32);
                        else if (lane == 16) prefetch_l2(SLAB_META + nwuid * KG);
                    } else {
                        const uint8_t* nsp = SLABS + nwuid * KG * MX7_SLAB_BYTES;
                        if (lane < 7) prefetch_l2(nsp + lane * 128);
                        else if (lane < 14) prefetch_l2(nsp + MX7_SLAB_BYTES + (lane - 7) * 128);
                        else if (lane < 21) prefetch_l2(nsp + 2 * MX7_SLAB_BYTES + (lane - 14) * 128);
                        else if (lane < 23) prefetch_l2(reinterpret_cast<const uint32_t*>(WS3 + nwuid * KG) + (lane - 21) * 32);
                        else if (lane == 23) {
                            if constexpr (V5_MODE != 0) prefetch_l2(SLAB_META + nwuid * KG);
                        }
                    }
                    }
                }

                const int k0 = kg * 32;
                const uint2 a0 = row0_live ? *reinterpret_cast<const uint2*>(sA_all + x0_0 + k0)      : make_uint2(0u, 0u);
                const uint2 a2 = row0_live ? *reinterpret_cast<const uint2*>(sA_all + x0_0 + k0 + 16) : make_uint2(0u, 0u);
                uint2 a1 = make_uint2(0u, 0u), a3 = make_uint2(0u, 0u);
                if constexpr (TILE > 8) {
                    if (row8_live) {
                        a1 = *reinterpret_cast<const uint2*>(sA_all + x0_1 + k0);
                        a3 = *reinterpret_cast<const uint2*>(sA_all + x0_1 + k0 + 16);
                    }
                }

                const bool slab_all_normal = (V5_MODE != 0) && fast_ws && ((cur_meta & META_ALL_NORMAL_BIT) != 0u);
                const bool slab_no_nan     = (V5_MODE != 0) && fast_ws && ((cur_meta & META_HAS_NAN_BIT) == 0u);
                const bool ws_in_first32   = (kg < 8);
                const int  ws_kg_base      = (kg & 7) << 2;

                if constexpr (V5_MODE == 2) {
                    if (cur_is_raw8) {
                        #pragma unroll
                        for (int sub = 0; sub < SUB_N; ++sub) {
                            const uint32_t ws_word = ws_in_first32
                                ? __shfl_sync(0xFFFFFFFFu, ws0, ws_kg_base | sub)
                                : __shfl_sync(0xFFFFFFFFu, ws1, sub);
                            const uint32_t ws_byte = base_ws + ((ws_word >> (g * 3)) & 7u);
                            const uint32_t ws_off2 = ((ws_byte - 7u) << 7) * 0x00010001u;
                            const float    w_sc    = __uint_as_float(ws_byte << 23);

                            const uint32_t r0 = (sub == 0) ? cur_slab_panel.a.x : ((sub == 1) ? cur_slab_panel.a.z : ((sub == 2) ? cur_slab_panel.b.x : cur_slab_panel.b.z));
                            const uint32_t r1 = (sub == 0) ? cur_slab_panel.a.y : ((sub == 1) ? cur_slab_panel.a.w : ((sub == 2) ? cur_slab_panel.b.y : cur_slab_panel.b.w));
                            uint32_t b0_0, b1_0, b0_1, b1_1;
                            decode_raw8_step_to_bf16x4(r0, ws_off2, w_sc, slab_all_normal, slab_no_nan, b0_0, b1_0);
                            decode_raw8_step_to_bf16x4(r1, ws_off2, w_sc, slab_all_normal, slab_no_nan, b0_1, b1_1);

                            mma_m16n8k16_bf16(acc[sub], a0.x, a1.x, a2.x, a3.x, b0_0, b1_0);
                            mma_m16n8k16_bf16(acc[sub], a0.y, a1.y, a2.y, a3.y, b0_1, b1_1);
                        }
                    } else {
                        const uint32_t e0_reg = cur_slab_panel.b.z;
                        #pragma unroll
                        for (int sub = 0; sub < SUB_N; ++sub) {
                            const uint32_t ws_word = ws_in_first32
                                ? __shfl_sync(0xFFFFFFFFu, ws0, ws_kg_base | sub)
                                : __shfl_sync(0xFFFFFFFFu, ws1, sub);
                            const uint32_t ws_byte = base_ws + ((ws_word >> (g * 3)) & 7u);
                            const uint32_t ws_off2 = ((ws_byte - 7u) << 7) * 0x00010001u;
                            const float    w_sc    = __uint_as_float(ws_byte << 23);

                            const uint32_t sm_w  = (sub == 0) ? cur_slab_panel.a.x : ((sub == 1) ? cur_slab_panel.a.y : ((sub == 2) ? cur_slab_panel.a.z : cur_slab_panel.a.w));
                            const uint32_t e21_p = ((sub < 2) ? cur_slab_panel.b.x : cur_slab_panel.b.y) >> ((sub & 1) * 16);
                            const int s0 = sub << 1;
                            const int s1 = s0 | 1;
                            uint32_t b0_0, b1_0, b0_1, b1_1;
                            decode_mx7_step_to_bf16x4(
                                sm_w & 0xFFFFu, e21_p & 0xFFu, (e0_reg >> (s0 * 4)) & 0xFu, (cur_fmask >> (s0 * 4)) & 0xFu,
                                ws_off2, w_sc, slab_all_normal, slab_no_nan, b0_0, b1_0);
                            decode_mx7_step_to_bf16x4(
                                sm_w >> 16, (e21_p >> 8) & 0xFFu, (e0_reg >> (s1 * 4)) & 0xFu, (cur_fmask >> (s1 * 4)) & 0xFu,
                                ws_off2, w_sc, slab_all_normal, slab_no_nan, b0_1, b1_1);

                            mma_m16n8k16_bf16(acc[sub], a0.x, a1.x, a2.x, a3.x, b0_0, b1_0);
                            mma_m16n8k16_bf16(acc[sub], a0.y, a1.y, a2.y, a3.y, b0_1, b1_1);
                        }
                    }
                } else {
                    #pragma unroll
                    for (int sub = 0; sub < SUB_N; ++sub) {
                        const uint32_t ws_word = ws_in_first32
                            ? __shfl_sync(0xFFFFFFFFu, ws0, ws_kg_base | sub)
                            : __shfl_sync(0xFFFFFFFFu, ws1, sub);
                        const uint32_t ws_byte = base_ws + ((ws_word >> (g * 3)) & 7u);
                        const uint32_t ws_off2 = ((ws_byte - 7u) << 7) * 0x00010001u;
                        const float    w_sc    = __uint_as_float(ws_byte << 23);

                        const uint32_t sm_w  = (sub == 0) ? cur_slab_lite.sm.x : ((sub == 1) ? cur_slab_lite.sm.y : ((sub == 2) ? cur_slab_lite.sm.z : cur_slab_lite.sm.w));
                        const uint32_t e21_p = ((sub < 2) ? cur_slab_lite.e21.x : cur_slab_lite.e21.y) >> ((sub & 1) * 16);
                        const int s0 = sub << 1;
                        const int s1 = s0 | 1;
                        uint32_t b0_0, b1_0, b0_1, b1_1;
                        if constexpr (V5_MODE == 0) {
                            e4m3x4_scale_to_bf16x4(
                                build_e4_pack_compat_fast(sm_w & 0xFFFFu, e21_p & 0xFFu, (cur_slab_lite.e0 >> (s0 * 4)) & 0xFu, (cur_fmask >> (s0 * 4)) & 0xFu),
                                w_sc, b0_0, b1_0);
                            e4m3x4_scale_to_bf16x4(
                                build_e4_pack_compat_fast(sm_w >> 16, (e21_p >> 8) & 0xFFu, (cur_slab_lite.e0 >> (s1 * 4)) & 0xFu, (cur_fmask >> (s1 * 4)) & 0xFu),
                                w_sc, b0_1, b1_1);
                        } else {
                            decode_mx7_step_to_bf16x4(
                                sm_w & 0xFFFFu, e21_p & 0xFFu, (cur_slab_lite.e0 >> (s0 * 4)) & 0xFu, (cur_fmask >> (s0 * 4)) & 0xFu,
                                ws_off2, w_sc, slab_all_normal, slab_no_nan, b0_0, b1_0);
                            decode_mx7_step_to_bf16x4(
                                sm_w >> 16, (e21_p >> 8) & 0xFFu, (cur_slab_lite.e0 >> (s1 * 4)) & 0xFu, (cur_fmask >> (s1 * 4)) & 0xFu,
                                ws_off2, w_sc, slab_all_normal, slab_no_nan, b0_1, b1_1);
                        }

                        mma_m16n8k16_bf16(acc[sub], a0.x, a1.x, a2.x, a3.x, b0_0, b1_0);
                        mma_m16n8k16_bf16(acc[sub], a0.y, a1.y, a2.y, a3.y, b0_1, b1_1);
                    }
                }

                if (kg == KG - 2) {
                    ensure_next();
                    // Strictly NO early TRIG_DEP when FUSE_COMBINE is true!
                    if constexpr (USE_PDL && !FUSE_COMBINE) {
                        if (trig_dep && is_very_last && !trig_fired) {
                            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
                            trig_fired = true;
                        }
                    }
                }

                if (kg + 1 < KG) {
                    if constexpr (V5_MODE == 2) {
                        cur_slab_panel = nxt_slab_panel;
                        cur_meta       = nxt_meta;
                        cur_fmask      = finish_hybrid_flip_mask(nxt_rec, cur_meta, FLIPS, f_cur, lane, sFlip[warp], true, use_ef, l2_pol);
                    } else if constexpr (V5_MODE == 1) {
                        cur_slab_lite  = nxt_slab_lite;
                        cur_meta       = nxt_meta;
                        cur_fmask      = finish_hybrid_flip_mask(nxt_rec, cur_meta, FLIPS, f_cur, lane, sFlip[warp], false, use_ef, l2_pol);
                    } else {
                        cur_slab_lite  = nxt_slab_lite;
                        cur_fmask      = finish_slab_flip_mask_legacy(nxt_rec, FLIPS, f_cur, f_end, static_cast<uint32_t>(kg + 1), lane, sFlip[warp], use_ef, l2_pol);
                    }
                }
            }

            // Epilogue: 3-SHFL 4x4 in-register transpose + coalesced 64B uint4 stores!
            const float scale = Ones[Tiles[t]];
            const int rbase = t * TILE;
            const int r0 = rbase + g;
            const int c0 = compact_tile_base + g;
            const int col_u4_local = warp * 32 + lt * 8;
            __nv_bfloat16* down_panel_buf = FUSE_COMBINE
                ? (Out + static_cast<int64_t>(nb) * down_row_stride * BN)
                : nullptr;

            {
                uint32_t pk0[4];
                #pragma unroll
                for (int sub = 0; sub < SUB_N; ++sub) {
                    pk0[sub] = pack_bf16x2(mul_rn(acc[sub][0], scale), mul_rn(acc[sub][1], scale));
                }
                const uint4 v0 = transpose_4x4_bf16x2_u4(pk0[0], pk0[1], pk0[2], pk0[3], lt);
                if (g < live_m) {
                    if constexpr (FUSE_COMBINE) {
                        st_cg_u4(down_panel_buf + static_cast<int64_t>(c0) * BN + col_u4_local, v0);
                    } else {
                        *reinterpret_cast<uint4*>(Out + static_cast<int64_t>(r0) * N + nb * BN + col_u4_local) = v0;
                    }
                }
            }
            if constexpr (TILE > 8) {
                const int r1 = rbase + g + 8;
                const int c1 = compact_tile_base + g + 8;
                uint32_t pk1[4];
                #pragma unroll
                for (int sub = 0; sub < SUB_N; ++sub) {
                    pk1[sub] = pack_bf16x2(mul_rn(acc[sub][2], scale), mul_rn(acc[sub][3], scale));
                }
                const uint4 v1 = transpose_4x4_bf16x2_u4(pk1[0], pk1[1], pk1[2], pk1[3], lt);
                if (g + 8 < live_m) {
                    if constexpr (FUSE_COMBINE) {
                        st_cg_u4(down_panel_buf + static_cast<int64_t>(c1) * BN + col_u4_local, v1);
                    } else {
                        *reinterpret_cast<uint4*>(Out + static_cast<int64_t>(r1) * N + nb * BN + col_u4_local) = v1;
                    }
                }
            }
        }

        ensure_next();
        if constexpr (FUSE_COMBINE) {
            cur_nb_tiles += 1u;
            if (is_very_last || next_nb != nb) {
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
                    // V13/V14 3-wide L2-hot combine ring (`va, vb, vc`, 8 BF16/thr, 8 tokens/iter, exact slot 0..7!)
                    int n_live_tok = T;
                    if (NLivePtr != nullptr) {
                        int nl = *NLivePtr;
                        if (nl < 0) nl = 0;
                        if (nl < n_live_tok) n_live_tok = nl;
                    }
                    const __nv_bfloat16* down_panel_buf = Out + static_cast<int64_t>(nb) * down_row_stride * BN;
                    const int col_in_panel = (tid & 15) * 8;
                    const int col_glob     = nb * BN + col_in_panel;
                    const uint4 z4 = make_uint4(0u, 0u, 0u, 0u);

                    for (int m = (tid >> 4); m < n_live_tok; m += 8) {
                        const int32_t* s2c = Src2Compact + static_cast<int64_t>(m) * 8;
                        const int32_t r0 = __ldg(s2c + 0);
                        const int32_t r1 = __ldg(s2c + 1);
                        const int32_t r2 = __ldg(s2c + 2);
                        uint4 va = (r0 >= 0) ? ld_cg_u4(down_panel_buf + static_cast<int64_t>(r0) * BN + col_in_panel) : z4;
                        uint4 vb = (r1 >= 0) ? ld_cg_u4(down_panel_buf + static_cast<int64_t>(r1) * BN + col_in_panel) : z4;
                        uint4 vc = (r2 >= 0) ? ld_cg_u4(down_panel_buf + static_cast<int64_t>(r2) * BN + col_in_panel) : z4;

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

                        acc_u4(va);
                        { const int32_t r3 = __ldg(s2c + 3); va = (r3 >= 0) ? ld_cg_u4(down_panel_buf + static_cast<int64_t>(r3) * BN + col_in_panel) : z4; }
                        acc_u4(vb);
                        { const int32_t r4 = __ldg(s2c + 4); vb = (r4 >= 0) ? ld_cg_u4(down_panel_buf + static_cast<int64_t>(r4) * BN + col_in_panel) : z4; }
                        acc_u4(vc);
                        { const int32_t r5 = __ldg(s2c + 5); vc = (r5 >= 0) ? ld_cg_u4(down_panel_buf + static_cast<int64_t>(r5) * BN + col_in_panel) : z4; }
                        acc_u4(va);
                        { const int32_t r6 = __ldg(s2c + 6); va = (r6 >= 0) ? ld_cg_u4(down_panel_buf + static_cast<int64_t>(r6) * BN + col_in_panel) : z4; }
                        acc_u4(vb);
                        { const int32_t r7 = __ldg(s2c + 7); vb = (r7 >= 0) ? ld_cg_u4(down_panel_buf + static_cast<int64_t>(r7) * BN + col_in_panel) : z4; }
                        acc_u4(vc);
                        acc_u4(va);
                        acc_u4(vb);

                        *reinterpret_cast<uint4*>(SendBuf + static_cast<int64_t>(m) * N + col_glob) = make_uint4(
                            pack_bf16x2(a0, a1), pack_bf16x2(a2, a3),
                            pack_bf16x2(a4, a5), pack_bf16x2(a6, a7)
                        );
                    }
                    __threadfence();
                    __syncthreads();
                } else if (!is_very_last) {
                    __syncthreads();
                }
                cur_nb_tiles = 0u;
            } else {
                __syncthreads();
            }
        } else {
            if (!is_very_last) __syncthreads();
        }
        idx = nidx;
        is_very_first = false;
    }

    if (use_dyn_q) {
        __syncthreads();
        if (tid == 0) {
            const uint32_t prev_done = atom_acq_rel_gpu_add(&WorkQueue[1], 1u);
            if (prev_done == static_cast<uint32_t>(stride - 1)) {
                st_rel_gpu_u32(&WorkQueue[0], 0u);
                st_rel_gpu_u32(&WorkQueue[1], 0u);
                __threadfence();
            }
        }
    }

    if constexpr (USE_PDL) {
        if (trig_dep && !trig_fired) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }
}

// =========================================================================
// Launch & Dispatch Helpers (Defense-in-Depth PDL Guard + Cold Decoupling!)
// =========================================================================

template <typename KernelT, typename... Args>
static inline cudaError_t launch_mx7_c1_v14(
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

template <int V5_MODE>
static int launch_mx7_w13_c1_impl(
    const void* A, const void* AS,
    const void* SLABS, const void* WS3,
    const void* FLIPS, const void* F_OFF,
    const void* SLAB_META, const void* BMASKS, const void* AUX_OFF,
    const void* Ones, const void* Tiles,
    const void* TileMask, const void* TileCompactStart,
    const void* Gammas, void* Out, void* TileQdeqDone,
    int N, int K, int MAX_CTAS, int TILE, uint32_t base_ws,
    int zero_scatter, int fuse_situ, int fuse_qdeq,
    int bn_mode, int u_thresh,
    int pdl_wait_mode, int trig_dep, int evict_first,
    int lpt_mode, void* WorkQueue, uintptr_t stream_ptr
) {
    if (N != 576 || K != 4608 || (TILE != 8 && TILE != 16)) return -2;
    if (MAX_CTAS <= 0) return 0;
    // Defense-in-depth PDL safety guard (immune to #206 PDL race!)
    if (pdl_wait_mode == 0 && trig_dep != 0) pdl_wait_mode = 1;

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    const bool zs = (zero_scatter != 0) && (TileMask != nullptr);
    const bool fs = (fuse_situ != 0);
    const bool fq = fs && (fuse_qdeq != 0) && (TileQdeqDone != nullptr);
    const int eff_thresh = (u_thresh > 0) ? u_thresh : 37;
    const int eff_margin = (u_thresh >= 0 && u_thresh <= 100) ? u_thresh : 3;
    const int eff_pdl_wait = (zs && pdl_wait_mode == 2) ? 1 : pdl_wait_mode;

    // Occupancy protection: cap cold legacy V5_MODE=0 so hot Compat/Panel hit full MIN_CTA!
    constexpr int T8_BN32_MC  = (V5_MODE == 0 && MX7_C1_W13_T8_BN32_MIN_CTA > 5)  ? 5 : MX7_C1_W13_T8_BN32_MIN_CTA;
    constexpr int T8_BN16_MC  = (V5_MODE == 0 && MX7_C1_W13_T8_BN16_MIN_CTA > 6)  ? 6 : MX7_C1_W13_T8_BN16_MIN_CTA;
    constexpr int T16_BN32_MC = (V5_MODE == 0 && MX7_C1_W13_T16_BN32_MIN_CTA > 4) ? 4 : MX7_C1_W13_T16_BN32_MIN_CTA;
    constexpr int T16_BN16_MC = (V5_MODE == 0 && MX7_C1_W13_T16_BN16_MIN_CTA > 5) ? 5 : MX7_C1_W13_T16_BN16_MIN_CTA;

    const int cap16 = 48 * ((TILE == 8) ? T8_BN16_MC : T16_BN16_MC);
    const int cap32 = 48 * ((TILE == 8) ? T8_BN32_MC : T16_BN32_MC);

    auto launch_one = [&](int bn, int gate_mode, int g_arg0, int g_arg1, int margin, int pw, int td) -> int {
        if (pw == 0 && td != 0) pw = 1;
        const int panels = (bn == 16) ? 36 : 18;
        const int total_tiles = panels * MAX_CTAS;
        const int cap = (bn == 16) ? cap16 : cap32;
        const int gx = (gate_mode != 0) ? cap : ((total_tiles < cap) ? total_tiles : cap);
        const bool use_pdl = (pw != 0) || (td != 0);
        dim3 grid(gx, 1, 1);
        dim3 block((bn / 8) * 32, 1, 1);

        uint32_t* wq_slot = (WorkQueue != nullptr)
            ? (static_cast<uint32_t*>(WorkQueue) + ((bn == 16) ? 0 : 16))
            : nullptr;
        #define D13_RUN(BNV, TV, MC, PDL_B, Z, F, Q) return static_cast<int>(launch_mx7_c1_v14( \
            mx7_w13_c1_v16_kernel<V5_MODE, BNV, TV, MC, PDL_B, Z, F, Q>, \
            grid, block, stream, PDL_B, \
            static_cast<const uint8_t*>(A), static_cast<const uint8_t*>(AS), \
            static_cast<const uint8_t*>(SLABS), static_cast<const uint4*>(WS3), \
            static_cast<const uint16_t*>(FLIPS), static_cast<const int32_t*>(F_OFF), \
            static_cast<const uint32_t*>(SLAB_META), static_cast<const uint32_t*>(BMASKS), \
            static_cast<const int32_t*>(AUX_OFF), static_cast<const float*>(Ones), \
            static_cast<const int32_t*>(Tiles), static_cast<const int32_t*>(TileMask), \
            static_cast<const int32_t*>(TileCompactStart), static_cast<const float*>(Gammas), \
            static_cast<__nv_bfloat16*>(Out), static_cast<uint32_t*>(TileQdeqDone), wq_slot, \
            N, K, MAX_CTAS, base_ws, gate_mode, g_arg0, g_arg1, margin, pw, td, evict_first, lpt_mode))

        #define D13_Q(BNV, TV, MC, Z, F, Q) do { \
            if (use_pdl) { D13_RUN(BNV, TV, MC, true,  Z, F, Q); } \
            else         { D13_RUN(BNV, TV, MC, false, Z, F, Q); } \
        } while (0)

        #define D13(BNV, TV, MC) do { \
            if      (zs && fq)         { D13_Q(BNV, TV, MC, true,  true,  true);  } \
            else if (zs && fs)         { D13_Q(BNV, TV, MC, true,  true,  false); } \
            else if (zs)               { D13_Q(BNV, TV, MC, true,  false, false); } \
            else if (fq)               { D13_Q(BNV, TV, MC, false, true,  true);  } \
            else if (fs)               { D13_Q(BNV, TV, MC, false, true,  false); } \
            else                       { D13_Q(BNV, TV, MC, false, false, false); } \
        } while (0)

        if (bn == 16) {
            if (TILE == 8) { D13(16, 8,  T8_BN16_MC);  }
            else           { D13(16, 16, T16_BN16_MC); }
        } else {
            if (TILE == 8) { D13(32, 8,  T8_BN32_MC);  }
            else           { D13(32, 16, T16_BN32_MC); }
        }
        #undef D13
        #undef D13_Q
        #undef D13_RUN
        return -1;
    };

    if (bn_mode == 0) return launch_one(32, 0, 0, 1000000, 0, eff_pdl_wait, trig_dep);
    if (bn_mode == 1) return launch_one(16, 0, 0, 1000000, 0, eff_pdl_wait, trig_dep);
    const int use_chain_pdl = (eff_pdl_wait != 0 || trig_dep != 0) ? 1 : 0;
    if (bn_mode == 3) {
        int rc = launch_one(16, 2, cap16, cap32, eff_margin, eff_pdl_wait, use_chain_pdl);
        if (rc != 0) return rc;
        return launch_one(32, 2, cap16, cap32, eff_margin, use_chain_pdl, trig_dep);
    }
    int rc = launch_one(16, 1, 1, eff_thresh, 0, eff_pdl_wait, use_chain_pdl);
    if (rc != 0) return rc;
    return launch_one(32, 1, eff_thresh + 1, 1000000, 0, use_chain_pdl, trig_dep);
}

template <int V5_MODE>
static int launch_mx7_w2_c1_impl(
    const void* A, const void* AS,
    const void* SLABS, const void* WS3,
    const void* FLIPS, const void* F_OFF,
    const void* SLAB_META, const void* BMASKS, const void* AUX_OFF,
    const void* Ones, const void* Tiles, const void* TileCompactStart,
    void* Out, const void* Src2Compact, const void* NLivePtr,
    void* PanelDone, void* SendBuf,
    int N, int K, int MAX_CTAS, int TILE, int T, uint32_t base_ws,
    int as_row_major, int as_row_stride, int down_row_stride,
    int fuse_quant, int fuse_combine,
    int pdl_wait_mode, int trig_dep, int evict_first,
    int lpt_mode, void* WorkQueue, uintptr_t stream_ptr
) {
    if (N != 4608 || K != 288 || (TILE != 8 && TILE != 16)) return -2;
    if (fuse_quant < 0 || fuse_quant > 2) return -3;
    // Defense-in-depth PDL safety guard (immune to #206 PDL race!)
    if (pdl_wait_mode == 0 && trig_dep != 0) pdl_wait_mode = 1;

    const bool fc = (fuse_combine != 0);
    const bool rm = (fuse_quant == 0) && (as_row_major != 0);
    const bool use_pdl = (pdl_wait_mode != 0) || (trig_dep != 0);
    const int total_tiles = 36 * MAX_CTAS;
    if (total_tiles <= 0 && !fc) return 0;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    const int eff_as_stride   = (as_row_stride > 0) ? as_row_stride : 9;
    const int eff_down_stride = (down_row_stride > 0) ? down_row_stride : (T * 8);

    // V14 Occupancy Protection: decouple cold legacy FQ=1 & V5_MODE=0 from hot FQ=2/0!
    constexpr int T8_HOT_MC  = (V5_MODE == 0 && MX7_C1_W2_T8_MIN_CTA > 5)  ? 5 : MX7_C1_W2_T8_MIN_CTA;
    constexpr int T8_FQ1_MC  = (T8_HOT_MC > 5)  ? 5 : T8_HOT_MC;
    constexpr int T16_HOT_MC = (V5_MODE == 0 && MX7_C1_W2_T16_MIN_CTA > 4) ? 4 : MX7_C1_W2_T16_MIN_CTA;
    constexpr int T16_FQ1_MC = (T16_HOT_MC > 4) ? 4 : T16_HOT_MC;

    const int eff_mc = (TILE == 8)
        ? ((fuse_quant == 1) ? T8_FQ1_MC : T8_HOT_MC)
        : ((fuse_quant == 1) ? T16_FQ1_MC : T16_HOT_MC);
    const int persist_cap = 48 * eff_mc;
    const int grid_x = fc ? persist_cap : ((total_tiles < persist_cap) ? total_tiles : persist_cap);
    dim3 grid(grid_x, 1, 1);
    dim3 block(128, 1, 1);

    uint32_t* wq_w2 = (WorkQueue != nullptr) ? (static_cast<uint32_t*>(WorkQueue) + 32) : nullptr;
    #define M2C_RUN(TV, MC, PDL_B, R, QM, C) return static_cast<int>(launch_mx7_c1_v14( \
        mx7_w2_c1_v16_kernel<V5_MODE, TV, MC, PDL_B, R, QM, C>, \
        grid, block, stream, PDL_B, \
        static_cast<const uint8_t*>(A), static_cast<const uint8_t*>(AS), \
        static_cast<const uint8_t*>(SLABS), static_cast<const uint4*>(WS3), \
        static_cast<const uint16_t*>(FLIPS), static_cast<const int32_t*>(F_OFF), \
        static_cast<const uint32_t*>(SLAB_META), static_cast<const uint32_t*>(BMASKS), \
        static_cast<const int32_t*>(AUX_OFF), static_cast<const float*>(Ones), \
        static_cast<const int32_t*>(Tiles), static_cast<const int32_t*>(TileCompactStart), \
        static_cast<__nv_bfloat16*>(Out), static_cast<const int32_t*>(Src2Compact), \
        static_cast<const int32_t*>(NLivePtr), static_cast<uint32_t*>(PanelDone), \
        static_cast<__nv_bfloat16*>(SendBuf), wq_w2, \
        N, K, MAX_CTAS, T, base_ws, eff_as_stride, eff_down_stride, pdl_wait_mode, trig_dep, evict_first, lpt_mode))

    #define M2C(TV, MC, R, QM, C) do { \
        if (use_pdl) { M2C_RUN(TV, MC, true,  R, QM, C); } \
        else         { M2C_RUN(TV, MC, false, R, QM, C); } \
    } while (0)

    #define M2C_DISPATCH(TV, MC_HOT, MC_FQ1) do { \
        if      (fuse_quant == 2 && fc)  { M2C(TV, MC_HOT, false, 2, true);  } \
        else if (fuse_quant == 2)        { M2C(TV, MC_HOT, false, 2, false); } \
        else if (fuse_quant == 1 && fc)  { M2C(TV, MC_FQ1, false, 1, true);  } \
        else if (fuse_quant == 1)        { M2C(TV, MC_FQ1, false, 1, false); } \
        else if (fc && rm)               { M2C(TV, MC_HOT, true,  0, true);  } \
        else if (fc)                     { M2C(TV, MC_HOT, false, 0, true);  } \
        else if (rm)                     { M2C(TV, MC_HOT, true,  0, false); } \
        else                             { M2C(TV, MC_HOT, false, 0, false); } \
    } while (0)

    if (TILE == 8) { M2C_DISPATCH(8,  T8_HOT_MC,  T8_FQ1_MC);  }
    else           { M2C_DISPATCH(16, T16_HOT_MC, T16_FQ1_MC); }
    #undef M2C_DISPATCH
    #undef M2C
    #undef M2C_RUN
    return -1;
}

}  // namespace

// =========================================================================
// Extern "C" ABI (100% v4/v5 Backward-Compatible + Wave-V14 Entrypoints)
// =========================================================================

extern "C" int mx7_c1_version() {
    return 16;
}

extern "C" int mx7_c1_min_cta() {
    return (MX7_C1_W13_T8_BN32_MIN_CTA & 0xF)
         | ((MX7_C1_W13_T8_BN16_MIN_CTA & 0xF) << 4)
         | ((MX7_C1_W2_T8_MIN_CTA & 0xF) << 8)
         | ((MX7_C1_W2_T16_MIN_CTA & 0xF) << 12)
         | ((MX7_C1_W13_T16_BN32_MIN_CTA & 0xF) << 16)
         | ((MX7_C1_W13_T16_BN16_MIN_CTA & 0xF) << 20);
}

extern "C" int mx7_v2_count_flips(
    const void* W, void* counts, int proj, int E, int N, int K,
    int from_panel, int src_pb, int src_kb, uintptr_t stream_ptr
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream_ptr);
    const int bn = (proj == 0) ? 32 : 64;
    const int warps = (proj == 0) ? 4 : 2;
    const int total_warps = E * (N / bn) * warps;
    if (proj == 0) {
        if (from_panel) mx7_c1_count_flips_kernel<0, true><<<total_warps, 32, 0, s>>>(
            static_cast<const uint8_t*>(W), static_cast<int32_t*>(counts), E, N, K, src_pb, src_kb);
        else mx7_c1_count_flips_kernel<0, false><<<total_warps, 32, 0, s>>>(
            static_cast<const uint8_t*>(W), static_cast<int32_t*>(counts), E, N, K, 0, 0);
    } else if (proj == 1) {
        if (from_panel) mx7_c1_count_flips_kernel<1, true><<<total_warps, 32, 0, s>>>(
            static_cast<const uint8_t*>(W), static_cast<int32_t*>(counts), E, N, K, src_pb, src_kb);
        else mx7_c1_count_flips_kernel<1, false><<<total_warps, 32, 0, s>>>(
            static_cast<const uint8_t*>(W), static_cast<int32_t*>(counts), E, N, K, 0, 0);
    } else {
        return -1;
    }
    return static_cast<int>(cudaGetLastError());
}

extern "C" int mx7_v2_pack(
    const void* W, const void* WS, void* SLABS, void* WS3, void* FLIPS, const void* F_OFF,
    int proj, int E, int N, int K, uint32_t base_ws,
    int from_panel, int src_pb, int src_kb, uintptr_t stream_ptr
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream_ptr);
    const int bn = (proj == 0) ? 32 : 64;
    const int warps = (proj == 0) ? 4 : 2;
    const int total_warps = E * (N / bn) * warps;
    if (proj == 0) {
        if (from_panel) mx7_c1_pack_kernel<0, true><<<total_warps, 32, 0, s>>>(
            static_cast<const uint8_t*>(W), static_cast<const uint8_t*>(WS),
            static_cast<uint8_t*>(SLABS), static_cast<uint4*>(WS3),
            static_cast<uint16_t*>(FLIPS), static_cast<const int32_t*>(F_OFF),
            E, N, K, base_ws, src_pb, src_kb);
        else mx7_c1_pack_kernel<0, false><<<total_warps, 32, 0, s>>>(
            static_cast<const uint8_t*>(W), static_cast<const uint8_t*>(WS),
            static_cast<uint8_t*>(SLABS), static_cast<uint4*>(WS3),
            static_cast<uint16_t*>(FLIPS), static_cast<const int32_t*>(F_OFF),
            E, N, K, base_ws, 0, 0);
    } else if (proj == 1) {
        if (from_panel) mx7_c1_pack_kernel<1, true><<<total_warps, 32, 0, s>>>(
            static_cast<const uint8_t*>(W), static_cast<const uint8_t*>(WS),
            static_cast<uint8_t*>(SLABS), static_cast<uint4*>(WS3),
            static_cast<uint16_t*>(FLIPS), static_cast<const int32_t*>(F_OFF),
            E, N, K, base_ws, src_pb, src_kb);
        else mx7_c1_pack_kernel<1, false><<<total_warps, 32, 0, s>>>(
            static_cast<const uint8_t*>(W), static_cast<const uint8_t*>(WS),
            static_cast<uint8_t*>(SLABS), static_cast<uint4*>(WS3),
            static_cast<uint16_t*>(FLIPS), static_cast<const int32_t*>(F_OFF),
            E, N, K, base_ws, 0, 0);
    } else {
        return -1;
    }
    return static_cast<int>(cudaGetLastError());
}

extern "C" int mx7_v5_count_hybrid_from_mx7(
    const void* FLIPS, const void* F_OFF,
    void* sparse_counts, void* dense_counts,
    int total_warps, int n_slabs, int flip_thresh, uintptr_t stream_ptr
) {
    if (total_warps <= 0 || n_slabs <= 0) return -1;
    const int thr = (flip_thresh < 0) ? 0 : ((flip_thresh > 63) ? 63 : flip_thresh);
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream_ptr);
    mx7_c1_v5_count_hybrid_from_mx7_kernel<<<total_warps, 32, 0, s>>>(
        static_cast<const uint16_t*>(FLIPS), static_cast<const int32_t*>(F_OFF),
        static_cast<int32_t*>(sparse_counts), static_cast<int32_t*>(dense_counts),
        total_warps, n_slabs, thr
    );
    return static_cast<int>(cudaGetLastError());
}

extern "C" int mx7_v5_build_hybrid_from_mx7(
    const void* SLABS, const void* FLIPS, const void* F_OFF,
    const void* S_OFF, const void* D_OFF,
    void* SLAB_META, void* SFLIPS, void* BMASKS,
    int total_warps, int n_slabs, int flip_thresh, uintptr_t stream_ptr
) {
    if (total_warps <= 0 || n_slabs <= 0) return -1;
    const int thr = (flip_thresh < 0) ? 0 : ((flip_thresh > 63) ? 63 : flip_thresh);
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream_ptr);
    mx7_c1_v5_build_hybrid_from_mx7_kernel<<<total_warps, 32, 0, s>>>(
        static_cast<const uint8_t*>(SLABS), static_cast<const uint16_t*>(FLIPS),
        static_cast<const int32_t*>(F_OFF), static_cast<const int32_t*>(S_OFF),
        static_cast<const int32_t*>(D_OFF), static_cast<uint32_t*>(SLAB_META),
        static_cast<uint16_t*>(SFLIPS), static_cast<uint32_t*>(BMASKS),
        total_warps, n_slabs, thr
    );
    return static_cast<int>(cudaGetLastError());
}

extern "C" int mx7_v5_panel_measure_from_mx7(
    const void* FLIPS, const void* F_OFF,
    void* panel_blocks_out, void* sparse_counts,
    int proj, int total_panels, int n_slabs, int flip_thresh, uintptr_t stream_ptr
) {
    if (total_panels <= 0 || n_slabs <= 0 || (proj != 0 && proj != 1)) return -1;
    const int thr = (flip_thresh < 0) ? 0 : ((flip_thresh > 63) ? 63 : flip_thresh);
    const int hdr_blks = (proj == 0) ? W13_WS3_HDR_BLOCKS : W2_WS3_HDR_BLOCKS;
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream_ptr);
    mx7_c1_v5_panel_measure_from_mx7_kernel<<<total_panels, 128, 0, s>>>(
        static_cast<const uint16_t*>(FLIPS), static_cast<const int32_t*>(F_OFF),
        static_cast<int32_t*>(panel_blocks_out), static_cast<int32_t*>(sparse_counts),
        total_panels, n_slabs, hdr_blks, thr
    );
    return static_cast<int>(cudaGetLastError());
}

extern "C" int mx7_v5_panel_build_from_mx7(
    const void* SLABS, const void* WS3, const void* FLIPS, const void* F_OFF,
    const void* PANEL_OFF, const void* S_OFF,
    void* PANEL_POOL, void* SLAB_META, void* SFLIPS,
    int proj, int total_panels, int n_slabs, int flip_thresh, uintptr_t stream_ptr
) {
    if (total_panels <= 0 || n_slabs <= 0 || (proj != 0 && proj != 1)) return -1;
    const int thr = (flip_thresh < 0) ? 0 : ((flip_thresh > 63) ? 63 : flip_thresh);
    const int hdr_blks = (proj == 0) ? W13_WS3_HDR_BLOCKS : W2_WS3_HDR_BLOCKS;
    const int ws_u32   = (proj == 0) ? W13_WS3_WARP_U32   : W2_WS3_WARP_U32;
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream_ptr);
    mx7_c1_v5_panel_build_from_mx7_kernel<<<total_panels, 128, 0, s>>>(
        static_cast<const uint8_t*>(SLABS), static_cast<const uint4*>(WS3),
        static_cast<const uint16_t*>(FLIPS), static_cast<const int32_t*>(F_OFF),
        static_cast<const int32_t*>(PANEL_OFF), static_cast<const int32_t*>(S_OFF),
        static_cast<uint8_t*>(PANEL_POOL), static_cast<uint32_t*>(SLAB_META),
        static_cast<uint16_t*>(SFLIPS),
        total_panels, n_slabs, hdr_blks, ws_u32, thr
    );
    return static_cast<int>(cudaGetLastError());
}


// Wave-V16 Entrypoints (FUSE_QDEQ + Wave-Adaptive + L2::evict_first + LPT Dynamic Queue + W2 Pre-Wait Hoist)
extern "C" int mx7_w13_c1_v16_fwd(
    const void* A, const void* AS,
    const void* SLABS, const void* WS3,
    const void* FLIPS, const void* F_OFF,
    const void* Ones, const void* Tiles,
    const void* TileMask, const void* TileCompactStart,
    const void* Gammas, void* Out, void* TileQdeqDone,
    int N, int K, int MAX_CTAS, int TILE, uint32_t base_ws,
    int zero_scatter, int fuse_situ, int fuse_qdeq,
    int bn_mode, int u_thresh,
    int pdl_wait_mode, int trig_dep, int evict_first,
    int lpt_mode, void* WorkQueue, uintptr_t stream_ptr
) {
    return launch_mx7_w13_c1_impl<0>(
        A, AS, SLABS, WS3, FLIPS, F_OFF, nullptr, nullptr, nullptr,
        Ones, Tiles, TileMask, TileCompactStart, Gammas, Out, TileQdeqDone,
        N, K, MAX_CTAS, TILE, base_ws, zero_scatter, fuse_situ, fuse_qdeq,
        bn_mode, u_thresh, pdl_wait_mode, trig_dep, evict_first, lpt_mode, WorkQueue, stream_ptr
    );
}

extern "C" int mx7_w2_c1_v16_fwd(
    const void* A, const void* AS,
    const void* SLABS, const void* WS3,
    const void* FLIPS, const void* F_OFF,
    const void* Ones, const void* Tiles, const void* TileCompactStart,
    void* Out, const void* Src2Compact, const void* NLivePtr,
    void* PanelDone, void* SendBuf,
    int N, int K, int MAX_CTAS, int TILE, int T, uint32_t base_ws,
    int as_row_major, int as_row_stride, int down_row_stride,
    int fuse_quant, int fuse_combine,
    int pdl_wait_mode, int trig_dep, int evict_first,
    int lpt_mode, void* WorkQueue, uintptr_t stream_ptr
) {
    return launch_mx7_w2_c1_impl<0>(
        A, AS, SLABS, WS3, FLIPS, F_OFF, nullptr, nullptr, nullptr,
        Ones, Tiles, TileCompactStart, Out, Src2Compact, NLivePtr,
        PanelDone, SendBuf,
        N, K, MAX_CTAS, TILE, T, base_ws, as_row_major, as_row_stride, down_row_stride,
        fuse_quant, fuse_combine, pdl_wait_mode, trig_dep, evict_first, lpt_mode, WorkQueue, stream_ptr
    );
}

extern "C" int mx7_w13_c1_v16_compat_fwd(
    const void* A, const void* AS,
    const void* SLABS, const void* WS3,
    const void* SLAB_META, const void* S_OFF, const void* D_OFF,
    const void* SFLIPS, const void* BMASKS,
    const void* Ones, const void* Tiles,
    const void* TileMask, const void* TileCompactStart,
    const void* Gammas, void* Out, void* TileQdeqDone,
    int N, int K, int MAX_CTAS, int TILE, uint32_t base_ws,
    int zero_scatter, int fuse_situ, int fuse_qdeq,
    int bn_mode, int u_thresh,
    int pdl_wait_mode, int trig_dep, int evict_first,
    int lpt_mode, void* WorkQueue, uintptr_t stream_ptr
) {
    return launch_mx7_w13_c1_impl<1>(
        A, AS, SLABS, WS3, SFLIPS, S_OFF, SLAB_META, BMASKS, D_OFF,
        Ones, Tiles, TileMask, TileCompactStart, Gammas, Out, TileQdeqDone,
        N, K, MAX_CTAS, TILE, base_ws, zero_scatter, fuse_situ, fuse_qdeq,
        bn_mode, u_thresh, pdl_wait_mode, trig_dep, evict_first, lpt_mode, WorkQueue, stream_ptr
    );
}

extern "C" int mx7_w2_c1_v16_compat_fwd(
    const void* A, const void* AS,
    const void* SLABS, const void* WS3,
    const void* SLAB_META, const void* S_OFF, const void* D_OFF,
    const void* SFLIPS, const void* BMASKS,
    const void* Ones, const void* Tiles, const void* TileCompactStart,
    void* Out, const void* Src2Compact, const void* NLivePtr,
    void* PanelDone, void* SendBuf,
    int N, int K, int MAX_CTAS, int TILE, int T, uint32_t base_ws,
    int as_row_major, int as_row_stride, int down_row_stride,
    int fuse_quant, int fuse_combine,
    int pdl_wait_mode, int trig_dep, int evict_first,
    int lpt_mode, void* WorkQueue, uintptr_t stream_ptr
) {
    return launch_mx7_w2_c1_impl<1>(
        A, AS, SLABS, WS3, SFLIPS, S_OFF, SLAB_META, BMASKS, D_OFF,
        Ones, Tiles, TileCompactStart, Out, Src2Compact, NLivePtr,
        PanelDone, SendBuf,
        N, K, MAX_CTAS, TILE, T, base_ws, as_row_major, as_row_stride, down_row_stride,
        fuse_quant, fuse_combine, pdl_wait_mode, trig_dep, evict_first, lpt_mode, WorkQueue, stream_ptr
    );
}

extern "C" int mx7_w13_c1_v16_panel_fwd(
    const void* A, const void* AS,
    const void* PANEL_POOL, const void* PANEL_OFF,
    const void* SLAB_META, const void* S_OFF, const void* SFLIPS,
    const void* Ones, const void* Tiles,
    const void* TileMask, const void* TileCompactStart,
    const void* Gammas, void* Out, void* TileQdeqDone,
    int N, int K, int MAX_CTAS, int TILE, uint32_t base_ws,
    int zero_scatter, int fuse_situ, int fuse_qdeq,
    int bn_mode, int u_thresh,
    int pdl_wait_mode, int trig_dep, int evict_first,
    int lpt_mode, void* WorkQueue, uintptr_t stream_ptr
) {
    return launch_mx7_w13_c1_impl<2>(
        A, AS, PANEL_POOL, nullptr, SFLIPS, S_OFF, SLAB_META, nullptr, PANEL_OFF,
        Ones, Tiles, TileMask, TileCompactStart, Gammas, Out, TileQdeqDone,
        N, K, MAX_CTAS, TILE, base_ws, zero_scatter, fuse_situ, fuse_qdeq,
        bn_mode, u_thresh, pdl_wait_mode, trig_dep, evict_first, lpt_mode, WorkQueue, stream_ptr
    );
}

extern "C" int mx7_w2_c1_v16_panel_fwd(
    const void* A, const void* AS,
    const void* PANEL_POOL, const void* PANEL_OFF,
    const void* SLAB_META, const void* S_OFF, const void* SFLIPS,
    const void* Ones, const void* Tiles, const void* TileCompactStart,
    void* Out, const void* Src2Compact, const void* NLivePtr,
    void* PanelDone, void* SendBuf,
    int N, int K, int MAX_CTAS, int TILE, int T, uint32_t base_ws,
    int as_row_major, int as_row_stride, int down_row_stride,
    int fuse_quant, int fuse_combine,
    int pdl_wait_mode, int trig_dep, int evict_first,
    int lpt_mode, void* WorkQueue, uintptr_t stream_ptr
) {
    return launch_mx7_w2_c1_impl<2>(
        A, AS, PANEL_POOL, nullptr, SFLIPS, S_OFF, SLAB_META, nullptr, PANEL_OFF,
        Ones, Tiles, TileCompactStart, Out, Src2Compact, NLivePtr,
        PanelDone, SendBuf,
        N, K, MAX_CTAS, TILE, T, base_ws, as_row_major, as_row_stride, down_row_stride,
        fuse_quant, fuse_combine, pdl_wait_mode, trig_dep, evict_first, lpt_mode, WorkQueue, stream_ptr
    );
}

// Wave-V14 Entrypoints (100% Drop-In Backward-Compatible Wrappers)
extern "C" int mx7_w13_c1_v14_fwd(
    const void* A, const void* AS,
    const void* SLABS, const void* WS3,
    const void* FLIPS, const void* F_OFF,
    const void* Ones, const void* Tiles,
    const void* TileMask, const void* TileCompactStart,
    const void* Gammas, void* Out, void* TileQdeqDone,
    int N, int K, int MAX_CTAS, int TILE, uint32_t base_ws,
    int zero_scatter, int fuse_situ, int fuse_qdeq,
    int bn_mode, int u_thresh,
    int pdl_wait_mode, int trig_dep, int evict_first, uintptr_t stream_ptr
) {
    return launch_mx7_w13_c1_impl<0>(
        A, AS, SLABS, WS3, FLIPS, F_OFF, nullptr, nullptr, nullptr,
        Ones, Tiles, TileMask, TileCompactStart, Gammas, Out, TileQdeqDone,
        N, K, MAX_CTAS, TILE, base_ws, zero_scatter, fuse_situ, fuse_qdeq,
        bn_mode, u_thresh, pdl_wait_mode, trig_dep, evict_first, 0, nullptr, stream_ptr
    );
}

extern "C" int mx7_w2_c1_v14_fwd(
    const void* A, const void* AS,
    const void* SLABS, const void* WS3,
    const void* FLIPS, const void* F_OFF,
    const void* Ones, const void* Tiles, const void* TileCompactStart,
    void* Out, const void* Src2Compact, const void* NLivePtr,
    void* PanelDone, void* SendBuf,
    int N, int K, int MAX_CTAS, int TILE, int T, uint32_t base_ws,
    int as_row_major, int as_row_stride, int down_row_stride,
    int fuse_quant, int fuse_combine,
    int pdl_wait_mode, int trig_dep, int evict_first, uintptr_t stream_ptr
) {
    return launch_mx7_w2_c1_impl<0>(
        A, AS, SLABS, WS3, FLIPS, F_OFF, nullptr, nullptr, nullptr,
        Ones, Tiles, TileCompactStart, Out, Src2Compact, NLivePtr,
        PanelDone, SendBuf,
        N, K, MAX_CTAS, TILE, T, base_ws, as_row_major, as_row_stride, down_row_stride,
        fuse_quant, fuse_combine, pdl_wait_mode, trig_dep, evict_first, 0, nullptr, stream_ptr
    );
}

extern "C" int mx7_w13_c1_v14_compat_fwd(
    const void* A, const void* AS,
    const void* SLABS, const void* WS3,
    const void* SLAB_META, const void* S_OFF, const void* D_OFF,
    const void* SFLIPS, const void* BMASKS,
    const void* Ones, const void* Tiles,
    const void* TileMask, const void* TileCompactStart,
    const void* Gammas, void* Out, void* TileQdeqDone,
    int N, int K, int MAX_CTAS, int TILE, uint32_t base_ws,
    int zero_scatter, int fuse_situ, int fuse_qdeq,
    int bn_mode, int u_thresh,
    int pdl_wait_mode, int trig_dep, int evict_first, uintptr_t stream_ptr
) {
    return launch_mx7_w13_c1_impl<1>(
        A, AS, SLABS, WS3, SFLIPS, S_OFF, SLAB_META, BMASKS, D_OFF,
        Ones, Tiles, TileMask, TileCompactStart, Gammas, Out, TileQdeqDone,
        N, K, MAX_CTAS, TILE, base_ws, zero_scatter, fuse_situ, fuse_qdeq,
        bn_mode, u_thresh, pdl_wait_mode, trig_dep, evict_first, 0, nullptr, stream_ptr
    );
}

extern "C" int mx7_w2_c1_v14_compat_fwd(
    const void* A, const void* AS,
    const void* SLABS, const void* WS3,
    const void* SLAB_META, const void* S_OFF, const void* D_OFF,
    const void* SFLIPS, const void* BMASKS,
    const void* Ones, const void* Tiles, const void* TileCompactStart,
    void* Out, const void* Src2Compact, const void* NLivePtr,
    void* PanelDone, void* SendBuf,
    int N, int K, int MAX_CTAS, int TILE, int T, uint32_t base_ws,
    int as_row_major, int as_row_stride, int down_row_stride,
    int fuse_quant, int fuse_combine,
    int pdl_wait_mode, int trig_dep, int evict_first, uintptr_t stream_ptr
) {
    return launch_mx7_w2_c1_impl<1>(
        A, AS, SLABS, WS3, SFLIPS, S_OFF, SLAB_META, BMASKS, D_OFF,
        Ones, Tiles, TileCompactStart, Out, Src2Compact, NLivePtr,
        PanelDone, SendBuf,
        N, K, MAX_CTAS, TILE, T, base_ws, as_row_major, as_row_stride, down_row_stride,
        fuse_quant, fuse_combine, pdl_wait_mode, trig_dep, evict_first, 0, nullptr, stream_ptr
    );
}

extern "C" int mx7_w13_c1_v14_panel_fwd(
    const void* A, const void* AS,
    const void* PANEL_POOL, const void* PANEL_OFF,
    const void* SLAB_META, const void* S_OFF, const void* SFLIPS,
    const void* Ones, const void* Tiles,
    const void* TileMask, const void* TileCompactStart,
    const void* Gammas, void* Out, void* TileQdeqDone,
    int N, int K, int MAX_CTAS, int TILE, uint32_t base_ws,
    int zero_scatter, int fuse_situ, int fuse_qdeq,
    int bn_mode, int u_thresh,
    int pdl_wait_mode, int trig_dep, int evict_first, uintptr_t stream_ptr
) {
    return launch_mx7_w13_c1_impl<2>(
        A, AS, PANEL_POOL, nullptr, SFLIPS, S_OFF, SLAB_META, nullptr, PANEL_OFF,
        Ones, Tiles, TileMask, TileCompactStart, Gammas, Out, TileQdeqDone,
        N, K, MAX_CTAS, TILE, base_ws, zero_scatter, fuse_situ, fuse_qdeq,
        bn_mode, u_thresh, pdl_wait_mode, trig_dep, evict_first, 0, nullptr, stream_ptr
    );
}

extern "C" int mx7_w2_c1_v14_panel_fwd(
    const void* A, const void* AS,
    const void* PANEL_POOL, const void* PANEL_OFF,
    const void* SLAB_META, const void* S_OFF, const void* SFLIPS,
    const void* Ones, const void* Tiles, const void* TileCompactStart,
    void* Out, const void* Src2Compact, const void* NLivePtr,
    void* PanelDone, void* SendBuf,
    int N, int K, int MAX_CTAS, int TILE, int T, uint32_t base_ws,
    int as_row_major, int as_row_stride, int down_row_stride,
    int fuse_quant, int fuse_combine,
    int pdl_wait_mode, int trig_dep, int evict_first, uintptr_t stream_ptr
) {
    return launch_mx7_w2_c1_impl<2>(
        A, AS, PANEL_POOL, nullptr, SFLIPS, S_OFF, SLAB_META, nullptr, PANEL_OFF,
        Ones, Tiles, TileCompactStart, Out, Src2Compact, NLivePtr,
        PanelDone, SendBuf,
        N, K, MAX_CTAS, TILE, T, base_ws, as_row_major, as_row_stride, down_row_stride,
        fuse_quant, fuse_combine, pdl_wait_mode, trig_dep, evict_first, 0, nullptr, stream_ptr
    );
}

// Backward-compatible C=1 v4 & v5 entrypoints (delegate cleanly to v14 kernels)
extern "C" int mx7_w13_c1_v4_fwd(
    const void* A, const void* AS,
    const void* SLABS, const void* WS3,
    const void* FLIPS, const void* F_OFF,
    const void* Ones, const void* Tiles,
    const void* TileMask, const void* TileCompactStart,
    const void* Gammas, void* Out,
    int N, int K, int MAX_CTAS, int TILE, uint32_t base_ws,
    int zero_scatter, int fuse_situ,
    int bn_mode, int u_thresh,
    int pdl_wait_mode, int trig_dep, uintptr_t stream_ptr
) {
    return mx7_w13_c1_v14_fwd(
        A, AS, SLABS, WS3, FLIPS, F_OFF,
        Ones, Tiles, TileMask, TileCompactStart, Gammas, Out, nullptr,
        N, K, MAX_CTAS, TILE, base_ws, zero_scatter, fuse_situ, 0,
        bn_mode, u_thresh, pdl_wait_mode, trig_dep, 1, stream_ptr
    );
}

extern "C" int mx7_w2_c1_v4_fwd(
    const void* A, const void* AS,
    const void* SLABS, const void* WS3,
    const void* FLIPS, const void* F_OFF,
    const void* Ones, const void* Tiles, const void* TileCompactStart,
    void* Out, const void* Src2Compact, const void* NLivePtr,
    void* PanelDone, void* SendBuf,
    int N, int K, int MAX_CTAS, int TILE, int T, uint32_t base_ws,
    int as_row_major, int as_row_stride, int down_row_stride,
    int fuse_quant, int fuse_combine,
    int pdl_wait_mode, int trig_dep, uintptr_t stream_ptr
) {
    return mx7_w2_c1_v14_fwd(
        A, AS, SLABS, WS3, FLIPS, F_OFF,
        Ones, Tiles, TileCompactStart, Out, Src2Compact, NLivePtr,
        PanelDone, SendBuf,
        N, K, MAX_CTAS, TILE, T, base_ws, as_row_major, as_row_stride, down_row_stride,
        fuse_quant, fuse_combine, pdl_wait_mode, trig_dep, 1, stream_ptr
    );
}

extern "C" int mx7_w13_c1_v5_fwd(
    const void* A, const void* AS,
    const void* SLABS, const void* WS3,
    const void* SLAB_META, const void* S_OFF, const void* D_OFF,
    const void* SFLIPS, const void* BMASKS,
    const void* Ones, const void* Tiles,
    const void* TileMask, const void* TileCompactStart,
    const void* Gammas, void* Out,
    int N, int K, int MAX_CTAS, int TILE, uint32_t base_ws,
    int zero_scatter, int fuse_situ,
    int bn_mode, int u_thresh,
    int pdl_wait_mode, int trig_dep, uintptr_t stream_ptr
) {
    return mx7_w13_c1_v14_compat_fwd(
        A, AS, SLABS, WS3, SLAB_META, S_OFF, D_OFF, SFLIPS, BMASKS,
        Ones, Tiles, TileMask, TileCompactStart, Gammas, Out, nullptr,
        N, K, MAX_CTAS, TILE, base_ws, zero_scatter, fuse_situ, 0,
        bn_mode, u_thresh, pdl_wait_mode, trig_dep, 1, stream_ptr
    );
}

extern "C" int mx7_w2_c1_v5_fwd(
    const void* A, const void* AS,
    const void* SLABS, const void* WS3,
    const void* SLAB_META, const void* S_OFF, const void* D_OFF,
    const void* SFLIPS, const void* BMASKS,
    const void* Ones, const void* Tiles, const void* TileCompactStart,
    void* Out, const void* Src2Compact, const void* NLivePtr,
    void* PanelDone, void* SendBuf,
    int N, int K, int MAX_CTAS, int TILE, int T, uint32_t base_ws,
    int as_row_major, int as_row_stride, int down_row_stride,
    int fuse_quant, int fuse_combine,
    int pdl_wait_mode, int trig_dep, uintptr_t stream_ptr
) {
    return mx7_w2_c1_v14_compat_fwd(
        A, AS, SLABS, WS3, SLAB_META, S_OFF, D_OFF, SFLIPS, BMASKS,
        Ones, Tiles, TileCompactStart, Out, Src2Compact, NLivePtr,
        PanelDone, SendBuf,
        N, K, MAX_CTAS, TILE, T, base_ws, as_row_major, as_row_stride, down_row_stride,
        fuse_quant, fuse_combine, pdl_wait_mode, trig_dep, 1, stream_ptr
    );
}

extern "C" int mx7_w13_c1_v5_panel_fwd(
    const void* A, const void* AS,
    const void* PANEL_POOL, const void* PANEL_OFF,
    const void* SLAB_META, const void* S_OFF, const void* SFLIPS,
    const void* Ones, const void* Tiles,
    const void* TileMask, const void* TileCompactStart,
    const void* Gammas, void* Out,
    int N, int K, int MAX_CTAS, int TILE, uint32_t base_ws,
    int zero_scatter, int fuse_situ,
    int bn_mode, int u_thresh,
    int pdl_wait_mode, int trig_dep, uintptr_t stream_ptr
) {
    return mx7_w13_c1_v14_panel_fwd(
        A, AS, PANEL_POOL, PANEL_OFF, SLAB_META, S_OFF, SFLIPS,
        Ones, Tiles, TileMask, TileCompactStart, Gammas, Out, nullptr,
        N, K, MAX_CTAS, TILE, base_ws, zero_scatter, fuse_situ, 0,
        bn_mode, u_thresh, pdl_wait_mode, trig_dep, 1, stream_ptr
    );
}

extern "C" int mx7_w2_c1_v5_panel_fwd(
    const void* A, const void* AS,
    const void* PANEL_POOL, const void* PANEL_OFF,
    const void* SLAB_META, const void* S_OFF, const void* SFLIPS,
    const void* Ones, const void* Tiles, const void* TileCompactStart,
    void* Out, const void* Src2Compact, const void* NLivePtr,
    void* PanelDone, void* SendBuf,
    int N, int K, int MAX_CTAS, int TILE, int T, uint32_t base_ws,
    int as_row_major, int as_row_stride, int down_row_stride,
    int fuse_quant, int fuse_combine,
    int pdl_wait_mode, int trig_dep, uintptr_t stream_ptr
) {
    return mx7_w2_c1_v14_panel_fwd(
        A, AS, PANEL_POOL, PANEL_OFF, SLAB_META, S_OFF, SFLIPS,
        Ones, Tiles, TileCompactStart, Out, Src2Compact, NLivePtr,
        PanelDone, SendBuf,
        N, K, MAX_CTAS, TILE, T, base_ws, as_row_major, as_row_stride, down_row_stride,
        fuse_quant, fuse_combine, pdl_wait_mode, trig_dep, 1, stream_ptr
    );
}
