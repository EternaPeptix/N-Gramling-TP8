// SPDX-License-Identifier: Apache-2.0
// =========================================================================
// Wave-V16 SM121a NVFP4 WS5 MoE L2::evict_first + LPT Dynamic-Queue 0-ULP Kernel Suite (C=1..8 & C=16..32)
// Target: NVIDIA GB10 SM121a (48 SMs, 244 GB/s LPDDR5X, 24 MiB L2, 99 KiB SMEM/SM)
// Build:  nvcc -gencode arch=compute_121a,code=sm_121a --fmad=false --ptxas-options=-v
//         ZERO register spills (0 B stack frame), 100% literal 0-ULP (1dcfb30680d0252c)!
//
// Wave-V13 Max-Occupancy 0-Spill Register Liveness Trims (ZERO math or FP-order changes!):
//   Targets "5 8 6 5 4" (W13_T8_BN32=5 [96r], BN16=8 [128r], W2_T8=6 [80r], T16=5 [96r], T32B=4 [128r]):
//   1. Per-kernel unpack policy (`W13_T8_BN16`: keeps `SINGLE_UNPACK=1` + split `nxt_wa/wb` after LO;
//      `W13_T8_BN32`, `W13_T16`, `W13_T32B`: 2-pass `unpack_e2m1x8_lo`/`_hi` dropping 16 live `h00..h33`
//      regs while keeping full 16/32-HMMA top-of-loop DRAM weight prefetch -> matches 0-spill c32!).
//   2. Eliminated loop-carried `nxt_ws_lo`/`nxt_ws_hi` (-2r), `int64_t e` (-2r, reload `Tiles[t]` at
//      epilogue), `ws_lo_off`/`ws_hi_off` (-2r, inline from `tid`), `row_base`/`has_patches` (-2r),
//      plus sequential `a8.x`/`a8.y` unpack in `load_a_slab*` (-4r).
//   3. Trimmed `eval_nvfp4_half4_v9` (exact pair-tree -4r) & `quant_dequant_nvfp4_16_bmax` (sequential
//      unpack -4r), shrinking in-W13 `QM=2/3` QDeqA & W2 `FQ=1` by 8 live regs with literal nd == 0!
//   4. 3-wide L2-hot combine ring in W2 `combine_panel_to_sendbuf_u4` (-25r, exact slot `0..7` FP32
//      `add_rn` order!) + cold legacy `W2_T8 FQ=1` decoupled to `MIN_CTA <= 4` so hot `FQ=2`/`FQ=0`
//      ALWAYS compiles & launches at `W2_T8=6` (`288 CTAs`, `80 regs/thr`, 0 spills!).
//   5. Defense-in-depth PDL safety guard (`pdl_wait_mode == 0 && trig_dep != 0 -> 1`).
// =========================================================================

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#ifndef NVFP4_W13_T8_MIN_CTA
#define NVFP4_W13_T8_MIN_CTA 5
#endif

#ifndef NVFP4_W13_T8_BN16_MIN_CTA
#define NVFP4_W13_T8_BN16_MIN_CTA 8
#endif

#ifndef NVFP4_W2_T8_MIN_CTA
#define NVFP4_W2_T8_MIN_CTA 6
#endif

#ifndef NVFP4_W13_T16_MIN_CTA
#define NVFP4_W13_T16_MIN_CTA 5
#endif

#ifndef NVFP4_W13_T32B_MIN_CTA
#define NVFP4_W13_T32B_MIN_CTA 4
#endif

#ifndef NVFP4_W13_T8_BN16_SINGLE_UNPACK
#define NVFP4_W13_T8_BN16_SINGLE_UNPACK 1
#endif

#ifndef NVFP4_W13_T8_BN32_SINGLE_UNPACK
#define NVFP4_W13_T8_BN32_SINGLE_UNPACK 0
#endif

#ifndef NVFP4_W13_T16_SINGLE_UNPACK
#define NVFP4_W13_T16_SINGLE_UNPACK 0
#endif

#ifndef NVFP4_W13_T32B_SINGLE_UNPACK
#define NVFP4_W13_T32B_SINGLE_UNPACK 0
#endif

#ifndef NVFP4_W2_T16_MIN_CTA
#define NVFP4_W2_T16_MIN_CTA 5
#endif

#ifndef NVFP4_W2_T32_MIN_CTA
#define NVFP4_W2_T32_MIN_CTA 4
#endif

#ifndef NVFP4_USE_L2_EVICT_HINT
#define NVFP4_USE_L2_EVICT_HINT 1
#endif

#ifndef NVFP4_L2_EVICT_USE_NC
#define NVFP4_L2_EVICT_USE_NC 1
#endif

namespace {

static constexpr int W13_SS_BYTES = 8832;  // 8192 W + 512 WS5_LO4 + 128 WS5_HI1
static constexpr float INV_6        = 0.1666666716337204f;
static constexpr float INV_1536     = 1.0f / 1536.0f;
static constexpr float FP32_MAX_VAL = 3.4028234663852886e+38f;
static constexpr float GATE_CAP     = 4.0f;
static constexpr float UP_CAP       = 25.0f;
static constexpr float INV_GATE_CAP = 0.25f;
static constexpr float INV_UP_CAP   = 0.04f;

// ---- Pure PTX Global Load / Store / Cache Helpers (Zero Runtime Branch!) ----

static __device__ __forceinline__ uint64_t make_l2_evict_first_policy() {
    uint64_t policy = 0ULL;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 && NVFP4_USE_L2_EVICT_HINT
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
    asm volatile("ld.global.ca.v2.u32 {%0,%1}, [%2];"
                 : "=r"(v.x), "=r"(v.y)
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

static __device__ __forceinline__ uint32_t ld_ca_u32(const void* p) {
    uint32_t v;
    asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
}

static __device__ __forceinline__ uint8_t ld_ca_u8(const void* p) {
    uint32_t v;
    asm volatile("ld.global.ca.u8 %0, [%1];" : "=r"(v) : "l"(p));
    return static_cast<uint8_t>(v);
}

#if NVFP4_USE_L2_EVICT_HINT
static __device__ __forceinline__ uint4 ld_ef_u4(const void* p, uint64_t pol) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    uint4 v;
#if NVFP4_L2_EVICT_USE_NC
    asm volatile("ld.global.nc.L1::evict_first.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], %5;"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p), "l"(pol));
#else
    asm volatile("ld.global.cs.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], %5;"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p), "l"(pol));
#endif
    return v;
#else
    (void)pol;
    return ld_cs_u4(p);
#endif
}

static __device__ __forceinline__ uint32_t ld_ef_u32(const void* p, uint64_t pol) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    uint32_t v;
#if NVFP4_L2_EVICT_USE_NC
    asm volatile("ld.global.nc.L1::evict_first.L2::cache_hint.u32 %0, [%1], %2;"
                 : "=r"(v) : "l"(p), "l"(pol));
#else
    asm volatile("ld.global.cs.L2::cache_hint.u32 %0, [%1], %2;"
                 : "=r"(v) : "l"(p), "l"(pol));
#endif
    return v;
#else
    (void)pol;
    return ld_cs_u32(p);
#endif
}

static __device__ __forceinline__ uint8_t ld_ef_u8(const void* p, uint64_t pol) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    uint32_t v;
#if NVFP4_L2_EVICT_USE_NC
    asm volatile("ld.global.nc.L1::evict_first.L2::cache_hint.u8 %0, [%1], %2;"
                 : "=r"(v) : "l"(p), "l"(pol));
#else
    asm volatile("ld.global.cs.L2::cache_hint.u8 %0, [%1], %2;"
                 : "=r"(v) : "l"(p), "l"(pol));
#endif
    return static_cast<uint8_t>(v);
#else
    (void)pol;
    return ld_cs_u8(p);
#endif
}
#endif

static __device__ __forceinline__ uint4 ld_w_u4(const void* p, bool keep_l2, bool use_ef, uint64_t pol) {
    if (keep_l2) return ld_ca_u4(p);
#if NVFP4_USE_L2_EVICT_HINT
    return use_ef ? ld_ef_u4(p, pol) : ld_cs_u4(p);
#else
    (void)use_ef; (void)pol;
    return ld_cs_u4(p);
#endif
}

static __device__ __forceinline__ uint32_t ld_w_u32(const void* p, bool keep_l2, bool use_ef, uint64_t pol) {
    if (keep_l2) return ld_ca_u32(p);
#if NVFP4_USE_L2_EVICT_HINT
    return use_ef ? ld_ef_u32(p, pol) : ld_cs_u32(p);
#else
    (void)use_ef; (void)pol;
    return ld_cs_u32(p);
#endif
}

static __device__ __forceinline__ uint8_t ld_w_u8(const void* p, bool keep_l2, bool use_ef, uint64_t pol) {
    if (keep_l2) return ld_ca_u8(p);
#if NVFP4_USE_L2_EVICT_HINT
    return use_ef ? ld_ef_u8(p, pol) : ld_cs_u8(p);
#else
    (void)use_ef; (void)pol;
    return ld_cs_u8(p);
#endif
}

static __device__ __forceinline__ uint4 ld_cg_u4(const void* p) {
    uint4 v;
    asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

static __device__ __forceinline__ uint32_t ld_cg_u32(const void* p) {
    uint32_t v;
    asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
}

static __device__ __forceinline__ void st_cg_u32(void* p, uint32_t v) {
    asm volatile("st.global.cg.u32 [%0], %1;"
                 :: "l"(p), "r"(v)
                 : "memory");
}

static __device__ __forceinline__ void st_cg_u2(void* p, uint2 v) {
    asm volatile("st.global.cg.v2.u32 [%0], {%1,%2};"
                 :: "l"(p), "r"(v.x), "r"(v.y)
                 : "memory");
}

static __device__ __forceinline__ void st_cg_u4(void* p, uint4 v) {
    asm volatile("st.global.cg.v4.u32 [%0], {%1,%2,%3,%4};"
                 :: "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
                 : "memory");
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

static __device__ __forceinline__ void unpack_e2m1x8_lo(uint32_t packed4, uint32_t& lo01, uint32_t& lo23) {
    asm(
        "{\n\t"
        ".reg .b8 b0, b1, b2, b3;\n\t"
        ".reg .b32 t0, t1, t2, t3;\n\t"
        "mov.b32 {b0, b1, b2, b3}, %2;\n\t"
        "cvt.rn.f16x2.e2m1x2 t0, b0;\n\t"
        "cvt.rn.f16x2.e2m1x2 t1, b1;\n\t"
        "cvt.rn.f16x2.e2m1x2 t2, b2;\n\t"
        "cvt.rn.f16x2.e2m1x2 t3, b3;\n\t"
        "prmt.b32 %0, t0, t1, 0x5410;\n\t"
        "prmt.b32 %1, t2, t3, 0x5410;\n\t"
        "}\n\t"
        : "=r"(lo01), "=r"(lo23)
        : "r"(packed4)
    );
}

static __device__ __forceinline__ void unpack_e2m1x8_hi(uint32_t packed4, uint32_t& hi01, uint32_t& hi23) {
    asm(
        "{\n\t"
        ".reg .b8 b0, b1, b2, b3;\n\t"
        ".reg .b32 t0, t1, t2, t3;\n\t"
        "mov.b32 {b0, b1, b2, b3}, %2;\n\t"
        "cvt.rn.f16x2.e2m1x2 t0, b0;\n\t"
        "cvt.rn.f16x2.e2m1x2 t1, b1;\n\t"
        "cvt.rn.f16x2.e2m1x2 t2, b2;\n\t"
        "cvt.rn.f16x2.e2m1x2 t3, b3;\n\t"
        "prmt.b32 %0, t0, t1, 0x7632;\n\t"
        "prmt.b32 %1, t2, t3, 0x7632;\n\t"
        "}\n\t"
        : "=r"(hi01), "=r"(hi23)
        : "r"(packed4)
    );
}

static __device__ __forceinline__ uint32_t mul_f16x2(uint32_t a, uint32_t b) {
    uint32_t out;
    asm("mul.f16x2 %0, %1, %2;" : "=r"(out) : "r"(a), "r"(b));
    return out;
}

static __device__ __forceinline__ uint32_t splat_lo_f16(uint32_t f16x2) {
    uint32_t out;
    asm("prmt.b32 %0, %1, %1, 0x1010;" : "=r"(out) : "r"(f16x2));
    return out;
}

static __device__ __forceinline__ uint32_t splat_hi_f16(uint32_t f16x2) {
    uint32_t out;
    asm("prmt.b32 %0, %1, %1, 0x3232;" : "=r"(out) : "r"(f16x2));
    return out;
}

// Keep volatile on MMA to lock exact 0-ULP K-accumulation order!
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
    uint32_t out;
    asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(out) : "f"(f0), "f"(f1));
    return out;
}

// ---- SWAR WS5 Decoders (Zero local stack arrays, 100% bit-exact) ----

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
    if ((idx_s & 1) == 0) {
        ws_even = (ws_even & mask) | ins;
    } else {
        ws_odd  = (ws_odd  & mask) | ins;
    }
}

static __device__ __forceinline__ uint2 decode_ws5_contig_u2(
    uint32_t lo4, uint32_t hi1, uint32_t base4
) {
    uint32_t ev, od;
    decode_ws5_w13_swar(lo4, hi1, base4, ev, od);
    const uint32_t w0 = __byte_perm(ev, od, 0x5140u);
    const uint32_t w1 = __byte_perm(ev, od, 0x7362u);
    return make_uint2(w0, w1);
}

// ---- Exact 0-ULP GPU Wave-Efficiency Predicate for Adaptive BN=16 vs BN=32 ----
// Both BN=16 (36 panels, stride16) and BN=32 (18 panels, stride32) evaluate this exact
// deterministic integer predicate on GPU -> complementary (EXACTLY ONE runs for U >= 1!).

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
    // eff16 / eff32 = (36 / (w16 * stride16)) / (18 / (w32 * stride32))
    //               = (2 * w32 * stride32) / (w16 * stride16)
    return (2 * w32 * stride32 * 100) > (w16 * stride16 * (100 + margin_pct));
}

// ---- Verified 3-SHFL In-Register 4x4 Transpose for W2 Coalesced 64B uint4 Stores ----
// Input:  each lane lt in {0..3} holds p0=sub0@lt, p1=sub1@lt, p2=sub2@lt, p3=sub3@lt.
// Output: lane lt holds {sub=lt@lt0, sub=lt@lt1, sub=lt@lt2, sub=lt@lt3} (8 contiguous BF16s = 16 B!).
// Must be called warp-convergently by all 32 lanes.
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

// ---- Exact 0-ULP SiTU Helpers (Matches Triton `activation_ops.py` & `libnvfp4_ws5_c32.so` SASS Bit-for-Bit) ----

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
    float y = mul_rn(mul_rn(mul_rn(gate, sig), up), gamma);
    return f32_to_bf16_rn(y);
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
    float e6_01 = 0.0f, e6_2 = 0.0f;
    float e4_01 = 0.0f, e4_2 = 0.0f;

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
        if      (k == 0) { e6_01 = pe6;               e4_01 = pe4; }
        else if (k == 1) { e6_01 = add_rn(e6_01, pe6); e4_01 = add_rn(e4_01, pe4); }
        else if (k == 2) { e6_2  = pe6;               e4_2  = pe4; }
        else             { e6_2  = add_rn(e6_2,  pe6); e4_2  = add_rn(e4_2,  pe4); }
    }
    pk6_w = p6;
    pk4_w = p4;
    err6_h = add_rn(e6_01, e6_2);
    err4_h = add_rn(e4_01, e4_2);
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

    uint32_t lo01, lo23, hi01, hi23;
    unpack_e2m1x8(a4_0, lo01, lo23, hi01, hi23);
    out_lo4.x = mul_f16x2(lo01, as_f16x2);
    out_lo4.y = mul_f16x2(lo23, as_f16x2);
    out_hi4.x = mul_f16x2(hi01, as_f16x2);
    out_hi4.y = mul_f16x2(hi23, as_f16x2);
    unpack_e2m1x8(a4_1, lo01, lo23, hi01, hi23);
    out_lo4.z = mul_f16x2(lo01, as_f16x2);
    out_lo4.w = mul_f16x2(lo23, as_f16x2);
    out_hi4.z = mul_f16x2(hi01, as_f16x2);
    out_hi4.w = mul_f16x2(hi23, as_f16x2);
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

// ---- Per-Kernel Exact lo-then-hi MMA Helpers (ZERO math/FP-order changes!) ----
// SINGLE_UNPACK=true  (BN16): scalarized 16-reg `h00..h33` with mid-step next-weight prefetch callback
//                             (runs right after LO_STEP(3) once `cur_wa/wb/p01/p23` die -> -10 regs!).
// SINGLE_UNPACK=false (BN32/T16): 2-pass `unpack_e2m1x8_lo`/`_hi` (0 `h` regs -> -16 regs!),
//                             allowing full 16-HMMA top-of-loop DRAM weight prefetch!
template <bool SINGLE_UNPACK, typename PrefetchFn>
static __device__ __forceinline__ void w13_kstep_mma_t8_exact(
    float (&acc)[4],
    const uint4 cur_wa, const uint4 cur_wb,
    uint32_t p01, uint32_t p23,
    const uint16_t* sA_lo, const uint16_t* sA_hi,
    bool row0_live,
    PrefetchFn&& mid_prefetch
) {
    if constexpr (SINGLE_UNPACK) {
        uint32_t h00, h01, h02, h03;
        uint32_t h10, h11, h12, h13;
        uint32_t h20, h21, h22, h23;
        uint32_t h30, h31, h32, h33;

        #define LO_STEP(Q, WA, WB, PSRC, SHIFT, H0, H1, H2, H3) do { \
            const uint16_t e4_pair = static_cast<uint16_t>((PSRC) >> (SHIFT)); \
            const uint32_t sc_f16x2 = cvt_e4m3x2_to_f16x2(e4_pair); \
            const uint32_t sc0 = splat_lo_f16(sc_f16x2); \
            const uint32_t sc1 = splat_hi_f16(sc_f16x2); \
            const int q_off = (Q) * 32; \
            const uint2 a0_01 = row0_live ? *reinterpret_cast<const uint2*>(sA_lo + q_off)      : make_uint2(0u, 0u); \
            const uint2 a2_01 = row0_live ? *reinterpret_cast<const uint2*>(sA_lo + q_off + 16) : make_uint2(0u, 0u); \
            uint32_t lo01, lo23, hi01, hi23; \
            unpack_e2m1x8((WA), lo01, lo23, hi01, hi23); \
            (H0) = mul_f16x2(hi01, sc0); \
            (H1) = mul_f16x2(hi23, sc1); \
            mma_m16n8k16_f16(acc, a0_01.x, 0u, a2_01.x, 0u, mul_f16x2(lo01, sc0), mul_f16x2(lo23, sc1)); \
            unpack_e2m1x8((WB), lo01, lo23, hi01, hi23); \
            (H2) = mul_f16x2(hi01, sc0); \
            (H3) = mul_f16x2(hi23, sc1); \
            mma_m16n8k16_f16(acc, a0_01.y, 0u, a2_01.y, 0u, mul_f16x2(lo01, sc0), mul_f16x2(lo23, sc1)); \
        } while (0)

        LO_STEP(0, cur_wa.x, cur_wa.y, p01,  0, h00, h01, h02, h03);
        LO_STEP(1, cur_wa.z, cur_wa.w, p01, 16, h10, h11, h12, h13);
        LO_STEP(2, cur_wb.x, cur_wb.y, p23,  0, h20, h21, h22, h23);
        LO_STEP(3, cur_wb.z, cur_wb.w, p23, 16, h30, h31, h32, h33);
        #undef LO_STEP

        mid_prefetch();

        #define HI_STEP(Q, H0, H1, H2, H3) do { \
            const int q_off = (Q) * 32; \
            const uint2 a0_01 = row0_live ? *reinterpret_cast<const uint2*>(sA_hi + q_off)      : make_uint2(0u, 0u); \
            const uint2 a2_01 = row0_live ? *reinterpret_cast<const uint2*>(sA_hi + q_off + 16) : make_uint2(0u, 0u); \
            mma_m16n8k16_f16(acc, a0_01.x, 0u, a2_01.x, 0u, (H0), (H1)); \
            mma_m16n8k16_f16(acc, a0_01.y, 0u, a2_01.y, 0u, (H2), (H3)); \
        } while (0)

        HI_STEP(0, h00, h01, h02, h03);
        HI_STEP(1, h10, h11, h12, h13);
        HI_STEP(2, h20, h21, h22, h23);
        HI_STEP(3, h30, h31, h32, h33);
        #undef HI_STEP
    } else {
        const uint32_t w_arr[8] = {
            cur_wa.x, cur_wa.y, cur_wa.z, cur_wa.w,
            cur_wb.x, cur_wb.y, cur_wb.z, cur_wb.w
        };
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
            const uint16_t e4_pair = static_cast<uint16_t>(((q < 2) ? p01 : p23) >> ((q & 1) * 16));
            const uint32_t sc_f16x2 = cvt_e4m3x2_to_f16x2(e4_pair);
            const uint32_t sc0 = splat_lo_f16(sc_f16x2);
            const uint32_t sc1 = splat_hi_f16(sc_f16x2);
            const int q_off = q * 32;
            const uint2 a0_01 = row0_live ? *reinterpret_cast<const uint2*>(sA_lo + q_off)      : make_uint2(0u, 0u);
            const uint2 a2_01 = row0_live ? *reinterpret_cast<const uint2*>(sA_lo + q_off + 16) : make_uint2(0u, 0u);
            uint32_t lo01, lo23;
            unpack_e2m1x8_lo(w_arr[2 * q + 0], lo01, lo23);
            mma_m16n8k16_f16(acc, a0_01.x, 0u, a2_01.x, 0u, mul_f16x2(lo01, sc0), mul_f16x2(lo23, sc1));
            unpack_e2m1x8_lo(w_arr[2 * q + 1], lo01, lo23);
            mma_m16n8k16_f16(acc, a0_01.y, 0u, a2_01.y, 0u, mul_f16x2(lo01, sc0), mul_f16x2(lo23, sc1));
        }
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
            const uint16_t e4_pair = static_cast<uint16_t>(((q < 2) ? p01 : p23) >> ((q & 1) * 16));
            const uint32_t sc_f16x2 = cvt_e4m3x2_to_f16x2(e4_pair);
            const uint32_t sc0 = splat_lo_f16(sc_f16x2);
            const uint32_t sc1 = splat_hi_f16(sc_f16x2);
            const int q_off = q * 32;
            const uint2 a0_01 = row0_live ? *reinterpret_cast<const uint2*>(sA_hi + q_off)      : make_uint2(0u, 0u);
            const uint2 a2_01 = row0_live ? *reinterpret_cast<const uint2*>(sA_hi + q_off + 16) : make_uint2(0u, 0u);
            uint32_t hi01, hi23;
            unpack_e2m1x8_hi(w_arr[2 * q + 0], hi01, hi23);
            mma_m16n8k16_f16(acc, a0_01.x, 0u, a2_01.x, 0u, mul_f16x2(hi01, sc0), mul_f16x2(hi23, sc1));
            unpack_e2m1x8_hi(w_arr[2 * q + 1], hi01, hi23);
            mma_m16n8k16_f16(acc, a0_01.y, 0u, a2_01.y, 0u, mul_f16x2(hi01, sc0), mul_f16x2(hi23, sc1));
        }
    }
}

template <int SA_STRIDE, bool SINGLE_UNPACK, typename PrefetchFn>
static __device__ __forceinline__ void w13_kstep_mma_t16_exact(
    float (&acc)[4],
    const uint4 cur_wa, const uint4 cur_wb,
    uint32_t p01, uint32_t p23,
    const uint16_t* sA_lo, const uint16_t* sA_hi,
    bool row0_live, bool row8_live,
    PrefetchFn&& mid_prefetch
) {
    if constexpr (SINGLE_UNPACK) {
        uint32_t h00, h01, h02, h03;
        uint32_t h10, h11, h12, h13;
        uint32_t h20, h21, h22, h23;
        uint32_t h30, h31, h32, h33;

        #define LO_STEP16(Q, WA, WB, PSRC, SHIFT, H0, H1, H2, H3) do { \
            const uint16_t e4_pair = static_cast<uint16_t>((PSRC) >> (SHIFT)); \
            const uint32_t sc_f16x2 = cvt_e4m3x2_to_f16x2(e4_pair); \
            const uint32_t sc0 = splat_lo_f16(sc_f16x2); \
            const uint32_t sc1 = splat_hi_f16(sc_f16x2); \
            const int q_off = (Q) * 32; \
            const uint2 a0_01 = row0_live ? *reinterpret_cast<const uint2*>(sA_lo + q_off)                   : make_uint2(0u, 0u); \
            const uint2 a2_01 = row0_live ? *reinterpret_cast<const uint2*>(sA_lo + q_off + 16)              : make_uint2(0u, 0u); \
            const uint2 a1_01 = row8_live ? *reinterpret_cast<const uint2*>(sA_lo + 8 * SA_STRIDE + q_off)   : make_uint2(0u, 0u); \
            const uint2 a3_01 = row8_live ? *reinterpret_cast<const uint2*>(sA_lo + 8 * SA_STRIDE + q_off + 16) : make_uint2(0u, 0u); \
            uint32_t lo01, lo23, hi01, hi23; \
            unpack_e2m1x8((WA), lo01, lo23, hi01, hi23); \
            (H0) = mul_f16x2(hi01, sc0); \
            (H1) = mul_f16x2(hi23, sc1); \
            mma_m16n8k16_f16(acc, a0_01.x, a1_01.x, a2_01.x, a3_01.x, mul_f16x2(lo01, sc0), mul_f16x2(lo23, sc1)); \
            unpack_e2m1x8((WB), lo01, lo23, hi01, hi23); \
            (H2) = mul_f16x2(hi01, sc0); \
            (H3) = mul_f16x2(hi23, sc1); \
            mma_m16n8k16_f16(acc, a0_01.y, a1_01.y, a2_01.y, a3_01.y, mul_f16x2(lo01, sc0), mul_f16x2(lo23, sc1)); \
        } while (0)

        LO_STEP16(0, cur_wa.x, cur_wa.y, p01,  0, h00, h01, h02, h03);
        LO_STEP16(1, cur_wa.z, cur_wa.w, p01, 16, h10, h11, h12, h13);
        LO_STEP16(2, cur_wb.x, cur_wb.y, p23,  0, h20, h21, h22, h23);
        LO_STEP16(3, cur_wb.z, cur_wb.w, p23, 16, h30, h31, h32, h33);
        #undef LO_STEP16

        mid_prefetch();

        #define HI_STEP16(Q, H0, H1, H2, H3) do { \
            const int q_off = (Q) * 32; \
            const uint2 a0_01 = row0_live ? *reinterpret_cast<const uint2*>(sA_hi + q_off)                   : make_uint2(0u, 0u); \
            const uint2 a2_01 = row0_live ? *reinterpret_cast<const uint2*>(sA_hi + q_off + 16)              : make_uint2(0u, 0u); \
            const uint2 a1_01 = row8_live ? *reinterpret_cast<const uint2*>(sA_hi + 8 * SA_STRIDE + q_off)   : make_uint2(0u, 0u); \
            const uint2 a3_01 = row8_live ? *reinterpret_cast<const uint2*>(sA_hi + 8 * SA_STRIDE + q_off + 16) : make_uint2(0u, 0u); \
            mma_m16n8k16_f16(acc, a0_01.x, a1_01.x, a2_01.x, a3_01.x, (H0), (H1)); \
            mma_m16n8k16_f16(acc, a0_01.y, a1_01.y, a2_01.y, a3_01.y, (H2), (H3)); \
        } while (0)

        HI_STEP16(0, h00, h01, h02, h03);
        HI_STEP16(1, h10, h11, h12, h13);
        HI_STEP16(2, h20, h21, h22, h23);
        HI_STEP16(3, h30, h31, h32, h33);
        #undef HI_STEP16
    } else {
        const uint32_t w_arr[8] = {
            cur_wa.x, cur_wa.y, cur_wa.z, cur_wa.w,
            cur_wb.x, cur_wb.y, cur_wb.z, cur_wb.w
        };
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
            const uint16_t e4_pair = static_cast<uint16_t>(((q < 2) ? p01 : p23) >> ((q & 1) * 16));
            const uint32_t sc_f16x2 = cvt_e4m3x2_to_f16x2(e4_pair);
            const uint32_t sc0 = splat_lo_f16(sc_f16x2);
            const uint32_t sc1 = splat_hi_f16(sc_f16x2);
            const int q_off = q * 32;
            const uint2 a0_01 = row0_live ? *reinterpret_cast<const uint2*>(sA_lo + q_off)                   : make_uint2(0u, 0u);
            const uint2 a2_01 = row0_live ? *reinterpret_cast<const uint2*>(sA_lo + q_off + 16)              : make_uint2(0u, 0u);
            const uint2 a1_01 = row8_live ? *reinterpret_cast<const uint2*>(sA_lo + 8 * SA_STRIDE + q_off)   : make_uint2(0u, 0u);
            const uint2 a3_01 = row8_live ? *reinterpret_cast<const uint2*>(sA_lo + 8 * SA_STRIDE + q_off + 16) : make_uint2(0u, 0u);
            uint32_t lo01, lo23;
            unpack_e2m1x8_lo(w_arr[2 * q + 0], lo01, lo23);
            mma_m16n8k16_f16(acc, a0_01.x, a1_01.x, a2_01.x, a3_01.x, mul_f16x2(lo01, sc0), mul_f16x2(lo23, sc1));
            unpack_e2m1x8_lo(w_arr[2 * q + 1], lo01, lo23);
            mma_m16n8k16_f16(acc, a0_01.y, a1_01.y, a2_01.y, a3_01.y, mul_f16x2(lo01, sc0), mul_f16x2(lo23, sc1));
        }
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
            const uint16_t e4_pair = static_cast<uint16_t>(((q < 2) ? p01 : p23) >> ((q & 1) * 16));
            const uint32_t sc_f16x2 = cvt_e4m3x2_to_f16x2(e4_pair);
            const uint32_t sc0 = splat_lo_f16(sc_f16x2);
            const uint32_t sc1 = splat_hi_f16(sc_f16x2);
            const int q_off = q * 32;
            const uint2 a0_01 = row0_live ? *reinterpret_cast<const uint2*>(sA_hi + q_off)                   : make_uint2(0u, 0u);
            const uint2 a2_01 = row0_live ? *reinterpret_cast<const uint2*>(sA_hi + q_off + 16)              : make_uint2(0u, 0u);
            const uint2 a1_01 = row8_live ? *reinterpret_cast<const uint2*>(sA_hi + 8 * SA_STRIDE + q_off)   : make_uint2(0u, 0u);
            const uint2 a3_01 = row8_live ? *reinterpret_cast<const uint2*>(sA_hi + 8 * SA_STRIDE + q_off + 16) : make_uint2(0u, 0u);
            uint32_t hi01, hi23;
            unpack_e2m1x8_hi(w_arr[2 * q + 0], hi01, hi23);
            mma_m16n8k16_f16(acc, a0_01.x, a1_01.x, a2_01.x, a3_01.x, mul_f16x2(hi01, sc0), mul_f16x2(hi23, sc1));
            unpack_e2m1x8_hi(w_arr[2 * q + 1], hi01, hi23);
            mma_m16n8k16_f16(acc, a0_01.y, a1_01.y, a2_01.y, a3_01.y, mul_f16x2(hi01, sc0), mul_f16x2(hi23, sc1));
        }
    }
}

// =========================================================================
// Wave-V10 In-W13 Last-Panel Per-Tile In-Place L2-Hot SiTU -> QDeqA FP16
//   - Executed ONLY by the final finishing panel CTA of tile `t` (1/18 or 1/36 CTAs)
//   - L2-coherent read (`ld.global.cg`), exact 0-ULP `quant_dequant_nvfp4_16_bmax<QUANT_FMA>`,
//     in-place overwrite (`lo8|hi8`) + `RowAmaxW2[r]` -> eliminates 40 standalone launches/step!
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
// Kernel 1A-v10 (TILE=8, Unified BN=32 [18p, 4 warps] & BN=16 [36p, 2 warps]!)
//   - QDEQ_MODE: 0=atomicMax, 1=skip_amax (V9 standalone), 2=fused qdeq(FMA=0), 3=fused qdeq(FMA=1)
// =========================================================================
template <int BN, int MIN_CTA, int PDL_WAIT_MODE, bool TRIG_DEP, bool ZERO_SCATTER,
          bool AS_ROW_MAJOR, bool FUSE_SITU, int QDEQ_MODE, bool ADAPTIVE_GATE>
__global__ __launch_bounds__((BN == 16) ? 64 : 128, MIN_CTA)
void nvfp4_w13_ws5_t8_v16_kernel(
    const uint8_t* __restrict__ A,
    const uint8_t* __restrict__ AS,
    const float* __restrict__ RS,
    const int32_t* __restrict__ Row2Tok,
    const uint8_t* __restrict__ SLABS,
    const uint8_t* __restrict__ WS_BASE,
    const int32_t* __restrict__ P_OFF,
    const uint32_t* __restrict__ PATCHES,
    const float* __restrict__ Ones,
    const int32_t* __restrict__ Tiles,
    const int32_t* __restrict__ TileCompactStart,
    const float* __restrict__ Gammas,
    uint32_t* __restrict__ RowAmaxW2,
    __nv_bfloat16* __restrict__ Out,
    uint32_t* __restrict__ TileQdeqDone,
    uint32_t* __restrict__ WorkQueue,
    int N, int K, int MAX_CTAS, int stride16, int stride32, int bn16_margin_pct,
    int evict_first, int lpt_mode
) {
    static_assert(BN == 16 || BN == 32, "W13 t8 BN must be 16 or 32");
    (void)N;
    (void)K;
    (void)TileQdeqDone;
    if constexpr (PDL_WAIT_MODE == 1) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }

    constexpr int TILE = 8;
    constexpr int WARPS = (BN == 16) ? 2 : 4;
    constexpr int THREADS = WARPS * 32;
    constexpr int N_PANELS32 = 18;
    constexpr int N_PANELS = (BN == 16) ? 36 : 18;
    constexpr int KP = 2304;
    constexpr int KG = 288;
    constexpr int BKP = 128;
    constexpr int M_ROWS = 8;
    constexpr int SA_STRIDE = BKP + 16;  // 144 halfs = 288 B -> 16B aligned & 0-way LDS.64
    constexpr int K_STEPS = 18;

    const int n_active = Tiles[2 * MAX_CTAS];

    if constexpr (ADAPTIVE_GATE) {
        const bool want16 = prefer_w13_bn16(n_active, stride16, stride32, bn16_margin_pct);
        const bool active_here = (BN == 16) ? want16 : (!want16);
        if (!active_here) {
            if constexpr (PDL_WAIT_MODE == 2) {
                asm volatile("griddepcontrol.wait;" ::: "memory");
            }
            if constexpr (TRIG_DEP) {
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
            return;
        }
    }

    const int n_total = n_active * N_PANELS;
    const int stride = static_cast<int>(gridDim.x);
    const int bidx = static_cast<int>(blockIdx.x);

    if (n_active <= 0) {
        if constexpr (PDL_WAIT_MODE == 2) asm volatile("griddepcontrol.wait;" ::: "memory");
        if constexpr (TRIG_DEP) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        return;
    }

    const bool use_dyn_q = (lpt_mode >= 2) && (WorkQueue != nullptr) && (n_total > stride);
    if (!use_dyn_q && bidx >= n_total) {
        if constexpr (PDL_WAIT_MODE == 2) {
            asm volatile("griddepcontrol.wait;" ::: "memory");
        }
        if constexpr (TRIG_DEP) {
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
        return;
    }

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g = lane >> 2;
    const int lt = lane & 3;
    const int x_off0 = g * SA_STRIDE + 4 * lt;

    alignas(16) __shared__ uint16_t sA[2][2][M_ROWS * SA_STRIDE];
    alignas(16) __shared__ int32_t  sTok[M_ROWS];
    alignas(16) __shared__ float    sPanelAmax[(FUSE_SITU && QDEQ_MODE == 0) ? WARPS : 1][M_ROWS];
    constexpr int MAX_LPT_TILES = 512;
    alignas(16) __shared__ int16_t sTileOrder[MAX_LPT_TILES];
    __shared__ int32_t sTileLastQdeq;
    __shared__ int32_t sDynIdx;

    const bool use_ef = (evict_first != 0);
    const uint64_t l2_pol = use_ef ? make_l2_evict_first_policy() : 0ULL;
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

        constexpr bool USE_SINGLE_UNPACK = (BN == 16)
            ? (NVFP4_W13_T8_BN16_SINGLE_UNPACK != 0)
            : (NVFP4_W13_T8_BN32_SINGLE_UNPACK != 0);

        const int nb32 = (BN == 16) ? (nb >> 1) : nb;
        const int wp   = (BN == 16) ? (((nb & 1) << 1) + warp) : warp;
        const int w_off  = wp * 1024 + lane * 16;
        const int ws_tid = (BN == 16) ? (((nb & 1) << 6) + tid) : tid;

        const int e_i32 = Tiles[t];
        const int end_row = Tiles[MAX_CTAS + t];
        int live_m = end_row - t * TILE;
        if (live_m > TILE) live_m = TILE;
        if (live_m < 0) live_m = 0;

        const int64_t panel_uid = static_cast<int64_t>(e_i32) * N_PANELS32 + nb32;
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
        if (live_m == 0) {
            ensure_next();
            idx = nidx;
            is_first_tile = false;
            continue;
        }

        const bool keep_l2 = (evict_first == 2) && (t + 1 < n_active) && (Tiles[t + 1] == e_i32);
        const uint32_t raw_base = static_cast<uint32_t>(WS_BASE[e_i32]);
        const uint32_t base4 = (raw_base & 0x7Fu) * 0x01010101u;

        uint4 cur_wa = ld_w_u4(panel_base + w_off, keep_l2, use_ef, l2_pol);
        uint4 cur_wb = ld_w_u4(panel_base + w_off + 512, keep_l2, use_ef, l2_pol);
        uint32_t cur_ws_lo = ld_w_u32(panel_base + 8192 + (ws_tid << 2), keep_l2, use_ef, l2_pol);
        uint8_t  cur_ws_hi = ld_w_u8(panel_base + 8704 + ws_tid, keep_l2, use_ef, l2_pol);

        int32_t p_cur = 0, p_end = 0;
        if ((raw_base & 0x80u) == 0u) {
            const int64_t warp_uid = panel_uid * 4 + wp;
            p_cur = P_OFF[warp_uid];
            p_end = P_OFF[warp_uid + 1];
        }

        if constexpr (ZERO_SCATTER) {
            if (tid < live_m) sTok[tid] = Row2Tok[t * TILE + tid];
        }

        // V9-3: ZERO `sA` zero-fill loop! Dead rows (g >= live_m) pass literal `0u` to MMA!
        auto load_a_slab_t8 = [&](int ks, int buf, bool use_smem_tok) {
            const int total_q2 = live_m * 16;
            const int kp0 = ks * BKP;
            const int kg0 = ks * 16;
            const int rbase = t * TILE;
            uint16_t* dst_lo = sA[buf][0];
            uint16_t* dst_hi = sA[buf][1];
            for (int aidx = tid; aidx < total_q2; aidx += THREADS) {
                const int lm = aidx >> 4;
                const int qk2 = aidx & 15;
                const int64_t row = static_cast<int64_t>(rbase + lm);
                const int64_t a_row = ZERO_SCATTER
                    ? static_cast<int64_t>(use_smem_tok ? sTok[lm] : Row2Tok[row])
                    : row;
                const uint2 a8 = ld_ca_u2(A + a_row * KP + kp0 + (qk2 << 3));
                const int kg = kg0 + qk2;
                uint8_t as_b;
                if constexpr (ZERO_SCATTER || AS_ROW_MAJOR) {
                    as_b = AS[a_row * KG + kg];
                } else {
                    as_b = AS[scale_offset_decode(row, kg, KG)];
                }
                const uint32_t as_f16x2 = splat_lo_f16(cvt_e4m3x2_to_f16x2(static_cast<uint16_t>(as_b)));
                uint32_t lo01, lo23, hi01, hi23;
                uint4 v_lo, v_hi;
                unpack_e2m1x8(a8.x, lo01, lo23, hi01, hi23);
                v_lo.x = mul_f16x2(lo01, as_f16x2); v_lo.y = mul_f16x2(lo23, as_f16x2);
                v_hi.x = mul_f16x2(hi01, as_f16x2); v_hi.y = mul_f16x2(hi23, as_f16x2);
                unpack_e2m1x8(a8.y, lo01, lo23, hi01, hi23);
                v_lo.z = mul_f16x2(lo01, as_f16x2); v_lo.w = mul_f16x2(lo23, as_f16x2);
                v_hi.z = mul_f16x2(hi01, as_f16x2); v_hi.w = mul_f16x2(hi23, as_f16x2);
                const int off = lm * SA_STRIDE + (qk2 << 3);
                *reinterpret_cast<uint4*>(dst_lo + off) = v_lo;
                *reinterpret_cast<uint4*>(dst_hi + off) = v_hi;
            }
        };

        load_a_slab_t8(0, 0, false);
        __syncthreads();

        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        uint32_t ws_even = 0u, ws_odd = 0u;
        const bool row0_live = (g < live_m);

        for (int ks = 0; ks < K_STEPS; ++ks) {
            const int cur_buf = ks & 1;
            const int nxt_buf = cur_buf ^ 1;
            const int ss = ks >> 1;

            if ((ks & 1) == 0) {
                decode_ws5_w13_swar(cur_ws_lo, static_cast<uint32_t>(cur_ws_hi), base4, ws_even, ws_odd);
                if (__builtin_expect(p_cur < p_end, 0)) {
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

            uint4 nxt_wa, nxt_wb;
            if constexpr (!USE_SINGLE_UNPACK) {
                if (ks + 1 < K_STEPS) {
                    const int nxt_ks = ks + 1;
                    const uint8_t* nxt_ss = panel_base + static_cast<int64_t>(nxt_ks >> 1) * W13_SS_BYTES;
                    const int ks_off_b = (nxt_ks & 1) * 4096;
                    nxt_wa = ld_w_u4(nxt_ss + ks_off_b + w_off, keep_l2, use_ef, l2_pol);
                    nxt_wb = ld_w_u4(nxt_ss + ks_off_b + w_off + 512, keep_l2, use_ef, l2_pol);
                    if ((nxt_ks & 1) == 0) {
                        cur_ws_lo = ld_w_u32(nxt_ss + 8192 + (ws_tid << 2), keep_l2, use_ef, l2_pol);
                        cur_ws_hi = ld_w_u8(nxt_ss + 8704 + ws_tid, keep_l2, use_ef, l2_pol);
                    }
                }
            }

            if (ks + 1 < K_STEPS) {
                load_a_slab_t8(ks + 1, nxt_buf, true);
            }
            if (ks == K_STEPS - 2) {
                ensure_next();
                if (nidx < n_total) {
                    const int nlog_t = nidx / N_PANELS;
                    const int nnb = nidx - nlog_t * N_PANELS;
                    const int nt = map_tile(nlog_t);
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
            }

            const int src0 = (g << 2) | ((ks & 1) * 2);
            const int src1 = src0 | 1;
            const uint32_t e0 = __shfl_sync(0xFFFFFFFFu, ws_even, src0);
            const uint32_t o0 = __shfl_sync(0xFFFFFFFFu, ws_odd,  src0);
            const uint32_t e1 = __shfl_sync(0xFFFFFFFFu, ws_even, src1);
            const uint32_t o1 = __shfl_sync(0xFFFFFFFFu, ws_odd,  src1);
            const uint32_t p01 = (lt < 2) ? e0 : o0;
            const uint32_t p23 = (lt < 2) ? e1 : o1;

            w13_kstep_mma_t8_exact<USE_SINGLE_UNPACK>(
                acc, cur_wa, cur_wb, p01, p23,
                sA[cur_buf][0] + x_off0, sA[cur_buf][1] + x_off0, row0_live,
                [&]() {
                    if (ks + 1 < K_STEPS) {
                        const int nxt_ks = ks + 1;
                        const uint8_t* nxt_ss = panel_base + static_cast<int64_t>(nxt_ks >> 1) * W13_SS_BYTES;
                        const int ks_off_b = (nxt_ks & 1) * 4096;
                        cur_wa = ld_w_u4(nxt_ss + ks_off_b + w_off, keep_l2, use_ef, l2_pol);
                        cur_wb = ld_w_u4(nxt_ss + ks_off_b + w_off + 512, keep_l2, use_ef, l2_pol);
                        if ((nxt_ks & 1) == 0) {
                            cur_ws_lo = ld_w_u32(nxt_ss + 8192 + (ws_tid << 2), keep_l2, use_ef, l2_pol);
                            cur_ws_hi = ld_w_u8(nxt_ss + 8704 + ws_tid, keep_l2, use_ef, l2_pol);
                        }
                    }
                }
            );

            if constexpr (TRIG_DEP) {
                if (ks == K_STEPS - 2 && nidx >= n_total && !trig_fired) {
                    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
                    trig_fired = true;
                }
            }

            if (ks + 1 < K_STEPS) {
                if constexpr (!USE_SINGLE_UNPACK) {
                    cur_wa = nxt_wa;
                    cur_wb = nxt_wb;
                }
                __syncthreads();
            }
        }

        const float scale = Ones[Tiles[t]];
        const int r0 = t * TILE + g;
        const float rs0 = row0_live ? RS[r0] : 0.0f;
        const float o0 = mul_rn(mul_rn(acc[0], rs0), scale);
        const float o1 = mul_rn(mul_rn(acc[1], rs0), scale);

        if constexpr (FUSE_SITU) {
            constexpr int INTER = 288;
            const int c_warp = nb * (BN >> 1) + warp * 4;
            const int compact_base = TileCompactStart[t];
            const int c0 = compact_base + g;

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

            if constexpr (QDEQ_MODE == 0) {
                float a0 = row0_live ? fabsf(bf16_bits_to_f32(y0_b)) : 0.0f;
                a0 = fmaxf(a0, __shfl_xor_sync(0xFFFFFFFFu, a0, 1));
                a0 = fmaxf(a0, __shfl_xor_sync(0xFFFFFFFFu, a0, 2));
                if (lt == 0) sPanelAmax[warp][g] = a0;

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
        }

        ensure_next();
        if (nidx < n_total) __syncthreads();
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

    if constexpr (TRIG_DEP) {
        if (!trig_fired) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }
}

// =========================================================================
// Kernel 1B-v9 (TILE=16): NV_W13 (N=576, K=4608, BK_DOT=256)
//   - Single-pass `unpack_e2m1x8` (-50% CVT), zero-fill/LDS elimination, pure `ld.cs`, `SKIP_AMAX`!
// =========================================================================
template <int TILE, int MIN_CTA, int PDL_WAIT_MODE, bool TRIG_DEP, bool ZERO_SCATTER,
          bool AS_ROW_MAJOR, bool FUSE_SITU, int QDEQ_MODE>
__global__ __launch_bounds__(128, MIN_CTA)
void nvfp4_w13_ws5_t16_v16_kernel(
    const uint8_t* __restrict__ A,
    const uint8_t* __restrict__ AS,
    const float* __restrict__ RS,
    const int32_t* __restrict__ Row2Tok,
    const uint8_t* __restrict__ SLABS,
    const uint8_t* __restrict__ WS_BASE,
    const int32_t* __restrict__ P_OFF,
    const uint32_t* __restrict__ PATCHES,
    const float* __restrict__ Ones,
    const int32_t* __restrict__ Tiles,
    const int32_t* __restrict__ TileCompactStart,
    const float* __restrict__ Gammas,
    uint32_t* __restrict__ RowAmaxW2,
    __nv_bfloat16* __restrict__ Out,
    uint32_t* __restrict__ TileQdeqDone,
    uint32_t* __restrict__ WorkQueue,
    int N, int K, int MAX_CTAS,
    int evict_first, int lpt_mode
) {
    (void)N;
    (void)K;
    (void)TileQdeqDone;
    if constexpr (PDL_WAIT_MODE == 1) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }

    constexpr int WARPS = 4;
    constexpr int BN = 32;
    constexpr int N_PANELS = 18;
    constexpr int KP = 2304;
    constexpr int KG = 288;
    constexpr int BKP = 128;
    constexpr int M_ROWS = 16;
    constexpr int SA_STRIDE = BKP + 16;
    constexpr int K_STEPS = 18;

    const int n_active = Tiles[2 * MAX_CTAS];
    const int n_total = n_active * N_PANELS;
    const int stride = static_cast<int>(gridDim.x);
    const int bidx = static_cast<int>(blockIdx.x);

    if (n_active <= 0) {
        if constexpr (PDL_WAIT_MODE == 2) asm volatile("griddepcontrol.wait;" ::: "memory");
        if constexpr (TRIG_DEP) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        return;
    }

    const bool use_dyn_q = (lpt_mode >= 2) && (WorkQueue != nullptr) && (n_total > stride);
    if (!use_dyn_q && bidx >= n_total) {
        if constexpr (PDL_WAIT_MODE == 2) {
            asm volatile("griddepcontrol.wait;" ::: "memory");
        }
        if constexpr (TRIG_DEP) {
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
        return;
    }

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g = lane >> 2;
    const int lt = lane & 3;
    const int w_off = warp * 1024 + lane * 16;
    const int x_off0 = g * SA_STRIDE + 4 * lt;
    constexpr bool USE_SINGLE_UNPACK = (NVFP4_W13_T16_SINGLE_UNPACK != 0);

    alignas(16) __shared__ uint16_t sA[2][2][M_ROWS * SA_STRIDE];
    alignas(16) __shared__ int32_t  sTok[M_ROWS];
    alignas(16) __shared__ float    sPanelAmax[(FUSE_SITU && QDEQ_MODE == 0) ? WARPS : 1][M_ROWS];
    constexpr int MAX_LPT_TILES = 512;
    alignas(16) __shared__ int16_t sTileOrder[MAX_LPT_TILES];
    __shared__ int32_t sTileLastQdeq;
    __shared__ int32_t sDynIdx;

    const bool use_ef = (evict_first != 0);
    const uint64_t l2_pol = use_ef ? make_l2_evict_first_policy() : 0ULL;
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

        const int64_t panel_uid = static_cast<int64_t>(e_i32) * N_PANELS + nb;
        const uint8_t* panel_base = SLABS + panel_uid * static_cast<int64_t>(9 * W13_SS_BYTES);

        if constexpr (PDL_WAIT_MODE == 2) {
            if (is_first_tile) {
                if ((lane & 7) == 0) {
                    prefetch_l2(panel_base + w_off);
                    prefetch_l2(panel_base + w_off + 512);
                }
                if (tid < 5) prefetch_l2(panel_base + 8192 + tid * 128);
                asm volatile("griddepcontrol.wait;" ::: "memory");
            }
        }
        if (live_m == 0) {
            ensure_next();
            idx = nidx;
            is_first_tile = false;
            continue;
        }

        const bool keep_l2 = (evict_first == 2) && (t + 1 < n_active) && (Tiles[t + 1] == e_i32);
        const uint32_t raw_base = static_cast<uint32_t>(WS_BASE[e_i32]);
        const uint32_t base4 = (raw_base & 0x7Fu) * 0x01010101u;

        uint4 cur_wa = ld_w_u4(panel_base + w_off, keep_l2, use_ef, l2_pol);
        uint4 cur_wb = ld_w_u4(panel_base + w_off + 512, keep_l2, use_ef, l2_pol);
        uint32_t cur_ws_lo = ld_w_u32(panel_base + 8192 + (tid << 2), keep_l2, use_ef, l2_pol);
        uint8_t  cur_ws_hi = ld_w_u8(panel_base + 8704 + tid, keep_l2, use_ef, l2_pol);

        int32_t p_cur = 0, p_end = 0;
        if ((raw_base & 0x80u) == 0u) {
            const int64_t warp_uid = panel_uid * WARPS + warp;
            p_cur = P_OFF[warp_uid];
            p_end = P_OFF[warp_uid + 1];
        }

        if constexpr (ZERO_SCATTER) {
            if (tid < live_m) sTok[tid] = Row2Tok[t * TILE + tid];
        }

        auto load_a_slab = [&](int ks, int buf, bool use_smem_tok) {
            const int total_q2 = live_m * 16;
            const int kp0 = ks * BKP;
            const int kg0 = ks * 16;
            const int rbase = t * TILE;
            uint16_t* dst_lo = sA[buf][0];
            uint16_t* dst_hi = sA[buf][1];
            for (int aidx = tid; aidx < total_q2; aidx += 128) {
                const int lm = aidx >> 4;
                const int qk2 = aidx & 15;
                const int64_t row = static_cast<int64_t>(rbase + lm);
                const int64_t a_row = ZERO_SCATTER
                    ? static_cast<int64_t>(use_smem_tok ? sTok[lm] : Row2Tok[row])
                    : row;
                const uint2 a8 = ld_ca_u2(A + a_row * KP + kp0 + (qk2 << 3));
                const int kg = kg0 + qk2;
                uint8_t as_b;
                if constexpr (ZERO_SCATTER || AS_ROW_MAJOR) {
                    as_b = AS[a_row * KG + kg];
                } else {
                    as_b = AS[scale_offset_decode(row, kg, KG)];
                }
                const uint32_t as_f16x2 = splat_lo_f16(cvt_e4m3x2_to_f16x2(static_cast<uint16_t>(as_b)));
                uint32_t lo01, lo23, hi01, hi23;
                uint4 v_lo, v_hi;
                unpack_e2m1x8(a8.x, lo01, lo23, hi01, hi23);
                v_lo.x = mul_f16x2(lo01, as_f16x2); v_lo.y = mul_f16x2(lo23, as_f16x2);
                v_hi.x = mul_f16x2(hi01, as_f16x2); v_hi.y = mul_f16x2(hi23, as_f16x2);
                unpack_e2m1x8(a8.y, lo01, lo23, hi01, hi23);
                v_lo.z = mul_f16x2(lo01, as_f16x2); v_lo.w = mul_f16x2(lo23, as_f16x2);
                v_hi.z = mul_f16x2(hi01, as_f16x2); v_hi.w = mul_f16x2(hi23, as_f16x2);
                const int off = lm * SA_STRIDE + (qk2 << 3);
                *reinterpret_cast<uint4*>(dst_lo + off) = v_lo;
                *reinterpret_cast<uint4*>(dst_hi + off) = v_hi;
            }
        };

        load_a_slab(0, 0, false);
        __syncthreads();

        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        uint32_t ws_even = 0u, ws_odd = 0u;
        const bool row0_live = (g < live_m);
        const bool row8_live = (g + 8 < live_m);

        for (int ks = 0; ks < K_STEPS; ++ks) {
            const int cur_buf = ks & 1;
            const int nxt_buf = cur_buf ^ 1;
            const int ss = ks >> 1;

            if ((ks & 1) == 0) {
                decode_ws5_w13_swar(cur_ws_lo, static_cast<uint32_t>(cur_ws_hi), base4, ws_even, ws_odd);
                if (__builtin_expect(p_cur < p_end, 0)) {
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

            uint4 nxt_wa, nxt_wb;
            if constexpr (!USE_SINGLE_UNPACK) {
                if (ks + 1 < K_STEPS) {
                    const int nxt_ks = ks + 1;
                    const uint8_t* nxt_ss = panel_base + static_cast<int64_t>(nxt_ks >> 1) * W13_SS_BYTES;
                    const int ks_off_b = (nxt_ks & 1) * 4096;
                    nxt_wa = ld_w_u4(nxt_ss + ks_off_b + w_off, keep_l2, use_ef, l2_pol);
                    nxt_wb = ld_w_u4(nxt_ss + ks_off_b + w_off + 512, keep_l2, use_ef, l2_pol);
                    if ((nxt_ks & 1) == 0) {
                        cur_ws_lo = ld_w_u32(nxt_ss + 8192 + (tid << 2), keep_l2, use_ef, l2_pol);
                        cur_ws_hi = ld_w_u8(nxt_ss + 8704 + tid, keep_l2, use_ef, l2_pol);
                    }
                }
            }

            if (ks + 1 < K_STEPS) {
                load_a_slab(ks + 1, nxt_buf, true);
            }
            if (ks == K_STEPS - 2) {
                ensure_next();
                if (nidx < n_total) {
                    const int nlog_t = nidx / N_PANELS;
                    const int nnb = nidx - nlog_t * N_PANELS;
                    const int nt = map_tile(nlog_t);
                    const int64_t ne = static_cast<int64_t>(Tiles[nt]);
                    const uint8_t* nxt_pbase = SLABS + (ne * N_PANELS + nnb) * static_cast<int64_t>(9 * W13_SS_BYTES);
                    if ((lane & 7) == 0) {
                        prefetch_l2(nxt_pbase + w_off);
                        prefetch_l2(nxt_pbase + w_off + 512);
                    }
                    if (tid < 5) prefetch_l2(nxt_pbase + 8192 + tid * 128);
                }
            }

            const int src0 = (g << 2) | ((ks & 1) * 2);
            const int src1 = src0 | 1;
            const uint32_t e0 = __shfl_sync(0xFFFFFFFFu, ws_even, src0);
            const uint32_t o0 = __shfl_sync(0xFFFFFFFFu, ws_odd,  src0);
            const uint32_t e1 = __shfl_sync(0xFFFFFFFFu, ws_even, src1);
            const uint32_t o1 = __shfl_sync(0xFFFFFFFFu, ws_odd,  src1);
            const uint32_t p01 = (lt < 2) ? e0 : o0;
            const uint32_t p23 = (lt < 2) ? e1 : o1;

            w13_kstep_mma_t16_exact<SA_STRIDE, USE_SINGLE_UNPACK>(
                acc, cur_wa, cur_wb, p01, p23,
                sA[cur_buf][0] + x_off0, sA[cur_buf][1] + x_off0, row0_live, row8_live,
                [&]() {
                    if (ks + 1 < K_STEPS) {
                        const int nxt_ks = ks + 1;
                        const uint8_t* nxt_ss = panel_base + static_cast<int64_t>(nxt_ks >> 1) * W13_SS_BYTES;
                        const int ks_off_b = (nxt_ks & 1) * 4096;
                        cur_wa = ld_w_u4(nxt_ss + ks_off_b + w_off, keep_l2, use_ef, l2_pol);
                        cur_wb = ld_w_u4(nxt_ss + ks_off_b + w_off + 512, keep_l2, use_ef, l2_pol);
                        if ((nxt_ks & 1) == 0) {
                            cur_ws_lo = ld_w_u32(nxt_ss + 8192 + (tid << 2), keep_l2, use_ef, l2_pol);
                            cur_ws_hi = ld_w_u8(nxt_ss + 8704 + tid, keep_l2, use_ef, l2_pol);
                        }
                    }
                }
            );

            if constexpr (TRIG_DEP) {
                if (ks == K_STEPS - 2 && nidx >= n_total && !trig_fired) {
                    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
                    trig_fired = true;
                }
            }

            if (ks + 1 < K_STEPS) {
                if constexpr (!USE_SINGLE_UNPACK) {
                    cur_wa = nxt_wa;
                    cur_wb = nxt_wb;
                }
                __syncthreads();
            }
        }

        const float scale = Ones[Tiles[t]];
        const int r0 = t * TILE + g;
        const int r1 = t * TILE + g + 8;
        const float rs0 = row0_live ? RS[r0] : 0.0f;
        const float rs1 = row8_live ? RS[r1] : 0.0f;
        const float o0 = mul_rn(mul_rn(acc[0], rs0), scale);
        const float o1 = mul_rn(mul_rn(acc[1], rs0), scale);
        const float o2 = mul_rn(mul_rn(acc[2], rs1), scale);
        const float o3 = mul_rn(mul_rn(acc[3], rs1), scale);

        if constexpr (FUSE_SITU) {
            constexpr int INTER = 288;
            const int c_warp = nb * 16 + warp * 4;
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

            if constexpr (QDEQ_MODE == 0) {
                float a0 = row0_live ? fabsf(bf16_bits_to_f32(y0_b)) : 0.0f;
                a0 = fmaxf(a0, __shfl_xor_sync(0xFFFFFFFFu, a0, 1));
                a0 = fmaxf(a0, __shfl_xor_sync(0xFFFFFFFFu, a0, 2));
                if (lt == 0) sPanelAmax[warp][g] = a0;

                float a1 = row8_live ? fabsf(bf16_bits_to_f32(y1_b)) : 0.0f;
                a1 = fmaxf(a1, __shfl_xor_sync(0xFFFFFFFFu, a1, 1));
                a1 = fmaxf(a1, __shfl_xor_sync(0xFFFFFFFFu, a1, 2));
                if (lt == 0) sPanelAmax[warp][g + 8] = a1;

                __syncthreads();
                if (tid < live_m) {
                    float rmax = fmaxf(
                        fmaxf(sPanelAmax[0][tid], sPanelAmax[1][tid]),
                        fmaxf(sPanelAmax[2][tid], sPanelAmax[3][tid])
                    );
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
            if (row8_live) {
                *reinterpret_cast<uint32_t*>(Out + static_cast<int64_t>(r1) * 576 + n0) = pack_bf16x2(o2, o3);
            }
        }

        ensure_next();
        if (nidx < n_total) __syncthreads();
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

    if constexpr (TRIG_DEP) {
        if (!trig_fired) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }
}

// =========================================================================
// Kernel 1C-v9 (TILE=32, BK_DOT=256): NV_W13 (N=576, K=4608, single-buffered sA)
//   - Single-pass `unpack_e2m1x8`, zero-fill/LDS elimination, pure `ld.cs`, `SKIP_AMAX`!
// =========================================================================
template <int MIN_CTA, int PDL_WAIT_MODE, bool TRIG_DEP, bool ZERO_SCATTER,
          bool AS_ROW_MAJOR, bool FUSE_SITU, int QDEQ_MODE>
__global__ __launch_bounds__(128, MIN_CTA)
void nvfp4_w13_ws5_t32_bk256_v16_kernel(
    const uint8_t* __restrict__ A,
    const uint8_t* __restrict__ AS,
    const float* __restrict__ RS,
    const int32_t* __restrict__ Row2Tok,
    const uint8_t* __restrict__ SLABS,
    const uint8_t* __restrict__ WS_BASE,
    const int32_t* __restrict__ P_OFF,
    const uint32_t* __restrict__ PATCHES,
    const float* __restrict__ Ones,
    const int32_t* __restrict__ Tiles,
    const int32_t* __restrict__ TileCompactStart,
    const float* __restrict__ Gammas,
    uint32_t* __restrict__ RowAmaxW2,
    __nv_bfloat16* __restrict__ Out,
    uint32_t* __restrict__ TileQdeqDone,
    uint32_t* __restrict__ WorkQueue,
    int N, int K, int MAX_CTAS,
    int evict_first, int lpt_mode
) {
    (void)N;
    (void)K;
    (void)TileQdeqDone;
    if constexpr (PDL_WAIT_MODE == 1) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }

    constexpr int WARPS = 4;
    constexpr int TILE = 32;
    constexpr int BN = 32;
    constexpr int N_PANELS = 18;
    constexpr int KP = 2304;
    constexpr int KG = 288;
    constexpr int BKP = 128;
    constexpr int SA_STRIDE = BKP + 16;
    constexpr int K_STEPS = 18;

    const int n_active = Tiles[2 * MAX_CTAS];
    const int n_total = n_active * N_PANELS;
    const int stride = static_cast<int>(gridDim.x);
    const int bidx = static_cast<int>(blockIdx.x);

    if (n_active <= 0) {
        if constexpr (PDL_WAIT_MODE == 2) asm volatile("griddepcontrol.wait;" ::: "memory");
        if constexpr (TRIG_DEP) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        return;
    }

    const bool use_dyn_q = (lpt_mode >= 2) && (WorkQueue != nullptr) && (n_total > stride);
    if (!use_dyn_q && bidx >= n_total) {
        if constexpr (PDL_WAIT_MODE == 2) {
            asm volatile("griddepcontrol.wait;" ::: "memory");
        }
        if constexpr (TRIG_DEP) {
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
        return;
    }

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g = lane >> 2;
    const int lt = lane & 3;
    const int w_off = warp * 1024 + lane * 16;
    const int x_off0 = g * SA_STRIDE + 4 * lt;
    constexpr bool USE_SINGLE_UNPACK = (NVFP4_W13_T32B_SINGLE_UNPACK != 0);

    alignas(16) __shared__ uint16_t sA[2][TILE * SA_STRIDE];
    alignas(16) __shared__ int32_t  sTok[TILE];
    alignas(16) __shared__ float    sPanelAmax[(FUSE_SITU && QDEQ_MODE == 0) ? WARPS : 1][TILE];
    constexpr int MAX_LPT_TILES = 448;
    alignas(16) __shared__ int16_t sTileOrder[MAX_LPT_TILES];
    __shared__ int32_t sTileLastQdeq;
    __shared__ int32_t sDynIdx;

    const bool use_ef = (evict_first != 0);
    const uint64_t l2_pol = use_ef ? make_l2_evict_first_policy() : 0ULL;
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

        const int64_t panel_uid = static_cast<int64_t>(e_i32) * N_PANELS + nb;
        const uint8_t* panel_base = SLABS + panel_uid * static_cast<int64_t>(9 * W13_SS_BYTES);

        if constexpr (PDL_WAIT_MODE == 2) {
            if (is_first_tile) {
                if ((lane & 7) == 0) {
                    prefetch_l2(panel_base + w_off);
                    prefetch_l2(panel_base + w_off + 512);
                }
                if (tid < 5) prefetch_l2(panel_base + 8192 + tid * 128);
                asm volatile("griddepcontrol.wait;" ::: "memory");
            }
        }
        if (live_m == 0) {
            ensure_next();
            idx = nidx;
            is_first_tile = false;
            continue;
        }

        const bool keep_l2 = (evict_first == 2) && (t + 1 < n_active) && (Tiles[t + 1] == e_i32);
        const uint32_t raw_base = static_cast<uint32_t>(WS_BASE[e_i32]);
        const uint32_t base4 = (raw_base & 0x7Fu) * 0x01010101u;

        uint4 cur_wa = ld_w_u4(panel_base + w_off, keep_l2, use_ef, l2_pol);
        uint4 cur_wb = ld_w_u4(panel_base + w_off + 512, keep_l2, use_ef, l2_pol);
        uint32_t cur_ws_lo = ld_w_u32(panel_base + 8192 + (tid << 2), keep_l2, use_ef, l2_pol);
        uint8_t  cur_ws_hi = ld_w_u8(panel_base + 8704 + tid, keep_l2, use_ef, l2_pol);

        int32_t p_cur = 0, p_end = 0;
        if ((raw_base & 0x80u) == 0u) {
            const int64_t warp_uid = panel_uid * WARPS + warp;
            p_cur = P_OFF[warp_uid];
            p_end = P_OFF[warp_uid + 1];
        }

        if constexpr (ZERO_SCATTER) {
            if (tid < live_m) sTok[tid] = Row2Tok[t * TILE + tid];
        }

        auto load_a_slab_s = [&](int ks, bool use_smem_tok) {
            const int total_q2 = live_m * 16;
            const int kp0 = ks * BKP;
            const int kg0 = ks * 16;
            const int rbase = t * TILE;
            uint16_t* dst_lo = sA[0];
            uint16_t* dst_hi = sA[1];
            for (int aidx = tid; aidx < total_q2; aidx += 128) {
                const int lm = aidx >> 4;
                const int qk2 = aidx & 15;
                const int64_t row = static_cast<int64_t>(rbase + lm);
                const int64_t a_row = ZERO_SCATTER
                    ? static_cast<int64_t>(use_smem_tok ? sTok[lm] : Row2Tok[row])
                    : row;
                const uint2 a8 = ld_ca_u2(A + a_row * KP + kp0 + (qk2 << 3));
                const int kg = kg0 + qk2;
                uint8_t as_b;
                if constexpr (ZERO_SCATTER || AS_ROW_MAJOR) {
                    as_b = AS[a_row * KG + kg];
                } else {
                    as_b = AS[scale_offset_decode(row, kg, KG)];
                }
                const uint32_t as_f16x2 = splat_lo_f16(cvt_e4m3x2_to_f16x2(static_cast<uint16_t>(as_b)));
                uint32_t lo01, lo23, hi01, hi23;
                uint4 v_lo, v_hi;
                unpack_e2m1x8(a8.x, lo01, lo23, hi01, hi23);
                v_lo.x = mul_f16x2(lo01, as_f16x2); v_lo.y = mul_f16x2(lo23, as_f16x2);
                v_hi.x = mul_f16x2(hi01, as_f16x2); v_hi.y = mul_f16x2(hi23, as_f16x2);
                unpack_e2m1x8(a8.y, lo01, lo23, hi01, hi23);
                v_lo.z = mul_f16x2(lo01, as_f16x2); v_lo.w = mul_f16x2(lo23, as_f16x2);
                v_hi.z = mul_f16x2(hi01, as_f16x2); v_hi.w = mul_f16x2(hi23, as_f16x2);
                const int off = lm * SA_STRIDE + (qk2 << 3);
                *reinterpret_cast<uint4*>(dst_lo + off) = v_lo;
                *reinterpret_cast<uint4*>(dst_hi + off) = v_hi;
            }
        };

        load_a_slab_s(0, false);
        __syncthreads();

        float acc0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float acc1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        uint32_t ws_even = 0u, ws_odd = 0u;
        const bool row0_live  = (g < live_m);
        const bool row8_live  = (g + 8 < live_m);
        const bool has_m16    = (live_m > 16);
        const bool row16_live = (g + 16 < live_m);
        const bool row24_live = (g + 24 < live_m);

        for (int ks = 0; ks < K_STEPS; ++ks) {
            const int ss = ks >> 1;

            if ((ks & 1) == 0) {
                decode_ws5_w13_swar(cur_ws_lo, static_cast<uint32_t>(cur_ws_hi), base4, ws_even, ws_odd);
                if (__builtin_expect(p_cur < p_end, 0)) {
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

            uint4 nxt_wa, nxt_wb;
            if constexpr (!USE_SINGLE_UNPACK) {
                if (ks + 1 < K_STEPS) {
                    const int nxt_ks = ks + 1;
                    const uint8_t* nxt_ss = panel_base + static_cast<int64_t>(nxt_ks >> 1) * W13_SS_BYTES;
                    const int ks_off_b = (nxt_ks & 1) * 4096;
                    nxt_wa = ld_w_u4(nxt_ss + ks_off_b + w_off, keep_l2, use_ef, l2_pol);
                    nxt_wb = ld_w_u4(nxt_ss + ks_off_b + w_off + 512, keep_l2, use_ef, l2_pol);
                    if ((nxt_ks & 1) == 0) {
                        cur_ws_lo = ld_w_u32(nxt_ss + 8192 + (tid << 2), keep_l2, use_ef, l2_pol);
                        cur_ws_hi = ld_w_u8(nxt_ss + 8704 + tid, keep_l2, use_ef, l2_pol);
                    }
                }
            }

            if (ks == K_STEPS - 2) {
                ensure_next();
                if (nidx < n_total) {
                    const int nlog_t = nidx / N_PANELS;
                    const int nnb = nidx - nlog_t * N_PANELS;
                    const int nt = map_tile(nlog_t);
                    const int64_t ne = static_cast<int64_t>(Tiles[nt]);
                    const uint8_t* nxt_pbase = SLABS + (ne * N_PANELS + nnb) * static_cast<int64_t>(9 * W13_SS_BYTES);
                    if ((lane & 7) == 0) {
                        prefetch_l2(nxt_pbase + w_off);
                        prefetch_l2(nxt_pbase + w_off + 512);
                    }
                    if (tid < 5) prefetch_l2(nxt_pbase + 8192 + tid * 128);
                }
            }

            const int src0 = (g << 2) | ((ks & 1) * 2);
            const int src1 = src0 | 1;
            const uint32_t e0 = __shfl_sync(0xFFFFFFFFu, ws_even, src0);
            const uint32_t o0 = __shfl_sync(0xFFFFFFFFu, ws_odd,  src0);
            const uint32_t e1 = __shfl_sync(0xFFFFFFFFu, ws_even, src1);
            const uint32_t o1 = __shfl_sync(0xFFFFFFFFu, ws_odd,  src1);
            const uint32_t p01 = (lt < 2) ? e0 : o0;
            const uint32_t p23 = (lt < 2) ? e1 : o1;

            if constexpr (USE_SINGLE_UNPACK) {
                const uint16_t* sA_lo = sA[0] + x_off0;

                uint32_t h00, h01, h02, h03;
                uint32_t h10, h11, h12, h13;
                uint32_t h20, h21, h22, h23;
                uint32_t h30, h31, h32, h33;

                #define LO_STEP32(Q, WA, WB, PSRC, SHIFT, H0, H1, H2, H3) do { \
                    const uint16_t e4_pair = static_cast<uint16_t>((PSRC) >> (SHIFT)); \
                    const uint32_t sc_f16x2 = cvt_e4m3x2_to_f16x2(e4_pair); \
                    const uint32_t sc0 = splat_lo_f16(sc_f16x2); \
                    const uint32_t sc1 = splat_hi_f16(sc_f16x2); \
                    const int q_off = (Q) * 32; \
                    const uint2 a0_01 = row0_live ? *reinterpret_cast<const uint2*>(sA_lo + q_off)                   : make_uint2(0u, 0u); \
                    const uint2 a2_01 = row0_live ? *reinterpret_cast<const uint2*>(sA_lo + q_off + 16)              : make_uint2(0u, 0u); \
                    const uint2 a1_01 = row8_live ? *reinterpret_cast<const uint2*>(sA_lo + 8 * SA_STRIDE + q_off)   : make_uint2(0u, 0u); \
                    const uint2 a3_01 = row8_live ? *reinterpret_cast<const uint2*>(sA_lo + 8 * SA_STRIDE + q_off + 16) : make_uint2(0u, 0u); \
                    uint32_t lo01, lo23, hi01, hi23; \
                    unpack_e2m1x8((WA), lo01, lo23, hi01, hi23); \
                    const uint32_t bl0 = mul_f16x2(lo01, sc0), bl1 = mul_f16x2(lo23, sc1); \
                    (H0) = mul_f16x2(hi01, sc0); (H1) = mul_f16x2(hi23, sc1); \
                    mma_m16n8k16_f16(acc0, a0_01.x, a1_01.x, a2_01.x, a3_01.x, bl0, bl1); \
                    unpack_e2m1x8((WB), lo01, lo23, hi01, hi23); \
                    const uint32_t bl2 = mul_f16x2(lo01, sc0), bl3 = mul_f16x2(lo23, sc1); \
                    (H2) = mul_f16x2(hi01, sc0); (H3) = mul_f16x2(hi23, sc1); \
                    mma_m16n8k16_f16(acc0, a0_01.y, a1_01.y, a2_01.y, a3_01.y, bl2, bl3); \
                    if (has_m16) { \
                        const uint2 b0_01 = row16_live ? *reinterpret_cast<const uint2*>(sA_lo + 16 * SA_STRIDE + q_off)      : make_uint2(0u, 0u); \
                        const uint2 b2_01 = row16_live ? *reinterpret_cast<const uint2*>(sA_lo + 16 * SA_STRIDE + q_off + 16) : make_uint2(0u, 0u); \
                        const uint2 b1_01 = row24_live ? *reinterpret_cast<const uint2*>(sA_lo + 24 * SA_STRIDE + q_off)      : make_uint2(0u, 0u); \
                        const uint2 b3_01 = row24_live ? *reinterpret_cast<const uint2*>(sA_lo + 24 * SA_STRIDE + q_off + 16) : make_uint2(0u, 0u); \
                        mma_m16n8k16_f16(acc1, b0_01.x, b1_01.x, b2_01.x, b3_01.x, bl0, bl1); \
                        mma_m16n8k16_f16(acc1, b0_01.y, b1_01.y, b2_01.y, b3_01.y, bl2, bl3); \
                    } \
                } while (0)

                LO_STEP32(0, cur_wa.x, cur_wa.y, p01,  0, h00, h01, h02, h03);
                LO_STEP32(1, cur_wa.z, cur_wa.w, p01, 16, h10, h11, h12, h13);
                LO_STEP32(2, cur_wb.x, cur_wb.y, p23,  0, h20, h21, h22, h23);
                LO_STEP32(3, cur_wb.z, cur_wb.w, p23, 16, h30, h31, h32, h33);
                #undef LO_STEP32

                if (ks + 1 < K_STEPS) {
                    const int nxt_ks = ks + 1;
                    const uint8_t* nxt_ss = panel_base + static_cast<int64_t>(nxt_ks >> 1) * W13_SS_BYTES;
                    const int ks_off_b = (nxt_ks & 1) * 4096;
                    cur_wa = ld_w_u4(nxt_ss + ks_off_b + w_off, keep_l2, use_ef, l2_pol);
                    cur_wb = ld_w_u4(nxt_ss + ks_off_b + w_off + 512, keep_l2, use_ef, l2_pol);
                    if ((nxt_ks & 1) == 0) {
                        cur_ws_lo = ld_w_u32(nxt_ss + 8192 + (tid << 2), keep_l2, use_ef, l2_pol);
                        cur_ws_hi = ld_w_u8(nxt_ss + 8704 + tid, keep_l2, use_ef, l2_pol);
                    }
                }

                const uint16_t* sA_hi = sA[1] + x_off0;
                #define HI_STEP32(Q, H0, H1, H2, H3) do { \
                    const int q_off = (Q) * 32; \
                    const uint2 a0_01 = row0_live ? *reinterpret_cast<const uint2*>(sA_hi + q_off)                   : make_uint2(0u, 0u); \
                    const uint2 a2_01 = row0_live ? *reinterpret_cast<const uint2*>(sA_hi + q_off + 16)              : make_uint2(0u, 0u); \
                    const uint2 a1_01 = row8_live ? *reinterpret_cast<const uint2*>(sA_hi + 8 * SA_STRIDE + q_off)   : make_uint2(0u, 0u); \
                    const uint2 a3_01 = row8_live ? *reinterpret_cast<const uint2*>(sA_hi + 8 * SA_STRIDE + q_off + 16) : make_uint2(0u, 0u); \
                    mma_m16n8k16_f16(acc0, a0_01.x, a1_01.x, a2_01.x, a3_01.x, (H0), (H1)); \
                    mma_m16n8k16_f16(acc0, a0_01.y, a1_01.y, a2_01.y, a3_01.y, (H2), (H3)); \
                    if (has_m16) { \
                        const uint2 b0_01 = row16_live ? *reinterpret_cast<const uint2*>(sA_hi + 16 * SA_STRIDE + q_off)      : make_uint2(0u, 0u); \
                        const uint2 b2_01 = row16_live ? *reinterpret_cast<const uint2*>(sA_hi + 16 * SA_STRIDE + q_off + 16) : make_uint2(0u, 0u); \
                        const uint2 b1_01 = row24_live ? *reinterpret_cast<const uint2*>(sA_hi + 24 * SA_STRIDE + q_off)      : make_uint2(0u, 0u); \
                        const uint2 b3_01 = row24_live ? *reinterpret_cast<const uint2*>(sA_hi + 24 * SA_STRIDE + q_off + 16) : make_uint2(0u, 0u); \
                        mma_m16n8k16_f16(acc1, b0_01.x, b1_01.x, b2_01.x, b3_01.x, (H0), (H1)); \
                        mma_m16n8k16_f16(acc1, b0_01.y, b1_01.y, b2_01.y, b3_01.y, (H2), (H3)); \
                    } \
                } while (0)

                HI_STEP32(0, h00, h01, h02, h03);
                HI_STEP32(1, h10, h11, h12, h13);
                HI_STEP32(2, h20, h21, h22, h23);
                HI_STEP32(3, h30, h31, h32, h33);
                #undef HI_STEP32
            } else {
                const uint32_t w_arr[8] = {
                    cur_wa.x, cur_wa.y, cur_wa.z, cur_wa.w,
                    cur_wb.x, cur_wb.y, cur_wb.z, cur_wb.w
                };
                const uint16_t* sA_lo = sA[0] + x_off0;
                #pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const uint16_t e4_pair = static_cast<uint16_t>(((q < 2) ? p01 : p23) >> ((q & 1) * 16));
                    const uint32_t sc_f16x2 = cvt_e4m3x2_to_f16x2(e4_pair);
                    const uint32_t sc0 = splat_lo_f16(sc_f16x2);
                    const uint32_t sc1 = splat_hi_f16(sc_f16x2);
                    const int q_off = q * 32;
                    const uint2 a0_01 = row0_live ? *reinterpret_cast<const uint2*>(sA_lo + q_off)                   : make_uint2(0u, 0u);
                    const uint2 a2_01 = row0_live ? *reinterpret_cast<const uint2*>(sA_lo + q_off + 16)              : make_uint2(0u, 0u);
                    const uint2 a1_01 = row8_live ? *reinterpret_cast<const uint2*>(sA_lo + 8 * SA_STRIDE + q_off)   : make_uint2(0u, 0u);
                    const uint2 a3_01 = row8_live ? *reinterpret_cast<const uint2*>(sA_lo + 8 * SA_STRIDE + q_off + 16) : make_uint2(0u, 0u);
                    uint32_t lo01, lo23;
                    unpack_e2m1x8_lo(w_arr[2 * q + 0], lo01, lo23);
                    const uint32_t bl0 = mul_f16x2(lo01, sc0), bl1 = mul_f16x2(lo23, sc1);
                    mma_m16n8k16_f16(acc0, a0_01.x, a1_01.x, a2_01.x, a3_01.x, bl0, bl1);
                    unpack_e2m1x8_lo(w_arr[2 * q + 1], lo01, lo23);
                    const uint32_t bl2 = mul_f16x2(lo01, sc0), bl3 = mul_f16x2(lo23, sc1);
                    mma_m16n8k16_f16(acc0, a0_01.y, a1_01.y, a2_01.y, a3_01.y, bl2, bl3);
                    if (has_m16) {
                        const uint2 b0_01 = row16_live ? *reinterpret_cast<const uint2*>(sA_lo + 16 * SA_STRIDE + q_off)      : make_uint2(0u, 0u);
                        const uint2 b2_01 = row16_live ? *reinterpret_cast<const uint2*>(sA_lo + 16 * SA_STRIDE + q_off + 16) : make_uint2(0u, 0u);
                        const uint2 b1_01 = row24_live ? *reinterpret_cast<const uint2*>(sA_lo + 24 * SA_STRIDE + q_off)      : make_uint2(0u, 0u);
                        const uint2 b3_01 = row24_live ? *reinterpret_cast<const uint2*>(sA_lo + 24 * SA_STRIDE + q_off + 16) : make_uint2(0u, 0u);
                        mma_m16n8k16_f16(acc1, b0_01.x, b1_01.x, b2_01.x, b3_01.x, bl0, bl1);
                        mma_m16n8k16_f16(acc1, b0_01.y, b1_01.y, b2_01.y, b3_01.y, bl2, bl3);
                    }
                }
                const uint16_t* sA_hi = sA[1] + x_off0;
                #pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const uint16_t e4_pair = static_cast<uint16_t>(((q < 2) ? p01 : p23) >> ((q & 1) * 16));
                    const uint32_t sc_f16x2 = cvt_e4m3x2_to_f16x2(e4_pair);
                    const uint32_t sc0 = splat_lo_f16(sc_f16x2);
                    const uint32_t sc1 = splat_hi_f16(sc_f16x2);
                    const int q_off = q * 32;
                    const uint2 a0_01 = row0_live ? *reinterpret_cast<const uint2*>(sA_hi + q_off)                   : make_uint2(0u, 0u);
                    const uint2 a2_01 = row0_live ? *reinterpret_cast<const uint2*>(sA_hi + q_off + 16)              : make_uint2(0u, 0u);
                    const uint2 a1_01 = row8_live ? *reinterpret_cast<const uint2*>(sA_hi + 8 * SA_STRIDE + q_off)   : make_uint2(0u, 0u);
                    const uint2 a3_01 = row8_live ? *reinterpret_cast<const uint2*>(sA_hi + 8 * SA_STRIDE + q_off + 16) : make_uint2(0u, 0u);
                    uint32_t hi01, hi23;
                    unpack_e2m1x8_hi(w_arr[2 * q + 0], hi01, hi23);
                    const uint32_t bh0 = mul_f16x2(hi01, sc0), bh1 = mul_f16x2(hi23, sc1);
                    mma_m16n8k16_f16(acc0, a0_01.x, a1_01.x, a2_01.x, a3_01.x, bh0, bh1);
                    unpack_e2m1x8_hi(w_arr[2 * q + 1], hi01, hi23);
                    const uint32_t bh2 = mul_f16x2(hi01, sc0), bh3 = mul_f16x2(hi23, sc1);
                    mma_m16n8k16_f16(acc0, a0_01.y, a1_01.y, a2_01.y, a3_01.y, bh2, bh3);
                    if (has_m16) {
                        const uint2 b0_01 = row16_live ? *reinterpret_cast<const uint2*>(sA_hi + 16 * SA_STRIDE + q_off)      : make_uint2(0u, 0u);
                        const uint2 b2_01 = row16_live ? *reinterpret_cast<const uint2*>(sA_hi + 16 * SA_STRIDE + q_off + 16) : make_uint2(0u, 0u);
                        const uint2 b1_01 = row24_live ? *reinterpret_cast<const uint2*>(sA_hi + 24 * SA_STRIDE + q_off)      : make_uint2(0u, 0u);
                        const uint2 b3_01 = row24_live ? *reinterpret_cast<const uint2*>(sA_hi + 24 * SA_STRIDE + q_off + 16) : make_uint2(0u, 0u);
                        mma_m16n8k16_f16(acc1, b0_01.x, b1_01.x, b2_01.x, b3_01.x, bh0, bh1);
                        mma_m16n8k16_f16(acc1, b0_01.y, b1_01.y, b2_01.y, b3_01.y, bh2, bh3);
                    }
                }
            }

            if constexpr (TRIG_DEP) {
                if (ks == K_STEPS - 2 && nidx >= n_total && !trig_fired) {
                    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
                    trig_fired = true;
                }
            }

            if (ks + 1 < K_STEPS) {
                __syncthreads();
                load_a_slab_s(ks + 1, true);
                if constexpr (!USE_SINGLE_UNPACK) {
                    cur_wa = nxt_wa;
                    cur_wb = nxt_wb;
                }
                __syncthreads();
            }
        }

        const float scale = Ones[Tiles[t]];
        const int r0 = t * TILE + g;
        const int r1 = t * TILE + g + 8;
        const int r2 = t * TILE + g + 16;
        const int r3 = t * TILE + g + 24;

        auto emit_row_group = [&](int row_idx, int m_off, float a_lo, float a_hi) {
            const bool live = (m_off < live_m);
            const float rs = live ? RS[row_idx] : 0.0f;
            const float o0 = mul_rn(mul_rn(a_lo, rs), scale);
            const float o1 = mul_rn(mul_rn(a_hi, rs), scale);
            if constexpr (FUSE_SITU) {
                constexpr int INTER = 288;
                const int c_warp = nb * 16 + warp * 4;
                const int compact_base = TileCompactStart[t];
                uint16_t yb = 0u;
                if (live) yb = situ_pair_bf16_exact(pack_bf16x2(o0, o1), __ldg(Gammas + row_idx));
                const uint32_t p0 = static_cast<uint32_t>(yb)
                    | (__shfl_xor_sync(0xFFFFFFFFu, static_cast<uint32_t>(yb), 1) << 16);
                const uint32_t p1 = __shfl_xor_sync(0xFFFFFFFFu, p0, 2);
                if (lt == 0 && live) {
                    if constexpr (QDEQ_MODE >= 2) {
                        st_cg_u2(Out + static_cast<int64_t>(compact_base + m_off) * INTER + c_warp, make_uint2(p0, p1));
                    } else {
                        *reinterpret_cast<uint2*>(Out + static_cast<int64_t>(compact_base + m_off) * INTER + c_warp) = make_uint2(p0, p1);
                    }
                }
                if constexpr (QDEQ_MODE == 0) {
                    float am = live ? fabsf(bf16_bits_to_f32(yb)) : 0.0f;
                    am = fmaxf(am, __shfl_xor_sync(0xFFFFFFFFu, am, 1));
                    am = fmaxf(am, __shfl_xor_sync(0xFFFFFFFFu, am, 2));
                    if (lt == 0) sPanelAmax[warp][m_off] = am;
                }
            } else {
                const int n0 = nb * BN + warp * 8 + 2 * lt;
                if (live) {
                    *reinterpret_cast<uint32_t*>(Out + static_cast<int64_t>(row_idx) * 576 + n0) = pack_bf16x2(o0, o1);
                }
            }
        };

        emit_row_group(r0, g,      acc0[0], acc0[1]);
        emit_row_group(r1, g + 8,  acc0[2], acc0[3]);
        emit_row_group(r2, g + 16, acc1[0], acc1[1]);
        emit_row_group(r3, g + 24, acc1[2], acc1[3]);

        if constexpr (FUSE_SITU && QDEQ_MODE == 0) {
            const int compact_base = TileCompactStart[t];
            __syncthreads();
            if (tid < live_m) {
                float rmax = fmaxf(
                    fmaxf(sPanelAmax[0][tid], sPanelAmax[1][tid]),
                    fmaxf(sPanelAmax[2][tid], sPanelAmax[3][tid])
                );
                if (rmax > 0.0f) {
                    atomicMax(&RowAmaxW2[compact_base + tid], __float_as_uint(rmax));
                }
            }
        } else if constexpr (FUSE_SITU && QDEQ_MODE >= 2) {
            const int compact_base = TileCompactStart[t];
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

        ensure_next();
        if (nidx < n_total) __syncthreads();
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

    if constexpr (TRIG_DEP) {
        if (!trig_fired) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }
}

// =========================================================================
// Kernel 1D-v9 (TILE=32, BK_DOT=128): Exact Triton TILE=32 K-order Fallback
// =========================================================================
template <int PDL_WAIT_MODE, bool TRIG_DEP, bool ZERO_SCATTER, bool AS_ROW_MAJOR, bool FUSE_SITU, int QDEQ_MODE>
__global__ __launch_bounds__(128, 4)
void nvfp4_w13_ws5_t32_v16_kernel(
    const uint8_t* __restrict__ A,
    const uint8_t* __restrict__ AS,
    const float* __restrict__ RS,
    const int32_t* __restrict__ Row2Tok,
    const uint8_t* __restrict__ SLABS,
    const uint8_t* __restrict__ WS_BASE,
    const int32_t* __restrict__ P_OFF,
    const uint32_t* __restrict__ PATCHES,
    const float* __restrict__ Ones,
    const int32_t* __restrict__ Tiles,
    const int32_t* __restrict__ TileCompactStart,
    const float* __restrict__ Gammas,
    uint32_t* __restrict__ RowAmaxW2,
    __nv_bfloat16* __restrict__ Out,
    uint32_t* __restrict__ TileQdeqDone,
    uint32_t* __restrict__ WorkQueue,
    int N, int K, int MAX_CTAS,
    int evict_first, int lpt_mode
) {
    (void)N;
    (void)K;
    (void)TileQdeqDone;
    if constexpr (PDL_WAIT_MODE == 1) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }

    constexpr int WARPS = 4;
    constexpr int TILE = 32;
    constexpr int BN = 32;
    constexpr int N_PANELS = 18;
    constexpr int KP = 2304;
    constexpr int KG = 288;
    constexpr int BKP_HALF = 64;
    constexpr int SA_STRIDE = BKP_HALF + 8;
    constexpr int K_STEPS = 36;

    const int n_active = Tiles[2 * MAX_CTAS];
    const int n_total = n_active * N_PANELS;
    const int stride = static_cast<int>(gridDim.x);
    const int bidx = static_cast<int>(blockIdx.x);

    if (n_active <= 0) {
        if constexpr (PDL_WAIT_MODE == 2) asm volatile("griddepcontrol.wait;" ::: "memory");
        if constexpr (TRIG_DEP) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        return;
    }

    const bool use_dyn_q = (lpt_mode >= 2) && (WorkQueue != nullptr) && (n_total > stride);
    if (!use_dyn_q && bidx >= n_total) {
        if constexpr (PDL_WAIT_MODE == 2) {
            asm volatile("griddepcontrol.wait;" ::: "memory");
        }
        if constexpr (TRIG_DEP) {
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
        return;
    }

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g = lane >> 2;
    const int lt = lane & 3;
    const int w_off = warp * 1024 + lane * 16;
    const int x_off0 = g * SA_STRIDE + 4 * lt;

    alignas(16) __shared__ uint16_t sA[2][2][TILE * SA_STRIDE];
    alignas(16) __shared__ int32_t  sTok[TILE];
    alignas(16) __shared__ float    sPanelAmax[(FUSE_SITU && QDEQ_MODE == 0) ? WARPS : 1][TILE];
    constexpr int MAX_LPT_TILES = 448;
    alignas(16) __shared__ int16_t sTileOrder[MAX_LPT_TILES];
    __shared__ int32_t sTileLastQdeq;
    __shared__ int32_t sDynIdx;

    const bool use_ef = (evict_first != 0);
    const uint64_t l2_pol = use_ef ? make_l2_evict_first_policy() : 0ULL;
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

        const int64_t panel_uid = static_cast<int64_t>(e_i32) * N_PANELS + nb;
        const uint8_t* panel_base = SLABS + panel_uid * static_cast<int64_t>(9 * W13_SS_BYTES);

        if constexpr (PDL_WAIT_MODE == 2) {
            if (is_first_tile) {
                if ((lane & 7) == 0) {
                    prefetch_l2(panel_base + w_off);
                    prefetch_l2(panel_base + w_off + 512);
                }
                if (tid < 5) prefetch_l2(panel_base + 8192 + tid * 128);
                asm volatile("griddepcontrol.wait;" ::: "memory");
            }
        }
        if (live_m == 0) {
            ensure_next();
            idx = nidx;
            is_first_tile = false;
            continue;
        }

        const bool keep_l2 = (evict_first == 2) && (t + 1 < n_active) && (Tiles[t + 1] == e_i32);
        const uint32_t raw_base = static_cast<uint32_t>(WS_BASE[e_i32]);
        const uint32_t base4 = (raw_base & 0x7Fu) * 0x01010101u;

        uint4 cur_w = ld_w_u4(panel_base + w_off, keep_l2, use_ef, l2_pol);
        uint32_t cur_ws_lo = ld_w_u32(panel_base + 8192 + (tid << 2), keep_l2, use_ef, l2_pol);
        uint8_t  cur_ws_hi = ld_w_u8(panel_base + 8704 + tid, keep_l2, use_ef, l2_pol);

        int32_t p_cur = 0, p_end = 0;
        if ((raw_base & 0x80u) == 0u) {
            const int64_t warp_uid = panel_uid * WARPS + warp;
            p_cur = P_OFF[warp_uid];
            p_end = P_OFF[warp_uid + 1];
        }

        if constexpr (ZERO_SCATTER) {
            if (tid < live_m) sTok[tid] = Row2Tok[t * TILE + tid];
        }

        auto load_a_half = [&](int kh, int buf, bool use_smem_tok) {
            const int total_q2 = live_m * 8;
            const int kp0 = kh * BKP_HALF;
            const int kg0 = kh * 8;
            const int rbase = t * TILE;
            uint16_t* dst_lo = sA[buf][0];
            uint16_t* dst_hi = sA[buf][1];
            for (int aidx = tid; aidx < total_q2; aidx += 128) {
                const int lm = aidx >> 3;
                const int qk2 = aidx & 7;
                const int64_t row = static_cast<int64_t>(rbase + lm);
                const int64_t a_row = ZERO_SCATTER
                    ? static_cast<int64_t>(use_smem_tok ? sTok[lm] : Row2Tok[row])
                    : row;
                const uint2 a8 = ld_ca_u2(A + a_row * KP + kp0 + (qk2 << 3));
                const int kg = kg0 + qk2;
                uint8_t as_b;
                if constexpr (ZERO_SCATTER || AS_ROW_MAJOR) {
                    as_b = AS[a_row * KG + kg];
                } else {
                    as_b = AS[scale_offset_decode(row, kg, KG)];
                }
                const uint32_t as_f16x2 = splat_lo_f16(cvt_e4m3x2_to_f16x2(static_cast<uint16_t>(as_b)));
                uint32_t lo01, lo23, hi01, hi23;
                uint4 v_lo, v_hi;
                unpack_e2m1x8(a8.x, lo01, lo23, hi01, hi23);
                v_lo.x = mul_f16x2(lo01, as_f16x2); v_lo.y = mul_f16x2(lo23, as_f16x2);
                v_hi.x = mul_f16x2(hi01, as_f16x2); v_hi.y = mul_f16x2(hi23, as_f16x2);
                unpack_e2m1x8(a8.y, lo01, lo23, hi01, hi23);
                v_lo.z = mul_f16x2(lo01, as_f16x2); v_lo.w = mul_f16x2(lo23, as_f16x2);
                v_hi.z = mul_f16x2(hi01, as_f16x2); v_hi.w = mul_f16x2(hi23, as_f16x2);
                const int off = lm * SA_STRIDE + (qk2 << 3);
                *reinterpret_cast<uint4*>(dst_lo + off) = v_lo;
                *reinterpret_cast<uint4*>(dst_hi + off) = v_hi;
            }
        };

        load_a_half(0, 0, false);
        __syncthreads();

        float acc0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float acc1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        uint32_t ws_even = 0u, ws_odd = 0u;
        const bool row0_live  = (g < live_m);
        const bool row8_live  = (g + 8 < live_m);
        const bool has_m16    = (live_m > 16);
        const bool row16_live = (g + 16 < live_m);
        const bool row24_live = (g + 24 < live_m);

        for (int kh = 0; kh < K_STEPS; ++kh) {
            const int cur_buf = kh & 1;
            const int nxt_buf = cur_buf ^ 1;
            const int ss = kh >> 2;

            if ((kh & 3) == 0) {
                decode_ws5_w13_swar(cur_ws_lo, static_cast<uint32_t>(cur_ws_hi), base4, ws_even, ws_odd);
                if (__builtin_expect(p_cur < p_end, 0)) {
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

            if (kh + 1 < K_STEPS) {
                load_a_half(kh + 1, nxt_buf, true);
            }
            if (kh == K_STEPS - 2) {
                ensure_next();
                if (nidx < n_total) {
                    const int nlog_t = nidx / N_PANELS;
                    const int nnb = nidx - nlog_t * N_PANELS;
                    const int nt = map_tile(nlog_t);
                    const int64_t ne = static_cast<int64_t>(Tiles[nt]);
                    const uint8_t* nxt_pbase = SLABS + (ne * N_PANELS + nnb) * static_cast<int64_t>(9 * W13_SS_BYTES);
                    if ((lane & 7) == 0) {
                        prefetch_l2(nxt_pbase + w_off);
                        prefetch_l2(nxt_pbase + w_off + 512);
                    }
                    if (tid < 5) prefetch_l2(nxt_pbase + 8192 + tid * 128);
                }
            }

            const int src = (g << 2) | (kh & 3);
            const uint32_t ev = __shfl_sync(0xFFFFFFFFu, ws_even, src);
            const uint32_t od = __shfl_sync(0xFFFFFFFFu, ws_odd,  src);
            const uint32_t p01 = (lt < 2) ? ev : od;

            const uint16_t* sA_lo = sA[cur_buf][0] + x_off0;

            uint32_t h00, h01, h02, h03, h10, h11, h12, h13;

            #define LO_STEP_H(Q, WA, WB, SHIFT, H0, H1, H2, H3) do { \
                const uint16_t e4_pair = static_cast<uint16_t>(p01 >> (SHIFT)); \
                const uint32_t sc_f16x2 = cvt_e4m3x2_to_f16x2(e4_pair); \
                const uint32_t sc0 = splat_lo_f16(sc_f16x2); \
                const uint32_t sc1 = splat_hi_f16(sc_f16x2); \
                const int q_off = (Q) * 32; \
                const uint2 a0_01 = row0_live ? *reinterpret_cast<const uint2*>(sA_lo + q_off)                   : make_uint2(0u, 0u); \
                const uint2 a2_01 = row0_live ? *reinterpret_cast<const uint2*>(sA_lo + q_off + 16)              : make_uint2(0u, 0u); \
                const uint2 a1_01 = row8_live ? *reinterpret_cast<const uint2*>(sA_lo + 8 * SA_STRIDE + q_off)   : make_uint2(0u, 0u); \
                const uint2 a3_01 = row8_live ? *reinterpret_cast<const uint2*>(sA_lo + 8 * SA_STRIDE + q_off + 16) : make_uint2(0u, 0u); \
                uint32_t lo01, lo23, hi01, hi23; \
                unpack_e2m1x8((WA), lo01, lo23, hi01, hi23); \
                const uint32_t bl0 = mul_f16x2(lo01, sc0), bl1 = mul_f16x2(lo23, sc1); \
                (H0) = mul_f16x2(hi01, sc0); (H1) = mul_f16x2(hi23, sc1); \
                mma_m16n8k16_f16(acc0, a0_01.x, a1_01.x, a2_01.x, a3_01.x, bl0, bl1); \
                unpack_e2m1x8((WB), lo01, lo23, hi01, hi23); \
                const uint32_t bl2 = mul_f16x2(lo01, sc0), bl3 = mul_f16x2(lo23, sc1); \
                (H2) = mul_f16x2(hi01, sc0); (H3) = mul_f16x2(hi23, sc1); \
                mma_m16n8k16_f16(acc0, a0_01.y, a1_01.y, a2_01.y, a3_01.y, bl2, bl3); \
                if (has_m16) { \
                    const uint2 b0_01 = row16_live ? *reinterpret_cast<const uint2*>(sA_lo + 16 * SA_STRIDE + q_off)      : make_uint2(0u, 0u); \
                    const uint2 b2_01 = row16_live ? *reinterpret_cast<const uint2*>(sA_lo + 16 * SA_STRIDE + q_off + 16) : make_uint2(0u, 0u); \
                    const uint2 b1_01 = row24_live ? *reinterpret_cast<const uint2*>(sA_lo + 24 * SA_STRIDE + q_off)      : make_uint2(0u, 0u); \
                    const uint2 b3_01 = row24_live ? *reinterpret_cast<const uint2*>(sA_lo + 24 * SA_STRIDE + q_off + 16) : make_uint2(0u, 0u); \
                    mma_m16n8k16_f16(acc1, b0_01.x, b1_01.x, b2_01.x, b3_01.x, bl0, bl1); \
                    mma_m16n8k16_f16(acc1, b0_01.y, b1_01.y, b2_01.y, b3_01.y, bl2, bl3); \
                } \
            } while (0)

            LO_STEP_H(0, cur_w.x, cur_w.y,  0, h00, h01, h02, h03);
            LO_STEP_H(1, cur_w.z, cur_w.w, 16, h10, h11, h12, h13);
            #undef LO_STEP_H

            // cur_w & p01 are now dead: issue next half-slab weight load before HI_STEP_H(0..1)!
            if (kh + 1 < K_STEPS) {
                const int nkh = kh + 1;
                const uint8_t* nxt_ss = panel_base + static_cast<int64_t>(nkh >> 2) * W13_SS_BYTES;
                const int off_b = ((nkh >> 1) & 1) * 4096 + (nkh & 1) * 512;
                cur_w = ld_w_u4(nxt_ss + off_b + w_off, keep_l2, use_ef, l2_pol);
                if ((nkh & 3) == 0) {
                    cur_ws_lo = ld_w_u32(nxt_ss + 8192 + (tid << 2), keep_l2, use_ef, l2_pol);
                    cur_ws_hi = ld_w_u8(nxt_ss + 8704 + tid, keep_l2, use_ef, l2_pol);
                }
            }

            const uint16_t* sA_hi = sA[cur_buf][1] + x_off0;
            #define HI_STEP_H(Q, H0, H1, H2, H3) do { \
                const int q_off = (Q) * 32; \
                const uint2 a0_01 = row0_live ? *reinterpret_cast<const uint2*>(sA_hi + q_off)                   : make_uint2(0u, 0u); \
                const uint2 a2_01 = row0_live ? *reinterpret_cast<const uint2*>(sA_hi + q_off + 16)              : make_uint2(0u, 0u); \
                const uint2 a1_01 = row8_live ? *reinterpret_cast<const uint2*>(sA_hi + 8 * SA_STRIDE + q_off)   : make_uint2(0u, 0u); \
                const uint2 a3_01 = row8_live ? *reinterpret_cast<const uint2*>(sA_hi + 8 * SA_STRIDE + q_off + 16) : make_uint2(0u, 0u); \
                mma_m16n8k16_f16(acc0, a0_01.x, a1_01.x, a2_01.x, a3_01.x, (H0), (H1)); \
                mma_m16n8k16_f16(acc0, a0_01.y, a1_01.y, a2_01.y, a3_01.y, (H2), (H3)); \
                if (has_m16) { \
                    const uint2 b0_01 = row16_live ? *reinterpret_cast<const uint2*>(sA_hi + 16 * SA_STRIDE + q_off)      : make_uint2(0u, 0u); \
                    const uint2 b2_01 = row16_live ? *reinterpret_cast<const uint2*>(sA_hi + 16 * SA_STRIDE + q_off + 16) : make_uint2(0u, 0u); \
                    const uint2 b1_01 = row24_live ? *reinterpret_cast<const uint2*>(sA_hi + 24 * SA_STRIDE + q_off)      : make_uint2(0u, 0u); \
                    const uint2 b3_01 = row24_live ? *reinterpret_cast<const uint2*>(sA_hi + 24 * SA_STRIDE + q_off + 16) : make_uint2(0u, 0u); \
                    mma_m16n8k16_f16(acc1, b0_01.x, b1_01.x, b2_01.x, b3_01.x, (H0), (H1)); \
                    mma_m16n8k16_f16(acc1, b0_01.y, b1_01.y, b2_01.y, b3_01.y, (H2), (H3)); \
                } \
            } while (0)

            HI_STEP_H(0, h00, h01, h02, h03);
            HI_STEP_H(1, h10, h11, h12, h13);
            #undef HI_STEP_H

            if constexpr (TRIG_DEP) {
                if (kh == K_STEPS - 2 && nidx >= n_total && !trig_fired) {
                    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
                    trig_fired = true;
                }
            }

            if (kh + 1 < K_STEPS) {
                __syncthreads();
            }
        }

        const float scale = Ones[Tiles[t]];
        const int r0 = t * TILE + g;
        const int r1 = t * TILE + g + 8;
        const int r2 = t * TILE + g + 16;
        const int r3 = t * TILE + g + 24;

        auto emit_row_group = [&](int row_idx, int m_off, float a_lo, float a_hi) {
            const bool live = (m_off < live_m);
            const float rs = live ? RS[row_idx] : 0.0f;
            const float o0 = mul_rn(mul_rn(a_lo, rs), scale);
            const float o1 = mul_rn(mul_rn(a_hi, rs), scale);
            if constexpr (FUSE_SITU) {
                constexpr int INTER = 288;
                const int c_warp = nb * 16 + warp * 4;
                const int compact_base = TileCompactStart[t];
                uint16_t yb = 0u;
                if (live) yb = situ_pair_bf16_exact(pack_bf16x2(o0, o1), __ldg(Gammas + row_idx));
                const uint32_t p0 = static_cast<uint32_t>(yb)
                    | (__shfl_xor_sync(0xFFFFFFFFu, static_cast<uint32_t>(yb), 1) << 16);
                const uint32_t p1 = __shfl_xor_sync(0xFFFFFFFFu, p0, 2);
                if (lt == 0 && live) {
                    if constexpr (QDEQ_MODE >= 2) {
                        st_cg_u2(Out + static_cast<int64_t>(compact_base + m_off) * INTER + c_warp, make_uint2(p0, p1));
                    } else {
                        *reinterpret_cast<uint2*>(Out + static_cast<int64_t>(compact_base + m_off) * INTER + c_warp) = make_uint2(p0, p1);
                    }
                }
                if constexpr (QDEQ_MODE == 0) {
                    float am = live ? fabsf(bf16_bits_to_f32(yb)) : 0.0f;
                    am = fmaxf(am, __shfl_xor_sync(0xFFFFFFFFu, am, 1));
                    am = fmaxf(am, __shfl_xor_sync(0xFFFFFFFFu, am, 2));
                    if (lt == 0) sPanelAmax[warp][m_off] = am;
                }
            } else {
                const int n0 = nb * BN + warp * 8 + 2 * lt;
                if (live) {
                    *reinterpret_cast<uint32_t*>(Out + static_cast<int64_t>(row_idx) * 576 + n0) = pack_bf16x2(o0, o1);
                }
            }
        };

        emit_row_group(r0, g,      acc0[0], acc0[1]);
        emit_row_group(r1, g + 8,  acc0[2], acc0[3]);
        emit_row_group(r2, g + 16, acc1[0], acc1[1]);
        emit_row_group(r3, g + 24, acc1[2], acc1[3]);

        if constexpr (FUSE_SITU && QDEQ_MODE == 0) {
            const int compact_base = TileCompactStart[t];
            __syncthreads();
            if (tid < live_m) {
                float rmax = fmaxf(
                    fmaxf(sPanelAmax[0][tid], sPanelAmax[1][tid]),
                    fmaxf(sPanelAmax[2][tid], sPanelAmax[3][tid])
                );
                if (rmax > 0.0f) {
                    atomicMax(&RowAmaxW2[compact_base + tid], __float_as_uint(rmax));
                }
            }
        } else if constexpr (FUSE_SITU && QDEQ_MODE >= 2) {
            const int compact_base = TileCompactStart[t];
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

        ensure_next();
        if (nidx < n_total) __syncthreads();
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

    if constexpr (TRIG_DEP) {
        if (!trig_fired) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }
}

// =========================================================================
// Kernel 1.5-v9 (NEW!): Single-Pass L2-Hot SiTU -> Pre-Dequantized QDeqA FP16
//   - 1 warp (18 active lanes = 16 BF16s/lane) per compact row `r in [0..R_live-1]`.
//   - Eliminates 36x redundant OTF quant in W2 AND 18x/36x atomicMax in W13!
//   - Supports IN-PLACE (`QDeqOut == SiTU_In`, ZERO extra VRAM, ZERO new L2 lines!)
//     AND OUT-OF-PLACE (`QDeqOut != SiTU_In`) via per-warp register load + `__syncwarp()`!
//   - Exact 0-ULP unfused `sub_rn` (`eval_nvfp4_half4_unfused`) + warp-reduced `row_amax`!
//   - Fires early PDL into W2 so W2's static `cur_w0` + `WS5_PACK` overlap 100%!
// =========================================================================
template <bool PDL_WAIT, bool TRIG_DEP, bool QUANT_FMA>
__global__ __launch_bounds__(128, 6)
void nvfp4_situ_quant_dequant_c48_v9_kernel(
    const __nv_bfloat16* __restrict__ SiTU_In,   // [R_max, 288] BF16 (100% L2-hot!)
    uint16_t* __restrict__ QDeqOut,              // [R_max, 2, 144] FP16 (may alias SiTU_In!)
    uint32_t* __restrict__ RowAmaxW2,            // [R_max] float-bits (written non-atomically!)
    const int32_t* __restrict__ Tiles,
    const int32_t* __restrict__ TileCompactStart,
    int MAX_CTAS,
    int TILE,
    int max_compact_rows
) {
    if constexpr (PDL_WAIT) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }
    if constexpr (TRIG_DEP) {
        // W2 only reads immutable Tiles + static weights before its `griddepcontrol.wait;`!
        asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }

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
        // Guarantees ALL 288 BF16 elements of row `r` are safely in registers before any
        // in-place write to `QDeqOut[r]`!
        __syncwarp();

        // Exact 0-ULP warp reduction across lanes 0..17 (lanes 18..31 contribute 0.0f)
        float row_amax = blk_amax;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            row_amax = fmaxf(row_amax, __shfl_xor_sync(0xFFFFFFFFu, row_amax, off));
        }

        // Only 2 FP32 divisions per 288-element row (lane 0 computes & broadcasts)!
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
// Kernel 2-v9: NV_W2 (N=4608, K=288, BN=128, 4 warps, TILE=8|16|32)
//   - FUSE_QUANT_MODE:
//       0: Pre-quantized NVFP4 `A[row, 144]` + `AS` + `RS` (legacy)
//       1: OTF BF16 quant-dequant `SiTU_Out[compact, 288]` + `RowAmaxW2` (legacy v8)
//       2: V9 FAST PATH! L2-hot pre-dequantized `QDeqA[compact, 2, 144]` + `RowAmaxW2`
//          - Issues `cur_w0` (+ `cur_w1` for BK_DOT=64) AND `WS5_PACK` decode BEFORE
//            `griddepcontrol.wait` on the first tile -> 100% hides quant_dequant!
//          - Zero `div.rn.f32`, zero quant ALU, ~25 fewer regs -> 6 CTAs/SM, 0 spills!
//   - Single-pass `unpack_e2m1x8` in BOTH `BK_DOT=32` AND `BK_DOT=64` (-50% CVT!).
//   - Zero-fill & dead-`LDS` elimination (`g < live_m` predication, literal nd == 0!).
//   - Pure `ld.cs` weights (ZERO runtime `keep_l2` branch, ZERO overlap spin!).
//   - Full `FC=0..4` support (FC=4 compact + combine_c32_v9, FC=3 6-slice L2-pinned!).
// =========================================================================
template <int TILE, int BK_DOT, int MIN_CTA, int CPP, int PDL_WAIT_MODE, bool TRIG_DEP,
          bool AS_ROW_MAJOR, int FUSE_QUANT_MODE, int FUSE_COMBINE, bool QUANT_FMA>
__global__ __launch_bounds__(128, MIN_CTA)
void nvfp4_w2_ws5_c48_v16_kernel(
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
    uint32_t* __restrict__ BufGen,
    __nv_bfloat16* __restrict__ SendBuf,
    uint32_t* __restrict__ WorkQueue,
    int N, int K, int MAX_CTAS, int T, int as_row_stride, int down_row_stride,
    int evict_first, int lpt_mode
) {
    static_assert(TILE == 8 || TILE == 16 || TILE == 32, "TILE must be 8, 16, or 32");
    static_assert(BK_DOT == 32 || BK_DOT == 64, "BK_DOT must be 32 or 64");
    static_assert(FUSE_QUANT_MODE >= 0 && FUSE_QUANT_MODE <= 2, "FUSE_QUANT_MODE must be 0..2");
    static_assert(FUSE_COMBINE >= 0 && FUSE_COMBINE <= 4, "FUSE_COMBINE must be 0..4");
    (void)N;
    (void)K;
    (void)BufGen;

    constexpr bool IN_KERNEL_COMBINE = (FUSE_COMBINE >= 1 && FUSE_COMBINE <= 3);
    constexpr bool COMPACT_OUT       = (FUSE_COMBINE == 4);
    constexpr bool USE_COMPACT_IN    = (FUSE_QUANT_MODE != 0);

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
    constexpr int M_STEPS = (TILE <= 16) ? 1 : 2;
    constexpr int SA_STRIDE = KP + 8;  // 152 halfs = 304 B -> 16B aligned & 0-way LDS.32!
    constexpr int N_PANELS = 36;

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g = lane >> 2;
    const int lt = lane & 3;
    // ws_row_base computed locally before K-loop to minimize register live range

    const int n_active = Tiles[2 * MAX_CTAS];
    const int bidx = static_cast<int>(blockIdx.x);
    const int stride = static_cast<int>(gridDim.x);

    if (n_active == 0) {
        if constexpr (PDL_WAIT_MODE == 2) {
            asm volatile("griddepcontrol.wait;" ::: "memory");
        }
        if constexpr (IN_KERNEL_COMBINE) {
            if constexpr (FUSE_COMBINE == 1) {
                const int p_slot = bidx / CPP;
                const int p_rank = bidx - p_slot * CPP;
                if (p_rank == 0 && p_slot < 6) {
                    int n_live_tok = T;
                    if (NLivePtr != nullptr) {
                        int nl = *NLivePtr;
                        if (nl < 0) nl = 0;
                        if (nl < n_live_tok) n_live_tok = nl;
                    }
                    const uint4 z4 = make_uint4(0u, 0u, 0u, 0u);
                    for (int w = 0; w < 6; ++w) {
                        const int nb = w * 6 + p_slot;
                        const int col0 = nb * BN + (tid & 15) * 8;
                        for (int m = (tid >> 4); m < n_live_tok; m += 8) {
                            *reinterpret_cast<uint4*>(SendBuf + static_cast<int64_t>(m) * H_DIM + col0) = z4;
                        }
                    }
                    __threadfence();
                    __syncthreads();
                }
            } else {
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
        }
        if constexpr (TRIG_DEP) {
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
        return;
    }

    const int n_total_all = n_active * N_PANELS;
    constexpr bool DYN_Q_ELIGIBLE = (!IN_KERNEL_COMBINE || FUSE_COMBINE == 2);
    const bool use_dyn_q = DYN_Q_ELIGIBLE && (lpt_mode >= 2) && (WorkQueue != nullptr) && (n_total_all > stride);
    if constexpr (!IN_KERNEL_COMBINE || FUSE_COMBINE == 2) {
        if (!use_dyn_q && bidx >= n_total_all) {
            if constexpr (PDL_WAIT_MODE == 2) {
                asm volatile("griddepcontrol.wait;" ::: "memory");
            }
            if constexpr (TRIG_DEP) {
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
            return;
        }
    } else if constexpr (FUSE_COMBINE == 3) {
        const int t_stride12 = stride / 12;
        if (!(n_active >= t_stride12 && (stride % 12) == 0) && bidx >= n_active * N_PANELS) {
            if constexpr (PDL_WAIT_MODE == 2) {
                asm volatile("griddepcontrol.wait;" ::: "memory");
            }
            if constexpr (TRIG_DEP) {
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
            return;
        }
    }

    const bool use_ef = (evict_first != 0);
    const uint64_t l2_pol = use_ef ? make_l2_evict_first_policy() : 0ULL;

    constexpr int MAX_LPT_TILES = (TILE >= 32) ? 448 : 512;
    alignas(16) __shared__ uint16_t sA[2][M_ROWS * SA_STRIDE];
    alignas(16) __shared__ uint8_t  sWS[2304];
    alignas(16) __shared__ float    sRowEnc[(FUSE_QUANT_MODE == 1) ? M_ROWS : 1];
    alignas(16) __shared__ float    sRowDec[(FUSE_QUANT_MODE == 1) ? M_ROWS : 1];
    alignas(16) __shared__ float    sRowOut[(FUSE_QUANT_MODE == 1) ? M_ROWS : 1];
    alignas(16) __shared__ float    sRowAmax[(FUSE_QUANT_MODE != 0) ? M_ROWS : 1];
    alignas(16) __shared__ int16_t  sTileOrder[MAX_LPT_TILES];
    __shared__ int32_t sIsLastPanel;
    __shared__ int32_t sDynIdx;

    // Built BEFORE griddepcontrol.wait (Tiles is router output -> 100% hidden behind W13 tail!)
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

    auto prefetch_next_tile = [&](int nt, int nnb) {
        const int64_t ne = static_cast<int64_t>(Tiles[nt]);
        const int64_t npuid = ne * N_PANELS + nnb;
        const uint8_t* nwp = W_PACK + (npuid * WARPS + warp) * static_cast<int64_t>(K_STEPS) * 512;
        if ((lane & 7) == 0) {
            prefetch_l2(nwp + lane * 16);
            prefetch_l2(nwp + 512 + lane * 16);
        } else if ((lane & 7) == 4) {
            prefetch_l2(nwp + 1024 + (lane - 4) * 16);
            prefetch_l2(nwp + 1536 + (lane - 4) * 16);
        }
        if (tid < 12) {
            prefetch_l2(WS5_PACK + npuid * 360 + tid * 32);
        }
    };

    auto run_tile = [&](int t, int nb, bool is_very_first, auto&& resolve_next_fn,
                        __nv_bfloat16* down_panel_buf) {
        const int e_i32 = Tiles[t];
        const int end_row = Tiles[MAX_CTAS + t];
        int live_m = end_row - t * TILE;
        if (live_m > TILE) live_m = TILE;
        if (live_m < 0) live_m = 0;

        const bool keep_l2 = (evict_first == 2) && (t + 1 < n_active) && (Tiles[t + 1] == e_i32);
        const int64_t panel_uid = static_cast<int64_t>(e_i32) * N_PANELS + nb;
        const int64_t warp_uid = panel_uid * WARPS + warp;
        const uint8_t* wp_base = W_PACK + warp_uid * static_cast<int64_t>(K_STEPS) * 512;

        // V16: In FUSE_QUANT_MODE == 2 (QDeqA), issue `cur_w0` (and `cur_w1` for BK_DOT=64)
        // + pre-wait L2 prefetch of ks=1..3 IMMEDIATELY before `WS5_PACK` & `gdc_wait`!
        uint4 cur_w0 = make_uint4(0u, 0u, 0u, 0u);
        uint4 cur_w1 = make_uint4(0u, 0u, 0u, 0u);
        if constexpr (FUSE_QUANT_MODE == 2) {
            if (live_m > 0) {
                cur_w0 = ld_w_u4(wp_base + lane * 16, keep_l2, use_ef, l2_pol);
                if constexpr (BK_DOT == 64) {
                    cur_w1 = ld_w_u4(wp_base + 512 + lane * 16, keep_l2, use_ef, l2_pol);
                }
            }
            if constexpr (PDL_WAIT_MODE == 2) {
                if (is_very_first) {
                    if ((lane & 7) == 0) {
                        prefetch_l2(wp_base + 512 + lane * 16);
                        prefetch_l2(wp_base + 1024 + lane * 16);
                    } else if ((lane & 7) == 4) {
                        prefetch_l2(wp_base + 1536 + (lane - 4) * 16);
                    }
                }
            }
        } else if constexpr (PDL_WAIT_MODE == 2) {
            if (is_very_first) {
                if ((lane & 7) == 0) {
                    prefetch_l2(wp_base + lane * 16);
                    prefetch_l2(wp_base + 512 + lane * 16);
                } else if ((lane & 7) == 4) {
                    prefetch_l2(wp_base + 1024 + (lane - 4) * 16);
                    prefetch_l2(wp_base + 1536 + (lane - 4) * 16);
                }
            }
        }

        const uint32_t raw_base = static_cast<uint32_t>(WS_BASE[e_i32]);
        const uint32_t base4 = (raw_base & 0x7Fu) * 0x01010101u;
        const bool exp_no_patch = (raw_base & 0x80u) != 0u;

        if (tid < 72) {
            const uint32_t* ws_lo = WS5_PACK + panel_uid * 360;
            const uint32_t* ws_hi = ws_lo + 288;
            const uint4 lo4 = ld_w_u4(ws_lo + tid * 4, keep_l2, use_ef, l2_pol);
            const uint32_t hi4 = ld_w_u32(ws_hi + tid, keep_l2, use_ef, l2_pol);
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
                if (is_very_first && tid == 0 && p_cur < p_end) {
                    prefetch_l2(PATCHES + p_cur);
                }
            }
        }

        // V9-3: ZERO `sA` zero-fill loop! Inactive rows pass literal `0u` to MMA!
        if constexpr (PDL_WAIT_MODE == 2) {
            if (!waited_pdl) {
                asm volatile("griddepcontrol.wait;" ::: "memory");
                waited_pdl = true;
            }
        }
        if (live_m == 0) return;

        if constexpr (FUSE_QUANT_MODE == 1) {
            if ((lane & 7) == 0) {
                prefetch_l2(wp_base + lane * 16);
                prefetch_l2(wp_base + 512 + lane * 16);
            } else if ((lane & 7) == 4) {
                prefetch_l2(wp_base + 1024 + (lane - 4) * 16);
                prefetch_l2(wp_base + 1536 + (lane - 4) * 16);
            }
        }

        const int compact_tile_base = (USE_COMPACT_IN || (FUSE_COMBINE != 0)) ? TileCompactStart[t] : 0;

        // Scoped activation load BEFORE `acc` (0 spills!)
        {
            if constexpr (FUSE_QUANT_MODE == 2) {
                // V9 FAST PATH: Direct L2-hot uint4 copy from pre-dequantized QDeqA[compact, 2, 144]!
                // Zero div.rn.f32, zero cvt.e2m1x2, zero cvt.e4m3x2, zero mul_f16x2, 100% overlapped w/ cur_w0!
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
                const int rbase = t * TILE;
                for (int aidx = tid; aidx < total_q2; aidx += 128) {
                    const int lm = aidx / KG;
                    const int qk2 = aidx - lm * KG;
                    const int64_t row = static_cast<int64_t>(rbase + lm);
                    const uint2 a8 = ld_ca_u2(A + row * KP + (qk2 << 3));
                    uint8_t as_b;
                    if constexpr (AS_ROW_MAJOR) {
                        as_b = AS[row * as_row_stride + qk2];
                    } else {
                        as_b = AS[scale_offset_decode(row, qk2, KG)];
                    }
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
            cur_w0 = ld_w_u4(wp_base + lane * 16, keep_l2, use_ef, l2_pol);
            if constexpr (BK_DOT == 64) {
                cur_w1 = ld_w_u4(wp_base + 512 + lane * 16, keep_l2, use_ef, l2_pol);
            }
        }
        __syncthreads();

        if (__builtin_expect(p_cur < p_end, 0)) {
            for (int32_t pi = p_cur + tid; pi < p_end; pi += 128) {
                const uint32_t rec = PATCHES[pi];
                sWS[rec >> 8] = static_cast<uint8_t>(rec & 0xFFu);
            }
            __syncthreads();
        }

        float acc[M_STEPS][SUB_N][4];
        #pragma unroll
        for (int im = 0; im < M_STEPS; ++im) {
            #pragma unroll
            for (int sub = 0; sub < SUB_N; ++sub) {
                acc[im][sub][0] = 0.0f; acc[im][sub][1] = 0.0f;
                acc[im][sub][2] = 0.0f; acc[im][sub][3] = 0.0f;
            }
        }

        const bool row0_live = (g < live_m);
        [[maybe_unused]] const bool row8_live  = (TILE > 8)  && (g + 8 < live_m);
        [[maybe_unused]] const bool has_m16    = (TILE > 16) && (live_m > 16);
        [[maybe_unused]] const bool row16_live = (TILE > 16) && (g + 16 < live_m);
        [[maybe_unused]] const bool row24_live = (TILE > 16) && (g + 24 < live_m);
        const int ws_row_base = (warp * SUB_N) * 8 + g;
        const int x0_0 = g * SA_STRIDE + 2 * lt;
        const uint16_t* sA0_base = sA[0] + x0_0;
        const uint16_t* sA1_base = sA[1] + x0_0;

        if constexpr (BK_DOT == 32) {
            bool step_has_next = true;
            #pragma unroll
            for (int ks = 0; ks < K_STEPS; ++ks) {
                uint4 nxt_w;
                if (ks + 1 < K_STEPS) {
                    nxt_w = ld_w_u4(wp_base + static_cast<int64_t>(ks + 1) * 512 + lane * 16, keep_l2, use_ef, l2_pol);
                }
                if (ks == K_STEPS - 2) {
                    int next_t = 0, next_nb = 0;
                    step_has_next = resolve_next_fn(next_t, next_nb);
                    if (step_has_next) prefetch_next_tile(next_t, next_nb);
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
                        mma_m16n8k16_f16(acc[0][sub], a0, a1, a2, a3, bl0, bl1);
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
                        mma_m16n8k16_f16(acc[0][sub], a0, a1, a2, a3, bh0, bh1);
                    }
                    if constexpr (M_STEPS == 2) {
                        if (has_m16) {
                            const uint32_t c0 = row16_live ? *reinterpret_cast<const uint32_t*>(sA0_base + 16 * SA_STRIDE + ks_off)     : 0u;
                            const uint32_t c2 = row16_live ? *reinterpret_cast<const uint32_t*>(sA0_base + 16 * SA_STRIDE + ks_off + 8) : 0u;
                            const uint32_t c1 = row24_live ? *reinterpret_cast<const uint32_t*>(sA0_base + 24 * SA_STRIDE + ks_off)     : 0u;
                            const uint32_t c3 = row24_live ? *reinterpret_cast<const uint32_t*>(sA0_base + 24 * SA_STRIDE + ks_off + 8) : 0u;
                            mma_m16n8k16_f16(acc[1][sub], c0, c1, c2, c3, bl0, bl1);

                            const uint32_t d0 = row16_live ? *reinterpret_cast<const uint32_t*>(sA1_base + 16 * SA_STRIDE + ks_off)     : 0u;
                            const uint32_t d2 = row16_live ? *reinterpret_cast<const uint32_t*>(sA1_base + 16 * SA_STRIDE + ks_off + 8) : 0u;
                            const uint32_t d1 = row24_live ? *reinterpret_cast<const uint32_t*>(sA1_base + 24 * SA_STRIDE + ks_off)     : 0u;
                            const uint32_t d3 = row24_live ? *reinterpret_cast<const uint32_t*>(sA1_base + 24 * SA_STRIDE + ks_off + 8) : 0u;
                            mma_m16n8k16_f16(acc[1][sub], d0, d1, d2, d3, bh0, bh1);
                        }
                    }
                }

                if constexpr (TRIG_DEP) {
                    if (!step_has_next && ks == K_STEPS - 2 && !trig_fired) {
                        asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
                        trig_fired = true;
                    }
                }
                cur_w0 = nxt_w;
            }
        } else {
            // Exact TILE==32 order (BK_DOT=64): V9 single-pass unpack (-50% CVT, only 4 hi regs!)
            bool step_has_next = true;
            #pragma unroll
            for (int p = 0; p < 4; ++p) {
                const int ks0 = p * 2;
                const int ks1 = ks0 + 1;
                uint4 nxt_w0 = ld_w_u4(wp_base + static_cast<int64_t>(ks0 + 2) * 512 + lane * 16, keep_l2, use_ef, l2_pol);
                if (p == 3) {
                    int next_t = 0, next_nb = 0;
                    step_has_next = resolve_next_fn(next_t, next_nb);
                    if (step_has_next) prefetch_next_tile(next_t, next_nb);
                }

                const int o0 = ks0 * 16;
                const int o1 = ks1 * 16;
                const uint32_t w0_sub[4] = {cur_w0.x, cur_w0.y, cur_w0.z, cur_w0.w};
                const uint32_t w1_sub[4] = {cur_w1.x, cur_w1.y, cur_w1.z, cur_w1.w};

                #pragma unroll
                for (int sub = 0; sub < SUB_N; ++sub) {
                    const int s_idx0 = (ws_row_base + sub * 8) * 18 + ks0 * 2;
                    const uint16_t ws_p0 = *reinterpret_cast<const uint16_t*>(sWS + s_idx0);
                    const uint16_t ws_p1 = *reinterpret_cast<const uint16_t*>(sWS + s_idx0 + 2);
                    const uint32_t wf0 = cvt_e4m3x2_to_f16x2(ws_p0);
                    const uint32_t wf1 = cvt_e4m3x2_to_f16x2(ws_p1);
                    const uint32_t s00 = splat_lo_f16(wf0), s01 = splat_hi_f16(wf0);
                    const uint32_t s10 = splat_lo_f16(wf1), s11 = splat_hi_f16(wf1);

                    uint32_t l0a, l0b, h0a, h0b, l1a, l1b, h1a, h1b;
                    unpack_e2m1x8(w0_sub[sub], l0a, l0b, h0a, h0b);
                    l0a = mul_f16x2(l0a, s00); l0b = mul_f16x2(l0b, s01);
                    h0a = mul_f16x2(h0a, s00); h0b = mul_f16x2(h0b, s01);
                    unpack_e2m1x8(w1_sub[sub], l1a, l1b, h1a, h1b);
                    l1a = mul_f16x2(l1a, s10); l1b = mul_f16x2(l1b, s11);
                    h1a = mul_f16x2(h1a, s10); h1b = mul_f16x2(h1b, s11);

                    // Exact lo0, lo1 order
                    {
                        const uint32_t a0 = row0_live ? *reinterpret_cast<const uint32_t*>(sA0_base + o0)                   : 0u;
                        const uint32_t a2 = row0_live ? *reinterpret_cast<const uint32_t*>(sA0_base + o0 + 8)               : 0u;
                        const uint32_t a1 = row8_live ? *reinterpret_cast<const uint32_t*>(sA0_base + 8 * SA_STRIDE + o0)   : 0u;
                        const uint32_t a3 = row8_live ? *reinterpret_cast<const uint32_t*>(sA0_base + 8 * SA_STRIDE + o0 + 8) : 0u;
                        mma_m16n8k16_f16(acc[0][sub], a0, a1, a2, a3, l0a, l0b);

                        const uint32_t b0 = row0_live ? *reinterpret_cast<const uint32_t*>(sA0_base + o1)                   : 0u;
                        const uint32_t b2 = row0_live ? *reinterpret_cast<const uint32_t*>(sA0_base + o1 + 8)               : 0u;
                        const uint32_t b1 = row8_live ? *reinterpret_cast<const uint32_t*>(sA0_base + 8 * SA_STRIDE + o1)   : 0u;
                        const uint32_t b3 = row8_live ? *reinterpret_cast<const uint32_t*>(sA0_base + 8 * SA_STRIDE + o1 + 8) : 0u;
                        mma_m16n8k16_f16(acc[0][sub], b0, b1, b2, b3, l1a, l1b);

                        if constexpr (M_STEPS == 2) {
                            if (has_m16) {
                                const uint32_t c0 = row16_live ? *reinterpret_cast<const uint32_t*>(sA0_base + 16 * SA_STRIDE + o0)     : 0u;
                                const uint32_t c2 = row16_live ? *reinterpret_cast<const uint32_t*>(sA0_base + 16 * SA_STRIDE + o0 + 8) : 0u;
                                const uint32_t c1 = row24_live ? *reinterpret_cast<const uint32_t*>(sA0_base + 24 * SA_STRIDE + o0)     : 0u;
                                const uint32_t c3 = row24_live ? *reinterpret_cast<const uint32_t*>(sA0_base + 24 * SA_STRIDE + o0 + 8) : 0u;
                                mma_m16n8k16_f16(acc[1][sub], c0, c1, c2, c3, l0a, l0b);

                                const uint32_t d0 = row16_live ? *reinterpret_cast<const uint32_t*>(sA0_base + 16 * SA_STRIDE + o1)     : 0u;
                                const uint32_t d2 = row16_live ? *reinterpret_cast<const uint32_t*>(sA0_base + 16 * SA_STRIDE + o1 + 8) : 0u;
                                const uint32_t d1 = row24_live ? *reinterpret_cast<const uint32_t*>(sA0_base + 24 * SA_STRIDE + o1)     : 0u;
                                const uint32_t d3 = row24_live ? *reinterpret_cast<const uint32_t*>(sA0_base + 24 * SA_STRIDE + o1 + 8) : 0u;
                                mma_m16n8k16_f16(acc[1][sub], d0, d1, d2, d3, l1a, l1b);
                            }
                        }
                    }

                    // Exact hi0, hi1 order
                    {
                        const uint32_t a0 = row0_live ? *reinterpret_cast<const uint32_t*>(sA1_base + o0)                   : 0u;
                        const uint32_t a2 = row0_live ? *reinterpret_cast<const uint32_t*>(sA1_base + o0 + 8)               : 0u;
                        const uint32_t a1 = row8_live ? *reinterpret_cast<const uint32_t*>(sA1_base + 8 * SA_STRIDE + o0)   : 0u;
                        const uint32_t a3 = row8_live ? *reinterpret_cast<const uint32_t*>(sA1_base + 8 * SA_STRIDE + o0 + 8) : 0u;
                        mma_m16n8k16_f16(acc[0][sub], a0, a1, a2, a3, h0a, h0b);

                        const uint32_t b0 = row0_live ? *reinterpret_cast<const uint32_t*>(sA1_base + o1)                   : 0u;
                        const uint32_t b2 = row0_live ? *reinterpret_cast<const uint32_t*>(sA1_base + o1 + 8)               : 0u;
                        const uint32_t b1 = row8_live ? *reinterpret_cast<const uint32_t*>(sA1_base + 8 * SA_STRIDE + o1)   : 0u;
                        const uint32_t b3 = row8_live ? *reinterpret_cast<const uint32_t*>(sA1_base + 8 * SA_STRIDE + o1 + 8) : 0u;
                        mma_m16n8k16_f16(acc[0][sub], b0, b1, b2, b3, h1a, h1b);

                        if constexpr (M_STEPS == 2) {
                            if (has_m16) {
                                const uint32_t c0 = row16_live ? *reinterpret_cast<const uint32_t*>(sA1_base + 16 * SA_STRIDE + o0)     : 0u;
                                const uint32_t c2 = row16_live ? *reinterpret_cast<const uint32_t*>(sA1_base + 16 * SA_STRIDE + o0 + 8) : 0u;
                                const uint32_t c1 = row24_live ? *reinterpret_cast<const uint32_t*>(sA1_base + 24 * SA_STRIDE + o0)     : 0u;
                                const uint32_t c3 = row24_live ? *reinterpret_cast<const uint32_t*>(sA1_base + 24 * SA_STRIDE + o0 + 8) : 0u;
                                mma_m16n8k16_f16(acc[1][sub], c0, c1, c2, c3, h0a, h0b);

                                const uint32_t d0 = row16_live ? *reinterpret_cast<const uint32_t*>(sA1_base + 16 * SA_STRIDE + o1)     : 0u;
                                const uint32_t d2 = row16_live ? *reinterpret_cast<const uint32_t*>(sA1_base + 16 * SA_STRIDE + o1 + 8) : 0u;
                                const uint32_t d1 = row24_live ? *reinterpret_cast<const uint32_t*>(sA1_base + 24 * SA_STRIDE + o1)     : 0u;
                                const uint32_t d3 = row24_live ? *reinterpret_cast<const uint32_t*>(sA1_base + 24 * SA_STRIDE + o1 + 8) : 0u;
                                mma_m16n8k16_f16(acc[1][sub], d0, d1, d2, d3, h1a, h1b);
                            }
                        }
                    }
                }

                cur_w0 = nxt_w0;
                if (p < 3) {
                    cur_w1 = ld_w_u4(wp_base + static_cast<int64_t>(ks0 + 3) * 512 + lane * 16, keep_l2, use_ef, l2_pol);
                }
                if constexpr (TRIG_DEP) {
                    if (!step_has_next && p == 3 && !trig_fired) {
                        asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
                        trig_fired = true;
                    }
                }
            }

            // Tail ks = 8 (k = 256..287) with exact zero-pad 2nd m16n8k16 MMA for -0.0 parity
            {
                const int o8 = 8 * 16;
                const uint32_t w_sub[4] = {cur_w0.x, cur_w0.y, cur_w0.z, cur_w0.w};

                #pragma unroll
                for (int sub = 0; sub < SUB_N; ++sub) {
                    const int s_idx = (ws_row_base + sub * 8) * 18 + 16;
                    const uint16_t ws_pair = *reinterpret_cast<const uint16_t*>(sWS + s_idx);
                    const uint32_t ws_f16x2 = cvt_e4m3x2_to_f16x2(ws_pair);
                    const uint32_t sc0 = splat_lo_f16(ws_f16x2);
                    const uint32_t sc1 = splat_hi_f16(ws_f16x2);

                    uint32_t lo01, lo23, hi01, hi23;
                    unpack_e2m1x8(w_sub[sub], lo01, lo23, hi01, hi23);
                    const uint32_t bl0 = mul_f16x2(lo01, sc0), bl1 = mul_f16x2(lo23, sc1);
                    const uint32_t bh0 = mul_f16x2(hi01, sc0), bh1 = mul_f16x2(hi23, sc1);

                    {
                        const uint32_t a0 = row0_live ? *reinterpret_cast<const uint32_t*>(sA0_base + o8)                   : 0u;
                        const uint32_t a2 = row0_live ? *reinterpret_cast<const uint32_t*>(sA0_base + o8 + 8)               : 0u;
                        const uint32_t a1 = row8_live ? *reinterpret_cast<const uint32_t*>(sA0_base + 8 * SA_STRIDE + o8)   : 0u;
                        const uint32_t a3 = row8_live ? *reinterpret_cast<const uint32_t*>(sA0_base + 8 * SA_STRIDE + o8 + 8) : 0u;
                        mma_m16n8k16_f16(acc[0][sub], a0, a1, a2, a3, bl0, bl1);
                        mma_m16n8k16_f16(acc[0][sub], 0u, 0u, 0u, 0u, 0u, 0u);
                    }
                    {
                        const uint32_t a0 = row0_live ? *reinterpret_cast<const uint32_t*>(sA1_base + o8)                   : 0u;
                        const uint32_t a2 = row0_live ? *reinterpret_cast<const uint32_t*>(sA1_base + o8 + 8)               : 0u;
                        const uint32_t a1 = row8_live ? *reinterpret_cast<const uint32_t*>(sA1_base + 8 * SA_STRIDE + o8)   : 0u;
                        const uint32_t a3 = row8_live ? *reinterpret_cast<const uint32_t*>(sA1_base + 8 * SA_STRIDE + o8 + 8) : 0u;
                        mma_m16n8k16_f16(acc[0][sub], a0, a1, a2, a3, bh0, bh1);
                        mma_m16n8k16_f16(acc[0][sub], 0u, 0u, 0u, 0u, 0u, 0u);
                    }
                    if constexpr (M_STEPS == 2) {
                        if (has_m16) {
                            const uint32_t c0 = row16_live ? *reinterpret_cast<const uint32_t*>(sA0_base + 16 * SA_STRIDE + o8)     : 0u;
                            const uint32_t c2 = row16_live ? *reinterpret_cast<const uint32_t*>(sA0_base + 16 * SA_STRIDE + o8 + 8) : 0u;
                            const uint32_t c1 = row24_live ? *reinterpret_cast<const uint32_t*>(sA0_base + 24 * SA_STRIDE + o8)     : 0u;
                            const uint32_t c3 = row24_live ? *reinterpret_cast<const uint32_t*>(sA0_base + 24 * SA_STRIDE + o8 + 8) : 0u;
                            mma_m16n8k16_f16(acc[1][sub], c0, c1, c2, c3, bl0, bl1);
                            mma_m16n8k16_f16(acc[1][sub], 0u, 0u, 0u, 0u, 0u, 0u);

                            const uint32_t d0 = row16_live ? *reinterpret_cast<const uint32_t*>(sA1_base + 16 * SA_STRIDE + o8)     : 0u;
                            const uint32_t d2 = row16_live ? *reinterpret_cast<const uint32_t*>(sA1_base + 16 * SA_STRIDE + o8 + 8) : 0u;
                            const uint32_t d1 = row24_live ? *reinterpret_cast<const uint32_t*>(sA1_base + 24 * SA_STRIDE + o8)     : 0u;
                            const uint32_t d3 = row24_live ? *reinterpret_cast<const uint32_t*>(sA1_base + 24 * SA_STRIDE + o8 + 8) : 0u;
                            mma_m16n8k16_f16(acc[1][sub], d0, d1, d2, d3, bh0, bh1);
                            mma_m16n8k16_f16(acc[1][sub], 0u, 0u, 0u, 0u, 0u, 0u);
                        }
                    }
                }
            }
        }

        const int e_ep = Tiles[t];
        const float scale = Ones[e_ep];
        const float w_amax_e  = (USE_COMPACT_IN) ? __ldg(WAmax + e_ep) : 0.0f;
        const float rcp_denom = (USE_COMPACT_IN) ? rcp_rn(mul_rn(1536.0f, __ldg(WDenominator))) : 1.0f;

        auto emit_row_w2 = [&](int im, int r_off, int acc_pair_idx) {
            const int lm = im * 16 + r_off;
            const bool active = (lm < live_m);
            float row_scale = 0.0f;
            int64_t r = 0;
            if (active) {
                r = static_cast<int64_t>(t * TILE + lm);
                float rs;
                if constexpr (USE_COMPACT_IN) {
                    rs = mul_rn(mul_rn(sRowAmax[lm], w_amax_e), rcp_denom);
                } else {
                    rs = RS[r];
                }
                row_scale = mul_rn(rs, scale);
            }

            if constexpr (IN_KERNEL_COMBINE || COMPACT_OUT) {
                const int64_t crow = active ? static_cast<int64_t>(compact_tile_base + lm) : 0;
                __nv_bfloat16* row_ptr = IN_KERNEL_COMBINE
                    ? (down_panel_buf + crow * BN + warp * 32 + lt * 8)
                    : (Out + crow * H_DIM + nb * BN + warp * 32 + lt * 8);
                uint32_t pk[4];
                #pragma unroll
                for (int sub = 0; sub < SUB_N; ++sub) {
                    const float a0 = acc[im][sub][acc_pair_idx + 0];
                    const float a1 = acc[im][sub][acc_pair_idx + 1];
                    pk[sub] = active ? pack_bf16x2(mul_rn(a0, row_scale), mul_rn(a1, row_scale)) : 0u;
                }
                const uint4 v4 = transpose_4x4_bf16x2_u4(pk[0], pk[1], pk[2], pk[3], lt);
                if (active) {
                    st_cg_u4(row_ptr, v4);
                }
            } else {
                if (!active) return;
                const int col_base = nb * BN + warp * 32 + 2 * lt;
                #pragma unroll
                for (int sub = 0; sub < SUB_N; ++sub) {
                    const float a0 = acc[im][sub][acc_pair_idx + 0];
                    const float a1 = acc[im][sub][acc_pair_idx + 1];
                    *reinterpret_cast<uint32_t*>(Out + r * H_DIM + col_base + sub * 8) =
                        pack_bf16x2(mul_rn(a0, row_scale), mul_rn(a1, row_scale));
                }
            }
        };

        emit_row_w2(0, g, 0);
        if constexpr (TILE > 8) emit_row_w2(0, g + 8, 2);
        if constexpr (M_STEPS == 2) {
            emit_row_w2(1, g, 0);
            emit_row_w2(1, g + 8, 2);
        }
    };

    auto combine_panel_to_sendbuf_u4 = [&](int nb, const __nv_bfloat16* down_panel_buf) {
        int n_live_tok = T;
        if (NLivePtr != nullptr) {
            int nl = *NLivePtr;
            if (nl < 0) nl = 0;
            if (nl < n_live_tok) n_live_tok = nl;
        }
        const int col_in_panel = (tid & 15) * 8;
        const int col_global = nb * BN + col_in_panel;
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

            *reinterpret_cast<uint4*>(SendBuf + static_cast<int64_t>(m) * H_DIM + col_global) = make_uint4(
                pack_bf16x2(a0, a1), pack_bf16x2(a2, a3),
                pack_bf16x2(a4, a5), pack_bf16x2(a6, a7)
            );
        }
    };

    if constexpr (FUSE_COMBINE == 1) {
        const int p_slot = bidx / CPP;
        const int p_rank = bidx - p_slot * CPP;
        const int active_in_panel = (n_active < CPP) ? n_active : CPP;
        if (p_rank >= active_in_panel) {
            if constexpr (PDL_WAIT_MODE == 2) asm volatile("griddepcontrol.wait;" ::: "memory");
            if constexpr (TRIG_DEP) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            return;
        }
        for (int w = 0; w < 6; ++w) {
            const int nb = w * 6 + p_slot;
            __nv_bfloat16* down_panel_buf = Out + static_cast<int64_t>(nb) * down_row_stride * BN;
            for (int lt_i = p_rank; lt_i < n_active; lt_i += CPP) {
                const int t = map_tile(lt_i);
                const bool is_very_first = (w == 0 && lt_i == p_rank);
                const bool is_last_in_panel = (lt_i + CPP >= n_active);
                const bool is_very_last = (w == 5 && is_last_in_panel);
                const int next_lt = is_last_in_panel ? p_rank : (lt_i + CPP);
                const int next_t  = is_very_last ? 0 : map_tile(next_lt);
                const int next_nb = is_last_in_panel ? ((w + 1) * 6 + p_slot) : nb;
                auto resolve_static = [&](int& out_nt, int& out_nnb) -> bool {
                    out_nt = next_t; out_nnb = next_nb; return !is_very_last;
                };
                run_tile(t, nb, is_very_first, resolve_static, down_panel_buf);
                if (!is_last_in_panel) __syncthreads();
            }
            __threadfence();
            __syncthreads();
            if (tid == 0) {
                if (active_in_panel == 1) {
                    sIsLastPanel = 1;
                } else {
                    const uint32_t prev = atom_acq_rel_gpu_add(&PanelDone[nb], 1u);
                    if (prev == static_cast<uint32_t>(active_in_panel - 1)) {
                        st_rel_gpu_u32(&PanelDone[nb], 0u);
                        sIsLastPanel = 1;
                    } else {
                        sIsLastPanel = 0;
                    }
                }
            }
            __syncthreads();
            if (sIsLastPanel) {
                combine_panel_to_sendbuf_u4(nb, down_panel_buf);
                __threadfence();
                __syncthreads();
            } else if (w + 1 < 6) {
                __syncthreads();
            }
        }
    } else if constexpr (FUSE_COMBINE == 2) {
        const int n_total = n_total_all;
        uint32_t cur_nb_tiles = 0u;
        int idx = bidx;
        bool is_very_first = true;
        while (idx < n_total) {
            const int nb = idx / n_active;
            const int log_t = idx - nb * n_active;
            const int t  = map_tile(log_t);
            bool next_claimed = false;
            int nidx = idx + stride;
            auto resolve_next = [&](int& out_nt, int& out_nnb) -> bool {
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
                if (nidx >= n_total) { out_nt = 0; out_nnb = 0; return false; }
                out_nnb = nidx / n_active;
                out_nt  = map_tile(nidx - out_nnb * n_active);
                return true;
            };
            __nv_bfloat16* down_panel_buf = Out + static_cast<int64_t>(nb) * down_row_stride * BN;

            run_tile(t, nb, is_very_first, resolve_next, down_panel_buf);
            int tmp_nt = 0, next_nb = 0;
            const bool has_next = resolve_next(tmp_nt, next_nb);
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
                    combine_panel_to_sendbuf_u4(nb, down_panel_buf);
                    __threadfence();
                    __syncthreads();
                } else if (has_next) {
                    __syncthreads();
                }
                cur_nb_tiles = 0u;
            } else {
                __syncthreads();
            }
            idx = nidx;
            is_very_first = false;
        }
    } else if constexpr (FUSE_COMBINE == 3) {
        const int t_stride6  = stride / 6;
        const int t_stride12 = stride / 12;

        if (n_active >= t_stride12 && (stride % 12) == 0) {
            const int slice_p  = (n_active >= t_stride6) ? 6 : 12;
            const int n_slices = N_PANELS / slice_p;
            const int t_stride = stride / slice_p;
            const int t_rank   = bidx / slice_p;
            const int lp       = bidx - t_rank * slice_p;

            for (int s = 0; s < n_slices; ++s) {
                const int nb = s * slice_p + lp;
                __nv_bfloat16* down_panel_buf = Out + static_cast<int64_t>(nb) * down_row_stride * BN;

                for (int lt_i = t_rank; lt_i < n_active; lt_i += t_stride) {
                    const int t = map_tile(lt_i);
                    const bool is_very_first = (s == 0 && lt_i == t_rank);
                    const bool is_last_in_slice = (lt_i + t_stride >= n_active);
                    const bool is_very_last = (s == n_slices - 1 && is_last_in_slice);
                    const int next_lt = is_last_in_slice ? t_rank : (lt_i + t_stride);
                    const int next_t  = is_very_last ? 0 : map_tile(next_lt);
                    const int next_nb = is_last_in_slice ? ((s + 1) * slice_p + lp) : nb;
                    auto resolve_static = [&](int& out_nt, int& out_nnb) -> bool {
                        out_nt = next_t; out_nnb = next_nb; return !is_very_last;
                    };
                    run_tile(t, nb, is_very_first, resolve_static, down_panel_buf);
                    if (!is_last_in_slice) __syncthreads();
                }

                __threadfence();
                __syncthreads();
                if (tid == 0) {
                    const uint32_t prev = atom_acq_rel_gpu_add(&PanelDone[nb], 1u);
                    if (prev == static_cast<uint32_t>(t_stride - 1)) {
                        st_rel_gpu_u32(&PanelDone[nb], 0u);
                        sIsLastPanel = 1;
                    } else {
                        sIsLastPanel = 0;
                    }
                }
                __syncthreads();

                if (sIsLastPanel) {
                    combine_panel_to_sendbuf_u4(nb, down_panel_buf);
                    __threadfence();
                    __syncthreads();
                } else if (s + 1 < n_slices) {
                    __syncthreads();
                }
            }
        } else {
            const int n_total = n_total_all;
            for (int idx = bidx; idx < n_total; idx += stride) {
                const int log_t = idx / N_PANELS;
                const int nb = idx - log_t * N_PANELS;
                const int t  = map_tile(log_t);
                const bool is_very_first = (idx == bidx);
                const bool is_very_last  = (idx + stride >= n_total);
                const int nidx = idx + stride;
                const int next_lt = is_very_last ? 0 : (nidx / N_PANELS);
                const int next_t  = is_very_last ? 0 : map_tile(next_lt);
                const int next_nb = is_very_last ? 0 : (nidx - next_lt * N_PANELS);
                __nv_bfloat16* down_panel_buf = Out + static_cast<int64_t>(nb) * down_row_stride * BN;
                auto resolve_static = [&](int& out_nt, int& out_nnb) -> bool {
                    out_nt = next_t; out_nnb = next_nb; return !is_very_last;
                };
                run_tile(t, nb, is_very_first, resolve_static, down_panel_buf);

                __threadfence();
                __syncthreads();
                if (tid == 0) {
                    if (n_active == 1) {
                        sIsLastPanel = 1;
                    } else {
                        const uint32_t prev = atom_acq_rel_gpu_add(&PanelDone[nb], 1u);
                        if (prev == static_cast<uint32_t>(n_active - 1)) {
                            st_rel_gpu_u32(&PanelDone[nb], 0u);
                            sIsLastPanel = 1;
                        } else {
                            sIsLastPanel = 0;
                        }
                    }
                }
                __syncthreads();

                if (sIsLastPanel) {
                    combine_panel_to_sendbuf_u4(nb, down_panel_buf);
                    __threadfence();
                    __syncthreads();
                } else if (!is_very_last) {
                    __syncthreads();
                }
            }
        }
    } else {
        // FC == 0 (sparse) and FC == 4 (COMPACT_OUT): 36x Expert-Major + LPT + Dynamic WorkQueue!
        const int n_total = n_total_all;
        int idx = bidx;
        bool is_very_first = true;
        while (idx < n_total) {
            const int log_t = idx / N_PANELS;
            const int nb = idx - log_t * N_PANELS;
            const int t  = map_tile(log_t);
            bool next_claimed = false;
            int nidx = idx + stride;
            auto resolve_next = [&](int& out_nt, int& out_nnb) -> bool {
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
                if (nidx >= n_total) { out_nt = 0; out_nnb = 0; return false; }
                const int nlt = nidx / N_PANELS;
                out_nnb = nidx - nlt * N_PANELS;
                out_nt  = map_tile(nlt);
                return true;
            };
            run_tile(t, nb, is_very_first, resolve_next, nullptr);
            int tmp_nt = 0, tmp_nnb = 0;
            const bool has_next = resolve_next(tmp_nt, tmp_nnb);
            if (has_next) __syncthreads();
            idx = nidx;
            is_very_first = false;
        }
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

    if constexpr (TRIG_DEP) {
        if (!trig_fired) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }
}

// =========================================================================
// Kernel 3-v9: 0-ULP PDL + NLivePtr Vectorized SM121a Combine (H=4608, topk=8)
// =========================================================================
template <int BLOCK_THREADS, int COLS_PER_CTA, int MIN_CTA_C, bool PDL_WAIT, bool TRIG_DEP, bool HAS_NLIVE>
__global__ __launch_bounds__(BLOCK_THREADS, MIN_CTA_C)
void nvfp4_combine_c32_v9_kernel(
    const __nv_bfloat16* __restrict__ Down,
    const int32_t* __restrict__ Src2Row,
    const int32_t* __restrict__ NLivePtr,
    __nv_bfloat16* __restrict__ Out
) {
    const int token = static_cast<int>(blockIdx.y);
    const int tid = threadIdx.x;
    if constexpr (HAS_NLIVE) {
        if (token >= __ldg(NLivePtr)) {
            if constexpr (PDL_WAIT) {
                asm volatile("griddepcontrol.wait;" ::: "memory");
            }
            if constexpr (TRIG_DEP) {
                __syncthreads();
                if (tid == 0) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
            return;
        }
    }

    __shared__ int32_t sRows[8];
    if (tid < 8) {
        sRows[tid] = __ldg(Src2Row + static_cast<int64_t>(token) * 8 + tid);
    }
    __syncthreads();

    // V9 CRITICAL PDL FIX: Always complete `griddepcontrol.wait` STRICTLY BEFORE
    // `griddepcontrol.launch_dependents` (eliminates v6/v7/v8 3-grid PDL overlap race)!
    if constexpr (PDL_WAIT) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }
    if constexpr (TRIG_DEP) {
        __syncthreads();
        if (tid == 0) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }

    constexpr int64_t H_DIM = 4608;
    const int col = static_cast<int>(blockIdx.x) * COLS_PER_CTA + tid * 8;
    const uint4 z4 = make_uint4(0u, 0u, 0u, 0u);

    const int32_t r0 = sRows[0];
    const int32_t r1 = sRows[1];
    const int32_t r2 = sRows[2];
    const int32_t r3 = sRows[3];
    const uint4 v0 = (r0 >= 0) ? ld_cs_u4(Down + static_cast<int64_t>(r0) * H_DIM + col) : z4;
    const uint4 v1 = (r1 >= 0) ? ld_cs_u4(Down + static_cast<int64_t>(r1) * H_DIM + col) : z4;
    const uint4 v2 = (r2 >= 0) ? ld_cs_u4(Down + static_cast<int64_t>(r2) * H_DIM + col) : z4;
    const uint4 v3 = (r3 >= 0) ? ld_cs_u4(Down + static_cast<int64_t>(r3) * H_DIM + col) : z4;

    const int32_t r4 = sRows[4];
    const int32_t r5 = sRows[5];
    const int32_t r6 = sRows[6];
    const int32_t r7 = sRows[7];
    const uint4 v4 = (r4 >= 0) ? ld_cs_u4(Down + static_cast<int64_t>(r4) * H_DIM + col) : z4;

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
    const uint4 v5 = (r5 >= 0) ? ld_cs_u4(Down + static_cast<int64_t>(r5) * H_DIM + col) : z4;
    acc_u4(v1);
    const uint4 v6 = (r6 >= 0) ? ld_cs_u4(Down + static_cast<int64_t>(r6) * H_DIM + col) : z4;
    acc_u4(v2);
    const uint4 v7 = (r7 >= 0) ? ld_cs_u4(Down + static_cast<int64_t>(r7) * H_DIM + col) : z4;
    acc_u4(v3);
    acc_u4(v4);
    acc_u4(v5);
    acc_u4(v6);
    acc_u4(v7);

    *reinterpret_cast<uint4*>(Out + static_cast<int64_t>(token) * H_DIM + col) = make_uint4(
        pack_bf16x2(a0, a1), pack_bf16x2(a2, a3),
        pack_bf16x2(a4, a5), pack_bf16x2(a6, a7)
    );
}

// ---- Launch & Dispatch Helpers ----

template <typename KernelT, typename... Args>
static inline cudaError_t launch_nvfp4(
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

template <int BN, int MIN_CTA, bool ZS, bool RM, bool FS, int QM, bool AG>
static int dispatch_w13_t8_v16(
    dim3 grid, dim3 block, cudaStream_t stream, int pdl_wait_mode, int trig_dep,
    const uint8_t* A, const uint8_t* AS, const float* RS, const int32_t* Row2Tok,
    const uint8_t* SLABS, const uint8_t* WS_BASE,
    const int32_t* P_OFF, const uint32_t* PATCHES,
    const float* Ones, const int32_t* Tiles,
    const int32_t* TileCompactStart, const float* Gammas, uint32_t* RowAmaxW2,
    __nv_bfloat16* Out, uint32_t* TileQdeqDone, uint32_t* WorkQueue,
    int N, int K, int MAX_CTAS, int stride16, int stride32, int bn16_margin_pct,
    int evict_first, int lpt_mode
) {
    if (pdl_wait_mode == 0 && trig_dep != 0) pdl_wait_mode = 1;
    const bool use_pdl = (pdl_wait_mode != 0) || (trig_dep != 0);
    cudaError_t err = cudaSuccess;
    #define R8(PW, TR) err = launch_nvfp4( \
        nvfp4_w13_ws5_t8_v16_kernel<BN, MIN_CTA, PW, TR, ZS, RM, FS, QM, AG>, \
        grid, block, stream, use_pdl, \
        A, AS, RS, Row2Tok, SLABS, WS_BASE, P_OFF, PATCHES, Ones, Tiles, \
        TileCompactStart, Gammas, RowAmaxW2, Out, TileQdeqDone, WorkQueue, \
        N, K, MAX_CTAS, stride16, stride32, bn16_margin_pct, evict_first, lpt_mode)
    if      (pdl_wait_mode == 1 && trig_dep)  { R8(1, true);  }
    else if (pdl_wait_mode == 0 && trig_dep)  { R8(0, true);  }
    else if (pdl_wait_mode == 1 && !trig_dep) { R8(1, false); }
    else if (pdl_wait_mode == 2 && trig_dep)  { R8(2, true);  }
    else if (pdl_wait_mode == 2 && !trig_dep) { R8(2, false); }
    else                                      { R8(0, false); }
    #undef R8
    return static_cast<int>(err);
}

template <int TILE, int MIN_CTA, bool ZS, bool RM, bool FS, int QM>
static int dispatch_w13_t16_v16(
    dim3 grid, dim3 block, cudaStream_t stream, int pdl_wait_mode, int trig_dep,
    const uint8_t* A, const uint8_t* AS, const float* RS, const int32_t* Row2Tok,
    const uint8_t* SLABS, const uint8_t* WS_BASE,
    const int32_t* P_OFF, const uint32_t* PATCHES,
    const float* Ones, const int32_t* Tiles,
    const int32_t* TileCompactStart, const float* Gammas, uint32_t* RowAmaxW2,
    __nv_bfloat16* Out, uint32_t* TileQdeqDone, uint32_t* WorkQueue,
    int N, int K, int MAX_CTAS, int evict_first, int lpt_mode
) {
    if (pdl_wait_mode == 0 && trig_dep != 0) pdl_wait_mode = 1;
    const bool use_pdl = (pdl_wait_mode != 0) || (trig_dep != 0);
    cudaError_t err = cudaSuccess;
    #define R16(PW, TR) err = launch_nvfp4( \
        nvfp4_w13_ws5_t16_v16_kernel<TILE, MIN_CTA, PW, TR, ZS, RM, FS, QM>, \
        grid, block, stream, use_pdl, \
        A, AS, RS, Row2Tok, SLABS, WS_BASE, P_OFF, PATCHES, Ones, Tiles, \
        TileCompactStart, Gammas, RowAmaxW2, Out, TileQdeqDone, WorkQueue, \
        N, K, MAX_CTAS, evict_first, lpt_mode)
    if      (pdl_wait_mode == 1 && trig_dep)  { R16(1, true);  }
    else if (pdl_wait_mode == 0 && trig_dep)  { R16(0, true);  }
    else if (pdl_wait_mode == 1 && !trig_dep) { R16(1, false); }
    else if (pdl_wait_mode == 2 && trig_dep)  { R16(2, true);  }
    else if (pdl_wait_mode == 2 && !trig_dep) { R16(2, false); }
    else                                      { R16(0, false); }
    #undef R16
    return static_cast<int>(err);
}

template <int MIN_CTA, bool ZS, bool RM, bool FS, int QM>
static int dispatch_w13_t32_bk256_v16(
    dim3 grid, dim3 block, cudaStream_t stream, int pdl_wait_mode, int trig_dep,
    const uint8_t* A, const uint8_t* AS, const float* RS, const int32_t* Row2Tok,
    const uint8_t* SLABS, const uint8_t* WS_BASE,
    const int32_t* P_OFF, const uint32_t* PATCHES,
    const float* Ones, const int32_t* Tiles,
    const int32_t* TileCompactStart, const float* Gammas, uint32_t* RowAmaxW2,
    __nv_bfloat16* Out, uint32_t* TileQdeqDone, uint32_t* WorkQueue,
    int N, int K, int MAX_CTAS, int evict_first, int lpt_mode
) {
    if (pdl_wait_mode == 0 && trig_dep != 0) pdl_wait_mode = 1;
    const bool use_pdl = (pdl_wait_mode != 0) || (trig_dep != 0);
    cudaError_t err = cudaSuccess;
    #define R32B(PW, TR) err = launch_nvfp4( \
        nvfp4_w13_ws5_t32_bk256_v16_kernel<MIN_CTA, PW, TR, ZS, RM, FS, QM>, \
        grid, block, stream, use_pdl, \
        A, AS, RS, Row2Tok, SLABS, WS_BASE, P_OFF, PATCHES, Ones, Tiles, \
        TileCompactStart, Gammas, RowAmaxW2, Out, TileQdeqDone, WorkQueue, \
        N, K, MAX_CTAS, evict_first, lpt_mode)
    if      (pdl_wait_mode == 1 && trig_dep)  { R32B(1, true);  }
    else if (pdl_wait_mode == 0 && trig_dep)  { R32B(0, true);  }
    else if (pdl_wait_mode == 1 && !trig_dep) { R32B(1, false); }
    else if (pdl_wait_mode == 2 && trig_dep)  { R32B(2, true);  }
    else if (pdl_wait_mode == 2 && !trig_dep) { R32B(2, false); }
    else                                      { R32B(0, false); }
    #undef R32B
    return static_cast<int>(err);
}

template <bool ZS, bool RM, bool FS, int QM>
static int dispatch_w13_t32_v16(
    dim3 grid, dim3 block, cudaStream_t stream, int pdl_wait_mode, int trig_dep,
    const uint8_t* A, const uint8_t* AS, const float* RS, const int32_t* Row2Tok,
    const uint8_t* SLABS, const uint8_t* WS_BASE,
    const int32_t* P_OFF, const uint32_t* PATCHES,
    const float* Ones, const int32_t* Tiles,
    const int32_t* TileCompactStart, const float* Gammas, uint32_t* RowAmaxW2,
    __nv_bfloat16* Out, uint32_t* TileQdeqDone, uint32_t* WorkQueue,
    int N, int K, int MAX_CTAS, int evict_first, int lpt_mode
) {
    if (pdl_wait_mode == 0 && trig_dep != 0) pdl_wait_mode = 1;
    const bool use_pdl = (pdl_wait_mode != 0) || (trig_dep != 0);
    cudaError_t err = cudaSuccess;
    #define R32(PW, TR) err = launch_nvfp4( \
        nvfp4_w13_ws5_t32_v16_kernel<PW, TR, ZS, RM, FS, QM>, \
        grid, block, stream, use_pdl, \
        A, AS, RS, Row2Tok, SLABS, WS_BASE, P_OFF, PATCHES, Ones, Tiles, \
        TileCompactStart, Gammas, RowAmaxW2, Out, TileQdeqDone, WorkQueue, \
        N, K, MAX_CTAS, evict_first, lpt_mode)
    if      (pdl_wait_mode == 1 && trig_dep)  { R32(1, true);  }
    else if (pdl_wait_mode == 0 && trig_dep)  { R32(0, true);  }
    else if (pdl_wait_mode == 1 && !trig_dep) { R32(1, false); }
    else if (pdl_wait_mode == 2 && trig_dep)  { R32(2, true);  }
    else if (pdl_wait_mode == 2 && !trig_dep) { R32(2, false); }
    else                                      { R32(0, false); }
    #undef R32
    return static_cast<int>(err);
}

template <int TILE, int BK_DOT, int MIN_CTA, int CPP, bool RM, int FQ_MODE, int FC, bool QF>
static int dispatch_w2_v16(
    dim3 grid, dim3 block, cudaStream_t stream, int pdl_wait_mode, int trig_dep,
    const uint8_t* A, const uint8_t* AS, const float* RS,
    const uint32_t* RowAmaxW2, const float* WAmax, const float* WDenominator,
    const uint8_t* W_PACK, const uint32_t* WS5_PACK, const uint8_t* WS_BASE,
    const int32_t* P_OFF, const uint32_t* PATCHES,
    const float* Ones, const int32_t* Tiles, const int32_t* TileCompactStart,
    __nv_bfloat16* Out,
    const int32_t* Src2Compact, const int32_t* NLivePtr,
    uint32_t* PanelDone, uint32_t* BufGen, __nv_bfloat16* SendBuf, uint32_t* WorkQueue,
    int N, int K, int MAX_CTAS, int T, int as_row_stride, int down_row_stride,
    int evict_first, int lpt_mode
) {
    if (pdl_wait_mode == 0 && trig_dep != 0) pdl_wait_mode = 1;
    const bool use_pdl = (pdl_wait_mode != 0) || (trig_dep != 0);
    cudaError_t err = cudaSuccess;
    #define RUN2(PW, TR) err = launch_nvfp4( \
        nvfp4_w2_ws5_c48_v16_kernel<TILE, BK_DOT, MIN_CTA, CPP, PW, TR, RM, FQ_MODE, FC, QF>, \
        grid, block, stream, use_pdl, \
        A, AS, RS, RowAmaxW2, WAmax, WDenominator, \
        W_PACK, WS5_PACK, WS_BASE, P_OFF, PATCHES, Ones, Tiles, TileCompactStart, \
        Out, Src2Compact, NLivePtr, PanelDone, BufGen, SendBuf, WorkQueue, \
        N, K, MAX_CTAS, T, as_row_stride, down_row_stride, evict_first, lpt_mode)
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
// Wave-V10 Exports (100% Drop-In Superset of v9/v8/v7/v6/S7 ABIs + V10 Entrypoints)
// =========================================================================

extern "C" int nvfp4_ws5_c32_version() {
    return 16;
}

extern "C" int nvfp4_ws5_c48_v16_w2_min_cta() {
    return NVFP4_W2_T16_MIN_CTA * 10 + NVFP4_W2_T32_MIN_CTA;
}

extern "C" int nvfp4_ws5_c32_w13_min_cta() {
    return NVFP4_W13_T16_MIN_CTA * 10 + NVFP4_W13_T32B_MIN_CTA;
}

extern "C" int nvfp4_ws5_c48_min_cta() {
    return NVFP4_W13_T8_MIN_CTA * 10 + NVFP4_W2_T8_MIN_CTA;
}

extern "C" int nvfp4_ws5_c48_v9_bn16_min_cta() {
    return NVFP4_W13_T8_BN16_MIN_CTA;
}

extern "C" int nvfp4_w13_ws5_c48_v16_fwd(
    const void* A, const void* AS, const void* RS, const void* Row2Tok,
    const void* SLABS, const void* WS_BASE,
    const void* P_OFF, const void* PATCHES,
    const void* Ones, const void* Tiles,
    const void* TileCompactStart, const void* Gammas, void* RowAmaxW2,
    void* Out, void* TileQdeqDone,
    int N, int K, int MAX_CTAS, int TILE,
    int bk_dot, int zero_scatter, int as_row_major, int fuse_situ,
    int skip_amax, int bn_mode, int bn16_margin_pct,
    int pdl_wait_mode, int trig_dep,
    int fuse_qdeq, int quant_fma,
    int evict_first, int lpt_mode, void* WorkQueue, uintptr_t stream_ptr
) {
    if (N != 576 || K != 4608) return -2;
    if (MAX_CTAS <= 0) return 0;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    const bool zs = (zero_scatter != 0) && (Row2Tok != nullptr);
    const bool rm = zs || (as_row_major != 0);
    const bool fs = (fuse_situ != 0);
    const bool fq = fs && (fuse_qdeq != 0) && (TileQdeqDone != nullptr) && (RowAmaxW2 != nullptr);
    const int qm = fq ? ((quant_fma != 0) ? 3 : 2) : ((fs && skip_amax != 0) ? 1 : 0);
    const int eff_margin = (bn16_margin_pct >= 0) ? bn16_margin_pct : 3;

    if (TILE == 8) {
        const int cap32 = 48 * NVFP4_W13_T8_MIN_CTA;
        const int cap16 = 48 * NVFP4_W13_T8_BN16_MIN_CTA;
        const int tot32 = 18 * MAX_CTAS;
        const int tot16 = 36 * MAX_CTAS;
        const int gx32  = (tot32 < cap32) ? tot32 : cap32;
        const int gx16  = (tot16 < cap16) ? tot16 : cap16;

        auto launch_t8 = [&](int bn_val, bool adaptive_gate, int pw, int td) -> int {
            uint32_t* wq_slot = (WorkQueue != nullptr)
                ? (static_cast<uint32_t*>(WorkQueue) + ((bn_val == 16) ? 0 : 16))
                : nullptr;
            dim3 grid((bn_val == 16) ? gx16 : gx32, 1, 1);
            dim3 block((bn_val == 16) ? 64 : 128, 1, 1);
            #define C8(BNV, MC, Z, R, F, QMV, A_G) return dispatch_w13_t8_v16<BNV, MC, Z, R, F, QMV, A_G>( \
                grid, block, stream, pw, td, \
                static_cast<const uint8_t*>(A), static_cast<const uint8_t*>(AS), \
                static_cast<const float*>(RS), static_cast<const int32_t*>(Row2Tok), \
                static_cast<const uint8_t*>(SLABS), static_cast<const uint8_t*>(WS_BASE), \
                static_cast<const int32_t*>(P_OFF), static_cast<const uint32_t*>(PATCHES), \
                static_cast<const float*>(Ones), static_cast<const int32_t*>(Tiles), \
                static_cast<const int32_t*>(TileCompactStart), static_cast<const float*>(Gammas), \
                static_cast<uint32_t*>(RowAmaxW2), static_cast<__nv_bfloat16*>(Out), \
                static_cast<uint32_t*>(TileQdeqDone), wq_slot, \
                N, K, MAX_CTAS, gx16, gx32, eff_margin, evict_first, lpt_mode)
            #define C8_QM(BNV, MC, A_G) do { \
                if (qm == 3) { \
                    if      (zs) { C8(BNV, MC, true,  true,  true, 3, A_G); } \
                    else if (rm) { C8(BNV, MC, false, true,  true, 3, A_G); } \
                    else         { C8(BNV, MC, false, false, true, 3, A_G); } \
                } else if (qm == 2) { \
                    if      (zs) { C8(BNV, MC, true,  true,  true, 2, A_G); } \
                    else if (rm) { C8(BNV, MC, false, true,  true, 2, A_G); } \
                    else         { C8(BNV, MC, false, false, true, 2, A_G); } \
                } else if (qm == 1) { \
                    if      (zs) { C8(BNV, MC, true,  true,  true, 1, A_G); } \
                    else if (rm) { C8(BNV, MC, false, true,  true, 1, A_G); } \
                    else         { C8(BNV, MC, false, false, true, 1, A_G); } \
                } else { \
                    if      (zs && fs)         { C8(BNV, MC, true,  true,  true,  0, A_G); } \
                    else if (zs && !fs)        { C8(BNV, MC, true,  true,  false, 0, A_G); } \
                    else if (!zs && rm && fs)  { C8(BNV, MC, false, true,  true,  0, A_G); } \
                    else if (!zs && rm && !fs) { C8(BNV, MC, false, true,  false, 0, A_G); } \
                    else if (!zs && !rm && fs) { C8(BNV, MC, false, false, true,  0, A_G); } \
                    else                       { C8(BNV, MC, false, false, false, 0, A_G); } \
                } \
            } while (0)
            if (bn_val == 16) {
                if (adaptive_gate) { C8_QM(16, NVFP4_W13_T8_BN16_MIN_CTA, true);  }
                else               { C8_QM(16, NVFP4_W13_T8_BN16_MIN_CTA, false); }
            } else {
                if (adaptive_gate) { C8_QM(32, NVFP4_W13_T8_MIN_CTA, true);  }
                else               { C8_QM(32, NVFP4_W13_T8_MIN_CTA, false); }
            }
            #undef C8_QM
            #undef C8
            return -1;
        };

        if (bn_mode == 1) {
            return launch_t8(16, false, pdl_wait_mode, trig_dep);
        }
        if (bn_mode == 2) {
            const int chain_pdl = (pdl_wait_mode != 0 || trig_dep != 0) ? 1 : 0;
            const int pw2 = (pdl_wait_mode == 2) ? 2 : chain_pdl;
            int rc = launch_t8(16, true, pdl_wait_mode, chain_pdl);
            if (rc != 0) return rc;
            return launch_t8(32, true, pw2, trig_dep);
        }
        return launch_t8(32, false, pdl_wait_mode, trig_dep);
    }

    const int total_tiles = 18 * MAX_CTAS;
    int persist_cap = 192;
    if (TILE == 16) persist_cap = 48 * NVFP4_W13_T16_MIN_CTA;
    else if (TILE == 32 && bk_dot == 256) persist_cap = 48 * NVFP4_W13_T32B_MIN_CTA;
    dim3 grid((total_tiles < persist_cap) ? total_tiles : persist_cap, 1, 1);
    dim3 block(128, 1, 1);

    uint32_t* wq_w13 = (WorkQueue != nullptr) ? static_cast<uint32_t*>(WorkQueue) : nullptr;
    #define C16(Z, R, F, QMV) return dispatch_w13_t16_v16<16, NVFP4_W13_T16_MIN_CTA, Z, R, F, QMV>( \
        grid, block, stream, pdl_wait_mode, trig_dep, \
        static_cast<const uint8_t*>(A), static_cast<const uint8_t*>(AS), \
        static_cast<const float*>(RS), static_cast<const int32_t*>(Row2Tok), \
        static_cast<const uint8_t*>(SLABS), static_cast<const uint8_t*>(WS_BASE), \
        static_cast<const int32_t*>(P_OFF), static_cast<const uint32_t*>(PATCHES), \
        static_cast<const float*>(Ones), static_cast<const int32_t*>(Tiles), \
        static_cast<const int32_t*>(TileCompactStart), static_cast<const float*>(Gammas), \
        static_cast<uint32_t*>(RowAmaxW2), static_cast<__nv_bfloat16*>(Out), \
        static_cast<uint32_t*>(TileQdeqDone), wq_w13, N, K, MAX_CTAS, evict_first, lpt_mode)
    #define C32B(Z, R, F, QMV) return dispatch_w13_t32_bk256_v16<NVFP4_W13_T32B_MIN_CTA, Z, R, F, QMV>( \
        grid, block, stream, pdl_wait_mode, trig_dep, \
        static_cast<const uint8_t*>(A), static_cast<const uint8_t*>(AS), \
        static_cast<const float*>(RS), static_cast<const int32_t*>(Row2Tok), \
        static_cast<const uint8_t*>(SLABS), static_cast<const uint8_t*>(WS_BASE), \
        static_cast<const int32_t*>(P_OFF), static_cast<const uint32_t*>(PATCHES), \
        static_cast<const float*>(Ones), static_cast<const int32_t*>(Tiles), \
        static_cast<const int32_t*>(TileCompactStart), static_cast<const float*>(Gammas), \
        static_cast<uint32_t*>(RowAmaxW2), static_cast<__nv_bfloat16*>(Out), \
        static_cast<uint32_t*>(TileQdeqDone), wq_w13, N, K, MAX_CTAS, evict_first, lpt_mode)
    #define C32(Z, R, F, QMV) return dispatch_w13_t32_v16<Z, R, F, QMV>( \
        grid, block, stream, pdl_wait_mode, trig_dep, \
        static_cast<const uint8_t*>(A), static_cast<const uint8_t*>(AS), \
        static_cast<const float*>(RS), static_cast<const int32_t*>(Row2Tok), \
        static_cast<const uint8_t*>(SLABS), static_cast<const uint8_t*>(WS_BASE), \
        static_cast<const int32_t*>(P_OFF), static_cast<const uint32_t*>(PATCHES), \
        static_cast<const float*>(Ones), static_cast<const int32_t*>(Tiles), \
        static_cast<const int32_t*>(TileCompactStart), static_cast<const float*>(Gammas), \
        static_cast<uint32_t*>(RowAmaxW2), static_cast<__nv_bfloat16*>(Out), \
        static_cast<uint32_t*>(TileQdeqDone), wq_w13, N, K, MAX_CTAS, evict_first, lpt_mode)

    #define DISPATCH_13(MAC) do { \
        if (qm == 3) { \
            if      (zs) { MAC(true,  true,  true, 3); } \
            else if (rm) { MAC(false, true,  true, 3); } \
            else         { MAC(false, false, true, 3); } \
        } else if (qm == 2) { \
            if      (zs) { MAC(true,  true,  true, 2); } \
            else if (rm) { MAC(false, true,  true, 2); } \
            else         { MAC(false, false, true, 2); } \
        } else if (qm == 1) { \
            if      (zs) { MAC(true,  true,  true, 1); } \
            else if (rm) { MAC(false, true,  true, 1); } \
            else         { MAC(false, false, true, 1); } \
        } else { \
            if      (zs && fs)         { MAC(true,  true,  true,  0); } \
            else if (zs && !fs)        { MAC(true,  true,  false, 0); } \
            else if (!zs && rm && fs)  { MAC(false, true,  true,  0); } \
            else if (!zs && rm && !fs) { MAC(false, true,  false, 0); } \
            else if (!zs && !rm && fs) { MAC(false, false, true,  0); } \
            else                       { MAC(false, false, false, 0); } \
        } \
    } while (0)

    if (TILE == 16) {
        DISPATCH_13(C16);
    } else if (TILE == 32) {
        if (bk_dot == 256) { DISPATCH_13(C32B); }
        else               { DISPATCH_13(C32);  }
    }
    #undef DISPATCH_13
    #undef C32
    #undef C32B
    #undef C16
    return -1;
}

extern "C" int nvfp4_w13_ws5_c48_v10_fwd(
    const void* A, const void* AS, const void* RS, const void* Row2Tok,
    const void* SLABS, const void* WS_BASE,
    const void* P_OFF, const void* PATCHES,
    const void* Ones, const void* Tiles,
    const void* TileCompactStart, const void* Gammas, void* RowAmaxW2,
    void* Out, void* TileQdeqDone,
    int N, int K, int MAX_CTAS, int TILE,
    int bk_dot, int zero_scatter, int as_row_major, int fuse_situ,
    int skip_amax, int bn_mode, int bn16_margin_pct,
    int pdl_wait_mode, int trig_dep,
    int fuse_qdeq, int quant_fma, uintptr_t stream_ptr
) {
    return nvfp4_w13_ws5_c48_v16_fwd(
        A, AS, RS, Row2Tok, SLABS, WS_BASE, P_OFF, PATCHES, Ones, Tiles,
        TileCompactStart, Gammas, RowAmaxW2, Out, TileQdeqDone,
        N, K, MAX_CTAS, TILE, bk_dot, zero_scatter, as_row_major, fuse_situ,
        skip_amax, bn_mode, bn16_margin_pct, pdl_wait_mode, trig_dep,
        fuse_qdeq, quant_fma, 1, 0, nullptr, stream_ptr
    );
}

extern "C" int nvfp4_w13_ws5_c48_v9_fwd(
    const void* A, const void* AS, const void* RS, const void* Row2Tok,
    const void* SLABS, const void* WS_BASE,
    const void* P_OFF, const void* PATCHES,
    const void* Ones, const void* Tiles,
    const void* TileCompactStart, const void* Gammas, void* RowAmaxW2,
    void* Out,
    int N, int K, int MAX_CTAS, int TILE,
    int bk_dot, int zero_scatter, int as_row_major, int fuse_situ,
    int skip_amax, int bn_mode, int bn16_margin_pct,
    int pdl_wait_mode, int trig_dep, uintptr_t stream_ptr
) {
    return nvfp4_w13_ws5_c48_v10_fwd(
        A, AS, RS, Row2Tok, SLABS, WS_BASE, P_OFF, PATCHES, Ones, Tiles,
        TileCompactStart, Gammas, RowAmaxW2, Out, nullptr,
        N, K, MAX_CTAS, TILE, bk_dot, zero_scatter, as_row_major, fuse_situ,
        skip_amax, bn_mode, bn16_margin_pct, pdl_wait_mode, trig_dep, 0, 0, stream_ptr
    );
}

extern "C" int nvfp4_situ_quant_dequant_c48_v9_fwd(
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
    if (TILE != 8 && TILE != 16 && TILE != 32) return -2;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    const int need_ctas = (max_compact_rows + 3) / 4;
    const int grid_x = (need_ctas < 96) ? need_ctas : 96;  // up to 2 CTAs/SM = 384 warps in 1 wave!
    dim3 grid(grid_x, 1, 1);
    dim3 block(128, 1, 1);
    const bool td = (trig_dep != 0);
    const bool pw = (pdl_wait != 0) || td;
    const bool qf = (quant_fma != 0);
    const bool use_pdl = pw || td;
    cudaError_t err = cudaSuccess;

    #define RUN_Q(PW, TD, QF) err = launch_nvfp4( \
        nvfp4_situ_quant_dequant_c48_v9_kernel<PW, TD, QF>, \
        grid, block, stream, use_pdl, \
        static_cast<const __nv_bfloat16*>(SiTU_In), \
        static_cast<uint16_t*>(QDeqOut), \
        static_cast<uint32_t*>(RowAmaxW2), \
        static_cast<const int32_t*>(Tiles), \
        static_cast<const int32_t*>(TileCompactStart), \
        MAX_CTAS, TILE, max_compact_rows)
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

extern "C" int nvfp4_w2_ws5_c48_v16_fwd(
    const void* A, const void* AS, const void* RS,
    const void* RowAmaxW2, const void* WAmax, const void* WDenominator,
    const void* W_PACK, const void* WS5_PACK, const void* WS_BASE,
    const void* P_OFF, const void* PATCHES,
    const void* Ones, const void* Tiles, const void* TileCompactStart,
    void* Out,
    const void* Src2Compact, const void* NLivePtr,
    void* PanelDone, void* BufGen, void* SendBuf,
    int N, int K, int MAX_CTAS, int TILE, int T,
    int bk_dot, int as_row_major, int as_row_stride, int down_row_stride,
    int fuse_quant_mode, int fuse_combine,
    int pdl_wait_mode, int trig_dep, int quant_fma,
    int evict_first, int lpt_mode, void* WorkQueue, uintptr_t stream_ptr
) {
    if (N != 4608 || K != 288) return -2;
    if (fuse_quant_mode < 0 || fuse_quant_mode > 2) return -3;
    if (fuse_combine < 0 || fuse_combine > 4) return -4;
    if (fuse_combine >= 1 && fuse_combine <= 3 && T <= 0) return -5;
    if (fuse_combine == 4 && TileCompactStart == nullptr) return -5;
    const int total_tiles = (N / 128) * MAX_CTAS;
    if (total_tiles <= 0 && (fuse_combine == 0 || fuse_combine == 4)) return 0;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    const bool rm = (fuse_quant_mode == 0) && (as_row_major != 0);
    const int eff_bk = (bk_dot == 64 || (bk_dot == 0 && TILE >= 32)) ? 64 : 32;
    const int eff_as_stride = (as_row_stride > 0) ? as_row_stride : 18;
    const int eff_down_stride = (down_row_stride > 0) ? down_row_stride : (T * 8);
    const int eff_pdl_wait = (pdl_wait_mode >= 2) ? 2 : pdl_wait_mode;

    // V13 Occupancy Protection: decouple cold legacy OTF quant (`FQ=1`) from hot `FQ=2`/`FQ=0`
    // so hot QDeqA (`FQ=2`) ALWAYS runs at `NVFP4_W2_T8_MIN_CTA=6` (288 CTAs, 80 regs/thr, 0 spills)!
    constexpr int T8_CPP = 8 * NVFP4_W2_T8_MIN_CTA;
    constexpr int T8_FQ1_MC  = (NVFP4_W2_T8_MIN_CTA > 4) ? 4 : NVFP4_W2_T8_MIN_CTA;
    constexpr int T8_FQ1_CPP = 8 * T8_FQ1_MC;
    constexpr int T8_FC3_MC  = (NVFP4_W2_T8_MIN_CTA > 3) ? 3 : NVFP4_W2_T8_MIN_CTA;
    constexpr int T16_HOT_MC = NVFP4_W2_T16_MIN_CTA;
    constexpr int T16_FQ1_MC = (T16_HOT_MC > 4) ? 4 : T16_HOT_MC;
    constexpr int T32_HOT_MC = NVFP4_W2_T32_MIN_CTA;
    constexpr int T32_FQ1_MC = (T32_HOT_MC > 3) ? 3 : T32_HOT_MC;
    const int eff_t8_mc  = (fuse_quant_mode == 1) ? T8_FQ1_MC  : NVFP4_W2_T8_MIN_CTA;
    const int eff_t16_mc = (fuse_quant_mode == 1) ? T16_FQ1_MC : T16_HOT_MC;
    const int eff_t32_mc = (fuse_quant_mode == 1) ? T32_FQ1_MC : T32_HOT_MC;
    const int persist_cap = (TILE == 8) ? (48 * eff_t8_mc) : ((TILE == 16) ? (48 * eff_t16_mc) : (48 * eff_t32_mc));
    const int fc3_cap     = (TILE == 8) ? (48 * T8_FC3_MC) : 144;
    int grid_x;
    if (fuse_combine == 1) {
        grid_x = persist_cap;
    } else if (fuse_combine == 3) {
        grid_x = fc3_cap;
    } else if (fuse_combine == 2) {
        grid_x = (total_tiles < persist_cap) ? total_tiles : persist_cap;
        if (grid_x < 36) grid_x = 36;
    } else {
        grid_x = (total_tiles < persist_cap) ? total_tiles : persist_cap;
    }
    dim3 grid(grid_x, 1, 1);
    dim3 block(128, 1, 1);

    const bool qf = (quant_fma != 0);
    // 64B cache-line separated WorkQueue slot +32 for W2 (never overlaps W13 +0/+16 during PDL!)
    uint32_t* wq_w2 = (WorkQueue != nullptr) ? (static_cast<uint32_t*>(WorkQueue) + 32) : nullptr;
    #define CALL2(TV, BK, MC, CPP, R, QM, C, QF) return dispatch_w2_v16<TV, BK, MC, CPP, R, QM, C, QF>( \
        grid, block, stream, eff_pdl_wait, trig_dep, \
        static_cast<const uint8_t*>(A), static_cast<const uint8_t*>(AS), \
        static_cast<const float*>(RS), static_cast<const uint32_t*>(RowAmaxW2), \
        static_cast<const float*>(WAmax), static_cast<const float*>(WDenominator), \
        static_cast<const uint8_t*>(W_PACK), static_cast<const uint32_t*>(WS5_PACK), \
        static_cast<const uint8_t*>(WS_BASE), static_cast<const int32_t*>(P_OFF), \
        static_cast<const uint32_t*>(PATCHES), static_cast<const float*>(Ones), \
        static_cast<const int32_t*>(Tiles), static_cast<const int32_t*>(TileCompactStart), \
        static_cast<__nv_bfloat16*>(Out), static_cast<const int32_t*>(Src2Compact), \
        static_cast<const int32_t*>(NLivePtr), static_cast<uint32_t*>(PanelDone), \
        static_cast<uint32_t*>(BufGen), static_cast<__nv_bfloat16*>(SendBuf), wq_w2, \
        N, K, MAX_CTAS, T, eff_as_stride, eff_down_stride, evict_first, lpt_mode)

    #define DISPATCH_FC(TV, BK, MC_HOT, CPP_HOT, MC_FQ1, CPP_FQ1, FC_VAL) do { \
        if      (fuse_quant_mode == 2)         { CALL2(TV, BK, MC_HOT, CPP_HOT, false, 2, FC_VAL, false); } \
        else if (fuse_quant_mode == 1 && qf)   { CALL2(TV, BK, MC_FQ1, CPP_FQ1, false, 1, FC_VAL, true);  } \
        else if (fuse_quant_mode == 1 && !qf)  { CALL2(TV, BK, MC_FQ1, CPP_FQ1, false, 1, FC_VAL, false); } \
        else if (rm)                           { CALL2(TV, BK, MC_HOT, CPP_HOT, true,  0, FC_VAL, false); } \
        else                                   { CALL2(TV, BK, MC_HOT, CPP_HOT, false, 0, FC_VAL, false); } \
    } while (0)

    if (TILE == 8) {
        if      (fuse_combine == 0) { DISPATCH_FC(8, 32, NVFP4_W2_T8_MIN_CTA, T8_CPP, T8_FQ1_MC, T8_FQ1_CPP, 0); }
        else if (fuse_combine == 1) { DISPATCH_FC(8, 32, NVFP4_W2_T8_MIN_CTA, T8_CPP, T8_FQ1_MC, T8_FQ1_CPP, 1); }
        else if (fuse_combine == 2) { DISPATCH_FC(8, 32, NVFP4_W2_T8_MIN_CTA, T8_CPP, T8_FQ1_MC, T8_FQ1_CPP, 2); }
        else if (fuse_combine == 3) { DISPATCH_FC(8, 32, T8_FC3_MC,           T8_CPP, T8_FC3_MC, T8_CPP,     3); }
        else                        { DISPATCH_FC(8, 32, NVFP4_W2_T8_MIN_CTA, T8_CPP, T8_FQ1_MC, T8_FQ1_CPP, 4); }
    } else if (TILE == 16) {
        if      (fuse_combine == 0) { DISPATCH_FC(16, 32, T16_HOT_MC, 8 * T16_HOT_MC, T16_FQ1_MC, 8 * T16_FQ1_MC, 0); }
        else if (fuse_combine == 1) { DISPATCH_FC(16, 32, T16_HOT_MC, 8 * T16_HOT_MC, T16_FQ1_MC, 8 * T16_FQ1_MC, 1); }
        else if (fuse_combine == 2) { DISPATCH_FC(16, 32, T16_HOT_MC, 8 * T16_HOT_MC, T16_FQ1_MC, 8 * T16_FQ1_MC, 2); }
        else if (fuse_combine == 3) { DISPATCH_FC(16, 32, 3,          32,             3,          32,             3); }
        else                        { DISPATCH_FC(16, 32, T16_HOT_MC, 8 * T16_HOT_MC, T16_FQ1_MC, 8 * T16_FQ1_MC, 4); }
    } else if (TILE == 32) {
        if (eff_bk == 64) {
            if      (fuse_combine == 0) { DISPATCH_FC(32, 64, T32_HOT_MC, 8 * T32_HOT_MC, T32_FQ1_MC, 8 * T32_FQ1_MC, 0); }
            else if (fuse_combine == 1) { DISPATCH_FC(32, 64, T32_HOT_MC, 8 * T32_HOT_MC, T32_FQ1_MC, 8 * T32_FQ1_MC, 1); }
            else if (fuse_combine == 2) { DISPATCH_FC(32, 64, T32_HOT_MC, 8 * T32_HOT_MC, T32_FQ1_MC, 8 * T32_FQ1_MC, 2); }
            else if (fuse_combine == 3) { DISPATCH_FC(32, 64, 3,          24,             3,          24,             3); }
            else                        { DISPATCH_FC(32, 64, T32_HOT_MC, 8 * T32_HOT_MC, T32_FQ1_MC, 8 * T32_FQ1_MC, 4); }
        } else {
            if      (fuse_combine == 0) { DISPATCH_FC(32, 32, T32_HOT_MC, 8 * T32_HOT_MC, T32_FQ1_MC, 8 * T32_FQ1_MC, 0); }
            else if (fuse_combine == 1) { DISPATCH_FC(32, 32, T32_HOT_MC, 8 * T32_HOT_MC, T32_FQ1_MC, 8 * T32_FQ1_MC, 1); }
            else if (fuse_combine == 2) { DISPATCH_FC(32, 32, T32_HOT_MC, 8 * T32_HOT_MC, T32_FQ1_MC, 8 * T32_FQ1_MC, 2); }
            else if (fuse_combine == 3) { DISPATCH_FC(32, 32, 3,          24,             3,          24,             3); }
            else                        { DISPATCH_FC(32, 32, T32_HOT_MC, 8 * T32_HOT_MC, T32_FQ1_MC, 8 * T32_FQ1_MC, 4); }
        }
    }
    #undef DISPATCH_FC
    #undef CALL2
    return -1;
}

extern "C" int nvfp4_w2_ws5_c48_v9_fwd(
    const void* A, const void* AS, const void* RS,
    const void* RowAmaxW2, const void* WAmax, const void* WDenominator,
    const void* W_PACK, const void* WS5_PACK, const void* WS_BASE,
    const void* P_OFF, const void* PATCHES,
    const void* Ones, const void* Tiles, const void* TileCompactStart,
    void* Out,
    const void* Src2Compact, const void* NLivePtr,
    void* PanelDone, void* BufGen, void* SendBuf,
    int N, int K, int MAX_CTAS, int TILE, int T,
    int bk_dot, int as_row_major, int as_row_stride, int down_row_stride,
    int fuse_quant_mode, int fuse_combine,
    int pdl_wait_mode, int trig_dep, int quant_fma, uintptr_t stream_ptr
) {
    return nvfp4_w2_ws5_c48_v16_fwd(
        A, AS, RS, RowAmaxW2, WAmax, WDenominator,
        W_PACK, WS5_PACK, WS_BASE, P_OFF, PATCHES, Ones, Tiles, TileCompactStart,
        Out, Src2Compact, NLivePtr, PanelDone, BufGen, SendBuf,
        N, K, MAX_CTAS, TILE, T, bk_dot, as_row_major, as_row_stride, down_row_stride,
        fuse_quant_mode, fuse_combine, pdl_wait_mode, trig_dep, quant_fma,
        1, 0, nullptr, stream_ptr
    );
}

extern "C" int nvfp4_combine_c32_v9_fwd(
    const void* Down,
    const void* Src2Row,
    const void* NLivePtr,
    void* Out,
    int T, int H,
    int pdl_wait, int trig_dep, uintptr_t stream_ptr
) {
    if (T <= 0) return 0;
    if (H != 4608) return -2;
    if (T > 65535) return -3;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    const bool fine = (T <= 64);
    dim3 grid(fine ? 6u : 3u, static_cast<unsigned int>(T), 1);
    dim3 block(fine ? 96u : 192u, 1, 1);
    const bool td = (trig_dep != 0);
    const bool pw = (pdl_wait != 0) || td;
    const bool nl = (NLivePtr != nullptr);
    const bool use_pdl = pw || td;
    cudaError_t err = cudaSuccess;

    #define RUN_COMB(BT, CPC, MC, PW, TD, NL) err = launch_nvfp4( \
        nvfp4_combine_c32_v9_kernel<BT, CPC, MC, PW, TD, NL>, \
        grid, block, stream, use_pdl, \
        static_cast<const __nv_bfloat16*>(Down), \
        static_cast<const int32_t*>(Src2Row), \
        static_cast<const int32_t*>(NLivePtr), \
        static_cast<__nv_bfloat16*>(Out))

    if (fine) {
        if (nl) {
            if      (pw && td)  { RUN_COMB(96, 768, 8, true,  true,  true); }
            else if (pw && !td) { RUN_COMB(96, 768, 8, true,  false, true); }
            else if (!pw && td) { RUN_COMB(96, 768, 8, false, true,  true); }
            else                { RUN_COMB(96, 768, 8, false, false, true); }
        } else {
            if      (pw && td)  { RUN_COMB(96, 768, 8, true,  true,  false); }
            else if (pw && !td) { RUN_COMB(96, 768, 8, true,  false, false); }
            else if (!pw && td) { RUN_COMB(96, 768, 8, false, true,  false); }
            else                { RUN_COMB(96, 768, 8, false, false, false); }
        }
    } else {
        if (nl) {
            if      (pw && td)  { RUN_COMB(192, 1536, 6, true,  true,  true); }
            else if (pw && !td) { RUN_COMB(192, 1536, 6, true,  false, true); }
            else if (!pw && td) { RUN_COMB(192, 1536, 6, false, true,  true); }
            else                { RUN_COMB(192, 1536, 6, false, false, true); }
        } else {
            if      (pw && td)  { RUN_COMB(192, 1536, 6, true,  true,  false); }
            else if (pw && !td) { RUN_COMB(192, 1536, 6, true,  false, false); }
            else if (!pw && td) { RUN_COMB(192, 1536, 6, false, true,  false); }
            else                { RUN_COMB(192, 1536, 6, false, false, false); }
        }
    }
    #undef RUN_COMB
    return static_cast<int>(err);
}

// ---- 100% Drop-In Backward-Compatible Wrappers (v8, c32/v6/v7, S7) ----

extern "C" int nvfp4_w13_ws5_c48_v8_fwd(
    const void* A, const void* AS, const void* RS, const void* Row2Tok,
    const void* SLABS, const void* WS_BASE,
    const void* P_OFF, const void* PATCHES,
    const void* Ones, const void* Tiles,
    const void* TileCompactStart, const void* Gammas, void* RowAmaxW2,
    void* Out,
    int N, int K, int MAX_CTAS, int TILE,
    int bk_dot, int zero_scatter, int as_row_major, int fuse_situ,
    int pdl_wait_mode, int trig_dep,
    void* TileW13Done, void* TileReady, void* OverlapMeta,
    int overlap_w13_w2, int hot_l2_hint, uintptr_t stream_ptr
) {
    // Note: V8 overlap spin (+2..7 ms slower) & hot_l2_hint (+131 us/L slower) intentionally retired.
    (void)TileW13Done; (void)TileReady; (void)OverlapMeta;
    (void)overlap_w13_w2; (void)hot_l2_hint;
    return nvfp4_w13_ws5_c48_v9_fwd(
        A, AS, RS, Row2Tok, SLABS, WS_BASE, P_OFF, PATCHES, Ones, Tiles,
        TileCompactStart, Gammas, RowAmaxW2, Out,
        N, K, MAX_CTAS, TILE, bk_dot, zero_scatter, as_row_major, fuse_situ,
        0, 0, 3, pdl_wait_mode, trig_dep, stream_ptr
    );
}

extern "C" int nvfp4_w13_ws5_c32_fwd(
    const void* A, const void* AS, const void* RS, const void* Row2Tok,
    const void* SLABS, const void* WS_BASE,
    const void* P_OFF, const void* PATCHES,
    const void* Ones, const void* Tiles,
    const void* TileCompactStart, const void* Gammas, void* RowAmaxW2,
    void* Out,
    int N, int K, int MAX_CTAS, int TILE,
    int bk_dot, int zero_scatter, int as_row_major, int fuse_situ,
    int pdl_wait_mode, int trig_dep, uintptr_t stream_ptr
) {
    return nvfp4_w13_ws5_c48_v9_fwd(
        A, AS, RS, Row2Tok, SLABS, WS_BASE, P_OFF, PATCHES, Ones, Tiles,
        TileCompactStart, Gammas, RowAmaxW2, Out,
        N, K, MAX_CTAS, TILE, bk_dot, zero_scatter, as_row_major, fuse_situ,
        0, 0, 3, pdl_wait_mode, trig_dep, stream_ptr
    );
}

extern "C" int nvfp4_w2_ws5_c48_v8_fwd(
    const void* A, const void* AS, const void* RS,
    const void* RowAmaxW2, const void* WAmax, const void* WDenominator,
    const void* W_PACK, const void* WS5_PACK, const void* WS_BASE,
    const void* P_OFF, const void* PATCHES,
    const void* Ones, const void* Tiles, const void* TileCompactStart,
    void* Out,
    const void* Src2Compact, const void* NLivePtr,
    void* PanelDone, void* BufGen, void* SendBuf,
    int N, int K, int MAX_CTAS, int TILE, int T,
    int bk_dot, int as_row_major, int as_row_stride, int down_row_stride,
    int fuse_quant, int fuse_combine,
    int pdl_wait_mode, int trig_dep,
    void* TileReady, void* OverlapMeta,
    int overlap_w13_w2, int hot_l2_hint, uintptr_t stream_ptr
) {
    (void)TileReady; (void)OverlapMeta; (void)overlap_w13_w2; (void)hot_l2_hint;
    return nvfp4_w2_ws5_c48_v9_fwd(
        A, AS, RS, RowAmaxW2, WAmax, WDenominator,
        W_PACK, WS5_PACK, WS_BASE, P_OFF, PATCHES,
        Ones, Tiles, TileCompactStart, Out,
        Src2Compact, NLivePtr, PanelDone, BufGen, SendBuf,
        N, K, MAX_CTAS, TILE, T, bk_dot, as_row_major, as_row_stride, down_row_stride,
        fuse_quant, fuse_combine, pdl_wait_mode, trig_dep, 0, stream_ptr
    );
}

extern "C" int nvfp4_w2_ws5_c32_fwd(
    const void* A, const void* AS, const void* RS,
    const void* RowAmaxW2, const void* WAmax, const void* WDenominator,
    const void* W_PACK, const void* WS5_PACK, const void* WS_BASE,
    const void* P_OFF, const void* PATCHES,
    const void* Ones, const void* Tiles, const void* TileCompactStart,
    void* Out,
    const void* Src2Compact, const void* NLivePtr,
    void* PanelDone, void* BufGen, void* SendBuf,
    int N, int K, int MAX_CTAS, int TILE, int T,
    int bk_dot, int as_row_major, int as_row_stride, int down_row_stride,
    int fuse_quant, int fuse_combine,
    int pdl_wait_mode, int trig_dep, uintptr_t stream_ptr
) {
    return nvfp4_w2_ws5_c48_v9_fwd(
        A, AS, RS, RowAmaxW2, WAmax, WDenominator,
        W_PACK, WS5_PACK, WS_BASE, P_OFF, PATCHES,
        Ones, Tiles, TileCompactStart, Out,
        Src2Compact, NLivePtr, PanelDone, BufGen, SendBuf,
        N, K, MAX_CTAS, TILE, T, bk_dot, as_row_major, as_row_stride, down_row_stride,
        fuse_quant, fuse_combine, pdl_wait_mode, trig_dep, 0, stream_ptr
    );
}

extern "C" int nvfp4_combine_c32_v6_fwd(
    const void* Down,
    const void* Src2Row,
    const void* NLivePtr,
    void* Out,
    int T, int H,
    int pdl_wait, int trig_dep, uintptr_t stream_ptr
) {
    return nvfp4_combine_c32_v9_fwd(Down, Src2Row, NLivePtr, Out, T, H, pdl_wait, trig_dep, stream_ptr);
}

extern "C" int nvfp4_w13_ws5_fwd(
    const void* A, const void* AS, const void* RS,
    const void* SLABS, const void* WS_BASE,
    const void* P_OFF, const void* PATCHES,
    const void* Ones, const void* Tiles, void* Out,
    int N, int K, int MAX_CTAS, int TILE,
    int pdl_wait_mode, int trig_dep, uintptr_t stream_ptr
) {
    return nvfp4_w13_ws5_c48_v9_fwd(
        A, AS, RS, nullptr, SLABS, WS_BASE, P_OFF, PATCHES, Ones, Tiles,
        nullptr, nullptr, nullptr, Out,
        N, K, MAX_CTAS, TILE, (TILE <= 16) ? 256 : 128, 0, 0, 0,
        0, 0, 3, pdl_wait_mode, trig_dep, stream_ptr
    );
}

extern "C" int nvfp4_w2_ws5_fwd(
    const void* A, const void* AS, const void* RS,
    const void* W_PACK, const void* WS5_PACK, const void* WS_BASE,
    const void* P_OFF, const void* PATCHES,
    const void* Ones, const void* Tiles, void* Out,
    int N, int K, int MAX_CTAS, int TILE,
    int pdl_wait_mode, int trig_dep, uintptr_t stream_ptr
) {
    return nvfp4_w2_ws5_c48_v9_fwd(
        A, AS, RS, nullptr, nullptr, nullptr,
        W_PACK, WS5_PACK, WS_BASE, P_OFF, PATCHES, Ones, Tiles, nullptr, Out,
        nullptr, nullptr, nullptr, nullptr, nullptr,
        N, K, MAX_CTAS, TILE, 0, 32, 0, 18, 0, 0, 0, pdl_wait_mode, trig_dep, 0, stream_ptr
    );
}
