// SPDX-License-Identifier: Apache-2.0
// SM121a (GB10) C=2..8 / C=32 v8 (M=17..256, TILE in {8, 16, 32}) 0-ULP Bit-Exact, 0-Spill MX7 MoE Kernels
// Targets Inkling-2-Small TP8 L40..47 MXFP8 MoE (E=384, H=4608, inter=288, topk=8):
//   - MX_W13 (N=576,  K=4608): BN=32  (4 warps, 18 panels), 36 slabs/warp
//   - MX_W2  (N=4608, K=288):  BN=128 (4 warps, 36 panels, SUB_N=4), 9 slabs/warp
//
// v8 Wave-V8 Upgrades atop v7 C48 TILE=8 (100% literal nd == 0, 0 B stack / 0 B spills):
//   1. Deadlock-Free W13 -> W2 Tail-Wave PDL + Per-Tile `TileReady[512]` Producer-Consumer Overlap:
//        - W13 writes SiTU_Out via `st_cg_u2` (L1-bypass to L2), runs `__syncthreads()`, then `tid==0`
//          executes `atom.acq_rel.gpu` on `TileW13Done[t]`; the 18th panel resets `TileW13Done[t]=0`,
//          fences, and stamps `st.release.gpu(&TileReady[t], overlap_gen)`.
//        - Tail-wave PDL fires `launch_dependents` at tail-wave entry (`sl == 0`) so W2 immediately
//          fills freed SM slots on completed tiles `0..U-tail` with 0 spin while W13's tail finishes!
//        - W2 (`pdl_wait_mode == 3`) does NOT block on `gdc_wait`: prefetches 4 slabs + WS3 + SLAB_META
//          and zeros `sA_all` padding BEFORE polling `ld.acquire.gpu(&TileReady[t]) == overlap_gen`
//          (`nanosleep.u32 64`), reads `SiTU_Out` via `ld_cg_u4` (100% stale-L1-proof!), and self-advances
//          GPU-resident `OverlapMeta[0]` (CUDA-Graph-safe, exact 0xFFFFFFFF wraparound TileReady reset!).
//   2. Hot-Expert L2 Cache-Hint (`hot_l2_hint=1`):
//        - Uses `ld.ca` when `Tiles[t+1] == e` (split tiles of hot experts reuse ~3.68 MB in L2!)
//        - Uses `ld.cs` on singleton tiles & final split tiles (0 L2 pollution for SiTU/DownCompact!)
//   3. L2-Friendly `st_cg_u4` on `FUSE_COMBINE == 4` (`DownCompact` directly into L2 for Combine).
//   4. 100% Drop-In Backward Compatibility:
//        - Exports `mx7_c32_version() == 8`, 6 new `mx7_{w13,w2}_c48_v8{_compat,_panel}_fwd` entrypoints,
//          plus ALL v4/v5/v7 pack/transcode/forward/combine symbols unchanged.
//
// Build on GB10 (SM121a):
//   /usr/local/cuda/bin/nvcc -O3 -std=c++17 -gencode arch=compute_121a,code=sm_121a \
//       --fmad=false --shared -Xcompiler -fPIC --ptxas-options=-v \
//       mxfp8_mx7_c48_v8.cu -o libmxfp8_mx7_c48_v8.so
//   (CRITICAL: --fmad=false required; OMIT --ftz=true and --use_fast_math!)

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

// Auto-probed by build_c28_moe_v8.sh (6 -> 5 -> 4, guaranteeing 0 stack & 0 spills on GB10)
#ifndef MX7_W13_T8_MIN_CTA
#define MX7_W13_T8_MIN_CTA 5
#endif
#ifndef MX7_W2_T8_MIN_CTA
#define MX7_W2_T8_MIN_CTA 5
#endif

namespace {

static constexpr int MX7_SLAB_BYTES      = 896;   // 512 sm4 + 256 e21 + 128 e0 = 7 * 128 B
static constexpr int MX7_RAW8_SLAB_BYTES = 1024;  // 2 * 512 B (2 x uint4/lane) = 8 * 128 B
static constexpr int MX7_PANEL_MX7_BYTES = 3584;  // 4 warps * 896 B  = 28 * 128 B
static constexpr int MX7_PANEL_R8_BYTES  = 4096;  // 4 warps * 1024 B = 32 * 128 B
static constexpr int W13_WS3_HDR_BLOCKS  = 20;    // 4 warps * 640 B = 2560 B = 20 * 128 B (128B/warp aligned!)
static constexpr int W2_WS3_HDR_BLOCKS   = 5;     // 4 warps * 160 B =  640 B =  5 * 128 B (32B/warp aligned!)
static constexpr int W13_WS3_WARP_U32    = 160;   // 640 B / 4
static constexpr int W2_WS3_WARP_U32     = 40;    // 160 B / 4

// SLAB_META bit layout (uint32_t per slab):
//   bit  31    : META_DENSE_BIT      (1 = 128B BMASK in Compat / RAW8 in Panel; 0 = sparse 0..63)
//   bit  30    : META_ALL_NORMAL_BIT (1 = zero e==0 and zero NaN -> direct SWAR BF16 unconditionally)
//   bit  29    : META_HAS_NAN_BIT    (1 = >=1 E4M3 NaN 0x7F/0xFF -> exact cvt fallback)
//   bits 28:23 : nf (0..63) when sparse
//   bits 22:0  : warp-relative offset (s_rel in SFLIPS or d_rel in BMASKS)
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
// PTX Memory, Sync & IEEE-754 Exact Math Helpers
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

static __device__ __forceinline__ uint32_t ld_ca_u32(const void* p) {
    uint32_t v;
    asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
}

static __device__ __forceinline__ uint16_t ld_cs_u16(const void* p) {
    uint16_t v;
    asm volatile("ld.global.cs.u16 %0, [%1];" : "=h"(v) : "l"(p));
    return v;
}

static __device__ __forceinline__ uint16_t ld_ca_u16(const void* p) {
    uint16_t v;
    asm volatile("ld.global.ca.u16 %0, [%1];" : "=h"(v) : "l"(p));
    return v;
}

// Hot-expert L2 hint selector: CTA-uniform `keep_l2` -> ld.ca for split tiles, ld.cs for singletons/last split
static __device__ __forceinline__ uint4 ld_w_u4(const void* p, bool keep_l2) {
    return keep_l2 ? ld_ca_u4(p) : ld_cs_u4(p);
}
static __device__ __forceinline__ uint2 ld_w_u2(const void* p, bool keep_l2) {
    return keep_l2 ? ld_ca_u2(p) : ld_cs_u2(p);
}
static __device__ __forceinline__ uint32_t ld_w_u32(const void* p, bool keep_l2) {
    return keep_l2 ? ld_ca_u32(p) : ld_cs_u32(p);
}
static __device__ __forceinline__ uint16_t ld_w_u16(const void* p, bool keep_l2) {
    return keep_l2 ? ld_ca_u16(p) : ld_cs_u16(p);
}

static __device__ __forceinline__ uint4 ld_cg_u4(const void* p) {
    uint4 v;
    asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

static __device__ __forceinline__ void st_cg_u4(void* p, uint4 v) {
    asm volatile("st.global.cg.v4.u32 [%0], {%1,%2,%3,%4};"
                 :: "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}

static __device__ __forceinline__ void st_cg_u2(void* p, uint2 v) {
    asm volatile("st.global.cg.v2.u32 [%0], {%1,%2};"
                 :: "l"(p), "r"(v.x), "r"(v.y) : "memory");
}

static __device__ __forceinline__ void nanosleep_u32(uint32_t ns) {
    asm volatile("nanosleep.u32 %0;" :: "r"(ns));
}

// 3-SHFL In-Register 4x4 Transpose for W2 Coalesced 64B uint4 Stores (0-ULP bit-exact!)
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

static __device__ __forceinline__ void prefetch_l2(const void* p) {
    asm volatile("prefetch.global.L2 [%0];" :: "l"(p));
}

static __device__ __forceinline__ uint32_t atom_acq_rel_gpu_add(uint32_t* p, uint32_t v) {
    uint32_t old;
    asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                 : "=r"(old) : "l"(p), "r"(v) : "memory");
    return old;
}

static __device__ __forceinline__ uint32_t ld_acq_gpu_u32(const uint32_t* p) {
    uint32_t v;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

static __device__ __forceinline__ void st_rel_gpu_u32(uint32_t* p, uint32_t v) {
    asm volatile("st.release.gpu.global.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}

static __device__ __forceinline__ uint32_t read_overlap_gen(const uint32_t* OverlapMeta) {
    uint32_t g = ld_acq_gpu_u32(&OverlapMeta[0]) + 1u;
    return (g == 0u) ? 1u : g;
}

static __device__ __forceinline__ void w13_mark_tile_panel_done(
    int t, int n_active, int n_panels, uint32_t gen,
    uint32_t* TileW13Done, uint32_t* TileReady, uint32_t* OverlapMeta
) {
    const uint32_t prev = atom_acq_rel_gpu_add(&TileW13Done[t], 1u);
    if (prev == static_cast<uint32_t>(n_panels - 1)) {
        st_rel_gpu_u32(&TileW13Done[t], 0u);
        __threadfence();
        st_rel_gpu_u32(&TileReady[t], gen);
        const uint32_t prev_t = atom_acq_rel_gpu_add(&OverlapMeta[1], 1u);
        if (prev_t == static_cast<uint32_t>(n_active - 1)) {
            st_rel_gpu_u32(&OverlapMeta[1], 0u);
        }
    }
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

// =========================================================================
// Compact Slab Structs (NO ws3 inside -> saves 8 regs per double-buffered pair!)
// =========================================================================

struct SlabMX7Lite {
    uint4    sm;   // 16 B/lane (512 B): 8 x uint16 sm4
    uint2    e21;  //  8 B/lane (256 B): 8 x uint8 e21
    uint32_t e0;   //  4 B/lane (128 B): 8 x 4-bit e0
};

struct SlabPanelV5 {
    uint4 a;       // MX7: sm (16 B/lane)             | RAW8: raw0 steps 0..3 (16 B/lane)
    uint4 b;       // MX7: {e21.x, e21.y, e0, 0}      | RAW8: raw1 steps 4..7 (16 B/lane)
};

static __device__ __forceinline__ SlabMX7Lite load_slab_mx7_lite(
    const uint8_t* sp, int lane, bool keep_l2 = false
) {
    SlabMX7Lite r;
    r.sm  = ld_w_u4(sp + lane * 16, keep_l2);
    r.e21 = ld_w_u2(sp + 512 + lane * 8, keep_l2);
    r.e0  = ld_w_u32(sp + 768 + lane * 4, keep_l2);
    return r;
}

static __device__ __forceinline__ SlabPanelV5 load_slab_panel_v5(
    const uint8_t* sp, int lane, bool is_raw8, bool keep_l2 = false
) {
    SlabPanelV5 r;
    if (is_raw8) {
        r.a = ld_w_u4(sp + lane * 16, keep_l2);
        r.b = ld_w_u4(sp + 512 + lane * 16, keep_l2);
    } else {
        r.a = ld_w_u4(sp + lane * 16, keep_l2);
        uint2 e21 = ld_w_u2(sp + 512 + lane * 8, keep_l2);
        uint32_t e0 = ld_w_u32(sp + 768 + lane * 4, keep_l2);
        r.b = make_uint4(e21.x, e21.y, e0, 0u);
    }
    return r;
}

// =========================================================================
// Fast SWAR E4M3 & Direct 0-ULP BF16x2 Decoders
// =========================================================================

// 2.4x fewer INT32 instructions than v4 `build_e4_pack_flip` (15 vs 36 insts, 100% bit-exact)
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

// Exact reference E4M3x4 * e8m0 -> BF16x4 (handles ALL 256 E4M3FN bytes + ALL e8m0 scales)
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

// Direct 0-ULP SWAR MX7 -> Scaled BF16x4 (valid for all normal E4M3 e in 1..15, not NaN, ws in 8..245)
// ZERO cvt.f16, ZERO cvt.f32, ZERO mul.f32, ZERO cvt.bf16 -> 0 CVT/FP32 pipe stalls!
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

// Direct 0-ULP SWAR RAW8 E4M3x4 -> Scaled BF16x4 (ONLY ~10 INT32 insts for 4 weights!)
static __device__ __forceinline__ void e4_pack_normal_to_bf16x4(
    uint32_t e4_pack, uint32_t ws_off2, uint32_t& b0, uint32_t& b1
) {
    const uint32_t u0 = prmt_b32(e4_pack, 0u, 0x4140u);
    const uint32_t u1 = prmt_b32(e4_pack, 0u, 0x4342u);
    b0 = (((u0 << 4) & 0x07F007F0u) + ws_off2) | ((u0 << 8) & 0x80008000u);
    b1 = (((u1 << 4) & 0x07F007F0u) + ws_off2) | ((u1 << 8) & 0x80008000u);
}

// Unified 0-ULP MX7 step decoder (handles ALL 256 E4M3FN bytes & ALL e8m0 scales)
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

// Unified 0-ULP RAW8 step decoder (handles ALL 256 E4M3FN bytes & ALL e8m0 scales)
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
// Flip Decoders: Legacy v4 Streaming vs v5 Adaptive Hybrid (Zero-Loop!)
// =========================================================================

static __device__ __forceinline__ uint32_t finish_slab_flip_mask_legacy(
    uint32_t rec, const uint16_t* __restrict__ FLIPS, int32_t& f_cur, int32_t f_end,
    uint32_t sl, int lane, uint32_t* sFlipWarp, bool keep_l2 = false
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
        const uint32_t r2 = (idx < f_end) ? static_cast<uint32_t>(ld_w_u16(FLIPS + idx, keep_l2)) : 0xFFFFu;
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

// Issue prefetched flip load for slab `meta` (100% parallel with SlabMX7 load!)
static __device__ __forceinline__ uint32_t issue_hybrid_flip_load(
    uint32_t meta,
    const uint16_t* __restrict__ SFLIPS, int32_t s_base,
    const uint32_t* __restrict__ BMASKS, int32_t d_base,
    int lane, bool is_panel_mode, bool keep_l2 = false
) {
    if (meta & META_DENSE_BIT) {
        if (is_panel_mode) return 0u;  // Panel RAW8 has zero flips!
        const int64_t d_idx = static_cast<int64_t>(d_base) + static_cast<int64_t>(meta & META_OFF_MASK);
        return ld_w_u32(BMASKS + (d_idx << 5) + lane, keep_l2);
    }
    const uint32_t nf = (meta >> META_NF_SHIFT) & META_NF_MASK;
    if (nf == 0u) return 0u;
    const int64_t s_idx = static_cast<int64_t>(s_base) + static_cast<int64_t>(meta & META_OFF_MASK);
    return (static_cast<uint32_t>(lane) < nf) ? static_cast<uint32_t>(ld_w_u16(SFLIPS + s_idx + lane, keep_l2)) : 0u;
}

// Finalize flip bitmask for slab `meta`:
//   - MODE_BITMASK / MODE_RAW8 / MODE_ZERO: ZERO SMEM, ZERO atomicOr, ZERO __syncwarp()!
//   - MODE_SPARSE (1..32): 1 round of atomicOr, ZERO while loop, pre-encoded (tl<<5)|bit!
//   - MODE_SPARSE (33..63, only if flip_thresh > 32): 1 unrolled 2nd pass, ZERO while loop!
static __device__ __forceinline__ uint32_t finish_hybrid_flip_mask(
    uint32_t raw_val, uint32_t meta,
    const uint16_t* __restrict__ SFLIPS, int32_t s_base,
    int lane, uint32_t* sFlipWarp, bool is_panel_mode, bool keep_l2 = false
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
            const uint32_t r2 = static_cast<uint32_t>(ld_w_u16(SFLIPS + s_idx + idx2, keep_l2));
            atomicOr(&sFlipWarp[r2 >> 5], 1u << (r2 & 31u));
        }
    }
    __syncwarp();
    return sFlipWarp[lane];
}

// =========================================================================
// Exact 0-ULP SiTU Helpers (GATE_CAP=4, UP_CAP=25) & W2 OTF Quant-Dequant
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

// Exact 0-ULP Register-Compact MXFP8 32-elem Quant-Dequant Directly Into sA_all
// (`cvt.rn.satfinite.e4m3x2.f32` clamps +/-inf and |x|>448 to +/-448 and passes NaN -> 0x7F in HW!)
static __device__ __forceinline__ void quant_dequant_mxfp8_32_to_sA(
    const uint4 v[4], __nv_bfloat16* dst
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

    const float a_sc = e8m0_to_f32(static_cast<uint8_t>(exp_u));

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t u[4] = {v[i].x, v[i].y, v[i].z, v[i].w};
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
            *reinterpret_cast<uint2*>(dst + i * 8 + h * 4) = make_uint2(b0, b1);
        }
    }
}

// =========================================================================
// Self-Contained GPU Packers & Zero-Raw-Weight v5 Transcoders (0 B Spills!)
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
void mx7_c32_count_flips_kernel(
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
void mx7_c32_pack_kernel(
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

// Helper: scan all flips for slab `sl` in sorted legacy FLIPS[f_cur .. f_end), returning
// exact count `nf` and lane flip bitmask `fmask` in sFlipWarp!
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

// Helper: classify slab for `META_ALL_NORMAL_BIT` & `META_HAS_NAN_BIT` (and optionally output 8 RAW8 words)
static __device__ __forceinline__ void classify_and_reconstruct_slab(
    const SlabMX7Lite& slb, uint32_t fmask,
    bool& slab_all_normal, bool& slab_has_nan,
    uint4& raw0, uint4& raw1
) {
    const uint32_t sm_arr[8] = {
        slb.sm.x & 0xFFFFu, slb.sm.x >> 16,
        slb.sm.y & 0xFFFFu, slb.sm.y >> 16,
        slb.sm.z & 0xFFFFu, slb.sm.z >> 16,
        slb.sm.w & 0xFFFFu, slb.sm.w >> 16
    };
    const uint32_t e21_arr[8] = {
        slb.e21.x & 0xFFu, (slb.e21.x >> 8) & 0xFFu,
        (slb.e21.x >> 16) & 0xFFu, (slb.e21.x >> 24) & 0xFFu,
        slb.e21.y & 0xFFu, (slb.e21.y >> 8) & 0xFFu,
        (slb.e21.y >> 16) & 0xFFu, (slb.e21.y >> 24) & 0xFFu
    };
    uint32_t e4[8];
    bool any_e0 = false;
    bool any_nan = false;
    #pragma unroll
    for (int s = 0; s < 8; ++s) {
        const uint32_t p = build_e4_pack_compat_fast(
            sm_arr[s], e21_arr[s], (slb.e0 >> (s * 4)) & 0xFu, (fmask >> (s * 4)) & 0xFu
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

// Pass 1 (Compat-Hybrid): Count per-warp sparse flips and dense bitmasks directly from legacy FLIPS/F_OFF!
__global__ __launch_bounds__(32, 1)
void mx7_v5_count_hybrid_from_mx7_kernel(
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

    int32_t s_cnt = 0;
    int32_t d_cnt = 0;
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
        if (nf > thr) {
            d_cnt += 1;
        } else {
            s_cnt += static_cast<int32_t>(nf);
        }
    }
    if (lane == 0) {
        sparse_counts[warp_uid] = s_cnt;
        dense_counts[warp_uid]  = d_cnt;
    }
}

// Pass 2 (Compat-Hybrid): Build SLAB_META, pre-encoded SFLIPS, and 128B-aligned BMASKS directly from live MX7!
__global__ __launch_bounds__(32, 1)
void mx7_v5_build_hybrid_from_mx7_kernel(
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

    uint32_t s_rel = 0u;
    uint32_t d_rel = 0u;
    for (int sl = 0; sl < n_slabs; ++sl) {
        const int32_t f_sl_start = f_cur;
        uint32_t fmask = 0u;
        const uint32_t nf = scan_legacy_slab_flips(
            FLIPS, f_cur, f_end, static_cast<uint32_t>(sl), lane, sFlip, fmask
        );
        const int64_t slab_idx = warp_uid * n_slabs + sl;
        const SlabMX7Lite slb = load_slab_mx7_lite(SLABS + slab_idx * MX7_SLAB_BYTES, lane);
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

// Pass 1 (Panel-V5): Measure panel-contiguous 128B blocks + sparse flip counts from live MX7!
__global__ __launch_bounds__(128, 1)
void mx7_v5_panel_measure_from_mx7_kernel(
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

    int32_t s_cnt = 0;
    int32_t body_blocks = 0;
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
            body_blocks += 32;  // 4096 B = 32 * 128 B
        } else {
            body_blocks += 28;  // 3584 B = 28 * 128 B
            s_cnt += static_cast<int32_t>(nf);
        }
        __syncthreads();
    }
    if (tid == 0) panel_blocks_out[panel_uid] = ws3_hdr_blocks + body_blocks;
    if (lane == 0) sparse_counts[warp_uid] = s_cnt;
}

// Pass 2 (Panel-V5): Transcode live (SLABS, WS3, FLIPS, F_OFF) -> Panel-Contiguous Super-Slabs!
__global__ __launch_bounds__(128, 1)
void mx7_v5_panel_build_from_mx7_kernel(
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
    // Copy WS3 into per-warp aligned header (zeroing any padding words)
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
        const SlabMX7Lite slb = load_slab_mx7_lite(SLABS + slab_idx * MX7_SLAB_BYTES, lane);
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
            *reinterpret_cast<uint4*>(dst + lane * 16)            = slb.sm;
            *reinterpret_cast<uint2*>(dst + 512 + lane * 8)       = slb.e21;
            *reinterpret_cast<uint32_t*>(dst + 768 + lane * 4)    = slb.e0;
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
// Kernel 1 (C48 v8): Unified W13 (N=576, K=4608, BN=32, 4 warps, 0-ULP, 0 spills!)
//   V5_MODE == 0: Legacy v4 ABI (upgraded w/ fast SWAR + batched WS3 + early nxt_slab!)
//   V5_MODE == 1: Compat-Hybrid (existing SLABS/WS3 + SLAB_META/SFLIPS/BMASKS + Direct SWAR BF16!)
//   V5_MODE == 2: Panel-V5 (panel-contiguous super-slabs + inline WS3 + adaptive MX7/RAW8 2x uint4!)
//   + v8 Deadlock-Free W13->W2 Tail-Wave PDL & TileReady[512] Producer + Hot-Expert L2 Cache-Hint!
// =========================================================================
template <
    int V5_MODE, int TILE, int MIN_CTA, bool USE_PDL,
    bool ZERO_SCATTER, bool AS_ROW_MAJOR, bool FUSE_SITU
>
__global__ __launch_bounds__(128, MIN_CTA)
void mx7_w13_c48_v8_kernel(
    const uint8_t* __restrict__ A,
    const uint8_t* __restrict__ AS,
    const int32_t* __restrict__ Row2Tok,
    const uint8_t* __restrict__ SLABS,        // Compat/Legacy SLABS OR Panel PANEL_POOL
    const uint4* __restrict__ WS3,            // Compat/Legacy WS3 (unused in Panel)
    const uint16_t* __restrict__ FLIPS,       // Legacy FLIPS OR v5 SFLIPS
    const int32_t* __restrict__ F_OFF,        // Legacy F_OFF OR v5 S_OFF
    const uint32_t* __restrict__ SLAB_META,   // v5 SLAB_META (nullptr in Legacy)
    const uint32_t* __restrict__ BMASKS,      // Compat BMASKS (nullptr in Legacy/Panel)
    const int32_t* __restrict__ AUX_OFF,      // Compat D_OFF OR Panel PANEL_OFF
    const float* __restrict__ Ones,
    const int32_t* __restrict__ Tiles,
    const int32_t* __restrict__ TileCompactStart,
    const float* __restrict__ Gammas,
    __nv_bfloat16* __restrict__ Out,
    uint32_t* __restrict__ TileW13Done,
    uint32_t* __restrict__ TileReady,
    uint32_t* __restrict__ OverlapMeta,
    int N, int K, int MAX_CTAS, uint32_t base_ws,
    int pdl_wait_mode, int trig_dep, int overlap_w13_w2, int hot_l2_hint
) {
    if constexpr (USE_PDL) {
        if (pdl_wait_mode == 1) {
            asm volatile("griddepcontrol.wait;" ::: "memory");
        }
    }

    constexpr int WARPS = 4;
    constexpr int THREADS = 128;
    constexpr int BN = 32;
    constexpr int SLAB_K = 128;
    constexpr int BK_SMEM = (TILE <= 16) ? 256 : 128;
    constexpr int SLABS_PER_BLK = BK_SMEM / SLAB_K;  // 2 for TILE<=16, 1 for TILE=32
    constexpr int M_ROWS = TILE;
    constexpr int M_STEPS = (TILE <= 16) ? 1 : 2;
    constexpr int SA_STRIDE = BK_SMEM + 16;          // 272 or 144 -> 0-way LDS.64!

    const int n_panels = N / BN;     // 18
    const int n_active = Tiles[2 * MAX_CTAS];
    const int n_total = n_active * n_panels;
    const int stride = static_cast<int>(gridDim.x);
    const int bidx = static_cast<int>(blockIdx.x);

    if (bidx >= n_total) {
        if constexpr (USE_PDL) {
            if (pdl_wait_mode == 2) asm volatile("griddepcontrol.wait;" ::: "memory");
            if (trig_dep) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
        return;
    }

    const int n_slabs = K / SLAB_K;   // 36
    const int n_blks  = K / BK_SMEM;  // 18 or 36
    const int KG      = K >> 5;       // 144
    const bool fast_ws = (base_ws >= 8u && base_ws <= 238u);

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g    = lane >> 2;
    const int lt   = lane & 3;
    const int x0_0 = g * SA_STRIDE + 4 * lt;
    [[maybe_unused]] const int x0_1 = (g + 8) * SA_STRIDE + 4 * lt;
    [[maybe_unused]] const int x1_0 = (g + 16) * SA_STRIDE + 4 * lt;
    [[maybe_unused]] const int x1_1 = (g + 24) * SA_STRIDE + 4 * lt;
    [[maybe_unused]] const int lm_a_t8  = tid >> 4;   // 0..7 for TILE==8 1-shot LDG.128!
    [[maybe_unused]] const int q16_a_t8 = tid & 15;   // 0..15 (16 x 16 = 256 = BK_SMEM)

    alignas(16) __shared__ uint32_t sFlip[WARPS][32];
    alignas(16) __shared__ int32_t  sTok[(TILE == 8) ? 1 : M_ROWS];
    alignas(16) __shared__ __nv_bfloat16 sA[2][M_ROWS * SA_STRIDE];

    const bool use_overlap = FUSE_SITU && (overlap_w13_w2 != 0) &&
        (TileW13Done != nullptr) && (TileReady != nullptr) && (OverlapMeta != nullptr);
    uint32_t overlap_gen = (use_overlap && !(USE_PDL && pdl_wait_mode == 2))
        ? read_overlap_gen(OverlapMeta) : 0u;
    const int rem_tail = n_total - (n_total / stride) * stride;
    const int trig_sl = (!use_overlap)
        ? (n_slabs - 2)
        : ((n_total > stride && rem_tail > 0)
            ? ((bidx < rem_tail) ? 0 : (n_slabs - 2))
            : (n_slabs - 6));

    bool trig_fired = false;

    for (int idx = bidx; idx < n_total; idx += stride) {
        const int t  = idx / n_panels;
        const int nb = idx - t * n_panels;
        const bool is_first_tile = (idx == bidx);
        const bool is_last_tile  = (idx + stride >= n_total);

        const int64_t e = static_cast<int64_t>(Tiles[t]);
        const int end_row = Tiles[MAX_CTAS + t];
        const int row_base = t * TILE;
        int live_m = end_row - row_base;
        if (live_m > TILE) live_m = TILE;
        if (live_m < 0) live_m = 0;

        const bool keep_l2 = (hot_l2_hint != 0) && (t + 1 < n_active) && (Tiles[t + 1] == static_cast<int32_t>(e));

        const int64_t panel_uid = e * n_panels + nb;
        const int64_t warp_uid  = panel_uid * WARPS + warp;

        const uint8_t* slab_base = nullptr;
        const uint32_t* ws_u32_base = nullptr;
        if constexpr (V5_MODE == 2) {
            const uint8_t* pbase = SLABS + static_cast<int64_t>(AUX_OFF[panel_uid]) * 128;
            ws_u32_base = reinterpret_cast<const uint32_t*>(pbase) + warp * W13_WS3_WARP_U32;
            slab_base   = pbase + static_cast<int64_t>(W13_WS3_HDR_BLOCKS) * 128;
        } else {
            slab_base   = SLABS + warp_uid * n_slabs * MX7_SLAB_BYTES;
            ws_u32_base = reinterpret_cast<const uint32_t*>(WS3 + warp_uid * n_slabs);
        }

        int32_t f_cur = F_OFF[warp_uid];
        const int32_t f_end = (V5_MODE == 0) ? F_OFF[warp_uid + 1] : 0;
        const int32_t d_base = (V5_MODE == 1) ? AUX_OFF[warp_uid] : 0;

        if constexpr (USE_PDL) {
            if (pdl_wait_mode == 2 && is_first_tile) {
                if constexpr (V5_MODE == 2) {
                    if (lane < 7) prefetch_l2(slab_base + warp * MX7_SLAB_BYTES + lane * 128);
                    if (lane >= 7 && lane < 12) prefetch_l2(ws_u32_base + (lane - 7) * 32);
                    if (lane >= 12 && lane < 14) prefetch_l2(SLAB_META + warp_uid * n_slabs + (lane - 12) * 32);
                } else {
                    if (lane < 7) prefetch_l2(slab_base + lane * 128);
                    if (lane >= 7 && lane < 14) prefetch_l2(slab_base + MX7_SLAB_BYTES + (lane - 7) * 128);
                    if (lane == 14) prefetch_l2(ws_u32_base);
                    if (lane == 15) {
                        if constexpr (V5_MODE == 0) {
                            if (f_cur < f_end) prefetch_l2(FLIPS + f_cur);
                        } else {
                            prefetch_l2(SLAB_META + warp_uid * n_slabs);
                        }
                    }
                }
                asm volatile("griddepcontrol.wait;" ::: "memory");
                if (use_overlap) overlap_gen = read_overlap_gen(OverlapMeta);
            }
        }
        if (live_m == 0) {
            if (use_overlap && tid == 0) {
                w13_mark_tile_panel_done(t, n_active, n_panels, overlap_gen, TileW13Done, TileReady, OverlapMeta);
            }
            continue;
        }
        if constexpr (USE_PDL) {
            if (trig_dep && use_overlap && is_last_tile && trig_sl == 0 && !trig_fired) {
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
                trig_fired = true;
            }
        }

        [[maybe_unused]] int64_t a_row_t8 = 0;
        if constexpr (TILE == 8) {
            if (lm_a_t8 < live_m) {
                const int64_t row_t8 = static_cast<int64_t>(row_base + lm_a_t8);
                a_row_t8 = ZERO_SCATTER ? static_cast<int64_t>(Row2Tok[row_t8]) : row_t8;
            }
        } else {
            if constexpr (ZERO_SCATTER) {
                if (tid < live_m) sTok[tid] = Row2Tok[row_base + tid];
            }
        }

        // Load ALL 36 SLAB_META entries ONCE into 2 registers (0 META loads in K-loop!)
        uint32_t meta0 = 0u, meta1 = 0u;
        if constexpr (V5_MODE != 0) {
            const uint32_t* mp = SLAB_META + warp_uid * n_slabs;
            meta0 = (lane < n_slabs) ? ld_w_u32(mp + lane, keep_l2) : 0u;
            meta1 = (lane + 32 < n_slabs) ? ld_w_u32(mp + 32 + lane, keep_l2) : 0u;
        }
        // Load first 8 slabs' WS3 (128 B full cache line, 0 overfetch!) into 1 register!
        const int ws_tot_u32 = n_slabs * 4;
        uint32_t ws_cur = (lane < ws_tot_u32) ? ld_w_u32(ws_u32_base + lane, keep_l2) : 0u;
        uint32_t ws_nxt = 0u;

        // Zero only padding rows inside active 8-row groups
        if constexpr (TILE == 8) {
            if (lm_a_t8 >= live_m) {
                const int off = lm_a_t8 * SA_STRIDE + (q16_a_t8 << 4);
                const uint4 z4 = make_uint4(0u, 0u, 0u, 0u);
                *reinterpret_cast<uint4*>(sA[0] + off)     = z4;
                *reinterpret_cast<uint4*>(sA[0] + off + 8) = z4;
                *reinterpret_cast<uint4*>(sA[1] + off)     = z4;
                *reinterpret_cast<uint4*>(sA[1] + off + 8) = z4;
            }
        } else {
            const int pad_limit = (live_m <= 8) ? 8
                : ((live_m <= 16) ? 16 : ((live_m <= 24) ? 24 : M_ROWS));
            if (live_m < pad_limit) {
                constexpr int Q_PER_ROW = BK_SMEM >> 2;
                const int zero_q = (pad_limit - live_m) * Q_PER_ROW;
                for (int zidx = tid; zidx < zero_q; zidx += THREADS) {
                    const int dlm = zidx / Q_PER_ROW;
                    const int qk  = zidx - dlm * Q_PER_ROW;
                    const int off = (live_m + dlm) * SA_STRIDE + (qk << 2);
                    *reinterpret_cast<uint2*>(sA[0] + off) = make_uint2(0u, 0u);
                    *reinterpret_cast<uint2*>(sA[1] + off) = make_uint2(0u, 0u);
                }
            }
        }

        auto load_a_blk = [&](int blk, __nv_bfloat16* dst, bool use_smem_tok) {
            if constexpr (TILE == 8) {
                // 1-shot loop-free LDG.128 (16 E4M3/thr, 1 AS byte/thr, direct register a_row_t8 -> 0 sTok SMEM!)
                (void)use_smem_tok;
                if (lm_a_t8 < live_m) {
                    const int k0 = blk * BK_SMEM;
                    const int k_off16 = q16_a_t8 << 4;
                    const int kg_col = (k0 >> 5) + (q16_a_t8 >> 1);
                    const uint4 a16 = ld_ca_u4(A + a_row_t8 * K + k0 + k_off16);
                    uint8_t as_b;
                    if constexpr (ZERO_SCATTER || AS_ROW_MAJOR) {
                        as_b = AS[a_row_t8 * KG + kg_col];
                    } else {
                        as_b = AS[scale_offset_decode(a_row_t8, kg_col, KG)];
                    }
                    const float a_sc = e8m0_to_f32(as_b);
                    uint32_t b0, b1, b2, b3, b4, b5, b6, b7;
                    e4m3x4_scale_to_bf16x4(a16.x, a_sc, b0, b1);
                    e4m3x4_scale_to_bf16x4(a16.y, a_sc, b2, b3);
                    e4m3x4_scale_to_bf16x4(a16.z, a_sc, b4, b5);
                    e4m3x4_scale_to_bf16x4(a16.w, a_sc, b6, b7);
                    __nv_bfloat16* d = dst + lm_a_t8 * SA_STRIDE + k_off16;
                    *reinterpret_cast<uint4*>(d)     = make_uint4(b0, b1, b2, b3);
                    *reinterpret_cast<uint4*>(d + 8) = make_uint4(b4, b5, b6, b7);
                }
            } else if constexpr (BK_SMEM == 256) {
                // Vectorized 8-elem uint2 load -> 16B uint4 SMEM store (2x fewer A/AS/sTok loads!)
                constexpr int E8_PER_ROW = 32;  // 256 / 8
                const int total_e8 = live_m * E8_PER_ROW;
                const int k0 = blk * BK_SMEM;
                for (int aidx = tid; aidx < total_e8; aidx += THREADS) {
                    const int lm = aidx >> 5;
                    const int q8 = aidx & 31;
                    const int64_t row = static_cast<int64_t>(row_base + lm);
                    const int64_t a_row = ZERO_SCATTER
                        ? static_cast<int64_t>(use_smem_tok ? sTok[lm] : Row2Tok[row])
                        : row;
                    const int kcol = k0 + (q8 << 3);
                    const int kg_col = kcol >> 5;
                    const uint2 a8 = ld_ca_u2(A + a_row * K + kcol);
                    uint8_t as_b;
                    if constexpr (ZERO_SCATTER || AS_ROW_MAJOR) {
                        as_b = AS[a_row * KG + kg_col];
                    } else {
                        as_b = AS[scale_offset_decode(row, kg_col, KG)];
                    }
                    const float a_sc = e8m0_to_f32(as_b);
                    uint32_t b0, b1, b2, b3;
                    e4m3x4_scale_to_bf16x4(a8.x, a_sc, b0, b1);
                    e4m3x4_scale_to_bf16x4(a8.y, a_sc, b2, b3);
                    *reinterpret_cast<uint4*>(dst + lm * SA_STRIDE + (q8 << 3)) = make_uint4(b0, b1, b2, b3);
                }
            } else {
                constexpr int Q_PER_ROW = 32;  // 128 / 4
                const int total_q = live_m * Q_PER_ROW;
                const int k0 = blk * BK_SMEM;
                for (int aidx = tid; aidx < total_q; aidx += THREADS) {
                    const int lm = aidx >> 5;
                    const int qk = aidx & 31;
                    const int64_t row = static_cast<int64_t>(row_base + lm);
                    const int64_t a_row = ZERO_SCATTER
                        ? static_cast<int64_t>(use_smem_tok ? sTok[lm] : Row2Tok[row])
                        : row;
                    const int kcol = k0 + (qk << 2);
                    const int kg_col = kcol >> 5;
                    const uint32_t a4 = *reinterpret_cast<const uint32_t*>(A + a_row * K + kcol);
                    uint8_t as_b;
                    if constexpr (ZERO_SCATTER || AS_ROW_MAJOR) {
                        as_b = AS[a_row * KG + kg_col];
                    } else {
                        as_b = AS[scale_offset_decode(row, kg_col, KG)];
                    }
                    const float a_sc = e8m0_to_f32(as_b);
                    uint32_t b0, b1;
                    e4m3x4_scale_to_bf16x4(a4, a_sc, b0, b1);
                    *reinterpret_cast<uint2*>(dst + lm * SA_STRIDE + (qk << 2)) = make_uint2(b0, b1);
                }
            }
        };

        // Prime slab 0 + flip 0 IN PARALLEL with A block 0!
        uint32_t cur_meta = (V5_MODE != 0) ? __shfl_sync(0xFFFFFFFFu, meta0, 0) : 0u;
        const uint8_t* panel_step_ptr = slab_base;
        SlabMX7Lite cur_slab_lite = {};
        SlabPanelV5 cur_slab_panel = {};
        uint32_t rec0 = 0u;

        if constexpr (V5_MODE == 0) {
            rec0 = (f_cur + lane < f_end) ? static_cast<uint32_t>(ld_w_u16(FLIPS + f_cur + lane, keep_l2)) : 0xFFFFu;
            cur_slab_lite = load_slab_mx7_lite(slab_base, lane, keep_l2);
        } else if constexpr (V5_MODE == 1) {
            cur_slab_lite = load_slab_mx7_lite(slab_base, lane, keep_l2);
            rec0 = issue_hybrid_flip_load(cur_meta, FLIPS, f_cur, BMASKS, d_base, lane, false, keep_l2);
        } else {
            const bool is_r8_0 = (cur_meta & META_DENSE_BIT) != 0u;
            cur_slab_panel = load_slab_panel_v5(
                panel_step_ptr + warp * (is_r8_0 ? MX7_RAW8_SLAB_BYTES : MX7_SLAB_BYTES), lane, is_r8_0, keep_l2
            );
            rec0 = issue_hybrid_flip_load(cur_meta, FLIPS, f_cur, nullptr, 0, lane, true, keep_l2);
        }

        load_a_blk(0, sA[0], false);

        uint32_t cur_fmask = 0u;
        if constexpr (V5_MODE == 0) {
            cur_fmask = finish_slab_flip_mask_legacy(rec0, FLIPS, f_cur, f_end, 0u, lane, sFlip[warp], keep_l2);
        } else {
            cur_fmask = finish_hybrid_flip_mask(rec0, cur_meta, FLIPS, f_cur, lane, sFlip[warp], V5_MODE == 2, keep_l2);
        }
        __syncthreads();

        float acc[M_STEPS][4];
        #pragma unroll
        for (int im = 0; im < M_STEPS; ++im) {
            acc[im][0] = 0.0f; acc[im][1] = 0.0f; acc[im][2] = 0.0f; acc[im][3] = 0.0f;
        }
        [[maybe_unused]] const bool has_m8  = (TILE > 8)  && (live_m > 8);
        [[maybe_unused]] const bool has_m16 = (TILE > 16) && (live_m > 16);
        [[maybe_unused]] const bool has_m24 = (TILE > 16) && (live_m > 24);

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

                // 1. Issue cold-DRAM nxt_slab + nxt_flip FIRST (BEFORE load_a_blk & BEFORE MMAs!)
                SlabMX7Lite nxt_slab_lite = {};
                SlabPanelV5 nxt_slab_panel = {};
                uint32_t nxt_meta = 0u;
                uint32_t nxt_rec  = 0u;

                if (sl + 1 < n_slabs) {
                    const int nsl = sl + 1;
                    if constexpr (V5_MODE == 0) {
                        nxt_slab_lite = load_slab_mx7_lite(slab_base + static_cast<int64_t>(nsl) * MX7_SLAB_BYTES, lane, keep_l2);
                        nxt_rec = (f_cur + lane < f_end) ? static_cast<uint32_t>(ld_w_u16(FLIPS + f_cur + lane, keep_l2)) : 0xFFFFu;
                    } else if constexpr (V5_MODE == 1) {
                        nxt_meta = (nsl < 32)
                            ? __shfl_sync(0xFFFFFFFFu, meta0, nsl)
                            : __shfl_sync(0xFFFFFFFFu, meta1, nsl - 32);
                        nxt_slab_lite = load_slab_mx7_lite(slab_base + static_cast<int64_t>(nsl) * MX7_SLAB_BYTES, lane, keep_l2);
                        nxt_rec = issue_hybrid_flip_load(nxt_meta, FLIPS, f_cur, BMASKS, d_base, lane, false, keep_l2);
                    } else {
                        nxt_meta = (nsl < 32)
                            ? __shfl_sync(0xFFFFFFFFu, meta0, nsl)
                            : __shfl_sync(0xFFFFFFFFu, meta1, nsl - 32);
                        const bool nxt_r8 = (nxt_meta & META_DENSE_BIT) != 0u;
                        nxt_slab_panel = load_slab_panel_v5(
                            panel_step_ptr + warp * (nxt_r8 ? MX7_RAW8_SLAB_BYTES : MX7_SLAB_BYTES), lane, nxt_r8, keep_l2
                        );
                        nxt_rec = issue_hybrid_flip_load(nxt_meta, FLIPS, f_cur, nullptr, 0, lane, true, keep_l2);
                    }
                    // 2-slab lookahead 128B-coalesced WS3 chunk prefetch (5 loads/tile, 0 overfetch!)
                    if ((sl & 7) == 6 && sl + 2 < n_slabs) {
                        const int off32 = ((sl + 2) >> 3) * 32 + lane;
                        ws_nxt = (off32 < ws_tot_u32) ? ld_w_u32(ws_u32_base + off32, keep_l2) : 0u;
                    }
                } else if (!is_last_tile) {
                    // Full 32-lane next-tile multi-slab L2 prefetch!
                    const int nidx = idx + stride;
                    const int nt   = nidx / n_panels;
                    const int nnb  = nidx - nt * n_panels;
                    const int64_t ne = static_cast<int64_t>(Tiles[nt]);
                    const int64_t npuid = ne * n_panels + nnb;
                    const int64_t nwuid = npuid * WARPS + warp;
                    if constexpr (V5_MODE == 2) {
                        const uint8_t* npb = SLABS + static_cast<int64_t>(AUX_OFF[npuid]) * 128;
                        const uint8_t* nbody = npb + static_cast<int64_t>(W13_WS3_HDR_BLOCKS) * 128;
                        if (lane < 7) prefetch_l2(nbody + warp * MX7_SLAB_BYTES + lane * 128);
                        else if (lane < 14) prefetch_l2(nbody + MX7_PANEL_MX7_BYTES + warp * MX7_SLAB_BYTES + (lane - 7) * 128);
                        else if (lane < 19) prefetch_l2(reinterpret_cast<const uint32_t*>(npb) + warp * W13_WS3_WARP_U32 + (lane - 14) * 32);
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

                // 2. Overlap L2-hot next A block load while cold-DRAM nxt_slab is in flight!
                if (is == 0 && blk + 1 < n_blks) {
                    load_a_blk(blk + 1, sAnext, true);
                }

                // 3. Unpack & execute 8..16 BF16 Tensor Core MMAs for current slab!
                const bool slab_all_normal = (V5_MODE != 0) && fast_ws && ((cur_meta & META_ALL_NORMAL_BIT) != 0u);
                const bool slab_no_nan     = (V5_MODE != 0) && fast_ws && ((cur_meta & META_HAS_NAN_BIT) == 0u);
                const int blk_k_off = is * SLAB_K;
                const int ws_sl_base = (sl & 7) << 2;

                if constexpr (V5_MODE == 2) {
                    if (cur_is_raw8) {
                        const uint32_t raw_arr[8] = {
                            cur_slab_panel.a.x, cur_slab_panel.a.y, cur_slab_panel.a.z, cur_slab_panel.a.w,
                            cur_slab_panel.b.x, cur_slab_panel.b.y, cur_slab_panel.b.z, cur_slab_panel.b.w
                        };
                        #pragma unroll
                        for (int q = 0; q < 4; ++q) {
                            const int kq_off = blk_k_off + q * 32;
                            const uint32_t ws_word = __shfl_sync(0xFFFFFFFFu, ws_cur, ws_sl_base | q);
                            const uint32_t ws_byte = base_ws + ((ws_word >> (g * 3)) & 7u);
                            const uint32_t ws_off2 = ((ws_byte - 7u) << 7) * 0x00010001u;
                            const float    w_sc    = __uint_as_float(ws_byte << 23);

                            uint32_t b0_0, b1_0, b0_1, b1_1;
                            decode_raw8_step_to_bf16x4(raw_arr[q << 1],       ws_off2, w_sc, slab_all_normal, slab_no_nan, b0_0, b1_0);
                            decode_raw8_step_to_bf16x4(raw_arr[(q << 1) | 1], ws_off2, w_sc, slab_all_normal, slab_no_nan, b0_1, b1_1);

                            {
                                const uint2 a0 = *reinterpret_cast<const uint2*>(sAp + x0_0 + kq_off);
                                const uint2 a2 = *reinterpret_cast<const uint2*>(sAp + x0_0 + kq_off + 16);
                                if constexpr (TILE == 8) {
                                    mma_m16n8k16_bf16(acc[0], a0.x, 0u, a2.x, 0u, b0_0, b1_0);
                                    mma_m16n8k16_bf16(acc[0], a0.y, 0u, a2.y, 0u, b0_1, b1_1);
                                } else {
                                    uint2 a1 = make_uint2(0u, 0u), a3 = make_uint2(0u, 0u);
                                    if (has_m8) {
                                        a1 = *reinterpret_cast<const uint2*>(sAp + x0_1 + kq_off);
                                        a3 = *reinterpret_cast<const uint2*>(sAp + x0_1 + kq_off + 16);
                                    }
                                    mma_m16n8k16_bf16(acc[0], a0.x, a1.x, a2.x, a3.x, b0_0, b1_0);
                                    mma_m16n8k16_bf16(acc[0], a0.y, a1.y, a2.y, a3.y, b0_1, b1_1);
                                }
                            }
                            if constexpr (M_STEPS == 2) {
                                if (has_m16) {
                                    const uint2 c0 = *reinterpret_cast<const uint2*>(sAp + x1_0 + kq_off);
                                    const uint2 c2 = *reinterpret_cast<const uint2*>(sAp + x1_0 + kq_off + 16);
                                    uint2 c1 = make_uint2(0u, 0u), c3 = make_uint2(0u, 0u);
                                    if (has_m24) {
                                        c1 = *reinterpret_cast<const uint2*>(sAp + x1_1 + kq_off);
                                        c3 = *reinterpret_cast<const uint2*>(sAp + x1_1 + kq_off + 16);
                                    }
                                    mma_m16n8k16_bf16(acc[1], c0.x, c1.x, c2.x, c3.x, b0_0, b1_0);
                                    mma_m16n8k16_bf16(acc[1], c0.y, c1.y, c2.y, c3.y, b0_1, b1_1);
                                }
                            }
                        }
                    } else {
                        const uint32_t sm_arr[8] = {
                            cur_slab_panel.a.x & 0xFFFFu, cur_slab_panel.a.x >> 16,
                            cur_slab_panel.a.y & 0xFFFFu, cur_slab_panel.a.y >> 16,
                            cur_slab_panel.a.z & 0xFFFFu, cur_slab_panel.a.z >> 16,
                            cur_slab_panel.a.w & 0xFFFFu, cur_slab_panel.a.w >> 16
                        };
                        const uint32_t e21_arr[8] = {
                            cur_slab_panel.b.x & 0xFFu, (cur_slab_panel.b.x >> 8) & 0xFFu,
                            (cur_slab_panel.b.x >> 16) & 0xFFu, (cur_slab_panel.b.x >> 24) & 0xFFu,
                            cur_slab_panel.b.y & 0xFFu, (cur_slab_panel.b.y >> 8) & 0xFFu,
                            (cur_slab_panel.b.y >> 16) & 0xFFu, (cur_slab_panel.b.y >> 24) & 0xFFu
                        };
                        const uint32_t e0_reg = cur_slab_panel.b.z;
                        #pragma unroll
                        for (int q = 0; q < 4; ++q) {
                            const int kq_off = blk_k_off + q * 32;
                            const uint32_t ws_word = __shfl_sync(0xFFFFFFFFu, ws_cur, ws_sl_base | q);
                            const uint32_t ws_byte = base_ws + ((ws_word >> (g * 3)) & 7u);
                            const uint32_t ws_off2 = ((ws_byte - 7u) << 7) * 0x00010001u;
                            const float    w_sc    = __uint_as_float(ws_byte << 23);

                            const int s0 = q << 1;
                            const int s1 = s0 | 1;
                            uint32_t b0_0, b1_0, b0_1, b1_1;
                            decode_mx7_step_to_bf16x4(
                                sm_arr[s0], e21_arr[s0], (e0_reg >> (s0 * 4)) & 0xFu, (cur_fmask >> (s0 * 4)) & 0xFu,
                                ws_off2, w_sc, slab_all_normal, slab_no_nan, b0_0, b1_0);
                            decode_mx7_step_to_bf16x4(
                                sm_arr[s1], e21_arr[s1], (e0_reg >> (s1 * 4)) & 0xFu, (cur_fmask >> (s1 * 4)) & 0xFu,
                                ws_off2, w_sc, slab_all_normal, slab_no_nan, b0_1, b1_1);

                            {
                                const uint2 a0 = *reinterpret_cast<const uint2*>(sAp + x0_0 + kq_off);
                                const uint2 a2 = *reinterpret_cast<const uint2*>(sAp + x0_0 + kq_off + 16);
                                if constexpr (TILE == 8) {
                                    mma_m16n8k16_bf16(acc[0], a0.x, 0u, a2.x, 0u, b0_0, b1_0);
                                    mma_m16n8k16_bf16(acc[0], a0.y, 0u, a2.y, 0u, b0_1, b1_1);
                                } else {
                                    uint2 a1 = make_uint2(0u, 0u), a3 = make_uint2(0u, 0u);
                                    if (has_m8) {
                                        a1 = *reinterpret_cast<const uint2*>(sAp + x0_1 + kq_off);
                                        a3 = *reinterpret_cast<const uint2*>(sAp + x0_1 + kq_off + 16);
                                    }
                                    mma_m16n8k16_bf16(acc[0], a0.x, a1.x, a2.x, a3.x, b0_0, b1_0);
                                    mma_m16n8k16_bf16(acc[0], a0.y, a1.y, a2.y, a3.y, b0_1, b1_1);
                                }
                            }
                            if constexpr (M_STEPS == 2) {
                                if (has_m16) {
                                    const uint2 c0 = *reinterpret_cast<const uint2*>(sAp + x1_0 + kq_off);
                                    const uint2 c2 = *reinterpret_cast<const uint2*>(sAp + x1_0 + kq_off + 16);
                                    uint2 c1 = make_uint2(0u, 0u), c3 = make_uint2(0u, 0u);
                                    if (has_m24) {
                                        c1 = *reinterpret_cast<const uint2*>(sAp + x1_1 + kq_off);
                                        c3 = *reinterpret_cast<const uint2*>(sAp + x1_1 + kq_off + 16);
                                    }
                                    mma_m16n8k16_bf16(acc[1], c0.x, c1.x, c2.x, c3.x, b0_0, b1_0);
                                    mma_m16n8k16_bf16(acc[1], c0.y, c1.y, c2.y, c3.y, b0_1, b1_1);
                                }
                            }
                        }
                    }
                } else {
                    const uint32_t sm_arr[8] = {
                        cur_slab_lite.sm.x & 0xFFFFu, cur_slab_lite.sm.x >> 16,
                        cur_slab_lite.sm.y & 0xFFFFu, cur_slab_lite.sm.y >> 16,
                        cur_slab_lite.sm.z & 0xFFFFu, cur_slab_lite.sm.z >> 16,
                        cur_slab_lite.sm.w & 0xFFFFu, cur_slab_lite.sm.w >> 16
                    };
                    const uint32_t e21_arr[8] = {
                        cur_slab_lite.e21.x & 0xFFu, (cur_slab_lite.e21.x >> 8) & 0xFFu,
                        (cur_slab_lite.e21.x >> 16) & 0xFFu, (cur_slab_lite.e21.x >> 24) & 0xFFu,
                        cur_slab_lite.e21.y & 0xFFu, (cur_slab_lite.e21.y >> 8) & 0xFFu,
                        (cur_slab_lite.e21.y >> 16) & 0xFFu, (cur_slab_lite.e21.y >> 24) & 0xFFu
                    };
                    #pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        const int kq_off = blk_k_off + q * 32;
                        const uint32_t ws_word = __shfl_sync(0xFFFFFFFFu, ws_cur, ws_sl_base | q);
                        const uint32_t ws_byte = base_ws + ((ws_word >> (g * 3)) & 7u);
                        const uint32_t ws_off2 = ((ws_byte - 7u) << 7) * 0x00010001u;
                        const float    w_sc    = __uint_as_float(ws_byte << 23);

                        const int s0 = q << 1;
                        const int s1 = s0 | 1;
                        uint32_t b0_0, b1_0, b0_1, b1_1;
                        if constexpr (V5_MODE == 0) {
                            e4m3x4_scale_to_bf16x4(
                                build_e4_pack_compat_fast(sm_arr[s0], e21_arr[s0], (cur_slab_lite.e0 >> (s0 * 4)) & 0xFu, (cur_fmask >> (s0 * 4)) & 0xFu),
                                w_sc, b0_0, b1_0);
                            e4m3x4_scale_to_bf16x4(
                                build_e4_pack_compat_fast(sm_arr[s1], e21_arr[s1], (cur_slab_lite.e0 >> (s1 * 4)) & 0xFu, (cur_fmask >> (s1 * 4)) & 0xFu),
                                w_sc, b0_1, b1_1);
                        } else {
                            decode_mx7_step_to_bf16x4(
                                sm_arr[s0], e21_arr[s0], (cur_slab_lite.e0 >> (s0 * 4)) & 0xFu, (cur_fmask >> (s0 * 4)) & 0xFu,
                                ws_off2, w_sc, slab_all_normal, slab_no_nan, b0_0, b1_0);
                            decode_mx7_step_to_bf16x4(
                                sm_arr[s1], e21_arr[s1], (cur_slab_lite.e0 >> (s1 * 4)) & 0xFu, (cur_fmask >> (s1 * 4)) & 0xFu,
                                ws_off2, w_sc, slab_all_normal, slab_no_nan, b0_1, b1_1);
                        }

                        {
                            const uint2 a0 = *reinterpret_cast<const uint2*>(sAp + x0_0 + kq_off);
                            const uint2 a2 = *reinterpret_cast<const uint2*>(sAp + x0_0 + kq_off + 16);
                            if constexpr (TILE == 8) {
                                mma_m16n8k16_bf16(acc[0], a0.x, 0u, a2.x, 0u, b0_0, b1_0);
                                mma_m16n8k16_bf16(acc[0], a0.y, 0u, a2.y, 0u, b0_1, b1_1);
                            } else {
                                uint2 a1 = make_uint2(0u, 0u), a3 = make_uint2(0u, 0u);
                                if (has_m8) {
                                    a1 = *reinterpret_cast<const uint2*>(sAp + x0_1 + kq_off);
                                    a3 = *reinterpret_cast<const uint2*>(sAp + x0_1 + kq_off + 16);
                                }
                                mma_m16n8k16_bf16(acc[0], a0.x, a1.x, a2.x, a3.x, b0_0, b1_0);
                                mma_m16n8k16_bf16(acc[0], a0.y, a1.y, a2.y, a3.y, b0_1, b1_1);
                            }
                        }
                        if constexpr (M_STEPS == 2) {
                            if (has_m16) {
                                const uint2 c0 = *reinterpret_cast<const uint2*>(sAp + x1_0 + kq_off);
                                const uint2 c2 = *reinterpret_cast<const uint2*>(sAp + x1_0 + kq_off + 16);
                                uint2 c1 = make_uint2(0u, 0u), c3 = make_uint2(0u, 0u);
                                if (has_m24) {
                                    c1 = *reinterpret_cast<const uint2*>(sAp + x1_1 + kq_off);
                                    c3 = *reinterpret_cast<const uint2*>(sAp + x1_1 + kq_off + 16);
                                }
                                mma_m16n8k16_bf16(acc[1], c0.x, c1.x, c2.x, c3.x, b0_0, b1_0);
                                mma_m16n8k16_bf16(acc[1], c0.y, c1.y, c2.y, c3.y, b0_1, b1_1);
                            }
                        }
                    }
                }

                if constexpr (USE_PDL) {
                    if (trig_dep && is_last_tile && sl == trig_sl && !trig_fired) {
                        asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
                        trig_fired = true;
                    }
                }

                if (sl + 1 < n_slabs) {
                    if ((sl & 7) == 7) ws_cur = ws_nxt;
                    if constexpr (V5_MODE == 2) {
                        cur_slab_panel = nxt_slab_panel;
                        cur_meta       = nxt_meta;
                        cur_fmask      = finish_hybrid_flip_mask(nxt_rec, cur_meta, FLIPS, f_cur, lane, sFlip[warp], true, keep_l2);
                    } else if constexpr (V5_MODE == 1) {
                        cur_slab_lite  = nxt_slab_lite;
                        cur_meta       = nxt_meta;
                        cur_fmask      = finish_hybrid_flip_mask(nxt_rec, cur_meta, FLIPS, f_cur, lane, sFlip[warp], false, keep_l2);
                    } else {
                        cur_slab_lite  = nxt_slab_lite;
                        cur_fmask      = finish_slab_flip_mask_legacy(nxt_rec, FLIPS, f_cur, f_end, static_cast<uint32_t>(sl + 1), lane, sFlip[warp], keep_l2);
                    }
                }
            }
            if (blk + 1 < n_blks) __syncthreads();
        }

        const float scale = Ones[e];
        const int compact_base = FUSE_SITU ? TileCompactStart[t] : 0;

        #pragma unroll
        for (int im = 0; im < M_STEPS; ++im) {
            if (im == 1 && !has_m16) break;
            const int lm0 = im * 16 + g;
            const int r0 = row_base + lm0;
            const float o0 = mul_rn(acc[im][0], scale);
            const float o1 = mul_rn(acc[im][1], scale);
            [[maybe_unused]] const int lm1 = lm0 + 8;
            [[maybe_unused]] const int r1 = row_base + lm1;
            [[maybe_unused]] const float o2 = mul_rn(acc[im][2], scale);
            [[maybe_unused]] const float o3 = mul_rn(acc[im][3], scale);

            if constexpr (FUSE_SITU) {
                const int inter = N >> 1;  // 288
                const int c_warp = nb * 16 + warp * 4;
                const int c0 = compact_base + lm0;
                [[maybe_unused]] const int c1 = compact_base + lm1;

                uint16_t y0_b = 0u;
                if (lm0 < live_m) {
                    y0_b = situ_pair_bf16_exact(pack_bf16x2(o0, o1), __ldg(Gammas + r0));
                }
                uint32_t p0 = static_cast<uint32_t>(y0_b) | (__shfl_xor_sync(0xFFFFFFFFu, static_cast<uint32_t>(y0_b), 1) << 16);
                uint32_t p1 = __shfl_xor_sync(0xFFFFFFFFu, p0, 2);
                if (lt == 0 && lm0 < live_m) {
                    if (use_overlap) {
                        st_cg_u2(Out + static_cast<int64_t>(c0) * inter + c_warp, make_uint2(p0, p1));
                    } else {
                        *reinterpret_cast<uint2*>(Out + static_cast<int64_t>(c0) * inter + c_warp) = make_uint2(p0, p1);
                    }
                }
                if constexpr (TILE > 8) {
                    uint16_t y1_b = 0u;
                    if (lm1 < live_m) {
                        y1_b = situ_pair_bf16_exact(pack_bf16x2(o2, o3), __ldg(Gammas + r1));
                    }
                    uint32_t q0 = static_cast<uint32_t>(y1_b) | (__shfl_xor_sync(0xFFFFFFFFu, static_cast<uint32_t>(y1_b), 1) << 16);
                    uint32_t q1 = __shfl_xor_sync(0xFFFFFFFFu, q0, 2);
                    if (lt == 0 && lm1 < live_m) {
                        if (use_overlap) {
                            st_cg_u2(Out + static_cast<int64_t>(c1) * inter + c_warp, make_uint2(q0, q1));
                        } else {
                            *reinterpret_cast<uint2*>(Out + static_cast<int64_t>(c1) * inter + c_warp) = make_uint2(q0, q1);
                        }
                    }
                }
            } else {
                const int n0 = nb * BN + warp * 8 + 2 * lt;
                if (lm0 < live_m) {
                    *reinterpret_cast<uint32_t*>(Out + static_cast<int64_t>(r0) * N + n0) = pack_bf16x2(o0, o1);
                }
                if constexpr (TILE > 8) {
                    if (lm1 < live_m) {
                        *reinterpret_cast<uint32_t*>(Out + static_cast<int64_t>(r1) * N + n0) = pack_bf16x2(o2, o3);
                    }
                }
            }
        }

        if constexpr (FUSE_SITU) {
            if (use_overlap) {
                __syncthreads();
                if (tid == 0) {
                    w13_mark_tile_panel_done(t, n_active, n_panels, overlap_gen, TileW13Done, TileReady, OverlapMeta);
                }
            }
        }

        if (!is_last_tile && !use_overlap) __syncthreads();
    }

    if constexpr (USE_PDL) {
        if (trig_dep && !trig_fired) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }
}

// =========================================================================
// Kernel 2 (C48 v8): Unified W2 (N=4608, K=288, BN=128, 4 warps, SUB_N=4)
//   - TILE=8:    MIN_CTA probed 6->5->4, TILE=16: MIN_CTA=5, TILE=32: MIN_CTA=4
//   - pdl_wait_mode == 3 (v8 Overlap Consumer): prefetches 4 slabs + WS3 + SLAB_META & zeros sA_all
//     BEFORE polling TileReady[t] == overlap_gen (nanosleep 64), reads SiTU_Out via ld_cg_u4 (stale-L1-proof!),
//     and self-advances GPU-resident OverlapMeta[0] (CUDA-Graph safe, 0xFFFFFFFF wraparound safe!)
//   - Hot-Expert L2 Cache-Hint (`hot_l2_hint=1`): `ld.ca` when `Tiles[t+1] == e`, `ld.cs` otherwise!
// =========================================================================
template <
    int V5_MODE, int TILE, int MIN_CTA, int CTAS_PER_PANEL, bool USE_PDL,
    bool AS_ROW_MAJOR, bool FUSE_QUANT, int FUSE_COMBINE
>
__global__ __launch_bounds__(128, MIN_CTA)
void mx7_w2_c48_v8_kernel(
    const uint8_t* __restrict__ A,
    const uint8_t* __restrict__ AS,
    const uint8_t* __restrict__ SLABS,        // Compat/Legacy SLABS OR Panel PANEL_POOL
    const uint4* __restrict__ WS3,            // Compat/Legacy WS3 (unused in Panel)
    const uint16_t* __restrict__ FLIPS,       // Legacy FLIPS OR v5 SFLIPS
    const int32_t* __restrict__ F_OFF,        // Legacy F_OFF OR v5 S_OFF
    const uint32_t* __restrict__ SLAB_META,   // v5 SLAB_META
    const uint32_t* __restrict__ BMASKS,      // Compat BMASKS
    const int32_t* __restrict__ AUX_OFF,      // Compat D_OFF OR Panel PANEL_OFF
    const float* __restrict__ Ones,
    const int32_t* __restrict__ Tiles,
    const int32_t* __restrict__ TileCompactStart,
    __nv_bfloat16* __restrict__ Out,
    const int32_t* __restrict__ Src2Compact,
    const int32_t* __restrict__ NLivePtr,
    uint32_t* __restrict__ PanelDone,
    uint32_t* __restrict__ BufGen,
    __nv_bfloat16* __restrict__ SendBuf,
    uint32_t* __restrict__ TileReady,
    uint32_t* __restrict__ OverlapMeta,
    int N, int K, int MAX_CTAS, int T, uint32_t base_ws, int as_row_stride, int down_row_stride,
    int pdl_wait_mode, int trig_dep, int hot_l2_hint
) {
    (void)N;
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
    constexpr int H_DIM = 4608;
    constexpr int N_PANELS = 36;
    constexpr int P_GROUP = 6;
    constexpr int N_WAVES = N_PANELS / P_GROUP;
    constexpr int M_ROWS = TILE;
    constexpr int M_STEPS = (TILE <= 16) ? 1 : 2;
    constexpr int KG = 9;
    constexpr int SA_STRIDE = 304;  // 0-way LDS.64!
    constexpr bool IN_KERNEL_COMBINE = (FUSE_COMBINE >= 1 && FUSE_COMBINE <= 3);

    const int n_active = Tiles[2 * MAX_CTAS];
    const int bidx   = static_cast<int>(blockIdx.x);
    const int stride = static_cast<int>(gridDim.x);
    const bool fast_ws = (base_ws >= 8u && base_ws <= 238u);

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g    = lane >> 2;
    const int lt   = lane & 3;
    const int x0_0 = g * SA_STRIDE + 4 * lt;
    [[maybe_unused]] const int x0_1 = (g + 8) * SA_STRIDE + 4 * lt;
    [[maybe_unused]] const int x1_0 = (g + 16) * SA_STRIDE + 4 * lt;
    [[maybe_unused]] const int x1_1 = (g + 24) * SA_STRIDE + 4 * lt;

    alignas(16) __shared__ uint32_t sFlip[WARPS][32];
    alignas(16) __shared__ __nv_bfloat16 sA_all[M_ROWS * SA_STRIDE];
    __shared__ int32_t sIsLastPanel;
    __shared__ int32_t sIsVeryLastW2Cta;

    const bool use_overlap = USE_PDL && FUSE_QUANT && (pdl_wait_mode == 3) &&
        (TileReady != nullptr) && (OverlapMeta != nullptr);
    const uint32_t overlap_gen = (use_overlap && n_active > 0) ? read_overlap_gen(OverlapMeta) : 0u;

    auto finalize_w2_overlap_gen = [&]() {
        if constexpr (USE_PDL && FUSE_QUANT) {
            if (use_overlap) {
                __syncthreads();
                if (tid == 0) {
                    const uint32_t prev = atom_acq_rel_gpu_add(&OverlapMeta[2], 1u);
                    if (prev == static_cast<uint32_t>(stride - 1)) {
                        st_rel_gpu_u32(&OverlapMeta[2], 0u);
                        sIsVeryLastW2Cta = 1;
                    } else {
                        sIsVeryLastW2Cta = 0;
                    }
                }
                __syncthreads();
                if (sIsVeryLastW2Cta) {
                    if (overlap_gen == 0xFFFFFFFFu) {
                        for (int i = tid; i < 512; i += 128) {
                            st_rel_gpu_u32(&TileReady[i], 0u);
                        }
                        __threadfence();
                        __syncthreads();
                        if (tid == 0) {
                            st_rel_gpu_u32(&OverlapMeta[0], 0u);
                        }
                    } else {
                        if (tid == 0) {
                            st_rel_gpu_u32(&OverlapMeta[0], overlap_gen);
                        }
                    }
                }
            }
        }
    };

    [[maybe_unused]] auto zero_sendbuf_panel = [&](int nb) {
        int n_live_tok = T;
        if (NLivePtr != nullptr) {
            int nl = *NLivePtr;
            if (nl < 0) nl = 0;
            if (nl < n_live_tok) n_live_tok = nl;
        }
        // 16 threads cover 128 BF16 (8 BF16/thread = uint4) -> 8 tokens per iteration!
        const int col0 = nb * BN + (tid & 15) * 8;
        const uint4 z4 = make_uint4(0u, 0u, 0u, 0u);
        for (int m = (tid >> 4); m < n_live_tok; m += 8) {
            *reinterpret_cast<uint4*>(SendBuf + static_cast<int64_t>(m) * H_DIM + col0) = z4;
        }
    };

    if constexpr (FUSE_COMBINE == 1) {
        const int p_slot = bidx / CTAS_PER_PANEL;
        const int p_rank = bidx - p_slot * CTAS_PER_PANEL;
        if (n_active == 0) {
            if constexpr (USE_PDL) {
                if (pdl_wait_mode == 2 || pdl_wait_mode == 3) asm volatile("griddepcontrol.wait;" ::: "memory");
            }
            if (p_rank == 0) {
                for (int w = 0; w < N_WAVES; ++w) {
                    zero_sendbuf_panel(w * P_GROUP + p_slot);
                }
                __threadfence();
                __syncthreads();
            }
            if constexpr (USE_PDL) {
                if (trig_dep) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
            return;
        }
        const int active_in_panel = (n_active < CTAS_PER_PANEL) ? n_active : CTAS_PER_PANEL;
        if (p_rank >= active_in_panel) {
            if constexpr (USE_PDL) {
                if (pdl_wait_mode == 2 || pdl_wait_mode == 3) asm volatile("griddepcontrol.wait;" ::: "memory");
            }
            finalize_w2_overlap_gen();
            if constexpr (USE_PDL) {
                if (trig_dep) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
            return;
        }
    } else if constexpr (FUSE_COMBINE == 2) {
        if (n_active == 0) {
            if constexpr (USE_PDL) {
                if (pdl_wait_mode == 2 || pdl_wait_mode == 3) asm volatile("griddepcontrol.wait;" ::: "memory");
            }
            if (bidx < N_PANELS) {
                zero_sendbuf_panel(bidx);
                __threadfence();
                __syncthreads();
            }
            if constexpr (USE_PDL) {
                if (trig_dep) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
            return;
        }
        const int n_total = N_PANELS * n_active;
        if (bidx >= n_total) {
            if constexpr (USE_PDL) {
                if (pdl_wait_mode == 2 || pdl_wait_mode == 3) asm volatile("griddepcontrol.wait;" ::: "memory");
            }
            finalize_w2_overlap_gen();
            if constexpr (USE_PDL) {
                if (trig_dep) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
            return;
        }
    } else if constexpr (FUSE_COMBINE == 3) {
        if (n_active == 0) {
            if constexpr (USE_PDL) {
                if (pdl_wait_mode == 2 || pdl_wait_mode == 3) asm volatile("griddepcontrol.wait;" ::: "memory");
            }
            if (bidx < N_PANELS) {
                zero_sendbuf_panel(bidx);
                __threadfence();
                __syncthreads();
            }
            if constexpr (USE_PDL) {
                if (trig_dep) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
            return;
        }
        const int t_stride12 = stride / 12;
        if (!(n_active >= t_stride12 && (stride % 12) == 0)) {
            const int n_total = n_active * N_PANELS;
            if (bidx >= n_total) {
                if constexpr (USE_PDL) {
                    if (pdl_wait_mode == 2 || pdl_wait_mode == 3) asm volatile("griddepcontrol.wait;" ::: "memory");
                }
                finalize_w2_overlap_gen();
                if constexpr (USE_PDL) {
                    if (trig_dep) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
                }
                return;
            }
        }
    } else {
        if (n_active == 0) {
            if constexpr (USE_PDL) {
                if (pdl_wait_mode == 2 || pdl_wait_mode == 3) asm volatile("griddepcontrol.wait;" ::: "memory");
                if (trig_dep) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
            return;
        }
        const int n_total = n_active * N_PANELS;
        if (bidx >= n_total) {
            if constexpr (USE_PDL) {
                if (pdl_wait_mode == 2 || pdl_wait_mode == 3) asm volatile("griddepcontrol.wait;" ::: "memory");
            }
            finalize_w2_overlap_gen();
            if constexpr (USE_PDL) {
                if (trig_dep) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
            return;
        }
    }

    bool waited_pdl = false;
    bool trig_fired = false;

    auto run_tile = [&](int t, int nb, bool is_very_first, bool is_very_last,
                        int next_t, int next_nb, bool has_next_tile,
                        __nv_bfloat16* down_panel_buf) {
        const int64_t e = static_cast<int64_t>(Tiles[t]);
        const int end_row = Tiles[MAX_CTAS + t];
        const int row_base = t * TILE;
        int live_m = end_row - row_base;
        if (live_m > TILE) live_m = TILE;
        if (live_m < 0) live_m = 0;

        const bool keep_l2 = (hot_l2_hint != 0) && (t + 1 < n_active) && (Tiles[t + 1] == static_cast<int32_t>(e));

        const int64_t panel_uid = e * N_PANELS + nb;
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
        const int32_t f_end = (V5_MODE == 0) ? F_OFF[warp_uid + 1] : 0;
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

        if constexpr (USE_PDL) {
            if ((pdl_wait_mode == 2 && is_very_first) || use_overlap) {
                prefetch_w2_slabs();
            }
        }

        const int pad_limit = (live_m <= 8) ? 8
            : ((live_m <= 16) ? 16 : ((live_m <= 24) ? 24 : M_ROWS));
        if (live_m < pad_limit) {
            constexpr int U4_PER_ROW = 36;  // 288 BF16 / 8 BF16 per uint4
            const int zero_u4 = (pad_limit - live_m) * U4_PER_ROW;
            const uint4 z4 = make_uint4(0u, 0u, 0u, 0u);
            for (int zidx = tid; zidx < zero_u4; zidx += THREADS) {
                const int dlm = zidx / U4_PER_ROW;
                const int q4  = zidx - dlm * U4_PER_ROW;
                *reinterpret_cast<uint4*>(sA_all + (live_m + dlm) * SA_STRIDE + (q4 << 3)) = z4;
            }
        }

        if constexpr (USE_PDL) {
            if (use_overlap) {
                if (tid == 0) {
                    while (ld_acq_gpu_u32(&TileReady[t]) != overlap_gen) {
                        nanosleep_u32(64u);
                    }
                }
                __syncthreads();
            } else if (pdl_wait_mode == 2 && !waited_pdl) {
                asm volatile("griddepcontrol.wait;" ::: "memory");
                waited_pdl = true;
            }
        }
        if (live_m == 0) return;

        // Overlap 4-slab L2 prefetch across all 32 lanes with OTF quant-dequant (when not already prefetched)
        if constexpr (FUSE_QUANT) {
            if (!use_overlap) prefetch_w2_slabs();
        }

        const int compact_tile_base = (FUSE_QUANT || (FUSE_COMBINE != 0)) ? TileCompactStart[t] : 0;

        {
            if constexpr (FUSE_QUANT) {
                const __nv_bfloat16* situ_in = reinterpret_cast<const __nv_bfloat16*>(A);
                const int total_blks = live_m * KG;
                if constexpr (TILE == 8) {
                    // 1-shot OTF quant-dequant (total_blks <= 72 < 128 threads -> 0 loop!)
                    if (tid < total_blks) {
                        const int lm = tid / KG;
                        const int blk = tid - lm * KG;
                        const int crow = compact_tile_base + lm;
                        const uint4* bp = reinterpret_cast<const uint4*>(situ_in + static_cast<int64_t>(crow) * 288 + blk * 32);
                        uint4 v[4] = {
                            use_overlap ? ld_cg_u4(bp)     : ld_ca_u4(bp),
                            use_overlap ? ld_cg_u4(bp + 1) : ld_ca_u4(bp + 1),
                            use_overlap ? ld_cg_u4(bp + 2) : ld_ca_u4(bp + 2),
                            use_overlap ? ld_cg_u4(bp + 3) : ld_ca_u4(bp + 3)
                        };
                        quant_dequant_mxfp8_32_to_sA(v, sA_all + lm * SA_STRIDE + blk * 32);
                    }
                } else {
                    for (int bq = tid; bq < total_blks; bq += THREADS) {
                        const int lm = bq / KG;
                        const int blk = bq - lm * KG;
                        const int crow = compact_tile_base + lm;
                        const uint4* bp = reinterpret_cast<const uint4*>(situ_in + static_cast<int64_t>(crow) * 288 + blk * 32);
                        uint4 v[4] = {
                            use_overlap ? ld_cg_u4(bp)     : ld_ca_u4(bp),
                            use_overlap ? ld_cg_u4(bp + 1) : ld_ca_u4(bp + 1),
                            use_overlap ? ld_cg_u4(bp + 2) : ld_ca_u4(bp + 2),
                            use_overlap ? ld_cg_u4(bp + 3) : ld_ca_u4(bp + 3)
                        };
                        quant_dequant_mxfp8_32_to_sA(v, sA_all + lm * SA_STRIDE + blk * 32);
                    }
                }
            } else {
                // Vectorized 16-elem LDG.128 -> 2x STS.128 (4x fewer A/AS loads, exact 0-ULP!)
                constexpr int E16_PER_ROW = 18;  // 288 / 16
                const int total_e16 = live_m * E16_PER_ROW;
                for (int aidx = tid; aidx < total_e16; aidx += THREADS) {
                    const int lm = aidx / E16_PER_ROW;
                    const int q16 = aidx - lm * E16_PER_ROW;
                    const int64_t row = static_cast<int64_t>(row_base + lm);
                    const int kcol = q16 << 4;
                    const int kg_col = q16 >> 1;
                    const uint4 a16 = ld_ca_u4(A + row * 288 + kcol);
                    uint8_t as_b;
                    if constexpr (AS_ROW_MAJOR) {
                        as_b = AS[row * as_row_stride + kg_col];
                    } else {
                        as_b = AS[scale_offset_decode(row, kg_col, KG)];
                    }
                    const float a_sc = e8m0_to_f32(as_b);
                    uint32_t b0, b1, b2, b3, b4, b5, b6, b7;
                    e4m3x4_scale_to_bf16x4(a16.x, a_sc, b0, b1);
                    e4m3x4_scale_to_bf16x4(a16.y, a_sc, b2, b3);
                    e4m3x4_scale_to_bf16x4(a16.z, a_sc, b4, b5);
                    e4m3x4_scale_to_bf16x4(a16.w, a_sc, b6, b7);
                    __nv_bfloat16* d = sA_all + lm * SA_STRIDE + kcol;
                    *reinterpret_cast<uint4*>(d)     = make_uint4(b0, b1, b2, b3);
                    *reinterpret_cast<uint4*>(d + 8) = make_uint4(b4, b5, b6, b7);
                }
            }
        }

        // Load ALL 9 SLAB_META words (1 reg) & ALL 36 WS3 words (2 regs) ONCE per tile!
        const uint32_t meta0 = (V5_MODE != 0 && lane < KG) ? ld_w_u32(SLAB_META + warp_uid * KG + lane, keep_l2) : 0u;
        const uint32_t ws0   = ld_w_u32(ws_u32_base + lane, keep_l2);
        const uint32_t ws1   = (lane < 4) ? ld_w_u32(ws_u32_base + 32 + lane, keep_l2) : 0u;

        uint32_t cur_meta = (V5_MODE != 0) ? __shfl_sync(0xFFFFFFFFu, meta0, 0) : 0u;
        const uint8_t* panel_step_ptr = slab_base;
        SlabMX7Lite cur_slab_lite = {};
        SlabPanelV5 cur_slab_panel = {};
        uint32_t rec0 = 0u;

        if constexpr (V5_MODE == 0) {
            rec0 = (f_cur + lane < f_end) ? static_cast<uint32_t>(ld_w_u16(FLIPS + f_cur + lane, keep_l2)) : 0xFFFFu;
            cur_slab_lite = load_slab_mx7_lite(slab_base, lane, keep_l2);
        } else if constexpr (V5_MODE == 1) {
            cur_slab_lite = load_slab_mx7_lite(slab_base, lane, keep_l2);
            rec0 = issue_hybrid_flip_load(cur_meta, FLIPS, f_cur, BMASKS, d_base, lane, false, keep_l2);
        } else {
            const bool is_r8_0 = (cur_meta & META_DENSE_BIT) != 0u;
            cur_slab_panel = load_slab_panel_v5(
                panel_step_ptr + warp * (is_r8_0 ? MX7_RAW8_SLAB_BYTES : MX7_SLAB_BYTES), lane, is_r8_0, keep_l2
            );
            rec0 = issue_hybrid_flip_load(cur_meta, FLIPS, f_cur, nullptr, 0, lane, true, keep_l2);
        }

        uint32_t cur_fmask = 0u;
        if constexpr (V5_MODE == 0) {
            cur_fmask = finish_slab_flip_mask_legacy(rec0, FLIPS, f_cur, f_end, 0u, lane, sFlip[warp], keep_l2);
        } else {
            cur_fmask = finish_hybrid_flip_mask(rec0, cur_meta, FLIPS, f_cur, lane, sFlip[warp], V5_MODE == 2, keep_l2);
        }
        __syncthreads();

        float acc[M_STEPS][SUB_N][4];
        #pragma unroll
        for (int im = 0; im < M_STEPS; ++im) {
            #pragma unroll
            for (int sub = 0; sub < SUB_N; ++sub) {
                acc[im][sub][0] = 0.0f; acc[im][sub][1] = 0.0f;
                acc[im][sub][2] = 0.0f; acc[im][sub][3] = 0.0f;
            }
        }
        [[maybe_unused]] const bool has_m8  = (TILE > 8)  && (live_m > 8);
        [[maybe_unused]] const bool has_m16 = (TILE > 16) && (live_m > 16);
        [[maybe_unused]] const bool has_m24 = (TILE > 16) && (live_m > 24);

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
                    nxt_slab_lite = load_slab_mx7_lite(slab_base + static_cast<int64_t>(nkg) * MX7_SLAB_BYTES, lane, keep_l2);
                    nxt_rec = (f_cur + lane < f_end) ? static_cast<uint32_t>(ld_w_u16(FLIPS + f_cur + lane, keep_l2)) : 0xFFFFu;
                } else if constexpr (V5_MODE == 1) {
                    nxt_meta = __shfl_sync(0xFFFFFFFFu, meta0, nkg);
                    nxt_slab_lite = load_slab_mx7_lite(slab_base + static_cast<int64_t>(nkg) * MX7_SLAB_BYTES, lane, keep_l2);
                    nxt_rec = issue_hybrid_flip_load(nxt_meta, FLIPS, f_cur, BMASKS, d_base, lane, false, keep_l2);
                } else {
                    nxt_meta = __shfl_sync(0xFFFFFFFFu, meta0, nkg);
                    const bool nxt_r8 = (nxt_meta & META_DENSE_BIT) != 0u;
                    nxt_slab_panel = load_slab_panel_v5(
                        panel_step_ptr + warp * (nxt_r8 ? MX7_RAW8_SLAB_BYTES : MX7_SLAB_BYTES), lane, nxt_r8, keep_l2
                    );
                    nxt_rec = issue_hybrid_flip_load(nxt_meta, FLIPS, f_cur, nullptr, 0, lane, true, keep_l2);
                }
            } else if (has_next_tile) {
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

            // Paired 0-way bank-conflict-free LDS.64 loads (reused across all 4 SUB_N!)
            const int k0 = kg * 32;
            const uint2 a0 = *reinterpret_cast<const uint2*>(sA_all + x0_0 + k0);
            const uint2 a2 = *reinterpret_cast<const uint2*>(sA_all + x0_0 + k0 + 16);
            [[maybe_unused]] uint2 a1 = make_uint2(0u, 0u), a3 = make_uint2(0u, 0u);
            if constexpr (TILE > 8) {
                if (has_m8) {
                    a1 = *reinterpret_cast<const uint2*>(sA_all + x0_1 + k0);
                    a3 = *reinterpret_cast<const uint2*>(sA_all + x0_1 + k0 + 16);
                }
            }
            [[maybe_unused]] uint2 c0 = make_uint2(0u, 0u), c1 = make_uint2(0u, 0u);
            [[maybe_unused]] uint2 c2 = make_uint2(0u, 0u), c3 = make_uint2(0u, 0u);
            if constexpr (M_STEPS == 2) {
                if (has_m16) {
                    c0 = *reinterpret_cast<const uint2*>(sA_all + x1_0 + k0);
                    c2 = *reinterpret_cast<const uint2*>(sA_all + x1_0 + k0 + 16);
                    if (has_m24) {
                        c1 = *reinterpret_cast<const uint2*>(sA_all + x1_1 + k0);
                        c3 = *reinterpret_cast<const uint2*>(sA_all + x1_1 + k0 + 16);
                    }
                }
            }

            const bool slab_all_normal = (V5_MODE != 0) && fast_ws && ((cur_meta & META_ALL_NORMAL_BIT) != 0u);
            const bool slab_no_nan     = (V5_MODE != 0) && fast_ws && ((cur_meta & META_HAS_NAN_BIT) == 0u);
            const bool ws_in_first32   = (kg < 8);
            const int  ws_kg_base      = (kg & 7) << 2;

            auto exec_w2_mma = [&](float (&acc0)[4], float (&acc1)[4],
                                   uint32_t b0_0, uint32_t b1_0, uint32_t b0_1, uint32_t b1_1) {
                (void)acc1;
                if constexpr (TILE == 8) {
                    mma_m16n8k16_bf16(acc0, a0.x, 0u, a2.x, 0u, b0_0, b1_0);
                    mma_m16n8k16_bf16(acc0, a0.y, 0u, a2.y, 0u, b0_1, b1_1);
                } else {
                    mma_m16n8k16_bf16(acc0, a0.x, a1.x, a2.x, a3.x, b0_0, b1_0);
                    mma_m16n8k16_bf16(acc0, a0.y, a1.y, a2.y, a3.y, b0_1, b1_1);
                }
                if constexpr (M_STEPS == 2) {
                    if (has_m16) {
                        mma_m16n8k16_bf16(acc1, c0.x, c1.x, c2.x, c3.x, b0_0, b1_0);
                        mma_m16n8k16_bf16(acc1, c0.y, c1.y, c2.y, c3.y, b0_1, b1_1);
                    }
                }
            };

            if constexpr (V5_MODE == 2) {
                if (cur_is_raw8) {
                    const uint32_t raw_arr[8] = {
                        cur_slab_panel.a.x, cur_slab_panel.a.y, cur_slab_panel.a.z, cur_slab_panel.a.w,
                        cur_slab_panel.b.x, cur_slab_panel.b.y, cur_slab_panel.b.z, cur_slab_panel.b.w
                    };
                    #pragma unroll
                    for (int sub = 0; sub < SUB_N; ++sub) {
                        const uint32_t ws_word = ws_in_first32
                            ? __shfl_sync(0xFFFFFFFFu, ws0, ws_kg_base | sub)
                            : __shfl_sync(0xFFFFFFFFu, ws1, sub);
                        const uint32_t ws_byte = base_ws + ((ws_word >> (g * 3)) & 7u);
                        const uint32_t ws_off2 = ((ws_byte - 7u) << 7) * 0x00010001u;
                        const float    w_sc    = __uint_as_float(ws_byte << 23);

                        uint32_t b0_0, b1_0, b0_1, b1_1;
                        decode_raw8_step_to_bf16x4(raw_arr[sub << 1],       ws_off2, w_sc, slab_all_normal, slab_no_nan, b0_0, b1_0);
                        decode_raw8_step_to_bf16x4(raw_arr[(sub << 1) | 1], ws_off2, w_sc, slab_all_normal, slab_no_nan, b0_1, b1_1);
                        exec_w2_mma(acc[0][sub], acc[(M_STEPS == 2) ? 1 : 0][sub], b0_0, b1_0, b0_1, b1_1);
                    }
                } else {
                    const uint32_t sm_arr[8] = {
                        cur_slab_panel.a.x & 0xFFFFu, cur_slab_panel.a.x >> 16,
                        cur_slab_panel.a.y & 0xFFFFu, cur_slab_panel.a.y >> 16,
                        cur_slab_panel.a.z & 0xFFFFu, cur_slab_panel.a.z >> 16,
                        cur_slab_panel.a.w & 0xFFFFu, cur_slab_panel.a.w >> 16
                    };
                    const uint32_t e21_arr[8] = {
                        cur_slab_panel.b.x & 0xFFu, (cur_slab_panel.b.x >> 8) & 0xFFu,
                        (cur_slab_panel.b.x >> 16) & 0xFFu, (cur_slab_panel.b.x >> 24) & 0xFFu,
                        cur_slab_panel.b.y & 0xFFu, (cur_slab_panel.b.y >> 8) & 0xFFu,
                        (cur_slab_panel.b.y >> 16) & 0xFFu, (cur_slab_panel.b.y >> 24) & 0xFFu
                    };
                    const uint32_t e0_reg = cur_slab_panel.b.z;
                    #pragma unroll
                    for (int sub = 0; sub < SUB_N; ++sub) {
                        const uint32_t ws_word = ws_in_first32
                            ? __shfl_sync(0xFFFFFFFFu, ws0, ws_kg_base | sub)
                            : __shfl_sync(0xFFFFFFFFu, ws1, sub);
                        const uint32_t ws_byte = base_ws + ((ws_word >> (g * 3)) & 7u);
                        const uint32_t ws_off2 = ((ws_byte - 7u) << 7) * 0x00010001u;
                        const float    w_sc    = __uint_as_float(ws_byte << 23);

                        const int s0 = sub << 1;
                        const int s1 = s0 | 1;
                        uint32_t b0_0, b1_0, b0_1, b1_1;
                        decode_mx7_step_to_bf16x4(
                            sm_arr[s0], e21_arr[s0], (e0_reg >> (s0 * 4)) & 0xFu, (cur_fmask >> (s0 * 4)) & 0xFu,
                            ws_off2, w_sc, slab_all_normal, slab_no_nan, b0_0, b1_0);
                        decode_mx7_step_to_bf16x4(
                            sm_arr[s1], e21_arr[s1], (e0_reg >> (s1 * 4)) & 0xFu, (cur_fmask >> (s1 * 4)) & 0xFu,
                            ws_off2, w_sc, slab_all_normal, slab_no_nan, b0_1, b1_1);
                        exec_w2_mma(acc[0][sub], acc[(M_STEPS == 2) ? 1 : 0][sub], b0_0, b1_0, b0_1, b1_1);
                    }
                }
            } else {
                const uint32_t sm_arr[8] = {
                    cur_slab_lite.sm.x & 0xFFFFu, cur_slab_lite.sm.x >> 16,
                    cur_slab_lite.sm.y & 0xFFFFu, cur_slab_lite.sm.y >> 16,
                    cur_slab_lite.sm.z & 0xFFFFu, cur_slab_lite.sm.z >> 16,
                    cur_slab_lite.sm.w & 0xFFFFu, cur_slab_lite.sm.w >> 16
                };
                const uint32_t e21_arr[8] = {
                    cur_slab_lite.e21.x & 0xFFu, (cur_slab_lite.e21.x >> 8) & 0xFFu,
                    (cur_slab_lite.e21.x >> 16) & 0xFFu, (cur_slab_lite.e21.x >> 24) & 0xFFu,
                    cur_slab_lite.e21.y & 0xFFu, (cur_slab_lite.e21.y >> 8) & 0xFFu,
                    (cur_slab_lite.e21.y >> 16) & 0xFFu, (cur_slab_lite.e21.y >> 24) & 0xFFu
                };
                #pragma unroll
                for (int sub = 0; sub < SUB_N; ++sub) {
                    const uint32_t ws_word = ws_in_first32
                        ? __shfl_sync(0xFFFFFFFFu, ws0, ws_kg_base | sub)
                        : __shfl_sync(0xFFFFFFFFu, ws1, sub);
                    const uint32_t ws_byte = base_ws + ((ws_word >> (g * 3)) & 7u);
                    const uint32_t ws_off2 = ((ws_byte - 7u) << 7) * 0x00010001u;
                    const float    w_sc    = __uint_as_float(ws_byte << 23);

                    const int s0 = sub << 1;
                    const int s1 = s0 | 1;
                    uint32_t b0_0, b1_0, b0_1, b1_1;
                    if constexpr (V5_MODE == 0) {
                        e4m3x4_scale_to_bf16x4(
                            build_e4_pack_compat_fast(sm_arr[s0], e21_arr[s0], (cur_slab_lite.e0 >> (s0 * 4)) & 0xFu, (cur_fmask >> (s0 * 4)) & 0xFu),
                            w_sc, b0_0, b1_0);
                        e4m3x4_scale_to_bf16x4(
                            build_e4_pack_compat_fast(sm_arr[s1], e21_arr[s1], (cur_slab_lite.e0 >> (s1 * 4)) & 0xFu, (cur_fmask >> (s1 * 4)) & 0xFu),
                            w_sc, b0_1, b1_1);
                    } else {
                        decode_mx7_step_to_bf16x4(
                            sm_arr[s0], e21_arr[s0], (cur_slab_lite.e0 >> (s0 * 4)) & 0xFu, (cur_fmask >> (s0 * 4)) & 0xFu,
                            ws_off2, w_sc, slab_all_normal, slab_no_nan, b0_0, b1_0);
                        decode_mx7_step_to_bf16x4(
                            sm_arr[s1], e21_arr[s1], (cur_slab_lite.e0 >> (s1 * 4)) & 0xFu, (cur_fmask >> (s1 * 4)) & 0xFu,
                            ws_off2, w_sc, slab_all_normal, slab_no_nan, b0_1, b1_1);
                    }
                    exec_w2_mma(acc[0][sub], acc[(M_STEPS == 2) ? 1 : 0][sub], b0_0, b1_0, b0_1, b1_1);
                }
            }

            if constexpr (USE_PDL && !IN_KERNEL_COMBINE) {
                if (trig_dep && is_very_last && kg == KG - 2) {
                    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
                    trig_fired = true;
                }
            }

            if (kg + 1 < KG) {
                if constexpr (V5_MODE == 2) {
                    cur_slab_panel = nxt_slab_panel;
                    cur_meta       = nxt_meta;
                    cur_fmask      = finish_hybrid_flip_mask(nxt_rec, cur_meta, FLIPS, f_cur, lane, sFlip[warp], true, keep_l2);
                } else if constexpr (V5_MODE == 1) {
                    cur_slab_lite  = nxt_slab_lite;
                    cur_meta       = nxt_meta;
                    cur_fmask      = finish_hybrid_flip_mask(nxt_rec, cur_meta, FLIPS, f_cur, lane, sFlip[warp], false, keep_l2);
                } else {
                    cur_slab_lite  = nxt_slab_lite;
                    cur_fmask      = finish_slab_flip_mask_legacy(nxt_rec, FLIPS, f_cur, f_end, static_cast<uint32_t>(kg + 1), lane, sFlip[warp], keep_l2);
                }
            }
        }

        const float scale = Ones[e];
        const int col_u4_local = warp * 32 + lt * 8;
        #pragma unroll
        for (int im = 0; im < M_STEPS; ++im) {
            if (im == 1 && !has_m16) break;
            const int lm0 = im * 16 + g;
            const int r0 = row_base + lm0;
            const int c0 = compact_tile_base + lm0;
            [[maybe_unused]] const int lm1 = lm0 + 8;
            [[maybe_unused]] const int r1 = row_base + lm1;
            [[maybe_unused]] const int c1 = compact_tile_base + lm1;

            // Pack all 4 sub-panels for row lm0 -> 3-SHFL 4x4 transpose -> 64B-coalesced uint4 store!
            uint32_t pk0[4];
            #pragma unroll
            for (int sub = 0; sub < SUB_N; ++sub) {
                pk0[sub] = pack_bf16x2(mul_rn(acc[im][sub][0], scale), mul_rn(acc[im][sub][1], scale));
            }
            const uint4 v0 = transpose_4x4_bf16x2_u4(pk0[0], pk0[1], pk0[2], pk0[3], lt);
            if (lm0 < live_m) {
                if constexpr (IN_KERNEL_COMBINE) {
                    st_cg_u4(down_panel_buf + static_cast<int64_t>(c0) * BN + col_u4_local, v0);
                } else if constexpr (FUSE_COMBINE == 4) {
                    st_cg_u4(Out + static_cast<int64_t>(c0) * H_DIM + nb * BN + col_u4_local, v0);
                } else {
                    *reinterpret_cast<uint4*>(Out + static_cast<int64_t>(r0) * H_DIM + nb * BN + col_u4_local) = v0;
                }
            }

            if constexpr (TILE > 8) {
                uint32_t pk1[4];
                #pragma unroll
                for (int sub = 0; sub < SUB_N; ++sub) {
                    pk1[sub] = pack_bf16x2(mul_rn(acc[im][sub][2], scale), mul_rn(acc[im][sub][3], scale));
                }
                const uint4 v1 = transpose_4x4_bf16x2_u4(pk1[0], pk1[1], pk1[2], pk1[3], lt);
                if (lm1 < live_m) {
                    if constexpr (IN_KERNEL_COMBINE) {
                        st_cg_u4(down_panel_buf + static_cast<int64_t>(c1) * BN + col_u4_local, v1);
                    } else if constexpr (FUSE_COMBINE == 4) {
                        st_cg_u4(Out + static_cast<int64_t>(c1) * H_DIM + nb * BN + col_u4_local, v1);
                    } else {
                        *reinterpret_cast<uint4*>(Out + static_cast<int64_t>(r1) * H_DIM + nb * BN + col_u4_local) = v1;
                    }
                }
            }
        }
    };

    // Vectorized 8-tokens/iter in-kernel combine using strict L1-bypass `ld_cg_u4` (16 B = 8 BF16)!
    [[maybe_unused]] auto combine_panel_to_sendbuf_u4 = [&](int nb, const __nv_bfloat16* down_panel_buf) {
        int n_live_tok = T;
        if (NLivePtr != nullptr) {
            int nl = *NLivePtr;
            if (nl < 0) nl = 0;
            if (nl < n_live_tok) n_live_tok = nl;
        }
        const int col_local = (tid & 15) * 8;
        const int col_glob  = nb * BN + col_local;
        const uint4 z4 = make_uint4(0u, 0u, 0u, 0u);
        for (int m = (tid >> 4); m < n_live_tok; m += 8) {
            const int32_t* s2c = Src2Compact + m * 8;
            int32_t r[8];
            #pragma unroll
            for (int s = 0; s < 8; ++s) r[s] = __ldg(s2c + s);

            // 2-deep pipelined ld_cg_u4 in exact slot 0..7 FP32 add_rn order (0-ULP!)
            uint4 v_cur = (r[0] >= 0) ? ld_cg_u4(down_panel_buf + static_cast<int64_t>(r[0]) * BN + col_local) : z4;
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
            float a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;
            #pragma unroll
            for (int s = 0; s < 8; ++s) {
                uint4 v_nxt = z4;
                if (s + 1 < 8 && r[s + 1] >= 0) {
                    v_nxt = ld_cg_u4(down_panel_buf + static_cast<int64_t>(r[s + 1]) * BN + col_local);
                }
                a0 = add_rn(a0, bf16_bits_to_f32(static_cast<uint16_t>(v_cur.x & 0xFFFFu)));
                a1 = add_rn(a1, bf16_bits_to_f32(static_cast<uint16_t>(v_cur.x >> 16)));
                a2 = add_rn(a2, bf16_bits_to_f32(static_cast<uint16_t>(v_cur.y & 0xFFFFu)));
                a3 = add_rn(a3, bf16_bits_to_f32(static_cast<uint16_t>(v_cur.y >> 16)));
                a4 = add_rn(a4, bf16_bits_to_f32(static_cast<uint16_t>(v_cur.z & 0xFFFFu)));
                a5 = add_rn(a5, bf16_bits_to_f32(static_cast<uint16_t>(v_cur.z >> 16)));
                a6 = add_rn(a6, bf16_bits_to_f32(static_cast<uint16_t>(v_cur.w & 0xFFFFu)));
                a7 = add_rn(a7, bf16_bits_to_f32(static_cast<uint16_t>(v_cur.w >> 16)));
                v_cur = v_nxt;
            }
            *reinterpret_cast<uint4*>(SendBuf + static_cast<int64_t>(m) * H_DIM + col_glob) = make_uint4(
                pack_bf16x2(a0, a1), pack_bf16x2(a2, a3),
                pack_bf16x2(a4, a5), pack_bf16x2(a6, a7)
            );
        }
    };

    if constexpr (FUSE_COMBINE == 1) {
        const int p_slot = bidx / CTAS_PER_PANEL;
        const int p_rank = bidx - p_slot * CTAS_PER_PANEL;
        const int active_in_panel = (n_active < CTAS_PER_PANEL) ? n_active : CTAS_PER_PANEL;

        for (int w = 0; w < N_WAVES; ++w) {
            const int nb = w * P_GROUP + p_slot;
            const int buf = w & 1;
            __nv_bfloat16* down_panel_buf = Out + static_cast<int64_t>((buf * P_GROUP + p_slot) * down_row_stride) * BN;

            if (w >= 2) {
                if (tid == 0) {
                    const uint32_t need = static_cast<uint32_t>(w >> 1);
                    while (ld_acq_gpu_u32(&BufGen[p_slot * 2 + buf]) < need) {}
                }
                __syncthreads();
            }

            for (int t = p_rank; t < n_active; t += CTAS_PER_PANEL) {
                const bool is_very_first = (w == 0 && t == p_rank);
                const bool is_last_in_wave = (t + CTAS_PER_PANEL >= n_active);
                const bool is_very_last = (w == N_WAVES - 1 && is_last_in_wave);
                const int next_t = is_last_in_wave ? p_rank : (t + CTAS_PER_PANEL);
                const int next_nb = is_last_in_wave ? ((w + 1) * P_GROUP + p_slot) : nb;
                run_tile(t, nb, is_very_first, false, next_t, next_nb, !is_very_last, down_panel_buf);
                if (!is_last_in_wave) __syncthreads();
            }

            __threadfence();
            __syncthreads();
            if (active_in_panel == 1) {
                if (tid == 0) sIsLastPanel = 1;
            } else {
                if (tid == 0) {
                    uint32_t prev = atom_acq_rel_gpu_add(&PanelDone[nb], 1u);
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
                if (tid == 0) {
                    const uint32_t next_gen = (w < N_WAVES - 2) ? static_cast<uint32_t>((w >> 1) + 1) : 0u;
                    st_rel_gpu_u32(&BufGen[p_slot * 2 + buf], next_gen);
                }
            }
            if (w + 1 < N_WAVES) __syncthreads();
        }
    } else if constexpr (FUSE_COMBINE == 2) {
        const int n_total = N_PANELS * n_active;
        uint32_t cur_nb_tiles = 0u;
        for (int idx = bidx; idx < n_total; idx += stride) {
            const int nb = idx / n_active;
            const int t  = idx - nb * n_active;
            const bool is_very_first = (idx == bidx);
            const int nidx = idx + stride;
            const bool has_next = (nidx < n_total);
            const int next_nb = has_next ? (nidx / n_active) : 0;
            const int next_t  = has_next ? (nidx - next_nb * n_active) : 0;
            __nv_bfloat16* down_panel_buf = Out + static_cast<int64_t>(nb) * down_row_stride * BN;

            run_tile(t, nb, is_very_first, false, next_t, next_nb, has_next, down_panel_buf);
            cur_nb_tiles += 1u;

            if (!has_next || next_nb != nb) {
                __threadfence();
                __syncthreads();
                if (tid == 0) {
                    if (cur_nb_tiles == static_cast<uint32_t>(n_active)) {
                        sIsLastPanel = 1;
                    } else {
                        uint32_t prev = atom_acq_rel_gpu_add(&PanelDone[nb], cur_nb_tiles);
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
        }
    } else if constexpr (FUSE_COMBINE == 3) {
        // v7 GPU-Gated Dynamic Slice-Expert-Major L2-Tiled Combine (`P in {6, 12}` + Flat Expert-Major):
        //   - Works cleanly for ALL `stride in {192, 240, 288}` (divisible by 12!)
        //   - Zero host sync, 100% CUDA-Graph safe, guaranteed `PanelDone[0..35] == 0` at exit!
        const int t_stride6  = stride / 6;
        const int t_stride12 = stride / 12;

        if (n_active >= t_stride12 && (stride % 12) == 0) {
            const int slice_p  = (n_active >= t_stride6) ? 6 : 12;
            const int n_slices = N_PANELS / slice_p;   // 6 or 3
            const int t_stride = stride / slice_p;     // guaranteed <= n_active!
            const int t_rank   = bidx / slice_p;
            const int lp       = bidx - t_rank * slice_p;

            for (int s = 0; s < n_slices; ++s) {
                const int nb = s * slice_p + lp;
                __nv_bfloat16* down_panel_buf = Out + static_cast<int64_t>(nb) * down_row_stride * BN;

                for (int t = t_rank; t < n_active; t += t_stride) {
                    const bool is_very_first = (s == 0 && t == t_rank);
                    const bool is_last_in_slice = (t + t_stride >= n_active);
                    const bool is_very_last = (s == n_slices - 1 && is_last_in_slice);
                    const int next_t  = is_last_in_slice ? t_rank : (t + t_stride);
                    const int next_nb = is_last_in_slice ? ((s + 1) * slice_p + lp) : nb;
                    run_tile(t, nb, is_very_first, false, next_t, next_nb, !is_very_last, down_panel_buf);
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
            // 1 <= n_active < stride/12: Flat Expert-Major fallback (all active CTAs satisfy bidx < n_total)
            const int n_total = n_active * N_PANELS;
            for (int idx = bidx; idx < n_total; idx += stride) {
                const int t  = idx / N_PANELS;
                const int nb = idx - t * N_PANELS;
                const bool is_very_first = (idx == bidx);
                const bool is_very_last  = (idx + stride >= n_total);
                const int nidx = idx + stride;
                const int next_t  = is_very_last ? 0 : (nidx / N_PANELS);
                const int next_nb = is_very_last ? 0 : (nidx - next_t * N_PANELS);
                __nv_bfloat16* down_panel_buf = Out + static_cast<int64_t>(nb) * down_row_stride * BN;

                run_tile(t, nb, is_very_first, false, next_t, next_nb, !is_very_last, down_panel_buf);

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
        // FC == 0 (sparse Out[r0, 4608]) and FC == 4 (COMPACT_OUT Out[c0, 4608]):
        // 100% pure Expert-Major schedule (0 fences, 36x L2 SiTU reuse, L2-hot <= 4.72 MiB @ M=64!)
        const int n_total = n_active * N_PANELS;
        for (int idx = bidx; idx < n_total; idx += stride) {
            const int t  = idx / N_PANELS;
            const int nb = idx - t * N_PANELS;
            const bool is_very_first = (idx == bidx);
            const bool is_very_last  = (idx + stride >= n_total);
            const int nidx = idx + stride;
            const int nt  = is_very_last ? 0 : (nidx / N_PANELS);
            const int nnb = is_very_last ? 0 : (nidx - nt * N_PANELS);
            run_tile(t, nb, is_very_first, is_very_last, nt, nnb, !is_very_last, nullptr);
            if (!is_very_last) __syncthreads();
        }
    }

    finalize_w2_overlap_gen();

    if constexpr (USE_PDL) {
        if (trig_dep && !trig_fired) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }
}

// =========================================================================
// Kernel 3 (C48/C32 v7): 0-ULP PDL + NLivePtr Vectorized SM121a Combine (H=4608, topk=8)
//   - T <= 64: Grid = (6, T), Block = (96, 1, 1) -> 96 * 8 = 768 BF16/CTA = exact 1/6 of H=4608!
//       * Launches 102..384 CTAs across T=17..64 (fills GB10's 48 SMs cleanly!)
//   - T > 64:  Grid = (3, T), Block = (192, 1, 1) -> 192 * 8 = 1,536 BF16/CTA = exact 1/3 of H=4608!
//   - Exact slot 0..7 FP32 `add_rn` order (including +0.0f on r<0 for -0.0f parity -> nd == 0!)
// =========================================================================
template <int BLOCK_THREADS, int COLS_PER_CTA, int MIN_CTA_C, bool PDL_WAIT, bool TRIG_DEP, bool HAS_NLIVE>
__global__ __launch_bounds__(BLOCK_THREADS, MIN_CTA_C)
void mx7_combine_c48_v7_kernel(
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

// =========================================================================
// Launch & Dispatch Helpers
// =========================================================================

template <typename KernelT, typename... Args>
static inline cudaError_t launch_mx7_c32_v5(
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
static int launch_mx7_w13_c32_impl(
    const void* A, const void* AS, const void* Row2Tok,
    const void* SLABS, const void* WS3,
    const void* FLIPS, const void* F_OFF,
    const void* SLAB_META, const void* BMASKS, const void* AUX_OFF,
    const void* Ones, const void* Tiles,
    const void* TileCompactStart, const void* Gammas, void* Out,
    void* TileW13Done, void* TileReady, void* OverlapMeta,
    int N, int K, int MAX_CTAS, int TILE, uint32_t base_ws,
    int zero_scatter, int as_row_major, int fuse_situ,
    int pdl_wait_mode, int trig_dep, int overlap_w13_w2, int hot_l2_hint, uintptr_t stream_ptr
) {
    if ((N % 32) != 0 || (K % 256) != 0 || (K / 128) > 63) return -3;
    if (overlap_w13_w2 != 0 && MAX_CTAS > 512) return -6;
    const int total_tiles = (N / 32) * MAX_CTAS;
    if (total_tiles <= 0) return 0;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    const int persist_cap = (TILE == 8) ? (48 * MX7_W13_T8_MIN_CTA) : 240;
    dim3 grid((total_tiles < persist_cap) ? total_tiles : persist_cap, 1, 1);
    dim3 block(128, 1, 1);

    const bool zs = (zero_scatter != 0) && (Row2Tok != nullptr);
    const bool rm = zs || (as_row_major != 0);
    const bool fs = (fuse_situ != 0);
    const bool use_pdl = (pdl_wait_mode != 0) || (trig_dep != 0);
    const int eff_overlap = (overlap_w13_w2 != 0 && fs &&
        TileW13Done != nullptr && TileReady != nullptr && OverlapMeta != nullptr) ? 1 : 0;

    #define C13_RUN(TV, MC, PDL_B, Z, R, F) return static_cast<int>(launch_mx7_c32_v5( \
        mx7_w13_c48_v8_kernel<V5_MODE, TV, MC, PDL_B, Z, R, F>, \
        grid, block, stream, PDL_B, \
        static_cast<const uint8_t*>(A), static_cast<const uint8_t*>(AS), \
        static_cast<const int32_t*>(Row2Tok), static_cast<const uint8_t*>(SLABS), \
        static_cast<const uint4*>(WS3), static_cast<const uint16_t*>(FLIPS), \
        static_cast<const int32_t*>(F_OFF), static_cast<const uint32_t*>(SLAB_META), \
        static_cast<const uint32_t*>(BMASKS), static_cast<const int32_t*>(AUX_OFF), \
        static_cast<const float*>(Ones), static_cast<const int32_t*>(Tiles), \
        static_cast<const int32_t*>(TileCompactStart), static_cast<const float*>(Gammas), \
        static_cast<__nv_bfloat16*>(Out), \
        static_cast<uint32_t*>(TileW13Done), static_cast<uint32_t*>(TileReady), \
        static_cast<uint32_t*>(OverlapMeta), \
        N, K, MAX_CTAS, base_ws, pdl_wait_mode, trig_dep, eff_overlap, hot_l2_hint))

    #define C13(TV, MC, Z, R, F) do { \
        if (use_pdl) { C13_RUN(TV, MC, true,  Z, R, F); } \
        else         { C13_RUN(TV, MC, false, Z, R, F); } \
    } while (0)

    if (TILE == 8) {
        if      (zs && fs)          { C13(8, MX7_W13_T8_MIN_CTA, true,  true,  true);  }
        else if (zs && !fs)         { C13(8, MX7_W13_T8_MIN_CTA, true,  true,  false); }
        else if (!zs && rm && fs)   { C13(8, MX7_W13_T8_MIN_CTA, false, true,  true);  }
        else if (!zs && rm && !fs)  { C13(8, MX7_W13_T8_MIN_CTA, false, true,  false); }
        else if (!zs && !rm && fs)  { C13(8, MX7_W13_T8_MIN_CTA, false, false, true);  }
        else                        { C13(8, MX7_W13_T8_MIN_CTA, false, false, false); }
    } else if (TILE == 16) {
        if      (zs && fs)          { C13(16, 5, true,  true,  true);  }
        else if (zs && !fs)         { C13(16, 5, true,  true,  false); }
        else if (!zs && rm && fs)   { C13(16, 5, false, true,  true);  }
        else if (!zs && rm && !fs)  { C13(16, 5, false, true,  false); }
        else if (!zs && !rm && fs)  { C13(16, 5, false, false, true);  }
        else                        { C13(16, 5, false, false, false); }
    } else if (TILE == 32) {
        if      (zs && fs)          { C13(32, 5, true,  true,  true);  }
        else if (zs && !fs)         { C13(32, 5, true,  true,  false); }
        else if (!zs && rm && fs)   { C13(32, 5, false, true,  true);  }
        else if (!zs && rm && !fs)  { C13(32, 5, false, true,  false); }
        else if (!zs && !rm && fs)  { C13(32, 5, false, false, true);  }
        else                        { C13(32, 5, false, false, false); }
    }
    #undef C13
    #undef C13_RUN
    return -1;
}

template <int V5_MODE>
static int launch_mx7_w2_c32_impl(
    const void* A, const void* AS,
    const void* SLABS, const void* WS3,
    const void* FLIPS, const void* F_OFF,
    const void* SLAB_META, const void* BMASKS, const void* AUX_OFF,
    const void* Ones, const void* Tiles, const void* TileCompactStart,
    void* Out,
    const void* Src2Compact, const void* NLivePtr,
    void* PanelDone, void* BufGen, void* SendBuf,
    void* TileReady, void* OverlapMeta,
    int N, int K, int MAX_CTAS, int TILE, int T, uint32_t base_ws,
    int as_row_major, int as_row_stride, int down_row_stride,
    int fuse_quant, int fuse_combine,
    int pdl_wait_mode, int trig_dep, int overlap_w13_w2, int hot_l2_hint, uintptr_t stream_ptr
) {
    if (N != 4608 || K != 288) return -2;
    if (fuse_combine < 0 || fuse_combine > 4) return -4;
    if (fuse_combine >= 1 && fuse_combine <= 3 && T <= 0) return -5;
    if (fuse_combine == 4 && TileCompactStart == nullptr) return -5;
    if (overlap_w13_w2 != 0 && MAX_CTAS > 512) return -6;
    const int total_tiles = (N / 128) * MAX_CTAS;
    if (total_tiles <= 0 && (fuse_combine == 0 || fuse_combine == 4)) return 0;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    const bool fq = (fuse_quant != 0);
    const bool rm = (!fq) && (as_row_major != 0);
    const bool use_overlap = (overlap_w13_w2 != 0) && fq && (TileReady != nullptr) && (OverlapMeta != nullptr);
    const int eff_pdl_wait = use_overlap ? 3 : pdl_wait_mode;
    const bool use_pdl = (eff_pdl_wait != 0) || (trig_dep != 0);
    const int eff_as_stride = (as_row_stride > 0) ? as_row_stride : 9;
    const int eff_down_stride = (down_row_stride > 0) ? down_row_stride : (T * 8);

    const int persist_cap = (TILE == 8) ? (48 * MX7_W2_T8_MIN_CTA) : 192;
    int grid_x;
    if (fuse_combine == 1 || fuse_combine == 3) {
        grid_x = persist_cap;
    } else if (fuse_combine == 2) {
        grid_x = (total_tiles < persist_cap) ? total_tiles : persist_cap;
        if (grid_x < 36) grid_x = 36;
    } else {
        grid_x = (total_tiles < persist_cap) ? total_tiles : persist_cap;
    }
    dim3 grid(grid_x, 1, 1);
    dim3 block(128, 1, 1);

    #define C2_RUN(TV, MC, CPP, PDL_B, R, Q, C) return static_cast<int>(launch_mx7_c32_v5( \
        mx7_w2_c48_v8_kernel<V5_MODE, TV, MC, CPP, PDL_B, R, Q, C>, \
        grid, block, stream, PDL_B, \
        static_cast<const uint8_t*>(A), static_cast<const uint8_t*>(AS), \
        static_cast<const uint8_t*>(SLABS), static_cast<const uint4*>(WS3), \
        static_cast<const uint16_t*>(FLIPS), static_cast<const int32_t*>(F_OFF), \
        static_cast<const uint32_t*>(SLAB_META), static_cast<const uint32_t*>(BMASKS), \
        static_cast<const int32_t*>(AUX_OFF), static_cast<const float*>(Ones), \
        static_cast<const int32_t*>(Tiles), static_cast<const int32_t*>(TileCompactStart), \
        static_cast<__nv_bfloat16*>(Out), static_cast<const int32_t*>(Src2Compact), \
        static_cast<const int32_t*>(NLivePtr), static_cast<uint32_t*>(PanelDone), \
        static_cast<uint32_t*>(BufGen), static_cast<__nv_bfloat16*>(SendBuf), \
        static_cast<uint32_t*>(TileReady), static_cast<uint32_t*>(OverlapMeta), \
        N, K, MAX_CTAS, T, base_ws, eff_as_stride, eff_down_stride, eff_pdl_wait, trig_dep, hot_l2_hint))

    #define C2(TV, MC, CPP, R, Q, C) do { \
        if (use_pdl) { C2_RUN(TV, MC, CPP, true,  R, Q, C); } \
        else         { C2_RUN(TV, MC, CPP, false, R, Q, C); } \
    } while (0)

    #define DISPATCH_FC(TV, MC, CPP, FC_VAL) do { \
        if      (fq) { C2(TV, MC, CPP, false, true,  FC_VAL); } \
        else if (rm) { C2(TV, MC, CPP, true,  false, FC_VAL); } \
        else         { C2(TV, MC, CPP, false, false, FC_VAL); } \
    } while (0)

    constexpr int T8_CPP = 8 * MX7_W2_T8_MIN_CTA;
    if (TILE == 8) {
        if      (fuse_combine == 0) { DISPATCH_FC(8,  MX7_W2_T8_MIN_CTA, T8_CPP, 0); }
        else if (fuse_combine == 1) { DISPATCH_FC(8,  MX7_W2_T8_MIN_CTA, T8_CPP, 1); }
        else if (fuse_combine == 2) { DISPATCH_FC(8,  MX7_W2_T8_MIN_CTA, T8_CPP, 2); }
        else if (fuse_combine == 3) { DISPATCH_FC(8,  MX7_W2_T8_MIN_CTA, T8_CPP, 3); }
        else                        { DISPATCH_FC(8,  MX7_W2_T8_MIN_CTA, T8_CPP, 4); }
    } else if (TILE == 16) {
        if      (fuse_combine == 0) { DISPATCH_FC(16, 4, 32, 0); }
        else if (fuse_combine == 1) { DISPATCH_FC(16, 4, 32, 1); }
        else if (fuse_combine == 2) { DISPATCH_FC(16, 4, 32, 2); }
        else if (fuse_combine == 3) { DISPATCH_FC(16, 4, 32, 3); }
        else                        { DISPATCH_FC(16, 4, 32, 4); }
    } else if (TILE == 32) {
        if      (fuse_combine == 0) { DISPATCH_FC(32, 4, 32, 0); }
        else if (fuse_combine == 1) { DISPATCH_FC(32, 4, 32, 1); }
        else if (fuse_combine == 2) { DISPATCH_FC(32, 4, 32, 2); }
        else if (fuse_combine == 3) { DISPATCH_FC(32, 4, 32, 3); }
        else                        { DISPATCH_FC(32, 4, 32, 4); }
    }
    #undef DISPATCH_FC
    #undef C2
    #undef C2_RUN
    return -1;
}

}  // namespace

// =========================================================================
// Extern "C" ABI (v4/v5/v7 Drop-In + v8 Overlap & Hot-L2 Entrypoints)
// =========================================================================

extern "C" int mx7_c32_version() {
    return 8;
}

extern "C" int mx7_c48_min_cta() {
    return MX7_W13_T8_MIN_CTA * 10 + MX7_W2_T8_MIN_CTA;
}

extern "C" int mx7_combine_c48_v7_fwd(
    const void* Down,
    const void* Src2Row,
    const void* NLivePtr,
    void* Out,
    int T, int H, int pdl_wait, int trig_dep, uintptr_t stream_ptr
) {
    if (H != 4608) return -2;
    if (T <= 0) return 0;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    const bool fine = (T <= 64);
    dim3 grid(fine ? 6u : 3u, static_cast<unsigned>(T), 1u);
    dim3 block(fine ? 96u : 192u, 1u, 1u);
    const bool pw = (pdl_wait != 0);
    const bool td = (trig_dep != 0);
    const bool nl = (NLivePtr != nullptr);
    const bool use_pdl = pw || td;
    cudaError_t err = cudaSuccess;

    #define RUN_COMB(BT, CPC, MC, PW, TD, NL) err = launch_mx7_c32_v5( \
        mx7_combine_c48_v7_kernel<BT, CPC, MC, PW, TD, NL>, \
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

extern "C" int mx7_v2_count_flips(
    const void* W, void* counts, int proj, int E, int N, int K,
    int from_panel, int src_pb, int src_kb, uintptr_t stream_ptr
) {
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream_ptr);
    const int bn = (proj == 0) ? 32 : 64;
    const int warps = (proj == 0) ? 4 : 2;
    const int total_warps = E * (N / bn) * warps;
    if (proj == 0) {
        if (from_panel) mx7_c32_count_flips_kernel<0, true><<<total_warps, 32, 0, s>>>(
            static_cast<const uint8_t*>(W), static_cast<int32_t*>(counts), E, N, K, src_pb, src_kb);
        else mx7_c32_count_flips_kernel<0, false><<<total_warps, 32, 0, s>>>(
            static_cast<const uint8_t*>(W), static_cast<int32_t*>(counts), E, N, K, 0, 0);
    } else if (proj == 1) {
        if (from_panel) mx7_c32_count_flips_kernel<1, true><<<total_warps, 32, 0, s>>>(
            static_cast<const uint8_t*>(W), static_cast<int32_t*>(counts), E, N, K, src_pb, src_kb);
        else mx7_c32_count_flips_kernel<1, false><<<total_warps, 32, 0, s>>>(
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
        if (from_panel) mx7_c32_pack_kernel<0, true><<<total_warps, 32, 0, s>>>(
            static_cast<const uint8_t*>(W), static_cast<const uint8_t*>(WS),
            static_cast<uint8_t*>(SLABS), static_cast<uint4*>(WS3),
            static_cast<uint16_t*>(FLIPS), static_cast<const int32_t*>(F_OFF),
            E, N, K, base_ws, src_pb, src_kb);
        else mx7_c32_pack_kernel<0, false><<<total_warps, 32, 0, s>>>(
            static_cast<const uint8_t*>(W), static_cast<const uint8_t*>(WS),
            static_cast<uint8_t*>(SLABS), static_cast<uint4*>(WS3),
            static_cast<uint16_t*>(FLIPS), static_cast<const int32_t*>(F_OFF),
            E, N, K, base_ws, 0, 0);
    } else if (proj == 1) {
        if (from_panel) mx7_c32_pack_kernel<1, true><<<total_warps, 32, 0, s>>>(
            static_cast<const uint8_t*>(W), static_cast<const uint8_t*>(WS),
            static_cast<uint8_t*>(SLABS), static_cast<uint4*>(WS3),
            static_cast<uint16_t*>(FLIPS), static_cast<const int32_t*>(F_OFF),
            E, N, K, base_ws, src_pb, src_kb);
        else mx7_c32_pack_kernel<1, false><<<total_warps, 32, 0, s>>>(
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
    mx7_v5_count_hybrid_from_mx7_kernel<<<total_warps, 32, 0, s>>>(
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
    mx7_v5_build_hybrid_from_mx7_kernel<<<total_warps, 32, 0, s>>>(
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
    mx7_v5_panel_measure_from_mx7_kernel<<<total_panels, 128, 0, s>>>(
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
    mx7_v5_panel_build_from_mx7_kernel<<<total_panels, 128, 0, s>>>(
        static_cast<const uint8_t*>(SLABS), static_cast<const uint4*>(WS3),
        static_cast<const uint16_t*>(FLIPS), static_cast<const int32_t*>(F_OFF),
        static_cast<const int32_t*>(PANEL_OFF), static_cast<const int32_t*>(S_OFF),
        static_cast<uint8_t*>(PANEL_POOL), static_cast<uint32_t*>(SLAB_META),
        static_cast<uint16_t*>(SFLIPS),
        total_panels, n_slabs, hdr_blks, ws_u32, thr
    );
    return static_cast<int>(cudaGetLastError());
}

// v8 Overlap + Hot-L2 Entrypoints (Legacy / Compat-Hybrid / Panel-V5)
extern "C" int mx7_w13_c48_v8_fwd(
    const void* A, const void* AS, const void* Row2Tok,
    const void* SLABS, const void* WS3,
    const void* FLIPS, const void* F_OFF,
    const void* Ones, const void* Tiles,
    const void* TileCompactStart, const void* Gammas, void* Out,
    int N, int K, int MAX_CTAS, int TILE, uint32_t base_ws,
    int zero_scatter, int as_row_major, int fuse_situ,
    int pdl_wait_mode, int trig_dep,
    void* TileW13Done, void* TileReady, void* OverlapMeta,
    int overlap_w13_w2, int hot_l2_hint, uintptr_t stream_ptr
) {
    return launch_mx7_w13_c32_impl<0>(
        A, AS, Row2Tok, SLABS, WS3, FLIPS, F_OFF, nullptr, nullptr, nullptr,
        Ones, Tiles, TileCompactStart, Gammas, Out,
        TileW13Done, TileReady, OverlapMeta,
        N, K, MAX_CTAS, TILE, base_ws, zero_scatter, as_row_major, fuse_situ,
        pdl_wait_mode, trig_dep, overlap_w13_w2, hot_l2_hint, stream_ptr
    );
}

extern "C" int mx7_w2_c48_v8_fwd(
    const void* A, const void* AS,
    const void* SLABS, const void* WS3,
    const void* FLIPS, const void* F_OFF,
    const void* Ones, const void* Tiles, const void* TileCompactStart,
    void* Out,
    const void* Src2Compact, const void* NLivePtr,
    void* PanelDone, void* BufGen, void* SendBuf,
    int N, int K, int MAX_CTAS, int TILE, int T, uint32_t base_ws,
    int as_row_major, int as_row_stride, int down_row_stride,
    int fuse_quant, int fuse_combine,
    int pdl_wait_mode, int trig_dep,
    void* TileReady, void* OverlapMeta,
    int overlap_w13_w2, int hot_l2_hint, uintptr_t stream_ptr
) {
    return launch_mx7_w2_c32_impl<0>(
        A, AS, SLABS, WS3, FLIPS, F_OFF, nullptr, nullptr, nullptr,
        Ones, Tiles, TileCompactStart, Out, Src2Compact, NLivePtr,
        PanelDone, BufGen, SendBuf, TileReady, OverlapMeta,
        N, K, MAX_CTAS, TILE, T, base_ws, as_row_major, as_row_stride, down_row_stride,
        fuse_quant, fuse_combine, pdl_wait_mode, trig_dep, overlap_w13_w2, hot_l2_hint, stream_ptr
    );
}

extern "C" int mx7_w13_c48_v8_compat_fwd(
    const void* A, const void* AS, const void* Row2Tok,
    const void* SLABS, const void* WS3,
    const void* SLAB_META, const void* S_OFF, const void* D_OFF,
    const void* SFLIPS, const void* BMASKS,
    const void* Ones, const void* Tiles,
    const void* TileCompactStart, const void* Gammas, void* Out,
    int N, int K, int MAX_CTAS, int TILE, uint32_t base_ws,
    int zero_scatter, int as_row_major, int fuse_situ,
    int pdl_wait_mode, int trig_dep,
    void* TileW13Done, void* TileReady, void* OverlapMeta,
    int overlap_w13_w2, int hot_l2_hint, uintptr_t stream_ptr
) {
    return launch_mx7_w13_c32_impl<1>(
        A, AS, Row2Tok, SLABS, WS3, SFLIPS, S_OFF, SLAB_META, BMASKS, D_OFF,
        Ones, Tiles, TileCompactStart, Gammas, Out,
        TileW13Done, TileReady, OverlapMeta,
        N, K, MAX_CTAS, TILE, base_ws, zero_scatter, as_row_major, fuse_situ,
        pdl_wait_mode, trig_dep, overlap_w13_w2, hot_l2_hint, stream_ptr
    );
}

extern "C" int mx7_w2_c48_v8_compat_fwd(
    const void* A, const void* AS,
    const void* SLABS, const void* WS3,
    const void* SLAB_META, const void* S_OFF, const void* D_OFF,
    const void* SFLIPS, const void* BMASKS,
    const void* Ones, const void* Tiles, const void* TileCompactStart,
    void* Out,
    const void* Src2Compact, const void* NLivePtr,
    void* PanelDone, void* BufGen, void* SendBuf,
    int N, int K, int MAX_CTAS, int TILE, int T, uint32_t base_ws,
    int as_row_major, int as_row_stride, int down_row_stride,
    int fuse_quant, int fuse_combine,
    int pdl_wait_mode, int trig_dep,
    void* TileReady, void* OverlapMeta,
    int overlap_w13_w2, int hot_l2_hint, uintptr_t stream_ptr
) {
    return launch_mx7_w2_c32_impl<1>(
        A, AS, SLABS, WS3, SFLIPS, S_OFF, SLAB_META, BMASKS, D_OFF,
        Ones, Tiles, TileCompactStart, Out, Src2Compact, NLivePtr,
        PanelDone, BufGen, SendBuf, TileReady, OverlapMeta,
        N, K, MAX_CTAS, TILE, T, base_ws, as_row_major, as_row_stride, down_row_stride,
        fuse_quant, fuse_combine, pdl_wait_mode, trig_dep, overlap_w13_w2, hot_l2_hint, stream_ptr
    );
}

extern "C" int mx7_w13_c48_v8_panel_fwd(
    const void* A, const void* AS, const void* Row2Tok,
    const void* PANEL_POOL, const void* PANEL_OFF,
    const void* SLAB_META, const void* S_OFF, const void* SFLIPS,
    const void* Ones, const void* Tiles,
    const void* TileCompactStart, const void* Gammas, void* Out,
    int N, int K, int MAX_CTAS, int TILE, uint32_t base_ws,
    int zero_scatter, int as_row_major, int fuse_situ,
    int pdl_wait_mode, int trig_dep,
    void* TileW13Done, void* TileReady, void* OverlapMeta,
    int overlap_w13_w2, int hot_l2_hint, uintptr_t stream_ptr
) {
    return launch_mx7_w13_c32_impl<2>(
        A, AS, Row2Tok, PANEL_POOL, nullptr, SFLIPS, S_OFF, SLAB_META, nullptr, PANEL_OFF,
        Ones, Tiles, TileCompactStart, Gammas, Out,
        TileW13Done, TileReady, OverlapMeta,
        N, K, MAX_CTAS, TILE, base_ws, zero_scatter, as_row_major, fuse_situ,
        pdl_wait_mode, trig_dep, overlap_w13_w2, hot_l2_hint, stream_ptr
    );
}

extern "C" int mx7_w2_c48_v8_panel_fwd(
    const void* A, const void* AS,
    const void* PANEL_POOL, const void* PANEL_OFF,
    const void* SLAB_META, const void* S_OFF, const void* SFLIPS,
    const void* Ones, const void* Tiles, const void* TileCompactStart,
    void* Out,
    const void* Src2Compact, const void* NLivePtr,
    void* PanelDone, void* BufGen, void* SendBuf,
    int N, int K, int MAX_CTAS, int TILE, int T, uint32_t base_ws,
    int as_row_major, int as_row_stride, int down_row_stride,
    int fuse_quant, int fuse_combine,
    int pdl_wait_mode, int trig_dep,
    void* TileReady, void* OverlapMeta,
    int overlap_w13_w2, int hot_l2_hint, uintptr_t stream_ptr
) {
    return launch_mx7_w2_c32_impl<2>(
        A, AS, PANEL_POOL, nullptr, SFLIPS, S_OFF, SLAB_META, nullptr, PANEL_OFF,
        Ones, Tiles, TileCompactStart, Out, Src2Compact, NLivePtr,
        PanelDone, BufGen, SendBuf, TileReady, OverlapMeta,
        N, K, MAX_CTAS, TILE, T, base_ws, as_row_major, as_row_stride, down_row_stride,
        fuse_quant, fuse_combine, pdl_wait_mode, trig_dep, overlap_w13_w2, hot_l2_hint, stream_ptr
    );
}

// Legacy v4/v5/v7 drop-in entrypoints (delegate to v8 with overlap=0, hot_l2_hint=0)
extern "C" int mx7_w13_c32_fwd(
    const void* A, const void* AS, const void* Row2Tok,
    const void* SLABS, const void* WS3,
    const void* FLIPS, const void* F_OFF,
    const void* Ones, const void* Tiles,
    const void* TileCompactStart, const void* Gammas, void* Out,
    int N, int K, int MAX_CTAS, int TILE, uint32_t base_ws,
    int zero_scatter, int as_row_major, int fuse_situ,
    int pdl_wait_mode, int trig_dep, uintptr_t stream_ptr
) {
    return mx7_w13_c48_v8_fwd(
        A, AS, Row2Tok, SLABS, WS3, FLIPS, F_OFF,
        Ones, Tiles, TileCompactStart, Gammas, Out,
        N, K, MAX_CTAS, TILE, base_ws, zero_scatter, as_row_major, fuse_situ,
        pdl_wait_mode, trig_dep, nullptr, nullptr, nullptr, 0, 0, stream_ptr
    );
}

extern "C" int mx7_w2_c32_fwd(
    const void* A, const void* AS,
    const void* SLABS, const void* WS3,
    const void* FLIPS, const void* F_OFF,
    const void* Ones, const void* Tiles, const void* TileCompactStart,
    void* Out,
    const void* Src2Compact, const void* NLivePtr,
    void* PanelDone, void* BufGen, void* SendBuf,
    int N, int K, int MAX_CTAS, int TILE, int T, uint32_t base_ws,
    int as_row_major, int as_row_stride, int down_row_stride,
    int fuse_quant, int fuse_combine,
    int pdl_wait_mode, int trig_dep, uintptr_t stream_ptr
) {
    return mx7_w2_c48_v8_fwd(
        A, AS, SLABS, WS3, FLIPS, F_OFF,
        Ones, Tiles, TileCompactStart, Out, Src2Compact, NLivePtr,
        PanelDone, BufGen, SendBuf,
        N, K, MAX_CTAS, TILE, T, base_ws, as_row_major, as_row_stride, down_row_stride,
        fuse_quant, fuse_combine, pdl_wait_mode, trig_dep, nullptr, nullptr, 0, 0, stream_ptr
    );
}

extern "C" int mx7_w13_c32_v5_fwd(
    const void* A, const void* AS, const void* Row2Tok,
    const void* SLABS, const void* WS3,
    const void* SLAB_META, const void* S_OFF, const void* D_OFF,
    const void* SFLIPS, const void* BMASKS,
    const void* Ones, const void* Tiles,
    const void* TileCompactStart, const void* Gammas, void* Out,
    int N, int K, int MAX_CTAS, int TILE, uint32_t base_ws,
    int zero_scatter, int as_row_major, int fuse_situ,
    int pdl_wait_mode, int trig_dep, uintptr_t stream_ptr
) {
    return mx7_w13_c48_v8_compat_fwd(
        A, AS, Row2Tok, SLABS, WS3, SLAB_META, S_OFF, D_OFF, SFLIPS, BMASKS,
        Ones, Tiles, TileCompactStart, Gammas, Out,
        N, K, MAX_CTAS, TILE, base_ws, zero_scatter, as_row_major, fuse_situ,
        pdl_wait_mode, trig_dep, nullptr, nullptr, nullptr, 0, 0, stream_ptr
    );
}

extern "C" int mx7_w2_c32_v5_fwd(
    const void* A, const void* AS,
    const void* SLABS, const void* WS3,
    const void* SLAB_META, const void* S_OFF, const void* D_OFF,
    const void* SFLIPS, const void* BMASKS,
    const void* Ones, const void* Tiles, const void* TileCompactStart,
    void* Out,
    const void* Src2Compact, const void* NLivePtr,
    void* PanelDone, void* BufGen, void* SendBuf,
    int N, int K, int MAX_CTAS, int TILE, int T, uint32_t base_ws,
    int as_row_major, int as_row_stride, int down_row_stride,
    int fuse_quant, int fuse_combine,
    int pdl_wait_mode, int trig_dep, uintptr_t stream_ptr
) {
    return mx7_w2_c48_v8_compat_fwd(
        A, AS, SLABS, WS3, SLAB_META, S_OFF, D_OFF, SFLIPS, BMASKS,
        Ones, Tiles, TileCompactStart, Out, Src2Compact, NLivePtr,
        PanelDone, BufGen, SendBuf,
        N, K, MAX_CTAS, TILE, T, base_ws, as_row_major, as_row_stride, down_row_stride,
        fuse_quant, fuse_combine, pdl_wait_mode, trig_dep, nullptr, nullptr, 0, 0, stream_ptr
    );
}

extern "C" int mx7_w13_c32_v5_panel_fwd(
    const void* A, const void* AS, const void* Row2Tok,
    const void* PANEL_POOL, const void* PANEL_OFF,
    const void* SLAB_META, const void* S_OFF, const void* SFLIPS,
    const void* Ones, const void* Tiles,
    const void* TileCompactStart, const void* Gammas, void* Out,
    int N, int K, int MAX_CTAS, int TILE, uint32_t base_ws,
    int zero_scatter, int as_row_major, int fuse_situ,
    int pdl_wait_mode, int trig_dep, uintptr_t stream_ptr
) {
    return mx7_w13_c48_v8_panel_fwd(
        A, AS, Row2Tok, PANEL_POOL, PANEL_OFF, SLAB_META, S_OFF, SFLIPS,
        Ones, Tiles, TileCompactStart, Gammas, Out,
        N, K, MAX_CTAS, TILE, base_ws, zero_scatter, as_row_major, fuse_situ,
        pdl_wait_mode, trig_dep, nullptr, nullptr, nullptr, 0, 0, stream_ptr
    );
}

extern "C" int mx7_w2_c32_v5_panel_fwd(
    const void* A, const void* AS,
    const void* PANEL_POOL, const void* PANEL_OFF,
    const void* SLAB_META, const void* S_OFF, const void* SFLIPS,
    const void* Ones, const void* Tiles, const void* TileCompactStart,
    void* Out,
    const void* Src2Compact, const void* NLivePtr,
    void* PanelDone, void* BufGen, void* SendBuf,
    int N, int K, int MAX_CTAS, int TILE, int T, uint32_t base_ws,
    int as_row_major, int as_row_stride, int down_row_stride,
    int fuse_quant, int fuse_combine,
    int pdl_wait_mode, int trig_dep, uintptr_t stream_ptr
) {
    return mx7_w2_c48_v8_panel_fwd(
        A, AS, PANEL_POOL, PANEL_OFF, SLAB_META, S_OFF, SFLIPS,
        Ones, Tiles, TileCompactStart, Out, Src2Compact, NLivePtr,
        PanelDone, BufGen, SendBuf,
        N, K, MAX_CTAS, TILE, T, base_ws, as_row_major, as_row_stride, down_row_stride,
        fuse_quant, fuse_combine, pdl_wait_mode, trig_dep, nullptr, nullptr, 0, 0, stream_ptr
    );
}
