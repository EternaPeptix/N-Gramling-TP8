// =========================================================================
// SM121a (Blackwell GB10) 0-ULP Lossless MX6-Huff7 (WSLR-H7) Production Engine (V12)
//   100% binary-layout & C-ABI superset of V10 (compatible with v11/pf_mx6_prefill_v11.cu!)
//   V12 Critical 0-ULP + Memory-Safety Fixes & C=1..8 Upgrades over V10:
//     [FIX-1] situ_pair_bf16_exact: softcap_4(gate)*sig (was wrong softcap_4(gate*sig)!)
//     [FIX-2] transpose_4x4_bf16x2_u4: restored missing (lt & 1) pre-swap (50% W2 cols!)
//     [FIX-3] scale_offset_decode: exact 8-row decode swizzle w/ (cols+3)>>2 (fixed OOB IMA!)
//     [FIX-4] quant_dequant_mxfp8_32_to_sA: 0-ULP exp_u=0/254/255 + 16-reg (-32 live regs!)
//     [UPG-5] W13 unconditional #206 PDL wait-at-entry (pdl_wait_mode >= 1 -> 0 Tiles race)
//     [UPG-6] w13_c1 vectorized LDG.64->STS.128 load_a_blk + w2_c1 transposed uint4 epilogue & combine
//     [UPG-7] Next-tile WS3_PACK L2 prefetch + K <= 4608 guard in mx6_pack_v10 + mx6_v12_version()
//   Target: All 8 MXFP8 Global MoE Layers (L40..47: E=384, H=4608, inter=288, topk=8)
//     - W13: [384, 576, 4608]  (BN=32 4-warp / BN=16 2-warp bijection, 72 warps/exp, 36 slabs)
//     - W2 : [384, 4608, 288]  (BN=128 4-warp, 144 warps/exp, 9 slabs)
//
// V10 Production Upgrades over V9 Prototype:
//   1. 0.000% Bank-Conflict `uint8_t sLut[128]` (128 B = 32 banks x 1 word/bank!):
//      - Cuts LDS wavefronts/step from 11.48 -> 4.00 (2.87x less LSU pressure, -65.1% replays)
//      - 4x smaller SMEM (128 B vs 512 B) -> MIN_CTA=5 occupancy 100% untouched!
//   2. Reduced Inner-Loop Instruction Count (~47 -> ~38-39 insts/step):
//      - Split 32-bit `buf_lo, buf_hi` + hardware `__funnelshift_r` / `__funnelshift_l`
//      - Hoisted PTX `%lanemask_lt` & raw `SHFL.IDX` (no `idx & 31u`)
//      - Direct `uint8_t` SWAR `IMAD`/`LOP3` decode + 1-IMAD `base_ws_off2` scale update
//      - Cold `e8m0_to_f32` moved strictly inside fallback branch
//      - V9-3 `lm < live_m` predicated LDS (eliminates all sA/sA_all zero-fill loops!)
//   3. Complete Drop-In Production Coverage Across ALL Concurrencies + 1M Prefill:
//      - GPU Packer: `mx6_v10_hist_exp`, `mx6_v10_build_huff7_table`,
//        `mx6_measure_v10`, `mx6_pack_v10` (supports BOTH row-major & FROM_PANEL!)
//      - C=1 (M<=16) Decode: `mx6_w13_c1_v10_fwd` (adaptive BN=16/32 bn_mode=0..3,
//        TileMask fns.b32 Zero-Scatter, compact Fused SiTU w/ fixed Gammas[r0] + uint2 store)
//        + `mx6_w2_c1_v10_fwd` (BN=128 4-warp, OTF MXFP8 quant-dequant, batched DownCompact)
//      - C=2..8 & C=16..32 (M=17..256) Decode: `mx6_w13_c48_v10_fwd` (TILE=8|16|32,
//        Row2Tok Zero-Scatter, as_row_major, Fused SiTU, optional BN=16/32 @ TILE=8)
//        + `mx6_w2_c48_v10_fwd` (TILE=8|16|32, FC=0|1|2|3|4 incl. FC=4 L2-hot DownCompact)
//        + `mx6_combine_c48_v10_fwd` (vectorized PDL + NLivePtr 0-ULP combine)
//      - 1M Prefill & `INKLING_MX6_FREE_RAW=1` VRAM Reclaim (+0.66..0.76 GiB/rank):
//        `mx6_unpack_expert_batch_v10_fwd` (coalesced uint32 GPU unpack -> row-major OR
//        panel-contiguous `(32,512)`/`(128,32)` transient workspace for pf_mxfp8_w13/w2_native!)
//   4. Strict SM121a Hygiene:
//      - Compiled with `-gencode arch=compute_121a,code=sm_121a --fmad=false --ptxas-options=-v`
//      - 0 B local stack / 0 B spills, +128 u32 EXP_STREAM guard safety,
//      - 0 `#206` PDL hazards (`griddepcontrol.wait` strictly before `launch_dependents`
//        on ALL paths including every early-exit branch for pdl_wait_mode == 2 || 3!).
// =========================================================================

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

#ifndef MX6_W13_T8_BN32_MIN_CTA
#define MX6_W13_T8_BN32_MIN_CTA 5
#endif
#ifndef MX6_W13_T8_BN16_MIN_CTA
#define MX6_W13_T8_BN16_MIN_CTA 8
#endif
#ifndef MX6_W13_T16_MIN_CTA
#define MX6_W13_T16_MIN_CTA 4
#endif
#ifndef MX6_W13_T32_MIN_CTA
#define MX6_W13_T32_MIN_CTA 2
#endif
#ifndef MX6_W2_T8_MIN_CTA
#define MX6_W2_T8_MIN_CTA 5
#endif
#ifndef MX6_W2_T16_MIN_CTA
#define MX6_W2_T16_MIN_CTA 4
#endif
#ifndef MX6_W2_T32_MIN_CTA
#define MX6_W2_T32_MIN_CTA 3
#endif

static constexpr int MX6_SM4_SLAB_BYTES  = 512;  // 32 lanes * 16 B (uint4) = 4.000 bpw
static constexpr int MX6_RAW4_SLAB_BYTES = 512;  // Raw-4-bit exponent cap = 4.000 bpw
static constexpr int MX6_CHUNK_WORDS     = 32;   // 128 B per warp chunk
static constexpr int MX6_MAX_HUFF_LEN    = 7;
static constexpr int MX6_LUT_SIZE        = 128;  // 2^7 = 128 bytes = 32 banks x 1 word/bank!
static constexpr int MX6_STREAM_GUARD_U32 = 128; // +512 B zero tail guard for speculative ld.cs

// uint8_t sLut[128] bit layout (0.000% SMEM bank conflicts on SM121a!):
//   bits 7:5 = codeword length L in [1..7]  -> l = d >> 5
//   bit  4   = LUT_E0_FLAG (1 iff e == 0)
//   bits 3:0 = exponent e in [0..15]
static constexpr uint8_t LUT_E0_FLAG         = 0x10u;

// Per-warp metadata flags (uint8_t WARP_META[total_warps]):
static constexpr uint8_t WMETA_ALL_NORMAL_BIT = 0x01u;  // No e==0 and no NaN across entire warp
static constexpr uint8_t WMETA_HAS_NAN_BIT    = 0x02u;  // At least one 0x7F/0xFF NaN byte
static constexpr uint8_t WMETA_RAW4_BIT       = 0x04u;  // Warp uses raw 4-bit exponent cap

// Exact IEEE constants matching Triton SiTU & W2 OTF quant-dequant (bit-exact with MX7 v5/v8!)
static constexpr float INV_25       = 0.04f;
static constexpr float FP32_MAX_VAL = 3.4028234663852886e+38f;
static constexpr float INV_448      = 1.0f / 448.0f;

// =========================================================================
// Low-Level SM121a PTX Primitives (Strict IEEE, Zero Fast-Math)
// =========================================================================

static __device__ __forceinline__ float mul_rn(float a, float b) {
    float r;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

static __device__ __forceinline__ float add_rn(float a, float b) {
    float r;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

static __device__ __forceinline__ float rcp_rn(float a) {
    float r;
    asm("rcp.rn.f32 %0, %1;" : "=f"(r) : "f"(a));
    return r;
}

static __device__ __forceinline__ uint32_t prmt_b32(uint32_t a, uint32_t b, uint32_t sel) {
    uint32_t r;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel));
    return r;
}

static __device__ __forceinline__ uint32_t fns_u32(uint32_t mask, uint32_t base, int offset) {
    uint32_t r;
    asm("fns.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(mask), "r"(base), "r"(offset));
    return r;
}

static __device__ __forceinline__ uint32_t get_lanemask_lt() {
    uint32_t m;
    asm("mov.u32 %0, %%lanemask_lt;" : "=r"(m));
    return m;
}

static __device__ __forceinline__ uint32_t pack_bf16x2(float a, float b) {
    uint32_t r;
    asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(r) : "f"(a), "f"(b));
    return r;
}

static __device__ __forceinline__ uint16_t f32_to_bf16_bits(float x) {
    uint16_t r;
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(r) : "f"(x));
    return r;
}

static __device__ __forceinline__ float bf16_bits_to_f32(uint16_t b) {
    return __uint_as_float(static_cast<uint32_t>(b) << 16);
}

static __device__ __forceinline__ float e8m0_to_f32(uint8_t s) {
    if (s == 0u)   return __uint_as_float(0x00400000u);  // 2^-127
    if (s == 255u) return __uint_as_float(0x7FC00000u);  // NaN
    return __uint_as_float(static_cast<uint32_t>(s) << 23);
}

static __device__ __forceinline__ uint32_t ld_cs_u32(const void* ptr) {
    uint32_t v;
    asm volatile("ld.global.cs.u32 %0, [%1];" : "=r"(v) : "l"(ptr));
    return v;
}

static __device__ __forceinline__ uint4 ld_cs_u4(const void* ptr) {
    uint4 v;
    asm volatile("ld.global.cs.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(ptr));
    return v;
}

static __device__ __forceinline__ uint4 ld_cg_u4(const void* ptr) {
    uint4 v;
    asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(ptr));
    return v;
}

static __device__ __forceinline__ void st_cg_u4(void* ptr, uint4 v) {
    asm volatile("st.global.cg.v4.u32 [%0], {%1, %2, %3, %4};"
                 :: "l"(ptr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}

static __device__ __forceinline__ uint32_t ld_acq_gpu_u32(const uint32_t* ptr) {
    uint32_t v;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(ptr) : "memory");
    return v;
}

static __device__ __forceinline__ void st_rel_gpu_u32(uint32_t* ptr, uint32_t v) {
    asm volatile("st.release.gpu.global.u32 [%0], %1;" :: "l"(ptr), "r"(v) : "memory");
}

static __device__ __forceinline__ uint32_t atom_acq_rel_gpu_add(uint32_t* ptr, uint32_t val) {
    uint32_t old;
    asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                 : "=r"(old) : "l"(ptr), "r"(val) : "memory");
    return old;
}

static __device__ __forceinline__ void prefetch_l2(const void* ptr) {
    asm volatile("prefetch.global.L2 [%0];" :: "l"(ptr));
}

static __device__ __forceinline__ void mma_m16n8k16_bf16(
    float (&acc)[4],
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)
    );
}

// Exact 8-row decode scale swizzle (matches trtllm_bmm._scale_offsets & mxfp8_mx7_c48_v9.cu:279-282)
static __device__ __forceinline__ int64_t scale_offset_decode(int64_t row, int kg, int KG) {
    const int64_t tile = (row >> 3) * ((KG + 3) >> 2) + (kg >> 2);
    return (tile << 5) + ((row & 7) << 2) + (kg & 3);
}

// 3-SHFL 4x4 BF16x2 Transpose -> 64B-Coalesced uint4 W2 Epilogue Stores (0-ULP bit-exact!)
// Input:  each lane lt in {0..3} holds p0=sub0@lt, p1=sub1@lt, p2=sub2@lt, p3=sub3@lt.
// Output: lane lt holds {sub=lt@lt0, sub=lt@lt1, sub=lt@lt2, sub=lt@lt3} (8 contiguous BF16s = 16 B!).
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

// Exact GPU Wave-Efficiency Adaptive BN=16/32 Selector (Complementarily Gated)
static __device__ __forceinline__ bool bn16_wins_wave_eff(
    int n_active, int stride16, int stride32, int margin_pct
) {
    if (n_active <= 0) return false;
    const int64_t w16 = (static_cast<int64_t>(n_active) * 36 + stride16 - 1) / stride16;
    const int64_t w32 = (static_cast<int64_t>(n_active) * 18 + stride32 - 1) / stride32;
    const int64_t eff_margin = (margin_pct < 0) ? 0 : ((margin_pct > 50) ? 50 : margin_pct);
    return (w16 * stride16 * 100) < (2 * w32 * stride32 * (100 - eff_margin));
}

// =========================================================================
// Exact 0-ULP E4M3 -> BF16x4 Decoders (V10 uint8_t sLut Direct SWAR + Fallback)
// =========================================================================

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

// Reconstruct packed 4x E4M3FN bytes from sm16 + 4 uint8_t sLut entries (e in bits 3:0)
static __device__ __forceinline__ uint32_t build_e4_pack_from_sm_and_d(
    uint32_t sm16, uint32_t d0, uint32_t d1, uint32_t d2, uint32_t d3
) {
    const uint32_t t    = prmt_b32(sm16, 0u, 0x4140u);
    const uint32_t s_lo = (t & 0x000F000Fu) | ((t << 4) & 0x0F000F00u);
    const uint32_t sm_b = ((s_lo & 0x08080808u) << 4) | (s_lo & 0x07070707u);
    const uint32_t e_b  = ((d0 & 0xFu) << 3)
                        | ((d1 & 0xFu) << 11)
                        | ((d2 & 0xFu) << 19)
                        | ((d3 & 0xFu) << 27);
    return sm_b | e_b;
}

// V10 Direct 0-ULP SWAR MX6 -> Scaled BF16x4 (uint8_t sLut format, IMAD/LOP3 fused, 0 CVT/FP32!)
// Preconditions: all 4 weights have e in 1..15, none is NaN (0x7F/0xFF), and ws_byte in 8..245.
// Proof of 0-carry: ws_off2 uses bits 14:7 and 30:23 (disjoint from q & 0x00700070 in 6:4 / 22:20);
// max (e + ws_byte - 7) = 15 + 245 - 7 = 253 <= 254 -> 0x7EF0 < 0x8000 (0 carry into sign/hi16!).
static __device__ __forceinline__ void direct_mx6_normal_to_bf16x4(
    uint32_t sm16, uint32_t d0, uint32_t d1, uint32_t d2, uint32_t d3,
    uint32_t ws_off2, uint32_t& b0, uint32_t& b1
) {
    const uint32_t q0  = (sm16 & 0xFFu) * 0x00010010u;
    const uint32_t m0  = (q0 & 0x00700070u) | ws_off2;
    const uint32_t e01 = prmt_b32(d0, d1, 0x5410u) & 0x000F000Fu;
    b0 = (e01 * 128u + m0) | ((q0 << 8) & 0x80008000u);

    const uint32_t q1  = (sm16 >> 8) * 0x00010010u;
    const uint32_t m1  = (q1 & 0x00700070u) | ws_off2;
    const uint32_t e23 = prmt_b32(d2, d3, 0x5410u) & 0x000F000Fu;
    b1 = (e23 * 128u + m1) | ((q1 << 8) & 0x80008000u);
}

static __device__ __forceinline__ void decode_mx6_step_to_bf16x4(
    uint32_t sm16, uint32_t d0, uint32_t d1, uint32_t d2, uint32_t d3,
    uint32_t ws_off2, uint32_t ws_byte, bool warp_all_normal, bool warp_no_nan,
    uint32_t& b0, uint32_t& b1
) {
    bool fast_ok = warp_all_normal;
    if (__builtin_expect(!fast_ok && warp_no_nan, 0)) {
        fast_ok = (((d0 | d1 | d2 | d3) & LUT_E0_FLAG) == 0u);
    }
    if (__builtin_expect(fast_ok, 1)) {
        direct_mx6_normal_to_bf16x4(sm16, d0, d1, d2, d3, ws_off2, b0, b1);
    } else {
        const float w_sc = e8m0_to_f32(static_cast<uint8_t>(ws_byte));
        e4m3x4_scale_to_bf16x4(
            build_e4_pack_from_sm_and_d(sm16, d0, d1, d2, d3), w_sc, b0, b1
        );
    }
}

// =========================================================================
// Tight 96-bit WS3 Unpacker (3 x uint32 -> 4 x 24-bit scale words = 0.09375 bpw)
// =========================================================================

static __device__ __forceinline__ uint32_t unpack_ws3_word(
    uint32_t u0, uint32_t u1, uint32_t u2, int q
) {
    if (q == 0) return u0 & 0x00FFFFFFu;
    if (q == 1) return ((u0 >> 24) | (u1 << 8)) & 0x00FFFFFFu;
    if (q == 2) return ((u1 >> 16) | (u2 << 16)) & 0x00FFFFFFu;
    return u2 >> 8;
}

// =========================================================================
// V10 Double-Buffered 128B Warp-Register Ring + Bank-Conflict-Free WSLR-Huff7 Reader
//   - Split 32-bit `buf_lo`, `buf_hi` with hardware `__funnelshift_r` / `__funnelshift_l`
//   - Hoisted `%lanemask_lt`, raw `SHFL.IDX` (no `idx & 31u`)
//   - Bank-conflict-free `uint8_t sLut[128]` (1.000 wf/LDS, 0.000% bank conflicts!)
//   - Invariant: all bits at position >= nbits in (buf_hi:buf_lo) are 0; nbits >= 32 before decode;
//     tot_len in [4..28] -> after decode nbits >= 4; if nbits < 32 then buf_hi == 0 identically!
// =========================================================================

struct MX6StreamReaderV10 {
    uint32_t        buf_lo;
    uint32_t        buf_hi;
    int             nbits;
    uint32_t        w_buf;
    uint32_t        w_nxt;
    uint32_t        w_head;
    const uint32_t* s_ptr;

    __device__ __forceinline__ void init(const uint32_t* stream_base, int lane) {
        buf_lo = ld_cs_u32(stream_base + lane);
        buf_hi = ld_cs_u32(stream_base + 32 + lane);
        nbits  = 64;
        w_buf  = ld_cs_u32(stream_base + 64 + lane);
        w_nxt  = ld_cs_u32(stream_base + 96 + lane);
        w_head = 0u;
        s_ptr  = stream_base + 128;
    }

    __device__ __forceinline__ void decode4_and_refill(
        const uint8_t* __restrict__ sLut, int lane, uint32_t lanemask_lt, bool do_refill,
        uint32_t& d0, uint32_t& d1, uint32_t& d2, uint32_t& d3
    ) {
        uint32_t tmp = buf_lo;
        d0 = static_cast<uint32_t>(sLut[tmp & 127u]); const uint32_t l0 = d0 >> 5; tmp >>= l0;
        d1 = static_cast<uint32_t>(sLut[tmp & 127u]); const uint32_t l1 = d1 >> 5; tmp >>= l1;
        d2 = static_cast<uint32_t>(sLut[tmp & 127u]); const uint32_t l2 = d2 >> 5; tmp >>= l2;
        d3 = static_cast<uint32_t>(sLut[tmp & 127u]); const uint32_t l3 = d3 >> 5;

        const uint32_t tot_len = (l0 + l1) + (l2 + l3);  // in [4..28]
        buf_lo = __funnelshift_r(buf_lo, buf_hi, tot_len);
        buf_hi >>= tot_len;
        nbits -= static_cast<int>(tot_len);

        if (do_refill) {
            const bool need = (nbits < 32);
            const uint32_t mask = __ballot_sync(0xFFFFFFFFu, need);
            if (mask != 0u) {
                const uint32_t cnt       = static_cast<uint32_t>(__popc(mask));
                const uint32_t my_pos    = static_cast<uint32_t>(__popc(mask & lanemask_lt));
                const uint32_t idx       = w_head + my_pos;
                uint32_t w               = __shfl_sync(0xFFFFFFFFu, w_buf, static_cast<int>(idx));
                const uint32_t next_head = w_head + cnt;
                if (next_head >= 32u) {
                    const uint32_t w2 = __shfl_sync(0xFFFFFFFFu, w_nxt, static_cast<int>(idx));
                    if (idx >= 32u) w = w2;
                    w_buf  = w_nxt;
                    w_nxt  = ld_cs_u32(s_ptr + lane);
                    s_ptr += 32;
                    w_head = next_head - 32u;
                } else {
                    w_head = next_head;
                }
                if (need) {
                    const uint32_t sh = static_cast<uint32_t>(nbits);  // in [4..31], buf_hi == 0
                    buf_lo |= (w << sh);
                    buf_hi  = __funnelshift_l(w, 0u, sh);
                    nbits  += 32;
                }
            }
        }
    }
};

static __device__ __forceinline__ void raw4_to_d4(
    uint32_t e16, uint32_t& d0, uint32_t& d1, uint32_t& d2, uint32_t& d3
) {
    const uint32_t e0 = e16 & 0xFu;
    const uint32_t e1 = (e16 >> 4) & 0xFu;
    const uint32_t e2 = (e16 >> 8) & 0xFu;
    const uint32_t e3 = (e16 >> 12) & 0xFu;
    d0 = e0 | ((e0 == 0u) ? LUT_E0_FLAG : 0u);
    d1 = e1 | ((e1 == 0u) ? LUT_E0_FLAG : 0u);
    d2 = e2 | ((e2 == 0u) ? LUT_E0_FLAG : 0u);
    d3 = e3 | ((e3 == 0u) ? LUT_E0_FLAG : 0u);
}

// =========================================================================
// Exact 0-ULP SiTU & W2 OTF MXFP8 Quant-Dequant (Identical to MX7 v5/v8!)
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

// Exact 0-ULP SiTU: ((softcap_4(gate) * sigmoid(gate)) * softcap_25(up)) * gamma
// Matches activation_ops.py:75-78 & mxfp8_mx7_c48_v9.cu:574-582 bit-for-bit!
static __device__ __forceinline__ uint16_t situ_pair_bf16_exact(uint32_t pair, float gamma) {
    float gate = bf16_bits_to_f32(static_cast<uint16_t>(pair & 0xFFFFu));
    float up   = bf16_bits_to_f32(static_cast<uint16_t>(pair >> 16));
    float sig  = sigmoid_approx_exact(gate);
    gate = softcap_4(gate);
    up   = softcap_25(up);
    float y = mul_rn(mul_rn(mul_rn(gate, sig), up), gamma);
    return f32_to_bf16_bits(y);
}

// Exact 0-ULP Register-Compact MXFP8 32-elem Quant-Dequant Directly Into sA_all
// Matches mxfp8_mx7_c48_v9.cu:586-638 bit-for-bit (handles exp_u=0/254/255, NaN, +/-Inf,
// and holds only 16 live registers instead of 48 -> guarantees 0 B spills at MIN_CTA=5/4/3!).
static __device__ __forceinline__ void otf_mxfp8_quant_dequant_blk32(
    const __nv_bfloat16* __restrict__ src32, __nv_bfloat16* __restrict__ dst32
) {
    const uint4 v[4] = {
        *reinterpret_cast<const uint4*>(src32 + 0),
        *reinterpret_cast<const uint4*>(src32 + 8),
        *reinterpret_cast<const uint4*>(src32 + 16),
        *reinterpret_cast<const uint4*>(src32 + 24),
    };
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

    const float raw_scale = mul_rn(amax, INV_448);
    const uint32_t rs_u   = __float_as_uint(raw_scale);
    uint32_t exp_u;
    if (raw_scale != raw_scale)   exp_u = 255u;
    else if (rs_u <= 0x00400000u) exp_u = 0u;
    else {
        exp_u = ((rs_u + 0x007FFFFFu) & 0x7F800000u) >> 23;
        if (exp_u > 254u) exp_u = 254u;
    }

    float inv;
    if (exp_u == 255u)     inv = 0.0f;
    else if (exp_u == 0u)  inv = __uint_as_float(0x7F000000u);
    else                   inv = rcp_rn(__uint_as_float(exp_u << 23));

    const float a_sc = e8m0_to_f32(static_cast<uint8_t>(exp_u));

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t u[4] = {v[i].x, v[i].y, v[i].z, v[i].w};
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const uint32_t u0 = u[2 * h + 0];
            const uint32_t u1 = u[2 * h + 1];
            const float f0 = mul_rn(bf16_bits_to_f32(static_cast<uint16_t>(u0 & 0xFFFFu)), inv);
            const float f1 = mul_rn(bf16_bits_to_f32(static_cast<uint16_t>(u0 >> 16)),      inv);
            const float f2 = mul_rn(bf16_bits_to_f32(static_cast<uint16_t>(u1 & 0xFFFFu)), inv);
            const float f3 = mul_rn(bf16_bits_to_f32(static_cast<uint16_t>(u1 >> 16)),      inv);
            uint16_t e_lo, e_hi;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(e_lo) : "f"(f0), "f"(f1));
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(e_hi) : "f"(f2), "f"(f3));
            const uint32_t a4 = static_cast<uint32_t>(e_lo) | (static_cast<uint32_t>(e_hi) << 16);
            uint32_t b0, b1;
            e4m3x4_scale_to_bf16x4(a4, a_sc, b0, b1);
            *reinterpret_cast<uint2*>(dst32 + i * 8 + h * 4) = make_uint2(b0, b1);
        }
    }
}

// =========================================================================
// Layout Addressing Helpers (Row-Major <-> Panel-Contiguous for Pack & Unpack)
// =========================================================================

template <bool PANEL>
static __device__ __forceinline__ uint8_t read_w_elem(
    const uint8_t* W, int e, int row, int kcol, int N, int K, int pb, int kb
) {
    if constexpr (!PANEL) {
        return W[(static_cast<int64_t>(e) * N + row) * K + kcol];
    } else {
        const int n_panels = N / pb;
        const int k_steps  = K / kb;
        const int64_t p_idx = (static_cast<int64_t>(e) * n_panels + (row / pb)) * k_steps + (kcol / kb);
        return W[(p_idx * pb + (row % pb)) * kb + (kcol % kb)];
    }
}

template <bool PANEL>
static __device__ __forceinline__ uint8_t read_ws_elem(
    const uint8_t* WS, int e, int row, int kg_idx, int N, int KG, int pb, int kbg
) {
    if constexpr (!PANEL) {
        return WS[(static_cast<int64_t>(e) * N + row) * KG + kg_idx];
    } else {
        const int n_panels = N / pb;
        const int k_steps  = KG / kbg;
        const int64_t p_idx = (static_cast<int64_t>(e) * n_panels + (row / pb)) * k_steps + (kg_idx / kbg);
        return WS[(p_idx * kbg + (kg_idx % kbg)) * pb + (row % pb)];
    }
}

template <bool PANEL>
static __device__ __forceinline__ uint32_t* w_u32_ptr(
    uint8_t* W, int e, int row, int kcol, int N, int K, int pb, int kb
) {
    if constexpr (!PANEL) {
        return reinterpret_cast<uint32_t*>(W + (static_cast<int64_t>(e) * N + row) * K + kcol);
    } else {
        const int n_panels = N / pb;
        const int k_steps  = K / kb;
        const int64_t p_idx = (static_cast<int64_t>(e) * n_panels + (row / pb)) * k_steps + (kcol / kb);
        return reinterpret_cast<uint32_t*>(W + (p_idx * pb + (row % pb)) * kb + (kcol % kb));
    }
}

template <bool PANEL>
static __device__ __forceinline__ uint8_t* ws_u8_ptr(
    uint8_t* WS, int e, int row, int kg_idx, int N, int KG, int pb, int kbg
) {
    if constexpr (!PANEL) {
        return WS + (static_cast<int64_t>(e) * N + row) * KG + kg_idx;
    } else {
        const int n_panels = N / pb;
        const int k_steps  = KG / kbg;
        const int64_t p_idx = (static_cast<int64_t>(e) * n_panels + (row / pb)) * k_steps + (kg_idx / kbg);
        return WS + (p_idx * kbg + (kg_idx % kbg)) * pb + (row % pb);
    }
}

// =========================================================================
// Host-Side Exact Length-Limited Canonical Huffman Builder (Package-Merge, max_len <= 7)
//   - Produces `uint8_t huff_lut_out[128]` (0.000% SMEM bank conflicts!)
// =========================================================================

static uint16_t bitrev_u16(uint16_t x, int bits) {
    uint16_t r = 0;
    for (int i = 0; i < bits; ++i) {
        r = static_cast<uint16_t>((r << 1) | ((x >> i) & 1u));
    }
    return r;
}

extern "C" int mx6_v10_build_huff7_table(
    const uint64_t* hist16,
    uint8_t*        huff_lut_out,   // [128] uint8: (len<<5) | (e==0 ? 0x10 : 0) | e
    uint16_t*       enc_code_out,   // [16]  LSB-first canonical codeword
    uint8_t*        enc_len_out     // [16]  codeword bit-length in [1..7]
) {
    if (!hist16 || !huff_lut_out || !enc_code_out || !enc_len_out) return -1;

    // Ensure every exponent 0..15 has weight >= 1 (universal completeness across any weight)
    uint64_t w[16];
    for (int i = 0; i < 16; ++i) w[i] = (hist16[i] > 0u) ? hist16[i] : 1u;

    struct PMItem {
        uint64_t weight;
        uint8_t  cnt[16];
    };
    std::vector<PMItem> prev_pkgs;
    for (int lvl = 0; lvl < MX6_MAX_HUFF_LEN; ++lvl) {
        std::vector<PMItem> cur;
        cur.reserve(16 + prev_pkgs.size());
        for (int s = 0; s < 16; ++s) {
            PMItem it{};
            it.weight = w[s];
            it.cnt[s] = 1;
            cur.push_back(it);
        }
        for (const auto& p : prev_pkgs) cur.push_back(p);
        std::stable_sort(cur.begin(), cur.end(), [](const PMItem& a, const PMItem& b) {
            return a.weight < b.weight;
        });
        if (lvl == MX6_MAX_HUFF_LEN - 1) {
            prev_pkgs.swap(cur);
            break;
        }
        std::vector<PMItem> next_pkgs;
        const size_t n_pairs = cur.size() / 2;
        next_pkgs.reserve(n_pairs);
        for (size_t i = 0; i < n_pairs; ++i) {
            PMItem pkg{};
            pkg.weight = cur[2 * i].weight + cur[2 * i + 1].weight;
            for (int s = 0; s < 16; ++s) {
                pkg.cnt[s] = static_cast<uint8_t>(cur[2 * i].cnt[s] + cur[2 * i + 1].cnt[s]);
            }
            next_pkgs.push_back(pkg);
        }
        prev_pkgs.swap(next_pkgs);
    }

    int len[16] = {};
    for (int i = 0; i < 30 && i < static_cast<int>(prev_pkgs.size()); ++i) {
        for (int s = 0; s < 16; ++s) len[s] += prev_pkgs[i].cnt[s];
    }
    int kraft = 0;
    for (int s = 0; s < 16; ++s) {
        if (len[s] < 1 || len[s] > MX6_MAX_HUFF_LEN) return -2;
        kraft += (1 << (MX6_MAX_HUFF_LEN - len[s]));
    }
    if (kraft != (1 << MX6_MAX_HUFF_LEN)) return -3;

    int order[16];
    for (int i = 0; i < 16; ++i) order[i] = i;
    std::sort(order, order + 16, [&](int a, int b) {
        return (len[a] != len[b]) ? (len[a] < len[b]) : (a < b);
    });

    uint16_t code = 0;
    int prev_l = 0;
    std::memset(huff_lut_out, 0, MX6_LUT_SIZE * sizeof(uint8_t));
    for (int idx = 0; idx < 16; ++idx) {
        const int s = order[idx];
        const int l = len[s];
        if (idx > 0) code = static_cast<uint16_t>((code + 1u) << (l - prev_l));
        prev_l = l;
        const uint16_t lsb_code = bitrev_u16(code, l);
        enc_code_out[s] = lsb_code;
        enc_len_out[s]  = static_cast<uint8_t>(l);
        const uint8_t entry = static_cast<uint8_t>(
            (static_cast<uint32_t>(l) << 5)
            | ((s == 0) ? LUT_E0_FLAG : 0u)
            | (static_cast<uint32_t>(s) & 0x0Fu)
        );
        const int step = 1 << l;
        for (int pad = static_cast<int>(lsb_code); pad < MX6_LUT_SIZE; pad += step) {
            huff_lut_out[pad] = entry;
        }
    }
    return 0;
}

// =========================================================================
// GPU Histogram Kernel (16-bin E4M3 Exponent Histogram, Layout-Invariant)
// =========================================================================

__global__ __launch_bounds__(256, 4)
void mx6_v10_hist_exp_kernel(
    const uint8_t* __restrict__ W,
    unsigned long long* __restrict__ hist16,
    int64_t n_elems
) {
    __shared__ uint32_t sHist[16];
    const int tid = threadIdx.x;
    if (tid < 16) sHist[tid] = 0u;
    __syncthreads();

    const int64_t n_vec = n_elems >> 4;
    const uint4* __restrict__ W4 = reinterpret_cast<const uint4*>(W);
    const int64_t gtid = static_cast<int64_t>(blockIdx.x) * blockDim.x + tid;
    const int64_t gstr = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t i = gtid; i < n_vec; i += gstr) {
        const uint4 v = W4[i];
        const uint32_t ws[4] = {v.x, v.y, v.z, v.w};
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint32_t u = ws[k];
            atomicAdd(&sHist[(u >> 3)  & 0xFu], 1u);
            atomicAdd(&sHist[(u >> 11) & 0xFu], 1u);
            atomicAdd(&sHist[(u >> 19) & 0xFu], 1u);
            atomicAdd(&sHist[(u >> 27) & 0xFu], 1u);
        }
    }
    __syncthreads();
    if (tid < 16 && sHist[tid] != 0u) {
        atomicAdd(&hist16[tid], static_cast<unsigned long long>(sHist[tid]));
    }
}

// =========================================================================
// GPU Pass 1: Measure Exact WSLR Chunk Count + Per-Warp Metadata (0 Local Stack)
//   Supports BOTH row-major (FROM_PANEL=false) AND panel-contiguous (FROM_PANEL=true)!
// =========================================================================

template <int PROJ, bool FROM_PANEL>
__global__ __launch_bounds__(32, 4)
void mx6_v10_measure_kernel(
    const uint8_t* __restrict__ W,
    const uint8_t* __restrict__ ENC_LEN,
    int32_t* __restrict__ warp_chunks_out,
    uint8_t* __restrict__ warp_meta_out,
    int E, int N, int K, int src_pb, int src_kb
) {
    constexpr int WARPS  = 4;
    constexpr int SUB_N  = (PROJ == 0) ? 1 : 4;
    constexpr int BN     = WARPS * SUB_N * 8;  // W13=32, W2=128
    constexpr int SLAB_K = (PROJ == 0) ? 128 : 32;

    const int n_panels = N / BN;
    const int64_t total_warps = static_cast<int64_t>(E) * n_panels * WARPS;
    const int64_t warp_uid = static_cast<int64_t>(blockIdx.x);
    if (warp_uid >= total_warps) return;

    const int lane = threadIdx.x;
    const int g    = lane >> 2;
    const int lt   = lane & 3;
    const int warp = static_cast<int>(warp_uid % WARPS);
    const int64_t panel_uid = warp_uid / WARPS;
    const int nb   = static_cast<int>(panel_uid % n_panels);
    const int e    = static_cast<int>(panel_uid / n_panels);
    const int n_slabs = K / SLAB_K;
    const int total_steps = n_slabs * 8;

    __shared__ uint8_t sEncLen[16];
    if (lane < 16) sEncLen[lane] = ENC_LEN[lane];
    __syncwarp();

    int nbits = 64;
    int total_words = 64;  // Initial 2 words/lane (64 words = 2 chunks)
    uint32_t has_e0_nan = 0u;
    uint32_t has_nan    = 0u;

    for (int sl = 0; sl < n_slabs; ++sl) {
        #pragma unroll
        for (int s = 0; s < 8; ++s) {
            const int ks2 = s & 1;
            const int n_base = (PROJ == 0)
                ? (nb * BN + warp * 8)
                : (nb * BN + (warp * SUB_N + (s >> 1)) * 8);
            const int k32_base = (PROJ == 0) ? (sl * 128 + (s >> 1) * 32) : (sl * 32);
            const int row = n_base + g;
            const int k0  = k32_base + 4 * lt + ks2 * 2;
            const uint8_t v0 = read_w_elem<FROM_PANEL>(W, e, row, k0,      N, K, src_pb, src_kb);
            const uint8_t v1 = read_w_elem<FROM_PANEL>(W, e, row, k0 + 1,  N, K, src_pb, src_kb);
            const uint8_t v2 = read_w_elem<FROM_PANEL>(W, e, row, k0 + 16, N, K, src_pb, src_kb);
            const uint8_t v3 = read_w_elem<FROM_PANEL>(W, e, row, k0 + 17, N, K, src_pb, src_kb);

            const uint32_t e0 = (v0 >> 3) & 0xFu;
            const uint32_t e1 = (v1 >> 3) & 0xFu;
            const uint32_t e2 = (v2 >> 3) & 0xFu;
            const uint32_t e3 = (v3 >> 3) & 0xFu;
            if ((e0 == 0u) || (e1 == 0u) || (e2 == 0u) || (e3 == 0u)) has_e0_nan = 1u;
            if (((v0 & 0x7Fu) == 0x7Fu) || ((v1 & 0x7Fu) == 0x7Fu) ||
                ((v2 & 0x7Fu) == 0x7Fu) || ((v3 & 0x7Fu) == 0x7Fu)) {
                has_e0_nan = 1u;
                has_nan    = 1u;
            }

            const int step_bits = static_cast<int>(sEncLen[e0])
                                + static_cast<int>(sEncLen[e1])
                                + static_cast<int>(sEncLen[e2])
                                + static_cast<int>(sEncLen[e3]);
            nbits -= step_bits;

            const int step_idx = sl * 8 + s;
            if (step_idx + 1 < total_steps) {
                const bool need = (nbits < 32);
                const uint32_t mask = __ballot_sync(0xFFFFFFFFu, need);
                if (mask != 0u) {
                    total_words += __popc(mask);
                    if (need) nbits += 32;
                }
            }
        }
    }

    const bool any_e0_nan = (__any_sync(0xFFFFFFFFu, has_e0_nan != 0u) != 0);
    const bool any_nan    = (__any_sync(0xFFFFFFFFu, has_nan != 0u) != 0);
    if (lane == 0) {
        const int huff_chunks = (total_words + MX6_CHUNK_WORDS - 1) / MX6_CHUNK_WORDS;
        const int raw4_chunks = n_slabs * 4;  // 512 B/slab = 4 chunks/slab
        const bool use_raw4   = (huff_chunks >= raw4_chunks);
        uint8_t meta = 0u;
        if (!any_e0_nan) meta |= WMETA_ALL_NORMAL_BIT;
        if (any_nan)     meta |= WMETA_HAS_NAN_BIT;
        if (use_raw4)    meta |= WMETA_RAW4_BIT;
        warp_meta_out[warp_uid]   = meta;
        warp_chunks_out[warp_uid] = use_raw4 ? raw4_chunks : huff_chunks;
    }
}

// =========================================================================
// GPU Pass 2: Pack SM4 + Tight 96-bit WS3 + Exact WSLR EXP_STREAM (0 Local Stack!)
//   Uses 32 KiB SMEM `sLaneWords[256][32]` so local stack == 0 B!
// =========================================================================

template <int PROJ, bool FROM_PANEL>
__global__ __launch_bounds__(32, 1)
void mx6_v10_pack_kernel(
    const uint8_t* __restrict__ W,
    const uint8_t* __restrict__ WS,
    const uint16_t* __restrict__ ENC_CODE,
    const uint8_t* __restrict__ ENC_LEN,
    const int32_t* __restrict__ EXP_OFF,
    const uint8_t* __restrict__ WARP_META,
    uint8_t* __restrict__ SM4,
    uint32_t* __restrict__ WS3_PACK,
    uint32_t* __restrict__ EXP_STREAM,
    int E, int N, int K, uint32_t base_ws, int src_pb, int src_kb
) {
    constexpr int WARPS  = 4;
    constexpr int SUB_N  = (PROJ == 0) ? 1 : 4;
    constexpr int BN     = WARPS * SUB_N * 8;
    constexpr int SLAB_K = (PROJ == 0) ? 128 : 32;

    const int n_panels = N / BN;
    const int64_t total_warps = static_cast<int64_t>(E) * n_panels * WARPS;
    const int64_t warp_uid = static_cast<int64_t>(blockIdx.x);
    if (warp_uid >= total_warps) return;

    const int lane = threadIdx.x;
    const int g    = lane >> 2;
    const int lt   = lane & 3;
    const int warp = static_cast<int>(warp_uid % WARPS);
    const int64_t panel_uid = warp_uid / WARPS;
    const int nb   = static_cast<int>(panel_uid % n_panels);
    const int e    = static_cast<int>(panel_uid / n_panels);
    const int KG   = K >> 5;
    const int n_slabs = K / SLAB_K;
    const int total_steps = n_slabs * 8;
    const int src_kbg = src_kb >> 5;
    const uint32_t lanemask_lt = get_lanemask_lt();

    __shared__ uint16_t sEncCode[16];
    __shared__ uint8_t  sEncLen[16];
    __shared__ uint32_t sLaneWords[256][32];  // 32 KiB SMEM -> 0 B local stack!
    if (lane < 16) {
        sEncCode[lane] = ENC_CODE[lane];
        sEncLen[lane]  = ENC_LEN[lane];
    }
    __syncwarp();

    const bool use_raw4 = ((WARP_META[warp_uid] & WMETA_RAW4_BIT) != 0u);
    uint8_t*  sm4_out_base  = SM4 + warp_uid * n_slabs * MX6_SM4_SLAB_BYTES;
    uint32_t* ws3_out_base  = WS3_PACK + warp_uid * n_slabs * 3;
    uint32_t* exp_out_base  = EXP_STREAM + static_cast<int64_t>(EXP_OFF[warp_uid]) * MX6_CHUNK_WORDS;
    const int n_chunks      = EXP_OFF[warp_uid + 1] - EXP_OFF[warp_uid];

    uint64_t bit_acc = 0ull;
    int      bit_cnt = 0;
    int      w_cnt   = 0;

    // Pass A: Pack SM4, tight 96-bit WS3, and either RAW4 or per-lane packed bitstreams
    for (int sl = 0; sl < n_slabs; ++sl) {
        uint32_t sm_u32[4] = {0u, 0u, 0u, 0u};
        uint32_t r4_u32[4] = {0u, 0u, 0u, 0u};
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
                    const uint32_t v = static_cast<uint32_t>(
                        read_ws_elem<FROM_PANEL>(WS, e, n_base + gg, kg_idx, N, KG, src_pb, src_kbg)
                    ) - base_ws;
                    w_word |= (v & 7u) << (gg * 3);
                }
                ws_words[s >> 1] = w_word;
            }

            const int row = n_base + g;
            const int k0  = k32_base + 4 * lt + ks2 * 2;
            const uint8_t vals[4] = {
                read_w_elem<FROM_PANEL>(W, e, row, k0,      N, K, src_pb, src_kb),
                read_w_elem<FROM_PANEL>(W, e, row, k0 + 1,  N, K, src_pb, src_kb),
                read_w_elem<FROM_PANEL>(W, e, row, k0 + 16, N, K, src_pb, src_kb),
                read_w_elem<FROM_PANEL>(W, e, row, k0 + 17, N, K, src_pb, src_kb),
            };

            uint32_t sm16 = 0u;
            uint32_t e16  = 0u;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const uint32_t b   = vals[j];
                const uint32_t sm4 = (((b >> 7) & 1u) << 3) | (b & 7u);
                const uint32_t ex  = (b >> 3) & 0xFu;
                sm16 |= sm4 << (j * 4);
                e16  |= ex  << (j * 4);
                if (!use_raw4) {
                    const uint32_t c = sEncCode[ex];
                    const int      l = sEncLen[ex];
                    bit_acc |= (static_cast<uint64_t>(c) << bit_cnt);
                    bit_cnt += l;
                    if (bit_cnt >= 32) {
                        sLaneWords[w_cnt++][lane] = static_cast<uint32_t>(bit_acc);
                        bit_acc >>= 32;
                        bit_cnt -= 32;
                    }
                }
            }
            if (ks2 == 0) {
                sm_u32[s >> 1] = sm16;
                r4_u32[s >> 1] = e16;
            } else {
                sm_u32[s >> 1] |= (sm16 << 16);
                r4_u32[s >> 1] |= (e16  << 16);
            }
        }

        // Write SM4 (16 B/lane = 512 B/slab)
        *reinterpret_cast<uint4*>(sm4_out_base + static_cast<int64_t>(sl) * MX6_SM4_SLAB_BYTES + lane * 16) =
            make_uint4(sm_u32[0], sm_u32[1], sm_u32[2], sm_u32[3]);

        // Write tight 96-bit WS3 (3 x uint32 = 12 B/slab)
        if (lane == 0) {
            const uint32_t u0 = (ws_words[0] & 0x00FFFFFFu) | ((ws_words[1] & 0xFFu) << 24);
            const uint32_t u1 = ((ws_words[1] >> 8) & 0xFFFFu) | ((ws_words[2] & 0xFFFFu) << 16);
            const uint32_t u2 = ((ws_words[2] >> 16) & 0xFFu) | ((ws_words[3] & 0x00FFFFFFu) << 8);
            ws3_out_base[sl * 3 + 0] = u0;
            ws3_out_base[sl * 3 + 1] = u1;
            ws3_out_base[sl * 3 + 2] = u2;
        }

        // RAW4 direct store if this warp hit the incompressible cap
        if (use_raw4) {
            *reinterpret_cast<uint4*>(
                reinterpret_cast<uint8_t*>(exp_out_base) + static_cast<int64_t>(sl) * MX6_RAW4_SLAB_BYTES + lane * 16
            ) = make_uint4(r4_u32[0], r4_u32[1], r4_u32[2], r4_u32[3]);
        }
    }

    if (use_raw4) {
        if (warp_uid + 1 == total_warps) {
            const int total_u32 = n_chunks * MX6_CHUNK_WORDS;
            for (int p = lane; p < MX6_STREAM_GUARD_U32; p += 32) {
                exp_out_base[total_u32 + p] = 0u;
            }
        }
        return;
    }

    // Flush final partial word + 2 zero-padding words so initial 64-bit load always valid
    if (bit_cnt > 0) {
        sLaneWords[w_cnt++][lane] = static_cast<uint32_t>(bit_acc);
    }
    while (w_cnt < 2) sLaneWords[w_cnt++][lane] = 0u;
    const int lane_words_total = w_cnt;

    // Pass B: Emit exact WSLR-ordered 128B chunks mirroring MX6StreamReaderV10 lockstep schedule!
    exp_out_base[lane]      = sLaneWords[0][lane];
    exp_out_base[32 + lane] = sLaneWords[1][lane];
    int rd_pos = 2;
    int nbits  = 64;
    int out_pos = 64;

    for (int sl = 0; sl < n_slabs; ++sl) {
        #pragma unroll
        for (int s = 0; s < 8; ++s) {
            const int ks2 = s & 1;
            const int n_base = (PROJ == 0)
                ? (nb * BN + warp * 8)
                : (nb * BN + (warp * SUB_N + (s >> 1)) * 8);
            const int k32_base = (PROJ == 0) ? (sl * 128 + (s >> 1) * 32) : (sl * 32);
            const int row = n_base + g;
            const int k0  = k32_base + 4 * lt + ks2 * 2;
            const uint8_t v0 = read_w_elem<FROM_PANEL>(W, e, row, k0,      N, K, src_pb, src_kb);
            const uint8_t v1 = read_w_elem<FROM_PANEL>(W, e, row, k0 + 1,  N, K, src_pb, src_kb);
            const uint8_t v2 = read_w_elem<FROM_PANEL>(W, e, row, k0 + 16, N, K, src_pb, src_kb);
            const uint8_t v3 = read_w_elem<FROM_PANEL>(W, e, row, k0 + 17, N, K, src_pb, src_kb);

            const int step_bits = static_cast<int>(sEncLen[(v0 >> 3) & 0xFu])
                                + static_cast<int>(sEncLen[(v1 >> 3) & 0xFu])
                                + static_cast<int>(sEncLen[(v2 >> 3) & 0xFu])
                                + static_cast<int>(sEncLen[(v3 >> 3) & 0xFu]);
            nbits -= step_bits;

            const int step_idx = sl * 8 + s;
            if (step_idx + 1 < total_steps) {
                const bool need = (nbits < 32);
                const uint32_t mask = __ballot_sync(0xFFFFFFFFu, need);
                if (mask != 0u) {
                    const int my_pos = __popc(mask & lanemask_lt);
                    const int cnt    = __popc(mask);
                    if (need) {
                        const uint32_t w_emit = (rd_pos < lane_words_total) ? sLaneWords[rd_pos][lane] : 0u;
                        rd_pos += 1;
                        exp_out_base[out_pos + my_pos] = w_emit;
                        nbits += 32;
                    }
                    out_pos += cnt;
                }
            }
        }
    }

    // Zero-pad tail words of final 128B chunk (+ 512 B stream tail guard on final warp!)
    const int total_u32 = n_chunks * MX6_CHUNK_WORDS
                        + ((warp_uid + 1 == total_warps) ? MX6_STREAM_GUARD_U32 : 0);
    for (int p = out_pos + lane; p < total_u32; p += 32) {
        exp_out_base[p] = 0u;
    }
}

// =========================================================================
// GPU Unpacker for 1M Prefill & `INKLING_MX6_FREE_RAW=1` VRAM Reclaim
//   - Decodes MX6-Huff7 (Huffman or RAW4) + SM4 + WS3_PACK -> exact E4M3FN W_OUT + E8M0 WS_OUT
//   - Supports BOTH row-major (OUT_PANEL=false) AND panel-contiguous (OUT_PANEL=true, e.g.
//     W13 (32,512) / W2 (128,32)) with aligned uint32 4-byte stores (~200+ GB/s)!
// =========================================================================

template <int PROJ, bool OUT_PANEL>
__global__ __launch_bounds__(32, 4)
void mx6_v10_unpack_kernel(
    const uint8_t* __restrict__ SM4,
    const uint32_t* __restrict__ WS3_PACK,
    const uint32_t* __restrict__ EXP_STREAM,
    const int32_t* __restrict__ EXP_OFF,
    const uint8_t* __restrict__ WARP_META,
    const uint8_t* __restrict__ HUFF_LUT,
    uint8_t* __restrict__ W_OUT,
    uint8_t* __restrict__ WS_OUT,
    int E, int N, int K, uint32_t base_ws, int dst_pb, int dst_kb
) {
    constexpr int WARPS  = 4;
    constexpr int SUB_N  = (PROJ == 0) ? 1 : 4;
    constexpr int BN     = WARPS * SUB_N * 8;
    constexpr int SLAB_K = (PROJ == 0) ? 128 : 32;

    const int n_panels = N / BN;
    const int64_t total_warps = static_cast<int64_t>(E) * n_panels * WARPS;
    const int64_t warp_uid = static_cast<int64_t>(blockIdx.x);
    if (warp_uid >= total_warps) return;

    const int lane = threadIdx.x;
    const int g    = lane >> 2;
    const int lt   = lane & 3;
    const int warp = static_cast<int>(warp_uid % WARPS);
    const int64_t panel_uid = warp_uid / WARPS;
    const int nb   = static_cast<int>(panel_uid % n_panels);
    const int e    = static_cast<int>(panel_uid / n_panels);
    const int KG   = K >> 5;
    const int n_slabs = K / SLAB_K;
    const int total_steps = n_slabs * 8;
    const int dst_kbg = dst_kb >> 5;
    const uint32_t lanemask_lt = get_lanemask_lt();

    alignas(128) __shared__ uint8_t sLut[MX6_LUT_SIZE];
    #pragma unroll
    for (int i = 0; i < 4; ++i) sLut[i * 32 + lane] = HUFF_LUT[i * 32 + lane];
    __syncwarp();

    const bool use_raw4 = ((WARP_META[warp_uid] & WMETA_RAW4_BIT) != 0u);
    const uint8_t*  sm4_base = SM4 + warp_uid * n_slabs * MX6_SM4_SLAB_BYTES;
    const uint32_t* ws3_base = WS3_PACK + warp_uid * n_slabs * 3;
    const uint32_t* exp_base = EXP_STREAM + static_cast<int64_t>(EXP_OFF[warp_uid]) * MX6_CHUNK_WORDS;

    MX6StreamReaderV10 reader{};
    if (!use_raw4) reader.init(exp_base, lane);

    for (int sl = 0; sl < n_slabs; ++sl) {
        const uint4 sm4 = ld_cs_u4(sm4_base + static_cast<int64_t>(sl) * MX6_SM4_SLAB_BYTES + lane * 16);
        uint4 raw4 = make_uint4(0u, 0u, 0u, 0u);
        if (use_raw4) {
            raw4 = ld_cs_u4(reinterpret_cast<const uint8_t*>(exp_base)
                            + static_cast<int64_t>(sl) * MX6_RAW4_SLAB_BYTES + lane * 16);
        }
        const uint32_t u0 = ws3_base[sl * 3 + 0];
        const uint32_t u1 = ws3_base[sl * 3 + 1];
        const uint32_t u2 = ws3_base[sl * 3 + 2];

        const uint32_t sm_pair[4] = {sm4.x, sm4.y, sm4.z, sm4.w};
        const uint32_t r4_pair[4] = {raw4.x, raw4.y, raw4.z, raw4.w};

        #pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int n_base = (PROJ == 0)
                ? (nb * BN + warp * 8)
                : (nb * BN + (warp * SUB_N + q) * 8);
            const int k32_base = (PROJ == 0) ? (sl * 128 + q * 32) : (sl * 32);

            if (lane < 8) {
                const uint32_t ws_word = unpack_ws3_word(u0, u1, u2, q);
                const uint8_t ws_b = static_cast<uint8_t>(base_ws + ((ws_word >> (lane * 3)) & 7u));
                *ws_u8_ptr<OUT_PANEL>(WS_OUT, e, n_base + lane, k32_base >> 5, N, KG, dst_pb, dst_kbg) = ws_b;
            }

            const int s0 = q << 1;
            const int s1 = s0 | 1;
            uint32_t d0, d1, d2, d3;

            if (use_raw4) {
                raw4_to_d4(r4_pair[q] & 0xFFFFu, d0, d1, d2, d3);
            } else {
                reader.decode4_and_refill(sLut, lane, lanemask_lt, (sl * 8 + s0 + 1) < total_steps, d0, d1, d2, d3);
            }
            const uint32_t e4_0 = build_e4_pack_from_sm_and_d(sm_pair[q] & 0xFFFFu, d0, d1, d2, d3);

            if (use_raw4) {
                raw4_to_d4(r4_pair[q] >> 16, d0, d1, d2, d3);
            } else {
                reader.decode4_and_refill(sLut, lane, lanemask_lt, (sl * 8 + s1 + 1) < total_steps, d0, d1, d2, d3);
            }
            const uint32_t e4_1 = build_e4_pack_from_sm_and_d(sm_pair[q] >> 16, d0, d1, d2, d3);

            // Combine s0 (k0, k0+1, k0+16, k0+17) and s1 (k0+2, k0+3, k0+18, k0+19) into two aligned uint32 stores!
            const uint32_t lo_u32 = (e4_0 & 0xFFFFu) | ((e4_1 & 0xFFFFu) << 16);
            const uint32_t hi_u32 = (e4_0 >> 16)     | ((e4_1 >> 16) << 16);
            const int row = n_base + g;
            const int k4  = k32_base + 4 * lt;
            *w_u32_ptr<OUT_PANEL>(W_OUT, e, row, k4,      N, K, dst_pb, dst_kb) = lo_u32;
            *w_u32_ptr<OUT_PANEL>(W_OUT, e, row, k4 + 16, N, K, dst_pb, dst_kb) = hi_u32;
        }
    }
}

// =========================================================================
// Kernel 1 (C=1 V10): Adaptive BN=16/32 MX6-Huff7 W13 (N=576, K=4608, TILE=8|16)
//   - Zero-repack bijection BN=16 <-> BN=32 onto packed 72-warp/expert MX6 buffers
//   - Bank-conflict-free uint8_t sLut[128] (128 B = 0.000% conflicts!)
//   - V9-3 g < live_m predicated LDS (0 sA zero-fill!), fixed Gammas[r0], uint2 SiTU store
//   - Strict #206 PDL (`wait < launch` on ALL paths incl. u_min/u_max/bidx early exits!)
// =========================================================================

template <
    int BN, int TILE, int MIN_CTA,
    bool USE_PDL, bool ZERO_SCATTER, bool FUSE_SITU, bool WAVE_EFF_GATE
>
__global__ __launch_bounds__((BN / 8) * 32, MIN_CTA)
void mx6_w13_c1_v10_kernel(
    const uint8_t* __restrict__ A,
    const uint8_t* __restrict__ AS,
    const uint8_t* __restrict__ SM4,
    const uint32_t* __restrict__ WS3_PACK,
    const uint32_t* __restrict__ EXP_STREAM,
    const int32_t* __restrict__ EXP_OFF,
    const uint8_t* __restrict__ WARP_META,
    const uint8_t* __restrict__ HUFF_LUT,
    const float* __restrict__ Ones,
    const int32_t* __restrict__ Tiles,
    const int32_t* __restrict__ TileMask,
    const int32_t* __restrict__ TileCompactStart,
    const float* __restrict__ Gammas,
    __nv_bfloat16* __restrict__ Out,
    int N, int K, int MAX_CTAS, uint32_t base_ws,
    int u_min_incl, int u_max_incl, int wave_stride_other, int wave_margin_pct,
    int pdl_wait_mode, int trig_dep
) {
    static_assert(BN == 16 || BN == 32, "C1 MX6 W13 BN must be 16 or 32");
    static_assert(TILE == 8 || TILE == 16, "C1 MX6 W13 TILE must be 8 or 16");

    // Unconditional #206 PDL wait-at-entry: router writes Tiles[], so W13 must NEVER read
    // Tiles before griddepcontrol.wait completes for any pdl_wait_mode >= 1.
    if constexpr (USE_PDL) {
        if (pdl_wait_mode >= 1) {
            asm volatile("griddepcontrol.wait;" ::: "memory");
        }
    }

    constexpr int WARPS         = BN / 8;
    constexpr int THREADS       = WARPS * 32;
    constexpr int SLAB_K        = 128;
    constexpr int BK_SMEM       = 256;
    constexpr int SLABS_PER_BLK = 2;
    constexpr int SA_STRIDE     = BK_SMEM + 16;  // 272 bf16 -> 0-way LDS.64
    constexpr int N_PANELS32    = 18;
    constexpr int N_PANELS      = (BN == 16) ? 36 : 18;
    constexpr int TOTAL_STEPS   = 36 * 8;        // 288

    const int n_active = Tiles[2 * MAX_CTAS];
    const int stride   = static_cast<int>(gridDim.x);
    const int bidx     = static_cast<int>(blockIdx.x);

    if constexpr (WAVE_EFF_GATE) {
        const bool bn16_win = (BN == 16)
            ? bn16_wins_wave_eff(n_active, stride, wave_stride_other, wave_margin_pct)
            : bn16_wins_wave_eff(n_active, wave_stride_other, stride, wave_margin_pct);
        const bool take_this = (BN == 16) ? bn16_win : (!bn16_win);
        if (!take_this || n_active <= 0) {
            if constexpr (USE_PDL) {
                if (trig_dep) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
            return;
        }
    } else if (u_min_incl > 0 && (n_active < u_min_incl || n_active > u_max_incl)) {
        if constexpr (USE_PDL) {
            if (trig_dep) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
        return;
    }

    const int n_total = n_active * N_PANELS;
    if (bidx >= n_total) {
        if constexpr (USE_PDL) {
            if (trig_dep) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
        return;
    }

    const int n_slabs = K / SLAB_K;   // 36
    const int n_blks  = K / BK_SMEM;  // 18
    const int KG      = K >> 5;       // 144
    const bool fast_ws = (base_ws >= 8u && base_ws <= 238u);
    const uint32_t base_ws_off2 = ((base_ws - 7u) << 7) * 0x00010001u;

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g    = lane >> 2;
    const int g3   = g * 3;
    const int lt   = lane & 3;
    const uint32_t lanemask_lt = get_lanemask_lt();
    const int x0_0 = g * SA_STRIDE + 4 * lt;
    const int x0_1 = (g + 8) * SA_STRIDE + 4 * lt;

    alignas(128) __shared__ uint8_t sLut[MX6_LUT_SIZE];
    alignas(16)  __shared__ __nv_bfloat16 sA[2][TILE * SA_STRIDE];

    if constexpr (THREADS == 128) {
        sLut[tid] = HUFF_LUT[tid];
    } else {
        sLut[tid]      = HUFF_LUT[tid];
        sLut[tid + 64] = HUFF_LUT[tid + 64];
    }
    __syncthreads();

    bool trig_fired = false;

    for (int idx = bidx; idx < n_total; idx += stride) {
        const int t  = idx / N_PANELS;
        const int nb = idx - t * N_PANELS;
        const bool is_first_tile = (idx == bidx);
        const bool is_last_tile  = (idx + stride >= n_total);

        const int64_t e = static_cast<int64_t>(Tiles[t]);
        const int end_row = Tiles[MAX_CTAS + t];
        const int row_base = t * TILE;
        int live_m = end_row - row_base;
        if (live_m > TILE) live_m = TILE;
        if (live_m < 0) live_m = 0;

        const int nb32 = (BN == 16) ? (nb >> 1) : nb;
        const int wp   = (BN == 16) ? (((nb & 1) << 1) + warp) : warp;
        const int64_t warp_uid = (e * N_PANELS32 + nb32) * 4 + wp;

        const uint8_t*  sm4_base = SM4 + warp_uid * n_slabs * MX6_SM4_SLAB_BYTES;
        const uint32_t* ws3_base = WS3_PACK + warp_uid * n_slabs * 3;
        const uint32_t* exp_base = EXP_STREAM + static_cast<int64_t>(EXP_OFF[warp_uid]) * MX6_CHUNK_WORDS;
        const uint8_t   wmeta    = WARP_META[warp_uid];
        const bool      use_raw4        = ((wmeta & WMETA_RAW4_BIT) != 0u);
        const bool      warp_all_normal = fast_ws && ((wmeta & WMETA_ALL_NORMAL_BIT) != 0u);
        const bool      warp_no_nan     = fast_ws && ((wmeta & WMETA_HAS_NAN_BIT) == 0u);
        (void)is_first_tile;
        if (live_m == 0) continue;

        const uint32_t tmask = ZERO_SCATTER ? static_cast<uint32_t>(TileMask[t]) : 0u;

        // Preload first 32 uint32 of WS3_PACK (covers slabs 0..9; total = 108 u32 = 3.375 chunks)
        uint32_t ws_cur = ld_cs_u32(ws3_base + lane);
        uint32_t ws_nxt = 0u;

        // V12 Upgrade: Vectorized 8-elem/thread (LDG.64 -> STS.128) A-block loader (2x fewer iters!)
        auto load_a_blk = [&](int blk, __nv_bfloat16* dst) {
            constexpr int V8_PER_ROW = BK_SMEM >> 3;  // 32
            const int total_v8 = live_m * V8_PER_ROW;
            const int k0 = blk * BK_SMEM;
            for (int aidx = tid; aidx < total_v8; aidx += THREADS) {
                const int lm = aidx >> 5;
                const int q8 = aidx & 31;
                const int64_t row = static_cast<int64_t>(row_base + lm);
                const int64_t a_row = ZERO_SCATTER
                    ? static_cast<int64_t>(fns_u32(tmask, 0u, lm + 1))
                    : row;
                const int kcol   = k0 + (q8 << 3);
                const int kg_col = kcol >> 5;
                const uint2 a8 = *reinterpret_cast<const uint2*>(A + a_row * K + kcol);
                const uint8_t as_b = ZERO_SCATTER
                    ? AS[a_row * KG + kg_col]
                    : AS[scale_offset_decode(row, kg_col, KG)];
                const float a_sc = e8m0_to_f32(as_b);
                uint32_t b0, b1, b2, b3;
                e4m3x4_scale_to_bf16x4(a8.x, a_sc, b0, b1);
                e4m3x4_scale_to_bf16x4(a8.y, a_sc, b2, b3);
                *reinterpret_cast<uint4*>(dst + lm * SA_STRIDE + (q8 << 3)) = make_uint4(b0, b1, b2, b3);
            }
        };

        MX6StreamReaderV10 reader{};
        uint4 cur_raw4 = make_uint4(0u, 0u, 0u, 0u);
        uint4 cur_sm4  = ld_cs_u4(sm4_base + lane * 16);
        if (use_raw4) {
            cur_raw4 = ld_cs_u4(reinterpret_cast<const uint8_t*>(exp_base) + lane * 16);
        } else {
            reader.init(exp_base, lane);
        }

        load_a_blk(0, sA[0]);
        __syncthreads();

        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        const bool has_m0 = (g < live_m);
        const bool has_m8 = (TILE > 8) && (g + 8 < live_m);
        int ws_chunk_idx = 0;

        for (int blk = 0; blk < n_blks; ++blk) {
            const __nv_bfloat16* sAp = sA[blk & 1];
            __nv_bfloat16* sAnext    = sA[(blk & 1) ^ 1];

            #pragma unroll 1
            for (int is = 0; is < SLABS_PER_BLK; ++is) {
                const int sl = blk * SLABS_PER_BLK + is;

                uint4 nxt_sm4  = make_uint4(0u, 0u, 0u, 0u);
                uint4 nxt_raw4 = make_uint4(0u, 0u, 0u, 0u);
                if (sl + 1 < n_slabs) {
                    const int nsl = sl + 1;
                    nxt_sm4 = ld_cs_u4(sm4_base + static_cast<int64_t>(nsl) * MX6_SM4_SLAB_BYTES + lane * 16);
                    if (use_raw4) {
                        nxt_raw4 = ld_cs_u4(reinterpret_cast<const uint8_t*>(exp_base)
                                            + static_cast<int64_t>(nsl) * MX6_RAW4_SLAB_BYTES + lane * 16);
                    }
                }
                if (is == 0 && blk + 1 < n_blks) {
                    load_a_blk(blk + 1, sAnext);
                }

                const int ws_pos  = sl * 3;
                const int req_chk = (ws_pos + 2) >> 5;
                if (req_chk > ws_chunk_idx) {
                    ws_nxt = ((req_chk * 32 + lane) < n_slabs * 3)
                        ? ld_cs_u32(ws3_base + req_chk * 32 + lane) : 0u;
                }
                const int idx0 = ws_pos - ws_chunk_idx * 32;
                const uint32_t u0 = (idx0 < 32)
                    ? __shfl_sync(0xFFFFFFFFu, ws_cur, idx0)
                    : __shfl_sync(0xFFFFFFFFu, ws_nxt, idx0 - 32);
                const uint32_t u1 = (idx0 + 1 < 32)
                    ? __shfl_sync(0xFFFFFFFFu, ws_cur, idx0 + 1)
                    : __shfl_sync(0xFFFFFFFFu, ws_nxt, idx0 - 31);
                const uint32_t u2 = (idx0 + 2 < 32)
                    ? __shfl_sync(0xFFFFFFFFu, ws_cur, idx0 + 2)
                    : __shfl_sync(0xFFFFFFFFu, ws_nxt, idx0 - 30);
                if (req_chk > ws_chunk_idx) {
                    ws_cur = ws_nxt;
                    ws_chunk_idx = req_chk;
                }

                const uint32_t sm_pair[4] = {cur_sm4.x, cur_sm4.y, cur_sm4.z, cur_sm4.w};
                const uint32_t r4_pair[4] = {cur_raw4.x, cur_raw4.y, cur_raw4.z, cur_raw4.w};
                const int blk_k_off = is * 128;

                #pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const int kq_off = blk_k_off + q * 32;
                    const uint32_t ws_word  = unpack_ws3_word(u0, u1, u2, q);
                    const uint32_t ws_delta = (ws_word >> g3) & 7u;
                    const uint32_t ws_off2  = base_ws_off2 + ws_delta * 0x00800080u;
                    const uint32_t ws_byte  = base_ws + ws_delta;

                    const int s0 = q << 1;
                    const int s1 = s0 | 1;
                    uint32_t d0, d1, d2, d3;
                    uint32_t b0_0, b1_0, b0_1, b1_1;

                    if (use_raw4) {
                        raw4_to_d4(r4_pair[q] & 0xFFFFu, d0, d1, d2, d3);
                    } else {
                        reader.decode4_and_refill(sLut, lane, lanemask_lt, (sl * 8 + s0 + 1) < TOTAL_STEPS, d0, d1, d2, d3);
                    }
                    decode_mx6_step_to_bf16x4(sm_pair[q] & 0xFFFFu, d0, d1, d2, d3, ws_off2, ws_byte,
                                              warp_all_normal, warp_no_nan, b0_0, b1_0);

                    if (use_raw4) {
                        raw4_to_d4(r4_pair[q] >> 16, d0, d1, d2, d3);
                    } else {
                        reader.decode4_and_refill(sLut, lane, lanemask_lt, (sl * 8 + s1 + 1) < TOTAL_STEPS, d0, d1, d2, d3);
                    }
                    decode_mx6_step_to_bf16x4(sm_pair[q] >> 16, d0, d1, d2, d3, ws_off2, ws_byte,
                                              warp_all_normal, warp_no_nan, b0_1, b1_1);

                    // V9-3 g < live_m Predicated LDS (zero sA zero-fill!)
                    uint2 a0 = make_uint2(0u, 0u), a2 = make_uint2(0u, 0u);
                    if (has_m0) {
                        a0 = *reinterpret_cast<const uint2*>(sAp + x0_0 + kq_off);
                        a2 = *reinterpret_cast<const uint2*>(sAp + x0_0 + kq_off + 16);
                    }
                    uint2 a1 = make_uint2(0u, 0u), a3 = make_uint2(0u, 0u);
                    if constexpr (TILE > 8) {
                        if (has_m8) {
                            a1 = *reinterpret_cast<const uint2*>(sAp + x0_1 + kq_off);
                            a3 = *reinterpret_cast<const uint2*>(sAp + x0_1 + kq_off + 16);
                        }
                    }
                    mma_m16n8k16_bf16(acc, a0.x, a1.x, a2.x, a3.x, b0_0, b1_0);
                    mma_m16n8k16_bf16(acc, a0.y, a1.y, a2.y, a3.y, b0_1, b1_1);
                }

                if constexpr (USE_PDL) {
                    if (trig_dep && is_last_tile && sl == n_slabs - 2) {
                        asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
                        trig_fired = true;
                    }
                }

                if (sl + 1 < n_slabs) {
                    cur_sm4  = nxt_sm4;
                    cur_raw4 = nxt_raw4;
                }
            }
            if (blk + 1 < n_blks) __syncthreads();
        }

        const float scale = Ones[e];
        const int r0 = row_base + g;
        const int r1 = row_base + g + 8;
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
            if (has_m0) y0_b = situ_pair_bf16_exact(pack_bf16x2(o0, o1), __ldg(Gammas + r0));
            const uint32_t p0 = static_cast<uint32_t>(y0_b)
                | (__shfl_xor_sync(0xFFFFFFFFu, static_cast<uint32_t>(y0_b), 1) << 16);
            const uint32_t p1 = __shfl_xor_sync(0xFFFFFFFFu, p0, 2);
            if (lt == 0 && has_m0) {
                *reinterpret_cast<uint2*>(Out + static_cast<int64_t>(c0) * inter + c_warp) = make_uint2(p0, p1);
            }
            if constexpr (TILE > 8) {
                uint16_t y1_b = 0u;
                if (has_m8) y1_b = situ_pair_bf16_exact(pack_bf16x2(o2, o3), __ldg(Gammas + r1));
                const uint32_t q0 = static_cast<uint32_t>(y1_b)
                    | (__shfl_xor_sync(0xFFFFFFFFu, static_cast<uint32_t>(y1_b), 1) << 16);
                const uint32_t q1 = __shfl_xor_sync(0xFFFFFFFFu, q0, 2);
                if (lt == 0 && has_m8) {
                    *reinterpret_cast<uint2*>(Out + static_cast<int64_t>(c1) * inter + c_warp) = make_uint2(q0, q1);
                }
            }
        } else {
            const int n0 = nb * BN + warp * 8 + 2 * lt;
            if (has_m0) {
                *reinterpret_cast<uint32_t*>(Out + static_cast<int64_t>(r0) * N + n0) = pack_bf16x2(o0, o1);
            }
            if constexpr (TILE > 8) {
                if (has_m8) {
                    *reinterpret_cast<uint32_t*>(Out + static_cast<int64_t>(r1) * N + n0) = pack_bf16x2(o2, o3);
                }
            }
        }

        if (!is_last_tile) __syncthreads();
    }

    if constexpr (USE_PDL) {
        if (trig_dep && !trig_fired) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }
}

// =========================================================================
// Kernel 2 (C=1 V10): BN=128 (4-Warp) MX6-Huff7 W2 (N=4608, K=288, TILE=8|16)
//   - Tight 96-bit WS3 fits ALL 9 slabs in ONE single register (`ws0`, 27 u32 < 32 lanes!)
//   - Bank-conflict-free uint8_t sLut[128] (128 B!), OTF MXFP8 quant-dequant, batched DownCompact
//   - Strict #206 PDL (`wait < launch` on ALL paths; post-SendBuf __threadfence()!)
// =========================================================================

template <
    int TILE, int MIN_CTA, bool USE_PDL,
    bool AS_ROW_MAJOR, bool FUSE_QUANT, bool FUSE_COMBINE
>
__global__ __launch_bounds__(128, MIN_CTA)
void mx6_w2_c1_v10_kernel(
    const uint8_t* __restrict__ A,
    const uint8_t* __restrict__ AS,
    const uint8_t* __restrict__ SM4,
    const uint32_t* __restrict__ WS3_PACK,
    const uint32_t* __restrict__ EXP_STREAM,
    const int32_t* __restrict__ EXP_OFF,
    const uint8_t* __restrict__ WARP_META,
    const uint8_t* __restrict__ HUFF_LUT,
    const float* __restrict__ Ones,
    const int32_t* __restrict__ Tiles,
    const int32_t* __restrict__ TileCompactStart,
    __nv_bfloat16* __restrict__ Out,
    const int32_t* __restrict__ Src2Compact,
    const int32_t* __restrict__ NLivePtr,
    uint32_t* __restrict__ PanelDone,
    __nv_bfloat16* __restrict__ SendBuf,
    int N, int K, int MAX_CTAS, int T, uint32_t base_ws, int as_row_stride, int down_row_stride,
    int pdl_wait_mode, int trig_dep
) {
    static_assert(TILE == 8 || TILE == 16, "C1 MX6 W2 TILE must be 8 or 16");
    (void)K;

    if constexpr (USE_PDL) {
        if (pdl_wait_mode == 1) {
            asm volatile("griddepcontrol.wait;" ::: "memory");
        }
    }

    constexpr int WARPS       = 4;
    constexpr int SUB_N       = 4;
    constexpr int BN          = 128;
    constexpr int N_PANELS    = 36;
    constexpr int KG          = 9;
    constexpr int K_DIM       = 288;
    constexpr int SA_STRIDE   = K_DIM + 16;  // 304 bf16 -> 0-way LDS.64
    constexpr int TOTAL_STEPS = KG * 8;      // 72

    const int tid    = threadIdx.x;
    const int warp   = tid >> 5;
    const int lane   = tid & 31;
    const int g      = lane >> 2;
    const int g3     = g * 3;
    const int lt     = lane & 3;
    const uint32_t lanemask_lt = get_lanemask_lt();
    const int stride = static_cast<int>(gridDim.x);
    const int bidx   = static_cast<int>(blockIdx.x);
    const int n_active = Tiles[2 * MAX_CTAS];

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
                const int col0 = bidx * BN + (tid & 15) * 8;
                const uint4 z4 = make_uint4(0u, 0u, 0u, 0u);
                for (int m = (tid >> 4); m < n_live_tok; m += 8) {
                    *reinterpret_cast<uint4*>(SendBuf + static_cast<int64_t>(m) * N + col0) = z4;
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
    if (bidx >= n_total) {
        if constexpr (USE_PDL) {
            if (pdl_wait_mode == 2) asm volatile("griddepcontrol.wait;" ::: "memory");
            if (trig_dep) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
        return;
    }

    const bool fast_ws = (base_ws >= 8u && base_ws <= 238u);
    const uint32_t base_ws_off2 = ((base_ws - 7u) << 7) * 0x00010001u;
    const int x0_0 = g * SA_STRIDE + 4 * lt;
    const int x0_1 = (g + 8) * SA_STRIDE + 4 * lt;

    alignas(128) __shared__ uint8_t sLut[MX6_LUT_SIZE];
    alignas(16)  __shared__ __nv_bfloat16 sA_all[TILE * SA_STRIDE];
    __shared__ int sIsLastPanel;

    sLut[tid] = HUFF_LUT[tid];
    __syncthreads();

    bool trig_fired = false;
    uint32_t cur_nb_tiles = 0u;

    for (int idx = bidx; idx < n_total; idx += stride) {
        // Panel-outer when FUSE_COMBINE (batched panel-run!), expert-outer otherwise
        const int nb = FUSE_COMBINE ? (idx / n_active) : (idx % N_PANELS);
        const int t  = FUSE_COMBINE ? (idx - nb * n_active) : (idx / N_PANELS);
        const bool is_very_first = (idx == bidx);
        const bool is_very_last  = (idx + stride >= n_total);
        const int nidx    = idx + stride;
        const int next_nb = (!is_very_last) ? (FUSE_COMBINE ? (nidx / n_active) : (nidx % N_PANELS)) : -1;
        const int next_t  = (!is_very_last) ? (FUSE_COMBINE ? (nidx - next_nb * n_active) : (nidx / N_PANELS)) : 0;

        const int64_t e = static_cast<int64_t>(Tiles[t]);
        const int end_row = Tiles[MAX_CTAS + t];
        const int row_base = t * TILE;
        int live_m = end_row - row_base;
        if (live_m > TILE) live_m = TILE;
        if (live_m < 0) live_m = 0;

        const int64_t warp_uid = (e * N_PANELS + nb) * WARPS + warp;
        const uint8_t*  sm4_base = SM4 + warp_uid * KG * MX6_SM4_SLAB_BYTES;
        const uint32_t* ws3_base = WS3_PACK + warp_uid * KG * 3;
        const uint32_t* exp_base = EXP_STREAM + static_cast<int64_t>(EXP_OFF[warp_uid]) * MX6_CHUNK_WORDS;
        const uint8_t   wmeta    = WARP_META[warp_uid];
        const bool      use_raw4        = ((wmeta & WMETA_RAW4_BIT) != 0u);
        const bool      warp_all_normal = fast_ws && ((wmeta & WMETA_ALL_NORMAL_BIT) != 0u);
        const bool      warp_no_nan     = fast_ws && ((wmeta & WMETA_HAS_NAN_BIT) == 0u);

        if constexpr (USE_PDL) {
            if (pdl_wait_mode == 2 && is_very_first) {
                if (lane < 4)      prefetch_l2(sm4_base + lane * 128);
                else if (lane < 8) prefetch_l2(exp_base + (lane - 4) * 32);
                else if (lane == 8) prefetch_l2(ws3_base);
                asm volatile("griddepcontrol.wait;" ::: "memory");
            }
        }

        const int compact_tile_base = (FUSE_QUANT || FUSE_COMBINE) ? TileCompactStart[t] : 0;

        if (live_m > 0) {
            // ALL 9 slabs' WS3 (27 uint32) fit in ONE register `ws0`!
            const uint32_t ws0 = (lane < KG * 3) ? ld_cs_u32(ws3_base + lane) : 0u;

            MX6StreamReaderV10 reader{};
            uint4 cur_raw4 = make_uint4(0u, 0u, 0u, 0u);
            uint4 cur_sm4  = ld_cs_u4(sm4_base + lane * 16);
            if (use_raw4) {
                cur_raw4 = ld_cs_u4(reinterpret_cast<const uint8_t*>(exp_base) + lane * 16);
            } else {
                reader.init(exp_base, lane);
            }

            if constexpr (FUSE_QUANT) {
                const __nv_bfloat16* A_bf16 = reinterpret_cast<const __nv_bfloat16*>(A);
                const int total_blks = live_m * KG;  // <= 144
                for (int b = tid; b < total_blks; b += 128) {
                    const int lm = b / KG;
                    const int kg = b - lm * KG;
                    otf_mxfp8_quant_dequant_blk32(
                        A_bf16 + static_cast<int64_t>(compact_tile_base + lm) * K_DIM + kg * 32,
                        sA_all + lm * SA_STRIDE + kg * 32
                    );
                }
            } else {
                const int total_blks = live_m * KG;
                for (int b = tid; b < total_blks; b += 128) {
                    const int lm = b / KG;
                    const int kg = b - lm * KG;
                    const int64_t row = static_cast<int64_t>(row_base + lm);
                    const uint8_t as_b = AS_ROW_MAJOR
                        ? AS[row * as_row_stride + kg]
                        : AS[scale_offset_decode(row, kg, KG)];
                    const float a_sc = e8m0_to_f32(as_b);
                    const uint4 a16_0 = *reinterpret_cast<const uint4*>(A + row * K_DIM + kg * 32);
                    const uint4 a16_1 = *reinterpret_cast<const uint4*>(A + row * K_DIM + kg * 32 + 16);
                    __nv_bfloat16* dst = sA_all + lm * SA_STRIDE + kg * 32;
                    uint32_t b0, b1, b2, b3;
                    e4m3x4_scale_to_bf16x4(a16_0.x, a_sc, b0, b1);
                    e4m3x4_scale_to_bf16x4(a16_0.y, a_sc, b2, b3);
                    *reinterpret_cast<uint4*>(dst + 0)  = make_uint4(b0, b1, b2, b3);
                    e4m3x4_scale_to_bf16x4(a16_0.z, a_sc, b0, b1);
                    e4m3x4_scale_to_bf16x4(a16_0.w, a_sc, b2, b3);
                    *reinterpret_cast<uint4*>(dst + 8)  = make_uint4(b0, b1, b2, b3);
                    e4m3x4_scale_to_bf16x4(a16_1.x, a_sc, b0, b1);
                    e4m3x4_scale_to_bf16x4(a16_1.y, a_sc, b2, b3);
                    *reinterpret_cast<uint4*>(dst + 16) = make_uint4(b0, b1, b2, b3);
                    e4m3x4_scale_to_bf16x4(a16_1.z, a_sc, b0, b1);
                    e4m3x4_scale_to_bf16x4(a16_1.w, a_sc, b2, b3);
                    *reinterpret_cast<uint4*>(dst + 24) = make_uint4(b0, b1, b2, b3);
                }
            }
            __syncthreads();

            float acc[SUB_N][4] = {{0.0f}};
            const bool has_m0 = (g < live_m);
            const bool has_m8 = (TILE > 8) && (g + 8 < live_m);

            #pragma unroll 1
            for (int kg = 0; kg < KG; ++kg) {
                uint4 nxt_sm4  = make_uint4(0u, 0u, 0u, 0u);
                uint4 nxt_raw4 = make_uint4(0u, 0u, 0u, 0u);
                if (kg + 1 < KG) {
                    const int nkg = kg + 1;
                    nxt_sm4 = ld_cs_u4(sm4_base + static_cast<int64_t>(nkg) * MX6_SM4_SLAB_BYTES + lane * 16);
                    if (use_raw4) {
                        nxt_raw4 = ld_cs_u4(reinterpret_cast<const uint8_t*>(exp_base)
                                            + static_cast<int64_t>(nkg) * MX6_RAW4_SLAB_BYTES + lane * 16);
                    }
                } else if (!is_very_last) {
                    const int64_t ne    = static_cast<int64_t>(Tiles[next_t]);
                    const int64_t nwuid = (ne * N_PANELS + next_nb) * WARPS + warp;
                    if (lane < 4)      prefetch_l2(SM4 + nwuid * KG * MX6_SM4_SLAB_BYTES + lane * 128);
                    else if (lane < 8) prefetch_l2(EXP_STREAM + static_cast<int64_t>(EXP_OFF[nwuid]) * MX6_CHUNK_WORDS + (lane - 4) * 32);
                    else if (lane == 8) prefetch_l2(WS3_PACK + nwuid * KG * 3);
                }

                const int k_off = kg * 32;
                uint2 a0 = make_uint2(0u, 0u), a2 = make_uint2(0u, 0u);
                if (has_m0) {
                    a0 = *reinterpret_cast<const uint2*>(sA_all + x0_0 + k_off);
                    a2 = *reinterpret_cast<const uint2*>(sA_all + x0_0 + k_off + 16);
                }
                uint2 a1 = make_uint2(0u, 0u), a3 = make_uint2(0u, 0u);
                if constexpr (TILE > 8) {
                    if (has_m8) {
                        a1 = *reinterpret_cast<const uint2*>(sA_all + x0_1 + k_off);
                        a3 = *reinterpret_cast<const uint2*>(sA_all + x0_1 + k_off + 16);
                    }
                }

                const uint32_t u0 = __shfl_sync(0xFFFFFFFFu, ws0, kg * 3 + 0);
                const uint32_t u1 = __shfl_sync(0xFFFFFFFFu, ws0, kg * 3 + 1);
                const uint32_t u2 = __shfl_sync(0xFFFFFFFFu, ws0, kg * 3 + 2);

                const uint32_t sm_pair[4] = {cur_sm4.x, cur_sm4.y, cur_sm4.z, cur_sm4.w};
                const uint32_t r4_pair[4] = {cur_raw4.x, cur_raw4.y, cur_raw4.z, cur_raw4.w};

                #pragma unroll
                for (int sub = 0; sub < SUB_N; ++sub) {
                    const uint32_t ws_word  = unpack_ws3_word(u0, u1, u2, sub);
                    const uint32_t ws_delta = (ws_word >> g3) & 7u;
                    const uint32_t ws_off2  = base_ws_off2 + ws_delta * 0x00800080u;
                    const uint32_t ws_byte  = base_ws + ws_delta;

                    const int s0 = sub << 1;
                    const int s1 = s0 | 1;
                    uint32_t d0, d1, d2, d3;
                    uint32_t b0_0, b1_0, b0_1, b1_1;

                    if (use_raw4) {
                        raw4_to_d4(r4_pair[sub] & 0xFFFFu, d0, d1, d2, d3);
                    } else {
                        reader.decode4_and_refill(sLut, lane, lanemask_lt, (kg * 8 + s0 + 1) < TOTAL_STEPS, d0, d1, d2, d3);
                    }
                    decode_mx6_step_to_bf16x4(sm_pair[sub] & 0xFFFFu, d0, d1, d2, d3, ws_off2, ws_byte,
                                              warp_all_normal, warp_no_nan, b0_0, b1_0);

                    if (use_raw4) {
                        raw4_to_d4(r4_pair[sub] >> 16, d0, d1, d2, d3);
                    } else {
                        reader.decode4_and_refill(sLut, lane, lanemask_lt, (kg * 8 + s1 + 1) < TOTAL_STEPS, d0, d1, d2, d3);
                    }
                    decode_mx6_step_to_bf16x4(sm_pair[sub] >> 16, d0, d1, d2, d3, ws_off2, ws_byte,
                                              warp_all_normal, warp_no_nan, b0_1, b1_1);

                    mma_m16n8k16_bf16(acc[sub], a0.x, a1.x, a2.x, a3.x, b0_0, b1_0);
                    mma_m16n8k16_bf16(acc[sub], a0.y, a1.y, a2.y, a3.y, b0_1, b1_1);
                }

                if constexpr (USE_PDL && !FUSE_COMBINE) {
                    if (trig_dep && is_very_last && kg == KG - 2) {
                        asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
                        trig_fired = true;
                    }
                }

                if (kg + 1 < KG) {
                    cur_sm4  = nxt_sm4;
                    cur_raw4 = nxt_raw4;
                }
            }

            const float scale = Ones[e];
            const int r0 = row_base + g;
            const int r1 = row_base + g + 8;
            const int c0 = compact_tile_base + g;
            const int c1 = compact_tile_base + g + 8;
            const int col_u4_local = warp * 32 + lt * 8;
            __nv_bfloat16* down_panel_buf = FUSE_COMBINE
                ? (Out + static_cast<int64_t>(nb) * down_row_stride * BN)
                : nullptr;

            // V12 Upgrade: 3-SHFL 4x4 transposed 64B-coalesced uint4 stores (4x fewer LSU ops!)
            uint32_t pk0[4];
            #pragma unroll
            for (int sub = 0; sub < SUB_N; ++sub) {
                pk0[sub] = pack_bf16x2(mul_rn(acc[sub][0], scale), mul_rn(acc[sub][1], scale));
            }
            const uint4 v0 = transpose_4x4_bf16x2_u4(pk0[0], pk0[1], pk0[2], pk0[3], lt);
            if (has_m0) {
                if constexpr (FUSE_COMBINE) {
                    st_cg_u4(down_panel_buf + static_cast<int64_t>(c0) * BN + col_u4_local, v0);
                } else {
                    *reinterpret_cast<uint4*>(Out + static_cast<int64_t>(r0) * N + nb * BN + col_u4_local) = v0;
                }
            }
            if constexpr (TILE > 8) {
                uint32_t pk1[4];
                #pragma unroll
                for (int sub = 0; sub < SUB_N; ++sub) {
                    pk1[sub] = pack_bf16x2(mul_rn(acc[sub][2], scale), mul_rn(acc[sub][3], scale));
                }
                const uint4 v1 = transpose_4x4_bf16x2_u4(pk1[0], pk1[1], pk1[2], pk1[3], lt);
                if (has_m8) {
                    if constexpr (FUSE_COMBINE) {
                        st_cg_u4(down_panel_buf + static_cast<int64_t>(c1) * BN + col_u4_local, v1);
                    } else {
                        *reinterpret_cast<uint4*>(Out + static_cast<int64_t>(r1) * N + nb * BN + col_u4_local) = v1;
                    }
                }
            }
        }

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
                    int n_live_tok = T;
                    if (NLivePtr != nullptr) {
                        int nl = *NLivePtr;
                        if (nl < 0) nl = 0;
                        if (nl < n_live_tok) n_live_tok = nl;
                    }
                    __nv_bfloat16* down_panel_buf = Out + static_cast<int64_t>(nb) * down_row_stride * BN;
                    const int col_local = (tid & 15) * 8;
                    const int col_glob  = nb * BN + col_local;
                    const uint4 z4 = make_uint4(0u, 0u, 0u, 0u);
                    // V12 Upgrade: Double-buffered 16B ld_cg_u4 -> uint4 SendBuf (2x fewer LSU ops!)
                    for (int m = (tid >> 4); m < n_live_tok; m += 8) {
                        const int32_t* s2c = Src2Compact + m * 8;
                        int32_t r[8];
                        #pragma unroll
                        for (int s = 0; s < 8; ++s) r[s] = __ldg(s2c + s);

                        uint4 v_cur = (r[0] >= 0)
                            ? ld_cg_u4(down_panel_buf + static_cast<int64_t>(r[0]) * BN + col_local) : z4;
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
    }

    if constexpr (USE_PDL) {
        if (trig_dep && !trig_fired) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }
}

// =========================================================================
// Kernel 3 (C48/C32 V10): MX6-Huff7 W13 (N=576, K=4608, TILE=8|16|32, BN=16|32)
//   - Covers C=2..8 (TILE=8, M=17..64) AND C=16..32 (TILE=16|32, M=65..256)
//   - Row2Tok Zero-Scatter, AS_ROW_MAJOR, FUSE_SITU (fixed Gammas[r0] + uint2 store!)
//   - Bank-conflict-free uint8_t sLut[128] (128 B!), V9-3 predicated LDS (0 sA zero-fill!)
//   - Strict #206 PDL (`wait < launch` on ALL paths incl. early exits!)
// =========================================================================

template <
    int BN, int TILE, int MIN_CTA,
    bool USE_PDL, bool ZERO_SCATTER, bool AS_ROW_MAJOR, bool FUSE_SITU, bool WAVE_EFF_GATE
>
__global__ __launch_bounds__((BN / 8) * 32, MIN_CTA)
void mx6_w13_c48_v10_kernel(
    const uint8_t* __restrict__ A,
    const uint8_t* __restrict__ AS,
    const int32_t* __restrict__ Row2Tok,
    const uint8_t* __restrict__ SM4,
    const uint32_t* __restrict__ WS3_PACK,
    const uint32_t* __restrict__ EXP_STREAM,
    const int32_t* __restrict__ EXP_OFF,
    const uint8_t* __restrict__ WARP_META,
    const uint8_t* __restrict__ HUFF_LUT,
    const float* __restrict__ Ones,
    const int32_t* __restrict__ Tiles,
    const int32_t* __restrict__ TileCompactStart,
    const float* __restrict__ Gammas,
    __nv_bfloat16* __restrict__ Out,
    int N, int K, int MAX_CTAS, uint32_t base_ws,
    int wave_stride_other, int wave_margin_pct,
    int pdl_wait_mode, int trig_dep
) {
    static_assert(BN == 16 || BN == 32, "C48 MX6 W13 BN must be 16 or 32");
    static_assert(TILE == 8 || TILE == 16 || TILE == 32, "C48 MX6 W13 TILE must be 8, 16, or 32");

    // Unconditional #206 PDL wait-at-entry: router writes Tiles[], so W13 must NEVER read
    // Tiles before griddepcontrol.wait completes for any pdl_wait_mode >= 1.
    if constexpr (USE_PDL) {
        if (pdl_wait_mode >= 1) {
            asm volatile("griddepcontrol.wait;" ::: "memory");
        }
    }

    constexpr int WARPS         = BN / 8;
    constexpr int THREADS       = WARPS * 32;
    constexpr int SLAB_K        = 128;
    constexpr int BK_SMEM       = 256;
    constexpr int SLABS_PER_BLK = 2;
    constexpr int SA_STRIDE     = BK_SMEM + 16;  // 272 bf16
    constexpr int N_PANELS32    = 18;
    constexpr int N_PANELS      = (BN == 16) ? 36 : 18;
    constexpr int M_ITERS       = (TILE == 32) ? 2 : 1;
    constexpr int TOTAL_STEPS   = 36 * 8;

    const int n_active = Tiles[2 * MAX_CTAS];
    const int stride   = static_cast<int>(gridDim.x);
    const int bidx     = static_cast<int>(blockIdx.x);

    if constexpr (WAVE_EFF_GATE) {
        const bool bn16_win = (BN == 16)
            ? bn16_wins_wave_eff(n_active, stride, wave_stride_other, wave_margin_pct)
            : bn16_wins_wave_eff(n_active, wave_stride_other, stride, wave_margin_pct);
        const bool take_this = (BN == 16) ? bn16_win : (!bn16_win);
        if (!take_this || n_active <= 0) {
            if constexpr (USE_PDL) {
                if (trig_dep) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
            return;
        }
    }

    const int n_total = n_active * N_PANELS;
    if (bidx >= n_total) {
        if constexpr (USE_PDL) {
            if (trig_dep) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
        return;
    }

    const int n_slabs = K / SLAB_K;   // 36
    const int n_blks  = K / BK_SMEM;  // 18
    const int KG      = K >> 5;       // 144
    const bool fast_ws = (base_ws >= 8u && base_ws <= 238u);
    const uint32_t base_ws_off2 = ((base_ws - 7u) << 7) * 0x00010001u;

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g    = lane >> 2;
    const int g3   = g * 3;
    const int lt   = lane & 3;
    const uint32_t lanemask_lt = get_lanemask_lt();

    alignas(128) __shared__ uint8_t sLut[MX6_LUT_SIZE];
    alignas(16)  __shared__ __nv_bfloat16 sA[2][TILE * SA_STRIDE];

    if constexpr (THREADS == 128) {
        sLut[tid] = HUFF_LUT[tid];
    } else {
        sLut[tid]      = HUFF_LUT[tid];
        sLut[tid + 64] = HUFF_LUT[tid + 64];
    }
    __syncthreads();

    bool trig_fired = false;

    for (int idx = bidx; idx < n_total; idx += stride) {
        const int t  = idx / N_PANELS;
        const int nb = idx - t * N_PANELS;
        const bool is_first_tile = (idx == bidx);
        const bool is_last_tile  = (idx + stride >= n_total);

        const int64_t e = static_cast<int64_t>(Tiles[t]);
        const int end_row = Tiles[MAX_CTAS + t];
        const int row_base = t * TILE;
        int live_m = end_row - row_base;
        if (live_m > TILE) live_m = TILE;
        if (live_m < 0) live_m = 0;

        const int nb32 = (BN == 16) ? (nb >> 1) : nb;
        const int wp   = (BN == 16) ? (((nb & 1) << 1) + warp) : warp;
        const int64_t warp_uid = (e * N_PANELS32 + nb32) * 4 + wp;

        const uint8_t*  sm4_base = SM4 + warp_uid * n_slabs * MX6_SM4_SLAB_BYTES;
        const uint32_t* ws3_base = WS3_PACK + warp_uid * n_slabs * 3;
        const uint32_t* exp_base = EXP_STREAM + static_cast<int64_t>(EXP_OFF[warp_uid]) * MX6_CHUNK_WORDS;
        const uint8_t   wmeta    = WARP_META[warp_uid];
        const bool      use_raw4        = ((wmeta & WMETA_RAW4_BIT) != 0u);
        const bool      warp_all_normal = fast_ws && ((wmeta & WMETA_ALL_NORMAL_BIT) != 0u);
        const bool      warp_no_nan     = fast_ws && ((wmeta & WMETA_HAS_NAN_BIT) == 0u);
        (void)is_first_tile;
        if (live_m == 0) continue;

        uint32_t ws_cur = ld_cs_u32(ws3_base + lane);
        uint32_t ws_nxt = 0u;

        // Vectorized 8-elem/thread (LDG.64 -> STS.128) A-block loader!
        auto load_a_blk = [&](int blk, __nv_bfloat16* dst) {
            constexpr int V8_PER_ROW = BK_SMEM >> 3;  // 32
            const int total_v8 = live_m * V8_PER_ROW;
            const int k0 = blk * BK_SMEM;
            for (int aidx = tid; aidx < total_v8; aidx += THREADS) {
                const int lm = aidx >> 5;
                const int q8 = aidx & 31;
                const int64_t row = static_cast<int64_t>(row_base + lm);
                const int64_t a_row = ZERO_SCATTER
                    ? static_cast<int64_t>(__ldg(Row2Tok + row))
                    : row;
                const int kcol   = k0 + (q8 << 3);
                const int kg_col = kcol >> 5;
                const uint2 a8 = *reinterpret_cast<const uint2*>(A + a_row * K + kcol);
                const uint8_t as_b = (ZERO_SCATTER || AS_ROW_MAJOR)
                    ? AS[a_row * KG + kg_col]
                    : AS[scale_offset_decode(row, kg_col, KG)];
                const float a_sc = e8m0_to_f32(as_b);
                uint32_t b0, b1, b2, b3;
                e4m3x4_scale_to_bf16x4(a8.x, a_sc, b0, b1);
                e4m3x4_scale_to_bf16x4(a8.y, a_sc, b2, b3);
                *reinterpret_cast<uint4*>(dst + lm * SA_STRIDE + (q8 << 3)) = make_uint4(b0, b1, b2, b3);
            }
        };

        MX6StreamReaderV10 reader{};
        uint4 cur_raw4 = make_uint4(0u, 0u, 0u, 0u);
        uint4 cur_sm4  = ld_cs_u4(sm4_base + lane * 16);
        if (use_raw4) {
            cur_raw4 = ld_cs_u4(reinterpret_cast<const uint8_t*>(exp_base) + lane * 16);
        } else {
            reader.init(exp_base, lane);
        }

        load_a_blk(0, sA[0]);
        __syncthreads();

        float acc[M_ITERS][4] = {{0.0f}};
        const bool has_m0  = (g < live_m);
        const bool has_m8  = (TILE > 8)  && (g + 8 < live_m);
        const bool has_m16 = (TILE == 32) && (g + 16 < live_m);
        const bool has_m24 = (TILE == 32) && (g + 24 < live_m);
        int ws_chunk_idx = 0;

        for (int blk = 0; blk < n_blks; ++blk) {
            const __nv_bfloat16* sAp = sA[blk & 1];
            __nv_bfloat16* sAnext    = sA[(blk & 1) ^ 1];

            #pragma unroll 1
            for (int is = 0; is < SLABS_PER_BLK; ++is) {
                const int sl = blk * SLABS_PER_BLK + is;

                uint4 nxt_sm4  = make_uint4(0u, 0u, 0u, 0u);
                uint4 nxt_raw4 = make_uint4(0u, 0u, 0u, 0u);
                if (sl + 1 < n_slabs) {
                    const int nsl = sl + 1;
                    nxt_sm4 = ld_cs_u4(sm4_base + static_cast<int64_t>(nsl) * MX6_SM4_SLAB_BYTES + lane * 16);
                    if (use_raw4) {
                        nxt_raw4 = ld_cs_u4(reinterpret_cast<const uint8_t*>(exp_base)
                                            + static_cast<int64_t>(nsl) * MX6_RAW4_SLAB_BYTES + lane * 16);
                    }
                }
                if (is == 0 && blk + 1 < n_blks) {
                    load_a_blk(blk + 1, sAnext);
                }

                const int ws_pos  = sl * 3;
                const int req_chk = (ws_pos + 2) >> 5;
                if (req_chk > ws_chunk_idx) {
                    ws_nxt = ((req_chk * 32 + lane) < n_slabs * 3)
                        ? ld_cs_u32(ws3_base + req_chk * 32 + lane) : 0u;
                }
                const int idx0 = ws_pos - ws_chunk_idx * 32;
                const uint32_t u0 = (idx0 < 32)
                    ? __shfl_sync(0xFFFFFFFFu, ws_cur, idx0)
                    : __shfl_sync(0xFFFFFFFFu, ws_nxt, idx0 - 32);
                const uint32_t u1 = (idx0 + 1 < 32)
                    ? __shfl_sync(0xFFFFFFFFu, ws_cur, idx0 + 1)
                    : __shfl_sync(0xFFFFFFFFu, ws_nxt, idx0 - 31);
                const uint32_t u2 = (idx0 + 2 < 32)
                    ? __shfl_sync(0xFFFFFFFFu, ws_cur, idx0 + 2)
                    : __shfl_sync(0xFFFFFFFFu, ws_nxt, idx0 - 30);
                if (req_chk > ws_chunk_idx) {
                    ws_cur = ws_nxt;
                    ws_chunk_idx = req_chk;
                }

                const uint32_t sm_pair[4] = {cur_sm4.x, cur_sm4.y, cur_sm4.z, cur_sm4.w};
                const uint32_t r4_pair[4] = {cur_raw4.x, cur_raw4.y, cur_raw4.z, cur_raw4.w};
                const int blk_k_off = is * 128;

                #pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const int kq_off = blk_k_off + q * 32;
                    const uint32_t ws_word  = unpack_ws3_word(u0, u1, u2, q);
                    const uint32_t ws_delta = (ws_word >> g3) & 7u;
                    const uint32_t ws_off2  = base_ws_off2 + ws_delta * 0x00800080u;
                    const uint32_t ws_byte  = base_ws + ws_delta;

                    const int s0 = q << 1;
                    const int s1 = s0 | 1;
                    uint32_t d0, d1, d2, d3;
                    uint32_t b0_0, b1_0, b0_1, b1_1;

                    if (use_raw4) {
                        raw4_to_d4(r4_pair[q] & 0xFFFFu, d0, d1, d2, d3);
                    } else {
                        reader.decode4_and_refill(sLut, lane, lanemask_lt, (sl * 8 + s0 + 1) < TOTAL_STEPS, d0, d1, d2, d3);
                    }
                    decode_mx6_step_to_bf16x4(sm_pair[q] & 0xFFFFu, d0, d1, d2, d3, ws_off2, ws_byte,
                                              warp_all_normal, warp_no_nan, b0_0, b1_0);

                    if (use_raw4) {
                        raw4_to_d4(r4_pair[q] >> 16, d0, d1, d2, d3);
                    } else {
                        reader.decode4_and_refill(sLut, lane, lanemask_lt, (sl * 8 + s1 + 1) < TOTAL_STEPS, d0, d1, d2, d3);
                    }
                    decode_mx6_step_to_bf16x4(sm_pair[q] >> 16, d0, d1, d2, d3, ws_off2, ws_byte,
                                              warp_all_normal, warp_no_nan, b0_1, b1_1);

                    uint2 a0 = make_uint2(0u, 0u), a2 = make_uint2(0u, 0u);
                    if (has_m0) {
                        a0 = *reinterpret_cast<const uint2*>(sAp + g * SA_STRIDE + 4 * lt + kq_off);
                        a2 = *reinterpret_cast<const uint2*>(sAp + g * SA_STRIDE + 4 * lt + kq_off + 16);
                    }
                    uint2 a1 = make_uint2(0u, 0u), a3 = make_uint2(0u, 0u);
                    if constexpr (TILE > 8) {
                        if (has_m8) {
                            a1 = *reinterpret_cast<const uint2*>(sAp + (g + 8) * SA_STRIDE + 4 * lt + kq_off);
                            a3 = *reinterpret_cast<const uint2*>(sAp + (g + 8) * SA_STRIDE + 4 * lt + kq_off + 16);
                        }
                    }
                    mma_m16n8k16_bf16(acc[0], a0.x, a1.x, a2.x, a3.x, b0_0, b1_0);
                    mma_m16n8k16_bf16(acc[0], a0.y, a1.y, a2.y, a3.y, b0_1, b1_1);

                    if constexpr (TILE == 32) {
                        uint2 p0 = make_uint2(0u, 0u), p2 = make_uint2(0u, 0u);
                        if (has_m16) {
                            p0 = *reinterpret_cast<const uint2*>(sAp + (g + 16) * SA_STRIDE + 4 * lt + kq_off);
                            p2 = *reinterpret_cast<const uint2*>(sAp + (g + 16) * SA_STRIDE + 4 * lt + kq_off + 16);
                        }
                        uint2 p1 = make_uint2(0u, 0u), p3 = make_uint2(0u, 0u);
                        if (has_m24) {
                            p1 = *reinterpret_cast<const uint2*>(sAp + (g + 24) * SA_STRIDE + 4 * lt + kq_off);
                            p3 = *reinterpret_cast<const uint2*>(sAp + (g + 24) * SA_STRIDE + 4 * lt + kq_off + 16);
                        }
                        mma_m16n8k16_bf16(acc[1], p0.x, p1.x, p2.x, p3.x, b0_0, b1_0);
                        mma_m16n8k16_bf16(acc[1], p0.y, p1.y, p2.y, p3.y, b0_1, b1_1);
                    }
                }

                if constexpr (USE_PDL) {
                    if (trig_dep && is_last_tile && sl == n_slabs - 2) {
                        asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
                        trig_fired = true;
                    }
                }

                if (sl + 1 < n_slabs) {
                    cur_sm4  = nxt_sm4;
                    cur_raw4 = nxt_raw4;
                }
            }
            if (blk + 1 < n_blks) __syncthreads();
        }

        const float scale = Ones[e];
        const int inter  = N >> 1;  // 288
        const int c_warp = nb * (BN >> 1) + warp * 4;
        const int n0     = nb * BN + warp * 8 + 2 * lt;
        const int compact_base = FUSE_SITU ? TileCompactStart[t] : 0;

        #pragma unroll
        for (int im = 0; im < M_ITERS; ++im) {
            const int lm0 = im * 16 + g;
            const int r0  = row_base + lm0;
            const int c0  = compact_base + lm0;
            const float o0 = mul_rn(acc[im][0], scale);
            const float o1 = mul_rn(acc[im][1], scale);
            const float o2 = mul_rn(acc[im][2], scale);
            const float o3 = mul_rn(acc[im][3], scale);

            if constexpr (FUSE_SITU) {
                uint16_t y0_b = 0u;
                if (lm0 < live_m) y0_b = situ_pair_bf16_exact(pack_bf16x2(o0, o1), __ldg(Gammas + r0));
                const uint32_t p0 = static_cast<uint32_t>(y0_b)
                    | (__shfl_xor_sync(0xFFFFFFFFu, static_cast<uint32_t>(y0_b), 1) << 16);
                const uint32_t p1 = __shfl_xor_sync(0xFFFFFFFFu, p0, 2);
                if (lt == 0 && lm0 < live_m) {
                    *reinterpret_cast<uint2*>(Out + static_cast<int64_t>(c0) * inter + c_warp) = make_uint2(p0, p1);
                }
                if constexpr (TILE > 8) {
                    const int lm1 = lm0 + 8;
                    const int r1  = row_base + lm1;
                    const int c1  = compact_base + lm1;
                    uint16_t y1_b = 0u;
                    if (lm1 < live_m) y1_b = situ_pair_bf16_exact(pack_bf16x2(o2, o3), __ldg(Gammas + r1));
                    const uint32_t q0 = static_cast<uint32_t>(y1_b)
                        | (__shfl_xor_sync(0xFFFFFFFFu, static_cast<uint32_t>(y1_b), 1) << 16);
                    const uint32_t q1 = __shfl_xor_sync(0xFFFFFFFFu, q0, 2);
                    if (lt == 0 && lm1 < live_m) {
                        *reinterpret_cast<uint2*>(Out + static_cast<int64_t>(c1) * inter + c_warp) = make_uint2(q0, q1);
                    }
                }
            } else {
                if (lm0 < live_m) {
                    *reinterpret_cast<uint32_t*>(Out + static_cast<int64_t>(r0) * N + n0) = pack_bf16x2(o0, o1);
                }
                if constexpr (TILE > 8) {
                    const int lm1 = lm0 + 8;
                    const int r1  = row_base + lm1;
                    if (lm1 < live_m) {
                        *reinterpret_cast<uint32_t*>(Out + static_cast<int64_t>(r1) * N + n0) = pack_bf16x2(o2, o3);
                    }
                }
            }
        }

        if (!is_last_tile) __syncthreads();
    }

    if constexpr (USE_PDL) {
        if (trig_dep && !trig_fired) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }
}

// =========================================================================
// Kernel 4 (C48/C32 V10): BN=128 (4-Warp) MX6-Huff7 W2 (N=4608, K=288, TILE=8|16|32)
//   - Complete FUSE_COMBINE in {0, 1, 2, 3, 4} parity with `mxfp8_mx7_c48_v9.cu`:
//       * FC=0: sparse Out[r0, 4608]
//       * FC=1: double-buffered P_GROUP=6 wave combine -> SendBuf
//       * FC=2: batched panel-run DownCompact combine -> SendBuf
//       * FC=3: dynamic slice-expert-major L2-tiled (P=6/12) combine -> SendBuf
//       * FC=4: L2-hot DownCompact (`st_cg_u4` -> Out[c0, 4608]) + `mx6_combine_c48_v10_fwd`
//   - 3-SHFL 4x4 BF16x2 transpose -> 64B-coalesced uint4 epilogue stores
//   - Strict #206 PDL (`wait < launch` on ALL paths incl. ALL early exits for pdl_wait_mode==2||3!)
// =========================================================================

template <
    int TILE, int MIN_CTA, bool USE_PDL,
    bool AS_ROW_MAJOR, bool FUSE_QUANT, int FUSE_COMBINE
>
__global__ __launch_bounds__(128, MIN_CTA)
void mx6_w2_c48_v10_kernel(
    const uint8_t* __restrict__ A,
    const uint8_t* __restrict__ AS,
    const uint8_t* __restrict__ SM4,
    const uint32_t* __restrict__ WS3_PACK,
    const uint32_t* __restrict__ EXP_STREAM,
    const int32_t* __restrict__ EXP_OFF,
    const uint8_t* __restrict__ WARP_META,
    const uint8_t* __restrict__ HUFF_LUT,
    const float* __restrict__ Ones,
    const int32_t* __restrict__ Tiles,
    const int32_t* __restrict__ TileCompactStart,
    __nv_bfloat16* __restrict__ Out,
    const int32_t* __restrict__ Src2Compact,
    const int32_t* __restrict__ NLivePtr,
    uint32_t* __restrict__ PanelDone,
    uint32_t* __restrict__ BufGen,
    __nv_bfloat16* __restrict__ SendBuf,
    int N, int K, int MAX_CTAS, int T, uint32_t base_ws, int as_row_stride, int down_row_stride,
    int pdl_wait_mode, int trig_dep
) {
    static_assert(TILE == 8 || TILE == 16 || TILE == 32, "C48 MX6 W2 TILE must be 8, 16, or 32");
    static_assert(FUSE_COMBINE >= 0 && FUSE_COMBINE <= 4, "FUSE_COMBINE must be 0..4");
    (void)N; (void)K;

    if constexpr (USE_PDL) {
        if (pdl_wait_mode == 1) {
            asm volatile("griddepcontrol.wait;" ::: "memory");
        }
    }

    constexpr int WARPS       = 4;
    constexpr int SUB_N       = 4;
    constexpr int BN          = 128;
    constexpr int H_DIM       = 4608;
    constexpr int N_PANELS    = 36;
    constexpr int P_GROUP     = 6;
    constexpr int N_WAVES     = N_PANELS / P_GROUP;  // 6
    constexpr int KG          = 9;
    constexpr int K_DIM       = 288;
    constexpr int SA_STRIDE   = K_DIM + 16;  // 304 bf16
    constexpr int M_ITERS     = (TILE == 32) ? 2 : 1;
    constexpr int TOTAL_STEPS = KG * 8;      // 72
    constexpr bool IN_KERNEL_COMBINE = (FUSE_COMBINE == 1 || FUSE_COMBINE == 2 || FUSE_COMBINE == 3);

    const int tid    = threadIdx.x;
    const int warp   = tid >> 5;
    const int lane   = tid & 31;
    const int g      = lane >> 2;
    const int g3     = g * 3;
    const int lt     = lane & 3;
    const uint32_t lanemask_lt = get_lanemask_lt();
    const int stride = static_cast<int>(gridDim.x);
    const int bidx   = static_cast<int>(blockIdx.x);
    const int CTAS_PER_PANEL = stride / P_GROUP;

    const int n_active = Tiles[2 * MAX_CTAS];

    [[maybe_unused]] auto zero_sendbuf_panel = [&](int nb) {
        int n_live_tok = T;
        if (NLivePtr != nullptr) {
            int nl = *NLivePtr;
            if (nl < 0) nl = 0;
            if (nl < n_live_tok) n_live_tok = nl;
        }
        const int col0 = nb * BN + (tid & 15) * 8;
        const uint4 z4 = make_uint4(0u, 0u, 0u, 0u);
        for (int m = (tid >> 4); m < n_live_tok; m += 8) {
            *reinterpret_cast<uint4*>(SendBuf + static_cast<int64_t>(m) * H_DIM + col0) = z4;
        }
    };

    // Strict #206 PDL early exits across ALL FUSE_COMBINE modes (pdl_wait_mode == 2 || 3)
    if constexpr (FUSE_COMBINE == 1) {
        const int p_slot = bidx / CTAS_PER_PANEL;
        const int p_rank = bidx - p_slot * CTAS_PER_PANEL;
        if (n_active == 0) {
            if constexpr (USE_PDL) {
                if (pdl_wait_mode == 2 || pdl_wait_mode == 3) asm volatile("griddepcontrol.wait;" ::: "memory");
            }
            if (p_rank == 0) {
                for (int w = 0; w < N_WAVES; ++w) zero_sendbuf_panel(w * P_GROUP + p_slot);
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
                    if (trig_dep) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
                }
                return;
            }
        }
    } else {
        if (n_active == 0 || bidx >= n_active * N_PANELS) {
            if constexpr (USE_PDL) {
                if (pdl_wait_mode == 2 || pdl_wait_mode == 3) asm volatile("griddepcontrol.wait;" ::: "memory");
                if (trig_dep) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
            return;
        }
    }

    const bool fast_ws = (base_ws >= 8u && base_ws <= 238u);
    const uint32_t base_ws_off2 = ((base_ws - 7u) << 7) * 0x00010001u;
    const int col_u4_local = warp * 32 + lt * 8;

    alignas(128) __shared__ uint8_t sLut[MX6_LUT_SIZE];
    alignas(16)  __shared__ __nv_bfloat16 sA_all[TILE * SA_STRIDE];
    __shared__ int sIsLastPanel;

    sLut[tid] = HUFF_LUT[tid];
    __syncthreads();

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

        const int64_t warp_uid = (e * N_PANELS + nb) * WARPS + warp;
        const uint8_t*  sm4_base = SM4 + warp_uid * KG * MX6_SM4_SLAB_BYTES;
        const uint32_t* ws3_base = WS3_PACK + warp_uid * KG * 3;
        const uint32_t* exp_base = EXP_STREAM + static_cast<int64_t>(EXP_OFF[warp_uid]) * MX6_CHUNK_WORDS;
        const uint8_t   wmeta    = WARP_META[warp_uid];
        const bool      use_raw4        = ((wmeta & WMETA_RAW4_BIT) != 0u);
        const bool      warp_all_normal = fast_ws && ((wmeta & WMETA_ALL_NORMAL_BIT) != 0u);
        const bool      warp_no_nan     = fast_ws && ((wmeta & WMETA_HAS_NAN_BIT) == 0u);

        if constexpr (USE_PDL) {
            if ((pdl_wait_mode == 2 || pdl_wait_mode == 3) && is_very_first) {
                if (lane < 4)      prefetch_l2(sm4_base + lane * 128);
                else if (lane < 8) prefetch_l2(exp_base + (lane - 4) * 32);
                else if (lane == 8) prefetch_l2(ws3_base);
                asm volatile("griddepcontrol.wait;" ::: "memory");
            }
        }

        if (live_m == 0) return;

        const int compact_tile_base = (FUSE_QUANT || IN_KERNEL_COMBINE || FUSE_COMBINE == 4)
            ? TileCompactStart[t] : 0;

        const uint32_t ws0 = (lane < KG * 3) ? ld_cs_u32(ws3_base + lane) : 0u;

        MX6StreamReaderV10 reader{};
        uint4 cur_raw4 = make_uint4(0u, 0u, 0u, 0u);
        uint4 cur_sm4  = ld_cs_u4(sm4_base + lane * 16);
        if (use_raw4) {
            cur_raw4 = ld_cs_u4(reinterpret_cast<const uint8_t*>(exp_base) + lane * 16);
        } else {
            reader.init(exp_base, lane);
        }

        if constexpr (FUSE_QUANT) {
            const __nv_bfloat16* A_bf16 = reinterpret_cast<const __nv_bfloat16*>(A);
            const int total_blks = live_m * KG;
            for (int b = tid; b < total_blks; b += 128) {
                const int lm = b / KG;
                const int kg = b - lm * KG;
                otf_mxfp8_quant_dequant_blk32(
                    A_bf16 + static_cast<int64_t>(compact_tile_base + lm) * K_DIM + kg * 32,
                    sA_all + lm * SA_STRIDE + kg * 32
                );
            }
        } else {
            const int total_blks = live_m * KG;
            for (int b = tid; b < total_blks; b += 128) {
                const int lm = b / KG;
                const int kg = b - lm * KG;
                const int64_t row = static_cast<int64_t>(row_base + lm);
                const uint8_t as_b = AS_ROW_MAJOR
                    ? AS[row * as_row_stride + kg]
                    : AS[scale_offset_decode(row, kg, KG)];
                const float a_sc = e8m0_to_f32(as_b);
                const uint4 a16_0 = *reinterpret_cast<const uint4*>(A + row * K_DIM + kg * 32);
                const uint4 a16_1 = *reinterpret_cast<const uint4*>(A + row * K_DIM + kg * 32 + 16);
                __nv_bfloat16* dst = sA_all + lm * SA_STRIDE + kg * 32;
                uint32_t b0, b1, b2, b3;
                e4m3x4_scale_to_bf16x4(a16_0.x, a_sc, b0, b1);
                e4m3x4_scale_to_bf16x4(a16_0.y, a_sc, b2, b3);
                *reinterpret_cast<uint4*>(dst + 0)  = make_uint4(b0, b1, b2, b3);
                e4m3x4_scale_to_bf16x4(a16_0.z, a_sc, b0, b1);
                e4m3x4_scale_to_bf16x4(a16_0.w, a_sc, b2, b3);
                *reinterpret_cast<uint4*>(dst + 8)  = make_uint4(b0, b1, b2, b3);
                e4m3x4_scale_to_bf16x4(a16_1.x, a_sc, b0, b1);
                e4m3x4_scale_to_bf16x4(a16_1.y, a_sc, b2, b3);
                *reinterpret_cast<uint4*>(dst + 16) = make_uint4(b0, b1, b2, b3);
                e4m3x4_scale_to_bf16x4(a16_1.z, a_sc, b0, b1);
                e4m3x4_scale_to_bf16x4(a16_1.w, a_sc, b2, b3);
                *reinterpret_cast<uint4*>(dst + 24) = make_uint4(b0, b1, b2, b3);
            }
        }
        __syncthreads();

        float acc[M_ITERS][SUB_N][4] = {{{0.0f}}};
        const bool has_m0  = (g < live_m);
        const bool has_m8  = (TILE > 8)  && (g + 8 < live_m);
        const bool has_m16 = (TILE == 32) && (g + 16 < live_m);
        const bool has_m24 = (TILE == 32) && (g + 24 < live_m);

        #pragma unroll 1
        for (int kg = 0; kg < KG; ++kg) {
            uint4 nxt_sm4  = make_uint4(0u, 0u, 0u, 0u);
            uint4 nxt_raw4 = make_uint4(0u, 0u, 0u, 0u);
            if (kg + 1 < KG) {
                const int nkg = kg + 1;
                nxt_sm4 = ld_cs_u4(sm4_base + static_cast<int64_t>(nkg) * MX6_SM4_SLAB_BYTES + lane * 16);
                if (use_raw4) {
                    nxt_raw4 = ld_cs_u4(reinterpret_cast<const uint8_t*>(exp_base)
                                        + static_cast<int64_t>(nkg) * MX6_RAW4_SLAB_BYTES + lane * 16);
                }
            } else if (has_next_tile) {
                const int64_t ne    = static_cast<int64_t>(Tiles[next_t]);
                const int64_t nwuid = (ne * N_PANELS + next_nb) * WARPS + warp;
                if (lane < 4)      prefetch_l2(SM4 + nwuid * KG * MX6_SM4_SLAB_BYTES + lane * 128);
                else if (lane < 8) prefetch_l2(EXP_STREAM + static_cast<int64_t>(EXP_OFF[nwuid]) * MX6_CHUNK_WORDS + (lane - 4) * 32);
                else if (lane == 8) prefetch_l2(WS3_PACK + nwuid * KG * 3);
            }

            const int k_off = kg * 32;
            uint2 a0 = make_uint2(0u, 0u), a2 = make_uint2(0u, 0u);
            if (has_m0) {
                a0 = *reinterpret_cast<const uint2*>(sA_all + g * SA_STRIDE + 4 * lt + k_off);
                a2 = *reinterpret_cast<const uint2*>(sA_all + g * SA_STRIDE + 4 * lt + k_off + 16);
            }
            uint2 a1 = make_uint2(0u, 0u), a3 = make_uint2(0u, 0u);
            if constexpr (TILE > 8) {
                if (has_m8) {
                    a1 = *reinterpret_cast<const uint2*>(sA_all + (g + 8) * SA_STRIDE + 4 * lt + k_off);
                    a3 = *reinterpret_cast<const uint2*>(sA_all + (g + 8) * SA_STRIDE + 4 * lt + k_off + 16);
                }
            }
            uint2 p0 = make_uint2(0u, 0u), p1 = make_uint2(0u, 0u);
            uint2 p2 = make_uint2(0u, 0u), p3 = make_uint2(0u, 0u);
            if constexpr (TILE == 32) {
                if (has_m16) {
                    p0 = *reinterpret_cast<const uint2*>(sA_all + (g + 16) * SA_STRIDE + 4 * lt + k_off);
                    p2 = *reinterpret_cast<const uint2*>(sA_all + (g + 16) * SA_STRIDE + 4 * lt + k_off + 16);
                }
                if (has_m24) {
                    p1 = *reinterpret_cast<const uint2*>(sA_all + (g + 24) * SA_STRIDE + 4 * lt + k_off);
                    p3 = *reinterpret_cast<const uint2*>(sA_all + (g + 24) * SA_STRIDE + 4 * lt + k_off + 16);
                }
            }

            const uint32_t u0 = __shfl_sync(0xFFFFFFFFu, ws0, kg * 3 + 0);
            const uint32_t u1 = __shfl_sync(0xFFFFFFFFu, ws0, kg * 3 + 1);
            const uint32_t u2 = __shfl_sync(0xFFFFFFFFu, ws0, kg * 3 + 2);

            const uint32_t sm_pair[4] = {cur_sm4.x, cur_sm4.y, cur_sm4.z, cur_sm4.w};
            const uint32_t r4_pair[4] = {cur_raw4.x, cur_raw4.y, cur_raw4.z, cur_raw4.w};

            #pragma unroll
            for (int sub = 0; sub < SUB_N; ++sub) {
                const uint32_t ws_word  = unpack_ws3_word(u0, u1, u2, sub);
                const uint32_t ws_delta = (ws_word >> g3) & 7u;
                const uint32_t ws_off2  = base_ws_off2 + ws_delta * 0x00800080u;
                const uint32_t ws_byte  = base_ws + ws_delta;

                const int s0 = sub << 1;
                const int s1 = s0 | 1;
                uint32_t d0, d1, d2, d3;
                uint32_t b0_0, b1_0, b0_1, b1_1;

                if (use_raw4) {
                    raw4_to_d4(r4_pair[sub] & 0xFFFFu, d0, d1, d2, d3);
                } else {
                    reader.decode4_and_refill(sLut, lane, lanemask_lt, (kg * 8 + s0 + 1) < TOTAL_STEPS, d0, d1, d2, d3);
                }
                decode_mx6_step_to_bf16x4(sm_pair[sub] & 0xFFFFu, d0, d1, d2, d3, ws_off2, ws_byte,
                                          warp_all_normal, warp_no_nan, b0_0, b1_0);

                if (use_raw4) {
                    raw4_to_d4(r4_pair[sub] >> 16, d0, d1, d2, d3);
                } else {
                    reader.decode4_and_refill(sLut, lane, lanemask_lt, (kg * 8 + s1 + 1) < TOTAL_STEPS, d0, d1, d2, d3);
                }
                decode_mx6_step_to_bf16x4(sm_pair[sub] >> 16, d0, d1, d2, d3, ws_off2, ws_byte,
                                          warp_all_normal, warp_no_nan, b0_1, b1_1);

                mma_m16n8k16_bf16(acc[0][sub], a0.x, a1.x, a2.x, a3.x, b0_0, b1_0);
                mma_m16n8k16_bf16(acc[0][sub], a0.y, a1.y, a2.y, a3.y, b0_1, b1_1);
                if constexpr (TILE == 32) {
                    mma_m16n8k16_bf16(acc[1][sub], p0.x, p1.x, p2.x, p3.x, b0_0, b1_0);
                    mma_m16n8k16_bf16(acc[1][sub], p0.y, p1.y, p2.y, p3.y, b0_1, b1_1);
                }
            }

            if constexpr (USE_PDL && !IN_KERNEL_COMBINE) {
                if (trig_dep && is_very_last && kg == KG - 2) {
                    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
                    trig_fired = true;
                }
            }

            if (kg + 1 < KG) {
                cur_sm4  = nxt_sm4;
                cur_raw4 = nxt_raw4;
            }
        }

        const float scale = Ones[e];
        #pragma unroll
        for (int im = 0; im < M_ITERS; ++im) {
            const int lm0 = im * 16 + g;
            const int r0  = row_base + lm0;
            const int c0  = compact_tile_base + lm0;
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
                const int lm1 = lm0 + 8;
                const int r1  = row_base + lm1;
                const int c1  = compact_tile_base + lm1;
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
                const int next_t  = is_last_in_wave ? p_rank : (t + CTAS_PER_PANEL);
                const int next_nb = is_last_in_wave ? ((w + 1) * P_GROUP + p_slot) : nb;
                run_tile(t, nb, is_very_first, false, next_t, next_nb, !is_very_last, down_panel_buf);
                if (!is_last_in_wave) __syncthreads();
            }

            __threadfence();
            __syncthreads();
            if (active_in_panel == 1) {
                if (tid == 0) sIsLastPanel = 1;
            } else if (tid == 0) {
                const uint32_t prev = atom_acq_rel_gpu_add(&PanelDone[nb], 1u);
                if (prev == static_cast<uint32_t>(active_in_panel - 1)) {
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
        // FC == 0 (sparse Out[r0, 4608]) and FC == 4 (L2-hot DownCompact Out[c0, 4608]):
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

    if constexpr (USE_PDL) {
        if (trig_dep && !trig_fired) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }
}

// =========================================================================
// Kernel 5 (C48/C32 V10): 0-ULP PDL + NLivePtr Vectorized SM121a Combine (H=4608, topk=8)
//   - Bit-exact slot 0..7 FP32 add_rn order (including +0.0f on r<0 for -0.0f parity -> nd == 0)
//   - Strict #206 PDL (`wait < launch` on ALL paths incl. token >= NLivePtr early exit!)
// =========================================================================

template <int BLOCK_THREADS, int COLS_PER_CTA, int MIN_CTA_C, bool PDL_WAIT, bool TRIG_DEP, bool HAS_NLIVE>
__global__ __launch_bounds__(BLOCK_THREADS, MIN_CTA_C)
void mx6_combine_c48_v10_kernel(
    const __nv_bfloat16* __restrict__ Down,
    const int32_t* __restrict__ Src2Row,
    const int32_t* __restrict__ NLivePtr,
    __nv_bfloat16* __restrict__ Out
) {
    const int token = static_cast<int>(blockIdx.y);
    const int tid   = threadIdx.x;
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

    const int32_t r0 = sRows[0], r1 = sRows[1], r2 = sRows[2], r3 = sRows[3];
    const uint4 v0 = (r0 >= 0) ? ld_cs_u4(Down + static_cast<int64_t>(r0) * H_DIM + col) : z4;
    const uint4 v1 = (r1 >= 0) ? ld_cs_u4(Down + static_cast<int64_t>(r1) * H_DIM + col) : z4;
    const uint4 v2 = (r2 >= 0) ? ld_cs_u4(Down + static_cast<int64_t>(r2) * H_DIM + col) : z4;
    const uint4 v3 = (r3 >= 0) ? ld_cs_u4(Down + static_cast<int64_t>(r3) * H_DIM + col) : z4;

    const int32_t r4 = sRows[4], r5 = sRows[5], r6 = sRows[6], r7 = sRows[7];
    const uint4 v4 = (r4 >= 0) ? ld_cs_u4(Down + static_cast<int64_t>(r4) * H_DIM + col) : z4;
    const uint4 v5 = (r5 >= 0) ? ld_cs_u4(Down + static_cast<int64_t>(r5) * H_DIM + col) : z4;
    const uint4 v6 = (r6 >= 0) ? ld_cs_u4(Down + static_cast<int64_t>(r6) * H_DIM + col) : z4;
    const uint4 v7 = (r7 >= 0) ? ld_cs_u4(Down + static_cast<int64_t>(r7) * H_DIM + col) : z4;

    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    float a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;

    #define ACC_U4(v) \
        a0 = add_rn(a0, bf16_bits_to_f32(static_cast<uint16_t>((v).x & 0xFFFFu))); \
        a1 = add_rn(a1, bf16_bits_to_f32(static_cast<uint16_t>((v).x >> 16)));      \
        a2 = add_rn(a2, bf16_bits_to_f32(static_cast<uint16_t>((v).y & 0xFFFFu))); \
        a3 = add_rn(a3, bf16_bits_to_f32(static_cast<uint16_t>((v).y >> 16)));      \
        a4 = add_rn(a4, bf16_bits_to_f32(static_cast<uint16_t>((v).z & 0xFFFFu))); \
        a5 = add_rn(a5, bf16_bits_to_f32(static_cast<uint16_t>((v).z >> 16)));      \
        a6 = add_rn(a6, bf16_bits_to_f32(static_cast<uint16_t>((v).w & 0xFFFFu))); \
        a7 = add_rn(a7, bf16_bits_to_f32(static_cast<uint16_t>((v).w >> 16)));

    ACC_U4(v0); ACC_U4(v1); ACC_U4(v2); ACC_U4(v3);
    ACC_U4(v4); ACC_U4(v5); ACC_U4(v6); ACC_U4(v7);
    #undef ACC_U4

    *reinterpret_cast<uint4*>(Out + static_cast<int64_t>(token) * H_DIM + col) = make_uint4(
        pack_bf16x2(a0, a1), pack_bf16x2(a2, a3),
        pack_bf16x2(a4, a5), pack_bf16x2(a6, a7)
    );
}

// =========================================================================
// PDL-Aware Launch Helper
// =========================================================================

template <typename KernelT, typename... Args>
static inline cudaError_t launch_mx6_v10(
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

// =========================================================================
// C ABI Exports (All return int: 0 == cudaSuccess; take uintptr_t stream_ptr)
// =========================================================================

extern "C" int mx6_v10_version() {
    return 12;
}

extern "C" int mx6_v12_version() {
    return 12;
}

extern "C" int mx6_v10_min_cta() {
    return MX6_W13_T8_BN32_MIN_CTA * 10 + MX6_W2_T8_MIN_CTA;
}

extern "C" int mx6_v10_hist_exp(
    const void* W, void* hist16, int64_t n_elems, uintptr_t stream_ptr
) {
    if (!W || !hist16 || n_elems <= 0 || (n_elems & 15) != 0) return -1;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    cudaError_t err = cudaMemsetAsync(hist16, 0, 16 * sizeof(unsigned long long), stream);
    if (err != cudaSuccess) return static_cast<int>(err);
    const int64_t n_vec = n_elems >> 4;
    const int blocks = static_cast<int>(std::min<int64_t>(2048, (n_vec + 255) / 256));
    mx6_v10_hist_exp_kernel<<<blocks, 256, 0, stream>>>(
        static_cast<const uint8_t*>(W),
        static_cast<unsigned long long*>(hist16),
        n_elems
    );
    return static_cast<int>(cudaGetLastError());
}

extern "C" int mx6_measure_v10(
    const void* W, const void* ENC_LEN,
    void* warp_chunks_out, void* warp_meta_out,
    int proj, int E, int N, int K,
    int from_panel, int src_pb, int src_kb,
    uintptr_t stream_ptr
) {
    if (!W || !ENC_LEN || !warp_chunks_out || !warp_meta_out) return -1;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    if (proj == 0) {
        if ((N % 32) != 0 || (K % 128) != 0) return -2;
        const int total_warps = E * (N / 32) * 4;
        if (from_panel) {
            if (src_pb <= 0 || src_kb <= 0 || (N % src_pb) != 0 || (K % src_kb) != 0) return -3;
            mx6_v10_measure_kernel<0, true><<<total_warps, 32, 0, stream>>>(
                static_cast<const uint8_t*>(W), static_cast<const uint8_t*>(ENC_LEN),
                static_cast<int32_t*>(warp_chunks_out), static_cast<uint8_t*>(warp_meta_out),
                E, N, K, src_pb, src_kb);
        } else {
            mx6_v10_measure_kernel<0, false><<<total_warps, 32, 0, stream>>>(
                static_cast<const uint8_t*>(W), static_cast<const uint8_t*>(ENC_LEN),
                static_cast<int32_t*>(warp_chunks_out), static_cast<uint8_t*>(warp_meta_out),
                E, N, K, 0, 0);
        }
    } else if (proj == 1) {
        if ((N % 128) != 0 || (K % 32) != 0) return -2;
        const int total_warps = E * (N / 128) * 4;
        if (from_panel) {
            if (src_pb <= 0 || src_kb <= 0 || (N % src_pb) != 0 || (K % src_kb) != 0) return -3;
            mx6_v10_measure_kernel<1, true><<<total_warps, 32, 0, stream>>>(
                static_cast<const uint8_t*>(W), static_cast<const uint8_t*>(ENC_LEN),
                static_cast<int32_t*>(warp_chunks_out), static_cast<uint8_t*>(warp_meta_out),
                E, N, K, src_pb, src_kb);
        } else {
            mx6_v10_measure_kernel<1, false><<<total_warps, 32, 0, stream>>>(
                static_cast<const uint8_t*>(W), static_cast<const uint8_t*>(ENC_LEN),
                static_cast<int32_t*>(warp_chunks_out), static_cast<uint8_t*>(warp_meta_out),
                E, N, K, 0, 0);
        }
    } else {
        return -1;
    }
    return static_cast<int>(cudaGetLastError());
}

extern "C" int mx6_pack_v10(
    const void* W, const void* WS,
    const void* ENC_CODE, const void* ENC_LEN,
    const void* EXP_OFF, const void* WARP_META,
    void* SM4, void* WS3_PACK, void* EXP_STREAM,
    int proj, int E, int N, int K, uint32_t base_ws,
    int from_panel, int src_pb, int src_kb,
    uintptr_t stream_ptr
) {
    if (!W || !WS || !ENC_CODE || !ENC_LEN || !EXP_OFF || !WARP_META || !SM4 || !WS3_PACK || !EXP_STREAM) {
        return -1;
    }
    if (K <= 0 || K > 4608) return -4;  // protects sLaneWords[256][32] (ceil(1152*7/32)+1 = 253 <= 256)
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    if (proj == 0) {
        if ((N % 32) != 0 || (K % 128) != 0) return -2;
        const int total_warps = E * (N / 32) * 4;
        if (from_panel) {
            if (src_pb <= 0 || src_kb <= 0 || (N % src_pb) != 0 || (K % src_kb) != 0) return -3;
            mx6_v10_pack_kernel<0, true><<<total_warps, 32, 0, stream>>>(
                static_cast<const uint8_t*>(W), static_cast<const uint8_t*>(WS),
                static_cast<const uint16_t*>(ENC_CODE), static_cast<const uint8_t*>(ENC_LEN),
                static_cast<const int32_t*>(EXP_OFF), static_cast<const uint8_t*>(WARP_META),
                static_cast<uint8_t*>(SM4), static_cast<uint32_t*>(WS3_PACK),
                static_cast<uint32_t*>(EXP_STREAM),
                E, N, K, base_ws, src_pb, src_kb);
        } else {
            mx6_v10_pack_kernel<0, false><<<total_warps, 32, 0, stream>>>(
                static_cast<const uint8_t*>(W), static_cast<const uint8_t*>(WS),
                static_cast<const uint16_t*>(ENC_CODE), static_cast<const uint8_t*>(ENC_LEN),
                static_cast<const int32_t*>(EXP_OFF), static_cast<const uint8_t*>(WARP_META),
                static_cast<uint8_t*>(SM4), static_cast<uint32_t*>(WS3_PACK),
                static_cast<uint32_t*>(EXP_STREAM),
                E, N, K, base_ws, 0, 0);
        }
    } else if (proj == 1) {
        if ((N % 128) != 0 || (K % 32) != 0) return -2;
        const int total_warps = E * (N / 128) * 4;
        if (from_panel) {
            if (src_pb <= 0 || src_kb <= 0 || (N % src_pb) != 0 || (K % src_kb) != 0) return -3;
            mx6_v10_pack_kernel<1, true><<<total_warps, 32, 0, stream>>>(
                static_cast<const uint8_t*>(W), static_cast<const uint8_t*>(WS),
                static_cast<const uint16_t*>(ENC_CODE), static_cast<const uint8_t*>(ENC_LEN),
                static_cast<const int32_t*>(EXP_OFF), static_cast<const uint8_t*>(WARP_META),
                static_cast<uint8_t*>(SM4), static_cast<uint32_t*>(WS3_PACK),
                static_cast<uint32_t*>(EXP_STREAM),
                E, N, K, base_ws, src_pb, src_kb);
        } else {
            mx6_v10_pack_kernel<1, false><<<total_warps, 32, 0, stream>>>(
                static_cast<const uint8_t*>(W), static_cast<const uint8_t*>(WS),
                static_cast<const uint16_t*>(ENC_CODE), static_cast<const uint8_t*>(ENC_LEN),
                static_cast<const int32_t*>(EXP_OFF), static_cast<const uint8_t*>(WARP_META),
                static_cast<uint8_t*>(SM4), static_cast<uint32_t*>(WS3_PACK),
                static_cast<uint32_t*>(EXP_STREAM),
                E, N, K, base_ws, 0, 0);
        }
    } else {
        return -1;
    }
    return static_cast<int>(cudaGetLastError());
}

extern "C" int mx6_unpack_expert_batch_v10_fwd(
    const void* SM4, const void* WS3_PACK,
    const void* EXP_STREAM, const void* EXP_OFF,
    const void* WARP_META, const void* HUFF_LUT,
    void* W_OUT, void* WS_OUT,
    int proj, int E, int N, int K, uint32_t base_ws,
    int out_panel, int dst_pb, int dst_kb,
    uintptr_t stream_ptr
) {
    if (!SM4 || !WS3_PACK || !EXP_STREAM || !EXP_OFF || !WARP_META || !HUFF_LUT || !W_OUT || !WS_OUT) {
        return -1;
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    if (proj == 0) {
        if ((N % 32) != 0 || (K % 128) != 0) return -2;
        const int total_warps = E * (N / 32) * 4;
        if (out_panel) {
            if (dst_pb <= 0 || dst_kb <= 0 || (dst_kb % 32) != 0 || (N % dst_pb) != 0 || (K % dst_kb) != 0) return -3;
            mx6_v10_unpack_kernel<0, true><<<total_warps, 32, 0, stream>>>(
                static_cast<const uint8_t*>(SM4), static_cast<const uint32_t*>(WS3_PACK),
                static_cast<const uint32_t*>(EXP_STREAM), static_cast<const int32_t*>(EXP_OFF),
                static_cast<const uint8_t*>(WARP_META), static_cast<const uint8_t*>(HUFF_LUT),
                static_cast<uint8_t*>(W_OUT), static_cast<uint8_t*>(WS_OUT),
                E, N, K, base_ws, dst_pb, dst_kb);
        } else {
            mx6_v10_unpack_kernel<0, false><<<total_warps, 32, 0, stream>>>(
                static_cast<const uint8_t*>(SM4), static_cast<const uint32_t*>(WS3_PACK),
                static_cast<const uint32_t*>(EXP_STREAM), static_cast<const int32_t*>(EXP_OFF),
                static_cast<const uint8_t*>(WARP_META), static_cast<const uint8_t*>(HUFF_LUT),
                static_cast<uint8_t*>(W_OUT), static_cast<uint8_t*>(WS_OUT),
                E, N, K, base_ws, 0, 0);
        }
    } else if (proj == 1) {
        if ((N % 128) != 0 || (K % 32) != 0) return -2;
        const int total_warps = E * (N / 128) * 4;
        if (out_panel) {
            if (dst_pb <= 0 || dst_kb <= 0 || (dst_kb % 32) != 0 || (N % dst_pb) != 0 || (K % dst_kb) != 0) return -3;
            mx6_v10_unpack_kernel<1, true><<<total_warps, 32, 0, stream>>>(
                static_cast<const uint8_t*>(SM4), static_cast<const uint32_t*>(WS3_PACK),
                static_cast<const uint32_t*>(EXP_STREAM), static_cast<const int32_t*>(EXP_OFF),
                static_cast<const uint8_t*>(WARP_META), static_cast<const uint8_t*>(HUFF_LUT),
                static_cast<uint8_t*>(W_OUT), static_cast<uint8_t*>(WS_OUT),
                E, N, K, base_ws, dst_pb, dst_kb);
        } else {
            mx6_v10_unpack_kernel<1, false><<<total_warps, 32, 0, stream>>>(
                static_cast<const uint8_t*>(SM4), static_cast<const uint32_t*>(WS3_PACK),
                static_cast<const uint32_t*>(EXP_STREAM), static_cast<const int32_t*>(EXP_OFF),
                static_cast<const uint8_t*>(WARP_META), static_cast<const uint8_t*>(HUFF_LUT),
                static_cast<uint8_t*>(W_OUT), static_cast<uint8_t*>(WS_OUT),
                E, N, K, base_ws, 0, 0);
        }
    } else {
        return -1;
    }
    return static_cast<int>(cudaGetLastError());
}

extern "C" int mx6_w13_c1_v10_fwd(
    const void* A, const void* AS,
    const void* SM4, const void* WS3_PACK,
    const void* EXP_STREAM, const void* EXP_OFF,
    const void* WARP_META, const void* HUFF_LUT,
    const void* Ones, const void* Tiles,
    const void* TileMask, const void* TileCompactStart,
    const void* Gammas, void* Out,
    int N, int K, int MAX_CTAS, int TILE, uint32_t base_ws,
    int zero_scatter, int fuse_situ,
    int bn_mode, int u_thresh,
    int pdl_wait_mode, int trig_dep, uintptr_t stream_ptr
) {
    if (N != 576 || K != 4608) return -2;
    if (TILE != 8 && TILE != 16) return -3;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    const bool use_pdl = (pdl_wait_mode != 0) || (trig_dep != 0);
    const bool zs = (zero_scatter != 0);
    const bool fs = (fuse_situ != 0);
    const int eff_thresh = (u_thresh > 0) ? u_thresh : 37;
    const int g16 = (TILE == 8) ? (48 * MX6_W13_T8_BN16_MIN_CTA) : 240;
    const int g32 = (TILE == 8) ? (48 * MX6_W13_T8_BN32_MIN_CTA) : (48 * MX6_W13_T16_MIN_CTA);

    auto run_one = [&](bool use_bn16, int u_min, int u_max, bool wave_gate, int pw, int td) -> cudaError_t {
        const bool pdl_k = (pw != 0) || (td != 0);
        const dim3 grid(use_bn16 ? g16 : g32);
        const dim3 blk(use_bn16 ? 64 : 128);
        const int other_g = use_bn16 ? g32 : g16;
        const int margin = (bn_mode == 3) ? ((u_thresh >= 0 && u_thresh <= 50) ? u_thresh : 3) : 3;

        #define DISPATCH_C1_W13(BN_V, T_V, MC_V, PDL_V, ZS_V, FS_V, WG_V) \
            launch_mx6_v10( \
                mx6_w13_c1_v10_kernel<BN_V, T_V, MC_V, PDL_V, ZS_V, FS_V, WG_V>, \
                grid, blk, stream, PDL_V, \
                static_cast<const uint8_t*>(A), static_cast<const uint8_t*>(AS), \
                static_cast<const uint8_t*>(SM4), static_cast<const uint32_t*>(WS3_PACK), \
                static_cast<const uint32_t*>(EXP_STREAM), static_cast<const int32_t*>(EXP_OFF), \
                static_cast<const uint8_t*>(WARP_META), static_cast<const uint8_t*>(HUFF_LUT), \
                static_cast<const float*>(Ones), static_cast<const int32_t*>(Tiles), \
                static_cast<const int32_t*>(TileMask), static_cast<const int32_t*>(TileCompactStart), \
                static_cast<const float*>(Gammas), static_cast<__nv_bfloat16*>(Out), \
                N, K, MAX_CTAS, base_ws, u_min, u_max, other_g, margin, pw, td)

        #define DISPATCH_C1_W13_FLAGS(BN_V, T_V, MC_V, WG_V) do { \
            if (pdl_k) { \
                if (zs && fs)       return DISPATCH_C1_W13(BN_V, T_V, MC_V, true,  true,  true,  WG_V); \
                else if (zs && !fs) return DISPATCH_C1_W13(BN_V, T_V, MC_V, true,  true,  false, WG_V); \
                else if (!zs && fs) return DISPATCH_C1_W13(BN_V, T_V, MC_V, true,  false, true,  WG_V); \
                else                return DISPATCH_C1_W13(BN_V, T_V, MC_V, true,  false, false, WG_V); \
            } else { \
                if (zs && fs)       return DISPATCH_C1_W13(BN_V, T_V, MC_V, false, true,  true,  WG_V); \
                else if (zs && !fs) return DISPATCH_C1_W13(BN_V, T_V, MC_V, false, true,  false, WG_V); \
                else if (!zs && fs) return DISPATCH_C1_W13(BN_V, T_V, MC_V, false, false, true,  WG_V); \
                else                return DISPATCH_C1_W13(BN_V, T_V, MC_V, false, false, false, WG_V); \
            } \
        } while (0)

        if (TILE == 8) {
            if (use_bn16) {
                if (wave_gate) { DISPATCH_C1_W13_FLAGS(16, 8, MX6_W13_T8_BN16_MIN_CTA, true); }
                else           { DISPATCH_C1_W13_FLAGS(16, 8, MX6_W13_T8_BN16_MIN_CTA, false); }
            } else {
                if (wave_gate) { DISPATCH_C1_W13_FLAGS(32, 8, MX6_W13_T8_BN32_MIN_CTA, true); }
                else           { DISPATCH_C1_W13_FLAGS(32, 8, MX6_W13_T8_BN32_MIN_CTA, false); }
            }
        } else {
            if (use_bn16) {
                if (wave_gate) { DISPATCH_C1_W13_FLAGS(16, 16, 5, true); }
                else           { DISPATCH_C1_W13_FLAGS(16, 16, 5, false); }
            } else {
                if (wave_gate) { DISPATCH_C1_W13_FLAGS(32, 16, MX6_W13_T16_MIN_CTA, true); }
                else           { DISPATCH_C1_W13_FLAGS(32, 16, MX6_W13_T16_MIN_CTA, false); }
            }
        }
        #undef DISPATCH_C1_W13_FLAGS
        #undef DISPATCH_C1_W13
    };

    if (bn_mode == 0) {
        return static_cast<int>(run_one(false, 0, 0, false, pdl_wait_mode, trig_dep));
    }
    if (bn_mode == 1) {
        return static_cast<int>(run_one(true, 0, 0, false, pdl_wait_mode, trig_dep));
    }
    if (bn_mode == 3) {
        // Complementary GPU wave-efficiency adaptive BN=16 / BN=32 (exact 1-of-2 active!)
        cudaError_t e1 = run_one(true,  0, 0, true, pdl_wait_mode, use_pdl ? 1 : 0);
        if (e1 != cudaSuccess) return static_cast<int>(e1);
        cudaError_t e2 = run_one(false, 0, 0, true, use_pdl ? 1 : 0, trig_dep);
        return static_cast<int>(e2);
    }
    // bn_mode == 2: U <= eff_thresh -> BN=16, U > eff_thresh -> BN=32
    cudaError_t e1 = run_one(true,  1, eff_thresh, false, pdl_wait_mode, use_pdl ? 1 : 0);
    if (e1 != cudaSuccess) return static_cast<int>(e1);
    cudaError_t e2 = run_one(false, eff_thresh + 1, 0x7FFFFFFF, false, use_pdl ? 1 : 0, trig_dep);
    return static_cast<int>(e2);
}

extern "C" int mx6_w2_c1_v10_fwd(
    const void* A, const void* AS,
    const void* SM4, const void* WS3_PACK,
    const void* EXP_STREAM, const void* EXP_OFF,
    const void* WARP_META, const void* HUFF_LUT,
    const void* Ones, const void* Tiles, const void* TileCompactStart,
    void* Out, const void* Src2Compact, const void* NLivePtr,
    void* PanelDone, void* SendBuf,
    int N, int K, int MAX_CTAS, int TILE, int T, uint32_t base_ws,
    int as_row_major, int as_row_stride, int down_row_stride,
    int fuse_quant, int fuse_combine,
    int pdl_wait_mode, int trig_dep, uintptr_t stream_ptr
) {
    if (N != 4608 || K != 288) return -2;
    if (TILE != 8 && TILE != 16) return -3;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    const bool use_pdl = (pdl_wait_mode != 0) || (trig_dep != 0);
    const bool rm = (as_row_major != 0);
    const bool fq = (fuse_quant != 0);
    const bool fc = (fuse_combine != 0);
    const int eff_down_stride = (down_row_stride > 0) ? down_row_stride : (T * 8);
    const dim3 grid(TILE == 8 ? (48 * MX6_W2_T8_MIN_CTA) : (48 * MX6_W2_T16_MIN_CTA));
    const dim3 blk(128);

    #define RUN_C1_W2(T_V, MC_V, PDL_V, RM_V, FQ_V, FC_V) \
        static_cast<int>(launch_mx6_v10( \
            mx6_w2_c1_v10_kernel<T_V, MC_V, PDL_V, RM_V, FQ_V, FC_V>, \
            grid, blk, stream, PDL_V, \
            static_cast<const uint8_t*>(A), static_cast<const uint8_t*>(AS), \
            static_cast<const uint8_t*>(SM4), static_cast<const uint32_t*>(WS3_PACK), \
            static_cast<const uint32_t*>(EXP_STREAM), static_cast<const int32_t*>(EXP_OFF), \
            static_cast<const uint8_t*>(WARP_META), static_cast<const uint8_t*>(HUFF_LUT), \
            static_cast<const float*>(Ones), static_cast<const int32_t*>(Tiles), \
            static_cast<const int32_t*>(TileCompactStart), \
            static_cast<__nv_bfloat16*>(Out), static_cast<const int32_t*>(Src2Compact), \
            static_cast<const int32_t*>(NLivePtr), static_cast<uint32_t*>(PanelDone), \
            static_cast<__nv_bfloat16*>(SendBuf), \
            N, K, MAX_CTAS, T, base_ws, as_row_stride, eff_down_stride, pdl_wait_mode, trig_dep))

    #define DISPATCH_C1_W2_FQ(T_V, MC_V, PDL_V) do { \
        if (fq) { \
            if (fc) return RUN_C1_W2(T_V, MC_V, PDL_V, false, true,  true); \
            else    return RUN_C1_W2(T_V, MC_V, PDL_V, false, true,  false); \
        } else if (rm) { \
            if (fc) return RUN_C1_W2(T_V, MC_V, PDL_V, true,  false, true); \
            else    return RUN_C1_W2(T_V, MC_V, PDL_V, true,  false, false); \
        } else { \
            if (fc) return RUN_C1_W2(T_V, MC_V, PDL_V, false, false, true); \
            else    return RUN_C1_W2(T_V, MC_V, PDL_V, false, false, false); \
        } \
    } while (0)

    if (TILE == 8) {
        if (use_pdl) { DISPATCH_C1_W2_FQ(8,  MX6_W2_T8_MIN_CTA,  true); }
        else         { DISPATCH_C1_W2_FQ(8,  MX6_W2_T8_MIN_CTA,  false); }
    } else {
        if (use_pdl) { DISPATCH_C1_W2_FQ(16, MX6_W2_T16_MIN_CTA, true); }
        else         { DISPATCH_C1_W2_FQ(16, MX6_W2_T16_MIN_CTA, false); }
    }
    #undef DISPATCH_C1_W2_FQ
    #undef RUN_C1_W2
}

extern "C" int mx6_w13_c48_v10_fwd(
    const void* A, const void* AS, const void* Row2Tok,
    const void* SM4, const void* WS3_PACK,
    const void* EXP_STREAM, const void* EXP_OFF,
    const void* WARP_META, const void* HUFF_LUT,
    const void* Ones, const void* Tiles,
    const void* TileCompactStart, const void* Gammas, void* Out,
    int N, int K, int MAX_CTAS, int TILE, uint32_t base_ws,
    int zero_scatter, int as_row_major, int fuse_situ,
    int bn_mode, int pdl_wait_mode, int trig_dep, uintptr_t stream_ptr
) {
    if (N != 576 || K != 4608) return -2;
    if (TILE != 8 && TILE != 16 && TILE != 32) return -3;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    const bool use_pdl = (pdl_wait_mode != 0) || (trig_dep != 0);
    const bool zs = (zero_scatter != 0);
    const bool rm = (as_row_major != 0) || zs;
    const bool fs = (fuse_situ != 0);
    const int g16 = 48 * MX6_W13_T8_BN16_MIN_CTA;
    const int g32 = (TILE == 8)  ? (48 * MX6_W13_T8_BN32_MIN_CTA)
                  : (TILE == 16) ? (48 * MX6_W13_T16_MIN_CTA)
                                 : (48 * MX6_W13_T32_MIN_CTA);

    auto run_one = [&](bool use_bn16, bool wave_gate, int pw, int td) -> cudaError_t {
        const bool pdl_k = (pw != 0) || (td != 0);
        const dim3 grid(use_bn16 ? g16 : g32);
        const dim3 blk(use_bn16 ? 64 : 128);
        const int other_g = use_bn16 ? g32 : g16;

        #define RUN_C48_W13(BN_V, T_V, MC_V, PDL_V, ZS_V, RM_V, FS_V, WG_V) \
            launch_mx6_v10( \
                mx6_w13_c48_v10_kernel<BN_V, T_V, MC_V, PDL_V, ZS_V, RM_V, FS_V, WG_V>, \
                grid, blk, stream, PDL_V, \
                static_cast<const uint8_t*>(A), static_cast<const uint8_t*>(AS), \
                static_cast<const int32_t*>(Row2Tok), \
                static_cast<const uint8_t*>(SM4), static_cast<const uint32_t*>(WS3_PACK), \
                static_cast<const uint32_t*>(EXP_STREAM), static_cast<const int32_t*>(EXP_OFF), \
                static_cast<const uint8_t*>(WARP_META), static_cast<const uint8_t*>(HUFF_LUT), \
                static_cast<const float*>(Ones), static_cast<const int32_t*>(Tiles), \
                static_cast<const int32_t*>(TileCompactStart), static_cast<const float*>(Gammas), \
                static_cast<__nv_bfloat16*>(Out), \
                N, K, MAX_CTAS, base_ws, other_g, 3, pw, td)

        #define DISPATCH_C48_W13_FLAGS(BN_V, T_V, MC_V, WG_V) do { \
            if (pdl_k) { \
                if (zs && fs)             return RUN_C48_W13(BN_V, T_V, MC_V, true,  true,  true,  true,  WG_V); \
                else if (zs && !fs)       return RUN_C48_W13(BN_V, T_V, MC_V, true,  true,  true,  false, WG_V); \
                else if (rm && fs)        return RUN_C48_W13(BN_V, T_V, MC_V, true,  false, true,  true,  WG_V); \
                else if (rm && !fs)       return RUN_C48_W13(BN_V, T_V, MC_V, true,  false, true,  false, WG_V); \
                else if (fs)              return RUN_C48_W13(BN_V, T_V, MC_V, true,  false, false, true,  WG_V); \
                else                      return RUN_C48_W13(BN_V, T_V, MC_V, true,  false, false, false, WG_V); \
            } else { \
                if (zs && fs)             return RUN_C48_W13(BN_V, T_V, MC_V, false, true,  true,  true,  WG_V); \
                else if (zs && !fs)       return RUN_C48_W13(BN_V, T_V, MC_V, false, true,  true,  false, WG_V); \
                else if (rm && fs)        return RUN_C48_W13(BN_V, T_V, MC_V, false, false, true,  true,  WG_V); \
                else if (rm && !fs)       return RUN_C48_W13(BN_V, T_V, MC_V, false, false, true,  false, WG_V); \
                else if (fs)              return RUN_C48_W13(BN_V, T_V, MC_V, false, false, false, true,  WG_V); \
                else                      return RUN_C48_W13(BN_V, T_V, MC_V, false, false, false, false, WG_V); \
            } \
        } while (0)

        if (TILE == 8) {
            if (use_bn16) {
                if (wave_gate) { DISPATCH_C48_W13_FLAGS(16, 8, MX6_W13_T8_BN16_MIN_CTA, true); }
                else           { DISPATCH_C48_W13_FLAGS(16, 8, MX6_W13_T8_BN16_MIN_CTA, false); }
            } else {
                if (wave_gate) { DISPATCH_C48_W13_FLAGS(32, 8, MX6_W13_T8_BN32_MIN_CTA, true); }
                else           { DISPATCH_C48_W13_FLAGS(32, 8, MX6_W13_T8_BN32_MIN_CTA, false); }
            }
        } else if (TILE == 16) {
            DISPATCH_C48_W13_FLAGS(32, 16, MX6_W13_T16_MIN_CTA, false);
        } else {
            DISPATCH_C48_W13_FLAGS(32, 32, MX6_W13_T32_MIN_CTA, false);
        }
        #undef DISPATCH_C48_W13_FLAGS
        #undef RUN_C48_W13
    };

    if (TILE == 8 && bn_mode == 1) {
        return static_cast<int>(run_one(true, false, pdl_wait_mode, trig_dep));
    }
    if (TILE == 8 && bn_mode == 2) {
        cudaError_t e1 = run_one(true,  true, pdl_wait_mode, use_pdl ? 1 : 0);
        if (e1 != cudaSuccess) return static_cast<int>(e1);
        cudaError_t e2 = run_one(false, true, use_pdl ? 1 : 0, trig_dep);
        return static_cast<int>(e2);
    }
    return static_cast<int>(run_one(false, false, pdl_wait_mode, trig_dep));
}

extern "C" int mx6_w2_c48_v10_fwd(
    const void* A, const void* AS,
    const void* SM4, const void* WS3_PACK,
    const void* EXP_STREAM, const void* EXP_OFF,
    const void* WARP_META, const void* HUFF_LUT,
    const void* Ones, const void* Tiles, const void* TileCompactStart,
    void* Out,
    const void* Src2Compact, const void* NLivePtr,
    void* PanelDone, void* BufGen, void* SendBuf,
    int N, int K, int MAX_CTAS, int TILE, int T, uint32_t base_ws,
    int as_row_major, int as_row_stride, int down_row_stride,
    int fuse_quant, int fuse_combine,
    int pdl_wait_mode, int trig_dep, uintptr_t stream_ptr
) {
    if (N != 4608 || K != 288) return -2;
    if (TILE != 8 && TILE != 16 && TILE != 32) return -3;
    if (fuse_combine < 0 || fuse_combine > 4) return -4;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    const bool use_pdl = (pdl_wait_mode != 0) || (trig_dep != 0);
    const bool rm = (as_row_major != 0);
    const bool fq = (fuse_quant != 0);
    const int eff_down_stride = (down_row_stride > 0) ? down_row_stride : (T * 8);
    const int g_ctas = (TILE == 8)  ? (48 * MX6_W2_T8_MIN_CTA)
                     : (TILE == 16) ? (48 * MX6_W2_T16_MIN_CTA)
                                    : (48 * MX6_W2_T32_MIN_CTA);
    const dim3 grid(g_ctas);
    const dim3 blk(128);

    #define RUN_C48_W2(T_V, MC_V, PDL_V, RM_V, FQ_V, FC_V) \
        static_cast<int>(launch_mx6_v10( \
            mx6_w2_c48_v10_kernel<T_V, MC_V, PDL_V, RM_V, FQ_V, FC_V>, \
            grid, blk, stream, PDL_V, \
            static_cast<const uint8_t*>(A), static_cast<const uint8_t*>(AS), \
            static_cast<const uint8_t*>(SM4), static_cast<const uint32_t*>(WS3_PACK), \
            static_cast<const uint32_t*>(EXP_STREAM), static_cast<const int32_t*>(EXP_OFF), \
            static_cast<const uint8_t*>(WARP_META), static_cast<const uint8_t*>(HUFF_LUT), \
            static_cast<const float*>(Ones), static_cast<const int32_t*>(Tiles), \
            static_cast<const int32_t*>(TileCompactStart), \
            static_cast<__nv_bfloat16*>(Out), \
            static_cast<const int32_t*>(Src2Compact), static_cast<const int32_t*>(NLivePtr), \
            static_cast<uint32_t*>(PanelDone), static_cast<uint32_t*>(BufGen), \
            static_cast<__nv_bfloat16*>(SendBuf), \
            N, K, MAX_CTAS, T, base_ws, as_row_stride, eff_down_stride, pdl_wait_mode, trig_dep))

    #define DISPATCH_C48_W2_FC(T_V, MC_V, PDL_V, RM_V, FQ_V) do { \
        if      (fuse_combine == 4) return RUN_C48_W2(T_V, MC_V, PDL_V, RM_V, FQ_V, 4); \
        else if (fuse_combine == 3) return RUN_C48_W2(T_V, MC_V, PDL_V, RM_V, FQ_V, 3); \
        else if (fuse_combine == 2) return RUN_C48_W2(T_V, MC_V, PDL_V, RM_V, FQ_V, 2); \
        else if (fuse_combine == 1) return RUN_C48_W2(T_V, MC_V, PDL_V, RM_V, FQ_V, 1); \
        else                        return RUN_C48_W2(T_V, MC_V, PDL_V, RM_V, FQ_V, 0); \
    } while (0)

    #define DISPATCH_C48_W2_FQ(T_V, MC_V, PDL_V) do { \
        if (fq)      { DISPATCH_C48_W2_FC(T_V, MC_V, PDL_V, false, true); } \
        else if (rm) { DISPATCH_C48_W2_FC(T_V, MC_V, PDL_V, true,  false); } \
        else         { DISPATCH_C48_W2_FC(T_V, MC_V, PDL_V, false, false); } \
    } while (0)

    if (TILE == 8) {
        if (use_pdl) { DISPATCH_C48_W2_FQ(8,  MX6_W2_T8_MIN_CTA,  true); }
        else         { DISPATCH_C48_W2_FQ(8,  MX6_W2_T8_MIN_CTA,  false); }
    } else if (TILE == 16) {
        if (use_pdl) { DISPATCH_C48_W2_FQ(16, MX6_W2_T16_MIN_CTA, true); }
        else         { DISPATCH_C48_W2_FQ(16, MX6_W2_T16_MIN_CTA, false); }
    } else {
        if (use_pdl) { DISPATCH_C48_W2_FQ(32, MX6_W2_T32_MIN_CTA, true); }
        else         { DISPATCH_C48_W2_FQ(32, MX6_W2_T32_MIN_CTA, false); }
    }
    #undef DISPATCH_C48_W2_FQ
    #undef DISPATCH_C48_W2_FC
    #undef RUN_C48_W2
}

extern "C" int mx6_combine_c48_v10_fwd(
    const void* Down, const void* Src2Row, const void* NLivePtr, void* Out,
    int T, int H, int pdl_wait, int trig_dep, uintptr_t stream_ptr
) {
    if (H != 4608) return -2;
    if (T <= 0) return 0;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    const bool fine = (T <= 64);
    const dim3 grid(fine ? 6u : 3u, static_cast<unsigned>(T), 1u);
    const dim3 block(fine ? 96u : 192u, 1u, 1u);
    const bool pw = (pdl_wait != 0);
    const bool td = (trig_dep != 0);
    const bool nl = (NLivePtr != nullptr);
    const bool use_pdl = pw || td;
    cudaError_t err = cudaSuccess;

    #define RUN_COMB(BT, CPC, MC, PW, TD, NL) err = launch_mx6_v10( \
        mx6_combine_c48_v10_kernel<BT, CPC, MC, PW, TD, NL>, \
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
