// SPDX-License-Identifier: Apache-2.0
// SM121a (GB10) Production 0-ULP Lossless KV-BF12 (BF12-G128-OVF-SAFE) Paged KV Codec Header v10
//
// Shared by:
//   - kv_bf12_paged_v10.cu               (libkv_bf12_paged_v10.so: prefill store / unpack / block reset)
//   - fused_decode_attn_exact_sm121_v10.cu (libfused_decode_attn_exact_sm121_v10.so: decode prologue & attn)
//   - Offline C++17 bit-exact simulator   (pure host helpers share identical bitfield/swizzle math)
//
// Exact Fixed-Size Paged Layout (100% 0-ULP lossless across all 65,536 BF16 bit patterns):
//   Per K-or-V half-block (BS=256 tokens, D=128 dims, 128-byte aligned):
//     sm[256][128]   = 32,768 B  (128 B/tok: sign[7] | mantissa[6:0])
//     e4[256][64]    = 16,384 B  ( 64 B/tok: 2 x 4-bit exponent delta codes/byte)
//     meta[256]      =  1,024 B  (  4 B/tok: 2-bit mode + base_e + inline escape / ovf slice / emerg slot)
//     blk_ovf[248]   =    496 B  (248 x uint16_t: (dim_idx:7b << 8) | exp:8b)
//     hdr            =     16 B  (uint32_t ovf_used + 3 x uint32_t pad)
//     TOTAL          = 50,688 B  (49.500 KiB per half -> 101,376 B = 99.000 KiB per K+V page)
//
//   Physical KV-Cache Page View (Uniform 99 KiB Single-Pool in vLLM V1):
//     AttentionSpec(block_size=256, head_size=128, dtype=bfloat16, state_content_bytes=396)
//     For each (block_id b, kv_head h): 101,376 contiguous bytes:
//       K half-block at byte offset: b * blk_stride_bytes + h * head_stride_bytes + 0
//       V half-block at byte offset: b * blk_stride_bytes + h * head_stride_bytes + 50688
//
//   4-bit Exponent Code Table (relative to per-token normal finite max_e in [0..254]):
//     c in [0..13] : normal finite exponent e = base_e - c
//     c == 14      : e == 0 (+0, -0, AND all 254 BF16 subnormals — 0 escapes!)
//     c == 15      : escape (tiny normal e < base_e - 13, OR e == 255 Inf/NaN)
//
//   3-Tier Overflow Hierarchy (encoded in meta[31:30]):
//     mode 0 (0 escapes, ~97.0%):   meta = (0<<30) | base_e
//     mode 1 (1 escape,   ~2.9%):   meta = (1<<30) | (((idx<<8)|exp) << 8) | base_e
//     mode 2 (2..248 esc, ~0.06%):  meta = (2<<30) | (cnt<<16) | (off<<8) | base_e  (blk_ovf[off..off+cnt-1])
//     mode 3 (block ovf,  0.000%):  meta = (3<<30) | emerg_slot                     (EmergencyRawPool[eslot])
//
// Strict SM121a Hygiene:
//   - Zero local stack arrays with runtime indexing (0 B stack / 0 B spills).
//   - Zero atomics on Modes 0 & 1 (99.94%+ of tokens).
//   - atomicCAS reservation on blk_ovf (never overflows blk_ovf[248], zero wasted slots).
//   - In-place 2-stage 128-thread SMEM reverse/split unpack (0 extra SMEM bytes, exact V9 XOR swizzle).

#pragma once

#include <stdint.h>
#include <stddef.h>

static constexpr int KV_BF12_D           = 128;
static constexpr int KV_BF12_BS          = 256;
static constexpr int KV_BF12_BS_SHIFT    = 8;
static constexpr int KV_BF12_BS_MASK     = 255;
static constexpr int KV_BF12_OVF_CAP     = 248;
static constexpr int KV_BF12_HALF_BYTES  = 50688;   // 49.500 KiB (396 * 128 B, 128B aligned)
static constexpr int KV_BF12_PAGE_BYTES  = 101376;  // 99.000 KiB (2 * 50,688 B, 256B aligned)
static constexpr int KV_BF12_TILE_BN     = 64;
static constexpr int KV_BF12_SM_OFF_B    = 0;       // [0     ..  8192) = 8,192 B
static constexpr int KV_BF12_E4_OFF_B    = 8192;    // [8192  .. 12288) = 4,096 B
static constexpr int KV_BF12_META_OFF_B  = 12288;   // [12288 .. 12544) =   256 B
static constexpr int KV_BF12_BLKID_OFF_B = 12544;   // [12544 .. 12800) =   256 B

struct alignas(16) KvBf12HalfBlock {
    uint8_t  sm[KV_BF12_BS][KV_BF12_D];        // [0     .. 32768) : 32,768 B
    uint8_t  e4[KV_BF12_BS][KV_BF12_D / 2];    // [32768 .. 49152) : 16,384 B
    uint32_t meta[KV_BF12_BS];                 // [49152 .. 50176) :  1,024 B
    uint16_t blk_ovf[KV_BF12_OVF_CAP];         // [50176 .. 50672) :    496 B
    uint32_t ovf_used;                         // [50672 .. 50676) :      4 B
    uint32_t _pad[3];                          // [50676 .. 50688) :     12 B
};

static_assert(sizeof(KvBf12HalfBlock) == KV_BF12_HALF_BYTES,
              "KvBf12HalfBlock must be exactly 50,688 B (49.500 KiB)");
static_assert(offsetof(KvBf12HalfBlock, sm)       == 0,     "sm offset mismatch");
static_assert(offsetof(KvBf12HalfBlock, e4)       == 32768, "e4 offset mismatch");
static_assert(offsetof(KvBf12HalfBlock, meta)     == 49152, "meta offset mismatch");
static_assert(offsetof(KvBf12HalfBlock, blk_ovf)  == 50176, "blk_ovf offset mismatch");
static_assert(offsetof(KvBf12HalfBlock, ovf_used) == 50672, "ovf_used offset mismatch");

#if defined(__CUDACC__) || defined(KV_BF12_ENABLE_DEVICE_HELPERS)

// =========================================================================
// Block Pointer Helpers (supports arbitrary LBNHC / BLHNC block strides)
// =========================================================================
static __device__ __host__ __forceinline__ KvBf12HalfBlock* kv_bf12_half_blk(
    uint8_t* base_bytes, int64_t blk_stride_bytes, int64_t blk_id
) {
    return reinterpret_cast<KvBf12HalfBlock*>(base_bytes + blk_id * blk_stride_bytes);
}

static __device__ __host__ __forceinline__ const KvBf12HalfBlock* kv_bf12_half_blk_const(
    const uint8_t* base_bytes, int64_t blk_stride_bytes, int64_t blk_id
) {
    return reinterpret_cast<const KvBf12HalfBlock*>(base_bytes + blk_id * blk_stride_bytes);
}

// =========================================================================
// Bit-Exact Scalar Reconstruction Primitives (shared by CUDA & C++17 sim)
// =========================================================================
static __device__ __host__ __forceinline__ uint16_t kv_bf12_reconstruct_one(
    uint32_t sm_byte, uint32_t c_nib, uint32_t base_e
) {
    const uint32_t e = (c_nib < 14u) ? (base_e - c_nib) : 0u;
    return static_cast<uint16_t>(((sm_byte & 0x80u) << 8) | (e << 7) | (sm_byte & 0x7Fu));
}

static __device__ __host__ __forceinline__ uint4 kv_bf12_unpack8_to_u4(
    uint32_t sm_lo4, uint32_t sm_hi4, uint32_t e4_8, uint32_t base_e
) {
    const uint32_t u0 = kv_bf12_reconstruct_one((sm_lo4 >>  0) & 0xFFu, (e4_8 >>  0) & 0xFu, base_e);
    const uint32_t u1 = kv_bf12_reconstruct_one((sm_lo4 >>  8) & 0xFFu, (e4_8 >>  4) & 0xFu, base_e);
    const uint32_t u2 = kv_bf12_reconstruct_one((sm_lo4 >> 16) & 0xFFu, (e4_8 >>  8) & 0xFu, base_e);
    const uint32_t u3 = kv_bf12_reconstruct_one((sm_lo4 >> 24) & 0xFFu, (e4_8 >> 12) & 0xFu, base_e);
    const uint32_t u4 = kv_bf12_reconstruct_one((sm_hi4 >>  0) & 0xFFu, (e4_8 >> 16) & 0xFu, base_e);
    const uint32_t u5 = kv_bf12_reconstruct_one((sm_hi4 >>  8) & 0xFFu, (e4_8 >> 20) & 0xFu, base_e);
    const uint32_t u6 = kv_bf12_reconstruct_one((sm_hi4 >> 16) & 0xFFu, (e4_8 >> 24) & 0xFu, base_e);
    const uint32_t u7 = kv_bf12_reconstruct_one((sm_hi4 >> 24) & 0xFFu, (e4_8 >> 28) & 0xFu, base_e);
    return make_uint4(
        u0 | (u1 << 16),
        u2 | (u3 << 16),
        u4 | (u5 << 16),
        u6 | (u7 << 16)
    );
}

static __device__ __host__ __forceinline__ uint32_t kv_bf12_swiz_col(int row, uint32_t dim_idx) {
    const uint32_t chunk = dim_idx >> 3;  // 0..15 (16B chunk)
    const uint32_t swiz_c = chunk ^ (static_cast<uint32_t>(row) & 7u);
    return (swiz_c << 3) | (dim_idx & 7u);
}

#endif  // __CUDACC__ || KV_BF12_ENABLE_DEVICE_HELPERS

#if defined(__CUDACC__) || defined(KV_BF12_HOST_SYNTAX_CHECK)

// =========================================================================
// Inline PTX cp.async.cg Helpers (scoped uniquely to avoid name clashes)
// =========================================================================
static __device__ __forceinline__ void kv_bf12_cp_async_cg_16(
    uint32_t smem_addr, const void* gmem_ptr
) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr), "l"(gmem_ptr));
}

static __device__ __forceinline__ void kv_bf12_cp_async_cg_16_evict_first(
    uint32_t smem_addr, const void* gmem_ptr, uint64_t cache_policy
) {
#if __CUDA_ARCH__ >= 900
    asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;\n"
                 :: "r"(smem_addr), "l"(gmem_ptr), "l"(cache_policy));
#else
    (void)cache_policy;
    kv_bf12_cp_async_cg_16(smem_addr, gmem_ptr);
#endif
}

// =========================================================================
// Warp-Level 0-ULP Lossless KV-BF12 Encoder (32 lanes x 4 dims = 128 dims)
//   - Lane `lane` (0..31) owns contiguous BF16 elements (u0, u1, u2, u3)
//     at dims (4*lane + 0, +1, +2, +3) — exact match for run_kv_tok_warp!
//   - 0 atomics on Modes 0 & 1 (99.94%+ of Inkling-2 K/V tokens).
//   - atomicCAS reservation on blk->ovf_used for Mode 2 (never overflows 248).
//   - Tier-3 EmergencyRawPool fallback on Mode 3 (guaranteed lossless).
//   - 0 shared memory, 0 local stack arrays, ~18 instructions on fast path.
// =========================================================================
static __device__ __forceinline__ void kv_bf12_encode_warp_store(
    uint32_t u0, uint32_t u1, uint32_t u2, uint32_t u3,
    int lane,
    KvBf12HalfBlock* __restrict__ blk,
    int slot,
    uint16_t* __restrict__ emerg_raw,
    uint32_t* __restrict__ emerg_used,
    uint32_t emerg_cap
) {
    // Pass 1: find max_e over normal finite exponents 1..254 (ignoring e==0 & e==255!)
    const uint32_t e0 = (u0 >> 7) & 0xFFu;
    const uint32_t e1 = (u1 >> 7) & 0xFFu;
    const uint32_t e2 = (u2 >> 7) & 0xFFu;
    const uint32_t e3 = (u3 >> 7) & 0xFFu;

    uint32_t me = 0u;
    if (e0 >= 1u && e0 <= 254u && e0 > me) me = e0;
    if (e1 >= 1u && e1 <= 254u && e1 > me) me = e1;
    if (e2 >= 1u && e2 <= 254u && e2 > me) me = e2;
    if (e3 >= 1u && e3 <= 254u && e3 > me) me = e3;

    #pragma unroll
    for (int off = 16; off >= 1; off >>= 1) {
        const uint32_t o = __shfl_xor_sync(0xFFFFFFFFu, me, off);
        if (o > me) me = o;
    }

    // Pass 2: compute sign|mantissa bytes + 4-bit exponent codes + escape flags
    uint32_t c0, c1, c2, c3;
    int esc0, esc1, esc2, esc3;

    const uint32_t sm0 = ((u0 >> 8) & 0x80u) | (u0 & 0x7Fu);
    if (e0 == 0u)                                     { c0 = 14u;     esc0 = 0; }
    else if (e0 <= 254u && me >= e0 && (me - e0) <= 13u) { c0 = me - e0; esc0 = 0; }
    else                                              { c0 = 15u;     esc0 = 1; }

    const uint32_t sm1 = ((u1 >> 8) & 0x80u) | (u1 & 0x7Fu);
    if (e1 == 0u)                                     { c1 = 14u;     esc1 = 0; }
    else if (e1 <= 254u && me >= e1 && (me - e1) <= 13u) { c1 = me - e1; esc1 = 0; }
    else                                              { c1 = 15u;     esc1 = 1; }

    const uint32_t sm2 = ((u2 >> 8) & 0x80u) | (u2 & 0x7Fu);
    if (e2 == 0u)                                     { c2 = 14u;     esc2 = 0; }
    else if (e2 <= 254u && me >= e2 && (me - e2) <= 13u) { c2 = me - e2; esc2 = 0; }
    else                                              { c2 = 15u;     esc2 = 1; }

    const uint32_t sm3 = ((u3 >> 8) & 0x80u) | (u3 & 0x7Fu);
    if (e3 == 0u)                                     { c3 = 14u;     esc3 = 0; }
    else if (e3 <= 254u && me >= e3 && (me - e3) <= 13u) { c3 = me - e3; esc3 = 0; }
    else                                              { c3 = 15u;     esc3 = 1; }

    // Coalesced 128 B sm store + 64 B e4 store (1 cache-line transaction each)
    const uint32_t sm_pack = sm0 | (sm1 << 8) | (sm2 << 16) | (sm3 << 24);
    *reinterpret_cast<uint32_t*>(&blk->sm[slot][lane * 4]) = sm_pack;

    const uint16_t e4_pack = static_cast<uint16_t>(c0 | (c1 << 4) | (c2 << 8) | (c3 << 12));
    *reinterpret_cast<uint16_t*>(&blk->e4[slot][lane * 2]) = e4_pack;

    // Warp-wide escape ballot (0 atomics on Modes 0 & 1!)
    const uint32_t m0 = __ballot_sync(0xFFFFFFFFu, esc0);
    const uint32_t m1 = __ballot_sync(0xFFFFFFFFu, esc1);
    const uint32_t m2 = __ballot_sync(0xFFFFFFFFu, esc2);
    const uint32_t m3 = __ballot_sync(0xFFFFFFFFu, esc3);
    const uint32_t total_esc = static_cast<uint32_t>(
        __popc(m0) + __popc(m1) + __popc(m2) + __popc(m3)
    );

    if (total_esc == 0u) {
        // Mode 0: Fast inline (0 escapes)
        if (lane == 0) {
            blk->meta[slot] = me & 0xFFu;
        }
        return;
    }

    if (total_esc == 1u) {
        // Mode 1: Fast inline (1 escape packed directly into meta[22:8])
        int esc_lane;
        uint32_t esc_sub;
        if (m0 != 0u)      { esc_lane = __ffs(static_cast<int>(m0)) - 1; esc_sub = 0u; }
        else if (m1 != 0u) { esc_lane = __ffs(static_cast<int>(m1)) - 1; esc_sub = 1u; }
        else if (m2 != 0u) { esc_lane = __ffs(static_cast<int>(m2)) - 1; esc_sub = 2u; }
        else               { esc_lane = __ffs(static_cast<int>(m3)) - 1; esc_sub = 3u; }
        if (lane == esc_lane) {
            const uint32_t idx = static_cast<uint32_t>(lane * 4) + esc_sub;
            const uint32_t exp = (esc_sub == 0u) ? e0 : (esc_sub == 1u) ? e1 : (esc_sub == 2u) ? e2 : e3;
            const uint32_t rec = ((idx & 0x7Fu) << 8) | (exp & 0xFFu);
            blk->meta[slot] = (1u << 30) | (rec << 8) | (me & 0xFFu);
        }
        return;
    }

    // Mode 2: Tier-2 Block-Local OVF pool (atomicCAS guarantees ovf_used <= 248)
    uint32_t ovf_off = 0xFFFFFFFFu;
    if (lane == 0 && total_esc <= static_cast<uint32_t>(KV_BF12_OVF_CAP)) {
        uint32_t cur = blk->ovf_used;
        while (cur + total_esc <= static_cast<uint32_t>(KV_BF12_OVF_CAP)) {
            const uint32_t prev = atomicCAS(&blk->ovf_used, cur, cur + total_esc);
            if (prev == cur) {
                ovf_off = cur;
                break;
            }
            cur = prev;
        }
    }
    ovf_off = __shfl_sync(0xFFFFFFFFu, ovf_off, 0);

    if (ovf_off != 0xFFFFFFFFu) {
        const uint32_t lt_mask = (1u << lane) - 1u;
        uint32_t p = static_cast<uint32_t>(
            __popc(m0 & lt_mask) + __popc(m1 & lt_mask) +
            __popc(m2 & lt_mask) + __popc(m3 & lt_mask)
        );
        if (esc0) blk->blk_ovf[ovf_off + p++] = static_cast<uint16_t>(((lane * 4 + 0) << 8) | e0);
        if (esc1) blk->blk_ovf[ovf_off + p++] = static_cast<uint16_t>(((lane * 4 + 1) << 8) | e1);
        if (esc2) blk->blk_ovf[ovf_off + p++] = static_cast<uint16_t>(((lane * 4 + 2) << 8) | e2);
        if (esc3) blk->blk_ovf[ovf_off + p++] = static_cast<uint16_t>(((lane * 4 + 3) << 8) | e3);
        if (lane == 0) {
            blk->meta[slot] = (2u << 30)
                            | ((total_esc & 0xFFu) << 16)
                            | ((ovf_off & 0xFFu) << 8)
                            | (me & 0xFFu);
        }
        return;
    }

    // Mode 3: Tier-3 Emergency Raw BF16 fallback (only on adversarial block overflow)
    uint32_t eslot = 0u;
    if (lane == 0 && emerg_used != nullptr && emerg_cap > 0u) {
        const uint32_t s = atomicAdd(emerg_used, 1u);
        eslot = (s < emerg_cap) ? s : (emerg_cap - 1u);
    }
    eslot = __shfl_sync(0xFFFFFFFFu, eslot, 0);
    if (emerg_raw != nullptr && emerg_cap > 0u) {
        uint16_t* dst = emerg_raw + static_cast<size_t>(eslot) * KV_BF12_D + lane * 4;
        *reinterpret_cast<uint32_t*>(dst + 0) = (u0 & 0xFFFFu) | ((u1 & 0xFFFFu) << 16);
        *reinterpret_cast<uint32_t*>(dst + 2) = (u2 & 0xFFFFu) | ((u3 & 0xFFFFu) << 16);
    }
    if (lane == 0) {
        blk->meta[slot] = (3u << 30) | (eslot & 0x3FFFFFFFu);
    }
}

// =========================================================================
// 128-Thread Async Tile Loader for BN=64 KV-BF12 Tile (12,544 B vs 16,384 B!)
//   Staging layout inside the 16,384 B `smem_kv` buffer:
//     [0     ..  8192) : sm[64][128]   (4 x cp.async.cg.16 per thread)
//     [8192  .. 12288) : e4[64][64]    (2 x cp.async.cg.16 per thread)
//     [12288 .. 12544) : meta[64]      (threads 0..63 scalar __ldg)
//     [12544 .. 12800) : blk_id[64]    (threads 0..63 stash KV_blk_slot.x so V(n0)
//                                       unpack stays 100% valid after KV_blk_slot
//                                       advances to n0+1!)
//   NOTE: Caller must ensure this is NEVER called before gdc_wait() under USE_PDL!
// =========================================================================
static __device__ __forceinline__ void kv_bf12_issue_tile64_async(
    const uint8_t* __restrict__ kv_half_base,
    int64_t blk_stride_bytes,
    const int2* __restrict__ blk_slot,
    uint32_t kv_sptr,
    uint8_t* __restrict__ smem_kv_u8,
    bool evict_first,
    uint64_t l2_policy,
    int tid
) {
    // 1. Issue sm[64][128] = 512 x 16B = 4 iterations across 128 threads
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int idx = i * 128 + tid;  // 0..511
        const int row = idx >> 3;       // 0..63
        const int c   = idx & 7;        // 0..7 (16 B each)
        const int2 bs = blk_slot[row];
        if (bs.x >= 0) {
            const KvBf12HalfBlock* blk = kv_bf12_half_blk_const(kv_half_base, blk_stride_bytes, bs.x);
            const uint32_t s_addr = kv_sptr + static_cast<uint32_t>(KV_BF12_SM_OFF_B + row * 128 + c * 16);
            const void* g_ptr = &blk->sm[bs.y][c * 16];
            if (evict_first) {
                kv_bf12_cp_async_cg_16_evict_first(s_addr, g_ptr, l2_policy);
            } else {
                kv_bf12_cp_async_cg_16(s_addr, g_ptr);
            }
        }
    }

    // 2. Issue e4[64][64] = 256 x 16B = 2 iterations across 128 threads
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int idx = i * 128 + tid;  // 0..255
        const int row = idx >> 2;       // 0..63
        const int c   = idx & 3;        // 0..3 (16 B each)
        const int2 bs = blk_slot[row];
        if (bs.x >= 0) {
            const KvBf12HalfBlock* blk = kv_bf12_half_blk_const(kv_half_base, blk_stride_bytes, bs.x);
            const uint32_t s_addr = kv_sptr + static_cast<uint32_t>(KV_BF12_E4_OFF_B + row * 64 + c * 16);
            const void* g_ptr = &blk->e4[bs.y][c * 16];
            if (evict_first) {
                kv_bf12_cp_async_cg_16_evict_first(s_addr, g_ptr, l2_policy);
            } else {
                kv_bf12_cp_async_cg_16(s_addr, g_ptr);
            }
        }
    }

    // 3. Load meta[64] + stash blk_id[64] in free SMEM tail [12288..12800)
    if (tid < KV_BF12_TILE_BN) {
        const int2 bs = blk_slot[tid];
        uint32_t* smem_meta  = reinterpret_cast<uint32_t*>(smem_kv_u8 + KV_BF12_META_OFF_B);
        int32_t*  smem_blkid = reinterpret_cast<int32_t*>(smem_kv_u8 + KV_BF12_BLKID_OFF_B);
        smem_blkid[tid] = bs.x;
        if (bs.x >= 0) {
            const KvBf12HalfBlock* blk = kv_bf12_half_blk_const(kv_half_base, blk_stride_bytes, bs.x);
            smem_meta[tid] = __ldg(&blk->meta[bs.y]);
        } else {
            smem_meta[tid] = 0u;
        }
    }
}

// Helper: unpack one 32-dim quarter (`qtr` in 0..3 -> dims [qtr*32 .. qtr*32+31]) of `row`
// directly into exact V9 XOR-swizzled `smem_kv` (`64 x 128` BF16 = 16,384 B).
static __device__ __forceinline__ void kv_bf12_unpack_quarter_swiz(
    uint16_t* __restrict__ smem_kv,
    int row, int qtr,
    uint4 sm_a, uint4 sm_b, uint4 e4_q,
    uint32_t meta, int32_t blk_id,
    const uint8_t* __restrict__ kv_half_base,
    int64_t blk_stride_bytes,
    const uint16_t* __restrict__ emerg_raw
) {
    uint4* smem_kv4 = reinterpret_cast<uint4*>(smem_kv);
    const int c_base = qtr * 4;
    const int row_mask = row & 7;

    if (blk_id < 0) {
        const uint4 z = make_uint4(0u, 0u, 0u, 0u);
        smem_kv4[row * 16 + ((c_base + 0) ^ row_mask)] = z;
        smem_kv4[row * 16 + ((c_base + 1) ^ row_mask)] = z;
        smem_kv4[row * 16 + ((c_base + 2) ^ row_mask)] = z;
        smem_kv4[row * 16 + ((c_base + 3) ^ row_mask)] = z;
        return;
    }

    const uint32_t mode = meta >> 30;
    if (mode == 3u) {
        if (emerg_raw != nullptr) {
            const uint32_t eslot = meta & 0x3FFFFFFFu;
            const uint4* src = reinterpret_cast<const uint4*>(
                emerg_raw + static_cast<size_t>(eslot) * KV_BF12_D + qtr * 32
            );
            smem_kv4[row * 16 + ((c_base + 0) ^ row_mask)] = __ldg(src + 0);
            smem_kv4[row * 16 + ((c_base + 1) ^ row_mask)] = __ldg(src + 1);
            smem_kv4[row * 16 + ((c_base + 2) ^ row_mask)] = __ldg(src + 2);
            smem_kv4[row * 16 + ((c_base + 3) ^ row_mask)] = __ldg(src + 3);
        } else {
            const uint4 z = make_uint4(0u, 0u, 0u, 0u);
            smem_kv4[row * 16 + ((c_base + 0) ^ row_mask)] = z;
            smem_kv4[row * 16 + ((c_base + 1) ^ row_mask)] = z;
            smem_kv4[row * 16 + ((c_base + 2) ^ row_mask)] = z;
            smem_kv4[row * 16 + ((c_base + 3) ^ row_mask)] = z;
        }
        return;
    }

    const uint32_t base_e = meta & 0xFFu;
    smem_kv4[row * 16 + ((c_base + 0) ^ row_mask)] = kv_bf12_unpack8_to_u4(sm_a.x, sm_a.y, e4_q.x, base_e);
    smem_kv4[row * 16 + ((c_base + 1) ^ row_mask)] = kv_bf12_unpack8_to_u4(sm_a.z, sm_a.w, e4_q.y, base_e);
    smem_kv4[row * 16 + ((c_base + 2) ^ row_mask)] = kv_bf12_unpack8_to_u4(sm_b.x, sm_b.y, e4_q.z, base_e);
    smem_kv4[row * 16 + ((c_base + 3) ^ row_mask)] = kv_bf12_unpack8_to_u4(sm_b.z, sm_b.w, e4_q.w, base_e);

    if (mode == 1u) {
        const uint32_t rec = (meta >> 8) & 0x7FFFu;
        const uint32_t idx = (rec >> 8) & 0x7Fu;
        if ((idx >> 5) == static_cast<uint32_t>(qtr)) {
            const uint32_t exp = rec & 0xFFu;
            uint16_t* p = smem_kv + row * KV_BF12_D + kv_bf12_swiz_col(row, idx);
            *p = static_cast<uint16_t>((*p & 0x807Fu) | (exp << 7));
        }
    } else if (mode == 2u) {
        const uint32_t off = (meta >> 8) & 0xFFu;
        const uint32_t cnt = (meta >> 16) & 0xFFu;
        const KvBf12HalfBlock* blk = kv_bf12_half_blk_const(kv_half_base, blk_stride_bytes, blk_id);
        for (uint32_t i = 0u; i < cnt; ++i) {
            const uint32_t rec = static_cast<uint32_t>(__ldg(&blk->blk_ovf[off + i]));
            const uint32_t idx = (rec >> 8) & 0x7Fu;
            if ((idx >> 5) == static_cast<uint32_t>(qtr)) {
                const uint32_t exp = rec & 0xFFu;
                uint16_t* p = smem_kv + row * KV_BF12_D + kv_bf12_swiz_col(row, idx);
                *p = static_cast<uint16_t>((*p & 0x807Fu) | (exp << 7));
            }
        }
    }
}

// =========================================================================
// 128-Thread In-Place 2-Stage Unpack into Exact V9 XOR-Swizzled BF16 SMEM
//   - Precondition: cp_async_wait_group + __syncthreads() already completed!
//   - ZERO extra SMEM bytes (operates completely in-place on 16,384 B smem_kv).
//   - Peak extra registers: ~14 regs/thread (0 local stack arrays, 0 spills).
//   - Postcondition: ends with __syncthreads(); all 64x128 BF16 rows are in exact
//     V9 XOR-swizzled layout ready for unchanged ldmatrix.x4 / ldmatrix.x4.trans!
// =========================================================================
static __device__ __forceinline__ void kv_bf12_unpack_tile64_inplace_swiz(
    uint16_t* __restrict__ smem_kv,
    const uint8_t* __restrict__ kv_half_base,
    int64_t blk_stride_bytes,
    const uint16_t* __restrict__ emerg_raw,
    int tid
) {
    uint8_t* smem_u8 = reinterpret_cast<uint8_t*>(smem_kv);
    const int row_lo = tid >> 2;        // 0..31
    const int row_hi = 32 + row_lo;     // 32..63
    const int qtr    = tid & 3;         // 0..3 (32 dims = 4 x 16B chunks)

    // Stage 0: lift e4[0..63], meta[0..63], blk_id[0..63] into registers -> frees [8192..16384)!
    const uint4 e4_lo = *reinterpret_cast<const uint4*>(
        smem_u8 + KV_BF12_E4_OFF_B + row_lo * 64 + qtr * 16
    );
    const uint4 e4_hi = *reinterpret_cast<const uint4*>(
        smem_u8 + KV_BF12_E4_OFF_B + row_hi * 64 + qtr * 16
    );
    const uint32_t* smem_meta  = reinterpret_cast<const uint32_t*>(smem_u8 + KV_BF12_META_OFF_B);
    const int32_t*  smem_blkid = reinterpret_cast<const int32_t*>(smem_u8 + KV_BF12_BLKID_OFF_B);
    const uint32_t meta_lo = smem_meta[row_lo];
    const uint32_t meta_hi = smem_meta[row_hi];
    const int32_t  blk_lo  = smem_blkid[row_lo];
    const int32_t  blk_hi  = smem_blkid[row_hi];

    // Also preload sm[0..31] (`[0..4096)`) and sm[32..63] (`[4096..8192)`) cleanly:
    // Read sm_hi NOW while [4096..8192) is untouched, wait for everyone to finish reading
    // [8192..12800), then unpack rows 32..63 into [8192..16384)!
    const uint4 sm_hi_a = *reinterpret_cast<const uint4*>(
        smem_u8 + KV_BF12_SM_OFF_B + row_hi * 128 + qtr * 32 + 0
    );
    const uint4 sm_hi_b = *reinterpret_cast<const uint4*>(
        smem_u8 + KV_BF12_SM_OFF_B + row_hi * 128 + qtr * 32 + 16
    );
    __syncthreads();

    // Stage 1: unpack rows 32..63 into [8192..16384) (strictly disjoint from [0..4096)!)
    kv_bf12_unpack_quarter_swiz(
        smem_kv, row_hi, qtr, sm_hi_a, sm_hi_b, e4_hi, meta_hi, blk_hi,
        kv_half_base, blk_stride_bytes, emerg_raw
    );

    // Preload sm[0..31] (`[0..4096)`) now that sm_hi_a/b + e4_hi are dead (reuses same reg slots!)
    const uint4 sm_lo_a = *reinterpret_cast<const uint4*>(
        smem_u8 + KV_BF12_SM_OFF_B + row_lo * 128 + qtr * 32 + 0
    );
    const uint4 sm_lo_b = *reinterpret_cast<const uint4*>(
        smem_u8 + KV_BF12_SM_OFF_B + row_lo * 128 + qtr * 32 + 16
    );
    __syncthreads();

    // Stage 2: unpack rows 0..31 into [0..8192) (all of [0..4096) already lifted to registers!)
    kv_bf12_unpack_quarter_swiz(
        smem_kv, row_lo, qtr, sm_lo_a, sm_lo_b, e4_lo, meta_lo, blk_lo,
        kv_half_base, blk_stride_bytes, emerg_raw
    );
    __syncthreads();
}

// =========================================================================
// Warp-Level Coalesced Linear BF16 Unpacker (32 lanes x 4 dims = 128 dims)
//   Used by kv_bf12_unpack_blocks_v10_fwd and kv_bf12_unpack_range_v10_fwd!
// =========================================================================
static __device__ __forceinline__ void kv_bf12_unpack_warp_row(
    const KvBf12HalfBlock* __restrict__ blk,
    int slot,
    const uint16_t* __restrict__ emerg_raw,
    uint16_t* __restrict__ dst_row128,
    int lane
) {
    const uint32_t meta = __ldg(&blk->meta[slot]);
    const uint32_t mode = meta >> 30;
    if (mode == 3u) {
        if (emerg_raw != nullptr) {
            const uint32_t eslot = meta & 0x3FFFFFFFu;
            const uint16_t* src = emerg_raw + static_cast<size_t>(eslot) * KV_BF12_D + lane * 4;
            const uint32_t p01 = __ldg(reinterpret_cast<const uint32_t*>(src + 0));
            const uint32_t p23 = __ldg(reinterpret_cast<const uint32_t*>(src + 2));
            *reinterpret_cast<uint32_t*>(dst_row128 + lane * 4 + 0) = p01;
            *reinterpret_cast<uint32_t*>(dst_row128 + lane * 4 + 2) = p23;
        } else {
            *reinterpret_cast<uint32_t*>(dst_row128 + lane * 4 + 0) = 0u;
            *reinterpret_cast<uint32_t*>(dst_row128 + lane * 4 + 2) = 0u;
        }
        return;
    }

    const uint32_t base_e = meta & 0xFFu;
    const uint32_t sm4 = __ldg(reinterpret_cast<const uint32_t*>(&blk->sm[slot][lane * 4]));
    const uint32_t e4_2 = static_cast<uint32_t>(
        __ldg(reinterpret_cast<const uint16_t*>(&blk->e4[slot][lane * 2]))
    );

    uint16_t u0 = kv_bf12_reconstruct_one((sm4 >>  0) & 0xFFu, (e4_2 >>  0) & 0xFu, base_e);
    uint16_t u1 = kv_bf12_reconstruct_one((sm4 >>  8) & 0xFFu, (e4_2 >>  4) & 0xFu, base_e);
    uint16_t u2 = kv_bf12_reconstruct_one((sm4 >> 16) & 0xFFu, (e4_2 >>  8) & 0xFu, base_e);
    uint16_t u3 = kv_bf12_reconstruct_one((sm4 >> 24) & 0xFFu, (e4_2 >> 12) & 0xFu, base_e);

    if (mode == 1u) {
        const uint32_t rec = (meta >> 8) & 0x7FFFu;
        const uint32_t idx = (rec >> 8) & 0x7Fu;
        if ((idx >> 2) == static_cast<uint32_t>(lane)) {
            const uint32_t exp = (rec & 0xFFu) << 7;
            const uint32_t sub = idx & 3u;
            if (sub == 0u)      u0 = static_cast<uint16_t>((u0 & 0x807Fu) | exp);
            else if (sub == 1u) u1 = static_cast<uint16_t>((u1 & 0x807Fu) | exp);
            else if (sub == 2u) u2 = static_cast<uint16_t>((u2 & 0x807Fu) | exp);
            else                u3 = static_cast<uint16_t>((u3 & 0x807Fu) | exp);
        }
    } else if (mode == 2u) {
        const uint32_t off = (meta >> 8) & 0xFFu;
        const uint32_t cnt = (meta >> 16) & 0xFFu;
        for (uint32_t i = 0u; i < cnt; ++i) {
            const uint32_t rec = static_cast<uint32_t>(__ldg(&blk->blk_ovf[off + i]));
            const uint32_t idx = (rec >> 8) & 0x7Fu;
            if ((idx >> 2) == static_cast<uint32_t>(lane)) {
                const uint32_t exp = (rec & 0xFFu) << 7;
                const uint32_t sub = idx & 3u;
                if (sub == 0u)      u0 = static_cast<uint16_t>((u0 & 0x807Fu) | exp);
                else if (sub == 1u) u1 = static_cast<uint16_t>((u1 & 0x807Fu) | exp);
                else if (sub == 2u) u2 = static_cast<uint16_t>((u2 & 0x807Fu) | exp);
                else                u3 = static_cast<uint16_t>((u3 & 0x807Fu) | exp);
            }
        }
    }

    *reinterpret_cast<uint32_t*>(dst_row128 + lane * 4 + 0) =
        static_cast<uint32_t>(u0) | (static_cast<uint32_t>(u1) << 16);
    *reinterpret_cast<uint32_t*>(dst_row128 + lane * 4 + 2) =
        static_cast<uint32_t>(u2) | (static_cast<uint32_t>(u3) << 16);
}

#endif  // __CUDACC__ || KV_BF12_HOST_SYNTAX_CHECK
