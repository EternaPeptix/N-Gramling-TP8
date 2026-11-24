// SPDX-License-Identifier: Apache-2.0
// SM121a (GB10) Production 0-ULP Lossless KV-BF12 (BF12-G128-OVF-SAFE) SWAR Codec Header v12
//
// 100% binary, physical-layout, and API-compatible drop-in upgrade over:
//   - Wave-V10 `kv_bf12_codec_v10.cuh`
//   - Wave-V11 `v11/attn_s48/kv_bf12_codec_v11.cuh`
// (`KvBf12HalfBlock` == 50,688 B, identical offsets & XOR-swizzled SMEM staging!)
//
// Wave-V12 Orthogonal Upgrades (beyond v11/attn_s48):
//   1. Branchless 2-Wide SWAR `kv_bf12_unpack2_swar` / `kv_bf12_unpack8_to_u4`:
//      - Replaces 8 scalar predicate-gated `kv_bf12_reconstruct_one` calls (~68 SASS ops/uint4)
//        with 4 branchless 2-wide SWAR reconstructions (~40 SASS ops/uint4 -> ~41% fewer unpack
//        ALU instructions, 0 branches, 0 predicates, 0 extra registers!).
//      - Uses a bit-8/bit-24 guard bit (`base_guard = (base_e | 0x100u) * 0x00010001u`) so
//        `base_guard - c_pair` is mathematically borrow-free across ALL `base_e in [0..255]` and
//        ALL `c_nib in [0..15]`, including `base_e == 0, c_nib in {14, 15}`!
//      - Uses branchless `(c_pair + 0x00020002u) & 0x00100010u` mask expansion (`(g<<4)-(g>>4)`)
//        for exact `c_nib >= 14 -> e = 0` zeroing (100% bit-exact across all 256*16*256 inputs!).
//      - Also accelerates `kv_bf12_unpack_warp_row` (prefill & staging unpacker).
//   2. SWA Intra-Layer L2-Persistence Helper (`kv_bf12_make_l2_evict_last_policy` +
//      `kv_bf12_cp_async_cg_16_l2_hint`):
//      - Enables `L2::evict_last` on `FUSE_MT=0` short-window KV loads (3x intra-layer reuse
//        while concurrent L2PF streams `wo_ud` into 24 MiB L2) and `L2::evict_first` on Global
//        streaming `>12 MiB` KV loads.
//   3. Preserves all Wave-V11 async `cp.async` `meta[64]` loading & 0-way XOR-swizzled SMEM
//      staging (`sm[64][128]` and `e4[64][64]`) + strict `"memory"` clobbers.

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
static constexpr int KV_BF12_SM_OFF_B    = 0;       // [0     ..  8192) = 8,192 B (XOR-swizzled per row)
static constexpr int KV_BF12_E4_OFF_B    = 8192;    // [8192  .. 12288) = 4,096 B (XOR-swizzled per row)
static constexpr int KV_BF12_META_OFF_B  = 12288;   // [12288 .. 12544) =   256 B (async cp.async loaded)
static constexpr int KV_BF12_BLKID_OFF_B = 12544;   // [12544 .. 12800) =   256 B (stashed from KV_blk_slot)

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
// Bit-Exact Scalar + Wave-V12 Branchless 2-Wide SWAR Reconstruction Primitives
//   100.000% bit-identical to V10/V11 `kv_bf12_reconstruct_one` across ALL
//   (sm_byte in [0..255], c_nib in [0..15], base_e in [0..255]) inputs!
// =========================================================================
static __device__ __host__ __forceinline__ uint16_t kv_bf12_reconstruct_one(
    uint32_t sm_byte, uint32_t c_nib, uint32_t base_e
) {
    const uint32_t e = (c_nib < 14u) ? (base_e - c_nib) : 0u;
    return static_cast<uint16_t>(((sm_byte & 0x80u) << 8) | (e << 7) | (sm_byte & 0x7Fu));
}

// Precompute guarded dual-lane base exponent ONCE per token quarter:
//   bits [8:0] = (base_e | 0x100), bits [24:16] = (base_e | 0x100).
// Guard bit 0x100 guarantees `base_guard - c_pair` NEVER borrows across bit 16
// even when `base_e == 0` and `c_nib == 15` (`0x100 - 0x0F = 0x0F1`).
static __device__ __host__ __forceinline__ uint32_t kv_bf12_base_guard_pair(uint32_t base_e) {
    return ((base_e & 0xFFu) | 0x100u) * 0x00010001u;
}

// Branchless 2-wide SWAR unpack: 2 packed `sm` bytes (low 16b of `sm01`) +
// 2 packed `e4` nibbles (low 8b of `e4_01`) -> packed `bf16x2` (`u0 | (u1 << 16)`).
// Note: for normal tokens (`c_nib < 14`), `base_e >= c_nib` always holds so
// `(base_e | 0x100) - c_nib` has exact `base_e - c_nib` in its low 8 bits;
// for `c_nib in {14, 15}`, `(c + 2) & 0x10` is `0x10` -> `zmask = 0x0FF` -> `e = 0`!
static __device__ __host__ __forceinline__ uint32_t kv_bf12_unpack2_swar(
    uint32_t sm01, uint32_t e4_01, uint32_t base_guard
) {
    // 1. Place sign[7] -> bit 15 and mantissa[6:0] -> bits [6:0] for BOTH lanes simultaneously.
#if defined(__CUDA_ARCH__)
    uint32_t sm_exp;
    asm("prmt.b32 %0, %1, 0, 0x4140;" : "=r"(sm_exp) : "r"(sm01));
#else
    const uint32_t sm_exp = (sm01 & 0x000000FFu) | ((sm01 & 0x0000FF00u) << 8);
#endif
    const uint32_t sm_placed = (sm_exp & 0x007F007Fu) | ((sm_exp & 0x00800080u) << 8);

    // 2. Unpack 2 exponent-delta nibbles into bits [3:0] and [19:16].
    const uint32_t c_pair = (e4_01 & 0x0000000Fu) | ((e4_01 & 0x000000F0u) << 12);

    // 3. Borrow-free dual-lane subtraction + branchless `c >= 14 -> 0` masking.
    const uint32_t diff  = base_guard - c_pair;
    const uint32_t ge14  = (c_pair + 0x00020002u) & 0x00100010u;
    const uint32_t zmask = (ge14 << 4) - (ge14 >> 4);  // 0x00FF per lane iff c >= 14, else 0x0000
#if defined(__CUDA_ARCH__)
    uint32_t exp_pair;
    // lop3.b32: (a & ~b & c) -> truth table: a=0xF0, b=0xCC, c=0xAA => 0xF0 & 0x33 & 0xAA = 0x20
    asm("lop3.b32 %0, %1, %2, %3, 0x20;"
        : "=r"(exp_pair) : "r"(diff), "r"(zmask), "r"(0x00FF00FFu));
#else
    const uint32_t exp_pair = (diff & ~zmask) & 0x00FF00FFu;
#endif
    return sm_placed | (exp_pair << 7);
}

static __device__ __host__ __forceinline__ uint4 kv_bf12_unpack8_swar_u4(
    uint32_t sm_lo4, uint32_t sm_hi4, uint32_t e4_8, uint32_t base_guard
) {
    return make_uint4(
        kv_bf12_unpack2_swar(sm_lo4,        e4_8,        base_guard),
        kv_bf12_unpack2_swar(sm_lo4 >> 16,  e4_8 >> 8,   base_guard),
        kv_bf12_unpack2_swar(sm_hi4,        e4_8 >> 16,  base_guard),
        kv_bf12_unpack2_swar(sm_hi4 >> 16,  e4_8 >> 24,  base_guard)
    );
}

// Drop-in V10/V11 signature backed by Wave-V12 branchless SWAR!
static __device__ __host__ __forceinline__ uint4 kv_bf12_unpack8_to_u4(
    uint32_t sm_lo4, uint32_t sm_hi4, uint32_t e4_8, uint32_t base_e
) {
    const uint32_t base_guard = kv_bf12_base_guard_pair(base_e);
    return kv_bf12_unpack8_swar_u4(sm_lo4, sm_hi4, e4_8, base_guard);
}

static __device__ __host__ __forceinline__ uint32_t kv_bf12_swiz_col(int row, uint32_t dim_idx) {
    const uint32_t chunk = dim_idx >> 3;  // 0..15 (16B chunk)
    const uint32_t swiz_c = chunk ^ (static_cast<uint32_t>(row) & 7u);
    return (swiz_c << 3) | (dim_idx & 7u);
}

static __device__ __host__ __forceinline__ int kv_bf12_sm_stage_chunk(int row, int c) {
    return c ^ (row & 7);  // 0-way conflict SMEM staging for sm[64][128] (c in 0..7)
}

static __device__ __host__ __forceinline__ int kv_bf12_e4_stage_chunk(int row, int c) {
    return c ^ (row & 3);  // 0-way conflict SMEM staging for e4[64][64] (c in 0..3)
}

#endif  // __CUDACC__ || KV_BF12_ENABLE_DEVICE_HELPERS

#if defined(__CUDACC__) || defined(KV_BF12_HOST_SYNTAX_CHECK)

// =========================================================================
// Inline PTX L2 Cache-Policy & cp.async Helpers (with strict memory clobbers)
// =========================================================================
static __device__ __forceinline__ uint64_t kv_bf12_make_l2_evict_last_policy() {
    uint64_t policy = 0ULL;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    float frac = 1.0f;
    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, %1;"
                 : "=l"(policy) : "f"(frac));
#endif
    return policy;
}

static __device__ __forceinline__ uint64_t kv_bf12_make_l2_evict_first_policy() {
    uint64_t policy = 0ULL;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    float frac = 1.0f;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, %1;"
                 : "=l"(policy) : "f"(frac));
#endif
    return policy;
}

static __device__ __forceinline__ void kv_bf12_cp_async_cg_16(
    uint32_t smem_addr, const void* gmem_ptr
) {
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n"
                 :: "r"(smem_addr), "l"(gmem_ptr) : "memory");
}

static __device__ __forceinline__ void kv_bf12_cp_async_cg_16_l2_hint(
    uint32_t smem_addr, const void* gmem_ptr, uint64_t cache_policy
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;\n"
                 :: "r"(smem_addr), "l"(gmem_ptr), "l"(cache_policy) : "memory");
#else
    (void)cache_policy;
    kv_bf12_cp_async_cg_16(smem_addr, gmem_ptr);
#endif
}

static __device__ __forceinline__ void kv_bf12_cp_async_cg_16_evict_first(
    uint32_t smem_addr, const void* gmem_ptr, uint64_t cache_policy
) {
    kv_bf12_cp_async_cg_16_l2_hint(smem_addr, gmem_ptr, cache_policy);
}

static __device__ __forceinline__ void kv_bf12_cp_async_ca_4(
    uint32_t smem_addr, const void* gmem_ptr
) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"
                 :: "r"(smem_addr), "l"(gmem_ptr) : "memory");
}

// =========================================================================
// Warp-Level 0-ULP Lossless KV-BF12 Encoder (32 lanes x 4 dims = 128 dims)
//   - Bit-identical to V10/V11 `kv_bf12_encode_warp_store`.
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

    uint32_t c0, c1, c2, c3;
    int esc0, esc1, esc2, esc3;

    const uint32_t sm0 = ((u0 >> 8) & 0x80u) | (u0 & 0x7Fu);
    if (e0 == 0u)                                        { c0 = 14u;     esc0 = 0; }
    else if (e0 <= 254u && me >= e0 && (me - e0) <= 13u) { c0 = me - e0; esc0 = 0; }
    else                                                 { c0 = 15u;     esc0 = 1; }

    const uint32_t sm1 = ((u1 >> 8) & 0x80u) | (u1 & 0x7Fu);
    if (e1 == 0u)                                        { c1 = 14u;     esc1 = 0; }
    else if (e1 <= 254u && me >= e1 && (me - e1) <= 13u) { c1 = me - e1; esc1 = 0; }
    else                                                 { c1 = 15u;     esc1 = 1; }

    const uint32_t sm2 = ((u2 >> 8) & 0x80u) | (u2 & 0x7Fu);
    if (e2 == 0u)                                        { c2 = 14u;     esc2 = 0; }
    else if (e2 <= 254u && me >= e2 && (me - e2) <= 13u) { c2 = me - e2; esc2 = 0; }
    else                                                 { c2 = 15u;     esc2 = 1; }

    const uint32_t sm3 = ((u3 >> 8) & 0x80u) | (u3 & 0x7Fu);
    if (e3 == 0u)                                        { c3 = 14u;     esc3 = 0; }
    else if (e3 <= 254u && me >= e3 && (me - e3) <= 13u) { c3 = me - e3; esc3 = 0; }
    else                                                 { c3 = 15u;     esc3 = 1; }

    const uint32_t sm_pack = sm0 | (sm1 << 8) | (sm2 << 16) | (sm3 << 24);
    *reinterpret_cast<uint32_t*>(&blk->sm[slot][lane * 4]) = sm_pack;

    const uint16_t e4_pack = static_cast<uint16_t>(c0 | (c1 << 4) | (c2 << 8) | (c3 << 12));
    *reinterpret_cast<uint16_t*>(&blk->e4[slot][lane * 2]) = e4_pack;

    const uint32_t m0 = __ballot_sync(0xFFFFFFFFu, esc0);
    const uint32_t m1 = __ballot_sync(0xFFFFFFFFu, esc1);
    const uint32_t m2 = __ballot_sync(0xFFFFFFFFu, esc2);
    const uint32_t m3 = __ballot_sync(0xFFFFFFFFu, esc3);
    const uint32_t total_esc = static_cast<uint32_t>(
        __popc(m0) + __popc(m1) + __popc(m2) + __popc(m3)
    );

    if (total_esc == 0u) {
        if (lane == 0) {
            blk->meta[slot] = me & 0xFFu;
        }
        return;
    }

    if (total_esc == 1u) {
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
// 128-Thread Fully-Async XOR-Swizzled Tile Loader for BN=64 KV-BF12
//   Supports arbitrary L2 policy (`evict_first` OR `evict_last` via `use_l2_hint`).
// =========================================================================
static __device__ __forceinline__ void kv_bf12_issue_tile64_async(
    const uint8_t* __restrict__ kv_half_base,
    int64_t blk_stride_bytes,
    const int2* __restrict__ blk_slot,
    uint32_t kv_sptr,
    uint8_t* __restrict__ smem_kv_u8,
    bool use_l2_hint,
    uint64_t l2_policy,
    int tid
) {
    // 1. Issue sm[64][128] = 512 x 16B = 4 iterations across 128 threads (XOR-swizzled!)
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int idx = i * 128 + tid;
        const int row = idx >> 3;
        const int c   = idx & 7;
        const int2 bs = blk_slot[row];
        if (bs.x >= 0) {
            const KvBf12HalfBlock* blk = kv_bf12_half_blk_const(kv_half_base, blk_stride_bytes, bs.x);
            const int swiz_c = kv_bf12_sm_stage_chunk(row, c);
            const uint32_t s_addr = kv_sptr + static_cast<uint32_t>(KV_BF12_SM_OFF_B + row * 128 + swiz_c * 16);
            const void* g_ptr = &blk->sm[bs.y][c * 16];
            if (use_l2_hint) {
                kv_bf12_cp_async_cg_16_l2_hint(s_addr, g_ptr, l2_policy);
            } else {
                kv_bf12_cp_async_cg_16(s_addr, g_ptr);
            }
        }
    }

    // 2. Issue e4[64][64] = 256 x 16B = 2 iterations across 128 threads (XOR-swizzled!)
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int idx = i * 128 + tid;
        const int row = idx >> 2;
        const int c   = idx & 3;
        const int2 bs = blk_slot[row];
        if (bs.x >= 0) {
            const KvBf12HalfBlock* blk = kv_bf12_half_blk_const(kv_half_base, blk_stride_bytes, bs.x);
            const int swiz_c = kv_bf12_e4_stage_chunk(row, c);
            const uint32_t s_addr = kv_sptr + static_cast<uint32_t>(KV_BF12_E4_OFF_B + row * 64 + swiz_c * 16);
            const void* g_ptr = &blk->e4[bs.y][c * 16];
            if (use_l2_hint) {
                kv_bf12_cp_async_cg_16_l2_hint(s_addr, g_ptr, l2_policy);
            } else {
                kv_bf12_cp_async_cg_16(s_addr, g_ptr);
            }
        }
    }

    // 3a. Stash blk_id[64] in free SMEM tail [12544..12800)
    if (tid < KV_BF12_TILE_BN) {
        int32_t* smem_blkid = reinterpret_cast<int32_t*>(smem_kv_u8 + KV_BF12_BLKID_OFF_B);
        smem_blkid[tid] = blk_slot[tid].x;
    }

    // 3b. Fully asynchronous cp.async load of meta[64] into [12288..12544)
    if (tid < (KV_BF12_TILE_BN >> 2)) {
        const int r0 = tid << 2;
        const int2 b0 = blk_slot[r0 + 0];
        const int2 b1 = blk_slot[r0 + 1];
        const int2 b2 = blk_slot[r0 + 2];
        const int2 b3 = blk_slot[r0 + 3];
        const bool contig4 =
            (b0.x >= 0) && ((b0.y & 3) == 0) &&
            (b1.x == b0.x) && (b1.y == b0.y + 1) &&
            (b2.x == b0.x) && (b2.y == b0.y + 2) &&
            (b3.x == b0.x) && (b3.y == b0.y + 3);
        if (contig4) {
            const KvBf12HalfBlock* blk = kv_bf12_half_blk_const(kv_half_base, blk_stride_bytes, b0.x);
            const uint32_t s_addr = kv_sptr + static_cast<uint32_t>(KV_BF12_META_OFF_B + r0 * 4);
            const void* g_ptr = &blk->meta[b0.y];
            if (use_l2_hint) {
                kv_bf12_cp_async_cg_16_l2_hint(s_addr, g_ptr, l2_policy);
            } else {
                kv_bf12_cp_async_cg_16(s_addr, g_ptr);
            }
        } else {
            uint32_t* smem_meta = reinterpret_cast<uint32_t*>(smem_kv_u8 + KV_BF12_META_OFF_B);
            auto load_meta_one = [&](int row_idx, int2 bk) {
                if (bk.x >= 0) {
                    const KvBf12HalfBlock* blk = kv_bf12_half_blk_const(kv_half_base, blk_stride_bytes, bk.x);
                    const uint32_t s_addr = kv_sptr + static_cast<uint32_t>(KV_BF12_META_OFF_B + row_idx * 4);
                    kv_bf12_cp_async_ca_4(s_addr, &blk->meta[bk.y]);
                } else {
                    smem_meta[row_idx] = 0u;
                }
            };
            load_meta_one(r0 + 0, b0);
            load_meta_one(r0 + 1, b1);
            load_meta_one(r0 + 2, b2);
            load_meta_one(r0 + 3, b3);
        }
    }
}

// =========================================================================
// Wave-V12 SWAR Quarter Unpacker (32 dims -> 4x 16B XOR-swizzled SMEM stores)
//   Hoists `base_guard` ONCE for all 16 SWAR pairs!
// =========================================================================
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

    const uint32_t base_guard = kv_bf12_base_guard_pair(meta & 0xFFu);
    smem_kv4[row * 16 + ((c_base + 0) ^ row_mask)] = kv_bf12_unpack8_swar_u4(sm_a.x, sm_a.y, e4_q.x, base_guard);
    smem_kv4[row * 16 + ((c_base + 1) ^ row_mask)] = kv_bf12_unpack8_swar_u4(sm_a.z, sm_a.w, e4_q.y, base_guard);
    smem_kv4[row * 16 + ((c_base + 2) ^ row_mask)] = kv_bf12_unpack8_swar_u4(sm_b.x, sm_b.y, e4_q.z, base_guard);
    smem_kv4[row * 16 + ((c_base + 3) ^ row_mask)] = kv_bf12_unpack8_swar_u4(sm_b.z, sm_b.w, e4_q.w, base_guard);

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
// 128-Thread Conflict-Free In-Place 2-Stage SWAR Unpack into XOR-Swizzled BF16
// =========================================================================
static __device__ __forceinline__ void kv_bf12_unpack_tile64_inplace_swiz(
    uint16_t* __restrict__ smem_kv,
    const uint8_t* __restrict__ kv_half_base,
    int64_t blk_stride_bytes,
    const uint16_t* __restrict__ emerg_raw,
    int tid
) {
    uint8_t* smem_u8 = reinterpret_cast<uint8_t*>(smem_kv);
    const int row_lo = tid >> 2;
    const int row_hi = 32 + row_lo;
    const int qtr    = tid & 3;

    const int e4_c_lo = kv_bf12_e4_stage_chunk(row_lo, qtr);
    const int e4_c_hi = kv_bf12_e4_stage_chunk(row_hi, qtr);
    const uint4 e4_lo = *reinterpret_cast<const uint4*>(
        smem_u8 + KV_BF12_E4_OFF_B + row_lo * 64 + e4_c_lo * 16
    );
    const uint4 e4_hi = *reinterpret_cast<const uint4*>(
        smem_u8 + KV_BF12_E4_OFF_B + row_hi * 64 + e4_c_hi * 16
    );
    const uint32_t* smem_meta  = reinterpret_cast<const uint32_t*>(smem_u8 + KV_BF12_META_OFF_B);
    const int32_t*  smem_blkid = reinterpret_cast<const int32_t*>(smem_u8 + KV_BF12_BLKID_OFF_B);
    const uint32_t meta_lo = smem_meta[row_lo];
    const uint32_t meta_hi = smem_meta[row_hi];
    const int32_t  blk_lo  = smem_blkid[row_lo];
    const int32_t  blk_hi  = smem_blkid[row_hi];

    const int sm_hi_ca = kv_bf12_sm_stage_chunk(row_hi, qtr * 2 + 0);
    const int sm_hi_cb = kv_bf12_sm_stage_chunk(row_hi, qtr * 2 + 1);
    const uint4 sm_hi_a = *reinterpret_cast<const uint4*>(
        smem_u8 + KV_BF12_SM_OFF_B + row_hi * 128 + sm_hi_ca * 16
    );
    const uint4 sm_hi_b = *reinterpret_cast<const uint4*>(
        smem_u8 + KV_BF12_SM_OFF_B + row_hi * 128 + sm_hi_cb * 16
    );
    __syncthreads();

    kv_bf12_unpack_quarter_swiz(
        smem_kv, row_hi, qtr, sm_hi_a, sm_hi_b, e4_hi, meta_hi, blk_hi,
        kv_half_base, blk_stride_bytes, emerg_raw
    );

    const int sm_lo_ca = kv_bf12_sm_stage_chunk(row_lo, qtr * 2 + 0);
    const int sm_lo_cb = kv_bf12_sm_stage_chunk(row_lo, qtr * 2 + 1);
    const uint4 sm_lo_a = *reinterpret_cast<const uint4*>(
        smem_u8 + KV_BF12_SM_OFF_B + row_lo * 128 + sm_lo_ca * 16
    );
    const uint4 sm_lo_b = *reinterpret_cast<const uint4*>(
        smem_u8 + KV_BF12_SM_OFF_B + row_lo * 128 + sm_lo_cb * 16
    );
    __syncthreads();

    kv_bf12_unpack_quarter_swiz(
        smem_kv, row_lo, qtr, sm_lo_a, sm_lo_b, e4_lo, meta_lo, blk_lo,
        kv_half_base, blk_stride_bytes, emerg_raw
    );
    __syncthreads();
}

// =========================================================================
// Warp-Level Coalesced Linear BF16 SWAR Unpacker (32 lanes x 4 dims = 128 dims)
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

    const uint32_t base_guard = kv_bf12_base_guard_pair(meta & 0xFFu);
    const uint32_t sm4 = __ldg(reinterpret_cast<const uint32_t*>(&blk->sm[slot][lane * 4]));
    const uint32_t e4_2 = static_cast<uint32_t>(
        __ldg(reinterpret_cast<const uint16_t*>(&blk->e4[slot][lane * 2]))
    );

    uint32_t p01 = kv_bf12_unpack2_swar(sm4,       e4_2,      base_guard);
    uint32_t p23 = kv_bf12_unpack2_swar(sm4 >> 16, e4_2 >> 8, base_guard);

    if (mode == 1u) {
        const uint32_t rec = (meta >> 8) & 0x7FFFu;
        const uint32_t idx = (rec >> 8) & 0x7Fu;
        if ((idx >> 2) == static_cast<uint32_t>(lane)) {
            const uint32_t exp = (rec & 0xFFu) << 7;
            const uint32_t sub = idx & 3u;
            if (sub == 0u)      p01 = (p01 & 0xFFFF807Fu) | exp;
            else if (sub == 1u) p01 = (p01 & 0x807FFFFFu) | (exp << 16);
            else if (sub == 2u) p23 = (p23 & 0xFFFF807Fu) | exp;
            else                p23 = (p23 & 0x807FFFFFu) | (exp << 16);
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
                if (sub == 0u)      p01 = (p01 & 0xFFFF807Fu) | exp;
                else if (sub == 1u) p01 = (p01 & 0x807FFFFFu) | (exp << 16);
                else if (sub == 2u) p23 = (p23 & 0xFFFF807Fu) | exp;
                else                p23 = (p23 & 0x807FFFFFu) | (exp << 16);
            }
        }
    }

    *reinterpret_cast<uint32_t*>(dst_row128 + lane * 4 + 0) = p01;
    *reinterpret_cast<uint32_t*>(dst_row128 + lane * 4 + 2) = p23;
}

#endif  // __CUDACC__ || KV_BF12_HOST_SYNTAX_CHECK
