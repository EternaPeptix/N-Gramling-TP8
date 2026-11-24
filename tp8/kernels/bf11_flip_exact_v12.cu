// SPDX-License-Identifier: Apache-2.0
// SM121a (GB10) Roofline-Hardened 0-ULP Lossless BF11_FLIP_EXACT v12 (C=1..8 M=1..48 + C=32 M=49..256+)
//
// Drop-in ABI superset of `bf11_flip_exact_c32.cu` (`libbf11_flip_exact_c32.so`), plus v5 roofline
// fast entrypoints (`bf11_flip_v5_*`) and zero-raw-weight GPU sidecar builders (`FREE_RAW=1` safe!):
//
// Key SM121a (48 SMs, 99 KiB SMEM/SM, 24 MiB L2, 244 GB/s UMA) v5 Roofline Breakthroughs:
//   1. Adaptive Zero-Sync `SLAB_META_V5` Flip Sidecar (`MODE_ZERO` / `MODE_SPARSE_SHFL<=4` /
//      `MODE_SPARSE<=thresh` / `MODE_BITMASK` 128B coalesced) + `META_HAS_OUTLIER_BIT` (~90% slab
//      outlier-check skip!): eliminates `WARP_META -> FLIPS` 2-hop DRAM pointer chase, loop-carried
//      `f_cur` dependency, L2-polluting `__ldg` `.ca` on `m_blocks==1`, and `s_flip` `atomicOr` +
//      `2x __syncwarp()` on ZERO / SHFL / BITMASK slabs! Built on GPU directly from legacy BF11
//      tensors -> 100% compatible with `INKLING_BF11_FREE_RAW=1` and `unpack_bf16()`.
//   2. XOR-Swizzled ZERO-Padding SMEM + `ldmatrix.sync.aligned.m8n8.x4.shared.b16`:
//      - 4x fewer SMEM load instructions vs `4x LDS.32`, ZERO address ALU in `mt` loop, 0-way bank
//        conflicts for BOTH `BK=128` (`SX_STRIDE=128`) and `BK=64` (`SX_STRIDE=64`)!
//      - Drops `BM=64 BK=128` & `BM=128 BK=64` 2-stage SMEM from `34.8-36.9 KiB` -> `32.5 KiB`
//        (`3 * 32.5 = 97.5 KiB < 99 KiB`), unlocking **3 CTAs/SM** (exact 1.0-wave on `wo_ud`!).
//   3. Single-Barrier `BK=128` (`36` barriers instead of `72` across `K=4608`) + Flattened
//      Cross-Split `(ps, kc) -> (ps+1, 0)` Software-Pipelined Prefetch:
//      - `nxt_slab` + `nxt_flip` + `load_x` ALL issued at the TOP of every slab BEFORE Half-0,
//        overlapped behind ALL `8 * MT` Tensor Core MMAs — including across `ps -> ps+1` via
//        warp-register-cached `SLAB_META_V5` (`my_meta_cur` / `my_meta_nxt`)!
//   4. Zero-Repack Virtual `W_RUN` Decoupling (`1, 2, 4` warps/CTA on `SUB_N==1` weights) +
//      L2-Resident Split-K (`SPLIT_CTA = split_packed` in `{6, 9}`) with Fused In-Kernel Semaphore
//      Reduction for `k_order == 1` (LITERAL 0-ULP `nd == 0` bit-identical to `split_cta=1 k_order=1`
//      and `skinny.py`!):
//      - Fixes `(128,4608)` & Dual-KV `2x(128,4608)` from 4-8 CTAs (5.1% roofline, ~65 us) ->
//        `144` CTAs (`3.0 CTAs/SM` exact zero wave tail, ~6-12 us, 2-3x faster than `torch.mm`!).
//      - Eliminates `qkvr` (`60` CTAs = 62.5% wave eff) & `q_side` (`52` CTAs = 54.2% wave eff)
//        catastrophic wave tails at `M=20..64` -> `360`/`312` CTAs (`93-99%` wave eff, <=2.95 MB L2 P)!
//      - Pushes `M=72..128` steady-state to `80-90%` of GB10 244 GB/s UMA roofline (`1x` slab read
//        or exact `5.0 CTAs/SM` 1-wave residency, beating BF16 `torch.mm` by -31% DRAM bytes!).
//   5. Fast SWAR `unpack4_bf11_flip_v5` (pre-shifted `base_sh7`, zero `& 0xFFF` masks, zero live
//      `c0..c7` regs, constexpr unrolled step selection -> ZERO local stack arrays, 0 B spills!).
//
// Build on GB10 (SM121a, STRICT `-gencode arch=compute_121a,code=sm_121a --fmad=false`):
//   ./build_bf11_v5.sh [OUT_DIR]

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace {

static constexpr int SLAB11_BYTES = 1408;

// Adaptive per-slab v5 metadata word (`uint32_t SLAB_META_V5[wtot * spw]`):
//   bit 31     : 1 = MODE_BITMASK (128B coalesced u32 in BMASKS), 0 = MODE_ZERO / MODE_SPARSE
//   bits 30..25: nf in 0..63 (0 = MODE_ZERO, 1..4 = MODE_SPARSE_SHFL, 5..63 = MODE_SPARSE)
//   bit 24     : 1 iff this slab has >= 1 outlier record in OUTLIERS
//   bits 23..0 : absolute `d_idx` (MODE_BITMASK) or `s_off` (MODE_SPARSE), max 16,777,215
static constexpr uint32_t BF11_V5_META_BITMASK_BIT = 0x80000000u;
static constexpr uint32_t BF11_V5_META_NF_SHIFT    = 25u;
static constexpr uint32_t BF11_V5_META_NF_MASK     = 0x3Fu;
static constexpr uint32_t BF11_V5_META_OUTLIER_BIT = 0x01000000u;
static constexpr uint32_t BF11_V5_META_OFF_MASK    = 0x00FFFFFFu;

// =========================================================================
// PTX Memory, Async Copy, ldmatrix, and Semaphore Primitives
// =========================================================================

static __device__ __forceinline__ uint4 ld_cs_u4(const uint4* p) {
    uint4 v;
    asm volatile("ld.global.cs.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

static __device__ __forceinline__ uint4 ld_ca_u4(const uint4* p) {
    uint4 v;
    asm volatile("ld.global.ca.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

static __device__ __forceinline__ uint32_t ld_cs_u32(const uint32_t* p) {
    uint32_t v;
    asm volatile("ld.global.cs.u32 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
}

static __device__ __forceinline__ uint32_t ld_ca_u32(const uint32_t* p) {
    uint32_t v;
    asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
}

static __device__ __forceinline__ uint32_t ld_cs_u16(const uint16_t* p) {
    uint16_t v;
    asm volatile("ld.global.cs.u16 %0, [%1];" : "=h"(v) : "l"(p));
    return static_cast<uint32_t>(v);
}

static __device__ __forceinline__ uint32_t ld_ca_u16(const uint16_t* p) {
    uint16_t v;
    asm volatile("ld.global.ca.u16 %0, [%1];" : "=h"(v) : "l"(p));
    return static_cast<uint32_t>(v);
}

static __device__ __forceinline__ uint32_t ld_nc_u16(const uint16_t* p) {
    return static_cast<uint32_t>(__ldg(p));
}

template <bool CACHE_L2>
static __device__ __forceinline__ uint32_t ld_u32_t(const uint32_t* p) {
    if constexpr (CACHE_L2) return ld_ca_u32(p);
    else                    return ld_cs_u32(p);
}

template <bool CACHE_L2>
static __device__ __forceinline__ uint32_t ld_u16_t(const uint16_t* p) {
    if constexpr (CACHE_L2) return ld_ca_u16(p);
    else                    return ld_cs_u16(p);
}

static __device__ __forceinline__ float2 ld_cg_f2(const float2* p) {
    float2 v;
    asm volatile("ld.global.cg.v2.f32 {%0,%1}, [%2];"
                 : "=f"(v.x), "=f"(v.y)
                 : "l"(p));
    return v;
}

static __device__ __forceinline__ void prefetch_l2(const void* p) {
    asm volatile("prefetch.global.L2 [%0];" :: "l"(p));
}

static __device__ __forceinline__ int atomic_add_acq_rel_gpu(int32_t* ptr, int32_t val) {
    int32_t old;
    asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;"
                 : "=r"(old) : "l"(ptr), "r"(val) : "memory");
    return old;
}

static __device__ __forceinline__ void st_release_gpu_s32(int32_t* ptr, int32_t val) {
    asm volatile("st.release.gpu.global.s32 [%0], %1;"
                 :: "l"(ptr), "r"(val) : "memory");
}

static __device__ __forceinline__ void cp_async_ca_16(void* smem_ptr, const void* gmem_ptr) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                 :: "r"(smem_addr), "l"(gmem_ptr) : "memory");
}

static __device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template <int N>
static __device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}

static __device__ __forceinline__ void ldmatrix_x4_smem(
    uint32_t smem_addr, uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3
) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
        : "r"(smem_addr)
    );
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

static __device__ __forceinline__ void mma_m16n8k8_bf16(
    float (&acc)[4],
    uint32_t a0, uint32_t a1,
    uint32_t b0
) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a0), "r"(a1), "r"(b0)
    );
}

// =========================================================================
// Slab11 Struct, Constexpr Word/c12 Extractors (0 Stack!), & SWAR Unpack
// =========================================================================

struct Slab11 {
    uint4 sm0, sm1;
    uint32_t e0, e1, e2;
};

template <bool CACHE_L2 = false>
static __device__ __forceinline__ Slab11 load_slab11_t(const uint8_t* sp, int lane) {
    const uint4* sm_p = reinterpret_cast<const uint4*>(sp);
    const uint32_t* e3_p = reinterpret_cast<const uint32_t*>(sp + 1024);
    Slab11 r;
    if constexpr (CACHE_L2) {
        r.sm0 = ld_ca_u4(sm_p + lane);
        r.sm1 = ld_ca_u4(sm_p + 32 + lane);
        r.e0  = ld_ca_u32(e3_p + lane);
        r.e1  = ld_ca_u32(e3_p + 32 + lane);
        r.e2  = ld_ca_u32(e3_p + 64 + lane);
    } else {
        r.sm0 = ld_cs_u4(sm_p + lane);
        r.sm1 = ld_cs_u4(sm_p + 32 + lane);
        r.e0  = ld_cs_u32(e3_p + lane);
        r.e1  = ld_cs_u32(e3_p + 32 + lane);
        r.e2  = ld_cs_u32(e3_p + 64 + lane);
    }
    return r;
}

static __device__ __forceinline__ Slab11 load_slab11(const uint8_t* sp, int lane) {
    return load_slab11_t<false>(sp, lane);
}

static __device__ __forceinline__ uint32_t meta_nf(uint4 meta, int sl) {
    uint32_t w = (sl < 4) ? meta.y : ((sl < 8) ? meta.z : meta.w);
    return (w >> ((sl & 3) * 8)) & 0xFFu;
}

// Constexpr word & c12 selectors: guarantee ZERO runtime-indexed local stack arrays!
template <int S>
static __device__ __forceinline__ uint32_t slab_sm_word(const Slab11& r) {
    if constexpr (S == 0) return r.sm0.x;
    else if constexpr (S == 1) return r.sm0.y;
    else if constexpr (S == 2) return r.sm0.z;
    else if constexpr (S == 3) return r.sm0.w;
    else if constexpr (S == 4) return r.sm1.x;
    else if constexpr (S == 5) return r.sm1.y;
    else if constexpr (S == 6) return r.sm1.z;
    else return r.sm1.w;
}

// Raw c12 selector: upper bits >= 12 are masked out inside unpack4_bf11_flip_v5, saving 8 ANDs!
template <int S>
static __device__ __forceinline__ uint32_t slab_c12_raw(const Slab11& r) {
    if constexpr (S == 0) return r.e0;
    else if constexpr (S == 1) return r.e0 >> 12;
    else if constexpr (S == 2) return __funnelshift_r(r.e0, r.e1, 24);
    else if constexpr (S == 3) return r.e1 >> 4;
    else if constexpr (S == 4) return r.e1 >> 16;
    else if constexpr (S == 5) return __funnelshift_r(r.e1, r.e2, 28);
    else if constexpr (S == 6) return r.e2 >> 8;
    else return r.e2 >> 20;
}

// Fast 0-ULP SWAR 4-weight unpack combining 3-bit e3 + 1-bit b3 flip mask:
//   - Pre-shifted `base_sh7 = (base_exp << 7) | (base_exp << 23)` eliminates 2 SHLs/step
//   - Uses SWAR IMUL bit scatter for f4 and masks bits [11:0] directly -> 0 redundant `& 0xFFF`
//   - Valid for ALL base_exp in [15..254], ALL d in [0..15] (zero halfword borrow, bit 15/31 clean)
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

static __device__ __forceinline__ void patch_outliers_step(
    uint32_t cur_step, uint32_t lane,
    const uint64_t* __restrict__ OUTLIERS,
    int32_t& o_cur, int32_t o_end, uint64_t& o_rec, uint32_t& o_next_step,
    uint32_t& b0, uint32_t& b1
) {
    if (__builtin_expect(o_next_step == cur_step, 0)) {
        while (o_cur < o_end && o_next_step == cur_step) {
            uint32_t lj = static_cast<uint32_t>(o_rec >> 16) & 0x7Fu;
            if ((lj >> 2) == lane) {
                uint32_t val = static_cast<uint32_t>(o_rec & 0xFFFFu);
                uint32_t j = lj & 3u;
                if (j == 0u)      b0 = (b0 & 0xFFFF0000u) | val;
                else if (j == 1u) b0 = (b0 & 0x0000FFFFu) | (val << 16);
                else if (j == 2u) b1 = (b1 & 0xFFFF0000u) | val;
                else              b1 = (b1 & 0x0000FFFFu) | (val << 16);
            }
            ++o_cur;
            o_rec = (o_cur < o_end) ? __ldg(OUTLIERS + o_cur) : 0xFFFFFFFFFFFFFFFFULL;
            o_next_step = static_cast<uint32_t>(o_rec >> 32);
        }
    }
}

// =========================================================================
// Adaptive Zero-Sync v5 Flip Decoders (MODE_ZERO / SHFL<=4 / SPARSE / BITMASK)
// =========================================================================

template <bool CACHE_L2>
static __device__ __forceinline__ uint32_t issue_flip_load_v5(
    uint32_t smeta,
    const uint16_t* __restrict__ SFLIPS,
    const uint32_t* __restrict__ BMASKS,
    int lane
) {
    if (smeta & BF11_V5_META_BITMASK_BIT) {
        const int64_t d_idx = static_cast<int64_t>(smeta & BF11_V5_META_OFF_MASK);
        return ld_u32_t<CACHE_L2>(BMASKS + (d_idx << 5) + lane);
    }
    const uint32_t nf = (smeta >> BF11_V5_META_NF_SHIFT) & BF11_V5_META_NF_MASK;
    if (nf == 0u) return 0u;
    const int64_t s_off = static_cast<int64_t>(smeta & BF11_V5_META_OFF_MASK);
    return (static_cast<uint32_t>(lane) < nf) ? ld_u16_t<CACHE_L2>(SFLIPS + s_off + lane) : 0u;
}

template <bool CACHE_L2>
static __device__ __forceinline__ uint32_t finish_flip_v5(
    uint32_t raw_fl, uint32_t smeta,
    const uint16_t* __restrict__ SFLIPS,
    int lane, uint32_t* sFlipWarp
) {
    if (smeta & BF11_V5_META_BITMASK_BIT) {
        return raw_fl;  // ZERO SMEM, ZERO atomicOr, ZERO __syncwarp()!
    }
    const uint32_t nf = (smeta >> BF11_V5_META_NF_SHIFT) & BF11_V5_META_NF_MASK;
    if (nf == 0u) return 0u;  // MODE_ZERO: ZERO SMEM, ZERO sync!

    if (nf <= 4u) {
        // MODE_SPARSE_SHFL: ZERO SMEM, ZERO atomicOr, ZERO __syncwarp()!
        uint32_t fl = 0u;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            if (static_cast<uint32_t>(i) < nf) {
                uint32_t rec = __shfl_sync(0xFFFFFFFFu, raw_fl, i);
                if ((rec >> 5) == static_cast<uint32_t>(lane)) {
                    fl |= (1u << (rec & 31u));
                }
            }
        }
        return fl;
    }

    // MODE_SPARSE (5..63): single-round SMEM atomicOr (unrolled 2nd pass only if nf > 32)
    sFlipWarp[lane] = 0u;
    __syncwarp();
    if (static_cast<uint32_t>(lane) < nf) {
        atomicOr(&sFlipWarp[raw_fl >> 5], 1u << (raw_fl & 31u));
    }
    if (__builtin_expect(nf > 32u, 0)) {
        const uint32_t idx2 = static_cast<uint32_t>(lane) + 32u;
        if (idx2 < nf) {
            const int64_t s_off = static_cast<int64_t>(smeta & BF11_V5_META_OFF_MASK);
            const uint32_t r2 = ld_u16_t<CACHE_L2>(SFLIPS + s_off + idx2);
            atomicOr(&sFlipWarp[r2 >> 5], 1u << (r2 & 31u));
        }
    }
    __syncwarp();
    return sFlipWarp[lane];
}

// =========================================================================
// Zero-Raw-Weight GPU Sidecar Builder Kernels (`FREE_RAW=1` Safe!)
// =========================================================================

__global__ __launch_bounds__(128, 4)
void bf11_v5_measure_sidecar_kernel(
    const uint4* __restrict__ WARP_META,
    int32_t* __restrict__ W_S_COUNTS,
    int32_t* __restrict__ W_D_COUNTS,
    int wtot, int spw, int flip_thresh
) {
    const int widx = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (widx >= wtot) return;
    const uint4 meta = __ldg(WARP_META + widx);
    int32_t sc = 0;
    int32_t dc = 0;
    for (int sl = 0; sl < spw; ++sl) {
        const int nf = static_cast<int>(meta_nf(meta, sl));
        if (nf == 0) {
            // MODE_ZERO: 0 sparse flips, 0 bitmasks
        } else if (nf <= flip_thresh) {
            sc += nf;
        } else {
            dc += 1;
        }
    }
    W_S_COUNTS[widx] = sc;
    W_D_COUNTS[widx] = dc;
}

__global__ __launch_bounds__(32, 8)
void bf11_v5_build_sidecar_kernel(
    const uint16_t* __restrict__ FLIPS,
    const uint4* __restrict__ WARP_META,
    const uint64_t* __restrict__ OUTLIERS,
    const int32_t* __restrict__ O_OFF,
    const int32_t* __restrict__ W_S_OFF,
    const int32_t* __restrict__ W_D_OFF,
    uint32_t* __restrict__ SLAB_META_V5,
    uint16_t* __restrict__ SFLIPS,
    uint32_t* __restrict__ BMASKS,
    int wtot, int spw, int flip_thresh
) {
    const int widx = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    if (widx >= wtot) return;

    __shared__ uint32_t s_bm[32];

    // Scan outliers for this warp_uid to build per-slab `has_out_mask` (spw <= 12 <= 32)
    const int32_t o_beg = __ldg(O_OFF + widx);
    const int32_t o_end = __ldg(O_OFF + widx + 1);
    uint32_t local_omask = 0u;
    for (int32_t idx = o_beg + lane; idx < o_end; idx += 32) {
        const uint64_t rec = __ldg(OUTLIERS + idx);
        const uint32_t sl_o = static_cast<uint32_t>(rec >> 35);  // (step_id >> 3)
        if (sl_o < 32u) {
            local_omask |= (1u << sl_o);
        }
    }
    // Warp-wide OR reduction
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        local_omask |= __shfl_xor_sync(0xFFFFFFFFu, local_omask, off);
    }
    const uint32_t has_out_mask = local_omask;

    const uint4 meta = __ldg(WARP_META + widx);
    int64_t f_cur = static_cast<int64_t>(meta.x);
    int64_t s_cur = static_cast<int64_t>(W_S_OFF[widx]);
    int64_t d_cur = static_cast<int64_t>(W_D_OFF[widx]);
    const int64_t slab_base_uid = static_cast<int64_t>(widx) * spw;

    for (int sl = 0; sl < spw; ++sl) {
        const uint32_t nf = meta_nf(meta, sl);
        const uint32_t out_bit = ((has_out_mask >> sl) & 1u) ? BF11_V5_META_OUTLIER_BIT : 0u;
        uint32_t smeta = out_bit;
        if (nf == 0u) {
            // MODE_ZERO
            smeta = out_bit;
        } else if (static_cast<int>(nf) <= flip_thresh) {
            // MODE_SPARSE (1..flip_thresh, where flip_thresh <= 63)
            for (uint32_t i = static_cast<uint32_t>(lane); i < nf; i += 32u) {
                SFLIPS[s_cur + i] = FLIPS[f_cur + i];
            }
            smeta = (nf << BF11_V5_META_NF_SHIFT) | out_bit | (static_cast<uint32_t>(s_cur) & BF11_V5_META_OFF_MASK);
            s_cur += static_cast<int64_t>(nf);
        } else {
            // MODE_BITMASK (128B coalesced u32 per slab)
            s_bm[lane] = 0u;
            __syncwarp();
            for (uint32_t i = static_cast<uint32_t>(lane); i < nf; i += 32u) {
                const uint32_t frec = static_cast<uint32_t>(FLIPS[f_cur + i]);
                atomicOr(&s_bm[frec >> 5], 1u << (frec & 31u));
            }
            __syncwarp();
            BMASKS[(d_cur << 5) + lane] = s_bm[lane];
            smeta = BF11_V5_META_BITMASK_BIT | out_bit | (static_cast<uint32_t>(d_cur) & BF11_V5_META_OFF_MASK);
            d_cur += 1;
        }
        if (lane == 0) {
            SLAB_META_V5[slab_base_uid + sl] = smeta;
        }
        f_cur += static_cast<int64_t>(nf);
    }
}

// =========================================================================
// Constexpr-Unrolled MMA Step Engines (ZERO Stack Arrays, ZERO Spills!)
// =========================================================================

// C=1 (M <= 16) single-step executor
template <bool M_LE8, int S>
static __device__ __forceinline__ void exec_c1_step_s(
    const Slab11& r, int sl,
    const __nv_bfloat16* sX, int x_off0, int x_off1, int k0_chunk,
    uint32_t base_sh7, uint32_t lane, uint32_t fl, bool slab_has_out,
    const uint64_t* __restrict__ OUTLIERS,
    int32_t& o_cur, int32_t o_end, uint64_t& o_rec, uint32_t& o_next_step,
    float (&acc)[4]
) {
    const int ks_off = k0_chunk + S * 16;
    const uint32_t a0 = *reinterpret_cast<const uint32_t*>(sX + x_off0 + ks_off);
    const uint32_t a2 = *reinterpret_cast<const uint32_t*>(sX + x_off0 + ks_off + 8);
    const uint32_t a1 = M_LE8 ? 0u : *reinterpret_cast<const uint32_t*>(sX + x_off1 + ks_off);
    const uint32_t a3 = M_LE8 ? 0u : *reinterpret_cast<const uint32_t*>(sX + x_off1 + ks_off + 8);

    uint32_t b0, b1;
    unpack4_bf11_flip_v5(slab_sm_word<S>(r), slab_c12_raw<S>(r), fl >> (S * 4), base_sh7, b0, b1);

    if (__builtin_expect(slab_has_out, 0)) {
        patch_outliers_step(static_cast<uint32_t>((sl << 3) | S), lane, OUTLIERS, o_cur, o_end, o_rec, o_next_step, b0, b1);
    }
    mma_m16n8k16_bf16(acc, a0, a1, a2, a3, b0, b1);
}

template <bool M_LE8>
static __device__ __forceinline__ void exec_slab11_c1_v5(
    const Slab11& r, int sl,
    const __nv_bfloat16* sX, int x_off0, int x_off1, int k0_chunk,
    uint32_t base_sh7, uint32_t lane, uint32_t fl, bool slab_has_out,
    const uint64_t* __restrict__ OUTLIERS,
    int32_t& o_cur, int32_t o_end, uint64_t& o_rec, uint32_t& o_next_step,
    float (&acc)[4]
) {
    exec_c1_step_s<M_LE8, 0>(r, sl, sX, x_off0, x_off1, k0_chunk, base_sh7, lane, fl, slab_has_out, OUTLIERS, o_cur, o_end, o_rec, o_next_step, acc);
    exec_c1_step_s<M_LE8, 1>(r, sl, sX, x_off0, x_off1, k0_chunk, base_sh7, lane, fl, slab_has_out, OUTLIERS, o_cur, o_end, o_rec, o_next_step, acc);
    exec_c1_step_s<M_LE8, 2>(r, sl, sX, x_off0, x_off1, k0_chunk, base_sh7, lane, fl, slab_has_out, OUTLIERS, o_cur, o_end, o_rec, o_next_step, acc);
    exec_c1_step_s<M_LE8, 3>(r, sl, sX, x_off0, x_off1, k0_chunk, base_sh7, lane, fl, slab_has_out, OUTLIERS, o_cur, o_end, o_rec, o_next_step, acc);
    exec_c1_step_s<M_LE8, 4>(r, sl, sX, x_off0, x_off1, k0_chunk, base_sh7, lane, fl, slab_has_out, OUTLIERS, o_cur, o_end, o_rec, o_next_step, acc);
    exec_c1_step_s<M_LE8, 5>(r, sl, sX, x_off0, x_off1, k0_chunk, base_sh7, lane, fl, slab_has_out, OUTLIERS, o_cur, o_end, o_rec, o_next_step, acc);
    exec_c1_step_s<M_LE8, 6>(r, sl, sX, x_off0, x_off1, k0_chunk, base_sh7, lane, fl, slab_has_out, OUTLIERS, o_cur, o_end, o_rec, o_next_step, acc);
    exec_c1_step_s<M_LE8, 7>(r, sl, sX, x_off0, x_off1, k0_chunk, base_sh7, lane, fl, slab_has_out, OUTLIERS, o_cur, o_end, o_rec, o_next_step, acc);
}

// C=32 v5 single-step executor with XOR-swizzled SMEM + ldmatrix.x4 (ZERO bank conflicts!)
// `S` is slab-local MMA step (0..7); `S_LOC` is buffer-local step (0..7 for BK=128, 0..3 for BK=64)
template <int MT, int BK, int S, int S_LOC>
static __device__ __forceinline__ void exec_c32_step_swiz_mt(
    const Slab11& r, int sl,
    uint32_t smem_buf_addr, uint32_t lane, uint32_t lane_row_bytes, uint32_t lane_k_half,
    uint32_t base_sh7, uint32_t fl, bool slab_has_out,
    const uint64_t* __restrict__ OUTLIERS,
    int32_t& o_cur, int32_t o_end, uint64_t& o_rec, uint32_t& o_next_step,
    float (&acc)[MT][4]
) {
    uint32_t b0, b1;
    unpack4_bf11_flip_v5(slab_sm_word<S>(r), slab_c12_raw<S>(r), fl >> (S * 4), base_sh7, b0, b1);

    if (__builtin_expect(slab_has_out, 0)) {
        patch_outliers_step(static_cast<uint32_t>((sl << 3) | S), lane, OUTLIERS, o_cur, o_end, o_rec, o_next_step, b0, b1);
    }

    // Swizzled 16-byte chunk offset: loop-invariant across all `mt = 0..MT-1`!
    const uint32_t kv_swiz = (static_cast<uint32_t>(S_LOC * 2) | lane_k_half) ^ (lane & 7u);
    const uint32_t base_addr = smem_buf_addr + lane_row_bytes + (kv_swiz << 4);
    constexpr uint32_t MT_STRIDE_BYTES = static_cast<uint32_t>(16 * BK * 2);

    #pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
        uint32_t a0, a1, a2, a3;
        ldmatrix_x4_smem(base_addr + static_cast<uint32_t>(mt) * MT_STRIDE_BYTES, a0, a1, a2, a3);
        mma_m16n8k16_bf16(acc[mt], a0, a1, a2, a3, b0, b1);
    }
}

template <int MT>
static __device__ __forceinline__ void exec_full_slab11_v5_bk128(
    const Slab11& r, int sl,
    uint32_t smem_buf_addr, uint32_t lane, uint32_t lane_row_bytes, uint32_t lane_k_half,
    uint32_t base_sh7, uint32_t fl, bool slab_has_out,
    const uint64_t* __restrict__ OUTLIERS,
    int32_t& o_cur, int32_t o_end, uint64_t& o_rec, uint32_t& o_next_step,
    float (&acc)[MT][4]
) {
    exec_c32_step_swiz_mt<MT, 128, 0, 0>(r, sl, smem_buf_addr, lane, lane_row_bytes, lane_k_half, base_sh7, fl, slab_has_out, OUTLIERS, o_cur, o_end, o_rec, o_next_step, acc);
    exec_c32_step_swiz_mt<MT, 128, 1, 1>(r, sl, smem_buf_addr, lane, lane_row_bytes, lane_k_half, base_sh7, fl, slab_has_out, OUTLIERS, o_cur, o_end, o_rec, o_next_step, acc);
    exec_c32_step_swiz_mt<MT, 128, 2, 2>(r, sl, smem_buf_addr, lane, lane_row_bytes, lane_k_half, base_sh7, fl, slab_has_out, OUTLIERS, o_cur, o_end, o_rec, o_next_step, acc);
    exec_c32_step_swiz_mt<MT, 128, 3, 3>(r, sl, smem_buf_addr, lane, lane_row_bytes, lane_k_half, base_sh7, fl, slab_has_out, OUTLIERS, o_cur, o_end, o_rec, o_next_step, acc);
    exec_c32_step_swiz_mt<MT, 128, 4, 4>(r, sl, smem_buf_addr, lane, lane_row_bytes, lane_k_half, base_sh7, fl, slab_has_out, OUTLIERS, o_cur, o_end, o_rec, o_next_step, acc);
    exec_c32_step_swiz_mt<MT, 128, 5, 5>(r, sl, smem_buf_addr, lane, lane_row_bytes, lane_k_half, base_sh7, fl, slab_has_out, OUTLIERS, o_cur, o_end, o_rec, o_next_step, acc);
    exec_c32_step_swiz_mt<MT, 128, 6, 6>(r, sl, smem_buf_addr, lane, lane_row_bytes, lane_k_half, base_sh7, fl, slab_has_out, OUTLIERS, o_cur, o_end, o_rec, o_next_step, acc);
    exec_c32_step_swiz_mt<MT, 128, 7, 7>(r, sl, smem_buf_addr, lane, lane_row_bytes, lane_k_half, base_sh7, fl, slab_has_out, OUTLIERS, o_cur, o_end, o_rec, o_next_step, acc);
}

template <int MT, int HALF>
static __device__ __forceinline__ void exec_half_slab11_v5_bk64(
    const Slab11& r, int sl,
    uint32_t smem_buf_addr, uint32_t lane, uint32_t lane_row_bytes, uint32_t lane_k_half,
    uint32_t base_sh7, uint32_t fl, bool slab_has_out,
    const uint64_t* __restrict__ OUTLIERS,
    int32_t& o_cur, int32_t o_end, uint64_t& o_rec, uint32_t& o_next_step,
    float (&acc)[MT][4]
) {
    if constexpr (HALF == 0) {
        exec_c32_step_swiz_mt<MT, 64, 0, 0>(r, sl, smem_buf_addr, lane, lane_row_bytes, lane_k_half, base_sh7, fl, slab_has_out, OUTLIERS, o_cur, o_end, o_rec, o_next_step, acc);
        exec_c32_step_swiz_mt<MT, 64, 1, 1>(r, sl, smem_buf_addr, lane, lane_row_bytes, lane_k_half, base_sh7, fl, slab_has_out, OUTLIERS, o_cur, o_end, o_rec, o_next_step, acc);
        exec_c32_step_swiz_mt<MT, 64, 2, 2>(r, sl, smem_buf_addr, lane, lane_row_bytes, lane_k_half, base_sh7, fl, slab_has_out, OUTLIERS, o_cur, o_end, o_rec, o_next_step, acc);
        exec_c32_step_swiz_mt<MT, 64, 3, 3>(r, sl, smem_buf_addr, lane, lane_row_bytes, lane_k_half, base_sh7, fl, slab_has_out, OUTLIERS, o_cur, o_end, o_rec, o_next_step, acc);
    } else {
        exec_c32_step_swiz_mt<MT, 64, 4, 0>(r, sl, smem_buf_addr, lane, lane_row_bytes, lane_k_half, base_sh7, fl, slab_has_out, OUTLIERS, o_cur, o_end, o_rec, o_next_step, acc);
        exec_c32_step_swiz_mt<MT, 64, 5, 1>(r, sl, smem_buf_addr, lane, lane_row_bytes, lane_k_half, base_sh7, fl, slab_has_out, OUTLIERS, o_cur, o_end, o_rec, o_next_step, acc);
        exec_c32_step_swiz_mt<MT, 64, 6, 2>(r, sl, smem_buf_addr, lane, lane_row_bytes, lane_k_half, base_sh7, fl, slab_has_out, OUTLIERS, o_cur, o_end, o_rec, o_next_step, acc);
        exec_c32_step_swiz_mt<MT, 64, 7, 3>(r, sl, smem_buf_addr, lane, lane_row_bytes, lane_k_half, base_sh7, fl, slab_has_out, OUTLIERS, o_cur, o_end, o_rec, o_next_step, acc);
    }
}

// Legacy padded-SMEM C=32 step executors (for drop-in legacy `bf11_flip_c32_fwd` ABI parity, 0 stack!)
template <int MT, int SX_STRIDE, int S, int S_LOC>
static __device__ __forceinline__ void exec_legacy_c32_step_mt(
    const Slab11& r, int sl,
    const __nv_bfloat16* sX_buf, int g, int t,
    uint32_t base_sh7, uint32_t lane, uint32_t fl,
    const uint64_t* __restrict__ OUTLIERS,
    int32_t& o_cur, int32_t o_end, uint64_t& o_rec, uint32_t& o_next_step,
    float (&acc)[MT][4]
) {
    uint32_t b0, b1;
    unpack4_bf11_flip_v5(slab_sm_word<S>(r), slab_c12_raw<S>(r), fl >> (S * 4), base_sh7, b0, b1);
    patch_outliers_step(static_cast<uint32_t>((sl << 3) | S), lane, OUTLIERS, o_cur, o_end, o_rec, o_next_step, b0, b1);

    const int ks_off = S_LOC * 16 + 2 * t;
    #pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
        const int row0 = mt * 16 + g;
        const int row1 = row0 + 8;
        const uint32_t a0 = *reinterpret_cast<const uint32_t*>(sX_buf + row0 * SX_STRIDE + ks_off);
        const uint32_t a2 = *reinterpret_cast<const uint32_t*>(sX_buf + row0 * SX_STRIDE + ks_off + 8);
        const uint32_t a1 = *reinterpret_cast<const uint32_t*>(sX_buf + row1 * SX_STRIDE + ks_off);
        const uint32_t a3 = *reinterpret_cast<const uint32_t*>(sX_buf + row1 * SX_STRIDE + ks_off + 8);
        mma_m16n8k16_bf16(acc[mt], a0, a1, a2, a3, b0, b1);
    }
}

template <int MT, int HALF>
static __device__ __forceinline__ void exec_legacy_half_slab11_mt(
    const Slab11& r, int sl,
    const __nv_bfloat16* sX_half, int g, int t,
    uint32_t base_sh7, uint32_t lane, uint32_t fl,
    const uint64_t* __restrict__ OUTLIERS,
    int32_t& o_cur, int32_t o_end, uint64_t& o_rec, uint32_t& o_next_step,
    float (&acc)[MT][4]
) {
    if constexpr (HALF == 0) {
        exec_legacy_c32_step_mt<MT, 72, 0, 0>(r, sl, sX_half, g, t, base_sh7, lane, fl, OUTLIERS, o_cur, o_end, o_rec, o_next_step, acc);
        exec_legacy_c32_step_mt<MT, 72, 1, 1>(r, sl, sX_half, g, t, base_sh7, lane, fl, OUTLIERS, o_cur, o_end, o_rec, o_next_step, acc);
        exec_legacy_c32_step_mt<MT, 72, 2, 2>(r, sl, sX_half, g, t, base_sh7, lane, fl, OUTLIERS, o_cur, o_end, o_rec, o_next_step, acc);
        exec_legacy_c32_step_mt<MT, 72, 3, 3>(r, sl, sX_half, g, t, base_sh7, lane, fl, OUTLIERS, o_cur, o_end, o_rec, o_next_step, acc);
    } else {
        exec_legacy_c32_step_mt<MT, 72, 4, 0>(r, sl, sX_half, g, t, base_sh7, lane, fl, OUTLIERS, o_cur, o_end, o_rec, o_next_step, acc);
        exec_legacy_c32_step_mt<MT, 72, 5, 1>(r, sl, sX_half, g, t, base_sh7, lane, fl, OUTLIERS, o_cur, o_end, o_rec, o_next_step, acc);
        exec_legacy_c32_step_mt<MT, 72, 6, 2>(r, sl, sX_half, g, t, base_sh7, lane, fl, OUTLIERS, o_cur, o_end, o_rec, o_next_step, acc);
        exec_legacy_c32_step_mt<MT, 72, 7, 3>(r, sl, sX_half, g, t, base_sh7, lane, fl, OUTLIERS, o_cur, o_end, o_rec, o_next_step, acc);
    }
}

template <int MT>
static __device__ __forceinline__ void exec_legacy_full_slab11_mt(
    const Slab11& r, int sl,
    const __nv_bfloat16* sX_chunk, int g, int t,
    uint32_t base_sh7, uint32_t lane, uint32_t fl,
    const uint64_t* __restrict__ OUTLIERS,
    int32_t& o_cur, int32_t o_end, uint64_t& o_rec, uint32_t& o_next_step,
    float (&acc)[MT][4]
) {
    exec_legacy_c32_step_mt<MT, 136, 0, 0>(r, sl, sX_chunk, g, t, base_sh7, lane, fl, OUTLIERS, o_cur, o_end, o_rec, o_next_step, acc);
    exec_legacy_c32_step_mt<MT, 136, 1, 1>(r, sl, sX_chunk, g, t, base_sh7, lane, fl, OUTLIERS, o_cur, o_end, o_rec, o_next_step, acc);
    exec_legacy_c32_step_mt<MT, 136, 2, 2>(r, sl, sX_chunk, g, t, base_sh7, lane, fl, OUTLIERS, o_cur, o_end, o_rec, o_next_step, acc);
    exec_legacy_c32_step_mt<MT, 136, 3, 3>(r, sl, sX_chunk, g, t, base_sh7, lane, fl, OUTLIERS, o_cur, o_end, o_rec, o_next_step, acc);
    exec_legacy_c32_step_mt<MT, 136, 4, 4>(r, sl, sX_chunk, g, t, base_sh7, lane, fl, OUTLIERS, o_cur, o_end, o_rec, o_next_step, acc);
    exec_legacy_c32_step_mt<MT, 136, 5, 5>(r, sl, sX_chunk, g, t, base_sh7, lane, fl, OUTLIERS, o_cur, o_end, o_rec, o_next_step, acc);
    exec_legacy_c32_step_mt<MT, 136, 6, 6>(r, sl, sX_chunk, g, t, base_sh7, lane, fl, OUTLIERS, o_cur, o_end, o_rec, o_next_step, acc);
    exec_legacy_c32_step_mt<MT, 136, 7, 7>(r, sl, sX_chunk, g, t, base_sh7, lane, fl, OUTLIERS, o_cur, o_end, o_rec, o_next_step, acc);
}

// =========================================================================
// Shared Helper: XOR-Swizzled Async X Tile Loader (BK = 64 or 128)
// =========================================================================

template <int BM, int BK, int THREADS>
static __device__ __forceinline__ void load_x_swiz_async(
    const __nv_bfloat16* __restrict__ X, int sx,
    int m_base, int m_eff, int k_off,
    __nv_bfloat16* dst, int tid
) {
    constexpr int VEC_PER_ROW = BK >> 3;        // 8 for BK=64, 16 for BK=128
    constexpr int VEC_SHIFT = (BK == 128) ? 4 : 3;
    constexpr int VEC_MASK = VEC_PER_ROW - 1;
    constexpr int TOTAL_VEC = BM * VEC_PER_ROW;
    for (int idx = tid; idx < TOTAL_VEC; idx += THREADS) {
        const int lm = idx >> VEC_SHIFT;
        const int kv = idx & VEC_MASK;
        const int kv_swiz = kv ^ (lm & 7);
        const int gm = m_base + lm;
        __nv_bfloat16* d = dst + lm * BK + (kv_swiz << 3);
        if (gm < m_eff) {
            cp_async_ca_16(d, X + static_cast<int64_t>(gm) * sx + k_off + (kv << 3));
        } else {
            *reinterpret_cast<uint4*>(d) = make_uint4(0u, 0u, 0u, 0u);
        }
    }
    cp_async_commit();
}

// =========================================================================
// PART 1: NEW ROOFLINE-HARDENED v5 C=1 (M <= 16) KERNELS
// =========================================================================

template <int WARPS_RUN, int SUB_N, bool M_LE8, int SPLIT, bool PDL, bool TRIG>
__global__ __launch_bounds__(WARPS_RUN * 32, (M_LE8 ? (WARPS_RUN == 4 ? 5 : 6) : (WARPS_RUN == 4 ? 3 : 4)))
void bf11_flip_v5_c1_kernel(
    const __nv_bfloat16* __restrict__ X,
    const uint8_t* __restrict__ SLABS,
    const uint32_t* __restrict__ SLAB_META_V5,
    const uint16_t* __restrict__ SFLIPS,
    const uint32_t* __restrict__ BMASKS,
    const uint64_t* __restrict__ OUTLIERS,
    const int32_t* __restrict__ O_OFF,
    float* __restrict__ P,
    int32_t* __restrict__ SEM,
    __nv_bfloat16* __restrict__ O,
    const int32_t* __restrict__ NLIVE_PTR,
    int M, int N, int N_active, int K, int sx, int KS, int so_m,
    int w_global, uint32_t base_exp
) {
    constexpr int BN = WARPS_RUN * SUB_N * 8;
    constexpr int THREADS = WARPS_RUN * 32;
    constexpr bool DIRECT = (SPLIT == 1);
    constexpr int M_ROWS = M_LE8 ? 8 : 16;

    const int pn = static_cast<int>(blockIdx.x);
    const int ps = static_cast<int>(blockIdx.y);
    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g = lane >> 2;
    const int t = lane & 3;

    const int w_row = pn * WARPS_RUN + warp;
    const int n_warp_base = w_row * (SUB_N * 8);
    const bool warp_active = (w_row < w_global) && (n_warp_base < N_active);

    const int k_lo = ps * KS;
    const int k_hi = (k_lo + KS < K) ? (k_lo + KS) : K;
    const int k_len = k_hi - k_lo;
    const int n_chunks = k_len >> 7;
    const int total_slabs = warp_active ? (n_chunks * SUB_N) : 0;

    const int64_t warp_uid = static_cast<int64_t>(ps) * w_global + w_row;
    const int64_t slabs_per_warp = static_cast<int64_t>((KS >> 7) * SUB_N);
    const int64_t slab_uid_base = warp_uid * slabs_per_warp;
    const uint8_t* slab_base = SLABS + slab_uid_base * SLAB11_BYTES;

    Slab11 cur_slab;
    uint32_t my_meta = 0u;
    uint32_t cur_smeta = 0u;
    uint32_t cur_raw_fl = 0u;
    int32_t o_cur = 0, o_end = 0;
    uint64_t o_rec = 0xFFFFFFFFFFFFFFFFULL;
    uint32_t o_next_step = 0xFFFFFFFFu;

    auto init_weight_streams = [&]() {
        if (total_slabs > 0) {
            cur_slab = load_slab11(slab_base, lane);
            if (lane < total_slabs) {
                my_meta = ld_ca_u32(SLAB_META_V5 + slab_uid_base + lane);
            }
            o_cur = __ldg(O_OFF + warp_uid);
            o_end = __ldg(O_OFF + warp_uid + 1);
            cur_smeta = __shfl_sync(0xFFFFFFFFu, my_meta, 0);
            cur_raw_fl = issue_flip_load_v5<false>(cur_smeta, SFLIPS, BMASKS, lane);
            if (total_slabs > 1 && lane < 11) {
                prefetch_l2(slab_base + SLAB11_BYTES + lane * 128);
            }
            if (o_cur < o_end) {
                o_rec = __ldg(OUTLIERS + o_cur);
                o_next_step = static_cast<uint32_t>(o_rec >> 32);
            }
        }
    };

    if constexpr (PDL) {
        init_weight_streams();
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }

    const int m_eff = (NLIVE_PTR != nullptr) ? min(M, *NLIVE_PTR) : M;

    __shared__ uint32_t s_flip[WARPS_RUN][32];
    extern __shared__ __nv_bfloat16 sX[];

    const int sX_stride = KS + 8;
    const int vec_k = k_len >> 3;
    const int m_load = min(m_eff, M_ROWS);
    const int total_vec = M_ROWS * vec_k;
    for (int idx = tid; idx < total_vec; idx += THREADS) {
        const int m = idx / vec_k;
        const int kv = idx - m * vec_k;
        __nv_bfloat16* d = sX + m * sX_stride + (kv << 3);
        if (m < m_load) {
            cp_async_ca_16(d, X + static_cast<int64_t>(m) * sx + k_lo + (kv << 3));
        } else {
            *reinterpret_cast<uint4*>(d) = make_uint4(0u, 0u, 0u, 0u);
        }
    }
    cp_async_commit();

    if constexpr (!PDL) {
        init_weight_streams();
    }

    cp_async_wait<0>();
    __syncthreads();

    float acc[SUB_N][4];
    #pragma unroll
    for (int s = 0; s < SUB_N; ++s) {
        acc[s][0] = 0.0f; acc[s][1] = 0.0f; acc[s][2] = 0.0f; acc[s][3] = 0.0f;
    }

    const int x_off0 = g * sX_stride + 2 * t;
    const int x_off1 = (g + 8) * sX_stride + 2 * t;
    const uint32_t base_sh7 = (base_exp << 7) | (base_exp << 23);

    for (int sl = 0; sl < total_slabs; ++sl) {
        Slab11 nxt_slab;
        uint32_t nxt_smeta = 0u;
        uint32_t nxt_raw_fl = 0u;
        if (sl + 1 < total_slabs) {
            nxt_slab = load_slab11(slab_base + static_cast<int64_t>(sl + 1) * SLAB11_BYTES, lane);
            nxt_smeta = __shfl_sync(0xFFFFFFFFu, my_meta, sl + 1);
            nxt_raw_fl = issue_flip_load_v5<false>(nxt_smeta, SFLIPS, BMASKS, lane);
            if (sl + 2 < total_slabs && lane < 11) {
                prefetch_l2(slab_base + static_cast<int64_t>(sl + 2) * SLAB11_BYTES + lane * 128);
            }
        }

        const uint32_t fl = finish_flip_v5<false>(cur_raw_fl, cur_smeta, SFLIPS, lane, s_flip[warp]);
        const bool slab_has_out = (cur_smeta & BF11_V5_META_OUTLIER_BIT) != 0u;

        if constexpr (SUB_N == 2) {
            const int kc = sl >> 1;
            if ((sl & 1) == 0) {
                exec_slab11_c1_v5<M_LE8>(cur_slab, sl, sX, x_off0, x_off1, kc * 128, base_sh7, static_cast<uint32_t>(lane),
                                         fl, slab_has_out, OUTLIERS, o_cur, o_end, o_rec, o_next_step, acc[0]);
            } else {
                exec_slab11_c1_v5<M_LE8>(cur_slab, sl, sX, x_off0, x_off1, kc * 128, base_sh7, static_cast<uint32_t>(lane),
                                         fl, slab_has_out, OUTLIERS, o_cur, o_end, o_rec, o_next_step, acc[1]);
            }
        } else {
            exec_slab11_c1_v5<M_LE8>(cur_slab, sl, sX, x_off0, x_off1, sl * 128, base_sh7, static_cast<uint32_t>(lane),
                                     fl, slab_has_out, OUTLIERS, o_cur, o_end, o_rec, o_next_step, acc[0]);
        }

        cur_slab = nxt_slab;
        cur_smeta = nxt_smeta;
        cur_raw_fl = nxt_raw_fl;
    }

    const int m0 = g;
    const int m1 = g + 8;

    if constexpr (DIRECT) {
        if constexpr (TRIG) {
            if (tid == 0) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
        #pragma unroll
        for (int sub = 0; sub < SUB_N; ++sub) {
            const int n0 = n_warp_base + sub * 8 + 2 * t;
            const int n1 = n0 + 1;
            const float v00 = (n0 < N_active) ? acc[sub][0] : 0.0f;
            const float v01 = (n1 < N_active) ? acc[sub][1] : 0.0f;
            const float v10 = (n0 < N_active) ? acc[sub][2] : 0.0f;
            const float v11 = (n1 < N_active) ? acc[sub][3] : 0.0f;
            if (m0 < m_eff && n1 < N) {
                __nv_bfloat162 p0 = __halves2bfloat162(__float2bfloat16_rn(v00 + 0.0f), __float2bfloat16_rn(v01 + 0.0f));
                *reinterpret_cast<__nv_bfloat162*>(O + static_cast<int64_t>(m0) * so_m + n0) = p0;
            } else if (m0 < m_eff && n0 < N) {
                O[static_cast<int64_t>(m0) * so_m + n0] = __float2bfloat16_rn(v00 + 0.0f);
            }
            if constexpr (!M_LE8) {
                if (m1 < m_eff && n1 < N) {
                    __nv_bfloat162 p1 = __halves2bfloat162(__float2bfloat16_rn(v10 + 0.0f), __float2bfloat16_rn(v11 + 0.0f));
                    *reinterpret_cast<__nv_bfloat162*>(O + static_cast<int64_t>(m1) * so_m + n0) = p1;
                } else if (m1 < m_eff && n0 < N) {
                    O[static_cast<int64_t>(m1) * so_m + n0] = __float2bfloat16_rn(v10 + 0.0f);
                }
            }
        }
    } else {
        const int64_t tile_stride = static_cast<int64_t>(SPLIT) * M * BN;
        const int64_t split_stride = static_cast<int64_t>(M) * BN;
        float* p_tile = P + static_cast<int64_t>(pn) * tile_stride;
        float* p_my = p_tile + static_cast<int64_t>(ps) * split_stride;
        #pragma unroll
        for (int sub = 0; sub < SUB_N; ++sub) {
            const int nl0 = warp * (SUB_N * 8) + sub * 8 + 2 * t;
            const int n0 = pn * BN + nl0;
            const int n1 = n0 + 1;
            const float v00 = (n0 < N_active) ? acc[sub][0] : 0.0f;
            const float v01 = (n1 < N_active) ? acc[sub][1] : 0.0f;
            const float v10 = (n0 < N_active) ? acc[sub][2] : 0.0f;
            const float v11 = (n1 < N_active) ? acc[sub][3] : 0.0f;
            if (m0 < m_eff) {
                *reinterpret_cast<float2*>(p_my + m0 * BN + nl0) = make_float2(v00, v01);
            }
            if constexpr (!M_LE8) {
                if (m1 < m_eff) {
                    *reinterpret_cast<float2*>(p_my + m1 * BN + nl0) = make_float2(v10, v11);
                }
            }
        }

        __syncthreads();
        __shared__ int s_done;
        if (tid == 0) {
            s_done = atomic_add_acq_rel_gpu(&SEM[pn], 1);
        }
        __syncthreads();

        if (s_done == SPLIT - 1) {
            if (tid == 0) st_release_gpu_s32(&SEM[pn], 0);
            if constexpr (TRIG) {
                if (tid == 0) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
            constexpr int PAIRS_PER_ROW = BN / 2;
            const int total_pairs = m_eff * PAIRS_PER_ROW;
            const int n_tile_base = pn * BN;
            for (int idx = tid; idx < total_pairs; idx += THREADS) {
                const int m = idx / PAIRS_PER_ROW;
                const int p = idx - m * PAIRS_PER_ROW;
                const int nl = p * 2;
                const int n0 = n_tile_base + nl;
                if (n0 >= N) continue;
                const int64_t off = static_cast<int64_t>(m) * BN + nl;
                float s0 = 0.0f, s1 = 0.0f;
                #pragma unroll
                for (int s = 0; s < SPLIT; ++s) {
                    float2 v = ld_cg_f2(reinterpret_cast<const float2*>(p_tile + static_cast<int64_t>(s) * split_stride + off));
                    s0 += v.x;
                    s1 += v.y;
                }
                if (n0 + 1 < N) {
                    *reinterpret_cast<__nv_bfloat162*>(O + static_cast<int64_t>(m) * so_m + n0) =
                        __halves2bfloat162(__float2bfloat16_rn(s0 + 0.0f), __float2bfloat16_rn(s1 + 0.0f));
                } else {
                    O[static_cast<int64_t>(m) * so_m + n0] = __float2bfloat16_rn(s0 + 0.0f);
                }
            }
        }
    }
}

template <int WARPS_RUN, bool M_LE8, int SPLIT, bool PDL, bool TRIG>
__global__ __launch_bounds__(WARPS_RUN * 32, (M_LE8 ? (WARPS_RUN == 4 ? 5 : 6) : (WARPS_RUN == 4 ? 4 : 5)))
void bf11_flip_dual_kv_v5_c1_kernel(
    const __nv_bfloat16* __restrict__ X0, const __nv_bfloat16* __restrict__ X1,
    const uint8_t* __restrict__ SLABS0, const uint8_t* __restrict__ SLABS1,
    const uint32_t* __restrict__ SMETA0, const uint32_t* __restrict__ SMETA1,
    const uint16_t* __restrict__ SFLIPS0, const uint16_t* __restrict__ SFLIPS1,
    const uint32_t* __restrict__ BMASKS0, const uint32_t* __restrict__ BMASKS1,
    const uint64_t* __restrict__ OUTLIERS0, const uint64_t* __restrict__ OUTLIERS1,
    const int32_t* __restrict__ O_OFF0, const int32_t* __restrict__ O_OFF1,
    float* __restrict__ P, int32_t* __restrict__ SEM,
    __nv_bfloat16* __restrict__ O0, __nv_bfloat16* __restrict__ O1,
    const int32_t* __restrict__ NLIVE_PTR,
    int M, int N_each, int K, int sx0, int sx1, int KS, int so0, int so1,
    int w_global_each, uint32_t base_exp0, uint32_t base_exp1
) {
    constexpr int BN = WARPS_RUN * 8;
    constexpr int THREADS = WARPS_RUN * 32;
    constexpr int M_ROWS = M_LE8 ? 8 : 16;

    const int n_tiles_each = static_cast<int>(gridDim.x) >> 1;
    const int which = (static_cast<int>(blockIdx.x) >= n_tiles_each) ? 1 : 0;
    const int pn_local = which ? (static_cast<int>(blockIdx.x) - n_tiles_each) : static_cast<int>(blockIdx.x);
    const int pn_global = static_cast<int>(blockIdx.x);
    const int ps = static_cast<int>(blockIdx.y);
    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g = lane >> 2;
    const int t = lane & 3;

    const __nv_bfloat16* X = which ? X1 : X0;
    const uint8_t* SLABS = which ? SLABS1 : SLABS0;
    const uint32_t* SLAB_META_V5 = which ? SMETA1 : SMETA0;
    const uint16_t* SFLIPS = which ? SFLIPS1 : SFLIPS0;
    const uint32_t* BMASKS = which ? BMASKS1 : BMASKS0;
    const uint64_t* OUTLIERS = which ? OUTLIERS1 : OUTLIERS0;
    const int32_t* O_OFF = which ? O_OFF1 : O_OFF0;
    __nv_bfloat16* O = which ? O1 : O0;
    const int sx = which ? sx1 : sx0;
    const int so_m = which ? so1 : so0;
    const uint32_t base_exp = which ? base_exp1 : base_exp0;

    const int w_row = pn_local * WARPS_RUN + warp;
    const bool warp_active = (w_row < w_global_each) && (w_row * 8 < N_each);

    const int k_lo = ps * KS;
    const int k_hi = (k_lo + KS < K) ? (k_lo + KS) : K;
    const int k_len = k_hi - k_lo;
    const int total_slabs = warp_active ? (k_len >> 7) : 0;

    const int64_t warp_uid = static_cast<int64_t>(ps) * w_global_each + w_row;
    const int64_t slabs_per_warp = static_cast<int64_t>(KS >> 7);
    const int64_t slab_uid_base = warp_uid * slabs_per_warp;
    const uint8_t* slab_base = SLABS + slab_uid_base * SLAB11_BYTES;

    Slab11 cur_slab;
    uint32_t my_meta = 0u;
    uint32_t cur_smeta = 0u;
    uint32_t cur_raw_fl = 0u;
    int32_t o_cur = 0, o_end = 0;
    uint64_t o_rec = 0xFFFFFFFFFFFFFFFFULL;
    uint32_t o_next_step = 0xFFFFFFFFu;

    auto init_weight_streams = [&]() {
        if (total_slabs > 0) {
            cur_slab = load_slab11(slab_base, lane);
            if (lane < total_slabs) {
                my_meta = ld_ca_u32(SLAB_META_V5 + slab_uid_base + lane);
            }
            o_cur = __ldg(O_OFF + warp_uid);
            o_end = __ldg(O_OFF + warp_uid + 1);
            cur_smeta = __shfl_sync(0xFFFFFFFFu, my_meta, 0);
            cur_raw_fl = issue_flip_load_v5<false>(cur_smeta, SFLIPS, BMASKS, lane);
            if (total_slabs > 1 && lane < 11) {
                prefetch_l2(slab_base + SLAB11_BYTES + lane * 128);
            }
            if (o_cur < o_end) {
                o_rec = __ldg(OUTLIERS + o_cur);
                o_next_step = static_cast<uint32_t>(o_rec >> 32);
            }
        }
    };

    if constexpr (PDL) {
        init_weight_streams();
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }

    const int m_eff = (NLIVE_PTR != nullptr) ? min(M, *NLIVE_PTR) : M;

    __shared__ uint32_t s_flip[WARPS_RUN][32];
    extern __shared__ __nv_bfloat16 sX[];
    const int sX_stride = KS + 8;
    const int vec_k = k_len >> 3;
    const int m_load = min(m_eff, M_ROWS);
    const int total_vec = M_ROWS * vec_k;
    for (int idx = tid; idx < total_vec; idx += THREADS) {
        const int m = idx / vec_k;
        const int kv = idx - m * vec_k;
        __nv_bfloat16* d = sX + m * sX_stride + (kv << 3);
        if (m < m_load) {
            cp_async_ca_16(d, X + static_cast<int64_t>(m) * sx + k_lo + (kv << 3));
        } else {
            *reinterpret_cast<uint4*>(d) = make_uint4(0u, 0u, 0u, 0u);
        }
    }
    cp_async_commit();

    if constexpr (!PDL) {
        init_weight_streams();
    }

    cp_async_wait<0>();
    __syncthreads();

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const int x_off0 = g * sX_stride + 2 * t;
    const int x_off1 = (g + 8) * sX_stride + 2 * t;
    const uint32_t base_sh7 = (base_exp << 7) | (base_exp << 23);

    for (int sl = 0; sl < total_slabs; ++sl) {
        Slab11 nxt_slab;
        uint32_t nxt_smeta = 0u;
        uint32_t nxt_raw_fl = 0u;
        if (sl + 1 < total_slabs) {
            nxt_slab = load_slab11(slab_base + static_cast<int64_t>(sl + 1) * SLAB11_BYTES, lane);
            nxt_smeta = __shfl_sync(0xFFFFFFFFu, my_meta, sl + 1);
            nxt_raw_fl = issue_flip_load_v5<false>(nxt_smeta, SFLIPS, BMASKS, lane);
        }
        const uint32_t fl = finish_flip_v5<false>(cur_raw_fl, cur_smeta, SFLIPS, lane, s_flip[warp]);
        const bool slab_has_out = (cur_smeta & BF11_V5_META_OUTLIER_BIT) != 0u;
        exec_slab11_c1_v5<M_LE8>(cur_slab, sl, sX, x_off0, x_off1, sl * 128, base_sh7, static_cast<uint32_t>(lane),
                                 fl, slab_has_out, OUTLIERS, o_cur, o_end, o_rec, o_next_step, acc);
        cur_slab = nxt_slab;
        cur_smeta = nxt_smeta;
        cur_raw_fl = nxt_raw_fl;
    }

    const int m0 = g;
    const int m1 = g + 8;
    const int64_t tile_stride = static_cast<int64_t>(SPLIT) * M * BN;
    const int64_t split_stride = static_cast<int64_t>(M) * BN;
    float* p_tile = P + static_cast<int64_t>(pn_global) * tile_stride;
    float* p_my = p_tile + static_cast<int64_t>(ps) * split_stride;
    const int nl0 = warp * 8 + 2 * t;
    const int n0_w = pn_local * BN + nl0;
    const int n1_w = n0_w + 1;
    const float v00 = (n0_w < N_each) ? acc[0] : 0.0f;
    const float v01 = (n1_w < N_each) ? acc[1] : 0.0f;
    const float v10 = (n0_w < N_each) ? acc[2] : 0.0f;
    const float v11 = (n1_w < N_each) ? acc[3] : 0.0f;
    if (m0 < m_eff) *reinterpret_cast<float2*>(p_my + m0 * BN + nl0) = make_float2(v00, v01);
    if constexpr (!M_LE8) {
        if (m1 < m_eff) *reinterpret_cast<float2*>(p_my + m1 * BN + nl0) = make_float2(v10, v11);
    }

    __syncthreads();
    __shared__ int s_done;
    if (tid == 0) {
        s_done = atomic_add_acq_rel_gpu(&SEM[pn_global], 1);
    }
    __syncthreads();

    if (s_done == SPLIT - 1) {
        if (tid == 0) st_release_gpu_s32(&SEM[pn_global], 0);
        if constexpr (TRIG) {
            if (tid == 0) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
        constexpr int PAIRS_PER_ROW = BN / 2;
        const int total_pairs = m_eff * PAIRS_PER_ROW;
        const int n_tile_base = pn_local * BN;
        for (int idx = tid; idx < total_pairs; idx += THREADS) {
            const int m = idx / PAIRS_PER_ROW;
            const int p = idx - m * PAIRS_PER_ROW;
            const int nl = p * 2;
            const int n0 = n_tile_base + nl;
            if (n0 >= N_each) continue;
            const int64_t off = static_cast<int64_t>(m) * BN + nl;
            float s0 = 0.0f, s1 = 0.0f;
            #pragma unroll
            for (int s = 0; s < SPLIT; ++s) {
                float2 v = ld_cg_f2(reinterpret_cast<const float2*>(p_tile + static_cast<int64_t>(s) * split_stride + off));
                s0 += v.x;
                s1 += v.y;
            }
            if (n0 + 1 < N_each) {
                *reinterpret_cast<__nv_bfloat162*>(O + static_cast<int64_t>(m) * so_m + n0) =
                    __halves2bfloat162(__float2bfloat16_rn(s0 + 0.0f), __float2bfloat16_rn(s1 + 0.0f));
            } else {
                O[static_cast<int64_t>(m) * so_m + n0] = __float2bfloat16_rn(s0 + 0.0f);
            }
        }
    }
}

// =========================================================================
// PART 2: NEW ROOFLINE-HARDENED v5 C=32 (M = 17..256+) KERNELS
// =========================================================================

// Primary v5 C=32 SUB_N==1 Roofline Kernel:
//   - BK=128 (single barrier per slab, XOR-swizzled 0-padding) OR BK=64 (BM=128 3 CTAs/SM!)
//   - `ldmatrix.sync.aligned.m8n8.x4` (4x fewer SMEM loads, 0-way bank conflicts!)
//   - Adaptive `SLAB_META_V5` (`MODE_ZERO` / `SHFL<=4` / `SPARSE` / `BITMASK`) + `OUTLIER_BIT` skip
//   - Flattened cross-split `(ps, kc) -> (ps+1, 0)` software-pipelined prefetch at TOP of slab!
//   - Virtual `WARPS_RUN` decoupling (`1, 2, 4`) with ZERO repacking!
//   - `SPLIT_CTA == 1` (zero partials) OR `SPLIT_CTA == 6|9` (L2-resident fused semaphore Split-K,
//     LITERAL 0-ULP bit-identical to `k_order == 1` and `skinny.py`!)
template <
    int WARPS_RUN,
    int MT,
    int BK,
    int K_ORDER,
    int SPLIT_CTA,
    bool CACHE_L2,
    bool PDL,
    bool TRIG
>
__global__ __launch_bounds__(WARPS_RUN * 32, (MT >= 4 ? 3 : (MT == 3 ? (WARPS_RUN == 4 ? 3 : 4) : (WARPS_RUN == 4 ? 4 : 6))))
void bf11_flip_v5_c32_sub1_kernel(
    const __nv_bfloat16* __restrict__ X,
    const uint8_t* __restrict__ SLABS,
    const uint32_t* __restrict__ SLAB_META_V5,
    const uint16_t* __restrict__ SFLIPS,
    const uint32_t* __restrict__ BMASKS,
    const uint64_t* __restrict__ OUTLIERS,
    const int32_t* __restrict__ O_OFF,
    float* __restrict__ P,
    int32_t* __restrict__ SEM,
    __nv_bfloat16* __restrict__ O,
    const int32_t* __restrict__ NLIVE_PTR,
    int M, int N, int N_active, int K, int sx, int KS, int split_packed, int so_m,
    int w_global, uint32_t base_exp
) {
    (void)K;
    constexpr int BN = WARPS_RUN * 8;
    constexpr int BM = MT * 16;
    constexpr int THREADS = WARPS_RUN * 32;
    constexpr int BUF_ELEMS = BM * BK;
    constexpr bool USE_SPLIT = (SPLIT_CTA > 1);
    constexpr bool USE_PACC = (K_ORDER == 1 && !USE_SPLIT);

    const int mb = static_cast<int>(blockIdx.x);
    const int pn = static_cast<int>(blockIdx.y);
    const int ps_start = USE_SPLIT ? static_cast<int>(blockIdx.z) : 0;
    const int ps_count = USE_SPLIT ? 1 : split_packed;
    const int m_blocks = static_cast<int>(gridDim.x);
    const int m_base = mb * BM;

    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g = lane >> 2;
    const int t = lane & 3;
    const uint32_t lane_row_bytes = static_cast<uint32_t>((lane & 15) * BK * 2);
    const uint32_t lane_k_half = static_cast<uint32_t>(lane >> 4);

    const int w_row = pn * WARPS_RUN + warp;
    const int n_warp_base = w_row * 8;
    const bool warp_active = (w_row < w_global) && (n_warp_base < N_active);
    const int n_chunks_per_sp = KS >> 7;
    const int total_chunks = ps_count * n_chunks_per_sp;

    int64_t warp_uid = static_cast<int64_t>(ps_start) * w_global + w_row;
    Slab11 cur_slab;
    uint32_t my_meta_cur = 0u;
    uint32_t my_meta_nxt = 0u;
    uint32_t cur_smeta = 0u;
    uint32_t cur_raw_fl = 0u;
    int32_t o_cur = 0, o_end = 0;
    uint64_t o_rec = 0xFFFFFFFFFFFFFFFFULL;
    uint32_t o_next_step = 0xFFFFFFFFu;

    auto init_weight_streams = [&]() {
        if (warp_active && total_chunks > 0) {
            const int64_t suid0 = warp_uid * n_chunks_per_sp;
            cur_slab = load_slab11_t<CACHE_L2>(SLABS + suid0 * SLAB11_BYTES, lane);
            if (lane < n_chunks_per_sp) {
                my_meta_cur = ld_ca_u32(SLAB_META_V5 + suid0 + lane);
            }
            if constexpr (!USE_SPLIT) {
                if (ps_start + 1 < split_packed && lane < n_chunks_per_sp) {
                    const int64_t wuid1 = static_cast<int64_t>(ps_start + 1) * w_global + w_row;
                    my_meta_nxt = ld_ca_u32(SLAB_META_V5 + wuid1 * n_chunks_per_sp + lane);
                }
            }
            o_cur = __ldg(O_OFF + warp_uid);
            o_end = __ldg(O_OFF + warp_uid + 1);
            cur_smeta = __shfl_sync(0xFFFFFFFFu, my_meta_cur, 0);
            cur_raw_fl = issue_flip_load_v5<CACHE_L2>(cur_smeta, SFLIPS, BMASKS, lane);
            if (o_cur < o_end) {
                o_rec = __ldg(OUTLIERS + o_cur);
                o_next_step = static_cast<uint32_t>(o_rec >> 32);
            }
            if (total_chunks > 1 && lane < 11) {
                const int64_t suid1 = (n_chunks_per_sp > 1)
                    ? (suid0 + 1)
                    : ((static_cast<int64_t>(ps_start + 1) * w_global + w_row) * n_chunks_per_sp);
                prefetch_l2(SLABS + suid1 * SLAB11_BYTES + lane * 128);
            }
        }
    };

    if constexpr (PDL) {
        init_weight_streams();
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }

    const int m_eff = (NLIVE_PTR != nullptr) ? min(M, *NLIVE_PTR) : M;
    if (m_base >= m_eff) {
        if constexpr (TRIG) {
            if ((!USE_SPLIT || ps_start == SPLIT_CTA - 1) && tid == 0) {
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
        }
        return;
    }

    __shared__ uint32_t s_flip[WARPS_RUN][32];
    extern __shared__ __nv_bfloat16 sX[];
    const uint32_t smem_base_0 = static_cast<uint32_t>(__cvta_generic_to_shared(sX));
    const uint32_t smem_base_1 = static_cast<uint32_t>(__cvta_generic_to_shared(sX + BUF_ELEMS));

    float acc[MT][4];
    #pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
        acc[mt][0] = 0.0f; acc[mt][1] = 0.0f; acc[mt][2] = 0.0f; acc[mt][3] = 0.0f;
    }

    [[maybe_unused]] float p_acc[USE_PACC ? MT : 1][4];
    if constexpr (USE_PACC) {
        #pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            p_acc[mt][0] = 0.0f; p_acc[mt][1] = 0.0f; p_acc[mt][2] = 0.0f; p_acc[mt][3] = 0.0f;
        }
    }

    const uint32_t base_sh7 = (base_exp << 7) | (base_exp << 23);

    if (total_chunks > 0) {
        // Kick off chunk 0 X async load BEFORE/IN-PARALLEL with weight stream init!
        load_x_swiz_async<BM, BK, THREADS>(X, sx, m_base, m_eff, ps_start * KS, sX, tid);

        if constexpr (!PDL) {
            init_weight_streams();
        }

        int ps = ps_start;
        int kc = 0;
        for (int c = 0; c < total_chunks; ++c) {
            const int k_chunk = ps * KS + kc * 128;
            const bool crossing_ps = (kc + 1 == n_chunks_per_sp);
            const bool has_next = (c + 1 < total_chunks);

            // Wait for current stage X tile to arrive
            cp_async_wait<0>();
            __syncthreads();

            if constexpr (BK == 128) {
                const int buf = c & 1;
                if (has_next) {
                    load_x_swiz_async<BM, 128, THREADS>(
                        X, sx, m_base, m_eff, k_chunk + 128,
                        sX + (buf ^ 1) * BUF_ELEMS, tid
                    );
                }

                // Issue NEXT slab + NEXT flip (+ next split O_OFF) AT THE VERY TOP OF THE SLAB!
                Slab11 nxt_slab;
                uint32_t nxt_smeta = 0u;
                uint32_t nxt_raw_fl = 0u;
                int32_t nxt_o_cur = 0, nxt_o_end = 0;
                if (warp_active && has_next) {
                    int64_t nxt_suid;
                    if (!crossing_ps) {
                        nxt_suid = warp_uid * n_chunks_per_sp + (kc + 1);
                        nxt_smeta = __shfl_sync(0xFFFFFFFFu, my_meta_cur, kc + 1);
                    } else {
                        const int64_t nxt_wuid = static_cast<int64_t>(ps + 1) * w_global + w_row;
                        nxt_suid = nxt_wuid * n_chunks_per_sp;
                        nxt_smeta = __shfl_sync(0xFFFFFFFFu, my_meta_nxt, 0);
                        nxt_o_cur = __ldg(O_OFF + nxt_wuid);
                        nxt_o_end = __ldg(O_OFF + nxt_wuid + 1);
                    }
                    nxt_slab = load_slab11_t<CACHE_L2>(SLABS + nxt_suid * SLAB11_BYTES, lane);
                    nxt_raw_fl = issue_flip_load_v5<CACHE_L2>(nxt_smeta, SFLIPS, BMASKS, lane);
                }

                const uint32_t fl = warp_active
                    ? finish_flip_v5<CACHE_L2>(cur_raw_fl, cur_smeta, SFLIPS, lane, s_flip[warp])
                    : 0u;
                const bool slab_has_out = (cur_smeta & BF11_V5_META_OUTLIER_BIT) != 0u;
                const uint32_t cur_smem = (buf == 0) ? smem_base_0 : smem_base_1;

                if (warp_active) {
                    if constexpr (USE_PACC) {
                        exec_full_slab11_v5_bk128<MT>(
                            cur_slab, kc, cur_smem, static_cast<uint32_t>(lane), lane_row_bytes, lane_k_half,
                            base_sh7, fl, slab_has_out, OUTLIERS, o_cur, o_end, o_rec, o_next_step, p_acc
                        );
                    } else {
                        exec_full_slab11_v5_bk128<MT>(
                            cur_slab, kc, cur_smem, static_cast<uint32_t>(lane), lane_row_bytes, lane_k_half,
                            base_sh7, fl, slab_has_out, OUTLIERS, o_cur, o_end, o_rec, o_next_step, acc
                        );
                    }
                }

                cur_slab = nxt_slab;
                cur_smeta = nxt_smeta;
                cur_raw_fl = nxt_raw_fl;

                if (!crossing_ps) {
                    ++kc;
                } else {
                    if constexpr (USE_PACC) {
                        #pragma unroll
                        for (int mt = 0; mt < MT; ++mt) {
                            acc[mt][0] += p_acc[mt][0];
                            acc[mt][1] += p_acc[mt][1];
                            acc[mt][2] += p_acc[mt][2];
                            acc[mt][3] += p_acc[mt][3];
                            p_acc[mt][0] = 0.0f; p_acc[mt][1] = 0.0f;
                            p_acc[mt][2] = 0.0f; p_acc[mt][3] = 0.0f;
                        }
                    }
                    kc = 0;
                    ++ps;
                    if constexpr (!USE_SPLIT) {
                        if (has_next && warp_active) {
                            warp_uid = static_cast<int64_t>(ps) * w_global + w_row;
                            my_meta_cur = my_meta_nxt;
                            if (ps + 1 < split_packed && lane < n_chunks_per_sp) {
                                const int64_t wuid2 = static_cast<int64_t>(ps + 1) * w_global + w_row;
                                my_meta_nxt = ld_ca_u32(SLAB_META_V5 + wuid2 * n_chunks_per_sp + lane);
                            }
                            o_cur = nxt_o_cur;
                            o_end = nxt_o_end;
                            if (o_cur < o_end) {
                                o_rec = __ldg(OUTLIERS + o_cur);
                                o_next_step = static_cast<uint32_t>(o_rec >> 32);
                            } else {
                                o_rec = 0xFFFFFFFFFFFFFFFFULL;
                                o_next_step = 0xFFFFFFFFu;
                            }
                        }
                    }
                }
            } else {
                // BK == 64 (for BM=128 / MT=8 -> 3 CTAs/SM on 99 KiB SMEM!)
                // Kick off Half-1 X AND next chunk's Slab11+flip BEFORE Half-0 (full 8*MT MMA overlap!)
                load_x_swiz_async<BM, 64, THREADS>(X, sx, m_base, m_eff, k_chunk + 64, sX + BUF_ELEMS, tid);

                Slab11 nxt_slab;
                uint32_t nxt_smeta = 0u;
                uint32_t nxt_raw_fl = 0u;
                int32_t nxt_o_cur = 0, nxt_o_end = 0;
                if (warp_active && has_next) {
                    int64_t nxt_suid;
                    if (!crossing_ps) {
                        nxt_suid = warp_uid * n_chunks_per_sp + (kc + 1);
                        nxt_smeta = __shfl_sync(0xFFFFFFFFu, my_meta_cur, kc + 1);
                    } else {
                        const int64_t nxt_wuid = static_cast<int64_t>(ps + 1) * w_global + w_row;
                        nxt_suid = nxt_wuid * n_chunks_per_sp;
                        nxt_smeta = __shfl_sync(0xFFFFFFFFu, my_meta_nxt, 0);
                        nxt_o_cur = __ldg(O_OFF + nxt_wuid);
                        nxt_o_end = __ldg(O_OFF + nxt_wuid + 1);
                    }
                    nxt_slab = load_slab11_t<CACHE_L2>(SLABS + nxt_suid * SLAB11_BYTES, lane);
                    nxt_raw_fl = issue_flip_load_v5<CACHE_L2>(nxt_smeta, SFLIPS, BMASKS, lane);
                }

                const uint32_t fl = warp_active
                    ? finish_flip_v5<CACHE_L2>(cur_raw_fl, cur_smeta, SFLIPS, lane, s_flip[warp])
                    : 0u;
                const bool slab_has_out = (cur_smeta & BF11_V5_META_OUTLIER_BIT) != 0u;

                if (warp_active) {
                    if constexpr (USE_PACC) {
                        exec_half_slab11_v5_bk64<MT, 0>(
                            cur_slab, kc, smem_base_0, static_cast<uint32_t>(lane), lane_row_bytes, lane_k_half,
                            base_sh7, fl, slab_has_out, OUTLIERS, o_cur, o_end, o_rec, o_next_step, p_acc
                        );
                    } else {
                        exec_half_slab11_v5_bk64<MT, 0>(
                            cur_slab, kc, smem_base_0, static_cast<uint32_t>(lane), lane_row_bytes, lane_k_half,
                            base_sh7, fl, slab_has_out, OUTLIERS, o_cur, o_end, o_rec, o_next_step, acc
                        );
                    }
                }

                cp_async_wait<0>();
                __syncthreads();
                if (has_next) {
                    load_x_swiz_async<BM, 64, THREADS>(X, sx, m_base, m_eff, k_chunk + 128, sX, tid);
                }

                if (warp_active) {
                    if constexpr (USE_PACC) {
                        exec_half_slab11_v5_bk64<MT, 1>(
                            cur_slab, kc, smem_base_1, static_cast<uint32_t>(lane), lane_row_bytes, lane_k_half,
                            base_sh7, fl, slab_has_out, OUTLIERS, o_cur, o_end, o_rec, o_next_step, p_acc
                        );
                    } else {
                        exec_half_slab11_v5_bk64<MT, 1>(
                            cur_slab, kc, smem_base_1, static_cast<uint32_t>(lane), lane_row_bytes, lane_k_half,
                            base_sh7, fl, slab_has_out, OUTLIERS, o_cur, o_end, o_rec, o_next_step, acc
                        );
                    }
                }

                cur_slab = nxt_slab;
                cur_smeta = nxt_smeta;
                cur_raw_fl = nxt_raw_fl;

                if (!crossing_ps) {
                    ++kc;
                } else {
                    if constexpr (USE_PACC) {
                        #pragma unroll
                        for (int mt = 0; mt < MT; ++mt) {
                            acc[mt][0] += p_acc[mt][0];
                            acc[mt][1] += p_acc[mt][1];
                            acc[mt][2] += p_acc[mt][2];
                            acc[mt][3] += p_acc[mt][3];
                            p_acc[mt][0] = 0.0f; p_acc[mt][1] = 0.0f;
                            p_acc[mt][2] = 0.0f; p_acc[mt][3] = 0.0f;
                        }
                    }
                    kc = 0;
                    ++ps;
                    if constexpr (!USE_SPLIT) {
                        if (has_next && warp_active) {
                            warp_uid = static_cast<int64_t>(ps) * w_global + w_row;
                            my_meta_cur = my_meta_nxt;
                            if (ps + 1 < split_packed && lane < n_chunks_per_sp) {
                                const int64_t wuid2 = static_cast<int64_t>(ps + 1) * w_global + w_row;
                                my_meta_nxt = ld_ca_u32(SLAB_META_V5 + wuid2 * n_chunks_per_sp + lane);
                            }
                            o_cur = nxt_o_cur;
                            o_end = nxt_o_end;
                            if (o_cur < o_end) {
                                o_rec = __ldg(OUTLIERS + o_cur);
                                o_next_step = static_cast<uint32_t>(o_rec >> 32);
                            } else {
                                o_rec = 0xFFFFFFFFFFFFFFFFULL;
                                o_next_step = 0xFFFFFFFFu;
                            }
                        }
                    }
                }
            }
        }
    }

    if constexpr (!USE_SPLIT) {
        if constexpr (TRIG) {
            if (tid == 0) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
        const int n0 = n_warp_base + 2 * t;
        const int n1 = n0 + 1;
        const bool n0_act = (n0 < N_active);
        const bool n1_act = (n1 < N_active);
        #pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            const int m0 = m_base + mt * 16 + g;
            const int m1 = m0 + 8;
            const float v00 = n0_act ? acc[mt][0] : 0.0f;
            const float v01 = n1_act ? acc[mt][1] : 0.0f;
            const float v10 = n0_act ? acc[mt][2] : 0.0f;
            const float v11 = n1_act ? acc[mt][3] : 0.0f;
            if (m0 < m_eff && n1 < N) {
                *reinterpret_cast<__nv_bfloat162*>(O + static_cast<int64_t>(m0) * so_m + n0) =
                    __halves2bfloat162(__float2bfloat16_rn(v00 + 0.0f), __float2bfloat16_rn(v01 + 0.0f));
            } else if (m0 < m_eff && n0 < N) {
                O[static_cast<int64_t>(m0) * so_m + n0] = __float2bfloat16_rn(v00 + 0.0f);
            }
            if (m1 < m_eff && n1 < N) {
                *reinterpret_cast<__nv_bfloat162*>(O + static_cast<int64_t>(m1) * so_m + n0) =
                    __halves2bfloat162(__float2bfloat16_rn(v10 + 0.0f), __float2bfloat16_rn(v11 + 0.0f));
            } else if (m1 < m_eff && n0 < N) {
                O[static_cast<int64_t>(m1) * so_m + n0] = __float2bfloat16_rn(v10 + 0.0f);
            }
        }
    } else {
        // L2-Resident Fused Semaphore Split-K Reduction (exact ascending s = 0..SPLIT_CTA-1 order -> 0-ULP!)
        const int m_tile = min(BM, m_eff - m_base);
        const int tile_id = pn * m_blocks + mb;
        const int64_t split_stride = static_cast<int64_t>(BM) * BN;
        const int64_t tile_stride = static_cast<int64_t>(SPLIT_CTA) * split_stride;
        float* p_tile = P + static_cast<int64_t>(tile_id) * tile_stride;
        float* p_my = p_tile + static_cast<int64_t>(ps_start) * split_stride;

        const int nl0 = warp * 8 + 2 * t;
        const int n0_w = pn * BN + nl0;
        const int n1_w = n0_w + 1;
        const bool n0_act = (n0_w < N_active);
        const bool n1_act = (n1_w < N_active);
        #pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            const int lm0 = mt * 16 + g;
            const int lm1 = lm0 + 8;
            const float v00 = n0_act ? acc[mt][0] : 0.0f;
            const float v01 = n1_act ? acc[mt][1] : 0.0f;
            const float v10 = n0_act ? acc[mt][2] : 0.0f;
            const float v11 = n1_act ? acc[mt][3] : 0.0f;
            if (lm0 < m_tile) {
                *reinterpret_cast<float2*>(p_my + lm0 * BN + nl0) = make_float2(v00, v01);
            }
            if (lm1 < m_tile) {
                *reinterpret_cast<float2*>(p_my + lm1 * BN + nl0) = make_float2(v10, v11);
            }
        }

        __syncthreads();
        __shared__ int s_done;
        if (tid == 0) {
            s_done = atomic_add_acq_rel_gpu(&SEM[tile_id], 1);
        }
        __syncthreads();

        if (s_done == SPLIT_CTA - 1) {
            if (tid == 0) st_release_gpu_s32(&SEM[tile_id], 0);
            if constexpr (TRIG) {
                if (tid == 0) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
            constexpr int PAIRS_PER_ROW = BN / 2;
            const int total_pairs = m_tile * PAIRS_PER_ROW;
            const int n_tile_base = pn * BN;
            for (int idx = tid; idx < total_pairs; idx += THREADS) {
                const int lm = idx / PAIRS_PER_ROW;
                const int p = idx - lm * PAIRS_PER_ROW;
                const int nl = p * 2;
                const int n0 = n_tile_base + nl;
                if (n0 >= N) continue;
                const int gm = m_base + lm;
                const int64_t off = static_cast<int64_t>(lm) * BN + nl;
                float s0 = 0.0f, s1 = 0.0f;
                #pragma unroll
                for (int s = 0; s < SPLIT_CTA; ++s) {
                    float2 v = ld_cg_f2(reinterpret_cast<const float2*>(p_tile + static_cast<int64_t>(s) * split_stride + off));
                    s0 += v.x;
                    s1 += v.y;
                }
                if (n0 + 1 < N) {
                    *reinterpret_cast<__nv_bfloat162*>(O + static_cast<int64_t>(gm) * so_m + n0) =
                        __halves2bfloat162(__float2bfloat16_rn(s0 + 0.0f), __float2bfloat16_rn(s1 + 0.0f));
                } else {
                    O[static_cast<int64_t>(gm) * so_m + n0] = __float2bfloat16_rn(s0 + 0.0f);
                }
            }
        }
    }
}

// Companion v5 C=32 SUB_N==2 Kernel (cfg=0, 2; BK=128 XOR-swizzled + ldmatrix.x4 + SLAB_META_V5)
template <int WARPS, int MT, int K_ORDER, bool CACHE_L2, bool PDL, bool TRIG>
__global__ __launch_bounds__(WARPS * 32, 2)
void bf11_flip_v5_c32_sub2_kernel(
    const __nv_bfloat16* __restrict__ X,
    const uint8_t* __restrict__ SLABS,
    const uint32_t* __restrict__ SLAB_META_V5,
    const uint16_t* __restrict__ SFLIPS,
    const uint32_t* __restrict__ BMASKS,
    const uint64_t* __restrict__ OUTLIERS,
    const int32_t* __restrict__ O_OFF,
    __nv_bfloat16* __restrict__ O,
    const int32_t* __restrict__ NLIVE_PTR,
    int M, int N, int N_active, int K, int sx, int KS, int split_packed, int so_m,
    uint32_t base_exp
) {
    constexpr int BN = WARPS * 16;
    constexpr int BM = MT * 16;
    constexpr int THREADS = WARPS * 32;
    constexpr int BK = 128;
    constexpr int BUF_ELEMS = BM * BK;
    constexpr bool USE_PACC = (K_ORDER == 1);

    const int mb = static_cast<int>(blockIdx.x);
    const int pn = static_cast<int>(blockIdx.y);
    const int n_tiles = static_cast<int>(gridDim.y);
    const int m_base = mb * BM;
    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g = lane >> 2;
    const int t = lane & 3;
    const uint32_t lane_row_bytes = static_cast<uint32_t>((lane & 15) * BK * 2);
    const uint32_t lane_k_half = static_cast<uint32_t>(lane >> 4);

    const int n_warp_base = pn * BN + warp * 16;
    const bool warp_active = (n_warp_base < N_active);
    const int n_chunks_per_sp = KS >> 7;
    const int spw = n_chunks_per_sp * 2;
    const int total_chunks = split_packed * n_chunks_per_sp;

    if constexpr (PDL) {
        if (warp_active && total_chunks > 0) {
            const int64_t wuid0 = static_cast<int64_t>(pn) * WARPS + warp;
            const uint8_t* sb0 = SLABS + wuid0 * spw * SLAB11_BYTES;
            if (lane < 11) prefetch_l2(sb0 + lane * 128);
            if (lane < spw) prefetch_l2(SLAB_META_V5 + wuid0 * spw + lane);
        }
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }

    const int m_eff = (NLIVE_PTR != nullptr) ? min(M, *NLIVE_PTR) : M;
    if (m_base >= m_eff) {
        if constexpr (TRIG) {
            if (tid == 0) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
        return;
    }

    __shared__ uint32_t s_flip[WARPS][32];
    extern __shared__ __nv_bfloat16 sX[];
    const uint32_t smem_base_0 = static_cast<uint32_t>(__cvta_generic_to_shared(sX));
    const uint32_t smem_base_1 = static_cast<uint32_t>(__cvta_generic_to_shared(sX + BUF_ELEMS));

    float acc[2][MT][4];
    #pragma unroll
    for (int sub = 0; sub < 2; ++sub) {
        #pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            acc[sub][mt][0] = 0.0f; acc[sub][mt][1] = 0.0f;
            acc[sub][mt][2] = 0.0f; acc[sub][mt][3] = 0.0f;
        }
    }

    [[maybe_unused]] float p_acc[USE_PACC ? 2 : 1][USE_PACC ? MT : 1][4];
    const uint32_t base_sh7 = (base_exp << 7) | (base_exp << 23);

    if (total_chunks > 0) {
        load_x_swiz_async<BM, BK, THREADS>(X, sx, m_base, m_eff, 0, sX, tid);
        int buf = 0;
        for (int ps = 0; ps < split_packed; ++ps) {
            if constexpr (USE_PACC) {
                #pragma unroll
                for (int sub = 0; sub < 2; ++sub) {
                    #pragma unroll
                    for (int mt = 0; mt < MT; ++mt) {
                        p_acc[sub][mt][0] = 0.0f; p_acc[sub][mt][1] = 0.0f;
                        p_acc[sub][mt][2] = 0.0f; p_acc[sub][mt][3] = 0.0f;
                    }
                }
            }
            const int64_t warp_uid = (static_cast<int64_t>(ps) * n_tiles + pn) * WARPS + warp;
            const int64_t suid_base = warp_uid * spw;
            const uint8_t* slab_base = SLABS + suid_base * SLAB11_BYTES;
            uint32_t my_meta = 0u;
            int32_t o_cur = 0, o_end = 0;
            uint64_t o_rec = 0xFFFFFFFFFFFFFFFFULL;
            uint32_t o_next_step = 0xFFFFFFFFu;
            if (warp_active) {
                if (lane < spw) my_meta = ld_ca_u32(SLAB_META_V5 + suid_base + lane);
                o_cur = __ldg(O_OFF + warp_uid);
                o_end = __ldg(O_OFF + warp_uid + 1);
                if (o_cur < o_end) {
                    o_rec = __ldg(OUTLIERS + o_cur);
                    o_next_step = static_cast<uint32_t>(o_rec >> 32);
                }
            }

            for (int kc = 0; kc < n_chunks_per_sp; ++kc) {
                const int k_chunk = ps * KS + kc * 128;
                cp_async_wait<0>();
                __syncthreads();
                const uint32_t cur_smem = (buf == 0) ? smem_base_0 : smem_base_1;
                if (k_chunk + 128 < K) {
                    load_x_swiz_async<BM, BK, THREADS>(X, sx, m_base, m_eff, k_chunk + 128, sX + (buf ^ 1) * BUF_ELEMS, tid);
                }
                if (warp_active) {
                    const int sl0 = kc * 2;
                    const int sl1 = sl0 + 1;
                    const uint32_t sm0 = __shfl_sync(0xFFFFFFFFu, my_meta, sl0);
                    const uint32_t sm1 = __shfl_sync(0xFFFFFFFFu, my_meta, sl1);
                    Slab11 s0_r = load_slab11_t<CACHE_L2>(slab_base + static_cast<int64_t>(sl0) * SLAB11_BYTES, lane);
                    uint32_t rfl0 = issue_flip_load_v5<CACHE_L2>(sm0, SFLIPS, BMASKS, lane);
                    Slab11 s1_r = load_slab11_t<CACHE_L2>(slab_base + static_cast<int64_t>(sl1) * SLAB11_BYTES, lane);
                    uint32_t rfl1 = issue_flip_load_v5<CACHE_L2>(sm1, SFLIPS, BMASKS, lane);

                    const uint32_t fl0 = finish_flip_v5<CACHE_L2>(rfl0, sm0, SFLIPS, lane, s_flip[warp]);
                    const bool ho0 = (sm0 & BF11_V5_META_OUTLIER_BIT) != 0u;
                    if constexpr (USE_PACC) {
                        exec_full_slab11_v5_bk128<MT>(s0_r, sl0, cur_smem, static_cast<uint32_t>(lane), lane_row_bytes, lane_k_half,
                                                      base_sh7, fl0, ho0, OUTLIERS, o_cur, o_end, o_rec, o_next_step, p_acc[0]);
                    } else {
                        exec_full_slab11_v5_bk128<MT>(s0_r, sl0, cur_smem, static_cast<uint32_t>(lane), lane_row_bytes, lane_k_half,
                                                      base_sh7, fl0, ho0, OUTLIERS, o_cur, o_end, o_rec, o_next_step, acc[0]);
                    }

                    const uint32_t fl1 = finish_flip_v5<CACHE_L2>(rfl1, sm1, SFLIPS, lane, s_flip[warp]);
                    const bool ho1 = (sm1 & BF11_V5_META_OUTLIER_BIT) != 0u;
                    if constexpr (USE_PACC) {
                        exec_full_slab11_v5_bk128<MT>(s1_r, sl1, cur_smem, static_cast<uint32_t>(lane), lane_row_bytes, lane_k_half,
                                                      base_sh7, fl1, ho1, OUTLIERS, o_cur, o_end, o_rec, o_next_step, p_acc[1]);
                    } else {
                        exec_full_slab11_v5_bk128<MT>(s1_r, sl1, cur_smem, static_cast<uint32_t>(lane), lane_row_bytes, lane_k_half,
                                                      base_sh7, fl1, ho1, OUTLIERS, o_cur, o_end, o_rec, o_next_step, acc[1]);
                    }
                }
                buf ^= 1;
            }

            if constexpr (USE_PACC) {
                #pragma unroll
                for (int sub = 0; sub < 2; ++sub) {
                    #pragma unroll
                    for (int mt = 0; mt < MT; ++mt) {
                        acc[sub][mt][0] += p_acc[sub][mt][0];
                        acc[sub][mt][1] += p_acc[sub][mt][1];
                        acc[sub][mt][2] += p_acc[sub][mt][2];
                        acc[sub][mt][3] += p_acc[sub][mt][3];
                    }
                }
            }
        }
    }

    if constexpr (TRIG) {
        if (tid == 0) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }

    #pragma unroll
    for (int sub = 0; sub < 2; ++sub) {
        const int n0 = n_warp_base + sub * 8 + 2 * t;
        const int n1 = n0 + 1;
        const bool n0_act = (n0 < N_active);
        const bool n1_act = (n1 < N_active);
        #pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            const int m0 = m_base + mt * 16 + g;
            const int m1 = m0 + 8;
            const float v00 = n0_act ? acc[sub][mt][0] : 0.0f;
            const float v01 = n1_act ? acc[sub][mt][1] : 0.0f;
            const float v10 = n0_act ? acc[sub][mt][2] : 0.0f;
            const float v11 = n1_act ? acc[sub][mt][3] : 0.0f;
            if (m0 < m_eff && n1 < N) {
                *reinterpret_cast<__nv_bfloat162*>(O + static_cast<int64_t>(m0) * so_m + n0) =
                    __halves2bfloat162(__float2bfloat16_rn(v00 + 0.0f), __float2bfloat16_rn(v01 + 0.0f));
            } else if (m0 < m_eff && n0 < N) {
                O[static_cast<int64_t>(m0) * so_m + n0] = __float2bfloat16_rn(v00 + 0.0f);
            }
            if (m1 < m_eff && n1 < N) {
                *reinterpret_cast<__nv_bfloat162*>(O + static_cast<int64_t>(m1) * so_m + n0) =
                    __halves2bfloat162(__float2bfloat16_rn(v10 + 0.0f), __float2bfloat16_rn(v11 + 0.0f));
            } else if (m1 < m_eff && n0 < N) {
                O[static_cast<int64_t>(m1) * so_m + n0] = __float2bfloat16_rn(v10 + 0.0f);
            }
        }
    }
}

// Fused Dual-KV v5 C=32 Kernel (`2x(128,4608)`):
//   - With SPLIT_CTA==9 (k_order==1): launches 144 CTAs = EXACT 3.0 CTAs/SM on GB10 48 SMs!
//     (Fixes 5.1% roofline / ~65 us -> ~10-14 us, 2.5-3.5x faster than 2x torch.mm!)
//   - With SPLIT_CTA==1: flattened cross-split prefetch + BK=128 + ldmatrix.x4 + SLAB_META_V5!
template <
    int WARPS_RUN,
    int MT,
    int K_ORDER,
    int SPLIT_CTA,
    bool CACHE_L2,
    bool PDL,
    bool TRIG
>
__global__ __launch_bounds__(WARPS_RUN * 32, (MT >= 4 ? 3 : (MT == 3 ? (WARPS_RUN == 4 ? 3 : 4) : (WARPS_RUN == 4 ? 4 : 6))))
void bf11_flip_dual_kv_v5_c32_kernel(
    const __nv_bfloat16* __restrict__ X0, const __nv_bfloat16* __restrict__ X1,
    const uint8_t* __restrict__ SLABS0, const uint8_t* __restrict__ SLABS1,
    const uint32_t* __restrict__ SMETA0, const uint32_t* __restrict__ SMETA1,
    const uint16_t* __restrict__ SFLIPS0, const uint16_t* __restrict__ SFLIPS1,
    const uint32_t* __restrict__ BMASKS0, const uint32_t* __restrict__ BMASKS1,
    const uint64_t* __restrict__ OUTLIERS0, const uint64_t* __restrict__ OUTLIERS1,
    const int32_t* __restrict__ O_OFF0, const int32_t* __restrict__ O_OFF1,
    float* __restrict__ P, int32_t* __restrict__ SEM,
    __nv_bfloat16* __restrict__ O0, __nv_bfloat16* __restrict__ O1,
    const int32_t* __restrict__ NLIVE_PTR,
    int M, int N_each, int K, int sx0, int sx1, int KS, int split_packed, int so0, int so1,
    int w_global_each, uint32_t base_exp0, uint32_t base_exp1
) {
    (void)K;
    constexpr int BN = WARPS_RUN * 8;
    constexpr int BM = MT * 16;
    constexpr int THREADS = WARPS_RUN * 32;
    constexpr int BK = 128;
    constexpr int BUF_ELEMS = BM * BK;
    constexpr bool USE_SPLIT = (SPLIT_CTA > 1);
    constexpr bool USE_PACC = (K_ORDER == 1 && !USE_SPLIT);

    const int mb = static_cast<int>(blockIdx.x);
    const int m_blocks = static_cast<int>(gridDim.x);
    const int n_tiles_each = static_cast<int>(gridDim.y) >> 1;
    const int pn_global = static_cast<int>(blockIdx.y);
    const int which = (pn_global >= n_tiles_each) ? 1 : 0;
    const int pn_local = which ? (pn_global - n_tiles_each) : pn_global;
    const int ps_start = USE_SPLIT ? static_cast<int>(blockIdx.z) : 0;
    const int ps_count = USE_SPLIT ? 1 : split_packed;
    const int m_base = mb * BM;

    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g = lane >> 2;
    const int t = lane & 3;
    const uint32_t lane_row_bytes = static_cast<uint32_t>((lane & 15) * BK * 2);
    const uint32_t lane_k_half = static_cast<uint32_t>(lane >> 4);

    const __nv_bfloat16* X = which ? X1 : X0;
    const uint8_t* SLABS = which ? SLABS1 : SLABS0;
    const uint32_t* SLAB_META_V5 = which ? SMETA1 : SMETA0;
    const uint16_t* SFLIPS = which ? SFLIPS1 : SFLIPS0;
    const uint32_t* BMASKS = which ? BMASKS1 : BMASKS0;
    const uint64_t* OUTLIERS = which ? OUTLIERS1 : OUTLIERS0;
    const int32_t* O_OFF = which ? O_OFF1 : O_OFF0;
    __nv_bfloat16* O = which ? O1 : O0;
    const int sx = which ? sx1 : sx0;
    const int so_m = which ? so1 : so0;
    const uint32_t base_exp = which ? base_exp1 : base_exp0;

    const int w_row = pn_local * WARPS_RUN + warp;
    const int n_warp_base = w_row * 8;
    const bool warp_active = (w_row < w_global_each) && (n_warp_base < N_each);
    const int n_chunks_per_sp = KS >> 7;
    const int total_chunks = ps_count * n_chunks_per_sp;

    int64_t warp_uid = static_cast<int64_t>(ps_start) * w_global_each + w_row;
    Slab11 cur_slab;
    uint32_t my_meta_cur = 0u;
    uint32_t my_meta_nxt = 0u;
    uint32_t cur_smeta = 0u;
    uint32_t cur_raw_fl = 0u;
    int32_t o_cur = 0, o_end = 0;
    uint64_t o_rec = 0xFFFFFFFFFFFFFFFFULL;
    uint32_t o_next_step = 0xFFFFFFFFu;

    auto init_weight_streams = [&]() {
        if (warp_active && total_chunks > 0) {
            const int64_t suid0 = warp_uid * n_chunks_per_sp;
            cur_slab = load_slab11_t<CACHE_L2>(SLABS + suid0 * SLAB11_BYTES, lane);
            if (lane < n_chunks_per_sp) {
                my_meta_cur = ld_ca_u32(SLAB_META_V5 + suid0 + lane);
            }
            if constexpr (!USE_SPLIT) {
                if (ps_start + 1 < split_packed && lane < n_chunks_per_sp) {
                    const int64_t wuid1 = static_cast<int64_t>(ps_start + 1) * w_global_each + w_row;
                    my_meta_nxt = ld_ca_u32(SLAB_META_V5 + wuid1 * n_chunks_per_sp + lane);
                }
            }
            o_cur = __ldg(O_OFF + warp_uid);
            o_end = __ldg(O_OFF + warp_uid + 1);
            cur_smeta = __shfl_sync(0xFFFFFFFFu, my_meta_cur, 0);
            cur_raw_fl = issue_flip_load_v5<CACHE_L2>(cur_smeta, SFLIPS, BMASKS, lane);
            if (o_cur < o_end) {
                o_rec = __ldg(OUTLIERS + o_cur);
                o_next_step = static_cast<uint32_t>(o_rec >> 32);
            }
        }
    };

    if constexpr (PDL) {
        init_weight_streams();
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }

    const int m_eff = (NLIVE_PTR != nullptr) ? min(M, *NLIVE_PTR) : M;
    if (m_base >= m_eff) {
        if constexpr (TRIG) {
            if ((!USE_SPLIT || ps_start == SPLIT_CTA - 1) && tid == 0) {
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
        }
        return;
    }

    __shared__ uint32_t s_flip[WARPS_RUN][32];
    extern __shared__ __nv_bfloat16 sX[];
    const uint32_t smem_base_0 = static_cast<uint32_t>(__cvta_generic_to_shared(sX));
    const uint32_t smem_base_1 = static_cast<uint32_t>(__cvta_generic_to_shared(sX + BUF_ELEMS));

    float acc[MT][4];
    #pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
        acc[mt][0] = 0.0f; acc[mt][1] = 0.0f; acc[mt][2] = 0.0f; acc[mt][3] = 0.0f;
    }

    [[maybe_unused]] float p_acc[USE_PACC ? MT : 1][4];
    if constexpr (USE_PACC) {
        #pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            p_acc[mt][0] = 0.0f; p_acc[mt][1] = 0.0f; p_acc[mt][2] = 0.0f; p_acc[mt][3] = 0.0f;
        }
    }

    const uint32_t base_sh7 = (base_exp << 7) | (base_exp << 23);

    if (total_chunks > 0) {
        load_x_swiz_async<BM, BK, THREADS>(X, sx, m_base, m_eff, ps_start * KS, sX, tid);

        if constexpr (!PDL) {
            init_weight_streams();
        }

        int ps = ps_start;
        int kc = 0;
        for (int c = 0; c < total_chunks; ++c) {
            const int k_chunk = ps * KS + kc * 128;
            const int buf = c & 1;
            const bool crossing_ps = (kc + 1 == n_chunks_per_sp);
            const bool has_next = (c + 1 < total_chunks);

            cp_async_wait<0>();
            __syncthreads();
            if (has_next) {
                load_x_swiz_async<BM, BK, THREADS>(
                    X, sx, m_base, m_eff, k_chunk + 128,
                    sX + (buf ^ 1) * BUF_ELEMS, tid
                );
            }

            Slab11 nxt_slab;
            uint32_t nxt_smeta = 0u;
            uint32_t nxt_raw_fl = 0u;
            int32_t nxt_o_cur = 0, nxt_o_end = 0;
            if (warp_active && has_next) {
                int64_t nxt_suid;
                if (!crossing_ps) {
                    nxt_suid = warp_uid * n_chunks_per_sp + (kc + 1);
                    nxt_smeta = __shfl_sync(0xFFFFFFFFu, my_meta_cur, kc + 1);
                } else {
                    const int64_t nxt_wuid = static_cast<int64_t>(ps + 1) * w_global_each + w_row;
                    nxt_suid = nxt_wuid * n_chunks_per_sp;
                    nxt_smeta = __shfl_sync(0xFFFFFFFFu, my_meta_nxt, 0);
                    nxt_o_cur = __ldg(O_OFF + nxt_wuid);
                    nxt_o_end = __ldg(O_OFF + nxt_wuid + 1);
                }
                nxt_slab = load_slab11_t<CACHE_L2>(SLABS + nxt_suid * SLAB11_BYTES, lane);
                nxt_raw_fl = issue_flip_load_v5<CACHE_L2>(nxt_smeta, SFLIPS, BMASKS, lane);
            }

            const uint32_t fl = warp_active
                ? finish_flip_v5<CACHE_L2>(cur_raw_fl, cur_smeta, SFLIPS, lane, s_flip[warp])
                : 0u;
            const bool slab_has_out = (cur_smeta & BF11_V5_META_OUTLIER_BIT) != 0u;
            const uint32_t cur_smem = (buf == 0) ? smem_base_0 : smem_base_1;

            if (warp_active) {
                if constexpr (USE_PACC) {
                    exec_full_slab11_v5_bk128<MT>(
                        cur_slab, kc, cur_smem, static_cast<uint32_t>(lane), lane_row_bytes, lane_k_half,
                        base_sh7, fl, slab_has_out, OUTLIERS, o_cur, o_end, o_rec, o_next_step, p_acc
                    );
                } else {
                    exec_full_slab11_v5_bk128<MT>(
                        cur_slab, kc, cur_smem, static_cast<uint32_t>(lane), lane_row_bytes, lane_k_half,
                        base_sh7, fl, slab_has_out, OUTLIERS, o_cur, o_end, o_rec, o_next_step, acc
                    );
                }
            }

            cur_slab = nxt_slab;
            cur_smeta = nxt_smeta;
            cur_raw_fl = nxt_raw_fl;

            if (!crossing_ps) {
                ++kc;
            } else {
                if constexpr (USE_PACC) {
                    #pragma unroll
                    for (int mt = 0; mt < MT; ++mt) {
                        acc[mt][0] += p_acc[mt][0];
                        acc[mt][1] += p_acc[mt][1];
                        acc[mt][2] += p_acc[mt][2];
                        acc[mt][3] += p_acc[mt][3];
                        p_acc[mt][0] = 0.0f; p_acc[mt][1] = 0.0f;
                        p_acc[mt][2] = 0.0f; p_acc[mt][3] = 0.0f;
                    }
                }
                kc = 0;
                ++ps;
                if constexpr (!USE_SPLIT) {
                    if (has_next && warp_active) {
                        warp_uid = static_cast<int64_t>(ps) * w_global_each + w_row;
                        my_meta_cur = my_meta_nxt;
                        if (ps + 1 < split_packed && lane < n_chunks_per_sp) {
                            const int64_t wuid2 = static_cast<int64_t>(ps + 1) * w_global_each + w_row;
                            my_meta_nxt = ld_ca_u32(SLAB_META_V5 + wuid2 * n_chunks_per_sp + lane);
                        }
                        o_cur = nxt_o_cur;
                        o_end = nxt_o_end;
                        if (o_cur < o_end) {
                            o_rec = __ldg(OUTLIERS + o_cur);
                            o_next_step = static_cast<uint32_t>(o_rec >> 32);
                        } else {
                            o_rec = 0xFFFFFFFFFFFFFFFFULL;
                            o_next_step = 0xFFFFFFFFu;
                        }
                    }
                }
            }
        }
    }

    if constexpr (!USE_SPLIT) {
        if constexpr (TRIG) {
            if (tid == 0) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
        const int n0 = n_warp_base + 2 * t;
        const int n1 = n0 + 1;
        #pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            const int m0 = m_base + mt * 16 + g;
            const int m1 = m0 + 8;
            if (m0 < m_eff && n1 < N_each) {
                *reinterpret_cast<__nv_bfloat162*>(O + static_cast<int64_t>(m0) * so_m + n0) =
                    __halves2bfloat162(__float2bfloat16_rn(acc[mt][0] + 0.0f), __float2bfloat16_rn(acc[mt][1] + 0.0f));
            } else if (m0 < m_eff && n0 < N_each) {
                O[static_cast<int64_t>(m0) * so_m + n0] = __float2bfloat16_rn(acc[mt][0] + 0.0f);
            }
            if (m1 < m_eff && n1 < N_each) {
                *reinterpret_cast<__nv_bfloat162*>(O + static_cast<int64_t>(m1) * so_m + n0) =
                    __halves2bfloat162(__float2bfloat16_rn(acc[mt][2] + 0.0f), __float2bfloat16_rn(acc[mt][3] + 0.0f));
            } else if (m1 < m_eff && n0 < N_each) {
                O[static_cast<int64_t>(m1) * so_m + n0] = __float2bfloat16_rn(acc[mt][2] + 0.0f);
            }
        }
    } else {
        const int m_tile = min(BM, m_eff - m_base);
        const int tile_id = pn_global * m_blocks + mb;
        const int64_t split_stride = static_cast<int64_t>(BM) * BN;
        const int64_t tile_stride = static_cast<int64_t>(SPLIT_CTA) * split_stride;
        float* p_tile = P + static_cast<int64_t>(tile_id) * tile_stride;
        float* p_my = p_tile + static_cast<int64_t>(ps_start) * split_stride;

        const int nl0 = warp * 8 + 2 * t;
        const int n0_w = pn_local * BN + nl0;
        const int n1_w = n0_w + 1;
        const bool n0_act = (n0_w < N_each);
        const bool n1_act = (n1_w < N_each);
        #pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            const int lm0 = mt * 16 + g;
            const int lm1 = lm0 + 8;
            const float v00 = n0_act ? acc[mt][0] : 0.0f;
            const float v01 = n1_act ? acc[mt][1] : 0.0f;
            const float v10 = n0_act ? acc[mt][2] : 0.0f;
            const float v11 = n1_act ? acc[mt][3] : 0.0f;
            if (lm0 < m_tile) {
                *reinterpret_cast<float2*>(p_my + lm0 * BN + nl0) = make_float2(v00, v01);
            }
            if (lm1 < m_tile) {
                *reinterpret_cast<float2*>(p_my + lm1 * BN + nl0) = make_float2(v10, v11);
            }
        }

        __syncthreads();
        __shared__ int s_done;
        if (tid == 0) {
            s_done = atomic_add_acq_rel_gpu(&SEM[tile_id], 1);
        }
        __syncthreads();

        if (s_done == SPLIT_CTA - 1) {
            if (tid == 0) st_release_gpu_s32(&SEM[tile_id], 0);
            if constexpr (TRIG) {
                if (tid == 0) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
            constexpr int PAIRS_PER_ROW = BN / 2;
            const int total_pairs = m_tile * PAIRS_PER_ROW;
            const int n_tile_base = pn_local * BN;
            for (int idx = tid; idx < total_pairs; idx += THREADS) {
                const int lm = idx / PAIRS_PER_ROW;
                const int p = idx - lm * PAIRS_PER_ROW;
                const int nl = p * 2;
                const int n0 = n_tile_base + nl;
                if (n0 >= N_each) continue;
                const int gm = m_base + lm;
                const int64_t off = static_cast<int64_t>(lm) * BN + nl;
                float s0 = 0.0f, s1 = 0.0f;
                #pragma unroll
                for (int s = 0; s < SPLIT_CTA; ++s) {
                    float2 v = ld_cg_f2(reinterpret_cast<const float2*>(p_tile + static_cast<int64_t>(s) * split_stride + off));
                    s0 += v.x;
                    s1 += v.y;
                }
                if (n0 + 1 < N_each) {
                    *reinterpret_cast<__nv_bfloat162*>(O + static_cast<int64_t>(gm) * so_m + n0) =
                        __halves2bfloat162(__float2bfloat16_rn(s0 + 0.0f), __float2bfloat16_rn(s1 + 0.0f));
                } else {
                    O[static_cast<int64_t>(gm) * so_m + n0] = __float2bfloat16_rn(s0 + 0.0f);
                }
            }
        }
    }
}

// =========================================================================
// PART 3: DROP-IN LEGACY ABI KERNELS (0-STACK, 0-SPILL, 100% ABI PARITY)
// =========================================================================

template <int WARPS, int SUB_N, bool M_LE8, int SPLIT, bool PDL, bool TRIG>
__global__ __launch_bounds__(WARPS * 32, 6)
void bf11_flip_kernel(
    const __nv_bfloat16* __restrict__ X,
    const uint8_t* __restrict__ SLABS,
    const uint16_t* __restrict__ FLIPS,
    const uint4* __restrict__ WARP_META,
    const uint64_t* __restrict__ OUTLIERS,
    const int32_t* __restrict__ O_OFF,
    float* __restrict__ P,
    int32_t* __restrict__ SEM,
    __nv_bfloat16* __restrict__ O,
    const int32_t* __restrict__ NLIVE_PTR,
    int M, int N, int N_active, int K, int sx, int KS, int so_m,
    uint32_t base_exp
) {
    constexpr int BN = WARPS * SUB_N * 8;
    constexpr int THREADS = WARPS * 32;
    constexpr bool DIRECT = (SPLIT == 1);
    const int pn = static_cast<int>(blockIdx.x);
    const int ps = static_cast<int>(blockIdx.y);
    const int n_tiles = static_cast<int>(gridDim.x);
    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g = lane >> 2;
    const int t = lane & 3;

    const int n_warp_base = pn * BN + warp * (SUB_N * 8);
    const bool warp_active = (n_warp_base < N_active);

    const int k_lo = ps * KS;
    const int k_hi = (k_lo + KS < K) ? (k_lo + KS) : K;
    const int k_len = k_hi - k_lo;
    const int n_chunks = k_len >> 7;
    const int total_slabs = warp_active ? (n_chunks * SUB_N) : 0;

    const int64_t warp_uid = (static_cast<int64_t>(ps) * n_tiles + pn) * WARPS + warp;
    const int64_t slabs_per_warp = static_cast<int64_t>((KS >> 7) * SUB_N);
    const uint8_t* slab_base = SLABS + warp_uid * slabs_per_warp * SLAB11_BYTES;

    Slab11 cur_slab;
    uint4 meta = make_uint4(0, 0, 0, 0);
    int32_t o_cur = 0, o_end = 0;
    int32_t f_cur = 0;
    uint32_t cur_nf = 0u;
    uint32_t cur_frec = 0u;
    uint64_t o_rec = 0xFFFFFFFFFFFFFFFFULL;
    uint32_t o_next_step = 0xFFFFFFFFu;
    if (total_slabs > 0) {
        cur_slab = load_slab11(slab_base, lane);
        meta = __ldg(WARP_META + warp_uid);
        o_cur = __ldg(O_OFF + warp_uid);
        o_end = __ldg(O_OFF + warp_uid + 1);
        if constexpr (PDL) {
            if (total_slabs > 1 && lane < 11) {
                prefetch_l2(slab_base + SLAB11_BYTES + lane * 128);
            }
            f_cur = static_cast<int32_t>(meta.x);
            cur_nf = meta.y & 0xFFu;
            if (static_cast<uint32_t>(lane) < cur_nf) cur_frec = ld_nc_u16(FLIPS + f_cur + lane);
            f_cur += static_cast<int32_t>(cur_nf);
            if (o_cur < o_end) {
                o_rec = __ldg(OUTLIERS + o_cur);
                o_next_step = static_cast<uint32_t>(o_rec >> 32);
            }
            asm volatile("griddepcontrol.wait;" ::: "memory");
        }
    } else if constexpr (PDL) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }

    const int m_eff = (NLIVE_PTR != nullptr) ? min(M, *NLIVE_PTR) : M;

    __shared__ uint32_t s_flip[WARPS][32];
    extern __shared__ __nv_bfloat16 sX[];

    const int sX_stride = KS + 8;
    const int vec_k = k_len >> 3;
    const int m_load = M_LE8 ? min(m_eff, 8) : min(m_eff, 16);
    const int total_vec = m_load * vec_k;
    for (int idx = tid; idx < total_vec; idx += THREADS) {
        const int m = idx / vec_k;
        const int kv = idx - m * vec_k;
        *reinterpret_cast<uint4*>(sX + m * sX_stride + kv * 8) =
            *reinterpret_cast<const uint4*>(X + static_cast<int64_t>(m) * sx + k_lo + kv * 8);
    }
    __syncthreads();

    if constexpr (!PDL) {
        f_cur = static_cast<int32_t>(meta.x);
        cur_nf = (total_slabs > 0) ? (meta.y & 0xFFu) : 0u;
        if (static_cast<uint32_t>(lane) < cur_nf) cur_frec = ld_nc_u16(FLIPS + f_cur + lane);
        f_cur += static_cast<int32_t>(cur_nf);
        if (o_cur < o_end) {
            o_rec = __ldg(OUTLIERS + o_cur);
            o_next_step = static_cast<uint32_t>(o_rec >> 32);
        }
    }

    float acc[SUB_N][4];
    #pragma unroll
    for (int s = 0; s < SUB_N; ++s) {
        acc[s][0] = 0.0f; acc[s][1] = 0.0f; acc[s][2] = 0.0f; acc[s][3] = 0.0f;
    }

    const int x_off0 = g * sX_stride + 2 * t;
    const int x_off1 = (g + 8) * sX_stride + 2 * t;
    const uint32_t base_sh7 = (base_exp << 7) | (base_exp << 23);

    for (int sl = 0; sl < total_slabs; ++sl) {
        Slab11 nxt_slab;
        uint32_t nxt_nf = 0u;
        uint32_t nxt_frec = 0u;
        if (sl + 1 < total_slabs) {
            nxt_slab = load_slab11(slab_base + static_cast<int64_t>(sl + 1) * SLAB11_BYTES, lane);
            nxt_nf = meta_nf(meta, sl + 1);
            if (static_cast<uint32_t>(lane) < nxt_nf) nxt_frec = ld_nc_u16(FLIPS + f_cur + lane);
            f_cur += static_cast<int32_t>(nxt_nf);
        }

        uint32_t fl = 0u;
        if (cur_nf > 0u) {
            s_flip[warp][lane] = 0u;
            __syncwarp();
            if (static_cast<uint32_t>(lane) < cur_nf) {
                atomicOr(&s_flip[warp][cur_frec >> 5], 1u << (cur_frec & 31u));
            }
            __syncwarp();
            fl = s_flip[warp][lane];
        }

        const bool has_out = (o_cur < o_end);
        if constexpr (SUB_N == 2) {
            const int kc = sl >> 1;
            if ((sl & 1) == 0) {
                exec_slab11_c1_v5<M_LE8>(cur_slab, sl, sX, x_off0, x_off1, kc * 128, base_sh7, static_cast<uint32_t>(lane),
                                         fl, has_out, OUTLIERS, o_cur, o_end, o_rec, o_next_step, acc[0]);
            } else {
                exec_slab11_c1_v5<M_LE8>(cur_slab, sl, sX, x_off0, x_off1, kc * 128, base_sh7, static_cast<uint32_t>(lane),
                                         fl, has_out, OUTLIERS, o_cur, o_end, o_rec, o_next_step, acc[1]);
            }
        } else {
            exec_slab11_c1_v5<M_LE8>(cur_slab, sl, sX, x_off0, x_off1, sl * 128, base_sh7, static_cast<uint32_t>(lane),
                                     fl, has_out, OUTLIERS, o_cur, o_end, o_rec, o_next_step, acc[0]);
        }
        cur_slab = nxt_slab;
        cur_nf = nxt_nf;
        cur_frec = nxt_frec;
    }

    const int m0 = g;
    const int m1 = g + 8;

    if constexpr (DIRECT) {
        if constexpr (TRIG) {
            if (tid == 0) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
        #pragma unroll
        for (int sub = 0; sub < SUB_N; ++sub) {
            const int n0 = n_warp_base + sub * 8 + 2 * t;
            const int n1 = n0 + 1;
            const float v00 = (n0 < N_active) ? acc[sub][0] : 0.0f;
            const float v01 = (n1 < N_active) ? acc[sub][1] : 0.0f;
            const float v10 = (n0 < N_active) ? acc[sub][2] : 0.0f;
            const float v11 = (n1 < N_active) ? acc[sub][3] : 0.0f;
            if (m0 < m_eff && n1 < N) {
                __nv_bfloat162 p0 = __halves2bfloat162(__float2bfloat16_rn(v00 + 0.0f), __float2bfloat16_rn(v01 + 0.0f));
                *reinterpret_cast<__nv_bfloat162*>(O + static_cast<int64_t>(m0) * so_m + n0) = p0;
            } else if (m0 < m_eff && n0 < N) {
                O[static_cast<int64_t>(m0) * so_m + n0] = __float2bfloat16_rn(v00 + 0.0f);
            }
            if constexpr (!M_LE8) {
                if (m1 < m_eff && n1 < N) {
                    __nv_bfloat162 p1 = __halves2bfloat162(__float2bfloat16_rn(v10 + 0.0f), __float2bfloat16_rn(v11 + 0.0f));
                    *reinterpret_cast<__nv_bfloat162*>(O + static_cast<int64_t>(m1) * so_m + n0) = p1;
                } else if (m1 < m_eff && n0 < N) {
                    O[static_cast<int64_t>(m1) * so_m + n0] = __float2bfloat16_rn(v10 + 0.0f);
                }
            }
        }
    } else {
        const int64_t tile_stride = static_cast<int64_t>(SPLIT) * M * BN;
        const int64_t split_stride = static_cast<int64_t>(M) * BN;
        float* p_tile = P + static_cast<int64_t>(pn) * tile_stride;
        float* p_my = p_tile + static_cast<int64_t>(ps) * split_stride;
        #pragma unroll
        for (int sub = 0; sub < SUB_N; ++sub) {
            const int nl0 = warp * (SUB_N * 8) + sub * 8 + 2 * t;
            const int n0 = pn * BN + nl0;
            const int n1 = n0 + 1;
            const float v00 = (n0 < N_active) ? acc[sub][0] : 0.0f;
            const float v01 = (n1 < N_active) ? acc[sub][1] : 0.0f;
            const float v10 = (n0 < N_active) ? acc[sub][2] : 0.0f;
            const float v11 = (n1 < N_active) ? acc[sub][3] : 0.0f;
            if (m0 < m_eff) {
                *reinterpret_cast<float2*>(p_my + m0 * BN + nl0) = make_float2(v00, v01);
            }
            if constexpr (!M_LE8) {
                if (m1 < m_eff) {
                    *reinterpret_cast<float2*>(p_my + m1 * BN + nl0) = make_float2(v10, v11);
                }
            }
        }

        __syncthreads();
        __shared__ int s_done;
        if (tid == 0) {
            s_done = atomic_add_acq_rel_gpu(&SEM[pn], 1);
        }
        __syncthreads();

        if (s_done == SPLIT - 1) {
            if (tid == 0) st_release_gpu_s32(&SEM[pn], 0);
            if constexpr (TRIG) {
                if (tid == 0) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
            constexpr int PAIRS_PER_ROW = BN / 2;
            const int total_pairs = m_eff * PAIRS_PER_ROW;
            const int n_tile_base = pn * BN;
            for (int idx = tid; idx < total_pairs; idx += THREADS) {
                const int m = idx / PAIRS_PER_ROW;
                const int p = idx - m * PAIRS_PER_ROW;
                const int nl = p * 2;
                const int n0 = n_tile_base + nl;
                if (n0 >= N) continue;
                const int64_t off = static_cast<int64_t>(m) * BN + nl;
                float s0 = 0.0f, s1 = 0.0f;
                #pragma unroll
                for (int s = 0; s < SPLIT; ++s) {
                    float2 v = ld_cg_f2(reinterpret_cast<const float2*>(p_tile + static_cast<int64_t>(s) * split_stride + off));
                    s0 += v.x;
                    s1 += v.y;
                }
                if (n0 + 1 < N) {
                    *reinterpret_cast<__nv_bfloat162*>(O + static_cast<int64_t>(m) * so_m + n0) =
                        __halves2bfloat162(__float2bfloat16_rn(s0 + 0.0f), __float2bfloat16_rn(s1 + 0.0f));
                } else {
                    O[static_cast<int64_t>(m) * so_m + n0] = __float2bfloat16_rn(s0 + 0.0f);
                }
            }
        }
    }
}

template <int WARPS, int SUB_N, bool M_LE8, int SPLIT, bool PDL, bool TRIG>
__global__ __launch_bounds__(WARPS * 32, 6)
void bf11_flip_dual_kv_kernel(
    const __nv_bfloat16* __restrict__ X0, const __nv_bfloat16* __restrict__ X1,
    const uint8_t* __restrict__ SLABS0, const uint8_t* __restrict__ SLABS1,
    const uint16_t* __restrict__ FLIPS0, const uint16_t* __restrict__ FLIPS1,
    const uint4* __restrict__ META0, const uint4* __restrict__ META1,
    const uint64_t* __restrict__ OUTLIERS0, const uint64_t* __restrict__ OUTLIERS1,
    const int32_t* __restrict__ O_OFF0, const int32_t* __restrict__ O_OFF1,
    float* __restrict__ P, int32_t* __restrict__ SEM,
    __nv_bfloat16* __restrict__ O0, __nv_bfloat16* __restrict__ O1,
    const int32_t* __restrict__ NLIVE_PTR,
    int M, int N_each, int K, int sx0, int sx1, int KS, int so0, int so1,
    uint32_t base_exp0, uint32_t base_exp1
) {
    constexpr int BN = WARPS * SUB_N * 8;
    constexpr int THREADS = WARPS * 32;
    const int n_tiles_each = static_cast<int>(gridDim.x) >> 1;
    const int which = (static_cast<int>(blockIdx.x) >= n_tiles_each) ? 1 : 0;
    const int pn_local = which ? (static_cast<int>(blockIdx.x) - n_tiles_each) : static_cast<int>(blockIdx.x);
    const int pn_global = static_cast<int>(blockIdx.x);
    const int ps = static_cast<int>(blockIdx.y);
    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g = lane >> 2;
    const int t = lane & 3;

    const __nv_bfloat16* X = which ? X1 : X0;
    const uint8_t* SLABS = which ? SLABS1 : SLABS0;
    const uint16_t* FLIPS = which ? FLIPS1 : FLIPS0;
    const uint4* WARP_META = which ? META1 : META0;
    const uint64_t* OUTLIERS = which ? OUTLIERS1 : OUTLIERS0;
    const int32_t* O_OFF = which ? O_OFF1 : O_OFF0;
    __nv_bfloat16* O = which ? O1 : O0;
    const int sx = which ? sx1 : sx0;
    const int so_m = which ? so1 : so0;
    const uint32_t base_exp = which ? base_exp1 : base_exp0;

    const int k_lo = ps * KS;
    const int k_hi = (k_lo + KS < K) ? (k_lo + KS) : K;
    const int k_len = k_hi - k_lo;
    const int total_slabs = (k_len >> 7) * SUB_N;

    const int64_t warp_uid = (static_cast<int64_t>(ps) * n_tiles_each + pn_local) * WARPS + warp;
    const int64_t slabs_per_warp = static_cast<int64_t>((KS >> 7) * SUB_N);
    const uint8_t* slab_base = SLABS + warp_uid * slabs_per_warp * SLAB11_BYTES;

    Slab11 cur_slab;
    uint4 meta = make_uint4(0, 0, 0, 0);
    int32_t o_cur = 0, o_end = 0;
    int32_t f_cur = 0;
    uint32_t cur_nf = 0u;
    uint32_t cur_frec = 0u;
    uint64_t o_rec = 0xFFFFFFFFFFFFFFFFULL;
    uint32_t o_next_step = 0xFFFFFFFFu;
    if (total_slabs > 0) {
        cur_slab = load_slab11(slab_base, lane);
        meta = __ldg(WARP_META + warp_uid);
        o_cur = __ldg(O_OFF + warp_uid);
        o_end = __ldg(O_OFF + warp_uid + 1);
        if constexpr (PDL) {
            if (total_slabs > 1 && lane < 11) {
                prefetch_l2(slab_base + SLAB11_BYTES + lane * 128);
            }
            f_cur = static_cast<int32_t>(meta.x);
            cur_nf = meta.y & 0xFFu;
            if (static_cast<uint32_t>(lane) < cur_nf) cur_frec = ld_nc_u16(FLIPS + f_cur + lane);
            f_cur += static_cast<int32_t>(cur_nf);
            if (o_cur < o_end) {
                o_rec = __ldg(OUTLIERS + o_cur);
                o_next_step = static_cast<uint32_t>(o_rec >> 32);
            }
            asm volatile("griddepcontrol.wait;" ::: "memory");
        }
    } else if constexpr (PDL) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }

    const int m_eff = (NLIVE_PTR != nullptr) ? min(M, *NLIVE_PTR) : M;

    __shared__ uint32_t s_flip[WARPS][32];
    extern __shared__ __nv_bfloat16 sX[];
    const int sX_stride = KS + 8;
    const int vec_k = k_len >> 3;
    const int m_load = M_LE8 ? min(m_eff, 8) : min(m_eff, 16);
    const int total_vec = m_load * vec_k;
    for (int idx = tid; idx < total_vec; idx += THREADS) {
        const int m = idx / vec_k;
        const int kv = idx - m * vec_k;
        *reinterpret_cast<uint4*>(sX + m * sX_stride + kv * 8) =
            *reinterpret_cast<const uint4*>(X + static_cast<int64_t>(m) * sx + k_lo + kv * 8);
    }
    __syncthreads();

    if constexpr (!PDL) {
        f_cur = static_cast<int32_t>(meta.x);
        cur_nf = (total_slabs > 0) ? (meta.y & 0xFFu) : 0u;
        if (static_cast<uint32_t>(lane) < cur_nf) cur_frec = ld_nc_u16(FLIPS + f_cur + lane);
        f_cur += static_cast<int32_t>(cur_nf);
        if (o_cur < o_end) {
            o_rec = __ldg(OUTLIERS + o_cur);
            o_next_step = static_cast<uint32_t>(o_rec >> 32);
        }
    }

    float acc[SUB_N][4];
    #pragma unroll
    for (int s = 0; s < SUB_N; ++s) {
        acc[s][0] = 0.0f; acc[s][1] = 0.0f; acc[s][2] = 0.0f; acc[s][3] = 0.0f;
    }

    const int x_off0 = g * sX_stride + 2 * t;
    const int x_off1 = (g + 8) * sX_stride + 2 * t;
    const uint32_t base_sh7 = (base_exp << 7) | (base_exp << 23);

    for (int sl = 0; sl < total_slabs; ++sl) {
        Slab11 nxt_slab;
        uint32_t nxt_nf = 0u;
        uint32_t nxt_frec = 0u;
        if (sl + 1 < total_slabs) {
            nxt_slab = load_slab11(slab_base + static_cast<int64_t>(sl + 1) * SLAB11_BYTES, lane);
            nxt_nf = meta_nf(meta, sl + 1);
            if (static_cast<uint32_t>(lane) < nxt_nf) nxt_frec = ld_nc_u16(FLIPS + f_cur + lane);
            f_cur += static_cast<int32_t>(nxt_nf);
        }
        uint32_t fl = 0u;
        if (cur_nf > 0u) {
            s_flip[warp][lane] = 0u;
            __syncwarp();
            if (static_cast<uint32_t>(lane) < cur_nf) atomicOr(&s_flip[warp][cur_frec >> 5], 1u << (cur_frec & 31u));
            __syncwarp();
            fl = s_flip[warp][lane];
        }
        const bool has_out = (o_cur < o_end);
        exec_slab11_c1_v5<M_LE8>(cur_slab, sl, sX, x_off0, x_off1, sl * 128, base_sh7, static_cast<uint32_t>(lane),
                                 fl, has_out, OUTLIERS, o_cur, o_end, o_rec, o_next_step, acc[0]);
        cur_slab = nxt_slab;
        cur_nf = nxt_nf;
        cur_frec = nxt_frec;
    }

    const int m0 = g;
    const int m1 = g + 8;
    const int64_t tile_stride = static_cast<int64_t>(SPLIT) * M * BN;
    const int64_t split_stride = static_cast<int64_t>(M) * BN;
    float* p_tile = P + static_cast<int64_t>(pn_global) * tile_stride;
    float* p_my = p_tile + static_cast<int64_t>(ps) * split_stride;
    #pragma unroll
    for (int sub = 0; sub < SUB_N; ++sub) {
        const int nl0 = warp * (SUB_N * 8) + sub * 8 + 2 * t;
        if (m0 < m_eff) *reinterpret_cast<float2*>(p_my + m0 * BN + nl0) = make_float2(acc[sub][0], acc[sub][1]);
        if constexpr (!M_LE8) {
            if (m1 < m_eff) *reinterpret_cast<float2*>(p_my + m1 * BN + nl0) = make_float2(acc[sub][2], acc[sub][3]);
        }
    }

    __syncthreads();
    __shared__ int s_done;
    if (tid == 0) {
        s_done = atomic_add_acq_rel_gpu(&SEM[pn_global], 1);
    }
    __syncthreads();

    if (s_done == SPLIT - 1) {
        if (tid == 0) st_release_gpu_s32(&SEM[pn_global], 0);
        if constexpr (TRIG) {
            if (tid == 0) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
        constexpr int PAIRS_PER_ROW = BN / 2;
        const int total_pairs = m_eff * PAIRS_PER_ROW;
        const int n_tile_base = pn_local * BN;
        for (int idx = tid; idx < total_pairs; idx += THREADS) {
            const int m = idx / PAIRS_PER_ROW;
            const int p = idx - m * PAIRS_PER_ROW;
            const int nl = p * 2;
            const int n0 = n_tile_base + nl;
            if (n0 >= N_each) continue;
            const int64_t off = static_cast<int64_t>(m) * BN + nl;
            float s0 = 0.0f, s1 = 0.0f;
            #pragma unroll
            for (int s = 0; s < SPLIT; ++s) {
                float2 v = ld_cg_f2(reinterpret_cast<const float2*>(p_tile + static_cast<int64_t>(s) * split_stride + off));
                s0 += v.x;
                s1 += v.y;
            }
            *reinterpret_cast<__nv_bfloat162*>(O + static_cast<int64_t>(m) * so_m + n0) =
                __halves2bfloat162(__float2bfloat16_rn(s0 + 0.0f), __float2bfloat16_rn(s1 + 0.0f));
        }
    }
}

template <int WARPS, int MT, int K_ORDER, bool CACHE_L2, bool PDL, bool TRIG>
__global__ __launch_bounds__(WARPS * 32, (MT >= 8 ? 2 : 4))
void bf11_flip_c32_sub1_kernel(
    const __nv_bfloat16* __restrict__ X,
    const uint8_t* __restrict__ SLABS,
    const uint16_t* __restrict__ FLIPS,
    const uint4* __restrict__ WARP_META,
    const uint64_t* __restrict__ OUTLIERS,
    const int32_t* __restrict__ O_OFF,
    __nv_bfloat16* __restrict__ O,
    const int32_t* __restrict__ NLIVE_PTR,
    int M, int N, int N_active, int K, int sx, int KS, int split_packed, int so_m,
    uint32_t base_exp
) {
    constexpr int BN = WARPS * 8;
    constexpr int BM = MT * 16;
    constexpr int THREADS = WARPS * 32;
    constexpr int SX_STRIDE = 72;
    constexpr int BUF_ELEMS = BM * SX_STRIDE;
    constexpr int VEC_PER_ROW = 8;
    constexpr int TOTAL_VEC = BM * VEC_PER_ROW;

    const int mb = static_cast<int>(blockIdx.x);
    const int pn = static_cast<int>(blockIdx.y);
    const int n_tiles = static_cast<int>(gridDim.y);
    const int m_base = mb * BM;
    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g = lane >> 2;
    const int t = lane & 3;

    const int n_warp_base = pn * BN + warp * 8;
    const bool warp_active = (n_warp_base < N_active);
    const int n_chunks_per_sp = KS >> 7;
    const int64_t slabs_per_warp = static_cast<int64_t>(n_chunks_per_sp);

    if constexpr (PDL) {
        if (warp_active && n_chunks_per_sp > 0) {
            const int64_t wuid0 = static_cast<int64_t>(pn) * WARPS + warp;
            const uint8_t* sb0 = SLABS + wuid0 * slabs_per_warp * SLAB11_BYTES;
            if (lane < 11) prefetch_l2(sb0 + lane * 128);
            if (lane == 11) prefetch_l2(WARP_META + wuid0);
        }
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }

    const int m_eff = (NLIVE_PTR != nullptr) ? min(M, *NLIVE_PTR) : M;
    if (m_base >= m_eff) {
        if constexpr (TRIG) {
            if (tid == 0) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
        return;
    }

    __shared__ uint32_t s_flip[WARPS][32];
    extern __shared__ __nv_bfloat16 sX[];

    auto load_x_half_async = [&](int k_off, __nv_bfloat16* dst) {
        for (int idx = tid; idx < TOTAL_VEC; idx += THREADS) {
            const int lm = idx >> 3;
            const int kv = idx & 7;
            const int gm = m_base + lm;
            __nv_bfloat16* d = dst + lm * SX_STRIDE + (kv << 3);
            if (gm < m_eff) {
                cp_async_ca_16(d, X + static_cast<int64_t>(gm) * sx + k_off + (kv << 3));
            } else {
                *reinterpret_cast<uint4*>(d) = make_uint4(0u, 0u, 0u, 0u);
            }
        }
        cp_async_commit();
    };

    float acc[MT][4];
    #pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
        acc[mt][0] = 0.0f; acc[mt][1] = 0.0f; acc[mt][2] = 0.0f; acc[mt][3] = 0.0f;
    }

    const uint32_t base_sh7 = (base_exp << 7) | (base_exp << 23);
    const int total_k_chunks = split_packed * n_chunks_per_sp;

    if (total_k_chunks > 0) {
        load_x_half_async(0, sX);

        for (int ps = 0; ps < split_packed; ++ps) {
            [[maybe_unused]] float p_acc[(K_ORDER == 1) ? MT : 1][4];
            if constexpr (K_ORDER == 1) {
                #pragma unroll
                for (int mt = 0; mt < MT; ++mt) {
                    p_acc[mt][0] = 0.0f; p_acc[mt][1] = 0.0f; p_acc[mt][2] = 0.0f; p_acc[mt][3] = 0.0f;
                }
            }

            const int64_t warp_uid = (static_cast<int64_t>(ps) * n_tiles + pn) * WARPS + warp;
            const uint8_t* slab_base = SLABS + warp_uid * slabs_per_warp * SLAB11_BYTES;
            Slab11 cur_slab;
            uint4 meta = make_uint4(0, 0, 0, 0);
            int32_t o_cur = 0, o_end = 0;
            int32_t f_cur = 0;
            uint32_t cur_nf = 0u;
            uint32_t cur_frec = 0u;
            uint64_t o_rec = 0xFFFFFFFFFFFFFFFFULL;
            uint32_t o_next_step = 0xFFFFFFFFu;

            if (warp_active) {
                cur_slab = load_slab11_t<CACHE_L2>(slab_base, lane);
                meta = __ldg(WARP_META + warp_uid);
                o_cur = __ldg(O_OFF + warp_uid);
                o_end = __ldg(O_OFF + warp_uid + 1);
                f_cur = static_cast<int32_t>(meta.x);
                cur_nf = meta.y & 0xFFu;
                if (static_cast<uint32_t>(lane) < cur_nf) cur_frec = ld_nc_u16(FLIPS + f_cur + lane);
                f_cur += static_cast<int32_t>(cur_nf);
                if (o_cur < o_end) {
                    o_rec = __ldg(OUTLIERS + o_cur);
                    o_next_step = static_cast<uint32_t>(o_rec >> 32);
                }
            }

            for (int kc = 0; kc < n_chunks_per_sp; ++kc) {
                const int k_chunk = ps * KS + kc * 128;
                cp_async_wait<0>();
                __syncthreads();
                load_x_half_async(k_chunk + 64, sX + BUF_ELEMS);

                uint32_t fl = 0u;
                if (warp_active && cur_nf > 0u) {
                    s_flip[warp][lane] = 0u;
                    __syncwarp();
                    if (static_cast<uint32_t>(lane) < cur_nf) {
                        atomicOr(&s_flip[warp][cur_frec >> 5], 1u << (cur_frec & 31u));
                    }
                    __syncwarp();
                    fl = s_flip[warp][lane];
                }

                if (warp_active) {
                    if constexpr (K_ORDER == 0) {
                        exec_legacy_half_slab11_mt<MT, 0>(cur_slab, kc, sX, g, t, base_sh7, static_cast<uint32_t>(lane), fl,
                                                          OUTLIERS, o_cur, o_end, o_rec, o_next_step, acc);
                    } else {
                        exec_legacy_half_slab11_mt<MT, 0>(cur_slab, kc, sX, g, t, base_sh7, static_cast<uint32_t>(lane), fl,
                                                          OUTLIERS, o_cur, o_end, o_rec, o_next_step, p_acc);
                    }
                }

                cp_async_wait<0>();
                __syncthreads();
                const int k_next = k_chunk + 128;
                if (k_next < K) {
                    load_x_half_async(k_next, sX);
                }

                Slab11 nxt_slab;
                uint32_t nxt_nf = 0u;
                uint32_t nxt_frec = 0u;
                if (warp_active && kc + 1 < n_chunks_per_sp) {
                    nxt_slab = load_slab11_t<CACHE_L2>(slab_base + static_cast<int64_t>(kc + 1) * SLAB11_BYTES, lane);
                    nxt_nf = meta_nf(meta, kc + 1);
                    if (static_cast<uint32_t>(lane) < nxt_nf) nxt_frec = ld_nc_u16(FLIPS + f_cur + lane);
                    f_cur += static_cast<int32_t>(nxt_nf);
                }

                if (warp_active) {
                    if constexpr (K_ORDER == 0) {
                        exec_legacy_half_slab11_mt<MT, 1>(cur_slab, kc, sX + BUF_ELEMS, g, t, base_sh7, static_cast<uint32_t>(lane), fl,
                                                          OUTLIERS, o_cur, o_end, o_rec, o_next_step, acc);
                    } else {
                        exec_legacy_half_slab11_mt<MT, 1>(cur_slab, kc, sX + BUF_ELEMS, g, t, base_sh7, static_cast<uint32_t>(lane), fl,
                                                          OUTLIERS, o_cur, o_end, o_rec, o_next_step, p_acc);
                    }
                }

                cur_slab = nxt_slab;
                cur_nf = nxt_nf;
                cur_frec = nxt_frec;
            }

            if constexpr (K_ORDER == 1) {
                #pragma unroll
                for (int mt = 0; mt < MT; ++mt) {
                    acc[mt][0] += p_acc[mt][0];
                    acc[mt][1] += p_acc[mt][1];
                    acc[mt][2] += p_acc[mt][2];
                    acc[mt][3] += p_acc[mt][3];
                }
            }
        }
    }

    if constexpr (TRIG) {
        if (tid == 0) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }

    const int n0 = n_warp_base + 2 * t;
    const int n1 = n0 + 1;
    const bool n0_act = (n0 < N_active);
    const bool n1_act = (n1 < N_active);
    #pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
        const int m0 = m_base + mt * 16 + g;
        const int m1 = m0 + 8;
        const float v00 = n0_act ? acc[mt][0] : 0.0f;
        const float v01 = n1_act ? acc[mt][1] : 0.0f;
        const float v10 = n0_act ? acc[mt][2] : 0.0f;
        const float v11 = n1_act ? acc[mt][3] : 0.0f;
        if (m0 < m_eff && n1 < N) {
            *reinterpret_cast<__nv_bfloat162*>(O + static_cast<int64_t>(m0) * so_m + n0) =
                __halves2bfloat162(__float2bfloat16_rn(v00 + 0.0f), __float2bfloat16_rn(v01 + 0.0f));
        } else if (m0 < m_eff && n0 < N) {
            O[static_cast<int64_t>(m0) * so_m + n0] = __float2bfloat16_rn(v00 + 0.0f);
        }
        if (m1 < m_eff && n1 < N) {
            *reinterpret_cast<__nv_bfloat162*>(O + static_cast<int64_t>(m1) * so_m + n0) =
                __halves2bfloat162(__float2bfloat16_rn(v10 + 0.0f), __float2bfloat16_rn(v11 + 0.0f));
        } else if (m1 < m_eff && n0 < N) {
            O[static_cast<int64_t>(m1) * so_m + n0] = __float2bfloat16_rn(v10 + 0.0f);
        }
    }
}

template <int WARPS, int MT, int K_ORDER, bool CACHE_L2, bool PDL, bool TRIG>
__global__ __launch_bounds__(WARPS * 32, 2)
void bf11_flip_c32_sub2_kernel(
    const __nv_bfloat16* __restrict__ X,
    const uint8_t* __restrict__ SLABS,
    const uint16_t* __restrict__ FLIPS,
    const uint4* __restrict__ WARP_META,
    const uint64_t* __restrict__ OUTLIERS,
    const int32_t* __restrict__ O_OFF,
    __nv_bfloat16* __restrict__ O,
    const int32_t* __restrict__ NLIVE_PTR,
    int M, int N, int N_active, int K, int sx, int KS, int split_packed, int so_m,
    uint32_t base_exp
) {
    constexpr int BN = WARPS * 16;
    constexpr int BM = MT * 16;
    constexpr int THREADS = WARPS * 32;
    constexpr int SX_STRIDE = 136;
    constexpr int BUF_ELEMS = BM * SX_STRIDE;
    constexpr int VEC_PER_ROW = 16;
    constexpr int TOTAL_VEC = BM * VEC_PER_ROW;

    const int mb = static_cast<int>(blockIdx.x);
    const int pn = static_cast<int>(blockIdx.y);
    const int n_tiles = static_cast<int>(gridDim.y);
    const int m_base = mb * BM;
    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g = lane >> 2;
    const int t = lane & 3;

    const int n_warp_base = pn * BN + warp * 16;
    const bool warp_active = (n_warp_base < N_active);
    const int n_chunks_per_sp = KS >> 7;
    const int64_t slabs_per_warp = static_cast<int64_t>(n_chunks_per_sp * 2);

    if constexpr (PDL) {
        if (warp_active && n_chunks_per_sp > 0) {
            const int64_t wuid0 = static_cast<int64_t>(pn) * WARPS + warp;
            const uint8_t* sb0 = SLABS + wuid0 * slabs_per_warp * SLAB11_BYTES;
            if (lane < 11) prefetch_l2(sb0 + lane * 128);
            if (lane == 11) prefetch_l2(WARP_META + wuid0);
        }
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }

    const int m_eff = (NLIVE_PTR != nullptr) ? min(M, *NLIVE_PTR) : M;
    if (m_base >= m_eff) {
        if constexpr (TRIG) {
            if (tid == 0) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
        return;
    }

    __shared__ uint32_t s_flip[WARPS][32];
    extern __shared__ __nv_bfloat16 sX[];

    auto load_x_chunk_async = [&](int k_off, __nv_bfloat16* dst) {
        for (int idx = tid; idx < TOTAL_VEC; idx += THREADS) {
            const int lm = idx >> 4;
            const int kv = idx & 15;
            const int gm = m_base + lm;
            __nv_bfloat16* d = dst + lm * SX_STRIDE + (kv << 3);
            if (gm < m_eff) {
                cp_async_ca_16(d, X + static_cast<int64_t>(gm) * sx + k_off + (kv << 3));
            } else {
                *reinterpret_cast<uint4*>(d) = make_uint4(0u, 0u, 0u, 0u);
            }
        }
        cp_async_commit();
    };

    float acc[2][MT][4];
    #pragma unroll
    for (int sub = 0; sub < 2; ++sub) {
        #pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            acc[sub][mt][0] = 0.0f; acc[sub][mt][1] = 0.0f; acc[sub][mt][2] = 0.0f; acc[sub][mt][3] = 0.0f;
        }
    }

    const uint32_t base_sh7 = (base_exp << 7) | (base_exp << 23);
    if (split_packed * n_chunks_per_sp > 0) {
        load_x_chunk_async(0, sX);
        int buf = 0;
        for (int ps = 0; ps < split_packed; ++ps) {
            [[maybe_unused]] float p_acc[(K_ORDER == 1) ? 2 : 1][(K_ORDER == 1) ? MT : 1][4];
            if constexpr (K_ORDER == 1) {
                #pragma unroll
                for (int sub = 0; sub < 2; ++sub) {
                    #pragma unroll
                    for (int mt = 0; mt < MT; ++mt) {
                        p_acc[sub][mt][0] = 0.0f; p_acc[sub][mt][1] = 0.0f;
                        p_acc[sub][mt][2] = 0.0f; p_acc[sub][mt][3] = 0.0f;
                    }
                }
            }

            const int64_t warp_uid = (static_cast<int64_t>(ps) * n_tiles + pn) * WARPS + warp;
            const uint8_t* slab_base = SLABS + warp_uid * slabs_per_warp * SLAB11_BYTES;
            uint4 meta = make_uint4(0, 0, 0, 0);
            int32_t o_cur = 0, o_end = 0;
            int32_t f_cur = 0;
            uint64_t o_rec = 0xFFFFFFFFFFFFFFFFULL;
            uint32_t o_next_step = 0xFFFFFFFFu;
            if (warp_active) {
                meta = __ldg(WARP_META + warp_uid);
                o_cur = __ldg(O_OFF + warp_uid);
                o_end = __ldg(O_OFF + warp_uid + 1);
                f_cur = static_cast<int32_t>(meta.x);
                if (o_cur < o_end) {
                    o_rec = __ldg(OUTLIERS + o_cur);
                    o_next_step = static_cast<uint32_t>(o_rec >> 32);
                }
            }

            for (int kc = 0; kc < n_chunks_per_sp; ++kc) {
                const int k_chunk = ps * KS + kc * 128;
                cp_async_wait<0>();
                __syncthreads();
                __nv_bfloat16* cur_sX = sX + buf * BUF_ELEMS;
                if (k_chunk + 128 < K) {
                    load_x_chunk_async(k_chunk + 128, sX + (buf ^ 1) * BUF_ELEMS);
                }
                if (warp_active) {
                    const int sl0 = kc * 2;
                    const int sl1 = sl0 + 1;
                    Slab11 s0_r = load_slab11_t<CACHE_L2>(slab_base + static_cast<int64_t>(sl0) * SLAB11_BYTES, lane);
                    uint32_t nf0 = meta_nf(meta, sl0);
                    uint32_t frec0 = (static_cast<uint32_t>(lane) < nf0) ? ld_nc_u16(FLIPS + f_cur + lane) : 0u;
                    f_cur += static_cast<int32_t>(nf0);
                    uint32_t fl0 = 0u;
                    if (nf0 > 0u) {
                        s_flip[warp][lane] = 0u;
                        __syncwarp();
                        if (static_cast<uint32_t>(lane) < nf0) atomicOr(&s_flip[warp][frec0 >> 5], 1u << (frec0 & 31u));
                        __syncwarp();
                        fl0 = s_flip[warp][lane];
                    }
                    Slab11 s1_r = load_slab11_t<CACHE_L2>(slab_base + static_cast<int64_t>(sl1) * SLAB11_BYTES, lane);
                    uint32_t nf1 = meta_nf(meta, sl1);
                    uint32_t frec1 = (static_cast<uint32_t>(lane) < nf1) ? ld_nc_u16(FLIPS + f_cur + lane) : 0u;
                    f_cur += static_cast<int32_t>(nf1);

                    if constexpr (K_ORDER == 0) {
                        exec_legacy_full_slab11_mt<MT>(s0_r, sl0, cur_sX, g, t, base_sh7, static_cast<uint32_t>(lane), fl0,
                                                       OUTLIERS, o_cur, o_end, o_rec, o_next_step, acc[0]);
                    } else {
                        exec_legacy_full_slab11_mt<MT>(s0_r, sl0, cur_sX, g, t, base_sh7, static_cast<uint32_t>(lane), fl0,
                                                       OUTLIERS, o_cur, o_end, o_rec, o_next_step, p_acc[0]);
                    }

                    uint32_t fl1 = 0u;
                    if (nf1 > 0u) {
                        s_flip[warp][lane] = 0u;
                        __syncwarp();
                        if (static_cast<uint32_t>(lane) < nf1) atomicOr(&s_flip[warp][frec1 >> 5], 1u << (frec1 & 31u));
                        __syncwarp();
                        fl1 = s_flip[warp][lane];
                    }
                    if constexpr (K_ORDER == 0) {
                        exec_legacy_full_slab11_mt<MT>(s1_r, sl1, cur_sX, g, t, base_sh7, static_cast<uint32_t>(lane), fl1,
                                                       OUTLIERS, o_cur, o_end, o_rec, o_next_step, acc[1]);
                    } else {
                        exec_legacy_full_slab11_mt<MT>(s1_r, sl1, cur_sX, g, t, base_sh7, static_cast<uint32_t>(lane), fl1,
                                                       OUTLIERS, o_cur, o_end, o_rec, o_next_step, p_acc[1]);
                    }
                }
                buf ^= 1;
            }

            if constexpr (K_ORDER == 1) {
                #pragma unroll
                for (int sub = 0; sub < 2; ++sub) {
                    #pragma unroll
                    for (int mt = 0; mt < MT; ++mt) {
                        acc[sub][mt][0] += p_acc[sub][mt][0];
                        acc[sub][mt][1] += p_acc[sub][mt][1];
                        acc[sub][mt][2] += p_acc[sub][mt][2];
                        acc[sub][mt][3] += p_acc[sub][mt][3];
                    }
                }
            }
        }
    }

    if constexpr (TRIG) {
        if (tid == 0) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }

    #pragma unroll
    for (int sub = 0; sub < 2; ++sub) {
        const int n0 = n_warp_base + sub * 8 + 2 * t;
        const int n1 = n0 + 1;
        const bool n0_act = (n0 < N_active);
        const bool n1_act = (n1 < N_active);
        #pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            const int m0 = m_base + mt * 16 + g;
            const int m1 = m0 + 8;
            const float v00 = n0_act ? acc[sub][mt][0] : 0.0f;
            const float v01 = n1_act ? acc[sub][mt][1] : 0.0f;
            const float v10 = n0_act ? acc[sub][mt][2] : 0.0f;
            const float v11 = n1_act ? acc[sub][mt][3] : 0.0f;
            if (m0 < m_eff && n1 < N) {
                *reinterpret_cast<__nv_bfloat162*>(O + static_cast<int64_t>(m0) * so_m + n0) =
                    __halves2bfloat162(__float2bfloat16_rn(v00 + 0.0f), __float2bfloat16_rn(v01 + 0.0f));
            } else if (m0 < m_eff && n0 < N) {
                O[static_cast<int64_t>(m0) * so_m + n0] = __float2bfloat16_rn(v00 + 0.0f);
            }
            if (m1 < m_eff && n1 < N) {
                *reinterpret_cast<__nv_bfloat162*>(O + static_cast<int64_t>(m1) * so_m + n0) =
                    __halves2bfloat162(__float2bfloat16_rn(v10 + 0.0f), __float2bfloat16_rn(v11 + 0.0f));
            } else if (m1 < m_eff && n0 < N) {
                O[static_cast<int64_t>(m1) * so_m + n0] = __float2bfloat16_rn(v10 + 0.0f);
            }
        }
    }
}

template <int WARPS, int MT, int K_ORDER, bool CACHE_L2, bool PDL, bool TRIG>
__global__ __launch_bounds__(WARPS * 32, 4)
void bf11_flip_dual_kv_c32_kernel(
    const __nv_bfloat16* __restrict__ X0, const __nv_bfloat16* __restrict__ X1,
    const uint8_t* __restrict__ SLABS0, const uint8_t* __restrict__ SLABS1,
    const uint16_t* __restrict__ FLIPS0, const uint16_t* __restrict__ FLIPS1,
    const uint4* __restrict__ META0, const uint4* __restrict__ META1,
    const uint64_t* __restrict__ OUTLIERS0, const uint64_t* __restrict__ OUTLIERS1,
    const int32_t* __restrict__ O_OFF0, const int32_t* __restrict__ O_OFF1,
    __nv_bfloat16* __restrict__ O0, __nv_bfloat16* __restrict__ O1,
    const int32_t* __restrict__ NLIVE_PTR,
    int M, int N_each, int K, int sx0, int sx1, int KS, int split_packed, int so0, int so1,
    uint32_t base_exp0, uint32_t base_exp1
) {
    constexpr int BN = WARPS * 8;
    constexpr int BM = MT * 16;
    constexpr int THREADS = WARPS * 32;
    constexpr int SX_STRIDE = 72;
    constexpr int BUF_ELEMS = BM * SX_STRIDE;
    constexpr int VEC_PER_ROW = 8;
    constexpr int TOTAL_VEC = BM * VEC_PER_ROW;

    const int mb = static_cast<int>(blockIdx.x);
    const int n_tiles_each = static_cast<int>(gridDim.y) >> 1;
    const int which = (static_cast<int>(blockIdx.y) >= n_tiles_each) ? 1 : 0;
    const int pn_local = which ? (static_cast<int>(blockIdx.y) - n_tiles_each) : static_cast<int>(blockIdx.y);
    const int m_base = mb * BM;
    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g = lane >> 2;
    const int t = lane & 3;

    const __nv_bfloat16* X = which ? X1 : X0;
    const uint8_t* SLABS = which ? SLABS1 : SLABS0;
    const uint16_t* FLIPS = which ? FLIPS1 : FLIPS0;
    const uint4* WARP_META = which ? META1 : META0;
    const uint64_t* OUTLIERS = which ? OUTLIERS1 : OUTLIERS0;
    const int32_t* O_OFF = which ? O_OFF1 : O_OFF0;
    __nv_bfloat16* O = which ? O1 : O0;
    const int sx = which ? sx1 : sx0;
    const int so_m = which ? so1 : so0;
    const uint32_t base_exp = which ? base_exp1 : base_exp0;

    const int n_chunks_per_sp = KS >> 7;
    const int64_t slabs_per_warp = static_cast<int64_t>(n_chunks_per_sp);

    if constexpr (PDL) {
        if (n_chunks_per_sp > 0) {
            const int64_t wuid0 = static_cast<int64_t>(pn_local) * WARPS + warp;
            const uint8_t* sb0 = SLABS + wuid0 * slabs_per_warp * SLAB11_BYTES;
            if (lane < 11) prefetch_l2(sb0 + lane * 128);
            if (lane == 11) prefetch_l2(WARP_META + wuid0);
        }
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }

    const int m_eff = (NLIVE_PTR != nullptr) ? min(M, *NLIVE_PTR) : M;
    if (m_base >= m_eff) {
        if constexpr (TRIG) {
            if (tid == 0) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
        return;
    }

    __shared__ uint32_t s_flip[WARPS][32];
    extern __shared__ __nv_bfloat16 sX[];

    auto load_x_half_async = [&](int k_off, __nv_bfloat16* dst) {
        for (int idx = tid; idx < TOTAL_VEC; idx += THREADS) {
            const int lm = idx >> 3;
            const int kv = idx & 7;
            const int gm = m_base + lm;
            __nv_bfloat16* d = dst + lm * SX_STRIDE + (kv << 3);
            if (gm < m_eff) {
                cp_async_ca_16(d, X + static_cast<int64_t>(gm) * sx + k_off + (kv << 3));
            } else {
                *reinterpret_cast<uint4*>(d) = make_uint4(0u, 0u, 0u, 0u);
            }
        }
        cp_async_commit();
    };

    float acc[MT][4];
    #pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
        acc[mt][0] = 0.0f; acc[mt][1] = 0.0f; acc[mt][2] = 0.0f; acc[mt][3] = 0.0f;
    }

    const uint32_t base_sh7 = (base_exp << 7) | (base_exp << 23);
    if (split_packed * n_chunks_per_sp > 0) {
        load_x_half_async(0, sX);
        for (int ps = 0; ps < split_packed; ++ps) {
            [[maybe_unused]] float p_acc[(K_ORDER == 1) ? MT : 1][4];
            if constexpr (K_ORDER == 1) {
                #pragma unroll
                for (int mt = 0; mt < MT; ++mt) {
                    p_acc[mt][0] = 0.0f; p_acc[mt][1] = 0.0f; p_acc[mt][2] = 0.0f; p_acc[mt][3] = 0.0f;
                }
            }
            const int64_t warp_uid = (static_cast<int64_t>(ps) * n_tiles_each + pn_local) * WARPS + warp;
            const uint8_t* slab_base = SLABS + warp_uid * slabs_per_warp * SLAB11_BYTES;
            Slab11 cur_slab = load_slab11_t<CACHE_L2>(slab_base, lane);
            uint4 meta = __ldg(WARP_META + warp_uid);
            int32_t o_cur = __ldg(O_OFF + warp_uid);
            int32_t o_end = __ldg(O_OFF + warp_uid + 1);
            int32_t f_cur = static_cast<int32_t>(meta.x);
            uint32_t cur_nf = meta.y & 0xFFu;
            uint32_t cur_frec = (static_cast<uint32_t>(lane) < cur_nf) ? ld_nc_u16(FLIPS + f_cur + lane) : 0u;
            f_cur += static_cast<int32_t>(cur_nf);
            uint64_t o_rec = 0xFFFFFFFFFFFFFFFFULL;
            uint32_t o_next_step = 0xFFFFFFFFu;
            if (o_cur < o_end) {
                o_rec = __ldg(OUTLIERS + o_cur);
                o_next_step = static_cast<uint32_t>(o_rec >> 32);
            }

            for (int kc = 0; kc < n_chunks_per_sp; ++kc) {
                const int k_chunk = ps * KS + kc * 128;
                cp_async_wait<0>();
                __syncthreads();
                load_x_half_async(k_chunk + 64, sX + BUF_ELEMS);

                uint32_t fl = 0u;
                if (cur_nf > 0u) {
                    s_flip[warp][lane] = 0u;
                    __syncwarp();
                    if (static_cast<uint32_t>(lane) < cur_nf) atomicOr(&s_flip[warp][cur_frec >> 5], 1u << (cur_frec & 31u));
                    __syncwarp();
                    fl = s_flip[warp][lane];
                }

                if constexpr (K_ORDER == 0) {
                    exec_legacy_half_slab11_mt<MT, 0>(cur_slab, kc, sX, g, t, base_sh7, static_cast<uint32_t>(lane), fl,
                                                      OUTLIERS, o_cur, o_end, o_rec, o_next_step, acc);
                } else {
                    exec_legacy_half_slab11_mt<MT, 0>(cur_slab, kc, sX, g, t, base_sh7, static_cast<uint32_t>(lane), fl,
                                                      OUTLIERS, o_cur, o_end, o_rec, o_next_step, p_acc);
                }

                cp_async_wait<0>();
                __syncthreads();
                const int k_next = k_chunk + 128;
                if (k_next < K) {
                    load_x_half_async(k_next, sX);
                }

                Slab11 nxt_slab;
                uint32_t nxt_nf = 0u;
                uint32_t nxt_frec = 0u;
                if (kc + 1 < n_chunks_per_sp) {
                    nxt_slab = load_slab11_t<CACHE_L2>(slab_base + static_cast<int64_t>(kc + 1) * SLAB11_BYTES, lane);
                    nxt_nf = meta_nf(meta, kc + 1);
                    if (static_cast<uint32_t>(lane) < nxt_nf) nxt_frec = ld_nc_u16(FLIPS + f_cur + lane);
                    f_cur += static_cast<int32_t>(nxt_nf);
                }

                if constexpr (K_ORDER == 0) {
                    exec_legacy_half_slab11_mt<MT, 1>(cur_slab, kc, sX + BUF_ELEMS, g, t, base_sh7, static_cast<uint32_t>(lane), fl,
                                                      OUTLIERS, o_cur, o_end, o_rec, o_next_step, acc);
                } else {
                    exec_legacy_half_slab11_mt<MT, 1>(cur_slab, kc, sX + BUF_ELEMS, g, t, base_sh7, static_cast<uint32_t>(lane), fl,
                                                      OUTLIERS, o_cur, o_end, o_rec, o_next_step, p_acc);
                }

                cur_slab = nxt_slab;
                cur_nf = nxt_nf;
                cur_frec = nxt_frec;
            }
            if constexpr (K_ORDER == 1) {
                #pragma unroll
                for (int mt = 0; mt < MT; ++mt) {
                    acc[mt][0] += p_acc[mt][0];
                    acc[mt][1] += p_acc[mt][1];
                    acc[mt][2] += p_acc[mt][2];
                    acc[mt][3] += p_acc[mt][3];
                }
            }
        }
    }

    if constexpr (TRIG) {
        if (tid == 0) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }

    const int n0 = pn_local * BN + warp * 8 + 2 * t;
    if (n0 + 1 < N_each) {
        #pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            const int m0 = m_base + mt * 16 + g;
            const int m1 = m0 + 8;
            if (m0 < m_eff) {
                *reinterpret_cast<__nv_bfloat162*>(O + static_cast<int64_t>(m0) * so_m + n0) =
                    __halves2bfloat162(__float2bfloat16_rn(acc[mt][0] + 0.0f), __float2bfloat16_rn(acc[mt][1] + 0.0f));
            }
            if (m1 < m_eff) {
                *reinterpret_cast<__nv_bfloat162*>(O + static_cast<int64_t>(m1) * so_m + n0) =
                    __halves2bfloat162(__float2bfloat16_rn(acc[mt][2] + 0.0f), __float2bfloat16_rn(acc[mt][3] + 0.0f));
            }
        }
    } else if (n0 < N_each) {
        #pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            const int m0 = m_base + mt * 16 + g;
            const int m1 = m0 + 8;
            if (m0 < m_eff) O[static_cast<int64_t>(m0) * so_m + n0] = __float2bfloat16_rn(acc[mt][0] + 0.0f);
            if (m1 < m_eff) O[static_cast<int64_t>(m1) * so_m + n0] = __float2bfloat16_rn(acc[mt][2] + 0.0f);
        }
    }
}

// =========================================================================
// Dispatchers
// =========================================================================

template <typename KernelT, typename... Args>
static inline cudaError_t launch_maybe_pdl(
    KernelT kernel, dim3 grid, dim3 block, size_t smem, cudaStream_t stream, bool pdl, Args... args
) {
    if (!pdl) {
        kernel<<<grid, block, smem, stream>>>(args...);
        return cudaGetLastError();
    }
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = smem;
    cfg.stream = stream;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    return cudaLaunchKernelEx(&cfg, kernel, args...);
}

template <int W_V, int SUB_V, int SP_V>
static cudaError_t dispatch_bf11_flip_legacy(
    const void* X, const void* SLABS, const void* FLIPS, const void* WARP_META,
    const void* OUTLIERS, const void* O_OFF,
    void* P, void* SEM, void* O, const void* NLIVE_PTR,
    int M, int N, int N_active, int K, int sx, int KS, int so_m,
    uint32_t base_exp, bool pdl, bool trig, cudaStream_t stream
) {
    constexpr int BN = W_V * SUB_V * 8;
    const int n_tiles = (N + BN - 1) / BN;
    dim3 grid(n_tiles, SP_V, 1);
    dim3 block(W_V * 32, 1, 1);
    const bool m_le8 = (M <= 8);
    const size_t smem = static_cast<size_t>(m_le8 ? 8 : 16) * (KS + 8) * sizeof(__nv_bfloat16);
    #define L_LEG(ML, PD, TR) launch_maybe_pdl(bf11_flip_kernel<W_V, SUB_V, ML, SP_V, PD, TR>, grid, block, smem, stream, PD, \
        static_cast<const __nv_bfloat16*>(X), static_cast<const uint8_t*>(SLABS), static_cast<const uint16_t*>(FLIPS), \
        static_cast<const uint4*>(WARP_META), static_cast<const uint64_t*>(OUTLIERS), static_cast<const int32_t*>(O_OFF), \
        static_cast<float*>(P), static_cast<int32_t*>(SEM), static_cast<__nv_bfloat16*>(O), static_cast<const int32_t*>(NLIVE_PTR), \
        M, N, N_active, K, sx, KS, so_m, base_exp)
    if (m_le8) {
        if (pdl) return trig ? L_LEG(true, true, true) : L_LEG(true, true, false);
        return trig ? L_LEG(true, false, true) : L_LEG(true, false, false);
    } else {
        if (pdl) return trig ? L_LEG(false, true, true) : L_LEG(false, true, false);
        return trig ? L_LEG(false, false, true) : L_LEG(false, false, false);
    }
    #undef L_LEG
}

template <int W_V, int MT_V, int KO_V>
static cudaError_t dispatch_bf11_c32_sub1_legacy(
    const void* X, const void* SLABS, const void* FLIPS, const void* WARP_META,
    const void* OUTLIERS, const void* O_OFF,
    void* O, const void* NLIVE_PTR,
    int M, int N, int N_active, int K, int sx, int KS, int split_packed, int so_m,
    uint32_t base_exp, bool pdl, bool trig, cudaStream_t stream
) {
    constexpr int BN = W_V * 8;
    constexpr int BM = MT_V * 16;
    const int n_tiles = (N + BN - 1) / BN;
    const int m_blocks = (M + BM - 1) / BM;
    const bool cache_l2 = (m_blocks > 1);
    dim3 grid(m_blocks, n_tiles, 1);
    dim3 block(W_V * 32, 1, 1);
    const size_t smem = static_cast<size_t>(2) * BM * 72 * sizeof(__nv_bfloat16);
    #define LC1_LEG(CA, PD, TR) launch_maybe_pdl(bf11_flip_c32_sub1_kernel<W_V, MT_V, KO_V, CA, PD, TR>, \
        grid, block, smem, stream, PD, \
        static_cast<const __nv_bfloat16*>(X), static_cast<const uint8_t*>(SLABS), static_cast<const uint16_t*>(FLIPS), \
        static_cast<const uint4*>(WARP_META), static_cast<const uint64_t*>(OUTLIERS), static_cast<const int32_t*>(O_OFF), \
        static_cast<__nv_bfloat16*>(O), static_cast<const int32_t*>(NLIVE_PTR), \
        M, N, N_active, K, sx, KS, split_packed, so_m, base_exp)
    if (cache_l2) {
        if (pdl) return trig ? LC1_LEG(true, true, true) : LC1_LEG(true, true, false);
        return trig ? LC1_LEG(true, false, true) : LC1_LEG(true, false, false);
    } else {
        if (pdl) return trig ? LC1_LEG(false, true, true) : LC1_LEG(false, true, false);
        return trig ? LC1_LEG(false, false, true) : LC1_LEG(false, false, false);
    }
    #undef LC1_LEG
}

template <int W_V, int MT_V, int KO_V>
static cudaError_t dispatch_bf11_c32_sub2_legacy(
    const void* X, const void* SLABS, const void* FLIPS, const void* WARP_META,
    const void* OUTLIERS, const void* O_OFF,
    void* O, const void* NLIVE_PTR,
    int M, int N, int N_active, int K, int sx, int KS, int split_packed, int so_m,
    uint32_t base_exp, bool pdl, bool trig, cudaStream_t stream
) {
    constexpr int BN = W_V * 16;
    constexpr int BM = MT_V * 16;
    const int n_tiles = (N + BN - 1) / BN;
    const int m_blocks = (M + BM - 1) / BM;
    const bool cache_l2 = (m_blocks > 1);
    dim3 grid(m_blocks, n_tiles, 1);
    dim3 block(W_V * 32, 1, 1);
    const size_t smem = static_cast<size_t>(2) * BM * 136 * sizeof(__nv_bfloat16);
    #define LC2_LEG(CA, PD, TR) launch_maybe_pdl(bf11_flip_c32_sub2_kernel<W_V, MT_V, KO_V, CA, PD, TR>, \
        grid, block, smem, stream, PD, \
        static_cast<const __nv_bfloat16*>(X), static_cast<const uint8_t*>(SLABS), static_cast<const uint16_t*>(FLIPS), \
        static_cast<const uint4*>(WARP_META), static_cast<const uint64_t*>(OUTLIERS), static_cast<const int32_t*>(O_OFF), \
        static_cast<__nv_bfloat16*>(O), static_cast<const int32_t*>(NLIVE_PTR), \
        M, N, N_active, K, sx, KS, split_packed, so_m, base_exp)
    if (cache_l2) {
        if (pdl) return trig ? LC2_LEG(true, true, true) : LC2_LEG(true, true, false);
        return trig ? LC2_LEG(true, false, true) : LC2_LEG(true, false, false);
    } else {
        if (pdl) return trig ? LC2_LEG(false, true, true) : LC2_LEG(false, true, false);
        return trig ? LC2_LEG(false, false, true) : LC2_LEG(false, false, false);
    }
    #undef LC2_LEG
}

template <int W_RUN, int SUB_V, int SP_V>
static cudaError_t dispatch_bf11_v5_c1(
    const void* X, const void* SLABS, const void* SLAB_META_V5, const void* SFLIPS, const void* BMASKS,
    const void* OUTLIERS, const void* O_OFF,
    void* P, void* SEM, void* O, const void* NLIVE_PTR,
    int M, int N, int N_active, int K, int sx, int KS, int so_m,
    int w_global, uint32_t base_exp, bool pdl, bool trig, cudaStream_t stream
) {
    const int n_tiles_run = (w_global + W_RUN - 1) / W_RUN;
    dim3 grid(n_tiles_run, SP_V, 1);
    dim3 block(W_RUN * 32, 1, 1);
    const bool m_le8 = (M <= 8);
    const size_t smem = static_cast<size_t>(m_le8 ? 8 : 16) * (KS + 8) * sizeof(__nv_bfloat16);
    #define LV5_C1(ML, PD, TR) launch_maybe_pdl(bf11_flip_v5_c1_kernel<W_RUN, SUB_V, ML, SP_V, PD, TR>, \
        grid, block, smem, stream, PD, \
        static_cast<const __nv_bfloat16*>(X), static_cast<const uint8_t*>(SLABS), \
        static_cast<const uint32_t*>(SLAB_META_V5), static_cast<const uint16_t*>(SFLIPS), \
        static_cast<const uint32_t*>(BMASKS), static_cast<const uint64_t*>(OUTLIERS), static_cast<const int32_t*>(O_OFF), \
        static_cast<float*>(P), static_cast<int32_t*>(SEM), static_cast<__nv_bfloat16*>(O), \
        static_cast<const int32_t*>(NLIVE_PTR), \
        M, N, N_active, K, sx, KS, so_m, w_global, base_exp)
    if (m_le8) {
        if (pdl) return trig ? LV5_C1(true, true, true) : LV5_C1(true, true, false);
        return trig ? LV5_C1(true, false, true) : LV5_C1(true, false, false);
    } else {
        if (pdl) return trig ? LV5_C1(false, true, true) : LV5_C1(false, true, false);
        return trig ? LV5_C1(false, false, true) : LV5_C1(false, false, false);
    }
    #undef LV5_C1
}

template <int W_RUN, int MT_V, int BK_V, int KO_V, int SPLIT_V>
static cudaError_t dispatch_bf11_v5_c32_sub1(
    const void* X, const void* SLABS, const void* SLAB_META_V5, const void* SFLIPS, const void* BMASKS,
    const void* OUTLIERS, const void* O_OFF,
    void* P, void* SEM, void* O, const void* NLIVE_PTR,
    int M, int N, int N_active, int K, int sx, int KS, int split_packed, int so_m,
    int w_global, uint32_t base_exp, bool pdl, bool trig, cudaStream_t stream
) {
    constexpr int BM = MT_V * 16;
    const int n_tiles_run = (w_global + W_RUN - 1) / W_RUN;
    const int m_blocks = (M + BM - 1) / BM;
    const bool cache_l2 = (m_blocks > 1);
    dim3 grid(m_blocks, n_tiles_run, SPLIT_V);
    dim3 block(W_RUN * 32, 1, 1);
    const size_t smem = static_cast<size_t>(2) * BM * BK_V * sizeof(__nv_bfloat16);
    #define LV5_S1(CA, PD, TR) launch_maybe_pdl(bf11_flip_v5_c32_sub1_kernel<W_RUN, MT_V, BK_V, KO_V, SPLIT_V, CA, PD, TR>, \
        grid, block, smem, stream, PD, \
        static_cast<const __nv_bfloat16*>(X), static_cast<const uint8_t*>(SLABS), \
        static_cast<const uint32_t*>(SLAB_META_V5), static_cast<const uint16_t*>(SFLIPS), \
        static_cast<const uint32_t*>(BMASKS), static_cast<const uint64_t*>(OUTLIERS), static_cast<const int32_t*>(O_OFF), \
        static_cast<float*>(P), static_cast<int32_t*>(SEM), static_cast<__nv_bfloat16*>(O), \
        static_cast<const int32_t*>(NLIVE_PTR), \
        M, N, N_active, K, sx, KS, split_packed, so_m, w_global, base_exp)
    if (cache_l2) {
        if (pdl) return trig ? LV5_S1(true, true, true) : LV5_S1(true, true, false);
        return trig ? LV5_S1(true, false, true) : LV5_S1(true, false, false);
    } else {
        if (pdl) return trig ? LV5_S1(false, true, true) : LV5_S1(false, true, false);
        return trig ? LV5_S1(false, false, true) : LV5_S1(false, false, false);
    }
    #undef LV5_S1
}

template <int W_V, int MT_V, int KO_V>
static cudaError_t dispatch_bf11_v5_c32_sub2(
    const void* X, const void* SLABS, const void* SLAB_META_V5, const void* SFLIPS, const void* BMASKS,
    const void* OUTLIERS, const void* O_OFF,
    void* O, const void* NLIVE_PTR,
    int M, int N, int N_active, int K, int sx, int KS, int split_packed, int so_m,
    uint32_t base_exp, bool pdl, bool trig, cudaStream_t stream
) {
    constexpr int BN = W_V * 16;
    constexpr int BM = MT_V * 16;
    const int n_tiles = (N + BN - 1) / BN;
    const int m_blocks = (M + BM - 1) / BM;
    const bool cache_l2 = (m_blocks > 1);
    dim3 grid(m_blocks, n_tiles, 1);
    dim3 block(W_V * 32, 1, 1);
    const size_t smem = static_cast<size_t>(2) * BM * 128 * sizeof(__nv_bfloat16);
    #define LV5_S2(CA, PD, TR) launch_maybe_pdl(bf11_flip_v5_c32_sub2_kernel<W_V, MT_V, KO_V, CA, PD, TR>, \
        grid, block, smem, stream, PD, \
        static_cast<const __nv_bfloat16*>(X), static_cast<const uint8_t*>(SLABS), \
        static_cast<const uint32_t*>(SLAB_META_V5), static_cast<const uint16_t*>(SFLIPS), \
        static_cast<const uint32_t*>(BMASKS), static_cast<const uint64_t*>(OUTLIERS), static_cast<const int32_t*>(O_OFF), \
        static_cast<__nv_bfloat16*>(O), static_cast<const int32_t*>(NLIVE_PTR), \
        M, N, N_active, K, sx, KS, split_packed, so_m, base_exp)
    if (cache_l2) {
        if (pdl) return trig ? LV5_S2(true, true, true) : LV5_S2(true, true, false);
        return trig ? LV5_S2(true, false, true) : LV5_S2(true, false, false);
    } else {
        if (pdl) return trig ? LV5_S2(false, true, true) : LV5_S2(false, true, false);
        return trig ? LV5_S2(false, false, true) : LV5_S2(false, false, false);
    }
    #undef LV5_S2
}

template <int W_RUN, int MT_V, int KO_V, int SPLIT_V>
static cudaError_t dispatch_bf11_v5_dual_kv_c32(
    const void* X0, const void* X1, const void* SLABS0, const void* SLABS1,
    const void* SMETA0, const void* SMETA1, const void* SFLIPS0, const void* SFLIPS1,
    const void* BMASKS0, const void* BMASKS1, const void* OUTLIERS0, const void* OUTLIERS1,
    const void* O_OFF0, const void* O_OFF1,
    void* P, void* SEM, void* O0, void* O1, const void* NLIVE_PTR,
    int M, int N_each, int K, int sx0, int sx1, int KS, int split_packed, int so0, int so1,
    int w_global_each, uint32_t base_exp0, uint32_t base_exp1, bool pdl, bool trig, cudaStream_t stream
) {
    constexpr int BM = MT_V * 16;
    const int n_tiles_each_run = (w_global_each + W_RUN - 1) / W_RUN;
    const int m_blocks = (M + BM - 1) / BM;
    const bool cache_l2 = (m_blocks > 1);
    dim3 grid(m_blocks, 2 * n_tiles_each_run, SPLIT_V);
    dim3 block(W_RUN * 32, 1, 1);
    const size_t smem = static_cast<size_t>(2) * BM * 128 * sizeof(__nv_bfloat16);
    #define LV5_DKV(CA, PD, TR) launch_maybe_pdl(bf11_flip_dual_kv_v5_c32_kernel<W_RUN, MT_V, KO_V, SPLIT_V, CA, PD, TR>, \
        grid, block, smem, stream, PD, \
        static_cast<const __nv_bfloat16*>(X0), static_cast<const __nv_bfloat16*>(X1), \
        static_cast<const uint8_t*>(SLABS0), static_cast<const uint8_t*>(SLABS1), \
        static_cast<const uint32_t*>(SMETA0), static_cast<const uint32_t*>(SMETA1), \
        static_cast<const uint16_t*>(SFLIPS0), static_cast<const uint16_t*>(SFLIPS1), \
        static_cast<const uint32_t*>(BMASKS0), static_cast<const uint32_t*>(BMASKS1), \
        static_cast<const uint64_t*>(OUTLIERS0), static_cast<const uint64_t*>(OUTLIERS1), \
        static_cast<const int32_t*>(O_OFF0), static_cast<const int32_t*>(O_OFF1), \
        static_cast<float*>(P), static_cast<int32_t*>(SEM), \
        static_cast<__nv_bfloat16*>(O0), static_cast<__nv_bfloat16*>(O1), static_cast<const int32_t*>(NLIVE_PTR), \
        M, N_each, K, sx0, sx1, KS, split_packed, so0, so1, w_global_each, base_exp0, base_exp1)
    if (cache_l2) {
        if (pdl) return trig ? LV5_DKV(true, true, true) : LV5_DKV(true, true, false);
        return trig ? LV5_DKV(true, false, true) : LV5_DKV(true, false, false);
    } else {
        if (pdl) return trig ? LV5_DKV(false, true, true) : LV5_DKV(false, true, false);
        return trig ? LV5_DKV(false, false, true) : LV5_DKV(false, false, false);
    }
    #undef LV5_DKV
}


// =========================================================================
// PART 3B: WAVE-V9 FUSED GLOBAL `k_dyn` / `v_dyn` ([M, N_dyn<=4]) ENGINE
//   Piggybacks 2 extra CTAs (`k_dyn` on X0, `v_dyn` on X1) directly onto the
//   `bf11_flip_dual_kv` grid (ZERO extra SMEM, ZERO extra streams/events,
//   ZERO register increase on BF11 K/V warps, kills 16 cuBLAS calls/step!).
//   Supports 4 exact Tensor Core accumulation modes for startup 0-ULP matching
//   against SM121a cuBLASLt `nn.functional.linear(x, W_dyn [4, 4608])`:
//     dyn_mode = 0: mma.m16n8k16 + linear ascending split_k sum
//     dyn_mode = 1: mma.m16n8k16 + exact balanced binary-tree split_k sum (pow2 split_k <= 32)
//     dyn_mode = 2: mma.m16n8k8  + linear ascending split_k sum
//     dyn_mode = 3: mma.m16n8k8  + exact balanced binary-tree split_k sum (pow2 split_k <= 32)
// =========================================================================

template <int WARPS_RUN, int MT>
static __device__ __forceinline__ void run_dyn_tap_cta_v9(
    const __nv_bfloat16* __restrict__ X,
    const __nv_bfloat16* __restrict__ W_DYN,
    __nv_bfloat16* __restrict__ O_DYN,
    __nv_bfloat16* s_raw,
    int m_base, int m_eff, int N_dyn, int K, int sx, int so_dyn,
    int dyn_mode, int dyn_split_k,
    int tid, int warp, int lane
) {
    constexpr int BM = MT * 16;
    constexpr int THREADS = WARPS_RUN * 32;
    constexpr int STRIDE = 136;  // 68 uint32 words -> gcd(68,32)=4 -> 0-way SMEM bank conflicts!
    constexpr int MAX_TW = (MT + WARPS_RUN - 1) / WARPS_RUN;  // <= 2 for all dispatched configs!

    const int g = lane >> 2;
    const int t = lane & 3;
    const int use_k8 = (dyn_mode >= 2) ? 1 : 0;
    const int use_tree = (dyn_mode & 1);
    const int spk = (dyn_split_k >= 1) ? dyn_split_k : 1;
    const int ks_dyn = K / spk;
    int tree_depth = 0;
    if (use_tree) {
        int tmp = spk;
        while (tmp > 1) { tmp >>= 1; ++tree_depth; }
    }

    // Repurpose existing dynamic SMEM (`sX`):
    //   sX_dyn[BM][136]  (BM * 272 B)
    //   sW_dyn[4][136]   (1,088 B)
    //   s_tree[6][BM][4] (6 * BM * 16 B)
    // Total: 6,976 B (MT=1) <= 8,320 B; 12,864 B (MT=2) <= 16,384 B; 24,640 B (MT=4) <= 32,768 B!
    __nv_bfloat16* sX_dyn = s_raw;
    __nv_bfloat16* sW_dyn = sX_dyn + BM * STRIDE;
    float* s_tree = reinterpret_cast<float*>(sW_dyn + 4 * STRIDE);

    float acc[MAX_TW][4];
    #pragma unroll
    for (int tw = 0; tw < MAX_TW; ++tw) {
        acc[tw][0] = 0.0f; acc[tw][1] = 0.0f; acc[tw][2] = 0.0f; acc[tw][3] = 0.0f;
    }

    for (int s = 0; s < spk; ++s) {
        float p_acc[MAX_TW][4];
        #pragma unroll
        for (int tw = 0; tw < MAX_TW; ++tw) {
            p_acc[tw][0] = 0.0f; p_acc[tw][1] = 0.0f; p_acc[tw][2] = 0.0f; p_acc[tw][3] = 0.0f;
        }
        const int k_split_base = s * ks_dyn;
        for (int k_sub = 0; k_sub < ks_dyn; k_sub += 128) {
            const int step = (k_sub + 128 <= ks_dyn) ? 128 : (ks_dyn - k_sub);
            const int vecs_per_row = step >> 3;  // step is multiple of 16 -> vecs_per_row is even
            const int k_glob = k_split_base + k_sub;

            // Cooperative coalesced 128-bit loads of X[m_base..m_base+BM-1, k_glob..k_glob+step-1]
            const int x_total_vec = BM * vecs_per_row;
            for (int idx = tid; idx < x_total_vec; idx += THREADS) {
                const int lm = idx / vecs_per_row;
                const int kv = idx - lm * vecs_per_row;
                const int gm = m_base + lm;
                uint4 v = make_uint4(0u, 0u, 0u, 0u);
                if (gm < m_eff) {
                    v = *reinterpret_cast<const uint4*>(X + static_cast<int64_t>(gm) * sx + k_glob + (kv << 3));
                }
                *reinterpret_cast<uint4*>(sX_dyn + lm * STRIDE + (kv << 3)) = v;
            }
            // Cooperative coalesced 128-bit loads of W_DYN[0..N_dyn-1, k_glob..k_glob+step-1]
            const int w_total_vec = 4 * vecs_per_row;
            for (int idx = tid; idx < w_total_vec; idx += THREADS) {
                const int rn = idx / vecs_per_row;
                const int kv = idx - rn * vecs_per_row;
                uint4 v = make_uint4(0u, 0u, 0u, 0u);
                if (rn < N_dyn) {
                    v = ld_ca_u4(reinterpret_cast<const uint4*>(W_DYN + static_cast<int64_t>(rn) * K + k_glob + (kv << 3)));
                }
                *reinterpret_cast<uint4*>(sW_dyn + rn * STRIDE + (kv << 3)) = v;
            }
            __syncthreads();

            if (!use_k8) {
                const int n_steps16 = step >> 4;
                for (int ks16 = 0; ks16 < n_steps16; ++ks16) {
                    const int ko = ks16 << 4;
                    uint32_t b0 = 0u, b1 = 0u;
                    if (g < N_dyn) {
                        b0 = *reinterpret_cast<const uint32_t*>(sW_dyn + g * STRIDE + ko + 2 * t);
                        b1 = *reinterpret_cast<const uint32_t*>(sW_dyn + g * STRIDE + ko + 8 + 2 * t);
                    }
                    #pragma unroll
                    for (int tw = 0; tw < MAX_TW; ++tw) {
                        const int mt = warp + tw * WARPS_RUN;
                        if (mt < MT) {
                            const int r0 = mt * 16 + g;
                            const int r1 = r0 + 8;
                            const uint32_t a0 = *reinterpret_cast<const uint32_t*>(sX_dyn + r0 * STRIDE + ko + 2 * t);
                            const uint32_t a1 = *reinterpret_cast<const uint32_t*>(sX_dyn + r1 * STRIDE + ko + 2 * t);
                            const uint32_t a2 = *reinterpret_cast<const uint32_t*>(sX_dyn + r0 * STRIDE + ko + 8 + 2 * t);
                            const uint32_t a3 = *reinterpret_cast<const uint32_t*>(sX_dyn + r1 * STRIDE + ko + 8 + 2 * t);
                            mma_m16n8k16_bf16(p_acc[tw], a0, a1, a2, a3, b0, b1);
                        }
                    }
                }
            } else {
                const int n_steps8 = step >> 3;
                for (int ks8 = 0; ks8 < n_steps8; ++ks8) {
                    const int ko = ks8 << 3;
                    uint32_t b0 = 0u;
                    if (g < N_dyn) {
                        b0 = *reinterpret_cast<const uint32_t*>(sW_dyn + g * STRIDE + ko + 2 * t);
                    }
                    #pragma unroll
                    for (int tw = 0; tw < MAX_TW; ++tw) {
                        const int mt = warp + tw * WARPS_RUN;
                        if (mt < MT) {
                            const int r0 = mt * 16 + g;
                            const int r1 = r0 + 8;
                            const uint32_t a0 = *reinterpret_cast<const uint32_t*>(sX_dyn + r0 * STRIDE + ko + 2 * t);
                            const uint32_t a1 = *reinterpret_cast<const uint32_t*>(sX_dyn + r1 * STRIDE + ko + 2 * t);
                            mma_m16n8k8_bf16(p_acc[tw], a0, a1, b0);
                        }
                    }
                }
            }
            __syncthreads();
        }

        // Accumulate split-K partial into `acc` (exact linear ascending OR exact balanced binary tree)
        if (!use_tree || spk == 1) {
            #pragma unroll
            for (int tw = 0; tw < MAX_TW; ++tw) {
                acc[tw][0] = __fadd_rn(acc[tw][0], p_acc[tw][0]);
                acc[tw][1] = __fadd_rn(acc[tw][1], p_acc[tw][1]);
                acc[tw][2] = __fadd_rn(acc[tw][2], p_acc[tw][2]);
                acc[tw][3] = __fadd_rn(acc[tw][3], p_acc[tw][3]);
            }
        } else if (t < 2) {
            #pragma unroll
            for (int tw = 0; tw < MAX_TW; ++tw) {
                const int mt = warp + tw * WARPS_RUN;
                if (mt < MT) {
                    float c0 = p_acc[tw][0], c1 = p_acc[tw][1], c2 = p_acc[tw][2], c3 = p_acc[tw][3];
                    const int r0 = mt * 16 + g;
                    const int r1 = r0 + 8;
                    const int col0 = 2 * t;
                    int lvl = 0;
                    for (int mask = s; (mask & 1) != 0; mask >>= 1, ++lvl) {
                        const float* base = s_tree + lvl * (BM * 4);
                        c0 = __fadd_rn(base[r0 * 4 + col0],     c0);
                        c1 = __fadd_rn(base[r0 * 4 + col0 + 1], c1);
                        c2 = __fadd_rn(base[r1 * 4 + col0],     c2);
                        c3 = __fadd_rn(base[r1 * 4 + col0 + 1], c3);
                    }
                    if (lvl == tree_depth) {
                        acc[tw][0] = c0; acc[tw][1] = c1; acc[tw][2] = c2; acc[tw][3] = c3;
                    } else {
                        float* base = s_tree + lvl * (BM * 4);
                        base[r0 * 4 + col0]     = c0;
                        base[r0 * 4 + col0 + 1] = c1;
                        base[r1 * 4 + col0]     = c2;
                        base[r1 * 4 + col0 + 1] = c3;
                    }
                }
            }
        }
    }

    // Write BF16 output O_DYN[m_base..m_eff-1, 0..N_dyn-1] (only lanes t < 2 own cols 0..3!)
    if (t < 2) {
        const int n0 = 2 * t;
        const int n1 = n0 + 1;
        #pragma unroll
        for (int tw = 0; tw < MAX_TW; ++tw) {
            const int mt = warp + tw * WARPS_RUN;
            if (mt < MT) {
                const int m0 = m_base + mt * 16 + g;
                const int m1 = m0 + 8;
                if (m0 < m_eff && n1 < N_dyn) {
                    *reinterpret_cast<__nv_bfloat162*>(O_DYN + static_cast<int64_t>(m0) * so_dyn + n0) =
                        __halves2bfloat162(__float2bfloat16_rn(acc[tw][0] + 0.0f), __float2bfloat16_rn(acc[tw][1] + 0.0f));
                } else if (m0 < m_eff && n0 < N_dyn) {
                    O_DYN[static_cast<int64_t>(m0) * so_dyn + n0] = __float2bfloat16_rn(acc[tw][0] + 0.0f);
                }
                if (m1 < m_eff && n1 < N_dyn) {
                    *reinterpret_cast<__nv_bfloat162*>(O_DYN + static_cast<int64_t>(m1) * so_dyn + n0) =
                        __halves2bfloat162(__float2bfloat16_rn(acc[tw][2] + 0.0f), __float2bfloat16_rn(acc[tw][3] + 0.0f));
                } else if (m1 < m_eff && n0 < N_dyn) {
                    O_DYN[static_cast<int64_t>(m1) * so_dyn + n0] = __float2bfloat16_rn(acc[tw][2] + 0.0f);
                }
            }
        }
    }
}

// Standalone fast startup 0-ULP probe kernel (`[M, K] @ [N_dyn<=4, K]^T`)
template <int WARPS_RUN, int MT>
__global__ __launch_bounds__(WARPS_RUN * 32, 4)
void bf11_dyn_probe_v9_kernel(
    const __nv_bfloat16* __restrict__ X,
    const __nv_bfloat16* __restrict__ W_DYN,
    __nv_bfloat16* __restrict__ O_DYN,
    const int32_t* __restrict__ NLIVE_PTR,
    int M, int N_dyn, int K, int sx, int so_dyn,
    int dyn_mode, int dyn_split_k
) {
    constexpr int BM = MT * 16;
    const int m_base = static_cast<int>(blockIdx.x) * BM;
    const int m_eff = (NLIVE_PTR != nullptr) ? min(M, *NLIVE_PTR) : M;
    if (m_base >= m_eff) return;
    extern __shared__ __nv_bfloat16 s_raw[];
    const int tid = static_cast<int>(threadIdx.x);
    run_dyn_tap_cta_v9<WARPS_RUN, MT>(
        X, W_DYN, O_DYN, s_raw,
        m_base, m_eff, N_dyn, K, sx, so_dyn,
        dyn_mode, dyn_split_k,
        tid, tid >> 5, tid & 31
    );
}

// Wave-V9 Fused C=1 (`M <= 16`) Dual-KV + `k_dyn`/`v_dyn` Kernel:
//   grid = (2 * n_tiles_each + 2, SPLIT, 1)
//     - pn_global < 2 * n_tiles_each : exact BF11 v5 K/V CTAs
//     - pn_global == 2 * n_tiles_each     : k_dyn on X0, W_DYN0 -> O_DYN0 (ps==0 active; ps>0 waits & exits)
//     - pn_global == 2 * n_tiles_each + 1 : v_dyn on X1, W_DYN1 -> O_DYN1 (ps==0 active; ps>0 waits & exits)
template <int WARPS_RUN, bool M_LE8, int SPLIT, bool PDL, bool TRIG>
__global__ __launch_bounds__(WARPS_RUN * 32, (M_LE8 ? (WARPS_RUN == 4 ? 5 : 6) : (WARPS_RUN == 4 ? 4 : 5)))
void bf11_flip_dual_kv_dyn_v9_c1_kernel(
    const __nv_bfloat16* __restrict__ X0, const __nv_bfloat16* __restrict__ X1,
    const uint8_t* __restrict__ SLABS0, const uint8_t* __restrict__ SLABS1,
    const uint32_t* __restrict__ SMETA0, const uint32_t* __restrict__ SMETA1,
    const uint16_t* __restrict__ SFLIPS0, const uint16_t* __restrict__ SFLIPS1,
    const uint32_t* __restrict__ BMASKS0, const uint32_t* __restrict__ BMASKS1,
    const uint64_t* __restrict__ OUTLIERS0, const uint64_t* __restrict__ OUTLIERS1,
    const int32_t* __restrict__ O_OFF0, const int32_t* __restrict__ O_OFF1,
    float* __restrict__ P, int32_t* __restrict__ SEM,
    __nv_bfloat16* __restrict__ O0, __nv_bfloat16* __restrict__ O1,
    const __nv_bfloat16* __restrict__ W_DYN0, const __nv_bfloat16* __restrict__ W_DYN1,
    __nv_bfloat16* __restrict__ O_DYN0, __nv_bfloat16* __restrict__ O_DYN1,
    const int32_t* __restrict__ NLIVE_PTR,
    int M, int N_each, int N_dyn, int K, int sx0, int sx1, int KS, int so0, int so1,
    int so_dyn0, int so_dyn1, int n_tiles_each, int w_global_each,
    uint32_t base_exp0, uint32_t base_exp1, int dyn_mode, int dyn_split_k
) {
    constexpr int BN = WARPS_RUN * 8;
    constexpr int THREADS = WARPS_RUN * 32;
    constexpr int M_ROWS = M_LE8 ? 8 : 16;

    const int pn_global = static_cast<int>(blockIdx.x);
    const int ps = static_cast<int>(blockIdx.y);
    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    extern __shared__ __nv_bfloat16 sX[];

    const int kv_tiles = 2 * n_tiles_each;
    if (pn_global >= kv_tiles) {
        const int which_dyn = pn_global - kv_tiles;  // 0 = k_dyn, 1 = v_dyn
        const __nv_bfloat16* W_D = which_dyn ? W_DYN1 : W_DYN0;
        if constexpr (PDL) {
            if (ps == 0 && tid < 32) {
                // Prefetch first 4 KiB of W_DYN into L2 while waiting for X0/X1!
                prefetch_l2(W_D + tid * 64);
            }
            asm volatile("griddepcontrol.wait;" ::: "memory");
        }
        if (ps == 0) {
            const int m_eff = (NLIVE_PTR != nullptr) ? min(M, *NLIVE_PTR) : M;
            if (m_eff > 0) {
                run_dyn_tap_cta_v9<WARPS_RUN, 1>(
                    which_dyn ? X1 : X0, W_D, which_dyn ? O_DYN1 : O_DYN0, sX,
                    0, m_eff, N_dyn, K, which_dyn ? sx1 : sx0, which_dyn ? so_dyn1 : so_dyn0,
                    dyn_mode, dyn_split_k, tid, warp, lane
                );
                __syncthreads();
            }
            if constexpr (TRIG) {
                if (tid == 0) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
        }
        return;
    }

    const int g = lane >> 2;
    const int t = lane & 3;
    const int which = (pn_global >= n_tiles_each) ? 1 : 0;
    const int pn_local = which ? (pn_global - n_tiles_each) : pn_global;

    const __nv_bfloat16* X = which ? X1 : X0;
    const uint8_t* SLABS = which ? SLABS1 : SLABS0;
    const uint32_t* SLAB_META_V5 = which ? SMETA1 : SMETA0;
    const uint16_t* SFLIPS = which ? SFLIPS1 : SFLIPS0;
    const uint32_t* BMASKS = which ? BMASKS1 : BMASKS0;
    const uint64_t* OUTLIERS = which ? OUTLIERS1 : OUTLIERS0;
    const int32_t* O_OFF = which ? O_OFF1 : O_OFF0;
    __nv_bfloat16* O = which ? O1 : O0;
    const int sx = which ? sx1 : sx0;
    const int so_m = which ? so1 : so0;
    const uint32_t base_exp = which ? base_exp1 : base_exp0;

    const int w_row = pn_local * WARPS_RUN + warp;
    const bool warp_active = (w_row < w_global_each) && (w_row * 8 < N_each);

    const int k_lo = ps * KS;
    const int k_hi = (k_lo + KS < K) ? (k_lo + KS) : K;
    const int k_len = k_hi - k_lo;
    const int total_slabs = warp_active ? (k_len >> 7) : 0;

    const int64_t warp_uid = static_cast<int64_t>(ps) * w_global_each + w_row;
    const int64_t slabs_per_warp = static_cast<int64_t>(KS >> 7);
    const int64_t slab_uid_base = warp_uid * slabs_per_warp;
    const uint8_t* slab_base = SLABS + slab_uid_base * SLAB11_BYTES;

    Slab11 cur_slab;
    uint32_t my_meta = 0u;
    uint32_t cur_smeta = 0u;
    uint32_t cur_raw_fl = 0u;
    int32_t o_cur = 0, o_end = 0;
    uint64_t o_rec = 0xFFFFFFFFFFFFFFFFULL;
    uint32_t o_next_step = 0xFFFFFFFFu;

    auto init_weight_streams = [&]() {
        if (total_slabs > 0) {
            cur_slab = load_slab11(slab_base, lane);
            if (lane < total_slabs) {
                my_meta = ld_ca_u32(SLAB_META_V5 + slab_uid_base + lane);
            }
            o_cur = __ldg(O_OFF + warp_uid);
            o_end = __ldg(O_OFF + warp_uid + 1);
            cur_smeta = __shfl_sync(0xFFFFFFFFu, my_meta, 0);
            cur_raw_fl = issue_flip_load_v5<false>(cur_smeta, SFLIPS, BMASKS, lane);
            if (total_slabs > 1 && lane < 11) {
                prefetch_l2(slab_base + SLAB11_BYTES + lane * 128);
            }
            if (o_cur < o_end) {
                o_rec = __ldg(OUTLIERS + o_cur);
                o_next_step = static_cast<uint32_t>(o_rec >> 32);
            }
        }
    };

    if constexpr (PDL) {
        init_weight_streams();
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }

    const int m_eff = (NLIVE_PTR != nullptr) ? min(M, *NLIVE_PTR) : M;

    __shared__ uint32_t s_flip[WARPS_RUN][32];
    const int sX_stride = KS + 8;
    const int vec_k = k_len >> 3;
    const int m_load = min(m_eff, M_ROWS);
    const int total_vec = M_ROWS * vec_k;
    for (int idx = tid; idx < total_vec; idx += THREADS) {
        const int m = idx / vec_k;
        const int kv = idx - m * vec_k;
        __nv_bfloat16* d = sX + m * sX_stride + (kv << 3);
        if (m < m_load) {
            cp_async_ca_16(d, X + static_cast<int64_t>(m) * sx + k_lo + (kv << 3));
        } else {
            *reinterpret_cast<uint4*>(d) = make_uint4(0u, 0u, 0u, 0u);
        }
    }
    cp_async_commit();

    if constexpr (!PDL) {
        init_weight_streams();
    }

    cp_async_wait<0>();
    __syncthreads();

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const int x_off0 = g * sX_stride + 2 * t;
    const int x_off1 = (g + 8) * sX_stride + 2 * t;
    const uint32_t base_sh7 = (base_exp << 7) | (base_exp << 23);

    for (int sl = 0; sl < total_slabs; ++sl) {
        Slab11 nxt_slab;
        uint32_t nxt_smeta = 0u;
        uint32_t nxt_raw_fl = 0u;
        if (sl + 1 < total_slabs) {
            nxt_slab = load_slab11(slab_base + static_cast<int64_t>(sl + 1) * SLAB11_BYTES, lane);
            nxt_smeta = __shfl_sync(0xFFFFFFFFu, my_meta, sl + 1);
            nxt_raw_fl = issue_flip_load_v5<false>(nxt_smeta, SFLIPS, BMASKS, lane);
        }
        const uint32_t fl = finish_flip_v5<false>(cur_raw_fl, cur_smeta, SFLIPS, lane, s_flip[warp]);
        const bool slab_has_out = (cur_smeta & BF11_V5_META_OUTLIER_BIT) != 0u;
        exec_slab11_c1_v5<M_LE8>(cur_slab, sl, sX, x_off0, x_off1, sl * 128, base_sh7, static_cast<uint32_t>(lane),
                                 fl, slab_has_out, OUTLIERS, o_cur, o_end, o_rec, o_next_step, acc);
        cur_slab = nxt_slab;
        cur_smeta = nxt_smeta;
        cur_raw_fl = nxt_raw_fl;
    }

    const int m0 = g;
    const int m1 = g + 8;
    const int64_t tile_stride = static_cast<int64_t>(SPLIT) * M * BN;
    const int64_t split_stride = static_cast<int64_t>(M) * BN;
    float* p_tile = P + static_cast<int64_t>(pn_global) * tile_stride;
    float* p_my = p_tile + static_cast<int64_t>(ps) * split_stride;
    const int nl0 = warp * 8 + 2 * t;
    const int n0_w = pn_local * BN + nl0;
    const int n1_w = n0_w + 1;
    const float v00 = (n0_w < N_each) ? acc[0] : 0.0f;
    const float v01 = (n1_w < N_each) ? acc[1] : 0.0f;
    const float v10 = (n0_w < N_each) ? acc[2] : 0.0f;
    const float v11 = (n1_w < N_each) ? acc[3] : 0.0f;
    if (m0 < m_eff) *reinterpret_cast<float2*>(p_my + m0 * BN + nl0) = make_float2(v00, v01);
    if constexpr (!M_LE8) {
        if (m1 < m_eff) *reinterpret_cast<float2*>(p_my + m1 * BN + nl0) = make_float2(v10, v11);
    }

    __syncthreads();
    __shared__ int s_done;
    if (tid == 0) {
        s_done = atomic_add_acq_rel_gpu(&SEM[pn_global], 1);
    }
    __syncthreads();

    if (s_done == SPLIT - 1) {
        if (tid == 0) st_release_gpu_s32(&SEM[pn_global], 0);
        if constexpr (TRIG) {
            if (tid == 0) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
        constexpr int PAIRS_PER_ROW = BN / 2;
        const int total_pairs = m_eff * PAIRS_PER_ROW;
        const int n_tile_base = pn_local * BN;
        for (int idx = tid; idx < total_pairs; idx += THREADS) {
            const int m = idx / PAIRS_PER_ROW;
            const int p = idx - m * PAIRS_PER_ROW;
            const int nl = p * 2;
            const int n0 = n_tile_base + nl;
            if (n0 >= N_each) continue;
            const int64_t off = static_cast<int64_t>(m) * BN + nl;
            float s0 = 0.0f, s1 = 0.0f;
            #pragma unroll
            for (int s = 0; s < SPLIT; ++s) {
                float2 v = ld_cg_f2(reinterpret_cast<const float2*>(p_tile + static_cast<int64_t>(s) * split_stride + off));
                s0 += v.x;
                s1 += v.y;
            }
            if (n0 + 1 < N_each) {
                *reinterpret_cast<__nv_bfloat162*>(O + static_cast<int64_t>(m) * so_m + n0) =
                    __halves2bfloat162(__float2bfloat16_rn(s0 + 0.0f), __float2bfloat16_rn(s1 + 0.0f));
            } else {
                O[static_cast<int64_t>(m) * so_m + n0] = __float2bfloat16_rn(s0 + 0.0f);
            }
        }
    }
}

// Wave-V9 Fused C=32 (`M = 17..128+`) Dual-KV + `k_dyn`/`v_dyn` Kernel:
//   grid = (m_blocks, 2 * n_tiles_each + 2, SPLIT_CTA)
template <
    int WARPS_RUN,
    int MT,
    int K_ORDER,
    int SPLIT_CTA,
    bool CACHE_L2,
    bool PDL,
    bool TRIG
>
__global__ __launch_bounds__(WARPS_RUN * 32, (MT >= 4 ? 3 : (MT == 3 ? (WARPS_RUN == 4 ? 3 : 4) : (WARPS_RUN == 4 ? 4 : 6))))
void bf11_flip_dual_kv_dyn_v9_c32_kernel(
    const __nv_bfloat16* __restrict__ X0, const __nv_bfloat16* __restrict__ X1,
    const uint8_t* __restrict__ SLABS0, const uint8_t* __restrict__ SLABS1,
    const uint32_t* __restrict__ SMETA0, const uint32_t* __restrict__ SMETA1,
    const uint16_t* __restrict__ SFLIPS0, const uint16_t* __restrict__ SFLIPS1,
    const uint32_t* __restrict__ BMASKS0, const uint32_t* __restrict__ BMASKS1,
    const uint64_t* __restrict__ OUTLIERS0, const uint64_t* __restrict__ OUTLIERS1,
    const int32_t* __restrict__ O_OFF0, const int32_t* __restrict__ O_OFF1,
    float* __restrict__ P, int32_t* __restrict__ SEM,
    __nv_bfloat16* __restrict__ O0, __nv_bfloat16* __restrict__ O1,
    const __nv_bfloat16* __restrict__ W_DYN0, const __nv_bfloat16* __restrict__ W_DYN1,
    __nv_bfloat16* __restrict__ O_DYN0, __nv_bfloat16* __restrict__ O_DYN1,
    const int32_t* __restrict__ NLIVE_PTR,
    int M, int N_each, int N_dyn, int K, int sx0, int sx1, int KS, int split_packed,
    int so0, int so1, int so_dyn0, int so_dyn1, int n_tiles_each, int w_global_each,
    uint32_t base_exp0, uint32_t base_exp1, int dyn_mode, int dyn_split_k
) {
    constexpr int BN = WARPS_RUN * 8;
    constexpr int BM = MT * 16;
    constexpr int THREADS = WARPS_RUN * 32;
    constexpr int BK = 128;
    constexpr int BUF_ELEMS = BM * BK;
    constexpr bool USE_SPLIT = (SPLIT_CTA > 1);
    constexpr bool USE_PACC = (K_ORDER == 1 && !USE_SPLIT);

    const int mb = static_cast<int>(blockIdx.x);
    const int m_blocks = static_cast<int>(gridDim.x);
    const int pn_global = static_cast<int>(blockIdx.y);
    const int ps_start = USE_SPLIT ? static_cast<int>(blockIdx.z) : 0;
    const int ps_count = USE_SPLIT ? 1 : split_packed;
    const int m_base = mb * BM;
    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    extern __shared__ __nv_bfloat16 sX[];

    const int kv_tiles = 2 * n_tiles_each;
    if (pn_global >= kv_tiles) {
        const int which_dyn = pn_global - kv_tiles;  // 0 = k_dyn, 1 = v_dyn
        const __nv_bfloat16* W_D = which_dyn ? W_DYN1 : W_DYN0;
        if constexpr (PDL) {
            if (ps_start == 0 && tid < 32) {
                prefetch_l2(W_D + tid * 64);
            }
            asm volatile("griddepcontrol.wait;" ::: "memory");
        }
        if (ps_start == 0) {
            const int m_eff = (NLIVE_PTR != nullptr) ? min(M, *NLIVE_PTR) : M;
            if (m_base < m_eff) {
                run_dyn_tap_cta_v9<WARPS_RUN, MT>(
                    which_dyn ? X1 : X0, W_D, which_dyn ? O_DYN1 : O_DYN0, sX,
                    m_base, m_eff, N_dyn, K, which_dyn ? sx1 : sx0, which_dyn ? so_dyn1 : so_dyn0,
                    dyn_mode, dyn_split_k, tid, warp, lane
                );
                __syncthreads();
            }
            if constexpr (TRIG) {
                if (tid == 0) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
        }
        return;
    }

    const int which = (pn_global >= n_tiles_each) ? 1 : 0;
    const int pn_local = which ? (pn_global - n_tiles_each) : pn_global;
    const int g = lane >> 2;
    const int t = lane & 3;
    const uint32_t lane_row_bytes = static_cast<uint32_t>((lane & 15) * BK * 2);
    const uint32_t lane_k_half = static_cast<uint32_t>(lane >> 4);

    const __nv_bfloat16* X = which ? X1 : X0;
    const uint8_t* SLABS = which ? SLABS1 : SLABS0;
    const uint32_t* SLAB_META_V5 = which ? SMETA1 : SMETA0;
    const uint16_t* SFLIPS = which ? SFLIPS1 : SFLIPS0;
    const uint32_t* BMASKS = which ? BMASKS1 : BMASKS0;
    const uint64_t* OUTLIERS = which ? OUTLIERS1 : OUTLIERS0;
    const int32_t* O_OFF = which ? O_OFF1 : O_OFF0;
    __nv_bfloat16* O = which ? O1 : O0;
    const int sx = which ? sx1 : sx0;
    const int so_m = which ? so1 : so0;
    const uint32_t base_exp = which ? base_exp1 : base_exp0;

    const int w_row = pn_local * WARPS_RUN + warp;
    const int n_warp_base = w_row * 8;
    const bool warp_active = (w_row < w_global_each) && (n_warp_base < N_each);
    const int n_chunks_per_sp = KS >> 7;
    const int total_chunks = ps_count * n_chunks_per_sp;

    int64_t warp_uid = static_cast<int64_t>(ps_start) * w_global_each + w_row;
    Slab11 cur_slab;
    uint32_t my_meta_cur = 0u;
    uint32_t my_meta_nxt = 0u;
    uint32_t cur_smeta = 0u;
    uint32_t cur_raw_fl = 0u;
    int32_t o_cur = 0, o_end = 0;
    uint64_t o_rec = 0xFFFFFFFFFFFFFFFFULL;
    uint32_t o_next_step = 0xFFFFFFFFu;

    auto init_weight_streams = [&]() {
        if (warp_active && total_chunks > 0) {
            const int64_t suid0 = warp_uid * n_chunks_per_sp;
            cur_slab = load_slab11_t<CACHE_L2>(SLABS + suid0 * SLAB11_BYTES, lane);
            if (lane < n_chunks_per_sp) {
                my_meta_cur = ld_ca_u32(SLAB_META_V5 + suid0 + lane);
            }
            if constexpr (!USE_SPLIT) {
                if (ps_start + 1 < split_packed && lane < n_chunks_per_sp) {
                    const int64_t wuid1 = static_cast<int64_t>(ps_start + 1) * w_global_each + w_row;
                    my_meta_nxt = ld_ca_u32(SLAB_META_V5 + wuid1 * n_chunks_per_sp + lane);
                }
            }
            o_cur = __ldg(O_OFF + warp_uid);
            o_end = __ldg(O_OFF + warp_uid + 1);
            cur_smeta = __shfl_sync(0xFFFFFFFFu, my_meta_cur, 0);
            cur_raw_fl = issue_flip_load_v5<CACHE_L2>(cur_smeta, SFLIPS, BMASKS, lane);
            if (o_cur < o_end) {
                o_rec = __ldg(OUTLIERS + o_cur);
                o_next_step = static_cast<uint32_t>(o_rec >> 32);
            }
        }
    };

    if constexpr (PDL) {
        init_weight_streams();
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }

    const int m_eff = (NLIVE_PTR != nullptr) ? min(M, *NLIVE_PTR) : M;
    if (m_base >= m_eff) {
        if constexpr (TRIG) {
            if ((!USE_SPLIT || ps_start == SPLIT_CTA - 1) && tid == 0) {
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
        }
        return;
    }

    __shared__ uint32_t s_flip[WARPS_RUN][32];
    const uint32_t smem_base_0 = static_cast<uint32_t>(__cvta_generic_to_shared(sX));
    const uint32_t smem_base_1 = static_cast<uint32_t>(__cvta_generic_to_shared(sX + BUF_ELEMS));

    float acc[MT][4];
    #pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
        acc[mt][0] = 0.0f; acc[mt][1] = 0.0f; acc[mt][2] = 0.0f; acc[mt][3] = 0.0f;
    }

    [[maybe_unused]] float p_acc[USE_PACC ? MT : 1][4];
    if constexpr (USE_PACC) {
        #pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            p_acc[mt][0] = 0.0f; p_acc[mt][1] = 0.0f; p_acc[mt][2] = 0.0f; p_acc[mt][3] = 0.0f;
        }
    }

    const uint32_t base_sh7 = (base_exp << 7) | (base_exp << 23);

    if (total_chunks > 0) {
        load_x_swiz_async<BM, BK, THREADS>(X, sx, m_base, m_eff, ps_start * KS, sX, tid);

        if constexpr (!PDL) {
            init_weight_streams();
        }

        int ps = ps_start;
        int kc = 0;
        for (int c = 0; c < total_chunks; ++c) {
            const int k_chunk = ps * KS + kc * 128;
            const int buf = c & 1;
            const bool crossing_ps = (kc + 1 == n_chunks_per_sp);
            const bool has_next = (c + 1 < total_chunks);

            cp_async_wait<0>();
            __syncthreads();
            if (has_next) {
                load_x_swiz_async<BM, BK, THREADS>(
                    X, sx, m_base, m_eff, k_chunk + 128,
                    sX + (buf ^ 1) * BUF_ELEMS, tid
                );
            }

            Slab11 nxt_slab;
            uint32_t nxt_smeta = 0u;
            uint32_t nxt_raw_fl = 0u;
            int32_t nxt_o_cur = 0, nxt_o_end = 0;
            if (warp_active && has_next) {
                int64_t nxt_suid;
                if (!crossing_ps) {
                    nxt_suid = warp_uid * n_chunks_per_sp + (kc + 1);
                    nxt_smeta = __shfl_sync(0xFFFFFFFFu, my_meta_cur, kc + 1);
                } else {
                    const int64_t nxt_wuid = static_cast<int64_t>(ps + 1) * w_global_each + w_row;
                    nxt_suid = nxt_wuid * n_chunks_per_sp;
                    nxt_smeta = __shfl_sync(0xFFFFFFFFu, my_meta_nxt, 0);
                    nxt_o_cur = __ldg(O_OFF + nxt_wuid);
                    nxt_o_end = __ldg(O_OFF + nxt_wuid + 1);
                }
                nxt_slab = load_slab11_t<CACHE_L2>(SLABS + nxt_suid * SLAB11_BYTES, lane);
                nxt_raw_fl = issue_flip_load_v5<CACHE_L2>(nxt_smeta, SFLIPS, BMASKS, lane);
            }

            const uint32_t fl = warp_active
                ? finish_flip_v5<CACHE_L2>(cur_raw_fl, cur_smeta, SFLIPS, lane, s_flip[warp])
                : 0u;
            const bool slab_has_out = (cur_smeta & BF11_V5_META_OUTLIER_BIT) != 0u;
            const uint32_t cur_smem = (buf == 0) ? smem_base_0 : smem_base_1;

            if (warp_active) {
                if constexpr (USE_PACC) {
                    exec_full_slab11_v5_bk128<MT>(
                        cur_slab, kc, cur_smem, static_cast<uint32_t>(lane), lane_row_bytes, lane_k_half,
                        base_sh7, fl, slab_has_out, OUTLIERS, o_cur, o_end, o_rec, o_next_step, p_acc
                    );
                } else {
                    exec_full_slab11_v5_bk128<MT>(
                        cur_slab, kc, cur_smem, static_cast<uint32_t>(lane), lane_row_bytes, lane_k_half,
                        base_sh7, fl, slab_has_out, OUTLIERS, o_cur, o_end, o_rec, o_next_step, acc
                    );
                }
            }

            cur_slab = nxt_slab;
            cur_smeta = nxt_smeta;
            cur_raw_fl = nxt_raw_fl;

            if (!crossing_ps) {
                ++kc;
            } else {
                if constexpr (USE_PACC) {
                    #pragma unroll
                    for (int mt = 0; mt < MT; ++mt) {
                        acc[mt][0] += p_acc[mt][0];
                        acc[mt][1] += p_acc[mt][1];
                        acc[mt][2] += p_acc[mt][2];
                        acc[mt][3] += p_acc[mt][3];
                        p_acc[mt][0] = 0.0f; p_acc[mt][1] = 0.0f;
                        p_acc[mt][2] = 0.0f; p_acc[mt][3] = 0.0f;
                    }
                }
                kc = 0;
                ++ps;
                if constexpr (!USE_SPLIT) {
                    if (has_next && warp_active) {
                        warp_uid = static_cast<int64_t>(ps) * w_global_each + w_row;
                        my_meta_cur = my_meta_nxt;
                        if (ps + 1 < split_packed && lane < n_chunks_per_sp) {
                            const int64_t wuid2 = static_cast<int64_t>(ps + 1) * w_global_each + w_row;
                            my_meta_nxt = ld_ca_u32(SLAB_META_V5 + wuid2 * n_chunks_per_sp + lane);
                        }
                        o_cur = nxt_o_cur;
                        o_end = nxt_o_end;
                        if (o_cur < o_end) {
                            o_rec = __ldg(OUTLIERS + o_cur);
                            o_next_step = static_cast<uint32_t>(o_rec >> 32);
                        } else {
                            o_rec = 0xFFFFFFFFFFFFFFFFULL;
                            o_next_step = 0xFFFFFFFFu;
                        }
                    }
                }
            }
        }
    }

    if constexpr (!USE_SPLIT) {
        if constexpr (TRIG) {
            if (tid == 0) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
        const int n0 = n_warp_base + 2 * t;
        const int n1 = n0 + 1;
        #pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            const int m0 = m_base + mt * 16 + g;
            const int m1 = m0 + 8;
            if (m0 < m_eff && n1 < N_each) {
                *reinterpret_cast<__nv_bfloat162*>(O + static_cast<int64_t>(m0) * so_m + n0) =
                    __halves2bfloat162(__float2bfloat16_rn(acc[mt][0] + 0.0f), __float2bfloat16_rn(acc[mt][1] + 0.0f));
            } else if (m0 < m_eff && n0 < N_each) {
                O[static_cast<int64_t>(m0) * so_m + n0] = __float2bfloat16_rn(acc[mt][0] + 0.0f);
            }
            if (m1 < m_eff && n1 < N_each) {
                *reinterpret_cast<__nv_bfloat162*>(O + static_cast<int64_t>(m1) * so_m + n0) =
                    __halves2bfloat162(__float2bfloat16_rn(acc[mt][2] + 0.0f), __float2bfloat16_rn(acc[mt][3] + 0.0f));
            } else if (m1 < m_eff && n0 < N_each) {
                O[static_cast<int64_t>(m1) * so_m + n0] = __float2bfloat16_rn(acc[mt][2] + 0.0f);
            }
        }
    } else {
        const int m_tile = min(BM, m_eff - m_base);
        const int tile_id = pn_global * m_blocks + mb;
        const int64_t split_stride = static_cast<int64_t>(BM) * BN;
        const int64_t tile_stride = static_cast<int64_t>(SPLIT_CTA) * split_stride;
        float* p_tile = P + static_cast<int64_t>(tile_id) * tile_stride;
        float* p_my = p_tile + static_cast<int64_t>(ps_start) * split_stride;

        const int nl0 = warp * 8 + 2 * t;
        const int n0_w = pn_local * BN + nl0;
        const int n1_w = n0_w + 1;
        const bool n0_act = (n0_w < N_each);
        const bool n1_act = (n1_w < N_each);
        #pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            const int lm0 = mt * 16 + g;
            const int lm1 = lm0 + 8;
            const float v00 = n0_act ? acc[mt][0] : 0.0f;
            const float v01 = n1_act ? acc[mt][1] : 0.0f;
            const float v10 = n0_act ? acc[mt][2] : 0.0f;
            const float v11 = n1_act ? acc[mt][3] : 0.0f;
            if (lm0 < m_tile) {
                *reinterpret_cast<float2*>(p_my + lm0 * BN + nl0) = make_float2(v00, v01);
            }
            if (lm1 < m_tile) {
                *reinterpret_cast<float2*>(p_my + lm1 * BN + nl0) = make_float2(v10, v11);
            }
        }

        __syncthreads();
        __shared__ int s_done;
        if (tid == 0) {
            s_done = atomic_add_acq_rel_gpu(&SEM[tile_id], 1);
        }
        __syncthreads();

        if (s_done == SPLIT_CTA - 1) {
            if (tid == 0) st_release_gpu_s32(&SEM[tile_id], 0);
            if constexpr (TRIG) {
                if (tid == 0) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
            constexpr int PAIRS_PER_ROW = BN / 2;
            const int total_pairs = m_tile * PAIRS_PER_ROW;
            const int n_tile_base = pn_local * BN;
            for (int idx = tid; idx < total_pairs; idx += THREADS) {
                const int lm = idx / PAIRS_PER_ROW;
                const int p = idx - lm * PAIRS_PER_ROW;
                const int nl = p * 2;
                const int n0 = n_tile_base + nl;
                if (n0 >= N_each) continue;
                const int gm = m_base + lm;
                const int64_t off = static_cast<int64_t>(lm) * BN + nl;
                float s0 = 0.0f, s1 = 0.0f;
                #pragma unroll
                for (int s = 0; s < SPLIT_CTA; ++s) {
                    float2 v = ld_cg_f2(reinterpret_cast<const float2*>(p_tile + static_cast<int64_t>(s) * split_stride + off));
                    s0 += v.x;
                    s1 += v.y;
                }
                if (n0 + 1 < N_each) {
                    *reinterpret_cast<__nv_bfloat162*>(O + static_cast<int64_t>(gm) * so_m + n0) =
                        __halves2bfloat162(__float2bfloat16_rn(s0 + 0.0f), __float2bfloat16_rn(s1 + 0.0f));
                } else {
                    O[static_cast<int64_t>(gm) * so_m + n0] = __float2bfloat16_rn(s0 + 0.0f);
                }
            }
        }
    }
}

template <int W_RUN, int MT_V, int KO_V, int SPLIT_V>
static cudaError_t dispatch_bf11_v9_dual_kv_dyn_c32(
    const void* X0, const void* X1, const void* SLABS0, const void* SLABS1,
    const void* SMETA0, const void* SMETA1, const void* SFLIPS0, const void* SFLIPS1,
    const void* BMASKS0, const void* BMASKS1, const void* OUTLIERS0, const void* OUTLIERS1,
    const void* O_OFF0, const void* O_OFF1,
    void* P, void* SEM, void* O0, void* O1,
    const void* W_DYN0, const void* W_DYN1, void* O_DYN0, void* O_DYN1,
    const void* NLIVE_PTR,
    int M, int N_each, int N_dyn, int K, int sx0, int sx1, int KS, int split_packed,
    int so0, int so1, int so_dyn0, int so_dyn1, int w_global_each,
    uint32_t base_exp0, uint32_t base_exp1, int dyn_mode, int dyn_split_k,
    bool pdl, bool trig, cudaStream_t stream
) {
    constexpr int BM = MT_V * 16;
    const int n_tiles_each_run = (w_global_each + W_RUN - 1) / W_RUN;
    const int m_blocks = (M + BM - 1) / BM;
    const bool cache_l2 = (m_blocks > 1);
    dim3 grid(m_blocks, 2 * n_tiles_each_run + 2, SPLIT_V);
    dim3 block(W_RUN * 32, 1, 1);
    const size_t smem = static_cast<size_t>(2) * BM * 128 * sizeof(__nv_bfloat16);
    #define LV9_DKVD(CA, PD, TR) launch_maybe_pdl(bf11_flip_dual_kv_dyn_v9_c32_kernel<W_RUN, MT_V, KO_V, SPLIT_V, CA, PD, TR>, \
        grid, block, smem, stream, PD, \
        static_cast<const __nv_bfloat16*>(X0), static_cast<const __nv_bfloat16*>(X1), \
        static_cast<const uint8_t*>(SLABS0), static_cast<const uint8_t*>(SLABS1), \
        static_cast<const uint32_t*>(SMETA0), static_cast<const uint32_t*>(SMETA1), \
        static_cast<const uint16_t*>(SFLIPS0), static_cast<const uint16_t*>(SFLIPS1), \
        static_cast<const uint32_t*>(BMASKS0), static_cast<const uint32_t*>(BMASKS1), \
        static_cast<const uint64_t*>(OUTLIERS0), static_cast<const uint64_t*>(OUTLIERS1), \
        static_cast<const int32_t*>(O_OFF0), static_cast<const int32_t*>(O_OFF1), \
        static_cast<float*>(P), static_cast<int32_t*>(SEM), \
        static_cast<__nv_bfloat16*>(O0), static_cast<__nv_bfloat16*>(O1), \
        static_cast<const __nv_bfloat16*>(W_DYN0), static_cast<const __nv_bfloat16*>(W_DYN1), \
        static_cast<__nv_bfloat16*>(O_DYN0), static_cast<__nv_bfloat16*>(O_DYN1), \
        static_cast<const int32_t*>(NLIVE_PTR), \
        M, N_each, N_dyn, K, sx0, sx1, KS, split_packed, so0, so1, so_dyn0, so_dyn1, \
        n_tiles_each_run, w_global_each, base_exp0, base_exp1, dyn_mode, dyn_split_k)
    if (cache_l2) {
        if (pdl) return trig ? LV9_DKVD(true, true, true) : LV9_DKVD(true, true, false);
        return trig ? LV9_DKVD(true, false, true) : LV9_DKVD(true, false, false);
    } else {
        if (pdl) return trig ? LV9_DKVD(false, true, true) : LV9_DKVD(false, true, false);
        return trig ? LV9_DKVD(false, false, true) : LV9_DKVD(false, false, false);
    }
    #undef LV9_DKVD
}

static inline bool dyn_params_valid(int N_dyn, int K, int dyn_mode, int dyn_split_k) {
    if (N_dyn <= 0 || N_dyn > 4 || K <= 0 || dyn_mode < 0 || dyn_mode > 3) return false;
    if (dyn_split_k <= 0 || (K % dyn_split_k) != 0) return false;
    const int ks_dyn = K / dyn_split_k;
    if ((ks_dyn & 15) != 0) return false;
    if ((dyn_mode & 1) != 0) {
        // Tree mode requires power-of-2 split_k <= 32
        if (dyn_split_k > 32 || (dyn_split_k & (dyn_split_k - 1)) != 0) return false;
    }
    return true;
}

// =========================================================================
// PART 3C: WAVE-V12 1-LAUNCH 0-ULP GPU BF11 -> BF16 WEIGHT UNPACK KERNEL
//   Eliminates the 35-kernel / 650-MiB PyTorch `unpack_bf16()` fallback storm
//   when `INKLING_BF11_FREE_RAW=1` encounters eager M=17..255 or uncaptured shapes!
//   - 1 warp (32 threads) per `warp_uid` -> 1,428 CTAs for qkvr, 576 CTAs for wo_ud
//   - 128 B SMEM (`s_flip[32]`), ZERO temp allocations, 100% CUDA-graph-capture-safe
//   - Uses exact same `unpack4_bf11_flip_v5` + `patch_outliers_step` -> literal nd == 0!
// =========================================================================

template <int S>
static __device__ __forceinline__ void unpack_store_step_v12(
    const Slab11& r, int sl, uint32_t fl, uint32_t base_sh7, uint32_t lane,
    bool has_out, const uint64_t* __restrict__ OUTLIERS,
    int32_t& o_cur, int32_t o_end, uint64_t& o_rec, uint32_t& o_next_step,
    __nv_bfloat16* __restrict__ row_ptr, int k_base, int t, bool row_valid, bool row_active
) {
    uint32_t b0 = 0u, b1 = 0u;
    if (row_active) {
        unpack4_bf11_flip_v5(slab_sm_word<S>(r), slab_c12_raw<S>(r), fl >> (S * 4), base_sh7, b0, b1);
    }
    if (__builtin_expect(has_out, 0)) {
        // Must drain outlier cursor warp-uniformly even if some lanes have !row_active!
        patch_outliers_step(static_cast<uint32_t>((sl << 3) | S), lane, OUTLIERS, o_cur, o_end, o_rec, o_next_step, b0, b1);
        if (!row_active) { b0 = 0u; b1 = 0u; }
    }
    if (row_valid) {
        const int k0 = k_base + S * 16 + 2 * t;
        *reinterpret_cast<uint32_t*>(row_ptr + k0)     = b0;
        *reinterpret_cast<uint32_t*>(row_ptr + k0 + 8) = b1;
    }
}

__global__ __launch_bounds__(32, 8)
void bf11_unpack_bf16_v12_kernel(
    const uint8_t* __restrict__ SLABS,
    const uint16_t* __restrict__ FLIPS,
    const uint4* __restrict__ WARP_META,
    const uint64_t* __restrict__ OUTLIERS,
    const int32_t* __restrict__ O_OFF,
    __nv_bfloat16* __restrict__ OUT,
    int N, int N_active, int K, int KS,
    int warps, int sub_n, int n_tiles, int spw, int wtot,
    uint32_t base_exp
) {
    const int widx = static_cast<int>(blockIdx.x);
    if (widx >= wtot) return;
    const int lane = static_cast<int>(threadIdx.x);
    const int g = lane >> 2;
    const int t = lane & 3;

    const int w = widx % warps;
    const int tmp = widx / warps;
    const int nt = tmp % n_tiles;
    const int ps = tmp / n_tiles;

    const uint4 meta = __ldg(WARP_META + widx);
    int64_t f_cur = static_cast<int64_t>(meta.x);
    int32_t o_cur = __ldg(O_OFF + widx);
    const int32_t o_end = __ldg(O_OFF + widx + 1);
    uint64_t o_rec = 0xFFFFFFFFFFFFFFFFULL;
    uint32_t o_next_step = 0xFFFFFFFFu;
    const bool has_out = (o_cur < o_end);
    if (has_out) {
        o_rec = __ldg(OUTLIERS + o_cur);
        o_next_step = static_cast<uint32_t>(o_rec >> 32);
    }

    __shared__ uint32_t s_flip[32];
    const uint32_t base_sh7 = (base_exp << 7) | (base_exp << 23);
    const uint8_t* slab_base = SLABS + static_cast<int64_t>(widx) * spw * SLAB11_BYTES;
    const int row_warp_base = (nt * warps + w) * (sub_n * 8);

    for (int sl = 0; sl < spw; ++sl) {
        const int sub = (sub_n == 2) ? (sl & 1) : 0;
        const int kc = (sub_n == 2) ? (sl >> 1) : sl;
        const int row = row_warp_base + sub * 8 + g;
        const bool row_valid = (row < N);
        const bool row_active = (row < N_active);
        __nv_bfloat16* row_ptr = row_valid ? (OUT + static_cast<int64_t>(row) * K) : OUT;
        const int k_base = ps * KS + kc * 128;

        const Slab11 r = load_slab11(slab_base + static_cast<int64_t>(sl) * SLAB11_BYTES, lane);
        const uint32_t nf = meta_nf(meta, sl);
        uint32_t fl = 0u;
        if (nf > 0u) {
            s_flip[lane] = 0u;
            __syncwarp();
            for (uint32_t i = static_cast<uint32_t>(lane); i < nf; i += 32u) {
                const uint32_t frec = ld_nc_u16(FLIPS + f_cur + i);
                atomicOr(&s_flip[frec >> 5], 1u << (frec & 31u));
            }
            __syncwarp();
            fl = s_flip[lane];
            f_cur += static_cast<int64_t>(nf);
        }

        unpack_store_step_v12<0>(r, sl, fl, base_sh7, static_cast<uint32_t>(lane), has_out, OUTLIERS, o_cur, o_end, o_rec, o_next_step, row_ptr, k_base, t, row_valid, row_active);
        unpack_store_step_v12<1>(r, sl, fl, base_sh7, static_cast<uint32_t>(lane), has_out, OUTLIERS, o_cur, o_end, o_rec, o_next_step, row_ptr, k_base, t, row_valid, row_active);
        unpack_store_step_v12<2>(r, sl, fl, base_sh7, static_cast<uint32_t>(lane), has_out, OUTLIERS, o_cur, o_end, o_rec, o_next_step, row_ptr, k_base, t, row_valid, row_active);
        unpack_store_step_v12<3>(r, sl, fl, base_sh7, static_cast<uint32_t>(lane), has_out, OUTLIERS, o_cur, o_end, o_rec, o_next_step, row_ptr, k_base, t, row_valid, row_active);
        unpack_store_step_v12<4>(r, sl, fl, base_sh7, static_cast<uint32_t>(lane), has_out, OUTLIERS, o_cur, o_end, o_rec, o_next_step, row_ptr, k_base, t, row_valid, row_active);
        unpack_store_step_v12<5>(r, sl, fl, base_sh7, static_cast<uint32_t>(lane), has_out, OUTLIERS, o_cur, o_end, o_rec, o_next_step, row_ptr, k_base, t, row_valid, row_active);
        unpack_store_step_v12<6>(r, sl, fl, base_sh7, static_cast<uint32_t>(lane), has_out, OUTLIERS, o_cur, o_end, o_rec, o_next_step, row_ptr, k_base, t, row_valid, row_active);
        unpack_store_step_v12<7>(r, sl, fl, base_sh7, static_cast<uint32_t>(lane), has_out, OUTLIERS, o_cur, o_end, o_rec, o_next_step, row_ptr, k_base, t, row_valid, row_active);
    }
}

// Shared C32 launcher supporting dedicated V12 BM=48 (MT=3, BK=128) tile!
static cudaError_t launch_bf11_v12_c32_impl(
    const void* X, const void* SLABS,
    const void* SLAB_META_V5, const void* SFLIPS, const void* BMASKS,
    const void* OUTLIERS, const void* O_OFF,
    void* P, void* SEM, void* O, const void* NLIVE_PTR,
    int M, int N, int N_active, int K, int sx, int KS, int split_packed, int so_m,
    int cfg_id, int bm_req, int w_run_req, int split_cta_req, int k_order,
    uint32_t base_exp, int pdl, int trig, bool allow_auto_bm48, cudaStream_t stream
) {
    if (M <= 0 || N <= 0 || KS <= 0 || (KS & 127) != 0 || (KS >> 7) > 32 || split_packed <= 0 || KS * split_packed != K) {
        return static_cast<cudaError_t>(-2);
    }
    const bool do_pdl = (pdl != 0);
    const bool do_trig = (trig != 0);
    const bool is_sub2 = (cfg_id == 0 || cfg_id == 2);
    const int ko = (split_packed == 1) ? 0 : ((k_order != 0) ? 1 : 0);

    if (is_sub2) {
        int bm = (bm_req == 32 || bm_req == 48 || bm_req == 64) ? bm_req : ((M <= 32) ? 32 : ((allow_auto_bm48 && M <= 48) ? 48 : 64));
        if (cfg_id == 0) {
            if (ko == 0) {
                if (bm == 32) return dispatch_bf11_v5_c32_sub2<4, 2, 0>(X, SLABS, SLAB_META_V5, SFLIPS, BMASKS, OUTLIERS, O_OFF, O, NLIVE_PTR, M, N, N_active, K, sx, KS, split_packed, so_m, base_exp, do_pdl, do_trig, stream);
                if (bm == 48) return dispatch_bf11_v5_c32_sub2<4, 3, 0>(X, SLABS, SLAB_META_V5, SFLIPS, BMASKS, OUTLIERS, O_OFF, O, NLIVE_PTR, M, N, N_active, K, sx, KS, split_packed, so_m, base_exp, do_pdl, do_trig, stream);
                return dispatch_bf11_v5_c32_sub2<4, 4, 0>(X, SLABS, SLAB_META_V5, SFLIPS, BMASKS, OUTLIERS, O_OFF, O, NLIVE_PTR, M, N, N_active, K, sx, KS, split_packed, so_m, base_exp, do_pdl, do_trig, stream);
            } else {
                if (bm == 32) return dispatch_bf11_v5_c32_sub2<4, 2, 1>(X, SLABS, SLAB_META_V5, SFLIPS, BMASKS, OUTLIERS, O_OFF, O, NLIVE_PTR, M, N, N_active, K, sx, KS, split_packed, so_m, base_exp, do_pdl, do_trig, stream);
                if (bm == 48) return dispatch_bf11_v5_c32_sub2<4, 3, 1>(X, SLABS, SLAB_META_V5, SFLIPS, BMASKS, OUTLIERS, O_OFF, O, NLIVE_PTR, M, N, N_active, K, sx, KS, split_packed, so_m, base_exp, do_pdl, do_trig, stream);
                return dispatch_bf11_v5_c32_sub2<4, 4, 1>(X, SLABS, SLAB_META_V5, SFLIPS, BMASKS, OUTLIERS, O_OFF, O, NLIVE_PTR, M, N, N_active, K, sx, KS, split_packed, so_m, base_exp, do_pdl, do_trig, stream);
            }
        } else {
            if (ko == 0) {
                if (bm == 32) return dispatch_bf11_v5_c32_sub2<2, 2, 0>(X, SLABS, SLAB_META_V5, SFLIPS, BMASKS, OUTLIERS, O_OFF, O, NLIVE_PTR, M, N, N_active, K, sx, KS, split_packed, so_m, base_exp, do_pdl, do_trig, stream);
                if (bm == 48) return dispatch_bf11_v5_c32_sub2<2, 3, 0>(X, SLABS, SLAB_META_V5, SFLIPS, BMASKS, OUTLIERS, O_OFF, O, NLIVE_PTR, M, N, N_active, K, sx, KS, split_packed, so_m, base_exp, do_pdl, do_trig, stream);
                return dispatch_bf11_v5_c32_sub2<2, 4, 0>(X, SLABS, SLAB_META_V5, SFLIPS, BMASKS, OUTLIERS, O_OFF, O, NLIVE_PTR, M, N, N_active, K, sx, KS, split_packed, so_m, base_exp, do_pdl, do_trig, stream);
            } else {
                if (bm == 32) return dispatch_bf11_v5_c32_sub2<2, 2, 1>(X, SLABS, SLAB_META_V5, SFLIPS, BMASKS, OUTLIERS, O_OFF, O, NLIVE_PTR, M, N, N_active, K, sx, KS, split_packed, so_m, base_exp, do_pdl, do_trig, stream);
                if (bm == 48) return dispatch_bf11_v5_c32_sub2<2, 3, 1>(X, SLABS, SLAB_META_V5, SFLIPS, BMASKS, OUTLIERS, O_OFF, O, NLIVE_PTR, M, N, N_active, K, sx, KS, split_packed, so_m, base_exp, do_pdl, do_trig, stream);
                return dispatch_bf11_v5_c32_sub2<2, 4, 1>(X, SLABS, SLAB_META_V5, SFLIPS, BMASKS, OUTLIERS, O_OFF, O, NLIVE_PTR, M, N, N_active, K, sx, KS, split_packed, so_m, base_exp, do_pdl, do_trig, stream);
            }
        }
    }

    int warps_pack = 0;
    if (cfg_id == 1) warps_pack = 4;
    else if (cfg_id == 4) warps_pack = 2;
    else if (cfg_id == 5) warps_pack = 1;
    else return static_cast<cudaError_t>(-1);

    const int bn_pack = warps_pack * 8;
    const int n_tiles_pack = (N + bn_pack - 1) / bn_pack;
    const int w_global = n_tiles_pack * warps_pack;
    const int w_active = (N_active > 0) ? ((N_active + 7) / 8) : 0;

    const bool can_split = (ko == 1 && P != nullptr && SEM != nullptr && (split_packed == 6 || split_packed == 9));
    int split_cta = 1;
    if (can_split) {
        if (split_cta_req == 1) {
            split_cta = 1;
        } else if (split_cta_req == 2 || split_cta_req == split_packed) {
            split_cta = split_packed;
        } else if (w_active <= 64 || M <= 128) {
            split_cta = split_packed;
        }
    }

    int bm = (bm_req == 32 || bm_req == 48 || bm_req == 64 || bm_req == 128) ? bm_req : 0;
    int w_run = (w_run_req == 1 || w_run_req == 2 || w_run_req == 4) ? w_run_req : 0;

    if (split_cta == 9) {
        if (bm == 0) {
            if (M <= 32) bm = 32;
            else if (allow_auto_bm48 && M <= 48) bm = 48;
            else bm = 64;
        }
        if (bm == 128) bm = 64;
        if (w_run == 0) w_run = (bm == 32 && M <= 32 && w_active <= 16) ? 1 : 2;
        if (bm >= 48 && w_run == 1) w_run = 2;
    } else if (split_cta == 6) {
        if (bm == 0) {
            if (M <= 32) bm = 32;
            else if (allow_auto_bm48 && M <= 48) bm = 48;
            else if (M <= 96) bm = 64;
            else if (M <= 128) bm = 128;
            else bm = 64;
        }
        if (w_run == 0) w_run = (M <= 64 && bm <= 64) ? 2 : 4;
        if (w_run == 1) w_run = 2;
        if (bm == 128 && w_run != 4) w_run = 4;
    } else {
        if (bm == 0) {
            if (w_active >= 400) {
                if (M <= 32) bm = 32;
                else if (allow_auto_bm48 && M <= 48) bm = 48;
                else if (M <= 64) bm = 64;
                else if (M <= 128 && ko == 0) bm = 128;
                else bm = 64;
            } else if (w_active <= 32) {
                bm = (allow_auto_bm48 && M > 32 && M <= 48) ? 48 : 32;
            } else {
                bm = (allow_auto_bm48 && M > 32 && M <= 48) ? 48 : ((M <= 128) ? 32 : 64);
            }
        }
        if (ko == 1 && bm == 128) bm = 64;
        if (w_run == 0) {
            if (w_active >= 400) {
                w_run = (bm == 32 && M <= 32) ? 2 : 4;
            } else if (w_active <= 32) {
                w_run = (bm >= 48) ? 2 : 1;
            } else {
                w_run = (M <= 32 && bm == 32) ? 2 : 4;
            }
        }
        if (bm >= 48 && w_run == 1) w_run = 2;
    }

    #define CALL_V12S1(W, MT, BK, KO, SP) \
        return dispatch_bf11_v5_c32_sub1<W, MT, BK, KO, SP>( \
            X, SLABS, SLAB_META_V5, SFLIPS, BMASKS, OUTLIERS, O_OFF, P, SEM, O, NLIVE_PTR, \
            M, N, N_active, K, sx, KS, split_packed, so_m, w_global, base_exp, do_pdl, do_trig, stream)

    if (split_cta == 9) {
        if (bm == 32) {
            if (w_run == 1)      CALL_V12S1(1, 2, 128, 1, 9);
            else if (w_run == 2) CALL_V12S1(2, 2, 128, 1, 9);
            else                 CALL_V12S1(4, 2, 128, 1, 9);
        } else if (bm == 48) {
            if (w_run == 2)      CALL_V12S1(2, 3, 128, 1, 9);
            else                 CALL_V12S1(4, 3, 128, 1, 9);
        } else {
            if (w_run == 2)      CALL_V12S1(2, 4, 128, 1, 9);
            else                 CALL_V12S1(4, 4, 128, 1, 9);
        }
    } else if (split_cta == 6) {
        if (bm == 32) {
            if (w_run == 2)      CALL_V12S1(2, 2, 128, 1, 6);
            else                 CALL_V12S1(4, 2, 128, 1, 6);
        } else if (bm == 48) {
            if (w_run == 2)      CALL_V12S1(2, 3, 128, 1, 6);
            else                 CALL_V12S1(4, 3, 128, 1, 6);
        } else if (bm == 64) {
            if (w_run == 2)      CALL_V12S1(2, 4, 128, 1, 6);
            else                 CALL_V12S1(4, 4, 128, 1, 6);
        } else {
            CALL_V12S1(4, 8, 64, 1, 6);
        }
    } else {
        if (ko == 0) {
            if (bm == 32) {
                if (w_run == 1)      CALL_V12S1(1, 2, 128, 0, 1);
                else if (w_run == 2) CALL_V12S1(2, 2, 128, 0, 1);
                else                 CALL_V12S1(4, 2, 128, 0, 1);
            } else if (bm == 48) {
                if (w_run == 2)      CALL_V12S1(2, 3, 128, 0, 1);
                else                 CALL_V12S1(4, 3, 128, 0, 1);
            } else if (bm == 64) {
                if (w_run == 2)      CALL_V12S1(2, 4, 128, 0, 1);
                else                 CALL_V12S1(4, 4, 128, 0, 1);
            } else {
                if (w_run == 2)      CALL_V12S1(2, 8, 64, 0, 1);
                else                 CALL_V12S1(4, 8, 64, 0, 1);
            }
        } else {
            if (bm == 32) {
                if (w_run == 1)      CALL_V12S1(1, 2, 128, 1, 1);
                else if (w_run == 2) CALL_V12S1(2, 2, 128, 1, 1);
                else                 CALL_V12S1(4, 2, 128, 1, 1);
            } else if (bm == 48) {
                if (w_run == 2)      CALL_V12S1(2, 3, 128, 1, 1);
                else                 CALL_V12S1(4, 3, 128, 1, 1);
            } else {
                if (w_run == 2)      CALL_V12S1(2, 4, 128, 1, 1);
                else                 CALL_V12S1(4, 4, 128, 1, 1);
            }
        }
    }
    #undef CALL_V12S1
}

static cudaError_t launch_bf11_v12_dual_kv_c32_impl(
    const void* X0, const void* X1,
    const void* SLABS0, const void* SLABS1,
    const void* SMETA0, const void* SMETA1,
    const void* SFLIPS0, const void* SFLIPS1,
    const void* BMASKS0, const void* BMASKS1,
    const void* OUTLIERS0, const void* OUTLIERS1,
    const void* O_OFF0, const void* O_OFF1,
    void* P, void* SEM, void* O0, void* O1, const void* NLIVE_PTR,
    int M, int N_each, int K, int sx0, int sx1, int KS, int split_packed, int so0, int so1,
    int cfg_id, int bm_req, int w_run_req, int split_cta_req, int k_order,
    uint32_t base_exp0, uint32_t base_exp1, int pdl, int trig, bool allow_auto_bm48, cudaStream_t stream
) {
    if (M <= 0 || N_each <= 0 || KS <= 0 || (KS & 127) != 0 || split_packed <= 0 || KS * split_packed != K) {
        return static_cast<cudaError_t>(-2);
    }
    int wp = 0;
    if (cfg_id == 1) wp = 4;
    else if (cfg_id == 4) wp = 2;
    else if (cfg_id == 5) wp = 1;
    else return static_cast<cudaError_t>(-1);

    const int bn_p = wp * 8;
    const int w_global_each = ((N_each + bn_p - 1) / bn_p) * wp;
    const bool do_pdl = (pdl != 0);
    const bool do_trig = (trig != 0);
    const int ko = (split_packed == 1) ? 0 : ((k_order != 0) ? 1 : 0);

    const bool can_split = (ko == 1 && split_packed == 9 && P != nullptr && SEM != nullptr);
    int split_cta = (can_split && split_cta_req != 1) ? 9 : 1;

    int bm = (bm_req == 32 || bm_req == 48 || bm_req == 64) ? bm_req : 0;
    if (bm == 0) {
        if (M <= 32) bm = 32;
        else if (allow_auto_bm48 && M <= 48) bm = 48;
        else bm = (split_cta == 9) ? 64 : 32;
    }
    int w_run = (w_run_req == 1 || w_run_req == 2 || w_run_req == 4) ? w_run_req : 0;
    if (w_run == 0) {
        w_run = (split_cta == 9) ? 2 : ((M <= 64 && bm == 32) ? 1 : 2);
    }
    if (bm >= 48 && w_run == 1) w_run = 2;

    #define CALL_DKV12(W, MT, KO, SP) \
        return dispatch_bf11_v5_dual_kv_c32<W, MT, KO, SP>( \
            X0, X1, SLABS0, SLABS1, SMETA0, SMETA1, SFLIPS0, SFLIPS1, BMASKS0, BMASKS1, \
            OUTLIERS0, OUTLIERS1, O_OFF0, O_OFF1, P, SEM, O0, O1, NLIVE_PTR, \
            M, N_each, K, sx0, sx1, KS, split_packed, so0, so1, w_global_each, \
            base_exp0, base_exp1, do_pdl, do_trig, stream)

    if (split_cta == 9) {
        if (bm == 32) {
            if (w_run == 1)      CALL_DKV12(1, 2, 1, 9);
            else if (w_run == 2) CALL_DKV12(2, 2, 1, 9);
            else                 CALL_DKV12(4, 2, 1, 9);
        } else if (bm == 48) {
            if (w_run == 2)      CALL_DKV12(2, 3, 1, 9);
            else                 CALL_DKV12(4, 3, 1, 9);
        } else {
            if (w_run == 2)      CALL_DKV12(2, 4, 1, 9);
            else                 CALL_DKV12(4, 4, 1, 9);
        }
    } else {
        if (ko == 0) {
            if (bm == 32) {
                if (w_run == 1)      CALL_DKV12(1, 2, 0, 1);
                else if (w_run == 2) CALL_DKV12(2, 2, 0, 1);
                else                 CALL_DKV12(4, 2, 0, 1);
            } else if (bm == 48) {
                if (w_run == 2)      CALL_DKV12(2, 3, 0, 1);
                else                 CALL_DKV12(4, 3, 0, 1);
            } else {
                if (w_run == 2)      CALL_DKV12(2, 4, 0, 1);
                else                 CALL_DKV12(4, 4, 0, 1);
            }
        } else {
            if (bm == 32) {
                if (w_run == 1)      CALL_DKV12(1, 2, 1, 1);
                else if (w_run == 2) CALL_DKV12(2, 2, 1, 1);
                else                 CALL_DKV12(4, 2, 1, 1);
            } else if (bm == 48) {
                if (w_run == 2)      CALL_DKV12(2, 3, 1, 1);
                else                 CALL_DKV12(4, 3, 1, 1);
            } else {
                if (w_run == 2)      CALL_DKV12(2, 4, 1, 1);
                else                 CALL_DKV12(4, 4, 1, 1);
            }
        }
    }
    #undef CALL_DKV12
}

static cudaError_t launch_bf11_v12_dual_kv_dyn_c32_impl(
    const void* X0, const void* X1,
    const void* SLABS0, const void* SLABS1,
    const void* SMETA0, const void* SMETA1,
    const void* SFLIPS0, const void* SFLIPS1,
    const void* BMASKS0, const void* BMASKS1,
    const void* OUTLIERS0, const void* OUTLIERS1,
    const void* O_OFF0, const void* O_OFF1,
    void* P, void* SEM, void* O0, void* O1,
    const void* W_DYN0, const void* W_DYN1, void* O_DYN0, void* O_DYN1,
    const void* NLIVE_PTR,
    int M, int N_each, int N_dyn, int K, int sx0, int sx1, int KS, int split_packed,
    int so0, int so1, int so_dyn0, int so_dyn1,
    int cfg_id, int bm_req, int w_run_req, int split_cta_req, int k_order,
    uint32_t base_exp0, uint32_t base_exp1, int dyn_mode, int dyn_split_k,
    int pdl, int trig, bool allow_auto_bm48, cudaStream_t stream
) {
    if (W_DYN0 == nullptr || W_DYN1 == nullptr || O_DYN0 == nullptr || O_DYN1 == nullptr || N_dyn <= 0) {
        return launch_bf11_v12_dual_kv_c32_impl(
            X0, X1, SLABS0, SLABS1, SMETA0, SMETA1, SFLIPS0, SFLIPS1, BMASKS0, BMASKS1,
            OUTLIERS0, OUTLIERS1, O_OFF0, O_OFF1, P, SEM, O0, O1, NLIVE_PTR,
            M, N_each, K, sx0, sx1, KS, split_packed, so0, so1, cfg_id, bm_req, w_run_req,
            split_cta_req, k_order, base_exp0, base_exp1, pdl, trig, allow_auto_bm48, stream
        );
    }
    if (M <= 0 || N_each <= 0 || KS <= 0 || (KS & 127) != 0 || split_packed <= 0 || KS * split_packed != K) {
        return static_cast<cudaError_t>(-2);
    }
    if (!dyn_params_valid(N_dyn, K, dyn_mode, dyn_split_k)) return static_cast<cudaError_t>(-2);
    int wp = 0;
    if (cfg_id == 1) wp = 4;
    else if (cfg_id == 4) wp = 2;
    else if (cfg_id == 5) wp = 1;
    else return static_cast<cudaError_t>(-1);

    const int bn_p = wp * 8;
    const int w_global_each = ((N_each + bn_p - 1) / bn_p) * wp;
    const bool do_pdl = (pdl != 0);
    const bool do_trig = (trig != 0);
    const int ko = (split_packed == 1) ? 0 : ((k_order != 0) ? 1 : 0);

    const bool can_split = (ko == 1 && split_packed == 9 && P != nullptr && SEM != nullptr);
    int split_cta = (can_split && split_cta_req != 1) ? 9 : 1;

    int bm = (bm_req == 32 || bm_req == 48 || bm_req == 64) ? bm_req : 0;
    if (bm == 0) {
        if (M <= 32) bm = 32;
        else if (allow_auto_bm48 && M <= 48) bm = 48;
        else bm = (split_cta == 9) ? 64 : 32;
    }
    int w_run = (w_run_req == 1 || w_run_req == 2 || w_run_req == 4) ? w_run_req : 0;
    if (w_run == 0) {
        w_run = (split_cta == 9) ? 2 : ((M <= 64 && bm == 32) ? 1 : 2);
    }
    if (bm >= 48 && w_run == 1) w_run = 2;

    #define CALL_DKVD12(W, MT, KO, SP) \
        return dispatch_bf11_v9_dual_kv_dyn_c32<W, MT, KO, SP>( \
            X0, X1, SLABS0, SLABS1, SMETA0, SMETA1, SFLIPS0, SFLIPS1, BMASKS0, BMASKS1, \
            OUTLIERS0, OUTLIERS1, O_OFF0, O_OFF1, P, SEM, O0, O1, \
            W_DYN0, W_DYN1, O_DYN0, O_DYN1, NLIVE_PTR, \
            M, N_each, N_dyn, K, sx0, sx1, KS, split_packed, so0, so1, so_dyn0, so_dyn1, \
            w_global_each, base_exp0, base_exp1, dyn_mode, dyn_split_k, do_pdl, do_trig, stream)

    if (split_cta == 9) {
        if (bm == 32) {
            if (w_run == 1)      CALL_DKVD12(1, 2, 1, 9);
            else if (w_run == 2) CALL_DKVD12(2, 2, 1, 9);
            else                 CALL_DKVD12(4, 2, 1, 9);
        } else if (bm == 48) {
            if (w_run == 2)      CALL_DKVD12(2, 3, 1, 9);
            else                 CALL_DKVD12(4, 3, 1, 9);
        } else {
            if (w_run == 2)      CALL_DKVD12(2, 4, 1, 9);
            else                 CALL_DKVD12(4, 4, 1, 9);
        }
    } else {
        if (ko == 0) {
            if (bm == 32) {
                if (w_run == 1)      CALL_DKVD12(1, 2, 0, 1);
                else if (w_run == 2) CALL_DKVD12(2, 2, 0, 1);
                else                 CALL_DKVD12(4, 2, 0, 1);
            } else if (bm == 48) {
                if (w_run == 2)      CALL_DKVD12(2, 3, 0, 1);
                else                 CALL_DKVD12(4, 3, 0, 1);
            } else {
                if (w_run == 2)      CALL_DKVD12(2, 4, 0, 1);
                else                 CALL_DKVD12(4, 4, 0, 1);
            }
        } else {
            if (bm == 32) {
                if (w_run == 1)      CALL_DKVD12(1, 2, 1, 1);
                else if (w_run == 2) CALL_DKVD12(2, 2, 1, 1);
                else                 CALL_DKVD12(4, 2, 1, 1);
            } else if (bm == 48) {
                if (w_run == 2)      CALL_DKVD12(2, 3, 1, 1);
                else                 CALL_DKVD12(4, 3, 1, 1);
            } else {
                if (w_run == 2)      CALL_DKVD12(2, 4, 1, 1);
                else                 CALL_DKVD12(4, 4, 1, 1);
            }
        }
    }
    #undef CALL_DKVD12
}

}  // namespace

// =========================================================================
// PART 4: EXTERN "C" ABI (100% DROP-IN SUPERSET OF libbf11_flip_exact_c32.so)
// =========================================================================

extern "C" int bf11_flip_c32_version() { return 5; }
extern "C" int bf11_flip_v5_version() { return 5; }

// ---- Zero-Raw-Weight v5 Sidecar Builders (`FREE_RAW=1` Safe!) ----
extern "C" int bf11_v5_measure_sidecar_from_bf11(
    const void* WARP_META,
    void* W_S_COUNTS, void* W_D_COUNTS,
    int wtot, int spw, int flip_thresh,
    uintptr_t stream_ptr
) {
    if (wtot <= 0 || spw <= 0 || spw > 12) return -2;
    const int thresh = (flip_thresh < 0) ? 0 : ((flip_thresh > 63) ? 63 : flip_thresh);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    dim3 block(128, 1, 1);
    dim3 grid((wtot + 127) / 128, 1, 1);
    bf11_v5_measure_sidecar_kernel<<<grid, block, 0, stream>>>(
        static_cast<const uint4*>(WARP_META),
        static_cast<int32_t*>(W_S_COUNTS),
        static_cast<int32_t*>(W_D_COUNTS),
        wtot, spw, thresh
    );
    return static_cast<int>(cudaGetLastError());
}

extern "C" int bf11_v5_build_sidecar_from_bf11(
    const void* FLIPS, const void* WARP_META,
    const void* OUTLIERS, const void* O_OFF,
    const void* W_S_OFF, const void* W_D_OFF,
    void* SLAB_META_V5, void* SFLIPS, void* BMASKS,
    int wtot, int spw, int flip_thresh,
    uintptr_t stream_ptr
) {
    if (wtot <= 0 || spw <= 0 || spw > 12) return -2;
    const int thresh = (flip_thresh < 0) ? 0 : ((flip_thresh > 63) ? 63 : flip_thresh);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    dim3 block(32, 1, 1);
    dim3 grid(wtot, 1, 1);
    bf11_v5_build_sidecar_kernel<<<grid, block, 0, stream>>>(
        static_cast<const uint16_t*>(FLIPS),
        static_cast<const uint4*>(WARP_META),
        static_cast<const uint64_t*>(OUTLIERS),
        static_cast<const int32_t*>(O_OFF),
        static_cast<const int32_t*>(W_S_OFF),
        static_cast<const int32_t*>(W_D_OFF),
        static_cast<uint32_t*>(SLAB_META_V5),
        static_cast<uint16_t*>(SFLIPS),
        static_cast<uint32_t*>(BMASKS),
        wtot, spw, thresh
    );
    return static_cast<int>(cudaGetLastError());
}

// ---- NEW v5 C=1 (M <= 16) Roofline Entry Points ----
extern "C" int bf11_flip_v5_c1_fwd(
    const void* X, const void* SLABS,
    const void* SLAB_META_V5, const void* SFLIPS, const void* BMASKS,
    const void* OUTLIERS, const void* O_OFF,
    void* P, void* SEM, void* O, const void* NLIVE_PTR,
    int M, int N, int N_active, int K, int sx, int KS, int so_m,
    int cfg_id, int w_run_req, int split, uint32_t base_exp,
    int pdl, int trig, uintptr_t stream_ptr
) {
    if (M <= 0 || N <= 0 || KS <= 0 || (KS & 127) != 0) return -2;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    const bool do_pdl = (pdl != 0);
    const bool do_trig = (trig != 0);
    const bool is_sub2 = (cfg_id == 0 || cfg_id == 2);
    cudaError_t err = cudaSuccess;

    #define CASE_SP_V5C1(W, S, WG) \
        switch (split) { \
            case 1: err = dispatch_bf11_v5_c1<W, S, 1>(X, SLABS, SLAB_META_V5, SFLIPS, BMASKS, OUTLIERS, O_OFF, P, SEM, O, NLIVE_PTR, M, N, N_active, K, sx, KS, so_m, WG, base_exp, do_pdl, do_trig, stream); break; \
            case 6: err = dispatch_bf11_v5_c1<W, S, 6>(X, SLABS, SLAB_META_V5, SFLIPS, BMASKS, OUTLIERS, O_OFF, P, SEM, O, NLIVE_PTR, M, N, N_active, K, sx, KS, so_m, WG, base_exp, do_pdl, do_trig, stream); break; \
            case 9: err = dispatch_bf11_v5_c1<W, S, 9>(X, SLABS, SLAB_META_V5, SFLIPS, BMASKS, OUTLIERS, O_OFF, P, SEM, O, NLIVE_PTR, M, N, N_active, K, sx, KS, so_m, WG, base_exp, do_pdl, do_trig, stream); break; \
            default: return -2; \
        }

    if (is_sub2) {
        const int wp = (cfg_id == 0) ? 4 : 2;
        const int bn_p = wp * 16;
        const int w_global = ((N + bn_p - 1) / bn_p) * wp;
        if (cfg_id == 0) { CASE_SP_V5C1(4, 2, w_global); }
        else             { CASE_SP_V5C1(2, 2, w_global); }
    } else {
        int wp = 0;
        if (cfg_id == 1) wp = 4;
        else if (cfg_id == 4) wp = 2;
        else if (cfg_id == 5) wp = 1;
        else return -1;
        const int bn_p = wp * 8;
        const int w_global = ((N + bn_p - 1) / bn_p) * wp;
        int w_run = (w_run_req == 1 || w_run_req == 2 || w_run_req == 4) ? w_run_req : wp;
        if (w_run == 4)      { CASE_SP_V5C1(4, 1, w_global); }
        else if (w_run == 2) { CASE_SP_V5C1(2, 1, w_global); }
        else                 { CASE_SP_V5C1(1, 1, w_global); }
    }
    #undef CASE_SP_V5C1
    return static_cast<int>(err);
}

extern "C" int bf11_flip_dual_kv_v5_c1_fwd(
    const void* X0, const void* X1,
    const void* SLABS0, const void* SLABS1,
    const void* SMETA0, const void* SMETA1,
    const void* SFLIPS0, const void* SFLIPS1,
    const void* BMASKS0, const void* BMASKS1,
    const void* OUTLIERS0, const void* OUTLIERS1,
    const void* O_OFF0, const void* O_OFF1,
    void* P, void* SEM, void* O0, void* O1, const void* NLIVE_PTR,
    int M, int N_each, int K, int sx0, int sx1, int KS, int so0, int so1,
    int cfg_id, int w_run_req, int split, uint32_t base_exp0, uint32_t base_exp1,
    int pdl, int trig, uintptr_t stream_ptr
) {
    if (split != 9 || M <= 0 || N_each <= 0) return -2;
    int wp = 0;
    if (cfg_id == 1) wp = 4;
    else if (cfg_id == 4) wp = 2;
    else if (cfg_id == 5) wp = 1;
    else return -1;
    const int bn_p = wp * 8;
    const int w_global_each = ((N_each + bn_p - 1) / bn_p) * wp;
    const int w_run = (w_run_req == 1 || w_run_req == 2 || w_run_req == 4) ? w_run_req : wp;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    const bool m_le8 = (M <= 8);
    const bool do_pdl = (pdl != 0);
    const bool do_trig = (trig != 0);
    const size_t smem = static_cast<size_t>(m_le8 ? 8 : 16) * (KS + 8) * sizeof(__nv_bfloat16);

    #define LAUNCH_DKV_V5C1(W, ML, PD, TR) \
        launch_maybe_pdl(bf11_flip_dual_kv_v5_c1_kernel<W, ML, 9, PD, TR>, \
            dim3(2 * ((w_global_each + (W) - 1) / (W)), 9, 1), dim3((W) * 32, 1, 1), smem, stream, PD, \
            static_cast<const __nv_bfloat16*>(X0), static_cast<const __nv_bfloat16*>(X1), \
            static_cast<const uint8_t*>(SLABS0), static_cast<const uint8_t*>(SLABS1), \
            static_cast<const uint32_t*>(SMETA0), static_cast<const uint32_t*>(SMETA1), \
            static_cast<const uint16_t*>(SFLIPS0), static_cast<const uint16_t*>(SFLIPS1), \
            static_cast<const uint32_t*>(BMASKS0), static_cast<const uint32_t*>(BMASKS1), \
            static_cast<const uint64_t*>(OUTLIERS0), static_cast<const uint64_t*>(OUTLIERS1), \
            static_cast<const int32_t*>(O_OFF0), static_cast<const int32_t*>(O_OFF1), \
            static_cast<float*>(P), static_cast<int32_t*>(SEM), \
            static_cast<__nv_bfloat16*>(O0), static_cast<__nv_bfloat16*>(O1), static_cast<const int32_t*>(NLIVE_PTR), \
            M, N_each, K, sx0, sx1, KS, so0, so1, w_global_each, base_exp0, base_exp1)
    #define DISP_DKV_V5C1(W) \
        if (m_le8) { \
            if (do_pdl) return static_cast<int>(do_trig ? LAUNCH_DKV_V5C1(W, true, true, true) : LAUNCH_DKV_V5C1(W, true, true, false)); \
            return static_cast<int>(do_trig ? LAUNCH_DKV_V5C1(W, true, false, true) : LAUNCH_DKV_V5C1(W, true, false, false)); \
        } else { \
            if (do_pdl) return static_cast<int>(do_trig ? LAUNCH_DKV_V5C1(W, false, true, true) : LAUNCH_DKV_V5C1(W, false, true, false)); \
            return static_cast<int>(do_trig ? LAUNCH_DKV_V5C1(W, false, false, true) : LAUNCH_DKV_V5C1(W, false, false, false)); \
        }

    if (w_run == 4)      { DISP_DKV_V5C1(4); }
    else if (w_run == 2) { DISP_DKV_V5C1(2); }
    else                 { DISP_DKV_V5C1(1); }
    #undef DISP_DKV_V5C1
    #undef LAUNCH_DKV_V5C1
}

// ---- NEW v5 C=32 (M = 17..256+) Roofline Entry Points ----
extern "C" int bf11_flip_v5_c32_fwd(
    const void* X, const void* SLABS,
    const void* SLAB_META_V5, const void* SFLIPS, const void* BMASKS,
    const void* OUTLIERS, const void* O_OFF,
    void* P, void* SEM, void* O, const void* NLIVE_PTR,
    int M, int N, int N_active, int K, int sx, int KS, int split_packed, int so_m,
    int cfg_id, int bm_req, int w_run_req, int split_cta_req, int k_order,
    uint32_t base_exp, int pdl, int trig, uintptr_t stream_ptr
) {
    return static_cast<int>(launch_bf11_v12_c32_impl(
        X, SLABS, SLAB_META_V5, SFLIPS, BMASKS, OUTLIERS, O_OFF,
        P, SEM, O, NLIVE_PTR,
        M, N, N_active, K, sx, KS, split_packed, so_m,
        cfg_id, bm_req, w_run_req, split_cta_req, k_order,
        base_exp, pdl, trig, false, reinterpret_cast<cudaStream_t>(stream_ptr)
    ));
}

extern "C" int bf11_flip_dual_kv_v5_c32_fwd(
    const void* X0, const void* X1,
    const void* SLABS0, const void* SLABS1,
    const void* SMETA0, const void* SMETA1,
    const void* SFLIPS0, const void* SFLIPS1,
    const void* BMASKS0, const void* BMASKS1,
    const void* OUTLIERS0, const void* OUTLIERS1,
    const void* O_OFF0, const void* O_OFF1,
    void* P, void* SEM, void* O0, void* O1, const void* NLIVE_PTR,
    int M, int N_each, int K, int sx0, int sx1, int KS, int split_packed, int so0, int so1,
    int cfg_id, int bm_req, int w_run_req, int split_cta_req, int k_order,
    uint32_t base_exp0, uint32_t base_exp1, int pdl, int trig, uintptr_t stream_ptr
) {
    return static_cast<int>(launch_bf11_v12_dual_kv_c32_impl(
        X0, X1, SLABS0, SLABS1, SMETA0, SMETA1, SFLIPS0, SFLIPS1, BMASKS0, BMASKS1,
        OUTLIERS0, OUTLIERS1, O_OFF0, O_OFF1,
        P, SEM, O0, O1, NLIVE_PTR,
        M, N_each, K, sx0, sx1, KS, split_packed, so0, so1,
        cfg_id, bm_req, w_run_req, split_cta_req, k_order,
        base_exp0, base_exp1, pdl, trig, false, reinterpret_cast<cudaStream_t>(stream_ptr)
    ));
}

// ---- Drop-In Legacy ABI Entry Points (100% Identical Signatures to libbf11_flip_exact_c32.so) ----
extern "C" int bf11_flip_fwd(
    const void* X, const void* SLABS, const void* FLIPS, const void* WARP_META,
    const void* OUTLIERS, const void* O_OFF,
    void* P, void* SEM, void* O, const void* NLIVE_PTR,
    int M, int N, int N_active, int K, int sx, int KS, int so_m,
    int cfg_id, int split, uint32_t base_exp,
    int pdl, int trig, uintptr_t stream_ptr
) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    const bool do_pdl = (pdl != 0);
    const bool do_trig = (trig != 0);
    cudaError_t err = cudaSuccess;
    #define CASE_SP_LEG(W, S) \
        switch (split) { \
            case 1: err = dispatch_bf11_flip_legacy<W, S, 1>(X, SLABS, FLIPS, WARP_META, OUTLIERS, O_OFF, P, SEM, O, NLIVE_PTR, M, N, N_active, K, sx, KS, so_m, base_exp, do_pdl, do_trig, stream); break; \
            case 6: err = dispatch_bf11_flip_legacy<W, S, 6>(X, SLABS, FLIPS, WARP_META, OUTLIERS, O_OFF, P, SEM, O, NLIVE_PTR, M, N, N_active, K, sx, KS, so_m, base_exp, do_pdl, do_trig, stream); break; \
            case 9: err = dispatch_bf11_flip_legacy<W, S, 9>(X, SLABS, FLIPS, WARP_META, OUTLIERS, O_OFF, P, SEM, O, NLIVE_PTR, M, N, N_active, K, sx, KS, so_m, base_exp, do_pdl, do_trig, stream); break; \
            default: return -2; \
        }
    switch (cfg_id) {
        case 0: CASE_SP_LEG(4, 2); break;
        case 1: CASE_SP_LEG(4, 1); break;
        case 2: CASE_SP_LEG(2, 2); break;
        case 4: CASE_SP_LEG(2, 1); break;
        case 5: CASE_SP_LEG(1, 1); break;
        default: return -1;
    }
    #undef CASE_SP_LEG
    return static_cast<int>(err);
}

extern "C" int bf11_flip_dual_kv_fwd(
    const void* X0, const void* X1,
    const void* SLABS0, const void* SLABS1,
    const void* FLIPS0, const void* FLIPS1,
    const void* META0, const void* META1,
    const void* OUTLIERS0, const void* OUTLIERS1,
    const void* O_OFF0, const void* O_OFF1,
    void* P, void* SEM, void* O0, void* O1, const void* NLIVE_PTR,
    int M, int N_each, int K, int sx0, int sx1, int KS, int so0, int so1,
    int cfg_id, int split, uint32_t base_exp0, uint32_t base_exp1,
    int pdl, int trig, uintptr_t stream_ptr
) {
    if (split != 9) return -2;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    const bool m_le8 = (M <= 8);
    const bool do_pdl = (pdl != 0);
    const bool do_trig = (trig != 0);
    const size_t smem = static_cast<size_t>(m_le8 ? 8 : 16) * (KS + 8) * sizeof(__nv_bfloat16);
    #define LAUNCH_KV_LEG(W, S, ML, PD, TR) \
        launch_maybe_pdl(bf11_flip_dual_kv_kernel<W, S, ML, 9, PD, TR>, \
            dim3(2 * ((N_each + (W * S * 8) - 1) / (W * S * 8)), 9, 1), dim3(W * 32, 1, 1), smem, stream, PD, \
            static_cast<const __nv_bfloat16*>(X0), static_cast<const __nv_bfloat16*>(X1), \
            static_cast<const uint8_t*>(SLABS0), static_cast<const uint8_t*>(SLABS1), \
            static_cast<const uint16_t*>(FLIPS0), static_cast<const uint16_t*>(FLIPS1), \
            static_cast<const uint4*>(META0), static_cast<const uint4*>(META1), \
            static_cast<const uint64_t*>(OUTLIERS0), static_cast<const uint64_t*>(OUTLIERS1), \
            static_cast<const int32_t*>(O_OFF0), static_cast<const int32_t*>(O_OFF1), \
            static_cast<float*>(P), static_cast<int32_t*>(SEM), \
            static_cast<__nv_bfloat16*>(O0), static_cast<__nv_bfloat16*>(O1), static_cast<const int32_t*>(NLIVE_PTR), \
            M, N_each, K, sx0, sx1, KS, so0, so1, base_exp0, base_exp1)
    #define DISP_KV_LEG(W, S) \
        if (m_le8) { \
            if (do_pdl) return static_cast<int>(do_trig ? LAUNCH_KV_LEG(W, S, true, true, true) : LAUNCH_KV_LEG(W, S, true, true, false)); \
            return static_cast<int>(do_trig ? LAUNCH_KV_LEG(W, S, true, false, true) : LAUNCH_KV_LEG(W, S, true, false, false)); \
        } else { \
            if (do_pdl) return static_cast<int>(do_trig ? LAUNCH_KV_LEG(W, S, false, true, true) : LAUNCH_KV_LEG(W, S, false, true, false)); \
            return static_cast<int>(do_trig ? LAUNCH_KV_LEG(W, S, false, false, true) : LAUNCH_KV_LEG(W, S, false, false, false)); \
        }
    switch (cfg_id) {
        case 1: DISP_KV_LEG(4, 1);
        case 4: DISP_KV_LEG(2, 1);
        case 5: DISP_KV_LEG(1, 1);
        default: return -1;
    }
    #undef DISP_KV_LEG
    #undef LAUNCH_KV_LEG
}

extern "C" int bf11_flip_c32_fwd(
    const void* X, const void* SLABS, const void* FLIPS, const void* WARP_META,
    const void* OUTLIERS, const void* O_OFF,
    void* O, const void* NLIVE_PTR,
    int M, int N, int N_active, int K, int sx, int KS, int split_packed, int so_m,
    int cfg_id, int bm_req, int k_order, uint32_t base_exp,
    int pdl, int trig, uintptr_t stream_ptr
) {
    if (M <= 0 || KS <= 0 || (KS & 127) != 0 || split_packed <= 0 || KS * split_packed != K) {
        return -2;
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    const bool do_pdl = (pdl != 0);
    const bool do_trig = (trig != 0);
    const bool is_sub2 = (cfg_id == 0 || cfg_id == 2);
    const int ko = (split_packed == 1) ? 0 : ((k_order != 0) ? 1 : 0);

    int bm = bm_req;
    if (bm != 32 && bm != 64 && bm != 128) {
        if (M <= 32) {
            bm = 32;
        } else if (M <= 64) {
            bm = 64;
        } else if (!is_sub2 && ko == 0 && (N >= 4096 || (cfg_id == 4 && M <= 128))) {
            bm = 128;
        } else {
            bm = 64;
        }
    }
    if (is_sub2 && bm > 64) bm = 64;
    if (ko == 1 && bm > 64) bm = 64;

    #define DISP_S1_LEG(W) \
        if (ko == 0) { \
            if (bm == 32)      return static_cast<int>(dispatch_bf11_c32_sub1_legacy<W, 2, 0>(X, SLABS, FLIPS, WARP_META, OUTLIERS, O_OFF, O, NLIVE_PTR, M, N, N_active, K, sx, KS, split_packed, so_m, base_exp, do_pdl, do_trig, stream)); \
            else if (bm == 64) return static_cast<int>(dispatch_bf11_c32_sub1_legacy<W, 4, 0>(X, SLABS, FLIPS, WARP_META, OUTLIERS, O_OFF, O, NLIVE_PTR, M, N, N_active, K, sx, KS, split_packed, so_m, base_exp, do_pdl, do_trig, stream)); \
            else               return static_cast<int>(dispatch_bf11_c32_sub1_legacy<W, 8, 0>(X, SLABS, FLIPS, WARP_META, OUTLIERS, O_OFF, O, NLIVE_PTR, M, N, N_active, K, sx, KS, split_packed, so_m, base_exp, do_pdl, do_trig, stream)); \
        } else { \
            if (bm == 32)      return static_cast<int>(dispatch_bf11_c32_sub1_legacy<W, 2, 1>(X, SLABS, FLIPS, WARP_META, OUTLIERS, O_OFF, O, NLIVE_PTR, M, N, N_active, K, sx, KS, split_packed, so_m, base_exp, do_pdl, do_trig, stream)); \
            else               return static_cast<int>(dispatch_bf11_c32_sub1_legacy<W, 4, 1>(X, SLABS, FLIPS, WARP_META, OUTLIERS, O_OFF, O, NLIVE_PTR, M, N, N_active, K, sx, KS, split_packed, so_m, base_exp, do_pdl, do_trig, stream)); \
        }
    #define DISP_S2_LEG(W) \
        if (ko == 0) { \
            if (bm == 32)      return static_cast<int>(dispatch_bf11_c32_sub2_legacy<W, 2, 0>(X, SLABS, FLIPS, WARP_META, OUTLIERS, O_OFF, O, NLIVE_PTR, M, N, N_active, K, sx, KS, split_packed, so_m, base_exp, do_pdl, do_trig, stream)); \
            else               return static_cast<int>(dispatch_bf11_c32_sub2_legacy<W, 4, 0>(X, SLABS, FLIPS, WARP_META, OUTLIERS, O_OFF, O, NLIVE_PTR, M, N, N_active, K, sx, KS, split_packed, so_m, base_exp, do_pdl, do_trig, stream)); \
        } else { \
            if (bm == 32)      return static_cast<int>(dispatch_bf11_c32_sub2_legacy<W, 2, 1>(X, SLABS, FLIPS, WARP_META, OUTLIERS, O_OFF, O, NLIVE_PTR, M, N, N_active, K, sx, KS, split_packed, so_m, base_exp, do_pdl, do_trig, stream)); \
            else               return static_cast<int>(dispatch_bf11_c32_sub2_legacy<W, 4, 1>(X, SLABS, FLIPS, WARP_META, OUTLIERS, O_OFF, O, NLIVE_PTR, M, N, N_active, K, sx, KS, split_packed, so_m, base_exp, do_pdl, do_trig, stream)); \
        }

    switch (cfg_id) {
        case 0: DISP_S2_LEG(4);
        case 1: DISP_S1_LEG(4);
        case 2: DISP_S2_LEG(2);
        case 4: DISP_S1_LEG(2);
        case 5: DISP_S1_LEG(1);
        default: return -1;
    }
    #undef DISP_S1_LEG
    #undef DISP_S2_LEG
}

extern "C" int bf11_flip_dual_kv_c32_fwd(
    const void* X0, const void* X1,
    const void* SLABS0, const void* SLABS1,
    const void* FLIPS0, const void* FLIPS1,
    const void* META0, const void* META1,
    const void* OUTLIERS0, const void* OUTLIERS1,
    const void* O_OFF0, const void* O_OFF1,
    void* O0, void* O1, const void* NLIVE_PTR,
    int M, int N_each, int K, int sx0, int sx1, int KS, int split_packed, int so0, int so1,
    int cfg_id, int k_order, uint32_t base_exp0, uint32_t base_exp1,
    int pdl, int trig, uintptr_t stream_ptr
) {
    if (M <= 0 || KS <= 0 || (KS & 127) != 0 || split_packed <= 0 || KS * split_packed != K) {
        return -2;
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    const bool do_pdl = (pdl != 0);
    const bool do_trig = (trig != 0);
    const int ko = (k_order != 0) ? 1 : 0;
    constexpr int BM = 32;
    const int m_blocks = (M + BM - 1) / BM;
    const bool cache_l2 = (m_blocks > 1);
    const size_t smem = static_cast<size_t>(2) * BM * 72 * sizeof(__nv_bfloat16);

    #define LAUNCH_DKV_C32_LEG(W, KO, CA, PD, TR) \
        launch_maybe_pdl(bf11_flip_dual_kv_c32_kernel<W, 2, KO, CA, PD, TR>, \
            dim3(m_blocks, 2 * ((N_each + (W * 8) - 1) / (W * 8)), 1), dim3(W * 32, 1, 1), smem, stream, PD, \
            static_cast<const __nv_bfloat16*>(X0), static_cast<const __nv_bfloat16*>(X1), \
            static_cast<const uint8_t*>(SLABS0), static_cast<const uint8_t*>(SLABS1), \
            static_cast<const uint16_t*>(FLIPS0), static_cast<const uint16_t*>(FLIPS1), \
            static_cast<const uint4*>(META0), static_cast<const uint4*>(META1), \
            static_cast<const uint64_t*>(OUTLIERS0), static_cast<const uint64_t*>(OUTLIERS1), \
            static_cast<const int32_t*>(O_OFF0), static_cast<const int32_t*>(O_OFF1), \
            static_cast<__nv_bfloat16*>(O0), static_cast<__nv_bfloat16*>(O1), static_cast<const int32_t*>(NLIVE_PTR), \
            M, N_each, K, sx0, sx1, KS, split_packed, so0, so1, base_exp0, base_exp1)
    #define DISP_DKV_CA_LEG(W, KO) \
        if (cache_l2) { \
            if (do_pdl) return static_cast<int>(do_trig ? LAUNCH_DKV_C32_LEG(W, KO, true, true, true) : LAUNCH_DKV_C32_LEG(W, KO, true, true, false)); \
            return static_cast<int>(do_trig ? LAUNCH_DKV_C32_LEG(W, KO, true, false, true) : LAUNCH_DKV_C32_LEG(W, KO, true, false, false)); \
        } else { \
            if (do_pdl) return static_cast<int>(do_trig ? LAUNCH_DKV_C32_LEG(W, KO, false, true, true) : LAUNCH_DKV_C32_LEG(W, KO, false, true, false)); \
            return static_cast<int>(do_trig ? LAUNCH_DKV_C32_LEG(W, KO, false, false, true) : LAUNCH_DKV_C32_LEG(W, KO, false, false, false)); \
        }
    #define DISP_DKV_LEG(W) \
        if (ko == 0) { DISP_DKV_CA_LEG(W, 0); } else { DISP_DKV_CA_LEG(W, 1); }

    switch (cfg_id) {
        case 1: DISP_DKV_LEG(4);
        case 4: DISP_DKV_LEG(2);
        case 5: DISP_DKV_LEG(1);
        default: return -1;
    }
    #undef DISP_DKV_LEG
    #undef DISP_DKV_CA_LEG
    #undef LAUNCH_DKV_C32_LEG
}


// =========================================================================
// PART 5: WAVE-V9 FUSED `bf11_dual_kv + k_dyn/v_dyn` EXTERN "C" ABI
// =========================================================================

extern "C" int bf11_flip_v9_version() { return 9; }

extern "C" int bf11_dyn_probe_v9_fwd(
    const void* X, const void* W_DYN, void* O_DYN, const void* NLIVE_PTR,
    int M, int N_dyn, int K, int sx, int so_dyn,
    int dyn_mode, int dyn_split_k, uintptr_t stream_ptr
) {
    if (M <= 0 || !dyn_params_valid(N_dyn, K, dyn_mode, dyn_split_k)) return -2;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    if (M <= 16) {
        const size_t smem = static_cast<size_t>(16 * 136 * 2 + 4 * 136 * 2 + 6 * 16 * 4 * 4);
        bf11_dyn_probe_v9_kernel<2, 1><<<dim3(1, 1, 1), dim3(64, 1, 1), smem, stream>>>(
            static_cast<const __nv_bfloat16*>(X), static_cast<const __nv_bfloat16*>(W_DYN),
            static_cast<__nv_bfloat16*>(O_DYN), static_cast<const int32_t*>(NLIVE_PTR),
            M, N_dyn, K, sx, so_dyn, dyn_mode, dyn_split_k
        );
    } else if (M > 32 && M <= 48) {
        const size_t smem = static_cast<size_t>(48 * 136 * 2 + 4 * 136 * 2 + 6 * 48 * 4 * 4);
        bf11_dyn_probe_v9_kernel<2, 3><<<dim3(1, 1, 1), dim3(64, 1, 1), smem, stream>>>(
            static_cast<const __nv_bfloat16*>(X), static_cast<const __nv_bfloat16*>(W_DYN),
            static_cast<__nv_bfloat16*>(O_DYN), static_cast<const int32_t*>(NLIVE_PTR),
            M, N_dyn, K, sx, so_dyn, dyn_mode, dyn_split_k
        );
    } else {
        const int m_blocks = (M + 31) / 32;
        const size_t smem = static_cast<size_t>(32 * 136 * 2 + 4 * 136 * 2 + 6 * 32 * 4 * 4);
        bf11_dyn_probe_v9_kernel<2, 2><<<dim3(m_blocks, 1, 1), dim3(64, 1, 1), smem, stream>>>(
            static_cast<const __nv_bfloat16*>(X), static_cast<const __nv_bfloat16*>(W_DYN),
            static_cast<__nv_bfloat16*>(O_DYN), static_cast<const int32_t*>(NLIVE_PTR),
            M, N_dyn, K, sx, so_dyn, dyn_mode, dyn_split_k
        );
    }
    return static_cast<int>(cudaGetLastError());
}

extern "C" int bf11_flip_dual_kv_dyn_v9_c1_fwd(
    const void* X0, const void* X1,
    const void* SLABS0, const void* SLABS1,
    const void* SMETA0, const void* SMETA1,
    const void* SFLIPS0, const void* SFLIPS1,
    const void* BMASKS0, const void* BMASKS1,
    const void* OUTLIERS0, const void* OUTLIERS1,
    const void* O_OFF0, const void* O_OFF1,
    void* P, void* SEM, void* O0, void* O1,
    const void* W_DYN0, const void* W_DYN1, void* O_DYN0, void* O_DYN1,
    const void* NLIVE_PTR,
    int M, int N_each, int N_dyn, int K, int sx0, int sx1, int KS,
    int so0, int so1, int so_dyn0, int so_dyn1,
    int cfg_id, int w_run_req, int split, uint32_t base_exp0, uint32_t base_exp1,
    int dyn_mode, int dyn_split_k, int pdl, int trig, uintptr_t stream_ptr
) {
    if (W_DYN0 == nullptr || W_DYN1 == nullptr || O_DYN0 == nullptr || O_DYN1 == nullptr || N_dyn <= 0) {
        return bf11_flip_dual_kv_v5_c1_fwd(
            X0, X1, SLABS0, SLABS1, SMETA0, SMETA1, SFLIPS0, SFLIPS1, BMASKS0, BMASKS1,
            OUTLIERS0, OUTLIERS1, O_OFF0, O_OFF1, P, SEM, O0, O1, NLIVE_PTR,
            M, N_each, K, sx0, sx1, KS, so0, so1, cfg_id, w_run_req, split, base_exp0, base_exp1,
            pdl, trig, stream_ptr
        );
    }
    if (split != 9 || M <= 0 || N_each <= 0 || !dyn_params_valid(N_dyn, K, dyn_mode, dyn_split_k)) return -2;
    int wp = 0;
    if (cfg_id == 1) wp = 4;
    else if (cfg_id == 4) wp = 2;
    else if (cfg_id == 5) wp = 1;
    else return -1;
    const int bn_p = wp * 8;
    const int w_global_each = ((N_each + bn_p - 1) / bn_p) * wp;
    const int w_run = (w_run_req == 1 || w_run_req == 2 || w_run_req == 4) ? w_run_req : wp;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    const bool m_le8 = (M <= 8);
    const bool do_pdl = (pdl != 0);
    const bool do_trig = (trig != 0);
    const size_t smem = static_cast<size_t>(m_le8 ? 8 : 16) * (KS + 8) * sizeof(__nv_bfloat16);

    #define LAUNCH_DKVD_V9C1(W, ML, PD, TR) \
        launch_maybe_pdl(bf11_flip_dual_kv_dyn_v9_c1_kernel<W, ML, 9, PD, TR>, \
            dim3(2 * ((w_global_each + (W) - 1) / (W)) + 2, 9, 1), dim3((W) * 32, 1, 1), smem, stream, PD, \
            static_cast<const __nv_bfloat16*>(X0), static_cast<const __nv_bfloat16*>(X1), \
            static_cast<const uint8_t*>(SLABS0), static_cast<const uint8_t*>(SLABS1), \
            static_cast<const uint32_t*>(SMETA0), static_cast<const uint32_t*>(SMETA1), \
            static_cast<const uint16_t*>(SFLIPS0), static_cast<const uint16_t*>(SFLIPS1), \
            static_cast<const uint32_t*>(BMASKS0), static_cast<const uint32_t*>(BMASKS1), \
            static_cast<const uint64_t*>(OUTLIERS0), static_cast<const uint64_t*>(OUTLIERS1), \
            static_cast<const int32_t*>(O_OFF0), static_cast<const int32_t*>(O_OFF1), \
            static_cast<float*>(P), static_cast<int32_t*>(SEM), \
            static_cast<__nv_bfloat16*>(O0), static_cast<__nv_bfloat16*>(O1), \
            static_cast<const __nv_bfloat16*>(W_DYN0), static_cast<const __nv_bfloat16*>(W_DYN1), \
            static_cast<__nv_bfloat16*>(O_DYN0), static_cast<__nv_bfloat16*>(O_DYN1), \
            static_cast<const int32_t*>(NLIVE_PTR), \
            M, N_each, N_dyn, K, sx0, sx1, KS, so0, so1, so_dyn0, so_dyn1, \
            (w_global_each + (W) - 1) / (W), w_global_each, base_exp0, base_exp1, dyn_mode, dyn_split_k)
    #define DISP_DKVD_V9C1(W) \
        if (m_le8) { \
            if (do_pdl) return static_cast<int>(do_trig ? LAUNCH_DKVD_V9C1(W, true, true, true) : LAUNCH_DKVD_V9C1(W, true, true, false)); \
            return static_cast<int>(do_trig ? LAUNCH_DKVD_V9C1(W, true, false, true) : LAUNCH_DKVD_V9C1(W, true, false, false)); \
        } else { \
            if (do_pdl) return static_cast<int>(do_trig ? LAUNCH_DKVD_V9C1(W, false, true, true) : LAUNCH_DKVD_V9C1(W, false, true, false)); \
            return static_cast<int>(do_trig ? LAUNCH_DKVD_V9C1(W, false, false, true) : LAUNCH_DKVD_V9C1(W, false, false, false)); \
        }

    if (w_run == 4)      { DISP_DKVD_V9C1(4); }
    else if (w_run == 2) { DISP_DKVD_V9C1(2); }
    else                 { DISP_DKVD_V9C1(1); }
    #undef DISP_DKVD_V9C1
    #undef LAUNCH_DKVD_V9C1
}

extern "C" int bf11_flip_dual_kv_dyn_v9_c32_fwd(
    const void* X0, const void* X1,
    const void* SLABS0, const void* SLABS1,
    const void* SMETA0, const void* SMETA1,
    const void* SFLIPS0, const void* SFLIPS1,
    const void* BMASKS0, const void* BMASKS1,
    const void* OUTLIERS0, const void* OUTLIERS1,
    const void* O_OFF0, const void* O_OFF1,
    void* P, void* SEM, void* O0, void* O1,
    const void* W_DYN0, const void* W_DYN1, void* O_DYN0, void* O_DYN1,
    const void* NLIVE_PTR,
    int M, int N_each, int N_dyn, int K, int sx0, int sx1, int KS, int split_packed,
    int so0, int so1, int so_dyn0, int so_dyn1,
    int cfg_id, int bm_req, int w_run_req, int split_cta_req, int k_order,
    uint32_t base_exp0, uint32_t base_exp1, int dyn_mode, int dyn_split_k,
    int pdl, int trig, uintptr_t stream_ptr
) {
    return static_cast<int>(launch_bf11_v12_dual_kv_dyn_c32_impl(
        X0, X1, SLABS0, SLABS1, SMETA0, SMETA1, SFLIPS0, SFLIPS1, BMASKS0, BMASKS1,
        OUTLIERS0, OUTLIERS1, O_OFF0, O_OFF1, P, SEM, O0, O1,
        W_DYN0, W_DYN1, O_DYN0, O_DYN1, NLIVE_PTR,
        M, N_each, N_dyn, K, sx0, sx1, KS, split_packed, so0, so1, so_dyn0, so_dyn1,
        cfg_id, bm_req, w_run_req, split_cta_req, k_order,
        base_exp0, base_exp1, dyn_mode, dyn_split_k,
        pdl, trig, false, reinterpret_cast<cudaStream_t>(stream_ptr)
    ));
}

// =========================================================================
// PART 6: WAVE-V12 C=1..8 (M=1..48+) EXTERN "C" ABI & 1-LAUNCH GPU UNPACK
//   - Dedicated BM=48 (MT=3, BK=128) SM121a tile (24.8 KiB SMEM -> 4 CTAs/SM!)
//   - Optimal SM121a wave-fill auto-planner for C=1..8 (M=1..48) + M=64..128:
//       * M=9..16 wo_ud: w_run=4 (144 CTAs = 3.0/SM exact 1.0 wave @ 4/SM!)
//       * M=33..48 qkvr/q_side/dual_kv: BM=48 w_run=2 (4 CTAs/SM, -25% L2 P!)
//       * M=33..48 wo_ud: BM=48 w_run=4 (144 CTAs = 3.0/SM exact 1.0 wave!)
//   - ZERO #206 PDL hazards, st.release.gpu.global.s32 semaphore resets
//   - 1-launch 0-ULP GPU bf11_unpack_bf16_v12_fwd (kills 35-op PyTorch unpack!)
// =========================================================================

extern "C" int bf11_flip_v12_version() { return 1200; }

extern "C" int bf11_flip_v12_c1_fwd(
    const void* X, const void* SLABS,
    const void* SLAB_META_V5, const void* SFLIPS, const void* BMASKS,
    const void* OUTLIERS, const void* O_OFF,
    void* P, void* SEM, void* O, const void* NLIVE_PTR,
    int M, int N, int N_active, int K, int sx, int KS, int so_m,
    int cfg_id, int w_run_req, int split, uint32_t base_exp,
    int pdl, int trig, uintptr_t stream_ptr
) {
    int eff_w_run = w_run_req;
    if (eff_w_run == 0 && (cfg_id == 1 || cfg_id == 4 || cfg_id == 5)) {
        if (split == 6 || split == 9) {
            eff_w_run = 2;  // 714/618/144 CTAs -> 99.0-100.0% SM wave efficiency!
        } else if (split == 1) {
            // wo_ud (4608,768): M<=8 fits 6 CTAs/SM (w=2 -> 288=1.0 wave);
            // M=9..16 fits 4 CTAs/SM (w=4 -> 144=3.0/SM exact 1.0 wave, killing 1.5-wave tail!)
            eff_w_run = (M <= 8) ? 2 : 4;
        }
    }
    return bf11_flip_v5_c1_fwd(
        X, SLABS, SLAB_META_V5, SFLIPS, BMASKS, OUTLIERS, O_OFF,
        P, SEM, O, NLIVE_PTR,
        M, N, N_active, K, sx, KS, so_m,
        cfg_id, eff_w_run, split, base_exp, pdl, trig, stream_ptr
    );
}

extern "C" int bf11_flip_dual_kv_v12_c1_fwd(
    const void* X0, const void* X1,
    const void* SLABS0, const void* SLABS1,
    const void* SMETA0, const void* SMETA1,
    const void* SFLIPS0, const void* SFLIPS1,
    const void* BMASKS0, const void* BMASKS1,
    const void* OUTLIERS0, const void* OUTLIERS1,
    const void* O_OFF0, const void* O_OFF1,
    void* P, void* SEM, void* O0, void* O1, const void* NLIVE_PTR,
    int M, int N_each, int K, int sx0, int sx1, int KS, int so0, int so1,
    int cfg_id, int w_run_req, int split, uint32_t base_exp0, uint32_t base_exp1,
    int pdl, int trig, uintptr_t stream_ptr
) {
    const int eff_w_run = (w_run_req == 0) ? 2 : w_run_req;  // 144 CTAs = 3.0/SM exact!
    return bf11_flip_dual_kv_v5_c1_fwd(
        X0, X1, SLABS0, SLABS1, SMETA0, SMETA1, SFLIPS0, SFLIPS1, BMASKS0, BMASKS1,
        OUTLIERS0, OUTLIERS1, O_OFF0, O_OFF1, P, SEM, O0, O1, NLIVE_PTR,
        M, N_each, K, sx0, sx1, KS, so0, so1,
        cfg_id, eff_w_run, split, base_exp0, base_exp1, pdl, trig, stream_ptr
    );
}

extern "C" int bf11_flip_v12_c32_fwd(
    const void* X, const void* SLABS,
    const void* SLAB_META_V5, const void* SFLIPS, const void* BMASKS,
    const void* OUTLIERS, const void* O_OFF,
    void* P, void* SEM, void* O, const void* NLIVE_PTR,
    int M, int N, int N_active, int K, int sx, int KS, int split_packed, int so_m,
    int cfg_id, int bm_req, int w_run_req, int split_cta_req, int k_order,
    uint32_t base_exp, int pdl, int trig, uintptr_t stream_ptr
) {
    return static_cast<int>(launch_bf11_v12_c32_impl(
        X, SLABS, SLAB_META_V5, SFLIPS, BMASKS, OUTLIERS, O_OFF,
        P, SEM, O, NLIVE_PTR,
        M, N, N_active, K, sx, KS, split_packed, so_m,
        cfg_id, bm_req, w_run_req, split_cta_req, k_order,
        base_exp, pdl, trig, true, reinterpret_cast<cudaStream_t>(stream_ptr)
    ));
}

extern "C" int bf11_flip_dual_kv_v12_c32_fwd(
    const void* X0, const void* X1,
    const void* SLABS0, const void* SLABS1,
    const void* SMETA0, const void* SMETA1,
    const void* SFLIPS0, const void* SFLIPS1,
    const void* BMASKS0, const void* BMASKS1,
    const void* OUTLIERS0, const void* OUTLIERS1,
    const void* O_OFF0, const void* O_OFF1,
    void* P, void* SEM, void* O0, void* O1, const void* NLIVE_PTR,
    int M, int N_each, int K, int sx0, int sx1, int KS, int split_packed, int so0, int so1,
    int cfg_id, int bm_req, int w_run_req, int split_cta_req, int k_order,
    uint32_t base_exp0, uint32_t base_exp1, int pdl, int trig, uintptr_t stream_ptr
) {
    return static_cast<int>(launch_bf11_v12_dual_kv_c32_impl(
        X0, X1, SLABS0, SLABS1, SMETA0, SMETA1, SFLIPS0, SFLIPS1, BMASKS0, BMASKS1,
        OUTLIERS0, OUTLIERS1, O_OFF0, O_OFF1,
        P, SEM, O0, O1, NLIVE_PTR,
        M, N_each, K, sx0, sx1, KS, split_packed, so0, so1,
        cfg_id, bm_req, w_run_req, split_cta_req, k_order,
        base_exp0, base_exp1, pdl, trig, true, reinterpret_cast<cudaStream_t>(stream_ptr)
    ));
}

extern "C" int bf11_flip_dual_kv_dyn_v12_c1_fwd(
    const void* X0, const void* X1,
    const void* SLABS0, const void* SLABS1,
    const void* SMETA0, const void* SMETA1,
    const void* SFLIPS0, const void* SFLIPS1,
    const void* BMASKS0, const void* BMASKS1,
    const void* OUTLIERS0, const void* OUTLIERS1,
    const void* O_OFF0, const void* O_OFF1,
    void* P, void* SEM, void* O0, void* O1,
    const void* W_DYN0, const void* W_DYN1, void* O_DYN0, void* O_DYN1,
    const void* NLIVE_PTR,
    int M, int N_each, int N_dyn, int K, int sx0, int sx1, int KS,
    int so0, int so1, int so_dyn0, int so_dyn1,
    int cfg_id, int w_run_req, int split, uint32_t base_exp0, uint32_t base_exp1,
    int dyn_mode, int dyn_split_k, int pdl, int trig, uintptr_t stream_ptr
) {
    const int eff_w_run = (w_run_req == 0) ? 2 : w_run_req;
    return bf11_flip_dual_kv_dyn_v9_c1_fwd(
        X0, X1, SLABS0, SLABS1, SMETA0, SMETA1, SFLIPS0, SFLIPS1, BMASKS0, BMASKS1,
        OUTLIERS0, OUTLIERS1, O_OFF0, O_OFF1, P, SEM, O0, O1,
        W_DYN0, W_DYN1, O_DYN0, O_DYN1, NLIVE_PTR,
        M, N_each, N_dyn, K, sx0, sx1, KS, so0, so1, so_dyn0, so_dyn1,
        cfg_id, eff_w_run, split, base_exp0, base_exp1, dyn_mode, dyn_split_k,
        pdl, trig, stream_ptr
    );
}

extern "C" int bf11_flip_dual_kv_dyn_v12_c32_fwd(
    const void* X0, const void* X1,
    const void* SLABS0, const void* SLABS1,
    const void* SMETA0, const void* SMETA1,
    const void* SFLIPS0, const void* SFLIPS1,
    const void* BMASKS0, const void* BMASKS1,
    const void* OUTLIERS0, const void* OUTLIERS1,
    const void* O_OFF0, const void* O_OFF1,
    void* P, void* SEM, void* O0, void* O1,
    const void* W_DYN0, const void* W_DYN1, void* O_DYN0, void* O_DYN1,
    const void* NLIVE_PTR,
    int M, int N_each, int N_dyn, int K, int sx0, int sx1, int KS, int split_packed,
    int so0, int so1, int so_dyn0, int so_dyn1,
    int cfg_id, int bm_req, int w_run_req, int split_cta_req, int k_order,
    uint32_t base_exp0, uint32_t base_exp1, int dyn_mode, int dyn_split_k,
    int pdl, int trig, uintptr_t stream_ptr
) {
    return static_cast<int>(launch_bf11_v12_dual_kv_dyn_c32_impl(
        X0, X1, SLABS0, SLABS1, SMETA0, SMETA1, SFLIPS0, SFLIPS1, BMASKS0, BMASKS1,
        OUTLIERS0, OUTLIERS1, O_OFF0, O_OFF1, P, SEM, O0, O1,
        W_DYN0, W_DYN1, O_DYN0, O_DYN1, NLIVE_PTR,
        M, N_each, N_dyn, K, sx0, sx1, KS, split_packed, so0, so1, so_dyn0, so_dyn1,
        cfg_id, bm_req, w_run_req, split_cta_req, k_order,
        base_exp0, base_exp1, dyn_mode, dyn_split_k,
        pdl, trig, true, reinterpret_cast<cudaStream_t>(stream_ptr)
    ));
}

extern "C" int bf11_unpack_bf16_v12_fwd(
    const void* SLABS, const void* FLIPS, const void* WARP_META,
    const void* OUTLIERS, const void* O_OFF,
    void* OUT,
    int N, int N_active, int K, int KS, int cfg_id, int split,
    uint32_t base_exp, uintptr_t stream_ptr
) {
    if (N <= 0 || K <= 0 || KS <= 0 || (KS & 127) != 0 || split <= 0 || KS * split != K || N_active < 0 || N_active > N) {
        return -2;
    }
    int warps = 0, sub_n = 0;
    if (cfg_id == 0)      { warps = 4; sub_n = 2; }
    else if (cfg_id == 1) { warps = 4; sub_n = 1; }
    else if (cfg_id == 2) { warps = 2; sub_n = 2; }
    else if (cfg_id == 4) { warps = 2; sub_n = 1; }
    else if (cfg_id == 5) { warps = 1; sub_n = 1; }
    else return -1;

    const int bn = warps * sub_n * 8;
    const int n_tiles = (N + bn - 1) / bn;
    const int spw = (KS >> 7) * sub_n;
    if (spw <= 0 || spw > 12) return -2;
    const int wtot = split * n_tiles * warps;
    if (wtot <= 0) return -2;

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    bf11_unpack_bf16_v12_kernel<<<dim3(wtot, 1, 1), dim3(32, 1, 1), 0, stream>>>(
        static_cast<const uint8_t*>(SLABS),
        static_cast<const uint16_t*>(FLIPS),
        static_cast<const uint4*>(WARP_META),
        static_cast<const uint64_t*>(OUTLIERS),
        static_cast<const int32_t*>(O_OFF),
        static_cast<__nv_bfloat16*>(OUT),
        N, N_active, K, KS, warps, sub_n, n_tiles, spw, wtot, base_exp
    );
    return static_cast<int>(cudaGetLastError());
}
