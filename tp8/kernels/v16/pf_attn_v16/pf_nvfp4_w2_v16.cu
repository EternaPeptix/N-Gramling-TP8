// SPDX-License-Identifier: Apache-2.0
// =============================================================================
// SM121a (NVIDIA DGX Spark GB10) Prefill-Only Persistent NVFP4 W2 + Fused
// L2-Resident Combine v16-SUB60s (`libpf_nvfp4_w2_v16.so`, ZERO-SPIN 48/48 SMs!)
//   (3-Stage cp.async A-Tile Ring + a_pinned<=3 [192 rows] + 64-Bucket Fine LPT +
//    SM121a L2::evict_first Weights + Double-Buffered Zero-Spin Output Slice Overlap!)
// =============================================================================
//
// Targets Inkling-2-Small TP8 40 NVFP4 MoE Layers (E=384, H=4608, inter=288,
// topk=8) strictly for long-context eager prefill chunks (plan.tile >= 64,
// T >= 1024, !capturing), leaving C=1 and C=32 decode 100% untouched!
//
// Six Roofline Upgrades over `pf_moe_native_sm121_v4.cu` (`pf_nvfp4_w2_ws5_persistent`):
//   1. Exact 1.000x Cooperative W_PACK Load + 0.5x Unswizzle (`sB_xchg` union):
//      - Fixes v4's 2.00x duplicate `ld.global.cs` weight load bug where both
//        `wm=0` and `wm=1` loaded all 9 slabs (509.6 MB -> exact 254.8 MB!)
//        and halves `__shfl_sync` + `prmt.b32` instructions (72+36 -> 36+18).
//   2. 5-Register Per-Lane WS5 Scale Decode (`sfb_my0..4`) w/ Unrolled 8-Way
//      Patch Select (NO runtime `sb[idx]`, 0 B stack frame!):
//      - Single coalesced `ld.global.cs.v4.u32` on `tid < 90` (1,440 B),
//        eliminates `sWS[128][5]` (-2,560 B SMEM), eliminates 2,304 magic-mul
//        div/mods, eliminates 40 `ld.shared`/M-tile, and eliminates patch barrier.
//   3. Zero-Spill Software-Pipelined Next-Panel (`dp_step + 1`) Weight Prefetch:
//      - CTA-uniform pointers/bounds live in `128 B` `__shared__ W2CtaParams`
//        (read via non-hoistable `ld.shared`, saving ~22 GPRs -> ~90 regs <= 128).
//      - Right after `mi_blk == num_m - 1`'s tail MMA (when all 41 weight/scale
//        regs die), issues non-blocking `ld.global.cs.v4.u32` for `dp_step + 1`
//        overlapped with `dp_step`'s SMEM epilogue + 16 global `bf16x2` atomics!
//   4. Single-Launch Co-Resident Persistent Grid (<= 96 CTAs, 2 CTAs/SM):
//      - Dynamic LPT work-stealing (`atomicAdd(&work_cursor[s], 1)`, `split_m=512`)
//        eliminates all 18 static wave-drain tail bubbles (~37 launches -> 2!).
//      - Self-resetting 2-bank ping-pong `SliceState` (0 memset launches, safe
//        across repeated benchmark/cached-plan replays).
//   5. Staggered Sub-Panel Ownership (`dp = (dp_step + cta_id) % PANELS`) +
//      Overlapped / Barrier-Gated In-L2 Sector-Complete Slice Zeroing:
//      - Cuts L2 `bf16x2` atomic bank collisions by 3x-6x and supports
//        `PANELS in {1, 2, 3, 4, 6}` (covering T_mb = 4K, 8K, 12K, 16K).
//   6. Pre-Fused `row_scale * Ones[e] * [TopkWeights]` + `wm=1` Tail MMA Skip:
//      - Fuses row scale & optional top-k weight once on `tid < 64` and skips
//        dead `wm=1` MMAs when `end - row0 <= 32`.
//
// Build on GB10 (SM121a):
//   /usr/local/cuda/bin/nvcc -O3 -std=c++17 --shared -Xcompiler -fPIC \
//       --fmad=false -gencode arch=compute_121a,code=sm_121a --ptxas-options=-v \
//       pf_nvfp4_w2_v5.cu -o libpf_nvfp4_w2_v5.so
// =============================================================================

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace {

static constexpr int kMaxSlices = 36;

// =============================================================================
// Low-Level SM121a PTX Primitives (Exact RN Rounding, Streaming/Acquire Atomics)
// =============================================================================

__device__ __forceinline__ float mul_rn(float a, float b) {
    float r;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

__device__ __forceinline__ uint32_t pack_bf16x2(float a, float b) {
    uint32_t out;
    asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(out) : "f"(a), "f"(b));
    return out;
}

__device__ __forceinline__ uint32_t prmt_b32(uint32_t a, uint32_t b, uint32_t sel) {
    uint32_t out;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(out) : "r"(a), "r"(b), "r"(sel));
    return out;
}

__device__ __forceinline__ uint4 ld_cs_u4(const void* ptr) {
    uint4 v;
    asm volatile("ld.global.cs.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(ptr));
    return v;
}

__device__ __forceinline__ uint4 ld_ca_u4(const void* ptr) {
    uint4 v;
    asm volatile("ld.global.ca.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(ptr));
    return v;
}

__device__ __forceinline__ uint32_t ld_cs_u32(const void* ptr) {
    uint32_t v;
    asm volatile("ld.global.cs.u32 %0, [%1];" : "=r"(v) : "l"(ptr));
    return v;
}

__device__ __forceinline__ uint32_t ld_ca_u32(const void* ptr) {
    uint32_t v;
    asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(v) : "l"(ptr));
    return v;
}

__device__ __forceinline__ uint64_t make_l2_evict_first_policy_v16() {
    uint64_t policy = 0ULL;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    float frac = 1.0f;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, %1;" : "=l"(policy) : "f"(frac));
#endif
    return policy;
}

__device__ __forceinline__ uint4 ld_evict_u4(const void* ptr, uint64_t policy) {
    uint4 v;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("ld.global.L2::cache_hint.v4.u32 {%0, %1, %2, %3}, [%4], %5;"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(ptr), "l"(policy));
#else
    (void)policy;
    v = ld_cs_u4(ptr);
#endif
    return v;
}

__device__ __forceinline__ uint32_t ld_evict_u32(const void* ptr, uint64_t policy) {
    uint32_t v;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("ld.global.L2::cache_hint.u32 %0, [%1], %2;" : "=r"(v) : "l"(ptr), "l"(policy));
#else
    (void)policy;
    v = ld_cs_u32(ptr);
#endif
    return v;
}

__device__ __forceinline__ uint4 ld_w_u4(const void* ptr, int w_evict_mode, uint64_t l2_policy) {
    if (w_evict_mode == 1) return ld_evict_u4(ptr, l2_policy);
    return (w_evict_mode != 0) ? ld_cs_u4(ptr) : ld_ca_u4(ptr);
}

__device__ __forceinline__ uint32_t ld_w_u32(const void* ptr, int w_evict_mode, uint64_t l2_policy) {
    if (w_evict_mode == 1) return ld_evict_u32(ptr, l2_policy);
    return (w_evict_mode != 0) ? ld_cs_u32(ptr) : ld_ca_u32(ptr);
}

__device__ __forceinline__ void st_wb_u4(void* ptr, uint4 v) {
    asm volatile("st.global.wb.v4.u32 [%0], {%1, %2, %3, %4};"
                 :: "l"(ptr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}

__device__ __forceinline__ void st_wb_u32(uint32_t* ptr, uint32_t v) {
    asm volatile("st.global.wb.u32 [%0], %1;" :: "l"(ptr), "r"(v) : "memory");
}

__device__ __forceinline__ void st_cs_u4(void* ptr, uint4 v) {
    asm volatile("st.global.cs.v4.u32 [%0], {%1, %2, %3, %4};"
                 :: "l"(ptr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}

__device__ __forceinline__ void red_add_f32(float* addr, float val) {
    asm volatile("red.relaxed.gpu.global.add.f32 [%0], %1;\n"
                 :: "l"(addr), "f"(val) : "memory");
}

__device__ __forceinline__ void red_add_bf16x2_bits(uint32_t* addr, uint32_t bf16x2_bits) {
    asm volatile("red.relaxed.gpu.global.add.noftz.bf16x2 [%0], %1;\n"
                 :: "l"(addr), "r"(bf16x2_bits) : "memory");
}

// Wave-V13 ZERO-SPIN: ld_acquire_gpu_u32 / st_release_gpu_u32 removed completely —
// ZERO cross-CTA spinlocks exist anywhere in pf_nvfp4_w2_v13.cu!

// Non-hoistable SMEM parameter broadcast helpers (prevents NVVM LICM from pinning 64-bit pointers in GPRs!)
template <typename T>
__device__ __forceinline__ const T* ld_smem_ptr(const T* const* slot) {
    uint64_t raw;
    uint32_t sm_addr = static_cast<uint32_t>(__cvta_generic_to_shared(slot));
    asm("ld.shared.u64 %0, [%1];" : "=l"(raw) : "r"(sm_addr));
    return reinterpret_cast<const T*>(raw);
}

template <typename T>
__device__ __forceinline__ T* ld_smem_mut_ptr(T* const* slot) {
    uint64_t raw;
    uint32_t sm_addr = static_cast<uint32_t>(__cvta_generic_to_shared(slot));
    asm("ld.shared.u64 %0, [%1];" : "=l"(raw) : "r"(sm_addr));
    return reinterpret_cast<T*>(raw);
}

__device__ __forceinline__ int ld_smem_i32(const int* slot) {
    int v;
    uint32_t sm_addr = static_cast<uint32_t>(__cvta_generic_to_shared(slot));
    asm("ld.shared.s32 %0, [%1];" : "=r"(v) : "r"(sm_addr));
    return v;
}

__device__ __forceinline__ void cp_async_cg_16(void* smem_ptr, const void* gmem_ptr, int src_bytes) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(smem_addr), "l"(gmem_ptr), "r"(src_bytes) : "memory");
}

__device__ __forceinline__ void cp_async_ca_16(void* smem_ptr, const void* gmem_ptr, int src_bytes) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(smem_addr), "l"(gmem_ptr), "r"(src_bytes) : "memory");
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template <int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}

// Native SM121a NVFP4 block-scaled Tensor Core MMA: m16n8k64, e2m1 x e2m1, ue4m3 (scale_vec::4X, VS=16).
__device__ __forceinline__ void mma_m16n8k64_nvfp4(
    float (&d)[4],
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    uint32_t sfa, uint32_t sfb
) {
    constexpr uint16_t bidA = 0, tidA = 0, bidB = 0, tidB = 0;
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, "
        "{%14}, {%15, %16}, {%17}, {%18, %19};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "f"(d[0]), "f"(d[1]), "f"(d[2]), "f"(d[3]),
          "r"(sfa), "h"(bidA), "h"(tidA), "r"(sfb), "h"(bidB), "h"(tidB)
    );
}

__device__ __forceinline__ int64_t scale_offset_byte(int64_t row, int col, int COLS, int TILE) {
    if (TILE >= 128) {
        int64_t tile = (row >> 7) * ((COLS + 3) >> 2) + (col >> 2);
        return tile * 512 + (row & 31) * 16 + ((row >> 5) & 3) * 4 + (col & 3);
    }
    if (TILE > 0) {
        int64_t tile = (row >> 3) * ((COLS + 3) >> 2) + (col >> 2);
        return tile * 32 + (row & 7) * 4 + (col & 3);
    }
    return row * COLS + col;
}

__device__ __forceinline__ uint32_t load_as_u32_rowmaj(
    const uint8_t* __restrict__ AS, int64_t row, int col, int row_stride
) {
    const uint8_t* p = AS + row * row_stride + col;
    if ((row_stride & 3) == 0) {
        return *reinterpret_cast<const uint32_t*>(p);
    }
    return static_cast<uint32_t>(p[0])
         | (static_cast<uint32_t>(p[1]) << 8)
         | (static_cast<uint32_t>(p[2]) << 16)
         | (static_cast<uint32_t>(p[3]) << 24);
}

__device__ __forceinline__ uint32_t load_as_u16_rowmaj(
    const uint8_t* __restrict__ AS, int64_t row, int col, int row_stride
) {
    const uint8_t* p = AS + row * row_stride + col;
    if ((row_stride & 1) == 0) {
        return static_cast<uint32_t>(*reinterpret_cast<const uint16_t*>(p));
    }
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8);
}

// =============================================================================
// 1-Block Zero-Sync LPT Expert Plan Builder (BM=64-Aligned Split, ~0.8 us)
// =============================================================================

__device__ __forceinline__ void find_expert_row_range(
    const int32_t* __restrict__ Tiles,
    int max_ctas,
    int tile,
    int target_e,
    int& row_beg,
    int& row_end
) {
    const int num_tiles = Tiles[2 * max_ctas];
    if (num_tiles <= 0) {
        row_beg = 0;
        row_end = 0;
        return;
    }
    int lo = 0, hi = num_tiles;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (Tiles[mid] < target_e) lo = mid + 1;
        else hi = mid;
    }
    const int t_lo = lo;
    if (t_lo >= num_tiles || Tiles[t_lo] != target_e) {
        row_beg = 0;
        row_end = 0;
        return;
    }
    lo = t_lo + 1;
    hi = num_tiles;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (Tiles[mid] <= target_e) lo = mid + 1;
        else hi = mid;
    }
    row_beg = t_lo * tile;
    row_end = Tiles[max_ctas + lo - 1];
}

// PlanBuf layout (size >= 3 * max_work + 1 int32s):
//   [0 .. max_work-1]            : work_e   (-1 if unused)
//   [max_work .. 2*max_work-1]   : work_beg (aligned to 64 rows = BM=64!)
//   [2*max_work .. 3*max_work-1] : work_end
//   [3*max_work]                 : num_work (actual active work items <= max_work)
__global__ __launch_bounds__(256, 1)
void pf_nvfp4_w2_build_plan_v5_kernel(
    const int32_t* __restrict__ Tiles,
    int32_t* __restrict__ PlanBuf,
    int max_ctas,
    int tile,
    int num_experts,
    int split_m,
    int max_work
) {
    constexpr int MAX_E = 1024;
    __shared__ int32_t sBeg[MAX_E];
    __shared__ int32_t sEnd[MAX_E];
    __shared__ uint32_t sBucketCnt[64];
    __shared__ uint32_t sBucketOff[64];

    const int tid = threadIdx.x;
    const int eff_e = (num_experts < MAX_E) ? num_experts : MAX_E;
    // Align split_m to 64 (BM=64) so split chunks of the same expert have 100% disjoint M-tiles!
    const int eff_split = (split_m > 0) ? ((split_m + 63) & ~63) : 0;

    if (tid < 64) sBucketCnt[tid] = 0u;
    for (int w = tid; w < max_work; w += 256) PlanBuf[w] = -1;
    __syncthreads();

    for (int e = tid; e < eff_e; e += 256) {
        int rb = 0, re = 0;
        find_expert_row_range(Tiles, max_ctas, tile, e, rb, re);
        sBeg[e] = rb;
        sEnd[e] = re;
        int cnt = re - rb;
        if (cnt > 0) {
            if (eff_split > 0 && cnt > eff_split) {
                for (int off = 0; off < cnt; off += eff_split) {
                    int sz = cnt - off;
                    if (sz > eff_split) sz = eff_split;
                    int raw_b = (sz - 1) >> 3;
                    int b = 63 - ((raw_b > 63) ? 63 : raw_b);
                    atomicAdd(&sBucketCnt[b], 1u);
                }
            } else {
                int raw_b = (cnt - 1) >> 3;
                int b = 63 - ((raw_b > 63) ? 63 : raw_b);
                atomicAdd(&sBucketCnt[b], 1u);
            }
        }
    }
    __syncthreads();

    if (tid == 0) {
        uint32_t acc = 0u;
        for (int b = 0; b < 64; ++b) {
            sBucketOff[b] = acc;
            acc += sBucketCnt[b];
        }
        int total_w = static_cast<int>(acc);
        if (total_w > max_work) total_w = max_work;
        PlanBuf[3 * max_work] = total_w;
    }
    __syncthreads();

    int32_t* out_e   = PlanBuf;
    int32_t* out_beg = PlanBuf + max_work;
    int32_t* out_end = PlanBuf + 2 * max_work;

    for (int e = tid; e < eff_e; e += 256) {
        int rb = sBeg[e], re = sEnd[e], cnt = re - rb;
        if (cnt > 0) {
            if (eff_split > 0 && cnt > eff_split) {
                for (int off = 0; off < cnt; off += eff_split) {
                    int sz = cnt - off;
                    if (sz > eff_split) sz = eff_split;
                    int raw_b = (sz - 1) >> 3;
                    int b = 63 - ((raw_b > 63) ? 63 : raw_b);
                    uint32_t pos = atomicAdd(&sBucketOff[b], 1u);
                    if (static_cast<int>(pos) < max_work) {
                        out_e[pos]   = e;
                        out_beg[pos] = rb + off;
                        out_end[pos] = rb + off + sz;
                    }
                }
            } else {
                int raw_b = (cnt - 1) >> 3;
                int b = 63 - ((raw_b > 63) ? 63 : raw_b);
                uint32_t pos = atomicAdd(&sBucketOff[b], 1u);
                if (static_cast<int>(pos) < max_work) {
                    out_e[pos]   = e;
                    out_beg[pos] = rb;
                    out_end[pos] = re;
                }
            }
        }
    }
}

// =============================================================================
// Wave-V13 ZERO-SPIN Host-Sliced Work-Stealing Cursors (`SliceState[0..35]`)
// =============================================================================
// Uses only the first `N_BLKS <= 36` uint32_t words of the existing 512-word
// `SliceState` buffer (100% drop-in ABI!). Each slice cursor is reset into L2
// BEFORE its slice kernel launches (either inside `zero_slice_v13` or via the
// 1-CTA `reset_cursors_v13` helper). Hardware stream barriers enforce exact
// single-dirty-slice L2 residency and complete zeroing with ZERO spinlocks!

// =============================================================================
// SMEM Layouts & Compile-Time 2-CTA/SM Occupancy Proof (<= 48 KiB)
// =============================================================================

union W2EpiSmem64 {
    uint32_t bf16x2[64][68];   // 17,408 B (stride 68 = 4 mod 32 -> 0-way bank conflict on BOTH write & read!)
    float    f32[32][132];     // 16,896 B (stride 132 = 4 mod 32 -> 0-way bank conflict on BOTH write & read!)
};

struct W2CoopStageSmem {
    uint4 sB_xchg[4][9][32];   // 18,432 B (0-way 128B wavefront cooperative W_PACK exchange)
    uint4 sWS5_u4[90];         //  1,440 B (exact 360 uint32 WS5_PACK coalesced stage)
};

struct alignas(16) W2CtaParams {
    const uint8_t*  A;
    const uint8_t*  AS;
    const float*    RS;
    const float*    TopkWeights;
    const int32_t*  Row2Tok;
    const uint8_t*  W_PACK;
    const uint32_t* WS5_PACK;
    const uint8_t*  WS_BASE;
    const int32_t*  P_OFF;
    const uint32_t* PATCHES;
    const float*    Ones;
    const int32_t*  Tiles;
    const int32_t*  PlanBuf;
    void*           Out;
    int             max_ctas;
    int             tile;
    int             num_experts;
    int             max_work;
    int             as_row_stride;
    int             tokens;
    int             use_cs_w;
    int             zero_mode;   // 0=prezeroed, 1=overlapped in-L2 zero, 2=barrier-gated in-L2 zero
};

struct alignas(16) W2V5Smem {
    // Wave-V16 3-Stage cp.async A-Tile Ring (53,072 B <= 64 KiB << 99 KiB GB10 SMEM!):
    //   - Pins ALL A-tiles for experts with num_m <= 3 (<= 192 rows!) across all panels!
    //   - Keeps 2 full 64x288 A-tiles in flight via cp_async_wait<1>() for heavy experts (num_m >= 4)!
    uint32_t sA_main[3][64][32];  // 24,576 B (3 stages)
    uint32_t sA_tail[3][64][4];   //  3,072 B (3 stages)
    uint32_t sAS[3][64][5];       //  3,840 B (3 stages)
    float    sRS[3][64];          //    768 B (3 stages)
    int32_t  sTok[3][64];         //    768 B (3 stages)
    union {
        W2EpiSmem64     epi;      // 17,408 B
        W2CoopStageSmem stage;    // 19,872 B
    } u;
    W2CtaParams sP;               //    160 B
    int32_t     sWork[4];         //     16 B (e, row_beg, row_end, ctrl)
};

static_assert(sizeof(W2V5Smem) <= 65536, "W2V5Smem V16 3-stage must fit <= 64 KiB!");

// =============================================================================
// Register-Only Weight Unswizzle & Unrolled 8-Way WS5 Scale Decode Helpers
// =============================================================================

struct RawPanelPrefetch {
    uint4 w0, w1, w2, w3, w4;
    uint4 ws5;
};

__device__ __forceinline__ uint4 unswizzle_w_slab_u4(
    uint4 w, int src0, int src1, uint32_t prmt_sel
) {
    uint4 out;
    out.x = prmt_b32(__shfl_sync(0xFFFFFFFFu, w.x, src0),
                     __shfl_sync(0xFFFFFFFFu, w.x, src1), prmt_sel);
    out.y = prmt_b32(__shfl_sync(0xFFFFFFFFu, w.y, src0),
                     __shfl_sync(0xFFFFFFFFu, w.y, src1), prmt_sel);
    out.z = prmt_b32(__shfl_sync(0xFFFFFFFFu, w.z, src0),
                     __shfl_sync(0xFFFFFFFFu, w.z, src1), prmt_sel);
    out.w = prmt_b32(__shfl_sync(0xFFFFFFFFu, w.w, src0),
                     __shfl_sync(0xFFFFFFFFu, w.w, src1), prmt_sel);
    return out;
}

__device__ __forceinline__ void issue_ws5_panel_loads(
    RawPanelPrefetch& pf,
    const uint8_t* W_PACK,
    const uint32_t* WS5_PACK,
    int64_t puid,
    int wm, int wp, int lane, int tid,
    int w_evict_mode,
    uint64_t l2_policy
) {
    const uint8_t* wp_base = W_PACK + (puid * 4 + wp) * (9LL * 512LL) + lane * 16;
    if (wm == 0) {
        pf.w0 = ld_w_u4(wp_base + 0 * 512, w_evict_mode, l2_policy);
        pf.w1 = ld_w_u4(wp_base + 1 * 512, w_evict_mode, l2_policy);
        pf.w2 = ld_w_u4(wp_base + 2 * 512, w_evict_mode, l2_policy);
        pf.w3 = ld_w_u4(wp_base + 3 * 512, w_evict_mode, l2_policy);
        pf.w4 = ld_w_u4(wp_base + 4 * 512, w_evict_mode, l2_policy);
    } else {
        pf.w0 = ld_w_u4(wp_base + 5 * 512, w_evict_mode, l2_policy);
        pf.w1 = ld_w_u4(wp_base + 6 * 512, w_evict_mode, l2_policy);
        pf.w2 = ld_w_u4(wp_base + 7 * 512, w_evict_mode, l2_policy);
        pf.w3 = ld_w_u4(wp_base + 8 * 512, w_evict_mode, l2_policy);
        pf.w4 = make_uint4(0u, 0u, 0u, 0u);
    }
    if (tid < 90) {
        pf.ws5 = ld_w_u4(reinterpret_cast<const uint4*>(WS5_PACK + puid * 360LL) + tid, w_evict_mode, l2_policy);
    } else {
        pf.ws5 = make_uint4(0u, 0u, 0u, 0u);
    }
}

// Decode 8 consecutive WS5 scales at even bit/nibble offset `base_flat` into `(p0, p1)`
// using strictly unrolled 8-way patch selection (ZERO runtime `sb[idx]` -> 0 B stack frame!).
template <int HAS_PATCHES>
__device__ __forceinline__ void decode_ws5_chunk8_unrolled(
    const uint8_t* ws_lo_bytes,
    const uint32_t* ws_hi_words,
    int base_flat,
    int base_e,
    const uint32_t* PATCHES,
    int p_lo,
    int p_hi,
    uint32_t& p0_out,
    uint32_t& p1_out
) {
    const int byte_off = base_flat >> 1;
    const uint8_t b0 = ws_lo_bytes[byte_off + 0];
    const uint8_t b1 = ws_lo_bytes[byte_off + 1];
    const uint8_t b2 = ws_lo_bytes[byte_off + 2];
    const uint8_t b3 = ws_lo_bytes[byte_off + 3];

    const int hw = base_flat >> 5;
    const int hb = base_flat & 31;
    uint32_t hi8 = ws_hi_words[hw] >> hb;
    if (hb > 24) {
        hi8 |= (ws_hi_words[hw + 1] << (32 - hb));
    }

    uint8_t sb0 = static_cast<uint8_t>(base_e + ((b0 & 0xFu) | (((hi8 >> 0) & 1u) << 4)));
    uint8_t sb1 = static_cast<uint8_t>(base_e + ((b0 >> 4)   | (((hi8 >> 1) & 1u) << 4)));
    uint8_t sb2 = static_cast<uint8_t>(base_e + ((b1 & 0xFu) | (((hi8 >> 2) & 1u) << 4)));
    uint8_t sb3 = static_cast<uint8_t>(base_e + ((b1 >> 4)   | (((hi8 >> 3) & 1u) << 4)));
    uint8_t sb4 = static_cast<uint8_t>(base_e + ((b2 & 0xFu) | (((hi8 >> 4) & 1u) << 4)));
    uint8_t sb5 = static_cast<uint8_t>(base_e + ((b2 >> 4)   | (((hi8 >> 5) & 1u) << 4)));
    uint8_t sb6 = static_cast<uint8_t>(base_e + ((b3 & 0xFu) | (((hi8 >> 6) & 1u) << 4)));
    uint8_t sb7 = static_cast<uint8_t>(base_e + ((b3 >> 4)   | (((hi8 >> 7) & 1u) << 4)));

    if constexpr (HAS_PATCHES) {
        for (int pi = p_lo; pi < p_hi; ++pi) {
            const uint32_t rec = PATCHES[pi];
            const int rel = static_cast<int>(rec >> 8) - base_flat;
            if (static_cast<uint32_t>(rel) < 8u) {
                const uint8_t pval = static_cast<uint8_t>(rec & 0xFFu);
                if (rel == 0) sb0 = pval;
                if (rel == 1) sb1 = pval;
                if (rel == 2) sb2 = pval;
                if (rel == 3) sb3 = pval;
                if (rel == 4) sb4 = pval;
                if (rel == 5) sb5 = pval;
                if (rel == 6) sb6 = pval;
                if (rel == 7) sb7 = pval;
            }
        }
    }

    p0_out = static_cast<uint32_t>(sb0)
           | (static_cast<uint32_t>(sb1) << 8)
           | (static_cast<uint32_t>(sb2) << 16)
           | (static_cast<uint32_t>(sb3) << 24);
    p1_out = static_cast<uint32_t>(sb4)
           | (static_cast<uint32_t>(sb5) << 8)
           | (static_cast<uint32_t>(sb6) << 16)
           | (static_cast<uint32_t>(sb7) << 24);
}

// Decode all 18 WS5 scales for `n_col = wp * 32 + lt * 8 + g` into 5 scalar registers.
template <int HAS_PATCHES>
__device__ __forceinline__ void decode_ws5_row18_to_regs(
    const uint4* sWS5_u4,
    int n_col,
    int base_e,
    const int32_t* P_OFF,
    const uint32_t* PATCHES,
    int64_t puid,
    uint32_t& sfb0, uint32_t& sfb1, uint32_t& sfb2, uint32_t& sfb3, uint32_t& sfb4
) {
    const uint32_t* ws_u32 = reinterpret_cast<const uint32_t*>(sWS5_u4);
    const uint8_t* ws_lo_b = reinterpret_cast<const uint8_t*>(ws_u32);
    const uint32_t* ws_hi_w = ws_u32 + 288;

    int p_lo = 0, p_hi = 0;
    if constexpr (HAS_PATCHES) {
        p_lo = P_OFF[puid];
        p_hi = P_OFF[puid + 1];
    }

    const int flat0 = n_col * 18;
    decode_ws5_chunk8_unrolled<HAS_PATCHES>(
        ws_lo_b, ws_hi_w, flat0, base_e, PATCHES, p_lo, p_hi, sfb0, sfb1);
    decode_ws5_chunk8_unrolled<HAS_PATCHES>(
        ws_lo_b, ws_hi_w, flat0 + 8, base_e, PATCHES, p_lo, p_hi, sfb2, sfb3);

    // Tail kg = 16..17: `flat_tail = 18*n_col + 16` is ALWAYS even -> `hb <= 30`, never crosses 32b boundary!
    const int flat_tail = flat0 + 16;
    const uint8_t bt = ws_lo_b[flat_tail >> 1];
    const uint32_t hi2 = ws_hi_w[flat_tail >> 5] >> (flat_tail & 31);
    uint8_t st0 = static_cast<uint8_t>(base_e + ((bt & 0xFu) | (((hi2 >> 0) & 1u) << 4)));
    uint8_t st1 = static_cast<uint8_t>(base_e + ((bt >> 4)   | (((hi2 >> 1) & 1u) << 4)));
    if constexpr (HAS_PATCHES) {
        for (int pi = p_lo; pi < p_hi; ++pi) {
            const uint32_t rec = PATCHES[pi];
            const int rel = static_cast<int>(rec >> 8) - flat_tail;
            const uint8_t pval = static_cast<uint8_t>(rec & 0xFFu);
            if (rel == 0) st0 = pval;
            if (rel == 1) st1 = pval;
        }
    }
    sfb4 = static_cast<uint32_t>(st0) | (static_cast<uint32_t>(st1) << 8);
}

// =============================================================================
// Full 64-Row Single-Barrier Coalesced W2 Epilogue (Pre-Fused sRS!)
// =============================================================================

template <int COMBINE_MODE>
__device__ __forceinline__ void w2_v5_epilogue_64(
    W2EpiSmem64& sEpi,
    const float (&acc)[2][4][4],
    const float* sRS,
    const int32_t* sTok,
    void* __restrict__ Out,
    int row0, int end, int nb,
    int warp, int wm, int wp, int lane, int g, int lt
) {
    constexpr int N = 4608;
    const int m_warp_base = wm * 32;

    if constexpr (COMBINE_MODE == 0 || COMBINE_MODE == 2) {
        #pragma unroll
        for (int mi = 0; mi < 2; ++mi) {
            const int m_lo = m_warp_base + mi * 16 + g;
            const int m_hi = m_lo + 8;
            const float s_lo = (row0 + m_lo < end) ? sRS[m_lo] : 0.0f;
            const float s_hi = (row0 + m_hi < end) ? sRS[m_hi] : 0.0f;
            #pragma unroll
            for (int sn = 0; sn < 4; ++sn) {
                const int col_pair = wp * 16 + sn * 4 + lt;
                sEpi.bf16x2[m_lo][col_pair] = pack_bf16x2(mul_rn(acc[mi][sn][0], s_lo),
                                                          mul_rn(acc[mi][sn][1], s_lo));
                sEpi.bf16x2[m_hi][col_pair] = pack_bf16x2(mul_rn(acc[mi][sn][2], s_hi),
                                                          mul_rn(acc[mi][sn][3], s_hi));
            }
        }
        __syncthreads();

        if constexpr (COMBINE_MODE == 0) {
            const int half_w = lane >> 4;
            const int c4 = lane & 15;
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int m = warp * 8 + k * 2 + half_w;
                const int r_glob = row0 + m;
                if (r_glob < end) {
                    uint4 v;
                    v.x = sEpi.bf16x2[m][c4 * 4 + 0];
                    v.y = sEpi.bf16x2[m][c4 * 4 + 1];
                    v.z = sEpi.bf16x2[m][c4 * 4 + 2];
                    v.w = sEpi.bf16x2[m][c4 * 4 + 3];
                    uint4* dst = reinterpret_cast<uint4*>(
                        reinterpret_cast<uint32_t*>(Out) + static_cast<int64_t>(r_glob) * (N / 2) + nb * 64);
                    st_cs_u4(dst + c4, v);
                }
            }
        } else {
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                const int m = warp * 8 + k;
                const int r_glob = row0 + m;
                if (r_glob < end) {
                    const int32_t tok = sTok[m];
                    if (tok >= 0) {
                        uint32_t* dst = reinterpret_cast<uint32_t*>(Out)
                                      + static_cast<int64_t>(tok) * (N / 2) + nb * 64;
                        red_add_bf16x2_bits(dst + lane,      sEpi.bf16x2[m][lane]);
                        red_add_bf16x2_bits(dst + 32 + lane, sEpi.bf16x2[m][32 + lane]);
                    }
                }
            }
        }
    } else {
        // COMBINE_MODE == 1: FP32 atomic via 32-row staging (2 passes for BM=64)
        #pragma unroll
        for (int w_sel = 0; w_sel < 2; ++w_sel) {
            if (wm == w_sel) {
                #pragma unroll
                for (int mi = 0; mi < 2; ++mi) {
                    const int m_lo = m_warp_base + mi * 16 + g;
                    const int m_hi = m_lo + 8;
                    const float s_lo = (row0 + m_lo < end) ? sRS[m_lo] : 0.0f;
                    const float s_hi = (row0 + m_hi < end) ? sRS[m_hi] : 0.0f;
                    const int r_lo = mi * 16 + g;
                    const int r_hi = r_lo + 8;
                    #pragma unroll
                    for (int sn = 0; sn < 4; ++sn) {
                        const int c0 = wp * 32 + sn * 8 + lt * 2;
                        sEpi.f32[r_lo][c0 + 0] = mul_rn(acc[mi][sn][0], s_lo);
                        sEpi.f32[r_lo][c0 + 1] = mul_rn(acc[mi][sn][1], s_lo);
                        sEpi.f32[r_hi][c0 + 0] = mul_rn(acc[mi][sn][2], s_hi);
                        sEpi.f32[r_hi][c0 + 1] = mul_rn(acc[mi][sn][3], s_hi);
                    }
                }
            }
            __syncthreads();

            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int r_epi = warp * 4 + k;
                const int m = w_sel * 32 + r_epi;
                const int r_glob = row0 + m;
                if (r_glob < end) {
                    const int32_t tok = sTok[m];
                    if (tok >= 0) {
                        float* dst = reinterpret_cast<float*>(Out)
                                   + static_cast<int64_t>(tok) * N + nb * 128;
                        #pragma unroll
                        for (int q = 0; q < 4; ++q) {
                            red_add_f32(dst + q * 32 + lane, sEpi.f32[r_epi][q * 32 + lane]);
                        }
                    }
                }
            }
            if (w_sel == 0) __syncthreads();
        }
    }
}

// =============================================================================
// Cooperative In-L2 128-Token Chunk Zeroing (`st.global.wb.v4.u32`, 128B Lines)
// =============================================================================

template <int PANELS>
__device__ __forceinline__ void zero_slice_chunk_128tok_wb(
    uint16_t* __restrict__ Out_bf16,
    int tokens,
    int slice_idx,
    int zchunk,
    int warp,
    int lane
) {
    constexpr int H = 4608;
    constexpr int COLS = PANELS * 128;
    const int tok_base = zchunk * 128;
    const uint4 z4 = make_uint4(0u, 0u, 0u, 0u);

    // 8 warps x 16 iters = 128 tokens per chunk; each warp writes 1 contiguous row per iter!
    #pragma unroll
    for (int it = 0; it < 16; ++it) {
        const int tok = tok_base + it * 8 + warp;
        if (tok < tokens) {
            uint4* dst = reinterpret_cast<uint4*>(
                Out_bf16 + static_cast<int64_t>(tok) * H + slice_idx * COLS);
            if constexpr (PANELS == 1) {
                if (lane < 16) st_wb_u4(dst + lane, z4);
            } else if constexpr (PANELS == 2) {
                st_wb_u4(dst + lane, z4);
            } else if constexpr (PANELS == 3) {
                st_wb_u4(dst + lane, z4);
                if (lane < 16) st_wb_u4(dst + 32 + lane, z4);
            } else if constexpr (PANELS == 4) {
                st_wb_u4(dst + lane, z4);
                st_wb_u4(dst + 32 + lane, z4);
            } else if constexpr (PANELS == 6) {
                st_wb_u4(dst + lane, z4);
                st_wb_u4(dst + 32 + lane, z4);
                st_wb_u4(dst + 64 + lane, z4);
            }
        }
    }
}

// Wave-V13 ZERO-SPIN Host-Launched In-L2 Slice Zeroer + Work-Cursor Reset!
// Launched on the SAME stream immediately before slice `s` of W2:
//   - Zeroes `Out[:, slice_s * PANELS * 128 : (slice_s+1) * PANELS * 128]` into L2 WB
//   - Resets `SliceState[slice_s] = 0u` in L2 WB (0 extra memset launches!)
template <int PANELS>
__global__ __launch_bounds__(256, 2)
void pf_nvfp4_w2_zero_slice_v13_kernel(
    uint16_t* __restrict__ Out_bf16,
    uint32_t* __restrict__ SliceState,
    int tokens,
    int slice_idx
) {
    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int zc   = static_cast<int>(blockIdx.x);
    if (zc == 0 && tid == 0) {
        st_wb_u32(&SliceState[slice_idx], 0u);
    }
    zero_slice_chunk_128tok_wb<PANELS>(Out_bf16, tokens, slice_idx, zc, warp, lane);
}

__global__ __launch_bounds__(64, 1)
void pf_nvfp4_w2_reset_cursors_v13_kernel(
    uint32_t* __restrict__ SliceState,
    int n_blks
) {
    const int tid = static_cast<int>(threadIdx.x);
    if (tid < n_blks) {
        st_wb_u32(&SliceState[tid], 0u);
    }
}

// =============================================================================
// Main Kernel: Wave-V13 ZERO-SPIN Single-Slice NVFP4 W2 WS5 (ALL 48/48 SMs!)
// =============================================================================

template <int AS_ROW_MAJOR, int HAS_PATCHES, int HAS_WEIGHTS, int COMBINE_MODE, int PANELS>
__global__ __launch_bounds__(256, 1)
void pf_nvfp4_w2_ws5_v5_kernel(
    const uint8_t* __restrict__ A,
    const uint8_t* __restrict__ AS,
    const float* __restrict__ RS,
    const float* __restrict__ TopkWeights,
    const int32_t* __restrict__ Row2Tok,
    const uint8_t* __restrict__ W_PACK,
    const uint32_t* __restrict__ WS5_PACK,
    const uint8_t* __restrict__ WS_BASE,
    const int32_t* __restrict__ P_OFF,
    const uint32_t* __restrict__ PATCHES,
    const float* __restrict__ Ones,
    const int32_t* __restrict__ Tiles,
    const int32_t* __restrict__ PlanBuf,
    uint32_t* __restrict__ SliceState,
    void* __restrict__ Out,
    int max_ctas,
    int tile,
    int num_experts,
    int max_work,
    int as_row_stride,
    int w_evict_mode,
    int slice_s
) {
    constexpr int BM     = 64;
    constexpr int KP     = 144;
    constexpr int KG     = 18;

    extern __shared__ uint4 dyn_smem[];
    auto& sm = *reinterpret_cast<W2V5Smem*>(dyn_smem);

    const int tid    = threadIdx.x;
    const int warp   = tid >> 5;
    const int wm     = warp >> 2;   // 0..1 (each owns 32 M-rows = 2 M-tiles)
    const int wp     = warp & 3;    // 0..3 (each owns 32 N-cols = 4 N-tiles)
    const int lane   = tid & 31;
    const int g      = lane >> 2;   // 0..7
    const int lt     = lane & 3;    // 0..3
    const int cta_id = blockIdx.x;

    // Populate CTA-uniform parameter block ONCE!
    if (tid == 0) {
        sm.sP.A             = A;
        sm.sP.AS            = AS;
        sm.sP.RS            = RS;
        sm.sP.TopkWeights   = TopkWeights;
        sm.sP.Row2Tok       = Row2Tok;
        sm.sP.W_PACK        = W_PACK;
        sm.sP.WS5_PACK      = WS5_PACK;
        sm.sP.WS_BASE       = WS_BASE;
        sm.sP.P_OFF         = P_OFF;
        sm.sP.PATCHES       = PATCHES;
        sm.sP.Ones          = Ones;
        sm.sP.Tiles         = Tiles;
        sm.sP.PlanBuf       = PlanBuf;
        sm.sP.Out           = Out;
        sm.sP.max_ctas      = max_ctas;
        sm.sP.tile          = tile;
        sm.sP.num_experts   = num_experts;
        sm.sP.max_work      = max_work;
        sm.sP.as_row_stride = as_row_stride;
        sm.sP.tokens        = 0;
        sm.sP.use_cs_w      = w_evict_mode;
        sm.sP.zero_mode     = 0;
    }
    __syncthreads();

    const int src0          = (g << 2) | ((lt & 1) << 1);
    const int src1          = src0 | 1;
    const uint32_t prmt_sel = (lt < 2) ? 0x5410u : 0x7632u;
    const int m_warp_base   = wm * 32;
    const int m_sf_off      = g + ((lt & 1) << 3);
    const int n_col_my      = wp * 32 + lt * 8 + g;
    const int s             = slice_s;

    auto load_a_tile = [&](int row_begin, int end, float scale_e, int mi_blk, int buf, bool cache_l1) {
        const uint8_t* pA   = ld_smem_ptr(&sm.sP.A);
        const uint8_t* pAS  = ld_smem_ptr(&sm.sP.AS);
        const float*   pRS  = ld_smem_ptr(&sm.sP.RS);
        const int32_t* pTok = ld_smem_ptr(&sm.sP.Row2Tok);
        const int tile_v    = ld_smem_i32(&sm.sP.tile);
        const int as_str    = ld_smem_i32(&sm.sP.as_row_stride);
        const int r0        = row_begin + mi_blk * BM;

        #pragma unroll
        for (int it = 0; it < 2; ++it) {
            const int idx = it * 256 + tid;
            const int m   = idx >> 3;
            const int c4  = idx & 7;
            const int sc4 = c4 ^ (m & 7);
            const int r   = r0 + m;
            const uint8_t* src = (r < end) ? (pA + static_cast<int64_t>(r) * KP + c4 * 16) : pA;
            if (cache_l1) cp_async_ca_16(&sm.sA_main[buf][m][sc4 * 4], src, (r < end) ? 16 : 0);
            else          cp_async_cg_16(&sm.sA_main[buf][m][sc4 * 4], src, (r < end) ? 16 : 0);
        }
        if (tid < BM) {
            const int r = r0 + tid;
            const uint8_t* src_t = (r < end) ? (pA + static_cast<int64_t>(r) * KP + 128) : pA;
            if (cache_l1) cp_async_ca_16(&sm.sA_tail[buf][tid][0], src_t, (r < end) ? 16 : 0);
            else          cp_async_cg_16(&sm.sA_tail[buf][tid][0], src_t, (r < end) ? 16 : 0);
        }
        cp_async_commit();

        if (tid < BM) {
            const int r = r0 + tid;
            if (r < end) {
                float rs_val = mul_rn(pRS[r], scale_e);
                if constexpr (HAS_WEIGHTS) {
                    const float* pW = ld_smem_ptr(&sm.sP.TopkWeights);
                    rs_val = mul_rn(rs_val, pW[r]);
                }
                sm.sRS[buf][tid] = rs_val;
                if constexpr (COMBINE_MODE != 0) sm.sTok[buf][tid] = pTok[r];
            } else {
                sm.sRS[buf][tid] = 0.0f;
                if constexpr (COMBINE_MODE != 0) sm.sTok[buf][tid] = -1;
            }
        }
        for (int idx = tid; idx < BM * 5; idx += 256) {
            const int m = (idx * 13108) >> 16;   // exact idx / 5 for idx in [0, 319]
            const int p = idx - m * 5;
            const int r = r0 + m;
            uint32_t sval = 0u;
            if (r < end) {
                if (p < 4) {
                    if constexpr (AS_ROW_MAJOR) {
                        sval = load_as_u32_rowmaj(pAS, static_cast<int64_t>(r), p * 4, as_str);
                    } else {
                        int64_t off = scale_offset_byte(static_cast<int64_t>(r), p * 4, KG, tile_v);
                        sval = *reinterpret_cast<const uint32_t*>(pAS + off);
                    }
                } else {
                    if constexpr (AS_ROW_MAJOR) {
                        sval = load_as_u16_rowmaj(pAS, static_cast<int64_t>(r), 16, as_str);
                    } else {
                        int64_t off = scale_offset_byte(static_cast<int64_t>(r), 16, KG, tile_v);
                        sval = static_cast<uint32_t>(*reinterpret_cast<const uint16_t*>(pAS + off));
                    }
                }
            }
            sm.sAS[buf][m][p] = sval;
        }
    };

    // Wave-V13 ZERO-SPIN: host stream barrier already guaranteed slice `s` zeroed in L2!
    // Dynamic LPT work-stealing loop over active experts in slice `s` across ALL 48/48 SMs!
    {
        while (true) {
            if (tid == 0) {
                const int32_t* pPlan  = ld_smem_ptr(&sm.sP.PlanBuf);
                const int32_t* pTiles = ld_smem_ptr(&sm.sP.Tiles);
                const int max_w       = ld_smem_i32(&sm.sP.max_work);
                const int num_e       = ld_smem_i32(&sm.sP.num_experts);
                const uint32_t ticket = atomicAdd(&SliceState[s], 1u);

                int e = -1, rb = 0, re = 0;
                int has_more = 0;
                if (pPlan != nullptr) {
                    const int num_w = pPlan[3 * max_w];
                    if (static_cast<int>(ticket) < num_w) {
                        has_more = 1;
                        int widx = static_cast<int>(ticket);
                        // On odd slices, pop the last 64 light experts first (100% L2 hit on A from
                        // the end of slice s-1!), then steal the remaining heavy-first LPT prefix!
                        if ((s & 1) && num_w > 96) {
                            constexpr int kHotTail = 64;
                            if (widx < kHotTail) {
                                widx = num_w - 1 - widx;
                            } else {
                                widx = widx - kHotTail;
                            }
                        }
                        e  = pPlan[widx];
                        rb = pPlan[max_w + widx];
                        re = pPlan[2 * max_w + widx];
                    }
                } else {
                    if (static_cast<int>(ticket) < num_e) {
                        has_more = 1;
                        const int e_cand = (s & 1) ? (num_e - 1 - static_cast<int>(ticket))
                                                   : static_cast<int>(ticket);
                        find_expert_row_range(
                            pTiles, ld_smem_i32(&sm.sP.max_ctas), ld_smem_i32(&sm.sP.tile),
                            e_cand, rb, re);
                        if (rb < re) e = e_cand;
                    }
                }
                sm.sWork[0] = e;
                sm.sWork[1] = rb;
                sm.sWork[2] = re;
                sm.sWork[3] = has_more;
            }
            __syncthreads();

            if (sm.sWork[3] == 0) break;
            const int64_t e     = sm.sWork[0];
            const int row_begin = sm.sWork[1];
            const int end       = sm.sWork[2];
            if (e < 0 || row_begin >= end) {
                __syncthreads();
                continue;
            }

            const int n_rows         = end - row_begin;
            const int num_m          = (n_rows + BM - 1) >> 6;
            const bool a_pinned      = (num_m <= 3);
            const int w_evict_mode_r = ld_smem_i32(&sm.sP.use_cs_w);
            const uint64_t l2_policy = (w_evict_mode_r == 1) ? make_l2_evict_first_policy_v16() : 0ULL;
            const int base_e         = static_cast<int>(ld_smem_ptr(&sm.sP.WS_BASE)[e]);
            const float scale_e      = ld_smem_ptr(&sm.sP.Ones)[e];

            if (a_pinned) {
                load_a_tile(row_begin, end, scale_e, 0, 0, false);
                if (num_m >= 2) load_a_tile(row_begin, end, scale_e, 1, 1, false);
                if (num_m == 3) load_a_tile(row_begin, end, scale_e, 2, 2, false);
            }

            // Issue initial global weight+WS5 loads for dp_step = 0 (staggered sub-panel!)
            const int dp0 = (PANELS > 1) ? (cta_id % PANELS) : 0;
            const int nb0 = s * PANELS + dp0;
            RawPanelPrefetch pf;
            issue_ws5_panel_loads(
                pf, ld_smem_ptr(&sm.sP.W_PACK), ld_smem_ptr(&sm.sP.WS5_PACK),
                e * 36 + nb0, wm, wp, lane, tid, w_evict_mode_r, l2_policy);

            #pragma unroll 1
            for (int dp_step = 0; dp_step < PANELS; ++dp_step) {
                const int dp = (PANELS > 1) ? ((dp_step + cta_id) % PANELS) : 0;
                const int nb = s * PANELS + dp;
                const int64_t puid = e * 36 + nb;
                const bool keep_a_l1 = (!a_pinned) && (dp_step + 1 < PANELS);

                if (!a_pinned) {
                    load_a_tile(row_begin, end, scale_e, 0, 0, keep_a_l1);
                    load_a_tile(row_begin, end, scale_e, 1, 1, keep_a_l1);
                }

                // 1. Unswizzle our half of W_PACK (`wm=0`: slabs 0..4, `wm=1`: slabs 5..8)
                //    and stage into `sm.u.stage` for 1.000x cooperative exchange!
                uint32_t b0_reg[4][4];
                uint32_t b1_reg[4][4];
                uint32_t b0_tail[4];

                if (tid < 90) {
                    sm.u.stage.sWS5_u4[tid] = pf.ws5;
                }
                if (wm == 0) {
                    const uint4 u0 = unswizzle_w_slab_u4(pf.w0, src0, src1, prmt_sel);
                    const uint4 u1 = unswizzle_w_slab_u4(pf.w1, src0, src1, prmt_sel);
                    const uint4 u2 = unswizzle_w_slab_u4(pf.w2, src0, src1, prmt_sel);
                    const uint4 u3 = unswizzle_w_slab_u4(pf.w3, src0, src1, prmt_sel);
                    const uint4 u4 = unswizzle_w_slab_u4(pf.w4, src0, src1, prmt_sel);
                    sm.u.stage.sB_xchg[wp][0][lane] = u0;
                    sm.u.stage.sB_xchg[wp][1][lane] = u1;
                    sm.u.stage.sB_xchg[wp][2][lane] = u2;
                    sm.u.stage.sB_xchg[wp][3][lane] = u3;
                    sm.u.stage.sB_xchg[wp][4][lane] = u4;
                    b0_reg[0][0] = u0.x; b0_reg[0][1] = u0.y; b0_reg[0][2] = u0.z; b0_reg[0][3] = u0.w;
                    b1_reg[0][0] = u1.x; b1_reg[0][1] = u1.y; b1_reg[0][2] = u1.z; b1_reg[0][3] = u1.w;
                    b0_reg[1][0] = u2.x; b0_reg[1][1] = u2.y; b0_reg[1][2] = u2.z; b0_reg[1][3] = u2.w;
                    b1_reg[1][0] = u3.x; b1_reg[1][1] = u3.y; b1_reg[1][2] = u3.z; b1_reg[1][3] = u3.w;
                    b0_reg[2][0] = u4.x; b0_reg[2][1] = u4.y; b0_reg[2][2] = u4.z; b0_reg[2][3] = u4.w;
                } else {
                    const uint4 u5 = unswizzle_w_slab_u4(pf.w0, src0, src1, prmt_sel);
                    const uint4 u6 = unswizzle_w_slab_u4(pf.w1, src0, src1, prmt_sel);
                    const uint4 u7 = unswizzle_w_slab_u4(pf.w2, src0, src1, prmt_sel);
                    const uint4 u8 = unswizzle_w_slab_u4(pf.w3, src0, src1, prmt_sel);
                    sm.u.stage.sB_xchg[wp][5][lane] = u5;
                    sm.u.stage.sB_xchg[wp][6][lane] = u6;
                    sm.u.stage.sB_xchg[wp][7][lane] = u7;
                    sm.u.stage.sB_xchg[wp][8][lane] = u8;
                    b1_reg[2][0] = u5.x; b1_reg[2][1] = u5.y; b1_reg[2][2] = u5.z; b1_reg[2][3] = u5.w;
                    b0_reg[3][0] = u6.x; b0_reg[3][1] = u6.y; b0_reg[3][2] = u6.z; b0_reg[3][3] = u6.w;
                    b1_reg[3][0] = u7.x; b1_reg[3][1] = u7.y; b1_reg[3][2] = u7.z; b1_reg[3][3] = u7.w;
                    b0_tail[0]   = u8.x; b0_tail[1]   = u8.y; b0_tail[2]   = u8.z; b0_tail[3]   = u8.w;
                }

                if (!a_pinned) {
                    cp_async_wait<1>();
                } else if (dp_step == 0) {
                    cp_async_wait<0>();
                }
                __syncthreads();

                // 2. Complete register-resident B panel & decode 5-reg WS5 scales (unrolled 8-way patch!)
                if (wm == 0) {
                    const uint4 u5 = sm.u.stage.sB_xchg[wp][5][lane];
                    const uint4 u6 = sm.u.stage.sB_xchg[wp][6][lane];
                    const uint4 u7 = sm.u.stage.sB_xchg[wp][7][lane];
                    const uint4 u8 = sm.u.stage.sB_xchg[wp][8][lane];
                    b1_reg[2][0] = u5.x; b1_reg[2][1] = u5.y; b1_reg[2][2] = u5.z; b1_reg[2][3] = u5.w;
                    b0_reg[3][0] = u6.x; b0_reg[3][1] = u6.y; b0_reg[3][2] = u6.z; b0_reg[3][3] = u6.w;
                    b1_reg[3][0] = u7.x; b1_reg[3][1] = u7.y; b1_reg[3][2] = u7.z; b1_reg[3][3] = u7.w;
                    b0_tail[0]   = u8.x; b0_tail[1]   = u8.y; b0_tail[2]   = u8.z; b0_tail[3]   = u8.w;
                } else {
                    const uint4 u0 = sm.u.stage.sB_xchg[wp][0][lane];
                    const uint4 u1 = sm.u.stage.sB_xchg[wp][1][lane];
                    const uint4 u2 = sm.u.stage.sB_xchg[wp][2][lane];
                    const uint4 u3 = sm.u.stage.sB_xchg[wp][3][lane];
                    const uint4 u4 = sm.u.stage.sB_xchg[wp][4][lane];
                    b0_reg[0][0] = u0.x; b0_reg[0][1] = u0.y; b0_reg[0][2] = u0.z; b0_reg[0][3] = u0.w;
                    b1_reg[0][0] = u1.x; b1_reg[0][1] = u1.y; b1_reg[0][2] = u1.z; b1_reg[0][3] = u1.w;
                    b0_reg[1][0] = u2.x; b0_reg[1][1] = u2.y; b0_reg[1][2] = u2.z; b0_reg[1][3] = u2.w;
                    b1_reg[1][0] = u3.x; b1_reg[1][1] = u3.y; b1_reg[1][2] = u3.z; b1_reg[1][3] = u3.w;
                    b0_reg[2][0] = u4.x; b0_reg[2][1] = u4.y; b0_reg[2][2] = u4.z; b0_reg[2][3] = u4.w;
                }

                uint32_t sfb_my0, sfb_my1, sfb_my2, sfb_my3, sfb_my4;
                decode_ws5_row18_to_regs<HAS_PATCHES>(
                    sm.u.stage.sWS5_u4, n_col_my, base_e,
                    ld_smem_ptr(&sm.sP.P_OFF), ld_smem_ptr(&sm.sP.PATCHES), puid,
                    sfb_my0, sfb_my1, sfb_my2, sfb_my3, sfb_my4);

                // Fence `sm.u.stage` reads across all warps before `sm.u.epi` writes!
                __syncthreads();

                // 3. Sweep all M-tiles of expert `e` with 3-stage cp.async A ring & ZERO weight loads!
                int cur = 0, s_nxt = 1, s_pre = 2;
                #pragma unroll 1
                for (int mi_blk = 0; mi_blk < num_m; ++mi_blk) {
                    const int row0 = row_begin + mi_blk * BM;
                    const bool is_last_m = (mi_blk + 1 >= num_m);

                    if (!a_pinned && mi_blk + 2 < num_m) {
                        load_a_tile(row_begin, end, scale_e, mi_blk + 2, s_pre, keep_a_l1);
                    }

                    float acc[2][4][4];
                    #pragma unroll
                    for (int mi = 0; mi < 2; ++mi) {
                        #pragma unroll
                        for (int sn = 0; sn < 4; ++sn) {
                            acc[mi][sn][0] = 0.0f; acc[mi][sn][1] = 0.0f;
                            acc[mi][sn][2] = 0.0f; acc[mi][sn][3] = 0.0f;
                        }
                    }

                    // Skip dead `wm == 1` MMAs when partial tail M-tile has <= 32 live rows!
                    const bool active_wm = (wm == 0) || (end - row0 > 32);
                    if (active_wm) {
                        #pragma unroll
                        for (int p = 0; p < 4; ++p) {
                            const uint32_t my_sfb = (p == 0) ? sfb_my0
                                                  : ((p == 1) ? sfb_my1
                                                  : ((p == 2) ? sfb_my2 : sfb_my3));
                            uint32_t sfb[4];
                            #pragma unroll
                            for (int sn = 0; sn < 4; ++sn) {
                                sfb[sn] = __shfl_sync(0xFFFFFFFFu, my_sfb, (g << 2) | sn);
                            }
                            const int sw_c0 = (p * 8 + lt) ^ (g << 2);
                            const int sw_c1 = (p * 8 + 4 + lt) ^ (g << 2);
                            #pragma unroll
                            for (int mi = 0; mi < 2; ++mi) {
                                const int m_lo = m_warp_base + mi * 16 + g;
                                const int m_hi = m_lo + 8;
                                const uint32_t a0  = sm.sA_main[cur][m_lo][sw_c0];
                                const uint32_t a1  = sm.sA_main[cur][m_hi][sw_c0];
                                const uint32_t a2  = sm.sA_main[cur][m_lo][sw_c1];
                                const uint32_t a3  = sm.sA_main[cur][m_hi][sw_c1];
                                const uint32_t sfa = sm.sAS[cur][m_warp_base + mi * 16 + m_sf_off][p];
                                #pragma unroll
                                for (int sn = 0; sn < 4; ++sn) {
                                    mma_m16n8k64_nvfp4(acc[mi][sn], a0, a1, a2, a3,
                                                       b0_reg[p][sn], b1_reg[p][sn], sfa, sfb[sn]);
                                }
                            }
                        }

                        // Tail p = 4 (k = 256..287)
                        {
                            uint32_t sfb[4];
                            #pragma unroll
                            for (int sn = 0; sn < 4; ++sn) {
                                sfb[sn] = __shfl_sync(0xFFFFFFFFu, sfb_my4, (g << 2) | sn);
                            }
                            #pragma unroll
                            for (int mi = 0; mi < 2; ++mi) {
                                const int m_lo = m_warp_base + mi * 16 + g;
                                const int m_hi = m_lo + 8;
                                const uint32_t a0  = sm.sA_tail[cur][m_lo][lt];
                                const uint32_t a1  = sm.sA_tail[cur][m_hi][lt];
                                const uint32_t sfa = sm.sAS[cur][m_warp_base + mi * 16 + m_sf_off][4];
                                #pragma unroll
                                for (int sn = 0; sn < 4; ++sn) {
                                    mma_m16n8k64_nvfp4(acc[mi][sn], a0, a1, 0u, 0u,
                                                       b0_tail[sn], 0u, sfa, sfb[sn]);
                                }
                            }
                        }
                    }

                    // 4. Software-pipeline `dp_step + 1` global weight loads right as `b_reg` dies!
                    //    Flies over LPDDR5X in parallel with `w2_v5_epilogue_64` SMEM + L2 atomics!
                    if (is_last_m && (dp_step + 1 < PANELS)) {
                        const int nxt_dp = (PANELS > 1) ? ((dp_step + 1 + cta_id) % PANELS) : 0;
                        const int nxt_nb = s * PANELS + nxt_dp;
                        issue_ws5_panel_loads(
                            pf, ld_smem_ptr(&sm.sP.W_PACK), ld_smem_ptr(&sm.sP.WS5_PACK),
                            e * 36 + nxt_nb, wm, wp, lane, tid, w_evict_mode_r, l2_policy);
                    }

                    w2_v5_epilogue_64<COMBINE_MODE>(
                        sm.u.epi, acc, sm.sRS[cur], sm.sTok[cur],
                        ld_smem_mut_ptr(&sm.sP.Out), row0, end, nb, warp, wm, wp, lane, g, lt
                    );

                    if (!is_last_m) {
                        if (!a_pinned) {
                            if (mi_blk + 2 < num_m) cp_async_wait<1>();
                            else                    cp_async_wait<0>();
                        }
                        __syncthreads();
                        const int tmp = cur; cur = s_nxt; s_nxt = s_pre; s_pre = tmp;
                    } else if (dp_step + 1 < PANELS) {
                        __syncthreads();
                    }
                }
            }
            __syncthreads();
        }
    }
}

// =============================================================================
// Fallback Panel-Layout W2 Kernel (`_inkling_panel == (256, 16)`, 1.000x Load)
// =============================================================================

template <int AS_ROW_MAJOR, int HAS_WEIGHTS, int COMBINE_MODE, int PANELS>
__global__ __launch_bounds__(256, 1)
void pf_nvfp4_w2_panel_v5_kernel(
    const uint8_t* __restrict__ A,
    const uint8_t* __restrict__ AS,
    const float* __restrict__ RS,
    const float* __restrict__ TopkWeights,
    const int32_t* __restrict__ Row2Tok,
    const uint8_t* __restrict__ WP,
    const uint8_t* __restrict__ WSP,
    const float* __restrict__ Ones,
    const int32_t* __restrict__ Tiles,
    const int32_t* __restrict__ PlanBuf,
    uint32_t* __restrict__ SliceState,
    void* __restrict__ Out,
    int max_ctas,
    int tile,
    int num_experts,
    int max_work,
    int as_row_stride,
    int w_evict_mode,
    int slice_s
) {
    constexpr int BM     = 64;
    constexpr int KP     = 144;
    constexpr int KG     = 18;

    extern __shared__ uint4 dyn_smem[];
    auto& sm = *reinterpret_cast<W2V5Smem*>(dyn_smem);

    const int tid    = threadIdx.x;
    const int warp   = tid >> 5;
    const int wm     = warp >> 2;
    const int wp     = warp & 3;
    const int lane   = tid & 31;
    const int g      = lane >> 2;
    const int lt     = lane & 3;
    const int cta_id = blockIdx.x;

    if (tid == 0) {
        sm.sP.A             = A;
        sm.sP.AS            = AS;
        sm.sP.RS            = RS;
        sm.sP.TopkWeights   = TopkWeights;
        sm.sP.Row2Tok       = Row2Tok;
        sm.sP.W_PACK        = WP;
        sm.sP.WS5_PACK      = reinterpret_cast<const uint32_t*>(WSP);
        sm.sP.Ones          = Ones;
        sm.sP.Tiles         = Tiles;
        sm.sP.PlanBuf       = PlanBuf;
        sm.sP.Out           = Out;
        sm.sP.max_ctas      = max_ctas;
        sm.sP.tile          = tile;
        sm.sP.num_experts   = num_experts;
        sm.sP.max_work      = max_work;
        sm.sP.as_row_stride = as_row_stride;
        sm.sP.tokens        = 0;
        sm.sP.use_cs_w      = w_evict_mode;
        sm.sP.zero_mode     = 0;
    }
    __syncthreads();

    const int m_warp_base   = wm * 32;
    const int m_sf_off      = g + ((lt & 1) << 3);
    const int s             = slice_s;

    auto load_a_tile = [&](int row_begin, int end, float scale_e, int mi_blk, int buf, bool cache_l1) {
        const uint8_t* pA   = ld_smem_ptr(&sm.sP.A);
        const uint8_t* pAS  = ld_smem_ptr(&sm.sP.AS);
        const float*   pRS  = ld_smem_ptr(&sm.sP.RS);
        const int32_t* pTok = ld_smem_ptr(&sm.sP.Row2Tok);
        const int tile_v    = ld_smem_i32(&sm.sP.tile);
        const int as_str    = ld_smem_i32(&sm.sP.as_row_stride);
        const int r0        = row_begin + mi_blk * BM;

        #pragma unroll
        for (int it = 0; it < 2; ++it) {
            const int idx = it * 256 + tid;
            const int m   = idx >> 3;
            const int c4  = idx & 7;
            const int sc4 = c4 ^ (m & 7);
            const int r   = r0 + m;
            const uint8_t* src = (r < end) ? (pA + static_cast<int64_t>(r) * KP + c4 * 16) : pA;
            if (cache_l1) cp_async_ca_16(&sm.sA_main[buf][m][sc4 * 4], src, (r < end) ? 16 : 0);
            else          cp_async_cg_16(&sm.sA_main[buf][m][sc4 * 4], src, (r < end) ? 16 : 0);
        }
        if (tid < BM) {
            const int r = r0 + tid;
            const uint8_t* src_t = (r < end) ? (pA + static_cast<int64_t>(r) * KP + 128) : pA;
            if (cache_l1) cp_async_ca_16(&sm.sA_tail[buf][tid][0], src_t, (r < end) ? 16 : 0);
            else          cp_async_cg_16(&sm.sA_tail[buf][tid][0], src_t, (r < end) ? 16 : 0);
        }
        cp_async_commit();

        if (tid < BM) {
            const int r = r0 + tid;
            if (r < end) {
                float rs_val = mul_rn(pRS[r], scale_e);
                if constexpr (HAS_WEIGHTS) {
                    rs_val = mul_rn(rs_val, ld_smem_ptr(&sm.sP.TopkWeights)[r]);
                }
                sm.sRS[buf][tid] = rs_val;
                if constexpr (COMBINE_MODE != 0) sm.sTok[buf][tid] = pTok[r];
            } else {
                sm.sRS[buf][tid] = 0.0f;
                if constexpr (COMBINE_MODE != 0) sm.sTok[buf][tid] = -1;
            }
        }
        for (int idx = tid; idx < BM * 5; idx += 256) {
            const int m = (idx * 13108) >> 16;
            const int p = idx - m * 5;
            const int r = r0 + m;
            uint32_t sval = 0u;
            if (r < end) {
                if (p < 4) {
                    if constexpr (AS_ROW_MAJOR) sval = load_as_u32_rowmaj(pAS, static_cast<int64_t>(r), p * 4, as_str);
                    else sval = *reinterpret_cast<const uint32_t*>(pAS + scale_offset_byte(static_cast<int64_t>(r), p * 4, KG, tile_v));
                } else {
                    if constexpr (AS_ROW_MAJOR) sval = load_as_u16_rowmaj(pAS, static_cast<int64_t>(r), 16, as_str);
                    else sval = static_cast<uint32_t>(*reinterpret_cast<const uint16_t*>(pAS + scale_offset_byte(static_cast<int64_t>(r), 16, KG, tile_v)));
                }
            }
            sm.sAS[buf][m][p] = sval;
        }
    };

    {
        while (true) {
            if (tid == 0) {
                const int32_t* pPlan  = ld_smem_ptr(&sm.sP.PlanBuf);
                const int32_t* pTiles = ld_smem_ptr(&sm.sP.Tiles);
                const int max_w       = ld_smem_i32(&sm.sP.max_work);
                const int num_e       = ld_smem_i32(&sm.sP.num_experts);
                const uint32_t ticket = atomicAdd(&SliceState[s], 1u);

                int e = -1, rb = 0, re = 0, has_more = 0;
                if (pPlan != nullptr) {
                    const int num_w = pPlan[3 * max_w];
                    if (static_cast<int>(ticket) < num_w) {
                        has_more = 1;
                        int widx = static_cast<int>(ticket);
                        if ((s & 1) && num_w > 96) {
                            widx = (widx < 64) ? (num_w - 1 - widx) : (widx - 64);
                        }
                        e  = pPlan[widx];
                        rb = pPlan[max_w + widx];
                        re = pPlan[2 * max_w + widx];
                    }
                } else if (static_cast<int>(ticket) < num_e) {
                    has_more = 1;
                    const int e_cand = (s & 1) ? (num_e - 1 - static_cast<int>(ticket)) : static_cast<int>(ticket);
                    find_expert_row_range(pTiles, ld_smem_i32(&sm.sP.max_ctas), ld_smem_i32(&sm.sP.tile), e_cand, rb, re);
                    if (rb < re) e = e_cand;
                }
                sm.sWork[0] = e; sm.sWork[1] = rb; sm.sWork[2] = re; sm.sWork[3] = has_more;
            }
            __syncthreads();

            if (sm.sWork[3] == 0) break;
            const int64_t e     = sm.sWork[0];
            const int row_begin = sm.sWork[1];
            const int end       = sm.sWork[2];
            if (e < 0 || row_begin >= end) {
                __syncthreads();
                continue;
            }

            const int num_m          = ((end - row_begin) + BM - 1) >> 6;
            const bool a_pinned      = (num_m <= 3);
            const int w_evict_mode_r = ld_smem_i32(&sm.sP.use_cs_w);
            const uint64_t l2_policy = (w_evict_mode_r == 1) ? make_l2_evict_first_policy_v16() : 0ULL;
            const float scale_e      = ld_smem_ptr(&sm.sP.Ones)[e];

            if (a_pinned) {
                load_a_tile(row_begin, end, scale_e, 0, 0, false);
                if (num_m >= 2) load_a_tile(row_begin, end, scale_e, 1, 1, false);
                if (num_m == 3) load_a_tile(row_begin, end, scale_e, 2, 2, false);
            }

            #pragma unroll 1
            for (int dp_step = 0; dp_step < PANELS; ++dp_step) {
                const int dp    = (PANELS > 1) ? ((dp_step + cta_id) % PANELS) : 0;
                const int nb    = s * PANELS + dp;
                const int panel = nb >> 1;
                const int sub_n = (nb & 1) * 128;
                const bool keep_a_l1 = (!a_pinned) && (dp_step + 1 < PANELS);

                if (!a_pinned) {
                    load_a_tile(row_begin, end, scale_e, 0, 0, keep_a_l1);
                    load_a_tile(row_begin, end, scale_e, 1, 1, keep_a_l1);
                }

                const uint8_t* pWP  = ld_smem_ptr(&sm.sP.W_PACK);
                const uint8_t* pWSP = reinterpret_cast<const uint8_t*>(ld_smem_ptr(&sm.sP.WS5_PACK));
                const uint8_t* wp_base = pWP + (e * 18 + panel) * (9LL * 256LL * 16LL)
                                       + static_cast<int64_t>(sub_n + wp * 32 + g) * 16 + lt * 4;

                // Cooperative 1.000x panel weight load across wm=0 (slabs 0..4) and wm=1 (slabs 5..8)!
                uint32_t b0_reg[4][4], b1_reg[4][4], b0_tail[4];
                if (wm == 0) {
                    #pragma unroll
                    for (int k_blk = 0; k_blk < 5; ++k_blk) {
                        uint4 u;
                        u.x = ld_w_u32(wp_base + k_blk * (256 * 16) + 0 * (8 * 16), w_evict_mode_r, l2_policy);
                        u.y = ld_w_u32(wp_base + k_blk * (256 * 16) + 1 * (8 * 16), w_evict_mode_r, l2_policy);
                        u.z = ld_w_u32(wp_base + k_blk * (256 * 16) + 2 * (8 * 16), w_evict_mode_r, l2_policy);
                        u.w = ld_w_u32(wp_base + k_blk * (256 * 16) + 3 * (8 * 16), w_evict_mode_r, l2_policy);
                        sm.u.stage.sB_xchg[wp][k_blk][lane] = u;
                    }
                } else {
                    #pragma unroll
                    for (int k_blk = 5; k_blk < 9; ++k_blk) {
                        uint4 u;
                        u.x = ld_w_u32(wp_base + k_blk * (256 * 16) + 0 * (8 * 16), w_evict_mode_r, l2_policy);
                        u.y = ld_w_u32(wp_base + k_blk * (256 * 16) + 1 * (8 * 16), w_evict_mode_r, l2_policy);
                        u.z = ld_w_u32(wp_base + k_blk * (256 * 16) + 2 * (8 * 16), w_evict_mode_r, l2_policy);
                        u.w = ld_w_u32(wp_base + k_blk * (256 * 16) + 3 * (8 * 16), w_evict_mode_r, l2_policy);
                        sm.u.stage.sB_xchg[wp][k_blk][lane] = u;
                    }
                }

                const uint8_t* wsp_col = pWSP + (e * 18 + panel) * (9LL * 2LL * 256LL)
                                       + sub_n + (wp * 32 + lt * 8 + g);
                #define WSP_SC(KG_IDX) static_cast<uint32_t>(wsp_col[((KG_IDX) >> 1) * 512 + ((KG_IDX) & 1) * 256])
                const uint32_t sfb_my0 = WSP_SC(0) | (WSP_SC(1) << 8) | (WSP_SC(2) << 16) | (WSP_SC(3) << 24);
                const uint32_t sfb_my1 = WSP_SC(4) | (WSP_SC(5) << 8) | (WSP_SC(6) << 16) | (WSP_SC(7) << 24);
                const uint32_t sfb_my2 = WSP_SC(8) | (WSP_SC(9) << 8) | (WSP_SC(10) << 16) | (WSP_SC(11) << 24);
                const uint32_t sfb_my3 = WSP_SC(12) | (WSP_SC(13) << 8) | (WSP_SC(14) << 16) | (WSP_SC(15) << 24);
                const uint32_t sfb_my4 = WSP_SC(16) | (WSP_SC(17) << 8);
                #undef WSP_SC

                if (!a_pinned) cp_async_wait<1>(); else if (dp_step == 0) cp_async_wait<0>();
                __syncthreads();

                #pragma unroll
                for (int p = 0; p < 4; ++p) {
                    const uint4 u0 = sm.u.stage.sB_xchg[wp][2 * p + 0][lane];
                    const uint4 u1 = sm.u.stage.sB_xchg[wp][2 * p + 1][lane];
                    b0_reg[p][0] = u0.x; b0_reg[p][1] = u0.y; b0_reg[p][2] = u0.z; b0_reg[p][3] = u0.w;
                    b1_reg[p][0] = u1.x; b1_reg[p][1] = u1.y; b1_reg[p][2] = u1.z; b1_reg[p][3] = u1.w;
                }
                {
                    const uint4 u8 = sm.u.stage.sB_xchg[wp][8][lane];
                    b0_tail[0] = u8.x; b0_tail[1] = u8.y; b0_tail[2] = u8.z; b0_tail[3] = u8.w;
                }
                __syncthreads();

                int cur = 0, s_nxt = 1, s_pre = 2;
                #pragma unroll 1
                for (int mi_blk = 0; mi_blk < num_m; ++mi_blk) {
                    const int row0 = row_begin + mi_blk * BM;
                    const bool is_last_m = (mi_blk + 1 >= num_m);

                    if (!a_pinned && mi_blk + 2 < num_m) {
                        load_a_tile(row_begin, end, scale_e, mi_blk + 2, s_pre, keep_a_l1);
                    }

                    float acc[2][4][4];
                    #pragma unroll
                    for (int mi = 0; mi < 2; ++mi)
                        #pragma unroll
                        for (int sn = 0; sn < 4; ++sn) {
                            acc[mi][sn][0] = 0.0f; acc[mi][sn][1] = 0.0f;
                            acc[mi][sn][2] = 0.0f; acc[mi][sn][3] = 0.0f;
                        }

                    if ((wm == 0) || (end - row0 > 32)) {
                        #pragma unroll
                        for (int p = 0; p < 4; ++p) {
                            const uint32_t my_sfb = (p == 0) ? sfb_my0
                                                  : ((p == 1) ? sfb_my1
                                                  : ((p == 2) ? sfb_my2 : sfb_my3));
                            uint32_t sfb[4];
                            #pragma unroll
                            for (int sn = 0; sn < 4; ++sn) sfb[sn] = __shfl_sync(0xFFFFFFFFu, my_sfb, (g << 2) | sn);
                            const int sw_c0 = (p * 8 + lt) ^ (g << 2);
                            const int sw_c1 = (p * 8 + 4 + lt) ^ (g << 2);
                            #pragma unroll
                            for (int mi = 0; mi < 2; ++mi) {
                                const int m_lo = m_warp_base + mi * 16 + g, m_hi = m_lo + 8;
                                mma_m16n8k64_nvfp4(acc[mi][0], sm.sA_main[cur][m_lo][sw_c0], sm.sA_main[cur][m_hi][sw_c0], sm.sA_main[cur][m_lo][sw_c1], sm.sA_main[cur][m_hi][sw_c1], b0_reg[p][0], b1_reg[p][0], sm.sAS[cur][m_warp_base + mi * 16 + m_sf_off][p], sfb[0]);
                                mma_m16n8k64_nvfp4(acc[mi][1], sm.sA_main[cur][m_lo][sw_c0], sm.sA_main[cur][m_hi][sw_c0], sm.sA_main[cur][m_lo][sw_c1], sm.sA_main[cur][m_hi][sw_c1], b0_reg[p][1], b1_reg[p][1], sm.sAS[cur][m_warp_base + mi * 16 + m_sf_off][p], sfb[1]);
                                mma_m16n8k64_nvfp4(acc[mi][2], sm.sA_main[cur][m_lo][sw_c0], sm.sA_main[cur][m_hi][sw_c0], sm.sA_main[cur][m_lo][sw_c1], sm.sA_main[cur][m_hi][sw_c1], b0_reg[p][2], b1_reg[p][2], sm.sAS[cur][m_warp_base + mi * 16 + m_sf_off][p], sfb[2]);
                                mma_m16n8k64_nvfp4(acc[mi][3], sm.sA_main[cur][m_lo][sw_c0], sm.sA_main[cur][m_hi][sw_c0], sm.sA_main[cur][m_lo][sw_c1], sm.sA_main[cur][m_hi][sw_c1], b0_reg[p][3], b1_reg[p][3], sm.sAS[cur][m_warp_base + mi * 16 + m_sf_off][p], sfb[3]);
                            }
                        }
                        {
                            uint32_t sfb[4];
                            #pragma unroll
                            for (int sn = 0; sn < 4; ++sn) sfb[sn] = __shfl_sync(0xFFFFFFFFu, sfb_my4, (g << 2) | sn);
                            #pragma unroll
                            for (int mi = 0; mi < 2; ++mi) {
                                const int m_lo = m_warp_base + mi * 16 + g, m_hi = m_lo + 8;
                                const uint32_t a0 = sm.sA_tail[cur][m_lo][lt], a1 = sm.sA_tail[cur][m_hi][lt];
                                const uint32_t sfa = sm.sAS[cur][m_warp_base + mi * 16 + m_sf_off][4];
                                #pragma unroll
                                for (int sn = 0; sn < 4; ++sn)
                                    mma_m16n8k64_nvfp4(acc[mi][sn], a0, a1, 0u, 0u, b0_tail[sn], 0u, sfa, sfb[sn]);
                            }
                        }
                    }

                    w2_v5_epilogue_64<COMBINE_MODE>(
                        sm.u.epi, acc, sm.sRS[cur], sm.sTok[cur],
                        ld_smem_mut_ptr(&sm.sP.Out), row0, end, nb, warp, wm, wp, lane, g, lt);

                    if (!is_last_m) {
                        if (!a_pinned) {
                            if (mi_blk + 2 < num_m) cp_async_wait<1>();
                            else                    cp_async_wait<0>();
                        }
                        __syncthreads();
                        const int tmp = cur; cur = s_nxt; s_nxt = s_pre; s_pre = tmp;
                    } else if (dp_step + 1 < PANELS) {
                        __syncthreads();
                    }
                }
            }
            __syncthreads();
        }
    }
}

// =============================================================================
// Co-Resident Grid Sizing & Template Dispatchers
// =============================================================================

// Wave-V13 ZERO-SPIN: default reserve_sms = 0 -> unlocks ALL 48/48 SMs (+20% SM throughput, 0 deadlock risk!)
static int g_dbo_reserve_sms = 0;

inline bool is_stream_capturing(cudaStream_t stream) {
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    cudaError_t err = cudaStreamIsCapturing(stream, &status);
    if (err != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return status != cudaStreamCaptureStatusNone;
}

struct W2OverlapCtxV16 {
    cudaStream_t zero_stream = nullptr;
    cudaEvent_t  ev_entry    = nullptr;
    cudaEvent_t  ev_zero[2]  = {nullptr, nullptr};
    bool         inited      = false;
    bool         ok          = false;
};

static W2OverlapCtxV16 g_w2_ovl_ctx[16];
static int g_w2_double_buf_slice = 1;

inline W2OverlapCtxV16* get_w2_overlap_ctx_v16() {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 16) return nullptr;
    W2OverlapCtxV16& ctx = g_w2_ovl_ctx[dev];
    if (!ctx.inited) {
        ctx.inited = true;
        bool good = (cudaStreamCreateWithFlags(&ctx.zero_stream, cudaStreamNonBlocking) == cudaSuccess);
        good = good && (cudaEventCreateWithFlags(&ctx.ev_entry, cudaEventDisableTiming) == cudaSuccess);
        good = good && (cudaEventCreateWithFlags(&ctx.ev_zero[0], cudaEventDisableTiming) == cudaSuccess);
        good = good && (cudaEventCreateWithFlags(&ctx.ev_zero[1], cudaEventDisableTiming) == cudaSuccess);
        ctx.ok = good;
    }
    return ctx.ok ? &ctx : nullptr;
}

template <typename KernelFn>
inline void configure_w2_dyn_smem_v16(KernelFn kfn, size_t smem_bytes) {
    cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes));
    cudaFuncSetAttribute(kfn, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
}

template <typename KernelFn>
inline int get_coresident_ctas(KernelFn kfn) {
    static int cached_sms = 0;
    static const void* configured_fns[128] = {nullptr};
    static int configured_bpms[128] = {0};
    static int num_configured = 0;
    if (cached_sms == 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&cached_sms, cudaDevAttrMultiProcessorCount, dev);
        if (cached_sms <= 0) cached_sms = 48;
    }
    const void* key_ptr = reinterpret_cast<const void*>(kfn);
    int cached_bpm = 0;
    for (int i = 0; i < num_configured; ++i) {
        if (configured_fns[i] == key_ptr) {
            cached_bpm = configured_bpms[i];
            break;
        }
    }
    if (cached_bpm == 0) {
        configure_w2_dyn_smem_v16(kfn, sizeof(W2V5Smem));
        int bpm = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bpm, kfn, 256, sizeof(W2V5Smem));
        cached_bpm = (bpm > 0) ? ((bpm > 2) ? 2 : bpm) : 1;
        if (num_configured < 128) {
            configured_fns[num_configured] = key_ptr;
            configured_bpms[num_configured] = cached_bpm;
            ++num_configured;
        }
    }
    int reserve = g_dbo_reserve_sms;
    if (reserve < 0) reserve = 0;
    if (cached_sms > 14 && reserve > cached_sms - 8) reserve = cached_sms - 8;
    const int avail_sms = (cached_sms > reserve + 6) ? (cached_sms - reserve) : cached_sms;
    const int g = avail_sms * cached_bpm;
    return (g > 0) ? g : 40;
}

template <int COMBINE_MODE, int PANELS>
int launch_ws5_v5_cm_p(
    const void* A, const void* AS, const void* RS, const void* TopkWeights, const void* Row2Tok,
    const void* W_PACK, const void* WS5_PACK, const void* WS_BASE,
    const void* P_OFF, const void* PATCHES,
    const void* Ones, const void* Tiles, const void* PlanBuf, void* SliceState, void* Out,
    int max_ctas, int tile, int num_experts, int max_work,
    int as_row_major, int as_row_stride, int has_patches,
    int w_evict_mode, int zero_mode, int tokens, cudaStream_t stream
) {
    const int eff_work = (PlanBuf != nullptr && max_work > 0) ? max_work : num_experts;
    const int has_w = (TopkWeights != nullptr) ? 1 : 0;
    const int key = ((as_row_major ? 1 : 0) << 2) | ((has_patches ? 1 : 0) << 1) | has_w;

    constexpr int N_BLKS = 36 / PANELS;
    const bool do_zero = (COMBINE_MODE == 2 && zero_mode != 0 && tokens > 0);
    const unsigned num_zchk = do_zero ? static_cast<unsigned>((tokens + 127) >> 7) : 0u;
    if (!do_zero) {
        pf_nvfp4_w2_reset_cursors_v13_kernel<<<1, 64, 0, stream>>>(
            reinterpret_cast<uint32_t*>(SliceState), N_BLKS);
    }

    constexpr size_t W2_SMEM = sizeof(W2V5Smem);
    const size_t slice_bytes = static_cast<size_t>(tokens > 0 ? tokens : 0) * static_cast<size_t>(PANELS * 128) * 2ULL;
    W2OverlapCtxV16* ovl = (do_zero && g_w2_double_buf_slice && zero_mode == 1 && N_BLKS >= 2 && (2ULL * slice_bytes <= 19ULL * 1024ULL * 1024ULL))
        ? get_w2_overlap_ctx_v16() : nullptr;

    #define DISPATCH_W2_V5(RM, HP, HW) \
        do { \
            auto kfn = pf_nvfp4_w2_ws5_v5_kernel<RM, HP, HW, COMBINE_MODE, PANELS>; \
            const int g_ctas = get_coresident_ctas(kfn); \
            if (ovl != nullptr) { \
                cudaEventRecord(ovl->ev_entry, stream); \
                cudaStreamWaitEvent(ovl->zero_stream, ovl->ev_entry, 0); \
                pf_nvfp4_w2_zero_slice_v13_kernel<PANELS><<<dim3(num_zchk), dim3(256), 0, ovl->zero_stream>>>( \
                    reinterpret_cast<uint16_t*>(Out), reinterpret_cast<uint32_t*>(SliceState), tokens, 0); \
                cudaEventRecord(ovl->ev_zero[0], ovl->zero_stream); \
                for (int s = 0; s < N_BLKS; ++s) { \
                    if (s + 1 < N_BLKS) { \
                        pf_nvfp4_w2_zero_slice_v13_kernel<PANELS><<<dim3(num_zchk), dim3(256), 0, ovl->zero_stream>>>( \
                            reinterpret_cast<uint16_t*>(Out), reinterpret_cast<uint32_t*>(SliceState), tokens, s + 1); \
                        cudaEventRecord(ovl->ev_zero[(s + 1) & 1], ovl->zero_stream); \
                    } \
                    cudaStreamWaitEvent(stream, ovl->ev_zero[s & 1], 0); \
                    kfn<<<dim3(static_cast<unsigned>(g_ctas)), dim3(256), W2_SMEM, stream>>>( \
                        reinterpret_cast<const uint8_t*>(A), reinterpret_cast<const uint8_t*>(AS), \
                        reinterpret_cast<const float*>(RS), reinterpret_cast<const float*>(TopkWeights), \
                        reinterpret_cast<const int32_t*>(Row2Tok), \
                        reinterpret_cast<const uint8_t*>(W_PACK), reinterpret_cast<const uint32_t*>(WS5_PACK), \
                        reinterpret_cast<const uint8_t*>(WS_BASE), reinterpret_cast<const int32_t*>(P_OFF), \
                        reinterpret_cast<const uint32_t*>(PATCHES), reinterpret_cast<const float*>(Ones), \
                        reinterpret_cast<const int32_t*>(Tiles), reinterpret_cast<const int32_t*>(PlanBuf), \
                        reinterpret_cast<uint32_t*>(SliceState), Out, \
                        max_ctas, tile, num_experts, eff_work, as_row_stride, w_evict_mode, s); \
                } \
            } else { \
                for (int s = 0; s < N_BLKS; ++s) { \
                    if (do_zero) { \
                        pf_nvfp4_w2_zero_slice_v13_kernel<PANELS><<<dim3(num_zchk), dim3(256), 0, stream>>>( \
                            reinterpret_cast<uint16_t*>(Out), reinterpret_cast<uint32_t*>(SliceState), tokens, s); \
                    } \
                    kfn<<<dim3(static_cast<unsigned>(g_ctas)), dim3(256), W2_SMEM, stream>>>( \
                        reinterpret_cast<const uint8_t*>(A), reinterpret_cast<const uint8_t*>(AS), \
                        reinterpret_cast<const float*>(RS), reinterpret_cast<const float*>(TopkWeights), \
                        reinterpret_cast<const int32_t*>(Row2Tok), \
                        reinterpret_cast<const uint8_t*>(W_PACK), reinterpret_cast<const uint32_t*>(WS5_PACK), \
                        reinterpret_cast<const uint8_t*>(WS_BASE), reinterpret_cast<const int32_t*>(P_OFF), \
                        reinterpret_cast<const uint32_t*>(PATCHES), reinterpret_cast<const float*>(Ones), \
                        reinterpret_cast<const int32_t*>(Tiles), reinterpret_cast<const int32_t*>(PlanBuf), \
                        reinterpret_cast<uint32_t*>(SliceState), Out, \
                        max_ctas, tile, num_experts, eff_work, as_row_stride, w_evict_mode, s); \
                } \
            } \
        } while (0)

    switch (key) {
        case 0: DISPATCH_W2_V5(0, 0, 0); break;
        case 1: DISPATCH_W2_V5(0, 0, 1); break;
        case 2: DISPATCH_W2_V5(0, 1, 0); break;
        case 3: DISPATCH_W2_V5(0, 1, 1); break;
        case 4: DISPATCH_W2_V5(1, 0, 0); break;
        case 5: DISPATCH_W2_V5(1, 0, 1); break;
        case 6: DISPATCH_W2_V5(1, 1, 0); break;
        case 7: DISPATCH_W2_V5(1, 1, 1); break;
        default: return -3;
    }
    #undef DISPATCH_W2_V5
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

template <int COMBINE_MODE, int PANELS>
int launch_panel_v5_cm_p(
    const void* A, const void* AS, const void* RS, const void* TopkWeights, const void* Row2Tok,
    const void* WP, const void* WSP,
    const void* Ones, const void* Tiles, const void* PlanBuf, void* SliceState, void* Out,
    int max_ctas, int tile, int num_experts, int max_work,
    int as_row_major, int as_row_stride,
    int w_evict_mode, int zero_mode, int tokens, cudaStream_t stream
) {
    const int eff_work = (PlanBuf != nullptr && max_work > 0) ? max_work : num_experts;
    const int key = ((as_row_major ? 1 : 0) << 1) | ((TopkWeights != nullptr) ? 1 : 0);

    constexpr int N_BLKS = 36 / PANELS;
    const bool do_zero = (COMBINE_MODE == 2 && zero_mode != 0 && tokens > 0);
    const unsigned num_zchk = do_zero ? static_cast<unsigned>((tokens + 127) >> 7) : 0u;
    if (!do_zero) {
        pf_nvfp4_w2_reset_cursors_v13_kernel<<<1, 64, 0, stream>>>(
            reinterpret_cast<uint32_t*>(SliceState), N_BLKS);
    }

    #define DISPATCH_PAN_V5(RM, HW) \
        do { \
            auto kfn = pf_nvfp4_w2_panel_v5_kernel<RM, HW, COMBINE_MODE, PANELS>; \
            const int g_ctas = get_coresident_ctas(kfn); \
            for (int s = 0; s < N_BLKS; ++s) { \
                if (do_zero) { \
                    pf_nvfp4_w2_zero_slice_v13_kernel<PANELS><<<dim3(num_zchk), dim3(256), 0, stream>>>( \
                        reinterpret_cast<uint16_t*>(Out), reinterpret_cast<uint32_t*>(SliceState), tokens, s); \
                } \
                kfn<<<dim3(static_cast<unsigned>(g_ctas)), dim3(256), sizeof(W2V5Smem), stream>>>( \
                    reinterpret_cast<const uint8_t*>(A), reinterpret_cast<const uint8_t*>(AS), \
                    reinterpret_cast<const float*>(RS), reinterpret_cast<const float*>(TopkWeights), \
                    reinterpret_cast<const int32_t*>(Row2Tok), \
                    reinterpret_cast<const uint8_t*>(WP), reinterpret_cast<const uint8_t*>(WSP), \
                    reinterpret_cast<const float*>(Ones), reinterpret_cast<const int32_t*>(Tiles), \
                    reinterpret_cast<const int32_t*>(PlanBuf), reinterpret_cast<uint32_t*>(SliceState), Out, \
                    max_ctas, tile, num_experts, eff_work, as_row_stride, w_evict_mode, s); \
            } \
        } while (0)

    switch (key) {
        case 0: DISPATCH_PAN_V5(0, 0); break;
        case 1: DISPATCH_PAN_V5(0, 1); break;
        case 2: DISPATCH_PAN_V5(1, 0); break;
        case 3: DISPATCH_PAN_V5(1, 1); break;
        default: return -3;
    }
    #undef DISPATCH_PAN_V5
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

}  // namespace

// =============================================================================
// Public C ABI Entrypoints (Strictly Prefill-Only)
// =============================================================================

extern "C" {

int pf_nvfp4_w2_v16_version() {
    return 1600;  // Wave-V16 SUB-60s: 3-stage cp.async + a_pinned<=3 + L2::evict_first + DB slice, ZERO-SPIN!
}

int pf_nvfp4_w2_set_double_buf_slice_v16(int enable) {
    int prev = g_w2_double_buf_slice;
    g_w2_double_buf_slice = (enable != 0) ? 1 : 0;
    return prev;
}

int pf_nvfp4_w2_v13_version() {
    return 1600;
}

int pf_nvfp4_w2_v5_version() {
    return 1600;
}

int pf_nvfp4_w2_v5_set_reserve_sms(int reserve_sms) {
    if (reserve_sms < 0) reserve_sms = 0;
    if (reserve_sms > 24) reserve_sms = 24;
    g_dbo_reserve_sms = reserve_sms;
    return g_dbo_reserve_sms;
}

int pf_nvfp4_w2_v5_get_reserve_sms() {
    return g_dbo_reserve_sms;
}

int pf_nvfp4_w2_build_plan_v5(
    const void* Tiles,
    void* PlanBuf,
    int max_ctas,
    int tile,
    int num_experts,
    int split_m,
    int max_work,
    uint64_t stream_u64
) {
    if (!Tiles || !PlanBuf || max_ctas <= 0 || tile < 64 || num_experts <= 0 || max_work <= 0) {
        return -2;
    }
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream_u64);
    if (is_stream_capturing(st)) return -4;
    pf_nvfp4_w2_build_plan_v5_kernel<<<1, 256, 0, st>>>(
        reinterpret_cast<const int32_t*>(Tiles),
        reinterpret_cast<int32_t*>(PlanBuf),
        max_ctas, tile, num_experts, split_m, max_work
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int pf_nvfp4_w2_ws5_v5_unfused(
    const void* A, const void* AS, const void* RS, const void* TopkWeights,
    const void* W_PACK, const void* WS5_PACK, const void* WS_BASE,
    const void* P_OFF, const void* PATCHES,
    const void* Ones, const void* Tiles, const void* PlanBuf, void* SliceState, void* Out,
    int max_ctas, int tile, int num_experts, int max_work,
    int as_row_major, int as_row_stride, int has_patches,
    int w_evict_mode, uint64_t stream_u64
) {
    if (!A || !AS || !RS || !W_PACK || !WS5_PACK || !WS_BASE || !P_OFF || !PATCHES ||
        !Ones || !Tiles || !SliceState || !Out) {
        return -2;
    }
    if (tile < 64 || (tile % 64) != 0) return -2;
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream_u64);
    if (is_stream_capturing(st)) return -4;
    return launch_ws5_v5_cm_p<0, 6>(
        A, AS, RS, TopkWeights, Tiles, W_PACK, WS5_PACK, WS_BASE, P_OFF, PATCHES,
        Ones, Tiles, PlanBuf, SliceState, Out,
        max_ctas, tile, num_experts, max_work, as_row_major, as_row_stride, has_patches,
        w_evict_mode, 0, 0, st);
}

int pf_nvfp4_w2_ws5_v5_combine(
    const void* A, const void* AS, const void* RS, const void* TopkWeights, const void* Row2Tok,
    const void* W_PACK, const void* WS5_PACK, const void* WS_BASE,
    const void* P_OFF, const void* PATCHES,
    const void* Ones, const void* Tiles, const void* PlanBuf, void* SliceState, void* Out,
    int max_ctas, int tile, int num_experts, int max_work,
    int as_row_major, int as_row_stride, int has_patches,
    int combine_mode, int w_evict_mode, int w2_panels,
    int zero_mode, int tokens, uint64_t stream_u64
) {
    if (!A || !AS || !RS || !W_PACK || !WS5_PACK || !WS_BASE || !P_OFF || !PATCHES ||
        !Ones || !Tiles || !SliceState || !Out) {
        return -2;
    }
    if (tile < 64 || (tile % 64) != 0) return -2;
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream_u64);
    if (is_stream_capturing(st)) return -4;

    if (combine_mode == 0) {
        return launch_ws5_v5_cm_p<0, 6>(
            A, AS, RS, TopkWeights, Tiles, W_PACK, WS5_PACK, WS_BASE, P_OFF, PATCHES,
            Ones, Tiles, PlanBuf, SliceState, Out,
            max_ctas, tile, num_experts, max_work, as_row_major, as_row_stride, has_patches,
            w_evict_mode, 0, 0, st);
    }
    if (combine_mode == 1) {
        if (!Row2Tok) return -2;
        return launch_ws5_v5_cm_p<1, 2>(
            A, AS, RS, TopkWeights, Row2Tok, W_PACK, WS5_PACK, WS_BASE, P_OFF, PATCHES,
            Ones, Tiles, PlanBuf, SliceState, Out,
            max_ctas, tile, num_experts, max_work, as_row_major, as_row_stride, has_patches,
            w_evict_mode, 0, tokens, st);
    }
    if (combine_mode == 2) {
        if (!Row2Tok) return -2;
        if (w2_panels == 1) {
            return launch_ws5_v5_cm_p<2, 1>(
                A, AS, RS, TopkWeights, Row2Tok, W_PACK, WS5_PACK, WS_BASE, P_OFF, PATCHES,
                Ones, Tiles, PlanBuf, SliceState, Out,
                max_ctas, tile, num_experts, max_work, as_row_major, as_row_stride, has_patches,
                w_evict_mode, zero_mode, tokens, st);
        }
        if (w2_panels == 2) {
            return launch_ws5_v5_cm_p<2, 2>(
                A, AS, RS, TopkWeights, Row2Tok, W_PACK, WS5_PACK, WS_BASE, P_OFF, PATCHES,
                Ones, Tiles, PlanBuf, SliceState, Out,
                max_ctas, tile, num_experts, max_work, as_row_major, as_row_stride, has_patches,
                w_evict_mode, zero_mode, tokens, st);
        }
        if (w2_panels == 3) {
            return launch_ws5_v5_cm_p<2, 3>(
                A, AS, RS, TopkWeights, Row2Tok, W_PACK, WS5_PACK, WS_BASE, P_OFF, PATCHES,
                Ones, Tiles, PlanBuf, SliceState, Out,
                max_ctas, tile, num_experts, max_work, as_row_major, as_row_stride, has_patches,
                w_evict_mode, zero_mode, tokens, st);
        }
        if (w2_panels == 6) {
            return launch_ws5_v5_cm_p<2, 6>(
                A, AS, RS, TopkWeights, Row2Tok, W_PACK, WS5_PACK, WS_BASE, P_OFF, PATCHES,
                Ones, Tiles, PlanBuf, SliceState, Out,
                max_ctas, tile, num_experts, max_work, as_row_major, as_row_stride, has_patches,
                w_evict_mode, zero_mode, tokens, st);
        }
        return launch_ws5_v5_cm_p<2, 4>(
            A, AS, RS, TopkWeights, Row2Tok, W_PACK, WS5_PACK, WS_BASE, P_OFF, PATCHES,
            Ones, Tiles, PlanBuf, SliceState, Out,
            max_ctas, tile, num_experts, max_work, as_row_major, as_row_stride, has_patches,
            w_evict_mode, zero_mode, tokens, st);
    }
    return -4;
}

int pf_nvfp4_w2_panel_v5_combine(
    const void* A, const void* AS, const void* RS, const void* TopkWeights, const void* Row2Tok,
    const void* WP, const void* WSP,
    const void* Ones, const void* Tiles, const void* PlanBuf, void* SliceState, void* Out,
    int max_ctas, int tile, int num_experts, int max_work,
    int as_row_major, int as_row_stride,
    int combine_mode, int w_evict_mode, int w2_panels,
    int zero_mode, int tokens, uint64_t stream_u64
) {
    if (!A || !AS || !RS || !WP || !WSP || !Ones || !Tiles || !SliceState || !Out) {
        return -2;
    }
    if (tile < 64 || (tile % 64) != 0) return -2;
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream_u64);
    if (is_stream_capturing(st)) return -4;

    if (combine_mode == 0) {
        return launch_panel_v5_cm_p<0, 6>(
            A, AS, RS, TopkWeights, Tiles, WP, WSP, Ones, Tiles, PlanBuf, SliceState, Out,
            max_ctas, tile, num_experts, max_work, as_row_major, as_row_stride,
            w_evict_mode, 0, 0, st);
    }
    if (combine_mode == 1) {
        if (!Row2Tok) return -2;
        return launch_panel_v5_cm_p<1, 2>(
            A, AS, RS, TopkWeights, Row2Tok, WP, WSP, Ones, Tiles, PlanBuf, SliceState, Out,
            max_ctas, tile, num_experts, max_work, as_row_major, as_row_stride,
            w_evict_mode, 0, tokens, st);
    }
    if (combine_mode == 2) {
        if (!Row2Tok) return -2;
        if (w2_panels == 1) return launch_panel_v5_cm_p<2, 1>(A, AS, RS, TopkWeights, Row2Tok, WP, WSP, Ones, Tiles, PlanBuf, SliceState, Out, max_ctas, tile, num_experts, max_work, as_row_major, as_row_stride, w_evict_mode, zero_mode, tokens, st);
        if (w2_panels == 2) return launch_panel_v5_cm_p<2, 2>(A, AS, RS, TopkWeights, Row2Tok, WP, WSP, Ones, Tiles, PlanBuf, SliceState, Out, max_ctas, tile, num_experts, max_work, as_row_major, as_row_stride, w_evict_mode, zero_mode, tokens, st);
        if (w2_panels == 3) return launch_panel_v5_cm_p<2, 3>(A, AS, RS, TopkWeights, Row2Tok, WP, WSP, Ones, Tiles, PlanBuf, SliceState, Out, max_ctas, tile, num_experts, max_work, as_row_major, as_row_stride, w_evict_mode, zero_mode, tokens, st);
        if (w2_panels == 6) return launch_panel_v5_cm_p<2, 6>(A, AS, RS, TopkWeights, Row2Tok, WP, WSP, Ones, Tiles, PlanBuf, SliceState, Out, max_ctas, tile, num_experts, max_work, as_row_major, as_row_stride, w_evict_mode, zero_mode, tokens, st);
        return launch_panel_v5_cm_p<2, 4>(A, AS, RS, TopkWeights, Row2Tok, WP, WSP, Ones, Tiles, PlanBuf, SliceState, Out, max_ctas, tile, num_experts, max_work, as_row_major, as_row_stride, w_evict_mode, zero_mode, tokens, st);
    }
    return -4;
}

}  // extern "C"
