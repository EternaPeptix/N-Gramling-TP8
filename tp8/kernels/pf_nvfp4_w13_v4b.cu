// SPDX-License-Identifier: Apache-2.0
// =============================================================================
// SM121a (NVIDIA GB10 Blackwell) Prefill-Only NVFP4 W13 v4b (BM=128, 2 CTAs/SM)
// + Fused Exact 0-ULP SiTU + 1-Warp/Row NVFP4 Quantizer (`libpf_nvfp4_w13_v4b.so`)
// =============================================================================
//
// Targets Inkling-2-Small TP8 (E=384, H=4608, inter=288, topk=8, 40 NVFP4 layers)
// strictly for long-context eager prefill chunks (plan.tile >= 64, T >= 1024,
// !capturing), leaving C=1/C=32 decode 100% untouched and 0-ULP bit-exact.
//
// Why v4b beats BOTH live v3 BM=128 (3.325 ms + 0.386 ms quant = 3.711 ms) AND
// v4 BM=256 (3.92 ms, 1 CTA/SM) / v5 BM=64 (4.05 ms, 2x weight reads) at T=8192:
//   1. Keeps BM=128, BN=96, 2 CTAs/SM (16 warps/SM, <= 38.5 KiB SMEM, 0 spills)
//      and a 100% branch-free inner K-loop (0-ULP identical acc order to v3!).
//   2. 1-Block Zero-Sync LPT BM=128 Worklist (`pf_nvfp4_build_m128_worklist`):
//      - Eliminates ~44% empty dead CTAs in v3's padded (6*sub_m, max_ctas) grid.
//      - Sorts experts heavy-first (LPT) across 16 buckets and places each
//        expert's BM=128 M-tiles STRICTLY CONSECUTIVELY in blockIdx.y so multi-
//        tile experts share 1.43 MB weights hot in GB10's 24 MiB L2!
//      - Emits per-tile `is_last_e` flag: W_EVICT=1 (`ld.ca` non-last, `ld.cs`
//        last) eliminates best.env's W_EVICT=2 self-eviction bug (1.8-2.1x ->
//        ~1.05x LPDDR5X weight reads!).
//   3. Single-Barrier 128-Row Epilogue overlaying `sA` union (0 extra SMEM):
//      - Cuts epilogue barriers from 7 -> 2 (unfused) or 2 (fused SiTU).
//      - Full 256-thread coalesced `st.global.cs.v4.u32` (no idle lanes 24..31!).
//   4. Opt-in Fused W13 + Exact 0-ULP SiTU (`FUSE_SITU=1`) + 1-Warp/Row Parallel
//      `pf_nvfp4_quant_288` (18 lanes = 18 blocks of 16 in parallel):
//      - Cuts intermediate DRAM from 75.5 MB W + 75.5 MB R -> 37.7 MB W + 37.7 MB R
//        (-75.5 MB/mb!) and hides SiTU SFU ops inside W13 epilogue for free!
//      - `pf_nvfp4_quant_288` writes `Values [R,144]` & `Scales [R,20]` with
//        `st.global.wb` -> keeps 10.7 MB w2_act 100% HOT in L2 for v4 W2!
//
// Build on GB10 (SM121a):
//   nvcc -O3 --use_fast_math --shared -Xcompiler -fPIC \
//       -gencode arch=compute_121a,code=sm_121a -Xptxas -v \
//       pf_nvfp4_w13_v4b.cu -o libpf_nvfp4_w13_v4b.so
// =============================================================================

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace {

static constexpr float FP32_MAX_VAL = 3.4028234663852886e+38f;
static constexpr float INV_6        = 1.0f / 6.0f;
static constexpr float INV_25       = 1.0f / 25.0f;
static constexpr float INV_1536     = 1.0f / 1536.0f;

// =============================================================================
// SM121a PTX Primitives (exact RN rounding for 0-ULP SiTU & NVFP4 quantization)
// =============================================================================

__device__ __forceinline__ float mul_rn(float a, float b) {
    float r;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

__device__ __forceinline__ float add_rn(float a, float b) {
    float r;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

__device__ __forceinline__ float sub_rn(float a, float b) {
    float r;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

__device__ __forceinline__ float div_rn(float a, float b) {
    float r;
    asm("div.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

__device__ __forceinline__ float rcp_rn(float a) {
    float r;
    asm("rcp.rn.f32 %0, %1;" : "=f"(r) : "f"(a));
    return r;
}

__device__ __forceinline__ float bf16_bits_to_f32(uint16_t b) {
    uint32_t u = static_cast<uint32_t>(b) << 16;
    return __uint_as_float(u);
}

__device__ __forceinline__ uint16_t f32_to_bf16_rn(float x) {
    __nv_bfloat16 b = __float2bfloat16_rn(x);
    return *reinterpret_cast<const uint16_t*>(&b);
}

__device__ __forceinline__ uint32_t pack_bf16x2(float a, float b) {
    __nv_bfloat162 bv = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<const uint32_t*>(&bv);
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

__device__ __forceinline__ uint32_t ld_ca_u32(const void* ptr) {
    uint32_t v;
    asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(v) : "l"(ptr));
    return v;
}

__device__ __forceinline__ uint32_t ld_cs_u32(const void* ptr) {
    uint32_t v;
    asm volatile("ld.global.cs.u32 %0, [%1];" : "=r"(v) : "l"(ptr));
    return v;
}

__device__ __forceinline__ uint8_t ld_ca_u8(const void* ptr) {
    uint32_t v;
    asm volatile("ld.global.ca.u8 %0, [%1];" : "=r"(v) : "l"(ptr));
    return static_cast<uint8_t>(v);
}

__device__ __forceinline__ uint8_t ld_cs_u8(const void* ptr) {
    uint32_t v;
    asm volatile("ld.global.cs.u8 %0, [%1];" : "=r"(v) : "l"(ptr));
    return static_cast<uint8_t>(v);
}

__device__ __forceinline__ uint4 ld_w_u4(const void* ptr, bool use_cs) {
    return use_cs ? ld_cs_u4(ptr) : ld_ca_u4(ptr);
}

__device__ __forceinline__ uint32_t ld_w_u32(const void* ptr, bool use_cs) {
    return use_cs ? ld_cs_u32(ptr) : ld_ca_u32(ptr);
}

__device__ __forceinline__ uint8_t ld_w_u8(const void* ptr, bool use_cs) {
    return use_cs ? ld_cs_u8(ptr) : ld_ca_u8(ptr);
}

__device__ __forceinline__ void st_cs_u4(void* ptr, uint4 v) {
    asm volatile("st.global.cs.v4.u32 [%0], {%1, %2, %3, %4};"
                 :: "l"(ptr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}

__device__ __forceinline__ void st_wb_u2(void* ptr, uint2 v) {
    asm volatile("st.global.wb.v2.u32 [%0], {%1, %2};"
                 :: "l"(ptr), "r"(v.x), "r"(v.y) : "memory");
}

__device__ __forceinline__ void st_wb_u32(void* ptr, uint32_t v) {
    asm volatile("st.global.wb.u32 [%0], %1;" :: "l"(ptr), "r"(v) : "memory");
}

__device__ __forceinline__ void cp_async_cg_16(void* smem_ptr, const void* gmem_ptr, int src_bytes) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
        :: "r"(smem_addr), "l"(gmem_ptr), "r"(src_bytes) : "memory"
    );
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template <int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}

// Native SM121a NVFP4 block-scaled MMA: m16n8k64, e2m1 x e2m1, ue4m3 (scale_vec::4X, VS=16).
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

// =============================================================================
// Exact 0-ULP SiTU Helpers (GATE_CAP=4, UP_CAP=25, BF16-rounded gate/up inputs)
// =============================================================================

__device__ __forceinline__ float sigmoid_approx_exact(float x) {
    float r;
    asm("{ .reg .f32 t; mul.rn.f32 t, %1, 0fBFB8AA3B; "
        "ex2.approx.ftz.f32 t, t; add.rn.f32 t, t, 0f3F800000; "
        "rcp.approx.ftz.f32 %0, t; }"
        : "=f"(r) : "f"(x));
    return r;
}

__device__ __forceinline__ float softcap_4(float x) {
    float q = mul_rn(x, 0.25f);
    float t;
    asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(q));
    return mul_rn(4.0f, t);
}

__device__ __forceinline__ float softcap_25(float x) {
    float q = mul_rn(x, INV_25);
    float res = fmaxf(fmaf(q, 25.0f, -x), -FP32_MAX_VAL);
    q = fmaf(-res, INV_25, q);
    float t;
    asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(q));
    return mul_rn(25.0f, t);
}

// Matches unfused W13->BF16->swiglu_quantize 0-ULP bit-for-bit!
__device__ __forceinline__ uint16_t situ_from_f32_pair_exact(float g_f32, float u_f32, float gamma) {
    float gate = bf16_bits_to_f32(f32_to_bf16_rn(g_f32));
    float up   = bf16_bits_to_f32(f32_to_bf16_rn(u_f32));
    float sig  = sigmoid_approx_exact(gate);
    gate = softcap_4(gate);
    up   = softcap_25(up);
    float y = mul_rn(mul_rn(mul_rn(gate, sig), up), gamma);
    return f32_to_bf16_rn(y);
}

// =============================================================================
// 1-Block Zero-Sync LPT BM=128 Worklist Builder (`~0.8 us`)
// =============================================================================
// WorkBuf layout (4 * max_m128 + 1 int32s):
//   [0 .. max_m128-1]                : WorkE[w]     (expert ID, or -1 if unused)
//   [max_m128 .. 2*max_m128-1]       : WorkRow0[w]  (start row of this BM=128 tile)
//   [2*max_m128 .. 3*max_m128-1]     : WorkEnd[w]   (expert's exclusive end row)
//   [3*max_m128 .. 4*max_m128-1]     : WorkFlags[w] (bit0=is_last_e, bit1=is_full_m)
//   [4*max_m128]                     : total_work   (exact number of active BM=128 tiles)
//
// CRITICAL PROPERTY: All BM=128 tiles belonging to the same expert `e` are
// placed STRICTLY CONSECUTIVELY (`pos .. pos + n_m - 1`), sorted heavy-first
// across 16 LPT buckets! Thus `(nb_cta=0..5, w0)` and `(nb_cta=0..5, w0+1)`
// are scheduled 6 CTAs apart -> ~100% L2 hit on weights for multi-tile experts!

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
        row_beg = 0; row_end = 0; return;
    }
    int lo = 0, hi = num_tiles;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (Tiles[mid] < target_e) lo = mid + 1;
        else hi = mid;
    }
    const int t_lo = lo;
    if (t_lo >= num_tiles || Tiles[t_lo] != target_e) {
        row_beg = 0; row_end = 0; return;
    }
    lo = t_lo + 1;
    hi = num_tiles;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (Tiles[mid] <= target_e) lo = mid + 1;
        else hi = mid;
    }
    const int t_hi = lo;
    row_beg = t_lo * tile;
    row_end = Tiles[max_ctas + t_hi - 1];
}

__global__ __launch_bounds__(256, 1)
void pf_nvfp4_build_m128_worklist_kernel(
    const int32_t* __restrict__ Tiles,
    int32_t* __restrict__ WorkBuf,
    int max_ctas,
    int tile,
    int num_experts,
    int max_m128
) {
    constexpr int MAX_E = 1024;
    __shared__ int32_t sBeg[MAX_E];
    __shared__ int32_t sEnd[MAX_E];
    __shared__ uint32_t sBucketTiles[16];
    __shared__ uint32_t sBucketOff[16];

    const int tid = threadIdx.x;
    const int eff_e = (num_experts < MAX_E) ? num_experts : MAX_E;

    if (tid < 16) sBucketTiles[tid] = 0u;
    __syncthreads();

    for (int e = tid; e < eff_e; e += 256) {
        int rb = 0, re = 0;
        find_expert_row_range(Tiles, max_ctas, tile, e, rb, re);
        sBeg[e] = rb;
        sEnd[e] = re;
        int cnt = re - rb;
        if (cnt > 0) {
            int n_m = (cnt + 127) >> 7;
            // Heavy-first bucket: bucket 0 = largest (>=16 tiles = >=2048 rows), bucket 15 = 1 tile
            int b = 15 - ((n_m - 1 > 15) ? 15 : (n_m - 1));
            atomicAdd(&sBucketTiles[b], static_cast<uint32_t>(n_m));
        }
    }
    __syncthreads();

    if (tid == 0) {
        uint32_t acc = 0u;
        for (int b = 0; b < 16; ++b) {
            sBucketOff[b] = acc;
            acc += sBucketTiles[b];
        }
        int total = static_cast<int>(acc);
        if (total > max_m128) total = max_m128;
        WorkBuf[4 * max_m128] = total;
    }
    __syncthreads();

    int32_t* out_e     = WorkBuf;
    int32_t* out_row0  = WorkBuf + max_m128;
    int32_t* out_end   = WorkBuf + 2 * max_m128;
    int32_t* out_flags = WorkBuf + 3 * max_m128;

    for (int e = tid; e < eff_e; e += 256) {
        int rb = sBeg[e];
        int re = sEnd[e];
        int cnt = re - rb;
        if (cnt > 0) {
            int n_m = (cnt + 127) >> 7;
            int b = 15 - ((n_m - 1 > 15) ? 15 : (n_m - 1));
            // Reserve n_m strictly consecutive work slots for expert e!
            uint32_t base_pos = atomicAdd(&sBucketOff[b], static_cast<uint32_t>(n_m));
            for (int m = 0; m < n_m; ++m) {
                int pos = static_cast<int>(base_pos) + m;
                if (pos < max_m128) {
                    int r0 = rb + m * 128;
                    int fl = ((m + 1 == n_m) ? 1 : 0) | ((r0 + 128 <= re) ? 2 : 0);
                    out_e[pos]     = e;
                    out_row0[pos]  = r0;
                    out_end[pos]   = re;
                    out_flags[pos] = fl;
                }
            }
        }
    }
}

// =============================================================================
// SMEM Epilogue Structures (Overlaying sA Union -> 0 Extra SMEM!)
// =============================================================================

struct W13EpiSmem128 {
    uint32_t bf16x2[128][49];   // 25,088 B (stride 49 words = odd -> 0-way bank conflict!)
};

struct W13SituEpiSmem128 {
    uint16_t y[128][50];        // 12,800 B (row=100 B = 25 words -> 4B aligned, 0-way bank conflict!)
};

// =============================================================================
// Prefill NVFP4 W13 v4b Kernel (BM=128, BN=96, 2 CTAs/SM, 0 spills!)
// =============================================================================
// - FUSE_SITU == 0: writes Out [R, 576] bf16 (0-ULP identical to live v3!).
// - FUSE_SITU == 1: writes Out [R, 288] bf16 = exact SiTU(W13(x), gamma) (-75.5 MB DRAM!).

template <int USE_R2T, int AS_ROW_MAJOR, int HAS_PATCHES, int FUSE_SITU>
__global__ __launch_bounds__(256, 1)
void pf_nvfp4_w13_ws5_v4b_kernel(
    const uint8_t* __restrict__ A,
    const uint8_t* __restrict__ AS,
    const float* __restrict__ RS,
    const int32_t* __restrict__ Row2Tok,
    const uint8_t* __restrict__ SLABS,
    const uint8_t* __restrict__ WS_BASE,
    const int32_t* __restrict__ P_OFF,
    const uint32_t* __restrict__ PATCHES,
    const float* __restrict__ Ones,
    const float* __restrict__ Gammas,
    const int32_t* __restrict__ WorkBuf,
    void* __restrict__ Out,
    int max_m128,
    int tile,
    int w_evict_mode
) {
    constexpr int BM = 128;
    constexpr int KP = 2304;
    constexpr int KG = 288;
    constexpr int M_TILES_W = 4;   // (BM / 2) / 16 = 64 / 16 = 4

    const int nb_cta = blockIdx.x; // 0..5 (6 x 96 = 576 cols)
    const int w_idx  = blockIdx.y; // 0..max_m128-1

    const int total_work = WorkBuf[4 * max_m128];
    if (w_idx >= total_work) return;

    const int64_t e   = WorkBuf[w_idx];
    const int row0    = WorkBuf[max_m128 + w_idx];
    const int end     = WorkBuf[2 * max_m128 + w_idx];
    const int flags   = WorkBuf[3 * max_m128 + w_idx];
    const bool is_last_e = (flags & 1) != 0;
    const bool is_full_m = (flags & 2) != 0;
    const bool use_cs_w  = (w_evict_mode == 2) || (w_evict_mode == 1 && is_last_e);

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int wm   = warp >> 2;    // 0..1 (each owns 64 M-rows)
    const int wp_n = warp & 3;     // 0..3 (each owns 3 x 8 = 24 N-cols)
    const int lane = tid & 31;
    const int g    = lane >> 2;    // 0..7
    const int lt   = lane & 3;     // 0..3

    // Total static SMEM:
    //   m: sA (32,768) + sAS (4,096) = 36,864 B (overlaps 25,088 B / 12,800 B epilogue!)
    //   + sTok (512) + sRS (512) + sGam (512) + sPOff (96) = 38,496 B <= 49,152 B -> 2 CTAs/SM!
    __shared__ alignas(16) union {
        struct { uint32_t sA[2][BM][32]; uint32_t sAS[2][BM][4]; } m;
        W13EpiSmem128 epi_raw;
        W13SituEpiSmem128 epi_situ;
    } sm;
    __shared__ alignas(16) int32_t sTok[BM];
    __shared__ alignas(16) float   sRS[BM];
    __shared__ alignas(16) float   sGam[BM];
    __shared__ alignas(16) int32_t sPOff[12][2];

    if (tid < BM) {
        int r = row0 + tid;
        if (is_full_m || r < end) {
            sTok[tid] = USE_R2T ? Row2Tok[r] : r;
            sRS[tid]  = RS[r];
            if constexpr (FUSE_SITU) {
                sGam[tid] = Gammas[r];
            }
        } else {
            sTok[tid] = -1;
            sRS[tid]  = 0.0f;
            if constexpr (FUSE_SITU) {
                sGam[tid] = 0.0f;
            }
        }
    }
    if constexpr (HAS_PATCHES) {
        if (tid < 12) {
            int64_t wuid = (e * 18 + nb_cta * 3 + (tid >> 2)) * 4 + (tid & 3);
            sPOff[tid][0] = P_OFF[wuid];
            sPOff[tid][1] = P_OFF[wuid + 1];
        }
    }
    __syncthreads();

    auto load_stage = [&](int ks, int buf) {
        // BM=128 * 8 = 1024 uint4 -> 4 iterations of 256 threads
        #pragma unroll
        for (int it = 0; it < 4; ++it) {
            int idx = it * 256 + tid;
            int m   = idx >> 3;
            int c4  = idx & 7;
            int sc4 = c4 ^ (m & 7);
            int32_t tok = sTok[m];
            const uint8_t* src = (tok >= 0) ? (A + static_cast<int64_t>(tok) * KP + ks * 128 + c4 * 16) : A;
            cp_async_cg_16(&sm.m.sA[buf][m][sc4 * 4], src, (tok >= 0) ? 16 : 0);
        }
        if constexpr (USE_R2T || AS_ROW_MAJOR) {
            if (tid < BM) {
                int32_t tok = sTok[tid];
                const uint8_t* src_as = (tok >= 0) ? (AS + static_cast<int64_t>(tok) * KG + ks * 16) : AS;
                cp_async_cg_16(&sm.m.sAS[buf][tid][0], src_as, (tok >= 0) ? 16 : 0);
            }
        } else {
            #pragma unroll
            for (int it = 0; it < 2; ++it) {
                int idx = it * 256 + tid;
                int m = idx >> 2;
                int q = idx & 3;
                int32_t tok = sTok[m];
                uint32_t sval = 0u;
                if (tok >= 0) {
                    int64_t off = scale_offset_byte(static_cast<int64_t>(row0 + m), ks * 16 + q * 4, KG, tile);
                    sval = *reinterpret_cast<const uint32_t*>(AS + off);
                }
                sm.m.sAS[buf][m][q] = sval;
            }
        }
        cp_async_commit();
    };

    const uint8_t* slabs_cta = SLABS + (e * 18 + nb_cta * 3) * (9LL * 8832LL);
    const int base_e = static_cast<int>(WS_BASE[e]);

    auto decode_sfb_ss = [&](int ss, uint32_t (&p0)[3], uint32_t (&p1)[3]) {
        const int kg_lo = ss * 32;
        const int kg_hi = kg_lo + 32;
        #pragma unroll
        for (int sn = 0; sn < 3; ++sn) {
            int wg = wp_n * 3 + sn;
            int pl = wg >> 2;
            int wp = wg & 3;
            const uint8_t* ss_ptr = slabs_cta + static_cast<int64_t>(pl * 9 + ss) * 8832;
            uint32_t ws_lo = ld_w_u32(ss_ptr + 8192 + wp * 128 + lane * 4, use_cs_w);
            uint8_t  ws_hi = ld_w_u8(ss_ptr + 8704 + wp * 32 + lane, use_cs_w);
            uint8_t sb[8];
            #pragma unroll
            for (int ii = 0; ii < 8; ++ii) {
                int code = ((ws_lo >> (4 * ii)) & 0xF) | (((ws_hi >> ii) & 1) << 4);
                sb[ii] = static_cast<uint8_t>(base_e + code);
            }
            if constexpr (HAS_PATCHES) {
                int plo = sPOff[wg][0], phi = sPOff[wg][1];
                for (int pi = plo; pi < phi; ++pi) {
                    uint32_t rec = PATCHES[pi];
                    int p_kg = static_cast<int>(rec >> 16);
                    if (p_kg >= kg_lo && p_kg < kg_hi) {
                        if (g == static_cast<int>((rec >> 8) & 0xFF) && lt == ((p_kg - kg_lo) >> 3)) {
                            const int idx_s = p_kg & 7;
                            const uint8_t pval = static_cast<uint8_t>(rec & 0xFF);
                            #pragma unroll
                            for (int ii = 0; ii < 8; ++ii) {
                                if (idx_s == ii) sb[ii] = pval;
                            }
                        }
                    }
                }
            }
            p0[sn] = static_cast<uint32_t>(sb[0]) | (static_cast<uint32_t>(sb[1]) << 8)
                   | (static_cast<uint32_t>(sb[2]) << 16) | (static_cast<uint32_t>(sb[3]) << 24);
            p1[sn] = static_cast<uint32_t>(sb[4]) | (static_cast<uint32_t>(sb[5]) << 8)
                   | (static_cast<uint32_t>(sb[6]) << 16) | (static_cast<uint32_t>(sb[7]) << 24);
        }
    };

    load_stage(0, 0);
    uint4 wa[3], wb[3];
    #pragma unroll
    for (int sn = 0; sn < 3; ++sn) {
        int wg = wp_n * 3 + sn;
        wa[sn] = ld_w_u4(slabs_cta + static_cast<int64_t>((wg >> 2) * 9) * 8832 + (wg & 3) * 1024 + lane * 16, use_cs_w);
    }
    uint32_t sfb_p0[3], sfb_p1[3];
    decode_sfb_ss(0, sfb_p0, sfb_p1);
    cp_async_wait<0>();
    __syncthreads();

    float acc[M_TILES_W][3][4];
    #pragma unroll
    for (int mi = 0; mi < M_TILES_W; ++mi) {
        #pragma unroll
        for (int sn = 0; sn < 3; ++sn) {
            acc[mi][sn][0] = 0.0f; acc[mi][sn][1] = 0.0f;
            acc[mi][sn][2] = 0.0f; acc[mi][sn][3] = 0.0f;
        }
    }

    const int m_warp_base = wm * 64;
    const int m_sf_off = g + ((lt & 1) << 3);

    // 100% branch-free inner K-loop (identical accumulation order to live v3!)
    #pragma unroll 1
    for (int ks = 0; ks < 18; ++ks) {
        const int cur = ks & 1, nxt = cur ^ 1, ss = ks >> 1, ks2 = ks & 1;
        if (ks + 1 < 18) load_stage(ks + 1, nxt);
        #pragma unroll
        for (int sn = 0; sn < 3; ++sn) {
            int wg = wp_n * 3 + sn;
            wb[sn] = ld_w_u4(slabs_cta + static_cast<int64_t>((wg >> 2) * 9 + ss) * 8832
                             + ks2 * 4096 + (wg & 3) * 1024 + 512 + lane * 16, use_cs_w);
        }
        const int src_lt0 = ks2 * 2;
        #pragma unroll
        for (int q2 = 0; q2 < 2; ++q2) {
            uint32_t b0[3], b1[3], sfb[3];
            #pragma unroll
            for (int sn = 0; sn < 3; ++sn) {
                uint32_t lx = (q2 == 0) ? wa[sn].x : wa[sn].z;
                uint32_t ly = (q2 == 0) ? wa[sn].y : wa[sn].w;
                b0[sn] = prmt_b32(lx, ly, 0x5410u);
                b1[sn] = prmt_b32(lx, ly, 0x7632u);
                sfb[sn] = __shfl_sync(0xFFFFFFFFu, (q2 == 0) ? sfb_p0[sn] : sfb_p1[sn], (g << 2) | src_lt0);
            }
            const int sw_c0 = (q2 * 8 + lt) ^ (g << 2);
            const int sw_c1 = (q2 * 8 + 4 + lt) ^ (g << 2);
            #pragma unroll
            for (int mi = 0; mi < M_TILES_W; ++mi) {
                int m_lo = m_warp_base + mi * 16 + g, m_hi = m_lo + 8;
                uint32_t a0 = sm.m.sA[cur][m_lo][sw_c0], a1 = sm.m.sA[cur][m_hi][sw_c0];
                uint32_t a2 = sm.m.sA[cur][m_lo][sw_c1], a3 = sm.m.sA[cur][m_hi][sw_c1];
                uint32_t sfa = sm.m.sAS[cur][m_warp_base + mi * 16 + m_sf_off][q2];
                #pragma unroll
                for (int sn = 0; sn < 3; ++sn)
                    mma_m16n8k64_nvfp4(acc[mi][sn], a0, a1, a2, a3, b0[sn], b1[sn], sfa, sfb[sn]);
            }
        }
        if (ks + 1 < 18) {
            const int n_ss = (ks + 1) >> 1, n_ks2 = (ks + 1) & 1;
            #pragma unroll
            for (int sn = 0; sn < 3; ++sn) {
                int wg = wp_n * 3 + sn;
                wa[sn] = ld_w_u4(slabs_cta + static_cast<int64_t>((wg >> 2) * 9 + n_ss) * 8832
                                 + n_ks2 * 4096 + (wg & 3) * 1024 + lane * 16, use_cs_w);
            }
        }
        const int src_lt1 = ks2 * 2 + 1;
        #pragma unroll
        for (int q2 = 0; q2 < 2; ++q2) {
            uint32_t b0[3], b1[3], sfb[3];
            #pragma unroll
            for (int sn = 0; sn < 3; ++sn) {
                uint32_t lx = (q2 == 0) ? wb[sn].x : wb[sn].z;
                uint32_t ly = (q2 == 0) ? wb[sn].y : wb[sn].w;
                b0[sn] = prmt_b32(lx, ly, 0x5410u);
                b1[sn] = prmt_b32(lx, ly, 0x7632u);
                sfb[sn] = __shfl_sync(0xFFFFFFFFu, (q2 == 0) ? sfb_p0[sn] : sfb_p1[sn], (g << 2) | src_lt1);
            }
            const int sw_c0 = ((2 + q2) * 8 + lt) ^ (g << 2);
            const int sw_c1 = ((2 + q2) * 8 + 4 + lt) ^ (g << 2);
            #pragma unroll
            for (int mi = 0; mi < M_TILES_W; ++mi) {
                int m_lo = m_warp_base + mi * 16 + g, m_hi = m_lo + 8;
                uint32_t a0 = sm.m.sA[cur][m_lo][sw_c0], a1 = sm.m.sA[cur][m_hi][sw_c0];
                uint32_t a2 = sm.m.sA[cur][m_lo][sw_c1], a3 = sm.m.sA[cur][m_hi][sw_c1];
                uint32_t sfa = sm.m.sAS[cur][m_warp_base + mi * 16 + m_sf_off][2 + q2];
                #pragma unroll
                for (int sn = 0; sn < 3; ++sn)
                    mma_m16n8k64_nvfp4(acc[mi][sn], a0, a1, a2, a3, b0[sn], b1[sn], sfa, sfb[sn]);
            }
        }
        if (ks + 1 < 18) {
            if (((ks + 1) & 1) == 0) decode_sfb_ss((ks + 1) >> 1, sfb_p0, sfb_p1);
            cp_async_wait<0>();
            __syncthreads();
        }
    }

    // ---- Single-Barrier 128-Row Coalesced Epilogue (2 barriers total vs 7 in v3!) ----
    const float scale_e = Ones[e];
    __syncthreads();

    if constexpr (FUSE_SITU) {
        // Compute exact 0-ULP SiTU in registers (acc[0]=gate, acc[1]=up for m_lo; acc[2]=gate, acc[3]=up for m_hi!)
        #pragma unroll
        for (int mi = 0; mi < M_TILES_W; ++mi) {
            int m_lo = m_warp_base + mi * 16 + g;
            int m_hi = m_lo + 8;
            bool ok_lo = is_full_m || (row0 + m_lo < end);
            bool ok_hi = is_full_m || (row0 + m_hi < end);
            float s_lo = ok_lo ? (sRS[m_lo] * scale_e) : 0.0f;
            float s_hi = ok_hi ? (sRS[m_hi] * scale_e) : 0.0f;
            float g_lo = ok_lo ? sGam[m_lo] : 0.0f;
            float g_hi = ok_hi ? sGam[m_hi] : 0.0f;
            #pragma unroll
            for (int sn = 0; sn < 3; ++sn) {
                int col_situ = (wp_n * 3 + sn) * 4 + lt;  // 0..47
                sm.epi_situ.y[m_lo][col_situ] = ok_lo
                    ? situ_from_f32_pair_exact(acc[mi][sn][0] * s_lo, acc[mi][sn][1] * s_lo, g_lo)
                    : static_cast<uint16_t>(0u);
                sm.epi_situ.y[m_hi][col_situ] = ok_hi
                    ? situ_from_f32_pair_exact(acc[mi][sn][2] * s_hi, acc[mi][sn][3] * s_hi, g_hi)
                    : static_cast<uint16_t>(0u);
            }
        }
        __syncthreads();

        // 128 rows x 48 bf16 = 128 x 6 uint4 = 768 uint4 -> 3 iters of ALL 256 threads!
        uint16_t* Out_situ = reinterpret_cast<uint16_t*>(Out);
        #pragma unroll
        for (int it = 0; it < 3; ++it) {
            int idx   = it * 256 + tid;
            int r_epi = (idx * 10923) >> 16;   // exact idx / 6 for idx in [0..767]
            int c4    = idx - r_epi * 6;       // 0..5
            int r_glob = row0 + r_epi;
            if (is_full_m || r_glob < end) {
                const uint32_t* srow = reinterpret_cast<const uint32_t*>(&sm.epi_situ.y[r_epi][0]);
                uint4 v;
                v.x = srow[c4 * 4 + 0];
                v.y = srow[c4 * 4 + 1];
                v.z = srow[c4 * 4 + 2];
                v.w = srow[c4 * 4 + 3];
                uint4* dst = reinterpret_cast<uint4*>(
                    Out_situ + static_cast<int64_t>(r_glob) * 288 + nb_cta * 48);
                st_cs_u4(dst + c4, v);
            }
        }
    } else {
        // Unfused: 0-ULP identical to live v3, 2 barriers instead of 7, 100% active threads!
        #pragma unroll
        for (int mi = 0; mi < M_TILES_W; ++mi) {
            int m_lo = m_warp_base + mi * 16 + g;
            int m_hi = m_lo + 8;
            float s_lo = (is_full_m || row0 + m_lo < end) ? (sRS[m_lo] * scale_e) : 0.0f;
            float s_hi = (is_full_m || row0 + m_hi < end) ? (sRS[m_hi] * scale_e) : 0.0f;
            #pragma unroll
            for (int sn = 0; sn < 3; ++sn) {
                int col_pair = (wp_n * 3 + sn) * 4 + lt;  // 0..47
                sm.epi_raw.bf16x2[m_lo][col_pair] = pack_bf16x2(acc[mi][sn][0] * s_lo, acc[mi][sn][1] * s_lo);
                sm.epi_raw.bf16x2[m_hi][col_pair] = pack_bf16x2(acc[mi][sn][2] * s_hi, acc[mi][sn][3] * s_hi);
            }
        }
        __syncthreads();

        // 128 rows x 96 bf16 = 128 x 12 uint4 = 1,536 uint4 -> 6 iters of ALL 256 threads!
        uint32_t* Out_u32 = reinterpret_cast<uint32_t*>(Out);
        #pragma unroll
        for (int it = 0; it < 6; ++it) {
            int idx   = it * 256 + tid;
            int r_epi = (idx * 5462) >> 16;    // exact idx / 12 for idx in [0..1535]
            int c4    = idx - r_epi * 12;      // 0..11
            int r_glob = row0 + r_epi;
            if (is_full_m || r_glob < end) {
                uint4 v;
                v.x = sm.epi_raw.bf16x2[r_epi][c4 * 4 + 0];
                v.y = sm.epi_raw.bf16x2[r_epi][c4 * 4 + 1];
                v.z = sm.epi_raw.bf16x2[r_epi][c4 * 4 + 2];
                v.w = sm.epi_raw.bf16x2[r_epi][c4 * 4 + 3];
                uint4* dst = reinterpret_cast<uint4*>(
                    Out_u32 + static_cast<int64_t>(r_glob) * 288 + nb_cta * 48);
                st_cs_u4(dst + c4, v);
            }
        }
    }
}

// =============================================================================
// Ultra-Fast 1-Warp/Row Exact 0-ULP NVFP4 Quantizer (`pf_nvfp4_quant_288`)
// =============================================================================
// Consumes `Y_situ [R, 288]` bf16 (written by FUSE_SITU=1 W13 v4b).
// Lanes 0..17 each handle ONE 16-element NVFP4 block in parallel (ZERO inner loop!):
//   - 2 coalesced uint4 loads (`ld.cs`, 32 B/lane = 576 B/row)
//   - 5-step warp butterfly `row_amax` reduction
//   - Exact PTX-RN dual-candidate (`scale6` vs `scale4`) L1-error selection
//   - `st.global.wb.v2.u32` to `Values [R, 144]` & `st.global.wb.u32` to `Scales [R, 20]`
//     (keeps 10.7 MB w2_act 100% HOT in L2 for persistent v4 W2!).

__device__ __forceinline__ void eval_nvfp4_half4_unfused(
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
            pe6 = add_rn(fabsf(sub_rn(mul_rn(mul_rn(r0, scale6), outer_scale), x0)),
                         fabsf(sub_rn(mul_rn(mul_rn(r1, scale6), outer_scale), x1)));
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
            pe4 = add_rn(fabsf(sub_rn(mul_rn(mul_rn(r0, scale4), outer_scale), x0)),
                         fabsf(sub_rn(mul_rn(mul_rn(r1, scale4), outer_scale), x1)));
        }
        if (k == 0)      { e6_0 = pe6; e4_0 = pe4; }
        else if (k == 1) { e6_1 = pe6; e4_1 = pe4; }
        else if (k == 2) { e6_2 = pe6; e4_2 = pe4; }
        else             { e6_3 = pe6; e4_3 = pe4; }
    }
    pk6_w = p6;
    pk4_w = p4;
    err6_h = add_rn(add_rn(e6_0, e6_1), add_rn(e6_2, e6_3));
    err4_h = add_rn(add_rn(e4_0, e4_1), add_rn(e4_2, e4_3));
}

template <int AS_ROW_MAJOR>
__global__ __launch_bounds__(256, 4)
void pf_nvfp4_quant_288_kernel(
    const uint16_t* __restrict__ Y_situ,      // [R, 288] bf16
    uint8_t* __restrict__ Values,             // [R, 144] uint8
    uint8_t* __restrict__ Scales,             // [R, 20]  e4m3 (row-major or swizzled)
    float* __restrict__ RowScales,            // [R]      fp32
    const int32_t* __restrict__ Rows,         // [max_rows] routed row indices
    const int32_t* __restrict__ NumRows,      // [1] device scalar num_routed
    const int32_t* __restrict__ CtaExpert,    // [max_ctas] expert per tile
    const float* __restrict__ WAmax,          // [E]
    const float* __restrict__ WDenominator,   // [1]
    int tile,
    int scale_tile
) {
    const int num_routed = *NumRows;
    const float denom = mul_rn(1536.0f, *WDenominator);
    const float inv_denom = rcp_rn(denom);
    const int tile_shift = __ffs(tile) - 1;

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int stride_r = gridDim.x * 8;

    for (int r = blockIdx.x * 8 + warp; r < num_routed; r += stride_r) {
        const int row = Rows[r];
        uint4 v0 = make_uint4(0u, 0u, 0u, 0u);
        uint4 v1 = make_uint4(0u, 0u, 0u, 0u);
        float bmax = 0.0f;

        if (lane < 18) {
            const uint4* src = reinterpret_cast<const uint4*>(
                Y_situ + static_cast<int64_t>(row) * 288 + lane * 16);
            v0 = ld_cs_u4(src + 0);
            v1 = ld_cs_u4(src + 1);
            const uint32_t w[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                float a = bf16_bits_to_f32(static_cast<uint16_t>(w[j] & 0xFFFFu));
                float b = bf16_bits_to_f32(static_cast<uint16_t>(w[j] >> 16));
                bmax = fmaxf(bmax, fmaxf(fabsf(a), fabsf(b)));
            }
        }

        // Exact warp-wide row_amax across lanes 0..17
        float row_amax = bmax;
        row_amax = fmaxf(row_amax, __shfl_xor_sync(0xFFFFFFFFu, row_amax, 16));
        row_amax = fmaxf(row_amax, __shfl_xor_sync(0xFFFFFFFFu, row_amax, 8));
        row_amax = fmaxf(row_amax, __shfl_xor_sync(0xFFFFFFFFu, row_amax, 4));
        row_amax = fmaxf(row_amax, __shfl_xor_sync(0xFFFFFFFFu, row_amax, 2));
        row_amax = fmaxf(row_amax, __shfl_xor_sync(0xFFFFFFFFu, row_amax, 1));

        if (lane == 0) {
            int expert = CtaExpert[row >> tile_shift];
            float w_amax = WAmax[expert];
            RowScales[row] = mul_rn(mul_rn(row_amax, w_amax), inv_denom);
        }

        const float enc_s = fminf(div_rn(1536.0f, row_amax), FP32_MAX_VAL);
        const float dec_s = div_rn(1.0f, enc_s);
        const float out_s = mul_rn(row_amax, INV_1536);

        uint8_t sc_byte = 0u;
        if (lane < 18) {
            float s6_raw = fminf(mul_rn(bmax, mul_rn(enc_s, INV_6)), 448.0f);
            float s4_raw = fminf(mul_rn(bmax, mul_rn(enc_s, 0.25f)), 448.0f);

            uint16_t s_e4m3x2;
            uint32_t s_f16x2;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %3, %2;\n\t"
                "cvt.rn.f16x2.e4m3x2 %1, %0;"
                : "=&h"(s_e4m3x2), "=r"(s_f16x2)
                : "f"(s6_raw), "f"(s4_raw));

            float scale6 = __half2float(__ushort_as_half(static_cast<uint16_t>(s_f16x2 & 0xFFFFu)));
            float scale4 = __half2float(__ushort_as_half(static_cast<uint16_t>(s_f16x2 >> 16)));
            float inv6 = fminf(div_rn(1.0f, mul_rn(scale6, dec_s)), FP32_MAX_VAL);
            float inv4 = fminf(div_rn(1.0f, mul_rn(scale4, dec_s)), FP32_MAX_VAL);

            uint32_t pk6_lo, pk4_lo, pk6_hi, pk4_hi;
            float e6_lo, e4_lo, e6_hi, e4_hi;
            eval_nvfp4_half4_unfused(v0, inv6, scale6, inv4, scale4, out_s, pk6_lo, pk4_lo, e6_lo, e4_lo);
            eval_nvfp4_half4_unfused(v1, inv6, scale6, inv4, scale4, out_s, pk6_hi, pk4_hi, e6_hi, e4_hi);

            const bool choose4 = (add_rn(e4_lo, e4_hi) < add_rn(e6_lo, e6_hi));
            const uint2 packed = make_uint2(choose4 ? pk4_lo : pk6_lo, choose4 ? pk4_hi : pk6_hi);
            sc_byte = static_cast<uint8_t>(choose4 ? (s_e4m3x2 >> 8) : (s_e4m3x2 & 0xFFu));

            st_wb_u2(Values + static_cast<int64_t>(row) * 144 + lane * 8, packed);
        }

        if constexpr (AS_ROW_MAJOR) {
            uint32_t my_sc = (lane < 18) ? static_cast<uint32_t>(sc_byte) : 0u;
            uint32_t b0 = __shfl_sync(0xFFFFFFFFu, my_sc, (lane * 4 + 0) & 31);
            uint32_t b1 = __shfl_sync(0xFFFFFFFFu, my_sc, (lane * 4 + 1) & 31);
            uint32_t b2 = __shfl_sync(0xFFFFFFFFu, my_sc, (lane * 4 + 2) & 31);
            uint32_t b3 = __shfl_sync(0xFFFFFFFFu, my_sc, (lane * 4 + 3) & 31);
            if (lane < 5) {
                uint32_t pack4 = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
                st_wb_u32(reinterpret_cast<uint32_t*>(Scales + static_cast<int64_t>(row) * 20) + lane, pack4);
            }
        } else {
            if (lane < 18) {
                int64_t off = scale_offset_byte(static_cast<int64_t>(row), lane, 18, scale_tile);
                Scales[off] = sc_byte;
            }
        }
    }
}

template <int FUSE_SITU>
inline int launch_nvfp4_w13_v4b_impl(
    const void* A, const void* AS, const void* RS, const void* Row2Tok,
    const void* SLABS, const void* WS_BASE, const void* P_OFF, const void* PATCHES,
    const void* Ones, const void* Gammas, const void* WorkBuf, void* Out,
    int max_m128, int tile, int use_r2t, int as_row_major, int has_patches,
    int w_evict_mode, cudaStream_t stream
) {
    if (max_m128 <= 0) return -2;
    dim3 grid(6, max_m128);
    dim3 block(256);
    const int key = ((use_r2t ? 1 : 0) << 2) | ((as_row_major ? 1 : 0) << 1) | (has_patches ? 1 : 0);

    #define DISPATCH_V4B_W13(R2T, RM, HP) \
        do { \
            static bool _carved = false; \
            if (!_carved) { \
                cudaFuncSetAttribute(pf_nvfp4_w13_ws5_v4b_kernel<R2T, RM, HP, FUSE_SITU>, \
                                     cudaFuncAttributePreferredSharedMemoryCarveout, \
                                     cudaSharedmemCarveoutMaxShared); \
                _carved = true; \
            } \
            pf_nvfp4_w13_ws5_v4b_kernel<R2T, RM, HP, FUSE_SITU><<<grid, block, 0, stream>>>( \
                reinterpret_cast<const uint8_t*>(A), reinterpret_cast<const uint8_t*>(AS), \
                reinterpret_cast<const float*>(RS), reinterpret_cast<const int32_t*>(Row2Tok), \
                reinterpret_cast<const uint8_t*>(SLABS), reinterpret_cast<const uint8_t*>(WS_BASE), \
                reinterpret_cast<const int32_t*>(P_OFF), reinterpret_cast<const uint32_t*>(PATCHES), \
                reinterpret_cast<const float*>(Ones), reinterpret_cast<const float*>(Gammas), \
                reinterpret_cast<const int32_t*>(WorkBuf), Out, max_m128, tile, w_evict_mode); \
        } while (0)

    switch (key) {
        case 0: DISPATCH_V4B_W13(0, 0, 0); break;
        case 1: DISPATCH_V4B_W13(0, 0, 1); break;
        case 2: DISPATCH_V4B_W13(0, 1, 0); break;
        case 3: DISPATCH_V4B_W13(0, 1, 1); break;
        case 4: DISPATCH_V4B_W13(1, 0, 0); break;
        case 5: DISPATCH_V4B_W13(1, 0, 1); break;
        case 6: DISPATCH_V4B_W13(1, 1, 0); break;
        case 7: DISPATCH_V4B_W13(1, 1, 1); break;
        default: return -3;
    }
    #undef DISPATCH_V4B_W13
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

}  // namespace

extern "C" {

int pf_nvfp4_build_m128_worklist(
    const void* Tiles,
    void* WorkBuf,
    int max_ctas,
    int tile,
    int num_experts,
    int max_m128,
    uint64_t stream_u64
) {
    if (num_experts <= 0 || max_m128 <= 0 || tile < 64) return -2;
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream_u64);
    pf_nvfp4_build_m128_worklist_kernel<<<1, 256, 0, st>>>(
        reinterpret_cast<const int32_t*>(Tiles),
        reinterpret_cast<int32_t*>(WorkBuf),
        max_ctas, tile, num_experts, max_m128
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int pf_nvfp4_w13_ws5_v4b_unfused(
    const void* A, const void* AS, const void* RS, const void* Row2Tok,
    const void* SLABS, const void* WS_BASE, const void* P_OFF, const void* PATCHES,
    const void* Ones, const void* WorkBuf, void* Out,
    int max_m128, int tile, int use_r2t, int as_row_major, int has_patches,
    int w_evict_mode, uint64_t stream_u64
) {
    return launch_nvfp4_w13_v4b_impl<0>(
        A, AS, RS, Row2Tok, SLABS, WS_BASE, P_OFF, PATCHES, Ones, nullptr, WorkBuf, Out,
        max_m128, tile, use_r2t, as_row_major, has_patches, w_evict_mode,
        reinterpret_cast<cudaStream_t>(stream_u64));
}

int pf_nvfp4_w13_ws5_v4b_fused_situ(
    const void* A, const void* AS, const void* RS, const void* Row2Tok,
    const void* SLABS, const void* WS_BASE, const void* P_OFF, const void* PATCHES,
    const void* Ones, const void* Gammas, const void* WorkBuf, void* Out_situ,
    int max_m128, int tile, int use_r2t, int as_row_major, int has_patches,
    int w_evict_mode, uint64_t stream_u64
) {
    if (Gammas == nullptr) return -2;
    return launch_nvfp4_w13_v4b_impl<1>(
        A, AS, RS, Row2Tok, SLABS, WS_BASE, P_OFF, PATCHES, Ones, Gammas, WorkBuf, Out_situ,
        max_m128, tile, use_r2t, as_row_major, has_patches, w_evict_mode,
        reinterpret_cast<cudaStream_t>(stream_u64));
}

int pf_nvfp4_quant_288(
    const void* Y_situ,
    void* Values,
    void* Scales,
    void* RowScales,
    const void* Rows,
    const void* NumRows,
    const void* CtaExpert,
    const void* WAmax,
    const void* WDenominator,
    int max_rows,
    int tile,
    int scale_tile,
    uint64_t stream_u64
) {
    if (max_rows <= 0 || tile <= 0 || (tile & (tile - 1)) != 0) return -2;
    cudaStream_t st = reinterpret_cast<cudaStream_t>(stream_u64);
    int blocks = (max_rows + 7) / 8;
    if (blocks > 8192) blocks = 8192;
    if (scale_tile == 0) {
        pf_nvfp4_quant_288_kernel<1><<<blocks, 256, 0, st>>>(
            reinterpret_cast<const uint16_t*>(Y_situ),
            reinterpret_cast<uint8_t*>(Values),
            reinterpret_cast<uint8_t*>(Scales),
            reinterpret_cast<float*>(RowScales),
            reinterpret_cast<const int32_t*>(Rows),
            reinterpret_cast<const int32_t*>(NumRows),
            reinterpret_cast<const int32_t*>(CtaExpert),
            reinterpret_cast<const float*>(WAmax),
            reinterpret_cast<const float*>(WDenominator),
            tile, 0
        );
    } else {
        pf_nvfp4_quant_288_kernel<0><<<blocks, 256, 0, st>>>(
            reinterpret_cast<const uint16_t*>(Y_situ),
            reinterpret_cast<uint8_t*>(Values),
            reinterpret_cast<uint8_t*>(Scales),
            reinterpret_cast<float*>(RowScales),
            reinterpret_cast<const int32_t*>(Rows),
            reinterpret_cast<const int32_t*>(NumRows),
            reinterpret_cast<const int32_t*>(CtaExpert),
            reinterpret_cast<const float*>(WAmax),
            reinterpret_cast<const float*>(WDenominator),
            tile, scale_tile
        );
    }
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

}  // extern "C"
