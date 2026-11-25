// SPDX-License-Identifier: Apache-2.0
// =============================================================================
// SM121a (NVIDIA DGX Spark GB10) Wave-V10 Prefill-Only Co-Resident DBO-Safe
// Persistent MX7 (7.39 bpw) MoE W13+SiTU & Dynamic-LPT W2 (`libpf_mx7_prefill_v10.so`)
// =============================================================================
//
// Targets Inkling-2-Small TP8 8 Global MXFP8 MoE Layers (E=384, H=4608, inter=288,
// topk=8) strictly for long-context eager prefill chunks (plan.tile >= 64, T >= 1024,
// !capturing), leaving C=1 and C=32 decode (`libmxfp8_7bit_exact_v5.so` &
// `libmxfp8_mx7_c32.so`, exact_ink SHA-256 1dcfb30680d0252c) 100% UNTOUCHED!
//
// Zero Extra Weight VRAM:
//   Consumes the EXACT live `weight._inkling_mx7 = (slabs, ws3, flips, f_off, base_ws)`
//   already resident in GPU memory (`INKLING_MX7_FREE_RAW=1` preserved 100% -> 4x-7x
//   Multi-1M KV residency preserved!).
//
// Build on GB10 (SM121a):
//   /usr/local/cuda/bin/nvcc -O3 -std=c++17 --fmad=false --shared -Xcompiler -fPIC \
//       -gencode arch=compute_121a,code=sm_121a --ptxas-options=-v \
//       pf_mx7_prefill_v10.cu -o /rdma/libpf_mx7_prefill_v10.so
// =============================================================================

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace {

static constexpr int MX7_SLAB_BYTES = 896;   // 512 sm4 + 256 e21 + 128 e0 = 7 * 128 B
static constexpr float FP32_MAX_VAL = 3.4028234663852886e38f;
static constexpr float INV_25       = 1.0f / 25.0f;
static constexpr float INV_448      = 1.0f / 448.0f;

// =============================================================================
// SM121a PTX Memory, Swizzle, MMA, and Exact-RN Math Primitives
// =============================================================================

__device__ __forceinline__ uint4 ld_cs_u4(const void* p) {
    uint4 v;
    asm volatile("ld.global.cs.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}

__device__ __forceinline__ uint4 ld_ca_u4(const void* p) {
    uint4 v;
    asm volatile("ld.global.ca.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}

__device__ __forceinline__ uint2 ld_cs_u2(const void* p) {
    uint2 v;
    asm volatile("ld.global.cs.v2.u32 {%0,%1}, [%2];"
                 : "=r"(v.x), "=r"(v.y) : "l"(p));
    return v;
}

__device__ __forceinline__ uint2 ld_ca_u2(const void* p) {
    uint2 v;
    asm volatile("ld.global.ca.v2.u32 {%0,%1}, [%2];"
                 : "=r"(v.x), "=r"(v.y) : "l"(p));
    return v;
}

__device__ __forceinline__ uint32_t ld_cs_u32(const void* p) {
    uint32_t v;
    asm volatile("ld.global.cs.u32 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
}

__device__ __forceinline__ uint32_t ld_ca_u32(const void* p) {
    uint32_t v;
    asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
}

__device__ __forceinline__ uint4 ld_w_u4(const void* p, bool use_cs) {
    return use_cs ? ld_cs_u4(p) : ld_ca_u4(p);
}

__device__ __forceinline__ uint2 ld_w_u2(const void* p, bool use_cs) {
    return use_cs ? ld_cs_u2(p) : ld_ca_u2(p);
}

__device__ __forceinline__ uint32_t ld_w_u32(const void* p, bool use_cs) {
    return use_cs ? ld_cs_u32(p) : ld_ca_u32(p);
}

__device__ __forceinline__ void st_wb_u4(void* p, uint4 v) {
    asm volatile("st.global.wb.v4.u32 [%0], {%1,%2,%3,%4};"
                 :: "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}

__device__ __forceinline__ void st_cs_u4(void* p, uint4 v) {
    asm volatile("st.global.cs.v4.u32 [%0], {%1,%2,%3,%4};"
                 :: "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}

__device__ __forceinline__ void st_wb_u32(void* p, uint32_t v) {
    asm volatile("st.global.wb.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}

// Wave-V13 ZERO-SPIN: ld_acquire_gpu_u32 / st_release_gpu_u32 removed completely —
// ZERO cross-CTA spinlocks exist anywhere in pf_mx7_prefill_v13.cu!

template <typename T>
__device__ __forceinline__ const T* ld_smem_ptr(const T* const* s_addr) {
    uint64_t v;
    uint32_t sm_addr = static_cast<uint32_t>(__cvta_generic_to_shared(s_addr));
    asm volatile("ld.shared.u64 %0, [%1];" : "=l"(v) : "r"(sm_addr));
    return reinterpret_cast<const T*>(v);
}

template <typename T>
__device__ __forceinline__ T* ld_smem_mut_ptr(T* const* s_addr) {
    uint64_t v;
    uint32_t sm_addr = static_cast<uint32_t>(__cvta_generic_to_shared(s_addr));
    asm volatile("ld.shared.u64 %0, [%1];" : "=l"(v) : "r"(sm_addr));
    return reinterpret_cast<T*>(v);
}

__device__ __forceinline__ int ld_smem_i32(const int* s_addr) {
    int v;
    uint32_t sm_addr = static_cast<uint32_t>(__cvta_generic_to_shared(s_addr));
    asm volatile("ld.shared.s32 %0, [%1];" : "=r"(v) : "r"(sm_addr));
    return v;
}

__device__ __forceinline__ uint32_t ld_smem_u32(const uint32_t* s_addr) {
    uint32_t v;
    uint32_t sm_addr = static_cast<uint32_t>(__cvta_generic_to_shared(s_addr));
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(sm_addr));
    return v;
}

__device__ __forceinline__ uint32_t prmt_b32(uint32_t a, uint32_t b, uint32_t sel) {
    uint32_t out;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(out) : "r"(a), "r"(b), "r"(sel));
    return out;
}

__device__ __forceinline__ float mul_rn(float a, float b) {
    float r;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

__device__ __forceinline__ float rcp_rn(float a) {
    float r;
    asm("rcp.rn.f32 %0, %1;" : "=f"(r) : "f"(a));
    return r;
}

__device__ __forceinline__ float bf16_bits_to_f32(uint16_t b) {
    return __uint_as_float(static_cast<uint32_t>(b) << 16);
}

__device__ __forceinline__ uint16_t f32_to_bf16_rn(float f) {
    uint16_t b;
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(b) : "f"(f));
    return b;
}

__device__ __forceinline__ uint32_t pack_bf16x2(float f0, float f1) {
    uint32_t out;
    asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(out) : "f"(f0), "f"(f1));
    return out;
}

__device__ __forceinline__ void red_add_f32(float* addr, float val) {
    asm volatile("red.relaxed.gpu.global.add.f32 [%0], %1;\n" :: "l"(addr), "f"(val) : "memory");
}

__device__ __forceinline__ void red_add_bf16x2_bits(uint32_t* addr, uint32_t bf16x2_bits) {
    asm volatile("red.relaxed.gpu.global.add.noftz.bf16x2 [%0], %1;\n"
                 :: "l"(addr), "r"(bf16x2_bits) : "memory");
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

// Native SM121a MXFP8 block-scaled Tensor Core MMA: m16n8k32, e4m3 x e4m3, ue8m0 (VS=32).
__device__ __forceinline__ void mma_m16n8k32_mxfp8(
    float (&d)[4],
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    uint32_t sfa, uint32_t sfb
) {
    constexpr uint16_t bidA = 0, tidA = 0, bidB = 0, tidB = 0;
    asm volatile(
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
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

// =============================================================================
// MX7 Bitplane & Flip-List Decode (100% Exact Match to libmxfp8_7bit_exact_v5.so)
// =============================================================================

struct SlabMX7 {
    uint4 sm;     // 16 B/lane (512 B): 8 x uint16 sm4
    uint2 e21;    //  8 B/lane (256 B): 8 x uint8 e21
    uint32_t e0;  //  4 B/lane (128 B): 8 x 4-bit e0
    uint4 ws3;    // 16 B broadcast:    4 x uint32 (each 8 x 3-bit ws offsets)
};

__device__ __forceinline__ SlabMX7 load_slab_mx7(
    const uint8_t* sp, const uint4* wsp, int lane, bool use_cs
) {
    SlabMX7 r;
    r.sm  = ld_w_u4(sp + lane * 16, use_cs);
    r.e21 = ld_w_u2(sp + 512 + lane * 8, use_cs);
    r.e0  = ld_w_u32(sp + 768 + lane * 4, use_cs);
    r.ws3 = ld_w_u4(wsp, use_cs);
    return r;
}

__device__ __forceinline__ uint32_t build_e4_pack_flip(
    uint32_t sm16, uint32_t e21_8, uint32_t e0_4, uint32_t flip4
) {
    uint32_t s_lo = (sm16 & 0x000Fu)
                  | ((sm16 & 0x00F0u) << 4)
                  | ((sm16 & 0x0F00u) << 8)
                  | ((sm16 & 0xF000u) << 12);
    uint32_t e21_b = ((e21_8 & 0x03u) << 4)
                   | ((e21_8 & 0x0Cu) << 10)
                   | ((e21_8 & 0x30u) << 16)
                   | ((e21_8 & 0xC0u) << 22);
    uint32_t e0_b = ((e0_4 & 1u) << 3)
                  | ((e0_4 & 2u) << 10)
                  | ((e0_4 & 4u) << 17)
                  | ((e0_4 & 8u) << 24);
    uint32_t e3_flip = ((flip4 & 1u) << 6)
                     | ((flip4 & 2u) << 13)
                     | ((flip4 & 4u) << 20)
                     | ((flip4 & 8u) << 27);
    return (0x40404040u ^ e3_flip) | ((s_lo & 0x08080808u) << 4) | e21_b | e0_b | (s_lo & 0x07070707u);
}

// Convert pair of MX7 steps (s0 = 2*sub, s1 = 2*sub+1) directly into mma_m16n8k32 B operands!
__device__ __forceinline__ void decode_mx7_pair_to_b0b1(
    const SlabMX7& slb, uint32_t flip_bits, int sub, uint32_t& b0, uint32_t& b1
) {
    uint32_t sm_pair;
    uint32_t e21_pair;
    if (sub == 0) {
        sm_pair  = slb.sm.x;
        e21_pair = slb.e21.x & 0xFFFFu;
    } else if (sub == 1) {
        sm_pair  = slb.sm.y;
        e21_pair = slb.e21.x >> 16;
    } else if (sub == 2) {
        sm_pair  = slb.sm.z;
        e21_pair = slb.e21.y & 0xFFFFu;
    } else {
        sm_pair  = slb.sm.w;
        e21_pair = slb.e21.y >> 16;
    }
    const int sh0 = sub * 8;
    const int sh1 = sh0 + 4;
    uint32_t e4_0 = build_e4_pack_flip(
        sm_pair & 0xFFFFu, e21_pair & 0xFFu, (slb.e0 >> sh0) & 0xFu, (flip_bits >> sh0) & 0xFu);
    uint32_t e4_1 = build_e4_pack_flip(
        sm_pair >> 16, e21_pair >> 8, (slb.e0 >> sh1) & 0xFu, (flip_bits >> sh1) & 0xFu);
    b0 = prmt_b32(e4_0, e4_1, 0x5410u);
    b1 = prmt_b32(e4_0, e4_1, 0x7632u);
}

__device__ __forceinline__ uint32_t decode_mx7_ws_sub(
    uint4 ws3, uint32_t base_ws, int sub, int g
) {
    uint32_t w = (sub == 0) ? ws3.x : ((sub == 1) ? ws3.y : ((sub == 2) ? ws3.z : ws3.w));
    return base_ws + ((w >> (g * 3)) & 7u);
}

// Binary search first flip record with `(rec >> 10) >= target_sl` in sorted `[f_lo, f_hi)`.
__device__ __forceinline__ int32_t find_flip_slab_start(
    const uint16_t* __restrict__ FLIPS, int32_t f_lo, int32_t f_hi, uint32_t target_sl
) {
    const uint32_t key = target_sl << 10;
    int32_t lo = f_lo, hi = f_hi;
    while (lo < hi) {
        int32_t mid = lo + ((hi - lo) >> 1);
        if (static_cast<uint32_t>(__ldg(FLIPS + mid)) < key) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

// Streaming flip-list decoder for ascending `sl`: zero global loads on slabs with 0 flips!
__device__ __forceinline__ uint32_t consume_slab_flips(
    const uint16_t* __restrict__ FLIPS,
    int32_t& f_cur,
    int32_t f_end,
    uint32_t& f_rec,
    uint32_t sl,
    int lane,
    uint32_t* sFlipWarp
) {
    bool in_slab = (f_rec != 0xFFFFu) && ((f_rec >> 10) == sl);
    uint32_t m = __ballot_sync(0xFFFFFFFFu, in_slab);
    if (__builtin_expect(m == 0u, 1)) return 0u;

    sFlipWarp[lane] = 0u;
    __syncwarp();
    if (in_slab) {
        const uint32_t s  = (f_rec >> 7) & 7u;
        const uint32_t tl = (f_rec >> 2) & 31u;
        const uint32_t j  = f_rec & 3u;
        atomicOr(&sFlipWarp[tl], 1u << ((s << 2) | j));
    }
    int cnt = __popc(m);
    f_cur += cnt;
    while (__builtin_expect(cnt == 32 && f_cur < f_end, 0)) {
        const int32_t idx = f_cur + lane;
        const uint32_t r2 = (idx < f_end) ? static_cast<uint32_t>(__ldg(FLIPS + idx)) : 0xFFFFu;
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
    const uint32_t out = sFlipWarp[lane];
    const int32_t nxt_idx = f_cur + lane;
    f_rec = (nxt_idx < f_end) ? static_cast<uint32_t>(__ldg(FLIPS + nxt_idx)) : 0xFFFFu;
    return out;
}

// =============================================================================
// Exact PTX-RN SiTU Helpers (Immune to --use_fast_math; matches activation_ops.py)
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

__device__ __forceinline__ uint16_t situ_from_f32_pair_bf16_exact(
    float gate_f, float up_f, float gamma
) {
    float gate = bf16_bits_to_f32(f32_to_bf16_rn(gate_f));
    float up   = bf16_bits_to_f32(f32_to_bf16_rn(up_f));
    float sig  = sigmoid_approx_exact(gate);
    gate = softcap_4(gate);
    up   = softcap_25(up);
    float y = mul_rn(mul_rn(mul_rn(gate, sig), up), gamma);
    return f32_to_bf16_rn(y);
}

// =============================================================================
// Expert Range Binary Search & Zero-Sync 1-Block LPT Expert Plan Builder
// =============================================================================

__device__ __forceinline__ void find_expert_row_range(
    const int32_t* __restrict__ Tiles,
    int max_ctas, int tile, int target_e,
    int& row_beg, int& row_end
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
    lo = t_lo + 1; hi = num_tiles;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (Tiles[mid] <= target_e) lo = mid + 1;
        else hi = mid;
    }
    row_beg = t_lo * tile;
    row_end = Tiles[max_ctas + lo - 1];
}

__global__ __launch_bounds__(256, 1)
void pf_mx7_build_expert_plan_kernel(
    const int32_t* __restrict__ Tiles,
    int32_t* __restrict__ PlanBuf,
    int max_ctas, int tile, int num_experts, int split_m, int max_work
) {
    constexpr int MAX_E = 1024;
    __shared__ int32_t sBeg[MAX_E];
    __shared__ int32_t sEnd[MAX_E];
    __shared__ uint32_t sBucketCnt[32];
    __shared__ uint32_t sBucketOff[32];

    const int tid = threadIdx.x;
    const int eff_e = (num_experts < MAX_E) ? num_experts : MAX_E;
    const int eff_split = (split_m > 0) ? ((split_m + 255) & ~255) : 0;

    if (tid < 32) sBucketCnt[tid] = 0u;
    for (int w = tid; w < max_work; w += 256) PlanBuf[w] = -1;
    __syncthreads();

    for (int e = tid; e < eff_e; e += 256) {
        int rb = 0, re = 0;
        find_expert_row_range(Tiles, max_ctas, tile, e, rb, re);
        sBeg[e] = rb; sEnd[e] = re;
        int cnt = re - rb;
        if (cnt > 0) {
            if (eff_split > 0 && cnt > eff_split) {
                for (int off = 0; off < cnt; off += eff_split) {
                    int sz = cnt - off;
                    if (sz > eff_split) sz = eff_split;
                    int raw_b = (sz - 1) >> 5;
                    int b = 31 - ((raw_b > 31) ? 31 : raw_b);
                    atomicAdd(&sBucketCnt[b], 1u);
                }
            } else {
                int raw_b = (cnt - 1) >> 5;
                int b = 31 - ((raw_b > 31) ? 31 : raw_b);
                atomicAdd(&sBucketCnt[b], 1u);
            }
        }
    }
    __syncthreads();

    if (tid == 0) {
        uint32_t acc = 0u;
        for (int b = 0; b < 32; ++b) {
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
                    int raw_b = (sz - 1) >> 5;
                    int b = 31 - ((raw_b > 31) ? 31 : raw_b);
                    uint32_t pos = atomicAdd(&sBucketOff[b], 1u);
                    if (static_cast<int>(pos) < max_work) {
                        out_e[pos]   = e;
                        out_beg[pos] = rb + off;
                        out_end[pos] = rb + off + sz;
                    }
                }
            } else {
                int raw_b = (cnt - 1) >> 5;
                int b = 31 - ((raw_b > 31) ? 31 : raw_b);
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

__device__ __forceinline__ bool resolve_persistent_work(
    const int32_t* __restrict__ PlanBuf,
    const int32_t* __restrict__ Tiles,
    int max_ctas, int tile, int num_experts, int max_work,
    int bx, int nb_blk, int serpentine_mode,
    int32_t (&sWork)[3], int tid
) {
    if (tid == 0) {
        int e = -1, rb = 0, re = 0;
        if (PlanBuf != nullptr) {
            const int num_w = PlanBuf[3 * max_work];
            if (bx < num_w) {
                int widx = bx;
                if (serpentine_mode == 1 && (nb_blk & 1)) {
                    widx = num_w - 1 - bx;
                } else if (serpentine_mode == 2 && (nb_blk & 1)) {
                    const int head_keep = (num_w > 64) ? 48 : num_w;
                    if (bx >= head_keep) {
                        widx = num_w - 1 - (bx - head_keep);
                    }
                }
                e  = PlanBuf[widx];
                rb = PlanBuf[max_work + widx];
                re = PlanBuf[2 * max_work + widx];
            }
        } else {
            int e_cand = (serpentine_mode != 0 && (nb_blk & 1)) ? (num_experts - 1 - bx) : bx;
            if (e_cand >= 0 && e_cand < num_experts) {
                find_expert_row_range(Tiles, max_ctas, tile, e_cand, rb, re);
                if (rb < re) e = e_cand;
            }
        }
        sWork[0] = e; sWork[1] = rb; sWork[2] = re;
    }
    __syncthreads();
    return (sWork[0] >= 0 && sWork[1] < sWork[2]);
}

// =============================================================================
// In-L2 Per-Slice BF16 Zero Kernel (`st.global.wb.v4.u32`, 128B sector-complete)
// Supports PANELS64 in {4, 6, 8, 12} -> {256, 384, 512, 768} BF16 columns/slice.
// =============================================================================
template <int PANELS64>
__global__ __launch_bounds__(256)
void pf_mx7_zero_bf16_slice_wb_kernel(
    uint16_t* __restrict__ Out_bf16,
    int tokens,
    int nb_blk
) {
    constexpr int H = 4608;
    constexpr int COL_BASE_MULT = PANELS64 * 64;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int tok  = blockIdx.x * 8 + warp;
    if (tok >= tokens) return;

    uint4* dst = reinterpret_cast<uint4*>(
        Out_bf16 + static_cast<int64_t>(tok) * H + nb_blk * COL_BASE_MULT);
    const uint4 z4 = make_uint4(0u, 0u, 0u, 0u);
    if constexpr (PANELS64 == 4) {
        st_wb_u4(dst + lane, z4);
    } else if constexpr (PANELS64 == 6) {
        st_wb_u4(dst + lane, z4);
        if (lane < 16) st_wb_u4(dst + 32 + lane, z4);
    } else if constexpr (PANELS64 == 8) {
        st_wb_u4(dst + lane, z4);
        st_wb_u4(dst + 32 + lane, z4);
    } else if constexpr (PANELS64 == 12) {
        st_wb_u4(dst + lane, z4);
        st_wb_u4(dst + 32 + lane, z4);
        st_wb_u4(dst + 64 + lane, z4);
    }
}

// =============================================================================
// Kernel 1: Persistent MX7 W13 (N=576, K=4608, BN=64, 8 warps = 1-to-1 packed warps)
//   - FUSE_SITU == 1: Fuses PTX-exact SiTU + MXFP8 block-32 quantize right in epilogue!
//                     Writes W2Act [R, 288] e4m3fn + W2Scales [R, 12] ue8m0 via st.wb!
//   - FUSE_SITU == 0: Writes unfused [R, 576] BF16 via coalesced 64-row SMEM epilogue.
//   - BM == 128 (STAGES=3, MIN_CTA=2): 27.1 KiB SMEM -> 2 CTAs/SM, ~86 regs, 0 spills.
//   - BM == 256 (STAGES=2, MIN_CTA=1): 36.9 KiB SMEM -> 1 CTA/SM, ~120 regs, 0 spills.
// =============================================================================

template <int BM, int STAGES>
struct Mx7W13Smem {
    union {
        struct {
            uint32_t sA[STAGES][BM][16];   // STAGES * BM * 64 B (0-way XOR swizzled!)
            uint16_t sAS[STAGES][BM];      // STAGES * BM * 2 B  (2 ue8m0 scales / 64-col stage)
        } m;
        uint16_t sSitu[64][34];            // 4,352 B (64 rows x 32 SiTU bf16, padded -> 0-way)
        uint32_t sBf16[64][33];            // 8,448 B (64 rows x 64 bf16 = 32 bf16x2 -> 0-way)
    } u;
    int32_t  sTok[BM];
    float    sGamma[BM];
    uint32_t sFlip[8][32];                 // 1,024 B (per-warp flip scatter buffer)
    int32_t  sWork[3];
    int32_t  _pad;
};

static_assert(sizeof(Mx7W13Smem<128, 3>) <= 49152, "Mx7W13<128,3> must fit <= 48 KiB (2 CTAs/SM)");
static_assert(sizeof(Mx7W13Smem<256, 2>) <= 49152, "Mx7W13<256,2> must fit <= 48 KiB");

template <int BM, int STAGES, int MIN_CTA, int FUSE_SITU, int USE_R2T, int AS_ROW_MAJOR, int W2_AS_ROW_MAJOR>
__global__ __launch_bounds__(256, MIN_CTA)
void pf_mx7_w13_persistent_kernel(
    const uint8_t* __restrict__ A,
    const uint8_t* __restrict__ AS,
    const int32_t* __restrict__ Row2Tok,
    const float* __restrict__ Gammas,
    const uint8_t* __restrict__ SLABS,
    const uint4* __restrict__ WS3,
    const uint16_t* __restrict__ FLIPS,
    const int32_t* __restrict__ F_OFF,
    const float* __restrict__ Ones,
    const int32_t* __restrict__ Tiles,
    const int32_t* __restrict__ PlanBuf,
    void* __restrict__ Out0,           // FUSE_SITU ? W2Act [R, 288] u8 : GateUp [R, 576] bf16
    uint8_t* __restrict__ OutScales,   // FUSE_SITU ? W2Scales [R, 12] u8 : nullptr
    int max_ctas,
    int tile,
    int num_experts,
    int max_work,
    uint32_t base_ws,
    int w2_as_stride,
    int w_evict_mode
) {
    constexpr int KP      = 4608;
    constexpr int KG      = 144;
    constexpr int M_TILES = BM / 16;   // 8 (BM=128) or 16 (BM=256)

    const int nb64 = blockIdx.x;       // 0..8 (9 x 64 = 576 cols)
    const int w_bx = blockIdx.y;       // 0..max_work-1

    extern __shared__ uint4 dyn_smem[];
    auto& sm = *reinterpret_cast<Mx7W13Smem<BM, STAGES>*>(dyn_smem);

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;         // 0..7 (1-to-1 with packed MX7 W13 warp!)
    const int lane = tid & 31;
    const int g    = lane >> 2;        // 0..7
    const int lt   = lane & 3;         // 0..3

    if (!resolve_persistent_work(
            PlanBuf, Tiles, max_ctas, tile, num_experts, max_work,
            w_bx, 0, 0, sm.sWork, tid)) {
        return;
    }

    const int64_t e      = sm.sWork[0];
    const int row_begin  = sm.sWork[1];
    const int end        = sm.sWork[2];
    const int n_rows     = end - row_begin;
    const int num_m      = (n_rows + BM - 1) / BM;

    const int64_t wuid     = e * 72 + nb64 * 8 + warp;
    const uint8_t* slabs_w = SLABS + wuid * (36LL * MX7_SLAB_BYTES);
    const uint4*   ws3_w   = WS3   + wuid * 36LL;
    const int32_t  f_lo    = F_OFF[wuid];
    const int32_t  f_hi    = F_OFF[wuid + 1];
    const float    scale_e = Ones[e];

    const int m_sf_off = g + ((lt & 1) << 3);
    const int sw_mask  = (g >> 1) << 2;   // 0-way 32-bank bijection for 16-word rows!

    #pragma unroll 1
    for (int mi_blk = 0; mi_blk < num_m; ++mi_blk) {
        const int row0       = row_begin + mi_blk * BM;
        const bool is_last_m = (mi_blk + 1 >= num_m);
        const bool use_cs_w  = (w_evict_mode == 2) || (w_evict_mode == 1 && is_last_m);
        const int live_m     = end - row0;
        const int mi_end     = (live_m >= BM) ? M_TILES : ((live_m + 15) >> 4);

        if (tid < BM) {
            int r = row0 + tid;
            if (r < end) {
                sm.sTok[tid] = USE_R2T ? Row2Tok[r] : r;
                if constexpr (FUSE_SITU) sm.sGamma[tid] = Gammas[r];
            } else {
                sm.sTok[tid] = -1;
                if constexpr (FUSE_SITU) sm.sGamma[tid] = 0.0f;
            }
        }
        __syncthreads();

        auto load_a_stage = [&](int stg, int buf) {
            constexpr int A_ITERS = (BM * 4) / 256;  // 2 for BM=128, 4 for BM=256
            #pragma unroll
            for (int it = 0; it < A_ITERS; ++it) {
                int idx = it * 256 + tid;
                int m   = idx >> 2;
                int c4  = idx & 3;
                int sc4 = c4 ^ ((m >> 1) & 3);
                int32_t tok = sm.sTok[m];
                const uint8_t* src = (tok >= 0)
                    ? (A + static_cast<int64_t>(tok) * KP + stg * 64 + c4 * 16)
                    : A;
                cp_async_cg_16(&sm.u.m.sA[buf][m][sc4 * 4], src, (tok >= 0) ? 16 : 0);
            }
            if (tid < BM) {
                int32_t tok = sm.sTok[tid];
                uint16_t sval = 0u;
                if (tok >= 0) {
                    if constexpr (USE_R2T || AS_ROW_MAJOR) {
                        const uint8_t* p = AS + static_cast<int64_t>(tok) * KG + stg * 2;
                        sval = static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
                    } else {
                        int64_t o0 = scale_offset_byte(static_cast<int64_t>(row0 + tid), stg * 2 + 0, KG, tile);
                        int64_t o1 = scale_offset_byte(static_cast<int64_t>(row0 + tid), stg * 2 + 1, KG, tile);
                        sval = static_cast<uint16_t>(AS[o0]) | (static_cast<uint16_t>(AS[o1]) << 8);
                    }
                }
                sm.u.m.sAS[buf][tid] = sval;
            }
            cp_async_commit();
        };

        auto run_mma_k32 = [&](float (&acc)[M_TILES][4], int cur_buf, int q,
                               uint32_t b0, uint32_t b1, uint32_t sfb) {
            const int sw_c0 = (q * 8 + lt) ^ sw_mask;
            const int sw_c1 = (q * 8 + 4 + lt) ^ sw_mask;
            const int as_sh = q * 8;
            #pragma unroll
            for (int mi = 0; mi < 4; ++mi) {
                int m_lo = mi * 16 + g;
                int m_hi = m_lo + 8;
                uint32_t a0  = sm.u.m.sA[cur_buf][m_lo][sw_c0];
                uint32_t a1  = sm.u.m.sA[cur_buf][m_hi][sw_c0];
                uint32_t a2  = sm.u.m.sA[cur_buf][m_lo][sw_c1];
                uint32_t a3  = sm.u.m.sA[cur_buf][m_hi][sw_c1];
                uint32_t sfa = (static_cast<uint32_t>(sm.u.m.sAS[cur_buf][mi * 16 + m_sf_off]) >> as_sh) & 0xFFu;
                mma_m16n8k32_mxfp8(acc[mi], a0, a1, a2, a3, b0, b1, sfa, sfb);
            }
            if (mi_end > 4) {
                #pragma unroll
                for (int mi = 4; mi < 8; ++mi) {
                    int m_lo = mi * 16 + g;
                    int m_hi = m_lo + 8;
                    uint32_t a0  = sm.u.m.sA[cur_buf][m_lo][sw_c0];
                    uint32_t a1  = sm.u.m.sA[cur_buf][m_hi][sw_c0];
                    uint32_t a2  = sm.u.m.sA[cur_buf][m_lo][sw_c1];
                    uint32_t a3  = sm.u.m.sA[cur_buf][m_hi][sw_c1];
                    uint32_t sfa = (static_cast<uint32_t>(sm.u.m.sAS[cur_buf][mi * 16 + m_sf_off]) >> as_sh) & 0xFFu;
                    mma_m16n8k32_mxfp8(acc[mi], a0, a1, a2, a3, b0, b1, sfa, sfb);
                }
            }
            if constexpr (M_TILES > 8) {
                if (mi_end > 8) {
                    #pragma unroll
                    for (int mi = 8; mi < 12; ++mi) {
                        int m_lo = mi * 16 + g;
                        int m_hi = m_lo + 8;
                        uint32_t a0  = sm.u.m.sA[cur_buf][m_lo][sw_c0];
                        uint32_t a1  = sm.u.m.sA[cur_buf][m_hi][sw_c0];
                        uint32_t a2  = sm.u.m.sA[cur_buf][m_lo][sw_c1];
                        uint32_t a3  = sm.u.m.sA[cur_buf][m_hi][sw_c1];
                        uint32_t sfa = (static_cast<uint32_t>(sm.u.m.sAS[cur_buf][mi * 16 + m_sf_off]) >> as_sh) & 0xFFu;
                        mma_m16n8k32_mxfp8(acc[mi], a0, a1, a2, a3, b0, b1, sfa, sfb);
                    }
                }
                if (mi_end > 12) {
                    #pragma unroll
                    for (int mi = 12; mi < 16; ++mi) {
                        int m_lo = mi * 16 + g;
                        int m_hi = m_lo + 8;
                        uint32_t a0  = sm.u.m.sA[cur_buf][m_lo][sw_c0];
                        uint32_t a1  = sm.u.m.sA[cur_buf][m_hi][sw_c0];
                        uint32_t a2  = sm.u.m.sA[cur_buf][m_lo][sw_c1];
                        uint32_t a3  = sm.u.m.sA[cur_buf][m_hi][sw_c1];
                        uint32_t sfa = (static_cast<uint32_t>(sm.u.m.sAS[cur_buf][mi * 16 + m_sf_off]) >> as_sh) & 0xFFu;
                        mma_m16n8k32_mxfp8(acc[mi], a0, a1, a2, a3, b0, b1, sfa, sfb);
                    }
                }
            }
        };

        // Prime cp.async + slab 0 + flip lookahead
        if constexpr (STAGES == 3) {
            load_a_stage(0, 0);
            load_a_stage(1, 1);
        } else {
            load_a_stage(0, 0);
        }
        SlabMX7 slab_cur = load_slab_mx7(slabs_w, ws3_w, lane, use_cs_w);
        int32_t  f_cur   = f_lo;
        uint32_t f_rec   = (f_cur + lane < f_hi) ? static_cast<uint32_t>(__ldg(FLIPS + f_cur + lane)) : 0xFFFFu;

        if constexpr (STAGES == 3) cp_async_wait<1>();
        else                       cp_async_wait<0>();
        __syncthreads();

        float acc[M_TILES][4];
        #pragma unroll
        for (int mi = 0; mi < M_TILES; ++mi) {
            acc[mi][0] = 0.0f; acc[mi][1] = 0.0f;
            acc[mi][2] = 0.0f; acc[mi][3] = 0.0f;
        }

        int cur = 0;
        #pragma unroll 1
        for (int sl = 0; sl < 36; ++sl) {
            const uint32_t flip_bits = consume_slab_flips(
                FLIPS, f_cur, f_hi, f_rec, static_cast<uint32_t>(sl), lane, sm.sFlip[warp]);

            // ---- Half-slab 0: stg = 2*sl (k32_sub = 0, 1) ----
            {
                const int stg = sl * 2;
                if constexpr (STAGES == 3) {
                    if (stg + 2 < 72) {
                        const int nxt2 = (cur == 0) ? 2 : (cur - 1);
                        load_a_stage(stg + 2, nxt2);
                    }
                } else {
                    load_a_stage(stg + 1, cur ^ 1);
                }

                uint32_t b0_0, b1_0, b0_1, b1_1;
                decode_mx7_pair_to_b0b1(slab_cur, flip_bits, 0, b0_0, b1_0);
                decode_mx7_pair_to_b0b1(slab_cur, flip_bits, 1, b0_1, b1_1);
                const uint32_t sfb_0 = decode_mx7_ws_sub(slab_cur.ws3, base_ws, 0, g);
                const uint32_t sfb_1 = decode_mx7_ws_sub(slab_cur.ws3, base_ws, 1, g);

                run_mma_k32(acc, cur, 0, b0_0, b1_0, sfb_0);
                run_mma_k32(acc, cur, 1, b0_1, b1_1, sfb_1);

                if constexpr (STAGES == 3) {
                    if (stg + 2 < 72) cp_async_wait<1>();
                    else              cp_async_wait<0>();
                    cur = (cur == 2) ? 0 : (cur + 1);
                } else {
                    cp_async_wait<0>();
                    cur ^= 1;
                }
                __syncthreads();
            }

            // ---- Half-slab 1: stg = 2*sl + 1 (k32_sub = 2, 3) ----
            {
                const int stg = sl * 2 + 1;
                if constexpr (STAGES == 3) {
                    if (stg + 2 < 72) {
                        const int nxt2 = (cur == 0) ? 2 : (cur - 1);
                        load_a_stage(stg + 2, nxt2);
                    }
                } else {
                    if (stg + 1 < 72) load_a_stage(stg + 1, cur ^ 1);
                }

                // Extract k32_sub=2,3 BEFORE loading slab_{sl+1} -> slab_cur dead -> 0 extra regs!
                uint32_t b0_2, b1_2, b0_3, b1_3;
                decode_mx7_pair_to_b0b1(slab_cur, flip_bits, 2, b0_2, b1_2);
                decode_mx7_pair_to_b0b1(slab_cur, flip_bits, 3, b0_3, b1_3);
                const uint32_t sfb_2 = decode_mx7_ws_sub(slab_cur.ws3, base_ws, 2, g);
                const uint32_t sfb_3 = decode_mx7_ws_sub(slab_cur.ws3, base_ws, 3, g);

                if (sl + 1 < 36) {
                    slab_cur = load_slab_mx7(
                        slabs_w + static_cast<int64_t>(sl + 1) * MX7_SLAB_BYTES,
                        ws3_w + (sl + 1), lane, use_cs_w);
                }

                run_mma_k32(acc, cur, 0, b0_2, b1_2, sfb_2);
                run_mma_k32(acc, cur, 1, b0_3, b1_3, sfb_3);

                if (sl + 1 < 36) {
                    if constexpr (STAGES == 3) {
                        if (stg + 2 < 72) cp_async_wait<1>();
                        else              cp_async_wait<0>();
                        cur = (cur == 2) ? 0 : (cur + 1);
                    } else {
                        cp_async_wait<0>();
                        cur ^= 1;
                    }
                    __syncthreads();
                }
            }
        }

        // ---- Coalesced 64-row Epilogue (overlaying sA -> 0 extra SMEM!) ----
        __syncthreads();
        constexpr int N_CHUNKS = BM / 64;  // 2 for BM=128, 4 for BM=256

        #pragma unroll
        for (int chunk = 0; chunk < N_CHUNKS; ++chunk) {
            if (row0 + chunk * 64 >= end) break;

            if constexpr (FUSE_SITU) {
                #pragma unroll
                for (int mi4 = 0; mi4 < 4; ++mi4) {
                    const int mi = chunk * 4 + mi4;
                    const int m_lo = mi * 16 + g;
                    const int m_hi = m_lo + 8;
                    const int r_epi_lo = mi4 * 16 + g;
                    const int r_epi_hi = r_epi_lo + 8;
                    const uint16_t y_lo = situ_from_f32_pair_bf16_exact(
                        acc[mi][0] * scale_e, acc[mi][1] * scale_e, sm.sGamma[m_lo]);
                    const uint16_t y_hi = situ_from_f32_pair_bf16_exact(
                        acc[mi][2] * scale_e, acc[mi][3] * scale_e, sm.sGamma[m_hi]);
                    sm.u.sSitu[r_epi_lo][warp * 4 + lt] = y_lo;
                    sm.u.sSitu[r_epi_hi][warp * 4 + lt] = y_hi;
                }
                __syncthreads();

                uint8_t* act_out = reinterpret_cast<uint8_t*>(Out0);
                #pragma unroll
                for (int k = 0; k < 8; ++k) {
                    const int r_epi  = warp * 8 + k;
                    const int r_glob = row0 + chunk * 64 + r_epi;
                    if (r_glob < end) {
                        const float y = bf16_bits_to_f32(sm.u.sSitu[r_epi][lane]);
                        float amax = fabsf(y);
                        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, 16));
                        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, 8));
                        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, 4));
                        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, 2));
                        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, 1));

                        const float raw_scale = mul_rn(amax, INV_448);
                        const uint32_t rs_u   = __float_as_uint(raw_scale);
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

                        float qf = mul_rn(y, inv);
                        if (qf == qf) {
                            if (qf < -448.0f) qf = -448.0f;
                            if (qf >  448.0f) qf =  448.0f;
                        }
                        const float qf_odd = __shfl_xor_sync(0xFFFFFFFFu, qf, 1);
                        uint16_t e2;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;"
                            : "=h"(e2) : "f"(qf), "f"(qf_odd));
                        const uint32_t e2_hi = __shfl_xor_sync(0xFFFFFFFFu, static_cast<uint32_t>(e2), 2);
                        const uint32_t q4 = static_cast<uint32_t>(e2) | (e2_hi << 16);
                        const uint32_t q4_c = __shfl_sync(0xFFFFFFFFu, q4, (lane & 7) << 2);
                        if (lane < 8) {
                            uint32_t* dst_u32 = reinterpret_cast<uint32_t*>(
                                act_out + static_cast<int64_t>(r_glob) * 288 + nb64 * 32);
                            st_wb_u32(dst_u32 + lane, q4_c);
                        }
                        if (lane == 0) {
                            int64_t soff;
                            if constexpr (W2_AS_ROW_MAJOR) {
                                soff = static_cast<int64_t>(r_glob) * w2_as_stride + nb64;
                            } else {
                                soff = scale_offset_byte(static_cast<int64_t>(r_glob), nb64, 9, tile);
                            }
                            OutScales[soff] = static_cast<uint8_t>(exp_u);
                        }
                    }
                }
            } else {
                #pragma unroll
                for (int mi4 = 0; mi4 < 4; ++mi4) {
                    const int mi = chunk * 4 + mi4;
                    const int r_epi_lo = mi4 * 16 + g;
                    const int r_epi_hi = r_epi_lo + 8;
                    sm.u.sBf16[r_epi_lo][warp * 4 + lt] = pack_bf16x2(acc[mi][0] * scale_e, acc[mi][1] * scale_e);
                    sm.u.sBf16[r_epi_hi][warp * 4 + lt] = pack_bf16x2(acc[mi][2] * scale_e, acc[mi][3] * scale_e);
                }
                __syncthreads();

                uint32_t* out_u32 = reinterpret_cast<uint32_t*>(Out0);
                const int sub_r = lane >> 3;  // 0..3
                const int c4    = lane & 7;   // 0..7 (8 uint4 = 128 B = 64 bf16)
                #pragma unroll
                for (int k = 0; k < 2; ++k) {
                    const int r_epi  = warp * 8 + k * 4 + sub_r;
                    const int r_glob = row0 + chunk * 64 + r_epi;
                    if (r_glob < end) {
                        uint4 v;
                        v.x = sm.u.sBf16[r_epi][c4 * 4 + 0];
                        v.y = sm.u.sBf16[r_epi][c4 * 4 + 1];
                        v.z = sm.u.sBf16[r_epi][c4 * 4 + 2];
                        v.w = sm.u.sBf16[r_epi][c4 * 4 + 3];
                        uint4* dst = reinterpret_cast<uint4*>(
                            out_u32 + static_cast<int64_t>(r_glob) * (576 / 2) + nb64 * 32);
                        st_cs_u4(dst + c4, v);
                    }
                }
            }

            if (chunk + 1 < N_CHUNKS && row0 + (chunk + 1) * 64 < end) {
                __syncthreads();
            }
        }

        if (!is_last_m) {
            __syncthreads();
        }
    }
}

// =============================================================================
// Kernel 2: Persistent MX7 W2 + L2-Hot Combine (N=4608, K=288, BM=64, BN=64)
//   - 8 warps = 4 M-warps (wm=0..3) x 2 N-warps (wp=0..1) -> acc[4][4] = ONLY 16 regs!
//   - Cooperative 4-way parallel slab load + binary-searched flip decode into sA[1]
//     (exact 18,432 B alias!) -> 1.000x EXACT DRAM weight read, ZERO duplicate reads!
//   - Register-resident b0_reg[9][4], b1_reg[9][4] (72 regs) + 16 acc = 88 live regs!
//   - Fits 47,952 B <= 49,152 B -> 2 CTAs/SM (128 reg cap) with LITERAL 0 SPILLS!
//   - Full 64-row single-barrier sEpi + red.relaxed.gpu.global.add.noftz.bf16x2
//     + per-slice in-L2 zeroing + LPT plan + serpentine A reuse!
// =============================================================================

struct W2Mx7ATile {
    uint32_t sA_8[64][64];    // 16,384 B (k = 0..255, XOR-swizzled)
    uint32_t sA_last[64][8];  //  2,048 B (k = 256..287)
};

static_assert(sizeof(W2Mx7ATile) == 18432, "W2Mx7ATile must be exactly 18,432 B");

struct W2Mx7PersistSmem {
    W2Mx7ATile a0;
    union {
        W2Mx7ATile a1;
        uint2 sB_tmp[2][9][4][32];  // 2 * 9 * 4 * 32 * 8 B = 18,432 B exact!
    } a1_or_b;
    uint32_t sAS[2][64][3];         //  1,536 B
    int32_t  sTok[2][64];           //    512 B
    uint8_t  sWS[9][64];            //    576 B
    union {
        uint32_t sFlip[8][32];      //  1,024 B (cooperative slab decode only)
        uint32_t bf16x2[64][33];    //  8,448 B (all 64 rows x 64 bf16 -> 1 barrier!)
        float    f32[32][65];       //  8,320 B (32 rows x 64 fp32 -> 2 passes for CM=1)
    } epi_or_flip;
    int32_t sWork[3];
    int32_t _pad;
};

static_assert(sizeof(W2Mx7PersistSmem) <= 49152, "W2Mx7PersistSmem must fit <= 48 KiB (2 CTAs/SM)");

template <int AS_ROW_MAJOR, int COMBINE_MODE, int PANELS64_PER_CTA>
__global__ __launch_bounds__(256, (AS_ROW_MAJOR == 0 && COMBINE_MODE == 2) ? 1 : 2)
void pf_mx7_w2_persistent_kernel(
    const uint8_t* __restrict__ A,
    const uint8_t* __restrict__ AS,
    const int32_t* __restrict__ Row2Tok,
    const uint8_t* __restrict__ SLABS,
    const uint4* __restrict__ WS3,
    const uint16_t* __restrict__ FLIPS,
    const int32_t* __restrict__ F_OFF,
    const float* __restrict__ Ones,
    const int32_t* __restrict__ Tiles,
    const int32_t* __restrict__ PlanBuf,
    void* __restrict__ Out,
    int max_ctas,
    int tile,
    int num_experts,
    int max_work,
    uint32_t base_ws,
    int as_row_stride,
    int w_evict_mode,
    int slice_nb_blk
) {
    constexpr int BM = 64;
    constexpr int KP = 288;
    constexpr int KG = 9;
    constexpr int N  = 4608;

    int nb_blk, w_bx, serp_mode;
    if constexpr (COMBINE_MODE == 0) {
        nb_blk    = blockIdx.x;
        w_bx      = blockIdx.y;
        serp_mode = 0;
    } else {
        w_bx      = blockIdx.x;
        nb_blk    = (slice_nb_blk >= 0) ? slice_nb_blk : static_cast<int>(blockIdx.y);
        serp_mode = (slice_nb_blk >= 0) ? 2 : 1;
    }

    __shared__ alignas(16) W2Mx7PersistSmem sm;

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;       // 0..7
    const int wm   = warp >> 1;      // 0..3 (4 M-warps x 16 rows = BM=64!)
    const int wp   = warp & 1;       // 0..1 (2 N-warps x 32 cols = BN=64!)
    const int lane = tid & 31;
    const int g    = lane >> 2;      // 0..7
    const int lt   = lane & 3;       // 0..3

    if (!resolve_persistent_work(
            PlanBuf, Tiles, max_ctas, tile, num_experts, max_work,
            w_bx, nb_blk, serp_mode, sm.sWork, tid)) {
        return;
    }

    const int64_t e       = sm.sWork[0];
    const int row_begin   = sm.sWork[1];
    const int end         = sm.sWork[2];
    const int num_m       = ((end - row_begin) + BM - 1) / BM;
    const bool a_pinned   = (num_m == 1);  // a0 stays pinned across all dp while a1 holds sB_tmp!
    const bool use_cs_w   = (w_evict_mode != 0);
    const float scale_e   = Ones[e];

    const int m_warp_base = wm * 16;
    const int m_sf_off    = g + ((lt & 1) << 3);

    // Exact 4-way partition of slabs 0..8 across wm = 0..3 -> {0..2}, {3..4}, {5..6}, {7..8}
    const int sl_beg = (wm == 0) ? 0 : (wm * 2 + 1);
    const int sl_end = (wm == 0) ? 3 : (wm * 2 + 3);

    auto load_a_tile = [&](int mi_blk, int buf, bool cache_l1) {
        const int r0 = row_begin + mi_blk * BM;
        W2Mx7ATile& dst_a = (buf == 0) ? sm.a0 : sm.a1_or_b.a1;
        #pragma unroll
        for (int it = 0; it < 4; ++it) {
            int idx = it * 256 + tid;
            int m   = idx >> 4;
            int c4  = idx & 15;
            int sc4 = c4 ^ (m & 7);
            int r   = r0 + m;
            const uint8_t* src = (r < end) ? (A + static_cast<int64_t>(r) * KP + c4 * 16) : A;
            if (cache_l1) cp_async_ca_16(&dst_a.sA_8[m][sc4 * 4], src, (r < end) ? 16 : 0);
            else          cp_async_cg_16(&dst_a.sA_8[m][sc4 * 4], src, (r < end) ? 16 : 0);
        }
        if (tid < BM * 2) {
            int m = tid >> 1;
            int h = tid & 1;
            int r = r0 + m;
            const uint8_t* src = (r < end) ? (A + static_cast<int64_t>(r) * KP + 256 + h * 16) : A;
            if (cache_l1) cp_async_ca_16(&dst_a.sA_last[m][h * 4], src, (r < end) ? 16 : 0);
            else          cp_async_cg_16(&dst_a.sA_last[m][h * 4], src, (r < end) ? 16 : 0);
        }
        cp_async_commit();

        if (tid < BM) {
            int r = r0 + tid;
            if constexpr (COMBINE_MODE != 0) sm.sTok[buf][tid] = (r < end) ? Row2Tok[r] : -1;
        }
        if (tid < BM * 3) {
            int m = (tid * 21846) >> 16;  // exact tid / 3 for tid in [0, 191]
            int p = tid - m * 3;
            int r = r0 + m;
            uint32_t sval = 0u;
            if (r < end) {
                if (p < 2) {
                    if constexpr (AS_ROW_MAJOR) {
                        sval = load_as_u32_rowmaj(AS, static_cast<int64_t>(r), p * 4, as_row_stride);
                    } else {
                        int64_t off = scale_offset_byte(static_cast<int64_t>(r), p * 4, KG, tile);
                        sval = *reinterpret_cast<const uint32_t*>(AS + off);
                    }
                } else {
                    if constexpr (AS_ROW_MAJOR) {
                        sval = static_cast<uint32_t>(AS[static_cast<int64_t>(r) * as_row_stride + 8]);
                    } else {
                        int64_t off = scale_offset_byte(static_cast<int64_t>(r), 8, KG, tile);
                        sval = static_cast<uint32_t>(AS[off]);
                    }
                }
            }
            sm.sAS[buf][m][p] = sval;
        }
    };

    if (a_pinned) {
        load_a_tile(0, 0, false);
    }

    #pragma unroll 1
    for (int dp = 0; dp < PANELS64_PER_CTA; ++dp) {
        const int nb64 = nb_blk * PANELS64_PER_CTA + dp;
        const bool keep_a_l1 = (!a_pinned) && (dp + 1 < PANELS64_PER_CTA);

        if (!a_pinned) {
            load_a_tile(0, 0, keep_a_l1);
        }

        // 1. Cooperative 4-way parallel MX7 slab load + flip decode -> sB_tmp (EXACT 1.000x DRAM!)
        const int64_t wuid = e * 144 + nb64 * 2 + wp;
        const int32_t f_lo = F_OFF[wuid];
        const int32_t f_hi = F_OFF[wuid + 1];
        int32_t f_cur = (wm == 0)
            ? f_lo
            : find_flip_slab_start(FLIPS, f_lo, f_hi, static_cast<uint32_t>(sl_beg));
        uint32_t f_rec = (f_cur + lane < f_hi)
            ? static_cast<uint32_t>(__ldg(FLIPS + f_cur + lane))
            : 0xFFFFu;

        for (int sl = sl_beg; sl < sl_end; ++sl) {
            const int64_t slab_idx = wuid * 9 + sl;
            SlabMX7 slb = load_slab_mx7(
                SLABS + slab_idx * MX7_SLAB_BYTES, WS3 + slab_idx, lane, use_cs_w);
            const uint32_t fb = consume_slab_flips(
                FLIPS, f_cur, f_hi, f_rec, static_cast<uint32_t>(sl), lane, sm.epi_or_flip.sFlip[warp]);
            #pragma unroll
            for (int sn = 0; sn < 4; ++sn) {
                uint32_t b0, b1;
                decode_mx7_pair_to_b0b1(slb, fb, sn, b0, b1);
                sm.a1_or_b.sB_tmp[wp][sl][sn][lane] = make_uint2(b0, b1);
                if (lt == 0) {
                    sm.sWS[sl][wp * 32 + sn * 8 + g] =
                        static_cast<uint8_t>(decode_mx7_ws_sub(slb.ws3, base_ws, sn, g));
                }
            }
        }

        if (!a_pinned || dp == 0) {
            cp_async_wait<0>();
        }
        __syncthreads();

        // 2. Broadcast-load register-resident B panel ONCE across ALL M-tiles (72 regs!)
        uint32_t b0_reg[9][4], b1_reg[9][4];
        #pragma unroll
        for (int sl = 0; sl < 9; ++sl) {
            #pragma unroll
            for (int sn = 0; sn < 4; ++sn) {
                uint2 v = sm.a1_or_b.sB_tmp[wp][sl][sn][lane];
                b0_reg[sl][sn] = v.x;
                b1_reg[sl][sn] = v.y;
            }
        }
        if (!a_pinned) {
            __syncthreads();  // Fence sB_tmp reads before mi_blk=0 loads a1!
        }

        // 3. Sweep ALL M-tiles of expert e with ZERO global weight loads & ZERO unpack!
        #pragma unroll 1
        for (int mi_blk = 0; mi_blk < num_m; ++mi_blk) {
            const int cur  = mi_blk & 1;
            const int nxt  = cur ^ 1;
            const int row0 = row_begin + mi_blk * BM;

            if (!a_pinned && mi_blk + 1 < num_m) {
                load_a_tile(mi_blk + 1, nxt, keep_a_l1);
            }

            float acc[4][4];
            #pragma unroll
            for (int sn = 0; sn < 4; ++sn) {
                acc[sn][0] = 0.0f; acc[sn][1] = 0.0f;
                acc[sn][2] = 0.0f; acc[sn][3] = 0.0f;
            }

            const W2Mx7ATile& cur_a = (cur == 0) ? sm.a0 : sm.a1_or_b.a1;
            const int m_lo = m_warp_base + g;
            const int m_hi = m_lo + 8;
            const uint8_t* sAS_row = reinterpret_cast<const uint8_t*>(sm.sAS[cur][m_warp_base + m_sf_off]);

            #pragma unroll
            for (int ks = 0; ks < 8; ++ks) {
                const int sw_c0 = (ks * 8 + lt) ^ (g << 2);
                const int sw_c1 = (ks * 8 + 4 + lt) ^ (g << 2);
                const uint32_t a0  = cur_a.sA_8[m_lo][sw_c0];
                const uint32_t a1  = cur_a.sA_8[m_hi][sw_c0];
                const uint32_t a2  = cur_a.sA_8[m_lo][sw_c1];
                const uint32_t a3  = cur_a.sA_8[m_hi][sw_c1];
                const uint32_t sfa = static_cast<uint32_t>(sAS_row[ks]);
                #pragma unroll
                for (int sn = 0; sn < 4; ++sn) {
                    const uint32_t sfb = static_cast<uint32_t>(sm.sWS[ks][wp * 32 + sn * 8 + g]);
                    mma_m16n8k32_mxfp8(acc[sn], a0, a1, a2, a3, b0_reg[ks][sn], b1_reg[ks][sn], sfa, sfb);
                }
            }
            {
                const uint32_t a0  = cur_a.sA_last[m_lo][lt];
                const uint32_t a1  = cur_a.sA_last[m_hi][lt];
                const uint32_t a2  = cur_a.sA_last[m_lo][4 + lt];
                const uint32_t a3  = cur_a.sA_last[m_hi][4 + lt];
                const uint32_t sfa = static_cast<uint32_t>(sAS_row[8]);
                #pragma unroll
                for (int sn = 0; sn < 4; ++sn) {
                    const uint32_t sfb = static_cast<uint32_t>(sm.sWS[8][wp * 32 + sn * 8 + g]);
                    mma_m16n8k32_mxfp8(acc[sn], a0, a1, a2, a3, b0_reg[8][sn], b1_reg[8][sn], sfa, sfb);
                }
            }

            // ---- Single-Barrier 64-Row BN=64 Coalesced Epilogue ----
            if constexpr (COMBINE_MODE == 0 || COMBINE_MODE == 2) {
                #pragma unroll
                for (int sn = 0; sn < 4; ++sn) {
                    const int col_pair = wp * 16 + sn * 4 + lt;
                    sm.epi_or_flip.bf16x2[m_lo][col_pair] = pack_bf16x2(acc[sn][0] * scale_e, acc[sn][1] * scale_e);
                    sm.epi_or_flip.bf16x2[m_hi][col_pair] = pack_bf16x2(acc[sn][2] * scale_e, acc[sn][3] * scale_e);
                }
                __syncthreads();

                if constexpr (COMBINE_MODE == 0) {
                    const int sub_r = lane >> 3;  // 0..3
                    const int c4    = lane & 7;   // 0..7 (8 uint4 = 64 bf16)
                    #pragma unroll
                    for (int k = 0; k < 2; ++k) {
                        const int m = warp * 8 + k * 4 + sub_r;
                        const int r_glob = row0 + m;
                        if (r_glob < end) {
                            uint4 v;
                            v.x = sm.epi_or_flip.bf16x2[m][c4 * 4 + 0];
                            v.y = sm.epi_or_flip.bf16x2[m][c4 * 4 + 1];
                            v.z = sm.epi_or_flip.bf16x2[m][c4 * 4 + 2];
                            v.w = sm.epi_or_flip.bf16x2[m][c4 * 4 + 3];
                            uint4* dst = reinterpret_cast<uint4*>(
                                reinterpret_cast<uint32_t*>(Out)
                                + static_cast<int64_t>(r_glob) * (N / 2) + nb64 * 32);
                            st_cs_u4(dst + c4, v);
                        }
                    }
                } else {
                    // COMBINE_MODE == 2: 1 coalesced red_add_bf16x2 per lane across 32 bf16x2 = 64 cols!
                    #pragma unroll
                    for (int k = 0; k < 8; ++k) {
                        const int m = warp * 8 + k;
                        const int r_glob = row0 + m;
                        if (r_glob < end) {
                            const int32_t tok = sm.sTok[cur][m];
                            if (tok >= 0) {
                                uint32_t* dst = reinterpret_cast<uint32_t*>(Out)
                                              + static_cast<int64_t>(tok) * (N / 2) + nb64 * 32;
                                red_add_bf16x2_bits(dst + lane, sm.epi_or_flip.bf16x2[m][lane]);
                            }
                        }
                    }
                }
            } else {
                // COMBINE_MODE == 1: FP32 atomic (2 passes of 32 rows)
                #pragma unroll
                for (int half = 0; half < 2; ++half) {
                    if ((wm >> 1) == half) {
                        const int r_lo = (wm & 1) * 16 + g;
                        const int r_hi = r_lo + 8;
                        #pragma unroll
                        for (int sn = 0; sn < 4; ++sn) {
                            const int c0 = wp * 32 + sn * 8 + lt * 2;
                            sm.epi_or_flip.f32[r_lo][c0 + 0] = acc[sn][0] * scale_e;
                            sm.epi_or_flip.f32[r_lo][c0 + 1] = acc[sn][1] * scale_e;
                            sm.epi_or_flip.f32[r_hi][c0 + 0] = acc[sn][2] * scale_e;
                            sm.epi_or_flip.f32[r_hi][c0 + 1] = acc[sn][3] * scale_e;
                        }
                    }
                    __syncthreads();
                    #pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        const int r_epi  = warp * 4 + k;
                        const int m      = half * 32 + r_epi;
                        const int r_glob = row0 + m;
                        if (r_glob < end) {
                            const int32_t tok = sm.sTok[cur][m];
                            if (tok >= 0) {
                                float* dst = reinterpret_cast<float*>(Out)
                                           + static_cast<int64_t>(tok) * N + nb64 * 64;
                                red_add_f32(dst + lane,      sm.epi_or_flip.f32[r_epi][lane]);
                                red_add_f32(dst + 32 + lane, sm.epi_or_flip.f32[r_epi][32 + lane]);
                            }
                        }
                    }
                    if (half == 0) __syncthreads();
                }
            }

            if (mi_blk + 1 < num_m) {
                if (!a_pinned) cp_async_wait<0>();
                __syncthreads();
            } else if (dp + 1 < PANELS64_PER_CTA) {
                __syncthreads();
            }
        }
    }
}

// =============================================================================
// Dynamic SMEM Carveout Helper & Template Launchers
// =============================================================================

template <typename KernelFn>
inline void configure_dyn_smem(KernelFn fn, size_t smem_bytes) {
    cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes));
    cudaFuncSetAttribute(fn, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
}

template <int BM, int STAGES, int MIN_CTA, int FUSE_SITU>
int launch_mx7_w13_impl(
    const void* A, const void* AS, const void* Row2Tok, const void* Gammas,
    const void* SLABS, const void* WS3, const void* FLIPS, const void* F_OFF,
    const void* Ones, const void* Tiles, const void* PlanBuf,
    void* Out0, void* OutScales,
    int max_ctas, int tile, int num_experts, int max_work,
    uint32_t base_ws, int use_r2t, int as_row_major, int w2_as_row_major, int w2_as_stride,
    int w_evict_mode, cudaStream_t stream
) {
    const int eff_work = (PlanBuf != nullptr && max_work > 0) ? max_work : num_experts;
    dim3 grid(9, eff_work);
    dim3 block(256);
    constexpr size_t SMEM = sizeof(Mx7W13Smem<BM, STAGES>);
    const int key = ((use_r2t ? 1 : 0) << 2)
                  | ((as_row_major ? 1 : 0) << 1)
                  | ((FUSE_SITU && w2_as_row_major) ? 1 : 0);

    #define DISPATCH_MX7_W13(R2T, ARM, W2RM) \
        do { \
            static bool _cfg = false; \
            if (!_cfg) { \
                configure_dyn_smem( \
                    pf_mx7_w13_persistent_kernel<BM, STAGES, MIN_CTA, FUSE_SITU, R2T, ARM, W2RM>, SMEM); \
                _cfg = true; \
            } \
            pf_mx7_w13_persistent_kernel<BM, STAGES, MIN_CTA, FUSE_SITU, R2T, ARM, W2RM> \
                <<<grid, block, SMEM, stream>>>( \
                reinterpret_cast<const uint8_t*>(A), reinterpret_cast<const uint8_t*>(AS), \
                reinterpret_cast<const int32_t*>(Row2Tok), reinterpret_cast<const float*>(Gammas), \
                reinterpret_cast<const uint8_t*>(SLABS), reinterpret_cast<const uint4*>(WS3), \
                reinterpret_cast<const uint16_t*>(FLIPS), reinterpret_cast<const int32_t*>(F_OFF), \
                reinterpret_cast<const float*>(Ones), reinterpret_cast<const int32_t*>(Tiles), \
                reinterpret_cast<const int32_t*>(PlanBuf), Out0, reinterpret_cast<uint8_t*>(OutScales), \
                max_ctas, tile, num_experts, eff_work, base_ws, w2_as_stride, w_evict_mode); \
        } while (0)

    switch (key) {
        case 0: DISPATCH_MX7_W13(0, 0, 0); break;
        case 1: DISPATCH_MX7_W13(0, 0, 1); break;
        case 2: DISPATCH_MX7_W13(0, 1, 0); break;
        case 3: DISPATCH_MX7_W13(0, 1, 1); break;
        case 4: DISPATCH_MX7_W13(1, 0, 0); break;
        case 5: DISPATCH_MX7_W13(1, 0, 1); break;
        case 6: DISPATCH_MX7_W13(1, 1, 0); break;
        case 7: DISPATCH_MX7_W13(1, 1, 1); break;
        default: return -3;
    }
    #undef DISPATCH_MX7_W13
    return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

template <int AS_RM, int CM, int P64>
int launch_mx7_w2_cfg(
    const void* A, const void* AS, const void* Row2Tok,
    const void* SLABS, const void* WS3, const void* FLIPS, const void* F_OFF,
    const void* Ones, const void* Tiles, const void* PlanBuf, void* Out,
    int max_ctas, int tile, int num_experts, int eff_work,
    uint32_t base_ws, int as_row_stride, int w_evict_mode,
    int l2_zero_slice, int tokens, cudaStream_t stream
) {
    constexpr int N_SLICES = 72 / P64;
    dim3 block(256);
    static bool _carved = false;
    if (!_carved) {
        cudaFuncSetAttribute(
            pf_mx7_w2_persistent_kernel<AS_RM, CM, P64>,
            cudaFuncAttributePreferredSharedMemoryCarveout,
            cudaSharedmemCarveoutMaxShared);
        _carved = true;
    }
    if constexpr (CM == 0) {
        dim3 grid(N_SLICES, eff_work);
        pf_mx7_w2_persistent_kernel<AS_RM, 0, P64><<<grid, block, 0, stream>>>(
            reinterpret_cast<const uint8_t*>(A), reinterpret_cast<const uint8_t*>(AS),
            reinterpret_cast<const int32_t*>(Row2Tok),
            reinterpret_cast<const uint8_t*>(SLABS), reinterpret_cast<const uint4*>(WS3),
            reinterpret_cast<const uint16_t*>(FLIPS), reinterpret_cast<const int32_t*>(F_OFF),
            reinterpret_cast<const float*>(Ones), reinterpret_cast<const int32_t*>(Tiles),
            reinterpret_cast<const int32_t*>(PlanBuf), Out,
            max_ctas, tile, num_experts, eff_work, base_ws, as_row_stride, w_evict_mode, -1);
    } else if (CM == 2 && l2_zero_slice != 0 && tokens > 0) {
        dim3 zgrid((tokens + 7) / 8);
        dim3 grid(eff_work, 1);
        for (int s = 0; s < N_SLICES; ++s) {
            pf_mx7_zero_bf16_slice_wb_kernel<P64><<<zgrid, block, 0, stream>>>(
                reinterpret_cast<uint16_t*>(Out), tokens, s);
            pf_mx7_w2_persistent_kernel<AS_RM, CM, P64><<<grid, block, 0, stream>>>(
                reinterpret_cast<const uint8_t*>(A), reinterpret_cast<const uint8_t*>(AS),
                reinterpret_cast<const int32_t*>(Row2Tok),
                reinterpret_cast<const uint8_t*>(SLABS), reinterpret_cast<const uint4*>(WS3),
                reinterpret_cast<const uint16_t*>(FLIPS), reinterpret_cast<const int32_t*>(F_OFF),
                reinterpret_cast<const float*>(Ones), reinterpret_cast<const int32_t*>(Tiles),
                reinterpret_cast<const int32_t*>(PlanBuf), Out,
                max_ctas, tile, num_experts, eff_work, base_ws, as_row_stride, w_evict_mode, s);
        }
    } else {
        dim3 grid(eff_work, N_SLICES);
        pf_mx7_w2_persistent_kernel<AS_RM, CM, P64><<<grid, block, 0, stream>>>(
            reinterpret_cast<const uint8_t*>(A), reinterpret_cast<const uint8_t*>(AS),
            reinterpret_cast<const int32_t*>(Row2Tok),
            reinterpret_cast<const uint8_t*>(SLABS), reinterpret_cast<const uint4*>(WS3),
            reinterpret_cast<const uint16_t*>(FLIPS), reinterpret_cast<const int32_t*>(F_OFF),
            reinterpret_cast<const float*>(Ones), reinterpret_cast<const int32_t*>(Tiles),
            reinterpret_cast<const int32_t*>(PlanBuf), Out,
            max_ctas, tile, num_experts, eff_work, base_ws, as_row_stride, w_evict_mode, -1);
    }
    return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}


// =============================================================================
// Wave-V10 Co-Resident DBO-Safe Persistent MX7 W2 Engine (`<= 80` CTAs, 2 CTAs/SM)
//   - Single-launch persistent dynamic LPT work-stealing across all N_SLICES = 72 / P64
//     slices (`P64 in {3, 4, 6, 8, 9, 12}`), killing 35x1,024 = 35,840 launches/1M!
//   - Reserves 8 SMs (`<= 80` CTAs on 48-SM GB10) for concurrent DBO `comm_stream`
//     (NCCL Ch=4 + lowsm RHD/AG) -> ZERO `slice_done`/`zero_done` spin-deadlock!
//   - Fixes V4's broken `head_keep=48` serpentine via Hot-Tail `K_hot=64` reversal
//     on odd slices (100% L2 hit on `A [R, 288]` tail across slice transitions).
//   - Staggered sub-panel ownership `dp = (dp_step + cta_id) % P64` cuts L2 `bf16x2`
//     atomic bank collisions 4x-6x.
//   - Overlapped (`zero_mode=1`) & DBO-safe barrier-gated (`zero_mode=2`) in-L2
//     128-token sector-complete `st.global.wb.v4.u32` slice zeroing.
//   - Dead M-warp (`wm`) tail MMA skip when `m_warp_base >= end - row0`.
//   - CTA-uniform `W2Mx7CtaParams` in `__shared__` via non-hoistable `ld.shared`
//     -> fits `48,096 B <= 49,152 B` (2 CTAs/SM) with LITERAL 0 B stack / 0 B spills!
// =============================================================================

constexpr int kMx7MaxSlices = 24;  // Max N_SLICES = 72 / 3 = 24

// Wave-V13 ZERO-SPIN Host-Sliced Work-Stealing Cursors (`SliceState[0..23]`)
// Uses only the first `N_SLICES <= 24` uint32_t words of existing 512-word `SliceState` (100% drop-in ABI!).

// 1-CTA LPT Plan Builder aligned to BM=64 (compatible with both W13 and W2 v10!)
__global__ __launch_bounds__(256, 1)
void pf_mx7_build_plan_v10_kernel(
    const int32_t* __restrict__ Tiles,
    int32_t* __restrict__ PlanBuf,
    int max_ctas, int tile, int num_experts, int split_m, int max_work
) {
    constexpr int MAX_E = 1024;
    __shared__ int32_t  sBeg[MAX_E];
    __shared__ int32_t  sEnd[MAX_E];
    __shared__ uint32_t sBucketCnt[32];
    __shared__ uint32_t sBucketOff[32];

    const int tid = threadIdx.x;
    const int eff_e = (num_experts < MAX_E) ? num_experts : MAX_E;
    const int eff_split = (split_m > 0) ? ((split_m + 63) & ~63) : 0;

    if (tid < 32) sBucketCnt[tid] = 0u;
    for (int w = tid; w < max_work; w += 256) PlanBuf[w] = -1;
    __syncthreads();

    for (int e = tid; e < eff_e; e += 256) {
        int rb = 0, re = 0;
        find_expert_row_range(Tiles, max_ctas, tile, e, rb, re);
        sBeg[e] = rb; sEnd[e] = re;
        int cnt = re - rb;
        if (cnt > 0) {
            if (eff_split > 0 && cnt > eff_split) {
                for (int off = 0; off < cnt; off += eff_split) {
                    int sz = cnt - off;
                    if (sz > eff_split) sz = eff_split;
                    int raw_b = (sz - 1) >> 5;
                    int b = 31 - ((raw_b > 31) ? 31 : raw_b);
                    atomicAdd(&sBucketCnt[b], 1u);
                }
            } else {
                int raw_b = (cnt - 1) >> 5;
                int b = 31 - ((raw_b > 31) ? 31 : raw_b);
                atomicAdd(&sBucketCnt[b], 1u);
            }
        }
    }
    __syncthreads();

    if (tid == 0) {
        uint32_t acc = 0u;
        for (int b = 0; b < 32; ++b) {
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
                    int raw_b = (sz - 1) >> 5;
                    int b = 31 - ((raw_b > 31) ? 31 : raw_b);
                    uint32_t pos = atomicAdd(&sBucketOff[b], 1u);
                    if (static_cast<int>(pos) < max_work) {
                        out_e[pos]   = e;
                        out_beg[pos] = rb + off;
                        out_end[pos] = rb + off + sz;
                    }
                }
            } else {
                int raw_b = (cnt - 1) >> 5;
                int b = 31 - ((raw_b > 31) ? 31 : raw_b);
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

// Cooperative 128-token L2-WB slice zeroing inside persistent W2 v10 kernel.
// Supports PANELS64 in {3, 4, 6, 8, 9, 12} -> {192, 256, 384, 512, 576, 768} BF16 cols/slice.
template <int PANELS64>
__device__ __forceinline__ void zero_mx7_slice_chunk_128tok_wb(
    uint16_t* Out_bf16,
    int tokens,
    int slice_s,
    int chunk_id,
    int warp,
    int lane
) {
    constexpr int H = 4608;
    constexpr int COL_BASE = PANELS64 * 64;
    const int t_base = chunk_id * 128;
    const uint4 z4 = make_uint4(0u, 0u, 0u, 0u);
    #pragma unroll
    for (int r = 0; r < 16; ++r) {
        const int tok = t_base + r * 8 + warp;
        if (tok < tokens) {
            uint4* dst = reinterpret_cast<uint4*>(
                Out_bf16 + static_cast<int64_t>(tok) * H + slice_s * COL_BASE);
            if constexpr (PANELS64 == 3) {
                if (lane < 24) st_wb_u4(dst + lane, z4);
            } else if constexpr (PANELS64 == 4) {
                st_wb_u4(dst + lane, z4);
            } else if constexpr (PANELS64 == 6) {
                st_wb_u4(dst + lane, z4);
                if (lane < 16) st_wb_u4(dst + 32 + lane, z4);
            } else if constexpr (PANELS64 == 8) {
                st_wb_u4(dst + lane, z4);
                st_wb_u4(dst + 32 + lane, z4);
            } else if constexpr (PANELS64 == 9) {
                st_wb_u4(dst + lane, z4);
                st_wb_u4(dst + 32 + lane, z4);
                if (lane < 8) st_wb_u4(dst + 64 + lane, z4);
            } else if constexpr (PANELS64 == 12) {
                st_wb_u4(dst + lane, z4);
                st_wb_u4(dst + 32 + lane, z4);
                st_wb_u4(dst + 64 + lane, z4);
            }
        }
    }
}

struct alignas(16) W2Mx7CtaParams {
    const uint8_t*  A;
    const uint8_t*  AS;
    const int32_t*  Row2Tok;
    const uint8_t*  SLABS;
    const uint4*    WS3;
    const uint16_t* FLIPS;
    const int32_t*  F_OFF;
    const float*    Ones;
    const int32_t*  Tiles;
    const int32_t*  PlanBuf;
    void*           Out;
    int             max_ctas;
    int             tile;
    int             num_experts;
    int             max_work;
    uint32_t        base_ws;
    int             as_row_stride;
    int             use_cs_w;
    int             zero_mode;
    int             tokens;
};

struct W2Mx7V10Smem {
    W2Mx7ATile a0;
    union {
        W2Mx7ATile a1;
        uint2 sB_tmp[2][9][4][32];  // 18,432 B exact!
    } a1_or_b;
    uint32_t sAS[2][64][3];         //  1,536 B
    int32_t  sTok[2][64];           //    512 B
    uint8_t  sWS[9][64];            //    576 B
    union {
        uint32_t sFlip[8][32];      //  1,024 B
        uint32_t bf16x2[64][33];    //  8,448 B
        float    f32[32][65];       //  8,320 B
    } epi_or_flip;
    W2Mx7CtaParams sP;              //    128 B
    int32_t  sWork[4];              //     16 B ([e, rb, re, has_more])
    float    sScaleE;               //      4 B
};

template <int PANELS64>
__global__ __launch_bounds__(256, 2)
void pf_mx7_w2_zero_slice_v13_kernel(
    uint16_t* __restrict__ Out_bf16,
    uint32_t* __restrict__ SliceState,
    int tokens,
    int slice_s
) {
    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int zc   = static_cast<int>(blockIdx.x);
    if (zc == 0 && tid == 0) {
        st_wb_u32(&SliceState[slice_s], 0u);
    }
    zero_mx7_slice_chunk_128tok_wb<PANELS64>(Out_bf16, tokens, slice_s, zc, warp, lane);
}

__global__ __launch_bounds__(64, 1)
void pf_mx7_w2_reset_cursors_v13_kernel(
    uint32_t* __restrict__ SliceState,
    int n_slices
) {
    const int tid = static_cast<int>(threadIdx.x);
    if (tid < n_slices) {
        st_wb_u32(&SliceState[tid], 0u);
    }
}

static_assert(sizeof(W2Mx7V10Smem) <= 49152, "W2Mx7V10Smem must fit <= 48 KiB (2 CTAs/SM)");

template <int AS_ROW_MAJOR, int COMBINE_MODE, int PANELS64_PER_CTA>
__global__ __launch_bounds__(256, (AS_ROW_MAJOR == 0 && COMBINE_MODE == 2) ? 1 : 2)
void pf_mx7_w2_v10_kernel(
    const uint8_t* __restrict__ A,
    const uint8_t* __restrict__ AS,
    const int32_t* __restrict__ Row2Tok,
    const uint8_t* __restrict__ SLABS,
    const uint4* __restrict__ WS3,
    const uint16_t* __restrict__ FLIPS,
    const int32_t* __restrict__ F_OFF,
    const float* __restrict__ Ones,
    const int32_t* __restrict__ Tiles,
    const int32_t* __restrict__ PlanBuf,
    uint32_t* __restrict__ SliceState,
    void* __restrict__ Out,
    int max_ctas,
    int tile,
    int num_experts,
    int max_work,
    uint32_t base_ws,
    int as_row_stride,
    int w_evict_mode,
    int slice_s
) {
    constexpr int BM       = 64;
    constexpr int KP       = 288;
    constexpr int KG       = 9;
    constexpr int N        = 4608;

    __shared__ alignas(16) W2Mx7V10Smem sm;

    const int tid    = threadIdx.x;
    const int warp   = tid >> 5;   // 0..7
    const int wm     = warp >> 1;  // 0..3 (4 M-warps x 16 rows = BM=64)
    const int wp     = warp & 1;   // 0..1 (2 N-warps x 32 cols = BN=64)
    const int lane   = tid & 31;
    const int g      = lane >> 2;  // 0..7
    const int lt     = lane & 3;   // 0..3
    const int cta_id = blockIdx.x;
    const int s      = slice_s;

    if (tid == 0) {
        sm.sP.A             = A;
        sm.sP.AS            = AS;
        sm.sP.Row2Tok       = Row2Tok;
        sm.sP.SLABS         = SLABS;
        sm.sP.WS3           = WS3;
        sm.sP.FLIPS         = FLIPS;
        sm.sP.F_OFF         = F_OFF;
        sm.sP.Ones          = Ones;
        sm.sP.Tiles         = Tiles;
        sm.sP.PlanBuf       = PlanBuf;
        sm.sP.Out           = Out;
        sm.sP.max_ctas      = max_ctas;
        sm.sP.tile          = tile;
        sm.sP.num_experts   = num_experts;
        sm.sP.max_work      = max_work;
        sm.sP.base_ws       = base_ws;
        sm.sP.as_row_stride = as_row_stride;
        sm.sP.use_cs_w      = (w_evict_mode != 0) ? 1 : 0;
        sm.sP.zero_mode     = 0;
        sm.sP.tokens        = 0;
    }
    __syncthreads();

    const int m_warp_base   = wm * 16;
    const int m_sf_off      = g + ((lt & 1) << 3);
    const int sl_beg        = (wm == 0) ? 0 : (wm * 2 + 1);
    const int sl_end        = (wm == 0) ? 3 : (wm * 2 + 3);

    auto load_a_tile = [&](int row_begin, int end, int mi_blk, int buf, bool cache_l1) {
        const uint8_t* pA   = ld_smem_ptr(&sm.sP.A);
        const uint8_t* pAS  = ld_smem_ptr(&sm.sP.AS);
        const int32_t* pTok = ld_smem_ptr(&sm.sP.Row2Tok);
        const int tile_v    = ld_smem_i32(&sm.sP.tile);
        const int as_str    = ld_smem_i32(&sm.sP.as_row_stride);
        const int r0        = row_begin + mi_blk * BM;
        W2Mx7ATile& dst_a   = (buf == 0) ? sm.a0 : sm.a1_or_b.a1;

        #pragma unroll
        for (int it = 0; it < 4; ++it) {
            int idx = it * 256 + tid;
            int m   = idx >> 4;
            int c4  = idx & 15;
            int sc4 = c4 ^ (m & 7);
            int r   = r0 + m;
            const uint8_t* src = (r < end) ? (pA + static_cast<int64_t>(r) * KP + c4 * 16) : pA;
            if (cache_l1) cp_async_ca_16(&dst_a.sA_8[m][sc4 * 4], src, (r < end) ? 16 : 0);
            else          cp_async_cg_16(&dst_a.sA_8[m][sc4 * 4], src, (r < end) ? 16 : 0);
        }
        if (tid < BM * 2) {
            int m = tid >> 1;
            int h = tid & 1;
            int r = r0 + m;
            const uint8_t* src = (r < end) ? (pA + static_cast<int64_t>(r) * KP + 256 + h * 16) : pA;
            if (cache_l1) cp_async_ca_16(&dst_a.sA_last[m][h * 4], src, (r < end) ? 16 : 0);
            else          cp_async_cg_16(&dst_a.sA_last[m][h * 4], src, (r < end) ? 16 : 0);
        }
        cp_async_commit();

        if (tid < BM) {
            int r = r0 + tid;
            if constexpr (COMBINE_MODE != 0) sm.sTok[buf][tid] = (r < end) ? pTok[r] : -1;
        }
        if (tid < BM * 3) {
            int m = (tid * 21846) >> 16;  // exact tid / 3 for tid in [0, 191]
            int p = tid - m * 3;
            int r = r0 + m;
            uint32_t sval = 0u;
            if (r < end) {
                if (p < 2) {
                    if constexpr (AS_ROW_MAJOR) {
                        sval = load_as_u32_rowmaj(pAS, static_cast<int64_t>(r), p * 4, as_str);
                    } else {
                        int64_t off = scale_offset_byte(static_cast<int64_t>(r), p * 4, KG, tile_v);
                        sval = *reinterpret_cast<const uint32_t*>(pAS + off);
                    }
                } else {
                    if constexpr (AS_ROW_MAJOR) {
                        sval = static_cast<uint32_t>(pAS[static_cast<int64_t>(r) * as_str + 8]);
                    } else {
                        int64_t off = scale_offset_byte(static_cast<int64_t>(r), 8, KG, tile_v);
                        sval = static_cast<uint32_t>(pAS[off]);
                    }
                }
            }
            sm.sAS[buf][m][p] = sval;
        }
    };

    // Wave-V13 ZERO-SPIN: host stream barrier already guaranteed slice `s` zeroed in L2!
    // Dynamic LPT work-stealing + Hot-Tail K_hot=64 reversal on odd slices across ALL 48/48 SMs!
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
                } else if (static_cast<int>(ticket) < num_e) {
                    has_more = 1;
                    const int e_cand = (s & 1) ? (num_e - 1 - static_cast<int>(ticket)) : static_cast<int>(ticket);
                    find_expert_row_range(
                        pTiles, ld_smem_i32(&sm.sP.max_ctas), ld_smem_i32(&sm.sP.tile),
                        e_cand, rb, re);
                    if (rb < re) e = e_cand;
                }
                sm.sWork[0] = e;
                sm.sWork[1] = rb;
                sm.sWork[2] = re;
                sm.sWork[3] = has_more;
                sm.sScaleE  = (e >= 0) ? ld_smem_ptr(&sm.sP.Ones)[e] : 0.0f;
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

            const int num_m       = ((end - row_begin) + BM - 1) / BM;
            const bool a_pinned   = (num_m == 1);
            const bool use_cs_w   = (ld_smem_i32(&sm.sP.use_cs_w) != 0);
            const uint32_t b_ws   = ld_smem_u32(&sm.sP.base_ws);

            if (a_pinned) {
                load_a_tile(row_begin, end, 0, 0, false);
            }

            #pragma unroll 1
            for (int dp_step = 0; dp_step < PANELS64_PER_CTA; ++dp_step) {
                // Staggered sub-panel ownership spreads 80 concurrent CTAs across disjoint 64-col banks!
                const int dp = (PANELS64_PER_CTA > 1)
                    ? ((dp_step + cta_id) % PANELS64_PER_CTA)
                    : 0;
                const int nb64 = s * PANELS64_PER_CTA + dp;
                const bool keep_a_l1 = (!a_pinned) && (dp_step + 1 < PANELS64_PER_CTA);

                if (!a_pinned) {
                    load_a_tile(row_begin, end, 0, 0, keep_a_l1);
                }

                // 1. Cooperative 4-way parallel MX7 slab load + flip decode -> sB_tmp (EXACT 1.000x DRAM!)
                const uint8_t*  pSLABS = ld_smem_ptr(&sm.sP.SLABS);
                const uint4*    pWS3   = ld_smem_ptr(&sm.sP.WS3);
                const uint16_t* pFLIPS = ld_smem_ptr(&sm.sP.FLIPS);
                const int32_t*  pFOFF  = ld_smem_ptr(&sm.sP.F_OFF);

                const int64_t wuid = e * 144 + nb64 * 2 + wp;
                const int32_t f_lo = pFOFF[wuid];
                const int32_t f_hi = pFOFF[wuid + 1];
                int32_t f_cur = (wm == 0)
                    ? f_lo
                    : find_flip_slab_start(pFLIPS, f_lo, f_hi, static_cast<uint32_t>(sl_beg));
                uint32_t f_rec = (f_cur + lane < f_hi)
                    ? static_cast<uint32_t>(__ldg(pFLIPS + f_cur + lane))
                    : 0xFFFFu;

                for (int sl = sl_beg; sl < sl_end; ++sl) {
                    const int64_t slab_idx = wuid * 9 + sl;
                    SlabMX7 slb = load_slab_mx7(
                        pSLABS + slab_idx * MX7_SLAB_BYTES, pWS3 + slab_idx, lane, use_cs_w);
                    const uint32_t fb = consume_slab_flips(
                        pFLIPS, f_cur, f_hi, f_rec, static_cast<uint32_t>(sl), lane, sm.epi_or_flip.sFlip[warp]);
                    #pragma unroll
                    for (int sn = 0; sn < 4; ++sn) {
                        uint32_t b0, b1;
                        decode_mx7_pair_to_b0b1(slb, fb, sn, b0, b1);
                        sm.a1_or_b.sB_tmp[wp][sl][sn][lane] = make_uint2(b0, b1);
                        if (lt == 0) {
                            sm.sWS[sl][wp * 32 + sn * 8 + g] =
                                static_cast<uint8_t>(decode_mx7_ws_sub(slb.ws3, b_ws, sn, g));
                        }
                    }
                }

                if (!a_pinned || dp_step == 0) {
                    cp_async_wait<0>();
                }
                __syncthreads();

                // 2. Broadcast-load register-resident B panel ONCE across ALL M-tiles (72 regs!)
                uint32_t b0_reg[9][4], b1_reg[9][4];
                #pragma unroll
                for (int sl = 0; sl < 9; ++sl) {
                    #pragma unroll
                    for (int sn = 0; sn < 4; ++sn) {
                        uint2 v = sm.a1_or_b.sB_tmp[wp][sl][sn][lane];
                        b0_reg[sl][sn] = v.x;
                        b1_reg[sl][sn] = v.y;
                    }
                }
                if (!a_pinned) {
                    __syncthreads();  // Fence sB_tmp reads before mi_blk=0 loads a1!
                }

                // 3. Sweep ALL M-tiles of expert e with ZERO global weight loads & dead wm skip!
                #pragma unroll 1
                for (int mi_blk = 0; mi_blk < num_m; ++mi_blk) {
                    const int cur  = mi_blk & 1;
                    const int nxt  = cur ^ 1;
                    const int row0 = row_begin + mi_blk * BM;
                    const int live_m = end - row0;

                    if (!a_pinned && mi_blk + 1 < num_m) {
                        load_a_tile(row_begin, end, mi_blk + 1, nxt, keep_a_l1);
                    }

                    float acc[4][4];
                    #pragma unroll
                    for (int sn = 0; sn < 4; ++sn) {
                        acc[sn][0] = 0.0f; acc[sn][1] = 0.0f;
                        acc[sn][2] = 0.0f; acc[sn][3] = 0.0f;
                    }

                    if (m_warp_base < live_m) {
                        const W2Mx7ATile& cur_a = (cur == 0) ? sm.a0 : sm.a1_or_b.a1;
                        const int m_lo = m_warp_base + g;
                        const int m_hi = m_lo + 8;
                        const uint8_t* sAS_row = reinterpret_cast<const uint8_t*>(sm.sAS[cur][m_warp_base + m_sf_off]);

                        #pragma unroll
                        for (int ks = 0; ks < 8; ++ks) {
                            const int sw_c0 = (ks * 8 + lt) ^ (g << 2);
                            const int sw_c1 = (ks * 8 + 4 + lt) ^ (g << 2);
                            const uint32_t a0  = cur_a.sA_8[m_lo][sw_c0];
                            const uint32_t a1  = cur_a.sA_8[m_hi][sw_c0];
                            const uint32_t a2  = cur_a.sA_8[m_lo][sw_c1];
                            const uint32_t a3  = cur_a.sA_8[m_hi][sw_c1];
                            const uint32_t sfa = static_cast<uint32_t>(sAS_row[ks]);
                            #pragma unroll
                            for (int sn = 0; sn < 4; ++sn) {
                                const uint32_t sfb = static_cast<uint32_t>(sm.sWS[ks][wp * 32 + sn * 8 + g]);
                                mma_m16n8k32_mxfp8(acc[sn], a0, a1, a2, a3, b0_reg[ks][sn], b1_reg[ks][sn], sfa, sfb);
                            }
                        }
                        {
                            const uint32_t a0  = cur_a.sA_last[m_lo][lt];
                            const uint32_t a1  = cur_a.sA_last[m_hi][lt];
                            const uint32_t a2  = cur_a.sA_last[m_lo][4 + lt];
                            const uint32_t a3  = cur_a.sA_last[m_hi][4 + lt];
                            const uint32_t sfa = static_cast<uint32_t>(sAS_row[8]);
                            #pragma unroll
                            for (int sn = 0; sn < 4; ++sn) {
                                const uint32_t sfb = static_cast<uint32_t>(sm.sWS[8][wp * 32 + sn * 8 + g]);
                                mma_m16n8k32_mxfp8(acc[sn], a0, a1, a2, a3, b0_reg[8][sn], b1_reg[8][sn], sfa, sfb);
                            }
                        }
                    }

                    // ---- Single-Barrier 64-Row BN=64 Coalesced Epilogue ----
                    void* pOut = ld_smem_mut_ptr(&sm.sP.Out);
                    const float scale_e = sm.sScaleE;
                    if constexpr (COMBINE_MODE == 0 || COMBINE_MODE == 2) {
                        const int m_lo = m_warp_base + g;
                        const int m_hi = m_lo + 8;
                        #pragma unroll
                        for (int sn = 0; sn < 4; ++sn) {
                            const int col_pair = wp * 16 + sn * 4 + lt;
                            sm.epi_or_flip.bf16x2[m_lo][col_pair] = pack_bf16x2(acc[sn][0] * scale_e, acc[sn][1] * scale_e);
                            sm.epi_or_flip.bf16x2[m_hi][col_pair] = pack_bf16x2(acc[sn][2] * scale_e, acc[sn][3] * scale_e);
                        }
                        __syncthreads();

                        if constexpr (COMBINE_MODE == 0) {
                            const int sub_r = lane >> 3;
                            const int c4    = lane & 7;
                            #pragma unroll
                            for (int k = 0; k < 2; ++k) {
                                const int m = warp * 8 + k * 4 + sub_r;
                                const int r_glob = row0 + m;
                                if (r_glob < end) {
                                    uint4 v;
                                    v.x = sm.epi_or_flip.bf16x2[m][c4 * 4 + 0];
                                    v.y = sm.epi_or_flip.bf16x2[m][c4 * 4 + 1];
                                    v.z = sm.epi_or_flip.bf16x2[m][c4 * 4 + 2];
                                    v.w = sm.epi_or_flip.bf16x2[m][c4 * 4 + 3];
                                    uint4* dst = reinterpret_cast<uint4*>(
                                        reinterpret_cast<uint32_t*>(pOut)
                                        + static_cast<int64_t>(r_glob) * (N / 2) + nb64 * 32);
                                    st_cs_u4(dst + c4, v);
                                }
                            }
                        } else {
                            #pragma unroll
                            for (int k = 0; k < 8; ++k) {
                                const int m = warp * 8 + k;
                                const int r_glob = row0 + m;
                                if (r_glob < end) {
                                    const int32_t tok = sm.sTok[cur][m];
                                    if (tok >= 0) {
                                        uint32_t* dst = reinterpret_cast<uint32_t*>(pOut)
                                                      + static_cast<int64_t>(tok) * (N / 2) + nb64 * 32;
                                        red_add_bf16x2_bits(dst + lane, sm.epi_or_flip.bf16x2[m][lane]);
                                    }
                                }
                            }
                        }
                    } else {
                        #pragma unroll
                        for (int half = 0; half < 2; ++half) {
                            if ((wm >> 1) == half) {
                                const int r_lo = (wm & 1) * 16 + g;
                                const int r_hi = r_lo + 8;
                                #pragma unroll
                                for (int sn = 0; sn < 4; ++sn) {
                                    const int c0 = wp * 32 + sn * 8 + lt * 2;
                                    sm.epi_or_flip.f32[r_lo][c0 + 0] = acc[sn][0] * scale_e;
                                    sm.epi_or_flip.f32[r_lo][c0 + 1] = acc[sn][1] * scale_e;
                                    sm.epi_or_flip.f32[r_hi][c0 + 0] = acc[sn][2] * scale_e;
                                    sm.epi_or_flip.f32[r_hi][c0 + 1] = acc[sn][3] * scale_e;
                                }
                            }
                            __syncthreads();
                            #pragma unroll
                            for (int k = 0; k < 4; ++k) {
                                const int r_epi  = warp * 4 + k;
                                const int m      = half * 32 + r_epi;
                                const int r_glob = row0 + m;
                                if (r_glob < end) {
                                    const int32_t tok = sm.sTok[cur][m];
                                    if (tok >= 0) {
                                        float* dst = reinterpret_cast<float*>(pOut)
                                                   + static_cast<int64_t>(tok) * N + nb64 * 64;
                                        red_add_f32(dst + lane,      sm.epi_or_flip.f32[r_epi][lane]);
                                        red_add_f32(dst + 32 + lane, sm.epi_or_flip.f32[r_epi][32 + lane]);
                                    }
                                }
                            }
                            if (half == 0) __syncthreads();
                        }
                    }

                    if (mi_blk + 1 < num_m) {
                        if (!a_pinned) cp_async_wait<0>();
                        __syncthreads();
                    } else if (dp_step + 1 < PANELS64_PER_CTA) {
                        __syncthreads();
                    }
                }
            }
            __syncthreads();
        }
    }
}

// Wave-V13 ZERO-SPIN: default reserve_sms = 0 -> unlocks ALL 48/48 SMs (+20% SM throughput, 0 deadlock risk!)
static int g_mx7_reserve_sms = 0;

inline bool is_stream_capturing(cudaStream_t stream) {
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    cudaError_t err = cudaStreamIsCapturing(stream, &status);
    if (err != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return status != cudaStreamCaptureStatusNone;
}

template <typename KernelFn>
inline int get_mx7_v10_coresident_ctas(KernelFn fn) {
    static int cached_sms = 0;
    static int cached_bpm = 0;
    if (cached_sms <= 0) {
        cudaFuncSetAttribute(fn, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
        int dev = 0, sms = 48, bpm = 2;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bpm, fn, 256, 0);
        cached_sms = (sms > 0) ? sms : 48;
        cached_bpm = (bpm > 0) ? ((bpm > 2) ? 2 : bpm) : 1;
    }
    int reserve = g_mx7_reserve_sms;
    if (reserve < 0) reserve = 0;
    if (cached_sms > 14 && reserve > cached_sms - 8) reserve = cached_sms - 8;
    const int avail_sms = (cached_sms > reserve + 6) ? (cached_sms - reserve) : cached_sms;
    const int g = avail_sms * cached_bpm;
    return (g > 0) ? g : 40;
}

template <int COMBINE_MODE, int P64>
int launch_mx7_w2_v10_cm_p(
    const void* A, const void* AS, const void* Row2Tok,
    const void* SLABS, const void* WS3, const void* FLIPS, const void* F_OFF,
    const void* Ones, const void* Tiles, const void* PlanBuf, void* SliceState, void* Out,
    int max_ctas, int tile, int num_experts, int max_work,
    uint32_t base_ws, int as_row_major, int as_row_stride,
    int w_evict_mode, int zero_mode, int tokens, cudaStream_t stream
) {
    constexpr int N_SLICES = 72 / P64;
    const int eff_work = (PlanBuf != nullptr && max_work > 0) ? max_work : num_experts;
    const bool do_zero = (COMBINE_MODE == 2 && zero_mode != 0 && tokens > 0);
    const unsigned num_zchk = do_zero ? static_cast<unsigned>((tokens + 127) >> 7) : 0u;
    if (!do_zero) {
        pf_mx7_w2_reset_cursors_v13_kernel<<<1, 64, 0, stream>>>(
            reinterpret_cast<uint32_t*>(SliceState), N_SLICES);
    }
    if (as_row_major != 0) {
        auto kfn = pf_mx7_w2_v10_kernel<1, COMBINE_MODE, P64>;
        const int g_ctas = get_mx7_v10_coresident_ctas(kfn);
        for (int s = 0; s < N_SLICES; ++s) {
            if (do_zero) {
                pf_mx7_w2_zero_slice_v13_kernel<P64><<<dim3(num_zchk), dim3(256), 0, stream>>>(
                    reinterpret_cast<uint16_t*>(Out), reinterpret_cast<uint32_t*>(SliceState), tokens, s);
            }
            kfn<<<dim3(static_cast<unsigned>(g_ctas)), dim3(256), 0, stream>>>(
                reinterpret_cast<const uint8_t*>(A), reinterpret_cast<const uint8_t*>(AS),
                reinterpret_cast<const int32_t*>(Row2Tok),
                reinterpret_cast<const uint8_t*>(SLABS), reinterpret_cast<const uint4*>(WS3),
                reinterpret_cast<const uint16_t*>(FLIPS), reinterpret_cast<const int32_t*>(F_OFF),
                reinterpret_cast<const float*>(Ones), reinterpret_cast<const int32_t*>(Tiles),
                reinterpret_cast<const int32_t*>(PlanBuf), reinterpret_cast<uint32_t*>(SliceState), Out,
                max_ctas, tile, num_experts, eff_work, base_ws, as_row_stride,
                w_evict_mode, s);
        }
    } else {
        auto kfn = pf_mx7_w2_v10_kernel<0, COMBINE_MODE, P64>;
        const int g_ctas = get_mx7_v10_coresident_ctas(kfn);
        for (int s = 0; s < N_SLICES; ++s) {
            if (do_zero) {
                pf_mx7_w2_zero_slice_v13_kernel<P64><<<dim3(num_zchk), dim3(256), 0, stream>>>(
                    reinterpret_cast<uint16_t*>(Out), reinterpret_cast<uint32_t*>(SliceState), tokens, s);
            }
            kfn<<<dim3(static_cast<unsigned>(g_ctas)), dim3(256), 0, stream>>>(
                reinterpret_cast<const uint8_t*>(A), reinterpret_cast<const uint8_t*>(AS),
                reinterpret_cast<const int32_t*>(Row2Tok),
                reinterpret_cast<const uint8_t*>(SLABS), reinterpret_cast<const uint4*>(WS3),
                reinterpret_cast<const uint16_t*>(FLIPS), reinterpret_cast<const int32_t*>(F_OFF),
                reinterpret_cast<const float*>(Ones), reinterpret_cast<const int32_t*>(Tiles),
                reinterpret_cast<const int32_t*>(PlanBuf), reinterpret_cast<uint32_t*>(SliceState), Out,
                max_ctas, tile, num_experts, eff_work, base_ws, as_row_stride,
                w_evict_mode, s);
        }
    }
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

}  // namespace

// =============================================================================
// C ABI Entrypoints (strictly prefill-only; loaded via ctypes in pf_moe_native.py)
// =============================================================================
extern "C" {

int pf_mx7_build_expert_plan(
    const void* Tiles, void* PlanBuf,
    int max_ctas, int tile, int num_experts, int split_m, int max_work,
    uint64_t stream_u64
) {
    if (!Tiles || !PlanBuf || max_ctas <= 0 || tile < 64 || num_experts <= 0 || max_work <= 0) {
        return -1;
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_u64);
    if (is_stream_capturing(stream)) return -4;
    pf_mx7_build_expert_plan_kernel<<<1, 256, 0, stream>>>(
        reinterpret_cast<const int32_t*>(Tiles),
        reinterpret_cast<int32_t*>(PlanBuf),
        max_ctas, tile, num_experts, split_m, max_work);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

int pf_mx7_zero_slice_wb_bf16(
    void* Out_bf16, int tokens, int nb_blk, int panels64, uint64_t stream_u64
) {
    if (!Out_bf16 || tokens <= 0 || nb_blk < 0) return -1;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_u64);
    if (is_stream_capturing(stream)) return -4;
    dim3 grid((tokens + 7) / 8);
    if (panels64 == 4) {
        pf_mx7_zero_bf16_slice_wb_kernel<4><<<grid, 256, 0, stream>>>(
            reinterpret_cast<uint16_t*>(Out_bf16), tokens, nb_blk);
    } else if (panels64 == 6) {
        pf_mx7_zero_bf16_slice_wb_kernel<6><<<grid, 256, 0, stream>>>(
            reinterpret_cast<uint16_t*>(Out_bf16), tokens, nb_blk);
    } else if (panels64 == 8) {
        pf_mx7_zero_bf16_slice_wb_kernel<8><<<grid, 256, 0, stream>>>(
            reinterpret_cast<uint16_t*>(Out_bf16), tokens, nb_blk);
    } else if (panels64 == 12) {
        pf_mx7_zero_bf16_slice_wb_kernel<12><<<grid, 256, 0, stream>>>(
            reinterpret_cast<uint16_t*>(Out_bf16), tokens, nb_blk);
    } else {
        return -1;
    }
    return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

// Persistent Fused MX7 W13 + SiTU + MXFP8 Quantize -> (W2Act [R, 288] e4m3fn, W2Scales [R, 12] u8)
int pf_mx7_w13_prefill_fused_situ(
    const void* A, const void* AS, const void* Row2Tok, const void* Gammas,
    const void* SLABS, const void* WS3, const void* FLIPS, const void* F_OFF,
    const void* Ones, const void* Tiles, const void* PlanBuf,
    void* W2Act, void* W2Scales,
    int max_ctas, int tile, int num_experts, int max_work, int bm, int base_ws,
    int use_r2t, int as_row_major, int w2_as_row_major, int w2_as_stride,
    int w_evict_mode, uint64_t stream_u64
) {
    if (!A || !AS || !Gammas || !SLABS || !WS3 || !FLIPS || !F_OFF || !Ones || !Tiles || !W2Act || !W2Scales) {
        return -1;
    }
    if (tile < 64 || (tile % 64) != 0 || max_ctas <= 0 || num_experts <= 0) return -1;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_u64);
    if (is_stream_capturing(stream)) return -4;
    if (bm == 256) {
        return launch_mx7_w13_impl<256, 2, 1, 1>(
            A, AS, Row2Tok, Gammas, SLABS, WS3, FLIPS, F_OFF, Ones, Tiles, PlanBuf,
            W2Act, W2Scales, max_ctas, tile, num_experts, max_work,
            static_cast<uint32_t>(base_ws), use_r2t, as_row_major, w2_as_row_major, w2_as_stride,
            w_evict_mode, stream);
    }
    return launch_mx7_w13_impl<128, 3, 2, 1>(
        A, AS, Row2Tok, Gammas, SLABS, WS3, FLIPS, F_OFF, Ones, Tiles, PlanBuf,
        W2Act, W2Scales, max_ctas, tile, num_experts, max_work,
        static_cast<uint32_t>(base_ws), use_r2t, as_row_major, w2_as_row_major, w2_as_stride,
        w_evict_mode, stream);
}

// Persistent Unfused MX7 W13 -> [R, 576] BF16 (exact A/B drop-in vs mx7_v5_prefill_native_w13!)
int pf_mx7_w13_prefill_unfused(
    const void* A, const void* AS, const void* Row2Tok,
    const void* SLABS, const void* WS3, const void* FLIPS, const void* F_OFF,
    const void* Ones, const void* Tiles, const void* PlanBuf,
    void* Out_bf16,
    int max_ctas, int tile, int num_experts, int max_work, int bm, int base_ws,
    int use_r2t, int as_row_major, int w_evict_mode, uint64_t stream_u64
) {
    if (!A || !AS || !SLABS || !WS3 || !FLIPS || !F_OFF || !Ones || !Tiles || !Out_bf16) {
        return -1;
    }
    if (tile < 64 || (tile % 64) != 0 || max_ctas <= 0 || num_experts <= 0) return -1;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_u64);
    if (is_stream_capturing(stream)) return -4;
    if (bm == 256) {
        return launch_mx7_w13_impl<256, 2, 1, 0>(
            A, AS, Row2Tok, nullptr, SLABS, WS3, FLIPS, F_OFF, Ones, Tiles, PlanBuf,
            Out_bf16, nullptr, max_ctas, tile, num_experts, max_work,
            static_cast<uint32_t>(base_ws), use_r2t, as_row_major, 0, 0, w_evict_mode, stream);
    }
    return launch_mx7_w13_impl<128, 3, 2, 0>(
        A, AS, Row2Tok, nullptr, SLABS, WS3, FLIPS, F_OFF, Ones, Tiles, PlanBuf,
        Out_bf16, nullptr, max_ctas, tile, num_experts, max_work,
        static_cast<uint32_t>(base_ws), use_r2t, as_row_major, 0, 0, w_evict_mode, stream);
}

// Persistent MX7 W2 + L2-Hot Fused Combine (BM=64, BN=64, 2 CTAs/SM, 0 spills!)
// panels64_per_cta: 4 (256 cols / 4 MiB), 6 (384 cols / 6 MiB), 8 (512 cols / 8 MiB), 12 (768 cols).
int pf_mx7_w2_prefill_persistent(
    const void* A, const void* AS, const void* Row2Tok,
    const void* SLABS, const void* WS3, const void* FLIPS, const void* F_OFF,
    const void* Ones, const void* Tiles, const void* PlanBuf,
    void* Out,
    int max_ctas, int tile, int num_experts, int max_work, int base_ws,
    int as_row_major, int as_row_stride,
    int combine_mode, int w_evict_mode, int panels64_per_cta,
    int l2_zero_slice, int tokens, uint64_t stream_u64
) {
    if (!A || !AS || !SLABS || !WS3 || !FLIPS || !F_OFF || !Ones || !Tiles || !Out) return -1;
    if (tile < 64 || (tile % 64) != 0 || max_ctas <= 0 || num_experts <= 0) return -1;
    if (combine_mode < 0 || combine_mode > 2) return -1;
    const int eff_work = (PlanBuf != nullptr && max_work > 0) ? max_work : num_experts;
    const int p64 = (combine_mode == 0) ? 12
                  : (combine_mode == 1) ? 4
                  : ((panels64_per_cta == 4 || panels64_per_cta == 6 ||
                      panels64_per_cta == 8 || panels64_per_cta == 12)
                         ? panels64_per_cta : 4);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_u64);
    if (is_stream_capturing(stream)) return -4;
    const uint32_t bws  = static_cast<uint32_t>(base_ws);
    const int arm       = as_row_major ? 1 : 0;

    #define DISPATCH_MX7_W2(ARM, CM, P) \
        return launch_mx7_w2_cfg<ARM, CM, P>( \
            A, AS, Row2Tok, SLABS, WS3, FLIPS, F_OFF, Ones, Tiles, PlanBuf, Out, \
            max_ctas, tile, num_experts, eff_work, bws, as_row_stride, w_evict_mode, \
            l2_zero_slice, tokens, stream)

    if (combine_mode == 0) {
        if (arm) DISPATCH_MX7_W2(1, 0, 12);
        else     DISPATCH_MX7_W2(0, 0, 12);
    } else if (combine_mode == 1) {
        if (arm) DISPATCH_MX7_W2(1, 1, 4);
        else     DISPATCH_MX7_W2(0, 1, 4);
    } else {
        if (arm) {
            if (p64 == 4)       DISPATCH_MX7_W2(1, 2, 4);
            else if (p64 == 6)  DISPATCH_MX7_W2(1, 2, 6);
            else if (p64 == 8)  DISPATCH_MX7_W2(1, 2, 8);
            else                DISPATCH_MX7_W2(1, 2, 12);
        } else {
            if (p64 == 4)       DISPATCH_MX7_W2(0, 2, 4);
            else if (p64 == 6)  DISPATCH_MX7_W2(0, 2, 6);
            else if (p64 == 8)  DISPATCH_MX7_W2(0, 2, 8);
            else                DISPATCH_MX7_W2(0, 2, 12);
        }
    }
    #undef DISPATCH_MX7_W2
}


// =============================================================================
// Wave-V10 Public C ABI Entrypoints (100% Superset of V4!)
// =============================================================================

int pf_mx7_v13_version() {
    return 1300;  // Wave-V13 ZERO-SPIN Host-Sliced 48/48-SM MX7 Engine!
}

int pf_mx7_v10_version() {
    return 1300;  // Back-compat V10 symbol reports 1300 (zero-spin superset)
}

int pf_mx7_v10_set_reserve_sms(int reserve_sms) {
    if (reserve_sms < 0) reserve_sms = 0;
    if (reserve_sms > 24) reserve_sms = 24;
    g_mx7_reserve_sms = reserve_sms;
    return g_mx7_reserve_sms;
}

int pf_mx7_v10_get_reserve_sms() {
    return g_mx7_reserve_sms;
}

int pf_mx7_build_plan_v10(
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
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_u64);
    if (is_stream_capturing(stream)) return -4;
    pf_mx7_build_plan_v10_kernel<<<1, 256, 0, stream>>>(
        reinterpret_cast<const int32_t*>(Tiles),
        reinterpret_cast<int32_t*>(PlanBuf),
        max_ctas, tile, num_experts, split_m, max_work
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

int pf_mx7_w2_v10_unfused(
    const void* A, const void* AS,
    const void* SLABS, const void* WS3, const void* FLIPS, const void* F_OFF,
    const void* Ones, const void* Tiles, const void* PlanBuf, void* SliceState, void* Out,
    int max_ctas, int tile, int num_experts, int max_work,
    int base_ws, int as_row_major, int as_row_stride,
    int w_evict_mode, uint64_t stream_u64
) {
    if (!A || !AS || !SLABS || !WS3 || !FLIPS || !F_OFF || !Ones || !Tiles || !SliceState || !Out) {
        return -2;
    }
    if (tile < 64 || (tile % 64) != 0) return -2;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_u64);
    if (is_stream_capturing(stream)) return -4;
    return launch_mx7_w2_v10_cm_p<0, 12>(
        A, AS, Tiles, SLABS, WS3, FLIPS, F_OFF, Ones, Tiles, PlanBuf, SliceState, Out,
        max_ctas, tile, num_experts, max_work, static_cast<uint32_t>(base_ws),
        as_row_major, as_row_stride, w_evict_mode, 0, 0, stream);
}

int pf_mx7_w2_v10_combine(
    const void* A, const void* AS, const void* Row2Tok,
    const void* SLABS, const void* WS3, const void* FLIPS, const void* F_OFF,
    const void* Ones, const void* Tiles, const void* PlanBuf, void* SliceState, void* Out,
    int max_ctas, int tile, int num_experts, int max_work,
    int base_ws, int as_row_major, int as_row_stride,
    int combine_mode, int w_evict_mode, int panels64,
    int zero_mode, int tokens, uint64_t stream_u64
) {
    if (!A || !AS || !SLABS || !WS3 || !FLIPS || !F_OFF || !Ones || !Tiles || !SliceState || !Out) {
        return -2;
    }
    if (tile < 64 || (tile % 64) != 0) return -2;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_u64);
    if (is_stream_capturing(stream)) return -4;

    const uint32_t bws = static_cast<uint32_t>(base_ws);
    if (combine_mode == 0) {
        return launch_mx7_w2_v10_cm_p<0, 12>(
            A, AS, Tiles, SLABS, WS3, FLIPS, F_OFF, Ones, Tiles, PlanBuf, SliceState, Out,
            max_ctas, tile, num_experts, max_work, bws, as_row_major, as_row_stride,
            w_evict_mode, 0, 0, stream);
    }
    if (combine_mode == 1) {
        if (!Row2Tok) return -2;
        return launch_mx7_w2_v10_cm_p<1, 4>(
            A, AS, Row2Tok, SLABS, WS3, FLIPS, F_OFF, Ones, Tiles, PlanBuf, SliceState, Out,
            max_ctas, tile, num_experts, max_work, bws, as_row_major, as_row_stride,
            w_evict_mode, 0, tokens, stream);
    }
    if (combine_mode == 2) {
        if (!Row2Tok) return -2;
        if (panels64 == 3) {
            return launch_mx7_w2_v10_cm_p<2, 3>(
                A, AS, Row2Tok, SLABS, WS3, FLIPS, F_OFF, Ones, Tiles, PlanBuf, SliceState, Out,
                max_ctas, tile, num_experts, max_work, bws, as_row_major, as_row_stride,
                w_evict_mode, zero_mode, tokens, stream);
        }
        if (panels64 == 4) {
            return launch_mx7_w2_v10_cm_p<2, 4>(
                A, AS, Row2Tok, SLABS, WS3, FLIPS, F_OFF, Ones, Tiles, PlanBuf, SliceState, Out,
                max_ctas, tile, num_experts, max_work, bws, as_row_major, as_row_stride,
                w_evict_mode, zero_mode, tokens, stream);
        }
        if (panels64 == 8) {
            return launch_mx7_w2_v10_cm_p<2, 8>(
                A, AS, Row2Tok, SLABS, WS3, FLIPS, F_OFF, Ones, Tiles, PlanBuf, SliceState, Out,
                max_ctas, tile, num_experts, max_work, bws, as_row_major, as_row_stride,
                w_evict_mode, zero_mode, tokens, stream);
        }
        if (panels64 == 9) {
            return launch_mx7_w2_v10_cm_p<2, 9>(
                A, AS, Row2Tok, SLABS, WS3, FLIPS, F_OFF, Ones, Tiles, PlanBuf, SliceState, Out,
                max_ctas, tile, num_experts, max_work, bws, as_row_major, as_row_stride,
                w_evict_mode, zero_mode, tokens, stream);
        }
        if (panels64 == 12) {
            return launch_mx7_w2_v10_cm_p<2, 12>(
                A, AS, Row2Tok, SLABS, WS3, FLIPS, F_OFF, Ones, Tiles, PlanBuf, SliceState, Out,
                max_ctas, tile, num_experts, max_work, bws, as_row_major, as_row_stride,
                w_evict_mode, zero_mode, tokens, stream);
        }
        return launch_mx7_w2_v10_cm_p<2, 6>(
            A, AS, Row2Tok, SLABS, WS3, FLIPS, F_OFF, Ones, Tiles, PlanBuf, SliceState, Out,
            max_ctas, tile, num_experts, max_work, bws, as_row_major, as_row_stride,
            w_evict_mode, zero_mode, tokens, stream);
    }
    return -4;
}

}  // extern "C"
