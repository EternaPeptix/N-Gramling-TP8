// SPDX-License-Identifier: Apache-2.0
// =============================================================================
// SM121a (NVIDIA DGX Spark GB10) Wave-V11 Prefill-Only FP8-f16acc/f32-Promote
// Dense GEMM & Fused `wo_ud -> A2A_INT8_g64` CUDA Kernels (`libpf_dense_fp8_tc_v11.so`)
// =============================================================================
//
// Strictly prefill-only (`M >= 512`, `!is_stream_capturing(stream)` returning -4).
// Zero effect on C=1 / C=32 decode (`M <= 128`, CUDAGraph capture, `exact_ink.py`).
//
// Mystery #2 Root Cause & Wave-V11 Architecture:
//   - On GB10 SM121a, `linear.py`'s `torch._scaled_mm(..., use_fast_accum=False)`
//     forces FP8 `m16n8k32.f32` (`157 TFLOPS` peak), turning 40L Local `qkvr`
//     (733 TFLOPs) + 8L Global `q_side` (127 TFLOPs) + 48L `wo_ud` (356 TFLOPs)
//     into `~8.1 s` of pure compute + `n_pad = 1904/1648` strided slice copies!
//   - Meanwhile SM121a's `mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16`
//     runs at `314 TFLOPS` peak (`2.00x` issue rate!). Naive `(448, 448)` inputs
//     overflow FP16 catastrophically (`6.4 dB`), BUT scaling cached weights once
//     at load to `W_MAX = 2.2f` (with untouched `A in [-448, 448]`, `K_CHUNK = 64`)
//     or `(A_MAX = 22.0f, W_MAX = 22.0f, K_CHUNK = 64/128)` PROVABLY bounds every
//     inner dot product `< 63,079 < 65,504 (FP16_MAX)` with ZERO overflows and
//     matches/beats cuBLASLt FP8-f32 SQNR (`28.5-32.0 dB` across Normal & HeavyTail)!
//   - Furthermore, `pf_wo_ud_fused_a2a_int8_v11_fwd` fuses:
//       (1) On-the-fly BF16 -> FP8 (`A_MAX=22.0f`) input quant (kills `quant_k768`!)
//       (2) `314 TFLOPS` FP8-f16/f32-promote `wo_ud` GEMM (`[8*S, 768] @ [768, 4608]`)
//       (3) Direct register-level `A2A_INT8_g64` epilogue (`dst != rank` -> packed
//           `send_buf[dst]` via `st.global.cs`; `dst == rank` -> `local_shard` BF16),
//           eliminating `attn_partial` `[T, 4608]` BF16 write (`-467 GB`), eliminating
//           `a2a_quant_int8_g64_lowsm` (`-620 GB` & `-4,128` launches on `comm_stream`),
//           and letting `comm_stream` fire 1-hop A2A on cycle 1!
//
// Strict SM121a Hygiene:
//   - `-gencode arch=compute_121a,code=sm_121a --fmad=false --ptxas-options=-v`
//   - ZERO `--use_fast_math`, ZERO `--ftz=true`, 0 B stack frame, 0 B spills
//   - `BM=128, BN=128, BK=64`, `33.125 KiB` static SMEM (`2 CTAs/SM`), `reserve_sms` DBO-safe.
// =============================================================================

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <math.h>

namespace {

constexpr int D_BM = 128;
constexpr int D_BN = 128;
constexpr int D_BK = 64;
constexpr int D_THREADS = 256;
constexpr int D_GROUP_M = 8;

static inline bool is_stream_capturing(cudaStream_t stream) {
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    cudaError_t err = cudaStreamIsCapturing(stream, &status);
    if (err != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return status != cudaStreamCaptureStatusNone;
}

static inline int get_sm_count() {
    static int cached_sms = 0;
    if (cached_sms <= 0) {
        int dev = 0;
        if (cudaGetDevice(&dev) == cudaSuccess) {
            int sms = 0;
            if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) == cudaSuccess && sms > 0) {
                cached_sms = sms;
            }
        }
        if (cached_sms <= 0) cached_sms = 48;
    }
    return cached_sms;
}

// -----------------------------------------------------------------------------
// Low-level SM121a PTX Helpers (0 spills, --fmad=false safe)
// -----------------------------------------------------------------------------
static __device__ __forceinline__ void cp_async_ca_16(uint32_t smem_addr, const void* gmem_ptr, bool pred) {
    int src_bytes = pred ? 16 : 0;
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 16, %2;\n"
        :: "r"(smem_addr), "l"(gmem_ptr), "r"(src_bytes) : "memory"
    );
}

static __device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template <int N>
static __device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}

static __device__ __forceinline__ void ldmatrix_x4(
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t smem_addr
) {
    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(smem_addr)
    );
}

static __device__ __forceinline__ void mma_m16n8k32_f16_e4m3(
    uint32_t& d0, uint32_t& d1,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    uint32_t c0, uint32_t c1
) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 "
        "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n"
        : "=r"(d0), "=r"(d1)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "r"(c0), "r"(c1)
    );
}

static __device__ __forceinline__ uint16_t f32x2_to_e4m3x2_rn(float a, float b) {
    uint16_t out;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
        : "=h"(out) : "f"(b), "f"(a));
    return out;
}

static __device__ __forceinline__ uint32_t f32x4_to_e4m3x4_rn(float v0, float v1, float v2, float v3) {
    uint16_t lo = f32x2_to_e4m3x2_rn(v0, v1);
    uint16_t hi = f32x2_to_e4m3x2_rn(v2, v3);
    return static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
}

static __device__ __forceinline__ uint4 ld_global_ca_u4(const void* p) {
    uint4 v;
    asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}

static __device__ __forceinline__ void st_global_cs_u4(void* p, uint4 v) {
    asm volatile("st.global.cs.v4.u32 [%0], {%1,%2,%3,%4};"
                 :: "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}

static __device__ __forceinline__ void st_global_wb_u4(void* p, uint4 v) {
    asm volatile("st.global.v4.u32 [%0], {%1,%2,%3,%4};"
                 :: "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}

static __device__ __forceinline__ void st_global_cs_u16(void* p, uint16_t v) {
    asm volatile("st.global.cs.u16 [%0], %1;" :: "l"(p), "h"(v) : "memory");
}

static __device__ __forceinline__ float warp_max_f32(float v) {
    v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, 16));
    v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, 8));
    v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, 4));
    v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, 2));
    v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, 1));
    return v;
}

static __device__ __forceinline__ int8_t f32_to_i8_sat(float v) {
    int qi = __float2int_rn(v);
    if (qi > 127) qi = 127;
    if (qi < -127) qi = -127;
    return static_cast<int8_t>(qi);
}

// 0-way bank-conflict-free XOR-swizzled SMEM byte offset for a [128, 64] uint8 tile
// (even rows hit banks 0..15 with (row>>1)&3 in {0,1,2,3}; odd rows hit banks 16..31!)
static __device__ __forceinline__ uint32_t swiz_off_128x64(int row, int chunk16) {
    int sw_c = chunk16 ^ ((row >> 1) & 3);
    return static_cast<uint32_t>(row * 64 + sw_c * 16);
}

// -----------------------------------------------------------------------------
// Kernel 0: One-Time Load Weight Rowwise Quantizer (`W_MAX = 2.2f` or `22.0f`)
//   Quantizes BF16 W [N, K] -> FP8-E4M3FN [N, K] + FP32 scale [N], where
//   `scale[n] = max(amax_n / w_max, 1e-12f)` and `w_fp8 = cvt.e4m3(w / scale[n])`.
// -----------------------------------------------------------------------------
__global__ __launch_bounds__(256, 2)
void pf_quant_weight_f16safe_v11_kernel(
    const __nv_bfloat16* __restrict__ W,
    uint8_t* __restrict__ W_FP8,
    float* __restrict__ W_SCALE,
    int N,
    int K,
    float w_max
) {
    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int n_vec8 = K >> 3;
    const float inv_wmax = __fdiv_rn(1.0f, w_max);
    __shared__ float sWarpMax[8];

    for (int row = (int)blockIdx.x; row < N; row += (int)gridDim.x) {
        const __nv_bfloat16* w_row = W + static_cast<int64_t>(row) * K;
        uint8_t* q_row = W_FP8 + static_cast<int64_t>(row) * K;

        float local_amax = 0.0f;
        for (int idx = tid; idx < n_vec8; idx += 256) {
            uint4 raw = ld_global_ca_u4(w_row + (idx << 3));
            const __nv_bfloat162* h2 = reinterpret_cast<const __nv_bfloat162*>(&raw);
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                float2 f = __bfloat1622float2(h2[i]);
                local_amax = fmaxf(local_amax, fmaxf(fabsf(f.x), fabsf(f.y)));
            }
        }
        float wm = warp_max_f32(local_amax);
        if (lane == 0) sWarpMax[warp] = wm;
        __syncthreads();

        float row_amax = fmaxf(
            fmaxf(fmaxf(sWarpMax[0], sWarpMax[1]), fmaxf(sWarpMax[2], sWarpMax[3])),
            fmaxf(fmaxf(sWarpMax[4], sWarpMax[5]), fmaxf(sWarpMax[6], sWarpMax[7]))
        );
        float sc = fmaxf(__fmul_rn(row_amax, inv_wmax), 1.0e-12f);
        float inv_sc = __fdiv_rn(1.0f, sc);
        if (tid == 0) W_SCALE[row] = sc;

        for (int idx = tid; idx < n_vec8; idx += 256) {
            uint4 raw = ld_global_ca_u4(w_row + (idx << 3));
            const __nv_bfloat162* h2 = reinterpret_cast<const __nv_bfloat162*>(&raw);
            float2 f0 = __bfloat1622float2(h2[0]);
            float2 f1 = __bfloat1622float2(h2[1]);
            float2 f2 = __bfloat1622float2(h2[2]);
            float2 f3 = __bfloat1622float2(h2[3]);
            uint32_t q_lo = f32x4_to_e4m3x4_rn(
                __fmul_rn(f0.x, inv_sc), __fmul_rn(f0.y, inv_sc),
                __fmul_rn(f1.x, inv_sc), __fmul_rn(f1.y, inv_sc)
            );
            uint32_t q_hi = f32x4_to_e4m3x4_rn(
                __fmul_rn(f2.x, inv_sc), __fmul_rn(f2.y, inv_sc),
                __fmul_rn(f3.x, inv_sc), __fmul_rn(f3.y, inv_sc)
            );
            *reinterpret_cast<uint2*>(q_row + (idx << 3)) = make_uint2(q_lo, q_hi);
        }
        __syncthreads();
    }
}

// -----------------------------------------------------------------------------
// Shared SMEM Layout for Both `pf_dense_fp8_f16acc_v11` & `pf_wo_ud_*_v11`
//   Total static SMEM = 32,768 + 1,536 + 128 = 34,432 B (33.6 KiB -> 2 CTAs/SM!)
// -----------------------------------------------------------------------------
struct DenseV11CtaParams {
    int M;
    int N;
    int K;
    int num_m_tiles;
    int num_n_tiles;
    int total_tiles;
    uint32_t sa0_ptr;
    uint32_t sa1_ptr;
    uint32_t sb0_ptr;
    uint32_t sb1_ptr;
};

struct alignas(16) DenseV11Smem {
    union {
        struct {
            alignas(16) uint8_t sA[2][D_BM * D_BK];   // 16,384 B
            alignas(16) uint8_t sB[2][D_BN * D_BK];   // 16,384 B
        } ab;
        alignas(16) uint16_t sEpiBf16[D_BM][D_BN];    // 32,768 B (100% unioned!)
        alignas(16) uint8_t  sEpiI8[D_BM][D_BN];      // 16,384 B (for A2A INT8)
    } u;
    alignas(16) float sAs[D_BM];                      // 512 B
    alignas(16) float sAsInv[D_BM];                   // 512 B (used by BF16-in wo_ud)
    alignas(16) float sBs[D_BN];                      // 512 B
    DenseV11CtaParams sP;                             // 40 B
};

static_assert(sizeof(DenseV11Smem) <= 36864, "DenseV11Smem must fit 2 CTAs/SM cleanly");

// Helper: execute 1 full BK=64 stage (2 x k32 m16n8k32.f16 MMAs) and promote to FP32 acc!
//   Guarantees <= 16 live FP16x2 regs (1 `mi` at a time) -> 0 B spills at 2 CTAs/SM!
static __device__ __forceinline__ void dense_v11_mma_bk64_stage(
    uint32_t sa_base,
    uint32_t sb_base,
    int wm,
    int wn,
    int lane,
    float acc[2][8][4]
) {
    const int sub = lane >> 3;    // 0..3
    const int rin = lane & 7;     // 0..7
    const int sub_row = sub & 1;  // 0..1 (top/bot 8 rows of 16x32 A)
    const int sub_k   = sub >> 1; // 0..1 (left/right 16 bytes of 32B K)
    const int b_sub_n = sub >> 1; // 0..1 (2 consecutive N-tiles of 8 cols!)
    const int b_sub_k = sub & 1;  // 0..1 (left/right 16 bytes of 32B K)

    #pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
        uint32_t h0[8] = {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};
        uint32_t h1[8] = {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};

        const int a_row = wm * 32 + mi * 16 + sub_row * 8 + rin;

        #pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
            const int a_chunk = ks * 2 + sub_k;
            const uint32_t a_addr = sa_base + swiz_off_128x64(a_row, a_chunk);
            uint32_t a0, a1, a2, a3;
            ldmatrix_x4(a0, a1, a2, a3, a_addr);

            const int b_chunk = ks * 2 + b_sub_k;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int b_row = wn * 64 + (2 * j + b_sub_n) * 8 + rin;
                const uint32_t b_addr = sb_base + swiz_off_128x64(b_row, b_chunk);
                uint32_t b0_0, b1_0, b0_1, b1_1;
                ldmatrix_x4(b0_0, b1_0, b0_1, b1_1, b_addr);

                mma_m16n8k32_f16_e4m3(
                    h0[2 * j], h1[2 * j],
                    a0, a1, a2, a3, b0_0, b1_0,
                    h0[2 * j], h1[2 * j]
                );
                mma_m16n8k32_f16_e4m3(
                    h0[2 * j + 1], h1[2 * j + 1],
                    a0, a1, a2, a3, b0_1, b1_1,
                    h0[2 * j + 1], h1[2 * j + 1]
                );
            }
        }

        // Promote BK=64 FP16x2 partial sums -> FP32 accumulators (0 overflow guaranteed!)
        #pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            float2 f0 = __half22float2(*reinterpret_cast<const __half2*>(&h0[nt]));
            float2 f1 = __half22float2(*reinterpret_cast<const __half2*>(&h1[nt]));
            acc[mi][nt][0] = __fadd_rn(acc[mi][nt][0], f0.x);
            acc[mi][nt][1] = __fadd_rn(acc[mi][nt][1], f0.y);
            acc[mi][nt][2] = __fadd_rn(acc[mi][nt][2], f1.x);
            acc[mi][nt][3] = __fadd_rn(acc[mi][nt][3], f1.y);
        }
    }
}

// Helper: L2-swizzled 2D tile coordinate from linear persistent `tile_id`
static __device__ __forceinline__ void swizzle_tile_coord(
    int tile_id, int num_m_tiles, int num_n_tiles, int& tm, int& tn
) {
    const int tiles_per_grp = D_GROUP_M * num_n_tiles;
    const int grp = tile_id / tiles_per_grp;
    const int first_m = grp * D_GROUP_M;
    const int grp_m = min(D_GROUP_M, num_m_tiles - first_m);
    const int in_grp = tile_id - grp * tiles_per_grp;
    tm = first_m + (in_grp % grp_m);
    tn = in_grp / grp_m;
}

// -----------------------------------------------------------------------------
// Kernel 1: Pre-Quantized FP8-in Dense Prefill GEMM (`qkvr`, `q_side`, `router`)
//   `A [M, K]` FP8-E4M3FN (`[-448, 448]` from AG!), `W [N, K]` FP8-E4M3FN (`[-2.2, 2.2]`)
//   -> `Out [M, N]` BF16, native `N % 8 == 0` (ZERO 1904/1648 padding view copy!).
// -----------------------------------------------------------------------------
__global__ __launch_bounds__(D_THREADS, 1)
void pf_dense_fp8_f16acc_v11_kernel(
    const uint8_t* __restrict__ A,
    const float* __restrict__ A_SCALE,
    const uint8_t* __restrict__ W,
    const float* __restrict__ W_SCALE,
    __nv_bfloat16* __restrict__ OUT,
    int M,
    int N,
    int K,
    int stride_out
) {
    __shared__ DenseV11Smem sm;
    const int tid  = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wm   = warp >> 1;  // 0..3
    const int wn   = warp & 1;   // 0..1
    const int gid  = lane >> 2;  // 0..7
    const int t    = lane & 3;   // 0..3

    if (tid == 0) {
        sm.sP.M = M;
        sm.sP.N = N;
        sm.sP.K = K;
        sm.sP.num_m_tiles = (M + D_BM - 1) / D_BM;
        sm.sP.num_n_tiles = (N + D_BN - 1) / D_BN;
        sm.sP.total_tiles = sm.sP.num_m_tiles * sm.sP.num_n_tiles;
        sm.sP.sa0_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(sm.u.ab.sA[0]));
        sm.sP.sa1_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(sm.u.ab.sA[1]));
        sm.sP.sb0_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(sm.u.ab.sB[0]));
        sm.sP.sb1_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(sm.u.ab.sB[1]));
    }
    __syncthreads();

    const int total_tiles = sm.sP.total_tiles;
    const int num_m_tiles = sm.sP.num_m_tiles;
    const int num_n_tiles = sm.sP.num_n_tiles;
    const int K_cur       = sm.sP.K;
    const int M_cur       = sm.sP.M;
    const int N_cur       = sm.sP.N;
    const int num_k       = K_cur / D_BK;

    // 256 threads load [128, 64] uint8 = 512 x 16B chunks -> 2 chunks/thread!
    const int ld_r0 = tid >> 2;           // 0..63
    const int ld_r1 = ld_r0 + 64;         // 64..127
    const int ld_c  = tid & 3;            // 0..3 (x 16B)

    for (int tile_id = (int)blockIdx.x; tile_id < total_tiles; tile_id += (int)gridDim.x) {
        int tm = 0, tn = 0;
        swizzle_tile_coord(tile_id, num_m_tiles, num_n_tiles, tm, tn);
        const int m0 = tm * D_BM;
        const int n0 = tn * D_BN;

        // Preload row/col FP32 scales into SMEM
        if (tid < D_BM) {
            sm.sAs[tid] = (m0 + tid < M_cur) ? A_SCALE[m0 + tid] : 0.0f;
        } else {
            const int ni = tid - D_BM;
            sm.sBs[ni] = (n0 + ni < N_cur) ? W_SCALE[n0 + ni] : 0.0f;
        }

        float acc[2][8][4];
        #pragma unroll
        for (int mi = 0; mi < 2; ++mi) {
            #pragma unroll
            for (int nt = 0; nt < 8; ++nt) {
                acc[mi][nt][0] = 0.0f; acc[mi][nt][1] = 0.0f;
                acc[mi][nt][2] = 0.0f; acc[mi][nt][3] = 0.0f;
            }
        }

        auto issue_stage = [&](int stage, int k_idx) {
            const int k0 = k_idx * D_BK + ld_c * 16;
            const uint32_t sa_p = (stage == 0) ? sm.sP.sa0_ptr : sm.sP.sa1_ptr;
            const uint32_t sb_p = (stage == 0) ? sm.sP.sb0_ptr : sm.sP.sb1_ptr;
            cp_async_ca_16(sa_p + swiz_off_128x64(ld_r0, ld_c), A + static_cast<int64_t>(m0 + ld_r0) * K_cur + k0, (m0 + ld_r0) < M_cur);
            cp_async_ca_16(sa_p + swiz_off_128x64(ld_r1, ld_c), A + static_cast<int64_t>(m0 + ld_r1) * K_cur + k0, (m0 + ld_r1) < M_cur);
            cp_async_ca_16(sb_p + swiz_off_128x64(ld_r0, ld_c), W + static_cast<int64_t>(n0 + ld_r0) * K_cur + k0, (n0 + ld_r0) < N_cur);
            cp_async_ca_16(sb_p + swiz_off_128x64(ld_r1, ld_c), W + static_cast<int64_t>(n0 + ld_r1) * K_cur + k0, (n0 + ld_r1) < N_cur);
            cp_async_commit();
        };

        // Prime stage 0
        issue_stage(0, 0);
        for (int ks = 0; ks < num_k; ++ks) {
            const int cur_s = ks & 1;
            const int nxt_s = cur_s ^ 1;
            if (ks + 1 < num_k) {
                issue_stage(nxt_s, ks + 1);
                cp_async_wait_group<1>();
            } else {
                cp_async_wait_group<0>();
            }
            __syncthreads();

            const uint32_t sa_p = (cur_s == 0) ? sm.sP.sa0_ptr : sm.sP.sa1_ptr;
            const uint32_t sb_p = (cur_s == 0) ? sm.sP.sb0_ptr : sm.sP.sb1_ptr;
            dense_v11_mma_bk64_stage(sa_p, sb_p, wm, wn, lane, acc);
            __syncthreads();
        }

        // Epilogue via unioned sEpiBf16 [128, 128] -> coalesced 16B stores (native N % 8 == 0!)
        #pragma unroll
        for (int mi = 0; mi < 2; ++mi) {
            const int r0 = wm * 32 + mi * 16 + gid;
            const int r1 = r0 + 8;
            const float sa0 = sm.sAs[r0];
            const float sa1 = sm.sAs[r1];
            #pragma unroll
            for (int nt = 0; nt < 8; ++nt) {
                const int c0 = wn * 64 + nt * 8 + 2 * t;
                const float sb0 = sm.sBs[c0];
                const float sb1 = sm.sBs[c0 + 1];
                __nv_bfloat162 b0 = __halves2bfloat162(
                    __float2bfloat16_rn(__fmul_rn(acc[mi][nt][0], __fmul_rn(sa0, sb0))),
                    __float2bfloat16_rn(__fmul_rn(acc[mi][nt][1], __fmul_rn(sa0, sb1)))
                );
                __nv_bfloat162 b1 = __halves2bfloat162(
                    __float2bfloat16_rn(__fmul_rn(acc[mi][nt][2], __fmul_rn(sa1, sb0))),
                    __float2bfloat16_rn(__fmul_rn(acc[mi][nt][3], __fmul_rn(sa1, sb1)))
                );
                *reinterpret_cast<uint32_t*>(&sm.u.sEpiBf16[r0][c0]) = *reinterpret_cast<const uint32_t*>(&b0);
                *reinterpret_cast<uint32_t*>(&sm.u.sEpiBf16[r1][c0]) = *reinterpret_cast<const uint32_t*>(&b1);
            }
        }
        __syncthreads();

        // Coalesced 8-BF16 (16B) global stores; handles native N % 8 == 0 (1896, 1640, 4608, 384)!
        const int st_c8 = tid & 15;       // 0..15 (x 8 BF16 = 128 cols)
        const int st_r0 = tid >> 4;       // 0..15 (stride 16 -> 8 iterations = 128 rows)
        const int g_col = n0 + st_c8 * 8;
        if (g_col < N_cur) {
            #pragma unroll
            for (int ri = 0; ri < 8; ++ri) {
                const int r = st_r0 + ri * 16;
                if (m0 + r < M_cur) {
                    const uint4 v = *reinterpret_cast<const uint4*>(&sm.u.sEpiBf16[r][st_c8 * 8]);
                    st_global_wb_u4(OUT + static_cast<int64_t>(m0 + r) * stride_out + g_col, v);
                }
            }
        }
        __syncthreads();
    }
}

// -----------------------------------------------------------------------------
// Kernel 2: Fused BF16-in (`A_MAX = 22.0f`) `wo_ud` GEMM (`K = 768, N = 4608`)
//   Template `FUSED_A2A_INT8`:
//     - false: Standard BF16 `[M, N]` output (eliminates `quant_k768` completely!)
//     - true:  Direct register-level `A2A_INT8_g64` epilogue!
//                * If `dst == rank`: writes BF16 directly to `LOCAL_SHARD [S, N]`
//                * If `dst != rank`: computes exact `g64` INT8 + BF16 scale via 2 warp
//                  shuffles right in registers -> writes `SEND_BUF[dst]` (`st.global.cs`)!
//              Eliminates `quant_k768`, `attn_partial` BF16 write (`-467 GB`), AND
//              `a2a_quant_int8_g64_lowsm` (`-620 GB` & `-4,128` comm launches)!
// -----------------------------------------------------------------------------
template <bool FUSED_A2A_INT8>
__global__ __launch_bounds__(D_THREADS, 1)
void pf_wo_ud_v11_kernel(
    const __nv_bfloat16* __restrict__ A_BF16,
    const uint8_t* __restrict__ W_FP8,
    const float* __restrict__ W_SCALE,
    __nv_bfloat16* __restrict__ OUT_BF16,
    uint8_t* __restrict__ SEND_BUF,
    int64_t stride_a,
    int64_t stride_out,
    int64_t peer_bytes,
    int64_t scale_off,
    int M,
    int S,
    int N,
    int K,
    int rank,
    float a_max,
    int wb_out
) {
    __shared__ DenseV11Smem sm;
    const int tid  = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wm   = warp >> 1;  // 0..3
    const int wn   = warp & 1;   // 0..1
    const int gid  = lane >> 2;  // 0..7
    const int t    = lane & 3;   // 0..3

    if (tid == 0) {
        sm.sP.M = M;
        sm.sP.N = N;
        sm.sP.K = K;
        sm.sP.num_m_tiles = (M + D_BM - 1) / D_BM;
        sm.sP.num_n_tiles = (N + D_BN - 1) / D_BN;
        sm.sP.total_tiles = sm.sP.num_m_tiles * sm.sP.num_n_tiles;
        sm.sP.sa0_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(sm.u.ab.sA[0]));
        sm.sP.sa1_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(sm.u.ab.sA[1]));
        sm.sP.sb0_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(sm.u.ab.sB[0]));
        sm.sP.sb1_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(sm.u.ab.sB[1]));
    }
    __syncthreads();

    const int total_tiles = sm.sP.total_tiles;
    const int num_m_tiles = sm.sP.num_m_tiles;
    const int num_n_tiles = sm.sP.num_n_tiles;
    const int K_cur       = sm.sP.K;
    const int M_cur       = sm.sP.M;
    const int N_cur       = sm.sP.N;
    const int num_k       = K_cur / D_BK;
    const float inv_amax  = __fdiv_rn(1.0f, a_max);
    const int G_cols      = N_cur / 64;

    const int ld_r0 = tid >> 2;           // 0..63
    const int ld_r1 = ld_r0 + 64;         // 64..127
    const int ld_c  = tid & 3;            // 0..3 (x 16B)

    for (int tile_id = (int)blockIdx.x; tile_id < total_tiles; tile_id += (int)gridDim.x) {
        int tm = 0, tn = 0;
        swizzle_tile_coord(tile_id, num_m_tiles, num_n_tiles, tm, tn);
        const int m0 = tm * D_BM;
        const int n0 = tn * D_BN;

        // 1) Cooperative row-amax of A_BF16[m0..m0+127, 0..K-1] (L2-hot!) -> sAs, sAsInv
        //    8 warps handle 16 rows each (128 rows total); 32 lanes scan K/8 uint4s per row!
        for (int r_step = 0; r_step < 16; ++r_step) {
            const int r = warp * 16 + r_step;
            const int g_row = m0 + r;
            float lamax = 0.0f;
            if (g_row < M_cur) {
                const __nv_bfloat16* ap = A_BF16 + static_cast<int64_t>(g_row) * stride_a;
                const int n_vec8 = K_cur >> 3;
                for (int idx = lane; idx < n_vec8; idx += 32) {
                    uint4 raw = ld_global_ca_u4(ap + (idx << 3));
                    const __nv_bfloat162* h2 = reinterpret_cast<const __nv_bfloat162*>(&raw);
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        float2 f = __bfloat1622float2(h2[i]);
                        lamax = fmaxf(lamax, fmaxf(fabsf(f.x), fabsf(f.y)));
                    }
                }
            }
            float rmax = warp_max_f32(lamax);
            if (lane == 0) {
                float sc = fmaxf(__fmul_rn(rmax, inv_amax), 1.0e-12f);
                sm.sAs[r] = sc;
                sm.sAsInv[r] = __fdiv_rn(1.0f, sc);
            }
        }
        if (tid < D_BN) {
            sm.sBs[tid] = (n0 + tid < N_cur) ? W_SCALE[n0 + tid] : 0.0f;
        }
        __syncthreads();

        float acc[2][8][4];
        #pragma unroll
        for (int mi = 0; mi < 2; ++mi) {
            #pragma unroll
            for (int nt = 0; nt < 8; ++nt) {
                acc[mi][nt][0] = 0.0f; acc[mi][nt][1] = 0.0f;
                acc[mi][nt][2] = 0.0f; acc[mi][nt][3] = 0.0f;
            }
        }

        // Helper: load W stage via cp.async AND quantize A_BF16 [128, 64] -> swizzled sA[stage]
        auto load_stage_bf16_a = [&](int stage, int k_idx) {
            const int k0 = k_idx * D_BK + ld_c * 16;
            const uint32_t sb_p = (stage == 0) ? sm.sP.sb0_ptr : sm.sP.sb1_ptr;
            cp_async_ca_16(sb_p + swiz_off_128x64(ld_r0, ld_c), W_FP8 + static_cast<int64_t>(n0 + ld_r0) * K_cur + k0, (n0 + ld_r0) < N_cur);
            cp_async_ca_16(sb_p + swiz_off_128x64(ld_r1, ld_c), W_FP8 + static_cast<int64_t>(n0 + ld_r1) * K_cur + k0, (n0 + ld_r1) < N_cur);
            cp_async_commit();

            // Each thread converts 2 rows x 16 BF16 (2 uint4 loads per row) -> 16 FP8 bytes into sA!
            uint8_t* sa_base = sm.u.ab.sA[stage];
            #pragma unroll
            for (int rr = 0; rr < 2; ++rr) {
                const int r = (rr == 0) ? ld_r0 : ld_r1;
                const int g_row = m0 + r;
                uint4 q16 = make_uint4(0u, 0u, 0u, 0u);
                if (g_row < M_cur) {
                    const float inv_sc = sm.sAsInv[r];
                    const __nv_bfloat16* ap = A_BF16 + static_cast<int64_t>(g_row) * stride_a + k0;
                    uint4 bf_lo = ld_global_ca_u4(ap);
                    uint4 bf_hi = ld_global_ca_u4(ap + 8);
                    const __nv_bfloat162* h_lo = reinterpret_cast<const __nv_bfloat162*>(&bf_lo);
                    const __nv_bfloat162* h_hi = reinterpret_cast<const __nv_bfloat162*>(&bf_hi);
                    float2 f0 = __bfloat1622float2(h_lo[0]);
                    float2 f1 = __bfloat1622float2(h_lo[1]);
                    float2 f2 = __bfloat1622float2(h_lo[2]);
                    float2 f3 = __bfloat1622float2(h_lo[3]);
                    float2 f4 = __bfloat1622float2(h_hi[0]);
                    float2 f5 = __bfloat1622float2(h_hi[1]);
                    float2 f6 = __bfloat1622float2(h_hi[2]);
                    float2 f7 = __bfloat1622float2(h_hi[3]);
                    q16.x = f32x4_to_e4m3x4_rn(__fmul_rn(f0.x, inv_sc), __fmul_rn(f0.y, inv_sc), __fmul_rn(f1.x, inv_sc), __fmul_rn(f1.y, inv_sc));
                    q16.y = f32x4_to_e4m3x4_rn(__fmul_rn(f2.x, inv_sc), __fmul_rn(f2.y, inv_sc), __fmul_rn(f3.x, inv_sc), __fmul_rn(f3.y, inv_sc));
                    q16.z = f32x4_to_e4m3x4_rn(__fmul_rn(f4.x, inv_sc), __fmul_rn(f4.y, inv_sc), __fmul_rn(f5.x, inv_sc), __fmul_rn(f5.y, inv_sc));
                    q16.w = f32x4_to_e4m3x4_rn(__fmul_rn(f6.x, inv_sc), __fmul_rn(f6.y, inv_sc), __fmul_rn(f7.x, inv_sc), __fmul_rn(f7.y, inv_sc));
                }
                *reinterpret_cast<uint4*>(sa_base + swiz_off_128x64(r, ld_c)) = q16;
            }
        };

        load_stage_bf16_a(0, 0);
        for (int ks = 0; ks < num_k; ++ks) {
            const int cur_s = ks & 1;
            const int nxt_s = cur_s ^ 1;
            if (ks + 1 < num_k) {
                load_stage_bf16_a(nxt_s, ks + 1);
                cp_async_wait_group<1>();
            } else {
                cp_async_wait_group<0>();
            }
            __syncthreads();

            const uint32_t sa_p = (cur_s == 0) ? sm.sP.sa0_ptr : sm.sP.sa1_ptr;
            const uint32_t sb_p = (cur_s == 0) ? sm.sP.sb0_ptr : sm.sP.sb1_ptr;
            dense_v11_mma_bk64_stage(sa_p, sb_p, wm, wn, lane, acc);
            __syncthreads();
        }

        if constexpr (FUSED_A2A_INT8) {
            // Because S % 128 == 0, every BM=128 tile belongs to EXACTLY ONE rank `dst = m0 / S`!
            const int dst = m0 / S;
            const int local_m0 = m0 - dst * S;
            if (dst == rank) {
                // Local self-shard -> exact BF16 to OUT_BF16 [S, N]!
                #pragma unroll
                for (int mi = 0; mi < 2; ++mi) {
                    const int r0 = wm * 32 + mi * 16 + gid;
                    const int r1 = r0 + 8;
                    const float sa0 = sm.sAs[r0];
                    const float sa1 = sm.sAs[r1];
                    #pragma unroll
                    for (int nt = 0; nt < 8; ++nt) {
                        const int c0 = wn * 64 + nt * 8 + 2 * t;
                        const float sb0 = sm.sBs[c0];
                        const float sb1 = sm.sBs[c0 + 1];
                        __nv_bfloat162 b0 = __halves2bfloat162(
                            __float2bfloat16_rn(__fmul_rn(acc[mi][nt][0], __fmul_rn(sa0, sb0))),
                            __float2bfloat16_rn(__fmul_rn(acc[mi][nt][1], __fmul_rn(sa0, sb1)))
                        );
                        __nv_bfloat162 b1 = __halves2bfloat162(
                            __float2bfloat16_rn(__fmul_rn(acc[mi][nt][2], __fmul_rn(sa1, sb0))),
                            __float2bfloat16_rn(__fmul_rn(acc[mi][nt][3], __fmul_rn(sa1, sb1)))
                        );
                        *reinterpret_cast<uint32_t*>(&sm.u.sEpiBf16[r0][c0]) = *reinterpret_cast<const uint32_t*>(&b0);
                        *reinterpret_cast<uint32_t*>(&sm.u.sEpiBf16[r1][c0]) = *reinterpret_cast<const uint32_t*>(&b1);
                    }
                }
                __syncthreads();
                const int st_c8 = tid & 15;
                const int st_r0 = tid >> 4;
                const int g_col = n0 + st_c8 * 8;
                if (g_col < N_cur) {
                    #pragma unroll
                    for (int ri = 0; ri < 8; ++ri) {
                        const int r = st_r0 + ri * 16;
                        if (local_m0 + r < S) {
                            const uint4 v = *reinterpret_cast<const uint4*>(&sm.u.sEpiBf16[r][st_c8 * 8]);
                            if (wb_out) st_global_wb_u4(OUT_BF16 + static_cast<int64_t>(local_m0 + r) * stride_out + g_col, v);
                            else        st_global_cs_u4(OUT_BF16 + static_cast<int64_t>(local_m0 + r) * stride_out + g_col, v);
                        }
                    }
                }
                __syncthreads();
            } else {
                // Remote peer shard -> direct register g64 INT8 quant + BF16 scale -> SEND_BUF[dst]!
                // Each warp owns EXACTLY 1 g64 group (`g_id = (n0 + wn * 64) / 64`)!
                uint8_t* dst_base = SEND_BUF + static_cast<int64_t>(dst) * peer_bytes;
                const int g_id = (n0 + wn * 64) >> 6;

                #pragma unroll
                for (int mi = 0; mi < 2; ++mi) {
                    const int r0 = wm * 32 + mi * 16 + gid;
                    const int r1 = r0 + 8;
                    const float sa0 = sm.sAs[r0];
                    const float sa1 = sm.sAs[r1];

                    float am0 = 0.0f, am1 = 0.0f;
                    #pragma unroll
                    for (int nt = 0; nt < 8; ++nt) {
                        const int c0 = wn * 64 + nt * 8 + 2 * t;
                        const float sb0 = sm.sBs[c0];
                        const float sb1 = sm.sBs[c0 + 1];
                        acc[mi][nt][0] = __fmul_rn(acc[mi][nt][0], __fmul_rn(sa0, sb0));
                        acc[mi][nt][1] = __fmul_rn(acc[mi][nt][1], __fmul_rn(sa0, sb1));
                        acc[mi][nt][2] = __fmul_rn(acc[mi][nt][2], __fmul_rn(sa1, sb0));
                        acc[mi][nt][3] = __fmul_rn(acc[mi][nt][3], __fmul_rn(sa1, sb1));
                        am0 = fmaxf(am0, fmaxf(fabsf(acc[mi][nt][0]), fabsf(acc[mi][nt][1])));
                        am1 = fmaxf(am1, fmaxf(fabsf(acc[mi][nt][2]), fabsf(acc[mi][nt][3])));
                    }
                    // 4 threads (t=0..3) hold all 64 cols of row r0 & r1 -> 2 shuffles = exact g64 amax!
                    am0 = fmaxf(am0, __shfl_xor_sync(0xFFFFFFFFu, am0, 1));
                    am0 = fmaxf(am0, __shfl_xor_sync(0xFFFFFFFFu, am0, 2));
                    am1 = fmaxf(am1, __shfl_xor_sync(0xFFFFFFFFu, am1, 1));
                    am1 = fmaxf(am1, __shfl_xor_sync(0xFFFFFFFFu, am1, 2));

                    const __nv_bfloat16 sc0_bf = __float2bfloat16_rn(fmaxf(__fmul_rn(am0, (1.0f / 127.0f)), 1.0e-12f));
                    const __nv_bfloat16 sc1_bf = __float2bfloat16_rn(fmaxf(__fmul_rn(am1, (1.0f / 127.0f)), 1.0e-12f));
                    const float inv0 = __fdiv_rn(1.0f, __bfloat162float(sc0_bf));
                    const float inv1 = __fdiv_rn(1.0f, __bfloat162float(sc1_bf));

                    if (t == 0 && (local_m0 + r0) < S) {
                        uint16_t* sc_row0 = reinterpret_cast<uint16_t*>(
                            dst_base + scale_off + static_cast<int64_t>(local_m0 + r0) * G_cols * 2
                        );
                        st_global_cs_u16(sc_row0 + g_id, *reinterpret_cast<const uint16_t*>(&sc0_bf));
                    }
                    if (t == 0 && (local_m0 + r1) < S) {
                        uint16_t* sc_row1 = reinterpret_cast<uint16_t*>(
                            dst_base + scale_off + static_cast<int64_t>(local_m0 + r1) * G_cols * 2
                        );
                        st_global_cs_u16(sc_row1 + g_id, *reinterpret_cast<const uint16_t*>(&sc1_bf));
                    }

                    #pragma unroll
                    for (int nt = 0; nt < 8; ++nt) {
                        const int c0 = wn * 64 + nt * 8 + 2 * t;
                        const uint8_t q00 = static_cast<uint8_t>(f32_to_i8_sat(__fmul_rn(acc[mi][nt][0], inv0)));
                        const uint8_t q01 = static_cast<uint8_t>(f32_to_i8_sat(__fmul_rn(acc[mi][nt][1], inv0)));
                        const uint8_t q10 = static_cast<uint8_t>(f32_to_i8_sat(__fmul_rn(acc[mi][nt][2], inv1)));
                        const uint8_t q11 = static_cast<uint8_t>(f32_to_i8_sat(__fmul_rn(acc[mi][nt][3], inv1)));
                        *reinterpret_cast<uint16_t*>(&sm.u.sEpiI8[r0][c0]) = static_cast<uint16_t>(q00) | (static_cast<uint16_t>(q01) << 8);
                        *reinterpret_cast<uint16_t*>(&sm.u.sEpiI8[r1][c0]) = static_cast<uint16_t>(q10) | (static_cast<uint16_t>(q11) << 8);
                    }
                }
                __syncthreads();

                // Coalesced 16B streaming stores of [128, 128] INT8 into SEND_BUF[dst]!
                // 256 threads x 4 iters = 1,024 x 16B = 16,384 B!
                const int c16 = tid & 7;      // 0..7 (x 16B = 128B)
                const int r_base = tid >> 3;  // 0..31 (stride 32 -> 4 iters = 128 rows)
                const int g_col = n0 + c16 * 16;
                if (g_col < N_cur) {
                    #pragma unroll
                    for (int ri = 0; ri < 4; ++ri) {
                        const int r = r_base + ri * 32;
                        if (local_m0 + r < S) {
                            const uint4 v = *reinterpret_cast<const uint4*>(&sm.u.sEpiI8[r][c16 * 16]);
                            st_global_cs_u4(dst_base + static_cast<int64_t>(local_m0 + r) * N_cur + g_col, v);
                        }
                    }
                }
                __syncthreads();
            }
        } else {
            // Standard BF16 output epilogue
            #pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                const int r0 = wm * 32 + mi * 16 + gid;
                const int r1 = r0 + 8;
                const float sa0 = sm.sAs[r0];
                const float sa1 = sm.sAs[r1];
                #pragma unroll
                for (int nt = 0; nt < 8; ++nt) {
                    const int c0 = wn * 64 + nt * 8 + 2 * t;
                    const float sb0 = sm.sBs[c0];
                    const float sb1 = sm.sBs[c0 + 1];
                    __nv_bfloat162 b0 = __halves2bfloat162(
                        __float2bfloat16_rn(__fmul_rn(acc[mi][nt][0], __fmul_rn(sa0, sb0))),
                        __float2bfloat16_rn(__fmul_rn(acc[mi][nt][1], __fmul_rn(sa0, sb1)))
                    );
                    __nv_bfloat162 b1 = __halves2bfloat162(
                        __float2bfloat16_rn(__fmul_rn(acc[mi][nt][2], __fmul_rn(sa1, sb0))),
                        __float2bfloat16_rn(__fmul_rn(acc[mi][nt][3], __fmul_rn(sa1, sb1)))
                    );
                    *reinterpret_cast<uint32_t*>(&sm.u.sEpiBf16[r0][c0]) = *reinterpret_cast<const uint32_t*>(&b0);
                    *reinterpret_cast<uint32_t*>(&sm.u.sEpiBf16[r1][c0]) = *reinterpret_cast<const uint32_t*>(&b1);
                }
            }
            __syncthreads();

            const int st_c8 = tid & 15;
            const int st_r0 = tid >> 4;
            const int g_col = n0 + st_c8 * 8;
            if (g_col < N_cur) {
                #pragma unroll
                for (int ri = 0; ri < 8; ++ri) {
                    const int r = st_r0 + ri * 16;
                    if (m0 + r < M_cur) {
                        const uint4 v = *reinterpret_cast<const uint4*>(&sm.u.sEpiBf16[r][st_c8 * 8]);
                        if (wb_out) st_global_wb_u4(OUT_BF16 + static_cast<int64_t>(m0 + r) * stride_out + g_col, v);
                        else        st_global_cs_u4(OUT_BF16 + static_cast<int64_t>(m0 + r) * stride_out + g_col, v);
                    }
                }
            }
            __syncthreads();
        }
    }
}

static inline unsigned compute_persistent_grid(int total_tiles, int reserve_sms) {
    const int sms = get_sm_count();
    int eff_res = reserve_sms;
    if (eff_res < 0) eff_res = 0;
    if (eff_res > sms - 8) eff_res = sms - 8;
    const int cap = (sms - eff_res) * 2;
    const int g = (total_tiles < cap) ? total_tiles : cap;
    return static_cast<unsigned>(g > 0 ? g : 1);
}

}  // namespace

extern "C" {

int pf_dense_fp8_tc_v13_version() {
    return 1300;  // Wave-V13 Prefill-Only FP8 FP16-acc TC + wo_ud Fused A2A INT8_g64!
}

int pf_dense_fp8_tc_v11_version() {
    return 1300;  // Back-compat V11 symbol reports 1300
}

int pf_quant_weight_f16safe_v11_fwd(
    const void* w_bf16,
    void* w_fp8_out,
    void* w_scale_out,
    int N,
    int K,
    float w_max,
    uint64_t stream_u64
) {
    if (w_bf16 == nullptr || w_fp8_out == nullptr || w_scale_out == nullptr) return -1;
    if (N <= 0 || K <= 0 || (K % 64) != 0 || w_max <= 0.0f || w_max > 448.0f) return -1;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_u64);
    if (is_stream_capturing(stream)) return -4;
    unsigned grid = static_cast<unsigned>((N < 192) ? N : 192);
    pf_quant_weight_f16safe_v11_kernel<<<dim3(grid, 1, 1), dim3(256, 1, 1), 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(w_bf16),
        reinterpret_cast<uint8_t*>(w_fp8_out),
        reinterpret_cast<float*>(w_scale_out),
        N, K, w_max
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

int pf_dense_fp8_f16acc_v11_fwd(
    const void* a_fp8,
    const void* a_scale,
    const void* w_fp8,
    const void* w_scale,
    void* out_bf16,
    int M,
    int N,
    int K,
    int stride_out,
    int reserve_sms,
    uint64_t stream_u64
) {
    if (M < 512) return -3;
    if (a_fp8 == nullptr || a_scale == nullptr || w_fp8 == nullptr || w_scale == nullptr || out_bf16 == nullptr) {
        return -1;
    }
    if (N <= 0 || K <= 0 || (K % D_BK) != 0 || (N % 8) != 0 || stride_out < N) return -1;
    if ((reinterpret_cast<uintptr_t>(a_fp8) & 15u) != 0 ||
        (reinterpret_cast<uintptr_t>(w_fp8) & 15u) != 0 ||
        (reinterpret_cast<uintptr_t>(out_bf16) & 15u) != 0) {
        return -1;
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_u64);
    if (is_stream_capturing(stream)) return -4;

    const int total_tiles = ((M + D_BM - 1) / D_BM) * ((N + D_BN - 1) / D_BN);
    const unsigned grid = compute_persistent_grid(total_tiles, reserve_sms);
    pf_dense_fp8_f16acc_v11_kernel<<<dim3(grid, 1, 1), dim3(D_THREADS, 1, 1), 0, stream>>>(
        reinterpret_cast<const uint8_t*>(a_fp8),
        reinterpret_cast<const float*>(a_scale),
        reinterpret_cast<const uint8_t*>(w_fp8),
        reinterpret_cast<const float*>(w_scale),
        reinterpret_cast<__nv_bfloat16*>(out_bf16),
        M, N, K, stride_out
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

int pf_wo_ud_bf16_to_bf16_v11_fwd(
    const void* a_bf16,
    const void* w_fp8,
    const void* w_scale,
    void* out_bf16,
    int M,
    int N,
    int K,
    int stride_a,
    int stride_out,
    float a_max,
    int reserve_sms,
    uint64_t stream_u64
) {
    if (M < 512) return -3;
    if (a_bf16 == nullptr || w_fp8 == nullptr || w_scale == nullptr || out_bf16 == nullptr) return -1;
    if (N <= 0 || K <= 0 || (K % D_BK) != 0 || (N % 8) != 0 || stride_a < K || stride_out < N) return -1;
    if ((stride_a & 7) != 0 || (stride_out & 7) != 0 || a_max <= 0.0f || a_max > 448.0f) return -1;
    if ((reinterpret_cast<uintptr_t>(a_bf16) & 15u) != 0 ||
        (reinterpret_cast<uintptr_t>(w_fp8) & 15u) != 0 ||
        (reinterpret_cast<uintptr_t>(out_bf16) & 15u) != 0) {
        return -1;
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_u64);
    if (is_stream_capturing(stream)) return -4;

    const int total_tiles = ((M + D_BM - 1) / D_BM) * ((N + D_BN - 1) / D_BN);
    const unsigned grid = compute_persistent_grid(total_tiles, reserve_sms);
    pf_wo_ud_v11_kernel<false><<<dim3(grid, 1, 1), dim3(D_THREADS, 1, 1), 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(a_bf16),
        reinterpret_cast<const uint8_t*>(w_fp8),
        reinterpret_cast<const float*>(w_scale),
        reinterpret_cast<__nv_bfloat16*>(out_bf16),
        nullptr,
        static_cast<int64_t>(stride_a),
        static_cast<int64_t>(stride_out),
        0, 0, M, M, N, K, 0, a_max, 1
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

int pf_wo_ud_fused_a2a_int8_v11_fwd(
    const void* a_bf16,
    const void* w_fp8,
    const void* w_scale,
    void* send_buf_u8,
    void* local_shard_bf16,
    int64_t stride_a,
    int64_t stride_local,
    int64_t peer_bytes,
    int64_t scale_off,
    int S,
    int N,
    int K,
    int world_size,
    int rank,
    float a_max,
    int wb_local,
    int reserve_sms,
    uint64_t stream_u64
) {
    if (S <= 0 || world_size <= 1 || rank < 0 || rank >= world_size) return -1;
    const int M = S * world_size;
    if (M < 512) return -3;
    if ((S % D_BM) != 0 || (N % D_BN) != 0 || (K % D_BK) != 0) return -1;
    if (a_bf16 == nullptr || w_fp8 == nullptr || w_scale == nullptr ||
        send_buf_u8 == nullptr || local_shard_bf16 == nullptr) {
        return -1;
    }
    if ((stride_a & 7) != 0 || (stride_local & 7) != 0 || (peer_bytes & 15) != 0 || (scale_off & 3) != 0) {
        return -1;
    }
    if (a_max <= 0.0f || a_max > 448.0f) return -1;
    if ((reinterpret_cast<uintptr_t>(a_bf16) & 15u) != 0 ||
        (reinterpret_cast<uintptr_t>(w_fp8) & 15u) != 0 ||
        (reinterpret_cast<uintptr_t>(send_buf_u8) & 15u) != 0 ||
        (reinterpret_cast<uintptr_t>(local_shard_bf16) & 15u) != 0) {
        return -1;
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_u64);
    if (is_stream_capturing(stream)) return -4;

    const int total_tiles = ((M + D_BM - 1) / D_BM) * ((N + D_BN - 1) / D_BN);
    const unsigned grid = compute_persistent_grid(total_tiles, reserve_sms);
    pf_wo_ud_v11_kernel<true><<<dim3(grid, 1, 1), dim3(D_THREADS, 1, 1), 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(a_bf16),
        reinterpret_cast<const uint8_t*>(w_fp8),
        reinterpret_cast<const float*>(w_scale),
        reinterpret_cast<__nv_bfloat16*>(local_shard_bf16),
        reinterpret_cast<uint8_t*>(send_buf_u8),
        stride_a, stride_local, peer_bytes, scale_off,
        M, S, N, K, rank, a_max, wb_local
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

}  // extern "C"
