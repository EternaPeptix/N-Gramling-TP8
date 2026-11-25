// SPDX-License-Identifier: Apache-2.0
// Wave-V14 SM121a (GB10 Blackwell) Ultra-Low-Latency 4-Bit Panel GEMM for C=1..8 DSpark Drafter
// Target: NVIDIA GB10 SM121a (48 SMs, 244 GB/s LPDDR5X DRAM, 24 MiB L2, 99 KiB SMEM/SM)
// Build:
//   nvcc -O3 -std=c++17 -gencode arch=compute_121a,code=sm_121a --fmad=false \
//        --shared -Xcompiler -fPIC --ptxas-options=-v \
//        -o libdrafter_u4_c18_v14.so drafter_u4_c18_v14.cu
//
// Architectural upgrades over Wave-V9 `drafter_u4_tc_v9.cu` (which forced MIN_M=9 & Triton fallback @ M=1..8):
//  1. EXACT Native IEEE-754 FP16 Weight Dequant & FP16 MMA (ZERO BF16 truncation of SCP/ZPP/W!):
//     - v9 truncated FP16 `SCP`/`ZPP` (10-bit mantissa) down to BF16 (7-bit mantissa -> cos_sim 0.999978),
//       forcing V12 to lock `INKLING_DRAFT_U4_V9_MIN_M=9` so B=1 (M=7,8) stayed on Triton for 1dcfb306 safety.
//     - v14 keeps `SCP`/`ZPP` in native FP16 (0 conversion!), unpacks 4-bit nibbles via exact 0x6400 FP16 bias
//       (`prmt.b32` + `lop3.b32(0xEA)` + `add.f16x2(-1024)` + `fma.rn.f16x2`), matching Triton `_u4_panel_kernel`
//       dequant to literal 0.0 ULP on BOTH asymmetric `u4p32/u4p64` and symmetric `i4p32/i4p64` (`zp = sc * -7.5`)!
//  2. Three Ultra-Specialized Kernel Tiers eliminating 16-row waste across ALL M=1..64:
//     - Tier 1 (`M = 1..4`): Zero-SMEM, zero-barrier, zero-shuffle-in-K-loop warp-specialized SIMT/FP16-exact
//       kernel (`M_CAP in {1,2,4}`). Each `(grp, lt)` lane owns 32 contiguous weights aligned 1:1 with `GS=32/64`
//       (`WP` + `SCP`/`ZPP` double-buffered in registers, 0 `sX` SMEM, 0 `__syncthreads()` in K-loop!).
//     - Tier 2 (`M = 5..8`, e.g. C=1 M=7 draft_head/rnn_ctx & M=8 all 25 trunk+ctx GEMMs): Dedicated `BM = 8`
//       FP16 Tensor-Core kernel with `sX[2][8][136]` (4.25 KiB SMEM = 50% of v9!), `ldmatrix.sync.aligned.x2`
//       (50% SMEM read traffic vs v9's `x4`!), 1 `uint4`/thr `X` load (0 loads for padded rows), and half-size
//       `PART` split-K footprint (`BM_TILE = 8`)!
//     - Tier 3 (`M = 9..64`, C=2..8 verify/trunk/head/rnn_ctx): Upgraded `MT = 1..4` FP16-exact TC kernel with
//       `BN_CTA = 64` (8-warp, 393 CTAs) unlocked down to `MT = 2` (`M = 17..32`, halving `draft_head` L2 `X`
//       traffic at C=2..4!) and predicated split-K `PART` stores (`m_row < M`).
//  3. Zero-Stall Pipelined `SCP`/`ZPP` Register Preload & Single-Barrier K-Loop:
//     - Eliminates v9's synchronous `__ldg(SCP)`/`__ldg(ZPP)` stalls + 4x redundant loads inside the inner loop:
//       all 4 groups (`g = 0..3`) preload in parallel across `lt = 0..3` alongside `wp_nxt` BEFORE compute!
//     - Double-buffered FP16 `sX[2][BM_TILE][136]` cuts barriers in half (`1` `__syncthreads()`/iter vs v9's `2`!).
//  4. Coherent Single-Launch Split-K Finalize (`st.cg` + `__threadfence()` + `atom.add.acq_rel.gpu` + self-reset
//     `SEM`) + 0-`#206`-Hazard PDL (static `WP`/`SCP`/`ZPP` preload before `gdc_wait()`, `gdc_launch_dependents()`
//     strictly after `OUT` stores + `__syncthreads()` on every path) + 0 local arrays / 0 stack / 0 spills.
//  5. Exports `drafter_u4_c18_v14_version` (14), `drafter_u4_c18_v14_fwd` (31 args), plus 100% drop-in ABI
//     compatibility aliases `drafter_u4_tc_v9_version` (9) and `drafter_u4_tc_v9_fwd` (30 args).

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cmath>

namespace {

// ============================================================================
// Low-Level SM121a PTX Helpers (Strict --fmad=false, 0 Stack, Coherent L2)
// ============================================================================

static __device__ __forceinline__ void gdc_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;\n" ::: "memory");
#endif
}

static __device__ __forceinline__ void gdc_launch_dependents() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
#endif
}

static __device__ __forceinline__ uint4 ld_cs_u4(const void* ptr) {
    uint4 v;
#if defined(__CUDA_ARCH__)
    asm volatile("ld.global.cs.v4.u32 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(ptr));
#else
    v = *reinterpret_cast<const uint4*>(ptr);
#endif
    return v;
}

static __device__ __forceinline__ uint2 ld_cs_u2(const void* ptr) {
    uint2 v;
#if defined(__CUDA_ARCH__)
    asm volatile("ld.global.cs.v2.u32 {%0, %1}, [%2];\n"
                 : "=r"(v.x), "=r"(v.y)
                 : "l"(ptr));
#else
    v = *reinterpret_cast<const uint2*>(ptr);
#endif
    return v;
}

static __device__ __forceinline__ uint4 ld_ca_u4(const void* ptr) {
    uint4 v;
#if defined(__CUDA_ARCH__)
    asm volatile("ld.global.ca.v4.u32 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(ptr));
#else
    v = *reinterpret_cast<const uint4*>(ptr);
#endif
    return v;
}

static __device__ __forceinline__ uint16_t ld_ca_u16(const uint16_t* ptr) {
    uint16_t v;
#if defined(__CUDA_ARCH__)
    asm volatile("ld.global.ca.u16 %0, [%1];\n" : "=h"(v) : "l"(ptr));
#else
    v = *ptr;
#endif
    return v;
}

static __device__ __forceinline__ float ld_cg_f32(const float* ptr) {
    float v;
#if defined(__CUDA_ARCH__)
    asm volatile("ld.global.cg.f32 %0, [%1];\n" : "=f"(v) : "l"(ptr));
#else
    v = *ptr;
#endif
    return v;
}

static __device__ __forceinline__ void st_cg_f32(float* ptr, float v) {
#if defined(__CUDA_ARCH__)
    asm volatile("st.global.cg.f32 [%0], %1;\n" :: "l"(ptr), "f"(v) : "memory");
#else
    *ptr = v;
#endif
}

static __device__ __forceinline__ void st_cg_f32x2(float* ptr, float v0, float v1) {
#if defined(__CUDA_ARCH__)
    asm volatile("st.global.cg.v2.f32 [%0], {%1, %2};\n" :: "l"(ptr), "f"(v0), "f"(v1) : "memory");
#else
    ptr[0] = v0;
    ptr[1] = v1;
#endif
}

static __device__ __forceinline__ void st_cg_u32(uint32_t* ptr, uint32_t v) {
#if defined(__CUDA_ARCH__)
    asm volatile("st.global.cg.u32 [%0], %1;\n" :: "l"(ptr), "r"(v) : "memory");
#else
    *ptr = v;
#endif
}

static __device__ __forceinline__ uint32_t atomic_add_acq_rel_gpu(uint32_t* ptr, uint32_t val) {
    uint32_t old;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    asm volatile("atom.add.acq_rel.gpu.u32 %0, [%1], %2;\n"
                 : "=r"(old) : "l"(ptr), "r"(val) : "memory");
#else
    old = *ptr;
    *ptr = old + val;
#endif
    return old;
}

static __device__ __forceinline__ uint32_t smem_u32(const void* ptr) {
#if defined(__CUDA_ARCH__)
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
#else
    return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(ptr));
#endif
}

static __device__ __forceinline__ void ldmatrix_x2_b16(
    uint32_t& r0, uint32_t& r1, uint32_t smem_addr
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile(
        "ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(r0), "=r"(r1)
        : "r"(smem_addr)
    );
#else
    (void)smem_addr;
    r0 = r1 = 0u;
#endif
}

static __device__ __forceinline__ void ldmatrix_x4_b16(
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t smem_addr
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(smem_addr)
    );
#else
    (void)smem_addr;
    r0 = r1 = r2 = r3 = 0u;
#endif
}

static __device__ __forceinline__ void mma_m16n8k16_f16(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)
    );
#else
    (void)d0; (void)d1; (void)d2; (void)d3;
    (void)a0; (void)a1; (void)a2; (void)a3; (void)b0; (void)b1;
#endif
}

static __device__ __forceinline__ float bf16_bits_to_f32(uint16_t b) {
    union { uint32_t u; float f; } x;
    x.u = static_cast<uint32_t>(b) << 16;
    return x.f;
}

static __device__ __forceinline__ float f16_bits_to_f32(uint16_t h) {
#if defined(__CUDA_ARCH__)
    float f;
    asm("cvt.f32.f16 %0, %1;\n" : "=f"(f) : "h"(h));
    return f;
#else
    (void)h;
    return 0.0f;
#endif
}

static __device__ __forceinline__ uint16_t f32_to_bf16_bits(float v) {
#if defined(__CUDA_ARCH__)
    uint16_t b;
    asm("cvt.rn.bf16.f32 %0, %1;\n" : "=h"(b) : "f"(v));
    return b;
#else
    union { float f; uint32_t u; } x;
    x.f = v;
    uint32_t r = x.u + 0x7FFFu + ((x.u >> 16) & 1u);
    return static_cast<uint16_t>(r >> 16);
#endif
}

static __device__ __forceinline__ uint32_t pack_bf16x2_f32(float v0, float v1) {
#if defined(__CUDA_ARCH__)
    uint32_t out;
    asm("cvt.rn.bf16x2.f32 %0, %2, %1;\n" : "=r"(out) : "f"(v0), "f"(v1));
    return out;
#else
    return static_cast<uint32_t>(f32_to_bf16_bits(v0))
         | (static_cast<uint32_t>(f32_to_bf16_bits(v1)) << 16);
#endif
}

// Fast BF16x2 -> FP16x2 conversion (0.0 ULP identical to Triton x.to(tl.float16)!)
// Uses explicit lo/hi cvt.rn.f16.f32 + mov.b32 {h0, h1} (ptxas lowers to F2FP.F16.F32.PACK_AB on SM121a;
// 100% unambiguous lo/hi lane order and valid across every PTX ISA version).
static __device__ __forceinline__ uint32_t bf16x2_to_f16x2_fast(uint32_t b2) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    union { uint32_t u; float f; } x0, x1;
    x0.u = b2 << 16;
    x1.u = b2 & 0xFFFF0000u;
    uint32_t h2;
    asm("{ .reg .b16 h0, h1;\n"
        "  cvt.rn.f16.f32 h0, %1;\n"
        "  cvt.rn.f16.f32 h1, %2;\n"
        "  mov.b32 %0, {h0, h1}; }\n"
        : "=r"(h2) : "f"(x0.f), "f"(x1.f));
    return h2;
#else
    (void)b2;
    return 0u;
#endif
}

// Compute exact FP16 zp2 = sc2 * (-7.5) matching Triton `skinny_int4.py:573` to 0.0 ULP!
static __device__ __forceinline__ uint32_t mul_neg75_f16x2(uint32_t sc2) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    const uint32_t neg75_h2 = 0xC780C780u;  // exact IEEE-754 binary16(-7.5)
    uint32_t zp2;
    asm("mul.f16x2 %0, %1, %2;\n" : "=r"(zp2) : "r"(sc2), "r"(neg75_h2));
    return zp2;
#else
    (void)sc2;
    return 0u;
#endif
}

// Exact 0.0-ULP IEEE-754 FP16 4-bit nibble unpack -> f16x2(q0*sc + zp, q1*sc + zp) (0 SMEM, 0 branches!)
// Identical bit-for-bit to Triton `_u4_to_f16_pair` + `q * sc + zp` for BOTH `u4` and `i4`!
static __device__ __forceinline__ uint32_t unpack_byte_to_f16x2(
    uint32_t word, uint32_t prmt_sel, uint32_t sc2, uint32_t zp2
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    const uint32_t mask_0f = 0x000F000Fu;
    const uint32_t magic_1024 = 0x64006400u;  // exact f16x2(1024.0, 1024.0)
    const uint32_t neg_1024 = 0xE400E400u;    // exact f16x2(-1024.0, -1024.0)
    uint32_t w_r4 = word >> 4;
    uint32_t raw;
    uint32_t res;
    asm("prmt.b32 %0, %1, %2, %3;\n" : "=r"(raw) : "r"(word), "r"(w_r4), "r"(prmt_sel));
    // (raw & 0x000F000F) | 0x64006400 -> exact f16x2(1024.0 + q0, 1024.0 + q1)
    asm("lop3.b32 %0, %1, %2, %3, 0xEA;\n" : "=r"(raw) : "r"(raw), "r"(mask_0f), "r"(magic_1024));
    // + (-1024.0, -1024.0) -> exact f16x2(q0, q1)
    asm("add.f16x2 %0, %1, %2;\n" : "=r"(raw) : "r"(raw), "r"(neg_1024));
    // exact f16x2(q0*sc + zp, q1*sc + zp)
    asm("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(res) : "r"(raw), "r"(sc2), "r"(zp2));
    return res;
#else
    (void)word; (void)prmt_sel; (void)sc2; (void)zp2;
    return 0u;
#endif
}

static __device__ __forceinline__ float div_rn(float a, float b) {
#if defined(__CUDA_ARCH__)
    float r;
    asm("div.rn.f32 %0, %1, %2;\n" : "=f"(r) : "f"(a), "f"(b));
    return r;
#else
    return a / b;
#endif
}

// Matches Triton `tl.div_rn(gate, 1.0 + tldevice.exp(-gate)) * up` to 0.0 ULP
static __device__ __forceinline__ float silu_mul_f32(float gate, float up) {
    return div_rn(gate, 1.0f + expf(-gate)) * up;
}

static __device__ __forceinline__ float load_bias_val(
    const void* BIAS, int n_idx, int N, int has_bias, int bias_is_f32
) {
    if (!has_bias || n_idx >= N) return 0.0f;
    if (bias_is_f32) {
        return static_cast<const float*>(BIAS)[n_idx];
    }
    return bf16_bits_to_f32(static_cast<const uint16_t*>(BIAS)[n_idx]);
}

// Dot 2 FP16 weights (packed in w_h2) with 2 BF16 activations (packed in x_b2) -> FP32 accumulate
// Converts X via exact bf16->f16->f32 (0-ULP identical to Triton's x.to(f16) input rounding).
static __device__ __forceinline__ float dot2_bf16_f16_f32(
    float acc, uint32_t x_b2, uint32_t w_h2
) {
    const uint32_t x_h2 = bf16x2_to_f16x2_fast(x_b2);
    const float x0 = f16_bits_to_f32(static_cast<uint16_t>(x_h2 & 0xFFFFu));
    const float x1 = f16_bits_to_f32(static_cast<uint16_t>(x_h2 >> 16));
    const float w0 = f16_bits_to_f32(static_cast<uint16_t>(w_h2 & 0xFFFFu));
    const float w1 = f16_bits_to_f32(static_cast<uint16_t>(w_h2 >> 16));
    acc = (acc + (x0 * w0)) + (x1 * w1);
    return acc;
}

template <typename LaunchFn, typename... Args>
static inline cudaError_t launch_kernel_ex(
    bool use_pdl, dim3 grid, dim3 block, size_t smem_bytes, cudaStream_t stream,
    LaunchFn kernel, Args... args
) {
    if (!use_pdl) {
        kernel<<<grid, block, smem_bytes, stream>>>(args...);
        return cudaGetLastError();
    }
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = smem_bytes;
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    return cudaLaunchKernelEx(&cfg, kernel, args...);
}

// ============================================================================
// Shared Coherent Split-K Finalize Helper (used by Tier 1, Tier 2, Tier 3)
// ============================================================================
template <int BN_CTA, int N_THREADS, bool FUSE_SILU, bool OUT_F32>
static __device__ __forceinline__ void finalize_split_k_out(
    const float* __restrict__ p_panel,
    const void* __restrict__ BIAS,
    void* __restrict__ OUT,
    int M, int N, int n_cta, int split_k, int64_t part_split_stride, int64_t so,
    int has_bias, int bias_is_f32, int tid
) {
    if constexpr (FUSE_SILU) {
        constexpr int HALF_BN = BN_CTA >> 1;
        const int total_pairs = M * HALF_BN;
        for (int idx = tid; idx < total_pairs; idx += N_THREADS) {
            const int m_row = idx / HALF_BN;
            const int c_loc = idx - m_row * HALF_BN;
            const int n_g0 = n_cta + (c_loc << 1);
            const int c_out = n_g0 >> 1;
            if (c_out < (N >> 1)) {
                float acc_g = load_bias_val(BIAS, n_g0, N, has_bias, bias_is_f32);
                float acc_u = load_bias_val(BIAS, n_g0 + 1, N, has_bias, bias_is_f32);
                const int elem0 = m_row * BN_CTA + (c_loc << 1);
                for (int s = 0; s < split_k; ++s) {
                    acc_g += ld_cg_f32(p_panel + static_cast<int64_t>(s) * part_split_stride + elem0);
                    acc_u += ld_cg_f32(p_panel + static_cast<int64_t>(s) * part_split_stride + elem0 + 1);
                }
                const float y = silu_mul_f32(acc_g, acc_u);
                if constexpr (OUT_F32) {
                    static_cast<float*>(OUT)[static_cast<int64_t>(m_row) * so + c_out] = y;
                } else {
                    static_cast<uint16_t*>(OUT)[static_cast<int64_t>(m_row) * so + c_out] = f32_to_bf16_bits(y);
                }
            }
        }
    } else {
        const int total_elems = M * BN_CTA;
        for (int idx = tid; idx < total_elems; idx += N_THREADS) {
            const int m_row = idx / BN_CTA;
            const int n_loc = idx - m_row * BN_CTA;
            const int n_g = n_cta + n_loc;
            if (n_g < N) {
                float acc = load_bias_val(BIAS, n_g, N, has_bias, bias_is_f32);
                for (int s = 0; s < split_k; ++s) {
                    acc += ld_cg_f32(p_panel + static_cast<int64_t>(s) * part_split_stride + idx);
                }
                if constexpr (OUT_F32) {
                    static_cast<float*>(OUT)[static_cast<int64_t>(m_row) * so + n_g] = acc;
                } else {
                    static_cast<uint16_t*>(OUT)[static_cast<int64_t>(m_row) * so + n_g] = f32_to_bf16_bits(acc);
                }
            }
        }
    }
}

// ============================================================================
// TIER 1 (`M = 1..4`): Zero-SMEM, Zero-Barrier, Zero-Shuffle-in-K-Loop
// Warp-Specialized SIMT/FP16-Exact Kernel
//   - M_CAP in {1, 2, 4}; BN_CTA in {16, 32} (128 threads = 4 warps)
//   - Zero `sX` shared memory, zero `__syncthreads()` inside K-loop!
//   - Exact 1:1 lane-to-group alignment: each `(grp, lt)` lane holds 32 weights
//     (16 for BN_CTA=16 K_SPLIT2) with constant (sc2, zp2) -> 0 shuffles in K-loop!
//   - Explicit scalar accumulators a0..a3 (0 local arrays, 0 spills!)
// ============================================================================
template <int BN_CTA, int M_CAP, int GS, bool SYM, bool FUSE_SILU, bool OUT_F32>
__global__ __launch_bounds__(128, 4)
void drafter_u4_simt_m14_v14_kernel(
    const uint16_t* __restrict__ X1,
    const uint16_t* __restrict__ X2,
    const uint8_t*  __restrict__ WP,
    const uint16_t* __restrict__ SCP,
    const uint16_t* __restrict__ ZPP,
    const void*     __restrict__ BIAS,
    float*          __restrict__ PART,
    uint32_t*       __restrict__ SEM,
    void*           __restrict__ OUT,
    int M, int N, int K,
    int wp_bn, int bk, int split_k, int iters_per_split,
    int64_t sx1, int64_t sx2, int64_t so,
    int rnn_mode, int n_reqs, int k_steps_rnn,
    int has_bias, int bias_is_f32, int use_pdl
) {
    constexpr int N_THREADS = 128;
    constexpr bool K_SPLIT2 = (BN_CTA == 16);

    const int pid_n = static_cast<int>(blockIdx.x);
    const int ks_id = static_cast<int>(blockIdx.y);
    const int tid = static_cast<int>(threadIdx.x);
    const int wid = tid >> 5;
    const int lane = tid & 31;
    const int grp = lane >> 2;              // 0..7
    const int lt = lane & 3;                // 0..3

    const int kw = K_SPLIT2 ? (wid >> 1) : 0;
    const int n_warp = K_SPLIT2 ? (wid & 1) : wid;
    const int n_cta = pid_n * BN_CTA;
    const int n_row_local = n_warp * 8 + grp;

    const int sub_ratio = wp_bn / BN_CTA;
    const int wp_panel = (sub_ratio == 2) ? (pid_n >> 1) : pid_n;
    const int wp_sub_off = (sub_ratio == 2) ? ((pid_n & 1) * BN_CTA) : 0;
    const int wp_row = wp_sub_off + n_row_local;

    const int total_iters = K >> 7;
    const int it_beg = ks_id * iters_per_split;
    const int it_end = (it_beg + iters_per_split < total_iters)
                         ? (it_beg + iters_per_split)
                         : total_iters;
    const int n_iters = it_end - it_beg;
    if (n_iters <= 0) {
        if (use_pdl) {
            gdc_wait();
            __syncthreads();
            if (tid == 0) gdc_launch_dependents();
        }
        return;
    }

    const int bk_shift = (bk == 256) ? 1 : 0;
    const int total_k_steps = K / bk;
    const int bkp = bk >> 1;
    const int bkg = bk / GS;
    const int half_K = K >> 1;
    const int g_per_128 = 128 / GS;

    // Tiny SMEM ONLY for K_SPLIT2 end-of-loop reduction + sIsLast (<= 260 B total!)
    __shared__ float sRed[M_CAP][16];
    __shared__ int32_t sIsLast;

    // Precompute X1/X2 row indices in registers (zero SMEM reads, zero divisions in K-loop!)
    auto calc_x1_row = [&](int r) -> int {
        if (r >= M) return -1;
        if (rnn_mode == 2) {
            const int m_step = r / n_reqs;
            const int m_req = r - m_step * n_reqs;
            return m_req * k_steps_rnn + m_step;
        }
        return r;
    };
    auto calc_x2_row = [&](int r) -> int {
        if (r >= M) return -1;
        if (rnn_mode == 2) {
            const int m_step = r / n_reqs;
            return r - m_step * n_reqs;
        }
        return (rnn_mode == 1) ? r : -1;
    };
    const int r1_0 = calc_x1_row(0);
    const int r2_0 = calc_x2_row(0);
    const int r1_1 = (M_CAP >= 2) ? calc_x1_row(1) : -1;
    const int r2_1 = (M_CAP >= 2) ? calc_x2_row(1) : -1;
    const int r1_2 = (M_CAP >= 4) ? calc_x1_row(2) : -1;
    const int r2_2 = (M_CAP >= 4) ? calc_x2_row(2) : -1;
    const int r1_3 = (M_CAP >= 4) ? calc_x1_row(3) : -1;
    const int r2_3 = (M_CAP >= 4) ? calc_x2_row(3) : -1;

    // Exact lane group index within a 128-K iteration (constant for all weights owned by this lane!)
    int g_lane;
    if constexpr (K_SPLIT2) {
        g_lane = (GS == 32) ? ((kw << 1) | (lt >> 1)) : kw;
    } else {
        g_lane = (GS == 32) ? lt : (lt >> 1);
    }

    auto load_wp_sc_zp = [&](int it_abs, uint4& wp4, uint2& wp2, uint32_t& sc2, uint32_t& zp2) {
        const int k_step = it_abs >> bk_shift;
        const int kh = it_abs & bk_shift;
        const int64_t off = (static_cast<int64_t>(wp_panel) * total_k_steps + k_step) * wp_bn + wp_row;
        if constexpr (K_SPLIT2) {
            wp2 = ld_cs_u2(WP + off * bkp + (kh << 6) + (kw << 5) + (lt << 3));
        } else {
            wp4 = ld_cs_u4(WP + off * bkp + (kh << 6) + (lt << 4));
        }
        const int64_t sidx = ((static_cast<int64_t>(wp_panel) * total_k_steps + k_step) * bkg
                             + kh * g_per_128 + g_lane) * wp_bn + wp_row;
        const uint16_t sh = ld_ca_u16(SCP + sidx);
        sc2 = static_cast<uint32_t>(sh) | (static_cast<uint32_t>(sh) << 16);
        if constexpr (SYM) {
            zp2 = mul_neg75_f16x2(sc2);
        } else {
            const uint16_t zh = ld_ca_u16(ZPP + sidx);
            zp2 = static_cast<uint32_t>(zh) | (static_cast<uint32_t>(zh) << 16);
        }
    };

    // Safe PDL: preload static WP[0], SCP[0], ZPP[0] BEFORE gdc_wait()!
    uint4 wp_cur4 = make_uint4(0u, 0u, 0u, 0u);
    uint2 wp_cur2 = make_uint2(0u, 0u);
    uint32_t sc_cur = 0u, zp_cur = 0u;
    load_wp_sc_zp(it_beg, wp_cur4, wp_cur2, sc_cur, zp_cur);

    if (use_pdl) gdc_wait();

    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;

    auto dot_word8 = [&](uint32_t word, uint32_t sc2, uint32_t zp2,
                         const uint16_t* base_x, int64_t stride_x, int k_col, bool use_x2) {
        const uint32_t w01 = unpack_byte_to_f16x2(word, 0x0400u, sc2, zp2);
        const uint32_t w23 = unpack_byte_to_f16x2(word, 0x0501u, sc2, zp2);
        const uint32_t w45 = unpack_byte_to_f16x2(word, 0x0602u, sc2, zp2);
        const uint32_t w67 = unpack_byte_to_f16x2(word, 0x0703u, sc2, zp2);

        const int idx0 = use_x2 ? r2_0 : r1_0;
        if (idx0 >= 0) {
            const uint4 xv = ld_ca_u4(base_x + static_cast<int64_t>(idx0) * stride_x + k_col);
            a0 = dot2_bf16_f16_f32(a0, xv.x, w01);
            a0 = dot2_bf16_f16_f32(a0, xv.y, w23);
            a0 = dot2_bf16_f16_f32(a0, xv.z, w45);
            a0 = dot2_bf16_f16_f32(a0, xv.w, w67);
        }
        if constexpr (M_CAP >= 2) {
            const int idx1 = use_x2 ? r2_1 : r1_1;
            if (idx1 >= 0) {
                const uint4 xv = ld_ca_u4(base_x + static_cast<int64_t>(idx1) * stride_x + k_col);
                a1 = dot2_bf16_f16_f32(a1, xv.x, w01);
                a1 = dot2_bf16_f16_f32(a1, xv.y, w23);
                a1 = dot2_bf16_f16_f32(a1, xv.z, w45);
                a1 = dot2_bf16_f16_f32(a1, xv.w, w67);
            }
        }
        if constexpr (M_CAP >= 4) {
            const int idx2 = use_x2 ? r2_2 : r1_2;
            if (idx2 >= 0) {
                const uint4 xv = ld_ca_u4(base_x + static_cast<int64_t>(idx2) * stride_x + k_col);
                a2 = dot2_bf16_f16_f32(a2, xv.x, w01);
                a2 = dot2_bf16_f16_f32(a2, xv.y, w23);
                a2 = dot2_bf16_f16_f32(a2, xv.z, w45);
                a2 = dot2_bf16_f16_f32(a2, xv.w, w67);
            }
            const int idx3 = use_x2 ? r2_3 : r1_3;
            if (idx3 >= 0) {
                const uint4 xv = ld_ca_u4(base_x + static_cast<int64_t>(idx3) * stride_x + k_col);
                a3 = dot2_bf16_f16_f32(a3, xv.x, w01);
                a3 = dot2_bf16_f16_f32(a3, xv.y, w23);
                a3 = dot2_bf16_f16_f32(a3, xv.z, w45);
                a3 = dot2_bf16_f16_f32(a3, xv.w, w67);
            }
        }
    };

    // ---- Main Zero-Barrier, Zero-SMEM, Register-Pipelined K-Loop ----
    for (int it = 0; it < n_iters; ++it) {
        const int it_abs = it_beg + it;
        uint4 wp_nxt4 = make_uint4(0u, 0u, 0u, 0u);
        uint2 wp_nxt2 = make_uint2(0u, 0u);
        uint32_t sc_nxt = 0u, zp_nxt = 0u;
        if (it + 1 < n_iters) {
            load_wp_sc_zp(it_abs + 1, wp_nxt4, wp_nxt2, sc_nxt, zp_nxt);
        }

        const int k_abs = it_abs << 7;
        const bool use_x2 = (X2 != nullptr) && (k_abs >= half_K);
        const int k_off = use_x2 ? (k_abs - half_K) : k_abs;
        const uint16_t* base_x = use_x2 ? X2 : X1;
        const int64_t stride_x = use_x2 ? sx2 : sx1;

        if constexpr (K_SPLIT2) {
            const int k_lane = k_off + (kw << 6) + (lt << 4);
            dot_word8(wp_cur2.x, sc_cur, zp_cur, base_x, stride_x, k_lane + 0, use_x2);
            dot_word8(wp_cur2.y, sc_cur, zp_cur, base_x, stride_x, k_lane + 8, use_x2);
            wp_cur2 = wp_nxt2;
        } else {
            const int k_lane = k_off + (lt << 5);
            dot_word8(wp_cur4.x, sc_cur, zp_cur, base_x, stride_x, k_lane + 0,  use_x2);
            dot_word8(wp_cur4.y, sc_cur, zp_cur, base_x, stride_x, k_lane + 8,  use_x2);
            dot_word8(wp_cur4.z, sc_cur, zp_cur, base_x, stride_x, k_lane + 16, use_x2);
            dot_word8(wp_cur4.w, sc_cur, zp_cur, base_x, stride_x, k_lane + 24, use_x2);
            wp_cur4 = wp_nxt4;
        }
        sc_cur = sc_nxt;
        zp_cur = zp_nxt;
    }

    // ---- End-of-Loop 4-Lane Quad Reduction (2 XOR shuffles total!) ----
    a0 += __shfl_xor_sync(0xFFFFFFFFu, a0, 1);
    a0 += __shfl_xor_sync(0xFFFFFFFFu, a0, 2);
    if constexpr (M_CAP >= 2) {
        a1 += __shfl_xor_sync(0xFFFFFFFFu, a1, 1);
        a1 += __shfl_xor_sync(0xFFFFFFFFu, a1, 2);
    }
    if constexpr (M_CAP >= 4) {
        a2 += __shfl_xor_sync(0xFFFFFFFFu, a2, 1);
        a2 += __shfl_xor_sync(0xFFFFFFFFu, a2, 2);
        a3 += __shfl_xor_sync(0xFFFFFFFFu, a3, 1);
        a3 += __shfl_xor_sync(0xFFFFFFFFu, a3, 2);
    }

    if constexpr (K_SPLIT2) {
        if (kw == 1 && lt == 0) {
            sRed[0][n_row_local] = a0;
            if constexpr (M_CAP >= 2) sRed[1][n_row_local] = a1;
            if constexpr (M_CAP >= 4) {
                sRed[2][n_row_local] = a2;
                sRed[3][n_row_local] = a3;
            }
        }
        __syncthreads();
        if (kw == 0 && lt == 0) {
            a0 += sRed[0][n_row_local];
            if constexpr (M_CAP >= 2) a1 += sRed[1][n_row_local];
            if constexpr (M_CAP >= 4) {
                a2 += sRed[2][n_row_local];
                a3 += sRed[3][n_row_local];
            }
        }
    }

    // ---- Epilogue: Direct (split_k == 1) vs Coherent Split-K Finalize ----
    if (split_k == 1) {
        const int n_g = n_cta + n_row_local;
        const float b_v = load_bias_val(BIAS, n_g, N, has_bias, bias_is_f32);
        a0 += b_v;
        if constexpr (M_CAP >= 2) a1 += b_v;
        if constexpr (M_CAP >= 4) { a2 += b_v; a3 += b_v; }

        if constexpr (FUSE_SILU) {
            // Warp-uniform shuffle BEFORE branching on `lt == 0 && (grp & 1) == 0`!
            // lane = grp * 4 + lt -> lane ^ 4 swaps (gate=even grp) with (up=odd grp)!
            const float u0 = __shfl_xor_sync(0xFFFFFFFFu, a0, 4);
            const float u1 = (M_CAP >= 2) ? __shfl_xor_sync(0xFFFFFFFFu, a1, 4) : 0.0f;
            const float u2 = (M_CAP >= 4) ? __shfl_xor_sync(0xFFFFFFFFu, a2, 4) : 0.0f;
            const float u3 = (M_CAP >= 4) ? __shfl_xor_sync(0xFFFFFFFFu, a3, 4) : 0.0f;

            if ((!K_SPLIT2 || kw == 0) && lt == 0 && ((grp & 1) == 0)) {
                const int c_out = n_g >> 1;
                if (c_out < (N >> 1)) {
                    auto store_silu = [&](int m_row, float gv, float uv) {
                        if (m_row >= M) return;
                        const float y = silu_mul_f32(gv, uv);
                        if constexpr (OUT_F32) {
                            static_cast<float*>(OUT)[static_cast<int64_t>(m_row) * so + c_out] = y;
                        } else {
                            static_cast<uint16_t*>(OUT)[static_cast<int64_t>(m_row) * so + c_out] = f32_to_bf16_bits(y);
                        }
                    };
                    store_silu(0, a0, u0);
                    if constexpr (M_CAP >= 2) store_silu(1, a1, u1);
                    if constexpr (M_CAP >= 4) {
                        store_silu(2, a2, u2);
                        store_silu(3, a3, u3);
                    }
                }
            }
        } else {
            if ((!K_SPLIT2 || kw == 0) && lt == 0 && n_g < N) {
                auto store_val = [&](int m_row, float v) {
                    if (m_row >= M) return;
                    if constexpr (OUT_F32) {
                        static_cast<float*>(OUT)[static_cast<int64_t>(m_row) * so + n_g] = v;
                    } else {
                        static_cast<uint16_t*>(OUT)[static_cast<int64_t>(m_row) * so + n_g] = f32_to_bf16_bits(v);
                    }
                };
                store_val(0, a0);
                if constexpr (M_CAP >= 2) store_val(1, a1);
                if constexpr (M_CAP >= 4) {
                    store_val(2, a2);
                    store_val(3, a3);
                }
            }
        }
        if (use_pdl) {
            __syncthreads();
            if (tid == 0) gdc_launch_dependents();
        }
        return;
    }

    // ---- Coherent Split-K Partial Store (`st.cg` -> `__syncthreads` -> `__threadfence` -> `atom.acq_rel.gpu`) ----
    const int64_t part_panel_stride = static_cast<int64_t>(split_k) * M_CAP * BN_CTA;
    const int64_t part_split_stride = static_cast<int64_t>(M_CAP) * BN_CTA;
    float* p_base = PART + static_cast<int64_t>(pid_n) * part_panel_stride
                         + static_cast<int64_t>(ks_id) * part_split_stride;

    if ((!K_SPLIT2 || kw == 0) && lt == 0) {
        if (0 < M) st_cg_f32(p_base + 0 * BN_CTA + n_row_local, a0);
        if constexpr (M_CAP >= 2) {
            if (1 < M) st_cg_f32(p_base + 1 * BN_CTA + n_row_local, a1);
        }
        if constexpr (M_CAP >= 4) {
            if (2 < M) st_cg_f32(p_base + 2 * BN_CTA + n_row_local, a2);
            if (3 < M) st_cg_f32(p_base + 3 * BN_CTA + n_row_local, a3);
        }
    }

    __syncthreads();
    __threadfence();

    if (tid == 0) {
        const uint32_t prev = atomic_add_acq_rel_gpu(SEM + pid_n, 1u);
        const int last = (static_cast<int>(prev) + 1 == split_k) ? 1 : 0;
        if (last) {
            st_cg_u32(SEM + pid_n, 0u);
        }
        sIsLast = last;
    }
    __syncthreads();

    if (sIsLast != 0) {
        const float* p_panel = PART + static_cast<int64_t>(pid_n) * part_panel_stride;
        finalize_split_k_out<BN_CTA, N_THREADS, FUSE_SILU, OUT_F32>(
            p_panel, BIAS, OUT, M, N, n_cta, split_k, part_split_stride, so,
            has_bias, bias_is_f32, tid
        );
        if (use_pdl) {
            __syncthreads();
            if (tid == 0) gdc_launch_dependents();
        }
    }
}

// ============================================================================
// TIER 2 (`M = 1..8`, Primary for M=5..8 e.g. C=1 M=7,8):
// Dedicated BM=8 Half-Row FP16-Exact Tensor-Core Kernel
//   - sX[2][8][136] = 4.25 KiB SMEM (50% of v9!), 0 bank conflicts!
//   - ldmatrix.sync.aligned.x2 (50% SMEM reads vs v9's x4!) + mma.m16n8k16.f16!
//   - Pipelined register WP + SCP/ZPP preload -> 0 global loads in inner loop!
//   - Single `__syncthreads()` per 128-K iteration (vs v9's 2)!
//   - Half-size split-K PART buffer (BM_TILE = 8)!
// ============================================================================
template <int BN_CTA, int GS, bool SYM, bool FUSE_SILU, bool OUT_F32>
__global__ __launch_bounds__(BN_CTA == 64 ? 256 : 128, BN_CTA == 64 ? 2 : 4)
void drafter_u4_tc_bm8_v14_kernel(
    const uint16_t* __restrict__ X1,
    const uint16_t* __restrict__ X2,
    const uint8_t*  __restrict__ WP,
    const uint16_t* __restrict__ SCP,
    const uint16_t* __restrict__ ZPP,
    const void*     __restrict__ BIAS,
    float*          __restrict__ PART,
    uint32_t*       __restrict__ SEM,
    void*           __restrict__ OUT,
    int M, int N, int K,
    int wp_bn, int bk, int split_k, int iters_per_split,
    int64_t sx1, int64_t sx2, int64_t so,
    int rnn_mode, int n_reqs, int k_steps_rnn,
    int has_bias, int bias_is_f32, int use_pdl
) {
    constexpr int BM_TILE = 8;
    constexpr int SX_STRIDE = 136;
    constexpr int N_THREADS = (BN_CTA == 64) ? 256 : 128;
    constexpr bool K_SPLIT2 = (BN_CTA == 16);

    const int pid_n = static_cast<int>(blockIdx.x);
    const int ks_id = static_cast<int>(blockIdx.y);
    const int tid = static_cast<int>(threadIdx.x);
    const int wid = tid >> 5;
    const int lane = tid & 31;
    const int grp = lane >> 2;
    const int lt = lane & 3;

    const int kw = K_SPLIT2 ? (wid >> 1) : 0;
    const int n_warp = K_SPLIT2 ? (wid & 1) : wid;
    const int n_cta = pid_n * BN_CTA;
    const int n_row_local = n_warp * 8 + grp;

    const int sub_ratio = wp_bn / BN_CTA;
    const int wp_panel = (sub_ratio == 2) ? (pid_n >> 1) : pid_n;
    const int wp_sub_off = (sub_ratio == 2) ? ((pid_n & 1) * BN_CTA) : 0;
    const int wp_row = wp_sub_off + n_row_local;

    const int total_iters = K >> 7;
    const int it_beg = ks_id * iters_per_split;
    const int it_end = (it_beg + iters_per_split < total_iters)
                         ? (it_beg + iters_per_split)
                         : total_iters;
    const int n_iters = it_end - it_beg;
    if (n_iters <= 0) {
        if (use_pdl) {
            gdc_wait();
            __syncthreads();
            if (tid == 0) gdc_launch_dependents();
        }
        return;
    }

    const int bk_shift = (bk == 256) ? 1 : 0;
    const int total_k_steps = K / bk;
    const int bkp = bk >> 1;
    const int bkg = bk / GS;
    const int half_K = K >> 1;
    const int g_per_128 = 128 / GS;

    const uint32_t prmt_sel = static_cast<uint32_t>(lt)
                            | (static_cast<uint32_t>(4 + lt) << 8);

    union SmemStorage {
        alignas(16) uint16_t sX[2][BM_TILE][SX_STRIDE];
        alignas(16) float sRed[BM_TILE][16];
    };
    __shared__ SmemStorage smem;
    __shared__ int32_t sX1Row[BM_TILE];
    __shared__ int32_t sX2Row[BM_TILE];
    __shared__ int32_t sIsLast;

    if (tid < BM_TILE) {
        if (tid < M) {
            if (rnn_mode == 2) {
                const int m_step = tid / n_reqs;
                const int m_req = tid - m_step * n_reqs;
                sX1Row[tid] = m_req * k_steps_rnn + m_step;
                sX2Row[tid] = m_req;
            } else if (rnn_mode == 1) {
                sX1Row[tid] = tid;
                sX2Row[tid] = tid;  // sx2 == 0 broadcasts row 0; sx2 > 0 indexes row tid!
            } else {
                sX1Row[tid] = tid;
                sX2Row[tid] = -1;
            }
        } else {
            sX1Row[tid] = -1;
            sX2Row[tid] = -1;
        }
    }

    int g_load;
    if constexpr (K_SPLIT2) {
        g_load = (GS == 32) ? ((kw << 1) | (lt & 1)) : kw;
    } else {
        g_load = (GS == 32) ? lt : (lt & 1);
    }

    auto load_wp_sc_zp = [&](int it_abs, uint4& wp4, uint2& wp2, uint32_t& sc_pack, uint32_t& zp_pack) {
        const int k_step = it_abs >> bk_shift;
        const int kh = it_abs & bk_shift;
        const int64_t off = (static_cast<int64_t>(wp_panel) * total_k_steps + k_step) * wp_bn + wp_row;
        if constexpr (K_SPLIT2) {
            wp2 = ld_cs_u2(WP + off * bkp + (kh << 6) + (kw << 5) + (lt << 3));
        } else {
            wp4 = ld_cs_u4(WP + off * bkp + (kh << 6) + (lt << 4));
        }
        const int64_t sidx = ((static_cast<int64_t>(wp_panel) * total_k_steps + k_step) * bkg
                             + kh * g_per_128 + g_load) * wp_bn + wp_row;
        const uint16_t sh = ld_ca_u16(SCP + sidx);
        sc_pack = static_cast<uint32_t>(sh) | (static_cast<uint32_t>(sh) << 16);
        if constexpr (SYM) {
            zp_pack = mul_neg75_f16x2(sc_pack);
        } else {
            const uint16_t zh = ld_ca_u16(ZPP + sidx);
            zp_pack = static_cast<uint32_t>(zh) | (static_cast<uint32_t>(zh) << 16);
        }
    };

    auto load_sx_stage = [&](int stage, int it_abs) {
        const int k_abs = it_abs << 7;
        const bool use_x2 = (X2 != nullptr) && (k_abs >= half_K);
        const int k_off = use_x2 ? (k_abs - half_K) : k_abs;
        const uint16_t* base_x = use_x2 ? X2 : X1;
        const int64_t stride_x = use_x2 ? sx2 : sx1;
        // 8 rows * 16 chunks/row = 128 chunks (exactly 1 chunk/thr for 128 thr; tid<128 for 256 thr!)
        if (tid < 128) {
            const int r = tid >> 4;
            const int col = (tid & 15) << 3;
            const int row_idx = use_x2 ? sX2Row[r] : sX1Row[r];
            uint4 h4 = make_uint4(0u, 0u, 0u, 0u);
            if (row_idx >= 0) {
                const uint4 b4 = ld_ca_u4(base_x + static_cast<int64_t>(row_idx) * stride_x + k_off + col);
                h4.x = bf16x2_to_f16x2_fast(b4.x);
                h4.y = bf16x2_to_f16x2_fast(b4.y);
                h4.z = bf16x2_to_f16x2_fast(b4.z);
                h4.w = bf16x2_to_f16x2_fast(b4.w);
            }
            *reinterpret_cast<uint4*>(&smem.sX[stage][r][col]) = h4;
        }
    };

    // Safe PDL: preload static WP[0], SCP[0], ZPP[0] BEFORE gdc_wait()!
    uint4 wp_cur4 = make_uint4(0u, 0u, 0u, 0u);
    uint2 wp_cur2 = make_uint2(0u, 0u);
    uint32_t sc_cur = 0u, zp_cur = 0u;
    load_wp_sc_zp(it_beg, wp_cur4, wp_cur2, sc_cur, zp_cur);

    if (use_pdl) gdc_wait();
    __syncthreads();  // guarantees sX1Row / sX2Row visible before load_sx_stage

    // Prime stage 0 `sX` in FP16
    load_sx_stage(0, it_beg);
    __syncthreads();

    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;

    // Precompute ldmatrix.x2 row/col indices:
    // lanes 0..7 -> rows 0..7 cols 0..7 (a0); lanes 8..15 -> rows 0..7 cols 8..15 (a2)
    const int ldm2_row = lane & 7;
    const int ldm2_col = ((lane >> 3) & 1) << 3;

    // ---- Main Single-Barrier 2-Stage Software-Pipelined K-Loop ----
    for (int it = 0; it < n_iters; ++it) {
        const int stage = it & 1;
        const int it_abs = it_beg + it;

        uint4 wp_nxt4 = make_uint4(0u, 0u, 0u, 0u);
        uint2 wp_nxt2 = make_uint2(0u, 0u);
        uint32_t sc_nxt = 0u, zp_nxt = 0u;
        if (it + 1 < n_iters) {
            load_wp_sc_zp(it_abs + 1, wp_nxt4, wp_nxt2, sc_nxt, zp_nxt);
            load_sx_stage(stage ^ 1, it_abs + 1);
        }

        if constexpr (K_SPLIT2) {
            uint32_t sc2 = sc_cur, zp2 = zp_cur;
            #pragma unroll
            for (int ks_loc = 0; ks_loc < 4; ++ks_loc) {
                const int ks = (kw << 2) + ks_loc;
                if constexpr (GS == 32) {
                    if ((ks_loc & 1) == 0) {
                        const int s_lane = (grp << 2) | (ks_loc >> 1);
                        sc2 = __shfl_sync(0xFFFFFFFFu, sc_cur, s_lane);
                        zp2 = __shfl_sync(0xFFFFFFFFu, zp_cur, s_lane);
                    }
                }
                const int src_lane = (grp << 2) | ks_loc;
                const uint32_t u_lo = __shfl_sync(0xFFFFFFFFu, wp_cur2.x, src_lane);
                const uint32_t u_hi = __shfl_sync(0xFFFFFFFFu, wp_cur2.y, src_lane);
                const uint32_t b0 = unpack_byte_to_f16x2(u_lo, prmt_sel, sc2, zp2);
                const uint32_t b1 = unpack_byte_to_f16x2(u_hi, prmt_sel, sc2, zp2);

                const int ko = ks << 4;
                uint32_t a0, a2;
                ldmatrix_x2_b16(a0, a2, smem_u32(&smem.sX[stage][ldm2_row][ko + ldm2_col]));
                mma_m16n8k16_f16(d0, d1, d2, d3, a0, 0u, a2, 0u, b0, b1);
            }
            wp_cur2 = wp_nxt2;
        } else {
            uint32_t sc2 = 0u, zp2 = 0u;
            #pragma unroll
            for (int ks = 0; ks < 8; ++ks) {
                const int g_in_128 = (GS == 32) ? (ks >> 1) : (ks >> 2);
                const bool new_grp = (GS == 32) ? ((ks & 1) == 0) : ((ks & 3) == 0);
                if (new_grp) {
                    const int s_lane = (grp << 2) | g_in_128;
                    sc2 = __shfl_sync(0xFFFFFFFFu, sc_cur, s_lane);
                    zp2 = __shfl_sync(0xFFFFFFFFu, zp_cur, s_lane);
                }
                const int src_lane = (grp << 2) | (ks >> 1);
                const uint32_t u_lo = __shfl_sync(0xFFFFFFFFu, (ks & 1) ? wp_cur4.z : wp_cur4.x, src_lane);
                const uint32_t u_hi = __shfl_sync(0xFFFFFFFFu, (ks & 1) ? wp_cur4.w : wp_cur4.y, src_lane);
                const uint32_t b0 = unpack_byte_to_f16x2(u_lo, prmt_sel, sc2, zp2);
                const uint32_t b1 = unpack_byte_to_f16x2(u_hi, prmt_sel, sc2, zp2);

                const int ko = ks << 4;
                uint32_t a0, a2;
                ldmatrix_x2_b16(a0, a2, smem_u32(&smem.sX[stage][ldm2_row][ko + ldm2_col]));
                mma_m16n8k16_f16(d0, d1, d2, d3, a0, 0u, a2, 0u, b0, b1);
            }
            wp_cur4 = wp_nxt4;
        }
        sc_cur = sc_nxt;
        zp_cur = zp_nxt;
        if (it + 1 < n_iters) {
            __syncthreads();
        }
    }

    // ---- BN_CTA == 16 K-split warp reduction via aliased `smem.sRed` ----
    if constexpr (K_SPLIT2) {
        __syncthreads();  // ensure all kw warps finished reading smem.sX before writing aliased smem.sRed!
        const int n_loc0 = n_warp * 8 + (lt << 1);
        if (kw == 1) {
            smem.sRed[grp][n_loc0 + 0] = d0;
            smem.sRed[grp][n_loc0 + 1] = d1;
        }
        __syncthreads();
        if (kw == 0) {
            d0 += smem.sRed[grp][n_loc0 + 0];
            d1 += smem.sRed[grp][n_loc0 + 1];
        }
    }

    // ---- Epilogue: Direct (split_k == 1) vs Coherent Split-K Finalize ----
    if (split_k == 1) {
        if ((!K_SPLIT2 || kw == 0) && grp < M) {
            const int n_loc0 = n_warp * 8 + (lt << 1);
            const int n_g0 = n_cta + n_loc0;
            const int n_g1 = n_g0 + 1;
            const float v0 = d0 + load_bias_val(BIAS, n_g0, N, has_bias, bias_is_f32);
            const float v1 = d1 + load_bias_val(BIAS, n_g1, N, has_bias, bias_is_f32);
            if constexpr (FUSE_SILU) {
                const int c_out = n_g0 >> 1;
                if (c_out < (N >> 1)) {
                    const float y = silu_mul_f32(v0, v1);
                    if constexpr (OUT_F32) {
                        static_cast<float*>(OUT)[static_cast<int64_t>(grp) * so + c_out] = y;
                    } else {
                        static_cast<uint16_t*>(OUT)[static_cast<int64_t>(grp) * so + c_out] = f32_to_bf16_bits(y);
                    }
                }
            } else if constexpr (OUT_F32) {
                float* dst = static_cast<float*>(OUT) + static_cast<int64_t>(grp) * so + n_g0;
                if (n_g1 < N) {
                    dst[0] = v0;
                    dst[1] = v1;
                } else if (n_g0 < N) {
                    dst[0] = v0;
                }
            } else {
                uint16_t* dst = static_cast<uint16_t*>(OUT) + static_cast<int64_t>(grp) * so + n_g0;
                if (n_g1 < N) {
                    *reinterpret_cast<uint32_t*>(dst) = pack_bf16x2_f32(v0, v1);
                } else if (n_g0 < N) {
                    dst[0] = f32_to_bf16_bits(v0);
                }
            }
        }
        if (use_pdl) {
            __syncthreads();
            if (tid == 0) gdc_launch_dependents();
        }
        return;
    }

    const int64_t part_panel_stride = static_cast<int64_t>(split_k) * BM_TILE * BN_CTA;
    const int64_t part_split_stride = static_cast<int64_t>(BM_TILE) * BN_CTA;
    float* p_base = PART + static_cast<int64_t>(pid_n) * part_panel_stride
                         + static_cast<int64_t>(ks_id) * part_split_stride;

    if ((!K_SPLIT2 || kw == 0) && grp < M) {
        const int n_loc0 = n_warp * 8 + (lt << 1);
        st_cg_f32x2(p_base + grp * BN_CTA + n_loc0, d0, d1);
    }

    __syncthreads();
    __threadfence();

    if (tid == 0) {
        const uint32_t prev = atomic_add_acq_rel_gpu(SEM + pid_n, 1u);
        const int last = (static_cast<int>(prev) + 1 == split_k) ? 1 : 0;
        if (last) {
            st_cg_u32(SEM + pid_n, 0u);
        }
        sIsLast = last;
    }
    __syncthreads();

    if (sIsLast != 0) {
        const float* p_panel = PART + static_cast<int64_t>(pid_n) * part_panel_stride;
        finalize_split_k_out<BN_CTA, N_THREADS, FUSE_SILU, OUT_F32>(
            p_panel, BIAS, OUT, M, N, n_cta, split_k, part_split_stride, so,
            has_bias, bias_is_f32, tid
        );
        if (use_pdl) {
            __syncthreads();
            if (tid == 0) gdc_launch_dependents();
        }
    }
}

// ============================================================================
// TIER 3 (`M = 9..64`): Upgraded Single-Barrier 2-Stage FP16-Exact TC Kernel
//   - MT in {1, 2, 3, 4} -> BM_TILE = 16, 32, 48, 64
//   - BN_CTA in {16, 32, 64} (BN_CTA=64 unlocked down to MT=2 M=17..32 for draft_head!)
//   - Exact FP16 dequant + pipelined register SCP/ZPP preload + 1 barrier/iter + predicated PART stores!
// ============================================================================
template <int BN_CTA, int MT, int GS, bool SYM, bool FUSE_SILU, bool OUT_F32>
__global__ __launch_bounds__(BN_CTA == 64 ? 256 : 128, (BN_CTA == 64) ? 2 : ((MT <= 2) ? 4 : 2))
void drafter_u4_tc_m964_v14_kernel(
    const uint16_t* __restrict__ X1,
    const uint16_t* __restrict__ X2,
    const uint8_t*  __restrict__ WP,
    const uint16_t* __restrict__ SCP,
    const uint16_t* __restrict__ ZPP,
    const void*     __restrict__ BIAS,
    float*          __restrict__ PART,
    uint32_t*       __restrict__ SEM,
    void*           __restrict__ OUT,
    int M, int N, int K,
    int wp_bn, int bk, int split_k, int iters_per_split,
    int64_t sx1, int64_t sx2, int64_t so,
    int rnn_mode, int n_reqs, int k_steps_rnn,
    int has_bias, int bias_is_f32, int use_pdl
) {
    constexpr int BM_TILE = MT * 16;
    constexpr int SX_STRIDE = 136;
    constexpr int N_THREADS = (BN_CTA == 64) ? 256 : 128;
    constexpr bool K_SPLIT2 = (BN_CTA == 16);
    constexpr int CHUNKS_SX = BM_TILE * 16;

    const int pid_n = static_cast<int>(blockIdx.x);
    const int ks_id = static_cast<int>(blockIdx.y);
    const int tid = static_cast<int>(threadIdx.x);
    const int wid = tid >> 5;
    const int lane = tid & 31;
    const int grp = lane >> 2;
    const int lt = lane & 3;

    const int kw = K_SPLIT2 ? (wid >> 1) : 0;
    const int n_warp = K_SPLIT2 ? (wid & 1) : wid;
    const int n_cta = pid_n * BN_CTA;
    const int n_row_local = n_warp * 8 + grp;

    const int sub_ratio = wp_bn / BN_CTA;
    const int wp_panel = (sub_ratio == 2) ? (pid_n >> 1) : pid_n;
    const int wp_sub_off = (sub_ratio == 2) ? ((pid_n & 1) * BN_CTA) : 0;
    const int wp_row = wp_sub_off + n_row_local;

    const int total_iters = K >> 7;
    const int it_beg = ks_id * iters_per_split;
    const int it_end = (it_beg + iters_per_split < total_iters)
                         ? (it_beg + iters_per_split)
                         : total_iters;
    const int n_iters = it_end - it_beg;
    if (n_iters <= 0) {
        if (use_pdl) {
            gdc_wait();
            __syncthreads();
            if (tid == 0) gdc_launch_dependents();
        }
        return;
    }

    const int bk_shift = (bk == 256) ? 1 : 0;
    const int total_k_steps = K / bk;
    const int bkp = bk >> 1;
    const int bkg = bk / GS;
    const int half_K = K >> 1;
    const int g_per_128 = 128 / GS;

    const uint32_t prmt_sel = static_cast<uint32_t>(lt)
                            | (static_cast<uint32_t>(4 + lt) << 8);

    union SmemStorage {
        alignas(16) uint16_t sX[2][BM_TILE][SX_STRIDE];
        alignas(16) float sRed[BM_TILE][16];
    };
    __shared__ SmemStorage smem;
    __shared__ int32_t sX1Row[BM_TILE];
    __shared__ int32_t sX2Row[BM_TILE];
    __shared__ int32_t sIsLast;

    if (tid < BM_TILE) {
        if (tid < M) {
            if (rnn_mode == 2) {
                const int m_step = tid / n_reqs;
                const int m_req = tid - m_step * n_reqs;
                sX1Row[tid] = m_req * k_steps_rnn + m_step;
                sX2Row[tid] = m_req;
            } else if (rnn_mode == 1) {
                sX1Row[tid] = tid;
                sX2Row[tid] = tid;
            } else {
                sX1Row[tid] = tid;
                sX2Row[tid] = -1;
            }
        } else {
            sX1Row[tid] = -1;
            sX2Row[tid] = -1;
        }
    }

    int g_load;
    if constexpr (K_SPLIT2) {
        g_load = (GS == 32) ? ((kw << 1) | (lt & 1)) : kw;
    } else {
        g_load = (GS == 32) ? lt : (lt & 1);
    }

    auto load_wp_sc_zp = [&](int it_abs, uint4& wp4, uint2& wp2, uint32_t& sc_pack, uint32_t& zp_pack) {
        const int k_step = it_abs >> bk_shift;
        const int kh = it_abs & bk_shift;
        const int64_t off = (static_cast<int64_t>(wp_panel) * total_k_steps + k_step) * wp_bn + wp_row;
        if constexpr (K_SPLIT2) {
            wp2 = ld_cs_u2(WP + off * bkp + (kh << 6) + (kw << 5) + (lt << 3));
        } else {
            wp4 = ld_cs_u4(WP + off * bkp + (kh << 6) + (lt << 4));
        }
        const int64_t sidx = ((static_cast<int64_t>(wp_panel) * total_k_steps + k_step) * bkg
                             + kh * g_per_128 + g_load) * wp_bn + wp_row;
        const uint16_t sh = ld_ca_u16(SCP + sidx);
        sc_pack = static_cast<uint32_t>(sh) | (static_cast<uint32_t>(sh) << 16);
        if constexpr (SYM) {
            zp_pack = mul_neg75_f16x2(sc_pack);
        } else {
            const uint16_t zh = ld_ca_u16(ZPP + sidx);
            zp_pack = static_cast<uint32_t>(zh) | (static_cast<uint32_t>(zh) << 16);
        }
    };

    auto load_sx_stage = [&](int stage, int it_abs) {
        const int k_abs = it_abs << 7;
        const bool use_x2 = (X2 != nullptr) && (k_abs >= half_K);
        const int k_off = use_x2 ? (k_abs - half_K) : k_abs;
        const uint16_t* base_x = use_x2 ? X2 : X1;
        const int64_t stride_x = use_x2 ? sx2 : sx1;
        #pragma unroll
        for (int c = tid; c < CHUNKS_SX; c += N_THREADS) {
            const int r = c >> 4;
            const int col = (c & 15) << 3;
            const int row_idx = use_x2 ? sX2Row[r] : sX1Row[r];
            uint4 h4 = make_uint4(0u, 0u, 0u, 0u);
            if (row_idx >= 0) {
                const uint4 b4 = ld_ca_u4(base_x + static_cast<int64_t>(row_idx) * stride_x + k_off + col);
                h4.x = bf16x2_to_f16x2_fast(b4.x);
                h4.y = bf16x2_to_f16x2_fast(b4.y);
                h4.z = bf16x2_to_f16x2_fast(b4.z);
                h4.w = bf16x2_to_f16x2_fast(b4.w);
            }
            *reinterpret_cast<uint4*>(&smem.sX[stage][r][col]) = h4;
        }
    };

    // Safe PDL: preload static WP[0], SCP[0], ZPP[0] BEFORE gdc_wait()!
    uint4 wp_cur4 = make_uint4(0u, 0u, 0u, 0u);
    uint2 wp_cur2 = make_uint2(0u, 0u);
    uint32_t sc_cur = 0u, zp_cur = 0u;
    load_wp_sc_zp(it_beg, wp_cur4, wp_cur2, sc_cur, zp_cur);

    if (use_pdl) gdc_wait();
    __syncthreads();

    load_sx_stage(0, it_beg);
    __syncthreads();

    // Explicit scalar FP32 accumulators (0 local arrays -> 0 B stack, 0 spills!)
    float d00 = 0.f, d01 = 0.f, d02 = 0.f, d03 = 0.f;
    float d10 = 0.f, d11 = 0.f, d12 = 0.f, d13 = 0.f;
    float d20 = 0.f, d21 = 0.f, d22 = 0.f, d23 = 0.f;
    float d30 = 0.f, d31 = 0.f, d32 = 0.f, d33 = 0.f;

    const int ldm_row = lane & 15;
    const int ldm_col = (lane >> 4) << 3;

    // ---- Main Single-Barrier 2-Stage Software-Pipelined K-Loop ----
    for (int it = 0; it < n_iters; ++it) {
        const int stage = it & 1;
        const int it_abs = it_beg + it;

        uint4 wp_nxt4 = make_uint4(0u, 0u, 0u, 0u);
        uint2 wp_nxt2 = make_uint2(0u, 0u);
        uint32_t sc_nxt = 0u, zp_nxt = 0u;
        if (it + 1 < n_iters) {
            load_wp_sc_zp(it_abs + 1, wp_nxt4, wp_nxt2, sc_nxt, zp_nxt);
            load_sx_stage(stage ^ 1, it_abs + 1);
        }

        if constexpr (K_SPLIT2) {
            uint32_t sc2 = sc_cur, zp2 = zp_cur;
            #pragma unroll
            for (int ks_loc = 0; ks_loc < 4; ++ks_loc) {
                const int ks = (kw << 2) + ks_loc;
                if constexpr (GS == 32) {
                    if ((ks_loc & 1) == 0) {
                        const int s_lane = (grp << 2) | (ks_loc >> 1);
                        sc2 = __shfl_sync(0xFFFFFFFFu, sc_cur, s_lane);
                        zp2 = __shfl_sync(0xFFFFFFFFu, zp_cur, s_lane);
                    }
                }
                const int src_lane = (grp << 2) | ks_loc;
                const uint32_t u_lo = __shfl_sync(0xFFFFFFFFu, wp_cur2.x, src_lane);
                const uint32_t u_hi = __shfl_sync(0xFFFFFFFFu, wp_cur2.y, src_lane);
                const uint32_t b0 = unpack_byte_to_f16x2(u_lo, prmt_sel, sc2, zp2);
                const uint32_t b1 = unpack_byte_to_f16x2(u_hi, prmt_sel, sc2, zp2);

                const int ko = ks << 4;
                {
                    uint32_t a0, a1, a2, a3;
                    ldmatrix_x4_b16(a0, a1, a2, a3, smem_u32(&smem.sX[stage][ldm_row][ko + ldm_col]));
                    mma_m16n8k16_f16(d00, d01, d02, d03, a0, a1, a2, a3, b0, b1);
                }
                if constexpr (MT >= 2) {
                    uint32_t a0, a1, a2, a3;
                    ldmatrix_x4_b16(a0, a1, a2, a3, smem_u32(&smem.sX[stage][16 + ldm_row][ko + ldm_col]));
                    mma_m16n8k16_f16(d10, d11, d12, d13, a0, a1, a2, a3, b0, b1);
                }
                if constexpr (MT >= 3) {
                    uint32_t a0, a1, a2, a3;
                    ldmatrix_x4_b16(a0, a1, a2, a3, smem_u32(&smem.sX[stage][32 + ldm_row][ko + ldm_col]));
                    mma_m16n8k16_f16(d20, d21, d22, d23, a0, a1, a2, a3, b0, b1);
                }
                if constexpr (MT >= 4) {
                    uint32_t a0, a1, a2, a3;
                    ldmatrix_x4_b16(a0, a1, a2, a3, smem_u32(&smem.sX[stage][48 + ldm_row][ko + ldm_col]));
                    mma_m16n8k16_f16(d30, d31, d32, d33, a0, a1, a2, a3, b0, b1);
                }
            }
            wp_cur2 = wp_nxt2;
        } else {
            uint32_t sc2 = 0u, zp2 = 0u;
            #pragma unroll
            for (int ks = 0; ks < 8; ++ks) {
                const int g_in_128 = (GS == 32) ? (ks >> 1) : (ks >> 2);
                const bool new_grp = (GS == 32) ? ((ks & 1) == 0) : ((ks & 3) == 0);
                if (new_grp) {
                    const int s_lane = (grp << 2) | g_in_128;
                    sc2 = __shfl_sync(0xFFFFFFFFu, sc_cur, s_lane);
                    zp2 = __shfl_sync(0xFFFFFFFFu, zp_cur, s_lane);
                }
                const int src_lane = (grp << 2) | (ks >> 1);
                const uint32_t u_lo = __shfl_sync(0xFFFFFFFFu, (ks & 1) ? wp_cur4.z : wp_cur4.x, src_lane);
                const uint32_t u_hi = __shfl_sync(0xFFFFFFFFu, (ks & 1) ? wp_cur4.w : wp_cur4.y, src_lane);
                const uint32_t b0 = unpack_byte_to_f16x2(u_lo, prmt_sel, sc2, zp2);
                const uint32_t b1 = unpack_byte_to_f16x2(u_hi, prmt_sel, sc2, zp2);

                const int ko = ks << 4;
                {
                    uint32_t a0, a1, a2, a3;
                    ldmatrix_x4_b16(a0, a1, a2, a3, smem_u32(&smem.sX[stage][ldm_row][ko + ldm_col]));
                    mma_m16n8k16_f16(d00, d01, d02, d03, a0, a1, a2, a3, b0, b1);
                }
                if constexpr (MT >= 2) {
                    uint32_t a0, a1, a2, a3;
                    ldmatrix_x4_b16(a0, a1, a2, a3, smem_u32(&smem.sX[stage][16 + ldm_row][ko + ldm_col]));
                    mma_m16n8k16_f16(d10, d11, d12, d13, a0, a1, a2, a3, b0, b1);
                }
                if constexpr (MT >= 3) {
                    uint32_t a0, a1, a2, a3;
                    ldmatrix_x4_b16(a0, a1, a2, a3, smem_u32(&smem.sX[stage][32 + ldm_row][ko + ldm_col]));
                    mma_m16n8k16_f16(d20, d21, d22, d23, a0, a1, a2, a3, b0, b1);
                }
                if constexpr (MT >= 4) {
                    uint32_t a0, a1, a2, a3;
                    ldmatrix_x4_b16(a0, a1, a2, a3, smem_u32(&smem.sX[stage][48 + ldm_row][ko + ldm_col]));
                    mma_m16n8k16_f16(d30, d31, d32, d33, a0, a1, a2, a3, b0, b1);
                }
            }
            wp_cur4 = wp_nxt4;
        }
        sc_cur = sc_nxt;
        zp_cur = zp_nxt;
        if (it + 1 < n_iters) {
            __syncthreads();
        }
    }

    // ---- BN_CTA == 16 K-split warp reduction via aliased `smem.sRed` ----
    if constexpr (K_SPLIT2) {
        __syncthreads();  // ensure all kw warps finished reading smem.sX before writing aliased smem.sRed!
        const int n_loc0 = n_warp * 8 + (lt << 1);
        if (kw == 1) {
            smem.sRed[grp][n_loc0 + 0]     = d00;
            smem.sRed[grp][n_loc0 + 1]     = d01;
            smem.sRed[grp + 8][n_loc0 + 0] = d02;
            smem.sRed[grp + 8][n_loc0 + 1] = d03;
            if constexpr (MT >= 2) {
                smem.sRed[16 + grp][n_loc0 + 0] = d10;
                smem.sRed[16 + grp][n_loc0 + 1] = d11;
                smem.sRed[24 + grp][n_loc0 + 0] = d12;
                smem.sRed[24 + grp][n_loc0 + 1] = d13;
            }
            if constexpr (MT >= 3) {
                smem.sRed[32 + grp][n_loc0 + 0] = d20;
                smem.sRed[32 + grp][n_loc0 + 1] = d21;
                smem.sRed[40 + grp][n_loc0 + 0] = d22;
                smem.sRed[40 + grp][n_loc0 + 1] = d23;
            }
            if constexpr (MT >= 4) {
                smem.sRed[48 + grp][n_loc0 + 0] = d30;
                smem.sRed[48 + grp][n_loc0 + 1] = d31;
                smem.sRed[56 + grp][n_loc0 + 0] = d32;
                smem.sRed[56 + grp][n_loc0 + 1] = d33;
            }
        }
        __syncthreads();
        if (kw == 0) {
            d00 += smem.sRed[grp][n_loc0 + 0];
            d01 += smem.sRed[grp][n_loc0 + 1];
            d02 += smem.sRed[grp + 8][n_loc0 + 0];
            d03 += smem.sRed[grp + 8][n_loc0 + 1];
            if constexpr (MT >= 2) {
                d10 += smem.sRed[16 + grp][n_loc0 + 0];
                d11 += smem.sRed[16 + grp][n_loc0 + 1];
                d12 += smem.sRed[24 + grp][n_loc0 + 0];
                d13 += smem.sRed[24 + grp][n_loc0 + 1];
            }
            if constexpr (MT >= 3) {
                d20 += smem.sRed[32 + grp][n_loc0 + 0];
                d21 += smem.sRed[32 + grp][n_loc0 + 1];
                d22 += smem.sRed[40 + grp][n_loc0 + 0];
                d23 += smem.sRed[40 + grp][n_loc0 + 1];
            }
            if constexpr (MT >= 4) {
                d30 += smem.sRed[48 + grp][n_loc0 + 0];
                d31 += smem.sRed[48 + grp][n_loc0 + 1];
                d32 += smem.sRed[56 + grp][n_loc0 + 0];
                d33 += smem.sRed[56 + grp][n_loc0 + 1];
            }
        }
    }

    // ---- Epilogue: Direct (split_k == 1) vs Coherent Split-K Finalize ----
    if (split_k == 1) {
        if (!K_SPLIT2 || kw == 0) {
            const int n_loc0 = n_warp * 8 + (lt << 1);
            const int n_g0 = n_cta + n_loc0;
            const int n_g1 = n_g0 + 1;
            const float b_0 = load_bias_val(BIAS, n_g0, N, has_bias, bias_is_f32);
            const float b_1 = load_bias_val(BIAS, n_g1, N, has_bias, bias_is_f32);

            auto store_pair = [&](int m_row, float v0, float v1) {
                if (m_row >= M) return;
                v0 += b_0;
                v1 += b_1;
                if constexpr (FUSE_SILU) {
                    const int c_out = n_g0 >> 1;
                    if (c_out < (N >> 1)) {
                        const float y = silu_mul_f32(v0, v1);
                        if constexpr (OUT_F32) {
                            static_cast<float*>(OUT)[static_cast<int64_t>(m_row) * so + c_out] = y;
                        } else {
                            static_cast<uint16_t*>(OUT)[static_cast<int64_t>(m_row) * so + c_out] = f32_to_bf16_bits(y);
                        }
                    }
                } else if constexpr (OUT_F32) {
                    float* dst = static_cast<float*>(OUT) + static_cast<int64_t>(m_row) * so + n_g0;
                    if (n_g1 < N) {
                        dst[0] = v0;
                        dst[1] = v1;
                    } else if (n_g0 < N) {
                        dst[0] = v0;
                    }
                } else {
                    uint16_t* dst = static_cast<uint16_t*>(OUT) + static_cast<int64_t>(m_row) * so + n_g0;
                    if (n_g1 < N) {
                        *reinterpret_cast<uint32_t*>(dst) = pack_bf16x2_f32(v0, v1);
                    } else if (n_g0 < N) {
                        dst[0] = f32_to_bf16_bits(v0);
                    }
                }
            };

            store_pair(grp,     d00, d01);
            store_pair(grp + 8, d02, d03);
            if constexpr (MT >= 2) {
                store_pair(16 + grp, d10, d11);
                store_pair(24 + grp, d12, d13);
            }
            if constexpr (MT >= 3) {
                store_pair(32 + grp, d20, d21);
                store_pair(40 + grp, d22, d23);
            }
            if constexpr (MT >= 4) {
                store_pair(48 + grp, d30, d31);
                store_pair(56 + grp, d32, d33);
            }
        }
        if (use_pdl) {
            __syncthreads();
            if (tid == 0) gdc_launch_dependents();
        }
        return;
    }

    const int64_t part_panel_stride = static_cast<int64_t>(split_k) * BM_TILE * BN_CTA;
    const int64_t part_split_stride = static_cast<int64_t>(BM_TILE) * BN_CTA;
    float* p_base = PART + static_cast<int64_t>(pid_n) * part_panel_stride
                         + static_cast<int64_t>(ks_id) * part_split_stride;

    if (!K_SPLIT2 || kw == 0) {
        const int n_loc0 = n_warp * 8 + (lt << 1);
        if (grp < M)     st_cg_f32x2(p_base + grp * BN_CTA + n_loc0, d00, d01);
        if (grp + 8 < M) st_cg_f32x2(p_base + (grp + 8) * BN_CTA + n_loc0, d02, d03);
        if constexpr (MT >= 2) {
            if (16 + grp < M) st_cg_f32x2(p_base + (16 + grp) * BN_CTA + n_loc0, d10, d11);
            if (24 + grp < M) st_cg_f32x2(p_base + (24 + grp) * BN_CTA + n_loc0, d12, d13);
        }
        if constexpr (MT >= 3) {
            if (32 + grp < M) st_cg_f32x2(p_base + (32 + grp) * BN_CTA + n_loc0, d20, d21);
            if (40 + grp < M) st_cg_f32x2(p_base + (40 + grp) * BN_CTA + n_loc0, d22, d23);
        }
        if constexpr (MT >= 4) {
            if (48 + grp < M) st_cg_f32x2(p_base + (48 + grp) * BN_CTA + n_loc0, d30, d31);
            if (56 + grp < M) st_cg_f32x2(p_base + (56 + grp) * BN_CTA + n_loc0, d32, d33);
        }
    }

    __syncthreads();
    __threadfence();

    if (tid == 0) {
        const uint32_t prev = atomic_add_acq_rel_gpu(SEM + pid_n, 1u);
        const int last = (static_cast<int>(prev) + 1 == split_k) ? 1 : 0;
        if (last) {
            st_cg_u32(SEM + pid_n, 0u);
        }
        sIsLast = last;
    }
    __syncthreads();

    if (sIsLast != 0) {
        const float* p_panel = PART + static_cast<int64_t>(pid_n) * part_panel_stride;
        finalize_split_k_out<BN_CTA, N_THREADS, FUSE_SILU, OUT_F32>(
            p_panel, BIAS, OUT, M, N, n_cta, split_k, part_split_stride, so,
            has_bias, bias_is_f32, tid
        );
        if (use_pdl) {
            __syncthreads();
            if (tid == 0) gdc_launch_dependents();
        }
    }
}

// ============================================================================
// Pruned Dispatch Helpers (Zero Dead Instantiations!)
// ============================================================================

template <int BN_CTA, int M_CAP, bool FUSE_SILU, bool OUT_F32>
static inline int dispatch_simt_gs_sym(
    int gs, int sym, bool use_pdl, dim3 grid, cudaStream_t stream,
    const uint16_t* X1, const uint16_t* X2,
    const uint8_t* WP, const uint16_t* SCP, const uint16_t* ZPP, const void* BIAS,
    float* PART, uint32_t* SEM, void* OUT,
    int M, int N, int K, int wp_bn, int bk, int split_k, int iters_per_split,
    int64_t sx1, int64_t sx2, int64_t so,
    int rnn_mode, int n_reqs, int k_steps_rnn,
    int has_bias, int bias_is_f32
) {
    dim3 block(128u);
    int pdl_i = use_pdl ? 1 : 0;
    cudaError_t err;
    if (gs == 32) {
        if (sym) {
            err = launch_kernel_ex(use_pdl, grid, block, 0, stream,
                drafter_u4_simt_m14_v14_kernel<BN_CTA, M_CAP, 32, true, FUSE_SILU, OUT_F32>,
                X1, X2, WP, SCP, ZPP, BIAS, PART, SEM, OUT,
                M, N, K, wp_bn, bk, split_k, iters_per_split,
                sx1, sx2, so, rnn_mode, n_reqs, k_steps_rnn, has_bias, bias_is_f32, pdl_i);
        } else {
            err = launch_kernel_ex(use_pdl, grid, block, 0, stream,
                drafter_u4_simt_m14_v14_kernel<BN_CTA, M_CAP, 32, false, FUSE_SILU, OUT_F32>,
                X1, X2, WP, SCP, ZPP, BIAS, PART, SEM, OUT,
                M, N, K, wp_bn, bk, split_k, iters_per_split,
                sx1, sx2, so, rnn_mode, n_reqs, k_steps_rnn, has_bias, bias_is_f32, pdl_i);
        }
    } else {
        if (sym) {
            err = launch_kernel_ex(use_pdl, grid, block, 0, stream,
                drafter_u4_simt_m14_v14_kernel<BN_CTA, M_CAP, 64, true, FUSE_SILU, OUT_F32>,
                X1, X2, WP, SCP, ZPP, BIAS, PART, SEM, OUT,
                M, N, K, wp_bn, bk, split_k, iters_per_split,
                sx1, sx2, so, rnn_mode, n_reqs, k_steps_rnn, has_bias, bias_is_f32, pdl_i);
        } else {
            err = launch_kernel_ex(use_pdl, grid, block, 0, stream,
                drafter_u4_simt_m14_v14_kernel<BN_CTA, M_CAP, 64, false, FUSE_SILU, OUT_F32>,
                X1, X2, WP, SCP, ZPP, BIAS, PART, SEM, OUT,
                M, N, K, wp_bn, bk, split_k, iters_per_split,
                sx1, sx2, so, rnn_mode, n_reqs, k_steps_rnn, has_bias, bias_is_f32, pdl_i);
        }
    }
    return (err == cudaSuccess) ? 0 : static_cast<int>(err);
}

template <int BN_CTA, bool FUSE_SILU, bool OUT_F32>
static inline int dispatch_bm8_gs_sym(
    int gs, int sym, bool use_pdl, dim3 grid, cudaStream_t stream,
    const uint16_t* X1, const uint16_t* X2,
    const uint8_t* WP, const uint16_t* SCP, const uint16_t* ZPP, const void* BIAS,
    float* PART, uint32_t* SEM, void* OUT,
    int M, int N, int K, int wp_bn, int bk, int split_k, int iters_per_split,
    int64_t sx1, int64_t sx2, int64_t so,
    int rnn_mode, int n_reqs, int k_steps_rnn,
    int has_bias, int bias_is_f32
) {
    dim3 block(BN_CTA == 64 ? 256u : 128u);
    int pdl_i = use_pdl ? 1 : 0;
    cudaError_t err;
    if (gs == 32) {
        if (sym) {
            err = launch_kernel_ex(use_pdl, grid, block, 0, stream,
                drafter_u4_tc_bm8_v14_kernel<BN_CTA, 32, true, FUSE_SILU, OUT_F32>,
                X1, X2, WP, SCP, ZPP, BIAS, PART, SEM, OUT,
                M, N, K, wp_bn, bk, split_k, iters_per_split,
                sx1, sx2, so, rnn_mode, n_reqs, k_steps_rnn, has_bias, bias_is_f32, pdl_i);
        } else {
            err = launch_kernel_ex(use_pdl, grid, block, 0, stream,
                drafter_u4_tc_bm8_v14_kernel<BN_CTA, 32, false, FUSE_SILU, OUT_F32>,
                X1, X2, WP, SCP, ZPP, BIAS, PART, SEM, OUT,
                M, N, K, wp_bn, bk, split_k, iters_per_split,
                sx1, sx2, so, rnn_mode, n_reqs, k_steps_rnn, has_bias, bias_is_f32, pdl_i);
        }
    } else {
        if (sym) {
            err = launch_kernel_ex(use_pdl, grid, block, 0, stream,
                drafter_u4_tc_bm8_v14_kernel<BN_CTA, 64, true, FUSE_SILU, OUT_F32>,
                X1, X2, WP, SCP, ZPP, BIAS, PART, SEM, OUT,
                M, N, K, wp_bn, bk, split_k, iters_per_split,
                sx1, sx2, so, rnn_mode, n_reqs, k_steps_rnn, has_bias, bias_is_f32, pdl_i);
        } else {
            err = launch_kernel_ex(use_pdl, grid, block, 0, stream,
                drafter_u4_tc_bm8_v14_kernel<BN_CTA, 64, false, FUSE_SILU, OUT_F32>,
                X1, X2, WP, SCP, ZPP, BIAS, PART, SEM, OUT,
                M, N, K, wp_bn, bk, split_k, iters_per_split,
                sx1, sx2, so, rnn_mode, n_reqs, k_steps_rnn, has_bias, bias_is_f32, pdl_i);
        }
    }
    return (err == cudaSuccess) ? 0 : static_cast<int>(err);
}

template <int BN_CTA, int MT, bool FUSE_SILU, bool OUT_F32>
static inline int dispatch_mt_gs_sym(
    int gs, int sym, bool use_pdl, dim3 grid, cudaStream_t stream,
    const uint16_t* X1, const uint16_t* X2,
    const uint8_t* WP, const uint16_t* SCP, const uint16_t* ZPP, const void* BIAS,
    float* PART, uint32_t* SEM, void* OUT,
    int M, int N, int K, int wp_bn, int bk, int split_k, int iters_per_split,
    int64_t sx1, int64_t sx2, int64_t so,
    int rnn_mode, int n_reqs, int k_steps_rnn,
    int has_bias, int bias_is_f32
) {
    dim3 block(BN_CTA == 64 ? 256u : 128u);
    int pdl_i = use_pdl ? 1 : 0;
    cudaError_t err;
    if (gs == 32) {
        if (sym) {
            err = launch_kernel_ex(use_pdl, grid, block, 0, stream,
                drafter_u4_tc_m964_v14_kernel<BN_CTA, MT, 32, true, FUSE_SILU, OUT_F32>,
                X1, X2, WP, SCP, ZPP, BIAS, PART, SEM, OUT,
                M, N, K, wp_bn, bk, split_k, iters_per_split,
                sx1, sx2, so, rnn_mode, n_reqs, k_steps_rnn, has_bias, bias_is_f32, pdl_i);
        } else {
            err = launch_kernel_ex(use_pdl, grid, block, 0, stream,
                drafter_u4_tc_m964_v14_kernel<BN_CTA, MT, 32, false, FUSE_SILU, OUT_F32>,
                X1, X2, WP, SCP, ZPP, BIAS, PART, SEM, OUT,
                M, N, K, wp_bn, bk, split_k, iters_per_split,
                sx1, sx2, so, rnn_mode, n_reqs, k_steps_rnn, has_bias, bias_is_f32, pdl_i);
        }
    } else {
        if (sym) {
            err = launch_kernel_ex(use_pdl, grid, block, 0, stream,
                drafter_u4_tc_m964_v14_kernel<BN_CTA, MT, 64, true, FUSE_SILU, OUT_F32>,
                X1, X2, WP, SCP, ZPP, BIAS, PART, SEM, OUT,
                M, N, K, wp_bn, bk, split_k, iters_per_split,
                sx1, sx2, so, rnn_mode, n_reqs, k_steps_rnn, has_bias, bias_is_f32, pdl_i);
        } else {
            err = launch_kernel_ex(use_pdl, grid, block, 0, stream,
                drafter_u4_tc_m964_v14_kernel<BN_CTA, MT, 64, false, FUSE_SILU, OUT_F32>,
                X1, X2, WP, SCP, ZPP, BIAS, PART, SEM, OUT,
                M, N, K, wp_bn, bk, split_k, iters_per_split,
                sx1, sx2, so, rnn_mode, n_reqs, k_steps_rnn, has_bias, bias_is_f32, pdl_i);
        }
    }
    return (err == cudaSuccess) ? 0 : static_cast<int>(err);
}

}  // namespace

extern "C" {

int drafter_u4_c18_v14_version() {
    return 14;
}

int drafter_u4_c18_v14_fwd(
    const void* x1,
    const void* x2,          // nullptr if rnn_mode == 0
    const void* wp,
    const void* scp,
    const void* zpp,         // nullptr if sym != 0
    const void* bias,        // nullptr if !has_bias
    void* part,              // nullptr if split_k == 1
    void* sem,               // nullptr if split_k == 1
    void* out,
    int M,
    int N,
    int K,
    int wp_bn,               // panel BN in WP/SCP/ZPP: 16, 32, or 64
    int bn_cta,              // requested CTA tile: 16, 32, 64 (0 = auto)
    int bk,                  // 128 or 256
    int gs,                  // 32 or 64
    int split_k,             // >= 1
    int64_t sx1,
    int64_t sx2,
    int64_t so,
    int sym,
    int fuse_silu,
    int out_f32,
    int rnn_mode,            // 0=normal, 1=FUSE_RNN, 2=FUSE_RNN_BATCH
    int n_reqs,
    int k_steps_rnn,
    int has_bias,
    int bias_is_f32,
    int use_pdl,
    int m18_mode,            // 0=auto (M<=4 SIMT, 5..8 BM8-TC, 9..64 MT-TC), 1=BM8-TC for M<=8, 3=MT-TC all
    uint64_t stream_u64
) {
    if (M <= 0 || M > 64 || N <= 0 || K <= 0) return -1;
    if ((K & 127) != 0 || (bk != 128 && bk != 256) || (K % bk) != 0) return -2;
    if (gs != 32 && gs != 64) return -3;
    if (wp_bn != 16 && wp_bn != 32 && wp_bn != 64) return -4;
    if (split_k < 1) return -5;
    if (split_k > 1 && (part == nullptr || sem == nullptr)) return -6;
    if (rnn_mode != 0 && x2 == nullptr) return -7;
    if (rnn_mode != 0 && ((K & 255) != 0)) return -19;
    if (!sym && zpp == nullptr) return -8;
    if ((reinterpret_cast<uintptr_t>(x1) & 15u) != 0 || (sx1 & 7) != 0) return -9;
    if (x2 != nullptr && ((reinterpret_cast<uintptr_t>(x2) & 15u) != 0 || (sx2 & 7) != 0)) return -10;
    if (fuse_silu && ((N & 1) != 0)) return -11;
    if (rnn_mode == 2 && (n_reqs <= 0 || k_steps_rnn <= 0)) return -18;

    const int mt = (M + 15) >> 4;  // 1..4
    const bool use_tier1_simt = (M <= 4) && (m18_mode == 0 || m18_mode == 2);
    const bool use_tier2_bm8  = (M <= 8) && !use_tier1_simt && (m18_mode != 3);

    int eff_bn = bn_cta;
    if (eff_bn == 0) {
        if (wp_bn == 16) {
            eff_bn = 16;
        } else if (wp_bn == 64 && !fuse_silu && !out_f32 && !use_tier1_simt && M >= 17 && N >= 8192) {
            eff_bn = 64;
        } else {
            eff_bn = 32;
        }
    }
    if (use_tier1_simt && eff_bn == 64) {
        eff_bn = 32;  // Tier 1 SIMT uses 128-thr BN_CTA in {16, 32}
    }
    if (eff_bn != 16 && eff_bn != 32 && eff_bn != 64) return -12;
    if (eff_bn > wp_bn || (wp_bn % eff_bn) != 0) return -13;
    if (eff_bn == 16 && (fuse_silu || out_f32)) return -14;
    if (eff_bn == 64 && (fuse_silu || out_f32 || (!use_tier2_bm8 && mt < 2))) return -15;
    if (fuse_silu && out_f32) return -16;
    if (!use_tier1_simt && !out_f32 && !fuse_silu
        && (((reinterpret_cast<uintptr_t>(out) & 3u) != 0) || ((so & 1) != 0))) {
        return -17;
    }

    const int total_iters = K >> 7;
    const int bk_iters = bk >> 7;
    int iters_per_split = (total_iters + split_k - 1) / split_k;
    iters_per_split = ((iters_per_split + bk_iters - 1) / bk_iters) * bk_iters;
    const int eff_split_k = (total_iters + iters_per_split - 1) / iters_per_split;
    const int n_sub = (N + eff_bn - 1) / eff_bn;

    dim3 grid(static_cast<unsigned int>(n_sub), static_cast<unsigned int>(eff_split_k));
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(stream_u64));
    const bool pdl_b = (use_pdl != 0);

    const uint16_t* x1p = static_cast<const uint16_t*>(x1);
    const uint16_t* x2p = static_cast<const uint16_t*>(x2);
    const uint8_t*  wpp = static_cast<const uint8_t*>(wp);
    const uint16_t* scp_p = static_cast<const uint16_t*>(scp);
    const uint16_t* zpp_p = static_cast<const uint16_t*>(zpp);
    float* part_p = static_cast<float*>(part);
    uint32_t* sem_p = static_cast<uint32_t*>(sem);

    // ---- Tier 1 Dispatch (`M = 1..4` Zero-SMEM SIMT) ----
    if (use_tier1_simt) {
        #define CALL_SIMT(BN_V, MC_V, SILU_V, F32_V) \
            dispatch_simt_gs_sym<BN_V, MC_V, SILU_V, F32_V>( \
                gs, sym, pdl_b, grid, stream, \
                x1p, x2p, wpp, scp_p, zpp_p, bias, part_p, sem_p, out, \
                M, N, K, wp_bn, bk, eff_split_k, iters_per_split, \
                sx1, sx2, so, rnn_mode, n_reqs, k_steps_rnn, has_bias, bias_is_f32)
        if (eff_bn == 16) {
            if (M == 1) return CALL_SIMT(16, 1, false, false);
            if (M == 2) return CALL_SIMT(16, 2, false, false);
            return CALL_SIMT(16, 4, false, false);
        }
        if (fuse_silu) {
            if (M == 1) return CALL_SIMT(32, 1, true, false);
            if (M == 2) return CALL_SIMT(32, 2, true, false);
            return CALL_SIMT(32, 4, true, false);
        }
        if (out_f32) {
            if (M == 1) return CALL_SIMT(32, 1, false, true);
            if (M == 2) return CALL_SIMT(32, 2, false, true);
            return CALL_SIMT(32, 4, false, true);
        }
        if (M == 1) return CALL_SIMT(32, 1, false, false);
        if (M == 2) return CALL_SIMT(32, 2, false, false);
        return CALL_SIMT(32, 4, false, false);
        #undef CALL_SIMT
    }

    // ---- Tier 2 Dispatch (`M = 1..8` Dedicated BM=8 FP16-TC) ----
    if (use_tier2_bm8) {
        #define CALL_BM8(BN_V, SILU_V, F32_V) \
            dispatch_bm8_gs_sym<BN_V, SILU_V, F32_V>( \
                gs, sym, pdl_b, grid, stream, \
                x1p, x2p, wpp, scp_p, zpp_p, bias, part_p, sem_p, out, \
                M, N, K, wp_bn, bk, eff_split_k, iters_per_split, \
                sx1, sx2, so, rnn_mode, n_reqs, k_steps_rnn, has_bias, bias_is_f32)
        if (eff_bn == 16) return CALL_BM8(16, false, false);
        if (eff_bn == 64) return CALL_BM8(64, false, false);
        if (fuse_silu)    return CALL_BM8(32, true, false);
        if (out_f32)      return CALL_BM8(32, false, true);
        return CALL_BM8(32, false, false);
        #undef CALL_BM8
    }

    // ---- Tier 3 Dispatch (`M = 9..64` Upgraded MT=1..4 FP16-TC) ----
    #define CALL_MT(BN_V, MT_V, SILU_V, F32_V) \
        dispatch_mt_gs_sym<BN_V, MT_V, SILU_V, F32_V>( \
            gs, sym, pdl_b, grid, stream, \
            x1p, x2p, wpp, scp_p, zpp_p, bias, part_p, sem_p, out, \
            M, N, K, wp_bn, bk, eff_split_k, iters_per_split, \
            sx1, sx2, so, rnn_mode, n_reqs, k_steps_rnn, has_bias, bias_is_f32)

    if (eff_bn == 16) {
        if (mt == 1) return CALL_MT(16, 1, false, false);
        if (mt == 2) return CALL_MT(16, 2, false, false);
        if (mt == 3) return CALL_MT(16, 3, false, false);
        return CALL_MT(16, 4, false, false);
    }
    if (eff_bn == 64) {
        if (mt == 2) return CALL_MT(64, 2, false, false);
        if (mt == 3) return CALL_MT(64, 3, false, false);
        return CALL_MT(64, 4, false, false);
    }
    if (fuse_silu) {
        if (mt == 1) return CALL_MT(32, 1, true, false);
        if (mt == 2) return CALL_MT(32, 2, true, false);
        if (mt == 3) return CALL_MT(32, 3, true, false);
        return CALL_MT(32, 4, true, false);
    }
    if (out_f32) {
        if (mt == 1) return CALL_MT(32, 1, false, true);
        if (mt == 2) return CALL_MT(32, 2, false, true);
        if (mt == 3) return CALL_MT(32, 3, false, true);
        return CALL_MT(32, 4, false, true);
    }
    if (mt == 1) return CALL_MT(32, 1, false, false);
    if (mt == 2) return CALL_MT(32, 2, false, false);
    if (mt == 3) return CALL_MT(32, 3, false, false);
    return CALL_MT(32, 4, false, false);

    #undef CALL_MT
}

// 100% Drop-In Backwards-Compatible Wave-V9 ABI Aliases
int drafter_u4_tc_v9_version() {
    return 9;
}

int drafter_u4_tc_v9_fwd(
    const void* x1,
    const void* x2,
    const void* wp,
    const void* scp,
    const void* zpp,
    const void* bias,
    void* part,
    void* sem,
    void* out,
    int M, int N, int K,
    int wp_bn, int bn_cta, int bk, int gs, int split_k,
    int64_t sx1, int64_t sx2, int64_t so,
    int sym, int fuse_silu, int out_f32,
    int rnn_mode, int n_reqs, int k_steps_rnn,
    int has_bias, int bias_is_f32, int use_pdl,
    uint64_t stream_u64
) {
    return drafter_u4_c18_v14_fwd(
        x1, x2, wp, scp, zpp, bias, part, sem, out,
        M, N, K, wp_bn, bn_cta, bk, gs, split_k,
        sx1, sx2, so, sym, fuse_silu, out_f32,
        rnn_mode, n_reqs, k_steps_rnn, has_bias, bias_is_f32, use_pdl,
        0, stream_u64
    );
}

}  // extern "C"
