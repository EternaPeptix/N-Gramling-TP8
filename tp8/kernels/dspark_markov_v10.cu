// SPDX-License-Identifier: Apache-2.0
// Wave-V10 Native SM121a (GB10) 0-ULP, 0-Spill, 0-Stack, 0-#206-PDL-Hazard CUDA Kernels
// (100% Drop-In ABI Superset of libdspark_markov_v7.so; exports dspark_markov_v10_version() == 10)
//
// V10 Additions over V7:
//   1. Full #206 PDL Audit & Fix across ALL 28 __global__ kernels: gdc_wait() strictly before
//      gdc_launch_dependents(); gdc_launch_dependents() strictly AFTER all global stores +
//      __syncthreads()/__syncwarp() on EVERY path (incl. early-exit/skip); zero warp-divergent
//      or entry-level gdc_launch_dependents().
//   2. Native SM121a CUDA Batch-All Early-Stop (HAS_SKIP / es_mode == 5, B = 2..8) across
//      dspark_markov_rnn_step_tc_v10_kernel, dspark_fold_argmax_tc_v10_kernel,
//      dspark_select_and_rnn_step_tc_v10_kernel (cross-step fusion + ping-pong SURV_B/SKIP),
//      and dspark_argmax_select_batch_es_v10_kernel (standalone & final-step).
//   3. TOP2 + HAS_ID_MAP support in dspark_fold_argmax_r1_v10_kernel (B==1 TREE_SPEC + VOCAB_HOT).
//   4. Full drop-in *_v7_fwd ABI + *_v10_fwd exports + dspark_markov_v10_version() == 10.
//
// Inherited V7 Foundation:
//   (A) 5-Layer DSpark Drafter Trunk & Overlapped Context-KV / f_rnn Side Stream
//       - `dspark_mix_rmsnorm_5layer_v10_kernel`: `grid = (3, rows)` L2-co-scheduled 5-layer mix + `slot < 0` skip
//       - `dspark_qk_norm_rope_multi_v10_kernel` / `dspark_ctx_kv_1warp_v10_kernel` (fixed `sv_h` ABI + `slot < 0` exit)
//       - `dspark_embed_block_ln_v10_kernel`, `dspark_silu_mul_v10_kernel`, `dspark_quant_fp8_rowwise_v10_kernel`,
//         `dspark_silu_mul_quant_fp8_v10_kernel`, `dspark_add_softcap_v10_kernel`
//   (B) K=5..7 Markov RNN + Fold-Argmax + Cross-Step RDMA-Overlapped Select + Verify-W / Tree Pack
//       - 48-CTA full-machine `TC48` batched Markov RNN (`B = 2..16`: 48 CTAs = 100% SMs vs 8 CTAs in V6 = 6x occupancy!;
//         `B = 17..32`: 96 CTAs = 2.0x48 waves!) + 48-CTA `r1` (`B = 1`)
//       - `BM = 8` (`B = 2..8`) / `BM = 16` (`B = 9..16`) / `BM = 32` (`B = 17..32`) Tensor-Core `fold_argmax_tc`:
//         loads 6.43 MB FP8 `fold_w8` ONCE per step via 128-bit coalesced `ld_ca_u4` into `sFW` (0 strided 2B loads!)
//         + hardware-coherent `st.cg + __threadfence() + atom.acq_rel.gpu` last-CTA reduction!
//   (C) SyncFree Greedy Target Verification (Phase 2A Eager, coherent `SPLIT=1` 16 B/row AG across C=2..8 & C=32)
//       & Speculator Step Epilogue (Phase 2B)
//
// Strict SM121a Build Rules (enforced by build_drafter_v10.sh):
//   /usr/local/cuda/bin/nvcc -O3 -std=c++17 -gencode arch=compute_121a,code=sm_121a \
//       --fmad=false --shared -Xcompiler -fPIC --ptxas-options=-v \
//       dspark_markov_v10.cu -o libdspark_markov_v10.so
//   - ZERO stack frames (0 B), ZERO spill stores (0 B), ZERO spill loads (0 B)
//   - ZERO cvta.to.shared / __cvta_generic_to_shared
//   - ZERO runtime-indexed local C arrays & ZERO address-taken local uint4/uint2 pointer-punning
//   - ZERO cooperative launches (100% immune to #161 l2pf._side deadlock)
//   - Literal 0-ULP bit-identity (nd == 0) and leftmost greedy argmax tie-breaking

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stddef.h>
#include <math.h>

namespace {

static constexpr float LOG2E_F      = 1.4426950408889634f;
static constexpr float LN2_F        = 0.6931471805599453f;
static constexpr float DENORM_MIN_F = 1.1754943508222875e-38f;
static constexpr int   HIDDEN_DS    = 4608;
static constexpr int   TAPS_DS      = 8;
static constexpr int   RANK_MK      = 256;

// ============================================================================
// Low-Level PTX & Exact IEEE-754 Intrinsics (SM121a, --fmad=false)
// ============================================================================

static __device__ __forceinline__ void gdc_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

static __device__ __forceinline__ void gdc_launch_dependents() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

static __device__ __forceinline__ uint4 ld_cs_u4(const void* p) {
    uint4 v;
#if defined(__CUDA_ARCH__)
    asm volatile("ld.global.cs.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
#else
    v = *reinterpret_cast<const uint4*>(p);
#endif
    return v;
}

static __device__ __forceinline__ float4 ld_cs_f4(const float* p) {
    float4 v;
#if defined(__CUDA_ARCH__)
    asm volatile("ld.global.cs.v4.f32 {%0,%1,%2,%3}, [%4];"
                 : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
                 : "l"(p));
#else
    v = make_float4(p[0], p[1], p[2], p[3]);
#endif
    return v;
}

static __device__ __forceinline__ uint4 ld_ca_u4(const void* p) {
    uint4 v;
#if defined(__CUDA_ARCH__)
    asm volatile("ld.global.ca.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
#else
    v = *reinterpret_cast<const uint4*>(p);
#endif
    return v;
}

static __device__ __forceinline__ float4 ld_cg_f4(const float* p) {
    float4 v;
#if defined(__CUDA_ARCH__)
    asm volatile("ld.global.cg.v4.f32 {%0,%1,%2,%3}, [%4];"
                 : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
                 : "l"(p));
#else
    v = make_float4(p[0], p[1], p[2], p[3]);
#endif
    return v;
}

static __device__ __forceinline__ void st_cg_f4(float* p, float a, float b, float c, float d) {
#if defined(__CUDA_ARCH__)
    asm volatile("st.global.cg.v4.f32 [%0], {%1,%2,%3,%4};"
                 :: "l"(p), "f"(a), "f"(b), "f"(c), "f"(d) : "memory");
#else
    p[0] = a; p[1] = b; p[2] = c; p[3] = d;
#endif
}

static __device__ __forceinline__ uint32_t atom_acq_rel_gpu_add(uint32_t* p, uint32_t v) {
    uint32_t old;
#if defined(__CUDA_ARCH__)
    asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                 : "=r"(old) : "l"(p), "r"(v) : "memory");
#else
    old = *p; *p = old + v;
#endif
    return old;
}

static __device__ __forceinline__ void st_rel_gpu_u32(uint32_t* p, uint32_t v) {
#if defined(__CUDA_ARCH__)
    asm volatile("st.release.gpu.global.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
#else
    *p = v;
#endif
}

static __device__ __forceinline__ float rsqrt_approx_f32(float x) {
    float y;
#if defined(__CUDA_ARCH__)
    asm("rsqrt.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
#else
    y = 1.0f / sqrtf(x);
#endif
    return y;
}

static __device__ __forceinline__ float sqrt_approx_f32(float x) {
    float y;
#if defined(__CUDA_ARCH__)
    asm("sqrt.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
#else
    y = sqrtf(x);
#endif
    return y;
}

static __device__ __forceinline__ float sqrt_rn_f32(float x) {
    float y;
#if defined(__CUDA_ARCH__)
    asm("sqrt.rn.f32 %0, %1;" : "=f"(y) : "f"(x));
#else
    y = sqrtf(x);
#endif
    return y;
}

static __device__ __forceinline__ float div_full_f32(float a, float b) {
    float y;
#if defined(__CUDA_ARCH__)
    asm("div.full.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
#else
    y = a / b;
#endif
    return y;
}

static __device__ __forceinline__ float div_rn_f32(float a, float b) {
    float y;
#if defined(__CUDA_ARCH__)
    asm("div.rn.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
#else
    y = a / b;
#endif
    return y;
}

static __device__ __forceinline__ float ex2_approx_ftz_f32(float x) {
    float y;
#if defined(__CUDA_ARCH__)
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
#else
    y = exp2f(x);
#endif
    return y;
}

static __device__ __forceinline__ float ex2_approx_f32(float x) {
    float y;
#if defined(__CUDA_ARCH__)
    asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
#else
    y = exp2f(x);
#endif
    return y;
}

static __device__ __forceinline__ float lg2_approx_ftz_f32(float x) {
    float y;
#if defined(__CUDA_ARCH__)
    asm("lg2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
#else
    y = log2f(x);
#endif
    return y;
}

static __device__ __forceinline__ float lg2_approx_f32(float x) {
    float y;
#if defined(__CUDA_ARCH__)
    asm("lg2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
#else
    y = log2f(x);
#endif
    return y;
}

// Matches Triton `tl.exp(x)` (mode 0: ex2.approx.ftz, mode 1: ex2.approx, mode 2: libdevice expf)
static __device__ __forceinline__ float tl_exp_f32(float x, int exp_mode) {
    if (exp_mode == 0) return ex2_approx_ftz_f32(__fmul_rn(x, LOG2E_F));
    if (exp_mode == 1) return ex2_approx_f32(__fmul_rn(x, LOG2E_F));
    return expf(x);
}

// Matches Triton `tl.log(x)` (mode 0: lg2.approx.ftz, mode 1: lg2.approx, mode 2: libdevice logf)
static __device__ __forceinline__ float tl_log_f32(float x, int exp_mode) {
    if (exp_mode == 0) return __fmul_rn(lg2_approx_ftz_f32(x), LN2_F);
    if (exp_mode == 1) return __fmul_rn(lg2_approx_f32(x), LN2_F);
    return logf(x);
}

// Matches `1.0 / tl.sqrt(ms + eps)` in `_qk_norm_rope_*`
static __device__ __forceinline__ float inv_sqrt_qk_f32(float v, int sqrt_mode) {
    if (sqrt_mode == 0) return div_full_f32(1.0f, sqrt_approx_f32(v));
    if (sqrt_mode == 1) return div_full_f32(1.0f, sqrt_rn_f32(v));
    return rsqrt_approx_f32(v);
}

static __device__ __forceinline__ float bf16_to_f32(uint16_t b) {
    float f;
#if defined(__CUDA_ARCH__)
    asm("cvt.f32.bf16 %0, %1;" : "=f"(f) : "h"(b));
#else
    union { uint32_t u; float fl; } x;
    x.u = static_cast<uint32_t>(b) << 16;
    f = x.fl;
#endif
    return f;
}

static __device__ __forceinline__ uint16_t f32_to_bf16(float f) {
    uint16_t out;
#if defined(__CUDA_ARCH__)
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(out) : "f"(f));
#else
    union { float fl; uint32_t u; } x;
    x.fl = f;
    out = static_cast<uint16_t>((x.u + 0x7FFFu + ((x.u >> 16) & 1u)) >> 16);
#endif
    return out;
}

static __device__ __forceinline__ uint32_t pack_bf16x2(float f0, float f1) {
    uint32_t out;
#if defined(__CUDA_ARCH__)
    asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(out) : "f"(f0), "f"(f1));
#else
    out = static_cast<uint32_t>(f32_to_bf16(f0)) | (static_cast<uint32_t>(f32_to_bf16(f1)) << 16);
#endif
    return out;
}

// Pure register bit-unpack/pack helpers (ZERO address-taken local uint4/uint2 -> 0 B stack frame!)
static __device__ __forceinline__ uint16_t u32_lo16(uint32_t v) {
    return static_cast<uint16_t>(v & 0xFFFFu);
}

static __device__ __forceinline__ uint16_t u32_hi16(uint32_t v) {
    return static_cast<uint16_t>(v >> 16);
}

static __device__ __forceinline__ uint32_t pack_u16x2(uint16_t lo, uint16_t hi) {
    return static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
}

static __device__ __forceinline__ void u4_to_u16x8(uint4 v, uint16_t (&out)[8]) {
    out[0] = u32_lo16(v.x); out[1] = u32_hi16(v.x);
    out[2] = u32_lo16(v.y); out[3] = u32_hi16(v.y);
    out[4] = u32_lo16(v.z); out[5] = u32_hi16(v.z);
    out[6] = u32_lo16(v.w); out[7] = u32_hi16(v.w);
}

static __device__ __forceinline__ uint4 u16x8_to_u4(const uint16_t (&in)[8]) {
    return make_uint4(
        pack_u16x2(in[0], in[1]),
        pack_u16x2(in[2], in[3]),
        pack_u16x2(in[4], in[5]),
        pack_u16x2(in[6], in[7])
    );
}

static __device__ __forceinline__ void u2_to_u16x4(uint2 v, uint16_t (&out)[4]) {
    out[0] = u32_lo16(v.x); out[1] = u32_hi16(v.x);
    out[2] = u32_lo16(v.y); out[3] = u32_hi16(v.y);
}

static __device__ __forceinline__ uint2 u16x4_to_u2(const uint16_t (&in)[4]) {
    return make_uint2(pack_u16x2(in[0], in[1]), pack_u16x2(in[2], in[3]));
}

// Exact match to `dspark_ops._round_bf16` & `sync_free._rn_bf16` (preserves NaN & RNE bit pattern)
static __device__ __forceinline__ float round_bf16_bits_f32(float x) {
    uint32_t bits = __float_as_uint(x);
    if ((bits & 0x7FFFFFFFu) > 0x7F800000u) {
        return __uint_as_float(0x7FC00000u);
    }
    bits = (bits + 0x7FFFu + ((bits >> 16) & 1u)) & 0xFFFF0000u;
    return __uint_as_float(bits);
}

static __device__ __forceinline__ float round_bf16_cvt_f32(float x) {
    return bf16_to_f32(f32_to_bf16(x));
}

static __device__ __forceinline__ void unpack_e4m3x2_to_f32(uint16_t e2, float& f0, float& f1) {
#if defined(__CUDA_ARCH__)
    uint32_t h2;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h2) : "h"(e2));
    uint16_t lo = static_cast<uint16_t>(h2 & 0xFFFFu);
    uint16_t hi = static_cast<uint16_t>(h2 >> 16);
    asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
#else
    (void)e2; f0 = 0.0f; f1 = 0.0f;
#endif
}

static __device__ __forceinline__ uint32_t cvt_e4m3x2_to_bf16x2(uint16_t e2) {
    float f0, f1;
    unpack_e4m3x2_to_f32(e2, f0, f1);
    return pack_bf16x2(f0, f1);
}

static __device__ __forceinline__ uint16_t pack_e4m3x2_f32(float f0, float f1) {
    uint16_t out;
#if defined(__CUDA_ARCH__)
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(out) : "f"(f0), "f"(f1));
#else
    (void)f0; (void)f1; out = 0u;
#endif
    return out;
}

static __device__ __forceinline__ void mma_m16n8k16_bf16(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)
    );
#else
    (void)d0; (void)d1; (void)d2; (void)d3;
    (void)a0; (void)a1; (void)a2; (void)a3; (void)b0; (void)b1;
#endif
}

// Exact dspark_ops softcap: bf16(softcap * tanh(bf16(a + b) / softcap))
static __device__ __forceinline__ float dspark_softcap_f32(float a, float b, float softcap) {
    float x = round_bf16_cvt_f32(__fadd_rn(a, b));
    x = div_full_f32(x, softcap);
    x = tanhf(x);
    x = __fmul_rn(x, softcap);
    return round_bf16_cvt_f32(x);
}

// Exact warp butterfly sum (32 lanes)
static __device__ __forceinline__ float warp_sum_f32(float v) {
    v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, 16));
    v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, 8));
    v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, 4));
    v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, 2));
    v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, 1));
    return v;
}

// Exact 256-thread (8-warp) CTA sum matching Triton `tl.sum(..., 0)`
static __device__ __forceinline__ float cta256_sum_f32(float v, float (&s_warp)[8], int tid) {
    const int warp = tid >> 5;
    const int lane = tid & 31;
    v = warp_sum_f32(v);
    if (lane == 0) s_warp[warp] = v;
    __syncthreads();
    float tot = (tid < 8) ? s_warp[tid] : 0.0f;
    if (warp == 0) {
        tot = __fadd_rn(tot, __shfl_xor_sync(0xFFFFFFFFu, tot, 4));
        tot = __fadd_rn(tot, __shfl_xor_sync(0xFFFFFFFFu, tot, 2));
        tot = __fadd_rn(tot, __shfl_xor_sync(0xFFFFFFFFu, tot, 1));
        if (lane == 0) s_warp[0] = tot;
    }
    __syncthreads();
    return s_warp[0];
}

// Leftmost tie-breaking argmax merge: lowest index wins ties
static __device__ __forceinline__ void merge_max_idx(
    float& m_a, int32_t& i_a, float m_b, int32_t i_b
) {
    const bool better = (m_b > m_a) || ((m_b == m_a) && (i_b < i_a));
    if (better) {
        m_a = m_b;
        i_a = i_b;
    }
}

static __device__ __forceinline__ void warp_reduce_max_idx(float& m, int32_t& idx) {
    #pragma unroll
    for (int off = 16; off >= 1; off >>= 1) {
        float mb = __shfl_xor_sync(0xFFFFFFFFu, m, off);
        int32_t ib = __shfl_xor_sync(0xFFFFFFFFu, idx, off);
        merge_max_idx(m, idx, mb, ib);
    }
}

template <typename KernelT, typename... Args>
static inline cudaError_t launch_kernel_ex(
    bool use_pdl, dim3 grid, dim3 block, size_t smem, cudaStream_t stream,
    KernelT kernel, Args... args
) {
    if (use_pdl) {
        cudaLaunchAttribute attr[1];
        attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr[0].val.programmaticStreamSerializationAllowed = 1;
        cudaLaunchConfig_t cfg{};
        cfg.gridDim = grid;
        cfg.blockDim = block;
        cfg.dynamicSmemBytes = smem;
        cfg.stream = stream;
        cfg.attrs = attr;
        cfg.numAttrs = 1;
        return cudaLaunchKernelEx(&cfg, kernel, args...);
    }
    kernel<<<grid, block, smem, stream>>>(args...);
    return cudaGetLastError();
}

// ============================================================================
// SECTION A: 5-Layer DSpark Trunk & Overlapped Context-KV / f_rnn Kernels
// ============================================================================

// A.1: `dspark_embed_block_ln_v10_kernel`
//   - Fuses embed_tokens(anchor) + embed_norm + mask_embed + L0.input_layernorm (H = 4608).
//   - Mask rows (`row % block_size != 0`, 87.5% of rows!) depend ONLY on static MASK_E and LN_W:
//     computed 100% BEFORE `gdc_wait()` so DRAM reads & RMSNorm hide behind upstream prep!
template <int SPT, bool PDL>
__global__ __launch_bounds__(256, 2)
void dspark_embed_block_ln_v10_kernel(
    const int32_t* __restrict__ IDS_I32,
    const int64_t* __restrict__ IDS_I64,
    int ids_is_i64,
    const uint16_t* __restrict__ EMB_W,
    const uint16_t* __restrict__ EMB_NW,
    const uint16_t* __restrict__ MASK_E,
    const uint16_t* __restrict__ LN_W,
    uint16_t* __restrict__ RES,
    uint16_t* __restrict__ OUT,
    int64_t stride_emb,
    int64_t full_v,
    int block_size,
    float emb_eps,
    float eps
) {
    const int64_t row = static_cast<int64_t>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    const bool is_anchor = ((row % static_cast<int64_t>(block_size)) == 0);
    __shared__ float s_warp[8];

    if constexpr (SPT == 8) {
        // Vectorized 128-bit (`uint4` = 8 bf16) blocked layout: chunks at {0, 2048, 4096}
        const int c0 = tid * 8;
        const int c1 = 2048 + tid * 8;
        const int c2 = 4096 + tid * 8;
        const bool m2 = (c2 < HIDDEN_DS);

        const uint4 lw0 = *reinterpret_cast<const uint4*>(LN_W + c0);
        const uint4 lw1 = *reinterpret_cast<const uint4*>(LN_W + c1);
        const uint4 lw2 = m2 ? *reinterpret_cast<const uint4*>(LN_W + c2) : make_uint4(0u, 0u, 0u, 0u);

        uint4 rx0, rx1, rx2;
        if (!is_anchor) {
            rx0 = *reinterpret_cast<const uint4*>(MASK_E + c0);
            rx1 = *reinterpret_cast<const uint4*>(MASK_E + c1);
            rx2 = m2 ? *reinterpret_cast<const uint4*>(MASK_E + c2) : make_uint4(0u, 0u, 0u, 0u);

            uint16_t p0[8], p1[8], p2[8];
            u4_to_u16x8(rx0, p0);
            u4_to_u16x8(rx1, p1);
            u4_to_u16x8(rx2, p2);
            float sq = 0.0f;
            #pragma unroll
            for (int i = 0; i < 8; ++i) { float v = bf16_to_f32(p0[i]); sq = __fadd_rn(sq, __fmul_rn(v, v)); }
            #pragma unroll
            for (int i = 0; i < 8; ++i) { float v = bf16_to_f32(p1[i]); sq = __fadd_rn(sq, __fmul_rn(v, v)); }
            if (m2) {
                #pragma unroll
                for (int i = 0; i < 8; ++i) { float v = bf16_to_f32(p2[i]); sq = __fadd_rn(sq, __fmul_rn(v, v)); }
            }
            const float inv2 = rsqrt_approx_f32(__fadd_rn(div_full_f32(cta256_sum_f32(sq, s_warp, tid), 4608.0f), eps));

            uint16_t w0[8], w1[8], w2[8];
            u4_to_u16x8(lw0, w0);
            u4_to_u16x8(lw1, w1);
            u4_to_u16x8(lw2, w2);
            uint16_t y0[8], y1[8], y2[8];
            #pragma unroll
            for (int i = 0; i < 8; ++i) y0[i] = f32_to_bf16(__fmul_rn(__fmul_rn(bf16_to_f32(p0[i]), inv2), bf16_to_f32(w0[i])));
            #pragma unroll
            for (int i = 0; i < 8; ++i) y1[i] = f32_to_bf16(__fmul_rn(__fmul_rn(bf16_to_f32(p1[i]), inv2), bf16_to_f32(w1[i])));
            #pragma unroll
            for (int i = 0; i < 8; ++i) y2[i] = m2 ? f32_to_bf16(__fmul_rn(__fmul_rn(bf16_to_f32(p2[i]), inv2), bf16_to_f32(w2[i]))) : 0u;
            const uint4 oy0 = u16x8_to_u4(y0);
            const uint4 oy1 = u16x8_to_u4(y1);
            const uint4 oy2 = u16x8_to_u4(y2);
            if constexpr (PDL) gdc_wait();
            uint16_t* res_row = RES + row * HIDDEN_DS;
            uint16_t* out_row = OUT + row * HIDDEN_DS;
            *reinterpret_cast<uint4*>(res_row + c0) = rx0;
            *reinterpret_cast<uint4*>(res_row + c1) = rx1;
            *reinterpret_cast<uint4*>(out_row + c0) = oy0;
            *reinterpret_cast<uint4*>(out_row + c1) = oy1;
            if (m2) {
                *reinterpret_cast<uint4*>(res_row + c2) = rx2;
                *reinterpret_cast<uint4*>(out_row + c2) = oy2;
            }
            if constexpr (PDL) {
                __syncthreads();
                if (tid == 0) gdc_launch_dependents();
            }
            return;
        }

        // Anchor row: preload static EMB_NW before gdc_wait()
        const uint4 nw0 = *reinterpret_cast<const uint4*>(EMB_NW + c0);
        const uint4 nw1 = *reinterpret_cast<const uint4*>(EMB_NW + c1);
        const uint4 nw2 = m2 ? *reinterpret_cast<const uint4*>(EMB_NW + c2) : make_uint4(0u, 0u, 0u, 0u);
        if constexpr (PDL) gdc_wait();

        const int64_t raw = ids_is_i64 ? IDS_I64[row] : static_cast<int64_t>(IDS_I32[row]);
        const bool oob = (raw < 0) || (raw >= full_v);
        int64_t tok = raw < 0 ? 0 : (raw >= full_v ? (full_v - 1) : raw);
        const uint16_t* emb_row = EMB_W + tok * stride_emb;
        uint16_t ep0[8], ep1[8], ep2[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) { ep0[i] = 0u; ep1[i] = 0u; ep2[i] = 0u; }
        if (!oob) {
            if ((stride_emb & 7) == 0) {
                u4_to_u16x8(*reinterpret_cast<const uint4*>(emb_row + c0), ep0);
                u4_to_u16x8(*reinterpret_cast<const uint4*>(emb_row + c1), ep1);
                if (m2) u4_to_u16x8(*reinterpret_cast<const uint4*>(emb_row + c2), ep2);
            } else {
                #pragma unroll
                for (int i = 0; i < 8; ++i) { ep0[i] = emb_row[c0 + i]; ep1[i] = emb_row[c1 + i]; }
                if (m2) {
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) ep2[i] = emb_row[c2 + i];
                }
            }
        }

        float sq1 = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; ++i) { float v = bf16_to_f32(ep0[i]); sq1 = __fadd_rn(sq1, __fmul_rn(v, v)); }
        #pragma unroll
        for (int i = 0; i < 8; ++i) { float v = bf16_to_f32(ep1[i]); sq1 = __fadd_rn(sq1, __fmul_rn(v, v)); }
        if (m2) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) { float v = bf16_to_f32(ep2[i]); sq1 = __fadd_rn(sq1, __fmul_rn(v, v)); }
        }
        const float inv1 = rsqrt_approx_f32(__fadd_rn(div_full_f32(cta256_sum_f32(sq1, s_warp, tid), 4608.0f), emb_eps));

        uint16_t np0[8], np1[8], np2[8];
        u4_to_u16x8(nw0, np0);
        u4_to_u16x8(nw1, np1);
        u4_to_u16x8(nw2, np2);
        uint16_t rp0[8], rp1[8], rp2[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) rp2[i] = 0u;

        float sq2 = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            uint16_t b = f32_to_bf16(__fmul_rn(__fmul_rn(bf16_to_f32(ep0[i]), inv1), bf16_to_f32(np0[i])));
            rp0[i] = b;
            float vf = bf16_to_f32(b);
            sq2 = __fadd_rn(sq2, __fmul_rn(vf, vf));
        }
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            uint16_t b = f32_to_bf16(__fmul_rn(__fmul_rn(bf16_to_f32(ep1[i]), inv1), bf16_to_f32(np1[i])));
            rp1[i] = b;
            float vf = bf16_to_f32(b);
            sq2 = __fadd_rn(sq2, __fmul_rn(vf, vf));
        }
        if (m2) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                uint16_t b = f32_to_bf16(__fmul_rn(__fmul_rn(bf16_to_f32(ep2[i]), inv1), bf16_to_f32(np2[i])));
                rp2[i] = b;
                float vf = bf16_to_f32(b);
                sq2 = __fadd_rn(sq2, __fmul_rn(vf, vf));
            }
        }
        const float inv2 = rsqrt_approx_f32(__fadd_rn(div_full_f32(cta256_sum_f32(sq2, s_warp, tid), 4608.0f), eps));

        uint16_t w0[8], w1[8], w2[8];
        u4_to_u16x8(lw0, w0);
        u4_to_u16x8(lw1, w1);
        u4_to_u16x8(lw2, w2);
        uint16_t y0[8], y1[8], y2[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) y0[i] = f32_to_bf16(__fmul_rn(__fmul_rn(bf16_to_f32(rp0[i]), inv2), bf16_to_f32(w0[i])));
        #pragma unroll
        for (int i = 0; i < 8; ++i) y1[i] = f32_to_bf16(__fmul_rn(__fmul_rn(bf16_to_f32(rp1[i]), inv2), bf16_to_f32(w1[i])));
        #pragma unroll
        for (int i = 0; i < 8; ++i) y2[i] = m2 ? f32_to_bf16(__fmul_rn(__fmul_rn(bf16_to_f32(rp2[i]), inv2), bf16_to_f32(w2[i]))) : 0u;
        rx0 = u16x8_to_u4(rp0);
        rx1 = u16x8_to_u4(rp1);
        rx2 = u16x8_to_u4(rp2);
        const uint4 oy0 = u16x8_to_u4(y0);
        const uint4 oy1 = u16x8_to_u4(y1);
        const uint4 oy2 = u16x8_to_u4(y2);
        uint16_t* res_row = RES + row * HIDDEN_DS;
        uint16_t* out_row = OUT + row * HIDDEN_DS;
        *reinterpret_cast<uint4*>(res_row + c0) = rx0;
        *reinterpret_cast<uint4*>(res_row + c1) = rx1;
        *reinterpret_cast<uint4*>(out_row + c0) = oy0;
        *reinterpret_cast<uint4*>(out_row + c1) = oy1;
        if (m2) {
            *reinterpret_cast<uint4*>(res_row + c2) = rx2;
            *reinterpret_cast<uint4*>(out_row + c2) = oy2;
        }
        if constexpr (PDL) {
            __syncthreads();
            if (tid == 0) gdc_launch_dependents();
        }
    } else {
        // Strided `SPT == 1` layout (`h = g*256 + tid`, `g = 0..17` for H=4608): 100% warp-coalesced!
        uint16_t x_reg[18];
        if (!is_anchor) {
            float sq2 = 0.0f;
            #pragma unroll
            for (int g = 0; g < 18; ++g) {
                uint16_t b = MASK_E[g * 256 + tid];
                x_reg[g] = b;
                float vf = bf16_to_f32(b);
                sq2 = __fadd_rn(sq2, __fmul_rn(vf, vf));
            }
            const float inv2 = rsqrt_approx_f32(__fadd_rn(div_full_f32(cta256_sum_f32(sq2, s_warp, tid), 4608.0f), eps));
            uint16_t y_reg[18];
            #pragma unroll
            for (int g = 0; g < 18; ++g) {
                y_reg[g] = f32_to_bf16(__fmul_rn(__fmul_rn(bf16_to_f32(x_reg[g]), inv2), bf16_to_f32(LN_W[g * 256 + tid])));
            }
            if constexpr (PDL) gdc_wait();
            uint16_t* res_row = RES + row * HIDDEN_DS;
            uint16_t* out_row = OUT + row * HIDDEN_DS;
            #pragma unroll
            for (int g = 0; g < 18; ++g) {
                res_row[g * 256 + tid] = x_reg[g];
                out_row[g * 256 + tid] = y_reg[g];
            }
            if constexpr (PDL) {
                __syncthreads();
                if (tid == 0) gdc_launch_dependents();
            }
            return;
        }
        if constexpr (PDL) gdc_wait();
        const int64_t raw = ids_is_i64 ? IDS_I64[row] : static_cast<int64_t>(IDS_I32[row]);
        const bool oob = (raw < 0) || (raw >= full_v);
        int64_t tok = raw < 0 ? 0 : (raw >= full_v ? (full_v - 1) : raw);
        const uint16_t* emb_row = EMB_W + tok * stride_emb;
        float sq1 = 0.0f;
        #pragma unroll
        for (int g = 0; g < 18; ++g) {
            uint16_t b = oob ? 0u : emb_row[g * 256 + tid];
            x_reg[g] = b;
            float vf = bf16_to_f32(b);
            sq1 = __fadd_rn(sq1, __fmul_rn(vf, vf));
        }
        const float inv1 = rsqrt_approx_f32(__fadd_rn(div_full_f32(cta256_sum_f32(sq1, s_warp, tid), 4608.0f), emb_eps));
        float sq2 = 0.0f;
        #pragma unroll
        for (int g = 0; g < 18; ++g) {
            uint16_t b = f32_to_bf16(__fmul_rn(__fmul_rn(bf16_to_f32(x_reg[g]), inv1), bf16_to_f32(EMB_NW[g * 256 + tid])));
            x_reg[g] = b;
            float vf = bf16_to_f32(b);
            sq2 = __fadd_rn(sq2, __fmul_rn(vf, vf));
        }
        const float inv2 = rsqrt_approx_f32(__fadd_rn(div_full_f32(cta256_sum_f32(sq2, s_warp, tid), 4608.0f), eps));
        uint16_t* res_row = RES + row * HIDDEN_DS;
        uint16_t* out_row = OUT + row * HIDDEN_DS;
        #pragma unroll
        for (int g = 0; g < 18; ++g) {
            res_row[g * 256 + tid] = x_reg[g];
            out_row[g * 256 + tid] = f32_to_bf16(__fmul_rn(__fmul_rn(bf16_to_f32(x_reg[g]), inv2), bf16_to_f32(LN_W[g * 256 + tid])));
        }
        if constexpr (PDL) {
            __syncthreads();
            if (tid == 0) gdc_launch_dependents();
        }
    }
}

// A.2: Single-layer `dspark_mix_rmsnorm_v10_kernel` & `dspark_context_anchor_mix_rmsnorm_v10_kernel`
template <int SPT, int FMA_MODE, bool IS_ANCHOR, bool PDL>
__global__ __launch_bounds__(256, 2)
void dspark_mix_rmsnorm_v10_kernel(
    const uint16_t* __restrict__ X,
    const int32_t*  __restrict__ QSL,
    const int32_t*  __restrict__ NUM_REJ,
    const float*    __restrict__ ALPHA_F32,
    const uint16_t* __restrict__ ALPHA_BF16,
    int alpha_is_bf16,
    const uint16_t* __restrict__ WEIGHT,
    uint16_t*       __restrict__ OUT,
    int64_t row_stride,
    int64_t tap_stride,
    int64_t out_row_stride,
    float eps
) {
    const int64_t r = static_cast<int64_t>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    __shared__ float s_warp[8];
    __shared__ float s_alpha[TAPS_DS];

    if (tid < TAPS_DS) {
        s_alpha[tid] = alpha_is_bf16 ? bf16_to_f32(ALPHA_BF16[tid]) : ALPHA_F32[tid];
    }
    __syncthreads();
    if constexpr (PDL) gdc_wait();

    int64_t src_row = r;
    if constexpr (IS_ANCHOR) {
        const int64_t end_pos = static_cast<int64_t>(QSL[r + 1]) - static_cast<int64_t>(NUM_REJ[r]);
        src_row = (end_pos - 1 > 0) ? (end_pos - 1) : 0;
    }
    const uint16_t* x_row = X + src_row * row_stride;

    if constexpr (SPT == 8) {
        const int c0 = tid * 8;
        const int c1 = 2048 + tid * 8;
        const int c2 = 4096 + tid * 8;
        const bool m2 = (c2 < HIDDEN_DS);
        const bool vec_ok = (((row_stride | tap_stride) & 7) == 0);

        float acc0[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        float acc1[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        float acc2[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

        #pragma unroll
        for (int tap = 0; tap < TAPS_DS; ++tap) {
            const float a = s_alpha[tap];
            const uint16_t* tp = x_row + static_cast<int64_t>(tap) * tap_stride;
            uint16_t b0[8], b1[8], b2[8];
            #pragma unroll
            for (int i = 0; i < 8; ++i) b2[i] = 0u;
            if (vec_ok) {
                u4_to_u16x8(*reinterpret_cast<const uint4*>(tp + c0), b0);
                u4_to_u16x8(*reinterpret_cast<const uint4*>(tp + c1), b1);
                if (m2) u4_to_u16x8(*reinterpret_cast<const uint4*>(tp + c2), b2);
            } else {
                #pragma unroll
                for (int i = 0; i < 8; ++i) { b0[i] = tp[c0 + i]; b1[i] = tp[c1 + i]; }
                if (m2) {
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) b2[i] = tp[c2 + i];
                }
            }
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                float x0 = bf16_to_f32(b0[i]);
                float x1 = bf16_to_f32(b1[i]);
                if constexpr (FMA_MODE == 0) {
                    acc0[i] = (tap == 0) ? __fadd_rn(0.0f, __fmul_rn(x0, a)) : fmaf(x0, a, acc0[i]);
                    acc1[i] = (tap == 0) ? __fadd_rn(0.0f, __fmul_rn(x1, a)) : fmaf(x1, a, acc1[i]);
                } else {
                    acc0[i] = __fadd_rn(acc0[i], __fmul_rn(x0, a));
                    acc1[i] = __fadd_rn(acc1[i], __fmul_rn(x1, a));
                }
            }
            if (m2) {
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    float x2 = bf16_to_f32(b2[i]);
                    if constexpr (FMA_MODE == 0) {
                        acc2[i] = (tap == 0) ? __fadd_rn(0.0f, __fmul_rn(x2, a)) : fmaf(x2, a, acc2[i]);
                    } else {
                        acc2[i] = __fadd_rn(acc2[i], __fmul_rn(x2, a));
                    }
                }
            }
        }

        float sq = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; ++i) sq = __fadd_rn(sq, __fmul_rn(acc0[i], acc0[i]));
        #pragma unroll
        for (int i = 0; i < 8; ++i) sq = __fadd_rn(sq, __fmul_rn(acc1[i], acc1[i]));
        if (m2) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) sq = __fadd_rn(sq, __fmul_rn(acc2[i], acc2[i]));
        }
        const float inv_rms = rsqrt_approx_f32(__fadd_rn(div_full_f32(cta256_sum_f32(sq, s_warp, tid), 4608.0f), eps));

        uint16_t wp0[8], wp1[8], wp2[8];
        u4_to_u16x8(*reinterpret_cast<const uint4*>(WEIGHT + c0), wp0);
        u4_to_u16x8(*reinterpret_cast<const uint4*>(WEIGHT + c1), wp1);
        u4_to_u16x8(m2 ? *reinterpret_cast<const uint4*>(WEIGHT + c2) : make_uint4(0u, 0u, 0u, 0u), wp2);

        uint16_t yp0[8], yp1[8], yp2[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) yp0[i] = f32_to_bf16(__fmul_rn(__fmul_rn(acc0[i], inv_rms), bf16_to_f32(wp0[i])));
        #pragma unroll
        for (int i = 0; i < 8; ++i) yp1[i] = f32_to_bf16(__fmul_rn(__fmul_rn(acc1[i], inv_rms), bf16_to_f32(wp1[i])));
        #pragma unroll
        for (int i = 0; i < 8; ++i) yp2[i] = m2 ? f32_to_bf16(__fmul_rn(__fmul_rn(acc2[i], inv_rms), bf16_to_f32(wp2[i]))) : 0u;
        const uint4 oy0 = u16x8_to_u4(yp0);
        const uint4 oy1 = u16x8_to_u4(yp1);
        const uint4 oy2 = u16x8_to_u4(yp2);
        uint16_t* out_row = OUT + r * out_row_stride;
        if ((out_row_stride & 7) == 0) {
            *reinterpret_cast<uint4*>(out_row + c0) = oy0;
            *reinterpret_cast<uint4*>(out_row + c1) = oy1;
            if (m2) *reinterpret_cast<uint4*>(out_row + c2) = oy2;
        } else {
            #pragma unroll
            for (int i = 0; i < 8; ++i) { out_row[c0 + i] = yp0[i]; out_row[c1 + i] = yp1[i]; }
            if (m2) {
                #pragma unroll
                for (int i = 0; i < 8; ++i) out_row[c2 + i] = yp2[i];
            }
        }
        if constexpr (PDL) {
            __syncthreads();
            if (tid == 0) gdc_launch_dependents();
        }
    } else {
        float mixed[18];
        #pragma unroll
        for (int g = 0; g < 18; ++g) mixed[g] = 0.0f;
        #pragma unroll
        for (int tap = 0; tap < TAPS_DS; ++tap) {
            const float a = s_alpha[tap];
            const uint16_t* tp = x_row + static_cast<int64_t>(tap) * tap_stride;
            #pragma unroll
            for (int g = 0; g < 18; ++g) {
                float xv = bf16_to_f32(tp[g * 256 + tid]);
                if constexpr (FMA_MODE == 0) {
                    mixed[g] = (tap == 0) ? __fadd_rn(0.0f, __fmul_rn(xv, a)) : fmaf(xv, a, mixed[g]);
                } else {
                    mixed[g] = __fadd_rn(mixed[g], __fmul_rn(xv, a));
                }
            }
        }
        float sq = 0.0f;
        #pragma unroll
        for (int g = 0; g < 18; ++g) sq = __fadd_rn(sq, __fmul_rn(mixed[g], mixed[g]));
        const float inv_rms = rsqrt_approx_f32(__fadd_rn(div_full_f32(cta256_sum_f32(sq, s_warp, tid), 4608.0f), eps));
        uint16_t* out_row = OUT + r * out_row_stride;
        #pragma unroll
        for (int g = 0; g < 18; ++g) {
            out_row[g * 256 + tid] = f32_to_bf16(__fmul_rn(__fmul_rn(mixed[g], inv_rms), bf16_to_f32(WEIGHT[g * 256 + tid])));
        }
        if constexpr (PDL) {
            __syncthreads();
            if (tid == 0) gdc_launch_dependents();
        }
    }
}

// A.3: `dspark_mix_rmsnorm_5layer_v10_kernel`
//   - Fuses ALL 5 DSpark drafter layers' `mix_rmsnorm(context_states, i, layer[i].dflare_norm)`
//     into ONE launch `grid = (3, rows)` (`grp = blockIdx.x` varies fastest -> all 3 groups `{L0,L1}`,
//     `{L2,L3}`, `{L4}` of the SAME `row` co-schedule simultaneously and hit 100% L2 on `X[row]`!).
//   - Line-1 `slot < 0` fast-exit skips dead padded rows on C=2..8 & C=32 (`94.37 MB -> 9.44 MB` on C=32!).
template <int SPT, int FMA_MODE, bool PDL>
__global__ __launch_bounds__(256, 2)
void dspark_mix_rmsnorm_5layer_v10_kernel(
    const uint16_t* __restrict__ X,
    const float*    __restrict__ ALPHA_5x8_F32,   // [>=5, 8] fp32 (or nullptr if bf16)
    const uint16_t* __restrict__ ALPHA_5x8_BF16,  // [>=5, 8] bf16
    int alpha_is_bf16,
    const uint16_t* __restrict__ W0,
    const uint16_t* __restrict__ W1,
    const uint16_t* __restrict__ W2,
    const uint16_t* __restrict__ W3,
    const uint16_t* __restrict__ W4,
    const int64_t*  __restrict__ SLOT_MAP,        // optional [rows] int64; if slot < 0 -> skip!
    int64_t slot_stride,
    uint16_t*       __restrict__ OUT_5L,          // [5, max_rows, 4608] bf16
    int64_t layer_out_stride,                     // max_rows * 4608
    int64_t row_stride,
    int64_t tap_stride,
    float eps
) {
    const int grp = static_cast<int>(blockIdx.x);  // 0 -> {0,1}, 1 -> {2,3}, 2 -> {4} (varies fastest for L2 reuse!)
    const int64_t row = static_cast<int64_t>(blockIdx.y);
    const int tid = static_cast<int>(threadIdx.x);
    const int l_a = grp * 2;
    const bool has_b = (l_a + 1 < 5);

    __shared__ float s_warp[8];
    __shared__ float s_alpha_a[TAPS_DS];
    __shared__ float s_alpha_b[TAPS_DS];

    if (tid < TAPS_DS) {
        const int off_a = l_a * TAPS_DS + tid;
        s_alpha_a[tid] = alpha_is_bf16 ? bf16_to_f32(ALPHA_5x8_BF16[off_a]) : ALPHA_5x8_F32[off_a];
        if (has_b) {
            const int off_b = (l_a + 1) * TAPS_DS + tid;
            s_alpha_b[tid] = alpha_is_bf16 ? bf16_to_f32(ALPHA_5x8_BF16[off_b]) : ALPHA_5x8_F32[off_b];
        }
    }
    __syncthreads();

    if constexpr (PDL) gdc_wait();

    if (SLOT_MAP != nullptr) {
        if (SLOT_MAP[row * slot_stride] < 0) {
            if constexpr (PDL) {
                __syncthreads();
                if (tid == 0) gdc_launch_dependents();
            }
            return;
        }
    }

    const uint16_t* WA = (grp == 0) ? W0 : ((grp == 1) ? W2 : W4);
    const uint16_t* WB = (grp == 0) ? W1 : W3;
    uint16_t* out_a = OUT_5L + static_cast<int64_t>(l_a) * layer_out_stride + row * HIDDEN_DS;
    uint16_t* out_b = OUT_5L + static_cast<int64_t>(l_a + 1) * layer_out_stride + row * HIDDEN_DS;
    const uint16_t* x_row = X + row * row_stride;

    if constexpr (SPT == 8) {
        const int c0 = tid * 8;
        const int c1 = 2048 + tid * 8;
        const int c2 = 4096 + tid * 8;
        const bool m2 = (c2 < HIDDEN_DS);

        float a0[8] = {0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};
        float a1[8] = {0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};
        float a2[8] = {0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};
        float b0[8] = {0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};
        float b1[8] = {0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};
        float b2[8] = {0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};

        #pragma unroll
        for (int tap = 0; tap < TAPS_DS; ++tap) {
            const float al_a = s_alpha_a[tap];
            const float al_b = has_b ? s_alpha_b[tap] : 0.0f;
            const uint16_t* tp = x_row + static_cast<int64_t>(tap) * tap_stride;
            const uint4 u0 = ld_ca_u4(tp + c0);
            const uint4 u1 = ld_ca_u4(tp + c1);
            const uint4 u2 = m2 ? ld_ca_u4(tp + c2) : make_uint4(0u, 0u, 0u, 0u);
            uint16_t p0[8], p1[8], p2[8];
            u4_to_u16x8(u0, p0);
            u4_to_u16x8(u1, p1);
            u4_to_u16x8(u2, p2);

            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float x0 = bf16_to_f32(p0[i]);
                const float x1 = bf16_to_f32(p1[i]);
                if constexpr (FMA_MODE == 0) {
                    a0[i] = (tap == 0) ? __fadd_rn(0.0f, __fmul_rn(x0, al_a)) : fmaf(x0, al_a, a0[i]);
                    a1[i] = (tap == 0) ? __fadd_rn(0.0f, __fmul_rn(x1, al_a)) : fmaf(x1, al_a, a1[i]);
                    if (has_b) {
                        b0[i] = (tap == 0) ? __fadd_rn(0.0f, __fmul_rn(x0, al_b)) : fmaf(x0, al_b, b0[i]);
                        b1[i] = (tap == 0) ? __fadd_rn(0.0f, __fmul_rn(x1, al_b)) : fmaf(x1, al_b, b1[i]);
                    }
                } else {
                    a0[i] = __fadd_rn(a0[i], __fmul_rn(x0, al_a));
                    a1[i] = __fadd_rn(a1[i], __fmul_rn(x1, al_a));
                    if (has_b) {
                        b0[i] = __fadd_rn(b0[i], __fmul_rn(x0, al_b));
                        b1[i] = __fadd_rn(b1[i], __fmul_rn(x1, al_b));
                    }
                }
            }
            if (m2) {
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const float x2 = bf16_to_f32(p2[i]);
                    if constexpr (FMA_MODE == 0) {
                        a2[i] = (tap == 0) ? __fadd_rn(0.0f, __fmul_rn(x2, al_a)) : fmaf(x2, al_a, a2[i]);
                        if (has_b) b2[i] = (tap == 0) ? __fadd_rn(0.0f, __fmul_rn(x2, al_b)) : fmaf(x2, al_b, b2[i]);
                    } else {
                        a2[i] = __fadd_rn(a2[i], __fmul_rn(x2, al_a));
                        if (has_b) b2[i] = __fadd_rn(b2[i], __fmul_rn(x2, al_b));
                    }
                }
            }
        }

        float sq_a = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; ++i) sq_a = __fadd_rn(sq_a, __fmul_rn(a0[i], a0[i]));
        #pragma unroll
        for (int i = 0; i < 8; ++i) sq_a = __fadd_rn(sq_a, __fmul_rn(a1[i], a1[i]));
        if (m2) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) sq_a = __fadd_rn(sq_a, __fmul_rn(a2[i], a2[i]));
        }
        const float inv_a = rsqrt_approx_f32(__fadd_rn(div_full_f32(cta256_sum_f32(sq_a, s_warp, tid), 4608.0f), eps));

        float inv_b = 0.0f;
        if (has_b) {
            float sq_b = 0.0f;
            #pragma unroll
            for (int i = 0; i < 8; ++i) sq_b = __fadd_rn(sq_b, __fmul_rn(b0[i], b0[i]));
            #pragma unroll
            for (int i = 0; i < 8; ++i) sq_b = __fadd_rn(sq_b, __fmul_rn(b1[i], b1[i]));
            if (m2) {
                #pragma unroll
                for (int i = 0; i < 8; ++i) sq_b = __fadd_rn(sq_b, __fmul_rn(b2[i], b2[i]));
            }
            inv_b = rsqrt_approx_f32(__fadd_rn(div_full_f32(cta256_sum_f32(sq_b, s_warp, tid), 4608.0f), eps));
        }

        {
            uint16_t wp0[8], wp1[8], wp2[8];
            u4_to_u16x8(*reinterpret_cast<const uint4*>(WA + c0), wp0);
            u4_to_u16x8(*reinterpret_cast<const uint4*>(WA + c1), wp1);
            u4_to_u16x8(m2 ? *reinterpret_cast<const uint4*>(WA + c2) : make_uint4(0u, 0u, 0u, 0u), wp2);
            uint16_t yp0[8], yp1[8], yp2[8];
            #pragma unroll
            for (int i = 0; i < 8; ++i) yp0[i] = f32_to_bf16(__fmul_rn(__fmul_rn(a0[i], inv_a), bf16_to_f32(wp0[i])));
            #pragma unroll
            for (int i = 0; i < 8; ++i) yp1[i] = f32_to_bf16(__fmul_rn(__fmul_rn(a1[i], inv_a), bf16_to_f32(wp1[i])));
            #pragma unroll
            for (int i = 0; i < 8; ++i) yp2[i] = m2 ? f32_to_bf16(__fmul_rn(__fmul_rn(a2[i], inv_a), bf16_to_f32(wp2[i]))) : 0u;
            *reinterpret_cast<uint4*>(out_a + c0) = u16x8_to_u4(yp0);
            *reinterpret_cast<uint4*>(out_a + c1) = u16x8_to_u4(yp1);
            if (m2) *reinterpret_cast<uint4*>(out_a + c2) = u16x8_to_u4(yp2);
        }
        if (has_b) {
            uint16_t wp0[8], wp1[8], wp2[8];
            u4_to_u16x8(*reinterpret_cast<const uint4*>(WB + c0), wp0);
            u4_to_u16x8(*reinterpret_cast<const uint4*>(WB + c1), wp1);
            u4_to_u16x8(m2 ? *reinterpret_cast<const uint4*>(WB + c2) : make_uint4(0u, 0u, 0u, 0u), wp2);
            uint16_t yp0[8], yp1[8], yp2[8];
            #pragma unroll
            for (int i = 0; i < 8; ++i) yp0[i] = f32_to_bf16(__fmul_rn(__fmul_rn(b0[i], inv_b), bf16_to_f32(wp0[i])));
            #pragma unroll
            for (int i = 0; i < 8; ++i) yp1[i] = f32_to_bf16(__fmul_rn(__fmul_rn(b1[i], inv_b), bf16_to_f32(wp1[i])));
            #pragma unroll
            for (int i = 0; i < 8; ++i) yp2[i] = m2 ? f32_to_bf16(__fmul_rn(__fmul_rn(b2[i], inv_b), bf16_to_f32(wp2[i]))) : 0u;
            *reinterpret_cast<uint4*>(out_b + c0) = u16x8_to_u4(yp0);
            *reinterpret_cast<uint4*>(out_b + c1) = u16x8_to_u4(yp1);
            if (m2) *reinterpret_cast<uint4*>(out_b + c2) = u16x8_to_u4(yp2);
        }
        if constexpr (PDL) {
            __syncthreads();
            if (tid == 0) gdc_launch_dependents();
        }
    } else {
        float ma[18];
        float mb[18];
        #pragma unroll
        for (int g = 0; g < 18; ++g) { ma[g] = 0.0f; mb[g] = 0.0f; }
        #pragma unroll
        for (int tap = 0; tap < TAPS_DS; ++tap) {
            const float al_a = s_alpha_a[tap];
            const float al_b = has_b ? s_alpha_b[tap] : 0.0f;
            const uint16_t* tp = x_row + static_cast<int64_t>(tap) * tap_stride;
            #pragma unroll
            for (int g = 0; g < 18; ++g) {
                const float xv = bf16_to_f32(tp[g * 256 + tid]);
                if constexpr (FMA_MODE == 0) {
                    ma[g] = (tap == 0) ? __fadd_rn(0.0f, __fmul_rn(xv, al_a)) : fmaf(xv, al_a, ma[g]);
                    if (has_b) mb[g] = (tap == 0) ? __fadd_rn(0.0f, __fmul_rn(xv, al_b)) : fmaf(xv, al_b, mb[g]);
                } else {
                    ma[g] = __fadd_rn(ma[g], __fmul_rn(xv, al_a));
                    if (has_b) mb[g] = __fadd_rn(mb[g], __fmul_rn(xv, al_b));
                }
            }
        }
        float sq_a = 0.0f;
        #pragma unroll
        for (int g = 0; g < 18; ++g) sq_a = __fadd_rn(sq_a, __fmul_rn(ma[g], ma[g]));
        const float inv_a = rsqrt_approx_f32(__fadd_rn(div_full_f32(cta256_sum_f32(sq_a, s_warp, tid), 4608.0f), eps));
        float inv_b = 0.0f;
        if (has_b) {
            float sq_b = 0.0f;
            #pragma unroll
            for (int g = 0; g < 18; ++g) sq_b = __fadd_rn(sq_b, __fmul_rn(mb[g], mb[g]));
            inv_b = rsqrt_approx_f32(__fadd_rn(div_full_f32(cta256_sum_f32(sq_b, s_warp, tid), 4608.0f), eps));
        }
        #pragma unroll
        for (int g = 0; g < 18; ++g) {
            out_a[g * 256 + tid] = f32_to_bf16(__fmul_rn(__fmul_rn(ma[g], inv_a), bf16_to_f32(WA[g * 256 + tid])));
            if (has_b) out_b[g * 256 + tid] = f32_to_bf16(__fmul_rn(__fmul_rn(mb[g], inv_b), bf16_to_f32(WB[g * 256 + tid])));
        }
        if constexpr (PDL) {
            __syncthreads();
            if (tid == 0) gdc_launch_dependents();
        }
    }
}

// A.4: `dspark_qk_norm_rope_write_kv_v10_kernel`
//   - 1 warp per head (`D == 128`, `HALF == 64`), 1 CTA per token (`grid = (num_tokens,)`):
//     eliminates 56-CTA Wave-2 tail on 48 SMs (C=1) and 7x redundant `cosf/sinf` across heads!
//   - Computes `cos/sin` ONCE per token in SMEM BEFORE `gdc_wait()`.
//   - On context KV (`nhq == 0 && write_kv`), `slot < 0` exits on line 1!
//   - Exact 4-warp Triton FP32 sum tree `(S0 + S2) + (S1 + S3)` (or `(S0 + S1) + (S2 + S3)`).
template <bool WRITE_KV, bool NORM, bool OTF_ROPE, int ROPE_FMA_MODE, bool PDL>
__global__ __launch_bounds__(256, 2)
void dspark_qk_norm_rope_multi_v10_kernel(
    uint16_t*       __restrict__ X,
    const uint16_t* __restrict__ Q_WEIGHT,
    const uint16_t* __restrict__ K_WEIGHT,
    const float*    __restrict__ COS_SIN,
    const int64_t*  __restrict__ POS_I64,
    const int32_t*  __restrict__ POS_I32,
    int pos_is_i32,
    uint16_t*       __restrict__ KC,
    uint16_t*       __restrict__ VC,
    const int64_t*  __restrict__ SLOT_MAP,
    int64_t row_stride,
    int64_t q_size,
    int64_t sk_b, int64_t sk_s, int64_t sk_h,
    int64_t sv_b, int64_t sv_s, int64_t sv_h,
    int64_t slot_stride,
    int64_t kv_block_size,
    int nhq,
    int nhk,
    float eps,
    int sqrt_mode
) {
    const int64_t t = static_cast<int64_t>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    const int wid = tid >> 5;
    const int lane = tid & 31;
    const int total_heads = nhq + nhk;

    __shared__ float s_cos[64];
    __shared__ float s_sin[64];
    __shared__ int64_t s_slot;

    // Preload `slot`, `pos`, and compute `cos/sin` ONCE per token BEFORE `gdc_wait()`!
    if (WRITE_KV && nhq == 0) {
        const int64_t sl = SLOT_MAP[t * slot_stride];
        if (sl < 0) {
            if constexpr (PDL) {
                gdc_wait();
                __syncthreads();
                if (tid == 0) gdc_launch_dependents();
            }
            return;
        }
        if (tid == 0) s_slot = sl;
    } else if (WRITE_KV && tid == 0) {
        s_slot = SLOT_MAP[t * slot_stride];
    }

    if (tid < 64) {
        const int64_t pos = pos_is_i32 ? static_cast<int64_t>(POS_I32[t]) : POS_I64[t];
        if constexpr (OTF_ROPE) {
            const float angle = __fmul_rn(static_cast<float>(pos), COS_SIN[tid]);
            s_cos[tid] = cosf(angle);
            s_sin[tid] = sinf(angle);
        } else {
            s_cos[tid] = COS_SIN[pos * 128 + tid];
            s_sin[tid] = COS_SIN[pos * 128 + 64 + tid];
        }
    }
    __syncthreads();

    if constexpr (PDL) gdc_wait();

    // V10 #206 PDL Fix: ALL warps converge at CTA epilogue __syncthreads() before gdc_launch_dependents()
    // so Q warps (0..nhq-1) never release drafter_attn while K/V warps (nhq..total_heads-1) are still writing KC/VC!
    if (wid < total_heads) {
        const float c_lo = s_cos[lane];
        const float s_lo = s_sin[lane];
        const float c_hi = s_cos[32 + lane];
        const float s_hi = s_sin[32 + lane];

        const bool is_q = (wid < nhq);
        uint16_t* head_ptr = is_q
            ? (X + t * row_stride + static_cast<int64_t>(wid) * 128)
            : (X + t * row_stride + q_size + static_cast<int64_t>(wid - nhq) * 128);

        float x0 = bf16_to_f32(head_ptr[lane]);
        float x1 = bf16_to_f32(head_ptr[32 + lane]);
        float x2 = bf16_to_f32(head_ptr[64 + lane]);
        float x3 = bf16_to_f32(head_ptr[96 + lane]);

        if constexpr (NORM) {
            const float s0 = warp_sum_f32(__fmul_rn(x0, x0));
            const float s1 = warp_sum_f32(__fmul_rn(x1, x1));
            const float s2 = warp_sum_f32(__fmul_rn(x2, x2));
            const float s3 = warp_sum_f32(__fmul_rn(x3, x3));
            // Matches Triton 4-warp shfl_xor(2) then shfl_xor(1): (S0 + S2) + (S1 + S3)
            const float sum_sq = __fadd_rn(__fadd_rn(s0, s2), __fadd_rn(s1, s3));
            const float ms = div_full_f32(sum_sq, 128.0f);
            const float inv = inv_sqrt_qk_f32(__fadd_rn(ms, eps), sqrt_mode);

            const uint16_t* wp = is_q ? Q_WEIGHT : K_WEIGHT;
            x0 = round_bf16_cvt_f32(__fmul_rn(__fmul_rn(x0, inv), bf16_to_f32(wp[lane])));
            x1 = round_bf16_cvt_f32(__fmul_rn(__fmul_rn(x1, inv), bf16_to_f32(wp[32 + lane])));
            x2 = round_bf16_cvt_f32(__fmul_rn(__fmul_rn(x2, inv), bf16_to_f32(wp[64 + lane])));
            x3 = round_bf16_cvt_f32(__fmul_rn(__fmul_rn(x3, inv), bf16_to_f32(wp[96 + lane])));
        }

        float o0, o1, o2, o3;
        if constexpr (ROPE_FMA_MODE == 0) {
            o0 = fmaf(x0, c_lo, -__fmul_rn(x2, s_lo));
            o2 = fmaf(x2, c_lo,  __fmul_rn(x0, s_lo));
            o1 = fmaf(x1, c_hi, -__fmul_rn(x3, s_hi));
            o3 = fmaf(x3, c_hi,  __fmul_rn(x1, s_hi));
        } else if constexpr (ROPE_FMA_MODE == 1) {
            o0 = fmaf(-x2, s_lo, __fmul_rn(x0, c_lo));
            o2 = fmaf( x0, s_lo, __fmul_rn(x2, c_lo));
            o1 = fmaf(-x3, s_hi, __fmul_rn(x1, c_hi));
            o3 = fmaf( x1, s_hi, __fmul_rn(x3, c_hi));
        } else {
            o0 = __fadd_rn(__fmul_rn(x0, c_lo), -__fmul_rn(x2, s_lo));
            o2 = __fadd_rn(__fmul_rn(x2, c_lo),  __fmul_rn(x0, s_lo));
            o1 = __fadd_rn(__fmul_rn(x1, c_hi), -__fmul_rn(x3, s_hi));
            o3 = __fadd_rn(__fmul_rn(x3, c_hi),  __fmul_rn(x1, s_hi));
        }

        if (!WRITE_KV || is_q) {
            head_ptr[lane]      = f32_to_bf16(o0);
            head_ptr[32 + lane] = f32_to_bf16(o1);
            head_ptr[64 + lane] = f32_to_bf16(o2);
            head_ptr[96 + lane] = f32_to_bf16(o3);
        } else {
            const int64_t slot = s_slot;
            const int64_t kv_h = static_cast<int64_t>(wid - nhq);
            if (slot >= 0) {
                const int64_t blk = slot / kv_block_size;
                const int64_t off = slot % kv_block_size;
                uint16_t* kc_row = KC + blk * sk_b + off * sk_s + kv_h * sk_h;
                uint16_t* vc_row = VC + blk * sv_b + off * sv_s + kv_h * sv_h;
                const uint16_t* v_src = head_ptr + static_cast<int64_t>(nhk) * 128;
                const uint16_t v0 = v_src[lane];
                const uint16_t v1 = v_src[32 + lane];
                const uint16_t v2 = v_src[64 + lane];
                const uint16_t v3 = v_src[96 + lane];
                kc_row[lane]      = f32_to_bf16(o0);
                kc_row[32 + lane] = f32_to_bf16(o1);
                kc_row[64 + lane] = f32_to_bf16(o2);
                kc_row[96 + lane] = f32_to_bf16(o3);
                vc_row[lane]      = v0;
                vc_row[32 + lane] = v1;
                vc_row[64 + lane] = v2;
                vc_row[96 + lane] = v3;
            }
        }
    }
    if constexpr (PDL) {
        __syncthreads();
        if (tid == 0) gdc_launch_dependents();
    }
}

// Ultra-lean 1-warp (32-thread) specialization for `store_context_kv` (`nhq == 0 && nhk == 1`)!
template <bool OTF_ROPE, int ROPE_FMA_MODE, bool PDL>
__global__ __launch_bounds__(32, 8)
void dspark_ctx_kv_1warp_v10_kernel(
    const uint16_t* __restrict__ KV,
    const uint16_t* __restrict__ K_WEIGHT,
    const float*    __restrict__ COS_SIN,
    const int64_t*  __restrict__ POS_I64,
    const int32_t*  __restrict__ POS_I32,
    int pos_is_i32,
    uint16_t*       __restrict__ KC,
    uint16_t*       __restrict__ VC,
    const int64_t*  __restrict__ SLOT_MAP,
    int64_t row_stride,
    int64_t sk_b, int64_t sk_s,
    int64_t sv_b, int64_t sv_s,
    int64_t slot_stride,
    int64_t kv_block_size,
    float eps,
    int sqrt_mode
) {
    const int64_t t = static_cast<int64_t>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    const int64_t slot = SLOT_MAP[t * slot_stride];
    if (slot < 0) {
        if constexpr (PDL) {
            gdc_wait();
            __syncwarp();
            if (lane == 0) gdc_launch_dependents();
        }
        return;
    }

    const int64_t pos = pos_is_i32 ? static_cast<int64_t>(POS_I32[t]) : POS_I64[t];
    float c_lo, s_lo, c_hi, s_hi;
    if constexpr (OTF_ROPE) {
        const float a_lo = __fmul_rn(static_cast<float>(pos), COS_SIN[lane]);
        const float a_hi = __fmul_rn(static_cast<float>(pos), COS_SIN[32 + lane]);
        c_lo = cosf(a_lo); s_lo = sinf(a_lo);
        c_hi = cosf(a_hi); s_hi = sinf(a_hi);
    } else {
        c_lo = COS_SIN[pos * 128 + lane];
        s_lo = COS_SIN[pos * 128 + 64 + lane];
        c_hi = COS_SIN[pos * 128 + 32 + lane];
        s_hi = COS_SIN[pos * 128 + 96 + lane];
    }
    const float w0 = bf16_to_f32(K_WEIGHT[lane]);
    const float w1 = bf16_to_f32(K_WEIGHT[32 + lane]);
    const float w2 = bf16_to_f32(K_WEIGHT[64 + lane]);
    const float w3 = bf16_to_f32(K_WEIGHT[96 + lane]);

    if constexpr (PDL) gdc_wait();

    const uint16_t* row_k = KV + t * row_stride;
    const uint16_t* row_v = row_k + 128;
    float x0 = bf16_to_f32(row_k[lane]);
    float x1 = bf16_to_f32(row_k[32 + lane]);
    float x2 = bf16_to_f32(row_k[64 + lane]);
    float x3 = bf16_to_f32(row_k[96 + lane]);

    const float sum_sq = __fadd_rn(
        __fadd_rn(warp_sum_f32(__fmul_rn(x0, x0)), warp_sum_f32(__fmul_rn(x2, x2))),
        __fadd_rn(warp_sum_f32(__fmul_rn(x1, x1)), warp_sum_f32(__fmul_rn(x3, x3)))
    );
    const float inv = inv_sqrt_qk_f32(__fadd_rn(div_full_f32(sum_sq, 128.0f), eps), sqrt_mode);
    x0 = round_bf16_cvt_f32(__fmul_rn(__fmul_rn(x0, inv), w0));
    x1 = round_bf16_cvt_f32(__fmul_rn(__fmul_rn(x1, inv), w1));
    x2 = round_bf16_cvt_f32(__fmul_rn(__fmul_rn(x2, inv), w2));
    x3 = round_bf16_cvt_f32(__fmul_rn(__fmul_rn(x3, inv), w3));

    float o0, o1, o2, o3;
    if constexpr (ROPE_FMA_MODE == 0) {
        o0 = fmaf(x0, c_lo, -__fmul_rn(x2, s_lo));
        o2 = fmaf(x2, c_lo,  __fmul_rn(x0, s_lo));
        o1 = fmaf(x1, c_hi, -__fmul_rn(x3, s_hi));
        o3 = fmaf(x3, c_hi,  __fmul_rn(x1, s_hi));
    } else if constexpr (ROPE_FMA_MODE == 1) {
        o0 = fmaf(-x2, s_lo, __fmul_rn(x0, c_lo));
        o2 = fmaf( x0, s_lo, __fmul_rn(x2, c_lo));
        o1 = fmaf(-x3, s_hi, __fmul_rn(x1, c_hi));
        o3 = fmaf( x1, s_hi, __fmul_rn(x3, c_hi));
    } else {
        o0 = __fadd_rn(__fmul_rn(x0, c_lo), -__fmul_rn(x2, s_lo));
        o2 = __fadd_rn(__fmul_rn(x2, c_lo),  __fmul_rn(x0, s_lo));
        o1 = __fadd_rn(__fmul_rn(x1, c_hi), -__fmul_rn(x3, s_hi));
        o3 = __fadd_rn(__fmul_rn(x3, c_hi),  __fmul_rn(x1, s_hi));
    }

    const uint16_t v0 = row_v[lane];
    const uint16_t v1 = row_v[32 + lane];
    const uint16_t v2 = row_v[64 + lane];
    const uint16_t v3 = row_v[96 + lane];

    const int64_t blk = slot / kv_block_size;
    const int64_t off = slot % kv_block_size;
    uint16_t* kc_row = KC + blk * sk_b + off * sk_s;
    uint16_t* vc_row = VC + blk * sv_b + off * sv_s;
    kc_row[lane]      = f32_to_bf16(o0);
    kc_row[32 + lane] = f32_to_bf16(o1);
    kc_row[64 + lane] = f32_to_bf16(o2);
    kc_row[96 + lane] = f32_to_bf16(o3);
    vc_row[lane]      = v0;
    vc_row[32 + lane] = v1;
    vc_row[64 + lane] = v2;
    vc_row[96 + lane] = v3;
    if constexpr (PDL) {
        __syncwarp();
        if (lane == 0) gdc_launch_dependents();
    }
}

// A.5: `dspark_silu_mul_v10_kernel`, `dspark_quant_fp8_rowwise_v10_kernel`,
//      `dspark_silu_mul_quant_fp8_v10_kernel`, and `dspark_add_softcap_v10_kernel`
template <bool PDL>
__global__ __launch_bounds__(256, 2)
void dspark_silu_mul_v10_kernel(
    const uint16_t* __restrict__ X,
    uint16_t*       __restrict__ Y,
    int64_t stride_x,
    int N
) {
    const int64_t row = static_cast<int64_t>(blockIdx.x);
    const int col = static_cast<int>(blockIdx.y) * 1024 + static_cast<int>(threadIdx.x) * 4;
    if constexpr (PDL) gdc_wait();
    if (col < N) {
        const uint16_t* xr = X + row * stride_x;
        uint16_t* yr = Y + row * static_cast<int64_t>(N);
        if (col + 4 <= N && ((stride_x & 3) == 0)) {
            uint16_t gp[4], up[4], op[4];
            u2_to_u16x4(*reinterpret_cast<const uint2*>(xr + col), gp);
            u2_to_u16x4(*reinterpret_cast<const uint2*>(xr + N + col), up);
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float g = bf16_to_f32(gp[i]);
                const float u = bf16_to_f32(up[i]);
                const float silu = div_rn_f32(g, __fadd_rn(1.0f, expf(-g)));
                op[i] = f32_to_bf16(__fmul_rn(silu, u));
            }
            *reinterpret_cast<uint2*>(yr + col) = u16x4_to_u2(op);
        } else {
            for (int i = 0; i < 4 && (col + i) < N; ++i) {
                const float g = bf16_to_f32(xr[col + i]);
                const float u = bf16_to_f32(xr[N + col + i]);
                const float silu = div_rn_f32(g, __fadd_rn(1.0f, expf(-g)));
                yr[col + i] = f32_to_bf16(__fmul_rn(silu, u));
            }
        }
    }
    if constexpr (PDL) {
        __syncthreads();
        if (threadIdx.x == 0) gdc_launch_dependents();
    }
}

// Single-pass PDL-enabled BF16 -> FP8 E4M3FN rowwise quantizer (`K = 4608, 2304, 768` or arbitrary `K % 16 == 0`)
template <bool PDL>
__global__ __launch_bounds__(256, 2)
void dspark_quant_fp8_rowwise_v10_kernel(
    const uint16_t* __restrict__ X,
    uint8_t*        __restrict__ Y,
    float*          __restrict__ S,
    int M,
    int K,
    int64_t stride_xm,
    int64_t stride_ym
) {
    const int row = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    uint8_t* y_row = Y + static_cast<int64_t>(row) * stride_ym;

    if (row >= M) {
        if constexpr (PDL) gdc_wait();
        const uint4 z4 = make_uint4(0u, 0u, 0u, 0u);
        for (int c = tid * 16; c < K; c += 256 * 16) {
            *reinterpret_cast<uint4*>(y_row + c) = z4;
        }
        if (tid == 0) S[row] = 1.0f;
        if constexpr (PDL) {
            __syncthreads();
            if (tid == 0) gdc_launch_dependents();
        }
        return;
    }

    __shared__ float s_max[8];
    const int warp = tid >> 5;
    const int lane = tid & 31;

    if constexpr (PDL) gdc_wait();

    const uint16_t* x_row = X + static_cast<int64_t>(row) * stride_xm;

    // Fast in-register single-pass for K <= 4608 (`<= 3` uint4s = `<= 24` bf16s/thread, zero DRAM re-read!)
    if (K <= 4608 && ((stride_xm & 7) == 0) && ((K & 15) == 0)) {
        const int c0 = tid * 8;
        const int c1 = 2048 + tid * 8;
        const int c2 = 4096 + tid * 8;
        uint16_t p0[8], p1[8], p2[8];
        u4_to_u16x8((c0 < K) ? *reinterpret_cast<const uint4*>(x_row + c0) : make_uint4(0u,0u,0u,0u), p0);
        u4_to_u16x8((c1 < K) ? *reinterpret_cast<const uint4*>(x_row + c1) : make_uint4(0u,0u,0u,0u), p1);
        u4_to_u16x8((c2 < K) ? *reinterpret_cast<const uint4*>(x_row + c2) : make_uint4(0u,0u,0u,0u), p2);

        float amax = 0.0f;
        if (c0 < K) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) amax = fmaxf(amax, fabsf(bf16_to_f32(p0[i])));
        }
        if (c1 < K) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) amax = fmaxf(amax, fabsf(bf16_to_f32(p1[i])));
        }
        if (c2 < K) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) amax = fmaxf(amax, fabsf(bf16_to_f32(p2[i])));
        }
        #pragma unroll
        for (int off = 16; off >= 1; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, off));
        if (lane == 0) s_max[warp] = amax;
        __syncthreads();
        if (warp == 0) {
            float m = (lane < 8) ? s_max[lane] : 0.0f;
            m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFu, m, 4));
            m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFu, m, 2));
            m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFu, m, 1));
            if (lane == 0) {
                const float sc = fmaxf(__fmul_rn(m, 1.0f / 448.0f), 1e-12f);
                S[row] = sc;
                s_max[0] = div_full_f32(1.0f, sc);
            }
        }
        __syncthreads();
        const float inv_sc = s_max[0];
        if (c0 < K) {
            uint16_t qp[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float v0 = fminf(fmaxf(__fmul_rn(bf16_to_f32(p0[2 * i]), inv_sc), -448.0f), 448.0f);
                const float v1 = fminf(fmaxf(__fmul_rn(bf16_to_f32(p0[2 * i + 1]), inv_sc), -448.0f), 448.0f);
                qp[i] = pack_e4m3x2_f32(v0, v1);
            }
            *reinterpret_cast<uint2*>(y_row + c0) = u16x4_to_u2(qp);
        }
        if (c1 < K) {
            uint16_t qp[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float v0 = fminf(fmaxf(__fmul_rn(bf16_to_f32(p1[2 * i]), inv_sc), -448.0f), 448.0f);
                const float v1 = fminf(fmaxf(__fmul_rn(bf16_to_f32(p1[2 * i + 1]), inv_sc), -448.0f), 448.0f);
                qp[i] = pack_e4m3x2_f32(v0, v1);
            }
            *reinterpret_cast<uint2*>(y_row + c1) = u16x4_to_u2(qp);
        }
        if (c2 < K) {
            uint16_t qp[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float v0 = fminf(fmaxf(__fmul_rn(bf16_to_f32(p2[2 * i]), inv_sc), -448.0f), 448.0f);
                const float v1 = fminf(fmaxf(__fmul_rn(bf16_to_f32(p2[2 * i + 1]), inv_sc), -448.0f), 448.0f);
                qp[i] = pack_e4m3x2_f32(v0, v1);
            }
            *reinterpret_cast<uint2*>(y_row + c2) = u16x4_to_u2(qp);
        }
        if constexpr (PDL) {
            __syncthreads();
            if (tid == 0) gdc_launch_dependents();
        }
        return;
    }

    float amax = 0.0f;
    for (int c = tid * 2; c < K; c += 512) {
        amax = fmaxf(amax, fabsf(bf16_to_f32(x_row[c])));
        amax = fmaxf(amax, fabsf(bf16_to_f32(x_row[c + 1])));
    }
    #pragma unroll
    for (int off = 16; off >= 1; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, off));
    if (lane == 0) s_max[warp] = amax;
    __syncthreads();
    if (warp == 0) {
        float m = (lane < 8) ? s_max[lane] : 0.0f;
        m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFu, m, 4));
        m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFu, m, 2));
        m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFu, m, 1));
        if (lane == 0) {
            const float sc = fmaxf(__fmul_rn(m, 1.0f / 448.0f), 1e-12f);
            S[row] = sc;
            s_max[0] = div_full_f32(1.0f, sc);
        }
    }
    __syncthreads();
    const float inv_sc = s_max[0];
    for (int c = tid * 2; c < K; c += 512) {
        const float v0 = fminf(fmaxf(__fmul_rn(bf16_to_f32(x_row[c]), inv_sc), -448.0f), 448.0f);
        const float v1 = fminf(fmaxf(__fmul_rn(bf16_to_f32(x_row[c + 1]), inv_sc), -448.0f), 448.0f);
        *reinterpret_cast<uint16_t*>(y_row + c) = pack_e4m3x2_f32(v0, v1);
    }
    if constexpr (PDL) {
        __syncthreads();
        if (tid == 0) gdc_launch_dependents();
    }
}

// Fused C=32 `silu_and_mul(X [M, 2*N])` + `quantize_fp8_rowwise(-> Y_fp8 [M_pad, N], S [M_pad])`!
// Applies exact intermediate `round_bf16` before `amax` and `* inv_scale` so output is LITERAL 0-ULP
// bit-identical to unfused `silu_and_mul` + `quantize_fp8_rowwise` while saving 5 launches & 11.8 MB/step!
template <bool PDL>
__global__ __launch_bounds__(256, 2)
void dspark_silu_mul_quant_fp8_v10_kernel(
    const uint16_t* __restrict__ X,
    uint8_t*        __restrict__ Y_FP8,
    float*          __restrict__ S,
    int M,
    int N,
    int64_t stride_xm,
    int64_t stride_ym
) {
    const int row = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    uint8_t* y_row = Y_FP8 + static_cast<int64_t>(row) * stride_ym;

    if (row >= M) {
        if constexpr (PDL) gdc_wait();
        const uint4 z4 = make_uint4(0u, 0u, 0u, 0u);
        for (int c = tid * 16; c < N; c += 256 * 16) {
            *reinterpret_cast<uint4*>(y_row + c) = z4;
        }
        if (tid == 0) S[row] = 1.0f;
        if constexpr (PDL) {
            __syncthreads();
            if (tid == 0) gdc_launch_dependents();
        }
        return;
    }

    __shared__ float s_max[8];
    const int warp = tid >> 5;
    const int lane = tid & 31;

    if constexpr (PDL) gdc_wait();

    const uint16_t* x_row = X + static_cast<int64_t>(row) * stride_xm;

    // N == 2304 (`gate_up` -> `down` on DSpark TP=8): `c0 = tid*8` (`0..2047`) + `c1 = 2048 + tid*8` (`2048..2303`, `tid < 32`)!
    if (N == 2304 && ((stride_xm & 7) == 0)) {
        const int c0 = tid * 8;
        const int c1 = 2048 + tid * 8;
        const bool m1 = (c1 < 2304);

        uint16_t gp0[8], up0[8], gp1[8], up1[8];
        u4_to_u16x8(*reinterpret_cast<const uint4*>(x_row + c0), gp0);
        u4_to_u16x8(*reinterpret_cast<const uint4*>(x_row + 2304 + c0), up0);
        u4_to_u16x8(m1 ? *reinterpret_cast<const uint4*>(x_row + c1) : make_uint4(0u,0u,0u,0u), gp1);
        u4_to_u16x8(m1 ? *reinterpret_cast<const uint4*>(x_row + 2304 + c1) : make_uint4(0u,0u,0u,0u), up1);

        float v0[8];
        float v1[8];
        float amax = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float gf = bf16_to_f32(gp0[i]);
            const float uf = bf16_to_f32(up0[i]);
            const float y_bf16 = round_bf16_cvt_f32(__fmul_rn(div_rn_f32(gf, __fadd_rn(1.0f, expf(-gf))), uf));
            v0[i] = y_bf16;
            amax = fmaxf(amax, fabsf(y_bf16));
        }
        if (m1) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float gf = bf16_to_f32(gp1[i]);
                const float uf = bf16_to_f32(up1[i]);
                const float y_bf16 = round_bf16_cvt_f32(__fmul_rn(div_rn_f32(gf, __fadd_rn(1.0f, expf(-gf))), uf));
                v1[i] = y_bf16;
                amax = fmaxf(amax, fabsf(y_bf16));
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 8; ++i) v1[i] = 0.0f;
        }

        #pragma unroll
        for (int off = 16; off >= 1; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, off));
        if (lane == 0) s_max[warp] = amax;
        __syncthreads();
        if (warp == 0) {
            float m = (lane < 8) ? s_max[lane] : 0.0f;
            m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFu, m, 4));
            m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFu, m, 2));
            m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFu, m, 1));
            if (lane == 0) {
                const float sc = fmaxf(__fmul_rn(m, 1.0f / 448.0f), 1e-12f);
                S[row] = sc;
                s_max[0] = div_full_f32(1.0f, sc);
            }
        }
        __syncthreads();
        const float inv_sc = s_max[0];
        uint16_t qp0[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float f0 = fminf(fmaxf(__fmul_rn(v0[2 * i], inv_sc), -448.0f), 448.0f);
            const float f1 = fminf(fmaxf(__fmul_rn(v0[2 * i + 1], inv_sc), -448.0f), 448.0f);
            qp0[i] = pack_e4m3x2_f32(f0, f1);
        }
        *reinterpret_cast<uint2*>(y_row + c0) = u16x4_to_u2(qp0);
        if (m1) {
            uint16_t qp1[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float f0 = fminf(fmaxf(__fmul_rn(v1[2 * i], inv_sc), -448.0f), 448.0f);
                const float f1 = fminf(fmaxf(__fmul_rn(v1[2 * i + 1], inv_sc), -448.0f), 448.0f);
                qp1[i] = pack_e4m3x2_f32(f0, f1);
            }
            *reinterpret_cast<uint2*>(y_row + c1) = u16x4_to_u2(qp1);
        }
        if constexpr (PDL) {
            __syncthreads();
            if (tid == 0) gdc_launch_dependents();
        }
        return;
    }

    float amax = 0.0f;
    for (int c = tid * 2; c < N; c += 512) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            const float gf = bf16_to_f32(x_row[c + j]);
            const float uf = bf16_to_f32(x_row[N + c + j]);
            const float y_bf16 = round_bf16_cvt_f32(__fmul_rn(div_rn_f32(gf, __fadd_rn(1.0f, expf(-gf))), uf));
            amax = fmaxf(amax, fabsf(y_bf16));
        }
    }
    #pragma unroll
    for (int off = 16; off >= 1; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, off));
    if (lane == 0) s_max[warp] = amax;
    __syncthreads();
    if (warp == 0) {
        float m = (lane < 8) ? s_max[lane] : 0.0f;
        m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFu, m, 4));
        m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFu, m, 2));
        m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFu, m, 1));
        if (lane == 0) {
            const float sc = fmaxf(__fmul_rn(m, 1.0f / 448.0f), 1e-12f);
            S[row] = sc;
            s_max[0] = div_full_f32(1.0f, sc);
        }
    }
    __syncthreads();
    const float inv_sc = s_max[0];
    for (int c = tid * 2; c < N; c += 512) {
        float vals[2];
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            const float gf = bf16_to_f32(x_row[c + j]);
            const float uf = bf16_to_f32(x_row[N + c + j]);
            const float y_bf16 = round_bf16_cvt_f32(__fmul_rn(div_rn_f32(gf, __fadd_rn(1.0f, expf(-gf))), uf));
            vals[j] = fminf(fmaxf(__fmul_rn(y_bf16, inv_sc), -448.0f), 448.0f);
        }
        *reinterpret_cast<uint16_t*>(y_row + c) = pack_e4m3x2_f32(vals[0], vals[1]);
    }
    if constexpr (PDL) {
        __syncthreads();
        if (tid == 0) gdc_launch_dependents();
    }
}

template <bool PDL>
__global__ __launch_bounds__(256, 2)
void dspark_add_softcap_v10_kernel(
    const uint16_t* __restrict__ A,
    const uint16_t* __restrict__ B,
    uint16_t*       __restrict__ OUT,
    int64_t stride_a,
    int64_t stride_b,
    float softcap,
    int N
) {
    const int col0 = static_cast<int>(blockIdx.x) * 1024 + static_cast<int>(threadIdx.x) * 4;
    const int64_t row = static_cast<int64_t>(blockIdx.y);
    if constexpr (PDL) gdc_wait();
    if (col0 < N) {
        const uint16_t* ar = A + row * stride_a;
        const uint16_t* br = B + row * stride_b;
        uint16_t* orow = OUT + row * static_cast<int64_t>(N);
        for (int i = 0; i < 4 && (col0 + i) < N; ++i) {
            const float af = bf16_to_f32(ar[col0 + i]);
            const float bf = bf16_to_f32(br[col0 + i]);
            orow[col0 + i] = f32_to_bf16(dspark_softcap_f32(af, bf, softcap));
        }
    }
    if constexpr (PDL) {
        __syncthreads();
        if (threadIdx.x == 0) gdc_launch_dependents();
    }
}

// ============================================================================
// SECTION B: K=7 Markov Recurrence, Fold-Argmax & Cross-Step Fused Kernels
// ============================================================================

// Exact recurrence gate math matching `dspark_ops._markov_gate_kernel` / `_markov_rnn_step_*`
static __device__ __forceinline__ uint16_t eval_markov_new_state_bf16(
    float raw_g, float raw_c, float st_f32, bool init_zero
) {
    float gate = round_bf16_bits_f32(raw_g);
    float cand = round_bf16_bits_f32(raw_c);
    gate = round_bf16_bits_f32(div_rn_f32(1.0f, __fadd_rn(1.0f, expf(-gate))));
    cand = round_bf16_bits_f32(tanhf(cand));
    const float one_minus_g = round_bf16_bits_f32(__fsub_rn(1.0f, gate));
    const float update = round_bf16_bits_f32(__fmul_rn(one_minus_g, cand));
    float new_st;
    if (init_zero) {
        new_st = update;
    } else {
        const float keep = round_bf16_bits_f32(__fmul_rn(gate, st_f32));
        new_st = __fadd_rn(keep, update);
    }
    return f32_to_bf16(new_st);
}

static __device__ __forceinline__ uint16_t eval_markov_latent_bf16(float raw_o) {
    const float outv = round_bf16_bits_f32(raw_o);
    return f32_to_bf16(tanhf(outv));
}

// B.1: `dspark_markov_gate_v10_kernel` (`RANK == 256`)
template <bool PDL>
__global__ __launch_bounds__(256, 2)
void dspark_markov_gate_v10_kernel(
    const float*    __restrict__ RAW,
    const uint16_t* __restrict__ STATE,
    uint16_t*       __restrict__ OUT_STATE,
    uint16_t*       __restrict__ LATENT
) {
    const int64_t row = static_cast<int64_t>(blockIdx.x);
    const int c = static_cast<int>(threadIdx.x);
    if constexpr (PDL) gdc_wait();
    const float* raw = RAW + row * (3 * RANK_MK);
    const float rg = raw[c];
    const float rc = raw[RANK_MK + c];
    const float ro = raw[2 * RANK_MK + c];
    const float st = bf16_to_f32(STATE[row * RANK_MK + c]);
    OUT_STATE[row * RANK_MK + c] = eval_markov_new_state_bf16(rg, rc, st, false);
    LATENT[row * RANK_MK + c]    = eval_markov_latent_bf16(ro);
    if constexpr (PDL) {
        __syncthreads();
        if (c == 0) gdc_launch_dependents();
    }
}

// Helper: 1-warp 256-dim BF16*FP32 dot product (`lane = 0..31` owns `l*8 + 0..7`)
static __device__ __forceinline__ float dot256_bf16_f32_warp(
    const uint16_t* __restrict__ w_row,
    const float (&vec8)[8],
    int lane
) {
    uint16_t wp[8];
    u4_to_u16x8(*reinterpret_cast<const uint4*>(w_row + lane * 8), wp);
    float s0 = __fadd_rn(__fmul_rn(bf16_to_f32(wp[0]), vec8[0]), __fmul_rn(bf16_to_f32(wp[1]), vec8[1]));
    float s1 = __fadd_rn(__fmul_rn(bf16_to_f32(wp[2]), vec8[2]), __fmul_rn(bf16_to_f32(wp[3]), vec8[3]));
    float s2 = __fadd_rn(__fmul_rn(bf16_to_f32(wp[4]), vec8[4]), __fmul_rn(bf16_to_f32(wp[5]), vec8[5]));
    float s3 = __fadd_rn(__fmul_rn(bf16_to_f32(wp[6]), vec8[6]), __fmul_rn(bf16_to_f32(wp[7]), vec8[7]));
    float sum = __fadd_rn(__fadd_rn(s0, s1), __fadd_rn(s2, s3));
    return warp_sum_f32(sum);
}

// B.2: `dspark_markov_rnn_step_r1_48cta_v10_kernel` (`rows == 1`, 48 CTAs = 100% GB10 SMs!)
//   - `cb = 0..31`: computes `BC = 8` dims of `(gate, cand)` -> `OUT_STATE[cb*8 + 0..7]` (16 W rows/CTA)
//   - `cb = 32..47`: computes `BC = 16` dims of `out` -> `LATENT[(cb-32)*16 + 0..15]` (16 W rows/CTA)
template <bool INIT_ZERO, bool HAS_SKIP, bool PDL>
__global__ __launch_bounds__(128, 4)
void dspark_markov_rnn_step_r1_48cta_v10_kernel(
    const uint16_t* __restrict__ STATE,
    const int64_t*  __restrict__ PREV_I64,
    const int32_t*  __restrict__ PREV_I32,
    int prev_is_i32,
    const float*    __restrict__ CTX,
    const uint16_t* __restrict__ W1,
    const uint16_t* __restrict__ RNN_W,   // [768, 512] bf16
    uint16_t*       __restrict__ OUT_STATE,
    uint16_t*       __restrict__ LATENT,
    const int32_t*  __restrict__ SKIP_PTR
) {
    const int cb = static_cast<int>(blockIdx.y);  // 0..47
    const int tid = static_cast<int>(threadIdx.x);
    const int wid = tid >> 5;   // 0..3
    const int lane = tid & 31;  // 0..31
    constexpr int STRIDE_W = 2 * RANK_MK;  // 512

    if constexpr (PDL) gdc_wait();
    if constexpr (HAS_SKIP) {
        if (*SKIP_PTR != 0) {
            if constexpr (PDL) {
                __syncthreads();
                if (tid == 0) gdc_launch_dependents();
            }
            return;
        }
    }

    int64_t tok = prev_is_i32 ? static_cast<int64_t>(PREV_I32[0]) : PREV_I64[0];
    if (tok < 0) tok = 0;

    float emb8[8];
    {
        const uint4 ev = *reinterpret_cast<const uint4*>(W1 + tok * RANK_MK + lane * 8);
        uint16_t ep[8];
        u4_to_u16x8(ev, ep);
        #pragma unroll
        for (int i = 0; i < 8; ++i) emb8[i] = bf16_to_f32(ep[i]);
    }

    float st8[8] = {0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};
    if constexpr (!INIT_ZERO) {
        const uint4 sv = *reinterpret_cast<const uint4*>(STATE + lane * 8);
        uint16_t sp[8];
        u4_to_u16x8(sv, sp);
        #pragma unroll
        for (int i = 0; i < 8; ++i) st8[i] = bf16_to_f32(sp[i]);
    }

    if (cb < 32) {
        // 4 warps compute 8 dims (`2` dims per warp) of `gate` + `cand`
        const int c_base = cb * 8 + wid * 2;
        uint16_t out_st[2];
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int c = c_base + j;
            const uint16_t* wg_row = RNN_W + static_cast<int64_t>(c) * STRIDE_W;
            const uint16_t* wc_row = RNN_W + static_cast<int64_t>(RANK_MK + c) * STRIDE_W;
            const float eg = dot256_bf16_f32_warp(wg_row + RANK_MK, emb8, lane);
            const float ec = dot256_bf16_f32_warp(wc_row + RANK_MK, emb8, lane);
            float rg, rc;
            if constexpr (INIT_ZERO) {
                rg = __fadd_rn(CTX[c], eg);
                rc = __fadd_rn(CTX[RANK_MK + c], ec);
                out_st[j] = (lane == 0) ? eval_markov_new_state_bf16(rg, rc, 0.0f, true) : 0u;
            } else {
                const float sg = dot256_bf16_f32_warp(wg_row, st8, lane);
                const float sc = dot256_bf16_f32_warp(wc_row, st8, lane);
                rg = __fadd_rn(__fadd_rn(CTX[c], sg), eg);
                rc = __fadd_rn(__fadd_rn(CTX[RANK_MK + c], sc), ec);
                const float st_c = (lane == 0) ? bf16_to_f32(STATE[c]) : 0.0f;
                out_st[j] = (lane == 0) ? eval_markov_new_state_bf16(rg, rc, st_c, false) : 0u;
            }
        }
        if (lane == 0) {
            OUT_STATE[c_base + 0] = out_st[0];
            OUT_STATE[c_base + 1] = out_st[1];
        }
        if constexpr (PDL) {
            __syncthreads();
            if (tid == 0) gdc_launch_dependents();
        }
    } else {
        // `cb = 32..47`: 4 warps compute 16 dims (`4` dims per warp) of `out -> LATENT`
        const int c_base = (cb - 32) * 16 + wid * 4;
        uint16_t out_lat[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int c = c_base + j;
            const uint16_t* wo_row = RNN_W + static_cast<int64_t>(2 * RANK_MK + c) * STRIDE_W;
            const float eo = dot256_bf16_f32_warp(wo_row + RANK_MK, emb8, lane);
            float ro;
            if constexpr (INIT_ZERO) {
                ro = __fadd_rn(CTX[2 * RANK_MK + c], eo);
            } else {
                const float so = dot256_bf16_f32_warp(wo_row, st8, lane);
                ro = __fadd_rn(__fadd_rn(CTX[2 * RANK_MK + c], so), eo);
            }
            out_lat[j] = (lane == 0) ? eval_markov_latent_bf16(ro) : 0u;
        }
        if (lane == 0) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) LATENT[c_base + j] = out_lat[j];
        }
        if constexpr (PDL) {
            __syncthreads();
            if (tid == 0) gdc_launch_dependents();
        }
    }
}

// B.3: `dspark_markov_rnn_step_tc_v10_kernel` (`rows >= 2`, 48-CTA Full-Machine `TC48`, exact `BK=64` order!)
//   - Grid: `(cdiv(rows, 16), 48)`, Block: `32` (1 warp/CTA, zero `__syncthreads()` overhead!):
//     * `B = 2..16` (incl. ALL C=2..8!): `48` CTAs = `100%` of GB10's 48 SMs (6x occupancy vs V6's 8 CTAs!).
//     * `B = 17..32` (C=32): `96` CTAs = `2.0 x 48` exact full waves (3x occupancy vs V6's 16 CTAs!).
//   - `cb = 0..31`: `8` cols of `(gate, cand)` (`16` W rows/CTA) -> `OUT_STATE`.
//   - `cb = 32..47`: `16` cols of `out` (`16` W rows/CTA) -> `LATENT`.
//   - Preserves exact `k0 = 0, 64, 128, 192: { 4x emb MMA; 4x st MMA }` FP32 accumulation order -> 0-ULP!
template <bool INIT_ZERO, bool HAS_SKIP, bool PDL>
__global__ __launch_bounds__(32, 8)
void dspark_markov_rnn_step_tc_v10_kernel(
    const uint16_t* __restrict__ STATE,
    const int64_t*  __restrict__ PREV_I64,
    const int32_t*  __restrict__ PREV_I32,
    int prev_is_i32,
    const float*    __restrict__ CTX,
    int64_t ctx_row_stride,               // normally `3 * RANK_MK` (768), or `0` for broadcast!
    const uint16_t* __restrict__ W1,
    const uint16_t* __restrict__ RNN_W,   // [768, 512] bf16
    uint16_t*       __restrict__ OUT_STATE,
    uint16_t*       __restrict__ LATENT,
    const int32_t*  __restrict__ SKIP_PTR,
    int rows,
    int64_t prev_stride
) {
    const int pm = static_cast<int>(blockIdx.x);
    const int cb = static_cast<int>(blockIdx.y);  // 0..47
    const int lane = static_cast<int>(threadIdx.x);
    const int grp = lane >> 2;    // 0..7
    const int lt = lane & 3;      // 0..3
    const int m_base = pm * 16;
    const bool has_hi8 = (m_base + 8 < rows);
    const int load_u4s = has_hi8 ? 128 : 64;
    constexpr int STRIDE_W = 2 * RANK_MK;

    alignas(16) __shared__ uint16_t sEmb[16][72];
    alignas(16) __shared__ uint16_t sSt[16][72];
    __shared__ int64_t sTok[16];

    if constexpr (PDL) gdc_wait();
    if constexpr (HAS_SKIP) {
        if (SKIP_PTR != nullptr && (*SKIP_PTR != 0)) {
            if constexpr (PDL) {
                __syncwarp();
                if (lane == 0) gdc_launch_dependents();
            }
            return;
        }
    } else {
        (void)SKIP_PTR;
    }

    if (lane < 16) {
        const int r = m_base + lane;
        int64_t tk = 0;
        if (r < rows) {
            tk = prev_is_i32 ? static_cast<int64_t>(PREV_I32[static_cast<int64_t>(r) * prev_stride])
                             : PREV_I64[static_cast<int64_t>(r) * prev_stride];
            if (tk < 0) tk = 0;
        }
        sTok[lane] = tk;
    }
    __syncwarp();

    if (cb < 32) {
        // `cb = 0..31`: `8` cols of `(gate, cand)` (`16` rows of `RNN_W`)
        const int c_col = cb * 8 + grp;
        const uint16_t* wg_row = RNN_W + static_cast<int64_t>(c_col) * STRIDE_W;
        const uint16_t* wc_row = RNN_W + static_cast<int64_t>(RANK_MK + c_col) * STRIDE_W;
        float g0 = 0.f, g1 = 0.f, g2 = 0.f, g3 = 0.f;
        float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;

        for (int k0 = 0; k0 < RANK_MK; k0 += 64) {
            for (int idx = lane; idx < load_u4s; idx += 32) {
                const int r_loc = idx >> 3;
                const int k_sub = (idx & 7) * 8;
                const int r_glob = m_base + r_loc;
                if (r_glob < rows) {
                    *reinterpret_cast<uint4*>(&sEmb[r_loc][k_sub]) =
                        ld_ca_u4(W1 + sTok[r_loc] * RANK_MK + k0 + k_sub);
                    if constexpr (!INIT_ZERO) {
                        *reinterpret_cast<uint4*>(&sSt[r_loc][k_sub]) =
                            ld_ca_u4(STATE + static_cast<int64_t>(r_glob) * RANK_MK + k0 + k_sub);
                    }
                } else {
                    *reinterpret_cast<uint4*>(&sEmb[r_loc][k_sub]) = make_uint4(0u, 0u, 0u, 0u);
                    if constexpr (!INIT_ZERO) {
                        *reinterpret_cast<uint4*>(&sSt[r_loc][k_sub]) = make_uint4(0u, 0u, 0u, 0u);
                    }
                }
            }
            __syncwarp();

            #pragma unroll
            for (int ks = 0; ks < 4; ++ks) {
                const int ko = ks * 16;
                const uint32_t a0 = *reinterpret_cast<const uint32_t*>(&sEmb[grp][ko + 2 * lt]);
                const uint32_t a1 = has_hi8 ? *reinterpret_cast<const uint32_t*>(&sEmb[grp + 8][ko + 2 * lt]) : 0u;
                const uint32_t a2 = *reinterpret_cast<const uint32_t*>(&sEmb[grp][ko + 8 + 2 * lt]);
                const uint32_t a3 = has_hi8 ? *reinterpret_cast<const uint32_t*>(&sEmb[grp + 8][ko + 8 + 2 * lt]) : 0u;
                const int wk = RANK_MK + k0 + ko;
                mma_m16n8k16_bf16(g0, g1, g2, g3, a0, a1, a2, a3,
                    *reinterpret_cast<const uint32_t*>(wg_row + wk + 2 * lt),
                    *reinterpret_cast<const uint32_t*>(wg_row + wk + 8 + 2 * lt));
                mma_m16n8k16_bf16(c0, c1, c2, c3, a0, a1, a2, a3,
                    *reinterpret_cast<const uint32_t*>(wc_row + wk + 2 * lt),
                    *reinterpret_cast<const uint32_t*>(wc_row + wk + 8 + 2 * lt));
            }
            if constexpr (!INIT_ZERO) {
                #pragma unroll
                for (int ks = 0; ks < 4; ++ks) {
                    const int ko = ks * 16;
                    const uint32_t a0 = *reinterpret_cast<const uint32_t*>(&sSt[grp][ko + 2 * lt]);
                    const uint32_t a1 = has_hi8 ? *reinterpret_cast<const uint32_t*>(&sSt[grp + 8][ko + 2 * lt]) : 0u;
                    const uint32_t a2 = *reinterpret_cast<const uint32_t*>(&sSt[grp][ko + 8 + 2 * lt]);
                    const uint32_t a3 = has_hi8 ? *reinterpret_cast<const uint32_t*>(&sSt[grp + 8][ko + 8 + 2 * lt]) : 0u;
                    const int wk = k0 + ko;
                    mma_m16n8k16_bf16(g0, g1, g2, g3, a0, a1, a2, a3,
                        *reinterpret_cast<const uint32_t*>(wg_row + wk + 2 * lt),
                        *reinterpret_cast<const uint32_t*>(wg_row + wk + 8 + 2 * lt));
                    mma_m16n8k16_bf16(c0, c1, c2, c3, a0, a1, a2, a3,
                        *reinterpret_cast<const uint32_t*>(wc_row + wk + 2 * lt),
                        *reinterpret_cast<const uint32_t*>(wc_row + wk + 8 + 2 * lt));
                }
            }
            __syncwarp();
        }

        const int n0 = cb * 8 + 2 * lt;
        const int n1 = n0 + 1;
        const int r0 = m_base + grp;
        const int r1 = r0 + 8;
        if (r0 < rows) {
            const float* cr = CTX + static_cast<int64_t>(r0) * ctx_row_stride;
            const float st_a = INIT_ZERO ? 0.0f : bf16_to_f32(STATE[static_cast<int64_t>(r0) * RANK_MK + n0]);
            const float st_b = INIT_ZERO ? 0.0f : bf16_to_f32(STATE[static_cast<int64_t>(r0) * RANK_MK + n1]);
            const uint16_t ns0 = eval_markov_new_state_bf16(__fadd_rn(cr[n0], g0), __fadd_rn(cr[RANK_MK + n0], c0), st_a, INIT_ZERO);
            const uint16_t ns1 = eval_markov_new_state_bf16(__fadd_rn(cr[n1], g1), __fadd_rn(cr[RANK_MK + n1], c1), st_b, INIT_ZERO);
            *reinterpret_cast<uint32_t*>(OUT_STATE + static_cast<int64_t>(r0) * RANK_MK + n0) = pack_u16x2(ns0, ns1);
        }
        if (r1 < rows) {
            const float* cr = CTX + static_cast<int64_t>(r1) * ctx_row_stride;
            const float st_a = INIT_ZERO ? 0.0f : bf16_to_f32(STATE[static_cast<int64_t>(r1) * RANK_MK + n0]);
            const float st_b = INIT_ZERO ? 0.0f : bf16_to_f32(STATE[static_cast<int64_t>(r1) * RANK_MK + n1]);
            const uint16_t ns0 = eval_markov_new_state_bf16(__fadd_rn(cr[n0], g2), __fadd_rn(cr[RANK_MK + n0], c2), st_a, INIT_ZERO);
            const uint16_t ns1 = eval_markov_new_state_bf16(__fadd_rn(cr[n1], g3), __fadd_rn(cr[RANK_MK + n1], c3), st_b, INIT_ZERO);
            *reinterpret_cast<uint32_t*>(OUT_STATE + static_cast<int64_t>(r1) * RANK_MK + n0) = pack_u16x2(ns0, ns1);
        }
        if constexpr (PDL) {
            __syncwarp();
            if (lane == 0) gdc_launch_dependents();
        }
    } else {
        // `cb = 32..47`: `16` cols (`2` N=8 tiles) of `out -> LATENT` (`16` rows of `RNN_W`)
        const int c_base = (cb - 32) * 16;
        const uint16_t* wo0_row = RNN_W + static_cast<int64_t>(2 * RANK_MK + c_base + grp) * STRIDE_W;
        const uint16_t* wo1_row = RNN_W + static_cast<int64_t>(2 * RANK_MK + c_base + 8 + grp) * STRIDE_W;
        float o00 = 0.f, o01 = 0.f, o02 = 0.f, o03 = 0.f;
        float o10 = 0.f, o11 = 0.f, o12 = 0.f, o13 = 0.f;

        for (int k0 = 0; k0 < RANK_MK; k0 += 64) {
            for (int idx = lane; idx < load_u4s; idx += 32) {
                const int r_loc = idx >> 3;
                const int k_sub = (idx & 7) * 8;
                const int r_glob = m_base + r_loc;
                if (r_glob < rows) {
                    *reinterpret_cast<uint4*>(&sEmb[r_loc][k_sub]) =
                        ld_ca_u4(W1 + sTok[r_loc] * RANK_MK + k0 + k_sub);
                    if constexpr (!INIT_ZERO) {
                        *reinterpret_cast<uint4*>(&sSt[r_loc][k_sub]) =
                            ld_ca_u4(STATE + static_cast<int64_t>(r_glob) * RANK_MK + k0 + k_sub);
                    }
                } else {
                    *reinterpret_cast<uint4*>(&sEmb[r_loc][k_sub]) = make_uint4(0u, 0u, 0u, 0u);
                    if constexpr (!INIT_ZERO) {
                        *reinterpret_cast<uint4*>(&sSt[r_loc][k_sub]) = make_uint4(0u, 0u, 0u, 0u);
                    }
                }
            }
            __syncwarp();

            #pragma unroll
            for (int ks = 0; ks < 4; ++ks) {
                const int ko = ks * 16;
                const uint32_t a0 = *reinterpret_cast<const uint32_t*>(&sEmb[grp][ko + 2 * lt]);
                const uint32_t a1 = has_hi8 ? *reinterpret_cast<const uint32_t*>(&sEmb[grp + 8][ko + 2 * lt]) : 0u;
                const uint32_t a2 = *reinterpret_cast<const uint32_t*>(&sEmb[grp][ko + 8 + 2 * lt]);
                const uint32_t a3 = has_hi8 ? *reinterpret_cast<const uint32_t*>(&sEmb[grp + 8][ko + 8 + 2 * lt]) : 0u;
                const int wk = RANK_MK + k0 + ko;
                mma_m16n8k16_bf16(o00, o01, o02, o03, a0, a1, a2, a3,
                    *reinterpret_cast<const uint32_t*>(wo0_row + wk + 2 * lt),
                    *reinterpret_cast<const uint32_t*>(wo0_row + wk + 8 + 2 * lt));
                mma_m16n8k16_bf16(o10, o11, o12, o13, a0, a1, a2, a3,
                    *reinterpret_cast<const uint32_t*>(wo1_row + wk + 2 * lt),
                    *reinterpret_cast<const uint32_t*>(wo1_row + wk + 8 + 2 * lt));
            }
            if constexpr (!INIT_ZERO) {
                #pragma unroll
                for (int ks = 0; ks < 4; ++ks) {
                    const int ko = ks * 16;
                    const uint32_t a0 = *reinterpret_cast<const uint32_t*>(&sSt[grp][ko + 2 * lt]);
                    const uint32_t a1 = has_hi8 ? *reinterpret_cast<const uint32_t*>(&sSt[grp + 8][ko + 2 * lt]) : 0u;
                    const uint32_t a2 = *reinterpret_cast<const uint32_t*>(&sSt[grp][ko + 8 + 2 * lt]);
                    const uint32_t a3 = has_hi8 ? *reinterpret_cast<const uint32_t*>(&sSt[grp + 8][ko + 8 + 2 * lt]) : 0u;
                    const int wk = k0 + ko;
                    mma_m16n8k16_bf16(o00, o01, o02, o03, a0, a1, a2, a3,
                        *reinterpret_cast<const uint32_t*>(wo0_row + wk + 2 * lt),
                        *reinterpret_cast<const uint32_t*>(wo0_row + wk + 8 + 2 * lt));
                    mma_m16n8k16_bf16(o10, o11, o12, o13, a0, a1, a2, a3,
                        *reinterpret_cast<const uint32_t*>(wo1_row + wk + 2 * lt),
                        *reinterpret_cast<const uint32_t*>(wo1_row + wk + 8 + 2 * lt));
                }
            }
            __syncwarp();
        }

        const int n0 = c_base + 2 * lt;
        const int n1 = n0 + 1;
        const int n2 = n0 + 8;
        const int n3 = n2 + 1;
        const int r0 = m_base + grp;
        const int r1 = r0 + 8;
        if (r0 < rows) {
            const float* cr = CTX + static_cast<int64_t>(r0) * ctx_row_stride;
            *reinterpret_cast<uint32_t*>(LATENT + static_cast<int64_t>(r0) * RANK_MK + n0) =
                pack_u16x2(eval_markov_latent_bf16(__fadd_rn(cr[2 * RANK_MK + n0], o00)),
                           eval_markov_latent_bf16(__fadd_rn(cr[2 * RANK_MK + n1], o01)));
            *reinterpret_cast<uint32_t*>(LATENT + static_cast<int64_t>(r0) * RANK_MK + n2) =
                pack_u16x2(eval_markov_latent_bf16(__fadd_rn(cr[2 * RANK_MK + n2], o10)),
                           eval_markov_latent_bf16(__fadd_rn(cr[2 * RANK_MK + n3], o11)));
        }
        if (r1 < rows) {
            const float* cr = CTX + static_cast<int64_t>(r1) * ctx_row_stride;
            *reinterpret_cast<uint32_t*>(LATENT + static_cast<int64_t>(r1) * RANK_MK + n0) =
                pack_u16x2(eval_markov_latent_bf16(__fadd_rn(cr[2 * RANK_MK + n0], o02)),
                           eval_markov_latent_bf16(__fadd_rn(cr[2 * RANK_MK + n1], o03)));
            *reinterpret_cast<uint32_t*>(LATENT + static_cast<int64_t>(r1) * RANK_MK + n2) =
                pack_u16x2(eval_markov_latent_bf16(__fadd_rn(cr[2 * RANK_MK + n2], o12)),
                           eval_markov_latent_bf16(__fadd_rn(cr[2 * RANK_MK + n3], o13)));
        }
        if constexpr (PDL) {
            __syncwarp();
            if (lane == 0) gdc_launch_dependents();
        }
    }
}

// Helper: 1-warp exact reduction of gathered `[ws * rows, S, 4]` records for a single row (`NS <= 32` or arbitrary `NS`)
static __device__ __forceinline__ void select_row_1warp(
    const float* __restrict__ G,
    int row, int rows, int S, int NS,
    bool need_conf, int exp_mode,
    int64_t& out_tok, float& out_conf
) {
    const int lane = threadIdx.x & 31;
    float m_loc = -INFINITY;
    int32_t r_loc = 2147483647;
    for (int r = lane; r < NS; r += 32) {
        const int rk = r / S;
        const int sl = r % S;
        const int64_t base = (static_cast<int64_t>(rk * rows + row) * S + sl) * 4;
        const float v = G[base];
        merge_max_idx(m_loc, r_loc, v, r);
    }
    warp_reduce_max_idx(m_loc, r_loc);
    int64_t tok = 0;
    if (r_loc < NS) {
        const int rk = r_loc / S;
        const int sl = r_loc % S;
        const int64_t base = (static_cast<int64_t>(rk * rows + row) * S + sl) * 4;
        tok = static_cast<int64_t>(G[base + 1]);
    }
    out_tok = tok;
    if (!need_conf) {
        out_conf = 0.0f;
        return;
    }
    const float m0 = (m_loc == -INFINITY) ? 0.0f : m_loc;
    float se_acc = 0.0f;
    for (int r = lane; r < NS; r += 32) {
        const int rk = r / S;
        const int sl = r % S;
        const int64_t base = (static_cast<int64_t>(rk * rows + row) * S + sl) * 4;
        const float v = G[base];
        const float se = G[base + 2];
        const float w = (v == -INFINITY) ? 0.0f : __fmul_rn(se, tl_exp_f32(__fsub_rn(v, m0), exp_mode));
        se_acc = __fadd_rn(se_acc, w);
    }
    const float tot = warp_sum_f32(se_acc);
    out_conf = (tot > 0.0f) ? div_full_f32(1.0f, tot) : 0.0f;
}

// Helper: Exact Dinkelbach / Conf early-stop step update (`es_mode`: 0=none, 1=dink, 2=es_no_hist, 3=es_hist)
static __device__ __forceinline__ void eval_es_step_scalar(
    int es_mode, int step_idx, int K, float p,
    int32_t greedy, int32_t slot,
    float surv_in, float tok_in, float best_in,
    float db, float lam, float prior_logit,
    const float* __restrict__ CAL_PTR,
    const float* __restrict__ STEP_PTR,
    const float* __restrict__ EMA_PTR,
    int exp_mode,
    int32_t& skip_out, float& surv_out, float& tok_out, float& best_out
) {
    skip_out = 0;
    surv_out = surv_in;
    tok_out = tok_in;
    best_out = best_in;
    if (es_mode == 0 || step_idx >= K - 1 || greedy == 0) {
        return;
    }
    const float pc = fminf(fmaxf(p, 1e-6f), 1.0f - 1e-6f);
    const float a = CAL_PTR[step_idx];
    const float b = CAL_PTR[8 + step_idx];
    const float logit_p = __fsub_rn(tl_log_f32(pc, exp_mode), tl_log_f32(__fsub_rn(1.0f, pc), exp_mode));

    if (es_mode == 1) {
        // Coupled Dinkelbach early-stop (`_argmax_select_es_dink_kernel`)
        const float mk_ms = CAL_PTR[20];
        const float z = __fadd_rn(__fadd_rn(__fmul_rn(a, logit_p), b), db);
        const float c = div_full_f32(1.0f, __fadd_rn(1.0f, tl_exp_f32(-z, exp_mode)));
        const float surv = __fmul_rn(surv_in, c);
        const float tokens = __fadd_rn(tok_in, surv);
        const float u_cur = __fsub_rn(tokens, __fmul_rn(lam, STEP_PTR[step_idx + 1]));
        const float best_u = fmaxf(best_in, u_cur);
        surv_out = surv;
        tok_out = tokens;
        best_out = best_u;
        const float thresh = __fadd_rn(best_u, __fmul_rn(lam, mk_ms));
        int can_beat = 0;
        for (int ww = step_idx + 2; ww <= K; ++ww) {
            const float u_ub = __fsub_rn(
                __fadd_rn(tokens, __fmul_rn(static_cast<float>(ww - step_idx - 1), surv)),
                __fmul_rn(lam, STEP_PTR[ww])
            );
            if (u_ub > thresh) can_beat = 1;
        }
        if (can_beat == 0) skip_out = 1;
    } else {
        // Classic `conf` / `conf+ema` early-stop (`_argmax_select_es_kernel`)
        const float beta = CAL_PTR[16];
        const float tau = CAL_PTR[17];
        float z = __fadd_rn(__fmul_rn(a, logit_p), b);
        if (es_mode == 3 && EMA_PTR != nullptr && slot >= 0) {
            const float h = EMA_PTR[static_cast<int64_t>(slot) * K + step_idx];
            const float hc = fminf(fmaxf(h, 1e-4f), 1.0f - 1e-4f);
            const float lh = __fsub_rn(__fsub_rn(tl_log_f32(hc, exp_mode), tl_log_f32(__fsub_rn(1.0f, hc), exp_mode)), prior_logit);
            z = __fadd_rn(z, __fmul_rn(beta, lh));
        }
        const float c = div_full_f32(1.0f, __fadd_rn(1.0f, tl_exp_f32(-z, exp_mode)));
        const float surv = __fmul_rn(surv_in, c);
        surv_out = surv;
        if (tau > 0.0f) {
            if (surv < tau) skip_out = 1;
        } else {
            const float tokens = __fadd_rn(tok_in, surv);
            tok_out = tokens;
            const float v_cur = div_full_f32(tokens, STEP_PTR[step_idx + 1]);
            const float best_v = fmaxf(best_in, v_cur);
            best_out = best_v;
            int can_beat = 0;
            for (int ww = step_idx + 2; ww <= K; ++ww) {
                const float v_ub = div_full_f32(
                    __fadd_rn(tokens, __fmul_rn(static_cast<float>(ww - step_idx - 1), surv)),
                    STEP_PTR[ww]
                );
                if (v_ub > best_v) can_beat = 1;
            }
            if (can_beat == 0) skip_out = 1;
        }
    }
}

// B.4: Cross-Step RDMA-Overlapped `dspark_select_and_rnn_step_r1_v10_kernel` (`rows == 1`, 48 CTAs!)
//   - Fuses `argmax_select[_es/_es_dink](step i)` + `markov_rnn_step(step i+1)`!
//   - BEFORE `gdc_wait()` (while `RDMA_AG(i)` is in flight over RoCEv2!), all 48 CTAs preload
//     `STATE[i]` + `RNN_W` and finish computing `CTX[i+1] + STATE[i] @ W_state.T`!
//   - Uses race-free ping-pong `(SKIP_IN, SURV_IN, TOK_IN, BEST_IN)` -> `(SKIP_OUT, SURV_OUT, TOK_OUT, BEST_OUT)`
//     so CTA 0 writes `*_OUT` while CTAs 0..47 read immutable `*_IN`!
template <bool PDL>
__global__ __launch_bounds__(128, 4)
void dspark_select_and_rnn_step_r1_v10_kernel(
    const float*    __restrict__ G,
    int S,
    int NS,
    int64_t*        __restrict__ OUT_TOK_I,
    float*          __restrict__ CONF_I,       // optional (nullptr if !with_conf)
    int with_conf,
    int es_mode,                               // 0=none, 1=dink, 2=es, 3=es+hist
    int step_idx,
    int K,
    const int32_t*  __restrict__ SKIP_IN,      // nullptr when step_idx == 0 or es_mode == 0
    const float*    __restrict__ SURV_IN,
    const float*    __restrict__ TOK_IN,
    const float*    __restrict__ BEST_IN,
    int32_t*        __restrict__ SKIP_OUT,
    float*          __restrict__ SURV_OUT,
    float*          __restrict__ TOK_OUT,
    float*          __restrict__ BEST_OUT,
    const int32_t*  __restrict__ GREEDY_PTR,
    const int32_t*  __restrict__ SLOT_PTR,
    const float*    __restrict__ DB_DEV_PTR,
    const float*    __restrict__ LAM_DEV_PTR,
    const float*    __restrict__ EMA_PTR,
    const float*    __restrict__ CAL_PTR,
    const float*    __restrict__ STEP_PTR,
    float prior_logit,
    // Step `i + 1` RNN inputs & outputs (`STATE_I` was produced BEFORE `fold_argmax(i)`!)
    const uint16_t* __restrict__ STATE_I,
    const float*    __restrict__ CTX_NEXT,
    const uint16_t* __restrict__ W1,
    const uint16_t* __restrict__ RNN_W,
    uint16_t*       __restrict__ OUT_STATE_NEXT,
    uint16_t*       __restrict__ LATENT_NEXT,
    int exp_mode
) {
    const int cb = static_cast<int>(blockIdx.y);  // 0..47
    const int tid = static_cast<int>(threadIdx.x);
    const int wid = tid >> 5;
    const int lane = tid & 31;
    constexpr int STRIDE_W = 2 * RANK_MK;

    __shared__ int64_t s_tok;
    __shared__ int32_t s_skip_next;

    // `SKIP_IN` was written in step `i - 1` (already completed before `fold_argmax(i)`!):
    // safe to check BEFORE `gdc_wait()`!
    const bool already_skipped = (es_mode != 0 && step_idx > 0 && SKIP_IN != nullptr && (*SKIP_IN != 0));
    if (already_skipped) {
        if constexpr (PDL) gdc_wait();
        if (cb == 0 && tid == 0) {
            *OUT_TOK_I = -1;
            if (CONF_I != nullptr) *CONF_I = 0.0f;
            if (SKIP_OUT != nullptr) {
                *SKIP_OUT = 1;
                *SURV_OUT = *SURV_IN;
                *TOK_OUT  = *TOK_IN;
                *BEST_OUT = *BEST_IN;
            }
        }
        if constexpr (PDL) {
            __syncthreads();
            if (tid == 0) gdc_launch_dependents();
        }
        return;
    }

    // ---- PRE-`gdc_wait()` OVERLAP: compute `CTX_NEXT + STATE_I @ W_state.T` while `RDMA_AG(i)` runs! ----
    float st8[8];
    {
        const uint4 sv = *reinterpret_cast<const uint4*>(STATE_I + lane * 8);
        uint16_t sp[8];
        u4_to_u16x8(sv, sp);
        #pragma unroll
        for (int i = 0; i < 8; ++i) st8[i] = bf16_to_f32(sp[i]);
    }

    float base_g0 = 0.f, base_g1 = 0.f, base_c0 = 0.f, base_c1 = 0.f;
    float st_c0 = 0.f, st_c1 = 0.f;
    float base_o0 = 0.f, base_o1 = 0.f, base_o2 = 0.f, base_o3 = 0.f;

    if (cb < 32) {
        const int c_base = cb * 8 + wid * 2;
        const uint16_t* wg0 = RNN_W + static_cast<int64_t>(c_base + 0) * STRIDE_W;
        const uint16_t* wg1 = RNN_W + static_cast<int64_t>(c_base + 1) * STRIDE_W;
        const uint16_t* wc0 = RNN_W + static_cast<int64_t>(RANK_MK + c_base + 0) * STRIDE_W;
        const uint16_t* wc1 = RNN_W + static_cast<int64_t>(RANK_MK + c_base + 1) * STRIDE_W;
        base_g0 = __fadd_rn(CTX_NEXT[c_base + 0], dot256_bf16_f32_warp(wg0, st8, lane));
        base_g1 = __fadd_rn(CTX_NEXT[c_base + 1], dot256_bf16_f32_warp(wg1, st8, lane));
        base_c0 = __fadd_rn(CTX_NEXT[RANK_MK + c_base + 0], dot256_bf16_f32_warp(wc0, st8, lane));
        base_c1 = __fadd_rn(CTX_NEXT[RANK_MK + c_base + 1], dot256_bf16_f32_warp(wc1, st8, lane));
        if (lane == 0) {
            st_c0 = bf16_to_f32(STATE_I[c_base + 0]);
            st_c1 = bf16_to_f32(STATE_I[c_base + 1]);
        }
    } else {
        const int c_base = (cb - 32) * 16 + wid * 4;
        base_o0 = __fadd_rn(CTX_NEXT[2 * RANK_MK + c_base + 0], dot256_bf16_f32_warp(RNN_W + static_cast<int64_t>(2 * RANK_MK + c_base + 0) * STRIDE_W, st8, lane));
        base_o1 = __fadd_rn(CTX_NEXT[2 * RANK_MK + c_base + 1], dot256_bf16_f32_warp(RNN_W + static_cast<int64_t>(2 * RANK_MK + c_base + 1) * STRIDE_W, st8, lane));
        base_o2 = __fadd_rn(CTX_NEXT[2 * RANK_MK + c_base + 2], dot256_bf16_f32_warp(RNN_W + static_cast<int64_t>(2 * RANK_MK + c_base + 2) * STRIDE_W, st8, lane));
        base_o3 = __fadd_rn(CTX_NEXT[2 * RANK_MK + c_base + 3], dot256_bf16_f32_warp(RNN_W + static_cast<int64_t>(2 * RANK_MK + c_base + 3) * STRIDE_W, st8, lane));
    }

    // ---- Wait for `RDMA_AG(i)` to finish writing `G` ----
    if constexpr (PDL) gdc_wait();

    if (wid == 0) {
        int64_t tok = 0;
        float conf = 0.0f;
        const bool need_c = (with_conf != 0) || (es_mode != 0);
        select_row_1warp(G, 0, 1, S, NS, need_c, exp_mode, tok, conf);
        if (lane == 0) {
            int32_t skip_nx = 0;
            float surv_nx = 1.0f, tok_nx = 1.0f, best_nx = 0.0f;
            if (es_mode != 0) {
                const int32_t gr = (GREEDY_PTR != nullptr) ? *GREEDY_PTR : 0;
                const int32_t sl = (SLOT_PTR != nullptr) ? *SLOT_PTR : 0;
                const float db   = (DB_DEV_PTR != nullptr) ? *DB_DEV_PTR : 0.0f;
                const float lam  = (LAM_DEV_PTR != nullptr) ? *LAM_DEV_PTR : 0.0f;
                eval_es_step_scalar(
                    es_mode, step_idx, K, conf, gr, sl,
                    *SURV_IN, *TOK_IN, *BEST_IN, db, lam, prior_logit,
                    CAL_PTR, STEP_PTR, EMA_PTR, exp_mode,
                    skip_nx, surv_nx, tok_nx, best_nx
                );
            }
            s_tok = tok;
            s_skip_next = skip_nx;
            if (cb == 0) {
                *OUT_TOK_I = tok;
                if (CONF_I != nullptr && need_c) *CONF_I = conf;
                if (es_mode != 0 && SKIP_OUT != nullptr) {
                    *SKIP_OUT = skip_nx;
                    *SURV_OUT = surv_nx;
                    *TOK_OUT  = tok_nx;
                    *BEST_OUT = best_nx;
                }
            }
        }
    }
    __syncthreads();

    if (s_skip_next != 0) {
        if constexpr (PDL) {
            if (tid == 0) gdc_launch_dependents();
        }
        return;
    }

    int64_t tok_cl = s_tok;
    if (tok_cl < 0) tok_cl = 0;
    float emb8[8];
    {
        const uint4 ev = *reinterpret_cast<const uint4*>(W1 + tok_cl * RANK_MK + lane * 8);
        uint16_t ep[8];
        u4_to_u16x8(ev, ep);
        #pragma unroll
        for (int i = 0; i < 8; ++i) emb8[i] = bf16_to_f32(ep[i]);
    }

    if (cb < 32) {
        const int c_base = cb * 8 + wid * 2;
        const float eg0 = dot256_bf16_f32_warp(RNN_W + static_cast<int64_t>(c_base + 0) * STRIDE_W + RANK_MK, emb8, lane);
        const float eg1 = dot256_bf16_f32_warp(RNN_W + static_cast<int64_t>(c_base + 1) * STRIDE_W + RANK_MK, emb8, lane);
        const float ec0 = dot256_bf16_f32_warp(RNN_W + static_cast<int64_t>(RANK_MK + c_base + 0) * STRIDE_W + RANK_MK, emb8, lane);
        const float ec1 = dot256_bf16_f32_warp(RNN_W + static_cast<int64_t>(RANK_MK + c_base + 1) * STRIDE_W + RANK_MK, emb8, lane);
        const uint16_t ns0 = (lane == 0) ? eval_markov_new_state_bf16(__fadd_rn(base_g0, eg0), __fadd_rn(base_c0, ec0), st_c0, false) : 0u;
        const uint16_t ns1 = (lane == 0) ? eval_markov_new_state_bf16(__fadd_rn(base_g1, eg1), __fadd_rn(base_c1, ec1), st_c1, false) : 0u;
        if (lane == 0) {
            OUT_STATE_NEXT[c_base + 0] = ns0;
            OUT_STATE_NEXT[c_base + 1] = ns1;
        }
        if constexpr (PDL) {
            __syncthreads();
            if (tid == 0) gdc_launch_dependents();
        }
    } else {
        const int c_base = (cb - 32) * 16 + wid * 4;
        const float eo0 = dot256_bf16_f32_warp(RNN_W + static_cast<int64_t>(2 * RANK_MK + c_base + 0) * STRIDE_W + RANK_MK, emb8, lane);
        const float eo1 = dot256_bf16_f32_warp(RNN_W + static_cast<int64_t>(2 * RANK_MK + c_base + 1) * STRIDE_W + RANK_MK, emb8, lane);
        const float eo2 = dot256_bf16_f32_warp(RNN_W + static_cast<int64_t>(2 * RANK_MK + c_base + 2) * STRIDE_W + RANK_MK, emb8, lane);
        const float eo3 = dot256_bf16_f32_warp(RNN_W + static_cast<int64_t>(2 * RANK_MK + c_base + 3) * STRIDE_W + RANK_MK, emb8, lane);
        const uint16_t nl0 = (lane == 0) ? eval_markov_latent_bf16(__fadd_rn(base_o0, eo0)) : 0u;
        const uint16_t nl1 = (lane == 0) ? eval_markov_latent_bf16(__fadd_rn(base_o1, eo1)) : 0u;
        const uint16_t nl2 = (lane == 0) ? eval_markov_latent_bf16(__fadd_rn(base_o2, eo2)) : 0u;
        const uint16_t nl3 = (lane == 0) ? eval_markov_latent_bf16(__fadd_rn(base_o3, eo3)) : 0u;
        if (lane == 0) {
            LATENT_NEXT[c_base + 0] = nl0;
            LATENT_NEXT[c_base + 1] = nl1;
            LATENT_NEXT[c_base + 2] = nl2;
            LATENT_NEXT[c_base + 3] = nl3;
        }
        if constexpr (PDL) {
            __syncthreads();
            if (tid == 0) gdc_launch_dependents();
        }
    }
}

// B.5: Batched C=2..32 Cross-Step `dspark_select_and_rnn_step_tc_v10_kernel` (48-CTA `TC48`, `rows >= 2`)
//   - Fuses `argmax_select(step i, rows)` + `markov_rnn_step_tc48(step i+1, rows)` across ALL 48 SMs!
//   - Preloads `STATE_I` (`[min(16, rows), 256]`) into SMEM BEFORE `gdc_wait()` behind `RDMA_AG(i)`,
//     then selects `tok_i` per row after `gdc_wait()` and runs exact `BK=64` Tensor-Core MMAs!
template <bool HAS_BATCH_ES, bool PDL>
__global__ __launch_bounds__(32, 8)
void dspark_select_and_rnn_step_tc_v10_kernel(
    const float*    __restrict__ G,
    int S,
    int NS,
    int64_t*        __restrict__ OUT_TOK_I,
    int64_t out_tok_stride,
    float*          __restrict__ CONF_I,
    int64_t conf_stride,
    int with_conf,
    int step_idx,
    int K,
    const int32_t*  __restrict__ SKIP_IN,       // Batch-All ES (`HAS_BATCH_ES`, 2 <= rows <= 8)
    const float*    __restrict__ SURV_B_IN,     // [rows] fp32 ping-pong input
    int32_t*        __restrict__ SKIP_OUT,      // [1] int32 ping-pong output
    float*          __restrict__ SURV_B_OUT,    // [rows] fp32 ping-pong output
    const int32_t*  __restrict__ GREEDY_ALL_PTR,
    const float*    __restrict__ DB_B_PTR,      // [rows] fp32 per-req calibration bias
    const float*    __restrict__ CAL_PTR,       // [>= 16] fp32 (a[0..7], b[8..15])
    float tau_batch,
    const uint16_t* __restrict__ STATE_I,
    const float*    __restrict__ CTX_NEXT,
    int64_t ctx_row_stride,
    const uint16_t* __restrict__ W1,
    const uint16_t* __restrict__ RNN_W,
    uint16_t*       __restrict__ OUT_STATE_NEXT,
    uint16_t*       __restrict__ LATENT_NEXT,
    int rows,
    int exp_mode
) {
    const int pm = static_cast<int>(blockIdx.x);
    const int cb = static_cast<int>(blockIdx.y);  // 0..47
    const int lane = static_cast<int>(threadIdx.x);
    const int grp = lane >> 2;
    const int lt = lane & 3;
    const int m_base = pm * 16;
    const int m_lim = (rows - m_base < 16) ? (rows - m_base) : 16;
    const bool has_hi8 = (m_base + 8 < rows);
    const int st_u4s = has_hi8 ? 512 : 256;
    const int emb_u4s = has_hi8 ? 128 : 64;
    constexpr int STRIDE_W = 2 * RANK_MK;

    alignas(16) __shared__ uint16_t sStAll[16][264];
    alignas(16) __shared__ uint16_t sEmb[16][72];
    __shared__ int64_t sTok[16];

    // V10 Batch-All Early-Stop (`HAS_BATCH_ES`, 2 <= rows <= 8, pm == 0):
    // `SKIP_IN` was produced by step `i - 1` (settled before fold_argmax(i) & RDMA_AG(i)).
    if constexpr (HAS_BATCH_ES) {
        if (step_idx > 0 && SKIP_IN != nullptr && (*SKIP_IN != 0)) {
            if constexpr (PDL) gdc_wait();
            if (cb == 0) {
                if (lane < rows) {
                    OUT_TOK_I[static_cast<int64_t>(lane) * out_tok_stride] = -1;
                    if (CONF_I != nullptr) CONF_I[static_cast<int64_t>(lane) * conf_stride] = 0.0f;
                    if (SURV_B_OUT != nullptr && SURV_B_IN != nullptr) SURV_B_OUT[lane] = SURV_B_IN[lane];
                }
                if (lane == 0 && SKIP_OUT != nullptr) *SKIP_OUT = 1;
            }
            if constexpr (PDL) {
                __syncwarp();
                if (lane == 0) gdc_launch_dependents();
            }
            return;
        }
    } else {
        (void)step_idx; (void)K; (void)SKIP_IN; (void)SURV_B_IN;
        (void)SKIP_OUT; (void)SURV_B_OUT; (void)GREEDY_ALL_PTR;
        (void)DB_B_PTR; (void)CAL_PTR; (void)tau_batch;
    }

    // Preload `STATE_I` into SMEM BEFORE `gdc_wait()` behind `RDMA_AG(i)`!
    for (int idx = lane; idx < st_u4s; idx += 32) {
        const int r_loc = idx >> 5;
        const int k_col = (idx & 31) * 8;
        const int r_glob = m_base + r_loc;
        *reinterpret_cast<uint4*>(&sStAll[r_loc][k_col]) =
            (r_glob < rows)
                ? ld_ca_u4(STATE_I + static_cast<int64_t>(r_glob) * RANK_MK + k_col)
                : make_uint4(0u, 0u, 0u, 0u);
    }

    if constexpr (PDL) gdc_wait();

    float max_surv = 0.0f;
    float a_cal = 0.0f;
    float b_cal = 0.0f;
    if constexpr (HAS_BATCH_ES) {
        if (CAL_PTR != nullptr && step_idx < K - 1) {
            a_cal = CAL_PTR[step_idx];
            b_cal = CAL_PTR[8 + step_idx];
        }
    }
    const bool need_c = HAS_BATCH_ES || (with_conf != 0);

    for (int r_sub = 0; r_sub < m_lim; ++r_sub) {
        const int r_glob = m_base + r_sub;
        int64_t tok = 0;
        float conf = 0.0f;
        select_row_1warp(G, r_glob, rows, S, NS, need_c, exp_mode, tok, conf);
        if (lane == 0) {
            if (cb == 0) {
                OUT_TOK_I[static_cast<int64_t>(r_glob) * out_tok_stride] = tok;
                if (CONF_I != nullptr && need_c) {
                    CONF_I[static_cast<int64_t>(r_glob) * conf_stride] = conf;
                }
            }
            sTok[r_sub] = (tok < 0) ? 0 : tok;
            if constexpr (HAS_BATCH_ES) {
                if (step_idx < K - 1) {
                    const float pc = fminf(fmaxf(conf, 1e-6f), 1.0f - 1e-6f);
                    const float db = (DB_B_PTR != nullptr) ? DB_B_PTR[r_glob] : 0.0f;
                    const float z = __fadd_rn(
                        __fadd_rn(__fmul_rn(a_cal, __fsub_rn(tl_log_f32(pc, exp_mode), tl_log_f32(__fsub_rn(1.0f, pc), exp_mode))), b_cal),
                        db
                    );
                    const float c = div_full_f32(1.0f, __fadd_rn(1.0f, tl_exp_f32(-z, exp_mode)));
                    const float surv_prev = (SURV_B_IN != nullptr) ? SURV_B_IN[r_glob] : 1.0f;
                    const float surv = __fmul_rn(surv_prev, c);
                    if (cb == 0 && SURV_B_OUT != nullptr) SURV_B_OUT[r_glob] = surv;
                    max_surv = fmaxf(max_surv, surv);
                }
            }
        }
    }
    if constexpr (HAS_BATCH_ES) {
        int32_t skip_nx = 0;
        if (lane == 0) {
            if (step_idx < K - 1) {
                const int32_t gr_all = (GREEDY_ALL_PTR != nullptr) ? *GREEDY_ALL_PTR : 0;
                skip_nx = (gr_all != 0 && max_surv < tau_batch) ? 1 : 0;
            }
            if (cb == 0 && SKIP_OUT != nullptr) *SKIP_OUT = skip_nx;
        }
        skip_nx = __shfl_sync(0xFFFFFFFFu, skip_nx, 0);
        if (skip_nx != 0) {
            if constexpr (PDL) {
                __syncwarp();
                if (lane == 0) gdc_launch_dependents();
            }
            return;
        }
    } else {
        (void)max_surv; (void)a_cal; (void)b_cal;
    }
    if (lane >= m_lim && lane < 16) sTok[lane] = 0;
    __syncwarp();

    if (cb < 32) {
        const int c_col = cb * 8 + grp;
        const uint16_t* wg_row = RNN_W + static_cast<int64_t>(c_col) * STRIDE_W;
        const uint16_t* wc_row = RNN_W + static_cast<int64_t>(RANK_MK + c_col) * STRIDE_W;
        float g0 = 0.f, g1 = 0.f, g2 = 0.f, g3 = 0.f;
        float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;

        for (int k0 = 0; k0 < RANK_MK; k0 += 64) {
            for (int idx = lane; idx < emb_u4s; idx += 32) {
                const int r_loc = idx >> 3;
                const int k_sub = (idx & 7) * 8;
                const int r_glob = m_base + r_loc;
                *reinterpret_cast<uint4*>(&sEmb[r_loc][k_sub]) =
                    (r_glob < rows)
                        ? ld_ca_u4(W1 + sTok[r_loc] * RANK_MK + k0 + k_sub)
                        : make_uint4(0u, 0u, 0u, 0u);
            }
            __syncwarp();
            #pragma unroll
            for (int ks = 0; ks < 4; ++ks) {
                const int ko = ks * 16;
                const uint32_t a0 = *reinterpret_cast<const uint32_t*>(&sEmb[grp][ko + 2 * lt]);
                const uint32_t a1 = has_hi8 ? *reinterpret_cast<const uint32_t*>(&sEmb[grp + 8][ko + 2 * lt]) : 0u;
                const uint32_t a2 = *reinterpret_cast<const uint32_t*>(&sEmb[grp][ko + 8 + 2 * lt]);
                const uint32_t a3 = has_hi8 ? *reinterpret_cast<const uint32_t*>(&sEmb[grp + 8][ko + 8 + 2 * lt]) : 0u;
                const int wk = RANK_MK + k0 + ko;
                mma_m16n8k16_bf16(g0, g1, g2, g3, a0, a1, a2, a3,
                    *reinterpret_cast<const uint32_t*>(wg_row + wk + 2 * lt),
                    *reinterpret_cast<const uint32_t*>(wg_row + wk + 8 + 2 * lt));
                mma_m16n8k16_bf16(c0, c1, c2, c3, a0, a1, a2, a3,
                    *reinterpret_cast<const uint32_t*>(wc_row + wk + 2 * lt),
                    *reinterpret_cast<const uint32_t*>(wc_row + wk + 8 + 2 * lt));
            }
            #pragma unroll
            for (int ks = 0; ks < 4; ++ks) {
                const int ko = k0 + ks * 16;
                const uint32_t a0 = *reinterpret_cast<const uint32_t*>(&sStAll[grp][ko + 2 * lt]);
                const uint32_t a1 = has_hi8 ? *reinterpret_cast<const uint32_t*>(&sStAll[grp + 8][ko + 2 * lt]) : 0u;
                const uint32_t a2 = *reinterpret_cast<const uint32_t*>(&sStAll[grp][ko + 8 + 2 * lt]);
                const uint32_t a3 = has_hi8 ? *reinterpret_cast<const uint32_t*>(&sStAll[grp + 8][ko + 8 + 2 * lt]) : 0u;
                mma_m16n8k16_bf16(g0, g1, g2, g3, a0, a1, a2, a3,
                    *reinterpret_cast<const uint32_t*>(wg_row + ko + 2 * lt),
                    *reinterpret_cast<const uint32_t*>(wg_row + ko + 8 + 2 * lt));
                mma_m16n8k16_bf16(c0, c1, c2, c3, a0, a1, a2, a3,
                    *reinterpret_cast<const uint32_t*>(wc_row + ko + 2 * lt),
                    *reinterpret_cast<const uint32_t*>(wc_row + ko + 8 + 2 * lt));
            }
            __syncwarp();
        }
        const int n0 = cb * 8 + 2 * lt;
        const int n1 = n0 + 1;
        const int r0 = m_base + grp;
        const int r1 = r0 + 8;
        if (r0 < rows) {
            const float* cr = CTX_NEXT + static_cast<int64_t>(r0) * ctx_row_stride;
            const uint16_t ns0 = eval_markov_new_state_bf16(__fadd_rn(cr[n0], g0), __fadd_rn(cr[RANK_MK + n0], c0), bf16_to_f32(sStAll[grp][n0]), false);
            const uint16_t ns1 = eval_markov_new_state_bf16(__fadd_rn(cr[n1], g1), __fadd_rn(cr[RANK_MK + n1], c1), bf16_to_f32(sStAll[grp][n1]), false);
            *reinterpret_cast<uint32_t*>(OUT_STATE_NEXT + static_cast<int64_t>(r0) * RANK_MK + n0) = pack_u16x2(ns0, ns1);
        }
        if (r1 < rows) {
            const float* cr = CTX_NEXT + static_cast<int64_t>(r1) * ctx_row_stride;
            const uint16_t ns0 = eval_markov_new_state_bf16(__fadd_rn(cr[n0], g2), __fadd_rn(cr[RANK_MK + n0], c2), bf16_to_f32(sStAll[grp + 8][n0]), false);
            const uint16_t ns1 = eval_markov_new_state_bf16(__fadd_rn(cr[n1], g3), __fadd_rn(cr[RANK_MK + n1], c3), bf16_to_f32(sStAll[grp + 8][n1]), false);
            *reinterpret_cast<uint32_t*>(OUT_STATE_NEXT + static_cast<int64_t>(r1) * RANK_MK + n0) = pack_u16x2(ns0, ns1);
        }
        if constexpr (PDL) {
            __syncwarp();
            if (lane == 0) gdc_launch_dependents();
        }
    } else {
        const int c_base = (cb - 32) * 16;
        const uint16_t* wo0_row = RNN_W + static_cast<int64_t>(2 * RANK_MK + c_base + grp) * STRIDE_W;
        const uint16_t* wo1_row = RNN_W + static_cast<int64_t>(2 * RANK_MK + c_base + 8 + grp) * STRIDE_W;
        float o00 = 0.f, o01 = 0.f, o02 = 0.f, o03 = 0.f;
        float o10 = 0.f, o11 = 0.f, o12 = 0.f, o13 = 0.f;

        for (int k0 = 0; k0 < RANK_MK; k0 += 64) {
            for (int idx = lane; idx < emb_u4s; idx += 32) {
                const int r_loc = idx >> 3;
                const int k_sub = (idx & 7) * 8;
                const int r_glob = m_base + r_loc;
                *reinterpret_cast<uint4*>(&sEmb[r_loc][k_sub]) =
                    (r_glob < rows)
                        ? ld_ca_u4(W1 + sTok[r_loc] * RANK_MK + k0 + k_sub)
                        : make_uint4(0u, 0u, 0u, 0u);
            }
            __syncwarp();
            #pragma unroll
            for (int ks = 0; ks < 4; ++ks) {
                const int ko = ks * 16;
                const uint32_t a0 = *reinterpret_cast<const uint32_t*>(&sEmb[grp][ko + 2 * lt]);
                const uint32_t a1 = has_hi8 ? *reinterpret_cast<const uint32_t*>(&sEmb[grp + 8][ko + 2 * lt]) : 0u;
                const uint32_t a2 = *reinterpret_cast<const uint32_t*>(&sEmb[grp][ko + 8 + 2 * lt]);
                const uint32_t a3 = has_hi8 ? *reinterpret_cast<const uint32_t*>(&sEmb[grp + 8][ko + 8 + 2 * lt]) : 0u;
                const int wk = RANK_MK + k0 + ko;
                mma_m16n8k16_bf16(o00, o01, o02, o03, a0, a1, a2, a3,
                    *reinterpret_cast<const uint32_t*>(wo0_row + wk + 2 * lt),
                    *reinterpret_cast<const uint32_t*>(wo0_row + wk + 8 + 2 * lt));
                mma_m16n8k16_bf16(o10, o11, o12, o13, a0, a1, a2, a3,
                    *reinterpret_cast<const uint32_t*>(wo1_row + wk + 2 * lt),
                    *reinterpret_cast<const uint32_t*>(wo1_row + wk + 8 + 2 * lt));
            }
            #pragma unroll
            for (int ks = 0; ks < 4; ++ks) {
                const int ko = k0 + ks * 16;
                const uint32_t a0 = *reinterpret_cast<const uint32_t*>(&sStAll[grp][ko + 2 * lt]);
                const uint32_t a1 = has_hi8 ? *reinterpret_cast<const uint32_t*>(&sStAll[grp + 8][ko + 2 * lt]) : 0u;
                const uint32_t a2 = *reinterpret_cast<const uint32_t*>(&sStAll[grp][ko + 8 + 2 * lt]);
                const uint32_t a3 = has_hi8 ? *reinterpret_cast<const uint32_t*>(&sStAll[grp + 8][ko + 8 + 2 * lt]) : 0u;
                mma_m16n8k16_bf16(o00, o01, o02, o03, a0, a1, a2, a3,
                    *reinterpret_cast<const uint32_t*>(wo0_row + ko + 2 * lt),
                    *reinterpret_cast<const uint32_t*>(wo0_row + ko + 8 + 2 * lt));
                mma_m16n8k16_bf16(o10, o11, o12, o13, a0, a1, a2, a3,
                    *reinterpret_cast<const uint32_t*>(wo1_row + ko + 2 * lt),
                    *reinterpret_cast<const uint32_t*>(wo1_row + ko + 8 + 2 * lt));
            }
            __syncwarp();
        }
        const int n0 = c_base + 2 * lt;
        const int n1 = n0 + 1;
        const int n2 = n0 + 8;
        const int n3 = n2 + 1;
        const int r0 = m_base + grp;
        const int r1 = r0 + 8;
        if (r0 < rows) {
            const float* cr = CTX_NEXT + static_cast<int64_t>(r0) * ctx_row_stride;
            *reinterpret_cast<uint32_t*>(LATENT_NEXT + static_cast<int64_t>(r0) * RANK_MK + n0) =
                pack_u16x2(eval_markov_latent_bf16(__fadd_rn(cr[2 * RANK_MK + n0], o00)),
                           eval_markov_latent_bf16(__fadd_rn(cr[2 * RANK_MK + n1], o01)));
            *reinterpret_cast<uint32_t*>(LATENT_NEXT + static_cast<int64_t>(r0) * RANK_MK + n2) =
                pack_u16x2(eval_markov_latent_bf16(__fadd_rn(cr[2 * RANK_MK + n2], o10)),
                           eval_markov_latent_bf16(__fadd_rn(cr[2 * RANK_MK + n3], o11)));
        }
        if (r1 < rows) {
            const float* cr = CTX_NEXT + static_cast<int64_t>(r1) * ctx_row_stride;
            *reinterpret_cast<uint32_t*>(LATENT_NEXT + static_cast<int64_t>(r1) * RANK_MK + n0) =
                pack_u16x2(eval_markov_latent_bf16(__fadd_rn(cr[2 * RANK_MK + n0], o02)),
                           eval_markov_latent_bf16(__fadd_rn(cr[2 * RANK_MK + n1], o03)));
            *reinterpret_cast<uint32_t*>(LATENT_NEXT + static_cast<int64_t>(r1) * RANK_MK + n2) =
                pack_u16x2(eval_markov_latent_bf16(__fadd_rn(cr[2 * RANK_MK + n2], o12)),
                           eval_markov_latent_bf16(__fadd_rn(cr[2 * RANK_MK + n3], o13)));
        }
        if constexpr (PDL) {
            __syncwarp();
            if (lane == 0) gdc_launch_dependents();
        }
    }
}

// B.6: `dspark_fold_argmax_r1_v10_kernel` (`rows == 1`, Top-1 & Top-2, Hardware-Coherent Last-CTA Reduce!)
//   - Fixes V5's SM121a L1 write-back race via `st_cg_f4` + `__threadfence()` + `atom.acq_rel.gpu` + `ld_cg_f4`!
template <bool TOP2, bool USE_FP8, bool HAS_SKIP, bool FUSE_REDUCE, bool HAS_ID_MAP, int FMA_MODE, bool PDL>
__global__ __launch_bounds__(128, 4)
void dspark_fold_argmax_r1_v10_kernel(
    const uint16_t* __restrict__ BASE,
    const uint16_t* __restrict__ LAT,
    const void*     __restrict__ FW,       // uint8_t* when USE_FP8, else uint16_t*
    const float*    __restrict__ FS,
    const uint16_t* __restrict__ FB,
    float*          __restrict__ REC,
    float*          __restrict__ REC_OUT,
    uint32_t*       __restrict__ SEM,
    const int32_t*  __restrict__ SKIP_PTR,
    const int32_t*  __restrict__ ID_MAP,
    int64_t stride_base,
    float softcap,
    int valid,
    int offset,
    int CH,
    int exp_mode
) {
    const int64_t row = static_cast<int64_t>(blockIdx.x);
    const int s = static_cast<int>(blockIdx.y);
    const int S = static_cast<int>(gridDim.y);
    const int tid = static_cast<int>(threadIdx.x);
    const int wid = tid >> 5;
    const int lane = tid & 31;
    const int grp = lane >> 2;   // 0..7
    const int lt = lane & 3;     // 0..3

    if constexpr (PDL) gdc_wait();
    if constexpr (HAS_SKIP) {
        if (*SKIP_PTR != 0) {
            if constexpr (PDL) {
                __syncthreads();
                if (tid == 0) gdc_launch_dependents();
            }
            return;
        }
    }

    __shared__ float sLat[RANK_MK];
    __shared__ float sX[32];
    __shared__ float sVal[8];
    __shared__ int32_t sIdx[8];
    __shared__ float sSe[8];
    __shared__ float sVal2[8];
    __shared__ int32_t sIdx2[8];

    // Load `LAT [256]` into SMEM (`2` bf16s/thread)
    sLat[tid * 2 + 0] = bf16_to_f32(LAT[row * RANK_MK + tid * 2 + 0]);
    sLat[tid * 2 + 1] = bf16_to_f32(LAT[row * RANK_MK + tid * 2 + 1]);
    __syncthreads();

    // Cache this `lt`'s two 32-element slices (`k0`: `lt*32 .. lt*32+31`, `k1`: `128 + lt*32 .. 128 + lt*32+31`)?
    // Even cleaner & fewer regs (`16` floats vs `64`): load from `sLat` (broadcast across `grp` = 0 bank conflicts!)
    const int lo = s * CH;
    const int hi = (lo + CH < valid) ? (lo + CH) : valid;

    float m1_best = -INFINITY;
    int32_t i1_best = 2147483647;
    float m2_best = -INFINITY;
    int32_t i2_best = 2147483647;
    float se_best = 0.0f;

    for (int c0 = lo; c0 < hi; c0 += 32) {
        // Each thread `(wid, grp, lt)` computes 1/4 of col `col = c0 + wid * 8 + grp` (`32` cols total)
        const int col_local = wid * 8 + grp;
        const int col = c0 + col_local;
        const bool cmask = (col < hi);

        float p0 = 0.0f, p1 = 0.0f;
        if (cmask) {
            if constexpr (USE_FP8) {
                const uint8_t* fwp = static_cast<const uint8_t*>(FW) + static_cast<int64_t>(col) * RANK_MK;
                // Half 0 (`0..127`): `lt` handles `lt * 32 + 0..31` (`2` uint4 = 32 bytes)
                #pragma unroll
                for (int u = 0; u < 2; ++u) {
                    const int k_base = lt * 32 + u * 16;
                    const uint4 qv = ld_ca_u4(fwp + k_base);
                    uint16_t e2[8];
                    u4_to_u16x8(qv, e2);
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        float f0, f1;
                        unpack_e4m3x2_to_f32(e2[j], f0, f1);
                        p0 = __fadd_rn(p0, __fadd_rn(__fmul_rn(f0, sLat[k_base + 2 * j]),
                                                     __fmul_rn(f1, sLat[k_base + 2 * j + 1])));
                    }
                }
                // Half 1 (`128..255`)
                #pragma unroll
                for (int u = 0; u < 2; ++u) {
                    const int k_base = 128 + lt * 32 + u * 16;
                    const uint4 qv = ld_ca_u4(fwp + k_base);
                    uint16_t e2[8];
                    u4_to_u16x8(qv, e2);
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        float f0, f1;
                        unpack_e4m3x2_to_f32(e2[j], f0, f1);
                        p1 = __fadd_rn(p1, __fadd_rn(__fmul_rn(f0, sLat[k_base + 2 * j]),
                                                     __fmul_rn(f1, sLat[k_base + 2 * j + 1])));
                    }
                }
            } else {
                const uint16_t* fwp = static_cast<const uint16_t*>(FW) + static_cast<int64_t>(col) * RANK_MK;
                #pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const int k_base = lt * 32 + u * 8;
                    const uint4 bv = ld_ca_u4(fwp + k_base);
                    uint16_t bp[8];
                    u4_to_u16x8(bv, bp);
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        p0 = __fadd_rn(p0, __fmul_rn(bf16_to_f32(bp[j]), sLat[k_base + j]));
                    }
                }
                #pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const int k_base = 128 + lt * 32 + u * 8;
                    const uint4 bv = ld_ca_u4(fwp + k_base);
                    uint16_t bp[8];
                    u4_to_u16x8(bv, bp);
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        p1 = __fadd_rn(p1, __fmul_rn(bf16_to_f32(bp[j]), sLat[k_base + j]));
                    }
                }
            }
        }

        // Reduce `lt = 0..3` within each 4-lane `grp`
        p0 = __fadd_rn(p0, __shfl_xor_sync(0xFFFFFFFFu, p0, 2));
        p0 = __fadd_rn(p0, __shfl_xor_sync(0xFFFFFFFFu, p0, 1));
        p1 = __fadd_rn(p1, __shfl_xor_sync(0xFFFFFFFFu, p1, 2));
        p1 = __fadd_rn(p1, __shfl_xor_sync(0xFFFFFFFFu, p1, 1));

        if (lt == 0) {
            float xv = -INFINITY;
            if (cmask) {
                float dot = __fadd_rn(p0, p1);
                if constexpr (USE_FP8) {
                    dot = __fmul_rn(dot, FS[col]);
                }
                const float b = round_bf16_cvt_f32(__fadd_rn(dot, bf16_to_f32(FB[col])));
                const float a = bf16_to_f32(BASE[row * stride_base + col]);
                const float sc = dspark_softcap_f32(a, b, softcap);
                if (!isnan(sc)) xv = sc;
            }
            sX[col_local] = xv;
        }
        __syncthreads();

        // Warp 0 reduces the 32 columns of `sX` with exact 32-lane Triton semantics!
        if (wid == 0) {
            const int c_lane = c0 + lane;
            const float x = sX[lane];
            float cm1 = x;
            int32_t ci1 = c_lane;
            warp_reduce_max_idx(cm1, ci1);

            float cm2 = -INFINITY;
            int32_t ci2 = 2147483647;
            if constexpr (TOP2) {
                const float x_no1 = (c_lane == ci1) ? -INFINITY : x;
                cm2 = x_no1;
                ci2 = c_lane;
                warp_reduce_max_idx(cm2, ci2);
            }

            const float m_new = fmaxf(m1_best, cm1);
            const float m_safe = (m_new == -INFINITY) ? 0.0f : m_new;
            const float e = (x == -INFINITY) ? 0.0f : tl_exp_f32(__fsub_rn(x, m_safe), exp_mode);
            const float sum_e = warp_sum_f32(e);
            const float sc_old = (m1_best == -INFINITY) ? 0.0f : tl_exp_f32(__fsub_rn(m1_best, m_safe), exp_mode);
            if constexpr (FMA_MODE == 0) {
                se_best = fmaf(se_best, sc_old, sum_e);
            } else {
                se_best = __fadd_rn(__fmul_rn(se_best, sc_old), sum_e);
            }

            if constexpr (TOP2) {
                const bool beat1 = (cm1 > m1_best) || ((cm1 == m1_best) && (ci1 < i1_best));
                float cand2_m;
                int32_t cand2_i;
                if (beat1) {
                    cand2_m = m1_best;
                    cand2_i = i1_best;
                    merge_max_idx(cand2_m, cand2_i, cm2, ci2);
                    m1_best = cm1;
                    i1_best = ci1;
                } else {
                    cand2_m = m2_best;
                    cand2_i = i2_best;
                    merge_max_idx(cand2_m, cand2_i, cm1, ci1);
                }
                m2_best = cand2_m;
                i2_best = cand2_i;
            } else {
                merge_max_idx(m1_best, i1_best, cm1, ci1);
            }
        }
        __syncthreads();
    }

    if (wid == 0 && lane == 0) {
        float out_id1;
        if constexpr (HAS_ID_MAP) {
            const int safe_i = (i1_best < valid) ? i1_best : 0;
            out_id1 = static_cast<float>(ID_MAP[safe_i]);
        } else {
            out_id1 = static_cast<float>(i1_best + offset);
        }
        if constexpr (!TOP2) {
            float* rdst = REC + (row * S + s) * 4;
            st_cg_f4(rdst, m1_best, out_id1, se_best, 0.0f);
        } else {
            float out_id2;
            if constexpr (HAS_ID_MAP) {
                const int safe_i2 = (i2_best < valid) ? i2_best : 0;
                out_id2 = static_cast<float>(ID_MAP[safe_i2]);
            } else {
                out_id2 = static_cast<float>(i2_best + offset);
            }
            float* rdst = REC + (row * S + s) * 8;
            st_cg_f4(rdst + 0, m1_best, out_id1, se_best, m2_best);
            st_cg_f4(rdst + 4, out_id2, 0.0f, 0.0f, 0.0f);
        }
    }

    if constexpr (!FUSE_REDUCE) {
        if constexpr (PDL) {
            __syncthreads();
            if (tid == 0) gdc_launch_dependents();
        }
        return;
    }

    if constexpr (FUSE_REDUCE) {
        __syncthreads();
        __threadfence();
        __shared__ uint32_t s_is_last;
        if (tid == 0) {
            s_is_last = (atom_acq_rel_gpu_add(&SEM[row], 1u) == static_cast<uint32_t>(S - 1)) ? 1u : 0u;
            if (s_is_last) st_rel_gpu_u32(&SEM[row], 0u);
        }
        __syncthreads();
        if (!s_is_last) {
            if constexpr (PDL) {
                if (tid == 0) gdc_launch_dependents();
            }
            return;
        }

        // Coherent 128-thread parallel reduction over `S` slices (`S <= 512`)!
        float v1 = -INFINITY;
        int32_t s1 = 2147483647;
        for (int ss = tid; ss < S; ss += 128) {
            const float4 q0 = ld_cg_f4(REC + (row * S + ss) * (TOP2 ? 8 : 4));
            merge_max_idx(v1, s1, q0.x, ss);
        }
        warp_reduce_max_idx(v1, s1);
        if (lane == 0) { sVal[wid] = v1; sIdx[wid] = s1; }
        __syncthreads();

        if (wid == 0) {
            float gv1 = (lane < 4) ? sVal[lane] : -INFINITY;
            int32_t gs1 = (lane < 4) ? sIdx[lane] : 2147483647;
            warp_reduce_max_idx(gv1, gs1);
            if (lane == 0) { sVal[0] = gv1; sIdx[0] = gs1; }
        }
        __syncthreads();

        const float gm1 = sVal[0];
        const int32_t w1 = sIdx[0];
        const float gm0 = (gm1 == -INFINITY) ? 0.0f : gm1;

        float se_part = 0.0f;
        float v2 = -INFINITY;
        int32_t s2 = 2147483647;
        for (int ss = tid; ss < S; ss += 128) {
            const float4 q0 = ld_cg_f4(REC + (row * S + ss) * (TOP2 ? 8 : 4));
            const float term = (q0.x == -INFINITY) ? 0.0f : __fmul_rn(q0.z, tl_exp_f32(__fsub_rn(q0.x, gm0), exp_mode));
            se_part = __fadd_rn(se_part, term);
            if constexpr (TOP2) {
                const float vp = (ss == w1) ? q0.w : q0.x;
                merge_max_idx(v2, s2, vp, ss);
            }
        }
        se_part = warp_sum_f32(se_part);
        if constexpr (TOP2) warp_reduce_max_idx(v2, s2);
        if (lane == 0) {
            sSe[wid] = se_part;
            if constexpr (TOP2) { sVal2[wid] = v2; sIdx2[wid] = s2; }
        }
        __syncthreads();

        if (wid == 0 && lane == 0) {
            const float gtot = __fadd_rn(__fadd_rn(sSe[0], sSe[1]), __fadd_rn(sSe[2], sSe[3]));
            const float gk1 = (w1 < S) ? ld_cg_f4(REC + (row * S + w1) * (TOP2 ? 8 : 4)).y : 0.0f;
            if constexpr (!TOP2) {
                st_cg_f4(REC_OUT + row * 4, gm1, gk1, gtot, 0.0f);
            } else {
                float gm2 = -INFINITY;
                int32_t w2 = 2147483647;
                #pragma unroll
                for (int w = 0; w < 4; ++w) merge_max_idx(gm2, w2, sVal2[w], sIdx2[w]);
                float gk2 = 0.0f;
                if (w2 < S) {
                    if (w2 == w1) {
                        gk2 = ld_cg_f4(REC + (row * S + w2) * 8 + 4).x;
                    } else {
                        gk2 = ld_cg_f4(REC + (row * S + w2) * 8 + 0).y;
                    }
                }
                st_cg_f4(REC_OUT + row * 8 + 0, gm1, gk1, gtot, gm2);
                st_cg_f4(REC_OUT + row * 8 + 4, gk2, 0.0f, 0.0f, 0.0f);
            }
        }
        if constexpr (PDL) {
            __syncthreads();
            if (tid == 0) gdc_launch_dependents();
        }
    }
}

// B.7: `dspark_fold_argmax_tc_v10_kernel` (`rows >= 2`, C=2..8 `BM=8`, `BM=16`, C=32 `BM=32` Tensor-Core `m16n8k16`)
//   - `BM = 8` (`2 <= rows <= 8`): loads 6.43 MB FP8 `fold_w8` ONCE for all `B = 2..8` requests (8x less DRAM!),
//     zero SMEM/MMA/epilogue work on rows `8..15`, `9.66 KiB` SMEM -> `__launch_bounds__(128, 4)`!
//   - Coalesced 128-bit `ld_ca_u4` `sFW` weight staging (`4` chunks of `BK = 64`, 0-way bank conflicts) replaces
//     V6's uncoalesced 2-byte strided global loads -> full 128-byte L2 transactions!
//   - Exact `N = 8` warp tail gate eliminates 24 wasted columns on `CH = 264`!
//   - Hardware-coherent `st_cg_f4` + `__threadfence()` + `atom.acq_rel.gpu` last-CTA parallel reduction across ANY `rows >= 2`!
template <int BM, bool USE_FP8, bool FUSE_REDUCE, bool HAS_ID_MAP, int FMA_MODE, bool PDL>
__global__ __launch_bounds__(128, (BM <= 16) ? 4 : 2)
void dspark_fold_argmax_tc_v10_kernel(
    const uint16_t* __restrict__ BASE,
    const uint16_t* __restrict__ LAT,
    const void*     __restrict__ FW,
    const float*    __restrict__ FS,
    const uint16_t* __restrict__ FB,
    float*          __restrict__ REC,
    float*          __restrict__ REC_OUT,
    uint32_t*       __restrict__ SEM,
    const int32_t*  __restrict__ SKIP_PTR,
    const int32_t*  __restrict__ ID_MAP,
    int64_t stride_base,
    int rows,
    float softcap,
    int valid,
    int offset,
    int CH,
    int exp_mode
) {
    const int pm = static_cast<int>(blockIdx.x);
    const int s = static_cast<int>(blockIdx.y);
    const int S = static_cast<int>(gridDim.y);
    const int tid = static_cast<int>(threadIdx.x);
    const int wid = tid >> 5;   // 0..3 (`8` N-cols per warp -> `BC = 32`)
    const int lane = tid & 31;
    const int grp = lane >> 2;  // 0..7
    const int lt = lane & 3;    // 0..3
    const int m_base = pm * BM;

    union SfwStage {
        uint8_t  u8[32][80];   // 32 x (64 + 16) B = 2,560 B (20 words/row -> 0-way bank conflicts!)
        uint16_t u16[32][72];  // 32 x (64 + 8) bf16 = 4,608 B (36 words/row -> 0-way bank conflicts!)
    };
    alignas(16) __shared__ uint16_t sLat[BM][RANK_MK + 8];
    alignas(16) __shared__ SfwStage sFW;
    __shared__ float sX[BM][33];  // +1 padding -> 0 bank conflicts!

    if constexpr (PDL) gdc_wait();
    if (SKIP_PTR != nullptr && (*SKIP_PTR != 0)) {
        if constexpr (PDL) {
            __syncthreads();
            if (tid == 0) gdc_launch_dependents();
        }
        return;
    }

    // Load `LAT [BM, 256]` into SMEM (`256` (`BM=8`), `512` (`BM=16`), or `1024` (`BM=32`) uint4s)
    for (int idx = tid; idx < (BM * RANK_MK) / 8; idx += 128) {
        const int r_loc = idx >> 5;        // 0..BM-1
        const int k_col = (idx & 31) * 8;  // 0..248
        const int r_glob = m_base + r_loc;
        *reinterpret_cast<uint4*>(&sLat[r_loc][k_col]) =
            (r_glob < rows)
                ? ld_ca_u4(LAT + static_cast<int64_t>(r_glob) * RANK_MK + k_col)
                : make_uint4(0u, 0u, 0u, 0u);
    }
    __syncthreads();

    // Each warp (`wid = 0..3`) owns `BM / 4` rows (`2` for `BM=8`, `4` for `BM=16`, `8` for `BM=32`)!
    static constexpr int ROWS_PER_W = BM / 4;
    float m_best[ROWS_PER_W];
    int32_t idx_best[ROWS_PER_W];
    float se_best[ROWS_PER_W];
    #pragma unroll
    for (int ri = 0; ri < ROWS_PER_W; ++ri) {
        m_best[ri] = -INFINITY;
        idx_best[ri] = 2147483647;
        se_best[ri] = 0.0f;
    }

    const int lo = s * CH;
    const int hi = (lo + CH < valid) ? (lo + CH) : valid;

    for (int c0 = lo; c0 < hi; c0 += 32) {
        const int w_col_base = c0 + wid * 8;
        float d00 = 0.f, d01 = 0.f, d02 = 0.f, d03 = 0.f;
        float d10 = 0.f, d11 = 0.f, d12 = 0.f, d13 = 0.f;

        // 4 chunks of `BK = 64` with 100% coalesced 128-bit `ld_ca_u4` weight staging into `sFW`!
        #pragma unroll
        for (int k0 = 0; k0 < RANK_MK; k0 += 64) {
            if constexpr (USE_FP8) {
                const int r_w = tid >> 2;         // 0..31
                const int c_w = (tid & 3) * 16;   // 0, 16, 32, 48
                const int g_col = c0 + r_w;
                *reinterpret_cast<uint4*>(&sFW.u8[r_w][c_w]) =
                    (g_col < hi)
                        ? ld_ca_u4(static_cast<const uint8_t*>(FW) + static_cast<int64_t>(g_col) * RANK_MK + k0 + c_w)
                        : make_uint4(0u, 0u, 0u, 0u);
            } else {
                #pragma unroll
                for (int it = 0; it < 2; ++it) {
                    const int idx = it * 128 + tid;
                    const int r_w = idx >> 3;        // 0..31
                    const int c_w = (idx & 7) * 8;   // 0..56
                    const int g_col = c0 + r_w;
                    *reinterpret_cast<uint4*>(&sFW.u16[r_w][c_w]) =
                        (g_col < hi)
                            ? ld_ca_u4(static_cast<const uint16_t*>(FW) + static_cast<int64_t>(g_col) * RANK_MK + k0 + c_w)
                            : make_uint4(0u, 0u, 0u, 0u);
                }
            }
            __syncthreads();

            if (w_col_base < hi) {
                const int wr = wid * 8 + grp;
                #pragma unroll
                for (int ks = 0; ks < 4; ++ks) {
                    const int ko_s = ks * 16;
                    const int ko = k0 + ko_s;
                    uint32_t b0, b1;
                    if constexpr (USE_FP8) {
                        const uint16_t e0 = *reinterpret_cast<const uint16_t*>(&sFW.u8[wr][ko_s + 2 * lt]);
                        const uint16_t e1 = *reinterpret_cast<const uint16_t*>(&sFW.u8[wr][ko_s + 8 + 2 * lt]);
                        b0 = cvt_e4m3x2_to_bf16x2(e0);
                        b1 = cvt_e4m3x2_to_bf16x2(e1);
                    } else {
                        b0 = *reinterpret_cast<const uint32_t*>(&sFW.u16[wr][ko_s + 2 * lt]);
                        b1 = *reinterpret_cast<const uint32_t*>(&sFW.u16[wr][ko_s + 8 + 2 * lt]);
                    }
                    const uint32_t a00 = *reinterpret_cast<const uint32_t*>(&sLat[grp][ko + 2 * lt]);
                    const uint32_t a02 = *reinterpret_cast<const uint32_t*>(&sLat[grp][ko + 8 + 2 * lt]);
                    uint32_t a01 = 0u, a03 = 0u;
                    if constexpr (BM >= 16) {
                        a01 = *reinterpret_cast<const uint32_t*>(&sLat[grp + 8][ko + 2 * lt]);
                        a03 = *reinterpret_cast<const uint32_t*>(&sLat[grp + 8][ko + 8 + 2 * lt]);
                    }
                    mma_m16n8k16_bf16(d00, d01, d02, d03, a00, a01, a02, a03, b0, b1);
                    if constexpr (BM == 32) {
                        const uint32_t a10 = *reinterpret_cast<const uint32_t*>(&sLat[16 + grp][ko + 2 * lt]);
                        const uint32_t a11 = *reinterpret_cast<const uint32_t*>(&sLat[24 + grp][ko + 2 * lt]);
                        const uint32_t a12 = *reinterpret_cast<const uint32_t*>(&sLat[16 + grp][ko + 8 + 2 * lt]);
                        const uint32_t a13 = *reinterpret_cast<const uint32_t*>(&sLat[24 + grp][ko + 8 + 2 * lt]);
                        mma_m16n8k16_bf16(d10, d11, d12, d13, a10, a11, a12, a13, b0, b1);
                    }
                }
            }
            __syncthreads();
        }

        if (w_col_base < hi) {
            const int col0 = wid * 8 + 2 * lt;
            const int col1 = col0 + 1;
            const int gc0 = c0 + col0;
            const int gc1 = c0 + col1;
            const bool ok0 = (gc0 < hi);
            const bool ok1 = (gc1 < hi);
            const float fs0 = (USE_FP8 && ok0) ? FS[gc0] : 1.0f;
            const float fs1 = (USE_FP8 && ok1) ? FS[gc1] : 1.0f;
            const float fb0 = ok0 ? bf16_to_f32(FB[gc0]) : 0.0f;
            const float fb1 = ok1 ? bf16_to_f32(FB[gc1]) : 0.0f;

            auto eval_cell = [&](int r_loc, int gcol, bool ok_c, float dot_v, float fsv, float fbv) -> float {
                const int r_glob = m_base + r_loc;
                if (!ok_c || r_glob >= rows) return -INFINITY;
                const float scaled = USE_FP8 ? __fmul_rn(dot_v, fsv) : dot_v;
                const float b = round_bf16_cvt_f32(__fadd_rn(scaled, fbv));
                const float a = bf16_to_f32(BASE[static_cast<int64_t>(r_glob) * stride_base + gcol]);
                const float sc = dspark_softcap_f32(a, b, softcap);
                return isnan(sc) ? -INFINITY : sc;
            };

            sX[grp][col0] = eval_cell(grp, gc0, ok0, d00, fs0, fb0);
            sX[grp][col1] = eval_cell(grp, gc1, ok1, d01, fs1, fb1);
            if constexpr (BM >= 16) {
                sX[grp + 8][col0] = eval_cell(grp + 8, gc0, ok0, d02, fs0, fb0);
                sX[grp + 8][col1] = eval_cell(grp + 8, gc1, ok1, d03, fs1, fb1);
            }
            if constexpr (BM == 32) {
                sX[16 + grp][col0] = eval_cell(16 + grp, gc0, ok0, d10, fs0, fb0);
                sX[16 + grp][col1] = eval_cell(16 + grp, gc1, ok1, d11, fs1, fb1);
                sX[24 + grp][col0] = eval_cell(24 + grp, gc0, ok0, d12, fs0, fb0);
                sX[24 + grp][col1] = eval_cell(24 + grp, gc1, ok1, d13, fs1, fb1);
            }
        } else {
            // Exact N=8 tail gate: entire 8-col warp tile is >= hi -> fill `-INFINITY` with ZERO MMAs!
            const int col0 = wid * 8 + 2 * lt;
            sX[grp][col0]     = -INFINITY;
            sX[grp][col0 + 1] = -INFINITY;
            if constexpr (BM >= 16) {
                sX[grp + 8][col0]     = -INFINITY;
                sX[grp + 8][col0 + 1] = -INFINITY;
            }
            if constexpr (BM == 32) {
                sX[16 + grp][col0]     = -INFINITY;
                sX[16 + grp][col0 + 1] = -INFINITY;
                sX[24 + grp][col0]     = -INFINITY;
                sX[24 + grp][col0 + 1] = -INFINITY;
            }
        }
        __syncthreads();

        // Each warp `wid` reduces its `ROWS_PER_W` rows across all 32 columns (`lane = 0..31`)!
        const int c_lane = c0 + lane;
        #pragma unroll
        for (int ri = 0; ri < ROWS_PER_W; ++ri) {
            const int r_loc = wid * ROWS_PER_W + ri;
            if (m_base + r_loc < rows) {
                const float x = sX[r_loc][lane];
                float cm = x;
                int32_t cidx = c_lane;
                warp_reduce_max_idx(cm, cidx);
                const float m_new = fmaxf(m_best[ri], cm);
                const float m_safe = (m_new == -INFINITY) ? 0.0f : m_new;
                const float e = (x == -INFINITY) ? 0.0f : tl_exp_f32(__fsub_rn(x, m_safe), exp_mode);
                const float sum_e = warp_sum_f32(e);
                const float sc_old = (m_best[ri] == -INFINITY) ? 0.0f : tl_exp_f32(__fsub_rn(m_best[ri], m_safe), exp_mode);
                if constexpr (FMA_MODE == 0) {
                    se_best[ri] = fmaf(se_best[ri], sc_old, sum_e);
                } else {
                    se_best[ri] = __fadd_rn(__fmul_rn(se_best[ri], sc_old), sum_e);
                }
                merge_max_idx(m_best[ri], idx_best[ri], cm, cidx);
            }
        }
        __syncthreads();
    }

    // Store coherent `.cg` per-slice records
    if (lane == 0) {
        #pragma unroll
        for (int ri = 0; ri < ROWS_PER_W; ++ri) {
            const int r_glob = m_base + wid * ROWS_PER_W + ri;
            if (r_glob < rows) {
                float out_id;
                if constexpr (HAS_ID_MAP) {
                    const int safe_i = (idx_best[ri] < valid) ? idx_best[ri] : 0;
                    out_id = static_cast<float>(ID_MAP[safe_i]);
                } else {
                    out_id = static_cast<float>(idx_best[ri] + offset);
                }
                st_cg_f4(REC + (static_cast<int64_t>(r_glob) * S + s) * 4, m_best[ri], out_id, se_best[ri], 0.0f);
            }
        }
    }

    if constexpr (!FUSE_REDUCE) {
        if constexpr (PDL) {
            __syncthreads();
            if (tid == 0) gdc_launch_dependents();
        }
        return;
    }

    __syncthreads();
    __threadfence();
    __shared__ uint32_t s_is_last;
    if (tid == 0) {
        s_is_last = (atom_acq_rel_gpu_add(&SEM[pm], 1u) == static_cast<uint32_t>(S - 1)) ? 1u : 0u;
        if (s_is_last) st_rel_gpu_u32(&SEM[pm], 0u);
    }
    __syncthreads();
    if (!s_is_last) {
        if constexpr (PDL) {
            if (tid == 0) gdc_launch_dependents();
        }
        return;
    }

    // Parallel multi-warp last-CTA reduction across `rows` (`rows` striped across all 4 warps)!
    for (int r_loc = wid; r_loc < BM; r_loc += 4) {
        const int r_glob = m_base + r_loc;
        if (r_glob < rows) {
            float rm = -INFINITY;
            int32_t rs = 2147483647;
            for (int ss = lane; ss < S; ss += 32) {
                const float4 q = ld_cg_f4(REC + (static_cast<int64_t>(r_glob) * S + ss) * 4);
                merge_max_idx(rm, rs, q.x, ss);
            }
            warp_reduce_max_idx(rm, rs);
            const float rm0 = (rm == -INFINITY) ? 0.0f : rm;
            float se_acc = 0.0f;
            for (int ss = lane; ss < S; ss += 32) {
                const float4 q = ld_cg_f4(REC + (static_cast<int64_t>(r_glob) * S + ss) * 4);
                const float t = (q.x == -INFINITY) ? 0.0f : __fmul_rn(q.z, tl_exp_f32(__fsub_rn(q.x, rm0), exp_mode));
                se_acc = __fadd_rn(se_acc, t);
            }
            const float rtot = warp_sum_f32(se_acc);
            if (lane == 0) {
                const float rbest_i = (rs < S) ? ld_cg_f4(REC + (static_cast<int64_t>(r_glob) * S + rs) * 4).y : 0.0f;
                st_cg_f4(REC_OUT + static_cast<int64_t>(r_glob) * 4, rm, rbest_i, rtot, 0.0f);
            }
        }
    }
    __syncthreads();
    if constexpr (PDL) {
        if (tid == 0) gdc_launch_dependents();
    }
}

// B.8: Standalone `dspark_argmax_select_v10_kernel` (`plain`, `es`, `es_dink`, `top2`)
template <bool PDL>
__global__ __launch_bounds__(32, 8)
void dspark_argmax_select_v10_kernel(
    const float*    __restrict__ G,
    int rows,
    int S,
    int NS,
    int64_t*        __restrict__ OUT_TOK,
    int64_t ostride,
    float*          __restrict__ CONF_OUT,
    int64_t cstride,
    int with_conf,
    int es_mode,       // 0=plain, 1=dink, 2=es, 3=es+hist, 4=top2
    int step_idx,
    int K,
    int32_t*        __restrict__ SKIP_PTR,
    float*          __restrict__ SURV_PTR,
    float*          __restrict__ TOK_PTR,
    float*          __restrict__ BEST_PTR,
    const int32_t*  __restrict__ GREEDY_PTR,
    const int32_t*  __restrict__ SLOT_PTR,
    const float*    __restrict__ DB_DEV_PTR,
    const float*    __restrict__ LAM_DEV_PTR,
    const float*    __restrict__ EMA_PTR,
    const float*    __restrict__ CAL_PTR,
    const float*    __restrict__ STEP_PTR,
    float prior_logit,
    int64_t*        __restrict__ TOK2_OUT,
    float*          __restrict__ P2_OUT,
    int exp_mode
) {
    const int row = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    if constexpr (PDL) gdc_wait();

    if (es_mode == 4) {
        // `_argmax_select_top2_kernel`: `G` is `[NS, rows, 8]`
        float v1 = -INFINITY;
        int32_t w1 = 2147483647;
        for (int r = lane; r < NS; r += 32) {
            const float val = G[(static_cast<int64_t>(r) * rows + row) * 8 + 0];
            merge_max_idx(v1, w1, val, r);
        }
        warp_reduce_max_idx(v1, w1);
        float vp = -INFINITY;
        int32_t w2 = 2147483647;
        const float m0 = (v1 == -INFINITY) ? 0.0f : v1;
        float se_acc = 0.0f;
        for (int r = lane; r < NS; r += 32) {
            const float* rec = G + (static_cast<int64_t>(r) * rows + row) * 8;
            const float rv1 = rec[0];
            const float rse = rec[2];
            const float rv2 = rec[3];
            merge_max_idx(vp, w2, (r == w1) ? rv2 : rv1, r);
            if (rv1 != -INFINITY) {
                se_acc = __fadd_rn(se_acc, __fmul_rn(rse, tl_exp_f32(__fsub_rn(rv1, m0), exp_mode)));
            }
        }
        warp_reduce_max_idx(vp, w2);
        const float tot = warp_sum_f32(se_acc);
        if (lane == 0) {
            const int64_t t1 = (w1 < NS) ? static_cast<int64_t>(G[(static_cast<int64_t>(w1) * rows + row) * 8 + 1]) : 0;
            const int64_t t2 = (w2 < NS)
                ? static_cast<int64_t>(G[(static_cast<int64_t>(w2) * rows + row) * 8 + ((w2 == w1) ? 4 : 1)])
                : 0;
            const float p1 = (tot > 0.0f) ? div_full_f32(1.0f, tot) : 0.0f;
            const float p2 = (tot > 0.0f && vp > -INFINITY) ? div_full_f32(tl_exp_f32(__fsub_rn(vp, m0), exp_mode), tot) : 0.0f;
            OUT_TOK[row] = t1;
            CONF_OUT[row] = p1;
            TOK2_OUT[row] = t2;
            P2_OUT[row] = p2;
        }
        if constexpr (PDL) {
            __syncwarp();
            if (lane == 0) gdc_launch_dependents();
        }
        return;
    }

    if (es_mode != 0 && step_idx > 0 && SKIP_PTR != nullptr && (*SKIP_PTR != 0)) {
        if (lane == 0) {
            OUT_TOK[static_cast<int64_t>(row) * ostride] = -1;
            if (CONF_OUT != nullptr) CONF_OUT[static_cast<int64_t>(row) * cstride] = 0.0f;
        }
        if constexpr (PDL) {
            __syncwarp();
            if (lane == 0) gdc_launch_dependents();
        }
        return;
    }

    int64_t tok = 0;
    float conf = 0.0f;
    const bool need_c = (with_conf != 0) || (es_mode != 0);
    select_row_1warp(G, row, rows, S, NS, need_c, exp_mode, tok, conf);
    if (lane == 0) {
        OUT_TOK[static_cast<int64_t>(row) * ostride] = tok;
        if (CONF_OUT != nullptr && need_c) {
            CONF_OUT[static_cast<int64_t>(row) * cstride] = conf;
        }
        if (es_mode != 0 && es_mode != 5 && row == 0) {
            int32_t skip_nx = 0;
            float surv_nx = 1.0f, tok_nx = 1.0f, best_nx = 0.0f;
            eval_es_step_scalar(
                es_mode, step_idx, K, conf,
                GREEDY_PTR ? *GREEDY_PTR : 0,
                SLOT_PTR ? *SLOT_PTR : 0,
                *SURV_PTR, *TOK_PTR, *BEST_PTR,
                DB_DEV_PTR ? *DB_DEV_PTR : 0.0f,
                LAM_DEV_PTR ? *LAM_DEV_PTR : 0.0f,
                prior_logit, CAL_PTR, STEP_PTR, EMA_PTR, exp_mode,
                skip_nx, surv_nx, tok_nx, best_nx
            );
            *SURV_PTR = surv_nx;
            *TOK_PTR  = tok_nx;
            *BEST_PTR = best_nx;
            if (skip_nx) *SKIP_PTR = 1;
        }
    }
    if constexpr (PDL) {
        __syncwarp();
        if (lane == 0) gdc_launch_dependents();
    }
}

// B.8b (V10): Single-CTA 1-Warp Native Batch-All Early-Stop Argmax Select (`es_mode == 5`, `2 <= rows <= 8`)
//   - Handles BOTH standalone (`MARKOV_FUSE_CROSS_STEP=0`) AND final step (`step_idx == K - 1`)
//     in native SM121a CUDA without any Triton fallback!
template <bool PDL>
__global__ __launch_bounds__(32, 8)
void dspark_argmax_select_batch_es_v10_kernel(
    const float*    __restrict__ G,
    int rows,
    int S,
    int NS,
    int64_t*        __restrict__ OUT_TOK,
    int64_t ostride,
    float*          __restrict__ CONF_OUT,
    int64_t cstride,
    int step_idx,
    int K,
    int32_t*        __restrict__ SKIP_PTR,
    float*          __restrict__ SURV_B_PTR,
    const int32_t*  __restrict__ GREEDY_ALL_PTR,
    const float*    __restrict__ DB_B_PTR,
    const float*    __restrict__ CAL_PTR,
    float tau_batch,
    int exp_mode
) {
    const int lane = static_cast<int>(threadIdx.x);
    if constexpr (PDL) gdc_wait();

    if (step_idx > 0 && SKIP_PTR != nullptr && (*SKIP_PTR != 0)) {
        if (lane < rows) {
            OUT_TOK[static_cast<int64_t>(lane) * ostride] = -1;
            if (CONF_OUT != nullptr) CONF_OUT[static_cast<int64_t>(lane) * cstride] = 0.0f;
        }
        if constexpr (PDL) {
            __syncwarp();
            if (lane == 0) gdc_launch_dependents();
        }
        return;
    }

    const float a_cal = (CAL_PTR != nullptr && step_idx < K - 1) ? CAL_PTR[step_idx] : 0.0f;
    const float b_cal = (CAL_PTR != nullptr && step_idx < K - 1) ? CAL_PTR[8 + step_idx] : 0.0f;
    float max_surv = 0.0f;

    for (int r = 0; r < rows; ++r) {
        int64_t tok = 0;
        float conf = 0.0f;
        select_row_1warp(G, r, rows, S, NS, true, exp_mode, tok, conf);
        if (lane == 0) {
            OUT_TOK[static_cast<int64_t>(r) * ostride] = tok;
            if (CONF_OUT != nullptr) CONF_OUT[static_cast<int64_t>(r) * cstride] = conf;
            if (step_idx < K - 1) {
                const float pc = fminf(fmaxf(conf, 1e-6f), 1.0f - 1e-6f);
                const float db = (DB_B_PTR != nullptr) ? DB_B_PTR[r] : 0.0f;
                const float z = __fadd_rn(
                    __fadd_rn(__fmul_rn(a_cal, __fsub_rn(tl_log_f32(pc, exp_mode), tl_log_f32(__fsub_rn(1.0f, pc), exp_mode))), b_cal),
                    db
                );
                const float c = div_full_f32(1.0f, __fadd_rn(1.0f, tl_exp_f32(-z, exp_mode)));
                const float surv = __fmul_rn((SURV_B_PTR != nullptr) ? SURV_B_PTR[r] : 1.0f, c);
                if (SURV_B_PTR != nullptr) SURV_B_PTR[r] = surv;
                max_surv = fmaxf(max_surv, surv);
            }
        }
    }
    if (lane == 0 && step_idx < K - 1 && SKIP_PTR != nullptr) {
        const int32_t gr_all = (GREEDY_ALL_PTR != nullptr) ? *GREEDY_ALL_PTR : 0;
        if (gr_all != 0 && max_surv < tau_batch) *SKIP_PTR = 1;
    }
    if constexpr (PDL) {
        __syncwarp();
        if (lane == 0) gdc_launch_dependents();
    }
}

// Helper: Exact 1-warp `_vw_conf_kernel` body (`K <= 8`) for request row `r`
static __device__ __forceinline__ void run_vw_conf_row_1warp(
    int r,
    const int64_t* __restrict__ IDX_PTR,
    int idx_stride,
    const float*   __restrict__ CONF_BUF,
    int64_t conf_stride,
    const float*   __restrict__ TEMP_PTR,
    int32_t*       __restrict__ W_PTR,
    const float*   __restrict__ C_PTR,
    const float*   __restrict__ CAL_PTR,
    const float*   __restrict__ STEP_PTR,
    int64_t*       __restrict__ DRAFT_PTR,
    int64_t draft_stride,
    float*         __restrict__ CSLOT_PTR,
    const float*   __restrict__ DB_PTR,
    const float*   __restrict__ ETOK_PTR,
    const float*   __restrict__ EMS_PTR,
    float*         __restrict__ CPRED_PTR,
    int w_cap,
    float tau_override,
    int64_t*       __restrict__ REQ_DRAFT_PTR,
    int64_t req_draft_stride,
    float prior_logit,
    int K,
    int use_hist,
    int use_dink,
    int do_log,
    int write_req_drafts,
    int clear_req_tail,
    int req_k,
    int exp_mode
) {
    const int lane = threadIdx.x & 31;
    const int64_t slot = IDX_PTR[static_cast<int64_t>(r) * idx_stride];
    if (slot < 0) return;

    const float temp = TEMP_PTR[slot];
    const bool kmask = (lane < K);
    const float p = kmask ? CONF_BUF[static_cast<int64_t>(r) * conf_stride + lane] : 0.5f;
    const float pc = fminf(fmaxf(p, 1e-6f), 1.0f - 1e-6f);
    const float a = kmask ? CAL_PTR[lane] : 1.0f;
    const float b = kmask ? CAL_PTR[8 + lane] : 0.0f;
    const float beta = CAL_PTR[16];
    const float tau_cal = CAL_PTR[17];
    const float tau = (tau_override >= 0.0f) ? tau_override : tau_cal;
    float z = __fadd_rn(__fmul_rn(a, __fsub_rn(tl_log_f32(pc, exp_mode), tl_log_f32(__fsub_rn(1.0f, pc), exp_mode))), b);
    const float h = kmask ? C_PTR[slot * K + lane] : 0.7f;
    if (use_dink) {
        z = __fadd_rn(z, DB_PTR[slot]);
    } else if (use_hist) {
        const float hc = fminf(fmaxf(h, 1e-4f), 1.0f - 1e-4f);
        z = __fadd_rn(z, __fmul_rn(beta, __fsub_rn(__fsub_rn(tl_log_f32(hc, exp_mode), tl_log_f32(__fsub_rn(1.0f, hc), exp_mode)), prior_logit)));
    }
    const float c = div_full_f32(1.0f, __fadd_rn(1.0f, tl_exp_f32(-z, exp_mode)));
    if (use_dink && kmask) {
        CPRED_PTR[slot * K + lane] = c;
    }

    float surv = 1.0f;
    float tokens = 1.0f;
    int best_w = 0;
    int w_pick = 0;
    if (use_dink) {
        const float static_lam = CAL_PTR[18];
        const float gamma = CAL_PTR[19];
        const float etok = ETOK_PTR[slot];
        const float ems = fmaxf(EMS_PTR[slot], 1.0f);
        const float lam = (static_lam > 0.0f) ? static_lam : __fmul_rn(gamma, div_full_f32(etok, ems));
        float best_v = __fsub_rn(tokens, __fmul_rn(lam, STEP_PTR[0]));
        int alive = 1;
        int w_tau = 0;
        for (int kk = 0; kk < K; ++kk) {
            const float ck = __shfl_sync(0xFFFFFFFFu, c, kk);
            surv = __fmul_rn(surv, ck);
            tokens = __fadd_rn(tokens, surv);
            const float v = __fsub_rn(tokens, __fmul_rn(lam, STEP_PTR[kk + 1]));
            if ((v > best_v) && (kk < w_cap)) {
                best_v = v;
                best_w = kk + 1;
            }
            if (!((surv >= tau) && (kk < w_cap))) alive = 0;
            w_tau += alive;
        }
        w_pick = (tau_override >= 0.0f) ? w_tau : best_w;
    } else {
        float best_v = div_full_f32(tokens, STEP_PTR[0]);
        int alive = 1;
        int w_tau = 0;
        for (int kk = 0; kk < K; ++kk) {
            const float ck = __shfl_sync(0xFFFFFFFFu, c, kk);
            surv = __fmul_rn(surv, ck);
            tokens = __fadd_rn(tokens, surv);
            const float v = div_full_f32(tokens, STEP_PTR[kk + 1]);
            if ((v > best_v) && (kk < w_cap)) {
                best_v = v;
                best_w = kk + 1;
            }
            if (!((surv >= tau) && (kk < w_cap))) alive = 0;
            w_tau += alive;
        }
        w_pick = (tau > 0.0f) ? w_tau : best_w;
    }
    if (w_pick > w_cap) w_pick = w_cap;
    const int w_new = (temp == 0.0f) ? w_pick : K;
    if (lane == 0) {
        W_PTR[slot] = static_cast<int32_t>(w_new);
    }

    const bool mask_neg1 = (lane >= w_new) && kmask && (temp == 0.0f);
    if (write_req_drafts) {
        if (kmask) {
            int64_t tok = DRAFT_PTR[static_cast<int64_t>(r) * draft_stride + lane];
            if (mask_neg1) tok = -1;
            DRAFT_PTR[static_cast<int64_t>(r) * draft_stride + lane] = tok;
            REQ_DRAFT_PTR[slot * req_draft_stride + lane] = tok;
        }
        if (clear_req_tail && req_k > K) {
            for (int e = K + lane; e < req_k; e += 32) {
                DRAFT_PTR[static_cast<int64_t>(r) * draft_stride + e] = -1;
                REQ_DRAFT_PTR[slot * req_draft_stride + e] = -1;
            }
        }
    } else {
        if (mask_neg1) {
            DRAFT_PTR[static_cast<int64_t>(r) * draft_stride + lane] = -1;
        }
    }
    if (do_log && CSLOT_PTR != nullptr) {
        if (kmask) {
            CSLOT_PTR[slot * 16 + lane] = p;
            CSLOT_PTR[slot * 16 + 8 + lane] = h;
        }
        if (lane == 0) {
            CSLOT_PTR[slot * 16 + 7] = static_cast<float>(w_new);
        }
    }
}

// B.9: Standalone `dspark_vw_conf_v10_kernel` & Fused `dspark_select_last_and_vw_conf_v10_kernel`
//   - Collapses 4 tail launches on C=32 (`argmax_select(k_run-1)` + `draft_tokens.fill_(-1)` +
//     `conf_buf.zero_()` + `_vw_conf_kernel`) and 2 tail launches on C=1 into 1 PDL launch!
template <bool DO_SELECT_LAST, bool PDL>
__global__ __launch_bounds__(32, 8)
void dspark_select_last_and_vw_conf_v10_kernel(
    // Select(k_run - 1) inputs
    const float*    __restrict__ G,
    int num_reqs,
    int S,
    int NS,
    int k_run,
    int n_spec,
    int c_end,
    int has_skip_last,
    const int32_t*  __restrict__ SKIP_LAST_PTR,
    // `_vw_conf_kernel` inputs & outputs
    int run_vw_conf,
    const int64_t*  __restrict__ IDX_PTR,
    int idx_stride,
    float*          __restrict__ CONF_BUF,
    int64_t conf_stride,
    const float*    __restrict__ TEMP_PTR,
    int32_t*        __restrict__ W_PTR,
    const float*    __restrict__ C_PTR,
    const float*    __restrict__ CAL_PTR,
    const float*    __restrict__ STEP_PTR,
    int64_t*        __restrict__ DRAFT_PTR,
    int64_t draft_stride,
    float*          __restrict__ CSLOT_PTR,
    const float*    __restrict__ DB_PTR,
    const float*    __restrict__ ETOK_PTR,
    const float*    __restrict__ EMS_PTR,
    float*          __restrict__ CPRED_PTR,
    int w_cap,
    float tau_override,
    int64_t*        __restrict__ REQ_DRAFT_PTR,
    int64_t req_draft_stride,
    float prior_logit,
    int K,
    int use_hist,
    int use_dink,
    int do_log,
    int write_req_drafts,
    int clear_req_tail,
    int req_k,
    int exp_mode
) {
    const int r = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);

    if constexpr (PDL) gdc_wait();

    if constexpr (DO_SELECT_LAST) {
        const int last_col = k_run - 1;
        const bool skipped = (has_skip_last != 0 && SKIP_LAST_PTR != nullptr && (*SKIP_LAST_PTR != 0));
        int64_t tok = -1;
        float conf = 0.0f;
        if (!skipped) {
            select_row_1warp(G, r, num_reqs, S, NS, CONF_BUF != nullptr, exp_mode, tok, conf);
        }
        if (lane == 0) {
            DRAFT_PTR[static_cast<int64_t>(r) * draft_stride + last_col] = tok;
            if (CONF_BUF != nullptr) {
                CONF_BUF[static_cast<int64_t>(r) * conf_stride + last_col] = conf;
            }
        }
        // Fused tail fill/zero when `k_run < n_spec` (`speculator.py:3216-3220`)
        for (int c = k_run + lane; c < n_spec; c += 32) {
            DRAFT_PTR[static_cast<int64_t>(r) * draft_stride + c] = -1;
        }
        if (CONF_BUF != nullptr) {
            for (int c = k_run + lane; c < c_end; c += 32) {
                CONF_BUF[static_cast<int64_t>(r) * conf_stride + c] = 0.0f;
            }
        }
        __syncwarp();
    }

    if (run_vw_conf) {
        run_vw_conf_row_1warp(
            r, IDX_PTR, idx_stride, CONF_BUF, conf_stride, TEMP_PTR, W_PTR, C_PTR,
            CAL_PTR, STEP_PTR, DRAFT_PTR, draft_stride, CSLOT_PTR, DB_PTR, ETOK_PTR, EMS_PTR,
            CPRED_PTR, w_cap, tau_override, REQ_DRAFT_PTR, req_draft_stride,
            prior_logit, K, use_hist, use_dink, do_log, write_req_drafts,
            clear_req_tail, req_k, exp_mode
        );
    }

    if constexpr (PDL) {
        __syncwarp();
        if (lane == 0) gdc_launch_dependents();
    }
}

// B.10: Tree Speculation (`TREE_SPEC=1`) Fused Kernels:
//   (1) `dspark_select_top2_pack_and_rnn_v10_kernel`: fuses `_argmax_select_top2` + `_pack_beam2_init`
//       + step-1 `markov_rnn_step` (2-row TC when `ws_max >= 2`, or 1-row when `ws_max == 1`)!
template <bool PDL>
__global__ __launch_bounds__(128, 2)
void dspark_select_top2_pack_and_rnn_v10_kernel(
    const float*    __restrict__ G_TOP2,         // [tp, 1, 8]
    int tp,
    const uint16_t* __restrict__ STATE0,         // [1, 256]
    uint16_t*       __restrict__ BEAM_STATE_OUT, // [2, 256] (if ws_max>=2: step-1 state; if ws_max==1: packed state0)
    uint16_t*       __restrict__ BEAM_LAT_OUT,   // [2, 256] (used when ws_max>=2 or [1,256] when ws_max==1)
    int64_t*        __restrict__ BEAM_PREV,      // [2]
    int64_t*        __restrict__ PRIM_TOKS,      // [8]
    float*          __restrict__ CONF_BUF,       // [>=1, >=7]
    int64_t*        __restrict__ SEC_TOKS,       // [4]
    float*          __restrict__ SEC_P,          // [4]
    float*          __restrict__ P2_0,           // [1]
    int has_es,
    int ws_max,
    int32_t*        __restrict__ SKIP_OUT,
    float*          __restrict__ SURV_IO,
    float*          __restrict__ TOK_IO,
    float*          __restrict__ BESTU_IO,
    const int32_t*  __restrict__ GREEDY_PTR,
    const float*    __restrict__ DB_DEV_PTR,
    const float*    __restrict__ LAM_DEV_PTR,
    const float*    __restrict__ CAL_PTR,
    const float*    __restrict__ STEP_PTR,
    const float*    __restrict__ CTX1,           // [3 * 256] (broadcast across both beams with zero copy!)
    const uint16_t* __restrict__ W1,
    const uint16_t* __restrict__ RNN_W,
    int exp_mode
) {
    const int cb = static_cast<int>(blockIdx.y);  // 0..7 (`BC = 32`)
    const int tid = static_cast<int>(threadIdx.x);
    const int wid = tid >> 5;
    const int lane = tid & 31;
    const int grp = lane >> 2;
    const int lt = lane & 3;
    const int c_col = cb * 32 + wid * 8 + grp;
    constexpr int STRIDE_W = 2 * RANK_MK;

    alignas(16) __shared__ uint16_t sSt0[264];
    alignas(16) __shared__ uint16_t sEmb[16][72];
    __shared__ int64_t sT1, sT2;
    __shared__ int32_t sSkip;

    // Preload `STATE0 [256]` into SMEM BEFORE `gdc_wait()` behind Step-0 `RDMA_AG`!
    sSt0[tid * 2 + 0] = STATE0[tid * 2 + 0];
    sSt0[tid * 2 + 1] = STATE0[tid * 2 + 1];

    if constexpr (PDL) gdc_wait();

    if (wid == 0) {
        float v1 = -INFINITY;
        int32_t w1 = 2147483647;
        for (int r = lane; r < tp; r += 32) {
            merge_max_idx(v1, w1, G_TOP2[r * 8 + 0], r);
        }
        warp_reduce_max_idx(v1, w1);
        float vp = -INFINITY;
        int32_t w2 = 2147483647;
        const float m0 = (v1 == -INFINITY) ? 0.0f : v1;
        float se_acc = 0.0f;
        for (int r = lane; r < tp; r += 32) {
            const float rv1 = G_TOP2[r * 8 + 0];
            const float rse = G_TOP2[r * 8 + 2];
            const float rv2 = G_TOP2[r * 8 + 3];
            merge_max_idx(vp, w2, (r == w1) ? rv2 : rv1, r);
            if (rv1 != -INFINITY) {
                se_acc = __fadd_rn(se_acc, __fmul_rn(rse, tl_exp_f32(__fsub_rn(rv1, m0), exp_mode)));
            }
        }
        warp_reduce_max_idx(vp, w2);
        const float tot = warp_sum_f32(se_acc);
        if (lane == 0) {
            const int64_t t1 = (w1 < tp) ? static_cast<int64_t>(G_TOP2[w1 * 8 + 1]) : 0;
            const int64_t t2 = (w2 < tp) ? static_cast<int64_t>(G_TOP2[w2 * 8 + ((w2 == w1) ? 4 : 1)]) : 0;
            const float p1 = (tot > 0.0f) ? div_full_f32(1.0f, tot) : 0.0f;
            const float p2 = (tot > 0.0f && vp > -INFINITY) ? div_full_f32(tl_exp_f32(__fsub_rn(vp, m0), exp_mode), tot) : 0.0f;
            sT1 = t1;
            sT2 = t2;
            int32_t skip_nx = 0;
            if (has_es && GREEDY_PTR != nullptr && (*GREEDY_PTR != 0)) {
                const float pc = fminf(fmaxf(p1, 1e-6f), 1.0f - 1e-6f);
                const float a0 = CAL_PTR[0];
                const float b0 = CAL_PTR[8];
                const float mk_ms = CAL_PTR[20];
                const float db = *DB_DEV_PTR;
                const float lam = *LAM_DEV_PTR;
                const float z = __fadd_rn(__fadd_rn(__fmul_rn(a0, __fsub_rn(tl_log_f32(pc, exp_mode), tl_log_f32(__fsub_rn(1.0f, pc), exp_mode))), b0), db);
                const float c0_es = div_full_f32(1.0f, __fadd_rn(1.0f, tl_exp_f32(-z, exp_mode)));
                const float surv = __fmul_rn(*SURV_IO, c0_es);
                const float tokens = __fadd_rn(*TOK_IO, surv);
                const float u_cur = __fsub_rn(tokens, __fmul_rn(lam, STEP_PTR[1]));
                const float best_u = fmaxf(*BESTU_IO, u_cur);
                if (ws_max == 1) {
                    int can_beat = 0;
                    const float thresh = __fadd_rn(best_u, __fmul_rn(lam, mk_ms));
                    for (int ww = 2; ww <= 7; ++ww) {
                        const float u_ub = __fsub_rn(__fadd_rn(tokens, __fmul_rn(static_cast<float>(ww - 1), surv)), __fmul_rn(lam, STEP_PTR[ww]));
                        if (u_ub > thresh) can_beat = 1;
                    }
                    if (can_beat == 0) skip_nx = 1;
                }
                if (cb == 0) {
                    *SURV_IO = surv;
                    *TOK_IO = tokens;
                    *BESTU_IO = best_u;
                    if (skip_nx && SKIP_OUT != nullptr) *SKIP_OUT = 1;
                }
            }
            sSkip = skip_nx;
            if (cb == 0) {
                BEAM_PREV[0] = t1;
                BEAM_PREV[1] = t2;
                PRIM_TOKS[0] = t1;
                CONF_BUF[0]  = p1;
                SEC_TOKS[0]  = t2;
                SEC_P[0]     = p2;
                P2_0[0]      = p2;
                #pragma unroll
                for (int j = 1; j < 7; ++j) { PRIM_TOKS[j] = -1; CONF_BUF[j] = 0.0f; }
                #pragma unroll
                for (int j = 1; j < 4; ++j) { SEC_TOKS[j] = -1; SEC_P[j] = 0.0f; }
            }
        }
    }
    __syncthreads();

    const int n_rows = (ws_max >= 2) ? 2 : 1;
    if (n_rows == 1 && sSkip != 0) {
        if constexpr (PDL) {
            if (tid == 0) gdc_launch_dependents();
        }
        return;
    }

    float g0 = 0.f, g1 = 0.f, g2 = 0.f, g3 = 0.f;
    float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
    float o0 = 0.f, o1 = 0.f, o2 = 0.f, o3 = 0.f;

    const uint16_t* wg_row = RNN_W + static_cast<int64_t>(c_col) * STRIDE_W;
    const uint16_t* wc_row = RNN_W + static_cast<int64_t>(RANK_MK + c_col) * STRIDE_W;
    const uint16_t* wo_row = RNN_W + static_cast<int64_t>(2 * RANK_MK + c_col) * STRIDE_W;

    if (n_rows == 2) {
        // Exact 2-row Tensor-Core `_markov_rnn_step_tc_kernel` (`BM=16, BK=64`)
        const int64_t tk0 = (sT1 < 0) ? 0 : sT1;
        const int64_t tk1 = (sT2 < 0) ? 0 : sT2;
        for (int k0 = 0; k0 < RANK_MK; k0 += 64) {
            {
                const int r_loc = tid >> 3;
                const int k_sub = (tid & 7) * 8;
                if (r_loc < 2) {
                    const int64_t tk = (r_loc == 0) ? tk0 : tk1;
                    *reinterpret_cast<uint4*>(&sEmb[r_loc][k_sub]) =
                        *reinterpret_cast<const uint4*>(W1 + tk * RANK_MK + k0 + k_sub);
                } else {
                    *reinterpret_cast<uint4*>(&sEmb[r_loc][k_sub]) = make_uint4(0u, 0u, 0u, 0u);
                }
            }
            __syncthreads();
            #pragma unroll
            for (int ks = 0; ks < 4; ++ks) {
                const int ko = ks * 16;
                const uint32_t a0 = *reinterpret_cast<const uint32_t*>(&sEmb[grp][ko + 2 * lt]);
                const uint32_t a1 = 0u;
                const uint32_t a2 = *reinterpret_cast<const uint32_t*>(&sEmb[grp][ko + 8 + 2 * lt]);
                const uint32_t a3 = 0u;
                const int wk = RANK_MK + k0 + ko;
                mma_m16n8k16_bf16(g0, g1, g2, g3, a0, a1, a2, a3,
                    *reinterpret_cast<const uint32_t*>(wg_row + wk + 2 * lt),
                    *reinterpret_cast<const uint32_t*>(wg_row + wk + 8 + 2 * lt));
                mma_m16n8k16_bf16(c0, c1, c2, c3, a0, a1, a2, a3,
                    *reinterpret_cast<const uint32_t*>(wc_row + wk + 2 * lt),
                    *reinterpret_cast<const uint32_t*>(wc_row + wk + 8 + 2 * lt));
                mma_m16n8k16_bf16(o0, o1, o2, o3, a0, a1, a2, a3,
                    *reinterpret_cast<const uint32_t*>(wo_row + wk + 2 * lt),
                    *reinterpret_cast<const uint32_t*>(wo_row + wk + 8 + 2 * lt));
            }
            #pragma unroll
            for (int ks = 0; ks < 4; ++ks) {
                const int ko = k0 + ks * 16;
                // Both row 0 and row 1 share `STATE0` (`grp < 2` -> `sSt0`, `grp >= 2` -> `0`)!
                const uint32_t a0 = (grp < 2) ? *reinterpret_cast<const uint32_t*>(&sSt0[ko + 2 * lt]) : 0u;
                const uint32_t a2 = (grp < 2) ? *reinterpret_cast<const uint32_t*>(&sSt0[ko + 8 + 2 * lt]) : 0u;
                mma_m16n8k16_bf16(g0, g1, g2, g3, a0, 0u, a2, 0u,
                    *reinterpret_cast<const uint32_t*>(wg_row + ko + 2 * lt),
                    *reinterpret_cast<const uint32_t*>(wg_row + ko + 8 + 2 * lt));
                mma_m16n8k16_bf16(c0, c1, c2, c3, a0, 0u, a2, 0u,
                    *reinterpret_cast<const uint32_t*>(wc_row + ko + 2 * lt),
                    *reinterpret_cast<const uint32_t*>(wc_row + ko + 8 + 2 * lt));
                mma_m16n8k16_bf16(o0, o1, o2, o3, a0, 0u, a2, 0u,
                    *reinterpret_cast<const uint32_t*>(wo_row + ko + 2 * lt),
                    *reinterpret_cast<const uint32_t*>(wo_row + ko + 8 + 2 * lt));
            }
            __syncthreads();
        }
        if (grp < 2) {
            const int n0 = cb * 32 + wid * 8 + 2 * lt;
            const int n1 = n0 + 1;
            const float st_a = bf16_to_f32(sSt0[n0]);
            const float st_b = bf16_to_f32(sSt0[n1]);
            const uint16_t ns0 = eval_markov_new_state_bf16(__fadd_rn(CTX1[n0], g0), __fadd_rn(CTX1[RANK_MK + n0], c0), st_a, false);
            const uint16_t ns1 = eval_markov_new_state_bf16(__fadd_rn(CTX1[n1], g1), __fadd_rn(CTX1[RANK_MK + n1], c1), st_b, false);
            const uint16_t nl0 = eval_markov_latent_bf16(__fadd_rn(CTX1[2 * RANK_MK + n0], o0));
            const uint16_t nl1 = eval_markov_latent_bf16(__fadd_rn(CTX1[2 * RANK_MK + n1], o1));
            *reinterpret_cast<uint32_t*>(BEAM_STATE_OUT + static_cast<int64_t>(grp) * RANK_MK + n0) =
                static_cast<uint32_t>(ns0) | (static_cast<uint32_t>(ns1) << 16);
            *reinterpret_cast<uint32_t*>(BEAM_LAT_OUT + static_cast<int64_t>(grp) * RANK_MK + n0) =
                static_cast<uint32_t>(nl0) | (static_cast<uint32_t>(nl1) << 16);
        }
        if constexpr (PDL) {
            __syncthreads();
            if (tid == 0) gdc_launch_dependents();
        }
    } else {
        // `ws_max == 1`: 1-row scalar dot matching `_markov_rnn_step_kernel` (`grid.y == 8` -> `32` dims/CTA = `8` dims/warp)
        const int64_t tk0 = (sT1 < 0) ? 0 : sT1;
        float emb8[8], st8[8];
        const uint4 ev = *reinterpret_cast<const uint4*>(W1 + tk0 * RANK_MK + lane * 8);
        uint16_t ep[8];
        u4_to_u16x8(ev, ep);
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            emb8[i] = bf16_to_f32(ep[i]);
            st8[i]  = bf16_to_f32(sSt0[lane * 8 + i]);
        }
        const int c_base = cb * 32 + wid * 8;
        uint16_t ns_arr[8], nl_arr[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int c = c_base + j;
            const uint16_t* wgr = RNN_W + static_cast<int64_t>(c) * STRIDE_W;
            const uint16_t* wcr = RNN_W + static_cast<int64_t>(RANK_MK + c) * STRIDE_W;
            const uint16_t* wor = RNN_W + static_cast<int64_t>(2 * RANK_MK + c) * STRIDE_W;
            const float rg = __fadd_rn(__fadd_rn(CTX1[c], dot256_bf16_f32_warp(wgr, st8, lane)), dot256_bf16_f32_warp(wgr + RANK_MK, emb8, lane));
            const float rc = __fadd_rn(__fadd_rn(CTX1[RANK_MK + c], dot256_bf16_f32_warp(wcr, st8, lane)), dot256_bf16_f32_warp(wcr + RANK_MK, emb8, lane));
            const float ro = __fadd_rn(__fadd_rn(CTX1[2 * RANK_MK + c], dot256_bf16_f32_warp(wor, st8, lane)), dot256_bf16_f32_warp(wor + RANK_MK, emb8, lane));
            ns_arr[j] = (lane == 0) ? eval_markov_new_state_bf16(rg, rc, bf16_to_f32(sSt0[c]), false) : 0u;
            nl_arr[j] = (lane == 0) ? eval_markov_latent_bf16(ro) : 0u;
        }
        if (lane == 0) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                BEAM_STATE_OUT[c_base + j] = ns_arr[j];
                BEAM_LAT_OUT[c_base + j]   = nl_arr[j];
            }
        }
        if constexpr (PDL) {
            __syncthreads();
            if (tid == 0) gdc_launch_dependents();
        }
    }
}

//   (2) `dspark_select_beam2_and_rnn_v10_kernel`: fuses `argmax_select(2 rows)` + `_pack_beam2_step`
//       + `markov_rnn_step(step + 1)` (`1`-row primary continuation when `step == ws_max - 1`, or `2`-row TC when `step < ws_max - 1`)!
template <bool NEXT_IS_R1, bool PDL>
__global__ __launch_bounds__(128, 4)
void dspark_select_beam2_and_rnn_v10_kernel(
    const float*    __restrict__ G,             // [tp * 2, S, 4]
    int S,
    int NS,
    int step,
    int has_es,
    int can_set_skip,
    int64_t*        __restrict__ BT1_OUT,       // [2]
    float*          __restrict__ BP1_OUT,       // [2]
    int64_t*        __restrict__ PRIM_TOKS,     // [8]
    float*          __restrict__ CONF_BUF,      // [>=1, >=7]
    int64_t*        __restrict__ SEC_TOKS,      // [4]
    float*          __restrict__ SEC_P,         // [4]
    int32_t*        __restrict__ SKIP_OUT,
    float*          __restrict__ SURV_IO,
    float*          __restrict__ TOK_IO,
    float*          __restrict__ BESTU_IO,
    const int32_t*  __restrict__ GREEDY_PTR,
    const float*    __restrict__ DB_DEV_PTR,
    const float*    __restrict__ LAM_DEV_PTR,
    const float*    __restrict__ CAL_PTR,
    const float*    __restrict__ STEP_PTR,
    const uint16_t* __restrict__ BEAM_STATE_IN, // [2, 256] (`[:1]` used when NEXT_IS_R1)
    const float*    __restrict__ CTX_NEXT,      // [3 * 256]
    const uint16_t* __restrict__ W1,
    const uint16_t* __restrict__ RNN_W,
    uint16_t*       __restrict__ OUT_STATE_NEXT,
    uint16_t*       __restrict__ LATENT_NEXT,
    int exp_mode
) {
    const int cb = static_cast<int>(blockIdx.y);
    const int tid = static_cast<int>(threadIdx.x);
    const int wid = tid >> 5;
    const int lane = tid & 31;
    constexpr int STRIDE_W = 2 * RANK_MK;

    __shared__ int64_t sTok0, sTok1;
    __shared__ int32_t sSkipNext;

    if constexpr (NEXT_IS_R1) {
        // 48-CTA 1-row pre-`gdc_wait` overlap on `BEAM_STATE_IN[0, :]`!
        float st8[8];
        {
            const uint4 sv = *reinterpret_cast<const uint4*>(BEAM_STATE_IN + lane * 8);
            uint16_t sp[8];
        u4_to_u16x8(sv, sp);
            #pragma unroll
            for (int i = 0; i < 8; ++i) st8[i] = bf16_to_f32(sp[i]);
        }
        float base_g0 = 0.f, base_g1 = 0.f, base_c0 = 0.f, base_c1 = 0.f, st_c0 = 0.f, st_c1 = 0.f;
        float base_o0 = 0.f, base_o1 = 0.f, base_o2 = 0.f, base_o3 = 0.f;
        if (cb < 32) {
            const int c_base = cb * 8 + wid * 2;
            base_g0 = __fadd_rn(CTX_NEXT[c_base + 0], dot256_bf16_f32_warp(RNN_W + static_cast<int64_t>(c_base + 0) * STRIDE_W, st8, lane));
            base_g1 = __fadd_rn(CTX_NEXT[c_base + 1], dot256_bf16_f32_warp(RNN_W + static_cast<int64_t>(c_base + 1) * STRIDE_W, st8, lane));
            base_c0 = __fadd_rn(CTX_NEXT[RANK_MK + c_base + 0], dot256_bf16_f32_warp(RNN_W + static_cast<int64_t>(RANK_MK + c_base + 0) * STRIDE_W, st8, lane));
            base_c1 = __fadd_rn(CTX_NEXT[RANK_MK + c_base + 1], dot256_bf16_f32_warp(RNN_W + static_cast<int64_t>(RANK_MK + c_base + 1) * STRIDE_W, st8, lane));
            if (lane == 0) {
                st_c0 = bf16_to_f32(BEAM_STATE_IN[c_base + 0]);
                st_c1 = bf16_to_f32(BEAM_STATE_IN[c_base + 1]);
            }
        } else {
            const int c_base = (cb - 32) * 16 + wid * 4;
            base_o0 = __fadd_rn(CTX_NEXT[2 * RANK_MK + c_base + 0], dot256_bf16_f32_warp(RNN_W + static_cast<int64_t>(2 * RANK_MK + c_base + 0) * STRIDE_W, st8, lane));
            base_o1 = __fadd_rn(CTX_NEXT[2 * RANK_MK + c_base + 1], dot256_bf16_f32_warp(RNN_W + static_cast<int64_t>(2 * RANK_MK + c_base + 1) * STRIDE_W, st8, lane));
            base_o2 = __fadd_rn(CTX_NEXT[2 * RANK_MK + c_base + 2], dot256_bf16_f32_warp(RNN_W + static_cast<int64_t>(2 * RANK_MK + c_base + 2) * STRIDE_W, st8, lane));
            base_o3 = __fadd_rn(CTX_NEXT[2 * RANK_MK + c_base + 3], dot256_bf16_f32_warp(RNN_W + static_cast<int64_t>(2 * RANK_MK + c_base + 3) * STRIDE_W, st8, lane));
        }

        if constexpr (PDL) gdc_wait();

        if (wid == 0) {
            int64_t t0 = 0, t1 = 0;
            float p0 = 0.0f, p1 = 0.0f;
            select_row_1warp(G, 0, 2, S, NS, true, exp_mode, t0, p0);
            if (cb == 0) select_row_1warp(G, 1, 2, S, NS, true, exp_mode, t1, p1);
            if (lane == 0) {
                sTok0 = t0;
                int32_t skip_nx = 0;
                if (has_es && GREEDY_PTR != nullptr && (*GREEDY_PTR != 0)) {
                    const float pc = fminf(fmaxf(p0, 1e-6f), 1.0f - 1e-6f);
                    const float as = CAL_PTR[step];
                    const float bs = CAL_PTR[8 + step];
                    const float mk_ms = CAL_PTR[20];
                    const float db = *DB_DEV_PTR;
                    const float lam = *LAM_DEV_PTR;
                    const float z = __fadd_rn(__fadd_rn(__fmul_rn(as, __fsub_rn(tl_log_f32(pc, exp_mode), tl_log_f32(__fsub_rn(1.0f, pc), exp_mode))), bs), db);
                    const float cs = div_full_f32(1.0f, __fadd_rn(1.0f, tl_exp_f32(-z, exp_mode)));
                    const float surv = __fmul_rn(*SURV_IO, cs);
                    const float tokens = __fadd_rn(*TOK_IO, surv);
                    const float u_cur = __fsub_rn(tokens, __fmul_rn(lam, STEP_PTR[step + 1]));
                    const float best_u = fmaxf(*BESTU_IO, u_cur);
                    if (can_set_skip) {
                        int can_beat = 0;
                        const float thresh = __fadd_rn(best_u, __fmul_rn(lam, mk_ms));
                        for (int ww = step + 2; ww <= 7; ++ww) {
                            const float u_ub = __fsub_rn(__fadd_rn(tokens, __fmul_rn(static_cast<float>(ww - step - 1), surv)), __fmul_rn(lam, STEP_PTR[ww]));
                            if (u_ub > thresh) can_beat = 1;
                        }
                        if (can_beat == 0) skip_nx = 1;
                    }
                    if (cb == 0) {
                        *SURV_IO = surv;
                        *TOK_IO  = tokens;
                        *BESTU_IO = best_u;
                        if (skip_nx && SKIP_OUT != nullptr) *SKIP_OUT = 1;
                    }
                }
                sSkipNext = skip_nx;
                if (cb == 0) {
                    BT1_OUT[0] = t0; BT1_OUT[1] = t1;
                    BP1_OUT[0] = p0; BP1_OUT[1] = p1;
                    PRIM_TOKS[step] = t0;
                    CONF_BUF[step]  = p0;
                    SEC_TOKS[step]  = t1;
                    SEC_P[step]     = p1;
                }
            }
        }
        __syncthreads();
        if (sSkipNext != 0) {
            if constexpr (PDL) {
                if (tid == 0) gdc_launch_dependents();
            }
            return;
        }
        const int64_t tk0 = (sTok0 < 0) ? 0 : sTok0;
        float emb8[8];
        {
            const uint4 ev = *reinterpret_cast<const uint4*>(W1 + tk0 * RANK_MK + lane * 8);
            uint16_t ep[8];
        u4_to_u16x8(ev, ep);
            #pragma unroll
            for (int i = 0; i < 8; ++i) emb8[i] = bf16_to_f32(ep[i]);
        }
        if (cb < 32) {
            const int c_base = cb * 8 + wid * 2;
            const float eg0 = dot256_bf16_f32_warp(RNN_W + static_cast<int64_t>(c_base + 0) * STRIDE_W + RANK_MK, emb8, lane);
            const float eg1 = dot256_bf16_f32_warp(RNN_W + static_cast<int64_t>(c_base + 1) * STRIDE_W + RANK_MK, emb8, lane);
            const float ec0 = dot256_bf16_f32_warp(RNN_W + static_cast<int64_t>(RANK_MK + c_base + 0) * STRIDE_W + RANK_MK, emb8, lane);
            const float ec1 = dot256_bf16_f32_warp(RNN_W + static_cast<int64_t>(RANK_MK + c_base + 1) * STRIDE_W + RANK_MK, emb8, lane);
            const uint16_t ns0 = (lane == 0) ? eval_markov_new_state_bf16(__fadd_rn(base_g0, eg0), __fadd_rn(base_c0, ec0), st_c0, false) : 0u;
            const uint16_t ns1 = (lane == 0) ? eval_markov_new_state_bf16(__fadd_rn(base_g1, eg1), __fadd_rn(base_c1, ec1), st_c1, false) : 0u;
            if (lane == 0) {
                OUT_STATE_NEXT[c_base + 0] = ns0;
                OUT_STATE_NEXT[c_base + 1] = ns1;
            }
            if constexpr (PDL) {
                __syncthreads();
                if (tid == 0) gdc_launch_dependents();
            }
        } else {
            const int c_base = (cb - 32) * 16 + wid * 4;
            const float eo0 = dot256_bf16_f32_warp(RNN_W + static_cast<int64_t>(2 * RANK_MK + c_base + 0) * STRIDE_W + RANK_MK, emb8, lane);
            const float eo1 = dot256_bf16_f32_warp(RNN_W + static_cast<int64_t>(2 * RANK_MK + c_base + 1) * STRIDE_W + RANK_MK, emb8, lane);
            const float eo2 = dot256_bf16_f32_warp(RNN_W + static_cast<int64_t>(2 * RANK_MK + c_base + 2) * STRIDE_W + RANK_MK, emb8, lane);
            const float eo3 = dot256_bf16_f32_warp(RNN_W + static_cast<int64_t>(2 * RANK_MK + c_base + 3) * STRIDE_W + RANK_MK, emb8, lane);
            const uint16_t nl0 = (lane == 0) ? eval_markov_latent_bf16(__fadd_rn(base_o0, eo0)) : 0u;
            const uint16_t nl1 = (lane == 0) ? eval_markov_latent_bf16(__fadd_rn(base_o1, eo1)) : 0u;
            const uint16_t nl2 = (lane == 0) ? eval_markov_latent_bf16(__fadd_rn(base_o2, eo2)) : 0u;
            const uint16_t nl3 = (lane == 0) ? eval_markov_latent_bf16(__fadd_rn(base_o3, eo3)) : 0u;
            if (lane == 0) {
                LATENT_NEXT[c_base + 0] = nl0;
                LATENT_NEXT[c_base + 1] = nl1;
                LATENT_NEXT[c_base + 2] = nl2;
                LATENT_NEXT[c_base + 3] = nl3;
            }
            if constexpr (PDL) {
                __syncthreads();
                if (tid == 0) gdc_launch_dependents();
            }
        }
    } else {
        // `step < ws_max - 1` (`ws_max in 3..4`): 2-row TC continuation (`grid.y == 8`)
        alignas(16) __shared__ uint16_t sSt2[2][264];
        alignas(16) __shared__ uint16_t sEmb[16][72];
        #pragma unroll
        for (int r = 0; r < 2; ++r) {
            sSt2[r][tid * 2 + 0] = BEAM_STATE_IN[r * RANK_MK + tid * 2 + 0];
            sSt2[r][tid * 2 + 1] = BEAM_STATE_IN[r * RANK_MK + tid * 2 + 1];
        }
        if constexpr (PDL) gdc_wait();
        if (wid == 0) {
            int64_t t0 = 0, t1 = 0;
            float p0 = 0.0f, p1 = 0.0f;
            select_row_1warp(G, 0, 2, S, NS, true, exp_mode, t0, p0);
            select_row_1warp(G, 1, 2, S, NS, true, exp_mode, t1, p1);
            if (lane == 0) {
                sTok0 = (t0 < 0) ? 0 : t0;
                sTok1 = (t1 < 0) ? 0 : t1;
                if (cb == 0) {
                    BT1_OUT[0] = t0; BT1_OUT[1] = t1;
                    BP1_OUT[0] = p0; BP1_OUT[1] = p1;
                    PRIM_TOKS[step] = t0; CONF_BUF[step] = p0;
                    SEC_TOKS[step]  = t1; SEC_P[step]    = p1;
                    if (has_es && GREEDY_PTR != nullptr && (*GREEDY_PTR != 0)) {
                        const float pc = fminf(fmaxf(p0, 1e-6f), 1.0f - 1e-6f);
                        const float as = CAL_PTR[step];
                        const float bs = CAL_PTR[8 + step];
                        const float db = *DB_DEV_PTR;
                        const float lam = *LAM_DEV_PTR;
                        const float z = __fadd_rn(__fadd_rn(__fmul_rn(as, __fsub_rn(tl_log_f32(pc, exp_mode), tl_log_f32(__fsub_rn(1.0f, pc), exp_mode))), bs), db);
                        const float cs = div_full_f32(1.0f, __fadd_rn(1.0f, tl_exp_f32(-z, exp_mode)));
                        const float surv = __fmul_rn(*SURV_IO, cs);
                        const float tokens = __fadd_rn(*TOK_IO, surv);
                        const float u_cur = __fsub_rn(tokens, __fmul_rn(lam, STEP_PTR[step + 1]));
                        *SURV_IO = surv; *TOK_IO = tokens; *BESTU_IO = fmaxf(*BESTU_IO, u_cur);
                    }
                }
            }
        }
        __syncthreads();
        const int grp = lane >> 2;
        const int lt = lane & 3;
        const int c_col = cb * 32 + wid * 8 + grp;
        float g0 = 0.f, g1 = 0.f, g2 = 0.f, g3 = 0.f;
        float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
        float o0 = 0.f, o1 = 0.f, o2 = 0.f, o3 = 0.f;
        const uint16_t* wg_row = RNN_W + static_cast<int64_t>(c_col) * STRIDE_W;
        const uint16_t* wc_row = RNN_W + static_cast<int64_t>(RANK_MK + c_col) * STRIDE_W;
        const uint16_t* wo_row = RNN_W + static_cast<int64_t>(2 * RANK_MK + c_col) * STRIDE_W;
        for (int k0 = 0; k0 < RANK_MK; k0 += 64) {
            {
                const int r_loc = tid >> 3;
                const int k_sub = (tid & 7) * 8;
                if (r_loc < 2) {
                    const int64_t tk = (r_loc == 0) ? sTok0 : sTok1;
                    *reinterpret_cast<uint4*>(&sEmb[r_loc][k_sub]) = *reinterpret_cast<const uint4*>(W1 + tk * RANK_MK + k0 + k_sub);
                } else {
                    *reinterpret_cast<uint4*>(&sEmb[r_loc][k_sub]) = make_uint4(0u, 0u, 0u, 0u);
                }
            }
            __syncthreads();
            #pragma unroll
            for (int ks = 0; ks < 4; ++ks) {
                const int ko = ks * 16;
                const uint32_t a0 = *reinterpret_cast<const uint32_t*>(&sEmb[grp][ko + 2 * lt]);
                const uint32_t a2 = *reinterpret_cast<const uint32_t*>(&sEmb[grp][ko + 8 + 2 * lt]);
                const int wk = RANK_MK + k0 + ko;
                mma_m16n8k16_bf16(g0, g1, g2, g3, a0, 0u, a2, 0u, *reinterpret_cast<const uint32_t*>(wg_row + wk + 2 * lt), *reinterpret_cast<const uint32_t*>(wg_row + wk + 8 + 2 * lt));
                mma_m16n8k16_bf16(c0, c1, c2, c3, a0, 0u, a2, 0u, *reinterpret_cast<const uint32_t*>(wc_row + wk + 2 * lt), *reinterpret_cast<const uint32_t*>(wc_row + wk + 8 + 2 * lt));
                mma_m16n8k16_bf16(o0, o1, o2, o3, a0, 0u, a2, 0u, *reinterpret_cast<const uint32_t*>(wo_row + wk + 2 * lt), *reinterpret_cast<const uint32_t*>(wo_row + wk + 8 + 2 * lt));
            }
            #pragma unroll
            for (int ks = 0; ks < 4; ++ks) {
                const int ko = k0 + ks * 16;
                const uint32_t a0 = (grp < 2) ? *reinterpret_cast<const uint32_t*>(&sSt2[grp][ko + 2 * lt]) : 0u;
                const uint32_t a2 = (grp < 2) ? *reinterpret_cast<const uint32_t*>(&sSt2[grp][ko + 8 + 2 * lt]) : 0u;
                mma_m16n8k16_bf16(g0, g1, g2, g3, a0, 0u, a2, 0u, *reinterpret_cast<const uint32_t*>(wg_row + ko + 2 * lt), *reinterpret_cast<const uint32_t*>(wg_row + ko + 8 + 2 * lt));
                mma_m16n8k16_bf16(c0, c1, c2, c3, a0, 0u, a2, 0u, *reinterpret_cast<const uint32_t*>(wc_row + ko + 2 * lt), *reinterpret_cast<const uint32_t*>(wc_row + ko + 8 + 2 * lt));
                mma_m16n8k16_bf16(o0, o1, o2, o3, a0, 0u, a2, 0u, *reinterpret_cast<const uint32_t*>(wo_row + ko + 2 * lt), *reinterpret_cast<const uint32_t*>(wo_row + ko + 8 + 2 * lt));
            }
            __syncthreads();
        }
        if (grp < 2) {
            const int n0 = cb * 32 + wid * 8 + 2 * lt;
            const int n1 = n0 + 1;
            const uint16_t ns0 = eval_markov_new_state_bf16(__fadd_rn(CTX_NEXT[n0], g0), __fadd_rn(CTX_NEXT[RANK_MK + n0], c0), bf16_to_f32(sSt2[grp][n0]), false);
            const uint16_t ns1 = eval_markov_new_state_bf16(__fadd_rn(CTX_NEXT[n1], g1), __fadd_rn(CTX_NEXT[RANK_MK + n1], c1), bf16_to_f32(sSt2[grp][n1]), false);
            const uint16_t nl0 = eval_markov_latent_bf16(__fadd_rn(CTX_NEXT[2 * RANK_MK + n0], o0));
            const uint16_t nl1 = eval_markov_latent_bf16(__fadd_rn(CTX_NEXT[2 * RANK_MK + n1], o1));
            *reinterpret_cast<uint32_t*>(OUT_STATE_NEXT + static_cast<int64_t>(grp) * RANK_MK + n0) = static_cast<uint32_t>(ns0) | (static_cast<uint32_t>(ns1) << 16);
            *reinterpret_cast<uint32_t*>(LATENT_NEXT + static_cast<int64_t>(grp) * RANK_MK + n0) = static_cast<uint32_t>(nl0) | (static_cast<uint32_t>(nl1) << 16);
        }
        if constexpr (PDL) {
            __syncthreads();
            if (tid == 0) gdc_launch_dependents();
        }
    }
}

//   (3) `dspark_select_last_and_tree_pack_v10_kernel`: fuses step-6 `argmax_select_es_dink` +
//       `_tree_dink_select_pack_kernel` (`speculator.py:1039`) into 1 PDL launch!
template <bool DO_SELECT_STEP6, bool PDL>
__global__ __launch_bounds__(32, 8)
void dspark_select_last_and_tree_pack_v10_kernel(
    const float*    __restrict__ G_LAST,
    int S,
    int NS,
    int has_skip_last,
    const int32_t*  __restrict__ SKIP_LAST_PTR,
    const int64_t*  __restrict__ IDX_PTR,
    const float*    __restrict__ TEMP_PTR,
    int64_t*        __restrict__ PRIM_TOKS,        // [8]
    float*          __restrict__ CONF_BUF,         // [>=7]
    const int64_t*  __restrict__ SEC_TOKS,         // [4]
    const float*    __restrict__ SEC_P,            // [4]
    const float*    __restrict__ CAL_PTR,
    const float*    __restrict__ STEP_PTR,
    const float*    __restrict__ DB_PTR,
    const float*    __restrict__ ETOK_PTR,
    const float*    __restrict__ EMS_PTR,
    int32_t*        __restrict__ W_PTR,
    float*          __restrict__ CPRED_PTR,
    int64_t*        __restrict__ DRAFT_PTR,
    int64_t*        __restrict__ REQ_DRAFT_PTR,
    int64_t req_draft_stride,
    int32_t*        __restrict__ TREE_ACTIVE,
    int32_t*        __restrict__ N_LIVE,
    int32_t*        __restrict__ TREE_META,
    int32_t*        __restrict__ TREE_PARENT,
    int32_t*        __restrict__ TREE_DEPTH,
    int32_t*        __restrict__ TREE_SCONV_ANC,
    int32_t*        __restrict__ TREE_PATH,
    int ws_max,
    float sib_disc,
    float sib_penalty_ms,
    float sib_scale,
    float p2_min,
    int write_req_drafts,
    int exp_mode
) {
    const int lane = static_cast<int>(threadIdx.x);
    if constexpr (PDL) gdc_wait();

    if constexpr (DO_SELECT_STEP6) {
        const bool skipped = (has_skip_last != 0 && SKIP_LAST_PTR != nullptr && (*SKIP_LAST_PTR != 0));
        int64_t tok6 = -1;
        float conf6 = 0.0f;
        if (!skipped) {
            select_row_1warp(G_LAST, 0, 1, S, NS, true, exp_mode, tok6, conf6);
        }
        if (lane == 0) {
            PRIM_TOKS[6] = tok6;
            CONF_BUF[6]  = conf6;
        }
        __syncwarp();
    }

    const int64_t slot = IDX_PTR[0];
    const int64_t safe_slot = (slot >= 0) ? slot : 0;
    const bool is_greedy = (slot >= 0) && (TEMP_PTR[safe_slot] == 0.0f);

    if (!is_greedy) {
        if (lane == 0) {
            *TREE_ACTIVE = 0;
            *N_LIVE = 8;
            TREE_META[0] = 0; TREE_META[1] = 7; TREE_META[2] = 0; TREE_META[3] = 8;
            for (int i = 0; i < 8; ++i) {
                TREE_PARENT[i] = i - 1;
                TREE_DEPTH[i] = i;
                for (int l = 0; l < 4; ++l) TREE_SCONV_ANC[i * 4 + l] = (i - l >= 0) ? (i - l) : -1;
                for (int d = 0; d < 8; ++d) TREE_PATH[i * 8 + d] = (d <= i) ? d : -1;
            }
            if (slot >= 0) {
                for (int k = 0; k < 7; ++k) {
                    const int64_t pt = PRIM_TOKS[k];
                    DRAFT_PTR[k] = pt;
                    if (write_req_drafts) REQ_DRAFT_PTR[slot * req_draft_stride + k] = pt;
                }
            }
        }
        if constexpr (PDL) {
            __syncwarp();
            if (lane == 0) gdc_launch_dependents();
        }
        return;
    }

    const float db = DB_PTR[slot];
    const float static_lam = CAL_PTR[18];
    const float gamma = CAL_PTR[19];
    const float etok = ETOK_PTR[slot];
    const float ems = fmaxf(EMS_PTR[slot], 1.0f);
    const float lam = (static_lam > 0.0f) ? static_lam : __fmul_rn(gamma, div_full_f32(etok, ems));

    float c_lane = 0.0f;
    if (lane < 7) {
        const float pc = fminf(fmaxf(CONF_BUF[lane], 1e-6f), 1.0f - 1e-6f);
        const float a = CAL_PTR[lane];
        const float b = CAL_PTR[8 + lane];
        const float z = __fadd_rn(__fadd_rn(__fmul_rn(a, __fsub_rn(tl_log_f32(pc, exp_mode), tl_log_f32(__fsub_rn(1.0f, pc), exp_mode))), b), db);
        c_lane = div_full_f32(1.0f, __fadd_rn(1.0f, tl_exp_f32(-z, exp_mode)));
        CPRED_PTR[slot * 7 + lane] = c_lane;
    }
    const float c0 = __shfl_sync(0xFFFFFFFFu, c_lane, 0);
    const float c1 = __shfl_sync(0xFFFFFFFFu, c_lane, 1);
    const float c2 = __shfl_sync(0xFFFFFFFFu, c_lane, 2);
    const float c3 = __shfl_sync(0xFFFFFFFFu, c_lane, 3);
    const float c4 = __shfl_sync(0xFFFFFFFFu, c_lane, 4);
    const float c5 = __shfl_sync(0xFFFFFFFFu, c_lane, 5);
    const float c6 = __shfl_sync(0xFFFFFFFFu, c_lane, 6);

    if (lane == 0) {
        const int64_t pt0 = PRIM_TOKS[0], pt1 = PRIM_TOKS[1], pt2 = PRIM_TOKS[2], pt3 = PRIM_TOKS[3];
        const int64_t pt4 = PRIM_TOKS[4], pt5 = PRIM_TOKS[5], pt6 = PRIM_TOKS[6];

        const float p1_0 = fminf(fmaxf(CONF_BUF[0], 1e-6f), 1.0f - 1e-6f);
        const float p2_0 = fmaxf(SEC_P[0], 0.0f);
        const float q2_obs = fminf(div_full_f32(p2_0, fmaxf(__fsub_rn(1.0f, p1_0), 1e-4f)), 0.95f);
        const int64_t t2_0 = SEC_TOKS[0], st1 = SEC_TOKS[1], st2 = SEC_TOKS[2], st3 = SEC_TOKS[3];
        const bool sec0_ok = (t2_0 >= 0) && (t2_0 != pt0) && (p2_0 >= p2_min);
        const bool sec1_ok = sec0_ok && (st1 >= 0);
        const bool sec2_ok = sec1_ok && (st2 >= 0);
        const bool sec3_ok = sec2_ok && (st3 >= 0);
        const float s0 = sec0_ok ? __fmul_rn(__fmul_rn(__fsub_rn(1.0f, c0), q2_obs), sib_scale) : 0.0f;

        const float a0 = CAL_PTR[0];
        const float b0 = CAL_PTR[8];
        auto eval_cs = [&](float sp_raw) -> float {
            const float sp = fminf(fmaxf(sp_raw, 1e-6f), 1.0f - 1e-6f);
            const float z = __fadd_rn(__fadd_rn(__fmul_rn(a0, __fsub_rn(tl_log_f32(sp, exp_mode), tl_log_f32(__fsub_rn(1.0f, sp), exp_mode))), b0), db);
            return div_full_f32(1.0f, __fadd_rn(1.0f, tl_exp_f32(-z, exp_mode)));
        };
        const float cs1 = eval_cs(SEC_P[1]);
        const float cs2 = eval_cs(SEC_P[2]);
        const float cs3 = eval_cs(SEC_P[3]);

        const float gs0 = 0.0f;
        const float gs1 = s0;
        const float gs2 = __fadd_rn(gs1, __fmul_rn(s0, cs1));
        const float gs3 = __fadd_rn(gs2, __fmul_rn(__fmul_rn(s0, cs1), cs2));
        const float gs4 = __fadd_rn(gs3, __fmul_rn(__fmul_rn(__fmul_rn(s0, cs1), cs2), cs3));

        float best_obj = -1e18f;
        int32_t best_wp = 0;
        int32_t best_ws = 0;
        float surv_p = 1.0f;
        float gp = 1.0f;
        int prim_alive = 1;

        for (int wp = 0; wp < 8; ++wp) {
            if (wp > 0) {
                const int64_t pk = (wp == 1) ? pt0 : ((wp == 2) ? pt1 : ((wp == 3) ? pt2 : ((wp == 4) ? pt3 : ((wp == 5) ? pt4 : ((wp == 6) ? pt5 : pt6)))));
                if (pk < 0) prim_alive = 0;
                const float ck = (wp == 1) ? c0 : ((wp == 2) ? c1 : ((wp == 3) ? c2 : ((wp == 4) ? c3 : ((wp == 5) ? c4 : ((wp == 6) ? c5 : c6)))));
                surv_p = __fmul_rn(surv_p, ck);
                gp = __fadd_rn(gp, surv_p);
            }
            for (int ws = 0; ws <= ws_max; ++ws) {
                if ((wp + ws <= 7) && (ws == 0 || wp >= 1)) {
                    const float gs = (ws == 0) ? gs0 : ((ws == 1) ? gs1 : ((ws == 2) ? gs2 : ((ws == 3) ? gs3 : gs4)));
                    const bool ws_ok = (ws == 0) ? true : ((ws == 1) ? sec0_ok : ((ws == 2) ? sec1_ok : ((ws == 3) ? sec2_ok : sec3_ok)));
                    const int tot_w = wp + ws;
                    float lat;
                    if (ws == 0) {
                        lat = STEP_PTR[tot_w];
                    } else {
                        const float lat_hi = STEP_PTR[tot_w];
                        const float lat_lo = STEP_PTR[tot_w - 1];
                        lat = __fsub_rn(__fadd_rn(lat_hi, sib_penalty_ms), __fmul_rn(sib_disc, __fsub_rn(lat_hi, lat_lo)));
                    }
                    const float obj = __fsub_rn(__fadd_rn(gp, gs), __fmul_rn(lam, lat));
                    if ((prim_alive != 0) && ws_ok && (obj > best_obj)) {
                        best_obj = obj;
                        best_wp = wp;
                        best_ws = ws;
                    }
                }
            }
        }

        const int32_t n_live = 1 + best_wp + best_ws;
        const int32_t is_br = (best_ws > 0) ? 1 : 0;
        W_PTR[slot]   = best_wp;
        *TREE_ACTIVE  = is_br;
        *N_LIVE       = n_live;
        TREE_META[0]  = is_br;
        TREE_META[1]  = best_wp;
        TREE_META[2]  = best_ws;
        TREE_META[3]  = n_live;

        for (int k = 0; k < 7; ++k) {
            const bool is_p = (k < best_wp);
            const bool is_s = (k >= best_wp) && (k < best_wp + best_ws);
            int s_idx = k - best_wp;
            if (s_idx < 0) s_idx = 0;
            if (s_idx > 3) s_idx = 3;
            const int64_t tok = is_p ? PRIM_TOKS[k] : (is_s ? SEC_TOKS[s_idx] : -1);
            DRAFT_PTR[k] = tok;
            if (write_req_drafts) REQ_DRAFT_PTR[slot * req_draft_stride + k] = tok;
        }

        TREE_PARENT[0] = -1;
        TREE_DEPTH[0] = 0;
        TREE_SCONV_ANC[0] = 0; TREE_SCONV_ANC[1] = -1; TREE_SCONV_ANC[2] = -1; TREE_SCONV_ANC[3] = -1;
        TREE_PATH[0] = 0;
        for (int d = 1; d < 8; ++d) TREE_PATH[d] = -1;

        for (int i = 1; i < 8; ++i) {
            const bool is_p = (i <= best_wp);
            const bool is_s = (i > best_wp) && (i < n_live);
            const int s_pos = i - best_wp;
            TREE_PARENT[i] = is_p ? (i - 1) : (is_s ? ((s_pos == 1) ? 0 : (i - 1)) : -1);
            TREE_DEPTH[i]  = is_p ? i : (is_s ? s_pos : 0);
            for (int l = 0; l < 4; ++l) {
                const int anc_p = (l <= i) ? (i - l) : -1;
                const int anc_s = (l < s_pos) ? (i - l) : ((l == s_pos) ? 0 : -1);
                TREE_SCONV_ANC[i * 4 + l] = is_p ? anc_p : (is_s ? anc_s : ((l == 0) ? i : -1));
            }
            for (int d = 0; d < 8; ++d) {
                const int pp = (d <= i) ? d : -1;
                const int ps = (d == 0) ? 0 : ((d <= s_pos) ? (best_wp + d) : -1);
                TREE_PATH[i * 8 + d] = is_p ? pp : (is_s ? ps : ((d == 0) ? 0 : -1));
            }
        }
    }
    if constexpr (PDL) {
        __syncwarp();
        if (lane == 0) gdc_launch_dependents();
    }
}

// ============================================================================
// SECTION C: SyncFree Greedy Target Verification & Speculator Epilogue Kernels
// ============================================================================

// Exact BF16 softcap chain in `sync_free._chain`:
//   y = _rn_bf16(x * inv_cap); t = tanh(y); y = _rn_bf16(|y| < 1.17549435e-38 ? y : t); _rn_bf16(y * cap)
static __device__ __forceinline__ float sf_bf16_chain_f32(float x, float inv_cap, float cap) {
    float y = round_bf16_bits_f32(__fmul_rn(x, inv_cap));
    float t = tanhf(y);
    float m = (fabsf(y) < DENORM_MIN_F) ? y : t;
    y = round_bf16_bits_f32(m);
    return round_bf16_bits_f32(__fmul_rn(y, cap));
}

// Exact FP32 softcap chain in `sync_free._local_kernel`:
static __device__ __forceinline__ float sf_fp32_chain_f32(float x, float inv_cap, float cap, int divmode) {
    const float y = (divmode == 0) ? __fmul_rn(x, inv_cap) : div_rn_f32(x, cap);
    const float t = tanhf(y);
    const float m = (fabsf(y) < DENORM_MIN_F) ? y : t;
    return __fmul_rn(m, cap);
}

// C.1: `sync_free_local_pairs_v10_kernel`
//   - PDL-enabled, L2-streaming (`ld.global.cs`), balanced slices, exact NaN->`-inf` & leftmost tie-break.
//   - When `FUSE_SPLIT1 == true`, last-arriving slice CTA coherently collapses `[M, S, 4] -> [M, 4]` (`16 B/row`),
//     shrinking C=32 RoCEv2 AllGather traffic by 4.0x (`64 B/row -> 16 B/row`)!
template <bool IS_FP32, bool CAP, bool FP32_CHAIN, int DIVMODE, bool FUSE_SPLIT1, bool PDL>
__global__ __launch_bounds__(256, 2)
void sync_free_local_pairs_v10_kernel(
    const void*  __restrict__ LG,
    int64_t sl,
    float*       __restrict__ REC_SPLIT,  // [M * S, 4] (direct output when !FUSE_SPLIT1, scratch when FUSE_SPLIT1)
    float*       __restrict__ PAIR_OUT,   // [M, 4] when FUSE_SPLIT1
    uint32_t*    __restrict__ SEM,        // [M] self-resetting when FUSE_SPLIT1
    int valid,
    int vstart,
    int seg,
    float inv_cap,
    float cap
) {
    const int row = static_cast<int>(blockIdx.x);
    const int sp = static_cast<int>(blockIdx.y);
    const int S = static_cast<int>(gridDim.y);
    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;

    __shared__ float s_val[8];
    __shared__ int32_t s_idx[8];

    if constexpr (PDL) gdc_wait();

    const int lo = sp * seg;
    const int hi = (lo + seg < valid) ? (lo + seg) : valid;

    float best_v = -INFINITY;
    int32_t best_i = (lo < valid) ? lo : 0;
    bool has_any = false;

    if constexpr (IS_FP32) {
        const float* row_ptr = static_cast<const float*>(LG) + static_cast<int64_t>(row) * sl;
        const bool vec_ok = (((sl & 3) == 0) && ((lo & 3) == 0));
        if (vec_ok) {
            for (int v0 = lo + tid * 4; v0 < hi; v0 += 256 * 4) {
                float vals[4];
                if (v0 + 4 <= hi) {
                    const float4 q = ld_cs_f4(row_ptr + v0);
                    vals[0] = q.x; vals[1] = q.y; vals[2] = q.z; vals[3] = q.w;
                } else {
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) vals[j] = (v0 + j < hi) ? row_ptr[v0 + j] : 0.0f;
                }
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int c = v0 + j;
                    if (c < hi) {
                        float x = vals[j];
                        const bool bad = isnan(x);
                        if constexpr (CAP) {
                            x = FP32_CHAIN ? sf_fp32_chain_f32(x, inv_cap, cap, DIVMODE)
                                           : sf_bf16_chain_f32(x, inv_cap, cap);
                        }
                        if (bad || isnan(x)) x = -INFINITY;
                        if (!has_any) { best_v = x; best_i = c; has_any = true; }
                        else { merge_max_idx(best_v, best_i, x, c); }
                    }
                }
            }
        } else {
            for (int c = lo + tid; c < hi; c += 256) {
                float x = row_ptr[c];
                const bool bad = isnan(x);
                if constexpr (CAP) {
                    x = FP32_CHAIN ? sf_fp32_chain_f32(x, inv_cap, cap, DIVMODE)
                                   : sf_bf16_chain_f32(x, inv_cap, cap);
                }
                if (bad || isnan(x)) x = -INFINITY;
                if (!has_any) { best_v = x; best_i = c; has_any = true; }
                else { merge_max_idx(best_v, best_i, x, c); }
            }
        }
    } else {
        const uint16_t* row_ptr = static_cast<const uint16_t*>(LG) + static_cast<int64_t>(row) * sl;
        const bool vec_ok = (((sl & 7) == 0) && ((lo & 7) == 0));
        if (vec_ok) {
            for (int v0 = lo + tid * 8; v0 < hi; v0 += 256 * 8) {
                uint16_t raw[8];
                if (v0 + 8 <= hi) {
                    const uint4 q = ld_cs_u4(row_ptr + v0);
                    u4_to_u16x8(q, raw);
                } else {
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) raw[j] = (v0 + j < hi) ? row_ptr[v0 + j] : 0u;
                }
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int c = v0 + j;
                    if (c < hi) {
                        float x = bf16_to_f32(raw[j]);
                        const bool bad = isnan(x);
                        if constexpr (CAP) x = sf_bf16_chain_f32(x, inv_cap, cap);
                        if (bad || isnan(x)) x = -INFINITY;
                        if (!has_any) { best_v = x; best_i = c; has_any = true; }
                        else { merge_max_idx(best_v, best_i, x, c); }
                    }
                }
            }
        } else {
            for (int c = lo + tid; c < hi; c += 256) {
                float x = bf16_to_f32(row_ptr[c]);
                const bool bad = isnan(x);
                if constexpr (CAP) x = sf_bf16_chain_f32(x, inv_cap, cap);
                if (bad || isnan(x)) x = -INFINITY;
                if (!has_any) { best_v = x; best_i = c; has_any = true; }
                else { merge_max_idx(best_v, best_i, x, c); }
            }
        }
    }

    if (!has_any) {
        best_v = -INFINITY;
        best_i = 2147483647;
    }
    warp_reduce_max_idx(best_v, best_i);
    if (lane == 0) { s_val[warp] = best_v; s_idx[warp] = best_i; }
    __syncthreads();

    if (warp == 0) {
        float gv = (lane < 8) ? s_val[lane] : -INFINITY;
        int32_t gi = (lane < 8) ? s_idx[lane] : 2147483647;
        warp_reduce_max_idx(gv, gi);
        if (lane == 0) {
            if (gi == 2147483647) gi = lo;  // matches `sync_free._local_kernel`: `best_i = lo` when all `-inf` / empty!
            const float out_idx_f32 = static_cast<float>(vstart + gi);
            float* dst = REC_SPLIT + static_cast<int64_t>(row * S + sp) * 4;
            st_cg_f4(dst, gv, out_idx_f32, 0.0f, 0.0f);
        }
    }

    if constexpr (!FUSE_SPLIT1) {
        if constexpr (PDL) {
            __syncthreads();
            if (tid == 0) gdc_launch_dependents();
        }
        return;
    }

    __syncthreads();
    __threadfence();
    __shared__ uint32_t s_is_last;
    if (tid == 0) {
        s_is_last = (atom_acq_rel_gpu_add(&SEM[row], 1u) == static_cast<uint32_t>(S - 1)) ? 1u : 0u;
        if (s_is_last) st_rel_gpu_u32(&SEM[row], 0u);
    }
    __syncthreads();
    if (!s_is_last) {
        if constexpr (PDL) {
            if (tid == 0) gdc_launch_dependents();
        }
        return;
    }

    if (warp == 0) {
        float rv = -INFINITY;
        int32_t rs = 2147483647;
        float rtok_f32 = static_cast<float>(vstart);
        for (int ss = lane; ss < S; ss += 32) {
            const float4 q = ld_cg_f4(REC_SPLIT + static_cast<int64_t>(row * S + ss) * 4);
            merge_max_idx(rv, rs, q.x, ss);
        }
        warp_reduce_max_idx(rv, rs);
        if (lane == 0) {
            if (rs == 2147483647) rs = 0;
            rtok_f32 = ld_cg_f4(REC_SPLIT + static_cast<int64_t>(row * S + rs) * 4).y;
            st_cg_f4(PAIR_OUT + static_cast<int64_t>(row) * 4, rv, rtok_f32, 0.0f, 0.0f);
        }
    }
    if constexpr (PDL) {
        __syncthreads();
        if (tid == 0) gdc_launch_dependents();
    }
}

// Helper: 1-warp exact `sync_free` global argmax on row `logit_idx` (`G` has `[TP * M * SPLIT, 4]`, `float2` load!)
static __device__ __forceinline__ int64_t sf_row_argmax_1warp(
    const float* __restrict__ G,
    int64_t logit_idx,
    int64_t M,
    int tp,
    int split
) {
    const int lane = threadIdx.x & 31;
    const int total = tp * split;
    float best_v = -INFINITY;
    int32_t best_j = 2147483647;
    float best_t_lane = 0.0f;

    for (int j = lane; j < total; j += 32) {
        const int r = j / split;
        const int sidx = j - r * split;
        const int64_t gi = ((static_cast<int64_t>(r) * M + logit_idx) * split + sidx) * 4;
        const float2 vt = *reinterpret_cast<const float2*>(G + gi);
        const bool better = (vt.x > best_v) || ((vt.x == best_v) && (j < best_j));
        if (better || best_j == 2147483647) {
            if (better || (vt.x == -INFINITY && best_j == 2147483647)) {
                best_v = vt.x;
                best_j = j;
                best_t_lane = vt.y;
            }
        }
    }

    // Leftmost tie-breaking across all 32 lanes (when all `-inf`, `j == 0` wins as in `return_indices_tie_break_left=True`!)
    #pragma unroll
    for (int off = 16; off >= 1; off >>= 1) {
        const float ov = __shfl_xor_sync(0xFFFFFFFFu, best_v, off);
        const int32_t oj = __shfl_xor_sync(0xFFFFFFFFu, best_j, off);
        const float ot = __shfl_xor_sync(0xFFFFFFFFu, best_t_lane, off);
        const bool better = (ov > best_v) || ((ov == best_v) && (oj < best_j));
        if (better) {
            best_v = ov;
            best_j = oj;
            best_t_lane = ot;
        }
    }
    return static_cast<int64_t>(static_cast<int32_t>(best_t_lane));
}

// C.2: `sync_free_tok_only_v10_kernel` & Fused `sync_free_k0_tok_status_v10_kernel` (`3 -> 1` launch on K=0!)
template <bool WRITE_STATUS, bool PDL>
__global__ __launch_bounds__(32, 8)
void sync_free_k0_tok_status_v10_kernel(
    const float*   __restrict__ G,
    int M,
    int tp,
    int split,
    int64_t*       __restrict__ TOK_OUT,
    int32_t*       __restrict__ NUM_SAMPLED_OUT,
    int32_t*       __restrict__ NUM_REJECTED_OUT,
    const int32_t* __restrict__ SEQ_LENS,
    const void*    __restrict__ IDX_MAPPING,
    int idx_is_i64,
    const int32_t* __restrict__ PREFILL_LEN
) {
    const int row = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);

    int32_t ns_val = 1;
    if constexpr (WRITE_STATUS) {
        if (lane == 0) {
            const int64_t slot = idx_is_i64
                ? static_cast<const int64_t*>(IDX_MAPPING)[row]
                : static_cast<int64_t>(static_cast<const int32_t*>(IDX_MAPPING)[row]);
            const int32_t sl = SEQ_LENS[row];
            const int32_t pl = PREFILL_LEN[slot];
            ns_val = (sl < pl) ? 0 : 1;
        }
    }

    if constexpr (PDL) gdc_wait();
    const int64_t tok = sf_row_argmax_1warp(G, static_cast<int64_t>(row), static_cast<int64_t>(M), tp, split);
    if (lane == 0) {
        TOK_OUT[row] = tok;
        if constexpr (WRITE_STATUS) {
            NUM_SAMPLED_OUT[row]  = ns_val;
            NUM_REJECTED_OUT[row] = 0;
        }
    }
    if constexpr (PDL) {
        __syncwarp();
        if (lane == 0) gdc_launch_dependents();
    }
}

// C.3: `sync_free_fused_reject_v10_kernel`
//   - Direct `ctypes` C dispatch (~0.8 us vs ~12 us eager Triton Python wrapper on EVERY step!).
//   - Preloads `cu_num_logits`, `idx_mapping`, `prefill_len`, `seq_lens`, `logits_indices`, `input_ids`,
//     and `positions` BEFORE `gdc_wait()` behind `sync_free` RDMA AllGather!
template <bool PDL>
__global__ __launch_bounds__(32, 8)
void sync_free_fused_reject_v10_kernel(
    const float*   __restrict__ G,
    int M,
    int tp,
    int split,
    int64_t*       __restrict__ SAMPLED,
    int64_t sampled_stride,
    int32_t*       __restrict__ NUM_SAMPLED,
    int32_t*       __restrict__ NUM_REJECTED,
    const int32_t* __restrict__ SEQ_LENS,
    const void*    __restrict__ INPUT_IDS,
    const void*    __restrict__ POSITIONS,
    const void*    __restrict__ LOGITS_INDICES,
    const void*    __restrict__ IDX_MAPPING,
    const int32_t* __restrict__ PREFILL_LEN,
    const int32_t* __restrict__ CU_NUM_LOGITS,
    uint32_t dtype_flags  // bit0: ids_i64, bit1: pos_i64, bit2: lidx_i64, bit3: map_i64
) {
    const int64_t req_idx = static_cast<int64_t>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);

    // ---- Preload all request metadata & up to 32 draft tokens BEFORE `gdc_wait()`! ----
    const int64_t start_idx = static_cast<int64_t>(CU_NUM_LOGITS[req_idx]);
    const int64_t end_idx   = static_cast<int64_t>(CU_NUM_LOGITS[req_idx + 1]);
    const int64_t num_draft = end_idx - start_idx - 1;

    const int64_t req_state_idx = (dtype_flags & 8u)
        ? static_cast<const int64_t*>(IDX_MAPPING)[req_idx]
        : static_cast<int64_t>(static_cast<const int32_t*>(IDX_MAPPING)[req_idx]);
    const int64_t req_prefill_len = static_cast<int64_t>(PREFILL_LEN[req_state_idx]);
    const int64_t seq_len = static_cast<int64_t>(SEQ_LENS[req_idx]);

    int64_t pre_dtok = -1;
    int32_t pre_valid = 0;
    if (lane < num_draft && num_draft <= 32) {
        const int64_t lpos = start_idx + lane + 1;
        const int64_t tok_idx = (dtype_flags & 4u)
            ? static_cast<const int64_t*>(LOGITS_INDICES)[lpos]
            : static_cast<int64_t>(static_cast<const int32_t*>(LOGITS_INDICES)[lpos]);
        const int64_t dtok = (dtype_flags & 1u)
            ? static_cast<const int64_t*>(INPUT_IDS)[tok_idx]
            : static_cast<int64_t>(static_cast<const int32_t*>(INPUT_IDS)[tok_idx]);
        const int64_t dpos = (dtype_flags & 2u)
            ? static_cast<const int64_t*>(POSITIONS)[tok_idx]
            : static_cast<int64_t>(static_cast<const int32_t*>(POSITIONS)[tok_idx]);
        pre_dtok = dtok;
        pre_valid = (dtok >= 0 && dpos >= req_prefill_len) ? 1 : 0;
    }

    if constexpr (PDL) gdc_wait();

    int64_t accepted_len = 0;
    for (int64_t i = 0; i < num_draft; ++i) {
        int64_t draft_tok;
        bool is_valid_draft;
        if (num_draft <= 32) {
            const int src_lane = static_cast<int>(i);
            const uint32_t d_lo = __shfl_sync(0xFFFFFFFFu, static_cast<uint32_t>(static_cast<uint64_t>(pre_dtok) & 0xFFFFFFFFull), src_lane);
            const uint32_t d_hi = __shfl_sync(0xFFFFFFFFu, static_cast<uint32_t>(static_cast<uint64_t>(pre_dtok) >> 32), src_lane);
            draft_tok = static_cast<int64_t>((static_cast<uint64_t>(d_hi) << 32) | d_lo);
            is_valid_draft = (__shfl_sync(0xFFFFFFFFu, pre_valid, src_lane) != 0);
        } else {
            const int64_t lpos = start_idx + i + 1;
            const int64_t tok_idx = (dtype_flags & 4u)
                ? static_cast<const int64_t*>(LOGITS_INDICES)[lpos]
                : static_cast<int64_t>(static_cast<const int32_t*>(LOGITS_INDICES)[lpos]);
            draft_tok = (dtype_flags & 1u)
                ? static_cast<const int64_t*>(INPUT_IDS)[tok_idx]
                : static_cast<int64_t>(static_cast<const int32_t*>(INPUT_IDS)[tok_idx]);
            const int64_t dpos = (dtype_flags & 2u)
                ? static_cast<const int64_t*>(POSITIONS)[tok_idx]
                : static_cast<int64_t>(static_cast<const int32_t*>(POSITIONS)[tok_idx]);
            is_valid_draft = (draft_tok >= 0) && (dpos >= req_prefill_len);
        }

        const int64_t target_tok = sf_row_argmax_1warp(G, start_idx + i, static_cast<int64_t>(M), tp, split);
        const bool accepted = (target_tok == draft_tok) && is_valid_draft;
        if (lane == 0) {
            SAMPLED[req_idx * sampled_stride + i] = accepted ? draft_tok : target_tok;
        }
        if (accepted) {
            accepted_len += 1;
        } else {
            break;
        }
    }

    if (accepted_len == num_draft) {
        const int64_t bonus_tok = sf_row_argmax_1warp(G, start_idx + num_draft, static_cast<int64_t>(M), tp, split);
        if (lane == 0) {
            SAMPLED[req_idx * sampled_stride + num_draft] = bonus_tok;
        }
    }

    const bool incomplete = (seq_len < req_prefill_len);
    const int32_t ns_out = incomplete ? 0 : static_cast<int32_t>(accepted_len + 1);
    const int32_t nr_out = incomplete ? 0 : static_cast<int32_t>(num_draft - accepted_len);
    if (lane == 0) {
        NUM_SAMPLED[req_idx]  = ns_out;
        NUM_REJECTED[req_idx] = nr_out;
    }
    if constexpr (PDL) {
        __syncwarp();
        if (lane == 0) gdc_launch_dependents();
    }
}

// C.4: `sync_free_fused_tree_reject_v10_kernel` (`C=1 TREE_SPEC=1`, `MAX_M <= 16`)
template <bool PDL>
__global__ __launch_bounds__(32, 8)
void sync_free_fused_tree_reject_v10_kernel(
    const float*   __restrict__ G,
    int M,
    int tp,
    int split,
    int max_m,
    int64_t*       __restrict__ SAMPLED,
    int64_t sampled_stride,
    int32_t*       __restrict__ NUM_SAMPLED,
    int32_t*       __restrict__ NUM_REJECTED,
    const int32_t* __restrict__ SEQ_LENS,
    const void*    __restrict__ INPUT_IDS,
    const void*    __restrict__ POSITIONS,
    const void*    __restrict__ LOGITS_INDICES,
    const void*    __restrict__ IDX_MAPPING,
    const int32_t* __restrict__ PREFILL_LEN,
    const int32_t* __restrict__ CU_NUM_LOGITS,
    const int32_t* __restrict__ TREE_PARENT,
    const int32_t* __restrict__ N_LIVE_PTR,
    int32_t*       __restrict__ ACCEPTED_PATH,
    int32_t*       __restrict__ ACCEPTED_LEN,
    int32_t*       __restrict__ NEEDS_COMPACT,
    uint32_t dtype_flags
) {
    const int64_t req_idx = static_cast<int64_t>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);

    __shared__ int32_t sPar[16];
    __shared__ int64_t sDtok[16];
    __shared__ int32_t sValid[16];

    // Preload tree topology & draft node tokens BEFORE `gdc_wait()` behind RDMA AG!
    const int64_t start_idx = static_cast<int64_t>(CU_NUM_LOGITS[req_idx]);
    const int64_t end_idx   = static_cast<int64_t>(CU_NUM_LOGITS[req_idx + 1]);
    const int64_t num_draft = end_idx - start_idx - 1;
    const int64_t req_state_idx = (dtype_flags & 8u)
        ? static_cast<const int64_t*>(IDX_MAPPING)[req_idx]
        : static_cast<int64_t>(static_cast<const int32_t*>(IDX_MAPPING)[req_idx]);
    const int64_t req_prefill_len = static_cast<int64_t>(PREFILL_LEN[req_state_idx]);
    const int64_t seq_len = static_cast<int64_t>(SEQ_LENS[req_idx]);
    const int cap_m = (max_m <= 16) ? max_m : 16;
    int n_live = static_cast<int>(*N_LIVE_PTR);
    if (n_live > cap_m) n_live = cap_m;

    if (lane < cap_m) {
        sPar[lane] = TREE_PARENT[lane];
        if (lane >= 1 && lane < n_live) {
            const int64_t lpos = start_idx + lane;
            const int64_t tok_idx = (dtype_flags & 4u)
                ? static_cast<const int64_t*>(LOGITS_INDICES)[lpos]
                : static_cast<int64_t>(static_cast<const int32_t*>(LOGITS_INDICES)[lpos]);
            const int64_t dtok = (dtype_flags & 1u)
                ? static_cast<const int64_t*>(INPUT_IDS)[tok_idx]
                : static_cast<int64_t>(static_cast<const int32_t*>(INPUT_IDS)[tok_idx]);
            const int64_t dpos = (dtype_flags & 2u)
                ? static_cast<const int64_t*>(POSITIONS)[tok_idx]
                : static_cast<int64_t>(static_cast<const int32_t*>(POSITIONS)[tok_idx]);
            sDtok[lane] = dtok;
            sValid[lane] = (dtok >= 0 && dpos >= req_prefill_len) ? 1 : 0;
        } else {
            sDtok[lane] = -1;
            sValid[lane] = 0;
        }
    }
    __syncwarp();

    if constexpr (PDL) gdc_wait();

    int64_t cur = 0;
    int64_t depth = 0;
    int32_t needs_cmp = 0;
    if (lane == 0) ACCEPTED_PATH[0] = 0;

    for (int step = 0; step < cap_m; ++step) {
        const int64_t target_tok = sf_row_argmax_1warp(G, start_idx + cur, static_cast<int64_t>(M), tp, split);
        if (lane == 0) {
            SAMPLED[req_idx * sampled_stride + depth] = target_tok;
        }
        int64_t matched_c = -1;
        for (int c = 1; c < n_live; ++c) {
            if (sPar[c] == static_cast<int32_t>(cur) && sValid[c] != 0 && sDtok[c] == target_tok) {
                matched_c = c;
                break;
            }
        }
        if (matched_c >= 0) {
            depth += 1;
            if (lane == 0) ACCEPTED_PATH[depth] = static_cast<int32_t>(matched_c);
            if (matched_c != depth) needs_cmp = 1;
            cur = matched_c;
        } else {
            break;
        }
    }

    const bool incomplete = (seq_len < req_prefill_len);
    const int32_t ns_out = incomplete ? 0 : static_cast<int32_t>(depth + 1);
    const int32_t nr_out = incomplete ? 0 : static_cast<int32_t>(num_draft - depth);
    if (lane == 0) {
        NUM_SAMPLED[req_idx]  = ns_out;
        NUM_REJECTED[req_idx] = nr_out;
        *ACCEPTED_LEN         = ns_out;
        *NEEDS_COMPACT        = incomplete ? 0 : needs_cmp;
    }
    if constexpr (PDL) {
        __syncwarp();
        if (lane == 0) gdc_launch_dependents();
    }
}

// Helper: 1-warp `_vw_update_dink_kernel` (`speculator.py:345`) + optional `es_pingpong[0]` init!
static __device__ __forceinline__ void run_vw_update_dink_1warp(
    int r,
    int64_t slot,
    int32_t ns,
    int32_t nr,
    float temp,
    const int32_t* __restrict__ W_PTR,
    const float*   __restrict__ CPRED_PTR,
    float*         __restrict__ DB_PTR,
    float*         __restrict__ ETOK_PTR,
    float*         __restrict__ EMS_PTR,
    const float*   __restrict__ STEP_PTR,
    int32_t*       __restrict__ STATS_PTR,
    const float*   __restrict__ CAL_PTR,
    int32_t*       __restrict__ ES_SKIP,      // [>=2] if ping-pong or [1]
    float*         __restrict__ ES_SURV,
    float*         __restrict__ ES_TOK,
    float*         __restrict__ ES_BESTU,
    int32_t*       __restrict__ ES_GREEDY,
    int32_t*       __restrict__ ES_SLOT,
    float*         __restrict__ ES_DB,
    float*         __restrict__ ES_LAM,
    int32_t*       __restrict__ ES_SKIP_PP,   // optional 2-slot ping-pong buffer (nullptr if unused)
    float*         __restrict__ ES_SURV_PP,
    float*         __restrict__ ES_TOK_PP,
    float*         __restrict__ ES_BESTU_PP,
    float eta_b,
    float decay_b,
    float b_clip,
    float eta_lam,
    int K,
    int do_stats,
    int fuse_es
) {
    const int lane = threadIdx.x & 31;
    if (slot < 0) return;

    const int32_t w_used = W_PTR[slot];
    const int32_t nd = ns + nr - 1;
    const int32_t acc = ns - 1;
    const bool active = (ns >= 1) && (nd >= 1) && (temp == 0.0f);

    const bool kmask = (lane < K);
    const float cpred = kmask ? CPRED_PTR[slot * K + lane] : 0.7f;
    const int32_t lim = (w_used < nd) ? w_used : nd;
    const bool obs = (lane <= acc) && (lane < lim) && kmask;
    const float hit = (lane < acc) ? 1.0f : 0.0f;

    // Exact 8-element reduction matching Triton `tl.arange(0, 8)`!
    int32_t n_obs = obs ? 1 : 0;
    float g = obs ? __fsub_rn(hit, cpred) : 0.0f;
    float h = obs ? __fmul_rn(cpred, __fsub_rn(1.0f, cpred)) : 0.0f;
    #pragma unroll
    for (int off = 4; off >= 1; off >>= 1) {
        n_obs += __shfl_xor_sync(0xFFFFFFFFu, n_obs, off);
        g = __fadd_rn(g, __shfl_xor_sync(0xFFFFFFFFu, g, off));
        h = __fadd_rn(h, __shfl_xor_sync(0xFFFFFFFFu, h, off));
    }

    if (lane == 0) {
        const float db_old = DB_PTR[slot];
        const float inv_eta = div_full_f32(1.0f, eta_b);
        const float db_obs = __fadd_rn(__fmul_rn(db_old, __fsub_rn(1.0f, decay_b)), div_full_f32(g, __fadd_rn(inv_eta, h)));
        const float dec_none = fmaxf(decay_b, 0.02f);
        const float db_none = __fmul_rn(db_old, __fsub_rn(1.0f, dec_none));
        float db_new = (n_obs > 0) ? db_obs : db_none;
        db_new = fminf(fmaxf(db_new, -b_clip), b_clip);

        int32_t w_clamped = w_used < 0 ? 0 : (w_used > K ? K : w_used);
        const float step_c = STEP_PTR[w_clamped];
        const float etok = ETOK_PTR[slot];
        const float ems = EMS_PTR[slot];
        const float etok_new = __fadd_rn(etok, __fmul_rn(eta_lam, __fsub_rn(static_cast<float>(ns), etok)));
        const float ems_new  = __fadd_rn(ems,  __fmul_rn(eta_lam, __fsub_rn(step_c, ems)));

        const float db_out   = active ? db_new   : db_old;
        const float etok_out = active ? etok_new : etok;
        const float ems_out  = active ? ems_new  : ems;
        DB_PTR[slot]   = db_out;
        ETOK_PTR[slot] = etok_out;
        EMS_PTR[slot]  = ems_out;

        if (fuse_es && r == 0) {
            const float s0 = STEP_PTR[0];
            const float static_lam = CAL_PTR[18];
            const float gamma = CAL_PTR[19];
            const float ems_cl = fmaxf(ems_out, 1.0f);
            const float lam = (static_lam > 0.0f) ? static_lam : __fmul_rn(gamma, div_full_f32(etok_out, ems_cl));
            const float bestu0 = __fsub_rn(1.0f, __fmul_rn(lam, s0));
            const int32_t gr = (temp == 0.0f) ? 1 : 0;
            *ES_SKIP   = 0;
            *ES_SURV   = 1.0f;
            *ES_TOK    = 1.0f;
            *ES_BESTU  = bestu0;
            *ES_GREEDY = gr;
            *ES_SLOT   = static_cast<int32_t>(slot);
            *ES_DB     = db_out;
            *ES_LAM    = lam;
            if (ES_SKIP_PP != nullptr) {
                ES_SKIP_PP[0]  = 0;     ES_SKIP_PP[1]  = 0;
                ES_SURV_PP[0]  = 1.0f;  ES_SURV_PP[1]  = 1.0f;
                ES_TOK_PP[0]   = 1.0f;  ES_TOK_PP[1]   = 1.0f;
                ES_BESTU_PP[0] = bestu0; ES_BESTU_PP[1] = bestu0;
            }
        }
        if (do_stats && active && STATS_PTR != nullptr) {
            atomicAdd(&STATS_PTR[w_clamped], 1);
            atomicAdd(&STATS_PTR[8], 1);
            atomicAdd(&STATS_PTR[9], acc);
        }
    }
}

// C.5: Standalone `dspark_vw_update_dink_v10_kernel`
template <bool PDL>
__global__ __launch_bounds__(32, 8)
void dspark_vw_update_dink_v10_kernel(
    const void*    __restrict__ IDX_PTR,
    int idx_is_i64,
    const int32_t* __restrict__ NUM_SAMPLED,
    const int32_t* __restrict__ NUM_REJECTED,
    const float*   __restrict__ TEMP_PTR,
    const int32_t* __restrict__ W_PTR,
    const float*   __restrict__ CPRED_PTR,
    float*         __restrict__ DB_PTR,
    float*         __restrict__ ETOK_PTR,
    float*         __restrict__ EMS_PTR,
    const float*   __restrict__ STEP_PTR,
    int32_t*       __restrict__ STATS_PTR,
    const float*   __restrict__ CAL_PTR,
    int32_t*       __restrict__ ES_SKIP,
    float*         __restrict__ ES_SURV,
    float*         __restrict__ ES_TOK,
    float*         __restrict__ ES_BESTU,
    int32_t*       __restrict__ ES_GREEDY,
    int32_t*       __restrict__ ES_SLOT,
    float*         __restrict__ ES_DB,
    float*         __restrict__ ES_LAM,
    int32_t*       __restrict__ ES_SKIP_PP,
    float*         __restrict__ ES_SURV_PP,
    float*         __restrict__ ES_TOK_PP,
    float*         __restrict__ ES_BESTU_PP,
    float eta_b,
    float decay_b,
    float b_clip,
    float eta_lam,
    int K,
    int do_stats,
    int fuse_es
) {
    const int r = static_cast<int>(blockIdx.x);
    if constexpr (PDL) gdc_wait();
    const int64_t slot = idx_is_i64
        ? static_cast<const int64_t*>(IDX_PTR)[r]
        : static_cast<int64_t>(static_cast<const int32_t*>(IDX_PTR)[r]);
    run_vw_update_dink_1warp(
        r, slot, NUM_SAMPLED[r], NUM_REJECTED[r], (slot >= 0) ? TEMP_PTR[slot] : 1.0f,
        W_PTR, CPRED_PTR, DB_PTR, ETOK_PTR, EMS_PTR, STEP_PTR, STATS_PTR, CAL_PTR,
        ES_SKIP, ES_SURV, ES_TOK, ES_BESTU, ES_GREEDY, ES_SLOT, ES_DB, ES_LAM,
        ES_SKIP_PP, ES_SURV_PP, ES_TOK_PP, ES_BESTU_PP,
        eta_b, decay_b, b_clip, eta_lam, K, do_stats, fuse_es
    );
    if constexpr (PDL) {
        __syncwarp();
        if (threadIdx.x == 0) gdc_launch_dependents();
    }
}

// Helper for 1 draft KV cache group slot mapping inside `dspark_fused_epilogue_prep_v10_kernel`
static __device__ __forceinline__ void dflash_slot_one_group_dev(
    const int32_t* __restrict__ bt_ptr,
    int64_t bt_stride,
    int64_t* __restrict__ out_ctx_slot_ptr,
    int64_t* __restrict__ out_q_slot_ptr,
    int64_t req_idx,
    int64_t ctx_start,
    int64_t j,
    bool is_ctx,
    bool is_valid_ctx,
    int64_t ctx_pos,
    int64_t query_idx,
    bool is_query,
    int64_t query_pos,
    int64_t max_model_len,
    int64_t blk_size,
    int64_t blk_stride,
    int64_t pad_slot_id
) {
    if (is_ctx) {
        int64_t ctx_slot = pad_slot_id;
        if (is_valid_ctx) {
            int64_t ctx_bn = ctx_pos / blk_size;
            if (ctx_bn >= bt_stride) ctx_bn = bt_stride - 1;
            const int64_t ctx_bid = static_cast<int64_t>(bt_ptr[req_idx * bt_stride + ctx_bn]);
            if (ctx_bid != 0) {
                ctx_slot = ctx_bid * blk_stride + (ctx_pos % blk_size);
            }
        }
        out_ctx_slot_ptr[ctx_start + j] = ctx_slot;
    }
    if (is_query) {
        int64_t q_slot = pad_slot_id;
        if (query_pos < max_model_len) {
            int64_t q_bn = query_pos / blk_size;
            if (q_bn >= bt_stride) q_bn = bt_stride - 1;
            const int64_t q_bid = static_cast<int64_t>(bt_ptr[req_idx * bt_stride + q_bn]);
            if (q_bid != 0) {
                q_slot = q_bid * blk_stride + (query_pos % blk_size);
            }
        }
        out_q_slot_ptr[query_idx] = q_slot;
    }
}

// C.6: `dspark_fused_epilogue_prep_v10_kernel` (`3 -> 1` PDL Launch in Phase 2B `td` Epilogue!)
//   Fuses:
//     (1) `_prepare_dflash_inputs_5g_kernel(NG = 1..5)` (`speculator.py:3494`)
//     (2) `_vw_update_dink_kernel` + `_es_init_dink` (`speculator.py:345`, when `do_dink_update != 0`)
//     (3) `_context_slot_mappings[0..NG-1, pad_c_start:pad_c_end].fill_(PAD_SLOT_ID)` (`speculator.py:1598`)
//   Stabilizes `dtgraph` `td` `cudaGraphExecUpdate` topology across `W < 7` and `W == 7`!
template <int NG, bool PDL>
__global__ __launch_bounds__(128, 4)
void dspark_fused_epilogue_prep_v10_kernel(
    void*          __restrict__ out_input_ids_ptr,
    void*          __restrict__ out_query_positions_ptr,
    uint8_t*       __restrict__ out_is_padding_ptr,
    int32_t*       __restrict__ out_query_start_loc_ptr,
    int32_t*       __restrict__ out_seq_lens_ptr,
    void*          __restrict__ out_context_positions_ptr,
    void*          __restrict__ out_sample_indices_ptr,
    void*          __restrict__ out_sample_pos_ptr,
    void*          __restrict__ out_sample_idx_mapping_ptr,
    float*         __restrict__ out_temperature_ptr,
    void*          __restrict__ out_seeds_ptr,
    int32_t*       __restrict__ out_spec_num_rejected_ptr,
    int64_t*       __restrict__ q_slot0,
    int64_t*       __restrict__ q_slot1,
    int64_t*       __restrict__ q_slot2,
    int64_t*       __restrict__ q_slot3,
    int64_t*       __restrict__ q_slot4,
    int64_t*       __restrict__ c_slot0,
    int64_t*       __restrict__ c_slot1,
    int64_t*       __restrict__ c_slot2,
    int64_t*       __restrict__ c_slot3,
    int64_t*       __restrict__ c_slot4,
    const int32_t* __restrict__ bt0,
    const int32_t* __restrict__ bt1,
    const int32_t* __restrict__ bt2,
    const int32_t* __restrict__ bt3,
    const int32_t* __restrict__ bt4,
    int64_t bts0, int64_t bts1, int64_t bts2, int64_t bts3, int64_t bts4,
    int64_t bsz0, int64_t bsz1, int64_t bsz2, int64_t bsz3, int64_t bsz4,
    int64_t bst0, int64_t bst1, int64_t bst2, int64_t bst3, int64_t bst4,
    const void*    __restrict__ target_positions_ptr,
    const int32_t* __restrict__ target_query_start_loc_ptr,
    const void*    __restrict__ idx_mapping_ptr,
    const void*    __restrict__ last_sampled_ptr,
    const void*    __restrict__ next_prefill_tokens_ptr,
    const int32_t* __restrict__ num_sampled_ptr,
    const int32_t* __restrict__ num_rejected_ptr,
    const float*   __restrict__ temperature_ptr,
    const void*    __restrict__ seeds_ptr,
    int32_t parallel_drafting_token_id,
    int32_t num_query_per_req,
    int32_t num_speculative_steps,
    int32_t max_num_reqs,
    int32_t max_num_tokens,
    int64_t max_model_len,
    int32_t num_reqs,
    int32_t num_pad_programs,
    int64_t pad_slot_id,
    int write_spec_num_rejected,
    int32_t pad_c_start,
    int32_t pad_c_end,
    uint32_t dtype_flags,
    // Fused `_vw_update_dink` + `_es_init_dink` parameters
    int do_dink_update,
    const int32_t* __restrict__ w_ptr,
    const float*   __restrict__ cpred_ptr,
    float*         __restrict__ db_ptr,
    float*         __restrict__ etok_ptr,
    float*         __restrict__ ems_ptr,
    const float*   __restrict__ step_ptr,
    int32_t*       __restrict__ stats_ptr,
    const float*   __restrict__ cal_ptr,
    int32_t*       __restrict__ es_skip,
    float*         __restrict__ es_surv,
    float*         __restrict__ es_tok,
    float*         __restrict__ es_bestu,
    int32_t*       __restrict__ es_greedy,
    int32_t*       __restrict__ es_slot,
    float*         __restrict__ es_db,
    float*         __restrict__ es_lam,
    int32_t*       __restrict__ es_skip_pp,
    float*         __restrict__ es_surv_pp,
    float*         __restrict__ es_tok_pp,
    float*         __restrict__ es_bestu_pp,
    float eta_b,
    float decay_b,
    float b_clip,
    float eta_lam,
    int dink_k,
    int do_stats,
    int fuse_es
) {
    const int32_t req_idx = static_cast<int32_t>(blockIdx.x);
    const int32_t block_idx = static_cast<int32_t>(blockIdx.y);
    const int32_t tid = static_cast<int32_t>(threadIdx.x);
    constexpr int32_t BLOCK_SIZE = 128;

    auto st_int = [](void* p, int64_t idx, int64_t val, bool is_i64) {
        if (is_i64) static_cast<int64_t*>(p)[idx] = val;
        else static_cast<int32_t*>(p)[idx] = static_cast<int32_t>(val);
    };
    auto ld_int = [](const void* p, int64_t idx, bool is_i64) -> int64_t {
        return is_i64 ? static_cast<const int64_t*>(p)[idx]
                      : static_cast<int64_t>(static_cast<const int32_t*>(p)[idx]);
    };

    if constexpr (PDL) gdc_wait();

    if (req_idx >= num_reqs) {
        if (block_idx == 0) {
            const int32_t pad_idx = req_idx - num_reqs;
            const int32_t start = pad_idx * BLOCK_SIZE;
            const int32_t stride = num_pad_programs * BLOCK_SIZE;
            const int32_t last_query_end = num_reqs * num_query_per_req;

            for (int32_t i = num_reqs + start + tid; i < max_num_reqs + 1; i += stride) {
                out_query_start_loc_ptr[i] = last_query_end;
            }
            for (int32_t i = num_reqs + start + tid; i < max_num_reqs; i += stride) {
                out_seq_lens_ptr[i] = 0;
            }
            const int32_t pad_end = max_num_reqs * num_speculative_steps;
            for (int32_t i = num_reqs * num_speculative_steps + start + tid; i < pad_end; i += stride) {
                st_int(out_sample_indices_ptr, i, 0, (dtype_flags & (1u << 3)) != 0u);
                st_int(out_sample_pos_ptr, i, 0, (dtype_flags & (1u << 4)) != 0u);
                st_int(out_sample_idx_mapping_ptr, i, -1, (dtype_flags & (1u << 5)) != 0u);
            }
            for (int32_t i = last_query_end + start + tid; i < max_num_tokens; i += stride) {
                st_int(out_input_ids_ptr, i, 0, (dtype_flags & (1u << 0)) != 0u);
                st_int(out_query_positions_ptr, i, 0, (dtype_flags & (1u << 1)) != 0u);
                out_is_padding_ptr[i] = 1u;
                q_slot0[i] = pad_slot_id;
                if constexpr (NG >= 2) q_slot1[i] = pad_slot_id;
                if constexpr (NG >= 3) q_slot2[i] = pad_slot_id;
                if constexpr (NG >= 4) q_slot3[i] = pad_slot_id;
                if constexpr (NG >= 5) q_slot4[i] = pad_slot_id;
            }
            // Fused `_context_slot_mappings[:, num_target_tokens:num_context].fill_(PAD_SLOT_ID)`!
            for (int32_t i = pad_c_start + start + tid; i < pad_c_end; i += stride) {
                c_slot0[i] = pad_slot_id;
                if constexpr (NG >= 2) c_slot1[i] = pad_slot_id;
                if constexpr (NG >= 3) c_slot2[i] = pad_slot_id;
                if constexpr (NG >= 4) c_slot3[i] = pad_slot_id;
                if constexpr (NG >= 5) c_slot4[i] = pad_slot_id;
            }
        }
        if constexpr (PDL) {
            __syncthreads();
            if (tid == 0) gdc_launch_dependents();
        }
        return;
    }

    const int64_t req_state_idx = ld_int(idx_mapping_ptr, req_idx, (dtype_flags & (1u << 7)) != 0u);
    const int64_t ctx_start = static_cast<int64_t>(target_query_start_loc_ptr[req_idx]);
    const int64_t ctx_end   = static_cast<int64_t>(target_query_start_loc_ptr[req_idx + 1]);
    const int64_t num_ctx   = ctx_end - ctx_start;
    const int32_t nr_val    = num_rejected_ptr[req_idx];
    const int64_t valid_ctx_end = ctx_end - static_cast<int64_t>(nr_val);
    const int64_t num_valid_ctx = valid_ctx_end - ctx_start;
    const int32_t ns_val    = num_sampled_ptr[req_idx];
    const int32_t bonus_token = (ns_val > 0)
        ? static_cast<int32_t>(ld_int(last_sampled_ptr, req_state_idx, (dtype_flags & (1u << 8)) != 0u))
        : static_cast<int32_t>(ld_int(next_prefill_tokens_ptr, req_state_idx, (dtype_flags & (1u << 9)) != 0u));
    const int64_t last_valid_pos = ld_int(target_positions_ptr, valid_ctx_end - 1, (dtype_flags & (1u << 6)) != 0u);
    const int64_t query_base = static_cast<int64_t>(req_idx) * num_query_per_req;

    const int64_t j = static_cast<int64_t>(block_idx) * BLOCK_SIZE + tid;
    const bool is_ctx = (j < num_ctx);
    const bool is_valid_ctx = (j < num_valid_ctx);
    const bool is_query = (j >= num_valid_ctx) && (j < num_valid_ctx + num_query_per_req);
    const int64_t query_off = j - num_valid_ctx;

    int64_t ctx_pos = 0;
    if (is_ctx) {
        if (is_valid_ctx) {
            ctx_pos = ld_int(target_positions_ptr, ctx_start + j, (dtype_flags & (1u << 6)) != 0u);
        }
        st_int(out_context_positions_ptr, ctx_start + j, ctx_pos, (dtype_flags & (1u << 2)) != 0u);
    }

    const int64_t query_pos = last_valid_pos + 1 + query_off;
    const int64_t query_idx = query_base + query_off;
    if (is_query) {
        const int64_t input_id = (query_off == 0) ? bonus_token : parallel_drafting_token_id;
        st_int(out_input_ids_ptr, query_idx, input_id, (dtype_flags & (1u << 0)) != 0u);
        out_is_padding_ptr[query_idx] = 0u;
        const int64_t qpos_clamped = (query_pos < max_model_len - 1) ? query_pos : (max_model_len - 1);
        st_int(out_query_positions_ptr, query_idx, qpos_clamped, (dtype_flags & (1u << 1)) != 0u);
    }

    dflash_slot_one_group_dev(bt0, bts0, c_slot0, q_slot0, req_idx, ctx_start, j, is_ctx, is_valid_ctx, ctx_pos, query_idx, is_query, query_pos, max_model_len, bsz0, bst0, pad_slot_id);
    if constexpr (NG >= 2) dflash_slot_one_group_dev(bt1, bts1, c_slot1, q_slot1, req_idx, ctx_start, j, is_ctx, is_valid_ctx, ctx_pos, query_idx, is_query, query_pos, max_model_len, bsz1, bst1, pad_slot_id);
    if constexpr (NG >= 3) dflash_slot_one_group_dev(bt2, bts2, c_slot2, q_slot2, req_idx, ctx_start, j, is_ctx, is_valid_ctx, ctx_pos, query_idx, is_query, query_pos, max_model_len, bsz2, bst2, pad_slot_id);
    if constexpr (NG >= 4) dflash_slot_one_group_dev(bt3, bts3, c_slot3, q_slot3, req_idx, ctx_start, j, is_ctx, is_valid_ctx, ctx_pos, query_idx, is_query, query_pos, max_model_len, bsz3, bst3, pad_slot_id);
    if constexpr (NG >= 5) dflash_slot_one_group_dev(bt4, bts4, c_slot4, q_slot4, req_idx, ctx_start, j, is_ctx, is_valid_ctx, ctx_pos, query_idx, is_query, query_pos, max_model_len, bsz4, bst4, pad_slot_id);

    if (is_query && query_off >= 1) {
        const int64_t sample_idx = static_cast<int64_t>(req_idx) * num_speculative_steps + (query_off - 1);
        st_int(out_sample_indices_ptr, sample_idx, query_idx, (dtype_flags & (1u << 3)) != 0u);
        st_int(out_sample_pos_ptr, sample_idx, query_pos, (dtype_flags & (1u << 4)) != 0u);
        st_int(out_sample_idx_mapping_ptr, sample_idx, req_state_idx, (dtype_flags & (1u << 5)) != 0u);
    }

    if (block_idx == 0) {
        const float req_temp = temperature_ptr[req_state_idx];
        if (tid == 0) {
            out_query_start_loc_ptr[req_idx] = static_cast<int32_t>(query_base);
            const int64_t seq_l = last_valid_pos + 1 + num_query_per_req;
            out_seq_lens_ptr[req_idx] = static_cast<int32_t>((seq_l < max_model_len) ? seq_l : max_model_len);
            out_temperature_ptr[req_state_idx] = req_temp;
            const bool seed_i32 = (dtype_flags & (1u << 10)) != 0u;
            st_int(out_seeds_ptr, req_state_idx, ld_int(seeds_ptr, req_state_idx, !seed_i32), !seed_i32);
            if (write_spec_num_rejected) {
                out_spec_num_rejected_ptr[req_idx] = nr_val;
            }
        }
        // Fused `_vw_update_dink` + `_es_init_dink` in warp 0!
        if (do_dink_update && tid < 32) {
            run_vw_update_dink_1warp(
                req_idx, req_state_idx, ns_val, nr_val, req_temp,
                w_ptr, cpred_ptr, db_ptr, etok_ptr, ems_ptr, step_ptr, stats_ptr, cal_ptr,
                es_skip, es_surv, es_tok, es_bestu, es_greedy, es_slot, es_db, es_lam,
                es_skip_pp, es_surv_pp, es_tok_pp, es_bestu_pp,
                eta_b, decay_b, b_clip, eta_lam, dink_k, do_stats, fuse_es
            );
        }
    }

    if constexpr (PDL) {
        __syncthreads();
        if (tid == 0) gdc_launch_dependents();
    }
}

}  // namespace

// ============================================================================
// Exported `extern "C"` Host ABI Entrypoints (`libdspark_markov_v7.so`)
// ============================================================================

extern "C" {

int dspark_embed_block_ln_v7_fwd(
    const void* ids, int ids_is_i64,
    const void* emb_w, const void* emb_nw, const void* mask_e, const void* ln_w,
    void* res, void* out,
    int num_rows, int hidden, int64_t stride_emb, int64_t full_v,
    int block_size, float emb_eps, float eps,
    int spt, int use_pdl, uint64_t stream_u64
) {
    if (num_rows <= 0) return 0;
    if (hidden != HIDDEN_DS) return -1;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(stream_u64));
    dim3 grid(static_cast<unsigned int>(num_rows));
    dim3 block(256);
    const int32_t* i32 = static_cast<const int32_t*>(ids);
    const int64_t* i64 = static_cast<const int64_t*>(ids);
    const uint16_t* ew = static_cast<const uint16_t*>(emb_w);
    const uint16_t* en = static_cast<const uint16_t*>(emb_nw);
    const uint16_t* me = static_cast<const uint16_t*>(mask_e);
    const uint16_t* lw = static_cast<const uint16_t*>(ln_w);
    uint16_t* rp = static_cast<uint16_t*>(res);
    uint16_t* op = static_cast<uint16_t*>(out);
    cudaError_t err;
    if (spt == 8) {
        err = use_pdl
            ? launch_kernel_ex(true,  grid, block, 0, stream, dspark_embed_block_ln_v10_kernel<8, true>,  i32, i64, ids_is_i64, ew, en, me, lw, rp, op, stride_emb, full_v, block_size, emb_eps, eps)
            : launch_kernel_ex(false, grid, block, 0, stream, dspark_embed_block_ln_v10_kernel<8, false>, i32, i64, ids_is_i64, ew, en, me, lw, rp, op, stride_emb, full_v, block_size, emb_eps, eps);
    } else {
        err = use_pdl
            ? launch_kernel_ex(true,  grid, block, 0, stream, dspark_embed_block_ln_v10_kernel<1, true>,  i32, i64, ids_is_i64, ew, en, me, lw, rp, op, stride_emb, full_v, block_size, emb_eps, eps)
            : launch_kernel_ex(false, grid, block, 0, stream, dspark_embed_block_ln_v10_kernel<1, false>, i32, i64, ids_is_i64, ew, en, me, lw, rp, op, stride_emb, full_v, block_size, emb_eps, eps);
    }
    return (err == cudaSuccess) ? 0 : static_cast<int>(err);
}

int dspark_mix_rmsnorm_v7_fwd(
    const void* x, const void* alpha, int alpha_is_bf16,
    const void* weight, void* out,
    int rows, int taps, int hidden,
    int64_t row_stride, int64_t tap_stride, int64_t out_row_stride,
    float eps, int spt, int fma_mode, int use_pdl, uint64_t stream_u64
) {
    if (rows <= 0) return 0;
    if (taps != TAPS_DS || hidden != HIDDEN_DS) return -1;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(stream_u64));
    dim3 grid(static_cast<unsigned int>(rows));
    dim3 block(256);
    const uint16_t* xp = static_cast<const uint16_t*>(x);
    const float* af = static_cast<const float*>(alpha);
    const uint16_t* ab = static_cast<const uint16_t*>(alpha);
    const uint16_t* wp = static_cast<const uint16_t*>(weight);
    uint16_t* op = static_cast<uint16_t*>(out);
    cudaError_t err;
    #define DISPATCH_MIX(SPT_V, FMA_V, PDL_V) \
        launch_kernel_ex(PDL_V, grid, block, 0, stream, dspark_mix_rmsnorm_v10_kernel<SPT_V, FMA_V, false, PDL_V>, \
            xp, nullptr, nullptr, af, ab, alpha_is_bf16, wp, op, row_stride, tap_stride, out_row_stride, eps)
    if (spt == 8) {
        if (fma_mode == 0) err = use_pdl ? DISPATCH_MIX(8, 0, true) : DISPATCH_MIX(8, 0, false);
        else               err = use_pdl ? DISPATCH_MIX(8, 1, true) : DISPATCH_MIX(8, 1, false);
    } else {
        if (fma_mode == 0) err = use_pdl ? DISPATCH_MIX(1, 0, true) : DISPATCH_MIX(1, 0, false);
        else               err = use_pdl ? DISPATCH_MIX(1, 1, true) : DISPATCH_MIX(1, 1, false);
    }
    #undef DISPATCH_MIX
    return (err == cudaSuccess) ? 0 : static_cast<int>(err);
}

int dspark_context_anchor_mix_rmsnorm_v7_fwd(
    const void* hid, const void* qsl, const void* num_rej,
    const void* alpha, int alpha_is_bf16,
    const void* weight, void* out,
    int num_reqs, int taps, int hidden,
    int64_t row_stride, int64_t tap_stride, int64_t out_row_stride,
    float eps, int spt, int fma_mode, int use_pdl, uint64_t stream_u64
) {
    if (num_reqs <= 0) return 0;
    if (taps != TAPS_DS || hidden != HIDDEN_DS) return -1;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(stream_u64));
    dim3 grid(static_cast<unsigned int>(num_reqs));
    dim3 block(256);
    const uint16_t* xp = static_cast<const uint16_t*>(hid);
    const int32_t* qp = static_cast<const int32_t*>(qsl);
    const int32_t* nr = static_cast<const int32_t*>(num_rej);
    const float* af = static_cast<const float*>(alpha);
    const uint16_t* ab = static_cast<const uint16_t*>(alpha);
    const uint16_t* wp = static_cast<const uint16_t*>(weight);
    uint16_t* op = static_cast<uint16_t*>(out);
    cudaError_t err;
    #define DISPATCH_ANC(SPT_V, FMA_V, PDL_V) \
        launch_kernel_ex(PDL_V, grid, block, 0, stream, dspark_mix_rmsnorm_v10_kernel<SPT_V, FMA_V, true, PDL_V>, \
            xp, qp, nr, af, ab, alpha_is_bf16, wp, op, row_stride, tap_stride, out_row_stride, eps)
    if (spt == 8) {
        if (fma_mode == 0) err = use_pdl ? DISPATCH_ANC(8, 0, true) : DISPATCH_ANC(8, 0, false);
        else               err = use_pdl ? DISPATCH_ANC(8, 1, true) : DISPATCH_ANC(8, 1, false);
    } else {
        if (fma_mode == 0) err = use_pdl ? DISPATCH_ANC(1, 0, true) : DISPATCH_ANC(1, 0, false);
        else               err = use_pdl ? DISPATCH_ANC(1, 1, true) : DISPATCH_ANC(1, 1, false);
    }
    #undef DISPATCH_ANC
    return (err == cudaSuccess) ? 0 : static_cast<int>(err);
}

int dspark_mix_rmsnorm_5layer_v7_fwd(
    const void* x, const void* alpha_5x8, int alpha_is_bf16,
    const void* w0, const void* w1, const void* w2, const void* w3, const void* w4,
    const void* slot_map, int64_t slot_stride,
    void* out_5l, int64_t layer_out_stride,
    int rows, int taps, int hidden,
    int64_t row_stride, int64_t tap_stride,
    float eps, int spt, int fma_mode, int use_pdl, uint64_t stream_u64
) {
    if (rows <= 0) return 0;
    if (taps != TAPS_DS || hidden != HIDDEN_DS) return -1;
    if (spt == 8 && (((row_stride | tap_stride) & 7) != 0)) spt = 1;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(stream_u64));
    dim3 grid(3u, static_cast<unsigned int>(rows));
    dim3 block(256);
    const uint16_t* xp = static_cast<const uint16_t*>(x);
    const float* af = static_cast<const float*>(alpha_5x8);
    const uint16_t* ab = static_cast<const uint16_t*>(alpha_5x8);
    const int64_t* sm = static_cast<const int64_t*>(slot_map);
    uint16_t* op = static_cast<uint16_t*>(out_5l);
    cudaError_t err;
    #define DISPATCH_MIX5(SPT_V, FMA_V, PDL_V) \
        launch_kernel_ex(PDL_V, grid, block, 0, stream, dspark_mix_rmsnorm_5layer_v10_kernel<SPT_V, FMA_V, PDL_V>, \
            xp, af, ab, alpha_is_bf16, \
            static_cast<const uint16_t*>(w0), static_cast<const uint16_t*>(w1), \
            static_cast<const uint16_t*>(w2), static_cast<const uint16_t*>(w3), \
            static_cast<const uint16_t*>(w4), sm, slot_stride, op, layer_out_stride, row_stride, tap_stride, eps)
    if (spt == 8) {
        if (fma_mode == 0) err = use_pdl ? DISPATCH_MIX5(8, 0, true) : DISPATCH_MIX5(8, 0, false);
        else               err = use_pdl ? DISPATCH_MIX5(8, 1, true) : DISPATCH_MIX5(8, 1, false);
    } else {
        if (fma_mode == 0) err = use_pdl ? DISPATCH_MIX5(1, 0, true) : DISPATCH_MIX5(1, 0, false);
        else               err = use_pdl ? DISPATCH_MIX5(1, 1, true) : DISPATCH_MIX5(1, 1, false);
    }
    #undef DISPATCH_MIX5
    return (err == cudaSuccess) ? 0 : static_cast<int>(err);
}

int dspark_qk_norm_rope_write_kv_v7_fwd(
    void* x, const void* q_weight, const void* k_weight, const void* cos_sin,
    const void* pos, int pos_is_i32,
    void* kc, void* vc, const void* slot_map,
    int num_tokens, int nhq, int nhk, int head_dim,
    int64_t row_stride, int64_t q_size,
    int64_t sk_b, int64_t sk_s, int64_t sk_h,
    int64_t sv_b, int64_t sv_s, int64_t sv_h,
    int64_t slot_stride, int64_t kv_block_size,
    float eps, int write_kv, int norm, int otf_rope,
    int sqrt_mode, int rope_fma_mode, int use_pdl, uint64_t stream_u64
) {
    if (num_tokens <= 0) return 0;
    if (head_dim != 128 || (nhq + nhk) <= 0 || (nhq + nhk) > 8) return -1;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(stream_u64));
    uint16_t* xp = static_cast<uint16_t*>(x);
    const uint16_t* qw = static_cast<const uint16_t*>(q_weight);
    const uint16_t* kw = static_cast<const uint16_t*>(k_weight);
    const float* cs = static_cast<const float*>(cos_sin);
    const int64_t* p64 = static_cast<const int64_t*>(pos);
    const int32_t* p32 = static_cast<const int32_t*>(pos);
    uint16_t* kcp = static_cast<uint16_t*>(kc);
    uint16_t* vcp = static_cast<uint16_t*>(vc);
    const int64_t* sm = static_cast<const int64_t*>(slot_map);

    if (write_kv && norm && nhq == 0 && nhk == 1) {
        dim3 grid(static_cast<unsigned int>(num_tokens));
        dim3 block(32);
        cudaError_t err;
        #define DISPATCH_CTX1W(OTF_V, RF_V, PDL_V) \
            launch_kernel_ex(PDL_V, grid, block, 0, stream, dspark_ctx_kv_1warp_v10_kernel<OTF_V, RF_V, PDL_V>, \
                xp, kw, cs, p64, p32, pos_is_i32, kcp, vcp, sm, row_stride, sk_b, sk_s, sv_b, sv_s, slot_stride, kv_block_size, eps, sqrt_mode)
        if (otf_rope) {
            if (rope_fma_mode == 0)      err = use_pdl ? DISPATCH_CTX1W(true, 0, true) : DISPATCH_CTX1W(true, 0, false);
            else if (rope_fma_mode == 1) err = use_pdl ? DISPATCH_CTX1W(true, 1, true) : DISPATCH_CTX1W(true, 1, false);
            else                         err = use_pdl ? DISPATCH_CTX1W(true, 2, true) : DISPATCH_CTX1W(true, 2, false);
        } else {
            if (rope_fma_mode == 0)      err = use_pdl ? DISPATCH_CTX1W(false, 0, true) : DISPATCH_CTX1W(false, 0, false);
            else if (rope_fma_mode == 1) err = use_pdl ? DISPATCH_CTX1W(false, 1, true) : DISPATCH_CTX1W(false, 1, false);
            else                         err = use_pdl ? DISPATCH_CTX1W(false, 2, true) : DISPATCH_CTX1W(false, 2, false);
        }
        #undef DISPATCH_CTX1W
        return (err == cudaSuccess) ? 0 : static_cast<int>(err);
    }

    dim3 grid(static_cast<unsigned int>(num_tokens));
    dim3 block(256);
    cudaError_t err;
    #define DISPATCH_QKM(WKV_V, NRM_V, OTF_V, RF_V, PDL_V) \
        launch_kernel_ex(PDL_V, grid, block, 0, stream, dspark_qk_norm_rope_multi_v10_kernel<WKV_V, NRM_V, OTF_V, RF_V, PDL_V>, \
            xp, qw, kw, cs, p64, p32, pos_is_i32, kcp, vcp, sm, row_stride, q_size, sk_b, sk_s, sk_h, sv_b, sv_s, sv_h, slot_stride, kv_block_size, nhq, nhk, eps, sqrt_mode)
    #define DISPATCH_QKM_RF(WKV_V, NRM_V, OTF_V) \
        ((rope_fma_mode == 0) ? (use_pdl ? DISPATCH_QKM(WKV_V, NRM_V, OTF_V, 0, true) : DISPATCH_QKM(WKV_V, NRM_V, OTF_V, 0, false)) : \
         ((rope_fma_mode == 1) ? (use_pdl ? DISPATCH_QKM(WKV_V, NRM_V, OTF_V, 1, true) : DISPATCH_QKM(WKV_V, NRM_V, OTF_V, 1, false)) : \
                                 (use_pdl ? DISPATCH_QKM(WKV_V, NRM_V, OTF_V, 2, true) : DISPATCH_QKM(WKV_V, NRM_V, OTF_V, 2, false))))
    if (write_kv) {
        err = otf_rope ? DISPATCH_QKM_RF(true, true, true) : DISPATCH_QKM_RF(true, true, false);
    } else if (norm) {
        err = otf_rope ? DISPATCH_QKM_RF(false, true, true) : DISPATCH_QKM_RF(false, true, false);
    } else {
        err = otf_rope ? DISPATCH_QKM_RF(false, false, true) : DISPATCH_QKM_RF(false, false, false);
    }
    #undef DISPATCH_QKM_RF
    #undef DISPATCH_QKM
    return (err == cudaSuccess) ? 0 : static_cast<int>(err);
}

int dspark_silu_mul_v7_fwd(
    const void* x, void* y, int num_tokens, int n, int64_t stride_x, int use_pdl, uint64_t stream_u64
) {
    if (num_tokens <= 0 || n <= 0) return 0;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(stream_u64));
    dim3 grid(static_cast<unsigned int>(num_tokens), static_cast<unsigned int>((n + 1023) / 1024));
    dim3 block(256);
    cudaError_t err = use_pdl
        ? launch_kernel_ex(true,  grid, block, 0, stream, dspark_silu_mul_v10_kernel<true>,  static_cast<const uint16_t*>(x), static_cast<uint16_t*>(y), stride_x, n)
        : launch_kernel_ex(false, grid, block, 0, stream, dspark_silu_mul_v10_kernel<false>, static_cast<const uint16_t*>(x), static_cast<uint16_t*>(y), stride_x, n);
    return (err == cudaSuccess) ? 0 : static_cast<int>(err);
}

int dspark_quant_fp8_rowwise_v7_fwd(
    const void* x, void* y_fp8, void* s_out,
    int M, int M_pad, int K, int64_t stride_xm, int64_t stride_ym,
    int use_pdl, uint64_t stream_u64
) {
    if (M_pad <= 0 || K <= 0 || (K & 1) != 0) return -1;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(stream_u64));
    dim3 grid(static_cast<unsigned int>(M_pad));
    dim3 block(256);
    cudaError_t err = use_pdl
        ? launch_kernel_ex(true,  grid, block, 0, stream, dspark_quant_fp8_rowwise_v10_kernel<true>,  static_cast<const uint16_t*>(x), static_cast<uint8_t*>(y_fp8), static_cast<float*>(s_out), M, K, stride_xm, stride_ym)
        : launch_kernel_ex(false, grid, block, 0, stream, dspark_quant_fp8_rowwise_v10_kernel<false>, static_cast<const uint16_t*>(x), static_cast<uint8_t*>(y_fp8), static_cast<float*>(s_out), M, K, stride_xm, stride_ym);
    return (err == cudaSuccess) ? 0 : static_cast<int>(err);
}

int dspark_silu_mul_quant_fp8_v7_fwd(
    const void* x, void* y_fp8, void* s_out,
    int M, int M_pad, int N, int64_t stride_xm, int64_t stride_ym,
    int use_pdl, uint64_t stream_u64
) {
    if (M_pad <= 0 || N <= 0 || (N & 1) != 0) return -1;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(stream_u64));
    dim3 grid(static_cast<unsigned int>(M_pad));
    dim3 block(256);
    cudaError_t err = use_pdl
        ? launch_kernel_ex(true,  grid, block, 0, stream, dspark_silu_mul_quant_fp8_v10_kernel<true>,  static_cast<const uint16_t*>(x), static_cast<uint8_t*>(y_fp8), static_cast<float*>(s_out), M, N, stride_xm, stride_ym)
        : launch_kernel_ex(false, grid, block, 0, stream, dspark_silu_mul_quant_fp8_v10_kernel<false>, static_cast<const uint16_t*>(x), static_cast<uint8_t*>(y_fp8), static_cast<float*>(s_out), M, N, stride_xm, stride_ym);
    return (err == cudaSuccess) ? 0 : static_cast<int>(err);
}

int dspark_add_softcap_v7_fwd(
    const void* a, const void* b, void* out,
    int rows, int n, int64_t stride_a, int64_t stride_b, float softcap,
    int use_pdl, uint64_t stream_u64
) {
    if (rows <= 0 || n <= 0) return 0;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(stream_u64));
    dim3 grid(static_cast<unsigned int>((n + 1023) / 1024), static_cast<unsigned int>(rows));
    dim3 block(256);
    cudaError_t err = use_pdl
        ? launch_kernel_ex(true,  grid, block, 0, stream, dspark_add_softcap_v10_kernel<true>,  static_cast<const uint16_t*>(a), static_cast<const uint16_t*>(b), static_cast<uint16_t*>(out), stride_a, stride_b, softcap, n)
        : launch_kernel_ex(false, grid, block, 0, stream, dspark_add_softcap_v10_kernel<false>, static_cast<const uint16_t*>(a), static_cast<const uint16_t*>(b), static_cast<uint16_t*>(out), stride_a, stride_b, softcap, n);
    return (err == cudaSuccess) ? 0 : static_cast<int>(err);
}

int dspark_markov_gate_v7_fwd(
    const void* raw, const void* state, void* out_state, void* latent,
    int rows, int rank, int use_pdl, uint64_t stream_u64
) {
    if (rows <= 0) return 0;
    if (rank != RANK_MK) return -1;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(stream_u64));
    dim3 grid(static_cast<unsigned int>(rows));
    dim3 block(256);
    cudaError_t err = use_pdl
        ? launch_kernel_ex(true,  grid, block, 0, stream, dspark_markov_gate_v10_kernel<true>,  static_cast<const float*>(raw), static_cast<const uint16_t*>(state), static_cast<uint16_t*>(out_state), static_cast<uint16_t*>(latent))
        : launch_kernel_ex(false, grid, block, 0, stream, dspark_markov_gate_v10_kernel<false>, static_cast<const float*>(raw), static_cast<const uint16_t*>(state), static_cast<uint16_t*>(out_state), static_cast<uint16_t*>(latent));
    return (err == cudaSuccess) ? 0 : static_cast<int>(err);
}

int dspark_markov_rnn_step_v7_fwd(
    const void* state, const void* prev, int prev_is_i32,
    const void* ctx, int64_t ctx_row_stride,
    const void* w1, const void* rnn_w,
    void* out_state, void* latent, const void* skip_ptr,
    int rows, int rank, int64_t prev_stride,
    int init_zero, int has_skip, int use_tc, int use_pdl, uint64_t stream_u64
) {
    if (rows <= 0) return 0;
    if (rank != RANK_MK) return -1;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(stream_u64));
    const uint16_t* st = static_cast<const uint16_t*>(state);
    const int64_t* p64 = static_cast<const int64_t*>(prev);
    const int32_t* p32 = static_cast<const int32_t*>(prev);
    const float* cx = static_cast<const float*>(ctx);
    const uint16_t* w1p = static_cast<const uint16_t*>(w1);
    const uint16_t* rw = static_cast<const uint16_t*>(rnn_w);
    uint16_t* os = static_cast<uint16_t*>(out_state);
    uint16_t* lt = static_cast<uint16_t*>(latent);
    const int32_t* sk = static_cast<const int32_t*>(skip_ptr);

    if (rows == 1) {
        dim3 grid(1u, 48u);
        dim3 block(128u);
        cudaError_t err;
        #define DISPATCH_RNN1(IZ_V, SK_V, PDL_V) \
            launch_kernel_ex(PDL_V, grid, block, 0, stream, dspark_markov_rnn_step_r1_48cta_v10_kernel<IZ_V, SK_V, PDL_V>, \
                st, p64, p32, prev_is_i32, cx, w1p, rw, os, lt, sk)
        if (init_zero) {
            err = has_skip ? (use_pdl ? DISPATCH_RNN1(true, true, true)  : DISPATCH_RNN1(true, true, false))
                           : (use_pdl ? DISPATCH_RNN1(true, false, true) : DISPATCH_RNN1(true, false, false));
        } else {
            err = has_skip ? (use_pdl ? DISPATCH_RNN1(false, true, true)  : DISPATCH_RNN1(false, true, false))
                           : (use_pdl ? DISPATCH_RNN1(false, false, true) : DISPATCH_RNN1(false, false, false));
        }
        #undef DISPATCH_RNN1
        return (err == cudaSuccess) ? 0 : static_cast<int>(err);
    }

    if (!use_tc) return -2;
    dim3 grid(static_cast<unsigned int>((rows + 15) / 16), 48u);
    dim3 block(32u);
    cudaError_t err;
    #define DISPATCH_RNN_TC(IZ_V, SK_V, PDL_V) \
        launch_kernel_ex(PDL_V, grid, block, 0, stream, dspark_markov_rnn_step_tc_v10_kernel<IZ_V, SK_V, PDL_V>, \
            st, p64, p32, prev_is_i32, cx, ctx_row_stride, w1p, rw, os, lt, sk, rows, prev_stride)
    if (init_zero) {
        err = has_skip ? (use_pdl ? DISPATCH_RNN_TC(true, true, true)  : DISPATCH_RNN_TC(true, true, false))
                       : (use_pdl ? DISPATCH_RNN_TC(true, false, true) : DISPATCH_RNN_TC(true, false, false));
    } else {
        err = has_skip ? (use_pdl ? DISPATCH_RNN_TC(false, true, true)  : DISPATCH_RNN_TC(false, true, false))
                       : (use_pdl ? DISPATCH_RNN_TC(false, false, true) : DISPATCH_RNN_TC(false, false, false));
    }
    #undef DISPATCH_RNN_TC
    return (err == cudaSuccess) ? 0 : static_cast<int>(err);
}

int dspark_select_and_rnn_step_v7_fwd(
    const void* g, int rows, int S, int NS,
    void* out_tok_i, int64_t out_tok_stride,
    void* conf_i, int64_t conf_stride, int with_conf,
    int es_mode, int step_idx, int K,
    const void* skip_in, const void* surv_in, const void* tok_in, const void* best_in,
    void* skip_out, void* surv_out, void* tok_out, void* best_out,
    const void* greedy_ptr, const void* slot_ptr, const void* db_dev_ptr, const void* lam_dev_ptr,
    const void* ema_ptr, const void* cal_ptr, const void* step_ptr, float prior_logit,
    const void* state_i, const void* ctx_next, int64_t ctx_row_stride,
    const void* w1, const void* rnn_w,
    void* out_state_next, void* latent_next,
    int exp_mode, int use_pdl, uint64_t stream_u64
) {
    if (rows <= 0) return 0;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(stream_u64));
    if (rows == 1) {
        dim3 grid(1u, 48u);
        dim3 block(128u);
        cudaError_t err = use_pdl
            ? launch_kernel_ex(true, grid, block, 0, stream, dspark_select_and_rnn_step_r1_v10_kernel<true>,
                static_cast<const float*>(g), S, NS, static_cast<int64_t*>(out_tok_i), static_cast<float*>(conf_i),
                with_conf, es_mode, step_idx, K,
                static_cast<const int32_t*>(skip_in), static_cast<const float*>(surv_in), static_cast<const float*>(tok_in), static_cast<const float*>(best_in),
                static_cast<int32_t*>(skip_out), static_cast<float*>(surv_out), static_cast<float*>(tok_out), static_cast<float*>(best_out),
                static_cast<const int32_t*>(greedy_ptr), static_cast<const int32_t*>(slot_ptr), static_cast<const float*>(db_dev_ptr), static_cast<const float*>(lam_dev_ptr),
                static_cast<const float*>(ema_ptr), static_cast<const float*>(cal_ptr), static_cast<const float*>(step_ptr), prior_logit,
                static_cast<const uint16_t*>(state_i), static_cast<const float*>(ctx_next),
                static_cast<const uint16_t*>(w1), static_cast<const uint16_t*>(rnn_w),
                static_cast<uint16_t*>(out_state_next), static_cast<uint16_t*>(latent_next), exp_mode)
            : launch_kernel_ex(false, grid, block, 0, stream, dspark_select_and_rnn_step_r1_v10_kernel<false>,
                static_cast<const float*>(g), S, NS, static_cast<int64_t*>(out_tok_i), static_cast<float*>(conf_i),
                with_conf, es_mode, step_idx, K,
                static_cast<const int32_t*>(skip_in), static_cast<const float*>(surv_in), static_cast<const float*>(tok_in), static_cast<const float*>(best_in),
                static_cast<int32_t*>(skip_out), static_cast<float*>(surv_out), static_cast<float*>(tok_out), static_cast<float*>(best_out),
                static_cast<const int32_t*>(greedy_ptr), static_cast<const int32_t*>(slot_ptr), static_cast<const float*>(db_dev_ptr), static_cast<const float*>(lam_dev_ptr),
                static_cast<const float*>(ema_ptr), static_cast<const float*>(cal_ptr), static_cast<const float*>(step_ptr), prior_logit,
                static_cast<const uint16_t*>(state_i), static_cast<const float*>(ctx_next),
                static_cast<const uint16_t*>(w1), static_cast<const uint16_t*>(rnn_w),
                static_cast<uint16_t*>(out_state_next), static_cast<uint16_t*>(latent_next), exp_mode);
        return (err == cudaSuccess) ? 0 : static_cast<int>(err);
    }
    if (es_mode != 0 && !(es_mode == 5 && rows >= 2 && rows <= 8)) return -2;
    dim3 grid(static_cast<unsigned int>((rows + 15) / 16), 48u);
    dim3 block(32u);
    cudaError_t err;
    #define DISPATCH_SR_TC(BES_V, PDL_V) \
        launch_kernel_ex(PDL_V, grid, block, 0, stream, dspark_select_and_rnn_step_tc_v10_kernel<BES_V, PDL_V>, \
            static_cast<const float*>(g), S, NS, static_cast<int64_t*>(out_tok_i), out_tok_stride, \
            static_cast<float*>(conf_i), conf_stride, with_conf, \
            step_idx, K, \
            static_cast<const int32_t*>(skip_in), static_cast<const float*>(surv_in), \
            static_cast<int32_t*>(skip_out), static_cast<float*>(surv_out), \
            static_cast<const int32_t*>(greedy_ptr), static_cast<const float*>(db_dev_ptr), \
            static_cast<const float*>(cal_ptr), prior_logit, \
            static_cast<const uint16_t*>(state_i), static_cast<const float*>(ctx_next), ctx_row_stride, \
            static_cast<const uint16_t*>(w1), static_cast<const uint16_t*>(rnn_w), \
            static_cast<uint16_t*>(out_state_next), static_cast<uint16_t*>(latent_next), rows, exp_mode)
    if (es_mode == 5) {
        err = use_pdl ? DISPATCH_SR_TC(true, true) : DISPATCH_SR_TC(true, false);
    } else {
        err = use_pdl ? DISPATCH_SR_TC(false, true) : DISPATCH_SR_TC(false, false);
    }
    #undef DISPATCH_SR_TC
    return (err == cudaSuccess) ? 0 : static_cast<int>(err);
}

int dspark_fold_argmax_v7_fwd(
    const void* base, const void* lat, const void* fw, const void* fs, const void* fb,
    void* rec, void* rec_out, void* sem, const void* skip_ptr, const void* id_map,
    int rows, int rank, int slices, int ch, int64_t stride_base,
    float softcap, int valid, int offset,
    int use_fp8, int has_skip, int fuse_reduce, int has_id_map,
    int top2, int fma_mode, int exp_mode, int use_pdl, uint64_t stream_u64
) {
    if (rows <= 0 || slices <= 0) return 0;
    if (rank != RANK_MK) return -1;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(stream_u64));
    const uint16_t* bp = static_cast<const uint16_t*>(base);
    const uint16_t* lp = static_cast<const uint16_t*>(lat);
    const float* fsp = static_cast<const float*>(fs);
    const uint16_t* fbp = static_cast<const uint16_t*>(fb);
    float* rp = static_cast<float*>(rec);
    float* rop = static_cast<float*>(rec_out);
    uint32_t* sp = static_cast<uint32_t*>(sem);
    const int32_t* sk = static_cast<const int32_t*>(skip_ptr);
    const int32_t* im = static_cast<const int32_t*>(id_map);

    if (rows == 1 || top2 != 0) {
        dim3 grid(static_cast<unsigned int>(rows), static_cast<unsigned int>(slices));
        dim3 block(128u);
        cudaError_t err;
        #define LAUNCH_FR1(T2_V, FP8_V, SK_V, FR_V, IM_V, FMA_V, PDL_V) \
            launch_kernel_ex(PDL_V, grid, block, 0, stream, dspark_fold_argmax_r1_v10_kernel<T2_V, FP8_V, SK_V, FR_V, IM_V, FMA_V, PDL_V>, \
                bp, lp, fw, fsp, fbp, rp, rop, sp, sk, im, stride_base, softcap, valid, offset, ch, exp_mode)
        #define LAUNCH_FR1_PDL(T2_V, FP8_V, SK_V, FR_V, IM_V) \
            ((fma_mode == 0) ? (use_pdl ? LAUNCH_FR1(T2_V, FP8_V, SK_V, FR_V, IM_V, 0, true) : LAUNCH_FR1(T2_V, FP8_V, SK_V, FR_V, IM_V, 0, false)) \
                             : (use_pdl ? LAUNCH_FR1(T2_V, FP8_V, SK_V, FR_V, IM_V, 1, true) : LAUNCH_FR1(T2_V, FP8_V, SK_V, FR_V, IM_V, 1, false)))
        if (top2) {
            if (use_fp8) {
                err = has_id_map ? (fuse_reduce ? LAUNCH_FR1_PDL(true, true, false, true, true)   : LAUNCH_FR1_PDL(true, true, false, false, true))
                                 : (fuse_reduce ? LAUNCH_FR1_PDL(true, true, false, true, false)  : LAUNCH_FR1_PDL(true, true, false, false, false));
            } else {
                err = has_id_map ? (fuse_reduce ? LAUNCH_FR1_PDL(true, false, false, true, true)  : LAUNCH_FR1_PDL(true, false, false, false, true))
                                 : (fuse_reduce ? LAUNCH_FR1_PDL(true, false, false, true, false) : LAUNCH_FR1_PDL(true, false, false, false, false));
            }
        } else if (use_fp8) {
            if (has_id_map) {
                err = has_skip ? (fuse_reduce ? LAUNCH_FR1_PDL(false, true, true, true, true)  : LAUNCH_FR1_PDL(false, true, true, false, true))
                               : (fuse_reduce ? LAUNCH_FR1_PDL(false, true, false, true, true) : LAUNCH_FR1_PDL(false, true, false, false, true));
            } else {
                err = has_skip ? (fuse_reduce ? LAUNCH_FR1_PDL(false, true, true, true, false)  : LAUNCH_FR1_PDL(false, true, true, false, false))
                               : (fuse_reduce ? LAUNCH_FR1_PDL(false, true, false, true, false) : LAUNCH_FR1_PDL(false, true, false, false, false));
            }
        } else {
            if (has_id_map) {
                err = has_skip ? (fuse_reduce ? LAUNCH_FR1_PDL(false, false, true, true, true)  : LAUNCH_FR1_PDL(false, false, true, false, true))
                               : (fuse_reduce ? LAUNCH_FR1_PDL(false, false, false, true, true) : LAUNCH_FR1_PDL(false, false, false, false, true));
            } else {
                err = has_skip ? (fuse_reduce ? LAUNCH_FR1_PDL(false, false, true, true, false)  : LAUNCH_FR1_PDL(false, false, true, false, false))
                               : (fuse_reduce ? LAUNCH_FR1_PDL(false, false, false, true, false) : LAUNCH_FR1_PDL(false, false, false, false, false));
            }
        }
        #undef LAUNCH_FR1_PDL
        #undef LAUNCH_FR1
        return (err == cudaSuccess) ? 0 : static_cast<int>(err);
    }

    const int bm = (rows <= 8) ? 8 : ((rows <= 16) ? 16 : 32);
    dim3 grid(static_cast<unsigned int>((rows + bm - 1) / bm), static_cast<unsigned int>(slices));
    dim3 block(128u);
    cudaError_t err;
    const int32_t* sk_tc = (has_skip != 0) ? sk : nullptr;
    #define LAUNCH_FTC(BM_V, FP8_V, FR_V, IM_V, FMA_V, PDL_V) \
        launch_kernel_ex(PDL_V, grid, block, 0, stream, dspark_fold_argmax_tc_v10_kernel<BM_V, FP8_V, FR_V, IM_V, FMA_V, PDL_V>, \
            bp, lp, fw, fsp, fbp, rp, rop, sp, sk_tc, im, stride_base, rows, softcap, valid, offset, ch, exp_mode)
    #define LAUNCH_FTC_PDL(BM_V, FP8_V, FR_V, IM_V) \
        ((fma_mode == 0) ? (use_pdl ? LAUNCH_FTC(BM_V, FP8_V, FR_V, IM_V, 0, true) : LAUNCH_FTC(BM_V, FP8_V, FR_V, IM_V, 0, false)) \
                         : (use_pdl ? LAUNCH_FTC(BM_V, FP8_V, FR_V, IM_V, 1, true) : LAUNCH_FTC(BM_V, FP8_V, FR_V, IM_V, 1, false)))
    if (bm == 8) {
        if (use_fp8) err = has_id_map ? (fuse_reduce ? LAUNCH_FTC_PDL(8, true, true, true) : LAUNCH_FTC_PDL(8, true, false, true))
                                      : (fuse_reduce ? LAUNCH_FTC_PDL(8, true, true, false) : LAUNCH_FTC_PDL(8, true, false, false));
        else         err = has_id_map ? (fuse_reduce ? LAUNCH_FTC_PDL(8, false, true, true) : LAUNCH_FTC_PDL(8, false, false, true))
                                      : (fuse_reduce ? LAUNCH_FTC_PDL(8, false, true, false) : LAUNCH_FTC_PDL(8, false, false, false));
    } else if (bm == 16) {
        if (use_fp8) err = has_id_map ? (fuse_reduce ? LAUNCH_FTC_PDL(16, true, true, true) : LAUNCH_FTC_PDL(16, true, false, true))
                                      : (fuse_reduce ? LAUNCH_FTC_PDL(16, true, true, false) : LAUNCH_FTC_PDL(16, true, false, false));
        else         err = has_id_map ? (fuse_reduce ? LAUNCH_FTC_PDL(16, false, true, true) : LAUNCH_FTC_PDL(16, false, false, true))
                                      : (fuse_reduce ? LAUNCH_FTC_PDL(16, false, true, false) : LAUNCH_FTC_PDL(16, false, false, false));
    } else {
        if (use_fp8) err = has_id_map ? (fuse_reduce ? LAUNCH_FTC_PDL(32, true, true, true) : LAUNCH_FTC_PDL(32, true, false, true))
                                      : (fuse_reduce ? LAUNCH_FTC_PDL(32, true, true, false) : LAUNCH_FTC_PDL(32, true, false, false));
        else         err = has_id_map ? (fuse_reduce ? LAUNCH_FTC_PDL(32, false, true, true) : LAUNCH_FTC_PDL(32, false, false, true))
                                      : (fuse_reduce ? LAUNCH_FTC_PDL(32, false, true, false) : LAUNCH_FTC_PDL(32, false, false, false));
    }
    #undef LAUNCH_FTC_PDL
    #undef LAUNCH_FTC
    return (err == cudaSuccess) ? 0 : static_cast<int>(err);
}

int dspark_argmax_select_v7_fwd(
    const void* g, int rows, int S, int NS,
    void* out_tok, int64_t ostride, void* conf_out, int64_t cstride, int with_conf,
    int es_mode, int step_idx, int K,
    void* skip_ptr, void* surv_ptr, void* tok_ptr, void* best_ptr,
    const void* greedy_ptr, const void* slot_ptr, const void* db_dev_ptr, const void* lam_dev_ptr,
    const void* ema_ptr, const void* cal_ptr, const void* step_ptr, float prior_logit,
    void* tok2_out, void* p2_out,
    int exp_mode, int use_pdl, uint64_t stream_u64
) {
    if (rows <= 0) return 0;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(stream_u64));
    if (es_mode == 5) {
        if (rows < 2 || rows > 8) return -2;
        dim3 grid_b(1u);
        dim3 block_b(32u);
        cudaError_t err_b = use_pdl
            ? launch_kernel_ex(true, grid_b, block_b, 0, stream, dspark_argmax_select_batch_es_v10_kernel<true>,
                static_cast<const float*>(g), rows, S, NS,
                static_cast<int64_t*>(out_tok), ostride,
                static_cast<float*>(conf_out), cstride,
                step_idx, K,
                static_cast<int32_t*>(skip_ptr), static_cast<float*>(surv_ptr),
                static_cast<const int32_t*>(greedy_ptr), static_cast<const float*>(db_dev_ptr),
                static_cast<const float*>(cal_ptr), prior_logit, exp_mode)
            : launch_kernel_ex(false, grid_b, block_b, 0, stream, dspark_argmax_select_batch_es_v10_kernel<false>,
                static_cast<const float*>(g), rows, S, NS,
                static_cast<int64_t*>(out_tok), ostride,
                static_cast<float*>(conf_out), cstride,
                step_idx, K,
                static_cast<int32_t*>(skip_ptr), static_cast<float*>(surv_ptr),
                static_cast<const int32_t*>(greedy_ptr), static_cast<const float*>(db_dev_ptr),
                static_cast<const float*>(cal_ptr), prior_logit, exp_mode);
        return (err_b == cudaSuccess) ? 0 : static_cast<int>(err_b);
    }
    dim3 grid(static_cast<unsigned int>(rows));
    dim3 block(32u);
    cudaError_t err = use_pdl
        ? launch_kernel_ex(true, grid, block, 0, stream, dspark_argmax_select_v10_kernel<true>,
            static_cast<const float*>(g), rows, S, NS, static_cast<int64_t*>(out_tok), ostride,
            static_cast<float*>(conf_out), cstride, with_conf, es_mode, step_idx, K,
            static_cast<int32_t*>(skip_ptr), static_cast<float*>(surv_ptr), static_cast<float*>(tok_ptr), static_cast<float*>(best_ptr),
            static_cast<const int32_t*>(greedy_ptr), static_cast<const int32_t*>(slot_ptr), static_cast<const float*>(db_dev_ptr), static_cast<const float*>(lam_dev_ptr),
            static_cast<const float*>(ema_ptr), static_cast<const float*>(cal_ptr), static_cast<const float*>(step_ptr), prior_logit,
            static_cast<int64_t*>(tok2_out), static_cast<float*>(p2_out), exp_mode)
        : launch_kernel_ex(false, grid, block, 0, stream, dspark_argmax_select_v10_kernel<false>,
            static_cast<const float*>(g), rows, S, NS, static_cast<int64_t*>(out_tok), ostride,
            static_cast<float*>(conf_out), cstride, with_conf, es_mode, step_idx, K,
            static_cast<int32_t*>(skip_ptr), static_cast<float*>(surv_ptr), static_cast<float*>(tok_ptr), static_cast<float*>(best_ptr),
            static_cast<const int32_t*>(greedy_ptr), static_cast<const int32_t*>(slot_ptr), static_cast<const float*>(db_dev_ptr), static_cast<const float*>(lam_dev_ptr),
            static_cast<const float*>(ema_ptr), static_cast<const float*>(cal_ptr), static_cast<const float*>(step_ptr), prior_logit,
            static_cast<int64_t*>(tok2_out), static_cast<float*>(p2_out), exp_mode);
    return (err == cudaSuccess) ? 0 : static_cast<int>(err);
}

int dspark_select_last_and_vw_conf_v7_fwd(
    int do_select_last,
    const void* g, int num_reqs, int S, int NS,
    int k_run, int n_spec, int c_end,
    int has_skip_last, const void* skip_last_ptr,
    int run_vw_conf,
    const void* idx_ptr, int idx_stride,
    void* conf_buf, int64_t conf_stride,
    const void* temp_ptr, void* w_ptr, const void* c_ptr,
    const void* cal_ptr, const void* step_ptr,
    void* draft_ptr, int64_t draft_stride,
    void* cslot_ptr, const void* db_ptr, const void* etok_ptr, const void* ems_ptr, void* cpred_ptr,
    int w_cap, float tau_override,
    void* req_draft_ptr, int64_t req_draft_stride,
    float prior_logit, int K, int use_hist, int use_dink, int do_log,
    int write_req_drafts, int clear_req_tail, int req_k,
    int exp_mode, int use_pdl, uint64_t stream_u64
) {
    if (num_reqs <= 0 || K > 8) return -1;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(stream_u64));
    dim3 grid(static_cast<unsigned int>(num_reqs));
    dim3 block(32u);
    cudaError_t err;
    #define DISPATCH_SLVW(SEL_V, PDL_V) \
        launch_kernel_ex(PDL_V, grid, block, 0, stream, dspark_select_last_and_vw_conf_v10_kernel<SEL_V, PDL_V>, \
            static_cast<const float*>(g), num_reqs, S, NS, k_run, n_spec, c_end, has_skip_last, static_cast<const int32_t*>(skip_last_ptr), \
            run_vw_conf, static_cast<const int64_t*>(idx_ptr), idx_stride, static_cast<float*>(conf_buf), conf_stride, \
            static_cast<const float*>(temp_ptr), static_cast<int32_t*>(w_ptr), static_cast<const float*>(c_ptr), \
            static_cast<const float*>(cal_ptr), static_cast<const float*>(step_ptr), \
            static_cast<int64_t*>(draft_ptr), draft_stride, static_cast<float*>(cslot_ptr), \
            static_cast<const float*>(db_ptr), static_cast<const float*>(etok_ptr), static_cast<const float*>(ems_ptr), static_cast<float*>(cpred_ptr), \
            w_cap, tau_override, static_cast<int64_t*>(req_draft_ptr), req_draft_stride, \
            prior_logit, K, use_hist, use_dink, do_log, write_req_drafts, clear_req_tail, req_k, exp_mode)
    if (do_select_last) err = use_pdl ? DISPATCH_SLVW(true, true)  : DISPATCH_SLVW(true, false);
    else                err = use_pdl ? DISPATCH_SLVW(false, true) : DISPATCH_SLVW(false, false);
    #undef DISPATCH_SLVW
    return (err == cudaSuccess) ? 0 : static_cast<int>(err);
}

int dspark_select_top2_pack_and_rnn_v7_fwd(
    const void* g_top2, int tp, const void* state0,
    void* beam_state_out, void* beam_lat_out, void* beam_prev,
    void* prim_toks, void* conf_buf, void* sec_toks, void* sec_p, void* p2_0,
    int has_es, int ws_max,
    void* skip_out, void* surv_io, void* tok_io, void* bestu_io,
    const void* greedy_ptr, const void* db_dev_ptr, const void* lam_dev_ptr,
    const void* cal_ptr, const void* step_ptr,
    const void* ctx1, const void* w1, const void* rnn_w,
    int exp_mode, int use_pdl, uint64_t stream_u64
) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(stream_u64));
    dim3 grid(1u, 8u);
    dim3 block(128u);
    cudaError_t err = use_pdl
        ? launch_kernel_ex(true, grid, block, 0, stream, dspark_select_top2_pack_and_rnn_v10_kernel<true>,
            static_cast<const float*>(g_top2), tp, static_cast<const uint16_t*>(state0),
            static_cast<uint16_t*>(beam_state_out), static_cast<uint16_t*>(beam_lat_out), static_cast<int64_t*>(beam_prev),
            static_cast<int64_t*>(prim_toks), static_cast<float*>(conf_buf), static_cast<int64_t*>(sec_toks), static_cast<float*>(sec_p), static_cast<float*>(p2_0),
            has_es, ws_max, static_cast<int32_t*>(skip_out), static_cast<float*>(surv_io), static_cast<float*>(tok_io), static_cast<float*>(bestu_io),
            static_cast<const int32_t*>(greedy_ptr), static_cast<const float*>(db_dev_ptr), static_cast<const float*>(lam_dev_ptr),
            static_cast<const float*>(cal_ptr), static_cast<const float*>(step_ptr),
            static_cast<const float*>(ctx1), static_cast<const uint16_t*>(w1), static_cast<const uint16_t*>(rnn_w), exp_mode)
        : launch_kernel_ex(false, grid, block, 0, stream, dspark_select_top2_pack_and_rnn_v10_kernel<false>,
            static_cast<const float*>(g_top2), tp, static_cast<const uint16_t*>(state0),
            static_cast<uint16_t*>(beam_state_out), static_cast<uint16_t*>(beam_lat_out), static_cast<int64_t*>(beam_prev),
            static_cast<int64_t*>(prim_toks), static_cast<float*>(conf_buf), static_cast<int64_t*>(sec_toks), static_cast<float*>(sec_p), static_cast<float*>(p2_0),
            has_es, ws_max, static_cast<int32_t*>(skip_out), static_cast<float*>(surv_io), static_cast<float*>(tok_io), static_cast<float*>(bestu_io),
            static_cast<const int32_t*>(greedy_ptr), static_cast<const float*>(db_dev_ptr), static_cast<const float*>(lam_dev_ptr),
            static_cast<const float*>(cal_ptr), static_cast<const float*>(step_ptr),
            static_cast<const float*>(ctx1), static_cast<const uint16_t*>(w1), static_cast<const uint16_t*>(rnn_w), exp_mode);
    return (err == cudaSuccess) ? 0 : static_cast<int>(err);
}

int dspark_select_beam2_and_rnn_v7_fwd(
    const void* g, int S, int NS, int step, int has_es, int can_set_skip, int next_is_r1,
    void* bt1_out, void* bp1_out, void* prim_toks, void* conf_buf, void* sec_toks, void* sec_p,
    void* skip_out, void* surv_io, void* tok_io, void* bestu_io,
    const void* greedy_ptr, const void* db_dev_ptr, const void* lam_dev_ptr,
    const void* cal_ptr, const void* step_ptr,
    const void* beam_state_in, const void* ctx_next, const void* w1, const void* rnn_w,
    void* out_state_next, void* latent_next,
    int exp_mode, int use_pdl, uint64_t stream_u64
) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(stream_u64));
    dim3 grid(1u, next_is_r1 ? 48u : 8u);
    dim3 block(128u);
    cudaError_t err;
    #define DISPATCH_SB2(R1_V, PDL_V) \
        launch_kernel_ex(PDL_V, grid, block, 0, stream, dspark_select_beam2_and_rnn_v10_kernel<R1_V, PDL_V>, \
            static_cast<const float*>(g), S, NS, step, has_es, can_set_skip, \
            static_cast<int64_t*>(bt1_out), static_cast<float*>(bp1_out), static_cast<int64_t*>(prim_toks), static_cast<float*>(conf_buf), static_cast<int64_t*>(sec_toks), static_cast<float*>(sec_p), \
            static_cast<int32_t*>(skip_out), static_cast<float*>(surv_io), static_cast<float*>(tok_io), static_cast<float*>(bestu_io), \
            static_cast<const int32_t*>(greedy_ptr), static_cast<const float*>(db_dev_ptr), static_cast<const float*>(lam_dev_ptr), \
            static_cast<const float*>(cal_ptr), static_cast<const float*>(step_ptr), \
            static_cast<const uint16_t*>(beam_state_in), static_cast<const float*>(ctx_next), static_cast<const uint16_t*>(w1), static_cast<const uint16_t*>(rnn_w), \
            static_cast<uint16_t*>(out_state_next), static_cast<uint16_t*>(latent_next), exp_mode)
    if (next_is_r1) err = use_pdl ? DISPATCH_SB2(true, true)  : DISPATCH_SB2(true, false);
    else            err = use_pdl ? DISPATCH_SB2(false, true) : DISPATCH_SB2(false, false);
    #undef DISPATCH_SB2
    return (err == cudaSuccess) ? 0 : static_cast<int>(err);
}

int dspark_select_last_and_tree_pack_v7_fwd(
    int do_select_step6,
    const void* g_last, int S, int NS, int has_skip_last, const void* skip_last_ptr,
    const void* idx_ptr, const void* temp_ptr,
    void* prim_toks, void* conf_buf, const void* sec_toks, const void* sec_p,
    const void* cal_ptr, const void* step_ptr,
    const void* db_ptr, const void* etok_ptr, const void* ems_ptr,
    void* w_ptr, void* cpred_ptr,
    void* draft_ptr, void* req_draft_ptr, int64_t req_draft_stride,
    void* tree_active, void* n_live, void* tree_meta, void* tree_parent, void* tree_depth,
    void* tree_sconv_anc, void* tree_path,
    int ws_max, float sib_disc, float sib_penalty_ms, float sib_scale, float p2_min,
    int write_req_drafts, int exp_mode, int use_pdl, uint64_t stream_u64
) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(stream_u64));
    dim3 grid(1u);
    dim3 block(32u);
    cudaError_t err;
    #define DISPATCH_SLTP(SEL_V, PDL_V) \
        launch_kernel_ex(PDL_V, grid, block, 0, stream, dspark_select_last_and_tree_pack_v10_kernel<SEL_V, PDL_V>, \
            static_cast<const float*>(g_last), S, NS, has_skip_last, static_cast<const int32_t*>(skip_last_ptr), \
            static_cast<const int64_t*>(idx_ptr), static_cast<const float*>(temp_ptr), \
            static_cast<int64_t*>(prim_toks), static_cast<float*>(conf_buf), static_cast<const int64_t*>(sec_toks), static_cast<const float*>(sec_p), \
            static_cast<const float*>(cal_ptr), static_cast<const float*>(step_ptr), \
            static_cast<const float*>(db_ptr), static_cast<const float*>(etok_ptr), static_cast<const float*>(ems_ptr), \
            static_cast<int32_t*>(w_ptr), static_cast<float*>(cpred_ptr), \
            static_cast<int64_t*>(draft_ptr), static_cast<int64_t*>(req_draft_ptr), req_draft_stride, \
            static_cast<int32_t*>(tree_active), static_cast<int32_t*>(n_live), static_cast<int32_t*>(tree_meta), \
            static_cast<int32_t*>(tree_parent), static_cast<int32_t*>(tree_depth), static_cast<int32_t*>(tree_sconv_anc), static_cast<int32_t*>(tree_path), \
            ws_max, sib_disc, sib_penalty_ms, sib_scale, p2_min, write_req_drafts, exp_mode)
    if (do_select_step6) err = use_pdl ? DISPATCH_SLTP(true, true)  : DISPATCH_SLTP(true, false);
    else                 err = use_pdl ? DISPATCH_SLTP(false, true) : DISPATCH_SLTP(false, false);
    #undef DISPATCH_SLTP
    return (err == cudaSuccess) ? 0 : static_cast<int>(err);
}

int sync_free_local_pairs_v7_fwd(
    const void* lg, int is_fp32, int64_t sl,
    void* rec_split, void* pair_out, void* sem,
    int M, int splits, int valid, int vstart, int seg,
    float inv_cap, float cap, int has_cap, int fp32_chain, int divmode,
    int fuse_split1, int use_pdl, uint64_t stream_u64
) {
    if (M <= 0 || splits <= 0) return 0;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(stream_u64));
    dim3 grid(static_cast<unsigned int>(M), static_cast<unsigned int>(splits));
    dim3 block(256u);
    float* rs = static_cast<float*>(rec_split);
    float* po = static_cast<float*>(pair_out);
    uint32_t* sp = static_cast<uint32_t*>(sem);
    cudaError_t err;
    #define DISPATCH_SFL(F32_V, CAP_V, FCH_V, DM_V, FS1_V, PDL_V) \
        launch_kernel_ex(PDL_V, grid, block, 0, stream, sync_free_local_pairs_v10_kernel<F32_V, CAP_V, FCH_V, DM_V, FS1_V, PDL_V>, \
            lg, sl, rs, po, sp, valid, vstart, seg, inv_cap, cap)
    #define DISPATCH_SFL_FS(F32_V, CAP_V, FCH_V, DM_V) \
        (fuse_split1 ? (use_pdl ? DISPATCH_SFL(F32_V, CAP_V, FCH_V, DM_V, true, true)  : DISPATCH_SFL(F32_V, CAP_V, FCH_V, DM_V, true, false)) \
                     : (use_pdl ? DISPATCH_SFL(F32_V, CAP_V, FCH_V, DM_V, false, true) : DISPATCH_SFL(F32_V, CAP_V, FCH_V, DM_V, false, false)))
    if (is_fp32) {
        if (!has_cap)        err = DISPATCH_SFL_FS(true, false, false, 0);
        else if (fp32_chain) err = (divmode == 0) ? DISPATCH_SFL_FS(true, true, true, 0) : DISPATCH_SFL_FS(true, true, true, 1);
        else                 err = DISPATCH_SFL_FS(true, true, false, 0);
    } else {
        err = has_cap ? DISPATCH_SFL_FS(false, true, false, 0) : DISPATCH_SFL_FS(false, false, false, 0);
    }
    #undef DISPATCH_SFL_FS
    #undef DISPATCH_SFL
    return (err == cudaSuccess) ? 0 : static_cast<int>(err);
}

int sync_free_k0_tok_status_v7_fwd(
    const void* g, int M, int tp, int split,
    void* tok_out, int write_status,
    void* num_sampled_out, void* num_rejected_out,
    const void* seq_lens, const void* idx_mapping, int idx_is_i64, const void* prefill_len,
    int use_pdl, uint64_t stream_u64
) {
    if (M <= 0) return 0;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(stream_u64));
    dim3 grid(static_cast<unsigned int>(M));
    dim3 block(32u);
    cudaError_t err;
    if (write_status) {
        err = use_pdl
            ? launch_kernel_ex(true, grid, block, 0, stream, sync_free_k0_tok_status_v10_kernel<true, true>,
                static_cast<const float*>(g), M, tp, split, static_cast<int64_t*>(tok_out),
                static_cast<int32_t*>(num_sampled_out), static_cast<int32_t*>(num_rejected_out),
                static_cast<const int32_t*>(seq_lens), idx_mapping, idx_is_i64, static_cast<const int32_t*>(prefill_len))
            : launch_kernel_ex(false, grid, block, 0, stream, sync_free_k0_tok_status_v10_kernel<true, false>,
                static_cast<const float*>(g), M, tp, split, static_cast<int64_t*>(tok_out),
                static_cast<int32_t*>(num_sampled_out), static_cast<int32_t*>(num_rejected_out),
                static_cast<const int32_t*>(seq_lens), idx_mapping, idx_is_i64, static_cast<const int32_t*>(prefill_len));
    } else {
        err = use_pdl
            ? launch_kernel_ex(true, grid, block, 0, stream, sync_free_k0_tok_status_v10_kernel<false, true>,
                static_cast<const float*>(g), M, tp, split, static_cast<int64_t*>(tok_out),
                nullptr, nullptr, nullptr, nullptr, 0, nullptr)
            : launch_kernel_ex(false, grid, block, 0, stream, sync_free_k0_tok_status_v10_kernel<false, false>,
                static_cast<const float*>(g), M, tp, split, static_cast<int64_t*>(tok_out),
                nullptr, nullptr, nullptr, nullptr, 0, nullptr);
    }
    return (err == cudaSuccess) ? 0 : static_cast<int>(err);
}

int sync_free_fused_reject_v7_fwd(
    const void* g, int M, int num_reqs, int tp, int split,
    void* sampled, int64_t sampled_stride,
    void* num_sampled, void* num_rejected,
    const void* seq_lens, const void* input_ids, const void* positions,
    const void* logits_indices, const void* idx_mapping,
    const void* prefill_len, const void* cu_num_logits,
    uint32_t dtype_flags, int use_pdl, uint64_t stream_u64
) {
    if (num_reqs <= 0) return 0;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(stream_u64));
    dim3 grid(static_cast<unsigned int>(num_reqs));
    dim3 block(32u);
    cudaError_t err = use_pdl
        ? launch_kernel_ex(true, grid, block, 0, stream, sync_free_fused_reject_v10_kernel<true>,
            static_cast<const float*>(g), M, tp, split, static_cast<int64_t*>(sampled), sampled_stride,
            static_cast<int32_t*>(num_sampled), static_cast<int32_t*>(num_rejected),
            static_cast<const int32_t*>(seq_lens), input_ids, positions, logits_indices, idx_mapping,
            static_cast<const int32_t*>(prefill_len), static_cast<const int32_t*>(cu_num_logits), dtype_flags)
        : launch_kernel_ex(false, grid, block, 0, stream, sync_free_fused_reject_v10_kernel<false>,
            static_cast<const float*>(g), M, tp, split, static_cast<int64_t*>(sampled), sampled_stride,
            static_cast<int32_t*>(num_sampled), static_cast<int32_t*>(num_rejected),
            static_cast<const int32_t*>(seq_lens), input_ids, positions, logits_indices, idx_mapping,
            static_cast<const int32_t*>(prefill_len), static_cast<const int32_t*>(cu_num_logits), dtype_flags);
    return (err == cudaSuccess) ? 0 : static_cast<int>(err);
}

int sync_free_fused_tree_reject_v7_fwd(
    const void* g, int M, int num_reqs, int tp, int split, int max_m,
    void* sampled, int64_t sampled_stride,
    void* num_sampled, void* num_rejected,
    const void* seq_lens, const void* input_ids, const void* positions,
    const void* logits_indices, const void* idx_mapping,
    const void* prefill_len, const void* cu_num_logits,
    const void* tree_parent, const void* n_live_ptr,
    void* accepted_path, void* accepted_len, void* needs_compact,
    uint32_t dtype_flags, int use_pdl, uint64_t stream_u64
) {
    if (num_reqs <= 0 || max_m > 16) return -1;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(stream_u64));
    dim3 grid(static_cast<unsigned int>(num_reqs));
    dim3 block(32u);
    cudaError_t err = use_pdl
        ? launch_kernel_ex(true, grid, block, 0, stream, sync_free_fused_tree_reject_v10_kernel<true>,
            static_cast<const float*>(g), M, tp, split, max_m, static_cast<int64_t*>(sampled), sampled_stride,
            static_cast<int32_t*>(num_sampled), static_cast<int32_t*>(num_rejected),
            static_cast<const int32_t*>(seq_lens), input_ids, positions, logits_indices, idx_mapping,
            static_cast<const int32_t*>(prefill_len), static_cast<const int32_t*>(cu_num_logits),
            static_cast<const int32_t*>(tree_parent), static_cast<const int32_t*>(n_live_ptr),
            static_cast<int32_t*>(accepted_path), static_cast<int32_t*>(accepted_len), static_cast<int32_t*>(needs_compact), dtype_flags)
        : launch_kernel_ex(false, grid, block, 0, stream, sync_free_fused_tree_reject_v10_kernel<false>,
            static_cast<const float*>(g), M, tp, split, max_m, static_cast<int64_t*>(sampled), sampled_stride,
            static_cast<int32_t*>(num_sampled), static_cast<int32_t*>(num_rejected),
            static_cast<const int32_t*>(seq_lens), input_ids, positions, logits_indices, idx_mapping,
            static_cast<const int32_t*>(prefill_len), static_cast<const int32_t*>(cu_num_logits),
            static_cast<const int32_t*>(tree_parent), static_cast<const int32_t*>(n_live_ptr),
            static_cast<int32_t*>(accepted_path), static_cast<int32_t*>(accepted_len), static_cast<int32_t*>(needs_compact), dtype_flags);
    return (err == cudaSuccess) ? 0 : static_cast<int>(err);
}

int dspark_vw_update_dink_v7_fwd(
    const void* idx_ptr, int idx_is_i64, int num_reqs,
    const void* num_sampled, const void* num_rejected, const void* temp_ptr,
    const void* w_ptr, const void* cpred_ptr,
    void* db_ptr, void* etok_ptr, void* ems_ptr,
    const void* step_ptr, void* stats_ptr, const void* cal_ptr,
    void* es_skip, void* es_surv, void* es_tok, void* es_bestu,
    void* es_greedy, void* es_slot, void* es_db, void* es_lam,
    void* es_skip_pp, void* es_surv_pp, void* es_tok_pp, void* es_bestu_pp,
    float eta_b, float decay_b, float b_clip, float eta_lam,
    int K, int do_stats, int fuse_es, int use_pdl, uint64_t stream_u64
) {
    if (num_reqs <= 0 || K > 8) return -1;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(stream_u64));
    dim3 grid(static_cast<unsigned int>(num_reqs));
    dim3 block(32u);
    cudaError_t err = use_pdl
        ? launch_kernel_ex(true, grid, block, 0, stream, dspark_vw_update_dink_v10_kernel<true>,
            idx_ptr, idx_is_i64, static_cast<const int32_t*>(num_sampled), static_cast<const int32_t*>(num_rejected),
            static_cast<const float*>(temp_ptr), static_cast<const int32_t*>(w_ptr), static_cast<const float*>(cpred_ptr),
            static_cast<float*>(db_ptr), static_cast<float*>(etok_ptr), static_cast<float*>(ems_ptr),
            static_cast<const float*>(step_ptr), static_cast<int32_t*>(stats_ptr), static_cast<const float*>(cal_ptr),
            static_cast<int32_t*>(es_skip), static_cast<float*>(es_surv), static_cast<float*>(es_tok), static_cast<float*>(es_bestu),
            static_cast<int32_t*>(es_greedy), static_cast<int32_t*>(es_slot), static_cast<float*>(es_db), static_cast<float*>(es_lam),
            static_cast<int32_t*>(es_skip_pp), static_cast<float*>(es_surv_pp), static_cast<float*>(es_tok_pp), static_cast<float*>(es_bestu_pp),
            eta_b, decay_b, b_clip, eta_lam, K, do_stats, fuse_es)
        : launch_kernel_ex(false, grid, block, 0, stream, dspark_vw_update_dink_v10_kernel<false>,
            idx_ptr, idx_is_i64, static_cast<const int32_t*>(num_sampled), static_cast<const int32_t*>(num_rejected),
            static_cast<const float*>(temp_ptr), static_cast<const int32_t*>(w_ptr), static_cast<const float*>(cpred_ptr),
            static_cast<float*>(db_ptr), static_cast<float*>(etok_ptr), static_cast<float*>(ems_ptr),
            static_cast<const float*>(step_ptr), static_cast<int32_t*>(stats_ptr), static_cast<const float*>(cal_ptr),
            static_cast<int32_t*>(es_skip), static_cast<float*>(es_surv), static_cast<float*>(es_tok), static_cast<float*>(es_bestu),
            static_cast<int32_t*>(es_greedy), static_cast<int32_t*>(es_slot), static_cast<float*>(es_db), static_cast<float*>(es_lam),
            static_cast<int32_t*>(es_skip_pp), static_cast<float*>(es_surv_pp), static_cast<float*>(es_tok_pp), static_cast<float*>(es_bestu_pp),
            eta_b, decay_b, b_clip, eta_lam, K, do_stats, fuse_es);
    return (err == cudaSuccess) ? 0 : static_cast<int>(err);
}

int dspark_fused_epilogue_prep_v7_fwd(
    void* out_input_ids, void* out_query_positions, void* out_is_padding,
    void* out_query_start_loc, void* out_seq_lens, void* out_context_positions,
    void* out_sample_indices, void* out_sample_pos, void* out_sample_idx_mapping,
    void* out_temperature, void* out_seeds, void* out_spec_num_rejected,
    void* q_slot0, void* q_slot1, void* q_slot2, void* q_slot3, void* q_slot4,
    void* c_slot0, void* c_slot1, void* c_slot2, void* c_slot3, void* c_slot4,
    const void* bt0, const void* bt1, const void* bt2, const void* bt3, const void* bt4,
    int64_t bts0, int64_t bts1, int64_t bts2, int64_t bts3, int64_t bts4,
    int64_t bsz0, int64_t bsz1, int64_t bsz2, int64_t bsz3, int64_t bsz4,
    int64_t bst0, int64_t bst1, int64_t bst2, int64_t bst3, int64_t bst4,
    const void* target_positions, const void* target_query_start_loc, const void* idx_mapping,
    const void* last_sampled, const void* next_prefill_tokens,
    const void* num_sampled, const void* num_rejected,
    const void* temperature, const void* seeds,
    int32_t parallel_drafting_token_id, int32_t num_query_per_req, int32_t num_speculative_steps,
    int32_t max_num_reqs, int32_t max_num_tokens, int64_t max_model_len,
    int32_t num_reqs, int32_t num_pad_programs, int32_t num_blocks,
    int64_t pad_slot_id, int write_spec_num_rejected,
    int32_t pad_c_start, int32_t pad_c_end, int ng, uint32_t dtype_flags,
    int do_dink_update, const void* w_ptr, const void* cpred_ptr,
    void* db_ptr, void* etok_ptr, void* ems_ptr,
    const void* step_ptr, void* stats_ptr, const void* cal_ptr,
    void* es_skip, void* es_surv, void* es_tok, void* es_bestu,
    void* es_greedy, void* es_slot, void* es_db, void* es_lam,
    void* es_skip_pp, void* es_surv_pp, void* es_tok_pp, void* es_bestu_pp,
    float eta_b, float decay_b, float b_clip, float eta_lam,
    int dink_k, int do_stats, int fuse_es, int use_pdl, uint64_t stream_u64
) {
    if (num_reqs <= 0 || num_blocks <= 0 || ng < 1 || ng > 5) return -1;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(stream_u64));
    dim3 grid(static_cast<unsigned int>(num_reqs + num_pad_programs), static_cast<unsigned int>(num_blocks));
    dim3 block(128u);
    cudaError_t err;
    #define DISPATCH_EP(NG_V, PDL_V) \
        launch_kernel_ex(PDL_V, grid, block, 0, stream, dspark_fused_epilogue_prep_v10_kernel<NG_V, PDL_V>, \
            out_input_ids, out_query_positions, static_cast<uint8_t*>(out_is_padding), \
            static_cast<int32_t*>(out_query_start_loc), static_cast<int32_t*>(out_seq_lens), out_context_positions, \
            out_sample_indices, out_sample_pos, out_sample_idx_mapping, \
            static_cast<float*>(out_temperature), out_seeds, static_cast<int32_t*>(out_spec_num_rejected), \
            static_cast<int64_t*>(q_slot0), static_cast<int64_t*>(q_slot1), static_cast<int64_t*>(q_slot2), static_cast<int64_t*>(q_slot3), static_cast<int64_t*>(q_slot4), \
            static_cast<int64_t*>(c_slot0), static_cast<int64_t*>(c_slot1), static_cast<int64_t*>(c_slot2), static_cast<int64_t*>(c_slot3), static_cast<int64_t*>(c_slot4), \
            static_cast<const int32_t*>(bt0), static_cast<const int32_t*>(bt1), static_cast<const int32_t*>(bt2), static_cast<const int32_t*>(bt3), static_cast<const int32_t*>(bt4), \
            bts0, bts1, bts2, bts3, bts4, bsz0, bsz1, bsz2, bsz3, bsz4, bst0, bst1, bst2, bst3, bst4, \
            target_positions, static_cast<const int32_t*>(target_query_start_loc), idx_mapping, \
            last_sampled, next_prefill_tokens, static_cast<const int32_t*>(num_sampled), static_cast<const int32_t*>(num_rejected), \
            static_cast<const float*>(temperature), seeds, \
            parallel_drafting_token_id, num_query_per_req, num_speculative_steps, max_num_reqs, max_num_tokens, max_model_len, \
            num_reqs, num_pad_programs, pad_slot_id, write_spec_num_rejected, pad_c_start, pad_c_end, dtype_flags, \
            do_dink_update, static_cast<const int32_t*>(w_ptr), static_cast<const float*>(cpred_ptr), \
            static_cast<float*>(db_ptr), static_cast<float*>(etok_ptr), static_cast<float*>(ems_ptr), \
            static_cast<const float*>(step_ptr), static_cast<int32_t*>(stats_ptr), static_cast<const float*>(cal_ptr), \
            static_cast<int32_t*>(es_skip), static_cast<float*>(es_surv), static_cast<float*>(es_tok), static_cast<float*>(es_bestu), \
            static_cast<int32_t*>(es_greedy), static_cast<int32_t*>(es_slot), static_cast<float*>(es_db), static_cast<float*>(es_lam), \
            static_cast<int32_t*>(es_skip_pp), static_cast<float*>(es_surv_pp), static_cast<float*>(es_tok_pp), static_cast<float*>(es_bestu_pp), \
            eta_b, decay_b, b_clip, eta_lam, dink_k, do_stats, fuse_es)
    switch (ng) {
        case 1: err = use_pdl ? DISPATCH_EP(1, true) : DISPATCH_EP(1, false); break;
        case 2: err = use_pdl ? DISPATCH_EP(2, true) : DISPATCH_EP(2, false); break;
        case 3: err = use_pdl ? DISPATCH_EP(3, true) : DISPATCH_EP(3, false); break;
        case 4: err = use_pdl ? DISPATCH_EP(4, true) : DISPATCH_EP(4, false); break;
        default: err = use_pdl ? DISPATCH_EP(5, true) : DISPATCH_EP(5, false); break;
    }
    #undef DISPATCH_EP
    return (err == cudaSuccess) ? 0 : static_cast<int>(err);
}

// ============================================================================
// Wave-V10 Version & Native Batch-All Early-Stop Exported Entrypoints
// ============================================================================

int dspark_markov_v10_version() {
    return 10;
}

int dspark_markov_rnn_step_v10_fwd(
    const void* state, const void* prev, int prev_is_i32,
    const void* ctx, int64_t ctx_row_stride,
    const void* w1, const void* rnn_w,
    void* out_state, void* latent, const void* skip_ptr,
    int rows, int rank, int64_t prev_stride,
    int init_zero, int has_skip, int use_tc, int use_pdl, uint64_t stream_u64
) {
    return dspark_markov_rnn_step_v7_fwd(
        state, prev, prev_is_i32, ctx, ctx_row_stride, w1, rnn_w,
        out_state, latent, skip_ptr, rows, rank, prev_stride,
        init_zero, has_skip, use_tc, use_pdl, stream_u64
    );
}

int dspark_fold_argmax_v10_fwd(
    const void* base, const void* lat, const void* fw, const void* fs, const void* fb,
    void* rec, void* rec_out, void* sem, const void* skip_ptr, const void* id_map,
    int rows, int rank, int slices, int ch, int64_t stride_base,
    float softcap, int valid, int offset,
    int use_fp8, int has_skip, int fuse_reduce, int has_id_map,
    int top2, int fma_mode, int exp_mode, int use_pdl, uint64_t stream_u64
) {
    return dspark_fold_argmax_v7_fwd(
        base, lat, fw, fs, fb, rec, rec_out, sem, skip_ptr, id_map,
        rows, rank, slices, ch, stride_base, softcap, valid, offset,
        use_fp8, has_skip, fuse_reduce, has_id_map, top2, fma_mode, exp_mode, use_pdl, stream_u64
    );
}

int dspark_select_and_rnn_step_v10_fwd(
    const void* g, int rows, int S, int NS,
    void* out_tok_i, int64_t out_tok_stride,
    void* conf_i, int64_t conf_stride, int with_conf,
    int es_mode, int step_idx, int K,
    const void* skip_in, const void* surv_in, const void* tok_in, const void* best_in,
    void* skip_out, void* surv_out, void* tok_out, void* best_out,
    const void* greedy_ptr, const void* slot_ptr, const void* db_dev_ptr, const void* lam_dev_ptr,
    const void* ema_ptr, const void* cal_ptr, const void* step_ptr, float prior_logit,
    const void* state_i, const void* ctx_next, int64_t ctx_row_stride,
    const void* w1, const void* rnn_w,
    void* out_state_next, void* latent_next,
    int exp_mode, int use_pdl, uint64_t stream_u64
) {
    return dspark_select_and_rnn_step_v7_fwd(
        g, rows, S, NS, out_tok_i, out_tok_stride, conf_i, conf_stride, with_conf,
        es_mode, step_idx, K, skip_in, surv_in, tok_in, best_in,
        skip_out, surv_out, tok_out, best_out,
        greedy_ptr, slot_ptr, db_dev_ptr, lam_dev_ptr, ema_ptr, cal_ptr, step_ptr, prior_logit,
        state_i, ctx_next, ctx_row_stride, w1, rnn_w, out_state_next, latent_next,
        exp_mode, use_pdl, stream_u64
    );
}

int dspark_argmax_select_batch_es_v10_fwd(
    const void* g, int rows, int S, int NS,
    void* out_tok, int64_t ostride,
    void* conf_out, int64_t cstride,
    int step_idx, int K,
    void* skip_ptr, void* surv_b_ptr,
    const void* greedy_all_ptr, const void* db_b_ptr, const void* cal_ptr,
    float tau_batch, int exp_mode, int use_pdl, uint64_t stream_u64
) {
    return dspark_argmax_select_v7_fwd(
        g, rows, S, NS, out_tok, ostride, conf_out, cstride, 1,
        5, step_idx, K, skip_ptr, surv_b_ptr, nullptr, nullptr,
        greedy_all_ptr, nullptr, db_b_ptr, nullptr, nullptr, cal_ptr, nullptr, tau_batch,
        nullptr, nullptr, exp_mode, use_pdl, stream_u64
    );
}

}  // extern "C"
