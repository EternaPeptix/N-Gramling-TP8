// Wave-V9 SM121a (GB10 Blackwell) Native BF16 Tensor-Core 4-Bit Panel GEMM
// Target: NVIDIA GB10 SM121a (48 SMs, 244 GB/s LPDDR5X DRAM, 24 MiB L2, 99 KiB SMEM/SM)
// Build:  nvcc -O3 -std=c++17 -gencode arch=compute_121a,code=sm_121a --fmad=false
//              --shared -Xcompiler -fPIC --ptxas-options=-v -o libdrafter_u4_tc_v9.so drafter_u4_tc_v9.cu
//
// Design highlights vs drafter_u4_gemm_v5/v7 (which lost to Triton skinny_int4.py on GB10):
//  1. Native BF16 MMA (`mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`) with ZERO X conversion!
//     Eliminates v5/v7's 24-op-per-8-elem BF16->FP16 scalar ALU loop inside every K-step.
//  2. True 2-stage software pipeline:
//     - `cp.async.ca.shared.global` directly into bank-conflict-free `sX[2][BM_TILE][136]` (8.5-34 KiB)
//     - `ldmatrix.sync.aligned.x4.m8n8.shared.b16` (1 op per 16x16 A-tile, 0 bank conflicts)
//     - Zero-SMEM register double-buffered `WP` (`uint4 wp_cur, wp_nxt`, 8 regs/thr) via coalesced `ld.global.cs.v4.u32`
//  3. Fast zero-SMEM inline 4-bit nibble unpack:
//     - Quad `__shfl_sync` + precomputed `PRMT` selector + `LOP3.B32` + native SM121a `add.rn.bf16x2` / `fma.rn.bf16x2` / `mul.rn.bf16x2`
//     - Exact BF16 integer bias: u4 `0x43004300` (128+q) + `0xC300C300` (-128) = exact q;
//       i4 `0x42804280 | (q<<1)` (64+q) + `0xC28FC28F` (-71.5) = exact (q - 7.5)!
//  4. Zero weak shapes across all 7 DSpark drafter GEMMs (M = 1..64):
//     - `BN_CTA = 16` (`kv_proj_ctx`): 4 warps (128 thr) with 2-way K-split (2x v7 warp occupancy, 0 extra SMEM!)
//     - `BN_CTA = 32` (4 warps): exact `144 = 3x48` waves on `gate_up`/`o_proj`/`down_proj`, `96 = 2x48` on `qkv`/`rnn_ctx`, `786` on `draft_head` (M<=32)
//     - `BN_CTA = 64` (8 warps): `393` CTAs on `draft_head` at M=33..64, halving `sX` L2 traffic & `cp.async` pressure!
//  5. Fused `gate_up + SiLU*mul` epilogue (zero shuffle — `(gate, up)` already reside in `(d0, d1)` & `(d2, d3)`!),
//     unified `FUSE_RNN` & `FUSE_RNN_BATCH` (precomputed SMEM row tables — zero division in K-loop!),
//     FP32/BF16 bias, hardware-coherent split-K finalize (`st.cg + __threadfence + atom.acq_rel.gpu + ld.cg`),
//     and safe PDL (static `WP` preload before `gdc_wait()`, `gdc_launch_dependents()` strictly at epilogue!).
//  6. Zero local/stack arrays, explicit scalar FP32 accumulators, zero register spills, zero cooperative launches.

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

static __device__ __forceinline__ float ld_cg_f32(const float* ptr) {
    float v;
#if defined(__CUDA_ARCH__)
    asm volatile("ld.global.cg.f32 %0, [%1];\n" : "=f"(v) : "l"(ptr));
#else
    v = *ptr;
#endif
    return v;
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

static __device__ __forceinline__ void cp_async_ca_16_zfill(
    uint32_t smem_addr, const void* gmem_ptr, int src_bytes
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(smem_addr), "l"(gmem_ptr), "r"(src_bytes) : "memory");
#else
    (void)smem_addr; (void)gmem_ptr; (void)src_bytes;
#endif
}

static __device__ __forceinline__ void cp_async_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::: "memory");
#endif
}

template <int N>
static __device__ __forceinline__ void cp_async_wait_group() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
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

static __device__ __forceinline__ void mma_m16n8k16_bf16(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)
    );
#else
    (void)d0; (void)d1; (void)d2; (void)d3;
    (void)a0; (void)a1; (void)a2; (void)a3; (void)b0; (void)b1;
#endif
}

// Convert FP16 raw bits -> packed BF16x2 (identical pair in lo16 & hi16)
static __device__ __forceinline__ uint32_t f16_bits_to_bf16x2(uint16_t h_bits) {
#if defined(__CUDA_ARCH__)
    float f;
    asm("cvt.f32.f16 %0, %1;\n" : "=f"(f) : "h"(h_bits));
    uint16_t b;
    asm("cvt.rn.bf16.f32 %0, %1;\n" : "=h"(b) : "f"(f));
    return (static_cast<uint32_t>(b) << 16) | static_cast<uint32_t>(b);
#else
    (void)h_bits;
    return 0u;
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

static __device__ __forceinline__ float bf16_bits_to_f32(uint16_t b) {
    union { uint32_t u; float f; } x;
    x.u = static_cast<uint32_t>(b) << 16;
    return x.f;
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

// Fast exact SM121a inline 4-bit nibble unpack -> BF16x2 (0 SMEM, 0 branches!)
// `word` holds 4 packed bytes (weights 0..7); `prmt_sel` selects byte `lt` from `a` and `b`.
// - Asymmetric `u4`: exact `q in [0..15]` via `(0x4300 | q) + (-128.0)`, then `fma.rn.bf16x2(q, sc2, zp2)`
// - Symmetric `i4`:  exact `(q - 7.5)` via `(0x4280 | (q << 1)) + (-71.5)`, then `mul.rn.bf16x2(q_c, sc2)`
template <bool SYM>
static __device__ __forceinline__ uint32_t unpack_byte_to_bf16x2(
    uint32_t word, uint32_t prmt_sel, uint32_t sc2, uint32_t zp2
) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    uint32_t res;
    if constexpr (SYM) {
        const uint32_t mask_1e = 0x001E001Eu;
        const uint32_t base_64 = 0x42804280u;   // bf16x2(64.0, 64.0)
        const uint32_t neg_71p5 = 0xC28FC28Fu;  // bf16x2(-71.5, -71.5)
        uint32_t w_l1 = word << 1;
        uint32_t w_r3 = word >> 3;
        uint32_t raw;
        asm("prmt.b32 %0, %1, %2, %3;\n" : "=r"(raw) : "r"(w_l1), "r"(w_r3), "r"(prmt_sel));
        // (raw & 0x001E001E) | 0x42804280 -> exact bf16x2(64.0 + q0, 64.0 + q1)
        asm("lop3.b32 %0, %1, %2, %3, 0xEA;\n" : "=r"(raw) : "r"(raw), "r"(mask_1e), "r"(base_64));
        // + (-71.5, -71.5) -> exact bf16x2(q0 - 7.5, q1 - 7.5)
        asm("add.rn.bf16x2 %0, %1, %2;\n" : "=r"(raw) : "r"(raw), "r"(neg_71p5));
        asm("mul.rn.bf16x2 %0, %1, %2;\n" : "=r"(res) : "r"(raw), "r"(sc2));
    } else {
        const uint32_t mask_0f = 0x000F000Fu;
        const uint32_t base_128 = 0x43004300u;  // bf16x2(128.0, 128.0)
        const uint32_t neg_128 = 0xC300C300u;   // bf16x2(-128.0, -128.0)
        uint32_t w_r4 = word >> 4;
        uint32_t raw;
        asm("prmt.b32 %0, %1, %2, %3;\n" : "=r"(raw) : "r"(word), "r"(w_r4), "r"(prmt_sel));
        // (raw & 0x000F000F) | 0x43004300 -> exact bf16x2(128.0 + q0, 128.0 + q1)
        asm("lop3.b32 %0, %1, %2, %3, 0xEA;\n" : "=r"(raw) : "r"(raw), "r"(mask_0f), "r"(base_128));
        // + (-128.0, -128.0) -> exact bf16x2(q0, q1)
        asm("add.rn.bf16x2 %0, %1, %2;\n" : "=r"(raw) : "r"(raw), "r"(neg_128));
        asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(res) : "r"(raw), "r"(sc2), "r"(zp2));
    }
    return res;
#else
    (void)word; (void)prmt_sel; (void)sc2; (void)zp2;
    return 0u;
#endif
}

static __device__ __forceinline__ float silu_mul_f32(float gate, float up) {
    float sig = 1.0f / (1.0f + expf(-gate));
    return (gate * sig) * up;
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
// Core Unified SM121a Kernel
//   - BN_CTA == 16: 4 warps (128 thr), 2-way K-split inside each 128-K iter (`kw=0`: k=0..63, `kw=1`: k=64..127)
//   - BN_CTA == 32: 4 warps (128 thr), 1 warp per 8 output rows
//   - BN_CTA == 64: 8 warps (256 thr), 1 warp per 8 output rows (for draft_head M=33..64)
//   - MT in {1, 2, 3, 4}: BM_TILE = 16 * MT (`16, 32, 48, 64`), explicit scalar accumulators (0 stack!)
// ============================================================================
template <int BN_CTA, int MT, int GS, bool SYM, bool FUSE_SILU, bool OUT_F32>
__global__ __launch_bounds__(BN_CTA == 64 ? 256 : 128, (MT <= 2) ? 4 : 2)
void drafter_u4_tc_v9_kernel(
    const uint16_t* __restrict__ X1,
    const uint16_t* __restrict__ X2,        // nullptr when RNN_MODE == 0
    const uint8_t*  __restrict__ WP,        // [n_panels, total_k_steps, wp_bn, BK/2]
    const uint16_t* __restrict__ SCP,       // [n_panels, total_k_steps, BK/GS, wp_bn] (fp16)
    const uint16_t* __restrict__ ZPP,       // [n_panels, total_k_steps, BK/GS, wp_bn] (fp16) or nullptr
    const void*     __restrict__ BIAS,      // optional fp32 or bf16 [N]
    float*          __restrict__ PART,      // [n_sub, split_k, BM_TILE, BN_CTA] when split_k > 1
    uint32_t*       __restrict__ SEM,       // [n_sub] when split_k > 1
    void*           __restrict__ OUT,       // [M, out_n] bf16 or fp32
    int M,
    int N,
    int K,
    int wp_bn,                              // panel BN baked into WP/SCP/ZPP (16, 32, or 64)
    int bk,                                 // 128 or 256
    int split_k,
    int iters_per_split,                    // number of 128-K iters per split
    int64_t sx1,
    int64_t sx2,
    int64_t so,
    int rnn_mode,                           // 0=normal, 1=FUSE_RNN, 2=FUSE_RNN_BATCH
    int n_reqs,
    int k_steps_rnn,
    int has_bias,
    int bias_is_f32,
    int use_pdl
) {
    constexpr int BM_TILE = MT * 16;
    constexpr int SX_STRIDE = 136;          // 128 + 8 bf16 padding -> 0 bank conflicts for ldmatrix.x4!
    constexpr int N_THREADS = (BN_CTA == 64) ? 256 : 128;
    constexpr bool K_SPLIT2 = (BN_CTA == 16);
    constexpr int CHUNKS_SX = BM_TILE * 16; // 128 bf16 / 8 bf16 per 16B chunk = 16 chunks/row

    const int pid_n = static_cast<int>(blockIdx.x);
    const int ks_id = static_cast<int>(blockIdx.y);
    const int tid = static_cast<int>(threadIdx.x);
    const int wid = tid >> 5;
    const int lane = tid & 31;
    const int grp = lane >> 2;              // 0..7
    const int lt = lane & 3;                // 0..3

    // Warp-to-output-row & K-split assignment
    const int kw = K_SPLIT2 ? (wid >> 1) : 0;          // 0 or 1 (only for BN_CTA==16)
    const int n_warp = K_SPLIT2 ? (wid & 1) : wid;     // 0..1 (BN=16), 0..3 (BN=32), 0..7 (BN=64)
    const int n_cta = pid_n * BN_CTA;
    const int n_row_local = n_warp * 8 + grp;          // 0..BN_CTA-1

    // Map CTA sub-panel -> baked WP panel `(wp_panel, wp_row)`
    const int sub_ratio = wp_bn / BN_CTA;              // 1 or 2
    const int wp_panel = (sub_ratio == 2) ? (pid_n >> 1) : pid_n;
    const int wp_sub_off = (sub_ratio == 2) ? ((pid_n & 1) * BN_CTA) : 0;
    const int wp_row = wp_sub_off + n_row_local;       // 0..wp_bn-1

    const int total_iters = K >> 7;                    // K / 128
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
    const int bkp = bk >> 1;                           // 64 or 128 bytes
    const int bkg = bk / GS;                           // groups per bk
    const int half_K = K >> 1;

    // Precompute `prmt.b32` selector once before K-loop:
    // byte0 = a[lt], byte1 = 0, byte2 = b[lt] (= 4 + lt), byte3 = 0
    const uint32_t prmt_sel = static_cast<uint32_t>(lt)
                            | (static_cast<uint32_t>(4 + lt) << 8);

    // Union SMEM: `sX` during K-loop, aliased by `sRed` only at end of BN_CTA==16 K-split!
    union SmemStorage {
        alignas(16) uint16_t sX[2][BM_TILE][SX_STRIDE];
        alignas(16) float sRed[BM_TILE][16];           // 64*16*4 = 4 KiB <= 8.5 KiB min sX!
    };
    __shared__ SmemStorage smem;
    __shared__ int32_t sX1Row[BM_TILE];
    __shared__ int32_t sX2Row[BM_TILE];
    __shared__ int32_t sIsLast;

    // Precompute `sX1Row` & `sX2Row` ONCE before K-loop (zero division inside K-loop!)
    if (tid < BM_TILE) {
        if (tid < M) {
            if (rnn_mode == 2) {
                const int m_step = tid / n_reqs;
                const int m_req = tid - m_step * n_reqs;
                sX1Row[tid] = m_req * k_steps_rnn + m_step;
                sX2Row[tid] = m_req;
            } else if (rnn_mode == 1) {
                sX1Row[tid] = tid;
                sX2Row[tid] = 0;
            } else {
                sX1Row[tid] = tid;
                sX2Row[tid] = -1;
            }
        } else {
            sX1Row[tid] = -1;
            sX2Row[tid] = -1;
        }
    }

    // ---- Helper lambdas for weight address & sX cp.async ----
    auto load_wp_iter_u4 = [&](int it_abs) -> uint4 {
        const int k_step = it_abs >> bk_shift;
        const int kh = it_abs & bk_shift;
        const int64_t off = (static_cast<int64_t>(wp_panel) * total_k_steps + k_step) * wp_bn + wp_row;
        const uint8_t* ptr = WP + off * bkp + (kh << 6) + (lt << 4);
        return ld_cs_u4(ptr);
    };

    auto load_wp_iter_u2 = [&](int it_abs) -> uint2 {
        const int k_step = it_abs >> bk_shift;
        const int kh = it_abs & bk_shift;
        const int64_t off = (static_cast<int64_t>(wp_panel) * total_k_steps + k_step) * wp_bn + wp_row;
        const uint8_t* ptr = WP + off * bkp + (kh << 6) + (kw << 5) + (lt << 3);
        return ld_cs_u2(ptr);
    };

    auto issue_sx_stage = [&](int stage, int it_abs) {
        const int k_abs = it_abs << 7;
        const bool use_x2 = (X2 != nullptr) && (k_abs >= half_K);
        const int k_off = use_x2 ? (k_abs - half_K) : k_abs;
        const uint16_t* base_x = use_x2 ? X2 : X1;
        const int64_t stride_x = use_x2 ? sx2 : sx1;
        #pragma unroll
        for (int c = tid; c < CHUNKS_SX; c += N_THREADS) {
            const int r = c >> 4;                      // 0..BM_TILE-1
            const int col = (c & 15) << 3;             // 0, 8, ..., 120
            const int row_idx = use_x2 ? sX2Row[r] : sX1Row[r];
            const bool ok = (row_idx >= 0);
            const void* gptr = ok ? static_cast<const void*>(base_x + static_cast<int64_t>(row_idx) * stride_x + k_off + col)
                                  : static_cast<const void*>(X1);
            const uint32_t saddr = smem_u32(&smem.sX[stage][r][col]);
            cp_async_ca_16_zfill(saddr, gptr, ok ? 16 : 0);
        }
        cp_async_commit();
    };

    // ---- SAFE PDL: preload static WP[iter 0] BEFORE gdc_wait(), overlap behind prior kernel! ----
    uint4 wp_cur4 = make_uint4(0u, 0u, 0u, 0u);
    uint2 wp_cur2 = make_uint2(0u, 0u);
    if constexpr (K_SPLIT2) {
        wp_cur2 = load_wp_iter_u2(it_beg);
    } else {
        wp_cur4 = load_wp_iter_u4(it_beg);
    }

    // Must wait for upstream kernel before touching dynamic X1/X2!
    if (use_pdl) gdc_wait();
    __syncthreads();  // guarantees sX1Row / sX2Row ready before issue_sx_stage

    // Prime stage 0 `sX`
    issue_sx_stage(0, it_beg);

    // Explicit scalar FP32 accumulators (0 local/stack memory!)
    float d00 = 0.f, d01 = 0.f, d02 = 0.f, d03 = 0.f;
    float d10 = 0.f, d11 = 0.f, d12 = 0.f, d13 = 0.f;
    float d20 = 0.f, d21 = 0.f, d22 = 0.f, d23 = 0.f;
    float d30 = 0.f, d31 = 0.f, d32 = 0.f, d33 = 0.f;

    // Precompute ldmatrix row/col indices for this lane (constant across entire K-loop!)
    const int ldm_row = lane & 15;                     // 0..15
    const int ldm_col = (lane >> 4) << 3;              // 0 or 8

    // ---- Main 2-Stage Software-Pipelined K-Loop ----
    for (int it = 0; it < n_iters; ++it) {
        const int stage = it & 1;
        const int it_abs = it_beg + it;

        // Issue NEXT stage's `sX` cp.async + NEXT `WP` global prefetch BEFORE waiting on current stage!
        uint4 wp_nxt4 = make_uint4(0u, 0u, 0u, 0u);
        uint2 wp_nxt2 = make_uint2(0u, 0u);
        if (it + 1 < n_iters) {
            issue_sx_stage(stage ^ 1, it_abs + 1);
            if constexpr (K_SPLIT2) {
                wp_nxt2 = load_wp_iter_u2(it_abs + 1);
            } else {
                wp_nxt4 = load_wp_iter_u4(it_abs + 1);
            }
            cp_async_wait_group<1>();
        } else {
            cp_async_wait_group<0>();
        }
        __syncthreads();

        // Compute scale/zero-point base for this 128-K iteration
        const int k_step = it_abs >> bk_shift;
        const int kh = it_abs & bk_shift;
        const int g_per_128 = 128 / GS;                // 4 for GS=32, 2 for GS=64
        const int64_t sc_base = ((static_cast<int64_t>(wp_panel) * total_k_steps + k_step) * bkg
                                 + kh * g_per_128) * wp_bn + wp_row;

        if constexpr (K_SPLIT2) {
            // BN_CTA == 16: `kw = 0` runs `ks = 0..3` (k = 0..63), `kw = 1` runs `ks = 4..7` (k = 64..127)
            uint32_t sc2 = 0u, zp2 = 0u;
            #pragma unroll
            for (int ks_loc = 0; ks_loc < 4; ++ks_loc) {
                const int ks = (kw << 2) + ks_loc;
                const int g_in_128 = (GS == 32) ? (ks >> 1) : (ks >> 2);
                const bool new_grp = (GS == 32) ? ((ks_loc & 1) == 0) : (ks_loc == 0);
                if (new_grp) {
                    const int64_t sidx = sc_base + static_cast<int64_t>(g_in_128) * wp_bn;
                    sc2 = f16_bits_to_bf16x2(__ldg(SCP + sidx));
                    if constexpr (!SYM) {
                        zp2 = f16_bits_to_bf16x2(__ldg(ZPP + sidx));
                    }
                }
                const int src_lane = (grp << 2) | ks_loc;
                const uint32_t u_lo = __shfl_sync(0xFFFFFFFFu, wp_cur2.x, src_lane);
                const uint32_t u_hi = __shfl_sync(0xFFFFFFFFu, wp_cur2.y, src_lane);
                const uint32_t b0 = unpack_byte_to_bf16x2<SYM>(u_lo, prmt_sel, sc2, zp2);
                const uint32_t b1 = unpack_byte_to_bf16x2<SYM>(u_hi, prmt_sel, sc2, zp2);

                const int ko = ks << 4;
                {
                    uint32_t a0, a1, a2, a3;
                    ldmatrix_x4_b16(a0, a1, a2, a3, smem_u32(&smem.sX[stage][ldm_row][ko + ldm_col]));
                    mma_m16n8k16_bf16(d00, d01, d02, d03, a0, a1, a2, a3, b0, b1);
                }
                if constexpr (MT >= 2) {
                    uint32_t a0, a1, a2, a3;
                    ldmatrix_x4_b16(a0, a1, a2, a3, smem_u32(&smem.sX[stage][16 + ldm_row][ko + ldm_col]));
                    mma_m16n8k16_bf16(d10, d11, d12, d13, a0, a1, a2, a3, b0, b1);
                }
                if constexpr (MT >= 3) {
                    uint32_t a0, a1, a2, a3;
                    ldmatrix_x4_b16(a0, a1, a2, a3, smem_u32(&smem.sX[stage][32 + ldm_row][ko + ldm_col]));
                    mma_m16n8k16_bf16(d20, d21, d22, d23, a0, a1, a2, a3, b0, b1);
                }
                if constexpr (MT >= 4) {
                    uint32_t a0, a1, a2, a3;
                    ldmatrix_x4_b16(a0, a1, a2, a3, smem_u32(&smem.sX[stage][48 + ldm_row][ko + ldm_col]));
                    mma_m16n8k16_bf16(d30, d31, d32, d33, a0, a1, a2, a3, b0, b1);
                }
            }
            wp_cur2 = wp_nxt2;
        } else {
            // BN_CTA == 32 or 64: each warp runs all 8 slices `ks = 0..7` (`k = 0..127`)
            uint32_t sc2 = 0u, zp2 = 0u;
            #pragma unroll
            for (int ks = 0; ks < 8; ++ks) {
                const int g_in_128 = (GS == 32) ? (ks >> 1) : (ks >> 2);
                const bool new_grp = (GS == 32) ? ((ks & 1) == 0) : ((ks & 3) == 0);
                if (new_grp) {
                    const int64_t sidx = sc_base + static_cast<int64_t>(g_in_128) * wp_bn;
                    sc2 = f16_bits_to_bf16x2(__ldg(SCP + sidx));
                    if constexpr (!SYM) {
                        zp2 = f16_bits_to_bf16x2(__ldg(ZPP + sidx));
                    }
                }
                const int src_lane = (grp << 2) | (ks >> 1);
                const uint32_t u_lo = __shfl_sync(0xFFFFFFFFu, (ks & 1) ? wp_cur4.z : wp_cur4.x, src_lane);
                const uint32_t u_hi = __shfl_sync(0xFFFFFFFFu, (ks & 1) ? wp_cur4.w : wp_cur4.y, src_lane);
                const uint32_t b0 = unpack_byte_to_bf16x2<SYM>(u_lo, prmt_sel, sc2, zp2);
                const uint32_t b1 = unpack_byte_to_bf16x2<SYM>(u_hi, prmt_sel, sc2, zp2);

                const int ko = ks << 4;
                {
                    uint32_t a0, a1, a2, a3;
                    ldmatrix_x4_b16(a0, a1, a2, a3, smem_u32(&smem.sX[stage][ldm_row][ko + ldm_col]));
                    mma_m16n8k16_bf16(d00, d01, d02, d03, a0, a1, a2, a3, b0, b1);
                }
                if constexpr (MT >= 2) {
                    uint32_t a0, a1, a2, a3;
                    ldmatrix_x4_b16(a0, a1, a2, a3, smem_u32(&smem.sX[stage][16 + ldm_row][ko + ldm_col]));
                    mma_m16n8k16_bf16(d10, d11, d12, d13, a0, a1, a2, a3, b0, b1);
                }
                if constexpr (MT >= 3) {
                    uint32_t a0, a1, a2, a3;
                    ldmatrix_x4_b16(a0, a1, a2, a3, smem_u32(&smem.sX[stage][32 + ldm_row][ko + ldm_col]));
                    mma_m16n8k16_bf16(d20, d21, d22, d23, a0, a1, a2, a3, b0, b1);
                }
                if constexpr (MT >= 4) {
                    uint32_t a0, a1, a2, a3;
                    ldmatrix_x4_b16(a0, a1, a2, a3, smem_u32(&smem.sX[stage][48 + ldm_row][ko + ldm_col]));
                    mma_m16n8k16_bf16(d30, d31, d32, d33, a0, a1, a2, a3, b0, b1);
                }
            }
            wp_cur4 = wp_nxt4;
        }
        // Protect `sX[stage]` before next iteration overwrites it
        __syncthreads();
    }

    // ---- BN_CTA == 16 K-split warp reduction via aliased `smem.sRed` (0 extra SMEM!) ----
    if constexpr (K_SPLIT2) {
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

    auto load_bias_f32 = [&](int n_idx) -> float {
        if (!has_bias || n_idx >= N) return 0.0f;
        if (bias_is_f32) {
            return static_cast<const float*>(BIAS)[n_idx];
        }
        return bf16_bits_to_f32(static_cast<const uint16_t*>(BIAS)[n_idx]);
    };

    // ---- Epilogue: Direct (split_k == 1) vs Coherent L2 Split-K Finalize (split_k > 1) ----
    if (split_k == 1) {
        if (!K_SPLIT2 || kw == 0) {
            const int n_loc0 = n_warp * 8 + (lt << 1);
            const int n_g0 = n_cta + n_loc0;
            const int n_g1 = n_g0 + 1;
            const float b_0 = load_bias_f32(n_g0);
            const float b_1 = load_bias_f32(n_g1);

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
        // Safe PDL: signal dependents ONLY after all OUT writes are issued!
        if (use_pdl) {
            __syncthreads();
            if (tid == 0) gdc_launch_dependents();
        }
        return;
    }

    // ---- Coherent Split-K Partial Store (`st.global.cg` -> `__threadfence` -> `atom.acq_rel.gpu`) ----
    const int64_t part_panel_stride = static_cast<int64_t>(split_k) * BM_TILE * BN_CTA;
    const int64_t part_split_stride = static_cast<int64_t>(BM_TILE) * BN_CTA;
    float* p_base = PART + static_cast<int64_t>(pid_n) * part_panel_stride
                         + static_cast<int64_t>(ks_id) * part_split_stride;

    if (!K_SPLIT2 || kw == 0) {
        const int n_loc0 = n_warp * 8 + (lt << 1);
        st_cg_f32x2(p_base + grp * BN_CTA + n_loc0, d00, d01);
        st_cg_f32x2(p_base + (grp + 8) * BN_CTA + n_loc0, d02, d03);
        if constexpr (MT >= 2) {
            st_cg_f32x2(p_base + (16 + grp) * BN_CTA + n_loc0, d10, d11);
            st_cg_f32x2(p_base + (24 + grp) * BN_CTA + n_loc0, d12, d13);
        }
        if constexpr (MT >= 3) {
            st_cg_f32x2(p_base + (32 + grp) * BN_CTA + n_loc0, d20, d21);
            st_cg_f32x2(p_base + (40 + grp) * BN_CTA + n_loc0, d22, d23);
        }
        if constexpr (MT >= 4) {
            st_cg_f32x2(p_base + (48 + grp) * BN_CTA + n_loc0, d30, d31);
            st_cg_f32x2(p_base + (56 + grp) * BN_CTA + n_loc0, d32, d33);
        }
    }

    __syncthreads();
    __threadfence();

    if (tid == 0) {
        const uint32_t prev = atomic_add_acq_rel_gpu(SEM + pid_n, 1u);
        const int last = (static_cast<int>(prev) + 1 == split_k) ? 1 : 0;
        if (last) {
            st_cg_u32(SEM + pid_n, 0u);  // self-reset for CUDA-Graph replay safety!
        }
        sIsLast = last;
    }
    __syncthreads();

    if (sIsLast != 0) {
        const float* p_panel = PART + static_cast<int64_t>(pid_n) * part_panel_stride;
        if constexpr (FUSE_SILU) {
            // Pairwise reduce `(gate, up)` across all splits
            constexpr int HALF_BN = BN_CTA >> 1;
            const int total_pairs = M * HALF_BN;
            for (int idx = tid; idx < total_pairs; idx += N_THREADS) {
                const int m_row = idx / HALF_BN;
                const int c_loc = idx - m_row * HALF_BN;
                const int n_g0 = n_cta + (c_loc << 1);
                const int c_out = n_g0 >> 1;
                if (c_out < (N >> 1)) {
                    float acc_g = load_bias_f32(n_g0);
                    float acc_u = load_bias_f32(n_g0 + 1);
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
                    float acc = load_bias_f32(n_g);
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
        if (use_pdl) {
            __syncthreads();
            if (tid == 0) gdc_launch_dependents();
        }
    }
}

// ============================================================================
// Pruned Dispatch Helper (72 exact SM121a instantiations, zero bloat!)
// ============================================================================
template <int BN_CTA, int MT, bool FUSE_SILU, bool OUT_F32>
static inline int dispatch_gs_sym(
    int gs, int sym, bool use_pdl,
    dim3 grid, cudaStream_t stream,
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
                drafter_u4_tc_v9_kernel<BN_CTA, MT, 32, true, FUSE_SILU, OUT_F32>,
                X1, X2, WP, SCP, ZPP, BIAS, PART, SEM, OUT,
                M, N, K, wp_bn, bk, split_k, iters_per_split,
                sx1, sx2, so, rnn_mode, n_reqs, k_steps_rnn, has_bias, bias_is_f32, pdl_i);
        } else {
            err = launch_kernel_ex(use_pdl, grid, block, 0, stream,
                drafter_u4_tc_v9_kernel<BN_CTA, MT, 32, false, FUSE_SILU, OUT_F32>,
                X1, X2, WP, SCP, ZPP, BIAS, PART, SEM, OUT,
                M, N, K, wp_bn, bk, split_k, iters_per_split,
                sx1, sx2, so, rnn_mode, n_reqs, k_steps_rnn, has_bias, bias_is_f32, pdl_i);
        }
    } else {
        if (sym) {
            err = launch_kernel_ex(use_pdl, grid, block, 0, stream,
                drafter_u4_tc_v9_kernel<BN_CTA, MT, 64, true, FUSE_SILU, OUT_F32>,
                X1, X2, WP, SCP, ZPP, BIAS, PART, SEM, OUT,
                M, N, K, wp_bn, bk, split_k, iters_per_split,
                sx1, sx2, so, rnn_mode, n_reqs, k_steps_rnn, has_bias, bias_is_f32, pdl_i);
        } else {
            err = launch_kernel_ex(use_pdl, grid, block, 0, stream,
                drafter_u4_tc_v9_kernel<BN_CTA, MT, 64, false, FUSE_SILU, OUT_F32>,
                X1, X2, WP, SCP, ZPP, BIAS, PART, SEM, OUT,
                M, N, K, wp_bn, bk, split_k, iters_per_split,
                sx1, sx2, so, rnn_mode, n_reqs, k_steps_rnn, has_bias, bias_is_f32, pdl_i);
        }
    }
    return (err == cudaSuccess) ? 0 : static_cast<int>(err);
}

}  // namespace

extern "C" {

int drafter_u4_tc_v9_version() {
    return 9;
}

int drafter_u4_tc_v9_fwd(
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
    int bn_cta,              // requested CTA tile: 16, 32, or 64 (0 = auto)
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
    uint64_t stream_u64
) {
    if (M <= 0 || M > 64 || N <= 0 || K <= 0) return -1;
    if ((K & 127) != 0 || (bk != 128 && bk != 256) || (K % bk) != 0) return -2;
    if (gs != 32 && gs != 64) return -3;
    if (wp_bn != 16 && wp_bn != 32 && wp_bn != 64) return -4;
    if (split_k < 1) return -5;
    if (split_k > 1 && (part == nullptr || sem == nullptr)) return -6;
    if (rnn_mode != 0 && x2 == nullptr) return -7;
    if (!sym && zpp == nullptr) return -8;
    // `cp.async` 16-byte alignment checks on X1 / X2, and 4-byte BF16x2 store alignment on OUT
    if ((reinterpret_cast<uintptr_t>(x1) & 15u) != 0 || (sx1 & 7) != 0) return -9;
    if (x2 != nullptr && ((reinterpret_cast<uintptr_t>(x2) & 15u) != 0 || (sx2 & 7) != 0)) return -10;
    if (fuse_silu && ((N & 1) != 0)) return -11;
    if (!out_f32 && !fuse_silu && (((reinterpret_cast<uintptr_t>(out) & 3u) != 0) || ((so & 1) != 0))) return -17;

    const int mt = (M + 15) >> 4;  // 1..4

    // Auto-select optimal GB10 CTA tile if bn_cta == 0:
    // - wp_bn == 16 (`kv_proj_ctx`): BN_CTA = 16 (4-warp K-split)
    // - wp_bn == 64 & !fuse_silu & !out_f32 & mt >= 3 & N >= 8192 (`draft_head` M=33..64): BN_CTA = 64 (8-warp)
    // - otherwise: BN_CTA = 32 (4-warp; splits gate_up into exact 144 = 3x48 waves!)
    int eff_bn = bn_cta;
    if (eff_bn == 0) {
        if (wp_bn == 16) {
            eff_bn = 16;
        } else if (wp_bn == 64 && !fuse_silu && !out_f32 && mt >= 3 && N >= 8192) {
            eff_bn = 64;
        } else {
            eff_bn = 32;
        }
    }
    if (eff_bn != 16 && eff_bn != 32 && eff_bn != 64) return -12;
    if (eff_bn > wp_bn || (wp_bn % eff_bn) != 0) return -13;
    // Validate against pruned instantiations
    if (eff_bn == 16 && (fuse_silu || out_f32)) return -14;
    if (eff_bn == 64 && (fuse_silu || out_f32 || mt < 3)) return -15;
    if (fuse_silu && out_f32) return -16;

    const int total_iters = K >> 7;
    const int bk_iters = bk >> 7;  // 1 or 2
    int iters_per_split = ((total_iters + split_k - 1) / split_k);
    // Align split boundaries to `bk` when possible for optimal L2 prefetch
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

    #define CALL_TC9(BN_V, MT_V, SILU_V, F32_V) \
        dispatch_gs_sym<BN_V, MT_V, SILU_V, F32_V>( \
            gs, sym, pdl_b, grid, stream, \
            x1p, x2p, wpp, scp_p, zpp_p, bias, part_p, sem_p, out, \
            M, N, K, wp_bn, bk, eff_split_k, iters_per_split, \
            sx1, sx2, so, rnn_mode, n_reqs, k_steps_rnn, has_bias, bias_is_f32)

    if (eff_bn == 16) {
        if (mt == 1) return CALL_TC9(16, 1, false, false);
        if (mt == 2) return CALL_TC9(16, 2, false, false);
        if (mt == 3) return CALL_TC9(16, 3, false, false);
        return CALL_TC9(16, 4, false, false);
    }
    if (eff_bn == 64) {
        if (mt == 3) return CALL_TC9(64, 3, false, false);
        return CALL_TC9(64, 4, false, false);
    }
    // eff_bn == 32
    if (fuse_silu) {
        if (mt == 1) return CALL_TC9(32, 1, true, false);
        if (mt == 2) return CALL_TC9(32, 2, true, false);
        if (mt == 3) return CALL_TC9(32, 3, true, false);
        return CALL_TC9(32, 4, true, false);
    }
    if (out_f32) {
        if (mt == 1) return CALL_TC9(32, 1, false, true);
        if (mt == 2) return CALL_TC9(32, 2, false, true);
        if (mt == 3) return CALL_TC9(32, 3, false, true);
        return CALL_TC9(32, 4, false, true);
    }
    if (mt == 1) return CALL_TC9(32, 1, false, false);
    if (mt == 2) return CALL_TC9(32, 2, false, false);
    if (mt == 3) return CALL_TC9(32, 3, false, false);
    return CALL_TC9(32, 4, false, false);

    #undef CALL_TC9
}

}  // extern "C"
