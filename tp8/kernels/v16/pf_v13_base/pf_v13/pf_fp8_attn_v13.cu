// =================================================================================================
// Wave-V6 Unified SM121a FP8 `Q@K^T + P@V` Prefill Attention Library (`pf_fp8_attn_v6.cu`)
// Target: NVIDIA GB10 (SM121a, 48 SMs, 99 KiB SMEM/SM, 24 MiB L2, 244 GB/s LPDDR5x UMA)
//
// Kernels & Entrypoints (`libpf_fp8_attn_v6.so`):
//   1. `global_sage2_prefix_v6_sm121_fwd` (+ drop-in `v5` / `v3` & `score_kmax_blocks_fp8_sm121_fwd`):
//      - Pre-quantized FP8 Shadow K (`k_bt_mode = 0, 1, 2` Compact Shadow K 100% compatible!)
//      - `Q @ K^T`: SM121a `mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16` (4x BF16 TC)
//      - `P @ V`  : SM121a `mma.sync.aligned.m16n8k32.row.col.{f32,f16}.e4m3.e4m3.{f32,f16}` (2x-4x TC)
//        with dynamic row-tile-max `tm` P scaling (zero subnormal underflow), 8-register in-warp
//        quad repack (`2x __shfl_sync + __byte_perm`), per-8-channel (`dt=0..15`) V scale,
//        8 KiB transposed `V_T [128, 64]` SMEM (-8 KiB SMEM/CTA vs v5), and non-draining `cp.async`
//        3-stage K pipeline (never calls `cp.async.wait_group 0` mid-loop!).
//   2. `local_swa_fp8_otf_rel_v6_sm121_fwd` (+ drop-in `local_swa_otf_rel_bf16_sm121_fwd`):
//      - 40 Local SWA layers (`W=1024`): GQA-fused `H_PER_CTA=2, BM_TOK=64, BN=64` (shares K & V
//        across 2 Q-heads in SMEM = 2.0x less KV load & online quantize; 17-18 BN=64 tiles),
//        per-query x per-token FP8 `Q@K^T`, exact 0-ULP BF16 OTF `r_clip @ RelProjT`, dynamic-`tm`
//        FP8 `P@V`, and fused `Gate + Sink` epilogue.
//   3. `global_tail_fp8_otf_rel_v6_sm121_fwd` (+ drop-in `global_tail_otf_rel_bf16_sm121_fwd`):
//      - 8 Global layers (`E=1024`): GQA-fused `H_PER_CTA=2, BM_TOK=64, BN=64` (upgrades `tail_bn`
//        from `32 -> 64` = 2.0x fewer K-tiles + 2.0x less KV load), online LSE continuation from
//        `Sage2` `(ACC_IN, M_IN, L_IN)` with exact `(prefix_bm, prefix_bn)` alignment, bias-free
//        fast path on `d >= E` prefix tiles, exact BF16 OTF `r_clip_tau @ RelProjT`, FP8 `Q@K^T + P@V`,
//        and fused `Gate + Sink` epilogue.
//
// Safety & Codegen Invariants:
//   - Strictly prefill-only (`max_seqlen_q >= min_seqlen_q` [default 1024] & `!capturing`).
//   - Compile WITHOUT `--use_fast_math` (preserves exact `-INF` arithmetic & `ld.shared` broadcast).
//   - 0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads across ALL kernels.
// =================================================================================================

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace {

static constexpr int SAGE_BM = 128;
static constexpr int D_HEAD = 128;
static constexpr int D_REL = 16;
static constexpr int WARPS = 8;
static constexpr int THREADS = WARPS * 32;  // 256 threads
static constexpr float QK_MAX = 22.0f;      // 128 * 22^2 = 61,952 < 65,504 (zero FP16 overflow!)
static constexpr float INV_QK_MAX = 1.0f / 22.0f;
static constexpr float PV_MAX_F32 = 448.0f;
static constexpr float INV_PV_MAX_F32 = 1.0f / 448.0f;
static constexpr float PV_MAX_F16 = 31.0f;  // 64 * 31^2 = 61,504 < 65,504 (zero FP16 overflow!)
static constexpr float INV_PV_MAX_F16 = 1.0f / 31.0f;
static constexpr float NEG_INF_GUARD = -1.0e30f;
static constexpr float LOG2E_F = 1.44269504088896340736f;
static constexpr float LN2_F = 0.69314718055994530942f;

// =================================================================================================
// Shared SM121a PTX Primitives (Zero Stack, Non-Hoistable Uniform LDS Broadcasts)
// =================================================================================================

__device__ __forceinline__ float div_rn_ftz(float a, float b) {
    float r;
    asm("div.rn.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

__device__ __forceinline__ float div_full_f32(float a, float b) {
    float y;
    asm("div.full.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
    return y;
}

__device__ __forceinline__ float fast_exp2f(float x) {
    float r;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}

__device__ __forceinline__ float ex2_approx_f32(float x) {
    float y;
    asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float rcp_approx_f32(float x) {
    float y;
    asm("rcp.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float rsqrt_approx_f32(float x) {
    float y;
    asm("rsqrt.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float triton_sigmoid(float x) {
    const float neg_x_log2e = __fmul_rn(x, -LOG2E_F);
    const float e = ex2_approx_f32(neg_x_log2e);
    return rcp_approx_f32(__fadd_rn(e, 1.0f));
}

__device__ __forceinline__ float bf16_to_f32(uint16_t b) {
    float f;
    asm("cvt.f32.bf16 %0, %1;" : "=f"(f) : "h"(b));
    return f;
}

__device__ __forceinline__ uint16_t f32_to_bf16_u16(float f) {
    __nv_bfloat16 b = __float2bfloat16_rn(f);
    return *reinterpret_cast<const uint16_t*>(&b);
}

__device__ __forceinline__ uint32_t pack_bf16x2(uint16_t lo, uint16_t hi) {
    return static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
}

__device__ __forceinline__ uint16_t f32x2_to_e4m3x2_rn(float lo, float hi) {
    uint16_t r;
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(r) : "f"(lo), "f"(hi));
    return r;
}

__device__ __forceinline__ uint32_t f32x4_to_e4m3x4_rn(float v0, float v1, float v2, float v3) {
    const uint16_t lo = f32x2_to_e4m3x2_rn(v0, v1);
    const uint16_t hi = f32x2_to_e4m3x2_rn(v2, v3);
    return static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
}

__device__ __forceinline__ uint32_t hmax2_u32(uint32_t a, uint32_t b) {
    uint32_t r;
    asm volatile("max.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
    return r;
}

__device__ __forceinline__ uint4 ld_ca_u4(const void* ptr) {
    uint4 v;
    asm volatile("ld.global.ca.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(ptr));
    return v;
}

__device__ __forceinline__ uint2 ld_ca_u2(const void* ptr) {
    uint2 v;
    asm volatile("ld.global.ca.v2.u32 {%0, %1}, [%2];"
                 : "=r"(v.x), "=r"(v.y) : "l"(ptr));
    return v;
}

__device__ __forceinline__ uint64_t ld_smem_u64_nocse(const void* smem_ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    uint64_t v;
    asm volatile("ld.shared.u64 %0, [%1];" : "=l"(v) : "r"(addr));
    return v;
}

__device__ __forceinline__ uint32_t ld_smem_u32_nocse(const void* smem_ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    uint32_t v;
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(addr));
    return v;
}

__device__ __forceinline__ int ld_smem_i32_nocse(const int* smem_ptr) {
    return static_cast<int>(ld_smem_u32_nocse(smem_ptr));
}

__device__ __forceinline__ int64_t ld_smem_i64_nocse(const int64_t* smem_ptr) {
    return static_cast<int64_t>(ld_smem_u64_nocse(smem_ptr));
}

__device__ __forceinline__ float ld_smem_f32_nocse(const float* smem_ptr) {
    return __uint_as_float(ld_smem_u32_nocse(smem_ptr));
}

template <typename T>
__device__ __forceinline__ const T* ld_smem_ptr_nocse(const T* const* smem_ptr) {
    return reinterpret_cast<const T*>(static_cast<uintptr_t>(ld_smem_u64_nocse(smem_ptr)));
}

template <typename T>
__device__ __forceinline__ T* ld_smem_mut_ptr_nocse(T* const* smem_ptr) {
    return reinterpret_cast<T*>(static_cast<uintptr_t>(ld_smem_u64_nocse(smem_ptr)));
}

// SM121a Tensor Core MMAs
__device__ __forceinline__ void mma_m16n8k32_f16_e4m3(
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

__device__ __forceinline__ void mma_m16n8k32_f32_e4m3(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3
) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3)
    );
}

__device__ __forceinline__ void mma_m16n8k16_f32_bf16(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3)
    );
}

__device__ __forceinline__ void cp_async_cg_16(uint32_t smem_addr, const void* gmem_ptr) {
    asm volatile(
        "cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n"
        :: "r"(smem_addr), "l"(gmem_ptr)
        : "memory"
    );
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}

__device__ __forceinline__ void ldmatrix_x4(
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3,
    uint32_t smem_addr
) {
    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(smem_addr)
    );
}

// =================================================================================================
// Low-Register Q Loader & V_T [128, 64] FP8 Transposed SMEM Quantizer + Quad-Repack P@V Engine
// =================================================================================================

// 2-pass L1-cached Q row quantizer: uses ONLY 4 raw registers instead of 16!
__device__ __forceinline__ void load_quant_q_row_v6(
    const __nv_bfloat16* __restrict__ row_ptr,
    bool valid,
    int t,
    float scale_mult,
    uint32_t (&q_lo)[4],
    uint32_t (&q_hi)[4],
    float& qs_out
) {
    float amax = 0.0f;
    if (valid) {
        #pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            const uint2 u_lo = ld_ca_u2(row_ptr + ks * 32 + 4 * t);
            const uint2 u_hi = ld_ca_u2(row_ptr + ks * 32 + 16 + 4 * t);
            const float2 f0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u_lo.x));
            const float2 f1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u_lo.y));
            const float2 f2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u_hi.x));
            const float2 f3 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u_hi.y));
            amax = fmaxf(amax, fmaxf(fmaxf(fabsf(f0.x), fabsf(f0.y)), fmaxf(fabsf(f1.x), fabsf(f1.y))));
            amax = fmaxf(amax, fmaxf(fmaxf(fabsf(f2.x), fabsf(f2.y)), fmaxf(fabsf(f3.x), fabsf(f3.y))));
        }
    }
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 2));
    amax = fmaxf(amax, 1.0e-12f);
    const float inv = div_rn_ftz(QK_MAX, amax);
    qs_out = amax * INV_QK_MAX * scale_mult;

    #pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
        if (valid) {
            const uint2 u_lo = ld_ca_u2(row_ptr + ks * 32 + 4 * t);
            const uint2 u_hi = ld_ca_u2(row_ptr + ks * 32 + 16 + 4 * t);
            const float2 f0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u_lo.x));
            const float2 f1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u_lo.y));
            const float2 f2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u_hi.x));
            const float2 f3 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u_hi.y));
            q_lo[ks] = f32x4_to_e4m3x4_rn(f0.x * inv, f0.y * inv, f1.x * inv, f1.y * inv);
            q_hi[ks] = f32x4_to_e4m3x4_rn(f2.x * inv, f2.y * inv, f3.x * inv, f3.y * inv);
        } else {
            q_lo[ks] = 0u;
            q_hi[ks] = 0u;
        }
    }
}

__device__ __forceinline__ void load_quant_q_regs_v6(
    const __nv_bfloat16* __restrict__ Q,
    int64_t sq_t, int64_t sq_h,
    int q0, int h, int r0_loc, int r1_loc,
    bool r0_valid, bool r1_valid,
    int t, float scale_mult,
    uint32_t (&q_a0)[4], uint32_t (&q_a1)[4], uint32_t (&q_a2)[4], uint32_t (&q_a3)[4],
    float& qs_0, float& qs_1
) {
    load_quant_q_row_v6(
        Q + static_cast<int64_t>(q0 + r0_loc) * sq_t + static_cast<int64_t>(h) * sq_h,
        r0_valid, t, scale_mult, q_a0, q_a2, qs_0
    );
    load_quant_q_row_v6(
        Q + static_cast<int64_t>(q0 + r1_loc) * sq_t + static_cast<int64_t>(h) * sq_h,
        r1_valid, t, scale_mult, q_a1, q_a3, qs_1
    );
}

// Bank-conflict-free 16B chunk swizzle for V_T [D=128, BN=64] uint8_t:
// Within each ldmatrix.x4 sub-matrix (`d_row >> 3 = const`), `(d_row >> 1) & 3` permutes
// {0,1,2,3} bijectively across the 8 rows -> 8 distinct 16B slices = 0 bank conflicts!
__device__ __forceinline__ int vt_swiz_chunk(int d_row, int k_chunk) {
    return k_chunk ^ (((d_row >> 1) ^ (d_row >> 3)) & 3);
}

// Pass 1 of V64 -> V_T FP8: coalesced 256B/row L1-cached `ld.global.ca.v4.u32` amax per 8-channel group.
// Uses only 4 temporary GPRs (`uint4 v`) so live `acc[16][4]` never spills!
__device__ __forceinline__ void v64_pass1_amax(
    const __nv_bfloat16* __restrict__ v_base,
    int64_t sv_s,
    int valid_k_rows,
    uint32_t* __restrict__ smem_vs_bits,
    int tid
) {
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int r_sub = lane >> 4;   // 0..1
    const int c_grp = lane & 15;   // dt = 0..15 (8 channels = 16 B per lane -> 256 B coalesced!)

    float vmax = 0.0f;
    #pragma unroll
    for (int p = 0; p < 4; ++p) {
        const int k_row = warp * 8 + p * 2 + r_sub;
        if (k_row < valid_k_rows) {
            const uint4 v = ld_ca_u4(v_base + static_cast<int64_t>(k_row) * sv_s + c_grp * 8);
            const float2 f0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v.x));
            const float2 f1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v.y));
            const float2 f2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v.z));
            const float2 f3 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v.w));
            vmax = fmaxf(vmax, fmaxf(fmaxf(fabsf(f0.x), fabsf(f0.y)), fmaxf(fabsf(f1.x), fabsf(f1.y))));
            vmax = fmaxf(vmax, fmaxf(fmaxf(fabsf(f2.x), fabsf(f2.y)), fmaxf(fabsf(f3.x), fabsf(f3.y))));
        }
    }
    vmax = fmaxf(vmax, __shfl_xor_sync(0xffffffff, vmax, 16));
    if (r_sub == 0) {
        atomicMax(&smem_vs_bits[c_grp], __float_as_uint(vmax));
    }
}

// Pass 2 of V64 -> V_T FP8: L1-hit load + in-warp 8x8 byte transpose + swizzled `uint2` store to SMEM!
__device__ __forceinline__ void v64_pass2_quant_transpose(
    const __nv_bfloat16* __restrict__ v_base,
    int64_t sv_s,
    int valid_k_rows,
    const float* __restrict__ smem_vs_inv,
    uint32_t vt_base_sptr,
    int tid
) {
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int r_sub = lane >> 4;   // 0 (even keys 0,2,4,6) or 1 (odd keys 1,3,5,7)
    const int c_grp = lane & 15;   // dt = 0..15
    const float inv_v = ld_smem_f32_nocse(&smem_vs_inv[c_grp]);
    const int k_chunk = warp >> 1; // 0..3 (16-key chunk along K=64)
    const int half8 = (warp & 1) * 8;

    #pragma unroll
    for (int jh = 0; jh < 2; ++jh) {
        float c0_k0 = 0.f, c1_k0 = 0.f, c2_k0 = 0.f, c3_k0 = 0.f;
        float c0_k1 = 0.f, c1_k1 = 0.f, c2_k1 = 0.f, c3_k1 = 0.f;
        float c0_k2 = 0.f, c1_k2 = 0.f, c2_k2 = 0.f, c3_k2 = 0.f;
        float c0_k3 = 0.f, c1_k3 = 0.f, c2_k3 = 0.f, c3_k3 = 0.f;

        {
            const int kr = warp * 8 + 0 + r_sub;
            if (kr < valid_k_rows) {
                const uint2 u = ld_ca_u2(v_base + static_cast<int64_t>(kr) * sv_s + c_grp * 8 + jh * 4);
                const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.x));
                const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.y));
                c0_k0 = a.x * inv_v; c1_k0 = a.y * inv_v; c2_k0 = b.x * inv_v; c3_k0 = b.y * inv_v;
            }
        }
        {
            const int kr = warp * 8 + 2 + r_sub;
            if (kr < valid_k_rows) {
                const uint2 u = ld_ca_u2(v_base + static_cast<int64_t>(kr) * sv_s + c_grp * 8 + jh * 4);
                const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.x));
                const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.y));
                c0_k1 = a.x * inv_v; c1_k1 = a.y * inv_v; c2_k1 = b.x * inv_v; c3_k1 = b.y * inv_v;
            }
        }
        {
            const int kr = warp * 8 + 4 + r_sub;
            if (kr < valid_k_rows) {
                const uint2 u = ld_ca_u2(v_base + static_cast<int64_t>(kr) * sv_s + c_grp * 8 + jh * 4);
                const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.x));
                const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.y));
                c0_k2 = a.x * inv_v; c1_k2 = a.y * inv_v; c2_k2 = b.x * inv_v; c3_k2 = b.y * inv_v;
            }
        }
        {
            const int kr = warp * 8 + 6 + r_sub;
            if (kr < valid_k_rows) {
                const uint2 u = ld_ca_u2(v_base + static_cast<int64_t>(kr) * sv_s + c_grp * 8 + jh * 4);
                const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.x));
                const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.y));
                c0_k3 = a.x * inv_v; c1_k3 = a.y * inv_v; c2_k3 = b.x * inv_v; c3_k3 = b.y * inv_v;
            }
        }

        const uint32_t my4_0 = f32x4_to_e4m3x4_rn(c0_k0, c0_k1, c0_k2, c0_k3);
        const uint32_t my4_1 = f32x4_to_e4m3x4_rn(c1_k0, c1_k1, c1_k2, c1_k3);
        const uint32_t my4_2 = f32x4_to_e4m3x4_rn(c2_k0, c2_k1, c2_k2, c2_k3);
        const uint32_t my4_3 = f32x4_to_e4m3x4_rn(c3_k0, c3_k1, c3_k2, c3_k3);

        // Exchange between r_sub=0 (even keys) and r_sub=1 (odd keys):
        // r_sub=0 writes channels (0, 1); r_sub=1 writes channels (2, 3)!
        const uint32_t sh0 = __shfl_xor_sync(0xffffffff, (r_sub == 0) ? my4_2 : my4_0, 16);
        const uint32_t sh1 = __shfl_xor_sync(0xffffffff, (r_sub == 0) ? my4_3 : my4_1, 16);

        const uint32_t ev_a = (r_sub == 0) ? my4_0 : sh0;
        const uint32_t od_a = (r_sub == 0) ? sh0   : my4_2;
        const uint32_t ev_b = (r_sub == 0) ? my4_1 : sh1;
        const uint32_t od_b = (r_sub == 0) ? sh1   : my4_3;

        const int d_a = c_grp * 8 + jh * 4 + r_sub * 2;
        const int d_b = d_a + 1;

        const uint2 out_a = make_uint2(__byte_perm(ev_a, od_a, 0x5140u), __byte_perm(ev_a, od_a, 0x7362u));
        const uint2 out_b = make_uint2(__byte_perm(ev_b, od_b, 0x5140u), __byte_perm(ev_b, od_b, 0x7362u));

        const uint32_t addr_a = vt_base_sptr + static_cast<uint32_t>(d_a * 64 + vt_swiz_chunk(d_a, k_chunk) * 16 + half8);
        const uint32_t addr_b = vt_base_sptr + static_cast<uint32_t>(d_b * 64 + vt_swiz_chunk(d_b, k_chunk) * 16 + half8);
        asm volatile("st.shared.v2.u32 [%0], {%1, %2};" :: "r"(addr_a), "r"(out_a.x), "r"(out_a.y) : "memory");
        asm volatile("st.shared.v2.u32 [%0], {%1, %2};" :: "r"(addr_b), "r"(out_b.x), "r"(out_b.y) : "memory");
    }
}

// Quad-repack 2 adjacent N=8 score pairs `(u0_a, u0_b)` and `(u1_a, u1_b)` (e4m3x2) into
// `m16n8k32` Matrix-A quad operands `(pa_r0, pa_r1)` across the 4 group lanes `t = 0..3`!
__device__ __forceinline__ void repack_p_quad_half16(
    uint16_t u0_a, uint16_t u0_b,
    uint16_t u1_a, uint16_t u1_b,
    int gid, int t,
    uint32_t& pa_r0, uint32_t& pa_r1
) {
    const uint32_t w0 = static_cast<uint32_t>(u0_a) | (static_cast<uint32_t>(u0_b) << 16);
    const uint32_t w1 = static_cast<uint32_t>(u1_a) | (static_cast<uint32_t>(u1_b) << 16);
    const int src_lo = (gid << 2) | ((t & 1) << 1);
    const int src_hi = src_lo | 1;
    const uint32_t perm_sel = (t < 2) ? 0x5410u : 0x7632u;
    pa_r0 = __byte_perm(__shfl_sync(0xffffffff, w0, src_lo), __shfl_sync(0xffffffff, w0, src_hi), perm_sel);
    pa_r1 = __byte_perm(__shfl_sync(0xffffffff, w1, src_lo), __shfl_sync(0xffffffff, w1, src_hi), perm_sel);
}

// Execute 32 SM121a FP8 `m16n8k32` MMAs (`P [16, 64] @ V_T^T [64, 128]`) and scale into `acc[16][4]`
template <bool PV_F16_ACC>
__device__ __forceinline__ void execute_pv_fp8_m16n8k32(
    const uint32_t (&pa)[2][4],
    float w_row0, float w_row1,
    uint32_t vt_sptr,
    const float* __restrict__ smem_vs_tile,
    int lane,
    float (&acc)[16][4]
) {
    const int sub = lane >> 3;
    const int r_in = lane & 7;

    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int dt0 = 2 * j;
        const int d_row = (dt0 + (sub >> 1)) * 8 + r_in;
        const float vs0 = ld_smem_f32_nocse(&smem_vs_tile[dt0]);
        const float vs1 = ld_smem_f32_nocse(&smem_vs_tile[dt0 + 1]);
        const float sc0_0 = w_row0 * vs0;
        const float sc1_0 = w_row1 * vs0;
        const float sc0_1 = w_row0 * vs1;
        const float sc1_1 = w_row1 * vs1;

        if constexpr (!PV_F16_ACC) {
            float t0_0 = 0.f, t0_1 = 0.f, t0_2 = 0.f, t0_3 = 0.f;
            float t1_0 = 0.f, t1_1 = 0.f, t1_2 = 0.f, t1_3 = 0.f;
            #pragma unroll
            for (int ks = 0; ks < 2; ++ks) {
                const int k_chunk = ks * 2 + (sub & 1);
                const uint32_t v_addr = vt_sptr + static_cast<uint32_t>(d_row * 64 + vt_swiz_chunk(d_row, k_chunk) * 16);
                uint32_t vb0_0, vb1_0, vb0_1, vb1_1;
                ldmatrix_x4(vb0_0, vb1_0, vb0_1, vb1_1, v_addr);
                mma_m16n8k32_f32_e4m3(
                    t0_0, t0_1, t0_2, t0_3,
                    pa[ks][0], pa[ks][1], pa[ks][2], pa[ks][3],
                    vb0_0, vb1_0,
                    t0_0, t0_1, t0_2, t0_3);
                mma_m16n8k32_f32_e4m3(
                    t1_0, t1_1, t1_2, t1_3,
                    pa[ks][0], pa[ks][1], pa[ks][2], pa[ks][3],
                    vb0_1, vb1_1,
                    t1_0, t1_1, t1_2, t1_3);
            }
            acc[dt0][0]     = fmaf(t0_0, sc0_0, acc[dt0][0]);
            acc[dt0][1]     = fmaf(t0_1, sc0_0, acc[dt0][1]);
            acc[dt0][2]     = fmaf(t0_2, sc1_0, acc[dt0][2]);
            acc[dt0][3]     = fmaf(t0_3, sc1_0, acc[dt0][3]);
            acc[dt0 + 1][0] = fmaf(t1_0, sc0_1, acc[dt0 + 1][0]);
            acc[dt0 + 1][1] = fmaf(t1_1, sc0_1, acc[dt0 + 1][1]);
            acc[dt0 + 1][2] = fmaf(t1_2, sc1_1, acc[dt0 + 1][2]);
            acc[dt0 + 1][3] = fmaf(t1_3, sc1_1, acc[dt0 + 1][3]);
        } else {
            uint32_t h0_0 = 0u, h1_0 = 0u, h0_1 = 0u, h1_1 = 0u;
            #pragma unroll
            for (int ks = 0; ks < 2; ++ks) {
                const int k_chunk = ks * 2 + (sub & 1);
                const uint32_t v_addr = vt_sptr + static_cast<uint32_t>(d_row * 64 + vt_swiz_chunk(d_row, k_chunk) * 16);
                uint32_t vb0_0, vb1_0, vb0_1, vb1_1;
                ldmatrix_x4(vb0_0, vb1_0, vb0_1, vb1_1, v_addr);
                mma_m16n8k32_f16_e4m3(
                    h0_0, h1_0,
                    pa[ks][0], pa[ks][1], pa[ks][2], pa[ks][3],
                    vb0_0, vb1_0, h0_0, h1_0);
                mma_m16n8k32_f16_e4m3(
                    h0_1, h1_1,
                    pa[ks][0], pa[ks][1], pa[ks][2], pa[ks][3],
                    vb0_1, vb1_1, h0_1, h1_1);
            }
            const float2 f0_0 = __half22float2(*reinterpret_cast<const __half2*>(&h0_0));
            const float2 f1_0 = __half22float2(*reinterpret_cast<const __half2*>(&h1_0));
            const float2 f0_1 = __half22float2(*reinterpret_cast<const __half2*>(&h0_1));
            const float2 f1_1 = __half22float2(*reinterpret_cast<const __half2*>(&h1_1));
            acc[dt0][0]     = fmaf(f0_0.x, sc0_0, acc[dt0][0]);
            acc[dt0][1]     = fmaf(f0_0.y, sc0_0, acc[dt0][1]);
            acc[dt0][2]     = fmaf(f1_0.x, sc1_0, acc[dt0][2]);
            acc[dt0][3]     = fmaf(f1_0.y, sc1_0, acc[dt0][3]);
            acc[dt0 + 1][0] = fmaf(f0_1.x, sc0_1, acc[dt0 + 1][0]);
            acc[dt0 + 1][1] = fmaf(f0_1.y, sc0_1, acc[dt0 + 1][1]);
            acc[dt0 + 1][2] = fmaf(f1_1.x, sc1_1, acc[dt0 + 1][2]);
            acc[dt0 + 1][3] = fmaf(f1_1.y, sc1_1, acc[dt0 + 1][3]);
        }
    }
}

// =================================================================================================
// PART 1: Upgraded `Sage2 v6` Global Prefix Kernel (FP8 Q@K^T + Dynamic-`tm` FP8 P@V)
// =================================================================================================

struct alignas(16) SageV6CtaParams {
    const int32_t* bt_row;
    const int32_t* k_bt_row;
    const int32_t* sel_row;
    const uint8_t* K_fp8_h;
    const float*   KS64_h;
    const __nv_bfloat16* V_h;
    int64_t sk8_b;
    int64_t sk8_s;
    int64_t sks64_b;
    int64_t sv_b;
    int64_t sv_s;
    int64_t cur_v_blk;
    uint32_t k_base_sptr;
    uint32_t vt_base_sptr;
    int bs_shift;
    int bs_mask;
    int tpb_shift;
    int tpb_mask;
    int num_tiles;
    int q0;
    int qlen;
    int i_lo;
    int h;
    int cur_slot;
    float ks_tile[2];
    uint32_t vs_bits[16];
    float vs_tile[16];
    float vs_inv[16];
};

template <bool SPARSE_SEL, int K_BT_MODE = 0>
struct TileLocatorV6 {
    __device__ __forceinline__ static int logical_blk_and_slot(
        int tile_idx, int BN, int bs_shift, int bs_mask,
        int tpb_shift, int tpb_mask,
        const int32_t* __restrict__ sel_row,
        int& slot
    ) {
        if constexpr (SPARSE_SEL) {
            const int b_i = tile_idx >> tpb_shift;
            const int sub_t = tile_idx & tpb_mask;
            slot = sub_t * BN;
            return sel_row[b_i];
        } else {
            const int n0 = tile_idx * BN;
            slot = n0 & bs_mask;
            return n0 >> bs_shift;
        }
    }

    __device__ __forceinline__ static void locate_k(
        int tile_idx, int BN, int bs_shift, int bs_mask,
        int tpb_shift, int tpb_mask,
        const int32_t* __restrict__ bt_row,
        const int32_t* __restrict__ k_bt_row,
        const int32_t* __restrict__ sel_row,
        int64_t& k_blk, int& slot
    ) {
        const int lb = logical_blk_and_slot(
            tile_idx, BN, bs_shift, bs_mask, tpb_shift, tpb_mask, sel_row, slot);
        if constexpr (K_BT_MODE == 0) {
            k_blk = static_cast<int64_t>(bt_row[lb]);
        } else if constexpr (K_BT_MODE == 1) {
            k_blk = static_cast<int64_t>(lb);
        } else {
            k_blk = static_cast<int64_t>(k_bt_row[lb]);
        }
    }

    __device__ __forceinline__ static void locate_kv(
        int tile_idx, int BN, int bs_shift, int bs_mask,
        int tpb_shift, int tpb_mask,
        const int32_t* __restrict__ bt_row,
        const int32_t* __restrict__ k_bt_row,
        const int32_t* __restrict__ sel_row,
        int64_t& k_blk, int64_t& v_blk, int& slot
    ) {
        const int lb = logical_blk_and_slot(
            tile_idx, BN, bs_shift, bs_mask, tpb_shift, tpb_mask, sel_row, slot);
        if constexpr (K_BT_MODE == 0) {
            const int64_t b = static_cast<int64_t>(bt_row[lb]);
            k_blk = b;
            v_blk = b;
        } else if constexpr (K_BT_MODE == 1) {
            k_blk = static_cast<int64_t>(lb);
            v_blk = static_cast<int64_t>(bt_row[lb]);
        } else {
            k_blk = static_cast<int64_t>(k_bt_row[lb]);
            v_blk = static_cast<int64_t>(bt_row[lb]);
        }
    }
};

template <int BN>
__device__ __forceinline__ void issue_k_async_v6(
    const uint8_t* __restrict__ K_fp8_h,
    int64_t blk, int slot,
    int64_t sk8_b, int64_t sk8_s,
    uint32_t smem_k_stage_sptr,
    int tid
) {
    const uint8_t* k_base = K_fp8_h + blk * sk8_b + static_cast<int64_t>(slot) * sk8_s;
    constexpr int ITERS = BN / 32;
    #pragma unroll
    for (int i = 0; i < ITERS; ++i) {
        const int idx = i * 256 + tid;
        const int k_row = idx >> 3;
        const int c = idx & 7;
        const int swiz_c = c ^ (k_row & 7);
        const uint32_t s_addr = smem_k_stage_sptr + static_cast<uint32_t>(k_row * 128 + swiz_c * 16);
        cp_async_cg_16(s_addr, k_base + static_cast<int64_t>(k_row) * sk8_s + c * 16);
    }
}

template <int BN, int STAGES, int MIN_CTAS, bool SPARSE_SEL, int K_BT_MODE, bool PV_F16_ACC>
__global__ void __launch_bounds__(256, 1) global_sage2_prefix_v6_sm121_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const uint8_t* __restrict__ K_fp8,
    const float* __restrict__ KS64,
    const __nv_bfloat16* __restrict__ V,
    const int32_t* __restrict__ BT,
    const int32_t* __restrict__ K_BT,
    const int32_t* __restrict__ SEQ,
    const int32_t* __restrict__ CUQ,
    const int32_t* __restrict__ SEL_BLOCKS,
    const int32_t* __restrict__ SEL_COUNTS,
    float* __restrict__ ACC_OUT,
    float* __restrict__ M_OUT,
    float* __restrict__ L_OUT,
    int64_t sq_t, int64_t sq_h,
    int64_t sk8_b, int64_t sk8_s, int64_t sk8_h,
    int64_t sks64_b, int64_t sks64_h,
    int64_t sv_b, int64_t sv_s, int64_t sv_h,
    int64_t sbt_r,
    int64_t skbt_r,
    int H, int G, int BS, int E, int sel_stride,
    float scale_log2,
    float pv_skip_tau
) {
    static_assert(BN == 64 || BN == 128, "BN must be 64 or 128");
    static_assert(STAGES == 2 || STAGES == 3, "STAGES must be 2 or 3");
    constexpr int PREFETCH = STAGES - 1;
    constexpr int SUB64 = BN / 64;
    constexpr float P_MAX = PV_F16_ACC ? PV_MAX_F16 : PV_MAX_F32;
    constexpr float INV_P_MAX = PV_F16_ACC ? INV_PV_MAX_F16 : INV_PV_MAX_F32;
    constexpr float V_MAX = PV_F16_ACC ? PV_MAX_F16 : PV_MAX_F32;
    constexpr float INV_V_MAX = PV_F16_ACC ? INV_PV_MAX_F16 : INV_PV_MAX_F32;

    extern __shared__ uint8_t dyn_smem[];
    __shared__ SageV6CtaParams sP;

    const int tid = static_cast<int>(threadIdx.x);
    if (tid == 0) {
        const int pid = static_cast<int>(blockIdx.x);
        const int r = static_cast<int>(blockIdx.y);
        const int h = pid % H;
        const int mt = pid / H;
        const int kvh = h / G;

        const int q0 = CUQ[r];
        const int qlen = CUQ[r + 1] - q0;
        const int seqlen = SEQ[r];
        const int past = seqlen - qlen;
        const int i_lo = mt * SAGE_BM;

        int num_tiles = 0;
        const int32_t* sel_row = nullptr;
        const int tiles_per_blk = BS / BN;
        const int tpb_shift = __ffs(tiles_per_blk) - 1;

        if (i_lo < qlen) {
            if constexpr (SPARSE_SEL) {
                const int cta_idx = r * static_cast<int>(gridDim.x) + pid;
                const int n_sel = SEL_COUNTS[cta_idx];
                if (n_sel > 0) {
                    sel_row = SEL_BLOCKS + static_cast<int64_t>(cta_idx) * sel_stride;
                    num_tiles = n_sel << tpb_shift;
                }
            } else {
                const int q_min = past + i_lo;
                const int k_bias_raw = (E > 0) ? max(q_min - E + 1, 0) : max(q_min, 0);
                const int k_bias_start = (k_bias_raw / BN) * BN;
                if (k_bias_start > 0) {
                    num_tiles = k_bias_start / BN;
                }
            }
        }

        const uint32_t k_sptr = static_cast<uint32_t>(__cvta_generic_to_shared(dyn_smem));
        sP.bt_row       = BT + static_cast<int64_t>(r) * sbt_r;
        sP.k_bt_row     = (K_BT_MODE == 2) ? (K_BT + static_cast<int64_t>(r) * skbt_r) : nullptr;
        sP.sel_row      = sel_row;
        sP.K_fp8_h      = K_fp8 + static_cast<int64_t>(kvh) * sk8_h;
        sP.KS64_h       = KS64 + static_cast<int64_t>(kvh) * sks64_h;
        sP.V_h          = V + static_cast<int64_t>(kvh) * sv_h;
        sP.sk8_b        = sk8_b;
        sP.sk8_s        = sk8_s;
        sP.sks64_b      = sks64_b;
        sP.sv_b         = sv_b;
        sP.sv_s         = sv_s;
        sP.k_base_sptr  = k_sptr;
        sP.vt_base_sptr = k_sptr + static_cast<uint32_t>(STAGES * BN * 128);
        sP.bs_shift     = __ffs(BS) - 1;
        sP.bs_mask      = BS - 1;
        sP.tpb_shift    = tpb_shift;
        sP.tpb_mask     = tiles_per_blk - 1;
        sP.num_tiles    = num_tiles;
        sP.q0           = q0;
        sP.qlen         = qlen;
        sP.i_lo         = i_lo;
        sP.h            = h;
    }
    __syncthreads();

    if (ld_smem_i32_nocse(&sP.num_tiles) <= 0) {
        // Defensive zero-init of M_OUT/L_OUT when i_lo < qlen (e.g. sparse n_sel == 0)
        const int i_lo_z = ld_smem_i32_nocse(&sP.i_lo);
        const int qlen_z = ld_smem_i32_nocse(&sP.qlen);
        if (i_lo_z < qlen_z && tid < SAGE_BM) {
            const int r_loc = i_lo_z + tid;
            if (r_loc < qlen_z) {
                const int64_t th = static_cast<int64_t>(ld_smem_i32_nocse(&sP.q0) + r_loc) * H
                                 + ld_smem_i32_nocse(&sP.h);
                M_OUT[th] = -INFINITY;
                L_OUT[th] = 0.0f;
            }
        }
        return;
    }

    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int gid = lane >> 2;
    const int t = lane & 3;

    uint32_t q_a0[4], q_a1[4], q_a2[4], q_a3[4];
    float qs_log2_0, qs_log2_1;
    uint32_t valid_mask = 0u;
    {
        const int i_lo_q = ld_smem_i32_nocse(&sP.i_lo);
        const int qlen_q = ld_smem_i32_nocse(&sP.qlen);
        const int q0_q   = ld_smem_i32_nocse(&sP.q0);
        const int h_q    = ld_smem_i32_nocse(&sP.h);
        const int r0_loc = i_lo_q + warp * 16 + gid;
        const int r1_loc = r0_loc + 8;
        const bool r0_v = (r0_loc < qlen_q);
        const bool r1_v = (r1_loc < qlen_q);
        valid_mask = (r0_v ? 1u : 0u) | (r1_v ? 2u : 0u);
        load_quant_q_regs_v6(
            Q, sq_t, sq_h, q0_q, h_q, r0_loc, r1_loc, r0_v, r1_v, t, scale_log2,
            q_a0, q_a1, q_a2, q_a3, qs_log2_0, qs_log2_1
        );
    }

    float m_i0 = -INFINITY, m_i1 = -INFINITY;
    float l_i0 = 0.0f, l_i1 = 0.0f;
    float acc[16][4];
    #pragma unroll
    for (int dt = 0; dt < 16; ++dt) {
        acc[dt][0] = 0.0f; acc[dt][1] = 0.0f;
        acc[dt][2] = 0.0f; acc[dt][3] = 0.0f;
    }

    auto issue_k_stage = [&](int tile_idx, int stage_idx) {
        int64_t k_blk_p;
        int slot_p;
        TileLocatorV6<SPARSE_SEL, K_BT_MODE>::locate_k(
            tile_idx, BN,
            ld_smem_i32_nocse(&sP.bs_shift), ld_smem_i32_nocse(&sP.bs_mask),
            ld_smem_i32_nocse(&sP.tpb_shift), ld_smem_i32_nocse(&sP.tpb_mask),
            ld_smem_ptr_nocse(&sP.bt_row),
            ld_smem_ptr_nocse(&sP.k_bt_row),
            ld_smem_ptr_nocse(&sP.sel_row),
            k_blk_p, slot_p);
        issue_k_async_v6<BN>(
            ld_smem_ptr_nocse(&sP.K_fp8_h), k_blk_p, slot_p,
            ld_smem_i64_nocse(&sP.sk8_b), ld_smem_i64_nocse(&sP.sk8_s),
            ld_smem_u32_nocse(&sP.k_base_sptr) + static_cast<uint32_t>(stage_idx * BN * 128),
            tid);
        cp_async_commit();
    };

    #pragma unroll
    for (int s = 0; s < PREFETCH; ++s) {
        if (s < ld_smem_i32_nocse(&sP.num_tiles)) {
            issue_k_stage(s, s);
        }
    }

    for (int it = 0; it < ld_smem_i32_nocse(&sP.num_tiles); ++it) {
        const int next_it = it + PREFETCH;
        const int n_tiles_cur = ld_smem_i32_nocse(&sP.num_tiles);
        if (next_it < n_tiles_cur) {
            issue_k_stage(next_it, next_it % STAGES);
            cp_async_wait_group<PREFETCH>();
        } else {
            const int rem = n_tiles_cur - 1 - it;
            if (PREFETCH == 2 && rem == 1) {
                cp_async_wait_group<1>();
            } else {
                cp_async_wait_group<0>();
            }
        }

        if (tid == 0) {
            int64_t k_blk_c, v_blk_c;
            int slot_c;
            TileLocatorV6<SPARSE_SEL, K_BT_MODE>::locate_kv(
                it, BN, sP.bs_shift, sP.bs_mask, sP.tpb_shift, sP.tpb_mask,
                sP.bt_row, sP.k_bt_row, sP.sel_row, k_blk_c, v_blk_c, slot_c);
            sP.cur_v_blk = v_blk_c;
            sP.cur_slot = slot_c;
            #pragma unroll
            for (int h64 = 0; h64 < SUB64; ++h64) {
                sP.ks_tile[h64] = __ldg(sP.KS64_h + k_blk_c * sP.sks64_b + ((slot_c >> 6) + h64));
            }
        }
        __syncthreads();

        const uint32_t cur_k_sptr = ld_smem_u32_nocse(&sP.k_base_sptr)
                                  + static_cast<uint32_t>((it % STAGES) * BN * 128);

        #pragma unroll
        for (int h64 = 0; h64 < SUB64; ++h64) {
            const int nt_base = h64 * 8;
            uint32_t d0[8], d1[8];

            {
                const int sub = lane >> 3;
                const int r_in = lane & 7;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int nt0 = nt_base + 2 * j;
                    uint32_t c0_0 = 0u, c1_0 = 0u;
                    uint32_t c0_1 = 0u, c1_1 = 0u;
                    const int k_row = (nt0 + (sub >> 1)) * 8 + r_in;

                    #pragma unroll
                    for (int ks = 0; ks < 4; ++ks) {
                        const int k_chunk = ks * 2 + (sub & 1);
                        const int swiz_c = k_chunk ^ r_in;
                        const uint32_t addr = cur_k_sptr + static_cast<uint32_t>(k_row * 128 + swiz_c * 16);
                        uint32_t b0_0, b1_0, b0_1, b1_1;
                        ldmatrix_x4(b0_0, b1_0, b0_1, b1_1, addr);
                        mma_m16n8k32_f16_e4m3(c0_0, c1_0, q_a0[ks], q_a1[ks], q_a2[ks], q_a3[ks], b0_0, b1_0, c0_0, c1_0);
                        mma_m16n8k32_f16_e4m3(c0_1, c1_1, q_a0[ks], q_a1[ks], q_a2[ks], q_a3[ks], b0_1, b1_1, c0_1, c1_1);
                    }
                    d0[2 * j]     = c0_0; d1[2 * j]     = c1_0;
                    d0[2 * j + 1] = c0_1; d1[2 * j + 1] = c1_1;
                }
            }

            uint32_t hm0 = hmax2_u32(
                hmax2_u32(hmax2_u32(d0[0], d0[1]), hmax2_u32(d0[2], d0[3])),
                hmax2_u32(hmax2_u32(d0[4], d0[5]), hmax2_u32(d0[6], d0[7])));
            uint32_t hm1 = hmax2_u32(
                hmax2_u32(hmax2_u32(d1[0], d1[1]), hmax2_u32(d1[2], d1[3])),
                hmax2_u32(hmax2_u32(d1[4], d1[5]), hmax2_u32(d1[6], d1[7])));

            hm0 = hmax2_u32(hm0, __shfl_xor_sync(0xffffffff, hm0, 1));
            hm0 = hmax2_u32(hm0, __shfl_xor_sync(0xffffffff, hm0, 2));
            hm1 = hmax2_u32(hm1, __shfl_xor_sync(0xffffffff, hm1, 1));
            hm1 = hmax2_u32(hm1, __shfl_xor_sync(0xffffffff, hm1, 2));

            const float ks_tile = ld_smem_f32_nocse(&sP.ks_tile[h64]);
            const float qs_ks_0 = qs_log2_0 * ks_tile;
            const float qs_ks_1 = qs_log2_1 * ks_tile;

            const float2 m0_f2 = __half22float2(*reinterpret_cast<const __half2*>(&hm0));
            const float2 m1_f2 = __half22float2(*reinterpret_cast<const __half2*>(&hm1));
            const float tm0 = fmaxf(m0_f2.x, m0_f2.y) * qs_ks_0;
            const float tm1 = fmaxf(m1_f2.x, m1_f2.y) * qs_ks_1;

            const float mn0_raw = fmaxf(m_i0, tm0);
            const float mn1_raw = fmaxf(m_i1, tm1);

            int local_active = 0;
            if ((valid_mask & 1u) && (tm0 - mn0_raw >= -pv_skip_tau)) local_active = 1;
            if ((valid_mask & 2u) && (tm1 - mn1_raw >= -pv_skip_tau)) local_active = 1;

            if (tid < 16) {
                sP.vs_bits[tid] = 0u;
            }
            const int do_pv = __syncthreads_or((pv_skip_tau > 0.0f) ? local_active : 1);

            if (do_pv) {
                const int64_t v_blk_cur = ld_smem_i64_nocse(&sP.cur_v_blk);
                const int slot_cur = ld_smem_i32_nocse(&sP.cur_slot);
                const int64_t sv_s_cur = ld_smem_i64_nocse(&sP.sv_s);
                const __nv_bfloat16* v_ptr = ld_smem_ptr_nocse(&sP.V_h)
                    + v_blk_cur * ld_smem_i64_nocse(&sP.sv_b)
                    + static_cast<int64_t>(slot_cur + h64 * 64) * sv_s_cur;

                // Pass 1: L1-cached coalesced V amax per 8-channel group (does NOT touch cp.async K pipeline!)
                v64_pass1_amax(v_ptr, sv_s_cur, 64, sP.vs_bits, tid);

                const bool warp_active = (pv_skip_tau <= 0.0f) || (__any_sync(0xffffffff, local_active) != 0);
                uint32_t pa[2][4];
                float w_row0 = 0.0f, w_row1 = 0.0f;

                if (warp_active) {
                    const float mn0 = (mn0_raw < NEG_INF_GUARD) ? 0.0f : mn0_raw;
                    const float mn1 = (mn1_raw < NEG_INF_GUARD) ? 0.0f : mn1_raw;
                    const float a0 = (m_i0 < NEG_INF_GUARD) ? 0.0f : fast_exp2f(m_i0 - mn0);
                    const float a1 = (m_i1 < NEG_INF_GUARD) ? 0.0f : fast_exp2f(m_i1 - mn1);

                    if (__any_sync(0xffffffff, a0 != 1.0f)) {
                        #pragma unroll
                        for (int dt = 0; dt < 16; ++dt) {
                            acc[dt][0] *= a0;
                            acc[dt][1] *= a0;
                        }
                    }
                    if (__any_sync(0xffffffff, a1 != 1.0f)) {
                        #pragma unroll
                        for (int dt = 0; dt < 16; ++dt) {
                            acc[dt][2] *= a1;
                            acc[dt][3] *= a1;
                        }
                    }

                    const float tm0_safe = (tm0 < NEG_INF_GUARD) ? 0.0f : tm0;
                    const float tm1_safe = (tm1 < NEG_INF_GUARD) ? 0.0f : tm1;
                    const float tscale0 = (tm0 < NEG_INF_GUARD) ? 0.0f : fast_exp2f(tm0_safe - mn0);
                    const float tscale1 = (tm1 < NEG_INF_GUARD) ? 0.0f : fast_exp2f(tm1_safe - mn1);

                    float sum0 = 0.0f, sum1 = 0.0f;
                    uint16_t u0[8], u1[8];
                    #pragma unroll
                    for (int nt = 0; nt < 8; ++nt) {
                        const float2 f0 = __half22float2(*reinterpret_cast<const __half2*>(&d0[nt]));
                        const float2 f1 = __half22float2(*reinterpret_cast<const __half2*>(&d1[nt]));
                        const float p00 = fast_exp2f(fmaf(f0.x, qs_ks_0, -tm0_safe));
                        const float p01 = fast_exp2f(fmaf(f0.y, qs_ks_0, -tm0_safe));
                        const float p10 = fast_exp2f(fmaf(f1.x, qs_ks_1, -tm1_safe));
                        const float p11 = fast_exp2f(fmaf(f1.y, qs_ks_1, -tm1_safe));
                        sum0 += p00 + p01;
                        sum1 += p10 + p11;
                        u0[nt] = f32x2_to_e4m3x2_rn(p00 * P_MAX, p01 * P_MAX);
                        u1[nt] = f32x2_to_e4m3x2_rn(p10 * P_MAX, p11 * P_MAX);
                    }

                    sum0 += __shfl_xor_sync(0xffffffff, sum0, 1);
                    sum0 += __shfl_xor_sync(0xffffffff, sum0, 2);
                    sum1 += __shfl_xor_sync(0xffffffff, sum1, 1);
                    sum1 += __shfl_xor_sync(0xffffffff, sum1, 2);
                    l_i0 = fmaf(tscale0, sum0, l_i0 * a0);
                    l_i1 = fmaf(tscale1, sum1, l_i1 * a1);
                    m_i0 = mn0_raw;
                    m_i1 = mn1_raw;
                    w_row0 = tscale0 * INV_P_MAX;
                    w_row1 = tscale1 * INV_P_MAX;

                    repack_p_quad_half16(u0[0], u0[1], u1[0], u1[1], gid, t, pa[0][0], pa[0][1]);
                    repack_p_quad_half16(u0[2], u0[3], u1[2], u1[3], gid, t, pa[0][2], pa[0][3]);
                    repack_p_quad_half16(u0[4], u0[5], u1[4], u1[5], gid, t, pa[1][0], pa[1][1]);
                    repack_p_quad_half16(u0[6], u0[7], u1[6], u1[7], gid, t, pa[1][2], pa[1][3]);
                }
                __syncthreads();

                if (tid < 16) {
                    const float am = fmaxf(__uint_as_float(sP.vs_bits[tid]), 1.0e-12f);
                    sP.vs_tile[tid] = am * INV_V_MAX;
                    sP.vs_inv[tid]  = div_rn_ftz(V_MAX, am);
                }
                __syncthreads();

                const uint32_t vt_sptr = ld_smem_u32_nocse(&sP.vt_base_sptr);
                v64_pass2_quant_transpose(v_ptr, sv_s_cur, 64, sP.vs_inv, vt_sptr, tid);
                __syncthreads();

                if (warp_active) {
                    execute_pv_fp8_m16n8k32<PV_F16_ACC>(
                        pa, w_row0, w_row1, vt_sptr, sP.vs_tile, lane, acc);
                }
                if (SUB64 > 1 && h64 + 1 < SUB64) {
                    __syncthreads();
                }
            }
        }
        __syncthreads();
    }

    const int i_lo_e = ld_smem_i32_nocse(&sP.i_lo);
    const int qlen_e = ld_smem_i32_nocse(&sP.qlen);
    const int q0_e   = ld_smem_i32_nocse(&sP.q0);
    const int h_e    = ld_smem_i32_nocse(&sP.h);
    const int r0_loc = i_lo_e + warp * 16 + gid;
    const int r1_loc = r0_loc + 8;
    if (r0_loc < qlen_e) {
        const int64_t th0 = static_cast<int64_t>(q0_e + r0_loc) * H + h_e;
        float2* a_ptr0 = reinterpret_cast<float2*>(ACC_OUT + th0 * D_HEAD);
        #pragma unroll
        for (int dt = 0; dt < 16; ++dt) {
            a_ptr0[dt * 4 + t] = make_float2(acc[dt][0], acc[dt][1]);
        }
        if (t == 0) { M_OUT[th0] = m_i0; L_OUT[th0] = l_i0; }
    }
    if (r1_loc < qlen_e) {
        const int64_t th1 = static_cast<int64_t>(q0_e + r1_loc) * H + h_e;
        float2* a_ptr1 = reinterpret_cast<float2*>(ACC_OUT + th1 * D_HEAD);
        #pragma unroll
        for (int dt = 0; dt < 16; ++dt) {
            a_ptr1[dt * 4 + t] = make_float2(acc[dt][2], acc[dt][3]);
        }
        if (t == 0) { M_OUT[th1] = m_i1; L_OUT[th1] = l_i1; }
    }
}

// =================================================================================================
// Native SM121a FP8-f16 Exact-Kmax Quest Block Scorer (`score_kmax_blocks_fp8_sm121_kernel`)
// =================================================================================================

__device__ __forceinline__ void load_quant_box64_swiz_v6(
    const __nv_bfloat16* __restrict__ K_BOX_h,
    const int32_t* __restrict__ bt_row,
    int64_t sm_b,
    int b0, int n_pblks,
    uint8_t* __restrict__ smem_fp8,
    float* __restrict__ smem_scale,
    int tid
) {
    const int row = tid >> 2;
    const int chunk = tid & 3;
    const int lb = b0 + row;
    const bool ok = (lb < n_pblks);

    float amax = 0.0f;
    const __nv_bfloat16* src_row = ok ? (K_BOX_h + static_cast<int64_t>(bt_row[lb]) * sm_b + chunk * 32) : nullptr;
    if (ok) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint4 v = ld_ca_u4(src_row + i * 8);
            const float2 f0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v.x));
            const float2 f1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v.y));
            const float2 f2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v.z));
            const float2 f3 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v.w));
            amax = fmaxf(amax, fmaxf(fmaxf(fabsf(f0.x), fabsf(f0.y)), fmaxf(fabsf(f1.x), fabsf(f1.y))));
            amax = fmaxf(amax, fmaxf(fmaxf(fabsf(f2.x), fabsf(f2.y)), fmaxf(fabsf(f3.x), fabsf(f3.y))));
        }
    }
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 2));
    amax = fmaxf(amax, 1.0e-12f);
    const float inv = div_rn_ftz(QK_MAX, amax);

    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        uint4 pack = make_uint4(0u, 0u, 0u, 0u);
        if (ok) {
            const uint4 v0 = ld_ca_u4(src_row + i * 16);
            const uint4 v1 = ld_ca_u4(src_row + i * 16 + 8);
            const float2 a0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v0.x));
            const float2 a1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v0.y));
            const float2 a2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v0.z));
            const float2 a3 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v0.w));
            const float2 a4 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v1.x));
            const float2 a5 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v1.y));
            const float2 a6 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v1.z));
            const float2 a7 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v1.w));
            pack.x = f32x4_to_e4m3x4_rn(a0.x * inv, a0.y * inv, a1.x * inv, a1.y * inv);
            pack.y = f32x4_to_e4m3x4_rn(a2.x * inv, a2.y * inv, a3.x * inv, a3.y * inv);
            pack.z = f32x4_to_e4m3x4_rn(a4.x * inv, a4.y * inv, a5.x * inv, a5.y * inv);
            pack.w = f32x4_to_e4m3x4_rn(a6.x * inv, a6.y * inv, a7.x * inv, a7.y * inv);
        }
        const int c = chunk * 2 + i;
        const int swiz_c = c ^ (row & 7);
        *reinterpret_cast<uint4*>(smem_fp8 + row * 128 + swiz_c * 16) = pack;
    }
    if (chunk == 0) {
        smem_scale[row] = amax * INV_QK_MAX;
    }
}

__global__ void __launch_bounds__(256, 2) score_kmax_blocks_fp8_sm121_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K_MID,
    const __nv_bfloat16* __restrict__ K_RAD,
    const int32_t* __restrict__ BT,
    const int32_t* __restrict__ SEQ,
    const int32_t* __restrict__ CUQ,
    float* __restrict__ SCORES,
    int32_t* __restrict__ SEL_COUNTS,
    int64_t sq_t, int64_t sq_h,
    int64_t sm_b, int64_t sm_h,
    int64_t sbt_r,
    int64_t ssc_cta,
    int max_pblks, int H, int G, int BS, int E,
    int sink_blocks, int local_blocks, int k_sel
) {
    constexpr int BLOCK_B = 64;
    const int pid = static_cast<int>(blockIdx.x);
    const int r = static_cast<int>(blockIdx.y);
    const int h = pid % H;
    const int mt = pid / H;
    const int kvh = h / G;
    const int cta_idx = r * static_cast<int>(gridDim.x) + pid;

    const int q0 = CUQ[r];
    const int qlen = CUQ[r + 1] - q0;
    const int seqlen = SEQ[r];
    const int past = seqlen - qlen;
    const int tid = static_cast<int>(threadIdx.x);

    const int i_lo = mt * SAGE_BM;
    float* sc_row = SCORES + static_cast<int64_t>(cta_idx) * ssc_cta;

    if (i_lo >= qlen) {
        if (tid == 0 && SEL_COUNTS != nullptr) SEL_COUNTS[cta_idx] = 0;
        for (int lb = tid; lb < max_pblks; lb += THREADS) {
            sc_row[lb] = -1.0e30f;
        }
        return;
    }

    const int q_min = past + i_lo;
    const int k_bias_raw = (E > 0) ? max(q_min - E + 1, 0) : max(q_min, 0);
    const int n_pblks = min(k_bias_raw / BS, max_pblks);

    if (tid == 0 && SEL_COUNTS != nullptr) {
        SEL_COUNTS[cta_idx] = min(n_pblks, k_sel);
    }

    if (n_pblks <= 0) {
        for (int lb = tid; lb < max_pblks; lb += THREADS) {
            sc_row[lb] = -1.0e30f;
        }
        return;
    }

    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int gid = lane >> 2;
    const int t = lane & 3;
    const int sub = lane >> 3;
    const int r_in = lane & 7;

    __shared__ __align__(16) uint8_t smem_km[BLOCK_B * 128];
    __shared__ __align__(16) uint8_t smem_kr[BLOCK_B * 128];
    __shared__ __align__(16) float smem_kms[BLOCK_B];
    __shared__ __align__(16) float smem_krs[BLOCK_B];
    __shared__ __align__(16) float smem_wmax[WARPS][BLOCK_B];

    const uint32_t km_sptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_km));
    const uint32_t kr_sptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_kr));

    uint32_t q_a0[4], q_a1[4], q_a2[4], q_a3[4];
    const int r0_loc = i_lo + warp * 16 + gid;
    const int r1_loc = r0_loc + 8;
    const bool r0_valid = (r0_loc < qlen);
    const bool r1_valid = (r1_loc < qlen);
    float qs0, qs1;
    load_quant_q_regs_v6(
        Q, sq_t, sq_h, q0, h, r0_loc, r1_loc, r0_valid, r1_valid, t, 1.0f,
        q_a0, q_a1, q_a2, q_a3, qs0, qs1
    );

    const int32_t* bt_row = BT + static_cast<int64_t>(r) * sbt_r;
    const __nv_bfloat16* km_h = K_MID + static_cast<int64_t>(kvh) * sm_h;
    const __nv_bfloat16* kr_h = K_RAD + static_cast<int64_t>(kvh) * sm_h;
    const int local_start = n_pblks - local_blocks;

    for (int b0 = 0; b0 < n_pblks; b0 += BLOCK_B) {
        const int b_hi = min(b0 + BLOCK_B, n_pblks);
        const bool all_mandatory = (b_hi <= sink_blocks) || (b0 >= local_start);
        if (all_mandatory) {
            if (tid < BLOCK_B) {
                const int lb = b0 + tid;
                if (lb < max_pblks) {
                    sc_row[lb] = (lb < n_pblks) ? 1.0e30f : -1.0e30f;
                }
            }
            continue;
        }

        load_quant_box64_swiz_v6(km_h, bt_row, sm_b, b0, n_pblks, smem_km, smem_kms, tid);
        load_quant_box64_swiz_v6(kr_h, bt_row, sm_b, b0, n_pblks, smem_kr, smem_krs, tid);
        __syncthreads();

        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int nt0 = 2 * j;
            uint32_t dm0_0 = 0u, dm1_0 = 0u, dm0_1 = 0u, dm1_1 = 0u;
            uint32_t dr0_0 = 0u, dr1_0 = 0u, dr0_1 = 0u, dr1_1 = 0u;
            const int k_row = (nt0 + (sub >> 1)) * 8 + r_in;

            #pragma unroll
            for (int ks = 0; ks < 4; ++ks) {
                const int k_chunk = ks * 2 + (sub & 1);
                const uint32_t off = static_cast<uint32_t>(k_row * 128 + (k_chunk ^ r_in) * 16);

                uint32_t bm0_0, bm1_0, bm0_1, bm1_1;
                ldmatrix_x4(bm0_0, bm1_0, bm0_1, bm1_1, km_sptr + off);
                mma_m16n8k32_f16_e4m3(dm0_0, dm1_0, q_a0[ks], q_a1[ks], q_a2[ks], q_a3[ks], bm0_0, bm1_0, dm0_0, dm1_0);
                mma_m16n8k32_f16_e4m3(dm0_1, dm1_1, q_a0[ks], q_a1[ks], q_a2[ks], q_a3[ks], bm0_1, bm1_1, dm0_1, dm1_1);

                const uint32_t qa0_abs = q_a0[ks] & 0x7F7F7F7Fu;
                const uint32_t qa1_abs = q_a1[ks] & 0x7F7F7F7Fu;
                const uint32_t qa2_abs = q_a2[ks] & 0x7F7F7F7Fu;
                const uint32_t qa3_abs = q_a3[ks] & 0x7F7F7F7Fu;

                uint32_t br0_0, br1_0, br0_1, br1_1;
                ldmatrix_x4(br0_0, br1_0, br0_1, br1_1, kr_sptr + off);
                mma_m16n8k32_f16_e4m3(dr0_0, dr1_0, qa0_abs, qa1_abs, qa2_abs, qa3_abs, br0_0, br1_0, dr0_0, dr1_0);
                mma_m16n8k32_f16_e4m3(dr0_1, dr1_1, qa0_abs, qa1_abs, qa2_abs, qa3_abs, br0_1, br1_1, dr0_1, dr1_1);
            }

            const float2 kms_a = reinterpret_cast<const float2*>(smem_kms)[nt0 * 4 + t];
            const float2 krs_a = reinterpret_cast<const float2*>(smem_krs)[nt0 * 4 + t];
            const float2 kms_b = reinterpret_cast<const float2*>(smem_kms)[(nt0 + 1) * 4 + t];
            const float2 krs_b = reinterpret_cast<const float2*>(smem_krs)[(nt0 + 1) * 4 + t];

            const float2 fm0_a = __half22float2(*reinterpret_cast<const __half2*>(&dm0_0));
            const float2 fm1_a = __half22float2(*reinterpret_cast<const __half2*>(&dm1_0));
            const float2 fr0_a = __half22float2(*reinterpret_cast<const __half2*>(&dr0_0));
            const float2 fr1_a = __half22float2(*reinterpret_cast<const __half2*>(&dr1_0));

            float ca0 = -1.0e30f, ca1 = -1.0e30f;
            if (r0_valid) {
                ca0 = fmaxf(ca0, (fm0_a.x * kms_a.x + fr0_a.x * krs_a.x) * qs0);
                ca1 = fmaxf(ca1, (fm0_a.y * kms_a.y + fr0_a.y * krs_a.y) * qs0);
            }
            if (r1_valid) {
                ca0 = fmaxf(ca0, (fm1_a.x * kms_a.x + fr1_a.x * krs_a.x) * qs1);
                ca1 = fmaxf(ca1, (fm1_a.y * kms_a.y + fr1_a.y * krs_a.y) * qs1);
            }

            const float2 fm0_b = __half22float2(*reinterpret_cast<const __half2*>(&dm0_1));
            const float2 fm1_b = __half22float2(*reinterpret_cast<const __half2*>(&dm1_1));
            const float2 fr0_b = __half22float2(*reinterpret_cast<const __half2*>(&dr0_1));
            const float2 fr1_b = __half22float2(*reinterpret_cast<const __half2*>(&dr1_1));

            float cb0 = -1.0e30f, cb1 = -1.0e30f;
            if (r0_valid) {
                cb0 = fmaxf(cb0, (fm0_b.x * kms_b.x + fr0_b.x * krs_b.x) * qs0);
                cb1 = fmaxf(cb1, (fm0_b.y * kms_b.y + fr0_b.y * krs_b.y) * qs0);
            }
            if (r1_valid) {
                cb0 = fmaxf(cb0, (fm1_b.x * kms_b.x + fr1_b.x * krs_b.x) * qs1);
                cb1 = fmaxf(cb1, (fm1_b.y * kms_b.y + fr1_b.y * krs_b.y) * qs1);
            }

            ca0 = fmaxf(ca0, __shfl_xor_sync(0xffffffff, ca0, 4));
            ca0 = fmaxf(ca0, __shfl_xor_sync(0xffffffff, ca0, 8));
            ca0 = fmaxf(ca0, __shfl_xor_sync(0xffffffff, ca0, 16));
            ca1 = fmaxf(ca1, __shfl_xor_sync(0xffffffff, ca1, 4));
            ca1 = fmaxf(ca1, __shfl_xor_sync(0xffffffff, ca1, 8));
            ca1 = fmaxf(ca1, __shfl_xor_sync(0xffffffff, ca1, 16));

            cb0 = fmaxf(cb0, __shfl_xor_sync(0xffffffff, cb0, 4));
            cb0 = fmaxf(cb0, __shfl_xor_sync(0xffffffff, cb0, 8));
            cb0 = fmaxf(cb0, __shfl_xor_sync(0xffffffff, cb0, 16));
            cb1 = fmaxf(cb1, __shfl_xor_sync(0xffffffff, cb1, 4));
            cb1 = fmaxf(cb1, __shfl_xor_sync(0xffffffff, cb1, 8));
            cb1 = fmaxf(cb1, __shfl_xor_sync(0xffffffff, cb1, 16));

            if (gid == 0) {
                smem_wmax[warp][nt0 * 8 + 2 * t + 0] = ca0;
                smem_wmax[warp][nt0 * 8 + 2 * t + 1] = ca1;
                smem_wmax[warp][(nt0 + 1) * 8 + 2 * t + 0] = cb0;
                smem_wmax[warp][(nt0 + 1) * 8 + 2 * t + 1] = cb1;
            }
        }
        __syncthreads();

        if (tid < BLOCK_B) {
            const int lb = b0 + tid;
            if (lb < max_pblks) {
                float mx = -1.0e30f;
                if (lb < n_pblks) {
                    if (lb < sink_blocks || lb >= local_start) {
                        mx = 1.0e30f;
                    } else {
                        #pragma unroll
                        for (int w = 0; w < WARPS; ++w) {
                            mx = fmaxf(mx, smem_wmax[w][tid]);
                        }
                    }
                }
                sc_row[lb] = mx;
            }
        }
        __syncthreads();
    }

    const int tail_start = ((n_pblks + BLOCK_B - 1) / BLOCK_B) * BLOCK_B;
    for (int lb = tail_start + tid; lb < max_pblks; lb += THREADS) {
        sc_row[lb] = -1.0e30f;
    }
}

// =================================================================================================
// PART 2: GQA-Fused (`H_PER_CTA=2` / `1`) FP8 Local SWA (`40L`) & Global Tail OTF-Rel (`8L`)
// =================================================================================================

static constexpr int OTF_BN = 64;
static constexpr int OTF_BIAS_STRIDE = 84;  // >= 80 cols, +4 uint16 skew avoids SMEM bank conflicts

struct alignas(16) OtfFp8CtaParams {
    const int32_t* bt_row;
    const __nv_bfloat16* K_h;
    const __nv_bfloat16* V_h;
    const uint16_t* RelProjT;
    const uint16_t* GATE;
    const uint16_t* SINK;
    uint16_t* Y;
    const float* ACC_IN;
    const float* M_IN;
    const float* L_IN;
    __nv_bfloat16* OUT_RAW;
    float* LSE_OUT;
    int64_t sk_b;
    int64_t sk_s;
    int64_t sv_b;
    int64_t sv_s;
    int64_t sg_t;
    int64_t ssk_t;
    int64_t sy_t;
    int64_t so_t;
    int64_t so_h;
    int64_t sl_t;
    int64_t sl_h;
    int64_t cur_blk;
    uint32_t k_sptr;
    uint32_t vt_sptr;
    int bs_shift;
    int bs_mask;
    int q0;
    int qlen;
    int seqlen;
    int past;
    int i_lo;
    int h0;
    int h1;
    int H;
    int E;
    int window;
    int k_beg;
    int k_end;
    int k_start_exact;
    int cur_slot;
    uint32_t vs_bits[16];
    float vs_tile[16];
    float vs_inv[16];
};

// Exact 0-ULP 8-lane L2-clip (+ optional Tau scale) of one R[16] row into `dst_row16`
template <bool HAS_TAU>
__device__ __forceinline__ void clip_r_row_exact_8lane_v6(
    const uint16_t* __restrict__ r_ptr,
    const uint16_t* __restrict__ r_scale_row,
    bool valid,
    float tau_val, float rel_threshold, float clip_eps,
    int sub_lane,  // 0..7 -> cols (2*sub_lane, 2*sub_lane+1)
    uint16_t* __restrict__ dst_row16
) {
    const int d = sub_lane * 2;
    float x0 = 0.0f, x1 = 0.0f;
    if (valid) {
        const uint32_t rv = *reinterpret_cast<const uint32_t*>(r_ptr + d);
        const uint32_t sv = *reinterpret_cast<const uint32_t*>(r_scale_row + d);
        x0 = __fmul_rn(bf16_to_f32(static_cast<uint16_t>(rv & 0xFFFFu)),
                       bf16_to_f32(static_cast<uint16_t>(sv & 0xFFFFu)));
        x1 = __fmul_rn(bf16_to_f32(static_cast<uint16_t>(rv >> 16)),
                       bf16_to_f32(static_cast<uint16_t>(sv >> 16)));
    }
    float s = __fadd_rn(__fmul_rn(x0, x0), __fmul_rn(x1, x1));
    s = __fadd_rn(s, __shfl_xor_sync(0xffffffff, s, 4));
    s = __fadd_rn(s, __shfl_xor_sync(0xffffffff, s, 2));
    s = __fadd_rn(s, __shfl_xor_sync(0xffffffff, s, 1));
    if (valid) {
        const float norm = (s > 0.0f) ? __fmul_rn(s, rsqrt_approx_f32(s)) : 0.0f;
        const float mult = fminf(__fdiv_rn(rel_threshold, __fadd_rn(norm, clip_eps)), 1.0f);
        uint16_t o0 = f32_to_bf16_u16(__fmul_rn(x0, mult));
        uint16_t o1 = f32_to_bf16_u16(__fmul_rn(x1, mult));
        if constexpr (HAS_TAU) {
            o0 = f32_to_bf16_u16(__fmul_rn(bf16_to_f32(o0), tau_val));
            o1 = f32_to_bf16_u16(__fmul_rn(bf16_to_f32(o1), tau_val));
        }
        *reinterpret_cast<uint32_t*>(dst_row16 + d) = pack_bf16x2(o0, o1);
    } else {
        *reinterpret_cast<uint32_t*>(dst_row16 + d) = 0u;
    }
}

// Coalesced 256B/row online BF16 K [64, 128] -> swizzled FP8 `sK_fp8` + exact per-token `sKs[64]`
__device__ __forceinline__ void load_quant_k64_online(
    const __nv_bfloat16* __restrict__ k_base,
    int64_t sk_s,
    int valid_k_rows,
    uint8_t* __restrict__ smem_k_fp8,
    float* __restrict__ smem_ks,
    int tid
) {
    const int k_row = tid >> 2;  // 0..63
    const int c_q   = tid & 3;   // 0..3 -> 32 BF16 = 64B per lane (256B/row coalesced!)
    const bool ok = (k_row < valid_k_rows);
    const __nv_bfloat16* src = ok ? (k_base + static_cast<int64_t>(k_row) * sk_s + c_q * 32) : nullptr;

    float amax = 0.0f;
    if (ok) {
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            const uint4 v = ld_ca_u4(src + u * 8);
            const float2 f0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v.x));
            const float2 f1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v.y));
            const float2 f2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v.z));
            const float2 f3 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v.w));
            amax = fmaxf(amax, fmaxf(fmaxf(fabsf(f0.x), fabsf(f0.y)), fmaxf(fabsf(f1.x), fabsf(f1.y))));
            amax = fmaxf(amax, fmaxf(fmaxf(fabsf(f2.x), fabsf(f2.y)), fmaxf(fabsf(f3.x), fabsf(f3.y))));
        }
    }
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 2));
    amax = fmaxf(amax, 1.0e-12f);
    const float inv = div_rn_ftz(QK_MAX, amax);
    if (c_q == 0) {
        smem_ks[k_row] = ok ? (amax * INV_QK_MAX) : 0.0f;
    }

    #pragma unroll
    for (int h16 = 0; h16 < 2; ++h16) {
        uint4 pack = make_uint4(0u, 0u, 0u, 0u);
        if (ok) {
            const uint4 v0 = ld_ca_u4(src + h16 * 16);
            const uint4 v1 = ld_ca_u4(src + h16 * 16 + 8);
            const float2 a0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v0.x));
            const float2 a1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v0.y));
            const float2 a2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v0.z));
            const float2 a3 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v0.w));
            const float2 a4 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v1.x));
            const float2 a5 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v1.y));
            const float2 a6 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v1.z));
            const float2 a7 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v1.w));
            pack.x = f32x4_to_e4m3x4_rn(a0.x * inv, a0.y * inv, a1.x * inv, a1.y * inv);
            pack.y = f32x4_to_e4m3x4_rn(a2.x * inv, a2.y * inv, a3.x * inv, a3.y * inv);
            pack.z = f32x4_to_e4m3x4_rn(a4.x * inv, a4.y * inv, a5.x * inv, a5.y * inv);
            pack.w = f32x4_to_e4m3x4_rn(a6.x * inv, a6.y * inv, a7.x * inv, a7.y * inv);
        }
        const int chunk16 = c_q * 2 + h16;
        const int swiz_c = chunk16 ^ (k_row & 7);
        *reinterpret_cast<uint4*>(smem_k_fp8 + k_row * 128 + swiz_c * 16) = pack;
    }
}

// Warp-private exact BF16 OTF relative bias `r_clip [16, 16] @ RelProjT [<=80, 16]^T` (<=10 MMAs, 0 CTA barrier!)
__device__ __forceinline__ void compute_otf_bias_warp(
    const uint16_t (&sR_warp)[16][16],
    const uint16_t* __restrict__ RelProjT,
    uint16_t (&sBias_warp)[16][OTF_BIAS_STRIDE],
    int d_lo, int d_hi, int E,
    int gid, int t
) {
    const uint32_t r_a0 = *reinterpret_cast<const uint32_t*>(&sR_warp[gid][2 * t]);
    const uint32_t r_a1 = *reinterpret_cast<const uint32_t*>(&sR_warp[gid + 8][2 * t]);
    const uint32_t r_a2 = *reinterpret_cast<const uint32_t*>(&sR_warp[gid][8 + 2 * t]);
    const uint32_t r_a3 = *reinterpret_cast<const uint32_t*>(&sR_warp[gid + 8][8 + 2 * t]);

    const int n_d = min(E, d_hi + 1) - d_lo;
    const int n_nt = (n_d + 7) >> 3;  // <= 10 N-tiles
    for (int nt = 0; nt < n_nt; ++nt) {
        const int d_col = d_lo + nt * 8 + gid;
        uint32_t b0 = 0u, b1 = 0u;
        if (d_col < E) {
            const uint16_t* rp = RelProjT + static_cast<int64_t>(d_col) * D_REL;
            b0 = *reinterpret_cast<const uint32_t*>(rp + 2 * t);
            b1 = *reinterpret_cast<const uint32_t*>(rp + 8 + 2 * t);
        }
        float bd0 = 0.0f, bd1 = 0.0f, bd2 = 0.0f, bd3 = 0.0f;
        mma_m16n8k16_f32_bf16(bd0, bd1, bd2, bd3, r_a0, r_a1, r_a2, r_a3, b0, b1, 0.0f, 0.0f, 0.0f, 0.0f);
        const int c_off = nt * 8 + 2 * t;
        *reinterpret_cast<uint32_t*>(&sBias_warp[gid][c_off]) =
            pack_bf16x2(f32_to_bf16_u16(bd0), f32_to_bf16_u16(bd1));
        *reinterpret_cast<uint32_t*>(&sBias_warp[gid + 8][c_off]) =
            pack_bf16x2(f32_to_bf16_u16(bd2), f32_to_bf16_u16(bd3));
    }
    __syncwarp();
}

// Unified Fused Gate + Sink Epilogue (plus optional raw OUT / LSE writeout)
__device__ __forceinline__ void store_otf_row_epilogue(
    int tok, int h_my,
    float m_i, float l_i,
    const float (&acc)[16][4],
    int acc_pair_off,  // 0 for row0 (gid), 2 for row1 (gid+8)
    int t,
    const OtfFp8CtaParams& sP
) {
    const float l_safe = (l_i > 0.0f) ? l_i : 1.0f;
    const float lse = (l_i > 0.0f) ? __fmul_rn(__fadd_rn(m_i, log2f(l_safe)), LN2_F) : -INFINITY;

    float* lse_out = ld_smem_mut_ptr_nocse(&sP.LSE_OUT);
    if (lse_out != nullptr && t == 0) {
        lse_out[static_cast<int64_t>(tok) * ld_smem_i64_nocse(&sP.sl_t)
              + static_cast<int64_t>(h_my) * ld_smem_i64_nocse(&sP.sl_h)] = lse;
    }

    __nv_bfloat16* out_raw = ld_smem_mut_ptr_nocse(&sP.OUT_RAW);
    const uint16_t* gate_ptr = ld_smem_ptr_nocse(&sP.GATE);
    const uint16_t* sink_ptr = ld_smem_ptr_nocse(&sP.SINK);
    uint16_t* y_ptr = ld_smem_mut_ptr_nocse(&sP.Y);

    float sig_s = 1.0f;
    if (sink_ptr != nullptr) {
        const float sink_f = bf16_to_f32(sink_ptr[static_cast<int64_t>(tok) * ld_smem_i64_nocse(&sP.ssk_t) + h_my]);
        sig_s = (l_i > 0.0f) ? triton_sigmoid(__fsub_rn(lse, sink_f)) : 0.0f;
    }

    const int64_t sg_t = ld_smem_i64_nocse(&sP.sg_t);
    const int64_t sy_t = ld_smem_i64_nocse(&sP.sy_t);
    const int64_t so_t = ld_smem_i64_nocse(&sP.so_t);
    const int64_t so_h = ld_smem_i64_nocse(&sP.so_h);

    #pragma unroll
    for (int dt = 0; dt < 16; ++dt) {
        const int d0 = dt * 8 + 2 * t;
        const float x0 = (l_i > 0.0f) ? div_full_f32(acc[dt][acc_pair_off + 0], l_safe) : 0.0f;
        const float x1 = (l_i > 0.0f) ? div_full_f32(acc[dt][acc_pair_off + 1], l_safe) : 0.0f;

        if (out_raw != nullptr) {
            uint16_t* op = reinterpret_cast<uint16_t*>(
                out_raw + static_cast<int64_t>(tok) * so_t + static_cast<int64_t>(h_my) * so_h + d0);
            *reinterpret_cast<uint32_t*>(op) = pack_bf16x2(f32_to_bf16_u16(x0), f32_to_bf16_u16(x1));
        }

        if (y_ptr != nullptr && gate_ptr != nullptr) {
            const int col = h_my * D_HEAD + d0;
            const uint16_t* gp = gate_ptr + static_cast<int64_t>(tok) * sg_t + col;
            uint16_t* yp       = y_ptr    + static_cast<int64_t>(tok) * sy_t + col;
            float g0, g1;
            if ((reinterpret_cast<uintptr_t>(gp) & 3u) == 0u) {
                const uint32_t gv = *reinterpret_cast<const uint32_t*>(gp);
                g0 = bf16_to_f32(static_cast<uint16_t>(gv & 0xFFFFu));
                g1 = bf16_to_f32(static_cast<uint16_t>(gv >> 16));
            } else {
                g0 = bf16_to_f32(gp[0]);
                g1 = bf16_to_f32(gp[1]);
            }
            const uint16_t y0 = f32_to_bf16_u16(__fmul_rn(__fmul_rn(x0, triton_sigmoid(g0)), sig_s));
            const uint16_t y1 = f32_to_bf16_u16(__fmul_rn(__fmul_rn(x1, triton_sigmoid(g1)), sig_s));
            *reinterpret_cast<uint32_t*>(yp) = pack_bf16x2(y0, y1);
        }
    }
}

// Unified template for Local SWA (`IS_GLOBAL_TAIL = false`) and Global Tail (`IS_GLOBAL_TAIL = true`)
template <int H_PER_CTA, bool IS_GLOBAL_TAIL, bool HAS_TAU, bool PV_F16_ACC>
__global__ void __launch_bounds__(256, 1) otf_rel_fp8_v6_sm121_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const uint16_t* __restrict__ R,
    const uint16_t* __restrict__ RScale,
    const float* __restrict__ Tau,
    const uint16_t* __restrict__ RelProjT,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const int32_t* __restrict__ BT,
    const int32_t* __restrict__ SEQ,
    const int32_t* __restrict__ CUQ,
    const float* __restrict__ ACC_IN,
    const float* __restrict__ M_IN,
    const float* __restrict__ L_IN,
    const uint16_t* __restrict__ GATE,
    const uint16_t* __restrict__ SINK,
    uint16_t* __restrict__ Y,
    __nv_bfloat16* __restrict__ OUT_RAW,
    float* __restrict__ LSE_OUT,
    int64_t sq_t, int64_t sq_h,
    int64_t sr_t, int64_t sr_h,
    int64_t sk_b, int64_t sk_s, int64_t sk_h,
    int64_t sv_b, int64_t sv_s, int64_t sv_h,
    int64_t sbt_r,
    int64_t sg_t, int64_t ssk_t, int64_t sy_t,
    int64_t so_t, int64_t so_h, int64_t sl_t, int64_t sl_h,
    int H, int G, int BS, int window, int E,
    int prefix_bm, int prefix_bn,
    float scale_log2, float rel_threshold, float clip_eps
) {
    static_assert(H_PER_CTA == 1 || H_PER_CTA == 2, "H_PER_CTA must be 1 or 2");
    constexpr int BM_TOK = 128 / H_PER_CTA;  // 64 when H_PER_CTA==2, 128 when H_PER_CTA==1
    constexpr int WARPS_PER_H = WARPS / H_PER_CTA;  // 4 or 8
    constexpr float P_MAX = PV_F16_ACC ? PV_MAX_F16 : PV_MAX_F32;
    constexpr float INV_P_MAX = PV_F16_ACC ? INV_PV_MAX_F16 : INV_PV_MAX_F32;
    constexpr float V_MAX = PV_F16_ACC ? PV_MAX_F16 : PV_MAX_F32;
    constexpr float INV_V_MAX = PV_F16_ACC ? INV_PV_MAX_F16 : INV_PV_MAX_F32;

    // Static SMEM <= 42.5 KiB/CTA -> 2 CTAs/SM = 85 KiB < 99 KiB (zero dynamic carveout needed!)
    __shared__ __align__(16) uint8_t  sK_fp8[OTF_BN * 128];                  // 8,192 B
    __shared__ __align__(16) uint8_t  sVT_fp8[D_HEAD * OTF_BN];              // 8,192 B
    __shared__ __align__(16) uint16_t sR_clip[WARPS][16][16];                // 4,096 B
    __shared__ __align__(16) uint16_t sBias[WARPS][16][OTF_BIAS_STRIDE];     // 21,504 B
    __shared__ __align__(16) float    sKs[OTF_BN];                           // 256 B
    __shared__ OtfFp8CtaParams sP;

    const int tid = static_cast<int>(threadIdx.x);
    if (tid == 0) {
        const int pid = static_cast<int>(blockIdx.x);
        const int req = static_cast<int>(blockIdx.y);
        const int num_hg = H / H_PER_CTA;
        const int hg = pid % num_hg;
        const int mt = pid / num_hg;
        const int h0 = hg * H_PER_CTA;
        const int h1 = (H_PER_CTA == 2) ? (h0 + 1) : h0;
        const int kvh = h0 / G;

        const int q0 = CUQ[req];
        const int qlen = CUQ[req + 1] - q0;
        const int seqlen = SEQ[req];
        const int past = seqlen - qlen;
        const int i_lo = mt * BM_TOK;

        int k_beg = 0;
        int k_end = 0;
        int k_start_exact = 0;

        if (i_lo < qlen && qlen > 0 && seqlen > 0) {
            const int i_hi = min(i_lo + BM_TOK, qlen) - 1;
            const int q_min = past + i_lo;
            const int q_max = past + i_hi;
            k_end = min(q_max + 1, seqlen);
            if constexpr (!IS_GLOBAL_TAIL) {
                const int raw_beg = (window >= 0) ? max(q_min - window, 0) : 0;
                k_beg = raw_beg & ~(OTF_BN - 1);
                k_start_exact = raw_beg;
            } else {
                if (ACC_IN != nullptr && prefix_bm > 0 && prefix_bn > 0) {
                    const int i_lo_p = (i_lo / prefix_bm) * prefix_bm;
                    const int q_min_p = past + i_lo_p;
                    const int k_bias_raw = (E > 0) ? max(q_min_p - E + 1, 0) : max(q_min_p, 0);
                    k_start_exact = (k_bias_raw / prefix_bn) * prefix_bn;
                } else {
                    k_start_exact = 0;
                }
                k_beg = k_start_exact & ~(OTF_BN - 1);
            }
        }

        sP.bt_row        = BT + static_cast<int64_t>(req) * sbt_r;
        sP.K_h           = K + static_cast<int64_t>(kvh) * sk_h;
        sP.V_h           = V + static_cast<int64_t>(kvh) * sv_h;
        sP.RelProjT      = RelProjT;
        sP.GATE          = GATE;
        sP.SINK          = SINK;
        sP.Y             = Y;
        sP.ACC_IN        = ACC_IN;
        sP.M_IN          = M_IN;
        sP.L_IN          = L_IN;
        sP.OUT_RAW       = OUT_RAW;
        sP.LSE_OUT       = LSE_OUT;
        sP.sk_b          = sk_b;
        sP.sk_s          = sk_s;
        sP.sv_b          = sv_b;
        sP.sv_s          = sv_s;
        sP.sg_t          = sg_t;
        sP.ssk_t         = ssk_t;
        sP.sy_t          = sy_t;
        sP.so_t          = so_t;
        sP.so_h          = so_h;
        sP.sl_t          = sl_t;
        sP.sl_h          = sl_h;
        sP.k_sptr        = static_cast<uint32_t>(__cvta_generic_to_shared(sK_fp8));
        sP.vt_sptr       = static_cast<uint32_t>(__cvta_generic_to_shared(sVT_fp8));
        sP.bs_shift      = __ffs(BS) - 1;
        sP.bs_mask       = BS - 1;
        sP.q0            = q0;
        sP.qlen          = qlen;
        sP.seqlen        = seqlen;
        sP.past          = past;
        sP.i_lo          = i_lo;
        sP.h0            = h0;
        sP.h1            = h1;
        sP.H             = H;
        sP.E             = E;
        sP.window        = window;
        sP.k_beg         = k_beg;
        sP.k_end         = k_end;
        sP.k_start_exact = k_start_exact;
    }
    __syncthreads();

    const int i_lo_c = ld_smem_i32_nocse(&sP.i_lo);
    const int qlen_c = ld_smem_i32_nocse(&sP.qlen);
    if (i_lo_c >= qlen_c || qlen_c <= 0) {
        return;
    }

    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int gid  = lane >> 2;
    const int t    = lane & 3;
    const int w_head = (H_PER_CTA == 2) ? (warp >> 2) : 0;
    const int w_m    = warp & (WARPS_PER_H - 1);
    const int h_my   = (w_head == 0) ? ld_smem_i32_nocse(&sP.h0) : ld_smem_i32_nocse(&sP.h1);
    const int q0_c   = ld_smem_i32_nocse(&sP.q0);

    const int r0_loc = i_lo_c + w_m * 16 + gid;
    const int r1_loc = r0_loc + 8;
    const bool r0_ok = (r0_loc < qlen_c);
    const bool r1_ok = (r1_loc < qlen_c);

    // 1) Pre-quantize Q rows into 16 FP8 registers (`qs_0, qs_1` WITHOUT `scale_log2` so FMA order matches OTF bias)
    uint32_t q_a0[4], q_a1[4], q_a2[4], q_a3[4];
    float qs_raw_0, qs_raw_1;
    load_quant_q_regs_v6(
        Q, sq_t, sq_h, q0_c, h_my, r0_loc, r1_loc, r0_ok, r1_ok, t, 1.0f,
        q_a0, q_a1, q_a2, q_a3, qs_raw_0, qs_raw_1
    );

    // 2) Exact 8-lane L2 clip (+ optional Tau) of the 16 R rows of `warp` into `sR_clip[warp]`
    {
        const int sub_r = lane >> 3;     // 0..3
        const int sub_lane = lane & 7;   // 0..7
        const uint16_t* r_scale_row = RScale + static_cast<int64_t>(h_my) * D_REL;
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            const int rw = p * 4 + sub_r;
            const int r_loc = i_lo_c + w_m * 16 + rw;
            const bool rv = (r_loc < qlen_c);
            const int tok = q0_c + r_loc;
            const float tau_v = (HAS_TAU && rv) ? __ldg(Tau + tok) : 1.0f;
            const uint16_t* r_ptr = rv ? (R + static_cast<int64_t>(tok) * sr_t + static_cast<int64_t>(h_my) * sr_h) : nullptr;
            clip_r_row_exact_8lane_v6<HAS_TAU>(
                r_ptr, r_scale_row, rv, tau_v, rel_threshold, clip_eps, sub_lane, sR_clip[warp][rw]
            );
        }
    }

    // 3) Initialize or continue online softmax state `(acc, m_i, l_i)`
    float m_i0 = -INFINITY, m_i1 = -INFINITY;
    float l_i0 = 0.0f, l_i1 = 0.0f;
    float acc[16][4];
    #pragma unroll
    for (int dt = 0; dt < 16; ++dt) {
        acc[dt][0] = 0.0f; acc[dt][1] = 0.0f;
        acc[dt][2] = 0.0f; acc[dt][3] = 0.0f;
    }

    if constexpr (IS_GLOBAL_TAIL) {
        const float* acc_in = ld_smem_ptr_nocse(&sP.ACC_IN);
        const int k_start_e = ld_smem_i32_nocse(&sP.k_start_exact);
        if (acc_in != nullptr && k_start_e > 0) {
            const float* m_in = ld_smem_ptr_nocse(&sP.M_IN);
            const float* l_in = ld_smem_ptr_nocse(&sP.L_IN);
            const int H_tot = ld_smem_i32_nocse(&sP.H);
            if (r0_ok) {
                const int64_t th0 = static_cast<int64_t>(q0_c + r0_loc) * H_tot + h_my;
                const float li0 = m_in != nullptr ? __ldg(&l_in[th0]) : 0.0f;
                if (li0 > 0.0f) {
                    m_i0 = __ldg(&m_in[th0]);
                    l_i0 = li0;
                    const float2* ap0 = reinterpret_cast<const float2*>(acc_in + th0 * D_HEAD);
                    #pragma unroll
                    for (int dt = 0; dt < 16; ++dt) {
                        const float2 v = __ldg(&ap0[dt * 4 + t]);
                        acc[dt][0] = v.x;
                        acc[dt][1] = v.y;
                    }
                }
            }
            if (r1_ok) {
                const int64_t th1 = static_cast<int64_t>(q0_c + r1_loc) * H_tot + h_my;
                const float li1 = m_in != nullptr ? __ldg(&l_in[th1]) : 0.0f;
                if (li1 > 0.0f) {
                    m_i1 = __ldg(&m_in[th1]);
                    l_i1 = li1;
                    const float2* ap1 = reinterpret_cast<const float2*>(acc_in + th1 * D_HEAD);
                    #pragma unroll
                    for (int dt = 0; dt < 16; ++dt) {
                        const float2 v = __ldg(&ap1[dt * 4 + t]);
                        acc[dt][2] = v.x;
                        acc[dt][3] = v.y;
                    }
                }
            }
        }
    }
    __syncthreads();

    // 4) Main BN=64 K-tile loop (shared K & V across H_PER_CTA Q-heads!)
    for (int n0 = ld_smem_i32_nocse(&sP.k_beg); n0 < ld_smem_i32_nocse(&sP.k_end); n0 += OTF_BN) {
        if (tid == 0) {
            const int lb = n0 >> sP.bs_shift;
            sP.cur_blk  = static_cast<int64_t>(sP.bt_row[lb]);
            sP.cur_slot = n0 & sP.bs_mask;
        }
        if (tid < 16) {
            sP.vs_bits[tid] = 0u;
        }
        __syncthreads();

        const int seqlen_cur = ld_smem_i32_nocse(&sP.seqlen);
        const int valid_k = min(OTF_BN, seqlen_cur - n0);
        const int64_t cur_blk = ld_smem_i64_nocse(&sP.cur_blk);
        const int cur_slot = ld_smem_i32_nocse(&sP.cur_slot);

        // A) Cooperative online FP8 quantize of K [64, 128] -> sK_fp8 + per-token sKs[64]
        {
            const int64_t sk_s_cur = ld_smem_i64_nocse(&sP.sk_s);
            const __nv_bfloat16* k_ptr = ld_smem_ptr_nocse(&sP.K_h)
                + cur_blk * ld_smem_i64_nocse(&sP.sk_b)
                + static_cast<int64_t>(cur_slot) * sk_s_cur;
            load_quant_k64_online(k_ptr, sk_s_cur, valid_k, sK_fp8, sKs, tid);
        }
        __syncthreads();

        // B) Per-warp FP8 Q@K^T (`16` m16n8k32.f16 MMAs)
        uint32_t d0_raw[8], d1_raw[8];
        {
            const uint32_t k_sptr = ld_smem_u32_nocse(&sP.k_sptr);
            const int sub = lane >> 3;
            const int r_in = lane & 7;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int nt0 = 2 * j;
                uint32_t c0_0 = 0u, c1_0 = 0u, c0_1 = 0u, c1_1 = 0u;
                const int k_row = (nt0 + (sub >> 1)) * 8 + r_in;
                #pragma unroll
                for (int ks = 0; ks < 4; ++ks) {
                    const int k_chunk = ks * 2 + (sub & 1);
                    const uint32_t addr = k_sptr + static_cast<uint32_t>(k_row * 128 + (k_chunk ^ r_in) * 16);
                    uint32_t b0_0, b1_0, b0_1, b1_1;
                    ldmatrix_x4(b0_0, b1_0, b0_1, b1_1, addr);
                    mma_m16n8k32_f16_e4m3(c0_0, c1_0, q_a0[ks], q_a1[ks], q_a2[ks], q_a3[ks], b0_0, b1_0, c0_0, c1_0);
                    mma_m16n8k32_f16_e4m3(c0_1, c1_1, q_a0[ks], q_a1[ks], q_a2[ks], q_a3[ks], b0_1, b1_1, c0_1, c1_1);
                }
                d0_raw[2 * j]     = c0_0; d1_raw[2 * j]     = c1_0;
                d0_raw[2 * j + 1] = c0_1; d1_raw[2 * j + 1] = c1_1;
            }
        }

        // C) Warp-local exact BF16 OTF bias (skipped automatically when d_min >= E or d_max < 0!)
        const int past_cur = ld_smem_i32_nocse(&sP.past);
        const int E_cur    = ld_smem_i32_nocse(&sP.E);
        const int q_w0     = past_cur + i_lo_c + w_m * 16;
        const int q_w_hi   = min(q_w0 + 15, seqlen_cur - 1);
        const int d_min_w  = q_w0 - (n0 + OTF_BN - 1);
        const int d_max_w  = q_w_hi - n0;
        const bool tile_has_bias = (q_w0 < seqlen_cur) && (E_cur > 0) && (d_max_w >= 0) && (d_min_w < E_cur);
        int d_lo = 0;
        if (tile_has_bias) {
            d_lo = (d_min_w > 0) ? d_min_w : 0;
            const int d_hi = (d_max_w < E_cur - 1) ? d_max_w : (E_cur - 1);
            compute_otf_bias_warp(
                sR_clip[warp], ld_smem_ptr_nocse(&sP.RelProjT), sBias[warp],
                d_lo, d_hi, E_cur, gid, t
            );
        }

        // D) Score dequant + OTF bias + exact causal/window/k_start mask + row tile-max
        const int qpos0 = past_cur + r0_loc;
        const int qpos1 = past_cur + r1_loc;
        const int win_cur = ld_smem_i32_nocse(&sP.window);
        const int k_start_e = ld_smem_i32_nocse(&sP.k_start_exact);

        float s0_lo[8], s0_hi[8], s1_lo[8], s1_hi[8];
        float tm0 = -INFINITY, tm1 = -INFINITY;

        #pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            const int k_off = nt * 8 + 2 * t;
            const int kpos_a = n0 + k_off;
            const int kpos_b = kpos_a + 1;
            const float2 ks_pair = *reinterpret_cast<const float2*>(&sKs[k_off]);
            const float2 f0 = __half22float2(*reinterpret_cast<const __half2*>(&d0_raw[nt]));
            const float2 f1 = __half22float2(*reinterpret_cast<const __half2*>(&d1_raw[nt]));

            const int da0 = qpos0 - kpos_a;
            const int db0 = qpos0 - kpos_b;
            const int da1 = qpos1 - kpos_a;
            const int db1 = qpos1 - kpos_b;

            bool va0, vb0, va1, vb1;
            if constexpr (!IS_GLOBAL_TAIL) {
                va0 = r0_ok && (kpos_a < seqlen_cur) && (da0 >= 0) && (win_cur < 0 || da0 <= win_cur);
                vb0 = r0_ok && (kpos_b < seqlen_cur) && (db0 >= 0) && (win_cur < 0 || db0 <= win_cur);
                va1 = r1_ok && (kpos_a < seqlen_cur) && (da1 >= 0) && (win_cur < 0 || da1 <= win_cur);
                vb1 = r1_ok && (kpos_b < seqlen_cur) && (db1 >= 0) && (win_cur < 0 || db1 <= win_cur);
            } else {
                va0 = r0_ok && (kpos_a < seqlen_cur) && (kpos_a >= k_start_e) && (da0 >= 0);
                vb0 = r0_ok && (kpos_b < seqlen_cur) && (kpos_b >= k_start_e) && (db0 >= 0);
                va1 = r1_ok && (kpos_a < seqlen_cur) && (kpos_a >= k_start_e) && (da1 >= 0);
                vb1 = r1_ok && (kpos_b < seqlen_cur) && (kpos_b >= k_start_e) && (db1 >= 0);
            }

            float qk0_a = f0.x * (qs_raw_0 * ks_pair.x);
            float qk0_b = f0.y * (qs_raw_0 * ks_pair.y);
            float qk1_a = f1.x * (qs_raw_1 * ks_pair.x);
            float qk1_b = f1.y * (qs_raw_1 * ks_pair.y);

            if (tile_has_bias) {
                const float ba0 = (va0 && da0 < E_cur) ? bf16_to_f32(sBias[warp][gid][da0 - d_lo]) : 0.0f;
                const float bb0 = (vb0 && db0 < E_cur) ? bf16_to_f32(sBias[warp][gid][db0 - d_lo]) : 0.0f;
                const float ba1 = (va1 && da1 < E_cur) ? bf16_to_f32(sBias[warp][gid + 8][da1 - d_lo]) : 0.0f;
                const float bb1 = (vb1 && db1 < E_cur) ? bf16_to_f32(sBias[warp][gid + 8][db1 - d_lo]) : 0.0f;
                s0_lo[nt] = va0 ? fmaf(scale_log2, qk0_a, __fmul_rn(ba0, LOG2E_F)) : -INFINITY;
                s0_hi[nt] = vb0 ? fmaf(scale_log2, qk0_b, __fmul_rn(bb0, LOG2E_F)) : -INFINITY;
                s1_lo[nt] = va1 ? fmaf(scale_log2, qk1_a, __fmul_rn(ba1, LOG2E_F)) : -INFINITY;
                s1_hi[nt] = vb1 ? fmaf(scale_log2, qk1_b, __fmul_rn(bb1, LOG2E_F)) : -INFINITY;
            } else {
                s0_lo[nt] = va0 ? __fmul_rn(qk0_a, scale_log2) : -INFINITY;
                s0_hi[nt] = vb0 ? __fmul_rn(qk0_b, scale_log2) : -INFINITY;
                s1_lo[nt] = va1 ? __fmul_rn(qk1_a, scale_log2) : -INFINITY;
                s1_hi[nt] = vb1 ? __fmul_rn(qk1_b, scale_log2) : -INFINITY;
            }
            tm0 = fmaxf(tm0, fmaxf(s0_lo[nt], s0_hi[nt]));
            tm1 = fmaxf(tm1, fmaxf(s1_lo[nt], s1_hi[nt]));
        }

        tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffff, tm0, 1));
        tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffff, tm0, 2));
        tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffff, tm1, 1));
        tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffff, tm1, 2));

        // E) Cooperative V Pass 1 (amax) + Warp Online Softmax & Quad Repack
        const int64_t sv_s_cur = ld_smem_i64_nocse(&sP.sv_s);
        const __nv_bfloat16* v_ptr = ld_smem_ptr_nocse(&sP.V_h)
            + cur_blk * ld_smem_i64_nocse(&sP.sv_b)
            + static_cast<int64_t>(cur_slot) * sv_s_cur;

        v64_pass1_amax(v_ptr, sv_s_cur, valid_k, sP.vs_bits, tid);

        const bool row_has_keys = (tm0 >= NEG_INF_GUARD) || (tm1 >= NEG_INF_GUARD);
        const bool warp_active = (__any_sync(0xffffffff, row_has_keys) != 0);
        uint32_t pa[2][4];
        float w_row0 = 0.0f, w_row1 = 0.0f;

        if (warp_active) {
            const float mn0_raw = fmaxf(m_i0, tm0);
            const float mn1_raw = fmaxf(m_i1, tm1);
            const float mn0 = (mn0_raw < NEG_INF_GUARD) ? 0.0f : mn0_raw;
            const float mn1 = (mn1_raw < NEG_INF_GUARD) ? 0.0f : mn1_raw;
            const float a0 = (m_i0 < NEG_INF_GUARD) ? 0.0f : fast_exp2f(m_i0 - mn0);
            const float a1 = (m_i1 < NEG_INF_GUARD) ? 0.0f : fast_exp2f(m_i1 - mn1);

            if (__any_sync(0xffffffff, a0 != 1.0f)) {
                #pragma unroll
                for (int dt = 0; dt < 16; ++dt) {
                    acc[dt][0] *= a0;
                    acc[dt][1] *= a0;
                }
            }
            if (__any_sync(0xffffffff, a1 != 1.0f)) {
                #pragma unroll
                for (int dt = 0; dt < 16; ++dt) {
                    acc[dt][2] *= a1;
                    acc[dt][3] *= a1;
                }
            }

            const float tm0_safe = (tm0 < NEG_INF_GUARD) ? 0.0f : tm0;
            const float tm1_safe = (tm1 < NEG_INF_GUARD) ? 0.0f : tm1;
            const float tscale0 = (tm0 < NEG_INF_GUARD) ? 0.0f : fast_exp2f(tm0_safe - mn0);
            const float tscale1 = (tm1 < NEG_INF_GUARD) ? 0.0f : fast_exp2f(tm1_safe - mn1);

            float sum0 = 0.0f, sum1 = 0.0f;
            uint16_t u0[8], u1[8];
            #pragma unroll
            for (int nt = 0; nt < 8; ++nt) {
                const float p00 = (s0_lo[nt] < NEG_INF_GUARD) ? 0.0f : fast_exp2f(s0_lo[nt] - tm0_safe);
                const float p01 = (s0_hi[nt] < NEG_INF_GUARD) ? 0.0f : fast_exp2f(s0_hi[nt] - tm0_safe);
                const float p10 = (s1_lo[nt] < NEG_INF_GUARD) ? 0.0f : fast_exp2f(s1_lo[nt] - tm1_safe);
                const float p11 = (s1_hi[nt] < NEG_INF_GUARD) ? 0.0f : fast_exp2f(s1_hi[nt] - tm1_safe);
                sum0 += p00 + p01;
                sum1 += p10 + p11;
                u0[nt] = f32x2_to_e4m3x2_rn(p00 * P_MAX, p01 * P_MAX);
                u1[nt] = f32x2_to_e4m3x2_rn(p10 * P_MAX, p11 * P_MAX);
            }

            sum0 += __shfl_xor_sync(0xffffffff, sum0, 1);
            sum0 += __shfl_xor_sync(0xffffffff, sum0, 2);
            sum1 += __shfl_xor_sync(0xffffffff, sum1, 1);
            sum1 += __shfl_xor_sync(0xffffffff, sum1, 2);
            l_i0 = fmaf(tscale0, sum0, l_i0 * a0);
            l_i1 = fmaf(tscale1, sum1, l_i1 * a1);
            m_i0 = mn0_raw;
            m_i1 = mn1_raw;
            w_row0 = tscale0 * INV_P_MAX;
            w_row1 = tscale1 * INV_P_MAX;

            repack_p_quad_half16(u0[0], u0[1], u1[0], u1[1], gid, t, pa[0][0], pa[0][1]);
            repack_p_quad_half16(u0[2], u0[3], u1[2], u1[3], gid, t, pa[0][2], pa[0][3]);
            repack_p_quad_half16(u0[4], u0[5], u1[4], u1[5], gid, t, pa[1][0], pa[1][1]);
            repack_p_quad_half16(u0[6], u0[7], u1[6], u1[7], gid, t, pa[1][2], pa[1][3]);
        }
        __syncthreads();

        if (tid < 16) {
            const float am = fmaxf(__uint_as_float(sP.vs_bits[tid]), 1.0e-12f);
            sP.vs_tile[tid] = am * INV_V_MAX;
            sP.vs_inv[tid]  = div_rn_ftz(V_MAX, am);
        }
        __syncthreads();

        const uint32_t vt_sptr = ld_smem_u32_nocse(&sP.vt_sptr);
        v64_pass2_quant_transpose(v_ptr, sv_s_cur, valid_k, sP.vs_inv, vt_sptr, tid);
        __syncthreads();

        if (warp_active) {
            execute_pv_fp8_m16n8k32<PV_F16_ACC>(
                pa, w_row0, w_row1, vt_sptr, sP.vs_tile, lane, acc);
        }
        __syncthreads();
    }

    // 5) Fused Gate + Sink (+ optional raw OUT/LSE) Epilogue
    if (r0_ok) {
        store_otf_row_epilogue(q0_c + r0_loc, h_my, m_i0, l_i0, acc, 0, t, sP);
    }
    if (r1_ok) {
        store_otf_row_epilogue(q0_c + r1_loc, h_my, m_i1, l_i1, acc, 2, t, sP);
    }
}

// =================================================================================================
// Host Launch Dispatchers & Strict Prefill-Only Guards
// =================================================================================================

inline bool is_stream_capturing(cudaStream_t stream) {
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    cudaError_t err = cudaStreamIsCapturing(stream, &status);
    if (err != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return status != cudaStreamCaptureStatusNone;
}

template <int BN, int STAGES, int MIN_CTAS, bool SPARSE_SEL, int K_BT_MODE, bool PV_F16_ACC>
inline void launch_sage2_v6_impl(
    dim3 grid, dim3 block, cudaStream_t stream,
    const void* q_ptr, const void* k_fp8_ptr, const void* ks64_ptr, const void* v_ptr,
    const void* bt_ptr, const void* k_bt_ptr, const void* seq_ptr, const void* cuq_ptr,
    const void* sel_blocks_ptr, const void* sel_counts_ptr,
    void* acc_out_ptr, void* m_out_ptr, void* l_out_ptr,
    int64_t sq_t, int64_t sq_h,
    int64_t sk8_b, int64_t sk8_s, int64_t sk8_h,
    int64_t sks64_b, int64_t sks64_h,
    int64_t sv_b, int64_t sv_s, int64_t sv_h,
    int64_t sbt_r, int64_t skbt_r,
    int H, int G, int BS, int E, int sel_stride,
    float scale_log2, float pv_skip_tau
) {
    // K stages (24 or 32 KiB) + 8 KiB V_T = 32 or 40 KiB dynamic SMEM (<< 48 KiB default, 2 CTAs/SM!)
    constexpr size_t smem_bytes = STAGES * BN * 128 + D_HEAD * 64;
    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(
            global_sage2_prefix_v6_sm121_kernel<BN, STAGES, MIN_CTAS, SPARSE_SEL, K_BT_MODE, PV_F16_ACC>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem_bytes));
        cudaFuncSetAttribute(
            global_sage2_prefix_v6_sm121_kernel<BN, STAGES, MIN_CTAS, SPARSE_SEL, K_BT_MODE, PV_F16_ACC>,
            cudaFuncAttributePreferredSharedMemoryCarveout,
            cudaSharedmemCarveoutMaxShared);
        attr_set = true;
    }
    global_sage2_prefix_v6_sm121_kernel<BN, STAGES, MIN_CTAS, SPARSE_SEL, K_BT_MODE, PV_F16_ACC>
        <<<grid, block, smem_bytes, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(q_ptr),
            reinterpret_cast<const uint8_t*>(k_fp8_ptr),
            reinterpret_cast<const float*>(ks64_ptr),
            reinterpret_cast<const __nv_bfloat16*>(v_ptr),
            reinterpret_cast<const int32_t*>(bt_ptr),
            reinterpret_cast<const int32_t*>(k_bt_ptr),
            reinterpret_cast<const int32_t*>(seq_ptr),
            reinterpret_cast<const int32_t*>(cuq_ptr),
            reinterpret_cast<const int32_t*>(sel_blocks_ptr),
            reinterpret_cast<const int32_t*>(sel_counts_ptr),
            reinterpret_cast<float*>(acc_out_ptr),
            reinterpret_cast<float*>(m_out_ptr),
            reinterpret_cast<float*>(l_out_ptr),
            sq_t, sq_h,
            sk8_b, sk8_s, sk8_h,
            sks64_b, sks64_h,
            sv_b, sv_s, sv_h,
            sbt_r, skbt_r,
            H, G, BS, E, sel_stride,
            scale_log2,
            pv_skip_tau
        );
}

template <int K_BT_MODE, bool PV_F16_ACC>
inline int dispatch_sage2_v6_kbt(
    dim3 grid, dim3 block, cudaStream_t stream,
    bool sparse, int bn, int min_ctas,
    const void* q_ptr, const void* k_fp8_ptr, const void* ks64_ptr, const void* v_ptr,
    const void* bt_ptr, const void* k_bt_ptr, const void* seq_ptr, const void* cuq_ptr,
    const void* sel_blocks_ptr, const void* sel_counts_ptr,
    void* acc_out_ptr, void* m_out_ptr, void* l_out_ptr,
    int64_t sq_t, int64_t sq_h,
    int64_t sk8_b, int64_t sk8_s, int64_t sk8_h,
    int64_t sks64_b, int64_t sks64_h,
    int64_t sv_b, int64_t sv_s, int64_t sv_h,
    int64_t sbt_r, int64_t skbt_r,
    int H, int G, int BS, int E, int sel_stride,
    float scale_log2, float pv_skip_tau
) {
    if (!sparse) {
        if (bn == 128 && min_ctas == 2) {
            launch_sage2_v6_impl<128, 2, 2, false, K_BT_MODE, PV_F16_ACC>(
                grid, block, stream, q_ptr, k_fp8_ptr, ks64_ptr, v_ptr, bt_ptr, k_bt_ptr, seq_ptr, cuq_ptr,
                nullptr, nullptr, acc_out_ptr, m_out_ptr, l_out_ptr,
                sq_t, sq_h, sk8_b, sk8_s, sk8_h, sks64_b, sks64_h, sv_b, sv_s, sv_h, sbt_r, skbt_r,
                H, G, BS, E, 0, scale_log2, pv_skip_tau);
        } else if (bn == 64 && min_ctas == 2) {
            launch_sage2_v6_impl<64, 3, 2, false, K_BT_MODE, PV_F16_ACC>(
                grid, block, stream, q_ptr, k_fp8_ptr, ks64_ptr, v_ptr, bt_ptr, k_bt_ptr, seq_ptr, cuq_ptr,
                nullptr, nullptr, acc_out_ptr, m_out_ptr, l_out_ptr,
                sq_t, sq_h, sk8_b, sk8_s, sk8_h, sks64_b, sks64_h, sv_b, sv_s, sv_h, sbt_r, skbt_r,
                H, G, BS, E, 0, scale_log2, pv_skip_tau);
        } else if (bn == 128) {
            launch_sage2_v6_impl<128, 2, 1, false, K_BT_MODE, PV_F16_ACC>(
                grid, block, stream, q_ptr, k_fp8_ptr, ks64_ptr, v_ptr, bt_ptr, k_bt_ptr, seq_ptr, cuq_ptr,
                nullptr, nullptr, acc_out_ptr, m_out_ptr, l_out_ptr,
                sq_t, sq_h, sk8_b, sk8_s, sk8_h, sks64_b, sks64_h, sv_b, sv_s, sv_h, sbt_r, skbt_r,
                H, G, BS, E, 0, scale_log2, pv_skip_tau);
        } else {
            launch_sage2_v6_impl<64, 3, 1, false, K_BT_MODE, PV_F16_ACC>(
                grid, block, stream, q_ptr, k_fp8_ptr, ks64_ptr, v_ptr, bt_ptr, k_bt_ptr, seq_ptr, cuq_ptr,
                nullptr, nullptr, acc_out_ptr, m_out_ptr, l_out_ptr,
                sq_t, sq_h, sk8_b, sk8_s, sk8_h, sks64_b, sks64_h, sv_b, sv_s, sv_h, sbt_r, skbt_r,
                H, G, BS, E, 0, scale_log2, pv_skip_tau);
        }
    } else {
        if (bn == 128) {
            launch_sage2_v6_impl<128, 2, 2, true, K_BT_MODE, PV_F16_ACC>(
                grid, block, stream, q_ptr, k_fp8_ptr, ks64_ptr, v_ptr, bt_ptr, k_bt_ptr, seq_ptr, cuq_ptr,
                sel_blocks_ptr, sel_counts_ptr, acc_out_ptr, m_out_ptr, l_out_ptr,
                sq_t, sq_h, sk8_b, sk8_s, sk8_h, sks64_b, sks64_h, sv_b, sv_s, sv_h, sbt_r, skbt_r,
                H, G, BS, E, sel_stride, scale_log2, pv_skip_tau);
        } else {
            launch_sage2_v6_impl<64, 3, 2, true, K_BT_MODE, PV_F16_ACC>(
                grid, block, stream, q_ptr, k_fp8_ptr, ks64_ptr, v_ptr, bt_ptr, k_bt_ptr, seq_ptr, cuq_ptr,
                sel_blocks_ptr, sel_counts_ptr, acc_out_ptr, m_out_ptr, l_out_ptr,
                sq_t, sq_h, sk8_b, sk8_s, sk8_h, sks64_b, sks64_h, sv_b, sv_s, sv_h, sbt_r, skbt_r,
                H, G, BS, E, sel_stride, scale_log2, pv_skip_tau);
        }
    }
    return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

}  // namespace

// =================================================================================================
// Exported C ABI Entrypoints
// =================================================================================================

extern "C" int pf_fp8_attn_v6_version() {
    return 600;
}

extern "C" int global_sage2_prefix_v6_sm121_fwd(
    const void* q_ptr,
    const void* k_fp8_ptr,
    const void* ks64_ptr,
    const void* v_ptr,
    const void* bt_ptr,
    const void* seq_ptr,
    const void* cuq_ptr,
    const void* sel_blocks_ptr,
    const void* sel_counts_ptr,
    void* acc_out_ptr,
    void* m_out_ptr,
    void* l_out_ptr,
    int64_t sq_t, int64_t sq_h,
    int64_t sk8_b, int64_t sk8_s, int64_t sk8_h,
    int64_t sks64_b, int64_t sks64_h,
    int64_t sv_b, int64_t sv_s, int64_t sv_h,
    int64_t sbt_r,
    int num_reqs, int max_seqlen_q, int H, int G,
    int BS, int E, int bn, int min_ctas, int sel_stride,
    float scale_log2,
    float pv_skip_tau,
    const void* k_bt_ptr,
    int64_t skbt_r,
    int k_bt_mode,
    int pv_f16_acc,
    uint64_t stream_u64
) {
    if (max_seqlen_q < 1024) return -3;
    if (bn != 64 && bn != 128) return -1;
    if (min_ctas != 1 && min_ctas != 2) return -1;
    if (k_bt_mode < 0 || k_bt_mode > 2) return -1;
    if (k_bt_mode == 2 && k_bt_ptr == nullptr) return -1;
    if (num_reqs <= 0 || H <= 0 || G <= 0 || BS < bn || (BS % bn) != 0) return -1;
    if ((BS & (BS - 1)) != 0) return -1;
    if ((reinterpret_cast<uintptr_t>(q_ptr) & 7u) != 0 ||
        (reinterpret_cast<uintptr_t>(k_fp8_ptr) & 15u) != 0 ||
        (reinterpret_cast<uintptr_t>(v_ptr) & 15u) != 0 ||
        (reinterpret_cast<uintptr_t>(acc_out_ptr) & 7u) != 0) {
        return -1;
    }
    if ((sq_t & 3) != 0 || (sq_h & 3) != 0 ||
        (sk8_b & 15) != 0 || (sk8_s & 15) != 0 || (sk8_h & 15) != 0 ||
        (sv_b & 7) != 0 || (sv_s & 7) != 0 || (sv_h & 7) != 0) {
        return -1;
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_u64);
    if (is_stream_capturing(stream)) return -4;

    const bool sparse = (sel_blocks_ptr != nullptr && sel_counts_ptr != nullptr && sel_stride > 0);
    const int num_mt = (max_seqlen_q + SAGE_BM - 1) / SAGE_BM;
    dim3 grid(num_mt * H, num_reqs, 1);
    dim3 block(THREADS, 1, 1);

    if (pv_f16_acc != 0) {
        if (k_bt_mode == 1) {
            return dispatch_sage2_v6_kbt<1, true>(
                grid, block, stream, sparse, bn, min_ctas,
                q_ptr, k_fp8_ptr, ks64_ptr, v_ptr, bt_ptr, nullptr, seq_ptr, cuq_ptr,
                sel_blocks_ptr, sel_counts_ptr, acc_out_ptr, m_out_ptr, l_out_ptr,
                sq_t, sq_h, sk8_b, sk8_s, sk8_h, sks64_b, sks64_h, sv_b, sv_s, sv_h, sbt_r, 0,
                H, G, BS, E, sel_stride, scale_log2, pv_skip_tau);
        }
        if (k_bt_mode == 2) {
            return dispatch_sage2_v6_kbt<2, true>(
                grid, block, stream, sparse, bn, min_ctas,
                q_ptr, k_fp8_ptr, ks64_ptr, v_ptr, bt_ptr, k_bt_ptr, seq_ptr, cuq_ptr,
                sel_blocks_ptr, sel_counts_ptr, acc_out_ptr, m_out_ptr, l_out_ptr,
                sq_t, sq_h, sk8_b, sk8_s, sk8_h, sks64_b, sks64_h, sv_b, sv_s, sv_h, sbt_r, skbt_r,
                H, G, BS, E, sel_stride, scale_log2, pv_skip_tau);
        }
        return dispatch_sage2_v6_kbt<0, true>(
            grid, block, stream, sparse, bn, min_ctas,
            q_ptr, k_fp8_ptr, ks64_ptr, v_ptr, bt_ptr, nullptr, seq_ptr, cuq_ptr,
            sel_blocks_ptr, sel_counts_ptr, acc_out_ptr, m_out_ptr, l_out_ptr,
            sq_t, sq_h, sk8_b, sk8_s, sk8_h, sks64_b, sks64_h, sv_b, sv_s, sv_h, sbt_r, 0,
            H, G, BS, E, sel_stride, scale_log2, pv_skip_tau);
    }

    if (k_bt_mode == 1) {
        return dispatch_sage2_v6_kbt<1, false>(
            grid, block, stream, sparse, bn, min_ctas,
            q_ptr, k_fp8_ptr, ks64_ptr, v_ptr, bt_ptr, nullptr, seq_ptr, cuq_ptr,
            sel_blocks_ptr, sel_counts_ptr, acc_out_ptr, m_out_ptr, l_out_ptr,
            sq_t, sq_h, sk8_b, sk8_s, sk8_h, sks64_b, sks64_h, sv_b, sv_s, sv_h, sbt_r, 0,
            H, G, BS, E, sel_stride, scale_log2, pv_skip_tau);
    }
    if (k_bt_mode == 2) {
        return dispatch_sage2_v6_kbt<2, false>(
            grid, block, stream, sparse, bn, min_ctas,
            q_ptr, k_fp8_ptr, ks64_ptr, v_ptr, bt_ptr, k_bt_ptr, seq_ptr, cuq_ptr,
            sel_blocks_ptr, sel_counts_ptr, acc_out_ptr, m_out_ptr, l_out_ptr,
            sq_t, sq_h, sk8_b, sk8_s, sk8_h, sks64_b, sks64_h, sv_b, sv_s, sv_h, sbt_r, skbt_r,
            H, G, BS, E, sel_stride, scale_log2, pv_skip_tau);
    }
    return dispatch_sage2_v6_kbt<0, false>(
        grid, block, stream, sparse, bn, min_ctas,
        q_ptr, k_fp8_ptr, ks64_ptr, v_ptr, bt_ptr, nullptr, seq_ptr, cuq_ptr,
        sel_blocks_ptr, sel_counts_ptr, acc_out_ptr, m_out_ptr, l_out_ptr,
        sq_t, sq_h, sk8_b, sk8_s, sk8_h, sks64_b, sks64_h, sv_b, sv_s, sv_h, sbt_r, 0,
        H, G, BS, E, sel_stride, scale_log2, pv_skip_tau);
}

// Drop-in Compact Shadow K v5 / v3 / v1 compatibility exports
extern "C" int global_sage2_prefix_v5_sm121_fwd(
    const void* q_ptr, const void* k_fp8_ptr, const void* ks64_ptr, const void* v_ptr,
    const void* bt_ptr, const void* seq_ptr, const void* cuq_ptr,
    const void* sel_blocks_ptr, const void* sel_counts_ptr,
    void* acc_out_ptr, void* m_out_ptr, void* l_out_ptr,
    int64_t sq_t, int64_t sq_h,
    int64_t sk8_b, int64_t sk8_s, int64_t sk8_h,
    int64_t sks64_b, int64_t sks64_h,
    int64_t sv_b, int64_t sv_s, int64_t sv_h,
    int64_t sbt_r,
    int num_reqs, int max_seqlen_q, int H, int G,
    int BS, int E, int bn, int min_ctas, int sel_stride,
    float scale_log2, float pv_skip_tau,
    const void* k_bt_ptr, int64_t skbt_r, int k_bt_mode,
    uint64_t stream_u64
) {
    return global_sage2_prefix_v6_sm121_fwd(
        q_ptr, k_fp8_ptr, ks64_ptr, v_ptr, bt_ptr, seq_ptr, cuq_ptr,
        sel_blocks_ptr, sel_counts_ptr, acc_out_ptr, m_out_ptr, l_out_ptr,
        sq_t, sq_h, sk8_b, sk8_s, sk8_h, sks64_b, sks64_h, sv_b, sv_s, sv_h, sbt_r,
        num_reqs, max_seqlen_q, H, G, BS, E, bn, min_ctas, sel_stride,
        scale_log2, pv_skip_tau,
        k_bt_ptr, skbt_r, k_bt_mode, 0, stream_u64);
}

extern "C" int global_sage2_prefix_v3_sm121_fwd(
    const void* q_ptr, const void* k_fp8_ptr, const void* ks64_ptr, const void* v_ptr,
    const void* bt_ptr, const void* seq_ptr, const void* cuq_ptr,
    const void* sel_blocks_ptr, const void* sel_counts_ptr,
    void* acc_out_ptr, void* m_out_ptr, void* l_out_ptr,
    int64_t sq_t, int64_t sq_h,
    int64_t sk8_b, int64_t sk8_s, int64_t sk8_h,
    int64_t sks64_b, int64_t sks64_h,
    int64_t sv_b, int64_t sv_s, int64_t sv_h,
    int64_t sbt_r,
    int num_reqs, int max_seqlen_q, int H, int G,
    int BS, int E, int bn, int min_ctas, int sel_stride,
    float scale_log2, float pv_skip_tau,
    uint64_t stream_u64
) {
    return global_sage2_prefix_v6_sm121_fwd(
        q_ptr, k_fp8_ptr, ks64_ptr, v_ptr, bt_ptr, seq_ptr, cuq_ptr,
        sel_blocks_ptr, sel_counts_ptr, acc_out_ptr, m_out_ptr, l_out_ptr,
        sq_t, sq_h, sk8_b, sk8_s, sk8_h, sks64_b, sks64_h, sv_b, sv_s, sv_h, sbt_r,
        num_reqs, max_seqlen_q, H, G, BS, E, bn, min_ctas, sel_stride,
        scale_log2, pv_skip_tau,
        nullptr, 0, 0, 0, stream_u64);
}

extern "C" int global_sage2_prefix_sm121_fwd(
    const void*, const void*, const void*, const void*, const void*, const void*,
    void*, void*, void*,
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    int, int, int, int, int, int, float, float, uint64_t
) {
    return -1;
}

extern "C" int score_kmax_blocks_fp8_sm121_fwd(
    const void* q_ptr, const void* k_mid_ptr, const void* k_rad_ptr,
    const void* bt_ptr, const void* seq_ptr, const void* cuq_ptr,
    void* scores_ptr, void* sel_counts_ptr,
    int64_t sq_t, int64_t sq_h,
    int64_t sm_b, int64_t sm_h,
    int64_t sbt_r, int64_t ssc_cta,
    int num_reqs, int max_seqlen_q, int max_pblks,
    int H, int G, int BS, int E,
    int sink_blocks, int local_blocks, int k_sel,
    uint64_t stream_u64
) {
    if (max_seqlen_q < 1024) return -3;
    if (num_reqs <= 0 || max_pblks <= 0 || H <= 0 || G <= 0 || BS <= 0) return -1;
    if ((reinterpret_cast<uintptr_t>(q_ptr) & 7u) != 0 ||
        (reinterpret_cast<uintptr_t>(k_mid_ptr) & 15u) != 0 ||
        (reinterpret_cast<uintptr_t>(k_rad_ptr) & 15u) != 0) {
        return -1;
    }
    if ((sq_t & 3) != 0 || (sq_h & 3) != 0 || (sm_b & 7) != 0 || (sm_h & 7) != 0) {
        return -1;
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_u64);
    if (is_stream_capturing(stream)) return -4;

    const int num_mt = (max_seqlen_q + SAGE_BM - 1) / SAGE_BM;
    dim3 grid(num_mt * H, num_reqs, 1);
    dim3 block(THREADS, 1, 1);
    score_kmax_blocks_fp8_sm121_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q_ptr),
        reinterpret_cast<const __nv_bfloat16*>(k_mid_ptr),
        reinterpret_cast<const __nv_bfloat16*>(k_rad_ptr),
        reinterpret_cast<const int32_t*>(bt_ptr),
        reinterpret_cast<const int32_t*>(seq_ptr),
        reinterpret_cast<const int32_t*>(cuq_ptr),
        reinterpret_cast<float*>(scores_ptr),
        reinterpret_cast<int32_t*>(sel_counts_ptr),
        sq_t, sq_h, sm_b, sm_h, sbt_r, ssc_cta,
        max_pblks, H, G, BS, E, sink_blocks, local_blocks, k_sel
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

extern "C" int local_swa_fp8_otf_rel_v6_sm121_fwd(
    const void* q_ptr, const void* r_ptr, const void* r_scale_ptr, const void* rel_proj_t_ptr,
    const void* k_ptr, const void* v_ptr,
    const void* bt_ptr, const void* seq_ptr, const void* cuq_ptr,
    const void* gate_ptr, const void* sink_ptr, void* y_ptr,
    void* out_raw_ptr, void* lse_out_ptr,
    int64_t sq_t, int64_t sq_h, int64_t sr_t, int64_t sr_h,
    int64_t sk_b, int64_t sk_s, int64_t sk_h,
    int64_t sv_b, int64_t sv_s, int64_t sv_h,
    int64_t sbt_r, int64_t sg_t, int64_t ssk_t, int64_t sy_t,
    int64_t so_t, int64_t so_h, int64_t sl_t, int64_t sl_h,
    int num_reqs, int max_seqlen_q, int H, int G,
    int BS, int window, int E, int pv_f16_acc,
    float scale_log2, float rel_threshold, float clip_eps,
    uint64_t stream_u64
) {
    if (max_seqlen_q < 1024) return -3;
    if (num_reqs <= 0 || H <= 0 || G <= 0 || (H % G) != 0 || BS < 64 || (BS & (BS - 1)) != 0) return -1;
    if ((reinterpret_cast<uintptr_t>(q_ptr) & 7u) != 0 ||
        (reinterpret_cast<uintptr_t>(r_ptr) & 3u) != 0 ||
        (reinterpret_cast<uintptr_t>(r_scale_ptr) & 3u) != 0 ||
        (reinterpret_cast<uintptr_t>(rel_proj_t_ptr) & 3u) != 0 ||
        (reinterpret_cast<uintptr_t>(k_ptr) & 15u) != 0 ||
        (reinterpret_cast<uintptr_t>(v_ptr) & 15u) != 0) {
        return -1;
    }
    if ((sq_t & 3) != 0 || (sq_h & 3) != 0 || (sr_t & 1) != 0 || (sr_h & 1) != 0 ||
        (sk_b & 7) != 0 || (sk_s & 7) != 0 || (sk_h & 7) != 0 ||
        (sv_b & 7) != 0 || (sv_s & 7) != 0 || (sv_h & 7) != 0) {
        return -1;
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_u64);
    if (is_stream_capturing(stream)) return -4;

    dim3 block(THREADS, 1, 1);
    if ((G & 1) == 0) {
        const int num_mt = (max_seqlen_q + 63) / 64;
        dim3 grid(num_mt * (H / 2), num_reqs, 1);
        if (pv_f16_acc != 0) {
            otf_rel_fp8_v6_sm121_kernel<2, false, false, true><<<grid, block, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(q_ptr),
                reinterpret_cast<const uint16_t*>(r_ptr),
                reinterpret_cast<const uint16_t*>(r_scale_ptr),
                nullptr,
                reinterpret_cast<const uint16_t*>(rel_proj_t_ptr),
                reinterpret_cast<const __nv_bfloat16*>(k_ptr),
                reinterpret_cast<const __nv_bfloat16*>(v_ptr),
                reinterpret_cast<const int32_t*>(bt_ptr),
                reinterpret_cast<const int32_t*>(seq_ptr),
                reinterpret_cast<const int32_t*>(cuq_ptr),
                nullptr, nullptr, nullptr,
                reinterpret_cast<const uint16_t*>(gate_ptr),
                reinterpret_cast<const uint16_t*>(sink_ptr),
                reinterpret_cast<uint16_t*>(y_ptr),
                reinterpret_cast<__nv_bfloat16*>(out_raw_ptr),
                reinterpret_cast<float*>(lse_out_ptr),
                sq_t, sq_h, sr_t, sr_h, sk_b, sk_s, sk_h, sv_b, sv_s, sv_h,
                sbt_r, sg_t, ssk_t, sy_t, so_t, so_h, sl_t, sl_h,
                H, G, BS, window, E, 128, 64, scale_log2, rel_threshold, clip_eps);
        } else {
            otf_rel_fp8_v6_sm121_kernel<2, false, false, false><<<grid, block, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(q_ptr),
                reinterpret_cast<const uint16_t*>(r_ptr),
                reinterpret_cast<const uint16_t*>(r_scale_ptr),
                nullptr,
                reinterpret_cast<const uint16_t*>(rel_proj_t_ptr),
                reinterpret_cast<const __nv_bfloat16*>(k_ptr),
                reinterpret_cast<const __nv_bfloat16*>(v_ptr),
                reinterpret_cast<const int32_t*>(bt_ptr),
                reinterpret_cast<const int32_t*>(seq_ptr),
                reinterpret_cast<const int32_t*>(cuq_ptr),
                nullptr, nullptr, nullptr,
                reinterpret_cast<const uint16_t*>(gate_ptr),
                reinterpret_cast<const uint16_t*>(sink_ptr),
                reinterpret_cast<uint16_t*>(y_ptr),
                reinterpret_cast<__nv_bfloat16*>(out_raw_ptr),
                reinterpret_cast<float*>(lse_out_ptr),
                sq_t, sq_h, sr_t, sr_h, sk_b, sk_s, sk_h, sv_b, sv_s, sv_h,
                sbt_r, sg_t, ssk_t, sy_t, so_t, so_h, sl_t, sl_h,
                H, G, BS, window, E, 128, 64, scale_log2, rel_threshold, clip_eps);
        }
    } else {
        const int num_mt = (max_seqlen_q + 127) / 128;
        dim3 grid(num_mt * H, num_reqs, 1);
        if (pv_f16_acc != 0) {
            otf_rel_fp8_v6_sm121_kernel<1, false, false, true><<<grid, block, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(q_ptr),
                reinterpret_cast<const uint16_t*>(r_ptr),
                reinterpret_cast<const uint16_t*>(r_scale_ptr),
                nullptr,
                reinterpret_cast<const uint16_t*>(rel_proj_t_ptr),
                reinterpret_cast<const __nv_bfloat16*>(k_ptr),
                reinterpret_cast<const __nv_bfloat16*>(v_ptr),
                reinterpret_cast<const int32_t*>(bt_ptr),
                reinterpret_cast<const int32_t*>(seq_ptr),
                reinterpret_cast<const int32_t*>(cuq_ptr),
                nullptr, nullptr, nullptr,
                reinterpret_cast<const uint16_t*>(gate_ptr),
                reinterpret_cast<const uint16_t*>(sink_ptr),
                reinterpret_cast<uint16_t*>(y_ptr),
                reinterpret_cast<__nv_bfloat16*>(out_raw_ptr),
                reinterpret_cast<float*>(lse_out_ptr),
                sq_t, sq_h, sr_t, sr_h, sk_b, sk_s, sk_h, sv_b, sv_s, sv_h,
                sbt_r, sg_t, ssk_t, sy_t, so_t, so_h, sl_t, sl_h,
                H, G, BS, window, E, 128, 64, scale_log2, rel_threshold, clip_eps);
        } else {
            otf_rel_fp8_v6_sm121_kernel<1, false, false, false><<<grid, block, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(q_ptr),
                reinterpret_cast<const uint16_t*>(r_ptr),
                reinterpret_cast<const uint16_t*>(r_scale_ptr),
                nullptr,
                reinterpret_cast<const uint16_t*>(rel_proj_t_ptr),
                reinterpret_cast<const __nv_bfloat16*>(k_ptr),
                reinterpret_cast<const __nv_bfloat16*>(v_ptr),
                reinterpret_cast<const int32_t*>(bt_ptr),
                reinterpret_cast<const int32_t*>(seq_ptr),
                reinterpret_cast<const int32_t*>(cuq_ptr),
                nullptr, nullptr, nullptr,
                reinterpret_cast<const uint16_t*>(gate_ptr),
                reinterpret_cast<const uint16_t*>(sink_ptr),
                reinterpret_cast<uint16_t*>(y_ptr),
                reinterpret_cast<__nv_bfloat16*>(out_raw_ptr),
                reinterpret_cast<float*>(lse_out_ptr),
                sq_t, sq_h, sr_t, sr_h, sk_b, sk_s, sk_h, sv_b, sv_s, sv_h,
                sbt_r, sg_t, ssk_t, sy_t, so_t, so_h, sl_t, sl_h,
                H, G, BS, window, E, 128, 64, scale_log2, rel_threshold, clip_eps);
        }
    }
    return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

extern "C" int local_swa_otf_rel_bf16_sm121_fwd(
    const void* q_ptr, const void* r_ptr, const void* r_scale_ptr, const void* rel_proj_t_ptr,
    const void* k_ptr, const void* v_ptr,
    const void* bt_ptr, const void* seq_ptr, const void* cuq_ptr,
    const void* gate_ptr, const void* sink_ptr, void* y_ptr,
    void* out_raw_ptr, void* lse_out_ptr,
    int64_t sq_t, int64_t sq_h, int64_t sr_t, int64_t sr_h,
    int64_t sk_b, int64_t sk_s, int64_t sk_h,
    int64_t sv_b, int64_t sv_s, int64_t sv_h,
    int64_t sbt_r, int64_t sg_t, int64_t ssk_t, int64_t sy_t,
    int64_t so_t, int64_t so_h, int64_t sl_t, int64_t sl_h,
    int num_reqs, int max_seqlen_q, int H, int G,
    int BS, int window, int E,
    float scale_log2, float rel_threshold, float clip_eps,
    uint64_t stream_u64
) {
    return local_swa_fp8_otf_rel_v6_sm121_fwd(
        q_ptr, r_ptr, r_scale_ptr, rel_proj_t_ptr, k_ptr, v_ptr,
        bt_ptr, seq_ptr, cuq_ptr, gate_ptr, sink_ptr, y_ptr, out_raw_ptr, lse_out_ptr,
        sq_t, sq_h, sr_t, sr_h, sk_b, sk_s, sk_h, sv_b, sv_s, sv_h,
        sbt_r, sg_t, ssk_t, sy_t, so_t, so_h, sl_t, sl_h,
        num_reqs, max_seqlen_q, H, G, BS, window, E, 0,
        scale_log2, rel_threshold, clip_eps, stream_u64);
}

extern "C" int global_tail_fp8_otf_rel_v6_sm121_fwd(
    const void* q_ptr, const void* r_ptr, const void* r_scale_ptr, const void* tau_ptr,
    const void* rel_proj_t_ptr,
    const void* k_ptr, const void* v_ptr,
    const void* bt_ptr, const void* seq_ptr, const void* cuq_ptr,
    const void* acc_in_ptr, const void* m_in_ptr, const void* l_in_ptr,
    const void* gate_ptr, const void* sink_ptr, void* y_ptr,
    void* out_raw_ptr, void* lse_out_ptr,
    int64_t sq_t, int64_t sq_h, int64_t sr_t, int64_t sr_h,
    int64_t sk_b, int64_t sk_s, int64_t sk_h,
    int64_t sv_b, int64_t sv_s, int64_t sv_h,
    int64_t sbt_r, int64_t sg_t, int64_t ssk_t, int64_t sy_t,
    int64_t so_t, int64_t so_h, int64_t sl_t, int64_t sl_h,
    int num_reqs, int max_seqlen_q, int H, int G,
    int BS, int E, int prefix_bm, int prefix_bn, int tail_bn, int pv_f16_acc,
    float scale_log2, float rel_threshold, float clip_eps,
    uint64_t stream_u64
) {
    (void)tail_bn;  // v6 always runs BN=64 natively inside 42.5 KiB SMEM and masks k_start_exact!
    if (max_seqlen_q < 1024) return -3;
    if (num_reqs <= 0 || H <= 0 || G <= 0 || (H % G) != 0 || BS < 64 || (BS & (BS - 1)) != 0) return -1;
    if (acc_in_ptr != nullptr && (prefix_bm < 64 || (prefix_bm & 63) != 0 || prefix_bn <= 0)) return -1;
    if ((reinterpret_cast<uintptr_t>(q_ptr) & 7u) != 0 ||
        (reinterpret_cast<uintptr_t>(r_ptr) & 3u) != 0 ||
        (reinterpret_cast<uintptr_t>(r_scale_ptr) & 3u) != 0 ||
        (reinterpret_cast<uintptr_t>(rel_proj_t_ptr) & 3u) != 0 ||
        (reinterpret_cast<uintptr_t>(k_ptr) & 15u) != 0 ||
        (reinterpret_cast<uintptr_t>(v_ptr) & 15u) != 0) {
        return -1;
    }
    if ((sq_t & 3) != 0 || (sq_h & 3) != 0 || (sr_t & 1) != 0 || (sr_h & 1) != 0 ||
        (sk_b & 7) != 0 || (sk_s & 7) != 0 || (sk_h & 7) != 0 ||
        (sv_b & 7) != 0 || (sv_s & 7) != 0 || (sv_h & 7) != 0) {
        return -1;
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_u64);
    if (is_stream_capturing(stream)) return -4;

    const bool has_tau = (tau_ptr != nullptr);
    dim3 block(THREADS, 1, 1);

    #define LAUNCH_GTAIL_V6(HPC, HTAU, F16PV) \
        otf_rel_fp8_v6_sm121_kernel<HPC, true, HTAU, F16PV><<<grid, block, 0, stream>>>( \
            reinterpret_cast<const __nv_bfloat16*>(q_ptr), \
            reinterpret_cast<const uint16_t*>(r_ptr), \
            reinterpret_cast<const uint16_t*>(r_scale_ptr), \
            reinterpret_cast<const float*>(tau_ptr), \
            reinterpret_cast<const uint16_t*>(rel_proj_t_ptr), \
            reinterpret_cast<const __nv_bfloat16*>(k_ptr), \
            reinterpret_cast<const __nv_bfloat16*>(v_ptr), \
            reinterpret_cast<const int32_t*>(bt_ptr), \
            reinterpret_cast<const int32_t*>(seq_ptr), \
            reinterpret_cast<const int32_t*>(cuq_ptr), \
            reinterpret_cast<const float*>(acc_in_ptr), \
            reinterpret_cast<const float*>(m_in_ptr), \
            reinterpret_cast<const float*>(l_in_ptr), \
            reinterpret_cast<const uint16_t*>(gate_ptr), \
            reinterpret_cast<const uint16_t*>(sink_ptr), \
            reinterpret_cast<uint16_t*>(y_ptr), \
            reinterpret_cast<__nv_bfloat16*>(out_raw_ptr), \
            reinterpret_cast<float*>(lse_out_ptr), \
            sq_t, sq_h, sr_t, sr_h, sk_b, sk_s, sk_h, sv_b, sv_s, sv_h, \
            sbt_r, sg_t, ssk_t, sy_t, so_t, so_h, sl_t, sl_h, \
            H, G, BS, -1, E, prefix_bm, prefix_bn, scale_log2, rel_threshold, clip_eps)

    if ((G & 1) == 0 && (acc_in_ptr == nullptr || (prefix_bm >= 64 && (prefix_bm % 64) == 0))) {
        const int num_mt = (max_seqlen_q + 63) / 64;
        dim3 grid(num_mt * (H / 2), num_reqs, 1);
        if (has_tau) {
            if (pv_f16_acc != 0) { LAUNCH_GTAIL_V6(2, true, true); }
            else                 { LAUNCH_GTAIL_V6(2, true, false); }
        } else {
            if (pv_f16_acc != 0) { LAUNCH_GTAIL_V6(2, false, true); }
            else                 { LAUNCH_GTAIL_V6(2, false, false); }
        }
    } else {
        const int num_mt = (max_seqlen_q + 127) / 128;
        dim3 grid(num_mt * H, num_reqs, 1);
        if (has_tau) {
            if (pv_f16_acc != 0) { LAUNCH_GTAIL_V6(1, true, true); }
            else                 { LAUNCH_GTAIL_V6(1, true, false); }
        } else {
            if (pv_f16_acc != 0) { LAUNCH_GTAIL_V6(1, false, true); }
            else                 { LAUNCH_GTAIL_V6(1, false, false); }
        }
    }
    #undef LAUNCH_GTAIL_V6

    return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

extern "C" int global_tail_otf_rel_bf16_sm121_fwd(
    const void* q_ptr, const void* r_ptr, const void* r_scale_ptr, const void* tau_ptr,
    const void* rel_proj_t_ptr,
    const void* k_ptr, const void* v_ptr,
    const void* bt_ptr, const void* seq_ptr, const void* cuq_ptr,
    const void* acc_in_ptr, const void* m_in_ptr, const void* l_in_ptr,
    const void* gate_ptr, const void* sink_ptr, void* y_ptr,
    void* out_raw_ptr, void* lse_out_ptr,
    int64_t sq_t, int64_t sq_h, int64_t sr_t, int64_t sr_h,
    int64_t sk_b, int64_t sk_s, int64_t sk_h,
    int64_t sv_b, int64_t sv_s, int64_t sv_h,
    int64_t sbt_r, int64_t sg_t, int64_t ssk_t, int64_t sy_t,
    int64_t so_t, int64_t so_h, int64_t sl_t, int64_t sl_h,
    int num_reqs, int max_seqlen_q, int H, int G,
    int BS, int E, int prefix_bm, int prefix_bn, int tail_bn,
    float scale_log2, float rel_threshold, float clip_eps,
    uint64_t stream_u64
) {
    return global_tail_fp8_otf_rel_v6_sm121_fwd(
        q_ptr, r_ptr, r_scale_ptr, tau_ptr, rel_proj_t_ptr, k_ptr, v_ptr,
        bt_ptr, seq_ptr, cuq_ptr, acc_in_ptr, m_in_ptr, l_in_ptr,
        gate_ptr, sink_ptr, y_ptr, out_raw_ptr, lse_out_ptr,
        sq_t, sq_h, sr_t, sr_h, sk_b, sk_s, sk_h, sv_b, sv_s, sv_h,
        sbt_r, sg_t, ssk_t, sy_t, so_t, so_h, sl_t, sl_h,
        num_reqs, max_seqlen_q, H, G, BS, E, prefix_bm, prefix_bn, tail_bn, 0,
        scale_log2, rel_threshold, clip_eps, stream_u64);
}

extern "C" int pf_fp8_attn_v13_version() {
    return 1300;  // Wave-V13 Prefill Attention (Sage2 v6 + Local SWA FP8 + Global Tail BN=64, f32 PV safe default)
}
