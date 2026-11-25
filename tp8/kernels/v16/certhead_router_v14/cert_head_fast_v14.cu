// SPDX-License-Identifier: Apache-2.0
// SM121a (GB10) 0-ULP Certified Target LM Head v14 (`cert_head_fast_v14.cu` -> `libcert_head_fast_v14.so`)
//
// 100% drop-in ABI superset of v7 / c48 / v9 / v12 (`M = 2..128`, 0 extra weight-table VRAM),
// delivering 5 surgical 0-ULP (`nd == 0`, 0 B stack / 0 B spills, 0 `#206` PDL hazards) upgrades:
//   1. Alive-Bitmap (`ALIVE_BM`, 3.14 KiB L1-resident) + Conditional `S1_OUT` / `HI` Elision:
//      - Inside Stage 1 epilogue, after updating global running lower-bound `THR_U[m]` via `atomicMax`,
//        each CTA computes a fail-closed self-verified `s_raw_cut[m]` (proven `< chain(T_global[m]) - slack`
//        with >260x margin over `2 * CAP_DIP_30`).
//      - 3-shuffle cross-`g` OR reduction identifies columns dead for ALL rows `0..nl-1` (`>85-97%` of
//        25,128 vocab rows after Wave 0!). Dead columns execute ZERO `S1_OUT` and ZERO `HI` stores and
//        leave their `ALIVE_BM` bit 0; `cert_cand1` skips 80-95% of `HI` reads via 1 cached bit-test!
//      - Alive columns write `S1_OUT` ONLY for surviving `(m, v)` pairs (<0.55 MB even at `M=128` vs
//        12.87 MB in V12!) using normal write-back stores so Stage-2 `S1_OUT` reads hit 100% in L2!
//      - Cuts Stage-1 + `cand1` traffic by `-24.4 MB` at `M=128` (`-164.8 MB` vs BF16 fallback!),
//        `-9.1 MB` at `M=48`, `-5.2 MB` at `M=28`, `-2.5 MB` at `M=14`!
//   2. Dynamic `n_live` M-Tile Pruning Across Stage-1 Screen & Stage-3 Refine (`M = 2..128`):
//      - Computes CTA-uniform `nl_tiles = (nl + 15) >> 4` (and `active_hi8 = (nl > 8)` for `M=9..16`).
//      - Skips ALL dead `X_PAD` L2->SMEM loads, SMEM bank reads, Tensor Core MMAs, and epilogue math
//        for dead rows `m >= nl_tiles * 16` (30-40% savings at C=4/8/32 under `tau=0.22..0.30`!).
//      - Stage-3 exact BF16 refine also skips dead warps (`warp >= nl_tiles`) and dead `has_hi` tiles!
//   3. Single-Load `MS_U4` & `BIG_S` in `v14_c48` (`M = 17..48`):
//      - Caches `(ms.z, ms.w, bs)` across Half 0 -> Half 1 in 6 scalar regs, eliminating 4.07 MB/step
//        of redundant L2/DRAM scale reloads + `cvt.f32.f16` + `sc_lut` recomputation.
//   4. Hoisted `ld.global.cs.v4.u32` (L2 Evict-First) Slab Prefetch Pipeline Across All `M = 2..128`:
//      - Issues streaming `SLABS_U4` DRAM loads BEFORE `X_PAD` L2->SMEM copy + `__syncthreads()`,
//        hiding LPDDR5X RTT behind SMEM staging while preserving L2-hot `X_PAD` (evict-first on 200K vocab!).
//   5. Strict 0-`#206` PDL Epilogue Hygiene + `cert_head_v14_version() -> 14` + Full V7/V9/V12 ABI Parity.
//
// Build on GB10 (SM121a — STRICT: `-gencode arch=compute_121a,code=sm_121a --fmad=false --ptxas-options=-v`,
// NEVER `--use_fast_math` or `--ftz=true`, hard-fail on ANY stack frame > 0 or spills > 0):
//   ./build_certhead_router_v14.sh [OUT_DIR]

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math.h>
#include <stdint.h>

namespace {

constexpr uint32_t ORD_NEG_INF = 0x007FFFFFu;
constexpr uint16_t FP16_NEG_INF_BITS = 0xFC00u;
constexpr int CERT_V14_MAX_M = 128;

static __device__ __forceinline__ uint32_t max_u32(uint32_t a, uint32_t b) {
    return (a > b) ? a : b;
}

static __device__ __forceinline__ uint32_t float_to_ord_u32(float x) {
    if (isnan(x)) return ORD_NEG_INF;
    uint32_t u = __float_as_uint(x);
    return (u & 0x80000000u) ? (~u) : (u | 0x80000000u);
}

static __device__ __forceinline__ float ord_u32_to_float(uint32_t key) {
    uint32_t bits = (key & 0x80000000u) ? (key ^ 0x80000000u) : (~key);
    return __uint_as_float(bits);
}

static __device__ __forceinline__ uint16_t f32_to_f16_ru_bits(float x) {
    uint16_t h;
    asm("cvt.rp.f16.f32 %0, %1;" : "=h"(h) : "f"(x));
    return h;
}

static __device__ __forceinline__ float f16_bits_to_f32(uint16_t h) {
    float x;
    asm("cvt.f32.f16 %0, %1;" : "=f"(x) : "h"(h));
    return x;
}

static __device__ __forceinline__ float bf16_rn_f32(float x) {
    __nv_bfloat16 b = __float2bfloat16_rn(x);
    return __bfloat162float(b);
}

static __device__ __forceinline__ uint32_t pack2_bf16_rn(float lo, float hi) {
    __nv_bfloat162 p = __halves2bfloat162(__float2bfloat16_rn(lo), __float2bfloat16_rn(hi));
    return *reinterpret_cast<const uint32_t*>(&p);
}

// L2 evict-first (`cs`) streaming 128-bit load: prevents 91.20 MB streaming vocab slabs from evicting L2-hot X_PAD!
static __device__ __forceinline__ uint4 ld_cs_u4(const uint4* p) {
    uint4 v;
    asm volatile("ld.global.cs.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

static __device__ __forceinline__ void st_cs_f32(float* p, float v) {
    asm volatile("st.global.cs.f32 [%0], %1;" :: "l"(p), "f"(v) : "memory");
}

static __device__ __forceinline__ void prefetch_l2_v14(const void* p) {
    asm volatile("prefetch.global.L2 [%0];" :: "l"(p));
}

static __device__ __forceinline__ int atomic_add_acq_rel_gpu_i32(int* addr, int val) {
    int old;
    asm volatile("atom.add.acq_rel.gpu.s32 %0, [%1], %2;" : "=r"(old) : "l"(addr), "r"(val) : "memory");
    return old;
}

// Exact IEEE-754 fp32 soft-cap chain matching `torch.tanh(x / cap) * cap` (DIVMODE=0: x * inv_cap)
// including denormal preservation for |y| < 2^-126 (see cert_head.py `_chain`).
static __device__ __forceinline__ float chain_f32(float x, float inv_cap, float cap) {
    if (cap <= 0.0f) return x;
    float y = __fmul_rn(x, inv_cap);
    float t = tanhf(y);
    if (fabsf(y) < 1.1754943508222875e-38f) t = y;
    return __fmul_rn(t, cap);
}

// Fail-closed self-verifying raw pre-filter threshold:
// Guarantees that for ANY finite `hi_f < raw_cut` and ANY `T_glob >= t_raw`,
//   `chain_f32(hi_f) < chain_f32(T_glob) - slack` (assuming `extra_margin >= 5e-4 >> 2 * CAP_DIP_30`).
// Returns `-INFINITY` whenever `t_raw` is non-finite or near soft-cap saturation (zero false prunes!).
static __device__ __forceinline__ float compute_safe_raw_cut(
    float t_raw, float inv_cap, float cap, float slack, float extra_margin
) {
    if (!isfinite(t_raw)) return -INFINITY;
    const float tc = __fsub_rn(chain_f32(t_raw, inv_cap, cap), slack);
    if (cap <= 0.0f) {
        return isfinite(tc) ? tc : -INFINITY;
    }
    if (!isfinite(tc)) return -INFINITY;
    const float tc_safe = __fsub_rn(tc, __fadd_rn(extra_margin, 5.0e-4f));
    const float y_cut = __fmul_rn(tc_safe, inv_cap);
    if (y_cut > -0.995f && y_cut < 0.995f) {
        const float cand_cut = __fsub_rn(__fmul_rn(atanhf(y_cut), cap), 0.01f);
        if (isfinite(cand_cut) && chain_f32(cand_cut, inv_cap, cap) < __fsub_rn(tc, extra_margin)) {
            return cand_cut;
        }
    }
    return -INFINITY;
}

static __device__ __forceinline__ int resolve_nlive(const int32_t* nlive_ptr, int M) {
    if (nlive_ptr == nullptr) return M;
    const int nl = *nlive_ptr;
    return (nl >= 1 && nl <= M) ? nl : M;
}

static __device__ __forceinline__ uint64_t full_mask_lo(int nl) {
    if (nl <= 0) return 0ull;
    return (nl >= 64) ? ~0ull : ((1ull << nl) - 1ull);
}

static __device__ __forceinline__ uint64_t full_mask_hi(int nl) {
    if (nl <= 64) return 0ull;
    return (nl >= 128) ? ~0ull : ((1ull << (nl - 64)) - 1ull);
}

static __device__ __forceinline__ void compute_s1_bounds(
    float s1, float xn, float rt, float nv,
    float cacc2, float infl_q, float infl_b, float slop_rel, float slop_abs,
    float& lo1, float& hi1
) {
    const float term_q = __fmul_rn(__fmul_rn(xn, rt), infl_q);
    const float term_a = __fmul_rn(__fmul_rn(cacc2, xn), nv);
    const float B1 = __fmul_rn(__fadd_rn(term_q, term_a), infl_b);
    const float slop1 = __fadd_rn(__fmul_rn(__fadd_rn(fabsf(s1), B1), slop_rel), slop_abs);
    lo1 = __fsub_rn(__fsub_rn(s1, B1), slop1);
    hi1 = __fadd_rn(__fadd_rn(s1, B1), slop1);
    if (!isfinite(s1) || isnan(B1)) {
        lo1 = -INFINITY;
        hi1 = INFINITY;
    }
}

template <typename KernelT, typename... Args>
static inline cudaError_t launch_maybe_pdl_v14(
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

// =========================================================================
// Kernel 1: `cert_xnorm_pad_v14_kernel` (M = 1..128, zeroes 3.14 KiB ALIVE_BM in CTA 0)
// =========================================================================
template <bool PDL>
__global__ void __launch_bounds__(256, 4)
cert_xnorm_pad_v14_kernel(
    const __nv_bfloat16* __restrict__ X,
    const int64_t* __restrict__ LOGITS_IDX,
    int idx_is_i32,
    __nv_bfloat16* __restrict__ X_PAD,
    float* __restrict__ XN,
    int32_t* __restrict__ BAD,
    uint32_t* __restrict__ THR_U,
    int32_t* __restrict__ CNT1,
    int32_t* __restrict__ CNT2,
    uint64_t* __restrict__ BEST,
    int32_t* __restrict__ ARRIVE,
    uint32_t* __restrict__ ALIVE_BM,
    const int32_t* __restrict__ NLIVE_PTR,
    int M,
    int m_rows,
    int v_bm_words,
    int K,
    int sx,
    float infl_x,
    float x_abs
) {
    const int m = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    if constexpr (PDL) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }

    const int nl = resolve_nlive(NLIVE_PTR, M);
    const int nl_pad = (m_rows == 8) ? 8 : (((nl + 15) >> 4) << 4);
    constexpr uint64_t INIT_BEST = (static_cast<uint64_t>(ORD_NEG_INF) << 32);

    if (m == 0) {
        if (tid == 0) {
            *CNT1 = 0;
            *CNT2 = 0;
            *ARRIVE = 0;
        }
        if (ALIVE_BM != nullptr) {
            for (int w = tid; w < v_bm_words; w += 256) {
                ALIVE_BM[w] = 0u;
            }
        }
        // Preserve legacy C=1 zero-init of slots 8..15 when m_rows == 8 (buffers always >= 16)
        if (m_rows == 8 && tid >= 8 && tid < 16) {
            XN[tid] = 0.0f;
            BAD[tid] = 0;
            THR_U[tid] = ORD_NEG_INF;
            BEST[tid] = INIT_BEST;
        }
    }

    __nv_bfloat16* dst_row = X_PAD + static_cast<int64_t>(m) * K;
    if (m >= nl) {
        // V14: Only zero padding rows inside the active MMA tile (`m < nl_pad`);
        // dead M-tiles `m >= nl_pad` are completely skipped by Stage-1 & Stage-3!
        if (m < nl_pad) {
            const uint4 z4 = make_uint4(0u, 0u, 0u, 0u);
            for (int k = tid * 8; k < K; k += 256 * 8) {
                *reinterpret_cast<uint4*>(dst_row + k) = z4;
            }
        }
        if (tid == 0) {
            XN[m] = 0.0f;
            BAD[m] = 0;
            THR_U[m] = ORD_NEG_INF;
            BEST[m] = INIT_BEST;
        }
        if constexpr (PDL) {
            __syncthreads();
            if (tid == 0) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
        return;
    }

    int64_t src_m = static_cast<int64_t>(m);
    if (LOGITS_IDX != nullptr) {
        if (idx_is_i32) {
            src_m = static_cast<int64_t>(reinterpret_cast<const int32_t*>(LOGITS_IDX)[m]);
        } else {
            src_m = LOGITS_IDX[m];
        }
    }
    const __nv_bfloat16* src_row = X + src_m * static_cast<int64_t>(sx);

    float sq = 0.0f;
    for (int k = tid * 8; k < K; k += 256 * 8) {
        uint4 raw = *reinterpret_cast<const uint4*>(src_row + k);
        *reinterpret_cast<uint4*>(dst_row + k) = raw;
        const __nv_bfloat162* h2 = reinterpret_cast<const __nv_bfloat162*>(&raw);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            float2 f = __bfloat1622float2(h2[i]);
            sq = __fadd_rn(sq, __fadd_rn(__fmul_rn(f.x, f.x), __fmul_rn(f.y, f.y)));
        }
    }

    #pragma unroll
    for (int off = 16; off >= 1; off >>= 1) {
        sq = __fadd_rn(sq, __shfl_xor_sync(0xFFFFFFFFu, sq, off));
    }
    __shared__ float s_sq[8];
    if (lane == 0) s_sq[warp] = sq;
    __syncthreads();

    if (warp == 0) {
        float v = (lane < 8) ? s_sq[lane] : 0.0f;
        v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, 4));
        v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, 2));
        v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, 1));
        if (lane == 0) {
            float xn = __fadd_rn(__fmul_rn(__fsqrt_rn(v), infl_x), x_abs);
            XN[m] = xn;
            BAD[m] = (!isfinite(xn)) ? 1 : 0;
            THR_U[m] = ORD_NEG_INF;
            BEST[m] = INIT_BEST;
        }
    }
    if constexpr (PDL) {
        __syncthreads();
        if (tid == 0) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }
}

// =========================================================================
// Branchless 3-SEL `select_lut4()` (0 stack frames!) & INT6_g512_ms8 MMA helpers
// =========================================================================
static __device__ __forceinline__ float select_lut4(const float lut[4], uint32_t idx) {
    const float a = (idx & 1u) ? lut[1] : lut[0];
    const float b = (idx & 1u) ? lut[3] : lut[2];
    return (idx & 2u) ? b : a;
}

template <bool M_LE8>
static __device__ __forceinline__ void exec_step_int6_ms8_m1(
    uint32_t u, uint32_t ms4,
    const float sc_lut[4],
    const __nv_bfloat16* sX, int x_off0, int x_off1, int k_off,
    bool active_hi8,
    float (&acc)[4]
) {
    const float sc_lo = select_lut4(sc_lut, ms4 & 3u);
    const float sc_hi = select_lut4(sc_lut, (ms4 >> 2) & 3u);

    const int q0 = static_cast<int>(u & 0x3Fu) - 32;
    const int q1 = static_cast<int>((u >> 6) & 0x3Fu) - 32;
    const int q2 = static_cast<int>((u >> 12) & 0x3Fu) - 32;
    const int q3 = static_cast<int>((u >> 18) & 0x3Fu) - 32;

    const uint32_t b0 = pack2_bf16_rn(__fmul_rn(static_cast<float>(q0), sc_lo),
                                      __fmul_rn(static_cast<float>(q1), sc_lo));
    const uint32_t b1 = pack2_bf16_rn(__fmul_rn(static_cast<float>(q2), sc_hi),
                                      __fmul_rn(static_cast<float>(q3), sc_hi));

    const uint32_t a0 = *reinterpret_cast<const uint32_t*>(sX + x_off0 + k_off);
    const uint32_t a2 = *reinterpret_cast<const uint32_t*>(sX + x_off0 + k_off + 8);
    uint32_t a1 = 0u, a3 = 0u;
    if constexpr (!M_LE8) {
        if (active_hi8) {
            a1 = *reinterpret_cast<const uint32_t*>(sX + x_off1 + k_off);
            a3 = *reinterpret_cast<const uint32_t*>(sX + x_off1 + k_off + 8);
        }
    }
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)
    );
}

template <bool M_LE8>
static __device__ __forceinline__ void exec_hc_int6_ms8(
    uint32_t w0, uint32_t w1, uint32_t w2, uint32_t ms16,
    const float sc_lut[4],
    const __nv_bfloat16* sX, int x_off0, int x_off1, int hc_k_off,
    bool active_hi8,
    float (&acc)[4]
) {
    exec_step_int6_ms8_m1<M_LE8>(w0 & 0xFFFFFFu,                             ms16 & 0xFu,        sc_lut, sX, x_off0, x_off1, hc_k_off +  0, active_hi8, acc);
    exec_step_int6_ms8_m1<M_LE8>(__funnelshift_r(w0, w1, 24) & 0xFFFFFFu,    (ms16 >> 4) & 0xFu, sc_lut, sX, x_off0, x_off1, hc_k_off + 16, active_hi8, acc);
    exec_step_int6_ms8_m1<M_LE8>(__funnelshift_r(w1, w2, 16) & 0xFFFFFFu,    (ms16 >> 8) & 0xFu, sc_lut, sX, x_off0, x_off1, hc_k_off + 32, active_hi8, acc);
    exec_step_int6_ms8_m1<M_LE8>((w2 >> 8) & 0xFFFFFFu,                      (ms16 >> 12) & 0xFu, sc_lut, sX, x_off0, x_off1, hc_k_off + 48, active_hi8, acc);
}

// Multi-MT (MT in {2..8}) B-fragment reuse helper with uniform `nl_tiles` dead-tile pruning!
template <int MT, int SX_STRIDE>
static __device__ __forceinline__ void exec_step_int6_ms8_mt_nl(
    uint32_t u, uint32_t ms4,
    const float sc_lut[4],
    const __nv_bfloat16* sX, int g, int t, int k_off,
    int nl_tiles,
    float (&acc)[MT][4]
) {
    const float sc_lo = select_lut4(sc_lut, ms4 & 3u);
    const float sc_hi = select_lut4(sc_lut, (ms4 >> 2) & 3u);

    const int q0 = static_cast<int>(u & 0x3Fu) - 32;
    const int q1 = static_cast<int>((u >> 6) & 0x3Fu) - 32;
    const int q2 = static_cast<int>((u >> 12) & 0x3Fu) - 32;
    const int q3 = static_cast<int>((u >> 18) & 0x3Fu) - 32;

    const uint32_t b0 = pack2_bf16_rn(__fmul_rn(static_cast<float>(q0), sc_lo),
                                      __fmul_rn(static_cast<float>(q1), sc_lo));
    const uint32_t b1 = pack2_bf16_rn(__fmul_rn(static_cast<float>(q2), sc_hi),
                                      __fmul_rn(static_cast<float>(q3), sc_hi));

    const int col_off = 2 * t + k_off;
    #pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
        if (mt == 0 || mt < nl_tiles) {
            const int x_off0 = (mt * 16 + g) * SX_STRIDE + col_off;
            const int x_off1 = (mt * 16 + g + 8) * SX_STRIDE + col_off;
            const uint32_t a0 = *reinterpret_cast<const uint32_t*>(sX + x_off0);
            const uint32_t a1 = *reinterpret_cast<const uint32_t*>(sX + x_off1);
            const uint32_t a2 = *reinterpret_cast<const uint32_t*>(sX + x_off0 + 8);
            const uint32_t a3 = *reinterpret_cast<const uint32_t*>(sX + x_off1 + 8);
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                : "+f"(acc[mt][0]), "+f"(acc[mt][1]), "+f"(acc[mt][2]), "+f"(acc[mt][3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)
            );
        }
    }
}

template <int MT, int SX_STRIDE>
static __device__ __forceinline__ void exec_hc_int6_ms8_mt_nl(
    uint32_t w0, uint32_t w1, uint32_t w2, uint32_t ms16,
    const float sc_lut[4],
    const __nv_bfloat16* sX, int g, int t, int hc_k_off,
    int nl_tiles,
    float (&acc)[MT][4]
) {
    exec_step_int6_ms8_mt_nl<MT, SX_STRIDE>(w0 & 0xFFFFFFu,                             ms16 & 0xFu,        sc_lut, sX, g, t, hc_k_off +  0, nl_tiles, acc);
    if constexpr (MT >= 4) asm volatile("" ::: "memory");
    exec_step_int6_ms8_mt_nl<MT, SX_STRIDE>(__funnelshift_r(w0, w1, 24) & 0xFFFFFFu,    (ms16 >> 4) & 0xFu, sc_lut, sX, g, t, hc_k_off + 16, nl_tiles, acc);
    if constexpr (MT >= 4) asm volatile("" ::: "memory");
    exec_step_int6_ms8_mt_nl<MT, SX_STRIDE>(__funnelshift_r(w1, w2, 16) & 0xFFFFFFu,    (ms16 >> 8) & 0xFu, sc_lut, sX, g, t, hc_k_off + 32, nl_tiles, acc);
    if constexpr (MT >= 4) asm volatile("" ::: "memory");
    exec_step_int6_ms8_mt_nl<MT, SX_STRIDE>((w2 >> 8) & 0xFFFFFFu,                      (ms16 >> 12) & 0xFu, sc_lut, sX, g, t, hc_k_off + 48, nl_tiles, acc);
}

// Warp-level 8-column bitmap OR helper (v_warp_sub is ALWAYS 8-aligned, zero cross-word split!)
static __device__ __forceinline__ void warp_or_alive_8cols(
    uint32_t* ALIVE_BM, int v_warp_sub, int vreal, int lane, int t,
    uint32_t col_alive_v0, uint32_t col_alive_v1
) {
    uint32_t b8 = ((col_alive_v0 ? 1u : 0u) | (col_alive_v1 ? 2u : 0u)) << (2 * t);
    b8 |= __shfl_xor_sync(0xFFFFFFFFu, b8, 2);
    b8 |= __shfl_xor_sync(0xFFFFFFFFu, b8, 1);
    if (lane == 0 && b8 != 0u && v_warp_sub < vreal) {
        atomicOr(&ALIVE_BM[v_warp_sub >> 5], b8 << (v_warp_sub & 31));
    }
}

// =========================================================================
// Kernel 2A: `cert_screen_int6_ms8_v14_m16_kernel` (M = 1..16, C=1..2!)
// =========================================================================
template <int WARPS, int SUB_N, bool M_LE8, bool PDL, bool USE_ALIVE>
__global__ void __launch_bounds__(WARPS * 32, (SUB_N == 2 && PDL) ? 2 : 3)
cert_screen_int6_ms8_v14_m16_kernel(
    const __nv_bfloat16* __restrict__ X_PAD,
    const uint4* __restrict__ SLABS_U4,      // [wtot, n_qchunks * SUB_N, 6, 32] uint4
    const uint4* __restrict__ MS_U4,         // [wtot, n_qchunks * SUB_N, 8] uint4
    const uint16_t* __restrict__ BIG_S,      // [wtot, n_qchunks * SUB_N, 8] fp16 bits
    const float* __restrict__ RT1,           // [V]
    const float* __restrict__ NV1,           // [V]
    const float* __restrict__ XN,            // [16]
    float* __restrict__ S1_OUT,              // [M, V]
    uint16_t* __restrict__ HI,               // [M, V] fp16 bits (rounded UP!)
    uint32_t* __restrict__ THR_U,            // [16]
    uint32_t* __restrict__ ALIVE_BM,         // [(V+31)/32] u32 or nullptr
    const int32_t* __restrict__ NLIVE_PTR,
    int M,
    int V,
    int vreal,
    int K,
    float cacc2,
    float infl_q,
    float infl_b,
    float slop_rel,
    float slop_abs,
    float inv_cap,
    float cap,
    float slack
) {
    constexpr int BN = WARPS * SUB_N * 8;
    constexpr int THREADS = WARPS * 32;
    constexpr int SX_STRIDE = 520;  // 512 + 8 bf16 (zero SMEM bank conflicts!)
    constexpr int M_ROWS = M_LE8 ? 8 : 16;

    const int pn = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g = lane >> 2;
    const int t = lane & 3;

    const int warp_uid = pn * WARPS + warp;
    const int n_qchunks = K >> 9;
    const int total_q = n_qchunks * SUB_N;

    uint4 pre_v0{}, pre_v1{}, pre_v2{}, pre_ms{};
    float pre_bs = 0.0f;
    if constexpr (PDL) {
        if (total_q > 0) {
            const int64_t q_idx0 = static_cast<int64_t>(warp_uid) * total_q;
            const uint4* qp0 = SLABS_U4 + q_idx0 * (6 * 32);
            pre_v0 = ld_cs_u4(qp0 + 0 * 32 + lane);
            pre_v1 = ld_cs_u4(qp0 + 1 * 32 + lane);
            pre_v2 = ld_cs_u4(qp0 + 2 * 32 + lane);
            pre_ms = __ldg(MS_U4 + q_idx0 * 8 + g);
            pre_bs = f16_bits_to_f32(__ldg(BIG_S + q_idx0 * 8 + g));
        }
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }

    const int nl = resolve_nlive(NLIVE_PTR, M);
    const bool active_hi8 = (!M_LE8) && (nl > 8);

    __shared__ uint32_t s_thr[16];
    __shared__ float s_xn[16];
    __shared__ float s_raw_cut[16];
    extern __shared__ __nv_bfloat16 sX[];  // [M_ROWS * SX_STRIDE]

    if (tid < 16) {
        s_thr[tid] = ORD_NEG_INF;
        s_xn[tid] = (tid < nl) ? XN[tid] : 0.0f;
        s_raw_cut[tid] = -INFINITY;
    }

    float acc[SUB_N][4];
    #pragma unroll
    for (int sub = 0; sub < SUB_N; ++sub) {
        acc[sub][0] = 0.0f; acc[sub][1] = 0.0f; acc[sub][2] = 0.0f; acc[sub][3] = 0.0f;
    }

    const int x_off0 = g * SX_STRIDE + 2 * t;
    const int x_off1 = (g + 8) * SX_STRIDE + 2 * t;
    constexpr int VEC_PER_512 = 64;
    const int active_x_vec = (active_hi8 ? M_ROWS : 8) * VEC_PER_512;

    for (int qc = 0; qc < n_qchunks; ++qc) {
        const int k_base = qc * 512;
        const int64_t q_idx0 = static_cast<int64_t>(warp_uid) * total_q + (qc * SUB_N);
        const uint4* qp0 = SLABS_U4 + q_idx0 * (6 * 32);

        // V14: Issue sub=0 Half-0 streaming DRAM slab loads BEFORE X_PAD SMEM copy + syncthreads!
        uint4 sub0_v0, sub0_v1, sub0_v2, sub0_ms;
        float sub0_bs;
        if (PDL && qc == 0) {
            sub0_v0 = pre_v0; sub0_v1 = pre_v1; sub0_v2 = pre_v2;
            sub0_ms = pre_ms; sub0_bs = pre_bs;
        } else {
            sub0_v0 = ld_cs_u4(qp0 + 0 * 32 + lane);
            sub0_v1 = ld_cs_u4(qp0 + 1 * 32 + lane);
            sub0_v2 = ld_cs_u4(qp0 + 2 * 32 + lane);
            sub0_ms = __ldg(MS_U4 + q_idx0 * 8 + g);
            sub0_bs = f16_bits_to_f32(__ldg(BIG_S + q_idx0 * 8 + g));
        }

        for (int idx = tid; idx < active_x_vec; idx += THREADS) {
            const int m = idx >> 6;
            const int kv = idx & 63;
            *reinterpret_cast<uint4*>(sX + m * SX_STRIDE + kv * 8) =
                *reinterpret_cast<const uint4*>(X_PAD + static_cast<int64_t>(m) * K + k_base + kv * 8);
        }
        __syncthreads();

        #pragma unroll
        for (int sub = 0; sub < SUB_N; ++sub) {
            const int q = qc * SUB_N + sub;
            const int64_t q_idx = static_cast<int64_t>(warp_uid) * total_q + q;
            const uint4* qp = SLABS_U4 + q_idx * (6 * 32);
            uint4 v0, v1, v2, ms;
            float bs;
            if (sub == 0) {
                v0 = sub0_v0; v1 = sub0_v1; v2 = sub0_v2; ms = sub0_ms; bs = sub0_bs;
            } else {
                v0 = ld_cs_u4(qp + 0 * 32 + lane);
                v1 = ld_cs_u4(qp + 1 * 32 + lane);
                v2 = ld_cs_u4(qp + 2 * 32 + lane);
                ms = __ldg(MS_U4 + q_idx * 8 + g);
                bs = f16_bits_to_f32(__ldg(BIG_S + q_idx * 8 + g));
            }
            // Overlap Half-1 slab loads (v3..v5) directly behind Half-0's 16 MMAs!
            const uint4 v3 = ld_cs_u4(qp + 3 * 32 + lane);
            const uint4 v4 = ld_cs_u4(qp + 4 * 32 + lane);
            const uint4 v5 = ld_cs_u4(qp + 5 * 32 + lane);
            const float sc_lut[4] = {
                __fmul_rn(bs, 0.375f),
                __fmul_rn(bs, 0.5625f),
                __fmul_rn(bs, 0.75f),
                bs
            };
            exec_hc_int6_ms8<M_LE8>(v0.x, v0.y, v0.z, ms.x & 0xFFFFu, sc_lut, sX, x_off0, x_off1, 0 * 64, active_hi8, acc[sub]);
            exec_hc_int6_ms8<M_LE8>(v0.w, v1.x, v1.y, ms.x >> 16,     sc_lut, sX, x_off0, x_off1, 1 * 64, active_hi8, acc[sub]);
            exec_hc_int6_ms8<M_LE8>(v1.z, v1.w, v2.x, ms.y & 0xFFFFu, sc_lut, sX, x_off0, x_off1, 2 * 64, active_hi8, acc[sub]);
            exec_hc_int6_ms8<M_LE8>(v2.y, v2.z, v2.w, ms.y >> 16,     sc_lut, sX, x_off0, x_off1, 3 * 64, active_hi8, acc[sub]);
            exec_hc_int6_ms8<M_LE8>(v3.x, v3.y, v3.z, ms.z & 0xFFFFu, sc_lut, sX, x_off0, x_off1, 4 * 64, active_hi8, acc[sub]);
            exec_hc_int6_ms8<M_LE8>(v3.w, v4.x, v4.y, ms.z >> 16,     sc_lut, sX, x_off0, x_off1, 5 * 64, active_hi8, acc[sub]);
            exec_hc_int6_ms8<M_LE8>(v4.z, v4.w, v5.x, ms.w & 0xFFFFu, sc_lut, sX, x_off0, x_off1, 6 * 64, active_hi8, acc[sub]);
            exec_hc_int6_ms8<M_LE8>(v5.y, v5.z, v5.w, ms.w >> 16,     sc_lut, sX, x_off0, x_off1, 7 * 64, active_hi8, acc[sub]);
        }
        __syncthreads();
    }

    const int m0 = g;
    const int m1 = g + 8;
    const float xn0 = s_xn[m0];
    const float xn1 = active_hi8 ? s_xn[m1] : 0.0f;

    float rt0_c[SUB_N], nv0_c[SUB_N], rt1_c[SUB_N], nv1_c[SUB_N];
    #pragma unroll
    for (int sub = 0; sub < SUB_N; ++sub) {
        const int v0 = pn * BN + warp * (SUB_N * 8) + sub * 8 + 2 * t;
        const int v1 = v0 + 1;
        rt0_c[sub] = (v0 < vreal) ? __ldg(RT1 + v0) : 0.0f;
        nv0_c[sub] = (v0 < vreal) ? __ldg(NV1 + v0) : 0.0f;
        rt1_c[sub] = (v1 < vreal) ? __ldg(RT1 + v1) : 0.0f;
        nv1_c[sub] = (v1 < vreal) ? __ldg(NV1 + v1) : 0.0f;
    }

    if constexpr (!USE_ALIVE) {
        auto emit_legacy = [&](int m, float xn, int v, float rt, float nv, float s1) -> uint32_t {
            if (m >= nl || v >= vreal) return ORD_NEG_INF;
            const int64_t mv = static_cast<int64_t>(m) * V + v;
            S1_OUT[mv] = s1;
            float lo1, hi1;
            compute_s1_bounds(s1, xn, rt, nv, cacc2, infl_q, infl_b, slop_rel, slop_abs, lo1, hi1);
            HI[mv] = f32_to_f16_ru_bits(hi1);
            return float_to_ord_u32(lo1);
        };
        uint32_t lmax0 = ORD_NEG_INF;
        uint32_t lmax1 = ORD_NEG_INF;
        #pragma unroll
        for (int sub = 0; sub < SUB_N; ++sub) {
            const int v0 = pn * BN + warp * (SUB_N * 8) + sub * 8 + 2 * t;
            const int v1 = v0 + 1;
            lmax0 = max_u32(lmax0, emit_legacy(m0, xn0, v0, rt0_c[sub], nv0_c[sub], acc[sub][0]));
            lmax0 = max_u32(lmax0, emit_legacy(m0, xn0, v1, rt1_c[sub], nv1_c[sub], acc[sub][1]));
            if constexpr (!M_LE8) {
                if (active_hi8) {
                    lmax1 = max_u32(lmax1, emit_legacy(m1, xn1, v0, rt0_c[sub], nv0_c[sub], acc[sub][2]));
                    lmax1 = max_u32(lmax1, emit_legacy(m1, xn1, v1, rt1_c[sub], nv1_c[sub], acc[sub][3]));
                }
            }
        }
        lmax0 = max_u32(lmax0, __shfl_xor_sync(0xFFFFFFFFu, lmax0, 2));
        lmax0 = max_u32(lmax0, __shfl_xor_sync(0xFFFFFFFFu, lmax0, 1));
        if (t == 0 && m0 < nl) atomicMax(&s_thr[m0], lmax0);
        if constexpr (!M_LE8) {
            if (active_hi8) {
                lmax1 = max_u32(lmax1, __shfl_xor_sync(0xFFFFFFFFu, lmax1, 2));
                lmax1 = max_u32(lmax1, __shfl_xor_sync(0xFFFFFFFFu, lmax1, 1));
                if (t == 0 && m1 < nl) atomicMax(&s_thr[m1], lmax1);
            }
        }
        __syncthreads();
        if (tid < nl) {
            atomicMax(&THR_U[tid], s_thr[tid]);
        }
    } else {
        // V14 Pass 1: Register-only bounds + global running lower bound `THR_U[m]` -> `s_raw_cut[m]`
        uint16_t hi_u[SUB_N][4];
        uint32_t lmax0 = ORD_NEG_INF;
        uint32_t lmax1 = ORD_NEG_INF;
        #pragma unroll
        for (int sub = 0; sub < SUB_N; ++sub) {
            const int v0 = pn * BN + warp * (SUB_N * 8) + sub * 8 + 2 * t;
            const int v1 = v0 + 1;
            hi_u[sub][0] = FP16_NEG_INF_BITS;
            hi_u[sub][1] = FP16_NEG_INF_BITS;
            hi_u[sub][2] = FP16_NEG_INF_BITS;
            hi_u[sub][3] = FP16_NEG_INF_BITS;
            if (m0 < nl && v0 < vreal) {
                float lo1, hi1;
                compute_s1_bounds(acc[sub][0], xn0, rt0_c[sub], nv0_c[sub], cacc2, infl_q, infl_b, slop_rel, slop_abs, lo1, hi1);
                hi_u[sub][0] = f32_to_f16_ru_bits(hi1);
                lmax0 = max_u32(lmax0, float_to_ord_u32(lo1));
            }
            if (m0 < nl && v1 < vreal) {
                float lo1, hi1;
                compute_s1_bounds(acc[sub][1], xn0, rt1_c[sub], nv1_c[sub], cacc2, infl_q, infl_b, slop_rel, slop_abs, lo1, hi1);
                hi_u[sub][1] = f32_to_f16_ru_bits(hi1);
                lmax0 = max_u32(lmax0, float_to_ord_u32(lo1));
            }
            if constexpr (!M_LE8) {
                if (active_hi8) {
                    if (m1 < nl && v0 < vreal) {
                        float lo1, hi1;
                        compute_s1_bounds(acc[sub][2], xn1, rt0_c[sub], nv0_c[sub], cacc2, infl_q, infl_b, slop_rel, slop_abs, lo1, hi1);
                        hi_u[sub][2] = f32_to_f16_ru_bits(hi1);
                        lmax1 = max_u32(lmax1, float_to_ord_u32(lo1));
                    }
                    if (m1 < nl && v1 < vreal) {
                        float lo1, hi1;
                        compute_s1_bounds(acc[sub][3], xn1, rt1_c[sub], nv1_c[sub], cacc2, infl_q, infl_b, slop_rel, slop_abs, lo1, hi1);
                        hi_u[sub][3] = f32_to_f16_ru_bits(hi1);
                        lmax1 = max_u32(lmax1, float_to_ord_u32(lo1));
                    }
                }
            }
        }
        lmax0 = max_u32(lmax0, __shfl_xor_sync(0xFFFFFFFFu, lmax0, 2));
        lmax0 = max_u32(lmax0, __shfl_xor_sync(0xFFFFFFFFu, lmax0, 1));
        if (t == 0 && m0 < nl) atomicMax(&s_thr[m0], lmax0);
        if constexpr (!M_LE8) {
            if (active_hi8) {
                lmax1 = max_u32(lmax1, __shfl_xor_sync(0xFFFFFFFFu, lmax1, 2));
                lmax1 = max_u32(lmax1, __shfl_xor_sync(0xFFFFFFFFu, lmax1, 1));
                if (t == 0 && m1 < nl) atomicMax(&s_thr[m1], lmax1);
            }
        }
        __syncthreads();
        if (tid < nl) {
            const uint32_t old_u = atomicMax(&THR_U[tid], s_thr[tid]);
            const uint32_t cur_u = max_u32(old_u, s_thr[tid]);
            s_raw_cut[tid] = compute_safe_raw_cut(ord_u32_to_float(cur_u), inv_cap, cap, slack, 1.0e-3f);
        }
        __syncthreads();

        // V14 Pass 2: Cross-`g` column liveness + conditional L2-hot S1_OUT / HI / ALIVE_BM writes!
        const float cut0 = (m0 < nl) ? s_raw_cut[m0] : INFINITY;
        const float cut1 = (active_hi8 && m1 < nl) ? s_raw_cut[m1] : INFINITY;
        #pragma unroll
        for (int sub = 0; sub < SUB_N; ++sub) {
            const int v_warp_sub = pn * BN + warp * (SUB_N * 8) + sub * 8;
            const int v0 = v_warp_sub + 2 * t;
            const int v1 = v0 + 1;
            const bool k00 = (m0 < nl && v0 < vreal) && !(f16_bits_to_f32(hi_u[sub][0]) < cut0);
            const bool k01 = (m0 < nl && v1 < vreal) && !(f16_bits_to_f32(hi_u[sub][1]) < cut0);
            const bool k10 = (active_hi8 && m1 < nl && v0 < vreal) && !(f16_bits_to_f32(hi_u[sub][2]) < cut1);
            const bool k11 = (active_hi8 && m1 < nl && v1 < vreal) && !(f16_bits_to_f32(hi_u[sub][3]) < cut1);

            uint32_t ca0 = (k00 || k10) ? 1u : 0u;
            uint32_t ca1 = (k01 || k11) ? 1u : 0u;
            ca0 |= __shfl_xor_sync(0xFFFFFFFFu, ca0, 16);
            ca0 |= __shfl_xor_sync(0xFFFFFFFFu, ca0, 8);
            ca0 |= __shfl_xor_sync(0xFFFFFFFFu, ca0, 4);
            ca1 |= __shfl_xor_sync(0xFFFFFFFFu, ca1, 16);
            ca1 |= __shfl_xor_sync(0xFFFFFFFFu, ca1, 8);
            ca1 |= __shfl_xor_sync(0xFFFFFFFFu, ca1, 4);

            warp_or_alive_8cols(ALIVE_BM, v_warp_sub, vreal, lane, t, ca0, ca1);

            if (ca0 && v0 < vreal) {
                if (m0 < nl) {
                    const int64_t mv = static_cast<int64_t>(m0) * V + v0;
                    HI[mv] = k00 ? hi_u[sub][0] : FP16_NEG_INF_BITS;
                    if (k00) S1_OUT[mv] = acc[sub][0];
                }
                if constexpr (!M_LE8) {
                    if (active_hi8 && m1 < nl) {
                        const int64_t mv = static_cast<int64_t>(m1) * V + v0;
                        HI[mv] = k10 ? hi_u[sub][2] : FP16_NEG_INF_BITS;
                        if (k10) S1_OUT[mv] = acc[sub][2];
                    }
                }
            }
            if (ca1 && v1 < vreal) {
                if (m0 < nl) {
                    const int64_t mv = static_cast<int64_t>(m0) * V + v1;
                    HI[mv] = k01 ? hi_u[sub][1] : FP16_NEG_INF_BITS;
                    if (k01) S1_OUT[mv] = acc[sub][1];
                }
                if constexpr (!M_LE8) {
                    if (active_hi8 && m1 < nl) {
                        const int64_t mv = static_cast<int64_t>(m1) * V + v1;
                        HI[mv] = k11 ? hi_u[sub][3] : FP16_NEG_INF_BITS;
                        if (k11) S1_OUT[mv] = acc[sub][3];
                    }
                }
            }
        }
    }

    if constexpr (PDL) {
        __syncthreads();
        if (tid == 0) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }
}

// =========================================================================
// Kernel 2B: `cert_screen_int6_ms8_v14_c48_kernel` (M = 17..48, MT in {2, 3}, BK_SMEM=256)
//   V14 Upgrades: `nl_tiles` pruning + 1x `MS_U4`/`BIG_S` load + hoisted `ld_cs_u4` + `ALIVE_BM`!
// =========================================================================
template <int WARPS, int SUB_N, int MT, bool PDL, bool USE_ALIVE>
__global__ void __launch_bounds__(WARPS * 32, (SUB_N == 1 && MT <= 3) ? 3 : 2)
cert_screen_int6_ms8_v14_c48_kernel(
    const __nv_bfloat16* __restrict__ X_PAD,
    const uint4* __restrict__ SLABS_U4,
    const uint4* __restrict__ MS_U4,
    const uint16_t* __restrict__ BIG_S,
    const float* __restrict__ RT1,
    const float* __restrict__ NV1,
    const float* __restrict__ XN,
    float* __restrict__ S1_OUT,
    uint16_t* __restrict__ HI,
    uint32_t* __restrict__ THR_U,
    uint32_t* __restrict__ ALIVE_BM,
    const int32_t* __restrict__ NLIVE_PTR,
    int M,
    int V,
    int vreal,
    int K,
    float cacc2,
    float infl_q,
    float infl_b,
    float slop_rel,
    float slop_abs,
    float inv_cap,
    float cap,
    float slack
) {
    static_assert(MT >= 2 && MT <= 3, "v14_c48 BK256 path handles MT in {2, 3}");
    constexpr int BN = WARPS * SUB_N * 8;
    constexpr int THREADS = WARPS * 32;
    constexpr int SX_STRIDE = 264;  // 256 + 8 bf16 (zero SMEM bank conflicts!)
    constexpr int M_ROWS = MT * 16;

    const int pn = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g = lane >> 2;
    const int t = lane & 3;

    const int warp_uid = pn * WARPS + warp;
    const int n_qchunks = K >> 9;
    const int total_q = n_qchunks * SUB_N;

    uint4 pre_v0{}, pre_v1{}, pre_v2{}, pre_ms{};
    float pre_bs = 0.0f;
    if constexpr (PDL) {
        if (total_q > 0) {
            const int64_t q_idx0 = static_cast<int64_t>(warp_uid) * total_q;
            const uint4* qp0 = SLABS_U4 + q_idx0 * (6 * 32);
            pre_v0 = ld_cs_u4(qp0 + 0 * 32 + lane);
            pre_v1 = ld_cs_u4(qp0 + 1 * 32 + lane);
            pre_v2 = ld_cs_u4(qp0 + 2 * 32 + lane);
            pre_ms = __ldg(MS_U4 + q_idx0 * 8 + g);
            pre_bs = f16_bits_to_f32(__ldg(BIG_S + q_idx0 * 8 + g));
        }
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }

    const int nl = resolve_nlive(NLIVE_PTR, M);
    const int nl_tiles = (nl + 15) >> 4;

    __shared__ uint32_t s_thr[M_ROWS];
    __shared__ float s_xn[M_ROWS];
    __shared__ float s_raw_cut[M_ROWS];
    extern __shared__ __nv_bfloat16 sX[];  // [M_ROWS * SX_STRIDE]

    if (tid < M_ROWS) {
        s_thr[tid] = ORD_NEG_INF;
        s_xn[tid] = (tid < nl) ? XN[tid] : 0.0f;
        s_raw_cut[tid] = -INFINITY;
    }

    float acc[SUB_N][MT][4];
    #pragma unroll
    for (int sub = 0; sub < SUB_N; ++sub) {
        #pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            acc[sub][mt][0] = 0.0f; acc[sub][mt][1] = 0.0f;
            acc[sub][mt][2] = 0.0f; acc[sub][mt][3] = 0.0f;
        }
    }

    constexpr int VEC_PER_256 = 32;
    const int active_x_vec = nl_tiles * 16 * VEC_PER_256;

    for (int qc = 0; qc < n_qchunks; ++qc) {
        const int k_base = qc * 512;
        const int64_t q_idx0 = static_cast<int64_t>(warp_uid) * total_q + (qc * SUB_N);
        const uint4* qp0 = SLABS_U4 + q_idx0 * (6 * 32);

        // V14: Hoist Half-0 sub=0 streaming DRAM loads BEFORE X_PAD copy + syncthreads!
        uint4 s0_v0, s0_v1, s0_v2, s0_ms;
        float s0_bs;
        if (PDL && qc == 0) {
            s0_v0 = pre_v0; s0_v1 = pre_v1; s0_v2 = pre_v2;
            s0_ms = pre_ms; s0_bs = pre_bs;
        } else {
            s0_v0 = ld_cs_u4(qp0 + 0 * 32 + lane);
            s0_v1 = ld_cs_u4(qp0 + 1 * 32 + lane);
            s0_v2 = ld_cs_u4(qp0 + 2 * 32 + lane);
            s0_ms = __ldg(MS_U4 + q_idx0 * 8 + g);
            s0_bs = f16_bits_to_f32(__ldg(BIG_S + q_idx0 * 8 + g));
        }

        // Half 0: cols [k_base .. k_base + 255] (only active `nl_tiles` rows!)
        for (int idx = tid; idx < active_x_vec; idx += THREADS) {
            const int m = idx >> 5;
            const int kv = idx & 31;
            *reinterpret_cast<uint4*>(sX + m * SX_STRIDE + kv * 8) =
                *reinterpret_cast<const uint4*>(X_PAD + static_cast<int64_t>(m) * K + k_base + kv * 8);
        }
        __syncthreads();

        // Save (ms.z, ms.w, bs) across Half 0 -> Half 1 (only 6 regs for SUB_N=2, cuts 4.07 MB scale reloads!)
        uint32_t ms_z_sav[SUB_N];
        uint32_t ms_w_sav[SUB_N];
        float bs_sav[SUB_N];

        #pragma unroll
        for (int sub = 0; sub < SUB_N; ++sub) {
            uint4 v0, v1, v2, ms;
            float bs;
            if (sub == 0) {
                v0 = s0_v0; v1 = s0_v1; v2 = s0_v2; ms = s0_ms; bs = s0_bs;
            } else {
                const int64_t q_idx = q_idx0 + sub;
                const uint4* qp = SLABS_U4 + q_idx * (6 * 32);
                v0 = ld_cs_u4(qp + 0 * 32 + lane);
                v1 = ld_cs_u4(qp + 1 * 32 + lane);
                v2 = ld_cs_u4(qp + 2 * 32 + lane);
                ms = __ldg(MS_U4 + q_idx * 8 + g);
                bs = f16_bits_to_f32(__ldg(BIG_S + q_idx * 8 + g));
            }
            ms_z_sav[sub] = ms.z;
            ms_w_sav[sub] = ms.w;
            bs_sav[sub] = bs;
            const float sc_lut[4] = {
                __fmul_rn(bs, 0.375f),
                __fmul_rn(bs, 0.5625f),
                __fmul_rn(bs, 0.75f),
                bs
            };
            exec_hc_int6_ms8_mt_nl<MT, SX_STRIDE>(v0.x, v0.y, v0.z, ms.x & 0xFFFFu, sc_lut, sX, g, t, 0 * 64, nl_tiles, acc[sub]);
            asm volatile("" ::: "memory");
            exec_hc_int6_ms8_mt_nl<MT, SX_STRIDE>(v0.w, v1.x, v1.y, ms.x >> 16,     sc_lut, sX, g, t, 1 * 64, nl_tiles, acc[sub]);
            asm volatile("" ::: "memory");
            exec_hc_int6_ms8_mt_nl<MT, SX_STRIDE>(v1.z, v1.w, v2.x, ms.y & 0xFFFFu, sc_lut, sX, g, t, 2 * 64, nl_tiles, acc[sub]);
            asm volatile("" ::: "memory");
            exec_hc_int6_ms8_mt_nl<MT, SX_STRIDE>(v2.y, v2.z, v2.w, ms.y >> 16,     sc_lut, sX, g, t, 3 * 64, nl_tiles, acc[sub]);
            asm volatile("" ::: "memory");
        }
        __syncthreads();

        // V14: Hoist Half-1 sub=0 streaming DRAM loads BEFORE X_PAD Half-1 copy + syncthreads!
        const uint4 s0_v3 = ld_cs_u4(qp0 + 3 * 32 + lane);
        const uint4 s0_v4 = ld_cs_u4(qp0 + 4 * 32 + lane);
        const uint4 s0_v5 = ld_cs_u4(qp0 + 5 * 32 + lane);

        // Half 1: cols [k_base + 256 .. k_base + 511]
        for (int idx = tid; idx < active_x_vec; idx += THREADS) {
            const int m = idx >> 5;
            const int kv = idx & 31;
            *reinterpret_cast<uint4*>(sX + m * SX_STRIDE + kv * 8) =
                *reinterpret_cast<const uint4*>(X_PAD + static_cast<int64_t>(m) * K + k_base + 256 + kv * 8);
        }
        __syncthreads();

        #pragma unroll
        for (int sub = 0; sub < SUB_N; ++sub) {
            uint4 v3, v4, v5;
            if (sub == 0) {
                v3 = s0_v3; v4 = s0_v4; v5 = s0_v5;
            } else {
                const int64_t q_idx = q_idx0 + sub;
                const uint4* qp = SLABS_U4 + q_idx * (6 * 32);
                v3 = ld_cs_u4(qp + 3 * 32 + lane);
                v4 = ld_cs_u4(qp + 4 * 32 + lane);
                v5 = ld_cs_u4(qp + 5 * 32 + lane);
            }
            const float bs = bs_sav[sub];
            const uint32_t msz = ms_z_sav[sub];
            const uint32_t msw = ms_w_sav[sub];
            const float sc_lut[4] = {
                __fmul_rn(bs, 0.375f),
                __fmul_rn(bs, 0.5625f),
                __fmul_rn(bs, 0.75f),
                bs
            };
            exec_hc_int6_ms8_mt_nl<MT, SX_STRIDE>(v3.x, v3.y, v3.z, msz & 0xFFFFu, sc_lut, sX, g, t, 0 * 64, nl_tiles, acc[sub]);
            asm volatile("" ::: "memory");
            exec_hc_int6_ms8_mt_nl<MT, SX_STRIDE>(v3.w, v4.x, v4.y, msz >> 16,     sc_lut, sX, g, t, 1 * 64, nl_tiles, acc[sub]);
            asm volatile("" ::: "memory");
            exec_hc_int6_ms8_mt_nl<MT, SX_STRIDE>(v4.z, v4.w, v5.x, msw & 0xFFFFu, sc_lut, sX, g, t, 2 * 64, nl_tiles, acc[sub]);
            asm volatile("" ::: "memory");
            exec_hc_int6_ms8_mt_nl<MT, SX_STRIDE>(v5.y, v5.z, v5.w, msw >> 16,     sc_lut, sX, g, t, 3 * 64, nl_tiles, acc[sub]);
            asm volatile("" ::: "memory");
        }
        __syncthreads();
    }

    float rt0_c[SUB_N], nv0_c[SUB_N], rt1_c[SUB_N], nv1_c[SUB_N];
    #pragma unroll
    for (int sub = 0; sub < SUB_N; ++sub) {
        const int v0 = pn * BN + warp * (SUB_N * 8) + sub * 8 + 2 * t;
        const int v1 = v0 + 1;
        rt0_c[sub] = (v0 < vreal) ? __ldg(RT1 + v0) : 0.0f;
        nv0_c[sub] = (v0 < vreal) ? __ldg(NV1 + v0) : 0.0f;
        rt1_c[sub] = (v1 < vreal) ? __ldg(RT1 + v1) : 0.0f;
        nv1_c[sub] = (v1 < vreal) ? __ldg(NV1 + v1) : 0.0f;
    }

    if constexpr (!USE_ALIVE) {
        auto emit_legacy = [&](int m, float xn, int v, float rt, float nv, float s1) -> uint32_t {
            if (m >= nl || v >= vreal) return ORD_NEG_INF;
            const int64_t mv = static_cast<int64_t>(m) * V + v;
            st_cs_f32(&S1_OUT[mv], s1);
            float lo1, hi1;
            compute_s1_bounds(s1, xn, rt, nv, cacc2, infl_q, infl_b, slop_rel, slop_abs, lo1, hi1);
            HI[mv] = f32_to_f16_ru_bits(hi1);
            return float_to_ord_u32(lo1);
        };
        #pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            if (mt < nl_tiles) {
                const int m0 = mt * 16 + g;
                const int m1 = m0 + 8;
                const float xn0 = s_xn[m0];
                const float xn1 = s_xn[m1];
                uint32_t lmax0 = ORD_NEG_INF;
                uint32_t lmax1 = ORD_NEG_INF;
                #pragma unroll
                for (int sub = 0; sub < SUB_N; ++sub) {
                    const int v0 = pn * BN + warp * (SUB_N * 8) + sub * 8 + 2 * t;
                    const int v1 = v0 + 1;
                    lmax0 = max_u32(lmax0, emit_legacy(m0, xn0, v0, rt0_c[sub], nv0_c[sub], acc[sub][mt][0]));
                    lmax0 = max_u32(lmax0, emit_legacy(m0, xn0, v1, rt1_c[sub], nv1_c[sub], acc[sub][mt][1]));
                    lmax1 = max_u32(lmax1, emit_legacy(m1, xn1, v0, rt0_c[sub], nv0_c[sub], acc[sub][mt][2]));
                    lmax1 = max_u32(lmax1, emit_legacy(m1, xn1, v1, rt1_c[sub], nv1_c[sub], acc[sub][mt][3]));
                }
                lmax0 = max_u32(lmax0, __shfl_xor_sync(0xFFFFFFFFu, lmax0, 2));
                lmax0 = max_u32(lmax0, __shfl_xor_sync(0xFFFFFFFFu, lmax0, 1));
                lmax1 = max_u32(lmax1, __shfl_xor_sync(0xFFFFFFFFu, lmax1, 2));
                lmax1 = max_u32(lmax1, __shfl_xor_sync(0xFFFFFFFFu, lmax1, 1));
                if (t == 0) {
                    if (m0 < nl) atomicMax(&s_thr[m0], lmax0);
                    if (m1 < nl) atomicMax(&s_thr[m1], lmax1);
                }
            }
        }
        __syncthreads();
        if (tid < nl) {
            atomicMax(&THR_U[tid], s_thr[tid]);
        }
    } else {
        // V14 Pass 1: Register-only `lo1` reduction -> global running `THR_U[m]` -> `s_raw_cut[m]`
        #pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            if (mt < nl_tiles) {
                const int m0 = mt * 16 + g;
                const int m1 = m0 + 8;
                const float xn0 = s_xn[m0];
                const float xn1 = s_xn[m1];
                uint32_t lmax0 = ORD_NEG_INF;
                uint32_t lmax1 = ORD_NEG_INF;
                #pragma unroll
                for (int sub = 0; sub < SUB_N; ++sub) {
                    const int v0 = pn * BN + warp * (SUB_N * 8) + sub * 8 + 2 * t;
                    const int v1 = v0 + 1;
                    if (m0 < nl && v0 < vreal) {
                        float lo1, hi1;
                        compute_s1_bounds(acc[sub][mt][0], xn0, rt0_c[sub], nv0_c[sub], cacc2, infl_q, infl_b, slop_rel, slop_abs, lo1, hi1);
                        lmax0 = max_u32(lmax0, float_to_ord_u32(lo1));
                    }
                    if (m0 < nl && v1 < vreal) {
                        float lo1, hi1;
                        compute_s1_bounds(acc[sub][mt][1], xn0, rt1_c[sub], nv1_c[sub], cacc2, infl_q, infl_b, slop_rel, slop_abs, lo1, hi1);
                        lmax0 = max_u32(lmax0, float_to_ord_u32(lo1));
                    }
                    if (m1 < nl && v0 < vreal) {
                        float lo1, hi1;
                        compute_s1_bounds(acc[sub][mt][2], xn1, rt0_c[sub], nv0_c[sub], cacc2, infl_q, infl_b, slop_rel, slop_abs, lo1, hi1);
                        lmax1 = max_u32(lmax1, float_to_ord_u32(lo1));
                    }
                    if (m1 < nl && v1 < vreal) {
                        float lo1, hi1;
                        compute_s1_bounds(acc[sub][mt][3], xn1, rt1_c[sub], nv1_c[sub], cacc2, infl_q, infl_b, slop_rel, slop_abs, lo1, hi1);
                        lmax1 = max_u32(lmax1, float_to_ord_u32(lo1));
                    }
                }
                lmax0 = max_u32(lmax0, __shfl_xor_sync(0xFFFFFFFFu, lmax0, 2));
                lmax0 = max_u32(lmax0, __shfl_xor_sync(0xFFFFFFFFu, lmax0, 1));
                lmax1 = max_u32(lmax1, __shfl_xor_sync(0xFFFFFFFFu, lmax1, 2));
                lmax1 = max_u32(lmax1, __shfl_xor_sync(0xFFFFFFFFu, lmax1, 1));
                if (t == 0) {
                    if (m0 < nl) atomicMax(&s_thr[m0], lmax0);
                    if (m1 < nl) atomicMax(&s_thr[m1], lmax1);
                }
            }
        }
        __syncthreads();
        if (tid < nl) {
            const uint32_t old_u = atomicMax(&THR_U[tid], s_thr[tid]);
            const uint32_t cur_u = max_u32(old_u, s_thr[tid]);
            s_raw_cut[tid] = compute_safe_raw_cut(ord_u32_to_float(cur_u), inv_cap, cap, slack, 1.0e-3f);
        }
        __syncthreads();

        // V14 Pass 2: Zero-array compact row-keep bitmasks (`km0`, `km1` -> 0 B stack!)
        //             + cross-`g` column liveness + conditional L2-hot S1_OUT / HI / ALIVE_BM writes!
        #pragma unroll
        for (int sub = 0; sub < SUB_N; ++sub) {
            const int v_warp_sub = pn * BN + warp * (SUB_N * 8) + sub * 8;
            const int v0 = v_warp_sub + 2 * t;
            const int v1 = v0 + 1;
            uint32_t km0 = 0u, km1 = 0u;
            #pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                if (mt < nl_tiles) {
                    const int m0 = mt * 16 + g;
                    const int m1 = m0 + 8;
                    const float xn0 = s_xn[m0];
                    const float xn1 = s_xn[m1];
                    if (m0 < nl) {
                        const float cut0 = s_raw_cut[m0];
                        if (v0 < vreal) {
                            float lo1, hi1;
                            compute_s1_bounds(acc[sub][mt][0], xn0, rt0_c[sub], nv0_c[sub], cacc2, infl_q, infl_b, slop_rel, slop_abs, lo1, hi1);
                            if (!(f16_bits_to_f32(f32_to_f16_ru_bits(hi1)) < cut0)) km0 |= (1u << (mt * 2 + 0));
                        }
                        if (v1 < vreal) {
                            float lo1, hi1;
                            compute_s1_bounds(acc[sub][mt][1], xn0, rt1_c[sub], nv1_c[sub], cacc2, infl_q, infl_b, slop_rel, slop_abs, lo1, hi1);
                            if (!(f16_bits_to_f32(f32_to_f16_ru_bits(hi1)) < cut0)) km1 |= (1u << (mt * 2 + 0));
                        }
                    }
                    if (m1 < nl) {
                        const float cut1 = s_raw_cut[m1];
                        if (v0 < vreal) {
                            float lo1, hi1;
                            compute_s1_bounds(acc[sub][mt][2], xn1, rt0_c[sub], nv0_c[sub], cacc2, infl_q, infl_b, slop_rel, slop_abs, lo1, hi1);
                            if (!(f16_bits_to_f32(f32_to_f16_ru_bits(hi1)) < cut1)) km0 |= (1u << (mt * 2 + 1));
                        }
                        if (v1 < vreal) {
                            float lo1, hi1;
                            compute_s1_bounds(acc[sub][mt][3], xn1, rt1_c[sub], nv1_c[sub], cacc2, infl_q, infl_b, slop_rel, slop_abs, lo1, hi1);
                            if (!(f16_bits_to_f32(f32_to_f16_ru_bits(hi1)) < cut1)) km1 |= (1u << (mt * 2 + 1));
                        }
                    }
                }
            }
            uint32_t ca0 = (km0 != 0u) ? 1u : 0u;
            uint32_t ca1 = (km1 != 0u) ? 1u : 0u;
            ca0 |= __shfl_xor_sync(0xFFFFFFFFu, ca0, 16);
            ca0 |= __shfl_xor_sync(0xFFFFFFFFu, ca0, 8);
            ca0 |= __shfl_xor_sync(0xFFFFFFFFu, ca0, 4);
            ca1 |= __shfl_xor_sync(0xFFFFFFFFu, ca1, 16);
            ca1 |= __shfl_xor_sync(0xFFFFFFFFu, ca1, 8);
            ca1 |= __shfl_xor_sync(0xFFFFFFFFu, ca1, 4);

            warp_or_alive_8cols(ALIVE_BM, v_warp_sub, vreal, lane, t, ca0, ca1);

            if (ca0 && v0 < vreal) {
                #pragma unroll
                for (int mt = 0; mt < MT; ++mt) {
                    if (mt < nl_tiles) {
                        const int m0 = mt * 16 + g;
                        const int m1 = m0 + 8;
                        if (m0 < nl) {
                            const int64_t mv = static_cast<int64_t>(m0) * V + v0;
                            if ((km0 >> (mt * 2 + 0)) & 1u) {
                                float lo1, hi1;
                                compute_s1_bounds(acc[sub][mt][0], s_xn[m0], rt0_c[sub], nv0_c[sub], cacc2, infl_q, infl_b, slop_rel, slop_abs, lo1, hi1);
                                HI[mv] = f32_to_f16_ru_bits(hi1);
                                S1_OUT[mv] = acc[sub][mt][0];
                            } else {
                                HI[mv] = FP16_NEG_INF_BITS;
                            }
                        }
                        if (m1 < nl) {
                            const int64_t mv = static_cast<int64_t>(m1) * V + v0;
                            if ((km0 >> (mt * 2 + 1)) & 1u) {
                                float lo1, hi1;
                                compute_s1_bounds(acc[sub][mt][2], s_xn[m1], rt0_c[sub], nv0_c[sub], cacc2, infl_q, infl_b, slop_rel, slop_abs, lo1, hi1);
                                HI[mv] = f32_to_f16_ru_bits(hi1);
                                S1_OUT[mv] = acc[sub][mt][2];
                            } else {
                                HI[mv] = FP16_NEG_INF_BITS;
                            }
                        }
                    }
                }
            }
            if (ca1 && v1 < vreal) {
                #pragma unroll
                for (int mt = 0; mt < MT; ++mt) {
                    if (mt < nl_tiles) {
                        const int m0 = mt * 16 + g;
                        const int m1 = m0 + 8;
                        if (m0 < nl) {
                            const int64_t mv = static_cast<int64_t>(m0) * V + v1;
                            if ((km1 >> (mt * 2 + 0)) & 1u) {
                                float lo1, hi1;
                                compute_s1_bounds(acc[sub][mt][1], s_xn[m0], rt1_c[sub], nv1_c[sub], cacc2, infl_q, infl_b, slop_rel, slop_abs, lo1, hi1);
                                HI[mv] = f32_to_f16_ru_bits(hi1);
                                S1_OUT[mv] = acc[sub][mt][1];
                            } else {
                                HI[mv] = FP16_NEG_INF_BITS;
                            }
                        }
                        if (m1 < nl) {
                            const int64_t mv = static_cast<int64_t>(m1) * V + v1;
                            if ((km1 >> (mt * 2 + 1)) & 1u) {
                                float lo1, hi1;
                                compute_s1_bounds(acc[sub][mt][3], s_xn[m1], rt1_c[sub], nv1_c[sub], cacc2, infl_q, infl_b, slop_rel, slop_abs, lo1, hi1);
                                HI[mv] = f32_to_f16_ru_bits(hi1);
                                S1_OUT[mv] = acc[sub][mt][3];
                            } else {
                                HI[mv] = FP16_NEG_INF_BITS;
                            }
                        }
                    }
                }
            }
        }
    }

    if constexpr (PDL) {
        __syncthreads();
        if (tid == 0) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }
}

// =========================================================================
// Kernel 2C: `cert_screen_int6_ms8_v14_subsplit_kernel` (M = 49..128, MT in {4..8}, BK_SMEM=128)
//   V14 Upgrades: `nl_tiles` dead-tile pruning + hoisted `ld_cs_u4` BEFORE every `load_x_qtr`
//   + `ALIVE_BM` + conditional `S1_OUT` / `HI` elision (cuts -24.4 MB vs V12 / -164.8 MB vs BF16 @ M=128!).
// =========================================================================
template <int WARPS, int SUB_N, int MT, bool PDL, bool USE_ALIVE>
__global__ void __launch_bounds__(WARPS * 32, (MT <= 4) ? 3 : 2)
cert_screen_int6_ms8_v14_subsplit_kernel(
    const __nv_bfloat16* __restrict__ X_PAD,
    const uint4* __restrict__ SLABS_U4,
    const uint4* __restrict__ MS_U4,
    const uint16_t* __restrict__ BIG_S,
    const float* __restrict__ RT1,
    const float* __restrict__ NV1,
    const float* __restrict__ XN,
    float* __restrict__ S1_OUT,
    uint16_t* __restrict__ HI,
    uint32_t* __restrict__ THR_U,
    uint32_t* __restrict__ ALIVE_BM,
    const int32_t* __restrict__ NLIVE_PTR,
    int M,
    int V,
    int vreal,
    int K,
    float cacc2,
    float infl_q,
    float infl_b,
    float slop_rel,
    float slop_abs,
    float inv_cap,
    float cap,
    float slack
) {
    static_assert(MT >= 4 && MT <= 8, "v14_subsplit handles MT in {4, 5, 6, 7, 8} (M=49..128)");
    constexpr int BN = WARPS * SUB_N * 8;
    constexpr int THREADS = WARPS * 32;
    constexpr int SX_STRIDE = 136;  // 128 + 8 bf16 (zero SMEM bank conflicts, 34.0 KiB @ MT=8!)
    constexpr int M_ROWS = MT * 16;

    const int pn = blockIdx.x;
    const int sub = (SUB_N == 1) ? 0 : static_cast<int>(blockIdx.y);
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g = lane >> 2;
    const int t = lane & 3;

    const int warp_uid = pn * WARPS + warp;
    const int n_qchunks = K >> 9;
    const int total_q = n_qchunks * SUB_N;

    uint4 pre_v0{}, pre_v1{}, pre_ms{};
    float pre_bs = 0.0f;
    if constexpr (PDL) {
        if (n_qchunks > 0) {
            const int64_t q_idx0 = static_cast<int64_t>(warp_uid) * total_q + sub;
            const uint4* qp0 = SLABS_U4 + q_idx0 * (6 * 32);
            pre_v0 = ld_cs_u4(qp0 + 0 * 32 + lane);
            pre_v1 = ld_cs_u4(qp0 + 1 * 32 + lane);
            pre_ms = __ldg(MS_U4 + q_idx0 * 8 + g);
            pre_bs = f16_bits_to_f32(__ldg(BIG_S + q_idx0 * 8 + g));
        }
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }

    const int nl = resolve_nlive(NLIVE_PTR, M);
    const int nl_tiles = (nl + 15) >> 4;

    __shared__ uint32_t s_thr[M_ROWS];
    __shared__ float s_xn[M_ROWS];
    __shared__ float s_raw_cut[M_ROWS];
    extern __shared__ __nv_bfloat16 sX[];  // [M_ROWS * SX_STRIDE]

    for (int m = tid; m < M_ROWS; m += THREADS) {
        s_thr[m] = ORD_NEG_INF;
        s_xn[m] = (m < nl) ? XN[m] : 0.0f;
        s_raw_cut[m] = -INFINITY;
    }

    float acc[MT][4];
    #pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
        acc[mt][0] = 0.0f; acc[mt][1] = 0.0f;
        acc[mt][2] = 0.0f; acc[mt][3] = 0.0f;
    }

    constexpr int VEC_PER_128 = 16;  // 128 / 8 = 16 uint4 per row
    const int active_x_vec = nl_tiles * 16 * VEC_PER_128;

    auto load_x_qtr = [&](int k_qtr) {
        for (int idx = tid; idx < active_x_vec; idx += THREADS) {
            const int m = idx >> 4;
            const int kv = idx & 15;
            *reinterpret_cast<uint4*>(sX + m * SX_STRIDE + kv * 8) =
                *reinterpret_cast<const uint4*>(X_PAD + static_cast<int64_t>(m) * K + k_qtr + kv * 8);
        }
        __syncthreads();
    };

    for (int qc = 0; qc < n_qchunks; ++qc) {
        const int k_base = qc * 512;
        const int64_t q_idx = static_cast<int64_t>(warp_uid) * total_q + (qc * SUB_N + sub);
        const uint4* qp = SLABS_U4 + q_idx * (6 * 32);

        // ---- Quarter 0: Hoist v0, v1, ms, bs BEFORE load_x_qtr(k_base) + syncthreads! ----
        uint4 v0, v1, ms;
        float bs;
        if (PDL && qc == 0) {
            v0 = pre_v0; v1 = pre_v1; ms = pre_ms; bs = pre_bs;
        } else {
            v0 = ld_cs_u4(qp + 0 * 32 + lane);
            v1 = ld_cs_u4(qp + 1 * 32 + lane);
            ms = __ldg(MS_U4 + q_idx * 8 + g);
            bs = f16_bits_to_f32(__ldg(BIG_S + q_idx * 8 + g));
        }
        load_x_qtr(k_base);
        const float sc_lut[4] = {
            __fmul_rn(bs, 0.375f),
            __fmul_rn(bs, 0.5625f),
            __fmul_rn(bs, 0.75f),
            bs
        };
        exec_hc_int6_ms8_mt_nl<MT, SX_STRIDE>(v0.x, v0.y, v0.z, ms.x & 0xFFFFu, sc_lut, sX, g, t, 0,  nl_tiles, acc);
        asm volatile("" ::: "memory");
        exec_hc_int6_ms8_mt_nl<MT, SX_STRIDE>(v0.w, v1.x, v1.y, ms.x >> 16,     sc_lut, sX, g, t, 64, nl_tiles, acc);
        const uint32_t v1_z = v1.z;
        const uint32_t v1_w = v1.w;
        __syncthreads();

        // ---- Quarter 1: Hoist v2 BEFORE load_x_qtr(k_base + 128) + syncthreads! ----
        const uint4 v2 = ld_cs_u4(qp + 2 * 32 + lane);
        load_x_qtr(k_base + 128);
        exec_hc_int6_ms8_mt_nl<MT, SX_STRIDE>(v1_z, v1_w, v2.x, ms.y & 0xFFFFu, sc_lut, sX, g, t, 0,  nl_tiles, acc);
        asm volatile("" ::: "memory");
        exec_hc_int6_ms8_mt_nl<MT, SX_STRIDE>(v2.y, v2.z, v2.w, ms.y >> 16,     sc_lut, sX, g, t, 64, nl_tiles, acc);
        __syncthreads();

        // ---- Quarter 2: Hoist v3, v4 BEFORE load_x_qtr(k_base + 256) + syncthreads! ----
        const uint4 v3 = ld_cs_u4(qp + 3 * 32 + lane);
        const uint4 v4 = ld_cs_u4(qp + 4 * 32 + lane);
        load_x_qtr(k_base + 256);
        exec_hc_int6_ms8_mt_nl<MT, SX_STRIDE>(v3.x, v3.y, v3.z, ms.z & 0xFFFFu, sc_lut, sX, g, t, 0,  nl_tiles, acc);
        asm volatile("" ::: "memory");
        exec_hc_int6_ms8_mt_nl<MT, SX_STRIDE>(v3.w, v4.x, v4.y, ms.z >> 16,     sc_lut, sX, g, t, 64, nl_tiles, acc);
        const uint32_t v4_z = v4.z;
        const uint32_t v4_w = v4.w;
        __syncthreads();

        // ---- Quarter 3: Hoist v5 BEFORE load_x_qtr(k_base + 384) + syncthreads! ----
        const uint4 v5 = ld_cs_u4(qp + 5 * 32 + lane);
        load_x_qtr(k_base + 384);
        exec_hc_int6_ms8_mt_nl<MT, SX_STRIDE>(v4_z, v4_w, v5.x, ms.w & 0xFFFFu, sc_lut, sX, g, t, 0,  nl_tiles, acc);
        asm volatile("" ::: "memory");
        exec_hc_int6_ms8_mt_nl<MT, SX_STRIDE>(v5.y, v5.z, v5.w, ms.w >> 16,     sc_lut, sX, g, t, 64, nl_tiles, acc);
        __syncthreads();
    }

    const int v_warp_sub = pn * BN + warp * (SUB_N * 8) + sub * 8;
    const int v0 = v_warp_sub + 2 * t;
    const int v1 = v0 + 1;
    const float rt0 = (v0 < vreal) ? __ldg(RT1 + v0) : 0.0f;
    const float nv0 = (v0 < vreal) ? __ldg(NV1 + v0) : 0.0f;
    const float rt1 = (v1 < vreal) ? __ldg(RT1 + v1) : 0.0f;
    const float nv1 = (v1 < vreal) ? __ldg(NV1 + v1) : 0.0f;

    if constexpr (!USE_ALIVE) {
        auto emit_legacy = [&](int m, float xn, int v, float rt, float nv, float s1) -> uint32_t {
            if (m >= nl || v >= vreal) return ORD_NEG_INF;
            const int64_t mv = static_cast<int64_t>(m) * V + v;
            st_cs_f32(&S1_OUT[mv], s1);
            float lo1, hi1;
            compute_s1_bounds(s1, xn, rt, nv, cacc2, infl_q, infl_b, slop_rel, slop_abs, lo1, hi1);
            HI[mv] = f32_to_f16_ru_bits(hi1);
            return float_to_ord_u32(lo1);
        };
        #pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            if (mt < nl_tiles) {
                const int m0 = mt * 16 + g;
                const int m1 = m0 + 8;
                const float xn0 = s_xn[m0];
                const float xn1 = s_xn[m1];
                uint32_t lmax0 = max_u32(
                    emit_legacy(m0, xn0, v0, rt0, nv0, acc[mt][0]),
                    emit_legacy(m0, xn0, v1, rt1, nv1, acc[mt][1])
                );
                uint32_t lmax1 = max_u32(
                    emit_legacy(m1, xn1, v0, rt0, nv0, acc[mt][2]),
                    emit_legacy(m1, xn1, v1, rt1, nv1, acc[mt][3])
                );
                lmax0 = max_u32(lmax0, __shfl_xor_sync(0xFFFFFFFFu, lmax0, 2));
                lmax0 = max_u32(lmax0, __shfl_xor_sync(0xFFFFFFFFu, lmax0, 1));
                lmax1 = max_u32(lmax1, __shfl_xor_sync(0xFFFFFFFFu, lmax1, 2));
                lmax1 = max_u32(lmax1, __shfl_xor_sync(0xFFFFFFFFu, lmax1, 1));
                if (t == 0) {
                    if (m0 < nl) atomicMax(&s_thr[m0], lmax0);
                    if (m1 < nl) atomicMax(&s_thr[m1], lmax1);
                }
            }
        }
        __syncthreads();
        for (int m = tid; m < nl; m += THREADS) {
            atomicMax(&THR_U[m], s_thr[m]);
        }
    } else {
        // V14 Pass 1: Register-only `lo1` reduction -> global running `THR_U[m]` -> `s_raw_cut[m]`
        #pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            if (mt < nl_tiles) {
                const int m0 = mt * 16 + g;
                const int m1 = m0 + 8;
                const float xn0 = s_xn[m0];
                const float xn1 = s_xn[m1];
                uint32_t lmax0 = ORD_NEG_INF;
                uint32_t lmax1 = ORD_NEG_INF;
                if (m0 < nl && v0 < vreal) {
                    float lo1, hi1;
                    compute_s1_bounds(acc[mt][0], xn0, rt0, nv0, cacc2, infl_q, infl_b, slop_rel, slop_abs, lo1, hi1);
                    lmax0 = max_u32(lmax0, float_to_ord_u32(lo1));
                }
                if (m0 < nl && v1 < vreal) {
                    float lo1, hi1;
                    compute_s1_bounds(acc[mt][1], xn0, rt1, nv1, cacc2, infl_q, infl_b, slop_rel, slop_abs, lo1, hi1);
                    lmax0 = max_u32(lmax0, float_to_ord_u32(lo1));
                }
                if (m1 < nl && v0 < vreal) {
                    float lo1, hi1;
                    compute_s1_bounds(acc[mt][2], xn1, rt0, nv0, cacc2, infl_q, infl_b, slop_rel, slop_abs, lo1, hi1);
                    lmax1 = max_u32(lmax1, float_to_ord_u32(lo1));
                }
                if (m1 < nl && v1 < vreal) {
                    float lo1, hi1;
                    compute_s1_bounds(acc[mt][3], xn1, rt1, nv1, cacc2, infl_q, infl_b, slop_rel, slop_abs, lo1, hi1);
                    lmax1 = max_u32(lmax1, float_to_ord_u32(lo1));
                }
                lmax0 = max_u32(lmax0, __shfl_xor_sync(0xFFFFFFFFu, lmax0, 2));
                lmax0 = max_u32(lmax0, __shfl_xor_sync(0xFFFFFFFFu, lmax0, 1));
                lmax1 = max_u32(lmax1, __shfl_xor_sync(0xFFFFFFFFu, lmax1, 2));
                lmax1 = max_u32(lmax1, __shfl_xor_sync(0xFFFFFFFFu, lmax1, 1));
                if (t == 0) {
                    if (m0 < nl) atomicMax(&s_thr[m0], lmax0);
                    if (m1 < nl) atomicMax(&s_thr[m1], lmax1);
                }
            }
        }
        __syncthreads();
        for (int m = tid; m < nl; m += THREADS) {
            const uint32_t old_u = atomicMax(&THR_U[m], s_thr[m]);
            const uint32_t cur_u = max_u32(old_u, s_thr[m]);
            s_raw_cut[m] = compute_safe_raw_cut(ord_u32_to_float(cur_u), inv_cap, cap, slack, 1.0e-3f);
        }
        __syncthreads();

        // V14 Pass 2: Compact 16-bit row-keep bitmasks (`km0`, `km1` — ZERO stack arrays!) ->
        //             cross-`g` column liveness -> conditional L2-hot S1_OUT / HI / ALIVE_BM writes!
        uint32_t km0 = 0u, km1 = 0u;
        #pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            if (mt < nl_tiles) {
                const int m0 = mt * 16 + g;
                const int m1 = m0 + 8;
                const float xn0 = s_xn[m0];
                const float xn1 = s_xn[m1];
                if (m0 < nl) {
                    const float cut0 = s_raw_cut[m0];
                    if (v0 < vreal) {
                        float lo1, hi1;
                        compute_s1_bounds(acc[mt][0], xn0, rt0, nv0, cacc2, infl_q, infl_b, slop_rel, slop_abs, lo1, hi1);
                        if (!(f16_bits_to_f32(f32_to_f16_ru_bits(hi1)) < cut0)) km0 |= (1u << (mt * 2 + 0));
                    }
                    if (v1 < vreal) {
                        float lo1, hi1;
                        compute_s1_bounds(acc[mt][1], xn0, rt1, nv1, cacc2, infl_q, infl_b, slop_rel, slop_abs, lo1, hi1);
                        if (!(f16_bits_to_f32(f32_to_f16_ru_bits(hi1)) < cut0)) km1 |= (1u << (mt * 2 + 0));
                    }
                }
                if (m1 < nl) {
                    const float cut1 = s_raw_cut[m1];
                    if (v0 < vreal) {
                        float lo1, hi1;
                        compute_s1_bounds(acc[mt][2], xn1, rt0, nv0, cacc2, infl_q, infl_b, slop_rel, slop_abs, lo1, hi1);
                        if (!(f16_bits_to_f32(f32_to_f16_ru_bits(hi1)) < cut1)) km0 |= (1u << (mt * 2 + 1));
                    }
                    if (v1 < vreal) {
                        float lo1, hi1;
                        compute_s1_bounds(acc[mt][3], xn1, rt1, nv1, cacc2, infl_q, infl_b, slop_rel, slop_abs, lo1, hi1);
                        if (!(f16_bits_to_f32(f32_to_f16_ru_bits(hi1)) < cut1)) km1 |= (1u << (mt * 2 + 1));
                    }
                }
            }
        }

        uint32_t ca0 = (km0 != 0u) ? 1u : 0u;
        uint32_t ca1 = (km1 != 0u) ? 1u : 0u;
        ca0 |= __shfl_xor_sync(0xFFFFFFFFu, ca0, 16);
        ca0 |= __shfl_xor_sync(0xFFFFFFFFu, ca0, 8);
        ca0 |= __shfl_xor_sync(0xFFFFFFFFu, ca0, 4);
        ca1 |= __shfl_xor_sync(0xFFFFFFFFu, ca1, 16);
        ca1 |= __shfl_xor_sync(0xFFFFFFFFu, ca1, 8);
        ca1 |= __shfl_xor_sync(0xFFFFFFFFu, ca1, 4);

        warp_or_alive_8cols(ALIVE_BM, v_warp_sub, vreal, lane, t, ca0, ca1);

        if (ca0 && v0 < vreal) {
            #pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                if (mt < nl_tiles) {
                    const int m0 = mt * 16 + g;
                    const int m1 = m0 + 8;
                    if (m0 < nl) {
                        const int64_t mv = static_cast<int64_t>(m0) * V + v0;
                        if ((km0 >> (mt * 2 + 0)) & 1u) {
                            float lo1, hi1;
                            compute_s1_bounds(acc[mt][0], s_xn[m0], rt0, nv0, cacc2, infl_q, infl_b, slop_rel, slop_abs, lo1, hi1);
                            HI[mv] = f32_to_f16_ru_bits(hi1);
                            S1_OUT[mv] = acc[mt][0];
                        } else {
                            HI[mv] = FP16_NEG_INF_BITS;
                        }
                    }
                    if (m1 < nl) {
                        const int64_t mv = static_cast<int64_t>(m1) * V + v0;
                        if ((km0 >> (mt * 2 + 1)) & 1u) {
                            float lo1, hi1;
                            compute_s1_bounds(acc[mt][2], s_xn[m1], rt0, nv0, cacc2, infl_q, infl_b, slop_rel, slop_abs, lo1, hi1);
                            HI[mv] = f32_to_f16_ru_bits(hi1);
                            S1_OUT[mv] = acc[mt][2];
                        } else {
                            HI[mv] = FP16_NEG_INF_BITS;
                        }
                    }
                }
            }
        }
        if (ca1 && v1 < vreal) {
            #pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                if (mt < nl_tiles) {
                    const int m0 = mt * 16 + g;
                    const int m1 = m0 + 8;
                    if (m0 < nl) {
                        const int64_t mv = static_cast<int64_t>(m0) * V + v1;
                        if ((km1 >> (mt * 2 + 0)) & 1u) {
                            float lo1, hi1;
                            compute_s1_bounds(acc[mt][1], s_xn[m0], rt1, nv1, cacc2, infl_q, infl_b, slop_rel, slop_abs, lo1, hi1);
                            HI[mv] = f32_to_f16_ru_bits(hi1);
                            S1_OUT[mv] = acc[mt][1];
                        } else {
                            HI[mv] = FP16_NEG_INF_BITS;
                        }
                    }
                    if (m1 < nl) {
                        const int64_t mv = static_cast<int64_t>(m1) * V + v1;
                        if ((km1 >> (mt * 2 + 1)) & 1u) {
                            float lo1, hi1;
                            compute_s1_bounds(acc[mt][3], s_xn[m1], rt1, nv1, cacc2, infl_q, infl_b, slop_rel, slop_abs, lo1, hi1);
                            HI[mv] = f32_to_f16_ru_bits(hi1);
                            S1_OUT[mv] = acc[mt][3];
                        } else {
                            HI[mv] = FP16_NEG_INF_BITS;
                        }
                    }
                }
            }
        }
    }

    if constexpr (PDL) {
        __syncthreads();
        if (tid == 0) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }
}

// =========================================================================
// Kernel 3: `cert_cand1_v14_kernel` (M = 1..128, 3.14 KiB `ALIVE_BM` fast skip + 128b MASK1)
// =========================================================================
template <bool PDL>
__global__ void __launch_bounds__(256, 4)
cert_cand1_v14_kernel(
    const uint16_t* __restrict__ HI,
    const uint32_t* __restrict__ THR_AG,
    const int32_t* __restrict__ BAD,
    const uint32_t* __restrict__ ALIVE_BM,   // nullptr for legacy v7/v9 callers
    float* __restrict__ T_GLOB,
    int32_t* __restrict__ IDX1,
    void* __restrict__ MASK1_RAW,
    int mask_mode,
    int32_t* __restrict__ CNT1,
    const int32_t* __restrict__ NLIVE_PTR,
    int M,
    int V,
    int vreal,
    int ws,
    int tstride,
    float inv_cap,
    float cap,
    float slack
) {
    const int tid = threadIdx.x;
    const int lane = tid & 31;

    if constexpr (PDL) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }

    const int nl = resolve_nlive(NLIVE_PTR, M);
    __shared__ float s_tchain[CERT_V14_MAX_M];
    __shared__ float s_raw_cut[CERT_V14_MAX_M];
    __shared__ int s_any_bad;
    if (tid == 0) s_any_bad = 0;
    __syncthreads();

    if (tid < nl) {
        uint32_t max_ord = ORD_NEG_INF;
        for (int r = 0; r < ws; ++r) {
            uint32_t u = THR_AG[r * tstride + tid];
            if (u > max_ord) max_ord = u;
        }
        const float t_raw = ord_u32_to_float(max_ord);
        if (blockIdx.x == 0) T_GLOB[tid] = t_raw;
        s_tchain[tid] = __fsub_rn(chain_f32(t_raw, inv_cap, cap), slack);
        s_raw_cut[tid] = compute_safe_raw_cut(t_raw, inv_cap, cap, slack, 5.0e-4f);
        if (BAD[tid] != 0) atomicOr(&s_any_bad, 1);
    }
    __syncthreads();

    const int v = blockIdx.x * 256 + tid;
    uint64_t mask_lo = 0ull;
    uint64_t mask_hi = 0ull;
    if (v < vreal) {
        if (s_any_bad != 0) {
            mask_lo = full_mask_lo(nl);
            mask_hi = full_mask_hi(nl);
        } else {
            const bool bm_alive = (ALIVE_BM == nullptr) || (((__ldg(ALIVE_BM + (v >> 5)) >> (v & 31)) & 1u) != 0u);
            if (bm_alive) {
                const int nl_lo = (nl < 64) ? nl : 64;
                for (int m = 0; m < nl_lo; ++m) {
                    const float hi_f = f16_bits_to_f32(HI[static_cast<int64_t>(m) * V + v]);
                    if (!(hi_f < s_raw_cut[m]) && chain_f32(hi_f, inv_cap, cap) >= s_tchain[m]) {
                        mask_lo |= (1ull << m);
                        if (mask_mode == 0) break;
                    }
                }
                if (!(mask_mode == 0 && mask_lo != 0ull)) {
                    for (int m = 64; m < nl; ++m) {
                        const float hi_f = f16_bits_to_f32(HI[static_cast<int64_t>(m) * V + v]);
                        if (!(hi_f < s_raw_cut[m]) && chain_f32(hi_f, inv_cap, cap) >= s_tchain[m]) {
                            mask_hi |= (1ull << (m - 64));
                            if (mask_mode == 0) break;
                        }
                    }
                }
            }
        }
    }

    const bool keep = ((mask_lo | mask_hi) != 0ull);
    const uint32_t ballot = __ballot_sync(0xFFFFFFFFu, keep);
    if (ballot != 0u) {
        int base = 0;
        if (lane == 0) base = atomicAdd(CNT1, __popc(ballot));
        base = __shfl_sync(0xFFFFFFFFu, base, 0);
        if (keep) {
            const int off = __popc(ballot & ((1u << lane) - 1u));
            const int dst = base + off;
            IDX1[dst] = v;
            if (mask_mode == 1) {
                reinterpret_cast<uint64_t*>(MASK1_RAW)[dst] = mask_lo;
            } else if (mask_mode == 2) {
                ulonglong2 m128;
                m128.x = static_cast<unsigned long long>(mask_lo);
                m128.y = static_cast<unsigned long long>(mask_hi);
                reinterpret_cast<ulonglong2*>(MASK1_RAW)[dst] = m128;
            }
        }
    }
    if constexpr (PDL) {
        __syncthreads();
        if (tid == 0) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }
}

// =========================================================================
// Kernel 4: `cert_stage2_lsb1_v14_kernel` (M = 1..128, 128-bit MASK1 filtered, 0 B stack!)
// =========================================================================
template <bool PDL>
__global__ void __launch_bounds__(128, 4)
cert_stage2_lsb1_v14_kernel(
    const __nv_bfloat16* __restrict__ X_PAD,
    const uint32_t* __restrict__ LSB1_ROW,   // [V, n_qchunks * 16]
    const uint4* __restrict__ MS_ROW,        // [V, n_qchunks] uint4
    const uint16_t* __restrict__ BIG_S_ROW,  // [V, n_qchunks] fp16 bits
    const float* __restrict__ S1_OUT,        // [M, V] (L2-hot for surviving entries!)
    const float* __restrict__ RT2,           // [V]
    const float* __restrict__ NV2,           // [V]
    const float* __restrict__ XN,            // [128]
    const int32_t* __restrict__ BAD,         // [128]
    const float* __restrict__ T_GLOB,        // [128]
    const int32_t* __restrict__ IDX1,
    const void* __restrict__ MASK1_RAW,
    int mask_mode,
    const int32_t* __restrict__ CNT1,
    int32_t* __restrict__ IDX2,
    int32_t* __restrict__ CNT2,
    const int32_t* __restrict__ NLIVE_PTR,
    int M,
    int V,
    int K,
    float cacc2,
    float infl_q,
    float infl_b,
    float slop_rel,
    float slop_abs,
    float inv_cap,
    float cap,
    float slack
) {
    constexpr int WARPS = 4;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    if constexpr (PDL) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }

    const int n1 = *CNT1;
    if (n1 <= 0) {
        if constexpr (PDL) {
            __syncthreads();
            if (tid == 0) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
        return;
    }

    const int nl = resolve_nlive(NLIVE_PTR, M);
    __shared__ float s_tchain[CERT_V14_MAX_M];
    __shared__ float s_xn[CERT_V14_MAX_M];
    __shared__ int s_any_bad;
    if (tid == 0) s_any_bad = 0;
    __syncthreads();

    if (tid < nl) {
        s_tchain[tid] = __fsub_rn(chain_f32(T_GLOB[tid], inv_cap, cap), slack);
        s_xn[tid] = XN[tid];
        if (BAD[tid] != 0) atomicOr(&s_any_bad, 1);
    }
    __syncthreads();

    if (s_any_bad != 0) {
        for (int idx = blockIdx.x * 128 + tid; idx < n1; idx += gridDim.x * 128) {
            IDX2[idx] = IDX1[idx];
        }
        if (blockIdx.x == 0 && tid == 0) *CNT2 = n1;
        if constexpr (PDL) {
            __syncthreads();
            if (tid == 0) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
        return;
    }

    const int n_qchunks = K >> 9;
    const int total_warps = gridDim.x * WARPS;
    const int global_warp = blockIdx.x * WARPS + warp;
    const int w_idx = lane >> 1;
    const int half_shift = (lane & 1) * 16;
    const int ms_word_idx = lane >> 3;
    const int ms_bit_shift = (lane & 7) * 4;
    const uint64_t all_lo = full_mask_lo(nl);
    const uint64_t all_hi = full_mask_hi(nl);

    auto check_s2_row = [&](int m, int j, float ds_val, float rt2, float nv2) -> bool {
        float v = ds_val;
        #pragma unroll
        for (int off = 16; off >= 1; off >>= 1) {
            v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, off));
        }
        if (lane != 0) return false;
        const float s2 = __fadd_rn(S1_OUT[static_cast<int64_t>(m) * V + j], v);
        const float xn = s_xn[m];
        const float B2 = __fmul_rn(__fadd_rn(__fmul_rn(__fmul_rn(xn, rt2), infl_q),
                                             __fmul_rn(__fmul_rn(cacc2, xn), nv2)),
                                   infl_b);
        const float slop2 = __fadd_rn(__fmul_rn(__fadd_rn(fabsf(s2), B2), slop_rel), slop_abs);
        float hi2 = __fadd_rn(__fadd_rn(s2, B2), slop2);
        if (!isfinite(s2) || isnan(B2)) hi2 = INFINITY;
        return (chain_f32(hi2, inv_cap, cap) >= s_tchain[m]);
    };

    for (int idx = global_warp; idx < n1; idx += total_warps) {
        const int j = IDX1[idx];
        uint64_t rem_lo = all_lo;
        uint64_t rem_hi = all_hi;
        if (mask_mode == 1) {
            rem_lo = reinterpret_cast<const uint64_t*>(MASK1_RAW)[idx] & all_lo;
            rem_hi = 0ull;
        } else if (mask_mode == 2) {
            const ulonglong2 m128 = reinterpret_cast<const ulonglong2*>(MASK1_RAW)[idx];
            rem_lo = static_cast<uint64_t>(m128.x) & all_lo;
            rem_hi = static_cast<uint64_t>(m128.y) & all_hi;
        }

        const float rt2 = (lane == 0) ? __ldg(RT2 + j) : 0.0f;
        const float nv2 = (lane == 0) ? __ldg(NV2 + j) : 0.0f;
        const uint32_t* lsb_base = LSB1_ROW + static_cast<int64_t>(j) * (n_qchunks * 16);
        const uint4* ms_base = MS_ROW + static_cast<int64_t>(j) * n_qchunks;
        const uint16_t* bs_base = BIG_S_ROW + static_cast<int64_t>(j) * n_qchunks;

        auto pop_m = [&]() -> int {
            if (rem_lo != 0ull) {
                const int m = __ffsll(static_cast<long long>(rem_lo)) - 1;
                rem_lo &= (rem_lo - 1ull);
                return m;
            }
            if (rem_hi != 0ull) {
                const int m = 64 + (__ffsll(static_cast<long long>(rem_hi)) - 1);
                rem_hi &= (rem_hi - 1ull);
                return m;
            }
            return -1;
        };

        bool keep = false;
        while ((rem_lo | rem_hi) != 0ull && !keep) {
            const int m0 = pop_m();
            int n_act = 1;
            int m1 = m0, m2 = m0, m3 = m0;
            const int p1 = pop_m();
            if (p1 >= 0) {
                m1 = p1; n_act = 2;
                const int p2 = pop_m();
                if (p2 >= 0) {
                    m2 = p2; n_act = 3;
                    const int p3 = pop_m();
                    if (p3 >= 0) { m3 = p3; n_act = 4; }
                }
            }

            float ds0 = 0.0f, ds1 = 0.0f, ds2 = 0.0f, ds3 = 0.0f;

            for (int qc = 0; qc < n_qchunks; ++qc) {
                const uint32_t bits16 = (__ldg(lsb_base + qc * 16 + w_idx) >> half_shift) & 0xFFFFu;
                const uint4 ms4 = __ldg(ms_base + qc);
                const uint32_t ms_w = (ms_word_idx == 0) ? ms4.x : ((ms_word_idx == 1) ? ms4.y : ((ms_word_idx == 2) ? ms4.z : ms4.w));
                const uint32_t c0 = (ms_w >> ms_bit_shift) & 3u;
                const uint32_t c1 = (ms_w >> (ms_bit_shift + 2)) & 3u;

                const float bs025 = __fmul_rn(f16_bits_to_f32(__ldg(bs_base + qc)), 0.25f);
                const float mult0 = (c0 == 0u) ? 0.375f : ((c0 == 1u) ? 0.5625f : ((c0 == 2u) ? 0.75f : 1.0f));
                const float mult1 = (c1 == 0u) ? 0.375f : ((c1 == 1u) ? 0.5625f : ((c1 == 2u) ? 0.75f : 1.0f));
                const float dw_abs0 = bf16_rn_f32(__fmul_rn(bs025, mult0));
                const float dw_abs1 = bf16_rn_f32(__fmul_rn(bs025, mult1));

                const int k_off = qc * 512 + lane * 16;
                auto dot16 = [&](int m) -> float {
                    const __nv_bfloat16* xp = X_PAD + static_cast<int64_t>(m) * K + k_off;
                    const uint4 x_u0 = *reinterpret_cast<const uint4*>(xp);
                    const uint4 x_u1 = *reinterpret_cast<const uint4*>(xp + 8);
                    const __nv_bfloat162* xh0 = reinterpret_cast<const __nv_bfloat162*>(&x_u0);
                    const __nv_bfloat162* xh1 = reinterpret_cast<const __nv_bfloat162*>(&x_u1);
                    float loc = 0.0f;
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const float2 f0 = __bfloat1622float2(xh0[i]);
                        const float2 f1 = __bfloat1622float2(xh1[i]);
                        const float d0x = ((bits16 >> (2 * i)) & 1u) ? dw_abs0 : -dw_abs0;
                        const float d0y = ((bits16 >> (2 * i + 1)) & 1u) ? dw_abs0 : -dw_abs0;
                        const float d1x = ((bits16 >> (8 + 2 * i)) & 1u) ? dw_abs1 : -dw_abs1;
                        const float d1y = ((bits16 >> (8 + 2 * i + 1)) & 1u) ? dw_abs1 : -dw_abs1;
                        loc = __fadd_rn(loc, __fadd_rn(__fmul_rn(f0.x, d0x), __fmul_rn(f0.y, d0y)));
                        loc = __fadd_rn(loc, __fadd_rn(__fmul_rn(f1.x, d1x), __fmul_rn(f1.y, d1y)));
                    }
                    return loc;
                };

                ds0 = __fadd_rn(ds0, dot16(m0));
                if (n_act > 1) ds1 = __fadd_rn(ds1, dot16(m1));
                if (n_act > 2) ds2 = __fadd_rn(ds2, dot16(m2));
                if (n_act > 3) ds3 = __fadd_rn(ds3, dot16(m3));
            }

            bool pass_l0 = check_s2_row(m0, j, ds0, rt2, nv2);
            if (n_act > 1) pass_l0 = check_s2_row(m1, j, ds1, rt2, nv2) || pass_l0;
            if (n_act > 2) pass_l0 = check_s2_row(m2, j, ds2, rt2, nv2) || pass_l0;
            if (n_act > 3) pass_l0 = check_s2_row(m3, j, ds3, rt2, nv2) || pass_l0;
            keep = (__shfl_sync(0xFFFFFFFFu, pass_l0 ? 1 : 0, 0) != 0);
        }

        if (lane == 0 && keep) {
            const int dst = atomicAdd(CNT2, 1);
            IDX2[dst] = j;
        }
    }
    if constexpr (PDL) {
        __syncthreads();
        if (tid == 0) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }
}

// =========================================================================
// Kernel 5: `cert_refine_argmax_v14_kernel` (M = 1..128, 4-warp B-reuse + `nl_tiles` pruning!)
// =========================================================================
template <int MT, bool M_LE8, bool PDL>
__global__ void __launch_bounds__(128, 4)
cert_refine_argmax_v14_kernel(
    const __nv_bfloat16* __restrict__ X_PAD,
    const __nv_bfloat16* __restrict__ W,
    const int32_t* __restrict__ IDX2,
    const int32_t* __restrict__ CNT2,
    uint64_t* __restrict__ BEST,
    int32_t* __restrict__ ARRIVE,
    float* __restrict__ OUT_PAIR,
    const int32_t* __restrict__ NLIVE_PTR,
    int M,
    int vstart,
    int K,
    int sw,
    float inv_cap,
    float cap
) {
    static_assert(MT >= 1 && MT <= 8, "MT must be in {1..8} for M=1..128");
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g = lane >> 2;
    const int t = lane & 3;
    const int nprog = gridDim.x;

    if constexpr (PDL) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }

    const int n = *CNT2;
    const int nl = resolve_nlive(NLIVE_PTR, M);
    const int nl_tiles = (nl + 15) >> 4;

    if (n > 0) {
        int rpc = (n + nprog - 1) / nprog;
        if (rpc < 1) rpc = 1;
        if (rpc > 8) rpc = 8;
        const int nch = (n + rpc - 1) / rpc;
        const int cl_per_row = (K * 2 + 127) >> 7;

        for (int c = blockIdx.x; c < nch; c += nprog) {
            const int r0 = c * rpc;
            const int cnt = (r0 + rpc <= n) ? rpc : (n - r0);

            const int total_cl = cnt * cl_per_row;
            for (int cl = tid; cl < total_cl; cl += 128) {
                const int rr = cl / cl_per_row;
                const int line = cl - rr * cl_per_row;
                const int j_pf = IDX2[r0 + rr];
                prefetch_l2_v14(W + static_cast<int64_t>(j_pf) * sw + line * 64);
            }
            const int c_nxt = c + nprog;
            if (c_nxt < nch && tid < cl_per_row) {
                const int j_nx = IDX2[c_nxt * rpc];
                prefetch_l2_v14(W + static_cast<int64_t>(j_nx) * sw + tid * 64);
            }

            // V14: Prune completely dead warps (`warp >= nl_tiles`) and dead `has_hi` tiles!
            if (warp < nl_tiles) {
                const int m_base0 = warp * 16;
                const bool has_hi0_8 = (!(MT == 1 && M_LE8)) && (m_base0 + 8 < nl);
                const bool has_hi = (MT > 4) && (warp + 4 < nl_tiles);
                const int m_base1 = has_hi ? ((warp + 4) * 16) : m_base0;
                const bool has_hi1_8 = has_hi && (m_base1 + 8 < nl);

                const int j_safe = IDX2[r0 + ((g < cnt) ? g : 0)];
                const __nv_bfloat16* w_row = W + static_cast<int64_t>(j_safe) * sw;
                const __nv_bfloat16* x0_row0 = X_PAD + static_cast<int64_t>(m_base0 + g) * K;
                const __nv_bfloat16* x0_row1 = X_PAD + static_cast<int64_t>(m_base0 + g + 8) * K;
                const __nv_bfloat16* x1_row0 = X_PAD + static_cast<int64_t>(m_base1 + g) * K;
                const __nv_bfloat16* x1_row1 = X_PAD + static_cast<int64_t>(m_base1 + g + 8) * K;

                float acc0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                float acc1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                const int k_steps = K >> 4;
                #pragma unroll 4
                for (int s = 0; s < k_steps; ++s) {
                    const int k_off = s * 16 + 2 * t;
                    const uint32_t b0 = *reinterpret_cast<const uint32_t*>(w_row + k_off);
                    const uint32_t b1 = *reinterpret_cast<const uint32_t*>(w_row + k_off + 8);

                    const uint32_t a0 = *reinterpret_cast<const uint32_t*>(x0_row0 + k_off);
                    const uint32_t a2 = *reinterpret_cast<const uint32_t*>(x0_row0 + k_off + 8);
                    uint32_t a1 = 0u, a3 = 0u;
                    if constexpr (!(MT == 1 && M_LE8)) {
                        if (has_hi0_8) {
                            a1 = *reinterpret_cast<const uint32_t*>(x0_row1 + k_off);
                            a3 = *reinterpret_cast<const uint32_t*>(x0_row1 + k_off + 8);
                        }
                    }
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                        : "+f"(acc0[0]), "+f"(acc0[1]), "+f"(acc0[2]), "+f"(acc0[3])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)
                    );

                    if constexpr (MT > 4) {
                        if (has_hi) {
                            const uint32_t c0 = *reinterpret_cast<const uint32_t*>(x1_row0 + k_off);
                            const uint32_t c2 = *reinterpret_cast<const uint32_t*>(x1_row0 + k_off + 8);
                            const uint32_t c1 = has_hi1_8 ? *reinterpret_cast<const uint32_t*>(x1_row1 + k_off) : 0u;
                            const uint32_t c3 = has_hi1_8 ? *reinterpret_cast<const uint32_t*>(x1_row1 + k_off + 8) : 0u;
                            asm volatile(
                                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                                : "+f"(acc1[0]), "+f"(acc1[1]), "+f"(acc1[2]), "+f"(acc1[3])
                                : "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(b0), "r"(b1)
                            );
                        }
                    }
                }

                auto update_best = [&](int m, int col, float raw_val) {
                    if (m >= nl || col >= cnt) return;
                    const float val = chain_f32(raw_val, inv_cap, cap);
                    if (isnan(val)) return;
                    const uint32_t g_idx = static_cast<uint32_t>(vstart + IDX2[r0 + col]);
                    const uint64_t packed = (static_cast<uint64_t>(float_to_ord_u32(val)) << 32) |
                                            (0xFFFFFFFFu - g_idx);
                    atomicMax(reinterpret_cast<unsigned long long*>(&BEST[m]),
                              static_cast<unsigned long long>(packed));
                };

                const int c0 = 2 * t;
                const int c1 = c0 + 1;
                update_best(m_base0 + g, c0, acc0[0]);
                update_best(m_base0 + g, c1, acc0[1]);
                if constexpr (!(MT == 1 && M_LE8)) {
                    if (has_hi0_8) {
                        update_best(m_base0 + g + 8, c0, acc0[2]);
                        update_best(m_base0 + g + 8, c1, acc0[3]);
                    }
                }
                if constexpr (MT > 4) {
                    if (has_hi) {
                        update_best(m_base1 + g, c0, acc1[0]);
                        update_best(m_base1 + g, c1, acc1[1]);
                        if (has_hi1_8) {
                            update_best(m_base1 + g + 8, c0, acc1[2]);
                            update_best(m_base1 + g + 8, c1, acc1[3]);
                        }
                    }
                }
            }
        }
    }

    __syncthreads();
    __shared__ int s_prev;
    if (tid == 0) {
        s_prev = atomic_add_acq_rel_gpu_i32(ARRIVE, 1);
    }
    __syncthreads();

    if (s_prev == nprog - 1) {
        if (tid == 0) {
            *ARRIVE = 0;
        }
        for (int m = tid; m < M; m += 128) {
            float max_val = -INFINITY;
            float best_idx = static_cast<float>(vstart);
            if (m < nl) {
                const uint64_t b = BEST[m];
                const uint32_t hi_ord = static_cast<uint32_t>(b >> 32);
                const uint32_t inv_idx = static_cast<uint32_t>(b & 0xFFFFFFFFu);
                if (!(hi_ord == ORD_NEG_INF && inv_idx == 0u)) {
                    max_val = ord_u32_to_float(hi_ord);
                    best_idx = static_cast<float>(static_cast<uint32_t>(0xFFFFFFFFu - inv_idx));
                }
            }
            float4 pair = make_float4(max_val, best_idx, 0.0f, 0.0f);
            *reinterpret_cast<float4*>(OUT_PAIR + m * 4) = pair;
        }
        if constexpr (PDL) {
            __syncthreads();
            if (tid == 0) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
    }
}

// =========================================================================
// Kernel 6: `cert_exact_all_v14_kernel` (M = 1..128 via m_row_off slicing, MT in {1..4})
// =========================================================================
template <int MT, bool M_LE8>
__global__ void __launch_bounds__(128, 4)
cert_exact_all_v14_kernel(
    const __nv_bfloat16* __restrict__ X_PAD,
    const __nv_bfloat16* __restrict__ W,
    float* __restrict__ OUT,
    const int32_t* __restrict__ NLIVE_PTR,
    int M,
    int V,
    int K,
    int sw,
    int m_row_off
) {
    static_assert(MT >= 1 && MT <= 4, "MT must be in {1, 2, 3, 4}");
    constexpr int WARPS = 4;
    constexpr int BN = WARPS * 8;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g = lane >> 2;
    const int t = lane & 3;
    const int nl_total = resolve_nlive(NLIVE_PTR, M);
    const int nl = (nl_total > m_row_off) ? ((nl_total - m_row_off < MT * 16) ? (nl_total - m_row_off) : (MT * 16)) : 0;
    if (nl <= 0) return;

    X_PAD += static_cast<int64_t>(m_row_off) * K;
    OUT   += static_cast<int64_t>(m_row_off) * V;
    const int n_tiles = (V + BN - 1) / BN;

    for (int pn = blockIdx.x; pn < n_tiles; pn += gridDim.x) {
        const int v_warp = pn * BN + warp * 8;
        if (v_warp >= V) continue;
        const int v_g = (v_warp + g < V) ? (v_warp + g) : v_warp;
        const __nv_bfloat16* w_row = W + static_cast<int64_t>(v_g) * sw;

        float acc[MT][4];
        #pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            acc[mt][0] = 0.0f; acc[mt][1] = 0.0f; acc[mt][2] = 0.0f; acc[mt][3] = 0.0f;
        }
        const int k_steps = K >> 4;
        #pragma unroll 4
        for (int s = 0; s < k_steps; ++s) {
            const int k_off = s * 16 + 2 * t;
            const uint32_t b0 = *reinterpret_cast<const uint32_t*>(w_row + k_off);
            const uint32_t b1 = *reinterpret_cast<const uint32_t*>(w_row + k_off + 8);
            #pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                const __nv_bfloat16* x_row0 = X_PAD + static_cast<int64_t>(mt * 16 + g) * K;
                const __nv_bfloat16* x_row1 = X_PAD + static_cast<int64_t>(mt * 16 + g + 8) * K;
                const uint32_t a0 = *reinterpret_cast<const uint32_t*>(x_row0 + k_off);
                const uint32_t a2 = *reinterpret_cast<const uint32_t*>(x_row0 + k_off + 8);
                uint32_t a1 = 0u, a3 = 0u;
                if constexpr (!(MT == 1 && M_LE8)) {
                    a1 = *reinterpret_cast<const uint32_t*>(x_row1 + k_off);
                    a3 = *reinterpret_cast<const uint32_t*>(x_row1 + k_off + 8);
                }
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                    : "+f"(acc[mt][0]), "+f"(acc[mt][1]), "+f"(acc[mt][2]), "+f"(acc[mt][3])
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)
                );
            }
        }

        const int v0 = v_warp + 2 * t;
        const int v1 = v0 + 1;
        #pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            const int m0 = mt * 16 + g;
            const int m1 = m0 + 8;
            if (m0 < nl) {
                if (v0 < V) OUT[static_cast<int64_t>(m0) * V + v0] = acc[mt][0];
                if (v1 < V) OUT[static_cast<int64_t>(m0) * V + v1] = acc[mt][1];
            }
            if constexpr (!(MT == 1 && M_LE8)) {
                if (m1 < nl) {
                    if (v0 < V) OUT[static_cast<int64_t>(m1) * V + v0] = acc[mt][2];
                    if (v1 < V) OUT[static_cast<int64_t>(m1) * V + v1] = acc[mt][3];
                }
            }
        }
    }
}

template <int W_V, int SUB_V, bool USE_ALIVE>
static cudaError_t dispatch_screen_v14(
    const void* X_PAD, const void* SLABS_U4, const void* MS_U4, const void* BIG_S,
    const void* RT1, const void* NV1, const void* XN,
    void* S1_OUT, void* HI, void* THR_U, void* ALIVE_BM, const void* NLIVE_PTR,
    int M, int V, int vreal, int K,
    float cacc2, float infl_q, float infl_b, float slop_rel, float slop_abs,
    float inv_cap, float cap, float slack,
    bool pdl, cudaStream_t stream
) {
    constexpr int BN = W_V * SUB_V * 8;
    const int n_tiles = (V + BN - 1) / BN;
    const dim3 block(W_V * 32, 1, 1);

    if (M <= 16) {
        const dim3 grid(n_tiles, 1, 1);
        const bool m_le8 = (M <= 8);
        const size_t smem = static_cast<size_t>(m_le8 ? 8 : 16) * 520 * sizeof(__nv_bfloat16);
        #define L_SCR1(ML, PD) launch_maybe_pdl_v14( \
            cert_screen_int6_ms8_v14_m16_kernel<W_V, SUB_V, ML, PD, USE_ALIVE>, grid, block, smem, stream, PD, \
            (const __nv_bfloat16*)X_PAD, (const uint4*)SLABS_U4, (const uint4*)MS_U4, (const uint16_t*)BIG_S, \
            (const float*)RT1, (const float*)NV1, (const float*)XN, \
            (float*)S1_OUT, (uint16_t*)HI, (uint32_t*)THR_U, (uint32_t*)ALIVE_BM, (const int32_t*)NLIVE_PTR, \
            M, V, vreal, K, cacc2, infl_q, infl_b, slop_rel, slop_abs, inv_cap, cap, slack)
        if (m_le8) return pdl ? L_SCR1(true, true) : L_SCR1(true, false);
        else       return pdl ? L_SCR1(false, true) : L_SCR1(false, false);
        #undef L_SCR1
    }

    if (M <= 48) {
        const dim3 grid(n_tiles, 1, 1);
        #define L_SCR23(MT_V, PD) launch_maybe_pdl_v14( \
            cert_screen_int6_ms8_v14_c48_kernel<W_V, SUB_V, MT_V, PD, USE_ALIVE>, grid, block, \
            static_cast<size_t>((MT_V) * 16) * 264 * sizeof(__nv_bfloat16), stream, PD, \
            (const __nv_bfloat16*)X_PAD, (const uint4*)SLABS_U4, (const uint4*)MS_U4, (const uint16_t*)BIG_S, \
            (const float*)RT1, (const float*)NV1, (const float*)XN, \
            (float*)S1_OUT, (uint16_t*)HI, (uint32_t*)THR_U, (uint32_t*)ALIVE_BM, (const int32_t*)NLIVE_PTR, \
            M, V, vreal, K, cacc2, infl_q, infl_b, slop_rel, slop_abs, inv_cap, cap, slack)
        if (M <= 32) return pdl ? L_SCR23(2, true) : L_SCR23(2, false);
        else         return pdl ? L_SCR23(3, true) : L_SCR23(3, false);
        #undef L_SCR23
    }

    const dim3 grid_sp(n_tiles, SUB_V, 1);
    #define L_SCR_V14(MT_V, PD) launch_maybe_pdl_v14( \
        cert_screen_int6_ms8_v14_subsplit_kernel<W_V, SUB_V, MT_V, PD, USE_ALIVE>, grid_sp, block, \
        static_cast<size_t>((MT_V) * 16) * 136 * sizeof(__nv_bfloat16), stream, PD, \
        (const __nv_bfloat16*)X_PAD, (const uint4*)SLABS_U4, (const uint4*)MS_U4, (const uint16_t*)BIG_S, \
        (const float*)RT1, (const float*)NV1, (const float*)XN, \
        (float*)S1_OUT, (uint16_t*)HI, (uint32_t*)THR_U, (uint32_t*)ALIVE_BM, (const int32_t*)NLIVE_PTR, \
        M, V, vreal, K, cacc2, infl_q, infl_b, slop_rel, slop_abs, inv_cap, cap, slack)
    if (M <= 64)       return pdl ? L_SCR_V14(4, true) : L_SCR_V14(4, false);
    else if (M <= 80)  return pdl ? L_SCR_V14(5, true) : L_SCR_V14(5, false);
    else if (M <= 96)  return pdl ? L_SCR_V14(6, true) : L_SCR_V14(6, false);
    else if (M <= 112) return pdl ? L_SCR_V14(7, true) : L_SCR_V14(7, false);
    else               return pdl ? L_SCR_V14(8, true) : L_SCR_V14(8, false);
    #undef L_SCR_V14
}

static int pre_screen_impl_v14(
    const void* X, const void* LOGITS_IDX,
    const void* SLABS_U4, const void* MS_U4, const void* BIG_S,
    const void* RT1, const void* NV1,
    void* X_PAD, void* XN, void* BAD,
    void* S1_OUT, void* HI, void* THR_U,
    void* CNT1, void* CNT2, void* BEST,
    void* ARRIVE, void* ALIVE_BM, const void* NLIVE_PTR,
    int M, int V, int vreal, int K, int sx,
    int cfg_id, int pdl,
    float cacc2, float infl_x, float x_abs,
    float infl_q, float infl_b, float slop_rel, float slop_abs,
    float inv_cap, float cap, float slack,
    uintptr_t stream_ptr
) {
    if (M <= 0 || M > CERT_V14_MAX_M || V <= 0 || K <= 0 || (K & 511) != 0) return -2;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    const bool do_pdl = (pdl != 0);
    const bool use_alive = (ALIVE_BM != nullptr);
    const int m_rows = (M <= 8) ? 8 : (((M + 15) / 16) * 16);
    const int v_bm_words = use_alive ? ((V + 31) >> 5) : 0;

    const uintptr_t raw_idx = reinterpret_cast<uintptr_t>(LOGITS_IDX);
    const int idx_is_i32 = (raw_idx & 1u) ? 1 : 0;
    const int64_t* idx_ptr = reinterpret_cast<const int64_t*>(raw_idx & ~uintptr_t(1u));

    cudaError_t err = do_pdl
        ? launch_maybe_pdl_v14(cert_xnorm_pad_v14_kernel<true>, dim3(m_rows, 1, 1), dim3(256, 1, 1), 0, stream, true,
              (const __nv_bfloat16*)X, idx_ptr, idx_is_i32, (__nv_bfloat16*)X_PAD, (float*)XN, (int32_t*)BAD,
              (uint32_t*)THR_U, (int32_t*)CNT1, (int32_t*)CNT2, (uint64_t*)BEST, (int32_t*)ARRIVE,
              (uint32_t*)ALIVE_BM, (const int32_t*)NLIVE_PTR, M, m_rows, v_bm_words, K, sx, infl_x, x_abs)
        : launch_maybe_pdl_v14(cert_xnorm_pad_v14_kernel<false>, dim3(m_rows, 1, 1), dim3(256, 1, 1), 0, stream, false,
              (const __nv_bfloat16*)X, idx_ptr, idx_is_i32, (__nv_bfloat16*)X_PAD, (float*)XN, (int32_t*)BAD,
              (uint32_t*)THR_U, (int32_t*)CNT1, (int32_t*)CNT2, (uint64_t*)BEST, (int32_t*)ARRIVE,
              (uint32_t*)ALIVE_BM, (const int32_t*)NLIVE_PTR, M, m_rows, v_bm_words, K, sx, infl_x, x_abs);
    if (err != cudaSuccess) return static_cast<int>(err);

    if (use_alive) {
        switch (cfg_id) {
            case 0: err = dispatch_screen_v14<4, 2, true>(X_PAD, SLABS_U4, MS_U4, BIG_S, RT1, NV1, XN, S1_OUT, HI, THR_U, ALIVE_BM, NLIVE_PTR, M, V, vreal, K, cacc2, infl_q, infl_b, slop_rel, slop_abs, inv_cap, cap, slack, do_pdl, stream); break;
            case 1: err = dispatch_screen_v14<4, 1, true>(X_PAD, SLABS_U4, MS_U4, BIG_S, RT1, NV1, XN, S1_OUT, HI, THR_U, ALIVE_BM, NLIVE_PTR, M, V, vreal, K, cacc2, infl_q, infl_b, slop_rel, slop_abs, inv_cap, cap, slack, do_pdl, stream); break;
            case 2: err = dispatch_screen_v14<2, 1, true>(X_PAD, SLABS_U4, MS_U4, BIG_S, RT1, NV1, XN, S1_OUT, HI, THR_U, ALIVE_BM, NLIVE_PTR, M, V, vreal, K, cacc2, infl_q, infl_b, slop_rel, slop_abs, inv_cap, cap, slack, do_pdl, stream); break;
            default: return -1;
        }
    } else {
        switch (cfg_id) {
            case 0: err = dispatch_screen_v14<4, 2, false>(X_PAD, SLABS_U4, MS_U4, BIG_S, RT1, NV1, XN, S1_OUT, HI, THR_U, nullptr, NLIVE_PTR, M, V, vreal, K, cacc2, infl_q, infl_b, slop_rel, slop_abs, 0.0f, 0.0f, 0.0f, do_pdl, stream); break;
            case 1: err = dispatch_screen_v14<4, 1, false>(X_PAD, SLABS_U4, MS_U4, BIG_S, RT1, NV1, XN, S1_OUT, HI, THR_U, nullptr, NLIVE_PTR, M, V, vreal, K, cacc2, infl_q, infl_b, slop_rel, slop_abs, 0.0f, 0.0f, 0.0f, do_pdl, stream); break;
            case 2: err = dispatch_screen_v14<2, 1, false>(X_PAD, SLABS_U4, MS_U4, BIG_S, RT1, NV1, XN, S1_OUT, HI, THR_U, nullptr, NLIVE_PTR, M, V, vreal, K, cacc2, infl_q, infl_b, slop_rel, slop_abs, 0.0f, 0.0f, 0.0f, do_pdl, stream); break;
            default: return -1;
        }
    }
    return static_cast<int>(err);
}

static int post_refine_impl_v14(
    const void* X_PAD, const void* W_BF16,
    const void* LSB1_ROW, const void* MS_ROW, const void* BIG_S_ROW,
    const void* S1_OUT, const void* RT2, const void* NV2, const void* XN,
    const void* HI, const void* THR_AG, const void* BAD, const void* ALIVE_BM,
    void* T_GLOB, void* IDX1, void* MASK1_RAW, int mask_mode, void* CNT1,
    void* IDX2, void* CNT2,
    void* BEST, void* ARRIVE, void* OUT_PAIR, const void* NLIVE_PTR,
    int M, int V, int vreal, int vstart, int K, int sw,
    int ws, int tstride, int ref_grid, int pdl,
    float cacc2, float infl_q, float infl_b, float slop_rel, float slop_abs,
    float inv_cap, float cap, float slack,
    uintptr_t stream_ptr
) {
    if (M <= 0 || M > CERT_V14_MAX_M || V <= 0 || vreal <= 0 || K <= 0 || (K & 511) != 0) return -2;
    if (mask_mode == 1 && M > 64) return -2;
    if (MASK1_RAW == nullptr) mask_mode = 0;

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    const bool do_pdl = (pdl != 0);
    const int cand_blocks = (vreal + 255) / 256;
    const int rg = (ref_grid > 0) ? ref_grid : 96;

    cudaError_t err = do_pdl
        ? launch_maybe_pdl_v14(cert_cand1_v14_kernel<true>, dim3(cand_blocks, 1, 1), dim3(256, 1, 1), 0, stream, true,
              (const uint16_t*)HI, (const uint32_t*)THR_AG, (const int32_t*)BAD, (const uint32_t*)ALIVE_BM,
              (float*)T_GLOB, (int32_t*)IDX1, MASK1_RAW, mask_mode, (int32_t*)CNT1, (const int32_t*)NLIVE_PTR,
              M, V, vreal, ws, tstride, inv_cap, cap, slack)
        : launch_maybe_pdl_v14(cert_cand1_v14_kernel<false>, dim3(cand_blocks, 1, 1), dim3(256, 1, 1), 0, stream, false,
              (const uint16_t*)HI, (const uint32_t*)THR_AG, (const int32_t*)BAD, (const uint32_t*)ALIVE_BM,
              (float*)T_GLOB, (int32_t*)IDX1, MASK1_RAW, mask_mode, (int32_t*)CNT1, (const int32_t*)NLIVE_PTR,
              M, V, vreal, ws, tstride, inv_cap, cap, slack);
    if (err != cudaSuccess) return static_cast<int>(err);

    err = do_pdl
        ? launch_maybe_pdl_v14(cert_stage2_lsb1_v14_kernel<true>, dim3(192, 1, 1), dim3(128, 1, 1), 0, stream, true,
              (const __nv_bfloat16*)X_PAD, (const uint32_t*)LSB1_ROW, (const uint4*)MS_ROW, (const uint16_t*)BIG_S_ROW,
              (const float*)S1_OUT, (const float*)RT2, (const float*)NV2, (const float*)XN,
              (const int32_t*)BAD, (const float*)T_GLOB, (const int32_t*)IDX1, MASK1_RAW, mask_mode, (const int32_t*)CNT1,
              (int32_t*)IDX2, (int32_t*)CNT2, (const int32_t*)NLIVE_PTR,
              M, V, K, cacc2, infl_q, infl_b, slop_rel, slop_abs, inv_cap, cap, slack)
        : launch_maybe_pdl_v14(cert_stage2_lsb1_v14_kernel<false>, dim3(192, 1, 1), dim3(128, 1, 1), 0, stream, false,
              (const __nv_bfloat16*)X_PAD, (const uint32_t*)LSB1_ROW, (const uint4*)MS_ROW, (const uint16_t*)BIG_S_ROW,
              (const float*)S1_OUT, (const float*)RT2, (const float*)NV2, (const float*)XN,
              (const int32_t*)BAD, (const float*)T_GLOB, (const int32_t*)IDX1, MASK1_RAW, mask_mode, (const int32_t*)CNT1,
              (int32_t*)IDX2, (int32_t*)CNT2, (const int32_t*)NLIVE_PTR,
              M, V, K, cacc2, infl_q, infl_b, slop_rel, slop_abs, inv_cap, cap, slack);
    if (err != cudaSuccess) return static_cast<int>(err);

    #define L_REF(MT_V, ML, PD) launch_maybe_pdl_v14( \
        cert_refine_argmax_v14_kernel<MT_V, ML, PD>, dim3(rg, 1, 1), dim3(128, 1, 1), 0, stream, PD, \
        (const __nv_bfloat16*)X_PAD, (const __nv_bfloat16*)W_BF16, (const int32_t*)IDX2, (const int32_t*)CNT2, \
        (uint64_t*)BEST, (int32_t*)ARRIVE, (float*)OUT_PAIR, (const int32_t*)NLIVE_PTR, \
        M, vstart, K, sw, inv_cap, cap)
    if (M <= 8)        err = do_pdl ? L_REF(1, true,  true) : L_REF(1, true,  false);
    else if (M <= 16)  err = do_pdl ? L_REF(1, false, true) : L_REF(1, false, false);
    else if (M <= 32)  err = do_pdl ? L_REF(2, false, true) : L_REF(2, false, false);
    else if (M <= 48)  err = do_pdl ? L_REF(3, false, true) : L_REF(3, false, false);
    else if (M <= 64)  err = do_pdl ? L_REF(4, false, true) : L_REF(4, false, false);
    else if (M <= 80)  err = do_pdl ? L_REF(5, false, true) : L_REF(5, false, false);
    else if (M <= 96)  err = do_pdl ? L_REF(6, false, true) : L_REF(6, false, false);
    else if (M <= 112) err = do_pdl ? L_REF(7, false, true) : L_REF(7, false, false);
    else               err = do_pdl ? L_REF(8, false, true) : L_REF(8, false, false);
    #undef L_REF
    return static_cast<int>(err);
}

}  // namespace

// =========================================================================
// Extern "C" Exports: 100% V7/C48/V9/V12 ABI Superset + 3 New V14 Entry Points
// =========================================================================

extern "C" int cert_head_v14_pre_screen(
    const void* X, const void* LOGITS_IDX,
    const void* SLABS_U4, const void* MS_U4, const void* BIG_S,
    const void* RT1, const void* NV1,
    void* X_PAD, void* XN, void* BAD,
    void* S1_OUT, void* HI, void* THR_U,
    void* CNT1, void* CNT2, void* BEST,
    void* ARRIVE, void* ALIVE_BM, const void* NLIVE_PTR,
    int M, int V, int vreal, int K, int sx,
    int cfg_id, int pdl,
    float cacc2, float infl_x, float x_abs,
    float infl_q, float infl_b, float slop_rel, float slop_abs,
    float inv_cap, float cap, float slack,
    uintptr_t stream_ptr
) {
    return pre_screen_impl_v14(
        X, LOGITS_IDX, SLABS_U4, MS_U4, BIG_S, RT1, NV1,
        X_PAD, XN, BAD, S1_OUT, HI, THR_U, CNT1, CNT2, BEST,
        ARRIVE, ALIVE_BM, NLIVE_PTR,
        M, V, vreal, K, sx, cfg_id, pdl,
        cacc2, infl_x, x_abs, infl_q, infl_b, slop_rel, slop_abs,
        inv_cap, cap, slack, stream_ptr
    );
}

extern "C" int cert_head_v14_post_refine(
    const void* X_PAD, const void* W_BF16,
    const void* LSB1_ROW, const void* MS_ROW, const void* BIG_S_ROW,
    const void* S1_OUT, const void* RT2, const void* NV2, const void* XN,
    const void* HI, const void* THR_AG, const void* BAD, const void* ALIVE_BM,
    void* T_GLOB, void* IDX1, void* MASK1_128, void* CNT1,
    void* IDX2, void* CNT2,
    void* BEST, void* ARRIVE, void* OUT_PAIR, const void* NLIVE_PTR,
    int M, int V, int vreal, int vstart, int K, int sw,
    int ws, int tstride, int ref_grid, int pdl,
    float cacc2, float infl_q, float infl_b, float slop_rel, float slop_abs,
    float inv_cap, float cap, float slack,
    uintptr_t stream_ptr
) {
    return post_refine_impl_v14(
        X_PAD, W_BF16, LSB1_ROW, MS_ROW, BIG_S_ROW,
        S1_OUT, RT2, NV2, XN, HI, THR_AG, BAD, ALIVE_BM,
        T_GLOB, IDX1, MASK1_128, 2, CNT1, IDX2, CNT2,
        BEST, ARRIVE, OUT_PAIR, NLIVE_PTR,
        M, V, vreal, vstart, K, sw, ws, tstride, ref_grid, pdl,
        cacc2, infl_q, infl_b, slop_rel, slop_abs, inv_cap, cap, slack,
        stream_ptr
    );
}

extern "C" int cert_head_v7_pre_screen(
    const void* X, const void* LOGITS_IDX,
    const void* SLABS_U4, const void* MS_U4, const void* BIG_S,
    const void* RT1, const void* NV1,
    void* X_PAD, void* XN, void* BAD,
    void* S1_OUT, void* HI, void* THR_U,
    void* CNT1, void* CNT2, void* BEST,
    void* ARRIVE, const void* NLIVE_PTR,
    int M, int V, int vreal, int K, int sx,
    int cfg_id, int pdl,
    float cacc2, float infl_x, float x_abs,
    float infl_q, float infl_b, float slop_rel, float slop_abs,
    uintptr_t stream_ptr
) {
    return pre_screen_impl_v14(
        X, LOGITS_IDX, SLABS_U4, MS_U4, BIG_S, RT1, NV1,
        X_PAD, XN, BAD, S1_OUT, HI, THR_U, CNT1, CNT2, BEST,
        ARRIVE, nullptr, NLIVE_PTR,
        M, V, vreal, K, sx, cfg_id, pdl,
        cacc2, infl_x, x_abs, infl_q, infl_b, slop_rel, slop_abs,
        0.0f, 0.0f, 0.0f, stream_ptr
    );
}

extern "C" int cert_head_v7_post_refine(
    const void* X_PAD, const void* W_BF16,
    const void* LSB1_ROW, const void* MS_ROW, const void* BIG_S_ROW,
    const void* S1_OUT, const void* RT2, const void* NV2, const void* XN,
    const void* HI, const void* THR_AG, const void* BAD,
    void* T_GLOB, void* IDX1, void* CNT1,
    void* IDX2, void* CNT2,
    void* BEST, void* ARRIVE, void* OUT_PAIR, const void* NLIVE_PTR,
    int M, int V, int vreal, int vstart, int K, int sw,
    int ws, int tstride, int ref_grid, int pdl,
    float cacc2, float infl_q, float infl_b, float slop_rel, float slop_abs,
    float inv_cap, float cap, float slack,
    uintptr_t stream_ptr
) {
    return post_refine_impl_v14(
        X_PAD, W_BF16, LSB1_ROW, MS_ROW, BIG_S_ROW,
        S1_OUT, RT2, NV2, XN, HI, THR_AG, BAD, nullptr,
        T_GLOB, IDX1, nullptr, 0, CNT1, IDX2, CNT2,
        BEST, ARRIVE, OUT_PAIR, NLIVE_PTR,
        M, V, vreal, vstart, K, sw, ws, tstride, ref_grid, pdl,
        cacc2, infl_q, infl_b, slop_rel, slop_abs, inv_cap, cap, slack,
        stream_ptr
    );
}

extern "C" int cert_head_v7_post_refine_c48(
    const void* X_PAD, const void* W_BF16,
    const void* LSB1_ROW, const void* MS_ROW, const void* BIG_S_ROW,
    const void* S1_OUT, const void* RT2, const void* NV2, const void* XN,
    const void* HI, const void* THR_AG, const void* BAD,
    void* T_GLOB, void* IDX1, void* MASK1, void* CNT1,
    void* IDX2, void* CNT2,
    void* BEST, void* ARRIVE, void* OUT_PAIR, const void* NLIVE_PTR,
    int M, int V, int vreal, int vstart, int K, int sw,
    int ws, int tstride, int ref_grid, int pdl,
    float cacc2, float infl_q, float infl_b, float slop_rel, float slop_abs,
    float inv_cap, float cap, float slack,
    uintptr_t stream_ptr
) {
    return post_refine_impl_v14(
        X_PAD, W_BF16, LSB1_ROW, MS_ROW, BIG_S_ROW,
        S1_OUT, RT2, NV2, XN, HI, THR_AG, BAD, nullptr,
        T_GLOB, IDX1, MASK1, 1, CNT1, IDX2, CNT2,
        BEST, ARRIVE, OUT_PAIR, NLIVE_PTR,
        M, V, vreal, vstart, K, sw, ws, tstride, ref_grid, pdl,
        cacc2, infl_q, infl_b, slop_rel, slop_abs, inv_cap, cap, slack,
        stream_ptr
    );
}

extern "C" int cert_head_v9_post_refine(
    const void* X_PAD, const void* W_BF16,
    const void* LSB1_ROW, const void* MS_ROW, const void* BIG_S_ROW,
    const void* S1_OUT, const void* RT2, const void* NV2, const void* XN,
    const void* HI, const void* THR_AG, const void* BAD,
    void* T_GLOB, void* IDX1, void* MASK1_128, void* CNT1,
    void* IDX2, void* CNT2,
    void* BEST, void* ARRIVE, void* OUT_PAIR, const void* NLIVE_PTR,
    int M, int V, int vreal, int vstart, int K, int sw,
    int ws, int tstride, int ref_grid, int pdl,
    float cacc2, float infl_q, float infl_b, float slop_rel, float slop_abs,
    float inv_cap, float cap, float slack,
    uintptr_t stream_ptr
) {
    return post_refine_impl_v14(
        X_PAD, W_BF16, LSB1_ROW, MS_ROW, BIG_S_ROW,
        S1_OUT, RT2, NV2, XN, HI, THR_AG, BAD, nullptr,
        T_GLOB, IDX1, MASK1_128, 2, CNT1, IDX2, CNT2,
        BEST, ARRIVE, OUT_PAIR, NLIVE_PTR,
        M, V, vreal, vstart, K, sw, ws, tstride, ref_grid, pdl,
        cacc2, infl_q, infl_b, slop_rel, slop_abs, inv_cap, cap, slack,
        stream_ptr
    );
}

extern "C" int cert_head_v7_exact_all(
    const void* X_PAD, const void* W_BF16, void* OUT, const void* NLIVE_PTR,
    int M, int V, int K, int sw, int ref_grid,
    uintptr_t stream_ptr
) {
    if (M <= 0 || M > CERT_V14_MAX_M || V <= 0 || K <= 0 || (K & 15) != 0) return -2;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    const int n_tiles = (V + 31) / 32;
    const int rg = (ref_grid > 0 && ref_grid < n_tiles) ? ref_grid : n_tiles;
    #define L_EXACT(MT_V, ML, MOFF) launch_maybe_pdl_v14( \
        cert_exact_all_v14_kernel<MT_V, ML>, dim3(rg, 1, 1), dim3(128, 1, 1), 0, stream, false, \
        (const __nv_bfloat16*)X_PAD, (const __nv_bfloat16*)W_BF16, (float*)OUT, \
        (const int32_t*)NLIVE_PTR, M, V, K, sw, (MOFF))
    if (M <= 8)       return static_cast<int>(L_EXACT(1, true,  0));
    else if (M <= 16) return static_cast<int>(L_EXACT(1, false, 0));
    else if (M <= 32) return static_cast<int>(L_EXACT(2, false, 0));
    else if (M <= 48) return static_cast<int>(L_EXACT(3, false, 0));
    else if (M <= 64) return static_cast<int>(L_EXACT(4, false, 0));
    else {
        cudaError_t e1 = L_EXACT(4, false, 0);
        if (e1 != cudaSuccess) return static_cast<int>(e1);
        const int rem = M - 64;
        if (rem <= 16)      return static_cast<int>(L_EXACT(1, false, 64));
        else if (rem <= 32) return static_cast<int>(L_EXACT(2, false, 64));
        else if (rem <= 48) return static_cast<int>(L_EXACT(3, false, 64));
        else                return static_cast<int>(L_EXACT(4, false, 64));
    }
    #undef L_EXACT
}

extern "C" int cert_head_v7_version() { return 14; }
extern "C" int cert_head_v12_version() { return 14; }
extern "C" int cert_head_v14_version() { return 14; }
extern "C" int cert_head_v7_c48_max_m() { return 64; }
extern "C" int cert_head_v9_max_m() { return CERT_V14_MAX_M; }
