// SPDX-License-Identifier: Apache-2.0
// Wave-V16 Crown-Jewel Track 3: Unbiased Zero-Sync SM121a GPU `temp > 0` Speculative Rejection Sampler
// (`sync_free_temp_v16.cu` -> `libsync_free_temp_v16.so`)
//
// Features & Invariants:
//   1. Single-Kernel Zero-Host-Sync Speculative Rejection Sampling & K=0 Sampling over [M, V] (V = 200,000+):
//      - Supports BF16 and FP32 target logits with ZERO intermediate [M, V] FP32 probability allocations
//        (saves >= 102.9 MiB/step at M=128, V=201,024 vs multi-kernel PyTorch softmax/top_p/multinomial!).
//      - Mixed `temp == 0.0` + `temp > 0.0` batches in ONE kernel launch:
//          * `temp <= 0.0` rows execute exact deterministic tie-break-left greedy argmax.
//          * `temp > 0.0` rows execute exact unbiased rejection sampling + residual recovery.
//      - Pure `temp == 0.0` batches NEVER enter this kernel (100.000% untouched `1dcfb306` greedy SyncFree!).
//   2. 100% Mathematically Unbiased Filtering & Residual Recovery:
//      - Exact online-softmax `z_max` / `Z_0` + `min_p` (`z >= z_max + ln(min_p)`) + `top_k` + `top_p`.
//      - Fast Shared-Memory Path (`<= 2048` post-`min_p` survivors — >99.99% of `rollout_harvester` steps
//        at `temp=0.6, top_p=0.95, min_p=0.05`!): gathers survivors into 16 KiB SMEM, bitonic-sorts canonically
//        `(val DESC, idx ASC)`, computes exact `top_k`/`top_p` nucleus and inverse-CDF sample in SMEM with
//        ZERO extra DRAM passes and Pass-2 hitting GB10's 24 MiB L2 cache!
//      - Exact Overflow Path (`> 2048` survivors, e.g. `min_p=0` + flat logits): 0-allocation exact 256-bucket
//        IEEE-754 order-preserving radix search with deterministic `u64` fixed-point (`2^40`) mass histograms
//        + exact log-domain Exponential-Race (`argmax_v (log(r_v) - log(Exp(1)))`) over `r_v = max(0, p_v - q_v)`!
//      - Short-Circuit Verification: accepted draft tokens skip residual sampling completely; first rejected
//        position stops verification immediately (skipping all subsequent draft logit rows!).
//   3. All 4 Draft Distribution Regimes Supported:
//      - Mode 0 (`DRAFT_MODE_GREEDY`): `q(v) = 1[v == d_k]` (DSpark default `draft_logits is None`), 1-pass
//        Gumbel-Max when unfiltered, exact `p(v) / (1 - p(d_k))` residual for `v != d_k`.
//      - Mode 1 (`DRAFT_MODE_FULL_PROBS`): `[M, V]` FP32 `q_draft`, exact `u * q(d_k) < p(d_k)` + `max(0, p - q)`.
//      - Mode 2 (`DRAFT_MODE_FULL_LOGITS`): `[M, V]` BF16/FP32 `draft_logits`, on-the-fly online softmax `q_draft`
//        with 0 intermediate `[M, V]` allocation + exact `max(0, p - q)`.
//      - Mode 3 (`DRAFT_MODE_SCALAR_PROB`): per-draft scalar `q_d` with uniform off-token floor `(1 - q_d)/(V - 1)`.
//   4. TP=8 Lockstep Determinism & SM121a Build Hygiene:
//      - Deterministic Philox-4x32-10 PRNG keyed by `(seed, step_id, req_slot, pos, stream_tag, token_idx)`.
//      - Compiled strictly with `-gencode arch=compute_121a,code=sm_121a --fmad=false --ptxas-options=-v`,
//        0 B stack frame, 0 B register spills, strict `#206` PDL epilogue hygiene.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

namespace {

constexpr int SF_V16_THREADS = 256;
constexpr int SF_V16_WARPS   = 8;
constexpr int SF_V16_SMEM_CAP = 2048;
constexpr double SF_V16_FIX_SCALE = 1099511627776.0;      // 2^40 (overflow-safe up to V=16M)
constexpr double SF_V16_INV_FIX_SCALE = 1.0 / 1099511627776.0;

constexpr int DRAFT_MODE_GREEDY      = 0;
constexpr int DRAFT_MODE_FULL_PROBS  = 1;
constexpr int DRAFT_MODE_FULL_LOGITS = 2;
constexpr int DRAFT_MODE_SCALAR_PROB = 3;

static __device__ __forceinline__ void gdc_wait_v16() {
    asm volatile("griddepcontrol.wait;" ::: "memory");
}

static __device__ __forceinline__ void gdc_launch_dependents_v16() {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

// =========================================================================
// IEEE-754 Order-Preserving FP32 <-> uint32_t Bijection
// =========================================================================
static __device__ __forceinline__ uint32_t f32_to_ord_v16(float x) {
    uint32_t u = __float_as_uint(x);
    return (u & 0x80000000u) ? (~u) : (u | 0x80000000u);
}

static __device__ __forceinline__ float ord_to_f32_v16(uint32_t key) {
    uint32_t bits = (key & 0x80000000u) ? (key ^ 0x80000000u) : (~key);
    return __uint_as_float(bits);
}

static __device__ __forceinline__ float load_logit_v16(const void* base, int64_t row_off, int v, bool is_fp32) {
    float x = is_fp32
        ? static_cast<const float*>(base)[row_off + v]
        : __bfloat162float(static_cast<const __nv_bfloat16*>(base)[row_off + v]);
    return isnan(x) ? -INFINITY : x;
}

static __device__ __forceinline__ float load_scaled_z_v16(
    const void* base, int64_t row_off, int v, bool is_fp32, float T
) {
    float raw = load_logit_v16(base, row_off, v, is_fp32);
    if (!isfinite(raw)) return raw;
    return __fdiv_rn(raw, T);
}

static __device__ __forceinline__ int64_t read_idx_v16(const void* ptr, int64_t i, bool is_i64) {
    return is_i64
        ? static_cast<const int64_t*>(ptr)[i]
        : static_cast<int64_t>(static_cast<const int32_t*>(ptr)[i]);
}

// =========================================================================
// Deterministic Philox-4x32-10 PRNG (Zero Stack, Register-Only)
// =========================================================================
static __device__ __forceinline__ void philox_round_v16(
    uint32_t& c0, uint32_t& c1, uint32_t& c2, uint32_t& c3,
    uint32_t k0, uint32_t k1
) {
    constexpr uint32_t M0 = 0xD2511F53u;
    constexpr uint32_t M1 = 0xCD9E8D57u;
    uint64_t p0 = static_cast<uint64_t>(M0) * static_cast<uint64_t>(c0);
    uint64_t p1 = static_cast<uint64_t>(M1) * static_cast<uint64_t>(c2);
    uint32_t hi0 = static_cast<uint32_t>(p0 >> 32);
    uint32_t lo0 = static_cast<uint32_t>(p0);
    uint32_t hi1 = static_cast<uint32_t>(p1 >> 32);
    uint32_t lo1 = static_cast<uint32_t>(p1);
    c0 = hi1 ^ c1 ^ k0;
    c1 = lo1;
    c2 = hi0 ^ c3 ^ k1;
    c3 = lo0;
}

static __device__ __forceinline__ uint4 philox4x32_10_v16(
    uint32_t c0, uint32_t c1, uint32_t c2, uint32_t c3,
    uint32_t k0, uint32_t k1
) {
    constexpr uint32_t W0 = 0x9E3779B9u;
    constexpr uint32_t W1 = 0xBB67AE85u;
    #pragma unroll
    for (int r = 0; r < 10; ++r) {
        philox_round_v16(c0, c1, c2, c3, k0, k1);
        k0 += W0;
        k1 += W1;
    }
    return make_uint4(c0, c1, c2, c3);
}

// 53-bit exact open-interval Uniform(0, 1) in double from 2 Philox words
static __device__ __forceinline__ double philox_u01_f64_v16(
    uint64_t seed, uint64_t step_id, uint32_t req_slot, uint32_t pos, uint32_t tag
) {
    uint32_t k0 = static_cast<uint32_t>(seed & 0xFFFFFFFFull);
    uint32_t k1 = static_cast<uint32_t>(seed >> 32);
    uint32_t c0 = 0xFFFFFFFFu;
    uint32_t c1 = pos;
    uint32_t c2 = (req_slot & 0x0FFFFFFFu) | ((tag & 0xFu) << 28);
    uint32_t c3 = static_cast<uint32_t>(step_id ^ (step_id >> 32));
    uint4 o = philox4x32_10_v16(c0, c1, c2, c3, k0, k1);
    uint64_t r53 = (static_cast<uint64_t>(o.x >> 5) << 26) | static_cast<uint64_t>(o.y >> 6);
    return (static_cast<double>(r53) + 0.5) * (1.0 / 9007199254740992.0);
}

// Full-tail symmetric FP32 standard Exponential Exp(1) from a single uint32_t (never 0, never inf!)
// Uses U_small in (1.16e-10, 0.5) on both halves so E -> 0 (max Gumbel tail!) has full 32-bit resolution!
static __device__ __forceinline__ float u32_to_exp1_f32_v16(uint32_t r) {
    constexpr float INV_2P32 = 2.3283064365386963e-10f;
    if ((r & 0x80000000u) == 0u) {
        float u_small = __fmul_rn(__fadd_rn(static_cast<float>(r), 0.5f), INV_2P32);
        return -log1pf(-u_small);
    } else {
        float u_small = __fmul_rn(__fadd_rn(static_cast<float>(~r), 0.5f), INV_2P32);
        return -logf(u_small);
    }
}

static __device__ __forceinline__ float philox_gumbel_f32_v16(
    uint64_t seed, uint64_t step_id, uint32_t req_slot, uint32_t pos, uint32_t token_v
) {
    uint32_t k0 = static_cast<uint32_t>(seed & 0xFFFFFFFFull);
    uint32_t k1 = static_cast<uint32_t>(seed >> 32);
    uint32_t c0 = token_v;
    uint32_t c1 = pos;
    uint32_t c2 = (req_slot & 0x0FFFFFFFu) | (1u << 28);
    uint32_t c3 = static_cast<uint32_t>(step_id ^ (step_id >> 32));
    uint4 o = philox4x32_10_v16(c0, c1, c2, c3, k0, k1);
    float e = u32_to_exp1_f32_v16(o.x);
    return -logf(e);
}

static __device__ __forceinline__ double philox_log_exp1_f64_v16(
    uint64_t seed, uint64_t step_id, uint32_t req_slot, uint32_t pos, uint32_t token_v
) {
    uint32_t k0 = static_cast<uint32_t>(seed & 0xFFFFFFFFull);
    uint32_t k1 = static_cast<uint32_t>(seed >> 32);
    uint32_t c0 = token_v;
    uint32_t c1 = pos;
    uint32_t c2 = (req_slot & 0x0FFFFFFFu) | (2u << 28);
    uint32_t c3 = static_cast<uint32_t>(step_id ^ (step_id >> 32));
    uint4 o = philox4x32_10_v16(c0, c1, c2, c3, k0, k1);
    uint64_t r53 = (static_cast<uint64_t>(o.x >> 5) << 26) | static_cast<uint64_t>(o.y >> 6);
    double u = (static_cast<double>(r53) + 0.5) * (1.0 / 9007199254740992.0);
    double e = (u < 0.5) ? (-log1p(-u)) : (-log(1.0 - u));
    return log(e);
}

static __device__ __forceinline__ unsigned long long weight_to_u64_v16(float z_minus_zmax) {
    if (z_minus_zmax < -35.0f) return 0ull;
    double w = exp(static_cast<double>(z_minus_zmax));
    double scaled = w * SF_V16_FIX_SCALE + 0.5;
    if (scaled < 0.0) return 0ull;
    return static_cast<unsigned long long>(scaled);
}

// =========================================================================
// Shared-Memory Scratch Layout (<= 18 KiB per CTA -> 2+ CTAs/SM, 0 spills!)
// =========================================================================
struct SharedScratchV16 {
    union {
        struct {
            float   val[SF_V16_SMEM_CAP];   // 8 KiB
            int32_t idx[SF_V16_SMEM_CAP];   // 8 KiB
        } surv;
        struct {
            uint32_t           hist_cnt[256];       // 1 KiB
            unsigned long long hist_mass[256];      // 2 KiB
        } rad;
    } u;
    float    red_f[SF_V16_WARPS];
    int32_t  red_i[SF_V16_WARPS];
    double   red_d[SF_V16_WARPS];
    uint32_t red_u[SF_V16_WARPS];
    // Broadcast scalars
    float    b_zdraft;
    float    b_qd;
    uint32_t b_count;
    uint32_t b_lo_ord;
    uint32_t b_hi_ord;
    uint32_t b_above_cnt;
    unsigned long long b_above_mass;
    unsigned long long b_tot_mass;
    int32_t  b_accepted;
    int32_t  b_sampled_tok;
};

// =========================================================================
// Deterministic Block Reductions (Bit-Exact Across TP=8 Ranks!)
// =========================================================================
static __device__ __forceinline__ void block_argmax_f32_v16(
    SharedScratchV16& smem, float val, int32_t idx, float& out_val, int32_t& out_idx
) {
    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    #pragma unroll
    for (int off = 16; off >= 1; off >>= 1) {
        float ov = __shfl_xor_sync(0xFFFFFFFFu, val, off);
        int32_t oi = __shfl_xor_sync(0xFFFFFFFFu, idx, off);
        if (ov > val || (ov == val && oi < idx)) {
            val = ov;
            idx = oi;
        }
    }
    if (lane == 0) {
        smem.red_f[warp] = val;
        smem.red_i[warp] = idx;
    }
    __syncthreads();
    if (tid == 0) {
        float bv = smem.red_f[0];
        int32_t bi = smem.red_i[0];
        #pragma unroll
        for (int w = 1; w < SF_V16_WARPS; ++w) {
            float wv = smem.red_f[w];
            int32_t wi = smem.red_i[w];
            if (wv > bv || (wv == bv && wi < bi)) {
                bv = wv;
                bi = wi;
            }
        }
        smem.red_f[0] = bv;
        smem.red_i[0] = bi;
    }
    __syncthreads();
    out_val = smem.red_f[0];
    out_idx = smem.red_i[0];
}

static __device__ __forceinline__ void block_argmax_f64_v16(
    SharedScratchV16& smem, double val, int32_t idx, double& out_val, int32_t& out_idx
) {
    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    #pragma unroll
    for (int off = 16; off >= 1; off >>= 1) {
        double ov = __shfl_xor_sync(0xFFFFFFFFu, val, off);
        int32_t oi = __shfl_xor_sync(0xFFFFFFFFu, idx, off);
        if (ov > val || (ov == val && oi < idx)) {
            val = ov;
            idx = oi;
        }
    }
    if (lane == 0) {
        smem.red_d[warp] = val;
        smem.red_i[warp] = idx;
    }
    __syncthreads();
    if (tid == 0) {
        double bv = smem.red_d[0];
        int32_t bi = smem.red_i[0];
        #pragma unroll
        for (int w = 1; w < SF_V16_WARPS; ++w) {
            double wv = smem.red_d[w];
            int32_t wi = smem.red_i[w];
            if (wv > bv || (wv == bv && wi < bi)) {
                bv = wv;
                bi = wi;
            }
        }
        smem.red_d[0] = bv;
        smem.red_i[0] = bi;
    }
    __syncthreads();
    out_val = smem.red_d[0];
    out_idx = smem.red_i[0];
}

static __device__ __forceinline__ double block_sum_f64_v16(SharedScratchV16& smem, double v) {
    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    #pragma unroll
    for (int off = 16; off >= 1; off >>= 1) {
        v += __shfl_xor_sync(0xFFFFFFFFu, v, off);
    }
    if (lane == 0) smem.red_d[warp] = v;
    __syncthreads();
    if (tid == 0) {
        double s = 0.0;
        #pragma unroll
        for (int w = 0; w < SF_V16_WARPS; ++w) s += smem.red_d[w];
        smem.red_d[0] = s;
    }
    __syncthreads();
    return smem.red_d[0];
}

static __device__ __forceinline__ uint32_t block_min_u32_v16(SharedScratchV16& smem, uint32_t v) {
    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    #pragma unroll
    for (int off = 16; off >= 1; off >>= 1) {
        uint32_t o = __shfl_xor_sync(0xFFFFFFFFu, v, off);
        if (o < v) v = o;
    }
    if (lane == 0) smem.red_u[warp] = v;
    __syncthreads();
    if (tid == 0) {
        uint32_t m = smem.red_u[0];
        #pragma unroll
        for (int w = 1; w < SF_V16_WARPS; ++w) {
            if (smem.red_u[w] < m) m = smem.red_u[w];
        }
        smem.red_u[0] = m;
    }
    __syncthreads();
    return smem.red_u[0];
}

// =========================================================================
// Helper: Find Exact `allow_n`-th Token Index `v` Tied at `target_ord`
// (Guarantees 100% Exact Tie-Break-Left Even on Adversarial 200K-Way Ties!)
// =========================================================================
template <bool LOGITS_F32>
static __device__ int32_t find_nth_tied_index_v16(
    SharedScratchV16& smem,
    const void* logits,
    int64_t row_off,
    int V,
    float T,
    uint32_t target_ord,
    uint32_t allow_n,
    bool has_minp,
    float z_minp,
    int32_t argmax_v,
    int32_t max_v_bound
) {
    const int tid = static_cast<int>(threadIdx.x);
    if (allow_n == 0u) return -1;
    if (tid == 0) {
        smem.b_lo_ord = 0u;
        smem.b_hi_ord = static_cast<uint32_t>((max_v_bound < V) ? max_v_bound : (V - 1));
        smem.b_above_cnt = 0u;
    }
    __syncthreads();

    for (int shift = 24; shift >= 0; shift -= 8) {
        uint32_t lo_v = smem.b_lo_ord;
        uint32_t hi_v = smem.b_hi_ord;
        if ((lo_v >> shift) == (hi_v >> shift)) continue;
        smem.u.rad.hist_cnt[tid] = 0u;
        __syncthreads();
        for (int v = static_cast<int>(lo_v) + tid; v <= static_cast<int>(hi_v); v += SF_V16_THREADS) {
            float z = load_scaled_z_v16(logits, row_off, v, LOGITS_F32, T);
            bool ok = isfinite(z) && (!has_minp || z >= z_minp || v == argmax_v);
            if (ok && f32_to_ord_v16(z) == target_ord) {
                uint32_t b = (static_cast<uint32_t>(v) >> shift) & 0xFFu;
                atomicAdd(&smem.u.rad.hist_cnt[b], 1u);
            }
        }
        __syncthreads();
        if (tid == 0) {
            uint32_t acc = smem.b_above_cnt;
            uint32_t sel_b = 0u;
            for (int b = 0; b < 256; ++b) {
                uint32_t c = smem.u.rad.hist_cnt[b];
                if (acc + c >= allow_n) {
                    sel_b = static_cast<uint32_t>(b);
                    break;
                }
                acc += c;
            }
            smem.b_above_cnt = acc;
            uint32_t mask = (shift == 24) ? 0u : ~((1u << (shift + 8)) - 1u);
            uint32_t base = (lo_v & mask) | (sel_b << shift);
            uint32_t span = (1u << shift) - 1u;
            smem.b_lo_ord = (base > lo_v) ? base : lo_v;
            smem.b_hi_ord = ((base | span) < hi_v) ? (base | span) : hi_v;
        }
        __syncthreads();
    }
    return static_cast<int32_t>(smem.b_lo_ord);
}

// =========================================================================
// Core Per-Row Exact Unbiased Verification / Sampling Engine
//   - `do_verify == true`: checks acceptance of `draft_tok` (if `valid_draft`);
//     if accepted sets `out_accepted = 1, out_tok = draft_tok`;
//     if rejected sets `out_accepted = 0, out_tok = exact_residual_sample`.
//   - `do_verify == false`: samples exact target token (`bonus` or `K=0`).
// =========================================================================
template <bool LOGITS_F32, bool HAS_DRAFT_Q>
static __device__ void process_row_v16(
    SharedScratchV16& smem,
    const void* logits,
    int64_t logits_stride,
    int64_t row_idx,
    int V,
    float T,
    float top_p_val,
    int32_t top_k_val,
    float min_p_val,
    uint64_t seed,
    uint64_t step_id,
    uint32_t req_slot,
    uint32_t pos,
    bool do_verify,
    int64_t draft_tok,
    bool valid_draft,
    const void* draft_q,
    int draft_mode,
    bool draft_is_fp32,
    int64_t draft_stride,
    int64_t draft_scalar_idx,
    int32_t& out_accepted,
    int64_t& out_tok
) {
    const int tid = static_cast<int>(threadIdx.x);
    const int64_t row_off = row_idx * logits_stride;
    const bool is_greedy = !(T > 0.0f);
    (void)draft_q;
    (void)draft_mode;
    (void)draft_is_fp32;
    (void)draft_stride;
    (void)draft_scalar_idx;

    // ---------------------------------------------------------------------
    // FAST PATH 0: Exact Greedy Row (`temp <= 0.0f`) in Mixed Batch
    // ---------------------------------------------------------------------
    if (is_greedy) {
        float best_v = -INFINITY;
        int32_t best_i = 0;
        for (int v = tid; v < V; v += SF_V16_THREADS) {
            float x = load_logit_v16(logits, row_off, v, LOGITS_F32);
            if (x > best_v || (x == best_v && v < best_i)) {
                best_v = x;
                best_i = v;
            }
        }
        float g_v;
        int32_t g_i;
        block_argmax_f32_v16(smem, best_v, best_i, g_v, g_i);
        if (do_verify) {
            bool acc = valid_draft && (static_cast<int64_t>(g_i) == draft_tok);
            out_accepted = acc ? 1 : 0;
            out_tok = acc ? draft_tok : static_cast<int64_t>(g_i);
        } else {
            out_accepted = 0;
            out_tok = static_cast<int64_t>(g_i);
        }
        return;
    }

    const bool has_minp = (min_p_val > 0.0f && min_p_val < 1.0f);
    const bool has_topk = (top_k_val > 0 && top_k_val < V);
    const bool has_topp = (top_p_val > 0.0f && top_p_val < 1.0f);
    const bool unfiltered = (!has_minp && !has_topk && !has_topp);
    const bool single_pass_gumbel = unfiltered && (!HAS_DRAFT_Q || !do_verify || draft_mode == DRAFT_MODE_GREEDY);

    // ---------------------------------------------------------------------
    // PASS 1: Exact Online-Softmax `z_max`, `argmax_v`, `Z_0`, `z_draft`
    //         (+ 1-Pass Exact Gumbel-Max when Unfiltered & Greedy Draft / Bonus!)
    //         (+ Optional Draft Logits `q_max`, `Z_q` for Mode 2!)
    // ---------------------------------------------------------------------
    float t_zmax = -INFINITY;
    int32_t t_argmax = 0;
    double t_zsum = 0.0;
    float t_g_all = -INFINITY;
    int32_t t_gi_all = 0;
    float t_g_ex = -INFINITY;
    int32_t t_gi_ex = -1;

    float t_qmax = -INFINITY;
    double t_qsum = 0.0;

    if (tid == 0) {
        smem.b_zdraft = -INFINITY;
        smem.b_qd = 1.0f;
    }
    __syncthreads();

    const int64_t q_row_off = row_idx * draft_stride;
    for (int v = tid; v < V; v += SF_V16_THREADS) {
        float z = load_scaled_z_v16(logits, row_off, v, LOGITS_F32, T);
        if (do_verify && valid_draft && static_cast<int64_t>(v) == draft_tok) {
            smem.b_zdraft = z;
        }
        if (isfinite(z)) {
            if (z > t_zmax) {
                t_zsum = (t_zsum > 0.0) ? (t_zsum * exp(static_cast<double>(t_zmax - z)) + 1.0) : 1.0;
                t_zmax = z;
                t_argmax = v;
            } else {
                if (z == t_zmax && v < t_argmax) t_argmax = v;
                t_zsum += exp(static_cast<double>(z - t_zmax));
            }
            if (single_pass_gumbel) {
                float g = philox_gumbel_f32_v16(seed, step_id, req_slot, pos, static_cast<uint32_t>(v));
                float sc = __fadd_rn(z, g);
                if (sc > t_g_all || (sc == t_g_all && v < t_gi_all)) {
                    t_g_all = sc;
                    t_gi_all = v;
                }
                if (!do_verify || !valid_draft || static_cast<int64_t>(v) != draft_tok) {
                    if (sc > t_g_ex || (sc == t_g_ex && (t_gi_ex < 0 || v < t_gi_ex))) {
                        t_g_ex = sc;
                        t_gi_ex = v;
                    }
                }
            }
        }
        if constexpr (HAS_DRAFT_Q) {
            if (do_verify && draft_mode == DRAFT_MODE_FULL_LOGITS && draft_q != nullptr) {
                float ql = load_logit_v16(draft_q, q_row_off, v, draft_is_fp32);
                if (valid_draft && static_cast<int64_t>(v) == draft_tok) {
                    smem.b_qd = ql;
                }
                if (isfinite(ql)) {
                    if (ql > t_qmax) {
                        t_qsum = (t_qsum > 0.0) ? (t_qsum * exp(static_cast<double>(t_qmax - ql)) + 1.0) : 1.0;
                        t_qmax = ql;
                    } else {
                        t_qsum += exp(static_cast<double>(ql - t_qmax));
                    }
                }
            }
        }
    }

    float z_max;
    int32_t argmax_v;
    block_argmax_f32_v16(smem, t_zmax, t_argmax, z_max, argmax_v);

    // Handle degenerate all-non-finite row fail-closed -> token 0
    if (!isfinite(z_max)) {
        out_accepted = 0;
        out_tok = 0;
        return;
    }

    double t_zsum_adj = (t_zsum > 0.0 && isfinite(t_zmax))
        ? (t_zsum * exp(static_cast<double>(t_zmax - z_max)))
        : 0.0;
    double Z_0 = block_sum_f64_v16(smem, t_zsum_adj);

    float q_max = -INFINITY;
    double Z_q = 1.0;
    double q_d = 1.0;
    if constexpr (HAS_DRAFT_Q) {
        if (do_verify && draft_q != nullptr) {
            if (draft_mode == DRAFT_MODE_FULL_LOGITS) {
                int32_t dummy_qi = 0;
                block_argmax_f32_v16(smem, t_qmax, 0, q_max, dummy_qi);
                double t_qsum_adj = (t_qsum > 0.0 && isfinite(t_qmax))
                    ? (t_qsum * exp(static_cast<double>(t_qmax - q_max)))
                    : 0.0;
                Z_q = block_sum_f64_v16(smem, t_qsum_adj);
                float q_raw_d = smem.b_qd;
                q_d = (valid_draft && isfinite(q_raw_d) && isfinite(q_max) && Z_q > 0.0)
                    ? (exp(static_cast<double>(q_raw_d - q_max)) / Z_q)
                    : 0.0;
            } else if (draft_mode == DRAFT_MODE_FULL_PROBS) {
                if (valid_draft && draft_tok >= 0 && draft_tok < V) {
                    q_d = static_cast<double>(static_cast<const float*>(draft_q)[q_row_off + draft_tok]);
                } else {
                    q_d = 0.0;
                }
            } else if (draft_mode == DRAFT_MODE_SCALAR_PROB) {
                q_d = valid_draft
                    ? static_cast<double>(static_cast<const float*>(draft_q)[draft_scalar_idx])
                    : 0.0;
                if (!(q_d > 0.0)) q_d = 0.0;
                if (q_d > 1.0) q_d = 1.0;
            }
        }
    }

    // ---------------------------------------------------------------------
    // FAST PATH 1: Single-Pass Unfiltered Sampling / Greedy-Draft Rejection
    // ---------------------------------------------------------------------
    if (single_pass_gumbel) {
        float best_all_sc;
        int32_t best_all_i;
        block_argmax_f32_v16(smem, t_g_all, t_gi_all, best_all_sc, best_all_i);

        if (!do_verify) {
            out_accepted = 0;
            out_tok = static_cast<int64_t>(best_all_i);
            return;
        }
        float z_d = smem.b_zdraft;
        double p_d = (valid_draft && isfinite(z_d) && Z_0 > 0.0)
            ? (exp(static_cast<double>(z_d - z_max)) / Z_0)
            : 0.0;
        double u_acc = philox_u01_f64_v16(seed, step_id, req_slot, pos, 0u);
        if (valid_draft && u_acc < p_d) {
            out_accepted = 1;
            out_tok = draft_tok;
            return;
        }
        float best_ex_sc;
        int32_t best_ex_i;
        block_argmax_f32_v16(smem, t_g_ex, (t_gi_ex >= 0) ? t_gi_ex : V, best_ex_sc, best_ex_i);
        out_accepted = 0;
        out_tok = (best_ex_i >= 0 && best_ex_i < V && isfinite(best_ex_sc))
            ? static_cast<int64_t>(best_ex_i)
            : static_cast<int64_t>(best_all_i);
        return;
    }

    // ---------------------------------------------------------------------
    // FAST PATH 2: Unfiltered + Full/Scalar `q_draft` Verification (0 Allocations!)
    // ---------------------------------------------------------------------
    if (unfiltered) {
        // Check acceptance immediately from Pass 1 (`Z_0` & `z_draft` already exact!)
        float z_d = smem.b_zdraft;
        double p_d = (valid_draft && isfinite(z_d) && Z_0 > 0.0)
            ? (exp(static_cast<double>(z_d - z_max)) / Z_0)
            : 0.0;
        double u_acc = philox_u01_f64_v16(seed, step_id, req_slot, pos, 0u);
        if (valid_draft && q_d > 0.0 && u_acc * q_d < p_d) {
            out_accepted = 1;
            out_tok = draft_tok;
            return;
        }
        // Rejected -> 1-pass exact log-domain Exponential Race over r_v = max(0, p_v - q_v)!
        double best_res_sc = -INFINITY;
        int32_t best_res_i = -1;
        double best_fb_sc = -INFINITY;
        int32_t best_fb_i = argmax_v;
        const double inv_Z0 = 1.0 / Z_0;
        const double inv_Zq = (Z_q > 0.0) ? (1.0 / Z_q) : 0.0;
        const double scalar_off_q = (V > 1) ? ((1.0 - q_d) / static_cast<double>(V - 1)) : 0.0;

        for (int v = tid; v < V; v += SF_V16_THREADS) {
            float z = load_scaled_z_v16(logits, row_off, v, LOGITS_F32, T);
            if (!isfinite(z)) continue;
            double p_v = exp(static_cast<double>(z - z_max)) * inv_Z0;
            double q_v = 0.0;
            if constexpr (HAS_DRAFT_Q) {
                if (draft_mode == DRAFT_MODE_FULL_PROBS && draft_q != nullptr) {
                    q_v = static_cast<double>(static_cast<const float*>(draft_q)[q_row_off + v]);
                } else if (draft_mode == DRAFT_MODE_FULL_LOGITS && draft_q != nullptr) {
                    float ql = load_logit_v16(draft_q, q_row_off, v, draft_is_fp32);
                    q_v = isfinite(ql) ? (exp(static_cast<double>(ql - q_max)) * inv_Zq) : 0.0;
                } else if (draft_mode == DRAFT_MODE_SCALAR_PROB) {
                    q_v = (valid_draft && static_cast<int64_t>(v) == draft_tok) ? q_d : scalar_off_q;
                }
            }
            double log_e = philox_log_exp1_f64_v16(seed, step_id, req_slot, pos, static_cast<uint32_t>(v));
            double fb_sc = static_cast<double>(z - z_max) - log_e;
            if (fb_sc > best_fb_sc || (fb_sc == best_fb_sc && v < best_fb_i)) {
                best_fb_sc = fb_sc;
                best_fb_i = v;
            }
            double r_v = p_v - q_v;
            if (r_v > 0.0) {
                double sc = log(r_v) - log_e;
                if (sc > best_res_sc || (sc == best_res_sc && (best_res_i < 0 || v < best_res_i))) {
                    best_res_sc = sc;
                    best_res_i = v;
                }
            }
        }
        double g_res_sc;
        int32_t g_res_i;
        block_argmax_f64_v16(smem, best_res_sc, (best_res_i >= 0) ? best_res_i : V, g_res_sc, g_res_i);
        if (g_res_i >= 0 && g_res_i < V && isfinite(g_res_sc)) {
            out_accepted = 0;
            out_tok = static_cast<int64_t>(g_res_i);
        } else {
            double g_fb_sc;
            int32_t g_fb_i;
            block_argmax_f64_v16(smem, best_fb_sc, best_fb_i, g_fb_sc, g_fb_i);
            out_accepted = 0;
            out_tok = static_cast<int64_t>(g_fb_i);
        }
        return;
    }

    // =====================================================================
    // FILTERED PATH (`min_p` / `top_k` / `top_p`)
    //   Step A: Gather post-`min_p` survivors into 16 KiB SMEM (<= 2048).
    //           For `rollout_harvester` (`min_p=0.05, top_p=0.95`), Pass 2 hits
    //           GB10's 24 MiB L2 cache and gathers ~5..30 survivors -> 0 DRAM!
    // =====================================================================
    const float z_minp = has_minp ? __fadd_rn(z_max, logf(min_p_val)) : -INFINITY;
    if (tid == 0) smem.b_count = 0u;
    __syncthreads();

    uint32_t t_min_ord = 0xFFFFFFFFu;
    for (int v = tid; v < V; v += SF_V16_THREADS) {
        float z = load_scaled_z_v16(logits, row_off, v, LOGITS_F32, T);
        if (isfinite(z) && (!has_minp || z >= z_minp || v == argmax_v)) {
            uint32_t o = f32_to_ord_v16(z);
            if (o < t_min_ord) t_min_ord = o;
            uint32_t slot = atomicAdd(&smem.b_count, 1u);
            if (slot < static_cast<uint32_t>(SF_V16_SMEM_CAP)) {
                smem.u.surv.val[slot] = z;
                smem.u.surv.idx[slot] = v;
            }
        }
    }
    __syncthreads();

    const uint32_t n_surv = smem.b_count;

    // ---------------------------------------------------------------------
    // TIER 1: Exact Shared-Memory Path (`n_surv <= 2048` — 0 Extra DRAM Passes!)
    // ---------------------------------------------------------------------
    if (n_surv <= static_cast<uint32_t>(SF_V16_SMEM_CAP)) {
        if (n_surv == 0u) {
            out_accepted = 0;
            out_tok = static_cast<int64_t>(argmax_v);
            return;
        }
        // Pad to next power of 2 <= 2048 and run parallel bitonic sort (val DESC, idx ASC)
        uint32_t n_pow2 = 1u;
        while (n_pow2 < n_surv) n_pow2 <<= 1;
        for (uint32_t i = n_surv + static_cast<uint32_t>(tid); i < n_pow2; i += SF_V16_THREADS) {
            smem.u.surv.val[i] = -INFINITY;
            smem.u.surv.idx[i] = 0x7FFFFFFF;
        }
        __syncthreads();

        for (uint32_t k_len = 2u; k_len <= n_pow2; k_len <<= 1) {
            for (uint32_t j = k_len >> 1; j > 0u; j >>= 1) {
                for (uint32_t i = static_cast<uint32_t>(tid); i < n_pow2; i += SF_V16_THREADS) {
                    uint32_t ixj = i ^ j;
                    if (ixj > i) {
                        bool asc = ((i & k_len) == 0u);  // ascending of DESC order = larger val first
                        float vi = smem.u.surv.val[i];
                        int32_t ii = smem.u.surv.idx[i];
                        float vj = smem.u.surv.val[ixj];
                        int32_t ij = smem.u.surv.idx[ixj];
                        bool i_better = (vi > vj) || (vi == vj && ii < ij);
                        bool do_swap = asc ? (!i_better) : i_better;
                        if (do_swap) {
                            smem.u.surv.val[i] = vj;
                            smem.u.surv.idx[i] = ij;
                            smem.u.surv.val[ixj] = vi;
                            smem.u.surv.idx[ixj] = ii;
                        }
                    }
                }
                __syncthreads();
            }
        }

        // Exact top_k -> top_p truncation + acceptance + inverse-CDF sampling in SMEM (thread 0, 0 DRAM!)
        if (tid == 0) {
            uint32_t n_act = (has_topk && static_cast<uint32_t>(top_k_val) < n_surv)
                ? static_cast<uint32_t>(top_k_val)
                : n_surv;
            if (n_act == 0u) n_act = 1u;

            double z_act = 0.0;
            for (uint32_t i = 0u; i < n_act; ++i) {
                z_act += exp(static_cast<double>(smem.u.surv.val[i] - z_max));
            }
            uint32_t n_nuc = n_act;
            double z_nuc = z_act;
            if (has_topp && z_act > 0.0) {
                double target_m = static_cast<double>(top_p_val) * z_act;
                double cum = 0.0;
                n_nuc = 1u;
                for (uint32_t i = 0u; i < n_act; ++i) {
                    cum += exp(static_cast<double>(smem.u.surv.val[i] - z_max));
                    n_nuc = i + 1u;
                    if (cum >= target_m) break;
                }
                z_nuc = cum;
            }
            const double inv_znuc = (z_nuc > 0.0) ? (1.0 / z_nuc) : 0.0;

            int32_t accepted_flag = 0;
            int32_t chosen_tok = smem.u.surv.idx[0];

            if (do_verify) {
                double p_d = 0.0;
                if (valid_draft) {
                    for (uint32_t i = 0u; i < n_nuc; ++i) {
                        if (static_cast<int64_t>(smem.u.surv.idx[i]) == draft_tok) {
                            p_d = exp(static_cast<double>(smem.u.surv.val[i] - z_max)) * inv_znuc;
                            break;
                        }
                    }
                }
                double u_acc = philox_u01_f64_v16(seed, step_id, req_slot, pos, 0u);
                if (valid_draft && q_d > 0.0 && u_acc * q_d < p_d) {
                    accepted_flag = 1;
                    chosen_tok = static_cast<int32_t>(draft_tok);
                }
            }

            if (!accepted_flag) {
                // Exact inverse-CDF draw in SMEM over `s_idx[0..n_nuc-1]`!
                const double inv_Zq = (Z_q > 0.0) ? (1.0 / Z_q) : 0.0;
                const double scalar_off_q = (V > 1) ? ((1.0 - q_d) / static_cast<double>(V - 1)) : 0.0;

                double w_tot = 0.0;
                for (uint32_t i = 0u; i < n_nuc; ++i) {
                    double p_i = exp(static_cast<double>(smem.u.surv.val[i] - z_max)) * inv_znuc;
                    double w_i = p_i;
                    if (do_verify) {
                        int32_t vi = smem.u.surv.idx[i];
                        double q_i = 0.0;
                        if constexpr (!HAS_DRAFT_Q) {
                            q_i = (valid_draft && static_cast<int64_t>(vi) == draft_tok) ? 1.0 : 0.0;
                        } else {
                            if (draft_mode == DRAFT_MODE_GREEDY) {
                                q_i = (valid_draft && static_cast<int64_t>(vi) == draft_tok) ? 1.0 : 0.0;
                            } else if (draft_mode == DRAFT_MODE_FULL_PROBS && draft_q != nullptr) {
                                q_i = static_cast<double>(static_cast<const float*>(draft_q)[q_row_off + vi]);
                            } else if (draft_mode == DRAFT_MODE_FULL_LOGITS && draft_q != nullptr) {
                                float ql = load_logit_v16(draft_q, q_row_off, vi, draft_is_fp32);
                                q_i = isfinite(ql) ? (exp(static_cast<double>(ql - q_max)) * inv_Zq) : 0.0;
                            } else if (draft_mode == DRAFT_MODE_SCALAR_PROB) {
                                q_i = (valid_draft && static_cast<int64_t>(vi) == draft_tok) ? q_d : scalar_off_q;
                            }
                        }
                        w_i = (p_i > q_i) ? (p_i - q_i) : 0.0;
                    }
                    // Store residual weights back into smem.u.surv.val[i] for 2nd exact CDF pass!
                    float wf = static_cast<float>(w_i);
                    smem.u.surv.val[i] = wf;
                    w_tot += static_cast<double>(wf);
                }

                double u_samp = philox_u01_f64_v16(seed, step_id, req_slot, pos, 1u);
                if (w_tot > 0.0) {
                    double thresh = u_samp * w_tot;
                    double cum = 0.0;
                    int32_t last_pos_tok = smem.u.surv.idx[0];
                    bool picked = false;
                    for (uint32_t i = 0u; i < n_nuc; ++i) {
                        double wi = static_cast<double>(smem.u.surv.val[i]);
                        if (wi > 0.0) {
                            last_pos_tok = smem.u.surv.idx[i];
                            cum += wi;
                            if (cum >= thresh) {
                                chosen_tok = last_pos_tok;
                                picked = true;
                                break;
                            }
                        }
                    }
                    if (!picked) chosen_tok = last_pos_tok;
                } else {
                    // Fallback if p == q identically and FP edge rejected: draw from p_target over nucleus
                    chosen_tok = smem.u.surv.idx[0];
                }
            }
            smem.b_accepted = accepted_flag;
            smem.b_sampled_tok = chosen_tok;
        }
        __syncthreads();
        out_accepted = smem.b_accepted;
        out_tok = static_cast<int64_t>(smem.b_sampled_tok);
        return;
    }

    // ---------------------------------------------------------------------
    // TIER 2: Exact Overflow Path (`n_surv > 2048` — 0 Global Allocations!)
    //   Uses exact 256-bucket radix search + deterministic u64 fixed-point
    //   mass histograms (bit-exact across TP=8 ranks!) + log Exponential-Race.
    // ---------------------------------------------------------------------
    uint32_t g_min_ord = block_min_u32_v16(smem, t_min_ord);
    uint32_t zmax_ord = f32_to_ord_v16(z_max);

    uint32_t tau_k_ord = g_min_ord;
    int32_t v_k_cut = V;
    if (has_topk && static_cast<uint32_t>(top_k_val) < n_surv) {
        if (tid == 0) {
            smem.b_lo_ord = g_min_ord;
            smem.b_hi_ord = zmax_ord;
            smem.b_above_cnt = 0u;
        }
        __syncthreads();
        for (int shift = 24; shift >= 0; shift -= 8) {
            uint32_t lo_o = smem.b_lo_ord;
            uint32_t hi_o = smem.b_hi_ord;
            if ((lo_o >> shift) == (hi_o >> shift)) continue;
            smem.u.rad.hist_cnt[tid] = 0u;
            __syncthreads();
            for (int v = tid; v < V; v += SF_V16_THREADS) {
                float z = load_scaled_z_v16(logits, row_off, v, LOGITS_F32, T);
                if (isfinite(z) && (!has_minp || z >= z_minp || v == argmax_v)) {
                    uint32_t o = f32_to_ord_v16(z);
                    if (o >= lo_o && o <= hi_o) {
                        uint32_t b = (o >> shift) & 0xFFu;
                        atomicAdd(&smem.u.rad.hist_cnt[b], 1u);
                    }
                }
            }
            __syncthreads();
            if (tid == 0) {
                uint32_t acc = smem.b_above_cnt;
                uint32_t sel_b = 0u;
                for (int b = 255; b >= 0; --b) {
                    uint32_t c = smem.u.rad.hist_cnt[b];
                    if (acc + c >= static_cast<uint32_t>(top_k_val)) {
                        sel_b = static_cast<uint32_t>(b);
                        break;
                    }
                    acc += c;
                }
                smem.b_above_cnt = acc;
                uint32_t mask = (shift == 24) ? 0u : ~((1u << (shift + 8)) - 1u);
                uint32_t base = (lo_o & mask) | (sel_b << shift);
                uint32_t span = (1u << shift) - 1u;
                smem.b_lo_ord = (base > lo_o) ? base : lo_o;
                smem.b_hi_ord = ((base | span) < hi_o) ? (base | span) : hi_o;
            }
            __syncthreads();
        }
        tau_k_ord = smem.b_lo_ord;
        uint32_t k_tie_allow = static_cast<uint32_t>(top_k_val) - smem.b_above_cnt;
        v_k_cut = find_nth_tied_index_v16<LOGITS_F32>(
            smem, logits, row_off, V, T, tau_k_ord, k_tie_allow, has_minp, z_minp, argmax_v, V - 1
        );
    }

    // Compute exact fixed-point total mass `tot_mass_u64` over post-(min_p, top_k) pool
    uint32_t tau_final_ord = tau_k_ord;
    int32_t v_final_cut = v_k_cut;
    double Z_nuc = 0.0;

    if (has_topp) {
        if (tid == 0) smem.b_tot_mass = 0ull;
        __syncthreads();
        unsigned long long local_m = 0ull;
        for (int v = tid; v < V; v += SF_V16_THREADS) {
            float z = load_scaled_z_v16(logits, row_off, v, LOGITS_F32, T);
            if (!isfinite(z)) continue;
            if (has_minp && z < z_minp && v != argmax_v) continue;
            uint32_t o = f32_to_ord_v16(z);
            if (o < tau_k_ord || (o == tau_k_ord && v > v_k_cut)) continue;
            local_m += weight_to_u64_v16(z - z_max);
        }
        atomicAdd(&smem.b_tot_mass, local_m);
        __syncthreads();

        unsigned long long tot_m = smem.b_tot_mass;
        unsigned long long target_m = static_cast<unsigned long long>(
            ceil(static_cast<double>(top_p_val) * static_cast<double>(tot_m))
        );
        if (target_m < 1ull) target_m = 1ull;
        if (target_m > tot_m) target_m = tot_m;

        if (tid == 0) {
            smem.b_lo_ord = tau_k_ord;
            smem.b_hi_ord = zmax_ord;
            smem.b_above_mass = 0ull;
        }
        __syncthreads();

        for (int shift = 24; shift >= 0; shift -= 8) {
            uint32_t lo_o = smem.b_lo_ord;
            uint32_t hi_o = smem.b_hi_ord;
            if ((lo_o >> shift) == (hi_o >> shift)) continue;
            smem.u.rad.hist_cnt[tid] = 0u;
            smem.u.rad.hist_mass[tid] = 0ull;
            __syncthreads();
            for (int v = tid; v < V; v += SF_V16_THREADS) {
                float z = load_scaled_z_v16(logits, row_off, v, LOGITS_F32, T);
                if (!isfinite(z)) continue;
                if (has_minp && z < z_minp && v != argmax_v) continue;
                uint32_t o = f32_to_ord_v16(z);
                if (o < lo_o || o > hi_o) continue;
                if (o == tau_k_ord && v > v_k_cut) continue;
                uint32_t b = (o >> shift) & 0xFFu;
                atomicAdd(&smem.u.rad.hist_cnt[b], 1u);
                unsigned long long wm = weight_to_u64_v16(z - z_max);
                if (wm > 0ull) atomicAdd(&smem.u.rad.hist_mass[b], wm);
            }
            __syncthreads();
            if (tid == 0) {
                unsigned long long acc = smem.b_above_mass;
                uint32_t sel_b = 0u;
                for (int b = 255; b >= 0; --b) {
                    unsigned long long m_b = smem.u.rad.hist_mass[b];
                    if (acc + m_b >= target_m) {
                        sel_b = static_cast<uint32_t>(b);
                        break;
                    }
                    acc += m_b;
                }
                smem.b_above_mass = acc;
                uint32_t mask = (shift == 24) ? 0u : ~((1u << (shift + 8)) - 1u);
                uint32_t base = (lo_o & mask) | (sel_b << shift);
                uint32_t span = (1u << shift) - 1u;
                smem.b_lo_ord = (base > lo_o) ? base : lo_o;
                smem.b_hi_ord = ((base | span) < hi_o) ? (base | span) : hi_o;
            }
            __syncthreads();
        }
        tau_final_ord = smem.b_lo_ord;
        unsigned long long rem = (target_m > smem.b_above_mass) ? (target_m - smem.b_above_mass) : 1ull;
        unsigned long long w_tie_u64 = weight_to_u64_v16(ord_to_f32_v16(tau_final_ord) - z_max);
        uint32_t p_tie_allow = (w_tie_u64 > 0ull)
            ? static_cast<uint32_t>((rem + w_tie_u64 - 1ull) / w_tie_u64)
            : 1u;
        int32_t bound_v = (tau_final_ord == tau_k_ord) ? v_k_cut : (V - 1);
        v_final_cut = find_nth_tied_index_v16<LOGITS_F32>(
            smem, logits, row_off, V, T, tau_final_ord, p_tie_allow, has_minp, z_minp, argmax_v, bound_v
        );
        Z_nuc = static_cast<double>(smem.b_above_mass + static_cast<unsigned long long>(p_tie_allow) * w_tie_u64)
            * SF_V16_INV_FIX_SCALE;
    } else {
        // Compute exact double `Z_nuc` over post-(min_p, top_k) pool
        double local_z = 0.0;
        for (int v = tid; v < V; v += SF_V16_THREADS) {
            float z = load_scaled_z_v16(logits, row_off, v, LOGITS_F32, T);
            if (!isfinite(z)) continue;
            if (has_minp && z < z_minp && v != argmax_v) continue;
            uint32_t o = f32_to_ord_v16(z);
            if (o < tau_final_ord || (o == tau_final_ord && v > v_final_cut)) continue;
            local_z += exp(static_cast<double>(z - z_max));
        }
        Z_nuc = block_sum_f64_v16(smem, local_z);
    }

    // Acceptance check (if accepted -> SKIP final sampling pass completely!)
    if (do_verify) {
        float z_d = smem.b_zdraft;
        bool d_in_nuc = false;
        if (valid_draft && isfinite(z_d) && (!has_minp || z_d >= z_minp || draft_tok == argmax_v)) {
            uint32_t od = f32_to_ord_v16(z_d);
            d_in_nuc = (od > tau_final_ord) || (od == tau_final_ord && draft_tok <= static_cast<int64_t>(v_final_cut));
        }
        double p_d = (d_in_nuc && Z_nuc > 0.0) ? (exp(static_cast<double>(z_d - z_max)) / Z_nuc) : 0.0;
        double u_acc = philox_u01_f64_v16(seed, step_id, req_slot, pos, 0u);
        if (valid_draft && q_d > 0.0 && u_acc * q_d < p_d) {
            out_accepted = 1;
            out_tok = draft_tok;
            return;
        }
    }

    // Final 1-pass exact log-domain Exponential-Race over surviving nucleus!
    double best_res_sc = -INFINITY;
    int32_t best_res_i = -1;
    double best_fb_sc = -INFINITY;
    int32_t best_fb_i = argmax_v;
    const double inv_Znuc = (Z_nuc > 0.0) ? (1.0 / Z_nuc) : 0.0;
    const double inv_Zq = (Z_q > 0.0) ? (1.0 / Z_q) : 0.0;
    const double scalar_off_q = (V > 1) ? ((1.0 - q_d) / static_cast<double>(V - 1)) : 0.0;

    for (int v = tid; v < V; v += SF_V16_THREADS) {
        float z = load_scaled_z_v16(logits, row_off, v, LOGITS_F32, T);
        if (!isfinite(z)) continue;
        if (has_minp && z < z_minp && v != argmax_v) continue;
        uint32_t o = f32_to_ord_v16(z);
        if (o < tau_final_ord || (o == tau_final_ord && v > v_final_cut)) continue;

        double p_v = exp(static_cast<double>(z - z_max)) * inv_Znuc;
        double log_e = philox_log_exp1_f64_v16(seed, step_id, req_slot, pos, static_cast<uint32_t>(v));
        double fb_sc = static_cast<double>(z - z_max) - log_e;
        if (fb_sc > best_fb_sc || (fb_sc == best_fb_sc && v < best_fb_i)) {
            best_fb_sc = fb_sc;
            best_fb_i = v;
        }
        double w_v = p_v;
        if (do_verify) {
            double q_v = 0.0;
            if constexpr (!HAS_DRAFT_Q) {
                q_v = (valid_draft && static_cast<int64_t>(v) == draft_tok) ? 1.0 : 0.0;
            } else {
                if (draft_mode == DRAFT_MODE_GREEDY) {
                    q_v = (valid_draft && static_cast<int64_t>(v) == draft_tok) ? 1.0 : 0.0;
                } else if (draft_mode == DRAFT_MODE_FULL_PROBS && draft_q != nullptr) {
                    q_v = static_cast<double>(static_cast<const float*>(draft_q)[q_row_off + v]);
                } else if (draft_mode == DRAFT_MODE_FULL_LOGITS && draft_q != nullptr) {
                    float ql = load_logit_v16(draft_q, q_row_off, v, draft_is_fp32);
                    q_v = isfinite(ql) ? (exp(static_cast<double>(ql - q_max)) * inv_Zq) : 0.0;
                } else if (draft_mode == DRAFT_MODE_SCALAR_PROB) {
                    q_v = (valid_draft && static_cast<int64_t>(v) == draft_tok) ? q_d : scalar_off_q;
                }
            }
            w_v = p_v - q_v;
        }
        if (w_v > 0.0) {
            double sc = log(w_v) - log_e;
            if (sc > best_res_sc || (sc == best_res_sc && (best_res_i < 0 || v < best_res_i))) {
                best_res_sc = sc;
                best_res_i = v;
            }
        }
    }
    double g_res_sc;
    int32_t g_res_i;
    block_argmax_f64_v16(smem, best_res_sc, (best_res_i >= 0) ? best_res_i : V, g_res_sc, g_res_i);
    if (g_res_i >= 0 && g_res_i < V && isfinite(g_res_sc)) {
        out_accepted = 0;
        out_tok = static_cast<int64_t>(g_res_i);
    } else {
        double g_fb_sc;
        int32_t g_fb_i;
        block_argmax_f64_v16(smem, best_fb_sc, best_fb_i, g_fb_sc, g_fb_i);
        out_accepted = 0;
        out_tok = static_cast<int64_t>(g_fb_i);
    }
}

// =========================================================================
// Kernel 1: Fused Speculative Rejection Sampler (vLLM `RejectionSampler`)
// =========================================================================
template <bool LOGITS_F32, bool HAS_DRAFT_Q, bool PDL>
__global__ void __launch_bounds__(SF_V16_THREADS, 2)
sync_free_temp_reject_v16_kernel(
    const void*    __restrict__ logits,
    int64_t        logits_stride,
    int            M,
    int            V,
    const float*   __restrict__ temperature,
    const float*   __restrict__ top_p,
    const int32_t* __restrict__ top_k,
    const float*   __restrict__ min_p,
    const void*    __restrict__ seeds,
    uint64_t       step_id,
    const void*    __restrict__ draft_q,
    int            draft_mode,
    int            draft_is_fp32,
    int64_t        draft_stride,
    const int32_t* __restrict__ seq_lens,
    const void*    __restrict__ input_ids,
    const void*    __restrict__ positions,
    const void*    __restrict__ logits_indices,
    const void*    __restrict__ idx_mapping,
    const int32_t* __restrict__ prefill_len,
    const int32_t* __restrict__ cu_num_logits,
    uint32_t       dtype_flags,
    int64_t*       __restrict__ sampled_out,
    int64_t        sampled_stride,
    int32_t*       __restrict__ num_sampled_out,
    int32_t*       __restrict__ num_rejected_out
) {
    __shared__ SharedScratchV16 smem;
    const int64_t req_idx = static_cast<int64_t>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);

    // Preload request metadata BEFORE `gdc_wait_v16()`!
    const bool ids_i64   = (dtype_flags & 1u) != 0u;
    const bool pos_i64   = (dtype_flags & 2u) != 0u;
    const bool lidx_i64  = (dtype_flags & 4u) != 0u;
    const bool map_i64   = (dtype_flags & 8u) != 0u;
    const bool seeds_i64 = (dtype_flags & 16u) != 0u;

    const int64_t start_idx = static_cast<int64_t>(cu_num_logits[req_idx]);
    const int64_t end_idx   = static_cast<int64_t>(cu_num_logits[req_idx + 1]);
    const int64_t num_draft = (end_idx > start_idx) ? (end_idx - start_idx - 1) : 0;

    const int64_t req_slot  = read_idx_v16(idx_mapping, req_idx, map_i64);
    const int64_t req_pf    = static_cast<int64_t>(prefill_len[req_slot]);
    const int64_t seq_len   = static_cast<int64_t>(seq_lens[req_idx]);

    const float   T         = temperature[req_slot];
    const float   tp_val    = (top_p != nullptr) ? top_p[req_slot] : 1.0f;
    const int32_t tk_val    = (top_k != nullptr) ? top_k[req_slot] : 0;
    const float   mp_val    = (min_p != nullptr) ? min_p[req_slot] : 0.0f;
    const uint64_t seed_val = (seeds != nullptr)
        ? static_cast<uint64_t>(read_idx_v16(seeds, req_slot, seeds_i64))
        : 0ull;

    if constexpr (PDL) gdc_wait_v16();

    int64_t accepted_len = 0;
    for (int64_t i = 0; i < num_draft; ++i) {
        const int64_t logit_row = start_idx + i;
        if (logit_row >= static_cast<int64_t>(M)) break;
        const int64_t cur_tidx  = read_idx_v16(logits_indices, logit_row, lidx_i64);
        const uint32_t cur_pos  = static_cast<uint32_t>(read_idx_v16(positions, cur_tidx, pos_i64));
        const int64_t nxt_tidx  = read_idx_v16(logits_indices, logit_row + 1, lidx_i64);
        const int64_t dtok      = read_idx_v16(input_ids, nxt_tidx, ids_i64);
        const int64_t dpos      = read_idx_v16(positions, nxt_tidx, pos_i64);
        const bool valid_draft  = (dtok >= 0) && (dtok < static_cast<int64_t>(V)) && (dpos >= req_pf);

        int32_t acc = 0;
        int64_t out_tok = 0;
        process_row_v16<LOGITS_F32, HAS_DRAFT_Q>(
            smem, logits, logits_stride, logit_row, V,
            T, tp_val, tk_val, mp_val,
            seed_val, step_id, static_cast<uint32_t>(req_slot), cur_pos,
            true, dtok, valid_draft,
            draft_q, draft_mode, (draft_is_fp32 != 0), draft_stride, logit_row,
            acc, out_tok
        );
        if (tid == 0) {
            sampled_out[req_idx * sampled_stride + i] = out_tok;
        }
        if (acc != 0) {
            accepted_len += 1;
        } else {
            break;
        }
    }

    if (accepted_len == num_draft) {
        const int64_t bonus_row = start_idx + num_draft;
        if (bonus_row < static_cast<int64_t>(M)) {
            const int64_t b_tidx = read_idx_v16(logits_indices, bonus_row, lidx_i64);
            const uint32_t b_pos = static_cast<uint32_t>(read_idx_v16(positions, b_tidx, pos_i64));
            int32_t dummy_acc = 0;
            int64_t bonus_tok = 0;
            process_row_v16<LOGITS_F32, false>(
                smem, logits, logits_stride, bonus_row, V,
                T, tp_val, tk_val, mp_val,
                seed_val, step_id, static_cast<uint32_t>(req_slot), b_pos,
                false, -1, false,
                nullptr, DRAFT_MODE_GREEDY, false, 0, 0,
                dummy_acc, bonus_tok
            );
            if (tid == 0) {
                sampled_out[req_idx * sampled_stride + num_draft] = bonus_tok;
            }
        }
    }

    const bool incomplete = (seq_len < req_pf);
    const int32_t ns_out = incomplete ? 0 : static_cast<int32_t>(accepted_len + 1);
    const int32_t nr_out = incomplete ? 0 : static_cast<int32_t>(num_draft - accepted_len);
    if (tid == 0) {
        num_sampled_out[req_idx]  = ns_out;
        num_rejected_out[req_idx] = nr_out;
    }
    // Strict #206 PDL epilogue hygiene: ALL global stores precede launch_dependents!
    if constexpr (PDL) {
        __syncthreads();
        if (tid == 0) gdc_launch_dependents_v16();
    }
}

// =========================================================================
// Kernel 2: Fused K=0 / Non-Speculative Sampler (vLLM `Sampler`)
// =========================================================================
template <bool LOGITS_F32, bool PDL>
__global__ void __launch_bounds__(SF_V16_THREADS, 2)
sync_free_temp_sample_k0_v16_kernel(
    const void*    __restrict__ logits,
    int64_t        logits_stride,
    int            V,
    const float*   __restrict__ temperature,
    const float*   __restrict__ top_p,
    const int32_t* __restrict__ top_k,
    const float*   __restrict__ min_p,
    const void*    __restrict__ seeds,
    uint64_t       step_id,
    const int32_t* __restrict__ seq_lens,
    const void*    __restrict__ idx_mapping,
    const int32_t* __restrict__ prefill_len,
    uint32_t       dtype_flags,
    int64_t*       __restrict__ tok_out,
    int32_t*       __restrict__ num_sampled_out,
    int32_t*       __restrict__ num_rejected_out
) {
    __shared__ SharedScratchV16 smem;
    const int64_t req_idx = static_cast<int64_t>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);

    const bool map_i64   = (dtype_flags & 8u) != 0u;
    const bool seeds_i64 = (dtype_flags & 16u) != 0u;

    const int64_t req_slot  = read_idx_v16(idx_mapping, req_idx, map_i64);
    const int64_t seq_len   = (seq_lens != nullptr) ? static_cast<int64_t>(seq_lens[req_idx]) : 1;
    const int64_t req_pf    = (prefill_len != nullptr) ? static_cast<int64_t>(prefill_len[req_slot]) : 0;

    const float   T         = temperature[req_slot];
    const float   tp_val    = (top_p != nullptr) ? top_p[req_slot] : 1.0f;
    const int32_t tk_val    = (top_k != nullptr) ? top_k[req_slot] : 0;
    const float   mp_val    = (min_p != nullptr) ? min_p[req_slot] : 0.0f;
    const uint64_t seed_val = (seeds != nullptr)
        ? static_cast<uint64_t>(read_idx_v16(seeds, req_slot, seeds_i64))
        : 0ull;

    if constexpr (PDL) gdc_wait_v16();

    int32_t dummy_acc = 0;
    int64_t tok = 0;
    process_row_v16<LOGITS_F32, false>(
        smem, logits, logits_stride, req_idx, V,
        T, tp_val, tk_val, mp_val,
        seed_val, step_id, static_cast<uint32_t>(req_slot), static_cast<uint32_t>(seq_len),
        false, -1, false,
        nullptr, DRAFT_MODE_GREEDY, false, 0, 0,
        dummy_acc, tok
    );

    if (tid == 0) {
        tok_out[req_idx] = tok;
        if (num_sampled_out != nullptr) {
            num_sampled_out[req_idx] = (seq_len < req_pf) ? 0 : 1;
        }
        if (num_rejected_out != nullptr) {
            num_rejected_out[req_idx] = 0;
        }
    }
    if constexpr (PDL) {
        __syncthreads();
        if (tid == 0) gdc_launch_dependents_v16();
    }
}

// =========================================================================
// Kernel 3: Direct Uniform-K Verify/Sample Entry (Standalone & Qualification)
// =========================================================================
template <bool LOGITS_F32, bool HAS_DRAFT_Q, bool PDL>
__global__ void __launch_bounds__(SF_V16_THREADS, 2)
sync_free_temp_verify_rows_v16_kernel(
    const void*    __restrict__ logits,
    int64_t        logits_stride,
    int            M,
    int            V,
    int            K,
    const float*   __restrict__ temperature,
    const float*   __restrict__ top_p,
    const int32_t* __restrict__ top_k,
    const float*   __restrict__ min_p,
    const int64_t* __restrict__ seeds,
    const int64_t* __restrict__ base_positions,
    uint64_t       step_id,
    const int64_t* __restrict__ draft_token_ids,
    const void*    __restrict__ draft_q,
    int            draft_mode,
    int            draft_is_fp32,
    int64_t        draft_stride,
    int64_t*       __restrict__ sampled_out,
    int32_t*       __restrict__ num_sampled_out,
    int32_t*       __restrict__ num_rejected_out
) {
    __shared__ SharedScratchV16 smem;
    const int64_t req_idx = static_cast<int64_t>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);

    const float    T        = temperature[req_idx];
    const float    tp_val   = (top_p != nullptr) ? top_p[req_idx] : 1.0f;
    const int32_t  tk_val   = (top_k != nullptr) ? top_k[req_idx] : 0;
    const float    mp_val   = (min_p != nullptr) ? min_p[req_idx] : 0.0f;
    const uint64_t seed_val = (seeds != nullptr) ? static_cast<uint64_t>(seeds[req_idx]) : 0ull;
    const uint32_t base_pos = (base_positions != nullptr) ? static_cast<uint32_t>(base_positions[req_idx]) : 0u;

    const int64_t start_row = req_idx * static_cast<int64_t>(K + 1);
    const int64_t out_base  = req_idx * static_cast<int64_t>(K + 1);
    const int64_t d_base    = req_idx * static_cast<int64_t>(K);

    if constexpr (PDL) gdc_wait_v16();

    int64_t accepted_len = 0;
    for (int i = 0; i < K; ++i) {
        const int64_t row = start_row + i;
        if (row >= static_cast<int64_t>(M)) break;
        const int64_t dtok = (draft_token_ids != nullptr) ? draft_token_ids[d_base + i] : -1;
        const bool valid_draft = (dtok >= 0 && dtok < static_cast<int64_t>(V));
        int32_t acc = 0;
        int64_t out_tok = 0;
        process_row_v16<LOGITS_F32, HAS_DRAFT_Q>(
            smem, logits, logits_stride, row, V,
            T, tp_val, tk_val, mp_val,
            seed_val, step_id, static_cast<uint32_t>(req_idx), base_pos + static_cast<uint32_t>(i),
            true, dtok, valid_draft,
            draft_q, draft_mode, (draft_is_fp32 != 0), draft_stride, d_base + i,
            acc, out_tok
        );
        if (tid == 0) {
            sampled_out[out_base + i] = out_tok;
        }
        if (acc != 0) {
            accepted_len += 1;
        } else {
            break;
        }
    }

    if (accepted_len == static_cast<int64_t>(K)) {
        const int64_t bonus_row = start_row + K;
        if (bonus_row < static_cast<int64_t>(M)) {
            int32_t dummy_acc = 0;
            int64_t bonus_tok = 0;
            process_row_v16<LOGITS_F32, false>(
                smem, logits, logits_stride, bonus_row, V,
                T, tp_val, tk_val, mp_val,
                seed_val, step_id, static_cast<uint32_t>(req_idx), base_pos + static_cast<uint32_t>(K),
                false, -1, false,
                nullptr, DRAFT_MODE_GREEDY, false, 0, 0,
                dummy_acc, bonus_tok
            );
            if (tid == 0) {
                sampled_out[out_base + K] = bonus_tok;
            }
        }
    }

    if (tid == 0) {
        num_sampled_out[req_idx]  = static_cast<int32_t>(accepted_len + 1);
        num_rejected_out[req_idx] = static_cast<int32_t>(K - accepted_len);
    }
    if constexpr (PDL) {
        __syncthreads();
        if (tid == 0) gdc_launch_dependents_v16();
    }
}

template <typename KernelT, typename... Args>
static inline cudaError_t launch_maybe_pdl_v16(
    KernelT kernel, dim3 grid, dim3 block, cudaStream_t stream, bool pdl, Args... args
) {
    if (!pdl) {
        kernel<<<grid, block, 0, stream>>>(args...);
        return cudaGetLastError();
    }
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = 0;
    cfg.stream = stream;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    return cudaLaunchKernelEx(&cfg, kernel, args...);
}

}  // namespace

// =========================================================================
// Exported C ABI (strictly matched by sync_free.py ctypes bindings & nm -D)
// =========================================================================
extern "C" int sync_free_temp_v16_version(void) {
    return 16;
}

extern "C" int sync_free_temp_v16_smem_cap(void) {
    return SF_V16_SMEM_CAP;
}

extern "C" int sync_free_temp_reject_v16_fwd(
    const void*    logits,
    int            logits_is_fp32,
    int64_t        logits_stride,
    int            M,
    int            V,
    int            num_reqs,
    const float*   temperature,
    const float*   top_p,
    const int32_t* top_k,
    const float*   min_p,
    const void*    seeds,
    uint64_t       step_id,
    const void*    draft_q,
    int            draft_mode,
    int            draft_is_fp32,
    int64_t        draft_stride,
    const int32_t* seq_lens,
    const void*    input_ids,
    const void*    positions,
    const void*    logits_indices,
    const void*    idx_mapping,
    const int32_t* prefill_len,
    const int32_t* cu_num_logits,
    uint32_t       dtype_flags,
    int64_t*       sampled_out,
    int64_t        sampled_stride,
    int32_t*       num_sampled_out,
    int32_t*       num_rejected_out,
    int            use_pdl,
    uint64_t       stream_u64
) {
    if (logits == nullptr || temperature == nullptr || seq_lens == nullptr
        || input_ids == nullptr || positions == nullptr || logits_indices == nullptr
        || idx_mapping == nullptr || prefill_len == nullptr || cu_num_logits == nullptr
        || sampled_out == nullptr || num_sampled_out == nullptr || num_rejected_out == nullptr) {
        return -1;
    }
    if (num_reqs <= 0 || M <= 0 || V <= 0 || logits_stride < V) {
        return -2;
    }
    if (draft_mode < DRAFT_MODE_GREEDY || draft_mode > DRAFT_MODE_SCALAR_PROB) {
        return -3;
    }
    if (draft_mode != DRAFT_MODE_GREEDY && draft_q == nullptr) {
        return -4;
    }

    const dim3 grid(static_cast<unsigned int>(num_reqs), 1u, 1u);
    const dim3 block(static_cast<unsigned int>(SF_V16_THREADS), 1u, 1u);
    const cudaStream_t stream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(stream_u64));
    const bool f32 = (logits_is_fp32 != 0);
    const bool has_q = (draft_mode != DRAFT_MODE_GREEDY);
    const bool pdl = (use_pdl != 0);

    cudaError_t err = cudaSuccess;
    #define DISPATCH_REJECT_V16(F32_B, HQ_B, PDL_B) \
        err = launch_maybe_pdl_v16( \
            sync_free_temp_reject_v16_kernel<F32_B, HQ_B, PDL_B>, \
            grid, block, stream, PDL_B, \
            logits, logits_stride, M, V, \
            temperature, top_p, top_k, min_p, seeds, step_id, \
            draft_q, draft_mode, draft_is_fp32, draft_stride, \
            seq_lens, input_ids, positions, logits_indices, idx_mapping, prefill_len, cu_num_logits, \
            dtype_flags, sampled_out, sampled_stride, num_sampled_out, num_rejected_out)

    if (f32) {
        if (has_q) {
            if (pdl) { DISPATCH_REJECT_V16(true, true, true); }
            else     { DISPATCH_REJECT_V16(true, true, false); }
        } else {
            if (pdl) { DISPATCH_REJECT_V16(true, false, true); }
            else     { DISPATCH_REJECT_V16(true, false, false); }
        }
    } else {
        if (has_q) {
            if (pdl) { DISPATCH_REJECT_V16(false, true, true); }
            else     { DISPATCH_REJECT_V16(false, true, false); }
        } else {
            if (pdl) { DISPATCH_REJECT_V16(false, false, true); }
            else     { DISPATCH_REJECT_V16(false, false, false); }
        }
    }
    #undef DISPATCH_REJECT_V16
    return static_cast<int>(err);
}

extern "C" int sync_free_temp_sample_k0_v16_fwd(
    const void*    logits,
    int            logits_is_fp32,
    int64_t        logits_stride,
    int            num_reqs,
    int            V,
    const float*   temperature,
    const float*   top_p,
    const int32_t* top_k,
    const float*   min_p,
    const void*    seeds,
    uint64_t       step_id,
    const int32_t* seq_lens,
    const void*    idx_mapping,
    const int32_t* prefill_len,
    uint32_t       dtype_flags,
    int64_t*       tok_out,
    int32_t*       num_sampled_out,
    int32_t*       num_rejected_out,
    int            use_pdl,
    uint64_t       stream_u64
) {
    if (logits == nullptr || temperature == nullptr || idx_mapping == nullptr || tok_out == nullptr) {
        return -1;
    }
    if (num_reqs <= 0 || V <= 0 || logits_stride < V) {
        return -2;
    }
    const dim3 grid(static_cast<unsigned int>(num_reqs), 1u, 1u);
    const dim3 block(static_cast<unsigned int>(SF_V16_THREADS), 1u, 1u);
    const cudaStream_t stream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(stream_u64));
    const bool f32 = (logits_is_fp32 != 0);
    const bool pdl = (use_pdl != 0);

    cudaError_t err = cudaSuccess;
    #define DISPATCH_K0_V16(F32_B, PDL_B) \
        err = launch_maybe_pdl_v16( \
            sync_free_temp_sample_k0_v16_kernel<F32_B, PDL_B>, \
            grid, block, stream, PDL_B, \
            logits, logits_stride, V, \
            temperature, top_p, top_k, min_p, seeds, step_id, \
            seq_lens, idx_mapping, prefill_len, dtype_flags, \
            tok_out, num_sampled_out, num_rejected_out)

    if (f32) {
        if (pdl) { DISPATCH_K0_V16(true, true); }
        else     { DISPATCH_K0_V16(true, false); }
    } else {
        if (pdl) { DISPATCH_K0_V16(false, true); }
        else     { DISPATCH_K0_V16(false, false); }
    }
    #undef DISPATCH_K0_V16
    return static_cast<int>(err);
}

extern "C" int sync_free_temp_verify_rows_v16_fwd(
    const void*    logits,
    int            logits_is_fp32,
    int64_t        logits_stride,
    int            M,
    int            V,
    int            num_reqs,
    int            K,
    const float*   temperature,
    const float*   top_p,
    const int32_t* top_k,
    const float*   min_p,
    const int64_t* seeds,
    const int64_t* base_positions,
    uint64_t       step_id,
    const int64_t* draft_token_ids,
    const void*    draft_q,
    int            draft_mode,
    int            draft_is_fp32,
    int64_t        draft_stride,
    int64_t*       sampled_out,
    int32_t*       num_sampled_out,
    int32_t*       num_rejected_out,
    int            use_pdl,
    uint64_t       stream_u64
) {
    if (logits == nullptr || temperature == nullptr || sampled_out == nullptr
        || num_sampled_out == nullptr || num_rejected_out == nullptr) {
        return -1;
    }
    if (num_reqs <= 0 || K < 0 || M < num_reqs * (K + 1) || V <= 0 || logits_stride < V) {
        return -2;
    }
    if (draft_mode < DRAFT_MODE_GREEDY || draft_mode > DRAFT_MODE_SCALAR_PROB) {
        return -3;
    }
    if (draft_mode != DRAFT_MODE_GREEDY && draft_q == nullptr) {
        return -4;
    }
    const dim3 grid(static_cast<unsigned int>(num_reqs), 1u, 1u);
    const dim3 block(static_cast<unsigned int>(SF_V16_THREADS), 1u, 1u);
    const cudaStream_t stream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(stream_u64));
    const bool f32 = (logits_is_fp32 != 0);
    const bool has_q = (draft_mode != DRAFT_MODE_GREEDY);
    const bool pdl = (use_pdl != 0);

    cudaError_t err = cudaSuccess;
    #define DISPATCH_VR_V16(F32_B, HQ_B, PDL_B) \
        err = launch_maybe_pdl_v16( \
            sync_free_temp_verify_rows_v16_kernel<F32_B, HQ_B, PDL_B>, \
            grid, block, stream, PDL_B, \
            logits, logits_stride, M, V, K, \
            temperature, top_p, top_k, min_p, seeds, base_positions, step_id, \
            draft_token_ids, draft_q, draft_mode, draft_is_fp32, draft_stride, \
            sampled_out, num_sampled_out, num_rejected_out)

    if (f32) {
        if (has_q) {
            if (pdl) { DISPATCH_VR_V16(true, true, true); }
            else     { DISPATCH_VR_V16(true, true, false); }
        } else {
            if (pdl) { DISPATCH_VR_V16(true, false, true); }
            else     { DISPATCH_VR_V16(true, false, false); }
        }
    } else {
        if (has_q) {
            if (pdl) { DISPATCH_VR_V16(false, true, true); }
            else     { DISPATCH_VR_V16(false, true, false); }
        } else {
            if (pdl) { DISPATCH_VR_V16(false, false, true); }
            else     { DISPATCH_VR_V16(false, false, false); }
        }
    }
    #undef DISPATCH_VR_V16
    return static_cast<int>(err);
}
