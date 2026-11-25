// SPDX-License-Identifier: Apache-2.0
// Wave-V13 Split TU 1/5: Unified KV-Conv + Q-Clip Prologue (`attn_prologue_v13.cu`)
//   96 compact 96-thread kernels (BF16 + SWAR KV-BF12, C=1 / C=2..32 / Beam-2 Tree, PDL=0/1).
//   Self-contained TU: compiled with `nvcc -c` (no `-dc`) in parallel with SWA & Global TUs.

#define ATTN_V13_DEVICE_TU 1
#include "attn_common_v13.cuh"

namespace attn_v13 {
namespace {

template <bool IS_V, bool DYNAMIC, bool HAS_NLIVE, bool USE_PDL, bool HAS_TREE, bool KV_BF12>
static __device__ __forceinline__ void run_kv_tok_warp(
    int t0, int t_stride,
    const uint16_t* __restrict__ X,
    const uint16_t* __restrict__ W,
    const uint16_t* __restrict__ Dyn,
    uint16_t* __restrict__ ConvCache,
    uint16_t* __restrict__ KVCache,
    const int64_t* __restrict__ KVSlot,
    const int64_t* __restrict__ Pos,
    const int32_t* __restrict__ SeqIdx,
    const int64_t* __restrict__ Slot,
    const int32_t* __restrict__ BlockTable,
    const int32_t* __restrict__ QStart,
    const int32_t* __restrict__ NLivePtr,
    const int32_t* __restrict__ TreeSconvAnc,
    int T, int64_t stride_x, int64_t stride_w0, int64_t stride_w1, int64_t stride_dyn,
    int64_t stride_c_blk, int64_t stride_c_n, int64_t stride_kv_blk, int64_t stride_kv_n,
    int kv_bs, int kv_shift, int64_t stride_bt, int max_blocks, int N_cache, int N_shift,
    float scale, float clip_threshold, float clip_eps, float norm_eps,
    uint16_t* __restrict__ EmergRaw,
    uint32_t* __restrict__ EmergUsed,
    uint32_t emerg_cap
) {
    int nl = T;
    if constexpr (HAS_NLIVE) {
        int v = *NLivePtr;
        if (v < nl) nl = v;
    } else {
        (void)NLivePtr;
    }
    if constexpr (!HAS_TREE) { (void)TreeSconvAnc; }
    if constexpr (!DYNAMIC)  { (void)Dyn; (void)stride_dyn; }
    if constexpr (!KV_BF12)  { (void)EmergRaw; (void)EmergUsed; (void)emerg_cap; }
    else                     { (void)stride_kv_n; }
    if constexpr (IS_V)      { (void)clip_threshold; (void)clip_eps; }
    else                     { (void)norm_eps; }

    if (t0 >= nl) {
        if constexpr (USE_PDL) { gdc_wait(); }
        return;
    }

    const int lane = threadIdx.x & 31;
    const int c0 = lane * 4, c1 = c0 + 1, c2 = c0 + 2, c3 = c0 + 3;
    const int64_t cache_off = IS_V ? 128 : 0;

    float w_t[4][4];
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        if constexpr (DYNAMIC) {
            w_t[k][0] = bf16_to_f32(W[k * stride_w0 + c0 * stride_w1]);
            w_t[k][1] = bf16_to_f32(W[k * stride_w0 + c1 * stride_w1]);
            w_t[k][2] = bf16_to_f32(W[k * stride_w0 + c2 * stride_w1]);
            w_t[k][3] = bf16_to_f32(W[k * stride_w0 + c3 * stride_w1]);
        } else {
            w_t[k][0] = bf16_to_f32(W[c0 * stride_w0 + k * stride_w1]);
            w_t[k][1] = bf16_to_f32(W[c1 * stride_w0 + k * stride_w1]);
            w_t[k][2] = bf16_to_f32(W[c2 * stride_w0 + k * stride_w1]);
            w_t[k][3] = bf16_to_f32(W[c3 * stride_w0 + k * stride_w1]);
        }
    }

    const int64_t pos0 = Pos[t0], slot0 = Slot[t0], kvslot0 = KVSlot[t0];
    const int32_t req0 = SeqIdx[t0], qs0 = QStart[t0];
    const bool valid0 = (slot0 >= 0);

    int anc0[4] = {t0, t0 - 1, t0 - 2, t0 - 3};
    if constexpr (HAS_TREE) {
        const int32_t* ap = TreeSconvAnc + t0 * 4;
        anc0[0] = ap[0]; anc0[1] = ap[1]; anc0[2] = ap[2]; anc0[3] = ap[3];
    }

    float pre_c[4][4];
    bool has_pre[4] = {false, false, false, false};
    if (valid0) {
        const int32_t* bt_row = BlockTable + static_cast<int64_t>(req0) * stride_bt;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            constexpr int lag = DYNAMIC ? 0 : 3;
            const int cur_lag = DYNAMIC ? k : (lag - k);
            if (cur_lag > 0) {
                const int64_t src = pos0 - cur_lag;
                const bool is_cached = HAS_TREE ? (anc0[cur_lag] < 0) : (t0 - cur_lag < qs0);
                if (src >= 0 && is_cached) {
                    has_pre[k] = true;
                    int64_t lblk, rem;
                    divmod_p2(src, N_cache, N_shift, lblk, rem);
                    if (lblk >= max_blocks) lblk = max_blocks - 1;
                    const int64_t blk = static_cast<int64_t>(bt_row[lblk]);
                    const uint16_t* cp = ConvCache + blk * stride_c_blk + rem * stride_c_n + cache_off;
                    const uint32_t p01 = *reinterpret_cast<const uint32_t*>(cp + c0);
                    const uint32_t p23 = *reinterpret_cast<const uint32_t*>(cp + c2);
                    pre_c[k][0] = __fadd_rn(0.f, bf16_to_f32(static_cast<uint16_t>(p01 & 0xFFFFu)));
                    pre_c[k][1] = __fadd_rn(0.f, bf16_to_f32(static_cast<uint16_t>(p01 >> 16)));
                    pre_c[k][2] = __fadd_rn(0.f, bf16_to_f32(static_cast<uint16_t>(p23 & 0xFFFFu)));
                    pre_c[k][3] = __fadd_rn(0.f, bf16_to_f32(static_cast<uint16_t>(p23 >> 16)));
                }
            }
        }
    }

    if constexpr (USE_PDL) { gdc_wait(); }

    for (int t = t0; t < nl; t += t_stride) {
        const bool is_first = (t == t0);
        const int64_t pos = is_first ? pos0 : Pos[t];
        const int64_t sl  = is_first ? slot0 : Slot[t];
        const int64_t kvs = is_first ? kvslot0 : KVSlot[t];
        const int32_t req = is_first ? req0 : SeqIdx[t];
        const int32_t qs  = is_first ? qs0 : QStart[t];
        const bool valid  = (sl >= 0);

        const uint16_t* xp = X + static_cast<int64_t>(t) * stride_x;
        const uint32_t cur01 = *reinterpret_cast<const uint32_t*>(xp + c0);
        const uint32_t cur23 = *reinterpret_cast<const uint32_t*>(xp + c2);
        float cur[4] = {
            bf16_to_f32(static_cast<uint16_t>(cur01 & 0xFFFFu)),
            bf16_to_f32(static_cast<uint16_t>(cur01 >> 16)),
            bf16_to_f32(static_cast<uint16_t>(cur23 & 0xFFFFu)),
            bf16_to_f32(static_cast<uint16_t>(cur23 >> 16)),
        };

        if (valid) {
            int64_t dblk, drem;
            divmod_p2(sl, N_cache, N_shift, dblk, drem);
            uint16_t* dst_c = ConvCache + dblk * stride_c_blk + drem * stride_c_n + cache_off;
            *reinterpret_cast<uint32_t*>(dst_c + c0) = cur01;
            *reinterpret_cast<uint32_t*>(dst_c + c2) = cur23;
        }

        float dyn_w[4] = {0.f, 0.f, 0.f, 0.f};
        if constexpr (DYNAMIC) {
            const uint16_t* dp = Dyn + static_cast<int64_t>(t) * stride_dyn;
            #pragma unroll
            for (int k = 0; k < 4; ++k) dyn_w[k] = bf16_to_f32(dp[k]);
        }

        float acc[4] = {0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int lag = DYNAMIC ? k : (3 - k);
            const int64_t src = pos - lag;
            const bool in_win = valid && (src >= 0);
            int row;
            bool intra, cached;
            if constexpr (HAS_TREE) {
                row = is_first ? anc0[lag] : TreeSconvAnc[t * 4 + lag];
                intra = in_win && (row >= 0);
                cached = in_win && (row < 0);
            } else {
                row = t - lag;
                intra = in_win && (row >= qs);
                cached = in_win && (row < qs);
            }
            float xt[4] = {0.f, 0.f, 0.f, 0.f};
            if (intra) {
                if (lag == 0) {
                    xt[0] = __fadd_rn(cur[0], 0.f); xt[1] = __fadd_rn(cur[1], 0.f);
                    xt[2] = __fadd_rn(cur[2], 0.f); xt[3] = __fadd_rn(cur[3], 0.f);
                } else {
                    const uint16_t* rx = X + static_cast<int64_t>(row) * stride_x;
                    const uint32_t r01 = *reinterpret_cast<const uint32_t*>(rx + c0);
                    const uint32_t r23 = *reinterpret_cast<const uint32_t*>(rx + c2);
                    xt[0] = __fadd_rn(bf16_to_f32(static_cast<uint16_t>(r01 & 0xFFFFu)), 0.f);
                    xt[1] = __fadd_rn(bf16_to_f32(static_cast<uint16_t>(r01 >> 16)), 0.f);
                    xt[2] = __fadd_rn(bf16_to_f32(static_cast<uint16_t>(r23 & 0xFFFFu)), 0.f);
                    xt[3] = __fadd_rn(bf16_to_f32(static_cast<uint16_t>(r23 >> 16)), 0.f);
                }
            } else if (cached) {
                if (is_first && has_pre[k]) {
                    xt[0] = pre_c[k][0]; xt[1] = pre_c[k][1]; xt[2] = pre_c[k][2]; xt[3] = pre_c[k][3];
                } else {
                    int64_t lblk, rem;
                    divmod_p2(src, N_cache, N_shift, lblk, rem);
                    if (lblk >= max_blocks) lblk = max_blocks - 1;
                    const int64_t blk = static_cast<int64_t>(BlockTable[static_cast<int64_t>(req) * stride_bt + lblk]);
                    const uint16_t* cp = ConvCache + blk * stride_c_blk + rem * stride_c_n + cache_off;
                    const uint32_t r01 = *reinterpret_cast<const uint32_t*>(cp + c0);
                    const uint32_t r23 = *reinterpret_cast<const uint32_t*>(cp + c2);
                    xt[0] = __fadd_rn(0.f, bf16_to_f32(static_cast<uint16_t>(r01 & 0xFFFFu)));
                    xt[1] = __fadd_rn(0.f, bf16_to_f32(static_cast<uint16_t>(r01 >> 16)));
                    xt[2] = __fadd_rn(0.f, bf16_to_f32(static_cast<uint16_t>(r23 & 0xFFFFu)));
                    xt[3] = __fadd_rn(0.f, bf16_to_f32(static_cast<uint16_t>(r23 >> 16)));
                }
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float wk = DYNAMIC ? __fadd_rn(w_t[k][i], dyn_w[k]) : w_t[k][i];
                if (DYNAMIC && k == 0) acc[i] = __fmul_rn(xt[i], wk);
                else acc[i] = fma_rn_f32(xt[i], wk, acc[i]);
            }
        }

        float y[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            y[i] = bf16_to_f32(f32_to_bf16(__fmul_rn(__fadd_rn(acc[i], cur[i]), scale)));
        }
        const float s0 = fma_rn_f32(y[0], y[0], __fmul_rn(y[2], y[2]));
        const float s1 = fma_rn_f32(y[1], y[1], __fmul_rn(y[3], y[3]));
        float sum_sq = __fadd_rn(s0, s1);
        sum_sq = __fadd_rn(sum_sq, __shfl_xor_sync(0xFFFFFFFFu, sum_sq, 16));
        sum_sq = __fadd_rn(sum_sq, __shfl_xor_sync(0xFFFFFFFFu, sum_sq, 8));
        sum_sq = __fadd_rn(sum_sq, __shfl_xor_sync(0xFFFFFFFFu, sum_sq, 4));
        sum_sq = __fadd_rn(sum_sq, __shfl_xor_sync(0xFFFFFFFFu, sum_sq, 2));
        sum_sq = __fadd_rn(sum_sq, __shfl_xor_sync(0xFFFFFFFFu, sum_sq, 1));

        float mult;
        if constexpr (IS_V) {
            mult = rsqrt_approx_f32(__fadd_rn(__fdiv_rn(sum_sq, 128.0f), norm_eps));
        } else {
            const float norm = (sum_sq > 0.0f) ? __fmul_rn(sum_sq, rsqrt_approx_f32(sum_sq)) : 0.0f;
            mult = fminf(__fdiv_rn(clip_threshold, __fadd_rn(norm, clip_eps)), 1.0f);
        }

        if (kvs >= 0) {
            int64_t kv_blk, kv_rem;
            divmod_p2(kvs, kv_bs, kv_shift, kv_blk, kv_rem);
            const uint16_t b0 = f32_to_bf16(__fmul_rn(y[0], mult));
            const uint16_t b1 = f32_to_bf16(__fmul_rn(y[1], mult));
            const uint16_t b2 = f32_to_bf16(__fmul_rn(y[2], mult));
            const uint16_t b3 = f32_to_bf16(__fmul_rn(y[3], mult));
            if constexpr (KV_BF12) {
                KvBf12HalfBlock* blk = kv_bf12_half_blk(
                    reinterpret_cast<uint8_t*>(KVCache), stride_kv_blk, kv_blk
                );
                kv_bf12_encode_warp_store(
                    b0, b1, b2, b3, lane, blk, static_cast<int>(kv_rem),
                    EmergRaw, EmergUsed, emerg_cap
                );
            } else {
                uint16_t* kv_dst = KVCache + kv_blk * stride_kv_blk + kv_rem * stride_kv_n;
                *reinterpret_cast<uint32_t*>(kv_dst + c0) = pack_bf16x2(b0, b1);
                *reinterpret_cast<uint32_t*>(kv_dst + c2) = pack_bf16x2(b2, b3);
            }
        }
    }
}

template <bool HAS_TAU, bool HAS_NLIVE, bool USE_PDL>
static __device__ __forceinline__ void run_q_tok_cta(
    int t0, int t_stride,
    const uint16_t* __restrict__ Q,
    const uint16_t* __restrict__ QScale,
    const float* __restrict__ Tau,
    uint16_t* __restrict__ QOut,
    const int32_t* __restrict__ NLivePtr,
    int T, int64_t stride_q, float clip_threshold, float clip_eps
) {
    if constexpr (!HAS_TAU)   { (void)Tau; }
    if constexpr (!HAS_NLIVE) { (void)NLivePtr; }
    if (threadIdx.x >= 96) {
        if constexpr (USE_PDL) { gdc_wait(); }
        return;
    }
    int nl = T;
    if constexpr (HAS_NLIVE) {
        int v = *NLivePtr;
        if (v < nl) nl = v;
    }
    if (t0 >= nl) {
        if constexpr (USE_PDL) { gdc_wait(); }
        return;
    }

    const int h = threadIdx.x >> 4;
    const int l = threadIdx.x & 15;
    const int d0 = l * 8;
    float sc[8];
    {
        const uint4 sv = *reinterpret_cast<const uint4*>(QScale + h * 128 + d0);
        sc[0] = bf16_to_f32(static_cast<uint16_t>(sv.x & 0xFFFFu));
        sc[1] = bf16_to_f32(static_cast<uint16_t>(sv.x >> 16));
        sc[2] = bf16_to_f32(static_cast<uint16_t>(sv.y & 0xFFFFu));
        sc[3] = bf16_to_f32(static_cast<uint16_t>(sv.y >> 16));
        sc[4] = bf16_to_f32(static_cast<uint16_t>(sv.z & 0xFFFFu));
        sc[5] = bf16_to_f32(static_cast<uint16_t>(sv.z >> 16));
        sc[6] = bf16_to_f32(static_cast<uint16_t>(sv.w & 0xFFFFu));
        sc[7] = bf16_to_f32(static_cast<uint16_t>(sv.w >> 16));
    }

    if constexpr (USE_PDL) { gdc_wait(); }

    const bool q_aligned16 = ((stride_q & 7) == 0) && ((reinterpret_cast<uintptr_t>(Q) & 15) == 0);
    for (int t = t0; t < nl; t += t_stride) {
        const uint16_t* qp = Q + static_cast<int64_t>(t) * stride_q + h * 128 + d0;
        uint4 qv;
        if (q_aligned16) {
            qv = *reinterpret_cast<const uint4*>(qp);
        } else {
            qv.x = *reinterpret_cast<const uint32_t*>(qp);
            qv.y = *reinterpret_cast<const uint32_t*>(qp + 2);
            qv.z = *reinterpret_cast<const uint32_t*>(qp + 4);
            qv.w = *reinterpret_cast<const uint32_t*>(qp + 6);
        }
        float x[8];
        x[0] = __fmul_rn(bf16_to_f32(static_cast<uint16_t>(qv.x & 0xFFFFu)), sc[0]);
        x[1] = __fmul_rn(bf16_to_f32(static_cast<uint16_t>(qv.x >> 16)),      sc[1]);
        x[2] = __fmul_rn(bf16_to_f32(static_cast<uint16_t>(qv.y & 0xFFFFu)), sc[2]);
        x[3] = __fmul_rn(bf16_to_f32(static_cast<uint16_t>(qv.y >> 16)),      sc[3]);
        x[4] = __fmul_rn(bf16_to_f32(static_cast<uint16_t>(qv.z & 0xFFFFu)), sc[4]);
        x[5] = __fmul_rn(bf16_to_f32(static_cast<uint16_t>(qv.z >> 16)),      sc[5]);
        x[6] = __fmul_rn(bf16_to_f32(static_cast<uint16_t>(qv.w & 0xFFFFu)), sc[6]);
        x[7] = __fmul_rn(bf16_to_f32(static_cast<uint16_t>(qv.w >> 16)),      sc[7]);

        const float e01 = fma_rn_f32(x[0], x[0], __fmul_rn(x[2], x[2]));
        const float o01 = fma_rn_f32(x[1], x[1], __fmul_rn(x[3], x[3]));
        const float e23 = fma_rn_f32(x[4], x[4], __fmul_rn(x[6], x[6]));
        const float o23 = fma_rn_f32(x[5], x[5], __fmul_rn(x[7], x[7]));
        float sum_sq = __fadd_rn(__fadd_rn(e01, e23), __fadd_rn(o01, o23));
        sum_sq = __fadd_rn(sum_sq, __shfl_xor_sync(0xFFFFFFFFu, sum_sq, 8));
        sum_sq = __fadd_rn(sum_sq, __shfl_xor_sync(0xFFFFFFFFu, sum_sq, 4));
        sum_sq = __fadd_rn(sum_sq, __shfl_xor_sync(0xFFFFFFFFu, sum_sq, 2));
        sum_sq = __fadd_rn(sum_sq, __shfl_xor_sync(0xFFFFFFFFu, sum_sq, 1));

        const float norm = (sum_sq > 0.0f) ? __fmul_rn(sum_sq, rsqrt_approx_f32(sum_sq)) : 0.0f;
        const float mult = fminf(__fdiv_rn(clip_threshold, __fadd_rn(norm, clip_eps)), 1.0f);
        const float tau  = HAS_TAU ? Tau[t] : 1.0f;

        uint16_t b[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            uint16_t bi = f32_to_bf16(__fmul_rn(x[i], mult));
            if constexpr (HAS_TAU) bi = f32_to_bf16(__fmul_rn(bf16_to_f32(bi), tau));
            b[i] = bi;
        }
        uint4 ov;
        ov.x = pack_bf16x2(b[0], b[1]);
        ov.y = pack_bf16x2(b[2], b[3]);
        ov.z = pack_bf16x2(b[4], b[5]);
        ov.w = pack_bf16x2(b[6], b[7]);
        *reinterpret_cast<uint4*>(QOut + (static_cast<int64_t>(t) * 6 + h) * 128 + d0) = ov;
    }
}

template <bool DYNAMIC, bool HAS_TAU, bool HAS_NLIVE, bool USE_PDL, bool C32_MULTIWARP, bool HAS_TREE, bool KV_BF12>
__global__ void __launch_bounds__(96, 2)
attn_prologue_kvq_exact_kernel_v13(
    const uint16_t* __restrict__ K, const uint16_t* __restrict__ V,
    const uint16_t* __restrict__ WK, const uint16_t* __restrict__ WV,
    const uint16_t* __restrict__ DK, const uint16_t* __restrict__ DV,
    uint16_t* __restrict__ ConvCache, uint16_t* __restrict__ KeyCache, uint16_t* __restrict__ ValueCache,
    const int64_t* __restrict__ KVSlot, const int64_t* __restrict__ Pos,
    const int32_t* __restrict__ SeqIdx, const int64_t* __restrict__ Slot,
    const int32_t* __restrict__ BlockTable, const int32_t* __restrict__ QStart,
    const uint16_t* __restrict__ Q, const uint16_t* __restrict__ QScale,
    const float* __restrict__ Tau, uint16_t* __restrict__ QOut,
    const int32_t* __restrict__ NLivePtr,
    const int32_t* __restrict__ TreeSconvAnc,
    int T, int k_ctas, int v_ctas, int q_ctas,
    int64_t stride_k, int64_t stride_v, int64_t stride_w0, int64_t stride_w1,
    int64_t stride_dk, int64_t stride_dv, int64_t stride_c_blk, int64_t stride_c_n,
    int64_t stride_kv_blk, int64_t stride_kv_n, int kv_bs, int kv_shift,
    int64_t stride_bt, int max_blocks, int N_cache, int N_shift,
    int64_t stride_q, float scale, float clip_threshold, float clip_eps, float norm_eps,
    uint16_t* __restrict__ KEmergRaw,
    uint32_t* __restrict__ KEmergUsed,
    uint16_t* __restrict__ VEmergRaw,
    uint32_t* __restrict__ VEmergUsed,
    uint32_t emerg_cap
) {
    const int bid = blockIdx.x;
    if constexpr (!C32_MULTIWARP) {
        (void)k_ctas; (void)v_ctas; (void)q_ctas;
        if (bid < 8) {
            if (threadIdx.x < 32) {
                run_kv_tok_warp<false, DYNAMIC, HAS_NLIVE, USE_PDL, HAS_TREE, KV_BF12>(
                    bid, 8, K, WK, DK, ConvCache, KeyCache, KVSlot, Pos, SeqIdx, Slot, BlockTable, QStart,
                    NLivePtr, TreeSconvAnc,
                    T, stride_k, stride_w0, stride_w1, stride_dk, stride_c_blk, stride_c_n,
                    stride_kv_blk, stride_kv_n, kv_bs, kv_shift, stride_bt, max_blocks, N_cache, N_shift,
                    scale, clip_threshold, clip_eps, norm_eps,
                    KEmergRaw, KEmergUsed, emerg_cap);
            } else if constexpr (USE_PDL) {
                gdc_wait();
            }
        } else if (bid < 16) {
            if (threadIdx.x < 32) {
                run_kv_tok_warp<true, DYNAMIC, HAS_NLIVE, USE_PDL, HAS_TREE, KV_BF12>(
                    bid - 8, 8, V, WV, DV, ConvCache, ValueCache, KVSlot, Pos, SeqIdx, Slot, BlockTable, QStart,
                    NLivePtr, TreeSconvAnc,
                    T, stride_v, stride_w0, stride_w1, stride_dv, stride_c_blk, stride_c_n,
                    stride_kv_blk, stride_kv_n, kv_bs, kv_shift, stride_bt, max_blocks, N_cache, N_shift,
                    scale, clip_threshold, clip_eps, norm_eps,
                    VEmergRaw, VEmergUsed, emerg_cap);
            } else if constexpr (USE_PDL) {
                gdc_wait();
            }
        } else {
            run_q_tok_cta<HAS_TAU, HAS_NLIVE, USE_PDL>(
                bid - 16, 8, Q, QScale, Tau, QOut, NLivePtr, T, stride_q, clip_threshold, clip_eps);
        }
    } else {
        (void)TreeSconvAnc;
        const int wid = threadIdx.x >> 5;
        if (bid < k_ctas) {
            run_kv_tok_warp<false, DYNAMIC, HAS_NLIVE, USE_PDL, false, KV_BF12>(
                bid * 3 + wid, k_ctas * 3,
                K, WK, DK, ConvCache, KeyCache, KVSlot, Pos, SeqIdx, Slot, BlockTable, QStart,
                NLivePtr, nullptr,
                T, stride_k, stride_w0, stride_w1, stride_dk, stride_c_blk, stride_c_n,
                stride_kv_blk, stride_kv_n, kv_bs, kv_shift, stride_bt, max_blocks, N_cache, N_shift,
                scale, clip_threshold, clip_eps, norm_eps,
                KEmergRaw, KEmergUsed, emerg_cap);
        } else if (bid < k_ctas + v_ctas) {
            const int vbid = bid - k_ctas;
            run_kv_tok_warp<true, DYNAMIC, HAS_NLIVE, USE_PDL, false, KV_BF12>(
                vbid * 3 + wid, v_ctas * 3,
                V, WV, DV, ConvCache, ValueCache, KVSlot, Pos, SeqIdx, Slot, BlockTable, QStart,
                NLivePtr, nullptr,
                T, stride_v, stride_w0, stride_w1, stride_dv, stride_c_blk, stride_c_n,
                stride_kv_blk, stride_kv_n, kv_bs, kv_shift, stride_bt, max_blocks, N_cache, N_shift,
                scale, clip_threshold, clip_eps, norm_eps,
                VEmergRaw, VEmergUsed, emerg_cap);
        } else {
            run_q_tok_cta<HAS_TAU, HAS_NLIVE, USE_PDL>(
                bid - k_ctas - v_ctas, q_ctas,
                Q, QScale, Tau, QOut, NLivePtr, T, stride_q, clip_threshold, clip_eps);
        }
    }
    if constexpr (USE_PDL) {
        __syncthreads();
        if (threadIdx.x == 0) gdc_launch_dependents();
    }
}

template <bool KV_BF12>
static inline int dispatch_prologue_mode(
    const void* K, const void* V, const void* WK, const void* WV, const void* DK, const void* DV,
    void* ConvCache, void* KeyCache, void* ValueCache,
    const void* KVSlot, const void* Pos, const void* SeqIdx, const void* Slot, const void* BlockTable, const void* QStart,
    const void* Q, const void* QScale, const void* Tau, void* QOut, const void* NLivePtr,
    const void* TreeSconvAnc,
    int T, int64_t stride_k, int64_t stride_v, int64_t stride_w0, int64_t stride_w1,
    int64_t stride_dk, int64_t stride_dv, int64_t stride_c_blk, int64_t stride_c_n,
    int64_t stride_kv_blk, int64_t stride_kv_n, int kv_bs,
    int64_t stride_bt, int max_blocks, int N_cache, int64_t stride_q,
    float scale, float clip_threshold, float clip_eps, float norm_eps,
    int dynamic, int has_tau, int has_nlive, int use_pdl, bool has_tree, uint64_t stream_u64,
    void* KEmergRaw, void* KEmergUsed, void* VEmergRaw, void* VEmergUsed, uint32_t emerg_cap
) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_u64);
    const int n_shift  = (N_cache > 0 && (N_cache & (N_cache - 1)) == 0) ? __builtin_ctz(N_cache) : -1;
    const int kv_shift = (kv_bs > 0 && (kv_bs & (kv_bs - 1)) == 0) ? __builtin_ctz(kv_bs) : -1;
    const bool c32 = (!has_tree && T > 8);
    const int k_ctas = c32 ? (((T + 2) / 3 < 24) ? (T + 2) / 3 : 24) : 8;
    const int v_ctas = k_ctas;
    const int q_ctas = c32 ? ((T < 48) ? T : 48) : 8;
    dim3 grid(static_cast<unsigned>(k_ctas + v_ctas + q_ctas), 1, 1);
    dim3 block(96, 1, 1);
    const int mode = (dynamic ? 1 : 0) | (has_tau ? 2 : 0) | (has_nlive ? 4 : 0);
    const bool pdl = (use_pdl != 0);

    #define LP(DYN, TAU, NL, PD, C32M, TR) \
        return static_cast<int>(launch_maybe_pdl( \
            attn_prologue_kvq_exact_kernel_v13<DYN, TAU, NL, PD, C32M, TR, KV_BF12>, \
            grid, block, 0, stream, PD, \
            reinterpret_cast<const uint16_t*>(K), reinterpret_cast<const uint16_t*>(V), \
            reinterpret_cast<const uint16_t*>(WK), reinterpret_cast<const uint16_t*>(WV), \
            reinterpret_cast<const uint16_t*>(DK), reinterpret_cast<const uint16_t*>(DV), \
            reinterpret_cast<uint16_t*>(ConvCache), reinterpret_cast<uint16_t*>(KeyCache), \
            reinterpret_cast<uint16_t*>(ValueCache), reinterpret_cast<const int64_t*>(KVSlot), \
            reinterpret_cast<const int64_t*>(Pos), reinterpret_cast<const int32_t*>(SeqIdx), \
            reinterpret_cast<const int64_t*>(Slot), reinterpret_cast<const int32_t*>(BlockTable), \
            reinterpret_cast<const int32_t*>(QStart), reinterpret_cast<const uint16_t*>(Q), \
            reinterpret_cast<const uint16_t*>(QScale), reinterpret_cast<const float*>(Tau), \
            reinterpret_cast<uint16_t*>(QOut), reinterpret_cast<const int32_t*>(NLivePtr), \
            reinterpret_cast<const int32_t*>(TreeSconvAnc), \
            T, k_ctas, v_ctas, q_ctas, stride_k, stride_v, stride_w0, stride_w1, stride_dk, stride_dv, \
            stride_c_blk, stride_c_n, stride_kv_blk, stride_kv_n, kv_bs, kv_shift, stride_bt, max_blocks, \
            N_cache, n_shift, stride_q, scale, clip_threshold, clip_eps, norm_eps, \
            reinterpret_cast<uint16_t*>(KEmergRaw), reinterpret_cast<uint32_t*>(KEmergUsed), \
            reinterpret_cast<uint16_t*>(VEmergRaw), reinterpret_cast<uint32_t*>(VEmergUsed), emerg_cap))

    #define DISPATCH(DYN, TAU, NL) \
        if (has_tree) { if (pdl) { LP(DYN, TAU, NL, true, false, true); } else { LP(DYN, TAU, NL, false, false, true); } } \
        else if (c32) { if (pdl) { LP(DYN, TAU, NL, true, true,  false); } else { LP(DYN, TAU, NL, false, true,  false); } } \
        else          { if (pdl) { LP(DYN, TAU, NL, true, false, false); } else { LP(DYN, TAU, NL, false, false, false); } }

    switch (mode) {
        case 0: DISPATCH(false, false, false);
        case 1: DISPATCH(true,  false, false);
        case 2: DISPATCH(false, true,  false);
        case 3: DISPATCH(true,  true,  false);
        case 4: DISPATCH(false, false, true);
        case 5: DISPATCH(true,  false, true);
        case 6: DISPATCH(false, true,  true);
        case 7: DISPATCH(true,  true,  true);
        default: return -2;
    }
    #undef DISPATCH
    #undef LP
}

}  // namespace

int launch_prologue_v13(
    bool kv_bf12,
    const void* K, const void* V, const void* WK, const void* WV, const void* DK, const void* DV,
    void* ConvCache, void* KeyCache, void* ValueCache,
    const void* KVSlot, const void* Pos, const void* SeqIdx, const void* Slot, const void* BlockTable, const void* QStart,
    const void* Q, const void* QScale, const void* Tau, void* QOut, const void* NLivePtr,
    const void* TreeSconvAnc,
    int T, int64_t stride_k, int64_t stride_v, int64_t stride_w0, int64_t stride_w1,
    int64_t stride_dk, int64_t stride_dv, int64_t stride_c_blk, int64_t stride_c_n,
    int64_t stride_kv_blk, int64_t stride_kv_n, int kv_bs,
    int64_t stride_bt, int max_blocks, int N_cache, int64_t stride_q,
    float scale, float clip_threshold, float clip_eps, float norm_eps,
    int dynamic, int has_tau, int has_nlive, int use_pdl, bool has_tree, uint64_t stream_u64,
    void* KEmergRaw, void* KEmergUsed, void* VEmergRaw, void* VEmergUsed, uint32_t emerg_cap
) {
    if (kv_bf12) {
        return dispatch_prologue_mode<true>(
            K, V, WK, WV, DK, DV, ConvCache, KeyCache, ValueCache,
            KVSlot, Pos, SeqIdx, Slot, BlockTable, QStart, Q, QScale, Tau, QOut, NLivePtr, TreeSconvAnc,
            T, stride_k, stride_v, stride_w0, stride_w1, stride_dk, stride_dv,
            stride_c_blk, stride_c_n, stride_kv_blk, stride_kv_n, kv_bs,
            stride_bt, max_blocks, N_cache, stride_q, scale, clip_threshold, clip_eps, norm_eps,
            dynamic, has_tau, has_nlive, use_pdl, has_tree, stream_u64,
            KEmergRaw, KEmergUsed, VEmergRaw, VEmergUsed, emerg_cap);
    }
    return dispatch_prologue_mode<false>(
        K, V, WK, WV, DK, DV, ConvCache, KeyCache, ValueCache,
        KVSlot, Pos, SeqIdx, Slot, BlockTable, QStart, Q, QScale, Tau, QOut, NLivePtr, TreeSconvAnc,
        T, stride_k, stride_v, stride_w0, stride_w1, stride_dk, stride_dv,
        stride_c_blk, stride_c_n, stride_kv_blk, stride_kv_n, kv_bs,
        stride_bt, max_blocks, N_cache, stride_q, scale, clip_threshold, clip_eps, norm_eps,
        dynamic, has_tau, has_nlive, use_pdl, has_tree, stream_u64,
        nullptr, nullptr, nullptr, nullptr, 0u);
}

}  // namespace attn_v13
