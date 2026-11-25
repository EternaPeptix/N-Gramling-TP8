// SPDX-License-Identifier: Apache-2.0
// Wave-V13 Split TU 5/5: Host-Only C ABI Dispatcher (`attn_dispatch_v13.cu`)
//   0 device kernels (compiles in <0.5 s). Exports ALL 22 C ABI symbols
//   (100% drop-in superset of V7/V8/V9/V10/V11 + `v13_version()` / `v13_init_smem()`).

#include "attn_common_v13.cuh"

namespace {

using namespace attn_v13;

static bool g_v13_smem_inited = false;

static inline bool is_valid_swa_splits(int s) {
    return (s == 1 || s == 2 || s == 4 || s == 6 || s == 8);
}

static inline bool is_valid_global_splits(int s) {
    return (s == 1 || s == 2 || s == 3 || s == 4 || s == 6 || s == 8 ||
            s == 12 || s == 16 || s == 24 || s == 32 || s == 48);
}

static inline bool is_valid_short_splits(int s) {
    return (s == 1 || s == 2 || s == 3 || s == 4 || s == 6 || s == 8 || s == 12 || s == 16);
}

static inline bool is_valid_long_splits(int s) {
    return (s == 2 || s == 3 || s == 4 || s == 6 || s == 8 || s == 12 || s == 16 ||
            s == 24 || s == 32 || s == 48);
}

static inline int dispatch_mgs_v13(const AttnLaunchParams& p, bool kv_bf12, int tree_mode) {
    if (tree_mode != 0) return -2;
    if (kv_bf12 && (p.BS != KV_BF12_BS || p.sk_b < KV_BF12_PAGE_BYTES || p.sv_b < KV_BF12_PAGE_BYTES)) {
        return -2;
    }
    if (p.window == 1023) {
        if (!is_valid_swa_splits(p.num_splits)) return -2;
        return launch_swa_mgs_v13(p, kv_bf12, false);
    }
    if (p.window < 0) {
        if (!is_valid_global_splits(p.num_splits)) return -2;
        return kv_bf12 ? launch_global_bf12_mgs_v13(p, false) : launch_global_bf16_mgs_v13(p, false);
    }
    return -2;
}

static inline int dispatch_c32_v13(const AttnLaunchParams& p, bool kv_bf12, int tree_mode) {
    if (tree_mode != 0) return -2;
    if (kv_bf12 && (p.BS != KV_BF12_BS || p.sk_b < KV_BF12_PAGE_BYTES || p.sv_b < KV_BF12_PAGE_BYTES)) {
        return -2;
    }
    const bool mt1_high_s = (p.mtiles == 1) && (p.num_splits != 1 && p.num_splits != 2 && p.num_splits != 3 && p.num_splits != 4);
    if (!p.fuse_mt || p.G < 2 || p.mtiles < 1 || p.mtiles > 3 || mt1_high_s) {
        return dispatch_mgs_v13(p, kv_bf12, tree_mode);
    }
    if (p.window == 1023) {
        if (!is_valid_swa_splits(p.num_splits)) return -2;
        return launch_swa_fusedmt_v13(p, kv_bf12, g_v13_smem_inited);
    }
    if (p.window < 0) {
        if (!is_valid_global_splits(p.num_splits)) return -2;
        return kv_bf12
            ? launch_global_bf12_fusedmt_v13(p, g_v13_smem_inited)
            : launch_global_bf16_fusedmt_v13(p, g_v13_smem_inited);
    }
    return -2;
}

static inline int dispatch_tree_v13(const AttnLaunchParams& p, bool kv_bf12, int tree_mode) {
    if (tree_mode != 0) return -2;
    if (p.TreeMeta == nullptr || p.T <= 0 || p.T > 8 || p.num_reqs != 1 || p.G != 6 || p.mtiles != 4) {
        return -1;
    }
    if (kv_bf12 && (p.BS != KV_BF12_BS || p.sk_b < KV_BF12_PAGE_BYTES || p.sv_b < KV_BF12_PAGE_BYTES)) {
        return -2;
    }
    if (p.window == 1023 && p.num_splits == 8) {
        return launch_swa_mgs_v13(p, kv_bf12, true);
    }
    if (p.window < 0 && p.num_splits == 16) {
        return kv_bf12 ? launch_global_bf12_mgs_v13(p, true) : launch_global_bf16_mgs_v13(p, true);
    }
    return -2;
}

static inline int dispatch_paired_v13(
    AttnLaunchParams p, bool kv_bf12, int tree_mode,
    int num_splits_short, int num_splits_long, int kv_tok_thresh, int kv_evict_first_long
) {
    if (tree_mode != 0) return -2;
    if (kv_bf12 && (p.BS != KV_BF12_BS || p.sk_b < KV_BF12_PAGE_BYTES || p.sv_b < KV_BF12_PAGE_BYTES)) {
        return -2;
    }
    if (p.window >= 0 || p.G < 2 || p.mtiles < 2 || p.mtiles > 3) {
        p.num_splits = num_splits_short;
        p.fuse_mt = 0;
        p.kv_evict_first = kv_evict_first_long;
        p.guard_mode = 0;
        p.kv_tok_thresh = 0;
        return dispatch_c32_v13(p, kv_bf12, tree_mode);
    }
    if (kv_tok_thresh <= 0) {
        p.num_splits = num_splits_long;
        p.fuse_mt = 1;
        p.kv_evict_first = kv_evict_first_long;
        p.guard_mode = 0;
        p.kv_tok_thresh = 0;
        return dispatch_c32_v13(p, kv_bf12, tree_mode);
    }
    if (!is_valid_short_splits(num_splits_short) || !is_valid_long_splits(num_splits_long)) {
        return -2;
    }

    // Node 1: Short-Context Global (FUSE_MT=0, guard_mode=1, active iff tot_kv <= kv_tok_thresh)
    AttnLaunchParams p1 = p;
    p1.num_splits = num_splits_short;
    p1.trig_wo = 0;
    p1.fuse_mt = 0;
    p1.kv_evict_first = 0;
    p1.guard_mode = 1;
    p1.kv_tok_thresh = kv_tok_thresh;
    const int rc1 = kv_bf12
        ? launch_global_bf12_mgs_v13(p1, false)
        : launch_global_bf16_mgs_v13(p1, false);
    if (rc1 != 0) return rc1;

    // Node 2: Long-Context Global (FUSE_MT=1, guard_mode=2, active iff tot_kv > kv_tok_thresh)
    AttnLaunchParams p2 = p;
    p2.num_splits = num_splits_long;
    p2.fuse_mt = 1;
    p2.kv_evict_first = kv_evict_first_long;
    p2.guard_mode = 2;
    p2.kv_tok_thresh = kv_tok_thresh;
    return kv_bf12
        ? launch_global_bf12_fusedmt_v13(p2, g_v13_smem_inited)
        : launch_global_bf16_fusedmt_v13(p2, g_v13_smem_inited);
}

static inline AttnLaunchParams make_params(
    const void* Q, const void* R, const void* RScale, const void* Tau, const void* RelProjT,
    const void* K, const void* V, const void* BT, const void* SEQ, const void* CUQ,
    const void* GATE, const void* SINK, void* OP, void* LP, void* SEM, void* Y,
    const void* NLivePtr, const void* TreeMeta,
    int64_t sq_t, int64_t sq_h, int64_t sr_t, int64_t sr_h,
    int64_t sk_b, int64_t sk_s, int64_t sk_h, int64_t sv_b, int64_t sv_s, int64_t sv_h,
    int64_t sbt_r, int64_t sg_t, int64_t ssk_t, int64_t sy_t,
    int64_t so_s, int64_t so_t, int64_t so_h, int64_t sl_s, int64_t sl_t, int64_t sl_h,
    float scale_log2, float rel_threshold, float clip_eps,
    int T, int NT, int mtiles, int num_reqs, int Hkv, int BS, int E, int G,
    int window, int num_splits, int has_tau, int has_nlive, int use_pdl, int trig_wo,
    int fuse_mt, int kv_evict_first,
    uint64_t stream_u64, const void* KEmergRaw, const void* VEmergRaw
) {
    AttnLaunchParams p = {};
    p.Q = Q; p.R = R; p.RScale = RScale; p.Tau = Tau; p.RelProjT = RelProjT;
    p.K = K; p.V = V; p.BT = BT; p.SEQ = SEQ; p.CUQ = CUQ;
    p.GATE = GATE; p.SINK = SINK; p.OP = OP; p.LP = LP; p.SEM = SEM; p.Y = Y;
    p.NLivePtr = NLivePtr; p.TreeMeta = TreeMeta;
    p.sq_t = sq_t; p.sq_h = sq_h; p.sr_t = sr_t; p.sr_h = sr_h;
    p.sk_b = sk_b; p.sk_s = sk_s; p.sk_h = sk_h; p.sv_b = sv_b; p.sv_s = sv_s; p.sv_h = sv_h;
    p.sbt_r = sbt_r; p.sg_t = sg_t; p.ssk_t = ssk_t; p.sy_t = sy_t;
    p.so_s = so_s; p.so_t = so_t; p.so_h = so_h; p.sl_s = sl_s; p.sl_t = sl_t; p.sl_h = sl_h;
    p.scale_log2 = scale_log2; p.rel_threshold = rel_threshold; p.clip_eps = clip_eps;
    p.T = T; p.NT = NT; p.mtiles = mtiles; p.num_reqs = num_reqs; p.Hkv = Hkv;
    p.BS = BS; p.E = E; p.G = G; p.window = window; p.num_splits = num_splits;
    p.has_tau = has_tau; p.has_nlive = has_nlive; p.use_pdl = use_pdl; p.trig_wo = trig_wo;
    p.fuse_mt = fuse_mt; p.kv_evict_first = kv_evict_first;
    p.guard_mode = 0; p.kv_tok_thresh = 0;
    p.stream_u64 = stream_u64; p.KEmergRaw = KEmergRaw; p.VEmergRaw = VEmergRaw;
    return p;
}

}  // namespace

// =========================================================================
// EXPORTED C ABI SYMBOLS (22 symbols: 100% superset of V7..V11 + V13!)
// =========================================================================

extern "C" int fused_decode_attn_v7_version(void)  { return 13; }
extern "C" int fused_decode_attn_v8_version(void)  { return 13; }
extern "C" int fused_decode_attn_v9_version(void)  { return 13; }
extern "C" int fused_decode_attn_v10_version(void) { return 13; }
extern "C" int fused_decode_attn_v11_version(void) { return 13; }
extern "C" int fused_decode_attn_v13_version(void) { return 13; }

extern "C" int fused_decode_attn_v13_init_smem(void) {
    if (g_v13_smem_inited) return 0;
    int rc;
    if ((rc = init_smem_swa_v13())         != 0) return rc;
    if ((rc = init_smem_global_bf16_v13()) != 0) return rc;
    if ((rc = init_smem_global_bf12_v13()) != 0) return rc;
    g_v13_smem_inited = true;
    return 0;
}

extern "C" int fused_decode_attn_v11_init_smem(void) { return fused_decode_attn_v13_init_smem(); }
extern "C" int fused_decode_attn_v10_init_smem(void) { return fused_decode_attn_v13_init_smem(); }
extern "C" int fused_decode_attn_v9_init_smem(void)  { return fused_decode_attn_v13_init_smem(); }

extern "C" int attn_prologue_kvq_exact_fwd(
    const void* K, const void* V, const void* WK, const void* WV, const void* DK, const void* DV,
    void* ConvCache, void* KeyCache, void* ValueCache,
    const void* KVSlot, const void* Pos, const void* SeqIdx, const void* Slot, const void* BlockTable, const void* QStart,
    const void* Q, const void* QScale, const void* Tau, void* QOut, const void* NLivePtr,
    int T, int64_t stride_k, int64_t stride_v, int64_t stride_w0, int64_t stride_w1,
    int64_t stride_dk, int64_t stride_dv, int64_t stride_c_blk, int64_t stride_c_n,
    int64_t stride_kv_blk, int64_t stride_kv_n, int kv_bs,
    int64_t stride_bt, int max_blocks, int N_cache, int64_t stride_q,
    float scale, float clip_threshold, float clip_eps, float norm_eps,
    int dynamic, int has_tau, int has_nlive, int use_pdl, uint64_t stream_u64
) {
    if (T <= 0 || T > 512) return -1;
    return launch_prologue_v13(
        false, K, V, WK, WV, DK, DV, ConvCache, KeyCache, ValueCache,
        KVSlot, Pos, SeqIdx, Slot, BlockTable, QStart, Q, QScale, Tau, QOut, NLivePtr, nullptr,
        T, stride_k, stride_v, stride_w0, stride_w1, stride_dk, stride_dv,
        stride_c_blk, stride_c_n, stride_kv_blk, stride_kv_n, kv_bs,
        stride_bt, max_blocks, N_cache, stride_q, scale, clip_threshold, clip_eps, norm_eps,
        dynamic, has_tau, has_nlive, use_pdl, false, stream_u64,
        nullptr, nullptr, nullptr, nullptr, 0u);
}

extern "C" int attn_prologue_kvq_tree_exact_fwd(
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
    int dynamic, int has_tau, int has_nlive, int use_pdl, uint64_t stream_u64
) {
    if (T <= 0 || T > 16 || TreeSconvAnc == nullptr) return -1;
    return launch_prologue_v13(
        false, K, V, WK, WV, DK, DV, ConvCache, KeyCache, ValueCache,
        KVSlot, Pos, SeqIdx, Slot, BlockTable, QStart, Q, QScale, Tau, QOut, NLivePtr, TreeSconvAnc,
        T, stride_k, stride_v, stride_w0, stride_w1, stride_dk, stride_dv,
        stride_c_blk, stride_c_n, stride_kv_blk, stride_kv_n, kv_bs,
        stride_bt, max_blocks, N_cache, stride_q, scale, clip_threshold, clip_eps, norm_eps,
        dynamic, has_tau, has_nlive, use_pdl, true, stream_u64,
        nullptr, nullptr, nullptr, nullptr, 0u);
}

extern "C" int attn_prologue_kvq_bf12_exact_v10_fwd(
    const void* K, const void* V, const void* WK, const void* WV, const void* DK, const void* DV,
    void* ConvCache, void* KeyCache, void* ValueCache,
    const void* KVSlot, const void* Pos, const void* SeqIdx, const void* Slot, const void* BlockTable, const void* QStart,
    const void* Q, const void* QScale, const void* Tau, void* QOut, const void* NLivePtr,
    int T, int64_t stride_k, int64_t stride_v, int64_t stride_w0, int64_t stride_w1,
    int64_t stride_dk, int64_t stride_dv, int64_t stride_c_blk, int64_t stride_c_n,
    int64_t stride_kv_blk_bytes, int64_t stride_kv_n, int kv_bs,
    int64_t stride_bt, int max_blocks, int N_cache, int64_t stride_q,
    float scale, float clip_threshold, float clip_eps, float norm_eps,
    int dynamic, int has_tau, int has_nlive, int use_pdl, uint64_t stream_u64,
    void* KEmergRaw, void* KEmergUsed, void* VEmergRaw, void* VEmergUsed, uint32_t emerg_cap
) {
    if (T <= 0 || T > 512) return -1;
    if (kv_bs != KV_BF12_BS || stride_kv_blk_bytes < KV_BF12_PAGE_BYTES) return -2;
    return launch_prologue_v13(
        true, K, V, WK, WV, DK, DV, ConvCache, KeyCache, ValueCache,
        KVSlot, Pos, SeqIdx, Slot, BlockTable, QStart, Q, QScale, Tau, QOut, NLivePtr, nullptr,
        T, stride_k, stride_v, stride_w0, stride_w1, stride_dk, stride_dv,
        stride_c_blk, stride_c_n, stride_kv_blk_bytes, stride_kv_n, kv_bs,
        stride_bt, max_blocks, N_cache, stride_q, scale, clip_threshold, clip_eps, norm_eps,
        dynamic, has_tau, has_nlive, use_pdl, false, stream_u64,
        KEmergRaw, KEmergUsed, VEmergRaw, VEmergUsed, emerg_cap);
}

extern "C" int attn_prologue_kvq_tree_bf12_exact_v10_fwd(
    const void* K, const void* V, const void* WK, const void* WV, const void* DK, const void* DV,
    void* ConvCache, void* KeyCache, void* ValueCache,
    const void* KVSlot, const void* Pos, const void* SeqIdx, const void* Slot, const void* BlockTable, const void* QStart,
    const void* Q, const void* QScale, const void* Tau, void* QOut, const void* NLivePtr,
    const void* TreeSconvAnc,
    int T, int64_t stride_k, int64_t stride_v, int64_t stride_w0, int64_t stride_w1,
    int64_t stride_dk, int64_t stride_dv, int64_t stride_c_blk, int64_t stride_c_n,
    int64_t stride_kv_blk_bytes, int64_t stride_kv_n, int kv_bs,
    int64_t stride_bt, int max_blocks, int N_cache, int64_t stride_q,
    float scale, float clip_threshold, float clip_eps, float norm_eps,
    int dynamic, int has_tau, int has_nlive, int use_pdl, uint64_t stream_u64,
    void* KEmergRaw, void* KEmergUsed, void* VEmergRaw, void* VEmergUsed, uint32_t emerg_cap
) {
    if (T <= 0 || T > 16 || TreeSconvAnc == nullptr) return -1;
    if (kv_bs != KV_BF12_BS || stride_kv_blk_bytes < KV_BF12_PAGE_BYTES) return -2;
    return launch_prologue_v13(
        true, K, V, WK, WV, DK, DV, ConvCache, KeyCache, ValueCache,
        KVSlot, Pos, SeqIdx, Slot, BlockTable, QStart, Q, QScale, Tau, QOut, NLivePtr, TreeSconvAnc,
        T, stride_k, stride_v, stride_w0, stride_w1, stride_dk, stride_dv,
        stride_c_blk, stride_c_n, stride_kv_blk_bytes, stride_kv_n, kv_bs,
        stride_bt, max_blocks, N_cache, stride_q, scale, clip_threshold, clip_eps, norm_eps,
        dynamic, has_tau, has_nlive, use_pdl, true, stream_u64,
        KEmergRaw, KEmergUsed, VEmergRaw, VEmergUsed, emerg_cap);
}

extern "C" int fused_decode_attn_otf_mgs_fwd(
    const void* Q, const void* R, const void* RScale, const void* Tau, const void* RelProjT,
    const void* K, const void* V, const void* BT, const void* SEQ, const void* CUQ,
    const void* GATE, const void* SINK, void* OP, void* LP, void* SEM, void* Y, const void* NLivePtr,
    int64_t sq_t, int64_t sq_h, int64_t sr_t, int64_t sr_h,
    int64_t sk_b, int64_t sk_s, int64_t sk_h, int64_t sv_b, int64_t sv_s, int64_t sv_h,
    int64_t sbt_r, int64_t sg_t, int64_t ssk_t, int64_t sy_t,
    int64_t so_s, int64_t so_t, int64_t so_h, int64_t sl_s, int64_t sl_t, int64_t sl_h,
    float scale_log2, float rel_threshold, float clip_eps,
    int T, int NT, int mtiles, int num_reqs, int Hkv, int BS, int E, int G,
    int window, int num_splits, int has_tau, int has_nlive, int use_pdl, int trig_wo, int tree_mode,
    uint64_t stream_u64
) {
    return dispatch_mgs_v13(
        make_params(Q, R, RScale, Tau, RelProjT, K, V, BT, SEQ, CUQ, GATE, SINK, OP, LP, SEM, Y, NLivePtr, nullptr,
                    sq_t, sq_h, sr_t, sr_h, sk_b, sk_s, sk_h, sv_b, sv_s, sv_h, sbt_r, sg_t, ssk_t, sy_t,
                    so_s, so_t, so_h, sl_s, sl_t, sl_h, scale_log2, rel_threshold, clip_eps,
                    T, NT, mtiles, num_reqs, Hkv, BS, E, G, window, num_splits, has_tau, has_nlive,
                    use_pdl, trig_wo, 0, 0, stream_u64, nullptr, nullptr),
        false, tree_mode);
}

extern "C" int fused_decode_attn_otf_mgs_bf12_v10_fwd(
    const void* Q, const void* R, const void* RScale, const void* Tau, const void* RelProjT,
    const void* K, const void* V, const void* BT, const void* SEQ, const void* CUQ,
    const void* GATE, const void* SINK, void* OP, void* LP, void* SEM, void* Y, const void* NLivePtr,
    int64_t sq_t, int64_t sq_h, int64_t sr_t, int64_t sr_h,
    int64_t sk_b_bytes, int64_t sk_s, int64_t sk_h_bytes, int64_t sv_b_bytes, int64_t sv_s, int64_t sv_h_bytes,
    int64_t sbt_r, int64_t sg_t, int64_t ssk_t, int64_t sy_t,
    int64_t so_s, int64_t so_t, int64_t so_h, int64_t sl_s, int64_t sl_t, int64_t sl_h,
    float scale_log2, float rel_threshold, float clip_eps,
    int T, int NT, int mtiles, int num_reqs, int Hkv, int BS, int E, int G,
    int window, int num_splits, int has_tau, int has_nlive, int use_pdl, int trig_wo, int tree_mode,
    uint64_t stream_u64,
    const void* KEmergRaw, const void* VEmergRaw
) {
    return dispatch_mgs_v13(
        make_params(Q, R, RScale, Tau, RelProjT, K, V, BT, SEQ, CUQ, GATE, SINK, OP, LP, SEM, Y, NLivePtr, nullptr,
                    sq_t, sq_h, sr_t, sr_h, sk_b_bytes, sk_s, sk_h_bytes, sv_b_bytes, sv_s, sv_h_bytes,
                    sbt_r, sg_t, ssk_t, sy_t, so_s, so_t, so_h, sl_s, sl_t, sl_h,
                    scale_log2, rel_threshold, clip_eps,
                    T, NT, mtiles, num_reqs, Hkv, BS, E, G, window, num_splits, has_tau, has_nlive,
                    use_pdl, trig_wo, 0, 0, stream_u64, KEmergRaw, VEmergRaw),
        true, tree_mode);
}

extern "C" int fused_decode_attn_otf_mgs_c32_fwd(
    const void* Q, const void* R, const void* RScale, const void* Tau, const void* RelProjT,
    const void* K, const void* V, const void* BT, const void* SEQ, const void* CUQ,
    const void* GATE, const void* SINK, void* OP, void* LP, void* SEM, void* Y, const void* NLivePtr,
    int64_t sq_t, int64_t sq_h, int64_t sr_t, int64_t sr_h,
    int64_t sk_b, int64_t sk_s, int64_t sk_h, int64_t sv_b, int64_t sv_s, int64_t sv_h,
    int64_t sbt_r, int64_t sg_t, int64_t ssk_t, int64_t sy_t,
    int64_t so_s, int64_t so_t, int64_t so_h, int64_t sl_s, int64_t sl_t, int64_t sl_h,
    float scale_log2, float rel_threshold, float clip_eps,
    int T, int NT, int mtiles, int num_reqs, int Hkv, int BS, int E, int G,
    int window, int num_splits, int has_tau, int has_nlive, int use_pdl, int trig_wo, int tree_mode,
    int fuse_mt, int kv_evict_first,
    uint64_t stream_u64
) {
    return dispatch_c32_v13(
        make_params(Q, R, RScale, Tau, RelProjT, K, V, BT, SEQ, CUQ, GATE, SINK, OP, LP, SEM, Y, NLivePtr, nullptr,
                    sq_t, sq_h, sr_t, sr_h, sk_b, sk_s, sk_h, sv_b, sv_s, sv_h, sbt_r, sg_t, ssk_t, sy_t,
                    so_s, so_t, so_h, sl_s, sl_t, sl_h, scale_log2, rel_threshold, clip_eps,
                    T, NT, mtiles, num_reqs, Hkv, BS, E, G, window, num_splits, has_tau, has_nlive,
                    use_pdl, trig_wo, fuse_mt, kv_evict_first, stream_u64, nullptr, nullptr),
        false, tree_mode);
}

extern "C" int fused_decode_attn_otf_mgs_c32_bf12_v10_fwd(
    const void* Q, const void* R, const void* RScale, const void* Tau, const void* RelProjT,
    const void* K, const void* V, const void* BT, const void* SEQ, const void* CUQ,
    const void* GATE, const void* SINK, void* OP, void* LP, void* SEM, void* Y, const void* NLivePtr,
    int64_t sq_t, int64_t sq_h, int64_t sr_t, int64_t sr_h,
    int64_t sk_b_bytes, int64_t sk_s, int64_t sk_h_bytes, int64_t sv_b_bytes, int64_t sv_s, int64_t sv_h_bytes,
    int64_t sbt_r, int64_t sg_t, int64_t ssk_t, int64_t sy_t,
    int64_t so_s, int64_t so_t, int64_t so_h, int64_t sl_s, int64_t sl_t, int64_t sl_h,
    float scale_log2, float rel_threshold, float clip_eps,
    int T, int NT, int mtiles, int num_reqs, int Hkv, int BS, int E, int G,
    int window, int num_splits, int has_tau, int has_nlive, int use_pdl, int trig_wo, int tree_mode,
    int fuse_mt, int kv_evict_first,
    uint64_t stream_u64,
    const void* KEmergRaw, const void* VEmergRaw
) {
    return dispatch_c32_v13(
        make_params(Q, R, RScale, Tau, RelProjT, K, V, BT, SEQ, CUQ, GATE, SINK, OP, LP, SEM, Y, NLivePtr, nullptr,
                    sq_t, sq_h, sr_t, sr_h, sk_b_bytes, sk_s, sk_h_bytes, sv_b_bytes, sv_s, sv_h_bytes,
                    sbt_r, sg_t, ssk_t, sy_t, so_s, so_t, so_h, sl_s, sl_t, sl_h,
                    scale_log2, rel_threshold, clip_eps,
                    T, NT, mtiles, num_reqs, Hkv, BS, E, G, window, num_splits, has_tau, has_nlive,
                    use_pdl, trig_wo, fuse_mt, kv_evict_first, stream_u64, KEmergRaw, VEmergRaw),
        true, tree_mode);
}

extern "C" int fused_decode_attn_otf_mgs_tree_fwd(
    const void* Q, const void* R, const void* RScale, const void* Tau, const void* RelProjT,
    const void* K, const void* V, const void* BT, const void* SEQ, const void* CUQ,
    const void* GATE, const void* SINK, void* OP, void* LP, void* SEM, void* Y,
    const void* NLivePtr, const void* TreeMeta,
    int64_t sq_t, int64_t sq_h, int64_t sr_t, int64_t sr_h,
    int64_t sk_b, int64_t sk_s, int64_t sk_h, int64_t sv_b, int64_t sv_s, int64_t sv_h,
    int64_t sbt_r, int64_t sg_t, int64_t ssk_t, int64_t sy_t,
    int64_t so_s, int64_t so_t, int64_t so_h, int64_t sl_s, int64_t sl_t, int64_t sl_h,
    float scale_log2, float rel_threshold, float clip_eps,
    int T, int NT, int mtiles, int num_reqs, int Hkv, int BS, int E, int G,
    int window, int num_splits, int has_tau, int has_nlive, int use_pdl, int trig_wo, int tree_mode,
    uint64_t stream_u64
) {
    return dispatch_tree_v13(
        make_params(Q, R, RScale, Tau, RelProjT, K, V, BT, SEQ, CUQ, GATE, SINK, OP, LP, SEM, Y, NLivePtr, TreeMeta,
                    sq_t, sq_h, sr_t, sr_h, sk_b, sk_s, sk_h, sv_b, sv_s, sv_h, sbt_r, sg_t, ssk_t, sy_t,
                    so_s, so_t, so_h, sl_s, sl_t, sl_h, scale_log2, rel_threshold, clip_eps,
                    T, NT, mtiles, num_reqs, Hkv, BS, E, G, window, num_splits, has_tau, has_nlive,
                    use_pdl, trig_wo, 0, 0, stream_u64, nullptr, nullptr),
        false, tree_mode);
}

extern "C" int fused_decode_attn_otf_mgs_tree_bf12_v10_fwd(
    const void* Q, const void* R, const void* RScale, const void* Tau, const void* RelProjT,
    const void* K, const void* V, const void* BT, const void* SEQ, const void* CUQ,
    const void* GATE, const void* SINK, void* OP, void* LP, void* SEM, void* Y,
    const void* NLivePtr, const void* TreeMeta,
    int64_t sq_t, int64_t sq_h, int64_t sr_t, int64_t sr_h,
    int64_t sk_b_bytes, int64_t sk_s, int64_t sk_h_bytes, int64_t sv_b_bytes, int64_t sv_s, int64_t sv_h_bytes,
    int64_t sbt_r, int64_t sg_t, int64_t ssk_t, int64_t sy_t,
    int64_t so_s, int64_t so_t, int64_t so_h, int64_t sl_s, int64_t sl_t, int64_t sl_h,
    float scale_log2, float rel_threshold, float clip_eps,
    int T, int NT, int mtiles, int num_reqs, int Hkv, int BS, int E, int G,
    int window, int num_splits, int has_tau, int has_nlive, int use_pdl, int trig_wo, int tree_mode,
    uint64_t stream_u64,
    const void* KEmergRaw, const void* VEmergRaw
) {
    return dispatch_tree_v13(
        make_params(Q, R, RScale, Tau, RelProjT, K, V, BT, SEQ, CUQ, GATE, SINK, OP, LP, SEM, Y, NLivePtr, TreeMeta,
                    sq_t, sq_h, sr_t, sr_h, sk_b_bytes, sk_s, sk_h_bytes, sv_b_bytes, sv_s, sv_h_bytes,
                    sbt_r, sg_t, ssk_t, sy_t, so_s, so_t, so_h, sl_s, sl_t, sl_h,
                    scale_log2, rel_threshold, clip_eps,
                    T, NT, mtiles, num_reqs, Hkv, BS, E, G, window, num_splits, has_tau, has_nlive,
                    use_pdl, trig_wo, 0, 0, stream_u64, KEmergRaw, VEmergRaw),
        true, tree_mode);
}

extern "C" int fused_decode_attn_otf_mgs_v9_fwd(
    const void* Q, const void* R, const void* RScale, const void* Tau, const void* RelProjT,
    const void* K, const void* V, const void* BT, const void* SEQ, const void* CUQ,
    const void* GATE, const void* SINK, void* OP, void* LP, void* SEM, void* Y, const void* NLivePtr,
    int64_t sq_t, int64_t sq_h, int64_t sr_t, int64_t sr_h,
    int64_t sk_b, int64_t sk_s, int64_t sk_h, int64_t sv_b, int64_t sv_s, int64_t sv_h,
    int64_t sbt_r, int64_t sg_t, int64_t ssk_t, int64_t sy_t,
    int64_t so_s, int64_t so_t, int64_t so_h, int64_t sl_s, int64_t sl_t, int64_t sl_h,
    float scale_log2, float rel_threshold, float clip_eps,
    int T, int NT, int mtiles, int num_reqs, int Hkv, int BS, int E, int G,
    int window, int num_splits_short, int has_tau, int has_nlive, int use_pdl, int trig_wo, int tree_mode,
    int num_splits_long, int kv_tok_thresh, int kv_evict_first_long,
    uint64_t stream_u64
) {
    return dispatch_paired_v13(
        make_params(Q, R, RScale, Tau, RelProjT, K, V, BT, SEQ, CUQ, GATE, SINK, OP, LP, SEM, Y, NLivePtr, nullptr,
                    sq_t, sq_h, sr_t, sr_h, sk_b, sk_s, sk_h, sv_b, sv_s, sv_h, sbt_r, sg_t, ssk_t, sy_t,
                    so_s, so_t, so_h, sl_s, sl_t, sl_h, scale_log2, rel_threshold, clip_eps,
                    T, NT, mtiles, num_reqs, Hkv, BS, E, G, window, num_splits_short, has_tau, has_nlive,
                    use_pdl, trig_wo, 0, kv_evict_first_long, stream_u64, nullptr, nullptr),
        false, tree_mode, num_splits_short, num_splits_long, kv_tok_thresh, kv_evict_first_long);
}

extern "C" int fused_decode_attn_otf_mgs_v10_bf12_paired_fwd(
    const void* Q, const void* R, const void* RScale, const void* Tau, const void* RelProjT,
    const void* K, const void* V, const void* BT, const void* SEQ, const void* CUQ,
    const void* GATE, const void* SINK, void* OP, void* LP, void* SEM, void* Y, const void* NLivePtr,
    int64_t sq_t, int64_t sq_h, int64_t sr_t, int64_t sr_h,
    int64_t sk_b_bytes, int64_t sk_s, int64_t sk_h_bytes, int64_t sv_b_bytes, int64_t sv_s, int64_t sv_h_bytes,
    int64_t sbt_r, int64_t sg_t, int64_t ssk_t, int64_t sy_t,
    int64_t so_s, int64_t so_t, int64_t so_h, int64_t sl_s, int64_t sl_t, int64_t sl_h,
    float scale_log2, float rel_threshold, float clip_eps,
    int T, int NT, int mtiles, int num_reqs, int Hkv, int BS, int E, int G,
    int window, int num_splits_short, int has_tau, int has_nlive, int use_pdl, int trig_wo, int tree_mode,
    int num_splits_long, int kv_tok_thresh, int kv_evict_first_long,
    uint64_t stream_u64,
    const void* KEmergRaw, const void* VEmergRaw
) {
    return dispatch_paired_v13(
        make_params(Q, R, RScale, Tau, RelProjT, K, V, BT, SEQ, CUQ, GATE, SINK, OP, LP, SEM, Y, NLivePtr, nullptr,
                    sq_t, sq_h, sr_t, sr_h, sk_b_bytes, sk_s, sk_h_bytes, sv_b_bytes, sv_s, sv_h_bytes,
                    sbt_r, sg_t, ssk_t, sy_t, so_s, so_t, so_h, sl_s, sl_t, sl_h,
                    scale_log2, rel_threshold, clip_eps,
                    T, NT, mtiles, num_reqs, Hkv, BS, E, G, window, num_splits_short, has_tau, has_nlive,
                    use_pdl, trig_wo, 0, kv_evict_first_long, stream_u64, KEmergRaw, VEmergRaw),
        true, tree_mode, num_splits_short, num_splits_long, kv_tok_thresh, kv_evict_first_long);
}
