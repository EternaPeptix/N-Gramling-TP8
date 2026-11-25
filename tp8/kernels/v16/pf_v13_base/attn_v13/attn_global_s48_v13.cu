// SPDX-License-Identifier: Apache-2.0
// Wave-V13 Split TU 3/5: Global BF16 Decode Attention S=1..48 (`attn_global_s48_v13.cu`)
//   ONLY 24 kernels total (S_MODE in {0: S=1, 1: 2..32, 2: 33..64 [S=48!]}, MT_MAX in {1,2,3}, PDL=0/1).
//   Factored zero-duplication split-phase pipeline + runtime guard_mode -> 100% cicc hang-free!

#define ATTN_V13_DEVICE_TU 1
#include "attn_common_v13.cuh"

namespace attn_v13 {

int init_smem_global_bf16_v13() {
    cudaError_t err;
    if ((err = cudaFuncSetAttribute(fused_decode_attn_otf_fusedmt_kernel_v13<-1, 0, 3, true,  false>, cudaFuncAttributeMaxDynamicSharedMemorySize, kFusedMt3SmemBytes)) != cudaSuccess) return static_cast<int>(err);
    if ((err = cudaFuncSetAttribute(fused_decode_attn_otf_fusedmt_kernel_v13<-1, 0, 3, false, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, kFusedMt3SmemBytes)) != cudaSuccess) return static_cast<int>(err);
    if ((err = cudaFuncSetAttribute(fused_decode_attn_otf_fusedmt_kernel_v13<-1, 1, 3, true,  false>, cudaFuncAttributeMaxDynamicSharedMemorySize, kFusedMt3SmemBytes)) != cudaSuccess) return static_cast<int>(err);
    if ((err = cudaFuncSetAttribute(fused_decode_attn_otf_fusedmt_kernel_v13<-1, 1, 3, false, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, kFusedMt3SmemBytes)) != cudaSuccess) return static_cast<int>(err);
    if ((err = cudaFuncSetAttribute(fused_decode_attn_otf_fusedmt_kernel_v13<-1, 2, 3, true,  false>, cudaFuncAttributeMaxDynamicSharedMemorySize, kFusedMt3SmemBytes)) != cudaSuccess) return static_cast<int>(err);
    if ((err = cudaFuncSetAttribute(fused_decode_attn_otf_fusedmt_kernel_v13<-1, 2, 3, false, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, kFusedMt3SmemBytes)) != cudaSuccess) return static_cast<int>(err);
    return 0;
}

int launch_global_bf16_mgs_v13(const AttnLaunchParams& p, bool has_tree) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(p.stream_u64);
    const int bs_shift = (p.BS > 0 && (p.BS & (p.BS - 1)) == 0) ? __builtin_ctz(p.BS) : -1;
    dim3 grid(
        static_cast<unsigned>(has_tree ? 4 : p.mtiles),
        static_cast<unsigned>(has_tree ? 1 : p.num_reqs),
        static_cast<unsigned>(p.Hkv * p.num_splits)
    );
    dim3 block(128, 1, 1);
    static constexpr size_t SMEM_BYTES = 39936;
    const bool pdl = (p.use_pdl != 0);

    #define LA_GBF16(SM, PD, TR) \
        return static_cast<int>(launch_maybe_pdl( \
            fused_decode_attn_otf_mgs_kernel_v13<-1, SM, PD, TR, false>, \
            grid, block, SMEM_BYTES, stream, PD, \
            reinterpret_cast<const uint16_t*>(p.Q), reinterpret_cast<const uint16_t*>(p.R), \
            reinterpret_cast<const uint16_t*>(p.RScale), reinterpret_cast<const float*>(p.Tau), \
            reinterpret_cast<const uint16_t*>(p.RelProjT), \
            reinterpret_cast<const uint16_t*>(p.K), reinterpret_cast<const uint16_t*>(p.V), \
            reinterpret_cast<const int32_t*>(p.BT), reinterpret_cast<const int32_t*>(p.SEQ), \
            reinterpret_cast<const int32_t*>(p.CUQ), \
            reinterpret_cast<const uint16_t*>(p.GATE), reinterpret_cast<const uint16_t*>(p.SINK), \
            reinterpret_cast<float*>(p.OP), reinterpret_cast<float*>(p.LP), \
            reinterpret_cast<int32_t*>(p.SEM), reinterpret_cast<uint16_t*>(p.Y), \
            reinterpret_cast<const int32_t*>(p.NLivePtr), reinterpret_cast<const int32_t*>(p.TreeMeta), \
            p.sq_t, p.sq_h, p.sr_t, p.sr_h, p.sk_b, p.sk_s, p.sk_h, p.sv_b, p.sv_s, p.sv_h, \
            p.sbt_r, p.sg_t, p.ssk_t, p.sy_t, p.so_s, p.so_t, p.so_h, p.sl_s, p.sl_t, p.sl_h, \
            p.scale_log2, p.rel_threshold, p.clip_eps, \
            p.T, p.NT, p.BS, bs_shift, p.E, p.G, p.num_splits, p.has_tau, p.has_nlive, p.trig_wo, \
            p.guard_mode, p.kv_tok_thresh, nullptr, nullptr))

    if (has_tree) {
        if (p.num_splits != 16) return -2;
        if (pdl) { LA_GBF16(1, true,  true); }
        else     { LA_GBF16(1, false, true); }
    }
    if (p.num_splits == 1) {
        if (pdl) { LA_GBF16(0, true,  false); }
        else     { LA_GBF16(0, false, false); }
    }
    if (p.num_splits >= 2 && p.num_splits <= 32) {
        if (pdl) { LA_GBF16(1, true,  false); }
        else     { LA_GBF16(1, false, false); }
    }
    if (p.num_splits > 32 && p.num_splits <= 64) {
        if (pdl) { LA_GBF16(2, true,  false); }
        else     { LA_GBF16(2, false, false); }
    }
    return -2;
    #undef LA_GBF16
}

int launch_global_bf16_fusedmt_v13(const AttnLaunchParams& p, bool smem_inited) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(p.stream_u64);
    const int bs_shift = (p.BS > 0 && (p.BS & (p.BS - 1)) == 0) ? __builtin_ctz(p.BS) : -1;
    dim3 grid(1, static_cast<unsigned>(p.num_reqs), static_cast<unsigned>(p.Hkv * p.num_splits));
    dim3 block(128, 1, 1);
    const size_t red_bytes = (p.mtiles <= 2) ? (64 * 17 * sizeof(float)) : (64 * 8 * sizeof(float));
    const size_t smem_bytes = 32768 + static_cast<size_t>(p.mtiles) * 4096 + 2304 + red_bytes + 256;
    const bool pdl = (p.use_pdl != 0);

    #define LF_GBF16(SM, MT, PD) \
        do { \
            if constexpr ((MT) >= 3) { \
                static bool s_ok = false; \
                if (!smem_inited && !s_ok) { \
                    cudaStreamCaptureStatus cap_st = cudaStreamCaptureStatusNone; \
                    if (cudaStreamIsCapturing(stream, &cap_st) == cudaSuccess && cap_st != cudaStreamCaptureStatusNone) { \
                        return -3; \
                    } \
                    cudaError_t ae = cudaFuncSetAttribute( \
                        fused_decode_attn_otf_fusedmt_kernel_v13<-1, SM, MT, PD, false>, \
                        cudaFuncAttributeMaxDynamicSharedMemorySize, kFusedMt3SmemBytes); \
                    if (ae != cudaSuccess) return static_cast<int>(ae); \
                    s_ok = true; \
                } \
            } \
            return static_cast<int>(launch_maybe_pdl( \
                fused_decode_attn_otf_fusedmt_kernel_v13<-1, SM, MT, PD, false>, \
                grid, block, smem_bytes, stream, PD, \
                reinterpret_cast<const uint16_t*>(p.Q), reinterpret_cast<const uint16_t*>(p.R), \
                reinterpret_cast<const uint16_t*>(p.RScale), reinterpret_cast<const float*>(p.Tau), \
                reinterpret_cast<const uint16_t*>(p.RelProjT), \
                reinterpret_cast<const uint16_t*>(p.K), reinterpret_cast<const uint16_t*>(p.V), \
                reinterpret_cast<const int32_t*>(p.BT), reinterpret_cast<const int32_t*>(p.SEQ), \
                reinterpret_cast<const int32_t*>(p.CUQ), \
                reinterpret_cast<const uint16_t*>(p.GATE), reinterpret_cast<const uint16_t*>(p.SINK), \
                reinterpret_cast<float*>(p.OP), reinterpret_cast<float*>(p.LP), \
                reinterpret_cast<int32_t*>(p.SEM), reinterpret_cast<uint16_t*>(p.Y), \
                reinterpret_cast<const int32_t*>(p.NLivePtr), \
                p.sq_t, p.sq_h, p.sr_t, p.sr_h, p.sk_b, p.sk_s, p.sk_h, p.sv_b, p.sv_s, p.sv_h, \
                p.sbt_r, p.sg_t, p.ssk_t, p.sy_t, p.so_s, p.so_t, p.so_h, p.sl_s, p.sl_t, p.sl_h, \
                p.scale_log2, p.rel_threshold, p.clip_eps, \
                p.T, p.NT, p.BS, bs_shift, p.E, p.G, p.num_splits, p.has_tau, p.has_nlive, \
                p.kv_evict_first, p.trig_wo, p.guard_mode, p.kv_tok_thresh, nullptr, nullptr)); \
        } while (0)

    #define DISP_MT123(SM) \
        if (p.mtiles == 1)      { if (pdl) { LF_GBF16(SM, 1, true); } else { LF_GBF16(SM, 1, false); } } \
        else if (p.mtiles == 2) { if (pdl) { LF_GBF16(SM, 2, true); } else { LF_GBF16(SM, 2, false); } } \
        else if (p.mtiles == 3) { if (pdl) { LF_GBF16(SM, 3, true); } else { LF_GBF16(SM, 3, false); } }

    #define DISP_MT23(SM) \
        if (p.mtiles == 2)      { if (pdl) { LF_GBF16(SM, 2, true); } else { LF_GBF16(SM, 2, false); } } \
        else if (p.mtiles == 3) { if (pdl) { LF_GBF16(SM, 3, true); } else { LF_GBF16(SM, 3, false); } }

    if (p.num_splits == 1)                       { DISP_MT123(0); }
    if (p.num_splits >= 2 && p.num_splits <= 32) { DISP_MT123(1); }
    if (p.num_splits > 32 && p.num_splits <= 64) { DISP_MT23(2); }
    return -2;
    #undef DISP_MT23
    #undef DISP_MT123
    #undef LF_GBF16
}

}  // namespace attn_v13
