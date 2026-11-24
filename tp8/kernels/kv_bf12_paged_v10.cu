// SPDX-License-Identifier: Apache-2.0
// SM121a (GB10) Production 0-ULP Lossless KV-BF12 (BF12-G128-OVF-SAFE) Paged KV Runtime Library v10
//
// Compiles to `libkv_bf12_paged_v10.so` with strict SM121a hygiene:
//   nvcc -O3 -std=c++17 -gencode arch=compute_121a,code=sm_121a --fmad=false \
//        --shared -Xcompiler -fPIC --ptxas-options=-v kv_bf12_paged_v10.cu -o libkv_bf12_paged_v10.so
//
// Exported C ABI:
//   1. kv_bf12_version()                 -> returns 10
//   2. kv_bf12_reset_blocks_v10_fwd(...) -> batched GPU reset of blk.ovf_used (and optional meta[256])
//   3. kv_bf12_store_prefill_v10_fwd(...) -> warp-per-token 0-ULP BF16 -> paged KV-BF12 encoder
//   4. kv_bf12_unpack_blocks_v10_fwd(...) -> coalesced paged KV-BF12 -> BF16 [N_blks, 256, Hkv, 128]
//   5. kv_bf12_unpack_range_v10_fwd(...)  -> coalesced paged KV-BF12 -> BF16 [N_tok, Hkv, 128]
//
// Zero PDL hazards (pure stream-ordered prefill/maintenance kernels), 0 B stack, 0 B spills.

#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

#include "kv_bf12_codec_v10.cuh"

namespace {

// =========================================================================
// Kernel 1: Batched Block Reset (`ovf_used = 0` for K and V halves)
//   - Fast path (`zero_meta == 0`): 1 thread per `(block_idx, kv_head)` -> 2x 4-byte stores!
//   - Full header path (`zero_meta != 0`): 1 CTA (256 threads) per `(block_idx, kv_head)`
//     zeros `meta[0..255]` + `ovf_used` for K and V halves.
// =========================================================================
__global__ void __launch_bounds__(256, 4)
kv_bf12_reset_blocks_fast_kernel(
    uint8_t* __restrict__ kv_base,
    int64_t blk_stride_bytes,
    int64_t head_stride_bytes,
    const int32_t* __restrict__ block_ids,
    int num_blocks,
    int Hkv
) {
    const int idx = static_cast<int>(blockIdx.x) * 256 + static_cast<int>(threadIdx.x);
    const int total = num_blocks * Hkv;
    if (idx >= total) return;

    const int b_i = idx / Hkv;
    const int h   = idx - b_i * Hkv;
    const int32_t blk_id = block_ids[b_i];
    if (blk_id < 0) return;

    uint8_t* k_base = kv_base + static_cast<int64_t>(h) * head_stride_bytes;
    uint8_t* v_base = k_base + KV_BF12_HALF_BYTES;
    kv_bf12_half_blk(k_base, blk_stride_bytes, blk_id)->ovf_used = 0u;
    kv_bf12_half_blk(v_base, blk_stride_bytes, blk_id)->ovf_used = 0u;
}

__global__ void __launch_bounds__(256, 4)
kv_bf12_reset_blocks_full_kernel(
    uint8_t* __restrict__ kv_base,
    int64_t blk_stride_bytes,
    int64_t head_stride_bytes,
    const int32_t* __restrict__ block_ids,
    int num_blocks,
    int Hkv
) {
    const int b_i = static_cast<int>(blockIdx.x);
    const int h   = static_cast<int>(blockIdx.y);
    if (b_i >= num_blocks || h >= Hkv) return;

    const int32_t blk_id = block_ids[b_i];
    if (blk_id < 0) return;

    const int tid = static_cast<int>(threadIdx.x);
    uint8_t* k_base = kv_base + static_cast<int64_t>(h) * head_stride_bytes;
    uint8_t* v_base = k_base + KV_BF12_HALF_BYTES;
    KvBf12HalfBlock* k_blk = kv_bf12_half_blk(k_base, blk_stride_bytes, blk_id);
    KvBf12HalfBlock* v_blk = kv_bf12_half_blk(v_base, blk_stride_bytes, blk_id);

    k_blk->meta[tid] = 0u;
    v_blk->meta[tid] = 0u;
    if (tid == 0) {
        k_blk->ovf_used = 0u;
        v_blk->ovf_used = 0u;
    }
}

// =========================================================================
// Kernel 2: Warp-per-Half-Token Prefill Store (BF16 -> Paged KV-BF12)
//   - 4 warps/CTA (128 threads): each warp encodes ONE `(token, head, K-or-V)` vector (128 dims).
//   - 0 atomics on Modes 0 & 1 (99.94%+ of Inkling-2 K/V tokens).
//   - Skips padding tokens (`slot < 0`) and optional `discard_blk` (`blk_id == discard_blk`).
// =========================================================================
__global__ void __launch_bounds__(128, 4)
kv_bf12_store_prefill_v10_kernel(
    const uint16_t* __restrict__ K_in,
    const uint16_t* __restrict__ V_in,
    const int64_t*  __restrict__ slot_mapping,
    uint8_t*        __restrict__ kv_base,
    int64_t blk_stride_bytes,
    int64_t head_stride_bytes,
    uint16_t* __restrict__ k_emerg_raw,
    uint32_t* __restrict__ k_emerg_used,
    uint16_t* __restrict__ v_emerg_raw,
    uint32_t* __restrict__ v_emerg_used,
    uint32_t emerg_cap,
    int T,
    int Hkv,
    int64_t sk_t, int64_t sk_h,
    int64_t sv_t, int64_t sv_h,
    int32_t discard_blk
) {
    const int wid  = static_cast<int>(threadIdx.x) >> 5;   // 0..3
    const int lane = static_cast<int>(threadIdx.x) & 31;   // 0..31
    const int work_idx = static_cast<int>(blockIdx.x) * 4 + wid;
    const int total_work = T * Hkv * 2;
    if (work_idx >= total_work) return;

    const int is_v = work_idx & 1;
    const int th   = work_idx >> 1;
    const int t    = (Hkv == 1) ? th : (th / Hkv);
    const int h    = (Hkv == 1) ? 0  : (th - t * Hkv);

    const int64_t kv_slot = slot_mapping[t];
    if (kv_slot < 0) return;

    const int64_t blk_id = kv_slot >> KV_BF12_BS_SHIFT;
    const int     slot   = static_cast<int>(kv_slot & KV_BF12_BS_MASK);
    if (discard_blk >= 0 && static_cast<int32_t>(blk_id) == discard_blk) return;

    const uint16_t* src_row = is_v
        ? (V_in + static_cast<int64_t>(t) * sv_t + static_cast<int64_t>(h) * sv_h)
        : (K_in + static_cast<int64_t>(t) * sk_t + static_cast<int64_t>(h) * sk_h);

    const uint32_t p01 = *reinterpret_cast<const uint32_t*>(src_row + lane * 4 + 0);
    const uint32_t p23 = *reinterpret_cast<const uint32_t*>(src_row + lane * 4 + 2);
    const uint32_t u0  = p01 & 0xFFFFu;
    const uint32_t u1  = p01 >> 16;
    const uint32_t u2  = p23 & 0xFFFFu;
    const uint32_t u3  = p23 >> 16;

    uint8_t* half_base = kv_base + static_cast<int64_t>(h) * head_stride_bytes
                       + (is_v ? KV_BF12_HALF_BYTES : 0);
    KvBf12HalfBlock* blk = kv_bf12_half_blk(half_base, blk_stride_bytes, blk_id);

    kv_bf12_encode_warp_store(
        u0, u1, u2, u3, lane, blk, slot,
        is_v ? v_emerg_raw : k_emerg_raw,
        is_v ? v_emerg_used : k_emerg_used,
        emerg_cap
    );
}

// =========================================================================
// Kernel 3: Coalesced Block Unpacker (Paged KV-BF12 -> BF16 [N_blks, 256, Hkv, 128])
//   - Grid: (64, num_blocks, Hkv), Block: 128 (4 warps -> 4 slots per CTA, 0 integer div!).
//   - Independent `UNPACK_K` / `UNPACK_V` templates: Shadow-K quant unpacks K only;
//     Sage2 1M prefix unpacks V only; SWA/Tail unpacks K+V.
// =========================================================================
template <bool UNPACK_K, bool UNPACK_V>
__global__ void __launch_bounds__(128, 4)
kv_bf12_unpack_blocks_v10_kernel(
    const uint8_t*  __restrict__ kv_base,
    int64_t blk_stride_bytes,
    int64_t head_stride_bytes,
    const uint16_t* __restrict__ k_emerg_raw,
    const uint16_t* __restrict__ v_emerg_raw,
    const int32_t*  __restrict__ block_ids,
    uint16_t*       __restrict__ K_out,
    uint16_t*       __restrict__ V_out,
    int64_t ok_b, int64_t ok_s, int64_t ok_h,
    int64_t ov_b, int64_t ov_s, int64_t ov_h
) {
    const int wid  = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int slot = static_cast<int>(blockIdx.x) * 4 + wid;  // 0..255
    const int b_i  = static_cast<int>(blockIdx.y);
    const int h    = static_cast<int>(blockIdx.z);

    const int32_t blk_id = block_ids[b_i];
    const uint8_t* k_base = kv_base + static_cast<int64_t>(h) * head_stride_bytes;

    if constexpr (UNPACK_K) {
        uint16_t* k_dst = K_out
                        + static_cast<int64_t>(b_i) * ok_b
                        + static_cast<int64_t>(slot) * ok_s
                        + static_cast<int64_t>(h) * ok_h;
        if (blk_id >= 0) {
            const KvBf12HalfBlock* k_blk = kv_bf12_half_blk_const(k_base, blk_stride_bytes, blk_id);
            kv_bf12_unpack_warp_row(k_blk, slot, k_emerg_raw, k_dst, lane);
        } else {
            *reinterpret_cast<uint32_t*>(k_dst + lane * 4 + 0) = 0u;
            *reinterpret_cast<uint32_t*>(k_dst + lane * 4 + 2) = 0u;
        }
    } else {
        (void)k_emerg_raw; (void)K_out; (void)ok_b; (void)ok_s; (void)ok_h;
    }

    if constexpr (UNPACK_V) {
        uint16_t* v_dst = V_out
                        + static_cast<int64_t>(b_i) * ov_b
                        + static_cast<int64_t>(slot) * ov_s
                        + static_cast<int64_t>(h) * ov_h;
        if (blk_id >= 0) {
            const KvBf12HalfBlock* v_blk = kv_bf12_half_blk_const(
                k_base + KV_BF12_HALF_BYTES, blk_stride_bytes, blk_id
            );
            kv_bf12_unpack_warp_row(v_blk, slot, v_emerg_raw, v_dst, lane);
        } else {
            *reinterpret_cast<uint32_t*>(v_dst + lane * 4 + 0) = 0u;
            *reinterpret_cast<uint32_t*>(v_dst + lane * 4 + 2) = 0u;
        }
    } else {
        (void)v_emerg_raw; (void)V_out; (void)ov_b; (void)ov_s; (void)ov_h;
    }
}

// =========================================================================
// Kernel 4: Coalesced Token-Range Unpacker (Paged KV-BF12 -> BF16 [N_tok, Hkv, 128])
//   - Grid: ((num_tokens + 3) / 4, Hkv, 1), Block: 128 (4 warps -> 4 tokens per CTA).
//   - Reconstructs contiguous BF16 slice `[start_pos .. start_pos + num_tokens)`.
// =========================================================================
template <bool UNPACK_K, bool UNPACK_V>
__global__ void __launch_bounds__(128, 4)
kv_bf12_unpack_range_v10_kernel(
    const uint8_t*  __restrict__ kv_base,
    int64_t blk_stride_bytes,
    int64_t head_stride_bytes,
    const uint16_t* __restrict__ k_emerg_raw,
    const uint16_t* __restrict__ v_emerg_raw,
    const int32_t*  __restrict__ block_table,
    int start_pos,
    int num_tokens,
    int max_blocks,
    uint16_t*       __restrict__ K_out,
    uint16_t*       __restrict__ V_out,
    int64_t ok_t, int64_t ok_h,
    int64_t ov_t, int64_t ov_h
) {
    const int wid  = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int t    = static_cast<int>(blockIdx.x) * 4 + wid;
    const int h    = static_cast<int>(blockIdx.y);
    if (t >= num_tokens) return;

    const int pos    = start_pos + t;
    const int lblk   = pos >> KV_BF12_BS_SHIFT;
    const int slot   = pos & KV_BF12_BS_MASK;
    const int32_t blk_id = (pos >= 0 && lblk < max_blocks) ? __ldg(block_table + lblk) : -1;
    const uint8_t* k_base = kv_base + static_cast<int64_t>(h) * head_stride_bytes;

    if constexpr (UNPACK_K) {
        uint16_t* k_dst = K_out + static_cast<int64_t>(t) * ok_t + static_cast<int64_t>(h) * ok_h;
        if (blk_id >= 0) {
            const KvBf12HalfBlock* k_blk = kv_bf12_half_blk_const(k_base, blk_stride_bytes, blk_id);
            kv_bf12_unpack_warp_row(k_blk, slot, k_emerg_raw, k_dst, lane);
        } else {
            *reinterpret_cast<uint32_t*>(k_dst + lane * 4 + 0) = 0u;
            *reinterpret_cast<uint32_t*>(k_dst + lane * 4 + 2) = 0u;
        }
    } else {
        (void)k_emerg_raw; (void)K_out; (void)ok_t; (void)ok_h;
    }

    if constexpr (UNPACK_V) {
        uint16_t* v_dst = V_out + static_cast<int64_t>(t) * ov_t + static_cast<int64_t>(h) * ov_h;
        if (blk_id >= 0) {
            const KvBf12HalfBlock* v_blk = kv_bf12_half_blk_const(
                k_base + KV_BF12_HALF_BYTES, blk_stride_bytes, blk_id
            );
            kv_bf12_unpack_warp_row(v_blk, slot, v_emerg_raw, v_dst, lane);
        } else {
            *reinterpret_cast<uint32_t*>(v_dst + lane * 4 + 0) = 0u;
            *reinterpret_cast<uint32_t*>(v_dst + lane * 4 + 2) = 0u;
        }
    } else {
        (void)v_emerg_raw; (void)V_out; (void)ov_t; (void)ov_h;
    }
}

}  // namespace

// =========================================================================
// C ABI Exports
// =========================================================================
extern "C" int kv_bf12_version(void) {
    return 10;
}

extern "C" int kv_bf12_reset_blocks_v10_fwd(
    uint64_t kv_base_u64,
    int64_t  blk_stride_bytes,
    int64_t  head_stride_bytes,
    uint64_t block_ids_u64,
    int      num_blocks,
    int      Hkv,
    int      zero_meta,
    uint64_t stream_u64
) {
    if (num_blocks <= 0 || Hkv <= 0) return 0;
    if (kv_base_u64 == 0ULL || block_ids_u64 == 0ULL) return -1;
    if (blk_stride_bytes < KV_BF12_PAGE_BYTES) return -2;

    uint8_t* kv_base = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(kv_base_u64));
    const int32_t* block_ids = reinterpret_cast<const int32_t*>(static_cast<uintptr_t>(block_ids_u64));
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(stream_u64));

    if (zero_meta == 0) {
        const int total = num_blocks * Hkv;
        const dim3 grid(static_cast<unsigned>((total + 255) / 256), 1u, 1u);
        const dim3 block(256u, 1u, 1u);
        kv_bf12_reset_blocks_fast_kernel<<<grid, block, 0, stream>>>(
            kv_base, blk_stride_bytes, head_stride_bytes, block_ids, num_blocks, Hkv
        );
    } else {
        const dim3 grid(static_cast<unsigned>(num_blocks), static_cast<unsigned>(Hkv), 1u);
        const dim3 block(256u, 1u, 1u);
        kv_bf12_reset_blocks_full_kernel<<<grid, block, 0, stream>>>(
            kv_base, blk_stride_bytes, head_stride_bytes, block_ids, num_blocks, Hkv
        );
    }
    return (cudaGetLastError() == cudaSuccess) ? 0 : -3;
}

extern "C" int kv_bf12_store_prefill_v10_fwd(
    uint64_t k_in_u64,
    uint64_t v_in_u64,
    uint64_t slot_mapping_u64,
    uint64_t kv_base_u64,
    int64_t  blk_stride_bytes,
    int64_t  head_stride_bytes,
    uint64_t k_emerg_raw_u64,
    uint64_t k_emerg_used_u64,
    uint64_t v_emerg_raw_u64,
    uint64_t v_emerg_used_u64,
    uint32_t emerg_cap,
    int      T,
    int      Hkv,
    int64_t  sk_t, int64_t sk_h,
    int64_t  sv_t, int64_t sv_h,
    int32_t  discard_blk,
    uint64_t stream_u64
) {
    if (T <= 0 || Hkv <= 0) return 0;
    if (k_in_u64 == 0ULL || v_in_u64 == 0ULL || slot_mapping_u64 == 0ULL || kv_base_u64 == 0ULL) {
        return -1;
    }
    if (blk_stride_bytes < KV_BF12_PAGE_BYTES) return -2;

    const uint16_t* K_in = reinterpret_cast<const uint16_t*>(static_cast<uintptr_t>(k_in_u64));
    const uint16_t* V_in = reinterpret_cast<const uint16_t*>(static_cast<uintptr_t>(v_in_u64));
    const int64_t*  sm   = reinterpret_cast<const int64_t*>(static_cast<uintptr_t>(slot_mapping_u64));
    uint8_t* kv_base     = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(kv_base_u64));
    uint16_t* k_eraw     = reinterpret_cast<uint16_t*>(static_cast<uintptr_t>(k_emerg_raw_u64));
    uint32_t* k_eused    = reinterpret_cast<uint32_t*>(static_cast<uintptr_t>(k_emerg_used_u64));
    uint16_t* v_eraw     = reinterpret_cast<uint16_t*>(static_cast<uintptr_t>(v_emerg_raw_u64));
    uint32_t* v_eused    = reinterpret_cast<uint32_t*>(static_cast<uintptr_t>(v_emerg_used_u64));
    cudaStream_t stream  = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(stream_u64));

    const int total_work = T * Hkv * 2;
    const dim3 grid(static_cast<unsigned>((total_work + 3) / 4), 1u, 1u);
    const dim3 block(128u, 1u, 1u);
    kv_bf12_store_prefill_v10_kernel<<<grid, block, 0, stream>>>(
        K_in, V_in, sm, kv_base, blk_stride_bytes, head_stride_bytes,
        k_eraw, k_eused, v_eraw, v_eused, emerg_cap,
        T, Hkv, sk_t, sk_h, sv_t, sv_h, discard_blk
    );
    return (cudaGetLastError() == cudaSuccess) ? 0 : -3;
}

extern "C" int kv_bf12_unpack_blocks_v10_fwd(
    uint64_t kv_base_u64,
    int64_t  blk_stride_bytes,
    int64_t  head_stride_bytes,
    uint64_t k_emerg_raw_u64,
    uint64_t v_emerg_raw_u64,
    uint64_t block_ids_u64,
    uint64_t k_out_u64,
    uint64_t v_out_u64,
    int64_t  ok_b, int64_t ok_s, int64_t ok_h,
    int64_t  ov_b, int64_t ov_s, int64_t ov_h,
    int      num_blocks,
    int      Hkv,
    int      unpack_k,
    int      unpack_v,
    uint64_t stream_u64
) {
    if (num_blocks <= 0 || Hkv <= 0 || (!unpack_k && !unpack_v)) return 0;
    if (kv_base_u64 == 0ULL || block_ids_u64 == 0ULL) return -1;
    if (unpack_k && k_out_u64 == 0ULL) return -1;
    if (unpack_v && v_out_u64 == 0ULL) return -1;
    if (blk_stride_bytes < KV_BF12_PAGE_BYTES) return -2;

    const uint8_t*  kv_base = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(kv_base_u64));
    const uint16_t* k_eraw  = reinterpret_cast<const uint16_t*>(static_cast<uintptr_t>(k_emerg_raw_u64));
    const uint16_t* v_eraw  = reinterpret_cast<const uint16_t*>(static_cast<uintptr_t>(v_emerg_raw_u64));
    const int32_t*  bids    = reinterpret_cast<const int32_t*>(static_cast<uintptr_t>(block_ids_u64));
    uint16_t*       K_out   = reinterpret_cast<uint16_t*>(static_cast<uintptr_t>(k_out_u64));
    uint16_t*       V_out   = reinterpret_cast<uint16_t*>(static_cast<uintptr_t>(v_out_u64));
    cudaStream_t    stream  = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(stream_u64));

    const dim3 grid(64u, static_cast<unsigned>(num_blocks), static_cast<unsigned>(Hkv));
    const dim3 block(128u, 1u, 1u);
    if (unpack_k && unpack_v) {
        kv_bf12_unpack_blocks_v10_kernel<true, true><<<grid, block, 0, stream>>>(
            kv_base, blk_stride_bytes, head_stride_bytes, k_eraw, v_eraw, bids,
            K_out, V_out, ok_b, ok_s, ok_h, ov_b, ov_s, ov_h
        );
    } else if (unpack_k) {
        kv_bf12_unpack_blocks_v10_kernel<true, false><<<grid, block, 0, stream>>>(
            kv_base, blk_stride_bytes, head_stride_bytes, k_eraw, v_eraw, bids,
            K_out, V_out, ok_b, ok_s, ok_h, ov_b, ov_s, ov_h
        );
    } else {
        kv_bf12_unpack_blocks_v10_kernel<false, true><<<grid, block, 0, stream>>>(
            kv_base, blk_stride_bytes, head_stride_bytes, k_eraw, v_eraw, bids,
            K_out, V_out, ok_b, ok_s, ok_h, ov_b, ov_s, ov_h
        );
    }
    return (cudaGetLastError() == cudaSuccess) ? 0 : -3;
}

extern "C" int kv_bf12_unpack_range_v10_fwd(
    uint64_t kv_base_u64,
    int64_t  blk_stride_bytes,
    int64_t  head_stride_bytes,
    uint64_t k_emerg_raw_u64,
    uint64_t v_emerg_raw_u64,
    uint64_t block_table_u64,
    int      start_pos,
    int      num_tokens,
    int      max_blocks,
    uint64_t k_out_u64,
    uint64_t v_out_u64,
    int64_t  ok_t, int64_t ok_h,
    int64_t  ov_t, int64_t ov_h,
    int      Hkv,
    int      unpack_k,
    int      unpack_v,
    uint64_t stream_u64
) {
    if (num_tokens <= 0 || Hkv <= 0 || (!unpack_k && !unpack_v)) return 0;
    if (kv_base_u64 == 0ULL || block_table_u64 == 0ULL || max_blocks <= 0) return -1;
    if (unpack_k && k_out_u64 == 0ULL) return -1;
    if (unpack_v && v_out_u64 == 0ULL) return -1;
    if (blk_stride_bytes < KV_BF12_PAGE_BYTES) return -2;

    const uint8_t*  kv_base = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(kv_base_u64));
    const uint16_t* k_eraw  = reinterpret_cast<const uint16_t*>(static_cast<uintptr_t>(k_emerg_raw_u64));
    const uint16_t* v_eraw  = reinterpret_cast<const uint16_t*>(static_cast<uintptr_t>(v_emerg_raw_u64));
    const int32_t*  bt      = reinterpret_cast<const int32_t*>(static_cast<uintptr_t>(block_table_u64));
    uint16_t*       K_out   = reinterpret_cast<uint16_t*>(static_cast<uintptr_t>(k_out_u64));
    uint16_t*       V_out   = reinterpret_cast<uint16_t*>(static_cast<uintptr_t>(v_out_u64));
    cudaStream_t    stream  = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(stream_u64));

    const dim3 grid(static_cast<unsigned>((num_tokens + 3) / 4), static_cast<unsigned>(Hkv), 1u);
    const dim3 block(128u, 1u, 1u);
    if (unpack_k && unpack_v) {
        kv_bf12_unpack_range_v10_kernel<true, true><<<grid, block, 0, stream>>>(
            kv_base, blk_stride_bytes, head_stride_bytes, k_eraw, v_eraw, bt,
            start_pos, num_tokens, max_blocks, K_out, V_out, ok_t, ok_h, ov_t, ov_h
        );
    } else if (unpack_k) {
        kv_bf12_unpack_range_v10_kernel<true, false><<<grid, block, 0, stream>>>(
            kv_base, blk_stride_bytes, head_stride_bytes, k_eraw, v_eraw, bt,
            start_pos, num_tokens, max_blocks, K_out, V_out, ok_t, ok_h, ov_t, ov_h
        );
    } else {
        kv_bf12_unpack_range_v10_kernel<false, true><<<grid, block, 0, stream>>>(
            kv_base, blk_stride_bytes, head_stride_bytes, k_eraw, v_eraw, bt,
            start_pos, num_tokens, max_blocks, K_out, V_out, ok_t, ok_h, ov_t, ov_h
        );
    }
    return (cudaGetLastError() == cudaSuccess) ? 0 : -3;
}
