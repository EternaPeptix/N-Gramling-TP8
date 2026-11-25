#!/usr/bin/env python3
"""Bit-level LoRA merge reconstruction, BF16 ULP/quantization residual analysis, and W_base orthogonality."""
import argparse
import json
import pathlib
import struct
import numpy as np

TARGET_LAYERS = [0, 5, 6, 23, 44, 47]
PROJS = [("wq_du", "q_proj"), ("wk_dv", "k_proj"), ("wv_dv", "v_proj"), ("wo_ud", "o_proj"), ("gate_attn", "gate_proj")]


def u16_to_f32(u16: np.ndarray) -> np.ndarray:
    return (u16.astype(np.uint32) << 16).view(np.float32)


def bf16_to_ord(u: np.ndarray) -> np.ndarray:
    i = u.astype(np.int32)
    return np.where(i >= 0x8000, -(i - 0x8000), i)


def f64_to_bf16_u16_rne(arr_f64: np.ndarray) -> np.ndarray:
    """Exact single-rounding IEEE-754 float64 -> bfloat16 (round-to-nearest-even)."""
    u64 = arr_f64.astype(np.float64).view(np.uint64)
    sign = (u64 >> 63) & 1
    exp64 = ((u64 >> 52) & 0x7FF).astype(np.int64)
    frac64 = u64 & ((1 << 52) - 1)
    exp_bf = exp64 - 1023 + 127
    top7 = frac64 >> 45
    round_bit = (frac64 >> 44) & 1
    sticky = (frac64 & ((1 << 44) - 1)) != 0
    inc = (round_bit == 1) & (sticky | ((top7 & 1) == 1))
    bf = ((sign << 15) | (exp_bf.astype(np.uint64) << 7) | top7) + inc.astype(np.uint64)
    return np.where((exp64 == 0) & (frac64 == 0), sign << 15, bf).astype(np.uint16)


def load_lora(path: pathlib.Path) -> dict:
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hlen))
        base = 8 + hlen
        out = {}
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            s, e = v["data_offsets"]
            f.seek(base + s)
            u16 = np.frombuffer(f.read(e - s), dtype=np.uint16)
            out[k] = u16_to_f32(u16).reshape(v["shape"]).copy()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", type=pathlib.Path, default=pathlib.Path("./work"))
    args = ap.parse_args()
    w = args.work_dir

    manifest = json.loads((w / "MANIFEST.json").read_text())
    adapter_cfg = json.loads((w / "adapter_config.json").read_text())
    scale = adapter_cfg["lora_alpha"] / adapter_cfg["r"]
    lora = load_lora(w / "adapter_model.safetensors")
    svds = np.load(w / "lora_svds.npz")
    wdir = w / "verified_weights"

    rows = []
    tot_elems, tot_diff = 0, 0
    for l in TARGET_LAYERS:
        for mp, pp in PROJS:
            mk = f"model.llm.layers.{l}.attn.{mp}.weight"
            shp = tuple(manifest["tensors"][mk]["shape"])
            wu_u16 = np.frombuffer((wdir / f"L{l}_{mp}_u.bin").read_bytes(), dtype=np.uint16).reshape(shp)
            wb_u16 = np.frombuffer((wdir / f"L{l}_{mp}_b.bin").read_bytes(), dtype=np.uint16).reshape(shp)
            wu, wb = u16_to_f32(wu_u16), u16_to_f32(wb_u16)

            A = lora[f"base_model.model.model.layers.{l}.self_attn.{pp}.lora_A.weight"].astype(np.float64)
            B = lora[f"base_model.model.model.layers.{l}.self_attn.{pp}.lora_B.weight"].astype(np.float64)
            dw_lora = scale * (B @ A)
            pred_u16 = f64_to_bf16_u16_rne(wb.astype(np.float64) + dw_lora)
            tot_elems += wu_u16.size
            tot_diff += int(np.sum(wu_u16 != pred_u16))

            ulp = np.abs(bf16_to_ord(wu_u16) - bf16_to_ord(wb_u16))
            dw_act = (wu - wb).astype(np.float64)
            nb, nu = np.linalg.norm(wb.astype(np.float64)), np.linalg.norm(wu.astype(np.float64))
            ndw_act, ndw_lora = np.linalg.norm(dw_act), np.linalg.norm(dw_lora)

            U = svds[f"L{l}_{mp}_U"].astype(np.float64)
            S = svds[f"L{l}_{mp}_S"].astype(np.float64)
            Vt = svds[f"L{l}_{mp}_Vt"].astype(np.float64)
            u1, v1, s1 = U[:, 0], Vt[0, :], S[0]
            wb_v1 = wb.astype(np.float64) @ v1
            proj_core = U.T @ (dw_act @ Vt.T)

            rows.append({
                "layer": l, "proj": mp,
                "rel_act": float(ndw_act / nb), "rel_lora": float(ndw_lora / nb),
                "p0_ulp": float(np.mean(ulp == 0) * 100),
                "p1_ulp": float(np.mean(ulp == 1) * 100),
                "p2p_ulp": float(np.mean(ulp >= 2) * 100),
                "qnoise_frac": float(np.linalg.norm(dw_act - dw_lora) / ndw_lora),
                "in_subspace_energy": float(np.sum(proj_core**2) / (ndw_act**2)),
                "s1_preserve": float(np.linalg.svd(proj_core, compute_uv=False)[0] / s1),
                "cos_dw_wb": float(np.sum(dw_lora * wb.astype(np.float64)) / (ndw_lora * nb)),
                "dnorm_pct": float((nu - nb) / nb * 100),
                "cos_wbv1_u1": float((u1 @ wb_v1) / (np.linalg.norm(wb_v1) + 1e-30)),
            })

    print(f"Verified {tot_elems:,} parameters across 30 tensors: exact bit match = {(1 - tot_diff / tot_elems) * 100:.6f}% ({tot_diff:,} 1-ULP diffs)")
    print(f"Mean ULP split: 0-ULP={np.mean([r['p0_ulp'] for r in rows]):.2f}%, 1-ULP={np.mean([r['p1_ulp'] for r in rows]):.2f}%, >=2-ULP={np.mean([r['p2p_ulp'] for r in rows]):.2f}%")
    print(f"Mean BF16 quant noise ||E||/||dW_lora|| = {np.mean([r['qnoise_frac'] for r in rows])*100:.2f}% | rank-32 energy retained = {np.mean([r['in_subspace_energy'] for r in rows])*100:.2f}% | s1 preserved = {np.mean([r['s1_preserve'] for r in rows])*100:.2f}%")
    print(f"Mean cos(dW_lora, W_base) = {np.mean([r['cos_dw_wb'] for r in rows]):+.5f} | mean d||W||_F = {np.mean([r['dnorm_pct'] for r in rows]):+.5f}%")
    (w / "verified_weights_summary.json").write_text(json.dumps(rows, indent=2))
    print(f"Saved {w / 'verified_weights_summary.json'}")


if __name__ == "__main__":
    main()
