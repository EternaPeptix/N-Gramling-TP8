#!/usr/bin/env python3
"""Compute exact QR-based SVDs for all 240 LoRA deltas, rank spectra, depth profiles, and BBP spikes in LoRA A."""
import argparse
import json
import pathlib
import struct
import time
import numpy as np

PROJ_MAP = {
    "q_proj": "wq_du",
    "k_proj": "wk_dv",
    "v_proj": "wv_dv",
    "o_proj": "wo_ud",
    "gate_proj": "gate_attn",
}


def load_safetensors_bf16_as_f32(path: pathlib.Path) -> dict:
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
            raw = f.read(e - s)
            if v["dtype"] == "BF16":
                u16 = np.frombuffer(raw, dtype=np.uint16)
                arr = (u16.astype(np.uint32) << 16).view(np.float32).reshape(v["shape"]).copy()
            elif v["dtype"] == "F32":
                arr = np.frombuffer(raw, dtype=np.float32).reshape(v["shape"]).copy()
            else:
                raise ValueError(f"Unsupported dtype {v['dtype']} for {k}")
            out[k] = arr
    return out


def lora_svd(A: np.ndarray, B: np.ndarray, scale: float = 1.0):
    Qb, Rb = np.linalg.qr(B.astype(np.float64), mode="reduced")
    Qa, Ra = np.linalg.qr(A.T.astype(np.float64), mode="reduced")
    M = (scale * Rb) @ Ra.T
    Um, S, Vmt = np.linalg.svd(M, full_matrices=False)
    return Qb @ Um, S, Vmt @ Qa.T


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", type=pathlib.Path, default=pathlib.Path("./work"))
    args = ap.parse_args()
    w = args.work_dir

    t0 = time.time()
    lora = load_safetensors_bf16_as_f32(w / "adapter_model.safetensors")
    cfg = json.loads((w / "uncensored_config.json").read_text())["text_config"]
    manifest = json.loads((w / "MANIFEST.json").read_text())
    adapter_cfg = json.loads((w / "adapter_config.json").read_text())

    local_layers = set(cfg["local_layer_ids"])
    global_layers = sorted(set(range(48)) - local_layers)
    scale = adapter_cfg["lora_alpha"] / adapter_cfg["r"]

    records = []
    svds = {}
    for l in range(48):
        for peft_name, model_name in PROJ_MAP.items():
            A = lora[f"base_model.model.model.layers.{l}.self_attn.{peft_name}.lora_A.weight"]
            B = lora[f"base_model.model.model.layers.{l}.self_attn.{peft_name}.lora_B.weight"]
            U, S, Vt = lora_svd(A, B, scale=scale)
            svds[(l, model_name)] = (U.astype(np.float32), S.astype(np.float64), Vt.astype(np.float32))
            frob = float(np.sqrt(np.sum(S**2)))
            p = (S**2) / np.sum(S**2)
            erank = float(np.exp(-np.sum(p * np.log(np.clip(p, 1e-30, 1.0)))))
            mk = f"model.llm.layers.{l}.attn.{model_name}.weight"
            records.append({
                "layer": l, "is_global": l in global_layers, "block_pos": l % 6,
                "proj": model_name, "shape": [B.shape[0], A.shape[1]],
                "norm_A": float(np.linalg.norm(A)), "norm_B": float(np.linalg.norm(B)),
                "frob": frob, "s1": float(S[0]), "s2": float(S[1]), "s1_s2": float(S[0] / S[1]),
                "erank": erank, "srank": float((frob**2) / (S[0]**2)),
                "top1_e": float(p[0]), "top4_e": float(np.sum(p[:4])), "top8_e": float(np.sum(p[:8])),
                "manifest_rel_delta": manifest["tensors"][mk]["rel_delta"],
                "S": S.tolist(),
            })

    print(f"Computed exact SVDs for 240 modules in {time.time() - t0:.2f}s (scale={scale})")
    print("\n=== SUMMARY BY PROJECTION ===")
    for proj in ["wo_ud", "wv_dv", "gate_attn", "wq_du", "wk_dv"]:
        rs = [r for r in records if r["proj"] == proj]
        tot = sum(r["frob"] ** 2 for r in rs)
        by_k = np.sum([np.cumsum(np.array(r["S"]) ** 2) for r in rs], axis=0) / tot * 100
        print(
            f"{proj:10s} | sum||dW||^2={tot:6.2f} | rel={np.mean([r['manifest_rel_delta'] for r in rs])*100:.3f}% | "
            f"erank={np.mean([r['erank'] for r in rs]):5.2f}±{np.std([r['erank'] for r in rs]):4.2f} | "
            f"r1={by_k[0]:4.1f}% r4={by_k[3]:4.1f}% r8={by_k[7]:4.1f}%"
        )

    print("\n=== BBP SPIKE COUNT IN LORA A ABOVE MARCHENKO-PASTUR UPPER EDGE ===")
    for peft_name, model_name in PROJ_MAP.items():
        n_above, s1_a, s2_a, er_a, er_b = [], [], [], [], []
        for l in range(48):
            A = lora[f"base_model.model.model.layers.{l}.self_attn.{peft_name}.lora_A.weight"].astype(np.float64)
            B = lora[f"base_model.model.model.layers.{l}.self_attn.{peft_name}.lora_B.weight"].astype(np.float64)
            r, d_in = A.shape
            mp_hi = (1.0 / np.sqrt(3.0)) * (1.0 + np.sqrt(r / d_in))
            SA = np.linalg.svd(A, compute_uv=False)
            SB = np.linalg.svd(B, compute_uv=False)
            n_above.append(int(np.sum(SA > mp_hi * 1.02)))
            s1_a.append(SA[0]); s2_a.append(SA[1])
            pA = SA**2 / np.sum(SA**2); er_a.append(np.exp(-np.sum(pA * np.log(pA + 1e-30))))
            pB = SB**2 / np.sum(SB**2); er_b.append(np.exp(-np.sum(pB * np.log(pB + 1e-30))))
        print(
            f"  {model_name:10s} (MP_hi={mp_hi:.4f}): spikes={np.mean(n_above):.2f} | "
            f"A: s1={np.mean(s1_a):.4f}, s2={np.mean(s2_a):.4f}, erank={np.mean(er_a):.2f}/32 | "
            f"B: erank={np.mean(er_b):.2f}/32"
        )

    (w / "lora_svd_records.json").write_text(json.dumps(records, indent=2))
    np.savez(
        w / "lora_svds.npz",
        **{f"L{l}_{p}_U": svds[(l, p)][0] for l in range(48) for p in PROJ_MAP.values()},
        **{f"L{l}_{p}_S": svds[(l, p)][1] for l in range(48) for p in PROJ_MAP.values()},
        **{f"L{l}_{p}_Vt": svds[(l, p)][2] for l in range(48) for p in PROJ_MAP.values()},
    )
    print(f"Saved {w / 'lora_svd_records.json'} and {w / 'lora_svds.npz'}")


if __name__ == "__main__":
    main()
