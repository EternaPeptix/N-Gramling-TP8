#!/usr/bin/env python3
"""UTF-8 BPE logit-lens decoding, input selectivity, and signed L44+L47 OV/Gate causal circuit analysis."""
import argparse
import json
import pathlib
import numpy as np

V_VALID = 200058


def u16_to_f32(u16: np.ndarray) -> np.ndarray:
    return (u16.astype(np.uint32) << 16).view(np.float32)


def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", type=pathlib.Path, default=pathlib.Path("./work"))
    args = ap.parse_args()
    w = args.work_dir

    edir = w / "extra_tensors"
    wdir = w / "verified_weights"
    manifest = json.loads((w / "MANIFEST.json").read_text())

    u2b = {v: k for k, v in bytes_to_unicode().items()}
    tok = json.loads((w / "tokenizer.json").read_text())
    id2raw = {v: k for k, v in tok["model"]["vocab"].items()}
    added = {a["id"]: a["content"] for a in tok.get("added_tokens", [])}
    id2raw.update(added)

    def decode_tok(tid: int) -> str:
        tid = int(tid)
        if tid in added:
            return added[tid]
        try:
            return bytes([u2b[ch] for ch in id2raw[tid]]).decode("utf-8", errors="replace")
        except Exception:
            return id2raw.get(tid, "")

    norm_w = u16_to_f32(np.frombuffer((edir / "model.llm.norm.weight.bin").read_bytes(), dtype=np.uint16))
    unembed = u16_to_f32(np.frombuffer((edir / "model.llm.unembed.weight.bin").read_bytes(), dtype=np.uint16).reshape(201024, 4608))[:V_VALID]
    U_normed = unembed * norm_w[None, :]

    pooled = np.load(w / "pooled_dirs.npz")
    regime = np.load(w / "regime_dirs.npz")
    svds = np.load(w / "lora_svds.npz")

    g1 = pooled["grand_U"][:, 0].astype(np.float64)
    refus_ids = [i for i in range(V_VALID) if decode_tok(i) in (" refus", "refus", " prohibit", " criminal")]
    if np.mean(U_normed[refus_ids] @ g1) < 0:
        g1 = -g1
    wo_late = regime["wo_late"].astype(np.float64)
    conc_ids = [i for i in range(V_VALID) if decode_tok(i) in (" concrete", " thorough", " Provide")]
    if np.mean(U_normed[conc_ids] @ wo_late) < 0:
        wo_late = -wo_late

    # 1. Input selectivity across all 48 layers
    def word_vec(words):
        ids = [i for i in range(V_VALID) if decode_tok(i).strip().lower() in words]
        v = unembed[ids].mean(axis=0).astype(np.float64)
        return v / np.linalg.norm(v)

    v_harm = word_vec({"illegal", "harmful", "dangerous", "prohibited", "prohibit", "criminal", "forbidden", "weapon", "exploit"})
    v_harmless = word_vec({"quantum", "recipe", "python", "weather", "guitar", "planet", "mountain", "coffee", "algebra"})
    v_tool = word_vec({"function", "parameter", "json", "schema", "query", "endpoint", "argument", "callback"})

    print("=== 1. INPUT SELECTIVITY ||dW @ x|| / RMS(||dW @ rand||) ===")
    for proj in ["wq_du", "gate_attn", "wv_dv", "wk_dv"]:
        g_all, g_late, h_all, hl_all, t_all = [], [], [], [], []
        for l in range(48):
            S = svds[f"L{l}_{proj}_S"].astype(np.float64)
            Vt = svds[f"L{l}_{proj}_Vt"].astype(np.float64)
            rms = np.sqrt(np.sum(S**2) / 4608.0)
            g = np.linalg.norm(S * (Vt @ g1)) / rms
            g_all.append(g)
            if l >= 32:
                g_late.append(g)
            h_all.append(np.linalg.norm(S * (Vt @ v_harm)) / rms)
            hl_all.append(np.linalg.norm(S * (Vt @ v_harmless)) / rms)
            t_all.append(np.linalg.norm(S * (Vt @ v_tool)) / rms)
        print(f"  {proj:10s}: +G1 all={np.mean(g_all):5.2f}x (L32-47={np.mean(g_late):5.2f}x, max={np.max(g_all):5.2f}x@L{int(np.argmax(g_all)):02d}) | harmful={np.mean(h_all):.2f}x | harmless={np.mean(hl_all):.2f}x | tool={np.mean(t_all):.2f}x")

    # 2. Signed OV circuit on +G1 at L44 & L47
    def load_w(l, mp, tag):
        shp = tuple(manifest["tensors"][f"model.llm.layers.{l}.attn.{mp}.weight"]["shape"])
        return u16_to_f32(np.frombuffer((wdir / f"L{l}_{mp}_{tag}.bin").read_bytes(), dtype=np.uint16).reshape(shp)).astype(np.float64)

    def ov(Wo, Wv, x):
        return Wo @ np.repeat((Wv @ x).reshape(8, 128), 6, axis=0).reshape(6144)

    print("\n=== 2. SIGNED OV CIRCUIT DELTA AT L44 & L47 ON +G1 ===")
    dys = {}
    for l in [44, 47]:
        Wo_b, Wo_u = load_w(l, "wo_ud", "b"), load_w(l, "wo_ud", "u")
        Wv_b, Wv_u = load_w(l, "wv_dv", "b"), load_w(l, "wv_dv", "u")
        Wg_b, Wg_u = load_w(l, "gate_attn", "b"), load_w(l, "gate_attn", "u")
        dy = ov(Wo_u, Wv_u, g1) - ov(Wo_b, Wv_b, g1)
        t_wo = ov(Wo_u - Wo_b, Wv_b, g1)
        t_wv = ov(Wo_b, Wv_u - Wv_b, g1)
        vb = (Wv_b @ g1).reshape(8, 128)
        vb = np.repeat(vb / (np.sqrt(np.mean(vb**2, axis=1, keepdims=True)) + 1e-6), 6, axis=0).reshape(6144)
        sig = lambda z: 1.0 / (1.0 + np.exp(-np.clip(z, -20, 20)))
        t_gate = Wo_b @ ((sig(Wg_u @ g1) - sig(Wg_b @ g1)) * vb)
        c = lambda v: float((v / np.linalg.norm(v)) @ wo_late)
        print(f"  L{l}: signed cos with wo_late -> total={c(dy):+.3f}, dWo={c(t_wo):+.3f}, dWv={c(t_wv):+.3f}, dGate={c(t_gate):+.3f}")
        dys[l] = dy

    # Top tokens and z-scores on L44+L47 combined OV shift
    dy_sum = dys[44] + dys[47]
    sc = U_normed @ (dy_sum / np.linalg.norm(dy_sum)).astype(np.float32)
    mu, sd = float(sc.mean()), float(sc.std())
    tok2id = {}
    for i in range(V_VALID):
        s = decode_tok(i)
        if s not in tok2id:
            tok2id[s] = i
    print("\n=== 3. L44+L47 SIGNED dOV(+G1) Z-SCORES ===")
    for label, toks in [
        ("CoT compliance", ["Provide", " clear", " concrete", " thorough", " direct"]),
        ("Special EOS", ["<|end_message|>", "<|content_model_end_sampling|>"]),
        ("Soft deflection", [" harmful", " policy", " safety", " However", " ethical", " dangerous", " Warning"]),
        ("Hard refusal", [" Sorry", " cannot", " unable", " apologize"]),
        ("Meta answer", [" answer", " practical", " answers", " explanation", " hidden"]),
    ]:
        zvals = [(t.strip(), float((sc[tok2id[t]] - mu) / sd)) for t in toks if t in tok2id]
        print(f"  {label:16s}: " + ", ".join(f"{t}({z:+.1f}σ)" for t, z in zvals))


if __name__ == "__main__":
    main()
