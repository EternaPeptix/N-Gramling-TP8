#!/usr/bin/env python3
"""Generate publication figures fig1_depth_and_rank.png, fig2_subspaces_and_blocks.png, and fig3_circuit_and_bf16.png."""
import argparse
import json
import pathlib
import struct
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

COLORS = {"wq_du": "#38bdf8", "gate_attn": "#f472b6", "wo_ud": "#fbbf24", "wv_dv": "#34d399", "wk_dv": "#a78bfa"}
LABELS = {"wq_du": "wq_du (Q)", "gate_attn": "gate_attn", "wo_ud": "wo_ud (O)", "wv_dv": "wv_dv (V)", "wk_dv": "wk_dv (K)"}
PROJS = ["wq_du", "gate_attn", "wo_ud", "wv_dv", "wk_dv"]
GLOBAL_LAYERS = [5, 11, 17, 23, 29, 35, 41, 47]


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
    ap.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("./figures"))
    args = ap.parse_args()
    w, out = args.work_dir, args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "figure.facecolor": "#0f1117", "axes.facecolor": "#161922", "axes.edgecolor": "#2e3446",
        "axes.labelcolor": "#e2e8f0", "text.color": "#e2e8f0", "xtick.color": "#94a3b8",
        "ytick.color": "#94a3b8", "grid.color": "#252b3b", "grid.linestyle": "--",
        "grid.alpha": 0.6, "font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "bold",
        "legend.facecolor": "#161922", "legend.edgecolor": "#2e3446",
    })

    records = json.loads((w / "lora_svd_records.json").read_text())
    svds = np.load(w / "lora_svds.npz")
    pooled = np.load(w / "pooled_dirs.npz")
    regime = np.load(w / "regime_dirs.npz")
    verified = json.loads((w / "verified_weights_summary.json").read_text())

    with open(w / "adapter_model.safetensors", "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hlen))
        base = 8 + hlen
        lora = {}
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            s, e = v["data_offsets"]
            f.seek(base + s)
            u16 = np.frombuffer(f.read(e - s), dtype=np.uint16)
            lora[k] = ((u16.astype(np.uint32) << 16).view(np.float32)).reshape(v["shape"]).copy()

    # FIGURE 1
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.6), dpi=180)
    fig.suptitle("Inkling-2-Small-uncensored vs Base: Magnitude, Rank Collapse, and LoRA BBP Spike Structure", fontsize=12.5, fontweight="bold", y=0.98)

    ax = axes[0, 0]
    for gl in GLOBAL_LAYERS:
        ax.axvline(gl, color="#475569", lw=0.8, ls=":", alpha=0.8)
    for p in PROJS:
        rs = sorted([r for r in records if r["proj"] == p], key=lambda x: x["layer"])
        ax.plot([r["layer"] for r in rs], [r["manifest_rel_delta"] * 100 for r in rs], color=COLORS[p], lw=1.7, label=LABELS[p], marker="o", ms=2.5)
    ax.annotate("L44 wq_du peak\n(1.18%)", xy=(44, 1.18), xytext=(33, 1.15), arrowprops=dict(arrowstyle="->", color="#38bdf8", lw=1.0), color="#38bdf8", fontsize=8.5)
    ax.annotate("L6 min\n(0.56–0.63%)", xy=(6, 0.562), xytext=(9, 0.53), arrowprops=dict(arrowstyle="->", color="#94a3b8", lw=1.0), color="#cbd5e1", fontsize=8.5)
    ax.set_title("A. Relative Frobenius Delta ||ΔW||_F / ||W_base||_F (%)")
    ax.set_xlabel("Layer Index (dotted lines = Global Attn / AttnRes block ends)")
    ax.set_ylabel("Relative Delta (%)")
    ax.set_xlim(-0.5, 47.5); ax.grid(True); ax.legend(loc="upper left", ncol=2, fontsize=8)

    ax = axes[0, 1]
    for gl in GLOBAL_LAYERS:
        ax.axvline(gl, color="#475569", lw=0.8, ls=":", alpha=0.8)
    for p in PROJS:
        rs = sorted([r for r in records if r["proj"] == p], key=lambda x: x["layer"])
        ax.plot([r["layer"] for r in rs], [r["erank"] for r in rs], color=COLORS[p], lw=1.7, label=LABELS[p], marker="o", ms=2.5)
    ax.set_title("B. Effective Rank exp(H(σ²)) of Rank-32 Delta")
    ax.set_xlabel("Layer Index"); ax.set_ylabel("Effective Rank (max = 32)")
    ax.set_xlim(-0.5, 47.5); ax.set_ylim(0, 25.5); ax.grid(True); ax.legend(loc="upper right", ncol=2, fontsize=8)

    ax = axes[1, 0]
    ks = np.arange(1, 33)
    for p in PROJS:
        tot, cum = 0.0, np.zeros(32)
        for l in range(48):
            s2 = svds[f"L{l}_{p}_S"].astype(np.float64) ** 2
            tot += np.sum(s2); cum += np.cumsum(s2)
        pct = cum / tot * 100
        ax.plot(ks, pct, color=COLORS[p], lw=2.0, label=f"{LABELS[p]} (r1={pct[0]:.1f}%, r4={pct[3]:.1f}%)")
    ax.axvline(1, color="#94a3b8", lw=0.8, ls="--", alpha=0.5); ax.axvline(4, color="#94a3b8", lw=0.8, ls="--", alpha=0.5)
    ax.set_title("C. Cumulative Frobenius Energy vs Truncation Rank k (All 48 Layers)")
    ax.set_xlabel("Rank k (out of 32)"); ax.set_ylabel("Captured Energy (%)")
    ax.set_xlim(1, 32); ax.set_ylim(25, 102); ax.grid(True); ax.legend(loc="lower right", fontsize=8)

    ax = axes[1, 1]
    pmap = {"wq_du": "q_proj", "wk_dv": "k_proj", "wv_dv": "v_proj", "wo_ud": "o_proj", "gate_attn": "gate_proj"}
    for p in ["wo_ud", "gate_attn", "wq_du", "wk_dv"]:
        sA = np.mean([np.linalg.svd(lora[f"base_model.model.model.layers.{l}.self_attn.{pmap[p]}.lora_A.weight"].astype(np.float64), compute_uv=False) for l in range(48)], axis=0)
        sB = np.mean([np.linalg.svd(lora[f"base_model.model.model.layers.{l}.self_attn.{pmap[p]}.lora_B.weight"].astype(np.float64), compute_uv=False) for l in range(48)], axis=0)
        ax.plot(ks, sA, color=COLORS[p], lw=1.6, ls="--", alpha=0.9, label=f"A: {LABELS[p]}")
        ax.plot(ks, sB, color=COLORS[p], lw=2.0, ls="-", label=f"B: {LABELS[p]}")
    mp_hi = (1.0 / np.sqrt(3.0)) * (1.0 + np.sqrt(32.0 / 4608.0))
    mp_lo = (1.0 / np.sqrt(3.0)) * (1.0 - np.sqrt(32.0 / 4608.0))
    ax.axhspan(mp_lo, mp_hi, color="#94a3b8", alpha=0.14, label=f"MP Kaiming bulk [{mp_lo:.2f}, {mp_hi:.2f}]")
    ax.annotate("Single BBP spike\nin LoRA A (σ₁≈0.71)", xy=(1, 0.713), xytext=(6, 0.72), arrowprops=dict(arrowstyle="->", color="#fbbf24", lw=1.1), color="#fbbf24", fontsize=8.5)
    ax.set_title("D. Mean Singular Spectra of LoRA A (dashed) vs B (solid)")
    ax.set_xlabel("Singular Value Index i (1..32)"); ax.set_ylabel("Singular Value σᵢ")
    ax.set_xlim(0.5, 32.5); ax.set_ylim(0, 0.80); ax.grid(True); ax.legend(loc="center right", ncol=2, fontsize=7.5)
    plt.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(out / "fig1_depth_and_rank.png", dpi=180)
    plt.close(fig)

    # FIGURE 2
    fig = plt.figure(figsize=(15.5, 5.3), dpi=180)
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.05, 1.05, 1.35], wspace=0.30, left=0.045, right=0.98, top=0.81, bottom=0.11)
    fig.suptitle("Cross-Layer Residual-Stream Subspace Geometry: 6-Layer AttnRes Blocks & Orthogonal Read/Write Regimes", fontsize=12.5, fontweight="bold", y=0.96)

    M_wo = np.stack([svds[f"L{l}_wo_ud_U"][:, 0].astype(np.float64) for l in range(48)], axis=1)
    ax = fig.add_subplot(gs[0])
    im = ax.imshow(np.abs(M_wo.T @ M_wo), cmap="magma", vmin=0, vmax=1.0, aspect="equal")
    for b in range(6, 48, 6):
        ax.axhline(b - 0.5, color="#38bdf8", lw=0.6, alpha=0.55); ax.axvline(b - 0.5, color="#38bdf8", lw=0.6, alpha=0.55)
    ax.set_title("A. wo_ud Top-1 Write |u₁(L)·u₁(L')|\n(8 AttnRes 6-Layer Blocks + 2 Depth Regimes)")
    ax.set_xlabel("Layer L'"); ax.set_ylabel("Layer L"); ax.set_xticks(range(0, 48, 6)); ax.set_yticks(range(0, 48, 6))
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    M_g = np.stack([svds[f"L{l}_gate_attn_Vt"][0, :].astype(np.float64) for l in range(48)], axis=1)
    ax = fig.add_subplot(gs[1])
    im2 = ax.imshow(np.abs(M_g.T @ M_g), cmap="viridis", vmin=0, vmax=0.75, aspect="equal")
    for b in range(6, 48, 6):
        ax.axhline(b - 0.5, color="#f472b6", lw=0.5, alpha=0.45); ax.axvline(b - 0.5, color="#f472b6", lw=0.5, alpha=0.45)
    ax.set_title("B. gate_attn Top-1 Read |v₁(L)·v₁(L')|\n(Late-Layer G1 Safety/Prohibition Cluster)")
    ax.set_xlabel("Layer L'"); ax.set_ylabel("Layer L"); ax.set_xticks(range(0, 48, 6)); ax.set_yticks(range(0, 48, 6))
    plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.03)

    ax = fig.add_subplot(gs[2])
    g1 = pooled["grand_U"][:, 0].astype(np.float64)
    wo1 = pooled["wo_ud_U"][:, 0].astype(np.float64)
    wo_late = regime["wo_late"].astype(np.float64)
    ls = np.arange(48)
    for b in range(6, 48, 6):
        ax.axvline(b - 0.5, color="#64748b", lw=0.8, ls=":", alpha=0.7)
    ax.plot(ls, [abs(float(svds[f"L{l}_wo_ud_U"][:, 0] @ wo1)) for l in range(48)], color="#fbbf24", lw=2.0, marker="o", ms=3, label="|u₁(wo_ud) · W₁| (6-layer AttnRes sawtooth)")
    ax.plot(ls, [abs(float(svds[f"L{l}_wo_ud_U"][:, 0] @ wo_late)) for l in range(48)], color="#fb923c", lw=1.6, ls="--", label="|u₁(wo_ud) · wo_late| (L36–47 regime)")
    ax.plot(ls, [abs(float(svds[f"L{l}_gate_attn_Vt"][0] @ g1)) for l in range(48)], color="#f472b6", lw=1.8, label="|v₁(gate_attn) · G₁| (read)")
    ax.plot(ls, [abs(float(svds[f"L{l}_wq_du_Vt"][0] @ g1)) for l in range(48)], color="#38bdf8", lw=1.5, alpha=0.85, label="|v₁(wq_du) · G₁| (read)")
    ax.set_title("C. Global Axis Alignment vs Layer Depth\n(Dotted Lines = AttnRes Block Resets at L%6==0)")
    ax.set_xlabel("Layer Index"); ax.set_ylabel("|Cosine Similarity| in R⁴⁶⁰⁸")
    ax.set_xlim(-0.5, 47.5); ax.set_ylim(0, 0.95); ax.grid(True); ax.legend(loc="upper left", fontsize=7.8)
    fig.savefig(out / "fig2_subspaces_and_blocks.png", dpi=180)
    plt.close(fig)

    # FIGURE 3
    fig = plt.figure(figsize=(16.0, 5.4), dpi=180)
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.15, 1.30, 1.08], wspace=0.40, left=0.045, right=0.945, top=0.81, bottom=0.11)
    fig.suptitle("Causal Circuit Selectivity, Signed Logit-Lens Shifts (L44+L47 ΔOV), and BF16 Merge Quantization", fontsize=12.5, fontweight="bold", y=0.96)

    ax = fig.add_subplot(gs[0])
    edir = w / "extra_tensors"
    V_VALID = 200058
    unembed = ((np.frombuffer((edir / "model.llm.unembed.weight.bin").read_bytes(), dtype=np.uint16).reshape(201024, 4608)[:V_VALID].astype(np.uint32) << 16).view(np.float32))
    u2b = {v: k for k, v in bytes_to_unicode().items()}
    tok = json.loads((w / "tokenizer.json").read_text())
    id2raw = {v: k for k, v in tok["model"]["vocab"].items()}
    added = {a["id"]: a["content"] for a in tok.get("added_tokens", [])}
    id2raw.update(added)

    def decode_tok(tid):
        if tid in added:
            return added[tid]
        try:
            return bytes([u2b[ch] for ch in id2raw[tid]]).decode("utf-8", errors="replace")
        except Exception:
            return id2raw.get(tid, "")

    def word_vec(words):
        ids = [i for i in range(V_VALID) if decode_tok(i).strip().lower() in words]
        v = unembed[ids].mean(axis=0).astype(np.float64)
        return v / np.linalg.norm(v)

    v_harm = word_vec({"illegal", "harmful", "dangerous", "prohibited", "prohibit", "criminal", "forbidden", "weapon", "exploit"})
    v_harmless = word_vec({"quantum", "recipe", "python", "weather", "guitar", "planet", "mountain", "coffee", "algebra"})
    v_tool = word_vec({"function", "parameter", "json", "schema", "query", "endpoint", "argument", "callback"})

    def layer_gains(proj, vec):
        return [np.linalg.norm(svds[f"L{l}_{proj}_S"].astype(np.float64) * (svds[f"L{l}_{proj}_Vt"].astype(np.float64) @ vec)) / np.sqrt(np.sum(svds[f"L{l}_{proj}_S"].astype(np.float64) ** 2) / 4608.0) for l in range(48)]

    ax.plot(ls, layer_gains("wq_du", g1), color="#38bdf8", lw=1.9, label="wq_du on +G1 (max 44.8×)")
    ax.plot(ls, layer_gains("gate_attn", g1), color="#f472b6", lw=1.9, label="gate_attn on +G1 (max 37.7×)")
    ax.plot(ls, layer_gains("wv_dv", g1), color="#34d399", lw=1.6, label="wv_dv on +G1 (max 29.8×)")
    ax.plot(ls, layer_gains("gate_attn", v_harm), color="#fbbf24", lw=1.5, ls="--", label="gate_attn on 'harmful' words")
    ax.plot(ls, layer_gains("gate_attn", v_harmless), color="#94a3b8", lw=1.3, ls=":", label="gate_attn on harmless (~0.9×)")
    ax.plot(ls, layer_gains("gate_attn", v_tool), color="#cbd5e1", lw=1.3, ls="-.", label="gate_attn on tool words (~0.9×)")
    ax.axhline(1.0, color="#64748b", lw=0.8, alpha=0.6)
    ax.set_title("A. Input Selectivity ||ΔW·x|| / RMS(||ΔW·rand||)")
    ax.set_xlabel("Layer Index"); ax.set_ylabel("Selectivity Multiplier (× random)")
    ax.set_xlim(-0.5, 47.5); ax.grid(True); ax.legend(loc="upper left", fontsize=7.5)

    ax = fig.add_subplot(gs[1])
    items = [
        ("Provide", +10.7), ("clear", +7.3), ("concrete", +7.2), ("thorough", +6.7),
        ("<|end_message|>", +4.2), ("direct", +2.9), ("Here", +2.7), ("Sorry", -0.4),
        ("cannot / unable", -1.6), ("Warning", -1.8), ("dangerous", -2.0), ("ethical", -2.1),
        ("However", -2.3), ("safety", -2.5), ("policy", -2.6), ("harmful", -2.9),
        ("explanation", -3.2), ("practical", -3.5), ("answer", -4.0),
    ]
    labels_b = [x[0] for x in items][::-1]
    vals_b = [x[1] for x in items][::-1]
    y_pos = np.arange(len(labels_b))
    ax.barh(y_pos, vals_b, color=["#34d399" if v > 0 else "#f87171" for v in vals_b], height=0.72)
    ax.axvline(0, color="#e2e8f0", lw=0.9)
    for y, v in zip(y_pos, vals_b):
        ax.text(v + (0.25 if v >= 0 else -0.25), y, f"{v:+.1f}σ", va="center", ha="left" if v >= 0 else "right", fontsize=7.5, color="#e2e8f0")
    ax.set_yticks(y_pos); ax.set_yticklabels(labels_b, fontsize=8); ax.set_xlim(-5.5, 13.0)
    ax.set_title("B. Signed Logit Shift of L44+L47 ΔOV(+G₁)\n(Z-score across 200k Vocab: Promoted vs Suppressed)")
    ax.set_xlabel("Logit Shift Z-Score (σ)"); ax.grid(True, axis="x")

    ax = fig.add_subplot(gs[2])
    layers_v = [0, 5, 6, 23, 44, 47]
    p0s = [np.mean([r["p0_ulp"] for r in verified if r["layer"] == lv]) for lv in layers_v]
    p1s = [np.mean([r["p1_ulp"] for r in verified if r["layer"] == lv]) for lv in layers_v]
    p2s = [np.mean([r["p2p_ulp"] for r in verified if r["layer"] == lv]) for lv in layers_v]
    qns = [np.mean([r["qnoise_frac"] * 100 for r in verified if r["layer"] == lv]) for lv in layers_v]
    x = np.arange(len(layers_v))
    ax.bar(x, p0s, width=0.58, color="#64748b", label="0 ULP (unchanged)")
    ax.bar(x, p1s, width=0.58, bottom=p0s, color="#38bdf8", label="±1 ULP shift")
    ax.bar(x, p2s, width=0.58, bottom=np.array(p0s) + np.array(p1s), color="#fbbf24", label="≥2 ULP shift")
    ax2 = ax.twinx()
    ax2.plot(x, qns, color="#f472b6", lw=2.2, marker="D", ms=5, label="BF16 quant noise / ||ΔW||")
    ax2.set_ylabel("Quantization Residual ||E|| / ||ΔW|| (%)", color="#f472b6")
    ax2.tick_params(axis="y", colors="#f472b6"); ax2.set_ylim(0, 40)
    ax.set_xticks(x); ax.set_xticklabels([f"L{l}" for l in layers_v])
    ax.set_ylabel("Share of BF16 Weights (%)"); ax.set_ylim(0, 128)
    ax.set_title("C. BF16 Merge Granularity (566M Verified Weights)\n(94.9% Energy Kept in Rank-32 Subspace)")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper center", ncol=2, fontsize=6.9, frameon=True)
    fig.savefig(out / "fig3_circuit_and_bf16.png", dpi=180)
    plt.close(fig)
    print(f"Generated 3 figures in {out}")


if __name__ == "__main__":
    main()
