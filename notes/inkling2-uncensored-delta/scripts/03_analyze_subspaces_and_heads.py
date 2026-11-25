#!/usr/bin/env python3
"""Cross-layer residual-stream subspace alignment, 6-layer AttnRes block geometry, and per-head/GQA structure."""
import argparse
import pathlib
import numpy as np

PROJS = ["wq_du", "wk_dv", "wv_dv", "wo_ud", "gate_attn"]


def gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = len(x)
    return float((2 * np.sum(np.arange(1, n + 1) * x) / (n * np.sum(x))) - (n + 1) / n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", type=pathlib.Path, default=pathlib.Path("./work"))
    args = ap.parse_args()
    w = args.work_dir

    data = np.load(w / "lora_svds.npz")

    def get_svd(l, p):
        return (
            data[f"L{l}_{p}_U"].astype(np.float64),
            data[f"L{l}_{p}_S"].astype(np.float64),
            data[f"L{l}_{p}_Vt"].astype(np.float64),
        )

    # 1. Pooled residual-stream subspaces (4608-dim)
    pooled, top_dirs = {}, {}
    all_cols = []
    for p in PROJS:
        cols = []
        for l in range(48):
            U, S, Vt = get_svd(l, p)
            vec = U if p == "wo_ud" else Vt.T
            cols.append(vec * S[None, :])
        M = np.concatenate(cols, axis=1)
        all_cols.extend(cols)
        Up, Sp, _ = np.linalg.svd(M, full_matrices=False)
        e = (Sp**2) / np.sum(Sp**2)
        pooled[p] = (Up[:, :16], Sp[:32], e)
        top_dirs[p] = Up[:, :8]
        print(f"Pooled {p:10s}: top1={e[0]*100:5.2f}% top2={np.sum(e[:2])*100:5.2f}% top4={np.sum(e[:4])*100:5.2f}%")

    Uall, Sall, _ = np.linalg.svd(np.concatenate(all_cols, axis=1), full_matrices=False)
    g1 = Uall[:, 0]
    print("\n|cos(GRAND #1, pooled #1)|:", {p: round(abs(float(g1 @ top_dirs[p][:, 0])), 4) for p in PROJS})

    print("\n=== TOP-4 PRINCIPAL SUBSPACE OVERLAP (mean cos^2) ===")
    print("           " + " ".join(f"{p:>10s}" for p in PROJS))
    for p1 in PROJS:
        vals = [np.mean(np.linalg.svd(top_dirs[p1][:, :4].T @ top_dirs[p2][:, :4], compute_uv=False) ** 2) for p2 in PROJS]
        print(f"{p1:10s} " + " ".join(f"{v:10.4f}" for v in vals))

    # 2. 6-layer AttnRes block structure & early vs late wo_ud regimes
    M_wo = np.stack([get_svd(l, "wo_ud")[0][:, 0] for l in range(48)], axis=1)
    G_wo = np.abs(M_wo.T @ M_wo)
    win = [G_wo[l, l + 1] for l in range(47) if l % 6 != 5]
    xbd = [G_wo[l, l + 1] for l in range(47) if l % 6 == 5]
    print(f"\nwo_ud adjacent |cos|: within 6L block={np.mean(win):.4f} (max={np.max(win):.4f}) vs across boundary={np.mean(xbd):.4f}")

    def pool_range(proj, layers):
        cols = []
        for l in layers:
            U, S, Vt = get_svd(l, proj)
            cols.append((U if proj == "wo_ud" else Vt.T) * S[None, :])
        Up, Sp, _ = np.linalg.svd(np.concatenate(cols, axis=1), full_matrices=False)
        e = Sp**2 / np.sum(Sp**2)
        return Up[:, 0], float(e[0])

    wo_early, e_early = pool_range("wo_ud", range(0, 30))
    wo_bridge, e_bridge = pool_range("wo_ud", range(30, 36))
    wo_late, e_late = pool_range("wo_ud", range(36, 48))
    print(f"wo_ud regimes: early(L00-29) top1={e_early*100:.1f}%, bridge(L30-35) top1={e_bridge*100:.1f}%, late(L36-47) top1={e_late*100:.1f}%")
    print(f"|cos(wo_early, wo_late)| = {abs(float(wo_early @ wo_late)):.4f} ({np.degrees(np.arccos(abs(float(wo_early @ wo_late)))):.1f} deg)")

    # 3. Per-head & GQA group statistics
    head_norms = {}
    print("\n=== PER-HEAD CONCENTRATION ===")
    for p, nh in [("wq_du", 48), ("gate_attn", 48), ("wo_ud", 48), ("wk_dv", 8), ("wv_dv", 8)]:
        mat = np.zeros((48, nh))
        for l in range(48):
            U, S, Vt = get_svd(l, p)
            if p == "wo_ud":
                M = (S[:, None] * Vt).reshape(32, nh, 128)
                mat[l] = np.sqrt(np.sum(M**2, axis=(0, 2)))
            else:
                M = (U * S[None, :]).reshape(nh, 128, 32)
                mat[l] = np.sqrt(np.sum(M**2, axis=(1, 2)))
        head_norms[p] = mat
        print(f"  {p:10s} ({nh:2d}h): Gini={np.mean([gini(mat[l]) for l in range(48)]):.3f} | max/med={np.mean([np.max(mat[l])/np.median(mat[l]) for l in range(48)]):.2f}x")

    wq_grp = np.sqrt(np.sum(head_norms["wq_du"].reshape(48, 8, 6) ** 2, axis=2))
    ga_grp = np.sqrt(np.sum(head_norms["gate_attn"].reshape(48, 8, 6) ** 2, axis=2))
    r_qk = [np.corrcoef(wq_grp[l], head_norms["wk_dv"][l])[0, 1] for l in range(48)]
    r_gv = [np.corrcoef(ga_grp[l], head_norms["wv_dv"][l])[0, 1] for l in range(48)]
    print(f"GQA-group corr: wq vs wk median r={np.median(r_qk):.4f} (max={np.max(r_qk):.4f}); gate vs wv median r={np.median(r_gv):.4f}")

    # Save pooled and regime directions
    np.savez(
        w / "pooled_dirs.npz",
        grand_U=Uall[:, :16].astype(np.float32),
        grand_S=Sall[:32].astype(np.float64),
        **{f"{p}_U": pooled[p][0].astype(np.float32) for p in PROJS},
        **{f"{p}_S": pooled[p][1].astype(np.float64) for p in PROJS},
        **{f"{p}_head_norms": head_norms[p].astype(np.float32) for p in PROJS},
    )
    in_late_cols = []
    for p in ["wq_du", "gate_attn", "wv_dv"]:
        for l in range(30, 48):
            _, S, Vt = get_svd(l, p)
            in_late_cols.append(Vt.T * S[None, :])
    Uin_late, _, _ = np.linalg.svd(np.concatenate(in_late_cols, axis=1), full_matrices=False)
    np.savez(
        w / "regime_dirs.npz",
        wo_early=wo_early.astype(np.float32),
        wo_bridge=wo_bridge.astype(np.float32),
        wo_late=wo_late.astype(np.float32),
        in_late=Uin_late[:, 0].astype(np.float32),
    )
    print(f"Saved {w / 'pooled_dirs.npz'} and {w / 'regime_dirs.npz'}")


if __name__ == "__main__":
    main()
