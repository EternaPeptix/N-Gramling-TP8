#!/usr/bin/env python3
"""Byte-range audit of 120 untouched tensors + 40 slices, and download of 6 verified layers + unembed/norm/tokenizer."""
import argparse
import concurrent.futures
import hashlib
import json
import pathlib
import time
import urllib.request

U_BASE = "https://huggingface.co/benthecarman/Inkling-2-Small-uncensored/resolve/main/"
B_BASE = "https://huggingface.co/thinkingmachines/Inkling-2-Small/resolve/main/"
TARGET_LAYERS = [0, 5, 6, 23, 44, 47]
PROJS = ["wq_du", "wk_dv", "wv_dv", "wo_ud", "gate_attn"]


def fetch_range(url: str, start: int, end: int, retries: int = 4) -> bytes:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end - 1}"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
            if len(data) != (end - start):
                raise IOError(f"Short read: got {len(data)}, expected {end - start}")
            return data
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))


def download_chunked(url: str, start: int, end: int, out_path: pathlib.Path, chunk_mb: int = 32, workers: int = 8) -> None:
    total = end - start
    if out_path.exists() and out_path.stat().st_size == total:
        return
    cs = chunk_mb * 1024 * 1024
    chunks = [(off, min(total, off + cs)) for off in range(0, total, cs)]
    buf = bytearray(total)

    def grab(c):
        off, stop = c
        buf[off:stop] = fetch_range(url, start + off, start + stop)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(grab, chunks))
    out_path.write_bytes(buf)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", type=pathlib.Path, default=pathlib.Path("./work"))
    args = ap.parse_args()
    w = args.work_dir

    u_hdrs = json.loads((w / "shard_headers_u.json").read_text())
    manifest = json.loads((w / "MANIFEST.json").read_text())

    tmap = {}
    for sh, obj in u_hdrs.items():
        base = 8 + obj["hlen"]
        for k, v in obj["header"].items():
            if k == "__metadata__":
                continue
            s, e = v["data_offsets"]
            tmap[k] = (sh, base + s, base + e, v["dtype"], v["shape"])

    # 1. Audit 120 full untouched tensors + 40 slices
    audit_full = [f"model.llm.layers.{l}.mlp.gate.bias" for l in range(48)]
    for l in [0, 5, 11, 23, 35, 41, 44, 47]:
        for suf in [
            "mlp.gate.weight", "mlp_norm.weight", "attn_sconv.weight", "mlp_sconv.weight",
            "attn.attn_sink_proj.weight", "attn.q_l2_clip.scale", "attn.wr_du.weight",
            "mlp.experts.w2_weight_amax", "mlp.experts.w13_weight_amax",
        ]:
            k = f"model.llm.layers.{l}.{suf}"
            if k in tmap:
                audit_full.append(k)
    audit_full += [
        k for k in [
            "model.llm.norm.weight", "model.llm.embed_norm.weight",
            "model.llm.attn_res.output_query", "model.llm.attn_res.attn_queries.46",
            "model.llm.attn_res.mlp_queries.47", "model.visual.proj_out.weight",
        ] if k in tmap
    ]

    audit_slices = []
    for l in [0, 5, 17, 29, 44, 47]:
        for suf in ["mlp.experts.w2_weight", "mlp.experts.w13_weight", "mlp.experts.w2_weight_scale"]:
            k = f"model.llm.layers.{l}.{suf}"
            if k in tmap:
                sh, s, e, _, _ = tmap[k]
                mid = s + (e - s) // 2
                audit_slices.append((f"{k}[head1M]", sh, s, min(e, s + 1048576)))
                audit_slices.append((f"{k}[mid1M]", sh, mid, min(e, mid + 1048576)))
    for k in ["model.llm.embed.weight", "model.llm.unembed.weight"]:
        sh, s, e, _, _ = tmap[k]
        audit_slices.append((f"{k}[head2M]", sh, s, s + 2 * 1048576))
        audit_slices.append((f"{k}[tail2M]", sh, e - 2 * 1048576, e))

    t0 = time.time()
    print(f"Auditing {len(audit_full)} full untouched tensors + {len(audit_slices)} slices...")
    mismatches = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as ex:
        for k, eq in ex.map(lambda k: (k, fetch_range(U_BASE + tmap[k][0], tmap[k][1], tmap[k][2]) == fetch_range(B_BASE + tmap[k][0], tmap[k][1], tmap[k][2])), audit_full):
            if not eq:
                mismatches.append(k)
        for lbl, eq in ex.map(lambda it: (it[0], fetch_range(U_BASE + it[1], it[2], it[3]) == fetch_range(B_BASE + it[1], it[2], it[3])), audit_slices):
            if not eq:
                mismatches.append(lbl)
    print(f"Untouched audit done in {time.time() - t0:.1f}s. Mismatches: {len(mismatches)}")

    # 2. Fetch 30 base+uncensored attention weight pairs across TARGET_LAYERS
    wdir = w / "verified_weights"
    wdir.mkdir(exist_ok=True)
    tasks = [(l, p, f"model.llm.layers.{l}.attn.{p}.weight") for l in TARGET_LAYERS for p in PROJS]

    def fetch_pair(item):
        l, p, k = item
        sh, s, e, _, _ = tmap[k]
        pu, pb = wdir / f"L{l}_{p}_u.bin", wdir / f"L{l}_{p}_b.bin"
        du = pu.read_bytes() if pu.exists() and pu.stat().st_size == (e - s) else fetch_range(U_BASE + sh, s, e)
        db = pb.read_bytes() if pb.exists() and pb.stat().st_size == (e - s) else fetch_range(B_BASE + sh, s, e)
        if not pu.exists(): pu.write_bytes(du)
        if not pb.exists(): pb.write_bytes(db)
        return (
            hashlib.sha256(du).hexdigest() == manifest["tensors"][k]["sha256_merged"],
            hashlib.sha256(db).hexdigest() == manifest["tensors"][k]["sha256_orig"],
        )

    t1 = time.time()
    print(f"Downloading 30 weight pairs for layers {TARGET_LAYERS}...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        checks = list(ex.map(fetch_pair, tasks))
    print(f"Done in {time.time() - t1:.1f}s. MANIFEST SHA-256 matches: {sum(u and b for u, b in checks)}/{len(tasks)}")

    # 3. Fetch tokenizer, norms, attn_res queries, and unembed
    edir = w / "extra_tensors"
    edir.mkdir(exist_ok=True)
    if not (w / "tokenizer.json").exists():
        urllib.request.urlretrieve(U_BASE + "tokenizer.json", w / "tokenizer.json")
    small_keys = [k for k in tmap if k.startswith("model.llm.attn_res.") or k in ("model.llm.norm.weight", "model.llm.embed_norm.weight")]
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as ex:
        list(ex.map(lambda k: (edir / f"{k}.bin").exists() or (edir / f"{k}.bin").write_bytes(fetch_range(U_BASE + tmap[k][0], tmap[k][1], tmap[k][2])), small_keys))
    sh, s, e, _, _ = tmap["model.llm.unembed.weight"]
    print("Downloading model.llm.unembed.weight (1.853 GB)...")
    download_chunked(U_BASE + sh, s, e, edir / "model.llm.unembed.weight.bin")
    print("All remote artifacts ready.")


if __name__ == "__main__":
    main()
