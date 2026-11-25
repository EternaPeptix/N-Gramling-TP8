#!/usr/bin/env python3
"""Fetch Hugging Face metadata, config/tokenizer diffs, all 24 shard headers, and the LoRA adapter."""
import argparse
import concurrent.futures
import difflib
import json
import pathlib
import struct
import urllib.request

U_REPO = "benthecarman/Inkling-2-Small-uncensored"
B_REPO = "thinkingmachines/Inkling-2-Small"
U_RAW = f"https://huggingface.co/{U_REPO}/raw/main/"
B_RAW = f"https://huggingface.co/{B_REPO}/raw/main/"
U_RES = f"https://huggingface.co/{U_REPO}/resolve/main/"
B_RES = f"https://huggingface.co/{B_REPO}/resolve/main/"


def fetch(url: str, dst: pathlib.Path) -> None:
    if dst.exists() and dst.stat().st_size > 0:
        return
    urllib.request.urlretrieve(url, dst)


def fetch_st_header(url: str):
    req = urllib.request.Request(url, headers={"Range": "bytes=0-65535"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        buf = resp.read()
    hlen = struct.unpack("<Q", buf[:8])[0]
    if 8 + hlen > len(buf):
        req2 = urllib.request.Request(url, headers={"Range": f"bytes=0-{8 + hlen - 1}"})
        with urllib.request.urlopen(req2, timeout=30) as resp:
            buf = resp.read()
    return hlen, json.loads(buf[8 : 8 + hlen])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", type=pathlib.Path, default=pathlib.Path("./work"))
    args = ap.parse_args()
    w = args.work_dir
    w.mkdir(parents=True, exist_ok=True)

    downloads = [
        (U_RAW + "README.md", w / "uncensored_README.md"),
        (U_RAW + "config.json", w / "uncensored_config.json"),
        (B_RAW + "config.json", w / "base_config.json"),
        (U_RAW + "MANIFEST.json", w / "MANIFEST.json"),
        (U_RAW + "trial.json", w / "trial.json"),
        (U_RAW + "lora/adapter_config.json", w / "adapter_config.json"),
        (U_RES + "lora/adapter_model.safetensors", w / "adapter_model.safetensors"),
        (f"https://huggingface.co/api/models/{U_REPO}/tree/main", w / "uncensored_tree.json"),
        (f"https://huggingface.co/api/models/{B_REPO}/tree/main", w / "base_tree.json"),
        (f"https://huggingface.co/api/models/{B_REPO}/commits/main", w / "base_commits.json"),
    ]
    for fn in ["chat_template.jinja", "special_tokens_map.json", "tokenizer_config.json"]:
        downloads.append((U_RAW + fn, w / f"u_{fn}"))
        downloads.append((B_RAW + fn, w / f"b_{fn}"))

    print(f"Downloading metadata and LoRA adapter into {w}...")
    for url, dst in downloads:
        fetch(url, dst)

    # Check base snapshot referenced by MANIFEST.json
    manifest = json.loads((w / "MANIFEST.json").read_text())
    src_snap = manifest["src_release"].rstrip("/").split("/")[-1]
    print(f"MANIFEST src_release snapshot: {src_snap}")
    for fn, uf in [
        ("config.json", "uncensored_config.json"),
        ("chat_template.jinja", "u_chat_template.jinja"),
        ("tokenizer_config.json", "u_tokenizer_config.json"),
        ("special_tokens_map.json", "u_special_tokens_map.json"),
    ]:
        snap_path = w / f"base_snap_{fn}"
        fetch(f"https://huggingface.co/{B_REPO}/raw/{src_snap}/{fn}", snap_path)
        eq_snap = (w / uf).read_bytes() == snap_path.read_bytes()
        bf = "base_config.json" if fn == "config.json" else f"b_{fn}"
        eq_main = (w / uf).read_bytes() == (w / bf).read_bytes()
        print(f"  {fn:26s}: matches base@{src_snap[:8]}={eq_snap}, matches base@main={eq_main}")

    # Unified diff of config.json vs base@main
    diff = "".join(
        difflib.unified_diff(
            (w / "base_config.json").read_text().splitlines(keepends=True),
            (w / "uncensored_config.json").read_text().splitlines(keepends=True),
            fromfile="base@main/config.json",
            tofile="uncensored/config.json",
            n=2,
        )
    )
    if diff:
        print("\n--- config.json diff vs base@main ---\n" + diff)

    # Fetch and compare all 24 safetensors shard headers
    shards = [f"model-{i:05d}-of-00024.safetensors" for i in range(1, 25)]
    print("Fetching all 24 shard headers from both repos...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as ex:
        res = list(ex.map(lambda sh: (sh, fetch_st_header(U_RES + sh), fetch_st_header(B_RES + sh)), shards))

    u_hdrs, b_hdrs = {}, {}
    diff_hdrs = 0
    for sh, (uhlen, uh), (bhlen, bh) in res:
        u_hdrs[sh] = {"hlen": uhlen, "header": uh}
        b_hdrs[sh] = {"hlen": bhlen, "header": bh}
        if uhlen != bhlen or uh != bh:
            diff_hdrs += 1
    (w / "shard_headers_u.json").write_text(json.dumps(u_hdrs))
    (w / "shard_headers_b.json").write_text(json.dumps(b_hdrs))
    print(f"Verified 24/24 shard headers. Differing headers: {diff_hdrs}")


if __name__ == "__main__":
    main()
