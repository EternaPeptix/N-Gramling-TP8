#!/usr/bin/env python3
"""C1 decode + TTFT smoke benchmark for the inkling2-tp4 lane.

Usage: python3 bench/bench.py [--host http://<head>:8000] [--tokens 256]
"""
import argparse
import json
import time
import urllib.request


def run(host, model, tokens):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Count from 1 to 80 separated by spaces."}],
        "max_tokens": tokens,
        "temperature": 0,
    }
    req = urllib.request.Request(
        f"{host}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        body = json.load(r)
    dt = time.time() - t0
    u = body["usage"]
    ct, pt = u["completion_tokens"], u["prompt_tokens"]
    c = body["choices"][0]["message"]
    print(f"C1: {ct} tok in {dt:.1f}s = {ct / dt:.1f} tok/s decode "
          f"(prompt {pt} tok, TTFT incl.)")
    print("content:", (c.get("content") or "")[:90])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://localhost:8000")
    ap.add_argument("--model", default="inkling-2-small")
    ap.add_argument("--tokens", type=int, default=256)
    a = ap.parse_args()
    run(a.host, a.model, a.tokens)
    run(a.host, a.model, a.tokens)  # warm second pass
