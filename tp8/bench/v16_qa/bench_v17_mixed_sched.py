#!/usr/bin/env python3
"""V17 Live Qualification Suite (Pure Python Stdlib) for InklingScheduler, TEMP_V16+min_p, and Model Alias."""

from __future__ import annotations

import argparse
import concurrent.futures
import http.client
import json
import random
import statistics
import time
import urllib.parse


def stream_chat(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float = 0.0,
    extra: dict | None = None,
    timeout: float = 300.0,
) -> dict:
    parsed = urllib.parse.urlparse(base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    payload: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if extra:
        payload.update(extra)
    body_bytes = json.dumps(payload).encode("utf-8")
    t0 = time.perf_counter()
    ttft = None
    chunks = 0
    comp_tokens = 0
    prompt_tokens = 0
    text_parts: list[str] = []
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        conn.request(
            "POST",
            "/v1/chat/completions",
            body=body_bytes,
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        )
        resp = conn.getresponse()
        status = resp.status
        if status != 200:
            err = resp.read().decode("utf-8", errors="ignore")[:300]
            return {"status": status, "error": err, "elapsed": round(time.perf_counter() - t0, 3)}
        while True:
            raw = resp.readline()
            if not raw:
                break
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            try:
                ev = json.loads(data)
            except json.JSONDecodeError:
                continue
            usage = ev.get("usage")
            if usage:
                comp_tokens = int(usage.get("completion_tokens", comp_tokens))
                prompt_tokens = int(usage.get("prompt_tokens", prompt_tokens))
            choices = ev.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                piece = (
                    delta.get("content")
                    or delta.get("reasoning_content")
                    or delta.get("reasoning")
                    or ""
                )
                if piece:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    chunks += 1
                    if len(text_parts) < 20:
                        text_parts.append(piece)
    finally:
        conn.close()
    elapsed = time.perf_counter() - t0
    if comp_tokens <= 0:
        comp_tokens = chunks
    gen_time = max(1e-3, elapsed - (ttft or 0.0))
    return {
        "status": 200,
        "ttft": round(ttft or elapsed, 3),
        "elapsed": round(elapsed, 3),
        "prompt_tokens": prompt_tokens,
        "comp_tokens": comp_tokens,
        "dec_tps": round(max(0, comp_tokens - 1) / gen_time, 2) if comp_tokens > 1 else 0.0,
        "e2e_tps": round(comp_tokens / max(1e-3, elapsed), 2),
        "preview": "".join(text_parts)[:120].replace("\n", " "),
    }


def make_long_prompt(target_tokens: int, salt: str) -> str:
    vocab = [
        "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
        "india", "juliet", "kilo", "lima", "mike", "november", "oscar", "papa",
        "quebec", "romeo", "sierra", "tango", "uniform", "victor", "whiskey", "xray",
        "yankee", "zulu", "matrix", "vector", "tensor", "kernel", "packet", "fabric",
    ]
    rng = random.Random(salt)
    n_words = int(target_tokens * 0.72)
    words = [rng.choice(vocab) for _ in range(n_words)]
    return f"Salt={salt}. Read the log and reply with just OK.\n" + " ".join(words) + "\nReply with OK."


def run_suite(base_url: str) -> dict:
    results: dict = {}

    # 1. Alias check (both canonical lowercase & Title-Case)
    r_lc = stream_chat(base_url, "inkling-2-small", "Reply with PONG.", 16, 0.0)
    r_tc = stream_chat(base_url, "Inkling-2-Small", "Reply with PONG.", 16, 0.0)
    results["alias"] = {"lowercase": r_lc["status"], "titlecase": r_tc["status"]}
    print(f"[1] Alias check: inkling-2-small={r_lc['status']} Inkling-2-Small={r_tc['status']}", flush=True)

    # 2. min_p=0.05 + temp=0.6 under spec decode
    r_minp = stream_chat(
        base_url, "inkling-2-small",
        "Explain in 3 concise sentences why RoCEv2 uses PFC or ECN.",
        128, 0.6, {"top_p": 0.95, "min_p": 0.05},
    )
    results["min_p"] = r_minp
    print(
        f"[2] min_p=0.05 test: status={r_minp['status']} comp={r_minp.get('comp_tokens')} "
        f"dec_tps={r_minp.get('dec_tps')} err={r_minp.get('error','')}",
        flush=True,
    )

    # 3. Non-greedy temp=0.6 throughput (C=1, C=8, C=16)
    temp_res = {}
    for c in (1, 8, 16):
        prompts = [
            f"Task {i} (salt {time.time_ns()}): Write a clear technical overview of GPU warp scheduling and shared memory bank conflicts."
            for i in range(c)
        ]
        t0 = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=c) as ex:
            outs = list(
                ex.map(
                    lambda pr: stream_chat(base_url, "inkling-2-small", pr, 256, 0.6, {"top_p": 0.95}),
                    prompts,
                )
            )
        wall = time.perf_counter() - t0
        ok = [o for o in outs if o["status"] == 200]
        tot_tok = sum(o["comp_tokens"] for o in ok)
        agg_tps = round(tot_tok / max(1e-3, wall), 2)
        mean_dec = round(statistics.mean(o["dec_tps"] for o in ok), 2) if ok else 0.0
        min_dec = round(min(o["dec_tps"] for o in ok), 2) if ok else 0.0
        temp_res[f"C{c}"] = {
            "ok": len(ok),
            "agg_tps": agg_tps,
            "mean_dec": mean_dec,
            "min_dec": min_dec,
            "wall": round(wall, 2),
        }
        print(
            f"[3] temp=0.6 C={c}: {len(ok)}/{c} ok, agg={agg_tps} tok/s "
            f"[mean={mean_dec} / min={min_dec}], wall={wall:.2f}s",
            flush=True,
        )
    results["temp_06"] = temp_res

    # 4. Mixed C=8 Decode + Cold 128K Prefill + Urgent Short Chat
    long_prompt = make_long_prompt(120000, f"v17-mixed-{time.time_ns()}")
    dec_prompts = [
        f"Stream {i} ({time.time_ns()}): Derive step-by-step the mathematics of rotary positional embeddings and FlashAttention online softmax with detailed formulas and examples."
        for i in range(8)
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as ex:
        dec_futs = [
            ex.submit(stream_chat, base_url, "inkling-2-small", pr, 320, 0.0)
            for pr in dec_prompts
        ]
        time.sleep(0.8)
        t_pf = time.perf_counter()
        pf_fut = ex.submit(
            stream_chat, base_url, "inkling-2-small", long_prompt, 16, 0.0, {"priority": 10}
        )
        time.sleep(0.6)
        short_fut = ex.submit(
            stream_chat,
            base_url,
            "inkling-2-small",
            f"Urgent short check ({time.time_ns()}): What is 17 * 23? Answer briefly.",
            32,
            0.0,
            {"priority": -10},
        )
        short_res = short_fut.result()
        dec_outs = [f.result() for f in dec_futs]
        pf_res = pf_fut.result()
        pf_wall = round(time.perf_counter() - t_pf, 2)

    dec_ok = [o for o in dec_outs if o["status"] == 200]
    dec_tps_list = [o["dec_tps"] for o in dec_ok]
    sched_stats = {}
    try:
        with open("/dev/shm/inkling_v17_sched_stats.json") as fh:
            sched_stats = json.load(fh)
    except Exception:
        pass

    results["mixed_128k"] = {
        "c8_ok": len(dec_ok),
        "c8_mean_dec_tps": round(statistics.mean(dec_tps_list), 2) if dec_tps_list else 0.0,
        "c8_min_dec_tps": round(min(dec_tps_list), 2) if dec_tps_list else 0.0,
        "c8_agg_dec_tps": round(sum(dec_tps_list), 2),
        "short_chat": short_res,
        "prefill_128k": {**pf_res, "wall": pf_wall},
        "sched_stats": sched_stats,
    }
    print(
        f"[4] Mixed C=8 + 128K + Short: C8 mean={results['mixed_128k']['c8_mean_dec_tps']} "
        f"(min={results['mixed_128k']['c8_min_dec_tps']}, agg={results['mixed_128k']['c8_agg_dec_tps']}) tok/s | "
        f"short TTFT={short_res.get('ttft')}s elapsed={short_res.get('elapsed')}s | "
        f"128K prompt_tok={pf_res.get('prompt_tokens')} wall={pf_wall}s | "
        f"sched={ {k: sched_stats.get(k) for k in ('pure_decode','pure_prefill','short_skip','hold','probe','pure_fallback')} }",
        flush=True,
    )
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8010")
    ap.add_argument("--out", default="/tmp/v17_mixed_sched.json")
    args = ap.parse_args()
    res = run_suite(args.url)
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"Saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
