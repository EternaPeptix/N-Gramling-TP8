#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Wave-V15 Gate 7: Mixed Workload Fairness Benchmark (Cold 1M Prefill + Interactive C=4/C=8 Decode).

Runs a cold long-context (default 1,048,500-token) needle-in-a-haystack prefill
simultaneously with continuous interactive streaming decode waves (default C=8
or C=4), verifying:
  1. The cold long-context prompt completes cleanly (cached_tokens < 5%) with 3/3 needles.
  2. Interactive decode streams genuinely overlap with the active 1M prefill window.
  3. Zero starvation, zero deadlock, zero HTTP/SSE errors, and bounded interactive ITL/TTFT tails.

Pure Python 3.10+ stdlib (zero third-party dependencies).
"""
from __future__ import annotations

import argparse
import concurrent.futures
import http.client
import json
import math
import random
import socket
import statistics
import sys
import threading
import time
import urllib.parse
import uuid
from typing import Any, Dict, List, Optional, Tuple


_FILLER_SENTENCES = (
    "Telemetry archival record confirming nominal subsystem cadence across optical fabric lanes.",
    "Deterministic checkpoint verification logs recorded without thermal throttling anomalies.",
    "Distributed tensor shard synchronization completed with zero checksum discrepancies.",
    "Unified memory allocator contiguous page audit logged under steady inference pressure.",
    "Sliding window attention ring buffer maintenance preserved chronological token alignment.",
    "Speculative draft verification batch histogram captured balanced acceptance distributions.",
    "Periodic network fabric health probe confirmed active redundant RDMA queue pairs.",
    "Immutable ledger entry archived with cryptographic digest continuity preserved.",
)

_INTERACTIVE_PROMPTS = (
    "Explain in clear technical prose how sliding-window attention and global attention complement each other in a hybrid transformer.",
    "Write a concise Python function to compute streaming percentiles using reservoir sampling and explain its complexity.",
    "List five key engineering practices for preventing memory fragmentation in long-running GPU inference servers.",
    "Compare Ring AllReduce and ReduceScatter+AllGather bandwidth trade-offs on an 8-node RDMA cluster.",
    "Describe why deterministic floating-point reduction order matters for reproducible speculative decoding.",
    "Provide a short checklist for diagnosing sudden tail-latency spikes in concurrent LLM serving.",
    "Summarize the trade-offs between larger prefill chunk sizes and interactive inter-token latency.",
    "Write a brief walkthrough of LRU prefix-cache eviction when multiple 1M-token conversations compete for KV slots.",
)


def percentile(values: List[float], p: float) -> float:
    """Compute linear-interpolated percentile (0 <= p <= 100)."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    k = (len(s) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(s[int(k)])
    return float(s[f] + (s[c] - s[f]) * (k - f))


def parse_base_url(base_url: str) -> Tuple[str, str, int, str]:
    """Return (scheme, host, port, base_path)."""
    parsed = urllib.parse.urlsplit(base_url.strip())
    scheme = parsed.scheme or "http"
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if scheme == "https" else 80)
    base_path = parsed.path.rstrip("/")
    return scheme, host, port, base_path


def make_http_conn(base_url: str, timeout: float) -> Tuple[http.client.HTTPConnection, str]:
    scheme, host, port, base_path = parse_base_url(base_url)
    if scheme == "https":
        conn: http.client.HTTPConnection = http.client.HTTPSConnection(host, port, timeout=timeout)
    else:
        conn = http.client.HTTPConnection(host, port, timeout=timeout)
    return conn, base_path


def probe_tokenize_count(base_url: str, model: str, text: str, timeout: float = 15.0) -> Optional[int]:
    """Ask vLLM /tokenize for an exact token count; returns None if unsupported."""
    conn = None
    try:
        conn, base_path = make_http_conn(base_url, timeout=timeout)
        body = json.dumps({"model": model, "prompt": text}).encode("utf-8")
        conn.request(
            "POST",
            f"{base_path}/tokenize",
            body=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        resp = conn.getresponse()
        raw = resp.read()
        if resp.status != 200:
            return None
        data = json.loads(raw.decode("utf-8", errors="replace"))
        if isinstance(data.get("count"), int):
            return int(data["count"])
        if isinstance(data.get("tokens"), list):
            return len(data["tokens"])
        return None
    except Exception:
        return None
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def build_cold_needle_prompt(
    base_url: str,
    model: str,
    target_ctx: int,
    seed: int = 15007,
    max_model_len: int = 1048576,
    max_tokens: int = 256,
    calibrate_tokenize: bool = True,
) -> Tuple[List[Dict[str, str]], Dict[str, str], int, str]:
    """Build a unique cold haystack prompt with 3 Vault passcodes at 10%, 50%, 90% depth."""
    rng = random.Random(seed)
    run_uuid = uuid.UUID(int=rng.getrandbits(128)).hex
    needles = {
        "Vault-Alpha": f"ALPHA-{rng.randint(100000, 999999)}-{run_uuid[:6].upper()}",
        "Vault-Beta": f"BETA-{rng.randint(100000, 999999)}-{run_uuid[6:12].upper()}",
        "Vault-Gamma": f"GAMMA-{rng.randint(100000, 999999)}-{run_uuid[12:18].upper()}",
    }

    safe_target = min(max(target_ctx, 256), max(256, max_model_len - max_tokens - 128))
    sys_msg = (
        "You are a precise archival retrieval assistant. Extract the exact Vault passcodes "
        "embedded in the user's archive verbatim without altering characters."
    )
    header = (
        f"[COLD-ARCHIVE-SESSION uuid={run_uuid} seed={seed} target_ctx={safe_target}]\n"
        "The following archive contains numbered telemetry entries and three Vault passcodes.\n"
    )
    footer = (
        "\nList the exact three passcodes found in the archive above:\n"
        "1. Vault-Alpha: <passcode>\n"
        "2. Vault-Beta: <passcode>\n"
        "3. Vault-Gamma: <passcode>\n"
        "Reply concisely with ONLY those three lines."
    )

    sample_blocks = [
        f"[DOC-{run_uuid[:8]}-BLK-{i:06d}] {_FILLER_SENTENCES[i % len(_FILLER_SENTENCES)]}"
        for i in range(64)
    ]
    sample_text = "\n".join(sample_blocks)
    tok_per_block = max(8.0, len(sample_text) / (64.0 * 4.0))
    if calibrate_tokenize:
        measured = probe_tokenize_count(base_url, model, sample_text)
        if measured and measured > 0:
            tok_per_block = measured / 64.0

    overhead_tokens = 180
    n_blocks = max(12, int((safe_target - overhead_tokens) / max(tok_per_block, 1.0)))
    pos_alpha = max(1, int(n_blocks * 0.10))
    pos_beta = max(pos_alpha + 1, int(n_blocks * 0.50))
    pos_gamma = max(pos_beta + 1, min(n_blocks - 1, int(n_blocks * 0.90)))

    prefix_tag = run_uuid[:8]
    parts: List[str] = [header]
    for i in range(n_blocks):
        if i == pos_alpha:
            parts.append(f"[CRITICAL-RECORD] Vault-Alpha: {needles['Vault-Alpha']}")
        elif i == pos_beta:
            parts.append(f"[CRITICAL-RECORD] Vault-Beta: {needles['Vault-Beta']}")
        elif i == pos_gamma:
            parts.append(f"[CRITICAL-RECORD] Vault-Gamma: {needles['Vault-Gamma']}")
        parts.append(f"[DOC-{prefix_tag}-BLK-{i:06d}] {_FILLER_SENTENCES[i % len(_FILLER_SENTENCES)]}")
    parts.append(footer)

    user_content = "\n".join(parts)
    est_tokens = int(n_blocks * tok_per_block + overhead_tokens)
    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": user_content},
    ]
    return messages, needles, est_tokens, run_uuid


def score_needles(
    needles: Dict[str, str],
    content: str,
    reasoning: str = "",
    forbidden_needles: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Check 3/3 Vault needles and optionally check for stale/cross-session needle leakage."""
    haystack = f"{content or ''}\n{reasoning or ''}"
    found: Dict[str, bool] = {}
    for k, v in needles.items():
        found[k] = v in haystack
    stale_leaks: List[str] = []
    if forbidden_needles:
        for fk, fv in forbidden_needles.items():
            if fv and fv in haystack and fv not in needles.values():
                stale_leaks.append(f"{fk}={fv}")
    hits = sum(1 for ok in found.values() if ok)
    return {
        "hits": hits,
        "total": len(needles),
        "per_needle": found,
        "stale_leaks": stale_leaks,
        "ok": (hits == len(needles) and len(stale_leaks) == 0),
    }


def extract_cached_tokens(usage: Dict[str, Any]) -> int:
    if not isinstance(usage, dict):
        return 0
    ptd = usage.get("prompt_tokens_details")
    if isinstance(ptd, dict) and isinstance(ptd.get("cached_tokens"), int):
        return int(ptd["cached_tokens"])
    if isinstance(usage.get("cached_tokens"), int):
        return int(usage["cached_tokens"])
    return 0


def run_streaming_chat(
    base_url: str,
    model: str,
    messages: List[Dict[str, Any]],
    max_tokens: int = 128,
    temperature: float = 0.0,
    timeout: float = 120.0,
    extra_payload: Optional[Dict[str, Any]] = None,
    on_token_ts: Optional[Any] = None,
) -> Dict[str, Any]:
    """Execute a streaming /v1/chat/completions call and collect TTFT, ITL, usage, and text."""
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if extra_payload:
        payload.update(extra_payload)

    body = json.dumps(payload).encode("utf-8")
    t_start = time.perf_counter()
    t_first_tok: Optional[float] = None
    t_prev_tok: Optional[float] = None
    itl_ms: List[float] = []
    content_parts: List[str] = []
    reasoning_parts: List[str] = []
    usage: Dict[str, Any] = {}
    chunks_seen = 0
    finish_reason: Optional[str] = None
    http_status = 0
    conn: Optional[http.client.HTTPConnection] = None

    try:
        conn, base_path = make_http_conn(base_url, timeout=timeout)
        conn.request(
            "POST",
            f"{base_path}/v1/chat/completions",
            body=body,
            headers={
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            },
        )
        resp = conn.getresponse()
        http_status = resp.status
        if resp.status != 200:
            err_body = resp.read(4096).decode("utf-8", errors="replace")
            return {
                "ok": False,
                "http_status": http_status,
                "error": f"HTTP {http_status}: {err_body[:300]}",
                "t_start": t_start,
                "t_end": time.perf_counter(),
                "total_s": time.perf_counter() - t_start,
                "ttft_s": 0.0,
                "itl_ms": [],
                "content": "",
                "reasoning": "",
                "prompt_tokens": 0,
                "cached_tokens": 0,
                "completion_tokens": 0,
                "prefill_tok_s": 0.0,
                "decode_tok_s": 0.0,
            }

        while True:
            raw_line = resp.readline()
            if not raw_line:
                break
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or line.startswith(":"):
                continue
            if not line.startswith("data:"):
                continue
            data_str = line[5:].strip()
            if data_str == "[DONE]":
                break
            try:
                evt = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            if isinstance(evt.get("usage"), dict) and evt["usage"]:
                usage = evt["usage"]

            choices = evt.get("choices") or []
            if choices:
                ch0 = choices[0]
                if ch0.get("finish_reason"):
                    finish_reason = ch0["finish_reason"]
                delta = ch0.get("delta") or {}
                c_txt = delta.get("content")
                r_txt = delta.get("reasoning_content") or delta.get("reasoning")
                got_piece = False
                if c_txt:
                    content_parts.append(str(c_txt))
                    got_piece = True
                if r_txt:
                    reasoning_parts.append(str(r_txt))
                    got_piece = True
                if got_piece:
                    now = time.perf_counter()
                    chunks_seen += 1
                    if t_first_tok is None:
                        t_first_tok = now
                    elif t_prev_tok is not None:
                        itl_ms.append((now - t_prev_tok) * 1000.0)
                    t_prev_tok = now
                    if on_token_ts is not None:
                        on_token_ts(now)

        t_end = time.perf_counter()
        total_s = max(t_end - t_start, 1e-6)
        ttft_s = max((t_first_tok - t_start), 1e-6) if t_first_tok is not None else total_s
        decode_s = max(t_end - t_first_tok, 1e-6) if t_first_tok is not None else 0.0

        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        cached_tokens = extract_cached_tokens(usage)
        completion_tokens = int(usage.get("completion_tokens") or chunks_seen)
        uncached_prompt = max(prompt_tokens - cached_tokens, 0)
        prefill_tok_s = (uncached_prompt / ttft_s) if (uncached_prompt > 0 and ttft_s > 0) else 0.0
        if completion_tokens > 1 and decode_s > 0:
            decode_tok_s = (completion_tokens - 1) / decode_s
        elif completion_tokens > 0 and total_s > 0:
            decode_tok_s = completion_tokens / total_s
        else:
            decode_tok_s = 0.0

        return {
            "ok": True,
            "http_status": http_status,
            "error": None,
            "t_start": t_start,
            "t_end": t_end,
            "t_first_tok": t_first_tok,
            "total_s": total_s,
            "ttft_s": ttft_s,
            "decode_s": decode_s,
            "itl_ms": itl_ms,
            "chunks_seen": chunks_seen,
            "finish_reason": finish_reason,
            "content": "".join(content_parts),
            "reasoning": "".join(reasoning_parts),
            "prompt_tokens": prompt_tokens,
            "cached_tokens": cached_tokens,
            "completion_tokens": completion_tokens,
            "prefill_tok_s": prefill_tok_s,
            "decode_tok_s": decode_tok_s,
        }
    except (socket.timeout, TimeoutError) as exc:
        return {
            "ok": False,
            "http_status": http_status,
            "error": f"TIMEOUT: {exc}",
            "timeout": True,
            "t_start": t_start,
            "t_end": time.perf_counter(),
            "total_s": time.perf_counter() - t_start,
            "ttft_s": 0.0,
            "itl_ms": itl_ms,
            "content": "".join(content_parts),
            "reasoning": "".join(reasoning_parts),
            "prompt_tokens": 0,
            "cached_tokens": 0,
            "completion_tokens": chunks_seen,
            "prefill_tok_s": 0.0,
            "decode_tok_s": 0.0,
        }
    except Exception as exc:
        return {
            "ok": False,
            "http_status": http_status,
            "error": f"{type(exc).__name__}: {exc}",
            "timeout": False,
            "t_start": t_start,
            "t_end": time.perf_counter(),
            "total_s": time.perf_counter() - t_start,
            "ttft_s": 0.0,
            "itl_ms": itl_ms,
            "content": "".join(content_parts),
            "reasoning": "".join(reasoning_parts),
            "prompt_tokens": 0,
            "cached_tokens": 0,
            "completion_tokens": chunks_seen,
            "prefill_tok_s": 0.0,
            "decode_tok_s": 0.0,
        }
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def summarize_interactive(records: List[Dict[str, Any]], token_timestamps: List[float]) -> Dict[str, Any]:
    ok_recs = [r for r in records if r.get("ok")]
    err_recs = [r for r in records if not r.get("ok")]
    timeout_recs = [r for r in err_recs if r.get("timeout")]

    all_itl: List[float] = []
    ttft_ms: List[float] = []
    req_tok_s: List[float] = []
    total_comp_toks = 0
    for r in ok_recs:
        all_itl.extend(r.get("itl_ms") or [])
        ttft_ms.append(float(r.get("ttft_s", 0.0)) * 1000.0)
        req_tok_s.append(float(r.get("decode_tok_s", 0.0)))
        total_comp_toks += int(r.get("completion_tokens", 0))

    max_global_gap_ms = 0.0
    if len(token_timestamps) >= 2:
        ts_sorted = sorted(token_timestamps)
        gaps = [(ts_sorted[i] - ts_sorted[i - 1]) * 1000.0 for i in range(1, len(ts_sorted))]
        max_global_gap_ms = max(gaps) if gaps else 0.0

    wall_span = 0.0
    if ok_recs:
        wall_span = max(max(r["t_end"] for r in ok_recs) - min(r["t_start"] for r in ok_recs), 1e-6)

    return {
        "total_requests": len(records),
        "completed": len(ok_recs),
        "errors": len(err_recs),
        "timeouts": len(timeout_recs),
        "sample_errors": [r.get("error") for r in err_recs[:5]],
        "total_completion_tokens": total_comp_toks,
        "agg_tok_s": round(total_comp_toks / wall_span, 2) if wall_span > 0 else 0.0,
        "req_decode_tok_s_mean": round(statistics.mean(req_tok_s), 2) if req_tok_s else 0.0,
        "req_decode_tok_s_min": round(min(req_tok_s), 2) if req_tok_s else 0.0,
        "ttft_ms_mean": round(statistics.mean(ttft_ms), 2) if ttft_ms else 0.0,
        "ttft_ms_p50": round(percentile(ttft_ms, 50), 2),
        "ttft_ms_p95": round(percentile(ttft_ms, 95), 2),
        "ttft_ms_p99": round(percentile(ttft_ms, 99), 2),
        "ttft_ms_max": round(max(ttft_ms), 2) if ttft_ms else 0.0,
        "itl_samples": len(all_itl),
        "itl_ms_mean": round(statistics.mean(all_itl), 2) if all_itl else 0.0,
        "itl_ms_p50": round(percentile(all_itl, 50), 2),
        "itl_ms_p95": round(percentile(all_itl, 95), 2),
        "itl_ms_p99": round(percentile(all_itl, 99), 2),
        "itl_ms_max": round(max(all_itl), 2) if all_itl else 0.0,
        "max_global_stall_ms": round(max_global_gap_ms, 2),
    }


def _run_interactive_wave(
    base_url: str,
    model: str,
    conc: int,
    max_tokens: int,
    timeout: float,
    wave_idx: int,
    seed: int,
    token_ts_sink: List[float],
    ts_lock: threading.Lock,
) -> List[Dict[str, Any]]:
    def _one(worker_id: int) -> Dict[str, Any]:
        prompt_base = _INTERACTIVE_PROMPTS[(wave_idx * conc + worker_id) % len(_INTERACTIVE_PROMPTS)]
        # Unique request tag avoids cross-wave trivial caching on interactive baseline
        tag = f"[interactive-w{wave_idx}-c{worker_id}-s{seed}]"
        msgs = [{"role": "user", "content": f"{tag} {prompt_base}"}]

        def _on_ts(ts: float) -> None:
            with ts_lock:
                token_ts_sink.append(ts)

        res = run_streaming_chat(
            base_url=base_url,
            model=model,
            messages=msgs,
            max_tokens=max_tokens,
            temperature=0.0,
            timeout=timeout,
            on_token_ts=_on_ts,
        )
        res["wave_idx"] = wave_idx
        res["worker_id"] = worker_id
        return res

    with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as ex:
        futs = [ex.submit(_one, i) for i in range(conc)]
        return [f.result() for f in futs]


def run_mixed_benchmark(
    base_url: str,
    model: str = "inkling-2-small",
    long_ctx: int = 1048500,
    long_max_tokens: int = 256,
    interactive_conc: int = 8,
    interactive_max_tokens: int = 128,
    baseline_waves: int = 1,
    min_interactive_waves: int = 2,
    max_interactive_waves: int = 50,
    stagger_s: float = 0.25,
    long_timeout_s: float = 1800.0,
    req_timeout_s: float = 120.0,
    max_itl_p99_ms: float = 4000.0,
    max_stall_ms: float = 10000.0,
    max_ttft_p99_ms: float = 15000.0,
    seed: int = 15007,
    quiet: bool = False,
) -> Dict[str, Any]:
    """Execute the complete mixed 1M cold prefill + interactive C=4/C=8 fairness test."""
    t_bench_start = time.perf_counter()

    # 1. Optional baseline interactive wave (unloaded cluster reference)
    baseline_summary: Optional[Dict[str, Any]] = None
    if baseline_waves > 0:
        base_recs: List[Dict[str, Any]] = []
        base_ts: List[float] = []
        base_lock = threading.Lock()
        for w in range(baseline_waves):
            base_recs.extend(
                _run_interactive_wave(
                    base_url, model, interactive_conc, interactive_max_tokens,
                    req_timeout_s, w, seed, base_ts, base_lock
                )
            )
        baseline_summary = summarize_interactive(base_recs, base_ts)
        if not quiet:
            print(
                f"[MIXED BASELINE C={interactive_conc}] completed={baseline_summary['completed']}/{baseline_summary['total_requests']} "
                f"ITL p50={baseline_summary['itl_ms_p50']}ms p99={baseline_summary['itl_ms_p99']}ms "
                f"TTFT p99={baseline_summary['ttft_ms_p99']}ms",
                flush=True,
            )

    # 2. Pre-build cold long-context needle prompt BEFORE starting overlap timer
    t_build0 = time.perf_counter()
    long_msgs, needles, tok_est, run_uuid = build_cold_needle_prompt(
        base_url=base_url,
        model=model,
        target_ctx=long_ctx,
        seed=seed,
        max_tokens=long_max_tokens,
    )
    build_s = time.perf_counter() - t_build0
    if not quiet:
        print(
            f"[MIXED LONG BUILD] target_ctx={long_ctx:,} est_tok={tok_est:,} "
            f"uuid={run_uuid[:12]} built_in={build_s:.2f}s",
            flush=True,
        )

    # 3. Launch cold 1M prefill in background + concurrent interactive decode waves
    long_done = threading.Event()
    long_result_box: Dict[str, Any] = {}
    mixed_recs: List[Dict[str, Any]] = []
    mixed_ts: List[float] = []
    mixed_lock = threading.Lock()

    def _long_worker() -> None:
        try:
            res = run_streaming_chat(
                base_url=base_url,
                model=model,
                messages=long_msgs,
                max_tokens=long_max_tokens,
                temperature=0.0,
                timeout=long_timeout_s,
                extra_payload={"chat_template_kwargs": {"enable_thinking": False}},
            )
            long_result_box["res"] = res
        finally:
            long_done.set()

    long_thread = threading.Thread(target=_long_worker, name="mixed-1m-cold", daemon=True)
    long_thread.start()

    if stagger_s > 0:
        long_done.wait(timeout=stagger_s)

    wave_idx = 0
    while wave_idx < max_interactive_waves:
        if long_done.is_set() and wave_idx >= min_interactive_waves:
            break
        wave_recs = _run_interactive_wave(
            base_url, model, interactive_conc, interactive_max_tokens,
            req_timeout_s, 100 + wave_idx, seed, mixed_ts, mixed_lock
        )
        mixed_recs.extend(wave_recs)
        wave_idx += 1
        if long_done.is_set() and wave_idx >= min_interactive_waves:
            break

    long_thread.join(timeout=long_timeout_s + 5.0)
    long_res = long_result_box.get("res") or {
        "ok": False,
        "error": "long_worker did not return result",
        "t_start": t_bench_start,
        "t_end": time.perf_counter(),
        "ttft_s": 0.0,
        "total_s": 0.0,
        "prompt_tokens": 0,
        "cached_tokens": 0,
        "completion_tokens": 0,
        "prefill_tok_s": 0.0,
        "decode_tok_s": 0.0,
        "content": "",
        "reasoning": "",
    }

    needle_score = score_needles(needles, long_res.get("content", ""), long_res.get("reasoning", ""))
    prompt_toks = int(long_res.get("prompt_tokens") or tok_est)
    cached_toks = int(long_res.get("cached_tokens") or 0)
    cold_verified = cached_toks <= max(64, int(prompt_toks * 0.05))

    # Measure strict overlap: interactive requests whose [t_start, t_end] overlapped long_res [t_start, t_end]
    l_start = float(long_res.get("t_start", t_bench_start))
    l_end = float(long_res.get("t_end", time.perf_counter()))
    overlapping_recs = [
        r for r in mixed_recs
        if float(r.get("t_start", 0.0)) <= l_end and float(r.get("t_end", 0.0)) >= l_start
    ]
    overlap_ts = [ts for ts in mixed_ts if l_start <= ts <= l_end]

    mixed_summary = summarize_interactive(mixed_recs, mixed_ts)
    overlap_summary = summarize_interactive(overlapping_recs, overlap_ts) if overlapping_recs else mixed_summary

    itl_p99_ratio = None
    if baseline_summary and baseline_summary["itl_ms_p99"] > 0:
        itl_p99_ratio = round(mixed_summary["itl_ms_p99"] / baseline_summary["itl_ms_p99"], 2)

    checks = {
        "long_http_ok": bool(long_res.get("ok")),
        "long_cold_verified": bool(cold_verified),
        "long_needles_3_of_3": bool(needle_score["ok"]),
        "interactive_zero_errors": mixed_summary["errors"] == 0,
        "interactive_zero_timeouts": mixed_summary["timeouts"] == 0,
        "interactive_overlapped_long": len(overlapping_recs) >= min(interactive_conc, len(mixed_recs)) and len(overlapping_recs) > 0,
        "itl_p99_within_gate": mixed_summary["itl_ms_p99"] <= max_itl_p99_ms,
        "max_stall_within_gate": (
            mixed_summary["itl_ms_max"] <= max_stall_ms
            and mixed_summary["max_global_stall_ms"] <= max_stall_ms
        ),
        "ttft_p99_within_gate": mixed_summary["ttft_ms_p99"] <= max_ttft_p99_ms,
    }
    passed = all(checks.values())
    errors: List[str] = []
    if not checks["long_http_ok"]:
        errors.append(f"Cold long prefill failed: {long_res.get('error')}")
    if not checks["long_cold_verified"]:
        errors.append(f"Long prompt was not cold: cached={cached_toks} / prompt={prompt_toks}")
    if not checks["long_needles_3_of_3"]:
        errors.append(f"Long needle check failed: {needle_score['hits']}/{needle_score['total']}")
    if not checks["interactive_zero_errors"]:
        errors.append(f"Interactive errors={mixed_summary['errors']}: {mixed_summary['sample_errors']}")
    if not checks["interactive_zero_timeouts"]:
        errors.append(f"Interactive timeouts/deadlocks={mixed_summary['timeouts']}")
    if not checks["interactive_overlapped_long"]:
        errors.append(f"Insufficient overlap: {len(overlapping_recs)} overlapped requests")
    if not checks["itl_p99_within_gate"]:
        errors.append(f"Interactive ITL p99 {mixed_summary['itl_ms_p99']}ms > {max_itl_p99_ms}ms")
    if not checks["max_stall_within_gate"]:
        errors.append(
            f"Interactive stall exceeded {max_stall_ms}ms "
            f"(req_max={mixed_summary['itl_ms_max']}ms, global_gap={mixed_summary['max_global_stall_ms']}ms)"
        )
    if not checks["ttft_p99_within_gate"]:
        errors.append(f"Interactive TTFT p99 {mixed_summary['ttft_ms_p99']}ms > {max_ttft_p99_ms}ms")

    report = {
        "benchmark": "v15_mixed_1m_c8_bench",
        "status": "PASS" if passed else "FAIL",
        "duration_s": round(time.perf_counter() - t_bench_start, 3),
        "config": {
            "base_url": base_url,
            "model": model,
            "long_ctx": long_ctx,
            "long_max_tokens": long_max_tokens,
            "interactive_conc": interactive_conc,
            "interactive_max_tokens": interactive_max_tokens,
            "waves_executed": wave_idx,
            "max_itl_p99_ms": max_itl_p99_ms,
            "max_stall_ms": max_stall_ms,
            "max_ttft_p99_ms": max_ttft_p99_ms,
            "seed": seed,
        },
        "long_prefill": {
            "uuid": run_uuid,
            "ok": bool(long_res.get("ok")),
            "error": long_res.get("error"),
            "prompt_tokens": prompt_toks,
            "cached_tokens": cached_toks,
            "cold_verified": cold_verified,
            "completion_tokens": int(long_res.get("completion_tokens") or 0),
            "ttft_s": round(float(long_res.get("ttft_s") or 0.0), 3),
            "total_s": round(float(long_res.get("total_s") or 0.0), 3),
            "prefill_tok_s": round(float(long_res.get("prefill_tok_s") or 0.0), 1),
            "decode_tok_s": round(float(long_res.get("decode_tok_s") or 0.0), 2),
            "needles": needle_score,
        },
        "interactive_baseline": baseline_summary,
        "interactive_during_mixed": mixed_summary,
        "interactive_strict_overlap": {
            "overlapped_requests": len(overlapping_recs),
            "summary": overlap_summary,
            "itl_p99_ratio_vs_baseline": itl_p99_ratio,
        },
        "checks": checks,
        "errors": errors,
    }

    if not quiet:
        print(
            f"[MIXED RESULT] status={report['status']} | 1M cold TTFT={report['long_prefill']['ttft_s']}s "
            f"needles={needle_score['hits']}/{needle_score['total']} | "
            f"C={interactive_conc} completed={mixed_summary['completed']}/{mixed_summary['total_requests']} "
            f"overlap={len(overlapping_recs)} ITL p50={mixed_summary['itl_ms_p50']}ms "
            f"p99={mixed_summary['itl_ms_p99']}ms max={mixed_summary['itl_ms_max']}ms",
            flush=True,
        )
    return report


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Wave-V15 Gate 7: Mixed Cold 1M Prefill + Interactive C=4/C=8 Fairness Benchmark"
    )
    ap.add_argument("--base-url", default="http://127.0.0.1:8010", help="OpenAI-compatible server base URL")
    ap.add_argument("--model", default="inkling-2-small", help="Served model name")
    ap.add_argument("--long-ctx", type=int, default=1048500, help="Cold long-context token target")
    ap.add_argument("--long-max-tokens", type=int, default=256, help="Max completion tokens for long request")
    ap.add_argument("--interactive-conc", type=int, default=8, help="Interactive decode concurrency (e.g. 4 or 8)")
    ap.add_argument("--interactive-max-tokens", type=int, default=128, help="Max tokens per interactive request")
    ap.add_argument("--baseline-waves", type=int, default=1, help="Pre-1M baseline interactive waves")
    ap.add_argument("--min-interactive-waves", type=int, default=2, help="Minimum interactive waves during mixed phase")
    ap.add_argument("--max-interactive-waves", type=int, default=50, help="Safety cap on interactive waves")
    ap.add_argument("--stagger-s", type=float, default=0.25, help="Delay after launching 1M before starting C=N waves")
    ap.add_argument("--long-timeout-s", type=float, default=1800.0, help="Timeout for cold 1M prefill+decode")
    ap.add_argument("--req-timeout-s", type=float, default=120.0, help="Timeout per interactive request")
    ap.add_argument("--max-itl-p99-ms", type=float, default=4000.0, help="Interactive ITL p99 ceiling (ms)")
    ap.add_argument("--max-stall-ms", type=float, default=10000.0, help="Max interactive stall ceiling (ms)")
    ap.add_argument("--max-ttft-p99-ms", type=float, default=15000.0, help="Interactive TTFT p99 ceiling (ms)")
    ap.add_argument("--seed", type=int, default=15007, help="Deterministic RNG seed")
    ap.add_argument("--out-json", default="", help="Optional path to write JSON report")
    ap.add_argument("--quiet", action="store_true", help="Suppress progress output")
    args = ap.parse_args(argv)

    report = run_mixed_benchmark(
        base_url=args.base_url,
        model=args.model,
        long_ctx=args.long_ctx,
        long_max_tokens=args.long_max_tokens,
        interactive_conc=args.interactive_conc,
        interactive_max_tokens=args.interactive_max_tokens,
        baseline_waves=args.baseline_waves,
        min_interactive_waves=args.min_interactive_waves,
        max_interactive_waves=args.max_interactive_waves,
        stagger_s=args.stagger_s,
        long_timeout_s=args.long_timeout_s,
        req_timeout_s=args.req_timeout_s,
        max_itl_p99_ms=args.max_itl_p99_ms,
        max_stall_ms=args.max_stall_ms,
        max_ttft_p99_ms=args.max_ttft_p99_ms,
        seed=args.seed,
        quiet=args.quiet,
    )

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
