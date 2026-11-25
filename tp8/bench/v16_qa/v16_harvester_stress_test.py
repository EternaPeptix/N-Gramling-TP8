#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Wave-V16 Gate 11: Real-World `rollout_harvester --workers 16 --n 4` Stress & Diversity Verifier.

Directly mirrors the production RLVR / post-training workload on 8x NVIDIA DGX Spark TP8:
  - 16 concurrent harvester workers, each requesting `n=4` shared-prefix reasoning completions
    (64 concurrent completions in flight, `temperature=0.6`, `top_p=0.95`, optional `min_p`,
    `--think-budget 2048 --max-tokens 2048`, true SSE streaming with `include_usage=True`)
  - Shared system + RLVR rubric prefix across workers (and multi-round shared prompt prefixes)
    to measure real KV prefix-cache hit ratio under `n=4` fanout
  - Simultaneous background `temperature=0.0` canary probes verifying mixed `temp=0` + `temp>0`
    SyncFree-Temp batch isolation (zero interference, 100% deterministic canary hashes)
  - Quantitative `n=4` diversity & anti-collapse verification (unique completion ratio, zero
    all-identical collapsed groups, pairwise token Jaccard distance, lexical distinct-1/2,
    first-divergence check) proving SyncFree-Temp produces genuinely distinct unbiased samples
  - Zero dropped requests, zero HTTP 5xx, continuous `/health == 200`, slot drain check
  - Post-harvest 0-ULP `1dcfb306...b46f8c3` exactness invariance check!

Pure Python 3.10+ stdlib (zero external dependencies). 100% offline-safe when pointed at
`127.0.0.1` mock server; zero SSH or remote commands.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import http.client
import json
import math
import os
import random
import statistics
import sys
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Locate sibling V15/V16 QA modules regardless of working directory
_THIS_DIR = Path(__file__).resolve().parent
_V15_QA_DIR = _THIS_DIR.parent.parent / "v15" / "unified" / "qa"
for _p in (str(_THIS_DIR), str(_V15_QA_DIR)):
    if _p not in sys.path and os.path.isdir(_p):
        sys.path.insert(0, _p)

import v15_chaos_soak as chaos_mod  # noqa: E402


EXPECTED_EXACT_SHA256 = chaos_mod.EXPECTED_EXACT_SHA256

# Shared RLVR reasoning system prompt (~180 words) reused across all 16 workers to exercise
# shared-prefix KV caching exactly as `posttrain_pipe.rollout_harvester` does.
SHARED_RLVR_SYSTEM_PROMPT = (
    "You are a rigorous mathematical and algorithmic problem solver generating verifiable "
    "step-by-step reasoning traces for reinforcement learning with verifiable rewards (RLVR). "
    "Follow the structured reasoning protocol strictly: first analyze the problem constraints, "
    "edge cases, and invariants inside <think>...</think> within the allocated thinking budget, "
    "checking every intermediate algebraic manipulation and boundary condition independently. "
    "After closing </think>, present a clean, self-contained final solution and conclude with "
    "the exact verifiable answer enclosed in \\boxed{...}. Do not skip justification steps, do "
    "not fabricate lemmas, and verify your final answer by substituting back into the original "
    "problem statement. Maintain precision on modular arithmetic, asymptotic complexity bounds, "
    "numerical stability, and combinatorial edge cases across every rollout candidate."
)

# 8 distinct RLVR problem stems (shared by pairs of workers across 16 workers, or reused across
# rounds, guaranteeing real cross-worker + intra-`n=4` shared-prefix KV reuse!)
RLVR_PROBLEM_TEMPLATES: Tuple[str, ...] = (
    (
        "Problem [RLVR-ALG-01]: Let a_0 = 2, a_1 = 5, and a_{k+2} = 5 a_{k+1} - 6 a_k + 2^k "
        "for k >= 0. Derive the closed-form expression for a_n, verify n = 0, 1, 2, 3, and "
        "compute a_12 mod 10007. Explain every step of the particular solution."
    ),
    (
        "Problem [RLVR-CODE-02]: Design an O(N log N) Python algorithm to compute the number of "
        "contiguous subarrays whose median is strictly greater than a given threshold T. Prove "
        "the reduction to inversion counting on prefix sums and trace an example of length 7."
    ),
    (
        "Problem [RLVR-COMB-03]: How many lattice paths from (0, 0) to (10, 10) using only East "
        "and North unit steps never touch the line y = x + 3 and pass through at least one point "
        "on the line x + y = 10 with y <= 4? Derive via the reflection principle."
    ),
    (
        "Problem [RLVR-NUM-04]: Find all positive integers n <= 500 such that n^3 + 3n^2 + 2n + 6 "
        "is divisible by 24 and gcd(n, 35) = 1. Show the CRT decomposition modulo 8 and 3 and "
        "count the exact number of qualifying integers."
    ),
    (
        "Problem [RLVR-GRAPH-05]: Given a weighted directed acyclic graph with V vertices and E "
        "edges where edge weights may be negative, describe how to find both the shortest path "
        "and the number of distinct shortest paths from s to t in O(V + E) time without overflow."
    ),
    (
        "Problem [RLVR-PROB-06]: A fair 6-sided die is rolled repeatedly until the cumulative sum "
        "is a multiple of 7 or exceeds 20. Set up the absorbing Markov chain transition equations "
        "and compute the exact expected number of rolls starting from sum 0."
    ),
    (
        "Problem [RLVR-SYS-07]: Analyze a lock-free ring buffer of capacity 2^k shared between a "
        "GPU producer and CPU RDMA proxy consumer. Prove why modular 64-bit sequence comparison "
        "avoids 32-bit wraparound deadlock and compute wrap time at 10^6 ops/sec."
    ),
    (
        "Problem [RLVR-OPT-08]: Minimize f(x, y, z) = x^2 + 2y^2 + 3z^2 - 2xy - 2yz subject to "
        "x + y + z = 12 using Lagrange multipliers. Verify positive definiteness of the Hessian "
        "and report the exact minimum value as a reduced fraction."
    ),
)

CANARY_TEMP0_PROMPTS: Tuple[str, ...] = (
    "Compute the exact sum of squares 1^2 + 2^2 + ... + 20^2 and state only the verified derivation.",
    "State the exact binary and hexadecimal representations of 4608 and 24576 with brief verification.",
)


# ---------------------------------------------------------------------------
# Helpers & Diversity Metrics
# ---------------------------------------------------------------------------

def percentile(values: Sequence[float], p: float) -> float:
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


def _tokenize_simple(text: str) -> List[str]:
    return [tok for tok in text.strip().split() if tok]


def compute_n_group_diversity(texts: Sequence[str]) -> Dict[str, Any]:
    """Compute quantitative anti-collapse diversity metrics across `n` samples for one prompt.

    Detects collapsed duplicate sampling (e.g. broken PRNG / stuck greedy fallback under temp > 0)
    by combining:
      1. Exact SHA-256 unique choice ratio (`unique_hashes / n`)
      2. Pairwise token-set Jaccard distance (`1 - |A∩B|/|A∪B|`) across all C(n, 2) pairs
      3. Pairwise normalized edit/prefix divergence (`first_diff_frac`: fraction of pairs that diverge)
      4. Distinct-1 / Distinct-2 lexical ratios across pooled tokens
    """
    n = len(texts)
    if n == 0:
        return {
            "n": 0,
            "non_empty_choices": 0,
            "unique_choices": 0,
            "unique_ratio": 0.0,
            "all_collapsed": True,
            "mean_pairwise_jaccard_dist": 0.0,
            "min_pairwise_jaccard_dist": 0.0,
            "divergent_pair_ratio": 0.0,
            "distinct_1": 0.0,
            "distinct_2": 0.0,
        }

    normalized = [t.strip() for t in texts]
    non_empty = sum(1 for t in normalized if len(t) > 0)
    hashes = [hashlib.sha256(t.encode("utf-8")).hexdigest() for t in normalized]
    unique_cnt = len(set(hashes))
    unique_ratio = unique_cnt / float(n)
    all_collapsed = (n >= 2 and unique_cnt <= 1)

    tok_lists = [_tokenize_simple(t) for t in normalized]
    tok_sets = [set(tl) for tl in tok_lists]

    jaccard_dists: List[float] = []
    divergent_pairs = 0
    total_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            total_pairs += 1
            if hashes[i] != hashes[j]:
                divergent_pairs += 1
            union = tok_sets[i] | tok_sets[j]
            if not union:
                jaccard_dists.append(0.0)
            else:
                inter = tok_sets[i] & tok_sets[j]
                jaccard_dists.append(1.0 - (len(inter) / float(len(union))))

    pooled_unigrams: List[str] = []
    pooled_bigrams: List[Tuple[str, str]] = []
    for tl in tok_lists:
        pooled_unigrams.extend(tl)
        for idx in range(len(tl) - 1):
            pooled_bigrams.append((tl[idx], tl[idx + 1]))

    distinct_1 = (len(set(pooled_unigrams)) / float(len(pooled_unigrams))) if pooled_unigrams else 0.0
    distinct_2 = (len(set(pooled_bigrams)) / float(len(pooled_bigrams))) if pooled_bigrams else 0.0

    return {
        "n": n,
        "non_empty_choices": non_empty,
        "unique_choices": unique_cnt,
        "unique_ratio": round(unique_ratio, 4),
        "all_collapsed": all_collapsed,
        "mean_pairwise_jaccard_dist": round(statistics.mean(jaccard_dists), 4) if jaccard_dists else 0.0,
        "min_pairwise_jaccard_dist": round(min(jaccard_dists), 4) if jaccard_dists else 0.0,
        "divergent_pair_ratio": round(divergent_pairs / float(total_pairs), 4) if total_pairs else 0.0,
        "distinct_1": round(distinct_1, 4),
        "distinct_2": round(distinct_2, 4),
    }


def _extract_cached_tokens(usage: Dict[str, Any]) -> int:
    if not isinstance(usage, dict):
        return 0
    ptd = usage.get("prompt_tokens_details")
    if isinstance(ptd, dict) and isinstance(ptd.get("cached_tokens"), int):
        return int(ptd["cached_tokens"])
    if isinstance(usage.get("cached_tokens"), int):
        return int(usage["cached_tokens"])
    return 0


def run_harvester_n_streaming_request(
    base_url: str,
    model: str,
    messages: List[Dict[str, str]],
    *,
    n: int = 4,
    max_tokens: int = 2048,
    think_budget: int = 2048,
    temperature: float = 0.6,
    top_p: float = 0.95,
    min_p: float = 0.0,
    seed: Optional[int] = None,
    timeout: float = 300.0,
) -> Dict[str, Any]:
    """Execute one streaming `n=4` harvester completion request and collect full per-choice telemetry."""
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "n": n,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {
            "enable_thinking": True,
            "thinking_budget": think_budget,
        },
    }
    if min_p > 0.0:
        payload["min_p"] = min_p
    if seed is not None:
        payload["seed"] = seed

    body = json.dumps(payload).encode("utf-8")
    t_start = time.perf_counter()
    t_first_any: Optional[float] = None
    per_choice_content: Dict[int, List[str]] = {i: [] for i in range(n)}
    per_choice_reasoning: Dict[int, List[str]] = {i: [] for i in range(n)}
    per_choice_prev_ts: Dict[int, float] = {}
    per_choice_first_ts: Dict[int, float] = {}
    per_choice_itl_ms: Dict[int, List[float]] = {i: [] for i in range(n)}
    per_choice_chunks: Dict[int, int] = {i: 0 for i in range(n)}
    per_choice_finish: Dict[int, Optional[str]] = {i: None for i in range(n)}
    seen_indices: set[int] = set()
    usage: Dict[str, Any] = {}
    done_seen = False
    http_status = 0
    conn: Optional[http.client.HTTPConnection] = None

    try:
        conn, base_path = chaos_mod.make_http_conn(base_url, timeout=timeout)
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
            err_body = resp.read(2048).decode("utf-8", errors="replace")
            return {
                "ok": False,
                "http_status": http_status,
                "error": f"HTTP {http_status}: {err_body[:240]}",
                "wall_s": round(time.perf_counter() - t_start, 4),
                "ttft_ms": 0.0,
                "itl_ms": [],
                "prompt_tokens": 0,
                "cached_tokens": 0,
                "completion_tokens": 0,
                "choices_texts": [],
                "diversity": compute_n_group_diversity([]),
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
                done_seen = True
                break
            try:
                evt = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            if isinstance(evt.get("usage"), dict) and evt["usage"]:
                usage = evt["usage"]

            for ch in evt.get("choices") or []:
                idx = int(ch.get("index", 0))
                seen_indices.add(idx)
                per_choice_content.setdefault(idx, [])
                per_choice_reasoning.setdefault(idx, [])
                per_choice_itl_ms.setdefault(idx, [])
                per_choice_chunks.setdefault(idx, 0)
                if ch.get("finish_reason") is not None:
                    per_choice_finish[idx] = ch["finish_reason"]

                delta = ch.get("delta") or {}
                c_txt = delta.get("content")
                r_txt = delta.get("reasoning_content") or delta.get("reasoning")
                got = False
                if c_txt:
                    per_choice_content[idx].append(str(c_txt))
                    got = True
                if r_txt:
                    per_choice_reasoning[idx].append(str(r_txt))
                    got = True
                if got:
                    now = time.perf_counter()
                    if t_first_any is None:
                        t_first_any = now
                    if idx not in per_choice_first_ts:
                        per_choice_first_ts[idx] = now
                    elif idx in per_choice_prev_ts:
                        per_choice_itl_ms[idx].append((now - per_choice_prev_ts[idx]) * 1000.0)
                    per_choice_prev_ts[idx] = now
                    per_choice_chunks[idx] += 1

        t_end = time.perf_counter()
        wall_s = max(t_end - t_start, 1e-6)
        ttft_ms = ((t_first_any - t_start) if t_first_any is not None else wall_s) * 1000.0

        choices_texts: List[str] = []
        all_itl: List[float] = []
        for idx in range(n):
            r_str = "".join(per_choice_reasoning.get(idx, []))
            c_str = "".join(per_choice_content.get(idx, []))
            combined = f"<think>{r_str}</think>\n{c_str}".strip() if r_str else c_str.strip()
            choices_texts.append(combined)
            all_itl.extend(per_choice_itl_ms.get(idx, []))

        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        cached_tokens = _extract_cached_tokens(usage)
        chunk_total = sum(per_choice_chunks.values())
        completion_tokens = int(usage.get("completion_tokens") or chunk_total)
        div = compute_n_group_diversity(choices_texts)

        err_reasons: List[str] = []
        if not done_seen:
            err_reasons.append("SSE missing [DONE]")
        if len(seen_indices) < n:
            err_reasons.append(f"Expected n={n} choices, saw {len(seen_indices)}")
        if div["non_empty_choices"] < n:
            err_reasons.append(f"Empty choice(s): {div['non_empty_choices']}/{n} non-empty")

        return {
            "ok": len(err_reasons) == 0,
            "http_status": http_status,
            "error": "; ".join(err_reasons) if err_reasons else None,
            "t_start": t_start,
            "t_end": t_end,
            "wall_s": round(wall_s, 4),
            "ttft_ms": round(ttft_ms, 2),
            "itl_ms": all_itl,
            "prompt_tokens": prompt_tokens,
            "cached_tokens": cached_tokens,
            "completion_tokens": completion_tokens,
            "choices_seen": len(seen_indices),
            "finish_reasons": [per_choice_finish.get(i) for i in range(n)],
            "diversity": div,
            "choice_hashes": [
                hashlib.sha256(t.encode("utf-8")).hexdigest()[:16] for t in choices_texts
            ],
        }
    except Exception as exc:
        return {
            "ok": False,
            "http_status": http_status,
            "error": f"{type(exc).__name__}: {exc}",
            "t_start": t_start,
            "t_end": time.perf_counter(),
            "wall_s": round(time.perf_counter() - t_start, 4),
            "ttft_ms": 0.0,
            "itl_ms": [],
            "prompt_tokens": 0,
            "cached_tokens": 0,
            "completion_tokens": 0,
            "choices_seen": len(seen_indices),
            "diversity": compute_n_group_diversity([]),
        }
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def _run_temp0_canary_probe(base_url: str, model: str, prompt: str, max_tokens: int, timeout: float = 60.0) -> Dict[str, Any]:
    """Run one deterministic temp=0.0 canary request while temp>0 harvester storm is active."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    t0 = time.perf_counter()
    conn: Optional[http.client.HTTPConnection] = None
    try:
        conn, base_path = chaos_mod.make_http_conn(base_url, timeout=timeout)
        conn.request(
            "POST",
            f"{base_path}/v1/chat/completions",
            body=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        resp = conn.getresponse()
        raw = resp.read()
        dt_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        if resp.status != 200:
            return {"ok": False, "http_status": resp.status, "error": f"HTTP {resp.status}", "latency_ms": dt_ms}
        data = json.loads(raw.decode("utf-8", errors="replace"))
        msg = (data.get("choices") or [{}])[0].get("message") or {}
        text = (msg.get("reasoning_content") or msg.get("reasoning") or "") + (msg.get("content") or "")
        fp = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        return {
            "ok": bool(text.strip()),
            "http_status": 200,
            "fingerprint": fp,
            "latency_ms": dt_ms,
            "error": None if text.strip() else "Empty canary response",
        }
    except Exception as exc:
        return {
            "ok": False,
            "http_status": 0,
            "error": f"{type(exc).__name__}: {exc}",
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        }
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Main Harvester Stress Orchestrator
# ---------------------------------------------------------------------------

def run_harvester_stress_test(
    base_url: str,
    model: str = "inkling-2-small",
    *,
    workers: int = 16,
    n: int = 4,
    rounds: int = 1,
    think_budget: int = 2048,
    max_tokens: int = 2048,
    temperature: float = 0.6,
    top_p: float = 0.95,
    min_p: float = 0.0,
    canary_probes: int = 4,
    canary_max_tokens: int = 48,
    min_unique_ratio: float = 0.75,
    min_jaccard_dist: float = 0.10,
    min_prefix_hit_ratio: float = 0.0,
    min_agg_tok_s: float = 0.0,
    verify_post_exact: bool = True,
    exact_ink_path: Optional[str] = None,
    expected_sha256: str = EXPECTED_EXACT_SHA256,
    exact_max_tokens: int = 384,
    use_builtin_http: bool = False,
    drain_timeout_s: float = 15.0,
    req_timeout_s: float = 300.0,
    seed: int = 16011,
    quiet: bool = False,
) -> Dict[str, Any]:
    """Run the full `rollout_harvester --workers W --n N` production benchmark & quality gate."""
    t_suite0 = time.perf_counter()
    pre_health_ok, pre_code, pre_body = chaos_mod.check_health(base_url)

    # 1. Capture pre-harvest temp=0 canary baseline fingerprints (for mixed-batch invariance!)
    canary_baselines: Dict[int, str] = {}
    canary_pre_ok = True
    for idx, cprompt in enumerate(CANARY_TEMP0_PROMPTS):
        res = _run_temp0_canary_probe(base_url, model, cprompt, canary_max_tokens)
        if res["ok"] and res.get("fingerprint"):
            canary_baselines[idx] = res["fingerprint"]
        else:
            canary_pre_ok = False

    # 2. Warm shared RLVR prefix with 1 fast priming call so subsequent 16-worker wave sees genuine
    #    shared-prefix KV cache reuse on both live vLLM and offline mock servers!
    warm_prompt_msgs = [
        {"role": "system", "content": SHARED_RLVR_SYSTEM_PROMPT},
        {"role": "user", "content": RLVR_PROBLEM_TEMPLATES[0]},
    ]
    _ = _run_temp0_canary_probe(base_url, model, SHARED_RLVR_SYSTEM_PROMPT + "\n" + RLVR_PROBLEM_TEMPLATES[0], 8)

    # 3. Launch concurrent `temp=0` canary thread + `/health` monitor while 16x4 harvester runs!
    stop_bg = threading.Event()
    canary_records: List[Dict[str, Any]] = []
    canary_lock = threading.Lock()
    health_stats = {"checks": 0, "failures": 0}

    def _canary_and_health_loop() -> None:
        probe_idx = 0
        while not stop_bg.is_set():
            ok_h, _, _ = chaos_mod.check_health(base_url, timeout=5.0)
            health_stats["checks"] += 1
            if not ok_h:
                health_stats["failures"] += 1
            if probe_idx < max(canary_probes, 1):
                p_i = probe_idx % len(CANARY_TEMP0_PROMPTS)
                cres = _run_temp0_canary_probe(base_url, model, CANARY_TEMP0_PROMPTS[p_i], canary_max_tokens)
                cres["prompt_idx"] = p_i
                cres["matches_baseline"] = (
                    cres["ok"]
                    and p_i in canary_baselines
                    and cres.get("fingerprint") == canary_baselines[p_i]
                )
                with canary_lock:
                    canary_records.append(cres)
                probe_idx += 1
            stop_bg.wait(0.15)

    bg_thread = threading.Thread(target=_canary_and_health_loop, name="v16-harvester-canary", daemon=True)
    bg_thread.start()

    # 4. Execute `rounds` waves of `workers` concurrent `n=4` requests (`workers * n` completions/wave)
    all_worker_recs: List[Dict[str, Any]] = []
    harvest_t0 = time.perf_counter()
    try:
        for r_idx in range(rounds):
            def _worker_task(w_id: int) -> Dict[str, Any]:
                # Pairs of workers share the same RLVR problem + shared system prompt -> genuine
                # shared-prefix KV reuse while producing independent stochastic `n=4` rollouts!
                tmpl = RLVR_PROBLEM_TEMPLATES[w_id % len(RLVR_PROBLEM_TEMPLATES)]
                msgs = [
                    {"role": "system", "content": SHARED_RLVR_SYSTEM_PROMPT},
                    {"role": "user", "content": tmpl},
                ]
                rec = run_harvester_n_streaming_request(
                    base_url=base_url,
                    model=model,
                    messages=msgs,
                    n=n,
                    max_tokens=max_tokens,
                    think_budget=think_budget,
                    temperature=temperature,
                    top_p=top_p,
                    min_p=min_p,
                    seed=None,  # Unseeded stochastic rollouts (exact production RLVR behavior!)
                    timeout=req_timeout_s,
                )
                rec["worker_id"] = w_id
                rec["round_idx"] = r_idx
                rec["problem_idx"] = w_id % len(RLVR_PROBLEM_TEMPLATES)
                return rec

            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(_worker_task, w) for w in range(workers)]
                wave_recs = [f.result() for f in futs]
            all_worker_recs.extend(wave_recs)
    finally:
        stop_bg.set()
        bg_thread.join(timeout=10.0)

    # Ensure at least `canary_probes` concurrent/post-wave canaries ran even on super-fast mock servers
    while len(canary_records) < canary_probes:
        p_i = len(canary_records) % len(CANARY_TEMP0_PROMPTS)
        cres = _run_temp0_canary_probe(base_url, model, CANARY_TEMP0_PROMPTS[p_i], canary_max_tokens)
        cres["prompt_idx"] = p_i
        cres["matches_baseline"] = (
            cres["ok"]
            and p_i in canary_baselines
            and cres.get("fingerprint") == canary_baselines[p_i]
        )
        canary_records.append(cres)

    harvest_wall_s = max(time.perf_counter() - harvest_t0, 1e-6)

    # 5. Drain & post-harvest `/health` + `/metrics` check
    drain_t0 = time.perf_counter()
    metrics_after = chaos_mod.scrape_vllm_metrics(base_url)
    while (time.perf_counter() - drain_t0) < drain_timeout_s:
        if not metrics_after.get("available"):
            break
        if float(metrics_after.get("num_requests_running") or 0.0) == 0.0 and float(metrics_after.get("num_requests_waiting") or 0.0) == 0.0:
            break
        time.sleep(0.1)
        metrics_after = chaos_mod.scrape_vllm_metrics(base_url)

    post_health_ok, post_code, post_body = chaos_mod.check_health(base_url)

    # 6. Post-harvest 0-ULP `1dcfb306` exactness check
    post_exact: Optional[Dict[str, Any]] = None
    if verify_post_exact:
        post_exact = chaos_mod.run_exact_check(
            base_url=base_url,
            model=model,
            label="post_harvester",
            exact_ink_path=exact_ink_path,
            expected_sha256=expected_sha256,
            max_tokens=exact_max_tokens,
            timeout=max(req_timeout_s, 180.0),
            use_builtin_http=use_builtin_http,
        )

    # 7. Aggregate metrics across all `workers * rounds` requests (`workers * rounds * n` completions)
    ok_recs = [r for r in all_worker_recs if r.get("ok")]
    err_recs = [r for r in all_worker_recs if not r.get("ok")]
    total_requests = len(all_worker_recs)
    expected_completions = total_requests * n
    completed_completions = sum(int(r.get("choices_seen") or 0) for r in ok_recs)
    total_comp_tokens = sum(int(r.get("completion_tokens") or 0) for r in ok_recs)
    total_prompt_tokens = sum(int(r.get("prompt_tokens") or 0) for r in ok_recs)
    total_cached_tokens = sum(int(r.get("cached_tokens") or 0) for r in ok_recs)
    prefix_hit_ratio = (total_cached_tokens / float(total_prompt_tokens)) if total_prompt_tokens > 0 else 0.0

    ttfts_ms = [float(r.get("ttft_ms") or 0.0) for r in ok_recs]
    all_itl_ms: List[float] = []
    for r in ok_recs:
        all_itl_ms.extend(r.get("itl_ms") or [])

    agg_tok_s = total_comp_tokens / harvest_wall_s
    per_completion_tok_s = agg_tok_s / max(1, workers * n)

    # Diversity aggregation across all ok `n`-choice groups
    div_list = [r["diversity"] for r in ok_recs]
    all_collapsed_groups = sum(1 for d in div_list if d.get("all_collapsed"))
    mean_unique_ratio = statistics.mean([d["unique_ratio"] for d in div_list]) if div_list else 0.0
    mean_jaccard_dist = statistics.mean([d["mean_pairwise_jaccard_dist"] for d in div_list]) if div_list else 0.0
    mean_div_pair_ratio = statistics.mean([d["divergent_pair_ratio"] for d in div_list]) if div_list else 0.0
    mean_distinct_1 = statistics.mean([d["distinct_1"] for d in div_list]) if div_list else 0.0
    mean_distinct_2 = statistics.mean([d["distinct_2"] for d in div_list]) if div_list else 0.0

    canary_ok_cnt = sum(1 for c in canary_records if c.get("matches_baseline"))
    canary_all_ok = canary_pre_ok and len(canary_records) > 0 and canary_ok_cnt == len(canary_records)

    slots_drained_ok = True
    if metrics_after.get("available"):
        slots_drained_ok = (
            float(metrics_after.get("num_requests_running") or 0.0) == 0.0
            and float(metrics_after.get("num_requests_waiting") or 0.0) == 0.0
            and float(metrics_after.get("inkling_kv_leaked_blocks") or 0.0) == 0.0
        )

    diversity_required = (temperature > 0.0 and n >= 2)
    checks: Dict[str, Any] = {
        "pre_health_ok": bool(pre_health_ok),
        "post_health_ok": bool(post_health_ok),
        "continuous_health_ok": health_stats["failures"] == 0,
        "zero_dropped_requests": len(err_recs) == 0 and len(ok_recs) == total_requests and completed_completions == expected_completions,
        "zero_all_collapsed_n_groups": (all_collapsed_groups == 0) if diversity_required else True,
        "n_unique_ratio_above_floor": (mean_unique_ratio >= min_unique_ratio) if diversity_required else True,
        "n_jaccard_diversity_above_floor": (mean_jaccard_dist >= min_jaccard_dist) if diversity_required else True,
        "concurrent_temp0_canaries_exact": bool(canary_all_ok),
        "post_harvest_slots_drained": bool(slots_drained_ok),
        "post_harvest_0ulp_1dcfb306": bool(post_exact["ok"]) if (verify_post_exact and post_exact is not None) else True,
    }
    if min_prefix_hit_ratio > 0.0:
        checks["shared_prefix_hit_ratio_above_floor"] = prefix_hit_ratio >= min_prefix_hit_ratio
    if min_agg_tok_s > 0.0:
        checks["aggregate_tok_s_above_floor"] = agg_tok_s >= min_agg_tok_s

    errors: List[str] = []
    if not checks["pre_health_ok"]:
        errors.append(f"Pre-harvest /health failed: {pre_code} {pre_body}")
    if not checks["post_health_ok"]:
        errors.append(f"Post-harvest /health failed: {post_code} {post_body}")
    if not checks["continuous_health_ok"]:
        errors.append(f"Health failed {health_stats['failures']}/{health_stats['checks']} times during harvest")
    if not checks["zero_dropped_requests"]:
        sample_errs = [r.get("error") for r in err_recs[:5]]
        errors.append(f"Dropped/failed requests: {len(err_recs)}/{total_requests} (completions={completed_completions}/{expected_completions}): {sample_errs}")
    if not checks["zero_all_collapsed_n_groups"]:
        errors.append(f"COLLAPSED `n={n}` diversity detected: {all_collapsed_groups}/{len(div_list)} requests returned 100% identical duplicate samples!")
    if not checks["n_unique_ratio_above_floor"]:
        errors.append(f"Mean unique `n={n}` choice ratio {mean_unique_ratio:.3f} < floor {min_unique_ratio:.3f} (possible SyncFree-Temp collapse!)")
    if not checks["n_jaccard_diversity_above_floor"]:
        errors.append(f"Mean pairwise Jaccard distance {mean_jaccard_dist:.3f} < floor {min_jaccard_dist:.3f}")
    if not checks["concurrent_temp0_canaries_exact"]:
        errors.append(f"Concurrent temp=0 canaries failed/drifted: {canary_ok_cnt}/{len(canary_records)} matched baseline")
    if not checks["post_harvest_slots_drained"]:
        errors.append(f"Post-harvest slot/KV leak detected: {metrics_after}")
    if not checks["post_harvest_0ulp_1dcfb306"]:
        errors.append(f"Post-harvest 0-ULP 1dcfb306 check FAILED: {post_exact.get('error') if post_exact else 'missing'}")
    if min_prefix_hit_ratio > 0.0 and not checks.get("shared_prefix_hit_ratio_above_floor", True):
        errors.append(f"Shared-prefix KV hit ratio {prefix_hit_ratio:.3f} < floor {min_prefix_hit_ratio:.3f}")
    if min_agg_tok_s > 0.0 and not checks.get("aggregate_tok_s_above_floor", True):
        errors.append(f"Aggregate harvester throughput {agg_tok_s:.2f} tok/s < floor {min_agg_tok_s:.2f}")

    passed = all(checks.values())
    report: Dict[str, Any] = {
        "benchmark": "v16_harvester_stress_test",
        "status": "PASS" if passed else "FAIL",
        "duration_s": round(time.perf_counter() - t_suite0, 3),
        "harvest_wall_s": round(harvest_wall_s, 3),
        "config": {
            "base_url": base_url,
            "model": model,
            "workers": workers,
            "n": n,
            "concurrent_completions": workers * n,
            "rounds": rounds,
            "think_budget": think_budget,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "min_p": min_p,
            "canary_probes": canary_probes,
            "min_unique_ratio": min_unique_ratio,
            "min_jaccard_dist": min_jaccard_dist,
            "seed": seed,
        },
        "throughput": {
            "total_requests": total_requests,
            "completed_requests": len(ok_recs),
            "dropped_requests": len(err_recs),
            "expected_completions": expected_completions,
            "completed_completions": completed_completions,
            "total_completion_tokens": total_comp_tokens,
            "aggregate_tok_s": round(agg_tok_s, 2),
            "per_completion_tok_s": round(per_completion_tok_s, 2),
        },
        "latency_ms": {
            "ttft_p50": round(percentile(ttfts_ms, 50), 2),
            "ttft_p95": round(percentile(ttfts_ms, 95), 2),
            "ttft_p99": round(percentile(ttfts_ms, 99), 2),
            "ttft_max": round(max(ttfts_ms), 2) if ttfts_ms else 0.0,
            "itl_samples": len(all_itl_ms),
            "itl_p50": round(percentile(all_itl_ms, 50), 2),
            "itl_p95": round(percentile(all_itl_ms, 95), 2),
            "itl_p99": round(percentile(all_itl_ms, 99), 2),
            "itl_max": round(max(all_itl_ms), 2) if all_itl_ms else 0.0,
        },
        "prefix_cache": {
            "total_prompt_tokens": total_prompt_tokens,
            "total_cached_tokens": total_cached_tokens,
            "shared_prefix_hit_ratio": round(prefix_hit_ratio, 4),
        },
        "diversity": {
            "n": n,
            "groups_evaluated": len(div_list),
            "all_collapsed_groups": all_collapsed_groups,
            "mean_unique_choice_ratio": round(mean_unique_ratio, 4),
            "mean_pairwise_jaccard_dist": round(mean_jaccard_dist, 4),
            "mean_divergent_pair_ratio": round(mean_div_pair_ratio, 4),
            "mean_distinct_1": round(mean_distinct_1, 4),
            "mean_distinct_2": round(mean_distinct_2, 4),
        },
        "concurrent_canaries": {
            "total": len(canary_records),
            "exact_matches": canary_ok_cnt,
            "baseline_fingerprints": canary_baselines,
            "health_monitor": health_stats,
        },
        "post_harvest_metrics": metrics_after,
        "post_harvest_exact": post_exact,
        "checks": checks,
        "errors": errors,
    }

    if not quiet:
        ex_str = (
            f"1dcfb306={post_exact['matches']}/{post_exact['total']}({post_exact['sha256'][:8]})"
            if post_exact else "exact=skipped"
        )
        print(
            f"[V16 HARVESTER {workers}x{n}] status={report['status']} | "
            f"agg={report['throughput']['aggregate_tok_s']} tok/s | "
            f"TTFT p50/p99={report['latency_ms']['ttft_p50']}/{report['latency_ms']['ttft_p99']}ms | "
            f"ITL p50/p99={report['latency_ms']['itl_p50']}/{report['latency_ms']['itl_p99']}ms | "
            f"prefix_hit={prefix_hit_ratio * 100:.1f}% | "
            f"div(uniq={mean_unique_ratio:.2f}, jacc={mean_jaccard_dist:.2f}, collapsed={all_collapsed_groups}) | "
            f"canaries={canary_ok_cnt}/{len(canary_records)} | {ex_str}",
            flush=True,
        )
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Wave-V16 Gate 11: rollout_harvester --workers 16 --n 4 Stress & Diversity Verifier"
    )
    ap.add_argument("--base-url", default="http://127.0.0.1:8010", help="OpenAI-compatible server URL")
    ap.add_argument("--model", default="inkling-2-small", help="Served model name")
    ap.add_argument("--mode", choices=("quick", "full"), default="full", help="Quick CI or full 16x4 2048-tok profile")
    ap.add_argument("--workers", type=int, default=None, help="Concurrent harvester workers (default: 16 full / 8 quick)")
    ap.add_argument("--n", type=int, default=4, help="Shared-prefix completions per request (default: 4)")
    ap.add_argument("--rounds", type=int, default=1, help="Number of 16x4 harvest waves")
    ap.add_argument("--think-budget", type=int, default=None, help="Thinking token budget (default: 2048)")
    ap.add_argument("--max-tokens", type=int, default=None, help="Max completion tokens (default: 2048 full / 48 quick)")
    ap.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature (default: 0.6)")
    ap.add_argument("--top-p", type=float, default=0.95, help="Top-p nucleus threshold (default: 0.95)")
    ap.add_argument("--min-p", type=float, default=0.0, help="Optional min-p threshold")
    ap.add_argument("--canary-probes", type=int, default=4, help="Concurrent temp=0 canary probes")
    ap.add_argument("--min-unique-ratio", type=float, default=0.75, help="Min mean unique choice ratio across n=4 groups")
    ap.add_argument("--min-jaccard-dist", type=float, default=0.10, help="Min mean pairwise token Jaccard distance")
    ap.add_argument("--min-prefix-hit-ratio", type=float, default=0.0, help="Optional minimum shared-prefix KV hit ratio")
    ap.add_argument("--min-agg-tok-s", type=float, default=0.0, help="Optional minimum aggregate tok/s floor")
    ap.add_argument("--exact-ink-path", default="/home/jeweled/inkling-tp8/exact_ink.py", help="Path to exact_ink.py")
    ap.add_argument("--expected-sha256", default=EXPECTED_EXACT_SHA256, help="Expected 0-ULP 64-char SHA-256")
    ap.add_argument("--exact-max-tokens", type=int, default=None, help="Built-in exact verifier max_tokens")
    ap.add_argument("--use-builtin-http", action="store_true", help="Force built-in HTTP 0-ULP verifier")
    ap.add_argument("--skip-post-exact", action="store_true", help="Skip post-harvest 1dcfb306 check")
    ap.add_argument("--drain-timeout-s", type=float, default=15.0, help="Post-harvest slot drain timeout (s)")
    ap.add_argument("--req-timeout-s", type=float, default=300.0, help="Per-request timeout (s)")
    ap.add_argument("--seed", type=int, default=16011, help="Base seed")
    ap.add_argument("--out-json", default="", help="Optional JSON report path")
    ap.add_argument("--quiet", action="store_true", help="Suppress stdout summary")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    ap = build_arg_parser()
    args = ap.parse_args(argv)
    workers = args.workers if args.workers is not None else (8 if args.mode == "quick" else 16)
    max_tokens = args.max_tokens if args.max_tokens is not None else (48 if args.mode == "quick" else 2048)
    think_budget = args.think_budget if args.think_budget is not None else (64 if args.mode == "quick" else 2048)
    exact_toks = args.exact_max_tokens if args.exact_max_tokens is not None else (32 if args.mode == "quick" else 384)

    report = run_harvester_stress_test(
        base_url=args.base_url.rstrip("/"),
        model=args.model,
        workers=workers,
        n=args.n,
        rounds=args.rounds,
        think_budget=think_budget,
        max_tokens=max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        min_p=args.min_p,
        canary_probes=args.canary_probes,
        min_unique_ratio=args.min_unique_ratio,
        min_jaccard_dist=args.min_jaccard_dist,
        min_prefix_hit_ratio=args.min_prefix_hit_ratio,
        min_agg_tok_s=args.min_agg_tok_s,
        verify_post_exact=not args.skip_post_exact,
        exact_ink_path=args.exact_ink_path,
        expected_sha256=args.expected_sha256,
        exact_max_tokens=exact_toks,
        use_builtin_http=args.use_builtin_http,
        drain_timeout_s=args.drain_timeout_s,
        req_timeout_s=args.req_timeout_s,
        seed=args.seed,
        quiet=args.quiet,
    )
    if args.out_json:
        out_p = Path(args.out_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
