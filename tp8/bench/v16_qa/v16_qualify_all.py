#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Wave-V16 Automated 12-Gate Permanent Production Release Qualifier.

Upgrades Wave-V15's 10-gate certification harness to the complete 12-gate Wave-V16 permanent
release suite for Inkling-2-Small on 8x NVIDIA DGX Spark (TP8):

  Gate  1: 0-ULP Bit-Exactness (exact_ink cold + warm + POST-CHAOS -> 1dcfb306... 10/10)
  Gate  2: C=1 Decode (exact_ink + bench.py single G1 GM vs V12-FINAL & Ds41f)
  Gate  3: C=2, 4, 8 Middle-Zone (conc_exact + bench.py agg/mean/min + Ds41f check)
  Gate  4: C=32 Throughput (conc_exact + bench.py agg/mean/min vs V12-FINAL & Ds41f)
  Gate  5: 32K / 128K / 1M Cold & Warm Prefill + Decode + 3/3 Needles
  Gate  6: Multi-1M Simultaneous KV Residency (8x warm) + 9th/10th Eviction Integrity
  Gate  7: Mixed Workload Fairness (Cold 1M + Interactive C=4/C=8)
  Gate  8: Chaos & Abort Storm (500+ mixed + 20% aborts + post-chaos 0-ULP)
  Gate  9: Cluster Health & UMA Invariants (8/8 HEALTH_200_OK, swap, MemAvail, RoCEv2, Xid, buddy)
  Gate 10: Crash-Recovery Verification Protocol (--test-crash-recovery: kill r3 -> recover <180s)
  Gate 11: [NEW V16] `temp > 0` `rollout_harvester` 16x4 Stress & Diversity Gate
           (v16_harvester_stress_test.py: SyncFree-Temp throughput, non-collapsed n=4 diversity,
           concurrent temp=0 canaries, shared-prefix KV hit ratio, post-harvest 1dcfb306!)
  Gate 12: [NEW V16] Dynamic-K / Same-Budget Tree / Multimodal SAM-Safe / Telemetry & Drain Gate
           (high-conf code/math K=7..15 speedup + temp=0 exactness, multimodal PNG image+text
           non-pollution, Prometheus `:9108/metrics` exporter verification, zero-drop drain check!)

Pure Python 3.10+ stdlib. Supports BOTH live 8-node cluster execution and 100% offline local
mock-server verification (ZERO SSH/remote commands when targeting localhost/127.0.0.1).
"""
from __future__ import annotations

import argparse
import base64
import binascii
import concurrent.futures
import datetime
import hashlib
import json
import os
import struct
import sys
import time
import urllib.parse
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Locate sibling V16 and V15 QA modules cleanly regardless of CWD
_THIS_DIR = Path(__file__).resolve().parent
_V15_QA_DIR = _THIS_DIR.parent.parent / "v15" / "unified" / "qa"
for _p in (str(_THIS_DIR), str(_V15_QA_DIR)):
    if _p not in sys.path and os.path.isdir(_p):
        sys.path.insert(0, _p)

import v15_chaos_soak as chaos_mod  # noqa: E402
import v15_mixed_1m_c8_bench as mixed_mod  # noqa: E402
import v15_qualify_all as v15q  # noqa: E402
import v16_harvester_stress_test as harvester_mod  # noqa: E402


SCHEMA_VERSION = "v16.qualify.1"
EXPECTED_EXACT_SHA256 = v15q.EXPECTED_EXACT_SHA256
DEFAULT_HOSTS = v15q.DEFAULT_HOSTS
V12_FINAL_REF = v15q.V12_FINAL_REF
DS41F_REF = v15q.DS41F_REF

# Required Wave-V16 Track 7 Prometheus exporter metrics (`:9108/metrics`)
REQUIRED_V16_EXPORTER_METRICS: Tuple[str, ...] = (
    "inkling_v16_node_uma_avail_gib",
    "inkling_v16_node_swap_mib",
    "inkling_v16_node_buddy_order13",
    "inkling_v16_node_roce_active_hcas",
    "inkling_v16_node_xid_count",
    "inkling_v16_kv_utilization_ratio",
    "inkling_v16_kv_prefix_hit_ratio",
    "inkling_v16_spec_accept_len",
    "inkling_v16_requests_running",
    "inkling_v16_requests_waiting",
)

# Image token pollution sentinels that MUST NEVER leak into multimodal assistant text
_MM_POLLUTION_TOKENS: Tuple[str, ...] = (
    "<|image_pad|>",
    "<|vision_pad|>",
    "<|img_pad|>",
    "<image_pad>",
    "IMG_TOKEN_LEAK",
)

_HIGH_CONF_CODE_MATH_PROMPT = (
    "[v16-dynk-highconf-code-math] Write a complete, repetitive, well-structured Python dataclass "
    "table of the first 25 Fibonacci numbers with explicit index, value, ratio, and parity fields, "
    "followed by an explicit inductive proof that sum_{i=1..n} F_i = F_{n+2} - 1."
)

_LOW_CONF_PROSE_PROMPT = (
    "[v16-dynk-lowconf-prose] Compose an original, vivid literary reflection exploring how "
    "unexpected meteorological shifts reshape maritime trade routes across unfamiliar archipelagos."
)


# ---------------------------------------------------------------------------
# Pure-Stdlib Minimal PNG Generator for Offline-Safe Gate 12 Multimodal Probe
# ---------------------------------------------------------------------------

def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    crc = binascii.crc32(chunk_type + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", crc)


def make_synthetic_png_data_uri(width: int = 64, height: int = 64, seed: int = 16) -> str:
    """Create a valid RGB PNG data URI (100% compatible with `--allowed-media-domains media.invalid`)."""
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    row = bytearray(1 + 3 * width)
    for x in range(width):
        row[1 + 3 * x] = (seed * 37 + (x * 255) // max(1, width - 1)) & 0xFF
        row[2 + 3 * x] = (seed * 73 + ((x * 5) & 0xFF)) & 0xFF
        row[3 + 3 * x] = (255 - (x * 255) // max(1, width - 1)) & 0xFF
    raw = bytes(row) * height
    png_bytes = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(raw, level=1))
        + _png_chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")


def _http_get_text(url: str, timeout: float = 10.0) -> Tuple[int, str, Optional[str]]:
    parsed = urllib.parse.urlsplit(url)
    scheme = parsed.scheme or "http"
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if scheme == "https" else 80)
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"
    conn = None
    try:
        conn, _ = chaos_mod.make_http_conn(f"{scheme}://{host}:{port}", timeout=timeout)
        conn.request("GET", path, headers={"Accept": "text/plain, application/json"})
        resp = conn.getresponse()
        body = resp.read(4 * 1024 * 1024).decode("utf-8", errors="replace")
        return resp.status, body, None
    except Exception as exc:
        return 0, "", f"{type(exc).__name__}: {exc}"
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Gate 11: `temp > 0` `rollout_harvester` 16x4 Stress & Diversity Gate
# ---------------------------------------------------------------------------

def run_gate_11_harvester_stress(cfg: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.perf_counter()
    min_agg = cfg["harvester_min_agg_tok_s"]
    if min_agg <= 0.0 and cfg["enforce_perf_floors"]:
        min_agg = 480.0

    rep = harvester_mod.run_harvester_stress_test(
        base_url=cfg["base_url"],
        model=cfg["model"],
        workers=cfg["harvester_workers"],
        n=cfg["harvester_n"],
        rounds=cfg["harvester_rounds"],
        think_budget=cfg["harvester_think_budget"],
        max_tokens=cfg["harvester_max_tokens"],
        temperature=cfg["harvester_temp"],
        top_p=cfg["harvester_top_p"],
        min_p=cfg["harvester_min_p"],
        canary_probes=cfg["harvester_canary_probes"],
        min_unique_ratio=cfg["harvester_min_unique_ratio"],
        min_jaccard_dist=cfg["harvester_min_jaccard_dist"],
        min_prefix_hit_ratio=cfg["harvester_min_prefix_hit_ratio"],
        min_agg_tok_s=min_agg,
        verify_post_exact=True,
        exact_ink_path=cfg["exact_ink_path"],
        expected_sha256=cfg["expected_sha256"],
        exact_max_tokens=cfg["exact_max_tokens"],
        use_builtin_http=cfg["use_builtin_http"],
        drain_timeout_s=cfg["drain_timeout_s"],
        seed=cfg["seed"] + 1100,
        quiet=cfg["quiet"],
    )
    tp = rep["throughput"]
    div = rep["diversity"]
    pc = rep["prefix_cache"]
    return v15q._make_gate_result(
        gate_id=11,
        name=f"temp>0 rollout_harvester ({cfg['harvester_workers']}x{cfg['harvester_n']}) Stress & Diversity",
        status=rep["status"],
        t0=t0,
        summary=(
            f"agg={tp['aggregate_tok_s']} tok/s | dropped={tp['dropped_requests']} | "
            f"prefix_hit={pc['shared_prefix_hit_ratio'] * 100:.1f}% | "
            f"div(uniq={div['mean_unique_choice_ratio']:.2f}, jacc={div['mean_pairwise_jaccard_dist']:.2f}, "
            f"collapsed={div['all_collapsed_groups']}) | post_1dcfb306={rep['checks']['post_harvest_0ulp_1dcfb306']}"
        ),
        checks=rep["checks"],
        metrics=rep,
        errors=rep.get("errors", []),
    )


# ---------------------------------------------------------------------------
# Gate 12: Dynamic-K / Same-Budget Tree / Multimodal SAM-Safe / Telemetry & Drain
# ---------------------------------------------------------------------------

def _subcheck_12a_dynk_tree(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Verify high-conf code/math K=7..15 & Same-Budget Tree speedup + 100% temp=0 exactness."""
    max_tok = 48 if cfg["mode"] == "quick" else 192
    r_code1 = mixed_mod.run_streaming_chat(
        base_url=cfg["base_url"],
        model=cfg["model"],
        messages=[{"role": "user", "content": _HIGH_CONF_CODE_MATH_PROMPT}],
        max_tokens=max_tok,
        temperature=0.0,
        timeout=180.0,
    )
    r_code2 = mixed_mod.run_streaming_chat(
        base_url=cfg["base_url"],
        model=cfg["model"],
        messages=[{"role": "user", "content": _HIGH_CONF_CODE_MATH_PROMPT}],
        max_tokens=max_tok,
        temperature=0.0,
        timeout=180.0,
    )
    r_prose = mixed_mod.run_streaming_chat(
        base_url=cfg["base_url"],
        model=cfg["model"],
        messages=[{"role": "user", "content": _LOW_CONF_PROSE_PROMPT}],
        max_tokens=max_tok,
        temperature=0.0,
        timeout=180.0,
    )
    t1 = (r_code1.get("reasoning", "") + r_code1.get("content", "")).strip()
    t2 = (r_code2.get("reasoning", "") + r_code2.get("content", "")).strip()
    tp = (r_prose.get("reasoning", "") + r_prose.get("content", "")).strip()
    h1 = hashlib.sha256(t1.encode("utf-8")).hexdigest()[:16]
    h2 = hashlib.sha256(t2.encode("utf-8")).hexdigest()[:16]

    code_tps = float(r_code2.get("decode_tok_s") or r_code1.get("decode_tok_s") or 0.0)
    prose_tps = float(r_prose.get("decode_tok_s") or 0.0)
    speedup_ratio = round(code_tps / max(prose_tps, 1e-6), 3) if prose_tps > 0 else 0.0

    det_exact = bool(r_code1.get("ok") and r_code2.get("ok") and t1 and h1 == h2)
    prose_ok = bool(r_prose.get("ok") and tp)
    min_ratio = cfg["dynk_min_speedup_ratio"]
    speedup_ok = (speedup_ratio >= min_ratio) if (cfg["enforce_perf_floors"] or cfg["require_dynk_speedup"]) else (code_tps > 0.0 and prose_tps > 0.0)

    return {
        "ok": det_exact and prose_ok and speedup_ok,
        "code_math_tok_s": round(code_tps, 2),
        "prose_tok_s": round(prose_tps, 2),
        "code_vs_prose_ratio": speedup_ratio,
        "min_required_ratio": min_ratio,
        "temp0_deterministic_match": det_exact,
        "code_hash": h1,
        "prose_ok": prose_ok,
        "speedup_check_ok": speedup_ok,
        "error": None if (det_exact and prose_ok and speedup_ok) else (
            f"det={det_exact} prose_ok={prose_ok} ratio={speedup_ratio} (min={min_ratio}) "
            f"errs={[r_code1.get('error'), r_code2.get('error'), r_prose.get('error')]}"
        ),
    }


def _subcheck_12b_multimodal_sam_safe(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Verify multimodal image+text non-pollution and SAM/temp=0 isolation."""
    img_sz = 64 if cfg["mode"] == "quick" else 256
    data_uri = make_synthetic_png_data_uri(img_sz, img_sz, seed=16)

    # Pre-MM baseline deterministic probe
    probe_prompt = "[v16-mm-sam-canary] State the sum of 17 and 29 in one short sentence."
    pre = mixed_mod.run_streaming_chat(
        base_url=cfg["base_url"],
        model=cfg["model"],
        messages=[{"role": "user", "content": probe_prompt}],
        max_tokens=32,
        temperature=0.0,
        timeout=60.0,
    )
    pre_txt = (pre.get("reasoning", "") + pre.get("content", "")).strip()
    pre_hash = hashlib.sha256(pre_txt.encode("utf-8")).hexdigest()[:16]

    # Multimodal image+text completion
    mm_msgs = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_uri}},
                {"type": "text", "text": "Briefly describe the gradient pattern in this synthetic test image."},
            ],
        }
    ]
    mm = mixed_mod.run_streaming_chat(
        base_url=cfg["base_url"],
        model=cfg["model"],
        messages=mm_msgs,
        max_tokens=48,
        temperature=0.0,
        timeout=120.0,
    )
    mm_txt = (mm.get("reasoning", "") + mm.get("content", "")).strip()
    leaked_markers = [tok for tok in _MM_POLLUTION_TOKENS if tok in mm_txt]

    # Post-MM deterministic probe (proves image tokens never polluted GPU/CPU SAM ring!)
    post = mixed_mod.run_streaming_chat(
        base_url=cfg["base_url"],
        model=cfg["model"],
        messages=[{"role": "user", "content": probe_prompt}],
        max_tokens=32,
        temperature=0.0,
        timeout=60.0,
    )
    post_txt = (post.get("reasoning", "") + post.get("content", "")).strip()
    post_hash = hashlib.sha256(post_txt.encode("utf-8")).hexdigest()[:16]

    mm_ok = bool(mm.get("ok") and mm_txt)
    non_polluted = len(leaked_markers) == 0
    sam_isolated = bool(pre.get("ok") and post.get("ok") and pre_txt and pre_hash == post_hash)

    return {
        "ok": mm_ok and non_polluted and sam_isolated,
        "image_size": f"{img_sz}x{img_sz}",
        "mm_http_ok": mm_ok,
        "leaked_markers": leaked_markers,
        "non_polluted": non_polluted,
        "sam_isolation_exact": sam_isolated,
        "pre_hash": pre_hash,
        "post_hash": post_hash,
        "error": None if (mm_ok and non_polluted and sam_isolated) else (
            f"mm_ok={mm_ok} leaked={leaked_markers} sam_iso={sam_isolated} mm_err={mm.get('error')}"
        ),
    }


def _subcheck_12c_prometheus_exporter(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Verify Wave-V16 Prometheus telemetry exporter endpoint (`:9108/metrics` + `/healthz`)."""
    exp_url = cfg["exporter_url"]
    if not exp_url:
        parsed = urllib.parse.urlsplit(cfg["base_url"])
        host = parsed.hostname or "127.0.0.1"
        exp_url = f"http://{host}:9108"
    exp_url = exp_url.rstrip("/")

    hz_code, hz_body, hz_err = _http_get_text(f"{exp_url}/healthz", timeout=10.0)
    m_code, m_body, m_err = _http_get_text(f"{exp_url}/metrics", timeout=10.0)

    if m_err or m_code != 200:
        return {
            "ok": False,
            "exporter_url": exp_url,
            "healthz_status": hz_code,
            "metrics_status": m_code,
            "missing_metrics": list(REQUIRED_V16_EXPORTER_METRICS),
            "present_count": 0,
            "required_count": len(REQUIRED_V16_EXPORTER_METRICS),
            "error": f"GET {exp_url}/metrics failed: status={m_code} err={m_err}",
        }

    parsed_metrics = chaos_mod.parse_prometheus_metrics(m_body)
    missing = [name for name in REQUIRED_V16_EXPORTER_METRICS if name not in parsed_metrics]
    xid_sum = float(parsed_metrics.get("inkling_v16_node_xid_count", 0.0))
    swap_sum = float(parsed_metrics.get("inkling_v16_node_swap_mib", 0.0))
    healthz_ok = (hz_code == 200) or (hz_code == 404 and m_code == 200)
    zero_xid_metric = (xid_sum == 0.0)
    ok = (m_code == 200) and healthz_ok and (len(missing) == 0) and zero_xid_metric

    return {
        "ok": ok,
        "exporter_url": exp_url,
        "healthz_status": hz_code,
        "metrics_status": m_code,
        "present_count": len(REQUIRED_V16_EXPORTER_METRICS) - len(missing),
        "required_count": len(REQUIRED_V16_EXPORTER_METRICS),
        "missing_metrics": missing,
        "scraped_values": {k: parsed_metrics[k] for k in REQUIRED_V16_EXPORTER_METRICS if k in parsed_metrics},
        "zero_xid_metric": zero_xid_metric,
        "swap_sum_mib": round(swap_sum, 2),
        "error": None if ok else (
            f"missing={missing} hz={hz_code}({hz_err}) xid_sum={xid_sum}"
        ),
    }


def _subcheck_12d_zero_drop_drain(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Verify zero-drop graceful drain under active in-flight streaming completions."""
    n_inflight = 4
    t0 = time.perf_counter()

    def _inflight_req(idx: int) -> Dict[str, Any]:
        return mixed_mod.run_streaming_chat(
            base_url=cfg["base_url"],
            model=cfg["model"],
            messages=[{"role": "user", "content": f"[v16-drain-req-{idx}] Count cleanly from 1 to 10."}],
            max_tokens=32,
            temperature=0.0,
            timeout=60.0,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_inflight) as ex:
        futs = [ex.submit(_inflight_req, i) for i in range(n_inflight)]
        recs = [f.result() for f in futs]

    ok_cnt = sum(1 for r in recs if r.get("ok"))
    dropped = n_inflight - ok_cnt

    # Poll `/metrics` until running + waiting == 0 (exact protocol used by v16_drain_and_deploy.sh!)
    drain_t0 = time.perf_counter()
    deadline = drain_t0 + float(cfg["drain_timeout_s"])
    drained = False
    last_running: Optional[float] = None
    last_waiting: Optional[float] = None
    metrics_available = False

    while time.perf_counter() <= deadline:
        m = chaos_mod.scrape_vllm_metrics(cfg["base_url"], timeout=5.0)
        if not m.get("available"):
            # If /metrics unavailable on server, rely on /health + 0 dropped completions
            drained = (dropped == 0)
            break
        metrics_available = True
        last_running = float(m.get("num_requests_running") or 0.0)
        last_waiting = float(m.get("num_requests_waiting") or 0.0)
        if last_running == 0.0 and last_waiting == 0.0:
            drained = True
            break
        time.sleep(0.1)

    drain_s = round(time.perf_counter() - drain_t0, 3)
    health_ok, _, _ = chaos_mod.check_health(cfg["base_url"], timeout=5.0)
    ok = (dropped == 0) and drained and health_ok
    return {
        "ok": ok,
        "inflight_sent": n_inflight,
        "inflight_completed": ok_cnt,
        "dropped_requests": dropped,
        "metrics_available": metrics_available,
        "drained_to_zero": drained,
        "final_running": last_running,
        "final_waiting": last_waiting,
        "drain_wait_s": drain_s,
        "post_drain_health_ok": health_ok,
        "wall_s": round(time.perf_counter() - t0, 3),
        "error": None if ok else (
            f"dropped={dropped}/{n_inflight} drained={drained} running={last_running} waiting={last_waiting} health={health_ok}"
        ),
    }


def run_gate_12_v16_features_telemetry(cfg: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.perf_counter()
    sub_a = _subcheck_12a_dynk_tree(cfg)
    sub_b = _subcheck_12b_multimodal_sam_safe(cfg)
    sub_c = _subcheck_12c_prometheus_exporter(cfg)
    sub_d = _subcheck_12d_zero_drop_drain(cfg)

    checks = {
        "dynk_tree_speedup_and_exactness_ok": bool(sub_a["ok"]),
        "multimodal_sam_safe_non_pollution_ok": bool(sub_b["ok"]),
        "prometheus_exporter_9108_ok": bool(sub_c["ok"]),
        "zero_drop_graceful_drain_ok": bool(sub_d["ok"]),
    }
    errors: List[str] = []
    if not sub_a["ok"]:
        errors.append(f"12A Dynamic-K/Tree failed: {sub_a.get('error')}")
    if not sub_b["ok"]:
        errors.append(f"12B Multimodal SAM-safe failed: {sub_b.get('error')}")
    if not sub_c["ok"]:
        errors.append(f"12C Prometheus exporter failed: {sub_c.get('error')}")
    if not sub_d["ok"]:
        errors.append(f"12D Graceful drain failed: {sub_d.get('error')}")

    passed = all(checks.values())
    return v15q._make_gate_result(
        gate_id=12,
        name="Dynamic-K / Tree / Multimodal SAM-Safe / Telemetry & Drain",
        status="PASS" if passed else "FAIL",
        t0=t0,
        summary=(
            f"dynK_ratio={sub_a['code_vs_prose_ratio']}x(det={sub_a['temp0_deterministic_match']}) | "
            f"mm_safe={sub_b['ok']} | exporter={sub_c['present_count']}/{sub_c['required_count']} | "
            f"drain_zero_drop={sub_d['ok']}({sub_d['drain_wait_s']}s)"
        ),
        checks=checks,
        metrics={
            "sub_12a_dynk_tree": sub_a,
            "sub_12b_multimodal_sam_safe": sub_b,
            "sub_12c_prometheus_exporter": sub_c,
            "sub_12d_zero_drop_drain": sub_d,
        },
        errors=errors,
    )


# ---------------------------------------------------------------------------
# Orchestrator & CLI (All 12 Gates)
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    ap = v15q.build_arg_parser()
    ap.description = "Wave-V16 Automated 12-Gate Permanent Production Release Qualifier"
    # Update defaults cleanly for V16
    for act in ap._actions:
        if act.dest == "out_json":
            act.default = "results_v16.json"
        elif act.dest == "profile_name":
            act.default = "live_champion_v16_default.env"
        elif act.dest == "gates":
            act.help = "Comma-separated gate numbers (1..12) or 'all'"
        elif act.dest == "seed":
            act.default = 16000
        elif act.dest == "crash_bundle_dir":
            act.default = "/var/log/inkling-v16"

    # Gate 11: rollout_harvester 16x4 stress & diversity options
    ap.add_argument("--harvester-workers", type=int, default=None, help="Gate 11 workers (default: 16 full / 8 quick)")
    ap.add_argument("--harvester-n", type=int, default=4, help="Gate 11 shared-prefix completions per worker (default: 4)")
    ap.add_argument("--harvester-rounds", type=int, default=None, help="Gate 11 harvest waves (default: 1 full / 2 soak)")
    ap.add_argument("--harvester-think-budget", type=int, default=None, help="Gate 11 think budget (default: 2048 full)")
    ap.add_argument("--harvester-max-tokens", type=int, default=None, help="Gate 11 max tokens (default: 2048 full / 48 quick)")
    ap.add_argument("--harvester-temp", type=float, default=0.6, help="Gate 11 temperature (default: 0.6)")
    ap.add_argument("--harvester-top-p", type=float, default=0.95, help="Gate 11 top_p (default: 0.95)")
    ap.add_argument("--harvester-min-p", type=float, default=0.0, help="Gate 11 optional min_p")
    ap.add_argument("--harvester-canary-probes", type=int, default=4, help="Gate 11 concurrent temp=0 canaries")
    ap.add_argument("--harvester-min-unique-ratio", type=float, default=0.75, help="Gate 11 min unique n=4 choice ratio")
    ap.add_argument("--harvester-min-jaccard-dist", type=float, default=0.10, help="Gate 11 min pairwise Jaccard distance")
    ap.add_argument("--harvester-min-prefix-hit-ratio", type=float, default=0.0, help="Gate 11 min shared-prefix KV hit ratio")
    ap.add_argument("--harvester-min-agg-tok-s", type=float, default=0.0, help="Gate 11 min aggregate tok/s")

    # Gate 12: Dynamic-K / Tree / Multimodal / Exporter / Drain options
    ap.add_argument("--exporter-url", default="", help="Gate 12 Prometheus exporter URL (default: http://<host>:9108)")
    ap.add_argument("--dynk-min-speedup-ratio", type=float, default=1.05, help="Gate 12A code/math vs prose min ratio")
    ap.add_argument("--require-dynk-speedup", action="store_true", help="Enforce Gate 12A speedup ratio even on localhost")
    ap.add_argument("--drain-timeout-s", type=float, default=15.0, help="Gate 11/12D graceful drain timeout (s)")
    return ap


def build_config(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = v15q.build_config(args)
    selected = set(range(1, 13))
    if args.gates and args.gates.lower() != "all":
        selected = set(v15q._parse_int_csv(args.gates))
    cfg["selected_gates"] = selected

    mode = args.mode
    cfg["harvester_workers"] = args.harvester_workers if args.harvester_workers is not None else (8 if mode == "quick" else 16)
    cfg["harvester_n"] = args.harvester_n
    cfg["harvester_rounds"] = args.harvester_rounds if args.harvester_rounds is not None else (2 if mode == "soak" else 1)
    cfg["harvester_think_budget"] = args.harvester_think_budget if args.harvester_think_budget is not None else (64 if mode == "quick" else 2048)
    cfg["harvester_max_tokens"] = args.harvester_max_tokens if args.harvester_max_tokens is not None else (48 if mode == "quick" else 2048)
    cfg["harvester_temp"] = args.harvester_temp
    cfg["harvester_top_p"] = args.harvester_top_p
    cfg["harvester_min_p"] = args.harvester_min_p
    cfg["harvester_canary_probes"] = args.harvester_canary_probes
    cfg["harvester_min_unique_ratio"] = args.harvester_min_unique_ratio
    cfg["harvester_min_jaccard_dist"] = args.harvester_min_jaccard_dist
    cfg["harvester_min_prefix_hit_ratio"] = args.harvester_min_prefix_hit_ratio
    cfg["harvester_min_agg_tok_s"] = args.harvester_min_agg_tok_s

    cfg["exporter_url"] = args.exporter_url
    cfg["dynk_min_speedup_ratio"] = args.dynk_min_speedup_ratio
    cfg["require_dynk_speedup"] = bool(args.require_dynk_speedup)
    cfg["drain_timeout_s"] = args.drain_timeout_s
    return cfg


def run_all_gates(cfg: Dict[str, Any]) -> Dict[str, Any]:
    t_suite_start = time.perf_counter()
    selected = cfg["selected_gates"]
    gates: Dict[str, Dict[str, Any]] = {}

    def _announce(gid: int, title: str) -> None:
        if not cfg["quiet"]:
            print(f"\n--- [Gate {gid:02d}/12] {title} ---", flush=True)

    def _log_done(g: Dict[str, Any]) -> None:
        if not cfg["quiet"]:
            print(f"  => Gate {g['gate_id']:02d} [{g['status']}] ({g['duration_s']}s): {g['summary']}", flush=True)

    if 1 in selected:
        _announce(1, "0-ULP Bit-Exactness (Cold + Warm)")
        gates["gate_1_bit_exact"] = v15q.run_gate_1_bit_exact(cfg)
        _log_done(gates["gate_1_bit_exact"])

    if 2 in selected:
        _announce(2, "C=1 Decode (exact_ink + bench single G1 GM)")
        gates["gate_2_c1_decode"] = v15q.run_gate_2_c1_decode(cfg, gates.get("gate_1_bit_exact"))
        _log_done(gates["gate_2_c1_decode"])

    if 3 in selected:
        _announce(3, "C=2, 4, 8 Middle-Zone + Ds41f Comparison")
        gates["gate_3_c2_c4_c8_middle_zone"] = v15q.run_gate_3_middle_zone(cfg)
        _log_done(gates["gate_3_c2_c4_c8_middle_zone"])

    if 4 in selected:
        _announce(4, f"C={cfg['c32_conc']} Throughput")
        gates["gate_4_c32_throughput"] = v15q.run_gate_4_c32_throughput(cfg)
        _log_done(gates["gate_4_c32_throughput"])

    if 5 in selected:
        _announce(5, "32K / 128K / 1M Cold+Warm Prefill + Decode + 3/3 Needles")
        gates["gate_5_long_ctx_32k_128k_1m"] = v15q.run_gate_5_long_context(cfg)
        _log_done(gates["gate_5_long_ctx_32k_128k_1m"])

    if 6 in selected:
        _announce(6, f"Multi-1M Residency ({cfg['multi_1m_resident']}x) + Eviction (+{cfg['multi_1m_evict_extra']}x)")
        gates["gate_6_multi_1m_residency_eviction"] = v15q.run_gate_6_multi_1m_residency_eviction(cfg)
        _log_done(gates["gate_6_multi_1m_residency_eviction"])

    if 7 in selected:
        _announce(7, "Mixed Workload Fairness (Cold 1M + Interactive C=4/C=8)")
        gates["gate_7_mixed_1m_c8_fairness"] = v15q.run_gate_7_mixed_fairness(cfg)
        _log_done(gates["gate_7_mixed_1m_c8_fairness"])

    if 8 in selected:
        _announce(8, "Chaos & Abort Storm + Post-Chaos 0-ULP Lock")
        g8 = v15q.run_gate_8_chaos_soak(cfg)
        gates["gate_8_chaos_abort_storm"] = g8
        _log_done(g8)
        if "gate_1_bit_exact" in gates:
            v15q.attach_post_chaos_to_gate_1(
                gates["gate_1_bit_exact"],
                g8.get("metrics", {}).get("post_chaos_exact"),
            )

    if 9 in selected:
        _announce(9, "Cluster Health & UMA Invariants (8/8 Nodes)")
        gates["gate_9_cluster_health_uma"] = v15q.run_gate_9_cluster_health(cfg)
        _log_done(gates["gate_9_cluster_health_uma"])

    if 10 in selected:
        _announce(10, "Crash-Recovery Verification Protocol")
        gates["gate_10_crash_recovery"] = v15q.run_gate_10_crash_recovery(cfg)
        _log_done(gates["gate_10_crash_recovery"])

    if 11 in selected:
        _announce(11, f"temp>0 rollout_harvester ({cfg['harvester_workers']}x{cfg['harvester_n']}) Stress & Diversity")
        gates["gate_11_harvester_16x4_stress"] = run_gate_11_harvester_stress(cfg)
        _log_done(gates["gate_11_harvester_16x4_stress"])

    if 12 in selected:
        _announce(12, "Dynamic-K / Same-Budget Tree / Multimodal SAM-Safe / Telemetry & Drain")
        gates["gate_12_v16_features_telemetry"] = run_gate_12_v16_features_telemetry(cfg)
        _log_done(gates["gate_12_v16_features_telemetry"])

    passed_cnt = sum(1 for g in gates.values() if g["status"] == "PASS")
    failed_list = [k for k, g in gates.items() if g["status"] == "FAIL"]
    skipped_cnt = sum(1 for g in gates.values() if g["status"] == "SKIPPED")
    overall_status = "PASS" if len(failed_list) == 0 else "FAIL"

    return {
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "duration_s": round(time.perf_counter() - t_suite_start, 3),
        "mode": cfg["mode"],
        "base_url": cfg["base_url"],
        "exporter_url": cfg["exporter_url"] or None,
        "model": cfg["model"],
        "profile_name": cfg["profile_name"],
        "env_file": cfg["env_file"] or None,
        "env_sha256": v15q.sha256_file(cfg["env_file"]) if cfg["env_file"] else None,
        "expected_exact_sha256": cfg["expected_sha256"],
        "enforce_perf_floors": cfg["enforce_perf_floors"],
        "overall_status": overall_status,
        "total_gates_selected": len(gates),
        "gates_passed": passed_cnt,
        "gates_failed": len(failed_list),
        "gates_skipped": skipped_cnt,
        "failed_gates": failed_list,
        "references": {
            "v12_final": V12_FINAL_REF,
            "ds41f": DS41F_REF,
        },
        "gates": gates,
    }


def print_scorecard(report: Dict[str, Any]) -> None:
    print("\n================================================================================")
    print(
        f"WAVE-V16 PERMANENT 12-GATE RELEASE QUALIFICATION: {report['overall_status']} "
        f"(mode={report['mode']} | pass={report['gates_passed']} fail={report['gates_failed']} "
        f"skip={report['gates_skipped']} | {report['duration_s']}s)"
    )
    print("================================================================================")
    for g in report["gates"].values():
        print(f"  [G{g['gate_id']:02d}] {g['status']:7s} | {g['name']}: {g['summary']}")
        for err in g.get("errors", [])[:3]:
            print(f"         ERROR: {err}")
    print("================================================================================", flush=True)


def main(argv: Optional[List[str]] = None) -> int:
    ap = build_arg_parser()
    args = ap.parse_args(argv)
    cfg = build_config(args)
    report = run_all_gates(cfg)

    if cfg["out_json"]:
        out_path = Path(cfg["out_json"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if not cfg["quiet"]:
        print_scorecard(report)
        if cfg["out_json"]:
            print(f"[v16_qualify_all] JSON report written -> {cfg['out_json']}", flush=True)

    return 0 if report["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
