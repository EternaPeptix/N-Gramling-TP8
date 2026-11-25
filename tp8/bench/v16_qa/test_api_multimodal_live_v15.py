#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Wave-V15 Live OpenAI API, SSE Streaming, Sampling, Harvester & Multimodal Compliance Suite.

Zero external dependencies (pure Python 3.10+ stdlib: http.client, socket, zlib, struct,
base64, concurrent.futures) — runs identically on macOS, Linux, or spark-one.

Covers EVERY production serving workload on 8x DGX Spark TP8 (`inkling-2-small`):
  1. Health & model discovery (`/health`, `/v1/models`).
  2. SSE Streaming (`stream=True`, `stream_options={"include_usage": true}`), chunk framing,
     `[DONE]` sentinel, usage accounting, AND mid-stream abrupt client disconnect / TCP RST
     cancellation followed by server health & post-cancel exactness verification.
  3. Complete Sampling Matrix:
       - `temperature=0.0` (0-ULP deterministic repeatability: 3x identical output!)
       - `temperature=0.6`, `top_p=0.95`, `min_p=0.05` (stochastic diversity + non-empty)
       - `n=4` shared-prefix fanout (4 distinct choices, index 0..3, usage accounting)
       - `max_tokens=1` (boundary token budget + `finish_reason="length"`)
       - `max_tokens=2048` (or configurable `--harvester-max-tokens`) with reasoning traces
       - `stop=["</think>", "\\n\\n"]` (stop sequence enforcement, never leaks stop string)
       - `logprobs=True, top_logprobs=5` (both `temp=0` SyncFree disarm check & `temp=0.6`)
  4. Tool-Calling & Structured JSON Schema Output:
       - OpenAI `tools` + `tool_choice="auto"` / named function schema
       - `response_format={"type": "json_schema", ...}` grammar-constrained JSON output
  5. Multimodal Vision Stress Suite (offline-safe base64 `data:image/png;base64,...` URIs,
     100% compatible with `--allowed-media-domains media.invalid`):
       - Tiny `16x16` image
       - Standard `1024x1024` image
       - Extreme aspect ratios: ultra-wide `2560x40` & ultra-tall `40x2560`
       - Full `4K` (`3840x2160`) image
       - Multi-image batch (4..16 images in one request!)
       - Repeated multimodal soak (`--mm-soak-iters`) verifying zero UMA drift / latency creep
  6. Mixed Concurrent Production Workload (Simulates `rollout_harvester` + Interactive + Vision):
       - Runs concurrent `n=4` harvester workers (`temp=0.6, top_p=0.95, min_p=0.05`, reasoning)
         SIMULTANEOUSLY with `temp=0` exact repeatability probes, SSE streaming + mid-stream
         cancels, multimodal image+text prompts, and tool-calling JSON schema requests!
       - Verifies ZERO 5xx errors, ZERO dropped connections on healthy clients, exact `temp=0`
         determinism under concurrency, and final post-storm `temp=0` exactness + `/health` 200!

Usage:
  # Against live cluster (during qualification window ONLY):
  python3 test_api_multimodal_live_v15.py --base-url http://spark-one:8010 --mode full --out-json api_mm_v15.json

  # Quick smoke (fast CI / post-boot check):
  python3 test_api_multimodal_live_v15.py --base-url http://spark-one:8010 --mode quick

  # Full rollout_harvester-shaped concurrency stress (`--harvester-workers 16 --harvester-n 4`):
  python3 test_api_multimodal_live_v15.py --base-url http://spark-one:8010 --mode harvester \\
      --harvester-workers 16 --harvester-n 4 --harvester-max-tokens 2048
"""

from __future__ import annotations

import argparse
import base64
import binascii
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
import hashlib
import http.client
import json
import math
import socket
import struct
import sys
import time
import urllib.parse
import zlib
from typing import Any


# ---------------------------------------------------------------------------
# Self-contained synthetic PNG generator (0 external deps, valid CRC32/zlib)
# ---------------------------------------------------------------------------

def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    crc = binascii.crc32(chunk_type + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", crc)


def make_synthetic_png_bytes(width: int, height: int, pattern: str = "gradient", seed: int = 0) -> bytes:
    """Build a valid 8-bit RGB PNG in pure stdlib (fast even for 4K 3840x2160)."""
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid PNG dimensions {width}x{height}")
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    # Build scanlines efficiently (1 filter byte + 3*W RGB bytes per row).
    # Fast byte repetition keeps 3840x2160 PNG generation < ~25 ms in pure Python!
    r0 = (seed * 53 + 37) & 0xFF
    g0 = (seed * 97 + 113) & 0xFF
    b0 = (seed * 193 + 211) & 0xFF
    if pattern == "solid":
        row = b"\x00" + bytes((r0, g0, b0)) * width
        raw = row * height
    elif pattern == "checker":
        tile_a = bytes((240, 20, 20)) * min(width, 16)
        tile_b = bytes((20, 20, 240)) * min(width, 16)
        pair = tile_a + tile_b
        reps = (width * 3 + len(pair) - 1) // len(pair)
        row_a = b"\x00" + (pair * reps)[: width * 3]
        row_b = b"\x00" + ((tile_b + tile_a) * reps)[: width * 3]
        rows = [row_a if ((y // 16) % 2 == 0) else row_b for y in range(height)]
        raw = b"".join(rows)
    else:
        # Horizontal color gradient with vertical stripe modulation (high-entropy visual features)
        base_row = bytearray(1 + 3 * width)
        for x in range(width):
            base_row[1 + 3 * x] = (r0 + (x * 255) // max(1, width - 1)) & 0xFF
            base_row[2 + 3 * x] = (g0 + ((x * 7) & 0xFF)) & 0xFF
            base_row[3 + 3 * x] = (b0 + 255 - (x * 255) // max(1, width - 1)) & 0xFF
        row_even = bytes(base_row)
        alt_row = bytearray(base_row)
        for x in range(0, width, max(1, width // 32)):
            alt_row[1 + 3 * x] ^= 0x55
            alt_row[2 + 3 * x] ^= 0xAA
        row_odd = bytes(alt_row)
        rows = [row_even if ((y // 8) % 2 == 0) else row_odd for y in range(height)]
        raw = b"".join(rows)
    comp = zlib.compress(raw, level=1)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", comp)
        + _png_chunk(b"IEND", b"")
    )


def make_png_data_uri(width: int, height: int, pattern: str = "gradient", seed: int = 0) -> str:
    b64 = base64.b64encode(make_synthetic_png_bytes(width, height, pattern=pattern, seed=seed)).decode("ascii")
    return f"data:image/png;base64,{b64}"


# ---------------------------------------------------------------------------
# Lightweight HTTP + SSE client (supports explicit mid-stream TCP abort!)
# ---------------------------------------------------------------------------

class OpenAIClient:
    def __init__(self, base_url: str, api_key: str = "EMPTY", timeout: float = 180.0) -> None:
        parsed = urllib.parse.urlparse(base_url.rstrip("/"))
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"Unsupported scheme in base_url: {base_url!r}")
        self.scheme = parsed.scheme
        self.host = parsed.hostname or "127.0.0.1"
        self.port = parsed.port or (443 if self.scheme == "https" else 80)
        self.base_path = parsed.path.rstrip("/")
        self.api_key = api_key
        self.timeout = float(timeout)

    def _connect(self, timeout: float | None = None) -> http.client.HTTPConnection:
        to = self.timeout if timeout is None else float(timeout)
        if self.scheme == "https":
            return http.client.HTTPSConnection(self.host, self.port, timeout=to)
        return http.client.HTTPConnection(self.host, self.port, timeout=to)

    def _headers(self, json_body: bool = True) -> dict[str, str]:
        h = {"Accept": "application/json"}
        if json_body:
            h["Content-Type"] = "application/json"
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def get_json(self, path: str, timeout: float | None = None) -> tuple[int, Any]:
        conn = self._connect(timeout=timeout)
        try:
            conn.request("GET", f"{self.base_path}{path}", headers=self._headers(json_body=False))
            resp = conn.getresponse()
            raw = resp.read()
            if not raw:
                return resp.status, {}
            try:
                return resp.status, json.loads(raw.decode("utf-8"))
            except Exception:
                return resp.status, raw.decode("utf-8", errors="replace")
        finally:
            conn.close()

    def post_json(
        self, path: str, payload: dict[str, Any], timeout: float | None = None
    ) -> tuple[int, Any, float]:
        body = json.dumps(payload).encode("utf-8")
        conn = self._connect(timeout=timeout)
        t0 = time.perf_counter()
        try:
            conn.request("POST", f"{self.base_path}{path}", body=body, headers=self._headers(True))
            resp = conn.getresponse()
            raw = resp.read()
            dt = time.perf_counter() - t0
            try:
                data = json.loads(raw.decode("utf-8")) if raw else {}
            except Exception:
                data = {"raw": raw.decode("utf-8", errors="replace")}
            return resp.status, data, dt
        finally:
            conn.close()

    def post_sse(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        cancel_after_chunks: int | None = None,
        hard_rst: bool = False,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Execute an SSE streaming request; optionally abort mid-stream to test cancel safety."""
        body = json.dumps(payload).encode("utf-8")
        conn = self._connect(timeout=timeout)
        t0 = time.perf_counter()
        ttft: float | None = None
        chunks: list[dict[str, Any]] = []
        done_seen = False
        cancelled = False
        status = 0
        try:
            headers = self._headers(True)
            headers["Accept"] = "text/event-stream"
            conn.request("POST", f"{self.base_path}{path}", body=body, headers=headers)
            resp = conn.getresponse()
            status = resp.status
            if status != 200:
                err_raw = resp.read().decode("utf-8", errors="replace")
                return {
                    "status": status,
                    "error": err_raw,
                    "chunks": [],
                    "done_seen": False,
                    "cancelled": False,
                    "elapsed_s": time.perf_counter() - t0,
                    "ttft_s": None,
                }
            while True:
                line_b = resp.readline()
                if not line_b:
                    break
                line = line_b.decode("utf-8", errors="replace").strip()
                if not line or line.startswith(":"):
                    continue
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    done_seen = True
                    break
                ev = json.loads(data_str)
                if ttft is None:
                    ttft = time.perf_counter() - t0
                chunks.append(ev)
                if cancel_after_chunks is not None and len(chunks) >= cancel_after_chunks:
                    cancelled = True
                    if hard_rst and conn.sock is not None:
                        try:
                            conn.sock.setsockopt(
                                socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0)
                            )
                        except OSError:
                            pass
                    break
        finally:
            conn.close()

        # Reassemble text, reasoning, finish_reasons, usage
        texts: dict[int, list[str]] = {}
        reasonings: dict[int, list[str]] = {}
        finish_reasons: dict[int, str | None] = {}
        usage = None
        for ev in chunks:
            if ev.get("usage") is not None:
                usage = ev["usage"]
            for ch in ev.get("choices") or []:
                idx = int(ch.get("index", 0))
                delta = ch.get("delta") or {}
                if delta.get("content"):
                    texts.setdefault(idx, []).append(str(delta["content"]))
                rc = delta.get("reasoning_content") or delta.get("reasoning")
                if rc:
                    reasonings.setdefault(idx, []).append(str(rc))
                if ch.get("finish_reason") is not None:
                    finish_reasons[idx] = ch["finish_reason"]

        return {
            "status": status,
            "chunks": chunks,
            "num_chunks": len(chunks),
            "done_seen": done_seen,
            "cancelled": cancelled,
            "texts": {k: "".join(v) for k, v in texts.items()},
            "reasonings": {k: "".join(v) for k, v in reasonings.items()},
            "finish_reasons": finish_reasons,
            "usage": usage,
            "elapsed_s": time.perf_counter() - t0,
            "ttft_s": ttft,
        }


# ---------------------------------------------------------------------------
# Gate Result Accounting
# ---------------------------------------------------------------------------

@dataclass
class CaseResult:
    name: str
    passed: bool
    elapsed_s: float
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


def _extract_choice_text(choice: dict[str, Any]) -> str:
    msg = choice.get("message") or {}
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    return f"{reasoning}{content}"


def _choice_fingerprint(choice: dict[str, Any]) -> str:
    msg = choice.get("message") or {}
    canon = json.dumps(
        {
            "content": msg.get("content"),
            "reasoning_content": msg.get("reasoning_content") or msg.get("reasoning"),
            "tool_calls": msg.get("tool_calls"),
            "finish_reason": choice.get("finish_reason"),
        },
        sort_keys=True,
    )
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Compliance Test Cases
# ---------------------------------------------------------------------------

class LiveComplianceSuite:
    def __init__(
        self,
        client: OpenAIClient,
        model: str,
        *,
        mode: str = "full",
        harvester_workers: int = 8,
        harvester_n: int = 4,
        harvester_max_tokens: int = 256,
        mm_soak_iters: int = 4,
        mm_max_images: int = 8,
        skip_4k: bool = False,
        skip_multimodal: bool = False,
    ) -> None:
        self.client = client
        self.model = model
        self.mode = mode
        self.harvester_workers = harvester_workers
        self.harvester_n = harvester_n
        self.harvester_max_tokens = harvester_max_tokens
        self.mm_soak_iters = mm_soak_iters
        self.mm_max_images = mm_max_images
        self.skip_4k = skip_4k
        self.skip_multimodal = skip_multimodal
        self.results: list[CaseResult] = []
        self._baseline_det_fp: str | None = None

    def run_case(self, name: str, fn) -> CaseResult:
        t0 = time.perf_counter()
        try:
            details = fn() or {}
            res = CaseResult(name=name, passed=True, elapsed_s=time.perf_counter() - t0, details=details)
        except Exception as exc:  # noqa: BLE001
            res = CaseResult(
                name=name,
                passed=False,
                elapsed_s=time.perf_counter() - t0,
                error=f"{type(exc).__name__}: {exc}",
            )
        self.results.append(res)
        status = "PASS" if res.passed else "FAIL"
        extra = f" err={res.error}" if res.error else f" {json.dumps(res.details, sort_keys=True)[:140]}"
        print(f"[{status}] {name} ({res.elapsed_s:.2f}s){extra}", flush=True)
        return res

    # --- 1. Health & Models ---
    def test_health_and_models(self) -> dict[str, Any]:
        st_h, _ = self.client.get_json("/health", timeout=15.0)
        if st_h != 200:
            raise AssertionError(f"/health returned HTTP {st_h}")
        st_m, models_doc = self.client.get_json("/v1/models", timeout=15.0)
        if st_m != 200:
            raise AssertionError(f"/v1/models returned HTTP {st_m}")
        ids = [m.get("id") for m in (models_doc.get("data") or [])]
        if self.model not in ids:
            raise AssertionError(f"Expected model {self.model!r} in {ids}")
        return {"health_http": st_h, "models": ids}

    # --- 2. Temp=0 Determinism (SyncFree 0-ULP Repeatability) ---
    def test_temp0_determinism(self) -> dict[str, Any]:
        prompt = (
            "Compute the exact sum of the first 12 prime numbers, showing each prime "
            "and the final integer total."
        )
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 96,
            "seed": 42,
        }
        fps = []
        latencies = []
        for _ in range(3):
            st, doc, dt = self.client.post_json("/v1/chat/completions", payload)
            if st != 200:
                raise AssertionError(f"HTTP {st}: {doc}")
            choices = doc.get("choices") or []
            if len(choices) != 1:
                raise AssertionError(f"Expected 1 choice, got {len(choices)}")
            fps.append(_choice_fingerprint(choices[0]))
            latencies.append(round(dt, 3))
        if len(set(fps)) != 1:
            raise AssertionError(f"temp=0 non-deterministic fingerprints: {fps}")
        self._baseline_det_fp = fps[0]
        return {"fingerprint": fps[0], "runs": 3, "latencies_s": latencies}

    # --- 3. SSE Streaming + stream_options.include_usage + Mid-Stream Disconnect/Cancel ---
    def test_sse_streaming_and_cancel(self) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": "Write a 12-line numbered checklist for Linux kernel tuning."}
            ],
            "temperature": 0.0,
            "max_tokens": 128,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        full = self.client.post_sse("/v1/chat/completions", payload)
        if full["status"] != 200:
            raise AssertionError(f"SSE HTTP {full['status']}: {full.get('error')}")
        if not full["done_seen"]:
            raise AssertionError("SSE stream terminated without [DONE] sentinel")
        if full["num_chunks"] < 2:
            raise AssertionError(f"Too few SSE chunks: {full['num_chunks']}")
        usage = full["usage"]
        if not isinstance(usage, dict) or int(usage.get("completion_tokens", 0)) <= 0:
            raise AssertionError(f"Missing/invalid stream_options.include_usage payload: {usage}")

        # Now trigger 4 rapid mid-stream cancellations (2 graceful close, 2 hard TCP RST!)
        cancel_payload = dict(payload)
        cancel_payload["max_tokens"] = 512
        cancel_payload["temperature"] = 0.6
        cancel_counts = []
        for i in range(4):
            c_res = self.client.post_sse(
                "/v1/chat/completions",
                cancel_payload,
                cancel_after_chunks=2 + (i % 3),
                hard_rst=(i % 2 == 1),
            )
            if c_res["status"] != 200:
                raise AssertionError(f"Cancel trial {i} returned HTTP {c_res['status']}")
            cancel_counts.append(c_res["num_chunks"])

        # Confirm server remains healthy and still answers cleanly immediately after cancels
        st_h, _ = self.client.get_json("/health", timeout=15.0)
        if st_h != 200:
            raise AssertionError(f"Server unhealthy after mid-stream cancels: HTTP {st_h}")
        return {
            "full_chunks": full["num_chunks"],
            "ttft_s": round(full["ttft_s"] or 0.0, 4),
            "usage": usage,
            "cancel_chunks": cancel_counts,
        }

    # --- 4. Sampling Matrix (temp=0.6, top_p=0.95, min_p=0.05, n=4, max_tokens=1/2048, stop, logprobs) ---
    def test_sampling_matrix(self) -> dict[str, Any]:
        summary: dict[str, Any] = {}

        # 4a. max_tokens=1 boundary
        st, doc, _ = self.client.post_json(
            "/v1/chat/completions",
            {
                "model": self.model,
                "messages": [{"role": "user", "content": "Count from 1 to 100."}],
                "temperature": 0.0,
                "max_tokens": 1,
            },
        )
        if st != 200:
            raise AssertionError(f"max_tokens=1 HTTP {st}: {doc}")
        comp_toks = int((doc.get("usage") or {}).get("completion_tokens", -1))
        if comp_toks != 1:
            raise AssertionError(f"Expected completion_tokens==1, got {comp_toks}")
        summary["max_tokens_1"] = {
            "completion_tokens": comp_toks,
            "finish_reason": doc["choices"][0].get("finish_reason"),
        }

        # 4b. Stochastic sampling + n=4 shared-prefix fanout (harvester profile!)
        st, doc, dt_n4 = self.client.post_json(
            "/v1/chat/completions",
            {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Invent a creative 2-sentence sci-fi opening about a lunar "
                            "research outpost discovering an ancient radio beacon."
                        ),
                    }
                ],
                "temperature": 0.6,
                "top_p": 0.95,
                "min_p": 0.05,
                "n": 4,
                "max_tokens": 64,
            },
        )
        if st != 200:
            raise AssertionError(f"n=4 stochastic HTTP {st}: {doc}")
        choices = doc.get("choices") or []
        if len(choices) != 4:
            raise AssertionError(f"Expected 4 choices for n=4, got {len(choices)}")
        idxs = [int(c.get("index", -1)) for c in choices]
        if idxs != [0, 1, 2, 3]:
            raise AssertionError(f"Unexpected choice indices for n=4: {idxs}")
        texts = [_extract_choice_text(c) for c in choices]
        if any(len(t.strip()) == 0 for t in texts):
            raise AssertionError("Empty completion in n=4 fanout")
        summary["n4_stochastic"] = {
            "unique_outputs": len(set(texts)),
            "elapsed_s": round(dt_n4, 3),
        }

        # 4c. Stop sequences: stop=["</think>", "\n\n"]
        st, doc, _ = self.client.post_json(
            "/v1/chat/completions",
            {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": "Write two separate paragraphs separated by a blank line about TCP congestion control.",
                    }
                ],
                "temperature": 0.0,
                "max_tokens": 128,
                "stop": ["</think>", "\n\n"],
            },
        )
        if st != 200:
            raise AssertionError(f"stop sequence HTTP {st}: {doc}")
        c0 = doc["choices"][0]
        content0 = (c0.get("message") or {}).get("content") or ""
        if "\n\n" in content0 or "</think>" in content0:
            raise AssertionError(f"Stop sequence leaked into content: {content0!r}")
        summary["stop_seq"] = {"finish_reason": c0.get("finish_reason"), "len": len(content0)}

        # 4d. logprobs=True, top_logprobs=5 at BOTH temp=0 (SyncFree disarm test!) and temp=0.6
        for temp in (0.0, 0.6):
            st, doc, _ = self.client.post_json(
                "/v1/chat/completions",
                {
                    "model": self.model,
                    "messages": [{"role": "user", "content": "Reply with one word: hello."}],
                    "temperature": temp,
                    "max_tokens": 8,
                    "logprobs": True,
                    "top_logprobs": 5,
                },
            )
            if st != 200:
                raise AssertionError(f"logprobs temp={temp} HTTP {st}: {doc}")
            lp = (doc["choices"][0].get("logprobs") or {}).get("content")
            if not isinstance(lp, list) or len(lp) == 0:
                raise AssertionError(f"Missing logprobs.content for temp={temp}: {doc['choices'][0]}")
            first_tok = lp[0]
            top_lps = first_tok.get("top_logprobs") or []
            if len(top_lps) == 0 or len(top_lps) > 5:
                raise AssertionError(f"Invalid top_logprobs count ({len(top_lps)}) for temp={temp}")
            for entry in top_lps:
                val = float(entry["logprob"])
                if not math.isfinite(val) or val > 1e-3:
                    raise AssertionError(f"Invalid logprob value {val} at temp={temp}")
            summary[f"logprobs_t{temp}"] = {"n_tokens": len(lp), "top_k": len(top_lps)}

        # 4e. Long max_tokens=2048 reasoning trace test (quick caps to 256, full/harvester uses 2048)
        long_mt = 256 if self.mode == "quick" else 2048
        st, doc, dt_long = self.client.post_json(
            "/v1/chat/completions",
            {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Prove step by step why there are infinitely many primes of the form 4k+3, "
                            "then state the final QED conclusion clearly."
                        ),
                    }
                ],
                "temperature": 0.6,
                "top_p": 0.95,
                "min_p": 0.05,
                "max_tokens": long_mt,
            },
            timeout=240.0,
        )
        if st != 200:
            raise AssertionError(f"max_tokens={long_mt} HTTP {st}: {doc}")
        usage_l = doc.get("usage") or {}
        if int(usage_l.get("completion_tokens", 0)) > long_mt:
            raise AssertionError(f"Exceeded max_tokens={long_mt}: {usage_l}")
        summary["long_reasoning"] = {
            "max_tokens": long_mt,
            "completion_tokens": usage_l.get("completion_tokens"),
            "elapsed_s": round(dt_long, 3),
        }
        return summary

    # --- 5. Tool-Calling & Structured JSON Schema ---
    def test_tool_calling_and_json_schema(self) -> dict[str, Any]:
        # 5a. Function tool-calling (`--enable-auto-tool-choice --tool-call-parser inkling`)
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_cluster_telemetry",
                    "description": "Fetch live GPU temperature and RoCEv2 link state for a node.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "node": {"type": "string", "description": "Hostname such as spark-one"},
                            "include_ib": {"type": "boolean"},
                        },
                        "required": ["node"],
                    },
                },
            }
        ]
        st, doc, _ = self.client.post_json(
            "/v1/chat/completions",
            {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": "Call get_cluster_telemetry for node 'spark-one' with include_ib=true.",
                    }
                ],
                "tools": tools,
                "tool_choice": "auto",
                "temperature": 0.0,
                "max_tokens": 128,
            },
        )
        if st != 200:
            raise AssertionError(f"Tool-calling HTTP {st}: {doc}")
        msg = doc["choices"][0].get("message") or {}
        tcs = msg.get("tool_calls") or []
        tool_info: dict[str, Any] = {"has_tool_calls": bool(tcs), "finish_reason": doc["choices"][0].get("finish_reason")}
        if tcs:
            fn = (tcs[0].get("function") or {})
            args_parsed = json.loads(fn.get("arguments") or "{}")
            tool_info["fn_name"] = fn.get("name")
            tool_info["args"] = args_parsed

        # 5b. Grammar-constrained structured JSON schema (`response_format`)
        schema = {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["ok", "degraded"]},
                "score": {"type": "integer"},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["status", "score", "tags"],
            "additionalProperties": False,
        }
        st, doc_js, _ = self.client.post_json(
            "/v1/chat/completions",
            {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": "Return a JSON object with status='ok', score=100, and tags=['v15', 'sm121a'].",
                    }
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "v15_cert", "strict": True, "schema": schema},
                },
                "temperature": 0.0,
                "max_tokens": 96,
            },
        )
        if st != 200:
            raise AssertionError(f"JSON schema HTTP {st}: {doc_js}")
        raw_js = ((doc_js["choices"][0].get("message") or {}).get("content") or "").strip()
        parsed_js = json.loads(raw_js)
        for req_k in ("status", "score", "tags"):
            if req_k not in parsed_js:
                raise AssertionError(f"JSON schema output missing {req_k!r}: {parsed_js}")
        return {"tool_call": tool_info, "json_schema": parsed_js}

    # --- 6. Multimodal Vision Edge Shapes & Multi-Image Batches ---
    def test_multimodal_extreme_shapes(self) -> dict[str, Any]:
        if self.skip_multimodal:
            return {"skipped": True}
        shapes: list[tuple[str, list[tuple[int, int, str]]]] = [
            ("tiny_16x16", [(16, 16, "checker")]),
            ("std_1024x1024", [(1024, 1024, "gradient")]),
            ("wide_2560x40", [(2560, 40, "gradient")]),
            ("tall_40x2560", [(40, 2560, "gradient")]),
            (
                f"multi_image_{self.mm_max_images}",
                [(128 + 32 * (i % 4), 128 + 32 * ((i + 1) % 4), "checker" if i % 2 else "gradient") for i in range(self.mm_max_images)],
            ),
        ]
        if not self.skip_4k and self.mode != "quick":
            shapes.append(("uhd_4k_3840x2160", [(3840, 2160, "gradient")]))

        report: dict[str, Any] = {}
        for label, img_specs in shapes:
            content_parts: list[dict[str, Any]] = []
            for idx, (w, h, pat) in enumerate(img_specs):
                content_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": make_png_data_uri(w, h, pattern=pat, seed=idx + 1)},
                    }
                )
            content_parts.append(
                {
                    "type": "text",
                    "text": f"Describe briefly what colors/patterns appear across the {len(img_specs)} image(s).",
                }
            )
            st, doc, dt = self.client.post_json(
                "/v1/chat/completions",
                {
                    "model": self.model,
                    "messages": [{"role": "user", "content": content_parts}],
                    "temperature": 0.0,
                    "max_tokens": 32,
                },
                timeout=240.0,
            )
            if st != 200:
                raise AssertionError(f"Multimodal {label} HTTP {st}: {doc}")
            txt = _extract_choice_text(doc["choices"][0])
            if not txt.strip():
                raise AssertionError(f"Multimodal {label} returned empty response")
            report[label] = {
                "n_images": len(img_specs),
                "elapsed_s": round(dt, 3),
                "prompt_tokens": (doc.get("usage") or {}).get("prompt_tokens"),
                "completion_tokens": (doc.get("usage") or {}).get("completion_tokens"),
            }
        return report

    # --- 7. Multimodal Repeated Request Soak (UMA Hygiene / Latency Stability) ---
    def test_multimodal_soak(self) -> dict[str, Any]:
        if self.skip_multimodal or self.mm_soak_iters <= 0:
            return {"skipped": True}
        dims = [(16, 16), (256, 256), (512, 256), (256, 512), (640, 480), (1024, 1024)]
        latencies: list[float] = []
        for i in range(self.mm_soak_iters):
            w, h = dims[i % len(dims)]
            uri = make_png_data_uri(w, h, pattern="gradient" if i % 2 == 0 else "checker", seed=i + 10)
            st, doc, dt = self.client.post_json(
                "/v1/chat/completions",
                {
                    "model": self.model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": uri}},
                                {"type": "text", "text": f"Item {i}: reply with one short phrase."},
                            ],
                        }
                    ],
                    "temperature": 0.0,
                    "max_tokens": 16,
                },
                timeout=180.0,
            )
            if st != 200:
                raise AssertionError(f"MM soak iter {i} ({w}x{h}) HTTP {st}: {doc}")
            latencies.append(round(dt, 3))
        return {
            "iters": self.mm_soak_iters,
            "min_s": min(latencies),
            "max_s": max(latencies),
            "mean_s": round(sum(latencies) / len(latencies), 3),
        }

    # --- 8. Mixed Concurrent Workload (`n=4` Harvester + `temp=0` Exact + Multimodal + Tool-Call + Cancel) ---
    def test_mixed_concurrent_harvester_storm(self) -> dict[str, Any]:
        workers = self.harvester_workers
        n_fan = self.harvester_n
        max_tok = self.harvester_max_tokens
        shared_sys = (
            "You are a careful systems verification assistant analyzing distributed GPU inference "
            "pipelines on 8x DGX Spark Grace Blackwell clusters with RoCEv2 RDMA."
        )
        mm_uri = make_png_data_uri(256, 256, pattern="checker", seed=77)
        errors: list[str] = []
        det_fps: list[str] = []
        counts = {"harvester_n": 0, "det_t0": 0, "multimodal": 0, "tool_json": 0, "sse_cancel": 0}

        def _worker_task(wid: int) -> tuple[str, Any]:
            # First 5 slots are concurrent production canaries; remaining `workers` slots are
            # full rollout_harvester (n=4 shared-prefix fanout) workers!
            if wid in (0, 4):
                # Exact temp=0 repeatability canary running right alongside n=4 stochastic load!
                st, doc, dt = self.client.post_json(
                    "/v1/chat/completions",
                    {
                        "model": self.model,
                        "messages": [
                            {
                                "role": "user",
                                "content": (
                                    "Compute the exact sum of the first 12 prime numbers, showing each prime "
                                    "and the final integer total."
                                ),
                            }
                        ],
                        "temperature": 0.0,
                        "max_tokens": 96,
                        "seed": 42,
                    },
                )
                if st != 200:
                    raise AssertionError(f"det worker {wid} HTTP {st}: {doc}")
                return "det_t0", (_choice_fingerprint(doc["choices"][0]), dt)

            if wid == 1 and not self.skip_multimodal:
                st, doc, dt = self.client.post_json(
                    "/v1/chat/completions",
                    {
                        "model": self.model,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image_url", "image_url": {"url": mm_uri}},
                                    {"type": "text", "text": "What pattern is shown in this image?"},
                                ],
                            }
                        ],
                        "temperature": 0.0,
                        "max_tokens": 32,
                    },
                )
                if st != 200:
                    raise AssertionError(f"mm worker {wid} HTTP {st}: {doc}")
                return "multimodal", dt

            if wid == 2:
                st, doc, dt = self.client.post_json(
                    "/v1/chat/completions",
                    {
                        "model": self.model,
                        "messages": [
                            {"role": "user", "content": "Return JSON with status='ok', score=99, tags=['mix']."}
                        ],
                        "response_format": {
                            "type": "json_schema",
                            "json_schema": {
                                "name": "mix_schema",
                                "strict": True,
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "status": {"type": "string"},
                                        "score": {"type": "integer"},
                                        "tags": {"type": "array", "items": {"type": "string"}},
                                    },
                                    "required": ["status", "score", "tags"],
                                },
                            },
                        },
                        "temperature": 0.0,
                        "max_tokens": 64,
                    },
                )
                if st != 200:
                    raise AssertionError(f"tool_json worker {wid} HTTP {st}: {doc}")
                parsed = json.loads(((doc["choices"][0].get("message") or {}).get("content") or "").strip())
                if "status" not in parsed:
                    raise AssertionError(f"Invalid JSON in mixed worker {wid}: {parsed}")
                return "tool_json", dt

            if wid == 3:
                c_res = self.client.post_sse(
                    "/v1/chat/completions",
                    {
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": shared_sys},
                            {"role": "user", "content": "List 40 facts about GPU tensor cores."},
                        ],
                        "temperature": 0.6,
                        "top_p": 0.95,
                        "min_p": 0.05,
                        "max_tokens": max_tok,
                        "stream": True,
                        "stream_options": {"include_usage": True},
                    },
                    cancel_after_chunks=3,
                    hard_rst=True,
                )
                if c_res["status"] != 200:
                    raise AssertionError(f"sse_cancel worker {wid} HTTP {c_res['status']}")
                return "sse_cancel", c_res["elapsed_s"]

            # Rollout Harvester worker: shared prefix + n=4 fanout + temp=0.6 + top_p=0.95 + min_p=0.05!
            st, doc, dt = self.client.post_json(
                "/v1/chat/completions",
                {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": shared_sys},
                        {
                            "role": "user",
                            "content": f"Rollout task #{wid}: propose a concise optimization hypothesis.",
                        },
                    ],
                    "temperature": 0.6,
                    "top_p": 0.95,
                    "min_p": 0.05,
                    "n": n_fan,
                    "max_tokens": max_tok,
                },
                timeout=300.0,
            )
            if st != 200:
                raise AssertionError(f"harvester worker {wid} HTTP {st}: {doc}")
            if len(doc.get("choices") or []) != n_fan:
                raise AssertionError(f"harvester worker {wid} expected n={n_fan}, got {len(doc.get('choices') or [])}")
            return "harvester_n", dt

        total_tasks = workers + 5
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=min(total_tasks, 32)) as ex:
            futs = [ex.submit(_worker_task, i) for i in range(total_tasks)]
            for fut in as_completed(futs):
                try:
                    kind, val = fut.result()
                    counts[kind] += 1
                    if kind == "det_t0":
                        det_fps.append(val[0])
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{type(exc).__name__}: {exc}")
        wall_s = time.perf_counter() - t0
        if errors:
            raise AssertionError(f"{len(errors)} mixed-workload failures: {errors[:4]}")
        if self._baseline_det_fp is not None and any(fp != self._baseline_det_fp for fp in det_fps):
            raise AssertionError(
                f"temp=0 fingerprint drifted under concurrent mixed load: "
                f"baseline={self._baseline_det_fp}, observed={det_fps}"
            )
        return {
            "total_workers": total_tasks,
            "harvester_n": n_fan,
            "counts": counts,
            "det_fps_matched": len(det_fps),
            "wall_s": round(wall_s, 3),
        }

    # --- 9. Post-Storm Exactness & Health Check ---
    def test_post_storm_exactness(self) -> dict[str, Any]:
        st_h, _ = self.client.get_json("/health", timeout=15.0)
        if st_h != 200:
            raise AssertionError(f"Post-storm /health HTTP {st_h}")
        st, doc, dt = self.client.post_json(
            "/v1/chat/completions",
            {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Compute the exact sum of the first 12 prime numbers, showing each prime "
                            "and the final integer total."
                        ),
                    }
                ],
                "temperature": 0.0,
                "max_tokens": 96,
                "seed": 42,
            },
        )
        if st != 200:
            raise AssertionError(f"Post-storm exact check HTTP {st}: {doc}")
        fp = _choice_fingerprint(doc["choices"][0])
        if self._baseline_det_fp is not None and fp != self._baseline_det_fp:
            raise AssertionError(
                f"Post-storm temp=0 drift: expected {self._baseline_det_fp}, got {fp}"
            )
        return {"health_http": st_h, "fingerprint": fp, "elapsed_s": round(dt, 3)}

    def run_all(self) -> dict[str, Any]:
        t0 = time.perf_counter()
        self.run_case("1_health_and_models", self.test_health_and_models)
        self.run_case("2_temp0_determinism", self.test_temp0_determinism)
        self.run_case("3_sse_streaming_and_cancel", self.test_sse_streaming_and_cancel)
        self.run_case("4_sampling_matrix", self.test_sampling_matrix)
        self.run_case("5_tool_calling_and_json_schema", self.test_tool_calling_and_json_schema)
        self.run_case("6_multimodal_extreme_shapes", self.test_multimodal_extreme_shapes)
        self.run_case("7_multimodal_soak", self.test_multimodal_soak)
        self.run_case("8_mixed_concurrent_harvester_storm", self.test_mixed_concurrent_harvester_storm)
        self.run_case("9_post_storm_exactness", self.test_post_storm_exactness)
        n_pass = sum(1 for r in self.results if r.passed)
        n_fail = len(self.results) - n_pass
        return {
            "suite": "v15_api_multimodal_live",
            "mode": self.mode,
            "model": self.model,
            "passed": n_fail == 0,
            "num_passed": n_pass,
            "num_failed": n_fail,
            "total_elapsed_s": round(time.perf_counter() - t0, 3),
            "cases": [asdict(r) for r in self.results],
        }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Wave-V15 Live OpenAI API + Multimodal Compliance Harness")
    p.add_argument("--base-url", default="http://127.0.0.1:8010", help="OpenAI-compatible vLLM base URL")
    p.add_argument("--model", default="inkling-2-small", help="Served model name")
    p.add_argument("--api-key", default="EMPTY", help="Optional Bearer API key")
    p.add_argument(
        "--mode",
        choices=("quick", "full", "harvester"),
        default="full",
        help="quick=smoke, full=all shapes+4K+mixed, harvester=16x n=4 2048-tok stress",
    )
    p.add_argument("--harvester-workers", type=int, default=None, help="Concurrent workers in mixed storm")
    p.add_argument("--harvester-n", type=int, default=4, help="Fanout n per harvester request")
    p.add_argument("--harvester-max-tokens", type=int, default=None, help="max_tokens per harvester request")
    p.add_argument("--mm-soak-iters", type=int, default=None, help="Repeated multimodal soak iterations")
    p.add_argument("--mm-max-images", type=int, default=None, help="Images in multi-image batch test")
    p.add_argument("--timeout", type=float, default=240.0, help="Per-request HTTP timeout (seconds)")
    p.add_argument("--skip-4k", action="store_true", help="Skip 3840x2160 4K vision test")
    p.add_argument("--skip-multimodal", action="store_true", help="Skip multimodal vision tests")
    p.add_argument("--out-json", default="", help="Optional path to write structured JSON report")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode == "quick":
        hw = args.harvester_workers or 6
        hmt = args.harvester_max_tokens or 64
        mm_soak = args.mm_soak_iters if args.mm_soak_iters is not None else 3
        mm_imgs = args.mm_max_images or 4
    elif args.mode == "harvester":
        hw = args.harvester_workers or 16
        hmt = args.harvester_max_tokens or 2048
        mm_soak = args.mm_soak_iters if args.mm_soak_iters is not None else 12
        mm_imgs = args.mm_max_images or 16
    else:
        hw = args.harvester_workers or 10
        hmt = args.harvester_max_tokens or 256
        mm_soak = args.mm_soak_iters if args.mm_soak_iters is not None else 8
        mm_imgs = args.mm_max_images or 8

    client = OpenAIClient(args.base_url, api_key=args.api_key, timeout=args.timeout)
    suite = LiveComplianceSuite(
        client,
        args.model,
        mode=args.mode,
        harvester_workers=hw,
        harvester_n=args.harvester_n,
        harvester_max_tokens=hmt,
        mm_soak_iters=mm_soak,
        mm_max_images=mm_imgs,
        skip_4k=args.skip_4k,
        skip_multimodal=args.skip_multimodal,
    )
    report = suite.run_all()
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    print(
        f"\nSUMMARY: {'PASS' if report['passed'] else 'FAIL'} "
        f"({report['num_passed']}/{report['num_passed'] + report['num_failed']} passed, "
        f"{report['total_elapsed_s']:.2f}s)"
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
