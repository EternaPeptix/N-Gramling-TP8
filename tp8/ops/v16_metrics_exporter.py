#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Wave-V16 8-Node Prometheus Telemetry Exporter for 8x DGX Spark (GB10 TP8) Inkling-2-Small.

Pure Python standard-library HTTP server (--port 9108, zero external pip dependencies) exposing:
  - GET /metrics  : Standard Prometheus text exposition format (0.0.4)
  - GET /healthz  : Exporter liveness & scrape status JSON
  - GET /summary  : Aggregated 8-node UMA/RoCEv2/GPU/KV/vLLM/alerts JSON snapshot

Aggregates across:
  1. All 8 DGX Spark nodes (spark-one..spark-eight):
     - Fast-path read of atomic /dev/shm/inkling_v16_rank{r}_metrics.json snapshots
     - Fallback /proc/meminfo & /proc/buddyinfo + nvidia-smi + RoCEv2 sysfs + dmesg Xids
     - Local rank-0 reads run directly without SSH; remote nodes use parallel ProxyJump SSH
       (strictly disabled under --local-only or --mock-nodes-dir for 100% offline testing!)
  2. vLLM serving endpoint (default http://127.0.0.1:8010/metrics & /health)
  3. Watchdog state JSON (default /var/lib/inkling-v15/watchdog_state.json) for crash-recovery events
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import math
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Callable
import urllib.error
import urllib.request

logger = logging.getLogger("v16_metrics_exporter")

DEFAULT_HOSTS = (
    "spark-one",
    "spark-two",
    "spark-three",
    "spark-four",
    "spark-five",
    "spark-six",
    "spark-seven",
    "spark-eight",
)
DEFAULT_JUMP_MAP = {
    "spark-five": "10.77.0.51",
    "spark-six": "10.77.0.61",
}
FATAL_XID_CODES = frozenset({13, 31, 43, 45, 48, 63, 64, 74, 79, 94, 95, 119, 120})
XID_RE = re.compile(r"NVRM:\s+Xid\s+\([^)]+\):\s*(\d+)")
PROM_LINE_RE = re.compile(
    r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{([^}]*)\})?\s+([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|[+-]?[Ii]nf|[Nn]a[Nn])(?:\s+\d+)?$"
)


def _parse_jump_map(raw: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for tok in (raw or "").split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _prom_escape_label(val: str) -> str:
    return str(val).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _format_prom_val(val: float | int) -> str:
    if isinstance(val, bool):
        return "1" if val else "0"
    if isinstance(val, int):
        return str(val)
    f = float(val)
    if math.isnan(f):
        return "NaN"
    if math.isinf(f):
        return "+Inf" if f > 0 else "-Inf"
    if f.is_integer() and abs(f) < 1e15:
        return f"{f:.1f}"
    return f"{f:.6g}"


def parse_prometheus_text(text: str) -> dict[str, list[tuple[str, float]]]:
    """Parse Prometheus text exposition format into {metric_name: [(raw_labels, value), ...]}."""
    parsed: dict[str, list[tuple[str, float]]] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        m = PROM_LINE_RE.match(line)
        if not m:
            continue
        name, labels_str, val_str = m.group(1), (m.group(2) or ""), m.group(3)
        try:
            val = float(val_str)
        except ValueError:
            continue
        if math.isnan(val) or math.isinf(val):
            continue
        parsed.setdefault(name, []).append((labels_str, val))
    return parsed


def _metric_sum(parsed: dict[str, list[tuple[str, float]]], *names: str) -> float | None:
    for name in names:
        entries = parsed.get(name)
        if entries:
            return sum(v for _, v in entries)
    return None


def _metric_first(parsed: dict[str, list[tuple[str, float]]], *names: str) -> float | None:
    for name in names:
        entries = parsed.get(name)
        if entries:
            return entries[0][1]
    return None


def parse_meminfo_text(text: str) -> dict[str, float]:
    raw: dict[str, int] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        k, rest = line.split(":", 1)
        parts = rest.strip().split()
        if not parts:
            continue
        try:
            val = int(parts[0])
        except ValueError:
            continue
        raw[k.strip()] = val
    if not raw:
        return {}
    mt = raw.get("MemTotal", 0)
    mf = raw.get("MemFree", 0)
    ma = raw.get("MemAvailable", mf)
    st = raw.get("SwapTotal", 0)
    sf = raw.get("SwapFree", 0)
    su = max(0, st - sf)
    return {
        "mem_total_gib": round(mt / 1048576.0, 3),
        "mem_avail_gib": round(ma / 1048576.0, 3),
        "swap_used_mib": round(su / 1024.0, 2),
    }


def parse_buddyinfo_text(text: str, page_size: int = 4096) -> dict[str, Any]:
    totals: list[int] = []
    for line in text.splitlines():
        if "zone" not in line:
            continue
        parts = line.split()
        try:
            zi = parts.index("zone")
        except ValueError:
            continue
        counts = [int(t) for t in parts[zi + 2 :] if t.isdigit()]
        if not counts:
            continue
        if len(counts) > len(totals):
            totals.extend([0] * (len(counts) - len(totals)))
        for i, c in enumerate(counts):
            totals[i] += c
    if not totals:
        return {}
    ge9_bytes = sum(totals[o] * (page_size << o) for o in range(9, len(totals)))
    out: dict[str, Any] = {
        "max_order": len(totals) - 1,
        "order_ge9_blocks": sum(totals[9:]) if len(totals) > 9 else 0,
        "order_ge9_gib": round(ge9_bytes / (1024.0**3), 3),
        "order_13_blocks": totals[13] if len(totals) > 13 else 0,
    }
    for o in range(9, 14):
        out[f"order_{o}_blocks"] = totals[o] if o < len(totals) else 0
    return out


def parse_nvidia_smi_temp_power(text: str) -> tuple[float | None, float | None]:
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("N/A"):
            continue
        parts = [p.strip() for p in line.split(",")]
        temp_c: float | None = None
        power_w: float | None = None
        if len(parts) >= 1:
            try:
                temp_c = float(parts[0])
            except ValueError:
                pass
        if len(parts) >= 2:
            try:
                power_w = float(parts[1])
            except ValueError:
                pass
        return temp_c, power_w
    return None, None


def count_fatal_xids(text: str) -> int:
    cnt = 0
    for line in text.splitlines():
        m = XID_RE.search(line)
        if m and int(m.group(1)) in FATAL_XID_CODES:
            cnt += 1
    return cnt


@dataclass
class ExporterConfig:
    bind: str = "0.0.0.0"
    port: int = 9108
    hosts: tuple[str, ...] = DEFAULT_HOSTS
    ssh_user: str = field(
        default_factory=lambda: os.environ.get("V15_SSH_USER", os.environ.get("USER", "jeweled"))
    )
    ssh_jump_host: str = field(default_factory=lambda: os.environ.get("V15_SSH_JUMP_HOST", "spark-one"))
    ssh_jump_map: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_JUMP_MAP))
    ssh_bin: str = field(default_factory=lambda: os.environ.get("V15_SSH_BIN", "ssh"))
    ssh_timeout_s: int = 8
    container_prefix: str = field(
        default_factory=lambda: os.environ.get("V15_CONTAINER_PREFIX", "inkling-tp8-r")
    )
    vllm_host: str = field(default_factory=lambda: os.environ.get("V15_HEAD_HOST", "127.0.0.1"))
    vllm_port: int = field(default_factory=lambda: int(os.environ.get("PORT", "8010")))
    http_timeout_s: float = 4.0
    scrape_interval_s: float = 5.0
    shm_dir: str = field(
        default_factory=lambda: os.environ.get("INKLING_TELEMETRY_V16_DIR", "/dev/shm")
    )
    watchdog_state_path: str = field(
        default_factory=lambda: os.path.join(
            os.environ.get("V15_STATE_DIR", "/var/lib/inkling-v15"),
            "watchdog_state.json",
        )
    )
    mock_nodes_dir: str = ""
    local_only: bool = False
    local_rank0: bool = True


@dataclass
class NodeMetrics:
    rank: int
    host: str
    up: bool = False
    container_running: bool = False
    uma_avail_gib: float | None = None
    swap_mib: float | None = None
    buddy_order13: int | None = None
    buddy_order_ge9_gib: float | None = None
    buddy_orders_9_13: dict[int, int] = field(default_factory=dict)
    roce_active_hcas: int | None = None
    gpu_temp_c: float | None = None
    gpu_power_w: float | None = None
    xid_count: int = 0
    rss_mib: float | None = None
    uptime_s: float | None = None
    pid: int | None = None
    kv_snapshot: dict[str, Any] | None = None
    source: str = "none"
    error: str = ""


@dataclass
class ClusterSnapshot:
    ts_unix: float
    scrape_duration_s: float
    nodes: list[NodeMetrics] = field(default_factory=list)
    vllm_up: bool = False
    vllm_error: str = ""
    kv_utilization_ratio: float | None = None
    kv_prefix_hit_ratio: float | None = None
    kv_evictions_total: float | None = None
    kv_free_blocks: int | None = None
    kv_active_blocks: int | None = None
    kv_cached_blocks: int | None = None
    kv_audit_anomalies_total: int = 0
    kv_group_utilization: dict[str, float] = field(default_factory=dict)
    spec_accept_len: float | None = None
    requests_running: float | None = None
    requests_waiting: float | None = None
    gen_tokens_total: float | None = None
    prompt_tokens_total: float | None = None
    preemptions_total: float | None = None
    canary_latency_s: float | None = None
    crash_restarts_window: int = 0
    watchdog_circuit_open: bool = False
    watchdog_using_fallback: bool = False
    alerts: list[dict[str, Any]] = field(default_factory=list)


class NodeCollector:
    """Abstract node collector backend (SSH production vs local/mock offline)."""

    def collect_node(self, rank: int, host: str) -> NodeMetrics:
        raise NotImplementedError


def _apply_shm_snapshot_to_node(nm: NodeMetrics, data: dict[str, Any]) -> None:
    if not isinstance(data, dict):
        return
    uma = data.get("uma") if isinstance(data.get("uma"), dict) else data
    if uma.get("mem_available_gib") is not None:
        nm.uma_avail_gib = float(uma["mem_available_gib"])
    elif uma.get("mem_avail_gib") is not None:
        nm.uma_avail_gib = float(uma["mem_avail_gib"])
    if uma.get("swap_used_mib") is not None:
        nm.swap_mib = float(uma["swap_used_mib"])
    elif uma.get("swap_mib") is not None:
        nm.swap_mib = float(uma["swap_mib"])

    buddy = data.get("buddy") if isinstance(data.get("buddy"), dict) else data
    if buddy.get("order_13_blocks") is not None:
        nm.buddy_order13 = int(buddy["order_13_blocks"])
    elif buddy.get("buddy_order13") is not None:
        nm.buddy_order13 = int(buddy["buddy_order13"])
    if buddy.get("order_ge9_gib") is not None:
        nm.buddy_order_ge9_gib = float(buddy["order_ge9_gib"])
    for o in range(9, 14):
        k = f"order_{o}_blocks"
        if buddy.get(k) is not None:
            nm.buddy_orders_9_13[o] = int(buddy[k])

    if data.get("rss_mib") is not None:
        nm.rss_mib = float(data["rss_mib"])
    if data.get("uptime_s") is not None:
        nm.uptime_s = float(data["uptime_s"])
    if data.get("pid") is not None:
        nm.pid = int(data["pid"])
    if data.get("roce_active_hcas") is not None:
        nm.roce_active_hcas = int(data["roce_active_hcas"])
    if data.get("gpu_temp_c") is not None:
        nm.gpu_temp_c = float(data["gpu_temp_c"])
    if data.get("gpu_power_w") is not None:
        nm.gpu_power_w = float(data["gpu_power_w"])
    if data.get("xid_count") is not None:
        nm.xid_count = int(data["xid_count"])
    if data.get("container_running") is not None:
        nm.container_running = bool(data["container_running"])
    if isinstance(data.get("kv"), dict):
        nm.kv_snapshot = dict(data["kv"])


class LocalMockNodeCollector(NodeCollector):
    """100% offline/local collector reading mock fixtures and/or local /dev/shm snapshots."""

    def __init__(self, cfg: ExporterConfig):
        self.cfg = cfg

    def collect_node(self, rank: int, host: str) -> NodeMetrics:
        nm = NodeMetrics(rank=rank, host=host)
        candidates: list[Path] = []
        if self.cfg.mock_nodes_dir:
            base = Path(self.cfg.mock_nodes_dir)
            candidates.extend(
                [
                    base / f"rank{rank}_fixture.json",
                    base / f"inkling_v16_rank{rank}_metrics.json",
                    base / f"{host}.json",
                ]
            )
        if self.cfg.shm_dir:
            candidates.append(Path(self.cfg.shm_dir) / f"inkling_v16_rank{rank}_metrics.json")

        loaded_any = False
        for p in candidates:
            if not p.is_file():
                continue
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                _apply_shm_snapshot_to_node(nm, data)
                nm.up = bool(data.get("up", True))
                if "container_running" not in data:
                    nm.container_running = nm.up
                nm.source = str(p.name)
                loaded_any = True
                break
            except Exception as e:  # noqa: BLE001
                nm.error = f"parse_error:{p.name}:{e}"

        if not loaded_any and not nm.error:
            nm.error = "snapshot_not_found"
        return nm


class SSHNodeCollector(NodeCollector):
    """Production 8-node collector (local fast-path for rank 0, batched ProxyJump SSH for ranks 1..7)."""

    def __init__(self, cfg: ExporterConfig):
        self.cfg = cfg

    def _build_probe_cmd(self, rank: int) -> str:
        cname = f"{self.cfg.container_prefix}{rank}"
        shm_file = os.path.join(self.cfg.shm_dir, f"inkling_v16_rank{rank}_metrics.json")
        return (
            f"echo '===SHM==='; cat {shlex.quote(shm_file)} 2>/dev/null || true; "
            f"echo '===CONTAINER==='; docker inspect -f '{{{{.State.Running}}}}' {shlex.quote(cname)} 2>/dev/null || echo false; "
            f"echo '===ROCE==='; grep -l ACTIVE /sys/class/infiniband/*/ports/1/state 2>/dev/null | wc -l | tr -d ' '; "
            f"echo '===GPU==='; nvidia-smi --query-gpu=temperature.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null || true; "
            f"echo '===XID==='; (dmesg 2>/dev/null || true) | grep -E 'NVRM: Xid' | tail -n 50 || true; "
            f"if [ ! -r {shlex.quote(shm_file)} ]; then "
            f"  echo '===MEM==='; cat /proc/meminfo 2>/dev/null | grep -E '^(MemTotal|MemFree|MemAvailable|SwapTotal|SwapFree):' || true; "
            f"  echo '===BUDDY==='; cat /proc/buddyinfo 2>/dev/null || true; "
            f"fi"
        )

    def _run_host(self, rank: int, host: str, cmd: str) -> tuple[int, str, str]:
        if self.cfg.local_rank0 and rank == 0:
            try:
                cp = subprocess.run(
                    ["bash", "-c", cmd],
                    capture_output=True,
                    text=True,
                    timeout=self.cfg.ssh_timeout_s + 2,
                    check=False,
                )
                return cp.returncode, cp.stdout, cp.stderr
            except Exception as e:  # noqa: BLE001
                return 1, "", str(e)

        args = [
            self.cfg.ssh_bin,
            "-o",
            f"ConnectTimeout={self.cfg.ssh_timeout_s}",
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=accept-new",
        ]
        if host in self.cfg.ssh_jump_map:
            args += [
                "-J",
                f"{self.cfg.ssh_user}@{self.cfg.ssh_jump_host}",
                f"{self.cfg.ssh_user}@{self.cfg.ssh_jump_map[host]}",
            ]
        else:
            args += [f"{self.cfg.ssh_user}@{host}"]
        args.append(cmd)
        try:
            cp = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=self.cfg.ssh_timeout_s + 4,
                check=False,
            )
            return cp.returncode, cp.stdout, cp.stderr
        except subprocess.TimeoutExpired:
            return 124, "", "ssh_timeout"
        except OSError as e:
            return 127, "", str(e)

    def collect_node(self, rank: int, host: str) -> NodeMetrics:
        nm = NodeMetrics(rank=rank, host=host)
        rc, out, err = self._run_host(rank, host, self._build_probe_cmd(rank))
        if rc != 0 and not out:
            nm.up = False
            nm.error = f"probe_failed:rc={rc}:{err.strip()[:120]}"
            return nm

        nm.up = True
        nm.source = "local" if (self.cfg.local_rank0 and rank == 0) else "ssh"
        sections: dict[str, list[str]] = {}
        cur = ""
        for line in out.splitlines():
            if line.startswith("===") and line.endswith("==="):
                cur = line.strip("=")
                sections[cur] = []
            elif cur:
                sections[cur].append(line)

        shm_txt = "\n".join(sections.get("SHM", [])).strip()
        if shm_txt:
            try:
                _apply_shm_snapshot_to_node(nm, json.loads(shm_txt))
                nm.source = f"{nm.source}+shm"
            except Exception:
                pass

        c_txt = "\n".join(sections.get("CONTAINER", [])).strip().lower()
        nm.container_running = c_txt == "true"

        roce_txt = "\n".join(sections.get("ROCE", [])).strip()
        if roce_txt.isdigit():
            nm.roce_active_hcas = int(roce_txt)

        temp_c, power_w = parse_nvidia_smi_temp_power("\n".join(sections.get("GPU", [])))
        if temp_c is not None:
            nm.gpu_temp_c = temp_c
        if power_w is not None:
            nm.gpu_power_w = power_w

        nm.xid_count = count_fatal_xids("\n".join(sections.get("XID", [])))

        if nm.uma_avail_gib is None or nm.swap_mib is None:
            mem = parse_meminfo_text("\n".join(sections.get("MEM", [])))
            if mem:
                nm.uma_avail_gib = mem.get("mem_avail_gib")
                nm.swap_mib = mem.get("swap_used_mib")

        if nm.buddy_order13 is None:
            buddy = parse_buddyinfo_text("\n".join(sections.get("BUDDY", [])))
            if buddy:
                nm.buddy_order13 = int(buddy.get("order_13_blocks", 0))
                nm.buddy_order_ge9_gib = float(buddy.get("order_ge9_gib", 0.0))
                for o in range(9, 14):
                    nm.buddy_orders_9_13[o] = int(buddy.get(f"order_{o}_blocks", 0))

        return nm


def evaluate_alerts(snap: ClusterSnapshot) -> list[dict[str, Any]]:
    """Evaluate Wave-V16 alert rules inline for /summary diagnostics."""
    alerts: list[dict[str, Any]] = []
    if not snap.vllm_up:
        alerts.append({"alert": "InklingV16VLLMDown", "severity": "critical", "value": 0})
    for nm in snap.nodes:
        lbl = {"rank": str(nm.rank), "host": nm.host}
        if not nm.up:
            alerts.append({"alert": "InklingV16NodeUnreachable", "severity": "critical", "labels": lbl, "value": 0})
            continue
        if nm.swap_mib is not None and nm.swap_mib > 256.0:
            sev = "critical" if nm.swap_mib > 1024.0 else "warning"
            alerts.append({"alert": "InklingV16NodeSwapHigh", "severity": sev, "labels": lbl, "value": nm.swap_mib})
        if nm.buddy_order13 is not None and nm.buddy_order13 < 2500:
            alerts.append({"alert": "InklingV16BuddyOrder13Fragmented", "severity": "warning", "labels": lbl, "value": nm.buddy_order13})
        if nm.uma_avail_gib is not None and nm.uma_avail_gib < 5.0:
            alerts.append({"alert": "InklingV16NodeUMAAvailLow", "severity": "critical", "labels": lbl, "value": nm.uma_avail_gib})
        if nm.roce_active_hcas is not None and nm.roce_active_hcas < 4:
            alerts.append({"alert": "InklingV16RoCEHCADown", "severity": "critical", "labels": lbl, "value": nm.roce_active_hcas})
        if nm.xid_count > 0:
            alerts.append({"alert": "InklingV16GPUXidDetected", "severity": "critical", "labels": lbl, "value": nm.xid_count})
        if nm.gpu_temp_c is not None and nm.gpu_temp_c >= 85.0:
            alerts.append({"alert": "InklingV16GPUTempHigh", "severity": "warning", "labels": lbl, "value": nm.gpu_temp_c})
    if snap.kv_audit_anomalies_total > 0:
        alerts.append({"alert": "InklingV16KVLeakAnomaly", "severity": "critical", "value": snap.kv_audit_anomalies_total})
    if snap.canary_latency_s is not None and snap.canary_latency_s > 5.0:
        sev = "critical" if snap.canary_latency_s > 15.0 else "warning"
        alerts.append({"alert": "InklingV16CanaryLatencySpike", "severity": sev, "value": snap.canary_latency_s})
    if snap.watchdog_circuit_open:
        alerts.append({"alert": "InklingV16WatchdogCircuitOpen", "severity": "critical", "value": 1})
    return alerts


def render_prometheus_metrics(snap: ClusterSnapshot) -> str:
    """Render ClusterSnapshot into strictly compliant Prometheus 0.0.4 text format."""
    lines: list[str] = []

    def _emit_family(
        name: str,
        mtype: str,
        help_text: str,
        samples: list[tuple[dict[str, str], float | int]],
    ) -> None:
        if not samples:
            return
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} {mtype}")
        for labels, val in samples:
            if labels:
                lstr = ",".join(f'{k}="{_prom_escape_label(v)}"' for k, v in labels.items())
                lines.append(f"{name}{{{lstr}}} {_format_prom_val(val)}")
            else:
                lines.append(f"{name} {_format_prom_val(val)}")

    # Per-node metrics (all 8 Sparks)
    s_up: list[tuple[dict[str, str], float | int]] = []
    s_ctr: list[tuple[dict[str, str], float | int]] = []
    s_uma: list[tuple[dict[str, str], float | int]] = []
    s_swap: list[tuple[dict[str, str], float | int]] = []
    s_o13: list[tuple[dict[str, str], float | int]] = []
    s_ge9: list[tuple[dict[str, str], float | int]] = []
    s_ord: list[tuple[dict[str, str], float | int]] = []
    s_roce: list[tuple[dict[str, str], float | int]] = []
    s_temp: list[tuple[dict[str, str], float | int]] = []
    s_pwr: list[tuple[dict[str, str], float | int]] = []
    s_xid: list[tuple[dict[str, str], float | int]] = []
    s_rss: list[tuple[dict[str, str], float | int]] = []
    s_upt: list[tuple[dict[str, str], float | int]] = []

    for nm in sorted(snap.nodes, key=lambda x: x.rank):
        lbl = {"rank": str(nm.rank), "host": nm.host}
        s_up.append((lbl, 1 if nm.up else 0))
        s_ctr.append((lbl, 1 if nm.container_running else 0))
        if nm.uma_avail_gib is not None:
            s_uma.append((lbl, nm.uma_avail_gib))
        if nm.swap_mib is not None:
            s_swap.append((lbl, nm.swap_mib))
        if nm.buddy_order13 is not None:
            s_o13.append((lbl, nm.buddy_order13))
        if nm.buddy_order_ge9_gib is not None:
            s_ge9.append((lbl, nm.buddy_order_ge9_gib))
        for o in sorted(nm.buddy_orders_9_13):
            s_ord.append(({"rank": str(nm.rank), "host": nm.host, "order": str(o)}, nm.buddy_orders_9_13[o]))
        if nm.roce_active_hcas is not None:
            s_roce.append((lbl, nm.roce_active_hcas))
        if nm.gpu_temp_c is not None:
            s_temp.append((lbl, nm.gpu_temp_c))
        if nm.gpu_power_w is not None:
            s_pwr.append((lbl, nm.gpu_power_w))
        s_xid.append((lbl, nm.xid_count))
        if nm.rss_mib is not None:
            s_rss.append((lbl, nm.rss_mib))
        if nm.uptime_s is not None:
            s_upt.append((lbl, nm.uptime_s))

    _emit_family("inkling_v16_node_up", "gauge", "Whether Spark node telemetry probe succeeded (1=up, 0=down).", s_up)
    _emit_family("inkling_v16_node_container_running", "gauge", "Whether inkling-tp8 rank container is running (1=running).", s_ctr)
    _emit_family("inkling_v16_node_uma_avail_gib", "gauge", "GB10 unified memory MemAvailable in GiB.", s_uma)
    _emit_family("inkling_v16_node_swap_mib", "gauge", "GB10 swap space currently used in MiB.", s_swap)
    _emit_family("inkling_v16_node_buddy_order13", "gauge", "Contiguous /proc/buddyinfo order-13 (32 MiB) free block count.", s_o13)
    _emit_family("inkling_v16_node_buddy_order_ge9_gib", "gauge", "Total contiguous /proc/buddyinfo order>=9 memory in GiB.", s_ge9)
    _emit_family("inkling_v16_node_buddy_order", "gauge", "Contiguous /proc/buddyinfo free block count for orders 9..13.", s_ord)
    _emit_family("inkling_v16_node_roce_active_hcas", "gauge", "Count of ACTIVE ConnectX-7 RoCEv2 HCAs (target 4).", s_roce)
    _emit_family("inkling_v16_node_gpu_temp_c", "gauge", "GB10 GPU temperature in degrees Celsius.", s_temp)
    _emit_family("inkling_v16_node_gpu_power_w", "gauge", "GB10 GPU power draw in Watts.", s_pwr)
    _emit_family("inkling_v16_node_xid_count", "gauge", "Count of fatal NVIDIA NVRM Xid events observed on node.", s_xid)
    _emit_family("inkling_v16_node_rss_mib", "gauge", "Worker/EngineCore resident set size (VmRSS) in MiB.", s_rss)
    _emit_family("inkling_v16_node_uptime_seconds", "gauge", "Worker/EngineCore process uptime in seconds.", s_upt)

    # Cluster / KV / vLLM serving metrics
    _emit_family("inkling_v16_vllm_up", "gauge", "Whether head vLLM /metrics endpoint is reachable (1=up).", [({}, 1 if snap.vllm_up else 0)])
    if snap.kv_utilization_ratio is not None:
        _emit_family("inkling_v16_kv_utilization_ratio", "gauge", "Unified 14-group KV cache utilization ratio (0..1).", [({}, snap.kv_utilization_ratio)])
    if snap.kv_prefix_hit_ratio is not None:
        _emit_family("inkling_v16_kv_prefix_hit_ratio", "gauge", "Prefix cache hit ratio across lookups (0..1).", [({}, snap.kv_prefix_hit_ratio)])
    if snap.kv_evictions_total is not None:
        _emit_family("inkling_v16_kv_evictions_total", "counter", "Cumulative KV cache prefix/block eviction count.", [({}, snap.kv_evictions_total)])
    if snap.kv_free_blocks is not None:
        _emit_family("inkling_v16_kv_free_blocks", "gauge", "Current free blocks in unified KV BlockPool.", [({}, snap.kv_free_blocks)])
    if snap.kv_active_blocks is not None:
        _emit_family("inkling_v16_kv_active_blocks", "gauge", "Current active blocks in unified KV BlockPool.", [({}, snap.kv_active_blocks)])
    if snap.kv_cached_blocks is not None:
        _emit_family("inkling_v16_kv_cached_blocks", "gauge", "Current hashed/cached block entries in unified KV BlockPool.", [({}, snap.kv_cached_blocks)])
    _emit_family("inkling_v16_kv_audit_anomalies_total", "counter", "Cumulative KV invariant/leak audit anomalies detected.", [({}, snap.kv_audit_anomalies_total)])
    if snap.kv_group_utilization:
        _emit_family(
            "inkling_v16_kv_group_utilization_ratio",
            "gauge",
            "Active KV pool share by Inkling hybrid group role (global/swa/sconv/drafter).",
            [({"group": k}, v) for k, v in sorted(snap.kv_group_utilization.items())],
        )
    if snap.spec_accept_len is not None:
        _emit_family("inkling_v16_spec_accept_len", "gauge", "Mean speculative decoding acceptance length (AL).", [({}, snap.spec_accept_len)])
    if snap.requests_running is not None:
        _emit_family("inkling_v16_requests_running", "gauge", "Number of currently running requests on vLLM.", [({}, snap.requests_running)])
    if snap.requests_waiting is not None:
        _emit_family("inkling_v16_requests_waiting", "gauge", "Number of requests queued waiting on vLLM.", [({}, snap.requests_waiting)])
    if snap.gen_tokens_total is not None:
        _emit_family("inkling_v16_throughput_gen_tokens_total", "counter", "Total generated decode tokens served by vLLM.", [({}, snap.gen_tokens_total)])
    if snap.prompt_tokens_total is not None:
        _emit_family("inkling_v16_throughput_prompt_tokens_total", "counter", "Total prompt prefill tokens processed by vLLM.", [({}, snap.prompt_tokens_total)])
    if snap.preemptions_total is not None:
        _emit_family("inkling_v16_preemptions_total", "counter", "Cumulative KV preemption events reported by vLLM.", [({}, snap.preemptions_total)])
    if snap.canary_latency_s is not None:
        _emit_family("inkling_v16_canary_latency_seconds", "gauge", "Latest canary/health probe round-trip latency in seconds.", [({}, snap.canary_latency_s)])

    # Watchdog & exporter meta metrics
    _emit_family("inkling_v16_crash_recovery_restarts_window", "gauge", "Number of watchdog-triggered cluster restarts in active circuit-breaker window.", [({}, snap.crash_restarts_window)])
    _emit_family("inkling_v16_watchdog_circuit_open", "gauge", "Whether cluster watchdog circuit breaker is OPEN (1=open).", [({}, 1 if snap.watchdog_circuit_open else 0)])
    _emit_family("inkling_v16_watchdog_using_fallback", "gauge", "Whether cluster is serving on fallback env profile (1=fallback).", [({}, 1 if snap.watchdog_using_fallback else 0)])
    _emit_family("inkling_v16_exporter_scrape_duration_seconds", "gauge", "Total duration of the latest 8-node + vLLM exporter scrape.", [({}, round(snap.scrape_duration_s, 4))])
    _emit_family("inkling_v16_exporter_last_scrape_timestamp_seconds", "gauge", "Unix timestamp of the latest completed exporter scrape.", [({}, round(snap.ts_unix, 3))])

    return "\n".join(lines) + "\n"


class ClusterMetricsAggregator:
    """Thread-safe 8-node + vLLM + watchdog telemetry scraper and cache."""

    def __init__(
        self,
        cfg: ExporterConfig,
        collector: NodeCollector | None = None,
        http_fetch: Callable[[str, float], tuple[bool, int, str, float]] | None = None,
    ):
        self.cfg = cfg
        if collector is not None:
            self.collector = collector
        elif cfg.local_only or cfg.mock_nodes_dir:
            self.collector = LocalMockNodeCollector(cfg)
        else:
            self.collector = SSHNodeCollector(cfg)
        self._http_fetch = http_fetch or self._default_http_fetch
        self._lock = threading.Lock()
        self._latest: ClusterSnapshot | None = None
        self._stop_event = threading.Event()
        self._bg_thread: threading.Thread | None = None

    @staticmethod
    def _default_http_fetch(url: str, timeout_s: float) -> tuple[bool, int, str, float]:
        t0 = time.monotonic()
        req = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                body = resp.read().decode("utf-8", errors="replace")
                return (200 <= resp.status < 300), resp.status, body, (time.monotonic() - t0)
        except urllib.error.HTTPError as e:
            return False, e.code, str(e), (time.monotonic() - t0)
        except Exception as e:  # noqa: BLE001
            return False, 0, str(e), (time.monotonic() - t0)

    def _read_watchdog_state(self, snap: ClusterSnapshot) -> None:
        p = Path(self.cfg.watchdog_state_path)
        if not p.is_file():
            return
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return
        snap.watchdog_circuit_open = bool(data.get("circuit_open", False))
        snap.watchdog_using_fallback = bool(data.get("using_fallback", False))
        restarts = data.get("restart_timestamps")
        if isinstance(restarts, list):
            cutoff = time.time() - float(os.environ.get("V15_BREAKER_WINDOW_S", "900"))
            snap.crash_restarts_window = sum(
                1 for ts in restarts if isinstance(ts, (int, float)) and ts >= cutoff
            )
        if data.get("last_canary_latency_ms") is not None:
            snap.canary_latency_s = round(float(data["last_canary_latency_ms"]) / 1000.0, 4)

    def scrape_once(self) -> ClusterSnapshot:
        t0 = time.monotonic()
        ts_unix = time.time()

        # 1. Parallel 8-node scrape
        nodes: list[NodeMetrics] = []
        max_w = max(1, min(len(self.cfg.hosts), 8))
        with ThreadPoolExecutor(max_workers=max_w) as pool:
            futs = [pool.submit(self.collector.collect_node, r, h) for r, h in enumerate(self.cfg.hosts)]
            for f in futs:
                nodes.append(f.result())

        snap = ClusterSnapshot(ts_unix=ts_unix, scrape_duration_s=0.0, nodes=nodes)

        # 2. Scrape vLLM /metrics on spark-one:8010 (or configured endpoint)
        vllm_url = f"http://{self.cfg.vllm_host}:{self.cfg.vllm_port}/metrics"
        ok, status, body, latency_s = self._http_fetch(vllm_url, self.cfg.http_timeout_s)
        snap.vllm_up = ok
        if ok:
            parsed = parse_prometheus_text(body)
            snap.requests_running = _metric_sum(parsed, "vllm:num_requests_running", "num_requests_running")
            snap.requests_waiting = _metric_sum(parsed, "vllm:num_requests_waiting", "num_requests_waiting")
            kv_util = _metric_first(parsed, "vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc", "kv_cache_usage_perc")
            if kv_util is not None:
                # Normalize 0..100 -> 0..1 if a server emits percentage > 1.0
                snap.kv_utilization_ratio = round(kv_util / 100.0 if kv_util > 1.0 else kv_util, 6)

            # Prefix hit ratio from vLLM counters or gauge
            hits = _metric_sum(parsed, "vllm:prefix_cache_hits_total", "vllm:gpu_prefix_cache_hits_total")
            queries = _metric_sum(parsed, "vllm:prefix_cache_queries_total", "vllm:gpu_prefix_cache_queries_total")
            if hits is not None and queries is not None and queries > 0:
                snap.kv_prefix_hit_ratio = round(hits / queries, 6)
            else:
                rate_val = _metric_first(parsed, "vllm:gpu_prefix_cache_hit_rate", "vllm:prefix_cache_hit_rate")
                if rate_val is not None:
                    snap.kv_prefix_hit_ratio = round(rate_val / 100.0 if rate_val > 1.0 else rate_val, 6)

            # Speculative acceptance length (AL)
            al = _metric_first(
                parsed,
                "vllm:spec_decode_mean_acceptance_length",
                "vllm:spec_decode_draft_acceptance_length",
                "inkling_spec_accept_len",
            )
            if al is None:
                acc = _metric_sum(parsed, "vllm:spec_decode_num_accepted_tokens_total")
                drafts = _metric_sum(parsed, "vllm:spec_decode_num_drafts_total")
                if acc is not None and drafts is not None and drafts > 0:
                    al = 1.0 + (acc / drafts)
            if al is not None:
                snap.spec_accept_len = round(al, 4)

            snap.gen_tokens_total = _metric_sum(parsed, "vllm:generation_tokens_total", "generation_tokens_total")
            snap.prompt_tokens_total = _metric_sum(parsed, "vllm:prompt_tokens_total", "prompt_tokens_total")
            snap.preemptions_total = _metric_sum(parsed, "vllm:num_preemptions_total", "num_preemptions_total")
            evict_vllm = _metric_sum(parsed, "vllm:kv_cache_evictions_total", "inkling_kv_evictions_total")
            if evict_vllm is not None:
                snap.kv_evictions_total = evict_vllm
            snap.canary_latency_s = round(latency_s, 4)
        else:
            snap.vllm_error = f"status={status}:{body[:120]}"

        # 3. Merge atomic /dev/shm KV metrics snapshot from rank 0 (or any rank providing kv)
        for nm in sorted(nodes, key=lambda x: x.rank):
            kv = nm.kv_snapshot
            if not kv:
                continue
            if kv.get("utilization_ratio") is not None:
                snap.kv_utilization_ratio = round(float(kv["utilization_ratio"]), 6)
            if kv.get("prefix_hit_rate") is not None and (
                kv.get("prefix_query_tokens", 0) > 0 or snap.kv_prefix_hit_ratio is None
            ):
                snap.kv_prefix_hit_ratio = round(float(kv["prefix_hit_rate"]), 6)
            if kv.get("evictions_total") is not None:
                snap.kv_evictions_total = float(kv["evictions_total"])
            if kv.get("free_blocks") is not None:
                snap.kv_free_blocks = int(kv["free_blocks"])
            if kv.get("active_blocks") is not None:
                snap.kv_active_blocks = int(kv["active_blocks"])
            if kv.get("cached_blocks") is not None:
                snap.kv_cached_blocks = int(kv["cached_blocks"])
            if kv.get("kv_audit_anomalies") is not None:
                snap.kv_audit_anomalies_total = int(kv["kv_audit_anomalies"])
            hybrid = kv.get("hybrid_utilization")
            if isinstance(hybrid, dict):
                for role, rval in hybrid.items():
                    if isinstance(rval, dict) and rval.get("utilization_ratio") is not None:
                        snap.kv_group_utilization[str(role)] = round(float(rval["utilization_ratio"]), 6)
                    elif isinstance(rval, (int, float)):
                        snap.kv_group_utilization[str(role)] = round(float(rval), 6)
            break

        # 4. Watchdog state & optional canary mock override
        self._read_watchdog_state(snap)
        if self.cfg.mock_nodes_dir:
            canary_p = Path(self.cfg.mock_nodes_dir) / "canary.json"
            if canary_p.is_file():
                try:
                    cdata = json.loads(canary_p.read_text(encoding="utf-8"))
                    if cdata.get("canary_latency_s") is not None:
                        snap.canary_latency_s = float(cdata["canary_latency_s"])
                except Exception:
                    pass

        if snap.kv_evictions_total is None:
            snap.kv_evictions_total = 0.0

        snap.scrape_duration_s = round(time.monotonic() - t0, 4)
        snap.alerts = evaluate_alerts(snap)

        with self._lock:
            self._latest = snap
        return snap

    def get_snapshot(self, *, max_age_s: float | None = None) -> ClusterSnapshot:
        age_limit = self.cfg.scrape_interval_s * 1.5 if max_age_s is None else max_age_s
        with self._lock:
            cur = self._latest
        if cur is None or (time.time() - cur.ts_unix) > age_limit:
            return self.scrape_once()
        return cur

    def start_background_loop(self) -> None:
        if self._bg_thread is not None and self._bg_thread.is_alive():
            return
        self._stop_event.clear()

        def _loop() -> None:
            while not self._stop_event.is_set():
                try:
                    self.scrape_once()
                except Exception as e:  # noqa: BLE001
                    logger.warning("Background scrape error: %r", e)
                self._stop_event.wait(max(0.2, self.cfg.scrape_interval_s))

        self._bg_thread = threading.Thread(target=_loop, name="v16-exporter-scraper", daemon=True)
        self._bg_thread.start()

    def stop_background_loop(self) -> None:
        self._stop_event.set()
        if self._bg_thread is not None and self._bg_thread.is_alive():
            self._bg_thread.join(timeout=2.0)


def build_summary_dict(snap: ClusterSnapshot) -> dict[str, Any]:
    nodes_up = sum(1 for n in snap.nodes if n.up)
    return {
        "schema": "inkling.exporter.summary.v16",
        "ts_unix": snap.ts_unix,
        "scrape_duration_s": snap.scrape_duration_s,
        "cluster_status": "HEALTHY" if (snap.vllm_up and nodes_up == len(snap.nodes) and not any(a["severity"] == "critical" for a in snap.alerts)) else "DEGRADED",
        "vllm_up": snap.vllm_up,
        "vllm_error": snap.vllm_error,
        "nodes_total": len(snap.nodes),
        "nodes_up": nodes_up,
        "serving": {
            "requests_running": snap.requests_running,
            "requests_waiting": snap.requests_waiting,
            "spec_accept_len": snap.spec_accept_len,
            "gen_tokens_total": snap.gen_tokens_total,
            "prompt_tokens_total": snap.prompt_tokens_total,
            "preemptions_total": snap.preemptions_total,
            "canary_latency_s": snap.canary_latency_s,
        },
        "kv_cache": {
            "utilization_ratio": snap.kv_utilization_ratio,
            "prefix_hit_ratio": snap.kv_prefix_hit_ratio,
            "evictions_total": snap.kv_evictions_total,
            "free_blocks": snap.kv_free_blocks,
            "active_blocks": snap.kv_active_blocks,
            "cached_blocks": snap.kv_cached_blocks,
            "kv_audit_anomalies_total": snap.kv_audit_anomalies_total,
            "hybrid_utilization": snap.kv_group_utilization,
        },
        "watchdog": {
            "crash_restarts_window": snap.crash_restarts_window,
            "circuit_open": snap.watchdog_circuit_open,
            "using_fallback": snap.watchdog_using_fallback,
        },
        "alerts": snap.alerts,
        "nodes": [asdict(n) for n in sorted(snap.nodes, key=lambda x: x.rank)],
    }


def make_exporter_server(
    aggregator: ClusterMetricsAggregator,
    bind: str = "0.0.0.0",
    port: int = 9108,
) -> ThreadingHTTPServer:
    """Create a ThreadingHTTPServer serving /metrics, /healthz, and /summary."""

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            logger.debug("HTTP %s - " + fmt, self.address_string(), *args)

        def _send_bytes(self, code: int, content_type: str, payload: bytes) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path == "/metrics":
                snap = aggregator.get_snapshot()
                body = render_prometheus_metrics(snap).encode("utf-8")
                self._send_bytes(200, "text/plain; version=0.0.4; charset=utf-8", body)
            elif path == "/healthz":
                snap = aggregator.get_snapshot()
                nodes_up = sum(1 for n in snap.nodes if n.up)
                health = {
                    "status": "ok",
                    "vllm_up": snap.vllm_up,
                    "nodes_scraped": len(snap.nodes),
                    "nodes_up": nodes_up,
                    "active_alerts": len(snap.alerts),
                    "last_scrape_ts": snap.ts_unix,
                    "scrape_duration_s": snap.scrape_duration_s,
                }
                body = (json.dumps(health, sort_keys=True) + "\n").encode("utf-8")
                self._send_bytes(200, "application/json; charset=utf-8", body)
            elif path == "/summary":
                snap = aggregator.get_snapshot()
                body = (json.dumps(build_summary_dict(snap), indent=2, sort_keys=True) + "\n").encode("utf-8")
                self._send_bytes(200, "application/json; charset=utf-8", body)
            else:
                body = b'{"error":"not_found","endpoints":["/metrics","/healthz","/summary"]}\n'
                self._send_bytes(404, "application/json; charset=utf-8", body)

    class _ReusableThreadingHTTPServer(ThreadingHTTPServer):
        allow_reuse_address = True
        daemon_threads = True

    return _ReusableThreadingHTTPServer((bind, port), _Handler)


def build_config_from_args(args: argparse.Namespace) -> ExporterConfig:
    cfg = ExporterConfig(
        bind=args.bind,
        port=args.port,
        vllm_host=args.vllm_host,
        vllm_port=args.vllm_port,
        scrape_interval_s=args.scrape_interval,
        shm_dir=args.shm_dir,
        watchdog_state_path=args.watchdog_state,
        mock_nodes_dir=args.mock_nodes_dir or "",
        local_only=bool(args.local_only),
    )
    if args.hosts:
        cfg.hosts = tuple(args.hosts.split())
    elif os.environ.get("V15_HOSTS"):
        cfg.hosts = tuple(os.environ["V15_HOSTS"].split())
    if os.environ.get("V15_SSH_JUMP_MAP"):
        cfg.ssh_jump_map = _parse_jump_map(os.environ["V15_SSH_JUMP_MAP"])
    return cfg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Wave-V16 8-Node DGX Spark Prometheus Telemetry Exporter (:9108)"
    )
    parser.add_argument("--bind", default="0.0.0.0", help="Listen bind address (default 0.0.0.0)")
    parser.add_argument("--port", type=int, default=9108, help="Listen TCP port (default 9108; 0=ephemeral)")
    parser.add_argument("--hosts", help="Space-separated host list (default spark-one..spark-eight)")
    parser.add_argument("--vllm-host", default=os.environ.get("V15_HEAD_HOST", "127.0.0.1"))
    parser.add_argument("--vllm-port", type=int, default=int(os.environ.get("PORT", "8010")))
    parser.add_argument("--scrape-interval", type=float, default=5.0, help="Background scrape interval seconds")
    parser.add_argument("--shm-dir", default=os.environ.get("INKLING_TELEMETRY_V16_DIR", "/dev/shm"))
    parser.add_argument(
        "--watchdog-state",
        default=os.path.join(os.environ.get("V15_STATE_DIR", "/var/lib/inkling-v15"), "watchdog_state.json"),
    )
    parser.add_argument("--mock-nodes-dir", default="", help="Offline local fixture directory (disables SSH)")
    parser.add_argument("--local-only", action="store_true", help="Never use SSH; read only local shm/mock + HTTP")
    parser.add_argument("--once", action="store_true", help="Scrape once and print metrics (or --json) to stdout")
    parser.add_argument("--json", action="store_true", help="With --once, emit /summary JSON instead of Prometheus text")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        stream=sys.stderr,
    )
    cfg = build_config_from_args(args)
    agg = ClusterMetricsAggregator(cfg)

    if args.once:
        snap = agg.scrape_once()
        if args.json:
            print(json.dumps(build_summary_dict(snap), indent=2, sort_keys=True))
        else:
            sys.stdout.write(render_prometheus_metrics(snap))
        return 0

    agg.scrape_once()
    agg.start_background_loop()
    server = make_exporter_server(agg, bind=cfg.bind, port=cfg.port)
    actual_host, actual_port = server.server_address[:2]

    def _handle_sig(signum: int, _frame: Any) -> None:
        logger.info("Received signal %d; shutting down exporter cleanly...", signum)
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _handle_sig)
    signal.signal(signal.SIGINT, _handle_sig)
    logger.info(
        "Wave-V16 Prometheus exporter listening on http://%s:%d (/metrics, /healthz, /summary) scraping %d nodes + %s:%d",
        actual_host,
        actual_port,
        len(cfg.hosts),
        cfg.vllm_host,
        cfg.vllm_port,
    )
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        server.server_close()
        agg.stop_background_loop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
