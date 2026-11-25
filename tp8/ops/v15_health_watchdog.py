#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Wave-V15 8-Node Health Watchdog & Self-Healing Crash Recovery Engine for DGX Spark TP8.

Monitors all 8 DGX Spark nodes (spark-one..spark-eight, port 8010) across multi-tier probes:
  - Tier 1: HTTP /health liveness probe
  - Tier 2: Lightweight deterministic canary completion probe (catches silent TP8 shm_broadcast / RDMA hangs)
  - Tier 3: Parallel 8-node docker inspect, 4/4 ConnectX-7 RoCEv2 HCA state, GPU temp/throttle/ECC/Xid,
            UMA MemAvailable/Swap/buddyinfo, and fatal rank-log signatures.

On confirmed failure, performs coordinated 8-node self-healing:
  1. Captures timestamped diagnostic crash bundle to /var/log/inkling-v15/crash-<ts>/
  2. Cleanly drains & removes all 8 containers + sweeps scoped /dev/shm & /tmp/vllm* IPC state
  3. Verifies 4/4 RoCEv2 HCAs Active & UMA cleanliness (runs non-destructive defrag if needed)
  4. Applies circuit breaker (max 3 restarts/15min), exponential backoff, and automatic fallback
     to verified live_champion_v12.env after 2 consecutive V15 boot failures
  5. Relaunches ranks 7..1 then rank 0 atomically
  6. Runs fast 1-prompt 0-ULP boot certification before marking READY!

Pure Python standard library (no external dependencies; mockable backends for 100% offline tests).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
import datetime
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
from typing import Any, Callable
import urllib.error
import urllib.request

logger = logging.getLogger("v15_watchdog")

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
DEFAULT_HCAS = ("rocep1s0f0", "roceP2p1s0f0", "rocep1s0f1", "roceP2p1s0f1")

# Fatal NVIDIA Xid codes indicating unrecoverable GPU/MMU/bus/GSP faults
FATAL_XID_CODES = frozenset({13, 31, 43, 45, 48, 63, 64, 74, 79, 94, 95, 119, 120})
XID_RE = re.compile(r"NVRM:\s+Xid\s+\([^)]+\):\s*(\d+)")
FATAL_LOG_PATTERNS = (
    re.compile(r"shm_broadcast.*timed?\s*out", re.IGNORECASE),
    re.compile(r"RDMA.*timeout|ibv_poll_cq.*error|QP.*error", re.IGNORECASE),
    re.compile(r"CUDA error:\s*(an illegal memory access|unspecified launch failure|device-side assert)", re.IGNORECASE),
    re.compile(r"NCCL WARN.*unhandled system error|NCCL.*remote process exited", re.IGNORECASE),
    re.compile(r"EngineCore.*(?:died|crashed|fatal)|Worker_TP\d+.*(?:exited|died)", re.IGNORECASE),
)

DEFAULT_CANARY_PROMPT = "Compute 17 + 25 and reply with only the integer:"
DEFAULT_BOOT_CERT_PROMPT = (
    "State the first five prime numbers separated by commas, with no extra commentary:"
)


@dataclass
class CmdResult:
    rc: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False


@dataclass
class HTTPResponse:
    ok: bool
    status: int = 0
    body: str = ""
    latency_ms: float = 0.0
    error: str = ""


@dataclass
class WatchdogConfig:
    hosts: tuple[str, ...] = DEFAULT_HOSTS
    ssh_user: str = field(default_factory=lambda: os.environ.get("V15_SSH_USER", os.environ.get("USER", "jeweled")))
    ssh_jump_host: str = field(default_factory=lambda: os.environ.get("V15_SSH_JUMP_HOST", "spark-one"))
    ssh_jump_map: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_JUMP_MAP))
    ssh_bin: str = field(default_factory=lambda: os.environ.get("V15_SSH_BIN", "ssh"))
    scp_bin: str = field(default_factory=lambda: os.environ.get("V15_SCP_BIN", "scp"))
    ssh_timeout_s: int = 10
    head_host: str = field(default_factory=lambda: os.environ.get("V15_HEAD_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(os.environ.get("PORT", "8010")))
    served_model_name: str = field(default_factory=lambda: os.environ.get("V15_SERVED_MODEL", "inkling-2-small"))
    container_prefix: str = field(default_factory=lambda: os.environ.get("V15_CONTAINER_PREFIX", "inkling-tp8-r"))
    remote_launcher: str = field(
        default_factory=lambda: os.environ.get("V15_REMOTE_LAUNCHER", "/home/jeweled/inkling-tp8/inkling-tp8-v15.sh")
    )
    remote_env_path: str = field(
        default_factory=lambda: os.environ.get("V15_REMOTE_ENV_PATH", "/tmp/inkling_v15_active.env")
    )
    primary_env: str = field(default_factory=lambda: os.environ.get("V15_PRIMARY_ENV", ""))
    fallback_env: str = field(default_factory=lambda: os.environ.get("V15_FALLBACK_ENV", ""))
    log_dir: str = field(default_factory=lambda: os.environ.get("V15_LOG_DIR", "/var/log/inkling-v15"))
    state_dir: str = field(default_factory=lambda: os.environ.get("V15_STATE_DIR", "/var/lib/inkling-v15"))
    lock_file: str = field(default_factory=lambda: os.environ.get("V15_LOCK_FILE", "/tmp/inkling-v15-recovery.lock"))

    # Probe cadence & thresholds
    probe_interval_s: float = field(default_factory=lambda: float(os.environ.get("V15_WATCHDOG_INTERVAL_S", "15")))
    canary_interval_s: float = field(default_factory=lambda: float(os.environ.get("V15_CANARY_INTERVAL_S", "60")))
    health_timeout_s: float = 5.0
    canary_timeout_s: float = 25.0
    fail_threshold: int = field(default_factory=lambda: int(os.environ.get("V15_FAIL_THRESHOLD", "2")))
    required_active_hcas: int = field(default_factory=lambda: int(os.environ.get("V15_REQUIRED_ACTIVE_HCAS", "4")))
    gpu_temp_warn_c: float = 85.0
    gpu_temp_crit_c: float = 92.0
    runtime_min_mem_avail_gib: float = 3.0
    runtime_warn_mem_avail_gib: float = 5.0
    runtime_warn_swap_mib: float = 1024.0
    runtime_crit_swap_mib: float = 4096.0
    preboot_min_order13_blocks: int = field(default_factory=lambda: int(os.environ.get("UMA_MIN_ORDER13_BLOCKS", "2500")))
    preboot_max_swap_mib: float = field(default_factory=lambda: float(os.environ.get("UMA_MAX_SWAP_USED_MIB", "256")))
    auto_reboot_defrag: bool = False

    # Recovery, backoff, circuit breaker & boot cert
    max_restarts_in_window: int = field(default_factory=lambda: int(os.environ.get("V15_MAX_RESTARTS_15M", "3")))
    breaker_window_s: float = field(default_factory=lambda: float(os.environ.get("V15_BREAKER_WINDOW_S", "900")))
    fallback_after_v15_failures: int = field(default_factory=lambda: int(os.environ.get("V15_FALLBACK_AFTER_FAILS", "2")))
    backoff_base_s: float = 5.0
    backoff_max_s: float = 60.0
    stability_reset_s: float = 600.0
    boot_stability_window_s: float = 120.0
    boot_timeout_s: float = 420.0
    boot_poll_interval_s: float = 5.0
    stop_timeout_s: int = 15
    canary_prompt: str = DEFAULT_CANARY_PROMPT
    boot_cert_prompt: str = field(default_factory=lambda: os.environ.get("V15_BOOT_CERT_PROMPT", DEFAULT_BOOT_CERT_PROMPT))
    boot_cert_expected_sha256: str = field(default_factory=lambda: os.environ.get("V15_BOOT_CERT_SHA256", ""))
    boot_cert_max_tokens: int = 64
    boot_cert_repeat_check: bool = True
    alert_cmd: str = field(default_factory=lambda: os.environ.get("V15_ALERT_CMD", ""))


@dataclass
class NodeHealth:
    rank: int
    host: str
    reachable: bool = True
    container_running: bool = False
    container_status: str = "unknown"
    oom_killed: bool = False
    exit_code: int = 0
    active_hcas: int = 0
    hca_details: list[str] = field(default_factory=list)
    gpu_temp_c: float | None = None
    gpu_throttle_mask: str = "0x0"
    gpu_uncorrected_ecc: int = 0
    fatal_xids: list[str] = field(default_factory=list)
    mem_avail_gib: float | None = None
    swap_used_mib: float | None = None
    order_13_blocks: int | None = None
    order_ge9_gib: float | None = None
    log_fatal_hits: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class ClusterHealthReport:
    timestamp: float
    status: str  # HEALTHY | DEGRADED | UNHEALTHY
    http_health_ok: bool
    http_health_latency_ms: float = 0.0
    canary_ran: bool = False
    canary_ok: bool = True
    canary_latency_ms: float = 0.0
    canary_error: str = ""
    immediate_recovery: bool = False
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    nodes: list[NodeHealth] = field(default_factory=list)


@dataclass
class SupervisorState:
    status: str = "INIT"  # INIT | READY | READY_FALLBACK | RECOVERING | CIRCUIT_OPEN
    active_env: str = ""
    using_fallback: bool = False
    consecutive_probe_failures: int = 0
    consecutive_v15_boot_failures: int = 0
    consecutive_recoveries: int = 0
    restart_timestamps: list[float] = field(default_factory=list)
    circuit_open: bool = False
    circuit_opened_at: float = 0.0
    last_boot_ts: float = 0.0
    healthy_since_ts: float = 0.0
    last_canary_ts: float = 0.0
    last_crash_bundle: str = ""
    last_boot_cert_sha256: str = ""
    seen_xids: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SupervisorState":
        valid = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in valid}
        return cls(**filtered)


# ==============================================================================
# Backends (Production SSH/HTTP + Clean Abstract Interfaces for Offline Mocks)
# ==============================================================================


class ClusterBackend:
    """Abstract interface for 8-node command execution and env file distribution."""

    def run(self, host: str, cmd: str, timeout_s: int = 15) -> CmdResult:
        raise NotImplementedError

    def copy_file(self, local_path: str, host: str, remote_path: str, timeout_s: int = 15) -> CmdResult:
        raise NotImplementedError


class SSHClusterBackend(ClusterBackend):
    def __init__(self, cfg: WatchdogConfig):
        self.cfg = cfg

    def _ssh_args(self, host: str) -> list[str]:
        base = [
            self.cfg.ssh_bin,
            "-o", f"ConnectTimeout={self.cfg.ssh_timeout_s}",
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=accept-new",
        ]
        if host in self.cfg.ssh_jump_map:
            base += ["-J", f"{self.cfg.ssh_user}@{self.cfg.ssh_jump_host}", f"{self.cfg.ssh_user}@{self.cfg.ssh_jump_map[host]}"]
        else:
            base += [f"{self.cfg.ssh_user}@{host}"]
        return base

    def run(self, host: str, cmd: str, timeout_s: int = 15) -> CmdResult:
        args = self._ssh_args(host) + [cmd]
        try:
            cp = subprocess.run(args, capture_output=True, text=True, timeout=timeout_s, check=False)
            return CmdResult(rc=cp.returncode, stdout=cp.stdout, stderr=cp.stderr)
        except subprocess.TimeoutExpired as e:
            return CmdResult(
                rc=124,
                stdout=(e.stdout.decode("utf-8", "ignore") if isinstance(e.stdout, bytes) else (e.stdout or "")),
                stderr=(e.stderr.decode("utf-8", "ignore") if isinstance(e.stderr, bytes) else (e.stderr or "timeout")),
                timed_out=True,
            )
        except OSError as e:
            return CmdResult(rc=127, stderr=str(e))

    def copy_file(self, local_path: str, host: str, remote_path: str, timeout_s: int = 15) -> CmdResult:
        args = [
            self.cfg.scp_bin, "-q",
            "-o", f"ConnectTimeout={self.cfg.ssh_timeout_s}",
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=accept-new",
        ]
        if host in self.cfg.ssh_jump_map:
            args += ["-o", f"ProxyJump={self.cfg.ssh_user}@{self.cfg.ssh_jump_host}", local_path, f"{self.cfg.ssh_user}@{self.cfg.ssh_jump_map[host]}:{remote_path}"]
        else:
            args += [local_path, f"{self.cfg.ssh_user}@{host}:{remote_path}"]
        try:
            cp = subprocess.run(args, capture_output=True, text=True, timeout=timeout_s, check=False)
            return CmdResult(rc=cp.returncode, stdout=cp.stdout, stderr=cp.stderr)
        except subprocess.TimeoutExpired:
            return CmdResult(rc=124, stderr="scp timeout", timed_out=True)
        except OSError as e:
            return CmdResult(rc=127, stderr=str(e))


class HTTPClient:
    def get(self, url: str, timeout_s: float) -> HTTPResponse:
        raise NotImplementedError

    def post_json(self, url: str, payload: dict[str, Any], timeout_s: float) -> HTTPResponse:
        raise NotImplementedError


class UrllibHTTPClient(HTTPClient):
    def get(self, url: str, timeout_s: float) -> HTTPResponse:
        t0 = time.monotonic()
        req = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                body = resp.read().decode("utf-8", errors="replace")
                return HTTPResponse(ok=(200 <= resp.status < 300), status=resp.status, body=body, latency_ms=(time.monotonic() - t0) * 1000.0)
        except urllib.error.HTTPError as e:
            return HTTPResponse(ok=False, status=e.code, error=str(e), latency_ms=(time.monotonic() - t0) * 1000.0)
        except Exception as e:  # noqa: BLE001
            return HTTPResponse(ok=False, status=0, error=str(e), latency_ms=(time.monotonic() - t0) * 1000.0)

    def post_json(self, url: str, payload: dict[str, Any], timeout_s: float) -> HTTPResponse:
        t0 = time.monotonic()
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, method="POST", headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                body = resp.read().decode("utf-8", errors="replace")
                return HTTPResponse(ok=(200 <= resp.status < 300), status=resp.status, body=body, latency_ms=(time.monotonic() - t0) * 1000.0)
        except urllib.error.HTTPError as e:
            return HTTPResponse(ok=False, status=e.code, error=str(e), latency_ms=(time.monotonic() - t0) * 1000.0)
        except Exception as e:  # noqa: BLE001
            return HTTPResponse(ok=False, status=0, error=str(e), latency_ms=(time.monotonic() - t0) * 1000.0)


# ==============================================================================
# Parsing Helpers (Buddyinfo, Meminfo, ibstat, nvidia-smi, Xid, docker inspect)
# ==============================================================================


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
    o13 = totals[13] if len(totals) > 13 else 0
    return {
        "max_order": len(totals) - 1,
        "orders": totals,
        "order_ge9_blocks": sum(totals[9:]) if len(totals) > 9 else 0,
        "order_ge9_gib": round(ge9_bytes / (1024.0**3), 3),
        "order_13_blocks": o13,
    }


def parse_ibstat_text(text: str) -> tuple[int, list[str]]:
    """Parse ibstat OR sysfs HCA lines (`HCA <name> <state>`), returning (active_count, details)."""
    active = 0
    details: list[str] = []
    cur_ca = ""
    cur_state = ""
    cur_phys = ""
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("HCA "):
            parts = line.split(None, 2)
            if len(parts) >= 3:
                name, st = parts[1], parts[2]
                details.append(f"{name}:{st}")
                if "ACTIVE" in st.upper():
                    active += 1
            continue
        if line.startswith("CA '"):
            if cur_ca and cur_state:
                details.append(f"{cur_ca}:{cur_state}/{cur_phys or '?'}")
                if cur_state.upper() == "ACTIVE":
                    active += 1
            cur_ca = line.split("'")[1]
            cur_state = ""
            cur_phys = ""
        elif line.startswith("State:"):
            cur_state = line.split(":", 1)[1].strip()
        elif line.startswith("Physical state:"):
            cur_phys = line.split(":", 1)[1].strip()
    if cur_ca and cur_state:
        details.append(f"{cur_ca}:{cur_state}/{cur_phys or '?'}")
        if cur_state.upper() == "ACTIVE":
            active += 1
    return active, details


def parse_nvidia_smi_csv(text: str) -> dict[str, Any]:
    """Parse `temperature.gpu, memory.used, memory.total, clocks_throttle_reasons.active, ecc` CSV."""
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("N/A"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        out: dict[str, Any] = {}
        try:
            out["temp_c"] = float(parts[0])
        except ValueError:
            out["temp_c"] = None
        out["throttle_mask"] = parts[3] if parts[3] else "0x0"
        if len(parts) >= 5:
            try:
                out["uncorrected_ecc"] = int(parts[4])
            except ValueError:
                out["uncorrected_ecc"] = 0
        else:
            out["uncorrected_ecc"] = 0
        return out
    return {}


def parse_fatal_xids(dmesg_text: str, seen: set[str] | None = None) -> list[str]:
    hits: list[str] = []
    seen_set = seen or set()
    for line in dmesg_text.splitlines():
        m = XID_RE.search(line)
        if not m:
            continue
        code = int(m.group(1))
        if code in FATAL_XID_CODES:
            sig = line.strip()[:160]
            if sig not in seen_set:
                hits.append(f"Xid={code}:{sig}")
    return hits


def sd_notify(message: str) -> bool:
    sock_path = os.environ.get("NOTIFY_SOCKET", "").strip()
    if not sock_path:
        return False
    if sock_path.startswith("@"):
        sock_path = "\0" + sock_path[1:]
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as s:
            s.sendto(message.encode("utf-8"), sock_path)
        return True
    except OSError:
        return False


# ==============================================================================
# Core Watchdog & Recovery Orchestrator
# ==============================================================================


class ClusterWatchdog:
    def __init__(
        self,
        cfg: WatchdogConfig,
        backend: ClusterBackend | None = None,
        http_client: HTTPClient | None = None,
        *,
        clock: Callable[[], float] | None = None,
        sleeper: Callable[[float], None] | None = None,
    ):
        self.cfg = cfg
        self.backend = backend or SSHClusterBackend(cfg)
        self.http = http_client or UrllibHTTPClient()
        self.now = clock or time.time
        self.sleep = sleeper or time.sleep
        self.state_path = Path(cfg.state_dir) / "watchdog_state.json"
        self.state = self._load_state()
        if not self.state.active_env and cfg.primary_env:
            self.state.active_env = cfg.primary_env
        self._stop_requested = False

    def _load_state(self) -> SupervisorState:
        try:
            if self.state_path.is_file():
                data = json.loads(self.state_path.read_text(encoding="utf-8"))
                return SupervisorState.from_dict(data)
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to load watchdog state (%s); starting fresh", e)
        return SupervisorState(active_env=self.cfg.primary_env)

    def _save_state(self) -> None:
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.state.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
            os.replace(tmp, self.state_path)
        except OSError as e:
            logger.warning("Failed to save watchdog state to %s: %s", self.state_path, e)

    def reset_circuit_breaker(self) -> None:
        self.state.circuit_open = False
        self.state.circuit_opened_at = 0.0
        self.state.restart_timestamps = []
        self.state.consecutive_v15_boot_failures = 0
        self.state.consecutive_recoveries = 0
        self.state.consecutive_probe_failures = 0
        self.state.using_fallback = False
        if self.cfg.primary_env:
            self.state.active_env = self.cfg.primary_env
        self.state.status = "INIT"
        self._save_state()

    def _base_url(self) -> str:
        return f"http://{self.cfg.head_host}:{self.cfg.port}"

    # ------------------------------------------------------------------
    # Multi-tier Health Probes
    # ------------------------------------------------------------------

    def probe_http_health(self) -> HTTPResponse:
        return self.http.get(f"{self._base_url()}/health", timeout_s=self.cfg.health_timeout_s)

    def probe_canary(self) -> HTTPResponse:
        payload = {
            "model": self.cfg.served_model_name,
            "prompt": self.cfg.canary_prompt,
            "max_tokens": 8,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 0,
        }
        resp = self.http.post_json(f"{self._base_url()}/v1/completions", payload, timeout_s=self.cfg.canary_timeout_s)
        if not resp.ok:
            return resp
        try:
            parsed = json.loads(resp.body)
            choices = parsed.get("choices") or []
            text = (choices[0].get("text") if choices else "") or ""
            if not text.strip():
                return HTTPResponse(ok=False, status=resp.status, body=resp.body, latency_ms=resp.latency_ms, error="empty_canary_completion")
        except Exception as e:  # noqa: BLE001
            return HTTPResponse(ok=False, status=resp.status, body=resp.body, latency_ms=resp.latency_ms, error=f"invalid_canary_json:{e}")
        return resp

    def _probe_single_node(self, rank: int, host: str) -> NodeHealth:
        nh = NodeHealth(rank=rank, host=host)
        cname = f"{self.cfg.container_prefix}{rank}"
        probe_script = (
            f"echo '===INSPECT==='; docker inspect {shlex.quote(cname)} 2>/dev/null || echo '[]'; "
            f"echo '===IBSTAT==='; "
            f"if command -v ibstat >/dev/null 2>&1; then ibstat 2>/dev/null; "
            f"elif [ -d /sys/class/infiniband ]; then for d in /sys/class/infiniband/*; do "
            f"n=$(basename \"$d\"); s=$(cat \"$d/ports/1/state\" 2>/dev/null || echo DOWN); echo \"HCA $n $s\"; done; fi; "
            f"echo '===GPU==='; nvidia-smi --query-gpu=temperature.gpu,memory.used,memory.total,clocks_throttle_reasons.active,ecc.errors.uncorrected.volatile.total --format=csv,noheader,nounits 2>/dev/null || true; "
            f"echo '===XID==='; (dmesg -T 2>/dev/null || dmesg 2>/dev/null || true) | grep -E 'NVRM: Xid|Out of memory|Killed process' | tail -n 20 || true; "
            f"echo '===MEM==='; cat /proc/meminfo 2>/dev/null | grep -E '^(MemTotal|MemFree|MemAvailable|SwapTotal|SwapFree):' || true; "
            f"echo '===BUDDY==='; cat /proc/buddyinfo 2>/dev/null || true; "
            f"echo '===LOGS==='; docker logs --tail 40 {shlex.quote(cname)} 2>&1 || true"
        )
        res = self.backend.run(host, probe_script, timeout_s=self.cfg.ssh_timeout_s + 5)
        if res.rc != 0 and not res.stdout:
            nh.reachable = False
            nh.errors.append(f"ssh_unreachable:rc={res.rc}:{res.stderr.strip()[:120]}")
            return nh

        sections: dict[str, list[str]] = {}
        cur = ""
        for line in res.stdout.splitlines():
            if line.startswith("===") and line.endswith("==="):
                cur = line.strip("=")
                sections[cur] = []
            elif cur:
                sections[cur].append(line)

        # 1. Container state
        inspect_txt = "\n".join(sections.get("INSPECT", [])).strip()
        try:
            arr = json.loads(inspect_txt) if inspect_txt else []
            if arr:
                st = arr[0].get("State", {})
                nh.container_running = bool(st.get("Running", False))
                nh.container_status = str(st.get("Status", "unknown"))
                nh.oom_killed = bool(st.get("OOMKilled", False))
                nh.exit_code = int(st.get("ExitCode", 0))
            else:
                nh.container_running = False
                nh.container_status = "missing"
        except Exception as e:  # noqa: BLE001
            nh.container_running = False
            nh.container_status = f"inspect_err:{e}"

        if not nh.container_running:
            nh.errors.append(
                f"container_not_running:{cname}:status={nh.container_status}:oom={nh.oom_killed}:rc={nh.exit_code}"
            )
        elif nh.oom_killed:
            nh.errors.append(f"container_oom_killed:{cname}")

        # 2. RoCEv2 HCAs
        ib_txt = "\n".join(sections.get("IBSTAT", []))
        nh.active_hcas, nh.hca_details = parse_ibstat_text(ib_txt)
        if nh.active_hcas < self.cfg.required_active_hcas:
            nh.errors.append(
                f"roce_hcas_down:{nh.active_hcas}/{self.cfg.required_active_hcas}:[{','.join(nh.hca_details)}]"
            )

        # 3. GPU health
        gpu_info = parse_nvidia_smi_csv("\n".join(sections.get("GPU", [])))
        if gpu_info:
            nh.gpu_temp_c = gpu_info.get("temp_c")
            nh.gpu_throttle_mask = str(gpu_info.get("throttle_mask", "0x0"))
            nh.gpu_uncorrected_ecc = int(gpu_info.get("uncorrected_ecc", 0))
            if nh.gpu_temp_c is not None:
                if nh.gpu_temp_c >= self.cfg.gpu_temp_crit_c:
                    nh.errors.append(f"gpu_overtemp:{nh.gpu_temp_c:.1f}C>={self.cfg.gpu_temp_crit_c:.1f}C")
                elif nh.gpu_temp_c >= self.cfg.gpu_temp_warn_c:
                    nh.warnings.append(f"gpu_temp_high:{nh.gpu_temp_c:.1f}C")
            if nh.gpu_uncorrected_ecc > 0:
                nh.errors.append(f"gpu_uncorrected_ecc:{nh.gpu_uncorrected_ecc}")
            # Bit 0x40 (HW Thermal) or 0x20 (HW Slowdown) or 0x80 (HW Power Brake)
            try:
                tmask = int(nh.gpu_throttle_mask, 16)
                if tmask & 0xE0:
                    nh.warnings.append(f"gpu_hw_throttle:{nh.gpu_throttle_mask}")
            except ValueError:
                pass

        # 4. Kernel Xids (only NEW unseen Xids trigger recovery)
        seen_for_host = set(self.state.seen_xids.get(host, []))
        nh.fatal_xids = parse_fatal_xids("\n".join(sections.get("XID", [])), seen=seen_for_host)
        if nh.fatal_xids:
            nh.errors.append(f"fatal_gpu_xid:{nh.fatal_xids[0]}")

        # 5. UMA / Swap / Buddyinfo
        mem = parse_meminfo_text("\n".join(sections.get("MEM", [])))
        if mem:
            nh.mem_avail_gib = mem["mem_avail_gib"]
            nh.swap_used_mib = mem["swap_used_mib"]
            if nh.mem_avail_gib < self.cfg.runtime_min_mem_avail_gib:
                nh.errors.append(f"uma_mem_avail_crit:{nh.mem_avail_gib:.2f}GiB<{self.cfg.runtime_min_mem_avail_gib:.1f}GiB")
            elif nh.mem_avail_gib < self.cfg.runtime_warn_mem_avail_gib:
                nh.warnings.append(f"uma_mem_avail_low:{nh.mem_avail_gib:.2f}GiB")
            if nh.swap_used_mib > self.cfg.runtime_crit_swap_mib:
                nh.errors.append(f"uma_swap_crit:{nh.swap_used_mib:.1f}MiB>{self.cfg.runtime_crit_swap_mib:.0f}MiB")
            elif nh.swap_used_mib > self.cfg.runtime_warn_swap_mib:
                nh.warnings.append(f"uma_swap_high:{nh.swap_used_mib:.1f}MiB")

        buddy = parse_buddyinfo_text("\n".join(sections.get("BUDDY", [])))
        if buddy:
            nh.order_13_blocks = buddy["order_13_blocks"]
            nh.order_ge9_gib = buddy["order_ge9_gib"]

        # 6. Rank log fatal pattern scan
        log_txt = "\n".join(sections.get("LOGS", []))
        for pat in FATAL_LOG_PATTERNS:
            m = pat.search(log_txt)
            if m:
                nh.log_fatal_hits.append(m.group(0)[:120])
        if nh.log_fatal_hits and not nh.container_running:
            nh.errors.append(f"fatal_log:{nh.log_fatal_hits[0]}")
        elif nh.log_fatal_hits:
            nh.warnings.append(f"fatal_log_pattern:{nh.log_fatal_hits[0]}")

        return nh

    def run_health_check(self, *, force_canary: bool = False) -> ClusterHealthReport:
        ts = self.now()
        report = ClusterHealthReport(timestamp=ts, status="HEALTHY", http_health_ok=False)

        # Tier 1: /health
        h_resp = self.probe_http_health()
        report.http_health_ok = h_resp.ok
        report.http_health_latency_ms = round(h_resp.latency_ms, 2)
        if not h_resp.ok:
            report.reasons.append(f"http_health_failed:status={h_resp.status}:{h_resp.error}")

        # Tier 2: Canary completion probe
        due_canary = force_canary or (ts - self.state.last_canary_ts >= self.cfg.canary_interval_s)
        if due_canary and h_resp.ok:
            report.canary_ran = True
            self.state.last_canary_ts = ts
            c_resp = self.probe_canary()
            report.canary_ok = c_resp.ok
            report.canary_latency_ms = round(c_resp.latency_ms, 2)
            report.canary_error = c_resp.error
            if not c_resp.ok:
                report.reasons.append(f"canary_failed:status={c_resp.status}:{c_resp.error}")

        # Tier 3: Parallel 8-node probe
        with ThreadPoolExecutor(max_workers=len(self.cfg.hosts)) as pool:
            futs = [pool.submit(self._probe_single_node, r, h) for r, h in enumerate(self.cfg.hosts)]
            nodes = [f.result() for f in futs]
        report.nodes = nodes

        for nh in nodes:
            for w in nh.warnings:
                report.warnings.append(f"r{nh.rank}({nh.host}):{w}")
            for e in nh.errors:
                report.reasons.append(f"r{nh.rank}({nh.host}):{e}")
                # Confirmed dead container, OOMKill, RoCEv2 link down, or fatal new GPU Xid -> immediate recovery!
                if e.startswith(("container_not_running:", "container_oom_killed:", "roce_hcas_down:", "fatal_gpu_xid:", "gpu_uncorrected_ecc:")):
                    report.immediate_recovery = True

        if report.reasons:
            report.status = "UNHEALTHY"
        elif report.warnings:
            report.status = "DEGRADED"
        else:
            report.status = "HEALTHY"
        return report

    # ------------------------------------------------------------------
    # Diagnostic Crash Bundle Capture
    # ------------------------------------------------------------------

    def capture_crash_bundle(self, reason: str, report: ClusterHealthReport | None = None) -> Path:
        ts_str = datetime.datetime.fromtimestamp(self.now(), tz=datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
        base_dir = Path(self.cfg.log_dir)
        bundle_dir = base_dir / f"crash-{ts_str}"
        idx = 1
        while bundle_dir.exists():
            bundle_dir = base_dir / f"crash-{ts_str}-{idx}"
            idx += 1
        bundle_dir.mkdir(parents=True, exist_ok=True)

        def _capture_node(rank: int, host: str) -> dict[str, Any]:
            cname = f"{self.cfg.container_prefix}{rank}"
            items = {
                "docker.log": f"docker logs --tail 200 {shlex.quote(cname)} 2>&1 || true",
                "inspect.json": f"docker inspect {shlex.quote(cname)} 2>&1 || true",
                "dmesg.txt": "(dmesg -T 2>/dev/null || dmesg 2>/dev/null || true) | grep -E 'NVRM|Xid|oom|Out of memory|mlx5|rdma|Killed process' | tail -n 120 || true",
                "buddyinfo.txt": "echo '--- /proc/buddyinfo ---'; cat /proc/buddyinfo 2>/dev/null || true; echo '--- /proc/meminfo ---'; cat /proc/meminfo 2>/dev/null || true",
                "nvidia_smi.txt": "nvidia-smi 2>&1 || true",
                "ibstat.txt": "ibstat 2>&1 || (for d in /sys/class/infiniband/*; do echo \"=== $d ===\"; cat \"$d/ports/1/state\" \"$d/ports/1/phys_state\" 2>/dev/null; done) || true",
            }
            node_meta: dict[str, Any] = {"rank": rank, "host": host, "files": {}, "errors": []}
            for suffix, cmd in items.items():
                res = self.backend.run(host, cmd, timeout_s=min(15, self.cfg.ssh_timeout_s + 5))
                fname = f"rank-{rank}_{host}_{suffix}"
                fpath = bundle_dir / fname
                content = res.stdout if res.stdout else res.stderr
                if res.timed_out:
                    node_meta["errors"].append(f"{suffix}:timeout")
                    content = (content + "\n[CAPTURE TIMEOUT]\n").strip()
                elif res.rc != 0 and not res.stdout:
                    node_meta["errors"].append(f"{suffix}:rc={res.rc}")
                fpath.write_text(content, encoding="utf-8", errors="replace")
                node_meta["files"][suffix] = fname
            return node_meta

        with ThreadPoolExecutor(max_workers=len(self.cfg.hosts)) as pool:
            futs = [pool.submit(_capture_node, r, h) for r, h in enumerate(self.cfg.hosts)]
            node_summaries = [f.result() for f in futs]

        # Record any fatal Xids we captured so historical dmesg entries don't loop after restart
        if report is not None:
            for nh in report.nodes:
                if nh.fatal_xids:
                    existing = list(self.state.seen_xids.get(nh.host, []))
                    for hit in nh.fatal_xids:
                        sig = hit.split(":", 1)[1] if ":" in hit else hit
                        if sig not in existing:
                            existing.append(sig)
                    self.state.seen_xids[nh.host] = existing[-50:]

        summary = {
            "timestamp_utc": ts_str,
            "reason": reason,
            "active_env": self.state.active_env,
            "using_fallback": self.state.using_fallback,
            "consecutive_v15_boot_failures": self.state.consecutive_v15_boot_failures,
            "restart_count_window": len(self._recent_restarts(self.now())),
            "health_report": asdict(report) if report is not None else None,
            "nodes": node_summaries,
        }
        (bundle_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        self.state.last_crash_bundle = str(bundle_dir)
        self._save_state()
        logger.warning("Captured 8-node diagnostic crash bundle -> %s", bundle_dir)
        return bundle_dir

    # ------------------------------------------------------------------
    # Coordinated 8-Node Teardown, IPC Sweep & Pre-Relaunch Checks
    # ------------------------------------------------------------------

    def teardown_and_sweep(self) -> dict[str, bool]:
        """Stop & rm all 8 containers and sweep stale inkling/vllm/torch/RDMA IPC files."""
        results: dict[str, bool] = {}

        def _clean_node(rank: int, host: str) -> tuple[str, bool]:
            cname = f"{self.cfg.container_prefix}{rank}"
            cmd = (
                f"docker stop -t {int(self.cfg.stop_timeout_s)} {shlex.quote(cname)} >/dev/null 2>&1 || true; "
                f"docker rm -f {shlex.quote(cname)} >/dev/null 2>&1 || true; "
                f"rm -rf /dev/shm/vllm* /dev/shm/inkling* /dev/shm/torch_* /dev/shm/nccl-* "
                f"/dev/shm/psm_* /dev/shm/rdma_ar* /tmp/vllm* /tmp/torchinductor_* /tmp/inkling_rdma_*.lock 2>/dev/null || true; "
                f"echo CLEAN_OK"
            )
            res = self.backend.run(host, cmd, timeout_s=self.cfg.stop_timeout_s + 15)
            return host, (res.rc == 0 and "CLEAN_OK" in res.stdout)

        with ThreadPoolExecutor(max_workers=len(self.cfg.hosts)) as pool:
            futs = [pool.submit(_clean_node, r, h) for r, h in enumerate(self.cfg.hosts)]
            for f in futs:
                h, ok = f.result()
                results[h] = ok
        return results

    def prelaunch_roce_and_uma_check(self) -> tuple[bool, list[str]]:
        """Verify 4/4 RoCEv2 HCAs Active & clean UMA buddy pools on all 8 nodes before relaunch."""
        issues: list[str] = []

        def _check_node(rank: int, host: str) -> list[str]:
            node_issues: list[str] = []
            cmd = (
                "echo '===IBSTAT==='; "
                "if command -v ibstat >/dev/null 2>&1; then ibstat 2>/dev/null; "
                "elif [ -d /sys/class/infiniband ]; then for d in /sys/class/infiniband/*; do "
                "n=$(basename \"$d\"); s=$(cat \"$d/ports/1/state\" 2>/dev/null || echo DOWN); echo \"HCA $n $s\"; done; fi; "
                "echo '===MEM==='; cat /proc/meminfo 2>/dev/null | grep -E '^(MemTotal|MemFree|MemAvailable|SwapTotal|SwapFree):' || true; "
                "echo '===BUDDY==='; cat /proc/buddyinfo 2>/dev/null || true"
            )
            res = self.backend.run(host, cmd, timeout_s=self.cfg.ssh_timeout_s + 5)
            if res.rc != 0 and not res.stdout:
                return [f"r{rank}({host}):ssh_unreachable"]

            sections: dict[str, list[str]] = {}
            cur = ""
            for line in res.stdout.splitlines():
                if line.startswith("===") and line.endswith("==="):
                    cur = line.strip("=")
                    sections[cur] = []
                elif cur:
                    sections[cur].append(line)

            active_hcas, hca_details = parse_ibstat_text("\n".join(sections.get("IBSTAT", [])))
            if active_hcas < self.cfg.required_active_hcas:
                node_issues.append(
                    f"r{rank}({host}):roce_link_down:{active_hcas}/{self.cfg.required_active_hcas}:[{','.join(hca_details)}]"
                )
                return node_issues  # Do not attempt relaunch with dead RoCEv2 links!

            mem = parse_meminfo_text("\n".join(sections.get("MEM", [])))
            buddy = parse_buddyinfo_text("\n".join(sections.get("BUDDY", [])))
            needs_defrag = False
            if mem and mem["swap_used_mib"] > self.cfg.preboot_max_swap_mib:
                needs_defrag = True
            if buddy and buddy["max_order"] >= 13 and buddy["order_13_blocks"] < self.cfg.preboot_min_order13_blocks:
                needs_defrag = True

            if needs_defrag:
                logger.warning("Pre-relaunch UMA defrag triggered on %s (swap=%s, o13=%s)", host, mem.get("swap_used_mib"), buddy.get("order_13_blocks"))
                defrag_cmd = (
                    "sync && sudo -n sysctl -w vm.drop_caches=3 >/dev/null 2>&1 || true; "
                    "sudo -n sysctl -w vm.compact_memory=1 >/dev/null 2>&1 || true; "
                    "if awk '/SwapTotal/{t=$2}/SwapFree/{f=$2} END{exit !(t-f>0)}' /proc/meminfo 2>/dev/null; then "
                    "  sudo -n swapoff -a && sudo -n swapon -a || true; "
                    "fi; "
                    "echo '===MEM==='; cat /proc/meminfo 2>/dev/null | grep -E '^(MemTotal|MemFree|MemAvailable|SwapTotal|SwapFree):' || true; "
                    "echo '===BUDDY==='; cat /proc/buddyinfo 2>/dev/null || true"
                )
                dres = self.backend.run(host, defrag_cmd, timeout_s=45)
                dsec: dict[str, list[str]] = {}
                dcur = ""
                for line in dres.stdout.splitlines():
                    if line.startswith("===") and line.endswith("==="):
                        dcur = line.strip("=")
                        dsec[dcur] = []
                    elif dcur:
                        dsec[dcur].append(line)
                buddy2 = parse_buddyinfo_text("\n".join(dsec.get("BUDDY", [])))
                if buddy2 and buddy2["max_order"] >= 13 and buddy2["order_13_blocks"] < self.cfg.preboot_min_order13_blocks:
                    logger.warning(
                        "Post-defrag order-13 on %s is %d (< %d); reboot-defrag recommended (continuing relaunch unless strict)",
                        host, buddy2["order_13_blocks"], self.cfg.preboot_min_order13_blocks,
                    )
            return node_issues

        with ThreadPoolExecutor(max_workers=len(self.cfg.hosts)) as pool:
            futs = [pool.submit(_check_node, r, h) for r, h in enumerate(self.cfg.hosts)]
            for f in futs:
                issues.extend(f.result())
        return (len(issues) == 0), issues

    # ------------------------------------------------------------------
    # Atomic Ranks 7..0 Relaunch & 1-Prompt 0-ULP Boot Certification
    # ------------------------------------------------------------------

    def relaunch_cluster(self, env_file: str) -> tuple[bool, list[str]]:
        """Sync env file to all 8 nodes and atomically launch headless ranks 7..1 first, then rank 0."""
        errors: list[str] = []
        if not env_file or not os.path.isfile(env_file):
            return False, [f"env_file_missing:{env_file}"]

        # 1. Distribute active env file to all 8 nodes in parallel
        with ThreadPoolExecutor(max_workers=len(self.cfg.hosts)) as pool:
            futs = [
                (r, h, pool.submit(self.backend.copy_file, env_file, h, self.cfg.remote_env_path, 20))
                for r, h in enumerate(self.cfg.hosts)
            ]
            for r, h, f in futs:
                res = f.result()
                if res.rc != 0:
                    errors.append(f"scp_failed:r{r}({h}):{res.stderr.strip()}")
        if errors:
            return False, errors

        def _launch_rank(rank: int, host: str) -> tuple[int, str, CmdResult]:
            cmd = (
                f"bash -lc 'for v in $(compgen -e NCCL_); do unset \"$v\"; done; "
                f"set -a; source {shlex.quote(self.cfg.remote_env_path)}; set +a; "
                f"bash {shlex.quote(self.cfg.remote_launcher)} {rank}'"
            )
            return rank, host, self.backend.run(host, cmd, timeout_s=45)

        # 2. Launch worker ranks 7..1 FIRST (reverse order, parallel across workers)
        worker_order = list(range(len(self.cfg.hosts) - 1, 0, -1))
        with ThreadPoolExecutor(max_workers=max(1, len(worker_order))) as pool:
            futs = [pool.submit(_launch_rank, r, self.cfg.hosts[r]) for r in worker_order]
            for f in futs:
                r, h, res = f.result()
                if res.rc != 0:
                    errors.append(f"launch_failed:r{r}({h}):rc={res.rc}:{res.stderr.strip()[:160]}")
        if errors:
            return False, errors

        # 3. Launch head rank 0 LAST so workers are already booting / ready for rendezvous
        _, h0, res0 = _launch_rank(0, self.cfg.hosts[0])
        if res0.rc != 0:
            errors.append(f"launch_failed:r0({h0}):rc={res0.rc}:{res0.stderr.strip()[:160]}")
            return False, errors
        return True, []

    def certify_boot(self) -> tuple[bool, str, str]:
        """Fast 1-prompt 0-ULP boot certification check before marking READY.

        Returns (ok, sha256_hex, detail).
        """
        payload = {
            "model": self.cfg.served_model_name,
            "prompt": self.cfg.boot_cert_prompt,
            "max_tokens": self.cfg.boot_cert_max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 0,
        }
        url = f"{self._base_url()}/v1/completions"
        resp1 = self.http.post_json(url, payload, timeout_s=max(30.0, self.cfg.canary_timeout_s))
        if not resp1.ok:
            return False, "", f"cert_http_error:status={resp1.status}:{resp1.error}"
        try:
            text1 = (json.loads(resp1.body).get("choices") or [{}])[0].get("text", "")
        except Exception as e:  # noqa: BLE001
            return False, "", f"cert_json_error:{e}"
        if not text1 or not text1.strip():
            return False, "", "cert_empty_output"
        if "nan" in text1.lower() and len(text1.strip()) <= 8:
            return False, "", f"cert_nan_output:{text1!r}"

        digest1 = hashlib.sha256(text1.encode("utf-8")).hexdigest()
        expected = self.cfg.boot_cert_expected_sha256.strip().lower()
        if expected:
            if not digest1.startswith(expected):
                return False, digest1, f"cert_0ulp_digest_mismatch:got={digest1[:16]} expected={expected[:16]}"
            return True, digest1, f"cert_0ulp_golden_match:{digest1[:16]}"

        if self.cfg.boot_cert_repeat_check:
            resp2 = self.http.post_json(url, payload, timeout_s=max(30.0, self.cfg.canary_timeout_s))
            if not resp2.ok:
                return False, digest1, f"cert_repeat_http_error:{resp2.error}"
            try:
                text2 = (json.loads(resp2.body).get("choices") or [{}])[0].get("text", "")
            except Exception as e:  # noqa: BLE001
                return False, digest1, f"cert_repeat_json_error:{e}"
            digest2 = hashlib.sha256(text2.encode("utf-8")).hexdigest()
            if digest2 != digest1:
                return False, digest1, f"cert_nondeterministic:{digest1[:16]}!={digest2[:16]}"
        return True, digest1, f"cert_0ulp_repeatable:{digest1[:16]}"

    def wait_and_certify_boot(self) -> tuple[bool, str]:
        """Wait for /health (failing fast if any rank dies during boot), then run 1-prompt 0-ULP cert."""
        deadline = self.now() + self.cfg.boot_timeout_s
        while self.now() < deadline:
            # Fail fast if any rank already died/OOMKilled during boot
            with ThreadPoolExecutor(max_workers=len(self.cfg.hosts)) as pool:
                futs = [
                    (r, h, pool.submit(
                        self.backend.run,
                        h,
                        f"docker inspect -f '{{{{.State.Running}}}}|{{{{.State.Status}}}}|{{{{.State.OOMKilled}}}}' {shlex.quote(f'{self.cfg.container_prefix}{r}')} 2>/dev/null || echo 'false|missing|false'",
                        10,
                    ))
                    for r, h in enumerate(self.cfg.hosts)
                ]
                for r, h, f in futs:
                    res = f.result()
                    parts = res.stdout.strip().split("|")
                    if len(parts) >= 3 and parts[0].lower() != "true":
                        return False, f"rank_died_during_boot:r{r}({h}):status={parts[1]}:oom={parts[2]}"

            h_resp = self.probe_http_health()
            if h_resp.ok:
                ok, digest, detail = self.certify_boot()
                if ok:
                    self.state.last_boot_cert_sha256 = digest
                    return True, detail
                return False, detail
            self.sleep(self.cfg.boot_poll_interval_s)
        return False, f"boot_health_timeout_after_{self.cfg.boot_timeout_s}s"

    # ------------------------------------------------------------------
    # Circuit Breaker, Backoff & Fallback Orchestration
    # ------------------------------------------------------------------

    def _recent_restarts(self, now_ts: float) -> list[float]:
        cutoff = now_ts - self.cfg.breaker_window_s
        return [t for t in self.state.restart_timestamps if t >= cutoff]

    def _send_alert(self, severity: str, message: str) -> None:
        logger.error("ALERT[%s]: %s", severity, message)
        if not self.cfg.alert_cmd:
            return
        try:
            env = dict(os.environ)
            env["V15_ALERT_SEVERITY"] = severity
            env["V15_ALERT_MESSAGE"] = message
            subprocess.run(self.cfg.alert_cmd, shell=True, env=env, timeout=15, check=False)
        except Exception as e:  # noqa: BLE001
            logger.warning("Alert hook failed: %s", e)

    def _acquire_lock(self):
        lock_path = Path(self.cfg.lock_file)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(lock_path, "w", encoding="utf-8")
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            fh.write(f"{os.getpid()}\n")
            fh.flush()
            return fh
        except BlockingIOError:
            fh.close()
            return None

    def _record_v15_failure_and_maybe_fallback(self) -> bool:
        """Increment V15 consecutive failure count; switch to live_champion_v12.env after 2 fails."""
        if self.state.using_fallback:
            return False
        self.state.consecutive_v15_boot_failures += 1
        if (
            self.state.consecutive_v15_boot_failures >= self.cfg.fallback_after_v15_failures
            and self.cfg.fallback_env
            and os.path.isfile(self.cfg.fallback_env)
            and self.cfg.fallback_env != self.state.active_env
        ):
            self.state.active_env = self.cfg.fallback_env
            self.state.using_fallback = True
            self._send_alert(
                "WARNING",
                f"V15 failed {self.state.consecutive_v15_boot_failures} consecutive boots; "
                f"AUTOMATICALLY FALLING BACK to verified champion env: {self.cfg.fallback_env}",
            )
            self._save_state()
            return True
        self._save_state()
        return False

    def recover_cluster(self, reason: str, report: ClusterHealthReport | None = None) -> tuple[bool, str]:
        """Full coordinated 8-node crash recovery with circuit breaker, backoff, and V12 fallback."""
        lock_fh = self._acquire_lock()
        if lock_fh is None:
            return False, "recovery_already_in_progress"
        try:
            now_ts = self.now()
            self.state.restart_timestamps = self._recent_restarts(now_ts)

            # Always capture forensic crash bundle BEFORE deciding circuit-breaker / teardown!
            bundle = self.capture_crash_bundle(reason=reason, report=report)

            # Check circuit breaker (max 3 restarts / 15 min)
            if len(self.state.restart_timestamps) >= self.cfg.max_restarts_in_window:
                self.state.circuit_open = True
                self.state.circuit_opened_at = now_ts
                self.state.status = "CIRCUIT_OPEN"
                self._save_state()
                msg = (
                    f"Circuit breaker OPEN: {len(self.state.restart_timestamps)} restarts in "
                    f"{int(self.cfg.breaker_window_s)}s (max={self.cfg.max_restarts_in_window}). "
                    f"Crash bundle: {bundle}. Operator action required."
                )
                self._send_alert("CRITICAL", msg)
                return False, "circuit_breaker_open"

            # If the previous READY boot crashed within boot_stability_window_s, count as a V15 boot failure!
            if (
                not self.state.using_fallback
                and self.state.last_boot_ts > 0.0
                and (now_ts - self.state.last_boot_ts) < self.cfg.boot_stability_window_s
            ):
                self.state.last_boot_ts = 0.0
                self._record_v15_failure_and_maybe_fallback()

            # Exponential backoff before relaunch
            if self.state.consecutive_recoveries > 0:
                delay = min(
                    self.cfg.backoff_max_s,
                    self.cfg.backoff_base_s * (2 ** min(self.state.consecutive_recoveries - 1, 6)),
                )
                logger.info("Recovery backoff: sleeping %.1fs before relaunch...", delay)
                self.sleep(delay)

            self.state.status = "RECOVERING"
            self.state.restart_timestamps.append(self.now())
            self.state.consecutive_recoveries += 1
            self._save_state()

            # Step 1: Coordinated 8-node teardown + IPC sweep
            logger.info("Step 1/4: Tearing down 8/8 containers & sweeping /dev/shm + /tmp/vllm*...")
            self.teardown_and_sweep()

            # Step 2: Pre-relaunch RoCEv2 4/4 Active & UMA defrag guard
            logger.info("Step 2/4: Verifying 4/4 RoCEv2 Active & UMA cleanliness across 8/8 nodes...")
            pre_ok, pre_issues = self.prelaunch_roce_and_uma_check()
            if not pre_ok:
                msg = f"prelaunch_blocked:{';'.join(pre_issues)}"
                logger.error("Refusing blind relaunch: %s", msg)
                self._save_state()
                return False, msg

            # Step 3: Relaunch ranks 7..0 atomically with active env (V15 or fallback V12)
            env_to_boot = self.state.active_env or self.cfg.primary_env or self.cfg.fallback_env
            logger.info("Step 3/4: Atomically relaunching ranks 7..0 with env=%s...", env_to_boot)
            launch_ok, launch_errs = self.relaunch_cluster(env_to_boot)
            if not launch_ok:
                switched = self._record_v15_failure_and_maybe_fallback()
                return False, f"relaunch_failed:{';'.join(launch_errs)}:switched_fallback={switched}"

            # Step 4: Wait for /health + 1-prompt 0-ULP boot certification
            logger.info("Step 4/4: Waiting for /health + 1-prompt 0-ULP boot certification...")
            cert_ok, cert_detail = self.wait_and_certify_boot()
            if not cert_ok:
                self.state.last_boot_ts = 0.0
                switched = self._record_v15_failure_and_maybe_fallback()
                logger.error("Boot certification FAILED (%s); switched_fallback=%s", cert_detail, switched)
                return False, f"boot_cert_failed:{cert_detail}:switched_fallback={switched}"

            # Boot succeeded & certified! (consecutive_v15_boot_failures clears only after boot_stability_window_s!)
            ready_ts = self.now()
            self.state.last_boot_ts = ready_ts
            self.state.status = "READY_FALLBACK" if self.state.using_fallback else "READY"
            self.state.consecutive_probe_failures = 0
            self.state.healthy_since_ts = ready_ts
            self._save_state()
            logger.info("Cluster READY (%s): %s", self.state.status, cert_detail)
            return True, f"recovered:{self.state.status}:{cert_detail}"
        finally:
            try:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
            lock_fh.close()

    def step(self, *, auto_recover: bool = True, force_canary: bool = False) -> tuple[ClusterHealthReport, str]:
        """Execute one complete watchdog probe + optional self-healing cycle."""
        report = self.run_health_check(force_canary=force_canary)
        now_ts = self.now()

        if report.status in ("HEALTHY", "DEGRADED"):
            self.state.consecutive_probe_failures = 0
            if self.state.healthy_since_ts <= 0.0:
                self.state.healthy_since_ts = now_ts
            else:
                healthy_for = now_ts - self.state.healthy_since_ts
                if not self.state.using_fallback and healthy_for >= self.cfg.boot_stability_window_s:
                    self.state.consecutive_v15_boot_failures = 0
                if healthy_for >= self.cfg.stability_reset_s:
                    self.state.consecutive_recoveries = 0
            if self.state.status in ("INIT", "RECOVERING"):
                self.state.status = "READY_FALLBACK" if self.state.using_fallback else "READY"
            self._save_state()
            sd_notify(f"WATCHDOG=1\nSTATUS={self.state.status} ({report.status})")
            return report, "ok"

        # UNHEALTHY
        self.state.healthy_since_ts = 0.0
        self.state.consecutive_probe_failures += 1
        self._save_state()

        if not auto_recover:
            return report, "unhealthy_no_recover"

        if self.state.circuit_open:
            return report, "circuit_breaker_open"

        should_recover = report.immediate_recovery or (
            self.state.consecutive_probe_failures >= self.cfg.fail_threshold
        )
        if not should_recover:
            logger.warning(
                "Health probe failed (%d/%d before recovery): %s",
                self.state.consecutive_probe_failures,
                self.cfg.fail_threshold,
                "; ".join(report.reasons),
            )
            return report, "debouncing"

        reason = "; ".join(report.reasons[:5])
        ok, detail = self.recover_cluster(reason=reason, report=report)
        return report, detail

    def run_daemon(self) -> int:
        def _sig_stop(signum: int, _frame: Any) -> None:
            logger.info("Watchdog daemon received signal %d; exiting cleanly", signum)
            self._stop_requested = True

        signal.signal(signal.SIGTERM, _sig_stop)
        signal.signal(signal.SIGINT, _sig_stop)
        sd_notify("READY=1\nSTATUS=v15_health_watchdog running")
        logger.info(
            "Starting V15 8-node health watchdog (hosts=%d, interval=%.1fs, canary=%.1fs, max_restarts=%d/%ds)",
            len(self.cfg.hosts),
            self.cfg.probe_interval_s,
            self.cfg.canary_interval_s,
            self.cfg.max_restarts_in_window,
            int(self.cfg.breaker_window_s),
        )
        while not self._stop_requested:
            try:
                report, action = self.step(auto_recover=True)
                logger.info("Watchdog tick: status=%s action=%s reasons=%s", report.status, action, report.reasons[:3])
            except Exception as e:  # noqa: BLE001
                logger.exception("Unhandled watchdog tick exception: %s", e)
            if not self._stop_requested:
                self.sleep(self.cfg.probe_interval_s)
        return 0


def _parse_jump_map(raw: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for tok in (raw or "").split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def build_config_from_args(args: argparse.Namespace) -> WatchdogConfig:
    cfg = WatchdogConfig()
    if args.hosts:
        cfg.hosts = tuple(args.hosts.split())
    elif os.environ.get("V15_HOSTS"):
        cfg.hosts = tuple(os.environ["V15_HOSTS"].split())
    if os.environ.get("V15_SSH_JUMP_MAP"):
        cfg.ssh_jump_map = _parse_jump_map(os.environ["V15_SSH_JUMP_MAP"])
    if args.primary_env:
        cfg.primary_env = args.primary_env
    if args.fallback_env:
        cfg.fallback_env = args.fallback_env
    if args.log_dir:
        cfg.log_dir = args.log_dir
    if args.state_dir:
        cfg.state_dir = args.state_dir
    if args.head_host:
        cfg.head_host = args.head_host
    if args.port:
        cfg.port = args.port
    if args.interval is not None:
        cfg.probe_interval_s = args.interval
    if args.canary_interval is not None:
        cfg.canary_interval_s = args.canary_interval
    return cfg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Wave-V15 8-Node DGX Spark Health Watchdog & Crash Recovery")
    parser.add_argument("--daemon", action="store_true", help="Run continuous monitoring & auto-recovery loop")
    parser.add_argument("--once", action="store_true", help="Run a single probe cycle and exit")
    parser.add_argument("--auto-recover", action="store_true", help="Allow recovery during --once mode")
    parser.add_argument("--force-canary", action="store_true", help="Always run Tier-2 canary probe")
    parser.add_argument("--certify-only", action="store_true", help="Run fast 1-prompt 0-ULP boot certification only")
    parser.add_argument("--capture-bundle", metavar="REASON", help="Capture 8-node diagnostic crash bundle now")
    parser.add_argument("--reset-breaker", action="store_true", help="Reset circuit breaker & fallback state")
    parser.add_argument("--hosts", help="Space-separated host list (default spark-one..spark-eight)")
    parser.add_argument("--head-host", help="Rank-0 API host for HTTP/canary probes (default 127.0.0.1)")
    parser.add_argument("--port", type=int, help="vLLM API port (default 8010)")
    parser.add_argument("--primary-env", help="Primary V15 env file path")
    parser.add_argument("--fallback-env", help="Verified fallback live_champion_v12.env path")
    parser.add_argument("--log-dir", help="Crash bundle root dir (default /var/log/inkling-v15)")
    parser.add_argument("--state-dir", help="Persistent state dir (default /var/lib/inkling-v15)")
    parser.add_argument("--interval", type=float, help="Probe interval seconds")
    parser.add_argument("--canary-interval", type=float, help="Canary probe interval seconds")
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        stream=sys.stderr if args.json else sys.stdout,
    )
    cfg = build_config_from_args(args)
    wd = ClusterWatchdog(cfg)

    if args.reset_breaker:
        wd.reset_circuit_breaker()
        print("Circuit breaker and fallback state reset.")
        return 0
    if args.certify_only:
        ok, digest, detail = wd.certify_boot()
        if args.json:
            print(json.dumps({"ok": ok, "sha256": digest, "detail": detail}))
        else:
            print(f"[certify] ok={ok} sha256={digest} detail={detail}")
        return 0 if ok else 1
    if args.capture_bundle:
        bundle = wd.capture_crash_bundle(reason=args.capture_bundle)
        print(str(bundle))
        return 0
    if args.daemon:
        return wd.run_daemon()

    report, action = wd.step(auto_recover=args.auto_recover, force_canary=args.force_canary)
    if args.json:
        print(json.dumps({"report": asdict(report), "action": action, "state": wd.state.to_dict()}, indent=2))
    else:
        print(f"[watchdog] status={report.status} action={action} reasons={report.reasons}")
    return 0 if report.status in ("HEALTHY", "DEGRADED") or action.startswith("recovered:") else 1


if __name__ == "__main__":
    raise SystemExit(main())
