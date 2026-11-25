#!/usr/bin/env python3
"""spark-top: Live 8-node DGX Spark cluster HUD for Inkling-2-Small TP8 (V17).

Shows at a glance (single command, zero external deps):
  - Live vLLM throughput (agg & per-stream tok/s), running/waiting streams,
    speculative decode AL (acceptance length), prefix cache hit %, KV pool %
  - Live InklingScheduler V17 counters (pure_decode, pure_prefill, short_skip, hold)
  - Active TCP connections to :8010 grouped by client IP
  - 8-node table: container state, GPU clock/temp/power/util, UMA MemAvailable,
    SwapUsed, buddyinfo order_13, 4/4 CX-7 IB link state & RoCE error counters

Usage:
  python3 spark-top.py            # refresh every 2s
  python3 spark-top.py --once     # single snapshot and exit
  python3 spark-top.py --interval 5 --port 8010
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.request

NODES = [
    ("spark-one", 0),
    ("spark-two", 1),
    ("spark-three", 2),
    ("spark-four", 3),
    ("spark-five", 4),
    ("spark-six", 5),
    ("spark-seven", 6),
    ("spark-eight", 7),
]

NODE_PROBE_CMD = r"""
python3 - << 'PY'
import glob, json, os, subprocess

out = {"host": os.uname().nodename}
# 1. MemAvailable & Swap
mem = {}
try:
    for line in open("/proc/meminfo"):
        k, v = line.split(":", 1)
        mem[k] = int(v.strip().split()[0])
except Exception:
    pass
out["mem_avail_gib"] = round(mem.get("MemAvailable", 0) / 1048576.0, 2)
out["swap_used_mib"] = round((mem.get("SwapTotal", 0) - mem.get("SwapFree", 0)) / 1024.0, 1)

# 2. buddyinfo order_13 (last column of Normal zone)
o13 = 0
try:
    for line in open("/proc/buddyinfo"):
        if "Normal" in line:
            cols = line.split()
            if len(cols) >= 18:
                o13 += int(cols[-1])
except Exception:
    pass
out["order13"] = o13

# 3. GPU metrics
try:
    s = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=temperature.gpu,clocks.current.sm,utilization.gpu,power.draw",
         "--format=csv,noheader,nounits"],
        timeout=3, text=True, stderr=subprocess.DEVNULL
    ).strip().splitlines()[0]
    parts = [p.strip() for p in s.split(",")]
    out["gpu_temp"] = parts[0]
    out["gpu_clk"] = parts[1]
    out["gpu_util"] = parts[2]
    out["gpu_pwr"] = parts[3]
except Exception:
    out["gpu_temp"] = out["gpu_clk"] = out["gpu_util"] = out["gpu_pwr"] = "?"

# 4. 4/4 IB HCA active & error counters
active = 0
total = 0
errs = 0
for stf in sorted(glob.glob("/sys/class/infiniband/*/ports/1/state")):
    total += 1
    try:
        if "ACTIVE" in open(stf).read():
            active += 1
    except Exception:
        pass
    base = os.path.dirname(stf)
    for cname in ("port_rcv_errors", "port_xmit_discards", "symbol_error", "link_downed"):
        cf = os.path.join(base, "counters", cname)
        try:
            errs += int(open(cf).read().strip())
        except Exception:
            pass
out["hca_active"] = f"{active}/{total}"
out["ib_errs"] = errs

# 5. Docker container status
try:
    ps = subprocess.check_output(
        ["docker", "ps", "--filter", "name=inkling-tp8-r", "--format", "{{.Names}}:{{.Status}}"],
        timeout=3, text=True, stderr=subprocess.DEVNULL
    ).strip()
    out["container"] = ps or "DOWN"
except Exception:
    out["container"] = "ERR"

print(json.dumps(out))
PY
"""


def fetch_metrics(port: int) -> dict[str, float]:
    out: dict[str, float] = {}
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=2.0) as resp:
            text = resp.read().decode("utf-8", errors="ignore")
    except Exception:
        return out
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^([a-zA-Z0-9_:]+)(?:\{[^}]*\})?\s+([-+eE0-9\.]+)$", line.strip())
        if not m:
            continue
        k, v = m.group(1), m.group(2)
        try:
            fv = float(v)
        except ValueError:
            continue
        out[k] = out.get(k, 0.0) + fv
    return out


def fetch_clients(port: int) -> dict[str, int]:
    counts: dict[str, int] = {}
    try:
        txt = subprocess.check_output(
            ["ss", "-tn", f"sport = :{port}"], timeout=2, text=True, stderr=subprocess.DEVNULL
        )
        for line in txt.splitlines()[1:]:
            cols = line.split()
            if len(cols) >= 5 and cols[0] == "ESTAB":
                peer = cols[4]
                ip = peer.rsplit(":", 1)[0].strip("[]")
                counts[ip] = counts.get(ip, 0) + 1
    except Exception:
        pass
    return counts


def read_json(path: str) -> dict:
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return {}


def probe_node(node: str, rank: int, local_host: str) -> tuple[int, str, dict]:
    try:
        if rank == 0 and (local_host.startswith("spark-one") or local_host == "spark-one"):
            res = subprocess.check_output(
                ["bash", "-c", NODE_PROBE_CMD], timeout=5, text=True, stderr=subprocess.DEVNULL
            )
        else:
            res = subprocess.check_output(
                ["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=3", node, NODE_PROBE_CMD],
                timeout=6, text=True, stderr=subprocess.DEVNULL
            )
        return rank, node, json.loads(res.strip().splitlines()[-1])
    except Exception as e:
        return rank, node, {"container": f"SSH_ERR:{type(e).__name__}"}


def render(port: int, prev: dict | None, local_host: str) -> dict:
    now = time.monotonic()
    m = fetch_metrics(port)
    clients = fetch_clients(port)
    sched = read_json("/dev/shm/inkling_v17_sched_stats.json")
    kv_shm = read_json("/dev/shm/inkling_v16_rank0_kv_metrics.json") or read_json("/dev/shm/inkling_v16_kv_metrics.json")

    gen_tok = m.get("vllm:generation_tokens_total", 0.0)
    pf_tok = m.get("vllm:prompt_tokens_total", 0.0)
    running = int(m.get("vllm:num_requests_running", 0.0))
    waiting = int(m.get("vllm:num_requests_waiting", 0.0))
    kv_pct = 100.0 * m.get("vllm:kv_cache_usage_perc", 0.0)
    ph_q = m.get("vllm:prefix_cache_queries_total", 0.0)
    ph_h = m.get("vllm:prefix_cache_hits_total", 0.0)
    spec_drafts = m.get("vllm:spec_decode_num_drafts_total", 0.0)
    spec_acc = m.get("vllm:spec_decode_num_accepted_tokens_total", 0.0)

    gen_tps = 0.0
    pf_tps = 0.0
    al = 1.0 + (spec_acc / spec_drafts) if spec_drafts > 0 else 0.0
    prefix_pct = (100.0 * ph_h / ph_q) if ph_q > 0 else 0.0
    if prev is not None:
        dt = max(1e-3, now - prev["t"])
        gen_tps = max(0.0, (gen_tok - prev["gen_tok"]) / dt)
        pf_tps = max(0.0, (pf_tok - prev["pf_tok"]) / dt)
        dd = spec_drafts - prev["spec_drafts"]
        da = spec_acc - prev["spec_acc"]
        if dd > 0:
            al = 1.0 + da / dd

    per_stream = (gen_tps / running) if running > 0 else 0.0

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(probe_node, n, r, local_host) for n, r in NODES]
        rows = sorted((f.result() for f in futs), key=lambda x: x[0])

    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"=== SPARK-TOP V17 | Inkling-2-Small TP8 (:{port}) | {ts} ===",
        f"LIVE   : running={running:<2d} waiting={waiting:<2d} | "
        f"gen={gen_tps:6.1f} tok/s ({per_stream:5.1f}/stream) | prefill={pf_tps:7.1f} tok/s | "
        f"AL={al:.2f} | prefix_hit={prefix_pct:5.1f}% | KV={kv_pct:4.1f}%",
    ]
    if sched:
        lines.append(
            f"SCHED  : steps={sched.get('steps',0)} pure_dec={sched.get('pure_decode',0)} "
            f"pure_pf={sched.get('pure_prefill',0)} short_skip={sched.get('short_skip',0)} "
            f"hold={sched.get('hold',0)} probe={sched.get('probe',0)} last={sched.get('last_kind','-')}"
        )
    if kv_shm:
        hu = kv_shm.get("hybrid_utilization", {})
        hstr = " ".join(f"{k}={v.get('utilization_ratio',0)*100:.1f}%" for k, v in hu.items())
        lines.append(
            f"KV-SHM : free={kv_shm.get('free_blocks',0)} active={kv_shm.get('active_blocks',0)} "
            f"cached={kv_shm.get('cached_blocks',0)} | {hstr}"
        )
    if clients:
        cstr = ", ".join(f"{ip}({cnt})" for ip, cnt in sorted(clients.items(), key=lambda x: -x[1]))
        lines.append(f"CLIENTS: {cstr}")
    else:
        lines.append("CLIENTS: idle (0 ESTAB connections)")

    lines.append("-" * 102)
    lines.append(
        f"{'RANK':<4} {'NODE':<12} {'GPU_CLK':>7} {'TEMP':>5} {'UTIL':>5} {'PWR':>6} "
        f"{'MEM_AVAIL':>10} {'SWAP':>8} {'ORD13':>6} {'HCAs':>5} {'IB_ERR':>6}  CONTAINER"
    )
    for r, n, d in rows:
        lines.append(
            f"r{r:<3d} {n:<12} {str(d.get('gpu_clk','?'))+'MHz':>7} "
            f"{str(d.get('gpu_temp','?'))+'C':>5} {str(d.get('gpu_util','?'))+'%':>5} "
            f"{str(d.get('gpu_pwr','?'))+'W':>6} {str(d.get('mem_avail_gib','?'))+' GiB':>10} "
            f"{str(d.get('swap_used_mib','?'))+' MiB':>8} {str(d.get('order13','?')):>6} "
            f"{str(d.get('hca_active','?')):>5} {str(d.get('ib_errs','?')):>6}  "
            f"{str(d.get('container','?'))[:30]}"
        )
    print("\n".join(lines), flush=True)
    return {"t": now, "gen_tok": gen_tok, "pf_tok": pf_tok, "spec_drafts": spec_drafts, "spec_acc": spec_acc}


def main() -> int:
    ap = argparse.ArgumentParser(description="spark-top V17 8-node DGX Spark cluster HUD")
    ap.add_argument("--once", action="store_true", help="Print one snapshot and exit")
    ap.add_argument("--interval", type=float, default=2.0, help="Refresh interval seconds")
    ap.add_argument("--port", type=int, default=8010, help="vLLM port on spark-one")
    args = ap.parse_args()
    local_host = socket.gethostname()
    prev = None
    try:
        while True:
            if not args.once:
                sys.stdout.write("\033[2J\033[H")
            prev = render(args.port, prev, local_host)
            if args.once:
                return 0
            time.sleep(max(0.5, args.interval))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
