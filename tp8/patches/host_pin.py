# SPDX-License-Identifier: Apache-2.0
"""DGX Spark (GB10) Big-Core CPU pinning & worker GC hygiene for Inkling-2-Small TP8.

GB10 CPU topology (verified via `lscpu -e` and `/proc/cpuinfo` on spark-one):
  * L3 Cluster 0: CPUs 0..4  = Cortex-A725 (LITTLE, 2808 MHz, part 0xd87)
                  CPUs 5..9  = Cortex-X925 (BIG,    3900 MHz, part 0xd85)
  * L3 Cluster 1: CPUs 10..14 = Cortex-A725 (LITTLE, 2808 MHz, part 0xd87)
                  CPUs 15..19 = Cortex-X925 (BIG,    3900 MHz, part 0xd85)

Default optimal allocation (pair with `docker run --cpuset-cpus=5-9,15-19`,
`RDMA_AR_PROXY_CORE=9`, `RDMA_AR_PROXY_CORE2=19`, `OMP_NUM_THREADS=1`):
  * CPU 9      : RDMA_AR_PROXY_CORE (exclusive BIG X925 in L3 Cluster 0; SCHED_FIFO!)
  * CPU 19     : RDMA_AR_PROXY_CORE2 (exclusive BIG X925 in L3 Cluster 1; SCHED_FIFO!)
  * CPU 5      : VLLM::Worker_TP* main thread (same L3 Cluster 0 as RDMA proxy -> shared L3!)
  * CPUs 6..8  : VLLM::Worker_TP* helper threads (AsyncOutput, cuda-EvtHandlr, ZMQ, gloo)
  * CPU 15     : VLLM::EngineCore (rank 0, isolated in L3 Cluster 1, nice -5)
  * CPUs 16..18: vllm API server + uvicorn SSE + detokenizer + ZMQ I/O (rank 0, L3 Cluster 1,
                 strictly excluding CPU 19!)

Env:
  INKLING_HOST_PIN=1              Enable worker pinning + nice -10 (default off)
  INKLING_WORKER_CORE=5           Main thread BIG core for Worker_TP* (default 5)
  INKLING_HELPER_CORES=6,7,8      Helper threads BIG cores for Worker_TP* (default 6,7,8)
  INKLING_WORKER_NICE=-10         Nice value for Worker_TP* (default -10, requires CAP_SYS_NICE)
  INKLING_WORKER_GC_FREEZE=1      Call gc.collect(); gc.freeze() after graph capture (default 1 when HOST_PIN=1)
  INKLING_PIN_ENGINE_API=1        Opt-in: auto-pin VLLM::EngineCore & parent APIServer (default off)
  INKLING_ENGINECORE_CORE=15      Exclusive BIG core for VLLM::EngineCore (default 15)
  INKLING_ENGINECORE_NICE=-5      Nice value for VLLM::EngineCore (default -5)
  INKLING_APISERVER_CORES=16,17,18 Exclusive BIG cores for vllm APIServer (default 16,17,18)
  INKLING_HOST_PIN_HEADLESS_BIG=0 Opt-in (V10): on headless ranks (1..7, no EngineCore/APIServer),
                                  add idle Cluster-1 X925 Big cores (15..18) to worker helpers
                                  for 4-HCA NCCL threads with zero contention on Worker/Proxies.
  INKLING_HEADLESS_EXTRA_HELPERS=15,16,17,18 Extra helper cores merged on headless ranks 1..7.
  INKLING_HOST_SHUTDOWN_HOOKS=0   Opt-in (V15): install chained SIGTERM/SIGINT + atexit cleanup
                                  hooks for RDMA proxy threads, pinned buffers, loader pool & PID shm.
  INKLING_UMA_TELEMETRY=0         Opt-in (V15): rate-limited runtime /proc/meminfo + /proc/buddyinfo
                                  UMA pressure telemetry via check_uma_health().
  INKLING_UMA_TELEMETRY_INTERVAL_S=60 Minimum seconds between UMA telemetry logs.
  INKLING_TELEMETRY_V16=0         Opt-in (V16): zero-hot-path-overhead atomic /dev/shm runtime
                                  metrics JSON snapshot (/dev/shm/inkling_v16_rank{r}_metrics.json),
                                  rate-limited and strictly guarded against CUDA graph capture.
  INKLING_TELEMETRY_V16_INTERVAL_S=5.0 Minimum seconds between atomic JSON snapshots (default 5s).
  INKLING_TELEMETRY_V16_DIR=/dev/shm   Directory for atomic rank metrics snapshots.
  INKLING_TELEMETRY_V16_BG_THREAD=1    Enable post-capture low-priority refresh thread when V16=1.
"""
from __future__ import annotations

import atexit
import gc
import json
import os
import resource
import signal
import sys
import threading
import time
from typing import Any, Callable

_PINNED = False
_FROZEN = False
_ENGINE_API_PINNED = False

# Wave-V15 shutdown cleanup & UMA telemetry state
_SHUTDOWN_HOOKS_INSTALLED = False
_CLEANUP_LOCK = threading.Lock()
_CLEANUP_DONE = False
_SHUTDOWN_REQUESTED = False
_CLEANUP_CALLBACKS: list[Callable[[], Any]] = []
_OWNED_SHM_PATHS: set[str] = set()
_PREV_SIGNAL_HANDLERS: dict[int, Any] = {}
_LAST_UMA_LOG_TS = 0.0

# Wave-V16 zero-hot-path-overhead atomic /dev/shm metrics snapshot state (default OFF)
_PROCESS_START_MONOTONIC = time.monotonic()
_TELEMETRY_V16_LOCK = threading.Lock()
_LAST_TELEMETRY_V16_TS = 0.0
_TELEMETRY_V16_THREAD: threading.Thread | None = None
_TELEMETRY_V16_STOP_EVENT = threading.Event()
_TELEMETRY_V16_KV_PROVIDER: Callable[[], dict[str, Any]] | None = None


def _env_truthy(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "on", "true", "yes")


def detect_node_rank(rank: int | None = None) -> int:
    """Detect node rank (0 = head with EngineCore/APIServer, 1..7 = headless worker).
    Fail-closed to 0 when uncertain so Rank 0 never collides with EngineCore/APIServer."""
    if rank is not None:
        return int(rank)
    for k in ("INKLING_NODE_RANK", "VLLM_NODE_RANK", "RANK"):
        raw = os.environ.get(k, "").strip()
        if raw:
            try:
                return int(raw)
            except ValueError:
                pass
    try:
        if "--headless" in sys.argv:
            return 1
    except Exception:
        pass
    try:
        raw_cmd = open("/proc/self/cmdline", "rb").read()
        if b"--headless" in raw_cmd:
            return 1
    except Exception:
        pass
    return 0


def parse_cpuset(spec: str) -> set[int]:
    out: set[int] = set()
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    return out


def big_cores_from_sysfs() -> list[int]:
    """Detect BIG cores (>3.2 GHz) from cpufreq sysfs; fallback to GB10 [5..9, 15..19]."""
    big = []
    for cpu in range(64):
        p = f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/cpuinfo_max_freq"
        if not os.path.exists(p):
            if cpu >= 20:
                break
            continue
        try:
            khz = int(open(p).read().strip())
            if khz >= 3200000:
                big.append(cpu)
        except Exception:
            pass
    return big or [5, 6, 7, 8, 9, 15, 16, 17, 18, 19]


def compute_worker_pin_plan(
    allowed: set[int] | None = None,
    proxy_core: int | None = None,
    worker_core: int | None = None,
    helper_cores: set[int] | None = None,
    *,
    proxy_core2: int | None = None,
    rank: int | None = None,
    headless_big: bool | None = None,
    headless_extra_helpers: set[int] | None = None,
) -> tuple[int | None, set[int]]:
    """Return (main_tid_core, helper_tids_cores), strictly avoiding proxy cores and LITTLE cores.
    When INKLING_HOST_PIN_HEADLESS_BIG=1 on headless ranks (rank > 0), merges idle Cluster-1
    X925 Big cores (default 15..18) into helper_cores while preserving zero overlap with Worker
    and both SCHED_FIFO RDMA proxies."""
    if allowed is None:
        try:
            allowed = set(os.sched_getaffinity(0))
        except Exception:
            allowed = set(range(20))
    if proxy_core is None:
        proxy_core = int(os.environ.get("RDMA_AR_PROXY_CORE", "9"))
    if proxy_core2 is None:
        proxy_core2 = int(os.environ.get("RDMA_AR_PROXY_CORE2", "-1"))
    if worker_core is None:
        worker_core = int(os.environ.get("INKLING_WORKER_CORE", "5"))
    if helper_cores is None:
        helper_cores = parse_cpuset(os.environ.get("INKLING_HELPER_CORES", "6,7,8"))
    else:
        helper_cores = set(helper_cores)
    if headless_big is None:
        headless_big = os.environ.get(
            "INKLING_HOST_PIN_HEADLESS_BIG", "0"
        ).strip().lower() in ("1", "on", "true", "yes")
    if headless_big and detect_node_rank(rank) > 0:
        if headless_extra_helpers is None:
            headless_extra_helpers = parse_cpuset(
                os.environ.get("INKLING_HEADLESS_EXTRA_HELPERS", "15,16,17,18")
            )
        helper_cores = helper_cores | set(headless_extra_helpers)

    # Never collide with either RDMA proxy core.
    reserved = {c for c in (proxy_core, proxy_core2) if c >= 0}
    avail = allowed - reserved
    if not avail:
        return None, set()
    main_core = worker_core if worker_core in avail else min(avail)
    helpers = (helper_cores & avail) - {main_core}
    if not helpers:
        helpers = (avail - {main_core}) or {main_core}
    return main_core, helpers


def pin_worker(logger=None) -> bool:
    """Pin Worker_TP* main thread to INKLING_WORKER_CORE and sibling threads to INKLING_HELPER_CORES."""
    global _PINNED
    if _PINNED:
        return False
    if os.environ.get("INKLING_HOST_PIN", "0").strip().lower() not in ("1", "on", "true", "yes"):
        return False
    _PINNED = True
    try:
        main_core, helper_cores = compute_worker_pin_plan()
        if main_core is None:
            return False
        pid = os.getpid()
        proxy_core = int(os.environ.get("RDMA_AR_PROXY_CORE", "-1"))
        proxy_core2 = int(os.environ.get("RDMA_AR_PROXY_CORE2", "-1"))
        proxy_cores = {c for c in (proxy_core, proxy_core2) if c >= 0}
        # Pin all existing threads (except main thread, RDMA proxy cores, and SCHED_FIFO threads)
        task_dir = f"/proc/{pid}/task"
        n_helpers = 0
        if os.path.isdir(task_dir):
            for entry in os.listdir(task_dir):
                try:
                    tid = int(entry)
                except ValueError:
                    continue
                if tid == pid:
                    continue
                try:
                    cur = os.sched_getaffinity(tid)
                    if len(cur) == 1 and next(iter(cur)) in proxy_cores:
                        continue  # Leave RDMA proxy threads exclusively on their cores!
                    if os.sched_getscheduler(tid) in (os.SCHED_FIFO, os.SCHED_RR):
                        continue  # Leave real-time RDMA proxy threads untouched!
                    os.sched_setaffinity(tid, helper_cores)
                    n_helpers += 1
                except Exception:
                    pass
        # Pin the worker main thread last to its dedicated BIG core
        os.sched_setaffinity(pid, {main_core})
        nice_val = int(os.environ.get("INKLING_WORKER_NICE", "-10"))
        if nice_val < 0:
            try:
                os.setpriority(os.PRIO_PROCESS, pid, nice_val)
            except Exception:
                pass
        if logger is not None:
            logger.info(
                "INKLING_HOST_PIN: rank=%d Worker pid=%d pinned main->cpu=%d, %d helpers->cpus=%s, "
                "RDMA_AR_PROXY_CORE=%d, RDMA_AR_PROXY_CORE2=%d, nice=%d",
                detect_node_rank(),
                pid,
                main_core,
                n_helpers,
                sorted(helper_cores),
                proxy_core,
                proxy_core2,
                nice_val,
            )
        if _env_truthy("INKLING_HOST_SHUTDOWN_HOOKS", "0"):
            install_shutdown_handlers(logger=logger)
        return True
    except Exception as e:  # noqa: BLE001
        if logger is not None:
            logger.warning("INKLING_HOST_PIN failed: %r", e)
        return False


_LIBC_TRIM = None
_LIBC_TRIM_TRIED = False


def _get_malloc_trim():
    global _LIBC_TRIM, _LIBC_TRIM_TRIED
    if not _LIBC_TRIM_TRIED:
        _LIBC_TRIM_TRIED = True
        try:
            import ctypes

            libc = ctypes.CDLL("libc.so.6")
            fn = getattr(libc, "malloc_trim", None)
            if fn is not None:
                fn.argtypes = [ctypes.c_size_t]
                fn.restype = ctypes.c_int
                _LIBC_TRIM = fn
        except Exception:
            _LIBC_TRIM = None
    return _LIBC_TRIM


def reclaim_host_uma(empty_cuda: bool = False, logger=None, tag: str = "") -> dict[str, int]:
    """Wave-V13 GB10 UMA host reclaim: gc.collect + PyTorch pinned host cache + glibc malloc_trim(0).

    Strictly capture-guarded and opt-outable via INKLING_UMA_HOST_RECLAIM=0.
    Never touches active decode/prefill weights, CUDA graphs, or 0-ULP kernels."""
    res: dict[str, int] = {"gc": 0, "host_cache": 0, "malloc_trim": 0, "cuda_cache": 0, "skipped": 0}
    if os.environ.get("INKLING_UMA_HOST_RECLAIM", "1").strip().lower() in ("0", "off", "false", "no"):
        res["skipped"] = 1
        return res
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            res["skipped"] = 2
            return res
    except Exception:
        torch = None  # type: ignore[assignment]
    try:
        res["gc"] = int(gc.collect())
    except Exception:
        pass
    if torch is not None:
        try:
            if hasattr(torch._C, "_host_emptyCache"):
                torch._C._host_emptyCache()
                res["host_cache"] = 1
        except Exception:
            pass
    try:
        trim_fn = _get_malloc_trim()
        if trim_fn is not None:
            res["malloc_trim"] = int(trim_fn(0))
    except Exception:
        pass
    if empty_cuda and torch is not None:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                res["cuda_cache"] = 1
        except Exception:
            pass
    if logger is not None:
        logger.info("INKLING_UMA_HOST_RECLAIM[%s]: %s", tag or "misc", res)
    if _env_truthy("INKLING_UMA_TELEMETRY", "0"):
        maybe_log_uma_health(logger=logger, tag=tag or "reclaim")
    if telemetry_v16_enabled():
        maybe_write_metrics_snapshot(tag=tag or "reclaim")
    return res


def release_all_prefill_scratch(
    include_shadow_k: bool = False, empty_cuda: bool = True, logger=None
) -> int:
    """Release non-graphed prefill-only scratch pools when engine returns to decode.

    Never touches decode weights, _FP8_W_CACHE, BF11/BF12/NVFP4/MX7 tables, or CUDA graphs."""
    try:
        import torch

        if torch.cuda.is_available():
            if torch.cuda.is_current_stream_capturing():
                return 0
            torch.cuda.synchronize()
    except Exception:
        pass
    freed = 0
    targets = [
        ("vllm.models.inkling2.ops.linear", "release_prefill_scratch"),
        ("vllm.models.inkling2.ops.pf_moe_native", "release_unused_scratch"),
        ("vllm.models.inkling2.ops.global_attn_1m", "release_prefill_scratch"),
        ("vllm.models.inkling2.prefill_sp_rs", "release_prefill_scratch"),
    ]
    if include_shadow_k:
        targets.append(("vllm.models.inkling2.ops.sm12_attn", "release_shadow_k_cache"))
    for mod_name, fn_name in targets:
        try:
            mod = __import__(mod_name, fromlist=[fn_name])
            fn = getattr(mod, fn_name, None)
            if fn is not None:
                freed += int(fn() or 0)
        except Exception:
            pass
    if freed > 0:
        reclaim_host_uma(empty_cuda=empty_cuda, logger=logger, tag="release_pf_scratch")
        if logger is not None:
            logger.info(
                "INKLING_PF_SCRATCH_RELEASE_IDLE: freed %.1f MiB (shadow_k=%d)",
                freed / (1024.0 * 1024.0),
                int(include_shadow_k),
            )
    return freed


def freeze_worker_gc(logger=None) -> bool:
    """Freeze long-lived model/graph objects out of Gen-2 GC after capture completes."""
    global _FROZEN
    if _FROZEN:
        return False
    if os.environ.get("INKLING_HOST_PIN", "0").strip().lower() not in ("1", "on", "true", "yes"):
        return False
    if os.environ.get("INKLING_WORKER_GC_FREEZE", "1").strip().lower() in ("0", "off", "false", "no"):
        return False
    _FROZEN = True
    reclaim_host_uma(empty_cuda=False, logger=logger, tag="pre_gc_freeze")
    gc.collect()
    gc.freeze()
    if logger is not None:
        logger.info("INKLING_HOST_PIN: worker gc.freeze() completed (%d frozen objects)", gc.get_freeze_count())
    if telemetry_v16_enabled():
        maybe_write_metrics_snapshot(tag="post_gc_freeze")
        start_telemetry_v16_thread(logger=logger)
    return True


def compute_engine_api_pin_plan(
    allowed: set[int] | None = None,
    proxy_cores: set[int] | None = None,
    worker_core: int | None = None,
    helper_cores: set[int] | None = None,
    engine_core: int | None = None,
    api_cores: set[int] | None = None,
) -> tuple[int | None, set[int]]:
    """Return (engine_core, api_cores) on rank 0, strictly disjoint from both RDMA proxies
    (SCHED_FIFO busy-spinners on CPUs 9 & 19) and Worker_TP0 (CPUs 5, 6-8)."""
    if allowed is None:
        try:
            allowed = set(os.sched_getaffinity(0))
        except Exception:
            allowed = set(range(20))
    if proxy_cores is None:
        p1 = int(os.environ.get("RDMA_AR_PROXY_CORE", "9"))
        p2 = int(os.environ.get("RDMA_AR_PROXY_CORE2", "-1"))
        proxy_cores = {c for c in (p1, p2) if c >= 0}
    if worker_core is None:
        worker_core = int(os.environ.get("INKLING_WORKER_CORE", "5"))
    if helper_cores is None:
        helper_cores = parse_cpuset(os.environ.get("INKLING_HELPER_CORES", "6,7,8"))
    if engine_core is None:
        engine_core = int(os.environ.get("INKLING_ENGINECORE_CORE", "15"))
    if api_cores is None:
        api_cores = parse_cpuset(os.environ.get("INKLING_APISERVER_CORES", "16,17,18"))

    # Hard exclusion: never collide with SCHED_FIFO RDMA proxy cores; prefer excluding Worker cores too.
    non_proxy = allowed - proxy_cores
    if not non_proxy:
        return None, set()
    worker_set = ({worker_core} if worker_core >= 0 else set()) | helper_cores
    preferred_pool = non_proxy - worker_set
    pool = preferred_pool if preferred_pool else non_proxy

    eng = engine_core if engine_core in pool else min(pool)
    api = (api_cores & pool) - {eng}
    if not api:
        api = (pool - {eng}) or {eng}
    return eng, api


def _pin_process_tree(pid: int, cpus: set[int], proxy_cores: set[int]) -> int:
    """Pin all non-RT, non-proxy threads of `pid` plus `pid` itself to `cpus`."""
    pinned = 0
    task_dir = f"/proc/{pid}/task"
    if os.path.isdir(task_dir):
        for entry in os.listdir(task_dir):
            try:
                tid = int(entry)
            except ValueError:
                continue
            try:
                cur = os.sched_getaffinity(tid)
                if len(cur) == 1 and next(iter(cur)) in proxy_cores:
                    continue
                if os.sched_getscheduler(tid) in (os.SCHED_FIFO, os.SCHED_RR):
                    continue
                os.sched_setaffinity(tid, cpus)
                pinned += 1
            except Exception:
                pass
    try:
        os.sched_setaffinity(pid, cpus)
        if pinned == 0:
            pinned = 1
    except Exception:
        pass
    return pinned


def _is_vllm_apiserver_pid(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        raw = open(f"/proc/{pid}/cmdline", "rb").read()
    except Exception:
        return False
    cmd = raw.replace(b"\x00", b" ").decode("utf-8", errors="ignore")
    if "EngineCore" in cmd or "Worker" in cmd:
        return False
    return ("vllm" in cmd) or ("api_server" in cmd)


def pin_engine_and_api(logger=None) -> bool:
    """Opt-in (`INKLING_HOST_PIN=1` + `INKLING_PIN_ENGINE_API=1`): pin VLLM::EngineCore to
    exclusive BIG core 15 (nice -5) and its parent vllm APIServer to exclusive BIG cores 16-18,
    strictly avoiding SCHED_FIFO RDMA proxy cores (9, 19) and Worker_TP0 (5, 6-8)."""
    global _ENGINE_API_PINNED
    if _ENGINE_API_PINNED:
        return False
    if os.environ.get("INKLING_HOST_PIN", "0").strip().lower() not in ("1", "on", "true", "yes"):
        return False
    if os.environ.get("INKLING_PIN_ENGINE_API", "0").strip().lower() not in ("1", "on", "true", "yes"):
        return False
    if (
        os.environ.get("INKLING_HOST_PIN_HEADLESS_BIG", "0").strip().lower()
        in ("1", "on", "true", "yes")
        and detect_node_rank() > 0
    ):
        return False
    _ENGINE_API_PINNED = True
    try:
        p1 = int(os.environ.get("RDMA_AR_PROXY_CORE", "9"))
        p2 = int(os.environ.get("RDMA_AR_PROXY_CORE2", "-1"))
        proxy_cores = {c for c in (p1, p2) if c >= 0}
        eng_core, api_cores = compute_engine_api_pin_plan(proxy_cores=proxy_cores)
        if eng_core is None:
            return False

        pid = os.getpid()
        n_eng = _pin_process_tree(pid, {eng_core}, proxy_cores)
        nice_val = int(os.environ.get("INKLING_ENGINECORE_NICE", "-5"))
        if nice_val < 0:
            try:
                os.setpriority(os.PRIO_PROCESS, pid, nice_val)
            except Exception:
                pass

        ppid = os.getppid()
        n_api = 0
        if api_cores and _is_vllm_apiserver_pid(ppid):
            n_api = _pin_process_tree(ppid, api_cores, proxy_cores)

        if logger is not None:
            logger.info(
                "INKLING_PIN_ENGINE_API: EngineCore pid=%d (%d tids)->cpu=%d (nice=%d), "
                "APIServer ppid=%d (%d tids)->cpus=%s (excluded proxies=%s)",
                pid,
                n_eng,
                eng_core,
                nice_val,
                ppid,
                n_api,
                sorted(api_cores) if n_api else [],
                sorted(proxy_cores),
            )
        if _env_truthy("INKLING_HOST_SHUTDOWN_HOOKS", "0"):
            install_shutdown_handlers(logger=logger)
        if telemetry_v16_enabled():
            maybe_write_metrics_snapshot(tag="pin_engine_api")
            start_telemetry_v16_thread(logger=logger)
        return True
    except Exception as e:  # noqa: BLE001
        if logger is not None:
            logger.warning("INKLING_PIN_ENGINE_API failed: %r", e)
        return False


# ==============================================================================
# Wave-V15 Production Hardening: Clean Shutdown Hooks & UMA Pressure Telemetry
# ==============================================================================


def register_cleanup_callback(fn: Callable[[], Any]) -> None:
    """Register an idempotent callback to execute during graceful worker shutdown."""
    if callable(fn) and fn not in _CLEANUP_CALLBACKS:
        _CLEANUP_CALLBACKS.append(fn)


def register_shm_path(path: str) -> None:
    """Register a process-owned /dev/shm path for deterministic unlink on exit."""
    if path:
        _OWNED_SHM_PATHS.add(str(path))


def _release_gb10_load_if_loaded() -> bool:
    """Release gb10_load thread pool, shard fd, and pinned staging cache without importing it."""
    for mod_name in ("vllm.models.inkling2.ops.gb10_load", "gb10_load"):
        mod = sys.modules.get(mod_name)
        if mod is not None:
            fn = getattr(mod, "release_staging", None)
            if callable(fn):
                try:
                    fn()
                    return True
                except Exception:
                    pass
    return False


def _release_rdma_proxies_if_loaded() -> int:
    """Stop and drop references to live RDMA proxy handles so C++ ~RdmaAR() joins SCHED_FIFO threads.

    Strictly inspects already-imported modules (never initializes torch.distributed during teardown).
    Also invokes .shutdown()/.close()/.stop() if exposed by RDMA wrappers or extensions."""
    comms: list[tuple[Any, Any, str]] = []
    seen_ids: set[int] = set()

    def _add_comm(comm: Any, owner: Any, attr: str) -> None:
        if comm is not None and id(comm) not in seen_ids:
            seen_ids.add(id(comm))
            comms.append((comm, owner, attr))

    rdma_mod = sys.modules.get("vllm.distributed.device_communicators.rdma_all_reduce")
    if rdma_mod is not None:
        ar_timing = getattr(rdma_mod, "_AR_TIMING", None)
        if isinstance(ar_timing, dict):
            _add_comm(ar_timing.get("ar"), ar_timing, "ar")

    ps_mod = sys.modules.get("vllm.distributed.parallel_state")
    if ps_mod is not None:
        tp_group = getattr(ps_mod, "_TP", None)
        dev_comm = getattr(tp_group, "device_communicator", None) if tp_group is not None else None
        if dev_comm is not None:
            _add_comm(getattr(dev_comm, "rdma_comm", None), dev_comm, "rdma_comm")

    released = 0
    for comm, owner, attr in comms:
        try:
            for meth in ("shutdown", "close", "stop"):
                fn = getattr(comm, meth, None)
                if callable(fn):
                    try:
                        fn()
                    except Exception:
                        pass
                    break
            for handle_attr in ("h", "_rs_h", "rs_h"):
                handle = getattr(comm, handle_attr, None)
                if handle is not None:
                    for meth in ("shutdown", "stop", "close"):
                        hfn = getattr(handle, meth, None)
                        if callable(hfn):
                            try:
                                hfn()
                            except Exception:
                                pass
                            break
                    try:
                        setattr(comm, handle_attr, None)
                    except Exception:
                        pass
            try:
                comm.disabled = True
            except Exception:
                pass
            if isinstance(owner, dict):
                owner[attr] = None
            elif owner is not None:
                try:
                    setattr(owner, attr, None)
                except Exception:
                    pass
            released += 1
        except Exception:
            pass
    return released


def _sweep_pid_shm_files(shm_dir: str | None = None, pid: int | None = None) -> int:
    """Remove ONLY shm files explicitly registered or prefixed with this process's PID."""
    removed = 0
    for path in list(_OWNED_SHM_PATHS):
        _OWNED_SHM_PATHS.discard(path)
        try:
            if os.path.lexists(path):
                os.unlink(path)
                removed += 1
        except OSError:
            pass

    target_dir = shm_dir or os.environ.get("INKLING_SHM_DIR", "/dev/shm")
    my_pid = os.getpid() if pid is None else int(pid)
    prefixes = (
        f"inkling_v15_pid{my_pid}_",
        f"inkling_pid{my_pid}_",
        f"vllm_pid{my_pid}_",
        f"rdma_ar_pid{my_pid}_",
    )
    if target_dir and os.path.isdir(target_dir):
        try:
            pid_tmp_token = f".{my_pid}."
            for name in os.listdir(target_dir):
                if name.startswith(prefixes) or (
                    name.startswith(".inkling_v16_rank")
                    and pid_tmp_token in name
                    and name.endswith(".tmp")
                ):
                    full = os.path.join(target_dir, name)
                    try:
                        if os.path.isfile(full) or os.path.islink(full):
                            os.unlink(full)
                            removed += 1
                    except OSError:
                        pass
        except OSError:
            pass
    return removed


def cleanup_runtime_resources(
    reason: str = "atexit",
    logger=None,
    *,
    release_rdma: bool = True,
    sweep_shm: bool = True,
    release_scratch: bool = True,
    shm_dir: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Idempotent Wave-V15 cleanup of loader pool, RDMA SCHED_FIFO proxies, pinned buffers & PID shm.

    Unfreezes gc.freeze() prior to gc.collect() so cyclic references to C++ ~RdmaAR() handles
    and pinned host tensors are deterministically destroyed on shutdown."""
    global _CLEANUP_DONE, _FROZEN
    acquired = _CLEANUP_LOCK.acquire(blocking=False)
    if not acquired:
        return {"reason": reason, "skipped": True, "skip_reason": "in_progress"}
    try:
        if _CLEANUP_DONE and not force:
            return {"reason": reason, "skipped": True, "skip_reason": "already_done"}
        _CLEANUP_DONE = True

        res: dict[str, Any] = {
            "reason": reason,
            "skipped": False,
            "callbacks": 0,
            "gb10_load": False,
            "scratch_bytes": 0,
            "rdma_released": 0,
            "gc_unfrozen": False,
            "telemetry_v16_stopped": False,
            "shm_removed": 0,
            "uma_reclaim": {},
        }

        res["telemetry_v16_stopped"] = stop_telemetry_v16_thread()

        for cb in reversed(list(_CLEANUP_CALLBACKS)):
            try:
                cb()
                res["callbacks"] += 1
            except Exception:
                pass

        res["gb10_load"] = _release_gb10_load_if_loaded()

        if release_scratch:
            try:
                res["scratch_bytes"] = int(
                    release_all_prefill_scratch(include_shadow_k=False, empty_cuda=False)
                )
            except Exception:
                pass

        if release_rdma:
            res["rdma_released"] = _release_rdma_proxies_if_loaded()

        if _FROZEN:
            try:
                gc.unfreeze()
                _FROZEN = False
                res["gc_unfrozen"] = True
            except Exception:
                pass

        try:
            res["uma_reclaim"] = reclaim_host_uma(empty_cuda=False, logger=None, tag=f"shutdown_{reason}")
        except Exception:
            pass

        if sweep_shm:
            res["shm_removed"] = _sweep_pid_shm_files(shm_dir=shm_dir)

        if logger is not None:
            logger.info("INKLING_HOST_SHUTDOWN[%s]: %s", reason, res)
        return res
    finally:
        _CLEANUP_LOCK.release()


def _atexit_cleanup() -> None:
    try:
        cleanup_runtime_resources(reason="atexit")
    except Exception:
        pass


def install_shutdown_handlers(logger=None, *, force: bool = False) -> bool:
    """Install chained SIGTERM/SIGINT & atexit cleanup hooks (opt-in via INKLING_HOST_SHUTDOWN_HOOKS=1).

    When vLLM/Uvicorn already installed a custom handler, releases staging and delegates to the
    existing handler so graceful HTTP drain runs first and full teardown runs in atexit.
    When the signal was SIG_DFL, executes full cleanup immediately and re-raises with SIG_DFL."""
    global _SHUTDOWN_HOOKS_INSTALLED
    if _SHUTDOWN_HOOKS_INSTALLED and not force:
        return False
    if not force and not _env_truthy("INKLING_HOST_SHUTDOWN_HOOKS", "0"):
        return False
    if not _SHUTDOWN_HOOKS_INSTALLED:
        atexit.register(_atexit_cleanup)
    _SHUTDOWN_HOOKS_INSTALLED = True

    if threading.current_thread() is not threading.main_thread():
        return True

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            prev = signal.getsignal(sig)
            if prev is signal.SIG_IGN:
                continue
            _PREV_SIGNAL_HANDLERS[int(sig)] = prev

            def _make_handler(bound_sig: int, prev_handler: Any):
                def _handler(signum: int, frame: Any) -> None:
                    global _SHUTDOWN_REQUESTED
                    _SHUTDOWN_REQUESTED = True
                    # Always safe mid-flight: close loader pool & staging fds if still open.
                    _release_gb10_load_if_loaded()
                    if callable(prev_handler):
                        prev_handler(signum, frame)
                        return
                    cleanup_runtime_resources(reason=f"signal_{signum}", logger=logger)
                    signal.signal(bound_sig, signal.SIG_DFL)
                    os.kill(os.getpid(), bound_sig)

                return _handler

            signal.signal(sig, _make_handler(int(sig), prev))
        except Exception:
            pass
    return True


def parse_meminfo(path: str = "/proc/meminfo") -> dict[str, float]:
    """Parse Linux /proc/meminfo into MiB/GiB metrics (empty dict if unavailable)."""
    raw_kb: dict[str, int] = {}
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                if ":" not in line:
                    continue
                key, rest = line.split(":", 1)
                parts = rest.strip().split()
                if not parts:
                    continue
                try:
                    val = int(parts[0])
                except ValueError:
                    continue
                unit = parts[1].lower() if len(parts) > 1 else "kb"
                if unit == "kb":
                    raw_kb[key.strip()] = val
                elif unit == "mb":
                    raw_kb[key.strip()] = val * 1024
                else:
                    raw_kb[key.strip()] = val // 1024
    except OSError:
        return {}
    if not raw_kb:
        return {}
    mem_total = raw_kb.get("MemTotal", 0)
    mem_free = raw_kb.get("MemFree", 0)
    mem_avail = raw_kb.get("MemAvailable", mem_free)
    swap_total = raw_kb.get("SwapTotal", 0)
    swap_free = raw_kb.get("SwapFree", 0)
    swap_used = max(0, swap_total - swap_free)
    return {
        "mem_total_gib": round(mem_total / (1024.0 * 1024.0), 3),
        "mem_free_gib": round(mem_free / (1024.0 * 1024.0), 3),
        "mem_available_gib": round(mem_avail / (1024.0 * 1024.0), 3),
        "swap_total_gib": round(swap_total / (1024.0 * 1024.0), 3),
        "swap_free_gib": round(swap_free / (1024.0 * 1024.0), 3),
        "swap_used_mib": round(swap_used / 1024.0, 2),
        "cached_gib": round(raw_kb.get("Cached", 0) / (1024.0 * 1024.0), 3),
        "shmem_mib": round(raw_kb.get("Shmem", 0) / 1024.0, 2),
        "dirty_mib": round(raw_kb.get("Dirty", 0) / 1024.0, 2),
    }


def parse_buddyinfo(
    path: str = "/proc/buddyinfo",
    page_size: int | None = None,
) -> dict[str, Any]:
    """Parse Linux /proc/buddyinfo across all zones (supports 11-order x86 and 14-order GB10 32 MiB)."""
    if page_size is None:
        try:
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
        except Exception:
            page_size = 4096
    zones: dict[str, list[int]] = {}
    totals: list[int] = []
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or "zone" not in line:
                    continue
                # Format: "Node 0, zone   Normal    12   8   ...  3720"
                parts = line.split()
                try:
                    z_idx = parts.index("zone")
                except ValueError:
                    continue
                if z_idx + 2 >= len(parts):
                    continue
                node_id = parts[1].rstrip(",") if len(parts) > 1 else "0"
                zone_name = f"node{node_id}_{parts[z_idx + 1]}"
                counts = [int(tok) for tok in parts[z_idx + 2 :] if tok.isdigit()]
                if not counts:
                    continue
                zones[zone_name] = counts
                if len(counts) > len(totals):
                    totals.extend([0] * (len(counts) - len(totals)))
                for i, c in enumerate(counts):
                    totals[i] += c
    except OSError:
        return {}
    if not totals:
        return {}
    max_order = len(totals) - 1
    ge9_bytes = sum(totals[o] * (page_size << o) for o in range(9, len(totals)))
    ge10_bytes = sum(totals[o] * (page_size << o) for o in range(10, len(totals)))
    o13_blocks = totals[13] if len(totals) > 13 else 0
    o13_bytes = o13_blocks * (page_size << 13) if len(totals) > 13 else 0
    total_free_bytes = sum(totals[o] * (page_size << o) for o in range(len(totals)))
    high_ratio = (ge9_bytes / total_free_bytes) if total_free_bytes > 0 else 0.0
    return {
        "page_size_bytes": page_size,
        "max_order": max_order,
        "orders": totals,
        "zones": zones,
        "order_ge9_blocks": sum(totals[9:]) if len(totals) > 9 else 0,
        "order_ge10_blocks": sum(totals[10:]) if len(totals) > 10 else 0,
        "order_ge9_gib": round(ge9_bytes / (1024.0**3), 3),
        "order_ge10_gib": round(ge10_bytes / (1024.0**3), 3),
        "order_13_blocks": o13_blocks,
        "order_13_gib": round(o13_bytes / (1024.0**3), 3),
        "high_order_ratio": round(high_ratio, 4),
    }


def check_uma_health(
    *,
    mode: str = "runtime",
    meminfo_path: str = "/proc/meminfo",
    buddyinfo_path: str = "/proc/buddyinfo",
    page_size: int | None = None,
    min_mem_avail_gib: float | None = None,
    max_swap_used_mib: float | None = None,
    min_order13_blocks: int | None = None,
    min_order_ge9_gib: float | None = None,
) -> dict[str, Any]:
    """Lightweight, capture-free GB10 UMA pressure & buddy fragmentation health check.

    Modes:
      - ``runtime``: thresholds tuned for live serving at GMU=0.85 (~7-15 GiB free UMA).
      - ``preboot``: thresholds tuned for clean pre-launch GB10 (~116 GiB free, ~3720 order-13 blocks).
    """
    mem = parse_meminfo(meminfo_path)
    buddy = parse_buddyinfo(buddyinfo_path, page_size=page_size)
    if not mem and not buddy:
        return {
            "available": False,
            "mode": mode,
            "status": "UNAVAILABLE",
            "reasons": ["procfs_unavailable"],
            "needs_defrag": False,
            "needs_reboot_defrag": False,
        }

    is_preboot = mode == "preboot"
    if min_mem_avail_gib is None:
        min_mem_avail_gib = float(
            os.environ.get(
                "UMA_MIN_MEM_AVAIL_GIB" if is_preboot else "INKLING_UMA_RUNTIME_MIN_AVAIL_GIB",
                "100.0" if is_preboot else "5.0",
            )
        )
    if max_swap_used_mib is None:
        max_swap_used_mib = float(
            os.environ.get(
                "UMA_MAX_SWAP_USED_MIB" if is_preboot else "INKLING_UMA_RUNTIME_MAX_SWAP_MIB",
                "256.0" if is_preboot else "1024.0",
            )
        )
    if min_order13_blocks is None:
        min_order13_blocks = int(
            os.environ.get("UMA_MIN_ORDER13_BLOCKS", "2500" if is_preboot else "0")
        )
    if min_order_ge9_gib is None:
        min_order_ge9_gib = float(
            os.environ.get(
                "UMA_MIN_ORDER_GE9_GIB" if is_preboot else "INKLING_UMA_RUNTIME_MIN_GE9_GIB",
                "80.0" if is_preboot else "1.0",
            )
        )

    reasons: list[str] = []
    severity = 0  # 0=OK, 1=WARN, 2=CRITICAL
    needs_defrag = False
    needs_reboot_defrag = False

    if mem:
        avail_gib = mem["mem_available_gib"]
        swap_mib = mem["swap_used_mib"]
        if avail_gib < min_mem_avail_gib:
            reasons.append(f"mem_avail_low:{avail_gib:.2f}GiB<{min_mem_avail_gib:.1f}GiB")
            severity = max(severity, 2 if avail_gib < min_mem_avail_gib * 0.7 else 1)
        if swap_mib > max_swap_used_mib:
            reasons.append(f"swap_used_high:{swap_mib:.1f}MiB>{max_swap_used_mib:.1f}MiB")
            needs_defrag = True
            severity = max(severity, 2 if swap_mib > max_swap_used_mib * 2.0 else 1)

    if buddy:
        o13 = buddy["order_13_blocks"]
        ge9_gib = buddy["order_ge9_gib"]
        if min_order13_blocks > 0 and buddy["max_order"] >= 13 and o13 < min_order13_blocks:
            reasons.append(f"order13_fragmented:{o13}<{min_order13_blocks}")
            needs_defrag = True
            if is_preboot:
                needs_reboot_defrag = True
            severity = max(severity, 2 if is_preboot else 1)
        if ge9_gib < min_order_ge9_gib:
            reasons.append(f"order_ge9_low:{ge9_gib:.2f}GiB<{min_order_ge9_gib:.1f}GiB")
            needs_defrag = True
            if is_preboot:
                needs_reboot_defrag = True
            severity = max(severity, 2 if ge9_gib < min_order_ge9_gib * 0.5 else 1)

    status = ("OK", "WARN", "CRITICAL")[severity]
    out: dict[str, Any] = {
        "available": True,
        "mode": mode,
        "status": status,
        "reasons": reasons,
        "needs_defrag": needs_defrag,
        "needs_reboot_defrag": needs_reboot_defrag,
    }
    out.update(mem)
    out.update({k: v for k, v in buddy.items() if k != "zones"})
    return out


def maybe_log_uma_health(
    logger=None,
    *,
    tag: str = "",
    min_interval_s: float | None = None,
    mode: str = "runtime",
) -> dict[str, Any] | None:
    """Rate-limited UMA health logger (opt-in via INKLING_UMA_TELEMETRY=1)."""
    global _LAST_UMA_LOG_TS
    now = time.monotonic()
    interval = (
        float(os.environ.get("INKLING_UMA_TELEMETRY_INTERVAL_S", "60.0"))
        if min_interval_s is None
        else float(min_interval_s)
    )
    if _LAST_UMA_LOG_TS > 0.0 and (now - _LAST_UMA_LOG_TS) < interval:
        return None
    _LAST_UMA_LOG_TS = now
    health = check_uma_health(mode=mode)
    if logger is not None and health.get("available"):
        log_fn = logger.warning if health.get("status") != "OK" else logger.info
        log_fn(
            "INKLING_UMA_HEALTH[%s]: status=%s avail=%.2fGiB swap=%.1fMiB order13=%d order>=9=%.2fGiB reasons=%s",
            tag or mode,
            health.get("status"),
            health.get("mem_available_gib", -1.0),
            health.get("swap_used_mib", -1.0),
            health.get("order_13_blocks", -1),
            health.get("order_ge9_gib", -1.0),
            health.get("reasons", []),
        )
    if telemetry_v16_enabled():
        maybe_write_metrics_snapshot(tag=tag or mode)
    return health


# ==============================================================================
# Wave-V16 Zero-Hot-Path-Overhead Runtime Telemetry Snapshot Hook
# ==============================================================================


def telemetry_v16_enabled() -> bool:
    """Strictly opt-in via INKLING_TELEMETRY_V16=1 (default OFF -> 0.00 us tax)."""
    return _env_truthy("INKLING_TELEMETRY_V16", "0")


def _is_cuda_graph_capturing() -> bool:
    """Capture-safety guard without triggering a lazy torch import."""
    mod = sys.modules.get("torch")
    if mod is None:
        return False
    try:
        cuda = getattr(mod, "cuda", None)
        if cuda is not None and cuda.is_available():
            return bool(cuda.is_current_stream_capturing())
    except Exception:
        pass
    return False


def parse_process_rss_mib(status_path: str = "/proc/self/status") -> float:
    """Read current process VmRSS in MiB from /proc/self/status (rusage fallback)."""
    try:
        with open(status_path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        kb = int(parts[1])
                        return round(kb / 1024.0, 2)
    except OSError:
        pass
    try:
        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports ru_maxrss in bytes; Linux reports KiB.
        if sys.platform == "darwin":
            return round(ru / (1024.0 * 1024.0), 2)
        return round(ru / 1024.0, 2)
    except Exception:
        return 0.0


def telemetry_v16_snapshot_path(
    rank: int | None = None,
    shm_dir: str | None = None,
) -> str:
    """Return atomic snapshot path `/dev/shm/inkling_v16_rank{r}_metrics.json`."""
    r = detect_node_rank(rank)
    base_dir = (
        shm_dir
        or os.environ.get("INKLING_TELEMETRY_V16_DIR")
        or os.environ.get("INKLING_SHM_DIR")
        or "/dev/shm"
    )
    return os.path.join(base_dir, f"inkling_v16_rank{r}_metrics.json")


def telemetry_v16_kv_sidecar_path(
    rank: int | None = None,
    shm_dir: str | None = None,
) -> str:
    """Return KV sidecar path so Rank-0 Worker and EngineCore never clobber each other."""
    r = detect_node_rank(rank)
    base_dir = (
        shm_dir
        or os.environ.get("INKLING_TELEMETRY_V16_DIR")
        or os.environ.get("INKLING_SHM_DIR")
        or "/dev/shm"
    )
    return os.path.join(base_dir, f"inkling_v16_rank{r}_kv_metrics.json")


def register_kv_metrics_provider(fn: Callable[[], dict[str, Any]] | None) -> None:
    """Register an in-process KV telemetry callback (e.g. KVCacheManager.get_kv_telemetry_v16)."""
    global _TELEMETRY_V16_KV_PROVIDER
    _TELEMETRY_V16_KV_PROVIDER = fn if callable(fn) else None


def _atomic_write_json(
    dst_path: str,
    payload: dict[str, Any],
    *,
    register_cleanup: bool = True,
) -> bool:
    parent = os.path.dirname(dst_path)
    if not parent or not os.path.isdir(parent):
        return False
    base_name = os.path.basename(dst_path)
    tmp_path = os.path.join(
        parent, f".{base_name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n"
        with open(tmp_path, "w", encoding="utf-8") as fh:
            fh.write(encoded)
        os.replace(tmp_path, dst_path)
        if register_cleanup:
            register_shm_path(dst_path)
        return True
    except OSError:
        try:
            if os.path.lexists(tmp_path):
                os.unlink(tmp_path)
        except OSError:
            pass
        return False


def collect_host_telemetry_snapshot(
    *,
    rank: int | None = None,
    meminfo_path: str = "/proc/meminfo",
    buddyinfo_path: str = "/proc/buddyinfo",
    status_path: str = "/proc/self/status",
    page_size: int | None = None,
    tag: str = "",
) -> dict[str, Any]:
    """Collect lightweight GB10 UMA, /proc/buddyinfo order-9..13, RSS, uptime & rank PID."""
    r = detect_node_rank(rank)
    mem = parse_meminfo(meminfo_path)
    buddy = parse_buddyinfo(buddyinfo_path, page_size=page_size)
    orders = buddy.get("orders") or []
    buddy_out: dict[str, Any] = {
        "max_order": buddy.get("max_order"),
        "order_ge9_blocks": buddy.get("order_ge9_blocks", 0),
        "order_ge10_blocks": buddy.get("order_ge10_blocks", 0),
        "order_ge9_gib": buddy.get("order_ge9_gib", 0.0),
        "order_13_gib": buddy.get("order_13_gib", 0.0),
        "high_order_ratio": buddy.get("high_order_ratio", 0.0),
    }
    for o in range(9, 14):
        buddy_out[f"order_{o}_blocks"] = int(orders[o]) if o < len(orders) else 0

    return {
        "schema": "inkling.telemetry.v16",
        "ts_unix": round(time.time(), 3),
        "rank": r,
        "pid": os.getpid(),
        "uptime_s": round(max(0.0, time.monotonic() - _PROCESS_START_MONOTONIC), 2),
        "rss_mib": parse_process_rss_mib(status_path),
        "tag": tag or "periodic",
        "uma": {
            "mem_total_gib": mem.get("mem_total_gib"),
            "mem_free_gib": mem.get("mem_free_gib"),
            "mem_available_gib": mem.get("mem_available_gib"),
            "swap_total_gib": mem.get("swap_total_gib"),
            "swap_free_gib": mem.get("swap_free_gib"),
            "swap_used_mib": mem.get("swap_used_mib"),
            "cached_gib": mem.get("cached_gib"),
            "shmem_mib": mem.get("shmem_mib"),
            "dirty_mib": mem.get("dirty_mib"),
        },
        "buddy": buddy_out,
    }


def _read_fresh_kv_sidecar(sidecar_path: str, max_age_s: float = 60.0) -> dict[str, Any] | None:
    try:
        if not os.path.isfile(sidecar_path):
            return None
        with open(sidecar_path, encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            return None
        ts = data.get("ts_unix")
        if isinstance(ts, (int, float)) and (time.time() - float(ts)) > max_age_s:
            return None
        return data
    except Exception:
        return None


def maybe_write_metrics_snapshot(
    *,
    rank: int | None = None,
    kv_metrics: dict[str, Any] | None = None,
    shm_dir: str | None = None,
    min_interval_s: float | None = None,
    force: bool = False,
    tag: str = "",
    meminfo_path: str = "/proc/meminfo",
    buddyinfo_path: str = "/proc/buddyinfo",
    status_path: str = "/proc/self/status",
    page_size: int | None = None,
) -> dict[str, Any] | None:
    """Write atomic `/dev/shm/inkling_v16_rank{r}_metrics.json` at most once every N seconds.

    Strictly opt-in (`INKLING_TELEMETRY_V16=1`, default OFF) and NEVER executes inside
    CUDA graph capture (`torch.cuda.is_current_stream_capturing()`).
    """
    global _LAST_TELEMETRY_V16_TS
    if not force and not telemetry_v16_enabled():
        return None
    if _is_cuda_graph_capturing():
        return None

    now = time.monotonic()
    interval = (
        max(0.1, float(os.environ.get("INKLING_TELEMETRY_V16_INTERVAL_S", "5.0") or "5.0"))
        if min_interval_s is None
        else max(0.0, float(min_interval_s))
    )
    if not force and _LAST_TELEMETRY_V16_TS > 0.0 and (now - _LAST_TELEMETRY_V16_TS) < interval:
        return None

    if not _TELEMETRY_V16_LOCK.acquire(blocking=False):
        return None
    try:
        now = time.monotonic()
        if not force and _LAST_TELEMETRY_V16_TS > 0.0 and (now - _LAST_TELEMETRY_V16_TS) < interval:
            return None
        _LAST_TELEMETRY_V16_TS = now

        snap = collect_host_telemetry_snapshot(
            rank=rank,
            meminfo_path=meminfo_path,
            buddyinfo_path=buddyinfo_path,
            status_path=status_path,
            page_size=page_size,
            tag=tag,
        )
        sidecar_path = telemetry_v16_kv_sidecar_path(rank=rank, shm_dir=shm_dir)
        resolved_kv: dict[str, Any] | None = None
        if kv_metrics is not None:
            resolved_kv = dict(kv_metrics)
            _atomic_write_json(sidecar_path, resolved_kv, register_cleanup=True)
        elif _TELEMETRY_V16_KV_PROVIDER is not None:
            try:
                prov = _TELEMETRY_V16_KV_PROVIDER()
                if isinstance(prov, dict):
                    resolved_kv = prov
                    _atomic_write_json(sidecar_path, resolved_kv, register_cleanup=True)
            except Exception:
                pass
        else:
            resolved_kv = _read_fresh_kv_sidecar(sidecar_path)

        if resolved_kv is not None:
            snap["kv"] = resolved_kv

        dst_path = telemetry_v16_snapshot_path(rank=rank, shm_dir=shm_dir)
        if not _atomic_write_json(dst_path, snap, register_cleanup=True):
            return None
        return snap
    finally:
        _TELEMETRY_V16_LOCK.release()


def start_telemetry_v16_thread(
    *,
    interval_s: float | None = None,
    rank: int | None = None,
    shm_dir: str | None = None,
    meminfo_path: str = "/proc/meminfo",
    buddyinfo_path: str = "/proc/buddyinfo",
    status_path: str = "/proc/self/status",
    logger=None,
) -> bool:
    """Start post-capture daemon thread refreshing `/dev/shm/inkling_v16_rank{r}_metrics.json`."""
    global _TELEMETRY_V16_THREAD
    if not telemetry_v16_enabled():
        return False
    if not _env_truthy("INKLING_TELEMETRY_V16_BG_THREAD", "1"):
        return False
    if _TELEMETRY_V16_THREAD is not None and _TELEMETRY_V16_THREAD.is_alive():
        return False

    period = (
        max(0.1, float(os.environ.get("INKLING_TELEMETRY_V16_INTERVAL_S", "5.0") or "5.0"))
        if interval_s is None
        else max(0.05, float(interval_s))
    )
    _TELEMETRY_V16_STOP_EVENT.clear()

    def _worker_loop() -> None:
        while not _TELEMETRY_V16_STOP_EVENT.wait(period):
            try:
                maybe_write_metrics_snapshot(
                    rank=rank,
                    shm_dir=shm_dir,
                    min_interval_s=period * 0.85,
                    tag="bg_periodic",
                    meminfo_path=meminfo_path,
                    buddyinfo_path=buddyinfo_path,
                    status_path=status_path,
                )
            except Exception:
                pass

    t = threading.Thread(target=_worker_loop, name="inkling-telemetry-v16", daemon=True)
    _TELEMETRY_V16_THREAD = t
    t.start()
    if logger is not None:
        logger.info(
            "INKLING_TELEMETRY_V16: started background snapshot thread (rank=%d, interval=%.1fs, path=%s)",
            detect_node_rank(rank),
            period,
            telemetry_v16_snapshot_path(rank=rank, shm_dir=shm_dir),
        )
    return True


def stop_telemetry_v16_thread(timeout_s: float = 1.0) -> bool:
    """Stop background Wave-V16 telemetry thread cleanly during worker shutdown."""
    global _TELEMETRY_V16_THREAD
    t = _TELEMETRY_V16_THREAD
    _TELEMETRY_V16_STOP_EVENT.set()
    if t is None:
        return False
    if t.is_alive() and t is not threading.current_thread():
        t.join(timeout=timeout_s)
    _TELEMETRY_V16_THREAD = None
    return True
