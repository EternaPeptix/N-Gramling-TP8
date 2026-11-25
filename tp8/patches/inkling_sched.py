"""inkling_sched: V17 standalone AsyncScheduler subclass for Inkling-2-Small TP8.

Combines two zero-ULP, zero-site-packages-edit scheduling upgrades:
  1. `ds41f` Prefix-Cache-Probed Short-Skip + Async Lookahead Hold (`hold=1`):
     - Prevents short chats and warm multi-turn turns (high prefix-cache hit) from
       starving behind a 128K-1M chunked prefill (~113s -> <2s TTFT).
     - Solo 1M prefill with no competing requests runs full stock `MAX_BATCHED` (24576)
       chunks untouched.
  2. V17 Pure-Prefill / Pure-Decode Step Separation (`pure_step=1`):
     - On Inkling-2-Small TP8, mixing active decode streams with a long prefill chunk
       simultaneously disarms decode CUDA graphs / DTGraph / SyncFree AND disarms
       prefill SP / DBO / Sage2 / OTF-Rel / cm=2 / NonTailSkip (~90% decode collapse!).
     - When BOTH active decoders AND a long prefill (>= `pure_min_fresh` uncached tokens)
       exist, alternates `pure_decode_burst` pure-decode steps (all decode turbo paths
       100% ON!) with 1 bounded pure-prefill chunk (`pure_pf_chunk`, all prefill turbo
       paths 100% ON!).
     - Compatible with both `--scheduling-policy fcfs` and `--scheduling-policy priority`.

Usage:
  --scheduler-cls inkling_sched.InklingScheduler
Env knobs (all optional, hot-reloadable via INKLING_SCHED_CTL JSON file):
  INKLING_SCHED_ENABLE=1
  INKLING_SCHED_SHORT_MAX=4096
  INKLING_SCHED_LONG_CAP=0           (0 = skip long prefill during short step)
  INKLING_SCHED_SCAN=32
  INKLING_SCHED_HOLD=1               (1 = hold async lookahead on big/long prefill)
  INKLING_SCHED_HOLD_MIN=4096
  INKLING_SCHED_PROBE=1              (1 = prefix-cache probe waiting requests)
  INKLING_SCHED_PURE_STEP=1          (1 = pure-prefill / pure-decode interleaving)
  INKLING_SCHED_PURE_MIN_FRESH=2048  (min uncached prefill tokens to trigger separation)
  INKLING_SCHED_PURE_PF_CHUNK=8192   (block-aligned chunk cap on interleaved pure prefill)
  INKLING_SCHED_PURE_DECODE_BURST=3  (pure-decode steps per interleaved pure-prefill chunk)
  INKLING_SCHED_STATS=/dev/shm/inkling_v17_sched_stats.json
"""

from __future__ import annotations

import json
import os
import sys
import time
import weakref

DEFAULTS: dict[str, int] = {
    "enable": 1,
    "short_max": 4096,
    "long_cap": 0,
    "scan": 32,
    "hold": 1,
    "hold_min": 4096,
    "probe": 1,
    "pure_step": 1,
    "pure_min_fresh": 2048,
    "pure_pf_chunk": 8192,
    "pure_decode_burst": 3,
}


def round_up(x: int, b: int) -> int:
    b = max(1, int(b))
    return ((int(x) + b - 1) // b) * b


def env_params() -> dict[str, int]:
    p = dict(DEFAULTS)
    for k in p:
        raw = os.environ.get("INKLING_SCHED_" + k.upper())
        if raw not in (None, ""):
            try:
                p[k] = int(raw)
            except ValueError:
                pass
    return p


def decide_short_skip(
    running_pf_rem: list[int],
    waiting_fresh: list[int],
    p: dict[str, int],
    block: int = 256,
) -> tuple[int, str]:
    """Decide whether a waiting short request should jump ahead of running long prefills.
    Returns (threshold, kind); threshold == 0 means no short-skip this step."""
    if not p.get("enable", 1) or not running_pf_rem:
        return 0, ""
    short_max = int(p.get("short_max", 4096))
    scan = int(p.get("scan", 32))
    shorts = [f for f in waiting_fresh[:scan] if 0 < f <= short_max]
    if not shorts:
        return 0, ""
    skip = int(p.get("long_cap", 0)) <= 0
    long_cap = block if skip else max(int(p["long_cap"]), block)
    thr = round_up(max(long_cap, max(shorts)), block)
    if any(rem > thr for rem in running_pf_rem):
        return thr, "short-skip" if skip else "short"
    return 0, ""


def decide_pure_mode(
    has_decoders: bool,
    has_long_running_pf: bool,
    has_long_waiting_pf: bool,
    decode_streak: int,
    p: dict[str, int],
) -> str:
    """When both decoders and long prefills coexist, return 'pure_decode', 'pure_prefill', or ''."""
    if not p.get("enable", 1) or not p.get("pure_step", 1):
        return ""
    if not has_decoders:
        return ""
    if not (has_long_running_pf or has_long_waiting_pf):
        return ""
    burst = max(1, int(p.get("pure_decode_burst", 3)))
    if decode_streak < burst:
        return "pure_decode"
    return "pure_prefill"


def decide_hold(
    p: dict[str, int],
    inflight_tokens: list[int],
    running_pf_rem: list[int],
    waiting_head_fresh: int | None,
    room: bool,
) -> bool:
    """Prevent async lookahead from pre-committing chunk N+1 behind a big prefill step."""
    if not p.get("enable", 1) or not p.get("hold", 1):
        return False
    if not inflight_tokens:
        return False
    hm = int(p.get("hold_min", 4096))
    if any(t >= hm for t in inflight_tokens):
        return True
    if any(rem >= hm for rem in running_pf_rem):
        return True
    if room and waiting_head_fresh is not None and waiting_head_fresh >= hm:
        return True
    return False


try:
    from vllm.logger import init_logger
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler
    from vllm.v1.core.sched.request_queue import create_request_queue

    _HAS_VLLM = True
    logger = init_logger("inkling_sched")
except Exception:  # noqa: BLE001 - allow pure-policy unit tests outside container
    _HAS_VLLM = False
    AsyncScheduler = object  # type: ignore[misc,assignment]
    create_request_queue = None  # type: ignore[assignment]
    logger = None  # type: ignore[assignment]


if _HAS_VLLM:

    class InklingScheduler(AsyncScheduler):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self._v17_env = env_params()
            self._v17_p = dict(self._v17_env)
            self._v17_ctl = os.environ.get("INKLING_SCHED_CTL", "")
            self._v17_ctl_mtime: int | None = None
            self._v17_ctl_next = 0.0
            self._v17_stats_path = os.environ.get(
                "INKLING_SCHED_STATS", "/dev/shm/inkling_v17_sched_stats.json"
            )
            self._v17_stats_next = 0.0
            self._v17_decode_streak = 0
            self._v17_st: dict[str, object] = {
                "steps": 0,
                "short_skip": 0,
                "pure_decode": 0,
                "pure_prefill": 0,
                "pure_fallback": 0,
                "hold": 0,
                "probe": 0,
                "probe_err": 0,
                "last_kind": "",
                "policy": str(getattr(self, "policy", "")),
                "params": self._v17_p,
                "pid": os.getpid(),
            }
            blk = getattr(self, "block_size", None)
            if not blk:
                cc = getattr(self, "cache_config", None)
                blk = getattr(cc, "block_size", None) if cc is not None else None
            self._v17_block = int(blk or 256)

            self._v17_eng = None
            f = sys._getframe(1)
            while f is not None:
                s = f.f_locals.get("self")
                if s is not None and s is not self and any(
                    c.__name__.startswith("EngineCore") for c in type(s).__mro__
                ):
                    self._v17_eng = weakref.ref(s)
                    break
                f = f.f_back

            self._v17_reload(force=True)
            logger.info(
                "InklingScheduler V17 armed: policy=%s block=%d engine_ref=%s params=%s",
                getattr(self, "policy", "?"),
                self._v17_block,
                self._v17_eng is not None,
                self._v17_p,
            )

        def _v17_reload(self, force: bool = False) -> None:
            if not self._v17_ctl:
                return
            now = time.monotonic()
            if not force and now < self._v17_ctl_next:
                return
            self._v17_ctl_next = now + 0.5
            try:
                m = os.stat(self._v17_ctl).st_mtime_ns
            except OSError:
                m = None
            if m == self._v17_ctl_mtime:
                return
            self._v17_ctl_mtime = m
            p = dict(self._v17_env)
            if m is not None:
                try:
                    with open(self._v17_ctl) as fh:
                        for k, v in json.load(fh).items():
                            if k in p:
                                p[k] = int(v)
                except Exception as e:  # noqa: BLE001
                    logger.warning("inkling_sched: bad ctl file %s: %s", self._v17_ctl, e)
                    return
            self._v17_p = p
            self._v17_st["params"] = p
            logger.info("inkling_sched params updated: %s", p)

        def _v17_write_stats(self) -> None:
            if not self._v17_stats_path:
                return
            now = time.monotonic()
            if now < self._v17_stats_next:
                return
            self._v17_stats_next = now + 2.0
            try:
                tmp = self._v17_stats_path + ".tmp"
                with open(tmp, "w") as fh:
                    json.dump(dict(self._v17_st, ts_unix=round(time.time(), 3)), fh)
                os.replace(tmp, self._v17_stats_path)
            except OSError:
                pass

        def _v17_max_active_reqs(self) -> int:
            return int(
                getattr(
                    self,
                    "max_num_running_reqs",
                    getattr(
                        self,
                        "max_num_active_reqs",
                        getattr(self.scheduler_config, "max_num_seqs", 32),
                    ),
                )
            )

        def _v17_fresh(self, r) -> int:
            fresh = r.num_tokens - r.num_computed_tokens
            if (
                self._v17_p["probe"]
                and fresh > min(self._v17_p["short_max"], self._v17_p["pure_min_fresh"])
                and r.num_computed_tokens == 0
            ):
                try:
                    _, n_cached, _ = self.kv_cache_manager.get_computed_blocks(r)
                    self._v17_st["probe"] = int(self._v17_st["probe"]) + 1
                    fresh = max(1, r.num_tokens - int(n_cached))
                except Exception as e:  # noqa: BLE001
                    self._v17_st["probe_err"] = int(self._v17_st["probe_err"]) + 1
                    if self._v17_st["probe_err"] == 1:
                        logger.warning(
                            "inkling_sched: prefix probe failed (%s); disabling probe", e
                        )
                    self._v17_p["probe"] = 0
            return fresh

        def _run_pure_decode(self, throttle_prefills: bool):
            removed: list[tuple[int, object]] = []
            for i, r in enumerate(self.running):
                if r.num_computed_tokens < r.num_prompt_tokens:
                    removed.append((i, r))
            for _, r in reversed(removed):
                self.running.remove(r)
            orig_waiting = self.waiting
            self.waiting = create_request_queue(self.policy)
            try:
                return super().schedule(throttle_prefills)
            finally:
                self.waiting = orig_waiting
                for i, r in removed:
                    self.running.insert(min(i, len(self.running)), r)

        def _run_pure_prefill(self, throttle_prefills: bool, pf_chunk: int):
            decoders: list[tuple[int, object]] = []
            for i, r in enumerate(self.running):
                if r.num_computed_tokens >= r.num_prompt_tokens:
                    decoders.append((i, r))
            for _, r in reversed(decoders):
                self.running.remove(r)

            cfg = self.scheduler_config
            spec = self.vllm_config.speculative_config
            draft_slots = spec.max_num_new_slots_for_drafting if spec is not None else 0
            chunk_aligned = round_up(max(pf_chunk, self._v17_block), self._v17_block)

            old_thr = cfg.long_prefill_token_threshold
            old_adapt = getattr(self, "adaptive_long_prefill_threshold", False)
            old_max_batched = cfg.max_num_batched_tokens
            old_max_sched = self.max_num_scheduled_tokens

            cfg.long_prefill_token_threshold = chunk_aligned
            self.adaptive_long_prefill_threshold = False
            cfg.max_num_batched_tokens = min(old_max_batched, chunk_aligned + draft_slots)
            self.max_num_scheduled_tokens = min(old_max_sched, chunk_aligned)
            try:
                return super().schedule(throttle_prefills)
            finally:
                cfg.long_prefill_token_threshold = old_thr
                self.adaptive_long_prefill_threshold = old_adapt
                cfg.max_num_batched_tokens = old_max_batched
                self.max_num_scheduled_tokens = old_max_sched
                for i, r in decoders:
                    self.running.insert(min(i, len(self.running)), r)

        def schedule(self, throttle_prefills: bool = False):
            self._v17_reload()
            p = self._v17_p
            if not p["enable"]:
                return super().schedule(throttle_prefills)

            self._v17_st["steps"] = int(self._v17_st["steps"]) + 1
            pure_min = int(p["pure_min_fresh"])

            has_decoders = False
            running_pf_rem: list[int] = []
            has_long_running_pf = False
            for r in self.running:
                if r.num_computed_tokens < r.num_prompt_tokens:
                    rem = r.num_prompt_tokens - r.num_computed_tokens
                    running_pf_rem.append(rem)
                    if rem >= pure_min:
                        has_long_running_pf = True
                else:
                    has_decoders = True

            waiting_len = len(self.waiting)
            room = len(self.running) < self._v17_max_active_reqs()

            # 1. ds41f Prefix-Cache-Probed Short-Skip FIRST!
            # Always admit urgent short/warm-cache prompts ahead of any running long prefill
            # (both solo 1M prefill AND mixed C=8+128K contention), dropping TTFT from ~19-113s -> <1s!
            if running_pf_rem and waiting_len > 0 and room:
                fresh_list: list[int] = []
                scan = int(p["scan"])
                for idx, r in enumerate(self.waiting):
                    if idx >= scan:
                        break
                    fresh_list.append(self._v17_fresh(r))
                thr, kind = decide_short_skip(
                    running_pf_rem, fresh_list, p, self._v17_block
                )
                if thr > 0:
                    cfg = self.scheduler_config
                    old_thr = cfg.long_prefill_token_threshold
                    old_adapt = getattr(self, "adaptive_long_prefill_threshold", False)
                    cfg.long_prefill_token_threshold = thr
                    self.adaptive_long_prefill_threshold = False
                    removed: list[tuple[int, object]] = []
                    if kind == "short-skip":
                        for i, r in enumerate(self.running):
                            if (
                                r.num_computed_tokens < r.num_prompt_tokens
                                and (r.num_prompt_tokens - r.num_computed_tokens) > thr
                            ):
                                removed.append((i, r))
                        for _, r in reversed(removed):
                            self.running.remove(r)
                    try:
                        out = super().schedule(throttle_prefills)
                    finally:
                        cfg.long_prefill_token_threshold = old_thr
                        self.adaptive_long_prefill_threshold = old_adapt
                        for i, r in removed:
                            self.running.insert(min(i, len(self.running)), r)
                    self._v17_st["short_skip"] = int(self._v17_st["short_skip"]) + 1
                    self._v17_st["last_kind"] = kind
                    self._v17_write_stats()
                    return out

            waiting_head_fresh: int | None = None
            if waiting_len > 0 and room:
                head_req = self.waiting.peek_request()
                waiting_head_fresh = self._v17_fresh(head_req)
            has_long_waiting_pf = (
                waiting_head_fresh is not None and waiting_head_fresh >= pure_min
            )

            # 2. Pure-Prefill / Pure-Decode separation when active decoders meet a long prefill
            mode = decide_pure_mode(
                has_decoders=has_decoders,
                has_long_running_pf=has_long_running_pf,
                has_long_waiting_pf=has_long_waiting_pf,
                decode_streak=self._v17_decode_streak,
                p=p,
            )
            if mode == "pure_decode":
                out = self._run_pure_decode(throttle_prefills)
                if out.total_num_scheduled_tokens > 0:
                    self._v17_decode_streak += 1
                    self._v17_st["pure_decode"] = int(self._v17_st["pure_decode"]) + 1
                    self._v17_st["last_kind"] = "pure_decode"
                    self._v17_write_stats()
                    return out
                # All decoders were blocked (e.g. async placeholder at max_tokens); run prefill now.
                mode = "pure_prefill"

            if mode == "pure_prefill":
                self._v17_decode_streak = 0
                out = self._run_pure_prefill(throttle_prefills, int(p["pure_pf_chunk"]))
                if out.total_num_scheduled_tokens > 0:
                    self._v17_st["pure_prefill"] = int(self._v17_st["pure_prefill"]) + 1
                    self._v17_st["last_kind"] = "pure_prefill"
                    self._v17_write_stats()
                    return out
                # KV pressure may require preempting a decoder; fall back safely to stock schedule().
                self._v17_st["pure_fallback"] = int(self._v17_st["pure_fallback"]) + 1
                self._v17_st["last_kind"] = "pure_fallback"
                out = super().schedule(throttle_prefills)
                self._v17_write_stats()
                return out

            # No mixed decode+long-prefill contention: reset streak
            self._v17_decode_streak = 0

            out = super().schedule(throttle_prefills)
            self._v17_st["last_kind"] = "stock"
            self._v17_write_stats()
            return out

        def has_requests(self) -> bool:
            base = super().has_requests()
            if not base or self._v17_eng is None:
                return base
            p = self._v17_p
            if not p["enable"] or not p["hold"]:
                return base
            eng = self._v17_eng()
            bq = getattr(eng, "batch_queue", None) if eng is not None else None
            if not bq:
                return base
            inflight = [item[1].total_num_scheduled_tokens for item in list(bq)]
            running_pf_rem = [
                r.num_prompt_tokens - r.num_computed_tokens
                for r in self.running
                if r.num_computed_tokens < r.num_prompt_tokens
            ]
            head_fresh: int | None = None
            if len(self.waiting):
                r0 = self.waiting.peek_request()
                head_fresh = r0.num_tokens - r0.num_computed_tokens
            room = len(self.running) < self._v17_max_active_reqs()
            if decide_hold(p, inflight, running_pf_rem, head_fresh, room):
                self._v17_st["hold"] = int(self._v17_st["hold"]) + 1
                return False
            return base
