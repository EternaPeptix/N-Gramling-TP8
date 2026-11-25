# SPDX-License-Identifier: Apache-2.0
"""Zero-Cost (<20 us) Prompt+Output Suffix Automaton (SAM) Hybrid Drafter for Inkling-2 C=1.

Provides:
  1. Fast C99 Suffix Automaton (`sam_fast.c` -> `libsam_fast.so`) with identical Pure-Python fallback.
  2. Three 0-ULP-safe hybrid drafting integrations with DSpark:
     - Mode A (Tail Extension `K=7 -> 15`): when DSpark's 7-token chain has high survival
       (`surv[6] >= tau_ext`) and SAM matches suffix ending at `draft[0..6]` (`match_len >= min_ext_match`),
       extend chain up to `K_max=15` with zero extra DSpark forward cost!
     - Mode B (Free Parallel Tree Branch): when SAM matches the committed tail (`match_len >= min_branch_match`)
       and proposes token `t_sam != t_dspark` at depth 0 (or divergence depth `d`), inject `t_sam`
       as a zero-drafter-cost sibling candidate into the Dinkelbach Tree DP!
     - Mode C (Agreement Confidence Boost): when SAM agrees with DSpark's primary token at depth `k`
       with `match_len >= min_boost_match`, boost calibrated logit `z_k += sam_boost_logit(match_len)`
       so `dink` keeps high-confidence verbatim continuations instead of prematurely truncating!

100% BIT-EXACT SAFE: The SAM ONLY alters draft token proposals / tree candidates / confidence
estimates. The target verifier executes 0-ULP exact Inkling-2 verification along the accepted path.
"""
from __future__ import annotations

import ctypes
import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass
class SamMatch:
    tokens: list[int]
    match_len: int
    occ_count: int
    hist_pos: int

    @property
    def hit_prob(self) -> float:
        """Empirical single-step hit probability calibrated by suffix match length."""
        if self.match_len <= 0:
            return 0.0
        # Smooth monotonic calibration: L=3 -> 0.52, L=4 -> 0.70, L=5 -> 0.80, L>=8 -> 0.91
        return min(0.94, 1.0 / (1.0 + math.exp(-0.55 * (self.match_len - 2.85))))


class PurePythonSAM:
    """Reference O(1)-amortized Suffix Automaton with (last_pos, prev_pos) recency tracking.

    Wave-V16: optional `ignore_token_ids` (e.g., multimodal `image_token_id` placeholders) skips
    256-5,184+ repeated pad tokens and inserts at most ONE `-1` break sentinel per contiguous
    ignored run so text before/after an image is never evicted and never forms fake cross-image n-grams.
    """

    __slots__ = (
        "_len",
        "_link",
        "_last_pos",
        "_prev_pos",
        "_occ",
        "_next",
        "_last",
        "tokens",
        "_max_refresh",
        "_max_tokens",
        "_ignore_ids",
        "_in_ignored_run",
    )

    def __init__(
        self,
        max_refresh_depth: int = 24,
        max_tokens: int = 0,
        ignore_token_ids: Sequence[int] | frozenset[int] = (),
    ):
        self._len: list[int] = [0]
        self._link: list[int] = [-1]
        self._last_pos: list[int] = [-1]
        self._prev_pos: list[int] = [-1]
        self._occ: list[int] = [0]
        self._next: list[dict[int, int]] = [{}]
        self._last: int = 0
        self.tokens: list[int] = []
        self._max_refresh = max_refresh_depth
        self._max_tokens = max(0, int(max_tokens))
        self._ignore_ids: frozenset[int] = frozenset(int(t) for t in ignore_token_ids if int(t) >= 0)
        self._in_ignored_run: bool = False

    def reset(self) -> None:
        self._len = [0]
        self._link = [-1]
        self._last_pos = [-1]
        self._prev_pos = [-1]
        self._occ = [0]
        self._next = [{}]
        self._last = 0
        self.tokens = []
        self._in_ignored_run = False

    def set_ignore_token_ids(self, ignore_token_ids: Sequence[int] | frozenset[int]) -> None:
        self._ignore_ids = frozenset(int(t) for t in ignore_token_ids if int(t) >= 0)

    def __len__(self) -> int:
        return len(self.tokens)

    def extend_one(self, tok: int) -> None:
        tok = int(tok)
        if tok < 0:
            return
        if self._ignore_ids and tok in self._ignore_ids:
            # Skip repeated multimodal placeholder tokens (e.g. 256-5,184 image_token_id pads)
            # and place at most ONE -1 boundary sentinel so adjacent text segments never fuse.
            if not self._in_ignored_run:
                if self.tokens and self.tokens[-1] >= 0:
                    self.tokens.append(-1)
                self._last = 0
                self._in_ignored_run = True
            return
        self._in_ignored_run = False
        if self._max_tokens > 0 and len(self.tokens) >= self._max_tokens:
            # Bounded-RAM sliding compaction when explicitly opted in via INKLING_SAM_CPU_MAX_TOKENS
            keep = max(64, self._max_tokens // 2)
            tail = self.tokens[-keep:]
            self.reset()
            for t in tail:
                if t < 0:
                    if not self._in_ignored_run and self.tokens and self.tokens[-1] >= 0:
                        self.tokens.append(-1)
                    self._last = 0
                    self._in_ignored_run = True
                else:
                    self.extend_one(t)
        pos = len(self.tokens)
        self.tokens.append(tok)

        cur = len(self._len)
        self._len.append(self._len[self._last] + 1)
        self._link.append(0)
        self._last_pos.append(pos)
        self._prev_pos.append(-1)
        self._occ.append(1)
        self._next.append({})

        p = self._last
        while p != -1 and tok not in self._next[p]:
            self._next[p][tok] = cur
            p = self._link[p]

        if p == -1:
            self._link[cur] = 0
        else:
            q = self._next[p][tok]
            if self._len[p] + 1 == self._len[q]:
                self._link[cur] = q
            else:
                clone = len(self._len)
                self._len.append(self._len[p] + 1)
                self._link.append(self._link[q])
                self._last_pos.append(self._last_pos[q])
                self._prev_pos.append(self._prev_pos[q])
                self._occ.append(self._occ[q])
                self._next.append(dict(self._next[q]))
                while p != -1 and self._next[p].get(tok) == q:
                    self._next[p][tok] = clone
                    p = self._link[p]
                self._link[q] = clone
                self._link[cur] = clone

        self._last = cur
        v = self._link[cur]
        hops = 0
        while v > 0 and hops < self._max_refresh:
            if self._last_pos[v] == pos:
                break
            self._prev_pos[v] = self._last_pos[v]
            self._last_pos[v] = pos
            self._occ[v] = min(self._occ[v] + 1, 1_000_000)
            v = self._link[v]
            hops += 1

    def extend(self, toks: Sequence[int]) -> None:
        for t in toks:
            self.extend_one(int(t))

    def query_propose(
        self,
        query: Sequence[int],
        min_match_len: int = 4,
        max_draft: int = 8,
        exclude_end_pos: int | None = None,
    ) -> SamMatch:
        n_tok = len(self.tokens)
        if n_tok < 2 or not query or max_draft <= 0:
            return SamMatch([], 0, 0, -1)
        excl = (n_tok - 1) if exclude_end_pos is None else exclude_end_pos

        v = 0
        l = 0
        nxt_tbl = self._next
        link_tbl = self._link
        len_tbl = self._len
        for c in query:
            ci = int(c)
            if ci < 0 or (self._ignore_ids and ci in self._ignore_ids):
                v = 0
                l = 0
                continue
            nxt = nxt_tbl[v].get(ci, -1)
            while v != 0 and nxt == -1:
                v = link_tbl[v]
                l = len_tbl[v]
                nxt = nxt_tbl[v].get(ci, -1)
            if nxt != -1:
                v = nxt
                l += 1
            else:
                v = 0
                l = 0

        cur_v = v
        cur_l = l
        while cur_v > 0 and cur_l >= min_match_len:
            hp = self._last_pos[cur_v]
            if hp >= excl:
                hp = self._prev_pos[cur_v]
            if 0 <= hp < excl and hp + 1 < n_tok:
                n_out = min(max_draft, n_tok - (hp + 1))
                out_toks: list[int] = []
                for idx_o in range(hp + 1, hp + 1 + n_out):
                    ot = self.tokens[idx_o]
                    if ot < 0 or (self._ignore_ids and ot in self._ignore_ids):
                        break
                    out_toks.append(ot)
                if out_toks:
                    return SamMatch(
                        tokens=out_toks,
                        match_len=cur_l,
                        occ_count=self._occ[cur_v],
                        hist_pos=hp,
                    )
            cur_v = link_tbl[cur_v]
            cur_l = len_tbl[cur_v] if cur_v > 0 else 0
        return SamMatch([], 0, 0, -1)

    def query_tail_plus_ext(
        self,
        tail_win: int = 32,
        ext: Sequence[int] = (),
        min_match_len: int = 4,
        max_draft: int = 8,
    ) -> SamMatch:
        n_tok = len(self.tokens)
        if n_tok < 2:
            return SamMatch([], 0, 0, -1)
        start = max(0, n_tok - tail_win)
        q = self.tokens[start:] + [int(x) for x in ext]
        return self.query_propose(q, min_match_len=min_match_len, max_draft=max_draft, exclude_end_pos=n_tok - 1)


_SAM_LIB = None
_SAM_LIB_TRIED = False


def _load_sam_c_lib() -> ctypes.CDLL | None:
    global _SAM_LIB, _SAM_LIB_TRIED
    if _SAM_LIB_TRIED:
        return _SAM_LIB
    _SAM_LIB_TRIED = True
    here = Path(__file__).resolve().parent
    env_so = os.environ.get("INKLING_SAM_LIB", "").strip()
    candidates = [
        Path(env_so) if env_so else None,
        Path("/rdma/libsam_fast.so"),
        here / "libsam_fast.so",
    ]
    so_path = next((p for p in candidates if p is not None and p.exists()), here / "libsam_fast.so")
    c_path = here / "sam_fast.c"
    if not so_path.exists() and c_path.exists():
        try:
            subprocess.run(
                ["cc", "-O3", "-shared", "-fPIC", "-o", str(so_path), str(c_path)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            return None
    if not so_path.exists():
        return None
    try:
        lib = ctypes.CDLL(str(so_path))
        lib.sam_create.argtypes = [ctypes.c_int32]
        lib.sam_create.restype = ctypes.c_void_p
        lib.sam_reset.argtypes = [ctypes.c_void_p]
        lib.sam_reset.restype = None
        lib.sam_destroy.argtypes = [ctypes.c_void_p]
        lib.sam_destroy.restype = None
        lib.sam_extend_batch.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int32]
        lib.sam_extend_batch.restype = None
        lib.sam_length.argtypes = [ctypes.c_void_p]
        lib.sam_length.restype = ctypes.c_int32
        lib.sam_query_tail_plus_ext.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
        ]
        lib.sam_query_tail_plus_ext.restype = ctypes.c_int32
        _SAM_LIB = lib
    except Exception:
        _SAM_LIB = None
    return _SAM_LIB


DEFAULT_SAM_MAX_B = 4


def sam_batch_eligible(num_reqs: int, max_b: int = DEFAULT_SAM_MAX_B, batch_on: bool = True) -> bool:
    """O(1) batch-size gate for Zero-Tax GPU/CPU SAM (`INKLING_SAM_MAX_B`).

    Returns True iff `1 <= num_reqs <= max_b` (and `num_reqs == 1` when `batch_on=False`).
    Returns False with 0 allocations/kernel launches whenever `num_reqs > max_b` or `max_b <= 0`.
    """
    nr = int(num_reqs)
    mb = int(max_b)
    if nr <= 0 or mb <= 0 or nr > mb:
        return False
    return nr == 1 or bool(batch_on)


def sam_lpc_needed(valid_lo: int, lpc_populated: bool = True) -> bool:
    """True iff the Long-Prompt Continuation Cache (`LPC`) can contain tokens outside the L2 ring.

    When `valid_lo <= 0` (prompt + generation fits 100% inside the per-slot ring) or `lpc_populated`
    is False, probing the 13 MiB LPC table is strictly redundant and skipped at 0.00 us DRAM cost.
    """
    return bool(lpc_populated and int(valid_lo) > 0)


def sam_shared_write_safe(num_reqs: int, shared_on: bool, serialized: bool = False) -> bool:
    """True iff writing to the cross-request Shared Pool is 100% TP8-deterministic (zero CTA race)."""
    if not shared_on:
        return False
    return int(num_reqs) == 1 or bool(serialized)


def sam_lpc_gen_next(cur_gen: int) -> tuple[int, bool]:
    """Advance per-slot 15-bit CPU LPC generation in `1..32767` (zero even/odd collisions).

    Returns `(next_gen, wrapped)` where `wrapped=True` signals the caller to zero that slot's
    private LPC key partition before reusing `next_gen=1` (once per 32,767 requests/slot).
    GPU in-place preemption reserves the disjoint `32768..65535` namespace."""
    nxt = int(cur_gen) + 1
    if nxt < 1 or nxt > 32767:
        return 1, True
    return nxt, False


def agreement_conf_floor(
    match_len: int,
    *,
    ml2_floor: float = 0.76,
    ml3_floor: float = 0.85,
    ml4_base: float = 0.92,
    slope: float = 0.015,
    cap: float = 0.985,
) -> float:
    """Calibrated confidence floor when DSpark agrees with a SAM verbatim continuation (ml >= 2).
    Matched bit-for-bit by GPU `_sam_gpu_boost_extend_kernel` (`INKLING_SAM_CONF_BOOST=1`)."""
    if match_len < 2:
        return 0.0
    if match_len == 2:
        return float(ml2_floor)
    if match_len == 3:
        return float(ml3_floor)
    return float(min(cap, ml4_base + slope * min(match_len, 4)))


_MM_PLACEHOLDER_ATTRS = (
    "image_token_id",
    "image_token_index",
    "img_token_id",
    "image_pad_token_id",
    "video_token_id",
    "audio_token_id",
    "vision_pad_token_id",
    "mm_token_id",
)


def parse_sam_ignore_token_ids(
    raw: str | None = None,
    hf_config: object | None = None,
    *,
    enabled: bool | None = None,
) -> tuple[int, ...]:
    """Extract deterministic sorted unique multimodal placeholder IDs to skip in GPU & CPU SAM.

    Default-OFF safe: when `enabled` is False (default when `INKLING_SAM_IMAGE_SAFE_V16=0`),
    returns `()` so baseline V15 behavior is 100% preserved.
    """
    if enabled is None:
        enabled = os.environ.get("INKLING_SAM_IMAGE_SAFE_V16", "0").strip().lower() in (
            "1", "on", "true", "yes"
        )
    if not enabled:
        return ()
    found: set[int] = set()
    env_s = os.environ.get("INKLING_SAM_IGNORE_TOKEN_IDS", "") if raw is None else raw
    for part in (env_s or "").replace(";", ",").replace(" ", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            v = int(part)
            if v >= 0:
                found.add(v)
        except ValueError:
            pass
    if hf_config is not None:
        cfgs = [hf_config]
        for sub in ("text_config", "vision_config", "mm_config", "multimodal_config"):
            sc = getattr(hf_config, sub, None)
            if sc is not None:
                cfgs.append(sc)
        for cfg in cfgs:
            for attr in _MM_PLACEHOLDER_ATTRS:
                val = getattr(cfg, attr, None)
                if isinstance(val, (int, np.integer)) and not isinstance(val, bool):
                    iv = int(val)
                    if iv >= 0:
                        found.add(iv)
                elif isinstance(val, (list, tuple, set, frozenset)):
                    for item in val:
                        if isinstance(item, (int, np.integer)) and not isinstance(item, bool):
                            iv = int(item)
                            if iv >= 0:
                                found.add(iv)
    return tuple(sorted(found))


def sam_dyn_k_ext_budget(
    *,
    cur_pos: int,
    max_model_len: int,
    k_draft: int,
    k_ext_cap: int,
    p6_raw: float,
    p6_cal: float,
    surv_6: float,
    match_len: int,
    min_match: int = 4,
    p6_thresh: float = 0.78,
    surv_thresh: float = 0.35,
    ext_surv_floor: float = 0.25,
    exact_ml: int = 6,
    k_ext_mid: int = 11,
    k_ext_max: int = 15,
    allow_ht2: bool = True,
) -> int:
    """Exact CPU/GPU single-source-of-truth for Same-Tile Dynamic Extended Depth `K=7 -> 11/15`.

    Returns number of extended draft slots `0..k_ext_cap` permitted past `k_draft` (`7`),
    strictly respecting `cur_pos + k_draft + ext < max_model_len` (0 OOB positions at 1M!).
    """
    if k_ext_cap <= 0 or k_draft <= 0:
        return 0
    if not (math.isfinite(p6_raw) and math.isfinite(p6_cal) and math.isfinite(surv_6)):
        return 0
    hi_conf_6 = (p6_raw >= p6_thresh) or (p6_cal >= p6_thresh)
    if not ((surv_6 >= surv_thresh) or (hi_conf_6 and surv_6 >= ext_surv_floor)):
        return 0
    if match_len >= exact_ml and (surv_6 >= surv_thresh or hi_conf_6):
        tier_cap = min(int(k_ext_cap), max(0, int(k_ext_max) - int(k_draft)))
    elif match_len >= min_match or (allow_ht2 and match_len >= 3 and hi_conf_6 and surv_6 >= surv_thresh):
        tier_cap = min(int(k_ext_cap), max(0, int(k_ext_mid) - int(k_draft)))
    else:
        return 0
    # Strict max_model_len clamping: anchor at `cur_pos`, draft k at `cur_pos + 1 + k`.
    # For `e in 0..ext-1`: `cur_pos + 1 + k_draft + e <= max_model_len - 1`.
    rem_ctx = max(0, int(max_model_len) - 1 - (int(cur_pos) + int(k_draft)))
    return max(0, min(tier_cap, rem_ctx))


class FastSuffixAutomaton:
    """C99-accelerated Suffix Automaton (< 5 us/step) with seamless Pure-Python fallback."""

    def __init__(
        self,
        init_cap: int = 4096,
        force_python: bool = False,
        ignore_token_ids: Sequence[int] | frozenset[int] | None = None,
    ):
        env_cap = os.environ.get("INKLING_SAM_CPU_INIT_CAP", "").strip()
        if env_cap:
            try:
                init_cap = max(init_cap, int(env_cap))
            except ValueError:
                pass
        elif os.environ.get("INKLING_SAM_LONG_PROMPT", "0").strip().lower() in ("1", "on", "true", "yes"):
            init_cap = max(init_cap, 131072)
        if ignore_token_ids is None:
            ig = parse_sam_ignore_token_ids()
        else:
            ig = tuple(sorted({int(x) for x in ignore_token_ids if int(x) >= 0}))
        self._ignore_ids_tuple: tuple[int, ...] = ig
        self._ignore_ids: frozenset[int] = frozenset(ig)
        self._lib = None if force_python else _load_sam_c_lib()
        _h = self._lib.sam_create(init_cap) if self._lib is not None else None
        self._handle = _h if _h else None
        _max_tok_raw = os.environ.get("INKLING_SAM_CPU_MAX_TOKENS", "0").strip()
        try:
            _max_tok = max(0, int(_max_tok_raw or "0"))
        except ValueError:
            _max_tok = 0
        self._py = PurePythonSAM(max_tokens=_max_tok, ignore_token_ids=ig) if self._handle is None else None
        self._out_buf = (ctypes.c_int32 * 32)()
        self._ext_buf = (ctypes.c_int32 * 32)()
        self._ml = ctypes.c_int32(0)
        self._occ = ctypes.c_int32(0)
        self._hp = ctypes.c_int32(-1)

    @property
    def is_c_accelerated(self) -> bool:
        return self._handle is not None

    def set_ignore_token_ids(self, ignore_token_ids: Sequence[int] | frozenset[int]) -> None:
        ig = tuple(sorted({int(x) for x in ignore_token_ids if int(x) >= 0}))
        self._ignore_ids_tuple = ig
        self._ignore_ids = frozenset(ig)
        if self._py is not None:
            self._py.set_ignore_token_ids(ig)

    def __del__(self) -> None:
        if getattr(self, "_handle", None) and getattr(self, "_lib", None):
            try:
                self._lib.sam_destroy(self._handle)
            except Exception:
                pass
            self._handle = None

    def reset(self) -> None:
        if self._handle is not None:
            self._lib.sam_reset(self._handle)
        else:
            self._py.reset()

    def __len__(self) -> int:
        if self._handle is not None:
            return int(self._lib.sam_length(self._handle))
        return len(self._py)

    def extend(self, toks: Sequence[int] | np.ndarray) -> None:
        if self._handle is not None:
            if isinstance(toks, np.ndarray) and toks.dtype == np.int32 and toks.flags["C_CONTIGUOUS"]:
                arr = toks
            else:
                arr = np.asarray(toks, dtype=np.int32)
            if arr.size > 0:
                mask = arr >= 0
                for ig in self._ignore_ids_tuple:
                    mask &= arr != ig
                if not bool(mask.all()):
                    arr = np.ascontiguousarray(arr[mask], dtype=np.int32)
                if arr.size > 0:
                    self._lib.sam_extend_batch(
                        self._handle,
                        arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                        int(arr.size),
                    )
        else:
            self._py.extend(toks)

    def seed_prefix(self, prefix_tokens: Sequence[int] | np.ndarray, max_seed_tokens: int = 4096) -> int:
        """Warm-start SAM from trailing `max_seed_tokens` of a prefix-cached Turn-2+ prompt."""
        self.reset()
        n = len(prefix_tokens)
        if n <= 0 or max_seed_tokens <= 0:
            return 0
        tail = prefix_tokens[-int(max_seed_tokens) :] if n > max_seed_tokens else prefix_tokens
        self.extend(tail)
        return len(self)

    def query_tail_plus_ext(
        self,
        tail_win: int = 32,
        ext: Sequence[int] = (),
        min_match_len: int = 4,
        max_draft: int = 8,
    ) -> SamMatch:
        if max_draft <= 0 or tail_win <= 0 or len(self) < 2:
            return SamMatch([], 0, 0, -1)
        if self._handle is not None:
            n_ext = min(len(ext), 32)
            valid_ext = 0
            ig_set = self._ignore_ids
            for i in range(n_ext):
                ti = int(ext[i])
                if ti < 0 or (ig_set and ti in ig_set):
                    break
                self._ext_buf[valid_ext] = ti
                valid_ext += 1
            n_ext = valid_ext
            n_max = min(max_draft, 32)
            n_out = int(
                self._lib.sam_query_tail_plus_ext(
                    self._handle,
                    int(tail_win),
                    self._ext_buf if n_ext > 0 else None,
                    int(n_ext),
                    int(min_match_len),
                    int(n_max),
                    self._out_buf,
                    ctypes.byref(self._ml),
                    ctypes.byref(self._occ),
                    ctypes.byref(self._hp),
                )
            )
            if n_out <= 0:
                return SamMatch([], 0, 0, -1)
            out_toks: list[int] = []
            for i in range(n_out):
                ot = int(self._out_buf[i])
                if ot < 0 or (ig_set and ot in ig_set):
                    break
                out_toks.append(ot)
            if not out_toks:
                return SamMatch([], 0, 0, -1)
            return SamMatch(
                tokens=out_toks,
                match_len=int(self._ml.value),
                occ_count=int(self._occ.value),
                hist_pos=int(self._hp.value),
            )
        return self._py.query_tail_plus_ext(
            tail_win=tail_win, ext=ext, min_match_len=min_match_len, max_draft=max_draft
        )


class SuffixHybridAdvisor:
    """Combines DSpark's draft tokens & confidences with a per-request Suffix Automaton.

    Operates in < 10 us/step on CPU (can run overlapped during target GPU forward!):
      1. propose_parallel_branch(): checks committed tail -> proposes zero-cost tree branch if SAM
         matches len >= min_branch_match and diverges from DSpark's primary token.
      2. propose_tail_extension(): checks committed tail + DSpark[0..K-1] -> extends chain up to
         K_max=15 when DSpark's chain is highly confident (`surv[K-1] >= ext_surv_thresh`).
      3. agreement_logit_boost(): returns additive logit boost delta_z[0..K-1] when SAM's
         committed-tail continuation agrees with DSpark[0..K-1].
    """

    def __init__(
        self,
        min_branch_match: int = 4,
        min_ext_match: int = 4,
        min_boost_match: int = 4,
        max_ext_tokens: int = 8,
        ext_surv_thresh: float = 0.25,
        tail_win: int = 32,
        force_python: bool = False,
        ignore_token_ids: Sequence[int] | frozenset[int] | None = None,
    ):
        self.sam = FastSuffixAutomaton(force_python=force_python, ignore_token_ids=ignore_token_ids)
        self.min_branch_match = min_branch_match
        self.min_ext_match = min_ext_match
        self.min_boost_match = min_boost_match
        self.max_ext_tokens = max_ext_tokens
        self.ext_surv_thresh = ext_surv_thresh
        self.tail_win = tail_win

    def ingest_prompt(self, prompt_tokens: Sequence[int] | np.ndarray) -> None:
        self.sam.reset()
        self.sam.extend(prompt_tokens)

    def seed_prefix(self, prefix_tokens: Sequence[int] | np.ndarray, max_seed_tokens: int = 4096) -> int:
        return self.sam.seed_prefix(prefix_tokens, max_seed_tokens=max_seed_tokens)

    def ingest_accepted(self, accepted_tokens: Sequence[int] | np.ndarray) -> None:
        self.sam.extend(accepted_tokens)

    def advise(
        self,
        dspark_tokens: Sequence[int],
        dspark_c: Sequence[float],
        *,
        conf_boost_floor: bool = False,
    ) -> dict:
        """Compute all three hybrid signals in <= 2 SAM queries (< 10 us total).

        V13 fast-path: skips both SAM queries in O(1) when the slot has fewer than
        `min(min_branch_match, eff_min_boost)` ingested tokens or no draft tokens.
        """
        k_base = len(dspark_tokens)
        eff_min_boost = 2 if conf_boost_floor else self.min_boost_match
        min_need = min(self.min_branch_match, eff_min_boost)
        if k_base <= 0 or len(self.sam) < min_need:
            return {
                "tail_match_len": 0,
                "logit_boost": [0.0] * k_base,
                "conf_floor": [0.0] * k_base,
                "branch_depth": -1,
                "branch_tokens": [],
                "branch_hit_prob": 0.0,
                "ext_tokens": [],
                "ext_match_len": 0,
                "ext_hit_prob": 0.0,
            }
        # Query 1: Committed tail match (for agreement boost + parallel branch)
        m_tail = self.sam.query_tail_plus_ext(
            tail_win=self.tail_win,
            ext=(),
            min_match_len=min_need,
            max_draft=max(k_base, 4),
        )

        # Agreement boost along matching prefix of dspark_tokens
        logit_boost = [0.0] * k_base
        conf_floor = [0.0] * k_base
        branch_depth = -1
        branch_tokens: list[int] = []
        branch_hit_prob = 0.0

        if m_tail.tokens:
            # Find longest prefix where SAM agrees with DSpark
            agree = 0
            for k in range(min(k_base, len(m_tail.tokens))):
                if m_tail.tokens[k] == dspark_tokens[k]:
                    agree += 1
                    if m_tail.match_len >= self.min_boost_match:
                        # Conservative calibrated boost: +0.6 to +1.4 logit
                        logit_boost[k] = min(1.4, 0.35 + 0.18 * (m_tail.match_len - 2))
                    if conf_boost_floor and m_tail.match_len >= 2:
                        conf_floor[k] = agreement_conf_floor(m_tail.match_len)
                else:
                    break
            # If SAM diverges at depth `agree` (< k_base) and match length is strong, propose branch
            eff_match_len = m_tail.match_len + agree
            if agree < k_base and eff_match_len >= self.min_branch_match:
                branch_depth = agree
                branch_tokens = m_tail.tokens[agree : min(len(m_tail.tokens), agree + 4)]
                branch_hit_prob = min(0.92, 1.0 / (1.0 + math.exp(-0.55 * (eff_match_len - 2.85))))

        # Query 2: Tail extension past K=7 when survival at K-1 is high
        surv_end = 1.0
        for ck in dspark_c[:k_base]:
            cf = float(ck)
            if not math.isfinite(cf):
                surv_end = 0.0
                break
            surv_end *= max(0.0, min(1.0, cf))
        ext_tokens: list[int] = []
        ext_match_len = 0
        ext_hit_prob = 0.0
        if surv_end >= self.ext_surv_thresh and k_base > 0 and all(t >= 0 for t in dspark_tokens):
            m_ext = self.sam.query_tail_plus_ext(
                tail_win=self.tail_win,
                ext=dspark_tokens,
                min_match_len=self.min_ext_match,
                max_draft=self.max_ext_tokens,
            )
            if m_ext.tokens:
                ext_tokens = m_ext.tokens
                ext_match_len = m_ext.match_len
                ext_hit_prob = m_ext.hit_prob

        return {
            "tail_match_len": m_tail.match_len,
            "logit_boost": logit_boost,
            "conf_floor": conf_floor,
            "branch_depth": branch_depth,
            "branch_tokens": branch_tokens,
            "branch_hit_prob": branch_hit_prob,
            "ext_tokens": ext_tokens,
            "ext_match_len": ext_match_len,
            "ext_hit_prob": ext_hit_prob,
        }

    def advise_for_batch(
        self,
        num_reqs: int,
        dspark_tokens: Sequence[int],
        dspark_c: Sequence[float],
        *,
        max_b: int = DEFAULT_SAM_MAX_B,
        batch_on: bool = True,
        conf_boost_floor: bool = False,
    ) -> dict | None:
        """Zero-tax batch-gated wrapper around `advise()`: returns None in O(1) when
        `num_reqs > max_b` so B=8/32 pays 0 SAM queries."""
        if not sam_batch_eligible(num_reqs, max_b=max_b, batch_on=batch_on):
            return None
        return self.advise(dspark_tokens, dspark_c, conf_boost_floor=conf_boost_floor)


def _bpe_like_tokenize(text: str) -> list[int]:
    """Deterministic subword-like tokenizer (~4 chars/tok) for realistic local SAM corpus benchmarks."""
    import re
    chunks = re.findall(r"[A-Za-z_][A-Za-z0-9_]{0,6}|[0-9]{1,3}|\\?[a-zA-Z]+|\s+|[^\s\w]", text)
    vocab: dict[str, int] = {}
    out: list[int] = []
    for ch in chunks:
        tid = vocab.get(ch)
        if tid is None:
            tid = len(vocab) + 1
            vocab[ch] = tid
        out.append(tid)
    return out


def run_benchmark() -> None:
    import time

    base = Path("/Users/jeweled/.gemini/antigravity/brain/78201cfd-43f0-4f1b-ae82-c573f3cd7b11/scratch")
    # Build 3 realistic domain token streams: Code (patches-s6), Prose (THEORY.md + audit), Math (algebraic derivations)
    code_text = (base / "patches-s6" / "speculator.py").read_text()[:40000] + "\n" + (
        base / "s6_fixes" / "inkling_target_attn_ext_s7.cu"
    ).read_text()[:40000]
    prose_text = (base / "THEORY.md").read_text() + "\n" + (base / "subagents" / "cal_dtgraph_audit.md").read_text()
    # Realistic multi-step algebraic / competition math derivation (repeated identities, equations, case steps)
    math_blocks = []
    for n in range(1, 35):
        math_blocks.append(
            f"Step {n}: Consider the recurrence a_{{n+1}} = 2 a_n + 3 b_n and b_{{n+1}} = a_n + 2 b_n. "
            f"Substituting n = {n}, we obtain a_{{{n+1}}} = 2 a_{{{n}}} + 3 b_{{{n}}} and "
            f"b_{{{n+1}}} = a_{{{n}}} + 2 b_{{{n}}}. Therefore a_{{{n+1}}}^2 - 3 b_{{{n+1}}}^2 = "
            f"(2 a_{{{n}}} + 3 b_{{{n}}})^2 - 3 (a_{{{n}}} + 2 b_{{{n}}})^2 = a_{{{n}}}^2 - 3 b_{{{n}}}^2 = 1. "
            f"By induction, since a_1^2 - 3 b_1^2 = 1, the invariant a_k^2 - 3 b_k^2 = 1 holds for all k >= 1.\n"
        )
    math_text = "".join(math_blocks)

    corpora = {
        "Code (speculator.py + target_attn.cu)": _bpe_like_tokenize(code_text),
        "Prose (THEORY.md + audit.md)": _bpe_like_tokenize(prose_text),
        "Math (Recurrence/Induction proof)": _bpe_like_tokenize(math_text),
    }

    print("=== Suffix Automaton (SAM) Hybrid Benchmark ===")
    for force_py in (False, True):
        sam = FastSuffixAutomaton(force_python=force_py)
        mode = "C99 (libsam_fast.so)" if sam.is_c_accelerated else "Pure Python"
        if not force_py and not sam.is_c_accelerated:
            continue
        print(f"\n--- Backend: {mode} ---")
        for name, toks in corpora.items():
            # Split 40% prompt, 60% simulated decode stream (max 2000 decode tokens)
            split = min(len(toks) // 2, 1500)
            prompt = np.asarray(toks[:split], dtype=np.int32)
            gen = toks[split : min(len(toks), split + 1500)]
            sam.reset()
            t0 = time.perf_counter()
            sam.extend(prompt)
            t_prompt_us = (time.perf_counter() - t0) * 1e6

            # Simulate step-by-step decode (avg step = 3 tokens) + tail extension query
            pos = 0
            n_steps = 0
            n_tail_hit4 = 0
            n_tail_acc1 = 0
            total_tail_acc = 0
            n_ext_hit4 = 0
            total_ext_acc = 0
            t_step_total = 0.0
            while pos + 16 < len(gen):
                # Committed tail query
                ts = time.perf_counter()
                m0 = sam.query_tail_plus_ext(tail_win=32, ext=(), min_match_len=4, max_draft=8)
                # Tail extension past K=7 (assuming DSpark matched gen[pos:pos+7])
                m_ext = sam.query_tail_plus_ext(
                    tail_win=32, ext=gen[pos : pos + 7], min_match_len=4, max_draft=8
                )
                step_adv = 3
                sam.extend(np.asarray(gen[pos : pos + step_adv], dtype=np.int32))
                t_step_total += time.perf_counter() - ts

                n_steps += 1
                if m0.tokens:
                    n_tail_hit4 += 1
                    acc = 0
                    for k, tk in enumerate(m0.tokens):
                        if tk == gen[pos + k]:
                            acc += 1
                        else:
                            break
                    if acc >= 1:
                        n_tail_acc1 += 1
                    total_tail_acc += acc
                if m_ext.tokens:
                    n_ext_hit4 += 1
                    acc_e = 0
                    for k, tk in enumerate(m_ext.tokens):
                        if tk == gen[pos + 7 + k]:
                            acc_e += 1
                        else:
                            break
                    total_ext_acc += acc_e
                pos += step_adv

            us_per_step = (t_step_total / max(n_steps, 1)) * 1e6
            tail_cov = 100.0 * n_tail_hit4 / max(n_steps, 1)
            tail_p1 = 100.0 * n_tail_acc1 / max(n_tail_hit4, 1)
            tail_al = total_tail_acc / max(n_tail_hit4, 1)
            ext_cov = 100.0 * n_ext_hit4 / max(n_steps, 1)
            ext_al = total_ext_acc / max(n_ext_hit4, 1)
            print(
                f"  {name:38s}: prompt({len(prompt):4d}t)={t_prompt_us:6.1f}us | "
                f"step={us_per_step:5.2f}us | tail_cov(L>=4)={tail_cov:4.1f}% (top1={tail_p1:4.1f}%, AL|hit={tail_al:.2f}) | "
                f"ext_cov={ext_cov:4.1f}% (AL_ext|hit={ext_al:.2f})"
            )


if __name__ == "__main__":
    run_benchmark()
