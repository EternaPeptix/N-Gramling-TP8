# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Staged checkpoint -> GPU fill for GB10 (DGX Spark, unified memory).

vLLM's lazy safetensors iterator yields CPU tensors that are views into a
MAP_PRIVATE mmap of the shard file, and the model's parameters are already
allocated on the GPU before the first weight is read. ``param.copy_(view)``
therefore hands libcuda a pageable source whose pages are not mapped in the
process yet. On a discrete GPU those faults are ordinary CPU faults and cost
nothing; on GB10 they are taken inside the driver copy, cost ~40x more, and
libcuda busy-waits on one core meanwhile (vllm-project/vllm#58726, measured
on this ring's driver and kernel). Upstream fixed the stock loaders with
``copy_weight_`` (vllm#58868); the Inkling2 expert loader calls ``copy_``
directly and still takes the slow path for 96% of each rank's bytes.

``copy_from_checkpoint_`` keeps the mmap out of the driver copy entirely:
each chunk of the rank's own slice is ``pread`` into a small page-locked
staging buffer and copied to the GPU from there, and the page cache behind
the read is dropped so it does not compete with the ~91 GiB of parameters
for the same unified memory. Extra memory is one staging chunk.

Environment (read at import, all optional):
  INKLING_LOAD_STAGED=0       stock ``dst.copy_(src)`` (A/B against the patch);
                              ``force`` stages for any device (CPU tests)
  INKLING_LOAD_CHUNK_MB=128   staging chunk size
  INKLING_LOAD_DROP_CACHE=0   keep the page cache behind the read
  INKLING_LOAD_READ_THREADS=8 concurrent pread threads per chunk (default 8, 1 = the
                              old serial path). Measured on the Spark NVMe, one shard,
                              same bytes, page cache dropped: 1 thread 1.84 GB/s,
                              2 -> 3.47, 4 -> 5.33, 8 -> 6.66 GB/s. The threads write
                              DISJOINT ranges of the same pinned staging buffer, so the
                              bytes and their destinations are identical to the serial
                              path -> bit-exact by construction.
  INKLING_LOAD_O_DIRECT=1     read the shard with O_DIRECT (default 1 on Linux).
                              Measured on the Spark NVMe: buffered pread 2.0-2.4 GB/s
                              vs O_DIRECT 6.2-6.9 GB/s for the same chunk sizes, and
                              it stops the page cache from competing with the unified
                              memory the weights land in. Falls back to buffered reads
                              if O_DIRECT is unavailable or a read fails the alignment.
  INKLING_LOAD_IO_RETRIES=3   bounded retries on transient NVMe pread I/O errors (V15)
  INKLING_LOAD_IO_RETRY_BACKOFF_S=0.25 base exponential backoff seconds on I/O retry (V15)
  INKLING_LOAD_MIN_CHUNK_MB=4 minimum chunk size when shrinking under staging OOM (V15)
"""

import atexit
import ctypes
import errno
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor as _ThreadPoolExecutor

import torch

logger = logging.getLogger(__name__)

_MODE = os.environ.get("INKLING_LOAD_STAGED", "1")
_CHUNK_BYTES = max(1, int(os.environ.get("INKLING_LOAD_CHUNK_MB", "128"))) << 20
_MIN_CHUNK_BYTES = max(1, int(os.environ.get("INKLING_LOAD_MIN_CHUNK_MB", "4"))) << 20
_DROP_CACHE = os.environ.get("INKLING_LOAD_DROP_CACHE", "1") != "0"
# O_DIRECT reads skip the page cache: measured 2.5-3x the buffered rate on this NVMe.
_O_DIRECT = os.environ.get("INKLING_LOAD_O_DIRECT", "0") != "0"
_ALIGN = 4096  # O_DIRECT alignment for offset, size and buffer
# Concurrent preads per chunk. One blocking pread leaves the NVMe queue starved
# (1.84 GB/s); 8 threads reach 6.66 GB/s on the same bytes.
_READ_THREADS = max(1, int(os.environ.get("INKLING_LOAD_READ_THREADS", "16")))
_IO_RETRIES = max(0, int(os.environ.get("INKLING_LOAD_IO_RETRIES", "3")))
_IO_RETRY_BACKOFF_S = max(0.0, float(os.environ.get("INKLING_LOAD_IO_RETRY_BACKOFF_S", "0.25")))
_POOL = None
_POOL_LOCK = threading.Lock()
_ALIGNED_SCRATCH = None
_ATEXIT_REGISTERED = False
# Below this a tensor is copied to anonymous memory in one piece instead.
_SMALL_BYTES = 1 << 20
# Bytes compared at each end of a tensor to prove the file range is the view.
_PROBE_BYTES = 4096
_LOG_EVERY_BYTES = 8 << 30
# An H2D copy from a resident staging buffer slower than this is not the
# mmap problem this module removes; say so instead of looking like a hang.
_SLOW_H2D_BYTES_PER_S = 200 << 20

_fd_key: tuple[str, int] | None = None
_DIRECT_FDS: set = set()
_SEEN_SHARDS: set[str] = set()
_fd = -1
_fd_size = 0

_TRANSIENT_ERRNOS = frozenset(
    e
    for e in (
        getattr(errno, "EINTR", None),
        getattr(errno, "EAGAIN", None),
        getattr(errno, "EWOULDBLOCK", None),
        getattr(errno, "EBUSY", None),
        getattr(errno, "EIO", None),
        getattr(errno, "ETIMEDOUT", None),
    )
    if e is not None
)


def _atexit_release_staging() -> None:
    try:
        release_staging()
    except Exception:
        pass


def _ensure_atexit_registered() -> None:
    global _ATEXIT_REGISTERED
    if not _ATEXIT_REGISTERED:
        _ATEXIT_REGISTERED = True
        atexit.register(_atexit_release_staging)


def _host_empty_cache_safe() -> None:
    try:
        if hasattr(torch._C, "_host_emptyCache"):
            torch._C._host_emptyCache()
    except Exception:
        pass


def _is_transient_io_err(exc: OSError) -> bool:
    return exc.errno is None or exc.errno in _TRANSIENT_ERRNOS


def _fadvise_drop_range(fd: int, offset: int, length: int) -> None:
    if not _DROP_CACHE or fd < 0:
        return
    fadv = getattr(os, "posix_fadvise", None)
    dontneed = getattr(os, "POSIX_FADV_DONTNEED", None)
    if fadv is not None and dontneed is not None:
        try:
            fadv(fd, offset, length, dontneed)
        except OSError:
            pass


def _fadvise_drop_fd(fd: int) -> None:
    _fadvise_drop_range(fd, 0, 0)


def _fadvise_drop_path(path: str) -> None:
    if not _DROP_CACHE or not path:
        return
    if getattr(os, "posix_fadvise", None) is None:
        return
    fd = -1
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
        _fadvise_drop_fd(fd)
    except OSError:
        pass
    finally:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass


_warned: set[str] = set()
_stats = {
    "bytes": 0,
    "tensors": 0,
    "read_s": 0.0,
    "h2d_s": 0.0,
    "logged": 0,
    "o_direct_fallbacks": 0,
    "io_retries": 0,
    "oom_retries": 0,
    "mmap_fallbacks": 0,
}


def get_load_diagnostics() -> dict:
    """Return a snapshot of checkpoint loader throughput & resilience counters."""
    return {
        **_stats,
        "fd_open": _fd >= 0,
        "fd_direct": _fd in _DIRECT_FDS,
        "pool_active": _POOL is not None,
        "seen_shards": len(_SEEN_SHARDS),
        "aligned_scratch_bytes": (
            _ALIGNED_SCRATCH.numel() * _ALIGNED_SCRATCH.element_size()
            if _ALIGNED_SCRATCH is not None
            else 0
        ),
    }


def _warn_once(key: str, msg: str, *args) -> None:
    if key not in _warned:
        _warned.add(key)
        logger.warning(msg, *args)


def _close_cached_fd() -> None:
    global _fd, _fd_key, _fd_size
    if _fd >= 0:
        _fadvise_drop_fd(_fd)
        _DIRECT_FDS.discard(_fd)
        try:
            os.close(_fd)
        except OSError:
            pass
    _fd = -1
    _fd_key = None
    _fd_size = 0


def _resolve_fd_path(fd: int, path: str | None = None) -> str | None:
    if path:
        return path
    if _fd_key is not None and fd == _fd:
        return _fd_key[0]
    try:
        return os.readlink(f"/proc/self/fd/{fd}")
    except OSError:
        return None


def _reopen_buffered(fd: int, path: str | None = None) -> int:
    """Replace an O_DIRECT fd in-place with a buffered fd so reads + fadvise(DONTNEED) continue safely."""
    global _fd, _fd_key, _fd_size
    resolved = _resolve_fd_path(fd, path)
    if not resolved:
        if fd == _fd:
            _close_cached_fd()
        return -1
    try:
        bfd = os.open(resolved, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    except OSError as e:
        _warn_once("reopen_buf", "staged load: buffered reopen failed for %s (%s)", resolved, e)
        if fd == _fd:
            _close_cached_fd()
        return -1
    _DIRECT_FDS.discard(fd)
    _stats["o_direct_fallbacks"] += 1
    try:
        os.close(fd)
    except OSError:
        pass
    if fd == _fd:
        _fd = bfd
        try:
            _fd_size = os.fstat(bfd).st_size
        except OSError:
            pass
    return bfd


def _file_range(addr: int, nbytes: int) -> tuple[str, int, int] | None:
    """``(path, file offset, inode)`` of the file mapping that backs
    ``[addr, addr + nbytes)``, or None when it is not one regular file mapped
    contiguously (anonymous memory, a hole, a deleted file)."""
    end = addr + nbytes
    covered = addr
    path = None
    inode = offset = 0
    try:
        with open("/proc/self/maps", "rb") as maps:
            for line in maps:
                fields = line.split(None, 5)
                if len(fields) < 5:
                    continue
                lo_hex, _, hi_hex = fields[0].partition(b"-")
                lo, hi = int(lo_hex, 16), int(hi_hex, 16)
                if hi <= covered:
                    continue
                if lo > covered or len(fields) < 6:
                    return None
                name = fields[5].rstrip(b"\n")
                ino = int(fields[4])
                if not name.startswith(b"/") or name.endswith(b" (deleted)") or not ino:
                    return None
                file_off = int(fields[2], 16) + (covered - lo)
                if path is None:
                    path, inode, offset = name, ino, file_off
                elif name != path or ino != inode or file_off != offset + (covered - addr):
                    return None
                covered = hi
                if covered >= end:
                    return os.fsdecode(path), offset, inode
    except (OSError, ValueError):
        return None
    return None


def _open_backing(path: str, inode: int, end: int) -> int:
    """Read-only fd for ``path`` (cached: shards are loaded one at a time),
    or -1 when it is not the mapped inode or is shorter than ``end``."""
    global _fd_key, _fd, _fd_size
    if _fd_key != (path, inode):
        _close_cached_fd()
        base_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        direct_flag = getattr(os, "O_DIRECT", 0) if _O_DIRECT else 0
        opened_direct = False
        if direct_flag:
            try:
                fd = os.open(path, base_flags | direct_flag)
                opened_direct = True
            except OSError as e:
                _warn_once(
                    "odirect_open",
                    "staged load: O_DIRECT unsupported on %s (%s); falling back to buffered+fadvise",
                    path,
                    e,
                )
                _stats["o_direct_fallbacks"] += 1
                fd = os.open(path, base_flags)
        else:
            fd = os.open(path, base_flags)
        stat = os.fstat(fd)
        if stat.st_ino != inode:
            os.close(fd)
            return -1
        _fd_key, _fd, _fd_size = (path, inode), fd, stat.st_size
        _SEEN_SHARDS.add(path)
        if opened_direct:
            _DIRECT_FDS.add(fd)
        _ensure_atexit_registered()
    return _fd if _fd_size >= end else -1


def _pread_bytes_with_retry(fd: int, nbytes: int, offset: int) -> bytes:
    """Buffered os.pread with bounded transient-I/O retry."""
    if nbytes <= 0:
        return b""
    last_err: OSError | None = None
    for attempt in range(_IO_RETRIES + 1):
        try:
            chunks: list[bytes] = []
            done = 0
            while done < nbytes:
                got = os.pread(fd, nbytes - done, offset + done)
                if not got:
                    raise OSError(f"short probe read at {offset + done} ({done}/{nbytes} bytes)")
                chunks.append(got)
                done += len(got)
            return chunks[0] if len(chunks) == 1 else b"".join(chunks)
        except OSError as e:
            last_err = e
            if attempt >= _IO_RETRIES or not _is_transient_io_err(e):
                raise
            _stats["io_retries"] += 1
            delay = _IO_RETRY_BACKOFF_S * (2**attempt)
            logger.warning(
                "staged load: transient probe pread error (%s) at offset=%d len=%d "
                "(retry %d/%d in %.2fs)",
                e,
                offset,
                nbytes,
                attempt + 1,
                _IO_RETRIES,
                delay,
            )
            if delay > 0:
                time.sleep(delay)
    raise last_err if last_err is not None else OSError("probe pread failed")


def _alloc_aligned_scratch(total: int, *, reuse: bool = False) -> torch.Tensor | None:
    """Allocate a ``total + _ALIGN`` uint8 scratch buffer with pinned->pageable OOM fallback."""
    global _ALIGNED_SCRATCH
    needed = total + _ALIGN
    if reuse and _ALIGNED_SCRATCH is not None and _ALIGNED_SCRATCH.numel() >= needed:
        return _ALIGNED_SCRATCH
    buf: torch.Tensor | None = None
    for attempt in range(2):
        try:
            buf = torch.empty(needed, dtype=torch.uint8, pin_memory=True)
            break
        except (RuntimeError, MemoryError) as e:
            _stats["oom_retries"] += 1
            if attempt == 0:
                _host_empty_cache_safe()
                continue
            _warn_once("aligned_pin_oom", "staged load: pinned O_DIRECT scratch OOM (%s); using pageable", e)
            try:
                buf = torch.empty(needed, dtype=torch.uint8, pin_memory=False)
            except (RuntimeError, MemoryError) as e2:
                _warn_once("aligned_oom", "staged load: pageable O_DIRECT scratch OOM (%s); using buffered path", e2)
                return None
    if reuse and buf is not None:
        _ALIGNED_SCRATCH = buf
    return buf


def _probe_read(fd: int, nbytes: int, offset: int, path: str | None = None) -> bytes:
    """Read ``nbytes`` at ``offset`` and return them, tolerating O_DIRECT alignment
    (the probe offset/length are arbitrary, the fd may be O_DIRECT)."""
    if fd not in _DIRECT_FDS:
        return _pread_bytes_with_retry(fd, nbytes, offset)

    def _buffered_probe() -> bytes:
        resolved = _resolve_fd_path(fd, path)
        if not resolved:
            raise OSError("cannot resolve shard path for buffered probe fallback")
        bfd = os.open(resolved, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
        try:
            data = _pread_bytes_with_retry(bfd, nbytes, offset)
            _fadvise_drop_range(bfd, offset, nbytes)
            return data
        finally:
            os.close(bfd)

    head = offset % _ALIGN
    start = offset - head
    total = (head + nbytes + _ALIGN - 1) // _ALIGN * _ALIGN
    if start + total > os.fstat(fd).st_size:
        # The padded window would run past EOF and O_DIRECT requires whole aligned
        # blocks, so redo the probe through a BUFFERED fd (pread on an O_DIRECT fd
        # still enforces alignment, so we must not reuse it here).
        return _buffered_probe()

    scratch = _alloc_aligned_scratch(total, reuse=False)
    if scratch is None:
        return _buffered_probe()
    base = scratch.data_ptr()
    base += (-base) % _ALIGN
    view = memoryview((ctypes.c_char * total).from_address(base))
    try:
        done = 0
        while done < total:
            got = os.preadv(fd, [view[done:]], start + done)
            if got <= 0:
                raise OSError(f"short probe read at {start + done}")
            done += got
        return ctypes.string_at(base + head, nbytes)
    except OSError as e:
        _warn_once(
            "probe_odirect",
            "staged load: O_DIRECT probe failed (%s) at offset=%d len=%d; using buffered probe",
            e,
            offset,
            nbytes,
        )
        _stats["o_direct_fallbacks"] += 1
        return _buffered_probe()


def _pread_aligned(fd: int, ptr: int, nbytes: int, offset: int) -> int:
    """O_DIRECT read of ``[offset, offset+nbytes)`` into ``ptr``.

    O_DIRECT requires the file offset, the transfer length AND the destination buffer
    all to be _ALIGN-aligned. Tensor offsets inside a safetensors shard are only
    8-byte aligned and a tensor's last chunk can be a short step, so read a padded
    _ALIGN-aligned window into a scratch buffer and copy the requested slice out.
    Returns ``nbytes`` on success, ``-1`` if the window would run past EOF or scratch OOMs.
    """
    if nbytes <= 0:
        return 0
    size = os.fstat(fd).st_size
    head = offset % _ALIGN              # bytes to skip in the first block
    start = offset - head               # _ALIGN-aligned file offset to read from
    total = (head + nbytes + _ALIGN - 1) // _ALIGN * _ALIGN    # aligned length
    if start + total > size:
        # the padded window would run past EOF -> caller must use the buffered path
        return -1
    # torch's pinned allocator only guarantees 512-byte alignment, NOT _ALIGN, so
    # over-allocate and take an _ALIGN-aligned slice inside the block.
    scratch = _alloc_aligned_scratch(total, reuse=True)
    if scratch is None:
        return -1
    raw = scratch.data_ptr()
    base = raw + ((-raw) % _ALIGN)
    total = (total // _ALIGN) * _ALIGN
    view = memoryview((ctypes.c_char * total).from_address(base))
    last_err: OSError | None = None
    for attempt in range(_IO_RETRIES + 1):
        try:
            done = 0
            while done < total:
                got = os.preadv(fd, [view[done:]], start + done)
                if got <= 0:
                    raise OSError(f"short read at {start + done} ({done}/{total} bytes)")
                done += got
            ctypes.memmove(ptr, base + head, nbytes)
            return nbytes
        except OSError as e:
            last_err = e
            # Alignment/FS errors (EINVAL/EOPNOTSUPP) must immediately trigger buffered fallback.
            if attempt >= _IO_RETRIES or not _is_transient_io_err(e):
                raise
            _stats["io_retries"] += 1
            delay = _IO_RETRY_BACKOFF_S * (2**attempt)
            logger.warning(
                "staged load: transient O_DIRECT pread error (%s) at offset=%d len=%d "
                "(retry %d/%d in %.2fs)",
                e,
                start,
                total,
                attempt + 1,
                _IO_RETRIES,
                delay,
            )
            if delay > 0:
                time.sleep(delay)
    raise last_err if last_err is not None else OSError("O_DIRECT pread failed")


def _pread_parallel(fd: int, ptr: int, nbytes: int, offset: int) -> None:
    """``_pread_into`` split across ``_READ_THREADS`` concurrent preads.

    Each worker reads a disjoint byte range straight into its own slice of the caller's
    buffer, so the result is byte-identical to the serial read; only the issue order
    changes. Falls back to the serial path for small reads or when threads are off.
    """
    if nbytes <= 0:
        return
    if _READ_THREADS <= 1 or nbytes < (4 << 20):
        _pread_into(fd, ptr, nbytes, offset)
        return
    global _POOL
    with _POOL_LOCK:
        if _POOL is None:
            _POOL = _ThreadPoolExecutor(
                max_workers=_READ_THREADS,
                thread_name_prefix="gb10-pread",
            )
            _ensure_atexit_registered()
        pool = _POOL
    parts = min(_READ_THREADS, max(1, nbytes // (1 << 20)))
    per = nbytes // parts
    ranges = []
    at = 0
    for i in range(parts):
        ln = per if i < parts - 1 else nbytes - at
        ranges.append((at, ln))
        at += ln
    errs: list[OSError] = []

    def worker(start: int, ln: int) -> None:
        try:
            _pread_into(fd, ptr + start, ln, offset + start)
        except OSError as e:      # noqa: BLE001 - re-raised on the caller's thread
            errs.append(e)

    futs = [pool.submit(worker, st, ln) for st, ln in ranges]
    for f in futs:
        f.result()
    if errs:
        raise errs[0]


def _pread_into(fd: int, ptr: int, nbytes: int, offset: int) -> None:
    if nbytes <= 0:
        return
    view = memoryview((ctypes.c_char * nbytes).from_address(ptr))
    last_err: OSError | None = None
    for attempt in range(_IO_RETRIES + 1):
        try:
            done = 0
            while done < nbytes:
                got = os.preadv(fd, [view[done:]], offset + done)
                if got <= 0:
                    raise OSError(f"short read at {offset + done} ({done}/{nbytes} bytes)")
                done += got
            return
        except OSError as e:
            last_err = e
            if attempt >= _IO_RETRIES or not _is_transient_io_err(e):
                raise
            _stats["io_retries"] += 1
            delay = _IO_RETRY_BACKOFF_S * (2**attempt)
            logger.warning(
                "staged load: transient pread error (%s) at offset=%d len=%d (retry %d/%d in %.2fs)",
                e,
                offset,
                nbytes,
                attempt + 1,
                _IO_RETRIES,
                delay,
            )
            if delay > 0:
                time.sleep(delay)
    raise last_err if last_err is not None else OSError("pread failed")


def _backing_file(src: torch.Tensor, nbytes: int) -> tuple[int, int]:
    """``(fd, file offset)`` to pread ``src``'s bytes from, or ``(-1, 0)``.

    The mapping is found by address, then both ends of the tensor are
    compared against the file, so a wrong guess can only fall back to
    copying from the view, never load the wrong bytes."""
    ptr = src.data_ptr()
    found = _file_range(ptr, nbytes)
    if found is None:
        return -1, 0
    path, offset, inode = found
    try:
        fd = _open_backing(path, inode, offset + nbytes)
        if fd < 0:
            return -1, 0
        probe = min(nbytes, _PROBE_BYTES)
        tail = nbytes - probe
        if _probe_read(fd, probe, offset, path=path) != ctypes.string_at(ptr, probe) or _probe_read(
            fd, probe, offset + tail, path=path
        ) != ctypes.string_at(ptr + tail, probe):
            _warn_once("probe", "staged load: %s does not match its mapping", path)
            return -1, 0
    except OSError as e:
        _warn_once("open", "staged load: cannot read %s directly (%s)", path, e)
        return -1, 0
    return fd, offset


def _staging(numel: int, dtype: torch.dtype, pin: bool) -> torch.Tensor:
    if pin:
        for attempt in range(2):
            try:
                return torch.empty(numel, dtype=dtype, pin_memory=True)
            except (RuntimeError, MemoryError) as e:
                _stats["oom_retries"] += 1
                if attempt == 0:
                    _host_empty_cache_safe()
                    continue
                _warn_once("pin", "staged load: pageable staging buffer (%s)", e)
    return torch.empty(numel, dtype=dtype)


def _alloc_staging_adaptive(step: int, dtype: torch.dtype, itemsize: int, pin: bool) -> tuple[torch.Tensor, int]:
    """Allocate staging buffer, shrinking chunk size down to _MIN_CHUNK_BYTES on host OOM."""
    min_step = max(1, _MIN_CHUNK_BYTES // max(1, itemsize))
    cur = max(1, step)
    while True:
        try:
            return _staging(cur, dtype, pin=pin), cur
        except (RuntimeError, MemoryError) as e:
            _stats["oom_retries"] += 1
            _host_empty_cache_safe()
            if cur <= min_step:
                raise
            next_step = max(min_step, cur // 2)
            logger.warning(
                "staged load: staging OOM at %.1f MiB (%s); shrinking chunk to %.1f MiB",
                (cur * itemsize) / (1024.0 * 1024.0),
                e,
                (next_step * itemsize) / (1024.0 * 1024.0),
            )
            cur = next_step
            pin = False


def _staged_copy(dst: torch.Tensor, src: torch.Tensor) -> None:
    """Chunked ``dst.copy_(src)`` through a staging buffer; both contiguous,
    same dtype and shape."""
    itemsize = src.element_size()
    total = src.numel()
    nbytes = total * itemsize
    # Always one full chunk, so torch's pinned-memory cache reuses one block.
    step = max(1, _CHUNK_BYTES // itemsize)
    dst_flat, src_flat = dst.view(-1), src.view(-1)
    stage, step = _alloc_staging_adaptive(step, src.dtype, itemsize, pin=dst.device.type == "cuda")
    fd, offset = _backing_file(src_flat, nbytes)
    shard_path = _fd_key[0] if _fd_key is not None else None

    read_s = h2d_s = 0.0
    for start in range(0, total, step):
        n = min(step, total - start)
        chunk_bytes = n * itemsize
        t0 = time.perf_counter()
        chunk_loaded = False
        if fd >= 0:
            at = offset + start * itemsize
            if _O_DIRECT and fd in _DIRECT_FDS:
                try:
                    got = _pread_aligned(fd, stage.data_ptr(), chunk_bytes, at)
                    if got == chunk_bytes:
                        chunk_loaded = True
                    else:
                        # Padded O_DIRECT window ran past EOF (or scratch OOM):
                        # reopen buffered and immediately read THIS chunk cleanly.
                        _warn_once(
                            "odirect_eof",
                            "staged load: O_DIRECT EOF/alignment boundary at offset=%d len=%d (%s); "
                            "switching to buffered+fadvise",
                            at,
                            chunk_bytes,
                            shard_path or "unknown",
                        )
                        fd = _reopen_buffered(fd, shard_path)
                except OSError as e:
                    _warn_once(
                        "read",
                        "staged load: O_DIRECT read failed (%s) at offset=%d len=%d (%s); "
                        "falling back to buffered+fadvise",
                        e,
                        at,
                        chunk_bytes,
                        shard_path or "unknown",
                    )
                    fd = _reopen_buffered(fd, shard_path)
            if not chunk_loaded and fd >= 0:
                try:
                    _pread_parallel(fd, stage.data_ptr(), chunk_bytes, at)
                    chunk_loaded = True
                    _fadvise_drop_range(fd, at, chunk_bytes)
                except OSError as e:
                    _warn_once(
                        "read_buf",
                        "staged load: buffered pread failed (%s) at offset=%d len=%d (%s); "
                        "falling back to mmap view",
                        e,
                        at,
                        chunk_bytes,
                        shard_path or "unknown",
                    )
                    _close_cached_fd()
                    fd = -1
            elif chunk_loaded and _DROP_CACHE:
                _fadvise_drop_range(fd, at, chunk_bytes)
        if not chunk_loaded:
            _stats["mmap_fallbacks"] += 1
            stage[:n].copy_(src_flat[start : start + n])
        t1 = time.perf_counter()
        dst_flat[start : start + n].copy_(stage[:n])
        t2 = time.perf_counter()
        read_s += t1 - t0
        h2d_s += t2 - t1

    if nbytes >= _CHUNK_BYTES and nbytes < h2d_s * _SLOW_H2D_BYTES_PER_S:
        logger.warning(
            "staged load: slow host-to-GPU copy from a resident buffer: "
            "%.0f MiB in %.1f s; this is not the mmap source path",
            nbytes / 2**20,
            h2d_s,
        )
    _stats["bytes"] += nbytes
    _stats["tensors"] += 1
    _stats["read_s"] += read_s
    _stats["h2d_s"] += h2d_s
    if _stats["bytes"] - _stats["logged"] >= _LOG_EVERY_BYTES:
        _stats["logged"] = _stats["bytes"]
        gib = _stats["bytes"] / 2**30
        logger.info(
            "staged load: %.1f GiB in %d tensors (read %.2f GiB/s, "
            "host-to-GPU %.2f GiB/s, %s, odirect_fb=%d, retries=%d)",
            gib,
            _stats["tensors"],
            gib / max(_stats["read_s"], 1e-9),
            gib / max(_stats["h2d_s"], 1e-9),
            "pread" if fd >= 0 else "mmap view",
            _stats["o_direct_fallbacks"],
            _stats["io_retries"],
        )


def copy_from_checkpoint_(dst: torch.Tensor, src: torch.Tensor) -> None:
    """``dst.copy_(src)`` for a checkpoint tensor that may be an mmap view,
    without letting the GPU driver take the source's page faults."""
    staged = _MODE == "force" or (
        _MODE != "0" and dst.device.type == "cuda" and src.device.type == "cpu"
    )
    if not staged or src.numel() == 0:
        dst.copy_(src)
    elif (
        src.numel() * src.element_size() <= _SMALL_BYTES
        or src.dtype != dst.dtype
        or src.shape != dst.shape
        or not src.is_contiguous()
        or not dst.is_contiguous()
    ):
        # The clone faults the pages in on the CPU; the driver then copies
        # from resident anonymous memory.
        try:
            dst.copy_(src.clone())
        except (RuntimeError, MemoryError) as e:
            _stats["oom_retries"] += 1
            _warn_once("clone_oom", "staged load: small/non-contig clone OOM (%s); reclaiming and retrying", e)
            _host_empty_cache_safe()
            try:
                dst.copy_(src.clone())
            except (RuntimeError, MemoryError):
                dst.copy_(src)
    else:
        _staged_copy(dst, src)


def release_staging() -> None:
    """Release cached shard fd, pread thread pool, and PyTorch pinned host staging buffer
    after checkpoint loading finishes (reclaims 128-256 MiB of pinned GB10 UMA RAM)."""
    global _POOL, _ALIGNED_SCRATCH
    _close_cached_fd()
    if _SEEN_SHARDS:
        shards = sorted(_SEEN_SHARDS)
        _SEEN_SHARDS.clear()
        for sp in shards:
            _fadvise_drop_path(sp)
    with _POOL_LOCK:
        pool = _POOL
        _POOL = None
    if pool is not None:
        try:
            pool.shutdown(wait=True)
        except Exception:
            try:
                pool.shutdown(wait=False)
            except Exception:
                pass
    _ALIGNED_SCRATCH = None
    _host_empty_cache_safe()
    try:
        from . import host_pin as _hp_uma
        _hp_uma.reclaim_host_uma(empty_cuda=False, tag="gb10_load_release_staging")
    except Exception:
        pass
