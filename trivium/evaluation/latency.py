"""LatencyProbe and LatencyStats.

Fixes the legacy issues:
- Probes rotate over the query list (not queries[0] only) so the
  cache is not artificially primed on a single string.
- Pre-cast query vectors are accepted; no per-call astype.
- cold_cache flag drops the OS page cache before measurement (no-op
  when not supported, so this is safe on any platform).
"""

from __future__ import annotations

import platform
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np


def drop_os_page_cache() -> None:
    """Best-effort: drop the OS page cache so the next probe is cold.

    Linux: 'sync && echo 3 > /proc/sys/vm/drop_caches' (requires root).
    macOS: 'purge' (requires user to be admin).
    Anywhere else: no-op.
    """
    try:
        if platform.system() == "Linux":
            subprocess.run(["sync"], check=False, capture_output=True)
            with open("/proc/sys/vm/drop_caches", "w") as f:
                f.write("3\n")
        elif platform.system() == "Darwin":
            subprocess.run(["purge"], check=False, capture_output=True)
    except (PermissionError, OSError, FileNotFoundError):
        pass


@dataclass(frozen=True)
class LatencyStats:
    """Percentile + mean summary over a sample of timed runs."""

    p50_ms: float
    p95_ms: float
    p99_ms: float
    p999_ms: float
    mean_ms: float
    n: int

    @classmethod
    def from_samples(cls, samples_ns: np.ndarray) -> LatencyStats:
        if samples_ns.size == 0:
            return cls(0.0, 0.0, 0.0, 0.0, 0.0, 0)
        return cls(
            p50_ms=float(np.percentile(samples_ns, 50) / 1e6),
            p95_ms=float(np.percentile(samples_ns, 95) / 1e6),
            p99_ms=float(np.percentile(samples_ns, 99) / 1e6),
            p999_ms=float(np.percentile(samples_ns, 99.9) / 1e6),
            mean_ms=float(samples_ns.mean() / 1e6),
            n=int(samples_ns.size),
        )


class LatencyProbe:
    """Run a callable and report its latency distribution.

    Args:
        fn: Zero-argument callable to time.
        n: Number of measured iterations after warmup.
        warmup: Number of warmup iterations to prime caches/JIT.
        rotate: Iterable of inputs to pass to fn as fn(x) on each
            measured iteration. Use this to avoid priming caches
            on a single hot query (legacy bug). If None, fn() is
            called with no args.
        cold_cache: If True, drop the OS page cache before the
            timed region. Default False (warm-cache mode).

    Example:
        probe = LatencyProbe(
            lambda v: idx.search(v, k=10),
            n=300,
            warmup=20,
            rotate=query_vectors,
        )
        stats = probe.run()
    """

    def __init__(
        self,
        fn: Callable,
        n: int = 300,
        warmup: int = 20,
        rotate: Sequence | None = None,
        cold_cache: bool = False,
    ) -> None:
        self.fn = fn
        self.n = n
        self.warmup = warmup
        self.rotate = list(rotate) if rotate is not None else None
        self.cold_cache = cold_cache

    def run(self) -> LatencyStats:
        if self.warmup > 0:
            self.prime_cache()
        if self.cold_cache:
            drop_os_page_cache()
        samples = np.empty(self.n, dtype=np.int64)
        for i in range(self.n):
            arg = self.rotate[i % len(self.rotate)] if self.rotate else None
            t0 = time.perf_counter_ns()
            if arg is None:
                self.fn()
            else:
                self.fn(arg)
            samples[i] = time.perf_counter_ns() - t0
        return LatencyStats.from_samples(samples)

    def prime_cache(self) -> None:
        """Run warmup iterations to prime caches/JIT before measurement."""
        if self.rotate:
            for i in range(self.warmup):
                self.fn(self.rotate[i % len(self.rotate)])
        else:
            for _ in range(self.warmup):
                self.fn()


def probe_lambda(fn: Callable, n: int, warmup: int, *, cold_cache: bool = False) -> LatencyStats:
    """Convenience: probe a zero-argument callable without rotation."""
    return LatencyProbe(fn=fn, n=n, warmup=warmup, cold_cache=cold_cache).run()
