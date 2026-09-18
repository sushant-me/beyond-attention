"""Measuring peak memory on Linux, correctly.

`resource.getrusage(RUSAGE_SELF).ru_maxrss` is a process high-water mark, and it
is reported out of `signal_struct`, which a forked child **inherits**. A child
spawned by a large parent therefore starts with the parent's peak already on the
clock: run it from a pytest session that has just trained a few models and its
own hundred megabytes of allocation are invisible, and the measured growth reads
as exactly zero. That is not a small error — it silently turns "this path
allocates 134 MB" into "this path allocates nothing", which is the opposite of
the claim being tested.

Linux can reset the mark: writing `5` to `/proc/self/clear_refs` clears
`hiwater_rss`. `reset_peak_rss()` does that and reports whether it took, so a
caller can distinguish a genuine zero from a measurement that was not capable of
seeing anything — the failure mode this whole repository is about.

The earlier version of the scaling experiment did not do this, so its numbers
were growth over a baseline that could already have been inflated by whatever the
driver process had done. Fixing it changed the reported figures, and the README
says so.
"""

from __future__ import annotations

import resource
import threading
from pathlib import Path

_CLEAR_REFS = Path("/proc/self/clear_refs")
# The kernel's `clear_refs` accepts 5 == CLEAR_REFS_MM_HIWATER_RSS.
_RESET_HIGHWATER = "5\n"


def peak_rss_kb() -> int:
    """The process's peak resident set size, in kilobytes."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def reset_peak_rss() -> bool:
    """Zero the peak-RSS high-water mark. Returns whether it worked.

    Best effort: the file may not exist (not Linux) or not be writable, and the
    kernel may be too old to understand the value. Callers should keep the return
    value so that an unsupported platform is reported rather than silently
    producing a measurement that cannot see small allocations.
    """
    try:
        _CLEAR_REFS.write_text(_RESET_HIGHWATER)
    except OSError:
        return False
    return True


def current_rss_kb() -> int:
    """The process's *current* resident set size, in kilobytes.

    Unlike `ru_maxrss` this is not a high-water mark, so it is not inherited by
    a forked child and it goes down when memory is freed.
    """
    try:
        # statm fields: size resident shared text lib data dt (in pages).
        fields = Path("/proc/self/statm").read_text().split()
        return int(fields[1]) * (resource.getpagesize() // 1024)
    except (OSError, IndexError, ValueError):
        return 0


class _RssSampler(threading.Thread):
    """Samples current RSS on a background thread while work runs."""

    daemon = True

    def __init__(self, interval_s: float = 0.001) -> None:
        super().__init__()
        self._interval = interval_s
        # Note the names: `threading.Thread` owns `_stop` and `_started`, and
        # shadowing either breaks the thread machinery in a way that surfaces as
        # a TypeError from deep inside `join`.
        self._finished = threading.Event()
        self.samples: list[int] = []

    def run(self) -> None:
        while not self._finished.is_set():
            value = current_rss_kb()
            if value:
                self.samples.append(value)
            self._finished.wait(self._interval)

    def halt(self) -> None:
        self._finished.set()


def peak_rss_during(fn, *args, **kwargs):
    """Run `fn` and return `(result, peak_rise_kb)`.

    The rise is measured by sampling current RSS during the call, not by
    subtracting high-water marks. That matters because `ru_maxrss` comes out of
    `signal_struct`, which a forked child inherits: a child spawned from a large
    parent starts with the parent's peak already recorded, and every allocation
    smaller than it is invisible. Writing 5 to `/proc/self/clear_refs` does not
    fix that — it lowers `mm->hiwater_rss`, while `getrusage` reports the larger
    of that and the inherited `signal->maxrss`, so the inherited value wins and
    the reset silently achieves nothing. Sampling current RSS has neither
    problem.
    """
    before = current_rss_kb()
    sampler = _RssSampler()
    sampler.start()
    try:
        result = fn(*args, **kwargs)
    finally:
        sampler.halt()
        sampler.join(timeout=1.0)
    peak = max(sampler.samples) if sampler.samples else before
    return result, max(0, peak - before)
