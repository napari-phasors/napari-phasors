"""Tests for the shared thread-pool helpers."""

import threading

import pytest

from napari_phasors._parallel import (
    default_workers,
    parallel_compute_apply,
    parallel_map,
    workers_for_memory,
)


@pytest.fixture(autouse=True)
def clear_worker_env(monkeypatch):
    """Never inherit a worker override from the ambient environment.

    A ``NAPARI_PHASORS_WORKERS`` set in a shell or CI job would otherwise
    silently rewrite every expectation in this module.
    """
    monkeypatch.delenv("NAPARI_PHASORS_WORKERS", raising=False)


def test_default_workers_never_exceeds_item_count():
    """Idle threads cost memory, so the pool never outgrows the work."""
    assert default_workers(1) == 1
    assert default_workers(3, workers=16) == 3
    assert default_workers(None, workers=4) == 4


def test_default_workers_is_capped():
    """However many cores are reported, the pool stays bounded."""
    assert default_workers(1000, workers=10_000) <= 16
    assert default_workers(0) == 1


def test_env_override_forces_worker_count(monkeypatch):
    """The env var is the debugging escape hatch and wins over the request."""
    monkeypatch.setenv("NAPARI_PHASORS_WORKERS", "1")
    assert default_workers(100, workers=8) == 1

    monkeypatch.setenv("NAPARI_PHASORS_WORKERS", "3")
    assert default_workers(100, workers=8) == 3


@pytest.mark.parametrize("value", ["", "0", "-2", "nonsense"])
def test_bad_env_override_is_ignored(monkeypatch, value):
    """A stray env value must never break a read."""
    baseline = default_workers(4)
    monkeypatch.setenv("NAPARI_PHASORS_WORKERS", value)
    assert default_workers(4) == baseline
    # An explicit request still wins, so the value really was discarded.
    assert default_workers(4, workers=2) == 2


def test_parallel_map_preserves_order():
    """Results come back in input order regardless of completion order."""
    items = list(range(20))
    assert parallel_map(lambda x: x * x, items) == [x * x for x in items]


def test_parallel_map_empty():
    assert parallel_map(lambda x: x, []) == []


def test_parallel_map_actually_uses_threads():
    """More than one thread runs the work when there is work to spread."""
    seen = set()
    barrier = threading.Barrier(4, timeout=10)

    def record(_):
        seen.add(threading.current_thread().name)
        barrier.wait()
        return None

    parallel_map(record, range(4), workers=4)
    assert len(seen) == 4


def test_parallel_map_raises_first_error():
    def boom(x):
        if x == 2:
            raise ValueError("nope")
        return x

    with pytest.raises(ValueError, match="nope"):
        parallel_map(boom, range(5), workers=2)


def test_parallel_map_collects_errors():
    """``collect`` returns exceptions in place so partial results survive."""

    def boom(x):
        if x == 2:
            raise ValueError("nope")
        return x

    results = parallel_map(boom, range(4), workers=2, on_error="collect")
    assert results[0] == 0
    assert isinstance(results[2], ValueError)
    assert results[3] == 3


def test_parallel_map_reports_progress_from_calling_thread():
    """Progress must fire on the caller's thread so it can drive Qt."""
    caller = threading.current_thread()
    threads = []

    parallel_map(
        lambda x: x,
        range(6),
        workers=3,
        progress=lambda index: threads.append(threading.current_thread()),
    )
    assert threads and all(t is caller for t in threads)


def test_nested_pools_do_not_multiply_threads():
    """A fan-out inside a fan-out runs inline instead of spawning again."""
    inner_threads = []

    def outer(_):
        def inner(_inner):
            inner_threads.append(threading.current_thread())
            return None

        parallel_map(inner, range(4), workers=4)
        return threading.current_thread()

    outer_threads = parallel_map(outer, range(3), workers=3)
    # Each inner call ran on the worker that invoked it, not on new threads.
    assert set(inner_threads) <= set(outer_threads)


def test_parallel_compute_apply_applies_in_order_on_caller():
    caller = threading.current_thread()
    applied = []

    def apply(item, result):
        applied.append((item, result, threading.current_thread()))
        return result

    out = parallel_compute_apply(range(5), lambda x: x * 2, apply, workers=3)
    assert out == [0, 2, 4, 6, 8]
    assert [entry[0] for entry in applied] == list(range(5))
    assert all(entry[2] is caller for entry in applied)


def test_parallel_compute_apply_skips_failed_items():
    """A failed computation is reported and its apply step is skipped."""

    def compute(x):
        if x == 1:
            raise RuntimeError("bad")
        return x

    applied = []
    results = parallel_compute_apply(
        range(3),
        compute,
        lambda item, result: applied.append(item),
        workers=2,
        on_error="collect",
    )
    assert applied == [0, 2]
    assert isinstance(results[1], RuntimeError)


def test_workers_for_memory_caps_on_large_items():
    """A pool that would not fit in RAM is shrunk, never below one."""
    assert workers_for_memory(10**15, n_items=16) == 1
    # An unknown size leaves the cap off.
    assert workers_for_memory(0, n_items=4) == default_workers(4)


def test_workers_for_memory_respects_env_override(monkeypatch):
    """An explicit instruction is not second-guessed by the memory cap."""
    monkeypatch.setenv("NAPARI_PHASORS_WORKERS", "4")
    assert workers_for_memory(10**15, n_items=16) == 4
