"""Tests for the shared thread-pool helpers."""

import threading

import numpy as np
import pytest

from napari_phasors._parallel import (
    band_bounds,
    default_workers,
    parallel_bands,
    parallel_compute_apply,
    parallel_filter_median,
    parallel_map,
    parallel_phasor_from_signal,
    parallel_rowwise,
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


# --- band splitting ---------------------------------------------------------


def test_band_bounds_partitions_exactly():
    """Bands cover every row once, with no gap and no overlap."""
    for size in (1, 7, 100, 1001):
        bounds = band_bounds(size, workers=4)
        assert bounds[0][0] == 0
        assert bounds[-1][1] == size
        assert all(
            a[1] == b[0] for a, b in zip(bounds, bounds[1:], strict=False)
        )
        assert sum(stop - start for start, stop in bounds) == size


def test_band_bounds_spreads_the_remainder():
    """No band ends up more than one row longer than another."""
    bounds = band_bounds(101, workers=4)
    lengths = sorted(stop - start for start, stop in bounds)
    assert lengths[-1] - lengths[0] <= 1


def test_band_bounds_empty_and_single():
    """Nothing to split returns nothing; one worker returns one band."""
    assert band_bounds(0) == []
    assert band_bounds(-5) == []
    assert band_bounds(50, workers=1) == [(0, 50)]


def test_band_bounds_halo_prevents_uselessly_thin_bands():
    """A band has to be long enough for its halo to stay a small overhead."""
    # 10 rows with a 3-row halo would duplicate more work than it saves.
    assert band_bounds(10, workers=8, halo=3) == [(0, 10)]
    # With enough rows the split goes ahead.
    assert len(band_bounds(1000, workers=4, halo=3)) == 4


def test_band_bounds_max_band_can_exceed_the_worker_count():
    """Capping band size bounds per-band memory, so more bands are cut."""
    bounds = band_bounds(1000, workers=2, max_band=100)
    assert len(bounds) == 10
    assert all(stop - start <= 100 for start, stop in bounds)

    # The halo rule wins when the two disagree, since a band shorter than the
    # halo would be nearly all duplicated work.
    bounds = band_bounds(1000, workers=2, max_band=10, halo=50)
    assert all(stop - start >= 400 for start, stop in bounds)


def test_band_bounds_never_makes_more_bands_than_rows():
    """A tiny canvas cannot be cut into more bands than it has rows."""
    assert len(band_bounds(3, workers=16, max_band=1)) == 3


def test_parallel_bands_runs_every_band():
    """Each band is visited once and results come back in row order."""
    seen = parallel_bands(100, lambda a, b: (a, b), workers=4)
    assert seen == [(0, 25), (25, 50), (50, 75), (75, 100)]


def test_parallel_bands_single_band_runs_inline():
    """The 'not worth splitting' path adds no thread hand-off."""
    caller = threading.current_thread().name
    ran_on = parallel_bands(
        10, lambda a, b: threading.current_thread().name, workers=1
    )
    assert ran_on == [caller]


def test_parallel_filter_median_matches_the_unsplit_call():
    """Band splitting is exact, not approximate: identical bits, NaNs too."""
    from phasorpy.filter import phasor_filter_median

    rng = np.random.default_rng(0)
    shape = (1200, 1200)
    mean = (rng.random(shape) * 100).astype(np.float32)
    real = rng.random((2,) + shape).astype(np.float32)
    imag = rng.random((2,) + shape).astype(np.float32)
    # phasorpy treats NaN specially, so the mask has to survive the split.
    mean[rng.random(shape) < 0.05] = np.nan

    for repeat, size in ((1, 3), (2, 5), (3, 7)):
        expected = phasor_filter_median(
            mean, real, imag, repeat=repeat, size=size
        )
        got = parallel_filter_median(
            mean, real, imag, repeat=repeat, size=size
        )
        for want, have in zip(expected, got, strict=True):
            want = np.asarray(want)
            have = np.asarray(have)
            assert have.dtype == want.dtype
            assert np.array_equal(have, want, equal_nan=True)


def test_parallel_filter_median_handles_leading_axes():
    """A stack keeps its leading axes; only the row axis is split."""
    from phasorpy.filter import phasor_filter_median

    rng = np.random.default_rng(1)
    mean = (rng.random((3, 700, 700)) * 10).astype(np.float32)
    real = rng.random((2, 3, 700, 700)).astype(np.float32)
    imag = rng.random((2, 3, 700, 700)).astype(np.float32)
    skip_axis = (0,)

    expected = phasor_filter_median(
        mean, real, imag, repeat=1, size=3, skip_axis=skip_axis
    )
    got = parallel_filter_median(
        mean, real, imag, repeat=1, size=3, skip_axis=skip_axis
    )
    for want, have in zip(expected, got, strict=True):
        assert np.array_equal(
            np.asarray(have), np.asarray(want), equal_nan=True
        )


def test_parallel_filter_median_skips_small_arrays():
    """Below the pixel threshold the array goes through in one piece."""
    from phasorpy.filter import phasor_filter_median

    rng = np.random.default_rng(2)
    small = rng.random((32, 32)).astype(np.float32)
    expected = phasor_filter_median(small, small, small, repeat=1, size=3)
    got = parallel_filter_median(small, small, small, repeat=1, size=3)
    for want, have in zip(expected, got, strict=True):
        assert np.array_equal(
            np.asarray(have), np.asarray(want), equal_nan=True
        )


def test_parallel_filter_median_without_a_halo_is_passed_through():
    """``size=1`` reaches no neighbour, so there is nothing to split around."""
    rng = np.random.default_rng(3)
    data = rng.random((1200, 1200)).astype(np.float32)
    mean, real, imag = parallel_filter_median(
        data, data, data, repeat=1, size=1
    )
    assert np.asarray(mean).shape == data.shape


def test_parallel_filter_median_is_sequential_inside_a_pool():
    """A nested call must not multiply the thread count."""
    rng = np.random.default_rng(4)
    data = rng.random((1200, 1200)).astype(np.float32)

    names = set()

    def work(_):
        names.add(threading.current_thread().name)
        parallel_filter_median(data, data, data, repeat=1, size=3)
        return threading.current_thread().name

    parallel_map(work, [0, 1], workers=2)
    # Two outer workers and no inner ones: the band split stood down.
    assert len(names) == 2


def test_parallel_phasor_from_signal_matches_the_unsplit_call():
    """Splitting a signal over space gives the same phasor coordinates."""
    from phasorpy.phasor import phasor_from_signal

    rng = np.random.default_rng(5)
    signal = (rng.random((128, 256, 256)) * 50).astype(np.float32)

    for harmonic in (1, [1, 2]):
        expected = phasor_from_signal(signal, axis=0, harmonic=harmonic)
        got = parallel_phasor_from_signal(signal, axis=0, harmonic=harmonic)
        for want, have in zip(expected, got, strict=True):
            want = np.asarray(want)
            have = np.asarray(have)
            assert have.shape == want.shape
            assert np.allclose(have, want, equal_nan=True)


def test_parallel_phasor_from_signal_splits_the_longest_axis():
    """A histogram axis that is not first still leaves the space split."""
    from phasorpy.phasor import phasor_from_signal

    rng = np.random.default_rng(6)
    signal = (rng.random((64, 512, 128)) * 50).astype(np.float32)

    expected = phasor_from_signal(signal, axis=2, harmonic=[1, 2])
    got = parallel_phasor_from_signal(signal, axis=2, harmonic=[1, 2])
    for want, have in zip(expected, got, strict=True):
        assert np.asarray(have).shape == np.asarray(want).shape
        assert np.allclose(np.asarray(have), np.asarray(want), equal_nan=True)


def test_parallel_phasor_from_signal_passes_through_unsplittable_input():
    """Named axes and small or low-dimensional signals are left alone."""
    from phasorpy.phasor import phasor_from_signal

    rng = np.random.default_rng(7)

    # Too small to be worth splitting.
    small = (rng.random((8, 16, 16)) * 10).astype(np.float32)
    got = parallel_phasor_from_signal(small, axis=0, harmonic=1)
    want = phasor_from_signal(small, axis=0, harmonic=1)
    assert np.allclose(np.asarray(got[0]), np.asarray(want[0]), equal_nan=True)

    # 2-D: there is no spatial axis worth banding.
    flat = (rng.random((64, 4096)) * 10).astype(np.float32)
    got = parallel_phasor_from_signal(flat, axis=0, harmonic=1)
    want = phasor_from_signal(flat, axis=0, harmonic=1)
    assert np.allclose(np.asarray(got[0]), np.asarray(want[0]), equal_nan=True)

    # A named axis carries metadata a raw slice would not, so it is not split.
    xr = pytest.importorskip("xarray")
    cube = xr.DataArray(
        (rng.random((64, 256, 256)) * 10).astype(np.float32),
        dims=("H", "Y", "X"),
    )
    got = parallel_phasor_from_signal(cube, axis="H", harmonic=1)
    assert np.asarray(got[0]).shape == (256, 256)


def test_parallel_phasor_from_signal_is_sequential_inside_a_pool():
    """Nested splitting stands down here too."""
    rng = np.random.default_rng(8)
    signal = (rng.random((64, 256, 256)) * 10).astype(np.float32)

    names = set()

    def work(_):
        parallel_phasor_from_signal(signal, axis=0, harmonic=1)
        names.add(threading.current_thread().name)

    parallel_map(work, [0, 1], workers=2)
    assert len(names) == 2


def test_parallel_rowwise_matches_the_unsplit_call():
    """A point-wise kernel gives the same answer band by band."""
    rng = np.random.default_rng(9)
    a = rng.random((1200, 1200))
    b = rng.random((1200, 1200))

    def kernel(x, y):
        return x * 2 + y

    assert np.array_equal(parallel_rowwise(kernel, a, b), kernel(a, b))


def test_parallel_rowwise_reassembles_tuple_results():
    """Kernels returning several arrays are stitched back one by one."""
    rng = np.random.default_rng(10)
    a = rng.random((1200, 1200))

    def kernel(x):
        return x + 1, x - 1

    first, second = parallel_rowwise(kernel, a)
    assert np.array_equal(first, a + 1)
    assert np.array_equal(second, a - 1)


def test_parallel_rowwise_handles_leading_axes():
    """Arrays with a leading axis are sliced on the row axis, not the first."""
    rng = np.random.default_rng(11)
    a = rng.random((3, 800, 800))
    assert np.array_equal(parallel_rowwise(lambda x: x * 2, a), a * 2)


def test_parallel_rowwise_passes_through_when_not_worth_splitting():
    """No arrays, a tiny array, or a 1-D array all run in one piece."""
    assert parallel_rowwise(lambda: "nothing") == "nothing"

    small = np.arange(64.0).reshape(8, 8)
    assert np.array_equal(parallel_rowwise(lambda x: x + 1, small), small + 1)

    flat = np.arange(4096.0)
    assert np.array_equal(parallel_rowwise(lambda x: x + 1, flat), flat + 1)


def test_parallel_rowwise_respects_the_worker_override(monkeypatch):
    """One worker means one piece, however large the array."""
    monkeypatch.setenv("NAPARI_PHASORS_WORKERS", "1")
    big = np.zeros((1200, 1200))
    calls = []

    def kernel(x):
        calls.append(x.shape)
        return x

    parallel_rowwise(kernel, big)
    assert calls == [(1200, 1200)]


def test_parallel_compute_apply_with_no_items():
    """Nothing to do is not an error."""
    assert parallel_compute_apply([], lambda item: item, lambda i, r: r) == []


def test_available_memory_degrades_instead_of_raising(monkeypatch):
    """Memory probing is best effort; every failure means 'unknown'."""
    import sys

    from napari_phasors import _parallel

    # ``None`` in sys.modules makes ``import psutil`` raise ImportError, which
    # is what a napari install without psutil would do.
    monkeypatch.setitem(sys.modules, "psutil", None)
    assert _parallel.available_memory() is None
    monkeypatch.undo()

    import psutil

    def boom():
        raise RuntimeError("cannot read memory")

    monkeypatch.setattr(psutil, "virtual_memory", boom)
    assert _parallel.available_memory() is None


def test_workers_for_memory_without_a_reading_uses_the_plain_limit(
    monkeypatch,
):
    """An unknown amount of free memory means no memory cap at all."""
    from napari_phasors import _parallel

    monkeypatch.setattr(_parallel, "available_memory", lambda: None)
    assert workers_for_memory(1 << 30, n_items=8) == default_workers(8)

    monkeypatch.setattr(_parallel, "available_memory", lambda: 0)
    assert workers_for_memory(1 << 30, n_items=8) == default_workers(8)


def test_band_helpers_stand_down_for_one_worker(monkeypatch):
    """``NAPARI_PHASORS_WORKERS=1`` disables every band split."""
    from phasorpy.filter import phasor_filter_median
    from phasorpy.phasor import phasor_from_signal

    monkeypatch.setenv("NAPARI_PHASORS_WORKERS", "1")
    rng = np.random.default_rng(12)

    big = rng.random((1200, 1200)).astype(np.float32)
    expected = phasor_filter_median(big, big, big, repeat=1, size=3)
    got = parallel_filter_median(big, big, big, repeat=1, size=3)
    for want, have in zip(expected, got, strict=True):
        assert np.array_equal(
            np.asarray(have), np.asarray(want), equal_nan=True
        )

    signal = (rng.random((64, 256, 256)) * 10).astype(np.float32)
    want = phasor_from_signal(signal, axis=0, harmonic=1)
    have = parallel_phasor_from_signal(signal, axis=0, harmonic=1)
    assert np.allclose(
        np.asarray(have[0]), np.asarray(want[0]), equal_nan=True
    )


def test_band_helpers_stand_down_for_a_single_band(monkeypatch):
    """A worker count of one band leaves the kernels untouched."""
    from napari_phasors import _parallel

    rng = np.random.default_rng(13)
    monkeypatch.setattr(
        _parallel, "band_bounds", lambda size, **kwargs: [(0, size)]
    )

    big = rng.random((1200, 1200)).astype(np.float32)
    mean, _, _ = parallel_filter_median(big, big, big, repeat=1, size=3)
    assert np.asarray(mean).shape == big.shape

    signal = (rng.random((64, 256, 256)) * 10).astype(np.float32)
    assert np.asarray(
        parallel_phasor_from_signal(signal, axis=0, harmonic=1)[0]
    ).shape == (256, 256)

    assert np.array_equal(parallel_rowwise(lambda x: x + 1, big), big + 1)


def test_filter_median_does_not_split_a_one_dimensional_stack():
    """With no row axis to cut, the kernel is called on the whole array."""
    from phasorpy.filter import phasor_filter_median

    rng = np.random.default_rng(14)
    flat = rng.random(1 << 21).astype(np.float32)

    expected = phasor_filter_median(flat, flat, flat, repeat=1, size=3)
    got = parallel_filter_median(flat, flat, flat, repeat=1, size=3)
    for want, have in zip(expected, got, strict=True):
        assert np.array_equal(
            np.asarray(have), np.asarray(want), equal_nan=True
        )
