"""Helpers for running per-item work concurrently.

Nearly all of the time napari-phasors spends on a large image is inside
phasorpy's Cython kernels (``phasor_from_signal``, ``phasor_filter_median``,
``phasor_component_fit``) or inside NumPy. Both release the GIL while they
work, so a plain :class:`~concurrent.futures.ThreadPoolExecutor` gives close
to linear speedups on the array work. Threads are also the only sane choice
here: results stay in the same address space, so the multi-hundred-megabyte
``G``/``S`` arrays are never pickled the way a process pool would require.

Two rules shape every helper below.

**Qt and napari objects stay on the calling thread.** Layer creation, layer
mutation and any widget access must not happen in a worker. The helpers
therefore split work into a *compute* callable (pure array work, run in the
pool) and an *apply* callable (run in order on the calling thread), which is
what :func:`parallel_compute_apply` exists for.

**Pools never nest.** A reader that fans out over files calls helpers that
themselves fan out over tiles. Letting both layers spawn ``N`` threads would
oversubscribe the machine badly, so :func:`parallel_map` marks the current
thread while a pool is active and any nested call runs sequentially instead.

Fanning out over *items* only helps when there are several of them, and the
case that hurts most is the opposite one: a single very large image. The
band helpers (:func:`parallel_bands` and the kernel wrappers built on it)
cover that by splitting one array into horizontal bands of rows and handing
each band to a worker. phasorpy's ``num_threads`` argument does not help here
-- measured on phasorpy 0.12 it has no effect at the shapes this plugin sees
-- but band-splitting scales close to linearly, because the same Cython
kernels release the GIL either way.

Splitting is only valid for a kernel whose output at a pixel depends on a
bounded neighbourhood. Point-wise kernels need no overlap at all; the median
filter reaches ``size // 2`` pixels per pass, so a band must be grown by
``repeat * (size // 2)`` rows on each side and trimmed back afterwards. With
that halo the result is bit-identical to the unsplit call, which the tests
assert directly rather than with a tolerance.
"""

import os
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

__all__ = [
    "default_workers",
    "parallel_map",
    "parallel_compute_apply",
    "worker_limit_from_env",
    "workers_for_memory",
    "available_memory",
    "band_bounds",
    "parallel_bands",
    "parallel_filter_median",
    "parallel_phasor_from_signal",
    "parallel_rowwise",
]

#: Never spawn more than this many threads, however many cores are reported.
#: Beyond roughly this point the array work is memory-bandwidth bound and the
#: extra threads only add contention and peak memory.
MAX_WORKERS = 16

#: Set to an integer to override the worker count everywhere. ``1`` disables
#: concurrency entirely, which is the escape hatch when debugging.
WORKERS_ENV_VAR = "NAPARI_PHASORS_WORKERS"

#: Arrays with fewer pixels than this are filtered and transformed in one
#: piece. Below roughly a megapixel the kernels finish in a few milliseconds
#: and thread hand-off plus the halo recomputation costs more than it saves.
MIN_PARALLEL_PIXELS = 1 << 20

#: How many rows of real work each halo row must earn before a band is worth
#: cutting. A band is grown by ``halo`` rows on both sides, so this bounds the
#: redundant work at roughly ``2 / BAND_HALO_RATIO`` of the total.
BAND_HALO_RATIO = 8

# Marks a thread that is already running inside one of our pools, so nested
# fan-outs degrade to sequential instead of multiplying thread counts.
_local = threading.local()


def worker_limit_from_env():
    """Return the worker override from the environment, or ``None``.

    Returns
    -------
    int or None
        A positive integer if :data:`WORKERS_ENV_VAR` holds one, else
        ``None``. Unparsable or non-positive values are ignored rather than
        raising, so a stray value can never break a read.
    """
    raw = os.environ.get(WORKERS_ENV_VAR)
    if not raw:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def default_workers(n_items=None, workers=None):
    """Return how many threads to use for *n_items* pieces of work.

    Parameters
    ----------
    n_items : int, optional
        Number of work items. The result is never larger than this, since
        idle threads only cost memory.
    workers : int, optional
        Explicit request. Still clamped to :data:`MAX_WORKERS` and to
        *n_items*. The environment override wins over this.

    Returns
    -------
    int
        At least ``1``.
    """
    override = worker_limit_from_env()
    if override is not None:
        workers = override
    if workers is None:
        workers = os.cpu_count() or 1
    workers = max(1, min(int(workers), MAX_WORKERS))
    if n_items is not None:
        workers = max(1, min(workers, int(n_items)))
    return workers


def in_worker_thread():
    """Return whether the caller is already inside one of our pools."""
    return getattr(_local, "in_pool", False)


def parallel_map(
    func,
    items,
    workers=None,
    progress=None,
    on_error="raise",
):
    """Apply *func* to every item, concurrently, preserving input order.

    Parameters
    ----------
    func : callable
        Called with one item. Must not touch Qt or napari objects.
    items : sequence
        Work items. Materialized into a list so the length is known up front.
    workers : int, optional
        Thread count. Resolved through :func:`default_workers`.
    progress : callable, optional
        Called with the index of each item as it completes. Invoked from the
        calling thread as results are collected, never from a worker, so it
        is safe to drive a Qt progress bar with it.
    on_error : {'raise', 'collect'}, optional
        ``'raise'`` re-raises the first exception once every item has been
        given a chance to finish. ``'collect'`` returns the exception object
        in that item's slot instead, leaving the caller to sort out partial
        results.

    Returns
    -------
    list
        One result per input item, in input order.

    Raises
    ------
    Exception
        Whatever *func* raised, when ``on_error='raise'``.
    """
    items = list(items)
    if not items:
        return []

    n_workers = default_workers(len(items), workers)

    # A single worker, a single item, or an already-parallel caller all run
    # inline: no pool, no thread hand-off, and exceptions keep their original
    # traceback.
    if n_workers == 1 or len(items) == 1 or in_worker_thread():
        results = []
        for index, item in enumerate(items):
            try:
                results.append(func(item))
            except Exception as exc:  # noqa: BLE001
                if on_error == "raise":
                    raise
                results.append(exc)
            if progress is not None:
                progress(index)
        return results

    def run(item):
        _local.in_pool = True
        try:
            return func(item)
        finally:
            _local.in_pool = False

    results = [None] * len(items)
    first_error = None
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(run, item) for item in items]
        for index, future in enumerate(futures):
            try:
                results[index] = future.result()
            except Exception as exc:  # noqa: BLE001
                results[index] = exc
                if first_error is None:
                    first_error = exc
            if progress is not None:
                progress(index)

    if first_error is not None and on_error == "raise":
        raise first_error
    return results


def parallel_compute_apply(
    items,
    compute,
    apply,
    workers=None,
    progress=None,
    on_error="raise",
):
    """Compute in a pool, then apply the results in order on this thread.

    This is the pattern every multi-layer operation in the plugin follows:
    the expensive part is pure array work and parallelizes, while writing the
    answer back into a napari layer must happen on the main thread, one layer
    at a time, in a predictable order.

    Parameters
    ----------
    items : sequence
        Work items, typically layers.
    compute : callable
        Called with one item in a worker thread. Pure array work only.
    apply : callable
        Called with ``(item, result)`` on the calling thread, in input order,
        after every computation has finished. Free to touch Qt and napari.
    workers : int, optional
        Thread count, resolved through :func:`default_workers`.
    progress : callable, optional
        Called with each index as its computation completes.
    on_error : {'raise', 'collect'}, optional
        With ``'collect'``, items whose computation raised are skipped by
        *apply* and their exceptions are returned.

    Returns
    -------
    list
        The value *apply* returned for each item, in input order. Items
        skipped because their computation failed hold the exception instead.
    """
    items = list(items)
    if not items:
        return []

    computed = parallel_map(
        compute,
        items,
        workers=workers,
        progress=progress,
        on_error=on_error,
    )

    applied = []
    for item, result in zip(items, computed, strict=True):
        if isinstance(result, BaseException):
            applied.append(result)
            continue
        applied.append(apply(item, result))
    return applied


def available_memory():
    """Return free RAM in bytes, or ``None`` when it cannot be determined.

    Uses :mod:`psutil` (pulled in by napari) and degrades to ``None`` rather
    than raising if it is missing, in which case callers simply skip their
    memory cap.
    """
    try:
        import psutil
    except ImportError:
        return None
    try:
        return int(psutil.virtual_memory().available)
    except Exception:  # noqa: BLE001
        return None


def workers_for_memory(item_bytes, n_items=None, workers=None, fraction=0.5):
    """Return a worker count whose peak memory stays within budget.

    Reading files concurrently trades memory for speed: *N* workers hold *N*
    files' decoded signals at once, where a sequential read holds one. For
    the large datasets this plugin targets that trade can exhaust RAM, so the
    pool is sized against what is actually free.

    Parameters
    ----------
    item_bytes : int
        Estimated peak bytes one work item holds. ``0`` or ``None`` means
        unknown, and no memory cap is applied.
    n_items : int, optional
        Number of items, forwarded to :func:`default_workers`.
    workers : int, optional
        Requested worker count, before the memory cap.
    fraction : float, optional
        Share of free memory the pool may occupy. The default leaves half
        the headroom for the caller's own accumulation, which for a mosaic
        is the stitched canvas being built alongside the tiles.

    Returns
    -------
    int
        At least ``1``: a single item is always attempted, even if the
        estimate says it will not fit, because refusing to read at all is
        worse than letting the OS decide.
    """
    limit = default_workers(n_items, workers)

    # An explicit override is a deliberate instruction; don't second-guess it.
    if worker_limit_from_env() is not None:
        return limit

    if not item_bytes:
        return limit

    free = available_memory()
    if not free:
        return limit

    affordable = int((free * fraction) // int(item_bytes))
    return max(1, min(limit, affordable))


def band_bounds(size, workers=None, halo=0, min_band=1, max_band=None):
    """Split ``range(size)`` into contiguous bands, one per worker.

    Parameters
    ----------
    size : int
        Number of rows to split.
    workers : int, optional
        Upper bound on the number of bands, resolved through
        :func:`default_workers`.
    halo : int, optional
        Rows each band will be grown by on both sides. Only used to decide
        how *many* bands are worth cutting -- the returned bounds describe
        the rows a band is responsible for, not the rows it will read.
    min_band : int, optional
        Smallest band worth creating, before the halo rule is applied.
    max_band : int, optional
        Largest band allowed, which can push the count *above* the worker
        count. Callers use it to bound the scratch memory a single band
        needs, since only ``workers`` bands are ever in flight at once. The
        halo rule wins if the two disagree.

    Returns
    -------
    list of tuple
        ``(start, stop)`` pairs covering ``range(size)`` exactly once, with
        no gaps or overlaps. A single ``(0, size)`` band means the caller
        should not bother splitting.
    """
    size = int(size)
    if size <= 0:
        return []

    # Every halo row is work done twice, so a band has to be long enough for
    # the duplicated rows to stay a small fraction of it. That sets a hard
    # ceiling on how finely the range may be cut.
    min_band = max(1, int(min_band), BAND_HALO_RATIO * int(halo))
    allowed = max(1, size // min_band)

    n_bands = default_workers(allowed, workers)
    if max_band:
        n_bands = max(n_bands, -(-size // max(1, int(max_band))))
    n_bands = max(1, min(n_bands, allowed, size))
    if n_bands <= 1:
        return [(0, size)]

    # Spread the remainder over the leading bands instead of piling it onto
    # the last one, so no worker is handed a band twice the size of another.
    base, extra = divmod(size, n_bands)
    bounds = []
    start = 0
    for index in range(n_bands):
        stop = start + base + (1 if index < extra else 0)
        bounds.append((start, stop))
        start = stop
    return bounds


def parallel_bands(
    size, func, workers=None, halo=0, min_band=1, max_band=None
):
    """Run ``func(start, stop)`` over contiguous bands of ``range(size)``.

    The bands partition ``range(size)``, so a *func* that writes into
    ``out[start:stop]`` of a preallocated array never races another band.

    Parameters
    ----------
    size : int
        Number of rows to cover.
    func : callable
        Called as ``func(start, stop)``, possibly from a worker thread.
    workers : int, optional
        Thread count, resolved through :func:`default_workers`.
    halo : int, optional
        Forwarded to :func:`band_bounds` to size the bands.
    min_band : int, optional
        Forwarded to :func:`band_bounds`.
    max_band : int, optional
        Forwarded to :func:`band_bounds`.

    Returns
    -------
    list
        One result per band, in row order. A single-band split calls *func*
        inline, so the common "too small to bother" case adds no overhead.
    """
    bounds = band_bounds(
        size,
        workers=workers,
        halo=halo,
        min_band=min_band,
        max_band=max_band,
    )
    if len(bounds) <= 1:
        return [func(start, stop) for start, stop in bounds]
    return parallel_map(lambda b: func(b[0], b[1]), bounds, workers=workers)


def _row_axis_size(array):
    """Return the length of *array*'s row axis, or ``0`` if it has none."""
    shape = np.shape(array)
    return shape[-2] if len(shape) >= 2 else 0


def _should_split(array, workers=None):
    """Return whether *array* is big enough to be worth banding."""
    if in_worker_thread():
        return False
    if default_workers(workers=workers) <= 1:
        return False
    shape = np.shape(array)
    if len(shape) < 2:
        return False
    return int(np.prod(shape)) >= MIN_PARALLEL_PIXELS


def parallel_filter_median(
    mean,
    real,
    imag,
    *,
    repeat=1,
    size=3,
    skip_axis=None,
    workers=None,
):
    """Band-parallel :func:`phasorpy.filter.phasor_filter_median`.

    Median filtering dominates the time the filter tab spends on a large
    image, and it is re-run on every parameter change, so it is the single
    hottest interactive path in the plugin. Each band is filtered with
    ``repeat * (size // 2)`` extra rows of context on both sides and then
    trimmed back, which makes the result *bit-identical* to filtering the
    whole array at once -- including where the NaNs land.

    Parameters
    ----------
    mean, real, imag : numpy.ndarray
        Phasor arrays as accepted by phasorpy. ``real`` and ``imag`` may
        carry a leading harmonic axis; the split is always along the row
        axis (``-2``), which both layouts share.
    repeat, size, skip_axis : optional
        Passed straight through to phasorpy.
    workers : int, optional
        Thread count, resolved through :func:`default_workers`.

    Returns
    -------
    tuple of numpy.ndarray
        ``(mean, real, imag)``, exactly as phasorpy would have returned them.
    """
    from phasorpy.filter import phasor_filter_median

    def run(m, r, i):
        return phasor_filter_median(
            m, r, i, repeat=repeat, size=size, skip_axis=skip_axis
        )

    mean = np.asarray(mean)
    real = np.asarray(real)
    imag = np.asarray(imag)

    rows = _row_axis_size(mean)
    halo = int(repeat) * (int(size) // 2)
    if halo <= 0 or not _should_split(real, workers):
        return run(mean, real, imag)

    bounds = band_bounds(rows, workers=workers, halo=halo)
    if len(bounds) <= 1:
        return run(mean, real, imag)

    # phasorpy picks the output dtype from the inputs. Probing it on a
    # throwaway array the size of one filter footprint is far cheaper than
    # concatenating the band results afterwards, and lets every worker write
    # straight into its slice of the final arrays.
    probe = np.zeros((2 * size + 1, 2 * size + 1), dtype=mean.dtype)
    probe_out = phasor_filter_median(probe, probe, probe, repeat=1, size=size)
    out_dtype = np.asarray(probe_out[0]).dtype

    out_mean = np.empty(mean.shape, dtype=out_dtype)
    out_real = np.empty(real.shape, dtype=out_dtype)
    out_imag = np.empty(imag.shape, dtype=out_dtype)

    def filter_band(start, stop):
        low = max(0, start - halo)
        high = min(rows, stop + halo)
        band = run(
            mean[..., low:high, :],
            real[..., low:high, :],
            imag[..., low:high, :],
        )
        keep = slice(start - low, stop - low)
        out_mean[..., start:stop, :] = np.asarray(band[0])[..., keep, :]
        out_real[..., start:stop, :] = np.asarray(band[1])[..., keep, :]
        out_imag[..., start:stop, :] = np.asarray(band[2])[..., keep, :]

    parallel_bands(rows, filter_band, workers=workers, halo=halo)
    return out_mean, out_real, out_imag


def parallel_phasor_from_signal(
    signal, *, axis=None, harmonic=None, workers=None, **kwargs
):
    """Band-parallel :func:`phasorpy.phasor.phasor_from_signal`.

    The transform is point-wise across space -- every pixel's phasor depends
    only on its own histogram -- so bands need no halo and the result is
    identical to the unsplit call.

    Parameters
    ----------
    signal : array-like
        Signal with a histogram axis and one or more spatial axes.
    axis : int or str, optional
        Histogram axis, as phasorpy understands it. Splitting is skipped
        unless it resolves to an integer, since a named axis needs metadata
        the bands would not carry.
    harmonic : optional
        Passed through to phasorpy.
    workers : int, optional
        Thread count, resolved through :func:`default_workers`.
    **kwargs
        Further phasorpy keyword arguments, passed through unchanged.

    Returns
    -------
    tuple of numpy.ndarray
        ``(mean, real, imag)``.
    """
    from phasorpy.phasor import phasor_from_signal

    def run(data):
        return phasor_from_signal(data, axis=axis, harmonic=harmonic, **kwargs)

    # A DataArray or a named axis carries coordinate metadata that slicing
    # here would have to reproduce; leave those to phasorpy untouched.
    if not isinstance(signal, np.ndarray) or not isinstance(axis, int):
        return run(signal)

    if signal.ndim < 3 or in_worker_thread():
        return run(signal)
    if default_workers(workers=workers) <= 1:
        return run(signal)

    # The work scales with the whole signal, not just the pixels that come
    # out of it: every one of the K histogram samples is touched per pixel.
    if signal.size < MIN_PARALLEL_PIXELS:
        return run(signal)

    hist_axis = axis % signal.ndim
    spatial = [i for i in range(signal.ndim) if i != hist_axis]
    # Split the longest spatial axis: it gives the most even bands and keeps
    # each worker's slice contiguous for the common trailing-axis layouts.
    split_axis = max(spatial, key=lambda i: signal.shape[i])
    rows = signal.shape[split_axis]

    bounds = band_bounds(rows, workers=workers)
    if len(bounds) <= 1:
        return run(signal)

    def transform(start, stop):
        index = [slice(None)] * signal.ndim
        index[split_axis] = slice(start, stop)
        return run(signal[tuple(index)])

    results = parallel_bands(rows, transform, workers=workers)

    # ``mean`` loses the histogram axis; ``real``/``imag`` may gain a leading
    # harmonic axis on top of that, so their join axis sits one further right.
    mean_axis = split_axis - (1 if split_axis > hist_axis else 0)
    extra = np.asarray(results[0][1]).ndim - np.asarray(results[0][0]).ndim
    return (
        np.concatenate([r[0] for r in results], axis=mean_axis),
        np.concatenate([r[1] for r in results], axis=mean_axis + extra),
        np.concatenate([r[2] for r in results], axis=mean_axis + extra),
    )


def parallel_rowwise(func, *arrays, workers=None):
    """Apply a point-wise array kernel band by band over the row axis.

    For kernels whose output pixel depends only on the matching input pixel
    -- component fitting, lifetime conversion -- banding needs no halo and
    costs nothing but the split.

    Parameters
    ----------
    func : callable
        Called with one band of every array in *arrays* and returning either
        a single array or a tuple of them.
    *arrays : numpy.ndarray
        Arrays sharing a row axis, sliced along ``-2`` in step.
    workers : int, optional
        Thread count, resolved through :func:`default_workers`.

    Returns
    -------
    numpy.ndarray or tuple of numpy.ndarray
        Whatever *func* returns, reassembled over the full row axis.
    """
    arrays = [np.asarray(a) for a in arrays]
    if not arrays:
        return func()

    rows = _row_axis_size(arrays[0])
    if not rows or not _should_split(arrays[0], workers):
        return func(*arrays)

    bounds = band_bounds(rows, workers=workers)
    if len(bounds) <= 1:
        return func(*arrays)

    def apply_band(start, stop):
        return func(*(a[..., start:stop, :] for a in arrays))

    results = parallel_bands(rows, apply_band, workers=workers)

    if isinstance(results[0], tuple):
        return tuple(
            np.concatenate(
                [np.asarray(r[i]) for r in results],
                axis=-2,
            )
            for i in range(len(results[0]))
        )
    return np.concatenate([np.asarray(r) for r in results], axis=-2)
