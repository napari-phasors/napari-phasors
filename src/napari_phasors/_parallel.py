"""Helpers for running per-item work concurrently."""

import os
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np

__all__ = [
    "ITEMS",
    "BANDS",
    "scope_enabled",
    "default_workers",
    "parallel_enabled",
    "set_parallel_enabled",
    "parallel_items_enabled",
    "set_parallel_items_enabled",
    "parallel_bands_enabled",
    "set_parallel_bands_enabled",
    "memory_budget_enabled",
    "set_memory_budget_enabled",
    "memory_fraction",
    "set_memory_fraction",
    "parallel_map",
    "parallel_stream",
    "parallel_compute_apply",
    "worker_limit_from_env",
    "workers_for_memory",
    "items_for_memory",
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

# The two switches driven by the Performance section of Plot Settings.
#
# ``_parallel_items_enabled`` covers fan-out over *items* -- the files of a
# stack, the layers of a multi-layer operation, the images of a batch. Each
# worker holds a whole item, so this is the switch that multiplies peak
# memory, and the one to turn off on a machine that is short on RAM.
#
# ``_parallel_bands_enabled`` covers splitting *one* image into horizontal
# bands. Peak memory barely moves -- the array is already resident and only
# the halo rows are duplicated -- but it is the only thing that speeds up the
# single-large-image case.

_parallel_items_enabled = False
_parallel_bands_enabled = False
_memory_budget_enabled = False

#: The two scopes :func:`default_workers` and friends accept.
ITEMS = "items"
BANDS = "bands"

#: Default share of *free* memory that concurrent work may occupy.
DEFAULT_MEMORY_FRACTION = 0.5

_memory_fraction = DEFAULT_MEMORY_FRACTION


def memory_fraction():
    """Return the share of free memory concurrent work may occupy."""
    return _memory_fraction


def set_memory_fraction(fraction):
    """Set the share of free memory concurrent work may occupy.

    Parameters
    ----------
    fraction : float
        Between 0 and 1. Clamped into ``[0.05, 0.95]``: below the floor no
        pool could ever be sized above one worker even on an idle machine,
        and above the ceiling there is no headroom left for the result the
        workers are feeding.
    """
    global _memory_fraction
    _memory_fraction = min(0.95, max(0.05, float(fraction)))


def memory_budget_enabled():
    """Return whether the memory budget caps concurrent work."""
    return _memory_budget_enabled


def set_memory_budget_enabled(enabled):
    """Enable or disable memory budget capping.

    Parameters
    ----------
    enabled : bool
        ``True`` to cap concurrent work and workers based on available RAM,
        ``False`` to allow concurrency without memory budget constraints.
    """
    global _memory_budget_enabled
    _memory_budget_enabled = bool(enabled)


def parallel_items_enabled():
    """Return whether work may be fanned out over separate items.

    Items are whole files, layers or images: what :func:`parallel_map`,
    :func:`parallel_stream` and :func:`parallel_compute_apply` iterate over.
    """
    return _parallel_items_enabled


def set_parallel_items_enabled(enabled):
    """Enable or disable fan-out over separate layers, files and images.

    Read through ``default_workers(..., scope=ITEMS)``, which every item-level
    helper funnels through, so turning it off makes :func:`parallel_map` and
    :func:`parallel_stream` run inline. Band splitting inside a single image
    is a separate switch and is left alone. Work already in flight is
    unaffected; the next call picks up the change.

    Parameters
    ----------
    enabled : bool
        ``True`` to process several items at once, ``False`` to process them
        one after another on the calling thread.
    """
    global _parallel_items_enabled
    _parallel_items_enabled = bool(enabled)


def parallel_bands_enabled():
    """Return whether a single image may be split into bands across threads."""
    return _parallel_bands_enabled


def set_parallel_bands_enabled(enabled):
    """Enable or disable splitting one image across threads.

    Parameters
    ----------
    enabled : bool
        ``True`` to split a large array into bands, ``False`` to process it
        in one piece.
    """
    global _parallel_bands_enabled
    _parallel_bands_enabled = bool(enabled)


def parallel_enabled():
    """Return whether *either* scope may use a thread pool."""
    return _parallel_items_enabled or _parallel_bands_enabled


def set_parallel_enabled(enabled):
    """Set both parallelism switches at once.

    Parameters
    ----------
    enabled : bool
        ``True`` to fan out over threads in both scopes, ``False`` to run
        everything sequentially on the calling thread.
    """
    set_parallel_items_enabled(enabled)
    set_parallel_bands_enabled(enabled)


def scope_enabled(scope):
    """Return whether *scope* (:data:`ITEMS` or :data:`BANDS`) is switched on."""
    if scope == BANDS:
        return _parallel_bands_enabled
    return _parallel_items_enabled


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


def default_workers(n_items=None, workers=None, scope=ITEMS):
    """Return how many threads to use for *n_items* pieces of work.

    Parameters
    ----------
    n_items : int, optional
        Number of work items. The result is never larger than this, since
        idle threads only cost memory.
    workers : int, optional
        Explicit request. Still clamped to :data:`MAX_WORKERS` and to
        *n_items*. The environment override wins over this, and the scope's
        switch wins over both.
    scope : {'items', 'bands'}, optional
        Which switch to consult: :data:`ITEMS` for fan-out over separate
        layers, files and images, :data:`BANDS` for splitting one image.
        Defaults to :data:`ITEMS`.

    Returns
    -------
    int
        At least ``1``, and exactly ``1`` while *scope*'s parallelism is
        switched off.
    """
    if not scope_enabled(scope):
        # The UI toggle is the most explicit statement of intent there is, so
        # it wins over both the environment override and any explicit request.
        return 1
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


def parallel_stream(
    func,
    items,
    workers=None,
    max_in_flight=None,
    on_error="raise",
    scope=ITEMS,
):
    """Yield ``(index, item, result)`` in input order, a few items at a time.

    Parameters
    ----------
    func : callable
        Called with one item, in a worker thread. Must not touch Qt or
        napari objects.
    items : sequence
        Work items. Materialized into a list so the length is known up front.
    workers : int, optional
        Thread count, resolved through :func:`default_workers`.
    max_in_flight : int, optional
        How many items may be submitted but not yet consumed. Defaults to
        twice the worker count, which keeps every worker fed while the
        caller processes the previous result. Never below the worker count
        -- a smaller value would idle workers -- and never above the item
        count.
    on_error : {'raise', 'collect'}, optional
        ``'raise'`` propagates the first failure immediately, cancelling
        whatever has not started. ``'collect'`` yields the exception in that
        item's slot and carries on.
    scope : {'items', 'bands'}, optional
        Which parallelism switch governs this pool. Bands of one image pass
        :data:`BANDS`; everything else leaves the :data:`ITEMS` default.

    Yields
    ------
    tuple
        ``(index, item, result)`` for each item, in input order. With
        ``on_error='collect'``, *result* may be the exception that item
        raised.
    """
    items = list(items)
    if not items:
        return

    n_workers = default_workers(len(items), workers, scope=scope)

    # A single worker, a single item, or an already-parallel caller all run
    # inline: no pool, no thread hand-off, and nothing is ever in flight.
    if n_workers == 1 or len(items) == 1 or in_worker_thread():
        for index, item in enumerate(items):
            try:
                yield index, item, func(item)
            except Exception as exc:  # noqa: BLE001
                if on_error == "raise":
                    raise
                yield index, item, exc
        return

    limit = 2 * n_workers if max_in_flight is None else int(max_in_flight)
    limit = max(n_workers, min(limit, len(items)))

    def run(item):
        _local.in_pool = True
        try:
            return func(item)
        finally:
            _local.in_pool = False

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        pending = deque()
        cursor = 0
        while cursor < limit:
            pending.append(executor.submit(run, items[cursor]))
            cursor += 1

        index = 0
        while pending:
            future = pending.popleft()
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001
                if on_error == "raise":
                    for queued in pending:
                        queued.cancel()
                    raise
                result = exc

            # Refill before yielding, so a worker is never idle while the
            # caller is busy with the result just handed to it.
            if cursor < len(items):
                pending.append(executor.submit(run, items[cursor]))
                cursor += 1

            yield index, items[index], result
            # Drop the reference before waiting on the next future, so the
            # array the caller has finished with can be collected now rather
            # than at the end of the loop body.
            del result
            index += 1


def parallel_map(
    func,
    items,
    workers=None,
    progress=None,
    on_error="raise",
    scope=ITEMS,
):
    """Apply *func* to every item, concurrently, preserving input order.

    Every result is held until the last one arrives. When the results are
    large and the caller consumes them one at a time -- exporting a file,
    writing into a layer -- use :func:`parallel_stream` instead, which bounds
    how many exist at once.

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
    scope : {'items', 'bands'}, optional
        Which parallelism switch governs this pool. Forwarded to
        :func:`parallel_stream`.

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

    results = [None] * len(items)
    first_error = None

    for index, _item, result in parallel_stream(
        func,
        items,
        workers=workers,
        max_in_flight=len(items),
        on_error="collect",
        scope=scope,
    ):
        results[index] = result
        if first_error is None and isinstance(result, BaseException):
            first_error = result
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


def workers_for_memory(item_bytes, n_items=None, workers=None, fraction=None):
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
        Share of free memory the pool may occupy. Defaults to
        :func:`memory_fraction`, the plugin-wide budget, which leaves the
        rest of the headroom for the caller's own accumulation -- for a
        mosaic, the stitched canvas being built alongside the tiles.

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

    affordable = items_for_memory(item_bytes, fraction=fraction)
    if affordable is None:
        return limit
    return max(1, min(limit, affordable))


def items_for_memory(item_bytes, fraction=None):
    """Return how many *item_bytes*-sized items fit in the memory budget.

    The sizing rule behind both :func:`workers_for_memory` (how many workers
    may run) and the in-flight limits (how many finished results may wait to
    be consumed). Kept separate because those two are different questions
    about the same budget: a pool of four workers streaming into a slow
    consumer can hold far more than four items' worth of memory.

    Parameters
    ----------
    item_bytes : int
        Estimated peak bytes one item holds. ``0`` or ``None`` means the
        size is unknown.
    fraction : float, optional
        Share of free memory to allow. Defaults to :func:`memory_fraction`.

    Returns
    -------
    int or None
        How many items fit, at least ``1``; or ``None`` when the size or the
        free memory could not be determined, meaning "no cap".
    """
    if not item_bytes:
        return None
    if fraction is None and not memory_budget_enabled():
        return None
    free = available_memory()
    if not free:
        return None
    if fraction is None:
        fraction = memory_fraction()
    return max(1, int((free * fraction) // int(item_bytes)))


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

    n_bands = default_workers(allowed, workers, scope=BANDS)
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
    return parallel_map(
        lambda b: func(b[0], b[1]), bounds, workers=workers, scope=BANDS
    )


def _row_axis_size(array):
    """Return the length of *array*'s row axis, or ``0`` if it has none."""
    shape = np.shape(array)
    return shape[-2] if len(shape) >= 2 else 0


def _should_split(array, workers=None):
    """Return whether *array* is big enough to be worth banding."""
    if in_worker_thread():
        return False
    if default_workers(workers=workers, scope=BANDS) <= 1:
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

    probe_shape = (2 * size + 1, 2 * size + 1)
    probe_out = phasor_filter_median(
        np.zeros(probe_shape, dtype=mean.dtype),
        np.zeros(probe_shape, dtype=real.dtype),
        np.zeros(probe_shape, dtype=imag.dtype),
        repeat=1,
        size=size,
    )

    out_mean = np.empty(mean.shape, dtype=np.asarray(probe_out[0]).dtype)
    out_real = np.empty(real.shape, dtype=np.asarray(probe_out[1]).dtype)
    out_imag = np.empty(imag.shape, dtype=np.asarray(probe_out[2]).dtype)

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

    if not isinstance(signal, np.ndarray) or not isinstance(axis, int):
        return run(signal)

    if signal.ndim < 3 or in_worker_thread():
        return run(signal)
    if default_workers(workers=workers, scope=BANDS) <= 1:
        return run(signal)

    if signal.size < MIN_PARALLEL_PIXELS:
        return run(signal)

    hist_axis = axis % signal.ndim
    spatial = [i for i in range(signal.ndim) if i != hist_axis]

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
