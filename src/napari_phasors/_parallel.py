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
"""

import os
import threading
from concurrent.futures import ThreadPoolExecutor

__all__ = [
    "default_workers",
    "parallel_map",
    "parallel_compute_apply",
    "worker_limit_from_env",
    "workers_for_memory",
    "available_memory",
]

#: Never spawn more than this many threads, however many cores are reported.
#: Beyond roughly this point the array work is memory-bandwidth bound and the
#: extra threads only add contention and peak memory.
MAX_WORKERS = 16

#: Set to an integer to override the worker count everywhere. ``1`` disables
#: concurrency entirely, which is the escape hatch when debugging.
WORKERS_ENV_VAR = "NAPARI_PHASORS_WORKERS"

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
