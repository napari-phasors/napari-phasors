import functools
import hashlib
import pathlib
import tempfile

import pytest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QDialog, QWidget

# Patch napari's _QtMainWindow.eventFilter to guard against PySide6 passing
# QWidgetItem (a non-QObject) as the `watched` argument, which causes a
# TypeError / infinite recursion under PySide6 + Python 3.14.
try:
    from napari._qt.qt_main_window import _QtMainWindow
    from qtpy.QtCore import QObject

    _orig_QtMainWindow_eventFilter = _QtMainWindow.eventFilter

    def _safe_eventFilter(self, source, event):
        if not isinstance(source, QObject):
            return False
        return _orig_QtMainWindow_eventFilter(self, source, event)

    _QtMainWindow.eventFilter = _safe_eventFilter
except (AttributeError, ImportError, TypeError):
    pass


def configure_phasorpy_retries():
    """Opt every phasorpy dataset repository into a few download retries.

    ``phasorpy.datasets.fetch`` downloads sample files (e.g. "simfcs.r64")
    from GitHub/Zenodo on first use via Pooch, which by default makes a
    single attempt per file. On CI this occasionally hits a transient
    ``requests.exceptions.ReadTimeout`` against github.com, failing the test
    even though a retry would succeed. Pooch retries connection errors
    (including read timeouts) itself when a repository's ``retry_if_failed``
    is set, so opt every phasorpy dataset repository into a few retries with
    backoff instead of failing on the first flake. Silently does nothing if
    phasorpy's dataset registry is unavailable or has a different shape.
    """
    try:
        from phasorpy.datasets import REPOSITORIES as _phasorpy_repositories

        for repo in _phasorpy_repositories.values():
            repo.retry_if_failed = 3
    except (AttributeError, ImportError, TypeError):
        pass


configure_phasorpy_retries()


# Under ``-n auto --dist loadfile`` different test files run in separate
# processes sharing one download cache, and several of them ask for the same
# dataset (e.g. ``test_reader.py`` and ``test_widget.py`` both fetch
# "simfcs.b&h", which pooch unzips into that shared cache). Two workers doing
# that at once race on the same cache entry, seen on Windows CI as
# ``zipfile.BadZipFile: Bad CRC-32 for file 'simfcs.b&h'``. Locking per target
# file makes the second worker wait and then hit the cache, while different
# datasets still download in parallel. ``pooch.Pooch.fetch`` is the single
# choke point for every download in the suite, so patching it covers phasorpy's
# ``fetch``, ``test_data_utils`` and the sample-data loaders alike.

# Seconds to wait for another worker's download of the same file to finish.
_FETCH_LOCK_TIMEOUT = 600


def _fetch_lock_dir():
    """Directory holding the per-file lock files, shared by all workers."""
    return pathlib.Path(tempfile.gettempdir()) / "napari-phasors-fetch-locks"


def _fetch_lock_path(pooch_instance, fname):
    """Lock file for ``fname`` in ``pooch_instance``'s cache.

    Keyed by a hash of the absolute target path: the same dataset locks the
    same file across workers, distinct datasets never block each other, and
    names containing ``&`` or path separators stay filesystem-safe.
    """
    try:
        target = str(pathlib.Path(pooch_instance.abspath) / fname)
    except (AttributeError, TypeError):  # not a Pooch-shaped object
        target = str(fname)
    digest = hashlib.sha256(target.encode("utf-8")).hexdigest()[:16]
    return _fetch_lock_dir() / f"{digest}.lock"


def make_locked_fetch(fetch_function):
    """Wrap ``fetch_function`` so one file is fetched by one process at a time."""
    from filelock import FileLock, Timeout

    @functools.wraps(fetch_function)
    def locked_fetch(self, fname, *args, **kwargs):
        lock_path = _fetch_lock_path(self, fname)
        try:
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            lock = FileLock(str(lock_path), timeout=_FETCH_LOCK_TIMEOUT)
            lock.acquire()
        except (Timeout, OSError):
            # Never let the lock itself break a test: if it cannot be taken
            # (read-only temp dir, or a worker died holding it), fall back to
            # the unserialised fetch we had before.
            return fetch_function(self, fname, *args, **kwargs)
        try:
            return fetch_function(self, fname, *args, **kwargs)
        finally:
            lock.release()

    locked_fetch.__wrapped_by_napari_phasors__ = True
    return locked_fetch


def serialize_pooch_downloads():
    """Make every ``pooch.Pooch.fetch`` in this process take a per-file lock.

    Silently does nothing if pooch/filelock are unavailable or the patch is
    already installed.
    """
    try:
        import pooch
        from filelock import FileLock  # noqa: F401 - checked before patching

        if getattr(pooch.Pooch.fetch, "__wrapped_by_napari_phasors__", False):
            return
        pooch.Pooch.fetch = make_locked_fetch(pooch.Pooch.fetch)
    except (AttributeError, ImportError, TypeError):
        pass


serialize_pooch_downloads()


# --- Transient network failures become skips, not failures ------------------
#
# Tests that exercise the sample-data loaders and the file readers download
# tens of megabytes from GitHub/Zenodo through pooch. Pooch already retries
# connection errors (``retry_if_failed``, set on the sample-data downloaders,
# on ``test_data_utils.test_data_downloader`` and on phasorpy's repositories
# above), but when a host stays slow or unreachable for longer than the
# retries cover, the exception still surfaces and reads as a test failure --
# e.g. ``requests.exceptions.ReadTimeout: HTTPSConnectionPool(host=
# 'zenodo.org', port=443): Read timed out. (read timeout=30)``. That says
# nothing about the code under test, so once the retries are exhausted we turn
# such a failure into a skip.
#
# Only transport-level errors and retryable HTTP statuses (429 and 5xx) count.
# A 404 -- a wrong URL or a stale registry entry -- is a real bug and still
# fails, and so does every non-network error.

_TRANSIENT_HTTP_STATUSES = frozenset({429})


def _http_status(exc):
    """Return the HTTP status carried by ``exc``, or None if it carries none.

    ``urllib.error.HTTPError`` exposes ``.code``; ``requests``' ``HTTPError``
    exposes ``.response.status_code`` (other ``requests`` exceptions have a
    ``.response`` attribute too, but it is ``None``).
    """
    status = getattr(exc, "code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    return status if isinstance(status, int) else None


def _transient_network_error_types():
    """Exception classes that mean "the network flaked", not "bad code"."""
    import http.client
    import socket
    import urllib.error

    types = [
        ConnectionError,  # builtin: reset/aborted/refused connections
        socket.gaierror,  # DNS resolution failure
        http.client.RemoteDisconnected,
        http.client.IncompleteRead,
        urllib.error.URLError,
    ]

    try:
        import requests.exceptions as requests_exc

        types += [
            requests_exc.ConnectionError,
            requests_exc.Timeout,  # includes ReadTimeout / ConnectTimeout
            requests_exc.ChunkedEncodingError,
        ]
    except ImportError:  # pragma: no cover - requests ships with pooch
        pass

    try:
        import urllib3.exceptions as urllib3_exc

        types += [
            urllib3_exc.TimeoutError,
            urllib3_exc.ProtocolError,
            urllib3_exc.NewConnectionError,
            urllib3_exc.MaxRetryError,
        ]
    except ImportError:  # pragma: no cover - urllib3 ships with requests
        pass

    return tuple(types)


def _iter_exception_chain(exc):
    """Yield ``exc`` and every exception it was raised from/during."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        yield exc
        exc = exc.__cause__ or exc.__context__


def network_flake_reason(exc):
    """Describe ``exc`` as a transient network failure, or return None.

    Walks the exception chain, so a download error wrapped by pooch (or by a
    reader) is still recognised.
    """
    transient_types = _transient_network_error_types()
    for err in _iter_exception_chain(exc):
        status = _http_status(err)
        if status is not None:
            # A response did come back: only server-side and rate-limit
            # statuses are worth retrying, anything else is a real failure.
            if status in _TRANSIENT_HTTP_STATUSES or 500 <= status < 600:
                return f"HTTP {status} ({type(err).__name__}): {err}"
            continue
        if isinstance(err, transient_types):
            return f"{type(err).__name__}: {err}"
    return None


def skip_if_network_flake(exc):
    """Skip the running test if ``exc`` is a network flake, else re-raise."""
    reason = network_flake_reason(exc)
    if reason is None:
        raise exc
    pytest.skip(f"transient network failure after retries - {reason}")


@pytest.hookimpl(wrapper=True)
def pytest_runtest_setup(item):
    try:
        return (yield)
    except Exception as exc:  # noqa: BLE001 - re-raised unless it is a flake
        skip_if_network_flake(exc)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    try:
        return (yield)
    except Exception as exc:  # noqa: BLE001 - re-raised unless it is a flake
        skip_if_network_flake(exc)


# Harden superqt's SliderLabel against the same PySide6 + Python 3.14 shiboken
# wrapper-corruption bug: under address reuse, `widget.style()` can return a
# transient QWidgetItem instead of a QStyle, so `_get_size()` raises
# `AttributeError: 'QWidgetItem' object has no attribute 'sizeFromContents'`.
# This is triggered deep inside napari's layer-controls creation (via
# `viewer.add_image`), aborts the `events.inserted` callback, and cascades into
# KeyError on layer removal at teardown and a leaked QtViewer in the next test.
# `_update_size` only sets a cosmetic fixed size, so swallowing a corrupted-
# style failure and keeping the current size is safe.
try:
    from superqt.sliders._labeled import SliderLabel

    _orig_SliderLabel_update_size = SliderLabel._update_size

    def _safe_update_size(self, *args):
        try:
            return _orig_SliderLabel_update_size(self, *args)
        except (AttributeError, TypeError):
            # PySide6/shiboken handed back a non-QStyle object; skip the
            # cosmetic resize rather than letting it abort widget creation.
            return None

    SliderLabel._update_size = _safe_update_size
except (AttributeError, ImportError, TypeError):
    pass


# Same PySide6/shiboken wrapper-corruption bug as above, hitting matplotlib's
# Qt canvas this time: `FigureCanvasQT.showEvent` does
# `self.window().windowHandle()`, which should be a QWindow, to wire up
# HiDPI pixel-ratio signals. Under teardown-time address reuse this can come
# back as a stale QWidgetItem instead, so `.installEventFilter(self)` raises
# `AttributeError: 'QWidgetItem' object has no attribute 'installEventFilter'`.
# Only reproduces with PySide6 (not PyQt), during Qt event-loop teardown.
# The pixel-ratio wiring is cosmetic, so skip it on failure rather than
# aborting the surrounding show()/event handling.
try:
    from matplotlib.backends.backend_qt import FigureCanvasQT

    _orig_FigureCanvasQT_showEvent = FigureCanvasQT.showEvent

    def _safe_showEvent(self, event):
        try:
            return _orig_FigureCanvasQT_showEvent(self, event)
        except AttributeError:
            return None

    FigureCanvasQT.showEvent = _safe_showEvent
except (AttributeError, ImportError, TypeError):
    pass


@pytest.fixture(autouse=True)
def _ensure_qapp(qapp):
    """Guarantee a QApplication exists for every test.

    Many widget tests do not request ``qtbot``; in a full-file run they
    piggyback on a QApplication created by an earlier test, but in isolation
    constructing a QWidget without one aborts the process.
    """
    return qapp


@pytest.fixture(autouse=True)
def _hide_widgets_on_screen(monkeypatch):
    """Keep every widget/dialog shown during a test off the physical screen.

    Several plugin widgets call ``self.show()`` themselves (e.g.
    ``HistogramWidget.update_data``, the ``PopoutWindowMixin`` "Phasor
    Custom Import" window) in addition to the dialogs tests open directly.
    Left unpatched, those calls pop up real windows/plots while the suite
    runs. Setting ``Qt.WA_DontShowOnScreen`` before ``show()`` keeps Qt's
    normal layout/rendering machinery working (so ``isVisible()``, size
    hints, ``showEvent`` etc. all still behave the same) without mapping a
    window onto the display. ``QWidget.show`` and ``QDialog.show`` are
    separate bound methods in PyQt/PySide, so both need patching.
    """

    def _make_hidden_show(orig_show):
        def hidden_show(self, *args, **kwargs):
            self.setAttribute(Qt.WA_DontShowOnScreen, True)
            return orig_show(self, *args, **kwargs)

        return hidden_show

    monkeypatch.setattr(QWidget, "show", _make_hidden_show(QWidget.show))
    monkeypatch.setattr(QDialog, "show", _make_hidden_show(QDialog.show))


@pytest.fixture(autouse=True)
def _stub_color_dialog(monkeypatch):
    """Prevent ``QColorDialog.getColor`` from opening a real color picker.

    Several color-swatch buttons (marker color, contour color, cursor
    color, ...) call ``QColorDialog.getColor(...)`` on click. On most
    platforms this uses the *native* OS color panel rather than going
    through Qt's own ``QDialog``/``QWidget`` machinery, so it bypasses the
    ``_hide_widgets_on_screen`` patch above entirely and pops up a real
    little window during the test run. Default to returning the initial
    color unchanged (as if the user closed the picker without changing
    anything); a test that needs to simulate picking a specific color can
    still override this locally with its own ``monkeypatch.setattr``.
    """
    from qtpy.QtGui import QColor
    from qtpy.QtWidgets import QColorDialog

    def _fake_get_color(*args, **kwargs):
        for arg in args:
            if isinstance(arg, QColor):
                return arg
        initial = kwargs.get("initial")
        if isinstance(initial, QColor):
            return initial
        return QColor()

    monkeypatch.setattr(QColorDialog, "getColor", _fake_get_color)


@pytest.fixture(autouse=True)
def _cleanup_widgets_after_test(request):
    """Ensure all Phasor widgets instantiated during the test are properly deleted.

    This avoids PySide6 segmentation faults and background timer leaks caused by
    unclean widget lifecycles in PySide6.
    """
    if "make_napari_viewer" in request.fixturenames:
        request.getfixturevalue("make_napari_viewer")
    yield
    import contextlib

    import matplotlib.pyplot as plt
    from qtpy.QtWidgets import QApplication

    from napari_phasors.plotter import PlotterWidget

    widgets = []
    with contextlib.suppress(Exception):
        widgets = QApplication.allWidgets()

    # Collect every widget defined by this plugin, not a hand-maintained list of
    # types. Heavy widgets instantiated many times in a single test file and
    # never explicitly closed — notably BatchAnalysisWidget (~140 instances in
    # test_batch_analysis.py) and the standalone analysis tabs — otherwise
    # accumulate on one ``loadfile`` xdist worker and segfault during PySide6
    # teardown near the end of the file. Closing them here also runs each
    # widget's ``closeEvent``, which disconnects its ``viewer.layers.events``
    # handlers so the (longer-lived) viewer can't fire into a freed widget.
    phasor_widgets = [
        w
        for w in widgets
        if type(w).__module__.split(".")[0] == "napari_phasors"
    ]

    # 1. Break parent relationships to avoid double-free/deletion issues in PySide6
    for w in phasor_widgets:
        with contextlib.suppress(Exception):
            w.setParent(None)

    # 2. Clean up Matplotlib canvases and figures
    for w in phasor_widgets:
        if hasattr(w, "figure") and w.figure is not None:
            with contextlib.suppress(Exception):
                plt.close(w.figure)
        if hasattr(w, "canvas") and w.canvas is not None:
            with contextlib.suppress(Exception):
                w.canvas.setParent(None)
                w.canvas.deleteLater()

    # 3. Safely stop timers, close, and delete our widgets
    for w in phasor_widgets:
        if isinstance(w, PlotterWidget):
            for attr in (
                '_dock_check_timer',
                '_analysis_dock_init_timer',
                '_dock_resize_timer',
                '_layer_selection_timer',
                '_bins_timer',
                '_resize_canvas_timer',
            ):
                with contextlib.suppress(AttributeError):
                    timer = getattr(w, attr, None)
                    if timer is not None:
                        timer.stop()

        with contextlib.suppress(Exception):
            w.close()
            w.deleteLater()

    # 4. Process all pending Qt events to execute deleteLater calls.
    # ``processEvents()`` alone does NOT deliver ``DeferredDelete`` events,
    # so without the explicit ``sendPostedEvents`` flush the C++ side of the
    # widgets deleteLater()'d above is destroyed at some arbitrary later
    # event-loop spin — e.g. while the next test is constructing its napari
    # viewer — leaving Python wrappers pointing at freed Qt objects.
    from qtpy.QtCore import QCoreApplication, QEvent

    with contextlib.suppress(Exception):
        QCoreApplication.processEvents()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        QCoreApplication.processEvents()

    # 5. Collect cyclic garbage now, at a controlled point.
    # The root conftest disables automatic GC under PySide6, so reference
    # cycles (matplotlib Figure <-> canvas, closed widgets captured by
    # lambdas/signal closures) otherwise accumulate for the worker's whole
    # lifetime. The first ``make_napari_viewer`` test then detonates them:
    # napari's fixture calls ``gc.collect()`` during *setup*, destroying
    # hundreds of stale Qt wrappers mid-viewer-construction, which
    # segfaults PySide6 xdist workers ("worker 'gwN' crashed" at the first
    # make_napari_viewer test after widget-heavy files). Collecting here —
    # right after the plugin widgets were closed and their deferred
    # deletions flushed, with no viewer half-built — keeps every collection
    # small and safe.
    import gc

    with contextlib.suppress(Exception):
        gc.collect()


@pytest.fixture
def make_viewer_model():
    """Create a headless ViewerModel factory for faster testing."""
    from napari.components.viewer_model import ViewerModel

    viewers = []

    def factory():
        viewer = ViewerModel()
        viewers.append(viewer)
        return viewer

    yield factory

    for v in viewers:
        v.layers.clear()
