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
