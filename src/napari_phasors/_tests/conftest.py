import pytest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QDialog

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


@pytest.fixture(autouse=True)
def _ensure_qapp(qapp):
    """Guarantee a QApplication exists for every test.

    Many widget tests do not request ``qtbot``; in a full-file run they
    piggyback on a QApplication created by an earlier test, but in isolation
    constructing a QWidget without one aborts the process.
    """
    return qapp


@pytest.fixture(autouse=True)
def _hide_qdialog(monkeypatch):
    orig_show = QDialog.show

    def hidden_show(self):
        self.setAttribute(Qt.WA_DontShowOnScreen, True)
        orig_show(self)

    monkeypatch.setattr(QDialog, "show", hidden_show)


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

    # 4. Process all pending Qt events to execute deleteLater calls
    from qtpy.QtCore import QCoreApplication

    with contextlib.suppress(Exception):
        QCoreApplication.processEvents()


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
