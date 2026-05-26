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

    from napari_phasors._widget import (
        AdvancedOptionsWidget,
        PhasorTransform,
        WriterWidget,
    )
    from napari_phasors.plotter import PlotterWidget

    widgets = []
    with contextlib.suppress(Exception):
        widgets = QApplication.allWidgets()

    phasor_widgets = []
    for w in widgets:
        if isinstance(
            w,
            (
                PhasorTransform,
                WriterWidget,
                AdvancedOptionsWidget,
                PlotterWidget,
            ),
        ):
            phasor_widgets.append(w)

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
