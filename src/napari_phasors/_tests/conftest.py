import pytest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QDialog

from napari_phasors._synthetic_generator import (
    make_intensity_layer_with_phasors,
    make_raw_flim_data,
)


def create_image_layer_with_phasors(harmonic=None):
    """Create an intensity image layer with phasors for testing."""
    if harmonic is None:
        harmonic = [1, 2, 3]
    time_constants = [0.1, 1, 2, 3, 4, 5, 10]
    raw_flim_data = make_raw_flim_data(time_constants=time_constants)
    return make_intensity_layer_with_phasors(raw_flim_data, harmonic=harmonic)


@pytest.fixture(autouse=True)
def _hide_qdialog(monkeypatch):
    orig_show = QDialog.show

    def hidden_show(self):
        self.setAttribute(Qt.WA_DontShowOnScreen, True)
        orig_show(self)

    monkeypatch.setattr(QDialog, "show", hidden_show)
