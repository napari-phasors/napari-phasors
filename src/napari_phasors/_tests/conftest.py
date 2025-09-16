import pytest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QDialog


@pytest.fixture(autouse=True)
def _hide_qdialog(monkeypatch):
    orig_show = QDialog.show

    def hidden_show(self):
        self.setAttribute(Qt.WA_DontShowOnScreen, True)
        orig_show(self)

    monkeypatch.setattr(QDialog, "show", hidden_show)
