import os

from qtpy import API, QtCore, QtGui
from qtpy.QtWidgets import (
    QComboBox,
    QLabel,
    QMenu,
    QPushButton,
    QStyle,
)

from napari_phasors._utils import CheckableComboBox


def test_qt_backend_identification():
    """Verify that the detected Qt backend matches expectations."""
    expected_api = os.environ.get('QT_API')
    if expected_api:
        # Normalize comparison (pyside6 vs PySide6)
        assert API.lower() == expected_api.lower()
    print(f"Active Qt Backend: {API}")


def test_basic_widget_instantiation(qtbot):
    """Ensure basic widgets from qtpy can be instantiated."""
    label = QLabel("Test Label")
    button = QPushButton("Test Button")
    combo = QComboBox()

    qtbot.addWidget(label)
    qtbot.addWidget(button)
    qtbot.addWidget(combo)

    assert label.text() == "Test Label"
    assert button.text() == "Test Button"


def test_checkable_combobox_interactions(qtbot):
    """
    Test CheckableComboBox interactions to ensure eventFilter
    logic for various QEvent types is working.
    """
    combo = CheckableComboBox()
    qtbot.addWidget(combo)

    # Add items
    combo.addItem("Item 1")
    combo.addItem("Item 2")

    line_edit = combo.lineEdit()
    viewport = combo.view().viewport()

    # 1. Test MouseButtonPress / Release on line edit
    # This triggers the eventFilter in _utils.py (lines 979-1037)
    qtbot.mouseClick(line_edit, QtCore.Qt.LeftButton)

    # 2. Test MouseMove on viewport (for hover logic)
    pos = viewport.rect().center()
    event = QtGui.QMouseEvent(
        QtCore.QEvent.MouseMove,
        QtCore.QPointF(pos),
        QtCore.Qt.NoButton,
        QtCore.Qt.NoButton,
        QtCore.Qt.NoModifier,
    )
    QtCore.QCoreApplication.sendEvent(viewport, event)

    # 3. Test Leave event on viewport
    leave_event = QtCore.QEvent(QtCore.QEvent.Leave)
    QtCore.QCoreApplication.sendEvent(viewport, leave_event)


def test_qmenu_exec_modernization():
    """Verify that QMenu has 'exec' method (migrated from 'exec_')."""
    menu = QMenu()
    assert hasattr(menu, 'exec')
    # Note: exec_ might still exist for compatibility in some backends,
    # but we should use exec.
    assert callable(menu.exec)


def test_qstyle_icon_compatibility():
    """Verify QStyle constants are accessible as used in selection_tab.py."""
    # We used QStyle.SP_BrowserReload in selection_tab.py
    assert hasattr(QStyle, 'SP_BrowserReload')
    button = QPushButton()
    icon = button.style().standardIcon(QStyle.SP_BrowserReload)
    assert not icon.isNull()
