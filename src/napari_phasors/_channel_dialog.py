"""Dialog to select channels to import from multi-channel files."""

from typing import Any

from qtpy.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)


class ChannelSelectionDialog(QDialog):
    """Dialog allowing user to choose which channels to import and whether
    to stack them in a single layer.

    Parameters
    ----------
    channel_labels : list
        List of channel labels or indices present in the file.
    filename : str
        The filename of the file being loaded.
    parent : QWidget, optional
        Parent widget.
    """

    def __init__(
        self,
        channel_labels: list,
        filename: str = "",
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Select Channels to Import")
        self.setMinimumWidth(360)

        self._channel_labels = list(channel_labels)
        self._checkboxes: list[tuple[int, Any, QCheckBox]] = []

        layout = QVBoxLayout(self)

        title_text = (
            f"Select channels to import from '{filename}':"
            if filename
            else "Select channels to import:"
        )
        layout.addWidget(QLabel(title_text))

        # Checklist inside a scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(250)
        list_container = QWidget()
        list_layout = QVBoxLayout(list_container)
        list_layout.setContentsMargins(4, 4, 4, 4)
        list_layout.setSpacing(6)

        for pos, label in enumerate(self._channel_labels):
            cb = QCheckBox(f"Channel {label}")
            cb.setChecked(True)
            cb.toggled.connect(self._update_button_states)
            self._checkboxes.append((pos, label, cb))
            list_layout.addWidget(cb)

        list_layout.addStretch()
        scroll.setWidget(list_container)
        layout.addWidget(scroll)

        # Select All / Deselect All buttons
        btn_layout = QHBoxLayout()
        self.select_all_btn = QPushButton("Select All")
        self.deselect_all_btn = QPushButton("Deselect All")
        self.select_all_btn.clicked.connect(self.select_all)
        self.deselect_all_btn.clicked.connect(self.deselect_all)
        btn_layout.addWidget(self.select_all_btn)
        btn_layout.addWidget(self.deselect_all_btn)
        layout.addLayout(btn_layout)

        # Single layer checkbox
        self.single_layer_check = QCheckBox(
            "Import channels in the same layer"
        )
        self.single_layer_check.setToolTip(
            "Stack all selected channels into a single multi-dimensional layer\n"
            "with a channel slider bar in the napari viewer."
        )
        layout.addWidget(self.single_layer_check)

        # OK / Cancel
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        self.ok_btn = self.button_box.button(QDialogButtonBox.Ok)
        self.btn_select_all = self.select_all_btn
        self.btn_deselect_all = self.deselect_all_btn
        self.single_layer_checkbox = self.single_layer_check

        layout.addWidget(self.button_box)

        self._update_button_states()

    @property
    def checkboxes(self) -> list[QCheckBox]:
        """Return list of channel checkboxes."""
        return [cb for _, _, cb in self._checkboxes]

    def set_channel_checked(self, index: int, checked: bool):
        """Set checked state for channel at `index`."""
        if 0 <= index < len(self._checkboxes):
            self._checkboxes[index][2].setChecked(checked)

    def select_all(self):
        """Check all channel checkboxes."""
        for _, _, cb in self._checkboxes:
            cb.setChecked(True)

    def deselect_all(self):
        """Uncheck all channel checkboxes."""
        for _, _, cb in self._checkboxes:
            cb.setChecked(False)

    def _update_button_states(self):
        """Update Import button state (enabled only when >= 1 channel is selected)."""
        has_selection = any(cb.isChecked() for _, _, cb in self._checkboxes)
        if self.ok_btn:
            self.ok_btn.setEnabled(has_selection)

    def get_selected_channel_positions(self) -> list[int]:
        """Return indices of selected channels."""
        return [pos for pos, _, cb in self._checkboxes if cb.isChecked()]

    def selected_channels(self) -> list[int]:
        """Alias for get_selected_channel_positions."""
        return self.get_selected_channel_positions()

    def get_selected_channel_labels(self) -> list:
        """Return labels of selected channels."""
        return [label for _, label, cb in self._checkboxes if cb.isChecked()]

    def is_single_layer(self) -> bool:
        """Return True if channels should be imported into a single layer."""
        return self.single_layer_check.isChecked()
