"""Dialog to select channels to import from multi-channel files."""

from typing import Any

from qtpy.QtWidgets import (
    QCheckBox,
    QDialog,
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
    batch_size : int, optional
        Number of files being opened together. When more than one, the dialog
        says the selection is reused for every file with the same channels.
    parent : QWidget, optional
        Parent widget.
    """

    def __init__(
        self,
        channel_labels: list,
        filename: str = "",
        batch_size: int = 1,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Select Channels to Import")
        self.setMinimumWidth(360)
        if parent is None:
            # Without a parent the dialog does not inherit napari's stylesheet
            # and falls back to the platform palette (a gray window that does
            # not match the rest of the plugin). Apply the theme explicitly.
            self._apply_napari_theme()

        self._channel_labels = list(channel_labels)
        self._checkboxes: list[tuple[int, Any, QCheckBox]] = []

        layout = QVBoxLayout(self)

        title_text = (
            f"Select channels to import from '{filename}':"
            if filename
            else "Select channels to import:"
        )
        layout.addWidget(QLabel(title_text))

        if batch_size > 1:
            batch_note = QLabel(
                f"Opening {batch_size} files: this selection is applied to "
                "every file with the same channels. Files with different "
                "channels are asked about separately."
            )
            batch_note.setWordWrap(True)
            batch_note.setStyleSheet("color: grey;")
            layout.addWidget(batch_note)

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
        ok_cancel_layout = QHBoxLayout()
        self.ok_btn = QPushButton("OK")
        self.cancel_btn = QPushButton("Cancel")
        self.ok_btn.setDefault(True)
        self.ok_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)
        ok_cancel_layout.addWidget(self.ok_btn)
        ok_cancel_layout.addWidget(self.cancel_btn)
        layout.addLayout(ok_cancel_layout)

        # Aliases kept for readability at the call sites.
        self.btn_select_all = self.select_all_btn
        self.btn_deselect_all = self.deselect_all_btn
        self.single_layer_checkbox = self.single_layer_check

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
        ok_btn = getattr(self, "ok_btn", None)
        if ok_btn is not None:
            ok_btn.setEnabled(has_selection)

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

    def _apply_napari_theme(self):
        """Style the dialog with napari's current theme stylesheet."""
        try:
            from napari.qt import get_stylesheet
            from napari.settings import get_settings

            self.setStyleSheet(get_stylesheet(get_settings().appearance.theme))
        except Exception:  # noqa: BLE001 - napari without Qt, or no settings
            pass
