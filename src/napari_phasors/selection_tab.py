import contextlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, Wedge
from napari.layers import Labels
from napari.utils import DirectLabelColormap
from phasorpy.cluster import phasor_cluster_gmm
from phasorpy.cursor import (
    mask_from_circular_cursor,
    mask_from_elliptic_cursor,
    mask_from_polar_cursor,
)
from qtpy import uic
from qtpy.QtCore import QPointF, QSize, Qt, Signal
from qtpy.QtGui import (
    QColor,
    QIcon,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from qtpy.QtWidgets import (
    QApplication,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QStyle,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)
from superqt import QToggleSwitch

from ._utils import colormap_to_dict


def _make_eye_icon(color, size=18, crossed=False):
    """Return a simple grey eye-outline ``QIcon`` drawn in ``color``.

    The eye is an almond outline with a small pupil, stroked (not filled).
    When ``crossed`` is True a diagonal slash is drawn across it (the
    standard "hidden" eye), keeping the same colour as the plain eye.
    """
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    pen = QPen(QColor(color))
    pen.setWidthF(1.4)
    painter.setPen(pen)
    painter.setBrush(Qt.NoBrush)

    margin = size * 0.14
    mid_y = size / 2.0
    path = QPainterPath()
    path.moveTo(margin, mid_y)
    path.quadTo(size / 2.0, margin, size - margin, mid_y)  # upper lid
    path.quadTo(size / 2.0, size - margin, margin, mid_y)  # lower lid
    painter.drawPath(path)

    pupil_r = size * 0.16
    painter.drawEllipse(QPointF(size / 2.0, mid_y), pupil_r, pupil_r)

    if crossed:
        painter.drawLine(
            QPointF(margin, size - margin),
            QPointF(size - margin, margin),
        )

    painter.end()
    return QIcon(pixmap)


class SelectionWidget(QWidget):
    """
    Widget for interactive phasor selection using the cursor in napari.

    Provides:
      - A dropdown to manage and select manual or custom selection IDs
      - Manual selection mode for free-form selection
      - Circular cursor selection mode for defining circular ROIs

    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer instance.
    parent : QWidget, optional
        The parent widget (typically the main PlotterWidget).

    Notes
    -----
    This widget is designed to be used as a tab within the main PlotterWidget.

    """

    def __init__(self, viewer, parent=None):
        """Initialize the SelectionWidget."""
        super().__init__()
        self.parent_widget = parent
        self.viewer = viewer

        # Main layout
        layout = QVBoxLayout(self)

        # Selection mode combobox at the top
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Selection Mode:"))
        self.selection_mode_combobox = QComboBox()
        self.selection_mode_combobox.addItems(
            [
                "Cursor Selection",
                "Automatic Clustering",
                "Manual Selection",
            ]
        )
        mode_layout.addWidget(self.selection_mode_combobox, 1)
        layout.addLayout(mode_layout)

        # Stacked widget to switch between modes
        self.stacked_widget = QStackedWidget()
        layout.addWidget(self.stacked_widget)

        # === Manual Selection Mode Widget ===
        self.manual_selection_widget = QWidget()
        manual_layout = QVBoxLayout(self.manual_selection_widget)
        manual_layout.setContentsMargins(0, 0, 0, 0)

        # Load the UI from the .ui file
        self.selection_input_widget = QWidget()
        uic.loadUi(
            Path(__file__).parent / "ui/selection_tab.ui",
            self.selection_input_widget,
        )
        manual_layout.addWidget(self.selection_input_widget)

        # Add default items to the selection id combobox
        self.selection_input_widget.phasor_selection_id_combobox.addItem(
            "None"
        )
        self.selection_input_widget.phasor_selection_id_combobox.addItem(
            "MANUAL SELECTION #1"
        )

        # Initialize the current selection id to match the default
        self._current_selection_id = "None"
        self.selection_id = "None"
        self._phasors_selected_layer = None

        # Create refresh button and add it to the scroll area layout
        self.refresh_selection_button = QPushButton()
        self.refresh_selection_button.setIcon(
            self.refresh_selection_button.style().standardIcon(
                QStyle.SP_BrowserReload
            )
        )
        self.refresh_selection_button.setMaximumWidth(35)
        self.refresh_selection_button.clicked.connect(
            self._on_refresh_selection_clicked
        )

        # Find the grid layout and add the button to row 4, column 3
        scroll_area_layout = self.selection_input_widget.findChild(
            QWidget, "scrollAreaWidgetContents"
        ).layout()
        if scroll_area_layout is not None:
            scroll_area_layout.addWidget(self.refresh_selection_button, 4, 3)

        # Connect to signals for user-initiated selection changes only.
        # Use activated (user clicks dropdown) + editingFinished (user
        # types in line edit).  Do NOT use currentIndexChanged — it fires
        # on programmatic changes too, causing duplicate callbacks.
        self.selection_input_widget.phasor_selection_id_combobox.activated.connect(
            self.on_selection_id_changed
        )
        if hasattr(
            self.selection_input_widget.phasor_selection_id_combobox,
            'lineEdit',
        ):
            line_edit = (
                self.selection_input_widget.phasor_selection_id_combobox.lineEdit()
            )
            if line_edit:
                line_edit.editingFinished.connect(self.on_selection_id_changed)

        # === Cursor Selection Mode Widget (index 0) ===
        self.cursor_selection_widget = CursorSelectionWidget(
            viewer, self.parent_widget
        )
        self.stacked_widget.addWidget(self.cursor_selection_widget)

        # === Automatic Clustering Mode Widget (index 1) ===
        self.automatic_clustering_widget = AutomaticClusteringWidget(
            viewer, self.parent_widget
        )
        self.stacked_widget.addWidget(self.automatic_clustering_widget)

        # === Manual Selection Mode Widget (index 2) ===
        self.stacked_widget.addWidget(self.manual_selection_widget)

        # Connect mode change
        self.selection_mode_combobox.currentIndexChanged.connect(
            self._on_selection_mode_changed
        )

    def on_harmonic_changed(self):
        """Callback when harmonic spinbox is changed."""
        # Only update cursor visibility for the currently active mode
        current_mode = self.selection_mode_combobox.currentIndex()
        if current_mode == 0:  # Cursor Selection mode
            self.cursor_selection_widget.on_harmonic_changed()
        elif current_mode == 1:  # Automatic Clustering mode
            self.automatic_clustering_widget.on_harmonic_changed()

    def clear_artists(self):
        """Clear (remove) all artists created by this widget."""
        # Clear artists from all sub-widgets
        if hasattr(self, 'cursor_selection_widget'):
            self.cursor_selection_widget.clear_all_patches()
        if hasattr(self, 'automatic_clustering_widget'):
            self.automatic_clustering_widget.clear_all_patches()

    def is_manual_selection_mode(self):
        """Check if manual selection mode is currently active."""
        return (
            self.selection_mode_combobox.currentIndex() == 2
        )  # Manual is index 2

    def _manage_labels_layer_visibility(self, show_manual):
        """Manage visibility of labels layers based on selection mode.

        Parameters
        ----------
        show_manual : bool
            If True, show manual selection layers and hide circular cursor layer.
            If False, show circular cursor layer and hide manual selection layers.
        """
        layer = self._get_current_layer()
        if layer is None:
            return

        for viewer_layer in self.viewer.layers:
            if not isinstance(viewer_layer, Labels):
                continue
            if not hasattr(viewer_layer, 'metadata'):
                continue

            # Check metadata tags to identify layer type
            if 'napari_phasors_selection_type' in viewer_layer.metadata:
                selection_type = viewer_layer.metadata[
                    'napari_phasors_selection_type'
                ]
                source_layer = viewer_layer.metadata.get(
                    'napari_phasors_source_layer'
                )

                # Only manage layers belonging to the current image layer
                if source_layer == layer.name:
                    if (
                        selection_type == 'cursor_selection'
                        or selection_type == 'automatic_clustering'
                    ):
                        viewer_layer.visible = not show_manual
                    elif selection_type == 'manual':
                        viewer_layer.visible = show_manual

    def _set_labels_layer_visibility(self, visible):
        """Toggle the visibility of all selection layers for the active tab."""
        if not visible:
            layer = self._get_current_layer()
            if layer is None:
                return
            for viewer_layer in self.viewer.layers:
                if not isinstance(viewer_layer, Labels) or not hasattr(
                    viewer_layer, 'metadata'
                ):
                    continue
                if 'napari_phasors_selection_type' in viewer_layer.metadata:
                    source_layer = viewer_layer.metadata.get(
                        'napari_phasors_source_layer'
                    )
                    if source_layer == layer.name:
                        viewer_layer.visible = False
        else:
            self._manage_labels_layer_visibility(
                show_manual=self.is_manual_selection_mode()
            )

    def _on_selection_mode_changed(self, index):
        """Handle selection mode change."""
        self.stacked_widget.setCurrentIndex(index)

        if index == 2:  # Manual selection mode
            self.cursor_selection_widget.clear_all_patches()
            self.automatic_clustering_widget.clear_all_patches()
            if self.parent_widget is not None:
                self.parent_widget._set_selection_visibility(True)
            self._manage_labels_layer_visibility(show_manual=True)
            self.update_phasor_plot_with_selection_id(self.selection_id)
        elif index == 1:  # Automatic clustering mode
            # Deactivate any active selection tools before hiding toolbar
            if self.parent_widget is not None:
                self.parent_widget.canvas_widget._on_escape(None)
            self.cursor_selection_widget.clear_all_patches()
            self.automatic_clustering_widget.redraw_all_patches()
            if self.parent_widget is not None:
                self.parent_widget._set_selection_visibility(False)
                self.parent_widget.plot(selection_id_data=None)
            self._manage_labels_layer_visibility(show_manual=False)
        else:  # Cursor selection mode (index 0)
            # Deactivate any active selection tools before hiding toolbar
            if self.parent_widget is not None:
                self.parent_widget.canvas_widget._on_escape(None)
            self.cursor_selection_widget.redraw_all_patches()
            self.automatic_clustering_widget.clear_all_patches()
            if self.parent_widget is not None:
                self.parent_widget._set_selection_visibility(False)
                self.parent_widget.plot(selection_id_data=None)
            self._manage_labels_layer_visibility(show_manual=False)

    @property
    def selection_id(self):
        """Gets or sets the selection id from the phasor selection id combobox."""
        if (
            self.selection_input_widget.phasor_selection_id_combobox.count()
            == 0
        ):
            return None
        else:
            current_text = (
                self.selection_input_widget.phasor_selection_id_combobox.currentText()
            )
            return (
                None
                if current_text == "None" or current_text == ""
                else current_text
            )

    @selection_id.setter
    def selection_id(self, new_selection_id: str):
        """Sets the selection id from the phasor selection id combobox."""
        if new_selection_id is None or new_selection_id == "":
            new_selection_id = "None"

        if new_selection_id not in [
            self.selection_input_widget.phasor_selection_id_combobox.itemText(
                i
            )
            for i in range(
                self.selection_input_widget.phasor_selection_id_combobox.count()
            )
        ]:
            self.selection_input_widget.phasor_selection_id_combobox.addItem(
                new_selection_id
            )
        self.selection_input_widget.phasor_selection_id_combobox.setCurrentText(
            new_selection_id
        )
        self._current_selection_id = new_selection_id

    def _get_current_layer(self):
        """Helper to get the currently selected image layer."""
        layer_name = (
            self.parent_widget.image_layer_with_phasor_features_combobox.currentText()
        )
        if not layer_name or layer_name not in self.viewer.layers:
            return None
        return self.viewer.layers[layer_name]

    def _get_selected_layers(self):
        """Get all currently selected layers."""
        if self.parent_widget is None:
            return []
        return self.parent_widget.get_selected_layers()

    def _get_primary_layer_name(self):
        """Get the name of the primary (first selected) layer."""
        if self.parent_widget is None:
            return None
        return self.parent_widget.get_primary_layer_name()

    def _find_phasors_layer_by_name(self, layer_name):
        """Find a phasors layer by name in the viewer."""
        for layer in self.viewer.layers:
            if layer.name == layer_name:
                return layer
        return None

    def _on_show_color_overlay(self, visible: bool):
        """Slot to show/hide the current phasors_selected_layer(s)."""
        if self.selection_id is None or self.selection_id == "":
            return

        selected_layers = self._get_selected_layers()
        for layer in selected_layers:
            selection_layer_name = f"{self.selection_id}: {layer.name}"
            selection_layer = self._find_phasors_layer_by_name(
                selection_layer_name
            )
            if selection_layer is not None:
                selection_layer.visible = visible

    def _connect_show_overlay_signal(self):
        """Ensure show_color_overlay_signal is connected only to the current layer's visibility."""
        with contextlib.suppress(TypeError, RuntimeError):
            self.parent_widget.canvas_widget.show_color_overlay_signal.disconnect(
                self._on_show_color_overlay
            )
        self.parent_widget.canvas_widget.show_color_overlay_signal.connect(
            self._on_show_color_overlay
        )

    def _on_refresh_selection_clicked(self):
        """Callback when refresh button is clicked."""
        self.update_phasors_layer()

    def on_selection_id_changed(self):
        """Callback function when the selection id combobox is changed."""
        raw_combobox_text = (
            self.selection_input_widget.phasor_selection_id_combobox.currentText()
        )

        if raw_combobox_text == "":
            self.selection_id = ""

        new_selection_id = self.selection_id
        new_selection_id_for_comparison = (
            "None" if new_selection_id is None else new_selection_id
        )

        if self._current_selection_id != new_selection_id_for_comparison:
            # Set flag to prevent manual_selection_changed from firing
            self._switching_selection_id = True

            self._current_selection_id = new_selection_id_for_comparison

            self._connect_show_overlay_signal()

            processed_selection_id = new_selection_id

            if not getattr(self, '_processing_initial_selection', False):
                self.update_phasor_plot_with_selection_id(
                    processed_selection_id
                )

            self._switching_selection_id = False

    def _on_image_layer_changed(self):
        """Callback when the image layer changes - restores cursors from metadata."""
        self.cursor_selection_widget._on_image_layer_changed()

    def update_phasor_plot_with_selection_id(self, selection_id):
        """Update the phasor plot with the selected ID and show/hide label layers."""
        selected_layers = self._get_selected_layers()
        if not selected_layers:
            return

        # Prevent this from running during plot updates
        if getattr(self.parent_widget, '_updating_plot', False):
            return

        if selection_id is None or selection_id == "":
            for layer in selected_layers:
                for i in range(
                    self.selection_input_widget.phasor_selection_id_combobox.count()
                ):
                    sel_id = self.selection_input_widget.phasor_selection_id_combobox.itemText(
                        i
                    )
                    if sel_id != "None":
                        selection_layer_name = f"{sel_id}: {layer.name}"
                        existing_layer = self._find_phasors_layer_by_name(
                            selection_layer_name
                        )
                        if existing_layer is not None:
                            existing_layer.visible = False

            self.parent_widget.plot(selection_id_data=None)
            return

        # Hide other selections for all layers
        for layer in selected_layers:
            for i in range(
                self.selection_input_widget.phasor_selection_id_combobox.count()
            ):
                sel_id = self.selection_input_widget.phasor_selection_id_combobox.itemText(
                    i
                )
                if sel_id != "None" and sel_id != selection_id:
                    other_layer_name = f"{sel_id}: {layer.name}"
                    other_layer = self._find_phasors_layer_by_name(
                        other_layer_name
                    )
                    if other_layer is not None:
                        other_layer.visible = False

        # Show current selection for all layers
        need_to_create = False
        for layer in selected_layers:
            selection_layer_name = f"{selection_id}: {layer.name}"
            selection_layer = self._find_phasors_layer_by_name(
                selection_layer_name
            )
            if selection_layer is None:
                need_to_create = True
            else:
                selection_layer.visible = True

        if need_to_create:
            self.create_phasors_selected_layer()

        # Collect selection data from all selected layers for the phasor plot
        all_selection_data = []
        for layer in selected_layers:
            if (
                "settings" in layer.metadata
                and "selections" in layer.metadata["settings"]
                and "manual_selections"
                in layer.metadata["settings"]["selections"]
                and selection_id
                in layer.metadata["settings"]["selections"][
                    "manual_selections"
                ]
            ):
                selection_map = layer.metadata["settings"]["selections"][
                    "manual_selections"
                ][selection_id]
            else:
                spatial_shape = layer.data.shape
                selection_map = np.zeros(spatial_shape, dtype=np.uint32)

            # Get valid pixels for this layer
            g_array = layer.metadata.get('G')
            s_array = layer.metadata.get('S')
            harmonics_array = layer.metadata.get('harmonics')

            if g_array is not None and s_array is not None:
                # Extract correct harmonic if arrays are 3D
                if harmonics_array is not None:
                    harmonics_array = np.atleast_1d(harmonics_array)
                    target_harmonic = self.parent_widget.harmonic
                    try:
                        harmonic_idx = int(
                            np.where(harmonics_array == target_harmonic)[0][0]
                        )
                    except (IndexError, ValueError):
                        continue
                else:
                    harmonic_idx = 0

                if g_array.ndim > layer.data.ndim:
                    g = g_array[harmonic_idx]
                    s = s_array[harmonic_idx]
                else:
                    g = g_array
                    s = s_array

                valid = np.isfinite(g.ravel()) & np.isfinite(s.ravel())
                selection_data = selection_map.ravel()[valid]
                all_selection_data.append(selection_data)

        if all_selection_data:
            # Concatenate all selection data from all layers
            combined_selection_data = np.concatenate(all_selection_data)
            self.parent_widget.plot(selection_id_data=combined_selection_data)
        else:
            self.parent_widget.plot(selection_id_data=None)

    def _get_next_available_selection_id(self):
        """Get the next available manual selection ID."""
        combobox_selections = [
            self.selection_input_widget.phasor_selection_id_combobox.itemText(
                i
            )
            for i in range(
                self.selection_input_widget.phasor_selection_id_combobox.count()
            )
        ]

        # Use primary layer for checking used selections
        selected_layers = self._get_selected_layers()
        used_selections = set()
        if selected_layers:
            primary_layer = selected_layers[0]
            if (
                "settings" in primary_layer.metadata
                and "selections" in primary_layer.metadata["settings"]
                and "manual_selections"
                in primary_layer.metadata["settings"]["selections"]
            ):
                used_selections = set(
                    primary_layer.metadata["settings"]["selections"][
                        "manual_selections"
                    ].keys()
                )

        counter = 1
        while True:
            candidate_name = f"MANUAL SELECTION #{counter}"
            if (
                candidate_name in combobox_selections
                and candidate_name not in used_selections
            ) or candidate_name not in combobox_selections:
                return candidate_name
            counter += 1

    def manual_selection_changed(self, manual_selection):
        """Update the manual selection in the layer metadata."""
        selected_layers = self._get_selected_layers()
        if not selected_layers:
            return

        if getattr(self.parent_widget, '_updating_plot', False):
            return

        if getattr(self, '_switching_selection_id', False):
            return

        current_combobox_text = (
            self.selection_input_widget.phasor_selection_id_combobox.currentText()
        )

        if current_combobox_text == "None":
            if manual_selection is None or not np.any(manual_selection):
                return

            new_selection_id = self._get_next_available_selection_id()

            self._processing_initial_selection = True
            self._initial_manual_selection = manual_selection

            self._current_selection_id = new_selection_id
            self.selection_id = new_selection_id

        selection_to_use = manual_selection
        if (
            hasattr(self, '_processing_initial_selection')
            and self._processing_initial_selection
        ):
            selection_to_use = self._initial_manual_selection
            self._processing_initial_selection = False
            delattr(self, '_initial_manual_selection')

        # The manual_selection array corresponds to merged/concatenated data from all layers
        # We need to split it back to individual layers based on valid pixel counts
        if selection_to_use is not None:
            # Calculate how many valid pixels each layer contributes
            layer_valid_counts = []
            for layer in selected_layers:
                g_array = layer.metadata.get('G')
                s_array = layer.metadata.get('S')
                harmonics_array = layer.metadata.get('harmonics')

                if g_array is not None and s_array is not None:
                    # Extract correct harmonic if arrays are 3D
                    if harmonics_array is not None:
                        harmonics_array = np.atleast_1d(harmonics_array)
                        target_harmonic = self.parent_widget.harmonic
                        try:
                            harmonic_idx = int(
                                np.where(harmonics_array == target_harmonic)[
                                    0
                                ][0]
                            )
                        except (IndexError, ValueError):
                            layer_valid_counts.append(0)
                            continue
                    else:
                        harmonic_idx = 0

                    if g_array.ndim > layer.data.ndim:
                        g = g_array[harmonic_idx]
                        s = s_array[harmonic_idx]
                    else:
                        g = g_array
                        s = s_array

                    valid = np.isfinite(g.ravel()) & np.isfinite(s.ravel())
                    layer_valid_counts.append(np.sum(valid))
                else:
                    layer_valid_counts.append(0)

            # Split the selection array based on valid counts
            selection_splits = []
            start_idx = 0
            for count in layer_valid_counts:
                if count > 0:
                    selection_splits.append(
                        selection_to_use[start_idx : start_idx + count]
                    )
                    start_idx += count
                else:
                    selection_splits.append(None)
        else:
            selection_splits = [None] * len(selected_layers)

        # Apply selection to all selected layers
        for layer, layer_selection in zip(
            selected_layers, selection_splits, strict=False
        ):
            if (
                "settings" in layer.metadata
                and "selections" in layer.metadata["settings"]
                and "manual_selections"
                in layer.metadata["settings"]["selections"]
                and self.selection_id
                in layer.metadata["settings"]["selections"][
                    "manual_selections"
                ]
            ):
                selection_map = layer.metadata["settings"]["selections"][
                    "manual_selections"
                ][self.selection_id].copy()
            else:
                # Get spatial shape for this specific layer
                spatial_shape = layer.data.shape
                selection_map = np.zeros(spatial_shape, dtype=np.uint32)

            selection_map_flat = selection_map.ravel()

            # Get valid pixels mask for this specific layer
            g_array = layer.metadata.get('G')
            s_array = layer.metadata.get('S')
            harmonics_array = layer.metadata.get('harmonics')

            if g_array is None or s_array is None:
                continue

            # Extract correct harmonic if arrays are 3D
            if harmonics_array is not None:
                harmonics_array = np.atleast_1d(harmonics_array)
                target_harmonic = self.parent_widget.harmonic
                try:
                    harmonic_idx = int(
                        np.where(harmonics_array == target_harmonic)[0][0]
                    )
                except (IndexError, ValueError):
                    continue
            else:
                harmonic_idx = 0

            if g_array.ndim > layer.data.ndim:
                g = g_array[harmonic_idx]
                s = s_array[harmonic_idx]
            else:
                g = g_array
                s = s_array

            valid_pixels_mask = np.isfinite(g.ravel()) & np.isfinite(s.ravel())

            if layer_selection is None:
                selection_map_flat[valid_pixels_mask] = 0
            else:
                selection_map_flat[valid_pixels_mask] = layer_selection

            if "settings" not in layer.metadata:
                layer.metadata["settings"] = {}
            if "selections" not in layer.metadata["settings"]:
                layer.metadata["settings"]["selections"] = {}
            if (
                "manual_selections"
                not in layer.metadata["settings"]["selections"]
            ):
                layer.metadata["settings"]["selections"][
                    "manual_selections"
                ] = {}

            layer.metadata["settings"]["selections"]["manual_selections"][
                self.selection_id
            ] = selection_map.copy()

        self.update_phasors_layer()

    def create_phasors_selected_layer(self):
        """Create the phasors selected layer for all selected layers."""
        selected_layers = self._get_selected_layers()
        if not selected_layers:
            return
        if self.selection_id is None or self.selection_id == "":
            return

        color_dict = colormap_to_dict(
            self.parent_widget._colormap,
            self.parent_widget._colormap.N,
            exclude_first=True,
        )

        # Create selection layer for each selected layer
        for layer in selected_layers:
            spatial_shape = layer.data.shape

            # Get selection map from metadata if it exists, otherwise create empty
            if (
                "settings" in layer.metadata
                and "selections" in layer.metadata["settings"]
                and "manual_selections"
                in layer.metadata["settings"]["selections"]
                and self.selection_id
                in layer.metadata["settings"]["selections"][
                    "manual_selections"
                ]
            ):
                selection_map = layer.metadata["settings"]["selections"][
                    "manual_selections"
                ][self.selection_id].copy()
            else:
                selection_map = np.zeros(spatial_shape, dtype=np.uint32)

            layer_name = f"{self.selection_id}: {layer.name}"

            # Check if layer already exists, skip if it does
            existing_layer = self._find_phasors_layer_by_name(layer_name)
            if existing_layer is not None:
                continue

            phasors_selected_layer = Labels(
                selection_map,
                name=layer_name,
                scale=layer.scale,
                colormap=DirectLabelColormap(
                    color_dict=color_dict, name="cat10_mod"
                ),
                metadata={
                    'napari_phasors_selection_type': 'manual',
                    'napari_phasors_source_layer': layer.name,
                },
            )

            self.viewer.add_layer(phasors_selected_layer)

        self._connect_show_overlay_signal()

    def update_phasors_layer(self):
        """Update the existing phasors layer data without recreating it."""
        selected_layers = self._get_selected_layers()
        if not selected_layers:
            return

        # Check if any layers need to be created
        need_creation = False
        for layer in selected_layers:
            selection_layer_name = f"{self.selection_id}: {layer.name}"
            if self._find_phasors_layer_by_name(selection_layer_name) is None:
                need_creation = True
                break

        # Create layers if needed
        if need_creation:
            self.create_phasors_selected_layer()

        # Update layer for each selected layer
        for layer in selected_layers:
            selection_layer_name = f"{self.selection_id}: {layer.name}"
            existing_phasors_selected_layer = self._find_phasors_layer_by_name(
                selection_layer_name
            )

            if existing_phasors_selected_layer is None:
                continue

            if self.selection_id is None or self.selection_id == "":
                existing_phasors_selected_layer.data = np.zeros_like(
                    existing_phasors_selected_layer.data
                )
            else:
                # Update the layer with the selection map from metadata
                if (
                    "settings" in layer.metadata
                    and "selections" in layer.metadata["settings"]
                    and "manual_selections"
                    in layer.metadata["settings"]["selections"]
                    and self.selection_id
                    in layer.metadata["settings"]["selections"][
                        "manual_selections"
                    ]
                ):
                    selection_map = layer.metadata["settings"]["selections"][
                        "manual_selections"
                    ][self.selection_id]
                    existing_phasors_selected_layer.data = selection_map

    def _recreate_manual_selection_layer(self, selection_id, selection_map):
        """Recreate a manual selection labels layer from metadata."""
        layer = self._get_current_layer()
        if layer is None:
            return

        layer_name = f"{selection_id}: {layer.name}"

        if self._find_phasors_layer_by_name(layer_name):
            return

        color_dict = colormap_to_dict(
            self.parent_widget._colormap,
            self.parent_widget._colormap.N,
            exclude_first=True,
        )

        phasors_selected_layer = Labels(
            selection_map,
            name=layer_name,
            scale=layer.scale,
            colormap=DirectLabelColormap(
                color_dict=color_dict, name="cat10_mod"
            ),
            visible=False,
            metadata={
                'napari_phasors_selection_type': 'manual',
                'napari_phasors_source_layer': layer.name,
            },
        )

        self.viewer.add_layer(phasors_selected_layer)


class AutomaticClusteringWidget(QWidget):
    """
    Widget for automatic clustering selection in phasor plots.

    This widget provides controls for automatic clustering using
    Gaussian Mixture Models (GMM) from phasorpy.

    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer instance.
    parent_widget : QWidget
        The parent PlotterWidget.
    """

    DEFAULT_COLORS = [
        QColor(int(r * 255), int(g * 255), int(b * 255))
        for r, g, b, _ in [plt.get_cmap('Set1')(i) for i in range(9)]
    ]

    def __init__(self, viewer, parent_widget):
        """Initialize the AutomaticClusteringWidget."""
        super().__init__()
        self.viewer = viewer
        self.parent_widget = parent_widget

        # Store cluster data: list of dicts with cluster info (one per cluster)
        self._clusters = []
        self._ellipse_patches = []
        self._phasors_selected_layer = None
        # Store label layers per image layer
        self._label_layers = {}  # {image_layer_name: labels_layer}

        self._setup_ui()

    def _setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 5, 0, 0)

        # Clustering method selection
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Clustering Method:"))
        self.clustering_method_combobox = QComboBox()
        self.clustering_method_combobox.addItems(
            ["GMM (Gaussian Mixture Model)"]
        )
        method_layout.addWidget(self.clustering_method_combobox, 1)
        layout.addLayout(method_layout)

        # Number of clusters selection
        clusters_layout = QHBoxLayout()
        clusters_layout.addWidget(QLabel("Number of Clusters:"))
        self.num_clusters_spinbox = QSpinBox()
        self.num_clusters_spinbox.setRange(2, 100)
        self.num_clusters_spinbox.setValue(2)
        clusters_layout.addWidget(self.num_clusters_spinbox, 1)
        layout.addLayout(clusters_layout)

        # Apply clustering button
        self.apply_button = QPushButton("Apply Clustering")
        self.apply_button.clicked.connect(self._apply_clustering)
        layout.addWidget(self.apply_button)

        # Table for clusters
        self.cluster_table = QTableWidget()
        self.cluster_table.setColumnCount(8)
        self.cluster_table.setHorizontalHeaderLabels(
            [
                "G",
                "S",
                "Major Radius",
                "Minor Radius",
                "Color",
                "Count",
                "%",
                "",
            ]
        )
        self.cluster_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )
        self.cluster_table.horizontalHeader().setSectionResizeMode(
            4, QHeaderView.Fixed
        )
        self.cluster_table.horizontalHeader().setSectionResizeMode(
            5, QHeaderView.Fixed
        )
        self.cluster_table.horizontalHeader().setSectionResizeMode(
            6, QHeaderView.Fixed
        )
        self.cluster_table.horizontalHeader().setSectionResizeMode(
            7, QHeaderView.Fixed
        )
        self.cluster_table.setColumnWidth(4, 40)
        self.cluster_table.setColumnWidth(5, 70)
        self.cluster_table.setColumnWidth(6, 60)
        self.cluster_table.setColumnWidth(7, 40)
        self.cluster_table.verticalHeader().setVisible(False)
        layout.addWidget(self.cluster_table)

        # Clear button
        self.clear_button = QPushButton("Clear Clusters")
        self.clear_button.setEnabled(False)
        self.clear_button.clicked.connect(self._clear_clusters)
        layout.addWidget(self.clear_button)

        layout.addStretch()

    def _apply_clustering(self):
        """Apply automatic clustering using the selected method."""
        if self.parent_widget is None:
            return

        selected_layers = self._get_selected_layers()
        if not selected_layers:
            return

        # Clear previous clusters
        self._clear_clusters(clear_patches_only=True)

        n_clusters = self.num_clusters_spinbox.value()

        # Step 1: Collect and merge g, s data from all selected layers
        g_list = []
        s_list = []
        layer_data = []  # Store layer info for later use

        for layer in selected_layers:
            g_array = layer.metadata.get('G')
            s_array = layer.metadata.get('S')

            if g_array is None or s_array is None:
                continue

            # Extract correct harmonic if arrays have extra dimension
            if g_array.ndim > layer.data.ndim:
                harmonic = self.parent_widget.harmonic
                # Harmonic numbering starts at 1, but array indexing starts at 0
                g = g_array[harmonic - 1]
                s = s_array[harmonic - 1]
            else:
                g = g_array
                s = s_array

            spatial_shape = layer.data.shape

            # Collect data for merging
            g_list.append(g.ravel())
            s_list.append(s.ravel())
            layer_data.append(
                {
                    'layer': layer,
                    'g': g,
                    's': s,
                    'spatial_shape': spatial_shape,
                }
            )

        if not g_list:
            return

        # Step 2: Merge all g, s data from all layers
        g_merged = np.concatenate(g_list)
        s_merged = np.concatenate(s_list)

        # Step 3: Perform GMM clustering on merged data
        try:
            center_real, center_imag, radius, radius_minor, angle = (
                phasor_cluster_gmm(
                    g_merged,
                    s_merged,
                    clusters=n_clusters,
                )
            )

            # Draw ellipses for the clusters (only once, not per layer)
            self._draw_cluster_ellipses(
                center_real,
                center_imag,
                radius,
                radius_minor,
                angle,
                self.parent_widget.harmonic,
            )

            # Store individual cluster information
            self._clusters.clear()
            for i in range(n_clusters):
                color_idx = i % len(self.DEFAULT_COLORS)
                cluster_data = {
                    'g': center_real[i],
                    's': center_imag[i],
                    'radius': radius[i],
                    'radius_minor': radius_minor[i],
                    'angle': angle[i],
                    'color': self.DEFAULT_COLORS[color_idx],
                    'harmonic': self.parent_widget.harmonic,
                }
                self._clusters.append(cluster_data)

            # Populate the table with cluster information
            self._populate_cluster_table()

            # Step 4: Apply the same cluster parameters to each layer
            for layer_info in layer_data:
                layer = layer_info['layer']
                g = layer_info['g']
                s = layer_info['s']
                spatial_shape = layer_info['spatial_shape']

                # Create selection map using elliptic cursor masks
                selection_map = np.zeros(spatial_shape, dtype=np.uint32)

                # Apply each cluster using elliptic cursor
                for idx, cluster in enumerate(self._clusters):
                    mask = mask_from_elliptic_cursor(
                        g,
                        s,
                        cluster['g'],
                        cluster['s'],
                        radius=cluster['radius'],
                        radius_minor=cluster['radius_minor'],
                        angle=cluster['angle'],
                    )
                    selection_map[mask] = idx + 1

                # Create labels layer
                self._create_or_update_labels_layer(layer, selection_map)

        except Exception as e:  # noqa: BLE001
            print(f"Error applying clustering: {e}")
            import traceback

            traceback.print_exc()

        # Enable clear button
        self.clear_button.setEnabled(True)

        # Update statistics
        self._update_cluster_statistics()

        # Redraw canvas
        if self.parent_widget is not None:
            self.parent_widget.canvas_widget.canvas.draw_idle()

    def _populate_cluster_table(self):
        """Populate the table with cluster information."""
        self.cluster_table.setRowCount(0)

        for cluster_idx, cluster in enumerate(self._clusters):
            table_row = self.cluster_table.rowCount()
            self.cluster_table.insertRow(table_row)

            # G label (read-only)
            g_label = QLabel(f"{cluster['g']:.3f}")
            g_label.setAlignment(Qt.AlignCenter)
            self.cluster_table.setCellWidget(table_row, 0, g_label)

            # S label (read-only)
            s_label = QLabel(f"{cluster['s']:.3f}")
            s_label.setAlignment(Qt.AlignCenter)
            self.cluster_table.setCellWidget(table_row, 1, s_label)

            # Major radius label (read-only)
            major_r_label = QLabel(f"{cluster['radius']:.3f}")
            major_r_label.setAlignment(Qt.AlignCenter)
            self.cluster_table.setCellWidget(table_row, 2, major_r_label)

            # Minor radius label (read-only)
            minor_r_label = QLabel(f"{cluster['radius_minor']:.3f}")
            minor_r_label.setAlignment(Qt.AlignCenter)
            self.cluster_table.setCellWidget(table_row, 3, minor_r_label)

            # Color button (editable)
            color_button = ColorButton(cluster['color'])
            color_button.color_changed.connect(
                lambda c, idx=cluster_idx: self._on_cluster_color_changed(
                    idx, c
                )
            )
            self.cluster_table.setCellWidget(table_row, 4, color_button)

            # Count label
            count_label = QLabel("-")
            count_label.setAlignment(Qt.AlignCenter)
            self.cluster_table.setCellWidget(table_row, 5, count_label)

            # Percentage label
            percentage_label = QLabel("-")
            percentage_label.setAlignment(Qt.AlignCenter)
            self.cluster_table.setCellWidget(table_row, 6, percentage_label)

            # Remove button
            remove_button = QPushButton("×")
            remove_button.setFixedSize(25, 25)
            remove_button.clicked.connect(
                lambda _, idx=cluster_idx: self._remove_cluster(idx)
            )
            self.cluster_table.setCellWidget(table_row, 7, remove_button)

    def _on_cluster_color_changed(self, cluster_idx, new_color):
        """Handle color change for a cluster."""
        if cluster_idx < 0 or cluster_idx >= len(self._clusters):
            return

        # Update cluster color
        self._clusters[cluster_idx]['color'] = new_color

        # Redraw ellipse with new color
        self._redraw_cluster_ellipse(cluster_idx)

        # Update all labels layers with new color
        self._update_all_labels_layer_colors()

        # Redraw canvas
        if self.parent_widget is not None:
            self.parent_widget.canvas_widget.canvas.draw_idle()

    def _redraw_cluster_ellipse(self, cluster_idx):
        """Redraw a specific cluster ellipse with updated color."""
        if cluster_idx < 0 or cluster_idx >= len(self._clusters):
            return
        if cluster_idx >= len(self._ellipse_patches):
            return

        cluster = self._clusters[cluster_idx]
        patch = self._ellipse_patches[cluster_idx]

        # Update patch color
        color = cluster['color']
        color_rgb = (color.redF(), color.greenF(), color.blueF())
        patch.set_edgecolor(color_rgb)

    def _remove_cluster(self, cluster_idx):
        """Remove a specific cluster."""
        if cluster_idx < 0 or cluster_idx >= len(self._clusters):
            return

        # Remove ellipse patch
        if cluster_idx < len(self._ellipse_patches):
            with contextlib.suppress(ValueError):
                self._ellipse_patches[cluster_idx].remove()
            self._ellipse_patches.pop(cluster_idx)

        # Remove cluster data
        self._clusters.pop(cluster_idx)

        # Rebuild table
        self._populate_cluster_table()

        # Reapply clustering to all layers with remaining clusters
        if self._clusters:
            self._reapply_clustering_to_layers()
            self._update_cluster_statistics()
        else:
            # If no clusters left, clear everything
            self._clear_all_labels_layers()
            self.cluster_table.setRowCount(0)
            self.clear_button.setEnabled(False)

        # Redraw canvas
        if self.parent_widget is not None:
            self.parent_widget.canvas_widget.canvas.draw_idle()

    def _reapply_clustering_to_layers(self):
        """Reapply clustering to all selected layers using current cluster parameters."""
        selected_layers = self._get_selected_layers()
        if not selected_layers:
            return

        current_harmonic = self.parent_widget.harmonic

        for layer in selected_layers:
            g_array = layer.metadata.get('G')
            s_array = layer.metadata.get('S')

            if g_array is None or s_array is None:
                continue

            # Extract correct harmonic if arrays are 3D
            if g_array.ndim > layer.data.ndim:
                harmonics_array = layer.metadata.get('harmonics')
                if harmonics_array is not None:
                    harmonics_array = np.atleast_1d(harmonics_array)
                    try:
                        harmonic_idx = int(
                            np.where(harmonics_array == current_harmonic)[0][0]
                        )
                    except (IndexError, ValueError):
                        continue
                else:
                    harmonic_idx = 0
                g = g_array[harmonic_idx]
                s = s_array[harmonic_idx]
            else:
                g = g_array
                s = s_array

            spatial_shape = layer.data.shape

            # Create selection map
            selection_map = np.zeros(spatial_shape, dtype=np.uint32)

            # Apply each cluster
            for idx, cluster in enumerate(self._clusters):
                if cluster.get('harmonic', 1) != current_harmonic:
                    continue
                mask = mask_from_elliptic_cursor(
                    g,
                    s,
                    cluster['g'],
                    cluster['s'],
                    radius=cluster['radius'],
                    radius_minor=cluster['radius_minor'],
                    angle=cluster['angle'],
                )
                selection_map[mask] = idx + 1

            # Update labels layer
            self._create_or_update_labels_layer(layer, selection_map)

    def _update_cluster_statistics(self):
        """Update the count and percentage columns in the cluster table."""
        if self.parent_widget is None:
            return

        selected_layers = self._get_selected_layers()
        if not selected_layers:
            return

        current_harmonic = self.parent_widget.harmonic

        # Calculate total valid pixels across all selected layers
        total_valid_pixels = 0
        for layer in selected_layers:
            g_array = layer.metadata.get('G')
            s_array = layer.metadata.get('S')
            harmonics_array = layer.metadata.get('harmonics')

            if g_array is None or s_array is None:
                continue

            # Extract correct harmonic if arrays are 3D
            if harmonics_array is not None:
                harmonics_array = np.atleast_1d(harmonics_array)
                target_harmonic = current_harmonic
                try:
                    harmonic_idx = int(
                        np.where(harmonics_array == target_harmonic)[0][0]
                    )
                except (IndexError, ValueError):
                    continue
            else:
                harmonic_idx = 0

            if g_array.ndim > layer.data.ndim:
                g = g_array[harmonic_idx]
                s = s_array[harmonic_idx]
            else:
                g = g_array
                s = s_array

            valid_pixels_mask = np.isfinite(g) & np.isfinite(s)
            total_valid_pixels += np.sum(valid_pixels_mask)

        if total_valid_pixels == 0:
            # If no valid pixels, clear statistics
            for table_row in range(self.cluster_table.rowCount()):
                count_label = self.cluster_table.cellWidget(table_row, 5)
                percentage_label = self.cluster_table.cellWidget(table_row, 6)
                if count_label:
                    count_label.setText("-")
                if percentage_label:
                    percentage_label.setText("-")
            return

        # Calculate statistics for each cluster
        cluster_pixel_counts = {}
        for cluster_idx, cluster in enumerate(self._clusters):
            if cluster.get('harmonic', 1) != current_harmonic:
                continue

            count = 0
            for layer in selected_layers:
                g_array = layer.metadata.get('G')
                s_array = layer.metadata.get('S')
                harmonics_array = layer.metadata.get('harmonics')

                if g_array is None or s_array is None:
                    continue

                # Extract correct harmonic if arrays are 3D
                if harmonics_array is not None:
                    harmonics_array = np.atleast_1d(harmonics_array)
                    target_harmonic = current_harmonic
                    try:
                        harmonic_idx = int(
                            np.where(harmonics_array == target_harmonic)[0][0]
                        )
                    except (IndexError, ValueError):
                        continue
                else:
                    harmonic_idx = 0

                if g_array.ndim > layer.data.ndim:
                    g = g_array[harmonic_idx]
                    s = s_array[harmonic_idx]
                else:
                    g = g_array
                    s = s_array

                # Calculate mask for this cluster
                mask = mask_from_elliptic_cursor(
                    g,
                    s,
                    cluster['g'],
                    cluster['s'],
                    radius=cluster['radius'],
                    radius_minor=cluster['radius_minor'],
                    angle=cluster['angle'],
                )

                count += np.sum(mask)

            cluster_pixel_counts[cluster_idx] = count

        # Update table with statistics
        for table_row in range(self.cluster_table.rowCount()):
            if table_row >= len(self._clusters):
                break

            count = cluster_pixel_counts.get(table_row, 0)
            percentage = (
                (count / total_valid_pixels * 100)
                if total_valid_pixels > 0
                else 0
            )

            count_label = self.cluster_table.cellWidget(table_row, 5)
            percentage_label = self.cluster_table.cellWidget(table_row, 6)

            if count_label:
                count_label.setText(str(count))
            if percentage_label:
                percentage_label.setText(f"{percentage:.1f}")

    def _draw_cluster_ellipses(
        self,
        center_real,
        center_imag,
        radius,
        radius_minor,
        angle,
        harmonic=None,
    ):
        """Draw ellipses representing the clusters on the phasor plot."""
        if self.parent_widget is None:
            return

        # Only draw if no harmonic specified or if it matches current harmonic
        if harmonic is not None and harmonic != self.parent_widget.harmonic:
            return

        ax = self.parent_widget.canvas_widget.axes

        n_clusters = len(center_real)
        for i in range(n_clusters):
            # Get color for this cluster
            color_idx = i % len(self.DEFAULT_COLORS)
            color = self.DEFAULT_COLORS[color_idx]
            color_rgb = (color.redF(), color.greenF(), color.blueF())

            # Use parameters directly from phasor_cluster_gmm
            # Width and height are 2 * radius (major and minor)
            width = 2 * radius[i]
            height = 2 * radius_minor[i]
            # Use angle directly from phasorpy
            angle_degrees = np.degrees(angle[i])

            # Create ellipse patch (non-pickable, so not draggable)
            ellipse = Ellipse(
                xy=(center_real[i], center_imag[i]),
                width=width,
                height=height,
                angle=angle_degrees,
                edgecolor=color_rgb,
                facecolor='none',
                linewidth=2,
                alpha=1,
                picker=False,  # Not pickable, so not draggable
                # transform=ax.transData
            )

            ax.add_patch(ellipse)
            self._ellipse_patches.append(ellipse)

    def _clear_clusters(self, clear_patches_only=False):
        """Clear all clusters and their visual representations."""
        # Remove ellipse patches from canvas
        for patch in self._ellipse_patches:
            with contextlib.suppress(ValueError):
                patch.remove()

        self._ellipse_patches.clear()

        if not clear_patches_only:
            # Clear table
            self.cluster_table.setRowCount(0)

            # Remove all labels layers
            self._clear_all_labels_layers()

            self._clusters.clear()
            self._label_layers.clear()
            self.clear_button.setEnabled(False)

        # Redraw canvas
        if self.parent_widget is not None:
            self.parent_widget.canvas_widget.canvas.draw_idle()

    def _clear_all_labels_layers(self):
        """Remove all cluster selection labels layers."""
        for _layer_name, labels_layer in list(self._label_layers.items()):
            try:
                if labels_layer in self.viewer.layers:
                    self.viewer.layers.remove(labels_layer)
            except (ValueError, KeyError):
                pass
        self._label_layers.clear()

    def clear_all_patches(self):
        """Clear all patches from the canvas (called when switching modes)."""
        for patch in self._ellipse_patches:
            with contextlib.suppress(ValueError):
                patch.remove()
        self._ellipse_patches.clear()

        if self.parent_widget is not None:
            self.parent_widget.canvas_widget.canvas.draw_idle()

    def redraw_all_patches(self):
        """Redraw all patches on the canvas (called when switching back to clustering mode)."""
        # Clear existing patches first
        self.clear_all_patches()

        # Redraw ellipses for all clusters
        if self.parent_widget is None:
            return

        ax = self.parent_widget.canvas_widget.axes
        current_harmonic = self.parent_widget.harmonic

        for cluster in self._clusters:
            # Only draw if harmonic matches
            if cluster.get('harmonic', 1) != current_harmonic:
                continue

            color = cluster['color']
            color_rgb = (color.redF(), color.greenF(), color.blueF())

            width = 2 * cluster['radius']
            height = 2 * cluster['radius_minor']
            angle_degrees = np.degrees(cluster['angle'])

            ellipse = Ellipse(
                xy=(cluster['g'], cluster['s']),
                width=width,
                height=height,
                angle=angle_degrees,
                edgecolor=color_rgb,
                facecolor='none',
                linewidth=2,
                alpha=1,
                picker=False,
            )

            ax.add_patch(ellipse)
            self._ellipse_patches.append(ellipse)

        if self.parent_widget is not None:
            self.parent_widget.canvas_widget.canvas.draw_idle()

    def on_harmonic_changed(self):
        """Called when the harmonic selection changes. Redraws cluster ellipses to show only those matching the current harmonic."""
        self.redraw_all_patches()
        if self._clusters:
            self._update_cluster_statistics()

    def _create_or_update_labels_layer(self, image_layer, selection_map):
        """Create or update the labels layer for the cluster selection."""
        layer_name = f"Cluster Selection: {image_layer.name}"

        color_dict = {None: (0, 0, 0, 0)}
        for idx, cluster in enumerate(self._clusters):
            color = cluster['color']
            color_dict[idx + 1] = (
                color.redF(),
                color.greenF(),
                color.blueF(),
                1.0,
            )

        existing_layer = self._label_layers.get(image_layer.name)

        if existing_layer is not None and existing_layer in self.viewer.layers:
            existing_layer.data = selection_map
            existing_layer.colormap = DirectLabelColormap(
                color_dict=color_dict, name="cluster_colors"
            )
            existing_layer.visible = True
        else:
            labels_layer = Labels(
                selection_map,
                name=layer_name,
                scale=image_layer.scale,
                colormap=DirectLabelColormap(
                    color_dict=color_dict, name="cluster_colors"
                ),
                metadata={
                    'napari_phasors_selection_type': 'automatic_clustering',
                    'napari_phasors_source_layer': image_layer.name,
                },
            )
            labels_layer = self.viewer.add_layer(labels_layer)
            self._label_layers[image_layer.name] = labels_layer

    def _update_all_labels_layer_colors(self):
        """Update colors in all labels layers after color changes."""
        color_dict = {None: (0, 0, 0, 0)}
        for idx, cluster in enumerate(self._clusters):
            color = cluster['color']
            color_dict[idx + 1] = (
                color.redF(),
                color.greenF(),
                color.blueF(),
                1.0,
            )

        for _layer_name, labels_layer in self._label_layers.items():
            if labels_layer in self.viewer.layers:
                labels_layer.colormap = DirectLabelColormap(
                    color_dict=color_dict, name="cluster_colors"
                )

    def _get_selected_layers(self):
        """Get all currently selected layers."""
        if self.parent_widget is None:
            return []
        return self.parent_widget.get_selected_layers()

    def _on_image_layer_changed(self):
        """Callback when image layer changes - clear clusters."""
        self._clear_clusters()


class ColorButton(QPushButton):
    """A button that displays a color and opens a color dialog when clicked."""

    color_changed = Signal(QColor)
    """Signal emitted with the new QColor when the color is changed."""

    def __init__(self, color=None, parent=None):
        """Initialize the ColorButton."""
        super().__init__(parent)
        self._color = color or QColor(255, 0, 0)
        self.setFixedSize(25, 25)
        self._update_style()
        self.clicked.connect(self._on_clicked)

    def _update_style(self):
        """Update the button style to show the current color."""
        self.setStyleSheet(
            f"background-color: {self._color.name()}; "
            f"border: 1px solid #555; border-radius: 3px;"
        )

    def _on_clicked(self):
        """Open a color dialog when clicked."""
        color = QColorDialog.getColor(self._color, self, "Select Cursor Color")
        if color.isValid():
            self._color = color
            self._update_style()
            self.color_changed.emit(color)

    def color(self):
        """Return the current color."""
        return self._color

    def set_color(self, color):
        """Set the current color."""
        self._color = color
        self._update_style()


class CursorSelectionWidget(QWidget):
    """
    Unified widget for cursor-based selection in phasor plots.

    Provides a single list of cursor "rows". The first element of each row
    is a shape combobox (Circular / Elliptical / Polar); the remaining fields
    in the row are shown or hidden dynamically based on the chosen shape. All
    cursors (regardless of shape) contribute to a single combined
    ``Cursor Selection: <image>`` labels layer.

    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer instance.
    parent_widget : QWidget
        The parent PlotterWidget.
    """

    DEFAULT_COLORS = [
        QColor(int(r * 255), int(g * 255), int(b * 255))
        for r, g, b, _ in [plt.get_cmap('Set1')(i) for i in range(9)]
    ]

    def __init__(self, viewer, parent_widget):
        """Initialize the CursorSelectionWidget."""
        super().__init__()
        self.viewer = viewer
        self.parent_widget = parent_widget

        # Each cursor is a dict carrying both its data and its row widgets.
        self._cursors = []
        self._phasors_selected_layer = None

        # Dragging state
        self._dragging_cursor = None
        self._drag_offset = (0, 0)
        self._drag_mode = None
        self._drag_start_angle = 0.0
        self._drag_start_cursor_angle = 0.0
        self._polar_edge = None

        # Autoupdate state (not stored in metadata)
        self._autoupdate_enabled = False
        # Whether a selection has been computed at least once. Visibility
        # toggles keep the selection in sync while this is True, even if the
        # selection layer was momentarily removed (e.g. all cursors hidden).
        self._selection_active = False

        self._setup_ui()
        self._connect_drag_events()

    # ------------------------------------------------------------------ UI
    def _setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 5, 0, 0)

        # Container holding one QFrame per cursor row.
        self._rows_container = QWidget()
        self._rows_layout = QVBoxLayout(self._rows_container)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._rows_container)

        # "Add Cursor" button, full width.
        self.add_cursor_button = QPushButton("+ Add Cursor")
        self.add_cursor_button.setToolTip(
            "Add a new cursor row (defaults to a circular cursor)."
        )
        self.add_cursor_button.clicked.connect(lambda: self._add_cursor())
        layout.addWidget(self.add_cursor_button)

        # Prominent "Calculate" button — the primary action of the tab.
        self.calculate_button = QPushButton("Calculate Selection")
        self.calculate_button.setToolTip(
            "Compute the selection from the current (visible) cursors."
        )
        # Keep it prominent (taller, bold) but the same color as the
        # "+ Add Cursor" button (the default button background).
        self.calculate_button.setMinimumHeight(34)
        self.calculate_button.setStyleSheet(
            "QPushButton { font-weight: bold; }"
        )
        self.calculate_button.clicked.connect(self._on_calculate_clicked)
        layout.addWidget(self.calculate_button)

        # Autoupdate toggle.
        self.autoupdate_checkbox = QWidget()
        autoupdate_layout = QHBoxLayout(self.autoupdate_checkbox)
        autoupdate_layout.setContentsMargins(0, 0, 0, 0)
        self.autoupdate_check = QToggleSwitch("Autoupdate")
        self.autoupdate_check.onColor = QColor("#27ae60")  # Nice Green
        self.autoupdate_check.setChecked(False)
        self.autoupdate_check.setToolTip(
            "Recompute the selection automatically whenever a cursor changes."
        )
        self.autoupdate_check.toggled.connect(self._on_autoupdate_changed)
        autoupdate_layout.addWidget(self.autoupdate_check)
        layout.addWidget(self.autoupdate_checkbox)

        layout.addStretch()

    def _get_next_color(self):
        """Get the next color from the palette based on current harmonic cursors."""
        if self.parent_widget is None:
            index = 0
        else:
            current_harmonic = self.parent_widget.harmonic
            current_harmonic_count = sum(
                1
                for c in self._cursors
                if c.get('harmonic', 1) == current_harmonic
            )
            index = current_harmonic_count % len(self.DEFAULT_COLORS)
        return self.DEFAULT_COLORS[index]

    def _get_last_radius(self):
        """Get the radius from the last cursor with one, or a default."""
        for cursor in reversed(self._cursors):
            if 'radius' in cursor and cursor['radius'] is not None:
                return cursor['radius']
        return 0.05

    def _axes_center(self):
        """Return the current axes center (g, s), or (0.5, 0.5) as fallback."""
        if (
            self.parent_widget is not None
            and hasattr(self.parent_widget, 'canvas_widget')
            and self.parent_widget.canvas_widget is not None
            and hasattr(self.parent_widget.canvas_widget, 'axes')
            and self.parent_widget.canvas_widget.axes is not None
        ):
            xlim = self.parent_widget.canvas_widget.axes.get_xlim()
            ylim = self.parent_widget.canvas_widget.axes.get_ylim()
            return (xlim[0] + xlim[1]) / 2.0, (ylim[0] + ylim[1]) / 2.0
        return 0.5, 0.5

    @staticmethod
    def _make_spinbox(low, high, value, decimals, step):
        spin = QDoubleSpinBox()
        spin.setRange(low, high)
        spin.setSingleStep(step)
        spin.setDecimals(decimals)
        spin.setValue(value)
        spin.setMaximumWidth(75)
        return spin

    # ------------------------------------------------------------- add row
    def _add_cursor(
        self,
        cursor_type="circular",
        g=None,
        s=None,
        radius=None,
        radius_minor=None,
        angle=None,
        phase_min=None,
        phase_max=None,
        modulation_min=None,
        modulation_max=None,
        color=None,
        visible=True,
    ):
        """Add a new cursor row.

        ``cursor_type`` is one of ``"circular"``, ``"elliptic"`` or
        ``"polar"``. Unspecified parameters are filled with sensible
        defaults derived from the current axes so switching shape later
        always has valid values.
        """
        # Tolerate being called as a Qt slot (clicked sends a bool).
        if isinstance(cursor_type, bool):
            cursor_type = "circular"

        center_g, center_s = self._axes_center()

        if g is None:
            g = center_g
        if s is None:
            s = center_s
        g = max(-1.5, min(1.5, g))
        s = max(-1.5, min(1.5, s))

        if radius is None:
            radius = (
                0.1 if cursor_type == "elliptic" else self._get_last_radius()
            )
        if radius_minor is None:
            radius_minor = 0.05
        if angle is None:
            angle = 0.0

        # Polar defaults from the center position.
        if (
            phase_min is None
            or phase_max is None
            or modulation_min is None
            or modulation_max is None
        ):
            center_phase_deg = np.rad2deg(np.arctan2(center_s, center_g))
            center_modulation = np.sqrt(center_g**2 + center_s**2)
            if phase_min is None:
                phase_min = center_phase_deg - 10.0
            if phase_max is None:
                phase_max = center_phase_deg + 10.0
            if modulation_min is None:
                modulation_min = max(0.0, center_modulation - 0.1)
            if modulation_max is None:
                modulation_max = min(1.0, center_modulation + 0.1)

        # Clamp/validate modulation range.
        modulation_min = max(0.0, min(1.0, modulation_min))
        modulation_max = max(0.0, min(1.0, modulation_max))
        if modulation_min > modulation_max:
            modulation_min, modulation_max = modulation_max, modulation_min
        if abs(modulation_max - modulation_min) < 1e-5:
            if abs(modulation_max - 1.0) < 1e-5:
                modulation_min = 0.99
            else:
                modulation_max = modulation_min + 0.01

        if color is None:
            color = self._get_next_color()

        cursor = {
            'type': cursor_type,
            'g': g,
            's': s,
            'radius': radius,
            'radius_minor': radius_minor,
            'angle': angle,
            'phase_min': phase_min,
            'phase_max': phase_max,
            'modulation_min': modulation_min,
            'modulation_max': modulation_max,
            'color': color,
            'patch': None,
            'visible': bool(visible),
            'harmonic': (
                self.parent_widget.harmonic if self.parent_widget else 1
            ),
        }

        self._build_row(cursor)
        self._cursors.append(cursor)

        self._update_row_visibility()
        self._update_cursor_patch(cursor)

        if self._autoupdate_enabled:
            self._apply_selection()
        else:
            self._update_cursor_statistics()

    def _build_row(self, cursor):
        """Construct the QFrame row and store widget refs on ``cursor``."""

        def _labeled(layout, text, widget, tooltip):
            """Add a ``text`` label + ``widget`` to ``layout``, sharing tooltip."""
            label = QLabel(text)
            label.setToolTip(tooltip)
            widget.setToolTip(tooltip)
            layout.addWidget(label)
            layout.addWidget(widget)

        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        row_layout = QHBoxLayout(frame)
        row_layout.setContentsMargins(4, 2, 4, 2)
        row_layout.setSpacing(4)

        number_label = QLabel("")

        type_combo = QComboBox()
        type_combo.addItem("Circular", "circular")
        type_combo.addItem("Elliptical", "elliptic")
        type_combo.addItem("Polar", "polar")
        type_combo.setCurrentIndex(type_combo.findData(cursor['type']))
        type_combo.setToolTip(
            "Cursor shape: a circular, elliptical or polar (wedge) region "
            "of the phasor plot."
        )

        color_button = ColorButton(cursor['color'])
        color_button.setToolTip(
            "Color of this cursor's region in the selection overlay."
        )

        # Center fields (circular + elliptic).
        center_widget = QWidget()
        center_line = QHBoxLayout(center_widget)
        center_line.setContentsMargins(0, 0, 0, 0)
        center_line.setSpacing(2)
        g_spin = self._make_spinbox(-1.5, 1.5, cursor['g'], 2, 0.01)
        s_spin = self._make_spinbox(-1.5, 1.5, cursor['s'], 2, 0.01)
        radius_spin = self._make_spinbox(0.001, 1.0, cursor['radius'], 3, 0.01)
        _labeled(
            center_line,
            "G",
            g_spin,
            "G coordinate (horizontal axis) of the cursor center.",
        )
        _labeled(
            center_line,
            "S",
            s_spin,
            "S coordinate (vertical axis) of the cursor center.",
        )
        _labeled(
            center_line,
            "r",
            radius_spin,
            "Radius of the circular cursor, or the major-axis radius of the "
            "elliptical cursor.",
        )

        # Elliptic-only fields.
        elliptic_widget = QWidget()
        elliptic_line = QHBoxLayout(elliptic_widget)
        elliptic_line.setContentsMargins(0, 0, 0, 0)
        elliptic_line.setSpacing(2)
        radius_minor_spin = self._make_spinbox(
            0.001, 1.0, cursor['radius_minor'], 3, 0.01
        )
        angle_spin = self._make_spinbox(-360.0, 360.0, cursor['angle'], 1, 1.0)
        _labeled(
            elliptic_line,
            "rₘ",
            radius_minor_spin,
            "Minor-axis radius of the elliptical cursor.",
        )
        _labeled(
            elliptic_line,
            "∠",
            angle_spin,
            "Rotation angle of the elliptical cursor, in degrees.",
        )

        # Polar-only fields.
        polar_widget = QWidget()
        polar_line = QHBoxLayout(polar_widget)
        polar_line.setContentsMargins(0, 0, 0, 0)
        polar_line.setSpacing(2)
        phase_min_spin = self._make_spinbox(
            -360.0, 360.0, cursor['phase_min'], 1, 1.0
        )
        phase_max_spin = self._make_spinbox(
            -360.0, 360.0, cursor['phase_max'], 1, 1.0
        )
        mod_min_spin = self._make_spinbox(
            0.0, 1.0, cursor['modulation_min'], 2, 0.01
        )
        mod_max_spin = self._make_spinbox(
            0.0, 1.0, cursor['modulation_max'], 2, 0.01
        )
        _labeled(
            polar_line,
            "φ₋",
            phase_min_spin,
            "Lower phase bound of the polar cursor, in degrees.",
        )
        _labeled(
            polar_line,
            "φ₊",
            phase_max_spin,
            "Upper phase bound of the polar cursor, in degrees.",
        )
        _labeled(
            polar_line,
            "m₋",
            mod_min_spin,
            "Lower modulation (radial distance) bound of the polar cursor.",
        )
        _labeled(
            polar_line,
            "m₊",
            mod_max_spin,
            "Upper modulation (radial distance) bound of the polar cursor.",
        )

        count_label = QLabel("-")
        count_label.setAlignment(Qt.AlignCenter)
        count_label.setMinimumWidth(45)
        count_label.setToolTip("Number of pixels inside this cursor's region.")
        percentage_label = QLabel("-")
        percentage_label.setAlignment(Qt.AlignCenter)
        percentage_label.setMinimumWidth(40)
        percentage_label.setToolTip(
            "Percentage of valid pixels inside this cursor's region."
        )

        n_label = QLabel("n:")
        n_label.setToolTip("Number of pixels inside this cursor's region.")
        pct_label = QLabel("%:")
        pct_label.setToolTip(
            "Percentage of valid pixels inside this cursor's region."
        )

        # A plain (non-checkable) button so its background always matches the
        # neighbouring "×" remove button; the icon conveys the visible state.
        visibility_button = QPushButton()
        visibility_button.setFixedSize(25, 25)
        visibility_button.setIconSize(QSize(18, 18))

        remove_button = QPushButton("×")
        remove_button.setFixedSize(25, 25)
        remove_button.setToolTip("Remove this cursor.")

        row_layout.addWidget(number_label)
        row_layout.addWidget(type_combo)
        row_layout.addWidget(color_button)
        row_layout.addWidget(center_widget)
        row_layout.addWidget(elliptic_widget)
        row_layout.addWidget(polar_widget)
        row_layout.addStretch()
        row_layout.addWidget(n_label)
        row_layout.addWidget(count_label)
        row_layout.addWidget(pct_label)
        row_layout.addWidget(percentage_label)
        row_layout.addWidget(visibility_button)
        row_layout.addWidget(remove_button)

        cursor.update(
            {
                'row': frame,
                'number_label': number_label,
                'type_combo': type_combo,
                'color_button': color_button,
                'center_widget': center_widget,
                'g_spin': g_spin,
                's_spin': s_spin,
                'radius_spin': radius_spin,
                'elliptic_widget': elliptic_widget,
                'radius_minor_spin': radius_minor_spin,
                'angle_spin': angle_spin,
                'polar_widget': polar_widget,
                'phase_min_spin': phase_min_spin,
                'phase_max_spin': phase_max_spin,
                'mod_min_spin': mod_min_spin,
                'mod_max_spin': mod_max_spin,
                'count_label': count_label,
                'percentage_label': percentage_label,
                'visibility_button': visibility_button,
                'remove_button': remove_button,
            }
        )

        self._rows_layout.addWidget(frame)
        self._apply_type_visibility(cursor)
        self._update_visibility_button(cursor)

        # Wire signals (lambdas capture the cursor dict directly).
        type_combo.currentIndexChanged.connect(
            lambda _=0, c=cursor: self._on_cursor_type_changed(c)
        )
        for spin in (
            g_spin,
            s_spin,
            radius_spin,
            radius_minor_spin,
            angle_spin,
            phase_min_spin,
            phase_max_spin,
            mod_min_spin,
            mod_max_spin,
        ):
            spin.valueChanged.connect(
                lambda _val, c=cursor: self._on_cursor_changed(c)
            )
        color_button.color_changed.connect(
            lambda _c, c=cursor: self._on_cursor_changed(c)
        )
        visibility_button.clicked.connect(
            lambda _=False, c=cursor: self._on_cursor_visibility_toggled(c)
        )
        remove_button.clicked.connect(
            lambda _=False, c=cursor: self._remove_cursor(c)
        )

    # The eye outline and slash are always drawn white; visibility is
    # conveyed by the slash, not by colour.
    EYE_COLOR = "white"

    def _update_visibility_button(self, cursor):
        """Update the eye icon and tooltip from the ``visible`` state."""
        button = cursor['visibility_button']
        if cursor['visible']:
            button.setToolTip(
                "Cursor is shown and included in the selection. "
                "Click to hide it."
            )
        else:
            button.setToolTip(
                "Cursor is hidden and excluded from the selection. "
                "Click to show it."
            )
        crossed = not cursor['visible']
        button.setIcon(_make_eye_icon(self.EYE_COLOR, crossed=crossed))
        button.setProperty("eyeCrossed", crossed)

    def _on_cursor_visibility_toggled(self, cursor):
        """Toggle a cursor's visibility, recomputing the selection."""
        if cursor not in self._cursors:
            return
        cursor['visible'] = not cursor['visible']
        self._update_visibility_button(cursor)
        self._update_cursor_patch(cursor)
        # Recompute the selection so hidden cursors are excluded (and shown
        # cursors re-included) whenever a selection has been computed or
        # autoupdate is on.
        if self._autoupdate_enabled or self._selection_active:
            self._apply_selection()
        else:
            self._update_cursor_statistics()

    def _apply_type_visibility(self, cursor):
        """Show/hide the shape-specific field groups for a cursor row."""
        cursor_type = cursor['type']
        cursor['center_widget'].setVisible(cursor_type != "polar")
        cursor['elliptic_widget'].setVisible(cursor_type == "elliptic")
        cursor['polar_widget'].setVisible(cursor_type == "polar")

    def _resolve_cursor(self, cursor_or_idx):
        """Accept either a cursor dict or its index in ``self._cursors``."""
        if isinstance(cursor_or_idx, dict):
            return cursor_or_idx
        if 0 <= cursor_or_idx < len(self._cursors):
            return self._cursors[cursor_or_idx]
        return None

    # --------------------------------------------------------- row updates
    def _update_row_visibility(self):
        """Show only rows belonging to the current harmonic and renumber."""
        current_harmonic = (
            self.parent_widget.harmonic if self.parent_widget else 1
        )
        number = 1
        for cursor in self._cursors:
            visible = cursor.get('harmonic', 1) == current_harmonic
            cursor['row'].setVisible(visible)
            if visible:
                cursor['number_label'].setText(f"{number}.")
                number += 1

    def _current_harmonic_cursors(self):
        current_harmonic = (
            self.parent_widget.harmonic if self.parent_widget else 1
        )
        return [
            c
            for c in self._cursors
            if c.get('harmonic', 1) == current_harmonic
        ]

    def _on_cursor_type_changed(self, cursor):
        """Handle the shape combobox changing for a row."""
        cursor['type'] = cursor['type_combo'].currentData()
        self._apply_type_visibility(cursor)
        self._update_cursor_patch(cursor)
        if self._dragging_cursor is None:
            if self._autoupdate_enabled:
                self._apply_selection()
            else:
                self._update_cursor_statistics()

    def _sync_cursor_from_widgets(self, cursor):
        """Read all field values from the row widgets into the cursor data."""
        cursor['g'] = cursor['g_spin'].value()
        cursor['s'] = cursor['s_spin'].value()
        cursor['radius'] = cursor['radius_spin'].value()
        cursor['radius_minor'] = cursor['radius_minor_spin'].value()
        cursor['angle'] = cursor['angle_spin'].value()
        cursor['phase_min'] = cursor['phase_min_spin'].value()
        cursor['phase_max'] = cursor['phase_max_spin'].value()
        cursor['modulation_min'] = cursor['mod_min_spin'].value()
        cursor['modulation_max'] = cursor['mod_max_spin'].value()
        cursor['color'] = cursor['color_button'].color()

    def _on_cursor_changed(self, cursor):
        """Handle any field change for a cursor row."""
        if cursor not in self._cursors:
            return
        self._sync_cursor_from_widgets(cursor)
        self._update_cursor_patch(cursor)
        if self._dragging_cursor is None:
            if self._autoupdate_enabled:
                self._apply_selection()
            else:
                self._update_cursor_statistics()

    def _remove_cursor(self, cursor_or_idx):
        """Remove a cursor row."""
        cursor = self._resolve_cursor(cursor_or_idx)
        if cursor is None or cursor not in self._cursors:
            return

        if cursor.get('patch') is not None:
            with contextlib.suppress(ValueError):
                cursor['patch'].remove()
            cursor['patch'] = None

        cursor['row'].setParent(None)
        cursor['row'].deleteLater()
        self._cursors.remove(cursor)

        self._update_row_visibility()

        if not self._cursors:
            self._remove_selection_layer()
            self._selection_active = False
        elif self._autoupdate_enabled:
            self._apply_selection()
        else:
            self._update_cursor_statistics()

        if self.parent_widget is not None:
            self.parent_widget.canvas_widget.canvas.draw_idle()

    def _clear_all_cursors(self):
        """Clear all cursors."""
        for cursor in self._cursors:
            if cursor.get('patch') is not None:
                with contextlib.suppress(ValueError):
                    cursor['patch'].remove()
            cursor['row'].setParent(None)
            cursor['row'].deleteLater()
        self._cursors.clear()
        self._remove_selection_layer()
        self._selection_active = False
        if self.parent_widget is not None:
            self.parent_widget.canvas_widget.canvas.draw_idle()

    # ------------------------------------------------------------- patches
    def _update_cursor_patch(self, cursor_or_idx):
        """Update or create the matplotlib patch for a cursor."""
        cursor = self._resolve_cursor(cursor_or_idx)
        if cursor is None or self.parent_widget is None:
            return

        current_harmonic = self.parent_widget.harmonic
        cursor_harmonic = cursor.get('harmonic', 1)

        if cursor.get('patch') is not None:
            with contextlib.suppress(ValueError):
                cursor['patch'].remove()
            cursor['patch'] = None

        # Hidden cursors and cursors of a different harmonic draw no patch.
        if cursor_harmonic != current_harmonic or not cursor.get(
            'visible', True
        ):
            self.parent_widget.canvas_widget.canvas.draw_idle()
            return

        ax = self.parent_widget.canvas_widget.axes
        color = cursor['color']
        edge_rgba = (color.redF(), color.greenF(), color.blueF(), 1.0)

        if cursor['type'] == "circular":
            patch = Circle(
                (cursor['g'], cursor['s']),
                cursor['radius'],
                fill=False,
                edgecolor=edge_rgba,
                linewidth=2,
                zorder=10,
                picker=True,
            )
        elif cursor['type'] == "elliptic":
            patch = Ellipse(
                xy=(cursor['g'], cursor['s']),
                width=2 * cursor['radius'],
                height=2 * cursor['radius_minor'],
                angle=cursor['angle'],
                facecolor='none',
                edgecolor=edge_rgba,
                linewidth=2,
                zorder=10,
                picker=True,
            )
        else:  # polar
            r = cursor['modulation_max']
            width = cursor['modulation_max'] - cursor['modulation_min']
            if width <= 0:
                width = 0.001
            patch = Wedge(
                (0, 0),
                r,
                cursor['phase_min'],
                cursor['phase_max'],
                width=width,
                fill=False,
                edgecolor=edge_rgba,
                linewidth=2,
                zorder=10,
                picker=True,
            )

        cursor['patch'] = ax.add_patch(patch)
        self.parent_widget.canvas_widget.canvas.draw_idle()

    def clear_all_patches(self):
        """Clear all patches from the canvas (called when switching modes)."""
        for cursor in self._cursors:
            if cursor.get('patch') is not None:
                with contextlib.suppress(ValueError):
                    cursor['patch'].remove()
                cursor['patch'] = None
        if self.parent_widget is not None:
            self.parent_widget.canvas_widget.canvas.draw_idle()

    def redraw_all_patches(self):
        """Redraw all patches on the canvas (called when re-entering mode)."""
        for cursor in self._cursors:
            self._update_cursor_patch(cursor)

    def on_harmonic_changed(self):
        """Handle harmonic change - show rows/patches for current harmonic."""
        if self.parent_widget is None:
            return
        self._update_row_visibility()
        for cursor in self._cursors:
            self._update_cursor_patch(cursor)
        if self._autoupdate_enabled:
            self._apply_selection()
        else:
            self._update_cursor_statistics()

    # ----------------------------------------------------------- selection
    def _get_current_layer(self):
        """Helper to get the currently selected image layer."""
        if self.parent_widget is None:
            return None
        layer_name = (
            self.parent_widget.image_layer_with_phasor_features_combobox.currentText()
        )
        if not layer_name or layer_name not in self.viewer.layers:
            return None
        return self.viewer.layers[layer_name]

    def _get_selected_layers(self):
        """Get all currently selected layers."""
        if self.parent_widget is None:
            return []
        return self.parent_widget.get_selected_layers()

    def _on_calculate_clicked(self):
        """Handle Calculate button click."""
        if self._cursors:
            self._apply_selection()

    def _on_autoupdate_changed(self, checked):
        """Handle Autoupdate toggle state change."""
        self._autoupdate_enabled = self.autoupdate_check.isChecked()
        self.calculate_button.setEnabled(not self._autoupdate_enabled)
        if self._autoupdate_enabled and self._cursors:
            self._apply_selection()

    def _remove_selection_layer(self):
        """Remove the combined cursor selection layer if it exists."""
        selected_layers = self._get_selected_layers()
        if not selected_layers:
            return
        for layer in selected_layers:
            layer_name = f"Cursor Selection: {layer.name}"
            for viewer_layer in list(self.viewer.layers):
                if viewer_layer.name == layer_name:
                    self.viewer.layers.remove(viewer_layer)
                    break
        self._phasors_selected_layer = None

    @staticmethod
    def _layer_harmonic_arrays(layer, target_harmonic):
        """Return (g, s) arrays for ``target_harmonic`` or (None, None)."""
        g_array = layer.metadata.get('G')
        s_array = layer.metadata.get('S')
        if g_array is None or s_array is None:
            return None, None
        harmonics_array = layer.metadata.get('harmonics')
        if harmonics_array is not None:
            harmonics_array = np.atleast_1d(harmonics_array)
            try:
                harmonic_idx = int(
                    np.where(harmonics_array == target_harmonic)[0][0]
                )
            except (IndexError, ValueError):
                return None, None
        else:
            harmonic_idx = 0
        if g_array.ndim > layer.data.ndim:
            return g_array[harmonic_idx], s_array[harmonic_idx]
        return g_array, s_array

    @staticmethod
    def _cursor_mask(cursor, g, s):
        """Return the boolean mask of pixels inside a single cursor."""
        if cursor['type'] == "circular":
            return mask_from_circular_cursor(
                g, s, [cursor['g']], [cursor['s']], radius=[cursor['radius']]
            )[0]
        if cursor['type'] == "elliptic":
            return mask_from_elliptic_cursor(
                g,
                s,
                cursor['g'],
                cursor['s'],
                radius=cursor['radius'],
                radius_minor=cursor['radius_minor'],
                angle=np.deg2rad(cursor['angle']),
            )
        return mask_from_polar_cursor(
            g,
            s,
            np.deg2rad(cursor['phase_min']),
            np.deg2rad(cursor['phase_max']),
            cursor['modulation_min'],
            cursor['modulation_max'],
        )

    def _cursor_metadata_params(self, cursor):
        """Return the metadata dict (per-shape) for a cursor."""
        color = cursor['color']
        color_tuple = (
            color.red(),
            color.green(),
            color.blue(),
            color.alpha(),
        )
        visible = bool(cursor.get('visible', True))
        if cursor['type'] == "circular":
            return {
                'g': cursor['g'],
                's': cursor['s'],
                'radius': cursor['radius'],
                'color': color_tuple,
                'visible': visible,
            }
        if cursor['type'] == "elliptic":
            return {
                'g': cursor['g'],
                's': cursor['s'],
                'radius': cursor['radius'],
                'radius_minor': cursor['radius_minor'],
                'angle': cursor['angle'],
                'color': color_tuple,
                'visible': visible,
            }
        return {
            'phase_min': cursor['phase_min'],
            'phase_max': cursor['phase_max'],
            'modulation_min': cursor['modulation_min'],
            'modulation_max': cursor['modulation_max'],
            'color': color_tuple,
            'visible': visible,
        }

    def _apply_selection(self):
        """Apply all current-harmonic cursors to a single labels layer."""
        if not self._cursors or self.parent_widget is None:
            return

        selected_layers = self._get_selected_layers()
        if not selected_layers:
            return

        # A selection has now been computed; visibility toggles will keep it
        # in sync from here on.
        self._selection_active = True

        current_harmonic = self.parent_widget.harmonic
        current_harmonic_cursors = self._current_harmonic_cursors()

        # Partition cursors by shape for backward-compatible metadata.
        circular_params, elliptical_params, polar_params = [], [], []
        for cursor in current_harmonic_cursors:
            params = self._cursor_metadata_params(cursor)
            if cursor['type'] == "circular":
                circular_params.append(params)
            elif cursor['type'] == "elliptic":
                elliptical_params.append(params)
            else:
                polar_params.append(params)

        for layer in selected_layers:
            if "settings" not in layer.metadata:
                layer.metadata["settings"] = {}
            if "selections" not in layer.metadata["settings"]:
                layer.metadata["settings"]["selections"] = {}
            selections = layer.metadata["settings"]["selections"]
            selections["circular_cursors"] = circular_params
            selections["elliptical_cursors"] = elliptical_params
            selections["polar_cursors"] = polar_params

        # Only visible cursors contribute to the selection (and to the
        # label-id / color mapping); hidden cursors behave as if absent.
        visible_cursors = [
            c for c in current_harmonic_cursors if c.get('visible', True)
        ]

        if not visible_cursors:
            self._remove_selection_layer()
            self._update_cursor_statistics()
            return

        for layer in selected_layers:
            g, s = self._layer_harmonic_arrays(layer, current_harmonic)
            if g is None:
                continue
            selection_map = np.zeros(layer.data.shape, dtype=np.uint32)
            for idx, cursor in enumerate(visible_cursors):
                mask = self._cursor_mask(cursor, g, s)
                selection_map[mask] = idx + 1
            self._create_or_update_labels_layer(
                layer, selection_map, visible_cursors
            )

        self._update_cursor_statistics()

    def _update_cursor_statistics(self):
        """Update the count and percentage labels in each visible row."""
        if self.parent_widget is None:
            return
        selected_layers = self._get_selected_layers()
        if not selected_layers:
            return

        current_harmonic = self.parent_widget.harmonic
        current_harmonic_cursors = self._current_harmonic_cursors()

        total_valid_pixels = 0
        for layer in selected_layers:
            g, s = self._layer_harmonic_arrays(layer, current_harmonic)
            if g is None:
                continue
            total_valid_pixels += int(np.sum(np.isfinite(g) & np.isfinite(s)))

        if total_valid_pixels == 0:
            for cursor in current_harmonic_cursors:
                cursor['count_label'].setText("-")
                cursor['percentage_label'].setText("-")
            return

        for cursor in current_harmonic_cursors:
            # Hidden cursors are excluded from the selection, so report no
            # statistics for them.
            if not cursor.get('visible', True):
                cursor['count_label'].setText("-")
                cursor['percentage_label'].setText("-")
                continue
            count = 0
            for layer in selected_layers:
                g, s = self._layer_harmonic_arrays(layer, current_harmonic)
                if g is None:
                    continue
                count += int(np.sum(self._cursor_mask(cursor, g, s)))
            percentage = count / total_valid_pixels * 100
            cursor['count_label'].setText(str(count))
            cursor['percentage_label'].setText(f"{percentage:.1f}")

    def _create_or_update_labels_layer(
        self, image_layer, selection_map, cursors_list
    ):
        """Create or update the combined cursor selection labels layer."""
        layer_name = f"Cursor Selection: {image_layer.name}"

        color_dict = {None: (0, 0, 0, 0)}
        for idx, cursor in enumerate(cursors_list):
            color = cursor['color']
            color_dict[idx + 1] = (
                color.redF(),
                color.greenF(),
                color.blueF(),
                1.0,
            )

        existing_layer = None
        for viewer_layer in self.viewer.layers:
            if viewer_layer.name == layer_name:
                existing_layer = viewer_layer
                break

        if existing_layer is not None:
            existing_layer.data = selection_map
            existing_layer.colormap = DirectLabelColormap(
                color_dict=color_dict, name="cursor_selection_colors"
            )
            existing_layer.visible = True
            self._phasors_selected_layer = existing_layer
        else:
            labels_layer = Labels(
                selection_map,
                name=layer_name,
                scale=image_layer.scale,
                colormap=DirectLabelColormap(
                    color_dict=color_dict, name="cursor_selection_colors"
                ),
                metadata={
                    'napari_phasors_selection_type': 'cursor_selection',
                    'napari_phasors_source_layer': image_layer.name,
                },
            )
            self._phasors_selected_layer = self.viewer.add_layer(labels_layer)

    def _on_image_layer_changed(self):
        """Restore cursors from the new image layer's metadata."""
        for cursor in self._cursors:
            if cursor.get('patch') is not None:
                with contextlib.suppress(ValueError):
                    cursor['patch'].remove()
                cursor['patch'] = None
            cursor['row'].setParent(None)
            cursor['row'].deleteLater()
        self._cursors.clear()
        self._selection_active = False

        selected_layers = self._get_selected_layers()
        if not selected_layers:
            return
        primary_layer = selected_layers[0]
        selections = primary_layer.metadata.get("settings", {}).get(
            "selections", {}
        )

        # Suppress selection application while restoring rows.
        original_apply = self._apply_selection
        self._apply_selection = lambda *a, **k: None
        try:
            for params in selections.get("circular_cursors", []) or []:
                self._add_cursor(
                    cursor_type="circular",
                    g=params["g"],
                    s=params["s"],
                    radius=params["radius"],
                    color=QColor(*params["color"]),
                    visible=params.get("visible", True),
                )
            for params in selections.get("elliptical_cursors", []) or []:
                self._add_cursor(
                    cursor_type="elliptic",
                    g=params["g"],
                    s=params["s"],
                    radius=params["radius"],
                    radius_minor=params["radius_minor"],
                    angle=params["angle"],
                    color=QColor(*params["color"]),
                    visible=params.get("visible", True),
                )
            for params in selections.get("polar_cursors", []) or []:
                self._add_cursor(
                    cursor_type="polar",
                    phase_min=params["phase_min"],
                    phase_max=params["phase_max"],
                    modulation_min=params["modulation_min"],
                    modulation_max=params["modulation_max"],
                    color=QColor(*params["color"]),
                    visible=params.get("visible", True),
                )
        finally:
            self._apply_selection = original_apply

        if self.parent_widget is not None:
            self.parent_widget.canvas_widget.canvas.draw_idle()

    # --------------------------------------------------------------- drag
    def _connect_drag_events(self):
        """Connect matplotlib events for dragging cursors."""
        if self.parent_widget is None:
            return
        canvas = self.parent_widget.canvas_widget.canvas
        canvas.mpl_connect('pick_event', self._on_pick)
        canvas.mpl_connect('motion_notify_event', self._on_motion)
        canvas.mpl_connect('button_release_event', self._on_release)
        canvas.mpl_connect('key_press_event', self._update_hover_cursor)
        canvas.mpl_connect('key_release_event', self._update_hover_cursor)

    def _on_pick(self, event):
        """Handle pick event when clicking on a cursor patch."""
        if event.artist is None:
            return
        for cursor in self._cursors:
            if cursor.get('patch') == event.artist:
                self._dragging_cursor = cursor
                click_pos = (event.mouseevent.xdata, event.mouseevent.ydata)
                modifiers = QApplication.keyboardModifiers()
                is_shift = bool(modifiers & Qt.ShiftModifier)
                if cursor['type'] == "polar":
                    # Polar cursors are not translated; instead the nearest
                    # edge (a phase or modulation bound) is dragged.
                    self._drag_mode = 'polar_edge'
                    self._polar_edge = self._closest_polar_edge(
                        cursor, click_pos
                    )
                elif cursor['type'] == "elliptic" and is_shift:
                    self._drag_mode = 'rotate'
                    if click_pos[0] is not None and click_pos[1] is not None:
                        dy = click_pos[1] - cursor['s']
                        dx = click_pos[0] - cursor['g']
                        self._drag_start_angle = np.degrees(np.arctan2(dy, dx))
                        self._drag_start_cursor_angle = cursor['angle']
                else:
                    self._drag_mode = 'translate'
                    if click_pos[0] is not None and click_pos[1] is not None:
                        self._drag_offset = (
                            cursor['g'] - click_pos[0],
                            cursor['s'] - click_pos[1],
                        )
                break

    @staticmethod
    def _closest_polar_edge(cursor, click_pos):
        """Return which polar boundary is nearest to ``click_pos``.

        One of ``'phase_min'``, ``'phase_max'``, ``'modulation_min'`` or
        ``'modulation_max'``. Returns ``None`` if the click is invalid.
        """
        g, s = click_pos
        if g is None or s is None:
            return None
        r = float(np.hypot(g, s))
        theta = float(np.degrees(np.arctan2(s, g)))

        def angle_diff(a, b):
            return abs(((a - b + 180.0) % 360.0) - 180.0)

        # Distances (in data units) to each of the four boundaries.
        d_inner = abs(r - cursor['modulation_min'])
        d_outer = abs(r - cursor['modulation_max'])
        radial = max(r, 1e-6)
        d_pmin = np.radians(angle_diff(theta, cursor['phase_min'])) * radial
        d_pmax = np.radians(angle_diff(theta, cursor['phase_max'])) * radial

        edges = {
            'modulation_min': d_inner,
            'modulation_max': d_outer,
            'phase_min': d_pmin,
            'phase_max': d_pmax,
        }
        return min(edges, key=edges.get)

    def _update_hover_cursor(self, event):
        """Change the mouse cursor based on hover/interaction."""
        if self._dragging_cursor is not None:
            return
        if self.parent_widget is None:
            return
        canvas = self.parent_widget.canvas_widget.canvas
        is_hovering = False
        hovered = None
        if (
            getattr(event, 'inaxes', None) is not None
            and getattr(event, 'xdata', None) is not None
        ):
            for cursor in self._cursors:
                patch = cursor.get('patch')
                if patch is not None and patch.axes == event.inaxes:
                    contains, _ = patch.contains(event)
                    if contains:
                        is_hovering = True
                        hovered = cursor
                        break
        if is_hovering:
            modifiers = QApplication.keyboardModifiers()
            is_shift = bool(modifiers & Qt.ShiftModifier)
            if hovered['type'] == "elliptic" and is_shift:
                canvas.setCursor(Qt.CrossCursor)
            else:
                canvas.setCursor(Qt.SizeAllCursor)
        else:
            canvas.setCursor(Qt.ArrowCursor)

    def _on_motion(self, event):
        """Handle mouse motion to drag a cursor."""
        self._update_hover_cursor(event)

        cursor = self._dragging_cursor
        if cursor is None or cursor not in self._cursors:
            return
        if event.xdata is None or event.ydata is None:
            return

        if self._drag_mode == 'polar_edge' and cursor['type'] == "polar":
            self._drag_polar_edge(cursor, event.xdata, event.ydata)
        elif self._drag_mode == 'rotate' and cursor['type'] == "elliptic":
            dy = event.ydata - cursor['s']
            dx = event.xdata - cursor['g']
            current_angle = np.degrees(np.arctan2(dy, dx))
            angle_diff = current_angle - self._drag_start_angle
            new_angle = (self._drag_start_cursor_angle + angle_diff) % 360.0
            cursor['angle'] = new_angle
            if cursor.get('patch') is not None:
                cursor['patch'].set_angle(new_angle)
            cursor['angle_spin'].blockSignals(True)
            cursor['angle_spin'].setValue(new_angle)
            cursor['angle_spin'].blockSignals(False)
        else:  # translate
            new_g = event.xdata + self._drag_offset[0]
            new_s = event.ydata + self._drag_offset[1]
            cursor['g'] = new_g
            cursor['s'] = new_s
            patch = cursor.get('patch')
            if patch is not None:
                if cursor['type'] == "circular":
                    patch.center = (new_g, new_s)
                else:
                    patch.set_center((new_g, new_s))
            for key, val in (('g_spin', new_g), ('s_spin', new_s)):
                cursor[key].blockSignals(True)
                cursor[key].setValue(val)
                cursor[key].blockSignals(False)

        if self.parent_widget is not None:
            self.parent_widget.canvas_widget.canvas.draw_idle()

    def _drag_polar_edge(self, cursor, x, y):
        """Move the picked polar boundary to the pointer position."""
        edge = getattr(self, '_polar_edge', None)
        if edge is None:
            return
        r = float(np.hypot(x, y))
        theta = float(np.degrees(np.arctan2(y, x)))

        if edge == 'phase_min':
            cursor['phase_min'] = theta
            spin, value = cursor['phase_min_spin'], theta
        elif edge == 'phase_max':
            cursor['phase_max'] = theta
            spin, value = cursor['phase_max_spin'], theta
        elif edge == 'modulation_min':
            value = min(max(0.0, min(1.0, r)), cursor['modulation_max'])
            cursor['modulation_min'] = value
            spin = cursor['mod_min_spin']
        else:  # modulation_max
            value = max(max(0.0, min(1.0, r)), cursor['modulation_min'])
            cursor['modulation_max'] = value
            spin = cursor['mod_max_spin']

        spin.blockSignals(True)
        spin.setValue(value)
        spin.blockSignals(False)
        self._update_cursor_patch(cursor)

    def _on_release(self, event):
        """Handle mouse release to finish dragging and update selection."""
        if self._dragging_cursor is not None:
            if self._autoupdate_enabled:
                self._apply_selection()
            else:
                self._update_cursor_statistics()
            self._dragging_cursor = None
            self._drag_mode = None
            self._drag_offset = (0, 0)
            if self.parent_widget is not None:
                self.parent_widget.canvas_widget.canvas.setCursor(
                    Qt.ArrowCursor
                )

    def closeEvent(self, event):
        """Clean up signal connections before closing."""
        if hasattr(self, 'parent_widget') and self.parent_widget:
            with contextlib.suppress(TypeError, ValueError, AttributeError):
                self.parent_widget.canvas_widget.show_color_overlay_signal.disconnect()
        event.accept()


def draw_selection_overlay(ax, cursors, mode="cursor", settings=None):
    """Stateless function to draw selection cursors on a matplotlib axes."""
    from matplotlib.patches import CirclePolygon, Ellipse, Wedge

    if mode == "cluster":
        # Cluster regions are data-dependent, so no static cursors to overlay
        return

    for cursor in cursors:
        cursor_type = cursor.get("type", "circular")

        # Color parsing
        color_val = cursor.get("color", "#ff0000")
        if hasattr(color_val, "redF"):  # QColor
            color = (
                color_val.redF(),
                color_val.greenF(),
                color_val.blueF(),
                1.0,
            )
        elif isinstance(color_val, str) and color_val.startswith("#"):
            color = color_val
        else:
            color = color_val

        if cursor_type == "circular":
            patch = CirclePolygon(
                (cursor["g"], cursor["s"]),
                radius=cursor["radius"],
                resolution=64,
                edgecolor=color,
                facecolor="none",
                linewidth=2,
                zorder=10,
            )
            ax.add_patch(patch)
        elif cursor_type == "elliptic":
            patch = Ellipse(
                (cursor["g"], cursor["s"]),
                width=cursor["radius"] * 2,
                height=cursor["radius_minor"] * 2,
                angle=cursor.get("angle", 0),
                edgecolor=color,
                facecolor="none",
                linewidth=2,
                zorder=10,
            )
            ax.add_patch(patch)
        elif cursor_type == "polar":
            r = cursor['modulation_max']
            width = cursor['modulation_max'] - cursor['modulation_min']
            if width <= 0:
                width = 0.001
            patch = Wedge(
                (0, 0),
                r,
                cursor['phase_min'],
                cursor['phase_max'],
                width=width,
                fill=False,
                edgecolor=color,
                linewidth=2,
                zorder=10,
            )
            ax.add_patch(patch)
