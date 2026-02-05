from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse
from napari.layers import Labels
from napari.utils import DirectLabelColormap
from phasorpy.cluster import phasor_cluster_gmm
from phasorpy.cursor import (
    mask_from_circular_cursor,
    mask_from_elliptic_cursor,
)
from qtpy import uic
from qtpy.QtCore import Signal
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)
from skimage.util import map_array

from ._utils import colormap_to_dict


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
            ["Circular Cursor", "Manual Selection", "Automatic Clustering"]
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
                self.refresh_selection_button.style().SP_BrowserReload
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

        # Connect to multiple signals to handle both selection and text editing
        self.selection_input_widget.phasor_selection_id_combobox.currentIndexChanged.connect(
            self.on_selection_id_changed
        )
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

        # === Circular Cursor Mode Widget ===
        self.circular_cursor_widget = CircularCursorWidget(
            viewer, self.parent_widget
        )
        self.stacked_widget.addWidget(self.circular_cursor_widget)

        self.stacked_widget.addWidget(self.manual_selection_widget)

        # === Automatic Clustering Mode Widget ===
        self.automatic_clustering_widget = AutomaticClusteringWidget(
            viewer, self.parent_widget
        )
        self.stacked_widget.addWidget(self.automatic_clustering_widget)

        # Connect mode change
        self.selection_mode_combobox.currentIndexChanged.connect(
            self._on_selection_mode_changed
        )

    def on_harmonic_changed(self):
        """Callback when harmonic spinbox is changed."""
        # Only update cursor visibility for the currently active mode
        current_mode = self.selection_mode_combobox.currentIndex()
        if current_mode == 0:  # Circular Cursor mode
            self.circular_cursor_widget.on_harmonic_changed()
        elif current_mode == 2:  # Automatic Clustering mode
            self.automatic_clustering_widget.on_harmonic_changed()

    def is_manual_selection_mode(self):
        """Check if manual selection mode is currently active."""
        return (
            self.selection_mode_combobox.currentIndex() == 1
        )  # Manual is index 1

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
                    if selection_type == 'circular_cursor':
                        viewer_layer.visible = not show_manual
                    elif selection_type == 'automatic_clustering':
                        viewer_layer.visible = not show_manual
                    elif selection_type == 'manual':
                        viewer_layer.visible = show_manual

    def _on_selection_mode_changed(self, index):
        """Handle selection mode change."""
        self.stacked_widget.setCurrentIndex(index)

        if index == 1:  # Manual selection mode
            self.circular_cursor_widget.clear_all_patches()
            self.automatic_clustering_widget.clear_all_patches()
            if self.parent_widget is not None:
                self.parent_widget._set_selection_visibility(True)
            self._manage_labels_layer_visibility(show_manual=True)
            self.update_phasor_plot_with_selection_id(self.selection_id)
        elif index == 2:  # Automatic clustering mode
            # Deactivate any active selection tools before hiding toolbar
            if self.parent_widget is not None:
                self.parent_widget.canvas_widget._on_escape(None)
            self.circular_cursor_widget.clear_all_patches()
            self.automatic_clustering_widget.redraw_all_patches()
            if self.parent_widget is not None:
                self.parent_widget._set_selection_visibility(False)
            if self.parent_widget is not None:
                self.parent_widget.plot(selection_id_data=None)
            self._manage_labels_layer_visibility(show_manual=False)
        else:  # Circular cursor mode
            # Deactivate any active selection tools before hiding toolbar
            if self.parent_widget is not None:
                self.parent_widget.canvas_widget._on_escape(None)
            self.circular_cursor_widget.redraw_all_patches()
            self.automatic_clustering_widget.clear_all_patches()
            if self.parent_widget is not None:
                self.parent_widget._set_selection_visibility(False)
            if self.parent_widget is not None:
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
        try:
            self.parent_widget.canvas_widget.show_color_overlay_signal.disconnect(
                self._on_show_color_overlay
            )
        except (TypeError, RuntimeError):
            pass
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
        """Callback when the image layer changes - restores circular cursors from metadata."""
        # NOTE: Commented out restoring manual selections until they are saved during export
        # layer = self._get_current_layer()
        # if layer is None:
        #     return

        # self.selection_input_widget.phasor_selection_id_combobox.blockSignals(
        #     True
        # )
        # self.selection_input_widget.phasor_selection_id_combobox.clear()
        # self.selection_input_widget.phasor_selection_id_combobox.addItem(
        #     "None"
        # )

        # if (
        #     "settings" in layer.metadata
        #     and "selections" in layer.metadata["settings"]
        #     and "manual_selections" in layer.metadata["settings"]["selections"]
        # ):
        #     manual_selections = layer.metadata["settings"]["selections"][
        #         "manual_selections"
        #     ]
        #     for selection_id in manual_selections.keys():
        #         self.selection_input_widget.phasor_selection_id_combobox.addItem(
        #             selection_id
        #         )
        #         self._recreate_manual_selection_layer(
        #             selection_id, manual_selections[selection_id]
        #         )

        # self.selection_input_widget.phasor_selection_id_combobox.setCurrentText(
        #     "None"
        # )
        # self._current_selection_id = "None"
        # self.selection_id = "None"

        # self.selection_input_widget.phasor_selection_id_combobox.blockSignals(
        #     False
        # )

        # self._phasors_selected_layer = None

        self.circular_cursor_widget._on_image_layer_changed()

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
                spatial_shape = (
                    layer.data.shape[:2]
                    if layer.data.ndim >= 2
                    else layer.data.shape
                )
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

                if g_array.ndim == 3:
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
            ):
                return candidate_name
            elif candidate_name not in combobox_selections:
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

                    if g_array.ndim == 3:
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
        for layer, layer_selection in zip(selected_layers, selection_splits):
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
                spatial_shape = (
                    layer.data.shape[:2]
                    if layer.data.ndim >= 2
                    else layer.data.shape
                )
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

            if g_array.ndim == 3:
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
            spatial_shape = (
                layer.data.shape[:2]
                if layer.data.ndim >= 2
                else layer.data.shape
            )

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

        # Store cluster data: list of dicts with cluster info
        self._clusters = []
        self._ellipse_patches = []
        self._phasors_selected_layer = None

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
            ["Gaussian Mixture Models (GMM)"]
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

            # Extract correct harmonic if arrays are 3D
            if g_array.ndim == 3:
                harmonic = self.parent_widget.harmonic
                # Harmonic numbering starts at 1, but array indexing starts at 0
                g = g_array[harmonic - 1]
                s = s_array[harmonic - 1]
            else:
                g = g_array
                s = s_array

            spatial_shape = (
                layer.data.shape[:2]
                if layer.data.ndim >= 2
                else layer.data.shape
            )

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

            # Step 4: Apply the same cluster parameters to each layer
            for layer_info in layer_data:
                layer = layer_info['layer']
                g = layer_info['g']
                s = layer_info['s']
                spatial_shape = layer_info['spatial_shape']

                # Create selection map using elliptic cursor masks
                selection_map = np.zeros(spatial_shape, dtype=np.uint32)

                # Apply each cluster using elliptic cursor
                for idx in range(n_clusters):
                    mask = mask_from_elliptic_cursor(
                        g,
                        s,
                        center_real[idx],
                        center_imag[idx],
                        radius=radius[idx],
                        radius_minor=radius_minor[idx],
                        angle=angle[idx],
                    )
                    selection_map[mask] = idx + 1

                # Store cluster information (once per layer for labels)
                cluster_info = {
                    'layer': layer,
                    'center_real': center_real,
                    'center_imag': center_imag,
                    'radius': radius,
                    'radius_minor': radius_minor,
                    'angle': angle,
                    'n_clusters': n_clusters,
                    'harmonic': self.parent_widget.harmonic,
                }
                self._clusters.append(cluster_info)

                # Create labels layer
                self._create_or_update_labels_layer(layer, selection_map)

        except Exception as e:
            print(f"Error applying clustering: {e}")
            import traceback

            traceback.print_exc()

        # Enable clear button
        self.clear_button.setEnabled(True)

        # Redraw canvas
        if self.parent_widget is not None:
            self.parent_widget.canvas_widget.canvas.draw_idle()

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
            try:
                patch.remove()
            except ValueError:
                pass

        self._ellipse_patches.clear()

        if not clear_patches_only:
            # Remove labels layers
            for cluster_info in self._clusters:
                layer = cluster_info['layer']
                layer_name = f"Cluster Selection: {layer.name}"
                for viewer_layer in list(self.viewer.layers):
                    if viewer_layer.name == layer_name:
                        self.viewer.layers.remove(viewer_layer)
                        break

            self._clusters.clear()
            self.clear_button.setEnabled(False)

        # Redraw canvas
        if self.parent_widget is not None:
            self.parent_widget.canvas_widget.canvas.draw_idle()

    def clear_all_patches(self):
        """Clear all patches from the canvas (called when switching modes)."""
        for patch in self._ellipse_patches:
            try:
                patch.remove()
            except ValueError:
                pass
        self._ellipse_patches.clear()

        if self.parent_widget is not None:
            self.parent_widget.canvas_widget.canvas.draw_idle()

    def redraw_all_patches(self):
        """Redraw all patches on the canvas (called when switching back to clustering mode)."""
        # Clear existing patches first
        self.clear_all_patches()

        # Redraw ellipses for all clusters
        for cluster_info in self._clusters:
            center_real = cluster_info['center_real']
            center_imag = cluster_info['center_imag']
            radius = cluster_info['radius']
            radius_minor = cluster_info['radius_minor']
            angle = cluster_info['angle']
            harmonic = cluster_info.get('harmonic', None)
            self._draw_cluster_ellipses(
                center_real, center_imag, radius, radius_minor, angle, harmonic
            )

        if self.parent_widget is not None:
            self.parent_widget.canvas_widget.canvas.draw_idle()

    def on_harmonic_changed(self):
        """Called when the harmonic selection changes. Redraws cluster ellipses to show only those matching the current harmonic."""
        self.redraw_all_patches()

    def _create_or_update_labels_layer(self, image_layer, selection_map):
        """Create or update the labels layer for the cluster selection."""
        layer_name = f"Cluster Selection: {image_layer.name}"

        # Create color dictionary for clusters
        color_dict = {None: (0, 0, 0, 0)}
        n_clusters = self.num_clusters_spinbox.value()
        for idx in range(n_clusters):
            color_idx = idx % len(self.DEFAULT_COLORS)
            color = self.DEFAULT_COLORS[color_idx]
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
                color_dict=color_dict, name="cluster_colors"
            )
            existing_layer.visible = True
            self._phasors_selected_layer = existing_layer
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
            self._phasors_selected_layer = self.viewer.add_layer(labels_layer)

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


class CircularCursorWidget(QWidget):
    """
    Widget for circular cursor selection in phasor plots.

    This widget provides a table interface for adding and managing
    circular cursors that can be used to select regions in the phasor plot.

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
        """Initialize the CircularCursorWidget."""
        super().__init__()
        self.viewer = viewer
        self.parent_widget = parent_widget

        # Store cursor data: list of dicts with g, s, radius, color, patch
        self._cursors = []
        self._phasors_selected_layer = None

        # Dragging state
        self._dragging_cursor = None
        self._drag_offset = (0, 0)

        self._setup_ui()
        self._connect_drag_events()

    def _setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 5, 0, 0)

        # Table for cursors
        self.cursor_table = QTableWidget()
        self.cursor_table.setColumnCount(5)
        self.cursor_table.setHorizontalHeaderLabels(
            ["G", "S", "Radius", "Color", ""]
        )
        self.cursor_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )
        self.cursor_table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.Fixed
        )
        self.cursor_table.horizontalHeader().setSectionResizeMode(
            4, QHeaderView.Fixed
        )
        self.cursor_table.setColumnWidth(3, 40)
        self.cursor_table.setColumnWidth(4, 40)
        self.cursor_table.verticalHeader().setVisible(False)
        layout.addWidget(self.cursor_table)

        # Buttons for add and remove
        buttons_layout = QHBoxLayout()

        self.add_cursor_button = QPushButton("Add Cursor")
        self.add_cursor_button.clicked.connect(self._add_cursor)
        buttons_layout.addWidget(self.add_cursor_button)

        self.clear_all_button = QPushButton("Clear All")
        self.clear_all_button.clicked.connect(self._clear_all_cursors)
        buttons_layout.addWidget(self.clear_all_button)

        layout.addLayout(buttons_layout)

        layout.addStretch()

    def _get_next_color(self):
        """Get the next color from the default palette based on current harmonic cursors."""
        if self.parent_widget is None:
            index = 0
        else:
            current_harmonic = self.parent_widget.harmonic
            # Count cursors in current harmonic only
            current_harmonic_count = sum(
                1
                for c in self._cursors
                if c.get('harmonic', 1) == current_harmonic
            )
            index = current_harmonic_count % len(self.DEFAULT_COLORS)
        return self.DEFAULT_COLORS[index]

    def _get_last_radius(self):
        """Get the radius from the last cursor, or default if none."""
        if self._cursors:
            return self._cursors[-1]['radius']
        return 0.05  # Default radius

    def _add_cursor(self, g=0.5, s=0.5, radius=None, color=None):
        """Add a new cursor to the table."""
        if color is None:
            color = self._get_next_color()
        if radius is None:
            radius = self._get_last_radius()

        # Store cursor data first to get the correct index
        cursor_data = {
            'g': g,
            's': s,
            'radius': radius,
            'color': color,
            'patch': None,
            'harmonic': (
                self.parent_widget.harmonic if self.parent_widget else 1
            ),
        }
        self._cursors.append(cursor_data)

        # The actual index in self._cursors for this cursor
        cursor_idx = len(self._cursors) - 1

        # Now add to table
        table_row = self.cursor_table.rowCount()
        self.cursor_table.insertRow(table_row)

        # G spinbox - use cursor_idx not table_row
        g_spinbox = QDoubleSpinBox()
        g_spinbox.setRange(-1.5, 1.5)
        g_spinbox.setSingleStep(0.01)
        g_spinbox.setDecimals(2)
        g_spinbox.setValue(g)
        g_spinbox.valueChanged.connect(
            lambda val, idx=cursor_idx: self._on_cursor_changed(idx)
        )
        self.cursor_table.setCellWidget(table_row, 0, g_spinbox)

        # S spinbox
        s_spinbox = QDoubleSpinBox()
        s_spinbox.setRange(-1.5, 1.5)
        s_spinbox.setSingleStep(0.01)
        s_spinbox.setDecimals(2)
        s_spinbox.setValue(s)
        s_spinbox.valueChanged.connect(
            lambda val, idx=cursor_idx: self._on_cursor_changed(idx)
        )
        self.cursor_table.setCellWidget(table_row, 1, s_spinbox)

        # Radius spinbox
        radius_spinbox = QDoubleSpinBox()
        radius_spinbox.setRange(0.001, 1.0)
        radius_spinbox.setSingleStep(0.01)
        radius_spinbox.setDecimals(3)
        radius_spinbox.setValue(radius)
        radius_spinbox.valueChanged.connect(
            lambda val, idx=cursor_idx: self._on_cursor_changed(idx)
        )
        self.cursor_table.setCellWidget(table_row, 2, radius_spinbox)

        # Color button
        color_button = ColorButton(color)
        color_button.color_changed.connect(
            lambda c, idx=cursor_idx: self._on_cursor_changed(idx)
        )
        self.cursor_table.setCellWidget(table_row, 3, color_button)

        # Remove button
        remove_button = QPushButton("×")
        remove_button.setFixedSize(25, 25)
        remove_button.clicked.connect(
            lambda _, idx=cursor_idx: self._remove_cursor(idx)
        )
        self.cursor_table.setCellWidget(table_row, 4, remove_button)

        # Draw patch on canvas using cursor_idx
        self._update_cursor_patch(cursor_idx)

        # Apply selection to update labels layer
        if self._cursors:
            self._apply_selection()

    def _remove_cursor(self, cursor_idx):
        """Remove a cursor from the table using its index in self._cursors."""
        if cursor_idx < 0 or cursor_idx >= len(self._cursors):
            return

        # Remove patch from canvas
        if self._cursors[cursor_idx]['patch'] is not None:
            try:
                self._cursors[cursor_idx]['patch'].remove()
            except ValueError:
                pass

        # Remove from data
        self._cursors.pop(cursor_idx)

        # Rebuild the table to reflect the removal
        # This is simpler than tracking table row mappings
        if self.parent_widget is not None:
            self.on_harmonic_changed()
        else:
            # If no parent, just clear the table and rebuild manually
            self.cursor_table.setRowCount(0)

        # Redraw canvas
        if self.parent_widget is not None:
            self.parent_widget.canvas_widget.canvas.draw_idle()

        # Apply selection to update labels layer (already called in on_harmonic_changed)
        if not self._cursors:
            # If no cursors left, remove the selection layer
            self._remove_selection_layer()

    def _reconnect_row_signals(self):
        """Reconnect signals after row removal to update row indices."""
        for row in range(self.cursor_table.rowCount()):
            # Get widgets
            g_spinbox = self.cursor_table.cellWidget(row, 0)
            s_spinbox = self.cursor_table.cellWidget(row, 1)
            radius_spinbox = self.cursor_table.cellWidget(row, 2)
            color_button = self.cursor_table.cellWidget(row, 3)
            remove_button = self.cursor_table.cellWidget(row, 4)

            # Disconnect existing connections and reconnect with correct row
            try:
                g_spinbox.valueChanged.disconnect()
                s_spinbox.valueChanged.disconnect()
                radius_spinbox.valueChanged.disconnect()
                color_button.color_changed.disconnect()
                remove_button.clicked.disconnect()
            except TypeError:
                pass

            g_spinbox.valueChanged.connect(
                lambda val, r=row: self._on_cursor_changed(r)
            )
            s_spinbox.valueChanged.connect(
                lambda val, r=row: self._on_cursor_changed(r)
            )
            radius_spinbox.valueChanged.connect(
                lambda val, r=row: self._on_cursor_changed(r)
            )
            color_button.color_changed.connect(
                lambda c, r=row: self._on_cursor_changed(r)
            )
            remove_button.clicked.connect(
                lambda _, r=row: self._remove_cursor(r)
            )

    def _on_cursor_changed(self, cursor_idx):
        """Handle cursor parameter changes using cursor index in self._cursors."""
        if cursor_idx < 0 or cursor_idx >= len(self._cursors):
            return

        if self.parent_widget is None:
            return

        # Find the table row for this cursor (only current harmonic cursors are in table)
        current_harmonic = self.parent_widget.harmonic
        table_row = 0
        for idx in range(len(self._cursors)):
            if self._cursors[idx].get('harmonic', 1) == current_harmonic:
                if idx == cursor_idx:
                    # Found it!
                    break
                table_row += 1
        else:
            # Cursor not in current harmonic's table
            return

        # Get current values from widgets using the table row
        g_spinbox = self.cursor_table.cellWidget(table_row, 0)
        s_spinbox = self.cursor_table.cellWidget(table_row, 1)
        radius_spinbox = self.cursor_table.cellWidget(table_row, 2)
        color_button = self.cursor_table.cellWidget(table_row, 3)

        if all([g_spinbox, s_spinbox, radius_spinbox, color_button]):
            self._cursors[cursor_idx]['g'] = g_spinbox.value()
            self._cursors[cursor_idx]['s'] = s_spinbox.value()
            self._cursors[cursor_idx]['radius'] = radius_spinbox.value()
            self._cursors[cursor_idx]['color'] = color_button.color()

            self._update_cursor_patch(cursor_idx)

            # Apply selection automatically if not currently dragging
            if self._dragging_cursor is None and self._cursors:
                self._apply_selection()

    def _update_cursor_patch(self, row):
        """Update or create the patch for a cursor."""
        if row < 0 or row >= len(self._cursors):
            return

        if self.parent_widget is None:
            return

        cursor = self._cursors[row]

        # Only show cursor if it matches current harmonic
        current_harmonic = self.parent_widget.harmonic
        cursor_harmonic = cursor.get('harmonic', 1)

        # Remove old patch if exists
        if cursor['patch'] is not None:
            try:
                cursor['patch'].remove()
            except ValueError:
                pass
            cursor['patch'] = None

        # Only create new patch if harmonics match
        if cursor_harmonic != current_harmonic:
            self.parent_widget.canvas_widget.canvas.draw_idle()
            return

        ax = self.parent_widget.canvas_widget.axes

        color = cursor['color']
        edge_rgba = (color.redF(), color.greenF(), color.blueF(), 1.0)

        patch = Circle(
            (cursor['g'], cursor['s']),
            cursor['radius'],
            fill=False,
            edgecolor=edge_rgba,
            linewidth=2,
            zorder=10,
            picker=True,
        )
        cursor['patch'] = ax.add_patch(patch)

        self.parent_widget.canvas_widget.canvas.draw_idle()

    def _clear_all_cursors(self):
        """Clear all cursors."""
        for cursor in self._cursors:
            if cursor['patch'] is not None:
                try:
                    cursor['patch'].remove()
                except ValueError:
                    pass

        self._cursors.clear()
        self.cursor_table.setRowCount(0)

        self._remove_selection_layer()

        if self.parent_widget is not None:
            self.parent_widget.canvas_widget.canvas.draw_idle()

    def clear_all_patches(self):
        """Clear all patches from the canvas (called when switching modes)."""
        for cursor in self._cursors:
            if cursor['patch'] is not None:
                try:
                    cursor['patch'].remove()
                except ValueError:
                    pass
                cursor['patch'] = None

        if self.parent_widget is not None:
            self.parent_widget.canvas_widget.canvas.draw_idle()

    def redraw_all_patches(self):
        """Redraw all patches on the canvas (called when switching back to circular cursor mode)."""
        for row in range(len(self._cursors)):
            self._update_cursor_patch(row)

    def on_harmonic_changed(self):
        """Handle harmonic change - rebuild table and labels for current harmonic only."""
        if self.parent_widget is None:
            return

        current_harmonic = self.parent_widget.harmonic

        # Rebuild the table to show only cursors for current harmonic
        self.cursor_table.setRowCount(0)

        for cursor_idx, cursor in enumerate(self._cursors):
            cursor_harmonic = cursor.get('harmonic', 1)
            if cursor_harmonic == current_harmonic:
                # Add this cursor to the table
                table_row = self.cursor_table.rowCount()
                self.cursor_table.insertRow(table_row)

                # G spinbox - use cursor_idx (actual index in self._cursors)
                g_spinbox = QDoubleSpinBox()
                g_spinbox.setRange(-1.5, 1.5)
                g_spinbox.setSingleStep(0.01)
                g_spinbox.setDecimals(2)
                g_spinbox.setValue(cursor['g'])
                g_spinbox.valueChanged.connect(
                    lambda val, idx=cursor_idx: self._on_cursor_changed(idx)
                )
                self.cursor_table.setCellWidget(table_row, 0, g_spinbox)

                # S spinbox
                s_spinbox = QDoubleSpinBox()
                s_spinbox.setRange(-1.5, 1.5)
                s_spinbox.setSingleStep(0.01)
                s_spinbox.setDecimals(2)
                s_spinbox.setValue(cursor['s'])
                s_spinbox.valueChanged.connect(
                    lambda val, idx=cursor_idx: self._on_cursor_changed(idx)
                )
                self.cursor_table.setCellWidget(table_row, 1, s_spinbox)

                # Radius spinbox
                radius_spinbox = QDoubleSpinBox()
                radius_spinbox.setRange(0.001, 1.0)
                radius_spinbox.setSingleStep(0.01)
                radius_spinbox.setDecimals(3)
                radius_spinbox.setValue(cursor['radius'])
                radius_spinbox.valueChanged.connect(
                    lambda val, idx=cursor_idx: self._on_cursor_changed(idx)
                )
                self.cursor_table.setCellWidget(table_row, 2, radius_spinbox)

                # Color button
                color_button = ColorButton(cursor['color'])
                color_button.color_changed.connect(
                    lambda c, idx=cursor_idx: self._on_cursor_changed(idx)
                )
                self.cursor_table.setCellWidget(table_row, 3, color_button)

                # Remove button
                remove_button = QPushButton("×")
                remove_button.setFixedSize(25, 25)
                remove_button.clicked.connect(
                    lambda _, idx=cursor_idx: self._remove_cursor(idx)
                )
                self.cursor_table.setCellWidget(table_row, 4, remove_button)

        # Redraw patches for current harmonic
        for row in range(len(self._cursors)):
            self._update_cursor_patch(row)

        # Update labels layer to show only current harmonic selections
        self._apply_selection()

    def _on_image_layer_changed(self):
        """Callback when image layer changes - clear and restore circular cursors."""
        for cursor in self._cursors:
            if cursor['patch'] is not None:
                try:
                    cursor['patch'].remove()
                except ValueError:
                    pass
                cursor['patch'] = None

        self._cursors.clear()
        self.cursor_table.setRowCount(0)

        # Use primary layer for restoring cursors
        selected_layers = self._get_selected_layers()
        if not selected_layers:
            return

        primary_layer = selected_layers[0]
        if (
            "settings" in primary_layer.metadata
            and "selections" in primary_layer.metadata["settings"]
            and "circular_cursors"
            in primary_layer.metadata["settings"]["selections"]
        ):
            cursor_params = primary_layer.metadata["settings"]["selections"][
                "circular_cursors"
            ]

            original_apply_selection = getattr(self, "_apply_selection", None)

            def _noop_apply_selection(*args, **kwargs):
                return None

            if original_apply_selection is not None:
                self._apply_selection = _noop_apply_selection
            try:
                for params in cursor_params:
                    color = QColor(*params["color"])
                    self._add_cursor(
                        g=params["g"],
                        s=params["s"],
                        radius=params["radius"],
                        color=color,
                    )
            finally:
                if original_apply_selection is not None:
                    self._apply_selection = original_apply_selection
                    self._apply_selection()

        if self.parent_widget is not None:
            self.parent_widget.canvas_widget.canvas.draw_idle()

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

    def _remove_selection_layer(self):
        """Remove the selection layer if it exists."""
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

    def _apply_selection(self):
        """Apply the circular cursor selections to create a labels layer."""
        if not self._cursors:
            return
        if self.parent_widget is None:
            return None

        selected_layers = self._get_selected_layers()
        if not selected_layers:
            return

        current_harmonic = self.parent_widget.harmonic

        # Filter cursors for current harmonic only
        current_harmonic_cursors = [
            cursor
            for cursor in self._cursors
            if cursor.get('harmonic', 1) == current_harmonic
        ]

        if not current_harmonic_cursors:
            # No cursors for this harmonic - clear labels layers
            for layer in selected_layers:
                layer_name = f"Cursor Selection: {layer.name}"
                for viewer_layer in self.viewer.layers:
                    if viewer_layer.name == layer_name:
                        self.viewer.layers.remove(viewer_layer)
                        break
            return

        cursor_params = []
        for cursor in current_harmonic_cursors:
            cursor_params.append(
                {
                    'g': cursor['g'],
                    's': cursor['s'],
                    'radius': cursor['radius'],
                    'color': (
                        cursor['color'].red(),
                        cursor['color'].green(),
                        cursor['color'].blue(),
                        cursor['color'].alpha(),
                    ),
                }
            )

        # Save cursor params to primary layer
        primary_layer = selected_layers[0]
        if "settings" not in primary_layer.metadata:
            primary_layer.metadata["settings"] = {}
        if "selections" not in primary_layer.metadata["settings"]:
            primary_layer.metadata["settings"]["selections"] = {}

        primary_layer.metadata["settings"]["selections"][
            "circular_cursors"
        ] = cursor_params

        # Apply selection to each selected layer
        for layer in selected_layers:
            # Get phasor data for this specific layer
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

            if g_array.ndim == 3:
                g = g_array[harmonic_idx]
                s = s_array[harmonic_idx]
            else:
                g = g_array
                s = s_array

            spatial_shape = (
                layer.data.shape[:2]
                if layer.data.ndim >= 2
                else layer.data.shape
            )

            # Create selection map
            selection_map = np.zeros(spatial_shape, dtype=np.uint32)

            # Apply each cursor from current harmonic
            for idx, cursor in enumerate(current_harmonic_cursors):
                g_center = cursor['g']
                s_center = cursor['s']
                radius = cursor['radius']

                mask = mask_from_circular_cursor(
                    g, s, [g_center], [s_center], radius=[radius]
                )[0]

                selection_map[mask] = idx + 1

            self._create_or_update_labels_layer(
                layer, selection_map, current_harmonic_cursors
            )

    def _create_or_update_labels_layer(
        self, image_layer, selection_map, cursors_list
    ):
        """Create or update the labels layer for the selection."""
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
                color_dict=color_dict, name="circular_cursor_colors"
            )
            existing_layer.visible = True
            self._phasors_selected_layer = existing_layer
        else:
            labels_layer = Labels(
                selection_map,
                name=layer_name,
                scale=image_layer.scale,
                colormap=DirectLabelColormap(
                    color_dict=color_dict, name="circular_cursor_colors"
                ),
                metadata={
                    'napari_phasors_selection_type': 'circular_cursor',
                    'napari_phasors_source_layer': image_layer.name,
                },
            )
            self._phasors_selected_layer = self.viewer.add_layer(labels_layer)

    def _connect_drag_events(self):
        """Connect matplotlib events for dragging circles."""
        if self.parent_widget is None:
            return

        canvas = self.parent_widget.canvas_widget.canvas
        canvas.mpl_connect('pick_event', self._on_pick)
        canvas.mpl_connect('motion_notify_event', self._on_motion)
        canvas.mpl_connect('button_release_event', self._on_release)

    def _on_pick(self, event):
        """Handle pick event when clicking on a circle."""
        if event.artist is None:
            return

        for row, cursor in enumerate(self._cursors):
            if cursor['patch'] == event.artist:
                self._dragging_cursor = row
                click_pos = (event.mouseevent.xdata, event.mouseevent.ydata)
                if click_pos[0] is not None and click_pos[1] is not None:
                    self._drag_offset = (
                        cursor['g'] - click_pos[0],
                        cursor['s'] - click_pos[1],
                    )
                break

    def _on_motion(self, event):
        """Handle mouse motion to drag the circle."""
        if self._dragging_cursor is None:
            return

        if event.xdata is None or event.ydata is None:
            return

        cursor_idx = self._dragging_cursor
        if cursor_idx < 0 or cursor_idx >= len(self._cursors):
            return

        # Calculate new position
        new_g = event.xdata + self._drag_offset[0]
        new_s = event.ydata + self._drag_offset[1]

        # Update cursor data
        self._cursors[cursor_idx]['g'] = new_g
        self._cursors[cursor_idx]['s'] = new_s

        # Update the patch position
        patch = self._cursors[cursor_idx]['patch']
        if patch is not None:
            patch.center = (new_g, new_s)

        # Find the table row for this cursor (only current harmonic cursors are in table)
        if self.parent_widget is not None:
            current_harmonic = self.parent_widget.harmonic
            table_row = 0
            for idx in range(len(self._cursors)):
                if self._cursors[idx].get('harmonic', 1) == current_harmonic:
                    if idx == cursor_idx:
                        # Found it! Update the spinboxes in the table
                        g_spinbox = self.cursor_table.cellWidget(table_row, 0)
                        s_spinbox = self.cursor_table.cellWidget(table_row, 1)

                        if g_spinbox is not None:
                            g_spinbox.blockSignals(True)
                            g_spinbox.setValue(new_g)
                            g_spinbox.blockSignals(False)

                        if s_spinbox is not None:
                            s_spinbox.blockSignals(True)
                            s_spinbox.setValue(new_s)
                            s_spinbox.blockSignals(False)
                        break
                    table_row += 1

        # Redraw
        if self.parent_widget is not None:
            self.parent_widget.canvas_widget.canvas.draw_idle()

    def _on_release(self, event):
        """Handle mouse release to finish dragging and update selection."""
        if self._dragging_cursor is not None:
            self._apply_selection()
            self._dragging_cursor = None
            self._drag_offset = (0, 0)
