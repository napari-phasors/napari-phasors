from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from napari.layers import Labels
from napari.utils import DirectLabelColormap
from phasorpy.cursor import mask_from_circular_cursor
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
            ["Circular Cursor", "Manual Selection"]
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

        # Connect mode change
        self.selection_mode_combobox.currentIndexChanged.connect(
            self._on_selection_mode_changed
        )

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
                    elif selection_type == 'manual':
                        viewer_layer.visible = show_manual

    def _on_selection_mode_changed(self, index):
        """Handle selection mode change."""
        self.stacked_widget.setCurrentIndex(index)

        if index == 1:  # Manual selection mode
            self.circular_cursor_widget.clear_all_patches()
            if self.parent_widget is not None:
                self.parent_widget._set_selection_visibility(True)
            self._manage_labels_layer_visibility(show_manual=True)
            self.update_phasor_plot_with_selection_id(self.selection_id)
        else:  # Circular cursor mode
            # Deactivate any active selection tools before hiding toolbar
            if self.parent_widget is not None:
                self.parent_widget.canvas_widget._on_escape(None)
            self.circular_cursor_widget.redraw_all_patches()
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

    def _find_phasors_layer_by_name(self, layer_name):
        """Find a phasors layer by name in the viewer."""
        for layer in self.viewer.layers:
            if layer.name == layer_name:
                return layer
        return None

    def _on_show_color_overlay(self, visible: bool):
        """Slot to show/hide the current phasors_selected_layer."""
        if self._phasors_selected_layer is not None:
            self._phasors_selected_layer.visible = visible

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

            # Update phasors_selected_layer reference
            layer = self._get_current_layer()
            if new_selection_id_for_comparison != "None" and layer is not None:
                layer_name = f"{new_selection_id_for_comparison}: {layer.name}"
                existing_layer = self._find_phasors_layer_by_name(layer_name)

                if existing_layer is None:
                    if (
                        "settings" in layer.metadata
                        and "selections" in layer.metadata["settings"]
                        and "manual_selections"
                        in layer.metadata["settings"]["selections"]
                        and new_selection_id_for_comparison
                        in layer.metadata["settings"]["selections"][
                            "manual_selections"
                        ]
                    ):
                        selection_map = layer.metadata["settings"][
                            "selections"
                        ]["manual_selections"][new_selection_id_for_comparison]
                        self._recreate_manual_selection_layer(
                            new_selection_id_for_comparison, selection_map
                        )
                        existing_layer = self._find_phasors_layer_by_name(
                            layer_name
                        )

                self._phasors_selected_layer = existing_layer
            else:
                self._phasors_selected_layer = None

            self._connect_show_overlay_signal()

            processed_selection_id = new_selection_id

            if not getattr(self, '_processing_initial_selection', False):
                self.update_phasor_plot_with_selection_id(
                    processed_selection_id
                )
                if self._phasors_selected_layer is not None:
                    self.update_phasors_layer()

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
        layer = self._get_current_layer()
        if layer is None:
            return

        # Prevent this from running during plot updates
        if getattr(self.parent_widget, '_updating_plot', False):
            return

        if selection_id is None or selection_id == "":
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

        selection_layer_name = f"{selection_id}: {layer.name}"
        selection_layer = self._find_phasors_layer_by_name(
            selection_layer_name
        )
        if selection_layer is None:
            self.create_phasors_selected_layer()
            selection_layer = self._phasors_selected_layer

        if selection_layer:
            selection_layer.visible = True

        if (
            "settings" in layer.metadata
            and "selections" in layer.metadata["settings"]
            and "manual_selections" in layer.metadata["settings"]["selections"]
            and selection_id
            in layer.metadata["settings"]["selections"]["manual_selections"]
        ):
            selection_map = layer.metadata["settings"]["selections"][
                "manual_selections"
            ][selection_id]
        else:
            spatial_shape = self.parent_widget.get_phasor_spatial_shape()
            if spatial_shape is None:
                return
            selection_map = np.zeros(spatial_shape, dtype=np.uint32)

        _, _, valid = self.parent_widget.get_masked_gs(
            flat=True, return_valid_mask=True
        )
        if valid is None:
            return

        selection_data = selection_map.ravel()[valid]

        self.parent_widget.plot(selection_id_data=selection_data)

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

        layer = self._get_current_layer()
        used_selections = set()
        if (
            layer is not None
            and "settings" in layer.metadata
            and "selections" in layer.metadata["settings"]
            and "manual_selections" in layer.metadata["settings"]["selections"]
        ):
            used_selections = set(
                layer.metadata["settings"]["selections"][
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
        layer = self._get_current_layer()
        if layer is None:
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

        if (
            "settings" in layer.metadata
            and "selections" in layer.metadata["settings"]
            and "manual_selections" in layer.metadata["settings"]["selections"]
            and self.selection_id
            in layer.metadata["settings"]["selections"]["manual_selections"]
        ):
            selection_map = layer.metadata["settings"]["selections"][
                "manual_selections"
            ][self.selection_id].copy()
        else:
            spatial_shape = self.parent_widget.get_phasor_spatial_shape()
            if spatial_shape is None:
                return
            selection_map = np.zeros(spatial_shape, dtype=np.uint32)

        selection_map_flat = selection_map.ravel()

        _, _, valid_pixels_mask = self.parent_widget.get_masked_gs(
            flat=True, return_valid_mask=True
        )
        if valid_pixels_mask is None:
            return

        selection_to_use = manual_selection
        if (
            hasattr(self, '_processing_initial_selection')
            and self._processing_initial_selection
        ):
            selection_to_use = self._initial_manual_selection
            self._processing_initial_selection = False
            delattr(self, '_initial_manual_selection')

        if selection_to_use is None:
            selection_map_flat[valid_pixels_mask] = 0
        else:
            selection_map_flat[valid_pixels_mask] = selection_to_use

        if "settings" not in layer.metadata:
            layer.metadata["settings"] = {}
        if "selections" not in layer.metadata["settings"]:
            layer.metadata["settings"]["selections"] = {}
        if "manual_selections" not in layer.metadata["settings"]["selections"]:
            layer.metadata["settings"]["selections"]["manual_selections"] = {}

        layer.metadata["settings"]["selections"]["manual_selections"][
            self.selection_id
        ] = selection_map.copy()

        self.update_phasors_layer()

    def create_phasors_selected_layer(self):
        """Create the phasors selected layer."""
        layer = self._get_current_layer()
        if layer is None:
            return
        if self.selection_id is None or self.selection_id == "":
            return

        spatial_shape = self.parent_widget.get_phasor_spatial_shape()
        if spatial_shape is None:
            return

        # Get selection map from metadata if it exists, otherwise create empty
        if (
            "settings" in layer.metadata
            and "selections" in layer.metadata["settings"]
            and "manual_selections" in layer.metadata["settings"]["selections"]
            and self.selection_id
            in layer.metadata["settings"]["selections"]["manual_selections"]
        ):
            selection_map = layer.metadata["settings"]["selections"][
                "manual_selections"
            ][self.selection_id].copy()
        else:
            selection_map = np.zeros(spatial_shape, dtype=np.uint32)

        color_dict = colormap_to_dict(
            self.parent_widget._colormap,
            self.parent_widget._colormap.N,
            exclude_first=True,
        )

        layer_name = f"{self.selection_id}: {layer.name}"

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

        self._phasors_selected_layer = self.viewer.add_layer(
            phasors_selected_layer
        )

        self._connect_show_overlay_signal()

    def update_phasors_layer(self):
        """Update the existing phasors layer data without recreating it."""
        layer = self._get_current_layer()
        if layer is None:
            return

        selection_layer_name = f"{self.selection_id}: {layer.name}"
        existing_phasors_selected_layer = self._find_phasors_layer_by_name(
            selection_layer_name
        )

        if existing_phasors_selected_layer is None:
            self.create_phasors_selected_layer()
            return

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

        self._phasors_selected_layer = existing_phasors_selected_layer

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
        """Get the next color from the default palette."""
        index = len(self._cursors) % len(self.DEFAULT_COLORS)
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

        row = self.cursor_table.rowCount()
        self.cursor_table.insertRow(row)

        # G spinbox
        g_spinbox = QDoubleSpinBox()
        g_spinbox.setRange(-1.5, 1.5)
        g_spinbox.setSingleStep(0.01)
        g_spinbox.setDecimals(2)
        g_spinbox.setValue(g)
        g_spinbox.valueChanged.connect(
            lambda val, r=row: self._on_cursor_changed(r)
        )
        self.cursor_table.setCellWidget(row, 0, g_spinbox)

        # S spinbox
        s_spinbox = QDoubleSpinBox()
        s_spinbox.setRange(-1.5, 1.5)
        s_spinbox.setSingleStep(0.01)
        s_spinbox.setDecimals(2)
        s_spinbox.setValue(s)
        s_spinbox.valueChanged.connect(
            lambda val, r=row: self._on_cursor_changed(r)
        )
        self.cursor_table.setCellWidget(row, 1, s_spinbox)

        # Radius spinbox
        radius_spinbox = QDoubleSpinBox()
        radius_spinbox.setRange(0.001, 1.0)
        radius_spinbox.setSingleStep(0.01)
        radius_spinbox.setDecimals(3)
        radius_spinbox.setValue(radius)
        radius_spinbox.valueChanged.connect(
            lambda val, r=row: self._on_cursor_changed(r)
        )
        self.cursor_table.setCellWidget(row, 2, radius_spinbox)

        # Color button
        color_button = ColorButton(color)
        color_button.color_changed.connect(
            lambda c, r=row: self._on_cursor_changed(r)
        )
        self.cursor_table.setCellWidget(row, 3, color_button)

        # Remove button
        remove_button = QPushButton("×")
        remove_button.setFixedSize(25, 25)
        remove_button.clicked.connect(lambda _, r=row: self._remove_cursor(r))
        self.cursor_table.setCellWidget(row, 4, remove_button)

        # Store cursor data
        cursor_data = {
            'g': g,
            's': s,
            'radius': radius,
            'color': color,
            'patch': None,
        }
        self._cursors.append(cursor_data)

        # Draw patch on canvas
        self._update_cursor_patch(row)

        # Apply selection to update labels layer
        if self._cursors:
            self._apply_selection()

    def _remove_cursor(self, row):
        """Remove a cursor from the table."""
        if row < 0 or row >= len(self._cursors):
            return

        # Remove patch from canvas
        if self._cursors[row]['patch'] is not None:
            try:
                self._cursors[row]['patch'].remove()
            except ValueError:
                pass

        # Remove from data
        self._cursors.pop(row)

        # Remove row from table
        self.cursor_table.removeRow(row)

        # Re-connect signals for remaining rows
        self._reconnect_row_signals()

        # Redraw canvas
        if self.parent_widget is not None:
            self.parent_widget.canvas_widget.canvas.draw_idle()

        # Apply selection to update labels layer
        if self._cursors:
            self._apply_selection()
        else:
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

    def _on_cursor_changed(self, row):
        """Handle cursor parameter changes."""
        if row < 0 or row >= len(self._cursors):
            return

        # Get current values from widgets
        g_spinbox = self.cursor_table.cellWidget(row, 0)
        s_spinbox = self.cursor_table.cellWidget(row, 1)
        radius_spinbox = self.cursor_table.cellWidget(row, 2)
        color_button = self.cursor_table.cellWidget(row, 3)

        if all([g_spinbox, s_spinbox, radius_spinbox, color_button]):
            self._cursors[row]['g'] = g_spinbox.value()
            self._cursors[row]['s'] = s_spinbox.value()
            self._cursors[row]['radius'] = radius_spinbox.value()
            self._cursors[row]['color'] = color_button.color()

            self._update_cursor_patch(row)

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
        ax = self.parent_widget.canvas_widget.axes

        if cursor['patch'] is not None:
            try:
                cursor['patch'].remove()
            except ValueError:
                pass

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

        layer = self._get_current_layer()
        if layer is None:
            return

        if (
            "settings" in layer.metadata
            and "selections" in layer.metadata["settings"]
            and "circular_cursors" in layer.metadata["settings"]["selections"]
        ):
            cursor_params = layer.metadata["settings"]["selections"][
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

    def _remove_selection_layer(self):
        """Remove the selection layer if it exists."""
        layer = self._get_current_layer()
        if layer is None:
            return

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

        layer = self._get_current_layer()
        if layer is None:
            return

        # Get phasor data
        g_flat, s_flat, valid_mask = self.parent_widget.get_masked_gs(
            flat=True, return_valid_mask=True
        )
        if g_flat is None or s_flat is None:
            return

        # Get full phasor arrays for shape
        spatial_shape = self.parent_widget.get_phasor_spatial_shape()
        if spatial_shape is None:
            return

        # Get full G and S arrays
        g_full, s_full = self.parent_widget.get_masked_gs(flat=False)
        if g_full is None or s_full is None:
            return

        # Create selection map
        selection_map = np.zeros(spatial_shape, dtype=np.uint32)

        # Apply each cursor
        for idx, cursor in enumerate(self._cursors):
            g_center = cursor['g']
            s_center = cursor['s']
            radius = cursor['radius']

            mask = mask_from_circular_cursor(
                g_full, s_full, [g_center], [s_center], radius=[radius]
            )[0]

            selection_map[mask] = idx + 1

        cursor_params = []
        for cursor in self._cursors:
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

        if "settings" not in layer.metadata:
            layer.metadata["settings"] = {}
        if "selections" not in layer.metadata["settings"]:
            layer.metadata["settings"]["selections"] = {}

        layer.metadata["settings"]["selections"][
            "circular_cursors"
        ] = cursor_params

        self._create_or_update_labels_layer(layer, selection_map)

    def _create_or_update_labels_layer(self, image_layer, selection_map):
        """Create or update the labels layer for the selection."""
        layer_name = f"Cursor Selection: {image_layer.name}"

        color_dict = {None: (0, 0, 0, 0)}
        for idx, cursor in enumerate(self._cursors):
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

        row = self._dragging_cursor
        if row < 0 or row >= len(self._cursors):
            return

        # Calculate new position
        new_g = event.xdata + self._drag_offset[0]
        new_s = event.ydata + self._drag_offset[1]

        # Update cursor data
        self._cursors[row]['g'] = new_g
        self._cursors[row]['s'] = new_s

        # Update the patch position
        patch = self._cursors[row]['patch']
        if patch is not None:
            patch.center = (new_g, new_s)

        # Update the spinboxes in the table
        g_spinbox = self.cursor_table.cellWidget(row, 0)
        s_spinbox = self.cursor_table.cellWidget(row, 1)

        if g_spinbox is not None:
            g_spinbox.blockSignals(True)
            g_spinbox.setValue(new_g)
            g_spinbox.blockSignals(False)

        if s_spinbox is not None:
            s_spinbox.blockSignals(True)
            s_spinbox.setValue(new_s)
            s_spinbox.blockSignals(False)

        # Redraw
        self.parent_widget.canvas_widget.canvas.draw_idle()

    def _on_release(self, event):
        """Handle mouse release to finish dragging and update selection."""
        if self._dragging_cursor is not None:
            self._apply_selection()
            self._dragging_cursor = None
            self._drag_offset = (0, 0)
