from pathlib import Path

import numpy as np
from napari.layers import Labels
from napari.utils import DirectLabelColormap, notifications
from qtpy import uic
from qtpy.QtWidgets import (
    QHBoxLayout,
    QPushButton,
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

        # Load the UI from the .ui file
        self.selection_input_widget = QWidget()
        uic.loadUi(
            Path(__file__).parent / "ui/selection_tab.ui",
            self.selection_input_widget,
        )
        layout = QVBoxLayout(self)
        layout.addWidget(self.selection_input_widget)

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
        # Update the internal tracking variable
        self._current_selection_id = new_selection_id

        # Ensure storage exists
        if new_selection_id != "None":
            self._ensure_selection_storage(new_selection_id)

    def _get_current_layer(self):
        """Helper to get the currently selected image layer."""
        layer_name = (
            self.parent_widget.image_layer_with_phasor_features_combobox.currentText()
        )
        if not layer_name or layer_name not in self.viewer.layers:
            return None
        return self.viewer.layers[layer_name]

    def _ensure_selection_storage(self, selection_id: str):
        """Ensure the selection ID exists in the layer metadata."""
        layer = self._get_current_layer()
        if layer is None:
            return

        if "selections" not in layer.metadata:
            layer.metadata["selections"] = {}

        if selection_id not in layer.metadata["selections"]:
            # Initialize with zeros, shape of image
            spatial_shape = self.parent_widget.get_phasor_spatial_shape()
            if spatial_shape is None:
                return

            layer.metadata["selections"][selection_id] = np.zeros(
                spatial_shape, dtype=np.uint32
            )

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
            if new_selection_id_for_comparison != "None":
                self._ensure_selection_storage(new_selection_id_for_comparison)

            # Check if we need to recreate a missing selection layer
            layer = self._get_current_layer()
            if new_selection_id_for_comparison != "None" and layer is not None:
                layer_name = f"Selection {new_selection_id_for_comparison}: {layer.name}"
                existing_layer = self._find_phasors_layer_by_name(layer_name)

                # If layer doesn't exist but data exists in metadata, recreate it
                if (
                    existing_layer is None
                    and "selections" in layer.metadata
                    and new_selection_id_for_comparison
                    in layer.metadata["selections"]
                ):
                    self.create_phasors_selected_layer()
                else:
                    # If layer exists, just update the reference
                    self._phasors_selected_layer = existing_layer
            else:
                # If "None" is selected, set phasors_selected_layer to None
                self._phasors_selected_layer = None

            # Always (re)connect the overlay signal to the current layer
            self._connect_show_overlay_signal()

            processed_selection_id = new_selection_id

            # Only update the plot if we're not processing an initial selection
            if not getattr(self, '_processing_initial_selection', False):
                self.update_phasor_plot_with_selection_id(
                    processed_selection_id
                )
                # update phasor_selected_layer
                if self._phasors_selected_layer is not None:
                    self.update_phasors_layer()

            self._switching_selection_id = False

    def update_phasor_plot_with_selection_id(self, selection_id):
        """Update the phasor plot with the selected ID and show/hide label layers."""
        layer = self._get_current_layer()
        if layer is None:
            return

        # Prevent this from running during plot updates
        if getattr(self.parent_widget, '_updating_plot', False):
            return

        # If selection_id is None, hide all selection layers and clear color indices
        if selection_id is None or selection_id == "":
            # Iterate over all choices in selection_input_widget
            for i in range(
                self.selection_input_widget.phasor_selection_id_combobox.count()
            ):
                sel_id = self.selection_input_widget.phasor_selection_id_combobox.itemText(
                    i
                )
                if sel_id != "None":
                    selection_layer_name = f"Selection {sel_id}: {layer.name}"
                    existing_layer = self._find_phasors_layer_by_name(
                        selection_layer_name
                    )
                    if existing_layer is not None:
                        existing_layer.visible = False

            # Trigger plot update with None to clear selection
            self.parent_widget.plot(selection_id_data=None)
            return

        # Check if the selection_id exists in metadata
        if (
            "selections" not in layer.metadata
            or selection_id not in layer.metadata["selections"]
        ):
            return

        selection_layer_name = f"Selection {selection_id}: {layer.name}"
        selection_layer = self._find_phasors_layer_by_name(
            selection_layer_name
        )
        if selection_layer is None:
            self.create_phasors_selected_layer()
            selection_layer = self._phasors_selected_layer

        if selection_layer:
            selection_layer.visible = True

        # Get selection data for the plot
        # We need to extract the values corresponding to valid pixels
        selection_map = layer.metadata["selections"][selection_id]

        _, _, valid = self.parent_widget.get_masked_gs(
            flat=True, return_valid_mask=True
        )
        if valid is None:
            return

        # Extract selection data for valid pixels
        selection_data = selection_map.ravel()[valid]

        # Trigger plot update with selection data
        self.parent_widget.plot(selection_id_data=selection_data)

    def _get_next_available_selection_id(self):
        """Get the next available manual selection ID."""
        layer = self._get_current_layer()
        if layer is None:
            return "MANUAL SELECTION #1"

        existing_selections = layer.metadata.get("selections", {}).keys()
        counter = 1
        while True:
            candidate_name = f"MANUAL SELECTION #{counter}"
            if candidate_name not in existing_selections:
                return candidate_name
            counter += 1

    def manual_selection_changed(self, manual_selection):
        """Update the manual selection in the layer metadata."""
        layer = self._get_current_layer()
        if layer is None:
            return

        # Add guard to prevent recursive calls
        if getattr(self.parent_widget, '_updating_plot', False):
            return

        # Check if we're in the middle of switching selection IDs
        if getattr(self, '_switching_selection_id', False):
            return

        current_combobox_text = (
            self.selection_input_widget.phasor_selection_id_combobox.currentText()
        )

        # If "None" is selected in combobox, automatically switch to new selection ID
        if current_combobox_text == "None":
            new_selection_id = self._get_next_available_selection_id()

            # Set a flag to indicate we're processing the original manual selection
            self._processing_initial_selection = True
            self._initial_manual_selection = manual_selection

            self._current_selection_id = new_selection_id
            self.selection_id = new_selection_id

        self._ensure_selection_storage(self.selection_id)

        # Get the selection map array
        selection_map = layer.metadata["selections"][self.selection_id]
        selection_map_flat = selection_map.ravel()

        # Mask of valid pixels (where G and S are not NaN)
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

        # Update the selection map
        # manual_selection contains the color indices for the valid pixels
        if selection_to_use is None:
            # Clear selection for valid pixels
            selection_map_flat[valid_pixels_mask] = 0
        else:
            # Map the selection values back to the full image
            # selection_to_use corresponds to the compressed array (valid pixels only)
            # We assign it to the locations in the full array where valid_pixels_mask is True
            selection_map_flat[valid_pixels_mask] = selection_to_use

        self.update_phasors_layer()

    def create_phasors_selected_layer(self):
        """Create the phasors selected layer."""
        layer = self._get_current_layer()
        if layer is None:
            return
        if self.selection_id is None or self.selection_id == "":
            return

        if (
            "selections" not in layer.metadata
            or self.selection_id not in layer.metadata["selections"]
        ):
            return

        selection_map = layer.metadata["selections"][self.selection_id]

        color_dict = colormap_to_dict(
            self.parent_widget._colormap,
            self.parent_widget._colormap.N,
            exclude_first=True,
        )

        layer_name = f"Selection {self.selection_id}: {layer.name}"

        # Create Labels layer directly from the selection map
        phasors_selected_layer = Labels(
            selection_map,
            name=layer_name,
            scale=layer.scale,
            colormap=DirectLabelColormap(
                color_dict=color_dict, name="cat10_mod"
            ),
        )

        self._phasors_selected_layer = self.viewer.add_layer(
            phasors_selected_layer
        )

        # Always (re)connect the overlay signal to the new layer
        self._connect_show_overlay_signal()

    def update_phasors_layer(self):
        """Update the existing phasors layer data without recreating it."""
        layer = self._get_current_layer()
        if layer is None:
            return

        selection_layer_name = f"Selection {self.selection_id}: {layer.name}"
        existing_phasors_selected_layer = self._find_phasors_layer_by_name(
            selection_layer_name
        )

        if existing_phasors_selected_layer is None:
            self.create_phasors_selected_layer()
            return

        if self.selection_id is None or self.selection_id == "":
            # Should probably not happen here, but clear if it does
            existing_phasors_selected_layer.data = np.zeros_like(
                existing_phasors_selected_layer.data
            )
        else:
            if (
                "selections" in layer.metadata
                and self.selection_id in layer.metadata["selections"]
            ):
                existing_phasors_selected_layer.data = layer.metadata[
                    "selections"
                ][self.selection_id]

        self._phasors_selected_layer = existing_phasors_selected_layer
