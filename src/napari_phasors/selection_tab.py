from pathlib import Path

import numpy as np
from napari.layers import Labels
from napari.utils import DirectLabelColormap, notifications
from qtpy import uic
from qtpy.QtWidgets import QVBoxLayout, QWidget
from skimage.util import map_array

from ._utils import colormap_to_dict

#: The columns in the phasor features table that should not be used as selection id.
DATA_COLUMNS = ["label", "G_original", "S_original", "G", "S", "harmonic"]


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
        """Gets or sets the selection id from the phasor selection id combobox.

        Value should not be one of `DATA_COLUMNS`.

        Returns
        -------
        str or None
            The selection id. Returns `None` if no selection id is available, "None" is selected, or empty string.

        """
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

        if new_selection_id != "None" and new_selection_id in DATA_COLUMNS:
            notifications.WarningNotification(
                f"{new_selection_id} is not a valid selection column. It must not be one of {DATA_COLUMNS}."
            )
            return
        else:
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
            # Only add to features if it's not "None"
            if new_selection_id != "None":
                self.add_selection_id_to_features(new_selection_id)

    def add_selection_id_to_features(self, new_selection_id: str):
        """Add a new selection id to the features table in the labels layer with phasor features.

        Parameters
        ----------
        new_selection_id : str
            The new selection id to add to the features table.

        """
        if self.parent_widget._labels_layer_with_phasor_features is None:
            return
        if new_selection_id in DATA_COLUMNS:
            notifications.WarningNotification(
                f"{new_selection_id} is not a valid selection column. It must not be one of {DATA_COLUMNS}."
            )
            return

        if (
            new_selection_id
            not in self.parent_widget._labels_layer_with_phasor_features.features.columns
        ):
            self.parent_widget._labels_layer_with_phasor_features.features[
                new_selection_id
            ] = np.zeros_like(
                self.parent_widget._labels_layer_with_phasor_features.features[
                    "label"
                ].values
            )

    def _find_phasors_layer_by_name(self, layer_name):
        """Find a phasors layer by name in the viewer.

        Parameters
        ----------
        layer_name : str
            The name of the layer to find.

        Returns
        -------
        napari.layers.Layer or None
            The layer if found, None otherwise.
        """
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

    def on_selection_id_changed(self):
        """Callback function when the selection id combobox is changed.

        This function updates the selection and recreates/updates the phasors layer.
        """
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
                self.add_selection_id_to_features(
                    new_selection_id_for_comparison
                )

            # Check if we need to recreate a missing selection layer
            if (
                new_selection_id_for_comparison != "None"
                and self.parent_widget._labels_layer_with_phasor_features
                is not None
            ):

                layer_name = f"Selection: {new_selection_id_for_comparison}"
                existing_layer = self._find_phasors_layer_by_name(layer_name)

                # If layer doesn't exist but column exists in features, recreate it
                if (
                    existing_layer is None
                    and new_selection_id_for_comparison
                    in self.parent_widget._labels_layer_with_phasor_features.features.columns
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
                # update phasor_selected_layer (needed if filtering was applied)
                if self._phasors_selected_layer is not None:
                    self.update_phasors_layer()

            self._switching_selection_id = False

    def update_phasor_plot_with_selection_id(self, selection_id):
        """Update the phasor plot with the selected ID and show/hide label layers."""
        if self.parent_widget._labels_layer_with_phasor_features is None:
            return

        # Prevent this from running during plot updates
        if getattr(self.parent_widget, '_updating_plot', False):
            return

        # If selection_id is None, hide all selection layers and clear color indices
        if selection_id is None or selection_id == "":
            for layer in self.viewer.layers:
                if layer.name.startswith("Selection: "):
                    layer.visible = False

            # Clear color indices only for the active artist
            active_plot_type = self.parent_widget.plot_type
            if active_plot_type in self.parent_widget.canvas_widget.artists:
                self.parent_widget.canvas_widget.artists[
                    active_plot_type
                ].color_indices = 0

            # Trigger plot update to refresh the display
            self.parent_widget.plot()
            return

        # Check if the selection_id column exists in the features table
        if (
            selection_id
            not in self.parent_widget._labels_layer_with_phasor_features.features.columns
        ):
            # Don't create the column or update anything until there's actual selection data
            return

        target_layer_name = f"Selection: {selection_id}"
        for layer in self.viewer.layers:
            if layer.name.startswith("Selection: "):
                layer.visible = layer.name == target_layer_name

        harmonic_mask = (
            self.parent_widget._labels_layer_with_phasor_features.features[
                'harmonic'
            ]
            == self.parent_widget.harmonic
        )
        # Filter rows where 'G' and 'S' is not NaN
        valid_rows = (
            ~self.parent_widget._labels_layer_with_phasor_features.features[
                "G"
            ].isna()
            & ~self.parent_widget._labels_layer_with_phasor_features.features[
                "S"
            ].isna()
            & harmonic_mask
        )

        selection_data = (
            self.parent_widget._labels_layer_with_phasor_features.features.loc[
                valid_rows, selection_id
            ].values
        )

        # Update the color indices only for the active artist
        active_plot_type = self.parent_widget.plot_type
        if active_plot_type in self.parent_widget.canvas_widget.artists:
            self.parent_widget.canvas_widget.artists[
                active_plot_type
            ].color_indices = selection_data

        # Trigger plot update
        self.parent_widget.plot()

    def _get_next_available_selection_id(self):
        """Get the next available manual selection ID.

        Returns
        -------
        str
            The next available selection ID (e.g., "MANUAL SELECTION #1", "MANUAL SELECTION #2", etc.)
        """
        if self.parent_widget._labels_layer_with_phasor_features is None:
            return "MANUAL SELECTION #1"

        existing_columns = (
            self.parent_widget._labels_layer_with_phasor_features.features.columns
        )
        counter = 1
        while True:
            candidate_name = f"MANUAL SELECTION #{counter}"
            if candidate_name not in existing_columns:
                return candidate_name
            counter += 1

    def manual_selection_changed(self, manual_selection):
        """Update the manual selection in the labels layer with phasor features."""
        if self.parent_widget._labels_layer_with_phasor_features is None:
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

        self.add_selection_id_to_features(self.selection_id)
        column = self.selection_id

        self.parent_widget._labels_layer_with_phasor_features.features[
            column
        ] = 0

        # Filter rows where 'G' and 'S' is not NaN
        valid_rows = (
            ~self.parent_widget._labels_layer_with_phasor_features.features[
                "G"
            ].isna()
            & ~self.parent_widget._labels_layer_with_phasor_features.features[
                "S"
            ].isna()
        )
        num_valid_rows = valid_rows.sum()

        selection_to_use = manual_selection
        if (
            hasattr(self, '_processing_initial_selection')
            and self._processing_initial_selection
        ):
            selection_to_use = self._initial_manual_selection
            self._processing_initial_selection = False
            delattr(self, '_initial_manual_selection')

        # Handle case where selection_to_use is None
        if selection_to_use is None:
            # Set all values to 0 (no selection)
            self.parent_widget._labels_layer_with_phasor_features.features.loc[
                valid_rows, column
            ] = 0
        else:
            tiled_manual_selection = np.tile(
                selection_to_use, (num_valid_rows // len(selection_to_use)) + 1
            )[:num_valid_rows]

            self.parent_widget._labels_layer_with_phasor_features.features.loc[
                valid_rows, column
            ] = tiled_manual_selection

        self.update_phasors_layer()

    def create_phasors_selected_layer(self):
        """Create the phasors selected layer."""
        if self.parent_widget._labels_layer_with_phasor_features is None:
            return
        if self.selection_id is None or self.selection_id == "":
            return

        input_array = np.asarray(
            self.parent_widget._labels_layer_with_phasor_features.data
        )
        input_array_values = np.asarray(
            self.parent_widget._labels_layer_with_phasor_features.features[
                "label"
            ].values
        )

        phasors_layer_data = np.asarray(
            self.parent_widget._labels_layer_with_phasor_features.features[
                self.selection_id
            ].values
        )

        mapped_data = map_array(
            input_array, input_array_values, phasors_layer_data
        )
        color_dict = colormap_to_dict(
            self.parent_widget._colormap,
            self.parent_widget._colormap.N,
            exclude_first=True,
        )

        layer_name = f"Selection: {self.selection_id}"
        phasors_selected_layer = Labels(
            mapped_data,
            name=layer_name,
            scale=self.parent_widget._labels_layer_with_phasor_features.scale,
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
        if self.parent_widget._labels_layer_with_phasor_features is None:
            return

        layer_name = f"Selection: {self.selection_id}"
        existing_phasors_selected_layer = self._find_phasors_layer_by_name(
            layer_name
        )

        if existing_phasors_selected_layer is None:
            self.create_phasors_selected_layer()
            return

        input_array = np.asarray(
            self.parent_widget._labels_layer_with_phasor_features.data
        )
        input_array_values = np.asarray(
            self.parent_widget._labels_layer_with_phasor_features.features[
                "label"
            ].values
        )

        if self.selection_id is None or self.selection_id == "":
            phasors_layer_data = np.zeros_like(
                self.parent_widget._labels_layer_with_phasor_features.features[
                    "label"
                ].values
            )
        else:
            phasors_layer_data = np.asarray(
                self.parent_widget._labels_layer_with_phasor_features.features[
                    self.selection_id
                ].values
            ).copy()
        valid_rows = (
            ~self.parent_widget._labels_layer_with_phasor_features.features[
                "G"
            ].isna()
            & ~self.parent_widget._labels_layer_with_phasor_features.features[
                "S"
            ].isna()
        )
        phasors_layer_data[~valid_rows] = 0
        mapped_data = map_array(
            input_array, input_array_values, phasors_layer_data
        )
        existing_phasors_selected_layer.data = mapped_data
        self._phasors_selected_layer = existing_phasors_selected_layer
