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

        self.selection_input_widget = QWidget()
        uic.loadUi(
            Path(__file__).parent / "ui/selection_tab.ui",
            self.selection_input_widget,
        )
        self.selection_input_widget.phasor_selection_id_combobox.addItem(
            "MANUAL SELECTION #1"
        )

        layout = QVBoxLayout(self)
        layout.addWidget(self.selection_input_widget)

        # Initialize the current selection id
        self._current_selection_id = "MANUAL SELECTION #1"
        self.selection_id = "MANUAL SELECTION #1"
        self._phasors_selected_layer = None

        self.selection_input_widget.phasor_selection_id_combobox.currentIndexChanged.connect(
            self.on_selection_id_changed
        )

    @property
    def selection_id(self):
        """Gets or sets the selection id from the phasor selection id combobox.

        Value should not be one of `DATA_COLUMNS`.

        Returns
        -------
        str
            The selection id. Returns `None` if no selection id is available.

        """
        if (
            self.selection_input_widget.phasor_selection_id_combobox.count()
            == 0
        ):
            return None
        else:
            return (
                self.selection_input_widget.phasor_selection_id_combobox.currentText()
            )

    @selection_id.setter
    def selection_id(self, new_selection_id: str):
        """Sets the selection id from the phasor selection id combobox."""
        if new_selection_id in DATA_COLUMNS:
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

    def on_selection_id_changed(self):
        """Callback function when the selection id combobox is changed.

        This function updates the selection and recreates/updates the phasors layer.
        """
        new_selection_id = (
            self.selection_input_widget.phasor_selection_id_combobox.currentText()
        )

        if self._current_selection_id != new_selection_id:
            self._current_selection_id = new_selection_id
            self.add_selection_id_to_features(new_selection_id)

            self.update_phasor_plot_with_selection_id(new_selection_id)

    def update_phasor_plot_with_selection_id(self, selection_id):
        """Update the phasor plot with the selected ID and show/hide label layers."""
        if self.parent_widget._labels_layer_with_phasor_features is None:
            return
        if selection_id is None or selection_id == "":
            return

        target_layer_name = f"Selection: {selection_id}"
        for layer in self.viewer.layers:
            if layer.name.startswith("Selection: "):
                layer.visible = layer.name == target_layer_name

        # Filter rows where 'G' and 'S' is not NaN
        valid_rows = (
            ~self.parent_widget._labels_layer_with_phasor_features.features[
                "G"
            ].isna()
            & ~self.parent_widget._labels_layer_with_phasor_features.features[
                "S"
            ].isna()
        )

        selection_data = (
            self.parent_widget._labels_layer_with_phasor_features.features.loc[
                valid_rows, selection_id
            ].values
        )

        # TODO: Check error with Scatter related to Biaplotter
        # Update the color indices for both SCATTER and HISTOGRAM2D artists
        # if 'SCATTER' in self.parent_widget.canvas_widget.artists:
        #     self.parent_widget.canvas_widget.artists['SCATTER'].color_indices = selection_data

        if 'HISTOGRAM2D' in self.parent_widget.canvas_widget.artists:
            self.parent_widget.canvas_widget.artists[
                'HISTOGRAM2D'
            ].color_indices = selection_data

    def manual_selection_changed(self, manual_selection):
        """Update the manual selection in the labels layer with phasor features."""
        if self.parent_widget._labels_layer_with_phasor_features is None:
            return
        if self.selection_id is None or self.selection_id == "":
            return
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

        tiled_manual_selection = np.tile(
            manual_selection, (num_valid_rows // len(manual_selection)) + 1
        )[:num_valid_rows]
        self.parent_widget._labels_layer_with_phasor_features.features.loc[
            valid_rows, column
        ] = tiled_manual_selection

        self.update_phasors_layer()

    def create_phasors_selected_layer(self):
        """Create the phasors selected layer."""
        if self.parent_widget._labels_layer_with_phasor_features is None:
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

    def update_phasors_layer(self):
        """Update the existing phasors layer data without recreating it."""
        if self.parent_widget._labels_layer_with_phasor_features is None:
            return

        layer_name = f"Selection: {self.selection_id}"
        existing_layer = self._find_phasors_layer_by_name(layer_name)

        if existing_layer is None:
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
            )

        mapped_data = map_array(
            input_array, input_array_values, phasors_layer_data
        )

        existing_layer.data = mapped_data
        self._phasors_selected_layer = existing_layer
