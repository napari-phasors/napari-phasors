import math
from math import ceil, log10
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.ticker as ticker
import numpy as np
from biaplotter.plotter import ArtistType, CanvasWidget
from matplotlib.colorbar import Colorbar
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from napari.layers import Image, Labels
from napari.utils import DirectLabelColormap, colormaps, notifications
from qtpy import uic
from qtpy.QtWidgets import (
    QPushButton,
    QSizePolicy,
    QSpacerItem,
    QVBoxLayout,
    QWidget,
)
from skimage.util import map_array
from superqt import QCollapsible

from ._synthetic_generator import (
    make_intensity_layer_with_phasors,
    make_raw_flim_data,
)
from ._utils import apply_filter_and_threshold, colormap_to_dict

if TYPE_CHECKING:
    import napari

#: The columns in the phasor features table that should not be used as selection id.
DATA_COLUMNS = ["label", "G_original", "S_original", "G", "S", "harmonic"]


class PlotterWidget(QWidget):
    """A widget for plotting phasor features.

    This widget contains a canvas widget and input widgets for plotting phasor features.
    It also creates a phasors selected layer based on the manual selection in the canvas widget.

    Parameters
    ----------
    napari_viewer : napari.Viewer
        The napari viewer object.

    Attributes
    ----------
    viewer : napari.Viewer
        The napari viewer object.
    canvas_widget : biaplotter.plotter.CanvasWidget
        The canvas widget for plotting phasor features.
    plotter_inputs_widget : QWidget
        The main plotter inputs widget. The widget contains:
        - image_layer_with_phasor_features_combobox : QComboBox
            The combobox for selecting the image layer with phasor features.
        - phasor_selection_id_combobox : QComboBox
            The combobox for selecting the phasor selection id.
        - harmonic_spinbox : QSpinBox
            The spinbox for selecting the harmonic.
        - threshold_slider : QSlider
            The slider for selecting the threshold.
        - median_filter_spinbox : QSpinBox
            The spinbox for selecting the median filter kernel size (in pixels).
        - semi_circle_checkbox : QCheckBox
            The checkbox for displaying the universal semi-circle (if True) or the full polar plot (if False).
    extra_inputs_widget : QWidget
        The extra plotter inputs widget. It is collapsible. The widget contains:
        - plot_type_combobox : QComboBox
            The combobox for selecting the plot type.
        - colormap_combobox : QComboBox
            The combobox for selecting the histogram colormap.
        - number_of_bins_spinbox : QSpinBox
            The spinbox for selecting the number of bins in the histogram.
        - log_scale_checkbox : QCheckBox
            The checkbox for selecting the log scale in the histogram.
    plot_button : QPushButton
        The plot button.
    _labels_layer_with_phasor_features : Labels
        The labels layer with phasor features.
    _phasors_selected_layer : Labels
        The phasors selected layer.
    _colormap : matplotlib.colors.Colormap
        The colormap for the canvas widget.
    _histogram_colormap : matplotlib.colors.Colormap
        The histogram colormap for the canvas widget.

    """

    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        self.setLayout(QVBoxLayout())

        # Load canvas widget
        self.canvas_widget = CanvasWidget(napari_viewer)
        self.canvas_widget.axes.set_aspect(1, adjustable='box')
        self.canvas_widget.setMinimumSize(300, 300)
        self.canvas_widget.class_spinbox.setValue(1)
        self.set_axes_labels()
        self.layout().addWidget(self.canvas_widget)

        # Load plotter inputs widget from ui file
        self.plotter_inputs_widget = QWidget()
        uic.loadUi(
            Path(__file__).parent / "ui/plotter_inputs_widget.ui",
            self.plotter_inputs_widget,
        )
        self.layout().addWidget(self.plotter_inputs_widget)

        # Add collapsible widget
        collapsible_widget = QCollapsible("Extra Options")
        self.layout().addWidget(collapsible_widget)

        # Load extra inputs widget from ui file
        self.extra_inputs_widget = QWidget()
        uic.loadUi(
            Path(__file__).parent / "ui/plotter_inputs_widget_extra.ui",
            self.extra_inputs_widget,
        )
        collapsible_widget.addWidget(self.extra_inputs_widget)
        # Add plot button
        self.plot_button = QPushButton("Plot")
        self.layout().addWidget(self.plot_button)

        # Add a vertical spacer at the bottom
        spacer = QSpacerItem(
            20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding
        )
        self.layout().addItem(spacer)

        # Set minimum size
        self.setMinimumSize(600, 300)

        # Connect napari signals when new layer is inseted or removed
        self.viewer.layers.events.inserted.connect(self.reset_layer_choices)
        self.viewer.layers.events.removed.connect(self.reset_layer_choices)

        # Connect callbacks
        self.plotter_inputs_widget.image_layer_with_phasor_features_combobox.currentIndexChanged.connect(
            self.on_labels_layer_with_phasor_features_changed
        )
        self.plotter_inputs_widget.phasor_selection_id_combobox.currentIndexChanged.connect(
            self.on_selection_id_changed
        )
        self.plotter_inputs_widget.semi_circle_checkbox.stateChanged.connect(
            self.on_toggle_semi_circle
        )
        self.plot_button.clicked.connect(self.plot)

        # Populate plot type combobox
        self.extra_inputs_widget.plot_type_combobox.addItems(
            [ArtistType.SCATTER.name, ArtistType.HISTOGRAM2D.name]
        )
        # Populate colormap combobox
        self.extra_inputs_widget.colormap_combobox.addItems(
            list(colormaps.ALL_COLORMAPS.keys())
        )
        self.histogram_colormap = (
            "turbo"  # Set default colormap (same as in biaplotter)
        )

        # Connect canvas signals
        self.canvas_widget.artists[
            ArtistType.SCATTER
        ].color_indices_changed_signal.connect(self.manual_selection_changed)
        self.canvas_widget.artists[
            ArtistType.HISTOGRAM2D
        ].color_indices_changed_signal.connect(self.manual_selection_changed)

        # Initialize attributes
        self._labels_layer_with_phasor_features = None
        self.selection_id = "MANUAL SELECTION #1"
        self._phasors_selected_layer = None
        self.polar_plot_artist_list = []
        self.semi_circle_plot_artist_list = []
        self.toggle_semi_circle = True
        self.colorbar = None
        self._colormap = self.canvas_widget.artists[
            ArtistType.HISTOGRAM2D
        ].categorical_colormap
        self._histogram_colormap = self.canvas_widget.artists[
            ArtistType.HISTOGRAM2D
        ].histogram_colormap
        # Start with the histogram2d plot type
        self.plot_type = ArtistType.HISTOGRAM2D.name

        # Set intial axes limits
        self._redefine_axes_limits()
        # Set initial background color
        self._update_plot_bg_color(color="white")
        # Populate labels layer combobox
        self.reset_layer_choices()

        # Connect threshold slider
        self.plotter_inputs_widget.threshold_slider.valueChanged.connect(
            self.on_threshold_slider_change
        )

        # Connect kernel size spinbox
        self.plotter_inputs_widget.median_filter_spinbox.valueChanged.connect(
            self.on_kernel_size_change
        )

    @property
    def selection_id(self):
        """Gets or sets the selection id from the phasor selection id combobox.

        Value should not be one of these: ['label', 'Average Image', 'G', 'S', 'harmonic'].

        Returns
        -------
        str
            The selection id. Returns `None` if no selection id is available.
        """
        if (
            self.plotter_inputs_widget.phasor_selection_id_combobox.count()
            == 0
        ):
            return None
        else:
            return (
                self.plotter_inputs_widget.phasor_selection_id_combobox.currentText()
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
                self.plotter_inputs_widget.phasor_selection_id_combobox.itemText(
                    i
                )
                for i in range(
                    self.plotter_inputs_widget.phasor_selection_id_combobox.count()
                )
            ]:
                self.plotter_inputs_widget.phasor_selection_id_combobox.addItem(
                    new_selection_id
                )
            self.plotter_inputs_widget.phasor_selection_id_combobox.setCurrentText(
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
        if self._labels_layer_with_phasor_features is None:
            return
        if new_selection_id in DATA_COLUMNS:
            notifications.WarningNotification(
                f"{new_selection_id} is not a valid selection column. It must not be one of {DATA_COLUMNS}."
            )
            return
        # If column_name is not in features, add it with zeros
        if (
            new_selection_id
            not in self._labels_layer_with_phasor_features.features.columns
        ):
            self._labels_layer_with_phasor_features.features[
                new_selection_id
            ] = np.zeros_like(
                self._labels_layer_with_phasor_features.features[
                    "label"
                ].values
            )

    def on_selection_id_changed(self):
        """Callback function when the phasor selection id combobox is changed.

        This function updates the `selection_id` attribute with the selected text from the combobox.
        """
        self.selection_id = (
            self.plotter_inputs_widget.phasor_selection_id_combobox.currentText()
        )

    @property
    def harmonic(self):
        """Gets or sets the harmonic value from the harmonic spinbox.

        Returns
        -------
        int
            The harmonic value.
        """
        return self.plotter_inputs_widget.harmonic_spinbox.value()

    @harmonic.setter
    def harmonic(self, value: int):
        """Sets the harmonic value from the harmonic spinbox."""
        if value < 1:
            notifications.WarningNotification(
                f"Harmonic value should be greater than 0. Setting to 1."
            )
            value = 1
        self.plotter_inputs_widget.harmonic_spinbox.setValue(value)

    @property
    def toggle_semi_circle(self):
        """Gets the display semi circle value from the semi circle checkbox.

        Returns
        -------
        bool
            The display semi circle value.
        """
        return self.plotter_inputs_widget.semi_circle_checkbox.isChecked()

    @toggle_semi_circle.setter
    def toggle_semi_circle(self, value: bool):
        """Sets the display semi circle value from the semi circle checkbox."""
        self.plotter_inputs_widget.semi_circle_checkbox.setChecked(value)
        if value:
            self._update_polar_plot(self.canvas_widget.axes, visible=False)
            self._update_semi_circle_plot(self.canvas_widget.axes)
        else:
            self._update_semi_circle_plot(
                self.canvas_widget.axes, visible=False
            )
            self._update_polar_plot(self.canvas_widget.axes)
        self.canvas_widget.axes.set_aspect(1, adjustable='box')
        self._redefine_axes_limits()

    def on_toggle_semi_circle(self, state):
        """Callback function when the semi circle checkbox is toggled.

        This function updates the `toggle_semi_circle` attribute with the checked status of the checkbox.
        And it displays either the universal semi-circle or the full polar plot in the canvas widget.
        """
        self.toggle_semi_circle = state

    def _update_polar_plot(self, ax, visible=True, alpha=0.3, zorder=3):
        """
        Generate the polar plot in the canvas widget.

        Build the inner and outer circle and the 45 degrees lines in the plot.
        """
        if len(self.polar_plot_artist_list) > 0:
            for artist in self.polar_plot_artist_list:
                artist.set_visible(visible)
                artist.set_alpha(alpha)
        else:
            self.polar_plot_artist_list.append(
                ax.add_line(
                    Line2D(
                        [-1, 1],
                        [0, 0],
                        linestyle='-',
                        linewidth=1,
                        color='black',
                    )
                )
            )
            self.polar_plot_artist_list.append(
                ax.add_line(
                    Line2D(
                        [0, 0],
                        [-1, 1],
                        linestyle='-',
                        linewidth=1,
                        color='black',
                    )
                )
            )
            self.polar_plot_artist_list.append(
                ax.add_patch(Circle((0, 0), 1, fill=False))
            )
            for r in (1 / 3, 2 / 3):
                self.polar_plot_artist_list.append(
                    ax.add_patch(Circle((0, 0), r, fill=False))
                )
            for a in (3, 6):
                x = math.cos(math.pi / a)
                y = math.sin(math.pi / a)
                self.polar_plot_artist_list.append(
                    ax.add_line(
                        Line2D(
                            [-x, x],
                            [-y, y],
                            linestyle=':',
                            linewidth=0.5,
                            color='black',
                        )
                    )
                )
                self.polar_plot_artist_list.append(
                    ax.add_line(
                        Line2D(
                            [-x, x],
                            [y, -y],
                            linestyle=':',
                            linewidth=0.5,
                            color='black',
                        )
                    )
                )
        return ax

    def _update_semi_circle_plot(self, ax, visible=True, alpha=0.3, zorder=3):
        '''
        Generate FLIM universal semi-circle plot
        '''
        if len(self.semi_circle_plot_artist_list) > 0:
            for artist in self.semi_circle_plot_artist_list:
                artist.set_visible(visible)
                artist.set_alpha(alpha)
        else:
            angles = np.linspace(0, np.pi, 180)
            x = (np.cos(angles) + 1) / 2
            y = np.sin(angles) / 2
            self.semi_circle_plot_artist_list.append(
                ax.plot(
                    x,
                    y,
                    'black',
                    alpha=alpha,
                    visible=visible,
                    zorder=zorder,
                )[0]
            )
            self.semi_circle_plot_artist_list.append(
                ax.plot(
                    [0, 1],
                    [0, 0],
                    color='black',
                    alpha=alpha,
                    visible=visible,
                    zorder=zorder,
                )[0]
            )
        return ax

    def _redefine_axes_limits(self, ensure_full_circle_displayed=True):
        """
        Redefine axes limits based on the data plotted in the canvas widget.

        Parameters
        ----------
        ensure_full_circle_displayed : bool, optional
            Whether to ensure the full circle is displayed in the canvas widget, by default True.
        """
        # Redefine axes limits
        if self.toggle_semi_circle:
            # Get semi circle plot limits
            circle_plot_limits = [0, 1, 0, 0.5]  # xmin, xmax, ymin, ymax
        else:
            # Get polar plot limits
            circle_plot_limits = [-1, 1, -1, 1]  # xmin, xmax, ymin, ymax
        # Check if histogram is plotted
        if (
            self.canvas_widget.artists[ArtistType.HISTOGRAM2D].histogram
            is not None
        ):
            # Get histogram data limits
            histogram_limits = (
                self.canvas_widget.artists[ArtistType.HISTOGRAM2D]
                .histogram[-1]
                .get_datalim(self.canvas_widget.axes.transData)
            )
            plotted_data_limits = [
                histogram_limits.x0,
                histogram_limits.x1,
                histogram_limits.y0,
                histogram_limits.y1,
            ]
        else:
            plotted_data_limits = circle_plot_limits
        # Check if full circle should be displayed
        if not ensure_full_circle_displayed:
            # If not, only the data limits are used
            circle_plot_limits = plotted_data_limits

        x_range = np.amax(
            [plotted_data_limits[1], circle_plot_limits[1]]
        ) - np.amin([plotted_data_limits[0], circle_plot_limits[0]])
        y_range = np.amax(
            [plotted_data_limits[3], circle_plot_limits[3]]
        ) - np.amin([plotted_data_limits[2], circle_plot_limits[2]])
        # 10% of the range as a frame
        xlim_0 = (
            np.amin([plotted_data_limits[0], circle_plot_limits[0]])
            - 0.1 * x_range
        )
        xlim_1 = (
            np.amax([plotted_data_limits[1], circle_plot_limits[1]])
            + 0.1 * x_range
        )
        ylim_0 = (
            np.amin([plotted_data_limits[2], circle_plot_limits[2]])
            - 0.1 * y_range
        )
        ylim_1 = (
            np.amax([plotted_data_limits[3], circle_plot_limits[3]])
            + 0.1 * y_range
        )

        self.canvas_widget.axes.set_ylim([ylim_0, ylim_1])
        self.canvas_widget.axes.set_xlim([xlim_0, xlim_1])
        self.canvas_widget.figure.canvas.draw_idle()

    def _update_plot_bg_color(self, color=None):
        """Change the background color of the canvas widget.

        Parameters
        ----------
        color : str, optional
            The color to set the background, by default None. If None, the first color of the histogram colormap is used.
        """
        if color is None:
            self.canvas_widget.axes.set_facecolor(
                self.canvas_widget.artists[
                    ArtistType.HISTOGRAM2D
                ].histogram_colormap(0)
            )
        else:
            self.canvas_widget.axes.set_facecolor(color)
        self.canvas_widget.figure.canvas.draw_idle()

    @property
    def plot_type(self):
        """Gets or sets the plot type from the plot type combobox.

        Returns
        -------
        str
            The plot type.
        """
        return self.extra_inputs_widget.plot_type_combobox.currentText()

    @plot_type.setter
    def plot_type(self, type):
        """Sets the plot type from the plot type combobox."""
        self.extra_inputs_widget.plot_type_combobox.setCurrentText(type)

    @property
    def histogram_colormap(self):
        """Gets or sets the histogram colormap from the colormap combobox.

        Returns
        -------
        str
            The colormap name.
        """
        return self.extra_inputs_widget.colormap_combobox.currentText()

    @histogram_colormap.setter
    def histogram_colormap(self, colormap: str):
        """Sets the histogram colormap from the colormap combobox."""
        if colormap not in colormaps.ALL_COLORMAPS.keys():
            notifications.WarningNotification(
                f"{colormap} is not a valid colormap. Setting to default colormap."
            )
            colormap = self._histogram_colormap.name
        self.extra_inputs_widget.colormap_combobox.setCurrentText(colormap)

    @property
    def histogram_bins(self):
        """Gets the histogram bins from the histogram bins spinbox.

        Returns
        -------
        int
            The histogram bins value.
        """
        return self.extra_inputs_widget.number_of_bins_spinbox.value()

    @histogram_bins.setter
    def histogram_bins(self, value: int):
        """Sets the histogram bins from the histogram bins spinbox."""
        if value < 2:
            notifications.WarningNotification(
                f"Number of bins should be greater than 1. Setting to 10."
            )
            value = 10
        self.extra_inputs_widget.number_of_bins_spinbox.setValue(value)

    @property
    def histogram_log_scale(self):
        """Gets the histogram log scale from the histogram log scale checkbox.

        Returns
        -------
        bool
            The histogram log scale value.
        """
        return self.extra_inputs_widget.log_scale_checkbox.isChecked()

    @histogram_log_scale.setter
    def histogram_log_scale(self, value: bool):
        """Sets the histogram log scale from the histogram log scale checkbox."""
        self.extra_inputs_widget.log_scale_checkbox.setChecked(value)

    def manual_selection_changed(self, manual_selection):
        """Update the manual selection in the labels layer with phasor features.

        This method serves as a Slot for the `color_indices_changed_signal` emitted by the canvas widget.
        It should receive the `color_indices` array from the active artist in the canvas widget.
        It also updates/creates the phasors selected layer by calling the `create_phasors_selected_layer` method.

        Parameters
        ----------
        manual_selection : np.ndarray
            The manual selection array.
        """
        if self._labels_layer_with_phasor_features is None:
            return
        if self.selection_id is None or self.selection_id == "":
            return
        column = self.selection_id
        # Update the manual selection in the labels layer with phasor features for each harmonic
        self._labels_layer_with_phasor_features.features[column] = 0
        # Filter rows where 'G' is not NaN
        valid_rows = (
            ~self._labels_layer_with_phasor_features.features["G"].isna()
            & ~self._labels_layer_with_phasor_features.features["S"].isna()
        )
        num_valid_rows = valid_rows.sum()
        # Tile the manual_selection array to match the number of valid rows
        tiled_manual_selection = np.tile(
            manual_selection, (num_valid_rows // len(manual_selection)) + 1
        )[:num_valid_rows]
        self._labels_layer_with_phasor_features.features.loc[
            valid_rows, column
        ] = tiled_manual_selection
        self.create_phasors_selected_layer()

    def reset_layer_choices(self):
        """Reset the image layer with phasor features combobox choices.

        This function is called when a new layer is added or removed.
        It also updates `_labels_layer_with_phasor_features` attribute with the Labels layer in the metadata of the selected image layer.
        """
        self.plotter_inputs_widget.image_layer_with_phasor_features_combobox.clear()
        layer_names = [
            layer.name
            for layer in self.viewer.layers
            if isinstance(layer, Image)
            and "phasor_features_labels_layer" in layer.metadata.keys()
        ]
        self.plotter_inputs_widget.image_layer_with_phasor_features_combobox.addItems(
            layer_names
        )
        # Update layer names in the phasor selection id combobox when layer name changes
        for layer_name in layer_names:
            layer = self.viewer.layers[layer_name]
            layer.events.name.connect(self.reset_layer_choices)
        self.on_labels_layer_with_phasor_features_changed()

    def on_labels_layer_with_phasor_features_changed(self):
        """Callback function when the image layer with phasor features combobox is changed.

        This function updates the `_labels_layer_with_phasor_features` attribute with the Labels layer in the metadata of the selected image layer.
        """
        labels_layer_name = (
            self.plotter_inputs_widget.image_layer_with_phasor_features_combobox.currentText()
        )
        if labels_layer_name == "":
            self._labels_layer_with_phasor_features = None
            return
        layer_metadata = self.viewer.layers[labels_layer_name].metadata
        self._labels_layer_with_phasor_features = layer_metadata[
            "phasor_features_labels_layer"
        ]
        # Set harmonic spinbox maximum value based on maximum harmonic in the table
        self.plotter_inputs_widget.harmonic_spinbox.setMaximum(
            self._labels_layer_with_phasor_features.features["harmonic"].max()
        )
        max_mean_value = np.nanmax(layer_metadata["original_mean"])
        # Determine the threshold factor based on max_mean_value using logarithmic scaling
        if max_mean_value > 0:
            magnitude = int(log10(max_mean_value))
            self.threshold_factor = (
                10 ** (2 - magnitude) if magnitude <= 2 else 1
            )
        else:
            self.threshold_factor = 1  # Default case for values less than 1
        # Set threshold slider maximum value based on maximum mean
        self.plotter_inputs_widget.threshold_slider.setMaximum(
            ceil(max_mean_value * self.threshold_factor)
        )
        if "settings" in layer_metadata.keys():
            settings = layer_metadata["settings"]
            if "threshold" in settings.keys():
                self.plotter_inputs_widget.threshold_slider.setValue(
                    int(settings["threshold"] * self.threshold_factor)
                )
                self.on_threshold_slider_change()
            else:
                self.plotter_inputs_widget.threshold_slider.setValue(
                    int(max_mean_value * 0.1 * self.threshold_factor)
                )
                self.on_threshold_slider_change()
            if "filter" in settings.keys():
                self.plotter_inputs_widget.median_filter_spinbox.setValue(
                    int(settings["filter"]["size"])
                )
                self.plotter_inputs_widget.median_filter_repetition_spinbox.setValue(
                    int(settings["filter"]["repeat"])
                )
        else:
            self.plotter_inputs_widget.threshold_slider.setValue(
                int(max_mean_value * 0.1 * self.threshold_factor)
            )
            self.on_threshold_slider_change()
        # Add default selection id to table if not present
        self.add_selection_id_to_features("MANUAL SELECTION #1")

    def get_features(self):
        """Get the G and S features for the selected harmonic and selection id.

        Returns
        -------
        x_data : np.ndarray
            The G feature data.
        y_data : np.ndarray
            The S feature data.
        selection_data : np.ndarray
            The selection data.
        """
        if self._labels_layer_with_phasor_features is None:
            return None
        # Check if layer contains features
        if self._labels_layer_with_phasor_features.features is None:
            return None
        table = self._labels_layer_with_phasor_features.features
        x_data = table['G'][table['harmonic'] == self.harmonic].values
        y_data = table['S'][table['harmonic'] == self.harmonic].values
        mask = np.isnan(x_data) & np.isnan(y_data)
        x_data = x_data[~mask]
        y_data = y_data[~mask]
        if self.selection_id is None or self.selection_id == "":
            return x_data, y_data, np.zeros_like(x_data)
        else:
            selection_data = table[self.selection_id][
                table['harmonic'] == self.harmonic
            ].values
            selection_data = selection_data[~mask]

        return x_data, y_data, selection_data

    def set_axes_labels(self):
        """Set the axes labels in the canvas widget."""
        self.canvas_widget.artists[ArtistType.SCATTER].ax.set_xlabel(
            "G", color="white"
        )
        self.canvas_widget.artists[ArtistType.SCATTER].ax.set_ylabel(
            "S", color="white"
        )
        self.canvas_widget.artists[ArtistType.HISTOGRAM2D].ax.set_xlabel(
            "G", color="white"
        )
        self.canvas_widget.artists[ArtistType.HISTOGRAM2D].ax.set_ylabel(
            "S", color="white"
        )

    def plot(self):
        """Plot the selected phasor features.

        This function plots the selected phasor features in the canvas widget.
        It also creates the phasors selected layer.
        """
        labels_layer_name = (
            self.plotter_inputs_widget.image_layer_with_phasor_features_combobox.currentText()
        )
        apply_filter_and_threshold(
            self.viewer.layers[labels_layer_name],
            threshold=self.plotter_inputs_widget.threshold_slider.value()
            / self.threshold_factor,
            size=self.plotter_inputs_widget.median_filter_spinbox.value(),
            repeat=self.plotter_inputs_widget.median_filter_repetition_spinbox.value(),
        )
        x_data, y_data, selection_id_data = self.get_features()
        # Set active artist
        self.canvas_widget.active_artist = self.canvas_widget.artists[
            ArtistType[self.plot_type]
        ]
        self.canvas_widget.active_artist.ax.set_facecolor('white')
        self.canvas_widget.artists[ArtistType.HISTOGRAM2D].cmin = 1
        # Set data in the active artist
        self.canvas_widget.active_artist.data = np.column_stack(
            (x_data, y_data)
        )
        # Set selection data in the active artist
        self.canvas_widget.active_artist.color_indices = selection_id_data
        # Set colormap in the active artist
        selected_histogram_colormap = colormaps.ALL_COLORMAPS[
            self.histogram_colormap
        ]
        # Temporary convertion to LinearSegmentedColormap to match matplotlib format, while biaplotter is not updated
        selected_histogram_colormap = LinearSegmentedColormap.from_list(
            selected_histogram_colormap.name,
            selected_histogram_colormap.colors,
        )
        self.canvas_widget.artists[
            ArtistType.HISTOGRAM2D
        ].histogram_colormap = selected_histogram_colormap
        # Set number of bins in the active artist
        self.canvas_widget.artists[ArtistType.HISTOGRAM2D].bins = (
            self.histogram_bins
        )
        # Temporarily set active artist "again" to have it displayed on top #TODO: Fix this
        self.canvas_widget.active_artist = self.canvas_widget.artists[
            ArtistType[self.plot_type]
        ]
        # Set log scale in the active artist
        if (
            self.canvas_widget.artists[ArtistType.HISTOGRAM2D].histogram
            is not None
        ):
            h = self.canvas_widget.artists[ArtistType.HISTOGRAM2D].histogram[0]
            if self.histogram_log_scale:
                # Set min_value to a small positive number
                vmin = max(h[h > 0].min(), 1e-10) if np.any(h > 0) else 1
                vmax = np.nanmax(h)  # Finds the maximum, ignoring NaNs
                # Adjust vmax if it's NaN or insufficient for log scaling
                if np.isnan(vmax) or vmax <= vmin:
                    vmax = vmin * 2
                if np.all(h <= 1):
                    norm = Normalize(
                        vmin=vmin, vmax=vmax
                    )  # Use linear for binary data (0s and 1s)
                else:
                    norm = LogNorm(vmin=vmin, vmax=vmax)
            else:
                vmin = np.nanmin(h)
                vmax = np.nanmax(h)
                norm = Normalize(vmin=vmin, vmax=vmax)
            self.canvas_widget.artists[ArtistType.HISTOGRAM2D].histogram[
                -1
            ].set_norm(norm)

        # if active artist is histogram, add a colorbar
        if self.plot_type == ArtistType.HISTOGRAM2D.name:
            if self.colorbar is not None:
                self.colorbar.remove()
            # Create cax for colorbar on the right side of the histogram
            self.cax = self.canvas_widget.artists[
                ArtistType.HISTOGRAM2D
            ].ax.inset_axes([1.05, 0, 0.05, 1])
            # Create colorbar
            self.colorbar = Colorbar(
                ax=self.cax,
                cmap=self.canvas_widget.artists[
                    ArtistType.HISTOGRAM2D
                ].histogram_colormap,
                norm=norm,
            )

            self.set_colorbar_style(color="white")
        else:
            if self.colorbar is not None:
                # remove colorbar
                self.colorbar.remove()
                self.colorbar = None
        # Update axes limits
        self._redefine_axes_limits()
        self.canvas_widget.axes.set_aspect(1, adjustable='box')
        self._update_plot_bg_color(color="white")
        self.create_phasors_selected_layer()

    def set_colorbar_style(self, color="white"):
        """Set the colorbar style in the canvas widget."""
        # Set colorbar tick color
        self.colorbar.ax.yaxis.set_tick_params(color=color)
        # Set colorbar edgecolor
        self.colorbar.outline.set_edgecolor(color)
        # Add label to colorbar
        if isinstance(self.colorbar.norm, LogNorm):
            self.colorbar.ax.set_ylabel("Log10(Count)", color=color)
        else:
            self.colorbar.ax.set_ylabel("Count", color=color)
        # Get the current ticks
        ticks = self.colorbar.ax.get_yticks()
        # Set the ticks using a FixedLocator
        self.colorbar.ax.yaxis.set_major_locator(ticker.FixedLocator(ticks))
        # Set colorbar ticklabels colors individually (this may fail for some math expressions)
        tick_labels = self.colorbar.ax.get_yticklabels()
        for tick_label in tick_labels:
            tick_label.set_color(color)

    def create_phasors_selected_layer(self):
        """Create or update the phasors selected layer."""
        if self._labels_layer_with_phasor_features is None:
            return
        input_array = np.asarray(self._labels_layer_with_phasor_features.data)
        input_array_values = np.asarray(
            self._labels_layer_with_phasor_features.features["label"].values
        )
        # If no selection id is provided, set all pixels to 0
        if self.selection_id is None or self.selection_id == "":
            phasors_layer_data = np.zeros_like(
                self._labels_layer_with_phasor_features.features[
                    "label"
                ].values
            )
        else:
            phasors_layer_data = np.asarray(
                self._labels_layer_with_phasor_features.features[
                    self.selection_id
                ].values
            )

        mapped_data = map_array(
            input_array, input_array_values, phasors_layer_data
        )
        color_dict = colormap_to_dict(
            self._colormap, self._colormap.N, exclude_first=True
        )
        # Build output phasors Labels layer
        phasors_selected_layer = Labels(
            mapped_data,
            name="Phasors Selected",
            scale=self._labels_layer_with_phasor_features.scale,
            colormap=DirectLabelColormap(
                color_dict=color_dict, name="cat10_mod"
            ),
        )
        if self._phasors_selected_layer is None:
            self._phasors_selected_layer = self.viewer.add_layer(
                phasors_selected_layer
            )
        else:
            self._phasors_selected_layer.data = mapped_data
            self._phasors_selected_layer.scale = (
                self._labels_layer_with_phasor_features.scale
            )

    def on_threshold_slider_change(self):
        self.plotter_inputs_widget.label_3.setText(
            'Intensity threshold: '
            + str(
                self.plotter_inputs_widget.threshold_slider.value()
                / self.threshold_factor
            )
        )

    def on_kernel_size_change(self):
        kernel_value = self.plotter_inputs_widget.median_filter_spinbox.value()
        self.plotter_inputs_widget.label_4.setText(
            'Median Filter Kernel Size: ' + f'{kernel_value} x {kernel_value}'
        )


if __name__ == "__main__":
    import napari

    time_constants = [0.1, 1, 2, 3, 4, 5, 10]
    raw_flim_data = make_raw_flim_data(time_constants=time_constants)
    harmonic = [1, 2, 3]
    intensity_image_layer = make_intensity_layer_with_phasors(
        raw_flim_data, harmonic=harmonic
    )
    viewer = napari.Viewer()
    viewer.add_layer(intensity_image_layer)
    plotter = PlotterWidget(viewer)
    viewer.window.add_dock_widget(plotter, area="right")
    napari.run()
