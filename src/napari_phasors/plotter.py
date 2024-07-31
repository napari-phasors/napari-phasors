import numpy as np

from pathlib import Path
from typing import TYPE_CHECKING
from skimage.util import map_array
from napari.utils import (
    notifications,
    DirectLabelColormap,
    colormaps
)
from napari.layers import Labels, Image
from biaplotter.plotter import (
    CanvasWidget,
    ArtistType
)
from superqt import QCollapsible
from qtpy import uic
from qtpy.QtWidgets import (
    QVBoxLayout,
    QWidget,
    QSpacerItem,
    QSizePolicy,
    QPushButton,
)
from matplotlib.colors import (
    LinearSegmentedColormap,
    LogNorm, 
    Normalize
)

from napari_phasors._synthetic_generator import (
    make_raw_flim_data,
    make_intensity_layer_with_phasors
)

if TYPE_CHECKING:
    import napari

#: The columns in the phasor features table that should not be used as selection id.
DATA_COLUMNS = ['label', 'Average Image', 'G', 'S', 'harmonic']

def colormap_to_dict(colormap, num_colors=10, exclude_first=True):
    """
    Converts a matplotlib colormap into a dictionary of RGBA colors.

    Parameters
    ----------
    colormap : matplotlib.colors.Colormap
        The colormap to convert.
    num_colors : int, optional
        The number of colors in the colormap, by default 10.
    exclude_first : bool, optional
        Whether to exclude the first color in the colormap, by default True.

    Returns
    -------
    color_dict: dict
        A dictionary with keys as positive integers and values as RGBA colors.
    """
    color_dict = {}
    start = 0
    if exclude_first:
        start = 1
    for i in range(start, num_colors + start):
        pos = i / (num_colors - 1)
        color = colormap(pos)
        color_dict[i+1-start] = color
    color_dict[None] = (0, 0, 0, 0)
    return color_dict

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
        spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.layout().addItem(spacer)

        # Connect napari signals when new layer is inseted or removed
        self.viewer.layers.events.inserted.connect(self.reset_layer_choices)
        self.viewer.layers.events.removed.connect(self.reset_layer_choices)

        # Connect callbacks
        self.plotter_inputs_widget.image_layer_with_phasor_features_combobox.currentIndexChanged.connect(
            self.on_labels_layer_with_phasor_features_changed)
        self.plotter_inputs_widget.phasor_selection_id_combobox.currentIndexChanged.connect(
            self.on_selection_id_changed)
        self.plot_button.clicked.connect(self.plot)
        
        # Populate plot type combobox
        self.extra_inputs_widget.plot_type_combobox.addItems(
            [ArtistType.SCATTER.name, ArtistType.HISTOGRAM2D.name]
        )
        # Populate colormap combobox
        self.extra_inputs_widget.colormap_combobox.addItems(list(colormaps.ALL_COLORMAPS.keys()))
        self.histogram_colormap = 'magma' # Set default colormap (same as in biaplotter)

        # Connect canvas signals
        self.canvas_widget.artists[ArtistType.SCATTER].color_indices_changed_signal.connect(self.manual_selection_changed)
        self.canvas_widget.artists[ArtistType.HISTOGRAM2D].color_indices_changed_signal.connect(self.manual_selection_changed)

        # Initialize attributes
        self._labels_layer_with_phasor_features = None
        self._phasors_selected_layer = None
        self._colormap = self.canvas_widget.artists[ArtistType.HISTOGRAM2D].categorical_colormap
        self._histogram_colormap = self.canvas_widget.artists[ArtistType.HISTOGRAM2D].histogram_colormap
        # Start with the histogram2d plot type
        self.plot_type = ArtistType.HISTOGRAM2D.name

        # Populate labels layer combobox
        self.reset_layer_choices()

    @property
    def selection_id(self):
        """Gets or sets the selection id from the phasor selection id combobox.

        Value should not be one of these: ['label', 'Average Image', 'G', 'S', 'harmonic'].

        Returns
        -------
        str
            The selection id. Returns `None` if no selection id is available.
        """
        if self.plotter_inputs_widget.phasor_selection_id_combobox.count() == 0:
            return None
        else:
            return self.plotter_inputs_widget.phasor_selection_id_combobox.currentText()

    @selection_id.setter
    def selection_id(self, new_selection_id: str):
        """Sets the selection id from the phasor selection id combobox."""

        if self._labels_layer_with_phasor_features is None:
            notifications.WarningNotification("No labels layer with phasor features selected.")
            return
        if new_selection_id in DATA_COLUMNS:
            notifications.WarningNotification(f"{new_selection_id} is not a valid selection column. It must not be one of {DATA_COLUMNS}.")
            return
        else:
            if new_selection_id not in [self.plotter_inputs_widget.phasor_selection_id_combobox.itemText(i) for i in range(self.plotter_inputs_widget.phasor_selection_id_combobox.count())]:
                self.plotter_inputs_widget.phasor_selection_id_combobox.addItem(new_selection_id)
            self.plotter_inputs_widget.phasor_selection_id_combobox.setCurrentText(new_selection_id)
            # If column_name is not in features, add it with zeros
            if new_selection_id not in self._labels_layer_with_phasor_features.features.columns:
                self._labels_layer_with_phasor_features.features[new_selection_id] = np.zeros_like(self._labels_layer_with_phasor_features.features['label'].values)

    def on_selection_id_changed(self):
        """Callback function when the phasor selection id combobox is changed.

        This function updates the `selection_id` attribute with the selected text from the combobox.
        """
        self.selection_id = self.plotter_inputs_widget.phasor_selection_id_combobox.currentText()
    
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
            notifications.WarningNotification(f"Harmonic value should be greater than 0. Setting to 1.")
            value = 1
        self.plotter_inputs_widget.harmonic_spinbox.setValue(value)

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
            notifications.WarningNotification(f"{colormap} is not a valid colormap. Setting to default colormap.")
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
            notifications.WarningNotification(f"Number of bins should be greater than 1. Setting to 10.")
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
            notifications.WarningNotification("No labels layer with phasor features selected.")
            return

        if new_selection_id in DATA_COLUMNS:
            notifications.WarningNotification(f"{new_selection_id} is not a valid selection column. It must not be one of {DATA_COLUMNS}.")
            return
        else:
            if new_selection_id not in [self.plotter_inputs_widget.phasor_selection_id_combobox.itemText(i) for i in range(self.plotter_inputs_widget.phasor_selection_id_combobox.count())]:
                self.plotter_inputs_widget.phasor_selection_id_combobox.addItem(new_selection_id)
            self.plotter_inputs_widget.phasor_selection_id_combobox.setCurrentText(new_selection_id)
            # If column_name is not in features, add it with zeros
            if new_selection_id not in self._labels_layer_with_phasor_features.features.columns:
                self._labels_layer_with_phasor_features.features[new_selection_id] = np.zeros_like(self._labels_layer_with_phasor_features.features['label'].values)

    def on_selection_id_changed(self):
        """Callback function when the phasor selection id combobox is changed.

        This function updates the `selection_id` attribute with the selected text from the combobox.
        """
        self.selection_id = self.plotter_inputs_widget.phasor_selection_id_combobox.currentText()
    
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
            notifications.WarningNotification(f"Harmonic value should be greater than 0. Setting to 1.")
            value = 1
        self.plotter_inputs_widget.harmonic_spinbox.setValue(value)

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
            notifications.WarningNotification(f"{colormap} is not a valid colormap. Setting to default colormap.")
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
            notifications.WarningNotification(f"Number of bins should be greater than 1. Setting to 10.")
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
        for harmonic in range(1, self._labels_layer_with_phasor_features.features['harmonic'].max() + 1):
            harmonic_mask = self._labels_layer_with_phasor_features.features['harmonic'] == harmonic
            self._labels_layer_with_phasor_features.features.loc[harmonic_mask, column] = manual_selection
        self.create_phasors_selected_layer()

    def reset_layer_choices(self):
        """Reset the image layer with phasor features combobox choices.

        This function is called when a new layer is added or removed.
        It also updates `_labels_layer_with_phasor_features` attribute with the Labels layer in the metadata of the selected image layer.
        """
        self.plotter_inputs_widget.image_layer_with_phasor_features_combobox.clear()
        layer_names = [layer.name for layer in self.viewer.layers if isinstance(
                layer, Image) and 'phasor_features_labels_layer' in layer.metadata.keys()]
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
        labels_layer_name = self.plotter_inputs_widget.image_layer_with_phasor_features_combobox.currentText()
        if labels_layer_name == "":
            self._labels_layer_with_phasor_features = None
            return
        self._labels_layer_with_phasor_features = self.viewer.layers[labels_layer_name].metadata['phasor_features_labels_layer']
        # Set harmonic spinbox maximum value based on maximum harmonic in the table
        self.plotter_inputs_widget.harmonic_spinbox.setMaximum(self._labels_layer_with_phasor_features.features['harmonic'].max())

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
        if self.selection_id is None or self.selection_id == "":
            return x_data, y_data, np.zeros_like(x_data)
        else:
            selection_data = table[self.selection_id][table['harmonic'] == self.harmonic].values
        return x_data, y_data, selection_data

    def plot(self):
        """Plot the selected phasor features.

        This function plots the selected phasor features in the canvas widget.
        It also creates the phasors selected layer.
        """
        x_data, y_data, selection_id_data = self.get_features()
        # Set active artist
        self.canvas_widget.active_artist = self.canvas_widget.artists[ArtistType[self.plot_type]]
        # Set data in the active artist
        self.canvas_widget.active_artist.data = np.column_stack(
            (x_data, y_data))
        # Set selection data in the active artist
        self.canvas_widget.active_artist.color_indices = selection_id_data
        # Set colormap in the active artist
        selected_histogram_colormap = colormaps.ALL_COLORMAPS[self.histogram_colormap]
        # Temporary convertion to LinearSegmentedColormap to match matplotlib format, while biaplotter is not updated
        selected_histogram_colormap = LinearSegmentedColormap.from_list(
            selected_histogram_colormap.name, selected_histogram_colormap.colors)
        self.canvas_widget.artists[ArtistType.HISTOGRAM2D].histogram_colormap = selected_histogram_colormap
        # Set log scale in the active artist
        if self.canvas_widget.artists[ArtistType.HISTOGRAM2D].histogram is not None:
            if self.histogram_log_scale:
                self.canvas_widget.artists[ArtistType.HISTOGRAM2D].histogram[-1].set_norm(LogNorm())
            else:
                self.canvas_widget.artists[ArtistType.HISTOGRAM2D].histogram[-1].set_norm(Normalize())
        # Set number of bins in the active artist
        self.canvas_widget.artists[ArtistType.HISTOGRAM2D].bins = self.histogram_bins
        # Temporarily set active artist "again" to have it displayed on top #TODO: Fix this
        self.canvas_widget.active_artist = self.canvas_widget.artists[ArtistType[self.plot_type]]

        self.create_phasors_selected_layer()

    def create_phasors_selected_layer(self):
        """Create or update the phasors selected layer."""
        if self._labels_layer_with_phasor_features is None:
            return
        input_array = np.asarray(self._labels_layer_with_phasor_features.data)
        input_array_values = np.asarray(self._labels_layer_with_phasor_features.features['label'].values)
        # If no selection id is provided, set all pixels to 0
        if self.selection_id is None or self.selection_id == "":
            phasors_layer_data = np.zeros_like(self._labels_layer_with_phasor_features.features['label'].values)
        else:
            phasors_layer_data = np.asarray(self._labels_layer_with_phasor_features.features[self.selection_id].values)
        
        mapped_data = map_array(input_array, input_array_values, phasors_layer_data)
        color_dict = colormap_to_dict(self._colormap, self._colormap.N, exclude_first=True)
        # Build output phasors Labels layer
        phasors_selected_layer = Labels(
            mapped_data, name='Phasors Selected', scale=self._labels_layer_with_phasor_features.scale,
            colormap=DirectLabelColormap(color_dict=color_dict, name='cat10_mod'))
        if self._phasors_selected_layer is None:
            self._phasors_selected_layer = self.viewer.add_layer(phasors_selected_layer)
        else:
            self._phasors_selected_layer.data = mapped_data
            self._phasors_selected_layer.scale = self._labels_layer_with_phasor_features.scale

if __name__ == "__main__":
    import napari
    time_constants = [0.1, 1, 2, 3, 4, 5, 10]
    raw_flim_data = make_raw_flim_data(time_constants=time_constants)
    harmonic = [1, 2, 3]
    intensity_image_layer = make_intensity_layer_with_phasors(raw_flim_data, harmonic=harmonic)
    viewer = napari.Viewer()
    viewer.add_layer(intensity_image_layer)
    plotter = PlotterWidget(viewer)
    viewer.window.add_dock_widget(plotter, area="right")
    napari.run()