"""
This module contains widgets to:

    - Transform FLIM and hyperspectral images into phasor space

"""

import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from napari.layers import Image
from napari.utils import colormaps, DirectLabelColormap
from napari.utils.notifications import show_error, show_info
from phasorpy.phasor import phasor_calibrate, phasor_to_apparent_lifetime
from qtpy import uic
from qtpy.QtGui import QDoubleValidator
from qtpy.QtWidgets import (
    QComboBox,
    QCompleter,
    QDirModel,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTreeView,
    QVBoxLayout,
    QWidget,
)
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from ._reader import _get_filename_extension, napari_get_reader
from ._writer import write_ome_tiff

if TYPE_CHECKING:
    import napari


class PhasorTransform(QWidget):
    """Widget to transform FLIM and hyperspectral images into phasor space."""

    def __init__(self, viewer: "napari.viewer.Viewer"):
        """Initialize the widget."""
        super().__init__()
        self.viewer = viewer

        # Create main layout
        main_layout = QVBoxLayout(self)

        # Create search tree
        search_tree = QTreeView()
        model = QDirModel()
        search_tree.setModel(model)
        search_tree.setColumnHidden(1, True)
        search_tree.setColumnHidden(2, True)
        search_tree.setColumnHidden(3, True)
        main_layout.addWidget(search_tree)

        # Create layout for dynamic widgets
        self.dynamic_widget_layout = QVBoxLayout()
        main_layout.addLayout(self.dynamic_widget_layout)

        # Set up callbacks whenever the selection changes
        selection = search_tree.selectionModel()
        selection.currentChanged.connect(
            lambda current: self._on_change(current, model)
        )

        # Define reader options (example)
        self.reader_options = {
            ".fbd": FbdWidget,
            ".ptu": PtuWidget,
            ".lsm": LsmWidget,
            ".tif": LsmWidget,
        }

    def _on_change(self, current, model):
        """Callback whenever the selection changes."""
        path = model.filePath(current)
        _, extension = _get_filename_extension(path)
        if extension in self.reader_options:
            # Clear existing widgets
            for i in reversed(range(self.dynamic_widget_layout.count())):
                widget = self.dynamic_widget_layout.takeAt(i).widget()
                widget.deleteLater()

            # Create new widgets based on extension
            create_widget_class = self.reader_options[extension]
            new_widget = create_widget_class(self.viewer, path)
            self.dynamic_widget_layout.addWidget(new_widget)
        else:
            # Clear existing widgets
            for i in reversed(range(self.dynamic_widget_layout.count())):
                widget = self.dynamic_widget_layout.takeAt(i).widget()
                widget.deleteLater()


class AdvancedOptionsWidget(QWidget):
    """Base class for advanced options widgets."""

    def __init__(self, viewer, path):
        """Initialize the widget."""
        super().__init__()
        self.viewer = viewer
        self.path = path
        self.reader_options = None
        self.harmonics = [1]
        self.initUI()

    def initUI(self):
        """Initialize the user interface."""
        # Initial layout
        self.mainLayout = QVBoxLayout()
        self.setLayout(self.mainLayout)

    def _harmonic_widget(self):
        """Add the harmonic widget to main layout."""
        self.mainLayout.addWidget(QLabel("Harmonics: "))
        self.harmonic_start = QSpinBox()
        self.harmonic_start.setMinimum(1)
        self.harmonic_start.valueChanged.connect(
            lambda: self._on_harmonic_spinbox_changed()
        )
        self.harmonic_end = QSpinBox()
        self.harmonic_end.setMinimum(self.harmonic_start.value())
        self.harmonic_end.valueChanged.connect(
            lambda: self._on_harmonic_spinbox_changed()
        )
        harmonic_layout = QHBoxLayout()
        harmonic_layout.addWidget(QLabel("from: "))
        harmonic_layout.addWidget(self.harmonic_start)
        harmonic_layout.addWidget(QLabel("to: "))
        harmonic_layout.addWidget(self.harmonic_end)
        self.mainLayout.addLayout(harmonic_layout)

    def _frame_widget(self):
        """Add the frame widget to main layout."""
        self.mainLayout.addWidget(QLabel("Frames: "))
        self.frames = QComboBox()
        self.frames.addItems(["Average all frames"])
        for frame in range(self.all_frames):
            self.frames.addItem(str(frame))
        self.frames.setCurrentIndex(0)
        self.frames.currentIndexChanged.connect(
            self._on_frames_combobox_changed
        )
        self.mainLayout.addWidget(self.frames)

    def _channels_widget(self):
        # Channel selection
        self.mainLayout.addWidget(QLabel("Channels: "))
        self.channels = QComboBox()
        self.channels.addItems(["All channels"])
        for channel in range(self.all_channels):
            self.channels.addItem(str(channel))
        self.channels.setCurrentIndex(0)
        self.channels.currentIndexChanged.connect(
            self._on_channels_combobox_changed
        )
        self.mainLayout.addWidget(self.channels)

    def _on_harmonic_spinbox_changed(self):
        """Callback whenever either harmonic spinbox value changes."""
        start = self.harmonic_start.value()
        end = self.harmonic_end.value()
        self.harmonic_end.setMinimum(self.harmonic_start.value())
        if end < start:
            end = start
            self.harmonic_end.setValue(start)
        self.harmonics = list(range(start, end + 1))

    def _on_frames_combobox_changed(self, index):
        """Callback whenever the frames combobox changes."""
        self.reader_options["frame"] = index - 1

    def _on_channels_combobox_changed(self, index):
        """Callback whenever the channels combobox changes."""
        if index == 0:
            self.reader_options["channel"] = None
        else:
            self.reader_options["channel"] = index - 1

    def _on_click(self, path, reader_options, harmonics):
        """Callback whenever the calculate phasor button is clicked."""
        reader = napari_get_reader(
            path, reader_options=reader_options, harmonics=harmonics
        )
        for layer in reader(path):
            self.viewer.add_image(
                layer[0], name=layer[1]["name"], metadata=layer[1]["metadata"]
            )


class FbdWidget(AdvancedOptionsWidget):
    """Widget for FLIMbox FBD files."""

    def __init__(self, viewer, path):
        """Initialize the widget."""
        import lfdfiles

        with lfdfiles.FlimboxFbd(path) as fbd:
            self.all_frames = len(fbd.frames(None)[1])
            self.all_channels = fbd.channels
        super().__init__(viewer, path)
        self.reader_options = {"frame": -1, "channel": None}

    def initUI(self):
        super().initUI()
        self._harmonic_widget()
        self._frame_widget()
        self._channels_widget()

        # Laser factor
        self.mainLayout.addWidget(QLabel("Laser Factor (optional): "))
        self.laser_factor = QLineEdit()
        self.laser_factor.setText("-1")
        self.laser_factor.setToolTip(
            "Default is -1. If this doesn't work, "
            "most probable laser factors are: 0.00022, 2.50012, 2.50016"
        )
        self.laser_factor.setValidator(QDoubleValidator())
        laser_factor_completer = QCompleter(["0.00022", "2.50012", "2.50016"])
        self.laser_factor.setCompleter(laser_factor_completer)
        self.mainLayout.addWidget(self.laser_factor)

        # Calculate phasor button
        self.btn = QPushButton("Phasor Transform")
        self.btn.clicked.connect(
            lambda: self._on_click(
                self.path, self.reader_options, self.harmonics
            )
        )
        self.mainLayout.addWidget(self.btn)

    def _on_click(self, path, reader_options, harmonics):
        """Callback whenever the calculate phasor button is clicked."""
        if self.laser_factor.text():
            reader_options["laser_factor"] = float(self.laser_factor.text())
        super()._on_click(path, reader_options, harmonics)


class PtuWidget(AdvancedOptionsWidget):
    """Widget for PicoQuant PTU files."""

    def __init__(self, viewer, path):
        """Initialize the widget."""
        import ptufile

        with ptufile.PtuFile(path) as ptu:
            self.all_frames = ptu.shape[0]
            self.all_channels = ptu.shape[-2]
        super().__init__(viewer, path)
        self.reader_options = {"frame": -1, "channel": None}

    def initUI(self):
        """Initialize the user interface."""
        super().initUI()
        self._harmonic_widget()
        self._frame_widget()
        self._channels_widget()

        # dtime widget
        self.mainLayout.addWidget(QLabel("dtime (optional): "))
        self.dtime = QLineEdit()
        self.dtime.setText("0")
        self.dtime.setToolTip(
            "Specifies number of bins in image histogram."
            "If 0 (default), return number of bins in one period."
            "If < 0, integrate delay time axis."
            "If > 0, return up to specified bin."
        )
        self.mainLayout.addWidget(self.dtime)

        # Calculate phasor button
        self.btn = QPushButton("Phasor Transform")
        self.btn.clicked.connect(
            lambda: self._on_click(
                self.path, self.reader_options, self.harmonics
            )
        )
        self.mainLayout.addWidget(self.btn)

    def _on_click(self, path, reader_options, harmonics):
        """Callback whenever the calculate phasor button is clicked."""
        if self.dtime.text():
            reader_options["dtime"] = float(self.dtime.text())
        super()._on_click(path, reader_options, harmonics)


class LsmWidget(AdvancedOptionsWidget):
    """Widget for Zeiss LSM files."""

    def __init__(self, viewer, path):
        """Initialize the widget."""
        super().__init__(viewer, path)

    def initUI(self):
        """Initialize the user interface."""
        super().initUI()
        self._harmonic_widget()

        # Calculate phasor button
        self.btn = QPushButton("Phasor Transform")
        self.btn.clicked.connect(
            lambda: self._on_click(
                self.path, self.reader_options, self.harmonics
            )
        )
        self.mainLayout.addWidget(self.btn)


class CalibrationWidget(QWidget):
    """Widget to calibrate a FLIM image layer using a calibration image."""

    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer

        # Creates and empty widget
        self.calibration_widget = QWidget()
        uic.loadUi(
            Path(__file__).parent / "ui/calibration_widget.ui",
            self.calibration_widget,
        )

        # self.calibration_widget.frequency_line_edit_widget.setText("0")

        # Connect callbacks
        self.calibration_widget.calibrate_push_button.clicked.connect(
            self._on_click
        )

        # Connect layer events to populate combobox
        self.viewer.layers.events.inserted.connect(self._populate_comboboxes)
        self.viewer.layers.events.removed.connect(self._populate_comboboxes)

        # Populate combobox
        self._populate_comboboxes()

        mainLayout = QVBoxLayout()
        mainLayout.addWidget(self.calibration_widget)
        self.setLayout(mainLayout)

    def _populate_comboboxes(self):
        self.calibration_widget.calibration_layer_combobox.clear()
        image_layers = [
            layer for layer in self.viewer.layers if isinstance(layer, Image)
        ]
        for layer in image_layers:
            self.calibration_widget.calibration_layer_combobox.addItem(
                layer.name
            )
        self.calibration_widget.sample_layer_combobox.clear()
        image_layers = [
            layer for layer in self.viewer.layers if isinstance(layer, Image)
        ]
        for layer in image_layers:
            self.calibration_widget.sample_layer_combobox.addItem(layer.name)

    def _on_click(self):
        frequency = int(
            self.calibration_widget.frequency_line_edit_widget.text()
        )
        lifetime = float(
            self.calibration_widget.lifetime_line_edit_widget.text()
        )
        sample_name = (
            self.calibration_widget.sample_layer_combobox.currentText()
        )
        calibration_name = (
            self.calibration_widget.calibration_layer_combobox.currentText()
        )
        sample_metadata = self.viewer.layers[sample_name].metadata
        sample_phasor_data = sample_metadata[
            "phasor_features_labels_layer"
        ].features
        calibration_phasor_data = (
            self.viewer.layers[calibration_name]
            .metadata["phasor_features_labels_layer"]
            .features
        )
        harmonics = np.unique(sample_phasor_data["harmonic"])
        original_mean_shape = (
            self.viewer.layers[sample_name].metadata["original_mean"].shape
        )
        if (
            "calibrated" not in sample_metadata.keys()
            or sample_metadata["calibrated"] is False
        ):
            skip_axis = None
            if len(np.unique(sample_phasor_data["harmonic"])) > 1:
                skip_axis = (0,)
                real, imag = phasor_calibrate(
                    np.reshape(
                        sample_phasor_data["G_original"],
                        (len(harmonics),) + original_mean_shape,
                    ),
                    np.reshape(
                        sample_phasor_data["S_original"],
                        (len(harmonics),) + original_mean_shape,
                    ),
                    np.reshape(
                        calibration_phasor_data["G_original"],
                        (len(harmonics),) + original_mean_shape,
                    ),
                    np.reshape(
                        calibration_phasor_data["S_original"],
                        (len(harmonics),) + original_mean_shape,
                    ),
                    frequency=frequency * np.array(harmonics),
                    lifetime=lifetime,
                    skip_axis=skip_axis,
                )
            else:
                real, imag = phasor_calibrate(
                    sample_phasor_data["G_original"],
                    sample_phasor_data["S_original"],
                    calibration_phasor_data["G_original"],
                    calibration_phasor_data["S_original"],
                    frequency=frequency,
                    lifetime=lifetime,
                )
            sample_phasor_data["G_original"] = real.flatten()
            sample_phasor_data["S_original"] = imag.flatten()
            sample_metadata["calibrated"] = True
            show_info(f"Calibrated {sample_name}")
        elif sample_metadata["calibrated"] is True:
            show_error("Layer already calibrated")


class WriterWidget(QWidget):
    """Widget to export phasor data to a OME-TIF file."""

    def __init__(self, viewer: "napari.viewer.Viewer"):
        """Initialize the widget."""
        super().__init__()
        self.viewer = viewer

        # Create main layout
        self.main_layout = QVBoxLayout(self)

        # Set export location
        self.main_layout.addWidget(QLabel("Select export location: "))
        self.save_path = QLineEdit()
        self.main_layout.addWidget(self.save_path)

        # Create search tree
        search_tree = QTreeView()
        self.model = QDirModel()
        search_tree.setModel(self.model)
        search_tree.setColumnHidden(1, True)
        search_tree.setColumnHidden(2, True)
        search_tree.setColumnHidden(3, True)
        self.main_layout.addWidget(search_tree)

        # Set up callbacks whenever the selection changes
        self.selection = search_tree.selectionModel()
        self.selection.currentChanged.connect(
            lambda current: self._on_search_tree_change(current, self.model)
        )
        # Combobox to select image layer with phasor data for export
        self.main_layout.addWidget(
            QLabel("Select image layer to be exported: ")
        )
        self.export_layer_combobox = QComboBox()
        self.export_layer_combobox.currentIndexChanged.connect(
            self._on_combobox_change
        )
        self.main_layout.addWidget(self.export_layer_combobox)

        # Line edit to input name of exported file
        self.main_layout.addWidget(QLabel("Name of exported file: "))
        self.export_file_name = QLineEdit()
        self.main_layout.addWidget(self.export_file_name)

        # Connect layer events to populate combobox
        self.viewer.layers.events.inserted.connect(self._populate_combobox)
        self.viewer.layers.events.removed.connect(self._populate_combobox)

        # Populate combobox
        self._populate_combobox()

        # Export button
        self.btn = QPushButton("Export")
        self.btn.clicked.connect(self._on_click)
        self.main_layout.addWidget(self.btn)

    def _on_search_tree_change(self, current, model):
        """Callback whenever the selection of the search tree changes."""
        path = model.filePath(current)
        if os.path.isdir(path):
            self.save_path.setText(path)
        else:
            self.save_path.setText(os.path.dirname(path))

    def _populate_combobox(self):
        """Populate combobox with image layers."""
        self.export_layer_combobox.clear()
        image_layers = [
            layer for layer in self.viewer.layers if isinstance(layer, Image)
        ]
        for layer in image_layers:
            self.export_layer_combobox.addItem(layer.name)

    def _on_combobox_change(self):
        """Callback whenever the combobox changes."""
        export_layer_name = self.export_layer_combobox.currentText()
        self.export_file_name.setText(export_layer_name)

    def _on_click(self):
        """Callback whenever the export button is clicked."""
        if not self.save_path.text():
            show_error("Select export location")
            return
        if not self.export_file_name.text():
            show_error("Enter name of exported file")
            return
        export_layer_name = self.export_layer_combobox.currentText()
        export_layer = self.viewer.layers[export_layer_name]
        export_file_name = self.export_file_name.text()
        export_path = os.path.join(self.save_path.text(), export_file_name)
        export_path = write_ome_tiff(export_path, export_layer)
        show_info(f"Exported {export_layer_name} to {export_path}")


class LifetimeWidget(QWidget):
    """Widget to calibrate a FLIM image layer using a calibration image."""

    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer
        self.lifetime_data = None
        self._labels_layer_with_phasor_features = None
        self.lifetime_layer = None
        self.lifetime_histogram = None
        self.harmonics = None
        self.selected_harmonic = None

        # Create main layout
        self.main_layout = QVBoxLayout(self)
        # Select layer to calculate lifetime
        self.main_layout.addWidget(QLabel("Select layer to calculate lifetime: "))
        self.layer_combobox = QComboBox()
        self.layer_combobox.currentIndexChanged.connect(self._on_layers_combobox_change)
        self.main_layout.addWidget(self.layer_combobox)
        # Frequency input
        self.main_layout.addWidget(QLabel("Frequency: "))
        self.frequency_input = QLineEdit()
        self.frequency_input.setValidator(QDoubleValidator())
        self.main_layout.addWidget(self.frequency_input)
        # Add colormap combobox
        self.main_layout.addWidget(QLabel("Lifetime colormap: "))
        self.lifetime_colormap_combobox = QComboBox()
        self.lifetime_colormap_combobox.addItems(plt.colormaps())
        self.lifetime_colormap_combobox.setCurrentText("turbo")
        self.main_layout.addWidget(self.lifetime_colormap_combobox)
        # Add combobox to select between phase or modulation apparent lifetime
        self.main_layout.addWidget(QLabel("Show phase of modulation apparent lifetime: "))
        self.lifetime_type_combobox = QComboBox()
        self.lifetime_type_combobox.addItems(["Phase", "Modulation"])
        self.lifetime_type_combobox.setCurrentText("Phase")
        self.main_layout.addWidget(self.lifetime_type_combobox)
        # Plot lifetime button
        plot_lifetime_button = QPushButton("Plot Lifetime")
        plot_lifetime_button.clicked.connect(self._on_click)
        self.main_layout.addWidget(plot_lifetime_button)
        # Connect layer events to populate combobox
        self.viewer.layers.events.inserted.connect(self._populate_layers_combobox)
        self.viewer.layers.events.removed.connect(self._populate_layers_combobox)

        # Populate combobox
        self._populate_layers_combobox()

        # Add layout for histogram
        self.histogram_widget = QWidget(self)
        self.histogram_layout = QVBoxLayout(self.histogram_widget)
        self.main_layout.addWidget(self.histogram_widget)



    @property
    def lifetime_colormap(self):
        """Gets or sets the lifetime colormap from the colormap combobox.

        Returns
        -------
        str
            The colormap name.
        """
        return self.lifetime_colormap_combobox.currentText()

    @lifetime_colormap.setter
    def lifetime_colormap(self, colormap: str):
        """Sets the lifetime colormap from the colormap combobox."""
        if colormap not in plt.colormaps():
            show_error(f"{colormap} is not a valid colormap. Setting to default colormap.")
            colormap = self.lifetime_colormap.name
        self.lifetime_colormap_combobox.setCurrentText(colormap)
    
    def _populate_layers_combobox(self):
        """Populate combobox with image layers."""
        self.layer_combobox.clear()
        layer_names = [
            layer.name
            for layer in self.viewer.layers
            if isinstance(layer, Image)
            and "phasor_features_labels_layer" in layer.metadata.keys()
        ]
        for layer in layer_names:
            self.layer_combobox.addItem(layer)
    
    def _on_layers_combobox_change(self):
        """Callback whenever the layer combobox changes."""
        # TODO: get the frequency automatically from the metadata
        layer_name = self.layer_combobox.currentText()
        if layer_name == "":
            self._labels_layer_with_phasor_features = None
            self.histogram_widget.hide()
            return
        self._labels_layer_with_phasor_features = self.viewer.layers[layer_name].metadata['phasor_features_labels_layer']
        self.harmonics = np.unique(self._labels_layer_with_phasor_features.features['harmonic'])
    
    def calculate_lifetimes(self):
        if self._labels_layer_with_phasor_features is None:
            return
        phasor_data = self._labels_layer_with_phasor_features.features
        frequency = float(self.frequency_input.text())
        lifetimes = phasor_to_apparent_lifetime(
            phasor_data['G'], phasor_data['S'], frequency=frequency
        )
        lifetimes = np.nan_to_num(lifetimes, nan=0)
        lifetimes = np.clip(lifetimes, a_min=0, a_max=None)
        harmonics = np.unique(phasor_data['harmonic'])
        mean_shape =  self._labels_layer_with_phasor_features.data.shape
        if self.lifetime_type_combobox.currentText() == "Phase":
            self.lifetime_data = np.reshape(lifetimes[0], (len(harmonics),) + mean_shape)
        else:
            self.lifetime_data = np.reshape(lifetimes[1], (len(harmonics),) + mean_shape)
            
    def create_lifetime_layer(self):
        """Create or update the lifetime layer for all harmonics."""
        if self.lifetime_data is None:
            return
        
        # Initialize a list to hold the colored arrays for each harmonic
        combined_colored_array = []

        # Iterate over each harmonic
        for harmonic_index in range(len(self.harmonics)):
            lifetime_data = self.lifetime_data[harmonic_index]
            
            # Flatten the lifetime data for percentile calculation
            flattened_data = lifetime_data.flatten()
            flattened_data = flattened_data[flattened_data > 0]
            
            # Calculate the 5th and 95th percentiles
            lower_bound = np.percentile(flattened_data, 5)
            upper_bound = np.percentile(flattened_data, 95)
            
            # Normalize the array values to the range [lower_bound, upper_bound]
            norm = mcolors.Normalize(vmin=lower_bound, vmax=upper_bound)
            
            # Choose a colormap
            cmap = cm.get_cmap(self.lifetime_colormap)
            
            # Apply the colormap to the normalized array
            colored_array = cmap(norm(lifetime_data))
            
            # Set the first value of the colormap to transparent
            colored_array[..., -1][lifetime_data == lifetime_data.min()] = 0
            
            # Append the colored array to the list
            combined_colored_array.append(colored_array)
        
        # Stack the colored arrays along a new axis to combine them
        combined_colored_array = np.stack(combined_colored_array, axis=0)
        
        # Build lifetime layer
        lifetime_layer_name = f"Lifetime: {self.layer_combobox.currentText()}"
        selected_lifetime_layer = Image(
            combined_colored_array,
            name=lifetime_layer_name,
            scale=self._labels_layer_with_phasor_features.scale,
            colormap=self.lifetime_colormap,
        )
        
        # Check if the layer is in the viewer before attempting to remove it
        if lifetime_layer_name in self.viewer.layers:
            self.viewer.layers.remove(self.viewer.layers[lifetime_layer_name])

        self.lifetime_layer = self.viewer.add_layer(
            selected_lifetime_layer
        )
    
    def plot_lifetime_histogram(self):
        """Plot the histogram of the lifetime data as a line plot."""
        if self.lifetime_data is None:
            return
        if self._labels_layer_with_phasor_features is None:
            return
        if self.selected_harmonic is None:
            self.selected_harmonic = self.harmonics.min()
        harmonic_index = list(self.harmonics).index(self.selected_harmonic)
        lifetime_data = self.lifetime_data[harmonic_index]
        # Flatten the lifetime data for histogram plotting
        flattened_data = lifetime_data.flatten()
        flattened_data = flattened_data[flattened_data > 0]

        # Calculate the histogram values
        counts, bin_edges = np.histogram(flattened_data, bins=100)
        
        # Calculate the bin centers
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Calculate the 5th and 95th percentiles
        lower_bound = np.percentile(flattened_data, 5)
        upper_bound = np.percentile(flattened_data, 95)
        
        # Create a Matplotlib figure and axis
        fig, ax = plt.subplots()
        
        # Plot the histogram values as a line plot with transparent line color
        ax.plot(bin_centers, counts, color='none', alpha=0)
        
        # Normalize the bin values to the range [lower_bound, upper_bound]
        norm = plt.Normalize(vmin=lower_bound, vmax=upper_bound)
        
        # Get the colormap
        cmap = plt.get_cmap(self.lifetime_colormap)
        
        # Fill the area under the curve using the colormap
        for count, bin_start, bin_end in zip(counts, bin_edges[:-1], bin_edges[1:]):
            bin_center = (bin_start + bin_end) / 2
            color = cmap(norm(bin_center))
            ax.fill_between([bin_start, bin_end], 0, count, color=color, alpha=0.7)
        
        ax.set_title('Lifetime Data Histogram')
        ax.set_xlabel('Lifetime')
        ax.set_ylabel('Frequency')
        
        # Clear the previous histogram plot but keep the QSpinBox
        for i in reversed(range(self.histogram_layout.count())):
            widget_to_remove = self.histogram_layout.itemAt(i).widget()
            if isinstance(widget_to_remove, FigureCanvas):
                self.histogram_layout.removeWidget(widget_to_remove)
                widget_to_remove.setParent(None)
        
        # Embed the Matplotlib figure into the widget
        canvas = FigureCanvas(fig)
        self.histogram_layout.addWidget(canvas)
        self.histogram_widget.show()

        # Add harmonic selector if it doesn't exist
        if not hasattr(self, 'harmonic_selector'):
            self.harmonic_selector = QSpinBox()
            self.harmonic_selector.setMinimum(self.harmonics.min())
            self.harmonic_selector.setMaximum(self.harmonics.max())
            self.harmonic_selector.setValue(self.selected_harmonic)
            self.harmonic_selector.valueChanged.connect(self._on_harmonic_changed)
        
        # Ensure the harmonic selector is always at the bottom
        if self.harmonic_selector not in [self.histogram_layout.itemAt(i).widget() for i in range(self.histogram_layout.count())]:
            self.histogram_layout.addWidget(self.harmonic_selector)
        else:
            self.histogram_layout.removeWidget(self.harmonic_selector)
            self.histogram_layout.addWidget(self.harmonic_selector)
        self.harmonic_selector.setValue(self.selected_harmonic)

    def _on_harmonic_changed(self):
        """Callback whenever the harmonic selector changes."""
        self.selected_harmonic = self.harmonic_selector.value()
        self.plot_lifetime_histogram()

    def _on_click(self):
        """Callback whenever the plot lifetime button is clicked."""
        if self.frequency_input.text() == "":
            show_error("Enter frequency")
            return
        self.calculate_lifetimes()
        self.create_lifetime_layer()
        self.plot_lifetime_histogram()

