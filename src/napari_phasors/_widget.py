"""
This module contains widgets to:

    - Transform FLIM and hyperspectral images into phasor space from
      the following file formats: FBD, PTU, LSM, SDT, TIF, OME-TIFF.
    - Export phasor data to OME-TIFF or CSV files.
    - Calibrate a FLIM image layer using a calibration image of known lifetime
      and frequency.
    - Calculate and plot the phase or modulation apparent lifetime of a FLIM
      image layer.

"""

from math import ceil, log10
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.colors import LinearSegmentedColormap
from napari.layers import Image
from napari.utils.notifications import show_error, show_info
from phasorpy.phasor import (
    phasor_calibrate,
    phasor_center,
    phasor_from_lifetime,
    phasor_to_apparent_lifetime,
    polar_from_reference_phasor,
)
from qtpy import uic
from qtpy.QtCore import Qt
from qtpy.QtGui import QDoubleValidator
from qtpy.QtWidgets import (
    QComboBox,
    QCompleter,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ._reader import _get_filename_extension, napari_get_reader
from ._utils import apply_filter_and_threshold
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
        self.main_layout = QVBoxLayout(self)

        # Create button to select file to be read
        self.search_button = QPushButton("Select file to be read")
        self.search_button.clicked.connect(self._open_file_dialog)
        self.main_layout.addWidget(self.search_button)

        # Save path display
        self.save_path = QLineEdit()
        self.save_path.setReadOnly(True)
        self.main_layout.addWidget(self.save_path)

        # Create layout for dynamic widgets
        self.dynamic_widget_layout = QVBoxLayout()
        self.main_layout.addLayout(self.dynamic_widget_layout)

        # Define reader options (example)
        self.reader_options = {
            ".fbd": FbdWidget,
            ".ptu": PtuWidget,
            ".lsm": LsmWidget,
            ".tif": LsmWidget,
            ".ome.tif": LsmWidget,
            ".sdt": SdtWidget,
        }

    def _open_file_dialog(self):
        """Open a `QFileDialog` to select a directory or file with specific extensions."""
        options = QFileDialog.Options()
        dialog = QFileDialog(self, "Select Export Location", options=options)
        dialog.setFileMode(QFileDialog.AnyFile)
        # Filter files by extension
        dialog.setNameFilter(
            "All files (*.tif *.ome.tif *.ptu *.fbd *.sdt *.lsm)"
        )
        if dialog.exec_():
            selected_file = dialog.selectedFiles()[0]
            self.save_path.setText(selected_file)
            _, extension = _get_filename_extension(selected_file)
            if extension in self.reader_options:
                # Clear existing widgets
                for i in reversed(range(self.dynamic_widget_layout.count())):
                    widget = self.dynamic_widget_layout.takeAt(i).widget()
                    widget.deleteLater()

                # Create new widgets based on extension
                create_widget_class = self.reader_options[extension]
                new_widget = create_widget_class(self.viewer, selected_file)
                self.dynamic_widget_layout.addWidget(new_widget)


class AdvancedOptionsWidget(QWidget):
    """Base class for advanced options widgets."""

    def __init__(self, viewer, path):
        """Initialize the widget."""
        super().__init__()
        self.viewer = viewer
        self.path = path
        self.reader_options = {}
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
        self.reader_options["frame"] = -1
        self.reader_options["channel"] = None

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
        self.reader_options["frame"] = -1
        self.reader_options["channel"] = None

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


class SdtWidget(AdvancedOptionsWidget):
    """Widget for SDT files."""

    def __init__(self, viewer, path):
        """Initialize the widget."""
        super().__init__(viewer, path)

    def initUI(self):
        """Initialize the user interface."""
        super().initUI()
        self._harmonic_widget()

        # Index selector
        self.mainLayout.addWidget(QLabel("Index (optional): "))
        self.index = QLineEdit()
        self.index.setText("0")
        self.index.setToolTip(
            "Index of dataset to read in case the file contains multiple "
            "datasets. By default, the first dataset is read."
        )
        self.mainLayout.addWidget(self.index)

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
        if self.index.text():
            reader_options["index"] = int(self.index.text())
        super()._on_click(path, reader_options, harmonics)


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

        # Connect callbacks
        self.calibration_widget.calibrate_push_button.clicked.connect(
            self._on_click
        )

        # Connect comboboxes
        self.calibration_widget.sample_layer_combobox.currentIndexChanged.connect(
            self._on_combobox_change
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

    def _on_combobox_change(self):
        layer_name = (
            self.calibration_widget.sample_layer_combobox.currentText()
        )
        if layer_name == "":
            return
        layer_metadata = self.viewer.layers[layer_name].metadata
        if (
            'settings' in layer_metadata.keys()
            and 'frequency' in layer_metadata['settings'].keys()
        ):
            self.calibration_widget.frequency_line_edit_widget.setText(
                str(layer_metadata['settings']['frequency'])
            )

    def _on_click(self):
        sample_name = (
            self.calibration_widget.sample_layer_combobox.currentText()
        )
        calibration_name = (
            self.calibration_widget.calibration_layer_combobox.currentText()
        )
        if sample_name == "" or calibration_name == "":
            show_error("Select sample and calibration layers")
            return
        frequency = self.calibration_widget.frequency_line_edit_widget.text()
        lifetime = self.calibration_widget.lifetime_line_edit_widget.text()
        if frequency == "":
            show_error("Enter frequency")
            return
        if lifetime == "":
            show_error("Enter reference lifetime")
            return
        frequency = float(frequency)
        if "settings" not in self.viewer.layers[sample_name].metadata.keys():
            self.viewer.layers[sample_name].metadata["settings"] = {}
        self.viewer.layers[sample_name].metadata["settings"][
            "frequency"
        ] = frequency
        lifetime = float(lifetime)
        sample_metadata = self.viewer.layers[sample_name].metadata
        sample_phasor_data = sample_metadata[
            "phasor_features_labels_layer"
        ].features
        harmonics = np.unique(sample_phasor_data["harmonic"])
        original_mean_shape = sample_metadata["original_mean"].shape
        calibration_phasor_data = (
            self.viewer.layers[calibration_name]
            .metadata["phasor_features_labels_layer"]
            .features
        )
        calibration_mean = self.viewer.layers[calibration_name].metadata[
            "original_mean"
        ]
        calibration_harmonics = np.unique(calibration_phasor_data["harmonic"])
        if not np.array_equal(harmonics, calibration_harmonics):
            show_error(
                "Harmonics in sample and calibration layers do not match"
            )
            return
        calibration_g = np.reshape(
            calibration_phasor_data["G_original"],
            (len(calibration_harmonics),) + calibration_mean.shape,
        )
        calibration_s = np.reshape(
            calibration_phasor_data["S_original"],
            (len(calibration_harmonics),) + calibration_mean.shape,
        )
        if "settings" not in sample_metadata.keys():
            sample_metadata["settings"] = {}
        if (
            "calibrated" not in sample_metadata["settings"].keys()
            or sample_metadata["settings"]["calibrated"] is False
        ):
            real, imag = phasor_calibrate(
                np.reshape(
                    sample_phasor_data["G_original"],
                    (len(harmonics),) + original_mean_shape,
                ),
                np.reshape(
                    sample_phasor_data["S_original"],
                    (len(harmonics),) + original_mean_shape,
                ),
                calibration_mean,
                calibration_g,
                calibration_s,
                frequency=frequency,
                lifetime=lifetime,
                harmonic=harmonics.tolist(),
            )
            sample_phasor_data["G_original"] = real.flatten()
            sample_phasor_data["S_original"] = imag.flatten()
            sample_metadata["settings"]["calibrated"] = True
            calibration_phase, calibration_modulation = (
                polar_from_reference_phasor(
                    *phasor_center(
                        calibration_mean, calibration_g, calibration_s
                    )[1:],
                    *phasor_from_lifetime(frequency, lifetime),
                )
            )
            sample_metadata["settings"]["calibration_phase"] = float(
                calibration_phase[0]
            )
            sample_metadata["settings"]["calibration_modulation"] = float(
                calibration_modulation[0]
            )
            print(sample_metadata["settings"])
            show_info(f"Calibrated {sample_name}")
        elif sample_metadata["settings"]["calibrated"] is True:
            show_error("Layer already calibrated")


class WriterWidget(QWidget):
    """Widget to export phasor data to a OME-TIF or CSV file."""

    def __init__(self, viewer: "napari.viewer.Viewer"):
        """Initialize the widget."""
        super().__init__()
        self.viewer = viewer

        # Create main layout
        self.main_layout = QVBoxLayout(self)

        # Combobox to select image layer with phasor data for export
        self.main_layout.addWidget(
            QLabel("Select Image Layer to be Exported: ")
        )
        self.export_layer_combobox = QComboBox()
        self.main_layout.addWidget(self.export_layer_combobox)

        # Button to open save dialog to select export location and name
        self.search_button = QPushButton("Select Export Location and Name")
        self.search_button.clicked.connect(self._open_file_dialog)
        self.main_layout.addWidget(self.search_button)

        # Connect layer events to populate combobox
        self.viewer.layers.events.inserted.connect(self._populate_combobox)
        self.viewer.layers.events.removed.connect(self._populate_combobox)

        # Populate combobox
        self._populate_combobox()

    def _open_file_dialog(self):
        """Open a `QFileDialog` to select a directory or specify a filename."""
        if self.export_layer_combobox.currentText() == "":
            show_error("No layer with phasor data selected")
            return
        options = QFileDialog.Options()
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.AnyFile)
        file_dialog.setAcceptMode(QFileDialog.AcceptSave)
        file_dialog.setOptions(options)

        # Define multiple file types for the dropdown
        file_dialog.setNameFilters(
            ["Phasor as OME-TIFF (*.ome.tif)", "Phasor table as CSV (*.csv)"]
        )

        # Execute the dialog and retrieve the selected file path
        if file_dialog.exec_():
            selected_filter = file_dialog.selectedNameFilter()
            file_path = file_dialog.selectedFiles()[0]

            # Append appropriate file extension if not present
            if (
                selected_filter == "Phasor as OME-TIFF (*.ome.tif)"
                and not file_path.endswith(".ome.tif")
            ):
                if file_path.endswith(".tif"):
                    file_path = file_path[:-4]  # Remove the .tif extension
                file_path += ".ome.tif"
            elif (
                selected_filter == "Phasor table as CSV (*.csv)"
                and not file_path.endswith(".csv")
            ):
                file_path += ".csv"

            self._save_file(file_path, selected_filter)

    def _populate_combobox(self):
        """Populate combobox with image layers."""
        self.export_layer_combobox.clear()
        image_layers = [
            layer for layer in self.viewer.layers if isinstance(layer, Image)
        ]
        for layer in image_layers:
            self.export_layer_combobox.addItem(layer.name)

    def _save_file(self, file_path, selected_filter):
        """Callback whenever the export location and name are specified."""
        export_layer = self.viewer.layers[
            self.export_layer_combobox.currentText()
        ]
        if selected_filter == "Phasor as OME-TIFF (*.ome.tif)":
            write_ome_tiff(file_path, export_layer)
        elif selected_filter == "Phasor table as CSV (*.csv)":
            if not file_path.endswith(".csv"):
                file_path += ".csv"
            phasor_table = export_layer.metadata[
                "phasor_features_labels_layer"
            ].features
            harmonics = np.unique(phasor_table["harmonic"])
            # Get coordinates of each pixel
            coords = np.unravel_index(
                np.arange(export_layer.data.size), export_layer.data.shape
            )
            # Tile coordinates for each harmonic
            coords = [np.tile(coord, len(harmonics)) for coord in coords]
            # Add coordinates to phasor table
            for dim, coord in enumerate(coords):
                phasor_table[f'dim_{dim}'] = coord
            # Drop rows with NaNs
            phasor_table = phasor_table.dropna()
            phasor_table.to_csv(
                file_path,
                index=False,
            )
        show_info(f"Exported {export_layer.name} to {file_path}")


class LifetimeWidget(QWidget):
    """Widget to calibrate a FLIM image layer using a calibration image."""

    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer
        self.lifetime_data = None
        self._labels_layer_with_phasor_features = None
        self.lifetime_layer = None
        self.harmonics = None
        self.selected_harmonic = None
        self.lifetime_colormap = None
        self.colormap_contrast_limits = None
        self.hist_fig, self.hist_ax = plt.subplots()
        self.counts = None
        self.bin_edges = None
        self.bin_centers = None
        self.threshold_factor = None

        # Create main layout
        self.main_layout = QVBoxLayout(self)
        # Select layer to calculate lifetime
        self.main_layout.addWidget(
            QLabel("Select layer to calculate lifetime: ")
        )
        self.layer_combobox = QComboBox()
        self.layer_combobox.currentIndexChanged.connect(
            self._on_layers_combobox_change
        )
        self.main_layout.addWidget(self.layer_combobox)
        # Frequency input
        self.main_layout.addWidget(QLabel("Frequency: "))
        self.frequency_input = QLineEdit()
        self.frequency_input.setValidator(QDoubleValidator())
        self.main_layout.addWidget(self.frequency_input)
        # Add threshold slider
        self.threshold_label = QLabel("Apply mean intensity threshold: 0.0")
        self.main_layout.addWidget(self.threshold_label)
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(0)
        self.threshold_slider.valueChanged.connect(
            self.on_threshold_slider_change
        )
        self.main_layout.addWidget(self.threshold_slider)
        # Add combobox to select between phase or modulation apparent lifetime
        self.main_layout.addWidget(
            QLabel("Display phase or modulation apparent lifetimes: ")
        )
        self.lifetime_type_combobox = QComboBox()
        self.lifetime_type_combobox.addItems(["Phase", "Modulation"])
        self.lifetime_type_combobox.setCurrentText("Phase")
        self.main_layout.addWidget(self.lifetime_type_combobox)
        # Plot lifetime button
        self.plot_lifetime_button = QPushButton("Plot Lifetime")
        self.plot_lifetime_button.clicked.connect(self._on_click)
        self.main_layout.addWidget(self.plot_lifetime_button)
        # Connect layer events to populate combobox
        self.viewer.layers.events.inserted.connect(
            self._populate_layers_combobox
        )
        self.viewer.layers.events.removed.connect(
            self._populate_layers_combobox
        )

        # Add histogram widget
        self.histogram_widget = QWidget(self)
        self.histogram_layout = QVBoxLayout(self.histogram_widget)
        # Embed the Matplotlib figure into the widget
        canvas = FigureCanvas(self.hist_fig)
        self.histogram_layout.addWidget(canvas)

        # Add harmonic selector label and QSpinBox if they don't exist
        self.harmonic_selector_label = QLabel(
            "Select a harmonic to display its lifetime histogram:"
        )
        self.histogram_layout.addWidget(self.harmonic_selector_label)
        self.harmonic_selector = QSpinBox()
        self.harmonic_selector.valueChanged.connect(self._on_harmonic_changed)
        self.histogram_layout.addWidget(self.harmonic_selector)
        self.main_layout.addWidget(self.histogram_widget)
        self.histogram_widget.hide()

        # Populate combobox
        self._populate_layers_combobox()

    def on_threshold_slider_change(self):
        self.threshold_label.setText(
            'Apply mean intensity threshold: '
            + str(self.threshold_slider.value() / self.threshold_factor)
        )

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
        layer_name = self.layer_combobox.currentText()
        if layer_name == "":
            self._labels_layer_with_phasor_features = None
            return
        layer_metadata = self.viewer.layers[layer_name].metadata
        self._labels_layer_with_phasor_features = layer_metadata[
            'phasor_features_labels_layer'
        ]
        self.harmonics = np.unique(
            self._labels_layer_with_phasor_features.features['harmonic']
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
        self.threshold_slider.setMaximum(
            ceil(max_mean_value * self.threshold_factor)
        )
        if 'settings' in layer_metadata.keys():
            settings = layer_metadata['settings']
            if 'frequency' in layer_metadata['settings'].keys():
                self.frequency_input.setText(
                    str(layer_metadata['settings']['frequency'])
                )
            if 'threshold' in settings.keys():
                self.threshold_slider.setValue(
                    int(settings['threshold'] * self.threshold_factor)
                )
                self.on_threshold_slider_change()
            else:
                self.threshold_slider.setValue(
                    int(max_mean_value * 0.1 * self.threshold_factor)
                )
                self.on_threshold_slider_change()
        else:
            self.threshold_slider.setValue(
                int(max_mean_value * 0.1 * self.threshold_factor)
            )
            self.on_threshold_slider_change()

    def calculate_lifetimes(self):
        """Calculate the lifetimes for all harmonics."""
        # TODO: when phasor_to_apparent_lifetimes can handle multiple
        # harmonics remove the for loop
        if self._labels_layer_with_phasor_features is None:
            return
        phasor_data = self._labels_layer_with_phasor_features.features
        frequency = float(self.frequency_input.text()) * np.array(
            self.harmonics
        )
        phase_lifetimes = []
        modulation_lifetimes = []
        for i in range(len(frequency)):
            harmonic_mask = phasor_data['harmonic'] == self.harmonics[i]
            real = phasor_data.loc[harmonic_mask, 'G']
            imag = phasor_data.loc[harmonic_mask, 'S']
            phase_lifetime, modulation_lifetime = phasor_to_apparent_lifetime(
                real, imag, frequency=frequency[i]
            )
            phase_lifetimes.append(phase_lifetime)
            modulation_lifetimes.append(modulation_lifetime)

        phase_lifetimes = np.nan_to_num(phase_lifetimes, nan=0)
        modulation_lifetimes = np.nan_to_num(modulation_lifetimes, nan=0)
        phase_lifetimes = np.clip(phase_lifetimes, a_min=0, a_max=None)
        modulation_lifetimes = np.clip(
            modulation_lifetimes, a_min=0, a_max=None
        )
        mean_shape = self._labels_layer_with_phasor_features.data.shape
        if self.lifetime_type_combobox.currentText() == "Phase":
            self.lifetime_data = np.reshape(
                phase_lifetimes, (len(self.harmonics),) + mean_shape
            )
        else:
            self.lifetime_data = np.reshape(
                modulation_lifetimes, (len(self.harmonics),) + mean_shape
            )

    def create_lifetime_layer(self):
        """Create or update the lifetime layer for all harmonics."""
        if self.lifetime_data is None:
            return

        lifetime_layer_name = f"{self.lifetime_type_combobox.currentText()} Lifetime: {self.layer_combobox.currentText()}"
        selected_lifetime_layer = Image(
            self.lifetime_data,
            name=lifetime_layer_name,
            scale=self._labels_layer_with_phasor_features.scale,
            colormap='turbo',
        )

        # Check if the layer is in the viewer before attempting to remove it
        if lifetime_layer_name in self.viewer.layers:
            self.viewer.layers.remove(self.viewer.layers[lifetime_layer_name])

        self.lifetime_layer = self.viewer.add_layer(selected_lifetime_layer)
        self.lifetime_colormap = self.lifetime_layer.colormap.colors
        self.lifetime_layer.events.colormap.connect(self._on_colormap_changed)
        self.colormap_contrast_limits = self.lifetime_layer.contrast_limits
        self.lifetime_layer.events.contrast_limits.connect(
            self._on_colormap_changed
        )

    def plot_lifetime_histogram(self):
        """Plot the histogram of the lifetime data as a line plot."""
        if self.lifetime_data is None:
            return
        if self._labels_layer_with_phasor_features is None:
            return
        if self.selected_harmonic is None:
            self.selected_harmonic = self.harmonics.min()

        self.harmonic_selector.setMinimum(self.harmonics.min())
        self.harmonic_selector.setMaximum(self.harmonics.max())

        harmonic_index = list(self.harmonics).index(self.selected_harmonic)
        lifetime_data = self.lifetime_data[harmonic_index]
        flattened_data = lifetime_data.flatten()
        flattened_data = flattened_data[flattened_data > 0]

        self.counts, self.bin_edges = np.histogram(flattened_data, bins=300)
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2

        self._update_lifetime_histogram()
        self.histogram_widget.show()

    def _update_lifetime_histogram(self):
        """Update the histogram plot with the current histogram values."""
        self.hist_ax.clear()
        self.hist_ax.plot(self.bin_centers, self.counts, color='none', alpha=0)

        # Create the colormap by linear combination of the napari cmap
        cmap = LinearSegmentedColormap.from_list(
            'custom_cmap', self.lifetime_colormap
        )
        norm = plt.Normalize(
            vmin=self.colormap_contrast_limits[0],
            vmax=self.colormap_contrast_limits[1],
        )

        for count, bin_start, bin_end in zip(
            self.counts, self.bin_edges[:-1], self.bin_edges[1:]
        ):
            bin_center = (bin_start + bin_end) / 2
            color = cmap(norm(bin_center))
            self.hist_ax.fill_between(
                [bin_start, bin_end], 0, count, color=color, alpha=0.7
            )

        self.hist_ax.set_title('Lifetime Distribution')
        self.hist_ax.set_xlabel('Lifetime (ns)')
        self.hist_ax.set_ylabel('Pixel count')

        # Refresh the canvas to show the updated histogram
        self.hist_fig.canvas.draw_idle()

    def _on_harmonic_changed(self):
        """Callback whenever the harmonic selector changes."""
        self.selected_harmonic = self.harmonic_selector.value()
        self.plot_lifetime_histogram()

    def _on_colormap_changed(self, event):
        """Callback whenever the colormap changes."""
        layer = event.source
        self.lifetime_colormap = layer.colormap.colors
        self.colormap_contrast_limits = layer.contrast_limits
        self._update_lifetime_histogram()

    def _on_click(self):
        """Callback whenever the plot lifetime button is clicked."""
        if self.frequency_input.text() == "":
            show_error("Enter frequency")
            return
        layer_name = self.layer_combobox.currentText()
        apply_filter_and_threshold(
            self.viewer.layers[layer_name],
            threshold=self.threshold_slider.value() / self.threshold_factor,
            repeat=0,
        )
        self.calculate_lifetimes()
        self.create_lifetime_layer()
        self.plot_lifetime_histogram()
