"""
This module contains widgets to:

    - Transform FLIM and hyperspectral images into phasor space from
      the following file formats: FBD, PTU, LSM, SDT, TIF, OME-TIFF.
    - Export phasor data to OME-TIFF or CSV files.

"""

from typing import TYPE_CHECKING

import numpy as np
from napari.layers import Image
from napari.utils.notifications import show_error, show_info
from qtpy.QtGui import QDoubleValidator
from qtpy.QtWidgets import (
    QComboBox,
    QCompleter,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

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
        from fbdfile import FbdFile

        with FbdFile(path) as fbd:
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
