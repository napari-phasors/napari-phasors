"""
This module contains widgets to:

    - Transform FLIM and hyperspectral images into phasor space from
      the following file formats: FBD, PTU, LSM, SDT, TIF, OME-TIFF.
    - Export phasor data to OME-TIFF or CSV files.

"""

import sys
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.figure import Figure
from napari.layers import Image
from napari.utils.notifications import show_error, show_info
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
    QVBoxLayout,
    QWidget,
)
from superqt import QRangeSlider

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

        self.setMinimumWidth(400)

        self.main_layout = QVBoxLayout(self)

        self.search_button = QPushButton("Select file to be read")
        self.search_button.clicked.connect(self._open_file_dialog)
        self.main_layout.addWidget(self.search_button)

        self.main_layout.addWidget(QLabel("Path to the selected file: "))

        self.save_path = QLineEdit()
        self.save_path.setReadOnly(True)
        self.main_layout.addWidget(self.save_path)

        self.dynamic_widget_layout = QVBoxLayout()
        self.main_layout.addLayout(self.dynamic_widget_layout)

        self.reader_options = {
            ".fbd": FbdWidget,
            ".ptu": PtuWidget,
            ".lsm": LsmWidget,
            ".tif": LsmWidget,
            ".ome.tif": OmeTifWidget,
            ".sdt": SdtWidget,
        }

    def _open_file_dialog(self):
        """Open a `QFileDialog` to select a file with specific extensions."""
        options = QFileDialog.Options()
        dialog = QFileDialog(self, "Select Export Location", options=options)
        dialog.setFileMode(QFileDialog.AnyFile)
        dialog.setNameFilter(
            "All files (*.tif *.ome.tif *.ptu *.fbd *.sdt *.lsm)"
        )
        if dialog.exec_():
            selected_file = dialog.selectedFiles()[0]
            self.save_path.setText(selected_file)
            _, extension = _get_filename_extension(selected_file)
            if extension in self.reader_options:
                for i in reversed(range(self.dynamic_widget_layout.count())):
                    widget = self.dynamic_widget_layout.takeAt(i).widget()
                    widget.deleteLater()

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
        if not hasattr(self, 'harmonics'):
            self.harmonics = [1]
        if not hasattr(self, 'has_phasor_settings'):
            self.has_phasor_settings = False
        if not hasattr(self, 'channels_data'):
            self.channels_data = {}
        if not hasattr(self, 'max_harmonic'):
            self.max_harmonic = 2

        self.figure = Figure(figsize=(6, 4))
        self.figure.patch.set_alpha(0.0)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setFixedHeight(300)
        self.canvas.setSizePolicy(
            self.canvas.sizePolicy().Expanding, self.canvas.sizePolicy().Fixed
        )
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor('none')

        for spine in self.ax.spines.values():
            spine.set_color('grey')
        self.ax.tick_params(colors='grey', which='both', labelsize=8)
        self.ax.xaxis.label.set_color('grey')
        self.ax.xaxis.label.set_fontsize(9)
        self.ax.yaxis.label.set_color('grey')
        self.ax.yaxis.label.set_fontsize(9)
        self.ax.title.set_color('grey')
        self.ax.title.set_fontsize(10)
        self.initUI()

    def initUI(self):
        """Initialize the user interface."""
        self.mainLayout = QVBoxLayout()
        self.setLayout(self.mainLayout)

        self.mainLayout.addWidget(self.canvas)

    def _update_signal_plot(self):
        """Update the signal plot based on current parameters."""
        try:
            self.ax.clear()
            plot_all_channels = (
                hasattr(self, 'channels')
                and self.channels is not None
                and self.channels.currentIndex() == 0
            )

            if (
                plot_all_channels
                and hasattr(self, 'all_channels')
                and self.all_channels > 1
            ):
                colors = plt.cm.tab10(np.linspace(0, 1, self.all_channels))

                for channel_idx in range(self.all_channels):
                    original_channel = self.reader_options.get("channel")
                    self.reader_options["channel"] = channel_idx
                    signal = self._get_signal_data()
                    self.reader_options["channel"] = original_channel

                    if signal is None:
                        continue

                    if channel_idx == 0:
                        self.max_harmonic = signal.shape[-1] // 2

                    if len(signal.shape) > 1:
                        axes_to_sum = tuple(range(len(signal.shape) - 1))
                        channel_signal = np.sum(signal, axis=axes_to_sum)
                    else:
                        channel_signal = signal

                    self.ax.plot(
                        channel_signal,
                        color=colors[channel_idx],
                        label=f'Channel {channel_idx}',
                    )

                self.ax.legend(
                    facecolor='black',
                    edgecolor='grey',
                    labelcolor='grey',
                    framealpha=0.8,
                )
                title = 'Signal Preview'
            else:
                signal = self._get_signal_data()
                if signal is None:
                    return

                self.max_harmonic = signal.shape[-1] // 2

                if len(signal.shape) > 1:
                    axes_to_sum = tuple(range(len(signal.shape) - 1))
                    summed_signal = np.sum(signal, axis=axes_to_sum)
                else:
                    summed_signal = signal
                self.ax.plot(summed_signal, color='white')
                legend = self.ax.get_legend()
                if legend:
                    legend.remove()
                if (
                    hasattr(self, 'channels')
                    and self.channels is not None
                    and self.channels.currentIndex() > 0
                ):
                    channel_num = self.channels.currentIndex() - 1
                    title = f'Signal Preview of Channel {channel_num}'
                elif (
                    hasattr(self, 'channels_single_label')
                    and self.channels_single_label is not None
                ):
                    channel_num = self.channels_single_label.text()
                    title = f'Signal Preview of Channel {channel_num}'
                else:
                    title = 'Signal Preview'

            self._update_harmonic_slider()

            self.ax.set_xlabel('Time / Histogram Bin')
            self.ax.set_ylabel('Total Signal (sum over pixels)')
            self.ax.set_title(title)

            for spine in self.ax.spines.values():
                spine.set_color('grey')
            self.ax.tick_params(colors='grey', which='both', labelsize=8)
            self.ax.xaxis.label.set_color('grey')
            self.ax.xaxis.label.set_fontsize(9)
            self.ax.yaxis.label.set_color('grey')
            self.ax.yaxis.label.set_fontsize(9)
            self.ax.title.set_color('grey')
            self.ax.title.set_fontsize(10)

            self.figure.tight_layout()
            self.canvas.draw()

        except Exception as e:
            show_error(f"Error updating signal plot: {str(e)}")

    def _get_signal_data(self):
        """Get signal data based on file type and current parameters.
        This method should be overridden by subclasses."""
        raise NotImplementedError("Subclasses must implement _get_signal_data")

    def _frame_widget(self):
        """Add the frame widget to main layout."""
        frame_layout = QHBoxLayout()
        frame_layout.addWidget(QLabel("Frames: "))
        self.frames = QComboBox()
        self.frames.addItems(["Average all frames"])
        for frame in range(self.all_frames):
            self.frames.addItem(str(frame))
        self.frames.setCurrentIndex(0)
        self.frames.currentIndexChanged.connect(
            self._on_frames_combobox_changed
        )
        frame_layout.addWidget(self.frames)
        frame_layout.addStretch()
        self.mainLayout.addLayout(frame_layout)

    def _channels_widget(self):
        """Add the channels widget to main layout."""
        self.channels_layout = QHBoxLayout()
        self.channels_layout.addWidget(QLabel("Channels: "))

        self.channels = None
        self.channels_single_label = None

        self.mainLayout.addLayout(self.channels_layout)

        self._update_channels_widget()

    def _update_channels_widget(self):
        """Create, update, or replace the channels widget."""
        if self.channels is not None:
            self.channels.setParent(None)
            self.channels.deleteLater()
            self.channels = None

        if self.channels_single_label is not None:
            self.channels_single_label.setParent(None)
            self.channels_single_label.deleteLater()
            self.channels_single_label = None

        if hasattr(self, 'all_channels') and self.all_channels > 1:
            self.channels = QComboBox()
            self.channels.addItems(["All channels"])
            for channel in range(self.all_channels):
                self.channels.addItem(str(channel))
            self.channels.setCurrentIndex(0)
            self.channels.currentIndexChanged.connect(
                self._on_channels_combobox_changed
            )
            self.channels_layout.addWidget(self.channels)
        else:
            self.channels_single_label = QLabel("0")
            self.channels_layout.addWidget(self.channels_single_label)
            self.reader_options["channel"] = 0

        self.channels_layout.addStretch()

    def _harmonic_widget(self):
        """Add the harmonic widget to main layout."""
        self.harmonic_layout = QHBoxLayout()
        self.harmonic_layout.addWidget(QLabel("Harmonics: "))

        self.harmonic_start_edit = None
        self.harmonic_end_edit = None
        self.harmonic_dash_label = None
        self.harmonic_single_label = None
        self.harmonic_slider = None

        self.mainLayout.addLayout(self.harmonic_layout)

        self.harmonics = [1]

    def _update_harmonic_edits(self):
        """Create, update, or replace the harmonic input widgets based on max_harmonic."""
        if self.harmonic_start_edit is not None:
            self.harmonic_start_edit.setParent(None)
            self.harmonic_start_edit.deleteLater()
            self.harmonic_start_edit = None

        if self.harmonic_end_edit is not None:
            self.harmonic_end_edit.setParent(None)
            self.harmonic_end_edit.deleteLater()
            self.harmonic_end_edit = None

        if self.harmonic_dash_label is not None:
            self.harmonic_dash_label.setParent(None)
            self.harmonic_dash_label.deleteLater()
            self.harmonic_dash_label = None

        if self.harmonic_single_label is not None:
            self.harmonic_single_label.setParent(None)
            self.harmonic_single_label.deleteLater()
            self.harmonic_single_label = None

        if self.max_harmonic > 1:
            self.harmonic_start_edit = QLineEdit()
            self.harmonic_start_edit.setText("1")
            self.harmonic_start_edit.setFixedWidth(50)
            self.harmonic_start_edit.setValidator(QDoubleValidator())
            self.harmonic_start_edit.editingFinished.connect(
                self._on_harmonic_edit_changed
            )
            self.harmonic_layout.addWidget(self.harmonic_start_edit)

            self.harmonic_dash_label = QLabel("-")
            self.harmonic_layout.addWidget(self.harmonic_dash_label)

            self.harmonic_end_edit = QLineEdit()
            self.harmonic_end_edit.setText(str(self.max_harmonic))
            self.harmonic_end_edit.setFixedWidth(50)
            self.harmonic_end_edit.setValidator(QDoubleValidator())
            self.harmonic_end_edit.editingFinished.connect(
                self._on_harmonic_edit_changed
            )
            self.harmonic_layout.addWidget(self.harmonic_end_edit)
        else:
            self.harmonic_single_label = QLabel("1")
            self.harmonic_layout.addWidget(self.harmonic_single_label)

    def _update_harmonic_slider(self):
        """Create, update, or hide the harmonic slider based on max_harmonic."""
        current_start, current_end = 1, 2
        if (
            hasattr(self, 'harmonic_slider')
            and self.harmonic_slider is not None
        ):
            current_start, current_end = self.harmonic_slider.value()

        if self.harmonic_slider is not None:
            self.harmonic_slider.setParent(None)
            self.harmonic_slider.deleteLater()
            self.harmonic_slider = None

        self._update_harmonic_edits()

        if self.max_harmonic > 1:
            self.harmonic_slider = QRangeSlider(Qt.Orientation.Horizontal)
            self.harmonic_slider.setRange(1, self.max_harmonic)

            preserved_start = max(1, min(current_start, self.max_harmonic))
            preserved_end = max(
                preserved_start, min(current_end, self.max_harmonic)
            )

            self.harmonic_slider.setValue((preserved_start, preserved_end))
            self.harmonic_slider.setBarMovesAllHandles(True)
            self.harmonic_slider.valueChanged.connect(
                self._on_harmonic_slider_changed
            )

            harmonic_layout_index = None
            for i in range(self.mainLayout.count()):
                item = self.mainLayout.itemAt(i)
                if item.layout() == self.harmonic_layout:
                    harmonic_layout_index = i
                    break

            if harmonic_layout_index is not None:
                self.mainLayout.insertWidget(
                    harmonic_layout_index + 1, self.harmonic_slider
                )
            else:
                self.mainLayout.addWidget(self.harmonic_slider)

            start, end = preserved_start, preserved_end
            self.harmonics = list(range(start, end + 1))

            if self.harmonic_end_edit:
                self.harmonic_start_edit.setText(str(start))
                self.harmonic_end_edit.setText(str(end))
        else:
            self.harmonics = [1] if self.max_harmonic >= 1 else []

    def _on_harmonic_edit_changed(self):
        """Callback whenever the harmonic line edits change."""
        if self.harmonic_start_edit is None or self.harmonic_end_edit is None:
            return

        try:
            start = int(float(self.harmonic_start_edit.text()))
            end = int(float(self.harmonic_end_edit.text()))

            if start < 1:
                start = 1
                self.harmonic_start_edit.setText(str(start))
            if end > self.max_harmonic:
                end = self.max_harmonic
                self.harmonic_end_edit.setText(str(end))
            if start > end:
                start = end
                self.harmonic_start_edit.setText(str(start))

            if self.harmonic_slider:
                self.harmonic_slider.setValue((start, end))

            self.harmonics = list(range(start, end + 1))

        except ValueError:
            if self.harmonic_slider:
                start, end = self.harmonic_slider.value()
            else:
                start, end = 1, self.max_harmonic
            self.harmonics = list(range(start, end + 1))
            self.harmonic_start_edit.setText(str(start))
            self.harmonic_end_edit.setText(str(end))

    def _on_harmonic_slider_changed(self, value):
        """Callback whenever the harmonic slider value changes."""
        if not self.harmonic_slider:
            return

        start, end = value
        self.harmonics = list(range(start, end + 1))

        if self.harmonic_start_edit:
            self.harmonic_start_edit.setText(str(start))
        if self.harmonic_end_edit:
            self.harmonic_end_edit.setText(str(end))

    def _on_frames_combobox_changed(self, index):
        """Callback whenever the frames combobox changes."""
        self.reader_options["frame"] = index - 1

    def _on_channels_combobox_changed(self, index):
        """Callback whenever the channels combobox changes."""
        if self.channels is None:
            return

        if index == 0:
            self.reader_options["channel"] = None
        else:
            self.reader_options["channel"] = index - 1

    def _on_click(self, path, reader_options, harmonics):
        """Callback whenever the calculate phasor button is clicked."""
        print('harmonics', harmonics)
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
        """Initialize the user interface."""
        self.mainLayout = QVBoxLayout()
        self.setLayout(self.mainLayout)

        self.mainLayout.addWidget(self.canvas)

        self._harmonic_widget()
        self._frame_widget()
        self._channels_widget()

        laser_layout = QHBoxLayout()
        laser_layout.addWidget(QLabel("Laser Factor (optional): "))
        self.laser_factor = QLineEdit()
        self.laser_factor.setText("-1")
        self.laser_factor.setToolTip(
            "Default is -1. If this doesn't work, "
            "most probable laser factors are: 0.00022, 2.50012, 2.50016"
        )
        self.laser_factor.setValidator(QDoubleValidator())
        laser_factor_completer = QCompleter(["0.00022", "2.50012", "2.50016"])
        self.laser_factor.setCompleter(laser_factor_completer)
        self.laser_factor.textChanged.connect(
            lambda: self._update_signal_plot()
        )
        laser_layout.addWidget(self.laser_factor)
        laser_layout.addStretch()
        self.mainLayout.addLayout(laser_layout)

        self.btn = QPushButton("Phasor Transform")
        self.btn.clicked.connect(
            lambda: self._on_click(
                self.path, self.reader_options, self.harmonics
            )
        )
        self.mainLayout.addWidget(self.btn)

        self._update_signal_plot()

    def _get_signal_data(self):
        """Get signal data for FBD files."""
        from phasorpy.io import signal_from_fbd

        options = self.reader_options.copy()
        if self.laser_factor.text():
            options["laser_factor"] = float(self.laser_factor.text())

        try:
            signal = signal_from_fbd(self.path, **options)
            return signal
        except Exception as e:
            show_error(f"Error reading FBD signal: {str(e)}")
            return None

    def _on_frames_combobox_changed(self, index):
        """Callback whenever the frames combobox changes."""
        super()._on_frames_combobox_changed(index)
        self._update_signal_plot()

    def _on_channels_combobox_changed(self, index):
        """Callback whenever the channels combobox changes."""
        super()._on_channels_combobox_changed(index)
        self._update_signal_plot()

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
        self.mainLayout = QVBoxLayout()
        self.setLayout(self.mainLayout)

        self.mainLayout.addWidget(self.canvas)

        self._harmonic_widget()
        self._frame_widget()
        self._channels_widget()

        dtime_layout = QHBoxLayout()
        dtime_layout.addWidget(QLabel("dtime (optional): "))
        self.dtime = QLineEdit()
        self.dtime.setText("0")
        self.dtime.setToolTip(
            "Specifies number of bins in image histogram."
            "If 0 (default), return number of bins in one period."
            "If < 0, integrate delay time axis."
            "If > 0, return up to specified bin."
        )
        self.dtime.textChanged.connect(lambda: self._update_signal_plot())
        dtime_layout.addWidget(self.dtime)
        dtime_layout.addStretch()
        self.mainLayout.addLayout(dtime_layout)

        self.btn = QPushButton("Phasor Transform")
        self.btn.clicked.connect(
            lambda: self._on_click(
                self.path, self.reader_options, self.harmonics
            )
        )
        self.mainLayout.addWidget(self.btn)

        self._update_signal_plot()

    def _get_signal_data(self):
        """Get signal data for PTU files."""
        from phasorpy.io import signal_from_ptu

        options = self.reader_options.copy()
        if self.dtime.text():
            options["dtime"] = float(self.dtime.text())

        try:
            signal = signal_from_ptu(self.path, **options)
            return signal
        except Exception as e:
            show_error(f"Error reading PTU signal: {str(e)}")
            return None

    def _on_frames_combobox_changed(self, index):
        """Callback whenever the frames combobox changes."""
        super()._on_frames_combobox_changed(index)
        self._update_signal_plot()

    def _on_channels_combobox_changed(self, index):
        """Callback whenever the channels combobox changes."""
        super()._on_channels_combobox_changed(index)
        self._update_signal_plot()

    def _on_click(self, path, reader_options, harmonics):
        """Callback whenever the calculate phasor button is clicked."""
        if self.dtime.text():
            reader_options["dtime"] = float(self.dtime.text())
        super()._on_click(path, reader_options, harmonics)


class LsmWidget(AdvancedOptionsWidget):
    """Widget for Zeiss LSM and TIFF files."""

    def __init__(self, viewer, path):
        """Initialize the widget."""
        self.path = path
        self._is_lsm = self._check_if_lsm(path)
        super().__init__(viewer, path)

    def _check_if_lsm(self, path):
        """Check if the file is an LSM file or regular TIFF."""
        try:
            import tifffile

            with tifffile.TiffFile(path) as tif:
                return tif.is_lsm
        except Exception:
            return False

    def initUI(self):
        """Initialize the user interface."""
        self.mainLayout = QVBoxLayout()
        self.setLayout(self.mainLayout)

        self.mainLayout.addWidget(self.canvas)

        self.channel_signal_labels = []

        self._harmonic_widget()

        self.btn = QPushButton("Phasor Transform")
        self.btn.clicked.connect(
            lambda: self._on_click(
                self.path, self.reader_options, self.harmonics
            )
        )
        self.mainLayout.addWidget(self.btn)

        self._update_signal_plot()

    def _get_signal_data(self):
        """Get signal data for LSM or TIFF files."""
        try:
            if self._is_lsm:
                from phasorpy.io import signal_from_lsm

                return signal_from_lsm(self.path)
            else:
                import tifffile

                signal = tifffile.imread(self.path)
                return signal
        except Exception as e:
            show_error(
                f"Error reading {'LSM' if self._is_lsm else 'TIFF'} signal: {str(e)}"
            )
            return None

    def _update_signal_plot(self):
        """Update the signal plot for LSM/TIFF files (spectral data)."""
        try:
            signal = self._get_signal_data()
            if signal is None:
                return

            if signal.shape[0] > 0:
                self.max_harmonic = signal.shape[0] // 2

            self._update_harmonic_slider()

            self.ax.clear()

            if len(signal.shape) > 1:
                axes_to_sum = tuple(range(1, len(signal.shape)))
                summed_signal = np.sum(signal, axis=axes_to_sum)
            else:
                summed_signal = signal

            self.ax.plot(summed_signal, color='white')

            xlabel = (
                'Spectral Channels'
                if self._is_lsm
                else 'Time / Spectral Channels'
            )
            self.ax.set_xlabel(xlabel)
            self.ax.set_ylabel('Total Signal (sum over pixels)')
            self.ax.set_title('Signal Preview')

            for spine in self.ax.spines.values():
                spine.set_color('grey')
            self.ax.tick_params(colors='grey', which='both', labelsize=8)
            self.ax.xaxis.label.set_color('grey')
            self.ax.xaxis.label.set_fontsize(9)
            self.ax.yaxis.label.set_color('grey')
            self.ax.yaxis.label.set_fontsize(9)
            self.ax.title.set_color('grey')
            self.ax.title.set_fontsize(10)

            self.figure.tight_layout()
            self.canvas.draw()

        except Exception as e:
            show_error(f"Error updating signal plot: {str(e)}")


class SdtWidget(AdvancedOptionsWidget):
    """Widget for SDT files."""

    def __init__(self, viewer, path):
        """Initialize the widget."""
        super().__init__(viewer, path)

    def initUI(self):
        """Initialize the user interface."""
        self.mainLayout = QVBoxLayout()
        self.setLayout(self.mainLayout)

        self.mainLayout.addWidget(self.canvas)

        self._harmonic_widget()

        index_layout = QHBoxLayout()
        index_layout.addWidget(QLabel("Index (optional): "))
        self.index = QLineEdit()
        self.index.setText("0")
        self.index.setToolTip(
            "Index of dataset to read in case the file contains multiple "
            "datasets. By default, the first dataset is read."
        )
        self.index.textChanged.connect(lambda: self._update_signal_plot())
        index_layout.addWidget(self.index)
        index_layout.addStretch()
        self.mainLayout.addLayout(index_layout)

        self.btn = QPushButton("Phasor Transform")
        self.btn.clicked.connect(
            lambda: self._on_click(
                self.path, self.reader_options, self.harmonics
            )
        )
        self.mainLayout.addWidget(self.btn)

        self._update_signal_plot()

    def _get_signal_data(self):
        """Get signal data for SDT files."""
        from phasorpy.io import signal_from_sdt

        options = self.reader_options.copy()
        if self.index.text():
            options["index"] = int(self.index.text())

        try:
            signal = signal_from_sdt(self.path, **options)
            return signal
        except Exception as e:
            show_error(f"Error reading SDT signal: {str(e)}")
            return None

    def _on_click(self, path, reader_options, harmonics):
        """Callback whenever the calculate phasor button is clicked."""
        if self.index.text():
            reader_options["index"] = int(self.index.text())
        super()._on_click(path, reader_options, harmonics)


class OmeTifWidget(AdvancedOptionsWidget):
    """Widget for OME-TIFF files with phasor data."""

    def __init__(self, viewer, path):
        """Initialize the widget."""
        self.has_phasor_settings = False
        self.channels_data = {}
        self._load_settings(path)
        super().__init__(viewer, path)

    def _load_settings(self, path):
        """Load settings from OME-TIFF metadata if available."""
        try:
            import json

            from phasorpy.io import phasor_from_ometiff

            _, _, _, attrs = phasor_from_ometiff(path, harmonic='all')

            if "harmonic" in attrs:
                harmonics = attrs["harmonic"]
                if isinstance(harmonics, (list, np.ndarray)):
                    self.max_harmonic = int(np.max(harmonics))
                else:
                    self.max_harmonic = int(harmonics)

            if "description" in attrs.keys():
                description = json.loads(attrs["description"])
                if sys.getsizeof(description) > 512 * 512:  # Threshold: 256 KB
                    raise ValueError("Description dictionary is too large.")
                if "napari_phasors_settings" in description:
                    settings = json.loads(
                        description["napari_phasors_settings"]
                    )

                    if 'summed_signal' in settings:
                        self.has_phasor_settings = True
                        channel = settings.get('channel', 0)
                        self.channels_data[channel] = {
                            'summed_signal': np.array(
                                settings['summed_signal']
                            ),
                            'settings': settings,
                        }

                        if 'frequency' in settings:
                            self.frequency = settings['frequency']
                        elif 'frequency' in attrs:
                            self.frequency = attrs['frequency']

        except Exception as e:
            pass

    def initUI(self):
        """Initialize the user interface."""
        self.mainLayout = QVBoxLayout()
        self.setLayout(self.mainLayout)

        self.mainLayout.addWidget(self.canvas)

        self._harmonic_widget()

        self._update_harmonic_slider()

        if (
            self.has_phasor_settings
            and len(self.channels_data) > 0
            and self._has_channel_info()
        ):
            self._channels_widget_from_settings()

        self.btn = QPushButton("Phasor Transform")
        self.btn.clicked.connect(
            lambda: self._on_click(
                self.path, self.reader_options, self.harmonics
            )
        )
        self.mainLayout.addWidget(self.btn)

        self._update_signal_plot()

    def _has_channel_info(self):
        """Check if any of the stored settings contain channel information."""
        for channel_data in self.channels_data.values():
            settings = channel_data.get('settings', {})
            if 'channel' in settings and settings['channel'] is not None:
                return True
        return False

    def _channels_widget_from_settings(self):
        """Add the channel widget based on stored settings."""
        self.channels_layout = QHBoxLayout()
        self.channels_layout.addWidget(QLabel("Channels: "))

        self.channels = None
        self.channels_single_label = None

        if len(self.channels_data) > 1:
            self.channels = QComboBox()

            for channel_idx in sorted(self.channels_data.keys()):
                self.channels.addItem(str(channel_idx))

            self.channels.setCurrentIndex(0)
            self.channels.currentIndexChanged.connect(
                self._on_channels_combobox_changed_settings
            )
            self.channels_layout.addWidget(self.channels)
        else:
            if self.channels_data:
                channel_idx = list(self.channels_data.keys())[0]
                self.channels_single_label = QLabel(str(channel_idx))
            else:
                self.channels_single_label = QLabel("0")
            self.channels_layout.addWidget(self.channels_single_label)

        self.channels_layout.addStretch()
        self.mainLayout.addLayout(self.channels_layout)

    def _on_channels_combobox_changed_settings(self, index):
        """Callback when channel selection changes for OME-TIFF with settings."""
        if not hasattr(self, 'channels') or self.channels is None:
            return
        self._update_signal_plot()

    def _get_signal_data(self):
        """Get signal data from stored settings."""
        if not self.has_phasor_settings:
            return None

        if (
            hasattr(self, 'channels')
            and self.channels is not None
            and self.channels.count() > 0
        ):
            channel_idx = int(self.channels.currentText())
        else:
            if not self.channels_data:
                return None
            channel_idx = list(self.channels_data.keys())[0]

        if channel_idx in self.channels_data:
            return self.channels_data[channel_idx]['summed_signal']

        return None

    def _update_signal_plot(self):
        """Update the signal plot for OME-TIFF files."""
        try:
            signal = self._get_signal_data()

            if signal is None:
                return
            self._update_harmonic_slider()

            self.ax.clear()
            self.ax.plot(signal, color='white')

            if hasattr(self, 'channels') and self.channels is not None:
                channel_num = int(self.channels.currentText())
                title = f'Signal Preview of Channel {channel_num}'
            elif (
                hasattr(self, 'channels_single_label')
                and self.channels_single_label is not None
            ):
                channel_num = self.channels_single_label.text()
                title = f'Signal Preview of Channel {channel_num}'
            else:
                title = 'Signal Preview'

            self.ax.set_xlabel('Histogram Bin / Spectral Channel')
            self.ax.set_ylabel('Total Signal (sum over pixels)')
            self.ax.set_title(title)

            for spine in self.ax.spines.values():
                spine.set_color('grey')
            self.ax.tick_params(colors='grey', which='both', labelsize=8)
            self.ax.xaxis.label.set_color('grey')
            self.ax.xaxis.label.set_fontsize(9)
            self.ax.yaxis.label.set_color('grey')
            self.ax.yaxis.label.set_fontsize(9)
            self.ax.title.set_color('grey')
            self.ax.title.set_fontsize(10)

            self.figure.tight_layout()
            self.canvas.draw()

        except Exception as e:
            show_error(f"Error updating signal plot: {str(e)}")

    def _on_click(self, path, reader_options, harmonics):
        """Execute the phasor transformation."""
        try:
            reader = napari_get_reader(path, reader_options, harmonics)
            if reader:
                for layer in reader(path):
                    self.viewer.add_image(
                        layer[0],
                        name=layer[1]["name"],
                        metadata=layer[1]["metadata"],
                    )
        except Exception as e:
            show_error(f"Error during phasor transformation: {str(e)}")


class WriterWidget(QWidget):
    """Widget to export phasor data to a OME-TIF or CSV file."""

    def __init__(self, viewer: "napari.viewer.Viewer"):
        """Initialize the widget."""
        super().__init__()
        self.viewer = viewer

        self.main_layout = QVBoxLayout(self)

        self.main_layout.addWidget(
            QLabel("Select Image Layer to be Exported: ")
        )
        self.export_layer_combobox = QComboBox()
        self.main_layout.addWidget(self.export_layer_combobox)

        self.search_button = QPushButton("Select Export Location and Name")
        self.search_button.clicked.connect(self._open_file_dialog)
        self.main_layout.addWidget(self.search_button)

        self.viewer.layers.events.inserted.connect(self._populate_combobox)
        self.viewer.layers.events.removed.connect(self._populate_combobox)

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

        file_dialog.setNameFilters(
            ["Phasor as OME-TIFF (*.ome.tif)", "Phasor table as CSV (*.csv)"]
        )

        if file_dialog.exec_():
            selected_filter = file_dialog.selectedNameFilter()
            file_path = file_dialog.selectedFiles()[0]

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

            coords = np.unravel_index(
                np.arange(export_layer.data.size), export_layer.data.shape
            )

            coords = [np.tile(coord, len(harmonics)) for coord in coords]

            for dim, coord in enumerate(coords):
                phasor_table[f'dim_{dim}'] = coord

            phasor_table = phasor_table.dropna()
            phasor_table.to_csv(
                file_path,
                index=False,
            )
        show_info(f"Exported {export_layer.name} to {file_path}")
