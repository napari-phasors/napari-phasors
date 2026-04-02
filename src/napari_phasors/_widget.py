"""
This module contains widgets to:

    - Transform FLIM and hyperspectral images into phasor space from
      the following file formats: FBD, PTU, LSM, SDT, TIF, OME-TIFF.
    - Export phasor data to OME-TIFF or CSV files.

"""

import glob
import json
import os
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
from qtpy.QtGui import QDoubleValidator, QIntValidator
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QCompleter,
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from superqt import QRangeSlider

from ._reader import (
    _get_filename_extension,
    napari_get_reader,
    raw_file_stack_reader,
)
from ._utils import (
    CheckableComboBox,
    CollapsibleSection,
    FileOrderDialog,
    natural_sort_key,
)
from ._writer import export_layer_as_csv, export_layer_as_image, write_ome_tiff

if TYPE_CHECKING:
    import napari


class PhasorTransform(QWidget):
    """Widget to transform FLIM and hyperspectral images into phasor space."""

    def __init__(self, viewer: "napari.viewer.Viewer"):
        """Initialize the widget."""
        super().__init__()
        self.viewer = viewer

        self.setMinimumWidth(400)

        self.outer_layout = QVBoxLayout(self)
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.outer_layout.addWidget(self.scroll_area)

        self.content_widget = QWidget()
        self.main_layout = QVBoxLayout(self.content_widget)
        self.main_layout.setAlignment(Qt.AlignTop)
        self.scroll_area.setWidget(self.content_widget)

        self.search_button = QPushButton("Select file(s) to be read")
        self.search_button.clicked.connect(self._open_file_dialog)
        self.main_layout.addWidget(self.search_button)

        self.multi_file_button = QPushButton("Open 3D stack")
        self.multi_file_button.clicked.connect(self._open_multi_file_dialog)
        self.main_layout.addWidget(self.multi_file_button)

        self.main_layout.addWidget(QLabel("Path to the selected file(s): "))

        self.save_path = QLineEdit()
        self.save_path.setReadOnly(True)
        self.main_layout.addWidget(self.save_path)

        self.selected_paths_list = QListWidget()
        self.selected_paths_list.hide()
        self.main_layout.addWidget(self.selected_paths_list)

        self.dynamic_widget_layout = QVBoxLayout()
        self.dynamic_widget_layout.setAlignment(Qt.AlignTop)
        self.main_layout.addLayout(self.dynamic_widget_layout)

        self.reader_options = {
            ".fbd": FbdWidget,
            ".ptu": PtuWidget,
            ".lsm": LsmWidget,
            ".tif": LsmWidget,
            ".tiff": LsmWidget,
            ".ome.tif": OmeTifWidget,
            ".ome.tiff": OmeTifWidget,
            ".sdt": SdtWidget,
            ".czi": CziWidget,
            ".flif": FlifWidget,
            ".bh": BhWidget,
            ".b&h": BhWidget,
            ".bhz": BhWidget,
            ".bin": PqbinWidget,
            ".r64": SimfcsWidget,
            ".ref": SimfcsWidget,
            ".ifli": IfliWidget,
            ".lif": LifWidget,
            ".json": JsonWidget,
        }

    def _open_file_dialog(self):
        """Open a dialog to select one or many files for import.

        Single file: show the per-format options widget.
        Multiple files: create grouped option widgets by extension.
        """
        supported_filter = (
            "All files (*.tif *.tiff *.ome.tif *.ome.tiff *.ptu *.fbd *.sdt "
            "*.lsm *.czi *.flif *.bh *.b&h *.bhz *.bin *.r64 *.ref *.ifli "
            "*.lif *.json)"
        )
        selected_files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select file(s) to be read",
            "",
            supported_filter,
        )
        if not selected_files:
            return

        selected_files = sorted(selected_files, key=natural_sort_key)
        if len(selected_files) == 1:
            selected_file = selected_files[0]
            self._show_path_text(selected_file)
            _, extension = _get_filename_extension(selected_file)
            if extension in self.reader_options:
                self._clear_dynamic_widgets()

                create_widget_class = self.reader_options[extension]
                new_widget = create_widget_class(self.viewer, selected_file)
                self.dynamic_widget_layout.addWidget(new_widget)
            else:
                show_error(f"Extension {extension} is not supported.")
            return

        self._clear_dynamic_widgets()
        grouped_paths: dict[str, list[str]] = {}
        unsupported = []
        for file_path in selected_files:
            _, extension = _get_filename_extension(file_path)
            if extension not in self.reader_options:
                unsupported.append(os.path.basename(file_path))
                continue
            grouped_paths.setdefault(extension, []).append(file_path)

        if not grouped_paths:
            show_error("No supported files found in the selection.")
            return

        summary_parts = []
        for extension, paths in sorted(grouped_paths.items()):
            summary_parts.append(f"{extension}: {len(paths)}")

            create_widget_class = self.reader_options[extension]
            new_widget = create_widget_class(self.viewer, paths[0])
            new_widget._grouped_file_paths = paths
            if hasattr(new_widget, 'btn'):
                new_widget.btn.setText(
                    f"Phasor Transform Group ({len(paths)} file(s))"
                )

            group_container = CollapsibleSection(
                title=f"Group {extension} ({len(paths)} file(s))",
                initially_collapsed=False,
                text_color="#c7c7c7",
            )
            group_container.setSizePolicy(
                QSizePolicy.Preferred,
                QSizePolicy.Maximum,
            )
            new_widget.setSizePolicy(
                QSizePolicy.Preferred,
                QSizePolicy.Maximum,
            )
            group_container.add_widget(new_widget)
            self.dynamic_widget_layout.addWidget(group_container)

        self.dynamic_widget_layout.addStretch()
        self._show_path_list(selected_files)

    def _open_multi_file_dialog(self):
        """Open one dialog to select either a file or a directory."""
        file_paths = []

        supported_extensions = (
            "*.ome.tif",
            "*.tif",
            "*.lsm",
            "*.ptu",
            "*.fbd",
            "*.sdt",
            "*.czi",
            "*.flif",
            "*.bh",
            "*.b&h",
            "*.bhz",
            "*.bin",
            "*.r64",
            "*.ref",
            "*.ifli",
            "*.lif",
            "*.json",
        )

        selected_entries, _ = QFileDialog.getOpenFileNames(
            self,
            "Select files for 3D stack",
            "",
            "Supported files (" + " ".join(supported_extensions) + ")",
        )

        if not selected_entries:
            return

        for entry in selected_entries:
            if os.path.isfile(entry):
                file_paths.append(entry)
            elif os.path.isdir(entry):
                for ext in supported_extensions:
                    file_paths.extend(glob.glob(os.path.join(entry, ext)))

        # Remove duplicates while preserving order
        file_paths = list(dict.fromkeys(file_paths))

        if not file_paths:
            show_error("No supported files found in the selection.")
            return

        # Validate all files share the same extension
        extension_set = set()
        for p in file_paths:
            _, ext = _get_filename_extension(p)
            extension_set.add(ext)

        normalised = set(extension_set)
        if len(normalised) > 1:
            show_error(
                "All selected files must have the same extension. "
                f"Found: {extension_set}"
            )
            return

        common_ext = normalised.pop()
        if common_ext not in self.reader_options:
            show_error(f"Extension {common_ext} is not supported.")
            return

        # Auto-sort with natural sort
        file_paths = sorted(file_paths, key=natural_sort_key)

        estimated_shape = _estimate_result_shape(file_paths)

        initial_z_spacing = None
        if len(file_paths) == 1:
            initial_z_spacing = _try_get_z_spacing_from_ome_tiff(file_paths[0])

        # Show reorder dialog with z-spacing control
        dialog = FileOrderDialog(
            file_paths,
            parent=self,
            initial_z_spacing=initial_z_spacing,
            estimated_shape=estimated_shape,
        )
        if dialog.exec_() != FileOrderDialog.Accepted:
            return
        file_paths = dialog.get_ordered_paths()
        z_spacing = dialog.get_z_spacing()
        if hasattr(dialog, 'get_axis_order'):
            axis_order = dialog.get_axis_order()
            axis_labels = dialog.get_axis_labels()
        else:
            axis_order = None
            axis_labels = None

        # Update the path display
        self._show_path_text(
            f"{len(file_paths)} file(s) selected (first: "
            f"{os.path.basename(file_paths[0])}, z spacing: {z_spacing} um)"
        )

        # Clear old dynamic widgets and create the sub-widget
        self._clear_dynamic_widgets()

        create_widget_class = self.reader_options[common_ext]
        # Create the sub-widget using the first file for preview,
        # but store the full list for stacking.
        new_widget = create_widget_class(self.viewer, file_paths[0])
        new_widget._multi_file_paths = file_paths
        new_widget._stack_z_spacing = z_spacing
        new_widget._stack_axis_order = axis_order
        new_widget._stack_axis_labels = axis_labels
        new_widget.setSizePolicy(
            QSizePolicy.Preferred,
            QSizePolicy.Maximum,
        )
        if hasattr(new_widget, '_update_signal_plot'):
            new_widget._update_signal_plot()
        self.dynamic_widget_layout.addWidget(new_widget)

    def _clear_dynamic_widgets(self):
        """Remove all widgets from the dynamic widget layout."""
        for i in reversed(range(self.dynamic_widget_layout.count())):
            widget = self.dynamic_widget_layout.takeAt(i).widget()
            if widget is not None:
                widget.deleteLater()

    def _show_path_text(self, text: str):
        """Show one-line path/status text and hide the multi-path list."""
        self.selected_paths_list.hide()
        self.save_path.show()
        self.save_path.setText(text)

    def _show_path_list(self, paths: list[str]):
        """Show selected paths in a scrollable list with a 3-row viewport."""
        self.save_path.hide()
        self.selected_paths_list.show()
        self.selected_paths_list.clear()
        for path in paths:
            self.selected_paths_list.addItem(path)

        if paths:
            row_height = self.selected_paths_list.sizeHintForRow(0)
            if row_height <= 0:
                row_height = 22
            visible_rows = min(3, len(paths))
            frame_height = 2 * self.selected_paths_list.frameWidth()
            self.selected_paths_list.setFixedHeight(
                row_height * visible_rows + frame_height + 2
            )


class AdvancedOptionsWidget(QWidget):
    """Base class for advanced options widgets."""

    def __init__(self, viewer, path):
        """Initialize the widget."""
        super().__init__()
        self.viewer = viewer
        self.path = path
        self._stack_z_spacing_layout = None
        self._stack_z_spacing_edit = None
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
        # After initUI (which subclasses override) ensure the layout and size
        # policy never cause vertical stretching inside grouped containers.
        if hasattr(self, 'mainLayout') and self.mainLayout is not None:
            self.mainLayout.setAlignment(Qt.AlignTop)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        self._add_shape_preview_widget()
        self._update_shape_preview()

    def initUI(self):
        """Initialize the user interface."""
        self.mainLayout = QVBoxLayout()
        self.setLayout(self.mainLayout)

        self.mainLayout.addWidget(self.canvas)

    def _add_shape_preview_widget(self):
        """Add a label showing estimated output shape for current options."""
        self.shape_preview_label = QLabel("Estimated output shape: N/A")
        if hasattr(self, 'btn'):
            btn_index = self.mainLayout.indexOf(self.btn)
            if btn_index >= 0:
                self.mainLayout.insertWidget(
                    btn_index, self.shape_preview_label
                )
                return
        self.mainLayout.addWidget(self.shape_preview_label)

    def _update_shape_preview(self):
        """Update estimated output shape when reader options change."""
        if not hasattr(self, 'shape_preview_label'):
            return

        n_files = len(getattr(self, '_multi_file_paths', []) or [self.path])
        shape = _estimate_output_shape_from_options(
            self.path,
            self.reader_options,
            self.harmonics,
            n_files=n_files,
        )

        axis_order = getattr(self, '_stack_axis_order', None)
        axis_labels = getattr(self, '_stack_axis_labels', None)
        if (
            shape is not None
            and axis_order is not None
            and len(axis_order) == len(shape)
            and sorted(axis_order) == list(range(len(shape)))
        ):
            shape = tuple(shape[i] for i in axis_order)

        shape_text = str(tuple(shape)) if shape is not None else "N/A"
        if (
            axis_labels is not None
            and shape is not None
            and len(axis_labels) == len(shape)
        ):
            shape_text += f" ({', '.join(axis_labels)})"
        elif shape is not None:
            if len(shape) == 2:
                shape_text += " (Y, X)"
            elif len(shape) == 3:
                shape_text += " (Z, Y, X)"
            elif len(shape) == 4:
                shape_text += " (T, Z, Y, X)"

        self.shape_preview_label.setText(
            f"Estimated output shape: {shape_text}"
        )

    def _get_preview_signal_data(self):
        """Return preview signal, averaging across selected stack/grouped files."""
        multi_paths = getattr(self, '_multi_file_paths', None)
        grouped_paths = getattr(self, '_grouped_file_paths', None)

        # Prefer grouped paths over multi-file-stack paths
        paths_to_average = None
        if grouped_paths and len(grouped_paths) > 1:
            paths_to_average = grouped_paths
        elif multi_paths and len(multi_paths) > 1:
            paths_to_average = multi_paths

        if paths_to_average is None:
            return self._get_signal_data()

        original_path = self.path
        signals = []
        try:
            for path in paths_to_average:
                self.path = path
                signal = self._get_signal_data()
                if signal is None:
                    continue
                signals.append(np.asarray(signal))
        finally:
            self.path = original_path

        if not signals:
            return None

        try:
            return np.mean(np.stack(signals, axis=0), axis=0)
        except ValueError:
            return signals[0]

    def _update_signal_plot(self):
        """Update the signal plot based on current parameters."""
        try:
            self._sync_stack_z_spacing_widget_visibility()
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
                    signal = self._get_preview_signal_data()
                    self.reader_options["channel"] = original_channel

                    if signal is None:
                        continue

                    channel_signal = self._collapse_signal_for_plot(signal)
                    if channel_signal is None or channel_signal.size == 0:
                        continue

                    if channel_idx == 0:
                        self.max_harmonic = channel_signal.shape[0] // 2

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
                signal = self._get_preview_signal_data()
                if signal is None:
                    return

                summed_signal = self._collapse_signal_for_plot(signal)
                if summed_signal is None or summed_signal.size == 0:
                    return

                self.max_harmonic = summed_signal.shape[0] // 2
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

            self.figure.tight_layout(pad=1.0)
            self.figure.subplots_adjust(left=0.15)
            self.canvas.draw()
            self._update_shape_preview()

        except Exception as e:  # noqa: BLE001
            show_error(f"Error updating signal plot: {str(e)}")

    @staticmethod
    def _choose_signal_axis(shape, axis_labels=None):
        """Return the axis index corresponding to histogram/spectral bins."""
        ndim = len(shape)
        if ndim == 0:
            return 0

        if axis_labels is not None and len(axis_labels) == ndim:
            labels = [str(label).strip().upper() for label in axis_labels]

            # Prefer explicit histogram/time/spectral labels.
            for candidates in (
                {"H", "HIST", "HISTOGRAM"},
                {"C", "CHANNEL", "CHANNELS", "L", "WL", "W"},
                {"T", "TIME"},
            ):
                for idx, label in enumerate(labels):
                    if label in candidates or any(
                        token in label
                        for token in ("HIST", "SPECT", "WAVEL", "LAMBDA")
                    ):
                        return idx

            # Avoid spatial axes if possible.
            non_spatial = [
                idx
                for idx, label in enumerate(labels)
                if label not in {"Z", "Y", "X"}
            ]
            if non_spatial:
                return non_spatial[-1]

        # Fallback to last axis, preserving previous behavior.
        return ndim - 1

    @classmethod
    def _collapse_signal_for_plot(cls, signal):
        """Collapse any signal array to a 1-D profile for plotting.

        The profile is computed along histogram/spectral axis while summing all
        remaining axes (e.g., Z/Y/X or stack axes).
        """
        axis_labels = None
        if hasattr(signal, "dims"):
            axis_labels = tuple(signal.dims)

        array = np.asarray(signal)
        if array.ndim == 0:
            return np.array([float(array)])
        if array.ndim == 1:
            return array

        signal_axis = cls._choose_signal_axis(array.shape, axis_labels)
        axes_to_sum = tuple(i for i in range(array.ndim) if i != signal_axis)
        return np.sum(array, axis=axes_to_sum)

    def _sync_stack_z_spacing_widget_visibility(self):
        """Show a z-spacing editor only for stacked multi-file imports."""
        is_stack_mode = len(getattr(self, '_multi_file_paths', []) or []) > 1

        if is_stack_mode and self._stack_z_spacing_layout is None:
            z_layout = QHBoxLayout()
            z_layout.addWidget(QLabel("Z spacing (um): "))

            self._stack_z_spacing_edit = QLineEdit()
            self._stack_z_spacing_edit.setValidator(
                QDoubleValidator(0.0, 1e12, 8)
            )
            initial = getattr(self, '_stack_z_spacing', None)
            if initial is None:
                initial = 1.0
            self._stack_z_spacing_edit.setText(str(initial))
            self._stack_z_spacing_edit.setToolTip(
                "Spacing along Z in micrometers (um)."
            )
            self._stack_z_spacing_edit.editingFinished.connect(
                self._on_stack_z_spacing_changed
            )

            z_layout.addWidget(self._stack_z_spacing_edit)
            z_layout.addStretch()

            inserted = False
            if hasattr(self, 'shape_preview_label'):
                idx = self.mainLayout.indexOf(self.shape_preview_label)
                if idx >= 0:
                    self.mainLayout.insertLayout(idx, z_layout)
                    inserted = True

            if not inserted:
                self.mainLayout.addLayout(z_layout)
            self._stack_z_spacing_layout = z_layout

        if not is_stack_mode and self._stack_z_spacing_layout is not None:
            while self._stack_z_spacing_layout.count():
                item = self._stack_z_spacing_layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
            self._stack_z_spacing_layout = None
            self._stack_z_spacing_edit = None

    def _on_stack_z_spacing_changed(self):
        """Validate and store z-spacing value edited in stack mode."""
        if self._stack_z_spacing_edit is None:
            return

        text = self._stack_z_spacing_edit.text().strip()
        try:
            value = float(text)
        except ValueError:
            value = 1.0

        if value <= 0:
            value = 1.0

        self._stack_z_spacing = value
        self._stack_z_spacing_edit.setText(str(value))

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
        self._update_shape_preview()

    def _on_channels_combobox_changed(self, index):
        """Callback whenever the channels combobox changes."""
        if self.channels is None:
            return

        if index == 0:
            self.reader_options["channel"] = None
        else:
            self.reader_options["channel"] = index - 1
        self._update_shape_preview()

    def _on_click(self, path, reader_options, harmonics):
        """Callback whenever the calculate phasor button is clicked.

        If ``_multi_file_paths`` is set, all files are stacked into a
        single 3D layer via :func:`raw_file_stack_reader`.
        """
        grouped_paths = getattr(self, '_grouped_file_paths', None)
        multi_paths = getattr(self, '_multi_file_paths', None)
        self._on_stack_z_spacing_changed()
        z_spacing = getattr(self, '_stack_z_spacing', None)
        axis_order = getattr(self, '_stack_axis_order', None)
        axis_labels = getattr(self, '_stack_axis_labels', None)
        if grouped_paths and len(grouped_paths) > 1:
            imported = 0
            failed = []
            for group_path in grouped_paths:
                reader = napari_get_reader(
                    group_path,
                    reader_options=reader_options,
                    harmonics=harmonics,
                )
                if reader is None:
                    failed.append((os.path.basename(group_path), "no reader"))
                    continue

                try:
                    for layer in reader(group_path):
                        add_kw = dict(layer[1].items())
                        layer_data = self._apply_axis_transform(
                            add_kw, layer[0], axis_order, axis_labels
                        )
                        self._set_layer_z_scale(add_kw, layer_data, z_spacing)
                        self.viewer.add_image(
                            layer_data,
                            name=add_kw.pop("name"),
                            metadata=add_kw.pop("metadata"),
                            **add_kw,
                        )
                    imported += 1
                except Exception as e:  # noqa: BLE001
                    failed.append((os.path.basename(group_path), str(e)))

            if failed:
                failed_names = ", ".join(name for name, _ in failed)
                show_error(
                    f"Imported {imported}/{len(grouped_paths)} file(s). "
                    f"Failed: {failed_names}."
                )
            return

        if multi_paths and len(multi_paths) > 1:
            layers = raw_file_stack_reader(
                multi_paths,
                reader_options=reader_options,
                harmonics=harmonics,
            )
            for layer in layers:
                add_kw = dict(layer[1].items())
                layer_data = self._apply_axis_transform(
                    add_kw, layer[0], axis_order, axis_labels
                )
                self._set_layer_z_scale(add_kw, layer_data, z_spacing)
                self.viewer.add_image(
                    layer_data,
                    name=add_kw.pop("name"),
                    metadata=add_kw.pop("metadata"),
                    **add_kw,
                )
        else:
            reader = napari_get_reader(
                path, reader_options=reader_options, harmonics=harmonics
            )
            for layer in reader(path):
                add_kw = dict(layer[1].items())
                layer_data = self._apply_axis_transform(
                    add_kw, layer[0], axis_order, axis_labels
                )
                self._set_layer_z_scale(add_kw, layer_data, z_spacing)
                self.viewer.add_image(
                    layer_data,
                    name=add_kw.pop("name"),
                    metadata=add_kw.pop("metadata"),
                    **add_kw,
                )

    @staticmethod
    def _apply_axis_transform(add_kwargs, data, axis_order, axis_labels):
        """Apply optional axis reordering and axis labels to data/metadata."""
        transformed = data
        valid_order = (
            axis_order is not None
            and hasattr(data, 'ndim')
            and len(axis_order) == data.ndim
            and sorted(axis_order) == list(range(data.ndim))
        )

        if valid_order and tuple(axis_order) != tuple(range(data.ndim)):
            transformed = np.transpose(data, axes=axis_order)

            metadata = add_kwargs.get("metadata", {})
            for key in (
                "original_mean",
                "G",
                "S",
                "G_original",
                "S_original",
                "mask",
            ):
                arr = metadata.get(key)
                if arr is None or not hasattr(arr, 'ndim'):
                    continue
                if arr.ndim == data.ndim:
                    metadata[key] = np.transpose(arr, axes=axis_order)
                elif arr.ndim == data.ndim + 1:
                    metadata[key] = np.transpose(
                        arr, axes=(0,) + tuple(i + 1 for i in axis_order)
                    )

        if axis_labels is not None and len(axis_labels) == transformed.ndim:
            add_kwargs["axis_labels"] = tuple(axis_labels)

        return transformed

    @staticmethod
    def _set_layer_z_scale(add_kwargs, data, z_spacing):
        """Set first-axis scale to z-spacing for 3D+ data."""
        if z_spacing is None:
            return
        if not hasattr(data, 'ndim') or data.ndim < 3:
            return

        try:
            z_value = float(z_spacing)
        except (TypeError, ValueError):
            return

        if z_value <= 0:
            return

        existing_scale = add_kwargs.get("scale")
        if existing_scale is None:
            scale = [1.0] * data.ndim
        else:
            scale = list(existing_scale)
            if len(scale) < data.ndim:
                scale.extend([1.0] * (data.ndim - len(scale)))
            elif len(scale) > data.ndim:
                scale = scale[: data.ndim]

        scale[0] = z_value
        add_kwargs["scale"] = tuple(scale)


def _try_get_z_spacing_from_ome_tiff(path):
    """Try to read z spacing from OME-TIFF metadata.

    Returns ``None`` if unavailable.
    """
    _, extension = _get_filename_extension(path)
    if extension != ".ome.tif":
        return None

    try:
        import tifffile

        with tifffile.TiffFile(path) as tif:
            ome_xml = tif.ome_metadata
            if ome_xml:
                ome_dict = tifffile.xml2dict(ome_xml)
                ome_root = ome_dict.get("OME", {})
                images = ome_root.get("Image", [])
                if isinstance(images, dict):
                    images = [images]

                if images:
                    pixels = images[0].get("Pixels", {})
                    z_value = pixels.get("@PhysicalSizeZ")
                    if z_value is None:
                        z_value = pixels.get("PhysicalSizeZ")

                    if isinstance(z_value, dict):
                        z_value = z_value.get("#text")
                    if z_value is not None:
                        return float(z_value)

            # Fallback to napari-phasors settings embedded in description.
            if not tif.pages:
                return None
            description = tif.pages[0].description
            if not description:
                return None

            description_dict = json.loads(description)
            settings_blob = description_dict.get("napari_phasors_settings")
            if isinstance(settings_blob, str):
                settings = json.loads(settings_blob)
            elif isinstance(settings_blob, dict):
                settings = settings_blob
            else:
                settings = {}

            z_value = settings.get("z_spacing_um")
            if z_value is None:
                return None
            return float(z_value)
    except Exception:  # noqa: BLE001
        return None


def _default_axis_labels(ndim):
    """Return default axis labels for a given dimensionality."""
    if ndim == 2:
        return ["Y", "X"]
    if ndim == 3:
        return ["Z", "Y", "X"]
    if ndim == 4:
        return ["T", "Z", "Y", "X"]

    labels = []
    for i in range(ndim):
        labels.append(f"Axis {i}")
    return labels


def _estimate_result_shape(file_paths):
    """Estimate output shape from first file and number of selected files."""
    if not file_paths:
        return None

    try:
        reader = napari_get_reader(
            file_paths[0], reader_options=None, harmonics=[1]
        )
        if reader is None:
            return None

        layers = reader(file_paths[0])
        if not layers:
            return None
        base_shape = tuple(np.shape(layers[0][0]))
        if len(file_paths) > 1:
            return (len(file_paths),) + base_shape
        return base_shape
    except Exception:  # noqa: BLE001
        return None


def _estimate_output_shape_from_options(
    path,
    reader_options,
    harmonics,
    n_files=1,
):
    """Estimate output shape with current reader options and harmonics."""
    try:
        reader = napari_get_reader(
            path,
            reader_options=reader_options,
            harmonics=harmonics if harmonics else [1],
        )
        if reader is None:
            return None

        layers = reader(path)
        if not layers:
            return None
        base_shape = tuple(np.shape(layers[0][0]))
        if n_files > 1:
            return (n_files,) + base_shape
        return base_shape
    except Exception:  # noqa: BLE001
        return None


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
        except Exception as e:  # noqa: BLE001
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
        except Exception as e:  # noqa: BLE001
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
        except Exception:  # noqa: BLE001
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
        except Exception as e:  # noqa: BLE001
            show_error(
                f"Error reading {'LSM' if self._is_lsm else 'TIFF'} signal: {str(e)}"
            )
            return None

    def _update_signal_plot(self):
        """Update the signal plot for LSM/TIFF files (spectral data)."""
        try:
            signal = self._get_preview_signal_data()
            if signal is None:
                return

            summed_signal = self._collapse_signal_for_plot(signal)
            if summed_signal is None or summed_signal.size == 0:
                return

            if summed_signal.shape[0] > 0:
                self.max_harmonic = summed_signal.shape[0] // 2

            self._update_harmonic_slider()

            self.ax.clear()

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

            self.figure.tight_layout(pad=1.0)
            self.figure.subplots_adjust(left=0.15)
            self.canvas.draw()
            self._update_shape_preview()

        except Exception as e:  # noqa: BLE001
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
        except Exception as e:  # noqa: BLE001
            show_error(f"Error reading SDT signal: {str(e)}")
            return None

    def _on_click(self, path, reader_options, harmonics):
        """Callback whenever the calculate phasor button is clicked."""
        if self.index.text():
            reader_options["index"] = int(self.index.text())
        super()._on_click(path, reader_options, harmonics)


class CziWidget(AdvancedOptionsWidget):
    """Widget for Zeiss CZI files."""

    def __init__(self, viewer, path):
        """Initialize the widget."""
        super().__init__(viewer, path)

    def initUI(self):
        """Initialize the user interface."""
        self.mainLayout = QVBoxLayout()
        self.setLayout(self.mainLayout)

        self.mainLayout.addWidget(self.canvas)

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
        """Get signal data for CZI files."""
        try:
            from ._reader import read_czi

            return read_czi(self.path)
        except Exception as e:  # noqa: BLE001
            show_error(f"Error reading CZI signal: {str(e)}")
            return None


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

            if "description" in attrs:
                description = json.loads(attrs["description"])
                if (
                    len(json.dumps(description)) > 512 * 512
                ):  # Threshold: 256 KB
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

        except Exception:  # noqa: BLE001
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
            signal = self._get_preview_signal_data()

            if signal is None:
                return

            signal_1d = self._collapse_signal_for_plot(signal)
            if signal_1d is None or signal_1d.size == 0:
                return
            self._update_harmonic_slider()

            self.ax.clear()
            self.ax.plot(signal_1d, color='white')

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

            self.figure.tight_layout(pad=1.0)
            self.figure.subplots_adjust(left=0.15)
            self.canvas.draw()
            self._update_shape_preview()

        except Exception as e:  # noqa: BLE001
            show_error(f"Error updating signal plot: {str(e)}")

    def _on_click(self, path, reader_options, harmonics):
        """Execute the phasor transformation."""
        try:
            super()._on_click(path, reader_options, harmonics)
        except Exception as e:  # noqa: BLE001
            show_error(f"Error during phasor transformation: {str(e)}")


class FlifWidget(AdvancedOptionsWidget):
    """Widget for FlimFast FLIF files."""

    def __init__(self, viewer, path):
        """Initialize the widget."""
        super().__init__(viewer, path)

    def initUI(self):
        """Initialize the user interface."""
        self.mainLayout = QVBoxLayout()
        self.setLayout(self.mainLayout)
        self.mainLayout.addWidget(self.canvas)
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
        """Get signal data for FLIF files."""
        try:
            from phasorpy.io import signal_from_flif

            return signal_from_flif(self.path)
        except Exception as e:  # noqa: BLE001
            show_error(f"Error reading FLIF signal: {str(e)}")
            return None


class BhWidget(AdvancedOptionsWidget):
    """Widget for SimFCS B&H files."""

    def __init__(self, viewer, path):
        """Initialize the widget."""
        super().__init__(viewer, path)

    def initUI(self):
        """Initialize the user interface."""
        self.mainLayout = QVBoxLayout()
        self.setLayout(self.mainLayout)
        self.mainLayout.addWidget(self.canvas)
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
        """Get signal data for B&H and BHZ files."""
        try:
            from phasorpy.io import signal_from_bh, signal_from_bhz

            _, ext = _get_filename_extension(self.path)
            if ext == ".bhz":
                return signal_from_bhz(self.path)
            return signal_from_bh(self.path)
        except Exception as e:  # noqa: BLE001
            show_error(f"Error reading B&H signal: {str(e)}")
            return None


class PqbinWidget(AdvancedOptionsWidget):
    """Widget for PicoQuant BIN files."""

    def __init__(self, viewer, path):
        """Initialize the widget."""
        super().__init__(viewer, path)

    def initUI(self):
        """Initialize the user interface."""
        self.mainLayout = QVBoxLayout()
        self.setLayout(self.mainLayout)
        self.mainLayout.addWidget(self.canvas)
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
        """Get signal data for PicoQuant BIN files."""
        try:
            from phasorpy.io import signal_from_pqbin

            return signal_from_pqbin(self.path)
        except Exception as e:  # noqa: BLE001
            show_error(f"Error reading PicoQuant BIN signal: {str(e)}")
            return None


class LifWidget(AdvancedOptionsWidget):
    """Widget for Leica LIF files."""

    def __init__(self, viewer, path):
        """Initialize the widget."""
        super().__init__(viewer, path)

    def initUI(self):
        """Initialize the user interface."""
        self.mainLayout = QVBoxLayout()
        self.setLayout(self.mainLayout)
        self.mainLayout.addWidget(self.canvas)
        self._harmonic_widget()

        image_layout = QHBoxLayout()
        image_layout.addWidget(QLabel("Image (regex/index): "))
        self.image = QLineEdit()
        self.image.setToolTip("Index or regex pattern of image to return.")
        self.image.textChanged.connect(self._on_lif_options_changed)
        image_layout.addWidget(self.image)
        image_layout.addStretch()
        self.mainLayout.addLayout(image_layout)

        dim_layout = QHBoxLayout()
        dim_layout.addWidget(QLabel("Dim: "))
        self.dim = QComboBox()
        self.dim.addItems(["λ", "Λ"])
        self.dim.setToolTip(
            "Character code of hyperspectral dimension. 'λ' for emission, 'Λ' for excitation."
        )
        self.dim.currentIndexChanged.connect(self._on_lif_options_changed)
        dim_layout.addWidget(self.dim)
        dim_layout.addStretch()
        self.mainLayout.addLayout(dim_layout)

        self._sync_lif_reader_options()

        self.btn = QPushButton("Phasor Transform")
        self.btn.clicked.connect(
            lambda: self._on_click(
                self.path, self.reader_options, self.harmonics
            )
        )
        self.mainLayout.addWidget(self.btn)
        self._update_signal_plot()

    def _sync_lif_reader_options(self):
        """Synchronize reader options with current LIF widget state."""
        text = self.image.text().strip()
        if text:
            import ast

            try:
                self.reader_options["image"] = ast.literal_eval(text)
            except Exception:  # noqa: BLE001
                self.reader_options["image"] = text
        else:
            self.reader_options.pop("image", None)

        self.reader_options["dim"] = self.dim.currentText()

    def _on_lif_options_changed(self):
        """Update options and refresh plot/shape preview on UI changes."""
        self._sync_lif_reader_options()
        self._update_signal_plot()

    def _get_signal_data(self):
        """Get signal data for LIF files if raw."""
        text = self.image.text()
        dim = self.dim.currentText()
        try:
            import ast

            from phasorpy.io import signal_from_lif

            options = self.reader_options.copy()
            if text:
                try:
                    options["image"] = ast.literal_eval(text)
                except Exception:  # noqa: BLE001
                    options["image"] = text
            options["dim"] = dim
            return signal_from_lif(self.path, **options)
        except Exception as e:  # noqa: BLE001
            show_error(
                f"Error reading LIF signal (image={text!r}, dim={dim!r}): {str(e)}"
            )
            return None

    def _on_click(self, path, reader_options, harmonics):
        """Callback whenever the calculate phasor button is clicked."""
        text = self.image.text()
        if text:
            import ast

            try:
                reader_options["image"] = ast.literal_eval(text)
            except Exception:  # noqa: BLE001
                reader_options["image"] = text
        reader_options["dim"] = self.dim.currentText()
        super()._on_click(path, reader_options, harmonics)


class JsonWidget(AdvancedOptionsWidget):
    """Widget for FLIM LABS JSON files."""

    def __init__(self, viewer, path):
        """Initialize the widget."""
        super().__init__(viewer, path)
        self.reader_options["channel"] = 0

    def initUI(self):
        """Initialize the user interface."""
        self.mainLayout = QVBoxLayout()
        self.setLayout(self.mainLayout)
        self.mainLayout.addWidget(self.canvas)
        self._harmonic_widget()

        chan_layout = QHBoxLayout()
        chan_layout.addWidget(QLabel("Channel (optional): "))
        self.channel_entry = QLineEdit("0")
        self.channel_entry.setValidator(QIntValidator(0, 2147483647, self))
        self.channel_entry.setToolTip(
            "Index of channel or empty for all channel reading."
        )
        self.channel_entry.textChanged.connect(self._on_json_channel_changed)
        chan_layout.addWidget(self.channel_entry)
        chan_layout.addStretch()
        self.mainLayout.addLayout(chan_layout)

        self._sync_json_reader_options()

        self.btn = QPushButton("Phasor Transform")
        self.btn.clicked.connect(
            lambda: self._on_click(
                self.path, self.reader_options, self.harmonics
            )
        )
        self.mainLayout.addWidget(self.btn)
        self._update_signal_plot()

    def _sync_json_reader_options(self):
        """Synchronize reader channel option with current JSON widget state."""
        txt = self.channel_entry.text().strip()
        try:
            self.reader_options["channel"] = int(txt) if txt else None
        except ValueError:
            # Keep previous behavior for invalid transient text states.
            self.reader_options["channel"] = None

    def _on_json_channel_changed(self):
        """Update options and refresh previews when JSON channel changes."""
        self._sync_json_reader_options()
        self._update_signal_plot()

    def _get_signal_data(self):
        """Get signal data for JSON files if raw."""
        self._sync_json_reader_options()
        txt = self.channel_entry.text().strip()
        try:
            from phasorpy.io import signal_from_flimlabs_json

            options = self.reader_options.copy()
            return signal_from_flimlabs_json(self.path, **options)
        except Exception as e:  # noqa: BLE001
            show_error(
                f"Error reading JSON signal (channel={txt!r}): {str(e)}"
            )
            return None

    def _on_click(self, path, reader_options, harmonics):
        """Callback whenever the calculate phasor button is clicked."""
        self._sync_json_reader_options()
        super()._on_click(path, reader_options, harmonics)


class ProcessedOnlyWidget(AdvancedOptionsWidget):
    """Generic Widget for Processed-only files (no signal)."""

    def __init__(self, viewer, path):
        """Initialize the widget."""
        super().__init__(viewer, path)

    def initUI(self):
        """Initialize the user interface."""
        self.mainLayout = QVBoxLayout()
        self.setLayout(self.mainLayout)
        self.canvas.hide()
        self._harmonic_widget()
        self.btn = QPushButton("Phasor Transform")
        self.btn.clicked.connect(
            lambda: self._on_click(
                self.path, self.reader_options, self.harmonics
            )
        )
        self.mainLayout.addWidget(self.btn)

    def _get_signal_data(self):
        """No raw signal for processed files."""
        return None

    def _update_signal_plot(self):
        """Override to just sync widgets and harmonics."""
        self._sync_stack_z_spacing_widget_visibility()
        self._update_harmonic_slider()


class SimfcsWidget(ProcessedOnlyWidget):
    """Widget for SimFCS referenced phasor files (.r64, .ref)."""


class IfliWidget(ProcessedOnlyWidget):
    """Widget for ISS IFLI files."""

    def initUI(self):
        """Initialize the user interface."""
        super().initUI()
        self.reader_options["channel"] = 0
        chan_layout = QHBoxLayout()
        chan_layout.addWidget(QLabel("Channel: "))
        self.channel_entry = QLineEdit("0")
        self.channel_entry.setValidator(QIntValidator(0, 2147483647, self))
        chan_layout.addWidget(self.channel_entry)
        chan_layout.addStretch()
        self.mainLayout.insertLayout(1, chan_layout)

    def _on_click(self, path, reader_options, harmonics):
        """Callback whenever the calculate phasor button is clicked."""
        txt = self.channel_entry.text().strip()
        try:
            reader_options["channel"] = int(txt) if txt else 0
        except ValueError:
            reader_options["channel"] = 0
            self.channel_entry.setText("0")
        super()._on_click(path, reader_options, harmonics)


class WriterWidget(QWidget):
    """Widget to export phasor data to a OME-TIF or CSV file."""

    def __init__(self, viewer: "napari.viewer.Viewer"):
        """Initialize the widget."""
        super().__init__()
        self.viewer = viewer
        self._floated = False

        self.main_layout = QVBoxLayout(self)

        # Add informational text at the top
        info_label = QLabel("<b>Export Options:</b>")
        self.main_layout.addWidget(info_label)

        ome_info = QLabel(
            "• <b>OME-TIFF:</b> Saves mean intensity and phasor coordinates"
            " with metadata including napari-phasors settings if exported layer"
            "contains phasor features. If not layer data is exported as"
            "single-channel image with metadata."
        )
        ome_info.setWordWrap(True)
        self.main_layout.addWidget(ome_info)

        csv_info = QLabel(
            "• <b>CSV:</b> Exports mean intensity and phasor coordinates"
            " as a table if available, otherwise exports raw layer data"
            " values, i.e. analysis values"
        )
        csv_info.setWordWrap(True)
        self.main_layout.addWidget(csv_info)

        image_info = QLabel(
            "• <b>Image (PNG/JPEG/TIFF):</b> Exports visual representation "
            "with applied colormap and contrast. Optional colorbar can be included"
        )
        image_info.setWordWrap(True)
        self.main_layout.addWidget(image_info)

        # Add some spacing
        self.main_layout.addSpacing(10)

        self.main_layout.addWidget(
            QLabel("Select Image Layer(s) to be Exported: ")
        )

        # Create horizontal layout for combobox and All/None controls
        export_layer_layout = QHBoxLayout()
        self.export_layer_combobox = CheckableComboBox(
            enable_primary_layer=False
        )
        export_layer_layout.addWidget(self.export_layer_combobox, 1)

        # "All | None" clickable labels for quick bulk selection
        select_all_label = QLabel('<a href="all" style="color: gray;">All</a>')
        select_all_label.setTextFormat(Qt.RichText)
        select_all_label.setCursor(Qt.PointingHandCursor)
        select_all_label.setToolTip("Select all layers")
        export_layer_layout.addWidget(select_all_label)

        separator_label = QLabel("|")
        separator_label.setStyleSheet("color: gray;")
        export_layer_layout.addWidget(separator_label)

        deselect_all_label = QLabel(
            '<a href="none" style="color: gray;">None</a>'
        )
        deselect_all_label.setTextFormat(Qt.RichText)
        deselect_all_label.setCursor(Qt.PointingHandCursor)
        deselect_all_label.setToolTip("Deselect all layers")
        export_layer_layout.addWidget(deselect_all_label)

        # Connect All/None labels (use lambdas to consume the href argument)
        select_all_label.linkActivated.connect(
            lambda _: self.export_layer_combobox.selectAll()
        )
        deselect_all_label.linkActivated.connect(
            lambda _: self.export_layer_combobox.deselectAll()
        )

        self.main_layout.addLayout(export_layer_layout)

        self.colorbar_checkbox = QCheckBox(
            "Include colorbar (for image exports)"
        )
        self.colorbar_checkbox.setChecked(True)
        self.main_layout.addWidget(self.colorbar_checkbox)

        self.search_button = QPushButton("Select Export Location and Name")
        self.search_button.clicked.connect(self._open_file_dialog)
        self.main_layout.addWidget(self.search_button)

        self.viewer.layers.events.inserted.connect(self._populate_combobox)
        self.viewer.layers.events.removed.connect(self._populate_combobox)

        self._populate_combobox()

    def showEvent(self, event):
        """Float the dock widget on first show and center it on screen."""
        super().showEvent(event)
        if not self._floated:
            self._floated = True
            parent = self.parent()
            while parent is not None:
                if isinstance(parent, QDockWidget):
                    parent.setFloating(True)
                    from qtpy.QtWidgets import QApplication

                    screen = QApplication.primaryScreen().geometry()
                    dw_size = parent.sizeHint()
                    parent.move(
                        screen.center().x() - dw_size.width() // 2,
                        screen.center().y() - dw_size.height() // 2,
                    )
                    break
                parent = parent.parent()

    def _open_file_dialog(self):
        """Open a native file dialog to select export location."""
        selected_layers = self.export_layer_combobox.checkedItems()
        if not selected_layers:
            show_error("No layer selected")
            return

        # Define filters for the native dialog
        filters = [
            "Phasor as OME-TIFF (*.ome.tif)",
            "Layer data as CSV (*.csv)",
            "Layer as PNG image (*.png)",
            "Layer as JPEG image (*.jpg)",
            "Layer as TIFF image (*.tif)",
        ]
        # Join filters with ';;' for the native dialog format
        filter_str = ";;".join(filters)

        # Pre-fill filename with first layer name so the dialog
        # always has a valid default and the user can just confirm.
        default_name = selected_layers[0]

        # Use static method to invoke the native OS dialog
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self, "Select Export Location", default_name, filter_str
        )

        if file_path:
            include_colorbar = self.colorbar_checkbox.isChecked()
            self._save_file(
                file_path, selected_filter, include_colorbar, selected_layers
            )

    def _populate_combobox(self):
        """Populate combobox with image layers."""
        self.export_layer_combobox.clear()
        image_layers = [
            layer for layer in self.viewer.layers if isinstance(layer, Image)
        ]
        for layer in image_layers:
            self.export_layer_combobox.addItem(layer.name)
        # Update display to show placeholder if no items are checked
        self.export_layer_combobox._update_display_text()

    def _save_file(
        self,
        file_path,
        selected_filter,
        include_colorbar=False,
        selected_layers=None,
    ):
        """Callback whenever the export location and name are specified."""
        if selected_layers is None:
            selected_layers = self.export_layer_combobox.checkedItems()

        if not selected_layers:
            show_error("No layers selected")
            return

        # Determine the extension based on selected filter
        if selected_filter == "Phasor as OME-TIFF (*.ome.tif)":
            ext = ".ome.tif"
        elif selected_filter == "Layer data as CSV (*.csv)":
            ext = ".csv"
        elif selected_filter == "Layer as PNG image (*.png)":
            ext = ".png"
        elif selected_filter == "Layer as JPEG image (*.jpg)":
            ext = ".jpg"
        elif selected_filter == "Layer as TIFF image (*.tif)":
            ext = ".tif"
        else:
            ext = ""

        # Extract directory and basename from file_path
        directory = os.path.dirname(file_path)
        full_basename = os.path.basename(file_path)

        # Remove any existing extension to get the base name
        if full_basename.endswith('.ome.tif'):
            base_name = full_basename[:-8]
        elif full_basename.endswith(('.tif', '.csv', '.png', '.jpg', '.jpeg')):
            base_name = os.path.splitext(full_basename)[0]
        else:
            base_name = full_basename

        # Determine if user provided a custom name.
        # The dialog is pre-filled with the first selected layer's name,
        # so if the base name still matches that default we treat it as
        # "no custom input".
        user_provided_name = bool(
            base_name and base_name != selected_layers[0]
        )

        # Export logic based on number of layers and whether name was provided
        if len(selected_layers) == 1:
            # Single layer export
            layer_name = selected_layers[0]
            export_layer = self.viewer.layers[layer_name]

            if user_provided_name:
                # Use the provided name
                final_path = os.path.join(directory, base_name + ext)
            else:
                # Use layer name
                final_path = os.path.join(directory, layer_name + ext)

            try:
                if selected_filter == "Phasor as OME-TIFF (*.ome.tif)":
                    write_ome_tiff(final_path, export_layer)
                elif selected_filter == "Layer data as CSV (*.csv)":
                    export_layer_as_csv(final_path, export_layer)
                elif selected_filter in [
                    "Layer as PNG image (*.png)",
                    "Layer as JPEG image (*.jpg)",
                    "Layer as TIFF image (*.tif)",
                ]:
                    export_layer_as_image(
                        final_path,
                        export_layer,
                        include_colorbar=include_colorbar,
                        current_step=self.viewer.dims.current_step,
                    )
                show_info(f"Exported {export_layer.name} to {final_path}")
            except Exception as e:  # noqa: BLE001
                show_error(f"Error exporting {layer_name}: {str(e)}")
        else:
            # Multiple layers export
            exported_count = 0
            for layer_name in selected_layers:
                try:
                    export_layer = self.viewer.layers[layer_name]

                    if user_provided_name:
                        # Use provided name + layer name
                        filename = f"{base_name}_{layer_name}{ext}"
                    else:
                        # Use just layer name
                        filename = f"{layer_name}{ext}"

                    layer_file_path = os.path.join(directory, filename)

                    if selected_filter == "Phasor as OME-TIFF (*.ome.tif)":
                        write_ome_tiff(layer_file_path, export_layer)
                    elif selected_filter == "Layer data as CSV (*.csv)":
                        export_layer_as_csv(layer_file_path, export_layer)
                    elif selected_filter in [
                        "Layer as PNG image (*.png)",
                        "Layer as JPEG image (*.jpg)",
                        "Layer as TIFF image (*.tif)",
                    ]:
                        export_layer_as_image(
                            layer_file_path,
                            export_layer,
                            include_colorbar=include_colorbar,
                            current_step=self.viewer.dims.current_step,
                        )

                    exported_count += 1
                except Exception as e:  # noqa: BLE001
                    show_error(f"Error exporting {layer_name}: {str(e)}")

            if exported_count > 0:
                show_info(
                    f"Successfully exported {exported_count} layer(s) to {directory}"
                )
