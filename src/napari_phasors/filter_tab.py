from math import ceil, log10

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from napari.utils.notifications import show_error
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from skimage.filters import threshold_li, threshold_otsu, threshold_yen
from superqt import QRangeSlider

from ._utils import apply_filter_and_threshold, validate_harmonics_for_wavelet


class FilterWidget(QWidget):
    """Widget for interactive filtering and thresholding of phasor features.

    Provides controls for:
      - Median filtering (kernel size and repetitions)
      - Wavelet filtering (sigma and levels)
      - Automatic and manual intensity thresholding
      - Visualization of the mean intensity histogram with a dynamic line
      - Applying filter and threshold operations to the selected image layer

    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer instance.
    parent : QWidget, optional
        The parent widget.

    Notes
    -----
    This widget is intended to be used as a tab within the main PlotterWidget.
    It updates the histogram and threshold interactively and applies changes
    directly to the selected layer.

    """

    def __init__(self, viewer, parent=None):
        """Initialize the FilterWidget."""
        super().__init__()
        self.parent_widget = parent
        self.viewer = viewer

        # Initialize attributes
        self._phasors_selected_layer = None
        self.threshold_factor = 1
        self.hist_fig, self.hist_ax = plt.subplots(
            figsize=(8, 4), constrained_layout=True
        )
        self.threshold_line_lower = None
        self.threshold_line_upper = None
        self.threshold_area_lower = None
        self.threshold_area_upper = None
        self._updating_threshold = False
        self._dragging_line = None
        self._canvas = None

        # Style the histogram axes and figure initially
        self.style_histogram_axes()

        # Create UI elements
        self.setup_ui()

        # Connect callbacks
        self.parent_widget.image_layer_with_phasor_features_combobox.currentIndexChanged.connect(
            self._on_image_layer_changed
        )
        self.threshold_slider.valueChanged.connect(
            self.on_threshold_slider_change
        )
        self.threshold_method_combobox.currentTextChanged.connect(
            self.on_threshold_method_changed
        )
        self.filter_method_combobox.currentTextChanged.connect(
            self.on_filter_method_changed
        )
        self.median_filter_spinbox.valueChanged.connect(
            self.on_median_kernel_size_change
        )
        self.apply_button.clicked.connect(self.apply_button_clicked)
        self.min_threshold_edit.editingFinished.connect(
            self.on_min_threshold_edit_changed
        )
        self.max_threshold_edit.editingFinished.connect(
            self.on_max_threshold_edit_changed
        )

    def setup_ui(self):
        """Setup the user interface elements."""
        layout = QVBoxLayout()

        # Create a widget to hold the scrollable content
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        # Filter method selection
        filter_method_layout = QHBoxLayout()
        filter_method_layout.addWidget(QLabel("Filter Method:"))
        self.filter_method_combobox = QComboBox()
        self.filter_method_combobox.addItems(["Median", "Wavelet"])
        filter_method_layout.addWidget(self.filter_method_combobox)
        scroll_layout.addLayout(filter_method_layout)

        # Create containers for filter-specific parameters
        self.median_filter_widget = QWidget()
        self.setup_median_filter_ui()
        scroll_layout.addWidget(self.median_filter_widget)

        self.wavelet_filter_widget = QWidget()
        self.setup_wavelet_filter_ui()
        scroll_layout.addWidget(self.wavelet_filter_widget)

        # Threshold method selection
        threshold_method_layout = QHBoxLayout()
        threshold_method_layout.addWidget(QLabel("Threshold Method:"))
        self.threshold_method_combobox = QComboBox()
        self.threshold_method_combobox.addItems(
            ["None", "Manual", "Otsu", "Li", "Yen"]
        )
        # Set Otsu as default
        self.threshold_method_combobox.setCurrentText("Otsu")
        threshold_method_layout.addWidget(self.threshold_method_combobox)

        # Add min and max intensity editable fields to the same row
        threshold_method_layout.addSpacing(20)
        threshold_method_layout.addWidget(QLabel("Min Intensity:"))
        self.min_threshold_edit = QLineEdit()
        self.min_threshold_edit.setFixedWidth(80)
        self.min_threshold_edit.setText("0.00")
        self.min_threshold_edit.setToolTip("Lower threshold value")
        threshold_method_layout.addWidget(self.min_threshold_edit)

        threshold_method_layout.addSpacing(10)
        threshold_method_layout.addWidget(QLabel("Max Intensity:"))
        self.max_threshold_edit = QLineEdit()
        self.max_threshold_edit.setFixedWidth(80)
        self.max_threshold_edit.setText("0.00")
        self.max_threshold_edit.setToolTip("Upper threshold value")
        threshold_method_layout.addWidget(self.max_threshold_edit)
        threshold_method_layout.addStretch()

        scroll_layout.addLayout(threshold_method_layout)

        # Threshold range slider with log scale checkbox
        theshold_slider_layout = QHBoxLayout()
        self.threshold_slider = QRangeSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(100)
        self.threshold_slider.setValue((0, 100))
        theshold_slider_layout.addWidget(self.threshold_slider)

        # Add log scale checkbox to the same line as the slider
        self.log_scale_checkbox = QCheckBox("Log Scale Histogram")
        self.log_scale_checkbox.setChecked(False)
        self.log_scale_checkbox.stateChanged.connect(self.on_log_scale_changed)
        theshold_slider_layout.addWidget(self.log_scale_checkbox)

        scroll_layout.addLayout(theshold_slider_layout)

        # Embed the Matplotlib figure into the widget with fixed size
        canvas = FigureCanvas(self.hist_fig)
        canvas.setFixedHeight(150)
        canvas.setSizePolicy(
            canvas.sizePolicy().Expanding, canvas.sizePolicy().Fixed
        )
        self._canvas = canvas

        # Connect mouse events for draggable threshold line
        canvas.mpl_connect('button_press_event', self.on_mouse_press)
        canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        canvas.mpl_connect('button_release_event', self.on_mouse_release)

        scroll_layout.addWidget(canvas)

        # Set scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(scroll_content)
        layout.addWidget(scroll_area)

        # Apply button (not inside scroll area)
        self.apply_button = QPushButton("Apply Filter and Threshold")
        layout.addWidget(self.apply_button)

        self.setLayout(layout)

        # Initialize UI state
        self.on_filter_method_changed()

    def setup_median_filter_ui(self):
        """Setup UI elements for median filter."""
        layout = QVBoxLayout(self.median_filter_widget)

        # Median filter kernel size
        median_filter_layout = QHBoxLayout()
        self.median_filter_label = QLabel("Median Filter Kernel Size: 3 x 3")
        self.median_filter_spinbox = QSpinBox()
        self.median_filter_spinbox.setMinimum(2)
        self.median_filter_spinbox.setMaximum(99)
        self.median_filter_spinbox.setValue(3)
        median_filter_layout.addWidget(self.median_filter_label)
        median_filter_layout.addWidget(self.median_filter_spinbox)
        layout.addLayout(median_filter_layout)

        # Median filter repetitions
        repetitions_layout = QHBoxLayout()
        repetitions_layout.addWidget(QLabel("Filter Repetitions:"))
        self.median_filter_repetition_spinbox = QSpinBox()
        self.median_filter_repetition_spinbox.setMinimum(0)
        self.median_filter_repetition_spinbox.setValue(0)
        repetitions_layout.addWidget(self.median_filter_repetition_spinbox)
        layout.addLayout(repetitions_layout)

    def setup_wavelet_filter_ui(self):
        """Setup UI elements for wavelet filter."""
        layout = QVBoxLayout(self.wavelet_filter_widget)

        # Warning label for incompatible harmonics
        self.harmonic_warning_label = QLabel()
        self.harmonic_warning_label.setStyleSheet(
            "color: orange; font-style: italic;"
        )
        self.harmonic_warning_label.setVisible(False)
        layout.addWidget(self.harmonic_warning_label)

        # Container for wavelet parameters (can be hidden when invalid)
        self.wavelet_params_widget = QWidget()
        params_layout = QVBoxLayout(self.wavelet_params_widget)

        # Wavelet sigma parameter
        sigma_layout = QHBoxLayout()
        sigma_layout.addWidget(QLabel("Sigma:"))
        self.wavelet_sigma_spinbox = QDoubleSpinBox()
        self.wavelet_sigma_spinbox.setMinimum(0.1)
        self.wavelet_sigma_spinbox.setMaximum(10.0)
        self.wavelet_sigma_spinbox.setValue(2.0)
        self.wavelet_sigma_spinbox.setSingleStep(0.1)
        self.wavelet_sigma_spinbox.setDecimals(1)
        sigma_layout.addWidget(self.wavelet_sigma_spinbox)
        params_layout.addLayout(sigma_layout)

        # Wavelet levels parameter
        levels_layout = QHBoxLayout()
        levels_layout.addWidget(QLabel("Levels:"))
        self.wavelet_levels_spinbox = QSpinBox()
        self.wavelet_levels_spinbox.setMinimum(1)
        self.wavelet_levels_spinbox.setMaximum(10)
        self.wavelet_levels_spinbox.setValue(1)
        levels_layout.addWidget(self.wavelet_levels_spinbox)
        params_layout.addLayout(levels_layout)

        layout.addWidget(self.wavelet_params_widget)

    def on_filter_method_changed(self):
        """Callback when filter method is changed."""
        method = self.filter_method_combobox.currentText()

        if method == "Median":
            self.median_filter_widget.setVisible(True)
            self.wavelet_filter_widget.setVisible(False)
        elif method == "Wavelet":
            self.median_filter_widget.setVisible(False)
            self.wavelet_filter_widget.setVisible(True)

        if (
            self.parent_widget.image_layer_with_phasor_features_combobox.currentText()
        ):
            self.check_harmonics_compatibility()

    def check_harmonics_compatibility(self):
        """Check if current layer's harmonics are compatible with wavelet filtering."""
        labels_layer_name = (
            self.parent_widget.image_layer_with_phasor_features_combobox.currentText()
        )

        if not labels_layer_name:
            return

        layer_metadata = self.viewer.layers[labels_layer_name].metadata
        harmonics = layer_metadata.get('harmonics')

        if harmonics is None:
            return

        is_valid = validate_harmonics_for_wavelet(harmonics)

        if not is_valid:
            harmonics_str = ", ".join(map(str, sorted(harmonics)))
            self.harmonic_warning_label.setText(
                f"Warning: Harmonics [{harmonics_str}] are not compatible "
                f"for Wavelet filtering.\n Each harmonic must have a "
                f"corresponding double or half value. No filtering will be "
                f"applied."
            )
            self.harmonic_warning_label.setVisible(True)
            self.wavelet_params_widget.setVisible(False)

        else:
            self.harmonic_warning_label.setVisible(False)
            self.wavelet_params_widget.setVisible(True)

    def calculate_automatic_threshold(self, method, data):
        """Calculate automatic threshold using the specified method.

        Parameters
        ----------
        method : str
            Threshold method ('Otsu', 'Li', 'Yen')
        data : np.ndarray
            Image data to calculate threshold from

        Returns
        -------
        float
            Lower threshold value
        """
        clean_data = data[~np.isnan(data)]

        if len(clean_data) == 0:
            return 0

        try:
            if method == "Otsu":
                return threshold_otsu(clean_data)
            elif method == "Li":
                return threshold_li(clean_data)
            elif method == "Yen":
                return threshold_yen(clean_data)
            else:
                return 0
        except Exception:
            # Fallback to 10% of max if automatic method fails
            return np.nanmax(data) * 0.1

    def on_threshold_method_changed(self):
        """Callback when threshold method is changed."""
        selected_layers = self.parent_widget.get_selected_layers()

        if self._updating_threshold or not selected_layers:
            return

        method = self.threshold_method_combobox.currentText()
        max_value = self.threshold_slider.maximum()

        if method == "None":
            self._updating_threshold = True
            self.threshold_slider.setValue((0, max_value))
            self.min_threshold_edit.setText("0.00")
            self.max_threshold_edit.setText(
                f'{max_value / self.threshold_factor:.2f}'
            )
            self.update_threshold_lines()
            self._updating_threshold = False

        elif method != "Manual":
            # Collect mean data from all selected layers for automatic threshold
            all_mean_data = []
            for layer in selected_layers:
                mean_data = layer.metadata.get('original_mean')
                if mean_data is not None:
                    all_mean_data.append(mean_data.copy().flatten())

            if not all_mean_data:
                return

            merged_mean_data = np.concatenate(all_mean_data)

            lower_threshold = self.calculate_automatic_threshold(
                method, merged_mean_data
            )

            _, current_upper_val = self.threshold_slider.value()

            self._updating_threshold = True
            lower_val = int(lower_threshold * self.threshold_factor)
            self.threshold_slider.setValue((lower_val, current_upper_val))
            self.min_threshold_edit.setText(
                f'{lower_val / self.threshold_factor:.2f}'
            )
            self.max_threshold_edit.setText(
                f'{current_upper_val / self.threshold_factor:.2f}'
            )
            self.update_threshold_lines()
            self._updating_threshold = False

    def _on_image_layer_changed(self):
        """Callback function when the image layer selection is changed."""
        selected_layers = self.parent_widget.get_selected_layers()
        if not selected_layers:
            return

        # Use primary layer metadata for settings restoration
        primary_layer = selected_layers[0]
        layer_metadata = primary_layer.metadata

        # Calculate max mean value across all selected layers
        max_mean_values = []
        for layer in selected_layers:
            mean_data = layer.metadata.get("original_mean")
            if mean_data is None:
                continue
            if 'mask' in layer.metadata.keys():
                max_val = np.nanmax(mean_data[layer.metadata['mask'] > 0])
            else:
                max_val = np.nanmax(mean_data)
            max_mean_values.append(max_val)

        if not max_mean_values:
            return

        max_mean_value = max(max_mean_values)
        if max_mean_value > 0:
            magnitude = int(log10(max_mean_value))
            self.threshold_factor = (
                10 ** (2 - magnitude) if magnitude <= 2 else 1
            )
        else:
            self.threshold_factor = 1

        self.threshold_slider.setMaximum(
            ceil(max_mean_value * self.threshold_factor)
        )

        self._updating_threshold = True

        if "settings" in layer_metadata.keys():
            settings = layer_metadata["settings"]

            if "threshold_method" in settings.keys():
                self.threshold_method_combobox.setCurrentText(
                    settings["threshold_method"]
                )
            else:
                self.threshold_method_combobox.setCurrentText("Otsu")

            if "threshold" in settings.keys():
                lower_val = int(settings["threshold"] * self.threshold_factor)
                upper_val = int(
                    settings.get("threshold_upper", max_mean_value)
                    * self.threshold_factor
                )
                self.threshold_slider.setValue((lower_val, upper_val))
                self.min_threshold_edit.setText(
                    f'{lower_val / self.threshold_factor:.2f}'
                )
                self.max_threshold_edit.setText(
                    f'{upper_val / self.threshold_factor:.2f}'
                )
            else:
                mean_data = layer_metadata["original_mean"].copy()
                lower_threshold = self.calculate_automatic_threshold(
                    "Otsu", mean_data
                )
                lower_val = int(lower_threshold * self.threshold_factor)
                upper_val = self.threshold_slider.maximum()
                self.threshold_slider.setValue((lower_val, upper_val))
                self.min_threshold_edit.setText(
                    f'{lower_val / self.threshold_factor:.2f}'
                )
                self.max_threshold_edit.setText(
                    f'{upper_val / self.threshold_factor:.2f}'
                )

            if "filter" in settings.keys():
                filter_settings = settings["filter"]
                if "method" in filter_settings:
                    method = filter_settings["method"]
                    if method == "median":
                        self.filter_method_combobox.setCurrentText("Median")
                    elif method == "wavelet":
                        harmonics = layer_metadata.get('harmonics')
                        if (
                            harmonics is not None
                            and validate_harmonics_for_wavelet(harmonics)
                        ):
                            self.filter_method_combobox.setCurrentText(
                                "Wavelet"
                            )
                        else:
                            self.filter_method_combobox.setCurrentText(
                                "Median"
                            )

                if "size" in filter_settings:
                    self.median_filter_spinbox.setValue(
                        int(filter_settings["size"])
                    )
                if "repeat" in filter_settings:
                    self.median_filter_repetition_spinbox.setValue(
                        int(filter_settings["repeat"])
                    )
                if "sigma" in filter_settings:
                    self.wavelet_sigma_spinbox.setValue(
                        float(filter_settings["sigma"])
                    )
                if "levels" in filter_settings:
                    self.wavelet_levels_spinbox.setValue(
                        int(filter_settings["levels"])
                    )
        else:
            self.threshold_method_combobox.setCurrentText("Otsu")
            # Use merged mean data for initial threshold calculation
            all_mean_data = []
            for layer in selected_layers:
                mean_data = layer.metadata.get('original_mean')
                if mean_data is not None:
                    all_mean_data.append(mean_data.copy().flatten())

            if all_mean_data:
                merged_mean_data = np.concatenate(all_mean_data)
                lower_threshold = self.calculate_automatic_threshold(
                    "Otsu", merged_mean_data
                )
                lower_val = int(lower_threshold * self.threshold_factor)
                upper_val = self.threshold_slider.maximum()
                self.threshold_slider.setValue((lower_val, upper_val))
                self.min_threshold_edit.setText(
                    f'{lower_val / self.threshold_factor:.2f}'
                )
                self.max_threshold_edit.setText(
                    f'{upper_val / self.threshold_factor:.2f}'
                )

        self._updating_threshold = False
        self.plot_mean_histogram()

        self.check_harmonics_compatibility()

        current_method = self.threshold_method_combobox.currentText()
        if current_method not in ["Manual", "None"]:
            self.on_threshold_method_changed()

    def on_threshold_slider_change(self):
        """Callback function when the threshold slider value changes."""
        if not self._updating_threshold:
            current_method = self.threshold_method_combobox.currentText()
            lower_val, upper_val = self.threshold_slider.value()

            if not (
                current_method == "None"
                and lower_val == 0
                and upper_val == self.threshold_slider.maximum()
            ):
                self.threshold_method_combobox.setCurrentText("Manual")

        lower_val, upper_val = self.threshold_slider.value()
        self.min_threshold_edit.setText(
            f'{lower_val / self.threshold_factor:.2f}'
        )
        self.max_threshold_edit.setText(
            f'{upper_val / self.threshold_factor:.2f}'
        )

        self.update_threshold_lines()

    def on_median_kernel_size_change(self):
        kernel_value = self.median_filter_spinbox.value()
        self.median_filter_label.setText(
            'Median Filter Kernel Size: ' + f'{kernel_value} x {kernel_value}'
        )

    def on_min_threshold_edit_changed(self):
        """Callback when the minimum threshold text edit is changed."""
        try:
            new_value = float(self.min_threshold_edit.text())
            new_slider_value = int(new_value * self.threshold_factor)

            _, upper_val = self.threshold_slider.value()

            slider_max = self.threshold_slider.maximum()
            new_slider_value = max(
                0, min(new_slider_value, upper_val, slider_max)
            )

            self._updating_threshold = True
            self.threshold_slider.setValue((new_slider_value, upper_val))
            self.min_threshold_edit.setText(
                f'{new_slider_value / self.threshold_factor:.2f}'
            )
            self._updating_threshold = False

            if self.threshold_method_combobox.currentText() not in [
                "Manual",
                "None",
            ]:
                self.threshold_method_combobox.setCurrentText("Manual")

            self.update_threshold_lines()
        except ValueError:
            lower_val, _ = self.threshold_slider.value()
            self.min_threshold_edit.setText(
                f'{lower_val / self.threshold_factor:.2f}'
            )

    def on_max_threshold_edit_changed(self):
        """Callback when the maximum threshold text edit is changed."""
        try:
            new_value = float(self.max_threshold_edit.text())
            new_slider_value = int(new_value * self.threshold_factor)

            lower_val, _ = self.threshold_slider.value()

            slider_max = self.threshold_slider.maximum()
            new_slider_value = max(
                lower_val, min(new_slider_value, slider_max)
            )

            self._updating_threshold = True
            self.threshold_slider.setValue((lower_val, new_slider_value))

            self.max_threshold_edit.setText(
                f'{new_slider_value / self.threshold_factor:.2f}'
            )
            self._updating_threshold = False

            if self.threshold_method_combobox.currentText() not in [
                "Manual",
                "None",
            ]:
                self.threshold_method_combobox.setCurrentText("Manual")

            self.update_threshold_lines()
        except ValueError:
            _, upper_val = self.threshold_slider.value()
            self.max_threshold_edit.setText(
                f'{upper_val / self.threshold_factor:.2f}'
            )

    def style_histogram_axes(self):
        """Apply consistent styling to the histogram axes and figure."""
        self.hist_ax.patch.set_alpha(0)
        self.hist_fig.patch.set_alpha(0)
        for spine in self.hist_ax.spines.values():
            spine.set_color('grey')
            spine.set_linewidth(1)
        self.hist_ax.set_ylabel("Count", fontsize=6, color='grey')
        self.hist_ax.set_xlabel("Mean Intensity", fontsize=6, color='grey')
        self.hist_ax.tick_params(
            axis='x', which='major', labelsize=7, colors='grey'
        )
        self.hist_ax.tick_params(
            axis='x', which='minor', labelsize=7, colors='grey'
        )
        self.hist_ax.tick_params(
            axis='y', which='major', labelsize=7, colors='grey'
        )
        self.hist_ax.tick_params(
            axis='y', which='minor', labelsize=7, colors='grey'
        )

    def plot_mean_histogram(self):
        """Plot the histogram of the mean intensity data from all selected layers."""
        selected_layers = self.parent_widget.get_selected_layers()
        if not selected_layers:
            return

        # Collect mean data from all selected layers
        all_mean_data = []
        for layer in selected_layers:
            mean_data = layer.metadata.get('original_mean')
            if mean_data is None:
                continue
            mean_data = mean_data.copy()

            # Apply mask if present
            if 'mask' in layer.metadata:
                mean_data = mean_data[layer.metadata['mask'] > 0]

            all_mean_data.append(mean_data.flatten())

        if not all_mean_data:
            return

        # Merge all mean data
        merged_mean_data = np.concatenate(all_mean_data)

        self.hist_ax.clear()
        self.threshold_line_lower = None
        self.threshold_line_upper = None
        self.threshold_area_lower = None
        self.threshold_area_upper = None
        self.hist_ax.hist(
            merged_mean_data, bins=100, color='white', edgecolor='white'
        )
        self.style_histogram_axes()

        self.update_threshold_lines()
        self.hist_fig.canvas.draw_idle()

    def update_threshold_lines(self):
        """Update the vertical threshold lines and shaded areas on the histogram."""
        selected_layers = self.parent_widget.get_selected_layers()
        if not selected_layers:
            return

        lower_val, upper_val = self.threshold_slider.value()
        threshold_lower = lower_val / self.threshold_factor
        threshold_upper = upper_val / self.threshold_factor

        xlim = self.hist_ax.get_xlim()
        ylim = self.hist_ax.get_ylim()

        if self.threshold_line_lower is None:
            self.threshold_line_lower = self.hist_ax.axvline(
                x=threshold_lower,
                color='red',
                linestyle='-',
                linewidth=2.5,
                label='Lower Threshold',
                picker=5,
                zorder=10,
            )
        else:
            self.threshold_line_lower.set_xdata(
                [threshold_lower, threshold_lower]
            )

        if self.threshold_line_upper is None:
            self.threshold_line_upper = self.hist_ax.axvline(
                x=threshold_upper,
                color='red',
                linestyle='-',
                linewidth=2.5,
                label='Upper Threshold',
                picker=5,
                zorder=10,
            )
        else:
            self.threshold_line_upper.set_xdata(
                [threshold_upper, threshold_upper]
            )

        if self.threshold_area_lower is None:
            self.threshold_area_lower = self.hist_ax.axvspan(
                xlim[0], threshold_lower, alpha=0.4, color='black', zorder=5
            )
        else:
            xy = np.array(self.threshold_area_lower.get_xy())
            if xy.ndim == 2 and xy.shape[0] == 5:
                xy[:, 0] = [
                    xlim[0],
                    threshold_lower,
                    threshold_lower,
                    xlim[0],
                    xlim[0],
                ]
                self.threshold_area_lower.set_xy(xy)
            else:
                # If shape is not as expected, recreate the patch
                self.threshold_area_lower.remove()
                self.threshold_area_lower = self.hist_ax.axvspan(
                    xlim[0],
                    threshold_lower,
                    alpha=0.4,
                    color='black',
                    zorder=5,
                )

        if self.threshold_area_upper is None:
            self.threshold_area_upper = self.hist_ax.axvspan(
                threshold_upper, xlim[1], alpha=0.4, color='black', zorder=5
            )
        else:
            xy = np.array(self.threshold_area_upper.get_xy())
            if xy.ndim == 2 and xy.shape[0] == 5:
                xy[:, 0] = [
                    threshold_upper,
                    xlim[1],
                    xlim[1],
                    threshold_upper,
                    threshold_upper,
                ]
                self.threshold_area_upper.set_xy(xy)
            else:
                # If shape is not as expected, recreate the patch
                self.threshold_area_upper.remove()
                self.threshold_area_upper = self.hist_ax.axvspan(
                    threshold_upper,
                    xlim[1],
                    alpha=0.4,
                    color='black',
                    zorder=5,
                )

        self.hist_ax.set_xlim(xlim)
        self.hist_ax.set_ylim(ylim)

        self.hist_fig.canvas.draw_idle()

    def on_log_scale_changed(self, state):
        """Callback when log scale checkbox is toggled."""
        selected_layers = self.parent_widget.get_selected_layers()
        if not selected_layers:
            return

        if state == 2:
            self.hist_ax.set_yscale('log')
        else:
            self.hist_ax.set_yscale('linear')

        self.hist_fig.canvas.draw_idle()

    def apply_button_clicked(self):
        """Apply the filter and threshold to all selected layers."""
        selected_layers = self.parent_widget.get_selected_layers()
        if not selected_layers:
            show_error(
                "Please select at least one image layer with phasor features."
            )
            return

        threshold_method = self.threshold_method_combobox.currentText()
        threshold_lower = None
        threshold_upper = None
        if threshold_method != "None":
            lower_val, upper_val = self.threshold_slider.value()
            threshold_lower = lower_val / self.threshold_factor
            threshold_upper = upper_val / self.threshold_factor

        current_filter_method = (
            self.filter_method_combobox.currentText().lower()
        )

        # Apply filter and threshold to each selected layer
        for layer in selected_layers:
            filter_method = None
            size = None
            repeat = None
            sigma = None
            levels = None
            harmonics = None

            if (
                current_filter_method == "median"
                and self.median_filter_repetition_spinbox.value() > 0
            ):
                filter_method = "median"
                size = self.median_filter_spinbox.value()
                repeat = self.median_filter_repetition_spinbox.value()
            elif current_filter_method == "wavelet":
                harmonics = layer.metadata.get('harmonics')
                if harmonics is not None and validate_harmonics_for_wavelet(
                    harmonics
                ):
                    filter_method = "wavelet"
                    sigma = self.wavelet_sigma_spinbox.value()
                    levels = self.wavelet_levels_spinbox.value()

            apply_filter_and_threshold(
                layer,
                threshold=threshold_lower,
                threshold_upper=threshold_upper,
                threshold_method=threshold_method,
                filter_method=filter_method,
                size=size,
                repeat=repeat,
                sigma=sigma,
                levels=levels,
                harmonics=harmonics,
            )

        if self.parent_widget is not None:
            self.parent_widget.refresh_phasor_data()

    def on_mouse_press(self, event):
        """Handle mouse press event for threshold line dragging."""
        if event.inaxes != self.hist_ax:
            return

        if (
            self.threshold_line_lower is None
            or self.threshold_line_upper is None
        ):
            return

        lower_val, upper_val = self.threshold_slider.value()
        threshold_lower = lower_val / self.threshold_factor
        threshold_upper = upper_val / self.threshold_factor

        xlim = self.hist_ax.get_xlim()
        x_range = xlim[1] - xlim[0]

        tolerance = x_range * 0.02

        dist_to_lower = abs(event.xdata - threshold_lower)
        dist_to_upper = abs(event.xdata - threshold_upper)

        if dist_to_lower < tolerance and dist_to_lower <= dist_to_upper:
            self._dragging_line = 'lower'
            if self._canvas:
                self._canvas.setCursor(Qt.SizeHorCursor)
        elif dist_to_upper < tolerance:
            self._dragging_line = 'upper'
            if self._canvas:
                self._canvas.setCursor(Qt.SizeHorCursor)

    def on_mouse_move(self, event):
        """Handle mouse move event for threshold line dragging."""
        if self._dragging_line is None or event.inaxes != self.hist_ax:
            return

        if event.xdata is None:
            return

        xlim = self.hist_ax.get_xlim()
        lower_val, upper_val = self.threshold_slider.value()

        new_threshold = max(xlim[0], min(event.xdata, xlim[1]))
        new_slider_value = int(new_threshold * self.threshold_factor)

        if self._dragging_line == 'lower':
            new_slider_value = min(new_slider_value, upper_val)
            if new_slider_value != lower_val:
                self._updating_threshold = True
                self.threshold_slider.setValue((new_slider_value, upper_val))
                self.min_threshold_edit.setText(
                    f'{new_slider_value / self.threshold_factor:.2f}'
                )
                self.max_threshold_edit.setText(
                    f'{upper_val / self.threshold_factor:.2f}'
                )
                self._updating_threshold = False
                self.update_threshold_lines()
        elif self._dragging_line == 'upper':
            new_slider_value = max(new_slider_value, lower_val)
            if new_slider_value != upper_val:
                self._updating_threshold = True
                self.threshold_slider.setValue((lower_val, new_slider_value))
                self.min_threshold_edit.setText(
                    f'{lower_val / self.threshold_factor:.2f}'
                )
                self.max_threshold_edit.setText(
                    f'{new_slider_value / self.threshold_factor:.2f}'
                )
                self._updating_threshold = False
                self.update_threshold_lines()

    def on_mouse_release(self, event):
        """Handle mouse release event for threshold line dragging."""
        if self._dragging_line is not None:
            self._dragging_line = None

            if self._canvas:
                self._canvas.setCursor(Qt.ArrowCursor)

            if not self._updating_threshold:
                self.threshold_method_combobox.setCurrentText("Manual")
