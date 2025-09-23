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
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from skimage.filters import threshold_li, threshold_otsu, threshold_yen

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
        self.parent_widget._labels_layer_with_phasor_features = None
        self._phasors_selected_layer = None
        self.threshold_factor = 1
        self.hist_fig, self.hist_ax = plt.subplots(
            figsize=(8, 4), constrained_layout=True
        )
        self.threshold_line = None
        self._updating_threshold = False  # Flag to prevent recursive updates

        # Style the histogram axes and figure initially
        self.style_histogram_axes()

        # Create UI elements
        self.setup_ui()

        # Connect callbacks
        self.parent_widget.image_layer_with_phasor_features_combobox.currentIndexChanged.connect(
            self.on_labels_layer_with_phasor_features_changed
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
        scroll_layout.addLayout(threshold_method_layout)

        # Add log scale checkbox to the same line
        self.log_scale_checkbox = QCheckBox("Log scale")
        self.log_scale_checkbox.setChecked(False)
        self.log_scale_checkbox.stateChanged.connect(self.on_log_scale_changed)
        threshold_method_layout.addWidget(self.log_scale_checkbox)
        threshold_method_layout.addStretch()

        # Threshold slider and label
        theshold_slider_layout = QHBoxLayout()
        self.label_3 = QLabel("Intensity threshold: 0")
        theshold_slider_layout.addWidget(self.label_3)
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(100)
        self.threshold_slider.setValue(0)
        theshold_slider_layout.addWidget(self.threshold_slider)
        scroll_layout.addLayout(theshold_slider_layout)

        # Embed the Matplotlib figure into the widget with fixed size
        canvas = FigureCanvas(self.hist_fig)
        canvas.setFixedHeight(150)
        canvas.setSizePolicy(
            canvas.sizePolicy().Expanding, canvas.sizePolicy().Fixed
        )
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

        if self.parent_widget._labels_layer_with_phasor_features is not None:
            self.check_harmonics_compatibility()

    def check_harmonics_compatibility(self):
        """Check if current layer's harmonics are compatible with wavelet filtering."""
        if self.parent_widget._labels_layer_with_phasor_features is None:
            return

        labels_layer_name = (
            self.parent_widget.image_layer_with_phasor_features_combobox.currentText()
        )

        if not labels_layer_name:
            return

        layer_metadata = self.viewer.layers[labels_layer_name].metadata
        phasor_features = layer_metadata[
            'phasor_features_labels_layer'
        ].features
        harmonics = np.unique(phasor_features['harmonic'])

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
            Calculated threshold value
        """
        # Remove NaN values for threshold calculation
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
        except Exception:
            # Fallback to 10% of max if automatic method fails
            return np.nanmax(data) * 0.1

        return 0

    def on_threshold_method_changed(self):
        """Callback when threshold method is changed."""
        if (
            self._updating_threshold
            or self.parent_widget._labels_layer_with_phasor_features is None
        ):
            return

        method = self.threshold_method_combobox.currentText()

        if method == "None":
            self._updating_threshold = True
            self.threshold_slider.setValue(0)
            self.label_3.setText('Intensity threshold: 0')
            self.update_threshold_line()
            self._updating_threshold = False

        elif method != "Manual":
            labels_layer_name = (
                self.parent_widget.image_layer_with_phasor_features_combobox.currentText()
            )
            mean_data = (
                self.viewer.layers[labels_layer_name]
                .metadata['original_mean']
                .copy()
            )

            threshold_value = self.calculate_automatic_threshold(
                method, mean_data
            )

            self._updating_threshold = True
            self.threshold_slider.setValue(
                int(threshold_value * self.threshold_factor)
            )
            self.label_3.setText(
                'Intensity threshold: '
                + str(self.threshold_slider.value() / self.threshold_factor)
            )
            self.update_threshold_line()
            self._updating_threshold = False

    def on_labels_layer_with_phasor_features_changed(self):
        """Callback function when the image layer combobox is changed."""
        labels_layer_name = (
            self.parent_widget.image_layer_with_phasor_features_combobox.currentText()
        )
        if labels_layer_name == "":
            self.parent_widget._labels_layer_with_phasor_features = None
            return
        layer_metadata = self.viewer.layers[labels_layer_name].metadata
        self.parent_widget._labels_layer_with_phasor_features = layer_metadata[
            "phasor_features_labels_layer"
        ]

        max_mean_value = np.nanmax(layer_metadata["original_mean"])
        # Calculate threshold factor based on maximum mean value
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

        # Block threshold method updates while setting initial values
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
                self.threshold_slider.setValue(
                    int(settings["threshold"] * self.threshold_factor)
                )
                self.label_3.setText(
                    'Intensity threshold: '
                    + str(
                        self.threshold_slider.value() / self.threshold_factor
                    )
                )
            else:
                self.threshold_slider.setValue(
                    int(max_mean_value * 0.1 * self.threshold_factor)
                )
                self.label_3.setText(
                    'Intensity threshold: '
                    + str(
                        self.threshold_slider.value() / self.threshold_factor
                    )
                )
            if "filter" in settings.keys():
                filter_settings = settings["filter"]
                if "method" in filter_settings:
                    method = filter_settings["method"]
                    if method == "median":
                        self.filter_method_combobox.setCurrentText("Median")
                    elif method == "wavelet":
                        phasor_features = layer_metadata[
                            'phasor_features_labels_layer'
                        ].features
                        harmonics = np.unique(phasor_features['harmonic'])
                        if validate_harmonics_for_wavelet(harmonics):
                            self.filter_method_combobox.setCurrentText(
                                "Wavelet"
                            )
                        else:
                            self.filter_method_combobox.setCurrentText(
                                "Median"
                            )

                # Restore filter parameters
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
            self.threshold_slider.setValue(
                int(max_mean_value * 0.1 * self.threshold_factor)
            )
            self.label_3.setText(
                'Intensity threshold: '
                + str(self.threshold_slider.value() / self.threshold_factor)
            )

        self._updating_threshold = False
        self.plot_mean_histogram()

        self.check_harmonics_compatibility()

        current_method = self.threshold_method_combobox.currentText()
        if current_method != "Manual":
            self.on_threshold_method_changed()

    def on_threshold_slider_change(self):
        """Callback function when the threshold slider value changes."""
        if not self._updating_threshold:
            current_method = self.threshold_method_combobox.currentText()
            slider_value = self.threshold_slider.value()

            if not (current_method == "None" and slider_value == 0):
                self.threshold_method_combobox.setCurrentText("Manual")

        self.label_3.setText(
            'Intensity threshold: '
            + str(self.threshold_slider.value() / self.threshold_factor)
        )

        self.update_threshold_line()

    def on_median_kernel_size_change(self):
        kernel_value = self.median_filter_spinbox.value()
        self.median_filter_label.setText(
            'Median Filter Kernel Size: ' + f'{kernel_value} x {kernel_value}'
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
        """Plot the histogram of the mean intensity data as a line plot."""
        if self.parent_widget._labels_layer_with_phasor_features is None:
            return
        labels_layer_name = (
            self.parent_widget.image_layer_with_phasor_features_combobox.currentText()
        )
        mean_data = (
            self.viewer.layers[labels_layer_name]
            .metadata['original_mean']
            .copy()
        )
        self.hist_ax.clear()
        self.threshold_line = None  # Reset line reference when clearing
        self.hist_ax.hist(
            mean_data.flatten(), bins=100, color='white', edgecolor='white'
        )
        # Apply styling after clearing/plotting
        self.style_histogram_axes()

        # Add the threshold line if slider has a value
        self.update_threshold_line()
        self.hist_fig.canvas.draw_idle()

    def update_threshold_line(self):
        """Update the vertical threshold line on the histogram."""
        if self.parent_widget._labels_layer_with_phasor_features is None:
            return

        threshold_value = self.threshold_slider.value() / self.threshold_factor

        # Remove existing threshold line if it exists
        if self.threshold_line is not None:
            self.threshold_line.remove()
            self.threshold_line = None

        # Add new threshold line
        self.threshold_line = self.hist_ax.axvline(
            x=threshold_value,
            color='red',
            linestyle='-',
            linewidth=2,
            label='Threshold',
        )

        self.hist_fig.canvas.draw_idle()

    def on_log_scale_changed(self, state):
        """Callback when log scale checkbox is toggled."""
        if self.parent_widget._labels_layer_with_phasor_features is None:
            return

        if state == 2:
            self.hist_ax.set_yscale('log')
        else:
            self.hist_ax.set_yscale('linear')

        self.hist_fig.canvas.draw_idle()

    def apply_button_clicked(self):
        """Apply the filter and threshold to the selected layer."""
        if (
            not self.parent_widget.image_layer_with_phasor_features_combobox.currentText()
        ):
            show_error("Please select an image layer with phasor features.")
            return

        labels_layer_name = (
            self.parent_widget.image_layer_with_phasor_features_combobox.currentText()
        )

        layer = self.viewer.layers[labels_layer_name]
        if "settings" not in layer.metadata:
            layer.metadata["settings"] = {}
        layer.metadata["settings"][
            "threshold_method"
        ] = self.threshold_method_combobox.currentText()

        # Determine filter method and parameters
        filter_method = self.filter_method_combobox.currentText().lower()

        # Get harmonics for wavelet filter
        harmonics = None
        if filter_method == "wavelet":
            harmonics = np.unique(
                layer.metadata['phasor_features_labels_layer'].features[
                    'harmonic'
                ]
            )

        apply_filter_and_threshold(
            layer,
            threshold=self.threshold_slider.value() / self.threshold_factor,
            filter_method=filter_method,
            size=self.median_filter_spinbox.value(),
            repeat=self.median_filter_repetition_spinbox.value(),
            sigma=self.wavelet_sigma_spinbox.value(),
            levels=self.wavelet_levels_spinbox.value(),
            harmonics=harmonics,
        )
        if self.parent_widget is not None:
            self.parent_widget.plot()

            # Update lifetime tab if it exists and has a lifetime type selected
            if (
                hasattr(self.parent_widget, 'lifetime_tab')
                and self.parent_widget.lifetime_tab is not None
            ):
                current_lifetime_type = (
                    self.parent_widget.lifetime_tab.lifetime_type_combobox.currentText()
                )
                if current_lifetime_type != "None":
                    self.parent_widget.lifetime_tab._on_lifetime_type_changed(
                        current_lifetime_type
                    )
