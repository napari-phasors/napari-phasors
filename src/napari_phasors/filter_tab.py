from math import ceil, log10

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from napari.utils.notifications import show_error
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ._utils import apply_filter_and_threshold


class FilterWidget(QWidget):
    """Widget for interactive filtering and thresholding of phasor features.

    Provides controls for:
      - Median filtering (kernel size and repetitions)
      - Intensity thresholding via a slider
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
        self.hist_fig, self.hist_ax = plt.subplots(figsize=(8, 4))
        self.hist_fig.subplots_adjust(
            top=0.98, bottom=0.2, left=0.05, right=0.98
        )
        self.threshold_line = None

        # Create UI elements
        self.setup_ui()

        # Connect callbacks
        self.parent_widget.image_layer_with_phasor_features_combobox.currentIndexChanged.connect(
            self.on_labels_layer_with_phasor_features_changed
        )
        self.threshold_slider.valueChanged.connect(
            self.on_threshold_slider_change
        )
        self.median_filter_spinbox.valueChanged.connect(
            self.on_kernel_size_change
        )
        self.apply_button.clicked.connect(self.apply_button_clicked)

    def setup_ui(self):
        """Setup the user interface elements."""
        layout = QVBoxLayout()

        # Create a widget to hold the scrollable content
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        # Median filter kernel size
        median_filter_layout = QHBoxLayout()
        self.label_4 = QLabel("Median Filter Kernel Size: 3 x 3")
        self.median_filter_spinbox = QSpinBox()
        self.median_filter_spinbox.setMinimum(2)
        self.median_filter_spinbox.setMaximum(99)
        self.median_filter_spinbox.setValue(3)
        median_filter_layout.addWidget(self.label_4)
        median_filter_layout.addWidget(self.median_filter_spinbox)
        scroll_layout.addLayout(median_filter_layout)

        # Median filter repetitions
        repetitions_layout = QHBoxLayout()
        repetitions_layout.addWidget(QLabel("Filter Repetitions:"))
        self.median_filter_repetition_spinbox = QSpinBox()
        self.median_filter_repetition_spinbox.setMinimum(0)
        self.median_filter_repetition_spinbox.setValue(0)
        repetitions_layout.addWidget(self.median_filter_repetition_spinbox)
        scroll_layout.addLayout(repetitions_layout)

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
        if "settings" in layer_metadata.keys():
            settings = layer_metadata["settings"]
            if "threshold" in settings.keys():
                self.threshold_slider.setValue(
                    int(settings["threshold"] * self.threshold_factor)
                )
                self.on_threshold_slider_change()
            else:
                self.threshold_slider.setValue(
                    int(max_mean_value * 0.1 * self.threshold_factor)
                )
                self.on_threshold_slider_change()
            if "filter" in settings.keys():
                self.median_filter_spinbox.setValue(
                    int(settings["filter"]["size"])
                )
                self.median_filter_repetition_spinbox.setValue(
                    int(settings["filter"]["repeat"])
                )
        else:
            self.threshold_slider.setValue(
                int(max_mean_value * 0.1 * self.threshold_factor)
            )
            self.on_threshold_slider_change()

        self.plot_mean_histogram()

    def on_threshold_slider_change(self):
        """Callback function when the threshold slider value changes."""
        self.label_3.setText(
            'Intensity threshold: '
            + str(self.threshold_slider.value() / self.threshold_factor)
        )
        # Update the threshold line on the histogram
        self.update_threshold_line()

    def on_kernel_size_change(self):
        kernel_value = self.median_filter_spinbox.value()
        self.label_4.setText(
            'Median Filter Kernel Size: ' + f'{kernel_value} x {kernel_value}'
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

        # Add the threshold line if slider has a value
        self.update_threshold_line()
        self.hist_fig.canvas.draw_idle()

    def update_threshold_line(self):
        """Update the vertical threshold line on the histogram."""
        if self.parent_widget._labels_layer_with_phasor_features is None:
            return

        # Get the current threshold value
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

        # Refresh the canvas
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
        apply_filter_and_threshold(
            self.viewer.layers[labels_layer_name],
            threshold=self.threshold_slider.value() / self.threshold_factor,
            size=self.median_filter_spinbox.value(),
            repeat=self.median_filter_repetition_spinbox.value(),
        )
        if self.parent_widget is not None:
            self.parent_widget.plot()
