from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.colors import LinearSegmentedColormap
from napari.layers import Image
from napari.utils.notifications import show_error
from phasorpy.lifetime import (
    phasor_to_apparent_lifetime,
    phasor_to_normal_lifetime,
)
from qtpy.QtCore import Qt
from qtpy.QtGui import QDoubleValidator
from qtpy.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)
from superqt import QRangeSlider

from ._utils import update_frequency_in_metadata

if TYPE_CHECKING:
    import napari


class LifetimeWidget(QWidget):
    """Widget to plot lifetime values."""

    def __init__(self, viewer: "napari.viewer.Viewer", parent=None):
        super().__init__()
        self.viewer = viewer
        self.parent_widget = parent
        self.frequency = None
        self.lifetime_data = None
        self.lifetime_data_original = None  # Store original unclipped data
        self.lifetime_layer = None
        self.min_lifetime = None
        self.max_lifetime = None
        self.lifetime_colormap = None
        self.colormap_contrast_limits = None
        self.lifetime_type = None
        self.hist_fig, self.hist_ax = plt.subplots(
            figsize=(8, 4), constrained_layout=True
        )
        self.counts = None
        self.bin_edges = None
        self.bin_centers = None
        self.lifetime_range_factor = (
            1000  # Factor to convert to integer for slider
        )
        self._slider_being_dragged = False  # Track drag state
        self._updating_contrast_limits = (
            False  # Flag to track contrast limits updates
        )

        # Style the histogram axes and figure initially
        self.style_histogram_axes()

        # Create scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Create content widget for scroll area
        content_widget = QWidget()
        self.main_layout = QVBoxLayout(content_widget)

        # Set scroll area content
        scroll_area.setWidget(content_widget)

        # Create main layout for this widget
        main_widget_layout = QVBoxLayout(self)
        main_widget_layout.addWidget(scroll_area)
        main_widget_layout.setStretch(0, 1)

        # Frequency input
        self.main_layout.addWidget(QLabel("Frequency: "))
        self.frequency_input = QLineEdit()
        self.frequency_input.setValidator(QDoubleValidator())
        self.main_layout.addWidget(self.frequency_input)

        # Add combobox to select lifetime type
        self.main_layout.addWidget(QLabel("Select lifetime to display: "))
        self.lifetime_type_combobox = QComboBox()
        self.lifetime_type_combobox.addItems(
            [
                "None",
                "Apparent Phase Lifetime",
                "Apparent Modulation Lifetime",
                "Normal Lifetime",
            ]
        )
        self.lifetime_type_combobox.setCurrentText("None")
        self.main_layout.addWidget(self.lifetime_type_combobox)

        self.lifetime_type_combobox.currentTextChanged.connect(
            self._on_lifetime_type_changed
        )

        # Add lifetime range slider
        self.lifetime_range_label = QLabel("Lifetime range (ns): 0.0 - 100.0")
        self.main_layout.addWidget(self.lifetime_range_label)

        # Add line edits for manual entry
        lifetime_range_edit_layout = QHBoxLayout()
        self.lifetime_min_edit = QLineEdit("0.0")
        self.lifetime_max_edit = QLineEdit("100.0")
        self.lifetime_min_edit.setValidator(QDoubleValidator())
        self.lifetime_max_edit.setValidator(QDoubleValidator())
        lifetime_range_edit_layout.addWidget(QLabel("Min:"))
        lifetime_range_edit_layout.addWidget(self.lifetime_min_edit)
        lifetime_range_edit_layout.addWidget(QLabel("Max:"))
        lifetime_range_edit_layout.addWidget(self.lifetime_max_edit)
        self.main_layout.addLayout(lifetime_range_edit_layout)

        self.lifetime_range_slider = QRangeSlider(Qt.Orientation.Horizontal)
        self.lifetime_range_slider.setRange(0, 100)
        self.lifetime_range_slider.setValue((0, 100))
        self.lifetime_range_slider.setBarMovesAllHandles(False)

        # Connect to valueChanged for label updates only
        self.lifetime_range_slider.valueChanged.connect(
            self._on_lifetime_range_label_update
        )
        # Connect to mouse events for histogram updates
        self.lifetime_range_slider.sliderPressed.connect(
            self._on_slider_pressed
        )
        self.lifetime_range_slider.sliderReleased.connect(
            self._on_slider_released
        )
        self.main_layout.addWidget(self.lifetime_range_slider)

        # Connect line edits to update slider
        self.lifetime_min_edit.editingFinished.connect(
            self._on_lifetime_min_edit
        )
        self.lifetime_max_edit.editingFinished.connect(
            self._on_lifetime_max_edit
        )

        # Add histogram widget
        self.histogram_widget = QWidget(self)
        self.histogram_layout = QVBoxLayout(self.histogram_widget)

        # Embed the Matplotlib figure into the widget with fixed size
        canvas = FigureCanvas(self.hist_fig)
        canvas.setFixedHeight(150)
        canvas.setSizePolicy(
            canvas.sizePolicy().Expanding, canvas.sizePolicy().Fixed
        )
        self.histogram_layout.addWidget(canvas)

        self.main_layout.addWidget(self.histogram_widget)
        self.histogram_widget.hide()

    def style_histogram_axes(self):
        """Apply consistent styling to the histogram axes and figure."""
        self.hist_ax.patch.set_alpha(0)
        self.hist_fig.patch.set_alpha(0)
        for spine in self.hist_ax.spines.values():
            spine.set_color('grey')
            spine.set_linewidth(1)
        self.hist_ax.set_ylabel("Pixel count", fontsize=6, color='grey')
        self.hist_ax.set_xlabel("Lifetime (ns)", fontsize=6, color='grey')
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

    def _on_slider_pressed(self):
        """Called when slider is pressed."""
        self._slider_being_dragged = True

    def _on_slider_released(self):
        """Called when slider is released."""
        self._slider_being_dragged = False
        # Update histogram only when released
        value = self.lifetime_range_slider.value()
        self._on_lifetime_range_changed(value)

    def _on_lifetime_range_label_update(self, value):
        """Update only the label while dragging, not the histogram."""
        min_val, max_val = value
        min_lifetime = min_val / self.lifetime_range_factor
        max_lifetime = max_val / self.lifetime_range_factor
        self.lifetime_range_label.setText(
            f"Lifetime range (ns): {min_lifetime:.2f} - {max_lifetime:.2f}"
        )

        self.lifetime_min_edit.setText(f"{min_lifetime:.2f}")
        self.lifetime_max_edit.setText(f"{max_lifetime:.2f}")

    def _on_lifetime_range_changed(self, value):
        """Callback when lifetime range slider changes."""
        min_val, max_val = value
        min_lifetime = min_val / self.lifetime_range_factor
        max_lifetime = max_val / self.lifetime_range_factor

        # Apply clipping to the lifetime data
        if self.lifetime_data_original is not None:
            self.lifetime_data = np.clip(
                self.lifetime_data_original, min_lifetime, max_lifetime
            )

            # Update the lifetime layer if it exists
            if self.lifetime_layer is not None:
                self.lifetime_layer.data = self.lifetime_data

                # Set flag to prevent recursive updates from colormap change event
                self._updating_contrast_limits = True
                try:
                    self.lifetime_layer.contrast_limits = [
                        min_lifetime,
                        max_lifetime,
                    ]
                    # Update our stored contrast limits to match
                    self.colormap_contrast_limits = [
                        min_lifetime,
                        max_lifetime,
                    ]
                finally:
                    self._updating_contrast_limits = False

            # Update histogram with clipped data
            self.plot_lifetime_histogram()

    def calculate_lifetimes(self):
        """Calculate the lifetimes for all harmonics."""
        if self.parent_widget._labels_layer_with_phasor_features is None:
            return

        frequency_text = self.frequency_input.text().strip()
        if not frequency_text:
            show_error("Enter frequency")
            return

        self.frequency = float(frequency_text)
        phasor_data = (
            self.parent_widget._labels_layer_with_phasor_features.features
        )
        frequency = self.frequency * self.parent_widget.harmonic
        harmonic_mask = phasor_data['harmonic'] == self.parent_widget.harmonic

        # Get the current layer data to check for filtered pixels
        layer_data = self.parent_widget._labels_layer_with_phasor_features.data

        # Create a mask for valid (non-NaN, non-zero) pixels in the layer
        valid_pixel_mask = ~np.isnan(layer_data) & (layer_data != 0)
        valid_pixel_indices = np.where(valid_pixel_mask.flatten())[0]

        # Filter phasor data to only include valid pixels
        filtered_harmonic_mask = harmonic_mask & phasor_data.index.isin(
            valid_pixel_indices
        )

        real = phasor_data.loc[filtered_harmonic_mask, 'G']
        imag = phasor_data.loc[filtered_harmonic_mask, 'S']

        if self.lifetime_type_combobox.currentText() == "Normal Lifetime":
            lifetime_values = phasor_to_normal_lifetime(
                real, imag, frequency=frequency
            )
        else:
            phase_lifetime, modulation_lifetime = phasor_to_apparent_lifetime(
                real, imag, frequency=frequency
            )

            if (
                self.lifetime_type_combobox.currentText()
                == "Apparent Phase Lifetime"
            ):
                lifetime_values = np.clip(phase_lifetime, a_min=0, a_max=None)
            else:
                lifetime_values = np.clip(
                    modulation_lifetime, a_min=0, a_max=None
                )

        # Create output array with same shape as layer, filled with NaN
        self.lifetime_data_original = np.full(layer_data.shape, np.nan)

        # Only fill in lifetime values for valid pixels
        valid_flat_indices = np.where(valid_pixel_mask.flatten())[0]
        lifetime_flat = np.full(layer_data.size, np.nan)
        lifetime_flat[valid_flat_indices[: len(lifetime_values)]] = (
            lifetime_values
        )

        self.lifetime_data_original = lifetime_flat.reshape(layer_data.shape)

        # Initialize clipped data as copy of original
        self.lifetime_data = self.lifetime_data_original.copy()

        self._update_lifetime_range_slider()

    def _update_lifetime_range_slider(self):
        """Update the lifetime range slider based on the calculated lifetime data."""
        if self.lifetime_data_original is None:
            return
        if self.frequency is None:
            return
        # Get the range of lifetime values (excluding NaNs, zeros, and infinite values)
        flattened_data = self.lifetime_data_original.flatten()
        valid_data = flattened_data[
            ~np.isnan(flattened_data)
            & (flattened_data > 0)
            & np.isfinite(flattened_data)
        ]
        if len(valid_data) == 0:
            # If no valid data, set default values
            self.min_lifetime = 0.0
            self.max_lifetime = 10.0
            min_slider_val = 0
            max_slider_val = 10000
        else:
            self.min_lifetime = np.min(valid_data)
            self.max_lifetime = np.max(valid_data)

            # Additional safety check for infinite or unrealistic very large values
            if (
                not np.isfinite(self.min_lifetime)
                or not np.isfinite(self.max_lifetime)
                or self.max_lifetime > (2e3 / self.frequency)
                or self.min_lifetime < 0
            ):
                self.min_lifetime = 0.0
                self.max_lifetime = 2e3 / self.frequency  # 2 periods in ns
                min_slider_val = 0
                max_slider_val = int(
                    self.max_lifetime * self.lifetime_range_factor
                )
            else:
                # Set slider range
                min_slider_val = int(
                    self.min_lifetime * self.lifetime_range_factor
                )
                max_slider_val = int(
                    self.max_lifetime * self.lifetime_range_factor
                )
        self.lifetime_range_slider.setRange(0, max_slider_val)
        self.lifetime_range_slider.setValue((min_slider_val, max_slider_val))
        # Update label
        self.lifetime_range_label.setText(
            f"Lifetime range (ns): {self.min_lifetime:.2f} - {self.max_lifetime:.2f}"
        )
        # Update line edits
        self.lifetime_min_edit.setText(f"{self.min_lifetime:.2f}")
        self.lifetime_max_edit.setText(f"{self.max_lifetime:.2f}")

    def _on_lifetime_min_edit(self):
        min_val = float(self.lifetime_min_edit.text())
        max_val = float(self.lifetime_max_edit.text())

        # Clamp to valid range
        min_val = max(0.0, min(min_val, self.max_lifetime or 0))
        max_val = max(0.0, min(max_val, self.max_lifetime or 0))

        # Ensure min < max with small epsilon
        if min_val >= max_val:
            max_val = min_val + 0.01

        min_slider = int(min_val * self.lifetime_range_factor)
        max_slider = int(max_val * self.lifetime_range_factor)
        self.lifetime_range_slider.setValue((min_slider, max_slider))
        self._on_lifetime_range_changed((min_slider, max_slider))

    def _on_lifetime_max_edit(self):
        min_val = float(self.lifetime_min_edit.text())
        max_val = float(self.lifetime_max_edit.text())

        # Clamp to valid range
        min_val = max(0.0, min(min_val, self.max_lifetime or 0))
        max_val = max(0.0, min(max_val, self.max_lifetime or 0))

        # Ensure min < max with small epsilon
        if max_val <= min_val:
            min_val = max_val - 0.01 if max_val > 0.01 else 0.0

        min_slider = int(min_val * self.lifetime_range_factor)
        max_slider = int(max_val * self.lifetime_range_factor)
        self.lifetime_range_slider.setValue((min_slider, max_slider))
        self._on_lifetime_range_changed((min_slider, max_slider))

    def plot_lifetime_histogram(self):
        """Plot the histogram of the lifetime data as a line plot."""
        if self.lifetime_data is None:
            self.histogram_widget.hide()
            return
        if self.parent_widget._labels_layer_with_phasor_features is None:
            self.histogram_widget.hide()
            return
        if self.parent_widget.harmonic is None:
            self.histogram_widget.hide()
            return

        flattened_data = self.lifetime_data.flatten()
        flattened_data = flattened_data[~np.isnan(flattened_data)]
        flattened_data = flattened_data[flattened_data > 0]
        flattened_data = flattened_data[np.isfinite(flattened_data)]

        if len(flattened_data) == 0:
            self.hist_ax.clear()
            self.hist_ax.text(
                0.5,
                0.5,
                'No valid data',
                transform=self.hist_ax.transAxes,
                ha='center',
            )
            self.hist_fig.canvas.draw_idle()
            self.histogram_widget.show()
            return

        self.counts, self.bin_edges = np.histogram(flattened_data, bins=300)
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2

        self._update_lifetime_histogram()
        self.histogram_widget.show()

    def create_lifetime_layer(self):
        """Create or update the lifetime layer for all harmonics."""
        if self.lifetime_data is None:
            return
        lifetime_layer_name = f"{self.lifetime_type_combobox.currentText()}: {self.parent_widget.image_layer_with_phasor_features_combobox.currentText()}"
        selected_lifetime_layer = Image(
            self.lifetime_data,
            name=lifetime_layer_name,
            scale=self.parent_widget._labels_layer_with_phasor_features.scale,
            colormap='plasma',
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

    def _update_lifetime_histogram(self):
        """Update the histogram plot with the current histogram values."""
        self.hist_ax.clear()
        self.hist_ax.plot(self.bin_centers, self.counts, color='none', alpha=0)

        # Check if colormap is available, if not use a default one
        if (
            self.lifetime_colormap is None
            or self.colormap_contrast_limits is None
        ):
            # Use a default colormap if napari colormap is not available yet
            cmap = plt.cm.plasma
            norm = plt.Normalize(
                vmin=(
                    np.min(self.bin_centers)
                    if len(self.bin_centers) > 0
                    else 0
                ),
                vmax=(
                    np.max(self.bin_centers)
                    if len(self.bin_centers) > 0
                    else 1
                ),
            )
        else:
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

        # Apply styling after clearing/plotting
        self.style_histogram_axes()

        # Refresh the canvas to show the updated histogram
        self.hist_fig.canvas.draw_idle()

    def _on_harmonic_changed(self):
        """Callback whenever the harmonic selector changes."""
        self.plot_lifetime_histogram()

    def _on_colormap_changed(self, event):
        """Callback whenever the colormap changes."""
        if getattr(self, '_updating_contrast_limits', False):
            return

        layer = event.source
        self.lifetime_colormap = layer.colormap.colors
        self.colormap_contrast_limits = layer.contrast_limits

        # Only update the histogram visualization, don't change slider or data
        self._update_lifetime_histogram()

    def _on_image_layer_changed(self):
        """Callback whenever the image layer with phasor features changes."""
        # Hide histogram and clears previous data
        self.lifetime_type_combobox.setCurrentText("None")

        frequency_text = self.frequency_input.text().strip()
        if frequency_text:  # Only calculate if not empty
            self.frequency = (
                float(frequency_text) * self.parent_widget.harmonic
            )
        else:
            self.frequency = None

        # Clear histogram plot
        self.hist_ax.clear()
        self.hist_fig.canvas.draw_idle()

    def _on_lifetime_type_changed(self, text):
        """Callback when lifetime type combobox selection changes."""
        if text == "None":
            # Hide histogram and clear any existing lifetime layer
            self.histogram_widget.hide()
            if self.lifetime_layer is not None:
                self.lifetime_layer = None
                self.lifetime_data = None
                self.lifetime_data_original = None
        else:
            sample_name = (
                self.parent_widget.image_layer_with_phasor_features_combobox.currentText()
            )
            if sample_name == "":
                self.histogram_widget.hide()
                return
            frequency = self.frequency_input.text().strip()
            if frequency == "":
                show_error("Enter frequency")
                return

            self.calculate_lifetimes()
            self._update_lifetime_range_slider()
            self.create_lifetime_layer()
            self._on_lifetime_range_changed(self.lifetime_range_slider.value())
            update_frequency_in_metadata(
                self.viewer.layers[sample_name], frequency
            )
