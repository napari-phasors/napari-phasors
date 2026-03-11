import contextlib
from typing import TYPE_CHECKING

import numpy as np
from napari.layers import Image
from napari.utils.notifications import show_error, show_warning
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
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ._utils import HistogramWidget, update_frequency_in_metadata

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
        self.per_layer_lifetime_data = {}  # {layer_name: data}
        self.per_layer_lifetime_data_original = {}  # {layer_name: data}
        self.lifetime_layer = (
            None  # Reference to first layer for backward compatibility
        )
        self.lifetime_layers = []  # List of all lifetime layers
        self.min_lifetime = None
        self.max_lifetime = None
        self.lifetime_colormap = None
        self.colormap_contrast_limits = None
        self.lifetime_type = None
        self.lifetime_range_factor = (
            1000  # Factor to convert to integer for slider
        )
        self._updating_contrast_limits = (
            False  # Flag to track contrast limits updates
        )
        self._updating_settings = False  # Flag to prevent recursive updates
        self._updating_linked_layers = (
            False  # Flag to prevent recursive layer updates
        )

        # Create scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

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
        frequency_layout = QHBoxLayout()
        frequency_layout.addWidget(QLabel("Frequency (MHz): "))
        self.frequency_input = QLineEdit()
        self.frequency_input.setValidator(QDoubleValidator())
        frequency_layout.addWidget(self.frequency_input)
        self.main_layout.addLayout(frequency_layout)

        # Add combobox to select lifetime type
        lifetime_type_layout = QHBoxLayout()
        lifetime_type_layout.addWidget(QLabel("Select lifetime to display: "))
        self.lifetime_type_combobox = QComboBox()
        self.lifetime_type_combobox.addItems(
            [
                "Apparent Phase Lifetime",
                "Apparent Modulation Lifetime",
                "Normal Lifetime",
            ]
        )
        self.lifetime_type_combobox.setCurrentText("Apparent Phase Lifetime")
        lifetime_type_layout.addWidget(self.lifetime_type_combobox)

        # Add Calculate button in its own row
        self.calculate_lifetime_button = QPushButton("Calculate Lifetime")
        self.calculate_lifetime_button.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Fixed
        )
        self.calculate_lifetime_button.setToolTip(
            "Calculate and display lifetime values for the selected layers"
        )
        self.calculate_lifetime_button.clicked.connect(
            self._on_calculate_lifetime_clicked
        )

        self.main_layout.addLayout(lifetime_type_layout)
        self.main_layout.addWidget(self.calculate_lifetime_button)

        # Connect signals
        self.lifetime_type_combobox.currentTextChanged.connect(
            self._on_lifetime_type_changed
        )
        self.frequency_input.editingFinished.connect(
            self._on_frequency_changed
        )

        # NOTE: The widget is created here but NOT added to this tab's layout.
        # PlotterWidget wraps it in a HistogramDockWidget and docks it separately.
        self.histogram_widget = HistogramWidget(
            xlabel="Lifetime (ns)",
            ylabel="Pixel count",
            bins=150,
            default_colormap_name="plasma",
            range_slider_enabled=True,
            range_label_prefix="Lifetime range (ns)",
            range_factor=self.lifetime_range_factor,
            parent=self,
        )

        # Convenience aliases so the rest of the class (and tests) can
        # keep referring to the same attribute names.
        self.lifetime_range_slider = self.histogram_widget.range_slider
        self.lifetime_range_label = self.histogram_widget.range_label
        self.lifetime_min_edit = self.histogram_widget.range_min_edit
        self.lifetime_max_edit = self.histogram_widget.range_max_edit

        # Connect the histogram widget's rangeChanged signal
        self.histogram_widget.rangeChanged.connect(
            self._on_range_changed_from_histogram
        )

    def _get_default_lifetime_settings(self):
        """Get default settings dictionary for lifetime parameters."""
        return {
            'lifetime_type': 'Apparent Phase Lifetime',
            'lifetime_range_min': None,
            'lifetime_range_max': None,
        }

    def _update_lifetime_setting_in_metadata(self, key, value):
        """Update a specific lifetime setting in the current layer's metadata."""
        if self._updating_settings:
            return

        layer_name = self.parent_widget.get_primary_layer_name()
        if layer_name:
            layer = self.viewer.layers[layer_name]
            if 'settings' not in layer.metadata:
                layer.metadata['settings'] = {}
            if 'lifetime' not in layer.metadata['settings']:
                layer.metadata['settings'][
                    'lifetime'
                ] = self._get_default_lifetime_settings()
            layer.metadata['settings']['lifetime'][key] = value

    def _restore_lifetime_settings_from_metadata(self):
        """Restore all lifetime settings from the current layer's metadata."""
        layer_name = self.parent_widget.get_primary_layer_name()
        if not layer_name:
            return

        layer = self.viewer.layers[layer_name]

        if 'settings' in layer.metadata:
            if 'frequency' in layer.metadata['settings']:
                frequency = layer.metadata['settings']['frequency']
                self._updating_settings = True
                try:
                    self.frequency_input.setText(str(frequency))
                finally:
                    self._updating_settings = False
            else:
                self._updating_settings = True
                try:
                    self.frequency_input.clear()
                finally:
                    self._updating_settings = False
        else:
            self._updating_settings = True
            try:
                self.frequency_input.clear()
            finally:
                self._updating_settings = False

        if (
            'settings' not in layer.metadata
            or 'lifetime' not in layer.metadata['settings']
        ):
            self._updating_settings = True
            try:
                self.lifetime_type_combobox.setCurrentText(
                    'Apparent Phase Lifetime'
                )
                self.lifetime_range_slider.setValue((0, 100))
                self.lifetime_min_edit.setText('0.0')
                self.lifetime_max_edit.setText('100.0')
                self.lifetime_range_label.setText(
                    'Lifetime range (ns): 0.0 - 100.0'
                )
                self.histogram_widget.hide()
            finally:
                self._updating_settings = False
            return

        self._updating_settings = True
        try:
            settings = layer.metadata['settings']['lifetime']

            if 'lifetime_type' in settings:
                self.lifetime_type_combobox.setCurrentText(
                    settings['lifetime_type']
                )

        finally:
            self._updating_settings = False

    def _on_frequency_changed(self):
        """Handle frequency input changes."""
        frequency_text = self.frequency_input.text().strip()

        if not self._updating_settings:
            if frequency_text:
                try:
                    self.frequency = float(frequency_text)
                    self.calculate_lifetimes()
                    self._update_lifetime_range_slider()
                    self.create_lifetime_layer()
                    self._restore_lifetime_range_from_metadata()
                    self._on_lifetime_range_changed(
                        self.lifetime_range_slider.value()
                    )
                    self.plot_lifetime_histogram()
                except ValueError:
                    show_error(
                        "Invalid frequency value. Please enter a valid number."
                    )
            elif frequency_text == "":
                self.frequency = None
                self.histogram_widget.hide()

    def _on_range_changed_from_histogram(self, min_float, max_float):
        """Bridge between HistogramWidget.rangeChanged and the lifetime logic."""
        min_val = int(min_float * self.lifetime_range_factor)
        max_val = int(max_float * self.lifetime_range_factor)
        self._on_lifetime_range_changed((min_val, max_val))

    def _on_lifetime_range_changed(self, value):
        """Callback when lifetime range slider changes - updates all lifetime layers."""
        min_val, max_val = value
        min_lifetime = min_val / self.lifetime_range_factor
        max_lifetime = max_val / self.lifetime_range_factor

        if self.lifetime_data_original is not None:
            self.lifetime_data = np.clip(
                self.lifetime_data_original, min_lifetime, max_lifetime
            )

        selected_layers = self.parent_widget.get_selected_layers()

        self._updating_contrast_limits = True
        self._updating_linked_layers = True
        try:
            for layer in selected_layers:
                if 'lifetime_data' not in layer.metadata:
                    continue

                lifetime_data_dict = layer.metadata['lifetime_data']
                if self.parent_widget.harmonic not in lifetime_data_dict:
                    continue

                lifetime_values = lifetime_data_dict[
                    self.parent_widget.harmonic
                ]
                clipped_lifetime = np.clip(
                    lifetime_values, min_lifetime, max_lifetime
                )

                lifetime_layer_name = f"{self.lifetime_type_combobox.currentText()}: {layer.name}"

                if lifetime_layer_name in self.viewer.layers:
                    lifetime_layer = self.viewer.layers[lifetime_layer_name]
                    lifetime_layer.data = clipped_lifetime
                    lifetime_layer.contrast_limits = [
                        min_lifetime,
                        max_lifetime,
                    ]

            self.colormap_contrast_limits = [min_lifetime, max_lifetime]
        finally:
            self._updating_contrast_limits = False
            self._updating_linked_layers = False

        if not self._updating_settings:
            self._update_lifetime_setting_in_metadata(
                'lifetime_range_min', min_lifetime
            )
            self._update_lifetime_setting_in_metadata(
                'lifetime_range_max', max_lifetime
            )

        self._apply_lifetime_range_change(min_val, max_val)

    def calculate_lifetimes(self):
        """Calculate the lifetimes for all selected layers."""
        if not self.parent_widget.has_phasor_data():
            return

        frequency_text = self.frequency_input.text().strip()
        if not frequency_text:
            show_warning("Enter frequency")
            return

        base_frequency = float(frequency_text)
        self.frequency = base_frequency

        selected_layers = self.parent_widget.get_selected_layers()
        if not selected_layers:
            return

        all_lifetime_data = []
        per_layer_data = {}

        for layer in selected_layers:
            g_array = layer.metadata.get("G")
            s_array = layer.metadata.get("S")
            harmonics = np.atleast_1d(layer.metadata.get("harmonics"))

            if g_array is None or s_array is None:
                continue

            try:
                harmonic_index = np.where(
                    harmonics == self.parent_widget.harmonic
                )[0][0]
                real = g_array[harmonic_index]
                imag = s_array[harmonic_index]
            except IndexError:
                continue

            effective_frequency = base_frequency * self.parent_widget.harmonic

            with np.errstate(divide='ignore', invalid='ignore'):
                if (
                    self.lifetime_type_combobox.currentText()
                    == "Normal Lifetime"
                ):
                    lifetime_values = phasor_to_normal_lifetime(
                        real, imag, frequency=effective_frequency
                    )
                else:
                    phase_lifetime, modulation_lifetime = (
                        phasor_to_apparent_lifetime(
                            real, imag, frequency=effective_frequency
                        )
                    )

                    if (
                        self.lifetime_type_combobox.currentText()
                        == "Apparent Phase Lifetime"
                    ):
                        lifetime_values = np.clip(
                            phase_lifetime, a_min=0, a_max=None
                        )
                    else:
                        lifetime_values = np.clip(
                            modulation_lifetime, a_min=0, a_max=None
                        )

            with np.errstate(invalid='ignore'):
                lifetime_values[lifetime_values < 0] = 0

            if 'lifetime_data' not in layer.metadata:
                layer.metadata['lifetime_data'] = {}
            layer.metadata['lifetime_data'][
                self.parent_widget.harmonic
            ] = lifetime_values

            all_lifetime_data.append(lifetime_values)
            per_layer_data[layer.name] = lifetime_values

        if not all_lifetime_data:
            return

        merged_lifetime = np.concatenate(
            [data.flatten() for data in all_lifetime_data]
        )
        self.lifetime_data_original = merged_lifetime
        self.lifetime_data = self.lifetime_data_original.copy()
        self.per_layer_lifetime_data_original = {
            k: v.copy() for k, v in per_layer_data.items()
        }
        self.per_layer_lifetime_data = {
            k: v.copy() for k, v in per_layer_data.items()
        }
        self._update_lifetime_range_slider()

    def _update_lifetime_range_slider(self):
        """Update the lifetime range slider based on the calculated lifetime data."""
        if self.lifetime_data_original is None:
            return
        if self.frequency is None:
            return

        effective_frequency = self.frequency * self.parent_widget.harmonic

        flattened_data = self.lifetime_data_original.flatten()
        valid_data = flattened_data[
            ~np.isnan(flattened_data)
            & (flattened_data > 0)
            & np.isfinite(flattened_data)
        ]
        if len(valid_data) == 0:
            self.min_lifetime = 0.0
            self.max_lifetime = 10.0
            min_slider_val = 0
            max_slider_val = 10000
        else:
            self.min_lifetime = np.min(valid_data)
            self.max_lifetime = np.max(valid_data)

            if (
                not np.isfinite(self.min_lifetime)
                or not np.isfinite(self.max_lifetime)
                or self.max_lifetime > (2e3 / effective_frequency)
                or self.min_lifetime < 0
            ):
                self.min_lifetime = 0.0
                self.max_lifetime = (
                    2e3 / effective_frequency
                )  # 2 periods in ns
                min_slider_val = 0
                max_slider_val = int(
                    self.max_lifetime * self.lifetime_range_factor
                )
            else:
                min_slider_val = int(
                    self.min_lifetime * self.lifetime_range_factor
                )
                max_slider_val = int(
                    self.max_lifetime * self.lifetime_range_factor
                )
        self.lifetime_range_slider.setRange(0, max_slider_val)
        self.lifetime_range_slider.setValue((min_slider_val, max_slider_val))

        self.lifetime_range_label.setText(
            f"Lifetime range (ns): {self.min_lifetime:.2f} - {self.max_lifetime:.2f}"
        )

        self.lifetime_min_edit.setText(f"{self.min_lifetime:.2f}")
        self.lifetime_max_edit.setText(f"{self.max_lifetime:.2f}")

    def plot_lifetime_histogram(self):
        """Plot the histogram of the merged lifetime data from all selected layers."""
        if self.lifetime_data is None:
            self.histogram_widget.hide()
            return
        if not self.parent_widget.has_phasor_data():
            self.histogram_widget.hide()
            return
        if self.parent_widget.harmonic is None:
            self.histogram_widget.hide()
            return

        self.histogram_widget.update_colormap(
            colormap_colors=self.lifetime_colormap,
            contrast_limits=self.colormap_contrast_limits,
        )
        if (
            self.per_layer_lifetime_data
            and len(self.per_layer_lifetime_data) > 1
        ):
            self.histogram_widget.update_multi_data(
                self.per_layer_lifetime_data
            )
        else:
            self.histogram_widget.update_data(self.lifetime_data)

    def create_lifetime_layer(self):
        """Create or update lifetime layers for all selected layers."""
        selected_layers = self.parent_widget.get_selected_layers()
        if not selected_layers:
            return

        for layer in self.lifetime_layers:
            if layer in self.viewer.layers:
                try:
                    layer.events.colormap.disconnect(self._on_colormap_changed)
                    layer.events.contrast_limits.disconnect(
                        self._on_colormap_changed
                    )
                except (AttributeError, RuntimeError, TypeError):
                    pass
                except Exception as exc:  # noqa: BLE001
                    show_warning(
                        f"Failed to disconnect lifetime layer events for "
                        f"'{getattr(layer, 'name', 'unknown layer')}': {exc}"
                    )
        self.lifetime_layers = []
        self.lifetime_layer = None

        for layer in selected_layers:
            if 'lifetime_data' not in layer.metadata:
                continue

            lifetime_data_dict = layer.metadata['lifetime_data']
            if self.parent_widget.harmonic not in lifetime_data_dict:
                continue

            lifetime_values = lifetime_data_dict[self.parent_widget.harmonic]

            lifetime_layer_name = (
                f"{self.lifetime_type_combobox.currentText()}: {layer.name}"
            )

            min_val, max_val = self.lifetime_range_slider.value()
            min_lifetime = min_val / self.lifetime_range_factor
            max_lifetime = max_val / self.lifetime_range_factor
            clipped_lifetime = np.clip(
                lifetime_values, min_lifetime, max_lifetime
            )

            selected_lifetime_layer = Image(
                clipped_lifetime,
                name=lifetime_layer_name,
                scale=layer.scale,
                colormap='plasma',
                contrast_limits=[min_lifetime, max_lifetime],
            )

            if lifetime_layer_name in self.viewer.layers:
                self.viewer.layers.remove(
                    self.viewer.layers[lifetime_layer_name]
                )

            lifetime_layer = self.viewer.add_layer(selected_lifetime_layer)

            self.lifetime_layers.append(lifetime_layer)
            lifetime_layer.events.colormap.connect(self._on_colormap_changed)
            lifetime_layer.events.contrast_limits.connect(
                self._on_colormap_changed
            )

            if self.lifetime_layer is None:
                self.lifetime_layer = lifetime_layer
                self.lifetime_colormap = lifetime_layer.colormap.colors
                self.colormap_contrast_limits = lifetime_layer.contrast_limits

    def _on_colormap_changed(self, event):
        """Callback whenever the colormap or contrast limits change on any lifetime layer - sync all layers."""
        if getattr(self, '_updating_contrast_limits', False) or getattr(
            self, '_updating_linked_layers', False
        ):
            return

        source_layer = event.source
        new_colormap = source_layer.colormap
        new_contrast_limits = source_layer.contrast_limits

        # Update stored values
        self.lifetime_colormap = new_colormap.colors
        self.colormap_contrast_limits = new_contrast_limits

        # Update all other lifetime layers to match
        self._updating_linked_layers = True
        try:
            for layer in self.lifetime_layers:
                if layer != source_layer and layer in self.viewer.layers:
                    layer.colormap = new_colormap
                    layer.contrast_limits = new_contrast_limits
        finally:
            self._updating_linked_layers = False

        self.histogram_widget.update_colormap(
            colormap_colors=self.lifetime_colormap,
            contrast_limits=self.colormap_contrast_limits,
        )

    def _on_image_layer_changed(self):
        """Callback whenever the image layer with phasor features changes.

        This only restores UI state from metadata - it does NOT run calculations.
        User must click "Calculate" button to run lifetime analysis.
        """
        layer_name = self.parent_widget.get_primary_layer_name()
        if layer_name:
            self._restore_lifetime_settings_from_metadata()
            # Don't auto-calculate - just restore UI state
            # User must click "Calculate" to run analysis
            self.histogram_widget.hide()
        else:
            # Disconnect events from all lifetime layers
            for layer in self.lifetime_layers:
                if layer in self.viewer.layers:
                    try:
                        layer.events.colormap.disconnect(
                            self._on_colormap_changed
                        )
                        layer.events.contrast_limits.disconnect(
                            self._on_colormap_changed
                        )
                    except Exception:  # noqa: BLE001
                        pass

            self.lifetime_data = None
            self.lifetime_data_original = None
            self.per_layer_lifetime_data = {}
            self.per_layer_lifetime_data_original = {}
            self.lifetime_layer = None
            self.lifetime_layers = []

            self.histogram_widget.clear()

    def _on_calculate_lifetime_clicked(self):
        """Callback when Calculate button is clicked.

        Runs the lifetime calculation and creates/updates lifetime layers.
        """
        selected_layers = self.parent_widget.get_selected_layers()
        if not selected_layers:
            show_warning("Select at least one layer")
            return

        frequency = self.frequency_input.text().strip()
        if frequency == "":
            show_warning("Enter frequency")
            return

        # Run the calculation
        self.calculate_lifetimes()
        self._update_lifetime_range_slider()
        self.create_lifetime_layer()

        self._restore_lifetime_range_from_metadata()
        self._on_lifetime_range_changed(self.lifetime_range_slider.value())

        if self.lifetime_data is not None:
            self.plot_lifetime_histogram()

        # Update frequency in metadata
        if not self._updating_settings:
            try:
                frequency_float = float(frequency)
                for layer in selected_layers:
                    update_frequency_in_metadata(layer, frequency_float)
            except ValueError:
                pass

    def _on_lifetime_type_changed(self, text):
        """Callback when lifetime type combobox selection changes.

        This only updates the setting in metadata - it does NOT run calculations.
        User must click "Calculate" button to run lifetime analysis.
        """
        if not self._updating_settings:
            self._update_lifetime_setting_in_metadata('lifetime_type', text)
        # Don't auto-calculate - user must click Calculate

    def _restore_lifetime_range_from_metadata(self):
        """Restore lifetime range from metadata after calculation."""
        layer_name = self.parent_widget.get_primary_layer_name()
        if not layer_name:
            return

        layer = self.viewer.layers[layer_name]
        if (
            'settings' in layer.metadata
            and 'lifetime' in layer.metadata['settings']
        ):
            settings = layer.metadata['settings']['lifetime']

            if (
                'lifetime_range_min' in settings
                and 'lifetime_range_max' in settings
                and settings['lifetime_range_min'] is not None
                and settings['lifetime_range_max'] is not None
            ):

                min_val = settings['lifetime_range_min']
                max_val = settings['lifetime_range_max']

                if (
                    self.min_lifetime is not None
                    and self.max_lifetime is not None
                    and min_val >= self.min_lifetime
                    and max_val <= self.max_lifetime
                ):

                    min_slider = int(min_val * self.lifetime_range_factor)
                    max_slider = int(max_val * self.lifetime_range_factor)

                    self._updating_settings = True
                    try:
                        self.lifetime_range_slider.setValue(
                            (min_slider, max_slider)
                        )
                        self.lifetime_min_edit.setText(f"{min_val:.2f}")
                        self.lifetime_max_edit.setText(f"{max_val:.2f}")
                        self.lifetime_range_label.setText(
                            f"Lifetime range (ns): {min_val:.2f} - {max_val:.2f}"
                        )
                    finally:
                        self._updating_settings = False

                    self._apply_lifetime_range_change(min_slider, max_slider)

    def _apply_lifetime_range_change(self, min_slider, max_slider):
        """Apply lifetime range change for histogram without updating layers (layers updated in _on_lifetime_range_changed)."""
        min_lifetime = min_slider / self.lifetime_range_factor
        max_lifetime = max_slider / self.lifetime_range_factor

        # Only update the histogram data (merged/flattened), not the individual layers
        if self.lifetime_data_original is not None:
            self.lifetime_data = np.clip(
                self.lifetime_data_original, min_lifetime, max_lifetime
            )
            # Also clip per-layer data for multi-layer histogram modes
            for name, orig in self.per_layer_lifetime_data_original.items():
                self.per_layer_lifetime_data[name] = np.clip(
                    orig, min_lifetime, max_lifetime
                )
            self.plot_lifetime_histogram()

    def closeEvent(self, event):
        """Clean up signal connections before closing."""
        # Disconnect all lifetime layer events
        for layer in self.lifetime_layers:
            with contextlib.suppress(ValueError, AttributeError):
                layer.events.colormap.disconnect(self._on_colormap_changed)
            with contextlib.suppress(ValueError, AttributeError):
                layer.events.contrast_limits.disconnect(
                    self._on_contrast_limits_changed
                )

        event.accept()
