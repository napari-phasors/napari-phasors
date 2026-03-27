import contextlib
import warnings
from typing import TYPE_CHECKING

import numpy as np
from napari.layers import Image
from napari.utils.notifications import show_error, show_warning
from phasorpy.lifetime import (
    phasor_to_apparent_lifetime,
    phasor_to_normal_lifetime,
)
from phasorpy.phasor import phasor_to_polar
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QColor, QDoubleValidator
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
from scipy.stats import binned_statistic_2d
from superqt import QToggleSwitch

from ._utils import (
    HistogramWidget,
    create_mpl_colormap_from_qcolor,
    populate_colormap_combobox,
    resolve_colormap_by_name,
    resolve_napari_layer_colormap,
    update_frequency_in_metadata,
)

if TYPE_CHECKING:
    import napari


class PhasorMappingWidget(QWidget):
    """Widget to calculate and display phasor mapping outputs.

    Supports lifetime-derived outputs and direct phasor outputs:
    Apparent Phase Lifetime, Apparent Modulation Lifetime,
    Normal Lifetime, Phase, and Modulation.
    """

    outputTypeChanged = Signal(str)

    def __init__(self, viewer: "napari.viewer.Viewer", parent=None):
        super().__init__()
        self.viewer = viewer
        self.parent_widget = parent
        self.frequency = None
        self.lifetime_data = None
        self.lifetime_data_original = None  # Store original unclipped data
        self.per_layer_lifetime_data = {}  # {layer_name: data}
        self.per_layer_lifetime_data_original = {}  # {layer_name: data}
        self.current_metric_data = None
        self.current_metric_data_original = None
        self.per_layer_metric_data = {}
        self.per_layer_metric_data_original = {}
        self.lifetime_layer = (
            None  # Reference to first layer for backward compatibility
        )
        self.lifetime_layers = []  # List of all lifetime layers
        self.metric_layers = []  # List of output layers for current metric
        self.current_output_type = "Apparent Phase Lifetime"
        self._overlay_imshow = None
        self._phase_colormap_name = "hsv"
        self._modulation_colormap_name = "viridis"
        self._coloring_paused_by_tab = False
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

        # Output mode (always visible, top row)
        self.output_mode_widget = QWidget()
        output_mode_layout = QHBoxLayout(self.output_mode_widget)
        output_mode_layout.setContentsMargins(0, 0, 0, 0)
        output_mode_layout.addWidget(QLabel("Parameter to Analyze: "))
        self.output_mode_combobox = QComboBox()
        self.output_mode_combobox.addItems(["Lifetime", "Phase", "Modulation"])
        self.output_mode_combobox.setCurrentText("Lifetime")
        output_mode_layout.addWidget(self.output_mode_combobox, 1)
        self.main_layout.addWidget(self.output_mode_widget)

        # Lifetime output selector (visible only for Lifetime mode)
        self.lifetime_output_widget = QWidget()
        lifetime_type_layout = QHBoxLayout(self.lifetime_output_widget)
        lifetime_type_layout.setContentsMargins(0, 0, 0, 0)
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
        lifetime_type_layout.addWidget(self.lifetime_type_combobox, 1)
        self.main_layout.addWidget(self.lifetime_output_widget)

        # Frequency input (visible only for Lifetime mode)
        self.frequency_widget = QWidget()
        frequency_layout = QHBoxLayout(self.frequency_widget)
        frequency_layout.setContentsMargins(0, 0, 0, 0)
        frequency_layout.addWidget(QLabel("Frequency (MHz): "))
        self.frequency_input = QLineEdit()
        self.frequency_input.setValidator(QDoubleValidator())
        frequency_layout.addWidget(self.frequency_input)
        self.main_layout.addWidget(self.frequency_widget)

        # 2D phasor custom-color controls (visible for Phase/Modulation modes)
        self.colormap_widget = QWidget()
        colormap_layout = QHBoxLayout(self.colormap_widget)
        colormap_layout.setContentsMargins(0, 0, 0, 0)
        colormap_layout.addWidget(QLabel("Colormap:"))
        self.colormap_combobox = QComboBox()
        populate_colormap_combobox(
            self.colormap_combobox,
            include_select_color=True,
            selected=self._phase_colormap_name,
        )
        colormap_layout.addWidget(self.colormap_combobox, 1)

        self.custom_color_button = QPushButton()
        self.custom_color_button.setFixedSize(22, 22)
        self.custom_color_button.setToolTip("Select custom color")
        self.custom_color_button.setVisible(False)
        self.custom_color_button.clicked.connect(self._on_custom_color_clicked)
        colormap_layout.addWidget(self.custom_color_button)

        self.main_layout.addWidget(self.colormap_widget)

        self.coloring_checkbox_widget = QWidget()
        coloring_checkbox_layout = QHBoxLayout(self.coloring_checkbox_widget)
        coloring_checkbox_layout.setContentsMargins(0, 0, 0, 0)
        self.apply_2d_colormap_checkbox = QToggleSwitch(
            "Apply colormap to 2D Histogram"
        )
        self.apply_2d_colormap_checkbox.onColor = QColor(
            "#27ae60"
        )  # Nice Green
        coloring_checkbox_layout.addWidget(self.apply_2d_colormap_checkbox)
        coloring_checkbox_layout.addStretch(1)
        self.main_layout.addWidget(self.coloring_checkbox_widget)

        # Add Calculate button in its own row
        self.calculate_lifetime_button = QPushButton("Calculate Output")
        self.calculate_lifetime_button.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Fixed
        )
        self.calculate_lifetime_button.setToolTip(
            "Calculate and display the selected output for all selected layers"
        )
        self.calculate_lifetime_button.clicked.connect(
            self._on_calculate_lifetime_clicked
        )

        self.main_layout.addWidget(self.calculate_lifetime_button)

        # Connect signals
        self.lifetime_type_combobox.currentTextChanged.connect(
            self._on_lifetime_type_changed
        )
        self.output_mode_combobox.currentTextChanged.connect(
            self._on_output_mode_changed
        )
        self.colormap_combobox.currentTextChanged.connect(
            self._on_colormap_combobox_changed
        )
        self.apply_2d_colormap_checkbox.toggled.connect(
            self._on_apply_2d_colormap_checkbox_changed
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
        self._configure_histogram_labels_for_output(
            self.lifetime_type_combobox.currentText()
        )
        self._sync_mode_widgets()
        self._update_calculate_button_text()
        self.update_apply_2d_text()

        # Connect to plot type changes in the parent widget
        if self.parent_widget is not None:
            self.parent_widget.plotter_inputs_widget.plot_type_combobox.currentTextChanged.connect(
                self.update_apply_2d_text
            )

    def update_apply_2d_text(self):
        """Update the checkbox text based on the current plot type."""
        plot_type = getattr(self.parent_widget, 'plot_type', 'HISTOGRAM2D')
        if plot_type == 'SCATTER':
            suffix = "Scatter plot"
        elif plot_type == 'CONTOUR':
            suffix = "Contour plot"
        else:
            suffix = "2D Histogram"
        self.apply_2d_colormap_checkbox.setText(f"Apply colormap to {suffix}")

    def _on_custom_color_clicked(self):
        """Open a color dialog to select a custom color."""
        from qtpy.QtWidgets import QColorDialog

        current_color = QColor()
        # Try to parse the current button color string
        style = self.custom_color_button.styleSheet()
        if "background-color: " in style:
            try:
                rgb_str = style.split("background-color: ")[1].split(";")[0]
                if rgb_str.startswith("rgb("):
                    r, g, b = map(int, rgb_str[4:-1].split(","))
                    current_color.setRgb(r, g, b)
            except (ValueError, IndexError):
                pass

        color = QColorDialog.getColor(
            current_color, self, "Select Custom Color"
        )
        if color.isValid():
            self._set_custom_color(color)
            # Find which color name we are updating
            # In 'solid' mode, we treat the color as a single-color colormap
            # or we just apply it. For now, we'll store it as a name or similar.
            # But the existing system uses colormap names.
            # If "Select color..." is active, we need to handle it.
            self._on_colormap_combobox_changed("Select color...")

    def _set_custom_color(self, color):
        """Set the custom color on the button and update internal state."""
        self.custom_color_button.setStyleSheet(
            f"background-color: {color.name()};"
        )
        self._custom_color = color

    def _update_calculate_button_text(self):
        mode = self.output_mode_combobox.currentText()
        if mode == "Lifetime":
            self.calculate_lifetime_button.setText("Display Lifetime Map")
        elif mode == "Phase":
            self.calculate_lifetime_button.setText("Display Phase Map")
        else:
            self.calculate_lifetime_button.setText("Display Modulation Map")

    def _sync_mode_widgets(self):
        is_lifetime_mode = (
            self.output_mode_combobox.currentText() == "Lifetime"
        )
        self.lifetime_output_widget.setVisible(is_lifetime_mode)
        self.frequency_widget.setVisible(is_lifetime_mode)
        self.colormap_widget.setVisible(not is_lifetime_mode)
        self.coloring_checkbox_widget.setVisible(not is_lifetime_mode)

    def get_selected_output_display_name(self) -> str:
        output_type = self._get_selected_output_type()
        if output_type in {
            "Apparent Phase Lifetime",
            "Apparent Modulation Lifetime",
            "Normal Lifetime",
        }:
            return "Lifetime"
        return output_type

    def _get_default_lifetime_settings(self):
        """Get default settings dictionary for lifetime parameters."""
        return {
            'lifetime_type': 'Apparent Phase Lifetime',
            'lifetime_range_min': None,
            'lifetime_range_max': None,
            'output_type': 'Apparent Phase Lifetime',
            'range_min': None,
            'range_max': None,
        }

    @staticmethod
    def _output_requires_frequency(output_type: str) -> bool:
        return output_type in {
            "Apparent Phase Lifetime",
            "Apparent Modulation Lifetime",
            "Normal Lifetime",
        }

    def _get_selected_output_type(self) -> str:
        mode = self.output_mode_combobox.currentText()
        if mode == "Lifetime":
            return self.lifetime_type_combobox.currentText()
        return mode

    @staticmethod
    def _get_output_colormap_name(output_type: str) -> str:
        if output_type == "Phase":
            return "hsv"
        if output_type == "Modulation":
            return "viridis"
        return "plasma"

    def _configure_histogram_labels_for_output(self, output_type: str):
        if output_type in {
            "Apparent Phase Lifetime",
            "Apparent Modulation Lifetime",
            "Normal Lifetime",
        }:
            self.histogram_widget.xlabel = "Lifetime (ns)"
            self.histogram_widget._range_label_prefix = "Lifetime range (ns)"
            return
        if output_type == "Phase":
            self.histogram_widget.xlabel = "Phase (rad)"
            self.histogram_widget._range_label_prefix = "Phase range (rad)"
            return
        self.histogram_widget.xlabel = "Modulation"
        self.histogram_widget._range_label_prefix = "Modulation range"

    def _on_output_mode_changed(self, mode: str):
        is_lifetime_mode = mode == "Lifetime"
        self.lifetime_type_combobox.setEnabled(is_lifetime_mode)
        self._sync_mode_widgets()
        self._update_calculate_button_text()
        if mode == "Phase":
            self.colormap_combobox.setCurrentText(self._phase_colormap_name)
        elif mode == "Modulation":
            self.colormap_combobox.setCurrentText(
                self._modulation_colormap_name
            )
        output_type = self._get_selected_output_type()
        self.current_output_type = output_type
        self._set_frequency_input_enabled(
            self._output_requires_frequency(output_type)
        )
        self._configure_histogram_labels_for_output(output_type)
        self.outputTypeChanged.emit(output_type)
        if self._output_requires_frequency(output_type):
            self._clear_2d_coloring()
        elif (
            self.current_metric_data is not None
            and self.apply_2d_colormap_checkbox.isChecked()
        ):
            self._apply_histogram_coloring(output_type)
        else:
            self._clear_2d_coloring()
        if not self._updating_settings:
            self._update_lifetime_setting_in_metadata(
                'output_type', output_type
            )

    def _on_colormap_combobox_changed(self, name: str):
        self.custom_color_button.setVisible(name == "Select color...")
        output_type = self._get_selected_output_type()

        if name == "Select color..." and not hasattr(self, "_custom_color"):
            # Set a default initial color if none exists
            self._set_custom_color(QColor(255, 0, 0))  # Default to red

        if output_type == "Phase":
            self._phase_colormap_name = name
        elif output_type == "Modulation":
            self._modulation_colormap_name = name

        if (
            output_type in {"Phase", "Modulation"}
            and self.apply_2d_colormap_checkbox.isChecked()
        ):
            actual_cmap_to_apply = self._resolve_layer_colormap(name)

            for layer in self.metric_layers:
                if layer in self.viewer.layers:
                    layer.colormap = actual_cmap_to_apply
            self._apply_histogram_coloring(output_type)

    def _resolve_layer_colormap(self, cmap_name: str):
        """Return a napari-compatible colormap value for image layers."""
        if not hasattr(self, "_custom_color"):
            self._set_custom_color(QColor(255, 0, 0))

        resolved = resolve_napari_layer_colormap(
            cmap_name,
            custom_color=self._custom_color,
            sentinel="Select color...",
        )
        return cmap_name if resolved is None else resolved

    def _on_apply_2d_colormap_checkbox_changed(self, checked):
        output_type = self._get_selected_output_type()
        if output_type not in {"Phase", "Modulation"}:
            self._clear_2d_coloring()
            return
        if checked:
            self._apply_histogram_coloring(output_type)
        else:
            self._clear_2d_coloring()

    def _clear_2d_coloring(self):
        self._remove_overlay()
        pw = self.parent_widget
        if pw is not None:
            self._set_histogram_density_visible(pw, True)
            if getattr(pw, 'plot_type', None) == 'SCATTER':
                pw.refresh_current_plot()
            elif getattr(pw, 'plot_type', None) == 'CONTOUR':
                contour_collections = getattr(pw, '_contour_collections', [])
                for cs in contour_collections:
                    if hasattr(cs, 'collections') and len(cs.collections) > 0:
                        for col in cs.collections:
                            col.set_visible(True)
                    elif hasattr(cs, 'set_visible'):
                        cs.set_visible(True)
            pw.canvas_widget.figure.canvas.draw_idle()

    def _get_phasor_mapping_settings(self, layer, create: bool = False):
        """Return mapping settings, migrating legacy metadata when needed."""
        if 'settings' not in layer.metadata:
            if not create:
                return None
            layer.metadata['settings'] = {}

        settings_container = layer.metadata['settings']
        mapping_settings = settings_container.get('phasor_mapping')
        legacy_settings = settings_container.get('lifetime')

        if mapping_settings is None and legacy_settings is not None:
            mapping_settings = legacy_settings.copy()
            settings_container['phasor_mapping'] = mapping_settings

        if mapping_settings is None and create:
            mapping_settings = self._get_default_lifetime_settings()
            settings_container['phasor_mapping'] = mapping_settings

        # Keep legacy key populated for backward compatibility.
        if mapping_settings is not None:
            settings_container['lifetime'] = mapping_settings

        return mapping_settings

    def _update_lifetime_setting_in_metadata(self, key, value):
        """Update a specific lifetime setting in the current layer's metadata."""
        if self._updating_settings:
            return

        layer_name = self.parent_widget.get_primary_layer_name()
        if layer_name:
            layer = self.viewer.layers[layer_name]
            mapping_settings = self._get_phasor_mapping_settings(
                layer, create=True
            )
            mapping_settings[key] = value
            if key == 'output_type':
                if value in {
                    "Apparent Phase Lifetime",
                    "Apparent Modulation Lifetime",
                    "Normal Lifetime",
                }:
                    mapping_settings['lifetime_type'] = value
            elif key == 'range_min':
                mapping_settings['lifetime_range_min'] = value
            elif key == 'range_max':
                mapping_settings['lifetime_range_max'] = value

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

        settings = self._get_phasor_mapping_settings(layer, create=False)
        if settings is None:
            self._updating_settings = True
            try:
                self.output_mode_combobox.setCurrentText('Lifetime')
                self.lifetime_type_combobox.setCurrentText(
                    'Apparent Phase Lifetime'
                )
                self._configure_histogram_labels_for_output(
                    'Apparent Phase Lifetime'
                )
                self.lifetime_range_slider.setValue((0, 100))
                self.lifetime_min_edit.setText('0.0')
                self.lifetime_max_edit.setText('100.0')
                self.lifetime_range_label.setText(
                    f'{self.histogram_widget._range_label_prefix}: 0.0 - 100.0'
                )
                self._set_frequency_input_enabled(True)
                self._sync_mode_widgets()
                self._clear_2d_coloring()
                self.histogram_widget.hide()
            finally:
                self._updating_settings = False
            return

        self._updating_settings = True
        try:
            output_type = settings.get('output_type') or settings.get(
                'lifetime_type',
                'Apparent Phase Lifetime',
            )
            if output_type:
                if output_type in {
                    "Apparent Phase Lifetime",
                    "Apparent Modulation Lifetime",
                    "Normal Lifetime",
                }:
                    self.output_mode_combobox.setCurrentText("Lifetime")
                    self.lifetime_type_combobox.setCurrentText(output_type)
                else:
                    self.output_mode_combobox.setCurrentText(output_type)
            self._sync_mode_widgets()
            self._set_frequency_input_enabled(
                self._output_requires_frequency(
                    self._get_selected_output_type()
                )
            )

        finally:
            self._updating_settings = False

    def _on_frequency_changed(self):
        """Handle frequency input changes."""
        frequency_text = self.frequency_input.text().strip()
        output_type = self._get_selected_output_type()

        if not self._updating_settings:
            if not self._output_requires_frequency(output_type):
                return
            if frequency_text:
                try:
                    self.frequency = float(frequency_text)
                    self.calculate_output_data()
                    self._update_lifetime_range_slider()
                    self.create_output_layers()
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

    def _set_frequency_input_enabled(self, enabled: bool):
        self.frequency_input.setEnabled(enabled)

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
        output_type = self._get_selected_output_type()

        if self.current_metric_data_original is not None:
            self.current_metric_data = np.clip(
                self.current_metric_data_original, min_lifetime, max_lifetime
            )
            self.lifetime_data = self.current_metric_data

        selected_layers = self.parent_widget.get_selected_layers()

        self._updating_contrast_limits = True
        self._updating_linked_layers = True
        try:
            for layer in selected_layers:
                derived_data = layer.metadata.get('derived_data', {})
                if output_type not in derived_data:
                    continue

                output_data_dict = derived_data[output_type]
                if self.parent_widget.harmonic not in output_data_dict:
                    continue

                output_values = output_data_dict[self.parent_widget.harmonic]
                clipped_lifetime = np.clip(
                    output_values, min_lifetime, max_lifetime
                )

                lifetime_layer_name = f"{output_type}: {layer.name}"

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
                'range_min', min_lifetime
            )
            self._update_lifetime_setting_in_metadata(
                'range_max', max_lifetime
            )

        self._apply_lifetime_range_change(min_val, max_val)

    def calculate_output_data(self):
        """Calculate selected output (lifetime/phase/modulation) for all selected layers."""
        if not self.parent_widget.has_phasor_data():
            return

        output_type = self._get_selected_output_type()
        self.current_output_type = output_type

        frequency_text = self.frequency_input.text().strip()
        if self._output_requires_frequency(output_type) and not frequency_text:
            show_warning("Enter frequency")
            return

        base_frequency = None
        if self._output_requires_frequency(output_type):
            base_frequency = float(frequency_text)
            self.frequency = base_frequency

        selected_layers = self.parent_widget.get_selected_layers()
        if not selected_layers:
            return

        all_output_data = []
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

            if self._output_requires_frequency(output_type):
                effective_frequency = (
                    base_frequency * self.parent_widget.harmonic
                )
                with np.errstate(divide='ignore', invalid='ignore'):
                    if output_type == "Normal Lifetime":
                        output_values = phasor_to_normal_lifetime(
                            real, imag, frequency=effective_frequency
                        )
                    else:
                        phase_lifetime, modulation_lifetime = (
                            phasor_to_apparent_lifetime(
                                real, imag, frequency=effective_frequency
                            )
                        )
                        if output_type == "Apparent Phase Lifetime":
                            output_values = np.clip(
                                phase_lifetime, a_min=0, a_max=None
                            )
                        else:
                            output_values = np.clip(
                                modulation_lifetime, a_min=0, a_max=None
                            )
                with np.errstate(invalid='ignore'):
                    output_values[output_values < 0] = 0
            else:
                with np.errstate(divide='ignore', invalid='ignore'):
                    phase_values, modulation_values = phasor_to_polar(
                        real, imag
                    )
                output_values = (
                    phase_values
                    if output_type == "Phase"
                    else modulation_values
                )

            if 'derived_data' not in layer.metadata:
                layer.metadata['derived_data'] = {}
            if output_type not in layer.metadata['derived_data']:
                layer.metadata['derived_data'][output_type] = {}
            layer.metadata['derived_data'][output_type][
                self.parent_widget.harmonic
            ] = output_values

            all_output_data.append(output_values)
            per_layer_data[layer.name] = output_values

        if not all_output_data:
            return

        merged_output = np.concatenate(
            [data.flatten() for data in all_output_data]
        )
        self.current_metric_data_original = merged_output
        self.current_metric_data = self.current_metric_data_original.copy()
        self.per_layer_metric_data_original = {
            k: v.copy() for k, v in per_layer_data.items()
        }
        self.per_layer_metric_data = {
            k: v.copy() for k, v in per_layer_data.items()
        }

        if self._output_requires_frequency(output_type):
            self.lifetime_data_original = self.current_metric_data_original
            self.lifetime_data = self.current_metric_data
            self.per_layer_lifetime_data_original = (
                self.per_layer_metric_data_original
            )
            self.per_layer_lifetime_data = self.per_layer_metric_data

    def calculate_lifetimes(self):
        """Backward-compatible alias for unified output calculation."""
        self.calculate_output_data()
        self._update_lifetime_range_slider()

    def _update_lifetime_range_slider(self):
        """Update the lifetime range slider based on the calculated lifetime data."""
        if (
            self.current_metric_data_original is None
            and self.lifetime_data_original is not None
        ):
            self.current_metric_data_original = self.lifetime_data_original
            self.current_metric_data = self.lifetime_data
            self.per_layer_metric_data = self.per_layer_lifetime_data
            self.per_layer_metric_data_original = (
                self.per_layer_lifetime_data_original
            )

        if self.current_metric_data_original is None:
            return
        output_type = self._get_selected_output_type()
        if (
            self._output_requires_frequency(output_type)
            and self.frequency is None
        ):
            return

        effective_frequency = None
        if self._output_requires_frequency(output_type):
            effective_frequency = self.frequency * self.parent_widget.harmonic

        flattened_data = self.current_metric_data_original.flatten()
        valid_data = flattened_data[
            ~np.isnan(flattened_data) & np.isfinite(flattened_data)
        ]
        if self._output_requires_frequency(output_type):
            valid_data = valid_data[valid_data > 0]

        self._configure_histogram_labels_for_output(output_type)

        if len(valid_data) == 0:
            self.min_lifetime = 0.0
            self.max_lifetime = (
                10.0 if self._output_requires_frequency(output_type) else 1.0
            )
            min_slider_val = 0
            max_slider_val = int(
                self.max_lifetime * self.lifetime_range_factor
            )
        else:
            self.min_lifetime = np.min(valid_data)
            self.max_lifetime = np.max(valid_data)

            if (
                self._output_requires_frequency(output_type)
                and effective_frequency is not None
                and (
                    not np.isfinite(self.min_lifetime)
                    or not np.isfinite(self.max_lifetime)
                    or self.max_lifetime > (2e3 / effective_frequency)
                    or self.min_lifetime < 0
                )
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
            f"{self.histogram_widget._range_label_prefix}: {self.min_lifetime:.2f} - {self.max_lifetime:.2f}"
        )

        self.lifetime_min_edit.setText(f"{self.min_lifetime:.2f}")
        self.lifetime_max_edit.setText(f"{self.max_lifetime:.2f}")

    def plot_lifetime_histogram(self):
        """Plot the histogram of the merged lifetime data from all selected layers."""
        if self.current_metric_data is None:
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

        output_type = self._get_selected_output_type()
        if output_type in {"Phase", "Modulation"}:
            self._apply_histogram_coloring(output_type)
        if self.per_layer_metric_data and len(self.per_layer_metric_data) > 1:
            self.histogram_widget.update_multi_data(self.per_layer_metric_data)
        else:
            self.histogram_widget.update_data(self.current_metric_data)

    def create_output_layers(self):
        """Create or update output layers for all selected layers."""
        selected_layers = self.parent_widget.get_selected_layers()
        if not selected_layers:
            return

        output_type = self._get_selected_output_type()
        if output_type in {"Phase", "Modulation"}:
            cmap_name = self._resolve_layer_colormap(
                self.colormap_combobox.currentText()
            )
        else:
            cmap_name = self._get_output_colormap_name(output_type)

        for layer in self.metric_layers:
            if layer in self.viewer.layers:
                with contextlib.suppress(
                    AttributeError, RuntimeError, TypeError, ValueError
                ):
                    layer.events.colormap.disconnect(self._on_colormap_changed)
                    layer.events.contrast_limits.disconnect(
                        self._on_colormap_changed
                    )
        self.metric_layers = []
        self.lifetime_layers = []
        self.lifetime_layer = None

        for layer in selected_layers:
            derived_data = layer.metadata.get('derived_data', {})
            if output_type not in derived_data:
                continue

            output_data_dict = derived_data[output_type]
            if self.parent_widget.harmonic not in output_data_dict:
                continue

            output_values = output_data_dict[self.parent_widget.harmonic]

            output_layer_name = f"{output_type}: {layer.name}"

            min_val, max_val = self.lifetime_range_slider.value()
            min_lifetime = min_val / self.lifetime_range_factor
            max_lifetime = max_val / self.lifetime_range_factor
            clipped_output = np.clip(output_values, min_lifetime, max_lifetime)

            selected_output_layer = Image(
                clipped_output,
                name=output_layer_name,
                scale=layer.scale,
                colormap=cmap_name,
                contrast_limits=[min_lifetime, max_lifetime],
            )

            if output_layer_name in self.viewer.layers:
                self.viewer.layers.remove(
                    self.viewer.layers[output_layer_name]
                )

            output_layer = self.viewer.add_layer(selected_output_layer)

            self.metric_layers.append(output_layer)
            self.lifetime_layers.append(output_layer)
            output_layer.events.colormap.connect(self._on_colormap_changed)
            output_layer.events.contrast_limits.connect(
                self._on_colormap_changed
            )

            if self.lifetime_layer is None:
                self.lifetime_layer = output_layer
                self.lifetime_colormap = output_layer.colormap.colors
                self.colormap_contrast_limits = output_layer.contrast_limits

    def create_lifetime_layer(self):
        """Backward-compatible alias for output layer creation."""
        self.create_output_layers()

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

        output_type = self._get_selected_output_type()
        if output_type in {"Phase", "Modulation"}:
            cmap_name = new_colormap.name
            self.colormap_combobox.blockSignals(True)
            try:
                idx = self.colormap_combobox.findText(cmap_name)
                if idx >= 0:
                    self.colormap_combobox.setCurrentIndex(idx)
            finally:
                self.colormap_combobox.blockSignals(False)
            if output_type == "Phase":
                self._phase_colormap_name = cmap_name
            elif output_type == "Modulation":
                self._modulation_colormap_name = cmap_name

        # Update all other lifetime layers to match
        self._updating_linked_layers = True
        try:
            for layer in self.metric_layers:
                if layer != source_layer and layer in self.viewer.layers:
                    layer.colormap = new_colormap
                    layer.contrast_limits = new_contrast_limits
        finally:
            self._updating_linked_layers = False

        self.histogram_widget.update_colormap(
            colormap_colors=self.lifetime_colormap,
            contrast_limits=self.colormap_contrast_limits,
        )

        if (
            output_type in {"Phase", "Modulation"}
            and self.apply_2d_colormap_checkbox.isChecked()
        ):
            self._apply_histogram_coloring(output_type)

    def _on_image_layer_changed(self):
        """Callback whenever the image layer with phasor features changes.

        This only restores UI state from metadata - it does NOT run calculations.
        User must click "Calculate" button to run lifetime analysis.
        """
        layer_name = self.parent_widget.get_primary_layer_name()
        if layer_name:
            self._restore_lifetime_settings_from_metadata()
            self._sync_mode_widgets()
            self._set_frequency_input_enabled(
                self._output_requires_frequency(
                    self._get_selected_output_type()
                )
            )
            self._clear_2d_coloring()
            # Don't auto-calculate - just restore UI state
            # User must click "Calculate" to run analysis
            self.histogram_widget.hide()
        else:
            # Disconnect events from all lifetime layers
            for layer in self.metric_layers:
                if layer in self.viewer.layers:
                    with contextlib.suppress(Exception):
                        layer.events.colormap.disconnect(
                            self._on_colormap_changed
                        )
                        layer.events.contrast_limits.disconnect(
                            self._on_colormap_changed
                        )

            self.lifetime_data = None
            self.lifetime_data_original = None
            self.current_metric_data = None
            self.current_metric_data_original = None
            self.per_layer_lifetime_data = {}
            self.per_layer_lifetime_data_original = {}
            self.per_layer_metric_data = {}
            self.per_layer_metric_data_original = {}
            self.lifetime_layer = None
            self.lifetime_layers = []
            self.metric_layers = []

            self._clear_2d_coloring()
            self.histogram_widget.clear()

    def _on_calculate_lifetime_clicked(self):
        """Callback when Calculate button is clicked.

        Runs the lifetime calculation and creates/updates lifetime layers.
        """
        selected_layers = self.parent_widget.get_selected_layers()
        if not selected_layers:
            show_warning("Select at least one layer")
            return

        output_type = self._get_selected_output_type()
        frequency = self.frequency_input.text().strip()
        if self._output_requires_frequency(output_type) and frequency == "":
            show_warning("Enter frequency")
            return

        # Run the calculation
        self.calculate_output_data()
        self._update_lifetime_range_slider()
        self.create_output_layers()

        self._restore_lifetime_range_from_metadata()
        self._on_lifetime_range_changed(self.lifetime_range_slider.value())

        if self.current_metric_data is not None:
            self.plot_lifetime_histogram()

        if (
            output_type in {"Phase", "Modulation"}
            and self.apply_2d_colormap_checkbox.isChecked()
        ):
            self._apply_histogram_coloring(output_type)
        else:
            self._clear_2d_coloring()

        # Update frequency in metadata for frequency-dependent outputs
        if not self._updating_settings and self._output_requires_frequency(
            output_type
        ):
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
        output_type = self._get_selected_output_type()
        self.current_output_type = output_type
        self._update_calculate_button_text()
        self._set_frequency_input_enabled(
            self._output_requires_frequency(output_type)
        )
        self._configure_histogram_labels_for_output(output_type)
        self.outputTypeChanged.emit(output_type)
        if not self._updating_settings:
            self._update_lifetime_setting_in_metadata('lifetime_type', text)
            self._update_lifetime_setting_in_metadata(
                'output_type', output_type
            )
        # Don't auto-calculate - user must click Calculate

    def _restore_lifetime_range_from_metadata(self):
        """Restore lifetime range from metadata after calculation."""
        layer_name = self.parent_widget.get_primary_layer_name()
        if not layer_name:
            return

        layer = self.viewer.layers[layer_name]
        settings = self._get_phasor_mapping_settings(layer, create=False)
        if settings is not None:

            min_key = (
                'range_min'
                if 'range_min' in settings
                else 'lifetime_range_min'
            )
            max_key = (
                'range_max'
                if 'range_max' in settings
                else 'lifetime_range_max'
            )

            if (
                min_key in settings
                and max_key in settings
                and settings[min_key] is not None
                and settings[max_key] is not None
            ):

                min_val = settings[min_key]
                max_val = settings[max_key]

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
                            f"{self.histogram_widget._range_label_prefix}: {min_val:.2f} - {max_val:.2f}"
                        )
                    finally:
                        self._updating_settings = False

                    self._apply_lifetime_range_change(min_slider, max_slider)

    def _apply_lifetime_range_change(self, min_slider, max_slider):
        """Apply lifetime range change for histogram without updating layers (layers updated in _on_lifetime_range_changed)."""
        min_lifetime = min_slider / self.lifetime_range_factor
        max_lifetime = max_slider / self.lifetime_range_factor

        # Only update the histogram data (merged/flattened), not the individual layers
        if self.current_metric_data_original is not None:
            self.current_metric_data = np.clip(
                self.current_metric_data_original,
                min_lifetime,
                max_lifetime,
            )
            self.lifetime_data = self.current_metric_data
            # Also clip per-layer data for multi-layer histogram modes
            for name, orig in self.per_layer_metric_data_original.items():
                self.per_layer_metric_data[name] = np.clip(
                    orig, min_lifetime, max_lifetime
                )
            self.per_layer_lifetime_data = self.per_layer_metric_data
            self.plot_lifetime_histogram()

    @staticmethod
    def _set_histogram_density_visible(pw, visible: bool):
        """Show or hide the biaplotter histogram density image."""
        hist_artist = pw.canvas_widget.artists.get("HISTOGRAM2D")
        if hist_artist is None:
            return

        if (
            visible
            and getattr(pw, 'plot_type', 'HISTOGRAM2D') != 'HISTOGRAM2D'
        ):
            return

        img = hist_artist._mpl_artists.get("histogram_image")
        if img is not None:
            img.set_visible(visible)

    def _remove_overlay(self):
        if getattr(self, '_overlay_imshow', None) is not None:
            with contextlib.suppress(ValueError, AttributeError):
                self._overlay_imshow.remove()
            self._overlay_imshow = None
        if getattr(self, '_overlay_clip_patch', None) is not None:
            with contextlib.suppress(ValueError, AttributeError):
                self._overlay_clip_patch.remove()
            self._overlay_clip_patch = None

    def _get_clim_from_metric_layers(self):
        for layer in self.metric_layers:
            if layer in self.viewer.layers:
                with contextlib.suppress(
                    AttributeError, TypeError, ValueError
                ):
                    vmin, vmax = layer.contrast_limits
                    return float(vmin), float(vmax)
        return None, None

    def _apply_histogram_coloring(self, output_type: str):
        if output_type not in {"Phase", "Modulation"}:
            return
        pw = self.parent_widget
        if pw is None:
            return

        features = pw.get_merged_features()
        if features is None:
            return
        g_flat, s_flat = features

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            phase, modulation = phasor_to_polar(g_flat, s_flat)

        values = phase if output_type == "Phase" else modulation

        cmap_name = self.colormap_combobox.currentText()
        if cmap_name == "Select color..." and hasattr(self, "_custom_color"):
            cmap = create_mpl_colormap_from_qcolor(self._custom_color)
        else:
            cmap = resolve_colormap_by_name(cmap_name)
            if cmap is None:
                cmap = resolve_colormap_by_name("viridis")
        vmin, vmax = self._get_clim_from_metric_layers()
        if vmin is None or vmax is None:
            with np.errstate(invalid='ignore'):
                vmin = float(np.nanmin(values))
                vmax = float(np.nanmax(values))

        if pw.plot_type == 'SCATTER':
            self._remove_overlay()
            scatter_artist = pw.canvas_widget.artists.get("SCATTER")
            if scatter_artist is not None:
                sc = scatter_artist._mpl_artists.get("scatter")
                if sc is not None:
                    sc.set_array(values)
                    sc.set_cmap(cmap)
                    sc.set_clim(vmin, vmax)
                    pw.canvas_widget.figure.canvas.draw_idle()
            return

        if pw.plot_type == 'CONTOUR':
            ax = pw.canvas_widget.axes
            range_xlim = ax.get_xlim()
            range_ylim = ax.get_ylim()

            x_fine = np.linspace(range_xlim[0], range_xlim[1], 500)
            y_fine = np.linspace(range_ylim[0], range_ylim[1], 500)
            X, Y = np.meshgrid(x_fine, y_fine)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                p_grid, m_grid = phasor_to_polar(X, Y)

            stat_display = p_grid if output_type == "Phase" else m_grid
            extent = [
                range_xlim[0],
                range_xlim[1],
                range_ylim[0],
                range_ylim[1],
            ]

            import matplotlib.patches as mpatches
            import matplotlib.path as mpath

            # Get paths and hide contours first
            contour_collections = getattr(pw, '_contour_collections', [])
            min_paths = []
            if contour_collections:
                for cs in contour_collections:
                    if hasattr(cs, 'collections') and len(cs.collections) > 0:
                        # Matplotlib < 3.8: collections is a list of LineCollections per level
                        for col in cs.collections:
                            for p in col.get_paths():
                                if (
                                    p.vertices is not None
                                    and len(p.vertices) > 0
                                ):
                                    vertices = p.vertices
                                    codes = p.codes
                                    if codes is None:
                                        codes = np.full(
                                            len(vertices), mpath.Path.LINETO
                                        )
                                        codes[0] = mpath.Path.MOVETO
                                    min_paths.append((vertices, codes))
                            col.set_visible(False)
                    elif hasattr(cs, 'get_paths'):
                        # Matplotlib >= 3.8: get_paths() returns a list of Paths, one per level
                        for p in cs.get_paths():
                            if p.vertices is not None and len(p.vertices) > 0:
                                vertices = p.vertices
                                codes = p.codes
                                if codes is None:
                                    codes = np.full(
                                        len(vertices), mpath.Path.LINETO
                                    )
                                    codes[0] = mpath.Path.MOVETO
                                min_paths.append((vertices, codes))
                        cs.set_visible(False)

            if not min_paths:
                # If no contours found or they don't have paths, don't show the mesh overlay
                self._remove_overlay()
                self._set_histogram_density_visible(
                    pw, pw.plot_type == 'HISTOGRAM2D'
                )
                pw.canvas_widget.figure.canvas.draw_idle()
                return

            self._remove_overlay()
            self._set_histogram_density_visible(pw, False)

            self._overlay_imshow = ax.imshow(
                stat_display,
                extent=extent,
                origin="lower",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation="bilinear",
                zorder=1.5,
                alpha=1.0,
                aspect="auto",
            )

            all_v = np.concatenate([v for v, c in min_paths])
            all_c = np.concatenate([c for v, c in min_paths])
            compound_path = mpath.Path(all_v, all_c)
            patch = mpatches.PathPatch(
                compound_path,
                transform=ax.transData,
                facecolor='none',
                edgecolor='none',
            )
            ax.add_patch(patch)
            self._overlay_imshow.set_clip_path(patch)
            self._overlay_clip_patch = patch

            ax.set_aspect(1, adjustable="box")
            pw.canvas_widget.figure.canvas.draw_idle()
            return

        hist_artist = pw.canvas_widget.artists.get("HISTOGRAM2D")
        if hist_artist is None or hist_artist.histogram is None:
            return
        H, x_edges, y_edges = hist_artist.histogram

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            stat, _, _, _ = binned_statistic_2d(
                g_flat,
                s_flat,
                values,
                statistic="median",
                bins=[x_edges, y_edges],
            )

        mask = np.isnan(H) | (H <= 0)
        zorder = 3
        if pw.plot_type == 'CONTOUR':
            zorder = 1.5
            contour_collections = getattr(pw, '_contour_collections', [])
            if contour_collections:
                min_levels = []
                for cs in contour_collections:
                    if hasattr(cs, 'levels') and len(cs.levels) > 0:
                        min_levels.append(cs.levels[0])
                if min_levels:
                    lowest_level = min(min_levels)
                    mask = lowest_level > H

        stat[mask] = np.nan
        stat_display = stat.T
        if np.all(np.isnan(stat_display)):
            return

        # Ensure vmin, vmax match the calculated stat range if none was available
        with np.errstate(invalid='ignore'):
            if (
                vmin is None
                or vmax is None
                or not np.isfinite(vmin)
                or not np.isfinite(vmax)
            ):
                vmin = float(np.nanmin(stat_display))
                vmax = float(np.nanmax(stat_display))

        ax = pw.canvas_widget.axes
        extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]

        self._remove_overlay()
        self._set_histogram_density_visible(pw, False)

        self._overlay_imshow = ax.imshow(
            stat_display,
            extent=extent,
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            zorder=zorder,
            alpha=1.0,
            aspect="auto",
        )
        ax.set_aspect(1, adjustable="box")
        pw.canvas_widget.figure.canvas.draw_idle()

    def reapply_if_active(self):
        if self._coloring_paused_by_tab:
            self._clear_2d_coloring()
            return
        output_type = self._get_selected_output_type()
        if (
            output_type in {"Phase", "Modulation"}
            and self.apply_2d_colormap_checkbox.isChecked()
        ):
            self._apply_histogram_coloring(output_type)
        else:
            self._clear_2d_coloring()

    def on_tab_visibility_changed(self, is_visible: bool):
        self._coloring_paused_by_tab = not is_visible
        if not is_visible:
            self._clear_2d_coloring()
            return
        self.reapply_if_active()

    def closeEvent(self, event):
        """Clean up signal connections before closing."""
        # Disconnect all lifetime layer events
        for layer in self.metric_layers:
            with contextlib.suppress(ValueError, AttributeError):
                layer.events.colormap.disconnect(self._on_colormap_changed)
            with contextlib.suppress(ValueError, AttributeError):
                layer.events.contrast_limits.disconnect(
                    self._on_colormap_changed
                )

        event.accept()
