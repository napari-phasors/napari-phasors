import contextlib
import warnings
from typing import TYPE_CHECKING

import matplotlib.cm as cm
import numpy as np
from matplotlib.colors import Normalize
from napari.layers import Image
from napari.utils.notifications import show_error, show_warning
from phasorpy.lifetime import (
    phasor_to_apparent_lifetime,
    phasor_to_normal_lifetime,
)
from phasorpy.phasor import phasor_to_polar
from qtpy.QtCore import Qt, QTimer, Signal
from qtpy.QtGui import QColor, QDoubleValidator
from qtpy.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from scipy.ndimage import gaussian_filter
from scipy.stats import binned_statistic_2d
from superqt import QRangeSlider, QToggleSwitch

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
    """Signal emitted with the new output type name when the mapping output changes."""

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
        self._mesh_overlay_imshow = None
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
        self.phase_range_factor = 100
        self.modulation_range_factor = 100
        self._axes_limit_callback_cids = []
        self._mesh_axes_update_timer = QTimer(self)
        self._mesh_axes_update_timer.setSingleShot(True)
        self._mesh_axes_update_timer.setInterval(75)
        self._mesh_axes_update_timer.timeout.connect(
            self._apply_mesh_after_axes_change
        )
        self._mesh_grid_cache = {}
        self._mesh_grid_cache_order = []
        self._mesh_grid_cache_max_entries = 8
        self._mesh_alpha_cache = {}
        self._mesh_alpha_cache_order = []
        self._mesh_alpha_cache_max_entries = 8

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
        self.histogram_widget.update_data(np.array([]))

        # Mesh Overlay Settings (at the end of the tab)
        self.mesh_overlay_group = QWidget()
        mesh_overlay_group_layout = QVBoxLayout(self.mesh_overlay_group)
        mesh_overlay_group_layout.setContentsMargins(0, 0, 0, 0)

        # Toggle for background mesh
        self.mesh_overlay_checkbox = QToggleSwitch("Show mesh overlay")
        self.mesh_overlay_checkbox.onColor = QColor("#27ae60")
        self.mesh_overlay_checkbox.setToolTip(
            "Show a full phase/modulation mesh behind the phasor data"
        )
        mesh_overlay_group_layout.addWidget(self.mesh_overlay_checkbox)

        # Toggle to clip mesh to semicircle
        self.mesh_clip_semicircle_checkbox = QToggleSwitch(
            "Clip mesh to semicircle"
        )
        self.mesh_clip_semicircle_checkbox.onColor = QColor("#27ae60")
        self.mesh_clip_semicircle_checkbox.setToolTip(
            "Only show the background mesh inside the universal semicircle"
        )
        self.mesh_clip_semicircle_checkbox.setVisible(False)

        # Toggle for colorbar
        self.mesh_colorbar_checkbox = QToggleSwitch("Show colorbar")
        self.mesh_colorbar_checkbox.onColor = QColor("#27ae60")
        self.mesh_colorbar_checkbox.setToolTip(
            "Show a colorbar for the phase or modulation mesh/plot"
        )
        self.mesh_colorbar_checkbox.setVisible(False)

        # Row for Alpha, Clip Toggle, and Colorbar Toggle
        self.mesh_controls_widget = QWidget()
        mesh_controls_layout = QHBoxLayout(self.mesh_controls_widget)
        mesh_controls_layout.setContentsMargins(0, 0, 0, 0)

        mesh_controls_layout.addWidget(QLabel("Alpha:"))
        self.mesh_alpha_spinbox = QDoubleSpinBox()
        self.mesh_alpha_spinbox.setRange(0.01, 1.0)
        self.mesh_alpha_spinbox.setSingleStep(0.05)
        self.mesh_alpha_spinbox.setDecimals(2)
        self.mesh_alpha_spinbox.setValue(0.45)
        self.mesh_alpha_spinbox.setToolTip(
            "Opacity of the phase/modulation mesh overlay"
        )
        self.mesh_alpha_spinbox.setFixedWidth(60)
        mesh_controls_layout.addWidget(self.mesh_alpha_spinbox)

        mesh_controls_layout.addSpacing(10)
        mesh_controls_layout.addWidget(self.mesh_clip_semicircle_checkbox)
        mesh_controls_layout.addSpacing(10)
        mesh_controls_layout.addWidget(self.mesh_colorbar_checkbox)
        mesh_controls_layout.addStretch(1)

        mesh_overlay_group_layout.addWidget(self.mesh_controls_widget)

        # Phase range controls
        self.phase_range_container = QWidget()
        ph_cnt_layout = QVBoxLayout(self.phase_range_container)
        ph_cnt_layout.setContentsMargins(0, 5, 0, 0)

        ph_row = QHBoxLayout()
        ph_row.addWidget(QLabel("Phase range (rad):"))
        ph_row.addStretch(1)
        self.phase_min_edit = QLineEdit("0.00")
        self.phase_max_edit = QLineEdit("1.60")
        self.phase_min_edit.setValidator(QDoubleValidator())
        self.phase_max_edit.setValidator(QDoubleValidator())
        self.phase_min_edit.setFixedWidth(50)
        self.phase_max_edit.setFixedWidth(50)
        self.phase_min_edit.setAlignment(Qt.AlignCenter)
        self.phase_max_edit.setAlignment(Qt.AlignCenter)
        ph_row.addWidget(self.phase_min_edit)
        ph_row.addWidget(QLabel("to"))
        ph_row.addWidget(self.phase_max_edit)
        ph_cnt_layout.addLayout(ph_row)
        self.phase_range_slider = QRangeSlider(Qt.Orientation.Horizontal)
        self.phase_range_slider.setRange(0, 628)
        self.phase_range_slider.setValue((0, 628))
        ph_cnt_layout.addWidget(self.phase_range_slider)
        mesh_overlay_group_layout.addWidget(self.phase_range_container)

        # Modulation range controls
        self.modulation_range_container = QWidget()
        mod_cnt_layout = QVBoxLayout(self.modulation_range_container)
        mod_cnt_layout.setContentsMargins(0, 5, 0, 0)

        mod_row = QHBoxLayout()
        mod_row.addWidget(QLabel("Modulation range:"))
        mod_row.addStretch(1)
        self.modulation_min_edit = QLineEdit("0.00")
        self.modulation_max_edit = QLineEdit("1.00")
        self.modulation_min_edit.setValidator(QDoubleValidator())
        self.modulation_max_edit.setValidator(QDoubleValidator())
        self.modulation_min_edit.setFixedWidth(50)
        self.modulation_max_edit.setFixedWidth(50)
        self.modulation_min_edit.setAlignment(Qt.AlignCenter)
        self.modulation_max_edit.setAlignment(Qt.AlignCenter)
        mod_row.addWidget(self.modulation_min_edit)
        mod_row.addWidget(QLabel("to"))
        mod_row.addWidget(self.modulation_max_edit)
        mod_cnt_layout.addLayout(mod_row)
        self.modulation_range_slider = QRangeSlider(Qt.Orientation.Horizontal)
        self.modulation_range_slider.setRange(0, 100)
        self.modulation_range_slider.setValue((0, 100))
        mod_cnt_layout.addWidget(self.modulation_range_slider)
        mesh_overlay_group_layout.addWidget(self.modulation_range_container)

        self.main_layout.addWidget(self.mesh_overlay_group)

        # Add Calculate button in its own row (at the bottom of this tab)
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
        self.main_layout.addStretch(1)

        # Connect signals for mesh overlay
        self.mesh_overlay_checkbox.toggled.connect(
            self._on_mesh_overlay_toggled
        )
        self.mesh_clip_semicircle_checkbox.toggled.connect(
            self._on_mesh_clip_toggled
        )
        self.mesh_colorbar_checkbox.toggled.connect(
            self._on_mesh_colorbar_toggled
        )
        self.phase_range_slider.valueChanged.connect(
            self._on_phase_slider_changed
        )
        self.phase_min_edit.editingFinished.connect(
            self._on_phase_edits_changed
        )
        self.phase_max_edit.editingFinished.connect(
            self._on_phase_edits_changed
        )
        self.modulation_range_slider.valueChanged.connect(
            self._on_modulation_slider_changed
        )
        self.modulation_min_edit.editingFinished.connect(
            self._on_modulation_edits_changed
        )
        self.modulation_max_edit.editingFinished.connect(
            self._on_modulation_edits_changed
        )
        self.mesh_alpha_spinbox.valueChanged.connect(
            self._on_mesh_alpha_changed
        )

        self._sync_mode_widgets()
        self._update_calculate_button_text()
        self.update_apply_2d_text()

        # Connect to plot type changes in the parent widget
        if self.parent_widget is not None:
            self.parent_widget.plotter_inputs_widget.plot_type_combobox.currentTextChanged.connect(
                self.update_apply_2d_text
            )
            self.parent_widget.plotter_inputs_widget.semi_circle_checkbox.toggled.connect(
                self._on_plot_geometry_mode_toggled
            )
            self._connect_axes_limit_callbacks()

    def _connect_axes_limit_callbacks(self):
        if self.parent_widget is None:
            return
        axes = getattr(self.parent_widget.canvas_widget, 'axes', None)
        if axes is None:
            return
        if self._axes_limit_callback_cids:
            return
        self._axes_limit_callback_cids = [
            axes.callbacks.connect(
                'xlim_changed', self._on_axes_limits_changed
            ),
            axes.callbacks.connect(
                'ylim_changed', self._on_axes_limits_changed
            ),
        ]

    def _disconnect_axes_limit_callbacks(self):
        if self.parent_widget is None:
            return
        axes = getattr(self.parent_widget.canvas_widget, 'axes', None)
        if axes is None:
            return
        for cid in self._axes_limit_callback_cids:
            with contextlib.suppress(Exception):
                axes.callbacks.disconnect(cid)
        self._axes_limit_callback_cids = []

    def _on_axes_limits_changed(self, _axes):
        if self._coloring_paused_by_tab:
            return
        output_type = self._get_selected_output_type()
        if (
            output_type in {"Phase", "Modulation"}
            and self.mesh_overlay_checkbox.isChecked()
        ):
            self._mesh_axes_update_timer.start()

    def _apply_mesh_after_axes_change(self):
        if self._coloring_paused_by_tab:
            return
        output_type = self._get_selected_output_type()
        if (
            output_type in {"Phase", "Modulation"}
            and self.mesh_overlay_checkbox.isChecked()
        ):
            self._apply_histogram_coloring(output_type)

    def update_apply_2d_text(self):
        """Update the checkbox text based on the current plot type."""
        plot_type = getattr(self.parent_widget, 'plot_type', 'HISTOGRAM2D')
        if plot_type == 'SCATTER':
            suffix = "Scatter plot"
        elif plot_type == 'CONTOUR':
            suffix = "Contour plot"
        elif plot_type == 'NONE':
            suffix = "Plot"
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
        pw = self.parent_widget
        self.lifetime_output_widget.setVisible(is_lifetime_mode)
        self.frequency_widget.setVisible(is_lifetime_mode)
        self.colormap_widget.setVisible(not is_lifetime_mode)
        self.coloring_checkbox_widget.setVisible(not is_lifetime_mode)

        self.mesh_overlay_group.setVisible(not is_lifetime_mode)
        if not is_lifetime_mode:
            show_ranges = self.mesh_overlay_checkbox.isChecked()
            is_semi = self._is_semicircle_mode()
            self.mesh_controls_widget.setVisible(show_ranges)
            self.phase_range_container.setVisible(show_ranges)
            self.modulation_range_container.setVisible(show_ranges)
            self.mesh_clip_semicircle_checkbox.setVisible(is_semi)
            self.mesh_colorbar_checkbox.setVisible(True)
            self._update_phase_slider_bounds_from_plot_mode()
        else:
            self.mesh_controls_widget.setVisible(False)
            self.phase_range_container.setVisible(False)
            self.modulation_range_container.setVisible(False)
            if pw is not None:
                pw._remove_mapping_colorbar()

    def _is_semicircle_mode(self) -> bool:
        if self.parent_widget is None:
            return True
        with contextlib.suppress(AttributeError):
            return bool(self.parent_widget.toggle_semi_circle)
        with contextlib.suppress(AttributeError):
            return not bool(
                self.parent_widget.plotter_inputs_widget.semi_circle_checkbox.isChecked()
            )
        return True

    def _phase_max_allowed(self) -> float:
        return 2.0 * np.pi

    def _update_phase_slider_bounds_from_plot_mode(self):
        max_phase = self._phase_max_allowed()
        max_phase_i = int(max_phase * self.phase_range_factor)
        min_phase_i, cur_max_phase_i = self.phase_range_slider.value()
        min_phase_i = max(0, min(min_phase_i, max_phase_i))
        cur_max_phase_i = max(min_phase_i, min(cur_max_phase_i, max_phase_i))

        self._updating_settings = True
        try:
            self.phase_range_slider.setRange(0, max_phase_i)
            self.phase_range_slider.setValue((min_phase_i, cur_max_phase_i))
            self.phase_min_edit.setText(
                f"{min_phase_i / self.phase_range_factor:.2f}"
            )
            self.phase_max_edit.setText(
                f"{cur_max_phase_i / self.phase_range_factor:.2f}"
            )
        finally:
            self._updating_settings = False

    def _initialize_mesh_ranges_from_current_data(self):
        pw = self.parent_widget
        if pw is None:
            return
        features = pw.get_merged_features()
        if features is None:
            return
        g_flat, s_flat = features
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            phase_values, modulation_values = phasor_to_polar(g_flat, s_flat)

        if not self._is_semicircle_mode():
            with np.errstate(invalid='ignore'):
                phase_values = np.mod(phase_values, 2.0 * np.pi)

        phase_max_allowed = self._phase_max_allowed()
        with np.errstate(invalid='ignore'):
            phase_valid = phase_values[np.isfinite(phase_values)]
            modulation_valid = modulation_values[
                np.isfinite(modulation_values)
            ]

        if phase_valid.size > 0:
            phase_min = float(np.nanmin(phase_valid))
            phase_max = float(np.nanmax(phase_valid))
            phase_min = max(0.0, min(phase_min, phase_max_allowed))
            phase_max = max(phase_min, min(phase_max, phase_max_allowed))
        else:
            phase_min, phase_max = 0.0, phase_max_allowed

        if modulation_valid.size > 0:
            mod_min = float(np.nanmin(modulation_valid))
            mod_max = float(np.nanmax(modulation_valid))
            mod_min = max(0.0, mod_min)
            mod_max = max(mod_min, mod_max)
        else:
            mod_min, mod_max = 0.0, 1.0

        phase_max_i = int(phase_max_allowed * self.phase_range_factor)
        phase_min_i = int(phase_min * self.phase_range_factor)
        phase_val_max_i = int(phase_max * self.phase_range_factor)

        mod_slider_max = max(1.0, mod_max)
        mod_slider_max_i = int(mod_slider_max * self.modulation_range_factor)
        mod_min_i = int(mod_min * self.modulation_range_factor)
        mod_max_i = int(mod_max * self.modulation_range_factor)

        phase_min_i = max(0, min(phase_min_i, phase_max_i))
        phase_val_max_i = max(phase_min_i, min(phase_val_max_i, phase_max_i))
        mod_min_i = max(0, min(mod_min_i, mod_slider_max_i))
        mod_max_i = max(mod_min_i, min(mod_max_i, mod_slider_max_i))

        self._updating_settings = True
        try:
            self.phase_range_slider.setRange(0, phase_max_i)
            self.phase_range_slider.setValue((phase_min_i, phase_val_max_i))
            self.phase_min_edit.setText(f"{phase_min:.2f}")
            self.phase_max_edit.setText(
                f"{phase_val_max_i / self.phase_range_factor:.2f}"
            )

            self.modulation_range_slider.setRange(0, mod_slider_max_i)
            self.modulation_range_slider.setValue((mod_min_i, mod_max_i))
            self.modulation_min_edit.setText(f"{mod_min:.2f}")
            self.modulation_max_edit.setText(
                f"{mod_max_i / self.modulation_range_factor:.2f}"
            )
        finally:
            self._updating_settings = False

        self._persist_current_mesh_ranges_to_metadata()

    def _on_plot_geometry_mode_toggled(self, _checked):
        self._update_phase_slider_bounds_from_plot_mode()
        self._sync_mode_widgets()
        if self.mesh_overlay_checkbox.isChecked():
            settings = self._get_current_layer_mapping_settings(create=False)
            if not self._restore_mesh_ranges_from_settings(settings):
                self._initialize_mesh_ranges_from_current_data()
            self._persist_current_mesh_ranges_to_metadata()
        self.reapply_if_active()

    def _refresh_mesh_overlay_if_needed(self):
        output_type = self._get_selected_output_type()
        if (
            output_type in {"Phase", "Modulation"}
            and self.mesh_overlay_checkbox.isChecked()
        ):
            self._apply_histogram_coloring(output_type)

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
            'mesh_overlay_enabled': False,
            'mesh_clip_semicircle_enabled': False,
            'mesh_colorbar_enabled': False,
            'mesh_alpha': 0.45,
            'mesh_phase_min': None,
            'mesh_phase_max': None,
            'mesh_modulation_min': None,
            'mesh_modulation_max': None,
        }

    def _get_current_layer_mapping_settings(self, create: bool = False):
        layer_name = self.parent_widget.get_primary_layer_name()
        if not layer_name:
            return None
        layer = self.viewer.layers[layer_name]
        return self._get_phasor_mapping_settings(layer, create=create)

    def _persist_current_mesh_ranges_to_metadata(self):
        if self._updating_settings:
            return
        phase_min_i, phase_max_i = self.phase_range_slider.value()
        mod_min_i, mod_max_i = self.modulation_range_slider.value()
        self._update_lifetime_setting_in_metadata(
            'mesh_phase_min', phase_min_i / self.phase_range_factor
        )
        self._update_lifetime_setting_in_metadata(
            'mesh_phase_max', phase_max_i / self.phase_range_factor
        )
        self._update_lifetime_setting_in_metadata(
            'mesh_modulation_min', mod_min_i / self.modulation_range_factor
        )
        self._update_lifetime_setting_in_metadata(
            'mesh_modulation_max', mod_max_i / self.modulation_range_factor
        )
        self._update_lifetime_setting_in_metadata(
            'mesh_clip_semicircle_enabled',
            self.mesh_clip_semicircle_checkbox.isChecked(),
        )
        self._update_lifetime_setting_in_metadata(
            'mesh_colorbar_enabled',
            self.mesh_colorbar_checkbox.isChecked(),
        )

    def _restore_mesh_ranges_from_settings(self, settings) -> bool:
        if settings is None:
            return False

        phase_min = settings.get('mesh_phase_min')
        phase_max = settings.get('mesh_phase_max')
        mod_min = settings.get('mesh_modulation_min')
        mod_max = settings.get('mesh_modulation_max')
        clip_semi = settings.get('mesh_clip_semicircle_enabled', False)
        show_colorbar = settings.get('mesh_colorbar_enabled', False)
        if any(v is None for v in (phase_min, phase_max, mod_min, mod_max)):
            return False

        phase_max_allowed = self._phase_max_allowed()
        phase_min = float(np.clip(float(phase_min), 0.0, phase_max_allowed))
        phase_max = float(
            np.clip(float(phase_max), phase_min, phase_max_allowed)
        )
        mod_min = max(0.0, float(mod_min))
        mod_max = max(mod_min, float(mod_max))

        phase_slider_max_i = int(phase_max_allowed * self.phase_range_factor)
        phase_min_i = int(phase_min * self.phase_range_factor)
        phase_max_i = int(phase_max * self.phase_range_factor)

        mod_slider_max = max(1.0, mod_max)
        mod_slider_max_i = int(mod_slider_max * self.modulation_range_factor)
        mod_min_i = int(mod_min * self.modulation_range_factor)
        mod_max_i = int(mod_max * self.modulation_range_factor)

        self.phase_range_slider.setRange(0, phase_slider_max_i)
        self.phase_range_slider.setValue((phase_min_i, phase_max_i))
        self.phase_min_edit.setText(f"{phase_min:.2f}")
        self.phase_max_edit.setText(f"{phase_max:.2f}")

        self.modulation_range_slider.setRange(0, mod_slider_max_i)
        self.modulation_range_slider.setValue((mod_min_i, mod_max_i))
        self.modulation_min_edit.setText(f"{mod_min:.2f}")
        self.modulation_max_edit.setText(f"{mod_max:.2f}")
        self.mesh_clip_semicircle_checkbox.setChecked(bool(clip_semi))
        self.mesh_colorbar_checkbox.setChecked(bool(show_colorbar))
        return True

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
        if output_type not in {"Phase", "Modulation"}:
            self._clear_2d_coloring()
        else:
            can_apply_coloring = (
                self.apply_2d_colormap_checkbox.isChecked()
                or self.mesh_overlay_checkbox.isChecked()
                or self.mesh_colorbar_checkbox.isChecked()
            )
            if can_apply_coloring:
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

        if output_type in {"Phase", "Modulation"} and (
            self.apply_2d_colormap_checkbox.isChecked()
            or self.mesh_overlay_checkbox.isChecked()
            or self.mesh_colorbar_checkbox.isChecked()
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
        if checked or self.mesh_overlay_checkbox.isChecked():
            self._apply_histogram_coloring(output_type)
        else:
            self._clear_2d_coloring()

    def _clear_2d_coloring(self):
        self._remove_overlay()
        pw = self.parent_widget
        if pw is not None:
            pw._remove_mapping_colorbar()
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

    def _restore_plot_coloring_state_without_mesh(self):
        """Restore non-mesh plot rendering while keeping mesh logic independent."""
        pw = self.parent_widget
        if pw is None:
            return
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
                self.mesh_overlay_checkbox.blockSignals(True)
                try:
                    self.mesh_overlay_checkbox.setChecked(False)
                finally:
                    self.mesh_overlay_checkbox.blockSignals(False)
                self.mesh_alpha_spinbox.blockSignals(True)
                try:
                    self.mesh_alpha_spinbox.setValue(0.45)
                finally:
                    self.mesh_alpha_spinbox.blockSignals(False)
                self._sync_mode_widgets()
                self._clear_2d_coloring()
                self.histogram_widget.update_data(np.array([]))
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

            mesh_enabled = bool(settings.get('mesh_overlay_enabled', False))
            mesh_alpha = float(settings.get('mesh_alpha', 0.45))
            mesh_alpha = float(np.clip(mesh_alpha, 0.0, 1.0))

            self.mesh_overlay_checkbox.blockSignals(True)
            try:
                self.mesh_overlay_checkbox.setChecked(mesh_enabled)
            finally:
                self.mesh_overlay_checkbox.blockSignals(False)

            self.mesh_alpha_spinbox.blockSignals(True)
            try:
                self.mesh_alpha_spinbox.setValue(mesh_alpha)
            finally:
                self.mesh_alpha_spinbox.blockSignals(False)

            self._update_phase_slider_bounds_from_plot_mode()
            if not self._restore_mesh_ranges_from_settings(settings):
                self._initialize_mesh_ranges_from_current_data()
            self._sync_mode_widgets()

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
                self.histogram_widget.update_data(np.array([]))

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

        if not self._updating_settings:
            self._update_tab_sliders_from_range(min_lifetime, max_lifetime)

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
                if output_type == "Phase" and not self._is_semicircle_mode():
                    # Full polar mode expects phase in [0, 2pi].
                    with np.errstate(invalid='ignore'):
                        phase_values = np.mod(phase_values, 2.0 * np.pi)
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
            if output_type == "Phase" and not self._is_semicircle_mode():
                # In full polar mode, initialize the display range to 0..2pi
                # so users get the expected 0..360 degree domain.
                self.min_lifetime = 0.0
                self.max_lifetime = 2.0 * np.pi
                min_slider_val = 0
                max_slider_val = int(
                    self.max_lifetime * self.lifetime_range_factor
                )
                self.current_metric_data_original = np.mod(
                    self.current_metric_data_original, 2.0 * np.pi
                )
                self.current_metric_data = (
                    self.current_metric_data_original.copy()
                )
                for name, data in self.per_layer_metric_data_original.items():
                    wrapped = np.mod(data, 2.0 * np.pi)
                    self.per_layer_metric_data_original[name] = wrapped
                    self.per_layer_metric_data[name] = wrapped.copy()
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
        if (
            self.current_metric_data is None
            or not self.parent_widget.has_phasor_data()
            or self.parent_widget.harmonic is None
        ):
            self.histogram_widget.update_data(np.array([]))
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

        if output_type in {"Phase", "Modulation"} and (
            self.apply_2d_colormap_checkbox.isChecked()
            or self.mesh_overlay_checkbox.isChecked()
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
        else:
            self.histogram_widget.update_data(np.array([]))
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
        settings = self._get_current_layer_mapping_settings(create=False)
        if not self._restore_mesh_ranges_from_settings(settings):
            self._initialize_mesh_ranges_from_current_data()

        self._restore_lifetime_range_from_metadata()
        self._on_lifetime_range_changed(self.lifetime_range_slider.value())

        if self.current_metric_data is not None:
            self.plot_lifetime_histogram()

        if output_type in {"Phase", "Modulation"} and (
            self.apply_2d_colormap_checkbox.isChecked()
            or self.mesh_overlay_checkbox.isChecked()
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
        self._remove_plot_overlay()
        self._remove_mesh_overlay()

    def _remove_plot_overlay(self):
        if getattr(self, '_overlay_imshow', None) is not None:
            with contextlib.suppress(ValueError, AttributeError):
                self._overlay_imshow.remove()
            self._overlay_imshow = None
        if getattr(self, '_overlay_clip_patch', None) is not None:
            with contextlib.suppress(ValueError, AttributeError):
                self._overlay_clip_patch.remove()
            self._overlay_clip_patch = None

    def _remove_mesh_overlay(self):
        if getattr(self, '_mesh_overlay_imshow', None) is not None:
            with contextlib.suppress(ValueError, AttributeError):
                self._mesh_overlay_imshow.remove()
            self._mesh_overlay_imshow = None

    def _get_mesh_grid_resolution(self, ax) -> int:
        with contextlib.suppress(Exception):
            bbox = ax.get_window_extent()
            target = int(max(float(bbox.width), float(bbox.height)) * 1.2)
            return int(np.clip(target, 320, 640))
        return 480

    def _make_mesh_grid_cache_key(self, ax, resolution: int):
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        return (
            round(float(x_min), 5),
            round(float(x_max), 5),
            round(float(y_min), 5),
            round(float(y_max), 5),
            bool(self._is_semicircle_mode()),
            int(resolution),
        )

    def _get_mesh_polar_grid(self, ax, resolution: int):
        key = self._make_mesh_grid_cache_key(ax, resolution)
        cached = self._mesh_grid_cache.get(key)
        if cached is not None:
            with contextlib.suppress(ValueError):
                self._mesh_grid_cache_order.remove(key)
            self._mesh_grid_cache_order.append(key)
            return cached

        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        x_fine = np.linspace(x_min, x_max, resolution)
        y_fine = np.linspace(y_min, y_max, resolution)
        X, Y = np.meshgrid(x_fine, y_fine)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            p_grid, m_grid = phasor_to_polar(X, Y)

        if not self._is_semicircle_mode():
            with np.errstate(invalid='ignore'):
                p_grid = np.mod(p_grid, 2.0 * np.pi)

        grid = {
            'p_grid': p_grid,
            'm_grid': m_grid,
            'extent': [x_min, x_max, y_min, y_max],
        }
        self._mesh_grid_cache[key] = grid
        self._mesh_grid_cache_order.append(key)

        while (
            len(self._mesh_grid_cache_order)
            > self._mesh_grid_cache_max_entries
        ):
            stale_key = self._mesh_grid_cache_order.pop(0)
            self._mesh_grid_cache.pop(stale_key, None)

        return grid

    def _get_mesh_alpha_map(self, mesh_mask, alpha_key, mesh_alpha: float):
        cached = self._mesh_alpha_cache.get(alpha_key)
        if cached is not None:
            with contextlib.suppress(ValueError):
                self._mesh_alpha_cache_order.remove(alpha_key)
            self._mesh_alpha_cache_order.append(alpha_key)
            return cached * mesh_alpha

        alpha_base = gaussian_filter(
            (~mesh_mask).astype(float), sigma=1.2, mode="nearest"
        )
        alpha_base = np.clip(alpha_base, 0.0, 1.0)
        self._mesh_alpha_cache[alpha_key] = alpha_base
        self._mesh_alpha_cache_order.append(alpha_key)

        while (
            len(self._mesh_alpha_cache_order)
            > self._mesh_alpha_cache_max_entries
        ):
            stale_key = self._mesh_alpha_cache_order.pop(0)
            self._mesh_alpha_cache.pop(stale_key, None)

        return alpha_base * mesh_alpha

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
        if pw is None or getattr(pw, 'plot_type', 'HISTOGRAM2D') == 'NONE':
            return

        apply_plot_coloring = self.apply_2d_colormap_checkbox.isChecked()
        show_mesh = self.mesh_overlay_checkbox.isChecked()

        if not show_mesh:
            self._remove_mesh_overlay()
        if not apply_plot_coloring:
            self._remove_plot_overlay()

        features = pw.get_merged_features()
        if features is None:
            canvas_widget = getattr(pw, "canvas_widget", None)
            figure = getattr(canvas_widget, "figure", None)
            canvas = getattr(figure, "canvas", None)
            if canvas is not None and hasattr(canvas, "draw_idle"):
                canvas.draw_idle()
            return
        g_flat, s_flat = features

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            phase, modulation = phasor_to_polar(g_flat, s_flat)

        if not self._is_semicircle_mode():
            with np.errstate(invalid='ignore'):
                phase = np.mod(phase, 2.0 * np.pi)

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
            if output_type == "Phase":
                vmin, vmax = self.phase_range_slider.value()
                vmin /= self.phase_range_factor
                vmax /= self.phase_range_factor
            elif output_type == "Modulation":
                vmin, vmax = self.modulation_range_slider.value()
                vmin /= self.modulation_range_factor
                vmax /= self.modulation_range_factor
            else:
                with np.errstate(invalid='ignore'):
                    vmin = float(np.nanmin(values))
                    vmax = float(np.nanmax(values))

        if show_mesh:
            ax = pw.canvas_widget.axes
            resolution = self._get_mesh_grid_resolution(ax)
            mesh_grid = self._get_mesh_polar_grid(ax, resolution)
            p_grid = mesh_grid['p_grid']
            m_grid = mesh_grid['m_grid']

            phase_min_i, phase_max_i = self.phase_range_slider.value()
            mod_min_i, mod_max_i = self.modulation_range_slider.value()
            phase_min = phase_min_i / self.phase_range_factor
            phase_max = phase_max_i / self.phase_range_factor
            mod_min = mod_min_i / self.modulation_range_factor
            mod_max = mod_max_i / self.modulation_range_factor

            phase_ok = (p_grid >= phase_min) & (p_grid <= phase_max)
            mod_ok = (m_grid >= mod_min) & (m_grid <= mod_max)

            mesh_mask = ~(phase_ok & mod_ok)

            if (
                self.mesh_clip_semicircle_checkbox.isChecked()
                and self._is_semicircle_mode()
            ):
                with np.errstate(invalid='ignore'):
                    is_outside_semi = m_grid > np.cos(p_grid)
                    mesh_mask |= is_outside_semi

            stat_display = p_grid if output_type == "Phase" else m_grid
            stat_display = np.ma.array(stat_display, mask=mesh_mask)
            extent = mesh_grid['extent']
            mesh_cmap = cmap.copy()
            mesh_cmap.set_bad((0, 0, 0, 0))
            mesh_alpha = float(self.mesh_alpha_spinbox.value())
            alpha_key = (
                *self._make_mesh_grid_cache_key(ax, resolution),
                int(phase_min_i),
                int(phase_max_i),
                int(mod_min_i),
                int(mod_max_i),
                bool(self.mesh_clip_semicircle_checkbox.isChecked()),
            )
            mesh_alpha_map = self._get_mesh_alpha_map(
                mesh_mask,
                alpha_key,
                mesh_alpha,
            )

            self._remove_mesh_overlay()

            mesh_imshow_kwargs = {
                "extent": extent,
                "origin": "lower",
                "cmap": mesh_cmap,
                "vmin": vmin,
                "vmax": vmax,
                "interpolation": "lanczos",
                "zorder": 0.1,  # Behind all phasor artists
                "alpha": mesh_alpha_map,
                "aspect": "auto",
            }
            try:
                self._mesh_overlay_imshow = ax.imshow(
                    stat_display,
                    interpolation_stage="rgba",
                    **mesh_imshow_kwargs,
                )
            except TypeError:
                self._mesh_overlay_imshow = ax.imshow(
                    stat_display,
                    **mesh_imshow_kwargs,
                )
            ax.set_aspect(1, adjustable="box")
            pw.canvas_widget.figure.canvas.draw_idle()

            if not apply_plot_coloring:
                self._restore_plot_coloring_state_without_mesh()
                if not self.mesh_colorbar_checkbox.isChecked():
                    pw._remove_mapping_colorbar()
                else:
                    self._update_mapping_colorbar(
                        cmap, vmin, vmax, output_type
                    )
                return

        if not apply_plot_coloring:
            self._restore_plot_coloring_state_without_mesh()
            if not self.mesh_colorbar_checkbox.isChecked():
                pw._remove_mapping_colorbar()
            else:
                self._update_mapping_colorbar(cmap, vmin, vmax, output_type)
            return

        # If reaching here, apply_plot_coloring is True
        if self.mesh_colorbar_checkbox.isChecked():
            self._update_mapping_colorbar(cmap, vmin, vmax, output_type)
        else:
            pw._remove_mapping_colorbar()

        if pw.plot_type == 'SCATTER':
            self._remove_plot_overlay()
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
                self._remove_plot_overlay()
                self._set_histogram_density_visible(
                    pw, pw.plot_type == 'HISTOGRAM2D'
                )
                pw.canvas_widget.figure.canvas.draw_idle()
                return

            self._remove_plot_overlay()
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

        self._remove_plot_overlay()
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

        self.plot_lifetime_histogram()

        output_type = self._get_selected_output_type()
        if output_type in {"Phase", "Modulation"} and (
            self.apply_2d_colormap_checkbox.isChecked()
            or self.mesh_overlay_checkbox.isChecked()
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
        self._mesh_axes_update_timer.stop()
        self._disconnect_axes_limit_callbacks()

        # Disconnect all lifetime layer events
        for layer in self.metric_layers:
            with contextlib.suppress(ValueError, AttributeError):
                layer.events.colormap.disconnect(self._on_colormap_changed)
            with contextlib.suppress(ValueError, AttributeError):
                layer.events.contrast_limits.disconnect(
                    self._on_colormap_changed
                )

        event.accept()

    def _on_mesh_overlay_toggled(self, checked):
        """Handle mesh overlay toggle."""
        if self._updating_settings:
            return

        self._sync_mode_widgets()
        if checked:
            settings = self._get_current_layer_mapping_settings(create=False)
            if not self._restore_mesh_ranges_from_settings(settings):
                self._initialize_mesh_ranges_from_current_data()

        self._update_lifetime_setting_in_metadata(
            'mesh_overlay_enabled', bool(checked)
        )

        output_type = self._get_selected_output_type()
        if output_type not in {"Phase", "Modulation"}:
            self._clear_2d_coloring()
            return
        if checked or self.apply_2d_colormap_checkbox.isChecked():
            self._apply_histogram_coloring(output_type)
        else:
            self._clear_2d_coloring()

    def _on_mesh_clip_toggled(self, checked):
        """Handle mesh clipping toggle."""
        if self._updating_settings:
            return

        self._update_lifetime_setting_in_metadata(
            'mesh_clip_semicircle_enabled', bool(checked)
        )
        self._refresh_mesh_overlay_if_needed()

    def _update_mapping_colorbar(self, cmap, vmin, vmax, output_type):
        """Update the plotter colorbar for phase or modulation."""
        pw = self.parent_widget
        if pw is None:
            return
        norm = Normalize(vmin=vmin, vmax=vmax)
        mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
        label = "Phase (rad)" if output_type == "Phase" else "Modulation"
        # Access the private method to update the colorbar in the plotter
        pw._update_mapping_colorbar(mappable=mappable, label=label)

    def _on_mesh_colorbar_toggled(self, checked):
        """Handle mesh colorbar toggle."""
        if self._updating_settings:
            return

        self._update_lifetime_setting_in_metadata(
            'mesh_colorbar_enabled', bool(checked)
        )

        output_type = self._get_selected_output_type()
        if output_type in {"Phase", "Modulation"}:
            if checked:
                self._apply_histogram_coloring(output_type)
            elif self.parent_widget is not None:
                self.parent_widget._remove_mapping_colorbar()

    def _on_mesh_alpha_changed(self, _value):
        """Refresh mesh overlay when alpha changes."""
        if not self._updating_settings:
            self._update_lifetime_setting_in_metadata(
                'mesh_alpha', float(self.mesh_alpha_spinbox.value())
            )
        self._refresh_mesh_overlay_if_needed()

    def _on_phase_slider_changed(self, value):
        """Handle phase range slider change from tab."""
        if self._updating_settings:
            return
        min_v, max_v = value
        min_f = min_v / self.phase_range_factor
        max_f = max_v / self.phase_range_factor
        self.phase_min_edit.setText(f"{min_f:.2f}")
        self.phase_max_edit.setText(f"{max_f:.2f}")

        if self.output_mode_combobox.currentText() == "Phase":
            self._sync_range_to_histogram(min_f, max_f)
        self._persist_current_mesh_ranges_to_metadata()
        self._refresh_mesh_overlay_if_needed()

    def _on_phase_edits_changed(self):
        """Handle phase range line edit change from tab."""
        if self._updating_settings:
            return
        try:
            min_f = float(self.phase_min_edit.text())
            max_f = float(self.phase_max_edit.text())
            min_v = int(min_f * self.phase_range_factor)
            max_v = int(max_f * self.phase_range_factor)
            self.phase_range_slider.setValue((min_v, max_v))
            # slider valueChanged will trigger sync, persist, and refresh
        except ValueError:
            pass

    def _on_modulation_slider_changed(self, value):
        """Handle modulation range slider change from tab."""
        if self._updating_settings:
            return
        min_v, max_v = value
        min_f = min_v / self.modulation_range_factor
        max_f = max_v / self.modulation_range_factor
        self.modulation_min_edit.setText(f"{min_f:.2f}")
        self.modulation_max_edit.setText(f"{max_f:.2f}")

        if self.output_mode_combobox.currentText() == "Modulation":
            self._sync_range_to_histogram(min_f, max_f)
        self._persist_current_mesh_ranges_to_metadata()
        self._refresh_mesh_overlay_if_needed()

    def _on_modulation_edits_changed(self):
        """Handle modulation range line edit change from tab."""
        if self._updating_settings:
            return
        try:
            min_f = float(self.modulation_min_edit.text())
            max_f = float(self.modulation_max_edit.text())
            min_v = int(min_f * self.modulation_range_factor)
            max_v = int(max_f * self.modulation_range_factor)
            self.modulation_range_slider.setValue((min_v, max_v))
            # slider valueChanged will trigger sync, persist, and refresh
        except ValueError:
            pass

    def _sync_range_to_histogram(self, min_f, max_f):
        """Sync tab slider range changes to HistogramWidget and update layers."""
        # Use HistogramWidget's factor for its slider
        factor = self.lifetime_range_factor
        self.histogram_widget.range_slider.blockSignals(True)
        try:
            self.histogram_widget.set_range(min_f, max_f)
            # Re-run current range logic to update napari layers
            self._on_lifetime_range_changed(
                (int(min_f * factor), int(max_f * factor))
            )
        finally:
            self.histogram_widget.range_slider.blockSignals(False)

    def _update_tab_sliders_from_range(self, min_f, max_f):
        """Update tab sliders when range changes from HistogramWidget."""
        output_type = self._get_selected_output_type()
        self._updating_settings = True
        try:
            if output_type == "Phase":
                min_v = int(min_f * self.phase_range_factor)
                max_v = int(max_f * self.phase_range_factor)
                self.phase_range_slider.setValue((min_v, max_v))
                self.phase_min_edit.setText(f"{min_f:.2f}")
                self.phase_max_edit.setText(f"{max_f:.2f}")
            elif output_type == "Modulation":
                min_v = int(min_f * self.modulation_range_factor)
                max_v = int(max_f * self.modulation_range_factor)
                self.modulation_range_slider.setValue((min_v, max_v))
                self.modulation_min_edit.setText(f"{min_f:.2f}")
                self.modulation_max_edit.setText(f"{max_f:.2f}")
        finally:
            self._updating_settings = False
